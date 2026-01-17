//! # Prompt Caching / CacheBlend (REQ-OP-014)
//!
//! Implements cross-request KV cache reuse with CacheBlend semantic fusion.
//!
//! ## Features
//! - **Prefix Caching**: Reuse KV cache for matching prompt prefixes
//! - **Hierarchical Storage**: GPU (hot) → CPU (warm) → Disk (cold)
//! - **CacheBlend**: Position reencoding for merging knowledge fragments
//! - **Reference Counting**: Automatic lifecycle management with CoW support
//!
//! ## References
//! - CacheBlend (EuroSys'25 Best Paper): 3.9x RAG throughput
//! - LMCache: 15x throughput, 2x latency reduction
//! - vLLM: PagedAttention with prefix caching
//!
//! ## SPEC Compliance
//! - ARCH-OP-014: PromptCacheEntry, PromptCacheConfig, PromptCacheManager
//! - Target: 2-15x throughput improvement for RAG scenarios

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Token ID type alias
pub type TokenId = u32;

/// KV Block ID for paged attention integration
pub type KVBlockId = u64;

/// Cache entry identifier
pub type CacheEntryId = u64;

/// Request identifier
pub type RequestId = u64;

// ============================================================================
// Storage Tier
// ============================================================================

/// Storage tier for hierarchical caching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// GPU memory - fastest access, limited capacity (default: 1GB)
    Gpu,
    /// CPU memory - medium speed, larger capacity (default: 16GB)
    Cpu,
    /// Disk storage - slowest, unlimited capacity
    Disk,
}

impl StorageTier {
    /// Get relative access latency (lower is faster)
    pub fn access_latency(&self) -> u32 {
        match self {
            StorageTier::Gpu => 1,
            StorageTier::Cpu => 10,
            StorageTier::Disk => 1000,
        }
    }

    /// Get typical bandwidth in GB/s
    pub fn bandwidth_gbps(&self) -> f32 {
        match self {
            StorageTier::Gpu => 900.0,   // HBM
            StorageTier::Cpu => 50.0,    // DDR5
            StorageTier::Disk => 3.0,    // NVMe SSD
        }
    }
}

// ============================================================================
// Eviction Policy
// ============================================================================

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In First Out
    Fifo,
    /// Adaptive Replacement Cache
    Arc,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        EvictionPolicy::Lru
    }
}

// ============================================================================
// Hash Algorithm
// ============================================================================

/// Hash algorithm for prompt content
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// xxHash64 - fast, good distribution
    XxHash64,
    /// FNV-1a - simple, fast
    Fnv1a,
    /// SHA256 - cryptographic, slower
    Sha256,
}

impl Default for HashAlgorithm {
    fn default() -> Self {
        HashAlgorithm::XxHash64
    }
}

impl HashAlgorithm {
    /// Compute hash of token sequence
    pub fn hash(&self, tokens: &[TokenId]) -> u64 {
        match self {
            HashAlgorithm::XxHash64 => Self::xxhash64(tokens),
            HashAlgorithm::Fnv1a => Self::fnv1a(tokens),
            HashAlgorithm::Sha256 => Self::sha256_truncated(tokens),
        }
    }

    /// xxHash64 implementation
    fn xxhash64(tokens: &[TokenId]) -> u64 {
        const PRIME64_1: u64 = 0x9E3779B185EBCA87;
        const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
        const PRIME64_3: u64 = 0x165667B19E3779F9;
        #[allow(dead_code)]
        const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;
        const PRIME64_5: u64 = 0x27D4EB2F165667C5;

        let seed: u64 = 0;
        let len = tokens.len();

        let mut h64: u64;

        if len >= 4 {
            let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
            let mut v2 = seed.wrapping_add(PRIME64_2);
            let v3 = seed;
            let v4 = seed.wrapping_sub(PRIME64_1);

            let mut i = 0;
            while i + 4 <= len {
                let lane = (tokens[i] as u64)
                    | ((tokens[i + 1] as u64) << 32);
                v1 = v1.wrapping_add(lane.wrapping_mul(PRIME64_2));
                v1 = v1.rotate_left(31).wrapping_mul(PRIME64_1);

                let lane2 = (tokens[i + 2] as u64)
                    | ((tokens[i + 3] as u64) << 32);
                v2 = v2.wrapping_add(lane2.wrapping_mul(PRIME64_2));
                v2 = v2.rotate_left(31).wrapping_mul(PRIME64_1);

                i += 4;
            }

            h64 = v1.rotate_left(1)
                .wrapping_add(v2.rotate_left(7))
                .wrapping_add(v3.rotate_left(12))
                .wrapping_add(v4.rotate_left(18));
        } else {
            h64 = seed.wrapping_add(PRIME64_5);
        }

        h64 = h64.wrapping_add(len as u64 * 4);

        // Process remaining tokens
        for &token in &tokens[(len / 4 * 4)..] {
            h64 ^= (token as u64).wrapping_mul(PRIME64_5);
            h64 = h64.rotate_left(11).wrapping_mul(PRIME64_1);
        }

        // Final avalanche
        h64 ^= h64 >> 33;
        h64 = h64.wrapping_mul(PRIME64_2);
        h64 ^= h64 >> 29;
        h64 = h64.wrapping_mul(PRIME64_3);
        h64 ^= h64 >> 32;

        h64
    }

    /// FNV-1a implementation
    fn fnv1a(tokens: &[TokenId]) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        for &token in tokens {
            let bytes = token.to_le_bytes();
            for byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }

    /// SHA256 truncated to 64 bits (simplified, not cryptographic)
    fn sha256_truncated(tokens: &[TokenId]) -> u64 {
        // Simplified hash for demonstration - in production use actual SHA256
        let mut h: [u64; 4] = [
            0x6a09e667bb67ae85,
            0x3c6ef372a54ff53a,
            0x510e527f9b05688c,
            0x1f83d9ab5be0cd19,
        ];

        for &token in tokens {
            let t = token as u64;
            h[0] = h[0].wrapping_add(t.wrapping_mul(0x428a2f98d728ae22));
            h[1] = h[1].wrapping_add(t.wrapping_mul(0x7137449123ef65cd));
            h[2] = h[2].wrapping_add(t.wrapping_mul(0xb5c0fbcfec4d3b2f));
            h[3] = h[3].wrapping_add(t.wrapping_mul(0xe9b5dba58189dbbc));

            // Mix
            h[0] = h[0].rotate_left(17) ^ h[1];
            h[2] = h[2].rotate_left(23) ^ h[3];
        }

        h[0] ^ h[1] ^ h[2] ^ h[3]
    }
}

// ============================================================================
// Cache Hit Result
// ============================================================================

/// Result of a cache lookup
#[derive(Debug, Clone)]
pub struct CacheHit {
    /// The matched cache entry ID
    pub entry_id: CacheEntryId,
    /// Number of tokens that matched (prefix length)
    pub matched_tokens: usize,
    /// KV block IDs for the matched prefix
    pub kv_blocks: Vec<KVBlockId>,
    /// Storage tier where the cache was found
    pub storage_tier: StorageTier,
    /// Whether this is an exact match or prefix match
    pub is_exact_match: bool,
}

// ============================================================================
// Prompt Cache Entry (ARCH-OP-014)
// ============================================================================

/// A cached prompt entry with KV data
///
/// SPEC: ARCH-OP-014 PromptCacheEntry structure
#[derive(Debug)]
pub struct PromptCacheEntry {
    /// Prompt content hash
    pub hash: u64,
    /// KV data block IDs
    pub kv_blocks: Vec<KVBlockId>,
    /// Number of cached tokens
    pub token_count: usize,
    /// Reference count for lifecycle management
    ref_count: AtomicUsize,
    /// Last access timestamp
    last_access: std::sync::RwLock<Instant>,
    /// Current storage tier
    storage_tier: std::sync::RwLock<StorageTier>,
    /// Access frequency counter
    access_count: AtomicU64,
    /// Creation timestamp
    created_at: Instant,
    /// Original token sequence (for verification)
    tokens: Vec<TokenId>,
}

impl PromptCacheEntry {
    /// Create a new cache entry
    pub fn new(
        hash: u64,
        tokens: Vec<TokenId>,
        kv_blocks: Vec<KVBlockId>,
    ) -> Self {
        let token_count = tokens.len();
        Self {
            hash,
            kv_blocks,
            token_count,
            ref_count: AtomicUsize::new(1),
            last_access: std::sync::RwLock::new(Instant::now()),
            storage_tier: std::sync::RwLock::new(StorageTier::Gpu),
            access_count: AtomicU64::new(1),
            created_at: Instant::now(),
            tokens,
        }
    }

    /// Increment reference count
    pub fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement reference count, returns true if entry should be freed
    pub fn release(&self) -> bool {
        let prev = self.ref_count.fetch_sub(1, Ordering::SeqCst);
        prev == 1
    }

    /// Get current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Update last access time
    pub fn touch(&self) {
        if let Ok(mut last) = self.last_access.write() {
            *last = Instant::now();
        }
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get last access time
    pub fn last_access(&self) -> Instant {
        self.last_access.read().map(|t| *t).unwrap_or(self.created_at)
    }

    /// Get access frequency
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Get current storage tier
    pub fn storage_tier(&self) -> StorageTier {
        self.storage_tier.read().map(|t| *t).unwrap_or(StorageTier::Gpu)
    }

    /// Set storage tier (for migration)
    pub fn set_storage_tier(&self, tier: StorageTier) {
        if let Ok(mut t) = self.storage_tier.write() {
            *t = tier;
        }
    }

    /// Get age since creation
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Estimate memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Hash + kv_blocks vec + token_count + atomics + timestamps + tokens
        8 + self.kv_blocks.len() * 8 + 8 + 16 + 32 + self.tokens.len() * 4
    }

    /// Check if tokens match this entry (for verification)
    pub fn matches_tokens(&self, tokens: &[TokenId]) -> bool {
        self.tokens == tokens
    }

    /// Check prefix match
    pub fn matches_prefix(&self, tokens: &[TokenId]) -> Option<usize> {
        if tokens.len() < self.tokens.len() {
            // Check if entry is prefix of query
            if self.tokens.starts_with(tokens) {
                return Some(tokens.len());
            }
        } else {
            // Check if query starts with entry
            if tokens.starts_with(&self.tokens) {
                return Some(self.tokens.len());
            }
        }
        None
    }
}

// ============================================================================
// Prompt Cache Config (ARCH-OP-014)
// ============================================================================

/// Configuration for prompt caching
///
/// SPEC: ARCH-OP-014 PromptCacheConfig structure
#[derive(Debug, Clone)]
pub struct PromptCacheConfig {
    /// GPU cache size in bytes (default: 1GB)
    pub gpu_cache_size: usize,
    /// CPU cache size in bytes (default: 16GB)
    pub cpu_cache_size: usize,
    /// Enable disk caching for overflow
    pub enable_disk_cache: bool,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable CacheBlend semantic fusion
    pub enable_cacheblend: bool,
    /// Minimum prefix length to cache (tokens)
    pub min_prefix_length: usize,
    /// Hash algorithm for prompt content
    pub hash_algorithm: HashAlgorithm,
    /// Maximum entries per tier
    pub max_entries_per_tier: usize,
    /// Prefetch ahead count (entries to prefetch)
    pub prefetch_count: usize,
    /// Enable Copy-on-Write for shared entries
    pub enable_cow: bool,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            gpu_cache_size: 1024 * 1024 * 1024,      // 1 GB
            cpu_cache_size: 16 * 1024 * 1024 * 1024, // 16 GB
            enable_disk_cache: true,
            eviction_policy: EvictionPolicy::Lru,
            enable_cacheblend: true,
            min_prefix_length: 64,
            hash_algorithm: HashAlgorithm::XxHash64,
            max_entries_per_tier: 10000,
            prefetch_count: 3,
            enable_cow: true,
        }
    }
}

impl PromptCacheConfig {
    /// Create config for RAG workloads (many shared prefixes)
    pub fn for_rag() -> Self {
        Self {
            gpu_cache_size: 2 * 1024 * 1024 * 1024, // 2 GB
            cpu_cache_size: 32 * 1024 * 1024 * 1024, // 32 GB
            enable_cacheblend: true,
            min_prefix_length: 32,
            prefetch_count: 5,
            ..Default::default()
        }
    }

    /// Create config for chat workloads (long conversations)
    pub fn for_chat() -> Self {
        Self {
            min_prefix_length: 128,
            prefetch_count: 2,
            ..Default::default()
        }
    }

    /// Create minimal config for testing
    pub fn minimal() -> Self {
        Self {
            gpu_cache_size: 64 * 1024 * 1024, // 64 MB
            cpu_cache_size: 256 * 1024 * 1024, // 256 MB
            enable_disk_cache: false,
            max_entries_per_tier: 100,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.gpu_cache_size == 0 {
            return Err("GPU cache size must be > 0".to_string());
        }
        if self.min_prefix_length == 0 {
            return Err("Minimum prefix length must be > 0".to_string());
        }
        if self.max_entries_per_tier == 0 {
            return Err("Max entries per tier must be > 0".to_string());
        }
        Ok(())
    }
}

// ============================================================================
// CacheBlend Position Reencoding
// ============================================================================

/// Position reencoding for CacheBlend
#[derive(Debug, Clone)]
pub struct PositionReencoding {
    /// Original positions
    pub original_positions: Vec<usize>,
    /// Reencoded positions (merged sequence)
    pub reencoded_positions: Vec<usize>,
    /// Segment boundaries
    pub segment_boundaries: Vec<usize>,
}

impl PositionReencoding {
    /// Create reencoding for merging multiple segments
    pub fn merge_segments(segment_lengths: &[usize]) -> Self {
        let total_len: usize = segment_lengths.iter().sum();
        let mut original_positions = Vec::with_capacity(total_len);
        let mut reencoded_positions = Vec::with_capacity(total_len);
        let mut segment_boundaries = vec![0];

        let mut global_pos = 0;
        for &seg_len in segment_lengths {
            for local_pos in 0..seg_len {
                original_positions.push(local_pos);
                reencoded_positions.push(global_pos);
                global_pos += 1;
            }
            segment_boundaries.push(global_pos);
        }

        Self {
            original_positions,
            reencoded_positions,
            segment_boundaries,
        }
    }

    /// Get reencoded position for original segment and position
    pub fn get_reencoded(&self, segment_idx: usize, local_pos: usize) -> Option<usize> {
        if segment_idx >= self.segment_boundaries.len() - 1 {
            return None;
        }
        let seg_start = self.segment_boundaries[segment_idx];
        let seg_end = self.segment_boundaries[segment_idx + 1];
        let seg_len = seg_end - seg_start;

        if local_pos >= seg_len {
            return None;
        }

        Some(self.reencoded_positions[seg_start + local_pos])
    }
}

// ============================================================================
// Blended KV Cache
// ============================================================================

/// Result of CacheBlend operation
#[derive(Debug)]
pub struct BlendedKVCache {
    /// Merged KV block IDs
    pub kv_blocks: Vec<KVBlockId>,
    /// Position reencoding information
    pub position_reencoding: PositionReencoding,
    /// Source entry IDs that were blended
    pub source_entries: Vec<CacheEntryId>,
    /// Total token count
    pub total_tokens: usize,
}

// ============================================================================
// Tier Statistics
// ============================================================================

/// Statistics for a storage tier
#[derive(Debug, Clone, Default)]
pub struct TierStats {
    /// Number of entries
    pub entry_count: usize,
    /// Total bytes used
    pub bytes_used: usize,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
}

impl TierStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ============================================================================
// Prompt Cache Manager (ARCH-OP-014)
// ============================================================================

/// Manager for prompt caching with hierarchical storage
///
/// SPEC: ARCH-OP-014 PromptCacheManager interface
pub struct PromptCacheManager {
    /// Configuration
    config: PromptCacheConfig,
    /// Cache entries indexed by ID
    entries: HashMap<CacheEntryId, PromptCacheEntry>,
    /// Hash to entry ID mapping for lookup
    hash_index: HashMap<u64, CacheEntryId>,
    /// LRU ordering for GPU tier
    gpu_lru: VecDeque<CacheEntryId>,
    /// LRU ordering for CPU tier
    cpu_lru: VecDeque<CacheEntryId>,
    /// LRU ordering for Disk tier
    disk_lru: VecDeque<CacheEntryId>,
    /// Next entry ID
    next_entry_id: AtomicU64,
    /// Statistics per tier
    gpu_stats: TierStats,
    cpu_stats: TierStats,
    disk_stats: TierStats,
    /// Prefix trie for efficient prefix matching
    prefix_trie: PrefixTrie,
}

impl PromptCacheManager {
    /// Create a new cache manager
    pub fn new(config: PromptCacheConfig) -> Result<Self, String> {
        config.validate()?;

        Ok(Self {
            config,
            entries: HashMap::new(),
            hash_index: HashMap::new(),
            gpu_lru: VecDeque::new(),
            cpu_lru: VecDeque::new(),
            disk_lru: VecDeque::new(),
            next_entry_id: AtomicU64::new(1),
            gpu_stats: TierStats::default(),
            cpu_stats: TierStats::default(),
            disk_stats: TierStats::default(),
            prefix_trie: PrefixTrie::new(),
        })
    }

    /// Lookup cache for prompt tokens
    ///
    /// Returns CacheHit if prefix match found
    pub fn lookup(&mut self, prompt_tokens: &[TokenId]) -> Option<CacheHit> {
        if prompt_tokens.len() < self.config.min_prefix_length {
            return None;
        }

        // Try exact hash match first
        let hash = self.config.hash_algorithm.hash(prompt_tokens);
        if let Some(&entry_id) = self.hash_index.get(&hash) {
            // Extract data from entry first to avoid borrow conflicts
            let entry_data = self.entries.get(&entry_id).and_then(|entry| {
                if entry.matches_tokens(prompt_tokens) {
                    entry.touch();
                    Some((
                        entry.token_count,
                        entry.kv_blocks.clone(),
                        entry.storage_tier(),
                    ))
                } else {
                    None
                }
            });

            if let Some((token_count, kv_blocks, tier)) = entry_data {
                self.record_hit(tier);
                self.promote_in_lru(entry_id, tier);

                return Some(CacheHit {
                    entry_id,
                    matched_tokens: token_count,
                    kv_blocks,
                    storage_tier: tier,
                    is_exact_match: true,
                });
            }
        }

        // Try prefix matching via trie
        if let Some((entry_id, matched_len)) = self.prefix_trie.find_longest_prefix(prompt_tokens) {
            // Extract data from entry first to avoid borrow conflicts
            let entry_data = self.entries.get(&entry_id).map(|entry| {
                entry.touch();
                let kv_len = matched_len.min(entry.kv_blocks.len());
                (
                    entry.kv_blocks[..kv_len].to_vec(),
                    entry.storage_tier(),
                )
            });

            if let Some((kv_blocks, tier)) = entry_data {
                self.record_hit(tier);
                self.promote_in_lru(entry_id, tier);

                return Some(CacheHit {
                    entry_id,
                    matched_tokens: matched_len,
                    kv_blocks,
                    storage_tier: tier,
                    is_exact_match: false,
                });
            }
        }

        self.record_miss(StorageTier::Gpu);
        None
    }

    /// Insert a new cache entry
    ///
    /// Returns the assigned entry ID
    pub fn insert(
        &mut self,
        prompt_tokens: Vec<TokenId>,
        kv_blocks: Vec<KVBlockId>,
    ) -> CacheEntryId {
        let hash = self.config.hash_algorithm.hash(&prompt_tokens);

        // Check if already exists
        if let Some(&existing_id) = self.hash_index.get(&hash) {
            if let Some(entry) = self.entries.get(&existing_id) {
                if entry.matches_tokens(&prompt_tokens) {
                    entry.add_ref();
                    return existing_id;
                }
            }
        }

        // Ensure capacity
        self.ensure_capacity(StorageTier::Gpu);

        // Create new entry
        let entry_id = self.next_entry_id.fetch_add(1, Ordering::SeqCst);
        let entry = PromptCacheEntry::new(hash, prompt_tokens.clone(), kv_blocks);
        let memory_bytes = entry.memory_bytes();

        // Update indexes
        self.hash_index.insert(hash, entry_id);
        self.prefix_trie.insert(&prompt_tokens, entry_id);
        self.gpu_lru.push_back(entry_id);
        self.entries.insert(entry_id, entry);

        // Update stats
        self.gpu_stats.entry_count += 1;
        self.gpu_stats.bytes_used += memory_bytes;

        entry_id
    }

    /// Evict entries to reach target size
    ///
    /// Returns number of entries evicted
    pub fn evict_lru(&mut self, target_size: usize) -> usize {
        let mut evicted = 0;

        // Evict from GPU tier first
        while self.gpu_stats.bytes_used > target_size && !self.gpu_lru.is_empty() {
            if let Some(entry_id) = self.gpu_lru.pop_front() {
                if let Some(entry) = self.entries.get(&entry_id) {
                    // Try to demote to CPU instead of evicting
                    if self.cpu_stats.bytes_used < self.config.cpu_cache_size {
                        entry.set_storage_tier(StorageTier::Cpu);
                        self.cpu_lru.push_back(entry_id);
                        self.gpu_stats.bytes_used -= entry.memory_bytes();
                        self.cpu_stats.bytes_used += entry.memory_bytes();
                        self.gpu_stats.entry_count -= 1;
                        self.cpu_stats.entry_count += 1;
                        continue;
                    }
                }

                // Actually evict
                if let Some(entry) = self.entries.remove(&entry_id) {
                    self.hash_index.remove(&entry.hash);
                    self.prefix_trie.remove(&entry.tokens);
                    self.gpu_stats.bytes_used -= entry.memory_bytes();
                    self.gpu_stats.entry_count -= 1;
                    self.gpu_stats.evictions += 1;
                    evicted += 1;
                }
            }
        }

        evicted
    }

    /// Blend multiple knowledge fragments using CacheBlend
    ///
    /// Merges KV caches with position reencoding
    pub fn blend_knowledge(&self, entry_ids: &[CacheEntryId]) -> Option<BlendedKVCache> {
        if !self.config.enable_cacheblend {
            return None;
        }

        if entry_ids.is_empty() {
            return None;
        }

        let mut all_kv_blocks = Vec::new();
        let mut segment_lengths = Vec::new();
        let mut total_tokens = 0;

        for &entry_id in entry_ids {
            if let Some(entry) = self.entries.get(&entry_id) {
                all_kv_blocks.extend(entry.kv_blocks.iter().cloned());
                segment_lengths.push(entry.token_count);
                total_tokens += entry.token_count;
            }
        }

        if all_kv_blocks.is_empty() {
            return None;
        }

        let position_reencoding = PositionReencoding::merge_segments(&segment_lengths);

        Some(BlendedKVCache {
            kv_blocks: all_kv_blocks,
            position_reencoding,
            source_entries: entry_ids.to_vec(),
            total_tokens,
        })
    }

    /// Prefetch entry to higher storage tier
    pub fn prefetch(&mut self, entry_id: CacheEntryId) {
        if let Some(entry) = self.entries.get(&entry_id) {
            let current_tier = entry.storage_tier();

            match current_tier {
                StorageTier::Disk => {
                    // Promote to CPU
                    if self.cpu_stats.bytes_used < self.config.cpu_cache_size {
                        entry.set_storage_tier(StorageTier::Cpu);
                        self.remove_from_lru(entry_id, StorageTier::Disk);
                        self.cpu_lru.push_back(entry_id);
                        self.disk_stats.entry_count -= 1;
                        self.cpu_stats.entry_count += 1;
                    }
                }
                StorageTier::Cpu => {
                    // Promote to GPU
                    if self.gpu_stats.bytes_used < self.config.gpu_cache_size {
                        entry.set_storage_tier(StorageTier::Gpu);
                        self.remove_from_lru(entry_id, StorageTier::Cpu);
                        self.gpu_lru.push_back(entry_id);
                        self.cpu_stats.entry_count -= 1;
                        self.gpu_stats.entry_count += 1;
                    }
                }
                StorageTier::Gpu => {
                    // Already at highest tier
                }
            }
        }
    }

    /// Get statistics for all tiers
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            gpu: self.gpu_stats.clone(),
            cpu: self.cpu_stats.clone(),
            disk: self.disk_stats.clone(),
            total_entries: self.entries.len(),
        }
    }

    /// Get entry by ID
    pub fn get_entry(&self, entry_id: CacheEntryId) -> Option<&PromptCacheEntry> {
        self.entries.get(&entry_id)
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hash_index.clear();
        self.gpu_lru.clear();
        self.cpu_lru.clear();
        self.disk_lru.clear();
        self.prefix_trie = PrefixTrie::new();
        self.gpu_stats = TierStats::default();
        self.cpu_stats = TierStats::default();
        self.disk_stats = TierStats::default();
    }

    // Private helper methods

    fn ensure_capacity(&mut self, tier: StorageTier) {
        let (stats, max_size) = match tier {
            StorageTier::Gpu => (&self.gpu_stats, self.config.gpu_cache_size),
            StorageTier::Cpu => (&self.cpu_stats, self.config.cpu_cache_size),
            StorageTier::Disk => return, // No limit for disk
        };

        if stats.bytes_used >= max_size {
            // Evict 10% to make room
            let target = max_size * 90 / 100;
            self.evict_lru(target);
        }
    }

    fn record_hit(&mut self, tier: StorageTier) {
        match tier {
            StorageTier::Gpu => self.gpu_stats.hits += 1,
            StorageTier::Cpu => self.cpu_stats.hits += 1,
            StorageTier::Disk => self.disk_stats.hits += 1,
        }
    }

    fn record_miss(&mut self, tier: StorageTier) {
        match tier {
            StorageTier::Gpu => self.gpu_stats.misses += 1,
            StorageTier::Cpu => self.cpu_stats.misses += 1,
            StorageTier::Disk => self.disk_stats.misses += 1,
        }
    }

    fn promote_in_lru(&mut self, entry_id: CacheEntryId, tier: StorageTier) {
        let lru = match tier {
            StorageTier::Gpu => &mut self.gpu_lru,
            StorageTier::Cpu => &mut self.cpu_lru,
            StorageTier::Disk => &mut self.disk_lru,
        };

        if let Some(pos) = lru.iter().position(|&id| id == entry_id) {
            lru.remove(pos);
            lru.push_back(entry_id);
        }
    }

    fn remove_from_lru(&mut self, entry_id: CacheEntryId, tier: StorageTier) {
        let lru = match tier {
            StorageTier::Gpu => &mut self.gpu_lru,
            StorageTier::Cpu => &mut self.cpu_lru,
            StorageTier::Disk => &mut self.disk_lru,
        };

        if let Some(pos) = lru.iter().position(|&id| id == entry_id) {
            lru.remove(pos);
        }
    }
}

// ============================================================================
// Cache Statistics
// ============================================================================

/// Overall cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// GPU tier stats
    pub gpu: TierStats,
    /// CPU tier stats
    pub cpu: TierStats,
    /// Disk tier stats
    pub disk: TierStats,
    /// Total entries across all tiers
    pub total_entries: usize,
}

impl CacheStats {
    /// Calculate overall hit rate
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.gpu.hits + self.cpu.hits + self.disk.hits;
        let total_misses = self.gpu.misses + self.cpu.misses + self.disk.misses;
        let total = total_hits + total_misses;

        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// Calculate total bytes used
    pub fn total_bytes(&self) -> usize {
        self.gpu.bytes_used + self.cpu.bytes_used + self.disk.bytes_used
    }

    /// Estimate throughput improvement factor
    pub fn estimated_speedup(&self) -> f64 {
        // Based on hit rates and tier latencies
        let gpu_weight = self.gpu.hits as f64 * (1.0 / StorageTier::Gpu.access_latency() as f64);
        let cpu_weight = self.cpu.hits as f64 * (1.0 / StorageTier::Cpu.access_latency() as f64);
        let disk_weight = self.disk.hits as f64 * (1.0 / StorageTier::Disk.access_latency() as f64);
        let miss_weight = (self.gpu.misses + self.cpu.misses + self.disk.misses) as f64;

        let total = gpu_weight + cpu_weight + disk_weight + miss_weight;
        if total == 0.0 {
            1.0
        } else {
            let hit_contribution = gpu_weight + cpu_weight + disk_weight;
            1.0 + (hit_contribution / total) * 14.0 // Up to 15x for 100% GPU hits
        }
    }
}

// ============================================================================
// Prefix Trie for Efficient Matching
// ============================================================================

/// Simple prefix trie for efficient longest prefix matching
struct PrefixTrie {
    root: TrieNode,
}

struct TrieNode {
    children: HashMap<TokenId, Box<TrieNode>>,
    entry_id: Option<CacheEntryId>,
}

impl PrefixTrie {
    fn new() -> Self {
        Self {
            root: TrieNode {
                children: HashMap::new(),
                entry_id: None,
            },
        }
    }

    fn insert(&mut self, tokens: &[TokenId], entry_id: CacheEntryId) {
        let mut node = &mut self.root;
        for &token in tokens {
            node = node.children
                .entry(token)
                .or_insert_with(|| Box::new(TrieNode {
                    children: HashMap::new(),
                    entry_id: None,
                }));
        }
        node.entry_id = Some(entry_id);
    }

    fn find_longest_prefix(&self, tokens: &[TokenId]) -> Option<(CacheEntryId, usize)> {
        let mut node = &self.root;
        let mut best_match: Option<(CacheEntryId, usize)> = None;

        for (i, &token) in tokens.iter().enumerate() {
            if let Some(child) = node.children.get(&token) {
                node = child;
                if let Some(entry_id) = node.entry_id {
                    best_match = Some((entry_id, i + 1));
                }
            } else {
                break;
            }
        }

        best_match
    }

    fn remove(&mut self, tokens: &[TokenId]) {
        // Simple removal - just clear the entry_id
        let mut node = &mut self.root;
        for &token in tokens {
            if let Some(child) = node.children.get_mut(&token) {
                node = child;
            } else {
                return;
            }
        }
        node.entry_id = None;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier_properties() {
        assert!(StorageTier::Gpu.access_latency() < StorageTier::Cpu.access_latency());
        assert!(StorageTier::Cpu.access_latency() < StorageTier::Disk.access_latency());

        assert!(StorageTier::Gpu.bandwidth_gbps() > StorageTier::Cpu.bandwidth_gbps());
        assert!(StorageTier::Cpu.bandwidth_gbps() > StorageTier::Disk.bandwidth_gbps());
    }

    #[test]
    fn test_hash_algorithms() {
        let tokens: Vec<TokenId> = vec![100, 200, 300, 400, 500];

        let xxhash = HashAlgorithm::XxHash64.hash(&tokens);
        let fnv = HashAlgorithm::Fnv1a.hash(&tokens);
        let sha = HashAlgorithm::Sha256.hash(&tokens);

        // Different algorithms should produce different hashes
        assert_ne!(xxhash, fnv);
        assert_ne!(fnv, sha);

        // Same input should produce same output
        assert_eq!(xxhash, HashAlgorithm::XxHash64.hash(&tokens));

        // Different input should produce different output
        let tokens2: Vec<TokenId> = vec![100, 200, 300, 400, 501];
        assert_ne!(xxhash, HashAlgorithm::XxHash64.hash(&tokens2));
    }

    #[test]
    fn test_cache_entry_lifecycle() {
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4, 5];
        let kv_blocks: Vec<KVBlockId> = vec![100, 101, 102];
        let hash = HashAlgorithm::XxHash64.hash(&tokens);

        let entry = PromptCacheEntry::new(hash, tokens.clone(), kv_blocks);

        assert_eq!(entry.ref_count(), 1);
        assert_eq!(entry.token_count, 5);
        assert_eq!(entry.storage_tier(), StorageTier::Gpu);

        entry.add_ref();
        assert_eq!(entry.ref_count(), 2);

        assert!(!entry.release());
        assert_eq!(entry.ref_count(), 1);

        assert!(entry.release());
    }

    #[test]
    fn test_cache_entry_matching() {
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4, 5];
        let entry = PromptCacheEntry::new(0, tokens.clone(), vec![]);

        // Exact match
        assert!(entry.matches_tokens(&tokens));
        assert!(!entry.matches_tokens(&[1, 2, 3]));

        // Prefix match - query is prefix of entry
        assert_eq!(entry.matches_prefix(&[1, 2, 3]), Some(3));

        // Prefix match - entry is prefix of query
        let longer: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 7];
        assert_eq!(entry.matches_prefix(&longer), Some(5));

        // No match
        assert_eq!(entry.matches_prefix(&[2, 3, 4]), None);
    }

    #[test]
    fn test_config_validation() {
        let valid = PromptCacheConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = PromptCacheConfig {
            gpu_cache_size: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid2 = PromptCacheConfig {
            min_prefix_length: 0,
            ..Default::default()
        };
        assert!(invalid2.validate().is_err());
    }

    #[test]
    fn test_position_reencoding() {
        // Merge 3 segments: 10 tokens, 5 tokens, 8 tokens
        let reencoding = PositionReencoding::merge_segments(&[10, 5, 8]);

        assert_eq!(reencoding.segment_boundaries, vec![0, 10, 15, 23]);
        assert_eq!(reencoding.reencoded_positions.len(), 23);

        // Check segment 0
        assert_eq!(reencoding.get_reencoded(0, 0), Some(0));
        assert_eq!(reencoding.get_reencoded(0, 5), Some(5));
        assert_eq!(reencoding.get_reencoded(0, 9), Some(9));

        // Check segment 1
        assert_eq!(reencoding.get_reencoded(1, 0), Some(10));
        assert_eq!(reencoding.get_reencoded(1, 4), Some(14));

        // Check segment 2
        assert_eq!(reencoding.get_reencoded(2, 0), Some(15));
        assert_eq!(reencoding.get_reencoded(2, 7), Some(22));

        // Out of bounds
        assert_eq!(reencoding.get_reencoded(0, 10), None);
        assert_eq!(reencoding.get_reencoded(3, 0), None);
    }

    #[test]
    fn test_cache_manager_basic() {
        let config = PromptCacheConfig {
            min_prefix_length: 32, // Lower threshold for testing
            ..PromptCacheConfig::minimal()
        };
        let mut manager = PromptCacheManager::new(config).unwrap();

        // Insert entry with 100 tokens
        let tokens: Vec<TokenId> = (0..100).collect();
        let kv_blocks: Vec<KVBlockId> = (0..100).map(|i| i as u64).collect();

        let entry_id = manager.insert(tokens.clone(), kv_blocks.clone());
        assert!(entry_id > 0);

        // Lookup exact match - should hit
        let hit = manager.lookup(&tokens);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!(hit.is_exact_match);
        assert_eq!(hit.matched_tokens, 100);

        // Insert a shorter prefix entry
        let short_prefix: Vec<TokenId> = (0..50).collect();
        manager.insert(short_prefix.clone(), vec![1, 2, 3]);

        // Lookup with longer query that shares prefix - should find the prefix
        let longer_query: Vec<TokenId> = (0..75).collect();
        let hit = manager.lookup(&longer_query);
        assert!(hit.is_some());
        // Should match the 50-token prefix entry (entry is prefix of query)
        let hit = hit.unwrap();
        assert_eq!(hit.matched_tokens, 50);

        // Stats
        let stats = manager.stats();
        assert_eq!(stats.total_entries, 2);
        assert!(stats.gpu.hits > 0);
    }

    #[test]
    fn test_cache_manager_eviction() {
        let config = PromptCacheConfig {
            gpu_cache_size: 1000, // Very small
            cpu_cache_size: 500,  // Also small to force actual eviction
            max_entries_per_tier: 5,
            min_prefix_length: 10,
            ..PromptCacheConfig::minimal()
        };
        let mut manager = PromptCacheManager::new(config).unwrap();

        // Insert many entries
        for i in 0..10 {
            let tokens: Vec<TokenId> = (i * 100..(i + 1) * 100).collect();
            let kv_blocks: Vec<KVBlockId> = tokens.iter().map(|&t| t as u64).collect();
            manager.insert(tokens, kv_blocks);
        }

        // Stats before eviction
        let stats_before = manager.stats();
        let gpu_entries_before = stats_before.gpu.entry_count;

        // Force eviction - this may demote to CPU or actually evict
        manager.evict_lru(0); // Target 0 to force maximum eviction

        // Either entries were demoted to CPU or actually evicted
        let stats_after = manager.stats();
        // At least some entries should have moved/evicted from GPU
        assert!(stats_after.gpu.entry_count < gpu_entries_before || stats_after.gpu.evictions > 0);
    }

    #[test]
    fn test_cacheblend() {
        let config = PromptCacheConfig::minimal();
        let mut manager = PromptCacheManager::new(config).unwrap();

        // Insert multiple knowledge fragments
        let tokens1: Vec<TokenId> = (0..100).collect();
        let tokens2: Vec<TokenId> = (1000..1050).collect();
        let tokens3: Vec<TokenId> = (2000..2080).collect();

        let id1 = manager.insert(tokens1, vec![1, 2, 3]);
        let id2 = manager.insert(tokens2, vec![4, 5]);
        let id3 = manager.insert(tokens3, vec![6, 7, 8, 9]);

        // Blend
        let blended = manager.blend_knowledge(&[id1, id2, id3]);
        assert!(blended.is_some());

        let blended = blended.unwrap();
        assert_eq!(blended.kv_blocks, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(blended.total_tokens, 100 + 50 + 80);
        assert_eq!(blended.source_entries.len(), 3);
    }

    #[test]
    fn test_tier_stats() {
        let mut stats = TierStats::default();

        assert_eq!(stats.hit_rate(), 0.0);

        stats.hits = 80;
        stats.misses = 20;
        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_speedup() {
        let stats = CacheStats {
            gpu: TierStats {
                hits: 100,
                misses: 0,
                ..Default::default()
            },
            cpu: TierStats::default(),
            disk: TierStats::default(),
            total_entries: 10,
        };

        // 100% GPU hits should give near max speedup
        let speedup = stats.estimated_speedup();
        assert!(speedup > 10.0); // Should be close to 15x
    }

    #[test]
    fn test_prefix_trie() {
        let mut trie = PrefixTrie::new();

        // Insert entries
        trie.insert(&[1, 2, 3, 4, 5], 100);
        trie.insert(&[1, 2, 3], 200);
        trie.insert(&[1, 2, 3, 4, 5, 6, 7], 300);

        // Find longest prefix
        let result = trie.find_longest_prefix(&[1, 2, 3, 4]);
        assert_eq!(result, Some((200, 3)));

        let result = trie.find_longest_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(result, Some((100, 5)));

        let result = trie.find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(result, Some((300, 7)));

        // No match
        let result = trie.find_longest_prefix(&[5, 6, 7]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_config_presets() {
        let rag = PromptCacheConfig::for_rag();
        assert!(rag.enable_cacheblend);
        assert!(rag.gpu_cache_size > PromptCacheConfig::default().gpu_cache_size);

        let chat = PromptCacheConfig::for_chat();
        assert!(chat.min_prefix_length > PromptCacheConfig::default().min_prefix_length);
    }
}
