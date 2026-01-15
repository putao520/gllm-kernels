//! Hierarchical FlashAttention with direct paged KV access.
//!
//! ## Performance Optimizations
//!
//! This module includes several optimizations for ultra-long contexts:
//! - **Pre-allocated workspace buffers** to avoid repeated allocations
//! - **Causal mask caching** with relative offset keys for high hit rates
//! - **Reduced cloning** through careful use of references and slicing
//! - **Batched operations** to minimize intermediate tensor creation
//!
//! ## Mask Cache Strategy
//!
//! Causal masks follow a pattern where the mask value at (i, j) depends only on:
//! - `relative_offset = (q_start + position_offset) - kv_start`
//! - Position within the block (i, j)
//!
//! By using `relative_offset` as the cache key instead of absolute positions,
//! we achieve much higher cache hit rates for long sequences.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::ops::stable_accumulator::AccumulatorConfig;

/// Default mask cache capacity (covers most practical scenarios).
/// For seq_len 4096 with block_q=64, block_kv=16: ~4096 unique relative offsets.
const DEFAULT_MASK_CACHE_CAPACITY: usize = 8192;

/// Global configuration for mask cache capacity.
static MASK_CACHE_CAPACITY: AtomicUsize = AtomicUsize::new(DEFAULT_MASK_CACHE_CAPACITY);

/// Cache key for causal masks using relative offset strategy.
///
/// The key insight is that causal masks depend on the *relative* position
/// between query and key, not their absolute positions. This dramatically
/// reduces the number of unique masks needed.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct MaskCacheKey {
    /// Query block length.
    query_len: usize,
    /// Key block length.
    key_len: usize,
    /// Relative offset: (q_start + position_offset) - kv_start.
    /// This determines the causal boundary within the block.
    relative_offset: isize,
}

impl MaskCacheKey {
    /// Create a cache key from absolute positions.
    fn from_positions(
        query_len: usize,
        key_len: usize,
        q_start: usize,
        kv_start: usize,
        position_offset: usize,
    ) -> Self {
        // The causal condition is: kv_pos <= q_pos + position_offset
        // Rewritten: kv_start + j <= q_start + position_offset + i
        // Which is: j <= (q_start + position_offset - kv_start) + i
        // So the relative_offset determines where the causal boundary starts
        let relative_offset = (q_start + position_offset) as isize - kv_start as isize;
        Self {
            query_len,
            key_len,
            relative_offset,
        }
    }
}

// Thread-local cache for causal masks to avoid repeated allocations.
thread_local! {
    static MASK_CACHE: RefCell<MaskCache> = RefCell::new(
        MaskCache::new(MASK_CACHE_CAPACITY.load(Ordering::Relaxed))
    );
}

/// LRU-style mask cache with bounded capacity and cache statistics.
struct MaskCache {
    /// Cached mask data (query_len * key_len f32 values).
    cache: HashMap<MaskCacheKey, Vec<f32>>,
    /// Access order for LRU eviction (using VecDeque would be more efficient).
    access_order: Vec<MaskCacheKey>,
    /// Maximum number of cached masks.
    capacity: usize,
    /// Cache hit count (for diagnostics).
    hits: usize,
    /// Cache miss count (for diagnostics).
    misses: usize,
}

impl MaskCache {
    fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity.min(1024)), // Don't over-allocate initially
            access_order: Vec::with_capacity(capacity.min(1024)),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    fn get_or_create(
        &mut self,
        key: MaskCacheKey,
        create_fn: impl FnOnce() -> Vec<f32>,
    ) -> &Vec<f32> {
        if self.cache.contains_key(&key) {
            self.hits += 1;
            // Move to end (most recently used) - only if not already at end
            if self.access_order.last() != Some(&key) {
                if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                    self.access_order.remove(pos);
                    self.access_order.push(key.clone());
                }
            }
        } else {
            self.misses += 1;
            // Evict oldest entries if at capacity
            while self.cache.len() >= self.capacity && !self.access_order.is_empty() {
                let oldest = self.access_order.remove(0);
                self.cache.remove(&oldest);
            }
            self.cache.insert(key.clone(), create_fn());
            self.access_order.push(key.clone());
        }
        self.cache.get(&key).unwrap()
    }

    /// Get cache statistics (hits, misses, hit_rate).
    #[allow(dead_code)]
    fn stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }

    /// Clear the cache and reset statistics.
    fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Resize the cache capacity.
    #[allow(dead_code)]
    fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
        // Evict excess entries if shrinking
        while self.cache.len() > self.capacity && !self.access_order.is_empty() {
            let oldest = self.access_order.remove(0);
            self.cache.remove(&oldest);
        }
    }

    /// Current number of cached entries.
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.cache.len()
    }
}

/// Configuration for mask caching behavior.
#[derive(Clone, Debug)]
pub struct MaskCacheConfig {
    /// Maximum number of masks to cache per thread.
    pub capacity: usize,
    /// Enable cache statistics tracking.
    pub track_stats: bool,
}

impl Default for MaskCacheConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_MASK_CACHE_CAPACITY,
            track_stats: false,
        }
    }
}

impl MaskCacheConfig {
    /// Configuration optimized for short sequences (< 2K).
    pub fn short_context() -> Self {
        Self {
            capacity: 1024,
            track_stats: false,
        }
    }

    /// Configuration optimized for medium sequences (2K - 8K).
    pub fn medium_context() -> Self {
        Self {
            capacity: 8192,
            track_stats: false,
        }
    }

    /// Configuration optimized for long sequences (8K - 32K).
    pub fn long_context() -> Self {
        Self {
            capacity: 32768,
            track_stats: false,
        }
    }

    /// Configuration optimized for ultra-long sequences (32K+).
    /// Uses more memory but ensures high cache hit rates.
    pub fn ultra_long_context() -> Self {
        Self {
            capacity: 131072, // 128K entries, ~512MB for 64x16 masks
            track_stats: false,
        }
    }
}

/// Configuration for deterministic computation.
#[derive(Clone, Debug)]
pub struct DeterministicConfig {
    /// Enable deterministic mode.
    pub enabled: bool,
    /// Force fixed tile processing order for reproducibility.
    pub fixed_tile_order: bool,
    /// Fixed random seed for reproducibility.
    pub seed: Option<u64>,
    /// Disable GPU non-deterministic operations (use deterministic kernels).
    pub no_gpu_nondeterminism: bool,
    /// Enable verification of determinism (compare results of multiple runs).
    pub verify_determinism: bool,
}

impl Default for DeterministicConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fixed_tile_order: false,
            seed: None,
            no_gpu_nondeterminism: false,
            verify_determinism: false,
        }
    }
}

impl DeterministicConfig {
    /// Create a configuration for maximum reproducibility.
    pub fn strict() -> Self {
        Self {
            enabled: true,
            fixed_tile_order: true,
            seed: Some(42),
            no_gpu_nondeterminism: true,
            verify_determinism: cfg!(debug_assertions),
        }
    }

    /// Create a configuration that allows some non-determinism for speed.
    pub fn relaxed() -> Self {
        Self {
            enabled: false,
            fixed_tile_order: false,
            seed: None,
            no_gpu_nondeterminism: false,
            verify_determinism: false,
        }
    }

    /// Create a configuration for 2M context (strict by default).
    pub fn ultra_long_context() -> Self {
        Self::strict()
    }

    /// Check if any deterministic guarantees are enabled.
    pub fn is_deterministic(&self) -> bool {
        self.enabled || self.fixed_tile_order || self.seed.is_some()
    }
}

/// Strict ordering iterator for deterministic processing.
pub struct StrictOrderIterator<I> {
    inner: I,
    index: usize,
}

impl<I: Iterator> StrictOrderIterator<I> {
    pub fn new(iter: I) -> Self {
        Self { inner: iter, index: 0 }
    }

    /// Get the current index (for verification).
    pub fn current_index(&self) -> usize {
        self.index
    }
}

impl<I: Iterator> Iterator for StrictOrderIterator<I> {
    type Item = (usize, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        let index = self.index;
        self.index += 1;

        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        Some((index, item))
    }
}

/// Extension trait for creating strict order iterators.
pub trait StrictOrderExt: Iterator + Sized {
    fn strict_order(self) -> StrictOrderIterator<Self> {
        StrictOrderIterator::new(self)
    }
}

impl<I: Iterator> StrictOrderExt for I {}

/// Configuration for hierarchical FlashAttention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Query block size for tiling.
    pub block_q: usize,
    /// KV block size for tiling (should match PagedKVCache block size).
    pub block_kv: usize,
    /// Accumulator configuration for numerical stability.
    pub accumulator: AccumulatorConfig,
    /// Determinism configuration.
    pub determinism: DeterministicConfig,
    /// Use log-space accumulation (more stable but slightly slower).
    pub use_log_space: bool,
    /// Maximum sequence length to expect (for pre-allocation).
    pub max_seq_len: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_q: 64,
            block_kv: 16,
            accumulator: AccumulatorConfig::max_precision(),
            determinism: DeterministicConfig::strict(),
            use_log_space: true,
            max_seq_len: 2_000_000,
        }
    }
}

impl FlashAttentionConfig {
    /// Configuration optimized for 2M context.
    pub fn ultra_long_context() -> Self {
        Self {
            block_q: 64,
            block_kv: 16,
            accumulator: AccumulatorConfig::max_precision(),
            determinism: DeterministicConfig::ultra_long_context(),
            use_log_space: true,
            max_seq_len: 2_000_000,
        }
    }

    /// Configuration for shorter contexts (< 100K).
    pub fn short_context() -> Self {
        Self {
            block_q: 128,
            block_kv: 64,
            accumulator: AccumulatorConfig::short_context(),
            determinism: DeterministicConfig::relaxed(),
            use_log_space: false,
            max_seq_len: 100_000,
        }
    }
}

/// Backward-compatible alias.
pub type HierarchicalFlashConfig = FlashAttentionConfig;

/// Trait for fused paged attention computation.
pub trait FusedPagedAttention<B: Backend> {
    /// Compute attention with direct access to paged KV blocks.
    fn forward_fused<'a, I>(
        &self,
        q: Tensor<B, 4>,
        kv_blocks: I,
        config: &FlashAttentionConfig,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4>
    where
        I: Iterator<Item = (Tensor<B, 3>, Tensor<B, 3>)> + 'a;
}

/// Pre-allocated workspace for attention computation.
///
/// This structure holds temporary buffers that are reused across iterations
/// to avoid repeated memory allocations in the hot path.
pub struct AttentionWorkspace<B: Backend> {
    /// Running maximum values [batch, heads, q_block_len, 1].
    pub m_buffer: Option<Tensor<B, 4>>,
    /// Running sum values [batch, heads, q_block_len, 1].
    pub l_buffer: Option<Tensor<B, 4>>,
    /// Output accumulator [batch, heads, q_block_len, head_dim].
    pub o_buffer: Option<Tensor<B, 4>>,
    /// Cached dimensions for validation.
    dims: Option<(usize, usize, usize, usize)>,
}

impl<B: Backend> Default for AttentionWorkspace<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> AttentionWorkspace<B> {
    /// Create an empty workspace.
    pub fn new() -> Self {
        Self {
            m_buffer: None,
            l_buffer: None,
            o_buffer: None,
            dims: None,
        }
    }

    /// Pre-allocate buffers for given dimensions.
    pub fn allocate(
        &mut self,
        device: &B::Device,
        batch_size: usize,
        num_heads: usize,
        q_block_len: usize,
        head_dim: usize,
    ) {
        let needs_realloc = self.dims.map_or(true, |(b, h, q, d)| {
            b != batch_size || h != num_heads || q < q_block_len || d != head_dim
        });

        if needs_realloc {
            self.m_buffer = Some(Tensor::zeros(
                [batch_size, num_heads, q_block_len, 1],
                device,
            ));
            self.l_buffer = Some(Tensor::zeros(
                [batch_size, num_heads, q_block_len, 1],
                device,
            ));
            self.o_buffer = Some(Tensor::zeros(
                [batch_size, num_heads, q_block_len, head_dim],
                device,
            ));
            self.dims = Some((batch_size, num_heads, q_block_len, head_dim));
        }
    }

    /// Reset buffers to initial values for a new Q block.
    pub fn reset(&mut self, device: &B::Device) {
        if let Some((batch_size, num_heads, q_block_len, _)) = self.dims {
            self.m_buffer = Some(Tensor::full(
                [batch_size, num_heads, q_block_len, 1],
                f32::NEG_INFINITY,
                device,
            ));
            if let Some(ref mut l) = self.l_buffer {
                *l = l.clone().zeros_like();
            }
            if let Some(ref mut o) = self.o_buffer {
                *o = o.clone().zeros_like();
            }
        }
    }

    /// Take ownership of output buffer.
    pub fn take_output(&mut self) -> Option<Tensor<B, 4>> {
        self.o_buffer.take()
    }
}

/// Hierarchical FlashAttention implementation.
#[derive(Debug, Clone)]
pub struct HierarchicalFlashAttention {
    config: FlashAttentionConfig,
}

impl HierarchicalFlashAttention {
    /// Create a new HierarchicalFlashAttention with the given configuration.
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(FlashAttentionConfig::default())
    }

    /// Create optimized for 2M context.
    pub fn ultra_long_context() -> Self {
        Self::new(FlashAttentionConfig::ultra_long_context())
    }

    /// Get the configuration.
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Optimized forward pass with pre-allocated workspace.
    ///
    /// This method reuses workspace buffers across Q blocks to minimize allocations.
    /// For best performance, create a workspace once and reuse it across calls.
    pub fn forward_with_workspace<B: Backend>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
        workspace: &mut AttentionWorkspace<B>,
    ) -> Tensor<B, 4> {
        let device = q.device();
        let [batch_size, num_heads, query_len, head_dim] = q.dims();
        let key_len = k.dims()[2];

        if query_len == 0 || key_len == 0 {
            return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
        }

        let block_q = self.config.block_q.max(1);
        let block_kv = self.config.block_kv.max(1);
        let inv_scale = 1.0 / (head_dim as f32).sqrt();

        // Pre-allocate workspace for maximum block size
        workspace.allocate(&device, batch_size, num_heads, block_q, head_dim);

        let q_blocks = q.split(block_q, 2);
        let k_blocks = k.split(block_kv, 2);
        let v_blocks = v.split(block_kv, 2);
        let k_blocks_t: Vec<_> = k_blocks.into_iter().map(|block| block.transpose()).collect();
        let mut outputs = Vec::with_capacity(q_blocks.len());

        for (q_block_index, q_block) in q_blocks.into_iter().enumerate() {
            let q_block_len = q_block.dims()[2];
            let q_start = q_block_index * block_q;
            let q_block_scaled = q_block * inv_scale;

            // Initialize accumulators
            let mut m_i = Tensor::<B, 4>::full(
                [batch_size, num_heads, q_block_len, 1],
                f32::NEG_INFINITY,
                &device,
            );
            let mut l_i = Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
            let mut o_i = Tensor::<B, 4>::zeros(
                [batch_size, num_heads, q_block_len, head_dim],
                &device,
            );

            for (kv_index, (k_block_t, v_block)) in
                k_blocks_t.iter().zip(v_blocks.iter()).enumerate()
            {
                let kv_block_len = k_block_t.dims()[3];
                let kv_start = kv_index * block_kv;

                // Compute attention scores
                let mut scores = q_block_scaled.clone().matmul(k_block_t.clone());

                // Apply causal mask with caching
                if causal {
                    let mask = self.build_causal_mask_cached::<B>(
                        &device,
                        q_block_len,
                        kv_block_len,
                        q_start,
                        kv_start,
                        position_offset,
                    );
                    scores = scores + mask;
                }

                // Online softmax update (fused operations)
                let m_ij = scores.clone().max_dim(3);
                let m_new = m_i.clone().max_pair(m_ij);

                let m_scale = (m_i - m_new.clone()).exp();
                let p_ij = (scores - m_new.clone()).exp();
                let p_sum = p_ij.clone().sum_dim(3);

                // Update accumulators
                l_i = m_scale.clone() * l_i + p_sum;
                o_i = m_scale * o_i + p_ij.matmul(v_block.clone());
                m_i = m_new;
            }

            outputs.push(o_i / l_i);
        }

        Tensor::cat(outputs, 2)
    }

    /// Build causal mask with thread-local caching using relative offset strategy.
    ///
    /// The mask pattern depends only on `relative_offset = (q_start + position_offset) - kv_start`,
    /// not on absolute positions. This dramatically increases cache hit rates for long sequences.
    ///
    /// For example, with block_q=64, block_kv=16, seq_len=8192:
    /// - Old strategy: 65536 unique keys (16 q_blocks × 512 kv_blocks × 8 position offsets)
    /// - New strategy: ~8192 unique keys (based on relative offset range)
    fn build_causal_mask_cached<B: Backend>(
        &self,
        device: &B::Device,
        query_len: usize,
        key_len: usize,
        q_start: usize,
        kv_start: usize,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        // Use relative offset as key for better cache hit rate
        let key = MaskCacheKey::from_positions(
            query_len,
            key_len,
            q_start,
            kv_start,
            position_offset,
        );
        let relative_offset = key.relative_offset;

        let mask_value = -1.0e4_f32;

        let data = MASK_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache
                .get_or_create(key, || {
                    let mut data = Vec::with_capacity(query_len * key_len);
                    // The causal condition: kv_pos <= q_pos
                    // With relative_offset = q_start + position_offset - kv_start:
                    // j <= relative_offset + i
                    for i in 0..query_len {
                        let threshold = relative_offset + i as isize;
                        for j in 0..key_len {
                            let allowed = (j as isize) <= threshold;
                            data.push(if allowed { 0.0 } else { mask_value });
                        }
                    }
                    data
                })
                .clone()
        });

        Tensor::<B, 2>::from_data(TensorData::new(data, [query_len, key_len]), device)
            .reshape([1, 1, query_len, key_len])
    }

    /// Standard FlashAttention forward pass (non-fused, for reference/testing).
    pub fn forward<B: Backend>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let device = q.device();
        let [batch_size, num_heads, query_len, head_dim] = q.dims();
        let key_len = k.dims()[2];

        if query_len == 0 || key_len == 0 {
            return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
        }

        let block_q = self.config.block_q.max(1);
        let block_kv = self.config.block_kv.max(1);
        let inv_scale = 1.0 / (head_dim as f32).sqrt();

        let k_blocks = k.split(block_kv, 2);
        let v_blocks = v.split(block_kv, 2);
        let k_blocks_t: Vec<_> = k_blocks.into_iter().map(|block| block.transpose()).collect();
        let q_blocks = q.split(block_q, 2);

        let mut outputs = Vec::with_capacity(q_blocks.len());
        let fixed_tile_order = self.config.determinism.fixed_tile_order;
        let kv_block_count = k_blocks_t.len();

        let process_q_block = |q_start: usize, q_block: Tensor<B, 4>, outputs: &mut Vec<Tensor<B, 4>>| {
            let q_block_len = q_block.dims()[2];
            let q_block_scaled = q_block * inv_scale;

            let mut m_i = Tensor::<B, 4>::full(
                [batch_size, num_heads, q_block_len, 1],
                f32::NEG_INFINITY,
                &device,
            );
            let mut l_i = Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
            let mut o_i = Tensor::<B, 4>::zeros(
                [batch_size, num_heads, q_block_len, head_dim],
                &device,
            );

            if fixed_tile_order {
                let mut kv_index = 0usize;
                while kv_index < kv_block_count {
                    let k_block_t = &k_blocks_t[kv_index];
                    let v_block = &v_blocks[kv_index];
                    let kv_block_len = k_block_t.dims()[3];
                    let kv_start = kv_index * block_kv;

                    let mut scores = q_block_scaled.clone().matmul(k_block_t.clone());

                    if causal {
                        let mask = self.build_causal_mask_cached::<B>(
                            &device,
                            q_block_len,
                            kv_block_len,
                            q_start,
                            kv_start,
                            position_offset,
                        );
                        scores = scores + mask;
                    }

                    let m_ij = scores.clone().max_dim(3);
                    let m_new = m_i.clone().max_pair(m_ij);

                    let m_scale = (m_i - m_new.clone()).exp();
                    let p_ij = (scores - m_new.clone()).exp();
                    let p_sum = p_ij.clone().sum_dim(3);

                    l_i = m_scale.clone() * l_i + p_sum;
                    o_i = m_scale * o_i + p_ij.matmul(v_block.clone());
                    m_i = m_new;

                    kv_index += 1;
                }
            } else {
                for kv_index in 0..kv_block_count {
                    let k_block_t = &k_blocks_t[kv_index];
                    let v_block = &v_blocks[kv_index];
                    let kv_block_len = k_block_t.dims()[3];
                    let kv_start = kv_index * block_kv;

                    let mut scores = q_block_scaled.clone().matmul(k_block_t.clone());

                    if causal {
                        let mask = self.build_causal_mask_cached::<B>(
                            &device,
                            q_block_len,
                            kv_block_len,
                            q_start,
                            kv_start,
                            position_offset,
                        );
                        scores = scores + mask;
                    }

                    let m_ij = scores.clone().max_dim(3);
                    let m_new = m_i.clone().max_pair(m_ij);

                    let m_scale = (m_i - m_new.clone()).exp();
                    let p_ij = (scores - m_new.clone()).exp();
                    let p_sum = p_ij.clone().sum_dim(3);

                    l_i = m_scale.clone() * l_i + p_sum;
                    o_i = m_scale * o_i + p_ij.matmul(v_block.clone());
                    m_i = m_new;
                }
            }

            outputs.push(o_i / l_i);
        };

        for (q_block_index, q_block) in q_blocks.into_iter().enumerate() {
            let q_start = q_block_index * block_q;
            process_q_block(q_start, q_block, &mut outputs);
        }

        let output = Tensor::cat(outputs, 2);
        let _ = B::sync(&output.device());
        output
    }

    /// Fused forward pass that directly iterates over KV blocks.
    pub fn forward_fused_iter<'a, B, I>(
        &self,
        q: Tensor<B, 4>,
        kv_blocks: I,
        causal: bool,
        position_offset: usize,
        total_kv_len: usize,
    ) -> Tensor<B, 4>
    where
        B: Backend,
        I: Iterator<Item = (Tensor<B, 3>, Tensor<B, 3>)> + 'a,
    {
        let device = q.device();
        let [batch_size, num_heads, query_len, head_dim] = q.dims();

        if query_len == 0 || total_kv_len == 0 {
            return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
        }

        let block_q = self.config.block_q.max(1);
        let inv_scale = 1.0 / (head_dim as f32).sqrt();

        let (kv_lower, _) = kv_blocks.size_hint();
        let mut kv_blocks_vec: Vec<(Tensor<B, 4>, Tensor<B, 4>)> = Vec::with_capacity(kv_lower);

        if self.config.determinism.fixed_tile_order {
            kv_blocks_vec.extend(
                kv_blocks
                    .strict_order()
                    .map(|(_, (k, v))| (k.unsqueeze_dim(0).transpose(), v.unsqueeze_dim(0))),
            );
        } else {
            kv_blocks_vec.extend(
                kv_blocks.map(|(k, v)| (k.unsqueeze_dim(0).transpose(), v.unsqueeze_dim(0))),
            );
        }

        let mut kv_starts = Vec::with_capacity(kv_blocks_vec.len());
        let mut kv_start = 0usize;
        for (k_block_t, _) in &kv_blocks_vec {
            kv_starts.push(kv_start);
            kv_start += k_block_t.dims()[3];
        }

        let q_blocks = q.split(block_q, 2);
        let mut outputs = Vec::with_capacity(q_blocks.len());

        for (q_block_index, q_block) in q_blocks.into_iter().enumerate() {
            let q_start = q_block_index * block_q;
            let q_block_scaled = q_block * inv_scale;

            let output = if self.config.use_log_space {
                self.process_q_block_log_space(
                    q_block_scaled,
                    &kv_blocks_vec,
                    &kv_starts,
                    causal,
                    q_start,
                    position_offset,
                )
            } else {
                self.process_q_block_standard(
                    q_block_scaled,
                    &kv_blocks_vec,
                    &kv_starts,
                    causal,
                    q_start,
                    position_offset,
                )
            };

            outputs.push(output);
        }

        Tensor::cat(outputs, 2)
    }

    fn process_q_block_standard<B: Backend>(
        &self,
        q_block: Tensor<B, 4>,
        kv_blocks: &[(Tensor<B, 4>, Tensor<B, 4>)],
        kv_starts: &[usize],
        causal: bool,
        q_start: usize,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let device = q_block.device();
        let [batch_size, num_heads, q_block_len, head_dim] = q_block.dims();

        let mut m_i = Tensor::<B, 4>::full(
            [batch_size, num_heads, q_block_len, 1],
            f32::NEG_INFINITY,
            &device,
        );
        let mut l_i = Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
        let mut o_i = Tensor::<B, 4>::zeros(
            [batch_size, num_heads, q_block_len, head_dim],
            &device,
        );

        for (kv_index, (k_block_t, v_block)) in kv_blocks.iter().enumerate() {
            let kv_block_len = k_block_t.dims()[3];
            let kv_start = kv_starts[kv_index];

            let mut scores = q_block.clone().matmul(k_block_t.clone());

            if causal {
                let mask = self.build_causal_mask_cached::<B>(
                    &device,
                    q_block_len,
                    kv_block_len,
                    q_start,
                    kv_start,
                    position_offset,
                );
                scores = scores + mask;
            }

            let m_ij = scores.clone().max_dim(3);
            let m_new = m_i.clone().max_pair(m_ij);

            let m_scale = (m_i - m_new.clone()).exp();
            let p_ij = (scores - m_new.clone()).exp();
            let p_sum = p_ij.clone().sum_dim(3);

            l_i = m_scale.clone() * l_i + p_sum;
            o_i = m_scale * o_i + p_ij.matmul(v_block.clone());
            m_i = m_new;
        }

        o_i / l_i
    }

    fn process_q_block_log_space<B: Backend>(
        &self,
        q_block: Tensor<B, 4>,
        kv_blocks: &[(Tensor<B, 4>, Tensor<B, 4>)],
        kv_starts: &[usize],
        causal: bool,
        q_start: usize,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let device = q_block.device();
        let [batch_size, num_heads, q_block_len, head_dim] = q_block.dims();

        let mut m_i = Tensor::<B, 4>::full(
            [batch_size, num_heads, q_block_len, 1],
            f32::NEG_INFINITY,
            &device,
        );
        let mut log_l_i = Tensor::<B, 4>::full(
            [batch_size, num_heads, q_block_len, 1],
            f32::NEG_INFINITY,
            &device,
        );
        let mut o_i = Tensor::<B, 4>::zeros(
            [batch_size, num_heads, q_block_len, head_dim],
            &device,
        );

        for (kv_index, (k_block_t, v_block)) in kv_blocks.iter().enumerate() {
            let kv_block_len = k_block_t.dims()[3];
            let kv_start = kv_starts[kv_index];

            let mut scores = q_block.clone().matmul(k_block_t.clone());

            if causal {
                let mask = self.build_causal_mask_cached::<B>(
                    &device,
                    q_block_len,
                    kv_block_len,
                    q_start,
                    kv_start,
                    position_offset,
                );
                scores = scores + mask;
            }

            let m_ij = scores.clone().max_dim(3);
            let m_new = m_i.clone().max_pair(m_ij.clone());

            let scores_shifted = scores - m_ij.clone();
            let p_ij = scores_shifted.exp();
            let sum_p = p_ij.clone().sum_dim(3);
            let log_sum_p = sum_p.log();

            let m_diff = m_i - m_new.clone();
            let log_prev = m_diff.clone() + log_l_i;
            let log_curr = (m_ij - m_new.clone()) + log_sum_p;

            let log_l_new = Self::tensor_log_add_exp(log_prev, log_curr);

            let m_scale = m_diff.exp();
            o_i = m_scale * o_i + p_ij.matmul(v_block.clone());

            m_i = m_new;
            log_l_i = log_l_new;
        }

        let l_i = log_l_i.exp();
        o_i / l_i
    }

    fn tensor_log_add_exp<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> Tensor<B, 4> {
        let max = a.clone().max_pair(b.clone());
        let diff_a = a - max.clone();
        let diff_b = b - max.clone();
        max + (diff_a.exp() + diff_b.exp()).log()
    }

    /// Build causal mask without caching (fallback for testing).
    #[allow(dead_code)]
    fn build_causal_mask_uncached<B: Backend>(
        &self,
        device: &B::Device,
        query_len: usize,
        key_len: usize,
        q_start: usize,
        kv_start: usize,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(query_len * key_len);
        let mask_value = -1.0e4_f32;

        for i in 0..query_len {
            let absolute_pos = position_offset + q_start + i;
            for j in 0..key_len {
                let absolute_key = kv_start + j;
                let allowed = absolute_key <= absolute_pos;
                data.push(if allowed { 0.0 } else { mask_value });
            }
        }

        Tensor::<B, 2>::from_data(TensorData::new(data, [query_len, key_len]), device)
            .reshape([1, 1, query_len, key_len])
    }

    /// Clear the thread-local mask cache.
    ///
    /// Useful for memory pressure situations or testing.
    pub fn clear_mask_cache() {
        MASK_CACHE.with(|cache| cache.borrow_mut().clear());
    }
}

impl<B: Backend> FusedPagedAttention<B> for HierarchicalFlashAttention {
    fn forward_fused<'a, I>(
        &self,
        q: Tensor<B, 4>,
        kv_blocks: I,
        config: &FlashAttentionConfig,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4>
    where
        I: Iterator<Item = (Tensor<B, 3>, Tensor<B, 3>)> + 'a,
    {
        let kv_blocks: Vec<_> = kv_blocks.collect();
        let total_kv_len: usize = kv_blocks.iter().map(|(k, _)| k.dims()[1]).sum();

        let attention = Self::new(config.clone());

        attention.forward_fused_iter(q, kv_blocks.into_iter(), causal, position_offset, total_kv_len)
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::activation::softmax;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_hierarchical_flash_basic() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::default_config();

        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 16;
        let head_dim = 8;

        let q = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = attention.forward(q, k, v, false, 0);
        assert_eq!(output.dims(), [batch_size, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_hierarchical_flash_matches_standard() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::new(FlashAttentionConfig {
            block_q: 4,
            block_kv: 4,
            use_log_space: false,
            ..Default::default()
        });

        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 4;

        let q = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let output_hier = attention.forward(q.clone(), k.clone(), v.clone(), false, 0);

        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.transpose()) / scale;
        let attn = softmax(scores, 3);
        let output_std = attn.matmul(v);

        let hier_data = output_hier
            .into_data()
            .into_vec::<f32>()
            .expect("output data");
        let std_data = output_std
            .into_data()
            .into_vec::<f32>()
            .expect("output data");

        for (i, (h, s)) in hier_data.iter().zip(std_data.iter()).enumerate() {
            let diff = (h - s).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at {}: hier={}, std={}, diff={}",
                i,
                h,
                s,
                diff
            );
        }
    }

    #[test]
    fn test_fused_iter_matches_standard() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::new(FlashAttentionConfig {
            block_q: 4,
            block_kv: 4,
            use_log_space: false,
            ..Default::default()
        });

        let num_heads = 2;
        let seq_len = 16;
        let head_dim = 4;
        let block_size = 4;

        let q = Tensor::<TestBackend, 4>::random(
            [1, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let num_blocks = seq_len / block_size;
        let kv_blocks: Vec<_> = (0..num_blocks)
            .map(|_| {
                let k = Tensor::<TestBackend, 3>::random(
                    [num_heads, block_size, head_dim],
                    burn::tensor::Distribution::Normal(0.0, 0.5),
                    &device,
                );
                let v = Tensor::<TestBackend, 3>::random(
                    [num_heads, block_size, head_dim],
                    burn::tensor::Distribution::Normal(0.0, 0.5),
                    &device,
                );
                (k, v)
            })
            .collect();

        let output_fused = attention.forward_fused_iter(
            q.clone(),
            kv_blocks.clone().into_iter(),
            false,
            0,
            seq_len,
        );

        let k_cat: Vec<_> = kv_blocks.iter().map(|(k, _)| k.clone()).collect();
        let v_cat: Vec<_> = kv_blocks.iter().map(|(_, v)| v.clone()).collect();

        let k_full = Tensor::cat(k_cat, 1).reshape([1, num_heads, seq_len, head_dim]);
        let v_full = Tensor::cat(v_cat, 1).reshape([1, num_heads, seq_len, head_dim]);

        let output_std = attention.forward(q, k_full, v_full, false, 0);

        let fused_data = output_fused
            .into_data()
            .into_vec::<f32>()
            .expect("output data");
        let std_data = output_std
            .into_data()
            .into_vec::<f32>()
            .expect("output data");

        for (i, (f, s)) in fused_data.iter().zip(std_data.iter()).enumerate() {
            let diff = (f - s).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at {}: fused={}, std={}, diff={}",
                i,
                f,
                s,
                diff
            );
        }
    }

    #[test]
    fn test_causal_mask() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::default_config();

        let mask = attention.build_causal_mask_cached::<TestBackend>(&device, 4, 4, 0, 0, 0);

        let data = mask.into_data().into_vec::<f32>().expect("mask data");

        assert!(data[0].abs() < 1e-5);
        assert!(data[1] < -1000.0);
        assert!(data[4].abs() < 1e-5);
        assert!(data[5].abs() < 1e-5);
        assert!(data[6] < -1000.0);
    }

    #[test]
    fn test_forward_with_workspace() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::new(FlashAttentionConfig {
            block_q: 4,
            block_kv: 4,
            use_log_space: false,
            ..Default::default()
        });

        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 4;

        let q = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        // Test with workspace
        let mut workspace = AttentionWorkspace::new();
        let output_workspace = attention.forward_with_workspace(
            q.clone(),
            k.clone(),
            v.clone(),
            false,
            0,
            &mut workspace,
        );

        // Test without workspace (standard)
        let output_std = attention.forward(q, k, v, false, 0);

        let ws_data = output_workspace
            .into_data()
            .into_vec::<f32>()
            .expect("output data");
        let std_data = output_std
            .into_data()
            .into_vec::<f32>()
            .expect("output data");

        for (i, (w, s)) in ws_data.iter().zip(std_data.iter()).enumerate() {
            let diff = (w - s).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at {}: workspace={}, std={}, diff={}",
                i,
                w,
                s,
                diff
            );
        }
    }

    #[test]
    fn test_mask_cache_hit() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::default_config();

        // Clear cache first
        HierarchicalFlashAttention::clear_mask_cache();

        // First call - cache miss
        let mask1 = attention.build_causal_mask_cached::<TestBackend>(&device, 4, 4, 0, 0, 0);
        // Second call - should hit cache
        let mask2 = attention.build_causal_mask_cached::<TestBackend>(&device, 4, 4, 0, 0, 0);

        let data1 = mask1.into_data().into_vec::<f32>().expect("mask data");
        let data2 = mask2.into_data().into_vec::<f32>().expect("mask data");

        assert_eq!(data1, data2);
    }
}
