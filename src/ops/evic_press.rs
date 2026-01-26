//! EvicPress: Joint Compression and Eviction for KV Cache.
//!
//! Based on EvicPress paper: Jointly optimizing quantization and eviction
//! strategies for efficient KV cache management.
//!
//! # Key Features
//! - Progressive quantization (FP16 -> INT8 -> INT2)
//! - Importance-based eviction (attention scores + position decay)
//! - Joint decision making for compression and eviction
//! - Three-zone architecture: Hot (FP16), Warm (INT8), Cold (INT2)

use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::kernel_types::KernelFloat;

use super::int2_quantizer::Int2PackedBuffer;

/// EvicPress configuration.
#[derive(Debug, Clone, Copy)]
pub struct EvicPressConfig {
    /// Maximum KV cache size in tokens.
    pub max_cache_size: usize,
    /// Cache utilization threshold to trigger eviction (default: 0.9).
    pub eviction_threshold: f32,
    /// Position decay factor for importance (default: 0.99).
    pub importance_decay: f32,
    /// Attention weight importance factor (default: 0.6).
    pub attention_weight: f32,
    /// Semantic importance factor (default: 0.4).
    pub semantic_weight: f32,
    /// Minimum tokens to keep (sink + critical).
    pub min_keep_tokens: usize,
    /// Hot zone size (FP16, most recent).
    pub hot_zone_size: usize,
    /// Warm zone size (INT8).
    pub warm_zone_size: usize,
    /// INT2 group size for cold zone.
    pub int2_group_size: usize,
}

impl Default for EvicPressConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            max_cache_size: 4096,
            eviction_threshold: 0.9,
            importance_decay: 0.99,
            attention_weight: 0.6,
            semantic_weight: 0.4,
            min_keep_tokens: 128,
            hot_zone_size: 64,
            warm_zone_size: 192,
            int2_group_size: 128,
        }
    }
}

impl EvicPressConfig {
    /// Validate configuration.
    #[inline(always)]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_cache_size == 0 {
            return Err("max_cache_size must be > 0");
        }
        if self.eviction_threshold <= 0.0 || self.eviction_threshold > 1.0 {
            return Err("eviction_threshold must be in (0, 1]");
        }
        if self.importance_decay <= 0.0 || self.importance_decay > 1.0 {
            return Err("importance_decay must be in (0, 1]");
        }
        if self.attention_weight < 0.0 || self.semantic_weight < 0.0 {
            return Err("weights must be >= 0");
        }
        if self.min_keep_tokens > self.max_cache_size {
            return Err("min_keep_tokens must be <= max_cache_size");
        }
        if self.hot_zone_size == 0 {
            return Err("hot_zone_size must be > 0");
        }
        if self.int2_group_size == 0 {
            return Err("int2_group_size must be > 0");
        }
        Ok(())
    }

    /// Cold zone capacity (remaining after hot + warm).
    #[inline(always)]
    pub fn cold_zone_capacity(&self) -> usize {
        self.max_cache_size
            .saturating_sub(self.hot_zone_size)
            .saturating_sub(self.warm_zone_size)
    }
}

/// Token importance score with metadata.
#[derive(Debug, Clone, Copy)]
pub struct TokenImportance {
    /// Token position in original sequence.
    pub position: usize,
    /// Cumulative attention score received.
    pub attention_score: f32,
    /// Semantic importance (from model).
    pub semantic_importance: f32,
    /// Combined importance score.
    pub combined_score: f32,
    /// Whether this is a sink token (always keep).
    pub is_sink: bool,
    /// Current storage zone.
    pub zone: StorageZone,
}

impl TokenImportance {
    /// Create a new token importance entry.
    #[inline(always)]
    pub fn new(position: usize, is_sink: bool) -> Self {
        Self {
            position,
            attention_score: 1.0,
            semantic_importance: 1.0,
            combined_score: 1.0,
            is_sink,
            zone: StorageZone::Hot,
        }
    }

    /// Update combined score with config weights.
    #[inline(always)]
    pub fn update_combined(&mut self, config: &EvicPressConfig) {
        self.combined_score = config.attention_weight * self.attention_score
            + config.semantic_weight * self.semantic_importance;
    }
}

/// Storage zone for KV data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageZone {
    /// Hot zone: FP16, most recent tokens.
    Hot,
    /// Warm zone: INT8 quantized.
    Warm,
    /// Cold zone: INT2 quantized.
    Cold,
    /// Evicted: removed from cache.
    Evicted,
}

/// KV entry in the progressive cache.
#[derive(Debug, Clone)]
pub enum KVEntry<T: KernelFloat> {
    /// Full precision (FP16/FP32).
    Full { k: Vec<T>, v: Vec<T> },
    /// INT8 quantized.
    Int8 {
        k: Vec<i8>,
        v: Vec<i8>,
        k_scale: f32,
        v_scale: f32,
    },
    /// INT2 packed.
    Int2 {
        k: Int2PackedBuffer,
        v: Int2PackedBuffer,
    },
}

impl<T: KernelFloat> KVEntry<T> {
    /// Get approximate memory size in bytes.
    #[inline(always)]
    pub fn memory_size(&self) -> usize {
        match self {
            KVEntry::Full { k, v } => (k.len() + v.len()) * std::mem::size_of::<T>(),
            KVEntry::Int8 { k, v, .. } => k.len() + v.len() + 8,
            KVEntry::Int2 { k, v } => {
                k.packed_size() + v.packed_size() + k.scales().len() * 4 + v.scales().len() * 4
            }
        }
    }

    /// Get head dimension (assumes k and v have same dim).
    #[inline(always)]
    pub fn head_dim(&self) -> usize {
        match self {
            KVEntry::Full { k, .. } => k.len(),
            KVEntry::Int8 { k, .. } => k.len(),
            KVEntry::Int2 { k, .. } => k.len(),
        }
    }
}

/// Progressive KV cache with three-zone architecture.
#[derive(Debug)]
pub struct ProgressiveKVCache<T: KernelFloat> {
    /// Configuration.
    config: EvicPressConfig,
    /// Hot zone entries (FP16, most recent).
    hot_zone: VecDeque<(TokenImportance, KVEntry<T>)>,
    /// Warm zone entries (INT8).
    warm_zone: VecDeque<(TokenImportance, KVEntry<T>)>,
    /// Cold zone entries (INT2).
    cold_zone: VecDeque<(TokenImportance, KVEntry<T>)>,
    /// Total tokens (including evicted for tracking).
    total_tokens: usize,
    /// Number of heads (used for future multi-head KV operations).
    #[allow(dead_code)]
    num_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Phantom marker.
    _marker: PhantomData<T>,
}

impl<T: KernelFloat> ProgressiveKVCache<T> {
    /// Create a new progressive KV cache.
    #[inline(always)]
    pub fn new(
        config: EvicPressConfig,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self, &'static str> {
        config.validate()?;

        Ok(Self {
            config,
            hot_zone: VecDeque::with_capacity(config.hot_zone_size),
            warm_zone: VecDeque::with_capacity(config.warm_zone_size),
            cold_zone: VecDeque::with_capacity(config.cold_zone_capacity()),
            total_tokens: 0,
            num_heads,
            head_dim,
            _marker: PhantomData,
        })
    }

    /// Get configuration.
    #[inline(always)]
    pub fn config(&self) -> &EvicPressConfig {
        &self.config
    }

    /// Get total number of cached tokens.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.hot_zone.len() + self.warm_zone.len() + self.cold_zone.len()
    }

    /// Check if cache is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get current utilization ratio.
    #[inline(always)]
    pub fn utilization(&self) -> f32 {
        self.len() as f32 / self.config.max_cache_size as f32
    }

    /// Append new KV pair.
    #[inline(always)]
    pub fn append(&mut self, k: &[T], v: &[T], is_sink: bool) -> Result<(), &'static str> {
        if k.len() != self.head_dim || v.len() != self.head_dim {
            return Err("k/v dimension mismatch");
        }

        let mut importance = TokenImportance::new(self.total_tokens, is_sink);
        importance.update_combined(&self.config);

        let entry = KVEntry::Full {
            k: k.to_vec(),
            v: v.to_vec(),
        };

        // Add to hot zone
        self.hot_zone.push_back((importance, entry));
        self.total_tokens += 1;

        // Manage zones if needed
        self.manage_zones()?;

        Ok(())
    }

    /// Append multiple KV pairs.
    #[inline(always)]
    pub fn append_batch(
        &mut self,
        k: &[T],
        v: &[T],
        num_tokens: usize,
        num_sinks: usize,
    ) -> Result<(), &'static str> {
        let expected_len = num_tokens * self.head_dim;
        if k.len() != expected_len || v.len() != expected_len {
            return Err("batch k/v dimension mismatch");
        }

        for i in 0..num_tokens {
            let start = i * self.head_dim;
            let end = start + self.head_dim;
            let is_sink = i < num_sinks;
            self.append(&k[start..end], &v[start..end], is_sink)?;
        }

        Ok(())
    }

    /// Update attention scores for importance tracking.
    #[inline(always)]
    pub fn update_attention_scores(&mut self, scores: &[f32]) {
        // Decay existing scores
        let decay = self.config.importance_decay;

        for (importance, _) in self.hot_zone.iter_mut() {
            importance.attention_score *= decay;
        }
        for (importance, _) in self.warm_zone.iter_mut() {
            importance.attention_score *= decay;
        }
        for (importance, _) in self.cold_zone.iter_mut() {
            importance.attention_score *= decay;
        }

        // Add new scores
        let total_cached = self.len();
        for (i, &score) in scores.iter().enumerate().take(total_cached) {
            self.update_token_attention(i, score);
        }

        // Recalculate combined scores
        self.recalculate_importance();
    }

    #[inline(always)]
    fn update_token_attention(&mut self, global_idx: usize, score: f32) {
        let hot_len = self.hot_zone.len();
        let warm_len = self.warm_zone.len();

        if global_idx < self.cold_zone.len() {
            if let Some((importance, _)) = self.cold_zone.get_mut(global_idx) {
                importance.attention_score += score;
            }
        } else if global_idx < self.cold_zone.len() + warm_len {
            let warm_idx = global_idx - self.cold_zone.len();
            if let Some((importance, _)) = self.warm_zone.get_mut(warm_idx) {
                importance.attention_score += score;
            }
        } else if global_idx < self.cold_zone.len() + warm_len + hot_len {
            let hot_idx = global_idx - self.cold_zone.len() - warm_len;
            if let Some((importance, _)) = self.hot_zone.get_mut(hot_idx) {
                importance.attention_score += score;
            }
        }
    }

    #[inline(always)]
    fn recalculate_importance(&mut self) {
        for (importance, _) in self.hot_zone.iter_mut() {
            importance.update_combined(&self.config);
        }
        for (importance, _) in self.warm_zone.iter_mut() {
            importance.update_combined(&self.config);
        }
        for (importance, _) in self.cold_zone.iter_mut() {
            importance.update_combined(&self.config);
        }
    }

    /// Manage zone transitions and eviction.
    #[inline(always)]
    fn manage_zones(&mut self) -> Result<(), &'static str> {
        // Move from hot to warm if hot zone is full
        while self.hot_zone.len() > self.config.hot_zone_size {
            if let Some((mut importance, entry)) = self.hot_zone.pop_front() {
                if importance.is_sink {
                    // Sink tokens stay in hot zone at the front
                    self.hot_zone.push_front((importance, entry));
                    break;
                }

                // Quantize to INT8
                let int8_entry = self.quantize_to_int8(entry)?;
                importance.zone = StorageZone::Warm;
                self.warm_zone.push_back((importance, int8_entry));
            }
        }

        // Move from warm to cold if warm zone is full
        while self.warm_zone.len() > self.config.warm_zone_size {
            if let Some((mut importance, entry)) = self.warm_zone.pop_front() {
                // Quantize to INT2
                let int2_entry = self.quantize_to_int2(entry)?;
                importance.zone = StorageZone::Cold;
                self.cold_zone.push_back((importance, int2_entry));
            }
        }

        // Evict if total exceeds max
        while self.len() > self.config.max_cache_size
            && self.len() > self.config.min_keep_tokens
        {
            self.evict_lowest_importance()?;
        }

        Ok(())
    }

    /// Quantize entry from FP to INT8.
    #[inline(always)]
    fn quantize_to_int8(&self, entry: KVEntry<T>) -> Result<KVEntry<T>, &'static str> {
        match entry {
            KVEntry::Full { k, v } => {
                let k_absmax = k
                    .iter()
                    .map(|&x| x.to_f32().abs())
                    .fold(0.0f32, f32::max);
                let v_absmax = v
                    .iter()
                    .map(|&x| x.to_f32().abs())
                    .fold(0.0f32, f32::max);

                let k_scale = if k_absmax > 0.0 { k_absmax / 127.0 } else { 1.0 };
                let v_scale = if v_absmax > 0.0 { v_absmax / 127.0 } else { 1.0 };

                let k_int8: Vec<i8> = k
                    .iter()
                    .map(|&x| (x.to_f32() / k_scale).round().clamp(-127.0, 127.0) as i8)
                    .collect();
                let v_int8: Vec<i8> = v
                    .iter()
                    .map(|&x| (x.to_f32() / v_scale).round().clamp(-127.0, 127.0) as i8)
                    .collect();

                Ok(KVEntry::Int8 {
                    k: k_int8,
                    v: v_int8,
                    k_scale,
                    v_scale,
                })
            }
            _ => Err("can only quantize FP to INT8"),
        }
    }

    /// Quantize entry to INT2.
    #[inline(always)]
    fn quantize_to_int2(&self, entry: KVEntry<T>) -> Result<KVEntry<T>, &'static str> {
        let (k_f32, v_f32) = match entry {
            KVEntry::Full { k, v } => {
                let k_f32: Vec<f32> = k.iter().map(|&x| x.to_f32()).collect();
                let v_f32: Vec<f32> = v.iter().map(|&x| x.to_f32()).collect();
                (k_f32, v_f32)
            }
            KVEntry::Int8 {
                k,
                v,
                k_scale,
                v_scale,
            } => {
                let k_f32: Vec<f32> = k.iter().map(|&x| x as f32 * k_scale).collect();
                let v_f32: Vec<f32> = v.iter().map(|&x| x as f32 * v_scale).collect();
                (k_f32, v_f32)
            }
            KVEntry::Int2 { .. } => return Err("already INT2"),
        };

        let k_packed = Int2PackedBuffer::from_f32(&k_f32, self.config.int2_group_size);
        let v_packed = Int2PackedBuffer::from_f32(&v_f32, self.config.int2_group_size);

        Ok(KVEntry::Int2 {
            k: k_packed,
            v: v_packed,
        })
    }

    /// Evict the token with lowest importance (excluding sinks).
    #[inline(always)]
    fn evict_lowest_importance(&mut self) -> Result<(), &'static str> {
        // Find lowest importance in cold zone (non-sink)
        let mut lowest_idx = None;
        let mut lowest_score = f32::INFINITY;

        for (i, (importance, _)) in self.cold_zone.iter().enumerate() {
            if !importance.is_sink && importance.combined_score < lowest_score {
                lowest_score = importance.combined_score;
                lowest_idx = Some(i);
            }
        }

        // Also check warm zone if cold is empty
        if lowest_idx.is_none() {
            for (i, (importance, _)) in self.warm_zone.iter().enumerate() {
                if !importance.is_sink && importance.combined_score < lowest_score {
                    lowest_score = importance.combined_score;
                    lowest_idx = Some(self.cold_zone.len() + i);
                }
            }
        }

        match lowest_idx {
            Some(idx) if idx < self.cold_zone.len() => {
                self.cold_zone.remove(idx);
                Ok(())
            }
            Some(idx) => {
                let warm_idx = idx - self.cold_zone.len();
                self.warm_zone.remove(warm_idx);
                Ok(())
            }
            None => Err("no evictable tokens found"),
        }
    }

    /// Get all KV pairs as contiguous tensors (for attention).
    #[inline(always)]
    pub fn get_all_kv(&self) -> Result<(Vec<T>, Vec<T>), &'static str> {
        let total = self.len();
        let mut all_k = Vec::with_capacity(total * self.head_dim);
        let mut all_v = Vec::with_capacity(total * self.head_dim);

        // Cold zone first (oldest)
        for (_, entry) in &self.cold_zone {
            let (k, v) = self.dequantize_entry(entry)?;
            all_k.extend(k);
            all_v.extend(v);
        }

        // Warm zone
        for (_, entry) in &self.warm_zone {
            let (k, v) = self.dequantize_entry(entry)?;
            all_k.extend(k);
            all_v.extend(v);
        }

        // Hot zone (newest)
        for (_, entry) in &self.hot_zone {
            let (k, v) = self.dequantize_entry(entry)?;
            all_k.extend(k);
            all_v.extend(v);
        }

        Ok((all_k, all_v))
    }

    #[inline(always)]
    fn dequantize_entry(&self, entry: &KVEntry<T>) -> Result<(Vec<T>, Vec<T>), &'static str> {
        match entry {
            KVEntry::Full { k, v } => Ok((k.clone(), v.clone())),
            KVEntry::Int8 {
                k,
                v,
                k_scale,
                v_scale,
            } => {
                let k_f32: Vec<T> = k
                    .iter()
                    .map(|&x| T::from_f32(x as f32 * k_scale))
                    .collect();
                let v_f32: Vec<T> = v
                    .iter()
                    .map(|&x| T::from_f32(x as f32 * v_scale))
                    .collect();
                Ok((k_f32, v_f32))
            }
            KVEntry::Int2 { k, v } => {
                let k_vec: Vec<T> = k.to_f32().into_iter().map(T::from_f32).collect();
                let v_vec: Vec<T> = v.to_f32().into_iter().map(T::from_f32).collect();
                Ok((k_vec, v_vec))
            }
        }
    }

    /// Get memory usage statistics.
    #[inline(always)]
    pub fn memory_stats(&self) -> MemoryStats {
        let hot_bytes: usize = self.hot_zone.iter().map(|(_, e)| e.memory_size()).sum();
        let warm_bytes: usize = self.warm_zone.iter().map(|(_, e)| e.memory_size()).sum();
        let cold_bytes: usize = self.cold_zone.iter().map(|(_, e)| e.memory_size()).sum();

        let fp16_equivalent = self.len() * self.head_dim * 2 * 2; // k + v, 2 bytes each

        MemoryStats {
            hot_zone_bytes: hot_bytes,
            warm_zone_bytes: warm_bytes,
            cold_zone_bytes: cold_bytes,
            total_bytes: hot_bytes + warm_bytes + cold_bytes,
            fp16_equivalent_bytes: fp16_equivalent,
            compression_ratio: if (hot_bytes + warm_bytes + cold_bytes) > 0 {
                fp16_equivalent as f32 / (hot_bytes + warm_bytes + cold_bytes) as f32
            } else {
                1.0
            },
            hot_zone_tokens: self.hot_zone.len(),
            warm_zone_tokens: self.warm_zone.len(),
            cold_zone_tokens: self.cold_zone.len(),
        }
    }

    /// Clear the cache.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.hot_zone.clear();
        self.warm_zone.clear();
        self.cold_zone.clear();
        self.total_tokens = 0;
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Bytes in hot zone.
    pub hot_zone_bytes: usize,
    /// Bytes in warm zone.
    pub warm_zone_bytes: usize,
    /// Bytes in cold zone.
    pub cold_zone_bytes: usize,
    /// Total bytes used.
    pub total_bytes: usize,
    /// Equivalent FP16 bytes.
    pub fp16_equivalent_bytes: usize,
    /// Compression ratio vs FP16.
    pub compression_ratio: f32,
    /// Tokens in hot zone.
    pub hot_zone_tokens: usize,
    /// Tokens in warm zone.
    pub warm_zone_tokens: usize,
    /// Tokens in cold zone.
    pub cold_zone_tokens: usize,
}

/// EvicPress decision for a token.
#[derive(Debug, Clone, Copy)]
pub enum EvicPressDecision {
    /// Keep in current zone.
    Keep,
    /// Compress to lower precision.
    Compress { target_zone: StorageZone },
    /// Evict from cache.
    Evict,
}

/// Make EvicPress decision based on cache state and importance.
#[inline(always)]
pub fn make_evicpress_decision(
    utilization: f32,
    importance: &TokenImportance,
    config: &EvicPressConfig,
) -> EvicPressDecision {
    // Sink tokens are never evicted or compressed
    if importance.is_sink {
        return EvicPressDecision::Keep;
    }

    // Low utilization: no action needed
    if utilization < 0.5 {
        return EvicPressDecision::Keep;
    }

    // Medium utilization: compress low-importance tokens
    if utilization < config.eviction_threshold {
        match importance.zone {
            StorageZone::Hot if importance.combined_score < 0.3 => {
                return EvicPressDecision::Compress {
                    target_zone: StorageZone::Warm,
                };
            }
            StorageZone::Warm if importance.combined_score < 0.2 => {
                return EvicPressDecision::Compress {
                    target_zone: StorageZone::Cold,
                };
            }
            _ => return EvicPressDecision::Keep,
        }
    }

    // High utilization: joint compress + evict
    match importance.zone {
        StorageZone::Hot if importance.combined_score < 0.5 => EvicPressDecision::Compress {
            target_zone: StorageZone::Warm,
        },
        StorageZone::Warm if importance.combined_score < 0.3 => EvicPressDecision::Compress {
            target_zone: StorageZone::Cold,
        },
        StorageZone::Cold if importance.combined_score < 0.1 => EvicPressDecision::Evict,
        _ => EvicPressDecision::Keep,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evicpress_config() {
        let config = EvicPressConfig::default();
        assert!(config.validate().is_ok());

        let invalid = EvicPressConfig {
            max_cache_size: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_token_importance() {
        let config = EvicPressConfig::default();
        let mut importance = TokenImportance::new(0, false);
        importance.attention_score = 0.8;
        importance.semantic_importance = 0.6;
        importance.update_combined(&config);

        let expected = 0.6 * 0.8 + 0.4 * 0.6;
        assert!((importance.combined_score - expected).abs() < 0.001);
    }

    #[test]
    fn test_progressive_cache_append() {
        let config = EvicPressConfig {
            max_cache_size: 100,
            hot_zone_size: 10,
            warm_zone_size: 20,
            min_keep_tokens: 8,
            ..Default::default()
        };

        let mut cache = ProgressiveKVCache::<f32>::new(config, 8, 64).unwrap();

        // Append tokens without sink tokens first (so they can be moved to warm zone)
        for i in 0..30 {
            let k = vec![i as f32; 64];
            let v = vec![i as f32; 64];
            let is_sink = false; // No sink tokens to allow zone transitions
            cache.append(&k, &v, is_sink).unwrap();
        }

        assert_eq!(cache.len(), 30);
        assert!(!cache.hot_zone.is_empty());
        // With 30 tokens and hot_zone_size=10, warm_zone should have tokens
        assert!(!cache.warm_zone.is_empty());
    }

    #[test]
    fn test_zone_transitions() {
        let config = EvicPressConfig {
            max_cache_size: 50,
            hot_zone_size: 5,
            warm_zone_size: 10,
            min_keep_tokens: 4,
            ..Default::default()
        };

        let mut cache = ProgressiveKVCache::<f32>::new(config, 1, 16).unwrap();

        // Fill beyond hot zone
        for i in 0..20 {
            let k = vec![i as f32; 16];
            let v = vec![i as f32; 16];
            cache.append(&k, &v, false).unwrap();
        }

        // Should have tokens in warm zone now
        assert!(!cache.warm_zone.is_empty());

        // Fill beyond warm zone
        for i in 20..40 {
            let k = vec![i as f32; 16];
            let v = vec![i as f32; 16];
            cache.append(&k, &v, false).unwrap();
        }

        // Should have tokens in cold zone now
        assert!(!cache.cold_zone.is_empty());
    }

    #[test]
    fn test_memory_stats() {
        let config = EvicPressConfig {
            max_cache_size: 100,
            hot_zone_size: 20,
            warm_zone_size: 30,
            min_keep_tokens: 8,
            ..Default::default()
        };

        let mut cache = ProgressiveKVCache::<f32>::new(config, 1, 64).unwrap();

        for i in 0..60 {
            let k = vec![i as f32; 64];
            let v = vec![i as f32; 64];
            cache.append(&k, &v, false).unwrap();
        }

        let stats = cache.memory_stats();
        assert!(stats.compression_ratio > 1.0); // Should have some compression
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_evicpress_decision() {
        let config = EvicPressConfig::default();

        // Sink token: always keep
        let sink = TokenImportance {
            position: 0,
            attention_score: 0.0,
            semantic_importance: 0.0,
            combined_score: 0.0,
            is_sink: true,
            zone: StorageZone::Hot,
        };
        assert!(matches!(
            make_evicpress_decision(0.95, &sink, &config),
            EvicPressDecision::Keep
        ));

        // Low importance cold token at high utilization: evict
        let cold_low = TokenImportance {
            position: 100,
            attention_score: 0.05,
            semantic_importance: 0.05,
            combined_score: 0.05,
            is_sink: false,
            zone: StorageZone::Cold,
        };
        assert!(matches!(
            make_evicpress_decision(0.95, &cold_low, &config),
            EvicPressDecision::Evict
        ));
    }
}
