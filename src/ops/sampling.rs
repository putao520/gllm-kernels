//! Sampling operations for language model token generation.
//!
//! This module provides pure Rust implementations of sampling algorithms
//! including top-k selection, temperature scaling, and nucleus (top-p) sampling.
//!
//! These are GPU-acceleratable operations commonly used in LLM inference.

use crate::kernel_types::KernelFloat;

/// Configuration for sampling operations.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for logit scaling (0.0 = greedy, 1.0 = original distribution).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold (0.0-1.0, 0.0 = disabled).
    pub top_p: f32,
    /// Top-k sampling limit (0 = disabled).
    pub top_k: usize,
    /// Random seed for reproducibility (None = use system entropy).
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Create a greedy sampling config (temperature=0, no randomness).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            seed: None,
        }
    }

    /// Create a config with specific temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }
}

/// Result of top-k selection operation.
#[derive(Debug, Clone)]
pub struct TopKResult {
    /// Indices of the top-k elements (sorted by value descending).
    pub indices: Vec<u32>,
    /// Values of the top-k elements (sorted descending).
    pub values: Vec<f32>,
}

/// Select top-k elements from each row of logits.
///
/// # Arguments
/// * `logits` - Input logits: [batch_size, vocab_size]
/// * `k` - Number of top elements to select
/// * `batch_size` - Number of rows
/// * `vocab_size` - Number of columns (vocabulary size)
///
/// # Returns
/// TopKResult with indices and values of shape [batch_size, k]
pub fn topk<T: KernelFloat>(
    logits: &[T],
    k: usize,
    batch_size: usize,
    vocab_size: usize,
) -> TopKResult {
    assert_eq!(logits.len(), batch_size * vocab_size);
    let k = k.min(vocab_size);

    let mut all_indices = Vec::with_capacity(batch_size * k);
    let mut all_values = Vec::with_capacity(batch_size * k);

    for batch in 0..batch_size {
        let row_start = batch * vocab_size;
        let row = &logits[row_start..row_start + vocab_size];

        // Create index-value pairs
        let mut indexed: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f32()))
            .collect();

        // Partial sort to get top-k (more efficient than full sort for large vocab)
        indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top-k and sort them
        let mut topk_items: Vec<_> = indexed[..k].to_vec();
        topk_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, val) in topk_items {
            all_indices.push(idx as u32);
            all_values.push(val);
        }
    }

    TopKResult {
        indices: all_indices,
        values: all_values,
    }
}

/// Apply temperature scaling to logits in-place.
///
/// # Arguments
/// * `logits` - Input/output logits
/// * `temperature` - Temperature value (> 0)
#[inline]
pub fn apply_temperature<T: KernelFloat>(logits: &mut [T], temperature: f32) {
    if temperature <= 0.0 || (temperature - 1.0).abs() < f32::EPSILON {
        return;
    }
    let inv_temp = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v = T::from_f32(v.to_f32() * inv_temp);
    }
}

/// Compute softmax probabilities from logits.
///
/// # Arguments
/// * `logits` - Input logits
/// * `probs` - Output probabilities (same length as logits)
pub fn softmax_1d(logits: &[f32], probs: &mut [f32]) {
    assert_eq!(logits.len(), probs.len());
    if logits.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let exp_v = (v - max_val).exp();
        probs[i] = exp_v;
        sum += exp_v;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
    }
}

/// Apply top-p (nucleus) filtering to sorted probabilities.
///
/// # Arguments
/// * `probs` - Sorted probabilities (descending)
/// * `top_p` - Cumulative probability threshold
///
/// # Returns
/// Number of elements to keep
pub fn apply_top_p(probs: &[f32], top_p: f32) -> usize {
    if top_p <= 0.0 || top_p >= 1.0 {
        return probs.len();
    }

    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            return i + 1;
        }
    }
    probs.len()
}

/// Simple LCG random number generator for reproducibility.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG constants from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Sample from a categorical distribution given probabilities.
///
/// # Arguments
/// * `probs` - Probabilities (should sum to ~1.0)
/// * `rng` - Random number generator state
///
/// # Returns
/// Sampled index
fn sample_categorical(probs: &[f32], rng: &mut SimpleRng) -> usize {
    let r = rng.next_f32();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Sample next tokens from logits using temperature, top-k, and top-p.
///
/// # Arguments
/// * `logits` - Input logits: [batch_size, vocab_size]
/// * `batch_size` - Number of sequences
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration
///
/// # Returns
/// Vector of sampled token indices (one per batch)
pub fn sample_tokens<T: KernelFloat>(
    logits: &[T],
    batch_size: usize,
    vocab_size: usize,
    config: &SamplingConfig,
) -> Vec<u32> {
    assert_eq!(logits.len(), batch_size * vocab_size);

    // Greedy decoding
    if config.temperature <= 0.0 {
        let mut results = Vec::with_capacity(batch_size);
        for batch in 0..batch_size {
            let row_start = batch * vocab_size;
            let row = &logits[row_start..row_start + vocab_size];
            let (max_idx, _) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.to_f32().partial_cmp(&b.to_f32()).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or((0, &T::from_f32(0.0)));
            results.push(max_idx as u32);
        }
        return results;
    }

    // Initialize RNG
    let seed = config.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    });
    let mut rng = SimpleRng::new(seed);

    let mut results = Vec::with_capacity(batch_size);

    // Determine effective k
    let k = if config.top_k > 0 {
        config.top_k.min(vocab_size)
    } else {
        vocab_size
    };

    for batch in 0..batch_size {
        let row_start = batch * vocab_size;
        let row = &logits[row_start..row_start + vocab_size];

        // Apply temperature and convert to f32
        let scaled: Vec<f32> = row
            .iter()
            .map(|v| v.to_f32() / config.temperature)
            .collect();

        // Get top-k candidates
        let mut indexed: Vec<(usize, f32)> = scaled
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Partial sort for top-k
        if k < vocab_size {
            indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
        }

        // Sort top-k by value (descending)
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute softmax on top-k logits
        let topk_logits: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
        let mut topk_probs = vec![0.0f32; topk_logits.len()];
        softmax_1d(&topk_logits, &mut topk_probs);

        // Apply top-p filtering
        let keep_count = apply_top_p(&topk_probs, config.top_p);
        let filtered_probs = &topk_probs[..keep_count];
        let filtered_indices: Vec<usize> = indexed[..keep_count].iter().map(|(i, _)| *i).collect();

        // Renormalize after top-p
        let sum: f32 = filtered_probs.iter().sum();
        let normalized: Vec<f32> = if sum > 0.0 {
            filtered_probs.iter().map(|p| p / sum).collect()
        } else {
            vec![1.0 / keep_count as f32; keep_count]
        };

        // Sample from distribution
        let sample_idx = sample_categorical(&normalized, &mut rng);
        results.push(filtered_indices[sample_idx] as u32);
    }

    results
}

/// Argmax operation - find index of maximum value in each row.
///
/// # Arguments
/// * `logits` - Input logits: [batch_size, vocab_size]
/// * `batch_size` - Number of rows
/// * `vocab_size` - Number of columns
///
/// # Returns
/// Vector of indices of maximum values
pub fn argmax<T: KernelFloat>(logits: &[T], batch_size: usize, vocab_size: usize) -> Vec<u32> {
    assert_eq!(logits.len(), batch_size * vocab_size);

    let mut results = Vec::with_capacity(batch_size);
    for batch in 0..batch_size {
        let row_start = batch * vocab_size;
        let row = &logits[row_start..row_start + vocab_size];
        let (max_idx, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.to_f32().partial_cmp(&b.to_f32()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &T::from_f32(0.0)));
        results.push(max_idx as u32);
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_basic() {
        let logits: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let result = topk(&logits, 3, 1, 5);

        assert_eq!(result.indices.len(), 3);
        assert_eq!(result.values.len(), 3);

        // Top 3 should be indices 1, 4, 2 with values 5.0, 4.0, 3.0
        assert_eq!(result.indices[0], 1); // 5.0
        assert_eq!(result.indices[1], 4); // 4.0
        assert_eq!(result.indices[2], 2); // 3.0
        assert!((result.values[0] - 5.0).abs() < 1e-6);
        assert!((result.values[1] - 4.0).abs() < 1e-6);
        assert!((result.values[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_topk_batch() {
        // 2 batches, 4 vocab
        let logits: Vec<f32> = vec![
            1.0, 4.0, 2.0, 3.0, // batch 0: max at 1
            5.0, 1.0, 3.0, 2.0, // batch 1: max at 0
        ];
        let result = topk(&logits, 2, 2, 4);

        assert_eq!(result.indices.len(), 4); // 2 batches * 2 k

        // Batch 0: top-2 are indices 1, 3 (values 4.0, 3.0)
        assert_eq!(result.indices[0], 1);
        assert_eq!(result.indices[1], 3);

        // Batch 1: top-2 are indices 0, 2 (values 5.0, 3.0)
        assert_eq!(result.indices[2], 0);
        assert_eq!(result.indices[3], 2);
    }

    #[test]
    fn test_softmax_1d() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut probs = vec![0.0; 3];
        softmax_1d(&logits, &mut probs);

        // Verify sum is 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Verify ordering: p[2] > p[1] > p[0]
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_apply_top_p() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];

        // top_p = 0.8 should keep first 2 (0.5 + 0.3 = 0.8)
        assert_eq!(apply_top_p(&probs, 0.8), 2);

        // top_p = 0.9 should keep first 3 (0.5 + 0.3 + 0.15 = 0.95)
        assert_eq!(apply_top_p(&probs, 0.9), 3);

        // top_p = 0.5 should keep first 1
        assert_eq!(apply_top_p(&probs, 0.5), 1);
    }

    #[test]
    fn test_greedy_sampling() {
        let logits: Vec<f32> = vec![
            1.0, 5.0, 3.0, 2.0, // batch 0: argmax = 1
            4.0, 1.0, 6.0, 2.0, // batch 1: argmax = 2
        ];
        let config = SamplingConfig::greedy();
        let result = sample_tokens(&logits, 2, 4, &config);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn test_argmax() {
        let logits: Vec<f32> = vec![
            1.0, 5.0, 3.0, 2.0,
            4.0, 1.0, 6.0, 2.0,
        ];
        let result = argmax(&logits, 2, 4);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn test_sampling_with_seed() {
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 5,
            seed: Some(42),
        };

        // With same seed, should get same result
        let result1 = sample_tokens(&logits, 1, 5, &config);
        let config2 = SamplingConfig {
            seed: Some(42),
            ..config.clone()
        };
        let result2 = sample_tokens(&logits, 1, 5, &config2);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_temperature_effect() {
        // High temperature should make distribution more uniform
        // Low temperature should make it more peaked
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0];

        let mut high_temp = logits.clone();
        apply_temperature(&mut high_temp, 2.0);

        let mut low_temp = logits.clone();
        apply_temperature(&mut low_temp, 0.5);

        // With high temp, differences should be smaller
        let high_diff = high_temp[2] - high_temp[0];
        let low_diff = low_temp[2] - low_temp[0];

        assert!(high_diff < low_diff);
    }
}
