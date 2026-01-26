//! Numerically stable accumulation for ultra-long sequences.
//!
//! This module provides Kahan summation and hierarchical accumulation to maintain
//! precision when accumulating millions of floating-point values.
//!
//! All implementations are pure Rust with no external dependencies.
//! For GPU-accelerated versions, use Backend implementations.

/// Kahan compensated summation for high-precision accumulation.
///
/// The Kahan summation algorithm tracks a compensation term `c` that captures
/// the low-order bits lost during each addition, then applies them in the next
/// iteration. This reduces rounding error from O(n) to O(1).
#[derive(Debug, Clone, Copy)]
pub struct KahanAccumulator<T> {
    sum: T,
    /// Compensation term for lost low-order bits.
    c: T,
}

impl<T: Default> Default for KahanAccumulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default> KahanAccumulator<T> {
    /// Create a new Kahan accumulator initialized to zero.
    pub fn new() -> Self {
        Self {
            sum: T::default(),
            c: T::default(),
        }
    }
}

impl KahanAccumulator<f64> {
    /// Create a Kahan accumulator with an initial value.
    pub fn with_value(value: f64) -> Self {
        Self { sum: value, c: 0.0 }
    }

    /// Add a value using Kahan summation.
    #[inline]
    pub fn add(&mut self, x: f64) {
        let y = x - self.c; // Compensate for previous error
        let t = self.sum + y; // New sum (some low bits lost)
        self.c = (t - self.sum) - y; // Recover lost low bits
        self.sum = t;
    }

    /// Add another Kahan sum (preserving precision).
    pub fn add_kahan(&mut self, other: &KahanAccumulator<f64>) {
        self.add(other.sum);
        self.add(-other.c);
    }

    /// Get the accumulated value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.sum
    }

    /// Get sum with compensation applied (slightly more accurate).
    #[inline]
    pub fn corrected_value(&self) -> f64 {
        self.sum - self.c
    }

    /// Reset the accumulator to zero.
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.c = 0.0;
    }
}

impl KahanAccumulator<f32> {
    /// Create a Kahan accumulator with an initial value.
    pub fn with_value(value: f32) -> Self {
        Self { sum: value, c: 0.0 }
    }

    /// Add a value using Kahan summation.
    #[inline]
    pub fn add(&mut self, x: f32) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    /// Get the accumulated value.
    #[inline]
    pub fn value(&self) -> f32 {
        self.sum
    }
}

/// Backward-compatible alias.
pub type KahanSum<T> = KahanAccumulator<T>;

/// Configuration for stable accumulation.
#[derive(Debug, Clone)]
pub struct AccumulatorConfig {
    /// Number of elements per level-0 block (default: 64).
    pub block_size: usize,
    /// Use FP64 for accumulation even if inputs are FP32.
    pub use_fp64: bool,
    /// Enable Kahan summation (recommended for 2M+ contexts).
    pub use_kahan: bool,
    /// Renormalization interval (blocks between explicit renormalization).
    pub renorm_interval: usize,
}

impl Default for AccumulatorConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            use_fp64: true,
            use_kahan: true,
            renorm_interval: 1024,
        }
    }
}

impl AccumulatorConfig {
    /// Configuration for maximum precision (recommended for 2M contexts).
    pub fn max_precision() -> Self {
        Self {
            block_size: 64,
            use_fp64: true,
            use_kahan: true,
            renorm_interval: 512,
        }
    }

    /// Configuration for balanced precision/performance.
    pub fn balanced() -> Self {
        Self {
            block_size: 128,
            use_fp64: true,
            use_kahan: true,
            renorm_interval: 1024,
        }
    }

    /// Configuration for shorter contexts (< 100K tokens).
    pub fn short_context() -> Self {
        Self {
            block_size: 256,
            use_fp64: true,
            use_kahan: false,
            renorm_interval: 2048,
        }
    }
}

/// Stable accumulator for online softmax in attention computation.
///
/// This accumulator tracks three values needed for online softmax:
/// - `m`: Running maximum (for numerical stability)
/// - `l`: Running sum of exp(x - m) (denominator)
#[derive(Debug, Clone)]
pub struct StableAccumulator {
    /// Running maximum value.
    m: f64,
    /// Running sum of exp(scores - m) using Kahan summation.
    l: KahanAccumulator<f64>,
    /// Count of accumulated elements.
    count: usize,
    /// Configuration.
    config: AccumulatorConfig,
}

impl StableAccumulator {
    /// Create a new stable accumulator.
    pub fn new(config: AccumulatorConfig) -> Self {
        Self {
            m: f64::NEG_INFINITY,
            l: KahanAccumulator::new(),
            count: 0,
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(AccumulatorConfig::default())
    }

    /// Update the accumulator with a new block of scores.
    ///
    /// # Arguments
    /// * `block_max` - Maximum value in the current block
    /// * `block_sum_exp` - Sum of exp(scores - block_max) in the block
    ///
    /// # Returns
    /// The scale factor to apply to previous output accumulation.
    pub fn update(&mut self, block_max: f64, block_sum_exp: f64) -> f64 {
        let m_new = self.m.max(block_max);

        let scale = if self.m.is_finite() {
            (self.m - m_new).exp()
        } else {
            0.0
        };

        let adjusted_sum = (block_max - m_new).exp() * block_sum_exp;

        if self.config.use_kahan {
            self.l = KahanAccumulator::<f64>::with_value(scale * self.l.value());
            self.l.add(adjusted_sum);
        } else {
            self.l = KahanAccumulator::<f64>::with_value(scale * self.l.value() + adjusted_sum);
        }

        self.m = m_new;
        self.count += 1;

        scale
    }

    /// Get current maximum value.
    #[inline]
    pub fn max(&self) -> f64 {
        self.m
    }

    /// Get current sum (denominator for softmax).
    #[inline]
    pub fn sum(&self) -> f64 {
        self.l.corrected_value()
    }

    /// Get the number of blocks accumulated.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if renormalization is recommended.
    #[inline]
    pub fn should_renormalize(&self) -> bool {
        self.count > 0 && self.count % self.config.renorm_interval == 0
    }

    /// Reset the accumulator for reuse.
    pub fn reset(&mut self) {
        self.m = f64::NEG_INFINITY;
        self.l.reset();
        self.count = 0;
    }

    /// Merge another accumulator into this one.
    pub fn merge(&mut self, other: &StableAccumulator) -> f64 {
        if other.count == 0 {
            return 1.0;
        }
        if self.count == 0 {
            self.m = other.m;
            self.l = other.l;
            self.count = other.count;
            return 0.0;
        }

        let m_new = self.m.max(other.m);

        let self_scale = (self.m - m_new).exp();
        let other_scale = (other.m - m_new).exp();

        let self_contrib = self_scale * self.l.value();
        let other_contrib = other_scale * other.l.value();

        self.l = KahanAccumulator::<f64>::with_value(self_contrib);
        self.l.add(other_contrib);
        self.m = m_new;
        self.count += other.count;

        self_scale
    }
}

/// Hierarchical accumulator for ultra-long sequences.
#[derive(Debug)]
pub struct HierarchicalAccumulator {
    /// Accumulators at each level.
    levels: Vec<Vec<StableAccumulator>>,
    /// Items per level before promotion.
    items_per_level: usize,
    /// Current counts at each level.
    counts: Vec<usize>,
    /// Configuration.
    config: AccumulatorConfig,
}

impl HierarchicalAccumulator {
    /// Create a new hierarchical accumulator.
    pub fn new(config: AccumulatorConfig, max_items: usize) -> Self {
        let items_per_level = config.block_size;

        let num_levels = if max_items <= items_per_level {
            1
        } else {
            (max_items as f64).log(items_per_level as f64).ceil() as usize + 1
        };

        let levels = (0..num_levels)
            .map(|level| {
                let capacity = (max_items / items_per_level.pow(level as u32)).max(1);
                Vec::with_capacity(capacity)
            })
            .collect();

        let counts = vec![0; num_levels];

        Self {
            levels,
            items_per_level,
            counts,
            config,
        }
    }

    /// Add a new accumulator at level 0.
    pub fn add(&mut self, acc: StableAccumulator) {
        self.add_at_level(0, acc);
    }

    fn add_at_level(&mut self, level: usize, acc: StableAccumulator) {
        if level >= self.levels.len() {
            self.levels.push(Vec::new());
            self.counts.push(0);
        }

        self.levels[level].push(acc);
        self.counts[level] += 1;

        if self.counts[level] >= self.items_per_level {
            self.promote_level(level);
        }
    }

    fn promote_level(&mut self, level: usize) {
        if self.levels[level].is_empty() {
            return;
        }

        let mut merged = StableAccumulator::new(self.config.clone());
        for acc in self.levels[level].drain(..) {
            merged.merge(&acc);
        }

        self.counts[level] = 0;
        self.add_at_level(level + 1, merged);
    }

    /// Finalize and get the merged result.
    pub fn finalize(mut self) -> StableAccumulator {
        for level in 0..self.levels.len() {
            if !self.levels[level].is_empty() {
                self.promote_level(level);
            }
        }

        for level in (0..self.levels.len()).rev() {
            if !self.levels[level].is_empty() {
                return self.levels[level].pop().unwrap();
            }
        }

        StableAccumulator::new(self.config)
    }

    /// Get the total number of items accumulated.
    pub fn total_count(&self) -> usize {
        let mut total = 0;
        for (level, count) in self.counts.iter().enumerate() {
            total += count * self.items_per_level.pow(level as u32);
        }
        total
    }
}

/// Output accumulator for attention computation.
#[derive(Debug, Clone)]
pub struct StableRowState<T> {
    /// Softmax accumulator (for m and l).
    pub softmax: StableAccumulator,
    /// Accumulated output (o = sum of p_ij * v_j).
    pub output: T,
}

impl<T> StableRowState<T> {
    pub fn with_output(output: T, config: AccumulatorConfig) -> Self {
        Self {
            softmax: StableAccumulator::new(config),
            output,
        }
    }
}

impl<T: Default> StableRowState<T> {
    pub fn new(config: AccumulatorConfig) -> Self {
        Self {
            softmax: StableAccumulator::new(config),
            output: T::default(),
        }
    }
}

// NOTE: Burn-based `StableRowState<Tensor<B, D>>` wrapper has been removed.
// Use Backend implementations for GPU-accelerated attention computation.

/// Backward-compatible alias.
pub type OutputAccumulator<T> = StableRowState<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum_precision() {
        let n = 1_000_000;
        let x = 1e-8_f64;
        let expected = (n as f64) * x;

        let mut naive_sum = 0.0_f64;
        for _ in 0..n {
            naive_sum += x;
        }

        let mut kahan = KahanAccumulator::<f64>::new();
        for _ in 0..n {
            kahan.add(x);
        }

        let naive_error = (naive_sum - expected).abs() / expected;
        let kahan_error = (kahan.value() - expected).abs() / expected;

        assert!(kahan_error < naive_error / 10.0);
        assert!(kahan_error < 1e-10);
    }

    #[test]
    fn test_kahan_sum_large_small() {
        let mut kahan = KahanAccumulator::<f64>::with_value(1e10_f64);

        for _ in 0..1000 {
            kahan.add(1.0);
        }

        let result = kahan.value();
        let expected = 1e10 + 1000.0;
        let error = (result - expected).abs();

        assert!(error < 1e-5, "Error too large: {}", error);
    }

    #[test]
    fn test_stable_accumulator_update() {
        let mut acc = StableAccumulator::default_config();

        let scale1 = acc.update(5.0, 10.0);
        assert_eq!(scale1, 0.0);
        assert_eq!(acc.max(), 5.0);

        let scale2 = acc.update(10.0, 20.0);
        assert!((scale2 - (-5.0_f64).exp()).abs() < 1e-10);
        assert_eq!(acc.max(), 10.0);
    }

    #[test]
    fn test_stable_accumulator_merge() {
        let config = AccumulatorConfig::default();

        let mut acc1 = StableAccumulator::new(config.clone());
        acc1.update(5.0, 100.0);

        let mut acc2 = StableAccumulator::new(config);
        acc2.update(3.0, 50.0);

        acc1.merge(&acc2);

        assert_eq!(acc1.max(), 5.0);
        assert_eq!(acc1.count(), 2);
    }

    #[test]
    fn test_hierarchical_accumulator() {
        let config = AccumulatorConfig {
            block_size: 4,
            ..Default::default()
        };

        let mut hier = HierarchicalAccumulator::new(config.clone(), 100);

        for i in 0..20 {
            let mut acc = StableAccumulator::new(config.clone());
            acc.update(i as f64, 1.0);
            hier.add(acc);
        }

        let result = hier.finalize();
        assert_eq!(result.count(), 20);
        assert_eq!(result.max(), 19.0);
    }

    #[test]
    fn test_hierarchical_accumulator_large() {
        let config = AccumulatorConfig::default();
        let n = 10000;

        let mut hier = HierarchicalAccumulator::new(config.clone(), n);

        for i in 0..n {
            let mut acc = StableAccumulator::new(config.clone());
            acc.update((i % 100) as f64, 1.0);
            hier.add(acc);
        }

        let result = hier.finalize();
        assert_eq!(result.count(), n);
    }
}
