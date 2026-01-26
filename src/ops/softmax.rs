//! Log-space softmax computation for numerical stability.
//!
//! This module provides pure-Rust implementations for log-space softmax operations,
//! optimized for numerical stability with 2M+ context lengths.
//!
//! For GPU-accelerated softmax, use the selected Backend implementation.

use crate::kernel_types::{KernelFloat, SoftmaxConfig};
use crate::ops::stable_accumulator::{AccumulatorConfig, StableAccumulator};

/// Compute log(exp(a) + exp(b)) in a numerically stable way.
#[inline]
pub fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }

    let max = a.max(b);
    let min = a.min(b);

    max + (1.0 + (min - max).exp()).ln()
}

/// Compute log(sum(exp(x))) in a numerically stable way.
#[inline]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max.is_infinite() {
        return max;
    }

    let sum: f64 = values.iter().map(|&x| (x - max).exp()).sum();

    max + sum.ln()
}

/// Compute log(sum(exp(x))) using Kahan summation for the exp sum.
pub fn log_sum_exp_kahan(values: &[f64]) -> f64 {
    use super::stable_accumulator::KahanAccumulator;

    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max.is_infinite() {
        return max;
    }

    let mut sum = KahanAccumulator::<f64>::new();
    for &x in values {
        sum.add((x - max).exp());
    }

    max + sum.value().ln()
}

/// CPU reference implementation of Softmax (generic).
#[inline(always)]
pub fn softmax<T: KernelFloat>(input: &[T], output: &mut [T], config: SoftmaxConfig) {
    if input.is_empty() {
        return;
    }

    let acc_config = if config.use_log_space || config.use_kahan {
        AccumulatorConfig::max_precision()
    } else {
        AccumulatorConfig::short_context()
    };

    let mut max_val = f64::NEG_INFINITY;
    for &x in input {
        max_val = max_val.max(x.to_f32() as f64);
    }

    if config.use_log_space || config.use_kahan {
        let mut stable = StableAccumulator::new(acc_config);
        let sum_exp: f64 = input.iter().map(|&x| ((x.to_f32() as f64) - max_val).exp()).sum();
        stable.update(max_val, sum_exp);

        let m = stable.max();
        let l = stable.sum();

        for (i, &x) in input.iter().enumerate() {
            let prob = if l > 0.0 { (((x.to_f32() as f64) - m).exp() / l) as f32 } else { 0.0 };
            output[i] = T::from_f32(prob);
        }
    } else {
        let max_f32 = input.iter().map(|x| x.to_f32()).fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = input.iter().map(|x| (x.to_f32() - max_f32).exp()).sum();
        for (i, &x) in input.iter().enumerate() {
            output[i] = T::from_f32((x.to_f32() - max_f32).exp() / sum);
        }
    }
}

/// Log-space softmax accumulator for online computation.
#[derive(Debug, Clone)]
pub struct LogSpaceSoftmax {
    /// Current maximum score.
    m: f64,
    /// Log of the running sum: log(sum(exp(scores - m))).
    log_l: f64,
    /// Number of blocks processed.
    count: usize,
}

impl Default for LogSpaceSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSpaceSoftmax {
    /// Create a new log-space softmax accumulator.
    pub fn new() -> Self {
        Self {
            m: f64::NEG_INFINITY,
            log_l: f64::NEG_INFINITY,
            count: 0,
        }
    }

    /// Update with a new block of scores.
    pub fn update(&mut self, block_max: f64, block_log_sum_exp: f64) -> f64 {
        let m_new = self.m.max(block_max);

        let prev_contrib = (self.m - m_new) + self.log_l;
        let new_contrib = (block_max - m_new) + block_log_sum_exp;
        let log_l_new = log_add_exp(prev_contrib, new_contrib);

        let log_scale = self.m - m_new;

        self.m = m_new;
        self.log_l = log_l_new;
        self.count += 1;

        log_scale
    }

    /// Update with raw block statistics (not in log-space).
    pub fn update_raw(&mut self, block_max: f64, block_sum_exp: f64) -> f64 {
        let block_log_sum_exp = block_sum_exp.ln();
        self.update(block_max, block_log_sum_exp)
    }

    /// Get the current maximum.
    #[inline]
    pub fn max(&self) -> f64 {
        self.m
    }

    /// Get the log of the running sum.
    #[inline]
    pub fn log_sum(&self) -> f64 {
        self.log_l
    }

    /// Get the running sum (converted from log-space).
    #[inline]
    pub fn sum(&self) -> f64 {
        self.log_l.exp()
    }

    /// Get the number of blocks processed.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Compute the final normalization factor: 1 / sum(exp(scores - m)).
    #[inline]
    pub fn normalization_factor(&self) -> f64 {
        (-self.log_l).exp()
    }

    /// Merge another log-space accumulator into this one.
    pub fn merge(&mut self, other: &LogSpaceSoftmax) -> f64 {
        if other.count == 0 {
            return 0.0;
        }
        if self.count == 0 {
            self.m = other.m;
            self.log_l = other.log_l;
            self.count = other.count;
            return f64::NEG_INFINITY;
        }

        let m_new = self.m.max(other.m);

        let self_contrib = (self.m - m_new) + self.log_l;
        let other_contrib = (other.m - m_new) + other.log_l;
        let log_l_new = log_add_exp(self_contrib, other_contrib);

        let log_scale = self.m - m_new;

        self.m = m_new;
        self.log_l = log_l_new;
        self.count += other.count;

        log_scale
    }

    /// Reset the accumulator.
    pub fn reset(&mut self) {
        self.m = f64::NEG_INFINITY;
        self.log_l = f64::NEG_INFINITY;
        self.count = 0;
    }
}

// NOTE: TensorLogOps and OnlineSoftmax (Burn-based wrappers) have been removed.
// Use a Backend implementation for GPU-accelerated softmax computation.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_add_exp_basic() {
        let result = log_add_exp(0.0, 0.0);
        assert!((result - 2.0_f64.ln()).abs() < 1e-10);

        let result = log_add_exp(1.0, 2.0);
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_add_exp_extreme() {
        let result = log_add_exp(1000.0, 1000.0);
        assert!((result - (1000.0 + 2.0_f64.ln())).abs() < 1e-10);

        let result = log_add_exp(-1000.0, -1000.0);
        assert!((result - (-1000.0 + 2.0_f64.ln())).abs() < 1e-10);

        let result = log_add_exp(1000.0, 0.0);
        assert!((result - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_add_exp_neg_infinity() {
        assert_eq!(log_add_exp(f64::NEG_INFINITY, 5.0), 5.0);
        assert_eq!(log_add_exp(5.0, f64::NEG_INFINITY), 5.0);
        assert_eq!(
            log_add_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_log_sum_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_large_sequence() {
        let values: Vec<f64> = (0..10000).map(|i| (i as f64) * 0.001).collect();
        let result = log_sum_exp(&values);

        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn test_log_space_softmax_update() {
        let mut acc = LogSpaceSoftmax::new();

        acc.update_raw(5.0, 10.0);
        assert_eq!(acc.max(), 5.0);
        assert!((acc.sum() - 10.0).abs() < 1e-10);

        acc.update_raw(3.0, 5.0);
        assert_eq!(acc.max(), 5.0);
        let expected_sum = 10.0 + 5.0 * (-2.0_f64).exp();
        assert!((acc.sum() - expected_sum).abs() < 1e-8);
    }

    #[test]
    fn test_log_space_softmax_merge() {
        let mut acc1 = LogSpaceSoftmax::new();
        acc1.update_raw(5.0, 10.0);

        let mut acc2 = LogSpaceSoftmax::new();
        acc2.update_raw(3.0, 5.0);

        acc1.merge(&acc2);

        assert_eq!(acc1.max(), 5.0);
        let expected_sum = 10.0 + 5.0 * (-2.0_f64).exp();
        assert!((acc1.sum() - expected_sum).abs() < 1e-8);
    }

    #[test]
    fn test_log_space_vs_standard() {
        use super::super::stable_accumulator::StableAccumulator;

        let blocks: Vec<(f64, f64)> = vec![(5.0, 10.0), (3.0, 5.0), (7.0, 20.0), (1.0, 2.0)];

        let mut std_acc = StableAccumulator::default_config();
        for (max, sum_exp) in blocks.iter() {
            std_acc.update(*max, *sum_exp);
        }

        let mut log_acc = LogSpaceSoftmax::new();
        for (max, sum_exp) in blocks.iter() {
            log_acc.update_raw(*max, *sum_exp);
        }

        assert_eq!(std_acc.max(), log_acc.max());
        let diff = (std_acc.sum() - log_acc.sum()).abs() / std_acc.sum();
        assert!(diff < 1e-10, "Relative difference: {}", diff);
    }
}
