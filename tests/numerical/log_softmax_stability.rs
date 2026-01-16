//! 2M Token Numerical Stability Tests (REQ-VERIFY-002).
//!
//! These tests verify that:
//! 1. LogSpaceSoftmax handles 2M+ tokens without NaN/Inf
//! 2. Results are properly normalized (exp sums to 1)
//! 3. KahanAccumulator provides better precision than naive accumulation

use gllm_kernels::ops::softmax::{log_add_exp, log_sum_exp, log_sum_exp_kahan, LogSpaceSoftmax};
use gllm_kernels::ops::stable_accumulator::{
    AccumulatorConfig, HierarchicalAccumulator, KahanAccumulator, StableAccumulator,
};

/// Generate deterministic test data.
fn generate_logits(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut data = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
        data.push(val * scale);
    }
    data
}

// =============================================================================
// Log-Space Softmax Tests
// =============================================================================

/// Test log_add_exp basic functionality.
#[test]
fn test_log_add_exp_basic() {
    // log(exp(0) + exp(0)) = log(2)
    let result = log_add_exp(0.0, 0.0);
    assert!(
        (result - 2.0_f64.ln()).abs() < 1e-10,
        "log_add_exp(0,0) should be ln(2), got {}",
        result
    );

    // log(exp(1) + exp(2)) = log(e + e^2)
    let result = log_add_exp(1.0, 2.0);
    let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
    assert!(
        (result - expected).abs() < 1e-10,
        "log_add_exp(1,2) = {}, expected {}",
        result,
        expected
    );
}

/// Test log_add_exp with extreme values.
#[test]
fn test_log_add_exp_extreme_values() {
    // Large positive values should not overflow
    let result = log_add_exp(1000.0, 1000.0);
    let expected = 1000.0 + 2.0_f64.ln();
    assert!(
        (result - expected).abs() < 1e-10,
        "log_add_exp with large positives failed: {} vs {}",
        result,
        expected
    );

    // Large negative values should not underflow
    let result = log_add_exp(-1000.0, -1000.0);
    let expected = -1000.0 + 2.0_f64.ln();
    assert!(
        (result - expected).abs() < 1e-10,
        "log_add_exp with large negatives failed: {} vs {}",
        result,
        expected
    );

    // Very different magnitudes
    let result = log_add_exp(1000.0, 0.0);
    assert!(
        (result - 1000.0).abs() < 1e-10,
        "log_add_exp with different magnitudes failed: {}",
        result
    );
}

/// Test log_add_exp with negative infinity.
#[test]
fn test_log_add_exp_neg_infinity() {
    assert_eq!(log_add_exp(f64::NEG_INFINITY, 5.0), 5.0);
    assert_eq!(log_add_exp(5.0, f64::NEG_INFINITY), 5.0);
    assert_eq!(
        log_add_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
}

/// Test log_sum_exp with a sequence.
#[test]
fn test_log_sum_exp_sequence() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = log_sum_exp(&values);
    let expected: f64 = values.iter().map(|x| x.exp()).sum::<f64>().ln();
    assert!(
        (result - expected).abs() < 1e-10,
        "log_sum_exp failed: {} vs {}",
        result,
        expected
    );
}

/// Test log_sum_exp with large sequence (10K elements).
#[test]
fn test_log_sum_exp_large_sequence() {
    let values = generate_logits(10_000, 42, 10.0);
    let result = log_sum_exp(&values);

    assert!(
        result.is_finite(),
        "log_sum_exp should produce finite result for large sequence"
    );
    assert!(
        result > 0.0,
        "log_sum_exp should be positive for non-trivial input"
    );
}

/// Test log_sum_exp vs log_sum_exp_kahan consistency.
#[test]
fn test_log_sum_exp_vs_kahan() {
    let values = generate_logits(100_000, 123, 5.0);

    let result_naive = log_sum_exp(&values);
    let result_kahan = log_sum_exp_kahan(&values);

    // Both should produce valid results
    assert!(result_naive.is_finite());
    assert!(result_kahan.is_finite());

    // Results should be very close
    let diff = (result_naive - result_kahan).abs();
    assert!(
        diff < 1e-8,
        "log_sum_exp and log_sum_exp_kahan differ by {}, naive={}, kahan={}",
        diff,
        result_naive,
        result_kahan
    );
}

// =============================================================================
// 2M Token Stability Tests (REQ-VERIFY-002)
// =============================================================================

/// Test 2M token LogSpaceSoftmax stability.
/// This is the core test for REQ-VERIFY-002.
#[test]
fn test_2m_context_log_softmax_stability() {
    let seq_len = 2_097_152; // 2M tokens
    let block_size = 4096; // Process in blocks

    let logits = generate_logits(seq_len, 42, 10.0);

    let mut acc = LogSpaceSoftmax::new();

    // Process in blocks (simulating attention computation)
    for block in logits.chunks(block_size) {
        let block_max = block.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let block_log_sum_exp = log_sum_exp(block);
        acc.update(block_max, block_log_sum_exp - block_max);
    }

    // Verify no NaN/Inf in accumulator state
    assert!(
        acc.max().is_finite(),
        "LogSpaceSoftmax max should be finite, got {}",
        acc.max()
    );
    assert!(
        acc.log_sum().is_finite(),
        "LogSpaceSoftmax log_sum should be finite, got {}",
        acc.log_sum()
    );
    assert!(
        acc.sum().is_finite(),
        "LogSpaceSoftmax sum should be finite, got {}",
        acc.sum()
    );

    // Verify count
    let expected_blocks = (seq_len + block_size - 1) / block_size;
    assert_eq!(
        acc.count(),
        expected_blocks,
        "Block count mismatch: {} vs {}",
        acc.count(),
        expected_blocks
    );
}

/// Test 2M token result normalization.
/// exp(log_softmax(x)) should sum to 1.
#[test]
fn test_2m_softmax_normalization() {
    // Use a smaller sequence for normalization verification
    // (full 2M would be slow to verify element by element)
    let seq_len = 100_000;
    let logits = generate_logits(seq_len, 100, 5.0);

    // Compute log-sum-exp
    let log_sum = log_sum_exp_kahan(&logits);
    assert!(log_sum.is_finite());

    // Verify normalization: sum(exp(logits - log_sum)) = 1
    let mut sum = KahanAccumulator::<f64>::new();
    for &logit in &logits {
        sum.add((logit - log_sum).exp());
    }

    let total = sum.corrected_value();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "Softmax should sum to 1, got {}",
        total
    );
}

// =============================================================================
// Kahan Accumulator Precision Tests
// =============================================================================

/// Test Kahan accumulator precision vs naive accumulation.
#[test]
fn test_kahan_vs_naive_accumulation() {
    let n = 1_000_000;
    let x = 1e-8_f64;
    let expected = (n as f64) * x;

    // Naive accumulation
    let mut naive_sum = 0.0_f64;
    for _ in 0..n {
        naive_sum += x;
    }

    // Kahan accumulation
    let mut kahan = KahanAccumulator::<f64>::new();
    for _ in 0..n {
        kahan.add(x);
    }

    let naive_error = (naive_sum - expected).abs() / expected;
    let kahan_error = (kahan.value() - expected).abs() / expected;

    // Kahan should be significantly more accurate
    assert!(
        kahan_error < naive_error / 10.0,
        "Kahan should be 10x more accurate: kahan_error={}, naive_error={}",
        kahan_error,
        naive_error
    );

    // Kahan should have very low error
    assert!(
        kahan_error < 1e-10,
        "Kahan error should be < 1e-10, got {}",
        kahan_error
    );
}

/// Test Kahan accumulator with alternating large/small values.
#[test]
fn test_kahan_large_small_values() {
    let mut kahan = KahanAccumulator::<f64>::with_value(1e10_f64);

    // Add many small values
    for _ in 0..10000 {
        kahan.add(1.0);
    }

    let result = kahan.value();
    let expected = 1e10 + 10000.0;
    let error = (result - expected).abs();

    assert!(
        error < 1e-3,
        "Kahan should handle large+small: error={}, result={}, expected={}",
        error,
        result,
        expected
    );
}

/// Test corrected_value vs value.
#[test]
fn test_kahan_corrected_value() {
    let n = 100_000;
    let x = 1e-9_f64;

    let mut kahan = KahanAccumulator::<f64>::new();
    for _ in 0..n {
        kahan.add(x);
    }

    let expected = (n as f64) * x;
    let error_value = (kahan.value() - expected).abs();
    let error_corrected = (kahan.corrected_value() - expected).abs();

    // corrected_value should be at least as accurate
    assert!(
        error_corrected <= error_value + 1e-15,
        "corrected_value should be at least as accurate: {} vs {}",
        error_corrected,
        error_value
    );
}

// =============================================================================
// Stable Accumulator Tests
// =============================================================================

/// Test StableAccumulator merge operation.
#[test]
fn test_stable_accumulator_merge() {
    let config = AccumulatorConfig::default();

    let mut acc1 = StableAccumulator::new(config.clone());
    acc1.update(5.0, 100.0);
    acc1.update(7.0, 200.0);

    let mut acc2 = StableAccumulator::new(config);
    acc2.update(3.0, 50.0);
    acc2.update(6.0, 150.0);

    let max_before = acc1.max();
    acc1.merge(&acc2);

    // Max should be updated correctly
    assert_eq!(
        acc1.max(),
        max_before.max(acc2.max()),
        "Max should be max of both accumulators"
    );

    // Count should be sum
    assert_eq!(acc1.count(), 4, "Count should be 4 after merge");
}

/// Test HierarchicalAccumulator for very long sequences.
#[test]
fn test_hierarchical_accumulator_long_sequence() {
    let config = AccumulatorConfig {
        block_size: 64,
        use_fp64: true,
        use_kahan: true,
        ..Default::default()
    };

    let n = 100_000;
    let mut hier = HierarchicalAccumulator::new(config.clone(), n);

    for i in 0..n {
        let mut acc = StableAccumulator::new(config.clone());
        acc.update((i % 1000) as f64, 1.0);
        hier.add(acc);
    }

    let result = hier.finalize();

    assert_eq!(result.count(), n, "Count should match number of additions");
    assert!(result.max().is_finite(), "Max should be finite");
    assert!(result.sum().is_finite(), "Sum should be finite");
}

// =============================================================================
// LogSpaceSoftmax Merge Tests
// =============================================================================

/// Test LogSpaceSoftmax merge operation.
#[test]
fn test_log_space_softmax_merge() {
    let mut acc1 = LogSpaceSoftmax::new();
    acc1.update_raw(5.0, 10.0);
    acc1.update_raw(7.0, 20.0);

    let mut acc2 = LogSpaceSoftmax::new();
    acc2.update_raw(3.0, 5.0);
    acc2.update_raw(6.0, 15.0);

    acc1.merge(&acc2);

    assert_eq!(acc1.count(), 4);
    assert!(acc1.max().is_finite());
    assert!(acc1.sum().is_finite());
    assert!(acc1.sum() > 0.0);
}

/// Test LogSpaceSoftmax consistency with StableAccumulator.
#[test]
fn test_log_space_vs_stable_accumulator() {
    let blocks: Vec<(f64, f64)> = vec![
        (5.0, 10.0),
        (3.0, 5.0),
        (7.0, 20.0),
        (1.0, 2.0),
        (8.0, 30.0),
    ];

    // StableAccumulator
    let mut std_acc = StableAccumulator::default_config();
    for (max, sum_exp) in blocks.iter() {
        std_acc.update(*max, *sum_exp);
    }

    // LogSpaceSoftmax
    let mut log_acc = LogSpaceSoftmax::new();
    for (max, sum_exp) in blocks.iter() {
        log_acc.update_raw(*max, *sum_exp);
    }

    // Should have same max
    assert_eq!(std_acc.max(), log_acc.max());

    // Sums should be very close
    let diff = (std_acc.sum() - log_acc.sum()).abs() / std_acc.sum();
    assert!(
        diff < 1e-10,
        "Relative difference: {}, std={}, log={}",
        diff,
        std_acc.sum(),
        log_acc.sum()
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test empty input handling.
#[test]
fn test_empty_input() {
    let empty: Vec<f64> = vec![];
    let result = log_sum_exp(&empty);
    assert_eq!(result, f64::NEG_INFINITY);
}

/// Test single element.
#[test]
fn test_single_element() {
    let single = vec![5.0];
    let result = log_sum_exp(&single);
    assert!((result - 5.0).abs() < 1e-10);
}

/// Test all same values.
#[test]
fn test_all_same_values() {
    let n = 1000;
    let val = 3.0;
    let values: Vec<f64> = vec![val; n];

    let result = log_sum_exp(&values);
    let expected = val + (n as f64).ln();

    assert!(
        (result - expected).abs() < 1e-10,
        "All same values: {} vs {}",
        result,
        expected
    );
}

/// Test very small values (underflow potential).
#[test]
fn test_very_small_values() {
    let values = vec![-1000.0, -1000.1, -1000.2, -999.9];
    let result = log_sum_exp(&values);

    assert!(result.is_finite());
    // log_sum_exp >= max, so result should be >= -999.9
    // With 4 values, result = max + log(sum(exp(x-max))) ≈ -999.9 + log(3.47) ≈ -998.66
    assert!(
        result >= -999.9 && result < -998.0,
        "Result {} should be in range [-999.9, -998.0)",
        result
    );
}

/// Test very large values (overflow potential).
#[test]
fn test_very_large_values() {
    let values = vec![1000.0, 1000.1, 1000.2, 999.9];
    let result = log_sum_exp(&values);

    assert!(result.is_finite());
    assert!(result > 1000.0); // Should be slightly larger than max
}
