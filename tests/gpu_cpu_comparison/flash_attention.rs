//! Flash Attention GPU vs CPU comparison tests (REQ-VERIFY-001).
//!
//! These tests verify that GPU Flash Attention implementations produce
//! results consistent with the CPU reference implementation within
//! defined precision tolerances.
//!
//! Precision tolerances (from SPEC/06-TESTING-STRATEGY.md):
//! - FP32: rtol=1e-5, atol=1e-6
//! - FP16: rtol=1e-3, atol=1e-4
//! - BF16: rtol=1e-2, atol=1e-3

use gllm_kernels::{BackendType, FlashAttentionConfig, KernelDispatcher};

/// Helper to generate deterministic random-like test data.
fn generate_test_data(size: usize, seed: u64) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    let mut state = seed;
    for _ in 0..size {
        // Simple LCG for reproducible "random" values
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0; // [-1, 1]
        data.push(val);
    }
    data
}

/// Check if two f32 slices are approximately equal within tolerances.
fn assert_close_f32(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        context,
        actual.len(),
        expected.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tolerance = atol + rtol * e.abs();
        assert!(
            diff <= tolerance,
            "{}: mismatch at index {}: actual={}, expected={}, diff={}, tolerance={}",
            context,
            i,
            a,
            e,
            diff,
            tolerance
        );
    }
}

/// Check that no NaN or Inf values exist in the output.
fn assert_no_nan_inf(data: &[f32], context: &str) {
    for (i, &v) in data.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{}: NaN/Inf at index {}: value={}",
            context,
            i,
            v
        );
    }
}

/// Test Flash Attention CPU reference implementation produces valid output.
#[test]
fn test_flash_attention_cpu_basic() {
    let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

    // Small test case: batch=1, heads=2, seq_len=4, head_dim=8
    let batch = 1;
    let heads = 2;
    let seq_len = 4;
    let head_dim = 8;
    let total_size = batch * heads * seq_len * head_dim;

    let q = generate_test_data(total_size, 42);
    let k = generate_test_data(total_size, 43);
    let v = generate_test_data(total_size, 44);
    let mut output = vec![0.0f32; total_size];

    let config = FlashAttentionConfig {
        batch_size: batch,
        num_heads: heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: false,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
        ..Default::default()
    };

    dispatcher.flash_attention(&q, &k, &v, &mut output, config);

    // Verify no NaN/Inf
    assert_no_nan_inf(&output, "CPU basic output");

    // Verify output is non-trivial (not all zeros)
    let sum: f32 = output.iter().sum();
    assert!(
        sum.abs() > 1e-10,
        "Output appears to be all zeros, sum={}",
        sum
    );
}

/// Test Flash Attention CPU vs CPU consistency (sanity check).
#[test]
fn test_flash_attention_cpu_consistency() {
    let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

    let batch = 2;
    let heads = 4;
    let seq_len = 16;
    let head_dim = 32;
    let total_size = batch * heads * seq_len * head_dim;

    let q = generate_test_data(total_size, 100);
    let k = generate_test_data(total_size, 101);
    let v = generate_test_data(total_size, 102);

    let config = FlashAttentionConfig {
        batch_size: batch,
        num_heads: heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
        ..Default::default()
    };

    // Run twice, should get identical results
    let mut output1 = vec![0.0f32; total_size];
    let mut output2 = vec![0.0f32; total_size];

    dispatcher.flash_attention(&q, &k, &v, &mut output1, config.clone());
    dispatcher.flash_attention(&q, &k, &v, &mut output2, config);

    // Exact equality for deterministic CPU implementation
    assert_eq!(output1, output2, "CPU implementation should be deterministic");
}

/// Test Flash Attention with SPEC-defined configuration.
/// Configuration: batch=2, heads=8, seq_len=1024, head_dim=64
#[test]
fn test_flash_attention_spec_config() {
    let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

    // SPEC/06-TESTING-STRATEGY.md configuration
    let batch = 2;
    let heads = 8;
    let seq_len = 1024;
    let head_dim = 64;
    let total_size = batch * heads * seq_len * head_dim;

    let q = generate_test_data(total_size, 200);
    let k = generate_test_data(total_size, 201);
    let v = generate_test_data(total_size, 202);
    let mut output = vec![0.0f32; total_size];

    let config = FlashAttentionConfig {
        batch_size: batch,
        num_heads: heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
        ..Default::default()
    };

    dispatcher.flash_attention(&q, &k, &v, &mut output, config);

    assert_no_nan_inf(&output, "SPEC config output");
}

/// CUDA GPU vs CPU comparison test.
/// Only runs when CUDA is available.
#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_cuda_vs_cpu_fp32() {
    use gllm_kernels::detect_backend;

    // Skip if CUDA not available
    let detected = detect_backend();
    if detected.backend != BackendType::Cuda {
        eprintln!("CUDA not available, skipping GPU comparison test");
        return;
    }

    let cpu_dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);
    let cuda_dispatcher = KernelDispatcher::with_backend(BackendType::Cuda);

    // SPEC configuration
    let batch = 2;
    let heads = 8;
    let seq_len = 1024;
    let head_dim = 64;
    let total_size = batch * heads * seq_len * head_dim;

    let q = generate_test_data(total_size, 300);
    let k = generate_test_data(total_size, 301);
    let v = generate_test_data(total_size, 302);

    let config = FlashAttentionConfig {
        batch_size: batch,
        num_heads: heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
        ..Default::default()
    };

    // CPU reference (Ground Truth)
    let mut cpu_output = vec![0.0f32; total_size];
    cpu_dispatcher.flash_attention(&q, &k, &v, &mut cpu_output, config.clone());

    // CUDA GPU
    let mut cuda_output = vec![0.0f32; total_size];
    cuda_dispatcher.flash_attention(&q, &k, &v, &mut cuda_output, config);

    // FP32 precision: rtol=1e-5, atol=1e-6
    assert_close_f32(
        &cuda_output,
        &cpu_output,
        1e-5,
        1e-6,
        "CUDA vs CPU FP32",
    );
}

/// Test Flash Attention with causal masking.
#[test]
fn test_flash_attention_causal_mask() {
    let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

    let batch = 1;
    let heads = 2;
    let seq_len = 8;
    let head_dim = 16;
    let total_size = batch * heads * seq_len * head_dim;

    let q = generate_test_data(total_size, 400);
    let k = generate_test_data(total_size, 401);
    let v = generate_test_data(total_size, 402);

    // With causal mask
    let config_causal = FlashAttentionConfig {
        batch_size: batch,
        num_heads: heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
        ..Default::default()
    };

    // Without causal mask
    let config_no_causal = FlashAttentionConfig {
        causal: false,
        ..config_causal.clone()
    };

    let mut output_causal = vec![0.0f32; total_size];
    let mut output_no_causal = vec![0.0f32; total_size];

    dispatcher.flash_attention(&q, &k, &v, &mut output_causal, config_causal);
    dispatcher.flash_attention(&q, &k, &v, &mut output_no_causal, config_no_causal);

    // Results should be different
    assert_ne!(
        output_causal, output_no_causal,
        "Causal and non-causal outputs should differ"
    );

    // Both should be valid
    assert_no_nan_inf(&output_causal, "Causal output");
    assert_no_nan_inf(&output_no_causal, "Non-causal output");
}

/// Test Flash Attention numerical stability options.
#[test]
fn test_flash_attention_stability_options() {
    let dispatcher = KernelDispatcher::with_backend(BackendType::Cpu);

    let batch = 1;
    let heads = 4;
    let seq_len = 256;
    let head_dim = 64;
    let total_size = batch * heads * seq_len * head_dim;

    // Use larger magnitude values to test stability
    let mut q = generate_test_data(total_size, 500);
    let mut k = generate_test_data(total_size, 501);
    let v = generate_test_data(total_size, 502);

    // Scale up to stress numerical stability
    for x in q.iter_mut() {
        *x *= 10.0;
    }
    for x in k.iter_mut() {
        *x *= 10.0;
    }

    // With log-space softmax and Kahan accumulator
    let config_stable = FlashAttentionConfig {
        batch_size: batch,
        num_heads: heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
        ..Default::default()
    };

    // Without stability features
    let config_basic = FlashAttentionConfig {
        use_log_space_softmax: false,
        use_kahan_accumulator: false,
        ..config_stable.clone()
    };

    let mut output_stable = vec![0.0f32; total_size];
    let mut output_basic = vec![0.0f32; total_size];

    dispatcher.flash_attention(&q, &k, &v, &mut output_stable, config_stable);
    dispatcher.flash_attention(&q, &k, &v, &mut output_basic, config_basic);

    // Both should produce valid output (no NaN/Inf)
    assert_no_nan_inf(&output_stable, "Stable config output");
    assert_no_nan_inf(&output_basic, "Basic config output");
}
