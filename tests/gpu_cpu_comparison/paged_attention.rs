//! Paged Attention GPU vs CPU comparison tests (REQ-VERIFY-001).
//!
//! These tests verify that GPU Paged Attention implementations produce
//! results consistent with the CPU reference implementation within
//! defined precision tolerances.
//!
//! Precision tolerances (from SPEC/06-TESTING-STRATEGY.md):
//! - FP32: rtol=1e-5, atol=1e-6
//! - FP16: rtol=1e-3, atol=1e-4
//! - BF16: rtol=1e-2, atol=1e-3

use gllm_kernels::backend::{Backend, TensorSlice, TensorSliceMut};
use gllm_kernels::{BackendType, CudaBackend, CpuBackend, PagedAttentionConfig};

fn run_paged_attention<B: Backend>(
    backend: &B,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    page_table: &[u32],
    seq_lens: &[u32],
    output: &mut [f32],
    config: PagedAttentionConfig,
) {
    backend
        .paged_attention(
            TensorSlice::F32(q),
            TensorSlice::F32(k_cache),
            TensorSlice::F32(v_cache),
            page_table,
            seq_lens,
            TensorSliceMut::F32(output),
            config,
        )
        .expect("paged_attention failed");
}

/// Helper to generate deterministic random-like test data.
fn generate_test_data(size: usize, seed: u64) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    let mut state = seed;
    for _ in 0..size {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        data.push(val);
    }
    data
}

/// Generate a simple page table for testing.
fn generate_page_table(num_seqs: usize, max_blocks_per_seq: usize) -> Vec<u32> {
    let mut table = Vec::with_capacity(num_seqs * max_blocks_per_seq);
    let mut page_idx = 0u32;
    for _ in 0..num_seqs {
        for _ in 0..max_blocks_per_seq {
            table.push(page_idx);
            page_idx += 1;
        }
    }
    table
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

/// Test Paged Attention CPU reference implementation produces valid output.
#[test]
fn test_paged_attention_cpu_basic() {
    let dispatcher = CpuBackend::new();

    // Small test case
    let num_seqs = 2;
    let num_heads = 4;
    let head_dim = 64;
    let page_size = 16;
    let num_blocks = 8;
    let seq_len = page_size * 2; // 2 blocks per sequence

    // Query: [num_seqs, num_heads, head_dim]
    let q_size = num_seqs * num_heads * head_dim;
    let q = generate_test_data(q_size, 42);

    // KV cache: [num_blocks, page_size, num_heads, head_dim]
    let cache_size = num_blocks * page_size * num_heads * head_dim;
    let k_cache = generate_test_data(cache_size, 43);
    let v_cache = generate_test_data(cache_size, 44);

    // Page table: [num_seqs, blocks_per_seq]
    let blocks_per_seq = 2;
    let page_table = generate_page_table(num_seqs, blocks_per_seq);

    // Sequence lengths (in blocks, not tokens - matches page_table entries per seq)
    let seq_lens: Vec<u32> = vec![blocks_per_seq as u32; num_seqs];

    // Output: [num_seqs, num_heads, head_dim]
    let mut output = vec![0.0f32; q_size];

    let config = PagedAttentionConfig {
        page_size,
        num_kv_heads: num_heads,
        head_dim,
        block_size: 128,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
    };

    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output,
        config,
    );

    // Verify no NaN/Inf
    assert_no_nan_inf(&output, "CPU basic output");

    // Verify output is non-trivial
    let sum: f32 = output.iter().sum();
    assert!(
        sum.abs() > 1e-10,
        "Output appears to be all zeros, sum={}",
        sum
    );
}

/// Test Paged Attention CPU consistency.
#[test]
fn test_paged_attention_cpu_consistency() {
    let dispatcher = CpuBackend::new();

    let num_seqs = 4;
    let num_heads = 8;
    let head_dim = 64;
    let page_size = 16;
    let num_blocks = 32;
    let blocks_per_seq = 4;

    let q_size = num_seqs * num_heads * head_dim;
    let q = generate_test_data(q_size, 100);

    let cache_size = num_blocks * page_size * num_heads * head_dim;
    let k_cache = generate_test_data(cache_size, 101);
    let v_cache = generate_test_data(cache_size, 102);

    let page_table = generate_page_table(num_seqs, blocks_per_seq);
    // seq_lens in blocks, not tokens
    let seq_lens: Vec<u32> = vec![blocks_per_seq as u32; num_seqs];

    let config = PagedAttentionConfig {
        page_size,
        num_kv_heads: num_heads,
        head_dim,
        block_size: 128,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
    };

    // Run twice
    let mut output1 = vec![0.0f32; q_size];
    let mut output2 = vec![0.0f32; q_size];

    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output1,
        config.clone(),
    );
    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output2,
        config,
    );

    // Should be identical for deterministic CPU implementation
    assert_eq!(
        output1, output2,
        "CPU implementation should be deterministic"
    );
}

/// Test Paged Attention with different sequence lengths.
#[test]
fn test_paged_attention_variable_seq_lengths() {
    let dispatcher = CpuBackend::new();

    let num_seqs = 3;
    let num_heads = 4;
    let head_dim = 32;
    let page_size = 16;
    let num_blocks = 12;
    let max_blocks_per_seq = 4;

    let q_size = num_seqs * num_heads * head_dim;
    let q = generate_test_data(q_size, 200);

    let cache_size = num_blocks * page_size * num_heads * head_dim;
    let k_cache = generate_test_data(cache_size, 201);
    let v_cache = generate_test_data(cache_size, 202);

    let page_table = generate_page_table(num_seqs, max_blocks_per_seq);

    // Different sequence lengths for each sequence (in blocks, not tokens)
    let seq_lens: Vec<u32> = vec![
        1,  // 1 block
        2,  // 2 blocks
        4,  // 4 blocks
    ];

    let mut output = vec![0.0f32; q_size];

    let config = PagedAttentionConfig {
        page_size,
        num_kv_heads: num_heads,
        head_dim,
        block_size: 128,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
    };

    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output,
        config,
    );

    assert_no_nan_inf(&output, "Variable seq lengths output");
}

/// CUDA GPU vs CPU comparison test for Paged Attention.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_cuda_vs_cpu_fp32() {
    use gllm_kernels::detect_backend;

    let detected = detect_backend();
    if detected.backend != BackendType::Cuda {
        eprintln!("CUDA not available, skipping GPU comparison test");
        return;
    }

    let cpu_dispatcher = CpuBackend::new();
    let cuda_dispatcher = CudaBackend::new();

    let num_seqs = 4;
    let num_heads = 8;
    let head_dim = 64;
    let page_size = 16;
    let num_blocks = 64;
    let blocks_per_seq = 8;

    let q_size = num_seqs * num_heads * head_dim;
    let q = generate_test_data(q_size, 300);

    let cache_size = num_blocks * page_size * num_heads * head_dim;
    let k_cache = generate_test_data(cache_size, 301);
    let v_cache = generate_test_data(cache_size, 302);

    let page_table = generate_page_table(num_seqs, blocks_per_seq);
    // seq_lens in blocks, not tokens
    let seq_lens: Vec<u32> = vec![blocks_per_seq as u32; num_seqs];

    let config = PagedAttentionConfig {
        page_size,
        num_kv_heads: num_heads,
        head_dim,
        block_size: 128,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
    };

    // CPU reference (Ground Truth)
    let mut cpu_output = vec![0.0f32; q_size];
    run_paged_attention(
        &cpu_dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut cpu_output,
        config.clone(),
    );

    // CUDA GPU
    let mut cuda_output = vec![0.0f32; q_size];
    run_paged_attention(
        &cuda_dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut cuda_output,
        config,
    );

    // FP32 precision: rtol=1e-5, atol=1e-6
    assert_close_f32(
        &cuda_output,
        &cpu_output,
        1e-5,
        1e-6,
        "CUDA vs CPU FP32 Paged Attention",
    );
}

/// Test Paged Attention stability options.
#[test]
fn test_paged_attention_stability_options() {
    let dispatcher = CpuBackend::new();

    let num_seqs = 2;
    let num_heads = 4;
    let head_dim = 64;
    let page_size = 16;
    let num_blocks = 16;
    let blocks_per_seq = 4;

    let q_size = num_seqs * num_heads * head_dim;
    let mut q = generate_test_data(q_size, 400);

    let cache_size = num_blocks * page_size * num_heads * head_dim;
    let mut k_cache = generate_test_data(cache_size, 401);
    let v_cache = generate_test_data(cache_size, 402);

    // Scale up to stress numerical stability
    for x in q.iter_mut() {
        *x *= 10.0;
    }
    for x in k_cache.iter_mut() {
        *x *= 10.0;
    }

    let page_table = generate_page_table(num_seqs, blocks_per_seq);
    // seq_lens in blocks, not tokens
    let seq_lens: Vec<u32> = vec![blocks_per_seq as u32; num_seqs];

    // With stability features
    let config_stable = PagedAttentionConfig {
        page_size,
        num_kv_heads: num_heads,
        head_dim,
        block_size: 128,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
    };

    // Without stability features
    let config_basic = PagedAttentionConfig {
        use_log_space_softmax: false,
        use_kahan_accumulator: false,
        ..config_stable.clone()
    };

    let mut output_stable = vec![0.0f32; q_size];
    let mut output_basic = vec![0.0f32; q_size];

    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output_stable,
        config_stable,
    );
    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output_basic,
        config_basic,
    );

    // Both should produce valid output
    assert_no_nan_inf(&output_stable, "Stable config output");
    assert_no_nan_inf(&output_basic, "Basic config output");
}

/// Test Paged Attention with GQA (grouped query attention).
/// num_kv_heads < num_heads
#[test]
fn test_paged_attention_gqa() {
    let dispatcher = CpuBackend::new();

    let num_seqs = 2;
    let num_heads = 8;       // Query heads
    let num_kv_heads = 2;    // KV heads (GQA: 4:1 ratio)
    let head_dim = 64;
    let page_size = 16;
    let num_blocks = 8;
    let blocks_per_seq = 2;

    // Query uses num_heads
    let q_size = num_seqs * num_heads * head_dim;
    let q = generate_test_data(q_size, 500);

    // KV cache uses num_kv_heads
    let cache_size = num_blocks * page_size * num_kv_heads * head_dim;
    let k_cache = generate_test_data(cache_size, 501);
    let v_cache = generate_test_data(cache_size, 502);

    let page_table = generate_page_table(num_seqs, blocks_per_seq);
    // seq_lens in blocks, not tokens
    let seq_lens: Vec<u32> = vec![blocks_per_seq as u32; num_seqs];

    let mut output = vec![0.0f32; q_size];

    let config = PagedAttentionConfig {
        page_size,
        num_kv_heads,
        head_dim,
        block_size: 128,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
    };

    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output,
        config,
    );

    assert_no_nan_inf(&output, "GQA output");
}

/// Test Paged Attention with single sequence.
#[test]
fn test_paged_attention_single_seq() {
    let dispatcher = CpuBackend::new();

    let num_seqs = 1;
    let num_heads = 4;
    let head_dim = 32;
    let page_size = 16;
    let num_blocks = 4;
    let blocks_per_seq = 4;

    let q_size = num_seqs * num_heads * head_dim;
    let q = generate_test_data(q_size, 600);

    let cache_size = num_blocks * page_size * num_heads * head_dim;
    let k_cache = generate_test_data(cache_size, 601);
    let v_cache = generate_test_data(cache_size, 602);

    let page_table = generate_page_table(num_seqs, blocks_per_seq);
    // seq_lens in blocks, not tokens
    let seq_lens: Vec<u32> = vec![blocks_per_seq as u32];

    let mut output = vec![0.0f32; q_size];

    let config = PagedAttentionConfig {
        page_size,
        num_kv_heads: num_heads,
        head_dim,
        block_size: 128,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
    };

    run_paged_attention(
        &dispatcher,
        &q,
        &k_cache,
        &v_cache,
        &page_table,
        &seq_lens,
        &mut output,
        config,
    );

    assert_no_nan_inf(&output, "Single sequence output");
}
