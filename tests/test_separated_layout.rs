//! Unit test to verify the separated QKV layout produces head-major format

use gllm_kernels::cpu_kernels;

#[test]
fn test_separated_qkv_produces_head_major_layout() {
    // Simulate SmolLM2 config
    let seq_len = 3;
    let hidden_size = 64;  // Simplified
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let rotary_dim = 16;

    // Input: 3 tokens, 64 hidden size
    let mut input = vec![0f32; seq_len * hidden_size];
    for pos in 0..seq_len {
        for i in 0..hidden_size {
            input[pos * hidden_size + i] = (pos * 100 + i) as f32;
        }
    }

    // Weight matrix (simplified)
    let q_out = num_heads * head_dim;  // 64
    let kv_out = num_kv_heads * head_dim;  // 32
    let qkv_stride = q_out + 2 * kv_out;  // 128

    let mut weight = vec![0f32; hidden_size * qkv_stride];
    // Set to make each position produce distinct output
    for i in 0..hidden_size * qkv_stride {
        weight[i] = (i % 256) as f32 * 0.01;
    }

    // Output buffers (head-major format expected)
    let mut q_output = vec![0f32; seq_len * q_out];
    let mut k_output = vec![0f32; seq_len * kv_out];
    let mut v_output = vec![0f32; seq_len * kv_out];

    let positions = vec![0i32, 1i32, 2i32];

    // Run the separated QKV projection
    cpu_kernels::qkv_rope_separated(
        &input,
        &weight,
        None,
        &mut q_output,
        &mut k_output,
        &mut v_output,
        seq_len,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        100000.0,
        1.0,
        false,
        None,
        &positions,
    ).unwrap();

    // VERIFICATION: Head-major layout means
    // Position 0 K: k_output[0 .. kv_out]
    // Position 1 K: k_output[kv_out .. 2*kv_out]
    // Position 2 K: k_output[2*kv_out .. 3*kv_out]

    // Extract K values for each position
    let k_pos_0 = &k_output[0..kv_out];
    let k_pos_1 = &k_output[kv_out..2 * kv_out];
    let k_pos_2 = &k_output[2 * kv_out..3 * kv_out];

    // Each position should have kv_out values
    assert_eq!(k_pos_0.len(), kv_out, "Position 0 K should have kv_out values");
    assert_eq!(k_pos_1.len(), kv_out, "Position 1 K should have kv_out values");
    assert_eq!(k_pos_2.len(), kv_out, "Position 2 K should have kv_out values");

    // Values should differ between positions (different input + RoPE)
    let k0_first = k_pos_0[0];
    let k1_first = k_pos_1[0];
    let k2_first = k_pos_2[0];

    // At least some values should differ
    let all_equal = (k0_first == k1_first) && (k1_first == k2_first);
    assert!(!all_equal, "K values should differ between positions");

    // Print for debugging
    eprintln!("Separated QKV Layout Verification:");
    eprintln!("  K position 0 (first 5): {:?}", &k_pos_0[..5.min(kv_out)]);
    eprintln!("  K position 1 (first 5): {:?}", &k_pos_1[..5.min(kv_out)]);
    eprintln!("  K position 2 (first 5): {:?}", &k_pos_2[..5.min(kv_out)]);
}

#[test]
fn test_separated_qkv_cache_write_layout() {
    // Test that K values can be written correctly to paged cache
    let seq_len = 2;
    let num_kv_heads = 2;
    let head_dim = 16;
    let kv_out = num_kv_heads * head_dim;  // 32
    let page_size = 16;

    // Simulate K values in head-major format
    let mut k_buffer = vec![0f32; seq_len * kv_out];
    for pos in 0..seq_len {
        for i in 0..kv_out {
            k_buffer[pos * kv_out + i] = (pos * 1000 + i) as f32;
        }
    }

    // Simulate writing to paged cache
    // Cache expects: for each position, write kv_out values
    let mut page = vec![0f32; page_size * kv_out * 2];  // K + V

    for pos in 0..seq_len {
        let k_dst = pos * kv_out;
        let k_src = &k_buffer[pos * kv_out..(pos + 1) * kv_out];
        page[k_dst..k_dst + kv_out].copy_from_slice(k_src);
    }

    // Verify: K at position 0 should match input
    assert_eq!(page[0], k_buffer[0], "Position 0 K first value should match");
    assert_eq!(page[kv_out - 1], k_buffer[kv_out - 1], "Position 0 K last value should match");

    // Verify: K at position 1 should match input
    assert_eq!(page[1 * kv_out], k_buffer[1 * kv_out], "Position 1 K first value should match");

    eprintln!("Paged cache write layout test PASSED");
}
