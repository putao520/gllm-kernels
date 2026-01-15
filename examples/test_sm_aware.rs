//! Test SM-aware PTX loading on the current GPU.
//!
//! Run with: RUST_LOG=info cargo run --example test_sm_aware

use std::sync::Arc;
use cudarc::driver::CudaContext;
use half::f16;

fn main() {
    // Initialize simple logging
    std::env::set_var("RUST_LOG", "info");

    println!("=== SM-Aware PTX Loading Test ===\n");

    // Create CUDA context
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => {
            println!("✓ CUDA context created");
            Arc::new(ctx)
        }
        Err(e) => {
            eprintln!("✗ Failed to create CUDA context: {}", e);
            return;
        }
    };

    // Test SM detection
    match gllm_kernels::cuda_kernels::detect_sm_version(&ctx) {
        Ok(sm) => println!("✓ Detected SM version: sm_{}", sm),
        Err(e) => {
            eprintln!("✗ Failed to detect SM version: {}", e);
            return;
        }
    }

    // Test FlashAttention kernel loading with SM-aware PTX
    println!("\n--- Testing FlashAttention Kernel ---");
    match gllm_kernels::cuda_kernels::FlashAttentionKernel::new(&ctx) {
        Ok(_kernel) => {
            println!("✓ FlashAttention kernel loaded successfully!");

            // Run a simple kernel test
            let stream = Arc::new(ctx.default_stream());

            // Test parameters
            let batch_size = 1;
            let num_heads = 4;
            let seq_len = 32;
            let head_dim = 64;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Allocate input tensors (f32)
            let total_size = batch_size * num_heads * seq_len * head_dim;
            let q_data: Vec<f32> = (0..total_size).map(|i| (i as f32 % 10.0) * 0.1).collect();
            let k_data: Vec<f32> = (0..total_size).map(|i| ((i + 1) as f32 % 10.0) * 0.1).collect();
            let v_data: Vec<f32> = (0..total_size).map(|i| ((i + 2) as f32 % 10.0) * 0.1).collect();

            let q = match stream.clone_htod(&q_data) {
                Ok(q) => q,
                Err(e) => {
                    eprintln!("✗ Failed to copy Q: {}", e);
                    return;
                }
            };
            let k = match stream.clone_htod(&k_data) {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("✗ Failed to copy K: {}", e);
                    return;
                }
            };
            let v = match stream.clone_htod(&v_data) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("✗ Failed to copy V: {}", e);
                    return;
                }
            };

            // Run forward pass
            match _kernel.forward_f32(
                &stream,
                &q,
                &k,
                &v,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                false, // is_causal
                scale,
                0,     // position_offset
            ) {
                Ok(output) => {
                    // Copy output back
                    let mut output_host = vec![0.0f32; total_size];
                    if let Err(e) = stream.memcpy_dtoh(&output, &mut output_host) {
                        eprintln!("✗ Failed to copy output: {}", e);
                        return;
                    }

                    // Verify output is valid (not NaN/Inf)
                    let valid = output_host.iter().all(|&x| x.is_finite());
                    if valid {
                        println!("✓ f32 kernel executed successfully!");
                        println!("  Output sample: [{:.4}, {:.4}, {:.4}, ...]",
                            output_host[0], output_host[1], output_host[2]);
                    } else {
                        eprintln!("✗ Output contains NaN or Inf values");
                    }
                }
                Err(e) => {
                    eprintln!("✗ Kernel execution failed: {}", e);
                }
            }

            // Test f16 kernel
            println!("\n--- Testing f16 Kernel ---");
            let q_f16: Vec<f16> = q_data.iter().map(|&x| f16::from_f32(x)).collect();
            let k_f16: Vec<f16> = k_data.iter().map(|&x| f16::from_f32(x)).collect();
            let v_f16: Vec<f16> = v_data.iter().map(|&x| f16::from_f32(x)).collect();

            let q16 = match stream.clone_htod(&q_f16) {
                Ok(q) => q,
                Err(e) => {
                    eprintln!("✗ Failed to copy Q f16: {}", e);
                    return;
                }
            };
            let k16 = match stream.clone_htod(&k_f16) {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("✗ Failed to copy K f16: {}", e);
                    return;
                }
            };
            let v16 = match stream.clone_htod(&v_f16) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("✗ Failed to copy V f16: {}", e);
                    return;
                }
            };

            match _kernel.forward_f16(
                &stream,
                &q16,
                &k16,
                &v16,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                false,
                scale,
                0,
            ) {
                Ok(output) => {
                    let mut output_host = vec![f16::ZERO; total_size];
                    if let Err(e) = stream.memcpy_dtoh(&output, &mut output_host) {
                        eprintln!("✗ Failed to copy f16 output: {}", e);
                        return;
                    }

                    let valid = output_host.iter().all(|x| x.to_f32().is_finite());
                    if valid {
                        println!("✓ f16 kernel executed successfully!");
                        println!("  Output sample: [{:.4}, {:.4}, {:.4}, ...]",
                            output_host[0].to_f32(), output_host[1].to_f32(), output_host[2].to_f32());
                    } else {
                        eprintln!("✗ f16 output contains NaN or Inf values");
                    }
                }
                Err(e) => {
                    eprintln!("✗ f16 kernel execution failed: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to load FlashAttention kernel: {}", e);
        }
    }

    println!("\n=== Test Complete ===");
}
