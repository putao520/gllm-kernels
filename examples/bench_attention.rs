//! Flash Attention Decode Benchmark
//! 
//! Benchmarks the optimized decode path vs. general path.

use std::time::Instant;
use gllm_kernels::backend::{auto_select_backend, TensorSlice, TensorSliceMut};
use gllm_kernels::{FlashAttentionConfig, current_simd_path};

fn main() {
    println!("================================================");
    println!("   FLASH ATTENTION DECODE BENCHMARK             ");
    println!("================================================");
    println!("SIMD Path: {}", current_simd_path());
    println!();
    
    let backend = auto_select_backend();
    
    // SmolLM2-135M attention dimensions
    let test_cases = vec![
        ("SmolLM2-135M (3 heads)", 3, 64, 64),   // num_heads, head_dim, seq_kv
        ("SmolLM2-135M (3 heads, 256 ctx)", 3, 64, 256),
        ("SmolLM2-360M (5 heads, 64 ctx)", 5, 64, 64),
        ("SmolLM2-360M (5 heads, 256 ctx)", 5, 64, 256),
        ("Llama-7B (32 heads, 64 ctx)", 32, 128, 64),
        ("Llama-7B (32 heads, 512 ctx)", 32, 128, 512),
    ];

    for (name, num_heads, head_dim, seq_kv) in test_cases {
        println!("{}:", name);
        println!("  Params: heads={}, head_dim={}, seq_kv={}", num_heads, head_dim, seq_kv);
        
        let q_size = num_heads * head_dim;
        let kv_size = num_heads * seq_kv * head_dim;
        
        let q: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..kv_size).map(|i| (i as f32 * 0.001).cos()).collect();
        let v: Vec<f32> = (0..kv_size).map(|i| (i as f32 * 0.002).sin()).collect();
        let mut output = vec![0.0f32; q_size];
        
        let config = FlashAttentionConfig {
            causal: true,
            num_heads,
            head_dim,
            seq_len_q: 1,  // DECODE path
            seq_len_kv: seq_kv,
            batch_size: 1,
            ..Default::default()
        };
        
        // Warmup
        for _ in 0..20 {
            backend
                .flash_attention(
                    TensorSlice::F32(&q),
                    TensorSlice::F32(&k),
                    TensorSlice::F32(&v),
                    TensorSliceMut::F32(&mut output),
                    config.clone(),
                )
                .expect("flash_attention failed");
        }
        
        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            backend
                .flash_attention(
                    TensorSlice::F32(&q),
                    TensorSlice::F32(&k),
                    TensorSlice::F32(&v),
                    TensorSliceMut::F32(&mut output),
                    config.clone(),
                )
                .expect("flash_attention failed");
        }
        let elapsed = start.elapsed();
        
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let tokens_per_sec = 1_000_000.0 / avg_us;
        
        println!("  Iterations: {}", iterations);
        println!("  Avg latency: {:.1} µs/token", avg_us);
        println!("  Throughput: {:.1} tokens/s (attention only)", tokens_per_sec);
        println!();
    }

    println!("================================================");
    println!("Benchmark completed.");
}
