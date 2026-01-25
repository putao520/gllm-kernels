//! SIMD Dot Product Benchmark
//! 
//! Benchmarks the performance of different dot product implementations
//! to validate the simd_asm module integration.

use std::time::Instant;
use gllm_kernels::{linear_forward, gemv_dispatch, gemv_single, gemv_parallel, ModelTier, SimdPath, current_simd_path};

fn main() {
    println!("================================================");
    println!("     SIMD DOT PRODUCT MICROBENCHMARK            ");
    println!("================================================");
    println!("SIMD Path: {}", current_simd_path());
    println!();
    
    // Test dimensions typical of LLM inference
    let test_cases = vec![
        ("SmolLM2-135M hidden", 576, 1),
        ("SmolLM2-360M hidden", 960, 1),
        ("Llama-2-7B hidden", 4096, 1),
        ("Llama-3-8B hidden", 4096, 1),
        ("Qwen2.5-72B hidden", 8192, 1),
    ];

    for (name, dim, batch) in &test_cases {
        let tier = ModelTier::from_hidden_size(*dim);
        println!("{}:", name);
        println!("  Dimension: {}x{} (Tier: {:?})", batch, dim, tier);
        
        // Prepare test data
        let input: Vec<f32> = (0..batch * dim)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let weight: Vec<f32> = (0..dim * dim)
            .map(|i| (i as f32 * 0.0001).cos())
            .collect();
        let mut output = vec![0.0f32; batch * dim];
        
        // Warmup
        for _ in 0..10 {
            linear_forward(&input, &weight, None, &mut output, *batch, *dim, *dim);
        }
        
        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            linear_forward(&input, &weight, None, &mut output, *batch, *dim, *dim);
        }
        let elapsed = start.elapsed();
        
        let ops_per_iter = 2.0 * (*batch as f64) * (*dim as f64) * (*dim as f64); // FLOPs per matmul
        let total_ops = ops_per_iter * (iterations as f64);
        let gflops = total_ops / elapsed.as_secs_f64() / 1e9;
        
        println!("  Iterations: {}", iterations);
        println!("  Time: {:.2?}", elapsed);
        println!("  GFLOPS: {:.2}", gflops);
        println!("  Avg per iter: {:.2?}", elapsed / iterations);
        println!();
    }

    println!("================================================");
    println!("     GEMV DISPATCH BENCHMARK (Batch=1)          ");
    println!("================================================\n");

    for (name, dim, _) in &test_cases {
        let tier = ModelTier::from_hidden_size(*dim);
        println!("{} (gemv_dispatch):", name);
        
        let input: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let weight: Vec<f32> = (0..dim * dim).map(|i| (i as f32 * 0.0001).cos()).collect();
        let mut output = vec![0.0f32; *dim];
        
        // Warmup
        for _ in 0..10 {
            gemv_dispatch(&input, &weight, &mut output, *dim, *dim);
        }
        
        let iterations = 500;
        let start = Instant::now();
        for _ in 0..iterations {
            gemv_dispatch(&input, &weight, &mut output, *dim, *dim);
        }
        let elapsed = start.elapsed();
        
        let gflops = 2.0 * (*dim as f64) * (*dim as f64) * (iterations as f64) / elapsed.as_secs_f64() / 1e9;
        println!("  Tier: {:?} | GFLOPS: {:.2} | Avg: {:.2?}", tier, gflops, elapsed / iterations);
    }

    println!("\n================================================");
    println!("Benchmark completed.");
}
