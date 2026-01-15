//! Rerank Pipeline Multi-Backend Performance Test
//! Tests all available backends: CUDA > ROCm > Metal > WGPU > CPU

use gllm_kernels::{KernelDispatcher, GpuRerankConfig};
use std::time::Instant;

fn main() {
    println!("=== Multi-Backend Rerank Pipeline Performance Test ===\n");
    
    let dispatcher = KernelDispatcher::new();
    println!("🎯 Selected Backend: {:?}", dispatcher.backend());
    println!("   (Priority: CUDA > ROCm > Metal > WGPU > CPU)\n");
    
    // Test configurations
    let configs = [
        (100_000, 512, 10000, 100, "100K vectors"),
        (500_000, 512, 10000, 100, "500K vectors"),
        (1_000_000, 512, 10000, 100, "1M vectors"),
    ];
    
    println!("=== Full Pipeline Benchmarks ===\n");
    
    for (num_vectors, dim, binary_k, int8_k, label) in configs {
        println!("--- {} (dim={}) ---", label, dim);
        
        let binary_elements_per_vec = dim / 32;
        let int8_elements_per_vec = dim / 4;
        
        // Generate test data
        let binary_query: Vec<u32> = (0..binary_elements_per_vec)
            .map(|i| 0xAAAA_5555u32.wrapping_add(i as u32))
            .collect();
        let int8_query: Vec<u32> = (0..int8_elements_per_vec)
            .map(|i| 0x7F7F_7F7Fu32.wrapping_sub(i as u32))
            .collect();
        let binary_database: Vec<u32> = (0..num_vectors * binary_elements_per_vec)
            .map(|i| (i as u32).wrapping_mul(0x9E3779B9))
            .collect();
        let int8_database: Vec<u32> = (0..num_vectors * int8_elements_per_vec)
            .map(|i| (i as u32).wrapping_mul(0x85EBCA6B))
            .collect();
        
        let config = GpuRerankConfig { binary_k, int8_k, dim };
        
        // Warmup
        let _ = dispatcher.rerank_pipeline(
            &binary_query, &binary_database,
            &int8_query, &int8_database,
            num_vectors, &config, 0.01,
        );
        
        // Benchmark
        let iterations = 5;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let result = dispatcher.rerank_pipeline(
                &binary_query, &binary_database,
                &int8_query, &int8_database,
                num_vectors, &config, 0.01,
            );
            std::hint::black_box(&result);
        }
        
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let throughput = num_vectors as f64 / avg_ms * 1000.0;
        
        println!("  Time: {:.2} ms", avg_ms);
        println!("  Throughput: {:.2}M vectors/sec", throughput / 1_000_000.0);
        
        // Memory usage
        let binary_mb = (binary_database.len() * 4) as f64 / 1024.0 / 1024.0;
        let int8_mb = (int8_database.len() * 4) as f64 / 1024.0 / 1024.0;
        println!("  Memory: binary={:.1}MB, int8={:.1}MB\n", binary_mb, int8_mb);
    }
    
    // Individual operations benchmark using dispatcher's native API
    println!("=== Individual Operations (1M vectors) ===\n");
    
    let num_vectors = 1_000_000;
    let dim = 512;
    
    // Binary IP Hamming (dispatcher uses u64)
    let binary_query_u64: Vec<u64> = (0..dim/64).map(|i| 0xAAAA_5555_AAAA_5555u64.wrapping_add(i as u64)).collect();
    let binary_database_u64: Vec<u64> = (0..num_vectors * dim / 64)
        .map(|i| (i as u64).wrapping_mul(0x9E3779B9_7F4A7C15u64))
        .collect();
    
    let binary_config = gllm_kernels::ops::embedding::BinaryIpConfig {
        dim, num_queries: 1, num_vectors,
    };
    let mut binary_scores = vec![0i32; num_vectors];
    
    let start = Instant::now();
    for _ in 0..5 {
        dispatcher.binary_ip_hamming(
            &binary_query_u64, &binary_database_u64, &mut binary_scores, binary_config.clone()
        );
        std::hint::black_box(&binary_scores);
    }
    let binary_time = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
    println!("Binary Hamming: {:.2} ms ({:.1}M vec/s)", 
             binary_time, num_vectors as f64 / binary_time / 1000.0);
    
    // Int8 Dot Product (dispatcher uses i8)
    let int8_query_i8: Vec<i8> = (0..dim).map(|i| ((i % 256) as i8).wrapping_sub(64)).collect();
    let int8_database_i8: Vec<i8> = (0..num_vectors * dim)
        .map(|i| ((i * 7 % 256) as i8).wrapping_sub(64))
        .collect();
    
    let int8_config = gllm_kernels::ops::embedding::Int8DotConfig {
        dim, num_queries: 1, num_vectors, scale: 0.01,
    };
    let mut int8_scores = vec![0f32; num_vectors];
    
    let start = Instant::now();
    for _ in 0..5 {
        dispatcher.int8_dot_product(
            &int8_query_i8, &int8_database_i8, &mut int8_scores, int8_config.clone()
        );
        std::hint::black_box(&int8_scores);
    }
    let int8_time = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
    println!("Int8 Dot Product: {:.2} ms ({:.1}M vec/s)", 
             int8_time, num_vectors as f64 / int8_time / 1000.0);
    
    // Summary
    println!("\n=== Performance Summary ===");
    println!("Backend: {:?}", dispatcher.backend());
    println!("Binary Hamming: {:.1}M vec/s", num_vectors as f64 / binary_time / 1000.0);
    println!("Int8 Dot Product: {:.1}M vec/s", num_vectors as f64 / int8_time / 1000.0);
    
    // Verify full pipeline output
    println!("\n=== Output Verification (1M vectors) ===");
    let config = GpuRerankConfig { binary_k: 10000, int8_k: 100, dim };
    let binary_query: Vec<u32> = (0..dim/32).map(|i| 0xAAAAu32.wrapping_add(i as u32)).collect();
    let int8_query: Vec<u32> = (0..dim/4).map(|i| 0x7F7F7F7Fu32.wrapping_sub(i as u32)).collect();
    let binary_database: Vec<u32> = (0..1_000_000 * dim / 32)
        .map(|i| (i as u32).wrapping_mul(0x9E3779B9))
        .collect();
    let int8_database: Vec<u32> = (0..1_000_000 * dim / 4)
        .map(|i| (i as u32).wrapping_mul(0x85EBCA6B))
        .collect();
    
    let start = Instant::now();
    let result = dispatcher.rerank_pipeline(
        &binary_query, &binary_database,
        &int8_query, &int8_database,
        1_000_000, &config, 0.01,
    );
    let pipeline_time = start.elapsed();
    
    match result {
        Ok(r) => {
            println!("✅ Pipeline completed in {:.2} ms", pipeline_time.as_secs_f64() * 1000.0);
            println!("   Throughput: {:.1}M vectors/sec", 1_000_000.0 / pipeline_time.as_secs_f64() / 1_000_000.0);
            println!("   Output: {} candidates", r.indices.len());
            println!("   Top 5 indices: {:?}", &r.indices[..5.min(r.indices.len())]);
            println!("   Top 5 scores: {:?}", &r.scores[..5.min(r.scores.len())]);
        }
        Err(e) => {
            println!("❌ Pipeline error: {}", e);
        }
    }
}
