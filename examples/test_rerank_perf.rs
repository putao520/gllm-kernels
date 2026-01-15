//! Rerank Pipeline GPU Performance Test

use gllm_kernels::{KernelDispatcher, GpuRerankConfig};
use std::time::Instant;

fn main() {
    println!("=== GPU Rerank Pipeline Performance Test ===\n");
    
    // Test configurations
    let configs = [
        (100_000, 1024, 10000, 100),   // 100K vectors, dim=1024, binary_k=10K, int8_k=100
        (500_000, 768, 10000, 100),    // 500K vectors, dim=768
        (1_000_000, 512, 10000, 100),  // 1M vectors, dim=512
    ];
    
    let dispatcher = KernelDispatcher::new();
    println!("Backend: {:?}\n", dispatcher.backend());
    
    for (num_vectors, dim, binary_k, int8_k) in configs {
        println!("--- Config: {} vectors, dim={}, binary_k={}, int8_k={} ---", 
                 num_vectors, dim, binary_k, int8_k);
        
        // Generate test data
        let binary_elements_per_vec = dim / 32;  // packed u32 for binary
        let int8_elements_per_vec = dim / 4;     // packed u32 for int8
        
        // Query embeddings
        let binary_query: Vec<u32> = (0..binary_elements_per_vec)
            .map(|i| 0xAAAA_5555u32.wrapping_add(i as u32))
            .collect();
        let int8_query: Vec<u32> = (0..int8_elements_per_vec)
            .map(|i| 0x7F7F_7F7Fu32.wrapping_sub(i as u32))
            .collect();
        
        // Database embeddings
        let binary_database: Vec<u32> = (0..num_vectors * binary_elements_per_vec)
            .map(|i| (i as u32).wrapping_mul(0x9E3779B9))
            .collect();
        let int8_database: Vec<u32> = (0..num_vectors * int8_elements_per_vec)
            .map(|i| (i as u32).wrapping_mul(0x85EBCA6B))
            .collect();
        
        let config = GpuRerankConfig {
            binary_k,
            int8_k,
            dim,
        };
        
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
            
            if let Ok(r) = &result {
                // Prevent optimization
                std::hint::black_box(&r.indices);
            }
        }
        
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let throughput = num_vectors as f64 / avg_ms * 1000.0;
        
        println!("  Average time: {:.2} ms", avg_ms);
        println!("  Throughput: {:.0} vectors/sec", throughput);
        println!("  Memory: binary={:.1}MB, int8={:.1}MB", 
                 binary_database.len() as f64 * 4.0 / 1024.0 / 1024.0,
                 int8_database.len() as f64 * 4.0 / 1024.0 / 1024.0);
        println!();
    }
    
    // Detailed stage timing
    println!("=== Detailed Stage Timing (1M vectors, dim=512) ===\n");
    
    let num_vectors = 1_000_000;
    let dim = 512;
    let binary_elements_per_vec = dim / 32;
    let int8_elements_per_vec = dim / 4;
    
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
    
    let config = GpuRerankConfig {
        binary_k: 10000,
        int8_k: 100,
        dim,
    };
    
    let start = Instant::now();
    let result = dispatcher.rerank_pipeline(
        &binary_query, &binary_database,
        &int8_query, &int8_database,
        num_vectors, &config, 0.01,
    );
    let total = start.elapsed();
    
    match result {
        Ok(r) => {
            println!("Total time: {:.2} ms", total.as_secs_f64() * 1000.0);
            println!("Output: {} candidates", r.indices.len());
            println!("Top 5 indices: {:?}", &r.indices[..5.min(r.indices.len())]);
            println!("Top 5 scores: {:?}", &r.scores[..5.min(r.scores.len())]);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}
