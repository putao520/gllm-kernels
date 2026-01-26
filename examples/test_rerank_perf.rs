//! Rerank Pipeline GPU Performance Test

use gllm_kernels::ops::embedding::{rerank_binary_stage, rerank_int8_stage, RerankResult};
use gllm_kernels::GpuRerankConfig;
use std::time::Instant;

fn rerank_pipeline_cpu(
    binary_query: &[u64],
    binary_database: &[u64],
    int8_query: &[i8],
    int8_database: &[i8],
    num_vectors: usize,
    config: &GpuRerankConfig,
    scale: f32,
) -> RerankResult {
    let binary = rerank_binary_stage(
        binary_query,
        binary_database,
        num_vectors,
        config.dim,
        config.binary_k,
    );
    rerank_int8_stage(
        int8_query,
        int8_database,
        &binary.indices,
        config.dim,
        scale,
        config.int8_k,
    )
}

fn main() {
    println!("=== GPU Rerank Pipeline Performance Test ===\n");
    
    // Test configurations
    let configs = [
        (100_000, 1024, 10000, 100),   // 100K vectors, dim=1024, binary_k=10K, int8_k=100
        (500_000, 768, 10000, 100),    // 500K vectors, dim=768
        (1_000_000, 512, 10000, 100),  // 1M vectors, dim=512
    ];
    
    println!("Backend: {:?}\n", gllm_kernels::detect_backend());
    
    for (num_vectors, dim, binary_k, int8_k) in configs {
        println!("--- Config: {} vectors, dim={}, binary_k={}, int8_k={} ---", 
                 num_vectors, dim, binary_k, int8_k);
        
        // Generate test data
        let binary_elements_per_vec = dim / 64; // packed u64 for binary
        
        // Query embeddings
        let binary_query: Vec<u64> = (0..binary_elements_per_vec)
            .map(|i| 0xAAAA_5555_AAAA_5555u64.wrapping_add(i as u64))
            .collect();
        let int8_query: Vec<i8> = (0..dim)
            .map(|i| ((i % 256) as i8).wrapping_sub(64))
            .collect();
        
        // Database embeddings
        let binary_database: Vec<u64> = (0..num_vectors * binary_elements_per_vec)
            .map(|i| (i as u64).wrapping_mul(0x9E3779B9_7F4A7C15u64))
            .collect();
        let int8_database: Vec<i8> = (0..num_vectors * dim)
            .map(|i| ((i * 7 % 256) as i8).wrapping_sub(64))
            .collect();
        
        let config = GpuRerankConfig {
            binary_k,
            int8_k,
            dim,
        };
        
        // Warmup
        let _ = rerank_pipeline_cpu(
            &binary_query,
            &binary_database,
            &int8_query,
            &int8_database,
            num_vectors,
            &config,
            0.01,
        );
        
        // Benchmark
        let iterations = 5;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let result = rerank_pipeline_cpu(
                &binary_query,
                &binary_database,
                &int8_query,
                &int8_database,
                num_vectors,
                &config,
                0.01,
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
    let int8_elements_per_vec = dim;
    
    let binary_query: Vec<u64> = (0..binary_elements_per_vec)
        .map(|i| 0xAAAA_5555_AAAA_5555u64.wrapping_add(i as u64))
        .collect();
    let int8_query: Vec<i8> = (0..int8_elements_per_vec)
        .map(|i| ((i % 256) as i8).wrapping_sub(64))
        .collect();
    let binary_database: Vec<u64> = (0..num_vectors * binary_elements_per_vec)
        .map(|i| (i as u64).wrapping_mul(0x9E3779B9_7F4A7C15u64))
        .collect();
    let int8_database: Vec<i8> = (0..num_vectors * int8_elements_per_vec)
        .map(|i| ((i * 7 % 256) as i8).wrapping_sub(64))
        .collect();
    
    let config = GpuRerankConfig {
        binary_k: 10000,
        int8_k: 100,
        dim,
    };
    
    let start = Instant::now();
    let result = rerank_pipeline_cpu(
        &binary_query,
        &binary_database,
        &int8_query,
        &int8_database,
        num_vectors,
        &config,
        0.01,
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
