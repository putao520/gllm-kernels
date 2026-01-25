//! GPU Backend Detection and Benchmark
//! 
//! Verifies which backend is active and benchmarks GPU performance.

use gllm_kernels::{KernelDispatcher, FlashAttentionConfig, BackendType, current_simd_path};
use std::time::Instant;

fn main() {
    println!("================================================");
    println!("        GPU BACKEND VERIFICATION                ");
    println!("================================================\n");
    
    // Enable logging to see backend selection
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    
    let dispatcher = KernelDispatcher::new();
    let backend = dispatcher.backend();
    
    println!("Active Backend: {:?}", backend);
    println!("CPU SIMD Path: {}", current_simd_path());
    println!();
    
    match backend {
        BackendType::Cuda => println!("✅ CUDA backend active"),
        BackendType::Rocm => println!("✅ ROCm/HSA backend active"),
        BackendType::Wgpu => println!("✅ WGPU backend active"),
        BackendType::Metal => println!("✅ Metal backend active"),
        BackendType::Cpu => println!("⚠️  CPU fallback active (GPU not detected)"),
    }
    println!();
    
    // Benchmark attention
    println!("--- Flash Attention Benchmark ---\n");
    
    let test_cases = vec![
        ("SmolLM2-135M decode", 3, 64, 1, 64),    // num_heads, head_dim, seq_q, seq_kv
        ("SmolLM2-135M prefill-64", 3, 64, 64, 64),
        ("Llama-7B decode", 32, 128, 1, 256),
        ("Llama-7B prefill-128", 32, 128, 128, 128),
    ];
    
    for (name, num_heads, head_dim, seq_q, seq_kv) in test_cases {
        println!("{}:", name);
        
        let q_size = num_heads * seq_q * head_dim;
        let kv_size = num_heads * seq_kv * head_dim;
        let out_size = num_heads * seq_q * head_dim;
        
        let q: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..kv_size).map(|i| (i as f32 * 0.001).cos()).collect();
        let v: Vec<f32> = (0..kv_size).map(|i| (i as f32 * 0.002).sin()).collect();
        let mut output = vec![0.0f32; out_size];
        
        let config = FlashAttentionConfig {
            causal: true,
            num_heads,
            head_dim,
            seq_len_q: seq_q,
            seq_len_kv: seq_kv,
            batch_size: 1,
            ..Default::default()
        };
        
        // Warmup
        for _ in 0..20 {
            dispatcher.flash_attention(&q, &k, &v, &mut output, config.clone());
        }
        
        // Benchmark
        let iterations = 500;
        let start = Instant::now();
        for _ in 0..iterations {
            dispatcher.flash_attention(&q, &k, &v, &mut output, config.clone());
        }
        let elapsed = start.elapsed();
        
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let tps = 1_000_000.0 / avg_us;
        
        println!("  Backend: {:?}", backend);
        println!("  Avg latency: {:.1} µs", avg_us);
        println!("  Throughput: {:.1} tokens/s", tps);
        println!();
    }
    
    println!("================================================");
    println!("Benchmark completed.");
}
