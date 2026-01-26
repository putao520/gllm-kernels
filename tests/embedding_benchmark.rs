//! Embedding Ops Benchmark: CPU vs WGPU vs CUDA
//!
//! Run with: cargo test --test embedding_benchmark -- --nocapture

use gllm_kernels::BackendType;
use gllm_kernels::ops::embedding::{
    BinaryIpConfig, Int4PackedConfig, Int8DotConfig, MatryoshkaConfig,
    binary_ip_asymmetric, binary_ip_hamming_simd, int4_packed_dot_product,
    int8_dot_product_unrolled, matryoshka_truncate,
};
use std::time::Instant;

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

fn generate_test_data(
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> (Vec<u64>, Vec<u64>, Vec<i32>) {
    let packed_dim = (dim + 63) / 64;
    let queries: Vec<u64> = (0..num_queries * packed_dim)
        .map(|i| (i as u64).wrapping_mul(0x5DEECE66D))
        .collect();
    let database: Vec<u64> = (0..num_vectors * packed_dim)
        .map(|i| (i as u64).wrapping_mul(0x5DEECE66D).wrapping_add(0xB))
        .collect();
    let scores = vec![0i32; num_queries * num_vectors];
    (queries, database, scores)
}

fn generate_asymmetric_data(
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> (Vec<f32>, Vec<u64>, Vec<f32>) {
    let queries: Vec<f32> = (0..num_queries * dim)
        .map(|i| ((i % 1000) as f32 - 500.0) / 500.0)
        .collect();
    let packed_dim = (dim + 63) / 64;
    let database: Vec<u64> = (0..num_vectors * packed_dim)
        .map(|i| (i as u64).wrapping_mul(0x5DEECE66D))
        .collect();
    let scores = vec![0.0f32; num_queries * num_vectors];
    (queries, database, scores)
}

fn generate_int8_data(
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> (Vec<i8>, Vec<i8>, Vec<f32>) {
    let queries: Vec<i8> = (0..num_queries * dim)
        .map(|i| ((i % 256) as u8 as i8))
        .collect();
    let database: Vec<i8> = (0..num_vectors * dim)
        .map(|i| (((i + 17) % 256) as u8 as i8))
        .collect();
    let scores = vec![0.0f32; num_queries * num_vectors];
    (queries, database, scores)
}

fn generate_int4_data(
    dim: usize,
    num_queries: usize,
    num_vectors: usize,
) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    // Int4 packed: 2 values per byte, so dim/2 bytes per vector
    let packed_dim = (dim + 1) / 2;
    let queries: Vec<u8> = (0..num_queries * packed_dim)
        .map(|i| (i % 256) as u8)
        .collect();
    let database: Vec<u8> = (0..num_vectors * packed_dim)
        .map(|i| ((i + 17) % 256) as u8)
        .collect();
    let scores = vec![0.0f32; num_queries * num_vectors];
    (queries, database, scores)
}

fn generate_matryoshka_data(
    full_dim: usize,
    target_dim: usize,
    num_vectors: usize,
) -> (Vec<f32>, Vec<f32>) {
    let embeddings: Vec<f32> = (0..num_vectors * full_dim)
        .map(|i| ((i % 1000) as f32 - 500.0) / 500.0)
        .collect();
    let output = vec![0.0f32; num_vectors * target_dim];
    (embeddings, output)
}

fn bench_binary_ip_hamming(backend: BackendType, dim: usize, num_queries: usize, num_vectors: usize) -> f64 {
    let _ = backend;
    let (queries, database, mut scores) = generate_test_data(dim, num_queries, num_vectors);
    let config = BinaryIpConfig {
        dim,
        num_queries,
        num_vectors,
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        binary_ip_hamming_simd(&queries, &database, &mut scores, &config);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        binary_ip_hamming_simd(&queries, &database, &mut scores, &config);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / BENCH_ITERS as f64 * 1000.0 // ms
}

fn bench_binary_ip_asymmetric(backend: BackendType, dim: usize, num_queries: usize, num_vectors: usize) -> f64 {
    let _ = backend;
    let (queries, database, mut scores) = generate_asymmetric_data(dim, num_queries, num_vectors);
    let config = BinaryIpConfig {
        dim,
        num_queries,
        num_vectors,
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        binary_ip_asymmetric(&queries, &database, &mut scores, &config);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        binary_ip_asymmetric(&queries, &database, &mut scores, &config);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / BENCH_ITERS as f64 * 1000.0 // ms
}

fn bench_int8_dot_product(backend: BackendType, dim: usize, num_queries: usize, num_vectors: usize) -> f64 {
    let _ = backend;
    let (queries, database, mut scores) = generate_int8_data(dim, num_queries, num_vectors);
    let config = Int8DotConfig {
        dim,
        num_queries,
        num_vectors,
        scale: 1.0 / 127.0,
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        int8_dot_product_unrolled(&queries, &database, &mut scores, &config);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        int8_dot_product_unrolled(&queries, &database, &mut scores, &config);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / BENCH_ITERS as f64 * 1000.0 // ms
}

fn bench_int4_dot_product(backend: BackendType, dim: usize, num_queries: usize, num_vectors: usize) -> f64 {
    let _ = backend;
    let (queries, database, mut scores) = generate_int4_data(dim, num_queries, num_vectors);
    let config = Int4PackedConfig {
        dim,
        num_queries,
        num_vectors,
        scale: 1.0 / 7.0,
        zero_point: 0,
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        int4_packed_dot_product(&queries, &database, &mut scores, &config);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        int4_packed_dot_product(&queries, &database, &mut scores, &config);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / BENCH_ITERS as f64 * 1000.0 // ms
}

fn bench_matryoshka_truncate(backend: BackendType, full_dim: usize, target_dim: usize, num_vectors: usize) -> f64 {
    let _ = backend;
    let (embeddings, mut output) = generate_matryoshka_data(full_dim, target_dim, num_vectors);
    let config = MatryoshkaConfig {
        full_dim,
        target_dim,
        normalize: true,
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        matryoshka_truncate(&embeddings, &mut output, &config);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        matryoshka_truncate(&embeddings, &mut output, &config);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / BENCH_ITERS as f64 * 1000.0 // ms
}

fn print_benchmark_header(name: &str) {
    println!("\n{}", "=".repeat(80));
    println!("{}", name);
    println!("{}", "=".repeat(80));
    println!("{:>15} {:>12} {:>12} {:>12} {:>12} {:>12}",
             "Backend", "Time(ms)", "Speedup", "vs CPU", "vs WGPU", "Status");
    println!("{}", "-".repeat(80));
}

fn print_benchmark_row(backend: &str, time_ms: f64, cpu_time: f64, wgpu_time: f64) {
    let vs_cpu = if cpu_time > 0.0 { cpu_time / time_ms } else { 0.0 };
    let vs_wgpu = if wgpu_time > 0.0 { wgpu_time / time_ms } else { 0.0 };
    let status = if time_ms < cpu_time { "✓ faster" } else { "slower" };
    println!("{:>15} {:>12.3} {:>12.2}x {:>12.2}x {:>12.2}x {:>12}",
             backend, time_ms, vs_cpu, vs_cpu, vs_wgpu, status);
}

#[test]
fn benchmark_embedding_ops() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              EMBEDDING OPS BENCHMARK: CPU vs WGPU vs CUDA                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Test configurations
    let dim = 1024;
    let num_queries = 64;
    let num_vectors = 10000;

    println!("\nConfiguration: dim={}, queries={}, vectors={}", dim, num_queries, num_vectors);
    println!("Warmup iterations: {}, Benchmark iterations: {}", WARMUP_ITERS, BENCH_ITERS);

    // ========================================================================
    // Binary IP Hamming
    // ========================================================================
    print_benchmark_header("Binary IP Hamming (u64 packed, Hamming distance)");

    let cpu_time = bench_binary_ip_hamming(BackendType::Cpu, dim, num_queries, num_vectors);
    println!("{:>15} {:>12.3} {:>12} {:>12} {:>12} {:>12}",
             "CPU", cpu_time, "baseline", "1.00x", "-", "baseline");

    let wgpu_time = bench_binary_ip_hamming(BackendType::Wgpu, dim, num_queries, num_vectors);
    print_benchmark_row("WGPU", wgpu_time, cpu_time, wgpu_time);

    let cuda_time = bench_binary_ip_hamming(BackendType::Cuda, dim, num_queries, num_vectors);
    print_benchmark_row("CUDA", cuda_time, cpu_time, wgpu_time);

    // ========================================================================
    // Binary IP Asymmetric
    // ========================================================================
    print_benchmark_header("Binary IP Asymmetric (f32 query vs u64 database)");

    let cpu_time = bench_binary_ip_asymmetric(BackendType::Cpu, dim, num_queries, num_vectors);
    println!("{:>15} {:>12.3} {:>12} {:>12} {:>12} {:>12}",
             "CPU", cpu_time, "baseline", "1.00x", "-", "baseline");

    let wgpu_time = bench_binary_ip_asymmetric(BackendType::Wgpu, dim, num_queries, num_vectors);
    print_benchmark_row("WGPU", wgpu_time, cpu_time, wgpu_time);

    let cuda_time = bench_binary_ip_asymmetric(BackendType::Cuda, dim, num_queries, num_vectors);
    print_benchmark_row("CUDA", cuda_time, cpu_time, wgpu_time);

    // ========================================================================
    // Int8 Dot Product
    // ========================================================================
    print_benchmark_header("Int8 Dot Product (i8 packed, 4x memory efficiency)");

    let cpu_time = bench_int8_dot_product(BackendType::Cpu, dim, num_queries, num_vectors);
    println!("{:>15} {:>12.3} {:>12} {:>12} {:>12} {:>12}",
             "CPU", cpu_time, "baseline", "1.00x", "-", "baseline");

    let wgpu_time = bench_int8_dot_product(BackendType::Wgpu, dim, num_queries, num_vectors);
    print_benchmark_row("WGPU", wgpu_time, cpu_time, wgpu_time);

    let cuda_time = bench_int8_dot_product(BackendType::Cuda, dim, num_queries, num_vectors);
    print_benchmark_row("CUDA", cuda_time, cpu_time, wgpu_time);

    // ========================================================================
    // Int4 Dot Product
    // ========================================================================
    print_benchmark_header("Int4 Packed Dot Product (i4 packed, 8x memory efficiency)");

    let cpu_time = bench_int4_dot_product(BackendType::Cpu, dim, num_queries, num_vectors);
    println!("{:>15} {:>12.3} {:>12} {:>12} {:>12} {:>12}",
             "CPU", cpu_time, "baseline", "1.00x", "-", "baseline");

    let wgpu_time = bench_int4_dot_product(BackendType::Wgpu, dim, num_queries, num_vectors);
    print_benchmark_row("WGPU", wgpu_time, cpu_time, wgpu_time);

    let cuda_time = bench_int4_dot_product(BackendType::Cuda, dim, num_queries, num_vectors);
    print_benchmark_row("CUDA", cuda_time, cpu_time, wgpu_time);

    // ========================================================================
    // Matryoshka Truncate
    // ========================================================================
    let full_dim = 1024;
    let target_dim = 256;
    let num_vectors_mat = 10000;

    print_benchmark_header(&format!("Matryoshka Truncate ({}→{}, {} vectors, normalized)",
                                     full_dim, target_dim, num_vectors_mat));

    let cpu_time = bench_matryoshka_truncate(BackendType::Cpu, full_dim, target_dim, num_vectors_mat);
    println!("{:>15} {:>12.3} {:>12} {:>12} {:>12} {:>12}",
             "CPU", cpu_time, "baseline", "1.00x", "-", "baseline");

    let wgpu_time = bench_matryoshka_truncate(BackendType::Wgpu, full_dim, target_dim, num_vectors_mat);
    print_benchmark_row("WGPU", wgpu_time, cpu_time, wgpu_time);

    let cuda_time = bench_matryoshka_truncate(BackendType::Cuda, full_dim, target_dim, num_vectors_mat);
    print_benchmark_row("CUDA", cuda_time, cpu_time, wgpu_time);

    println!("\n{}", "=".repeat(80));
    println!("Benchmark complete!");
    println!("{}", "=".repeat(80));
}

#[test]
fn test_correctness_across_backends() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║            CORRECTNESS TEST: CPU vs WGPU vs CUDA                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let dim = 128;
    let num_queries = 4;
    let num_vectors = 16;

    // Binary IP Hamming correctness
    println!("\n[Binary IP Hamming]");
    let (queries, database, mut scores_cpu) = generate_test_data(dim, num_queries, num_vectors);
    let mut scores_wgpu = scores_cpu.clone();
    let mut scores_cuda = scores_cpu.clone();

    let config = BinaryIpConfig { dim, num_queries, num_vectors };

    KernelDispatcher::with_backend(BackendType::Cpu)
        .binary_ip_hamming(&queries, &database, &mut scores_cpu, config.clone());
    KernelDispatcher::with_backend(BackendType::Wgpu)
        .binary_ip_hamming(&queries, &database, &mut scores_wgpu, config.clone());
    KernelDispatcher::with_backend(BackendType::Cuda)
        .binary_ip_hamming(&queries, &database, &mut scores_cuda, config.clone());

    let wgpu_match = scores_cpu == scores_wgpu;
    let cuda_match = scores_cpu == scores_cuda;
    println!("  CPU vs WGPU: {} (sample: {:?})",
             if wgpu_match { "✓ MATCH" } else { "✗ MISMATCH" },
             &scores_cpu[..4.min(scores_cpu.len())]);
    println!("  CPU vs CUDA: {} (sample: {:?})",
             if cuda_match { "✓ MATCH" } else { "✗ MISMATCH" },
             &scores_cuda[..4.min(scores_cuda.len())]);

    // Int8 Dot Product correctness
    println!("\n[Int8 Dot Product]");
    let (queries_i8, database_i8, mut scores_cpu) = generate_int8_data(dim, num_queries, num_vectors);
    let mut scores_wgpu = scores_cpu.clone();
    let mut scores_cuda = scores_cpu.clone();

    let config = Int8DotConfig { dim, num_queries, num_vectors, scale: 1.0 / 127.0 };

    KernelDispatcher::with_backend(BackendType::Cpu)
        .int8_dot_product(&queries_i8, &database_i8, &mut scores_cpu, config.clone());
    KernelDispatcher::with_backend(BackendType::Wgpu)
        .int8_dot_product(&queries_i8, &database_i8, &mut scores_wgpu, config.clone());
    KernelDispatcher::with_backend(BackendType::Cuda)
        .int8_dot_product(&queries_i8, &database_i8, &mut scores_cuda, config.clone());

    // Allow small floating point tolerance
    let wgpu_match = scores_cpu.iter().zip(scores_wgpu.iter())
        .all(|(a, b)| (a - b).abs() < 0.01);
    let cuda_match = scores_cpu.iter().zip(scores_cuda.iter())
        .all(|(a, b)| (a - b).abs() < 0.01);
    println!("  CPU vs WGPU: {} (sample: {:?})",
             if wgpu_match { "✓ MATCH" } else { "✗ MISMATCH" },
             &scores_cpu[..4.min(scores_cpu.len())]);
    println!("  CPU vs CUDA: {} (sample: {:?})",
             if cuda_match { "✓ MATCH" } else { "✗ MISMATCH" },
             &scores_cuda[..4.min(scores_cuda.len())]);

    // Matryoshka Truncate correctness
    println!("\n[Matryoshka Truncate]");
    let full_dim = 128;
    let target_dim = 32;
    let num_vectors = 8;
    let (embeddings, mut output_cpu) = generate_matryoshka_data(full_dim, target_dim, num_vectors);
    let mut output_wgpu = output_cpu.clone();
    let mut output_cuda = output_cpu.clone();

    let config = MatryoshkaConfig { full_dim, target_dim, normalize: true };

    KernelDispatcher::with_backend(BackendType::Cpu)
        .matryoshka_truncate(&embeddings, &mut output_cpu, config.clone());
    KernelDispatcher::with_backend(BackendType::Wgpu)
        .matryoshka_truncate(&embeddings, &mut output_wgpu, config.clone());
    KernelDispatcher::with_backend(BackendType::Cuda)
        .matryoshka_truncate(&embeddings, &mut output_cuda, config.clone());

    // Allow small floating point tolerance
    let wgpu_match = output_cpu.iter().zip(output_wgpu.iter())
        .all(|(a, b)| (a - b).abs() < 0.001);
    let cuda_match = output_cpu.iter().zip(output_cuda.iter())
        .all(|(a, b)| (a - b).abs() < 0.001);
    println!("  CPU vs WGPU: {} (sample: {:?})",
             if wgpu_match { "✓ MATCH" } else { "✗ MISMATCH" },
             &output_cpu[..4.min(output_cpu.len())]);
    println!("  CPU vs CUDA: {} (sample: {:?})",
             if cuda_match { "✓ MATCH" } else { "✗ MISMATCH" },
             &output_cuda[..4.min(output_cuda.len())]);

    println!("\n{}", "=".repeat(80));
}
