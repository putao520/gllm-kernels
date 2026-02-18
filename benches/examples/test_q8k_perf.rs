use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::quant::{BlockQ8K, QuantType};
use gllm_kernels::KernelOps;
use std::time::Instant;

fn main() {
    let kernels = CpuKernels::<f32>::new();

    // Create test data: 4096x4096 matrix (Q8K) * 4096 vector (f32)
    let m = 4096;
    let k = 4096;

    // Q8K: 256 values per block, so we need k/256 blocks per row
    let blocks_per_row = k / 256;
    let total_blocks = m * blocks_per_row;

    // Create quantized matrix (A)
    let mut a_blocks = vec![BlockQ8K::default(); total_blocks];
    for block in &mut a_blocks {
        block.d = 0.5;
        for i in 0..256 {
            block.qs[i] = (i % 128) as i8 - 64;
        }
    }
    let a_bytes = unsafe {
        std::slice::from_raw_parts(
            a_blocks.as_ptr() as *const u8,
            total_blocks * std::mem::size_of::<BlockQ8K>(),
        )
    };

    // Create f32 vector (x)
    let x = vec![1.0f32; k];

    // Output vector (y)
    let mut y = vec![0.0f32; m];

    // Warmup
    println!("Warming up...");
    for _ in 0..10 {
        kernels.gemv(QuantType::Q8K, m, k, a_bytes, &x, &mut y);
    }

    // Benchmark
    println!("Running benchmark...");
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        kernels.gemv(QuantType::Q8K, m, k, a_bytes, &x, &mut y);
    }
    let elapsed = start.elapsed();

    let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let flops = 2.0 * m as f64 * k as f64; // 2 ops per multiply-add
    let gflops = flops / (avg_time_ms / 1000.0) / 1e9;

    println!("\n=== Q8K GEMV Performance ({}x{}) ===", m, k);
    println!("  Average time: {:.3} ms", avg_time_ms);
    println!("  Throughput: {:.2} GFLOPS", gflops);
    println!("  First result: {:.6}", y[0]);
}
