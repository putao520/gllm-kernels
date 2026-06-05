use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use std::time::Instant;

fn bench_one(label: &str, m: usize, n: usize, k: usize, iters: usize) {
    let kernels = CpuKernels::<f32>::new();
    let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i % 53) as f32 * 0.01).collect();
    let mut c = vec![0.0f32; m * n];

    // Warmup
    kernels.gemm(&a, &b, &mut c, m, n, k);
    kernels.gemm(&a, &b, &mut c, m, n, k);

    // gemm
    let t = Instant::now();
    for _ in 0..iters {
        kernels.gemm(&a, &b, &mut c, m, n, k);
    }
    let elapsed = t.elapsed().as_secs_f64() / iters as f64;
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / elapsed / 1e9;
    println!("{label:>40} gemm     : {elapsed:>10.3}ms  {gflops:>6.1} GFLOPS",
             elapsed = elapsed * 1000.0);

    // gemm_bias
    let t = Instant::now();
    for _ in 0..iters {
        kernels.gemm_bias(&a, &b, &bias, &mut c, m, n, k);
    }
    let elapsed = t.elapsed().as_secs_f64() / iters as f64;
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / elapsed / 1e9;
    println!("{label:>40} gemm_bias: {elapsed:>10.3}ms  {gflops:>6.1} GFLOPS",
             elapsed = elapsed * 1000.0);
}

fn main() {
    println!("=== GEMM Benchmark ===\n");
    bench_one("256x256x256", 256, 256, 256, 100);
    println!();
    bench_one("512x512x512", 512, 512, 512, 20);
    println!();
    bench_one("1024x1024x1024", 1024, 1024, 1024, 5);
    println!();
    bench_one("128x4096x4096 (transformer)", 128, 4096, 4096, 5);
    println!();
    bench_one("1x4096x4096 (single token)", 1, 4096, 4096, 20);
}
