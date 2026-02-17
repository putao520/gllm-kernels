use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use std::time::Instant;

fn bench_gemm(m: usize, n: usize, k: usize, iters: usize) -> (f64, f64) {
    let kernels = CpuKernels::<f32>::new();
    let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect();
    let mut c = vec![0.0f32; m * n];

    // Warmup
    for _ in 0..5 {
        kernels.gemm(&a, &b, &mut c, m, n, k);
    }

    // Collect per-iteration times, take median
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        kernels.gemm(&a, &b, &mut c, m, n, k);
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[iters / 2];
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / median / 1e9;
    (median, gflops)
}

fn main() {
    let sizes: &[(usize, usize, usize, usize)] = &[
        (256, 256, 256, 200),
        (512, 512, 512, 50),
        (1024, 1024, 1024, 20),
        (2048, 2048, 2048, 5),
        (128, 4096, 4096, 10),
        (1, 4096, 4096, 50),
    ];

    println!("=== GEMM A/B Benchmark (median of N runs) ===\n");
    println!("{:>20}  {:>10}  {:>10}  {:>5}", "Size", "Time", "GFLOPS", "Iters");
    println!("{}", "-".repeat(52));
    for &(m, n, k, iters) in sizes {
        let label = format!("{}x{}x{}", m, n, k);
        let (secs, gflops) = bench_gemm(m, n, k, iters);
        println!("{:>20}  {:>8.3} ms  {:>7.1}  {:>5}", label, secs * 1000.0, gflops, iters);
    }
}
