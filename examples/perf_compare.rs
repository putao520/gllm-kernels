use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use std::time::Instant;

fn bench_gemm(label: &str, m: usize, n: usize, k: usize, warmup: usize, iters: usize) {
    let kernels = CpuKernels::<f32>::new();
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 89) as f32) * 0.01 - 0.5).collect();
    let mut c = vec![0.0f32; m * n];

    // Warmup
    for _ in 0..warmup {
        kernels.gemm(&a, &b, &mut c, m, n, k);
    }

    // Collect per-iteration times, take median
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        c.fill(0.0);
        let t = Instant::now();
        kernels.gemm(&a, &b, &mut c, m, n, k);
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[iters / 2];
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / median / 1e9;
    println!("{label:<45} {median_ms:>8.3}ms {gflops:>9.1}",
             median_ms = median * 1000.0);
}

fn bench_gemm_prepacked(label: &str, m: usize, n: usize, k: usize, warmup: usize, iters: usize) {
    let kernels = CpuKernels::<f32>::new();
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 89) as f32) * 0.01 - 0.5).collect();
    let mut c = vec![0.0f32; m * n];

    // Pre-pack B (amortized in real inference)
    let packed_b = kernels.pack_b(&b, n, k);

    // Warmup
    for _ in 0..warmup {
        kernels.gemm_prepacked(&a, &packed_b, &mut c, m, n, k);
    }

    // Collect per-iteration times, take median
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        c.fill(0.0);
        let t = Instant::now();
        kernels.gemm_prepacked(&a, &packed_b, &mut c, m, n, k);
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[iters / 2];
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / median / 1e9;
    println!("{label:<45} {median_ms:>8.3}ms {gflops:>9.1}",
             median_ms = median * 1000.0);
}

fn main() {
    // Print ISA detection
    let isa = gllm_kernels::cpu_kernels::get_isa_level();
    println!("Detected ISA: {:?}", isa);
    println!("Rayon threads: {}", rayon::current_num_threads());

    let shapes: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("M=1 K=4096 N=4096 (decode GEMV)",      1,    4096, 4096, 5, 50),
        ("M=32 K=4096 N=4096 (small batch)",      32,   4096, 4096, 5, 20),
        ("M=128 K=4096 N=11008 (FFN prefill)",    128,  4096, 11008, 3, 10),
        ("M=512 K=4096 N=4096 (medium prefill)",  512,  4096, 4096, 3, 10),
        ("M=2048 K=4096 N=4096 (large prefill)",  2048, 4096, 4096, 2, 5),
    ];

    println!("\n{}", "=".repeat(72));
    println!("  gllm-kernels SGEMM (unpacked B)");
    println!("{}", "=".repeat(72));
    println!("{:<45} {:>10} {:>10}", "Shape", "Time", "GFLOPS");
    println!("{}", "-".repeat(72));
    for &(name, m, n, k, warmup, iters) in shapes {
        bench_gemm(name, m, n, k, warmup, iters);
    }

    println!("\n{}", "=".repeat(72));
    println!("  gllm-kernels SGEMM (prepacked B)");
    println!("{}", "=".repeat(72));
    println!("{:<45} {:>10} {:>10}", "Shape", "Time", "GFLOPS");
    println!("{}", "-".repeat(72));
    for &(name, m, n, k, warmup, iters) in shapes {
        bench_gemm_prepacked(name, m, n, k, warmup, iters);
    }
}
