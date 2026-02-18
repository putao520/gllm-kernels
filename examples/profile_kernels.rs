//! Example: Profile gllm-kernels operators using the built-in profiler.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example profile_kernels
//!
//! Generates:
//!   /tmp/gllm-kernels-profile.json
//!   /tmp/gllm-kernels-profile.html

use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::profiling::{Profiler, counters};
use gllm_kernels::traits::Kernels;

fn rand_vec(n: usize) -> Vec<f32> {
    // Simple LCG — no rand dependency needed for examples
    let mut v = vec![0.0f32; n];
    let mut state = 0x12345678u64;
    for x in v.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
    }
    v
}

fn main() {
    let kernels = CpuKernels::<f32>::new();
    let mut profiler = Profiler::new();

    eprintln!("Hardware peak: {:.1} GFLOPS, {:.1} GB/s",
        profiler.hw_peak().gflops, profiler.hw_peak().bandwidth_gbs);
    eprintln!("HW counters available: {}", profiler.has_hw_counters());
    eprintln!();

    // ── GEMM (compute-bound) ────────────────────────────────────────
    for &(m, n, k) in &[(512, 512, 512), (1024, 1024, 1024), (128, 4096, 4096)] {
        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let mut c = vec![0.0f32; m * n];
        let name = format!("gemm_f32_{m}x{n}x{k}");
        let workload = counters::gemm_workload(m, n, k);

        // Warmup
        kernels.gemm(&a, &b, &mut c, m, n, k);

        profiler.measure_median(&name, workload, 5, || {
            kernels.gemm(&a, &b, &mut c, m, n, k);
        });
    }

    // ── GEMV (memory-bound) ─────────────────────────────────────────
    for &(m, n) in &[(4096, 4096), (1, 4096)] {
        let a = rand_vec(m * n);
        let x = rand_vec(n);
        let mut y = vec![0.0f32; m];
        let name = format!("gemv_f32_{m}x{n}");
        let workload = counters::gemv_workload(m, n);

        kernels.gemv(&a, &x, &mut y, m, n);

        profiler.measure_median(&name, workload, 5, || {
            kernels.gemv(&a, &x, &mut y, m, n);
        });
    }

    // ── Activations (memory-bound) ──────────────────────────────────
    for &n in &[4096, 16384, 65536] {
        let input = rand_vec(n);
        let mut out = vec![0.0f32; n];
        let name = format!("silu_f32_{n}");
        let workload = counters::elementwise_unary_workload(n, 4);

        kernels.silu(&input, &mut out);

        profiler.measure_median(&name, workload, 10, || {
            kernels.silu(&input, &mut out);
        });
    }

    // ── Softmax (memory-bound) ──────────────────────────────────────
    for &n in &[4096, 16384] {
        let input = rand_vec(n);
        let mut out = vec![0.0f32; n];
        let name = format!("softmax_f32_{n}");
        let workload = counters::softmax_workload(n, 4);

        kernels.softmax(&input, &mut out);

        profiler.measure_median(&name, workload, 10, || {
            kernels.softmax(&input, &mut out);
        });
    }

    // ── RMS Norm (memory-bound) ─────────────────────────────────────
    for &n in &[4096, 8192] {
        let input = rand_vec(n);
        let weight = rand_vec(n);
        let mut out = vec![0.0f32; n];
        let name = format!("rms_norm_f32_{n}");
        let workload = counters::rms_norm_workload(n, 4);

        kernels.rms_norm(&input, &weight, &mut out, 1e-5);

        profiler.measure_median(&name, workload, 10, || {
            kernels.rms_norm(&input, &weight, &mut out, 1e-5);
        });
    }

    // ── Dot product (mixed) ─────────────────────────────────────────
    for &n in &[4096, 65536] {
        let a = rand_vec(n);
        let b = rand_vec(n);
        let name = format!("vec_dot_f32_{n}");
        let workload = counters::dot_workload(n, 4);

        let _ = kernels.vec_dot(&a, &b);

        profiler.measure_median(&name, workload, 10, || {
            std::hint::black_box(kernels.vec_dot(&a, &b));
        });
    }

    // ── Print summary ───────────────────────────────────────────────
    profiler.print_summary();

    // ── Generate reports ────────────────────────────────────────────
    let report = profiler.report();

    let json_path = "/tmp/gllm-kernels-profile.json";
    let html_path = "/tmp/gllm-kernels-profile.html";

    report.write_json(json_path).expect("failed to write JSON report");
    report.write_html(html_path).expect("failed to write HTML report");

    eprintln!("Reports written:");
    eprintln!("  JSON: {json_path}");
    eprintln!("  HTML: {html_path}");
}
