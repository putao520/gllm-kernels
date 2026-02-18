//! Comprehensive baseline performance audit for gllm-kernels.
//!
//! Measures all memory-bound and compute-bound operators, compares against
//! theoretical hardware limits, and produces an efficiency report.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench baseline_audit

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use rand::Rng;

fn rand_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

// ============================================================================
// Memory-bound operators: measure GB/s
// ============================================================================

fn bench_vec_dot(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_vec_dot");

    for &n in &[1024, 4096, 65536, 1048576] {
        let a = rand_vec(n);
        let b = rand_vec(n);
        // Bytes: read 2 vectors of n f32 = 2*n*4 bytes
        group.throughput(Throughput::Bytes((2 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(kernels.vec_dot(black_box(&a), black_box(&b))))
        });
    }
    group.finish();
}

fn bench_vec_add(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_vec_add");

    for &n in &[1024, 4096, 65536, 1048576] {
        let a = rand_vec(n);
        let b = rand_vec(n);
        let mut out = vec![0.0f32; n];
        // Bytes: read 2 + write 1 = 3*n*4 bytes
        group.throughput(Throughput::Bytes((3 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kernels.vec_add(black_box(&a), black_box(&b), black_box(&mut out)))
        });
    }
    group.finish();
}

fn bench_vec_axpy(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_vec_axpy");

    for &n in &[4096, 65536] {
        let x = rand_vec(n);
        let mut y = rand_vec(n);
        // Bytes: read x + read/write y = 3*n*4 bytes
        group.throughput(Throughput::Bytes((3 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kernels.vec_axpy(black_box(&mut y), black_box(2.5f32), black_box(&x)))
        });
    }
    group.finish();
}

fn bench_silu(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_silu");

    for &n in &[4096, 32000] {
        let input = rand_vec(n);
        let mut out = vec![0.0f32; n];
        // Bytes: read 1 + write 1 = 2*n*4
        group.throughput(Throughput::Bytes((2 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kernels.silu(black_box(&input), black_box(&mut out)))
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_softmax");

    for &n in &[1024, 4096, 32000] {
        let data = rand_vec(n);
        let mut out = vec![0.0f32; n];
        // Softmax: 2-pass (max + exp+sum, then scale). Read ~2x, write 1x.
        group.throughput(Throughput::Bytes((3 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kernels.softmax(black_box(&data), black_box(&mut out)))
        });
    }
    group.finish();
}

fn bench_exp(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_exp");

    for &n in &[4096, 32000] {
        let input: Vec<f32> = rand_vec(n).iter().map(|x| x * 5.0).collect(); // range [-5, 5]
        let mut out = vec![0.0f32; n];
        // Bytes: read 1 + write 1 = 2*n*4
        group.throughput(Throughput::Bytes((2 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kernels.exp(black_box(&input), black_box(&mut out)))
        });
    }
    group.finish();
}

fn bench_rms_norm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_rms_norm");

    for &n in &[1024, 4096, 8192] {
        let input = rand_vec(n);
        let weight = rand_vec(n);
        let mut out = vec![0.0f32; n];
        // RMS norm: read x (sum_sq pass) + read x + read w + write out = 4*n*4
        group.throughput(Throughput::Bytes((4 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                kernels.rms_norm(
                    black_box(&input), black_box(&weight),
                    black_box(&mut out), 1e-5,
                )
            })
        });
    }
    group.finish();
}

fn bench_rope(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_rope");

    // Single token decode: 1 position, 32 heads, head_dim=128
    // RoPE treats data as [seq_len, head_dim] where seq_len = data.len()/head_dim.
    // cos/sin must be [seq_len, half] where half = head_dim/2.
    {
        let num_heads = 32;
        let head_dim = 128;
        let half = head_dim / 2;
        let total = num_heads * head_dim;
        let seq_len = total / head_dim; // = num_heads = 32
        let mut qk = rand_vec(total);
        let cos = rand_vec(seq_len * half);
        let sin = rand_vec(seq_len * half);
        // Bytes: read/write qk + read cos + read sin = 3*total*4 + 2*seq_len*half*4
        let bytes = (3 * total + 2 * seq_len * half) * 4;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function("1tok_32h_d128", |bench| {
            bench.iter(|| {
                kernels.rope(
                    black_box(&mut qk),
                    black_box(&cos), black_box(&sin),
                    head_dim, false,
                )
            })
        });
    }

    // Prefill: 32 positions, 32 heads, head_dim=128
    {
        let seq_len = 32;
        let num_heads = 32;
        let head_dim = 128;
        let half = head_dim / 2;
        let total = seq_len * num_heads * head_dim;
        let effective_seq = total / head_dim; // = seq_len * num_heads = 1024
        let mut qk = rand_vec(total);
        let cos = rand_vec(effective_seq * half);
        let sin = rand_vec(effective_seq * half);
        // Bytes: read/write qk + read cos + read sin
        let bytes = (3 * total + 2 * effective_seq * half) * 4;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function("32tok_32h_d128", |bench| {
            bench.iter(|| {
                kernels.rope(
                    black_box(&mut qk),
                    black_box(&cos), black_box(&sin),
                    head_dim, false,
                )
            })
        });
    }
    group.finish();
}

fn bench_dequant_q4k(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_dequant_q4k");

    let block_bytes = std::mem::size_of::<gllm_kernels::quant::BlockQ4K>();

    for &num_elements in &[256, 4096] {
        let num_blocks = num_elements / 256;
        let weight = vec![0u8; num_blocks * block_bytes];
        let mut out = vec![0.0f32; num_elements];
        // Bytes: read weight + write f32 output
        let bytes = (num_blocks * block_bytes + num_elements * 4) as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::from_parameter(num_elements), &num_elements, |bench, _| {
            bench.iter(|| {
                for b in 0..num_blocks {
                    let blk = &weight[b * block_bytes..(b + 1) * block_bytes];
                    let dst = &mut out[b * 256..(b + 1) * 256];
                    kernels.dequant_q4_k(black_box(blk), black_box(dst));
                }
            })
        });
    }
    group.finish();
}

fn bench_dequant_q8k(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_dequant_q8k");

    let block_bytes = std::mem::size_of::<gllm_kernels::quant::BlockQ8K>();

    for &num_elements in &[256, 4096] {
        let num_blocks = num_elements / 256;
        let weight = vec![0u8; num_blocks * block_bytes];
        let mut out = vec![0.0f32; num_elements];
        let bytes = (num_blocks * block_bytes + num_elements * 4) as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::from_parameter(num_elements), &num_elements, |bench, _| {
            bench.iter(|| {
                for b in 0..num_blocks {
                    let blk = &weight[b * block_bytes..(b + 1) * block_bytes];
                    let dst = &mut out[b * 256..(b + 1) * 256];
                    kernels.dequant_q8_k(black_box(blk), black_box(dst));
                }
            })
        });
    }
    group.finish();
}

// ============================================================================
// Compute-bound operators: measure GFLOPS
// ============================================================================

fn bench_gemm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("compute_gemm");
    group.sample_size(10);

    for &size in &[128, 256, 512, 1024, 2048] {
        let (m, n, k) = (size, size, size);
        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let mut out = vec![0.0f32; m * n];
        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));
        group.bench_with_input(BenchmarkId::new("gemm", format!("{size}x{size}x{size}")), &size, |bench, _| {
            bench.iter(|| {
                kernels.gemm(black_box(&a), black_box(&b), black_box(&mut out), m, n, k)
            })
        });
    }
    group.finish();
}

fn bench_gemm_prepacked(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("compute_gemm_prepacked");
    group.sample_size(10);

    for &size in &[128, 256, 512, 1024, 2048] {
        let (m, n, k) = (size, size, size);
        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let packed_b = kernels.pack_b(&b, n, k);
        let mut out = vec![0.0f32; m * n];
        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));
        group.bench_with_input(BenchmarkId::new("prepacked", format!("{size}x{size}x{size}")), &size, |bench, _| {
            bench.iter(|| {
                kernels.gemm_prepacked(black_box(&a), black_box(&packed_b), black_box(&mut out), m, n, k)
            })
        });
    }
    group.finish();
}

fn bench_gemv(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("compute_gemv");

    // Typical LLM shapes: M=4096,K=4096 and M=4096,K=11008
    for &(m, n) in &[(4096, 4096), (4096, 11008)] {
        let mat = rand_vec(m * n);
        let x = rand_vec(n);
        let mut y = vec![0.0f32; m];
        let flops = (2 * m * n) as u64;
        group.throughput(Throughput::Elements(flops));
        group.bench_with_input(BenchmarkId::new("gemv", format!("{m}x{n}")), &(m, n), |bench, _| {
            bench.iter(|| {
                kernels.gemv(black_box(&mat), black_box(&x), black_box(&mut y), m, n)
            })
        });
    }
    group.finish();
}

fn bench_gemv_q4(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("compute_gemv_q4");
    group.sample_size(10);

    let n = 4096;
    let block_bytes = std::mem::size_of::<gllm_kernels::quant::BlockQ4K>();
    let blocks = n / 256;
    let weight = vec![0u8; blocks * block_bytes];
    let x = rand_vec(n);
    // FLOPS: 2*n (multiply-add per element)
    let flops = (2 * n) as u64;
    group.throughput(Throughput::Elements(flops));
    group.bench_function(BenchmarkId::new("q4k_dot", n), |bench| {
        bench.iter(|| {
            black_box(kernels.gemv_q4(black_box(&weight), black_box(&x), 1.0, n))
        })
    });
    group.finish();
}

fn bench_gemv_q8(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("compute_gemv_q8");
    group.sample_size(10);

    let n = 4096;
    let block_bytes = std::mem::size_of::<gllm_kernels::quant::BlockQ8K>();
    let blocks = n / 256;
    let weight_raw = vec![0i8; blocks * block_bytes];
    let weight = unsafe { std::slice::from_raw_parts(weight_raw.as_ptr() as *const i8, weight_raw.len()) };
    let x = rand_vec(n);
    let flops = (2 * n) as u64;
    group.throughput(Throughput::Elements(flops));
    group.bench_function(BenchmarkId::new("q8k_dot", n), |bench| {
        bench.iter(|| {
            black_box(kernels.gemv_q8(black_box(weight), black_box(&x), 1.0, n))
        })
    });
    group.finish();
}

// ============================================================================
// Additional: GELU, LayerNorm, SwiGLU
// ============================================================================

fn bench_gelu(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_gelu");

    for &n in &[4096, 32000] {
        let input = rand_vec(n);
        let mut out = vec![0.0f32; n];
        group.throughput(Throughput::Bytes((2 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kernels.gelu(black_box(&input), black_box(&mut out)))
        });
    }
    group.finish();
}

fn bench_swiglu(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_swiglu");

    for &n in &[4096, 11008] {
        let gate = rand_vec(n);
        let up = rand_vec(n);
        let mut out = vec![0.0f32; n];
        // Read gate + up, write out = 3*n*4
        group.throughput(Throughput::Bytes((3 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kernels.swiglu(black_box(&gate), black_box(&up), black_box(&mut out)))
        });
    }
    group.finish();
}

fn bench_layer_norm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("mem_layer_norm");

    for &n in &[1024, 4096, 8192] {
        let input = rand_vec(n);
        let gamma = rand_vec(n);
        let beta = rand_vec(n);
        let mut out = vec![0.0f32; n];
        // Read x (2 passes) + gamma + beta + write out = 5*n*4
        group.throughput(Throughput::Bytes((5 * n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                kernels.layer_norm(
                    black_box(&input), black_box(&gamma), black_box(&beta),
                    black_box(&mut out), 1e-5,
                )
            })
        });
    }
    group.finish();
}

// ============================================================================
// Standalone throughput measurement (bypasses criterion for quick summary)
// ============================================================================

fn bench_summary(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("_summary_report");
    group.sample_size(10);

    // We use a single dummy benchmark to print the summary at the end
    group.bench_function("print_report", |bench| {
        bench.iter(|| {
            // Minimal work to keep criterion happy
            black_box(1 + 1);
        })
    });
    group.finish();

    // ---- Print hardware summary ----
    let sep = "=".repeat(80);
    eprintln!("\n{sep}");
    eprintln!("  BASELINE PERFORMANCE AUDIT REPORT");
    eprintln!("{sep}");
    eprintln!("  CPU: Intel i9-10900KF @ 3.70 GHz (turbo ~4.9 GHz)");
    eprintln!("  Cores: 10 physical / 20 threads");
    eprintln!("  ISA: AVX2 + FMA (no AVX-512)");
    eprintln!("  Cache: L1D=32KB, L2=256KB, L3=20MB");
    eprintln!("  FP32 Peak (1 core, 2 FMA ports): 2 * 8 * 2 * 4.9 = ~156.8 GFLOPS");
    eprintln!("  FP32 Peak (10 cores): ~1568 GFLOPS");
    eprintln!("  Memory BW (DDR4-3200 dual channel): ~51.2 GB/s theoretical");
    eprintln!();
    eprintln!("  NOTE: Efficiency numbers are in the criterion output above.");
    eprintln!("  Look at 'throughput' column:");
    eprintln!("    - mem_* groups: GB/s (compare vs ~40 GB/s practical DRAM BW)");
    eprintln!("    - compute_* groups: elements/sec = FLOPS (compare vs 156.8 GFLOPS peak)");
    eprintln!("{sep}\n");
}

// ============================================================================
// Criterion groups
// ============================================================================

criterion_group!(
    name = memory_bound;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(1)).measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_vec_dot,
        bench_vec_add,
        bench_vec_axpy,
        bench_silu,
        bench_softmax,
        bench_exp,
        bench_rms_norm,
        bench_rope,
        bench_dequant_q4k,
        bench_dequant_q8k,
        bench_gelu,
        bench_swiglu,
        bench_layer_norm,
);

criterion_group!(
    name = compute_bound;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(2)).measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_gemm,
        bench_gemm_prepacked,
        bench_gemv,
        bench_gemv_q4,
        bench_gemv_q8,
);

criterion_group!(
    name = report;
    config = Criterion::default();
    targets = bench_summary,
);

criterion_main!(memory_bound, compute_bound, report);
