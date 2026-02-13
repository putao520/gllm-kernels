use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use rand::Rng;

fn rand_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

// ============================================================
// GEMM at multiple sizes
// ============================================================
fn bench_gemm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("gemm");

    for &size in &[512, 1024, 2048] {
        let (m, n, k) = (size, size, size);
        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let mut out = vec![0.0f32; m * n];
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));
        group.bench_function(format!("{size}x{size}x{size}"), |bench| {
            bench.iter(|| {
                kernels.gemm(black_box(&a), black_box(&b), black_box(&mut out), m, n, k);
            })
        });
    }
    group.finish();
}

// ============================================================
// Activations: SiLU
// ============================================================
fn bench_activations(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("activations");
    let n = 4096;
    let input = rand_vec(n);
    let mut out = vec![0.0f32; n];

    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("silu_4096", |bench| {
        bench.iter(|| kernels.silu(black_box(&input), black_box(&mut out)))
    });
    group.finish();
}

// ============================================================
// RMS Norm
// ============================================================
fn bench_rms_norm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("rms_norm");

    for &n in &[1024, 4096, 8192] {
        let input = rand_vec(n);
        let weight = rand_vec(n);
        let mut out = vec![0.0f32; n];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("{n}"), |bench| {
            bench.iter(|| {
                kernels.rms_norm(
                    black_box(&input), black_box(&weight),
                    black_box(&mut out), 1e-5,
                );
            })
        });
    }
    group.finish();
}

// PLACEHOLDER_MORE_BENCHES

// ============================================================
// Softmax
// ============================================================
fn bench_softmax(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("softmax");

    for &n in &[1024, 4096, 32000] {
        let data = rand_vec(n);
        let mut out = vec![0.0f32; n];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("{n}"), |bench| {
            bench.iter(|| kernels.softmax(black_box(&data), black_box(&mut out)))
        });
    }
    group.finish();
}

// ============================================================
// Quantized GEMV (Q4_K)
// ============================================================
fn bench_quant_gemv(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("quant_gemv_q4k");
    group.sample_size(10);

    let n = 4096;
    let block_size = 144; // Q4_K block: 144 bytes per 256 elements
    let blocks_per_row = n / 256;
    let weight_bytes = blocks_per_row * block_size;
    let weight = vec![0u8; weight_bytes];
    let x = rand_vec(n);

    group.throughput(Throughput::Elements((2 * n) as u64));
    group.bench_function(format!("dot_{n}"), |bench| {
        bench.iter(|| {
            black_box(kernels.gemv_q4(
                black_box(&weight), black_box(&x), 1.0, n,
            ));
        })
    });
    group.finish();
}

// ============================================================
// RoPE
// ============================================================
fn bench_rope(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("rope");

    let seq_len = 512;
    let num_heads = 32;
    let head_dim = 128;
    let total = seq_len * num_heads * head_dim;
    let mut qk = rand_vec(total);
    let cos = rand_vec(seq_len * head_dim);
    let sin = rand_vec(seq_len * head_dim);

    group.throughput(Throughput::Elements(total as u64));
    group.bench_function(
        format!("seq{seq_len}_h{num_heads}_d{head_dim}"),
        |bench| {
            bench.iter(|| {
                kernels.rope(
                    black_box(&mut qk),
                    black_box(&cos), black_box(&sin),
                    head_dim, false,
                );
            })
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_gemm,
    bench_gemv,
    bench_activations,
    bench_rms_norm,
    bench_softmax,
    bench_quant_gemv,
    bench_rope,
);
criterion_main!(benches);
fn bench_gemv(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("gemv");

    for &n in &[1024, 4096] {
        let m = n;
        let mat = rand_vec(m * n);
        let x = rand_vec(n);
        let mut out = vec![0.0f32; m];
        group.throughput(Throughput::Elements((2 * m * n) as u64));
        group.bench_function(format!("{m}x{n}"), |bench| {
            bench.iter(|| {
                kernels.gemv(black_box(&mat), black_box(&x), black_box(&mut out), m, n);
            })
        });
    }
    group.finish();
}
