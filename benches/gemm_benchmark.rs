use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use rand::Rng;

fn benchmark_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");

    let m = 1024;
    let n = 1024;
    let k = 1024;

    group.throughput(Throughput::Elements((2 * m * n * k) as u64));

    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..m * k).map(|_| rng.gen()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.gen()).collect();
    let mut c_out = vec![0.0f32; m * n];

    let kernels = CpuKernels::<f32>::new();

    group.bench_function("gemm_1024", |b_bench| {
        b_bench.iter(|| {
            kernels.gemm(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_out),
                black_box(m),
                black_box(n),
                black_box(k),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_gemm_f32);
criterion_main!(benches);
