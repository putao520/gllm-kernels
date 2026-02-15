use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use rand::Rng;

fn benchmark_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");

    let sizes: &[(usize, usize, usize)] = &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        // Transformer-like shapes: (tokens, hidden, hidden)
        (128, 4096, 4096),
        (1, 4096, 4096),
    ];

    let kernels = CpuKernels::<f32>::new();
    let mut rng = rand::thread_rng();

    for &(m, n, k) in sizes {
        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));

        let a: Vec<f32> = (0..m * k).map(|_| rng.gen()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.gen()).collect();
        let bias: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
        let mut c_out = vec![0.0f32; m * n];

        // --- Regular gemm ---
        group.bench_function(&format!("gemm_{m}x{n}x{k}"), |bench| {
            bench.iter(|| {
                kernels.gemm(
                    black_box(&a), black_box(&b), black_box(&mut c_out),
                    black_box(m), black_box(n), black_box(k),
                )
            })
        });

        // --- Prepacked: pack_b + gemm_prepacked ---
        let packed_b = kernels.pack_b(&b, n, k);
        group.bench_function(&format!("gemm_prepacked_{m}x{n}x{k}"), |bench| {
            bench.iter(|| {
                kernels.gemm_prepacked(
                    black_box(&a), black_box(&packed_b), black_box(&mut c_out),
                    black_box(m), black_box(n), black_box(k),
                )
            })
        });

        // --- pack_b cost alone ---
        group.bench_function(&format!("pack_b_{m}x{n}x{k}"), |bench| {
            bench.iter(|| {
                black_box(kernels.pack_b(black_box(&b), black_box(n), black_box(k)));
            })
        });

        // --- Regular gemm_bias ---
        group.bench_function(&format!("gemm_bias_{m}x{n}x{k}"), |bench| {
            bench.iter(|| {
                kernels.gemm_bias(
                    black_box(&a), black_box(&b), black_box(&bias), black_box(&mut c_out),
                    black_box(m), black_box(n), black_box(k),
                )
            })
        });

        // --- Prepacked gemm_bias ---
        group.bench_function(&format!("gemm_bias_prepacked_{m}x{n}x{k}"), |bench| {
            bench.iter(|| {
                kernels.gemm_bias_prepacked(
                    black_box(&a), black_box(&packed_b), black_box(&bias), black_box(&mut c_out),
                    black_box(m), black_box(n), black_box(k),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_gemm_f32);
criterion_main!(benches);
