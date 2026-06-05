//! 量化 matmul 性能基准测试
//!
//! 算子: Q8 GEMV/GEMM, Q4 GEMV/GEMM, K-Quant matmul
//! 对比: 量化 vs f32 标量
//! 报告: GFLOPS (throughput = 2*M*N*K FLOPs)

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Duration;

#[path = "utils.rs"]
mod utils;

// ─── GEMV 基准 (M=1 路径) ───

/// Q8 GEMV: 逐行 dot product
fn bench_gemv_q8(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant/gemv_q8");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let sizes: &[(usize, usize)] = &[
        (4096, 4096),     // LLaMA-7B hidden
        (4096, 11008),    // LLaMA-7B FFN
        (4096, 16384),    // 大 FFN
    ];

    for &(k, n) in sizes {
        // GEMV: M=1, 每行 2*K FLOPs, 共 N 行
        let flops = 2 * k as u64 * n as u64;
        group.throughput(Throughput::Elements(flops));

        let input = utils::random_f32_vec(k);
        let scales = utils::random_scale_vec(n);
        // 权重: [N, K] i8
        let weight = utils::random_i8_vec(n * k);
        let mut output = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("q8", format!("1x{n}x{k}")),
            &(k, n),
            |bench, &(k, n)| {
                bench.iter(|| {
                    for j in 0..n {
                        let row_ptr = unsafe { weight.as_ptr().add(j * k) };
                        let val = gllm_scalar_ops::quant_matmul::scalar_gemv_q8(
                            black_box(row_ptr),
                            black_box(input.as_ptr()),
                            black_box(scales[j]),
                            k,
                        );
                        output[j] = val;
                    }
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// Q4 GEMV: 4-bit packed 逐行 dot product
fn bench_gemv_q4(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant/gemv_q4");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let sizes: &[(usize, usize)] = &[
        (4096, 4096),
        (4096, 11008),
    ];

    for &(k, n) in sizes {
        let flops = 2 * k as u64 * n as u64;
        group.throughput(Throughput::Elements(flops));

        let input = utils::random_f32_vec(k);
        let scales = utils::random_scale_vec(n);
        // 权重: [N, K/2] u8 (4-bit packed)
        let weight = utils::random_u8_vec(n * k / 2);

        let mut output = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("q4", format!("1x{n}x{k}")),
            &(k, n),
            |bench, &(k, n)| {
                bench.iter(|| {
                    for j in 0..n {
                        let row_ptr = unsafe { weight.as_ptr().add(j * k / 2) };
                        let val = gllm_scalar_ops::quant_matmul::scalar_gemv_q4(
                            black_box(row_ptr),
                            black_box(input.as_ptr()),
                            black_box(scales[j]),
                            k,
                        );
                        output[j] = val;
                    }
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// F32 GEMV 基线 (用 scalar_gemm M=1)
fn bench_gemv_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant/gemv_f32_baseline");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let sizes: &[(usize, usize)] = &[
        (4096, 4096),
        (4096, 11008),
    ];

    for &(k, n) in sizes {
        let flops = 2 * k as u64 * n as u64;
        group.throughput(Throughput::Elements(flops));

        let input = utils::random_f32_vec(k);
        let weight = utils::random_f32_vec(k * n);
        let mut output = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("f32", format!("1x{n}x{k}")),
            &(k, n),
            |bench, &(k, n)| {
                bench.iter(|| {
                    output.fill(0.0);
                    gllm_scalar_ops::blas::scalar_gemm(
                        black_box(input.as_ptr()),
                        black_box(weight.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        1,
                        n,
                        k,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

// ─── GEMM 基准 (M>1) ───

/// Q8 GEMM: batched quantized matmul
fn bench_gemm_q8(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant/gemm_q8");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
    ];

    for &(m, n, k) in sizes {
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let input = utils::random_f32_vec(m * k);
        let weight = utils::random_i8_vec(k * n);
        let scales = utils::random_scale_vec(n);
        let mut output = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("q8", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    output.fill(0.0);
                    gllm_scalar_ops::quant_matmul::scalar_gemm_q8(
                        black_box(weight.as_ptr()),
                        black_box(input.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        black_box(scales.as_ptr()),
                        m,
                        n,
                        k,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// Q4 GEMM: 4-bit packed batched matmul
fn bench_gemm_q4(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant/gemm_q4");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
    ];

    for &(m, n, k) in sizes {
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let input = utils::random_f32_vec(m * k);
        // Q4: [K, N/2] packed
        let weight = utils::random_u8_vec(k * n / 2);
        let scales = utils::random_scale_vec(n);
        let mut output = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("q4", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    output.fill(0.0);
                    gllm_scalar_ops::quant_matmul::scalar_gemm_q4(
                        black_box(weight.as_ptr()),
                        black_box(input.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        black_box(scales.as_ptr()),
                        m,
                        n,
                        k,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// F32 GEMM 基线 (对比量化)
fn bench_gemm_f32_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant/gemm_f32_baseline");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[
        (32, 4096, 4096),
        (64, 4096, 4096),
    ];

    for &(m, n, k) in sizes {
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let a = utils::random_f32_vec(m * k);
        let b = utils::random_f32_vec(k * n);
        let mut c_buf = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("f32", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    c_buf.fill(0.0);
                    gllm_scalar_ops::blas::scalar_gemm(
                        black_box(a.as_ptr()),
                        black_box(b.as_ptr()),
                        black_box(c_buf.as_mut_ptr()),
                        m,
                        n,
                        k,
                    );
                    black_box(&c_buf);
                });
            },
        );
    }
    group.finish();
}

/// K-Quant matmul 基准
fn bench_kquant_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant/kquant_matmul");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    // K-Quant: block_size=256, 模拟 Q4K (block_bytes=144)
    let block_size: usize = 256;
    let block_bytes: usize = 144;

    let sizes: &[(usize, usize, usize)] = &[
        (1, 4096, 4096),
        (32, 4096, 4096),
    ];

    for &(m, n, k) in sizes {
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let input = utils::random_f32_vec(m * k);
        let blocks_per_row = k / block_size;
        let row_bytes = blocks_per_row * block_bytes;
        let weight_blocks = utils::random_u8_vec(n * row_bytes);
        let mut output = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("q4k", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    output.fill(0.0);
                    gllm_scalar_ops::quant_matmul::scalar_kquant_matmul(
                        black_box(weight_blocks.as_ptr()),
                        black_box(input.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        block_bytes,
                        block_size,
                        m,
                        n,
                        k,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = quant_benches;
    config = Criterion::default();
    targets =
        bench_gemv_q8,
        bench_gemv_q4,
        bench_gemv_f32,
        bench_gemm_q8,
        bench_gemm_q4,
        bench_gemm_f32_baseline,
        bench_kquant_matmul,
);
criterion_main!(quant_benches);
