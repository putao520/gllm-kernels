//! GEMM 微内核性能基准测试
//!
//! 矩阵大小: 128×128 ~ 2048×2048
//! 对比: scalar_gemm vs JIT 编译延迟
//! 报告: GFLOPS (throughput = 2*M*N*K FLOPs)

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Duration;

#[path = "utils.rs"]
mod utils;

use gllm_kernels::compiler::graph::{CompilerGraph, OpKind};
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::types::DType;

const GEMM_SIZES: &[(usize, usize, usize)] = &[
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
];

fn build_gemm_graph(m: usize, n: usize, k: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let a = g.add_tensor("a", vec![m, k], dt);
    let b = g.add_tensor("b", vec![k, n], dt);
    let c = g.add_tensor("c", vec![m, n], dt);
    g.inputs = vec![a, b];
    g.outputs = vec![c];
    g.add_op(OpKind::Gemm { m, n, k }, vec![a, b], vec![c], "gemm");
    g
}

fn build_gemm_bias_graph(m: usize, n: usize, k: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let a = g.add_tensor("a", vec![m, k], dt);
    let b = g.add_tensor("b", vec![k, n], dt);
    let bias = g.add_tensor("bias", vec![n], dt);
    let c = g.add_tensor("c", vec![m, n], dt);
    g.inputs = vec![a, b, bias];
    g.outputs = vec![c];
    g.add_op(
        OpKind::GemmBias { m, n, k },
        vec![a, b, bias],
        vec![c],
        "gemm_bias",
    );
    g
}

/// scalar_gemm 直接调用基准 — 报告 GFLOPS
fn bench_scalar_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm/scalar");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    for &(m, n, k) in GEMM_SIZES {
        // 2048x2048 标量 GEMM 太慢，跳过
        if m * n * k > 512 * 512 * 512 {
            continue;
        }
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let a = utils::random_f32_vec(m * k);
        let b = utils::random_f32_vec(k * n);
        let mut c_buf = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("gemm", format!("{m}x{n}x{k}")),
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

/// scalar_gemm_bias 直接调用基准 — 报告 GFLOPS
fn bench_scalar_gemm_bias(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm/scalar_bias");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    for &(m, n, k) in GEMM_SIZES {
        if m * n * k > 512 * 512 * 512 {
            continue;
        }
        let flops = utils::gemm_flops(m, n, k) + (m * n) as u64;
        group.throughput(Throughput::Elements(flops));

        let a = utils::random_f32_vec(m * k);
        let b = utils::random_f32_vec(k * n);
        let bias = utils::random_f32_vec(n);
        let mut c_buf = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("gemm_bias", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    gllm_scalar_ops::blas::scalar_gemm_bias(
                        black_box(a.as_ptr()),
                        black_box(b.as_ptr()),
                        black_box(bias.as_ptr()),
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

/// JIT 编译 GEMM 图的延迟基准
fn bench_jit_compile_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm/jit_compile");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &(m, n, k) in GEMM_SIZES {
        let graph = build_gemm_graph(m, n, k);
        group.bench_with_input(
            BenchmarkId::new("gemm", format!("{m}x{n}x{k}")),
            &graph,
            |bench, graph| {
                bench.iter(|| {
                    let mut compiler = InferenceCompiler::new();
                    let result = compiler.compile_graph(black_box(graph));
                    black_box(result).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// JIT 编译 GEMM+bias 图的延迟基准
fn bench_jit_compile_gemm_bias(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm/jit_compile_bias");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &(m, n, k) in GEMM_SIZES {
        let graph = build_gemm_bias_graph(m, n, k);
        group.bench_with_input(
            BenchmarkId::new("gemm_bias", format!("{m}x{n}x{k}")),
            &graph,
            |bench, graph| {
                bench.iter(|| {
                    let mut compiler = InferenceCompiler::new();
                    let result = compiler.compile_graph(black_box(graph));
                    black_box(result).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = gemm_benches;
    config = Criterion::default();
    targets =
        bench_scalar_gemm,
        bench_scalar_gemm_bias,
        bench_jit_compile_gemm,
        bench_jit_compile_gemm_bias,
);
criterion_main!(gemm_benches);
