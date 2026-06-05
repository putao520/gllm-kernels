//! Elementwise 算子性能基准测试
//!
//! 算子: SiLU, GELU, RmsNorm, Softmax
//! 向量大小: 1K, 4K, 16K, 64K, 256K
//! 对比: scalar 直接调用 vs JIT 编译延迟
//! 报告: 内存吞吐量 (Bytes throughput)

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Duration;

#[path = "utils.rs"]
mod utils;

use gllm_kernels::compiler::graph::{CompilerGraph, OpKind};
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::types::DType;

const ELEM_SIZES: &[usize] = &[1024, 4096, 16384, 65536, 262144];

fn size_label(n: usize) -> String {
    match n {
        1024 => "1K".into(),
        4096 => "4K".into(),
        16384 => "16K".into(),
        65536 => "64K".into(),
        262144 => "256K".into(),
        _ => format!("{n}"),
    }
}

fn build_silu_graph(n: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let inp = g.add_tensor("input", vec![1, n], DType::F32);
    let out = g.add_tensor("output", vec![1, n], DType::F32);
    g.inputs = vec![inp];
    g.outputs = vec![out];
    g.add_op(OpKind::Silu, vec![inp], vec![out], "silu");
    g
}

fn build_gelu_graph(n: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let inp = g.add_tensor("input", vec![1, n], DType::F32);
    let out = g.add_tensor("output", vec![1, n], DType::F32);
    g.inputs = vec![inp];
    g.outputs = vec![out];
    g.add_op(OpKind::Gelu, vec![inp], vec![out], "gelu");
    g
}

fn build_rmsnorm_graph(n: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let inp = g.add_tensor("input", vec![1, n], DType::F32);
    let weight = g.add_tensor("weight", vec![n], DType::F32);
    let out = g.add_tensor("output", vec![1, n], DType::F32);
    g.inputs = vec![inp, weight];
    g.outputs = vec![out];
    g.add_op(
        OpKind::RmsNorm { eps: 1e-5 },
        vec![inp, weight],
        vec![out],
        "rmsnorm",
    );
    g
}

fn build_softmax_graph(n: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let inp = g.add_tensor("input", vec![1, n], DType::F32);
    let out = g.add_tensor("output", vec![1, n], DType::F32);
    g.inputs = vec![inp];
    g.outputs = vec![out];
    g.add_op(OpKind::Softmax, vec![inp], vec![out], "softmax");
    g
}

/// SiLU 标量基准 — 报告内存吞吐量
fn bench_scalar_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise/scalar_silu");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n in ELEM_SIZES {
        group.throughput(Throughput::Bytes(utils::elementwise_rw_bytes(n)));
        let input = utils::random_f32_vec(n);
        let mut output = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("silu", size_label(n)),
            &n,
            |bench, &n| {
                bench.iter(|| {
                    gllm_scalar_ops::activations::scalar_silu(
                        black_box(input.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        n,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// GELU 标量基准 — 报告内存吞吐量
fn bench_scalar_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise/scalar_gelu");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n in ELEM_SIZES {
        group.throughput(Throughput::Bytes(utils::elementwise_rw_bytes(n)));
        let input = utils::random_f32_vec(n);
        let mut output = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("gelu", size_label(n)),
            &n,
            |bench, &n| {
                bench.iter(|| {
                    gllm_scalar_ops::activations::scalar_gelu(
                        black_box(input.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        n,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// RmsNorm 标量基准 — 报告内存吞吐量
fn bench_scalar_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise/scalar_rmsnorm");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n in ELEM_SIZES {
        group.throughput(Throughput::Bytes(utils::rmsnorm_rw_bytes(n)));
        let input = utils::random_f32_vec(n);
        let weight = utils::random_f32_vec(n);
        let mut output = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("rmsnorm", size_label(n)),
            &n,
            |bench, &n| {
                bench.iter(|| {
                    gllm_scalar_ops::norms::scalar_rms_norm(
                        black_box(input.as_ptr()),
                        black_box(weight.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        n,
                        1e-5,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// Softmax 标量基准 — 报告内存吞吐量
fn bench_scalar_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise/scalar_softmax");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n in ELEM_SIZES {
        group.throughput(Throughput::Bytes(utils::softmax_rw_bytes(n)));
        let input = utils::random_f32_vec(n);
        let mut output = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("softmax", size_label(n)),
            &n,
            |bench, &n| {
                bench.iter(|| {
                    gllm_scalar_ops::blas::scalar_softmax(
                        black_box(input.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        n,
                    );
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

/// JIT 编译 elementwise 图的延迟基准
fn bench_jit_compile_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise/jit_compile");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let representative_n = 4096;

    let silu_graph = build_silu_graph(representative_n);
    group.bench_function(BenchmarkId::new("silu", "4K"), |bench| {
        bench.iter(|| {
            let mut compiler = InferenceCompiler::new();
            black_box(compiler.compile_graph(black_box(&silu_graph)).unwrap());
        });
    });

    let gelu_graph = build_gelu_graph(representative_n);
    group.bench_function(BenchmarkId::new("gelu", "4K"), |bench| {
        bench.iter(|| {
            let mut compiler = InferenceCompiler::new();
            black_box(compiler.compile_graph(black_box(&gelu_graph)).unwrap());
        });
    });

    let rmsnorm_graph = build_rmsnorm_graph(representative_n);
    group.bench_function(BenchmarkId::new("rmsnorm", "4K"), |bench| {
        bench.iter(|| {
            let mut compiler = InferenceCompiler::new();
            black_box(compiler.compile_graph(black_box(&rmsnorm_graph)).unwrap());
        });
    });

    let softmax_graph = build_softmax_graph(representative_n);
    group.bench_function(BenchmarkId::new("softmax", "4K"), |bench| {
        bench.iter(|| {
            let mut compiler = InferenceCompiler::new();
            black_box(compiler.compile_graph(black_box(&softmax_graph)).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    name = elementwise_benches;
    config = Criterion::default();
    targets =
        bench_scalar_silu,
        bench_scalar_gelu,
        bench_scalar_rmsnorm,
        bench_scalar_softmax,
        bench_jit_compile_elementwise,
);
criterion_main!(elementwise_benches);
