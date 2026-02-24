//! Benchmark JIT compilation latency for various graph configurations.
//!
//! REQ-COMPILER-016: JIT compile latency < 100ms per layer.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench jit_compile_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gllm_kernels::compiler::graph::{CompilerGraph, OpKind};
use gllm_kernels::compiler::ir::LayerIR;
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::types::{DType, ModelConfig};

/// Build a minimal graph with a single GEMM op.
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

/// Build a graph with GEMM followed by SiLU epilogue.
fn build_gemm_silu_graph(m: usize, n: usize, k: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let a = g.add_tensor("a", vec![m, k], dt);
    let b = g.add_tensor("b", vec![k, n], dt);
    let gemm_out = g.add_tensor("gemm_out", vec![m, n], dt);
    let silu_out = g.add_tensor("silu_out", vec![m, n], dt);
    g.inputs = vec![a, b];
    g.outputs = vec![silu_out];
    g.add_op(OpKind::Gemm { m, n, k }, vec![a, b], vec![gemm_out], "gemm");
    g.add_op(OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
    g
}

/// Build a 3-op elementwise chain: SiLU -> Add -> Mul.
fn build_elementwise_chain(n: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let input = g.add_tensor("input", vec![1, n], dt);
    let bias = g.add_tensor("bias", vec![1, n], dt);
    let scale = g.add_tensor("scale", vec![1, n], dt);
    let silu_out = g.add_tensor("silu_out", vec![1, n], dt);
    let add_out = g.add_tensor("add_out", vec![1, n], dt);
    let mul_out = g.add_tensor("mul_out", vec![1, n], dt);
    g.inputs = vec![input, bias, scale];
    g.outputs = vec![mul_out];
    g.add_op(OpKind::Silu, vec![input], vec![silu_out], "silu");
    g.add_op(OpKind::Add, vec![silu_out, bias], vec![add_out], "add");
    g.add_op(OpKind::Mul, vec![add_out, scale], vec![mul_out], "mul");
    g
}

fn bench_jit_gemm_standalone(c: &mut Criterion) {
    let graph = build_gemm_graph(8, 16, 12);
    c.bench_function("jit_gemm_standalone_8x16x12", |b| {
        b.iter(|| {
            let mut compiler = InferenceCompiler::new();
            let result = compiler.compile_graph(black_box(&graph));
            black_box(result).unwrap();
        })
    });
}

fn bench_jit_gemm_epilogue(c: &mut Criterion) {
    let graph = build_gemm_silu_graph(8, 16, 12);
    c.bench_function("jit_gemm_silu_epilogue_8x16x12", |b| {
        b.iter(|| {
            let mut compiler = InferenceCompiler::new();
            let result = compiler.compile_graph(black_box(&graph));
            black_box(result).unwrap();
        })
    });
}

fn bench_jit_elementwise_chain(c: &mut Criterion) {
    let graph = build_elementwise_chain(4096);
    c.bench_function("jit_elementwise_chain_3op_4096", |b| {
        b.iter(|| {
            let mut compiler = InferenceCompiler::new();
            let result = compiler.compile_graph(black_box(&graph));
            black_box(result).unwrap();
        })
    });
}

fn bench_jit_full_layer(c: &mut Criterion) {
    let config = ModelConfig::llama_7b();
    let ir = LayerIR::from_model_config(&config, 1);
    c.bench_function("jit_full_layer_llama7b", |b| {
        b.iter(|| {
            let mut compiler = InferenceCompiler::new();
            let result = compiler.compile_layer(black_box(&ir));
            black_box(result).unwrap();
        })
    });
}

criterion_group!(
    name = jit_compile;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(20);
    targets =
        bench_jit_gemm_standalone,
        bench_jit_gemm_epilogue,
        bench_jit_elementwise_chain,
        bench_jit_full_layer,
);
criterion_main!(jit_compile);
