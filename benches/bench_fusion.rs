//! 融合路径性能基准测试
//!
//! 对比: 融合 vs 非融合（标量逐算子执行 vs JIT 融合编译）
//! 场景:
//!   - LoopFusion: RmsNorm + SiLU 链
//!   - EpilogueInjection: GEMM + bias + SiLU
//!   - TileLevelFusion: RmsNorm → GEMM (大输出)
//!   - ComputeRoot: RmsNorm → GEMM (小输出)

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Duration;

#[path = "utils.rs"]
mod utils;

use gllm_kernels::compiler::graph::{CompilerGraph, OpKind};
use gllm_kernels::compiler::fusion::{self, FusionMode};
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::types::DType;

// ─── 图构建辅助函数 ───

/// LoopFusion 场景: RmsNorm → SiLU (elementwise 链)
fn build_rmsnorm_silu_graph(n: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let inp = g.add_tensor("input", vec![1, n], dt);
    let weight = g.add_tensor("weight", vec![n], dt);
    let norm_out = g.add_tensor("norm_out", vec![1, n], dt);
    let silu_out = g.add_tensor("silu_out", vec![1, n], dt);
    g.inputs = vec![inp, weight];
    g.outputs = vec![silu_out];
    g.add_op(
        OpKind::RmsNorm { eps: 1e-5 },
        vec![inp, weight],
        vec![norm_out],
        "rmsnorm",
    );
    g.add_op(OpKind::Silu, vec![norm_out], vec![silu_out], "silu");
    g
}

/// EpilogueInjection 场景: GEMM + SiLU
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

/// EpilogueInjection 场景: GEMM + bias + SiLU
fn build_gemm_bias_silu_graph(m: usize, n: usize, k: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let a = g.add_tensor("a", vec![m, k], dt);
    let b = g.add_tensor("b", vec![k, n], dt);
    let bias = g.add_tensor("bias", vec![n], dt);
    let gemm_out = g.add_tensor("gemm_out", vec![m, n], dt);
    let add_out = g.add_tensor("add_out", vec![m, n], dt);
    let silu_out = g.add_tensor("silu_out", vec![m, n], dt);
    g.inputs = vec![a, b, bias];
    g.outputs = vec![silu_out];
    g.add_op(OpKind::Gemm { m, n, k }, vec![a, b], vec![gemm_out], "gemm");
    g.add_op(OpKind::Add, vec![gemm_out, bias], vec![add_out], "add_bias");
    g.add_op(OpKind::Silu, vec![add_out], vec![silu_out], "silu");
    g
}

/// TileLevelFusion / ComputeRoot 场景: RmsNorm → GEMM
/// tile_level 当 norm 输出 > 75% L1 时触发，compute_root 当 <= 75% L1
fn build_rmsnorm_gemm_graph(m: usize, n: usize, k: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let inp = g.add_tensor("input", vec![m, k], dt);
    let weight = g.add_tensor("norm_weight", vec![k], dt);
    let norm_out = g.add_tensor("norm_out", vec![m, k], dt);
    let b = g.add_tensor("b", vec![k, n], dt);
    let gemm_out = g.add_tensor("gemm_out", vec![m, n], dt);
    g.inputs = vec![inp, weight, b];
    g.outputs = vec![gemm_out];
    g.add_op(
        OpKind::RmsNorm { eps: 1e-5 },
        vec![inp, weight],
        vec![norm_out],
        "rmsnorm",
    );
    g.add_op(
        OpKind::Gemm { m, n, k },
        vec![norm_out, b],
        vec![gemm_out],
        "gemm",
    );
    g
}

// ─── 非融合标量执行基准 ───

/// 非融合: RmsNorm + SiLU 分别执行
fn bench_unfused_rmsnorm_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/unfused_rmsnorm_silu");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let sizes: &[usize] = &[4096, 16384, 65536];
    for &n in sizes {
        // RmsNorm: 3n*4 bytes + SiLU: 2n*4 bytes = 5n*4 bytes total
        group.throughput(Throughput::Bytes(5 * n as u64 * 4));
        let input = utils::random_f32_vec(n);
        let weight = utils::random_f32_vec(n);
        let mut norm_out = vec![0.0f32; n];
        let mut silu_out = vec![0.0f32; n];

        group.bench_with_input(
            BenchmarkId::new("rmsnorm_then_silu", n),
            &n,
            |bench, &n| {
                bench.iter(|| {
                    gllm_scalar_ops::norms::scalar_rms_norm(
                        black_box(input.as_ptr()),
                        black_box(weight.as_ptr()),
                        black_box(norm_out.as_mut_ptr()),
                        n,
                        1e-5,
                    );
                    gllm_scalar_ops::activations::scalar_silu(
                        black_box(norm_out.as_ptr()),
                        black_box(silu_out.as_mut_ptr()),
                        n,
                    );
                    black_box(&silu_out);
                });
            },
        );
    }
    group.finish();
}

/// 非融合: GEMM + SiLU 分别执行
fn bench_unfused_gemm_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/unfused_gemm_silu");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[(128, 128, 128), (256, 256, 256)];
    for &(m, n, k) in sizes {
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let a = utils::random_f32_vec(m * k);
        let b = utils::random_f32_vec(k * n);
        let mut gemm_out = vec![0.0f32; m * n];
        let mut silu_out = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("gemm_then_silu", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    gemm_out.fill(0.0);
                    gllm_scalar_ops::blas::scalar_gemm(
                        black_box(a.as_ptr()),
                        black_box(b.as_ptr()),
                        black_box(gemm_out.as_mut_ptr()),
                        m,
                        n,
                        k,
                    );
                    gllm_scalar_ops::activations::scalar_silu(
                        black_box(gemm_out.as_ptr()),
                        black_box(silu_out.as_mut_ptr()),
                        m * n,
                    );
                    black_box(&silu_out);
                });
            },
        );
    }
    group.finish();
}

/// 非融合: GEMM + bias + SiLU 分别执行
fn bench_unfused_gemm_bias_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/unfused_gemm_bias_silu");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[(128, 128, 128), (256, 256, 256)];
    for &(m, n, k) in sizes {
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let a = utils::random_f32_vec(m * k);
        let b = utils::random_f32_vec(k * n);
        let bias = utils::random_f32_vec(n);
        let mut gemm_out = vec![0.0f32; m * n];
        let mut add_out = vec![0.0f32; m * n];
        let mut silu_out = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("gemm_bias_silu", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    gemm_out.fill(0.0);
                    gllm_scalar_ops::blas::scalar_gemm(
                        black_box(a.as_ptr()),
                        black_box(b.as_ptr()),
                        black_box(gemm_out.as_mut_ptr()),
                        m,
                        n,
                        k,
                    );
                    gllm_scalar_ops::blas::scalar_vec_add(
                        black_box(gemm_out.as_ptr()),
                        black_box(bias.as_ptr()),
                        black_box(add_out.as_mut_ptr()),
                        n,
                    );
                    gllm_scalar_ops::activations::scalar_silu(
                        black_box(add_out.as_ptr()),
                        black_box(silu_out.as_mut_ptr()),
                        m * n,
                    );
                    black_box(&silu_out);
                });
            },
        );
    }
    group.finish();
}

/// 非融合: RmsNorm + GEMM 分别执行
fn bench_unfused_rmsnorm_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/unfused_rmsnorm_gemm");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[(64, 128, 128), (128, 256, 256)];
    for &(m, n, k) in sizes {
        let flops = utils::gemm_flops(m, n, k);
        group.throughput(Throughput::Elements(flops));

        let input = utils::random_f32_vec(m * k);
        let weight = utils::random_f32_vec(k);
        let b = utils::random_f32_vec(k * n);
        let mut norm_out = vec![0.0f32; m * k];
        let mut gemm_out = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("rmsnorm_then_gemm", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    // RmsNorm 逐行
                    for row in 0..m {
                        gllm_scalar_ops::norms::scalar_rms_norm(
                            black_box(unsafe { input.as_ptr().add(row * k) }),
                            black_box(weight.as_ptr()),
                            black_box(unsafe { norm_out.as_mut_ptr().add(row * k) }),
                            k,
                            1e-5,
                        );
                    }
                    gemm_out.fill(0.0);
                    gllm_scalar_ops::blas::scalar_gemm(
                        black_box(norm_out.as_ptr()),
                        black_box(b.as_ptr()),
                        black_box(gemm_out.as_mut_ptr()),
                        m,
                        n,
                        k,
                    );
                    black_box(&gemm_out);
                });
            },
        );
    }
    group.finish();
}

// ─── JIT 融合编译基准 ───

/// JIT 编译融合图 — LoopFusion (RmsNorm + SiLU)
fn bench_fused_jit_compile_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/jit_loop_fusion");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[usize] = &[4096, 16384, 65536];
    for &n in sizes {
        let graph = build_rmsnorm_silu_graph(n);

        // 验证融合决策
        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);
        let has_loop_fusion = plan.groups.iter().any(|g| g.mode == FusionMode::LoopFusion);
        let has_any_fusion = plan.groups.iter().any(|g| g.mode != FusionMode::Standalone);

        group.bench_with_input(
            BenchmarkId::new(
                if has_loop_fusion {
                    "loop_fusion"
                } else if has_any_fusion {
                    "fused_other"
                } else {
                    "standalone"
                },
                n,
            ),
            &graph,
            |bench, graph| {
                bench.iter(|| {
                    let mut compiler = InferenceCompiler::new();
                    black_box(compiler.compile_graph(black_box(graph)).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// JIT 编译融合图 — EpilogueInjection (GEMM + SiLU)
fn bench_fused_jit_compile_epilogue(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/jit_epilogue_injection");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[(128, 128, 128), (256, 256, 256), (512, 512, 512)];
    for &(m, n, k) in sizes {
        let graph = build_gemm_silu_graph(m, n, k);

        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);
        let has_epilogue = plan
            .groups
            .iter()
            .any(|g| g.mode == FusionMode::EpilogueInjection);

        group.bench_with_input(
            BenchmarkId::new(
                if has_epilogue {
                    "epilogue"
                } else {
                    "standalone"
                },
                format!("{m}x{n}x{k}"),
            ),
            &graph,
            |bench, graph| {
                bench.iter(|| {
                    let mut compiler = InferenceCompiler::new();
                    black_box(compiler.compile_graph(black_box(graph)).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// JIT 编译融合图 — GEMM + bias + SiLU (3-op epilogue chain)
fn bench_fused_jit_compile_epilogue_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/jit_epilogue_chain");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[(usize, usize, usize)] = &[(128, 128, 128), (256, 256, 256)];
    for &(m, n, k) in sizes {
        let graph = build_gemm_bias_silu_graph(m, n, k);

        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);
        let fused_count = plan.num_fused_ops();

        group.bench_with_input(
            BenchmarkId::new(
                format!("fused_{fused_count}_ops"),
                format!("{m}x{n}x{k}"),
            ),
            &graph,
            |bench, graph| {
                bench.iter(|| {
                    let mut compiler = InferenceCompiler::new();
                    black_box(compiler.compile_graph(black_box(graph)).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// JIT 编译融合图 — TileLevelFusion / ComputeRoot (RmsNorm → GEMM)
fn bench_fused_jit_compile_tile(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/jit_tile_or_compute_root");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // 小输出 → ComputeRoot, 大输出 → TileLevelFusion
    let sizes: &[(usize, usize, usize)] = &[
        (64, 128, 128),   // 小: norm 输出 64*128*4 = 32KB, 可能 ComputeRoot
        (256, 512, 512),  // 大: norm 输出 256*512*4 = 512KB, 可能 TileLevelFusion
    ];

    for &(m, n, k) in sizes {
        let graph = build_rmsnorm_gemm_graph(m, n, k);

        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);
        let mode_label = plan
            .groups
            .iter()
            .find_map(|g| match &g.mode {
                FusionMode::TileLevelFusion { .. } => Some("tile_level"),
                FusionMode::ComputeRoot { .. } => Some("compute_root"),
                FusionMode::NormIntoGemm => Some("norm_into_gemm"),
                FusionMode::Standalone => None,
                _ => Some("other_fusion"),
            })
            .unwrap_or("standalone");

        group.bench_with_input(
            BenchmarkId::new(mode_label, format!("{m}x{n}x{k}")),
            &graph,
            |bench, graph| {
                bench.iter(|| {
                    let mut compiler = InferenceCompiler::new();
                    black_box(compiler.compile_graph(black_box(graph)).unwrap());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = fusion_benches;
    config = Criterion::default();
    targets =
        bench_unfused_rmsnorm_silu,
        bench_unfused_gemm_silu,
        bench_unfused_gemm_bias_silu,
        bench_unfused_rmsnorm_gemm,
        bench_fused_jit_compile_loop,
        bench_fused_jit_compile_epilogue,
        bench_fused_jit_compile_epilogue_chain,
        bench_fused_jit_compile_tile,
);
criterion_main!(fusion_benches);
