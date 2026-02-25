//! Benchmark: JIT 编译器生成代码 vs 手写 asm 微内核性能对比。
//!
//! 对比维度：
//! - GEMM (compute-bound): 手写 asm 微内核 vs JIT Phase 3 生成代码
//! - Elementwise SiLU (memory-bound): intrinsics/宏生成 vs JIT 生成
//! - Fused GEMM+SiLU (融合收益): 分步调用 vs JIT 融合图
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench jit_vs_asm

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use gllm_kernels::compiler::graph::{CompilerGraph, OpKind};
use gllm_kernels::compiler::InferenceCompiler;
use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use gllm_kernels::types::DType;
use rand::Rng;

// ── 工具函数 ──────────────────────────────────────────────────────────

fn rand_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

// ── Graph 构建 ────────────────────────────────────────────────────────

/// 构建单个 GEMM 算子的 CompilerGraph。
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

/// 构建单个 SiLU 算子的 CompilerGraph。
fn build_silu_graph(n: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let input = g.add_tensor("input", vec![1, n], dt);
    let output = g.add_tensor("output", vec![1, n], dt);
    g.inputs = vec![input];
    g.outputs = vec![output];
    g.add_op(OpKind::Silu, vec![input], vec![output], "silu");
    g
}

/// 构建 GEMM → SiLU 融合图。
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

// ============================================================
// GEMM: JIT vs ASM (compute-bound)
// ============================================================
fn bench_gemm_jit_vs_asm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("jit_vs_asm_gemm");
    group.sample_size(10);

    for &(m, n, k) in &[
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));

        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let mut c_out = vec![0.0f32; m * n];

        // ── ASM 路径: CpuKernels::gemm ──
        group.bench_function(
            BenchmarkId::new("asm_gemm", format!("{m}x{n}x{k}")),
            |bench| {
                bench.iter(|| {
                    kernels.gemm(
                        black_box(&a), black_box(&b), black_box(&mut c_out),
                        m, n, k,
                    );
                })
            },
        );

        // ── ASM 路径: prepacked ──
        let packed_b = kernels.pack_b(&b, n, k);
        group.bench_function(
            BenchmarkId::new("asm_gemm_prepacked", format!("{m}x{n}x{k}")),
            |bench| {
                bench.iter(|| {
                    kernels.gemm_prepacked(
                        black_box(&a), black_box(&packed_b), black_box(&mut c_out),
                        m, n, k,
                    );
                })
            },
        );

        // ── JIT 路径: compile_graph → CompiledLayer::execute ──
        // 编译阶段（一次性开销，不计入热路径）
        let graph = build_gemm_graph(m, n, k);
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph);

        match compiled {
            Ok(layer) => {
                let scratchpad_bytes = layer.scratchpad_bytes;
                let mut scratchpad = vec![0u8; scratchpad_bytes.max(1)];

                // JIT 生成代码的执行性能
                // NOTE: CompiledLayerFn 签名是 transformer-layer 级别的，
                // 对于单算子图，input/weights/output 通过 raw pointer 传入。
                // 当前 stub 实现只是 ret，实际 JIT 代码（jit-x86 feature）
                // 会生成完整的 GEMM 微内核。
                group.bench_function(
                    BenchmarkId::new("jit_gemm", format!("{m}x{n}x{k}")),
                    |bench| {
                        bench.iter(|| unsafe {
                            layer.execute(
                                black_box(a.as_ptr() as *const u8),
                                black_box(b.as_ptr() as *const u8),
                                std::ptr::null_mut(),       // kv_cache (unused)
                                std::ptr::null(),           // positions (unused)
                                std::ptr::null(),           // seq_lens (unused)
                                1,                          // batch_size
                                1,                          // seq_len
                                black_box(c_out.as_mut_ptr() as *mut u8),
                                black_box(scratchpad.as_mut_ptr()),
                            );
                        })
                    },
                );
            }
            Err(e) => {
                // 编译失败时跳过 JIT benchmark，记录原因
                eprintln!(
                    "[jit_vs_asm] GEMM {m}x{n}x{k} JIT 编译失败，跳过: {e}"
                );
            }
        }
    }
    group.finish();
}

// ============================================================
// Elementwise SiLU: JIT vs ASM (memory-bound)
// ============================================================
fn bench_silu_jit_vs_asm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("jit_vs_asm_silu");

    for &n in &[4096, 16384, 65536] {
        // SiLU: 读 n 个 f32 + 写 n 个 f32 = 2*n*4 bytes
        group.throughput(Throughput::Bytes((2 * n * 4) as u64));

        let input = rand_vec(n);
        let mut output = vec![0.0f32; n];

        // ── ASM 路径 ──
        group.bench_function(
            BenchmarkId::new("asm_silu", n),
            |bench| {
                bench.iter(|| {
                    kernels.silu(black_box(&input), black_box(&mut output));
                })
            },
        );

        // ── JIT 路径 ──
        let graph = build_silu_graph(n);
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph);

        match compiled {
            Ok(layer) => {
                let mut scratchpad = vec![0u8; layer.scratchpad_bytes.max(1)];

                group.bench_function(
                    BenchmarkId::new("jit_silu", n),
                    |bench| {
                        bench.iter(|| unsafe {
                            layer.execute(
                                black_box(input.as_ptr() as *const u8),
                                std::ptr::null(),           // weights (unused)
                                std::ptr::null_mut(),       // kv_cache
                                std::ptr::null(),           // positions
                                std::ptr::null(),           // seq_lens
                                1, 1,
                                black_box(output.as_mut_ptr() as *mut u8),
                                black_box(scratchpad.as_mut_ptr()),
                            );
                        })
                    },
                );
            }
            Err(e) => {
                eprintln!("[jit_vs_asm] SiLU n={n} JIT 编译失败，跳过: {e}");
            }
        }
    }
    group.finish();
}

// ============================================================
// Fused GEMM+SiLU: JIT 融合 vs ASM 分步 (融合收益)
// ============================================================
fn bench_fused_gemm_silu(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("jit_vs_asm_fused_gemm_silu");
    group.sample_size(10);

    for &(m, n, k) in &[
        (128, 128, 128),
        (512, 512, 512),
    ] {
        // GEMM FLOPS + SiLU 带宽（以 GEMM FLOPS 为主）
        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));

        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let mut gemm_out = vec![0.0f32; m * n];
        let mut silu_out = vec![0.0f32; m * n];

        // ── ASM 分步路径: GEMM → SiLU（两次内存往返）──
        group.bench_function(
            BenchmarkId::new("asm_gemm_then_silu", format!("{m}x{n}x{k}")),
            |bench| {
                bench.iter(|| {
                    kernels.gemm(
                        black_box(&a), black_box(&b), black_box(&mut gemm_out),
                        m, n, k,
                    );
                    kernels.silu(black_box(&gemm_out), black_box(&mut silu_out));
                })
            },
        );

        // ── JIT 融合路径: GEMM+SiLU 在累加器上原地执行 ──
        let graph = build_gemm_silu_graph(m, n, k);
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph);

        match compiled {
            Ok(layer) => {
                let mut scratchpad = vec![0u8; layer.scratchpad_bytes.max(1)];

                group.bench_function(
                    BenchmarkId::new("jit_fused_gemm_silu", format!("{m}x{n}x{k}")),
                    |bench| {
                        bench.iter(|| unsafe {
                            layer.execute(
                                black_box(a.as_ptr() as *const u8),
                                black_box(b.as_ptr() as *const u8),
                                std::ptr::null_mut(),
                                std::ptr::null(),
                                std::ptr::null(),
                                1, 1,
                                black_box(silu_out.as_mut_ptr() as *mut u8),
                                black_box(scratchpad.as_mut_ptr()),
                            );
                        })
                    },
                );
            }
            Err(e) => {
                eprintln!(
                    "[jit_vs_asm] Fused GEMM+SiLU {m}x{n}x{k} JIT 编译失败，跳过: {e}"
                );
            }
        }
    }
    group.finish();
}

// ============================================================
// JIT 编译延迟（冷/热路径）
// ============================================================
fn bench_jit_compile_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile_latency");

    // GEMM 图编译延迟
    for &(m, n, k) in &[(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)] {
        let graph = build_gemm_graph(m, n, k);
        group.bench_function(
            BenchmarkId::new("gemm_compile", format!("{m}x{n}x{k}")),
            |bench| {
                bench.iter(|| {
                    let mut compiler = InferenceCompiler::new();
                    let result = compiler.compile_graph(black_box(&graph));
                    black_box(result).unwrap();
                })
            },
        );
    }

    // 融合 GEMM+SiLU 图编译延迟
    let fused_graph = build_gemm_silu_graph(512, 512, 512);
    group.bench_function("gemm_silu_fused_compile_512", |bench| {
        bench.iter(|| {
            let mut compiler = InferenceCompiler::new();
            let result = compiler.compile_graph(black_box(&fused_graph));
            black_box(result).unwrap();
        })
    });

    // SiLU 图编译延迟
    let silu_graph = build_silu_graph(16384);
    group.bench_function("silu_compile_16384", |bench| {
        bench.iter(|| {
            let mut compiler = InferenceCompiler::new();
            let result = compiler.compile_graph(black_box(&silu_graph));
            black_box(result).unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    name = jit_vs_asm;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_gemm_jit_vs_asm,
        bench_silu_jit_vs_asm,
        bench_fused_gemm_silu,
        bench_jit_compile_latency,
);
criterion_main!(jit_vs_asm);
