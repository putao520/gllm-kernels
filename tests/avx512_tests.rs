//! AVX-512 specific end-to-end validation tests.
//!
//! Each test runtime-detects AVX-512 support and skips gracefully on
//! hardware without it. When AVX-512 is available, tests compile operator
//! graphs with `use_avx512=true` (zmm registers, simd_width=16), execute
//! the JIT code, and verify numerical correctness against scalar references.

#![feature(f16)]
#![cfg(target_arch = "x86_64")]
#![cfg(feature = "jit-x86")]

use std::collections::HashMap;

use gllm_kernels::compiler::buffer_alloc::BufferAllocation;
use gllm_kernels::compiler::codegen::x86_64::jit::X86CodeGen;
use gllm_kernels::compiler::executable::CompiledLayer;
use gllm_kernels::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
use gllm_kernels::compiler::graph::{CompilerGraph, OpId, OpKind, TensorId};
use gllm_kernels::compiler::registry::ScalarOpRegistry;
use gllm_kernels::dispatch::device_profile::IsaLevel;
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::inference::types::DType;

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

macro_rules! skip_without_avx512 {
    () => {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512 not supported on this CPU, skipping");
            return;
        }
    };
}

/// Build a DeviceProfile with AVX-512 forced on.
fn avx512_profile() -> DeviceProfile {
    let mut profile = DeviceProfile::detect();
    profile.kernel_config.use_avx512 = true;
    profile.isa = IsaLevel::Avx512;
    profile
}

fn build_alloc(total_bytes: usize) -> BufferAllocation {
    BufferAllocation {
        slots: vec![],
        total_bytes,
        num_tensors: 0,
        bytes_saved: 0,
    }
}

fn compile_avx512(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    alloc: &BufferAllocation,
) -> CompiledLayer {
    let profile = avx512_profile();
    let mut codegen = X86CodeGen::new(&profile);
    let output = codegen.emit_plan(plan, graph, alloc, &profile, None).unwrap();
    assert!(!output.code.is_empty(), "codegen produced empty code");
    CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
}

fn compile_avx512_with_registry(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    alloc: &BufferAllocation,
    registry: &ScalarOpRegistry,
) -> CompiledLayer {
    let profile = avx512_profile();
    let mut codegen = X86CodeGen::new(&profile);
    let output = codegen
        .emit_plan(plan, graph, alloc, &profile, Some(registry))
        .unwrap();
    assert!(!output.code.is_empty(), "codegen produced empty code");
    CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
}

// ── Execution helpers ────────────────────────────────────────────────

unsafe fn exec_unary(layer: &CompiledLayer, input: &[f32], output: &mut [f32]) {
    let f = layer.entry_point();
    f(
        input.as_ptr() as *const u8,
        std::ptr::null(),
        std::ptr::null_mut(),
        std::ptr::null(),
        output.as_mut_ptr() as *const usize,
        0,
        0,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
}

unsafe fn exec_binary(
    layer: &CompiledLayer,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    let f = layer.entry_point();
    f(
        a.as_ptr() as *const u8,
        b.as_ptr() as *const u8,
        std::ptr::null_mut(),
        std::ptr::null(),
        output.as_mut_ptr() as *const usize,
        0,
        0,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
}

unsafe fn exec_gemm(layer: &CompiledLayer, a: &[f32], b: &[f32], c: &mut [f32]) {
    let mut scratch = vec![0u8; layer.scratchpad_bytes];
    let scratch_ptr = if layer.scratchpad_bytes > 0 {
        scratch.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let f = layer.entry_point();
    f(
        a.as_ptr() as *const u8,
        b.as_ptr() as *const u8,
        std::ptr::null_mut(),
        std::ptr::null(),
        c.as_mut_ptr() as *const usize,
        0,
        0,
        std::ptr::null_mut(),
        scratch_ptr,
    );
}

unsafe fn exec_norm_gemm(
    layer: &CompiledLayer,
    a: &[f32],
    b: &[f32],
    norm_w: &[f32],
    c: &mut [f32],
) {
    let mut scratch = vec![0u8; layer.scratchpad_bytes];
    let scratch_ptr = if layer.scratchpad_bytes > 0 {
        scratch.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let f = layer.entry_point();
    f(
        a.as_ptr() as *const u8,
        b.as_ptr() as *const u8,
        norm_w.as_ptr() as *mut u8,
        std::ptr::null(),
        c.as_mut_ptr() as *const usize,
        0,
        0,
        std::ptr::null_mut(),
        scratch_ptr,
    );
}

// ── Graph builders ───────────────────────────────────────────────────

fn build_unary_graph(n: usize, kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
    let mut g = CompilerGraph::new();
    let input = g.add_tensor("input", vec![n], DType::F32);
    let output = g.add_tensor("output", vec![n], DType::F32);
    g.inputs = vec![input];
    g.outputs = vec![output];
    let op_id = g.add_op(kind, vec![input], vec![output], "op");
    (g, op_id, output)
}

fn build_gemm_graph(m: usize, n: usize, k: usize) -> (CompilerGraph, OpId) {
    let mut g = CompilerGraph::new();
    let a = g.add_tensor("A", vec![m, k], DType::F32);
    let b = g.add_tensor("B", vec![k, n], DType::F32);
    let c = g.add_tensor("C", vec![m, n], DType::F32);
    g.inputs = vec![a, b];
    g.outputs = vec![c];
    let op_id = g.add_op(OpKind::Gemm { m, n, k }, vec![a, b], vec![c], "gemm");
    (g, op_id)
}

fn build_gemm_epilogue_graph(
    m: usize,
    n: usize,
    k: usize,
    epi_kind: OpKind,
) -> (CompilerGraph, OpId, OpId) {
    let mut g = CompilerGraph::new();
    let a = g.add_tensor("A", vec![m, k], DType::F32);
    let b = g.add_tensor("B", vec![k, n], DType::F32);
    let gemm_out = g.add_tensor("gemm_out", vec![m, n], DType::F32);
    let epi_out = g.add_tensor("epi_out", vec![m, n], DType::F32);
    g.inputs = vec![a, b];
    g.outputs = vec![epi_out];
    let gemm_id = g.add_op(
        OpKind::Gemm { m, n, k },
        vec![a, b],
        vec![gemm_out],
        "gemm",
    );
    let epi_id = g.add_op(epi_kind, vec![gemm_out], vec![epi_out], "epilogue");
    (g, gemm_id, epi_id)
}

fn build_norm_gemm_graph(
    m: usize,
    n: usize,
    k: usize,
    eps: f32,
) -> (CompilerGraph, OpId, OpId) {
    let mut g = CompilerGraph::new();
    let a = g.add_tensor("A", vec![m, k], DType::F32);
    let norm_w = g.add_tensor("norm_w", vec![k], DType::F32);
    let normed = g.add_tensor("normed", vec![m, k], DType::F32);
    let b = g.add_tensor("B", vec![k, n], DType::F32);
    let c = g.add_tensor("C", vec![m, n], DType::F32);
    g.inputs = vec![a, b, norm_w];
    g.outputs = vec![c];
    let _norm_id = g.add_op(
        OpKind::RmsNorm { eps },
        vec![a, norm_w],
        vec![normed],
        "rms_norm",
    );
    let gemm_id = g.add_op(
        OpKind::Gemm { m, n, k },
        vec![normed, b],
        vec![c],
        "gemm",
    );
    // Return norm_id properly
    let norm_id = OpId(gemm_id.0 - 1);
    (g, norm_id, gemm_id)
}

fn build_chain_graph(
    n: usize,
    anchor_kind: OpKind,
    epilogue_kind: OpKind,
) -> (CompilerGraph, OpId, OpId) {
    let mut g = CompilerGraph::new();
    let input = g.add_tensor("input", vec![n], DType::F32);
    let bias = g.add_tensor("bias", vec![n], DType::F32);
    let mid = g.add_tensor("mid", vec![n], DType::F32);
    let output = g.add_tensor("output", vec![n], DType::F32);
    g.inputs = vec![input, bias];
    g.outputs = vec![output];
    let anchor_id = g.add_op(anchor_kind, vec![input], vec![mid], "anchor");
    let epi_id = g.add_op(epilogue_kind, vec![mid, bias], vec![output], "epilogue");
    (g, anchor_id, epi_id)
}

// ── Plan builders ────────────────────────────────────────────────────

fn build_elementwise_plan(op_id: OpId) -> FusionPlan {
    let mut op_to_group = HashMap::new();
    op_to_group.insert(op_id, 0);
    FusionPlan {
        groups: vec![FusionGroup {
            id: 0,
            anchor: op_id,
            epilogue: vec![],
            mode: FusionMode::LoopFusion,
            ops: vec![op_id],
        }],
        op_to_group,
    }
}

fn build_standalone_plan(op_id: OpId) -> FusionPlan {
    let mut op_to_group = HashMap::new();
    op_to_group.insert(op_id, 0);
    FusionPlan {
        groups: vec![FusionGroup {
            id: 0,
            anchor: op_id,
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![op_id],
        }],
        op_to_group,
    }
}

fn build_gemm_epilogue_plan(gemm_id: OpId, epi_id: OpId) -> FusionPlan {
    let mut op_to_group = HashMap::new();
    op_to_group.insert(gemm_id, 0);
    op_to_group.insert(epi_id, 0);
    FusionPlan {
        groups: vec![FusionGroup {
            id: 0,
            anchor: gemm_id,
            epilogue: vec![epi_id],
            mode: FusionMode::EpilogueInjection,
            ops: vec![gemm_id, epi_id],
        }],
        op_to_group,
    }
}

fn build_norm_gemm_plan(norm_id: OpId, gemm_id: OpId) -> FusionPlan {
    let mut op_to_group = HashMap::new();
    op_to_group.insert(norm_id, 0);
    op_to_group.insert(gemm_id, 0);
    FusionPlan {
        groups: vec![FusionGroup {
            id: 0,
            anchor: gemm_id,
            epilogue: vec![],
            mode: FusionMode::NormIntoGemm,
            ops: vec![norm_id, gemm_id],
        }],
        op_to_group,
    }
}

fn build_chain_plan(anchor_id: OpId, epi_id: OpId) -> FusionPlan {
    let mut op_to_group = HashMap::new();
    op_to_group.insert(anchor_id, 0);
    op_to_group.insert(epi_id, 0);
    FusionPlan {
        groups: vec![FusionGroup {
            id: 0,
            anchor: anchor_id,
            epilogue: vec![epi_id],
            mode: FusionMode::LoopFusion,
            ops: vec![anchor_id, epi_id],
        }],
        op_to_group,
    }
}

// ── Reference implementations ────────────────────────────────────────

fn ref_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn ref_gelu(x: f32) -> f32 {
    let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn ref_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = sum as f32;
        }
    }
}

fn ref_rms_norm(a: &[f32], w: &[f32], out: &mut [f32], rows: usize, cols: usize, eps: f32) {
    for row in 0..rows {
        let off = row * cols;
        let mut sum_sq = 0.0f32;
        for j in 0..cols {
            sum_sq += a[off + j] * a[off + j];
        }
        let scale = 1.0 / (sum_sq / cols as f32 + eps).sqrt();
        for j in 0..cols {
            out[off + j] = a[off + j] * scale * w[j];
        }
    }
}

fn ref_rms_norm_gemm(
    a: &[f32], b: &[f32], norm_w: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize, eps: f32,
) {
    let mut normed = vec![0.0f32; m * k];
    ref_rms_norm(a, norm_w, &mut normed, m, k, eps);
    ref_matmul(&normed, b, c, m, n, k);
}

fn fill_matrix(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let len = rows * cols;
    let mut v = Vec::with_capacity(len);
    let mut s = seed;
    for _ in 0..len {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push((s as f32) / (u32::MAX as f32) * 2.0 - 1.0);
    }
    v
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut max_diff = 0.0f32;
    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < tol,
            "{} mismatch at [{}]: got={}, expected={}, diff={:.2e}",
            label, i, got[i], expected[i], diff,
        );
    }
    eprintln!("{}: n={} max_diff={:.2e} (tol={:.0e})", label, got.len(), max_diff, tol);
}

fn assert_matrix_close(got: &[f32], expected: &[f32], m: usize, n: usize, tol: f32, label: &str) {
    assert_eq!(got.len(), expected.len());
    let mut max_diff = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let diff = (got[idx] - expected[idx]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < tol,
                "{} mismatch at [{},{}]: got={}, expected={}, diff={:.2e}",
                label, i, j, got[idx], expected[idx], diff,
            );
        }
    }
    eprintln!("{}: {}x{} max_diff={:.2e} (tol={:.0e})", label, m, n, max_diff, tol);
}

// ═══════════════════════════════════════════════════════════════════════
// Test cases
// ═══════════════════════════════════════════════════════════════════════

/// a) SiLU elementwise with AVX-512 (zmm registers, 16 f32 per vector).
#[test]
fn test_avx512_silu_elementwise() {
    skip_without_avx512!();

    let n = 64; // 4 zmm vectors
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Silu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile_avx512(&graph, &plan, &alloc);

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 8.0).collect();
    let mut output = vec![0.0f32; n];
    let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();

    unsafe { exec_unary(&layer, &input, &mut output) };

    assert_close(&output, &expected, 1e-4, "AVX512-SiLU-64");
}

/// b) GELU elementwise with AVX-512.
#[test]
fn test_avx512_gelu_elementwise() {
    skip_without_avx512!();

    let n = 64;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Gelu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile_avx512(&graph, &plan, &alloc);

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 8.0).collect();
    let mut output = vec![0.0f32; n];
    let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();

    unsafe { exec_unary(&layer, &input, &mut output) };

    assert_close(&output, &expected, 1e-4, "AVX512-GELU-64");
}

/// c) RmsNorm via NormIntoGemm fusion with AVX-512.
///    (Standalone RmsNorm is not implemented; test via NormIntoGemm.)
#[test]
fn test_avx512_rmsnorm() {
    skip_without_avx512!();

    let (m, n, k) = (8, 32, 64);
    let eps = 1e-5;
    let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
    let plan = build_norm_gemm_plan(norm_id, gemm_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile_avx512(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 42);
    let b = fill_matrix(k, n, 137);
    let norm_w: Vec<f32> = (0..k).map(|i| 0.8 + 0.4 * (i as f32) / k as f32).collect();
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_rms_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
    unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

    let tol = k as f32 * 1e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "AVX512-NormIntoGemm-8x32x64");
}

/// d) Small GEMM (16x16x16) with AVX-512.
#[test]
fn test_avx512_gemm_small() {
    skip_without_avx512!();

    let (m, n, k) = (16, 16, 16);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile_avx512(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 1234);
    let b = fill_matrix(k, n, 5678);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    let tol = k as f32 * 1e-4;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "AVX512-GEMM-16x16x16");
}

/// e) LoopFusion: SiLU + Add fused into a single loop with AVX-512.
#[test]
fn test_avx512_loop_fusion() {
    skip_without_avx512!();

    let n = 64;
    let (graph, anchor_id, epi_id) = build_chain_graph(n, OpKind::Silu, OpKind::Add);
    let plan = build_chain_plan(anchor_id, epi_id);
    let alloc = build_alloc(n * 4);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_avx512_with_registry(&graph, &plan, &alloc, &registry);

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 8.0).collect();
    let bias: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();
    let mut output = vec![0.0f32; n];
    let expected: Vec<f32> = input.iter().zip(bias.iter())
        .map(|(&x, &b)| ref_silu(x) + b)
        .collect();

    unsafe { exec_binary(&layer, &input, &bias, &mut output) };

    assert_close(&output, &expected, 1e-4, "AVX512-LoopFusion-SiLU+Add-64");
}

/// f) EpilogueInjection: GEMM + SiLU epilogue with AVX-512.
#[test]
fn test_avx512_epilogue_injection() {
    skip_without_avx512!();

    let (m, n, k) = (16, 16, 16);
    let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
    let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
    let alloc = build_alloc(0);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_avx512_with_registry(&graph, &plan, &alloc, &registry);

    let a = fill_matrix(m, k, 42);
    let b = fill_matrix(k, n, 99);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    for x in c_ref.iter_mut() {
        *x = ref_silu(*x);
    }

    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    let tol = k as f32 * 5e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "AVX512-EpilogueInjection-GEMM+SiLU-16x16x16");
}
