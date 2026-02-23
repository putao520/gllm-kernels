//! Comprehensive end-to-end validation tests for the x86_64 JIT compiler.
//!
//! These tests compile operator graphs into native code via `X86CodeGen`,
//! execute the generated machine code, and verify numerical correctness
//! against scalar reference implementations.
//!
//! Coverage:
//! - Elementwise ops: edge-case sizes, all unary/binary ops
//! - GEMM: degenerate shapes, non-aligned dims, large K accumulation
//! - Fused patterns: GEMM+GELU epilogue, NormIntoGemm edge cases
//! - Registry-driven chains: multi-op fusion, tail handling

#![feature(f16)]
#![cfg(target_arch = "x86_64")]
#![cfg(feature = "jit-x86")]

use std::collections::HashMap;

use gllm_kernels::compiler::buffer_alloc::BufferAllocation;
use gllm_kernels::compiler::codegen::x86_64::jit::X86CodeGen;
use gllm_kernels::compiler::executable::CompiledLayer;
use gllm_kernels::compiler::fusion::{FusionGroup, FusionPattern, FusionPlan};
use gllm_kernels::compiler::graph::{CompilerGraph, OpId, OpKind, TensorId};
use gllm_kernels::compiler::registry::ScalarOpRegistry;
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::inference::types::DType;

// ═══════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════

fn build_alloc(total_bytes: usize) -> BufferAllocation {
    BufferAllocation {
        slots: vec![],
        total_bytes,
        num_tensors: 0,
        bytes_saved: 0,
    }
}

fn compile(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    alloc: &BufferAllocation,
) -> CompiledLayer {
    let profile = DeviceProfile::detect();
    let mut codegen = X86CodeGen::new(&profile);
    let output = codegen.emit_plan(plan, graph, alloc, &profile, None).unwrap();
    assert!(!output.code.is_empty(), "codegen produced empty code");
    CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
}

fn compile_with_registry(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    alloc: &BufferAllocation,
    registry: &ScalarOpRegistry,
) -> CompiledLayer {
    let profile = DeviceProfile::detect();
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

fn build_binary_graph(n: usize, kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
    let mut g = CompilerGraph::new();
    let a = g.add_tensor("a", vec![n], DType::F32);
    let b = g.add_tensor("b", vec![n], DType::F32);
    let output = g.add_tensor("output", vec![n], DType::F32);
    g.inputs = vec![a, b];
    g.outputs = vec![output];
    let op_id = g.add_op(kind, vec![a, b], vec![output], "op");
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
    let norm_id = g.add_op(
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
    (g, norm_id, gemm_id)
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
            pattern: FusionPattern::ElementwiseChain,
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
            pattern: FusionPattern::Standalone,
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
            pattern: FusionPattern::GemmEpilogue,
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
            pattern: FusionPattern::NormIntoGemm,
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
            pattern: FusionPattern::ElementwiseChain,
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
    let inner =
        (2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn ref_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64; // f64 for reference accuracy
            for p in 0..k {
                sum += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = sum as f32;
        }
    }
}

fn ref_rms_norm_gemm(
    a: &[f32],
    b: &[f32],
    norm_w: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    eps: f32,
) {
    let mut normed = vec![0.0f32; m * k];
    for row in 0..m {
        let off = row * k;
        let mut sum_sq = 0.0f32;
        for j in 0..k {
            sum_sq += a[off + j] * a[off + j];
        }
        let scale = 1.0 / (sum_sq / k as f32 + eps).sqrt();
        for j in 0..k {
            normed[off + j] = a[off + j] * scale * norm_w[j];
        }
    }
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
                "{} mismatch at [{},{}]: got={}, expected={}, diff={}",
                label, i, j, got[idx], expected[idx], diff,
            );
        }
    }
    eprintln!(
        "{}: {}×{} max_diff={:.2e} (tol={:.0e})",
        label, m, n, max_diff, tol
    );
}

fn test_spread() -> Vec<f32> {
    vec![
        -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
    ]
}


// ═══════════════════════════════════════════════════════════════════════
// 1. GEMM edge-case shapes
// ═══════════════════════════════════════════════════════════════════════

/// 4×8×4 — small GEMM, single micro-tile (N=8 = one AVX2 vector).
#[test]
fn gemm_4x8x4() {
    let (m, n, k) = (4, 8, 4);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 1);
    let b = fill_matrix(k, n, 2);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, k as f32 * 1e-5, "GEMM-4x4x4");
}

/// Non-square: wide output (32×64×16).
#[test]
fn gemm_wide_output() {
    let (m, n, k) = (32, 64, 16);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 100);
    let b = fill_matrix(k, n, 200);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, k as f32 * 1e-4, "GEMM-32x64x16");
}

/// Non-square: tall output (64×16×32).
#[test]
fn gemm_tall_output() {
    let (m, n, k) = (64, 16, 32);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 300);
    let b = fill_matrix(k, n, 400);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, k as f32 * 1e-4, "GEMM-64x16x32");
}

/// K=1: degenerate inner dimension (outer product).
#[test]
fn gemm_k_equals_1() {
    let (m, n, k) = (8, 16, 1);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 500);
    let b = fill_matrix(k, n, 600);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, 1e-5, "GEMM-K=1");
}

/// M=1: matrix-vector product (single output row).
#[test]
fn gemm_m_equals_1() {
    let (m, n, k) = (1, 32, 16);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 700);
    let b = fill_matrix(k, n, 800);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, k as f32 * 1e-5, "GEMM-M=1");
}

/// N=16: narrow output (one NR=16 tile, minimum for BLIS micro-kernel).
#[test]
fn gemm_n_equals_16() {
    let (m, n, k) = (16, 16, 32);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 900);
    let b = fill_matrix(k, n, 1000);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, k as f32 * 1e-5, "GEMM-N=16");
}

/// 1×8×1: minimum GEMM — single row, one AVX2 vector width.
#[test]
fn gemm_1x8x1() {
    let (m, n, k) = (1, 8, 1);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = vec![2.5f32];
    let b = vec![3.0f32; 8];
    let mut c_jit = vec![0.0f32; 8];
    let mut c_ref = vec![0.0f32; 8];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, 1e-5, "GEMM-1x8x1");
}

/// Odd M and K that don't align to MR=6: 7×24×5 (N must be multiple of 8).
#[test]
fn gemm_odd_dims() {
    let (m, n, k) = (7, 24, 5);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 1111);
    let b = fill_matrix(k, n, 2222);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, k as f32 * 1e-4, "GEMM-7x24x5");
}

/// Prime M and K with N=24: 11×24×23 — M,K unaligned to tile sizes.
#[test]
fn gemm_prime_dims() {
    let (m, n, k) = (11, 24, 23);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 3333);
    let b = fill_matrix(k, n, 4444);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, k as f32 * 1e-4, "GEMM-11x24x23");
}

/// Large K accumulation accuracy: 12×16×2048.
/// Tests that FMA accumulation doesn't drift too far from f64 reference.
#[test]
fn gemm_large_k_accumulation() {
    let (m, n, k) = (12, 16, 2048);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 65535);
    let b = fill_matrix(k, n, 12321);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    // With k=2048 and values in [-1,1], expected magnitude ~sqrt(k)≈45.
    // Relative tolerance: k * eps ≈ 2048 * 1.2e-7 ≈ 2.5e-4.
    let tol = k as f32 * 2e-4;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-large-K-2048");
}

/// Stress: 100×200×300 — exercises all BLIS loops with remainders.
#[test]
fn gemm_stress_100x200x300() {
    let (m, n, k) = (100, 200, 300);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 99999);
    let b = fill_matrix(k, n, 88888);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    let tol = k as f32 * 1e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-100x200x300");
}


// ═══════════════════════════════════════════════════════════════════════
// 2. Elementwise edge cases
// ═══════════════════════════════════════════════════════════════════════

/// SiLU with exactly 8 elements (one AVX2 vector, no tail).
#[test]
fn silu_exact_vector_width() {
    let n = 8;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Silu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input: Vec<f32> = (-4..4).map(|i| i as f32 * 0.5).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    for i in 0..n {
        let expected = ref_silu(input[i]);
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-3, "SiLU-8 mismatch at [{}]: got={}, expected={}", i, output[i], expected);
    }
}

/// GELU with 16 elements: two full AVX2 vectors.
#[test]
fn gelu_two_vectors() {
    let n = 16;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Gelu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input: Vec<f32> = (0..n as i32).map(|i| (i as f32 - 4.0) * 0.7).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    for i in 0..n {
        let expected = ref_gelu(input[i]);
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-3, "GELU-16 mismatch at [{}]: got={}, expected={}", i, output[i], expected);
    }
}

/// SiLU with 256 elements — multiple loop iterations.
#[test]
fn silu_large_256() {
    let n = 256;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Silu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input: Vec<f32> = (0..n).map(|i| ((i as f32) - 128.0) / 32.0).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    let mut max_diff = 0.0f32;
    for i in 0..n {
        let expected = ref_silu(input[i]);
        let diff = (output[i] - expected).abs();
        max_diff = max_diff.max(diff);
        assert!(diff < 1e-3, "SiLU-256 mismatch at [{}]", i);
    }
    eprintln!("SiLU-256: max_diff={:.2e}", max_diff);
}

/// Add with 16 elements — one full AVX2 vector pair.
#[test]
fn add_aligned_16() {
    let n = 16;
    let (graph, op_id, _) = build_binary_graph(n, OpKind::Add);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 2.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.2).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_binary(&layer, &a, &b, &mut output) };

    for i in 0..n {
        let expected = a[i] + b[i];
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-5, "Add-16 mismatch at [{}]", i);
    }
}

/// Mul with 24 elements — 3 full AVX2 vectors.
#[test]
fn mul_aligned_24() {
    let n = 24;
    let (graph, op_id, _) = build_binary_graph(n, OpKind::Mul);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 4.0).collect();
    let b: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32) + 0.5).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_binary(&layer, &a, &b, &mut output) };

    for i in 0..n {
        let expected = a[i] * b[i];
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-4, "Mul-24 mismatch at [{}]", i);
    }
}

/// SiLU with extreme values: large positive/negative inputs.
#[test]
fn silu_extreme_values() {
    let n = 16;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Silu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input: Vec<f32> = vec![
        -100.0, -50.0, -20.0, -10.0, -5.0, -1.0, -0.001, 0.0,
        0.001, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 0.5,
    ];
    let mut output = vec![0.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    for i in 0..n {
        let expected = ref_silu(input[i]);
        let diff = (output[i] - expected).abs();
        // For large |x|, SiLU ≈ x (positive) or ≈ 0 (negative).
        // Allow relative tolerance for large values.
        let tol = if expected.abs() > 1.0 {
            expected.abs() * 1e-2
        } else {
            1e-3
        };
        assert!(
            diff < tol,
            "SiLU-extreme at [{}]: input={}, got={}, expected={}, diff={}",
            i, input[i], output[i], expected, diff
        );
    }
}

/// GELU with extreme values.
#[test]
fn gelu_extreme_values() {
    let n = 16;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Gelu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input: Vec<f32> = vec![
        -100.0, -50.0, -20.0, -10.0, -5.0, -1.0, -0.001, 0.0,
        0.001, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 0.5,
    ];
    let mut output = vec![0.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    for i in 0..n {
        let expected = ref_gelu(input[i]);
        let diff = (output[i] - expected).abs();
        let tol = if expected.abs() > 1.0 {
            expected.abs() * 1e-2
        } else {
            1e-3
        };
        assert!(
            diff < tol,
            "GELU-extreme at [{}]: input={}, got={}, expected={}, diff={}",
            i, input[i], output[i], expected, diff
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. GEMM + GELU epilogue
// ═══════════════════════════════════════════════════════════════════════

fn apply_gelu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = ref_gelu(*x);
    }
}

fn apply_silu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = ref_silu(*x);
    }
}

/// GEMM + GELU epilogue: small tile.
#[test]
fn gemm_gelu_epilogue_small() {
    let (m, n, k) = (4, 8, 4);
    let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Gelu);
    let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
    let alloc = build_alloc(0);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let a = fill_matrix(m, k, 42);
    let b = fill_matrix(k, n, 99);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    apply_gelu_inplace(&mut c_ref);

    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    assert_matrix_close(&c_jit, &c_ref, m, n, 5e-3, "GEMM+GELU-small");
}

/// GEMM + GELU epilogue: medium with remainders.
#[test]
fn gemm_gelu_epilogue_medium() {
    let (m, n, k) = (10, 24, 16);
    let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Gelu);
    let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
    let alloc = build_alloc(0);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let a = fill_matrix(m, k, 7);
    let b = fill_matrix(k, n, 13);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    apply_gelu_inplace(&mut c_ref);

    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    let tol = k as f32 * 5e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+GELU-medium");
}

/// GEMM + GELU epilogue: BLIS-scale.
#[test]
fn gemm_gelu_epilogue_blis() {
    let (m, n, k) = (64, 64, 64);
    let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Gelu);
    let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
    let alloc = build_alloc(0);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let a = fill_matrix(m, k, 31);
    let b = fill_matrix(k, n, 59);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    apply_gelu_inplace(&mut c_ref);

    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    let tol = k as f32 * 5e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+GELU-blis");
}

/// GEMM + SiLU epilogue with odd dimensions.
#[test]
fn gemm_silu_epilogue_odd_dims() {
    let (m, n, k) = (7, 24, 9);
    let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
    let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
    let alloc = build_alloc(0);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let a = fill_matrix(m, k, 555);
    let b = fill_matrix(k, n, 666);
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_matmul(&a, &b, &mut c_ref, m, n, k);
    apply_silu_inplace(&mut c_ref);

    unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

    let tol = k as f32 * 5e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+SiLU-odd-7x24x9");
}


// ═══════════════════════════════════════════════════════════════════════
// 4. NormIntoGemm edge cases
// ═══════════════════════════════════════════════════════════════════════

/// NormIntoGemm with M=1 (single row — single norm + matvec).
#[test]
fn norm_gemm_single_row() {
    let (m, n, k) = (1, 16, 32);
    let eps = 1e-5;
    let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
    let plan = build_norm_gemm_plan(norm_id, gemm_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 42);
    let b = fill_matrix(k, n, 137);
    let norm_w: Vec<f32> = (0..k).map(|i| 0.8 + 0.4 * (i as f32) / k as f32).collect();
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_rms_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
    unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

    let tol = k as f32 * 1e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormGemm-M=1");
}

/// NormIntoGemm with odd M and K (7×24×19, N must be multiple of 8).
#[test]
fn norm_gemm_odd_dims() {
    let (m, n, k) = (7, 24, 19);
    let eps = 1e-5;
    let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
    let plan = build_norm_gemm_plan(norm_id, gemm_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 777);
    let b = fill_matrix(k, n, 888);
    let norm_w: Vec<f32> = (0..k).map(|i| 1.0 + 0.1 * (i as f32 / k as f32 - 0.5)).collect();
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_rms_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
    unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

    let tol = k as f32 * 1e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormGemm-7x24x19");
}

/// NormIntoGemm with large K (256) — tests norm reduction accuracy.
#[test]
fn norm_gemm_large_k() {
    let (m, n, k) = (8, 32, 256);
    let eps = 1e-5;
    let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
    let plan = build_norm_gemm_plan(norm_id, gemm_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 271828);
    let b = fill_matrix(k, n, 314159);
    let norm_w: Vec<f32> = (0..k).map(|i| 0.5 + 1.0 * (i as f32) / k as f32).collect();
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_rms_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
    unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

    let tol = k as f32 * 1e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormGemm-large-K");
}

/// NormIntoGemm with uniform weights (all 1.0) — norm should be pure RMS scaling.
#[test]
fn norm_gemm_uniform_weights() {
    let (m, n, k) = (4, 8, 16);
    let eps = 1e-5;
    let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
    let plan = build_norm_gemm_plan(norm_id, gemm_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 42);
    let b = fill_matrix(k, n, 99);
    let norm_w = vec![1.0f32; k]; // uniform weights
    let mut c_jit = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    ref_rms_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
    unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

    let tol = k as f32 * 1e-3;
    assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormGemm-uniform-w");
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Registry-driven elementwise chains
// ═══════════════════════════════════════════════════════════════════════

fn build_chain_graph(
    n: usize,
    anchor_kind: OpKind,
    epilogue_kind: OpKind,
) -> (CompilerGraph, OpId, OpId, TensorId) {
    let mut g = CompilerGraph::new();
    let input = g.add_tensor("input", vec![n], DType::F32);
    let bias = g.add_tensor("bias", vec![n], DType::F32);
    let mid = g.add_tensor("mid", vec![n], DType::F32);
    let output = g.add_tensor("output", vec![n], DType::F32);
    g.inputs = vec![input, bias];
    g.outputs = vec![output];
    let anchor_id = g.add_op(anchor_kind, vec![input], vec![mid], "anchor");
    let epi_id = g.add_op(epilogue_kind, vec![mid, bias], vec![output], "epilogue");
    (g, anchor_id, epi_id, output)
}

/// GELU → Add chain.
#[test]
fn registry_gelu_add_chain() {
    let n = 32;
    let (graph, anchor_id, epi_id, _) = build_chain_graph(n, OpKind::Gelu, OpKind::Add);
    let plan = build_chain_plan(anchor_id, epi_id);
    let alloc = build_alloc(n * 4);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let input = test_spread().into_iter().cycle().take(n).collect::<Vec<_>>();
    let bias: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_binary(&layer, &input, &bias, &mut output) };

    for i in 0..n {
        let expected = ref_gelu(input[i]) + bias[i];
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-3, "GELU+Add at [{}]: got={}, expected={}", i, output[i], expected);
    }
}

/// GELU → Mul chain.
#[test]
fn registry_gelu_mul_chain() {
    let n = 32;
    let (graph, anchor_id, epi_id, _) = build_chain_graph(n, OpKind::Gelu, OpKind::Mul);
    let plan = build_chain_plan(anchor_id, epi_id);
    let alloc = build_alloc(n * 4);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let input = test_spread().into_iter().cycle().take(n).collect::<Vec<_>>();
    let scale: Vec<f32> = (0..n).map(|i| 0.5 + 0.05 * i as f32).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_binary(&layer, &input, &scale, &mut output) };

    for i in 0..n {
        let expected = ref_gelu(input[i]) * scale[i];
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-3, "GELU+Mul at [{}]: got={}, expected={}", i, output[i], expected);
    }
}

/// SiLU → Add chain with tail (n=25: 3 full vectors + 1 tail).
#[test]
fn registry_silu_add_chain_with_tail() {
    let n = 25;
    let (graph, anchor_id, epi_id, _) = build_chain_graph(n, OpKind::Silu, OpKind::Add);
    let plan = build_chain_plan(anchor_id, epi_id);
    let alloc = build_alloc(n * 4);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.3).collect();
    let bias: Vec<f32> = (0..n).map(|i| 0.2 * i as f32 - 2.0).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_binary(&layer, &input, &bias, &mut output) };

    for i in 0..n {
        let expected = ref_silu(input[i]) + bias[i];
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-3, "SiLU+Add-tail at [{}]: got={}, expected={}", i, output[i], expected);
    }
}

/// GELU with tail (n=11).
#[test]
fn registry_gelu_with_tail() {
    let n = 11;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Gelu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let registry = ScalarOpRegistry::with_defaults();
    let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 5.0) * 0.8).collect();
    let mut output = vec![0.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    for i in 0..n {
        let expected = ref_gelu(input[i]);
        let diff = (output[i] - expected).abs();
        assert!(diff < 1e-3, "GELU-tail-11 at [{}]: got={}, expected={}", i, output[i], expected);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Determinism: same inputs → same outputs across multiple runs
// ═══════════════════════════════════════════════════════════════════════

/// Verify GEMM produces bit-identical results across 5 invocations.
#[test]
fn gemm_deterministic() {
    let (m, n, k) = (16, 32, 24);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 42);
    let b = fill_matrix(k, n, 99);

    let mut first_result = vec![0.0f32; m * n];
    unsafe { exec_gemm(&layer, &a, &b, &mut first_result) };

    for run in 1..5 {
        let mut result = vec![0.0f32; m * n];
        unsafe { exec_gemm(&layer, &a, &b, &mut result) };
        assert_eq!(
            first_result, result,
            "GEMM non-deterministic on run {}",
            run
        );
    }
}

/// Verify SiLU produces bit-identical results across 5 invocations.
#[test]
fn silu_deterministic() {
    let n = 64;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Silu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 8.0).collect();

    let mut first_result = vec![0.0f32; n];
    unsafe { exec_unary(&layer, &input, &mut first_result) };

    for run in 1..5 {
        let mut result = vec![0.0f32; n];
        unsafe { exec_unary(&layer, &input, &mut result) };
        assert_eq!(
            first_result, result,
            "SiLU non-deterministic on run {}",
            run
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Zero / identity inputs
// ═══════════════════════════════════════════════════════════════════════

/// GEMM with all-zero A → output should be all zeros.
#[test]
fn gemm_zero_input() {
    let (m, n, k) = (8, 16, 8);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = vec![0.0f32; m * k];
    let b = fill_matrix(k, n, 42);
    let mut c = vec![999.0f32; m * n]; // pre-fill with non-zero

    unsafe { exec_gemm(&layer, &a, &b, &mut c) };

    for i in 0..m * n {
        assert!(
            c[i].abs() < 1e-10,
            "GEMM-zero: c[{}]={} should be ~0",
            i, c[i]
        );
    }
}

/// GEMM with identity-like B (k×k identity, n=k) → output should equal A.
#[test]
fn gemm_identity_b() {
    let (m, n, k) = (8, 16, 16);
    let (graph, op_id) = build_gemm_graph(m, n, k);
    let plan = build_standalone_plan(op_id);
    let alloc = build_alloc(m * n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a = fill_matrix(m, k, 42);
    let mut b = vec![0.0f32; k * n];
    for i in 0..k {
        b[i * n + i] = 1.0;
    }
    let mut c = vec![0.0f32; m * n];

    unsafe { exec_gemm(&layer, &a, &b, &mut c) };

    for i in 0..m * n {
        let diff = (c[i] - a[i]).abs();
        assert!(
            diff < 1e-5,
            "GEMM-identity: c[{}]={}, a[{}]={}, diff={}",
            i, c[i], i, a[i], diff
        );
    }
}

/// SiLU(0) should be 0.
#[test]
fn silu_zero_input() {
    let n = 16;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Silu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input = vec![0.0f32; n];
    let mut output = vec![999.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    for i in 0..n {
        assert!(
            output[i].abs() < 1e-6,
            "SiLU(0) at [{}]={} should be ~0",
            i, output[i]
        );
    }
}

/// GELU(0) should be 0.
#[test]
fn gelu_zero_input() {
    let n = 16;
    let (graph, op_id, _) = build_unary_graph(n, OpKind::Gelu);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let input = vec![0.0f32; n];
    let mut output = vec![999.0f32; n];

    unsafe { exec_unary(&layer, &input, &mut output) };

    for i in 0..n {
        assert!(
            output[i].abs() < 1e-6,
            "GELU(0) at [{}]={} should be ~0",
            i, output[i]
        );
    }
}

/// Add with zero second operand → identity.
#[test]
fn add_zero_identity() {
    let n = 32;
    let (graph, op_id, _) = build_binary_graph(n, OpKind::Add);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 5.0).collect();
    let b = vec![0.0f32; n];
    let mut output = vec![0.0f32; n];

    unsafe { exec_binary(&layer, &a, &b, &mut output) };

    for i in 0..n {
        let diff = (output[i] - a[i]).abs();
        assert!(diff < 1e-6, "Add-zero at [{}]: got={}, expected={}", i, output[i], a[i]);
    }
}

/// Mul with all-ones second operand → identity.
#[test]
fn mul_ones_identity() {
    let n = 32;
    let (graph, op_id, _) = build_binary_graph(n, OpKind::Mul);
    let plan = build_elementwise_plan(op_id);
    let alloc = build_alloc(n * 4);
    let layer = compile(&graph, &plan, &alloc);

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 5.0).collect();
    let b = vec![1.0f32; n];
    let mut output = vec![0.0f32; n];

    unsafe { exec_binary(&layer, &a, &b, &mut output) };

    for i in 0..n {
        let diff = (output[i] - a[i]).abs();
        assert!(diff < 1e-6, "Mul-ones at [{}]: got={}, expected={}", i, output[i], a[i]);
    }
}
