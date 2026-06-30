// P0.5 dtype-matrix 防假完成护栏测试 (BCE-20260630-MIXED-test §2)。
//
// 架构师防假完成要求：emit_gemm_* 的 B-matrix load 必须使用 **b_dtype**
// (per-tensor)，而非恒等于 a_dtype。否则 P0.5 "GEMM 实现层真用 b_dtype"
// 退化为 `let _ = (b_dtype, c_dtype);` 丢弃模式 (假完成)。
//
// 矩阵覆盖：trans_b={true,false} × backend={CPU blis, GPU tiled} = 4 组合。
// 对每个组合：
//   1. 结构断言：编译产物 VmInstr 中存在 B-matrix VecLoad 携带 dtype==b_dtype
//      (取 a_dtype≠b_dtype 的混合精度组合，使断言有区分力)
//   2. 反向断言：当 a_dtype==b_dtype 时所有 VecLoad dtype 一致 (回归基线)
//
// 注：纯结构断言 (VmInstr dtype 检查)，不依赖 JIT 执行，跨架构可移植。
// 数值等价 (vs scalar 参考) 已由 gemm_impls.rs 的 BF16/F32 OpImpl 对齐测试覆盖
// (verify_op_impl_aligns_scalar, BF16 容差 1e-2 / F32 1e-5)。

#[cfg(test)]
mod p05_dtype_matrix {
use crate::compiler::codegen::vm::instr::{SimdWidth, VRegKind, VmInstr, VmProgram};
use crate::compiler::codegen::vm::gemm_emit::{
    emit_gemm_blis_inline, emit_gemm_gpu_tiled_inline,
};
use crate::compiler::trace::QuantPrecision;

/// 断言 prog 中至少有一个 VecLoad 携带 `expect_dtype`。
/// 证明 B-matrix load 使用了 per-tensor dtype (而非恒 a_dtype)。
fn assert_has_vecload_with_dtype(prog: &VmProgram, expect_dtype: QuantPrecision, ctx: &str) {
    let found = prog.instrs.iter().any(|i| match i {
        VmInstr::VecLoad { dtype, .. } => *dtype == expect_dtype,
        _ => false,
    });
    assert!(
        found,
        "[P0.5-dtype-matrix/{}] 期望存在 dtype=={:?} 的 VecLoad (B-matrix load), \
         但未找到 — 说明 emit_gemm_* 未真正使用 b_dtype (假完成回归)",
        ctx, expect_dtype,
    );
}

/// 反向断言：当 a_dtype==b_dtype==c_dtype 时，不存在异于该 dtype 的 VecLoad。
/// 防止 P0.5 把 B-load 硬编码成某个固定 dtype 而非参数传入值。
fn assert_no_vecload_other_than(prog: &VmProgram, only_dtype: QuantPrecision, ctx: &str) {
    let stray = prog.instrs.iter().any(|i| match i {
        VmInstr::VecLoad { dtype, .. } => *dtype != only_dtype,
        _ => false,
    });
    assert!(
        !stray,
        "[P0.5-dtype-matrix/{}] 统一 dtype={:?} 下出现异 dtype VecLoad — \
         说明 B-load dtype 非来自参数 (硬编码嫌疑)",
        ctx, only_dtype,
    );
}

// ── CPU (BLIS) 路径 ──────────────────────────────────────────────────

#[test]
fn p05_dtype_matrix_cpu_blis_transb_false_b_load_uses_b_dtype() {
    // 混合精度：A=F32 激活, B=BF16 权重 → B-load 必须是 BF16 (非 F32)
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let a = prog.alloc_vreg(VRegKind::Ptr, width);
    let b = prog.alloc_vreg(VRegKind::Ptr, width);
    let c = prog.alloc_vreg(VRegKind::Ptr, width);
    emit_gemm_blis_inline(
        &mut prog, 4, 8, 16, width, a, b, c, 4, 2, None, 1,
        QuantPrecision::F32, QuantPrecision::BF16, QuantPrecision::F32, false,
    ).expect("blis GEMM mixed-precision emit");
    assert!(!prog.instrs.is_empty(), "blis GEMM should emit instructions");
    // B-load 用 b_dtype=BF16 (核心防假完成断言)
    assert_has_vecload_with_dtype(&prog, QuantPrecision::BF16, "cpu_blis/transb=false");
}

#[test]
fn p05_dtype_matrix_cpu_blis_transb_true_b_load_uses_b_dtype() {
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let a = prog.alloc_vreg(VRegKind::Ptr, width);
    let b = prog.alloc_vreg(VRegKind::Ptr, width);
    let c = prog.alloc_vreg(VRegKind::Ptr, width);
    emit_gemm_blis_inline(
        &mut prog, 4, 8, 16, width, a, b, c, 4, 2, None, 1,
        QuantPrecision::F32, QuantPrecision::BF16, QuantPrecision::F32, true,
    ).expect("blis GEMM mixed-precision trans_b emit");
    assert!(!prog.instrs.is_empty(), "blis GEMM trans_b should emit instructions");
    assert_has_vecload_with_dtype(&prog, QuantPrecision::BF16, "cpu_blis/transb=true");
}

// ── GPU (tiled) 路径 ────────────────────────────────────────────────

#[test]
fn p05_dtype_matrix_gpu_tiled_transb_false_b_load_uses_b_dtype() {
    let mut prog = VmProgram::new();
    let width = SimdWidth::Warp(32);
    let a = prog.alloc_vreg(VRegKind::Ptr, width);
    let b = prog.alloc_vreg(VRegKind::Ptr, width);
    let c = prog.alloc_vreg(VRegKind::Ptr, width);
    emit_gemm_gpu_tiled_inline(
        &mut prog, 16, 16, 16, width, a, b, c,
        16, 16, 16, 16, 16, 16,
        QuantPrecision::F32, QuantPrecision::BF16, QuantPrecision::F32, false,
    ).expect("gpu tiled GEMM mixed-precision emit");
    assert!(!prog.instrs.is_empty(), "gpu tiled GEMM should emit instructions");
    assert_has_vecload_with_dtype(&prog, QuantPrecision::BF16, "gpu_tiled/transb=false");
}

#[test]
fn p05_dtype_matrix_gpu_tiled_transb_true_b_load_uses_b_dtype() {
    let mut prog = VmProgram::new();
    let width = SimdWidth::Warp(32);
    let a = prog.alloc_vreg(VRegKind::Ptr, width);
    let b = prog.alloc_vreg(VRegKind::Ptr, width);
    let c = prog.alloc_vreg(VRegKind::Ptr, width);
    emit_gemm_gpu_tiled_inline(
        &mut prog, 16, 16, 16, width, a, b, c,
        16, 16, 16, 16, 16, 16,
        QuantPrecision::F32, QuantPrecision::BF16, QuantPrecision::F32, true,
    ).expect("gpu tiled GEMM mixed-precision trans_b emit");
    assert!(!prog.instrs.is_empty(), "gpu tiled GEMM trans_b should emit instructions");
    assert_has_vecload_with_dtype(&prog, QuantPrecision::BF16, "gpu_tiled/transb=true");
}

// ── 反向回归：统一 dtype 下不应出现异 dtype VecLoad ─────────────────

#[test]
fn p05_dtype_matrix_uniform_f32_no_stray_dtype() {
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let a = prog.alloc_vreg(VRegKind::Ptr, width);
    let b = prog.alloc_vreg(VRegKind::Ptr, width);
    let c = prog.alloc_vreg(VRegKind::Ptr, width);
    emit_gemm_blis_inline(
        &mut prog, 4, 8, 16, width, a, b, c, 4, 2, None, 1,
        QuantPrecision::F32, QuantPrecision::F32, QuantPrecision::F32, false,
    ).expect("blis GEMM uniform-F32 emit");
    assert_no_vecload_other_than(&prog, QuantPrecision::F32, "uniform_f32");
}
} // end mod p05_dtype_matrix
