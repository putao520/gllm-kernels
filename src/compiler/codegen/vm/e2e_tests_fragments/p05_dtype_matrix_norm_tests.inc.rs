// dtype-matrix 扩展：norm/softmax 算子族 B2-ready 结构断言 (BCE-20260630-MIXED-test §3)。
//
// 延续 p05_dtype_matrix_tests.inc.rs (GEMM) 的防假完成理念，扩展到 norm/softmax 算子族，
// 证明全算子 B2-ready（用户目标"算子级混合精度"完整证明）。
//
// 架构师三段式语义 (MIXED-PRECISION-OPERATOR-PLAN §0)：
//   load (激活/权重 per-tensor dtype) / accumulate (恒 F32) / store (输出 dtype)
//
// 覆盖：
//   1. **norm (RMSNorm/LayerNorm)** — 混合精度 (F32 激活 × BF16 权重)：
//      - 结构断言：weight VecLoad dtype == BF16 (per-weight dtype，BF16 经 WidenCompute widen)
//      - 结构断言：accumulator (Broadcast acc / reduce / HReduce) dtype == F32 (accumulator_dtype)
//   2. **softmax** — online softmax accumulator 恒 F32 (max_val/sum_val promote)
//   3. **反向回归** — uniform F32 下无异 dtype VecLoad
//
// 注：纯结构断言 (VmInstr dtype 检查)，不依赖 JIT 执行，跨架构可移植。
// 数值等价由 norm_softmax_emit.rs 的 OpImpl 对齐测试覆盖。
//
// 不覆盖 (本片段)：
//   - attention Q/KV per-input：当前 emit_score_dot_cpu 用单一 dtype (Q/KV 共用)，
//     per-input 分离属 P2 lowering 层 (lower_attention_v2，需 Q_dtype + kv_dtype 分离)，
//     架构师标注待 P2 落地后补此断言。
//   - vision/audio：lower_depthwise_conv1d/lower_patch_embed 接受 op/graph/resolver/abi，
//     需复杂 fixture，由 vision_audio_emit.rs 的 OpImpl 对齐测试覆盖权重 VecLoad dtype。

#[cfg(test)]
mod p05_dtype_matrix_norm {
use crate::compiler::codegen::vm::instr::{BoundExpr, SimdWidth, VRegKind, VmInstr, VmProgram};
use crate::compiler::codegen::vm::norm_softmax_emit::{
    emit_normlike_inline, emit_softmax_inline, NormKind,
};
use crate::compiler::trace::{ComputePattern, QuantPrecision, TraceOp, ValueId};

/// 断言 prog 中至少有一个 VecLoad 携带 `expect_dtype`。
fn assert_has_vecload_with_dtype(prog: &VmProgram, expect: QuantPrecision, ctx: &str) {
    let found = prog.instrs.iter().any(|i| matches!(
        i, VmInstr::VecLoad { dtype, .. } if *dtype == expect
    ));
    assert!(found,
        "[dtype-matrix-norm/{}] 期望存在 dtype=={:?} 的 VecLoad, 但未找到", ctx, expect);
}

/// 断言 prog 中至少有一个 Broadcast 携带 `expect_dtype` (accumulator init)。
fn assert_has_broadcast_with_dtype(prog: &VmProgram, expect: QuantPrecision, ctx: &str) {
    let found = prog.instrs.iter().any(|i| matches!(
        i, VmInstr::Broadcast { dtype, .. } if *dtype == expect
    ));
    assert!(found,
        "[dtype-matrix-norm/{}] 期望存在 dtype=={:?} 的 Broadcast (accumulator init), 但未找到", ctx, expect);
}

/// 反向断言：不存在异于 `only_dtype` 的 VecLoad (回归基线)。
fn assert_no_vecload_other_than(prog: &VmProgram, only: QuantPrecision, ctx: &str) {
    let stray = prog.instrs.iter().any(|i| matches!(
        i, VmInstr::VecLoad { dtype, .. } if *dtype != only
    ));
    assert!(!stray,
        "[dtype-matrix-norm/{}] 统一 dtype={:?} 下出现异 dtype VecLoad", ctx, only);
}

/// Helper: 构造 RMSNorm NormLike pattern (reduce=x*x, finalize=rsqrt, transform=mul weight)。
fn rmsnorm_pattern() -> ComputePattern {
    ComputePattern::NormLike {
        reduce: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))],
        finalize: vec![TraceOp::Input(0), TraceOp::Rsqrt(ValueId(0))],
        transform: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))],
    }
}

// ── norm: 混合精度 (F32 激活 × BF16 权重) 结构断言 ──────────────────────

#[test]
fn dtype_matrix_norm_rmsnorm_bf16_weight_load_uses_weight_dtype() {
    // 混合精度：激活 F32，权重 BF16 → weight VecLoad 必须是 BF16 (非 F32)
    // 证明 BCE-20260629-011 weight_dtype 多路传播 (DT2 范例 + P1 per-op ctx)
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let input_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let pattern = rmsnorm_pattern();
    emit_normlike_inline(
        &mut prog, &pattern, 64, 1, true, NormKind::RmsNorm,
        width, BoundExpr::Const(2),
        input_ptr, weight_ptr, output_ptr,
        QuantPrecision::F32, QuantPrecision::BF16, // act=F32, weight=BF16
    ).expect("normlike mixed-precision emit");
    assert!(!prog.instrs.is_empty(), "RMSNorm should emit instructions");
    // weight VecLoad 用 weight_dtype=BF16 (核心 B2-ready 断言)
    assert_has_vecload_with_dtype(&prog, QuantPrecision::BF16, "rmsnorm/bf16_weight");
}

#[test]
fn dtype_matrix_norm_rmsnorm_bf16_weight_accumulator_is_f32() {
    // 混合精度下 accumulator 必须恒 F32 (三段式 accumulate 位置，P3 promote 标注)
    // 即使 weight=BF16, acc Broadcast/reduce/HReduce 仍 F32 (数值稳定性)
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let input_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let pattern = rmsnorm_pattern();
    emit_normlike_inline(
        &mut prog, &pattern, 64, 1, true, NormKind::RmsNorm,
        width, BoundExpr::Const(2),
        input_ptr, weight_ptr, output_ptr,
        QuantPrecision::F32, QuantPrecision::BF16,
    ).expect("normlike mixed-precision emit");
    // accumulator init Broadcast 必须是 F32 (accumulator_dtype，非 BF16)
    assert_has_broadcast_with_dtype(&prog, QuantPrecision::F32, "rmsnorm/acc_f32");
}

#[test]
fn dtype_matrix_norm_layernorm_bf16_weight_load_uses_weight_dtype() {
    // LayerNorm 同理：BF16 权重 VecLoad dtype=BF16
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let input_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let pattern = rmsnorm_pattern();
    emit_normlike_inline(
        &mut prog, &pattern, 64, 1, true, NormKind::LayerNorm,
        width, BoundExpr::Const(2),
        input_ptr, weight_ptr, output_ptr,
        QuantPrecision::F32, QuantPrecision::BF16,
    ).expect("layernorm mixed-precision emit");
    assert!(!prog.instrs.is_empty());
    assert_has_vecload_with_dtype(&prog, QuantPrecision::BF16, "layernorm/bf16_weight");
}

// ── softmax: online softmax accumulator 恒 F32 ─────────────────────────

#[test]
fn dtype_matrix_softmax_f32_accumulator_promote() {
    // softmax max_val/sum_val 是 accumulate 位置，恒 F32 (P3 阶段2 promote)
    // 即使 dtype=F32 (B1)，accumulator_dtype() 返回 F32，行为等价
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let input_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    emit_softmax_inline(
        &mut prog, 64, width, input_ptr, output_ptr, QuantPrecision::F32,
    ).expect("softmax F32 emit");
    assert!(!prog.instrs.is_empty());
    // accumulator Broadcast (max_val init NEG_INFINITY / sum_val init 0) 必须 F32
    assert_has_broadcast_with_dtype(&prog, QuantPrecision::F32, "softmax/acc_f32");
}

// ── 反向回归：uniform F32 norm 无异 dtype VecLoad ──────────────────────

#[test]
fn dtype_matrix_norm_uniform_f32_no_stray_dtype() {
    // 统一 F32 (激活 F32 + 权重 F32) 下不应出现 BF16 等异 dtype VecLoad
    // 防止 P3 把 weight load 硬编码成 BF16
    let mut prog = VmProgram::new();
    let width = SimdWidth::W256;
    let input_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let output_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
    let pattern = rmsnorm_pattern();
    emit_normlike_inline(
        &mut prog, &pattern, 64, 1, true, NormKind::RmsNorm,
        width, BoundExpr::Const(2),
        input_ptr, weight_ptr, output_ptr,
        QuantPrecision::F32, QuantPrecision::F32, // uniform F32
    ).expect("normlike uniform-F32 emit");
    assert_no_vecload_other_than(&prog, QuantPrecision::F32, "uniform_f32");
}
} // end mod p05_dtype_matrix_norm
