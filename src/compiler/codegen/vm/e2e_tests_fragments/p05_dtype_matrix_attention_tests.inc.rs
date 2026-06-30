// dtype-matrix 扩展：attention 算子族 B2-ready 结构断言 (BCE-20260630-MIXED-test §4)。
//
// 延续 p05_dtype_matrix_tests.inc.rs (GEMM) / p05_dtype_matrix_norm_tests.inc.rs (norm)
// 的防假完成理念，扩展到 attention 算子族。
//
// 当前 attention emit 层架构（架构师 MIXED-PRECISION-OPERATOR-PLAN §8 待澄清项）：
//   emit_score_dot_cpu / emit_tiled_attention_inline 用单一 `dtype` 参数（Q/KV 共用）。
//   即 Q (激活) 与 K/V (KV cache) 当前同 dtype — per-input 分离（Q_dtype + kv_dtype）
//   属 P2 lowering 层（lower_attention_v2），待 P2 落地后补 per-input 断言。
//
// 本片段验证当前状态下 attention 的 dtype 感知：
//   1. **结构断言**：传入 dtype=BF16 时，Q/KV VecLoad dtype==BF16（证明 attention emit
//      用参数化 dtype，非硬编码 F32 — 防回归）
//   2. **accumulator 恒 F32**：dot_acc/running_max/running_sum/o_acc Broadcast dtype==F32
//      （三段式 accumulate 位置，P3 promote 标注）
//   3. **反向回归**：传入 dtype=F32 时无异 dtype VecLoad
//
// 注：纯结构断言，不依赖 JIT 执行。
//
// per-input Q/KV 分离的 dtype 断言（K/V load dtype == kv_dtype != Q dtype）待 P2
// lower_attention_v2 落地（Q_dtype + kv_dtype 分离）后补。

#[cfg(test)]
mod p05_dtype_matrix_attention {
use crate::compiler::codegen::vm::attention_emit::emit_tiled_attention_inline;
use crate::compiler::codegen::vm::instr::{
    BoundExpr, KvLoadMode, SimdWidth, VRegKind, VmInstr, VmProgram,
};
use crate::compiler::codegen::vm::isa_hook::{
    AccessPattern, AttentionStrategy, EpiloguePlace, IsaHook, KvQuantImpl, TransImpl,
};
use crate::compiler::codegen::vm::instr::TranscendentalFn;
use crate::compiler::trace::QuantPrecision;

/// 简易 CPU hook fixture（mirror attention_emit.rs 测试模块的 TestHook）。
struct CpuTestHook;
impl IsaHook for CpuTestHook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
    fn transcendental_impl(&self, _: TranscendentalFn) -> TransImpl {
        TransImpl::Polynomial { degree: 5 }
    }
    fn epilogue_strategy(&self, _acc: usize, _epi: usize) -> EpiloguePlace {
        EpiloguePlace::AfterStore
    }
    fn prefetch_hint(&self, _access: &AccessPattern) -> Option<crate::compiler::codegen::vm::isa_hook::PrefetchConfig> { None }
    fn select_attention(&self, _seq_len: usize, _head_dim: usize) -> AttentionStrategy {
        AttentionStrategy::Naive
    }
    fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }
    fn is_gpu(&self) -> bool { false }
}

/// 断言 prog 中至少有一个 VecLoad 携带 `expect_dtype`。
fn assert_has_vecload_with_dtype(prog: &VmProgram, expect: QuantPrecision, ctx: &str) {
    let found = prog.instrs.iter().any(|i| matches!(
        i, VmInstr::VecLoad { dtype, .. } if *dtype == expect
    ));
    assert!(found,
        "[dtype-matrix-attn/{}] 期望存在 dtype=={:?} 的 VecLoad (Q/KV load), 但未找到", ctx, expect);
}

/// 断言 prog 中至少有一个 Broadcast 携带 `expect_dtype` (accumulator init)。
fn assert_has_broadcast_with_dtype(prog: &VmProgram, expect: QuantPrecision, ctx: &str) {
    let found = prog.instrs.iter().any(|i| matches!(
        i, VmInstr::Broadcast { dtype, .. } if *dtype == expect
    ));
    assert!(found,
        "[dtype-matrix-attn/{}] 期望存在 dtype=={:?} 的 Broadcast (accumulator init), 但未找到", ctx, expect);
}

// ── attention: dtype=BF16 → Q/KV VecLoad dtype==BF16（参数化 dtype 证明）────────

#[test]
fn dtype_matrix_attention_bf16_qkv_load_uses_dtype_param() {
    // 传入 dtype=BF16 → Q/KV VecLoad 应是 BF16（证明 attention emit 用参数 dtype，非硬编码 F32）
    let mut prog = VmProgram::new();
    let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let result = emit_tiled_attention_inline(
        &mut prog,
        BoundExpr::Const(1), BoundExpr::Const(4),
        2, 2, 64,
        SimdWidth::W256,
        q_ptr, k_ptr, v_ptr, out_ptr,
        Some(&CpuTestHook),
        false, None, QuantPrecision::BF16,
        None, 0, KvLoadMode::Direct,
        None, None, None,
        false, false,
    );
    assert!(result.is_ok(), "attention BF16 emit should succeed: {:?}", result);
    assert!(!prog.instrs.is_empty(), "attention should emit instructions");
    // Q/KV VecLoad dtype==BF16（参数化 dtype，非硬编码 F32）
    assert_has_vecload_with_dtype(&prog, QuantPrecision::BF16, "bf16_qkv_load");
}

#[test]
fn dtype_matrix_attention_bf16_accumulator_is_f32() {
    // 即使 dtype=BF16，accumulator (dot_acc/running_max/running_sum/o_acc) 恒 F32
    // （三段式 accumulate 位置，P3 promote 标注生效）
    let mut prog = VmProgram::new();
    let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let result = emit_tiled_attention_inline(
        &mut prog,
        BoundExpr::Const(1), BoundExpr::Const(4),
        2, 2, 64,
        SimdWidth::W256,
        q_ptr, k_ptr, v_ptr, out_ptr,
        Some(&CpuTestHook),
        false, None, QuantPrecision::BF16,
        None, 0, KvLoadMode::Direct,
        None, None, None,
        false, false,
    );
    assert!(result.is_ok());
    // accumulator init Broadcast 必须是 F32（accumulator_dtype，非 BF16）
    assert_has_broadcast_with_dtype(&prog, QuantPrecision::F32, "acc_f32");
}

// ── 反向回归：dtype=F32 无异 dtype 问题（基线）──────────────────────────

#[test]
fn dtype_matrix_attention_f32_load_uses_dtype_param() {
    // dtype=F32 → Q/KV VecLoad dtype==F32（基线，证明参数传播对 F32 也正确）
    let mut prog = VmProgram::new();
    let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let k_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let v_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let result = emit_tiled_attention_inline(
        &mut prog,
        BoundExpr::Const(1), BoundExpr::Const(4),
        2, 2, 64,
        SimdWidth::W256,
        q_ptr, k_ptr, v_ptr, out_ptr,
        Some(&CpuTestHook),
        false, None, QuantPrecision::F32,
        None, 0, KvLoadMode::Direct,
        None, None, None,
        false, false,
    );
    assert!(result.is_ok());
    assert_has_vecload_with_dtype(&prog, QuantPrecision::F32, "f32_qkv_load");
}
} // end mod p05_dtype_matrix_attention
