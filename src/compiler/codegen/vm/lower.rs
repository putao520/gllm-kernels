//! TraceOp → VmInstr Lowering — 工具函数 + 兼容层 + QTapSTG
//!
//! 所有算术 VmInstr 发射已迁移至 auto_select.rs (auto_lower_trace_raw)。
//! 本文件保留:
//! - computation_elem_bytes / row_stride_bytes: 纯工具函数
//! - lower_trace_body_compat: 委托 auto_lower_trace 的兼容包装 (8 处生产调用: gemm_emit + pipeline.inc)
//! - lower_qtap_stg: Q-Tap ring buffer 写入 (纯控制流+内存操作, 零算术 VmInstr)
//!
//! 已删除的废弃函数（均已被 plan_lower.rs emit_*_inline + auto_lower_trace 替代）:
//! - lower_elementwise, lower_gemm, lower_gemm_tile
//! - lower_rope, lower_rope_full
//! - lower_gather, lower_column_slice
//! - lower_moe_dispatch_packed
//!
//! ARCH-DTYPE-JIT-TYPED: JIT 是多精度混合(BF16/F16/F32/I8 等),dtype 从 tensor metadata 推导。
//! lower_trace_body_compat 接收 dtype 参数,禁止硬编码 F32。

use super::instr::*;
use crate::compiler::graph::QTapPosition;
use crate::compiler::trace::{QuantPrecision, TraceOp};
use crate::types::{CompilerError, DType};

/// 计算元素字节数。多精度混合(BF16/F16/F32/I8/U8/Quant 等),dtype 由调用方传入,
/// 真实 dtype 走 `op_input_dtype(op, graph)` (ARCH-DTYPE-JIT-TYPED)。
///
/// 调用方必须从 tensor metadata / ctx.dtype 推导 dtype 传入,
/// 禁止使用 QuantPrecision::F32 兜底(测试代码除外)。
#[inline]
pub fn computation_elem_bytes(dtype: QuantPrecision) -> usize {
    dtype.elem_bytes()
}

/// Row-major 张量行步长（字节数）: 最内层维度大小 × elem_bytes(dtype)。
#[inline]
pub fn row_stride_bytes(inner_dim: usize, dtype: QuantPrecision) -> usize {
    inner_dim * computation_elem_bytes(dtype)
}

/// 兼容入口: primary + Option<secondary> → &[VRegId]。
///
/// 内部委托 auto_lower_trace，无手写 VmInstr。
/// dtype 由调用方传入(ARCH-DTYPE-JIT-TYPED:从 op inputs tensor metadata 推导)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_trace_body_compat(
    prog: &mut VmProgram,
    body: &[TraceOp],
    primary: VRegId,
    secondary: Option<VRegId>,
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    match secondary {
        Some(sec) => auto_lower_trace_inner(prog, body, &[primary, sec], width, dtype),
        None => auto_lower_trace_inner(prog, body, &[primary], width, dtype),
    }
}

fn auto_lower_trace_inner(
    prog: &mut VmProgram,
    body: &[TraceOp],
    inputs: &[VRegId],
    width: SimdWidth,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    super::auto_select::auto_lower_trace(prog, body, inputs, width, dtype)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Q-Tap STG — ring buffer 写入 (纯控制流 + 内存操作)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Q-Tap ring buffer 写入 + atomic step_index bump。
///
/// 纯控制流结构: LoadPtr + VecLoad/VecStore + MemFence + AtomicAdd。
/// 无算术 VmInstr (VecBinOp/Transcendental/Fma 等)。
#[allow(clippy::too_many_arguments)]
pub fn lower_qtap_stg(
    prog: &mut VmProgram,
    sink_ptr: u64,
    step_index_ptr: u64,
    dtype: DType,
    q_dim: usize,
    seq_bound: BoundExpr,
    position: QTapPosition,
    num_slots: usize,
    width: SimdWidth,
    q_input_ptr: VRegId,
) -> Result<(), CompilerError> {
    if num_slots < 2 {
        return Err(CompilerError::CodegenViolation(
            "lower_qtap_stg: num_slots must be >= 2 (double buffering rule)".into(),
        ));
    }
    if !num_slots.is_power_of_two() {
        return Err(CompilerError::CodegenViolation(format!(
            "lower_qtap_stg: num_slots = {num_slots} is not a power of two"
        )));
    }
    if q_dim == 0 {
        return Err(CompilerError::CodegenViolation(
            "lower_qtap_stg: q_dim must be > 0".into(),
        ));
    }
    // dtype 从调用方传入，通过 to_quant_precision() 转换为 VmInstr 层面的 QuantPrecision
    // 禁止硬编码 F32 限制，遵循 ARCH-DTYPE-JIT-TYPED 铁律
    if sink_ptr == 0 || step_index_ptr == 0 {
        return Err(CompilerError::CodegenViolation(
            "lower_qtap_stg: sink_ptr / step_index_ptr must be non-zero".into(),
        ));
    }

    let elem = dtype.size_bytes();
    let vp_dtype = dtype.to_quant_precision();
    let lanes = width.f32_lanes().max(1);
    let vec_step_bytes = lanes * elem;
    if q_dim % lanes != 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "lower_qtap_stg: q_dim = {q_dim} not divisible by SIMD lanes = {lanes}"
        )));
    }

    if num_slots != 2 {
        return Err(CompilerError::CodegenViolation(format!(
            "lower_qtap_stg: num_slots = {num_slots} currently only supports 2"
        )));
    }

    prog.emit(VmInstr::Comment("QTapSTG: ring buffer write + atomic step_index bump".into()));

    let sink_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: sink_base, src: PtrExpr::AbsAddr(sink_ptr) });
    let step_index_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::LoadPtr { dst: step_index_base, src: PtrExpr::AbsAddr(step_index_ptr) });

    prog.emit(VmInstr::MemFence { order: MemFenceOrder::Acquire });

    let dst_base = sink_base;

    match position {
        QTapPosition::LastToken => {
            let row_bytes = q_dim * elem;
            let src_row_ptr = match &seq_bound {
                BoundExpr::Const(n) => {
                    if *n == 0 {
                        return Err(CompilerError::CodegenViolation(
                            "lower_qtap_stg: seq_len must be > 0 for LastToken".into(),
                        ));
                    }
                    let offset = (*n - 1) * row_bytes;
                    let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: ptr, src: PtrExpr::VRegPlusConst(q_input_ptr, offset) });
                    ptr
                }
                BoundExpr::DynamicVReg(vreg) => {
                    let seq_minus_1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                    prog.emit(VmInstr::GprBinOp { dst: seq_minus_1, a: *vreg, b: GprOperand::Imm(0_i64), op: GprOp::Shl });
                    compute_row_ptr(prog, q_input_ptr, seq_minus_1, row_bytes)
                }
                BoundExpr::DynamicVRegPlusOne(vreg) => {
                    let seq_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                    prog.emit(VmInstr::GprBinOp { dst: seq_val, a: *vreg, b: GprOperand::Imm(0_i64), op: GprOp::Shl });
                    compute_row_ptr(prog, q_input_ptr, seq_val, row_bytes)
                }
                BoundExpr::Runtime(ptr) => {
                    let seq_minus_1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                    prog.emit(VmInstr::LoadPtr { dst: seq_minus_1, src: ptr.clone() });
                    prog.emit(VmInstr::GprBinOp { dst: seq_minus_1, a: seq_minus_1, b: GprOperand::Imm(1_i64), op: GprOp::Sub });
                    compute_row_ptr(prog, q_input_ptr, seq_minus_1, row_bytes)
                }
                _ => {
                    return Err(CompilerError::CodegenViolation(
                        "lower_qtap_stg: LastToken with unresolved Symbolic seq_bound".into(),
                    ));
                }
            };

            let num_vecs = q_dim / lanes;
            let acc = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit_loop(BoundExpr::Const(num_vecs), vec_step_bytes, |prog, _ctr, byte_off| {
                prog.emit(VmInstr::VecLoad {
                    dst: acc, base: src_row_ptr,
                    offset: OffsetExpr::LoopOffset(byte_off), width,
                    dtype: vp_dtype, predicate: None,
                });
                prog.emit(VmInstr::VecStore {
                    base: dst_base, offset: OffsetExpr::LoopOffset(byte_off),
                    src: acc, width,
                    dtype: vp_dtype, predicate: None,
                });
            });
        }
        QTapPosition::AllTokens => {
            let row_bytes = q_dim * elem;
            let num_vecs = q_dim / lanes;
            let acc = prog.alloc_vreg(VRegKind::Vec, width);

            prog.emit_loop(seq_bound, row_bytes, |prog, _seq_ctr, seq_byte_off| {
                let src_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: src_row, src: PtrExpr::VRegPlusVReg(q_input_ptr, seq_byte_off),
                });
                let dst_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: dst_row, src: PtrExpr::VRegPlusVReg(dst_base, seq_byte_off),
                });
                prog.emit_loop(BoundExpr::Const(num_vecs), vec_step_bytes, |prog, _ctr, off| {
                    prog.emit(VmInstr::VecLoad {
                        dst: acc, base: src_row,
                        offset: OffsetExpr::LoopOffset(off), width,
                        dtype: vp_dtype, predicate: None,
                    });
                    prog.emit(VmInstr::VecStore {
                        base: dst_row, offset: OffsetExpr::LoopOffset(off),
                        src: acc, width,
                        dtype: vp_dtype, predicate: None,
                    });
                });
            });
        }
    }

    prog.emit(VmInstr::MemFence { order: MemFenceOrder::Release });
    prog.emit(VmInstr::AtomicAdd {
        base: step_index_base, offset: OffsetExpr::Const(0), value: 1, elem_width: 8,
    });

    Ok(())
}

/// 从 seq_index 和 row_bytes 计算 q_input_ptr + seq_index * row_bytes。
/// 使用 shift-add 分解 row_bytes 来避免标量乘法指令。
fn compute_row_ptr(
    prog: &mut VmProgram,
    base_ptr: VRegId,
    seq_index: VRegId,
    row_bytes: usize,
) -> VRegId {
    let mut offset_vreg: Option<VRegId> = None;
    let mut bit = 0usize;
    let mut val = row_bytes;
    while val > 0 {
        if val & 1 != 0 {
            let partial = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp { dst: partial, a: seq_index, b: GprOperand::Imm(bit as u8  as i64), op: GprOp::Shl });
            match offset_vreg {
                None => offset_vreg = Some(partial),
                Some(prev) => {
                    let sum = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                    prog.emit(VmInstr::GprBinOp { dst: sum, a: prev, b: GprOperand::VReg(partial ), op: GprOp::Add });
                    offset_vreg = Some(sum);
                }
            }
        }
        val >>= 1;
        bit += 1;
    }
    let final_offset = offset_vreg.unwrap_or_else(|| {
        let z = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst: z, a: z, b: GprOperand::Imm(0_i64), op: GprOp::Sub });
        z
    });
    let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    prog.emit(VmInstr::GprBinOp { dst: ptr, a: base_ptr, b: GprOperand::VReg(final_offset ), op: GprOp::Add });
    ptr
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── computation_elem_bytes ──

    #[test]
    fn computation_elem_bytes_returns_4() {
        // Arrange & Act
        let bytes = computation_elem_bytes(QuantPrecision::F32);
        // Assert: f32 is 4 bytes
        assert_eq!(bytes, 4);
    }

    #[test]
    fn computation_elem_bytes_matches_sizeof_f32() {
        // Arrange & Act
        let bytes = computation_elem_bytes(QuantPrecision::F32);
        // Assert: must always equal std::mem::size_of::<f32>()
        assert_eq!(bytes, std::mem::size_of::<f32>());
    }

    // ── row_stride_bytes ──

    #[test]
    fn row_stride_bytes_inner_dim_1() {
        // Arrange
        let inner_dim = 1;
        // Act
        let stride = row_stride_bytes(inner_dim, QuantPrecision::F32);
        // Assert: 1 * 4 = 4
        assert_eq!(stride, 4);
    }

    #[test]
    fn row_stride_bytes_inner_dim_256() {
        // Arrange
        let inner_dim = 256;
        // Act
        let stride = row_stride_bytes(inner_dim, QuantPrecision::F32);
        // Assert: 256 * 4 = 1024
        assert_eq!(stride, 1024);
    }

    #[test]
    fn row_stride_bytes_inner_dim_0() {
        // Arrange
        let inner_dim = 0;
        // Act
        let stride = row_stride_bytes(inner_dim, QuantPrecision::F32);
        // Assert: 0 * 4 = 0
        assert_eq!(stride, 0);
    }

    // ── lower_qtap_stg validation errors ──

    fn make_default_qtap_args() -> (u64, u64, DType, usize, BoundExpr, QTapPosition, usize, SimdWidth, VRegId) {
        let mut prog = VmProgram::new();
        let q_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Return args but drop prog — we only need the VRegId as a dummy.
        // lower_qtap_stg creates its own VmProgram internally by the caller.
        // For validation tests we create a fresh prog inside each test.
        (0x1000, 0x2000, DType::F32, 8, BoundExpr::Const(1), QTapPosition::LastToken, 2, SimdWidth::W256, q_input_ptr)
    }

    #[test]
    fn lower_qtap_stg_rejects_num_slots_less_than_2() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 1,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("num_slots must be >= 2"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_rejects_non_power_of_two_slots() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act: 3 is not a power of two
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 3,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("not a power of two"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_rejects_zero_q_dim() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 0,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("q_dim must be > 0"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_rejects_zero_sink_ptr() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act: sink_ptr = 0
        let result = lower_qtap_stg(
            &mut prog, 0, 0x2000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("non-zero"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_rejects_zero_step_index_ptr() {
        // Arrange
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act: step_index_ptr = 0
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("non-zero"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_emits_valid_last_token_program() {
        // Arrange: valid args — F32, q_dim=8, W256 (8 lanes), 2 slots, seq=1
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let instrs_before = prog.len();

        // Act
        let result = lower_qtap_stg(
            &mut prog, 0xDEAD_0000, 0xBEEF_0000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );

        // Assert
        assert!(result.is_ok(), "expected Ok, got Err: {:?}", result.unwrap_err());
        // Must have emitted instructions (at minimum: Comment, LoadPtr x2, MemFence, loop body, MemFence, AtomicAdd)
        assert!(prog.len() > instrs_before, "program should have grown after lower_qtap_stg");
    }

    #[test]
    fn lower_qtap_stg_rejects_q_dim_not_divisible_by_lanes() {
        // Arrange: q_dim=5, W256 (8 lanes) → 5 % 8 != 0
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 5,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("not divisible by SIMD lanes"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    // ── Additional lowering tests ──

    #[test]
    fn lower_qtap_stg_rejects_num_slots_4_currently_unsupported() {
        // Arrange: num_slots=4 is power-of-two and >= 2, but only 2 is currently supported
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 4,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("currently only supports 2"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_rejects_zero_seq_len_last_token() {
        // Arrange: LastToken with BoundExpr::Const(0) → seq_len must be > 0
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(0), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("seq_len must be > 0"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_last_token_emits_mem_fences_and_atomic_add() {
        // Arrange: valid LastToken program
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0xDEAD_0000, 0xBEEF_0000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_ok());
        // Must contain MemFence(Acquire), MemFence(Release), and AtomicAdd
        let has_acquire = prog.instrs.iter().any(|i| matches!(i, VmInstr::MemFence { order: MemFenceOrder::Acquire }));
        let has_release = prog.instrs.iter().any(|i| matches!(i, VmInstr::MemFence { order: MemFenceOrder::Release }));
        let has_atomic_add = prog.instrs.iter().any(|i| matches!(i, VmInstr::AtomicAdd { .. }));
        assert!(has_acquire, "program must contain MemFence Acquire");
        assert!(has_release, "program must contain MemFence Release");
        assert!(has_atomic_add, "program must contain AtomicAdd");
    }

    #[test]
    fn lower_qtap_stg_all_tokens_emits_nested_loops() {
        // Arrange: valid AllTokens program with seq_bound=Const(2)
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0xDEAD_0000, 0xBEEF_0000, DType::F32, 8,
            BoundExpr::Const(2), QTapPosition::AllTokens, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_ok());
        // AllTokens emits an outer loop (seq) + inner loop (vec), so we expect
        // at least 2 LoopBegin and 2 LoopEnd instructions
        let loop_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        let loop_ends = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopEnd)).count();
        assert!(loop_begins >= 2, "AllTokens should have at least 2 LoopBegin (outer seq + inner vec), got {loop_begins}");
        assert_eq!(loop_begins, loop_ends, "LoopBegin/LoopEnd must be balanced");
    }

    #[test]
    fn lower_qtap_stg_last_token_valid_program_passes_structure_validation() {
        // Arrange: valid LastToken program
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        // Act
        let result = lower_qtap_stg(
            &mut prog, 0x5000, 0x6000, DType::F32, 16,
            BoundExpr::Const(4), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_ok());
        // The generated program must have balanced loops and scopes
        assert!(prog.validate_structure().is_ok(), "generated program must pass structure validation");
    }

    #[test]
    fn lower_qtap_stg_last_token_emits_abs_addr_loadptr() {
        // Arrange: verify that AbsAddr(sink_ptr) and AbsAddr(step_index_ptr) appear in LoadPtr
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sink_ptr: u64 = 0xCAFE_0000;
        let step_index_ptr: u64 = 0xBABE_0000;
        // Act
        let result = lower_qtap_stg(
            &mut prog, sink_ptr, step_index_ptr, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        // Assert
        assert!(result.is_ok());
        let has_sink_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoadPtr { src: PtrExpr::AbsAddr(addr), .. } if *addr == sink_ptr));
        let has_step_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoadPtr { src: PtrExpr::AbsAddr(addr), .. } if *addr == step_index_ptr));
        assert!(has_sink_load, "program must load sink_ptr via AbsAddr");
        assert!(has_step_load, "program must load step_index_ptr via AbsAddr");
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn row_stride_bytes_large_inner_dim() {
        let stride = row_stride_bytes(16384, QuantPrecision::F32);
        assert_eq!(stride, 65536); // 16384 * 4
    }

    #[test]
    fn computation_elem_bytes_is_constexpr_f32() {
        assert_eq!(computation_elem_bytes(QuantPrecision::F32), 4);
        assert_eq!(computation_elem_bytes(QuantPrecision::F32), core::mem::size_of::<f32>());
    }

    #[test]
    fn lower_qtap_stg_all_tokens_valid_structure() {
        // AllTokens with seq_bound=Const(1), q_dim=16, W256
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_qtap_stg(
            &mut prog, 0xA000, 0xB000, DType::F32, 16,
            BoundExpr::Const(1), QTapPosition::AllTokens, 2,
            SimdWidth::W256, q_input,
        );
        assert!(result.is_ok());
        assert!(prog.validate_structure().is_ok(), "AllTokens program must pass structure validation");
    }

    #[test]
    fn lower_qtap_stg_last_token_with_larger_q_dim() {
        // q_dim=32 with W256 (8 lanes): 32/8=4 vec iterations
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 32,
            BoundExpr::Const(2), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        assert!(result.is_ok());
        // Verify there are VecLoad and VecStore instructions
        let loads = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecLoad { .. })).count();
        let stores = prog.instrs.iter().filter(|i| matches!(i, VmInstr::VecStore { .. })).count();
        assert!(loads > 0, "should have VecLoad instructions");
        assert!(stores > 0, "should have VecStore instructions");
    }

    #[test]
    fn lower_qtap_stg_all_tokens_emits_mem_fences() {
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(2), QTapPosition::AllTokens, 2,
            SimdWidth::W256, q_input,
        );
        assert!(result.is_ok());
        let has_acquire = prog.instrs.iter().any(|i| matches!(i, VmInstr::MemFence { order: MemFenceOrder::Acquire }));
        let has_release = prog.instrs.iter().any(|i| matches!(i, VmInstr::MemFence { order: MemFenceOrder::Release }));
        assert!(has_acquire, "AllTokens must have Acquire fence");
        assert!(has_release, "AllTokens must have Release fence");
    }

    #[test]
    fn lower_qtap_stg_last_token_with_w128() {
        // q_dim=8 with W128 (4 lanes): 8/4=2 vec iterations
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W128, q_input,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn lower_qtap_stg_rejects_q_dim_not_aligned_to_w512() {
        // q_dim=12, W512 (16 lanes): 12 % 16 != 0
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 12,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W512, q_input,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("not divisible by SIMD lanes"));
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    #[test]
    fn lower_qtap_stg_emits_comment_instruction() {
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        assert!(result.is_ok());
        let has_comment = prog.instrs.iter().any(|i| matches!(i, VmInstr::Comment(_)));
        assert!(has_comment, "program should contain a Comment instruction");
    }

    #[test]
    fn lower_qtap_stg_atomic_add_at_end() {
        let mut prog = VmProgram::new();
        let q_input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let result = lower_qtap_stg(
            &mut prog, 0x1000, 0x2000, DType::F32, 8,
            BoundExpr::Const(1), QTapPosition::LastToken, 2,
            SimdWidth::W256, q_input,
        );
        assert!(result.is_ok());
        // The last non-Comment instruction should be AtomicAdd
        let last_real = prog.instrs.iter().rev().find(|i| !matches!(i, VmInstr::Comment(_)));
        assert!(last_real.is_some(), "should have at least one non-Comment instruction");
        assert!(
            matches!(last_real.unwrap(), VmInstr::AtomicAdd { .. }),
            "last real instruction should be AtomicAdd"
        );
    }
}
