//! StructuralOpBuilder — 符号化 structural op emit（lower_op_v2 驱动）。
//!
//! 提供 side-channel copy、SIMD injection、logit write-back 等
//! 常见模式的符号化构造方法，消除手工 VmInstr 序列中的计算错误。

use super::instr::*;
use super::auto_select;
use crate::compiler::trace::{QuantPrecision, TraceOp, ValueId};
use crate::types::CompilerError;

/// Symbolic builder for structural VmInstr patterns.
/// Replaces manual alloc_vreg + emit sequences with declarative calls.
pub struct StructuralOpBuilder;

impl StructuralOpBuilder {
    /// Emit a side-channel copy: hidden[0..dim] → base[offset..offset+dim].
    /// Used by SgDetect (copy hidden to SgSharedMemory).
    pub fn emit_side_channel_copy(
        prog: &mut VmProgram,
        src_base: VRegId,
        dst_base: VRegId,
        dst_offset: usize,
        hidden_dim: usize,
        width: SimdWidth,
    ) -> Result<(), CompilerError> {
        let step = width.f32_lanes();
        let byte_step = step * 4;
        let iterations = (hidden_dim + step - 1) / step;
        let ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter: ctr, byte_offset: byte_off,
            bound: BoundExpr::Const(iterations), step_bytes: byte_step,
        });
        let vec = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecLoad {
            dst: vec, base: src_base,
            offset: OffsetExpr::LoopOffset(byte_off), width,
            dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecStore {
            base: dst_base, src: vec,
            offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(dst_offset)),
                Box::new(OffsetExpr::LoopOffset(byte_off)),
            ),
            width,
            dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::LoopEnd);
        Ok(())
    }

    /// Emit SIMD injection: hidden[i] += confidence × knowledge_vector[i].
    /// Used by SgInject.
    pub fn emit_simd_injection(
        prog: &mut VmProgram,
        hidden_base: VRegId,
        knowledge_base: VRegId,
        conf_offset: usize,
        kv_offset: usize,
        hidden_dim: usize,
        width: SimdWidth,
    ) -> Result<(), CompilerError> {
        let step = width.f32_lanes();
        let byte_step = step * 4;
        let iterations = (hidden_dim + step - 1) / step;
        let ctr = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter: ctr, byte_offset: byte_off,
            bound: BoundExpr::Const(iterations), step_bytes: byte_step,
        });
        // Load knowledge_vector
        let kv_vec = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecLoad {
            dst: kv_vec, base: knowledge_base,
            offset: OffsetExpr::Add(
                Box::new(OffsetExpr::Const(kv_offset)),
                Box::new(OffsetExpr::LoopOffset(byte_off)),
            ),
            width,
            dtype: QuantPrecision::F32, predicate: None,
        });
        // Broadcast confidence
        let conf_bc = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast {
            dst: conf_bc,
            src: ScalarExpr::MemLoad(knowledge_base, OffsetExpr::Const(conf_offset)),
            width,
            dtype: QuantPrecision::F32,
        });
        // conf * kv + hidden (FMA via auto_lower_trace_raw)
        let hidden_vec = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::VecLoad {
            dst: hidden_vec, base: hidden_base,
            offset: OffsetExpr::LoopOffset(byte_off), width,
            dtype: QuantPrecision::F32, predicate: None,
        });
        let fma_body = [
            TraceOp::Input(0),  // conf_bc (slot 0)
            TraceOp::Input(1),  // kv_vec (slot 1)
            TraceOp::Mul(ValueId(0), ValueId(1)), // slot 2 = conf * kv
            TraceOp::Input(2),  // hidden_vec (slot 3)
            TraceOp::Add(ValueId(2), ValueId(3)), // slot 4 = hidden + conf*kv
        ];
        let fma_slots = auto_select::auto_lower_trace_raw(
            prog, &fma_body, &[conf_bc, kv_vec, hidden_vec], width, QuantPrecision::F32)?;
        let result = *fma_slots.last().ok_or_else(|| CompilerError::CodegenViolation(
            "emit_simd_injection: auto_lower_trace_raw returned empty".into(),
        ))?;
        prog.emit(VmInstr::VecStore {
            base: hidden_base, src: result,
            offset: OffsetExpr::LoopOffset(byte_off), width,
            dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::LoopEnd);
        Ok(())
    }

    /// Emit scalar logit write-back: for each target token, copy its logit to output.
    /// Used by WriteLogits and Argmax output.
    pub fn emit_scalar_writeback(
        prog: &mut VmProgram,
        logits_base: VRegId,
        output_base: VRegId,
        target_indices: &[u32],
    ) -> Result<(), CompilerError> {
        for (i, &token_idx) in target_indices.iter().enumerate() {
            let byte_offset = token_idx as usize * 4;
            let out_byte_offset = i * 4;
            let logit_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
            prog.emit(VmInstr::ScalarLoad {
                dst: logit_val, base: logits_base,
                offset: OffsetExpr::Const(byte_offset),
            });
            prog.emit(VmInstr::ScalarStore {
                base: output_base, src: logit_val,
                offset: OffsetExpr::Const(out_byte_offset),
            });
        }
        Ok(())
    }

    /// Emit conditional guard: load u32 flag from scratchpad at `probe_offset`,
    /// if flag != 0 → exit with output=victim_ptr.
    /// Used by GuardrailCheck and CotStepCheck.
    pub fn emit_conditional_guard(
        prog: &mut VmProgram,
        scratch_base: VRegId,
        probe_offset: usize,
        victim_ptr: VRegId,
    ) -> Result<(), CompilerError> {
        let flag = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::ScalarLoad {
            dst: flag,
            base: scratch_base,
            offset: OffsetExpr::Const(probe_offset),
        });
        prog.emit(VmInstr::ConditionalExit {
            condition: flag,
            output: victim_ptr,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::vm::instr::*;

    // ── 1. emit_side_channel_copy basic ───────────────────────────────

    #[test]
    fn side_channel_copy_emits_loop_load_store() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 16, SimdWidth::W256,
        ).unwrap();

        // Should have: LoopBegin, VecLoad, VecStore, LoopEnd
        let has_loop_begin = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        let has_vec_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecLoad { .. }));
        let has_vec_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::VecStore { .. }));
        let has_loop_end = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopEnd));

        assert!(has_loop_begin, "should emit LoopBegin");
        assert!(has_vec_load, "should emit VecLoad");
        assert!(has_vec_store, "should emit VecStore");
        assert!(has_loop_end, "should emit LoopEnd");
    }

    // ── 2. emit_side_channel_copy with nonzero offset ────────────────

    #[test]
    fn side_channel_copy_with_offset_uses_add_expr() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 4096, 32, SimdWidth::W256,
        ).unwrap();

        let store = prog.instrs.iter().find_map(|i| match i {
            VmInstr::VecStore { offset, .. } => Some(offset.clone()),
            _ => None,
        }).expect("should have VecStore");
        assert!(matches!(store, OffsetExpr::Add(_, _)), "store offset should be Add(Const, LoopOffset)");
    }

    // ── 3. emit_side_channel_copy hidden_dim=1 scalar ────────────────

    #[test]
    fn side_channel_copy_single_element() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 1, SimdWidth::Scalar,
        ).unwrap();

        // 1 element with scalar width = 1 iteration
        assert!(prog.instrs.len() >= 4, "should have at least 4 instructions for single element copy");
    }

    // ── 4. emit_simd_injection basic structure ────────────────────────

    #[test]
    fn simd_injection_emits_fma_pattern() {
        let mut prog = VmProgram::new();
        let hidden = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let knowledge = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_simd_injection(
            &mut prog, hidden, knowledge, 0, 0, 8, SimdWidth::W256,
        ).unwrap();

        let has_broadcast = prog.instrs.iter().any(|i| matches!(i, VmInstr::Broadcast { .. }));
        let has_loop = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopBegin { .. }));
        assert!(has_broadcast, "should emit Broadcast for confidence");
        assert!(has_loop, "should emit loop for hidden_dim iteration");
    }

    // ── 5. emit_simd_injection with offsets ───────────────────────────

    #[test]
    fn simd_injection_nonzero_offsets_in_kv_load() {
        let mut prog = VmProgram::new();
        let hidden = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let knowledge = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_simd_injection(
            &mut prog, hidden, knowledge, 1024, 2048, 16, SimdWidth::W256,
        ).unwrap();

        // At least one VecLoad should use an Add offset (for knowledge vector)
        let has_add_offset = prog.instrs.iter().any(|i| match i {
            VmInstr::VecLoad { offset: OffsetExpr::Add(_, _), .. } => true,
            _ => false,
        });
        assert!(has_add_offset, "knowledge vector load should use Add offset");
    }

    // ── 6. emit_scalar_writeback empty indices ────────────────────────

    #[test]
    fn scalar_writeback_empty_indices_no_new_instrs() {
        let mut prog = VmProgram::new();
        let logits = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let baseline = prog.instrs.len();

        StructuralOpBuilder::emit_scalar_writeback(&mut prog, logits, output, &[]).unwrap();

        assert_eq!(prog.instrs.len(), baseline, "empty indices should emit no new instructions");
    }

    // ── 7. emit_scalar_writeback single index ─────────────────────────

    #[test]
    fn scalar_writeback_single_index_loads_and_stores() {
        let mut prog = VmProgram::new();
        let logits = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_scalar_writeback(&mut prog, logits, output, &[42]).unwrap();

        // Should contain ScalarLoad with offset=42*4 and ScalarStore with offset=0
        let has_load = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::ScalarLoad { offset: OffsetExpr::Const(168), .. }));
        let has_store = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::ScalarStore { offset: OffsetExpr::Const(0), .. }));
        assert!(has_load, "should have ScalarLoad at offset 168 (=42*4)");
        assert!(has_store, "should have ScalarStore at offset 0");
    }

    // ── 8. emit_scalar_writeback multiple indices ─────────────────────

    #[test]
    fn scalar_writeback_three_indices_correct_offsets() {
        let mut prog = VmProgram::new();
        let logits = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_scalar_writeback(
            &mut prog, logits, output, &[0, 100, 999],
        ).unwrap();

        // 3 ScalarLoad + 3 ScalarStore = 6 new instrs (plus DeclareVReg for logit_val)
        let scalar_loads: Vec<_> = prog.instrs.iter().filter_map(|i| match i {
            VmInstr::ScalarLoad { offset: OffsetExpr::Const(off), .. } => Some(*off),
            _ => None,
        }).collect();
        let scalar_stores: Vec<_> = prog.instrs.iter().filter_map(|i| match i {
            VmInstr::ScalarStore { offset: OffsetExpr::Const(off), .. } => Some(*off),
            _ => None,
        }).collect();

        assert_eq!(scalar_loads.len(), 3, "should have 3 ScalarLoad");
        assert_eq!(scalar_stores.len(), 3, "should have 3 ScalarStore");
        assert_eq!(scalar_stores[1], 4, "second output at byte offset 4 (=1*4)");
    }

    // ── 9. emit_conditional_guard emits load_and_exit ─────────────────

    #[test]
    fn conditional_guard_emits_scalar_load_and_exit() {
        let mut prog = VmProgram::new();
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let victim = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_conditional_guard(
            &mut prog, scratch, 512, victim,
        ).unwrap();

        let has_load = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::ScalarLoad { offset: OffsetExpr::Const(512), .. }));
        let has_exit = prog.instrs.iter().any(|i| matches!(i, VmInstr::ConditionalExit { .. }));

        assert!(has_load, "should have ScalarLoad at probe_offset 512");
        assert!(has_exit, "should have ConditionalExit");
    }

    // ── 10. emit_conditional_guard zero offset ────────────────────────

    #[test]
    fn conditional_guard_zero_probe_offset() {
        let mut prog = VmProgram::new();
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let victim = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_conditional_guard(
            &mut prog, scratch, 0, victim,
        ).unwrap();

        let has_zero_load = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::ScalarLoad { offset: OffsetExpr::Const(0), .. }));
        assert!(has_zero_load, "should have ScalarLoad at offset 0");
    }

    // ── 11. StructuralOpBuilder is unit struct ────────────────────────

    #[test]
    fn structural_op_builder_is_unit() {
        let _builder = StructuralOpBuilder;
        // Unit struct: no fields, can be instantiated without {}
    }

    // ── 12. emit_side_channel_copy respects simd width ────────────────

    #[test]
    fn side_channel_copy_w512_fewer_declare_vregs() {
        let mut prog_w256 = VmProgram::new();
        let src2 = prog_w256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst2 = prog_w256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let mut prog_w512 = VmProgram::new();
        let src5 = prog_w512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst5 = prog_w512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog_w256, src2, dst2, 0, 64, SimdWidth::W256,
        ).unwrap();
        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog_w512, src5, dst5, 0, 64, SimdWidth::W512,
        ).unwrap();

        // Same instruction count (only loop bound differs)
        assert_eq!(prog_w256.instrs.len(), prog_w512.instrs.len(),
            "both should have same instruction count, just different loop bounds");
    }

    // ── 13. emit_scalar_writeback large_index_correct_offset ──────────

    #[test]
    fn scalar_writeback_large_index_correct_byte_offset() {
        let mut prog = VmProgram::new();
        let logits = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_scalar_writeback(
            &mut prog, logits, output, &[u32::MAX],
        ).unwrap();

        let expected_offset = u32::MAX as usize * 4;
        let has_load = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::ScalarLoad { offset: OffsetExpr::Const(off), .. } if *off == expected_offset));
        assert!(has_load, "should have ScalarLoad at byte offset u32::MAX * 4");
    }

    // ── 14. emit_side_channel_copy loop bound varies with width ──────────

    #[test]
    fn side_channel_copy_loop_bound_varies_with_width() {
        let mut prog_w128 = VmProgram::new();
        let src1 = prog_w128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst1 = prog_w128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let mut prog_w256 = VmProgram::new();
        let src2 = prog_w256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst2 = prog_w256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // hidden_dim=16: W128 (4 lanes) => 4 iterations, W256 (8 lanes) => 2 iterations
        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog_w128, src1, dst1, 0, 16, SimdWidth::W128,
        ).unwrap();
        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog_w256, src2, dst2, 0, 16, SimdWidth::W256,
        ).unwrap();

        let bound_w128 = prog_w128.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(n), .. } => Some(*n),
            _ => None,
        }).expect("should have LoopBegin with Const bound");
        let bound_w256 = prog_w256.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(n), .. } => Some(*n),
            _ => None,
        }).expect("should have LoopBegin with Const bound");

        assert_eq!(bound_w128, 4, "W128 with dim=16 should have 4 iterations (16/4)");
        assert_eq!(bound_w256, 2, "W256 with dim=16 should have 2 iterations (16/8)");
    }

    // ── 15. emit_side_channel_copy zero offset uses LoopOffset directly ───

    #[test]
    fn side_channel_copy_zero_offset_store_uses_loop_offset() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 8, SimdWidth::W256,
        ).unwrap();

        // When dst_offset=0, the store offset should be Add(Const(0), LoopOffset(_))
        let store_offset = prog.instrs.iter().find_map(|i| match i {
            VmInstr::VecStore { offset, .. } => Some(offset.clone()),
            _ => None,
        }).expect("should have VecStore");

        // With dst_offset=0, it's Add(Const(0), LoopOffset(_))
        assert!(matches!(store_offset, OffsetExpr::Add(_, _)),
            "store offset should be Add expression even with zero dst_offset");
    }

    // ── 16. emit_simd_injection produces VecStore result ─────────────────

    #[test]
    fn simd_injection_produces_vec_store_to_hidden() {
        let mut prog = VmProgram::new();
        let hidden = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let knowledge = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_simd_injection(
            &mut prog, hidden, knowledge, 0, 0, 8, SimdWidth::W256,
        ).unwrap();

        // The final VecStore should write back to hidden_base
        let has_store_to_hidden = prog.instrs.iter().any(|i| match i {
            VmInstr::VecStore { base, .. } if *base == hidden => true,
            _ => false,
        });
        assert!(has_store_to_hidden, "should have VecStore writing back to hidden_base");
    }

    // ── 17. emit_simd_injection scalar width produces correct iterations ──

    #[test]
    fn simd_injection_scalar_width_single_iteration() {
        let mut prog = VmProgram::new();
        let hidden = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let knowledge = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // hidden_dim=1 with Scalar width => 1 iteration
        StructuralOpBuilder::emit_simd_injection(
            &mut prog, hidden, knowledge, 0, 0, 1, SimdWidth::Scalar,
        ).unwrap();

        let bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(n), .. } => Some(*n),
            _ => None,
        }).expect("should have LoopBegin");
        assert_eq!(bound, 1, "Scalar width with dim=1 should have 1 iteration");
    }

    // ── 18. emit_conditional_guard uses correct scratch_base ─────────────

    #[test]
    fn conditional_guard_loads_from_correct_base() {
        let mut prog = VmProgram::new();
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let victim = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_conditional_guard(
            &mut prog, scratch, 256, victim,
        ).unwrap();

        // ScalarLoad should use scratch as base
        let load_base = prog.instrs.iter().find_map(|i| match i {
            VmInstr::ScalarLoad { base, .. } => Some(*base),
            _ => None,
        }).expect("should have ScalarLoad");
        assert_eq!(load_base, scratch, "ScalarLoad should read from scratch_base");

        // ConditionalExit should use victim as output
        let exit_output = prog.instrs.iter().find_map(|i| match i {
            VmInstr::ConditionalExit { output, .. } => Some(*output),
            _ => None,
        }).expect("should have ConditionalExit");
        assert_eq!(exit_output, victim, "ConditionalExit should use victim as output");
    }

    // ── 19. emit_side_channel_copy step_bytes matches width ──────────────

    #[test]
    fn side_channel_copy_step_bytes_matches_width() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // W512 = 16 f32 lanes = 64 bytes per step
        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 64, SimdWidth::W512,
        ).unwrap();

        let step_bytes = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { step_bytes, .. } => Some(*step_bytes),
            _ => None,
        }).expect("should have LoopBegin");
        assert_eq!(step_bytes, 64, "W512 step_bytes should be 64 (16 lanes * 4 bytes)");
    }

    // ── 20. emit_scalar_writeback index_zero_correct_offsets ─────────────

    #[test]
    fn scalar_writeback_index_zero_correct_offsets() {
        let mut prog = VmProgram::new();
        let logits = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Index 0: load from logits+0, store to output+0
        StructuralOpBuilder::emit_scalar_writeback(
            &mut prog, logits, output, &[0],
        ).unwrap();

        let has_load_at_zero = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::ScalarLoad { base, offset: OffsetExpr::Const(0), .. } if *base == logits));
        let has_store_at_zero = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::ScalarStore { base, offset: OffsetExpr::Const(0), .. } if *base == output));

        assert!(has_load_at_zero, "should have ScalarLoad from logits at offset 0");
        assert!(has_store_at_zero, "should have ScalarStore to output at offset 0");
    }

    // ── 21. emit_side_channel_copy allocates_counter_and_byte_offset ─────

    #[test]
    fn side_channel_copy_allocates_counter_and_byte_offset_vregs() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 8, SimdWidth::W256,
        ).unwrap();

        // Should have DeclareVReg for Counter and ByteOffset kinds
        let has_counter = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::DeclareVReg { kind: VRegKind::Counter, .. }));
        let has_byte_offset = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::DeclareVReg { kind: VRegKind::ByteOffset, .. }));

        assert!(has_counter, "should declare a Counter VReg for the loop");
        assert!(has_byte_offset, "should declare a ByteOffset VReg for the loop");
    }

    // ── 22. emit_simd_injection loop_end_present ─────────────────────────

    #[test]
    fn simd_injection_loop_end_present() {
        let mut prog = VmProgram::new();
        let hidden = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let knowledge = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_simd_injection(
            &mut prog, hidden, knowledge, 0, 0, 8, SimdWidth::W256,
        ).unwrap();

        // Should end with LoopEnd
        let has_loop_end = prog.instrs.iter().any(|i| matches!(i, VmInstr::LoopEnd));
        assert!(has_loop_end, "should have LoopEnd closing the injection loop");

        // Verify LoopBegin comes before LoopEnd
        let loop_begin_idx = prog.instrs.iter().position(|i| matches!(i, VmInstr::LoopBegin { .. }));
        let loop_end_idx = prog.instrs.iter().position(|i| matches!(i, VmInstr::LoopEnd));
        assert!(loop_begin_idx.is_some() && loop_end_idx.is_some(),
            "both LoopBegin and LoopEnd should exist");
        assert!(loop_begin_idx.unwrap() < loop_end_idx.unwrap(),
            "LoopBegin should come before LoopEnd");
    }

    // ── 23. emit_conditional_guard emits_exactly_two_instrs ──────────────

    #[test]
    fn conditional_guard_emits_exactly_two_non_declare_instrs() {
        let mut prog = VmProgram::new();
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let victim = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let baseline = prog.instrs.len();

        StructuralOpBuilder::emit_conditional_guard(
            &mut prog, scratch, 128, victim,
        ).unwrap();

        // Should emit: DeclareVReg(flag) + ScalarLoad + ConditionalExit
        // Non-DeclareVReg instructions = 2 (ScalarLoad + ConditionalExit)
        let non_declare_count = prog.instrs[baseline..].iter()
            .filter(|i| !matches!(i, VmInstr::DeclareVReg { .. }))
            .count();
        assert_eq!(non_declare_count, 2,
            "conditional_guard should emit exactly 2 non-declare instructions (ScalarLoad + ConditionalExit)");
    }

    // ── 24. emit_side_channel_copy_w128_iterations ─────────────────────────

    #[test]
    fn side_channel_copy_w128_correct_iterations() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // hidden_dim=9 with W128 (4 lanes) => ceil(9/4) = 3 iterations
        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 9, SimdWidth::W128,
        ).unwrap();

        let bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(n), .. } => Some(*n),
            _ => None,
        }).expect("should have LoopBegin");
        assert_eq!(bound, 3, "ceil(9/4) should be 3 iterations");
    }

    // ── 25. emit_simd_injection_w128_iterations ────────────────────────────

    #[test]
    fn simd_injection_w128_correct_iterations() {
        let mut prog = VmProgram::new();
        let hidden = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let knowledge = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // hidden_dim=17 with W128 (4 lanes) => ceil(17/4) = 5 iterations
        StructuralOpBuilder::emit_simd_injection(
            &mut prog, hidden, knowledge, 0, 0, 17, SimdWidth::W128,
        ).unwrap();

        let bound = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { bound: BoundExpr::Const(n), .. } => Some(*n),
            _ => None,
        }).expect("should have LoopBegin");
        assert_eq!(bound, 5, "ceil(17/4) should be 5 iterations");
    }

    // ── 26. emit_scalar_writeback_sequential_output_offsets ────────────────

    #[test]
    fn scalar_writeback_sequential_output_byte_offsets() {
        let mut prog = VmProgram::new();
        let logits = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_scalar_writeback(
            &mut prog, logits, output, &[10, 20, 30],
        ).unwrap();

        let store_offsets: Vec<usize> = prog.instrs.iter().filter_map(|i| match i {
            VmInstr::ScalarStore { offset: OffsetExpr::Const(off), .. } => Some(*off),
            _ => None,
        }).collect();

        assert_eq!(store_offsets, vec![0, 4, 8],
            "output offsets should be 0, 4, 8 for 3 sequential outputs");
    }

    // ── 27. emit_side_channel_copy_vec_load_src_base ────────────────────────

    #[test]
    fn side_channel_copy_vec_load_uses_src_base() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 8, SimdWidth::W256,
        ).unwrap();

        let load_base = prog.instrs.iter().find_map(|i| match i {
            VmInstr::VecLoad { base, .. } => Some(*base),
            _ => None,
        }).expect("should have VecLoad");
        assert_eq!(load_base, src, "VecLoad should read from src_base");

        let store_base = prog.instrs.iter().find_map(|i| match i {
            VmInstr::VecStore { base, .. } => Some(*base),
            _ => None,
        }).expect("should have VecStore");
        assert_eq!(store_base, dst, "VecStore should write to dst_base");
    }

    // ── 28. emit_simd_injection_broadcast_uses_conf_offset ─────────────────

    #[test]
    fn simd_injection_broadcast_uses_conf_offset() {
        let mut prog = VmProgram::new();
        let hidden = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let knowledge = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_simd_injection(
            &mut prog, hidden, knowledge, 512, 1024, 8, SimdWidth::W256,
        ).unwrap();

        // Broadcast should load confidence from knowledge_base at conf_offset
        let broadcast_src = prog.instrs.iter().find_map(|i| match i {
            VmInstr::Broadcast { src: ScalarExpr::MemLoad(base, OffsetExpr::Const(off)), .. }
                if *off == 512 => Some(*base),
            _ => None,
        });
        assert!(broadcast_src.is_some(),
            "Broadcast should load confidence from knowledge_base at conf_offset 512");
    }

    // ── 29. emit_side_channel_copy_w512_step_bytes ──────────────────────────

    #[test]
    fn side_channel_copy_w512_correct_step_bytes() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 32, SimdWidth::W512,
        ).unwrap();

        let step_bytes = prog.instrs.iter().find_map(|i| match i {
            VmInstr::LoopBegin { step_bytes, .. } => Some(*step_bytes),
            _ => None,
        }).expect("should have LoopBegin with step_bytes");

        // W512 = 16 f32 lanes * 4 bytes = 64
        assert_eq!(step_bytes, 64, "W512 step_bytes must be 64");
    }

    // ── 30. emit_conditional_guard_declares_scalar_vreg ────────────────────

    #[test]
    fn conditional_guard_declares_scalar_kind_vreg() {
        let mut prog = VmProgram::new();
        let scratch = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let victim = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_conditional_guard(
            &mut prog, scratch, 64, victim,
        ).unwrap();

        let has_scalar_declare = prog.instrs.iter().any(|i| match i {
            VmInstr::DeclareVReg { kind: VRegKind::Scalar, width: SimdWidth::Scalar, .. } => true,
            _ => false,
        });
        assert!(has_scalar_declare,
            "conditional_guard should declare a Scalar vreg for the flag");
    }

    // ── 31. emit_scalar_writeback_two_indices_correct_load_offsets ──────────

    #[test]
    fn scalar_writeback_two_indices_correct_load_byte_offsets() {
        let mut prog = VmProgram::new();
        let logits = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_scalar_writeback(
            &mut prog, logits, output, &[7, 13],
        ).unwrap();

        let load_offsets: Vec<usize> = prog.instrs.iter().filter_map(|i| match i {
            VmInstr::ScalarLoad { offset: OffsetExpr::Const(off), .. } => Some(*off),
            _ => None,
        }).collect();

        assert_eq!(load_offsets, vec![28, 52],
            "load offsets should be 7*4=28 and 13*4=52");
    }

    // ── 32. emit_side_channel_copy_loop_body_instruction_order ─────────────

    #[test]
    fn side_channel_copy_loop_body_order_is_load_then_store() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        StructuralOpBuilder::emit_side_channel_copy(
            &mut prog, src, dst, 0, 8, SimdWidth::W256,
        ).unwrap();

        // Find positions: LoopBegin, first VecLoad, first VecStore, LoopEnd
        let pos = |instrs: &[VmInstr], pred: &dyn Fn(&VmInstr) -> bool| -> Option<usize> {
            instrs.iter().position(|i| pred(i))
        };

        let loop_begin = pos(&prog.instrs, &|i| matches!(i, VmInstr::LoopBegin { .. })).unwrap();
        let vec_load = pos(&prog.instrs, &|i| matches!(i, VmInstr::VecLoad { .. })).unwrap();
        let vec_store = pos(&prog.instrs, &|i| matches!(i, VmInstr::VecStore { .. })).unwrap();
        let loop_end = pos(&prog.instrs, &|i| matches!(i, VmInstr::LoopEnd)).unwrap();

        assert!(loop_begin < vec_load, "LoopBegin before VecLoad");
        assert!(vec_load < vec_store, "VecLoad before VecStore");
        assert!(vec_store < loop_end, "VecStore before LoopEnd");
    }
}
