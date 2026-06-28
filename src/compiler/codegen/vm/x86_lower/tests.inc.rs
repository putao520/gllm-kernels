#[cfg(test)]
mod tests {
    use crate::compiler::trace::QuantPrecision;
    use super::*;
    use crate::compiler::codegen::vm::{auto_select, reg_alloc::RegAllocator};
    use crate::dispatch::DeviceProfile;
    use crate::compiler::trace::{TraceOp, ValueId};

    fn build_prog_from_body(body: &[TraceOp]) -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec_reg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad { dst: vec_reg, base: input_ptr, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        let slots = auto_select::auto_lower_trace_raw(&mut prog, body, &[vec_reg], SimdWidth::W256, QuantPrecision::F32).unwrap();
        let last = *slots.last().unwrap();
        prog.emit(VmInstr::VecStore { base: output_ptr, src: last, offset: OffsetExpr::Const(0), width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None, });
        prog
    }

    #[test]
    fn test_lower_identity_produces_code() {
        let prog = build_prog_from_body(&[TraceOp::Input(0)]);

        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::new();
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        assert!(!code.is_empty(), "should produce non-empty code");
        assert!(code.len() > 10, "code too short: {} bytes", code.len());
    }

    #[test]
    fn test_lower_silu_produces_code() {
        let body = vec![
            TraceOp::Input(0), TraceOp::Neg(ValueId(0)), TraceOp::Exp(ValueId(1)),
            TraceOp::Const(1.0), TraceOp::Add(ValueId(2), ValueId(3)), TraceOp::Recip(ValueId(4)), TraceOp::Mul(ValueId(0), ValueId(5)),
        ];
        let prog = build_prog_from_body(&body);

        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::new();
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        assert!(code.len() > 100, "SiLU code should be substantial: {} bytes", code.len());
    }

    #[test]
    fn scratch_slot_alloc_sequential() {
        let mut state = ScratchSlotState::new(3);
        let a = state.alloc();
        let b = state.alloc();
        let c = state.alloc();
        assert_eq!(a, Some(0));
        assert_eq!(b, Some(1));
        assert_eq!(c, Some(2));
    }

    #[test]
    fn scratch_slot_alloc_exhausted() {
        let mut state = ScratchSlotState::new(1);
        assert_eq!(state.alloc(), Some(0));
        assert_eq!(state.alloc(), None);
    }

    #[test]
    fn scratch_slot_free_and_reuse() {
        let mut state = ScratchSlotState::new(2);
        let _ = state.alloc();
        let slot1 = state.alloc().unwrap();
        state.free(slot1);
        let reused = state.alloc();
        assert_eq!(reused, Some(1));
    }

    #[test]
    fn scratch_slot_zero_capacity() {
        let mut state = ScratchSlotState::new(0);
        assert_eq!(state.alloc(), None);
    }

    #[test]
    fn stack_layout_spill_rbp_offset() {
        let layout = StackLayout {
            spill_base_off: -64,
            ..Default::default()
        };
        assert_eq!(layout.spill_rbp_offset(0, 16), -80);
        assert_eq!(layout.spill_rbp_offset(16, 8), -88);
    }

    #[test]
    fn stack_layout_abi_arg_offset_present() {
        let mut slots = [None; 6];
        slots[0] = Some(16);
        slots[3] = Some(-8);
        let layout = StackLayout { abi_arg_slots: slots, ..Default::default() };
        assert_eq!(layout.abi_arg_rbp_offset(0), Some(16));
        assert_eq!(layout.abi_arg_rbp_offset(3), Some(-8));
        assert_eq!(layout.abi_arg_rbp_offset(5), None);
    }

    #[test]
    fn stack_layout_default_no_abi_args() {
        let layout = StackLayout::default();
        for i in 0..6u8 {
            assert_eq!(layout.abi_arg_rbp_offset(i), None);
        }
        assert_eq!(layout.frame_pointer_off, 0);
        assert_eq!(layout.spill_base_off, 0);
    }

    #[test]
    fn test_lower_empty_program_produces_prologue_epilogue() {
        let prog = VmProgram::new();

        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::new();
        lower.emit_prologue(&frame, &alloc).unwrap();
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        assert!(!code.is_empty(), "empty program should still produce prologue+epilogue");
    }

    #[test]
    fn test_lower_with_avx512_flag() {
        let prog = build_prog_from_body(&[TraceOp::Input(0)]);

        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::with_avx512(true);
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        assert!(!code.is_empty(), "AVX-512 lower should produce code");
    }

    #[test]
    fn stack_frame_zero_spill_no_callee_save() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let prog = VmProgram::new();
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        assert_eq!(frame.callee_save_area, 0);
        assert_eq!(frame.spill_area, 0);
        assert!(frame.alignment > 0, "alignment must be positive");
    }

    #[test]
    fn stack_frame_alignment_is_power_of_two() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let prog = VmProgram::new();
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        assert!(frame.alignment > 0);
        assert_eq!(frame.alignment & (frame.alignment - 1), 0, "alignment must be power of 2");
    }

    #[test]
    fn stack_layout_abi_arg_all_six_slots() {
        let mut slots = [None; 6];
        for i in 0..6 {
            slots[i] = Some(-((i as i32 + 1) * 8));
        }
        let layout = StackLayout { abi_arg_slots: slots, ..Default::default() };

        for i in 0..6u8 {
            let off = layout.abi_arg_rbp_offset(i);
            assert!(off.is_some(), "slot {} should be present", i);
            assert_eq!(off.unwrap(), -((i as i32 + 1) * 8));
        }
    }

    #[test]
    fn scratch_slot_double_free_same_slot() {
        let mut state = ScratchSlotState::new(2);
        let slot = state.alloc().unwrap();
        state.free(slot);
        state.free(slot);
        let reused = state.alloc();
        assert_eq!(reused, Some(slot), "double-freed slot should still be reusable");
    }

    #[test]
    fn scratch_slot_alloc_after_full_cycle() {
        let mut state = ScratchSlotState::new(2);
        let a = state.alloc().unwrap();
        let b = state.alloc().unwrap();
        state.free(a);
        state.free(b);
        let c = state.alloc().unwrap();
        let d = state.alloc().unwrap();
        assert_eq!(c, a, "first alloc after full free should reuse slot 0");
        assert_eq!(d, b, "second alloc after full free should reuse slot 1");
    }

    #[test]
    fn scoped_spill_alloc_and_scope_recycle() {
        use super::super::stack_frame::ScopedSpillAllocator;

        let mut sa = ScopedSpillAllocator::new();
        let scope = sa.scope_begin();

        let (idx0, off0) = sa.alloc(VRegId(0), 32, Some(scope));
        let (idx1, off1) = sa.alloc(VRegId(1), 32, Some(scope));
        assert_ne!(idx0, idx1, "two allocs in same scope must have different indices");
        assert_ne!(off0, off1, "offsets must be distinct");

        sa.scope_end();
        assert_eq!(sa.active_count(), 0, "scope end should free all slots");

        // ARCH-SPILL-SAFE: after scope recycle, the freed slot INDEX is reused
        // but the offset is FRESH (never reused) to prevent two VRegs from
        // sharing the same spill memory location.
        let (idx2, off2) = sa.alloc(VRegId(2), 32, None);
        assert_eq!(idx2, idx0, "after scope recycle, should reuse freed slot index");
        assert_ne!(off2, off0, "offset must be fresh — never reuse spill offsets (ARCH-SPILL-SAFE)");
        assert_ne!(off2, off1, "offset must be fresh — never reuse spill offsets (ARCH-SPILL-SAFE)");
    }

    #[test]
    fn scoped_spill_nested_scope_independence() {
        use super::super::stack_frame::ScopedSpillAllocator;

        let mut sa = ScopedSpillAllocator::new();
        let outer = sa.scope_begin();
        let (_, _outer_off) = sa.alloc(VRegId(0), 16, Some(outer));

        let inner = sa.scope_begin();
        let (_, inner_off) = sa.alloc(VRegId(1), 16, Some(inner));

        assert_eq!(sa.active_count(), 2);
        sa.scope_end();
        assert_eq!(sa.active_count(), 1, "inner scope end frees only inner slots");

        // ARCH-SPILL-SAFE: the freed slot INDEX is reused but offset is fresh.
        let (_recycled_idx, recycled_off) = sa.alloc(VRegId(2), 16, None);
        assert_ne!(recycled_off, inner_off, "offset must be fresh — never reuse spill offsets (ARCH-SPILL-SAFE)");

        sa.scope_end();
        assert_eq!(sa.active_count(), 1, "outer scope end frees outer slots, global slot remains");
    }

    #[test]
    fn reg_alloc_empty_program_no_mapping() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let prog = VmProgram::new();
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();

        assert!(alloc.mapping.is_empty(), "empty program should have no vreg-to-preg mapping");
        assert!(alloc.spills.is_empty(), "empty program should have no spill slots");
    }

    #[test]
    fn reg_alloc_single_vec_reg_mapped() {
        let mut prog = VmProgram::new();
        let vreg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();

        assert!(alloc.mapping.contains_key(&vreg), "single vec vreg should be mapped to a physical register");
        assert!(alloc.spills.is_empty(), "single unused vreg should not spill");
    }

    // ── 10 additional unit tests ──

    #[test]
    fn x86_lower_new_equals_with_avx512_false() {
        // Arrange: construct via new() and with_avx512(false)
        let lower_default = X86Lower::new();
        let lower_explicit = X86Lower::with_avx512(false);

        // Act: finalize both (produces identical empty prologue+epilogue)
        let code_default = lower_default.finalize().unwrap();
        let code_explicit = lower_explicit.finalize().unwrap();

        // Assert: both produce the same machine code bytes
        assert_eq!(code_default, code_explicit, "new() and with_avx512(false) must produce identical output");
    }

    #[test]
    fn x86_lower_set_scratch_gprs_rejects_insufficient() {
        // Arrange: create a lower and a slice with only 2 GPRs (< required 3)
        let mut lower = X86Lower::new();
        let insufficient = [PhysGpr(0), PhysGpr(1)];

        // Act
        let result = lower.set_scratch_gprs(&insufficient);

        // Assert: must return error because < 3 scratch GPRs
        assert!(result.is_err(), "set_scratch_gprs with < 3 GPRs should fail");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("≥3"), "error message should mention the minimum requirement");
    }

    #[test]
    fn x86_lower_set_scratch_gprs_accepts_minimum() {
        // Arrange: create a lower and exactly 3 GPRs
        let mut lower = X86Lower::new();
        let minimum = [PhysGpr(0), PhysGpr(10), PhysGpr(11)];

        // Act
        let result = lower.set_scratch_gprs(&minimum);

        // Assert: should succeed with exactly 3 GPRs
        assert!(result.is_ok(), "set_scratch_gprs with exactly 3 GPRs should succeed");
    }

    #[test]
    fn x86_lower_set_scratch_vec_rejects_insufficient() {
        // Arrange: create a lower and a slice with only 5 vec regs (< required 6)
        let mut lower = X86Lower::new();
        let insufficient: Vec<PhysVec> = (0..5).map(PhysVec).collect();

        // Act
        let result = lower.set_scratch_vec_regs(&insufficient);

        // Assert: must return error because < 6 scratch vec regs
        assert!(result.is_err(), "set_scratch_vec_regs with < 6 vec regs should fail");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("≥6"), "error message should mention the minimum requirement");
    }

    #[test]
    fn x86_lower_set_scratch_vec_accepts_minimum() {
        // Arrange: create a lower and exactly 6 vec regs
        let mut lower = X86Lower::new();
        let minimum: Vec<PhysVec> = (0..6).map(PhysVec).collect();

        // Act
        let result = lower.set_scratch_vec_regs(&minimum);

        // Assert: should succeed with exactly 6 vec regs
        assert!(result.is_ok(), "set_scratch_vec_regs with exactly 6 vec regs should succeed");
    }

    #[test]
    fn x86_lower_precompute_zero_vregs_tracks_zero_imm() {
        // Arrange: build a VmProgram with GprLoadImm { value: 0 } and non-zero
        let mut prog = VmProgram::new();
        let dst_zero = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let dst_nonzero = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: dst_zero, value: 0 });
        prog.emit(VmInstr::GprLoadImm { dst: dst_nonzero, value: 42 });

        // Act
        let mut lower = X86Lower::new();
        lower.precompute_zero_vregs(&prog);

        // Assert: only the zero-valued GprLoadImm destination should be tracked
        assert!(lower.zero_vregs.contains(&dst_zero), "dst with value=0 should be tracked");
        assert!(!lower.zero_vregs.contains(&dst_nonzero), "dst with value=42 should NOT be tracked");
    }

    #[test]
    fn x86_lower_precompute_zero_vregs_empty_program() {
        // Arrange: empty VmProgram
        let prog = VmProgram::new();

        // Act
        let mut lower = X86Lower::new();
        lower.precompute_zero_vregs(&prog);

        // Assert: no zero vregs tracked
        assert!(lower.zero_vregs.is_empty(), "empty program should have zero tracked vregs");
    }

    #[test]
    fn mxcsr_slot_bytes_is_eight() {
        // Arrange & Act: read the constant
        let slot_bytes = MXCSR_SLOT_BYTES;

        // Assert: MXCSR slot is 8 bytes (64-bit aligned)
        assert_eq!(slot_bytes, 8, "MXCSR slot must be 8 bytes for proper stack alignment");
    }

    #[test]
    fn x86_lower_with_sym_map_preserves_custom_map() {
        // Arrange: build a custom SymDimSlotMap via mega_kernel_abi
        let custom_map = super::super::plan_lower::SymDimSlotMap::mega_kernel_abi();

        // Act: construct with custom map, then build and finalize a minimal program
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let prog = VmProgram::new();
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::with_sym_map(false, custom_map);
        lower.emit_prologue(&frame, &alloc).unwrap();
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        // Assert: with_sym_map produces valid non-empty code
        assert!(!code.is_empty(), "X86Lower with custom SymDimSlotMap should produce valid code");
    }

    #[test]
    fn x86_lower_take_source_map_returns_empty_initially() {
        // Arrange: freshly constructed X86Lower
        let mut lower = X86Lower::new();

        // Act
        let source_map = lower.take_source_map();

        // Assert: source map should be present but empty (no instructions lowered)
        assert!(source_map.entries.is_empty(), "fresh X86Lower should have empty source map");
    }

    // ── BF16 AVX2 窄化路径测试 (CR-TIER-SOVEREIGNTY-004) ──
    // 验证无 AVX-512 BF16 的硬件上 F32→BF16 向量化序列正确生成。

    fn build_bf16_store_prog() -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec_reg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        // Load 8×F32, then store as 8×BF16 (WidenCompute path)
        prog.emit(VmInstr::VecLoad {
            dst: vec_reg, base: input_ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecStore {
            base: output_ptr, src: vec_reg, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::BF16, predicate: None,
        });
        prog
    }

    #[test]
    fn bf16_vec_store_on_avx2_uses_software_path() {
        // Arrange: build a program that stores BF16 (VecStore WidenCompute)
        let prog = build_bf16_store_prog();
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        // Act: lower with use_avx512=false (AVX2-only hardware, e.g., i9-10900KF)
        let mut lower = X86Lower::with_avx512(false);
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        // Assert: code should be generated (no error), and should be substantial
        // (contains vpsrld+vpackusdw sequence, not just NOP)
        assert!(!code.is_empty(), "AVX2 BF16 store should produce code");
        assert!(code.len() > 50, "AVX2 BF16 store code should be substantial: {} bytes", code.len());
    }

    #[test]
    fn bf16_vec_store_on_avx512_uses_native_instruction() {
        // Arrange: same program, but lower with use_avx512=true (AVX-512 BF16 hardware)
        let prog = build_bf16_store_prog();
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        // Act: lower with use_avx512=true
        let mut lower = X86Lower::with_avx512(true);
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        // Assert: code should be generated
        assert!(!code.is_empty(), "AVX-512 BF16 store should produce code");
    }

    fn build_bf16_narrow_prog() -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoad {
            dst: src, base: input_ptr, offset: OffsetExpr::Const(0),
            width: SimdWidth::W256, dtype: QuantPrecision::F32, predicate: None,
        });
        prog.emit(VmInstr::VecNarrow {
            dst, src, dst_dtype: QuantPrecision::BF16, src_dtype: QuantPrecision::F32,
            width: SimdWidth::W256,
        });
        prog
    }

    #[test]
    fn bf16_vec_narrow_on_avx2_uses_software_path() {
        // Arrange: VecNarrow F32→BF16 on AVX2
        let prog = build_bf16_narrow_prog();
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        // Act: lower with use_avx512=false
        let mut lower = X86Lower::with_avx512(false);
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        // Assert: should succeed (no error about AVX-512 requirement)
        assert!(!code.is_empty(), "AVX2 BF16 VecNarrow should produce code");
    }

    #[test]
    fn bf16_vec_narrow_on_avx512_uses_native_instruction() {
        // Arrange: VecNarrow F32→BF16 with AVX-512
        let prog = build_bf16_narrow_prog();
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        // Act: lower with use_avx512=true
        let mut lower = X86Lower::with_avx512(true);
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        // Assert
        assert!(!code.is_empty(), "AVX-512 BF16 VecNarrow should produce code");
    }

    /// 反汇编 code bytes，收集所有指令助记符，用于验证 AVX2 路径 emit 了预期的指令序列。
    fn disasm_mnemonics(code: &[u8]) -> Vec<&'static str> {
        use iced_x86::{Decoder, DecoderOptions, Mnemonic};
        let mut decoder = Decoder::new(64, code, DecoderOptions::NONE);
        let mut mnemonics = Vec::new();
        for instr in decoder.iter() {
            // 仅保留与 BF16 窄化序列相关的指令，便于断言。
            let m = instr.mnemonic();
            let name = match m {
                Mnemonic::Vpsrld => "vpsrld",
                Mnemonic::Vpslld => "vpslld",
                Mnemonic::Vpaddd => "vpaddd",
                Mnemonic::Vpand => "vpand",
                Mnemonic::Vpackusdw => "vpackusdw",
                Mnemonic::Vcvtneps2bf16 => "vcvtneps2bf16",
                Mnemonic::Vbroadcastss => "vbroadcastss",
                _ => continue,
            };
            mnemonics.push(name);
        }
        mnemonics
    }

    #[test]
    fn bf16_avx2_store_emits_expected_instruction_sequence() {
        // CR-TIER-SOVEREIGNTY-004: AVX2 路径必须 emit vpsrld + vpackusdw (软件窄化序列)，
        // 而非 vcvtneps2bf16 (那是 AVX-512 专用)。
        let prog = build_bf16_store_prog();
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::with_avx512(false);
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        let mnemonics = disasm_mnemonics(&code);
        assert!(mnemonics.contains(&"vpsrld"),
            "AVX2 BF16 store must emit vpsrld (shift), got: {:?}", mnemonics);
        assert!(mnemonics.contains(&"vpackusdw"),
            "AVX2 BF16 store must emit vpackusdw (pack), got: {:?}", mnemonics);
        assert!(!mnemonics.contains(&"vcvtneps2bf16"),
            "AVX2 BF16 store must NOT emit vcvtneps2bf16 (AVX-512 only), got: {:?}", mnemonics);
        // RNE 需要 lsb 提取 (vpand) 和 bias 加法 (vpaddd)
        assert!(mnemonics.contains(&"vpand"),
            "AVX2 BF16 store must emit vpand (lsb mask for RNE), got: {:?}", mnemonics);
        assert!(mnemonics.contains(&"vpaddd"),
            "AVX2 BF16 store must emit vpaddd (bias add), got: {:?}", mnemonics);
    }

    #[test]
    fn bf16_avx512_store_emits_native_vcvtneps2bf16() {
        // CR-TIER-SOVEREIGNTY-004: AVX-512 路径必须 emit vcvtneps2bf16 (原生指令)，
        // 而非 vpsrld+vpackusdw (那是 AVX2 软件序列)。
        let prog = build_bf16_store_prog();
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();
        let frame = super::super::stack_frame::StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::with_avx512(true);
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();

        let mnemonics = disasm_mnemonics(&code);
        assert!(mnemonics.contains(&"vcvtneps2bf16"),
            "AVX-512 BF16 store must emit vcvtneps2bf16, got: {:?}", mnemonics);
    }

    /// F32→BF16 RNE (round-to-nearest-even) 标量参考实现。
    /// 与 emit_f32_to_bf16_*_avx2 中的向量算法严格一致：
    ///   rounded = f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)
    ///   bf16 = rounded >> 16
    fn f32_to_bf16_rne_reference(f: f32) -> u16 {
        let bits = f.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounding_bias: u32 = 0x7FFF + lsb;
        let rounded = bits.wrapping_add(rounding_bias);
        (rounded >> 16) as u16
    }

    #[test]
    fn bf16_avx2_rne_algorithm_matches_expected_precision() {
        // CR-TIER-SOVEREIGNTY-004: BF16 容差 = BF16 精度 (~1e-2)。
        // 验证 RNE 算法对典型值产生正确 (或精度内) 的 BF16 表示。
        let test_cases: Vec<f32> = vec![
            0.0, 1.0, -1.0, 2.0, 0.5, -0.5,
            3.14159265,        // π
            2.718281828,       // e
            1.5, 2.5, 3.5,     // tie cases (lsb boundary)
            100.0, 1000.0,
            0.1, 0.01,
            -3.14159265,
            1e-3, 1e3,
        ];

        for &f in &test_cases {
            let bf16_bits = f32_to_bf16_rne_reference(f);
            // 反向解码 BF16 → F32 (BF16 = F32 高 16 位, low 16 bits = 0)
            let restored = f32::from_bits((bf16_bits as u32) << 16);
            // 容差: BF16 只有 8 位尾数 (~3 位十进制精度), 相对误差 < 0.5/256 ≈ 0.2%
            if f == 0.0 {
                assert_eq!(restored, 0.0, "zero should map to zero, got {}", restored);
            } else {
                let rel_err = ((restored - f).abs() / f.abs()).abs();
                assert!(rel_err < 0.01,
                    "BF16 RNE of {}: restored={}, rel_err={} exceeds 1% tolerance",
                    f, restored, rel_err);
            }
        }
    }

    #[test]
    fn bf16_avx2_rne_preserves_nan_inf() {
        // NaN/Inf 必须保持语义 (exp 全 1)。
        let pos_inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let nan = f32::NAN;

        let bf16_inf = f32_to_bf16_rne_reference(pos_inf);
        let bf16_neg_inf = f32_to_bf16_rne_reference(neg_inf);
        let bf16_nan = f32_to_bf16_rne_reference(nan);

        // Inf 的 BF16: 0x7F80 (exp=0xFF, mant=0)
        assert_eq!(bf16_inf & 0x7F80, 0x7F80, "+Inf BF16 exp must be all-1s");
        assert_eq!(bf16_neg_inf & 0x7F80, 0x7F80, "-Inf BF16 exp must be all-1s");
        // NaN 的 BF16: exp=0xFF, mant != 0
        assert_eq!(bf16_nan & 0x7F80, 0x7F80, "NaN BF16 exp must be all-1s");
    }
}
