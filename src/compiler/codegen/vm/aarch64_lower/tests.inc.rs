#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::trace::QuantPrecision;

    #[test]
    fn test_aarch64_encoding_basics() {
        let lower = AArch64Lower::new();
        // MOV X0, X1 = ORR X0, XZR, X1
        let mov = lower.enc_mov_x(0, 1);
        assert_eq!(mov & 0xFF000000, 0xAA000000);

        // FADD V0.4S, V1.4S, V2.4S
        let fadd = lower.enc_fadd_4s(0, 1, 2);
        assert_eq!(fadd & 0xFFE00000, 0x4E200000);

        // RET
        assert_eq!(lower.enc_ret(), 0xD65F03C0);
    }

    #[test]
    fn test_aarch64_lower_produces_code() {
        let mut lower = AArch64Lower::new();
        let frame = StackFrame {
            total_size: 0, alignment: 16, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: true,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };
        lower.emit_prologue(&frame, &alloc).unwrap();
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();
        // stp + mov + ldp + ret = 4 instructions = 16 bytes
        assert_eq!(code.len(), 16);
    }

    fn make_test_jit_ctx() -> crate::compiler::jit_context::JitContext {
        let profile = IsaProfile::from_device_profile(
            &crate::dispatch::device_profile::DeviceProfile::detect(),
        );
        crate::compiler::jit_context::JitContext::new(&profile)
    }

    #[test]
    fn test_sve2_prologue_emits_ptrue() {
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let frame = StackFrame {
            total_size: 0, alignment: 16, callee_save_area: 0,
            spill_area: 0, scratchpad_area: 0, uses_red_zone: true,
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };
        lower.emit_prologue(&frame, &alloc).unwrap();
        lower.emit_epilogue(&frame, &alloc).unwrap();
        let code = lower.finalize().unwrap();
        // stp + mov + PTRUE + ldp + ret = 5 instructions = 20 bytes
        assert_eq!(code.len(), 20);
        // Verify PTRUE P7.S is at offset 8 (3rd instruction)
        let ptrue_bytes = &code[8..12];
        let ptrue_val = u32::from_le_bytes([ptrue_bytes[0], ptrue_bytes[1], ptrue_bytes[2], ptrue_bytes[3]]);
        assert_eq!(ptrue_val, 0x2598E007); // PTRUE P7.S
    }

    #[test]
    fn test_sve2_instruction_encodings() {
        let lower = AArch64Lower::new();

        // WHILELT p0.s, x0, x1
        let whilelt = lower.enc_whilelt_s(0, 0, 1);
        assert_eq!(whilelt & 0xFFE00C00, 0x25A00400 & 0xFFE00C00);

        // PTRUE p7.s
        let ptrue = lower.enc_ptrue_s(7);
        assert_eq!(ptrue, 0x2598E007);

        // RDVL x0, #1
        let rdvl = lower.enc_rdvl(0, 1);
        assert_eq!(rdvl & 0xFFFFFC00, 0x04BF5000);

        // SMSTART
        assert_eq!(lower.enc_smstart(), 0xD503437F);

        // SMSTOP
        assert_eq!(lower.enc_smstop(), 0xD503427F);
    }

    #[test]
    fn test_neon_loop_produces_back_branch() {
        // Test that NEON loop with LoopBegin + LoopEnd produces correct structure
        let mut lower = AArch64Lower::new();
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(2)));  // counter → x2
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(3)));  // byte_offset → x3
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        lower.lower_instr(&VmInstr::LoopBegin {
            counter: VRegId(0),
            byte_offset: VRegId(1),
            bound: BoundExpr::Const(16),
            step_bytes: 16,
        }, &alloc).unwrap();

        lower.lower_instr(&VmInstr::LoopEnd, &alloc).unwrap();

        let code = lower.finalize().unwrap();
        // Should produce non-zero code
        assert!(code.len() > 0);
        // Should be a multiple of 4 (all AArch64 instructions are 4 bytes)
        assert_eq!(code.len() % 4, 0);
    }

    #[test]
    fn test_sve2_loop_uses_whilelt() {
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(2)));
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(3)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        lower.lower_instr(&VmInstr::LoopBegin {
            counter: VRegId(0),
            byte_offset: VRegId(1),
            bound: BoundExpr::Const(64),
            step_bytes: 16,
        }, &alloc).unwrap();

        lower.lower_instr(&VmInstr::LoopEnd, &alloc).unwrap();

        let code = lower.finalize().unwrap();
        assert!(code.len() > 0);
        assert_eq!(code.len() % 4, 0);

        // Verify WHILELT is present in the code stream
        let mut found_whilelt = false;
        for i in (0..code.len()).step_by(4) {
            let instr = u32::from_le_bytes([code[i], code[i+1], code[i+2], code[i+3]]);
            // WHILELT p0.s has encoding mask 0xFFE0FC10 == 0x25A00410
            if instr & 0xFFE0FC10 == 0x25A00410 {
                found_whilelt = true;
                break;
            }
        }
        assert!(found_whilelt, "SVE2 loop should contain WHILELT instruction");

        // Verify INCW is present (SVE loop increment)
        let mut found_incw = false;
        for i in (0..code.len()).step_by(4) {
            let instr = u32::from_le_bytes([code[i], code[i+1], code[i+2], code[i+3]]);
            if instr & 0xFFFFF000 == 0x04B0E000 {
                found_incw = true;
                break;
            }
        }
        assert!(found_incw, "SVE2 loop should contain INCW instruction");
    }

    #[test]
    fn test_sme2_tile_mma_emits_fmopa_and_multi_vec() {
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: true, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));  // c
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(2)));  // a (even-aligned for multi-vec)
        mapping.insert(VRegId(2), PhysReg::Vec(PhysVec(4)));  // b
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        lower.lower_instr(&VmInstr::TileMma {
            c: VRegId(0),
            a: VRegId(1),
            b: VRegId(2),
            m: 1, n: 1, k: 1,
            dtype: DType::F32,
        }, &alloc).unwrap();

        let code = lower.finalize().unwrap();
        assert!(code.len() >= 12); // At least FMOPA + FMLA_VG2 + MOVA = 3 instructions

        // Verify FMOPA is present
        let first_instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(first_instr & 0xFF800000, 0x80800000, "First instruction should be FMOPA");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  Additional Unit Tests (10 tests)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_vec_bin_op_add_neon() {
        // Arrange: Create NEON-only lower (no SVE2) and register allocation.
        // Use dst == a (same physical register) to avoid MOV prefix,
        // so the first emitted instruction is directly FADD.
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(1))); // dst → V1
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1))); // a → V1 (same as dst)
        mapping.insert(VRegId(2), PhysReg::Vec(PhysVec(2))); // b → V2
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecBinOp::Add with dst == a (no MOV prefix needed)
        let result = lower.lower_instr(&VmInstr::VecBinOp {
            dst: VRegId(0),
            a: VRegId(1),
            b: VRegId(2),
            op: VecOp::Add,
            dtype: QuantPrecision::F32,
        }, &alloc);

        // Assert: Should succeed and produce exactly FADD V.4S (no MOV prefix)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 4, "With dst==a should produce exactly one instruction");
        // FADD Vd.4S, Vn.4S, Vm.4S: base 0x4E20_D400 | (vm << 16) | (vn << 5) | vd
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(instr & 0xFF20_FC00, 0x4E20_D400, "Should be FADD V.4S instruction");
    }

    #[test]
    fn test_vec_bin_op_mul_sve2() {
        // Arrange: Create SVE2-capable lower
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1)));
        mapping.insert(VRegId(2), PhysReg::Vec(PhysVec(2)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecBinOp::Mul
        let result = lower.lower_instr(&VmInstr::VecBinOp {
            dst: VRegId(0),
            a: VRegId(1),
            b: VRegId(2),
            op: VecOp::Mul,
            dtype: QuantPrecision::F32,
        }, &alloc);

        // Assert: Should succeed with SVE FMUL
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert!(code.len() >= 4);
    }

    #[test]
    fn test_vec_unary_neg_neon() {
        // Arrange: NEON-only lower for VecUnaryOp::Neg.
        // Use dst == a (same physical register) to avoid MOV prefix.
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(1))); // dst → V1
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1))); // src → V1 (same as dst)
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecUnaryOp::Neg with dst == a
        let result = lower.lower_instr(&VmInstr::VecUnaryOp {
            dst: VRegId(0),
            a: VRegId(1),
            op: VecUnaryOp::Neg,
        }, &alloc);

        // Assert: Should produce exactly FNEG V.4S (no MOV prefix)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 4, "With dst==a should produce exactly one instruction");
        // FNEG Vd.4S, Vn.4S: 0x6EA0_F800 | (vn << 5) | vd
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(instr & 0xFFBF_FC00, 0x6EA0_F800, "Should be FNEG V.4S");
    }

    #[test]
    fn test_fma_neon() {
        // Arrange: NEON-only lower for FMA.
        // Use dst == acc (same physical register) to avoid MOV prefix.
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(1))); // dst → V1
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1))); // acc → V1 (same as dst)
        mapping.insert(VRegId(2), PhysReg::Vec(PhysVec(2))); // a → V2
        mapping.insert(VRegId(3), PhysReg::Vec(PhysVec(3))); // b → V3
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower FMA with dst == acc
        let result = lower.lower_instr(&VmInstr::Fma {
            dst: VRegId(0),
            acc: VRegId(1),
            a: VRegId(2),
            b: VRegId(3),
            dtype: QuantPrecision::F32,
        }, &alloc);

        // Assert: Should produce exactly FMLA V.4S (no MOV prefix)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 4, "With dst==acc should produce exactly one instruction");
        // FMLA Vd.4S, Vn.4S, Vm.4S: 0x4E20_CC00 | (vm << 16) | (vn << 5) | vd
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(instr & 0xFF20_FC00, 0x4E20_CC00, "Should be FMLA V.4S");
    }

    #[test]
    fn test_mov_same_register_no_code() {
        // Arrange: Mov with dst == src should produce no code
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(0))); // Same register
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower Mov with same dst and src
        let result = lower.lower_instr(&VmInstr::Mov {
            dst: VRegId(0),
            src: VRegId(1),
            dtype: QuantPrecision::F32,
        }, &alloc);

        // Assert: Should succeed but produce no code (optimization)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 0, "Mov with same dst/src should produce no code");
    }

    #[test]
    fn test_hreduce_sum_neon() {
        // Arrange: NEON-only horizontal reduction (sum)
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0))); // dst
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1))); // src
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower HReduce::Sum
        let result = lower.lower_instr(&VmInstr::HReduce {
            dst: VRegId(0),
            src: VRegId(1),
            op: ReduceOp::Sum,
        }, &alloc);

        // Assert: Should produce FADDP cascade (2 instructions for 4-element reduction)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert!(code.len() >= 8, "HReduce Sum should produce at least 2 FADDP instructions");
    }

    #[test]
    fn test_accumulate_neon() {
        // Arrange: NEON-only accumulate (acc += src)
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0))); // acc
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1))); // src
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower Accumulate
        let result = lower.lower_instr(&VmInstr::Accumulate {
            acc: VRegId(0),
            src: VRegId(1),
        }, &alloc);

        // Assert: Should produce FADD instruction
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert!(code.len() >= 4);
    }

    #[test]
    fn test_broadcast_const_neon() {
        // Arrange: NEON-only broadcast of constant value
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0))); // dst
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower Broadcast with constant 1.0
        let result = lower.lower_instr(&VmInstr::Broadcast {
            dst: VRegId(0),
            src: ScalarExpr::Const(1.0),
            width: SimdWidth::W128,
            dtype: QuantPrecision::F32,
        }, &alloc);

        // Assert: Should produce MOV + MOVK + DUP sequence
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert!(code.len() >= 12, "Broadcast const should produce MOV+MOVK+DUP sequence");
    }

    #[test]
    fn test_load_ptr_abs_addr() {
        // Arrange: LoadPtr with absolute address (64-bit immediate)
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(0))); // dst
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Load 64-bit absolute address
        let test_addr: u64 = 0x1234_5678_9ABC_DEF0;
        let result = lower.lower_instr(&VmInstr::LoadPtr {
            dst: VRegId(0),
            src: PtrExpr::AbsAddr(test_addr),
        }, &alloc);

        // Assert: Should produce 4 MOVZ/MOVK instructions (16 bytes)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 16, "AbsAddr load should produce 4 instructions (16 bytes)");
    }

    #[test]
    fn test_vec_load_with_loop_offset_neon() {
        // Arrange: VecLoad with LoopOffset (requires offset register resolution)
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0))); // dst
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(1))); // base
        mapping.insert(VRegId(2), PhysReg::Gpr(PhysGpr(2))); // loop offset
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecLoad with LoopOffset
        let result = lower.lower_instr(&VmInstr::VecLoad {
            dst: VRegId(0),
            base: VRegId(1),
            offset: OffsetExpr::LoopOffset(VRegId(2)),
            width: SimdWidth::W128,
            dtype: QuantPrecision::F32, predicate: None,
        }, &alloc);

        // Assert: Should produce ADD + LD1 sequence
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert!(code.len() >= 8, "VecLoad with LoopOffset should produce ADD + LD1");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  Additional Unit Tests — Wave 2 (10 tests)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    // @trace BCE-X86HW-002-AARCH64-DOTDTYPE-PREDICATES [req:REQ-VR-002] [level:unit]
    fn test_dot_dtype_predicates() {
        // BCE-20260704-X86HW-002: 加入 Bf16xF32, Fp16xF32 混合精度变体谓词断言。
        // Arrange: exercise each DotDtype variant with its predicate function
        // Act & Assert: each predicate should return true only for its target variant
        assert!(dot_dtype_is_bf16(DotDtype::Bf16));
        assert!(!dot_dtype_is_bf16(DotDtype::Bf16xF32));
        assert!(!dot_dtype_is_bf16(DotDtype::Fp16));
        assert!(!dot_dtype_is_bf16(DotDtype::Fp16xF32));
        assert!(!dot_dtype_is_bf16(DotDtype::Int8));
        assert!(!dot_dtype_is_bf16(DotDtype::Int4x8));
        assert!(!dot_dtype_is_bf16(DotDtype::Fp4));

        // Bf16xF32 谓词: 只对 Bf16xF32 为 true
        assert!(dot_dtype_is_bf16xf32(DotDtype::Bf16xF32));
        assert!(!dot_dtype_is_bf16xf32(DotDtype::Bf16));
        assert!(!dot_dtype_is_bf16xf32(DotDtype::Fp16));
        assert!(!dot_dtype_is_bf16xf32(DotDtype::Fp16xF32));
        assert!(!dot_dtype_is_bf16xf32(DotDtype::Int8));

        assert!(dot_dtype_is_fp16(DotDtype::Fp16));
        assert!(!dot_dtype_is_fp16(DotDtype::Bf16));
        assert!(!dot_dtype_is_fp16(DotDtype::Fp16xF32));

        // Fp16xF32 谓词: 只对 Fp16xF32 为 true
        assert!(dot_dtype_is_fp16xf32(DotDtype::Fp16xF32));
        assert!(!dot_dtype_is_fp16xf32(DotDtype::Fp16));
        assert!(!dot_dtype_is_fp16xf32(DotDtype::Bf16));
        assert!(!dot_dtype_is_fp16xf32(DotDtype::Bf16xF32));

        assert!(dot_dtype_is_int8(DotDtype::Int8));
        assert!(!dot_dtype_is_int8(DotDtype::Bf16));

        assert!(dot_dtype_is_int4x8(DotDtype::Int4x8));
        assert!(!dot_dtype_is_int4x8(DotDtype::Fp4));

        assert!(dot_dtype_is_fp4(DotDtype::Fp4));
        assert!(!dot_dtype_is_fp4(DotDtype::Int4x8));
    }

    #[test]
    fn test_aarch64_features_default_is_neon_only() {
        // Arrange: construct default AArch64Features
        let features = AArch64Features::default();

        // Assert: default should be NEON baseline (no SVE/SVE2/SME2/BF16/DotProd/I8MM, zero VL)
        assert!(!features.has_sve, "Default should not have SVE");
        assert!(!features.has_sve2, "Default should not have SVE2");
        assert!(!features.has_sme2, "Default should not have SME2");
        assert!(!features.has_bf16, "Default should not have BF16");
        assert!(!features.has_dotprod, "Default should not have DotProd");
        assert!(!features.has_i8mm, "Default should not have I8MM");
        assert_eq!(features.sve_vl, 0, "Default SVE VL should be 0");
    }

    #[test]
    fn test_with_profile_extracts_all_aarch64_features() {
        // Regression for BCE-20260703-AARCH64-FEATURES-DROPPED:
        // with_profile 必须从 Platform::AArch64 提取全部特性字段, 不能用 `..` 丢弃
        // has_bf16/has_dotprod/has_i8mm/has_sve (原 bug 导致 lower 层无法感知这些特性,
        // BFDOT/SDOT 无条件发出 → 无该特性的 CPU SIGILL; SVE1-only CPU 被降级为 NEON)。
        //
        // SVE1-only CPU (has_sve=true, has_sve2=false, e.g. A64FX):
        //   has_dotprod=true (constructor sets), has_i8mm=has_sve2=false, has_bf16=true (arg)
        let profile = IsaProfile::aarch64(
            /* has_sve   */ true,
            /* has_sve2  */ false,   // SVE1-only
            /* sve_vl    */ 32,
            /* has_sme   */ false,
            /* has_sme2  */ false,
            /* has_bf16  */ true,
        );

        let lower = AArch64Lower::with_profile(&profile);

        // Assert: all features extracted (no field dropped via `..`)
        assert!(lower.platform.has_sve, "has_sve must be extracted (SVE1-only CPU needs SVE1 path, not NEON)");
        assert!(!lower.platform.has_sve2, "has_sve2 must be extracted");
        assert!(lower.platform.has_bf16, "has_bf16 must be extracted (FEAT_BF16 for BFDOT)");
        assert!(lower.platform.has_dotprod, "has_dotprod must be extracted (FEAT_DotProd for SDOT)");
        assert!(!lower.platform.has_i8mm, "has_i8mm must be extracted (= has_sve2 on this constructor)");
        assert_eq!(lower.platform.sve_vl, 32, "sve_vl must be extracted");
    }

    #[test]
    fn test_neon_encoding_register_fields() {
        // Arrange: verify register fields are correctly placed in encoded instructions
        let lower = AArch64Lower::new();

        // Act: encode LDR X5, [X3, #8]
        let ldr = lower.enc_ldr_x(5, 3, 1); // imm12=1 means offset=8 (scaled by 8)
        // Assert: Rd field is bits[4:0], Rn is bits[9:5]
        assert_eq!(ldr & 0x1F, 5, "LDR Rd should be 5");
        assert_eq!((ldr >> 5) & 0x1F, 3, "LDR Rn should be 3");

        // Act: encode STR X7, [X2, #16]
        let str_instr = lower.enc_str_x(7, 2, 2); // imm12=2 means offset=16
        // Assert
        assert_eq!(str_instr & 0x1F, 7, "STR Rd should be 7");
        assert_eq!((str_instr >> 5) & 0x1F, 2, "STR Rn should be 2");

        // Act: encode LD1 {V3.4S}, [X1]
        let ld1 = lower.enc_ld1_4s(3, 1);
        // Assert: Vt is bits[4:0], Xn is bits[9:5]
        assert_eq!(ld1 & 0x1F, 3, "LD1 Vt should be 3");
        assert_eq!((ld1 >> 5) & 0x1F, 1, "LD1 Rn should be 1");
    }

    #[test]
    fn test_vec_bin_op_sub_neon() {
        // Arrange: NEON-only lower for VecBinOp::Sub.
        // Use dst == a to avoid MOV prefix.
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(4))); // dst → V4
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(4))); // a → V4 (same as dst)
        mapping.insert(VRegId(2), PhysReg::Vec(PhysVec(5))); // b → V5
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecBinOp::Sub with dst == a
        let result = lower.lower_instr(&VmInstr::VecBinOp {
            dst: VRegId(0),
            a: VRegId(1),
            b: VRegId(2),
            op: VecOp::Sub,
            dtype: QuantPrecision::F32,
        }, &alloc);

        // Assert: Should produce exactly FSUB V.4S (no MOV prefix)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 4, "With dst==a should produce exactly one instruction");
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        // FSUB Vd.4S, Vn.4S, Vm.4S: base 0x4EA0_D400
        // Mask 0xFFE0FC00 clears only register fields (Rd, Rn, Rm), preserves opcode
        assert_eq!(instr & 0xFFE0_FC00, 0x4EA0_D400, "Should be FSUB V.4S instruction");
    }

    #[test]
    fn test_vec_unary_abs_neon() {
        // Arrange: NEON-only lower for VecUnaryOp::Abs.
        // Use dst == src to avoid MOV prefix.
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(3))); // dst → V3
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(3))); // src → V3 (same as dst)
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecUnaryOp::Abs with dst == src
        let result = lower.lower_instr(&VmInstr::VecUnaryOp {
            dst: VRegId(0),
            a: VRegId(1),
            op: VecUnaryOp::Abs,
        }, &alloc);

        // Assert: Should produce exactly FABS V.4S (no MOV prefix)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 4, "With dst==src should produce exactly one instruction");
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        // FABS Vd.4S, Vn.4S: 0x4EA0_F800 | (vn << 5) | vd (as encoded in lower_instr)
        // Mask 0xFFFF_FC00 clears only Rd/Rn register fields, preserves full opcode
        assert_eq!(instr & 0xFFFF_FC00, 0x4EA0_F800, "Should be FABS V.4S");
    }

    #[test]
    fn test_vec_load_const_offset_neon() {
        // Arrange: VecLoad with Const offset (non-zero) on NEON
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0))); // dst
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(1))); // base
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecLoad with Const offset of 64 bytes
        let result = lower.lower_instr(&VmInstr::VecLoad {
            dst: VRegId(0),
            base: VRegId(1),
            offset: OffsetExpr::Const(64),
            width: SimdWidth::W128,
            dtype: QuantPrecision::F32, predicate: None,
        }, &alloc);

        // Assert: Should produce ADD + LD1 (2 instructions = 8 bytes)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 8, "VecLoad Const(64) should produce ADD + LD1 = 8 bytes");
    }

    #[test]
    fn test_vec_store_zero_offset_neon() {
        // Arrange: VecStore with Const(0) offset — simplest case, just ST1
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(2))); // src → V2
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(1))); // base → X1
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower VecStore with zero offset
        let result = lower.lower_instr(&VmInstr::VecStore {
            base: VRegId(1),
            src: VRegId(0),
            offset: OffsetExpr::Const(0),
            width: SimdWidth::W128,
            dtype: QuantPrecision::F32, predicate: None,
        }, &alloc);

        // Assert: Should produce exactly ST1 (4 bytes)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 4, "VecStore Const(0) should produce exactly ST1");
        // ST1 {Vt.4S}, [Xn]: base 0x4C007800
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(instr & 0xFFFF_FC00, 0x4C007800, "Should be ST1 V.4S instruction");
    }

    #[test]
    fn test_resolve_gpr_missing_returns_error() {
        // Arrange: create a lower and an empty allocation (no mappings)
        let lower = AArch64Lower::new();
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: try to resolve a VRegId that has no physical register mapping
        let result = lower.resolve_gpr(VRegId(99), &alloc);

        // Assert: should return an error, not panic
        assert!(result.is_err());
        match result {
            Err(CompilerError::CodegenViolation(msg)) => {
                assert!(msg.contains("v99"), "Error message should mention the unmapped vreg");
            }
            _ => panic!("Expected CodegenViolation error, got {:?}", result),
        }
    }

    #[test]
    fn test_resolve_vreg_missing_returns_error() {
        // Arrange: create a lower and an empty allocation
        let lower = AArch64Lower::new();
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: try to resolve a VRegId that has no vector register mapping
        let result = lower.resolve_vreg(VRegId(42), &alloc);

        // Assert: should return an error
        assert!(result.is_err());
        match result {
            Err(CompilerError::CodegenViolation(msg)) => {
                assert!(msg.contains("v42"), "Error message should mention the unmapped vreg");
            }
            _ => panic!("Expected CodegenViolation error, got {:?}", result),
        }
    }

    #[test]
    fn test_sve2_mov_with_different_registers_emits_movprfx() {
        // Arrange: SVE2-capable lower with dst != src for Mov
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0))); // dst → Z0
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1))); // src → Z1
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };

        // Act: Lower Mov with different physical registers
        let result = lower.lower_instr(&VmInstr::Mov {
            dst: VRegId(0),
            src: VRegId(1),
            dtype: QuantPrecision::F32,
        }, &alloc);

        // Assert: SVE2 should emit ORR-based move (4 bytes)
        assert!(result.is_ok());
        let code = lower.finalize().unwrap();
        assert_eq!(code.len(), 4, "SVE2 Mov with different regs should emit one ORR instruction");
        // SVE ORR Zd.D, Zn.D, Zn.D: 0x04603000 | (zn << 5) | zd
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(instr & 0xFFE0_FC00, 0x04603000, "Should be SVE ORR (mov)");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  BCE-20260703-AARCH64-SVE2-I8MM regression:
    //  INT8 dot 必须优先 SMMLA (FEAT_I8MM) > SDOT (FEAT_DotProd); SDOT 编码必须正确。
    //  编码经 aarch64-linux-gnu-as/objdump 实测核对 (smmla v0,v1,v2 = 0x4E82A420,
    //  sdot v0,v1,v2 = 0x4E829420)。原 SDOT 字面量 0x4E409C00 反汇编为 0x4E429C20 = UNDEF。
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    fn make_lower_with(features: AArch64Features) -> AArch64Lower {
        AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: features,
            jit_ctx: make_test_jit_ctx(),
        }
    }

    #[test]
    fn test_int8_dot_emits_smmla_when_i8mm_present() {
        // Arrange: AArch64 with FEAT_I8MM (has_i8mm=true). has_dotprod also true
        // (典型 i8mm CPU 两者都有, 但 i8mm 优先).
        let mut lower = make_lower_with(AArch64Features {
            has_sve: false, has_sve2: true, has_sme2: false,
            has_bf16: false, has_dotprod: true, has_i8mm: true, sve_vl: 16,
        });

        // Act: lower INT8 dot-product with vd=0, vn=1, vm=2
        lower.lower_dot_product_native(0, 1, 2, DotDtype::Int8).unwrap();
        let code = lower.finalize().unwrap();

        // Assert: exactly one 32-bit instruction, equals SMMLA v0.4s, v1.16b, v2.16b
        assert_eq!(code.len(), 4, "SMMLA should emit one 32-bit instruction");
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(
            instr, 0x4E82A420,
            "i8mm=true must emit SMMLA (0x4E82A420 for v0,v1,v2), got {:#010X}",
            instr
        );
    }

    #[test]
    fn test_int8_dot_falls_back_to_sdot_without_i8mm() {
        // Arrange: AArch64 with FEAT_DotProd but no FEAT_I8MM.
        let mut lower = make_lower_with(AArch64Features {
            has_sve: false, has_sve2: false, has_sme2: false,
            has_bf16: false, has_dotprod: true, has_i8mm: false, sve_vl: 16,
        });

        // Act
        lower.lower_dot_product_native(0, 1, 2, DotDtype::Int8).unwrap();
        let code = lower.finalize().unwrap();

        // Assert: SDOT v0.4s, v1.16b, v2.16b = 0x4E829420 (corrected encoding)
        assert_eq!(code.len(), 4);
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(
            instr, 0x4E829420,
            "i8mm=false + has_dotprod=true must emit SDOT (0x4E829420 for v0,v1,v2), got {:#010X}",
            instr
        );
    }

    #[test]
    fn test_int8_dot_errors_without_i8mm_or_dotprod() {
        // Arrange: AArch64 with neither FEAT_I8MM nor FEAT_DotProd.
        let mut lower = make_lower_with(AArch64Features {
            has_sve: false, has_sve2: false, has_sme2: false,
            has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16,
        });

        // Act & Assert: NO-SILENT-FALLBACK — must return Err, not emit a SIGILL-inducing instr
        let result = lower.lower_dot_product_native(0, 1, 2, DotDtype::Int8);
        assert!(result.is_err(), "INT8 dot without i8mm/dotprod must Err (NO-SILENT-FALLBACK)");
        assert!(lower.code.is_empty(), "No instruction should be emitted on Err");
    }

    #[test]
    fn test_int8_dot_smmla_register_fields() {
        // Arrange: i8mm CPU — verify register fields land in correct bit positions.
        let mut lower = make_lower_with(AArch64Features {
            has_sve: false, has_sve2: true, has_sme2: false,
            has_bf16: false, has_dotprod: false, has_i8mm: true, sve_vl: 16,
        });

        // Act: vd=5, vn=6, vm=7
        lower.lower_dot_product_native(5, 6, 7, DotDtype::Int8).unwrap();
        let code = lower.finalize().unwrap();
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);

        // Assert: SMMLA base 0x4E80A400 | (Rm<<16) | (Rn<<5) | Rd
        assert_eq!(instr & 0x1F, 5, "Rd field (bits 4-0) must be 5");
        assert_eq!((instr >> 5) & 0x1F, 6, "Rn field (bits 9-5) must be 6");
        assert_eq!((instr >> 16) & 0x1F, 7, "Rm field (bits 20-16) must be 7");
        assert_eq!(instr & 0xFFE0FC00, 0x4E80A400, "SMMLA base opcode bits must match (mask Rm/Rn/Rd)");
    }

    #[test]
    fn test_int4x8_dot_widen_uses_corrected_sdot_encoding() {
        // Arrange: WidenCompute path (INT4x8 unpacked → SDOT). has_dotprod=true, no i8mm.
        let mut lower = make_lower_with(AArch64Features {
            has_sve: false, has_sve2: false, has_sme2: false,
            has_bf16: false, has_dotprod: true, has_i8mm: false, sve_vl: 16,
        });

        // Act
        lower.lower_dot_product_widen(0, 1, 2, DotDtype::Int4x8).unwrap();
        let code = lower.finalize().unwrap();
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);

        // Assert: corrected SDOT encoding 0x4E829420 (was buggy 0x4E429C20)
        assert_eq!(
            instr, 0x4E829420,
            "INT4x8 WidenCompute SDOT must use corrected encoding (0x4E829420), got {:#010X}",
            instr
        );
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  BCE-20260704-AARCH64 违宪根治回归测试
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// AARCH64-002: TileMma FMOPA 在无 SME2 的 CPU 必须返回 Err (NO-SILENT-FALLBACK),
    /// 而非无条件发射 FMOPA 导致 SIGILL。
    #[test]
    fn test_tile_mma_no_sme2_returns_err() {
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };
        let result = lower.lower_instr(&VmInstr::TileMma {
            c: VRegId(0), a: VRegId(1), b: VRegId(2),
            m: 1, n: 1, k: 1, dtype: DType::F32,
        }, &alloc);
        assert!(result.is_err(), "TileMma on non-SME2 platform must return Err (NO-SILENT-FALLBACK), got Ok");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("SME2") || err_msg.contains("SME"),
            "Err must mention SME/SME2 requirement, got: {}", err_msg);
    }

    /// AARCH64-005: GatherLoad 在 Scalable (SVE) width 必须发射 SVE gather 指令
    /// (LD1W ...SXTW #2), 不能静默 NOP (原 f32_lanes()=0 → for i in 0..0)。
    #[test]
    fn test_gather_load_sve_emits_gather_not_nop() {
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));  // dst
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(1))); // base
        mapping.insert(VRegId(2), PhysReg::Gpr(PhysGpr(2))); // indices
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        lower.lower_instr(&VmInstr::GatherLoad {
            dst: VRegId(0), base: VRegId(1), indices: VRegId(2),
            stride: 1, width: SimdWidth::Scalable,
            dtype: QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();

        let code = lower.finalize().unwrap();
        // Must emit ≥3 instructions: PTRUE + LD1W(idx load) + LD1W gather (stride==1)
        assert!(code.len() >= 12, "SVE gather must emit instructions (not NOP), got {} bytes", code.len());
        // Verify the SVE gather LD1W Zt.S, Pg/Z, [Xn, Zm, SXTW #2] base 0x85206000 is present.
        // Mask out variable fields (Zm[20:16], Pg[12:10], Xn[9:5], Zt[4:0]) → fixed bits 0xFFE0E000.
        let mut found_gather = false;
        for chunk in code.chunks_exact(4) {
            let instr = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if instr & 0xFFE0E000 == 0x85206000 {
                found_gather = true;
                break;
            }
        }
        assert!(found_gather, "SVE gather must emit LD1W [Xn,Zm,SXTW#2] (0x85206xxx), got code: {:08X?}", {
            let mut v = Vec::new();
            for c in code.chunks_exact(4) { v.push(u32::from_le_bytes([c[0],c[1],c[2],c[3]])); }
            v
        });
    }

    /// AARCH64-005: ScatterStore 在 Scalable (SVE) width 必须发射 SVE scatter 指令。
    #[test]
    fn test_scatter_store_sve_emits_scatter() {
        let mut lower = AArch64Lower {
            code: Vec::new(),
            const_pool: Vec::new(),
            data_tables: Vec::new(),
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: true, has_sve2: true, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 32 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));  // src
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(1))); // base
        mapping.insert(VRegId(2), PhysReg::Gpr(PhysGpr(2))); // indices
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };

        lower.lower_instr(&VmInstr::ScatterStore {
            base: VRegId(1), indices: VRegId(2), src: VRegId(0),
            stride: 1, width: SimdWidth::Scalable,
            dtype: QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();

        let code = lower.finalize().unwrap();
        assert!(code.len() >= 12, "SVE scatter must emit instructions (not NOP), got {} bytes", code.len());
        let mut found_scatter = false;
        for chunk in code.chunks_exact(4) {
            let instr = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if instr & 0xFFE0E000 == 0xE5206000 {
                found_scatter = true;
                break;
            }
        }
        assert!(found_scatter, "SVE scatter must emit ST1W [Xn,Zm,SXTW#2] (0xE5206xxx)");
    }

    /// AARCH64-007: VecNarrow F32→F16 必须发射 FCVTN (非 Err)。
    #[test]
    fn test_vec_narrow_f32_to_f16_emits_fcvtn() {
        let mut lower = AArch64Lower::new();
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        lower.lower_instr(&VmInstr::VecNarrow {
            dst: VRegId(0), src: VRegId(1),
            dst_dtype: QuantPrecision::F16, src_dtype: QuantPrecision::F32,
            width: SimdWidth::W128,
        }, &alloc).unwrap();
        let code = lower.finalize().unwrap();
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(instr & 0xFFFFFC00, 0x0E216800, "F32→F16 must emit FCVTN (0x0E216800), got {:#010X}", instr);
    }

    /// AARCH64-007: VecNarrow F32→BF16 需 has_bf16 (BFCVTN), 无 has_bf16 返回 Err。
    #[test]
    fn test_vec_narrow_f32_to_bf16_requires_has_bf16() {
        let mut lower = AArch64Lower {
            code: Vec::new(), const_pool: Vec::new(),
            data_tables: Vec::new(), loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            pending_labels: std::collections::HashMap::new(),
            resolved_labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve: false, has_sve2: false, has_sme2: false, has_bf16: false, has_dotprod: false, has_i8mm: false, sve_vl: 16 },
            jit_ctx: make_test_jit_ctx(),
        };
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        let result = lower.lower_instr(&VmInstr::VecNarrow {
            dst: VRegId(0), src: VRegId(1),
            dst_dtype: QuantPrecision::BF16, src_dtype: QuantPrecision::F32,
            width: SimdWidth::W128,
        }, &alloc);
        assert!(result.is_err(), "F32→BF16 without has_bf16 must Err (NO-SILENT-FALLBACK)");
    }

    /// AARCH64-007: VecWiden F16→F32 必须发射 FCVTL (非 Err)。
    #[test]
    fn test_vec_widen_f16_to_f32_emits_fcvtl() {
        let mut lower = AArch64Lower::new();
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        lower.lower_instr(&VmInstr::VecWiden {
            dst: VRegId(0), src: VRegId(1),
            dst_dtype: QuantPrecision::F32, src_dtype: QuantPrecision::F16,
            width: SimdWidth::W128,
        }, &alloc).unwrap();
        let code = lower.finalize().unwrap();
        let instr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(instr & 0xFFFFFC00, 0x0E617800, "F16→F32 must emit FCVTL (0x0E617800), got {:#010X}", instr);
    }

    /// AARCH64-011: FP8→F32 必须返回 Err (注明 FEAT_FP8 限制), 不静默 NOP。
    #[test]
    fn test_fp8_to_float_returns_err_no_silent_nop() {
        let mut lower = AArch64Lower::new();
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(1), PhysReg::Vec(PhysVec(1)));
        let alloc = RegAllocation { mapping, spills: vec![], callee_saved_used: vec![] };
        let result = lower.lower_instr(&VmInstr::VecUnaryOp {
            dst: VRegId(0), a: VRegId(1), op: VecUnaryOp::Fp8E4M3ToFloat,
        }, &alloc);
        assert!(result.is_err(), "FP8→F32 must Err (no has_fp8 + no software cvt), not silent NOP");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("FP8") || err_msg.contains("fp8") || err_msg.contains("FEAT_FP8"),
            "Err must mention FP8/FEAT_FP8 limitation, got: {}", err_msg);
        // 确保未发射任何指令 (静默 NOP 的反面: Err 时不写入 code)
        assert!(lower.finalize().unwrap().is_empty(), "Err path must not emit code bytes (no silent NOP)");
    }

    /// BCE-20260715-SIGSEGV-TABLE-IN-FALLTHROUGH regression (AArch64).
    ///
    /// LoadLayerWeightOffset emits ADR + LDR (LDR falls through). The table
    /// bytes must NOT be baked inline after LDR (CPU would execute them as
    /// instructions → SIGSEGV). Must be flushed past RET and the ADR immediate
    /// backpatched to reach it (ARCH-TABLE-OUT-OF-FALLTHROUGH, same-class x86 fix).
    #[test]
    fn bce_load_layer_weight_offset_table_out_of_fallthrough_aarch64() {
        use crate::compiler::codegen::vm::isa_profile::{PhysGpr, PhysReg};
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let layer_idx = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let offset_table = vec![0usize, 11370496, 22740992, 34111488];
        prog.emit(VmInstr::LoadLayerWeightOffset {
            dst, offset_table: offset_table.clone(), layer_idx_reg: layer_idx,
        });

        // Map dst → x1, layer_idx → x2 (arbitrary GPRs).
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(dst, PhysReg::Gpr(PhysGpr(1)));
        mapping.insert(layer_idx, PhysReg::Gpr(PhysGpr(2)));
        let alloc = RegAllocation {
            mapping, spills: vec![], callee_saved_used: vec![],
        };

        let mut lower = AArch64Lower::new();
        // Emit only the LoadLayerWeightOffset (skip DeclareVReg no-ops), then RET.
        let llwo = prog.instrs.iter()
            .find(|i| matches!(i, VmInstr::LoadLayerWeightOffset { .. }))
            .expect("LoadLayerWeightOffset must be emitted");
        lower.lower_instr(llwo, &alloc).unwrap();
        lower.emit32(lower.enc_ret()); // 0xD65F03C0
        let code = lower.finalize().unwrap();

        // 1. ADR is at offset 0; LDR at offset 4; RET at offset 8; table past RET.
        let adr = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        // ADR opcode top bits + Rd=1.
        assert_eq!(adr & 0x9F00001F, 0x10000001, "ADR x1, ... ; got {:08x}", adr);
        // RET must be at offset 8 (after ADR+LDR).
        let ret = u32::from_le_bytes([code[8], code[9], code[10], code[11]]);
        assert_eq!(ret, 0xD65F03C0, "RET must follow ADR+LDR at offset 8");

        // 2. Table bytes must live AFTER RET (offset >= 12), not inline (offset 8).
        let after_ret = &code[12..];
        for &off in &offset_table {
            let le = (off as u64).to_le_bytes();
            assert!(after_ret.windows(le.len()).any(|w| w == le),
                "BCE-20260715 aarch64: offset_table entry {} (LE {:?}) must be baked past RET",
                off, le);
        }
        // 3. Table must NOT leak into the inline path between LDR (offset 4-8).
        let inline = &code[8..12];
        let le1 = (offset_table[1] as u64).to_le_bytes();
        assert!(!inline.windows(le1.len()).any(|w| w == le1),
            "BCE-20260715 aarch64: table bytes leaked into inline fall-through path");

        // 4. ADR immediate correctly backpatched to reach the table (offset 12).
        //    imm21 (signed byte offset) = 12 - 0 = 12. immlo=12&3=0, immhi=(12>>2)=3.
        let immlo = (adr >> 29) & 0b11;
        let immhi = (adr >> 5) & 0x7FFFF;
        let delta = (immhi << 2) | immlo;
        assert_eq!(delta, 12, "ADR immediate must reach table at offset 12 (delta=12)");
    }

    // ── BCE-20260727-AARCH64-JUMPTOLABEL: label 回填机制单测 ──
    // 验证 pending_labels 两阶段回填：前向引用 + 后向引用 + 未回填 finalize 报错。

    #[test]
    fn test_aarch64_label_forward_reference_backpatch() {
        // 前向引用: JumpToLabel 先于 MarkLabel
        // CBZ X0, label  ; 占位 imm19=0
        // NOP            ; 一条指令
        // MarkLabel      ; label 目标 = NOP 之后
        // 期望: CBZ 的 imm19 回填为 +1 (跳过 NOP 到 MarkLabel)
        let mut lower = AArch64Lower::new();
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };
        let label_id = 5000usize;
        // CBZ X0, label (X0 = reg 0) — emit 占位 imm19=0
        let branch_offset = lower.current_offset();
        lower.emit32(0xB4000000 | 0u32); // CBZ X0
        lower.record_label_patch_site(label_id, branch_offset, false);
        // NOP (1 instruction)
        lower.emit32(0xD503201F); // NOP
        // MarkLabel — 触发回填
        let target_offset = lower.current_offset();
        lower.resolve_label(label_id, target_offset);
        // 验证 CBZ imm19 = +1 (跳过 1 条 NOP 到 MarkLabel)
        let cbz = u32::from_le_bytes(
            lower.code[branch_offset..branch_offset + 4].try_into().unwrap()
        );
        let imm19 = ((cbz >> 5) & 0x7FFFF) as i32;
        let imm19_signed = if imm19 >= (1 << 18) { imm19 - (1 << 19) } else { imm19 };
        assert_eq!(imm19_signed, 2, "CBZ imm19 must be +2 to skip NOP and land on MarkLabel (branch at 0, NOP at 4, MarkLabel at 8)");
    }

    #[test]
    fn test_aarch64_label_backward_reference_immediate_backpatch() {
        // 后向引用: MarkLabel 先于 JumpToLabel
        // MarkLabel      ; label 目标 = 此处
        // NOP            ; 一条指令
        // CBZ X0, label  ; label 已解析，立即回填 imm19
        // 期望: CBZ 的 imm19 = -1 (跳回 NOP 之前 = MarkLabel)
        let mut lower = AArch64Lower::new();
        let alloc = RegAllocation {
            mapping: std::collections::HashMap::new(),
            spills: vec![],
            callee_saved_used: vec![],
        };
        let label_id = 5001usize;
        // MarkLabel — 记录 resolved_labels
        let target_offset = lower.current_offset();
        lower.resolve_label(label_id, target_offset);
        // NOP
        lower.emit32(0xD503201F); // NOP
        // CBZ X0, label — label 已解析，立即回填
        let branch_offset = lower.current_offset();
        lower.emit32(0xB4000000 | 0u32); // CBZ X0
        lower.record_label_patch_site(label_id, branch_offset, false);
        // 验证 CBZ imm19 = -1 (跳回 MarkLabel，target 在 branch 之前 4 字节)
        let cbz = u32::from_le_bytes(
            lower.code[branch_offset..branch_offset + 4].try_into().unwrap()
        );
        let imm19 = ((cbz >> 5) & 0x7FFFF) as i32;
        let imm19_signed = if imm19 >= (1 << 18) { imm19 - (1 << 19) } else { imm19 };
        assert_eq!(imm19_signed, -1, "CBZ imm19 must be -1 to jump back to MarkLabel");
    }

    #[test]
    fn test_aarch64_unconditional_branch_imm26_backpatch() {
        // 无条件 B 指令用 imm26 (±128MB)，验证 imm26 回填而非 imm19。
        // B label  ; 占位 imm26=0
        // NOP ×3
        // MarkLabel
        // 期望: B 的 imm26 = +3 (跳过 3 条 NOP)
        let mut lower = AArch64Lower::new();
        let label_id = 5002usize;
        let branch_offset = lower.current_offset();
        lower.emit32(0x14000000u32); // B label (占位 imm26=0)
        lower.record_label_patch_site(label_id, branch_offset, true);
        for _ in 0..3 {
            lower.emit32(0xD503201F); // NOP ×3
        }
        let target_offset = lower.current_offset();
        lower.resolve_label(label_id, target_offset);
        let b_instr = u32::from_le_bytes(
            lower.code[branch_offset..branch_offset + 4].try_into().unwrap()
        );
        let imm26 = (b_instr & 0x3FFFFFF) as i32;
        let imm26_signed = if imm26 >= (1 << 25) { imm26 - (1 << 26) } else { imm26 };
        assert_eq!(imm26_signed, 4, "B imm26 must be +4 to skip B itself + 3 NOPs to MarkLabel (branch at 0, NOPs at 4/8/12, MarkLabel at 16)");
    }

    #[test]
    fn test_aarch64_finalize_errors_on_unresolved_label() {
        // 未回填的 pending label = MarkLabel 缺失 = 编译错误。
        // Emit CBZ X0, label 但不 emit MarkLabel → finalize 必须返回 Err。
        let mut lower = AArch64Lower::new();
        let label_id = 5003usize;
        let branch_offset = lower.current_offset();
        lower.emit32(0xB4000000 | 0u32); // CBZ X0
        lower.record_label_patch_site(label_id, branch_offset, false);
        // 不 emit MarkLabel — pending_labels 非空
        let result = lower.finalize();
        match result {
            Err(CompilerError::CodegenViolation(msg)) => {
                assert!(msg.contains("label") && msg.contains("MarkLabel"),
                    "finalize error must mention unresolved label and MarkLabel, got: {}", msg);
            }
            _ => panic!("finalize must return Err on unresolved pending_labels, got {:?}", result),
        }
    }

    #[test]
    fn test_aarch64_gpr_cond_action_jumptolabel_isnull_emits_cbz() {
        // GprCondAction { IsNull, JumpToLabel } 应 emit CBZ Xn, label
        // (BCE-20260727: 旧版直接 Err "GPU-only"，现支持)
        let mut lower = AArch64Lower::new();
        let mut mapping = std::collections::HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(5)));
        let alloc = RegAllocation {
            mapping,
            spills: vec![],
            callee_saved_used: vec![],
        };
        let label_id = 5004usize;
        let instr = VmInstr::GprCondAction {
            cond: GprCondition::IsNull(VRegId(0)),
            action: GprBranchAction::JumpToLabel(label_id),
        };
        // emit CBZ X5, label
        let result = lower.lower_gpr_cond_action_aarch64(&instr, &alloc);
        assert!(result.is_ok(), "GprCondAction IsNull+JumpToLabel must succeed on AArch64, got {:?}", result);
        // 验证 emit 了 CBZ X5 (0xB4 + reg 5)
        let cbz = u32::from_le_bytes(lower.code[0..4].try_into().unwrap());
        assert_eq!(cbz & 0xFF00001F, 0xB4000005, "must be CBZ X5, got {:#010x}", cbz);
        // pending_labels 应记录该 label
        assert!(lower.pending_labels.contains_key(&label_id),
            "pending_labels must contain label_id after JumpToLabel emit");
    }

}
