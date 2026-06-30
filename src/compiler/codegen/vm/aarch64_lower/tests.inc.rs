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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: true, has_sme2: false, sve_vl: 32 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: true, has_sme2: false, sve_vl: 32 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: true, has_sme2: true, sve_vl: 32 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: true, has_sme2: false, sve_vl: 32 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
    fn test_dot_dtype_predicates() {
        // Arrange: exercise each DotDtype variant with its predicate function
        // Act & Assert: each predicate should return true only for its target variant
        assert!(dot_dtype_is_bf16(DotDtype::Bf16));
        assert!(!dot_dtype_is_bf16(DotDtype::Fp16));
        assert!(!dot_dtype_is_bf16(DotDtype::Int8));
        assert!(!dot_dtype_is_bf16(DotDtype::Int4x8));
        assert!(!dot_dtype_is_bf16(DotDtype::Fp4));

        assert!(dot_dtype_is_fp16(DotDtype::Fp16));
        assert!(!dot_dtype_is_fp16(DotDtype::Bf16));

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

        // Assert: default should be NEON baseline (no SVE2, no SME2, zero VL)
        assert!(!features.has_sve2, "Default should not have SVE2");
        assert!(!features.has_sme2, "Default should not have SME2");
        assert_eq!(features.sve_vl, 0, "Default SVE VL should be 0");
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: false, has_sme2: false, sve_vl: 16 },
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
            loop_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            platform: AArch64Features { has_sve2: true, has_sme2: false, sve_vl: 32 },
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

}
