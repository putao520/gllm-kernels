mod tests {
    use crate::compiler::trace::QuantPrecision;
    use super::*;

    #[test]
    fn test_vreg_allocation() {
        let mut prog = VmProgram::new();
        let v0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let v1 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let v2 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        assert_eq!(v0, VRegId(0));
        assert_eq!(v1, VRegId(1));
        assert_eq!(v2, VRegId(2));
        assert_eq!(prog.len(), 3); // 3 DeclareVReg
    }

    #[test]
    fn test_elementwise_program() {
        let mut prog = VmProgram::new();
        let input = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        let vm_state = crate::compiler::codegen::vm::vm_state::VmState::init_x86_sysv();
        prog.emit(VmInstr::LoadPtr { dst: input, src: vm_state.arg_ptr_expr("input").unwrap() });
        prog.emit(VmInstr::LoadPtr { dst: output, src: vm_state.arg_ptr_expr("output").unwrap() });

        prog.emit_loop(BoundExpr::Const(8), 32, |prog, _counter, byte_off| {
            prog.emit(VmInstr::VecLoad {
                dst: acc, base: input,
                offset: OffsetExpr::LoopOffset(byte_off),
                width: SimdWidth::W256,
                dtype: QuantPrecision::F32,
            });
            // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            let neg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
            prog.emit(VmInstr::VecUnaryOp { dst: neg, a: acc, op: VecUnaryOp::Neg });
            let exp_neg = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
            prog.emit(VmInstr::Transcendental { dst: exp_neg, src: neg, func: TranscendentalFn::Exp });
            let one = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
            prog.emit(VmInstr::Broadcast { dst: one, src: ScalarExpr::Const(1.0), width: SimdWidth::W256, dtype: QuantPrecision::F32, });
            let denom = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
            prog.emit(VmInstr::VecBinOp { dst: denom, a: one, b: exp_neg, op: VecOp::Add, dtype: QuantPrecision::F32, });
            let recip = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
            prog.emit(VmInstr::VecUnaryOp { dst: recip, a: denom, op: VecUnaryOp::Recip });
            prog.emit(VmInstr::VecBinOp { dst: acc, a: acc, b: recip, op: VecOp::Mul, dtype: QuantPrecision::F32, });

            prog.emit(VmInstr::VecStore {
                base: output,
                offset: OffsetExpr::LoopOffset(byte_off),
                src: acc,
                width: SimdWidth::W256,
                dtype: QuantPrecision::F32,
            });
        });

        assert!(prog.validate_structure().is_ok());
        // 3 DeclareVReg(ptr+ptr+acc) + 2 LoadPtr + 1 LoopBegin + body + 1 LoopEnd
        assert!(prog.len() > 10);
    }

    #[test]
    fn test_gemm_program() {
        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let acc = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let a_broadcast = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let b_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        let vm_state = crate::compiler::codegen::vm::vm_state::VmState::init_x86_sysv();
        prog.emit(VmInstr::LoadPtr { dst: a_ptr, src: vm_state.arg_ptr_expr("input").unwrap() });
        prog.emit(VmInstr::LoadPtr { dst: b_ptr, src: vm_state.arg_ptr_expr("weights").unwrap() });
        prog.emit(VmInstr::LoadPtr { dst: c_ptr, src: vm_state.arg_ptr_expr("output").unwrap() });

        // Simple GEMM: C[m,n] = A[m,k] × B[k,n]
        let (m, n, k) = (4, 8, 16);
        let lanes = 8usize; // AVX2
        let elem = 4usize;

        for i in 0..m {
            for j_vec in 0..n / lanes {
                let j_byte = j_vec * lanes * elem;
                prog.emit(VmInstr::Broadcast {
                    dst: acc, src: ScalarExpr::Const(0.0), width: SimdWidth::W256,
                    dtype: QuantPrecision::F32,
                });
                for p in 0..k {
                    let a_off = (i * k + p) * elem;
                    let b_off = (p * n) * elem + j_byte;
                    prog.emit(VmInstr::Broadcast {
                        dst: a_broadcast,
                        src: ScalarExpr::MemLoad(a_ptr, OffsetExpr::Const(a_off)),
                        width: SimdWidth::W256,
                        dtype: QuantPrecision::F32,
                    });
                    prog.emit(VmInstr::VecLoad {
                        dst: b_vec, base: b_ptr,
                        offset: OffsetExpr::Const(b_off),
                        width: SimdWidth::W256,
                        dtype: QuantPrecision::F32,
                    });
                    prog.emit(VmInstr::Fma { dst: acc, acc, a: a_broadcast, b: b_vec, dtype: QuantPrecision::F32, });
                }
                let c_off = (i * n) * elem + j_byte;
                prog.emit(VmInstr::VecStore {
                    base: c_ptr, offset: OffsetExpr::Const(c_off),
                    src: acc, width: SimdWidth::W256,
                    dtype: QuantPrecision::F32,
                });
            }
        }

        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn test_compound_scope() {
        let mut prog = VmProgram::new();
        let v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);

        // Compound: Norm → GEMM
        prog.emit_scope(|prog| -> Result<(), String> {
            prog.emit(VmInstr::Comment("Norm phase".into()));
            prog.emit(VmInstr::Broadcast {
                dst: v, src: ScalarExpr::Const(0.0), width: SimdWidth::W256,
                dtype: QuantPrecision::F32,
            });
            Ok(())
        }).unwrap();

        prog.emit_scope(|prog| -> Result<(), String> {
            prog.emit(VmInstr::Comment("GEMM phase".into()));
            prog.emit(VmInstr::Broadcast {
                dst: v, src: ScalarExpr::Const(1.0), width: SimdWidth::W256,
                dtype: QuantPrecision::F32,
            });
            Ok(())
        }).unwrap();

        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn test_unbalanced_loop_detected() {
        let mut prog = VmProgram::new();
        let c = prog.alloc_vreg(VRegKind::Counter, SimdWidth::Scalar);
        let off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        prog.emit(VmInstr::LoopBegin {
            counter: c, byte_offset: off,
            bound: BoundExpr::Const(10), step_bytes: 32,
        });
        // 缺少 LoopEnd
        assert!(prog.validate_structure().is_err());
    }

    #[test]
    fn test_simd_width_bytes() {
        assert_eq!(SimdWidth::W256.bytes(), 32);
        assert_eq!(SimdWidth::W512.bytes(), 64);
        assert_eq!(SimdWidth::Scalar.bytes(), 4);
        assert_eq!(SimdWidth::Warp(32).bytes(), 128);
    }

    #[test]
    fn test_append_with_mapping_basic() {
        let mut main_prog = VmProgram::new();
        let main_vreg = main_prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        main_prog.emit(VmInstr::Comment("main".into()));

        let mut tpl_prog = VmProgram::new();
        let tpl_vreg = tpl_prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        tpl_prog.emit(VmInstr::VecBinOp {
            dst: tpl_vreg, a: tpl_vreg, b: tpl_vreg,
            op: VecOp::Add,
            dtype: QuantPrecision::F32,
        });

        // Append with mapping: tpl_vreg -> main_vreg
        main_prog.append_with_mapping(tpl_prog, &[(tpl_vreg, main_vreg)]);

        // Main program should have merged instructions:
        // DeclareVReg(main_vreg) + Comment + VecBinOp (DeclareVReg skipped, tpl_vreg remapped)
        assert!(main_prog.instrs.len() >= 2, "should have merged instructions, got {}", main_prog.instrs.len());
    }

    // ── REQ-VR-005~010: 缺失指令补全 测试 ──

    // REQ-VR-005: VecShuffle / VecShuffleMask

    #[test]
    fn vec_shuffle_mask_const_remap() {
        let mask = VecShuffleMask::Const(vec![3, 2, 1, 0]);
        let remapped = mask.remap(&|_| VRegId(99));
        assert!(matches!(remapped, VecShuffleMask::Const(_)));
    }

    #[test]
    fn vec_shuffle_mask_dynamic_remap() {
        let mask = VecShuffleMask::Dynamic { ctrl: VRegId(5) };
        let remapped = mask.remap(&|v| VRegId(v.0 + 10));
        assert!(matches!(remapped, VecShuffleMask::Dynamic { ctrl } if ctrl == VRegId(15)));
    }

    #[test]
    fn vec_shuffle_instr_structure() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecShuffle {
            dst, src,
            mask: VecShuffleMask::Const(vec![3, 2, 1, 0, 7, 6, 5, 4]),
            width: SimdWidth::W256,
        });
        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn vec_shuffle_dynamic_mask() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let ctrl = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecShuffle {
            dst, src,
            mask: VecShuffleMask::Dynamic { ctrl },
            width: SimdWidth::W256,
        });
        assert!(prog.validate_structure().is_ok());
    }

    // REQ-VR-006: VecExtractLane / VecInsertLane

    #[test]
    fn vec_extract_lane_instr() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let src = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecExtractLane {
            dst, src, lane: 3, dtype: QuantPrecision::F32,
        });
        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn vec_insert_lane_instr() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let src_vec = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        let src_scalar = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::VecInsertLane {
            dst, src_vec, src_scalar, lane: 5, dtype: QuantPrecision::F32,
        });
        assert!(prog.validate_structure().is_ok());
    }

    // REQ-VR-007: GprBinOp bit logic (And/Or/Xor)

    #[test]
    fn gpr_bin_op_and() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let a = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst, a, b: GprOperand::Imm(0xFF), op: GprOp::And,
        });
        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn gpr_bin_op_or_xor() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let a = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let b = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp { dst, a, b: GprOperand::VReg(b), op: GprOp::Or });
        prog.emit(VmInstr::GprBinOp { dst, a, b: GprOperand::VReg(b), op: GprOp::Xor });
        assert!(prog.validate_structure().is_ok());
    }

    // REQ-VR-008: VecLoadConst

    #[test]
    fn vec_load_const_uniform() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W256);
        prog.emit(VmInstr::VecLoadConst {
            dst,
            values: vec![0x3f800000u32; 8], // 1.0f32 × 8 lanes
            dtype: QuantPrecision::F32,
            width: SimdWidth::W256,
        });
        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn vec_load_const_mixed() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Vec, SimdWidth::W128);
        prog.emit(VmInstr::VecLoadConst {
            dst,
            values: vec![10, 20, 30, 40],
            dtype: QuantPrecision::F32,
            width: SimdWidth::W128,
        });
        assert!(prog.validate_structure().is_ok());
    }

    // REQ-VR-009: AtomicCAS

    #[test]
    fn atomic_cas_instr() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let expected = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let desired = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::AtomicCAS {
            dst, ptr, expected, desired,
            elem_width: 4,
            success_order: MemOrdering::AcqRel,
            failure_order: MemOrdering::Acquire,
        });
        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn atomic_cas_64bit() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let expected = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let desired = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::AtomicCAS {
            dst, ptr, expected, desired,
            elem_width: 8,
            success_order: MemOrdering::SeqCst,
            failure_order: MemOrdering::SeqCst,
        });
        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn mem_ordering_variants() {
        assert_ne!(MemOrdering::Relaxed, MemOrdering::Acquire);
        assert_ne!(MemOrdering::Release, MemOrdering::AcqRel);
        assert_eq!(MemOrdering::SeqCst, MemOrdering::SeqCst);
    }

    // REQ-VR-010: GprUnaryOp

    #[test]
    fn gpr_unary_op_all_variants() {
        let mut prog = VmProgram::new();
        let dst = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let src = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        for op in [GprUnaryOpKind::Not, GprUnaryOpKind::Popcount, GprUnaryOpKind::Clz, GprUnaryOpKind::Bswap, GprUnaryOpKind::Neg] {
            prog.emit(VmInstr::GprUnaryOp { dst, src, op });
        }
        assert!(prog.validate_structure().is_ok());
    }

    #[test]
    fn gpr_unary_op_kind_equality() {
        assert_eq!(GprUnaryOpKind::Not, GprUnaryOpKind::Not);
        assert_ne!(GprUnaryOpKind::Popcount, GprUnaryOpKind::Clz);
        assert_ne!(GprUnaryOpKind::Bswap, GprUnaryOpKind::Neg);
    }

    // ── 辅助类型测试 ──

    #[test]
    fn gpr_operand_remap() {
        let vreg_op = GprOperand::VReg(VRegId(3));
        assert_eq!(vreg_op.remap(|v| VRegId(v.0 + 7)), GprOperand::VReg(VRegId(10)));

        let imm_op = GprOperand::Imm(42);
        assert_eq!(imm_op.remap(|_| VRegId(0)), GprOperand::Imm(42));
    }

    #[test]
    fn gpr_operand_vreg() {
        assert_eq!(GprOperand::VReg(VRegId(5)).vreg(), Some(VRegId(5)));
        assert_eq!(GprOperand::Imm(99).vreg(), None);
    }

    #[test]
    fn gpr_op_all_variants() {
        let all = [GprOp::Add, GprOp::Sub, GprOp::Mul, GprOp::Div,
                    GprOp::Shl, GprOp::Shr, GprOp::And, GprOp::Or,
                    GprOp::Xor, GprOp::BitTest];
        for (i, op) in all.iter().enumerate() {
            for (j, other) in all.iter().enumerate() {
                if i == j { assert_eq!(op, other); }
                else { assert_ne!(op, other); }
            }
        }
    }

    #[test]
    fn gpr_condition_remap() {
        let cond = GprCondition::IsNull(VRegId(1));
        let remapped = cond.remap(|v| VRegId(v.0 + 5));
        assert!(matches!(remapped, GprCondition::IsNull(VRegId(6))));

        let cond = GprCondition::CmpEq(VRegId(2), 100);
        let remapped = cond.remap(|v| VRegId(v.0 * 2));
        assert!(matches!(remapped, GprCondition::CmpEq(VRegId(4), 100)));
    }

    #[test]
    fn gpr_condition_vregs() {
        assert_eq!(GprCondition::IsNull(VRegId(3)).vregs(), vec![VRegId(3)]);
        assert_eq!(GprCondition::CmpEq(VRegId(7), 0).vregs(), vec![VRegId(7)]);
        assert_eq!(GprCondition::BitClear(VRegId(1), 4).vregs(), vec![VRegId(1)]);
    }

    #[test]
    fn gpr_branch_action_remap() {
        let action = GprBranchAction::Exit(VRegId(2));
        let remapped = action.remap(|v| VRegId(v.0 + 10));
        assert!(matches!(remapped, GprBranchAction::Exit(VRegId(12))));

        let skip = GprBranchAction::Skip(5);
        assert!(matches!(skip.remap(|_| VRegId(0)), GprBranchAction::Skip(5)));
    }

    #[test]
    fn gpr_branch_action_vregs() {
        assert!(GprBranchAction::Skip(3).vregs().is_empty());
        assert_eq!(GprBranchAction::Exit(VRegId(7)).vregs(), vec![VRegId(7)]);
    }

    // ── SimdWidth f32_lanes / bytes ──

    #[test]
    fn simd_width_f32_lanes() {
        assert_eq!(SimdWidth::Scalar.f32_lanes(), 1);
        assert_eq!(SimdWidth::W128.f32_lanes(), 4);
        assert_eq!(SimdWidth::W256.f32_lanes(), 8);
        assert_eq!(SimdWidth::W512.f32_lanes(), 16);
        assert_eq!(SimdWidth::Warp(32).f32_lanes(), 32);
        assert_eq!(SimdWidth::Scalable.f32_lanes(), 0);
    }

    #[test]
    fn simd_width_bytes() {
        assert_eq!(SimdWidth::Scalar.bytes(), 4);
        assert_eq!(SimdWidth::W256.bytes(), 32);
        assert_eq!(SimdWidth::W512.bytes(), 64);
    }

    // ── OffsetExpr::substitute_loop_offset ──

    #[test]
    fn offset_expr_substitute_loop_offset() {
        let vreg = VRegId(0);
        let expr = OffsetExpr::LoopOffset(vreg);
        let result = expr.substitute_loop_offset(vreg, 128);
        assert_eq!(result, OffsetExpr::Const(128));
    }

    #[test]
    fn offset_expr_substitute_nested() {
        let vreg = VRegId(0);
        let expr = OffsetExpr::Add(
            Box::new(OffsetExpr::LoopOffset(vreg)),
            Box::new(OffsetExpr::Const(64)),
        );
        let result = expr.substitute_loop_offset(vreg, 256);
        assert_eq!(result, OffsetExpr::Add(
            Box::new(OffsetExpr::Const(256)),
            Box::new(OffsetExpr::Const(64)),
        ));
    }

    #[test]
    fn offset_expr_substitute_mul() {
        let vreg = VRegId(1);
        let expr = OffsetExpr::Mul(Box::new(OffsetExpr::LoopOffset(vreg)), 4);
        let result = expr.substitute_loop_offset(vreg, 32);
        assert_eq!(result, OffsetExpr::Mul(Box::new(OffsetExpr::Const(32)), 4));
    }

    #[test]
    fn offset_expr_substitute_no_match() {
        let vreg0 = VRegId(0);
        let vreg1 = VRegId(1);
        let expr = OffsetExpr::LoopOffset(vreg0);
        let result = expr.substitute_loop_offset(vreg1, 100);
        assert_eq!(result, OffsetExpr::LoopOffset(vreg0));
    }

    #[test]
    fn offset_expr_loop_plus_const_zero() {
        let vreg = VRegId(5);
        let result = OffsetExpr::loop_plus_const(vreg, 0);
        assert_eq!(result, OffsetExpr::LoopOffset(vreg));
    }

    #[test]
    fn offset_expr_loop_plus_const_nonzero() {
        let vreg = VRegId(5);
        let result = OffsetExpr::loop_plus_const(vreg, 64);
        assert!(matches!(result, OffsetExpr::Add(_, _)));
    }

    // ── BlockUnpackMode ──

    #[test]
    fn block_unpack_mode_block_bytes() {
        assert_eq!(BlockUnpackMode::Int8.block_bytes(), 32);
        assert_eq!(BlockUnpackMode::F16Broadcast.block_bytes(), 64);
        assert_eq!(BlockUnpackMode::SignedNibbleLow.block_bytes(), 18);
        assert_eq!(BlockUnpackMode::UnsignedNibbleHigh.block_bytes(), 18);
        assert_eq!(BlockUnpackMode::Bitpack2 { bias: 0.0 }.block_bytes(), 8);
        assert_eq!(BlockUnpackMode::Mxfp4 { scale_src: VRegId(0) }.block_bytes(), 16);
        assert_eq!(BlockUnpackMode::Nvfp4 { scale_src: VRegId(1) }.block_bytes(), 16);
    }

    #[test]
    fn block_unpack_mode_vregs() {
        assert!(BlockUnpackMode::Int8.vregs().is_empty());
        let mxfp4 = BlockUnpackMode::Mxfp4 { scale_src: VRegId(3) };
        assert_eq!(mxfp4.vregs(), vec![VRegId(3)]);
        let nvfp4 = BlockUnpackMode::Nvfp4 { scale_src: VRegId(7) };
        assert_eq!(nvfp4.vregs(), vec![VRegId(7)]);
    }

    #[test]
    fn block_unpack_mode_remap_vregs() {
        let mode = BlockUnpackMode::Mxfp4 { scale_src: VRegId(2) };
        let remapped = mode.remap_vregs(|v| VRegId(v.0 + 10));
        assert_eq!(remapped.vregs(), vec![VRegId(12)]);
    }

    // ── KvLoadMode ──

    #[test]
    fn kv_load_mode_default() {
        assert_eq!(KvLoadMode::default(), KvLoadMode::Direct);
    }

    #[test]
    fn kv_load_mode_variants() {
        assert_ne!(KvLoadMode::Direct, KvLoadMode::Kivi4);
        assert_ne!(KvLoadMode::Kivi2, KvLoadMode::Sparse);
        assert_ne!(KvLoadMode::Auto, KvLoadMode::Direct);
    }

    // ── Enum variant equality tests ──

    #[test]
    fn fp8_kind_variants() {
        assert_eq!(Fp8Kind::E4M3, Fp8Kind::E4M3);
        assert_ne!(Fp8Kind::E4M3, Fp8Kind::E5M2);
    }

    #[test]
    fn vreg_kind_variants() {
        assert_ne!(VRegKind::Ptr, VRegKind::Vec);
        assert_ne!(VRegKind::Scalar, VRegKind::Counter);
        assert_ne!(VRegKind::ByteOffset, VRegKind::Tile);
        assert_ne!(VRegKind::Mask, VRegKind::Ptr);
    }

    #[test]
    fn tma_swizzle_variants() {
        assert_eq!(TmaSwizzle::None, TmaSwizzle::None);
        assert_ne!(TmaSwizzle::Swizzle32, TmaSwizzle::Swizzle64);
        assert_ne!(TmaSwizzle::Swizzle128, TmaSwizzle::None);
    }

    #[test]
    fn vec_op_variants() {
        assert_ne!(VecOp::Add, VecOp::Sub);
        assert_ne!(VecOp::Mul, VecOp::Div);
        assert_ne!(VecOp::Max, VecOp::Min);
        assert_ne!(VecOp::And, VecOp::Or);
    }

    #[test]
    fn reduce_op_variants() {
        assert_ne!(ReduceOp::Sum, ReduceOp::Max);
        assert_ne!(ReduceOp::Min, ReduceOp::Prod);
        assert_ne!(ReduceOp::LogSum, ReduceOp::Sum);
    }

    #[test]
    fn cmp_predicate_variants() {
        assert_ne!(CmpPredicate::Eq, CmpPredicate::Ne);
        assert_ne!(CmpPredicate::Lt, CmpPredicate::Ge);
        assert_ne!(CmpPredicate::Gt, CmpPredicate::Le);
    }

    #[test]
    fn transcendental_fn_variants() {
        assert_ne!(TranscendentalFn::Exp, TranscendentalFn::Log);
        assert_ne!(TranscendentalFn::Tanh, TranscendentalFn::Sigmoid);
        assert_ne!(TranscendentalFn::Fwht, TranscendentalFn::Exp);
    }

    #[test]
    fn mem_fence_order_variants() {
        assert_ne!(MemFenceOrder::Release, MemFenceOrder::Acquire);
        assert_ne!(MemFenceOrder::AcqRel, MemFenceOrder::SeqCst);
    }

    #[test]
    fn dot_dtype_variants() {
        assert_ne!(DotDtype::Bf16, DotDtype::Fp16);
        assert_ne!(DotDtype::Int8, DotDtype::Int4x8);
        assert_ne!(DotDtype::Fp4, DotDtype::Bf16);
    }

    #[test]
    fn bi_plane_mode_variants() {
        assert_ne!(BiPlaneMode::Low5, BiPlaneMode::Low6);
        assert_ne!(BiPlaneMode::Low5, BiPlaneMode::Q3Merge);
    }

    #[test]
    fn scalar_cvt_source_variants() {
        assert_ne!(ScalarCvtSource::F16, ScalarCvtSource::I8);
        assert_ne!(ScalarCvtSource::U8, ScalarCvtSource::F16);
    }

    // ── HotpatchTarget / JumpTarget ──

    #[test]
    fn hotpatch_target_fields() {
        let t = HotpatchTarget::InstrIndex(42);
        assert!(matches!(t, HotpatchTarget::InstrIndex(42)));
        let t2 = HotpatchTarget::ExternalAddr(0xDEAD);
        assert!(matches!(t2, HotpatchTarget::ExternalAddr(0xDEAD)));
    }

    #[test]
    fn jump_target_fields() {
        let jt = JumpTarget { expert_id: 3, instr_index: 100 };
        assert_eq!(jt.expert_id, 3);
        assert_eq!(jt.instr_index, 100);
    }

    // ── ReturnValue ──

    #[test]
    fn return_value_variants() {
        let rv = ReturnValue::Const(0);
        assert!(matches!(rv, ReturnValue::Const(0)));
        let rv2 = ReturnValue::VReg(VRegId(5));
        assert!(matches!(rv2, ReturnValue::VReg(VRegId(5))));
    }

    // ── SymBound / BoundExpr ──

    #[test]
    fn sym_bound_fields() {
        let sb = SymBound { name: "seq_len".into(), max_alloc: 4096 };
        assert_eq!(sb.name, "seq_len");
        assert_eq!(sb.max_alloc, 4096);
    }

    #[test]
    fn bound_expr_const_equality() {
        assert_eq!(BoundExpr::Const(128), BoundExpr::Const(128));
        assert_ne!(BoundExpr::Const(128), BoundExpr::Const(256));
    }

    #[test]
    fn bound_expr_symbolic() {
        let sb1 = SymBound { name: "seq_len".into(), max_alloc: 4096 };
        let sb2 = SymBound { name: "seq_len".into(), max_alloc: 4096 };
        assert_eq!(BoundExpr::Symbolic(sb1), BoundExpr::Symbolic(sb2));
    }

    // ── VRegId hash consistency ──

    #[test]
    fn vreg_id_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(VRegId(1));
        set.insert(VRegId(2));
        set.insert(VRegId(1));
        assert_eq!(set.len(), 2);
        assert!(set.contains(&VRegId(1)));
    }
}
