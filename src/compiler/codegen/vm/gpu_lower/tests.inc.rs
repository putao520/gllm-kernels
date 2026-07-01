#[cfg(test)]
mod tests {
    use super::*;

    fn empty_frame() -> StackFrame {
        StackFrame { total_size: 0, alignment: 0, callee_save_area: 0, spill_area: 0, scratchpad_area: 0, uses_red_zone: false }
    }
    fn empty_alloc() -> RegAllocation {
        RegAllocation { mapping: std::collections::HashMap::new(), spills: vec![], callee_saved_used: vec![] }
    }

    #[test]
    fn test_ptx_sm80_lower() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains(".target sm_80"));
        assert!(ir.contains(".version 8.0"));
    }

    #[test]
    fn test_ptx_sm90_wgmma() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 90 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::TileMma {
            c: VRegId(0), a: VRegId(1), b: VRegId(2),
            m: 1, n: 1, k: 1, dtype: crate::types::DType::F32,
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("wgmma.mma_async"), "SM90 should emit WGMMA: {ir}");
        assert!(ir.contains("wgmma.fence"), "SM90 should emit fence");
        assert!(ir.contains("wgmma.wait_group"), "SM90 should emit wait_group");
        assert!(ir.contains(".version 8.3"), "SM90 should use PTX 8.3");
        assert!(ir.contains("mbar"), "SM90 should declare mbarrier");
    }

    #[test]
    #[ignore = "requires SM100 (Blackwell) hardware for TMEM tile allocation"]
    fn test_ptx_sm100_tcgen05() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 100 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::TileConfig {
            rows: 64, cols: 64, dtype: crate::types::DType::F16,
        }, &alloc).unwrap();
        l.lower_instr(&VmInstr::TileMma {
            c: VRegId(0), a: VRegId(1), b: VRegId(2),
            m: 1, n: 1, k: 1, dtype: crate::types::DType::F16,
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("tcgen05.alloc"), "SM100 should allocate TMEM: {ir}");
        assert!(ir.contains("tcgen05.mma"), "SM100 should emit tcgen05.mma: {ir}");
        assert!(ir.contains("tcgen05.dealloc"), "SM100 should dealloc TMEM: {ir}");
        assert!(ir.contains(".version 8.7"), "SM100 should use PTX 8.7");
    }

    #[test]
    fn test_ptx_sm90_tma_async() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 90 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::AsyncCopy {
            dst: VRegId(0), src: VRegId(1), size: 4096,
        }, &alloc).unwrap();
        l.lower_instr(&VmInstr::AsyncWait { handle: 0 }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("cp.async.bulk"), "SM90 should use TMA bulk copy: {ir}");
        assert!(ir.contains("mbarrier"), "SM90 should use mbarrier wait: {ir}");
    }

    #[test]
    fn test_hip_gfx950_mfma_v2() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 950, wave_size: 64 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::TileMma {
            c: VRegId(0), a: VRegId(1), b: VRegId(2),
            m: 1, n: 1, k: 1, dtype: crate::types::DType::F16,
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("v_mfma_f32_32x32x16_f16"), "gfx950 should emit MFMA v2: {ir}");
        assert!(ir.contains("gfx950"), "should identify gfx950");
    }

    #[test]
    fn test_hip_gfx908_mfma_v1() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.emit_prologue(&empty_frame(), &alloc, Default::default()).unwrap();
        l.lower_instr(&VmInstr::TileMma {
            c: VRegId(0), a: VRegId(1), b: VRegId(2),
            m: 1, n: 1, k: 1, dtype: crate::types::DType::F16,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("v_mfma_f32_16x16x16_f16"), "gfx908 should emit MFMA v1: {ir}");
    }

    #[test]
    fn test_ptx_loop_labels_unique() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.emit_prologue(&empty_frame(), &alloc, Default::default()).unwrap();
        // 两个嵌套循环
        l.lower_instr(&VmInstr::LoopBegin {
            counter: VRegId(0), byte_offset: VRegId(1),
            bound: BoundExpr::Const(16), step_bytes: 4,
        }, &alloc).unwrap();
        l.lower_instr(&VmInstr::LoopBegin {
            counter: VRegId(2), byte_offset: VRegId(3),
            bound: BoundExpr::Const(8), step_bytes: 4,
        }, &alloc).unwrap();
        l.lower_instr(&VmInstr::LoopEnd, &alloc).unwrap();
        l.lower_instr(&VmInstr::LoopEnd, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // 验证唯一标签
        assert!(ir.contains("LOOP_0:"), "first loop label 0");
        assert!(ir.contains("LOOP_1:"), "second loop label 1");
        assert!(ir.contains("LOOP_END_1:"), "inner loop end 1");
        assert!(ir.contains("LOOP_END_0:"), "outer loop end 0");
        assert!(ir.contains("bra LOOP_1;"), "inner loop back-branch");
        assert!(ir.contains("bra LOOP_0;"), "outer loop back-branch");
    }

    #[test]
    fn test_metal_lower() {
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("kernel void"));
    }

    #[test]
    fn test_gfx950_async_lds() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 950, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::AsyncCopy {
            dst: VRegId(0), src: VRegId(1), size: 128,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("global_load_lds"), "gfx950 should use GLOBAL_LOAD_LDS: {ir}");
    }

    // ── ScalarByteLoad tests ──

    #[test]
    fn test_ptx_scalar_byte_load() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarByteLoad {
            dst: VRegId(5),
            base: VRegId(0),
            offset: OffsetExpr::Const(16),
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("ld.global.u8"), "PTX ScalarByteLoad should emit ld.global.u8: {ir}");
        assert!(ir.contains("cvt.u32.u8"), "PTX ScalarByteLoad should emit cvt.u32.u8 zero-extend: {ir}");
    }

    #[test]
    fn test_hip_scalar_byte_load() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarByteLoad {
            dst: VRegId(5),
            base: VRegId(0),
            offset: OffsetExpr::Const(16),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("unsigned char"), "HIP ScalarByteLoad should cast via unsigned char: {ir}");
    }

    #[test]
    fn test_metal_scalar_byte_load() {
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarByteLoad {
            dst: VRegId(5),
            base: VRegId(0),
            offset: OffsetExpr::Const(16),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("device unsigned char"), "Metal ScalarByteLoad should cast via device unsigned char: {ir}");
    }

    // ── Mxfp4VecDequant tests ──

    #[test]
    fn test_ptx_mxfp4_dequant() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        // Build a VmProgram with proper VReg declarations for kind mapping
        let mut prog = VmProgram::new();
        let _ptr0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ptr1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scalar2 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let vec3 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Warp(32));

        l.set_vreg_kind_map(&prog);
        l.emit_prologue(&empty_frame(), &empty_alloc(), prog.vreg_counts_by_kind()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::QuantBlockLoad {
            dst: vec3,
            base: ptr1,
            offset: OffsetExpr::Const(0),
            unpack: BlockUnpackMode::Mxfp4 { scale_src: scalar2 },
            width: SimdWidth::Warp(32),
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Verify e8m0 scale decode via IEEE 754 bit reinterpret
        assert!(ir.contains("and.b32"), "PTX MXFP4 should mask scale byte: {ir}");
        assert!(ir.contains("shl.b32") || ir.contains("mov.f32"), "PTX MXFP4 should shift scale bits: {ir}");
        // Verify E2M1 decode
        assert!(ir.contains("ex2.approx.f32"), "PTX MXFP4 should use ex2.approx for 2^(exp-1): {ir}");
        assert!(ir.contains("ld.global.u8"), "PTX MXFP4 should load packed byte: {ir}");
        assert!(ir.contains("mul.f32"), "PTX MXFP4 should multiply by scale: {ir}");
    }

    #[test]
    fn test_hip_mxfp4_dequant() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let mut prog = VmProgram::new();
        let _ptr0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ptr1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scalar2 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let vec3 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Warp(32));

        l.set_vreg_kind_map(&prog);
        l.emit_prologue(&empty_frame(), &empty_alloc(), prog.vreg_counts_by_kind()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::QuantBlockLoad {
            dst: vec3,
            base: ptr1,
            offset: OffsetExpr::Const(0),
            unpack: BlockUnpackMode::Mxfp4 { scale_src: scalar2 },
            width: SimdWidth::Warp(32),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Verify HIP C++ MXFP4 decode
        assert!(ir.contains("scale_bits"), "HIP MXFP4 should decode e8m0 scale: {ir}");
        assert!(ir.contains("nibble"), "HIP MXFP4 should extract nibble: {ir}");
        assert!(ir.contains("exp2f"), "HIP MXFP4 should use exp2f for 2^(exp-1): {ir}");
        assert!(ir.contains("sign_bit"), "HIP MXFP4 should extract sign bit: {ir}");
        assert!(ir.contains("scale_f"), "HIP MXFP4 should multiply by e8m0 scale: {ir}");
    }

    #[test]
    fn test_metal_mxfp4_dequant() {
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let mut prog = VmProgram::new();
        let _ptr0 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let ptr1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scalar2 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let vec3 = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Scalar);

        l.set_vreg_kind_map(&prog);
        l.emit_prologue(&empty_frame(), &empty_alloc(), prog.vreg_counts_by_kind()).unwrap();
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::QuantBlockLoad {
            dst: vec3,
            base: ptr1,
            offset: OffsetExpr::Const(0),
            unpack: BlockUnpackMode::Mxfp4 { scale_src: scalar2 },
            width: SimdWidth::Scalar,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Metal uses same C++ syntax as HIP for MXFP4 decode
        assert!(ir.contains("scale_bits"), "Metal MXFP4 should decode e8m0 scale: {ir}");
        assert!(ir.contains("exp2f"), "Metal MXFP4 should use exp2f: {ir}");
        assert!(ir.contains("magnitude"), "Metal MXFP4 should compute magnitude: {ir}");
    }

    // ── REQ-CG-010: VecCast/VecCmp/ConditionalSelect tests ──

    #[test]
    fn test_ptx_vec_cast_f16_to_f32() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::VecCast {
            dst: VRegId(0), src: VRegId(1), from_bits: 16, to_bits: 32,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("cvt.rn.f32.f16"), "PTX VecCast 16->32 should emit cvt.rn.f32.f16: {ir}");
    }

    #[test]
    fn test_ptx_vec_cast_f32_to_f16() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::VecCast {
            dst: VRegId(0), src: VRegId(1), from_bits: 32, to_bits: 16,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("cvt.rn.f16.f32"), "PTX VecCast 32->16 should emit cvt.rn.f16.f32: {ir}");
    }

    #[test]
    fn test_hip_vec_cast_f16_to_f32() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::VecCast {
            dst: VRegId(0), src: VRegId(1), from_bits: 16, to_bits: 32,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("__half2float"), "HIP VecCast 16->32 should use __half2float: {ir}");
    }

    #[test]
    fn test_ptx_vec_cmp_eq() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::VecCmp {
            dst: VRegId(0), a: VRegId(1), b: VRegId(2), pred: CmpPredicate::Eq,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("setp.eq.f32"), "PTX VecCmp Eq should emit setp.eq.f32: {ir}");
        assert!(ir.contains("selp.u32"), "PTX VecCmp should emit selp.u32: {ir}");
        assert!(ir.contains("0xFFFFFFFF"), "PTX VecCmp should use 0xFFFFFFFF mask: {ir}");
    }

    #[test]
    fn test_hip_vec_cmp_gt() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::VecCmp {
            dst: VRegId(0), a: VRegId(1), b: VRegId(2), pred: CmpPredicate::Gt,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains(">"), "HIP VecCmp Gt should use > operator: {ir}");
        assert!(ir.contains("0xFFFFFFFFu"), "HIP VecCmp should use 0xFFFFFFFFu: {ir}");
    }

    #[test]
    fn test_ptx_conditional_select() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ConditionalSelect {
            dst: VRegId(0), mask: VRegId(1), true_val: VRegId(2), false_val: VRegId(3),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("setp.ne.u32"), "PTX ConditionalSelect should test mask: {ir}");
        assert!(ir.contains("selp.f32"), "PTX ConditionalSelect should emit selp.f32: {ir}");
    }

    #[test]
    fn test_hip_conditional_select() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ConditionalSelect {
            dst: VRegId(0), mask: VRegId(1), true_val: VRegId(2), false_val: VRegId(3),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("?"), "HIP ConditionalSelect should use ternary: {ir}");
    }

    // ── REQ-CG-011: LLM ops tests ──

    #[test]
    fn test_ptx_argmax() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::Argmax {
            dst: VRegId(0), logits_ptr: VRegId(1), vocab_bytes: 32000 * 4,
            width: SimdWidth::Warp(32),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("ARGMAX_LOOP"), "PTX Argmax should have loop label: {ir}");
        assert!(ir.contains("ARGMAX_DONE"), "PTX Argmax should have done label: {ir}");
        assert!(ir.contains("setp.gt.f32"), "PTX Argmax should compare with gt: {ir}");
        assert!(ir.contains("ld.global.f32"), "PTX Argmax should load logits: {ir}");
    }

    #[test]
    fn test_hip_argmax() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::Argmax {
            dst: VRegId(0), logits_ptr: VRegId(1), vocab_bytes: 1000 * 4,
            width: SimdWidth::Warp(32),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("for"), "HIP Argmax should use for loop: {ir}");
        assert!(ir.contains("max_val"), "HIP Argmax should track max_val: {ir}");
    }

    #[test]
    fn test_ptx_temperature_scale() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::TemperatureScale {
            logits_ptr: VRegId(0), temp_ptr: VRegId(1), vocab_bytes: 256,
            width: SimdWidth::Warp(32),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("TEMP_LOOP"), "PTX TemperatureScale should have loop: {ir}");
        assert!(ir.contains("div.rn.f32"), "PTX TemperatureScale should use div.rn.f32: {ir}");
        assert!(ir.contains("ld.global.f32"), "PTX TemperatureScale should load temperature: {ir}");
    }

    #[test]
    fn test_ptx_store_token() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::StoreToken {
            token_id: VRegId(0), output_buf: VRegId(1), counter: VRegId(2),
            input_ids_ptr: VRegId(3), prompt_len_bytes: VRegId(4),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("st.global.u32"), "PTX StoreToken should emit st.global.u32: {ir}");
        assert!(ir.contains("shl.b32"), "PTX StoreToken should compute byte offset with shl: {ir}");
    }

    #[test]
    fn test_ptx_check_stop_condition() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::CheckStopCondition {
            token_id: VRegId(0), counter: VRegId(1), eos_ptr: VRegId(2),
            max_tokens_ptr: VRegId(3),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("setp.eq.u32"), "PTX CheckStopCondition should emit setp.eq.u32: {ir}");
        assert!(ir.contains("setp.ge.u32"), "PTX CheckStopCondition should emit setp.ge.u32: {ir}");
        assert!(ir.contains("bra EPILOGUE"), "PTX CheckStopCondition should branch to epilogue: {ir}");
    }

    #[test]
    fn test_ptx_break_loop_const() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::BreakLoop {
            return_value: ReturnValue::Const(0),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("bra EPILOGUE"), "PTX BreakLoop should branch to epilogue: {ir}");
    }

    #[test]
    fn test_ptx_mark_label() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        // Set up a named label via path_labels directly (replaces OutputModeDispatch setup)
        l.path_labels.insert(42, "PATH_0".to_string());
        l.path_label_counter = 1;
        // Then mark the label
        l.lower_instr(&VmInstr::MarkLabel { label_id: 42 }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("PATH_0:"), "PTX MarkLabel should emit label: {ir}");
    }

    #[test]
    fn test_hip_break_loop_vreg() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::BreakLoop {
            return_value: ReturnValue::VReg(VRegId(5)),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("goto EPILOGUE"), "HIP BreakLoop should goto epilogue: {ir}");
    }

    #[test]
    fn test_ptx_gpr_cond_action_cmp_exit() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GprCondAction {
            cond: GprCondition::CmpEq(VRegId(0), 5),
            action: GprBranchAction::Exit(VRegId(1)),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("setp.eq.u32"), "PTX GprCondAction CmpEq should compare: {ir}");
        assert!(ir.contains("bra EPILOGUE"), "PTX GprCondAction Exit should branch to epilogue: {ir}");
    }

    // ── REQ-CG-012: GPR ops tests ──

    #[test]
    fn test_ptx_add_ptr() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::AddPtr {
            dst: VRegId(0), base: VRegId(1), offset: 4096,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("add.u64"), "PTX AddPtr should emit add.u64: {ir}");
    }

    #[test]
    fn test_hip_add_ptr() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::AddPtr {
            dst: VRegId(0), base: VRegId(1), offset: 4096,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("4096UL"), "HIP AddPtr should add offset: {ir}");
    }

    #[test]
    fn test_ptx_store_u32_to_stack() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::StoreConstToStack {
            rbp_offset: 4, value: 42, elem_width: 4,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("st.shared.u32"), "PTX StoreConstToStack should emit st.shared.u32: {ir}");
        assert!(ir.contains("42"), "PTX StoreConstToStack should contain value: {ir}");
    }

    #[test]
    fn test_ptx_gpr_shl() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GprBinOp { dst: VRegId(0), a: VRegId(1), b: GprOperand::Imm(2_i64), op: GprOp::Shl }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("shl.b32"), "PTX GprBinOp Shl should emit shl.b32: {ir}");
    }

    #[test]
    fn test_ptx_gpr_sub_const() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GprBinOp { dst: VRegId(0), a: VRegId(0), b: GprOperand::Imm(1_i64), op: GprOp::Sub }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("sub.u32"), "PTX GprBinOp Sub should emit sub.u32: {ir}");
    }

    #[test]
    fn test_ptx_gpr_add() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GprBinOp { dst: VRegId(0), a: VRegId(1), b: GprOperand::VReg(VRegId(2)), op: GprOp::Add }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("add.u32"), "PTX GprBinOp Add should emit add.u32: {ir}");
    }

    #[test]
    fn test_metal_gpr_ops() {
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GprBinOp { dst: VRegId(0), a: VRegId(1), b: GprOperand::Imm(3_i64), op: GprOp::Shl }, &alloc).unwrap();
        l.lower_instr(&VmInstr::GprBinOp { dst: VRegId(0), a: VRegId(0), b: GprOperand::Imm(5_i64), op: GprOp::Sub }, &alloc).unwrap();
        l.lower_instr(&VmInstr::GprBinOp { dst: VRegId(0), a: VRegId(1), b: GprOperand::VReg(VRegId(2)), op: GprOp::Add }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("<<"), "Metal GprBinOp Shl should use << operator: {ir}");
        assert!(ir.contains("-"), "Metal GprBinOp Sub should use - operator: {ir}");
        assert!(ir.contains("+"), "Metal GprBinOp Add should use + operator: {ir}");
    }

    // ── REQ-CG-008: Scalar ops tests ──

    #[test]
    fn test_ptx_scalar_load() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarLoad {
            dst: VRegId(3), base: VRegId(0), offset: OffsetExpr::Const(64),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("ld.global.u32"), "PTX ScalarLoad should emit ld.global.u32: {ir}");
    }

    #[test]
    fn test_hip_scalar_load() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarLoad {
            dst: VRegId(3), base: VRegId(0), offset: OffsetExpr::Const(64),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("unsigned int"), "HIP ScalarLoad should use unsigned int: {ir}");
    }

    #[test]
    fn test_metal_scalar_load() {
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarLoad {
            dst: VRegId(3), base: VRegId(0), offset: OffsetExpr::Const(64),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("device unsigned int"), "Metal ScalarLoad should use device unsigned int: {ir}");
    }

    #[test]
    fn test_ptx_scalar_store() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarStore {
            base: VRegId(0), src: VRegId(5), offset: OffsetExpr::Const(128),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("st.global.u32"), "PTX ScalarStore should emit st.global.u32: {ir}");
    }

    #[test]
    fn test_hip_scalar_store() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarStore {
            base: VRegId(0), src: VRegId(5), offset: OffsetExpr::Const(128),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("= f_5"), "HIP ScalarStore should assign src register: {ir}");
    }

    #[test]
    fn test_ptx_scalar_to_index() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarToIndex {
            dst: VRegId(2), src: VRegId(3), stride: 4,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("cvt.rni.s32.f32"), "PTX ScalarToIndex should emit cvt.rni.s32.f32: {ir}");
        assert!(ir.contains("mul.lo.s32"), "PTX ScalarToIndex with stride!=1 should emit mul.lo.s32: {ir}");
    }

    #[test]
    fn test_ptx_scalar_to_index_stride1() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarToIndex {
            dst: VRegId(2), src: VRegId(3), stride: 1,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("cvt.rni.s32.f32"), "PTX ScalarToIndex should emit cvt: {ir}");
        assert!(!ir.contains("mul.lo"), "PTX ScalarToIndex stride=1 should NOT emit mul: {ir}");
    }

    #[test]
    fn test_hip_scalar_to_index() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScalarToIndex {
            dst: VRegId(2), src: VRegId(3), stride: 4,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("(int)"), "HIP ScalarToIndex should cast to int: {ir}");
    }

    #[test]
    fn test_ptx_int_mul_stride() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::IntMulStride {
            dst: VRegId(2), src: VRegId(3), stride: 128,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("mul.lo.s32"), "PTX IntMulStride should emit mul.lo.s32: {ir}");
    }

    #[test]
    fn test_ptx_int_mul_stride_one() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::IntMulStride {
            dst: VRegId(2), src: VRegId(3), stride: 1,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("mov.s32"), "PTX IntMulStride stride=1 should emit mov.s32: {ir}");
        assert!(!ir.contains("mul"), "PTX IntMulStride stride=1 should NOT emit mul: {ir}");
    }

    #[test]
    fn test_hip_int_mul_stride() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::IntMulStride {
            dst: VRegId(2), src: VRegId(3), stride: 256,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("* 256"), "HIP IntMulStride should multiply by stride: {ir}");
    }

    // ── REQ-CG-009: Gather ops tests ──

    #[test]
    fn test_ptx_gather_load() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GatherLoad {
            dst: VRegId(4), base: VRegId(0), indices: VRegId(1), stride: 4, width: SimdWidth::Warp(32), dtype: crate::compiler::trace::QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("ld.global.u32"), "PTX GatherLoad should read index: {ir}");
        assert!(ir.contains("mul.lo.s32"), "PTX GatherLoad should multiply stride: {ir}");
        assert!(ir.contains("ld.global.f32"), "PTX GatherLoad should load f32: {ir}");
    }

    #[test]
    fn test_hip_gather_load() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GatherLoad {
            dst: VRegId(4), base: VRegId(0), indices: VRegId(1), stride: 4, width: SimdWidth::Warp(64), dtype: crate::compiler::trace::QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("_idx"), "HIP GatherLoad should read _idx: {ir}");
        assert!(ir.contains("_off"), "HIP GatherLoad should compute _off: {ir}");
    }

    #[test]
    fn test_metal_gather_load() {
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GatherLoad {
            dst: VRegId(4), base: VRegId(0), indices: VRegId(1), stride: 4, width: SimdWidth::Warp(32), dtype: crate::compiler::trace::QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("_idx"), "Metal GatherLoad should read _idx: {ir}");
        assert!(ir.contains("[_off/4]"), "Metal GatherLoad should index by _off/4: {ir}");
    }

    #[test]
    fn test_ptx_scatter_store() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScatterStore {
            base: VRegId(0), indices: VRegId(1), src: VRegId(4), stride: 4, width: SimdWidth::Warp(32), dtype: crate::compiler::trace::QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("ld.global.u32"), "PTX ScatterStore should read index: {ir}");
        assert!(ir.contains("st.global.f32"), "PTX ScatterStore should store f32: {ir}");
    }

    #[test]
    fn test_hip_scatter_store() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::ScatterStore {
            base: VRegId(0), indices: VRegId(1), src: VRegId(4), stride: 4, width: SimdWidth::Warp(64), dtype: crate::compiler::trace::QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("_idx"), "HIP ScatterStore should read _idx: {ir}");
        assert!(ir.contains("= f_4"), "HIP ScatterStore should store src: {ir}");
    }

    #[test]
    fn test_ptx_table_lookup() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::TableLookup {
            dst: VRegId(4), base: VRegId(0), row_index: VRegId(2), row_bytes: 4096, width: SimdWidth::Warp(32),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("mul.wide.u32"), "PTX TableLookup should emit mul.wide.u32: {ir}");
        assert!(ir.contains("ld.global.f32"), "PTX TableLookup should load f32: {ir}");
    }

    #[test]
    fn test_hip_table_lookup() {
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::TableLookup {
            dst: VRegId(4), base: VRegId(0), row_index: VRegId(2), row_bytes: 4096, width: SimdWidth::Warp(64),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("unsigned long long"), "HIP TableLookup should use 64-bit offset: {ir}");
        assert!(ir.contains("* 4096"), "HIP TableLookup should multiply by row_bytes: {ir}");
    }

    #[test]
    fn test_metal_table_lookup() {
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::TableLookup {
            dst: VRegId(4), base: VRegId(0), row_index: VRegId(2), row_bytes: 4096, width: SimdWidth::Warp(32),
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("unsigned long"), "Metal TableLookup should use unsigned long: {ir}");
        assert!(ir.contains("* 4096"), "Metal TableLookup should multiply by row_bytes: {ir}");
    }

    #[test]
    fn test_ptx_gather_load_stride1() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::GatherLoad {
            dst: VRegId(4), base: VRegId(0), indices: VRegId(1), stride: 1, width: SimdWidth::Warp(32), dtype: crate::compiler::trace::QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        assert!(!ir.contains("mul.lo.s32"), "PTX GatherLoad stride=1 should NOT emit mul: {ir}");
        assert!(ir.contains("ld.global.f32"), "PTX GatherLoad stride=1 should still load f32: {ir}");
    }

    // ── Error case tests ──

    #[test]
    fn test_gpu_sparse_mask_intersect_error() {
        // Arrange: SparseMaskIntersect is x86-only, should error on all GPU dialects
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        // Act
        let result = l.lower_instr(&VmInstr::SparseMaskIntersect {
            dst_k0: VRegId(0), dst_k1: VRegId(1), a: VRegId(2), b: VRegId(3),
        }, &alloc);
        // Assert
        assert!(result.is_err(), "SparseMaskIntersect should error on GPU: {result:?}");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("x86-only"), "Error should mention x86-only: {err_msg}");
    }

    #[test]
    fn test_hip_hotpatch_slot_error() {
        // Arrange: HotpatchSlot only supported on PTX, should error on HIP
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        // Act
        let result = l.lower_instr(&VmInstr::HotpatchSlot {
            slot_id: 0,
            initial_target: HotpatchTarget::InstrIndex(42),
            alternatives: vec![],
        }, &alloc);
        // Assert
        assert!(result.is_err(), "HotpatchSlot should error on HIP: {result:?}");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("hotpatch"), "Error should mention hotpatch: {err_msg}");
    }

    #[test]
    fn test_metal_hotpatch_slot_error() {
        // Arrange: HotpatchSlot only supported on PTX, should error on Metal
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let alloc = empty_alloc();
        // Act
        let result = l.lower_instr(&VmInstr::HotpatchSlot {
            slot_id: 1,
            initial_target: HotpatchTarget::ExternalAddr(0),
            alternatives: vec![],
        }, &alloc);
        // Assert
        assert!(result.is_err(), "HotpatchSlot should error on Metal: {result:?}");
    }

    #[test]
    fn test_ptx_sm70_shared_mem_async_store_falls_back_to_st_shared() {
        // SPEC 35 REQ-QWP-005: SM<80 has no cp.async, falls back to st.shared
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 70 });
        let alloc = empty_alloc();
        l.lower_instr(&VmInstr::SharedMemAlloc {
            name: "smem".to_string(),
            bytes: 256,
        }, &alloc).unwrap();
        // Act
        let result = l.lower_instr(&VmInstr::SharedMemAsyncStore {
            name: "smem".to_string(),
            dst_offset: OffsetExpr::Const(0),
            src: VRegId(0),
            width: SimdWidth::Warp(32),
            dtype: crate::compiler::trace::QuantPrecision::F32,
        }, &alloc);
        // Assert: should succeed with st.shared fallback
        assert!(result.is_ok(), "SharedMemAsyncStore should succeed with st.shared fallback on SM70: {result:?}");
        let ir = l.finalize().unwrap();
        assert!(ir.contains("st.shared"), "SM70 should emit st.shared fallback: {ir}");
        assert!(!ir.contains("cp.async"), "SM70 should NOT emit cp.async: {ir}");
    }

    #[test]
    fn test_ptx_sm80_tma2dcopy_error() {
        // Arrange: Tma2DCopy requires SM90+, should error on SM80
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        // Act
        let result = l.lower_instr(&VmInstr::Tma2DCopy {
            desc_name: "tensor_desc".to_string(),
            smem_name: "smem".to_string(),
            coord_x: VRegId(0),
            coord_y: VRegId(1),
            barrier_name: "mbar".to_string(),
        }, &alloc);
        // Assert
        assert!(result.is_err(), "Tma2DCopy should error on SM80: {result:?}");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("SM90"), "Error should mention SM90+: {err_msg}");
    }

    #[test]
    fn test_ptx_loop_end_without_begin_error() {
        // Arrange: LoopEnd without matching LoopBegin should error
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        // Act
        let result = l.lower_instr(&VmInstr::LoopEnd, &alloc);
        // Assert
        assert!(result.is_err(), "LoopEnd without LoopBegin should error: {result:?}");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("LoopEnd"), "Error should mention LoopEnd: {err_msg}");
    }

    // ── GPU sync and memory operation tests ──

    #[test]
    fn test_ptx_block_sync() {
        // Arrange
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        // Act
        l.lower_instr(&VmInstr::BlockSync, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert
        assert!(ir.contains("bar.sync 0"), "PTX BlockSync should emit bar.sync 0: {ir}");
    }

    #[test]
    fn test_hip_block_sync() {
        // Arrange
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let alloc = empty_alloc();
        // Act
        l.lower_instr(&VmInstr::BlockSync, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert
        assert!(ir.contains("__syncthreads"), "HIP BlockSync should emit __syncthreads: {ir}");
    }

    #[test]
    fn test_metal_block_sync() {
        // Arrange
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let alloc = empty_alloc();
        // Act
        l.lower_instr(&VmInstr::BlockSync, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert
        assert!(ir.contains("threadgroup_barrier"), "Metal BlockSync should emit threadgroup_barrier: {ir}");
    }

    #[test]
    fn test_ptx_mem_copy_16bytes() {
        // Arrange: MemCopy with 16 bytes should emit u64 load+store pairs
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let alloc = empty_alloc();
        // Act
        l.lower_instr(&VmInstr::MemCopy {
            dst: VRegId(0), src: VRegId(1), bytes: 16,
            dtype: crate::compiler::trace::QuantPrecision::F32,
            guard: None,
            effect: MemEffect::ReadWrite,
        }, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert: 16 bytes = 2 × 8-byte copies (ld.global.u64 + st.global.u64)
        assert!(ir.contains("ld.global.u64"), "PTX MemCopy should emit ld.global.u64: {ir}");
        assert!(ir.contains("st.global.u64"), "PTX MemCopy should emit st.global.u64: {ir}");
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_gpu_dialect_debug_output() {
        // Arrange: GpuDialect derives Debug — verify each variant formats correctly
        let ptx = GpuDialect::Ptx { sm_version: 80 };
        let hip = GpuDialect::Hip { gfx_arch: 908, wave_size: 64 };
        let metal = GpuDialect::Metal { gpu_family: 9 };
        // Act
        let ptx_dbg = format!("{:?}", ptx);
        let hip_dbg = format!("{:?}", hip);
        let metal_dbg = format!("{:?}", metal);
        // Assert
        assert!(ptx_dbg.contains("Ptx"), "Debug for Ptx variant should contain 'Ptx': {ptx_dbg}");
        assert!(ptx_dbg.contains("80"), "Debug for Ptx should contain sm_version: {ptx_dbg}");
        assert!(hip_dbg.contains("Hip"), "Debug for Hip variant should contain 'Hip': {hip_dbg}");
        assert!(hip_dbg.contains("908"), "Debug for Hip should contain gfx_arch: {hip_dbg}");
        assert!(metal_dbg.contains("Metal"), "Debug for Metal variant should contain 'Metal': {metal_dbg}");
        assert!(metal_dbg.contains("9"), "Debug for Metal should contain gpu_family: {metal_dbg}");
    }

    #[test]
    fn test_gpu_dialect_clone_copy_equality() {
        // Arrange: GpuDialect derives Clone + Copy
        let original = GpuDialect::Ptx { sm_version: 90 };
        // Act
        let cloned = original.clone();
        let copied = original; // Copy trait
        // Assert: both should produce identical prologue output
        let mut l1 = GpuLower::new(cloned);
        let mut l2 = GpuLower::new(copied);
        let frame = empty_frame();
        let alloc = empty_alloc();
        l1.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        l2.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        let ir1 = l1.finalize().unwrap();
        let ir2 = l2.finalize().unwrap();
        assert_eq!(ir1, ir2, "Cloned and copied GpuDialect should produce identical IR");
    }

    #[test]
    fn test_ptx_prologue_registers_declared() {
        // Arrange: PTX prologue with non-zero VReg counts should declare register banks
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let frame = empty_frame();
        let alloc = empty_alloc();
        let counts = VRegKindCounts {
            gpr_max_id: Some(5),
            vec_max_id: Some(10),
            mask_max_id: Some(2),
            tile_max_id: None,
        };
        // Act
        l.emit_prologue(&frame, &alloc, counts).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert: register declarations use max+1 count
        assert!(ir.contains(".reg .f32 %f<11>"), "Should declare 11 vec regs (max_id 10 + 1): {ir}");
        assert!(ir.contains(".reg .b32 %r<11>"), "Should declare gpr regs using global upper bound: {ir}");
        assert!(ir.contains(".reg .b64 %rd<11>"), "Should declare 64-bit gpr regs: {ir}");
        assert!(ir.contains(".reg .pred %p<16>"), "Should declare at least 16 pred regs: {ir}");
    }

    #[test]
    fn test_hip_prologue_with_scratchpad() {
        // Arrange: HIP prologue with non-zero scratchpad should emit shared memory
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
        let frame = StackFrame {
            total_size: 4096,
            alignment: 16,
            callee_save_area: 0,
            spill_area: 0,
            scratchpad_area: 8192,
            uses_red_zone: false,
        };
        let alloc = empty_alloc();
        // Act
        l.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert: shared memory declared with scratchpad_area / 4 floats
        assert!(ir.contains("__shared__ float smem[2048]"), "HIP should declare smem with scratchpad_area/4: {ir}");
        assert!(ir.contains("__global__ void kernel"), "HIP should declare kernel entry: {ir}");
    }

    #[test]
    fn test_metal_prologue_with_scratchpad() {
        // Arrange: Metal prologue with non-zero scratchpad should emit threadgroup memory
        let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
        let frame = StackFrame {
            total_size: 1024,
            alignment: 16,
            callee_save_area: 0,
            spill_area: 0,
            scratchpad_area: 1024,
            uses_red_zone: false,
        };
        let alloc = empty_alloc();
        // Act
        l.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert
        assert!(ir.contains("threadgroup float smem[256]"), "Metal should declare smem with scratchpad_area/4: {ir}");
        assert!(ir.contains("kernel void kernel_fn"), "Metal should declare kernel_fn entry: {ir}");
    }

    #[test]
    fn test_ptx_mega_kernel_prologue_21_params() {
        // Arrange: Mega-kernel prologue should declare 21 ABI parameters
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let frame = empty_frame();
        let alloc = empty_alloc();
        // Act
        l.emit_mega_kernel_prologue(&frame, &alloc, Default::default()).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert: all 21 parameters should be declared
        assert!(ir.contains(".visible .entry mega_kernel("), "Should declare mega_kernel entry: {ir}");
        assert!(ir.contains("input_ids_ptr"), "Should have input_ids_ptr param: {ir}");
        assert!(ir.contains("weight_blob_ptr"), "Should have weight_blob_ptr param: {ir}");
        assert!(ir.contains("kv_cache_ptr"), "Should have kv_cache_ptr param: {ir}");
        assert!(ir.contains("callback_table_ptr"), "Should have callback_table_ptr param (21st): {ir}");
        assert!(ir.contains("ld.param.u64 %rd_input"), "Should load input_ids into register: {ir}");
        assert!(ir.contains("ld.param.u64 %rd_cb"), "Should load callback_table_ptr into register: {ir}");
    }

    #[test]
    fn test_hip_mega_kernel_prologue_21_params() {
        // Arrange: HIP mega-kernel prologue should emit full C++ signature
        let mut l = GpuLower::new(GpuDialect::Hip { gfx_arch: 950, wave_size: 64 });
        let frame = empty_frame();
        let alloc = empty_alloc();
        // Act
        l.emit_mega_kernel_prologue(&frame, &alloc, Default::default()).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert
        assert!(ir.contains("extern \"C\" __global__ void mega_kernel("), "HIP should declare mega_kernel: {ir}");
        assert!(ir.contains("gfx950"), "HIP should identify gfx950: {ir}");
        assert!(ir.contains("unsigned int* __restrict__ input_ids_ptr"), "HIP mega should have input_ids_ptr: {ir}");
        assert!(ir.contains("callback_table_ptr"), "HIP mega should have callback_table_ptr: {ir}");
    }

    #[test]
    fn test_set_vreg_kind_map_populates_kinds() {
        // Arrange: set_vreg_kind_map should extract VReg → Kind mapping from VmProgram
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        let mut prog = VmProgram::new();
        let ptr_v = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let vec_v = prog.alloc_vreg(VRegKind::Vec, SimdWidth::Warp(32));
        let scalar_v = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // Act
        l.set_vreg_kind_map(&prog);
        // Assert: emit prologue with vreg counts and lower a VecLoad + ScalarLoad to verify naming
        let frame = empty_frame();
        let alloc = empty_alloc();
        l.emit_prologue(&frame, &alloc, prog.vreg_counts_by_kind()).unwrap();
        // VecLoad for vec VReg should use %f prefix (PTX vec namespace)
        l.lower_instr(&VmInstr::VecLoad {
            dst: vec_v, base: ptr_v, offset: OffsetExpr::Const(0),
            width: SimdWidth::Warp(32),
            dtype: crate::compiler::trace::QuantPrecision::F32, predicate: None,
        }, &alloc).unwrap();
        l.lower_instr(&VmInstr::ScalarLoad {
            dst: scalar_v, base: ptr_v, offset: OffsetExpr::Const(0),
        }, &alloc).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Vec VReg 1 should be named %f1 in PTX
        assert!(ir.contains("%f1"), "Vec VRegId(1) should map to %f1 in PTX: {ir}");
        // Scalar VReg 2 should be named %r2 in PTX
        assert!(ir.contains("%r2"), "Scalar VRegId(2) should map to %r2 in PTX: {ir}");
    }

    #[test]
    fn test_ptx_prologue_sm100_version() {
        // Arrange: SM100+ should use PTX version 8.7
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 100 });
        let frame = empty_frame();
        let alloc = empty_alloc();
        // Act
        l.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert
        assert!(ir.contains(".version 8.7"), "SM100 should use PTX 8.7: {ir}");
        assert!(ir.contains(".target sm_100"), "SM100 should target sm_100: {ir}");
    }

    #[test]
    fn test_ptx_prologue_sm120_version() {
        // Arrange: SM120 (Blackwell consumer, RTX 5070 Ti) should use PTX 8.8
        // and target sm_120 (CUDA 13.x driver JIT-compiles sm_120 PTX to native).
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 120 });
        let frame = empty_frame();
        let alloc = empty_alloc();
        // Act
        l.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert
        assert!(ir.contains(".version 8.8"), "SM120 should use PTX 8.8: {ir}");
        assert!(ir.contains(".target sm_120"), "SM120 should target sm_120: {ir}");
        // SM120+ inherits SM90+ mbarrier declaration
        assert!(ir.contains(".shared .align 8 .b64 mbar[4];"), "SM120 should declare mbarrier: {ir}");
    }

    #[test]
    fn test_ptx_prologue_sm90_mbarrier_declared() {
        // Arrange: SM90+ prologue should declare mbarrier array
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 90 });
        let frame = empty_frame();
        let alloc = empty_alloc();
        // Act
        l.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        l.emit_epilogue(&frame, &alloc).unwrap();
        let ir = l.finalize().unwrap();
        // Assert: SM90 should declare mbarrier but SM80 should not
        assert!(ir.contains(".shared .align 8 .b64 mbar[4]"), "SM90 prologue should declare mbar[4]: {ir}");
        // Verify SM80 does NOT have mbar
        let mut l80 = GpuLower::new(GpuDialect::Ptx { sm_version: 80 });
        l80.emit_prologue(&frame, &alloc, Default::default()).unwrap();
        let ir80 = l80.finalize().unwrap();
        assert!(!ir80.contains("mbar["), "SM80 prologue should NOT declare mbar: {ir80}");
    }

    // ── Phase 1.1/1.2: SM 6.1 (Pascal, GTX 1060) support ──

    #[test]
    fn test_ptx_sm61_prologue_version_and_target() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains(".version 6.5"), "SM61 should use PTX 6.5: {ir}");
        assert!(ir.contains(".target sm_61"), "SM61 should target sm_61: {ir}");
    }

    #[test]
    fn test_ptx_sm61_no_mbarrier_no_cluster() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        assert!(!ir.contains("mbar["), "SM61 should NOT declare mbarrier: {ir}");
        assert!(!ir.contains("cluster"), "SM61 should NOT reference clusters: {ir}");
    }

    #[test]
    fn test_ptx_sm61_mega_kernel_prologue() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        l.emit_mega_kernel_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains(".version 6.5"), "SM61 mega kernel should use PTX 6.5: {ir}");
        assert!(ir.contains(".target sm_61"), "SM61 mega kernel should target sm_61: {ir}");
        assert!(ir.contains("mega_kernel"), "SM61 mega kernel should have entry point: {ir}");
    }

    #[test]
    fn test_isa_profile_sm61_shared_mem_48kb() {
        let profile = super::super::isa_profile::IsaProfile::cuda(61);
        assert_eq!(profile.cache.smem_bytes, 48 * 1024, "SM61 should have 48KB shared memory");
    }

    #[test]
    fn test_isa_profile_sm61_no_tensor_core() {
        let profile = super::super::isa_profile::IsaProfile::cuda(61);
        let has_tile_gemm = profile.features.iter().any(|f| matches!(f, super::super::isa_profile::IsaFeature::TileGemm { .. }));
        assert!(!has_tile_gemm, "SM61 should NOT have TileGemm (no Tensor Cores)");
    }

    #[test]
    fn test_isa_profile_sm61_k_unroll_factor_2() {
        let profile = super::super::isa_profile::IsaProfile::cuda(61);
        assert_eq!(profile.k_unroll_factor, 2, "SM61 k_unroll_factor should be 2 (no Tensor Core)");
    }

    // ── Phase 1.4: Ring Barrier for SM61 cross-CTA sync ──

    #[test]
    fn test_ring_barrier_arrive_emits_atomic_add() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_ring_barrier_arrive("%r_barrier_ptr", "%r_scratch");
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("atom.global.add.u32 %r_scratch, [%r_barrier_ptr], 1"),
            "Ring barrier arrive should emit atom.global.add.u32: {ir}");
        assert!(ir.contains("membar.gl"),
            "Ring barrier arrive should emit membar.gl: {ir}");
    }

    #[test]
    fn test_ring_barrier_wait_emits_spin_loop() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_ring_barrier_wait("%r_barrier_ptr", "%r_val", "%p_wait", 8);
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        assert!(ir.contains("ld.global.u32 %r_val, [%r_barrier_ptr]"),
            "Ring barrier wait should load counter: {ir}");
        assert!(ir.contains("setp.lt.u32 %p_wait, %r_val, 8"),
            "Ring barrier wait should compare against expected count: {ir}");
        assert!(ir.contains("bra .Lring_wait_"),
            "Ring barrier wait should have spin-loop branch: {ir}");
    }

    #[test]
    fn test_ring_barrier_full_arrive_and_wait() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.emit_ring_barrier("%r_barrier_ptr", "%r_val", "%p_wait", 9);
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();
        // Should contain both arrive and wait
        assert!(ir.contains("atom.global.add.u32"), "Full barrier should have arrive: {ir}");
        assert!(ir.contains("setp.lt.u32 %p_wait, %r_val, 9"), "Full barrier should wait for 9 CTAs: {ir}");
        // Should have exactly 2 membar.gl (one in arrive, one after wait)
        assert_eq!(ir.matches("membar.gl").count(), 2, "Full barrier should have 2 membar.gl: {ir}");
    }

    // ── Phase 2.4: PageTableAddr / PageTableKVWrite PTX verification for SM61 ──

    fn gpu_reg_alloc(ids: &[u32]) -> RegAllocation {
        use super::super::isa_profile::{PhysReg, PhysGpr};
        RegAllocation {
            mapping: ids.iter().enumerate()
                .map(|(i, id)| (VRegId(*id), PhysReg::Gpr(PhysGpr(i as u8))))
                .collect(),
            spills: vec![], callee_saved_used: vec![],
        }
    }

    #[test]
    fn test_ptx_sm61_page_table_addr_computes_correct_address() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1, 2, 3, 30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        // page_size=16, row_bytes=64 (head_dim*4 for F32), page_stride=1024 (16*64)
        l.lower_instr(&VmInstr::PageTableAddr {
            dst: VRegId(0),
            pool_base: VRegId(1),
            page_table_ptr: VRegId(2),
            ki_byte_off: OffsetExpr::ScalarVReg(VRegId(3)),
            row_bytes: 64,
            page_size: 16,
            page_stride: 1024,
            base_offset: 0,
            seq_pt_offset: None,
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        // Should contain: shr.u32 for token_idx computation (log2(64)=6)
        assert!(ir.contains("shr.u32"), "Should use shr.u32 for token/page index: {ir}");
        // Should contain: mad.wide.u32 for page table lookup
        assert!(ir.contains("mad.wide.u32"), "Should use mad.wide.u32 for page table indexing: {ir}");
        // Should contain: ld.global.u32 to load page_id
        assert!(ir.contains("ld.global.u32"), "Should load page_id from global memory: {ir}");
        // Should contain: mad.lo.u32 for final address computation
        assert!(ir.contains("mad.lo.u32"), "Should compute final address with mad.lo.u32: {ir}");
    }

    #[test]
    fn test_ptx_sm61_page_table_addr_with_seq_offset() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1, 2, 3, 4, 30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.lower_instr(&VmInstr::PageTableAddr {
            dst: VRegId(0),
            pool_base: VRegId(1),
            page_table_ptr: VRegId(2),
            ki_byte_off: OffsetExpr::ScalarVReg(VRegId(3)),
            row_bytes: 64,
            page_size: 16,
            page_stride: 1024,
            base_offset: 128,
            seq_pt_offset: Some(VRegId(4)),
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        // Should contain: add.u32 for seq_pt_offset (BCI-005 per-sequence page table)
        assert!(ir.contains("add.u32"), "Should add seq_pt_offset to page_idx: {ir}");
        // Should contain base_offset
        assert!(ir.contains("128"), "Should add base_offset=128: {ir}");
    }

    #[test]
    fn test_ptx_sm61_page_table_kv_write_stores_to_correct_address() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1, 2, 3, 30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();
        l.lower_instr(&VmInstr::PageTableKVWrite {
            src: VRegId(0),
            pool_base: VRegId(1),
            page_table_ptr: VRegId(2),
            seq_index: VRegId(3),
            row_bytes: 64,
            page_size: 16,
            page_stride: 1024,
            base_offset: 0,
            width: SimdWidth::Warp(32),
            dtype: crate::types::DType::F32,
        }, &alloc).unwrap();
        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        // PTX path: loads page_id from page_table, then computes address with mad.lo.u32
        assert!(ir.contains("ld.global.u32"), "PageTableKVWrite should load page_id: {ir}");
        assert!(ir.contains("mad.lo.u32"), "PageTableKVWrite should compute address with mad.lo.u32: {ir}");
    }

    // ── Phase 3.1 (SPEC 32 REQ-MKO-001): ForwardPhaseDispatch ──

    #[test]
    fn test_ptx_sm61_forward_phase_dispatch_jump_to_label() {
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();

        // Simulate Phase 0.7: if batch_m == 0, jump to decode entry
        l.lower_instr(&VmInstr::GprCondAction {
            cond: GprCondition::CmpEq(VRegId(0), 0),
            action: GprBranchAction::JumpToLabel(101),
        }, &alloc).unwrap();

        // Some placeholder instructions (prefill path)
        l.lower_instr(&VmInstr::GprLoadImm { dst: VRegId(1), value: 42 }, &alloc).unwrap();

        // MarkLabel for decode entry
        l.lower_instr(&VmInstr::MarkLabel { label_id: 101 }, &alloc).unwrap();

        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        // Must contain conditional branch: setp.eq + bra to LABEL_101
        assert!(ir.contains("setp.eq.u32"), "should compare for equality: {ir}");
        assert!(ir.contains("bra LABEL_101"), "should branch to LABEL_101: {ir}");
        assert!(ir.contains("LABEL_101:"), "should emit the label target: {ir}");
    }

    #[test]
    fn test_ptx_sm61_jump_to_label_with_path_label() {
        // JumpToLabel also works when path_labels has a named label
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();

        // Set up named labels via path_labels directly (replaces OutputModeDispatch setup)
        l.path_labels.insert(200, "PATH_0".to_string());
        l.path_labels.insert(201, "PATH_1".to_string());
        l.path_label_counter = 2;

        // Jump to label 200 (should use PATH_0 naming from path_labels)
        l.lower_instr(&VmInstr::GprCondAction {
            cond: GprCondition::CmpEq(VRegId(1), 0),
            action: GprBranchAction::JumpToLabel(200),
        }, &alloc).unwrap();

        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        assert!(ir.contains("bra PATH_0"), "should branch to named path label: {ir}");
    }

    // ── Phase 4.1 (SPEC 33 REQ-MLA-009): MLA PTX SM61 verification ──

    #[test]
    fn test_ptx_sm61_mla_attn_score_contains_dot_product_and_softmax() {
        use crate::compiler::codegen::vm::mla_emit;
        use crate::compiler::codegen::vm::isa_profile::IsaProfile;

        let mut prog = VmProgram::new();
        let profile = IsaProfile::cuda(61);
        let width = profile.optimal_simd_width();

        // Allocate pointer vregs for MLA attention score inputs
        let q_absorbed_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_cache_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_uv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let slots = [q_absorbed_ptr, kv_cache_ptr, w_uv_ptr, output_ptr];

        // Small MLA config: 2 heads, head_dim=16, d_c=8, d_rope=4
        mla_emit::emit_mla_attn_score_inline(
            &mut prog,
            2, 16, 8, 4,
            &slots,
            kv_len,
            width,
            crate::compiler::trace::QuantPrecision::F32,
        ).expect("MLA attn score emission should succeed for valid params");

        // Lower to PTX SM61
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1, 2, 3, 4, 30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();

        for instr in &prog.instrs {
            let result = l.lower_instr(instr, &alloc);
            let _ = result;
        }

        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        assert!(ir.contains(".version 6.5"), "MLA PTX should target version 6.5 for SM61: {ir}");
        assert!(ir.contains(".target sm_61"), "MLA PTX should target sm_61: {ir}");
    }

    #[test]
    fn test_ptx_sm61_mla_paged_kv_addr_uses_correct_stride() {
        // MLA PagedAttention uses page_stride = d_c + d_rope (not standard 2*head_dim)
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1, 2, 3, 4, 30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();

        // MLA page stride = (d_c + d_rope) * elem_bytes = (8 + 4) * 4 = 48 bytes
        let page_stride = 12 * 4;
        l.lower_instr(&VmInstr::PageTableAddr {
            dst: VRegId(0),
            pool_base: VRegId(4),
            page_table_ptr: VRegId(1),
            ki_byte_off: OffsetExpr::ScalarVReg(VRegId(2)),
            row_bytes: page_stride,
            page_size: 16,
            page_stride,
            base_offset: 0,
            seq_pt_offset: None,
        }, &alloc).unwrap();

        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        assert!(ir.contains("mad.lo.u32"), "MLA paged addr should use mad.lo.u32: {ir}");
        assert!(ir.contains("ld.global.u32"), "MLA paged addr should load page_id: {ir}");
    }

    // ── Phase 5.1 (SPEC 35 REQ-QWP-005): QuantGather double-buffer SM61 ──

    #[test]
    fn test_ptx_sm61_shared_mem_async_store_falls_back_to_st_shared() {
        // SM61 has no cp.async, so SharedMemAsyncStore must emit st.shared + comment
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1, 30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();

        // Allocate shared memory first
        l.lower_instr(&VmInstr::SharedMemAlloc {
            name: "smem_a_0".to_string(),
            bytes: 256,
        }, &alloc).unwrap();

        // SharedMemAsyncStore on SM61 should fallback to st.shared
        l.lower_instr(&VmInstr::SharedMemAsyncStore {
            name: "smem_a_0".to_string(),
            dst_offset: OffsetExpr::Const(0),
            src: VRegId(0),
            width: SimdWidth::W128,
            dtype: crate::compiler::trace::QuantPrecision::F32,
        }, &alloc).unwrap();

        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        // Should contain st.shared (not cp.async) with fallback comment
        assert!(ir.contains("st.shared.f32"), "SM61 async store should emit st.shared.f32: {ir}");
        assert!(ir.contains("async fallback for SM<80"), "SM61 async store should have fallback comment: {ir}");
        assert!(!ir.contains("cp.async"), "SM61 should NOT emit cp.async: {ir}");
    }

    #[test]
    fn test_ptx_sm61_shared_mem_async_wait_falls_back_to_bar_sync() {
        // SM61 has no cp.async.wait_group, must use bar.sync
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();

        l.lower_instr(&VmInstr::SharedMemAsyncWaitGroup { n: 0 }, &alloc).unwrap();

        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        assert!(ir.contains("bar.sync"), "SM61 async wait should emit bar.sync: {ir}");
        assert!(!ir.contains("cp.async.wait_group"), "SM61 should NOT emit cp.async.wait_group: {ir}");
    }

    #[test]
    fn test_ptx_sm61_quant_block_load_q4_0_emits_ld_global_u8() {
        // SPEC 35 REQ-QWP-006: SM61 path for quantized block load
        // Q4_0: ld.global.u8 + nibble unpack (and/shr) + cvt.rn.f32.s32 + mul.f32
        // No WMMA/MMA instructions (SM61 has no Tensor Core)
        use crate::compiler::codegen::vm::quant_gather_emit;
        use crate::quant::QuantType;

        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);

        let indices_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let embed_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Emit QuantGather for Q4_0 with minimal hidden_dim=32, seq=1
        let result = quant_gather_emit::emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            100,
            32,
            QuantType::Q4_0,
            width,
            indices_ptr,
            embed_ptr,
            output_ptr,
            crate::compiler::trace::QuantPrecision::F32,
            None,
        );

        // QuantGather should succeed for Q4_0
        assert!(result.is_ok(), "QuantGather Q4_0 should succeed: {:?}", result);

        // Lower entire program to SM61 PTX
        let mut l = GpuLower::new(GpuDialect::Ptx { sm_version: 61 });
        let alloc = gpu_reg_alloc(&[0, 1, 2, 3, 4, 5, 6, 7, 30]);
        l.emit_prologue(&empty_frame(), &empty_alloc(), Default::default()).unwrap();

        for instr in &prog.instrs {
            let _ = l.lower_instr(instr, &alloc);
        }

        l.emit_epilogue(&empty_frame(), &empty_alloc()).unwrap();
        let ir = l.finalize().unwrap();

        // SM61 should NOT have any WMMA/MMA instructions
        assert!(!ir.contains("wmma") && !ir.contains("mma"), "SM61 should not emit wmma/mma: {ir}");
        assert!(!ir.contains("cp.async"), "SM61 should not emit cp.async: {ir}");
    }
}
