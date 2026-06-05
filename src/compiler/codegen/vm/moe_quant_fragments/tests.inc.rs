#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantType;
    use crate::compiler::quant_format::QuantAlgoKind;
    use crate::dispatch::device_profile::DotProductCap;
    use crate::compiler::trace::QuantPrecision;
    use crate::compiler::codegen::vm::instr::SimdWidth;

    #[test]
    fn test_q5_0_selects_highbit_merge() {
        let desc = QuantAlgoKind::Q5_0.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q5_0 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::HighBitMerge { scale_offset, low_offset, high_offset, bias, high_bits } => {
                assert_eq!(*scale_offset, 0);
                assert_eq!(*low_offset, 6);
                assert_eq!(*high_offset, 2);
                assert_eq!(*bias, 16.0);
                assert_eq!(*high_bits, 1);
            }
            other => panic!("Q5_0 should select HighBitMerge, got {:?}", other),
        }
    }

    #[test]
    fn test_q5_1_selects_highbit_merge_with_min() {
        let desc = QuantAlgoKind::Q5_1.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q5_1 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::HighBitMerge { bias, high_bits, .. } => {
                assert_eq!(*bias, 0.0);
                assert_eq!(*high_bits, 1);
            }
            other => panic!("Q5_1 should select HighBitMerge, got {:?}", other),
        }
    }

    #[test]
    fn test_q6_k_selects_highbit_merge() {
        let desc = QuantAlgoKind::Q6K.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q6_K plan derive should succeed");

        match &plan.kernel {
            GemmKernel::HighBitMerge { bias, high_bits, .. } => {
                assert_eq!(*high_bits, 2);
                assert_eq!(*bias, 32.0);
            }
            other => panic!("Q6_K should select HighBitMerge, got {:?}", other),
        }
    }

    #[test]
    fn test_q4_0_still_selects_assisted() {
        let desc = QuantAlgoKind::Q4_0.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q4_0 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Assisted { .. } => {}
            other => panic!("Q4_0 should select Assisted, got {:?}", other),
        }
    }

    #[test]
    fn test_q2k_selects_dequant_fma() {
        let desc = QuantAlgoKind::Q2K.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q2K plan derive should succeed");

        match &plan.kernel {
            GemmKernel::DequantFma { .. } => {}
            other => panic!("Q2K should select DequantFma, got {:?}", other),
        }
    }

    #[test]
    fn test_q3k_selects_dequant_fma() {
        let desc = QuantAlgoKind::Q3K.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q3K plan derive should succeed");

        match &plan.kernel {
            GemmKernel::DequantFma { .. } => {}
            other => panic!("Q3K should select DequantFma, got {:?}", other),
        }
    }

    #[test]
    fn test_q8_0_still_selects_int8native() {
        let desc = QuantAlgoKind::Q8_0.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::NativeInt8Simd,
        ).expect("Q8_0 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Int8Native { .. } => {}
            other => panic!("Q8_0 should select Int8Native, got {:?}", other),
        }
    }

    #[test]
    fn test_q5_0_emit_gemm_produces_biplane_load() {
        let mut prog = super::super::instr::VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        let result = emit_quant_gemm_inline(
            &mut prog, BoundExpr::Const(1), 1, 256, QuantType::Q5_0,
            SimdWidth::W256, input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        );

        assert!(result.is_ok(), "Q5_0 emit failed: {:?}", result.err());
        assert!(prog.instrs.len() > 10, "Q5_0 GEMM should produce > 10 instrs, got {}", prog.instrs.len());
        let has_biplane = prog.instrs.iter().any(|i| matches!(i, VmInstr::QuantBiPlaneLoad { .. }));
        assert!(has_biplane, "Q5_0 GEMM should emit QuantBiPlaneLoad");
    }

    // ── Additional format coverage tests ─────────────────────────────────

    #[test]
    fn test_q4_1_selects_assisted() {
        let desc = QuantAlgoKind::Q4_1.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q4_1 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Assisted { .. } => {}
            other => panic!("Q4_1 should select Assisted, got {:?}", other),
        }
    }

    #[test]
    fn test_q4k_selects_assisted() {
        let desc = QuantAlgoKind::Q4K.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q4K plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Assisted { .. } => {}
            other => panic!("Q4K should select Assisted, got {:?}", other),
        }
    }

    #[test]
    fn test_q5k_selects_highbit_merge() {
        let desc = QuantAlgoKind::Q5K.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Q5K plan derive should succeed");

        match &plan.kernel {
            GemmKernel::HighBitMerge { bias, high_bits, .. } => {
                assert_eq!(*bias, 0.0);
                assert!(*high_bits >= 1, "Q5K high_bits should be >= 1");
            }
            other => panic!("Q5K should select HighBitMerge, got {:?}", other),
        }
    }

    #[test]
    fn test_iq3xxs_selects_dequant_fma() {
        let desc = QuantAlgoKind::IQ3XXS.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("IQ3_XXS plan derive should succeed");

        match &plan.kernel {
            GemmKernel::DequantFma => {}
            other => panic!("IQ3_XXS should select DequantFma, got {:?}", other),
        }
    }

    #[test]
    fn test_awq4_selects_assisted() {
        let desc = QuantAlgoKind::Awq4.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 128, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Awq4 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Assisted { .. } => {}
            other => panic!("Awq4 should select Assisted, got {:?}", other),
        }
    }

    #[test]
    fn test_gptq4_selects_assisted() {
        let desc = QuantAlgoKind::Gptq4.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 128, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Gptq4 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Assisted { .. } => {}
            other => panic!("Gptq4 should select Assisted, got {:?}", other),
        }
    }

    #[test]
    fn test_mxfp4_selects_dequant_fma() {
        let desc = QuantAlgoKind::Mxfp4.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Mxfp4 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::DequantFma => {}
            other => panic!("Mxfp4 should select DequantFma, got {:?}", other),
        }
    }

    #[test]
    fn test_nvfp4_selects_dequant_fma() {
        let desc = QuantAlgoKind::Nvfp4.descriptor();
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("Nvfp4 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::DequantFma => {}
            other => panic!("Nvfp4 should select DequantFma, got {:?}", other),
        }
    }

    // ── Edge case: zero dimensions return error ──────────────────────────

    #[test]
    fn test_derive_with_zero_k_returns_error() {
        let desc = QuantAlgoKind::Q4_0.descriptor();
        let result = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 0, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        );
        assert!(result.is_err(), "k=0 should return error");
    }

    #[test]
    fn test_derive_with_zero_n_returns_error() {
        let desc = QuantAlgoKind::Q4_0.descriptor();
        let result = QuantGemmPlan::derive(
            BoundExpr::Const(1), 0, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        );
        assert!(result.is_err(), "n=0 should return error");
    }

    // ── Gemv vs General mode (verified via kernel type) ───────────────────

    #[test]
    fn test_gemv_mode_for_m_equals_1() {
        let desc = QuantAlgoKind::Q4_0.descriptor();
        // m_bound=Const(1) should produce Gemv mode; verify plan succeeds
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(1), 1, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("m=1 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Assisted { .. } => {}
            other => panic!("Q4_0 m=1 should select Assisted, got {:?}", other),
        }
    }

    #[test]
    fn test_general_mode_for_m_greater_than_1() {
        let desc = QuantAlgoKind::Q4_0.descriptor();
        // m_bound=Const(4) should produce General mode; verify plan succeeds
        let plan = QuantGemmPlan::derive(
            BoundExpr::Const(4), 4, 256, &desc, SimdWidth::W256,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("m=4 plan derive should succeed");

        match &plan.kernel {
            GemmKernel::Assisted { .. } => {}
            other => panic!("Q4_0 m=4 should select Assisted, got {:?}", other),
        }
    }
}
