mod tests {
    use super::*;
    use crate::compiler::graph::{Op, GemmSpec, NormSpec, RopeSpec};
    use crate::types::DType;

    #[test]
    fn registry_with_defaults_has_all_ops() {
        let reg = ScalarOpRegistry::with_defaults();

        let expected_keys = [
            OpKindKey::Silu,
            OpKindKey::Gelu,
            OpKindKey::Tanh,
            OpKindKey::Sigmoid,
            OpKindKey::SwiGlu,
            OpKindKey::SwiGluClipped,
            OpKindKey::GeGlu,
            OpKindKey::Add,
            OpKindKey::Mul,
            OpKindKey::Residual,
            OpKindKey::Softmax,
            OpKindKey::RmsNorm,
            OpKindKey::LayerNorm,
            OpKindKey::Gemm,
            OpKindKey::GemmBias,
            OpKindKey::RoPE,
            OpKindKey::Transpose,
            OpKindKey::Reshape,
            OpKindKey::QuantGemm,
            OpKindKey::Dequantize,
            OpKindKey::L2Normalize,
            OpKindKey::QkNorm,
            OpKindKey::HeadRmsNorm,
            OpKindKey::ValueNorm,
            OpKindKey::MeanPool,
            OpKindKey::MultiHeadAttention,
            OpKindKey::CachedGQA,
            OpKindKey::MoEGate,
            OpKindKey::TopK,
            OpKindKey::WeightedSum,
            // P4/P5 OpKind
            OpKindKey::GateMask,
            OpKindKey::AttentionSkipMask,
            OpKindKey::LayerBypass,
            OpKindKey::ResidualWithTelemetry,
            OpKindKey::MoEConditionalAdd,
            OpKindKey::SoftmaxWithEntropy,
            OpKindKey::FusedRmsNormGemm,
            OpKindKey::MaskedGemm,
            OpKindKey::EntropyGate,
            OpKindKey::VRangeQuant,
            OpKindKey::KvCentroidPrefetch,
            OpKindKey::VariableLengthBatch,
            OpKindKey::AltUpPredict,
            OpKindKey::AltUpCorrect,
            OpKindKey::AltUpInject,
            OpKindKey::DepthwiseConv1D,
            OpKindKey::PatchEmbed,
            OpKindKey::LearnedPos2D,
            OpKindKey::DequantizeMxfp4,
            OpKindKey::MoEDispatchPacked,
            OpKindKey::MoERouter,
            OpKindKey::Argmax,
            OpKindKey::LogitSoftcap,
            OpKindKey::Gather,
            OpKindKey::QuantGather,
            OpKindKey::ColumnSlice,
            OpKindKey::StoreToken,
            OpKindKey::CheckStopCondition,
            OpKindKey::WriteLogits,
            OpKindKey::EarlyExit,
            OpKindKey::GuardrailCheck,
            OpKindKey::SgInject,
            OpKindKey::SgDetect,
            OpKindKey::CotStepCheck,
            OpKindKey::SessionKvRestore,
            OpKindKey::MmHiddenInject,
            OpKindKey::MtpDraft,
        ];

        // All ops now have traces injected (QuantGather uses parameterized trace template)
        let no_trace_keys: &[OpKindKey] = &[];

        for key in &expected_keys {
            assert!(
                reg.get_signature(key).is_some(),
                "missing signature for {key:?}"
            );
            if !no_trace_keys.contains(key) {
                assert!(
                    reg.get_trace(key).is_some(),
                    "missing trace for {key:?}"
                );
            }
        }

        assert_eq!(reg.num_entries(), expected_keys.len());
        assert_eq!(reg.num_traces(), expected_keys.len() - no_trace_keys.len());
    }

    #[test]
    fn registry_key_from_op_roundtrip() {
        // OpKind enum 已删除，从 Op 派生 OpKindKey。
        let cases: Vec<(Op, OpKindKey)> = vec![
            (Op::Silu, OpKindKey::Silu),
            (Op::Gelu, OpKindKey::Gelu),
            (Op::SwiGlu, OpKindKey::SwiGlu),
            (Op::SwiGluClipped { limit: 7.0 }, OpKindKey::SwiGluClipped),
            (Op::GeGlu, OpKindKey::GeGlu),
            (Op::Add, OpKindKey::Add),
            (Op::Mul, OpKindKey::Mul),
            (Op::Residual, OpKindKey::Residual),
            (Op::Softmax, OpKindKey::Softmax),
            (Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-6, dtype: DType::F32, has_weight: true }), OpKindKey::RmsNorm),
            (Op::LayerNorm(NormSpec { feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true }), OpKindKey::LayerNorm),
            (Op::ValueNorm(NormSpec { feature_dim: 4096, eps: 1e-6, dtype: DType::F32, has_weight: false }), OpKindKey::ValueNorm),
            (Op::Gemm(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: false }), OpKindKey::Gemm),
            (Op::GemmBias(GemmSpec { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false, has_bias: true }), OpKindKey::GemmBias),
            (Op::RoPE(RopeSpec { num_heads: 0, head_dim: 128, theta: 10000.0, partial: 1.0, rope_scaling: None }), OpKindKey::RoPE),
            (Op::Transpose { perm: vec![1, 0] }, OpKindKey::Transpose),
            (Op::Reshape { target_shape: vec![1, 4096] }, OpKindKey::Reshape),
        ];

        let _reg = ScalarOpRegistry::with_defaults();
        for (op, expected_key) in &cases {
            let key = ScalarOpRegistry::key_from_op(op);
            assert_eq!(&key, expected_key, "key mismatch for {op:?}");
        }
    }

    #[test]
    fn registry_silu_trace_body_valid_ssa() {
        let reg = ScalarOpRegistry::with_defaults();
        let trace = reg.get_trace(&OpKindKey::Silu).unwrap();

        if let ComputePattern::Elementwise { body } = &trace.pattern {
            assert_eq!(body.len(), 6);
            assert_eq!(body[0], TraceOp::Input(0));
            assert_eq!(body[5], TraceOp::Div(ValueId(0), ValueId(4)));
        } else {
            panic!("SiLU should be Elementwise");
        }
    }

    #[test]
    fn registry_gelu_trace_body_valid_ssa() {
        let reg = ScalarOpRegistry::with_defaults();
        let trace = reg.get_trace(&OpKindKey::Gelu).unwrap();

        if let ComputePattern::Elementwise { body } = &trace.pattern {
            // The trace may come from symexec (auto-extracted) or the manual
            // fallback. Both are valid as long as the structural invariants hold.
            assert!(body.len() >= 10, "GELU trace too short: {} ops", body.len());
            assert_eq!(body[0], TraceOp::Input(0), "first op must be Input(0)");
            // Must contain Tanh and Mul (core GELU structure).
            let has_tanh = body.iter().any(|op| matches!(op, TraceOp::Tanh(_)));
            let has_mul = body.iter().any(|op| matches!(op, TraceOp::Mul(_, _)));
            assert!(has_tanh, "GELU missing Tanh");
            assert!(has_mul, "GELU missing Mul");
        } else {
            panic!("GELU should be Elementwise");
        }
    }

    #[test]
    fn registry_rms_norm_trace_is_normlike() {
        let reg = ScalarOpRegistry::with_defaults();
        let trace = reg.get_trace(&OpKindKey::RmsNorm).unwrap();
        assert!(
            matches!(trace.pattern, ComputePattern::NormLike { .. }),
            "RmsNorm should be NormLike"
        );
    }

    #[test]
    fn registry_gemm_trace_is_gemm() {
        let reg = ScalarOpRegistry::with_defaults();
        let trace = reg.get_trace(&OpKindKey::Gemm).unwrap();
        assert!(
            matches!(trace.pattern, ComputePattern::Gemm),
            "Gemm should be Gemm pattern"
        );
    }

    #[test]
    fn registry_rope_trace_is_injective() {
        let reg = ScalarOpRegistry::with_defaults();
        let trace = reg.get_trace(&OpKindKey::RoPE).unwrap();
        if let ComputePattern::Injective { num_inputs, num_outputs, body } = &trace.pattern {
            assert_eq!(*num_inputs, 4);
            assert_eq!(*num_outputs, 2);
            assert_eq!(body.len(), 10);
        } else {
            panic!("RoPE should be Injective");
        }
    }

    #[test]
    fn registry_fn_ptrs_are_non_null() {
        let reg = ScalarOpRegistry::with_defaults();
        for key in [
            OpKindKey::Silu,
            OpKindKey::Gelu,
            OpKindKey::Add,
            OpKindKey::Mul,
            OpKindKey::RmsNorm,
            OpKindKey::Gemm,
            OpKindKey::RoPE,
        ] {
            let sig = reg.get_signature(&key).unwrap();
            assert!(!sig.fn_ptr.is_null(), "fn_ptr is null for {key:?}");
        }
    }

    /// Verify that `auto_register_from_symexec` either succeeds or gracefully
    /// falls back for every default operator. The symexec engine is a stub
    /// (returns empty trace), so we expect `EmptyTrace` errors — the important
    /// thing is no panics or UB.
    #[test]
    fn test_symexec_extracts_from_compiled_scalar_ops() {
        let reg = ScalarOpRegistry::with_defaults();

        let test_keys = [
            OpKindKey::Silu,
            OpKindKey::Gelu,
            OpKindKey::Add,
            OpKindKey::Mul,
            OpKindKey::RmsNorm,
            OpKindKey::Softmax,
        ];

        for key in &test_keys {
            let sig = reg.get_signature(key).expect("missing signature");

            // Try symexec path on a fresh registry
            let mut fresh = ScalarOpRegistry::new();
            let result = fresh.auto_register_from_symexec(
                key.clone(),
                sig.clone(),
            );

            // Either succeeds or returns a well-formed error (no panic)
            match result {
                Ok(_pattern) => {
                    assert!(fresh.get_trace(key).is_some());
                }
                Err(_e) => {
                    // Expected: symexec stub returns empty/error
                }
            }
        }
    }

    #[test]
    fn registry_inject_overwrites() {
        let mut reg = ScalarOpRegistry::new();
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![],
        };
        let trace1 = OpTrace {
            pattern: ComputePattern::Elementwise { body: vec![TraceOp::Input(0)] },
            signature: sig.clone(),
        };
        let trace2 = OpTrace {
            pattern: ComputePattern::Elementwise {
                body: vec![TraceOp::Input(0), TraceOp::Neg(ValueId(0))],
            },
            signature: sig,
        };

        reg.inject_trace(OpKindKey::Silu, trace1);
        assert_eq!(
            reg.get_trace(&OpKindKey::Silu)
                .map(|t| match &t.pattern {
                    ComputePattern::Elementwise { body } => body.len(),
                    _ => 0,
                }),
            Some(1)
        );

        reg.inject_trace(OpKindKey::Silu, trace2);
        assert_eq!(
            reg.get_trace(&OpKindKey::Silu)
                .map(|t| match &t.pattern {
                    ComputePattern::Elementwise { body } => body.len(),
                    _ => 0,
                }),
            Some(2)
        );
    }

    // ── OpKindKey enum variant equality / hash tests ──

    #[test]
    fn opkindkey_equality_same_variants() {
        assert_eq!(OpKindKey::Silu, OpKindKey::Silu);
        assert_eq!(OpKindKey::Gemm, OpKindKey::Gemm);
        assert_eq!(OpKindKey::RoPE, OpKindKey::RoPE);
        assert_eq!(OpKindKey::MlaAttention, OpKindKey::MlaAttention);
        assert_eq!(OpKindKey::MtpDraft, OpKindKey::MtpDraft);
    }

    #[test]
    fn opkindkey_inequality_different_variants() {
        assert_ne!(OpKindKey::Silu, OpKindKey::Gelu);
        assert_ne!(OpKindKey::Gemm, OpKindKey::GemmBias);
        assert_ne!(OpKindKey::RmsNorm, OpKindKey::LayerNorm);
        assert_ne!(OpKindKey::MoEGate, OpKindKey::MoERouter);
        assert_ne!(OpKindKey::MlaKvCompress, OpKindKey::MlaQAbsorb);
    }

    #[test]
    fn opkindkey_clone_preserves_equality() {
        let key = OpKindKey::QuantGather;
        assert_eq!(key.clone(), key);
        let key2 = OpKindKey::DequantizeMxfp4;
        assert_eq!(key2.clone(), key2);
    }

    #[test]
    fn opkindkey_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OpKindKey::Silu);
        set.insert(OpKindKey::Gelu);
        set.insert(OpKindKey::Silu); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&OpKindKey::Silu));
        assert!(set.contains(&OpKindKey::Gelu));
    }

    #[test]
    fn opkindkey_norm_variants_distinct() {
        assert_ne!(OpKindKey::RmsNorm, OpKindKey::LayerNorm);
        assert_ne!(OpKindKey::RmsNorm, OpKindKey::QkNorm);
        assert_ne!(OpKindKey::RmsNorm, OpKindKey::HeadRmsNorm);
        assert_ne!(OpKindKey::RmsNorm, OpKindKey::ValueNorm);
        assert_ne!(OpKindKey::LayerNorm, OpKindKey::ValueNorm);
        assert_ne!(OpKindKey::QkNorm, OpKindKey::HeadRmsNorm);
    }

    #[test]
    fn opkindkey_mla_variants_distinct() {
        assert_ne!(OpKindKey::MlaKvCompress, OpKindKey::MlaQAbsorb);
        assert_ne!(OpKindKey::MlaVRestore, OpKindKey::MlaAttention);
        assert_ne!(OpKindKey::MlaRopeMerge, OpKindKey::MlaKvCompress);
    }

    #[test]
    fn opkindkey_moe_variants_distinct() {
        assert_ne!(OpKindKey::MoEGate, OpKindKey::MoERouter);
        assert_ne!(OpKindKey::MoEGate, OpKindKey::MoEDispatchPacked);
        assert_ne!(OpKindKey::MoERouter, OpKindKey::MoEDispatchPacked);
    }

    #[test]
    fn opkindkey_sg_business_variants_distinct() {
        assert_ne!(OpKindKey::SgInject, OpKindKey::SgDetect);
        assert_ne!(OpKindKey::GuardrailCheck, OpKindKey::SgDetect);
        assert_ne!(OpKindKey::CotStepCheck, OpKindKey::EarlyExit);
        assert_ne!(OpKindKey::SessionKvRestore, OpKindKey::MmHiddenInject);
        assert_ne!(OpKindKey::StoreToken, OpKindKey::WriteLogits);
        assert_ne!(OpKindKey::CheckStopCondition, OpKindKey::EarlyExit);
    }

    #[test]
    fn opkindkey_p4_p5_variants_distinct() {
        assert_ne!(OpKindKey::EntropyGate, OpKindKey::VRangeQuant);
        assert_ne!(OpKindKey::GateMask, OpKindKey::AttentionSkipMask);
        assert_ne!(OpKindKey::LayerBypass, OpKindKey::MaskedGemm);
        assert_ne!(OpKindKey::FusedRmsNormGemm, OpKindKey::SoftmaxWithEntropy);
        assert_ne!(OpKindKey::KvCentroidPrefetch, OpKindKey::VariableLengthBatch);
    }

    // ── RegistryError Display / From tests ──

    #[test]
    fn registry_error_display_empty_trace() {
        let err = RegistryError::EmptyTrace;
        assert_eq!(format!("{err}"), "extracted trace is empty");
    }

    #[test]
    fn registry_error_display_symexec() {
        let inner = SymExecError::DisassemblyFailed("bad bytes".into());
        let err = RegistryError::SymExec(inner);
        let msg = format!("{err}");
        assert!(msg.starts_with("symexec error:"));
        assert!(msg.contains("bad bytes"));
    }

    #[test]
    fn registry_error_from_symexec_error() {
        let inner = SymExecError::UnsupportedInstruction("vmovapd".into());
        let err: RegistryError = inner.into();
        match err {
            RegistryError::SymExec(_) => {}
            RegistryError::EmptyTrace => panic!("should be SymExec variant"),
        }
    }

    #[test]
    fn registry_error_symexec_no_return_value() {
        let inner = SymExecError::NoReturnValue;
        let err = RegistryError::SymExec(inner);
        let msg = format!("{err}");
        assert!(msg.contains("no return value"));
    }
}
