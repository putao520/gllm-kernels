mod tests {
    use super::*;
    use crate::compiler::ir::LayerIR;
    use crate::types::ModelConfig;
    use crate::dispatch::DeviceProfile;

    #[test]
    fn test_empty_graph() {
        let g = CompilerGraph::new();
        assert_eq!(g.num_ops(), 0);
        assert_eq!(g.num_tensors(), 0);
        assert!(g.topological_sort().is_empty());
    }

    #[test]
    fn test_simple_chain() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);
        let c = g.add_tensor_concrete("c", &[1, 4096], dt);

        let op0 = g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        let op1 = g.add_op(OpKind::Silu, vec![b], vec![c], "silu2");

        let sorted = g.topological_sort();
        assert_eq!(sorted, vec![op0, op1]);
    }

    #[test]
    fn test_diamond_dag() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("in", &[1, 4096], dt);
        let left = g.add_tensor_concrete("left", &[1, 4096], dt);
        let right = g.add_tensor_concrete("right", &[1, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);

        let op_l = g.add_op(OpKind::Silu, vec![input], vec![left], "left");
        let op_r = g.add_op(OpKind::Gelu, vec![input], vec![right], "right");
        let op_add = g.add_op(OpKind::Add, vec![left, right], vec![out], "add");

        let sorted = g.topological_sort();
        // op_l and op_r must come before op_add
        let pos_l = sorted.iter().position(|&x| x == op_l).unwrap();
        let pos_r = sorted.iter().position(|&x| x == op_r).unwrap();
        let pos_add = sorted.iter().position(|&x| x == op_add).unwrap();
        assert!(pos_l < pos_add);
        assert!(pos_r < pos_add);
    }

    #[test]
    fn test_def_use_chains() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);
        let c = g.add_tensor_concrete("c", &[1, 4096], dt);

        let op0 = g.add_op(OpKind::Silu, vec![a], vec![b], "silu");
        let op1 = g.add_op(OpKind::Silu, vec![b], vec![c], "silu2");

        let chains = g.def_use_chains();
        // 'a' is a graph input (no producer), consumed by op0
        assert_eq!(chains[&a].0, None);
        assert_eq!(chains[&a].1, vec![op0]);
        // 'b' produced by op0, consumed by op1
        assert_eq!(chains[&b].0, Some(op0));
        assert_eq!(chains[&b].1, vec![op1]);
        // 'c' produced by op1, no consumers
        assert_eq!(chains[&c].0, Some(op1));
        assert!(chains[&c].1.is_empty());
    }

    #[test]
    fn test_from_layer_ir_decoder() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let g = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        eprintln!("{g}");

        // Should have a reasonable number of ops
        assert!(g.num_ops() >= 14, "expected ≥14 ops, got {}", g.num_ops());
        // Should have inputs and outputs
        assert!(!g.inputs.is_empty());
        assert!(!g.outputs.is_empty());
        // Topological sort should succeed (no cycles)
        let sorted = g.topological_sort();
        assert_eq!(sorted.len(), g.num_ops());
    }

    #[test]
    fn test_from_layer_ir_gemma_geglu() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let g = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        // Should contain a GeGlu op
        let has_geglu = g.ops.iter().any(|op| matches!(op.kind, OpKind::GeGlu));
        assert!(has_geglu, "Gemma graph should have GeGlu op");
    }

    #[test]
    fn test_from_layer_ir_encoder() {
        let mut config = ModelConfig::llama_7b();
        // Override to make it an encoder-style model
        config.arch = crate::types::ModelArch::Gpt2;
        let ir = LayerIR::from_model_config(&config, 4); // seq_len=4
        assert!(!ir.activation.is_gated()); // GELU = non-gated (encoder-style)

        let profile = DeviceProfile::detect();
        let g = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir encoder failed");

        eprintln!("{g}");

        // Encoder graph: LN1 + QKV(3) + MHA + O + Resid1 + LN2 + Up + GELU + Down + Resid2 + MeanPool + L2Norm = 14 ops
        assert!(g.num_ops() >= 14, "expected ≥14 ops, got {}", g.num_ops());
        assert!(!g.inputs.is_empty());
        assert!(!g.outputs.is_empty());

        // Should have LayerNorm (not RmsNorm)
        let has_ln = g.ops.iter().any(|op| matches!(op.kind, OpKind::LayerNorm { .. }));
        assert!(has_ln, "Encoder graph should have LayerNorm");

        // Should have MHA
        let has_mha = g.ops.iter().any(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. }));
        assert!(has_mha, "Encoder graph should have MultiHeadAttention");

        // Should have MeanPool and L2Normalize
        let has_pool = g.ops.iter().any(|op| matches!(op.kind, OpKind::MeanPool { .. }));
        let has_l2 = g.ops.iter().any(|op| matches!(op.kind, OpKind::L2Normalize { .. }));
        assert!(has_pool, "Encoder graph should have MeanPool");
        assert!(has_l2, "Encoder graph should have L2Normalize");

        // Should NOT have RoPE or RmsNorm
        let has_rope = g.ops.iter().any(|op| matches!(op.kind, OpKind::RoPE { .. }));
        let has_rms = g.ops.iter().any(|op| matches!(op.kind, OpKind::RmsNorm { .. }));
        assert!(!has_rope, "Encoder graph should NOT have RoPE");
        assert!(!has_rms, "Encoder graph should NOT have RmsNorm");

        // Topological sort should succeed
        let sorted = g.topological_sort();
        assert_eq!(sorted.len(), g.num_ops());
    }

    #[test]
    fn test_multi_output_config_single() {
        let config = MultiOutputConfig::single();
        assert_eq!(config.num_outputs, 1);
        assert!(config.output_tensors.is_empty());
        assert!(!config.is_multi_output());
    }

    #[test]
    fn test_multi_output_config_multi() {
        let tensors = vec![TensorId(0), TensorId(1), TensorId(2)];
        let config = MultiOutputConfig::multi(tensors.clone());
        assert_eq!(config.num_outputs, 3);
        assert_eq!(config.output_tensors, tensors);
        assert!(config.is_multi_output());
    }

    #[test]
    fn test_multi_output_config_default() {
        let config = MultiOutputConfig::default();
        assert_eq!(config.num_outputs, 0);
        assert!(!config.is_multi_output());
    }

    // ── SymDim ────────────────────────────────────────────────────────

    #[test]
    fn symdim_concrete_resolve() {
        let dim = SymDim::Concrete(42);
        let binding = ShapeBinding::new();
        assert_eq!(dim.resolve(&binding).unwrap(), 42);
    }

    #[test]
    fn symdim_symbolic_resolve() {
        let dim = SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(2048) };
        let binding = ShapeBinding::new().bind("seq_len", 128);
        assert_eq!(dim.resolve(&binding).unwrap(), 128);
    }

    #[test]
    fn symdim_symbolic_unresolved() {
        let dim = SymDim::Symbolic { name: "missing".to_string(), max_value: None };
        let binding = ShapeBinding::new();
        assert!(dim.resolve(&binding).is_err());
    }

    #[test]
    fn symdim_as_concrete() {
        assert_eq!(SymDim::Concrete(64).as_concrete(), Some(64));
        assert_eq!(SymDim::Symbolic { name: "x".to_string(), max_value: None }.as_concrete(), None);
    }

    #[test]
    fn symdim_max_for_allocation_concrete() {
        let dim = SymDim::Concrete(100);
        assert_eq!(dim.max_for_allocation(999), 100);
    }

    #[test]
    fn symdim_max_for_allocation_symbolic_with_max() {
        let dim = SymDim::Symbolic { name: "seq".to_string(), max_value: Some(512) };
        assert_eq!(dim.max_for_allocation(0), 512);
    }

    #[test]
    fn symdim_max_for_allocation_symbolic_without_max() {
        let dim = SymDim::Symbolic { name: "seq".to_string(), max_value: None };
        assert_eq!(dim.max_for_allocation(256), 256);
    }

    #[test]
    fn symdim_max_for_allocation_strict_concrete() {
        let dim = SymDim::Concrete(64);
        assert_eq!(dim.max_for_allocation_strict().unwrap(), 64);
    }

    #[test]
    fn symdim_max_for_allocation_strict_symbolic_with_max() {
        let dim = SymDim::Symbolic { name: "s".to_string(), max_value: Some(1024) };
        assert_eq!(dim.max_for_allocation_strict().unwrap(), 1024);
    }

    #[test]
    fn symdim_max_for_allocation_strict_symbolic_no_max() {
        let dim = SymDim::Symbolic { name: "s".to_string(), max_value: None };
        assert!(dim.max_for_allocation_strict().is_err());
    }

    #[test]
    fn symdim_is_symbolic() {
        assert!(!SymDim::Concrete(1).is_symbolic());
        assert!(SymDim::Symbolic { name: "x".to_string(), max_value: None }.is_symbolic());
    }

    #[test]
    fn symdim_from_usize() {
        let dim: SymDim = 42usize.into();
        assert_eq!(dim, SymDim::Concrete(42));
    }

    // ── ShapeBinding ──────────────────────────────────────────────────

    #[test]
    fn shape_binding_new_empty() {
        let b = ShapeBinding::new();
        assert!(b.get("x").is_none());
    }

    #[test]
    fn shape_binding_builder() {
        let b = ShapeBinding::new().bind("a", 1).bind("b", 2);
        assert_eq!(b.get("a"), Some(&1));
        assert_eq!(b.get("b"), Some(&2));
    }

    #[test]
    fn shape_binding_insert() {
        let mut b = ShapeBinding::new();
        b.insert("x", 10);
        assert_eq!(b.get("x"), Some(&10));
    }

    #[test]
    fn shape_binding_from_array() {
        let b: ShapeBinding = [("seq", 64), ("batch", 4)].into();
        assert_eq!(b.get("seq"), Some(&64));
        assert_eq!(b.get("batch"), Some(&4));
    }

    #[test]
    fn shape_binding_resolve_helper() {
        let b = ShapeBinding::new().bind("dim", 8);
        let dim = SymDim::Symbolic { name: "dim".to_string(), max_value: Some(16) };
        assert_eq!(b.resolve(&dim).unwrap(), 8);
    }

    // ── TensorId / OpId ───────────────────────────────────────────────

    #[test]
    fn tensor_id_equality() {
        assert_eq!(TensorId(0), TensorId(0));
        assert_ne!(TensorId(1), TensorId(2));
    }

    #[test]
    fn op_id_equality() {
        assert_eq!(OpId(0), OpId(0));
        assert_ne!(OpId(1), OpId(2));
    }

    // ── WeightLayout ──────────────────────────────────────────────────

    #[test]
    fn weight_layout_offset_of_found() {
        let layout = WeightLayout {
            offsets: vec![(TensorId(0), 0), (TensorId(1), 256), (TensorId(2), 512)],
            total_bytes: 768,
        };
        assert_eq!(layout.offset_of(TensorId(1)), Some(256));
    }

    #[test]
    fn weight_layout_offset_of_not_found() {
        let layout = WeightLayout { offsets: vec![], total_bytes: 0 };
        assert_eq!(layout.offset_of(TensorId(99)), None);
    }

    // ── Op v2 (胖 opcode 自描述架构) 测试 ──

    #[test]
    fn op_v2_category_returns_correct_category() {
        // Activation
        assert_eq!(Op::Silu.category(), "activation");
        assert_eq!(Op::Gelu.category(), "activation");
        assert_eq!(Op::SwiGluClipped { limit: 7.0 }.category(), "activation");

        // Norm
        let norm = Op::RmsNorm(NormSpec {
            feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true,
        });
        assert_eq!(norm.category(), "norm");

        // Gemm
        let gemm = Op::Gemm(GemmSpec {
            m: SymDim::Concrete(1), n: 4096, k: 4096,
            dtype: DType::F32, trans_b: false, has_bias: false,
        });
        assert_eq!(gemm.category(), "gemm");

        // Attention
        let attn = Op::MultiHeadAttention(AttentionSpec {
            geometry: AttentionGeometry { num_q_heads: 32, num_kv_heads: 8, head_dim: 128 },
            mask: AttentionMask::Causal,
            kv_source: KvSource::FromCache,
            sinks: SinksSpec::None,
            seq_len: SymDim::Concrete(1),
            dtype: DType::F32,
        });
        assert_eq!(attn.category(), "attention");
    }

    #[test]
    fn op_v2_attention_mask_causal_vs_full() {
        assert_ne!(AttentionMask::Causal, AttentionMask::Full);
        assert_eq!(AttentionMask::Sliding { window: 512 }, AttentionMask::Sliding { window: 512 });
    }

    #[test]
    fn op_v2_norm_spec_has_weight_distinct() {
        let with_weight = NormSpec {
            feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true,
        };
        let without_weight = NormSpec {
            feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: false,
        };
        assert_ne!(with_weight, without_weight);
    }

    #[test]
    fn op_v2_kv_source_serializable() {
        // KvSource 是 Copy + Hash，可用于 JIT cache key
        let sources = [KvSource::FromTensor, KvSource::FromCache];
        assert_ne!(sources[0], sources[1]);
    }

    #[test]
    fn op_v2_from_op_kind_norm_activation_silu() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[4], DType::F32);
        let output = g.add_tensor_concrete("output", &[4], DType::F32);
        let op = g.add_op(OpKind::Silu, vec![input], vec![output], "silu");

        let op_v2 = Op::from_op_kind_norm_activation(g.op(op).unwrap(), &g);
        assert_eq!(op_v2, Some(Op::Silu));
    }

    #[test]
    fn op_v2_from_op_kind_rmsnorm_dtype_propagation() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[4096], DType::BF16);
        let weight = g.add_tensor_concrete("weight", &[4096], DType::BF16);
        let output = g.add_tensor_concrete("output", &[4096], DType::BF16);
        let op = g.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input, weight], vec![output], "rms",
        );

        let op_v2 = Op::from_op_kind_norm_activation(g.op(op).unwrap(), &g);
        if let Some(Op::RmsNorm(spec)) = op_v2 {
            assert_eq!(spec.feature_dim, 4096);
            assert_eq!(spec.eps, 1e-5);
            assert_eq!(spec.dtype, DType::BF16); // dtype 从输入 tensor 推导
            assert!(spec.has_weight);
        } else {
            panic!("expected Op::RmsNorm");
        }
    }

    #[test]
    fn op_v2_from_op_kind_valuenorm_has_weight_false() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[4096], DType::F32);
        let output = g.add_tensor_concrete("output", &[4096], DType::F32);
        let op = g.add_op(
            OpKind::ValueNorm { feature_dim: 4096, eps: 1e-6 },
            vec![input], vec![output], "vn",
        );

        let op_v2 = Op::from_op_kind_norm_activation(g.op(op).unwrap(), &g);
        if let Some(Op::ValueNorm(spec)) = op_v2 {
            assert!(!spec.has_weight); // ValueNorm 无学习参数
        } else {
            panic!("expected Op::ValueNorm");
        }
    }

    #[test]
    fn op_v2_from_op_kind_gem_returns_none_for_non_norm_activation() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[4096], DType::F32);
        let weight = g.add_tensor_concrete("weight", &[4096, 4096], DType::F32);
        let output = g.add_tensor_concrete("output", &[4096], DType::F32);
        let op = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![input, weight], vec![output], "gemm",
        );

        // Gemm 不是 Norm/Activation，应返回 None（Phase 5 处理）
        let op_v2 = Op::from_op_kind_norm_activation(g.op(op).unwrap(), &g);
        assert_eq!(op_v2, None);
    }

    // ── Phase 5: Gemm/Quant from_op_kind 测试 ──

    #[test]
    fn op_v2_from_op_kind_gemm_basic() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[1, 4096], DType::F32);
        let weight = g.add_tensor_concrete("weight", &[4096, 4096], DType::F32);
        let output = g.add_tensor_concrete("output", &[1, 4096], DType::F32);
        let op = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![input, weight], vec![output], "gemm",
        );

        let op_v2 = Op::from_op_kind_gemm_quant(g.op(op).unwrap(), &g);
        if let Some(Op::Gemm(spec)) = op_v2 {
            assert_eq!(spec.n, 4096);
            assert_eq!(spec.k, 4096);
            assert!(!spec.has_bias);
        } else {
            panic!("expected Op::Gemm");
        }
    }

    #[test]
    fn op_v2_from_op_kind_gemmbias_has_bias_true() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[1, 4096], DType::F32);
        let weight = g.add_tensor_concrete("weight", &[4096, 4096], DType::F32);
        let output = g.add_tensor_concrete("output", &[1, 4096], DType::F32);
        let op = g.add_op(
            OpKind::GemmBias { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![input, weight], vec![output], "gemmbias",
        );

        let op_v2 = Op::from_op_kind_gemm_quant(g.op(op).unwrap(), &g);
        if let Some(Op::GemmBias(spec)) = op_v2 {
            assert!(spec.has_bias);
        } else {
            panic!("expected Op::GemmBias");
        }
    }

    #[test]
    fn op_v2_from_op_kind_quantgemm_no_trans_b() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[1, 4096], DType::F32);
        let weight = g.add_tensor_concrete("weight", &[4096, 4096], DType::F32);
        let output = g.add_tensor_concrete("output", &[1, 4096], DType::F32);
        let op = g.add_op(
            OpKind::QuantGemm { m: SymDim::Concrete(1), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 },
            vec![input, weight], vec![output], "qgemm",
        );

        let op_v2 = Op::from_op_kind_gemm_quant(g.op(op).unwrap(), &g);
        assert!(matches!(op_v2, Some(Op::QuantGemm(_))));
    }

    #[test]
    fn op_v2_from_op_kind_dequantize_fields() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[1024], DType::F32);
        let output = g.add_tensor_concrete("output", &[1024], DType::F32);
        let op = g.add_op(
            OpKind::Dequantize { num_elements: 1024, block_size: 32, bits: 4 },
            vec![input], vec![output], "deq",
        );

        let op_v2 = Op::from_op_kind_gemm_quant(g.op(op).unwrap(), &g);
        if let Some(Op::Dequantize { num_elements, block_size, bits }) = op_v2 {
            assert_eq!(num_elements, 1024);
            assert_eq!(block_size, 32);
            assert_eq!(bits, 4);
        } else {
            panic!("expected Op::Dequantize");
        }
    }

    // ── Phase 6: Attention/MoE from_op_kind 测试 ──

    #[test]
    fn op_v2_from_op_kind_mha_kv_source_propagation() {
        let mut g = CompilerGraph::new();
        let q = g.add_tensor_concrete("q", &[1, 4096], DType::F32);
        let k = g.add_tensor_concrete("k", &[1, 4096], DType::F32);
        let v = g.add_tensor_concrete("v", &[1, 4096], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4096], DType::F32);
        let op = g.add_op(
            OpKind::MultiHeadAttention {
                seq_len: SymDim::Concrete(1), num_heads: 32, num_kv_heads: 8, head_dim: 128,
                causal: true, attention_sinks: false, kv_source: KvSource::FromCache,
            },
            vec![q, k, v], vec![out], "mha",
        );

        let op_v2 = Op::from_op_kind_attention_moe(g.op(op).unwrap(), &g);
        if let Some(Op::MultiHeadAttention(spec)) = op_v2 {
            assert_eq!(spec.kv_source, KvSource::FromCache);
            assert!(matches!(spec.mask, AttentionMask::Causal));
            assert!(matches!(spec.sinks, SinksSpec::None));
        } else {
            panic!("expected Op::MultiHeadAttention");
        }
    }

    #[test]
    fn op_v2_from_op_kind_mla_attention() {
        let mut g = CompilerGraph::new();
        let q = g.add_tensor_concrete("q", &[1, 4096], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 4096], DType::F32);
        let op = g.add_op(
            OpKind::MlaAttention {
                seq_len: SymDim::Concrete(1), num_heads: 32, head_dim: 128,
                d_c: 512, d_rope: 64, causal: true, kv_source: KvSource::FromTensor,
            },
            vec![q], vec![out], "mla",
        );

        let op_v2 = Op::from_op_kind_attention_moe(g.op(op).unwrap(), &g);
        assert!(matches!(op_v2, Some(Op::MlaAttention(_))));
    }

    #[test]
    fn op_v2_from_op_kind_moe_gate() {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[1, 4096], DType::F32);
        let out = g.add_tensor_concrete("out", &[1, 8], DType::F32);
        let op = g.add_op(
            OpKind::MoEGate { seq_len: 1, num_experts: 8, hidden: 4096, top_k: 2 },
            vec![input], vec![out], "moegate",
        );

        let op_v2 = Op::from_op_kind_attention_moe(g.op(op).unwrap(), &g);
        if let Some(Op::MoEGate { num_experts, top_k, .. }) = op_v2 {
            assert_eq!(num_experts, 8);
            assert_eq!(top_k, 2);
        } else {
            panic!("expected Op::MoEGate");
        }
    }

    // ── Phase 7: 统一 from_op_kind 入口测试 ──

    #[test]
    fn op_v2_unified_from_op_kind_all_categories() {
        let mut g = CompilerGraph::new();

        // Norm
        let input = g.add_tensor_concrete("in", &[4096], DType::F32);
        let weight = g.add_tensor_concrete("w", &[4096], DType::F32);
        let out = g.add_tensor_concrete("out", &[4096], DType::F32);
        let norm_op = g.add_op(
            OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![input, weight], vec![out], "norm",
        );
        assert!(Op::from_op_kind(g.op(norm_op).unwrap(), &g).is_some());

        // Gemm
        let g_in = g.add_tensor_concrete("gin", &[1, 4096], DType::F32);
        let g_w = g.add_tensor_concrete("gw", &[4096, 4096], DType::F32);
        let g_out = g.add_tensor_concrete("gout", &[1, 4096], DType::F32);
        let gemm_op = g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![g_in, g_w], vec![g_out], "gemm",
        );
        assert!(Op::from_op_kind(g.op(gemm_op).unwrap(), &g).is_some());

        // Activation
        let a_in = g.add_tensor_concrete("ain", &[4096], DType::F32);
        let a_out = g.add_tensor_concrete("aout", &[4096], DType::F32);
        let act_op = g.add_op(OpKind::Silu, vec![a_in], vec![a_out], "silu");
        assert!(Op::from_op_kind(g.op(act_op).unwrap(), &g).is_some());

        // Structural
        let t_in = g.add_tensor_concrete("tin", &[4096], DType::F32);
        let t_out = g.add_tensor_concrete("tout", &[4096], DType::F32);
        let struct_op = g.add_op(
            OpKind::Transpose { perm: vec![1, 0] },
            vec![t_in], vec![t_out], "transpose",
        );
        assert!(Op::from_op_kind(g.op(struct_op).unwrap(), &g).is_some());
    }
}
