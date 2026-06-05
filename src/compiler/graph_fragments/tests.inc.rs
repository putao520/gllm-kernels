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
        assert_eq!(ir.arch, crate::compiler::ir::LayerArch::Encoder);

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
}
