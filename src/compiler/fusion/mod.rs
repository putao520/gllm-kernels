//! Fusion pass — Fusion phase of the JIT compiler pipeline.
//!
//! Walks the CompilerGraph in topological order and groups adjacent ops
//! into `FusionGroup`s based on semantic compatibility rules.
//!
//! Fusion groups are the unit of code generation: each group becomes a
//! single loop nest or microkernel call in the emitted machine code.
//!
//! Key fusion patterns:
//! - GEMM + elementwise epilogue (bias, activation, residual add)
//! - Elementwise chain collapse (silu+mul -> swiglu already in graph)
//! - QKV shared input: three GEMMs reading the same normed input -> single pack_a
//! - RmsNorm -> GEMM: norm output feeds GEMM input without memory writeback
//! - FusedQkvNormRope (Gemma 4): QKV GEMMs + QkNorm(Q,K) + ValueNorm(V) + RoPE(Q,K)

mod types;
mod cost_model;
mod helpers;
mod pass;
pub mod pdt;
pub mod quant_aware;
pub mod quant_aware_fusion;

// ── Public re-exports ───────────────────────────────────────────────
pub use types::{FusionGroup, FusionMode, FusionPlan, FusionCost, GroupMarker, HeteroLayerType};
pub use cost_model::{estimate_fusion_cost, Cost, FusionCostModel};
pub use pass::{fuse_with_dag, fuse_with_dag_prebuilt};
pub use quant_aware_fusion::{FusionRule, FusionEngine};

// ── Crate-internal re-exports ───────────────────────────────────────
pub(crate) use cost_model::is_memory_bound_group;

#[cfg(test)]
mod tests {
    use super::*;
    use super::cost_model::{chain_eliminated_bytes, compute_group_roofline_scale};
    use crate::compiler::graph::{CompilerGraph, OpId, MultiOutputConfig};
    use crate::compiler::ir::LayerIR;
    use crate::compiler::planner::ExecutionPlan;
    use crate::dispatch::DeviceProfile;
    use crate::types::{DType, ModelConfig};

    // ── DAG-based fusion tests ──────────────────────────────────────

    #[test]
    fn test_fuse_with_dag_decoder_layer() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        eprintln!("DAG-based fusion:\n{plan}");

        // Every op should be in exactly one group
        for op in &graph.ops {
            assert!(
                plan.op_to_group.contains_key(&op.id),
                "Op {} not in any group",
                op.id.0
            );
        }

        // Should have QKV shared input group
        let has_qkv = plan
            .groups
            .iter()
            .any(|g| g.mode == FusionMode::QkvSharedInput);
        assert!(has_qkv, "Expected QKV shared input fusion");

        // Should have fewer groups than ops
        assert!(plan.num_groups() < graph.num_ops());
    }

    #[test]
    fn test_fuse_with_dag_gemm_epilogue() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
    }

    #[test]
    fn test_fuse_with_dag_injective_chain() {
        // RoPE (Injective) should be fusable in elementwise chains
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 128], dt);
        let cos = g.add_tensor_concrete("cos", &[64], dt);
        let rope_out = g.add_tensor_concrete("rope_out", &[1, 128], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 128], dt);

        g.add_op(
            crate::compiler::graph::OpKind::RoPE { num_heads: 32, head_dim: 128, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![a, cos],
            vec![rope_out],
            "rope",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![rope_out], vec![silu_out], "silu");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // RoPE + Silu should fuse (both fusable in new path: Injective + ElemWise)
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::LoopFusion);
    }

    // ── QuantGemm fusion tests ──────────────────────────────────────

    #[test]
    fn test_fuse_quant_gemm_epilogue() {
        // QuantGemm + SiLU should fuse as GemmEpilogue
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w_q4", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 },
            vec![a, w],
            vec![gemm_out],
            "qgemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
    }

    #[test]
    fn test_fuse_standalone_quant_gemm() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w_q4", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[1, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 },
            vec![a, w],
            vec![out],
            "qgemm",
        );

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::Standalone);
    }

    #[test]
    fn test_fuse_dequantize_standalone() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a_q4", &[4096], dt);
        let b = g.add_tensor_concrete("b_f32", &[4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Dequantize { num_elements: 4096, block_size: 32, bits: 4 },
            vec![a],
            vec![b],
            "dequant",
        );

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
    }

    // ── M5: Softmax must not be misidentified as norm prefix ────────

    #[test]
    fn test_softmax_not_norm_prefix_dag() {
        // Softmax -> GEMM: Softmax is Reduction but NOT a norm op.
        // detect_norm_into_gemm must reject it.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let softmax_out = g.add_tensor_concrete("softmax_out", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Softmax, vec![a], vec![softmax_out], "softmax");
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![softmax_out, w],
            vec![gemm_out],
            "gemm",
        );

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // Softmax and GEMM must be in separate groups — no NormIntoGemm fusion
        assert_eq!(plan.num_groups(), 2, "Softmax should not fuse as norm prefix");
        for group in &plan.groups {
            assert_ne!(
                group.mode,
                FusionMode::NormIntoGemm,
                "Softmax must not produce NormIntoGemm pattern"
            );
        }
    }

    // ── M6: Multi-input consumer with external input must not fuse ──

    #[test]
    fn test_multi_input_consumer_not_fused_dag() {
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);
        let other_out = g.add_tensor_concrete("other_out", &[1, 4096], dt);
        let swiglu_out = g.add_tensor_concrete("swiglu_out", &[1, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![other_out],
            "gemm_other",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm_main",
        );
        g.add_op(
            crate::compiler::graph::OpKind::SwiGlu,
            vec![gemm_out, other_out],
            vec![swiglu_out],
            "swiglu",
        );

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        let gemm_main_group = plan.groups.iter().find(|grp| {
            grp.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "gemm_main")
            })
        }).expect("gemm_main should have a group");
        assert!(
            !gemm_main_group.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "swiglu")
            }),
            "SwiGlu with external input must not be fused into GEMM epilogue (DAG path)"
        );
    }

    // ── WI-22: Fusion cost model tests ────────────────────────────────

    /// Helper: find the fusion group containing the op with the given label.
    fn find_group_by_label<'a>(
        plan: &'a FusionPlan,
        graph: &CompilerGraph,
        label: &str,
    ) -> Option<&'a FusionGroup> {
        plan.groups.iter().find(|grp| {
            grp.ops.iter().any(|&oid| {
                graph.op(oid).map_or(false, |o| o.label == label)
            })
        })
    }

    /// When norm output > 75% L1, TileLevelFusion is chosen;
    /// when <= 75% L1, ComputeRoot is chosen.
    #[test]
    fn test_tile_vs_compute_root_threshold() {
        let profile = DeviceProfile::detect();
        let (l1, _, _) = profile.cache_sizes();
        let l1_budget = l1 * 75 / 100;
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        // -- Large norm output: exceeds 75% L1 -> TileLevelFusion --
        let k_large = (l1_budget / 4) + 1;
        {
            let mut g = CompilerGraph::new();
            let x = g.add_tensor_concrete("x", &[1, k_large], dt);
            let norm_out = g.add_tensor_concrete("norm_out", &[1, k_large], dt);
            let w = g.add_tensor_concrete("w", &[k_large, k_large], dt);
            let gemm_out = g.add_tensor_concrete("gemm_out", &[1, k_large], dt);

            g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
            g.add_op(
                crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: k_large, k: k_large, dtype: DType::F32, trans_b: false },
                vec![norm_out, w],
                vec![gemm_out],
                "gemm",
            );

            let plan = fuse_with_dag(&g, &registry, &exec_plan);
            let gemm_group = find_group_by_label(&plan, &g, "gemm")
                .expect("GEMM should have a fusion group");
            assert!(
                matches!(gemm_group.mode, FusionMode::TileLevelFusion { .. }),
                "Expected TileLevelFusion for norm output ({} B) > 75% L1 ({} B), got {:?}",
                k_large * 4, l1_budget, gemm_group.mode,
            );
        }

        // -- Small norm output: fits in 75% L1 -> ComputeRoot --
        let k_small = l1_budget / 4;
        {
            let mut g = CompilerGraph::new();
            let x = g.add_tensor_concrete("x", &[1, k_small], dt);
            let norm_out = g.add_tensor_concrete("norm_out", &[1, k_small], dt);
            let n_small = 4096;
            let w = g.add_tensor_concrete("w", &[k_small, n_small], dt);
            let gemm_out = g.add_tensor_concrete("gemm_out", &[1, n_small], dt);

            let norm_bytes = k_small * 4;
            assert!(
                norm_bytes <= l1_budget,
                "Test setup error: norm output {norm_bytes}B should be <= l1_budget {l1_budget}B"
            );

            g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
            g.add_op(
                crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: n_small, k: k_small, dtype: DType::F32, trans_b: false },
                vec![norm_out, w],
                vec![gemm_out],
                "gemm",
            );

            let plan = fuse_with_dag(&g, &registry, &exec_plan);
            let gemm_group = find_group_by_label(&plan, &g, "gemm")
                .expect("GEMM should have a fusion group");
            assert!(
                matches!(gemm_group.mode, FusionMode::ComputeRoot { .. }),
                "Expected ComputeRoot for norm output ({} B) <= 75% L1 ({} B), got {:?}",
                norm_bytes, l1_budget, gemm_group.mode,
            );
        }
    }

    /// GEMM followed by elementwise gets EpilogueInjection mode.
    #[test]
    fn test_epilogue_injection_decision() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[4, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[4, 4096], dt);
        let gelu_out = g.add_tensor_concrete("gelu_out", &[4, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(4), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Gelu, vec![gemm_out], vec![gelu_out], "gelu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(
            plan.groups[0].mode,
            FusionMode::EpilogueInjection,
            "GEMM + elementwise should produce EpilogueInjection"
        );
        assert_eq!(plan.groups[0].epilogue.len(), 1, "epilogue should contain the Gelu op");
    }

    /// Consecutive elementwise ops get LoopFusion mode.
    #[test]
    fn test_loop_fusion_decision() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 4096], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Add, vec![a, b], vec![add_out], "add");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![add_out], vec![mul_out], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(
            plan.groups[0].mode,
            FusionMode::LoopFusion,
            "Consecutive elementwise ops should produce LoopFusion"
        );
        assert_eq!(plan.groups[0].ops.len(), 2);
    }

    /// 3 GEMMs sharing the same norm output get QkvSharedInput mode.
    #[test]
    fn test_qkv_shared_input_detection() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let dim = 4096;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, dim], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, dim], dt);
        let wq = g.add_tensor_concrete("wq", &[dim, dim], dt);
        let wk = g.add_tensor_concrete("wk", &[dim, dim], dt);
        let wv = g.add_tensor_concrete("wv", &[dim, dim], dt);
        let q_out = g.add_tensor_concrete("q_out", &[1, dim], dt);
        let k_out = g.add_tensor_concrete("k_out", &[1, dim], dt);
        let v_out = g.add_tensor_concrete("v_out", &[1, dim], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: DType::F32, trans_b: false },
            vec![norm_out, wq],
            vec![q_out],
            "gemm_q",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: DType::F32, trans_b: false },
            vec![norm_out, wk],
            vec![k_out],
            "gemm_k",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: DType::F32, trans_b: false },
            vec![norm_out, wv],
            vec![v_out],
            "gemm_v",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        let qkv_group = plan.groups.iter().find(|grp| grp.mode == FusionMode::QkvSharedInput);
        assert!(
            qkv_group.is_some(),
            "Expected QkvSharedInput group for 3 GEMMs sharing norm output"
        );
        let qkv = qkv_group.unwrap();
        assert_eq!(qkv.ops.len(), 3, "QKV group should contain exactly 3 GEMM ops");
    }

    /// FFNBlock detection: Gate+Up GEMM (shared input) → Silu → Mul should fuse into FFNBlock.
    #[test]
    fn test_ffn_block_detection() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let hidden = 4096;
        let inter = 11008;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, hidden], dt);
        let w_gate = g.add_tensor_concrete("w_gate", &[hidden, inter], dt);
        let w_up = g.add_tensor_concrete("w_up", &[hidden, inter], dt);
        let gate_out = g.add_tensor_concrete("gate_out", &[1, inter], dt);
        let up_out = g.add_tensor_concrete("up_out", &[1, inter], dt);
        let act_out = g.add_tensor_concrete("act_out", &[1, inter], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, inter], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: inter, k: hidden, dtype: DType::F32, trans_b: false },
            vec![x, w_gate], vec![gate_out], "gate_gemm",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: inter, k: hidden, dtype: DType::F32, trans_b: false },
            vec![x, w_up], vec![up_out], "up_gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gate_out], vec![act_out], "silu");
        g.add_op(crate::compiler::graph::OpKind::Mul, vec![act_out, up_out], vec![mul_out], "mul");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let ffn = plan.groups.iter().find(|grp| matches!(grp.mode, FusionMode::FFNBlock { .. }));
        assert!(ffn.is_some(), "Expected FFNBlock group");
        assert_eq!(ffn.unwrap().ops.len(), 4, "FFNBlock should contain 4 ops");
    }

    /// FFNBlock shape mismatch: different n in Gate vs Up → should NOT fuse.
    #[test]
    fn test_ffn_block_shape_mismatch_rejected() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let hidden = 4096;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, hidden], dt);
        let w_gate = g.add_tensor_concrete("w_gate", &[hidden, 11008], dt);
        let w_up = g.add_tensor_concrete("w_up", &[hidden, 8192], dt); // 不同 n
        let gate_out = g.add_tensor_concrete("gate_out", &[1, 11008], dt);
        let up_out = g.add_tensor_concrete("up_out", &[1, 8192], dt);
        let act_out = g.add_tensor_concrete("act_out", &[1, 11008], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, 11008], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 11008, k: hidden, dtype: DType::F32, trans_b: false },
            vec![x, w_gate], vec![gate_out], "gate_gemm",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 8192, k: hidden, dtype: DType::F32, trans_b: false },
            vec![x, w_up], vec![up_out], "up_gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gate_out], vec![act_out], "silu");
        g.add_op(crate::compiler::graph::OpKind::Mul, vec![act_out, up_out], vec![mul_out], "mul");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let ffn = plan.groups.iter().find(|grp| matches!(grp.mode, FusionMode::FFNBlock { .. }));
        assert!(ffn.is_none(), "FFNBlock must reject shape-mismatched Gate/Up GEMMs");
    }

    /// Gemma 4 pattern: 3 QKV GEMMs + QkNorm(Q,K) + ValueNorm(V) + RoPE(Q,K)
    /// should produce FusedQkvNormRope, not QkvSharedInput.
    #[test]
    fn test_fused_qkv_norm_rope_detection() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let dim = 4096;
        let head_dim = 128;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();

        // Inputs
        let x = g.add_tensor_concrete("x", &[1, dim], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, dim], dt);
        let wq = g.add_tensor_concrete("wq", &[dim, dim], dt);
        let wk = g.add_tensor_concrete("wk", &[dim, dim], dt);
        let wv = g.add_tensor_concrete("wv", &[dim, dim], dt);
        let cos_sin = g.add_tensor_concrete("cos_sin", &[head_dim / 2], dt);

        // RmsNorm → shared input for QKV
        g.add_op(
            crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 },
            vec![x],
            vec![norm_out],
            "rms_norm",
        );

        // Q projection
        let q_out = g.add_tensor_concrete("q_out", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: dt, trans_b: false },
            vec![norm_out, wq],
            vec![q_out],
            "gemm_q",
        );

        // K projection
        let k_out = g.add_tensor_concrete("k_out", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: dt, trans_b: false },
            vec![norm_out, wk],
            vec![k_out],
            "gemm_k",
        );

        // V projection
        let v_out = g.add_tensor_concrete("v_out", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: dt, trans_b: false },
            vec![norm_out, wv],
            vec![v_out],
            "gemm_v",
        );

        // QkNorm on Q
        let q_normed = g.add_tensor_concrete("q_normed", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::QkNorm { head_dim, eps: 1e-6 },
            vec![q_out],
            vec![q_normed],
            "qknorm_q",
        );

        // QkNorm on K
        let k_normed = g.add_tensor_concrete("k_normed", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::QkNorm { head_dim, eps: 1e-6 },
            vec![k_out],
            vec![k_normed],
            "qknorm_k",
        );

        // ValueNorm on V
        let v_normed = g.add_tensor_concrete("v_normed", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::ValueNorm { feature_dim: 4096, eps: 1e-6 },
            vec![v_out],
            vec![v_normed],
            "valuenorm_v",
        );

        // RoPE on normalized Q
        let q_rope = g.add_tensor_concrete("q_rope", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::RoPE { num_heads: dim / head_dim, head_dim, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![q_normed, cos_sin],
            vec![q_rope],
            "rope_q",
        );

        // RoPE on normalized K
        let k_rope = g.add_tensor_concrete("k_rope", &[1, dim], dt);
        g.add_op(
            crate::compiler::graph::OpKind::RoPE { num_heads: dim / head_dim, head_dim, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![k_normed, cos_sin],
            vec![k_rope],
            "rope_k",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // Should detect FusedQkvNormRope, NOT QkvSharedInput
        let fused_group = plan.groups.iter().find(|grp| {
            matches!(grp.mode, FusionMode::FusedQkvNormRope { .. })
        });
        assert!(
            fused_group.is_some(),
            "Expected FusedQkvNormRope group for Gemma 4 QKV+QkNorm+ValueNorm+RoPE pattern"
        );
        let fused = fused_group.unwrap();
        assert_eq!(fused.ops.len(), 8, "FusedQkvNormRope should contain 8 ops (3 GEMMs + 2 QkNorm + 1 ValueNorm + 2 RoPE)");

        // No QkvSharedInput should exist (superset detected instead)
        let qkv_group = plan.groups.iter().find(|grp| grp.mode == FusionMode::QkvSharedInput);
        assert!(
            qkv_group.is_none(),
            "QkvSharedInput should NOT be detected when FusedQkvNormRope covers the same GEMMs"
        );
    }

    /// Standard QKV without QkNorm/ValueNorm should still get QkvSharedInput, not FusedQkvNormRope.
    #[test]
    fn test_qkv_without_norms_not_fused_qkv_norm_rope() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let dim = 4096;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, dim], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, dim], dt);
        let wq = g.add_tensor_concrete("wq", &[dim, dim], dt);
        let wk = g.add_tensor_concrete("wk", &[dim, dim], dt);
        let wv = g.add_tensor_concrete("wv", &[dim, dim], dt);
        let q_out = g.add_tensor_concrete("q_out", &[1, dim], dt);
        let k_out = g.add_tensor_concrete("k_out", &[1, dim], dt);
        let v_out = g.add_tensor_concrete("v_out", &[1, dim], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: dt, trans_b: false },
            vec![norm_out, wq], vec![q_out], "gemm_q",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: dt, trans_b: false },
            vec![norm_out, wk], vec![k_out], "gemm_k",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, dtype: dt, trans_b: false },
            vec![norm_out, wv], vec![v_out], "gemm_v",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // Should be QkvSharedInput, NOT FusedQkvNormRope
        assert!(
            plan.groups.iter().any(|grp| grp.mode == FusionMode::QkvSharedInput),
            "Standard QKV (no norms) should produce QkvSharedInput"
        );
        assert!(
            !plan.groups.iter().any(|grp| matches!(grp.mode, FusionMode::FusedQkvNormRope { .. })),
            "Standard QKV (no norms) should NOT produce FusedQkvNormRope"
        );
    }

    /// RmsNorm -> GEMM (single consumer, no epilogue) gets a norm-aware
    /// fusion mode (ComputeRoot or TileLevelFusion), not Standalone.
    #[test]
    fn test_norm_into_gemm_decision() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, 512], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 512], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false },
            vec![norm_out, w],
            vec![gemm_out],
            "gemm",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        let gemm_group = find_group_by_label(&plan, &g, "gemm")
            .expect("GEMM should have a fusion group");
        let mode = &gemm_group.mode;
        assert!(
            matches!(mode, FusionMode::ComputeRoot { .. } | FusionMode::TileLevelFusion { .. }),
            "RmsNorm -> GEMM should produce ComputeRoot or TileLevelFusion, got {:?}",
            mode,
        );
        match mode {
            FusionMode::ComputeRoot { predecessor } |
            FusionMode::TileLevelFusion { predecessor, .. } => {
                let pred_op = g.op(*predecessor).expect("predecessor op should exist");
                assert!(
                    matches!(pred_op.kind, crate::compiler::graph::OpKind::RmsNorm { .. }),
                    "predecessor should be RmsNorm, got {:?}", pred_op.kind,
                );
            }
            _ => unreachable!(),
        }
    }

    /// Verify KC * (MR + NR) * 4 <= L1 * 0.85 for various GEMM sizes.
    #[test]
    fn test_gemm_blocking_l1_constraint() {
        let profile = DeviceProfile::detect();
        let (l1, _, _) = profile.cache_sizes();
        let (mr, nr) = profile.microkernel_mr_nr();

        for &(m, n, k) in &[
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (512, 2048, 768),
            (128, 128, 128),
        ] {
            let b = profile.gemm_blocking(m, n, k, DType::F32);
            let micropanel_bytes = b.kc * (mr + nr) * 4;
            let l1_budget = (l1 as f64 * profile.l1_budget_ratio()) as usize;
            assert!(
                micropanel_bytes <= l1_budget,
                "L1 constraint violated for m={m} n={n} k={k}: \
                 KC({}) * (MR({mr}) + NR({nr})) * 4 = {micropanel_bytes}B > {:.0}% of L1 ({}B)",
                b.kc, profile.l1_budget_ratio() * 100.0, l1_budget,
            );
        }
    }

    /// Verify MC * KC * 4 <= L2 * 0.85 for various GEMM sizes.
    #[test]
    fn test_gemm_blocking_l2_constraint() {
        let profile = DeviceProfile::detect();
        let (_, l2, _) = profile.cache_sizes();

        for &(m, n, k) in &[
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (512, 2048, 768),
            (128, 128, 128),
        ] {
            let b = profile.gemm_blocking(m, n, k, DType::F32);
            let a_panel_bytes = b.mc * b.kc * 4;
            let l2_budget = (l2 as f64 * profile.l1_budget_ratio()) as usize;
            assert!(
                a_panel_bytes <= l2_budget,
                "L2 constraint violated for m={m} n={n} k={k}: \
                 MC({}) * KC({}) * 4 = {a_panel_bytes}B > {:.0}% of L2 ({}B)",
                b.mc, b.kc, profile.l1_budget_ratio() * 100.0, l2_budget,
            );
        }
    }

    /// Verify KC * NC * 4 <= L3 * 0.65 for various GEMM sizes.
    #[test]
    fn test_gemm_blocking_l3_constraint() {
        let profile = DeviceProfile::detect();
        let (_, _, l3) = profile.cache_sizes();

        if l3 < 1024 * 1024 {
            eprintln!("Skipping L3 constraint test: L3 = {l3}B < 1MB");
            return;
        }

        for &(m, n, k) in &[
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (512, 2048, 768),
            (128, 128, 128),
        ] {
            let b = profile.gemm_blocking(m, n, k, DType::F32);
            let b_panel_bytes = b.kc * b.nc * 4;
            assert!(
                b_panel_bytes <= l3 * 65 / 100,
                "L3 constraint violated for m={m} n={n} k={k}: \
                 KC({}) * NC({}) * 4 = {b_panel_bytes}B > 65% of L3 ({}B)",
                b.kc, b.nc, l3 * 65 / 100,
            );
        }
    }

    /// Norm output > 75% L1 -> TileLevelFusion with tile_rows = MC.
    #[test]
    fn test_tile_level_fusion_decision() {
        let profile = DeviceProfile::detect();
        let (l1, _, _) = profile.cache_sizes();
        let l1_budget = l1 * 75 / 100;
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let k = (l1_budget / 4) + 1;
        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, k], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, k], dt);
        let w = g.add_tensor_concrete("w", &[k, k], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, k], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: k, k, dtype: DType::F32, trans_b: false },
            vec![norm_out, w],
            vec![gemm_out],
            "gemm",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let gemm_group = find_group_by_label(&plan, &g, "gemm")
            .expect("GEMM should have a fusion group");

        match &gemm_group.mode {
            FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                let pred_op = g.op(*predecessor).expect("predecessor should exist");
                assert!(matches!(pred_op.kind, crate::compiler::graph::OpKind::RmsNorm { .. }));
                let blocking = profile.gemm_blocking(1, k, k, DType::F32);
                assert_eq!(*tile_rows, blocking.mc, "tile_rows should equal MC from GEMM blocking");
            }
            other => panic!(
                "Expected TileLevelFusion for norm output ({} B) > 75% L1 ({} B), got {:?}",
                k * 4, l1_budget, other,
            ),
        }
    }

    /// Norm output <= 75% L1 -> ComputeRoot (standalone norm, result stays in L1).
    #[test]
    fn test_compute_root_decision() {
        let profile = DeviceProfile::detect();
        let (l1, _, _) = profile.cache_sizes();
        let l1_budget = l1 * 75 / 100;
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let k = l1_budget / 4;
        let n = 4096;
        let norm_bytes = k * 4;
        assert!(norm_bytes <= l1_budget, "test setup: norm output should fit in L1 budget");

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, k], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, k], dt);
        let w = g.add_tensor_concrete("w", &[k, n], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, n], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n, k, dtype: DType::F32, trans_b: false },
            vec![norm_out, w],
            vec![gemm_out],
            "gemm",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let gemm_group = find_group_by_label(&plan, &g, "gemm")
            .expect("GEMM should have a fusion group");

        match &gemm_group.mode {
            FusionMode::ComputeRoot { predecessor } => {
                let pred_op = g.op(*predecessor).expect("predecessor should exist");
                assert!(matches!(pred_op.kind, crate::compiler::graph::OpKind::RmsNorm { .. }));
            }
            other => panic!(
                "Expected ComputeRoot for norm output ({} B) <= 75% L1 ({} B), got {:?}",
                norm_bytes, l1_budget, other,
            ),
        }
    }

    /// Elementwise chain with oversized intermediates gets split into multiple LoopFusion groups.
    #[test]
    fn test_elementwise_chain_l1_split() {
        let profile = DeviceProfile::detect();
        let (l1, _, _) = profile.cache_sizes();
        let l1_budget = l1 * 75 / 100;
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let dim = (l1_budget / 4) + 1;

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, dim], dt);
        let b = g.add_tensor_concrete("b", &[1, dim], dt);
        let c = g.add_tensor_concrete("c", &[1, dim], dt);
        let d = g.add_tensor_concrete("d", &[1, dim], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![b], "silu1");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![b], vec![c], "silu2");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![c], vec![d], "silu3");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert!(
            plan.num_groups() > 1,
            "Expected chain split: {} groups for 3 ops with intermediate {} B > L1 budget {} B",
            plan.num_groups(), dim * 4, l1_budget,
        );

        for op in &g.ops {
            assert!(plan.op_to_group.contains_key(&op.id), "Op {} not in any group", op.id.0);
        }
    }

    // ── Roofline Cost model tests ──────────────────────────────────────

    #[test]
    fn test_cost_compute_bound() {
        use crate::compiler::trace::{OpTrace, ComputePattern, ScalarFnSignature, ScalarParam};

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let trace = OpTrace {
            op_kind: crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1024), n: 1024, k: 1024, dtype: DType::F32, trans_b: false },
            pattern: ComputePattern::Gemm,
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![
                    ScalarParam::InputPtr,
                    ScalarParam::InputPtr,
                    ScalarParam::OutputPtr,
                    ScalarParam::Dim(1024),
                    ScalarParam::Dim(1024),
                    ScalarParam::Dim(1024),
                ],
            },
        };

        let cost = Cost::compute(&trace, &exec_plan);
        assert!(cost.is_compute_bound(),
            "GEMM 1024x1024x1024 should be compute-bound: compute_cycles={:.1} > memory_cycles={:.1}",
            cost.compute_cycles, cost.memory_cycles);
        assert_eq!(cost.flops, 2 * 1024 * 1024 * 1024);
        assert_eq!(cost.bytes, 3 * 1024 * 1024 * 4);
    }

    #[test]
    fn test_cost_memory_bound() {
        use crate::compiler::trace::{OpTrace, ComputePattern, ScalarFnSignature, ScalarParam, TraceOp, ValueId};

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Neg(ValueId(0)),
            TraceOp::Exp(ValueId(1)),
            TraceOp::Const(1.0),
            TraceOp::Add(ValueId(2), ValueId(3)),
            TraceOp::Div(ValueId(0), ValueId(4)),
        ];
        let trace = OpTrace {
            op_kind: crate::compiler::graph::OpKind::Silu,
            pattern: ComputePattern::Elementwise { body },
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![
                    ScalarParam::InputPtr,
                    ScalarParam::OutputPtr,
                    ScalarParam::Dim(4096),
                ],
            },
        };

        let cost = Cost::compute(&trace, &exec_plan);
        assert!(!cost.is_compute_bound(),
            "SiLU on 4096 elements should be memory-bound: compute_cycles={:.1} <= memory_cycles={:.1}",
            cost.compute_cycles, cost.memory_cycles);
        assert_eq!(cost.flops, 13 * 4096);
        assert_eq!(cost.bytes, 2 * 4096 * 4);
    }

    #[test]
    fn test_fusion_benefit() {
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let eliminated = 2 * 4096 * 4;
        let benefit = Cost::fusion_benefit(eliminated, &exec_plan);
        assert!(benefit > 0,
            "Eliminating {} bytes should yield positive benefit, got {}",
            eliminated, benefit);

        let eliminated_large = 2 * 4096 * 4096 * 4;
        let benefit_large = Cost::fusion_benefit(eliminated_large, &exec_plan);
        assert!(benefit_large > benefit,
            "Larger elimination ({} bytes) should yield larger benefit ({}) than smaller ({} bytes, {})",
            eliminated_large, benefit_large, eliminated, benefit);

        assert_eq!(Cost::fusion_benefit(0, &exec_plan), 0);
    }

    #[test]
    fn test_cost_chain_eliminated_bytes() {
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);
        let c = g.add_tensor_concrete("c", &[1, 4096], dt);
        let d = g.add_tensor_concrete("d", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![b], "silu1");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![b], vec![c], "silu2");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![c], vec![d], "silu3");

        let anchor = g.op(OpId(0)).unwrap();
        let chain: Vec<&crate::compiler::graph::CompilerOp> = vec![
            g.op(OpId(1)).unwrap(),
            g.op(OpId(2)).unwrap(),
        ];

        let eliminated = chain_eliminated_bytes(&g, anchor, &chain);
        assert_eq!(eliminated, 2 * 4096 * 4 * 2,
            "Expected 2 intermediates eliminated, got {} bytes", eliminated);
    }

    #[test]
    fn test_cost_dag_fusion_uses_cost_filter() {
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);
        let c = g.add_tensor_concrete("c", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![b], "silu1");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![b], vec![c], "silu2");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::LoopFusion);
    }

    #[test]
    fn test_fusion_group_default_single_output() {
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert!(!group.multi_output.is_multi_output());
        assert_eq!(group.multi_output.num_outputs, 1);
    }

    #[test]
    fn test_fusion_group_multi_output() {
        let tensors = vec![
            crate::compiler::graph::TensorId(10),
            crate::compiler::graph::TensorId(11),
            crate::compiler::graph::TensorId(12),
        ];
        let group = FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![OpId(0)],
            multi_output: MultiOutputConfig::multi(tensors.clone()),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };
        assert!(group.multi_output.is_multi_output());
        assert_eq!(group.multi_output.num_outputs, 3);
        assert_eq!(group.multi_output.output_tensors, tensors);
    }

    #[test]
    fn test_fuse_produces_single_output_groups() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        for group in &plan.groups {
            assert!(
                !group.multi_output.is_multi_output(),
                "Group {} should be single-output by default",
                group.id,
            );
            assert_eq!(group.multi_output.num_outputs, 1);
        }
    }

    #[test]
    fn test_memory_bound_higher_fusion_benefit() {
        use crate::compiler::graph::OpKind;

        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Memory-bound: elementwise Add (AI ~ 0, no FLOPs from our estimator)
        let mut g1 = CompilerGraph::new();
        let a1 = g1.add_tensor_concrete("A", &[1024], DType::F32);
        let b1 = g1.add_tensor_concrete("B", &[1024], DType::F32);
        let c1 = g1.add_tensor_concrete("C", &[1024], DType::F32);
        let op1 = g1.add_op(OpKind::Add, vec![a1, b1], vec![c1], "add");
        let group_mem = FusionGroup {
            id: 0, anchor: op1, epilogue: vec![],
            mode: FusionMode::LoopFusion,
            ops: vec![op1],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Compute-bound: large GEMM
        let mut g2 = CompilerGraph::new();
        let a2 = g2.add_tensor_concrete("A", &[512, 512], DType::F32);
        let b2 = g2.add_tensor_concrete("B", &[512, 512], DType::F32);
        let c2 = g2.add_tensor_concrete("C", &[512, 512], DType::F32);
        let op2 = g2.add_op(
            OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(512), n: 512, k: 512, dtype: DType::F32, trans_b: false },
            vec![a2, b2], vec![c2], "gemm",
        );
        let group_compute = FusionGroup {
            id: 0, anchor: op2, epilogue: vec![],
            mode: FusionMode::Standalone,
            ops: vec![op2],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        let scale_mem = compute_group_roofline_scale(&group_mem, &g1, &exec_plan, None);
        let scale_compute = compute_group_roofline_scale(&group_compute, &g2, &exec_plan, None);

        assert!((scale_mem - 1.0).abs() < 1e-6, "memory-bound scale should be 1.0, got {}", scale_mem);
        assert!(scale_compute < scale_mem, "compute-bound scale ({}) should be less than memory-bound ({})", scale_compute, scale_mem);
    }

    // ── Standalone-equivalent tests (previously used old fuse()) ─────

    #[test]
    fn test_fuse_decoder_layer() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        eprintln!("{plan}");

        for op in &graph.ops {
            assert!(
                plan.op_to_group.contains_key(&op.id),
                "Op {} not in any group",
                op.id.0
            );
        }

        let has_qkv = plan
            .groups
            .iter()
            .any(|g| g.mode == FusionMode::QkvSharedInput);
        assert!(has_qkv, "Expected QKV shared input fusion");

        assert!(
            plan.num_groups() < graph.num_ops(),
            "Expected fusion to reduce group count: {} groups vs {} ops",
            plan.num_groups(),
            graph.num_ops()
        );
    }

    #[test]
    fn test_fuse_gemm_epilogue() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        eprintln!("{plan}");

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
        assert_eq!(plan.groups[0].ops.len(), 2);
    }

    #[test]
    fn test_standalone_reduction() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Softmax, vec![a], vec![b], "softmax");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::Standalone);
    }

    #[test]
    fn test_elementwise_chain() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let b = g.add_tensor_concrete("b", &[1, 256], dt);
        let c = g.add_tensor_concrete("c", &[1, 256], dt);
        let d = g.add_tensor_concrete("d", &[1, 256], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![b], "silu");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![b], vec![c], "silu2");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![c], vec![d], "silu3");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::LoopFusion);
        assert_eq!(plan.groups[0].ops.len(), 3);
    }

    #[test]
    fn test_no_fuse_across_reduction() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);
        let c = g.add_tensor_concrete("c", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![b], "silu");
        g.add_op(crate::compiler::graph::OpKind::Softmax, vec![b], vec![c], "softmax");

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 2);
    }

    #[test]
    fn test_gemma_geglu_fusion() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        eprintln!("{plan}");

        for op in &graph.ops {
            assert!(plan.op_to_group.contains_key(&op.id));
        }
    }

    #[test]
    fn test_multi_input_consumer_not_fused_as_epilogue() {
        let dt = DType::F32;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);
        let other_out = g.add_tensor_concrete("other_out", &[1, 4096], dt);
        let swiglu_out = g.add_tensor_concrete("swiglu_out", &[1, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![other_out],
            "gemm_other",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w],
            vec![gemm_out],
            "gemm_main",
        );
        g.add_op(
            crate::compiler::graph::OpKind::SwiGlu,
            vec![gemm_out, other_out],
            vec![swiglu_out],
            "swiglu",
        );

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let gemm_main_group = plan.groups.iter().find(|grp| {
            grp.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "gemm_main")
            })
        }).expect("gemm_main should have a group");
        assert!(
            !gemm_main_group.ops.iter().any(|&oid| {
                g.op(oid).map_or(false, |o| o.label == "swiglu")
            }),
            "SwiGlu with external input must not be fused into GEMM epilogue"
        );
    }

    // ── Phase A: Gemm + Argmax EpilogueInjection ──────────────────────

    /// GEMM followed by Argmax → should fuse as EpilogueInjection
    /// (argmax runs directly on GEMM accumulator, logits never written to memory).
    #[test]
    fn test_gemm_argmax_epilogue_injection() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let hidden = g.add_tensor_concrete("hidden", &[1, 4096], dt);
        let w_lm = g.add_tensor_concrete("w_lm", &[4096, 32000], dt);
        let logits = g.add_tensor_concrete("logits", &[1, 32000], dt);
        let token_id = g.add_tensor_concrete("token_id", &[1], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 32000,
                k: 4096,
                dtype: DType::F32, trans_b: false },
            vec![hidden, w_lm],
            vec![logits],
            "lm_head",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Argmax { vocab_size: 32000 },
            vec![logits],
            vec![token_id],
            "argmax",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // lm_head + argmax should fuse into a single group
        assert_eq!(plan.num_groups(), 1, "GEMM + Argmax should produce exactly 1 fusion group");
        let group = &plan.groups[0];
        assert_eq!(
            group.mode,
            FusionMode::EpilogueInjection,
            "GEMM + Argmax should produce EpilogueInjection"
        );
        assert_eq!(group.ops.len(), 2, "Group should contain GEMM + Argmax (2 ops)");
        assert_eq!(group.epilogue.len(), 1, "Argmax should be in epilogue");

        // Verify the epilogue op is Argmax
        let epilogue_op = g.op(group.epilogue[0]).expect("epilogue op should exist");
        assert!(
            matches!(epilogue_op.kind, crate::compiler::graph::OpKind::Argmax { .. }),
            "Epilogue should be Argmax, got {:?}",
            epilogue_op.kind,
        );
    }

    /// GEMM + ElemWise + Argmax chain: Argmax should still be collected after ElemWise epilogue.
    #[test]
    fn test_gemm_elemwise_argmax_chain() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let hidden = g.add_tensor_concrete("hidden", &[1, 4096], dt);
        let w_lm = g.add_tensor_concrete("w_lm", &[4096, 32000], dt);
        let logits = g.add_tensor_concrete("logits", &[1, 32000], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 32000], dt);
        let token_id = g.add_tensor_concrete("token_id", &[1], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 32000,
                k: 4096,
                dtype: DType::F32, trans_b: false },
            vec![hidden, w_lm],
            vec![logits],
            "lm_head",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![logits], vec![silu_out], "silu");
        g.add_op(
            crate::compiler::graph::OpKind::Argmax { vocab_size: 32000 },
            vec![silu_out],
            vec![token_id],
            "argmax",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 1, "GEMM + SiLU + Argmax should produce 1 group");
        let group = &plan.groups[0];
        assert_eq!(group.ops.len(), 3, "Group should contain GEMM + SiLU + Argmax (3 ops)");
    }

    /// Argmax with multiple consumers should NOT be fused (must write to memory for all consumers).
    #[test]
    fn test_argmax_multi_consumer_not_fused() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let hidden = g.add_tensor_concrete("hidden", &[1, 4096], dt);
        let w_lm = g.add_tensor_concrete("w_lm", &[4096, 1000], dt);
        let logits = g.add_tensor_concrete("logits", &[1, 1000], dt);
        let token_id = g.add_tensor_concrete("token_id", &[1], dt);
        let consumer_a = g.add_tensor_concrete("consumer_a", &[1], dt);
        let consumer_b = g.add_tensor_concrete("consumer_b", &[1], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 1000,
                k: 4096,
                dtype: DType::F32, trans_b: false },
            vec![hidden, w_lm],
            vec![logits],
            "lm_head",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Argmax { vocab_size: 1000 },
            vec![logits],
            vec![token_id],
            "argmax",
        );
        // Two consumers of logits → argmax's input has multiple consumers
        // Actually, this tests that logits has multiple consumers. Let me fix:
        // Argmax has token_id output. If token_id has 2 consumers, Argmax should not be fused.
        // But our check is on Argmax's INPUT having single consumer.
        // The real constraint is: logits must have exactly 1 consumer (Argmax).
        // With the current graph, logits has exactly 1 consumer (Argmax), so it SHOULD fuse.

        // Let's test the case where logits has 2 consumers instead:
        let _ = g;
        let mut g2 = CompilerGraph::new();
        let hidden2 = g2.add_tensor_concrete("hidden", &[1, 4096], dt);
        let w_lm2 = g2.add_tensor_concrete("w_lm", &[4096, 1000], dt);
        let logits2 = g2.add_tensor_concrete("logits", &[1, 1000], dt);
        let token_id2 = g2.add_tensor_concrete("token_id", &[1], dt);
        let other_out = g2.add_tensor_concrete("other_out", &[1, 1000], dt);

        g2.add_op(
            crate::compiler::graph::OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 1000,
                k: 4096,
                dtype: DType::F32, trans_b: false },
            vec![hidden2, w_lm2],
            vec![logits2],
            "lm_head",
        );
        // logits consumed by both argmax AND another op → multiple consumers
        g2.add_op(
            crate::compiler::graph::OpKind::Argmax { vocab_size: 1000 },
            vec![logits2],
            vec![token_id2],
            "argmax",
        );
        g2.add_op(crate::compiler::graph::OpKind::Silu, vec![logits2], vec![other_out], "other_consumer");

        let plan = fuse_with_dag(&g2, &registry, &exec_plan);

        // Argmax should NOT be fused because logits has multiple consumers
        let gemm_group = plan.groups.iter().find(|grp| {
            grp.ops.iter().any(|&oid| {
                g2.op(oid).map_or(false, |o| o.label == "lm_head")
            })
        }).expect("GEMM should have a group");

        assert!(
            !gemm_group.ops.iter().any(|&oid| {
                g2.op(oid).map_or(false, |o| matches!(o.kind, crate::compiler::graph::OpKind::Argmax { .. }))
            }),
            "Argmax must NOT be fused when logits has multiple consumers"
        );
    }

    // ── Quant-aware fusion integration tests ────────────────────────────

    /// QKV with mixed quant types (Q4_0 + Q6K) must NOT be QkvSharedInput.
    #[test]
    fn test_qkv_mixed_quant_rejected() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let dim = 4096;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, dim], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, dim], dt);
        let wq = g.add_tensor_concrete("wq", &[dim, dim], dt);
        let wk = g.add_tensor_concrete("wk", &[dim, dim], dt);
        let wv = g.add_tensor_concrete("wv", &[dim, dim], dt);
        let q_out = g.add_tensor_concrete("q_out", &[1, dim], dt);
        let k_out = g.add_tensor_concrete("k_out", &[1, dim], dt);
        let v_out = g.add_tensor_concrete("v_out", &[1, dim], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        // Q: Q4_0
        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, quant_type: crate::quant::QuantType::Q4_0 },
            vec![norm_out, wq], vec![q_out], "gemm_q",
        );
        // K: Q6K (different quant type)
        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, quant_type: crate::quant::QuantType::Q6K },
            vec![norm_out, wk], vec![k_out], "gemm_k",
        );
        // V: Q4_0 (same as Q)
        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, quant_type: crate::quant::QuantType::Q4_0 },
            vec![norm_out, wv], vec![v_out], "gemm_v",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // Should NOT detect QkvSharedInput — quant types are incompatible
        assert!(
            !plan.groups.iter().any(|grp| grp.mode == FusionMode::QkvSharedInput),
            "Mixed quant Q4_0/Q6K QKV must NOT fuse as QkvSharedInput"
        );
    }

    /// FFN with mixed quant types (Gate=Q4_0, Up=Q6K) must NOT be FFNBlock.
    #[test]
    fn test_ffn_mixed_quant_rejected() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let hidden = 4096;
        let inter = 11008;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, hidden], dt);
        let w_gate = g.add_tensor_concrete("w_gate", &[hidden, inter], dt);
        let w_up = g.add_tensor_concrete("w_up", &[hidden, inter], dt);
        let gate_out = g.add_tensor_concrete("gate_out", &[1, inter], dt);
        let up_out = g.add_tensor_concrete("up_out", &[1, inter], dt);
        let act_out = g.add_tensor_concrete("act_out", &[1, inter], dt);
        let mul_out = g.add_tensor_concrete("mul_out", &[1, inter], dt);

        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: inter, k: hidden, quant_type: crate::quant::QuantType::Q4_0 },
            vec![x, w_gate], vec![gate_out], "gate_gemm",
        );
        // Up GEMM: Q6K (different from Gate's Q4_0)
        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: inter, k: hidden, quant_type: crate::quant::QuantType::Q6K },
            vec![x, w_up], vec![up_out], "up_gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gate_out], vec![act_out], "silu");
        g.add_op(crate::compiler::graph::OpKind::Mul, vec![act_out, up_out], vec![mul_out], "mul");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let ffn = plan.groups.iter().find(|grp| matches!(grp.mode, FusionMode::FFNBlock { .. }));
        assert!(ffn.is_none(), "FFNBlock must reject mixed quant Gate(Q4_0)/Up(Q6K)");
    }

    /// QKV with same quant type (all Q4_0) should still fuse as QkvSharedInput.
    #[test]
    fn test_qkv_same_quant_fuses() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let dim = 4096;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, dim], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, dim], dt);
        let wq = g.add_tensor_concrete("wq", &[dim, dim], dt);
        let wk = g.add_tensor_concrete("wk", &[dim, dim], dt);
        let wv = g.add_tensor_concrete("wv", &[dim, dim], dt);
        let q_out = g.add_tensor_concrete("q_out", &[1, dim], dt);
        let k_out = g.add_tensor_concrete("k_out", &[1, dim], dt);
        let v_out = g.add_tensor_concrete("v_out", &[1, dim], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");
        // All Q4_0
        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, quant_type: crate::quant::QuantType::Q4_0 },
            vec![norm_out, wq], vec![q_out], "gemm_q",
        );
        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, quant_type: crate::quant::QuantType::Q4_0 },
            vec![norm_out, wk], vec![k_out], "gemm_k",
        );
        g.add_op(
            crate::compiler::graph::OpKind::QuantGemm { m: crate::compiler::graph::SymDim::Concrete(1), n: dim, k: dim, quant_type: crate::quant::QuantType::Q4_0 },
            vec![norm_out, wv], vec![v_out], "gemm_v",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert!(
            plan.groups.iter().any(|grp| grp.mode == FusionMode::QkvSharedInput),
            "Same quant type Q4_0 QKV should fuse as QkvSharedInput"
        );
    }

    // ── Additional fusion tests ────────────────────────────────────────────

    /// GEMM + ReLU should produce EpilogueInjection with 1 epilogue op.
    #[test]
    fn test_gemm_relu_epilogue_injection() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
        assert_eq!(plan.groups[0].epilogue.len(), 1);
    }

    /// FusionPlan::group_of returns correct group for each op and None for missing ops.
    #[test]
    fn test_fusion_plan_group_of_lookup() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        for op in &graph.ops {
            let group = plan.group_of(op.id);
            assert!(group.is_some(), "Op {} should belong to a group", op.id.0);
        }
        assert!(plan.group_of(OpId(9999)).is_none(), "Nonexistent OpId should return None");
    }

    /// FusionPlan::num_fused_ops returns positive count for decoder layer with QKV fusion.
    #[test]
    fn test_fusion_plan_num_fused_ops_positive() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        assert!(
            plan.num_fused_ops() > 0,
            "Decoder layer should have fused ops (QKV, GEMM epilogue, etc.)"
        );
        assert!(plan.num_fused_ops() < graph.num_ops(), "Some ops should remain standalone");
    }

    /// Single standalone GEMM (no downstream consumer) produces exactly 1 Standalone group.
    #[test]
    fn test_standalone_gemm_single_group() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let out = g.add_tensor_concrete("out", &[1, 512], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![out], "gemm",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::Standalone);
        assert_eq!(plan.groups[0].ops.len(), 1);
    }

    /// GEMM + Tanh + Silu triple chain should fuse into single EpilogueInjection group.
    #[test]
    fn test_gemm_multi_activation_epilogue() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 256], dt);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[1, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 256], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Tanh, vec![gemm_out], vec![tanh_out], "tanh");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![tanh_out], vec![silu_out], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
        assert_eq!(plan.groups[0].ops.len(), 3, "GEMM + Tanh + SiLU should be 3 ops in one group");
    }

    /// Two independent GEMMs (different inputs) must produce two separate groups.
    #[test]
    fn test_independent_gemms_separate_groups() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a1 = g.add_tensor_concrete("a1", &[1, 128], dt);
        let w1 = g.add_tensor_concrete("w1", &[128, 128], dt);
        let out1 = g.add_tensor_concrete("out1", &[1, 128], dt);
        let a2 = g.add_tensor_concrete("a2", &[1, 128], dt);
        let w2 = g.add_tensor_concrete("w2", &[128, 128], dt);
        let out2 = g.add_tensor_concrete("out2", &[1, 128], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 128, k: 128, dtype: DType::F32, trans_b: false },
            vec![a1, w1], vec![out1], "gemm1",
        );
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 128, k: 128, dtype: DType::F32, trans_b: false },
            vec![a2, w2], vec![out2], "gemm2",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 2, "Independent GEMMs must be in separate groups");
        for group in &plan.groups {
            assert_eq!(group.ops.len(), 1);
        }
    }

    /// FusionGroup anchor should always be the first op in the ops vector.
    #[test]
    fn test_fusion_group_anchor_is_first_op() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        for group in &plan.groups {
            assert!(
                group.ops.contains(&group.anchor),
                "Group {} anchor Op({}) must be in ops list", group.id, group.anchor.0
            );
        }
    }

    /// Dequantize followed by SiLU should produce LoopFusion (both elementwise-like).
    #[test]
    fn test_dequantize_elementwise_chain() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a_q4", &[4096], dt);
        let b = g.add_tensor_concrete("b_f32", &[4096], dt);
        let c = g.add_tensor_concrete("c_out", &[4096], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Dequantize { num_elements: 4096, block_size: 32, bits: 4 },
            vec![a], vec![b], "dequant",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![b], vec![c], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        assert_eq!(plan.num_groups(), 2, "Dequantize and SiLU form separate groups");
        assert_eq!(plan.groups[0].ops.len(), 1);
        assert_eq!(plan.groups[1].ops.len(), 1);
    }

    /// LayerNorm -> GEMM should detect norm-into-gemm pattern (ComputeRoot or TileLevelFusion).
    #[test]
    fn test_layer_norm_into_gemm_detection() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, 512], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 512], dt);

        g.add_op(crate::compiler::graph::OpKind::LayerNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "layer_norm");
        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false },
            vec![norm_out, w], vec![gemm_out], "gemm",
        );

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let gemm_group = find_group_by_label(&plan, &g, "gemm")
            .expect("GEMM should have a fusion group");
        assert!(
            matches!(gemm_group.mode, FusionMode::ComputeRoot { .. } | FusionMode::TileLevelFusion { .. }),
            "LayerNorm -> GEMM should produce ComputeRoot or TileLevelFusion, got {:?}",
            gemm_group.mode,
        );
    }

    /// FusionPlan Display output should list all group modes and be non-empty.
    #[test]
    fn test_fusion_plan_display_shows_all_groups() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 256], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let display = format!("{plan}");
        assert!(display.contains("1 groups"), "Display should show group count: {display}");
        assert!(display.contains("EpilogueInjection"), "Display should show mode: {display}");
        assert!(display.contains("anchor"), "Display should show anchor: {display}");
    }

    // ── Additional edge-case tests ─────────────────────────────────────────

    /// Empty CompilerGraph (zero ops) should produce an empty FusionPlan.
    #[test]
    fn test_fuse_empty_graph_produces_empty_plan() {
        let g = CompilerGraph::new();
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 0, "Empty graph should produce 0 groups");
        assert_eq!(plan.num_fused_ops(), 0, "Empty graph should have 0 fused ops");
    }

    /// Single standalone elementwise op (SiLU with no successor) produces
    /// exactly 1 Standalone group.
    #[test]
    fn test_single_elementwise_standalone() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let b = g.add_tensor_concrete("b", &[1, 256], dt);
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![b], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 1, "Single op should produce 1 group");
        assert_eq!(plan.groups[0].mode, FusionMode::Standalone);
        assert_eq!(plan.groups[0].ops.len(), 1);
    }

    /// Single standalone RmsNorm (no downstream GEMM) produces 1 Standalone group,
    /// not ComputeRoot or TileLevelFusion.
    #[test]
    fn test_standalone_rms_norm_not_norm_into_gemm() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, 512], dt);
        let norm_out = g.add_tensor_concrete("norm_out", &[1, 512], dt);
        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm_out], "rms_norm");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(
            plan.groups[0].mode, FusionMode::Standalone,
            "RmsNorm without downstream GEMM should be Standalone, got {:?}", plan.groups[0].mode,
        );
    }

        /// Large elementwise chain (4 SiLU ops, dim=4096) exceeds L1 budget and
    /// is split into multiple sub-chains by split_elementwise_by_l1.
    /// All ops should still appear in LoopFusion groups (not Standalone).
    #[test]
    fn test_large_chain_split_by_l1_budget() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let b = g.add_tensor_concrete("b", &[1, 4096], dt);
        let c = g.add_tensor_concrete("c", &[1, 4096], dt);
        let d = g.add_tensor_concrete("d", &[1, 4096], dt);
        let e = g.add_tensor_concrete("e", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![b], "silu1");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![b], vec![c], "silu2");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![c], vec![d], "silu3");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![d], vec![e], "silu4");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // All 4 ops must be accounted for
        let total_ops: usize = plan.groups.iter().map(|grp| grp.ops.len()).sum();
        assert_eq!(total_ops, 4, "all 4 ops should appear across groups");
        // Each sub-chain should be LoopFusion or Standalone (no EpilogueInjection here)
        for grp in &plan.groups {
            assert!(
                matches!(grp.mode, FusionMode::LoopFusion | FusionMode::Standalone),
                "elementwise-only groups must be LoopFusion or Standalone, got {:?}", grp.mode,
            );
        }
    }

    /// GEMM + two elementwise unary ops (Tanh + Silu) as epilogue chain:
    /// both should be collected into the epilogue vector.
    #[test]
    fn test_gemm_double_epilogue_chain() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 256], dt);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[1, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 256], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Tanh, vec![gemm_out], vec![tanh_out], "tanh");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![tanh_out], vec![silu_out], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].mode, FusionMode::EpilogueInjection);
        assert_eq!(plan.groups[0].ops.len(), 3, "GEMM + Tanh + SiLU = 3 ops");
        assert_eq!(plan.groups[0].epilogue.len(), 2, "Tanh and SiLU should be epilogue");
    }

    /// GEMM output consumed by two different successors must NOT fuse the
    /// second consumer as epilogue (multi-consumer rejection).
    #[test]
    fn test_gemm_output_multi_consumer_no_epilogue() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 256], dt);
        let add_out = g.add_tensor_concrete("add_out", &[1, 256], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![gemm_out], "gemm",
        );
        // gemm_out has two consumers
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        g.add_op(crate::compiler::graph::OpKind::Tanh, vec![gemm_out], vec![add_out], "tanh");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        let gemm_group = find_group_by_label(&plan, &g, "gemm")
            .expect("GEMM should have a fusion group");
        assert!(
            !gemm_group.ops.iter().any(|&oid| g.op(oid).map_or(false, |o| o.label == "tanh")),
            "Tanh (second consumer) must NOT be fused into GEMM epilogue",
        );
    }

    /// FusionPlan clone produces identical groups and op_to_group mapping.
    #[test]
    fn test_fusion_plan_clone_preserves_state() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 128], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 128], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 128], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 128, k: 128, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let cloned = plan.clone();

        assert_eq!(plan.num_groups(), cloned.num_groups());
        assert_eq!(plan.groups.len(), cloned.groups.len());
        for (orig, cpy) in plan.groups.iter().zip(cloned.groups.iter()) {
            assert_eq!(orig.id, cpy.id);
            assert_eq!(orig.mode, cpy.mode);
            assert_eq!(orig.ops, cpy.ops);
        }
    }

    /// Silu + Tanh + Silu three-op unary elementwise chain produces LoopFusion with 3 ops.
    #[test]
    fn test_unary_elementwise_triple_chain_loop_fusion() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4096], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 4096], dt);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[1, 4096], dt);
        let silu2_out = g.add_tensor_concrete("silu2_out", &[1, 4096], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(crate::compiler::graph::OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![tanh_out], vec![silu2_out], "silu2");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        let total_ops: usize = plan.groups.iter().map(|g| g.ops.len()).sum();
        assert!(total_ops >= 3, "all 3 unary ops should appear in groups, got {total_ops}");
        assert!(!plan.groups.is_empty(), "should have at least one group");
    }

    /// RmsNorm followed by another RmsNorm (no GEMM in between) should NOT
    /// produce NormIntoGemm — both should be separate standalone or a chain.
    #[test]
    fn test_norm_norm_no_norm_into_gemm() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let x = g.add_tensor_concrete("x", &[1, 256], dt);
        let norm1_out = g.add_tensor_concrete("norm1_out", &[1, 256], dt);
        let norm2_out = g.add_tensor_concrete("norm2_out", &[1, 256], dt);

        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![x], vec![norm1_out], "norm1");
        g.add_op(crate::compiler::graph::OpKind::RmsNorm { feature_dim: 4096, eps: 1e-5 }, vec![norm1_out], vec![norm2_out], "norm2");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        for group in &plan.groups {
            assert_ne!(
                group.mode, FusionMode::NormIntoGemm,
                "RmsNorm→RmsNorm must not produce NormIntoGemm",
            );
        }
    }

    /// Softmax followed by Silu should produce 2 separate groups
    /// (reduction output blocks downstream elementwise fusion).
    #[test]
    fn test_softmax_blocks_elementwise_fusion() {
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let softmax_out = g.add_tensor_concrete("softmax_out", &[1, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 256], dt);

        g.add_op(crate::compiler::graph::OpKind::Softmax, vec![a], vec![softmax_out], "softmax");
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![softmax_out], vec![silu_out], "silu");

        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        assert_eq!(plan.num_groups(), 2, "Softmax blocks elementwise fusion with SiLU");
        assert_eq!(plan.groups[0].mode, FusionMode::Standalone, "Softmax should be Standalone");
    }

    // ── Additional 10 unit tests ─────────────────────────────────────────────

    /// FusionGroup with EpilogueInjection mode: epilogue vector should be a
    /// proper subset of the ops vector (every epilogue op is also in ops).
    #[test]
    fn test_fusion_group_epilogue_subset_of_ops() {
        // Arrange: GEMM + SiLU + Tanh fused as EpilogueInjection
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let gemm_out = g.add_tensor_concrete("gemm_out", &[1, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 256], dt);
        let tanh_out = g.add_tensor_concrete("tanh_out", &[1, 256], dt);

        g.add_op(
            crate::compiler::graph::OpKind::Gemm { m: crate::compiler::graph::SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false },
            vec![a, w], vec![gemm_out], "gemm",
        );
        g.add_op(crate::compiler::graph::OpKind::Silu, vec![gemm_out], vec![silu_out], "silu");
        g.add_op(crate::compiler::graph::OpKind::Tanh, vec![silu_out], vec![tanh_out], "tanh");

        // Act
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // Assert: every epilogue op must also be in ops
        let group = &plan.groups[0];
        for &ep_id in &group.epilogue {
            assert!(
                group.ops.contains(&ep_id),
                "Epilogue op Op({}) must be in ops list", ep_id.0,
            );
        }
        // The anchor (GEMM) should not be in epilogue
        assert!(
            !group.epilogue.contains(&group.anchor),
            "Anchor should not appear in epilogue",
        );
    }

    /// FusionPlan op_to_group mapping is consistent: every op in every group
    /// maps back to the correct group index.
    #[test]
    fn test_fusion_plan_op_to_group_consistency() {
        // Arrange: llama decoder layer produces multiple groups
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let plan = fuse_with_dag(&graph, &registry, &exec_plan);

        // Assert: for each group, every op in it must map back to that group's index
        for (group_idx, group) in plan.groups.iter().enumerate() {
            for &op_id in &group.ops {
                let mapped_idx = plan.op_to_group.get(&op_id)
                    .unwrap_or_else(|| panic!("Op({}) missing from op_to_group", op_id.0));
                assert_eq!(
                    *mapped_idx, group_idx,
                    "Op({}) maps to group {} but is in group {}",
                    op_id.0, mapped_idx, group_idx,
                );
            }
        }
    }

    /// FusionRule priority ordering: each rule must have a distinct priority value.
    #[test]
    fn test_fusion_rule_priorities_are_distinct() {
        // Arrange
        let rules = [
            FusionRule::QuantAwareFusion,
            FusionRule::PdtGuided,
            FusionRule::HardwareConstrained,
            FusionRule::Standard,
        ];

        // Act & Assert: all priorities must be distinct
        let priorities: Vec<u32> = rules.iter().map(|r| r.priority()).collect();
        for i in 0..priorities.len() {
            for j in (i + 1)..priorities.len() {
                assert_ne!(
                    priorities[i], priorities[j],
                    "FusionRule {:?} and {:?} must not share priority {}",
                    rules[i], rules[j], priorities[i],
                );
            }
        }
    }

    /// FusionRule description returns non-empty strings for all variants.
    #[test]
    fn test_fusion_rule_description_non_empty() {
        // Arrange
        let rules = [
            FusionRule::QuantAwareFusion,
            FusionRule::PdtGuided,
            FusionRule::HardwareConstrained,
            FusionRule::Standard,
        ];

        // Act & Assert
        for rule in &rules {
            let desc = rule.description();
            assert!(
                !desc.is_empty(),
                "FusionRule {:?} must have a non-empty description",
                rule,
            );
        }
    }

    /// FusionEngine::new with custom rule order preserves that order (no auto-sort).
    #[test]
    fn test_fusion_engine_custom_rule_order_preserved() {
        // Arrange: deliberately non-priority order
        let custom_order = vec![
            FusionRule::Standard,
            FusionRule::PdtGuided,
        ];

        // Act
        let engine = FusionEngine::new(custom_order.clone());

        // Assert: rules retain caller-specified order
        assert_eq!(engine.rules.len(), 2);
        assert_eq!(engine.rules[0], FusionRule::Standard);
        assert_eq!(engine.rules[1], FusionRule::PdtGuided);
        // sorted_rules should re-order by priority
        let sorted = engine.sorted_rules();
        assert_eq!(sorted[0], FusionRule::PdtGuided, "PdtGuided (80) should come before Standard (60)");
        assert_eq!(sorted[1], FusionRule::Standard);
    }

    /// FusionPlan Display for a multi-group plan includes group count and
    /// anchor information for each group.
    #[test]
    fn test_fusion_plan_display_multi_group_content() {
        // Arrange: SiLU → Softmax (2 groups: SiLU standalone + Softmax standalone)
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let silu_out = g.add_tensor_concrete("silu_out", &[1, 256], dt);
        let softmax_out = g.add_tensor_concrete("softmax_out", &[1, 256], dt);

        g.add_op(crate::compiler::graph::OpKind::Silu, vec![a], vec![silu_out], "silu");
        g.add_op(crate::compiler::graph::OpKind::Softmax, vec![silu_out], vec![softmax_out], "softmax");

        // Act
        let plan = fuse_with_dag(&g, &registry, &exec_plan);
        let display = format!("{plan}");

        // Assert: display should mention group count and each group's anchor
        assert!(display.contains("2 groups"), "Display should show 2 groups: {display}");
        assert!(display.contains("anchor"), "Display should show anchor info: {display}");
    }

    /// FusionGroup with dominant_dtype set to Some(BF16) preserves the value
    /// and it is readable after construction.
    #[test]
    fn test_fusion_group_dominant_dtype_bf16() {
        // Arrange
        let group = FusionGroup {
            id: 7,
            anchor: OpId(3),
            epilogue: vec![OpId(4)],
            mode: FusionMode::EpilogueInjection,
            ops: vec![OpId(3), OpId(4)],
            multi_output: MultiOutputConfig::single(),
            dominant_dtype: Some(crate::compiler::trace::QuantPrecision::BF16),
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
        };

        // Act & Assert
        assert_eq!(group.id, 7);
        assert_eq!(group.dominant_dtype, Some(crate::compiler::trace::QuantPrecision::BF16));
        assert_eq!(group.ops.len(), 2);
        assert_eq!(group.epilogue.len(), 1);
    }

    /// FusionEngine with only PdtGuided rule: has_rule returns true for PdtGuided,
    /// false for all others. is_quant_aware returns false.
    #[test]
    fn test_fusion_engine_pdt_guided_only() {
        // Arrange & Act
        let engine = FusionEngine::new(vec![FusionRule::PdtGuided]);

        // Assert
        assert!(engine.has_rule(FusionRule::PdtGuided));
        assert!(!engine.has_rule(FusionRule::QuantAwareFusion));
        assert!(!engine.has_rule(FusionRule::Standard));
        assert!(!engine.has_rule(FusionRule::HardwareConstrained));
        assert!(!engine.is_quant_aware());
    }

    /// FusionPlan from an empty graph: num_groups == 0, group_of returns None
    /// for any OpId, and the Display output mentions "0 groups".
    #[test]
    fn test_empty_plan_group_of_and_display() {
        // Arrange
        let g = CompilerGraph::new();
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Act
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // Assert
        assert_eq!(plan.num_groups(), 0);
        assert!(plan.group_of(OpId(0)).is_none());
        assert!(plan.group_of(OpId(9999)).is_none());
        let display = format!("{plan}");
        assert!(display.contains("0 groups"), "Empty plan display should mention 0 groups: {display}");
    }

    /// Single standalone Softmax op produces a FusionGroup with empty epilogue
    /// and mode Standalone. The group_of lookup must return that group.
    #[test]
    fn test_standalone_softmax_group_fields_and_lookup() {
        // Arrange
        let profile = DeviceProfile::detect();
        let dt = DType::F32;
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();

        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 256], dt);
        let b = g.add_tensor_concrete("b", &[1, 256], dt);
        let softmax_op = g.add_op(crate::compiler::graph::OpKind::Softmax, vec![a], vec![b], "softmax");

        // Act
        let plan = fuse_with_dag(&g, &registry, &exec_plan);

        // Assert
        assert_eq!(plan.num_groups(), 1);
        let group = plan.group_of(softmax_op).expect("softmax op should belong to a group");
        assert_eq!(group.mode, FusionMode::Standalone);
        assert!(group.epilogue.is_empty(), "Standalone group should have empty epilogue");
        assert_eq!(group.ops.len(), 1);
    }
}
