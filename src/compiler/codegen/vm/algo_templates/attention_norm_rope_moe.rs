//! Attention + Norm + RoPE + MoE template instances (SPEC 27 REQ-AT-005)

use crate::compiler::codegen::vm::algo_template::*;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Attention Templates
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static ATTN_MHA: AlgoTemplate = AlgoTemplate {
    name: "ATTN_MHA",
    strategy: AlgoStrategy::AttnMha,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // QK^T = Q × K^T (seq × d_head × d_head × seq → seq × seq)
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "q" },
            AlgoTraceStep::LoadInput { name: "k" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "qk", a: "q", b: "k" },
        ]),
        // Scale by 1/sqrt(d_head)
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadConst { value: 0.0 }, // filled from graph config
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "qk_scaled", a: "qk", b: "scale" },
        ]),
        // Softmax(QK^T / sqrt(d_head))
        AlgoStep::Softmax,
        // Attn_weights × V
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "v" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "attn_out", a: "attn_weights", b: "v" },
        ]),
    ],
    params: &[
        ("seq_len", AlgoParam::FromGraph("seq_len")),
        ("num_heads", AlgoParam::FromGraph("num_heads")),
        ("head_dim", AlgoParam::FromGraph("head_dim")),
    ],
    micro_kernel: None,
};

pub static ATTN_GQA: AlgoTemplate = AlgoTemplate {
    name: "ATTN_GQA",
    strategy: AlgoStrategy::AttnGqa,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Same structure as MHA but with KV head grouping
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "q" },
            AlgoTraceStep::LoadInput { name: "k" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "qk", a: "q", b: "k" },
        ]),
        AlgoStep::Softmax,
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "v" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "attn_out", a: "attn_weights", b: "v" },
        ]),
    ],
    params: &[
        ("seq_len", AlgoParam::FromGraph("seq_len")),
        ("num_q_heads", AlgoParam::FromGraph("num_q_heads")),
        ("num_kv_heads", AlgoParam::FromGraph("num_kv_heads")),
        ("head_dim", AlgoParam::FromGraph("head_dim")),
        ("kv_group_size", AlgoParam::Derived { base: "num_q_heads", op: ParamArith::Div, operand: 0 }),
    ],
    micro_kernel: None,
};

pub static ATTN_MLA: AlgoTemplate = AlgoTemplate {
    name: "ATTN_MLA",
    strategy: AlgoStrategy::AttnMla,
    device_req: DeviceReq::CpuAvx2,
    steps: &[
        // MLA: absorbed Q × c_KV (low-rank compressed)
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "q_absorbed" },
            AlgoTraceStep::LoadInput { name: "c_kv" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "qk", a: "q_absorbed", b: "c_kv" },
        ]),
        AlgoStep::Softmax,
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "v" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "attn_out", a: "attn_weights", b: "v" },
        ]),
    ],
    params: &[
        ("seq_len", AlgoParam::FromGraph("seq_len")),
        ("kv_dim", AlgoParam::FromGraph("kv_dim")),
    ],
    micro_kernel: None,
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Norm Templates
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static NORM_RMS: AlgoTemplate = AlgoTemplate {
    name: "NORM_RMS",
    strategy: AlgoStrategy::NormRms,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Three-phase: reduce_sum(x^2) → rsqrt(sum/N + eps) → x * scale * rsqrt
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "x" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "x2", a: "x", b: "x" },
            AlgoTraceStep::HReduce { src: "x2", op: ReduceKind::Sum },
        ]),
        AlgoStep::Reduce { op: ReduceOp::Sum },
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadParam { name: "eps" }, // eps from NormSpec, resolved at instantiation
            AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "mean_eps", a: "sum", b: "eps" },
            AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Rsqrt, dst: "inv_rms", src: "mean_eps" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "norm_out", a: "x", b: "inv_rms" },
            AlgoTraceStep::LoadInput { name: "gamma" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "out", a: "norm_out", b: "gamma" },
        ]),
    ],
    params: &[
        ("hidden_dim", AlgoParam::FromGraph("hidden_dim")),
        ("eps", AlgoParam::FromGraph("rms_norm_eps")),
    ],
    micro_kernel: None,
};

pub static NORM_LAYER: AlgoTemplate = AlgoTemplate {
    name: "NORM_LAYER",
    strategy: AlgoStrategy::NormLayer,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Three-phase: reduce_mean(x) → x - mean → rsqrt(var + eps) → out
        AlgoStep::Reduce { op: ReduceOp::Sum },
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "x" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Sub, dst: "centered", a: "x", b: "mean" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "var", a: "centered", b: "centered" },
        ]),
        AlgoStep::Reduce { op: ReduceOp::Sum },
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadParam { name: "eps" }, // eps from NormSpec, resolved at instantiation
            AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "var_eps", a: "var", b: "eps" },
            AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Rsqrt, dst: "inv_std", src: "var_eps" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "norm_out", a: "centered", b: "inv_std" },
            AlgoTraceStep::LoadInput { name: "gamma" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "scaled", a: "norm_out", b: "gamma" },
            AlgoTraceStep::LoadInput { name: "beta" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "out", a: "scaled", b: "beta" },
        ]),
    ],
    params: &[
        ("hidden_dim", AlgoParam::FromGraph("hidden_dim")),
        ("eps", AlgoParam::FromGraph("norm_eps")),
    ],
    micro_kernel: None,
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RoPE Templates
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static ROPE_STANDARD: AlgoTemplate = AlgoTemplate {
    name: "ROPE_STANDARD",
    strategy: AlgoStrategy::RopeStandard,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // x_rot = x * cos + rotate_half(x) * sin
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "x" },
            AlgoTraceStep::LoadInput { name: "cos" },
            AlgoTraceStep::LoadInput { name: "sin" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "x_cos", a: "x", b: "cos" },
            // rotate_half: [-x2, x1] from [x1, x2]
            AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Neg, dst: "neg_x2", src: "x_half2" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "rot_sin", a: "neg_x2", b: "sin" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "out", a: "x_cos", b: "rot_sin" },
        ]),
    ],
    params: &[
        ("seq_len", AlgoParam::FromGraph("seq_len")),
        ("head_dim", AlgoParam::FromGraph("head_dim")),
    ],
    micro_kernel: None,
};

pub static ROPE_PARTIAL: AlgoTemplate = AlgoTemplate {
    name: "ROPE_PARTIAL",
    strategy: AlgoStrategy::RopePartial,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Partial RoPE: only first partial_dim gets rotation, rest passes through
        AlgoStep::Conditional {
            requirement: DeviceReq::CpuAny,
            body: &[
                AlgoStep::TraceBody(&[
                    AlgoTraceStep::LoadInput { name: "x_rot" },
                    AlgoTraceStep::LoadInput { name: "cos" },
                    AlgoTraceStep::LoadInput { name: "sin" },
                    AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "x_cos", a: "x_rot", b: "cos" },
                    AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "rot_sin", a: "x_rot_neg", b: "sin" },
                    AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "rotated", a: "x_cos", b: "rot_sin" },
                ]),
            ],
        },
    ],
    params: &[
        ("seq_len", AlgoParam::FromGraph("seq_len")),
        ("head_dim", AlgoParam::FromGraph("head_dim")),
        ("partial_dim", AlgoParam::FromGraph("partial_dim")),
    ],
    micro_kernel: None,
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MoE Templates
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static MOE_ROUTER_TOPK: AlgoTemplate = AlgoTemplate {
    name: "MOE_ROUTER_TOPK",
    strategy: AlgoStrategy::MoeRouterTopk,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Router: hidden → expert_scores → top-k selection
        AlgoStep::MoeRouterGemv {
            num_experts: "num_experts",
            hidden: "hidden_dim",
        },
        AlgoStep::Activation { kind: ActivationKind::Relu },
        AlgoStep::Softmax,
        AlgoStep::MoeTopK {
            num_experts: "num_experts",
            top_k: "top_k",
        },
    ],
    params: &[
        ("num_experts", AlgoParam::FromGraph("num_experts")),
        ("hidden_dim", AlgoParam::FromGraph("hidden_dim")),
        ("top_k", AlgoParam::FromGraph("top_k")),
    ],
    micro_kernel: None,
};

pub static MOE_PACKED_DISPATCH: AlgoTemplate = AlgoTemplate {
    name: "MOE_PACKED_DISPATCH",
    strategy: AlgoStrategy::MoePackedDispatch,
    device_req: DeviceReq::CpuAvx2,
    steps: &[
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "hidden" },
            AlgoTraceStep::LoadInput { name: "expert_weights" },
            AlgoTraceStep::LoadInput { name: "router_weights" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "gated", a: "hidden", b: "expert_weights" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "weighted", a: "gated", b: "router_weights" },
        ]),
    ],
    params: &[
        ("hidden_dim", AlgoParam::FromGraph("hidden_dim")),
        ("num_experts", AlgoParam::FromGraph("num_experts")),
        ("top_k", AlgoParam::FromGraph("top_k")),
    ],
    micro_kernel: None,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: ATTN_MHA static template field correctness ─────────────

    #[test]
    fn attn_mha_strategy_and_device_req() {
        // Arrange & Act: access the ATTN_MHA static directly
        // Assert: strategy maps to Attention family
        assert_eq!(ATTN_MHA.name, "ATTN_MHA");
        assert_eq!(ATTN_MHA.strategy, AlgoStrategy::AttnMha);
        assert_eq!(ATTN_MHA.strategy.family(), StrategyFamily::Attention);
        assert_eq!(ATTN_MHA.device_req, DeviceReq::CpuAny);
        assert!(ATTN_MHA.micro_kernel.is_none());
    }

    // ── Test 2: ATTN_GQA derived param for kv_group_size ──────────────

    #[test]
    fn attn_gqa_params_include_derived_kv_group() {
        // Arrange: ATTN_GQA template
        // Act: collect param names
        let param_names: Vec<&&str> = ATTN_GQA.params.iter().map(|(name, _)| name).collect();

        // Assert: kv_group_size is a Derived param with Div arithmetic
        assert!(param_names.iter().any(|n| **n == "kv_group_size"));
        let kv_group_param = ATTN_GQA.params.iter()
            .find(|(name, _)| *name == "kv_group_size")
            .map(|(_, p)| p)
            .expect("kv_group_size param must exist");
        match kv_group_param {
            AlgoParam::Derived { base, op: ParamArith::Div, operand: 0 } => {
                assert_eq!(*base, "num_q_heads");
            }
            _ => panic!("kv_group_size must be Derived with Div, got {:?}", kv_group_param),
        }
    }

    // ── Test 3: ATTN_MLA requires CpuAvx2 minimum ─────────────────────

    #[test]
    fn attn_mla_device_req_cpu_avx2() {
        // Arrange & Act: check ATTN_MLA device requirement
        // Assert: MLA uses compressed KV and requires AVX2 minimum
        assert_eq!(ATTN_MLA.device_req, DeviceReq::CpuAvx2);
        assert!(ATTN_MLA.device_req.priority() > DeviceReq::CpuAny.priority());
        assert_eq!(ATTN_MLA.strategy, AlgoStrategy::AttnMla);
    }

    // ── Test 4: NORM_RMS steps contain three phases (reduce/rsqrt/scale) ─

    #[test]
    fn norm_rms_steps_contain_reduce_and_trace_body() {
        // Arrange: NORM_RMS template
        // Act: classify step types
        let step_types: Vec<String> = NORM_RMS.steps.iter().map(|s| {
            match s {
                AlgoStep::TraceBody(_) => "TraceBody".to_string(),
                AlgoStep::Reduce { .. } => "Reduce".to_string(),
                AlgoStep::Softmax => "Softmax".to_string(),
                other => format!("{:?}", other),
            }
        }).collect();

        // Assert: has TraceBody and Reduce steps
        assert!(step_types.iter().any(|t| t == "TraceBody"),
            "NORM_RMS must have at least one TraceBody step");
        assert!(step_types.iter().any(|t| t == "Reduce"),
            "NORM_RMS must have at least one Reduce step");
    }

    // ── Test 5: NORM_RMS has eps param from graph ─────────────────────

    #[test]
    fn norm_rms_params_include_eps_from_graph() {
        // Arrange: NORM_RMS template
        // Act: find eps param
        let eps_param = NORM_RMS.params.iter()
            .find(|(name, _)| *name == "eps")
            .map(|(_, p)| p);

        // Assert: eps is FromGraph("rms_norm_eps")
        match eps_param {
            Some(AlgoParam::FromGraph(key)) => assert_eq!(*key, "rms_norm_eps"),
            Some(other) => panic!("eps param should be FromGraph, got {:?}", other),
            None => panic!("NORM_RMS must have eps parameter"),
        }
    }

    // ── Test 6: NORM_LAYER has beta input (unlike NORM_RMS) ───────────

    #[test]
    fn norm_layer_steps_include_beta_load() {
        // Arrange: NORM_LAYER template
        // Act: extract all LoadInput names from TraceBody steps
        let mut input_names: Vec<&str> = Vec::new();
        for step in NORM_LAYER.steps.iter() {
            if let AlgoStep::TraceBody(traces) = step {
                for trace in traces.iter() {
                    if let AlgoTraceStep::LoadInput { name } = trace {
                        input_names.push(name);
                    }
                }
            }
        }

        // Assert: beta is loaded (LayerNorm has shift parameter, RMSNorm does not)
        assert!(input_names.iter().any(|n| *n == "beta"),
            "NORM_LAYER must load 'beta' input (shift parameter)");
        assert!(input_names.iter().any(|n| *n == "gamma"),
            "NORM_LAYER must load 'gamma' input (scale parameter)");
    }

    // ── Test 7: ROPE_STANDARD step contains rotate_half via Neg unary ──

    #[test]
    fn rope_standard_trace_body_contains_neg_unary_op() {
        // Arrange: ROPE_STANDARD template
        // Act: find Neg unary op in TraceBody
        let has_neg = ROPE_STANDARD.steps.iter().any(|step| {
            if let AlgoStep::TraceBody(traces) = step {
                traces.iter().any(|t| {
                    matches!(t, AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Neg, .. })
                })
            } else {
                false
            }
        });

        // Assert: rotate_half uses Neg for [-x2, x1] transform
        assert!(has_neg, "ROPE_STANDARD must have Neg unary op for rotate_half");
    }

    // ── Test 8: ROPE_PARTIAL uses Conditional step structure ──────────

    #[test]
    fn rope_partial_step_is_conditional() {
        // Arrange: ROPE_PARTIAL template
        // Act: check first step type
        let first_step = &ROPE_PARTIAL.steps[0];

        // Assert: first step is Conditional (partial rotation, conditional execution)
        match first_step {
            AlgoStep::Conditional { requirement, body } => {
                assert_eq!(*requirement, DeviceReq::CpuAny);
                assert!(!body.is_empty(), "Conditional body must not be empty");
            }
            other => panic!("ROPE_PARTIAL first step must be Conditional, got {:?}", other),
        }
    }

    // ── Test 9: MOE_ROUTER_TOPK has Softmax and MoeTopK steps ─────────

    #[test]
    fn moe_router_topk_step_variants() {
        // Arrange: MOE_ROUTER_TOPK template
        // Act: collect step variant names
        let has_softmax = MOE_ROUTER_TOPK.steps.iter().any(|s| matches!(s, AlgoStep::Softmax));
        let has_topk = MOE_ROUTER_TOPK.steps.iter().any(|s| matches!(s, AlgoStep::MoeTopK { .. }));
        let has_activation = MOE_ROUTER_TOPK.steps.iter()
            .any(|s| matches!(s, AlgoStep::Activation { kind: ActivationKind::Relu }));

        // Assert: router has softmax normalization and top-k selection
        assert!(has_softmax, "MOE_ROUTER_TOPK must have Softmax step");
        assert!(has_topk, "MOE_ROUTER_TOPK must have MoeTopK step");
        assert!(has_activation, "MOE_ROUTER_TOPK must have Activation::Relu step");
    }

    // ── Test 10: MOE_ROUTER_TOPK MoeTopK step field extraction ────────

    #[test]
    fn moe_router_topk_step_field_values() {
        // Arrange: MOE_ROUTER_TOPK template
        // Act: extract MoeRouterGemv and MoeTopK step fields
        let router_gemv = MOE_ROUTER_TOPK.steps.iter().find_map(|s| {
            if let AlgoStep::MoeRouterGemv { num_experts, hidden } = s {
                Some((*num_experts, *hidden))
            } else {
                None
            }
        });
        let topk = MOE_ROUTER_TOPK.steps.iter().find_map(|s| {
            if let AlgoStep::MoeTopK { num_experts, top_k } = s {
                Some((*num_experts, *top_k))
            } else {
                None
            }
        });

        // Assert: field values reference correct parameter names
        assert_eq!(router_gemv, Some(("num_experts", "hidden_dim")));
        assert_eq!(topk, Some(("num_experts", "top_k")));
    }

    // ── Test 11: All attention templates share common Softmax step ─────

    #[test]
    fn all_attention_templates_have_softmax() {
        // Arrange: all attention template references
        let attention_templates: Vec<(&str, &AlgoTemplate)> = vec![
            ("ATTN_MHA", &ATTN_MHA),
            ("ATTN_GQA", &ATTN_GQA),
            ("ATTN_MLA", &ATTN_MLA),
        ];

        // Act & Assert: every attention template has a Softmax step
        for (name, tmpl) in attention_templates {
            let has_softmax = tmpl.steps.iter().any(|s| matches!(s, AlgoStep::Softmax));
            assert!(has_softmax, "{} must contain a Softmax step", name);
        }
    }

    // ── Test 12: Device requirement hierarchy across template categories ─

    #[test]
    fn device_requirement_specialization_per_category() {
        // Arrange: pick templates from different categories
        // Act: compare device requirement priorities
        let cpu_any = NORM_RMS.device_req.priority();
        let cpu_avx2_mla = ATTN_MLA.device_req.priority();
        let cpu_avx2_moe = MOE_PACKED_DISPATCH.device_req.priority();

        // Assert: CpuAny templates have lowest priority, AVX2 templates are higher
        assert_eq!(ATTN_MHA.device_req, DeviceReq::CpuAny);
        assert_eq!(NORM_RMS.device_req, DeviceReq::CpuAny);
        assert_eq!(ROPE_STANDARD.device_req, DeviceReq::CpuAny);
        assert_eq!(cpu_avx2_mla, DeviceReq::CpuAvx2.priority());
        assert_eq!(cpu_avx2_moe, DeviceReq::CpuAvx2.priority());
        assert!(cpu_any < cpu_avx2_mla);
        assert!(cpu_any < cpu_avx2_moe);
    }

    // ── Test 13: No template in this file has micro_kernel defined ─────

    #[test]
    fn no_template_has_micro_kernel() {
        // Arrange: all templates defined in this file
        let templates: Vec<(&str, &AlgoTemplate)> = vec![
            ("ATTN_MHA", &ATTN_MHA),
            ("ATTN_GQA", &ATTN_GQA),
            ("ATTN_MLA", &ATTN_MLA),
            ("NORM_RMS", &NORM_RMS),
            ("NORM_LAYER", &NORM_LAYER),
            ("ROPE_STANDARD", &ROPE_STANDARD),
            ("ROPE_PARTIAL", &ROPE_PARTIAL),
            ("MOE_ROUTER_TOPK", &MOE_ROUTER_TOPK),
            ("MOE_PACKED_DISPATCH", &MOE_PACKED_DISPATCH),
        ];

        // Act & Assert: none should have a micro_kernel
        for (name, tmpl) in templates {
            assert!(tmpl.micro_kernel.is_none(),
                "{} should not have micro_kernel (attention/norm/rope/moe are not GEMM)", name);
        }
    }

    // ── Test 14: ATTN_MHA params are all FromGraph ────────────────────

    #[test]
    fn attn_mha_params_all_from_graph() {
        // Arrange: ATTN_MHA template
        // Act: iterate params and check source type
        for (name, param) in ATTN_MHA.params.iter() {
            match param {
                AlgoParam::FromGraph(graph_key) => {
                    // Assert: graph key is non-empty and matches a known dimension
                    assert!(!graph_key.is_empty(),
                        "ATTN_MHA param '{}' must have non-empty graph key", name);
                }
                other => panic!(
                    "ATTN_MHA param '{}' should be FromGraph, got {:?}", name, other),
            }
        }
        // Assert: exactly 3 params (seq_len, num_heads, head_dim)
        assert_eq!(ATTN_MHA.params.len(), 3,
            "ATTN_MHA must have exactly 3 parameters");
    }

    // ── Test 15: NORM_RMS trace body contains HReduce with Sum ────────

    #[test]
    fn norm_rms_trace_body_contains_hreduce_sum() {
        // Arrange: NORM_RMS template
        // Act: search for HReduce with Sum in TraceBody steps
        let has_hreduce_sum = NORM_RMS.steps.iter().any(|step| {
            if let AlgoStep::TraceBody(traces) = step {
                traces.iter().any(|t| {
                    matches!(t, AlgoTraceStep::HReduce { op: ReduceKind::Sum, .. })
                })
            } else {
                false
            }
        });

        // Assert: RMS norm computes sum-of-squares via horizontal reduction
        assert!(has_hreduce_sum,
            "NORM_RMS must contain HReduce with Sum in its TraceBody");
    }

    // ── Test 16: NORM_LAYER has two Reduce steps (mean then variance) ─

    #[test]
    fn norm_layer_has_two_reduce_steps() {
        // Arrange: NORM_LAYER template
        // Act: count Reduce steps
        let reduce_count = NORM_LAYER.steps.iter()
            .filter(|s| matches!(s, AlgoStep::Reduce { .. }))
            .count();

        // Assert: LayerNorm has two phases — mean reduction and variance reduction
        assert_eq!(reduce_count, 2,
            "NORM_LAYER must have exactly 2 Reduce steps (mean + variance)");
    }

    // ── Test 17: ROPE_STANDARD trace body has Add as final op ─────────

    #[test]
    fn rope_standard_trace_body_final_binop_is_add() {
        // Arrange: ROPE_STANDARD template
        // Act: find the TraceBody and check the last BinOp
        let last_binop = ROPE_STANDARD.steps.iter()
            .find_map(|step| {
                if let AlgoStep::TraceBody(traces) = step {
                    traces.iter().rev().find_map(|t| {
                        if let AlgoTraceStep::BinOp { op, .. } = t {
                            Some(*op)
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            });

        // Assert: x_rot * cos + rotate_half(x) * sin is combined with Add
        match last_binop {
            Some(TraceBinOp::Add) => {},
            other => panic!(
                "ROPE_STANDARD final BinOp must be Add (cos + sin components), got {:?}",
                other),
        }
    }

    // ── Test 18: MOE_ROUTER_TOPK has three distinct FromGraph params ──

    #[test]
    fn moe_router_topk_params_count_and_source() {
        // Arrange: MOE_ROUTER_TOPK template
        // Act: collect FromGraph param names
        let from_graph_names: Vec<&str> = MOE_ROUTER_TOPK.params.iter()
            .filter_map(|(name, param)| {
                if let AlgoParam::FromGraph(_) = param {
                    Some(*name)
                } else {
                    None
                }
            })
            .collect();

        // Assert: all three params come from graph metadata
        assert_eq!(from_graph_names.len(), 3,
            "MOE_ROUTER_TOPK must have exactly 3 FromGraph params");
        assert!(from_graph_names.contains(&"num_experts"),
            "must include num_experts param");
        assert!(from_graph_names.contains(&"hidden_dim"),
            "must include hidden_dim param");
        assert!(from_graph_names.contains(&"top_k"),
            "must include top_k param");
    }

    // ── Test 19: ATTN_GQA has more params than ATTN_MHA ──────────────

    #[test]
    fn attn_gqa_has_more_params_than_mha() {
        // Arrange: both attention templates
        // Act: compare param counts
        let mha_count = ATTN_MHA.params.len();
        let gqa_count = ATTN_GQA.params.len();

        // Assert: GQA adds num_kv_heads and derived kv_group_size
        assert!(gqa_count > mha_count,
            "ATTN_GQA ({} params) must have more params than ATTN_MHA ({} params) \
             because GQA tracks num_kv_heads and derived kv_group_size",
            gqa_count, mha_count);
    }

    // ── Test 20: ROPE_PARTIAL has partial_dim param ───────────────────

    #[test]
    fn rope_partial_params_include_partial_dim() {
        // Arrange: ROPE_PARTIAL template
        // Act: find the partial_dim parameter
        let partial_dim_param = ROPE_PARTIAL.params.iter()
            .find(|(name, _)| *name == "partial_dim");

        // Assert: partial_dim exists and is FromGraph
        assert!(partial_dim_param.is_some(),
            "ROPE_PARTIAL must have partial_dim parameter");
        if let Some((_, param)) = partial_dim_param {
            match param {
                AlgoParam::FromGraph(key) => {
                    assert_eq!(*key, "partial_dim",
                        "partial_dim must reference graph key 'partial_dim'");
                }
                other => panic!(
                    "partial_dim must be FromGraph, got {:?}", other),
            }
        }
    }

    // ── Test 21: NORM_RMS has exactly one Reduce step ─────────────────

    #[test]
    fn norm_rms_has_exactly_one_reduce_step() {
        // Arrange: NORM_RMS template
        // Act: count Reduce steps
        let reduce_count = NORM_RMS.steps.iter()
            .filter(|s| matches!(s, AlgoStep::Reduce { .. }))
            .count();

        // Assert: RMS norm has a single reduction (sum of squares)
        assert_eq!(reduce_count, 1,
            "NORM_RMS must have exactly 1 Reduce step (sum of x^2)");
    }

    // ── Test 22: ATTN_MLA uses q_absorbed and c_kv inputs ─────────────

    #[test]
    fn attn_mla_loads_absorbed_inputs() {
        // Arrange: ATTN_MLA template
        // Act: collect all LoadInput names from TraceBody steps
        let input_names: Vec<&str> = ATTN_MLA.steps.iter()
            .filter_map(|step| {
                if let AlgoStep::TraceBody(traces) = step {
                    Some(traces.iter().filter_map(|t| {
                        if let AlgoTraceStep::LoadInput { name } = t {
                            Some(*name)
                        } else {
                            None
                        }
                    }))
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        // Assert: MLA uses compressed/absorbed inputs instead of raw Q/K
        assert!(input_names.iter().any(|n| *n == "q_absorbed"),
            "ATTN_MLA must load 'q_absorbed' input");
        assert!(input_names.iter().any(|n| *n == "c_kv"),
            "ATTN_MLA must load 'c_kv' (compressed KV) input");
    }

    // ── Test 23: MOE_PACKED_DISPATCH has exactly one TraceBody step ───

    #[test]
    fn moe_packed_dispatch_single_trace_body_step() {
        // Arrange: MOE_PACKED_DISPATCH template
        // Act: count TraceBody steps
        let trace_body_count = MOE_PACKED_DISPATCH.steps.iter()
            .filter(|s| matches!(s, AlgoStep::TraceBody(_)))
            .count();

        // Assert: packed dispatch is a single fused trace body
        assert_eq!(trace_body_count, 1,
            "MOE_PACKED_DISPATCH must have exactly 1 TraceBody step");
        assert_eq!(MOE_PACKED_DISPATCH.steps.len(), 1,
            "MOE_PACKED_DISPATCH must have exactly 1 total step");
    }

    // ── Test 24: All norm templates belong to StrategyFamily::Norm ────

    #[test]
    fn all_norm_templates_belong_to_norm_family() {
        // Arrange: both norm templates
        let norm_templates: Vec<(&str, &AlgoTemplate)> = vec![
            ("NORM_RMS", &NORM_RMS),
            ("NORM_LAYER", &NORM_LAYER),
        ];

        // Act & Assert: every norm template has Norm strategy family
        for (name, tmpl) in norm_templates {
            assert_eq!(tmpl.strategy.family(), StrategyFamily::Norm,
                "{} must belong to Norm family", name);
        }
    }

    // ── Test 25: All attention templates belong to Attention family ──────

    #[test]
    fn all_attention_templates_belong_to_attention_family() {
        let templates: Vec<(&str, &AlgoTemplate)> = vec![
            ("ATTN_MHA", &ATTN_MHA),
            ("ATTN_GQA", &ATTN_GQA),
            ("ATTN_MLA", &ATTN_MLA),
        ];
        for (name, tmpl) in templates {
            assert_eq!(tmpl.strategy.family(), StrategyFamily::Attention,
                "{} must belong to Attention family", name);
        }
    }

    // ── Test 26: ROPE templates belong to Rope family ──────────────────

    #[test]
    fn rope_templates_belong_to_rope_family() {
        assert_eq!(ROPE_STANDARD.strategy.family(), StrategyFamily::Rope);
        assert_eq!(ROPE_PARTIAL.strategy.family(), StrategyFamily::Rope);
    }

    // ── Test 27: MOE templates belong to Moe family ────────────────────

    #[test]
    fn moe_templates_belong_to_moe_family() {
        assert_eq!(MOE_ROUTER_TOPK.strategy.family(), StrategyFamily::Moe);
        assert_eq!(MOE_PACKED_DISPATCH.strategy.family(), StrategyFamily::Moe);
    }

    // ── Test 28: ATTN_MHA steps count ──────────────────────────────────

    #[test]
    fn attn_mha_steps_count() {
        // MHA: TraceBody(qk) + TraceBody(scale) + Softmax + TraceBody(attn_out) = 4
        assert_eq!(ATTN_MHA.steps.len(), 4, "ATTN_MHA must have 4 steps");
    }

    // ── Test 29: ATTN_GQA steps count ──────────────────────────────────

    #[test]
    fn attn_gqa_steps_count() {
        // GQA: TraceBody(qk) + Softmax + TraceBody(attn_out) = 3
        assert_eq!(ATTN_GQA.steps.len(), 3, "ATTN_GQA must have 3 steps");
    }

    // ── Test 30: ATTN_MLA params count ─────────────────────────────────

    #[test]
    fn attn_mla_params_count() {
        assert_eq!(ATTN_MLA.params.len(), 2, "ATTN_MLA must have 2 params (seq_len, kv_dim)");
    }

    // ── Test 31: NORM_RMS trace body has Rsqrt unary op ────────────────

    #[test]
    fn norm_rms_trace_body_has_rsqrt() {
        let has_rsqrt = NORM_RMS.steps.iter().any(|step| {
            if let AlgoStep::TraceBody(traces) = step {
                traces.iter().any(|t| {
                    matches!(t, AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Rsqrt, .. })
                })
            } else {
                false
            }
        });
        assert!(has_rsqrt, "NORM_RMS must contain Rsqrt unary op");
    }

    // ── Test 32: NORM_RMS loads gamma input ────────────────────────────

    #[test]
    fn norm_rms_loads_gamma() {
        let input_names: Vec<&str> = NORM_RMS.steps.iter()
            .filter_map(|step| {
                if let AlgoStep::TraceBody(traces) = step {
                    Some(traces.iter().filter_map(|t| {
                        if let AlgoTraceStep::LoadInput { name } = t { Some(*name) } else { None }
                    }))
                } else { None }
            })
            .flatten()
            .collect();
        assert!(input_names.iter().any(|n| *n == "gamma"),
            "NORM_RMS must load gamma (scale parameter)");
    }

    // ── Test 33: MOE_PACKED_DISPATCH loads three inputs ────────────────

    #[test]
    fn moe_packed_dispatch_loads_three_inputs() {
        let input_names: Vec<&str> = MOE_PACKED_DISPATCH.steps.iter()
            .filter_map(|step| {
                if let AlgoStep::TraceBody(traces) = step {
                    Some(traces.iter().filter_map(|t| {
                        if let AlgoTraceStep::LoadInput { name } = t { Some(*name) } else { None }
                    }))
                } else { None }
            })
            .flatten()
            .collect();
        assert!(input_names.iter().any(|n| *n == "hidden"));
        assert!(input_names.iter().any(|n| *n == "expert_weights"));
        assert!(input_names.iter().any(|n| *n == "router_weights"));
    }

    // ── Test 34: ROPE_STANDARD has two FromGraph params ────────────────

    #[test]
    fn rope_standard_params_count() {
        assert_eq!(ROPE_STANDARD.params.len(), 2, "ROPE_STANDARD must have 2 params");
        let param_names: Vec<&&str> = ROPE_STANDARD.params.iter().map(|(n, _)| n).collect();
        assert!(param_names.contains(&&"seq_len"));
        assert!(param_names.contains(&&"head_dim"));
    }

    // ── Test 35: ROPE_PARTIAL has three params ─────────────────────────

    #[test]
    fn rope_partial_params_count() {
        assert_eq!(ROPE_PARTIAL.params.len(), 3, "ROPE_PARTIAL must have 3 params");
    }
}
