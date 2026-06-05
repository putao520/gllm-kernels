//! Sampling template instances (SPEC 27 REQ-AT-010)

use crate::compiler::codegen::vm::algo_template::*;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Sampling Templates
//
// The sampling pipeline follows the architecture described in
// DOCS/architecture/gpu-ptx-codegen-reference.md:
//   Temperature→Top-K→Softmax→Top-P→Sample
//
// Templates express the high-level structure; detailed mask/threshold
// logic remains in the specialized emit functions.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static SAMPLING_ARGMAX: AlgoTemplate = AlgoTemplate {
    name: "SAMPLING_ARGMAX",
    strategy: AlgoStrategy::SamplingArgmax,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Argmax: HReduce(Max) over logits to find max value.
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "logits" },
            AlgoTraceStep::HReduce { src: "logits", op: ReduceKind::Max },
        ]),
        AlgoStep::Reduce { op: ReduceOp::Max },
    ],
    params: &[
        ("vocab_size", AlgoParam::FromGraph("vocab_size")),
    ],
    micro_kernel: None,
};

pub static SAMPLING_TEMPERATURE: AlgoTemplate = AlgoTemplate {
    name: "SAMPLING_TEMPERATURE",
    strategy: AlgoStrategy::SamplingTemperature,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Temperature scaling: logits / temperature
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "logits" },
            AlgoTraceStep::LoadInput { name: "temperature" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Div, dst: "scaled", a: "logits", b: "temperature" },
        ]),
    ],
    params: &[
        ("vocab_size", AlgoParam::FromGraph("vocab_size")),
    ],
    micro_kernel: None,
};

pub static SAMPLING_SOFTMAX: AlgoTemplate = AlgoTemplate {
    name: "SAMPLING_SOFTMAX",
    strategy: AlgoStrategy::SamplingSoftmax,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Softmax over logits: exp(logits) / sum(exp(logits))
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "logits" },
        ]),
        AlgoStep::Softmax,
    ],
    params: &[
        ("vocab_size", AlgoParam::FromGraph("vocab_size")),
    ],
    micro_kernel: None,
};

pub static SAMPLING_TOP_K: AlgoTemplate = AlgoTemplate {
    name: "SAMPLING_TOP_K",
    strategy: AlgoStrategy::SamplingTopK,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Top-K: find K-th largest value via HReduce(Max), then mask.
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "logits" },
            AlgoTraceStep::HReduce { src: "logits", op: ReduceKind::Max },
        ]),
        AlgoStep::Reduce { op: ReduceOp::Max },
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "threshold" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Sub, dst: "diff", a: "logits", b: "threshold" },
        ]),
    ],
    params: &[
        ("vocab_size", AlgoParam::FromGraph("vocab_size")),
        ("top_k", AlgoParam::FromGraph("top_k")),
    ],
    micro_kernel: None,
};

pub static SAMPLING_TOP_P: AlgoTemplate = AlgoTemplate {
    name: "SAMPLING_TOP_P",
    strategy: AlgoStrategy::SamplingTopP,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Top-P: softmax → cumulative distribution → threshold mask.
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "logits" },
        ]),
        AlgoStep::Softmax,
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "probs" },
            AlgoTraceStep::LoadInput { name: "top_p_threshold" },
            AlgoTraceStep::BinOp { op: TraceBinOp::Sub, dst: "remaining", a: "probs", b: "top_p_threshold" },
        ]),
    ],
    params: &[
        ("vocab_size", AlgoParam::FromGraph("vocab_size")),
    ],
    micro_kernel: None,
};

pub static SAMPLING_MULTINOMIAL: AlgoTemplate = AlgoTemplate {
    name: "SAMPLING_MULTINOMIAL",
    strategy: AlgoStrategy::SamplingMultinomial,
    device_req: DeviceReq::CpuAny,
    steps: &[
        // Multinomial: cumulative sum + random threshold comparison.
        AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "probs" },
            AlgoTraceStep::LoadInput { name: "random_val" },
            AlgoTraceStep::HReduce { src: "probs", op: ReduceKind::Sum },
        ]),
        AlgoStep::Reduce { op: ReduceOp::Sum },
    ],
    params: &[
        ("vocab_size", AlgoParam::FromGraph("vocab_size")),
    ],
    micro_kernel: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::vm::algo_template::{
        AlgoParam, AlgoStep, AlgoStrategy, AlgoTraceStep, DeviceReq, ReduceOp, StrategyFamily,
        TraceBinOp,
    };

    // ── Test 1: All 6 sampling templates have unique strategies ──────────

    #[test]
    fn all_sampling_templates_have_unique_strategies() {
        // Arrange
        let strategies: Vec<AlgoStrategy> = vec![
            SAMPLING_ARGMAX.strategy,
            SAMPLING_TEMPERATURE.strategy,
            SAMPLING_SOFTMAX.strategy,
            SAMPLING_TOP_K.strategy,
            SAMPLING_TOP_P.strategy,
            SAMPLING_MULTINOMIAL.strategy,
        ];

        // Act & Assert: all pairwise distinct
        for i in 0..strategies.len() {
            for j in (i + 1)..strategies.len() {
                assert_ne!(strategies[i], strategies[j],
                    "sampling strategies at indices {} and {} must differ", i, j);
            }
        }
        assert_eq!(strategies.len(), 6, "must have exactly 6 sampling templates");
    }

    // ── Test 2: All sampling templates belong to the Sampling family ─────

    #[test]
    fn all_sampling_templates_are_sampling_family() {
        // Arrange
        let templates: &[(&str, AlgoStrategy)] = &[
            ("SAMPLING_ARGMAX", SAMPLING_ARGMAX.strategy),
            ("SAMPLING_TEMPERATURE", SAMPLING_TEMPERATURE.strategy),
            ("SAMPLING_SOFTMAX", SAMPLING_SOFTMAX.strategy),
            ("SAMPLING_TOP_K", SAMPLING_TOP_K.strategy),
            ("SAMPLING_TOP_P", SAMPLING_TOP_P.strategy),
            ("SAMPLING_MULTINOMIAL", SAMPLING_MULTINOMIAL.strategy),
        ];

        // Act & Assert
        for (name, strategy) in templates {
            assert_eq!(strategy.family(), StrategyFamily::Sampling,
                "{} must belong to Sampling family", name);
        }
    }

    // ── Test 3: All sampling templates target CpuAny ─────────────────────

    #[test]
    fn all_sampling_templates_target_cpu_any() {
        // Arrange
        let templates: &[(&str, DeviceReq)] = &[
            ("SAMPLING_ARGMAX", SAMPLING_ARGMAX.device_req),
            ("SAMPLING_TEMPERATURE", SAMPLING_TEMPERATURE.device_req),
            ("SAMPLING_SOFTMAX", SAMPLING_SOFTMAX.device_req),
            ("SAMPLING_TOP_K", SAMPLING_TOP_K.device_req),
            ("SAMPLING_TOP_P", SAMPLING_TOP_P.device_req),
            ("SAMPLING_MULTINOMIAL", SAMPLING_MULTINOMIAL.device_req),
        ];

        // Act & Assert
        for (name, req) in templates {
            assert_eq!(*req, DeviceReq::CpuAny,
                "{} must target CpuAny", name);
        }
    }

    // ── Test 4: No sampling template has a micro_kernel ──────────────────

    #[test]
    fn no_sampling_template_has_micro_kernel() {
        // Arrange
        let templates: &[(&str, Option<_>)] = &[
            ("SAMPLING_ARGMAX", SAMPLING_ARGMAX.micro_kernel),
            ("SAMPLING_TEMPERATURE", SAMPLING_TEMPERATURE.micro_kernel),
            ("SAMPLING_SOFTMAX", SAMPLING_SOFTMAX.micro_kernel),
            ("SAMPLING_TOP_K", SAMPLING_TOP_K.micro_kernel),
            ("SAMPLING_TOP_P", SAMPLING_TOP_P.micro_kernel),
            ("SAMPLING_MULTINOMIAL", SAMPLING_MULTINOMIAL.micro_kernel),
        ];

        // Act & Assert
        for (name, mk) in templates {
            assert!(mk.is_none(), "{} must not have a micro_kernel", name);
        }
    }

    // ── Test 5: All sampling templates declare vocab_size param ──────────

    #[test]
    fn all_sampling_templates_have_vocab_size_param() {
        // Arrange
        let templates: &[(&str, &[(&str, AlgoParam)])] = &[
            ("SAMPLING_ARGMAX", SAMPLING_ARGMAX.params),
            ("SAMPLING_TEMPERATURE", SAMPLING_TEMPERATURE.params),
            ("SAMPLING_SOFTMAX", SAMPLING_SOFTMAX.params),
            ("SAMPLING_TOP_K", SAMPLING_TOP_K.params),
            ("SAMPLING_TOP_P", SAMPLING_TOP_P.params),
            ("SAMPLING_MULTINOMIAL", SAMPLING_MULTINOMIAL.params),
        ];

        // Act & Assert
        for (name, params) in templates {
            let vocab_entry = params.iter().find(|(pname, _)| *pname == "vocab_size");
            assert!(vocab_entry.is_some(), "{} must have vocab_size param", name);

            let (_, param) = vocab_entry.unwrap();
            assert!(matches!(param, AlgoParam::FromGraph("vocab_size")),
                "{} vocab_size must be FromGraph(\"vocab_size\")", name);
        }
    }

    // ── Test 6: SAMPLING_TOP_K has exactly 2 params (vocab_size + top_k) ─

    #[test]
    fn sampling_top_k_has_two_params() {
        // Arrange
        let params = SAMPLING_TOP_K.params;

        // Act
        let param_names: Vec<&&str> = params.iter().map(|(n, _)| n).collect();

        // Assert
        assert_eq!(params.len(), 2, "SAMPLING_TOP_K must have 2 params");
        assert!(param_names.contains(&&"vocab_size"), "must have vocab_size");
        assert!(param_names.contains(&&"top_k"), "must have top_k");

        // top_k must come from graph
        let top_k_param = params.iter().find(|(n, _)| *n == "top_k").unwrap().1;
        assert!(matches!(top_k_param, AlgoParam::FromGraph("top_k")),
            "top_k must be FromGraph(\"top_k\")");
    }

    // ── Test 7: SAMPLING_ARGMAX steps contain TraceBody + Reduce(Max) ────

    #[test]
    fn sampling_argmax_steps_structure() {
        // Arrange
        let steps = SAMPLING_ARGMAX.steps;

        // Assert: exactly 2 steps
        assert_eq!(steps.len(), 2, "ARGMAX must have 2 steps");

        // Step 0: TraceBody with LoadInput + HReduce(Max)
        assert!(matches!(&steps[0], AlgoStep::TraceBody(body) if body.len() == 2),
            "first step must be TraceBody with 2 trace steps");

        // Step 1: Reduce(Max)
        assert!(matches!(&steps[1], AlgoStep::Reduce { op } if matches!(op, ReduceOp::Max)),
            "second step must be Reduce(Max)");
    }

    // ── Test 8: SAMPLING_TEMPERATURE steps contain BinOp(Div) ────────────

    #[test]
    fn sampling_temperature_steps_have_div_binop() {
        // Arrange
        let steps = SAMPLING_TEMPERATURE.steps;

        // Assert: exactly 1 step (TraceBody)
        assert_eq!(steps.len(), 1, "TEMPERATURE must have 1 step");

        if let AlgoStep::TraceBody(body) = &steps[0] {
            assert_eq!(body.len(), 3, "TraceBody must have 3 trace steps");

            // Step 0: LoadInput("logits")
            assert!(matches!(&body[0], AlgoTraceStep::LoadInput { name } if *name == "logits"),
                "first trace step must load logits");

            // Step 1: LoadInput("temperature")
            assert!(matches!(&body[1], AlgoTraceStep::LoadInput { name } if *name == "temperature"),
                "second trace step must load temperature");

            // Step 2: BinOp(Div, "scaled", "logits", "temperature")
            assert!(matches!(&body[2],
                AlgoTraceStep::BinOp {
                    op: TraceBinOp::Div,
                    dst, a, b,
                } if *dst == "scaled" && *a == "logits" && *b == "temperature"),
                "third trace step must be Div producing 'scaled'");
        } else {
            panic!("first step must be TraceBody");
        }
    }

    // ── Test 9: SAMPLING_SOFTMAX has TraceBody + Softmax steps ───────────

    #[test]
    fn sampling_softmax_has_softmax_step() {
        // Arrange
        let steps = SAMPLING_SOFTMAX.steps;

        // Assert: 2 steps — TraceBody(LoadInput) + Softmax
        assert_eq!(steps.len(), 2, "SOFTMAX must have 2 steps");

        // Step 0: TraceBody loading logits
        if let AlgoStep::TraceBody(body) = &steps[0] {
            assert_eq!(body.len(), 1, "TraceBody must have 1 trace step");
            assert!(matches!(&body[0], AlgoTraceStep::LoadInput { name } if *name == "logits"),
                "must load logits");
        } else {
            panic!("step 0 must be TraceBody");
        }

        // Step 1: Softmax
        assert!(matches!(&steps[1], AlgoStep::Softmax),
            "step 1 must be Softmax");
    }

    // ── Test 10: SAMPLING_MULTINOMIAL steps structure ────────────────────

    #[test]
    fn sampling_multinomial_steps_structure() {
        let steps = SAMPLING_MULTINOMIAL.steps;
        assert_eq!(steps.len(), 2, "MULTINOMIAL must have 2 steps");

        // Step 0: TraceBody(LoadInput + LoadInput + HReduce(Sum))
        if let AlgoStep::TraceBody(body) = &steps[0] {
            assert_eq!(body.len(), 3, "TraceBody must have 3 trace steps");
            assert!(matches!(&body[0], AlgoTraceStep::LoadInput { name } if *name == "probs"));
            assert!(matches!(&body[1], AlgoTraceStep::LoadInput { name } if *name == "random_val"));
            assert!(matches!(&body[2], AlgoTraceStep::HReduce { op: ReduceKind::Sum, .. }));
        } else {
            panic!("step 0 must be TraceBody");
        }

        // Step 1: Reduce(Sum)
        assert!(matches!(&steps[1], AlgoStep::Reduce { op } if matches!(op, ReduceOp::Sum)),
            "step 1 must be Reduce(Sum)");
    }

    // ── Test 11: SAMPLING_TOP_K steps contain 3 steps ────────────────────

    #[test]
    fn sampling_top_k_three_steps() {
        let steps = SAMPLING_TOP_K.steps;
        assert_eq!(steps.len(), 3, "TOP_K must have 3 steps");

        // Step 0: TraceBody(LoadInput + HReduce)
        assert!(matches!(&steps[0], AlgoStep::TraceBody(body) if body.len() == 2));

        // Step 1: Reduce(Max)
        assert!(matches!(&steps[1], AlgoStep::Reduce { op: ReduceOp::Max }));

        // Step 2: TraceBody(LoadInput + BinOp(Sub))
        if let AlgoStep::TraceBody(body) = &steps[2] {
            assert_eq!(body.len(), 2, "final TraceBody must have 2 steps");
            assert!(matches!(&body[0], AlgoTraceStep::LoadInput { name } if *name == "threshold"));
            assert!(matches!(&body[1],
                AlgoTraceStep::BinOp { op: TraceBinOp::Sub, dst, a, b, .. }
                if *dst == "diff" && *a == "logits" && *b == "threshold"));
        } else {
            panic!("step 2 must be TraceBody");
        }
    }

    // ── Test 12: All sampling template names are well-formed ─────────────

    #[test]
    fn all_sampling_template_names_start_with_sampling() {
        for (name, tmpl) in [
            ("SAMPLING_ARGMAX", &SAMPLING_ARGMAX),
            ("SAMPLING_TEMPERATURE", &SAMPLING_TEMPERATURE),
            ("SAMPLING_SOFTMAX", &SAMPLING_SOFTMAX),
            ("SAMPLING_TOP_K", &SAMPLING_TOP_K),
            ("SAMPLING_TOP_P", &SAMPLING_TOP_P),
            ("SAMPLING_MULTINOMIAL", &SAMPLING_MULTINOMIAL),
        ] {
            assert_eq!(tmpl.name, name, "template name mismatch");
            assert!(name.starts_with("SAMPLING_"), "{} must start with SAMPLING_", name);
        }
    }

    // ── Test 13: SAMPLING_ARGMAX TraceBody contains HReduce(Max) ─────────

    #[test]
    fn sampling_argmax_trace_body_has_hreduce_max() {
        if let AlgoStep::TraceBody(body) = &SAMPLING_ARGMAX.steps[0] {
            assert!(matches!(&body[1], AlgoTraceStep::HReduce { src, op }
                if *src == "logits" && matches!(op, ReduceKind::Max)),
                "second trace step must be HReduce(Max) on logits");
        }
    }

    // ── Test 14: Non-TOP-K templates have exactly 1 param ────────────────

    #[test]
    fn non_top_k_templates_have_single_vocab_size_param() {
        for (name, tmpl) in [
            ("SAMPLING_ARGMAX", &SAMPLING_ARGMAX),
            ("SAMPLING_TEMPERATURE", &SAMPLING_TEMPERATURE),
            ("SAMPLING_SOFTMAX", &SAMPLING_SOFTMAX),
            ("SAMPLING_TOP_P", &SAMPLING_TOP_P),
            ("SAMPLING_MULTINOMIAL", &SAMPLING_MULTINOMIAL),
        ] {
            assert_eq!(tmpl.params.len(), 1, "{} must have exactly 1 param", name);
        }
    }

    // ── Test 10: SAMPLING_TOP_P has 3 steps with Softmax in the middle ───

    #[test]
    fn sampling_top_p_three_steps_with_middle_softmax() {
        // Arrange
        let steps = SAMPLING_TOP_P.steps;

        // Assert: 3 steps — TraceBody, Softmax, TraceBody
        assert_eq!(steps.len(), 3, "TOP_P must have 3 steps");

        // Step 0: TraceBody loading logits
        assert!(matches!(&steps[0], AlgoStep::TraceBody(_)),
            "step 0 must be TraceBody");

        // Step 1: Softmax
        assert!(matches!(&steps[1], AlgoStep::Softmax),
            "step 1 must be Softmax");

        // Step 2: TraceBody with BinOp(Sub) for threshold comparison
        if let AlgoStep::TraceBody(body) = &steps[2] {
            assert_eq!(body.len(), 3, "final TraceBody must have 3 steps");

            // First two: LoadInput("probs") and LoadInput("top_p_threshold")
            assert!(matches!(&body[0], AlgoTraceStep::LoadInput { name } if *name == "probs"),
                "must load probs");
            assert!(matches!(&body[1], AlgoTraceStep::LoadInput { name } if *name == "top_p_threshold"),
                "must load top_p_threshold");

            // Third: BinOp(Sub) producing "remaining"
            assert!(matches!(&body[2],
                AlgoTraceStep::BinOp {
                    op: TraceBinOp::Sub,
                    dst, a, b,
                } if *dst == "remaining" && *a == "probs" && *b == "top_p_threshold"),
                "last trace step must be Sub producing 'remaining'");
        } else {
            panic!("step 2 must be TraceBody");
        }
    }

    // ── Test 15: SAMPLING_ARGMAX loads logits input ──────────────────────

    #[test]
    fn sampling_argmax_loads_logits_input() {
        if let AlgoStep::TraceBody(body) = &SAMPLING_ARGMAX.steps[0] {
            assert!(matches!(&body[0], AlgoTraceStep::LoadInput { name } if *name == "logits"),
                "first trace step must load logits");
        }
    }

    // ── Test 16: SAMPLING_TEMPERATURE has exactly 1 param ────────────────

    #[test]
    fn sampling_temperature_single_param_is_vocab_size() {
        assert_eq!(SAMPLING_TEMPERATURE.params.len(), 1);
        let (name, param) = &SAMPLING_TEMPERATURE.params[0];
        assert_eq!(*name, "vocab_size");
        assert!(matches!(param, AlgoParam::FromGraph("vocab_size")));
    }

    // ── Test 17: SAMPLING_SOFTMAX has Softmax as second step ─────────────

    #[test]
    fn sampling_softmax_second_step_is_softmax_variant() {
        assert!(matches!(&SAMPLING_SOFTMAX.steps[1], AlgoStep::Softmax));
    }

    // ── Test 18: SAMPLING_MULTINOMIAL loads probs and random_val ─────────

    #[test]
    fn sampling_multinomial_loads_correct_inputs() {
        if let AlgoStep::TraceBody(body) = &SAMPLING_MULTINOMIAL.steps[0] {
            let names: Vec<&str> = body.iter().filter_map(|t| match t {
                AlgoTraceStep::LoadInput { name } => Some(*name),
                _ => None,
            }).collect();
            assert!(names.contains(&"probs"), "must load probs");
            assert!(names.contains(&"random_val"), "must load random_val");
        }
    }

    // ── Test 19: SAMPLING_TOP_K first TraceBody has HReduce(Max) ─────────

    #[test]
    fn sampling_top_k_first_trace_body_has_hreduce_max() {
        if let AlgoStep::TraceBody(body) = &SAMPLING_TOP_K.steps[0] {
            assert!(body.iter().any(|t| matches!(t,
                AlgoTraceStep::HReduce { src, op: ReduceKind::Max } if *src == "logits")),
                "first TraceBody must have HReduce(Max) on logits");
        }
    }

    // ── Test 20: All templates have at least one step ────────────────────

    #[test]
    fn all_sampling_templates_have_at_least_one_step() {
        for (name, tmpl) in [
            ("SAMPLING_ARGMAX", &SAMPLING_ARGMAX),
            ("SAMPLING_TEMPERATURE", &SAMPLING_TEMPERATURE),
            ("SAMPLING_SOFTMAX", &SAMPLING_SOFTMAX),
            ("SAMPLING_TOP_K", &SAMPLING_TOP_K),
            ("SAMPLING_TOP_P", &SAMPLING_TOP_P),
            ("SAMPLING_MULTINOMIAL", &SAMPLING_MULTINOMIAL),
        ] {
            assert!(!tmpl.steps.is_empty(), "{} must have at least one step", name);
        }
    }

    // ── Test 21: SAMPLING_TEMPERATURE first step is TraceBody ────────────

    #[test]
    fn sampling_temperature_first_step_is_trace_body() {
        assert!(matches!(&SAMPLING_TEMPERATURE.steps[0], AlgoStep::TraceBody(_)));
    }

    // ── Test 22: SAMPLING_TOP_P loads logits in first step ───────────────

    #[test]
    fn sampling_top_p_first_step_loads_logits() {
        if let AlgoStep::TraceBody(body) = &SAMPLING_TOP_P.steps[0] {
            assert!(matches!(&body[0], AlgoTraceStep::LoadInput { name } if *name == "logits"),
                "first trace step must load logits");
        }
    }

    // ── Test 23: SAMPLING_TOP_K second TraceBody has Sub BinOp ───────────

    #[test]
    fn sampling_top_k_second_trace_body_has_sub() {
        if let AlgoStep::TraceBody(body) = &SAMPLING_TOP_K.steps[2] {
            assert!(body.iter().any(|t| matches!(t,
                AlgoTraceStep::BinOp { op: TraceBinOp::Sub, .. })),
                "second TraceBody must have a Sub BinOp");
        }
    }
}
