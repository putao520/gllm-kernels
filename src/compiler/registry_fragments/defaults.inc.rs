impl ScalarOpRegistry {
    /// Create a registry pre-populated with all scalar functions and their OpTraces.
    // @trace REQ-AIS-007 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/scalar-registry-defaults]
    pub fn with_defaults() -> Self {
        use crate::scalar_ops::activations::*;
        use crate::scalar_ops::argmax::*;
        use crate::scalar_ops::blas::*;
        use crate::scalar_ops::norms::*;
        use crate::scalar_ops::rope::*;

        let mut reg = Self::new();

        // ── SiLU ──
        let silu_sig = ScalarFnSignature {
            fn_ptr: scalar_silu as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Silu,
            silu_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),   // [0] v
                        TraceOp::Neg(ValueId(0)),     // [1] -v
                        TraceOp::Exp(ValueId(1)),     // [2] exp(-v)
                        TraceOp::Const(1.0), // [3] 1.0
                        TraceOp::Add(ValueId(2), ValueId(3)),  // [4] 1 + exp(-v)
                        TraceOp::Div(ValueId(0), ValueId(4)),  // [5] v / (1 + exp(-v))
                    ],
                },
                signature: silu_sig,
            },
        );

        // ── GELU ──
        // SymExec auto-extraction verified: 14-op trace with correct Tanh(9)
        // covering the tanh-approximation GELU formula.
        let gelu_sig = ScalarFnSignature {
            fn_ptr: scalar_gelu as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Gelu,
            gelu_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),
                        TraceOp::Const(0.5),
                        TraceOp::Mul(ValueId(0), ValueId(1)),
                        TraceOp::Const(0.044715),
                        TraceOp::Mul(ValueId(0), ValueId(3)),
                        TraceOp::Mul(ValueId(4), ValueId(0)),
                        TraceOp::Mul(ValueId(5), ValueId(0)),
                        TraceOp::Add(ValueId(6), ValueId(0)),
                        TraceOp::Const(0.7978845608),
                        TraceOp::Mul(ValueId(7), ValueId(8)),
                        TraceOp::Tanh(ValueId(9)),
                        TraceOp::Const(1.0),
                        TraceOp::Add(ValueId(10), ValueId(11)),
                        TraceOp::Mul(ValueId(2), ValueId(12)),
                    ],
                },
                signature: gelu_sig,
            },
        );

        // ── Tanh ──
        let tanh_sig = ScalarFnSignature {
            fn_ptr: scalar_tanh as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Tanh,
            tanh_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0), // [0] x
                        TraceOp::Tanh(ValueId(0)),  // [1] tanh(x)
                    ],
                },
                signature: tanh_sig,
            },
        );

        // ── Sigmoid: 1 / (1 + exp(-x)) ──
        let sigmoid_sig = ScalarFnSignature {
            fn_ptr: scalar_tanh as *const u8, // placeholder fn_ptr; trace drives codegen
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Sigmoid,
            sigmoid_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),               // [0] x
                        TraceOp::Sigmoid(ValueId(0)),     // [1] sigmoid(x) = 1/(1+exp(-x))
                    ],
                },
                signature: sigmoid_sig,
            },
        );

        // ── SwiGLU ──
        let swiglu_sig = ScalarFnSignature {
            fn_ptr: scalar_swiglu as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::SwiGlu,
            swiglu_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),   // [0] gate
                        TraceOp::Input(1),   // [1] up
                        TraceOp::Neg(ValueId(0)),     // [2] -gate
                        TraceOp::Exp(ValueId(2)),     // [3] exp(-gate)
                        TraceOp::Const(1.0), // [4] 1.0
                        TraceOp::Add(ValueId(3), ValueId(4)),  // [5] 1 + exp(-gate)
                        TraceOp::Div(ValueId(0), ValueId(5)),  // [6] silu(gate)
                        TraceOp::Mul(ValueId(6), ValueId(1)),  // [7] silu(gate) * up
                    ],
                },
                signature: swiglu_sig,
            },
        );

        // ── Clipped SwiGLU (OpenAI gpt-oss-20b) ──
        //
        // The registry template trace bakes the canonical limit=7.0 clamp
        // constants into the body. At lower time,
        // `plan_lower::extract_op_trace` rewrites the two `Const(±limit)`
        // slots to match the `Op::SwiGluClipped { limit:  }` carried by
        // each individual op, so different layers / models can share the
        // same `OpKindKey` while producing distinct clamp thresholds.
        //
        // SSA layout (12 ops):
        //   [0] gate_raw   [1] up_raw
        //   [2] +limit     [3] -limit
        //   [4] min(gate_raw, +limit)  → upper-clamped gate
        //   [5] max([4], -limit)       → fully-clamped gate
        //   [6] min(up_raw, +limit)    → upper-clamped up
        //   [7] max([6], -limit)       → fully-clamped up
        //   [8] -gate'
        //   [9] exp(-gate')
        //   [10] 1.0
        //   [11] 1 + exp(-gate')
        //   [12] silu(gate') = gate' / (1+exp(-gate'))
        //   [13] silu(gate') * up'
        let swiglu_clipped_sig = ScalarFnSignature {
            fn_ptr: scalar_swiglu_clipped as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(7.0), // canonical template limit
            ],
        };
        // Template trace with canonical limit=7.0 — extract_op_trace
        // rewrites Const(±limit) per op at codegen time.
        reg.register(OpKindKey::SwiGluClipped, swiglu_clipped_sig.clone());
        reg.inject_trace(
            OpKindKey::SwiGluClipped,
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),     // [0] gate_raw
                        TraceOp::Input(1),     // [1] up_raw
                        TraceOp::Const(7.0),   // [2] +limit
                        TraceOp::Const(-7.0),  // [3] -limit
                        TraceOp::Min(ValueId(0), ValueId(2)),    // [4] min(gate_raw, +limit)
                        TraceOp::Max(ValueId(4), ValueId(3)),    // [5] clamp(gate_raw, ±limit)
                        TraceOp::Min(ValueId(1), ValueId(2)),    // [6] min(up_raw, +limit)
                        TraceOp::Max(ValueId(6), ValueId(3)),    // [7] clamp(up_raw, ±limit)
                        TraceOp::Neg(ValueId(5)),       // [8] -gate'
                        TraceOp::Exp(ValueId(8)),       // [9] exp(-gate')
                        TraceOp::Const(1.0),   // [10] 1.0
                        TraceOp::Add(ValueId(9), ValueId(10)),   // [11] 1 + exp(-gate')
                        TraceOp::Div(ValueId(5), ValueId(11)),   // [12] silu(gate')
                        TraceOp::Mul(ValueId(12), ValueId(7)),   // [13] silu(gate') * up'
                    ],
                },
                signature: swiglu_clipped_sig,
            },
        );

        // ── GeGLU ──
        let geglu_sig = ScalarFnSignature {
            fn_ptr: scalar_geglu as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::GeGlu,
            geglu_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),            // [0] gate
                        TraceOp::Input(1),            // [1] up
                        TraceOp::Mul(ValueId(0), ValueId(0)),           // [2] gate^2
                        TraceOp::Mul(ValueId(2), ValueId(0)),           // [3] gate^3
                        TraceOp::Const(0.044715),     // [4] coeff
                        TraceOp::Mul(ValueId(4), ValueId(3)),           // [5] coeff * gate^3
                        TraceOp::Add(ValueId(0), ValueId(5)),           // [6] gate + coeff*gate^3
                        TraceOp::Const(0.7978845608), // [7] sqrt(2/pi)
                        TraceOp::Mul(ValueId(7), ValueId(6)),           // [8] sqrt(2/pi) * (...)
                        TraceOp::Tanh(ValueId(8)),             // [9] tanh(...)
                        TraceOp::Const(1.0),          // [10] 1.0
                        TraceOp::Add(ValueId(10), ValueId(9)),          // [11] 1 + tanh(...)
                        TraceOp::Const(0.5),          // [12] 0.5
                        TraceOp::Mul(ValueId(12), ValueId(0)),          // [13] 0.5 * gate
                        TraceOp::Mul(ValueId(13), ValueId(11)),         // [14] gelu(gate)
                        TraceOp::Mul(ValueId(14), ValueId(1)),          // [15] gelu(gate) * up
                    ],
                },
                signature: geglu_sig,
            },
        );

        // ── Add ──
        let add_sig = ScalarFnSignature {
            fn_ptr: scalar_vec_add as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Add,
            add_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // [0] a
                        TraceOp::Input(1),  // [1] b
                        TraceOp::Add(ValueId(0), ValueId(1)), // [2] a + b
                    ],
                },
                signature: add_sig,
            },
        );

        // ── Mul ──
        let mul_sig = ScalarFnSignature {
            fn_ptr: scalar_vec_mul as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Mul,
            mul_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // [0] a
                        TraceOp::Input(1),  // [1] b
                        TraceOp::Mul(ValueId(0), ValueId(1)), // [2] a * b
                    ],
                },
                signature: mul_sig,
            },
        );

        // ── Residual (same as Add) ──
        let residual_sig = ScalarFnSignature {
            fn_ptr: scalar_vec_add as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Residual,
            residual_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] residual
                        TraceOp::Add(ValueId(0), ValueId(1)), // [2] x + residual
                    ],
                },
                signature: residual_sig,
            },
        );

        // ── LogitSoftcap: cap * tanh(x / cap) ──
        // cap is a compile-time constant from Op::LogitSoftcap { cap:  }.
        // The trace uses placeholder constants; `try_auto_dispatch_elementwise`
        // rewrites them via parameterized trace rewrite (like SwiGluClipped).
        // Trace: x / cap → tanh → * cap
        {
            // No dedicated scalar_fn — cap is compile-time. Use a dummy fn_ptr.
            // The trace is the authoritative definition.
            let softcap_sig = ScalarFnSignature {
                fn_ptr: scalar_tanh as *const u8, // closest scalar op
                params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
            };
            let cap_placeholder_f32 = 1.0f32;
            let cap_placeholder = 1.0f64;
            let inv_cap_placeholder = 1.0f64;
            reg.register_with_symexec_fallback(
                OpKindKey::LogitSoftcap,
                softcap_sig.clone(),
                OpTrace {
                    pattern: ComputePattern::Elementwise {
                        body: vec![
                            TraceOp::Input(0),                   // [0] x
                            TraceOp::Const(inv_cap_placeholder), // [1] 1/cap (placeholder)
                            TraceOp::Mul(ValueId(0), ValueId(1)),                  // [2] x * (1/cap)
                            TraceOp::Tanh(ValueId(2)),                    // [3] tanh(x/cap)
                            TraceOp::Const(cap_placeholder),     // [4] cap (placeholder)
                            TraceOp::Mul(ValueId(3), ValueId(4)),                  // [5] cap * tanh(x/cap)
                        ],
                    },
                    signature: softcap_sig,
                },
            );
        }

        // ── Softmax ──
        // SymExec auto-extraction verified: 3-pass Reduction (max → exp-sum → normalize).
        // Select pattern detection (W2.2) enables Max reduction from conditional branches.
        let softmax_sig = ScalarFnSignature {
            fn_ptr: scalar_softmax as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Softmax,
            softmax_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Reduction {
                    identity: f64::NEG_INFINITY,
                    combine: vec![
                        TraceOp::Input(0),  // [0] a (running max)
                        TraceOp::Input(1),  // [1] b (new element)
                        TraceOp::Max(ValueId(0), ValueId(1)), // [2] max(a, b)
                    ],
                    second_pass: Some(Box::new(ReductionSecondPass {
                        identity: 0.0,
                        element_transform: vec![
                            TraceOp::Input(0),  // [0] x (current element)
                            TraceOp::Input(1),  // [1] max (broadcast)
                            TraceOp::Sub(ValueId(0), ValueId(1)), // [2] x - max
                            TraceOp::Exp(ValueId(2)),    // [3] exp(x - max)
                        ],
                        combine: vec![
                            TraceOp::Input(0),  // [0] acc (running sum)
                            TraceOp::Input(1),  // [1] exp_val
                            TraceOp::Add(ValueId(0), ValueId(1)), // [2] acc + exp_val
                        ],
                    })),
                    normalize: Some(vec![
                        TraceOp::Input(0),  // [0] exp_val
                        TraceOp::Input(1),  // [1] inv_sum (broadcast)
                        TraceOp::Mul(ValueId(0), ValueId(1)), // [2] exp_val * inv_sum
                    ]),
                },
                signature: softmax_sig,
            },
        );

        // ── RmsNorm ──
        let rms_sig = ScalarFnSignature {
            fn_ptr: scalar_rms_norm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        };
        let default_eps: f32 = 1e-5;
        reg.register_with_symexec_fallback(
            OpKindKey::RmsNorm,
            rms_sig.clone(),
            OpTrace {
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Mul(ValueId(0), ValueId(0)), // [1] x^2
                    ],
                    finalize: vec![
                        TraceOp::Input(0),    // [0] sum_sq
                        TraceOp::Input(1),    // [1] n (as float)
                        TraceOp::Div(ValueId(0), ValueId(1)),   // [2] mean = sum_sq / n
                        TraceOp::Const(1e-5), // [3] default eps; codegen uses actual value from Op::RmsNorm(NormSpec { feature_dim: feature_dim, eps: , dtype: DType::F32, has_weight: true })
                        TraceOp::Add(ValueId(2), ValueId(3)),   // [4] mean + eps
                        TraceOp::Rsqrt(ValueId(4)),    // [5] rsqrt(mean + eps)
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] scale (from finalize)
                        TraceOp::Input(2),  // [2] weight
                        TraceOp::Mul(ValueId(0), ValueId(1)), // [3] x * scale
                        TraceOp::Mul(ValueId(3), ValueId(2)), // [4] x * scale * weight
                    ],
                },
                signature: rms_sig,
            },
        );

        // ── HeadRmsNorm (Qwen3 q_norm/k_norm: head-wise RMSNorm with weight) ──
        // 数学等价于 standard RmsNorm,但 feature_dim 是 head_dim (不是 input 最后一维)。
        // pattern 复用 RmsNorm (mean-based + weight × transform)。
        // lower 阶段按 head_dim 决定循环结构。
        let head_rms_sig = ScalarFnSignature {
            fn_ptr: scalar_rms_norm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::HeadRmsNorm,
            head_rms_sig.clone(),
            OpTrace {
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0),
                        TraceOp::Mul(ValueId(0), ValueId(0)),
                    ],
                    finalize: vec![
                        TraceOp::Input(0),
                        TraceOp::Input(1),
                        TraceOp::Div(ValueId(0), ValueId(1)),
                        TraceOp::Const(1e-5),
                        TraceOp::Add(ValueId(2), ValueId(3)),
                        TraceOp::Rsqrt(ValueId(4)),
                    ],
                    transform: vec![
                        TraceOp::Input(0),
                        TraceOp::Input(1),
                        TraceOp::Input(2),
                        TraceOp::Mul(ValueId(0), ValueId(1)),
                        TraceOp::Mul(ValueId(3), ValueId(2)),
                    ],
                },
                signature: head_rms_sig,
            },
        );

        // ── ValueNorm (RmsNorm without weight) ──
        let vnorm_sig = ScalarFnSignature {
            fn_ptr: scalar_value_norm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::ValueNorm,
            vnorm_sig.clone(),
            OpTrace {
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Mul(ValueId(0), ValueId(0)), // [1] x^2
                    ],
                    finalize: vec![
                        TraceOp::Input(0),    // [0] sum_sq
                        TraceOp::Input(1),    // [1] n (as float)
                        TraceOp::Div(ValueId(0), ValueId(1)),   // [2] mean = sum_sq / n
                        TraceOp::Const(1e-5), // [3] default eps; codegen uses actual value from Op::ValueNorm(NormSpec { feature_dim: feature_dim, eps: , dtype: DType::F32, has_weight: false })
                        TraceOp::Add(ValueId(2), ValueId(3)),   // [4] mean + eps
                        TraceOp::Rsqrt(ValueId(4)),    // [5] rsqrt(mean + eps)
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] scale (from finalize)
                        TraceOp::Mul(ValueId(0), ValueId(1)), // [2] x * scale — NO weight multiplication
                    ],
                },
                signature: vnorm_sig,
            },
        );

        // ── LayerNorm ──
        let ln_sig = ScalarFnSignature {
            fn_ptr: scalar_layer_norm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::LayerNorm,
            ln_sig.clone(),
            OpTrace {
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0), // [0] x  (used for both mean and variance)
                    ],
                    finalize: vec![
                        TraceOp::Input(0),    // [0] mean
                        TraceOp::Input(1),    // [1] var
                        TraceOp::Const(1e-5), // [2] default eps; codegen uses actual value from Op::LayerNorm(NormSpec { feature_dim: feature_dim, eps: , dtype: DType::F32, has_weight: true })
                        TraceOp::Add(ValueId(1), ValueId(2)),   // [3] var + eps
                        TraceOp::Rsqrt(ValueId(3)),    // [4] rsqrt(var + eps)
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] mean
                        TraceOp::Input(2),  // [2] scale (from finalize)
                        TraceOp::Input(3),  // [3] weight
                        TraceOp::Input(4),  // [4] bias
                        TraceOp::Sub(ValueId(0), ValueId(1)), // [5] x - mean
                        TraceOp::Mul(ValueId(5), ValueId(2)), // [6] (x - mean) * scale
                        TraceOp::Mul(ValueId(6), ValueId(3)), // [7] normed * weight
                        TraceOp::Add(ValueId(7), ValueId(4)), // [8] normed * weight + bias
                    ],
                },
                signature: ln_sig,
            },
        );

        // ── GEMM ──
        let gemm_sig = ScalarFnSignature {
            fn_ptr: scalar_gemm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Gemm,
            gemm_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: gemm_sig,
            },
        );

        // ── RoPE ──
        let rope_sig = ScalarFnSignature {
            fn_ptr: scalar_rope as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1.0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::RoPE,
            rope_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0),  // [0] x0
                        TraceOp::Input(1),  // [1] x1
                        TraceOp::Input(2),  // [2] cos
                        TraceOp::Input(3),  // [3] sin
                        TraceOp::Mul(ValueId(0), ValueId(2)), // [4] x0 * cos
                        TraceOp::Mul(ValueId(1), ValueId(3)), // [5] x1 * sin
                        TraceOp::Sub(ValueId(4), ValueId(5)), // [6] x0*cos - x1*sin  (out0)
                        TraceOp::Mul(ValueId(1), ValueId(2)), // [7] x1 * cos
                        TraceOp::Mul(ValueId(0), ValueId(3)), // [8] x0 * sin
                        TraceOp::Add(ValueId(7), ValueId(8)), // [9] x1*cos + x0*sin  (out1)
                    ],
                    num_inputs: 4,
                    num_outputs: 2,
                },
                signature: rope_sig,
            },
        );

        // --- GemmBias: same as Gemm but with bias add epilogue ---
        let gemm_bias_sig = ScalarFnSignature {
            fn_ptr: scalar_gemm_bias as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr, // bias
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::GemmBias,
            gemm_bias_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: gemm_bias_sig,
            },
        );

        // --- MultiHeadAttention: fused QKV attention ---
        let mha_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::attention::scalar_multi_head_attention as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // Q
                ScalarParam::InputPtr,  // K
                ScalarParam::InputPtr,  // V
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // num_heads
                ScalarParam::Dim(2),    // head_dim
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::MultiHeadAttention,
            mha_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: mha_sig,
            },
        );

        // --- Transpose: layout transform, no compute ---
        let transpose_sig = ScalarFnSignature {
            fn_ptr: scalar_transpose_2d as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Transpose,
            transpose_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: transpose_sig,
            },
        );

        // --- Reshape: layout transform, no compute ---
        let reshape_sig = ScalarFnSignature {
            fn_ptr: scalar_reshape as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Reshape,
            reshape_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: reshape_sig,
            },
        );

        // --- QuantGemm: quantized GEMM with on-the-fly dequantization ---
        let quant_gemm_sig = ScalarFnSignature {
            fn_ptr: scalar_quant_gemm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr, // scales
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
                ScalarParam::Dim(0), // block_size
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::QuantGemm,
            quant_gemm_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: quant_gemm_sig,
            },
        );

        // --- Dequantize: JIT QuantGemm 内部 epilogue 反量化 ---
        // JIT QuantGemm 在微核寄存器级完成反量化: (qw - zp) × scale (epilogue)。
        // 不再注册独立 scalar 反量化函数 (REQ-QCG9: NO_SCALAR)。
        // Registry 保留 OpKindKey::Dequantize 条目供 trace 查询，fn_ptr 指向
        // dequant_mxfp4 作为有效的 text-segment 参考地址（不会被调用）。
        let dequant_sig = ScalarFnSignature {
            fn_ptr: crate::quant_mxfp4::dequant_mxfp4 as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::Dequantize,
            dequant_sig.clone(),
            OpTrace {
                pattern: ComputePattern::QuantDecode {
                    block_size: 32,
                    decode: vec![
                        TraceOp::Input(0),
                        TraceOp::Input(1),
                        TraceOp::Mul(ValueId(0), ValueId(1)),
                    ],
                },
                signature: dequant_sig,
            },
        );

        // --- DequantizeMxfp4: OCP Microscaling FP4 (e2m1 LUT + e8m0 scale) ---
        //
        // Block layout (one OCP block, block_size=32 elements):
        //   blocks[0..16]  : 16 packed bytes; low nibble = even index, high nibble = odd index
        //   scales[0]      : 1 e8m0 byte → power-of-2 scale (2^(byte-127))
        //
        // Trace shape: Permute(nibble, e2m1_lut) → BlockScale(scale, block_size=32) → Mul.
        // The `Permute` TraceOp captures the 16-entry e2m1 LUT lookup; downstream codegen
        // can lower this to vpshufb (x86) / tbl (ARM) / prmt (PTX). The trace is intentionally
        // schematic — ISA Lowering reads the actual LUT from a constant pool emitted
        // during codegen, not from this trace.
        //
        // SymExec ground truth: `crate::quant_mxfp4::dequant_mxfp4_scalar` (function pointer
        // registered below). Lower passes that consume the OpTrace can call the scalar fn
        // directly to validate JIT-emitted code numerically.
        // fn_ptr uses `dequant_mxfp4` (runtime ISA dispatcher) as text-segment reference —
        // the pointer is a marker for binary analysis, never called through registry.
        let mxfp4_sig = ScalarFnSignature {
            fn_ptr: crate::quant_mxfp4::dequant_mxfp4 as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // blocks (packed e2m1 nibbles)
                ScalarParam::WeightPtr,  // scales (one e8m0 byte per block)
                ScalarParam::OutputPtr,  // dequantized f32 output
                ScalarParam::Dim(0),     // num_blocks
                ScalarParam::Dim(1),     // block_size (typically 32)
            ],
        };
        // Inject manual trace directly (no symexec fallback): the LUT lookup cannot be
        // discovered by linear symbolic execution — the scalar reference dereferences an
        // index-dependent f32 table.
        reg.register(OpKindKey::DequantizeMxfp4, mxfp4_sig.clone());
        reg.inject_trace(
            OpKindKey::DequantizeMxfp4,
            OpTrace {
                pattern: ComputePattern::QuantDecode {
                    block_size: 32,
                    decode: vec![
                        TraceOp::Input(0),                          // [0] packed nibble byte
                        TraceOp::Input(1),                          // [1] e2m1 LUT (16 × f32)
                        TraceOp::Permute { src: ValueId(0), indices: ValueId(1) },    // [2] LUT[nibble] → f32 value
                        TraceOp::Input(2),                          // [3] e8m0 scale (already decoded to f32 by ABI prologue)
                        TraceOp::BlockScale { data: ValueId(2), scale: ValueId(3), block_size: 32 }, // [4] apply per-block scale
                    ],
                },
                signature: mxfp4_sig,
            },
        );


        // ── MeanPool ──
        let meanpool_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::pooling::scalar_mean_pool as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0), // seq_len
                ScalarParam::Dim(1), // hidden
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::MeanPool,
            meanpool_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Reduction {
                    identity: 0.0,
                    combine: vec![
                        TraceOp::Input(0),  // [0] acc (running sum)
                        TraceOp::Input(1),  // [1] new element
                        TraceOp::Add(ValueId(0), ValueId(1)), // [2] acc + element
                    ],
                    second_pass: None,
                    normalize: Some(vec![
                        TraceOp::Input(0),  // [0] sum
                        TraceOp::Input(1),  // [1] inv_seq_len (broadcast)
                        TraceOp::Mul(ValueId(0), ValueId(1)), // [2] sum * inv_seq_len = mean
                    ]),
                },
                signature: meanpool_sig,
            },
        );

        // ── L2Normalize ──
        let l2norm_sig = ScalarFnSignature {
            fn_ptr: scalar_l2_normalize as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::L2Normalize,
            l2norm_sig.clone(),
            OpTrace {
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Mul(ValueId(0), ValueId(0)), // [1] x^2
                    ],
                    finalize: vec![
                        TraceOp::Input(0),      // [0] sum_sq
                        TraceOp::Const(1e-12),   // [1] eps
                        TraceOp::Add(ValueId(0), ValueId(1)),      // [2] sum_sq + eps
                        TraceOp::Rsqrt(ValueId(2)),       // [3] 1/sqrt(sum_sq + eps)
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] inv_norm (from finalize)
                        TraceOp::Mul(ValueId(0), ValueId(1)), // [2] x * inv_norm
                    ],
                },
                signature: l2norm_sig,
            },
        );

        // ── QkNorm ──
        let qk_norm_sig = ScalarFnSignature {
            fn_ptr: scalar_qk_norm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0), // seq_len
                ScalarParam::Dim(1), // num_heads
                ScalarParam::Dim(2), // head_dim
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::QkNorm,
            qk_norm_sig.clone(),
            OpTrace {
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Mul(ValueId(0), ValueId(0)), // [1] x^2
                    ],
                    finalize: vec![
                        TraceOp::Input(0),      // [0] sum_sq
                        TraceOp::Const(1e-6),   // [1] eps
                        TraceOp::Sqrt(ValueId(0)),       // [2] sqrt(sum_sq)
                        TraceOp::Add(ValueId(2), ValueId(1)),     // [3] sqrt(sum_sq) + eps
                        TraceOp::Const(1.0),    // [4] 1.0
                        TraceOp::Div(ValueId(4), ValueId(3)),     // [5] 1.0 / (sqrt(sum_sq) + eps) = inv_norm
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] inv_norm (from finalize)
                        TraceOp::Input(2),  // [2] scale (√head_dim, injected at codegen)
                        TraceOp::Mul(ValueId(0), ValueId(1)), // [3] x * inv_norm
                        TraceOp::Mul(ValueId(3), ValueId(2)), // [4] x * inv_norm * scale
                    ],
                },
                signature: qk_norm_sig,
            },
        );

        // ── CachedGQA ──
        let cached_gqa_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::attention::scalar_cached_gqa_attention as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // Q
                ScalarParam::InputPtr,  // K_cache
                ScalarParam::InputPtr,  // V_cache
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // total_seq
                ScalarParam::Dim(2),    // num_heads
                ScalarParam::Dim(3),    // num_kv_heads
                ScalarParam::Dim(4),    // head_dim
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::CachedGQA,
            cached_gqa_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: cached_gqa_sig,
            },
        );

        // ── MoEGate ──
        let moe_gate_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::moe::scalar_moe_gate as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // hidden_input
                ScalarParam::InputPtr,  // router_w
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // num_experts
                ScalarParam::Dim(2),    // hidden
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::MoEGate,
            moe_gate_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: moe_gate_sig,
            },
        );

        // ── TopK ──
        let topk_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::moe::scalar_topk as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // gate_probs
                ScalarParam::OutputPtr, // indices + weights
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // num_experts
                ScalarParam::Dim(2),    // top_k
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::TopK,
            topk_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Reduction {
                    identity: f64::NEG_INFINITY,
                    combine: vec![
                        TraceOp::Input(0),
                        TraceOp::Input(1),
                        TraceOp::Max(ValueId(0), ValueId(1)),
                    ],
                    second_pass: None,
                    normalize: Some(vec![
                        TraceOp::Input(0),
                        TraceOp::Input(1),
                        TraceOp::Mul(ValueId(0), ValueId(1)),
                    ]),
                },
                signature: topk_sig,
            },
        );

        // ── WeightedSum ──
        let wsum_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::moe::scalar_weighted_sum as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // expert_outputs
                ScalarParam::InputPtr,  // indices
                ScalarParam::InputPtr,  // weights
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // hidden
                ScalarParam::Dim(2),    // top_k
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::WeightedSum,
            wsum_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0),  // weight
                        TraceOp::Input(1),  // expert_val
                        TraceOp::Mul(ValueId(0), ValueId(1)), // weight * expert_val
                    ],
                    num_inputs: 2,
                    num_outputs: 1,
                },
                signature: wsum_sig,
            },
        );

        use crate::scalar_ops::p4_p5::*;

        // ── P4/P5 OpKind registrations ──

        // GateMask: elementwise comparison → mask[hidden]
        let gate_mask_sig = ScalarFnSignature {
            fn_ptr: scalar_gate_mask as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // activation
                ScalarParam::InputPtr,   // gate
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),     // hidden
                ScalarParam::Scalar(0.0), // threshold
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::GateMask,
            gate_mask_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),    // gate logit
                        TraceOp::Const(0.0),  // threshold
                        TraceOp::Const(1.0),  // true_val
                        TraceOp::ConditionalBranch(ValueId(0), ValueId(2), ValueId(1)), // gate > 0 ? 1.0 : 0.0
                    ],
                },
                signature: gate_mask_sig,
            },
        );

        // AttentionSkipMask: elementwise entropy comparison → skip mask
        let attn_skip_sig = ScalarFnSignature {
            fn_ptr: scalar_attention_skip_mask as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // tokens
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),     // seq_len
                ScalarParam::Scalar(0.0), // skip_token_id
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::AttentionSkipMask,
            attn_skip_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),    // entropy value
                        TraceOp::Const(0.0),  // zero
                        TraceOp::Const(1.0),  // one
                        TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2)), // entropy == 0 ? 0.0 : 1.0 (skip low entropy)
                    ],
                },
                signature: attn_skip_sig,
            },
        );

        // LayerBypass: elementwise threshold comparison
        let layer_bypass_sig = ScalarFnSignature {
            fn_ptr: scalar_layer_bypass as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // input hidden state
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),     // hidden
                ScalarParam::Scalar(0.001), // threshold
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::LayerBypass,
            layer_bypass_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),    // delta_rho
                        TraceOp::Const(0.0),  // zero
                        TraceOp::Const(1.0),  // one
                        TraceOp::ConditionalBranch(ValueId(0), ValueId(2), ValueId(1)), // delta_rho != 0 ? 1.0 : 0.0 (bypass when zero)
                    ],
                },
                signature: layer_bypass_sig,
            },
        );

        // ResidualWithTelemetry: binary add + telemetry output
        let residual_tel_sig = ScalarFnSignature {
            fn_ptr: scalar_vec_add as *const u8,
            params: vec![
                ScalarParam::InputPtr, ScalarParam::InputPtr,
                ScalarParam::OutputPtr, ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::ResidualWithTelemetry,
            residual_tel_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // x_in
                        TraceOp::Input(1),  // layer_out
                        TraceOp::Add(ValueId(0), ValueId(1)), // x_out = x_in + layer_out
                    ],
                },
                signature: residual_tel_sig,
            },
        );

        // MoEConditionalAdd: conditional binary add
        let moe_cond_sig = ScalarFnSignature {
            fn_ptr: scalar_vec_add as *const u8,
            params: vec![
                ScalarParam::InputPtr, ScalarParam::InputPtr,
                ScalarParam::OutputPtr, ScalarParam::Dim(0),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::MoEConditionalAdd,
            moe_cond_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // accumulator
                        TraceOp::Input(1),  // expert_output
                        TraceOp::Add(ValueId(0), ValueId(1)), // acc + expert_output (mask applied externally)
                    ],
                },
                signature: moe_cond_sig,
            },
        );

        // SoftmaxWithEntropy: norm-like (softmax + entropy)
        let softmax_ent_sig = ScalarFnSignature {
            fn_ptr: scalar_softmax_with_entropy as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // logits
                ScalarParam::OutputPtr,  // softmax probabilities + entropy
                ScalarParam::Dim(0),     // vocab_size
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::SoftmaxWithEntropy,
            softmax_ent_sig.clone(),
            OpTrace {
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0),  // x
                        TraceOp::Max(ValueId(0), ValueId(0)), // running max
                    ],
                    finalize: vec![
                        TraceOp::Input(0),  // max
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // x
                        TraceOp::Input(1),  // max
                        TraceOp::Sub(ValueId(0), ValueId(1)), // x - max
                        TraceOp::Exp(ValueId(2)),    // exp(x - max)
                    ],
                },
                signature: softmax_ent_sig,
            },
        );

        // FusedRmsNormGemm: Gemm pattern (RmsNorm fused into GEMM prologue)
        let fused_rn_gemm_sig = ScalarFnSignature {
            fn_ptr: scalar_gemm as *const u8,
            params: vec![
                ScalarParam::InputPtr, ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0), ScalarParam::Dim(1), ScalarParam::Dim(2),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::FusedRmsNormGemm,
            fused_rn_gemm_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: fused_rn_gemm_sig,
            },
        );

        // MaskedGemm: Gemm pattern (with row mask)
        let masked_gemm_sig = ScalarFnSignature {
            fn_ptr: scalar_gemm as *const u8,
            params: vec![
                ScalarParam::InputPtr, ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0), ScalarParam::Dim(1), ScalarParam::Dim(2),
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::MaskedGemm,
            masked_gemm_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: masked_gemm_sig,
            },
        );

        // EntropyGate: elementwise (entropy → gate decision)
        let entropy_gate_sig = ScalarFnSignature {
            fn_ptr: scalar_entropy_gate as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // entropy
                ScalarParam::OutputPtr,  // gate decision
                ScalarParam::Dim(0),     // vocab_size
                ScalarParam::Scalar(0.0), // threshold
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::EntropyGate,
            entropy_gate_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),    // entropy
                        TraceOp::Const(0.0),  // zero
                        TraceOp::Const(1.0),  // one
                        TraceOp::ConditionalBranch(ValueId(0), ValueId(2), ValueId(1)), // entropy != 0 ? 1.0 : 0.0 → write KV
                    ],
                },
                signature: entropy_gate_sig,
            },
        );

        // VRangeQuant: injective (value-range quantization)
        let vrange_sig = ScalarFnSignature {
            fn_ptr: scalar_vrange_quant as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // input values
                ScalarParam::OutputPtr,  // dequantized output
                ScalarParam::Dim(0),     // seq_len
                ScalarParam::Dim(1),     // kv_dim
                ScalarParam::Dim(2),     // block_size
                ScalarParam::Scalar(4.0), // bits (as f32 placeholder)
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::VRangeQuant,
            vrange_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0),    // value
                        TraceOp::Const(1.0),  // scale (computed from range)
                        TraceOp::Mul(ValueId(0), ValueId(1)),   // quantized = value * scale
                    ],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: vrange_sig,
            },
        );

        // KvCentroidPrefetch: injective (centroid → prefetch address)
        let prefetch_sig = ScalarFnSignature {
            fn_ptr: scalar_kv_centroid_prefetch as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // distances
                ScalarParam::OutputPtr,  // prefetch mask
                ScalarParam::Dim(0),     // total (seq_len * num_heads)
                ScalarParam::Scalar(0.0), // threshold
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::KvCentroidPrefetch,
            prefetch_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0),  // centroid position
                    ],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: prefetch_sig,
            },
        );

        // VariableLengthBatch: injective (ragged → compact)
        let vlb_sig = ScalarFnSignature {
            fn_ptr: scalar_variable_length_batch as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // input (ragged concatenated)
                ScalarParam::OutputPtr,  // output (padded row-major)
                ScalarParam::InputPtr,   // lengths (per-sequence lengths)
                ScalarParam::Dim(0),     // num_seqs
                ScalarParam::Dim(1),     // max_len
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::VariableLengthBatch,
            vlb_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0),  // ragged input
                    ],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: vlb_sig,
            },
        );

        // ── AltUp (Gemma 4 E2B/E4B) ──
        // AltUpPredict/AltUpCorrect/AltUpInject are Injective ops.
        // Registered with manual Injective trace bodies (scalar impls pending).
        // AltUpPredict: 2 inputs (stacked [P,S,H], coefs [S,P²]) → 1 output (predictions [P,S,H])
        let altup_pred_sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(), // scalar impl pending
            params: vec![
                ScalarParam::InputPtr,  // stacked
                ScalarParam::InputPtr,  // coefs
                ScalarParam::OutputPtr, // predictions
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // num_preds
                ScalarParam::Dim(2),    // hidden
            ],
        };
        reg.register(OpKindKey::AltUpPredict, altup_pred_sig);
        reg.inject_trace(
            OpKindKey::AltUpPredict,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0), TraceOp::Input(1)],
                    num_inputs: 2,
                    num_outputs: 1,
                },
                signature: ScalarFnSignature { fn_ptr: std::ptr::null(), params: vec![] },
            },
        );

        // AltUpCorrect: 3 inputs (predictions, coefs, activated) → 1 output (corrected)
        let altup_corr_sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::InputPtr,  // predictions
                ScalarParam::InputPtr,  // corrected_coefs
                ScalarParam::InputPtr,  // activated
                ScalarParam::OutputPtr, // corrected
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // num_preds
                ScalarParam::Dim(2),    // hidden
            ],
        };
        reg.register(OpKindKey::AltUpCorrect, altup_corr_sig);
        reg.inject_trace(
            OpKindKey::AltUpCorrect,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Input(2)],
                    num_inputs: 3,
                    num_outputs: 1,
                },
                signature: ScalarFnSignature { fn_ptr: std::ptr::null(), params: vec![] },
            },
        );

        // AltUpInject: 2 inputs (corrected, ple_projected) → 1 output (corrected, in-place)
        let altup_inj_sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::InputPtr,  // corrected
                ScalarParam::InputPtr,  // ple_projected
                ScalarParam::OutputPtr, // corrected (in-place)
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // num_preds
                ScalarParam::Dim(2),    // hidden
            ],
        };
        reg.register(OpKindKey::AltUpInject, altup_inj_sig);
        reg.inject_trace(
            OpKindKey::AltUpInject,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0), TraceOp::Input(1)],
                    num_inputs: 2,
                    num_outputs: 1,
                },
                signature: ScalarFnSignature { fn_ptr: std::ptr::null(), params: vec![] },
            },
        );

        // ── DepthwiseConv1D (USM Conformer convolution module) ──
        // Per-channel 1D conv: output[t, c] = Σ_k weight[c, k] * x[t - pad + k, c]。
        // SymExec 对 `causal: u32` 分支 + 三层循环 (t, c, k) 处理精度不足,
        // 直接注入 manual Injective trace (标记 2 输入 1 输出),
        // codegen lower 路径待真实 SIMD 实现后续补齐 (当前 Err 占位)。
        // fn_ptr 指向 scalar_depthwise_conv1d — 仅供 Scalar + SymExec 数值 ground truth / 测试,
        // 禁止运行时调用 (CLAUDE.md NO_SCALAR)。
        let dw_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::depthwise_conv1d::scalar_depthwise_conv1d as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // x
                ScalarParam::WeightPtr, // weight
                ScalarParam::OutputPtr, // out
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // channels
                ScalarParam::Dim(2),    // kernel_size
                ScalarParam::Scalar(1.0), // causal flag (u32 作为 f32 占位;真实传参由 JIT ABI 决定)
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::DepthwiseConv1D,
            dw_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0), // x
                        TraceOp::Input(1), // weight
                    ],
                    num_inputs: 2,
                    num_outputs: 1,
                },
                signature: dw_sig,
            },
        );

        // ── PatchEmbed (SigLIP / ViT vision tower, T44) ──
        // Conv2D 滑动窗口 (stride = patch_size) 把图像打成 patch token 序列。
        // SymExec 对五层嵌套循环 (p_row × p_col × e × c × kr × kc) 无法提取
        // 结构化 trace, 直接注入 manual Injective trace (标记 2 输入 1 输出);
        // codegen lower 路径待真实 SIMD 实现后续补齐 (当前 Err 占位)。
        // fn_ptr 指向 scalar_patch_embed — 仅供 Scalar + SymExec 数值 ground truth / 测试,
        // 禁止运行时调用 (CLAUDE.md NO_SCALAR)。
        let pe_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::patch_embed::scalar_patch_embed as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // image
                ScalarParam::WeightPtr, // kernel
                ScalarParam::OutputPtr, // patches
                ScalarParam::Dim(0),    // patch_size
                ScalarParam::Dim(1),    // embed_dim
                ScalarParam::Dim(2),    // in_channels
                ScalarParam::Dim(3),    // image_size
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::PatchEmbed,
            pe_sig.clone(),
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0), // image
                        TraceOp::Input(1), // kernel
                    ],
                    num_inputs: 2,
                    num_outputs: 1,
                },
                signature: pe_sig,
            },
        );

        // ── LearnedPos2D (SigLIP / ViT learned positional embedding, T44) ──
        // Pure binary elementwise add: out[p, d] = patches[p, d] + pos_table[p, d]。
        // 注入 BinaryElementwise trace 以便 codegen fallback 走通用 elementwise
        // 路径 (复用 emit_elementwise_inline); 本任务阶段仍在 emit_standalone_op
        // 显式返回 Err 占位 (待后续任务打通真实 lower 路径)。
        // fn_ptr 指向 scalar_learned_pos_2d — 仅供 Scalar + SymExec 数值 ground truth / 测试,
        // 禁止运行时调用 (CLAUDE.md NO_SCALAR)。
        let lp_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::learned_pos_2d::scalar_learned_pos_2d as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // patches
                ScalarParam::WeightPtr, // pos_table
                ScalarParam::OutputPtr, // out
                ScalarParam::Dim(0),    // num_patches
                ScalarParam::Dim(1),    // embed_dim
            ],
        };
        reg.register_with_symexec_fallback(
            OpKindKey::LearnedPos2D,
            lp_sig.clone(),
            OpTrace {
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),   // [0] patches[p, d]
                        TraceOp::Input(1),   // [1] pos_table[p, d]
                        TraceOp::Add(ValueId(0), ValueId(1)),  // [2] patches + pos_table
                    ],
                },
                signature: lp_sig,
            },
        );

        // ── MoEDispatchPacked (OpenAI gpt-oss-20b packed-expert + mxfp4 MoE) ──
        // Composite opaque op: index dispatch + mxfp4 dequant + clipped SwiGLU +
        // down GEMV + weighted accumulate。registry 只挂标量参考 `scalar_moe_dispatch_packed`
        // 作为 Scalar + SymExec ground truth (NOT runtime callable — CLAUDE.md NO_SCALAR)。
        // JIT codegen 走 plan_lower::lower_moe_dispatch_packed 专用分支 (类似
        // MultiHeadAttention / Gather / AltUp),不依赖 OpTrace。
        let moe_dp_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::moe_dispatch_packed::scalar_moe_dispatch_packed as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // hidden_input (placeholder; 真实 lower 通过 op.inputs 解析)
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // hidden
                ScalarParam::Dim(2),    // num_experts
            ],
        };
        // 不走 symexec fallback: 组合算子, symexec 无法追踪 (mxfp4 LUT + 索引分发)。
        reg.register(OpKindKey::MoEDispatchPacked, moe_dp_sig.clone());
        reg.inject_trace(
            OpKindKey::MoEDispatchPacked,
            OpTrace {
                // Opaque 复合算子,这里用 Gemm 占位 (表示有 heavy compute),真正 lower
                // 走 plan_lower 专用分支,不读取此 pattern。
                pattern: ComputePattern::Gemm,
                signature: moe_dp_sig,
            },
        );

        // ── MoERouter (MoE router: GEMM + softmax + top-k selection) ──
        // Opaque composite op producing router_weights [seq_len, top_k] and
        // router_indices [seq_len, top_k] (u32 in f32 bits).
        // extract_op_trace returns Ok(vec![]) — plan_lower handles specialized dispatch.
        // fn_ptr reuses scalar_moe_gate as placeholder (never called at runtime, NO_SCALAR).
        let moe_router_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::moe::scalar_moe_gate as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // hidden_input
                ScalarParam::InputPtr,  // router_weight
                ScalarParam::OutputPtr, // router_weights_out
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // num_experts
                ScalarParam::Dim(2),    // hidden
            ],
        };
        reg.register(OpKindKey::MoERouter, moe_router_sig.clone());
        reg.inject_trace(
            OpKindKey::MoERouter,
            OpTrace {
                pattern: ComputePattern::Gemm,
                signature: moe_router_sig,
            },
        );

        // ── Argmax (GRAPH-SHAPE-DRIVEN-MEGA-KERNEL §2.2) ──
        // Reduction op — manual trace (symexec cannot handle data-dependent branches).
        let argmax_sig = ScalarFnSignature {
            fn_ptr: scalar_argmax as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::Argmax, argmax_sig.clone());
        reg.inject_trace(
            OpKindKey::Argmax,
            OpTrace {
                pattern: ComputePattern::Reduction {
                    identity: f64::NEG_INFINITY,
                    combine: vec![
                        TraceOp::Input(0),  // [0] a (running max)
                        TraceOp::Input(1),  // [1] b (new element)
                        TraceOp::Max(ValueId(0), ValueId(1)), // [2] max(a, b)
                    ],
                    second_pass: None,
                    normalize: None,
                },
                signature: argmax_sig,
            },
        );

        // ── Gather (embedding lookup: output[i,j] = table[indices[i], j]) ──
        // Injective op — index-based memory read. SymExec 无法自动提取索引 load 语义,
        // 直接注入 manual Injective trace (标记 2 输入 1 输出),
        // codegen 走专用 lower_gather 路径。
        let gather_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::gather::scalar_gather as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // indices
                ScalarParam::InputPtr,  // table
                ScalarParam::OutputPtr, // output
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // embed_dim
                ScalarParam::Dim(2),    // table_rows
            ],
        };
        reg.register(OpKindKey::Gather, gather_sig.clone());
        reg.inject_trace(
            OpKindKey::Gather,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0), // indices
                        TraceOp::Input(1), // table
                    ],
                    num_inputs: 2,
                    num_outputs: 1,
                },
                signature: gather_sig,
            },
        );

        // ── ColumnSlice (row-major column slicing: output[s,j] = input[s,start+j]) ──
        // Injective op — stride-changing copy. SymExec 无法自动提取双层循环,
        // 直接注入 manual Injective trace (标记 1 输入 1 输出),
        // codegen 走专用 lower_column_slice 路径。
        let col_slice_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::gather::scalar_column_slice as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // input
                ScalarParam::OutputPtr, // output
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // input_inner
                ScalarParam::Dim(2),    // start
                ScalarParam::Dim(3),    // slice_dim
            ],
        };
        reg.register(OpKindKey::ColumnSlice, col_slice_sig.clone());
        reg.inject_trace(
            OpKindKey::ColumnSlice,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0), // input
                    ],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: col_slice_sig,
            },
        );

        // ── QuantGather (quantized embedding lookup with on-the-fly dequantization) ──
        // Structural op — dispatch_structural handles it via emit_quant_gather_inline.
        // NO trace injection: an Injective trace would cause try_dispatch_by_compute_pattern
        // QuantGather: 量化 embedding lookup (SPEC 24-QUANT-PIPELINE-JIT §3.1).
        // Trace 由 DecodeTraceBuilder 参数化模板生成（quant_decode.rs），不依赖 SymExec。
        // plan_lower 中 emit_quant_gather_trace_driven 对所有格式统一调用 DecodeTraceBuilder。
        let quant_gather_sig = ScalarFnSignature {
            fn_ptr: crate::scalar_ops::gather::scalar_quant_gather as *const u8,
            params: vec![
                ScalarParam::InputPtr,  // indices (token IDs as f32)
                ScalarParam::InputPtr,  // table_quant (quantized embed table bytes)
                ScalarParam::OutputPtr, // output (F32 embeddings)
                ScalarParam::Dim(0),    // seq_len
                ScalarParam::Dim(1),    // hidden_dim
                ScalarParam::Dim(2),    // vocab_size
                ScalarParam::Dim(3),    // block_size
                ScalarParam::Dim(4),    // block_bytes
                ScalarParam::Dim(5),    // header_bytes
            ],
        };
        reg.register(OpKindKey::QuantGather, quant_gather_sig.clone());
        // 注入默认 trace（Q4_0 格式）。plan_lower 在编译时会根据模型实际量化格式动态替换。
        // 使用 build_quant_gather_trace 生成完整的参数化 trace 模板。
        reg.inject_trace(
            OpKindKey::QuantGather,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: build_quant_gather_trace(crate::quant::QuantType::Q4_0, 0, 0),
                    num_inputs: 3,
                    num_outputs: 1,
                },
                signature: quant_gather_sig,
            },
        );

        // ── B 类: pass-through opaque ops (side-effect / control-flow) ──
        // Scalar reference: identity memcpy. JIT handles real logic (shared memory,
        // conditional JMP, buffer write). Trace: Injective with empty body (data pass-through).

        use crate::scalar_ops::control_flow::*;

        // StoreToken: write generated token to output buffer.
        let store_token_sig = ScalarFnSignature {
            fn_ptr: scalar_store_token as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::StoreToken, store_token_sig.clone());
        reg.inject_trace(
            OpKindKey::StoreToken,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: store_token_sig,
            },
        );

        // CheckStopCondition: check EOS / max_tokens for loop exit.
        let check_stop_sig = ScalarFnSignature {
            fn_ptr: scalar_check_stop_condition as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr],
        };
        reg.register(OpKindKey::CheckStopCondition, check_stop_sig.clone());
        reg.inject_trace(
            OpKindKey::CheckStopCondition,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: check_stop_sig,
            },
        );

        // WriteLogits: write selected logits to output buffer.
        let write_logits_sig = ScalarFnSignature {
            fn_ptr: scalar_write_logits as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::WriteLogits, write_logits_sig.clone());
        reg.inject_trace(
            OpKindKey::WriteLogits,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: write_logits_sig,
            },
        );

        // EarlyExit: conditional early exit at anchor layer.
        let early_exit_sig = ScalarFnSignature {
            fn_ptr: scalar_early_exit as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::EarlyExit, early_exit_sig.clone());
        reg.inject_trace(
            OpKindKey::EarlyExit,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: early_exit_sig,
            },
        );

        // GuardrailCheck: post-node veto probe.
        let guardrail_sig = ScalarFnSignature {
            fn_ptr: scalar_guardrail_check as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::GuardrailCheck, guardrail_sig.clone());
        reg.inject_trace(
            OpKindKey::GuardrailCheck,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: guardrail_sig,
            },
        );

        // SgInject: SG knowledge residual vector injection.
        let sg_inject_sig = ScalarFnSignature {
            fn_ptr: scalar_sg_inject as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::SgInject, sg_inject_sig.clone());
        reg.inject_trace(
            OpKindKey::SgInject,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: sg_inject_sig,
            },
        );

        // SgDetect: SG hidden state extraction.
        let sg_detect_sig = ScalarFnSignature {
            fn_ptr: scalar_sg_detect as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr],
        };
        reg.register(OpKindKey::SgDetect, sg_detect_sig.clone());
        reg.inject_trace(
            OpKindKey::SgDetect,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: sg_detect_sig,
            },
        );

        // CotStepCheck: CoT Step Hook step control flags check.
        let cot_step_sig = ScalarFnSignature {
            fn_ptr: scalar_cot_step_check as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr],
        };
        reg.register(OpKindKey::CotStepCheck, cot_step_sig.clone());
        reg.inject_trace(
            OpKindKey::CotStepCheck,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: cot_step_sig,
            },
        );

        // SessionKvRestore: session KV cache cross-turn restore.
        let session_kv_sig = ScalarFnSignature {
            fn_ptr: scalar_session_kv_restore as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::SessionKvRestore, session_kv_sig.clone());
        reg.inject_trace(
            OpKindKey::SessionKvRestore,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: session_kv_sig,
            },
        );

        // MmHiddenInject: multimodal fused hidden state injection.
        let mm_hidden_sig = ScalarFnSignature {
            fn_ptr: scalar_mm_hidden_inject as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::MmHiddenInject, mm_hidden_sig.clone());
        reg.inject_trace(
            OpKindKey::MmHiddenInject,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![TraceOp::Input(0)],
                    num_inputs: 1,
                    num_outputs: 1,
                },
                signature: mm_hidden_sig,
            },
        );

        // MtpDraft: Multi-Token Prediction draft candidate generation (MTP-001).
        // Structural op: depth iterations of GEMV + argmax + store.
        // Trace uses TraceOp::MtpDraft structural marker; auto_select expands it.
        // Scalar reference: pass-through identity (JIT handles real logic).
        let mtp_draft_sig = ScalarFnSignature {
            fn_ptr: scalar_store_token as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::MtpDraft, mtp_draft_sig.clone());
        reg.inject_trace(
            OpKindKey::MtpDraft,
            OpTrace {
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0),  // hidden_ptr
                        TraceOp::Input(1),  // weight_ptr
                        TraceOp::Input(2),  // output_tokens_ptr
                        TraceOp::MtpDraft { depth: 1, hidden_size: 1, vocab_size: 1 },
                    ],
                    num_inputs: 3,
                    num_outputs: 1,
                },
                signature: mtp_draft_sig,
            },
        );

        reg
    }
}
