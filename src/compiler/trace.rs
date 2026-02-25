use crate::compiler::graph::OpKind;

/// Phase 0 output: an operator's complete computational structure.
#[derive(Debug, Clone)]
pub struct OpTrace {
    /// Which graph-level operation this trace describes.
    pub op_kind: OpKind,
    /// Structural classification + SSA body.
    pub pattern: ComputePattern,
    /// Original scalar function pointer and parameter layout.
    pub signature: ScalarFnSignature,
}

/// Computational pattern — determines how Phase 3 vectorizes the operator.
#[derive(Debug, Clone)]
pub enum ComputePattern {
    /// `out[i] = f(in[i])` — single-input elementwise.
    Elementwise { body: Vec<TraceOp> },
    /// `out[i] = f(a[i], b[i])` — dual-input elementwise.
    BinaryElementwise { body: Vec<TraceOp> },
    /// Multi-input/multi-output elementwise (e.g. RoPE).
    Injective {
        body: Vec<TraceOp>,
        num_inputs: usize,
        num_outputs: usize,
    },
    /// Reduction with identity element and combine step.
    Reduction {
        identity: f64,
        combine: Vec<TraceOp>,
    },
    /// Two-pass normalize: reduce → finalize → per-element transform.
    NormLike {
        reduce: Vec<TraceOp>,
        finalize: Vec<TraceOp>,
        transform: Vec<TraceOp>,
    },
    /// Triple-loop matrix multiply (GEMM).
    Gemm,
    /// Quantization decode with fixed block size.
    QuantDecode {
        block_size: usize,
        decode: Vec<TraceOp>,
    },
}

impl ComputePattern {
    /// Return the primary computation body, if this pattern has one.
    ///
    /// For elementwise patterns this is the single body; for multi-phase
    /// patterns (NormLike, Reduction) this returns `None` — those require
    /// specialized codegen paths.
    pub fn body(&self) -> Option<&[TraceOp]> {
        match self {
            ComputePattern::Elementwise { body } => Some(body),
            ComputePattern::BinaryElementwise { body } => Some(body),
            ComputePattern::Injective { body, .. } => Some(body),
            ComputePattern::QuantDecode { decode, .. } => Some(decode),
            ComputePattern::Gemm
            | ComputePattern::Reduction { .. }
            | ComputePattern::NormLike { .. } => None,
        }
    }
}

/// Analyze a TraceOp sequence and classify its ComputePattern.
///
/// Rules:
/// - Empty body → `Injective` (layout-only op)
/// - Only `Input(0)` + unary/const ops → `Elementwise`
/// - `Input(0)` + `Input(1)` present → `BinaryElementwise`
/// - 3+ distinct inputs → `Injective`
pub fn classify_pattern(body: &[TraceOp]) -> ComputePattern {
    if body.is_empty() {
        return ComputePattern::Injective {
            body: vec![],
            num_inputs: 0,
            num_outputs: 1,
        };
    }

    let max_input = body.iter().filter_map(|op| {
        if let TraceOp::Input(idx) = op { Some(*idx) } else { None }
    }).max();

    let num_inputs = match max_input {
        Some(idx) => (idx + 1) as usize,
        None => 0,
    };

    match num_inputs {
        0 | 1 => ComputePattern::Elementwise { body: body.to_vec() },
        2 => ComputePattern::BinaryElementwise { body: body.to_vec() },
        _ => ComputePattern::Injective {
            body: body.to_vec(),
            num_inputs,
            num_outputs: 1,
        },
    }
}

/// SSA-form computation operation.
///
/// Each variant's `u32` fields reference the output index of a prior operation
/// in the same `Vec<TraceOp>`. Index 0 is the first op, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceOp {
    /// Load the i-th input element.
    Input(u32),
    /// Floating-point constant.
    Const(f64),
    // ── Arithmetic ──
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    /// Fused multiply-add: a * b + c
    Fma(u32, u32, u32),
    // ── Unary ──
    Neg(u32),
    Abs(u32),
    Exp(u32),
    Sqrt(u32),
    Rsqrt(u32),
    Tanh(u32),
    Recip(u32),
    /// Natural logarithm: ln(x).
    Log(u32),
    Max(u32, u32),
    Min(u32, u32),
}

/// Scalar function signature — pointer + parameter layout.
#[derive(Debug, Clone)]
pub struct ScalarFnSignature {
    /// Address of the `extern "C"` scalar function.
    pub fn_ptr: *const u8,
    /// Ordered parameter descriptors.
    pub params: Vec<ScalarParam>,
}

// SAFETY: fn_ptr points to a static extern "C" function in the binary's text segment.
// Static function pointers are inherently thread-safe (read-only, never deallocated).
unsafe impl Send for ScalarFnSignature {}
unsafe impl Sync for ScalarFnSignature {}

/// Describes one parameter of a scalar function.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarParam {
    InputPtr,
    OutputPtr,
    WeightPtr,
    Dim(usize),
    Scalar(f32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::OpKind;

    #[test]
    fn trace_silu_body_is_valid_ssa() {
        let body = vec![
            TraceOp::Input(0),   // [0] v
            TraceOp::Neg(0),     // [1] -v
            TraceOp::Exp(1),     // [2] exp(-v)
            TraceOp::Const(1.0), // [3] 1.0
            TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
            TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
        ];

        let trace = OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise { body: body.clone() },
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
            },
        };

        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::Neg(a) | TraceOp::Abs(a) | TraceOp::Exp(a)
                | TraceOp::Sqrt(a) | TraceOp::Rsqrt(a) | TraceOp::Tanh(a)
                | TraceOp::Recip(a) | TraceOp::Log(a) => {
                    assert!((*a as usize) < i, "SSA violation at index {i}: operand {a}");
                }
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b)
                | TraceOp::Div(a, b) | TraceOp::Max(a, b) | TraceOp::Min(a, b) => {
                    assert!((*a as usize) < i, "SSA violation at index {i}: operand {a}");
                    assert!((*b as usize) < i, "SSA violation at index {i}: operand {b}");
                }
                TraceOp::Fma(a, b, c) => {
                    assert!((*a as usize) < i, "SSA violation at index {i}: operand {a}");
                    assert!((*b as usize) < i, "SSA violation at index {i}: operand {b}");
                    assert!((*c as usize) < i, "SSA violation at index {i}: operand {c}");
                }
            }
        }

        assert!(matches!(trace.pattern, ComputePattern::Elementwise { .. }));
    }

    #[test]
    fn trace_gelu_body_is_valid_ssa() {
        let body = vec![
            TraceOp::Input(0),              // [0] x
            TraceOp::Mul(0, 0),             // [1] x^2
            TraceOp::Mul(1, 0),             // [2] x^3
            TraceOp::Const(0.044715),       // [3] 0.044715
            TraceOp::Mul(3, 2),             // [4] 0.044715 * x^3
            TraceOp::Add(0, 4),             // [5] x + 0.044715 * x^3
            TraceOp::Const(0.7978845608),   // [6] sqrt(2/pi)
            TraceOp::Mul(6, 5),             // [7] sqrt(2/pi) * (x + 0.044715*x^3)
            TraceOp::Tanh(7),               // [8] tanh(...)
            TraceOp::Const(1.0),            // [9] 1.0
            TraceOp::Add(9, 8),             // [10] 1 + tanh(...)
            TraceOp::Const(0.5),            // [11] 0.5
            TraceOp::Mul(11, 0),            // [12] 0.5 * x
            TraceOp::Mul(12, 10),           // [13] 0.5 * x * (1 + tanh(...))
        ];

        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::Mul(a, b) | TraceOp::Add(a, b) => {
                    assert!((*a as usize) < i);
                    assert!((*b as usize) < i);
                }
                TraceOp::Tanh(a) => {
                    assert!((*a as usize) < i);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn trace_rms_norm_pattern() {
        let reduce = vec![
            TraceOp::Input(0),   // [0] x
            TraceOp::Mul(0, 0),  // [1] x^2
        ];
        let finalize = vec![
            TraceOp::Input(0),       // [0] sum_sq (reduction result)
            TraceOp::Input(1),       // [1] n (dimension)
            TraceOp::Div(0, 1),      // [2] mean = sum_sq / n
            TraceOp::Const(1e-5),    // [3] eps
            TraceOp::Add(2, 3),      // [4] mean + eps
            TraceOp::Rsqrt(4),       // [5] rsqrt(mean + eps)
        ];
        let transform = vec![
            TraceOp::Input(0),   // [0] x
            TraceOp::Input(1),   // [1] scale (from finalize)
            TraceOp::Input(2),   // [2] weight
            TraceOp::Mul(0, 1),  // [3] x * scale
            TraceOp::Mul(3, 2),  // [4] x * scale * weight
        ];

        let pattern = ComputePattern::NormLike { reduce, finalize, transform };
        assert!(matches!(pattern, ComputePattern::NormLike { .. }));
    }

    #[test]
    fn trace_binary_elementwise_add() {
        let body = vec![
            TraceOp::Input(0),   // [0] a
            TraceOp::Input(1),   // [1] b
            TraceOp::Add(0, 1),  // [2] a + b
        ];
        let pattern = ComputePattern::BinaryElementwise { body };
        assert!(matches!(pattern, ComputePattern::BinaryElementwise { .. }));
    }

    #[test]
    fn trace_gemm_pattern() {
        let pattern = ComputePattern::Gemm;
        assert!(matches!(pattern, ComputePattern::Gemm));
    }

    #[test]
    fn trace_scalar_fn_signature_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ScalarFnSignature>();
    }

    #[test]
    fn test_trace_op_log_display() {
        // Verify TraceOp::Log Debug representation is correct
        let op = TraceOp::Log(0);
        assert_eq!(format!("{op:?}"), "Log(0)");

        let op5 = TraceOp::Log(5);
        assert_eq!(format!("{op5:?}"), "Log(5)");
    }
}
