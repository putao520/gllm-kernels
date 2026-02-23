use std::collections::HashMap;
use crate::compiler::graph::OpKind;
use crate::compiler::trace::{
    classify_pattern, ComputePattern, OpTrace, ScalarFnSignature, ScalarParam, TraceOp,
};
use crate::compiler::symexec::{SymbolicExecutor, SymExecError};

/// Hashable key for `OpKind` (OpKind contains f32/f64 fields that prevent `Hash`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
// TODO: parameterize eps in OpKindKey to support per-model values
pub enum OpKindKey {
    RmsNorm,
    LayerNorm,
    Gemm,
    GemmBias,
    Silu,
    Gelu,
    SwiGlu,
    GeGlu,
    Softmax,
    RoPE,
    Add,
    Mul,
    Residual,
    Reshape,
    Transpose,
    QuantGemm,
    Dequantize,
}

#[derive(Debug)]
pub enum RegistryError {
    /// Symbolic execution failed to produce a trace.
    SymExec(SymExecError),
    /// The extracted trace was empty or invalid.
    EmptyTrace,
    /// Key is already registered.
    AlreadyRegistered(OpKindKey),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::SymExec(e) => write!(f, "symexec error: {e}"),
            RegistryError::EmptyTrace => write!(f, "extracted trace is empty"),
            RegistryError::AlreadyRegistered(k) => write!(f, "key already registered: {k:?}"),
        }
    }
}

impl From<SymExecError> for RegistryError {
    fn from(e: SymExecError) -> Self {
        RegistryError::SymExec(e)
    }
}

/// Scalar operator registry: fn_ptr + cached OpTrace per operator.
pub struct ScalarOpRegistry {
    entries: HashMap<OpKindKey, ScalarFnSignature>,
    trace_cache: HashMap<OpKindKey, OpTrace>,
}

impl ScalarOpRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        ScalarOpRegistry {
            entries: HashMap::new(),
            trace_cache: HashMap::new(),
        }
    }

    /// Register a scalar function signature.
    pub fn register(&mut self, key: OpKindKey, sig: ScalarFnSignature) {
        self.entries.insert(key, sig);
    }

    /// Get the cached OpTrace for a key (returns `None` if not yet injected).
    pub fn get_trace(&self, key: &OpKindKey) -> Option<&OpTrace> {
        self.trace_cache.get(key)
    }

    /// Get the scalar function signature for a key.
    pub fn get_signature(&self, key: &OpKindKey) -> Option<&ScalarFnSignature> {
        self.entries.get(key)
    }

    /// Manually inject an OpTrace (temporary until Phase 0 symexec is implemented).
    pub fn inject_trace(&mut self, key: OpKindKey, trace: OpTrace) {
        self.trace_cache.insert(key, trace);
    }

    /// Convert an `OpKind` to its hashable `OpKindKey`.
    pub fn key_from_op_kind(kind: &OpKind) -> OpKindKey {
        match kind {
            OpKind::RmsNorm { .. } => OpKindKey::RmsNorm,
            OpKind::LayerNorm { .. } => OpKindKey::LayerNorm,
            OpKind::Gemm { .. } => OpKindKey::Gemm,
            OpKind::GemmBias { .. } => OpKindKey::GemmBias,
            OpKind::Silu => OpKindKey::Silu,
            OpKind::Gelu => OpKindKey::Gelu,
            OpKind::SwiGlu => OpKindKey::SwiGlu,
            OpKind::GeGlu => OpKindKey::GeGlu,
            OpKind::Softmax => OpKindKey::Softmax,
            OpKind::RoPE { .. } => OpKindKey::RoPE,
            OpKind::Add => OpKindKey::Add,
            OpKind::Mul => OpKindKey::Mul,
            OpKind::Residual => OpKindKey::Residual,
            OpKind::Transpose { .. } => OpKindKey::Transpose,
            OpKind::Reshape { .. } => OpKindKey::Reshape,
            OpKind::QuantGemm { .. } => OpKindKey::QuantGemm,
            OpKind::Dequantize { .. } => OpKindKey::Dequantize,
        }
    }

    /// Number of registered scalar functions.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Number of cached OpTraces.
    pub fn num_traces(&self) -> usize {
        self.trace_cache.len()
    }

    /// Create a registry pre-populated with all scalar functions and their OpTraces.
    pub fn with_defaults() -> Self {
        use crate::scalar_ops::activations::*;
        use crate::scalar_ops::blas::*;
        use crate::scalar_ops::norms::*;
        use crate::scalar_ops::rope::*;

        let mut reg = Self::new();

        // ── SiLU ──
        let silu_sig = ScalarFnSignature {
            fn_ptr: scalar_silu as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::Silu, silu_sig.clone());
        reg.inject_trace(
            OpKindKey::Silu,
            OpTrace {
                op_kind: OpKind::Silu,
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),   // [0] v
                        TraceOp::Neg(0),     // [1] -v
                        TraceOp::Exp(1),     // [2] exp(-v)
                        TraceOp::Const(1.0), // [3] 1.0
                        TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
                        TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
                    ],
                },
                signature: silu_sig,
            },
        );

        // ── GELU ──
        let gelu_sig = ScalarFnSignature {
            fn_ptr: scalar_gelu as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::Gelu, gelu_sig.clone());
        reg.inject_trace(
            OpKindKey::Gelu,
            OpTrace {
                op_kind: OpKind::Gelu,
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),            // [0] x
                        TraceOp::Mul(0, 0),           // [1] x^2
                        TraceOp::Mul(1, 0),           // [2] x^3
                        TraceOp::Const(0.044715),     // [3] coeff
                        TraceOp::Mul(3, 2),           // [4] coeff * x^3
                        TraceOp::Add(0, 4),           // [5] x + coeff*x^3
                        TraceOp::Const(0.7978845608), // [6] sqrt(2/pi)
                        TraceOp::Mul(6, 5),           // [7] sqrt(2/pi) * (...)
                        TraceOp::Tanh(7),             // [8] tanh(...)
                        TraceOp::Const(1.0),          // [9] 1.0
                        TraceOp::Add(9, 8),           // [10] 1 + tanh(...)
                        TraceOp::Const(0.5),          // [11] 0.5
                        TraceOp::Mul(11, 0),          // [12] 0.5 * x
                        TraceOp::Mul(12, 10),         // [13] result
                    ],
                },
                signature: gelu_sig,
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
        reg.register(OpKindKey::SwiGlu, swiglu_sig.clone());
        reg.inject_trace(
            OpKindKey::SwiGlu,
            OpTrace {
                op_kind: OpKind::SwiGlu,
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),   // [0] gate
                        TraceOp::Input(1),   // [1] up
                        TraceOp::Neg(0),     // [2] -gate
                        TraceOp::Exp(2),     // [3] exp(-gate)
                        TraceOp::Const(1.0), // [4] 1.0
                        TraceOp::Add(3, 4),  // [5] 1 + exp(-gate)
                        TraceOp::Div(0, 5),  // [6] silu(gate)
                        TraceOp::Mul(6, 1),  // [7] silu(gate) * up
                    ],
                },
                signature: swiglu_sig,
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
        reg.register(OpKindKey::GeGlu, geglu_sig.clone());
        reg.inject_trace(
            OpKindKey::GeGlu,
            OpTrace {
                op_kind: OpKind::GeGlu,
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),            // [0] gate
                        TraceOp::Input(1),            // [1] up
                        TraceOp::Mul(0, 0),           // [2] gate^2
                        TraceOp::Mul(2, 0),           // [3] gate^3
                        TraceOp::Const(0.044715),     // [4] coeff
                        TraceOp::Mul(4, 3),           // [5] coeff * gate^3
                        TraceOp::Add(0, 5),           // [6] gate + coeff*gate^3
                        TraceOp::Const(0.7978845608), // [7] sqrt(2/pi)
                        TraceOp::Mul(7, 6),           // [8] sqrt(2/pi) * (...)
                        TraceOp::Tanh(8),             // [9] tanh(...)
                        TraceOp::Const(1.0),          // [10] 1.0
                        TraceOp::Add(10, 9),          // [11] 1 + tanh(...)
                        TraceOp::Const(0.5),          // [12] 0.5
                        TraceOp::Mul(12, 0),          // [13] 0.5 * gate
                        TraceOp::Mul(13, 11),         // [14] gelu(gate)
                        TraceOp::Mul(14, 1),          // [15] gelu(gate) * up
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
        reg.register(OpKindKey::Add, add_sig.clone());
        reg.inject_trace(
            OpKindKey::Add,
            OpTrace {
                op_kind: OpKind::Add,
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // [0] a
                        TraceOp::Input(1),  // [1] b
                        TraceOp::Add(0, 1), // [2] a + b
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
        reg.register(OpKindKey::Mul, mul_sig.clone());
        reg.inject_trace(
            OpKindKey::Mul,
            OpTrace {
                op_kind: OpKind::Mul,
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // [0] a
                        TraceOp::Input(1),  // [1] b
                        TraceOp::Mul(0, 1), // [2] a * b
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
        reg.register(OpKindKey::Residual, residual_sig.clone());
        reg.inject_trace(
            OpKindKey::Residual,
            OpTrace {
                op_kind: OpKind::Residual,
                pattern: ComputePattern::BinaryElementwise {
                    body: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] residual
                        TraceOp::Add(0, 1), // [2] x + residual
                    ],
                },
                signature: residual_sig,
            },
        );

        // ── Softmax ──
        let softmax_sig = ScalarFnSignature {
            fn_ptr: scalar_softmax as *const u8,
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        reg.register(OpKindKey::Softmax, softmax_sig.clone());
        reg.inject_trace(
            OpKindKey::Softmax,
            OpTrace {
                op_kind: OpKind::Softmax,
                pattern: ComputePattern::Reduction {
                    identity: f64::NEG_INFINITY,
                    combine: vec![
                        TraceOp::Input(0),  // [0] a (running max / sum)
                        TraceOp::Input(1),  // [1] b (new element)
                        TraceOp::Max(0, 1), // [2] max(a, b)
                    ],
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
        reg.register(OpKindKey::RmsNorm, rms_sig.clone());
        reg.inject_trace(
            OpKindKey::RmsNorm,
            OpTrace {
                op_kind: OpKind::RmsNorm { eps: 1e-5 }, // default eps; actual value comes from graph OpKind at compile time
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Mul(0, 0), // [1] x^2
                    ],
                    finalize: vec![
                        TraceOp::Input(0),    // [0] sum_sq
                        TraceOp::Input(1),    // [1] n (as float)
                        TraceOp::Div(0, 1),   // [2] mean = sum_sq / n
                        TraceOp::Const(1e-5), // [3] placeholder eps; codegen must substitute actual value from OpKind::RmsNorm { eps }
                        TraceOp::Add(2, 3),   // [4] mean + eps
                        TraceOp::Rsqrt(4),    // [5] rsqrt(mean + eps)
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] scale (from finalize)
                        TraceOp::Input(2),  // [2] weight
                        TraceOp::Mul(0, 1), // [3] x * scale
                        TraceOp::Mul(3, 2), // [4] x * scale * weight
                    ],
                },
                signature: rms_sig,
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
        reg.register(OpKindKey::LayerNorm, ln_sig.clone());
        reg.inject_trace(
            OpKindKey::LayerNorm,
            OpTrace {
                op_kind: OpKind::LayerNorm { eps: 1e-5 }, // default eps; actual value comes from graph OpKind at compile time
                pattern: ComputePattern::NormLike {
                    reduce: vec![
                        TraceOp::Input(0), // [0] x  (used for both mean and variance)
                    ],
                    finalize: vec![
                        TraceOp::Input(0),    // [0] mean
                        TraceOp::Input(1),    // [1] var
                        TraceOp::Const(1e-5), // [2] placeholder eps; codegen must substitute actual value from OpKind::LayerNorm { eps }
                        TraceOp::Add(1, 2),   // [3] var + eps
                        TraceOp::Rsqrt(3),    // [4] rsqrt(var + eps)
                    ],
                    transform: vec![
                        TraceOp::Input(0),  // [0] x
                        TraceOp::Input(1),  // [1] mean
                        TraceOp::Input(2),  // [2] scale (from finalize)
                        TraceOp::Input(3),  // [3] weight
                        TraceOp::Input(4),  // [4] bias
                        TraceOp::Sub(0, 1), // [5] x - mean
                        TraceOp::Mul(5, 2), // [6] (x - mean) * scale
                        TraceOp::Mul(6, 3), // [7] normed * weight
                        TraceOp::Add(7, 4), // [8] normed * weight + bias
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
        reg.register(OpKindKey::Gemm, gemm_sig.clone());
        reg.inject_trace(
            OpKindKey::Gemm,
            OpTrace {
                op_kind: OpKind::Gemm { m: 0, n: 0, k: 0 },
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
            ],
        };
        reg.register(OpKindKey::RoPE, rope_sig.clone());
        reg.inject_trace(
            OpKindKey::RoPE,
            OpTrace {
                op_kind: OpKind::RoPE { head_dim: 0, theta: 0.0 },
                pattern: ComputePattern::Injective {
                    body: vec![
                        TraceOp::Input(0),  // [0] x0
                        TraceOp::Input(1),  // [1] x1
                        TraceOp::Input(2),  // [2] cos
                        TraceOp::Input(3),  // [3] sin
                        TraceOp::Mul(0, 2), // [4] x0 * cos
                        TraceOp::Mul(1, 3), // [5] x1 * sin
                        TraceOp::Sub(4, 5), // [6] x0*cos - x1*sin  (out0)
                        TraceOp::Mul(1, 2), // [7] x1 * cos
                        TraceOp::Mul(0, 3), // [8] x0 * sin
                        TraceOp::Add(7, 8), // [9] x1*cos + x0*sin  (out1)
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
        reg.register(OpKindKey::GemmBias, gemm_bias_sig.clone());
        reg.inject_trace(
            OpKindKey::GemmBias,
            OpTrace {
                op_kind: OpKind::GemmBias { m: 0, n: 0, k: 0 },
                pattern: ComputePattern::Gemm,
                signature: gemm_bias_sig,
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
        reg.register(OpKindKey::Transpose, transpose_sig.clone());
        reg.inject_trace(
            OpKindKey::Transpose,
            OpTrace {
                op_kind: OpKind::Transpose { perm: vec![1, 0] },
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
        reg.register(OpKindKey::Reshape, reshape_sig.clone());
        reg.inject_trace(
            OpKindKey::Reshape,
            OpTrace {
                op_kind: OpKind::Reshape { target_shape: vec![] },
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
        reg.register(OpKindKey::QuantGemm, quant_gemm_sig.clone());
        reg.inject_trace(
            OpKindKey::QuantGemm,
            OpTrace {
                op_kind: OpKind::QuantGemm { m: 0, n: 0, k: 0, block_size: 32, bits: 4 },
                pattern: ComputePattern::Gemm,
                signature: quant_gemm_sig,
            },
        );

        // --- Dequantize: scalar dequantization (quant * scale) ---
        let dequant_sig = ScalarFnSignature {
            fn_ptr: scalar_dequantize as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr, // scales
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Dim(0), // block_size
            ],
        };
        reg.register(OpKindKey::Dequantize, dequant_sig.clone());
        reg.inject_trace(
            OpKindKey::Dequantize,
            OpTrace {
                op_kind: OpKind::Dequantize { num_elements: 0, block_size: 32, bits: 4 },
                pattern: ComputePattern::QuantDecode {
                    block_size: 32,
                    decode: vec![
                        TraceOp::Input(0),  // [0] quantized value
                        TraceOp::Input(1),  // [1] scale
                        TraceOp::Mul(0, 1), // [2] dequantized = quant * scale
                    ],
                },
                signature: dequant_sig,
            },
        );

        reg
    }

    /// Auto-register a scalar function by running symbolic execution to extract its trace.
    pub fn auto_register_from_symexec(
        &mut self,
        key: OpKindKey,
        fn_sig: ScalarFnSignature,
    ) -> Result<ComputePattern, RegistryError> {
        if self.entries.contains_key(&key) {
            return Err(RegistryError::AlreadyRegistered(key));
        }

        // Count float and pointer params from the signature
        let n_float = fn_sig.params.iter().filter(|p| matches!(p, ScalarParam::Scalar(_))).count();
        let n_ptr = fn_sig.params.iter().filter(|p| matches!(p, ScalarParam::InputPtr | ScalarParam::OutputPtr | ScalarParam::WeightPtr)).count();

        let executor = SymbolicExecutor::new(n_float, n_ptr);
        let trace_ops = executor.extract_trace().map_err(RegistryError::SymExec)?;

        if trace_ops.is_empty() {
            return Err(RegistryError::EmptyTrace);
        }

        let pattern = classify_pattern(&trace_ops);
        let trace = OpTrace {
            op_kind: OpKind::Silu, // placeholder — caller should set real op_kind
            pattern: pattern.clone(),
            signature: fn_sig.clone(),
        };
        self.entries.insert(key.clone(), fn_sig);
        self.trace_cache.insert(key, trace);
        Ok(pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::OpKind;

    #[test]
    fn registry_with_defaults_has_all_ops() {
        let reg = ScalarOpRegistry::with_defaults();

        let expected_keys = [
            OpKindKey::Silu,
            OpKindKey::Gelu,
            OpKindKey::SwiGlu,
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
        ];

        for key in &expected_keys {
            assert!(
                reg.get_signature(key).is_some(),
                "missing signature for {key:?}"
            );
            assert!(
                reg.get_trace(key).is_some(),
                "missing trace for {key:?}"
            );
        }

        assert_eq!(reg.num_entries(), expected_keys.len());
        assert_eq!(reg.num_traces(), expected_keys.len());
    }

    #[test]
    fn registry_key_from_op_kind_roundtrip() {
        let cases: Vec<(OpKind, OpKindKey)> = vec![
            (OpKind::Silu, OpKindKey::Silu),
            (OpKind::Gelu, OpKindKey::Gelu),
            (OpKind::SwiGlu, OpKindKey::SwiGlu),
            (OpKind::GeGlu, OpKindKey::GeGlu),
            (OpKind::Add, OpKindKey::Add),
            (OpKind::Mul, OpKindKey::Mul),
            (OpKind::Residual, OpKindKey::Residual),
            (OpKind::Softmax, OpKindKey::Softmax),
            (OpKind::RmsNorm { eps: 1e-6 }, OpKindKey::RmsNorm),
            (OpKind::LayerNorm { eps: 1e-5 }, OpKindKey::LayerNorm),
            (OpKind::Gemm { m: 1, n: 4096, k: 4096 }, OpKindKey::Gemm),
            (OpKind::GemmBias { m: 1, n: 4096, k: 4096 }, OpKindKey::GemmBias),
            (OpKind::RoPE { head_dim: 128, theta: 10000.0 }, OpKindKey::RoPE),
            (OpKind::Transpose { perm: vec![1, 0] }, OpKindKey::Transpose),
            (OpKind::Reshape { target_shape: vec![1, 4096] }, OpKindKey::Reshape),
        ];

        let reg = ScalarOpRegistry::with_defaults();
        for (kind, expected_key) in &cases {
            let key = ScalarOpRegistry::key_from_op_kind(kind);
            assert_eq!(&key, expected_key, "key mismatch for {kind:?}");
        }
    }

    #[test]
    fn registry_silu_trace_body_valid_ssa() {
        let reg = ScalarOpRegistry::with_defaults();
        let trace = reg.get_trace(&OpKindKey::Silu).unwrap();

        if let ComputePattern::Elementwise { body } = &trace.pattern {
            assert_eq!(body.len(), 6);
            assert_eq!(body[0], TraceOp::Input(0));
            assert_eq!(body[5], TraceOp::Div(0, 4));
        } else {
            panic!("SiLU should be Elementwise");
        }
    }

    #[test]
    fn registry_gelu_trace_body_valid_ssa() {
        let reg = ScalarOpRegistry::with_defaults();
        let trace = reg.get_trace(&OpKindKey::Gelu).unwrap();

        if let ComputePattern::Elementwise { body } = &trace.pattern {
            assert_eq!(body.len(), 14);
            assert_eq!(body[0], TraceOp::Input(0));
            assert_eq!(body[8], TraceOp::Tanh(7));
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

    #[test]
    fn registry_inject_overwrites() {
        let mut reg = ScalarOpRegistry::new();
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![],
        };
        let trace1 = OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise { body: vec![TraceOp::Input(0)] },
            signature: sig.clone(),
        };
        let trace2 = OpTrace {
            op_kind: OpKind::Silu,
            pattern: ComputePattern::Elementwise {
                body: vec![TraceOp::Input(0), TraceOp::Neg(0)],
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
}
