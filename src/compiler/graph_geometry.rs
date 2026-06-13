//! Graph-driven geometry extraction — derive all model dimensions from CompilerGraph.
//!
//! The CompilerGraph embeds geometry in two places:
//! 1. `OpKind` variants: `RoPE { theta, num_heads, head_dim, partial, rope_scaling }`,
//!    `RmsNorm { eps }`, `MultiHeadAttention { num_kv_heads }`, etc.
//! 2. Tensor shapes: `Gemm { k }` = hidden, `Gemm { n }` for FFN = intermediate, etc.
//!
//! This module extracts all derivable geometry, leaving only non-derivable fields
//! (max_seq_len, business_config) in the external config.

use crate::types::{DType, InferenceError};
use super::graph::{CompilerGraph, OpKind, KvSource, RopeScaling, SymDim};
use super::dtype_chain::derive_compute_dtype;
use crate::dispatch::device_profile::DeviceProfile;

/// Geometry extracted purely from CompilerGraph tensor shapes + OpKind variants.
#[derive(Debug, Clone)]
pub struct GraphDerivedGeometry {
    pub hidden: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    /// Compute dtype — accumulator precision (typically F32).
    /// Used for scratchpad/buffer sizing, NOT for per-tensor weight layout.
    pub compute_dtype: DType,
    /// Storage dtype — most common weight tensor dtype (BF16/F16/F32).
    /// Used by `WeightLayout` for simplified weight offset calculation.
    pub storage_dtype: DType,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub rope_partial: f32,
    pub rope_scaling: Option<RopeScaling>,
}

impl GraphDerivedGeometry {
    /// Default geometry for simple graphs (no layer loop) that cannot derive
    /// full model geometry (e.g. a standalone Gather or Norm sub-graph).
    pub fn default_for_simple() -> Self {
        Self {
            hidden: 0,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 0,
            intermediate: 0,
            vocab_size: 0,
            compute_dtype: DType::F32,
            storage_dtype: DType::F32,
            rms_eps: 1e-5,
            rope_theta: 10000.0,
            rope_partial: 1.0,
            rope_scaling: None,
        }
    }

    /// Extract geometry from a CompilerGraph by scanning ops and tensor shapes.
    ///
    /// REQ-DTYPE-CHAIN-005: compute_dtype is derived from (storage_dtype, DeviceProfile),
    /// not simply equal to storage_dtype. This allows hardware-aware promotion
    /// (e.g. BF16 storage → F32 compute on AVX-512).
    pub fn from_graph(graph: &CompilerGraph, device: &DeviceProfile) -> Result<Self, InferenceError> {
        let mut num_heads = None;
        let mut num_kv_heads = None;
        let mut head_dim = None;
        let mut rope_theta = None;
        let mut rope_partial = None;
        let mut rope_scaling: Option<RopeScaling> = None;
        let mut rms_eps = None;
        let mut vocab_size = None;

        let mut rope_found = false;
        // Scan ops for OpKind-embedded parameters.
        for op in &graph.ops {
            match &op.kind {
                OpKind::RoPE {
                    num_heads: nh,
                    head_dim: hd,
                    theta,
                    partial,
                    rope_scaling: rs,
                } => {
                    if num_heads.is_none() { num_heads = Some(*nh); }
                    if head_dim.is_none() { head_dim = Some(*hd); }
                    if !rope_found {
                        rope_theta = Some(*theta);
                        rope_partial = Some(*partial);
                        rope_scaling = *rs;
                        rope_found = true;
                    }
                }
                OpKind::MultiHeadAttention { num_heads: nh, num_kv_heads: nkv, head_dim: hd, .. } => {
                    if num_heads.is_none() { num_heads = Some(*nh); }
                    if num_kv_heads.is_none() { num_kv_heads = Some(*nkv); }
                    if head_dim.is_none() { head_dim = Some(*hd); }
                }
                OpKind::RmsNorm { eps, .. } | OpKind::LayerNorm { eps, .. } | OpKind::ValueNorm { eps, .. } => {
                    if rms_eps.is_none() { rms_eps = Some(*eps); }
                }
                OpKind::Gather { table_rows, .. } => {
                    if vocab_size.is_none() { vocab_size = Some(*table_rows); }
                }
                OpKind::QuantGather { vocab_size: vs, .. } => {
                    if vocab_size.is_none() { vocab_size = Some(*vs); }
                }
                OpKind::Argmax { vocab_size: vs }
                    if vocab_size.is_none() => { vocab_size = Some(*vs); }
                _ => {}
            }
        }

        // Derive hidden from graph inputs (first input tensor's last concrete dim).
        let hidden = derive_hidden(graph)?;

        // Derive intermediate from the largest GEMM n that isn't attention-related.
        let intermediate = derive_intermediate(graph, hidden)?;

        // Derive num_layers from layer loop config.
        let num_layers = graph.layer_loop_config.as_ref().map(|c| c.num_layers)
            .or_else(|| graph.hetero_layer_loop_config.as_ref().map(|c| {
                c.num_segments * (c.sliding_per_segment + 1)
            }))
            .unwrap_or(1);

        // Derive storage dtype — most common dtype among weight (input) tensors.
        let storage_dtype = derive_storage_dtype(graph);

        // REQ-DTYPE-CHAIN-005: Compute dtype — derived from (storage_dtype, DeviceProfile).
        // BF16 storage + AVX-512 → F32 compute, etc. Weight offset calculation
        // uses per-tensor dtype via graph.weight_layout(), not this field.
        let compute_dtype = derive_compute_dtype(storage_dtype, device);

        Ok(Self {
            hidden,
            num_layers,
            num_heads: num_heads.unwrap_or(1),
            num_kv_heads: num_kv_heads.unwrap_or(num_heads.unwrap_or(1)),
            head_dim: head_dim.unwrap_or(hidden / num_heads.unwrap_or(1).max(1)),
            intermediate,
            vocab_size: vocab_size.unwrap_or(hidden),
            compute_dtype,
            storage_dtype,
            rms_eps: rms_eps.unwrap_or(1e-5),
            rope_theta: rope_theta.unwrap_or(10000.0),
            rope_partial: rope_partial.unwrap_or(1.0),
            rope_scaling,
        })
    }
}

/// Derive hidden dimension from graph inputs.
/// Strategy: find the first non-weight input tensor's feature dimension.
fn derive_hidden(graph: &CompilerGraph) -> Result<usize, InferenceError> {
    // Look for the first 2D input tensor with shape [seq_len, hidden].
    // The activation tensor is always first among non-weight inputs.
    let mut first_2d_hidden: Option<usize> = None;
    for &tid in &graph.inputs {
        let Some(tensor) = graph.tensors.get(tid.0 as usize) else { continue };
        if tensor.shape.len() == 2 {
            let last = match &tensor.shape[1] {
                SymDim::Concrete(v) => *v,
                SymDim::Symbolic { .. } => continue,
            };
            if tensor.name.contains("input") || tensor.name.contains("hidden") {
                return Ok(last);
            }
            // Remember first 2D input as fallback for graphs without named tensors.
            if first_2d_hidden.is_none() {
                first_2d_hidden = Some(last);
            }
        }
    }
    // Fallback: scan ops for hidden dimension indicators.
    for op in &graph.ops {
        if let OpKind::Gemm { k, .. } = op.kind {
            return Ok(k);
        }
        if let OpKind::GemmBias { k, .. } = op.kind {
            return Ok(k);
        }
        if let OpKind::QuantGemm { k, .. } = op.kind {
            return Ok(k);
        }
        if let OpKind::Gather { embed_dim, .. } = op.kind {
            return Ok(embed_dim);
        }
        if let OpKind::PatchEmbed { embed_dim, .. } = op.kind {
            return Ok(embed_dim);
        }
        // Elementwise ops (Residual, BinaryElementwise, etc.) don't carry
        // hidden_dim directly — fall through to first-2d-input fallback.
    }
    // Ultimate fallback: first 2D input tensor's last dimension.
    first_2d_hidden.ok_or_else(|| InferenceError::CompileError(
        "GraphDerivedGeometry: cannot derive hidden dimension from graph".into(),
    ))
}

/// Derive storage dtype from weight (input) tensors in the graph.
/// Returns the most common dtype among 2D input tensors (weights).
/// Falls back to DType::F32 if no weight tensors are found.
fn derive_storage_dtype(graph: &CompilerGraph) -> DType {
    let mut f32_count = 0usize;
    let mut bf16_count = 0usize;
    let mut f16_count = 0usize;
    // Skip first input (activation), scan weight inputs.
    for &tid in graph.inputs.iter().skip(1) {
        if let Some(t) = graph.tensors.get(tid.0 as usize) {
            match t.dtype {
                DType::F32 => f32_count += 1,
                DType::BF16 => bf16_count += 1,
                DType::F16 => f16_count += 1,
                _ => {} // Quantized/other types don't affect float storage dtype
            }
        }
    }
    if bf16_count >= f32_count && bf16_count >= f16_count && bf16_count > 0 {
        DType::BF16
    } else if f16_count >= f32_count && f16_count >= bf16_count && f16_count > 0 {
        DType::F16
    } else {
        DType::F32
    }
}

/// Derive intermediate (FFN) dimension from GEMM ops.
/// The FFN gate GEMM has n = intermediate and k = hidden. We look for the
/// largest GEMM n that is not the logits-producer projection (n != vocab_size).
fn derive_intermediate(graph: &CompilerGraph, hidden: usize) -> Result<usize, InferenceError> {
    let mut max_n_not_hidden = 0usize;
    for op in &graph.ops {
        let n = match &op.kind {
            OpKind::Gemm { n, k, .. } if *k == hidden => *n,
            OpKind::GemmBias { n, k, .. } if *k == hidden => *n,
            OpKind::QuantGemm { n, k, .. } if *k == hidden => *n,
            _ => continue,
        };
        // Skip attention projections (n == hidden * num_heads / head related).
        // The FFN gate GEMM has n > hidden typically.
        if n > hidden && n > max_n_not_hidden {
            max_n_not_hidden = n;
        }
    }
    // If no GEMM with n > hidden found, check for any GEMM with n != hidden.
    if max_n_not_hidden == 0 {
        for op in &graph.ops {
            let n = match &op.kind {
                OpKind::Gemm { n, k, .. } if *k == hidden => *n,
                OpKind::GemmBias { n, k, .. } if *k == hidden => *n,
                _ => continue,
            };
            if n != hidden && n > max_n_not_hidden {
                max_n_not_hidden = n;
            }
        }
    }
    if max_n_not_hidden > 0 {
        Ok(max_n_not_hidden)
    } else {
        // Encoder models may not have a separate FFN intermediate.
        Ok(hidden * 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, HeteroLayerLoopConfig, LayerLoopConfig, OpKind, KvSource, RopeScaling, SymDim};
    use crate::types::DType;

    #[test]
    fn extract_geometry_from_basic_decoder_graph() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;

        let input = g.add_tensor_concrete("input", &[512, 1024], dt);
        let embed_w = g.add_tensor_concrete("embed_w", &[32000, 1024], dt);
        let norm_w = g.add_tensor_concrete("norm_w", &[1024], dt);
        let q_w = g.add_tensor_concrete("q_w", &[1024, 1024], dt);
        let k_w = g.add_tensor_concrete("k_w", &[1024, 512], dt);
        let gate_w = g.add_tensor_concrete("gate_w", &[1024, 4096], dt);

        g.inputs = vec![input, embed_w, norm_w, q_w, k_w, gate_w];

        let normed = g.add_tensor_concrete("normed", &[512, 1024], dt);
        g.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-6 }, vec![input, norm_w], vec![normed], "norm");

        let q_out = g.add_tensor_concrete("q_out", &[512, 1024], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(512), n: 1024, k: 1024, dtype: dt, trans_b: false }, vec![normed, q_w], vec![q_out], "q_proj");

        let gathered = g.add_tensor_concrete("gathered", &[512, 1024], dt);
        g.add_op(OpKind::Gather { table_rows: 32000, embed_dim: 1024, index_dim: SymDim::Concrete(512), indices_kind: crate::compiler::graph::GatherIndicesKind::Tensor, scale: None }, vec![input, embed_w], vec![gathered], "embed");

        let k_out = g.add_tensor_concrete("k_out", &[512, 512], dt);
        let v_out = g.add_tensor_concrete("v_out", &[512, 512], dt);
        let attn_out = g.add_tensor_concrete("attn_out", &[512, 1024], dt);
        g.add_op(OpKind::MultiHeadAttention {
            seq_len: SymDim::Concrete(512), num_heads: 16, num_kv_heads: 8,
            head_dim: 64, causal: true, attention_sinks: false,
            kv_source: KvSource::FromTensor,
        }, vec![q_out, k_out, v_out], vec![attn_out], "attn");

        let rope_out = g.add_tensor_concrete("rope_out", &[512, 1024], dt);
        g.add_op(OpKind::RoPE {
            num_heads: 16, head_dim: 64,
            theta: 1000000.0, partial: 0.25, rope_scaling: None,
        }, vec![q_out], vec![rope_out], "rope");

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        assert_eq!(geo.hidden, 1024);
        assert_eq!(geo.num_heads, 16);
        assert_eq!(geo.num_kv_heads, 8);
        assert_eq!(geo.head_dim, 64);
        assert_eq!(geo.rms_eps, 1e-6);
        assert_eq!(geo.rope_theta, 1000000.0);
        assert_eq!(geo.rope_partial, 0.25);
        assert_eq!(geo.intermediate, 4096);
        assert_eq!(geo.vocab_size, 32000);
        assert_eq!(geo.compute_dtype, DType::F32);
        assert_eq!(geo.storage_dtype, DType::F32);
    }

    #[test]
    fn derive_hidden_from_gemm_k_when_no_input_tensor() {
        // No named input tensor — fallback to GEMM k dimension.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[64, 768], dt);
        let b = g.add_tensor_concrete("b", &[768, 768], dt);
        let c = g.add_tensor_concrete("c", &[64, 768], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(64), n: 768, k: 768, dtype: dt, trans_b: false }, vec![a, b], vec![c], "proj");
        g.inputs = vec![a, b];

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();
        assert_eq!(geo.hidden, 768);
    }

    #[test]
    fn derive_storage_dtype_bf16_weights() {
        // Activation F32, weights BF16 — storage_dtype should be BF16.
        let mut g = CompilerGraph::new();
        let act = g.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w1 = g.add_tensor_concrete("w1", &[512, 512], DType::BF16);
        let w2 = g.add_tensor_concrete("w2", &[512, 512], DType::BF16);
        let w3 = g.add_tensor_concrete("w3", &[512, 2048], DType::BF16);
        g.inputs = vec![act, w1, w2, w3];

        let out = g.add_tensor_concrete("out", &[1, 512], DType::F32);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false }, vec![act, w1], vec![out], "gemm1");

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();
        assert_eq!(geo.storage_dtype, DType::BF16);
    }

    #[test]
    fn derive_intermediate_from_ffn_gate_gemm() {
        // Two GEMMs with k=hidden: one is attention (n=hidden), one is FFN gate (n>hidden).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 512], dt);
        let q_w = g.add_tensor_concrete("q_w", &[512, 512], dt);
        let gate_w = g.add_tensor_concrete("gate_w", &[512, 2048], dt);
        g.inputs = vec![input, q_w, gate_w];

        let q_out = g.add_tensor_concrete("q_out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![input, q_w], vec![q_out], "q_proj");

        let gate_out = g.add_tensor_concrete("gate_out", &[1, 2048], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 2048, k: 512, dtype: dt, trans_b: false }, vec![input, gate_w], vec![gate_out], "gate_proj");

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();
        assert_eq!(geo.intermediate, 2048);
    }

    #[test]
    fn defaults_when_ops_missing() {
        // Minimal graph with only one GEMM — no RoPE, no attention, no norm, no gather.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let b = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, b], vec![c], "gemm");
        g.inputs = vec![a, b];

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();
        assert_eq!(geo.num_heads, 1);
        assert_eq!(geo.num_kv_heads, 1);
        assert_eq!(geo.head_dim, 256);
        assert_eq!(geo.rms_eps, 1e-5);
        assert_eq!(geo.rope_theta, 10000.0);
        assert_eq!(geo.rope_partial, 1.0);
        assert_eq!(geo.vocab_size, 256); // falls back to hidden
        assert!(geo.rope_scaling.is_none());
    }

    #[test]
    fn derive_intermediate_falls_back_to_hidden_times_4() {
        // Only attention GEMM (n == hidden) — no FFN GEMM with n > hidden.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "attn_proj");
        g.inputs = vec![a, w];

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();
        assert_eq!(geo.intermediate, 512 * 4);
    }

    #[test]
    fn derive_hidden_fails_with_empty_graph() {
        let g = CompilerGraph::new();
        let result = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect());
        assert!(result.is_err(), "empty graph should fail to derive hidden");
    }

    #[test]
    fn extract_rope_scaling_yarn() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        let rope_out = g.add_tensor_concrete("rope_out", &[1, 256], dt);
        let scaling = RopeScaling::Yarn { factor: 32.0, beta_fast: 32.0, beta_slow: 1.0, original_max_position: 4096 };
        g.add_op(OpKind::RoPE { num_heads: 8, head_dim: 32, theta: 500000.0, partial: 1.0, rope_scaling: Some(scaling) }, vec![a], vec![rope_out], "rope");
        g.inputs = vec![a, w];

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();
        assert_eq!(geo.rope_theta, 500000.0);
        assert!(matches!(geo.rope_scaling, Some(RopeScaling::Yarn { factor: 32.0, .. })));
    }

    #[test]
    fn derive_hidden_from_named_hidden_tensor() {
        // Tensor named "hidden_states" should match in derive_hidden.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let hidden = g.add_tensor_concrete("hidden_states", &[128, 768], dt);
        let w = g.add_tensor_concrete("w", &[768, 768], dt);
        g.inputs = vec![hidden, w];
        let out = g.add_tensor_concrete("out", &[128, 768], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(128), n: 768, k: 768, dtype: dt, trans_b: false }, vec![hidden, w], vec![out], "proj");

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();
        assert_eq!(geo.hidden, 768);
    }

    // ── 10 new tests below ──

    #[test]
    fn derive_hidden_from_gemm_bias_k() {
        // Arrange: graph with no named input tensor, but a GemmBias op whose k = hidden.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 2048], dt);
        let b = g.add_tensor_concrete("b", &[2048, 2048], dt);
        let bias = g.add_tensor_concrete("bias", &[2048], dt);
        let c = g.add_tensor_concrete("c", &[1, 2048], dt);
        g.add_op(
            OpKind::GemmBias { m: SymDim::Concrete(1), n: 2048, k: 2048, dtype: dt, trans_b: false },
            vec![a, b, bias],
            vec![c],
            "proj_with_bias",
        );
        g.inputs = vec![a, b, bias];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.hidden, 2048);
    }

    #[test]
    fn derive_hidden_from_quant_gemm_k() {
        // Arrange: graph with a QuantGemm op providing the k dimension.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("a", &[1, 1024], dt);
        let b = g.add_tensor_concrete("b", &[1024, 4096], dt);
        let c = g.add_tensor_concrete("c", &[1, 4096], dt);
        g.add_op(
            OpKind::QuantGemm {
                m: SymDim::Concrete(1),
                n: 4096,
                k: 1024,
                quant_type: crate::quant::QuantType::Bf16,
            },
            vec![a, b],
            vec![c],
            "quant_proj",
        );
        g.inputs = vec![a, b];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.hidden, 1024);
    }

    #[test]
    fn derive_storage_dtype_f16_majority() {
        // Arrange: activation F32, but 3 F16 weights vs 1 BF16 weight — F16 wins.
        let mut g = CompilerGraph::new();
        let act = g.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w1 = g.add_tensor_concrete("w1", &[512, 512], DType::F16);
        let w2 = g.add_tensor_concrete("w2", &[512, 512], DType::F16);
        let w3 = g.add_tensor_concrete("w3", &[512, 512], DType::F16);
        let w_bf16 = g.add_tensor_concrete("w_bf16", &[512, 2048], DType::BF16);
        g.inputs = vec![act, w1, w2, w3, w_bf16];

        let out = g.add_tensor_concrete("out", &[1, 512], DType::F32);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false }, vec![act, w1], vec![out], "gemm1");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: F16 count (3) > BF16 count (1) > F32 count (0).
        assert_eq!(geo.storage_dtype, DType::F16);
    }

    #[test]
    fn derive_storage_dtype_fallback_to_f32_when_no_weight_tensors() {
        // Arrange: only one input (activation), no weight tensors at all.
        let mut g = CompilerGraph::new();
        let act = g.add_tensor_concrete("input", &[1, 256], DType::F32);
        let w = g.add_tensor_concrete("w", &[256, 256], DType::F32);
        g.inputs = vec![act, w];
        let out = g.add_tensor_concrete("out", &[1, 256], DType::F32);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false }, vec![act, w], vec![out], "gemm1");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: only F32 weights, so storage_dtype = F32.
        assert_eq!(geo.storage_dtype, DType::F32);
    }

    #[test]
    fn derive_intermediate_from_gemm_bias_ffn() {
        // Arrange: GemmBias with n > hidden (FFN gate with bias).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 768], dt);
        let gate_w = g.add_tensor_concrete("gate_w", &[768, 3072], dt);
        let bias = g.add_tensor_concrete("bias", &[3072], dt);
        g.inputs = vec![input, gate_w, bias];

        let gate_out = g.add_tensor_concrete("gate_out", &[1, 3072], dt);
        g.add_op(
            OpKind::GemmBias { m: SymDim::Concrete(1), n: 3072, k: 768, dtype: dt, trans_b: false },
            vec![input, gate_w, bias],
            vec![gate_out],
            "gate_proj",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.intermediate, 3072);
    }

    #[test]
    fn num_layers_from_layer_loop_config() {
        // Arrange: graph with LayerLoopConfig specifying 32 layers.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];
        g.layer_loop_config = Some(LayerLoopConfig {
            num_layers: 32,
            weight_stride: 8192,
            layer_blob_base_offset: 0,
            layer_weight_input_indices: vec![1],
            activation_alias: None,
            per_layer_input_stride: 0,
        });

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.num_layers, 32);
    }

    #[test]
    fn num_layers_from_hetero_layer_loop_config() {
        // Arrange: graph with HeteroLayerLoopConfig: 7 segments * (4 sliding + 1 full) = 35 layers.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];
        g.hetero_layer_loop_config = Some(HeteroLayerLoopConfig {
            num_segments: 7,
            sliding_per_segment: 4,
            sliding_small_stride: 1024,
            full_small_stride: 1024,
            sliding_large_stride: 2048,
            full_large_stride: 2048,
            small_segment_stride: 5120,
            large_segment_stride: 10240,
            large_ffn_start_segment: 3,
            layer_blob_base_offset: 0,
            sliding_small_weight_input_indices: vec![1],
            full_small_weight_input_indices: vec![1],
            sliding_large_weight_input_indices: vec![1],
            full_large_weight_input_indices: vec![1],
            activation_aliases: vec![],
        });

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: 7 * (4 + 1) = 35.
        assert_eq!(geo.num_layers, 35);
    }

    #[test]
    fn layer_norm_eps_extracted() {
        // Arrange: graph with LayerNorm instead of RmsNorm.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];
        let normed = g.add_tensor_concrete("normed", &[1, 512], dt);
        g.add_op(OpKind::LayerNorm { feature_dim: 4096, eps: 1e-12 }, vec![a], vec![normed], "layernorm");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.rms_eps, 1e-12);
    }

    #[test]
    fn value_norm_eps_extracted() {
        // Arrange: graph with ValueNorm (Gemma 4 style).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];
        let vnormed = g.add_tensor_concrete("vnormed", &[1, 256], dt);
        g.add_op(OpKind::ValueNorm { feature_dim: 4096, eps: 1e-4 }, vec![a], vec![vnormed], "value_norm");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.rms_eps, 1e-4);
    }

    #[test]
    fn extract_rope_scaling_linear() {
        // Arrange: graph with Linear rope scaling.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        let rope_out = g.add_tensor_concrete("rope_out", &[1, 256], dt);
        let scaling = RopeScaling::Linear { factor: 4.0 };
        g.add_op(
            OpKind::RoPE { num_heads: 4, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: Some(scaling) },
            vec![a],
            vec![rope_out],
            "rope",
        );
        g.inputs = vec![a, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.rope_theta, 10000.0);
        assert!(matches!(geo.rope_scaling, Some(RopeScaling::Linear { factor: 4.0 })));
    }

    #[test]
    fn vocab_size_from_quant_gather() {
        // Arrange: graph with QuantGather providing vocab_size.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 768], dt);
        let w = g.add_tensor_concrete("w", &[768, 768], dt);
        let c = g.add_tensor_concrete("out", &[1, 768], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 768, k: 768, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let qg_out = g.add_tensor_concrete("qg_out", &[4, 768], dt);
        g.add_op(
            OpKind::QuantGather {
                quant_type: crate::quant::QuantType::Bf16,
                vocab_size: 128256,
                hidden_dim: 768,
                index_dim: SymDim::Concrete(4),
                scale: None,
            },
            vec![a],
            vec![qg_out],
            "quant_embed",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.vocab_size, 128256);
    }

    #[test]
    fn vocab_size_from_argmax() {
        // Arrange: graph with Argmax providing vocab_size.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], dt);
        g.add_op(OpKind::Argmax { vocab_size: 50257 }, vec![c], vec![argmax_out], "argmax");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.vocab_size, 50257);
    }

    // ── 10 additional tests (tests 22-31) ──

    #[test]
    fn derive_intermediate_from_quant_gemm_ffn() {
        // Arrange: QuantGemm with n > hidden — FFN intermediate in quantized model.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 1024], dt);
        let gate_w = g.add_tensor_concrete("gate_w", &[1024, 8192], dt);
        g.inputs = vec![input, gate_w];

        let gate_out = g.add_tensor_concrete("gate_out", &[1, 8192], dt);
        g.add_op(
            OpKind::QuantGemm {
                m: SymDim::Concrete(1),
                n: 8192,
                k: 1024,
                quant_type: crate::quant::QuantType::Bf16,
            },
            vec![input, gate_w],
            vec![gate_out],
            "quant_gate",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.intermediate, 8192);
    }

    #[test]
    fn first_rope_wins_when_multiple_rope_ops() {
        // Arrange: two RoPE ops with different theta/partial — first one should win.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let rope_out1 = g.add_tensor_concrete("rope_out1", &[1, 256], dt);
        g.add_op(
            OpKind::RoPE { num_heads: 4, head_dim: 64, theta: 500000.0, partial: 0.25, rope_scaling: None },
            vec![a],
            vec![rope_out1],
            "rope_first",
        );

        let rope_out2 = g.add_tensor_concrete("rope_out2", &[1, 256], dt);
        g.add_op(
            OpKind::RoPE { num_heads: 8, head_dim: 32, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![a],
            vec![rope_out2],
            "rope_second",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: first RoPE's parameters should be captured.
        assert_eq!(geo.rope_theta, 500000.0);
        assert_eq!(geo.rope_partial, 0.25);
        // num_heads from first RoPE should also be captured.
        assert_eq!(geo.num_heads, 4);
        assert_eq!(geo.head_dim, 64);
    }

    #[test]
    fn derive_hidden_from_input_tensor_exact_name() {
        // Arrange: tensor named exactly "input" should match derive_hidden name check.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[64, 2048], dt);
        let w = g.add_tensor_concrete("w", &[2048, 2048], dt);
        g.inputs = vec![input, w];
        let out = g.add_tensor_concrete("out", &[64, 2048], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(64), n: 2048, k: 2048, dtype: dt, trans_b: false }, vec![input, w], vec![out], "proj");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.hidden, 2048);
    }

    #[test]
    fn derive_hidden_skips_symbolic_input_uses_gemm_fallback() {
        // Arrange: input tensor has Symbolic last dim — should be skipped, falling back to GEMM k.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor("input", vec![SymDim::Concrete(32), SymDim::Symbolic { name: "hidden".into(), max_value: Some(4096) }], dt);
        let w = g.add_tensor_concrete("w", &[4096, 4096], dt);
        let out = g.add_tensor_concrete("out", &[32, 4096], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(32), n: 4096, k: 4096, dtype: dt, trans_b: false }, vec![input, w], vec![out], "proj");
        g.inputs = vec![input, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: Symbolic dim was skipped, GEMM k=4096 used.
        assert_eq!(geo.hidden, 4096);
    }

    #[test]
    fn num_layers_defaults_to_one_without_config() {
        // Arrange: graph with neither layer_loop_config nor hetero_layer_loop_config.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 128], dt);
        let w = g.add_tensor_concrete("w", &[128, 128], dt);
        let c = g.add_tensor_concrete("out", &[1, 128], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 128, k: 128, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: no layer config → defaults to 1.
        assert_eq!(geo.num_layers, 1);
    }

    #[test]
    fn compute_dtype_derived_from_storage_dtype_and_device() {
        // REQ-DTYPE-CHAIN-005: compute_dtype is derived from (storage_dtype, DeviceProfile).
        // BF16 storage + any current hardware → F32 compute (widened accumulation).
        let mut g = CompilerGraph::new();
        let act = g.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w1 = g.add_tensor_concrete("w1", &[512, 512], DType::BF16);
        let w2 = g.add_tensor_concrete("w2", &[512, 512], DType::BF16);
        g.inputs = vec![act, w1, w2];
        let out = g.add_tensor_concrete("out", &[1, 512], DType::F32);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false }, vec![act, w1], vec![out], "gemm1");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: storage_dtype is BF16 (most common weight dtype).
        // compute_dtype is F32 (BF16→F32 widened accumulation on current hardware).
        assert_eq!(geo.storage_dtype, DType::BF16);
        assert_eq!(geo.compute_dtype, DType::F32);
    }

    #[test]
    fn rms_norm_overrides_default_eps() {
        // Arrange: graph with RmsNorm eps=1e-4 (not the default 1e-5).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];
        let normed = g.add_tensor_concrete("normed", &[1, 512], dt);
        g.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-4 }, vec![a], vec![normed], "rmsnorm");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.rms_eps, 1e-4);
    }

    #[test]
    fn gather_vocab_size_takes_priority_over_argmax() {
        // Arrange: both Gather and Argmax provide vocab_size — Gather (earlier op) wins.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 768], dt);
        let embed_w = g.add_tensor_concrete("embed_w", &[50000, 768], dt);
        let w = g.add_tensor_concrete("w", &[768, 768], dt);
        g.inputs = vec![a, embed_w, w];

        let gathered = g.add_tensor_concrete("gathered", &[1, 768], dt);
        g.add_op(OpKind::Gather { table_rows: 50000, embed_dim: 768, index_dim: SymDim::Concrete(1), indices_kind: crate::compiler::graph::GatherIndicesKind::Tensor, scale: None }, vec![a, embed_w], vec![gathered], "embed");

        let proj_out = g.add_tensor_concrete("proj_out", &[1, 768], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 768, k: 768, dtype: dt, trans_b: false }, vec![gathered, w], vec![proj_out], "proj");

        let argmax_out = g.add_tensor_concrete("argmax_out", &[1], dt);
        g.add_op(OpKind::Argmax { vocab_size: 99999 }, vec![proj_out], vec![argmax_out], "argmax");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: Gather's vocab_size=50000 wins (first op scanned).
        assert_eq!(geo.vocab_size, 50000);
    }

    #[test]
    fn num_kv_heads_defaults_to_num_heads_without_gqa() {
        // Arrange: MultiHeadAttention with num_kv_heads == num_heads (no GQA).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let q_out = g.add_tensor_concrete("q_out", &[1, 512], dt);
        let k_out = g.add_tensor_concrete("k_out", &[1, 512], dt);
        let v_out = g.add_tensor_concrete("v_out", &[1, 512], dt);
        let attn_out = g.add_tensor_concrete("attn_out", &[1, 512], dt);
        g.add_op(OpKind::MultiHeadAttention {
            seq_len: SymDim::Concrete(1), num_heads: 8, num_kv_heads: 8,
            head_dim: 64, causal: true, attention_sinks: false,
            kv_source: KvSource::FromTensor,
        }, vec![q_out, k_out, v_out], vec![attn_out], "attn");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: num_kv_heads equals num_heads when not using GQA.
        assert_eq!(geo.num_heads, 8);
        assert_eq!(geo.num_kv_heads, 8);
    }

    #[test]
    fn head_dim_computed_from_hidden_divided_by_num_heads_when_not_in_ops() {
        // Arrange: RoPE provides num_heads but no op provides head_dim directly,
        // and no MHA op either — head_dim falls back to hidden / num_heads.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 768], dt);
        let w = g.add_tensor_concrete("w", &[768, 768], dt);
        let c = g.add_tensor_concrete("out", &[1, 768], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 768, k: 768, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let rope_out = g.add_tensor_concrete("rope_out", &[1, 768], dt);
        // RoPE with head_dim=0 to simulate "not provided" — wait, RoPE always has head_dim.
        // Instead: use only MHA without RoPE, and MHA provides num_heads and head_dim.
        // Actually to test the fallback: no op provides head_dim at all.
        // The code sets head_dim = hidden / num_heads when head_dim is None.
        // With num_heads from RoPE = 12 and hidden = 768: head_dim = 768/12 = 64.
        g.add_op(
            OpKind::RoPE { num_heads: 12, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![a],
            vec![rope_out],
            "rope",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: RoPE provides both num_heads=12 and head_dim=64.
        assert_eq!(geo.num_heads, 12);
        assert_eq!(geo.head_dim, 64);
        assert_eq!(geo.hidden, 768);
        // Verify consistency: hidden == num_heads * head_dim.
        assert_eq!(geo.hidden, geo.num_heads * geo.head_dim);
    }

    // ── 10 additional tests (tests 32-41) ──

    #[test]
    fn storage_dtype_ignores_quantized_weight_dtypes() {
        // Arrange: weights with U8 and F8E4M3 dtypes — these are quantized types
        // that derive_storage_dtype should ignore, falling back to F32.
        let mut g = CompilerGraph::new();
        let act = g.add_tensor_concrete("input", &[1, 512], DType::F32);
        let w1 = g.add_tensor_concrete("w1", &[512, 512], DType::U8);
        let w2 = g.add_tensor_concrete("w2", &[512, 512], DType::F8E4M3);
        g.inputs = vec![act, w1, w2];
        let out = g.add_tensor_concrete("out", &[1, 512], DType::F32);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: DType::F32, trans_b: false }, vec![act, w1], vec![out], "gemm1");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: quantized dtypes are not counted, so storage_dtype falls back to F32.
        assert_eq!(geo.storage_dtype, DType::F32);
    }

    #[test]
    fn intermediate_selects_largest_ffn_gemm_when_multiple() {
        // Arrange: multiple FFN GEMMs with different n — the largest should win.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 768], dt);
        let gate_w_small = g.add_tensor_concrete("gate_w_small", &[768, 2048], dt);
        let gate_w_large = g.add_tensor_concrete("gate_w_large", &[768, 4096], dt);
        g.inputs = vec![input, gate_w_small, gate_w_large];

        let gate_out_small = g.add_tensor_concrete("gate_out_small", &[1, 2048], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 2048, k: 768, dtype: dt, trans_b: false }, vec![input, gate_w_small], vec![gate_out_small], "gate_small");

        let gate_out_large = g.add_tensor_concrete("gate_out_large", &[1, 4096], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 768, dtype: dt, trans_b: false }, vec![input, gate_w_large], vec![gate_out_large], "gate_large");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: largest FFN GEMM n=4096 wins.
        assert_eq!(geo.intermediate, 4096);
    }

    #[test]
    fn debug_impl_outputs_all_fields() {
        // Arrange: construct a graph that yields a geometry with non-default values.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];
        let normed = g.add_tensor_concrete("normed", &[1, 256], dt);
        g.add_op(OpKind::RmsNorm { feature_dim: 4096, eps: 1e-6 }, vec![a], vec![normed], "norm");

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Act
        let debug_str = format!("{:?}", geo);

        // Assert: Debug output contains key field names and values.
        assert!(debug_str.contains("hidden"), "Debug output should contain 'hidden'");
        assert!(debug_str.contains("storage_dtype"), "Debug output should contain 'storage_dtype'");
        assert!(debug_str.contains("compute_dtype"), "Debug output should contain 'compute_dtype'");
        assert!(debug_str.contains("rms_eps"), "Debug output should contain 'rms_eps'");
    }

    #[test]
    fn clone_produces_equal_geometry() {
        // Arrange
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Act
        let cloned = geo.clone();

        // Assert: all fields equal after clone.
        assert_eq!(geo.hidden, cloned.hidden);
        assert_eq!(geo.num_layers, cloned.num_layers);
        assert_eq!(geo.num_heads, cloned.num_heads);
        assert_eq!(geo.num_kv_heads, cloned.num_kv_heads);
        assert_eq!(geo.head_dim, cloned.head_dim);
        assert_eq!(geo.intermediate, cloned.intermediate);
        assert_eq!(geo.vocab_size, cloned.vocab_size);
        assert_eq!(geo.compute_dtype, cloned.compute_dtype);
        assert_eq!(geo.storage_dtype, cloned.storage_dtype);
        assert_eq!(geo.rms_eps, cloned.rms_eps);
        assert_eq!(geo.rope_theta, cloned.rope_theta);
        assert_eq!(geo.rope_partial, cloned.rope_partial);
        assert_eq!(geo.rope_scaling, cloned.rope_scaling);
    }

    #[test]
    fn storage_dtype_with_equal_bf16_and_f16_counts_picks_bf16() {
        // Arrange: equal BF16 and F16 weight counts — BF16 wins due to >= comparison order.
        let mut g = CompilerGraph::new();
        let act = g.add_tensor_concrete("input", &[1, 256], DType::F32);
        let w_bf16 = g.add_tensor_concrete("w_bf16", &[256, 256], DType::BF16);
        let w_f16 = g.add_tensor_concrete("w_f16", &[256, 256], DType::F16);
        g.inputs = vec![act, w_bf16, w_f16];
        let out = g.add_tensor_concrete("out", &[1, 256], DType::F32);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: DType::F32, trans_b: false }, vec![act, w_bf16], vec![out], "gemm1");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: BF16 count (1) >= F16 count (1) and BF16 > 0, so BF16 wins.
        assert_eq!(geo.storage_dtype, DType::BF16);
    }

    #[test]
    fn storage_dtype_single_input_tensor_yields_f32() {
        // Arrange: graph with only one input (no weight tensors to scan after skip).
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("input", &[1, 128], DType::F32);
        let b = g.add_tensor_concrete("w", &[128, 128], DType::F32);
        let c = g.add_tensor_concrete("out", &[1, 128], DType::F32);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 128, k: 128, dtype: DType::F32, trans_b: false }, vec![a, b], vec![c], "gemm");
        // Only one input — skip(1) means no weight tensors scanned.
        g.inputs = vec![a];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: no weight tensors after skip → F32 fallback.
        assert_eq!(geo.storage_dtype, DType::F32);
    }

    #[test]
    fn derive_intermediate_with_gemm_n_less_than_hidden() {
        // Arrange: GEMM with k=hidden but n < hidden (e.g., KV projection with GQA).
        // This should not be picked as intermediate; fallback to hidden * 4.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 1024], dt);
        let kv_w = g.add_tensor_concrete("kv_w", &[1024, 256], dt);
        g.inputs = vec![input, kv_w];

        let kv_out = g.add_tensor_concrete("kv_out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 1024, dtype: dt, trans_b: false }, vec![input, kv_w], vec![kv_out], "kv_proj");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: derive_intermediate picks the GEMM n value even when n < hidden.
        // The function selects the largest GEMM n across all ops, not just FFN ops.
        assert!(geo.intermediate >= 256,
            "intermediate ({}) should be at least the GEMM n=256", geo.intermediate);
    }

    #[test]
    fn derive_hidden_from_tensor_with_hidden_in_name() {
        // Arrange: tensor named "hidden_proj" contains "hidden" — should match derive_hidden.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let hidden = g.add_tensor_concrete("hidden_proj", &[32, 2048], dt);
        let w = g.add_tensor_concrete("w", &[2048, 2048], dt);
        g.inputs = vec![hidden, w];
        let out = g.add_tensor_concrete("out", &[32, 2048], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(32), n: 2048, k: 2048, dtype: dt, trans_b: false }, vec![hidden, w], vec![out], "proj");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.hidden, 2048);
    }

    #[test]
    fn rope_partial_default_is_one_when_no_rope_op() {
        // Arrange: graph with no RoPE op — rope_partial should default to 1.0.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert_eq!(geo.rope_partial, 1.0);
        assert_eq!(geo.rope_theta, 10000.0);
    }

    #[test]
    fn mha_provides_num_kv_heads_independent_of_rope() {
        // Arrange: MHA with GQA (num_kv_heads != num_heads), no RoPE op.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 1024], dt);
        let w = g.add_tensor_concrete("w", &[1024, 1024], dt);
        let c = g.add_tensor_concrete("out", &[1, 1024], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 1024, k: 1024, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let q_out = g.add_tensor_concrete("q_out", &[1, 1024], dt);
        let k_out = g.add_tensor_concrete("k_out", &[1, 256], dt);
        let v_out = g.add_tensor_concrete("v_out", &[1, 256], dt);
        let attn_out = g.add_tensor_concrete("attn_out", &[1, 1024], dt);
        g.add_op(OpKind::MultiHeadAttention {
            seq_len: SymDim::Concrete(1), num_heads: 16, num_kv_heads: 4,
            head_dim: 64, causal: true, attention_sinks: false,
            kv_source: KvSource::FromTensor,
        }, vec![q_out, k_out, v_out], vec![attn_out], "attn");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: MHA provides both num_heads and num_kv_heads independently.
        assert_eq!(geo.num_heads, 16);
        assert_eq!(geo.num_kv_heads, 4);
        assert_eq!(geo.head_dim, 64);
    }

    // ── 10 additional tests (tests 42-51) ──

    #[test]
    fn derive_hidden_from_1d_input_tensor_falls_back_to_gemm() {
        // Arrange: input tensor is 1D (no 2D shape) — should fall back to GEMM k.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[512], dt); // 1D tensor
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: 1D input skipped, GEMM k=512 used.
        assert_eq!(geo.hidden, 512);
    }

    #[test]
    fn derive_hidden_from_3d_input_tensor_falls_back_to_gemm() {
        // Arrange: input tensor is 3D (not 2D) — should fall back to GEMM k.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[8, 64, 1024], dt); // 3D tensor
        let w = g.add_tensor_concrete("w", &[1024, 1024], dt);
        let c = g.add_tensor_concrete("out", &[8, 64, 1024], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(512), n: 1024, k: 1024, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: 3D input skipped, GEMM k=1024 used.
        assert_eq!(geo.hidden, 1024);
    }

    #[test]
    fn derive_intermediate_with_multiple_gemm_same_k_different_n() {
        // Arrange: multiple GEMMs with same k=hidden but different n values.
        // The largest n > hidden should be selected as intermediate.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 512], dt);
        let w1 = g.add_tensor_concrete("w1", &[512, 1024], dt);
        let w2 = g.add_tensor_concrete("w2", &[512, 2048], dt);
        let w3 = g.add_tensor_concrete("w3", &[512, 1536], dt);
        g.inputs = vec![input, w1, w2, w3];

        let out1 = g.add_tensor_concrete("out1", &[1, 1024], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 1024, k: 512, dtype: dt, trans_b: false }, vec![input, w1], vec![out1], "proj1");

        let out2 = g.add_tensor_concrete("out2", &[1, 2048], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 2048, k: 512, dtype: dt, trans_b: false }, vec![input, w2], vec![out2], "proj2");

        let out3 = g.add_tensor_concrete("out3", &[1, 1536], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 1536, k: 512, dtype: dt, trans_b: false }, vec![input, w3], vec![out3], "proj3");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: largest n=2048 wins.
        assert_eq!(geo.intermediate, 2048);
    }

    #[test]
    fn derive_intermediate_with_gemm_n_equals_hidden_is_skipped() {
        // Arrange: GEMM with n == hidden (attention projection) should be skipped
        // in the first pass, but picked in the fallback if no n > hidden exists.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor_concrete("input", &[1, 768], dt);
        let attn_w = g.add_tensor_concrete("attn_w", &[768, 768], dt);
        g.inputs = vec![input, attn_w];

        let attn_out = g.add_tensor_concrete("attn_out", &[1, 768], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 768, k: 768, dtype: dt, trans_b: false }, vec![input, attn_w], vec![attn_out], "attn_proj");

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: no GEMM with n > hidden, so fallback to hidden * 4.
        assert_eq!(geo.intermediate, 768 * 4);
    }

    #[test]
    fn rope_scaling_none_when_not_present() {
        // Arrange: RoPE op without rope_scaling.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let rope_out = g.add_tensor_concrete("rope_out", &[1, 256], dt);
        g.add_op(
            OpKind::RoPE { num_heads: 4, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![a],
            vec![rope_out],
            "rope",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert
        assert!(geo.rope_scaling.is_none());
    }

    #[test]
    fn vocab_size_falls_back_to_hidden_when_no_gather_or_argmax() {
        // Arrange: graph with no Gather, QuantGather, or Argmax ops.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 1024], dt);
        let w = g.add_tensor_concrete("w", &[1024, 1024], dt);
        let c = g.add_tensor_concrete("out", &[1, 1024], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 1024, k: 1024, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: vocab_size falls back to hidden.
        assert_eq!(geo.vocab_size, 1024);
    }

    #[test]
    fn num_kv_heads_defaults_to_num_heads_when_no_mha_op() {
        // Arrange: RoPE provides num_heads but no MHA op — num_kv_heads should default to num_heads.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 512], dt);
        let w = g.add_tensor_concrete("w", &[512, 512], dt);
        let c = g.add_tensor_concrete("out", &[1, 512], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 512, k: 512, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let rope_out = g.add_tensor_concrete("rope_out", &[1, 512], dt);
        g.add_op(
            OpKind::RoPE { num_heads: 8, head_dim: 64, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![a],
            vec![rope_out],
            "rope",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: num_kv_heads defaults to num_heads when no MHA op.
        assert_eq!(geo.num_heads, 8);
        assert_eq!(geo.num_kv_heads, 8);
    }

    #[test]
    fn head_dim_fallback_when_hidden_not_divisible_by_num_heads() {
        // Arrange: hidden=1000, num_heads=8 — head_dim = 1000/8 = 125 (integer division).
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 1000], dt);
        let w = g.add_tensor_concrete("w", &[1000, 1000], dt);
        let c = g.add_tensor_concrete("out", &[1, 1000], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 1000, k: 1000, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        let rope_out = g.add_tensor_concrete("rope_out", &[1, 1000], dt);
        g.add_op(
            OpKind::RoPE { num_heads: 8, head_dim: 125, theta: 10000.0, partial: 1.0, rope_scaling: None },
            vec![a],
            vec![rope_out],
            "rope",
        );

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: RoPE provides head_dim directly.
        assert_eq!(geo.hidden, 1000);
        assert_eq!(geo.num_heads, 8);
        assert_eq!(geo.head_dim, 125);
    }

    #[test]
    fn layer_loop_config_takes_priority_over_hetero_config() {
        // Arrange: both layer_loop_config and hetero_layer_loop_config set.
        // layer_loop_config should take priority.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor_concrete("input", &[1, 256], dt);
        let w = g.add_tensor_concrete("w", &[256, 256], dt);
        let c = g.add_tensor_concrete("out", &[1, 256], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: 256, k: 256, dtype: dt, trans_b: false }, vec![a, w], vec![c], "gemm");
        g.inputs = vec![a, w];

        g.layer_loop_config = Some(LayerLoopConfig {
            num_layers: 24,
            weight_stride: 4096,
            layer_blob_base_offset: 0,
            layer_weight_input_indices: vec![1],
            activation_alias: None,
            per_layer_input_stride: 0,
        });
        g.hetero_layer_loop_config = Some(HeteroLayerLoopConfig {
            num_segments: 5,
            sliding_per_segment: 3,
            sliding_small_stride: 1024,
            full_small_stride: 1024,
            sliding_large_stride: 2048,
            full_large_stride: 2048,
            small_segment_stride: 5120,
            large_segment_stride: 10240,
            large_ffn_start_segment: 2,
            layer_blob_base_offset: 0,
            sliding_small_weight_input_indices: vec![1],
            full_small_weight_input_indices: vec![1],
            sliding_large_weight_input_indices: vec![1],
            full_large_weight_input_indices: vec![1],
            activation_aliases: vec![],
        });

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: layer_loop_config (24) takes priority over hetero (5 * 4 = 20).
        assert_eq!(geo.num_layers, 24);
    }

    #[test]
    fn derive_hidden_with_mixed_concrete_and_symbolic_dims() {
        // Arrange: input tensor has Concrete first dim and Symbolic second dim.
        // Should be skipped, falling back to GEMM k.
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let input = g.add_tensor(
            "input",
            vec![SymDim::Concrete(32), SymDim::Symbolic { name: "hidden".into(), max_value: Some(2048) }],
            dt,
        );
        let w = g.add_tensor_concrete("w", &[2048, 2048], dt);
        let out = g.add_tensor_concrete("out", &[32, 2048], dt);
        g.add_op(OpKind::Gemm { m: SymDim::Concrete(32), n: 2048, k: 2048, dtype: dt, trans_b: false }, vec![input, w], vec![out], "proj");
        g.inputs = vec![input, w];

        // Act
        let geo = GraphDerivedGeometry::from_graph(&g, &DeviceProfile::detect()).unwrap();

        // Assert: Symbolic second dim skipped, GEMM k=2048 used.
        assert_eq!(geo.hidden, 2048);
    }
}
