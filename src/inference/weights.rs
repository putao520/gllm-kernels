//! Model weights — device-side weight storage with layer-indexed access.

use crate::types::{InferenceError, ModelArch, ModelConfig};
use crate::inference::tensor::DeviceTensor;

/// Per-layer weight set for a transformer layer (decoder or encoder).
pub struct LayerWeights {
    /// Attention input norm (RMSNorm / LayerNorm) weight (gamma): [hidden_size]
    pub attn_norm: DeviceTensor,
    /// Attention input norm bias (beta) for LayerNorm: [hidden_size]
    /// Zero-sized for RMSNorm models (Llama etc.), populated for LayerNorm models (GPT-2/BERT).
    pub attn_norm_bias: DeviceTensor,
    /// Q projection: [hidden_size, num_heads * head_dim]
    pub wq: DeviceTensor,
    /// K projection: [hidden_size, num_kv_heads * head_dim]
    pub wk: DeviceTensor,
    /// V projection: [hidden_size, num_kv_heads * head_dim]
    pub wv: DeviceTensor,
    /// Output projection: [num_heads * head_dim, hidden_size]
    pub wo: DeviceTensor,
    /// FFN input norm weight (gamma): [hidden_size]
    pub ffn_norm: DeviceTensor,
    /// FFN input norm bias (beta) for LayerNorm: [hidden_size]
    /// Zero-sized for RMSNorm models, populated for LayerNorm models.
    pub ffn_norm_bias: DeviceTensor,
    /// FFN gate projection: [hidden_size, intermediate_size]
    pub w_gate: DeviceTensor,
    /// FFN up projection: [hidden_size, intermediate_size]
    pub w_up: DeviceTensor,
    /// FFN down projection: [intermediate_size, hidden_size]
    pub w_down: DeviceTensor,
    /// Optional QKV bias: [q_dim + kv_dim + kv_dim] (Qwen has bias, Llama does not)
    pub qkv_bias: Option<DeviceTensor>,
}

/// Complete model weights on device.
pub struct ModelWeights {
    /// Token embedding: [vocab_size, hidden_size]
    pub embedding: DeviceTensor,
    /// Per-layer weights
    pub layers: Vec<LayerWeights>,
    /// Final norm weight: [hidden_size]
    pub final_norm: DeviceTensor,
    /// LM head (output projection): [hidden_size, vocab_size]
    /// May share storage with embedding (tied weights).
    pub lm_head: DeviceTensor,
    /// Model configuration
    pub config: ModelConfig,
}

impl ModelWeights {
    /// Allocate uninitialized weights on CPU for the given model config.
    /// Caller must fill the tensors with actual weight data.
    pub fn alloc_cpu(config: &ModelConfig) -> Result<Self, InferenceError> {
        let h = config.hidden_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let q_dim = config.num_heads * config.head_dim;
        let inter = config.intermediate_size;
        let dtype = config.dtype;

        let embedding = DeviceTensor::alloc_cpu(config.vocab_size * h, dtype)?;

        // LayerNorm models (GPT-2/BERT) need bias vectors; RMSNorm models get zero-sized.
        let needs_norm_bias = matches!(config.arch, ModelArch::Gpt2);
        let norm_bias_size = if needs_norm_bias { h } else { 0 };

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let qkv_bias = if config.has_qkv_bias {
                Some(DeviceTensor::alloc_cpu(q_dim + 2 * kv_dim, dtype)?)
            } else {
                None
            };
            layers.push(LayerWeights {
                attn_norm: DeviceTensor::alloc_cpu(h, dtype)?,
                attn_norm_bias: DeviceTensor::alloc_cpu(norm_bias_size, dtype)?,
                wq: DeviceTensor::alloc_cpu(h * q_dim, dtype)?,
                wk: DeviceTensor::alloc_cpu(h * kv_dim, dtype)?,
                wv: DeviceTensor::alloc_cpu(h * kv_dim, dtype)?,
                wo: DeviceTensor::alloc_cpu(q_dim * h, dtype)?,
                ffn_norm: DeviceTensor::alloc_cpu(h, dtype)?,
                ffn_norm_bias: DeviceTensor::alloc_cpu(norm_bias_size, dtype)?,
                w_gate: DeviceTensor::alloc_cpu(h * inter, dtype)?,
                w_up: DeviceTensor::alloc_cpu(h * inter, dtype)?,
                w_down: DeviceTensor::alloc_cpu(inter * h, dtype)?,
                qkv_bias,
            });
        }

        let final_norm = DeviceTensor::alloc_cpu(h, dtype)?;
        let lm_head = DeviceTensor::alloc_cpu(h * config.vocab_size, dtype)?;

        Ok(ModelWeights {
            embedding,
            layers,
            final_norm,
            lm_head,
            config: config.clone(),
        })
    }

    /// Number of layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DType, ModelArch};

    /// Helper: create a minimal ModelConfig for testing (small dimensions to keep tests fast).
    fn tiny_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 2,
            vocab_size: 256,
            max_seq_len: 512,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    // ---------------------------------------------------------------
    // Test 1: alloc_cpu succeeds and produces correct layer count
    // ---------------------------------------------------------------
    #[test]
    fn alloc_cpu_returns_correct_layer_count() {
        let config = tiny_config();
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        assert_eq!(weights.num_layers(), config.num_layers);
        assert_eq!(weights.num_layers(), 2);
    }

    // ---------------------------------------------------------------
    // Test 2: embedding tensor has correct element count (vocab_size * hidden_size)
    // ---------------------------------------------------------------
    #[test]
    fn embedding_has_correct_size() {
        let config = tiny_config();
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        let expected = config.vocab_size * config.hidden_size;
        assert_eq!(weights.embedding.num_elements(), expected);
        assert_eq!(weights.embedding.dtype(), config.dtype);
    }

    // ---------------------------------------------------------------
    // Test 3: lm_head tensor has correct element count (hidden_size * vocab_size)
    // ---------------------------------------------------------------
    #[test]
    fn lm_head_has_correct_size() {
        let config = tiny_config();
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        let expected = config.hidden_size * config.vocab_size;
        assert_eq!(weights.lm_head.num_elements(), expected);
    }

    // ---------------------------------------------------------------
    // Test 4: final_norm tensor has correct element count (hidden_size)
    // ---------------------------------------------------------------
    #[test]
    fn final_norm_has_correct_size() {
        let config = tiny_config();
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        assert_eq!(weights.final_norm.num_elements(), config.hidden_size);
    }

    // ---------------------------------------------------------------
    // Test 5: per-layer weight dimensions match config (Q/K/V/O projections)
    // ---------------------------------------------------------------
    #[test]
    fn layer_projections_have_correct_sizes() {
        let config = tiny_config();
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        let h = config.hidden_size;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        for (i, layer) in weights.layers.iter().enumerate() {
            assert_eq!(layer.wq.num_elements(), h * q_dim,
                "wq size mismatch at layer {i}");
            assert_eq!(layer.wk.num_elements(), h * kv_dim,
                "wk size mismatch at layer {i}");
            assert_eq!(layer.wv.num_elements(), h * kv_dim,
                "wv size mismatch at layer {i}");
            assert_eq!(layer.wo.num_elements(), q_dim * h,
                "wo size mismatch at layer {i}");
        }
    }

    // ---------------------------------------------------------------
    // Test 6: per-layer FFN weight dimensions match config
    // ---------------------------------------------------------------
    #[test]
    fn layer_ffn_weights_have_correct_sizes() {
        let config = tiny_config();
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        let h = config.hidden_size;
        let inter = config.intermediate_size;

        for (i, layer) in weights.layers.iter().enumerate() {
            assert_eq!(layer.w_gate.num_elements(), h * inter,
                "w_gate size mismatch at layer {i}");
            assert_eq!(layer.w_up.num_elements(), h * inter,
                "w_up size mismatch at layer {i}");
            assert_eq!(layer.w_down.num_elements(), inter * h,
                "w_down size mismatch at layer {i}");
        }
    }

    // ---------------------------------------------------------------
    // Test 7: per-layer norm tensors have correct sizes (RMSNorm: no bias)
    // ---------------------------------------------------------------
    #[test]
    fn layer_norms_rmsnorm_no_bias() {
        let config = tiny_config(); // Llama = RMSNorm
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        let h = config.hidden_size;

        for (i, layer) in weights.layers.iter().enumerate() {
            assert_eq!(layer.attn_norm.num_elements(), h,
                "attn_norm size mismatch at layer {i}");
            assert_eq!(layer.attn_norm_bias.num_elements(), 0,
                "attn_norm_bias should be zero-sized for RMSNorm at layer {i}");
            assert_eq!(layer.ffn_norm.num_elements(), h,
                "ffn_norm size mismatch at layer {i}");
            assert_eq!(layer.ffn_norm_bias.num_elements(), 0,
                "ffn_norm_bias should be zero-sized for RMSNorm at layer {i}");
        }
    }

    // ---------------------------------------------------------------
    // Test 8: GPT-2 arch gets norm bias vectors allocated
    // ---------------------------------------------------------------
    #[test]
    fn layernorm_arch_allocates_norm_bias() {
        let mut config = tiny_config();
        config.arch = ModelArch::Gpt2;
        let h = config.hidden_size;

        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        for (i, layer) in weights.layers.iter().enumerate() {
            assert_eq!(layer.attn_norm_bias.num_elements(), h,
                "GPT-2 attn_norm_bias should have hidden_size elements at layer {i}");
            assert_eq!(layer.ffn_norm_bias.num_elements(), h,
                "GPT-2 ffn_norm_bias should have hidden_size elements at layer {i}");
        }
    }

    // ---------------------------------------------------------------
    // Test 9: qkv_bias is None when has_qkv_bias is false (Llama)
    // ---------------------------------------------------------------
    #[test]
    fn qkv_bias_absent_when_config_false() {
        let config = tiny_config(); // has_qkv_bias: false
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        for (i, layer) in weights.layers.iter().enumerate() {
            assert!(layer.qkv_bias.is_none(),
                "qkv_bias should be None at layer {i} when has_qkv_bias=false");
        }
    }

    // ---------------------------------------------------------------
    // Test 10: qkv_bias is allocated when has_qkv_bias is true (Qwen-style)
    // ---------------------------------------------------------------
    #[test]
    fn qkv_bias_present_when_config_true() {
        let mut config = tiny_config();
        config.has_qkv_bias = true;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let expected_bias_elems = q_dim + 2 * kv_dim;

        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        for (i, layer) in weights.layers.iter().enumerate() {
            let bias = layer.qkv_bias.as_ref()
                .unwrap_or_else(|| panic!("qkv_bias should exist at layer {i}"));
            assert_eq!(bias.num_elements(), expected_bias_elems,
                "qkv_bias size mismatch at layer {i}");
        }
    }

    // ---------------------------------------------------------------
    // Test 11: config is stored correctly in ModelWeights
    // ---------------------------------------------------------------
    #[test]
    fn stored_config_matches_input() {
        let config = tiny_config();
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        assert_eq!(weights.config.hidden_size, config.hidden_size);
        assert_eq!(weights.config.num_layers, config.num_layers);
        assert_eq!(weights.config.vocab_size, config.vocab_size);
        assert_eq!(weights.config.num_heads, config.num_heads);
        assert_eq!(weights.config.num_kv_heads, config.num_kv_heads);
        assert_eq!(weights.config.head_dim, config.head_dim);
        assert_eq!(weights.config.intermediate_size, config.intermediate_size);
        assert_eq!(weights.config.dtype, config.dtype);
    }

    // ---------------------------------------------------------------
    // Test 12: single-layer model allocates exactly one layer
    // ---------------------------------------------------------------
    #[test]
    fn single_layer_model_has_one_layer() {
        let mut config = tiny_config();
        config.num_layers = 1;
        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        assert_eq!(weights.num_layers(), 1);
        assert_eq!(weights.layers.len(), 1);
    }

    // ---------------------------------------------------------------
    // Test 13: GQA config with num_kv_heads < num_heads allocates smaller K/V
    // ---------------------------------------------------------------
    #[test]
    fn gqa_config_allocates_smaller_kv_projections() {
        let mut config = tiny_config();
        config.num_kv_heads = 1; // GQA: 1 KV head, 4 query heads
        let h = config.hidden_size;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        assert!(kv_dim < q_dim, "KV dim should be smaller than Q dim in GQA");

        let weights = ModelWeights::alloc_cpu(&config).unwrap();
        for (i, layer) in weights.layers.iter().enumerate() {
            assert_eq!(layer.wq.num_elements(), h * q_dim,
                "wq should use full q_dim at layer {i}");
            assert_eq!(layer.wk.num_elements(), h * kv_dim,
                "wk should use reduced kv_dim at layer {i}");
            assert_eq!(layer.wv.num_elements(), h * kv_dim,
                "wv should use reduced kv_dim at layer {i}");
            assert_eq!(layer.wo.num_elements(), q_dim * h,
                "wo should use full q_dim at layer {i}");
        }
    }
}
