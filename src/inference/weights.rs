//! Model weights — device-side weight storage with layer-indexed access.

use crate::inference::types::{InferenceError, ModelArch, ModelConfig};
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
