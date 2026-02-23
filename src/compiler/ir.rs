//! Layer IR — structured representation of a transformer layer.
//!
//! Not a general-purpose DAG — this is a transformer-specific IR that
//! captures the computation graph of a single layer for the JIT compiler.

use crate::types::DType;
use crate::quant::QuantType;
use crate::traits::Activation;

/// Architecture of a single transformer layer.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerArch {
    /// Standard decoder: RMSNorm → Attn(QKV+RoPE+GQA+O) → Residual → RMSNorm → FFN(gate+up+SiLU+down) → Residual
    Decoder,
    /// MoE decoder: same as Decoder but FFN replaced by Router → TopK experts
    DecoderMoE { num_experts: usize, top_k: usize },
    /// Encoder: LayerNorm → SelfAttn(QKV+O) → Residual → LayerNorm → FFN(up+GELU+down) → Residual
    Encoder,
}

/// Complete description of a transformer layer for JIT compilation.
///
/// All shape and configuration parameters needed to generate machine code
/// for a single layer. The compiler uses this to determine tiling, fusion
/// opportunities, and buffer layouts.
#[derive(Debug, Clone)]
pub struct LayerIR {
    /// Layer architecture variant
    pub arch: LayerArch,
    /// Hidden dimension
    pub hidden: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (GQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// FFN intermediate dimension
    pub intermediate: usize,
    /// Weight quantization (None = full precision)
    pub quant: Option<QuantType>,
    /// Compute data type
    pub dtype: DType,
    /// RoPE base frequency
    pub rope_theta: f64,
    /// Normalization epsilon
    pub rms_eps: f32,
    /// Maximum batch size this compiled layer supports
    pub max_batch: usize,
    /// Maximum sequence length
    pub max_seq: usize,
    /// Fraction of head_dim to apply RoPE (0.0..=1.0)
    pub partial_rotary_factor: f32,
    /// FFN activation function
    pub activation: Activation,
}

impl LayerIR {
    /// Build a LayerIR from a ModelConfig.
    pub fn from_model_config(
        config: &crate::inference::types::ModelConfig,
        max_batch: usize,
    ) -> Self {
        use crate::inference::types::ModelArch;

        let arch = match config.arch {
            ModelArch::Llama | ModelArch::Mistral | ModelArch::Qwen | ModelArch::Gemma => {
                LayerArch::Decoder
            }
            ModelArch::Gpt2 => LayerArch::Encoder, // GPT-2 uses encoder-style blocks
            ModelArch::Phi => LayerArch::Decoder,
        };

        let activation = match config.arch {
            ModelArch::Gemma => Activation::GeGlu,
            ModelArch::Gpt2 => Activation::Gelu,
            _ => Activation::Silu, // LLaMA, Mistral, Qwen, Phi use SwiGLU
        };

        LayerIR {
            arch,
            hidden: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate: config.intermediate_size,
            quant: config.quant_type,
            dtype: config.dtype,
            rope_theta: config.rope_theta,
            rms_eps: config.norm_eps,
            max_batch,
            max_seq: config.max_seq_len,
            partial_rotary_factor: config.partial_rotary_factor,
            activation,
        }
    }

    /// Q projection output dimension.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// KV projection output dimension.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// GQA group size (num_heads / num_kv_heads).
    #[inline]
    pub fn gqa_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }

    /// Total FLOP count for a single forward pass of this layer (batch=1, seq=1).
    pub fn flops_per_token(&self) -> u64 {
        let h = self.hidden as u64;
        let q = self.q_dim() as u64;
        let kv = self.kv_dim() as u64;
        let inter = self.intermediate as u64;

        // QKV projections: 2*h*(q + 2*kv)
        let qkv_flops = 2 * h * (q + 2 * kv);
        // Output projection: 2*q*h
        let o_flops = 2 * q * h;
        // FFN: gate(2*h*inter) + up(2*h*inter) + down(2*inter*h) = 6*h*inter
        let ffn_flops = 6 * h * inter;
        // Attention: ~4*seq*head_dim per head (approximate for seq=1)
        let attn_flops = 4 * self.head_dim as u64 * self.num_heads as u64;

        qkv_flops + o_flops + ffn_flops + attn_flops
    }

    /// Total weight bytes for this layer.
    pub fn weight_bytes(&self) -> usize {
        let elem = self.dtype.size_bytes();
        let h = self.hidden;
        let q = self.q_dim();
        let kv = self.kv_dim();
        let inter = self.intermediate;

        // QKV + O + gate + up + down + 2 norms
        (h * q + h * kv + h * kv + q * h + h * inter + h * inter + inter * h) * elem
            + 2 * h * elem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::ModelConfig;

    #[test]
    fn test_layer_ir_from_config() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.hidden, 4096);
        assert_eq!(ir.num_heads, 32);
        assert_eq!(ir.q_dim(), 4096);
        assert_eq!(ir.kv_dim(), 4096);
        assert_eq!(ir.gqa_groups(), 1);
        assert!(ir.flops_per_token() > 0);
        assert!(ir.weight_bytes() > 0);
        assert_eq!(ir.arch, LayerArch::Decoder);
    }

    #[test]
    fn test_gqa_ir() {
        let mut config = ModelConfig::llama_7b();
        config.num_kv_heads = 8; // GQA with 4 groups
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.gqa_groups(), 4);
        assert_eq!(ir.kv_dim(), 1024); // 8 * 128
    }
}
