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
        config: &crate::types::ModelConfig,
        max_batch: usize,
    ) -> Self {
        use crate::types::ModelArch;

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
    use crate::types::ModelConfig;

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

    // ── LayerArch ────────────────────────────────────────────────────────

    #[test]
    fn layer_arch_equality() {
        assert_eq!(LayerArch::Decoder, LayerArch::Decoder);
        assert_ne!(LayerArch::Decoder, LayerArch::Encoder);
    }

    #[test]
    fn layer_arch_moe() {
        let moe = LayerArch::DecoderMoE { num_experts: 8, top_k: 2 };
        if let LayerArch::DecoderMoE { num_experts, top_k } = moe {
            assert_eq!(num_experts, 8);
            assert_eq!(top_k, 2);
        } else {
            panic!("expected DecoderMoE");
        }
    }

    #[test]
    fn layer_arch_clone() {
        let arch = LayerArch::DecoderMoE { num_experts: 4, top_k: 1 };
        let cloned = arch.clone();
        assert_eq!(cloned, arch);
    }

    // ── LayerIR dimension methods ────────────────────────────────────────

    #[test]
    fn q_dim_kv_dim_standard() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        // 32 heads × 128 head_dim = 4096
        assert_eq!(ir.q_dim(), 4096);
        // 32 kv_heads × 128 head_dim = 4096 (standard MHA)
        assert_eq!(ir.kv_dim(), 4096);
    }

    #[test]
    fn gqa_groups_single_kv_head() {
        let mut config = ModelConfig::llama_7b();
        config.num_kv_heads = 1;
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.gqa_groups(), 32); // 32 / 1
    }

    #[test]
    fn flops_per_token_positive() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let flops = ir.flops_per_token();
        // LLaMA-7B: hidden=4096, q_dim=4096, kv_dim=4096, inter=11008
        // QKV: 2*4096*(4096+2*4096) = 2*4096*12288 = 100663296
        // O: 2*4096*4096 = 33554432
        // FFN: 6*4096*11008 = 270598144
        assert!(flops > 400_000_000, "flops={flops}, expected >400M");
    }

    #[test]
    fn weight_bytes_positive() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let bytes = ir.weight_bytes();
        // F32 elem: (4096*4096 + 4096*4096 + 4096*4096 + 4096*4096 + 4096*11008 + 4096*11008 + 11008*4096) * 4 + 2*4096*4
        assert!(bytes > 0);
    }

    // ── LayerIR from encoder config ──────────────────────────────────────

    #[test]
    fn encoder_architecture() {
        let mut config = ModelConfig::llama_7b();
        config.arch = crate::types::ModelArch::Gpt2;
        let ir = LayerIR::from_model_config(&config, 4);
        assert_eq!(ir.arch, LayerArch::Encoder);
        assert_eq!(ir.activation, Activation::Gelu);
    }

    #[test]
    fn gemma_activation() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.activation, Activation::GeGlu);
    }

    // ── LayerIR max_batch field ──────────────────────────────────────────

    #[test]
    fn max_batch_propagated() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 8);
        assert_eq!(ir.max_batch, 8);
    }

    // ── LayerIR clone ────────────────────────────────────────────────────

    #[test]
    fn layer_ir_clone() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let cloned = ir.clone();
        assert_eq!(cloned.hidden, ir.hidden);
        assert_eq!(cloned.num_heads, ir.num_heads);
        assert_eq!(cloned.dtype, ir.dtype);
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn layer_ir_debug_format() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let debug = format!("{:?}", ir);
        // Debug should contain key fields
        assert!(debug.contains("4096"), "should contain hidden dim: {debug}");
        assert!(debug.contains("Decoder"), "should contain arch: {debug}");
    }

    #[test]
    fn layer_ir_max_seq_propagated() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.max_seq, config.max_seq_len);
    }

    #[test]
    fn layer_ir_rope_theta_propagated() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.rope_theta, config.rope_theta);
    }

    #[test]
    fn layer_ir_rms_eps_propagated() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.rms_eps, config.norm_eps);
    }

    #[test]
    fn layer_ir_quant_type_propagated() {
        let mut config = ModelConfig::llama_7b();
        config.quant_type = Some(QuantType::Q4_0);
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.quant, Some(QuantType::Q4_0));
    }

    #[test]
    fn layer_ir_dtype_propagated() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.dtype, config.dtype);
    }

    #[test]
    fn layer_ir_partial_rotary_factor_propagated() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.partial_rotary_factor, config.partial_rotary_factor);
    }

    #[test]
    fn layer_arch_moe_debug_format() {
        let moe = LayerArch::DecoderMoE { num_experts: 64, top_k: 8 };
        let debug = format!("{:?}", moe);
        assert!(debug.contains("64"), "should contain num_experts: {debug}");
        assert!(debug.contains("8"), "should contain top_k: {debug}");
    }

    #[test]
    fn flops_per_token_increases_with_hidden() {
        let config_small = ModelConfig::llama_7b();
        let ir_small = LayerIR::from_model_config(&config_small, 1);

        // Manually create a larger variant
        let mut config_big = config_small.clone();
        config_big.hidden_size = 8192;
        config_big.num_heads = 64;
        config_big.intermediate_size = 22016;
        let ir_big = LayerIR::from_model_config(&config_big, 1);

        assert!(ir_big.flops_per_token() > ir_small.flops_per_token(),
            "larger model should have more FLOPs");
    }

    #[test]
    fn weight_bytes_increases_with_hidden() {
        let config_small = ModelConfig::llama_7b();
        let ir_small = LayerIR::from_model_config(&config_small, 1);

        let mut config_big = config_small.clone();
        config_big.hidden_size = 8192;
        config_big.num_heads = 64;
        config_big.intermediate_size = 22016;
        let ir_big = LayerIR::from_model_config(&config_big, 1);

        assert!(ir_big.weight_bytes() > ir_small.weight_bytes(),
            "larger model should have more weight bytes");
    }

    #[test]
    fn qwen_architecture_maps_to_decoder() {
        let config = ModelConfig::qwen_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        assert_eq!(ir.arch, LayerArch::Decoder);
        assert_eq!(ir.activation, Activation::Silu);
    }

    #[test]
    fn gqa_groups_zero_kv_heads_safe() {
        // Edge case: num_kv_heads = 0 should use .max(1) to avoid division by zero
        let mut config = ModelConfig::llama_7b();
        config.num_kv_heads = 0;
        let ir = LayerIR::from_model_config(&config, 1);
        // gqa_groups = num_heads / max(num_kv_heads, 1)
        assert_eq!(ir.gqa_groups(), config.num_heads); // 32 / 1
    }
}
