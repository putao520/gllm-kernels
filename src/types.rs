//! Types and configuration for attention operations and kernel dispatch.

pub use crate::kernel_types::{
    FloatType, KernelFloat,
    FlashAttentionConfig, PagedAttentionConfig, SoftmaxConfig,
    Eagle3Config, Eagle3DraftResult, Eagle3VerifyConfig, Eagle3VerifyResult,
    SpecEEConfig, SpecEEForwardResult,
    FlashTreeAttentionConfig,
    Int2QuantConfig, Int2QuantResult,
    EvicPressCompression, EvicPressCompressConfig, EvicPressCompressionResult,
    EvicPressEvictConfig, EvicPressEvictResult,
    MedusaConfig, MedusaForwardResult, MedusaVerifyConfig, MedusaVerifyResult,
    PromptCacheLookupConfig, PromptCacheLookupResult, PromptCacheBlendConfig,
    ChunkedPrefillConfig, ChunkedPrefillResult,
    MatmulConfig,
};

/// Precision mode for kernel computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelPrecision {
    /// FP32 throughout (highest precision, slowest).
    FP32,
    /// FP16 compute with FP32 accumulation (balanced).
    #[default]
    FP16,
    /// BF16 compute with FP32 accumulation (best for training).
    BF16,
    /// FP8 compute with FP16 accumulation (fastest, Hopper+).
    FP8,
}

/// Configuration for Gpu Rerank.
#[derive(Clone, Debug)]
pub struct GpuRerankConfig {
    pub dim: usize,
    pub binary_k: usize,
    pub int8_k: usize,
}

impl Default for GpuRerankConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            binary_k: 100,
            int8_k: 10,
        }
    }
}

/// Result from rerank pipeline stage.
#[derive(Clone, Debug)]
pub struct GpuRerankStageResult {
    /// Vector indices (sorted by score).
    pub indices: Vec<u32>,
    /// Corresponding scores.
    pub scores: Vec<f32>,
}

/// Configuration for fused MoE forward pass.
#[derive(Clone, Copy, Debug)]
pub struct MoEForwardConfig {
    /// Hidden dimension of input/output.
    pub hidden_size: usize,
    /// Intermediate dimension of FFN.
    pub intermediate_size: usize,
    /// Number of tokens being processed.
    pub num_tokens: usize,
    /// Number of experts selected per token (top-k).
    pub top_k: usize,
    /// Total number of experts in the model.
    pub num_experts: usize,
}

impl MoEForwardConfig {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_tokens: usize,
        top_k: usize,
        num_experts: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_tokens,
            top_k,
            num_experts,
        }
    }
}

/// Configuration for GPU MoE routing.
#[derive(Clone, Copy, Debug)]
pub struct MoERoutingGpuConfig {
    pub num_tokens: usize,
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
}
