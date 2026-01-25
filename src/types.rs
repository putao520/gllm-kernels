//! Types and configuration for attention operations and kernel dispatch.

// use crate::kernel_dispatcher::KernelFloat;

/// Float type identifier for const-time kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatType {
    F32,
    F16,
    BF16,
}

impl FloatType {
    #[inline(always)]
    pub const fn as_u8(self) -> u8 {
        match self {
            FloatType::F32 => 0,
            FloatType::F16 => 1,
            FloatType::BF16 => 2,
        }
    }

    #[inline(always)]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(FloatType::F32),
            1 => Some(FloatType::F16),
            2 => Some(FloatType::BF16),
            _ => None,
        }
    }
}

/// Trait for kernel-compatible floating point types.
/// Implemented for f32, half::f16, and half::bf16. Zero-cost via monomorphization.
pub trait KernelFloat: Copy + Default + Send + Sync + 'static {
    const TYPE_ID: FloatType;
    const IS_F32: bool;
    const IS_F16: bool;

    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl KernelFloat for f32 {
    const TYPE_ID: FloatType = FloatType::F32;
    const IS_F32: bool = true;
    const IS_F16: bool = false;

    #[inline(always)] fn to_f32(self) -> f32 { self }
    #[inline(always)] fn from_f32(v: f32) -> Self { v }
    #[inline(always)] fn zero() -> Self { 0.0 }
    #[inline(always)] fn one() -> Self { 1.0 }
    #[inline(always)] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline(always)] fn exp(self) -> Self { f32::exp(self) }
    #[inline(always)] fn max(self, other: Self) -> Self { f32::max(self, other) }
}

impl KernelFloat for half::f16 {
    const TYPE_ID: FloatType = FloatType::F16;
    const IS_F32: bool = false;
    const IS_F16: bool = true;

    #[inline(always)] fn to_f32(self) -> f32 { half::f16::to_f32(self) }
    #[inline(always)] fn from_f32(v: f32) -> Self { half::f16::from_f32(v) }
    #[inline(always)] fn zero() -> Self { half::f16::ZERO }
    #[inline(always)] fn one() -> Self { half::f16::ONE }
    #[inline(always)] fn sqrt(self) -> Self { half::f16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)] fn exp(self) -> Self { half::f16::from_f32(self.to_f32().exp()) }
    #[inline(always)] fn max(self, other: Self) -> Self { if self.to_f32() >= other.to_f32() { self } else { other } }
}

impl KernelFloat for half::bf16 {
    const TYPE_ID: FloatType = FloatType::BF16;
    const IS_F32: bool = false;
    const IS_F16: bool = false;

    #[inline(always)] fn to_f32(self) -> f32 { half::bf16::to_f32(self) }
    #[inline(always)] fn from_f32(v: f32) -> Self { half::bf16::from_f32(v) }
    #[inline(always)] fn zero() -> Self { half::bf16::ZERO }
    #[inline(always)] fn one() -> Self { half::bf16::ONE }
    #[inline(always)] fn sqrt(self) -> Self { half::bf16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)] fn exp(self) -> Self { half::bf16::from_f32(self.to_f32().exp()) }
    #[inline(always)] fn max(self, other: Self) -> Self { if self.to_f32() >= other.to_f32() { self } else { other } }
}

/// Result type for configuration validation.
pub type ConfigResult<T> = std::result::Result<T, String>;

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

/// Configuration for Flash Attention kernel.
#[derive(Clone, Debug)]
pub struct FlashAttentionConfig {
    /// Query block size for tiling.
    pub block_size_q: usize,
    /// Key/Value block size for tiling.
    pub block_size_kv: usize,
    /// Whether to apply causal mask.
    pub causal: bool,
    /// Use log-space softmax for 2M+ context stability.
    pub use_log_space_softmax: bool,
    /// Use Kahan compensated summation for numerical stability.
    pub use_kahan_accumulator: bool,
    /// Dropout probability (0.0 = no dropout).
    pub dropout_prob: f32,
    /// Optional scale factor (default: 1/sqrt(head_dim)).
    pub scale: Option<f32>,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Sequence length for query.
    pub seq_len_q: usize,
    /// Sequence length for key/value.
    pub seq_len_kv: usize,
    /// Batch size.
    pub batch_size: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 128,
            block_size_kv: 128,
            causal: true,
            use_log_space_softmax: false,
            use_kahan_accumulator: false,
            dropout_prob: 0.0,
            scale: None,
            num_heads: 1,
            head_dim: 64,
            seq_len_q: 1,
            seq_len_kv: 1,
            batch_size: 1,
        }
    }
}

/// Configuration for Paged Attention kernel.
#[derive(Clone, Debug)]
pub struct PagedAttentionConfig {
    /// Number of tokens per page.
    pub page_size: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Block size for computation.
    pub block_size: usize,
    /// Use log-space softmax for 2M+ context stability.
    pub use_log_space_softmax: bool,
    /// Use Kahan compensated summation.
    pub use_kahan_accumulator: bool,
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            page_size: 16,
            num_kv_heads: 1,
            head_dim: 64,
            block_size: 128,
            use_log_space_softmax: true,
            use_kahan_accumulator: true,
        }
    }
}

/// Configuration for Softmax kernel.
#[derive(Clone, Debug)]
pub struct SoftmaxConfig {
    /// Use log-space computation for numerical stability.
    pub use_log_space: bool,
    /// Use Kahan compensated summation.
    pub use_kahan: bool,
    /// Axis along which to compute softmax.
    pub axis: i32,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self {
            use_log_space: true,
            use_kahan: true,
            axis: -1,
        }
    }
}

/// Configuration for EAGLE-3 draft/verify dispatch.
#[derive(Clone, Debug)]
pub struct Eagle3Config {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub fusion_layers: usize,
    pub min_draft_len: usize,
    pub max_draft_len: usize,
    pub confidence_threshold: f32,
}

impl Default for Eagle3Config {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_dim: 768,
            fusion_layers: 4,
            min_draft_len: 1,
            max_draft_len: 8,
            confidence_threshold: 0.5,
        }
    }
}

impl Eagle3Config {
    pub fn fused_dim(&self) -> usize {
        self.hidden_dim * self.fusion_layers
    }

    pub fn validate(&self) -> ConfigResult<()> {
        if self.batch_size == 0 || self.seq_len == 0 {
            return Err("batch_size and seq_len must be > 0".to_string());
        }
        if self.hidden_dim == 0 || self.fusion_layers == 0 {
            return Err("hidden_dim and fusion_layers must be > 0".to_string());
        }
        if self.min_draft_len == 0 || self.max_draft_len < self.min_draft_len {
            return Err("invalid draft length range".to_string());
        }
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err("confidence_threshold must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Result of EAGLE-3 draft generation.
#[derive(Clone, Debug)]
pub struct Eagle3DraftResult {
    pub draft_lengths: Vec<usize>,
    pub draft_tokens: Vec<i32>,
    pub max_draft_len: usize,
}

/// Configuration for EAGLE-3 verification.
#[derive(Clone, Debug)]
pub struct Eagle3VerifyConfig {
    pub batch_size: usize,
    pub draft_len: usize,
    pub vocab_size: usize,
}

/// Result of EAGLE-3 verification.
#[derive(Clone, Debug)]
pub struct Eagle3VerifyResult {
    pub accepted_lengths: Vec<usize>,
    pub acceptance_probs: Vec<f32>,
}

/// Configuration for SpecEE dispatch.
#[derive(Clone, Debug)]
pub struct SpecEEConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub skip_threshold: f32,
    pub exit_threshold: f32,
    pub current_layer: usize,
}

impl Default for SpecEEConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_dim: 768,
            skip_threshold: 0.9,
            exit_threshold: 0.95,
            current_layer: 0,
        }
    }
}

impl SpecEEConfig {
    pub fn validate(&self) -> ConfigResult<()> {
        if self.batch_size == 0 || self.seq_len == 0 || self.hidden_dim == 0 {
            return Err("batch_size, seq_len, hidden_dim must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.skip_threshold) {
            return Err("skip_threshold must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.exit_threshold) {
            return Err("exit_threshold must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Result of SpecEE forward dispatch.
#[derive(Clone, Debug)]
pub struct SpecEEForwardResult {
    pub confidence: Vec<f32>,
    pub skip_decisions: Vec<i32>,
    pub should_exit: Vec<i32>,
    pub exit_layer: Vec<i32>,
}

/// Configuration for Flash Tree-attention dispatch.
#[derive(Clone, Debug)]
pub struct FlashTreeAttentionConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub tree_size: usize,
    pub prefix_len: usize,
    pub head_dim: usize,
    pub scale: Option<f32>,
}

impl Default for FlashTreeAttentionConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 1,
            tree_size: 1,
            prefix_len: 0,
            head_dim: 64,
            scale: None,
        }
    }
}

/// Configuration for INT2 quantization dispatch.
#[derive(Clone, Debug)]
pub struct Int2QuantConfig {
    pub group_size: usize,
}

impl Default for Int2QuantConfig {
    fn default() -> Self {
        Self { group_size: 128 }
    }
}

/// Result of INT2 quantization.
#[derive(Clone, Debug)]
pub struct Int2QuantResult {
    pub quantized: Vec<i8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

/// Compression mode for EvicPress.
#[derive(Clone, Copy, Debug)]
pub enum EvicPressCompression {
    Int8,
    Int2,
}

/// Configuration for EvicPress compression.
#[derive(Clone, Debug)]
pub struct EvicPressCompressConfig {
    pub seq_len: usize,
    pub head_dim: usize,
    pub compression: EvicPressCompression,
}

/// Result of EvicPress compression.
#[derive(Clone, Debug)]
pub enum EvicPressCompressionResult {
    Int8 { data: Vec<i8>, scales: Vec<f32> },
    Int2 { data: Vec<u8>, scales: Vec<f32> },
}

/// Configuration for EvicPress eviction.
#[derive(Clone, Debug)]
pub struct EvicPressEvictConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub recency_weight: f32,
    pub attention_weight: f32,
    pub hot_threshold: f32,
    pub warm_threshold: f32,
    pub cache_pressure: f32,
}

impl Default for EvicPressEvictConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            recency_weight: 0.3,
            attention_weight: 0.7,
            hot_threshold: 0.8,
            warm_threshold: 0.4,
            cache_pressure: 0.0,
        }
    }
}

/// Result of EvicPress eviction decision.
#[derive(Clone, Debug)]
pub struct EvicPressEvictResult {
    pub importance: Vec<f32>,
    pub new_zones: Vec<i32>,
}

/// Configuration for Medusa forward dispatch.
#[derive(Clone, Debug)]
pub struct MedusaConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub top_k: usize,
    pub max_candidates: usize,
    pub temperature: f32,
}

impl Default for MedusaConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 4,
            vocab_size: 32000,
            top_k: 10,
            max_candidates: 64,
            temperature: 1.0,
        }
    }
}

/// Result of Medusa forward dispatch.
#[derive(Clone, Debug)]
pub struct MedusaForwardResult {
    pub candidate_tokens: Vec<i32>,
    pub candidate_probs: Vec<f32>,
    pub num_candidates: Vec<i32>,
}

/// Configuration for Medusa verification.
#[derive(Clone, Debug)]
pub struct MedusaVerifyConfig {
    pub batch_size: usize,
    pub num_candidates: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
}

/// Result of Medusa verification.
#[derive(Clone, Debug)]
pub struct MedusaVerifyResult {
    pub accepted_lengths: Vec<i32>,
    pub best_candidate: Vec<i32>,
}

/// Configuration for prompt cache lookup.
#[derive(Clone, Debug)]
pub struct PromptCacheLookupConfig {
    pub num_entries: usize,
    pub max_cache_len: usize,
    pub hash_seed: u64,
    pub min_match_len: usize,
}

impl Default for PromptCacheLookupConfig {
    fn default() -> Self {
        Self {
            num_entries: 1024,
            max_cache_len: 4096,
            hash_seed: 0x9e3779b97f4a7c15,
            min_match_len: 32,
        }
    }
}

/// Result of prompt cache lookup.
#[derive(Clone, Debug)]
pub struct PromptCacheLookupResult {
    pub best_entry: i32,
    pub match_length: usize,
    pub query_hashes: Vec<u64>,
}

/// Configuration for prompt cache blending.
#[derive(Clone, Debug)]
pub struct PromptCacheBlendConfig {
    pub match_len: usize,
    pub fresh_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub blend_window: usize,
}

impl Default for PromptCacheBlendConfig {
    fn default() -> Self {
        Self {
            match_len: 0,
            fresh_len: 0,
            num_heads: 1,
            head_dim: 64,
            blend_window: 16,
        }
    }
}

/// Configuration for chunked prefill attention dispatch.
#[derive(Clone, Debug)]
pub struct ChunkedPrefillConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub query_len: usize,
    pub chunk_len: usize,
    pub head_dim: usize,
    pub chunk_start: usize,
    pub causal: bool,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 1,
            query_len: 1,
            chunk_len: 1,
            head_dim: 64,
            chunk_start: 0,
            causal: true,
        }
    }
}

/// Result of chunked prefill attention dispatch.
#[derive(Clone, Debug)]
pub struct ChunkedPrefillResult<T: KernelFloat> {
    pub output: Vec<T>,
    pub log_sum_exp: Vec<f32>,
}

/// Matrix multiplication configuration.
#[derive(Clone, Copy, Debug)]
pub struct MatmulConfig {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub transpose_b: bool,
    pub alpha: f32,
    pub beta: f32,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            m: 1,
            k: 1,
            n: 1,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }
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
