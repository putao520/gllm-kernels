//! Kernel-related types shared across backends and ops.

/// Float type identifier for const-time kernel selection (ADR-001).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatType {
    F32,
    F16,
    BF16,
}

impl FloatType {
    /// Convert to u8 for serialization (zero-cost).
    #[inline(always)]
    pub const fn as_u8(self) -> u8 {
        match self {
            FloatType::F32 => 0,
            FloatType::F16 => 1,
            FloatType::BF16 => 2,
        }
    }

    /// Convert from u8 for deserialization.
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
    /// Compile-time type identifier for zero-cost kernel selection.
    const TYPE_ID: FloatType;

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

    #[inline(always)]
    fn to_f32(self) -> f32 { self }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { v }
    #[inline(always)]
    fn zero() -> Self { 0.0 }
    #[inline(always)]
    fn one() -> Self { 1.0 }
    #[inline(always)]
    fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline(always)]
    fn exp(self) -> Self { f32::exp(self) }
    #[inline(always)]
    fn max(self, other: Self) -> Self { f32::max(self, other) }
}

impl KernelFloat for half::f16 {
    const TYPE_ID: FloatType = FloatType::F16;

    #[inline(always)]
    fn to_f32(self) -> f32 { half::f16::to_f32(self) }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { half::f16::from_f32(v) }
    #[inline(always)]
    fn zero() -> Self { half::f16::ZERO }
    #[inline(always)]
    fn one() -> Self { half::f16::ONE }
    #[inline(always)]
    fn sqrt(self) -> Self { half::f16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)]
    fn exp(self) -> Self { half::f16::from_f32(self.to_f32().exp()) }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }
}

impl KernelFloat for half::bf16 {
    const TYPE_ID: FloatType = FloatType::BF16;

    #[inline(always)]
    fn to_f32(self) -> f32 { half::bf16::to_f32(self) }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { half::bf16::from_f32(v) }
    #[inline(always)]
    fn zero() -> Self { half::bf16::ZERO }
    #[inline(always)]
    fn one() -> Self { half::bf16::ONE }
    #[inline(always)]
    fn sqrt(self) -> Self { half::bf16::from_f32(self.to_f32().sqrt()) }
    #[inline(always)]
    fn exp(self) -> Self { half::bf16::from_f32(self.to_f32().exp()) }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }
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
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (for GQA/MQA). If 0, defaults to num_heads.
    pub num_kv_heads: usize,
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
            num_kv_heads: 0, // 0 means same as num_heads
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
            use_log_space_softmax: true, // Enable by default for safety
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
    /// Number of rows (batches) for GPU dispatch. If 0, auto-calculated as 1.
    pub num_rows: usize,
    /// Size of each row (softmax dimension). If 0, auto-calculated as input.len() / num_rows.
    pub row_size: usize,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self {
            use_log_space: true,
            use_kahan: true,
            axis: -1,
            num_rows: 0,
            row_size: 0,
        }
    }
}

impl SoftmaxConfig {
    /// Create config with explicit dimensions for GPU dispatch.
    pub fn with_dims(num_rows: usize, row_size: usize) -> Self {
        Self {
            num_rows,
            row_size,
            ..Default::default()
        }
    }

    /// Get effective num_rows (1 if not specified).
    pub fn effective_num_rows(&self, total_len: usize) -> usize {
        if self.num_rows > 0 {
            self.num_rows
        } else if self.row_size > 0 && total_len > 0 {
            total_len / self.row_size
        } else {
            1
        }
    }

    /// Get effective row_size (total length if not specified).
    pub fn effective_row_size(&self, total_len: usize) -> usize {
        if self.row_size > 0 {
            self.row_size
        } else if self.num_rows > 0 && total_len > 0 {
            total_len / self.num_rows
        } else {
            total_len
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
    pub(crate) fn fused_dim(&self) -> usize {
        self.hidden_dim * self.fusion_layers
    }

    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.batch_size == 0 || self.seq_len == 0 {
            return Err("batch_size and seq_len must be > 0");
        }
        if self.hidden_dim == 0 || self.fusion_layers == 0 {
            return Err("hidden_dim and fusion_layers must be > 0");
        }
        if self.min_draft_len == 0 || self.max_draft_len < self.min_draft_len {
            return Err("invalid draft length range");
        }
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err("confidence_threshold must be in [0, 1]");
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
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.batch_size == 0 || self.seq_len == 0 || self.hidden_dim == 0 {
            return Err("batch_size, seq_len, hidden_dim must be > 0");
        }
        if !(0.0..=1.0).contains(&self.skip_threshold) {
            return Err("skip_threshold must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&self.exit_threshold) {
            return Err("exit_threshold must be in [0, 1]");
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
///
/// Supports C = alpha * A * B + beta * C, with optional transposed storage for A/B.
#[derive(Clone, Debug)]
pub struct MatmulConfig {
    /// M dimension (rows of A, rows of C)
    pub m: usize,
    /// K dimension (cols of A, cols of B^T = rows of B)
    pub k: usize,
    /// N dimension (cols of B^T = cols of B, cols of C)
    pub n: usize,
    /// Whether A is stored transposed (A is [K, M])
    pub transpose_a: bool,
    /// Whether B is stored transposed (common for weight matrices)
    pub transpose_b: bool,
    /// Alpha scalar multiplier (C = alpha * A * B + beta * C)
    pub alpha: f32,
    /// Beta scalar multiplier (0.0 means C is overwritten)
    pub beta: f32,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            m: 1,
            k: 1,
            n: 1,
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

/// Linear layer kernel parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LinearParams {
    pub in_features: u32,
    pub out_features: u32,
    pub has_bias: u32,
    pub padding: u32,
}

// ============================================================================
// L2 Block-Level Operator Configurations (ARCH-GRANULARITY-001)
// ============================================================================

/// Activation function type for FFN blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Activation {
    /// SiLU/Swish activation (used in LLaMA, Mistral)
    #[default]
    SiLU,
    /// GELU activation (used in BERT, GPT)
    GELU,
    /// GELU with exact erf computation
    GELUExact,
    /// ReLU activation
    ReLU,
    /// No activation (identity)
    None,
}

/// Engram fusion hook configuration.
/// Defines how Engram conditional memory integrates with attention.
#[derive(Clone, Debug, Default)]
pub struct EngramHook {
    /// Whether to enable Engram fusion
    pub enabled: bool,
    /// Scaling factor for Engram output
    pub scale: f32,
    /// Bucket index for hash lookup (set by engram_lookup)
    pub bucket_indices: Vec<u64>,
}

/// Configuration for complete attention block (L2 block-level).
///
/// Fuses: QKV projection + RoPE + Softmax + Output projection
/// with optional Engram integration point.
#[derive(Clone, Debug)]
pub struct AttentionBlockConfig {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Number of query attention heads
    pub num_q_heads: usize,
    /// Number of key/value heads (for GQA/MQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Hidden dimension (num_q_heads * head_dim)
    pub hidden_size: usize,
    /// Whether to apply causal mask
    pub causal: bool,
    /// Whether to use RoPE position encoding
    pub use_rope: bool,
    /// RoPE theta parameter (default: 10000.0)
    pub rope_theta: f32,
    /// Position offset for RoPE (for incremental decoding)
    pub position_offset: usize,
    /// Optional scale factor (default: 1/sqrt(head_dim))
    pub scale: Option<f32>,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// Optional Engram fusion hook
    pub engram_hook: Option<EngramHook>,
    /// Use flash attention algorithm
    pub use_flash_attention: bool,
    /// Dropout probability (0.0 = no dropout)
    pub dropout_prob: f32,
}

impl Default for AttentionBlockConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            num_q_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            hidden_size: 4096,
            causal: true,
            use_rope: true,
            rope_theta: 10000.0,
            position_offset: 0,
            scale: None,
            rms_norm_eps: 1e-5,
            engram_hook: None,
            use_flash_attention: true,
            dropout_prob: 0.0,
        }
    }
}

/// Configuration for complete FFN block (L2 block-level).
///
/// Fuses: Gate projection + Up projection + Activation + Down projection
/// Supports both LLaMA-style (Gate*Up) and GPT-style (single Up) FFN.
#[derive(Clone, Debug)]
pub struct FFNBlockConfig {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Hidden dimension (input/output size)
    pub hidden_size: usize,
    /// Intermediate dimension (FFN expansion)
    pub intermediate_size: usize,
    /// Activation function type
    pub activation: Activation,
    /// Whether FFN uses gate projection (LLaMA-style)
    pub use_gate: bool,
    /// Whether to apply bias in linear layers
    pub use_bias: bool,
    /// RMS norm epsilon (for pre-norm)
    pub rms_norm_eps: f32,
}

impl Default for FFNBlockConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_size: 4096,
            intermediate_size: 11008,
            activation: Activation::SiLU,
            use_gate: true,
            use_bias: false,
            rms_norm_eps: 1e-5,
        }
    }
}

/// Configuration for embedding layer (L2 block-level).
///
/// Handles: Token lookup + optional position encoding
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding/hidden dimension
    pub hidden_size: usize,
    /// Maximum sequence length (for position embeddings)
    pub max_seq_len: usize,
    /// Whether to add position embeddings
    pub add_position_embedding: bool,
    /// Padding token ID (for masking)
    pub padding_idx: Option<u32>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            max_seq_len: 4096,
            add_position_embedding: false,
            padding_idx: None,
        }
    }
}

/// Configuration for language model head (L2 block-level).
///
/// Handles: Hidden state -> Vocabulary logits projection
#[derive(Clone, Debug)]
pub struct LMHeadConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length (usually 1 for generation)
    pub seq_len: usize,
    /// Whether to tie weights with input embedding
    pub tie_word_embeddings: bool,
    /// RMS norm epsilon (for final norm)
    pub rms_norm_eps: f32,
}

impl Default for LMHeadConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            batch_size: 1,
            seq_len: 1,
            tie_word_embeddings: true,
            rms_norm_eps: 1e-5,
        }
    }
}

/// Configuration for KV cache update (L2 block-level).
#[derive(Clone, Debug)]
pub struct KVCacheUpdateConfig {
    /// Batch size
    pub batch_size: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Maximum cache length
    pub max_cache_len: usize,
    /// Current position in sequence
    pub position: usize,
    /// Number of new tokens to add
    pub num_new_tokens: usize,
}

impl Default for KVCacheUpdateConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_kv_heads: 32,
            head_dim: 128,
            max_cache_len: 4096,
            position: 0,
            num_new_tokens: 1,
        }
    }
}

/// Configuration for mean pooling (L2 block-level).
///
/// Used for embedding model output aggregation.
#[derive(Clone, Debug)]
pub struct MeanPoolingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Optional attention mask (for variable length sequences)
    pub use_attention_mask: bool,
}

impl Default for MeanPoolingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 512,
            hidden_size: 768,
            use_attention_mask: true,
        }
    }
}

/// Configuration for CLS token pooling (L2 block-level).
///
/// Used for reranker and classification models.
#[derive(Clone, Debug)]
pub struct ClsPoolingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Position of CLS token (usually 0)
    pub cls_position: usize,
}

impl Default for ClsPoolingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            hidden_size: 768,
            cls_position: 0,
        }
    }
}

/// Configuration for L2 normalization (L2 block-level).
///
/// Used for embedding normalization before similarity computation.
#[derive(Clone, Debug)]
pub struct NormalizeConfig {
    /// Batch size
    pub batch_size: usize,
    /// Feature dimension to normalize
    pub dim: usize,
    /// Small epsilon for numerical stability
    pub eps: f32,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            dim: 768,
            eps: 1e-12,
        }
    }
}

/// Quantization format for dequantization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantFormat {
    /// 4-bit quantization (llama.cpp Q4_0)
    #[default]
    Q4_0,
    /// 4-bit quantization with K-quants
    Q4_K,
    /// 8-bit quantization
    Q8_0,
    /// AWQ 4-bit quantization
    AWQ,
}

/// Configuration for dequantization (L2 block-level).
#[derive(Clone, Debug)]
pub struct DequantizeConfig {
    /// Quantization format
    pub format: QuantFormat,
    /// Number of rows (output features)
    pub n: usize,
    /// Number of columns (input features)
    pub k: usize,
    /// Group size for grouped quantization
    pub group_size: usize,
}

impl Default for DequantizeConfig {
    fn default() -> Self {
        Self {
            format: QuantFormat::Q4_0,
            n: 1,
            k: 1,
            group_size: 128,
        }
    }
}

/// Configuration for Engram-Attention fusion (L2 block-level).
///
/// Merges standard attention output with Engram lookup results.
#[derive(Clone, Debug)]
pub struct EngramFuseConfig {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Scaling factor for Engram output
    pub engram_scale: f32,
    /// Scaling factor for attention output
    pub attention_scale: f32,
}

impl Default for EngramFuseConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_size: 4096,
            engram_scale: 1.0,
            attention_scale: 1.0,
        }
    }
}

// ============================================================================
// High-Level Inference API Types (ARCH-ADR-003)
// ============================================================================

/// Transformer layer weights for high-level forward API.
#[derive(Clone, Debug)]
pub struct TransformerLayerWeights<'a> {
    /// Input RMS norm weights [hidden_size]
    pub input_norm: &'a [f32],
    /// Query projection weights [num_q_heads * head_dim, hidden_size]
    pub q_weight: &'a [f32],
    /// Key projection weights [num_kv_heads * head_dim, hidden_size]
    pub k_weight: &'a [f32],
    /// Value projection weights [num_kv_heads * head_dim, hidden_size]
    pub v_weight: &'a [f32],
    /// Output projection weights [hidden_size, num_q_heads * head_dim]
    pub o_weight: &'a [f32],
    /// Post-attention RMS norm weights [hidden_size]
    pub post_attn_norm: &'a [f32],
    /// Gate projection weights (LLaMA-style) [intermediate_size, hidden_size]
    pub gate_weight: Option<&'a [f32]>,
    /// Up projection weights [intermediate_size, hidden_size]
    pub up_weight: &'a [f32],
    /// Down projection weights [hidden_size, intermediate_size]
    pub down_weight: &'a [f32],
}

/// GPU-resident transformer layer weights for zero-copy forward (ARCH-GPU-001).
///
/// Unlike `TransformerLayerWeights` which holds CPU slices, this holds GPU tensors
/// that are uploaded once at model load time and reused for all forward passes.
#[derive(Clone, Debug)]
pub struct TransformerLayerWeightsGpu {
    /// Input RMS norm weights [hidden_size]
    pub input_norm: crate::gpu_types::GpuTensor,
    /// Query projection weights [num_q_heads * head_dim, hidden_size]
    pub q_weight: crate::gpu_types::GpuTensor,
    /// Key projection weights [num_kv_heads * head_dim, hidden_size]
    pub k_weight: crate::gpu_types::GpuTensor,
    /// Value projection weights [num_kv_heads * head_dim, hidden_size]
    pub v_weight: crate::gpu_types::GpuTensor,
    /// Output projection weights [hidden_size, num_q_heads * head_dim]
    pub o_weight: crate::gpu_types::GpuTensor,
    /// Post-attention RMS norm weights [hidden_size]
    pub post_attn_norm: crate::gpu_types::GpuTensor,
    /// Gate projection weights (LLaMA-style) [intermediate_size, hidden_size]
    pub gate_weight: Option<crate::gpu_types::GpuTensor>,
    /// Up projection weights [intermediate_size, hidden_size]
    pub up_weight: crate::gpu_types::GpuTensor,
    /// Down projection weights [hidden_size, intermediate_size]
    pub down_weight: crate::gpu_types::GpuTensor,
    /// Optional RoPE cosine cache [max_seq_len, head_dim/2] (shared across layers)
    pub cos_cache: Option<crate::gpu_types::GpuTensor>,
    /// Optional RoPE sine cache [max_seq_len, head_dim/2] (shared across layers)
    pub sin_cache: Option<crate::gpu_types::GpuTensor>,
}

/// GPU-resident embedding model weights for zero-copy forward (ARCH-GPU-001).
#[derive(Clone, Debug)]
pub struct EmbeddingModelWeightsGpu {
    /// Token embedding weights [vocab_size, hidden_size]
    pub embedding: crate::gpu_types::GpuTensor,
    /// Transformer layer weights
    pub layers: Vec<TransformerLayerWeightsGpu>,
    /// Final layer norm weights [hidden_size]
    pub final_norm: crate::gpu_types::GpuTensor,
}

/// GPU-resident reranker model weights for zero-copy forward (ARCH-ADR-010).
#[derive(Clone, Debug)]
pub struct RerankerModelWeightsGpu {
    /// Token embedding weights [vocab_size, hidden_size]
    pub embedding: crate::gpu_types::GpuTensor,
    /// Transformer layer weights
    pub layers: Vec<TransformerLayerWeightsGpu>,
    /// Final layer norm weights [hidden_size]
    pub final_norm: crate::gpu_types::GpuTensor,
    /// Classifier weights [num_classes, hidden_size] (usually [1, hidden_size] for reranking)
    pub classifier_weight: crate::gpu_types::GpuTensor,
    /// Classifier bias [num_classes] (optional)
    pub classifier_bias: Option<crate::gpu_types::GpuTensor>,
}

/// GPU-resident generator/LLM model weights for zero-copy forward (ARCH-ADR-010).
#[derive(Clone, Debug)]
pub struct GeneratorModelWeightsGpu {
    /// Token embedding weights [vocab_size, hidden_size]
    pub embedding: crate::gpu_types::GpuTensor,
    /// Transformer layer weights
    pub layers: Vec<TransformerLayerWeightsGpu>,
    /// Final layer norm weights [hidden_size]
    pub final_norm: crate::gpu_types::GpuTensor,
    /// LM head weights [vocab_size, hidden_size]
    pub lm_head: crate::gpu_types::GpuTensor,
    /// RoPE cosine cache [max_seq_len, head_dim/2]
    pub cos_cache: crate::gpu_types::GpuTensor,
    /// RoPE sine cache [max_seq_len, head_dim/2]
    pub sin_cache: crate::gpu_types::GpuTensor,
}

/// GPU-resident KV cache for generator models (ARCH-ADR-010: GPU 常驻).
#[derive(Clone, Debug)]
pub struct KVCacheGpu {
    /// Key cache [num_layers, batch, num_kv_heads, max_len, head_dim]
    pub k_cache: crate::gpu_types::GpuTensor,
    /// Value cache [num_layers, batch, num_kv_heads, max_len, head_dim]
    pub v_cache: crate::gpu_types::GpuTensor,
    /// Current sequence length (position)
    pub seq_len: usize,
    /// Maximum cache length
    pub max_len: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

// ============================================================================
// MoE GPU Weights (ARCH-GPU-001)
// ============================================================================

/// GPU-resident single expert FFN weights for MoE.
#[derive(Clone, Debug)]
pub struct ExpertWeightsGpu {
    /// Gate projection [intermediate_size, hidden_size]
    pub gate: crate::gpu_types::GpuTensor,
    /// Up projection [intermediate_size, hidden_size]
    pub up: crate::gpu_types::GpuTensor,
    /// Down projection [hidden_size, intermediate_size]
    pub down: crate::gpu_types::GpuTensor,
}

/// GPU-resident MoE Transformer layer weights for zero-copy forward.
#[derive(Clone, Debug)]
pub struct MoETransformerLayerWeightsGpu {
    /// Input RMS norm weights [hidden_size]
    pub input_norm: crate::gpu_types::GpuTensor,
    /// Query projection weights [num_q_heads * head_dim, hidden_size]
    pub q_weight: crate::gpu_types::GpuTensor,
    /// Key projection weights [num_kv_heads * head_dim, hidden_size]
    pub k_weight: crate::gpu_types::GpuTensor,
    /// Value projection weights [num_kv_heads * head_dim, hidden_size]
    pub v_weight: crate::gpu_types::GpuTensor,
    /// Output projection weights [hidden_size, num_q_heads * head_dim]
    pub o_weight: crate::gpu_types::GpuTensor,
    /// Post-attention RMS norm weights [hidden_size]
    pub post_attn_norm: crate::gpu_types::GpuTensor,
    /// Router gate weights [hidden_size, num_experts] (transposed for GPU matmul)
    pub router_weight: crate::gpu_types::GpuTensor,
    /// Expert FFN weights
    pub experts: Vec<ExpertWeightsGpu>,
    /// Number of experts to activate per token
    pub num_experts_per_tok: usize,
    /// Number of experts total
    pub num_experts: usize,
    /// Optional RoPE cosine cache [max_seq_len, head_dim/2]
    pub cos_cache: Option<crate::gpu_types::GpuTensor>,
    /// Optional RoPE sine cache [max_seq_len, head_dim/2]
    pub sin_cache: Option<crate::gpu_types::GpuTensor>,
}

/// GPU-resident MoE generator model weights for zero-copy forward.
#[derive(Clone, Debug)]
pub struct MoEGeneratorModelWeightsGpu {
    /// Token embedding weights [vocab_size, hidden_size]
    pub embedding: crate::gpu_types::GpuTensor,
    /// MoE Transformer layer weights
    pub layers: Vec<MoETransformerLayerWeightsGpu>,
    /// Final layer norm weights [hidden_size]
    pub final_norm: crate::gpu_types::GpuTensor,
    /// LM head weights [vocab_size, hidden_size]
    pub lm_head: crate::gpu_types::GpuTensor,
    /// RoPE cosine cache [max_seq_len, head_dim/2]
    pub cos_cache: crate::gpu_types::GpuTensor,
    /// RoPE sine cache [max_seq_len, head_dim/2]
    pub sin_cache: crate::gpu_types::GpuTensor,
}

/// Configuration for GPU-native transformer layer forward (ARCH-ADR-010).
#[derive(Clone, Debug)]
pub struct TransformerLayerConfigGpu {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    /// Current position in KV cache
    pub position: usize,
    /// Whether to use SiLU (true) or GELU (false) activation
    pub use_silu: bool,
}

/// Single expert FFN weights for MoE.
#[derive(Clone, Debug)]
pub struct ExpertWeights<'a> {
    /// Gate projection [intermediate_size, hidden_size]
    pub gate: &'a [f32],
    /// Up projection [intermediate_size, hidden_size]
    pub up: &'a [f32],
    /// Down projection [hidden_size, intermediate_size]
    pub down: &'a [f32],
}

/// MoE Transformer layer weights for high-level forward API.
#[derive(Clone, Debug)]
pub struct MoETransformerLayerWeights<'a> {
    /// Input RMS norm weights [hidden_size]
    pub input_norm: &'a [f32],
    /// Query projection weights [num_q_heads * head_dim, hidden_size]
    pub q_weight: &'a [f32],
    /// Key projection weights [num_kv_heads * head_dim, hidden_size]
    pub k_weight: &'a [f32],
    /// Value projection weights [num_kv_heads * head_dim, hidden_size]
    pub v_weight: &'a [f32],
    /// Output projection weights [hidden_size, num_q_heads * head_dim]
    pub o_weight: &'a [f32],
    /// Post-attention RMS norm weights [hidden_size]
    pub post_attn_norm: &'a [f32],
    /// Router gate weights [num_experts, hidden_size]
    pub router_weight: &'a [f32],
    /// Expert FFN weights
    pub experts: Vec<ExpertWeights<'a>>,
    /// Number of experts to activate per token
    pub num_experts_per_tok: usize,
}

/// KV cache state for generation.
#[derive(Debug)]
pub struct KVCacheState<'a> {
    /// Key cache [num_layers, batch, num_kv_heads, max_len, head_dim]
    pub k_cache: &'a mut [f32],
    /// Value cache [num_layers, batch, num_kv_heads, max_len, head_dim]
    pub v_cache: &'a mut [f32],
    /// Current sequence length (position)
    pub seq_len: usize,
    /// Maximum cache length
    pub max_len: usize,
}

/// Configuration for generator forward (dense model).
#[derive(Clone, Debug)]
pub struct GeneratorForwardConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    /// Maximum sequence length for KV cache allocation
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub use_rope: bool,
    pub activation: Activation,
    pub position_offset: usize,
    pub final_logit_softcapping: Option<f32>,
}

impl Default for GeneratorForwardConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_size: 4096,
            num_layers: 32,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            vocab_size: 32000,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            use_rope: true,
            activation: Activation::SiLU,
            position_offset: 0,
            final_logit_softcapping: None,
        }
    }
}

/// Configuration for MoE generator forward.
#[derive(Clone, Debug)]
pub struct MoEGeneratorForwardConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    /// Expert FFN intermediate size (for MoE models, typically smaller than intermediate_size)
    pub moe_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Maximum sequence length for KV cache allocation
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub use_rope: bool,
    pub activation: Activation,
    pub position_offset: usize,
}

impl Default for MoEGeneratorForwardConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 1,
            hidden_size: 4096,
            num_layers: 32,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            moe_intermediate_size: None,
            vocab_size: 32000,
            num_experts: 8,
            num_experts_per_tok: 2,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            use_rope: true,
            activation: Activation::SiLU,
            position_offset: 0,
        }
    }
}

/// Architecture type for embedding models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingArchType {
    /// Decoder-style (Qwen3, LLaMA): Pre-LN + RMSNorm + RoPE + SwiGLU
    #[default]
    Decoder,
    /// BERT-style: Post-LN + LayerNorm + AbsolutePos + GELU FFN
    Bert,
}

pub struct EmbeddingForwardConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub activation: Activation,
    /// Pooling type: "mean", "cls", or "last"
    pub pooling: PoolingType,
    /// Whether to normalize output embeddings
    pub normalize: bool,
    /// Architecture type (BERT vs Decoder)
    pub arch_type: EmbeddingArchType,
    /// Max position embeddings (for BERT absolute position encoding)
    pub max_position_embeddings: usize,
}

/// Pooling type for embedding models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoolingType {
    #[default]
    Mean,
    Cls,
    Last,
}

impl Default for EmbeddingForwardConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 512,
            hidden_size: 768,
            num_layers: 12,
            num_q_heads: 12,
            num_kv_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            vocab_size: 30522,
            rms_norm_eps: 1e-5,
            activation: Activation::GELU,
            pooling: PoolingType::Mean,
            normalize: true,
            arch_type: EmbeddingArchType::default(),
            max_position_embeddings: 512,
        }
    }
}

// ============================================================================
// GPU-Native Logits for Zero-Copy Sampling (ARCH-PERF-001)
// ============================================================================

/// Logits tensor that can reside on CPU or GPU.
///
/// This abstraction enables zero-copy sampling by keeping logits on GPU
/// and only transferring the sampled token ID(s) back to CPU.
///
/// For a 32k vocabulary, this reduces per-token transfer from 128KB to 4 bytes.
#[derive(Clone, Debug)]
pub enum LogitsTensor {
    /// CPU-resident logits (Vec<f32>)
    Cpu(Vec<f32>),
    /// GPU-resident logits (GpuTensor)
    Gpu(crate::gpu_types::GpuTensor),
}

impl LogitsTensor {
    /// Create CPU logits from a Vec.
    pub fn from_cpu(data: Vec<f32>) -> Self {
        LogitsTensor::Cpu(data)
    }

    /// Create GPU logits from a GpuTensor.
    pub fn from_gpu(tensor: crate::gpu_types::GpuTensor) -> Self {
        LogitsTensor::Gpu(tensor)
    }

    /// Check if logits are on GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self, LogitsTensor::Gpu(_))
    }

    /// Check if logits are on CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self, LogitsTensor::Cpu(_))
    }

    /// Get CPU data if available (returns None for GPU tensors).
    pub fn as_cpu(&self) -> Option<&[f32]> {
        match self {
            LogitsTensor::Cpu(data) => Some(data),
            LogitsTensor::Gpu(_) => None,
        }
    }

    /// Get GPU tensor if available (returns None for CPU tensors).
    pub fn as_gpu(&self) -> Option<&crate::gpu_types::GpuTensor> {
        match self {
            LogitsTensor::Cpu(_) => None,
            LogitsTensor::Gpu(tensor) => Some(tensor),
        }
    }

    /// Get the vocabulary size (number of logits).
    pub fn vocab_size(&self) -> usize {
        match self {
            LogitsTensor::Cpu(data) => data.len(),
            LogitsTensor::Gpu(tensor) => tensor.shape.iter().product(),
        }
    }
}

/// Configuration for rerank forward.
#[derive(Clone, Debug)]
pub struct RerankForwardConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub activation: Activation,
}

impl Default for RerankForwardConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 512,
            hidden_size: 768,
            num_layers: 12,
            num_q_heads: 12,
            num_kv_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            vocab_size: 30522,
            rms_norm_eps: 1e-5,
            activation: Activation::GELU,
        }
    }
}
