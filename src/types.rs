//! Shared types used across inference and compiler layers.
//!
//! These types were originally defined in `inference::types` but are needed
//! by the compiler layer as well. Extracting them here avoids a dependency
//! from `compiler` → `inference`.

use std::fmt;
use crate::quant::QuantType;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    /// Size in bytes per element.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    /// Element ID matching `Element::ELEM_ID`.
    pub const fn elem_id(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::BF16 => 2,
        }
    }
}

/// Model architecture variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArch {
    /// LLaMA-style: RMSNorm + GQA + SwiGLU FFN
    Llama,
    /// GPT-2/GPT-J style: LayerNorm + MHA + GELU FFN
    Gpt2,
    /// Mistral: LLaMA-like with sliding window attention
    Mistral,
    /// Phi: similar to GPT but with partial rotary
    Phi,
    /// Qwen: LLaMA-like with different normalization
    Qwen,
    /// Gemma: LLaMA-like with GeGLU
    Gemma,
}

/// Complete model configuration for inference.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Architecture variant
    pub arch: ModelArch,
    /// Hidden dimension (embedding size)
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA; equals num_heads for MHA)
    pub num_kv_heads: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// FFN intermediate dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE theta (base frequency)
    pub rope_theta: f64,
    /// RMSNorm / LayerNorm epsilon
    pub norm_eps: f32,
    /// Compute dtype
    pub dtype: DType,
    /// Weight quantization type (None = full precision)
    pub quant_type: Option<QuantType>,
    /// Whether to use interleaved RoPE
    pub rope_interleaved: bool,
    /// Whether QKV projections have bias (Qwen: true, Llama: false)
    pub has_qkv_bias: bool,
    /// Fraction of head_dim to apply RoPE (0.0..=1.0).
    /// 1.0 = full rotary (Llama), 0.5 = partial rotary (Phi).
    pub partial_rotary_factor: f32,
    /// Sliding window size for attention (Mistral-style).
    /// `None` = full causal attention, `Some(W)` = attend to at most the last W positions.
    pub sliding_window: Option<usize>,
}

impl ModelConfig {
    /// Bytes per token for KV cache (both K and V, all layers).
    pub fn kv_cache_bytes_per_token(&self) -> usize {
        let kv_dim = self.num_kv_heads * self.head_dim;
        2 * kv_dim * self.dtype.size_bytes() * self.num_layers
    }

    /// Total weight bytes (approximate, for memory planning).
    pub fn approx_weight_bytes(&self) -> usize {
        let h = self.hidden_size;
        let inter = self.intermediate_size;
        let elem = match self.quant_type {
            Some(qt) => {
                let bits = qt.bits() as usize;
                // rough: bits/8 with ~10% block overhead
                (bits * 110) / (8 * 100)
            }
            None => self.dtype.size_bytes(),
        };
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        // QKV projections: Q is h*q_dim, K and V are each h*kv_dim
        let qkv = h * q_dim + 2 * h * kv_dim;
        let per_layer = (qkv + h * h + 2 * h * inter + inter * h) * elem
            + 2 * h * self.dtype.size_bytes(); // norm weights
        let embedding = self.vocab_size * h * self.dtype.size_bytes();
        per_layer * self.num_layers + 2 * embedding
    }

    /// Create a LLaMA-7B-like config for testing.
    pub fn llama_7b() -> Self {
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            num_layers: 32,
            vocab_size: 32000,
            max_seq_len: 4096,
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

    /// Create a Mistral-7B config for testing.
    pub fn mistral_7b() -> Self {
        ModelConfig {
            arch: ModelArch::Mistral,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            num_layers: 32,
            vocab_size: 32000,
            max_seq_len: 32768,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: Some(4096),
        }
    }

    /// Create a Qwen-7B config for testing.
    pub fn qwen_7b() -> Self {
        ModelConfig {
            arch: ModelArch::Qwen,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            num_layers: 32,
            vocab_size: 151936,
            max_seq_len: 8192,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: true,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    /// Create a Phi-2B-like config for testing (partial rotary).
    pub fn phi_2b() -> Self {
        ModelConfig {
            arch: ModelArch::Phi,
            hidden_size: 2560,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 80,
            intermediate_size: 10240,
            num_layers: 32,
            vocab_size: 51200,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 0.5,
            sliding_window: None,
        }
    }

    /// Create a Gemma-2B-like config for testing (GeGLU activation).
    pub fn gemma_2b() -> Self {
        ModelConfig {
            arch: ModelArch::Gemma,
            hidden_size: 2048,
            num_heads: 8,
            num_kv_heads: 1,
            head_dim: 256,
            intermediate_size: 16384,
            num_layers: 18,
            vocab_size: 256000,
            max_seq_len: 8192,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }
}

/// Errors from inference operations.
#[derive(Debug)]
pub enum InferenceError {
    /// Model configuration is invalid
    InvalidConfig(String),
    /// Memory allocation failed
    OutOfMemory { requested: usize, available: usize },
    /// JIT compilation failed
    CompileError(String),
    /// Runtime execution error
    RuntimeError(String),
    /// Shape mismatch
    ShapeMismatch { expected: String, got: String },
    /// Unsupported operation or configuration
    Unsupported(String),
    /// I/O error (weight loading, cache persistence)
    Io(std::io::Error),
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: need {requested} bytes, have {available}")
            }
            Self::CompileError(s) => write!(f, "compile error: {s}"),
            Self::RuntimeError(s) => write!(f, "runtime error: {s}"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<std::io::Error> for InferenceError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
