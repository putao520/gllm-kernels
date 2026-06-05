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
    /// Unsigned 8-bit integer (stub for P4/P5 features).
    U8,
    /// FP8 E4M3 (OCP standard, NVIDIA/AMD).
    F8E4M3,
    /// FP8 E5M2 (OCP standard, NVIDIA/AMD).
    F8E5M2,
    /// FP6 E3M2 (AMD CDNA4).
    F6E3M2,
    /// FP6 E2M3 (AMD CDNA4).
    F6E2M3,
    /// FP4 E2M1 (NVIDIA Blackwell / AMD CDNA4).
    F4E2M1,
}

impl DType {
    /// Size in bytes per element.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::U8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            // Sub-byte types: return 1 as minimum addressable unit.
            // Actual packing handled by quantization codegen.
            Self::F6E3M2 | Self::F6E2M3 | Self::F4E2M1 => 1,
        }
    }

    /// Element ID matching `Element::ELEM_ID`.
    pub const fn elem_id(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::BF16 => 2,
            Self::U8 => 3,
            Self::F8E4M3 => 4,
            Self::F8E5M2 => 5,
            Self::F6E3M2 => 6,
            Self::F6E2M3 => 7,
            Self::F4E2M1 => 8,
        }
    }

    /// GPU PTX/HIP/Metal 类型名。用于 lowering 层生成类型后缀。
    /// 禁止静默回退，未知类型返回 Err。
    pub fn gpu_type_name(self) -> Result<&'static str, ()> {
        match self {
            Self::F32 => Ok("f32"),
            Self::F16 => Ok("f16"),
            Self::BF16 => Ok("bf16"),
            Self::U8 => Ok("u8"),
            Self::F8E4M3 => Ok("e4m3"),
            Self::F8E5M2 => Ok("e5m2"),
            Self::F4E2M1 => Ok("e2m1"),
            Self::F6E3M2 | Self::F6E2M3 => Err(()),
        }
    }

    /// Convert to QuantPrecision for JIT dtype propagation.
    pub fn to_quant_precision(self) -> crate::compiler::trace::QuantPrecision {
        use crate::compiler::trace::QuantPrecision;
        match self {
            Self::F32 => QuantPrecision::F32,
            Self::F16 => QuantPrecision::F16,
            Self::BF16 => QuantPrecision::BF16,
            Self::F8E4M3 => QuantPrecision::FP8E4M3,
            Self::F8E5M2 => QuantPrecision::FP8E5M2,
            Self::F6E3M2 => QuantPrecision::FP6E3M2,
            Self::F6E2M3 => QuantPrecision::FP6E2M3,
            Self::F4E2M1 => QuantPrecision::FP4E2M1,
            Self::U8 => QuantPrecision::INT8,
        }
    }

    // ── GPU codegen helpers ──

    /// PTX type suffix for arithmetic instructions: `.f32` / `.f16` / `.bf16`
    pub const fn ptx_type(self) -> &'static str {
        match self {
            Self::F32 => ".f32",
            Self::F16 => ".f16",
            Self::BF16 => ".bf16",
            Self::U8 | Self::F8E4M3 | Self::F8E5M2 => ".b8",
            Self::F6E3M2 | Self::F6E2M3 | Self::F4E2M1 => ".b8", // sub-byte packed as bytes
        }
    }

    /// PTX register type: `.f32` / `.f16` / `.b16` (BF16 uses .b16 in registers)
    pub const fn ptx_reg_type(self) -> &'static str {
        match self {
            Self::F32 => ".f32",
            Self::F16 => ".f16",
            Self::BF16 => ".b16",
            Self::U8 | Self::F8E4M3 | Self::F8E5M2 => ".b8",
            Self::F6E3M2 | Self::F6E2M3 | Self::F4E2M1 => ".b8",
        }
    }

    /// PTX load/store type suffix: `.f32` / `.f16` / `.b16`
    pub const fn ptx_ld_type(self) -> &'static str {
        match self {
            Self::F32 => ".f32",
            Self::F16 => ".f16",
            Self::BF16 => ".b16",
            Self::U8 | Self::F8E4M3 | Self::F8E5M2 => ".u8",
            Self::F6E3M2 | Self::F6E2M3 | Self::F4E2M1 => ".u8",
        }
    }

    /// HIP/CUDA C++ type name: `float` / `half` / `__nv_bfloat16`
    pub const fn hip_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 => "half",
            Self::BF16 => "__nv_bfloat16",
            Self::U8 => "uint8_t",
            Self::F8E4M3 => "__nv_fp8_e4m3",
            Self::F8E5M2 => "__nv_fp8_e5m2",
            Self::F6E3M2 | Self::F6E2M3 | Self::F4E2M1 => "uint8_t", // packed
        }
    }

    /// Metal Shading Language type name: `float` / `half` / `bfloat`
    pub const fn msl_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 => "half",
            Self::BF16 => "bfloat",
            Self::U8 | Self::F8E4M3 | Self::F8E5M2 => "uchar",
            Self::F6E3M2 | Self::F6E2M3 | Self::F4E2M1 => "uchar",
        }
    }

    /// PTX arithmetic suffix for fma/add/mul: `.f32` / `.f16` / `.bf16`
    /// Note: BF16 arithmetic requires SM >= 80 (bf16 instructions).
    pub const fn ptx_arith_type(self) -> &'static str {
        match self {
            Self::F32 => ".f32",
            Self::F16 => ".f16",
            Self::BF16 => ".bf16",
            Self::U8 | Self::F8E4M3 | Self::F8E5M2 => ".b8",
            Self::F6E3M2 | Self::F6E2M3 | Self::F4E2M1 => ".b8",
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

/// Structured compiler error — replaces `String` in codegen/fusion/compiler layers.
///
/// Categorizes compilation failures so that downstream (gllm) can distinguish
/// between register pressure, unsupported ISA, invalid graph structure, etc.
#[derive(Debug, Clone)]
pub enum CompilerError {
    /// Register overflow: too many accumulators/scratch registers needed
    RegisterOverflow { needed: usize, available: usize, context: String },
    /// Unsupported DType for the target ISA
    UnsupportedDType { dtype: DType, isa: String },
    /// Invalid graph structure (cycle, disconnected output, etc.)
    InvalidGraph(String),
    /// Codegen constraint violation (alignment, tile size, etc.)
    CodegenViolation(String),
    /// Feature not available (e.g., jit-x86 feature flag disabled)
    FeatureDisabled(String),
    /// Generic internal error (should be minimized over time)
    Internal(String),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegisterOverflow { needed, available, context } => {
                write!(f, "register overflow: need {needed}, have {available} ({context})")
            }
            Self::UnsupportedDType { dtype, isa } => {
                write!(f, "unsupported dtype {dtype:?} for ISA {isa}")
            }
            Self::InvalidGraph(s) => write!(f, "invalid graph: {s}"),
            Self::CodegenViolation(s) => write!(f, "codegen violation: {s}"),
            Self::FeatureDisabled(s) => write!(f, "feature disabled: {s}"),
            Self::Internal(s) => write!(f, "internal: {s}"),
        }
    }
}

impl std::error::Error for CompilerError {}

impl From<String> for CompilerError {
    fn from(s: String) -> Self {
        Self::Internal(s)
    }
}

impl From<&str> for CompilerError {
    fn from(s: &str) -> Self {
        Self::Internal(s.to_string())
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
    CompileError(CompilerError),
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
            Self::CompileError(e) => write!(f, "compile error: {e}"),
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

impl From<CompilerError> for InferenceError {
    fn from(e: CompilerError) -> Self {
        Self::CompileError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DType::size_bytes ──

    #[test]
    fn dtype_size_bytes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::F8E4M3.size_bytes(), 1);
        assert_eq!(DType::F8E5M2.size_bytes(), 1);
        assert_eq!(DType::F6E3M2.size_bytes(), 1);
        assert_eq!(DType::F6E2M3.size_bytes(), 1);
        assert_eq!(DType::F4E2M1.size_bytes(), 1);
    }

    // ── DType::elem_id uniqueness ──

    #[test]
    fn dtype_elem_id_unique() {
        let ids: Vec<u8> = vec![
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ].iter().map(|d| d.elem_id()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len(), "elem_id values must be unique");
    }

    // ── DType::gpu_type_name ──

    #[test]
    fn dtype_gpu_type_name() {
        assert_eq!(DType::F32.gpu_type_name(), Ok("f32"));
        assert_eq!(DType::F16.gpu_type_name(), Ok("f16"));
        assert_eq!(DType::BF16.gpu_type_name(), Ok("bf16"));
        assert_eq!(DType::F8E4M3.gpu_type_name(), Ok("e4m3"));
        assert_eq!(DType::F8E5M2.gpu_type_name(), Ok("e5m2"));
        assert_eq!(DType::F4E2M1.gpu_type_name(), Ok("e2m1"));
        assert!(DType::F6E3M2.gpu_type_name().is_err());
        assert!(DType::F6E2M3.gpu_type_name().is_err());
    }

    // ── DType::to_quant_precision roundtrip ──

    #[test]
    fn dtype_to_quant_precision() {
        use crate::compiler::trace::QuantPrecision;
        assert_eq!(DType::F32.to_quant_precision(), QuantPrecision::F32);
        assert_eq!(DType::F16.to_quant_precision(), QuantPrecision::F16);
        assert_eq!(DType::BF16.to_quant_precision(), QuantPrecision::BF16);
        assert_eq!(DType::F8E4M3.to_quant_precision(), QuantPrecision::FP8E4M3);
        assert_eq!(DType::F8E5M2.to_quant_precision(), QuantPrecision::FP8E5M2);
        assert_eq!(DType::F6E3M2.to_quant_precision(), QuantPrecision::FP6E3M2);
        assert_eq!(DType::F6E2M3.to_quant_precision(), QuantPrecision::FP6E2M3);
        assert_eq!(DType::F4E2M1.to_quant_precision(), QuantPrecision::FP4E2M1);
        assert_eq!(DType::U8.to_quant_precision(), QuantPrecision::INT8);
    }

    // ── DType PTX/Metal/HIP type strings ──

    #[test]
    fn dtype_ptx_type() {
        assert_eq!(DType::F32.ptx_type(), ".f32");
        assert_eq!(DType::BF16.ptx_reg_type(), ".b16");
        assert_eq!(DType::F16.ptx_ld_type(), ".f16");
        assert_eq!(DType::F32.ptx_arith_type(), ".f32");
    }

    #[test]
    fn dtype_hip_msl_types() {
        assert_eq!(DType::F32.hip_type(), "float");
        assert_eq!(DType::F16.hip_type(), "half");
        assert_eq!(DType::BF16.hip_type(), "__nv_bfloat16");
        assert_eq!(DType::F32.msl_type(), "float");
        assert_eq!(DType::F16.msl_type(), "half");
        assert_eq!(DType::BF16.msl_type(), "bfloat");
    }

    // ── DType equality ──

    #[test]
    fn dtype_equality() {
        assert_eq!(DType::F32, DType::F32);
        assert_ne!(DType::F16, DType::BF16);
        assert_ne!(DType::F8E4M3, DType::F8E5M2);
    }

    // ── ModelArch variants ──

    #[test]
    fn model_arch_equality() {
        assert_eq!(ModelArch::Llama, ModelArch::Llama);
        assert_ne!(ModelArch::Llama, ModelArch::Mistral);
        assert_ne!(ModelArch::Phi, ModelArch::Gemma);
    }

    // ── ModelConfig presets ──

    #[test]
    fn llama_7b_config() {
        let cfg = ModelConfig::llama_7b();
        assert_eq!(cfg.arch, ModelArch::Llama);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 32);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.max_seq_len, 4096);
        assert!(!cfg.rope_interleaved);
        assert!(!cfg.has_qkv_bias);
        assert_eq!(cfg.partial_rotary_factor, 1.0);
        assert!(cfg.sliding_window.is_none());
    }

    #[test]
    fn mistral_7b_config() {
        let cfg = ModelConfig::mistral_7b();
        assert_eq!(cfg.arch, ModelArch::Mistral);
        assert_eq!(cfg.num_kv_heads, 8); // GQA
        assert_eq!(cfg.sliding_window, Some(4096));
    }

    #[test]
    fn qwen_7b_config() {
        let cfg = ModelConfig::qwen_7b();
        assert_eq!(cfg.arch, ModelArch::Qwen);
        assert!(cfg.has_qkv_bias);
        assert_eq!(cfg.vocab_size, 151936);
    }

    #[test]
    fn phi_2b_config() {
        let cfg = ModelConfig::phi_2b();
        assert_eq!(cfg.arch, ModelArch::Phi);
        assert_eq!(cfg.partial_rotary_factor, 0.5);
        assert_eq!(cfg.head_dim, 80);
    }

    #[test]
    fn gemma_2b_config() {
        let cfg = ModelConfig::gemma_2b();
        assert_eq!(cfg.arch, ModelArch::Gemma);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.head_dim, 256);
    }

    // ── ModelConfig::kv_cache_bytes_per_token ──

    #[test]
    fn kv_cache_bytes_llama_7b() {
        let cfg = ModelConfig::llama_7b();
        // 2 * (32 * 128) * 4 * 32 = 2 * 4096 * 4 * 32 = 1,048,576
        assert_eq!(cfg.kv_cache_bytes_per_token(), 1_048_576);
    }

    #[test]
    fn kv_cache_bytes_mistral_7b() {
        let cfg = ModelConfig::mistral_7b();
        // 2 * (8 * 128) * 4 * 32 = 2 * 1024 * 4 * 32 = 262,144
        assert_eq!(cfg.kv_cache_bytes_per_token(), 262_144);
    }

    // ── ModelConfig::approx_weight_bytes ──

    #[test]
    fn approx_weight_bytes_positive() {
        let cfg = ModelConfig::llama_7b();
        let bytes = cfg.approx_weight_bytes();
        assert!(bytes > 0, "weight bytes should be positive");
        // LLaMA-7B is ~13GB in FP32, check order of magnitude
        assert!(bytes > 1_000_000_000, "LLaMA-7B FP32 should be >1GB, got {bytes}");
    }

    // ── CompilerError Display ──

    #[test]
    fn compiler_error_display() {
        let e = CompilerError::RegisterOverflow { needed: 32, available: 16, context: "GEMM acc".into() };
        let msg = format!("{e}");
        assert!(msg.contains("32") && msg.contains("16") && msg.contains("GEMM acc"));

        let e = CompilerError::UnsupportedDType { dtype: DType::F6E3M2, isa: "AVX2".into() };
        let msg = format!("{e}");
        assert!(msg.contains("F6E3M2") && msg.contains("AVX2"));

        let e = CompilerError::InvalidGraph("cycle".into());
        assert!(format!("{e}").contains("cycle"));

        let e = CompilerError::CodegenViolation("alignment".into());
        assert!(format!("{e}").contains("alignment"));
    }

    #[test]
    fn compiler_error_from_string() {
        let e: CompilerError = "test error".into();
        assert!(matches!(e, CompilerError::Internal(s) if s == "test error"));

        let e: CompilerError = String::from("test").into();
        assert!(matches!(e, CompilerError::Internal(_)));
    }

    // ── InferenceError Display ──

    #[test]
    fn inference_error_display() {
        let e = InferenceError::OutOfMemory { requested: 1024, available: 512 };
        let msg = format!("{e}");
        assert!(msg.contains("1024") && msg.contains("512"));

        let e = InferenceError::ShapeMismatch { expected: "[2,3]".into(), got: "[4,5]".into() };
        let msg = format!("{e}");
        assert!(msg.contains("[2,3]") && msg.contains("[4,5]"));
    }

    #[test]
    fn inference_error_from_compiler() {
        let ce = CompilerError::Internal("test".into());
        let ie: InferenceError = ce.into();
        assert!(matches!(ie, InferenceError::CompileError(_)));
    }

    // ── Additional tests for uncovered logic paths ──

    #[test]
    fn dtype_elem_id_range_and_ordering() {
        // elem_id values should be sequential 0..=8.
        let ids: Vec<u8> = vec![
            DType::F32, DType::F16, DType::BF16, DType::U8,
            DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1,
        ].iter().map(|d| d.elem_id()).collect();
        assert_eq!(*ids.iter().min().unwrap(), 0);
        assert_eq!(*ids.iter().max().unwrap(), 8);
    }

    #[test]
    fn dtype_ptx_type_all_variants() {
        assert_eq!(DType::U8.ptx_type(), ".b8");
        assert_eq!(DType::F8E4M3.ptx_type(), ".b8");
        assert_eq!(DType::F8E5M2.ptx_type(), ".b8");
        assert_eq!(DType::F6E3M2.ptx_type(), ".b8");
        assert_eq!(DType::F6E2M3.ptx_type(), ".b8");
        assert_eq!(DType::F4E2M1.ptx_type(), ".b8");
        assert_eq!(DType::F16.ptx_type(), ".f16");
        assert_eq!(DType::BF16.ptx_type(), ".bf16");
    }

    #[test]
    fn dtype_ptx_reg_type_all_variants() {
        assert_eq!(DType::F32.ptx_reg_type(), ".f32");
        assert_eq!(DType::F16.ptx_reg_type(), ".f16");
        assert_eq!(DType::BF16.ptx_reg_type(), ".b16");
        assert_eq!(DType::U8.ptx_reg_type(), ".b8");
        assert_eq!(DType::F8E4M3.ptx_reg_type(), ".b8");
        assert_eq!(DType::F6E3M2.ptx_reg_type(), ".b8");
        assert_eq!(DType::F4E2M1.ptx_reg_type(), ".b8");
    }

    #[test]
    fn dtype_ptx_ld_type_all_variants() {
        assert_eq!(DType::F32.ptx_ld_type(), ".f32");
        assert_eq!(DType::F16.ptx_ld_type(), ".f16");
        assert_eq!(DType::BF16.ptx_ld_type(), ".b16");
        assert_eq!(DType::U8.ptx_ld_type(), ".u8");
        assert_eq!(DType::F8E4M3.ptx_ld_type(), ".u8");
        assert_eq!(DType::F6E3M2.ptx_ld_type(), ".u8");
        assert_eq!(DType::F4E2M1.ptx_ld_type(), ".u8");
    }

    #[test]
    fn dtype_hip_type_all_variants() {
        assert_eq!(DType::U8.hip_type(), "uint8_t");
        assert_eq!(DType::F8E4M3.hip_type(), "__nv_fp8_e4m3");
        assert_eq!(DType::F8E5M2.hip_type(), "__nv_fp8_e5m2");
        assert_eq!(DType::F6E3M2.hip_type(), "uint8_t");
        assert_eq!(DType::F6E2M3.hip_type(), "uint8_t");
        assert_eq!(DType::F4E2M1.hip_type(), "uint8_t");
    }

    #[test]
    fn dtype_msl_type_all_variants() {
        assert_eq!(DType::U8.msl_type(), "uchar");
        assert_eq!(DType::F8E4M3.msl_type(), "uchar");
        assert_eq!(DType::F8E5M2.msl_type(), "uchar");
        assert_eq!(DType::F6E3M2.msl_type(), "uchar");
        assert_eq!(DType::F6E2M3.msl_type(), "uchar");
        assert_eq!(DType::F4E2M1.msl_type(), "uchar");
    }

    #[test]
    fn dtype_ptx_arith_type_all_variants() {
        assert_eq!(DType::F16.ptx_arith_type(), ".f16");
        assert_eq!(DType::BF16.ptx_arith_type(), ".bf16");
        assert_eq!(DType::U8.ptx_arith_type(), ".b8");
        assert_eq!(DType::F8E4M3.ptx_arith_type(), ".b8");
        assert_eq!(DType::F6E3M2.ptx_arith_type(), ".b8");
        assert_eq!(DType::F4E2M1.ptx_arith_type(), ".b8");
    }

    #[test]
    fn gpu_type_name_u8_ok() {
        assert_eq!(DType::U8.gpu_type_name(), Ok("u8"));
    }

    #[test]
    fn compiler_error_feature_disabled_display() {
        let e = CompilerError::FeatureDisabled("nccl".into());
        let msg = format!("{e}");
        assert!(msg.contains("feature disabled"), "Display should contain 'feature disabled'");
        assert!(msg.contains("nccl"), "Display should contain 'nccl'");
    }

    #[test]
    fn inference_error_unsupported_display() {
        let e = InferenceError::Unsupported("MoE routing".into());
        let msg = format!("{e}");
        assert!(msg.contains("unsupported"), "Display should contain 'unsupported'");
        assert!(msg.contains("MoE routing"), "Display should contain 'MoE routing'");
    }

    #[test]
    fn inference_error_invalid_config_display() {
        let e = InferenceError::InvalidConfig("negative vocab_size".into());
        let msg = format!("{e}");
        assert!(msg.contains("invalid config"), "Display should contain 'invalid config'");
        assert!(msg.contains("negative vocab_size"));
    }

    #[test]
    fn inference_error_runtime_error_display() {
        let e = InferenceError::RuntimeError("stack overflow".into());
        let msg = format!("{e}");
        assert!(msg.contains("runtime error"));
        assert!(msg.contains("stack overflow"));
    }

    #[test]
    fn model_config_llama_7b_kv_cache_cross_check() {
        let cfg = ModelConfig::llama_7b();
        // Cross-check: kv_dim = num_kv_heads * head_dim = 32 * 128 = 4096
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        assert_eq!(kv_dim, cfg.hidden_size, "LLaMA-7B should have kv_dim == hidden_size (MHA)");
        // 2 * 4096 * 4 * 32 = 1,048,576
        assert_eq!(cfg.kv_cache_bytes_per_token(), 2 * kv_dim * 4 * cfg.num_layers);
    }

    #[test]
    fn model_config_mistral_gqa_ratio() {
        let cfg = ModelConfig::mistral_7b();
        // Mistral-7B uses GQA: num_kv_heads=8, num_heads=32, ratio=4:1
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 4,
            "Mistral GQA ratio should be 4:1");
        // kv_dim should be smaller than hidden_size due to GQA
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        assert!(kv_dim < cfg.hidden_size, "GQA kv_dim should be < hidden_size");
    }
}
