//! C ABI types — opaque handles and error codes for FFI consumers.

/// Error codes returned by C ABI functions.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GllmStatus {
    Ok = 0,
    InvalidArg = -1,
    OutOfMemory = -2,
    CompileError = -3,
    RuntimeError = -4,
    Unsupported = -5,
    IoError = -6,
}

impl From<crate::types::InferenceError> for GllmStatus {
    fn from(e: crate::types::InferenceError) -> Self {
        use crate::types::InferenceError;
        match e {
            InferenceError::InvalidConfig(_) => GllmStatus::InvalidArg,
            InferenceError::OutOfMemory { .. } => GllmStatus::OutOfMemory,
            InferenceError::CompileError(_) => GllmStatus::CompileError,
            InferenceError::RuntimeError(_) => GllmStatus::RuntimeError,
            InferenceError::ShapeMismatch { .. } => GllmStatus::InvalidArg,
            InferenceError::Unsupported(_) => GllmStatus::Unsupported,
            InferenceError::Io(_) => GllmStatus::IoError,
        }
    }
}

/// Opaque handle to an inference backend.
pub type GllmBackend = *mut std::ffi::c_void;

/// Opaque handle to a device tensor.
pub type GllmTensor = *mut std::ffi::c_void;

/// Opaque handle to a KV cache.
pub type GllmKvCache = *mut std::ffi::c_void;

/// Opaque handle to model weights.
pub type GllmWeights = *mut std::ffi::c_void;

/// Weight field identifiers for `gllm_weights_get_ptr`.
///
/// Fields 0..2 are global (layer_idx ignored).
/// Fields 10..21 are per-layer (require valid layer_idx).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GllmWeightField {
    /// Token embedding: [vocab_size, hidden_size]
    Embedding = 0,
    /// Final norm weight: [hidden_size]
    FinalNorm = 1,
    /// LM head: [hidden_size, vocab_size]
    LmHead = 2,
    /// Attention norm weight: [hidden_size]
    AttnNorm = 10,
    /// Q projection: [hidden_size, num_heads * head_dim]
    Wq = 11,
    /// K projection: [hidden_size, num_kv_heads * head_dim]
    Wk = 12,
    /// V projection: [hidden_size, num_kv_heads * head_dim]
    Wv = 13,
    /// Output projection: [num_heads * head_dim, hidden_size]
    Wo = 14,
    /// FFN norm weight: [hidden_size]
    FfnNorm = 15,
    /// FFN gate projection: [hidden_size, intermediate_size]
    WGate = 16,
    /// FFN up projection: [hidden_size, intermediate_size]
    WUp = 17,
    /// FFN down projection: [intermediate_size, hidden_size]
    WDown = 18,
    /// QKV bias: [q_dim + 2*kv_dim] (optional, may be absent)
    QkvBias = 19,
    /// Attention norm bias: [hidden_size] (LayerNorm models only)
    AttnNormBias = 20,
    /// FFN norm bias: [hidden_size] (LayerNorm models only)
    FfnNormBias = 21,
}

impl GllmWeightField {
    /// Convert from raw i32. Returns None for unknown values.
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Embedding),
            1 => Some(Self::FinalNorm),
            2 => Some(Self::LmHead),
            10 => Some(Self::AttnNorm),
            11 => Some(Self::Wq),
            12 => Some(Self::Wk),
            13 => Some(Self::Wv),
            14 => Some(Self::Wo),
            15 => Some(Self::FfnNorm),
            16 => Some(Self::WGate),
            17 => Some(Self::WUp),
            18 => Some(Self::WDown),
            19 => Some(Self::QkvBias),
            20 => Some(Self::AttnNormBias),
            21 => Some(Self::FfnNormBias),
            _ => None,
        }
    }

    /// Whether this field is global (not per-layer).
    pub fn is_global(self) -> bool {
        (self as i32) < 10
    }
}

/// Model configuration passed from C.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GllmModelConfig {
    pub arch: i32,           // 0=Llama, 1=Gpt2, 2=Mistral, 3=Phi, 4=Qwen, 5=Gemma
    pub hidden_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_theta: f64,
    pub norm_eps: f32,
    pub dtype: i32,          // 0=F32, 1=F16, 2=BF16
    pub quant_type: i32,     // -1=None, 0=Q4_0, 1=Q4_1, 2=Q8_0
    pub has_qkv_bias: i32,   // 0=false, 1=true
    pub partial_rotary_factor: f32, // 0.0..=1.0
}

impl GllmModelConfig {
    /// Convert to the Rust ModelConfig type.
    pub fn to_model_config(&self) -> Result<crate::types::ModelConfig, GllmStatus> {
        use crate::types::{DType, ModelArch, ModelConfig};

        let arch = match self.arch {
            0 => ModelArch::Llama,
            1 => ModelArch::Gpt2,
            2 => ModelArch::Mistral,
            3 => ModelArch::Phi,
            4 => ModelArch::Qwen,
            5 => ModelArch::Gemma,
            _ => return Err(GllmStatus::InvalidArg),
        };

        let dtype = match self.dtype {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::BF16,
            _ => return Err(GllmStatus::InvalidArg),
        };

        let quant_type = match self.quant_type {
            -1 => None,
            0 => Some(crate::quant::QuantType::Q4_0),
            1 => Some(crate::quant::QuantType::Q4_1),
            2 => Some(crate::quant::QuantType::Q8_0),
            _ => return Err(GllmStatus::InvalidArg),
        };

        Ok(ModelConfig {
            arch,
            hidden_size: self.hidden_size as usize,
            num_heads: self.num_heads as usize,
            num_kv_heads: self.num_kv_heads as usize,
            head_dim: self.head_dim as usize,
            intermediate_size: self.intermediate_size as usize,
            num_layers: self.num_layers as usize,
            vocab_size: self.vocab_size as usize,
            max_seq_len: self.max_seq_len as usize,
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            dtype,
            quant_type,
            rope_interleaved: false,
            has_qkv_bias: self.has_qkv_bias != 0,
            partial_rotary_factor: self.partial_rotary_factor,
            sliding_window: None, // not yet exposed via GllmModelConfig C ABI
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GllmStatus ──

    #[test]
    fn status_values() {
        assert_eq!(GllmStatus::Ok as i32, 0);
        assert_eq!(GllmStatus::InvalidArg as i32, -1);
        assert_eq!(GllmStatus::OutOfMemory as i32, -2);
        assert_eq!(GllmStatus::CompileError as i32, -3);
        assert_eq!(GllmStatus::RuntimeError as i32, -4);
        assert_eq!(GllmStatus::Unsupported as i32, -5);
        assert_eq!(GllmStatus::IoError as i32, -6);
    }

    #[test]
    fn status_from_inference_error() {
        use crate::types::InferenceError;
        assert_eq!(GllmStatus::from(InferenceError::InvalidConfig("x".into())), GllmStatus::InvalidArg);
        assert_eq!(GllmStatus::from(InferenceError::CompileError("x".into())), GllmStatus::CompileError);
        assert_eq!(GllmStatus::from(InferenceError::Unsupported("x".into())), GllmStatus::Unsupported);
    }

    // ── GllmWeightField ──

    #[test]
    fn weight_field_from_i32_roundtrip() {
        for (val, expected) in [
            (0, GllmWeightField::Embedding),
            (1, GllmWeightField::FinalNorm),
            (2, GllmWeightField::LmHead),
            (10, GllmWeightField::AttnNorm),
            (11, GllmWeightField::Wq),
            (18, GllmWeightField::WDown),
        ] {
            assert_eq!(GllmWeightField::from_i32(val), Some(expected));
        }
    }

    #[test]
    fn weight_field_unknown_returns_none() {
        assert!(GllmWeightField::from_i32(99).is_none());
        assert!(GllmWeightField::from_i32(-1).is_none());
    }

    #[test]
    fn weight_field_is_global() {
        assert!(GllmWeightField::Embedding.is_global());
        assert!(GllmWeightField::FinalNorm.is_global());
        assert!(GllmWeightField::LmHead.is_global());
        assert!(!GllmWeightField::AttnNorm.is_global());
        assert!(!GllmWeightField::Wq.is_global());
        assert!(!GllmWeightField::WDown.is_global());
    }

    // ── GllmModelConfig ──

    #[test]
    fn model_config_to_rust_valid() {
        let c = GllmModelConfig {
            arch: 0, // Llama
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 11008,
            num_layers: 32,
            vocab_size: 32000,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: 0, // F32
            quant_type: -1,
            has_qkv_bias: 0,
            partial_rotary_factor: 1.0,
        };
        let mc = c.to_model_config().unwrap();
        assert_eq!(mc.hidden_size, 4096);
        assert_eq!(mc.num_layers, 32);
        assert!(mc.quant_type.is_none());
    }

    #[test]
    fn model_config_invalid_arch() {
        let mut c = GllmModelConfig {
            arch: 99,
            hidden_size: 256,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate_size: 512,
            num_layers: 4,
            vocab_size: 1000,
            max_seq_len: 512,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: 0,
            quant_type: -1,
            has_qkv_bias: 0,
            partial_rotary_factor: 1.0,
        };
        assert_eq!(c.to_model_config().unwrap_err(), GllmStatus::InvalidArg);
        c.arch = 0;
        c.dtype = 99;
        assert_eq!(c.to_model_config().unwrap_err(), GllmStatus::InvalidArg);
    }

    // ── Test 8: GllmStatus Debug and Copy ──

    #[test]
    fn gllm_status_debug_and_copy() {
        let s = GllmStatus::Ok;
        let copied = s;
        assert_eq!(s, copied);
        let debug = format!("{:?}", GllmStatus::OutOfMemory);
        assert!(debug.contains("OutOfMemory"));
    }

    // ── Test 9: GllmWeightField all variants from_i32 ──

    #[test]
    fn weight_field_all_variants() {
        assert_eq!(GllmWeightField::from_i32(13), Some(GllmWeightField::Wv));
        assert_eq!(GllmWeightField::from_i32(14), Some(GllmWeightField::Wo));
        assert_eq!(GllmWeightField::from_i32(15), Some(GllmWeightField::FfnNorm));
        assert_eq!(GllmWeightField::from_i32(16), Some(GllmWeightField::WGate));
        assert_eq!(GllmWeightField::from_i32(17), Some(GllmWeightField::WUp));
        assert_eq!(GllmWeightField::from_i32(19), Some(GllmWeightField::QkvBias));
        assert_eq!(GllmWeightField::from_i32(20), Some(GllmWeightField::AttnNormBias));
        assert_eq!(GllmWeightField::from_i32(21), Some(GllmWeightField::FfnNormBias));
    }

    // ── Test 10: GllmWeightField Debug format ──

    #[test]
    fn weight_field_debug_format() {
        assert!(format!("{:?}", GllmWeightField::Embedding).contains("Embedding"));
        assert!(format!("{:?}", GllmWeightField::Wq).contains("Wq"));
        assert!(format!("{:?}", GllmWeightField::QkvBias).contains("QkvBias"));
    }

    // ── Test 11: GllmModelConfig all arch variants ──

    #[test]
    fn model_config_all_arch_variants() {
        let base = GllmModelConfig {
            arch: 0, hidden_size: 256, num_heads: 4, num_kv_heads: 4,
            head_dim: 64, intermediate_size: 512, num_layers: 4,
            vocab_size: 1000, max_seq_len: 512, rope_theta: 10000.0,
            norm_eps: 1e-5, dtype: 0, quant_type: -1,
            has_qkv_bias: 0, partial_rotary_factor: 1.0,
        };
        for arch in [0, 1, 2, 3, 4, 5] {
            let mut c = base.clone();
            c.arch = arch;
            assert!(c.to_model_config().is_ok());
        }
    }

    // ── Test 12: GllmModelConfig with quant_type ──

    #[test]
    fn model_config_with_quant_type() {
        let mut c = GllmModelConfig {
            arch: 0, hidden_size: 256, num_heads: 4, num_kv_heads: 4,
            head_dim: 64, intermediate_size: 512, num_layers: 4,
            vocab_size: 1000, max_seq_len: 512, rope_theta: 10000.0,
            norm_eps: 1e-5, dtype: 0, quant_type: 0,
            has_qkv_bias: 0, partial_rotary_factor: 1.0,
        };
        let mc = c.to_model_config().unwrap();
        assert!(mc.quant_type.is_some());

        c.quant_type = 99;
        assert_eq!(c.to_model_config().unwrap_err(), GllmStatus::InvalidArg);
    }

    // ── Test 13: GllmModelConfig has_qkv_bias ──

    #[test]
    fn model_config_has_qkv_bias() {
        let mut c = GllmModelConfig {
            arch: 0, hidden_size: 256, num_heads: 4, num_kv_heads: 4,
            head_dim: 64, intermediate_size: 512, num_layers: 4,
            vocab_size: 1000, max_seq_len: 512, rope_theta: 10000.0,
            norm_eps: 1e-5, dtype: 0, quant_type: -1,
            has_qkv_bias: 1, partial_rotary_factor: 1.0,
        };
        assert!(c.to_model_config().unwrap().has_qkv_bias);
        c.has_qkv_bias = 0;
        assert!(!c.to_model_config().unwrap().has_qkv_bias);
    }

    // ── Test 14: GllmWeightField is_global for bias fields ──

    #[test]
    fn weight_field_bias_not_global() {
        assert!(!GllmWeightField::QkvBias.is_global());
        assert!(!GllmWeightField::AttnNormBias.is_global());
        assert!(!GllmWeightField::FfnNormBias.is_global());
    }
}
