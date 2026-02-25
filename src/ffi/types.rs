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

impl From<crate::inference::types::InferenceError> for GllmStatus {
    fn from(e: crate::inference::types::InferenceError) -> Self {
        use crate::inference::types::InferenceError;
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
    pub sliding_window: i32,  // 0=disabled, >0=window size
}

impl GllmModelConfig {
    /// Convert to the Rust ModelConfig type.
    pub fn to_model_config(&self) -> Result<crate::inference::types::ModelConfig, GllmStatus> {
        use crate::inference::types::{DType, ModelArch, ModelConfig};

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
            sliding_window: if self.sliding_window > 0 { Some(self.sliding_window as usize) } else { None },
        })
    }
}
