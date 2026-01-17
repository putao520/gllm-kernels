//! CUDA EAGLE-3 adaptive draft length speculative decoding kernel.
//!
//! Based on EAGLE-3 (NeurIPS'25): Multi-layer feature fusion, token-level
//! confidence prediction, and adaptive draft length scheduling for 2-6x
//! inference acceleration.
//!
//! # Key Operations
//! - Multi-layer hidden state fusion (concatenation along hidden dim)
//! - Token-level confidence prediction via linear projection + sigmoid
//! - Draft generation with confidence-based early termination
//!
//! # SM-Aware PTX Loading
//!
//! - SM 61 (Pascal): GTX 1060/1070/1080
//! - SM 80 (Ampere): A100, RTX 30 series and higher
//!
//! 🚨 **Fat Binary Only**: NO runtime compilation fallback.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

// Kernel function names
const KERNEL_FUSE_HIDDEN_F32: &str = "eagle3_fuse_hidden_f32";
const KERNEL_FUSE_HIDDEN_F16: &str = "eagle3_fuse_hidden_f16";
const KERNEL_PREDICT_CONFIDENCE_F32: &str = "eagle3_predict_confidence_f32";
const KERNEL_PREDICT_CONFIDENCE_F16: &str = "eagle3_predict_confidence_f16";
const KERNEL_GENERATE_DRAFT_F32: &str = "eagle3_generate_draft_f32";
const KERNEL_GENERATE_DRAFT_F16: &str = "eagle3_generate_draft_f16";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for EAGLE-3 kernel.
/// PTX compiled for a lower SM version is forward-compatible with higher SM GPUs.
///
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static EAGLE3_PTX: PtxCollection = PtxCollection {
    kernel_name: "eagle3",
    ptx_versions: &[
        // SM 61 (Pascal) - GTX 1060/1070/1080
        (61, include_str!("kernels/eagle3_sm61.ptx")),
        // SM 80 (Ampere) - default for A100/RTX 30 series and higher
        (80, include_str!("kernels/eagle3.ptx")),
    ],
};

/// Errors surfaced by the CUDA EAGLE-3 kernels.
#[derive(Debug)]
pub enum Eagle3Error {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
    /// Dimension mismatch.
    DimensionMismatch(String),
}

impl fmt::Display for Eagle3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
            Self::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
        }
    }
}

impl std::error::Error for Eagle3Error {}

impl From<DriverError> for Eagle3Error {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for Eagle3Error {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// EAGLE-3 CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Multi-layer hidden state fusion
/// - Token-level confidence prediction
/// - Draft token generation with early termination
pub struct Eagle3Kernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Hidden state fusion kernels
    kernel_fuse_hidden_f32: CudaFunction,
    kernel_fuse_hidden_f16: CudaFunction,
    // Confidence prediction kernels
    kernel_predict_confidence_f32: CudaFunction,
    kernel_predict_confidence_f16: CudaFunction,
    // Draft generation kernels
    kernel_generate_draft_f32: CudaFunction,
    kernel_generate_draft_f16: CudaFunction,
}

impl Eagle3Kernel {
    /// Load EAGLE-3 kernel module on the given device.
    ///
    /// This method automatically selects the best PTX binary for the detected GPU.
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Eagle3Error> {
        let ptx = EAGLE3_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_fuse_hidden_f32 = module
            .load_function(KERNEL_FUSE_HIDDEN_F32)
            .map_err(|_| Eagle3Error::KernelMissing(KERNEL_FUSE_HIDDEN_F32))?;
        let kernel_fuse_hidden_f16 = module
            .load_function(KERNEL_FUSE_HIDDEN_F16)
            .map_err(|_| Eagle3Error::KernelMissing(KERNEL_FUSE_HIDDEN_F16))?;
        let kernel_predict_confidence_f32 = module
            .load_function(KERNEL_PREDICT_CONFIDENCE_F32)
            .map_err(|_| Eagle3Error::KernelMissing(KERNEL_PREDICT_CONFIDENCE_F32))?;
        let kernel_predict_confidence_f16 = module
            .load_function(KERNEL_PREDICT_CONFIDENCE_F16)
            .map_err(|_| Eagle3Error::KernelMissing(KERNEL_PREDICT_CONFIDENCE_F16))?;
        let kernel_generate_draft_f32 = module
            .load_function(KERNEL_GENERATE_DRAFT_F32)
            .map_err(|_| Eagle3Error::KernelMissing(KERNEL_GENERATE_DRAFT_F32))?;
        let kernel_generate_draft_f16 = module
            .load_function(KERNEL_GENERATE_DRAFT_F16)
            .map_err(|_| Eagle3Error::KernelMissing(KERNEL_GENERATE_DRAFT_F16))?;

        Ok(Self {
            module,
            kernel_fuse_hidden_f32,
            kernel_fuse_hidden_f16,
            kernel_predict_confidence_f32,
            kernel_predict_confidence_f16,
            kernel_generate_draft_f32,
            kernel_generate_draft_f16,
        })
    }

    /// Fuse hidden states from multiple layers (f32).
    ///
    /// Concatenates hidden states from the last `num_layers` layers along the hidden dimension.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for async execution
    /// * `layer_hidden_states` - Slice of layer hidden states, each [batch * seq_len * hidden_dim]
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension per layer
    /// * `num_layers` - Number of layers to fuse
    ///
    /// # Returns
    /// Fused hidden states: [batch * seq_len * (hidden_dim * num_layers)]
    pub fn fuse_hidden_states_f32(
        &self,
        stream: &Arc<CudaStream>,
        layer_hidden_states: &[&CudaSlice<f32>],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<CudaSlice<f32>, Eagle3Error> {
        if layer_hidden_states.len() < num_layers {
            return Err(Eagle3Error::InvalidConfig(format!(
                "Insufficient layers: got {}, need {}",
                layer_hidden_states.len(),
                num_layers
            )));
        }

        let fused_dim = hidden_dim * num_layers;
        let total_elements = batch_size * seq_len;
        let output_size = total_elements * fused_dim;

        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_elements + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_size_i32 = batch_size as i32;
        let seq_len_i32 = seq_len as i32;
        let hidden_dim_i32 = hidden_dim as i32;
        let num_layers_i32 = num_layers as i32;

        // We need to launch for each layer - fuse sequentially
        let start_layer = layer_hidden_states.len() - num_layers;
        for (layer_idx, layer_data) in layer_hidden_states[start_layer..].iter().enumerate() {
            let layer_idx_i32 = layer_idx as i32;
            unsafe {
                let mut builder = stream.launch_builder(&self.kernel_fuse_hidden_f32);
                builder.arg(*layer_data);
                builder.arg(&mut output);
                builder.arg(&batch_size_i32);
                builder.arg(&seq_len_i32);
                builder.arg(&hidden_dim_i32);
                builder.arg(&num_layers_i32);
                builder.arg(&layer_idx_i32);
                builder.launch(cfg)?;
            }
        }

        Ok(output)
    }

    /// Fuse hidden states from multiple layers (f16).
    pub fn fuse_hidden_states_f16(
        &self,
        stream: &Arc<CudaStream>,
        layer_hidden_states: &[&CudaSlice<f16>],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<CudaSlice<f16>, Eagle3Error> {
        if layer_hidden_states.len() < num_layers {
            return Err(Eagle3Error::InvalidConfig(format!(
                "Insufficient layers: got {}, need {}",
                layer_hidden_states.len(),
                num_layers
            )));
        }

        let fused_dim = hidden_dim * num_layers;
        let total_elements = batch_size * seq_len;
        let output_size = total_elements * fused_dim;

        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_elements + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_size_i32 = batch_size as i32;
        let seq_len_i32 = seq_len as i32;
        let hidden_dim_i32 = hidden_dim as i32;
        let num_layers_i32 = num_layers as i32;

        let start_layer = layer_hidden_states.len() - num_layers;
        for (layer_idx, layer_data) in layer_hidden_states[start_layer..].iter().enumerate() {
            let layer_idx_i32 = layer_idx as i32;
            unsafe {
                let mut builder = stream.launch_builder(&self.kernel_fuse_hidden_f16);
                builder.arg(*layer_data);
                builder.arg(&mut output);
                builder.arg(&batch_size_i32);
                builder.arg(&seq_len_i32);
                builder.arg(&hidden_dim_i32);
                builder.arg(&num_layers_i32);
                builder.arg(&layer_idx_i32);
                builder.launch(cfg)?;
            }
        }

        Ok(output)
    }

    /// Predict token-level confidence from fused hidden states (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `fused_hidden` - Fused hidden states [batch * seq_len * fused_dim]
    /// * `weight` - Linear projection weight [fused_dim]
    /// * `bias` - Bias scalar
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `fused_dim` - Fused hidden dimension
    ///
    /// # Returns
    /// Confidence scores: [batch * seq_len]
    pub fn predict_confidence_f32(
        &self,
        stream: &Arc<CudaStream>,
        fused_hidden: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: f32,
        batch_size: usize,
        seq_len: usize,
        fused_dim: usize,
    ) -> Result<CudaSlice<f32>, Eagle3Error> {
        let total_positions = batch_size * seq_len;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(total_positions)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_positions + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let total_positions_i32 = total_positions as i32;
        let fused_dim_i32 = fused_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_predict_confidence_f32);
            builder.arg(fused_hidden);
            builder.arg(weight);
            builder.arg(&bias);
            builder.arg(&mut output);
            builder.arg(&total_positions_i32);
            builder.arg(&fused_dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Predict token-level confidence from fused hidden states (f16).
    pub fn predict_confidence_f16(
        &self,
        stream: &Arc<CudaStream>,
        fused_hidden: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        bias: f32,
        batch_size: usize,
        seq_len: usize,
        fused_dim: usize,
    ) -> Result<CudaSlice<f32>, Eagle3Error> {
        let total_positions = batch_size * seq_len;
        // Output is always f32 for confidence scores
        let mut output: CudaSlice<f32> = stream.alloc_zeros(total_positions)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_positions + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let total_positions_i32 = total_positions as i32;
        let fused_dim_i32 = fused_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_predict_confidence_f16);
            builder.arg(fused_hidden);
            builder.arg(weight);
            builder.arg(&bias);
            builder.arg(&mut output);
            builder.arg(&total_positions_i32);
            builder.arg(&fused_dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Generate draft tokens with confidence-based early termination (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `logits` - Draft model logits [seq_len * vocab_size]
    /// * `confidence` - Pre-computed confidence scores [seq_len]
    /// * `confidence_threshold` - Threshold for early termination
    /// * `seq_len` - Maximum sequence length
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Tuple of (draft_tokens [seq_len], draft_length scalar, log_probs [seq_len])
    pub fn generate_draft_f32(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f32>,
        confidence: &CudaSlice<f32>,
        confidence_threshold: f32,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>, CudaSlice<f32>), Eagle3Error> {
        let mut draft_tokens: CudaSlice<i32> = stream.alloc_zeros(seq_len)?;
        let mut draft_length: CudaSlice<i32> = stream.alloc_zeros(1)?;
        let mut log_probs: CudaSlice<f32> = stream.alloc_zeros(seq_len)?;

        // Use one block per position for parallel argmax
        let cfg = LaunchConfig {
            grid_dim: (seq_len as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE.min(vocab_size as u32), 1, 1),
            shared_mem_bytes: (DEFAULT_BLOCK_SIZE as usize * 2 * std::mem::size_of::<f32>()) as u32,
        };

        let seq_len_i32 = seq_len as i32;
        let vocab_size_i32 = vocab_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_generate_draft_f32);
            builder.arg(logits);
            builder.arg(confidence);
            builder.arg(&confidence_threshold);
            builder.arg(&mut draft_tokens);
            builder.arg(&mut draft_length);
            builder.arg(&mut log_probs);
            builder.arg(&seq_len_i32);
            builder.arg(&vocab_size_i32);
            builder.launch(cfg)?;
        }

        Ok((draft_tokens, draft_length, log_probs))
    }

    /// Generate draft tokens with confidence-based early termination (f16).
    pub fn generate_draft_f16(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f16>,
        confidence: &CudaSlice<f32>,
        confidence_threshold: f32,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>, CudaSlice<f32>), Eagle3Error> {
        let mut draft_tokens: CudaSlice<i32> = stream.alloc_zeros(seq_len)?;
        let mut draft_length: CudaSlice<i32> = stream.alloc_zeros(1)?;
        let mut log_probs: CudaSlice<f32> = stream.alloc_zeros(seq_len)?;

        let cfg = LaunchConfig {
            grid_dim: (seq_len as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE.min(vocab_size as u32), 1, 1),
            shared_mem_bytes: (DEFAULT_BLOCK_SIZE as usize * 2 * std::mem::size_of::<f32>()) as u32,
        };

        let seq_len_i32 = seq_len as i32;
        let vocab_size_i32 = vocab_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_generate_draft_f16);
            builder.arg(logits);
            builder.arg(confidence);
            builder.arg(&confidence_threshold);
            builder.arg(&mut draft_tokens);
            builder.arg(&mut draft_length);
            builder.arg(&mut log_probs);
            builder.arg(&seq_len_i32);
            builder.arg(&vocab_size_i32);
            builder.launch(cfg)?;
        }

        Ok((draft_tokens, draft_length, log_probs))
    }
}

/// Configuration for EAGLE-3 CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct Eagle3CudaConfig {
    /// Number of layers to fuse for confidence prediction.
    pub fusion_layers: usize,
    /// Hidden dimension per layer.
    pub hidden_dim: usize,
    /// Confidence threshold for early termination.
    pub confidence_threshold: f32,
    /// Minimum draft length.
    pub min_draft_length: usize,
    /// Maximum draft length.
    pub max_draft_length: usize,
}

impl Default for Eagle3CudaConfig {
    fn default() -> Self {
        Self {
            fusion_layers: 4,
            hidden_dim: 768,
            confidence_threshold: 0.5,
            min_draft_length: 1,
            max_draft_length: 8,
        }
    }
}

impl Eagle3CudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), Eagle3Error> {
        if self.fusion_layers == 0 {
            return Err(Eagle3Error::InvalidConfig(
                "fusion_layers must be > 0".into(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(Eagle3Error::InvalidConfig("hidden_dim must be > 0".into()));
        }
        if self.confidence_threshold <= 0.0 || self.confidence_threshold > 1.0 {
            return Err(Eagle3Error::InvalidConfig(
                "confidence_threshold must be in (0, 1]".into(),
            ));
        }
        if self.min_draft_length == 0 {
            return Err(Eagle3Error::InvalidConfig(
                "min_draft_length must be > 0".into(),
            ));
        }
        if self.max_draft_length < self.min_draft_length {
            return Err(Eagle3Error::InvalidConfig(
                "max_draft_length must be >= min_draft_length".into(),
            ));
        }
        Ok(())
    }

    /// Get the fused hidden dimension.
    pub fn fused_dim(&self) -> usize {
        self.hidden_dim * self.fusion_layers
    }
}
