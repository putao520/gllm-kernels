//! CUDA SpecEE / LayerSkip early-exit decoding kernel.
//!
//! Based on SpecEE (ICML'24) and LayerSkip (Meta'24):
//! - Layer-level confidence computation
//! - Dynamic layer skipping based on confidence threshold
//! - Early exit when confidence is high enough
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
const KERNEL_COMPUTE_CONFIDENCE_F32: &str = "specee_compute_confidence_f32";
const KERNEL_COMPUTE_CONFIDENCE_F16: &str = "specee_compute_confidence_f16";
const KERNEL_LAYER_SKIP_DECISION_F32: &str = "specee_layer_skip_decision_f32";
const KERNEL_LAYER_SKIP_DECISION_F16: &str = "specee_layer_skip_decision_f16";
const KERNEL_EARLY_EXIT_F32: &str = "specee_early_exit_f32";
const KERNEL_EARLY_EXIT_F16: &str = "specee_early_exit_f16";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for SpecEE kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static SPECEE_PTX: PtxCollection = PtxCollection {
    kernel_name: "spec_ee",
    ptx_versions: &[
        (61, include_str!("kernels/spec_ee_sm61.ptx")),
        (80, include_str!("kernels/spec_ee.ptx")),
    ],
};

/// Errors surfaced by the CUDA SpecEE kernels.
#[derive(Debug)]
pub enum SpecEEError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
}

impl fmt::Display for SpecEEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for SpecEEError {}

impl From<DriverError> for SpecEEError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for SpecEEError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// SpecEE / LayerSkip CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Layer-level confidence computation
/// - Dynamic layer skip decisions
/// - Early exit processing
pub struct SpecEEKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Confidence computation kernels
    kernel_compute_confidence_f32: CudaFunction,
    kernel_compute_confidence_f16: CudaFunction,
    // Layer skip decision kernels
    kernel_layer_skip_f32: CudaFunction,
    kernel_layer_skip_f16: CudaFunction,
    // Early exit kernels
    kernel_early_exit_f32: CudaFunction,
    kernel_early_exit_f16: CudaFunction,
}

impl SpecEEKernel {
    /// Load SpecEE kernel module on the given device.
    ///
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, SpecEEError> {
        let ptx = SPECEE_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_compute_confidence_f32 = module
            .load_function(KERNEL_COMPUTE_CONFIDENCE_F32)
            .map_err(|_| SpecEEError::KernelMissing(KERNEL_COMPUTE_CONFIDENCE_F32))?;
        let kernel_compute_confidence_f16 = module
            .load_function(KERNEL_COMPUTE_CONFIDENCE_F16)
            .map_err(|_| SpecEEError::KernelMissing(KERNEL_COMPUTE_CONFIDENCE_F16))?;
        let kernel_layer_skip_f32 = module
            .load_function(KERNEL_LAYER_SKIP_DECISION_F32)
            .map_err(|_| SpecEEError::KernelMissing(KERNEL_LAYER_SKIP_DECISION_F32))?;
        let kernel_layer_skip_f16 = module
            .load_function(KERNEL_LAYER_SKIP_DECISION_F16)
            .map_err(|_| SpecEEError::KernelMissing(KERNEL_LAYER_SKIP_DECISION_F16))?;
        let kernel_early_exit_f32 = module
            .load_function(KERNEL_EARLY_EXIT_F32)
            .map_err(|_| SpecEEError::KernelMissing(KERNEL_EARLY_EXIT_F32))?;
        let kernel_early_exit_f16 = module
            .load_function(KERNEL_EARLY_EXIT_F16)
            .map_err(|_| SpecEEError::KernelMissing(KERNEL_EARLY_EXIT_F16))?;

        Ok(Self {
            module,
            kernel_compute_confidence_f32,
            kernel_compute_confidence_f16,
            kernel_layer_skip_f32,
            kernel_layer_skip_f16,
            kernel_early_exit_f32,
            kernel_early_exit_f16,
        })
    }

    /// Compute layer-level confidence scores (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `hidden_states` - Hidden states [batch * seq_len * hidden_dim]
    /// * `classifier_weight` - Classifier weight [hidden_dim]
    /// * `classifier_bias` - Classifier bias
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension
    ///
    /// # Returns
    /// Confidence scores: [batch * seq_len]
    pub fn compute_confidence_f32(
        &self,
        stream: &Arc<CudaStream>,
        hidden_states: &CudaSlice<f32>,
        classifier_weight: &CudaSlice<f32>,
        classifier_bias: f32,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<CudaSlice<f32>, SpecEEError> {
        let total_positions = batch_size * seq_len;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(total_positions)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_positions + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let total_i32 = total_positions as i32;
        let hidden_dim_i32 = hidden_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_compute_confidence_f32);
            builder.arg(hidden_states);
            builder.arg(classifier_weight);
            builder.arg(&classifier_bias);
            builder.arg(&mut output);
            builder.arg(&total_i32);
            builder.arg(&hidden_dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Compute layer-level confidence scores (f16).
    pub fn compute_confidence_f16(
        &self,
        stream: &Arc<CudaStream>,
        hidden_states: &CudaSlice<f16>,
        classifier_weight: &CudaSlice<f16>,
        classifier_bias: f32,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<CudaSlice<f32>, SpecEEError> {
        let total_positions = batch_size * seq_len;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(total_positions)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_positions + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let total_i32 = total_positions as i32;
        let hidden_dim_i32 = hidden_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_compute_confidence_f16);
            builder.arg(hidden_states);
            builder.arg(classifier_weight);
            builder.arg(&classifier_bias);
            builder.arg(&mut output);
            builder.arg(&total_i32);
            builder.arg(&hidden_dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Make layer skip decision based on confidence (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `confidence` - Confidence scores [batch * seq_len]
    /// * `skip_threshold` - Threshold for skipping layer
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Skip decisions: [batch * seq_len] (1 = skip, 0 = compute)
    pub fn layer_skip_decision_f32(
        &self,
        stream: &Arc<CudaStream>,
        confidence: &CudaSlice<f32>,
        skip_threshold: f32,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<i32>, SpecEEError> {
        let total = batch_size * seq_len;
        let mut output: CudaSlice<i32> = stream.alloc_zeros(total)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let total_i32 = total as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_layer_skip_f32);
            builder.arg(confidence);
            builder.arg(&skip_threshold);
            builder.arg(&mut output);
            builder.arg(&total_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Make layer skip decision based on confidence (f16).
    pub fn layer_skip_decision_f16(
        &self,
        stream: &Arc<CudaStream>,
        confidence: &CudaSlice<f32>,  // Confidence is always f32
        skip_threshold: f32,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<i32>, SpecEEError> {
        // Same as f32 since confidence is always f32
        self.layer_skip_decision_f32(stream, confidence, skip_threshold, batch_size, seq_len)
    }

    /// Process early exit (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `hidden_states` - Current hidden states [batch * seq_len * hidden_dim]
    /// * `confidence` - Confidence scores [batch * seq_len]
    /// * `exit_threshold` - Threshold for early exit
    /// * `current_layer` - Current layer index
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension
    ///
    /// # Returns
    /// Tuple of (should_exit [batch], exit_layer [batch])
    pub fn early_exit_f32(
        &self,
        stream: &Arc<CudaStream>,
        hidden_states: &CudaSlice<f32>,
        confidence: &CudaSlice<f32>,
        exit_threshold: f32,
        current_layer: i32,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>), SpecEEError> {
        let mut should_exit: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;
        let mut exit_layer: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let hidden_i32 = hidden_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_early_exit_f32);
            builder.arg(hidden_states);
            builder.arg(confidence);
            builder.arg(&exit_threshold);
            builder.arg(&current_layer);
            builder.arg(&mut should_exit);
            builder.arg(&mut exit_layer);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&hidden_i32);
            builder.launch(cfg)?;
        }

        Ok((should_exit, exit_layer))
    }

    /// Process early exit (f16).
    pub fn early_exit_f16(
        &self,
        stream: &Arc<CudaStream>,
        hidden_states: &CudaSlice<f16>,
        confidence: &CudaSlice<f32>,
        exit_threshold: f32,
        current_layer: i32,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>), SpecEEError> {
        let mut should_exit: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;
        let mut exit_layer: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let hidden_i32 = hidden_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_early_exit_f16);
            builder.arg(hidden_states);
            builder.arg(confidence);
            builder.arg(&exit_threshold);
            builder.arg(&current_layer);
            builder.arg(&mut should_exit);
            builder.arg(&mut exit_layer);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&hidden_i32);
            builder.launch(cfg)?;
        }

        Ok((should_exit, exit_layer))
    }
}

/// Configuration for SpecEE CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct SpecEECudaConfig {
    /// Confidence threshold for layer skipping.
    pub skip_threshold: f32,
    /// Confidence threshold for early exit.
    pub exit_threshold: f32,
    /// Minimum number of layers to compute.
    pub min_layers: usize,
    /// Maximum number of layers (total model layers).
    pub max_layers: usize,
}

impl Default for SpecEECudaConfig {
    fn default() -> Self {
        Self {
            skip_threshold: 0.9,
            exit_threshold: 0.95,
            min_layers: 4,
            max_layers: 32,
        }
    }
}

impl SpecEECudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), SpecEEError> {
        if self.skip_threshold <= 0.0 || self.skip_threshold > 1.0 {
            return Err(SpecEEError::InvalidConfig(
                "skip_threshold must be in (0, 1]".into(),
            ));
        }
        if self.exit_threshold <= 0.0 || self.exit_threshold > 1.0 {
            return Err(SpecEEError::InvalidConfig(
                "exit_threshold must be in (0, 1]".into(),
            ));
        }
        if self.min_layers == 0 {
            return Err(SpecEEError::InvalidConfig("min_layers must be > 0".into()));
        }
        if self.max_layers < self.min_layers {
            return Err(SpecEEError::InvalidConfig(
                "max_layers must be >= min_layers".into(),
            ));
        }
        Ok(())
    }
}
