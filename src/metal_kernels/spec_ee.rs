//! Metal SpecEE/LayerSkip early-exit decoding kernels.
//!
//! This module provides Metal GPU-accelerated kernels for SpecEE and LayerSkip:
//! - Layer-level confidence estimation
//! - Early exit decision making
//! - Adaptive layer skipping
//!
//! ## Precompiled metallib (Required)
//!
//! metallib must be precompiled before use:
//! ```bash
//! ./scripts/compile_metal_kernels.sh
//! ```
//!
//! metallib is Metal's intermediate format (like PTX/HSACO).
//! NO runtime compilation fallback - metallib must be precompiled and embedded.

use std::fmt;
use std::mem;
use std::os::raw::c_void;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};

use crate::metal_kernels::metallib_loader::{MetallibCollection, MetallibLoadError};

const KERNEL_COMPUTE_CONFIDENCE_F32: &str = "spec_ee_compute_confidence_f32";
const KERNEL_COMPUTE_CONFIDENCE_F16: &str = "spec_ee_compute_confidence_f16";
const KERNEL_LAYER_SKIP_DECISION_F32: &str = "spec_ee_layer_skip_decision_f32";
const KERNEL_LAYER_SKIP_DECISION_F16: &str = "spec_ee_layer_skip_decision_f16";
const KERNEL_EARLY_EXIT_F32: &str = "spec_ee_early_exit_f32";
const KERNEL_EARLY_EXIT_F16: &str = "spec_ee_early_exit_f16";

/// Metallib collection for SpecEE/LayerSkip kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static SPEC_EE_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "spec_ee",
    metallib_data: include_bytes!("kernels/spec_ee.metallib"),
};

/// Parameters for confidence computation kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ComputeConfidenceParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    num_layers: u32,
    temperature: f32,
    _pad: [u32; 3],
}

/// Parameters for layer skip decision kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct LayerSkipDecisionParams {
    batch_size: u32,
    seq_len: u32,
    num_layers: u32,
    confidence_threshold: f32,
    min_layers: u32,
    max_skip: u32,
    _pad: [u32; 2],
}

/// Parameters for early exit kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct EarlyExitParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    vocab_size: u32,
    current_layer: u32,
    total_layers: u32,
    confidence_threshold: f32,
    _pad: u32,
}

/// Errors surfaced by the Metal SpecEE/LayerSkip kernels.
#[derive(Debug)]
pub enum SpecEEError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for SpecEEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for SpecEEError {}

impl From<MetallibLoadError> for SpecEEError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Configuration for SpecEE/LayerSkip operations.
#[derive(Clone, Debug)]
pub struct SpecEEConfig {
    /// Temperature for confidence scaling.
    pub temperature: f32,
    /// Confidence threshold for early exit.
    pub confidence_threshold: f32,
    /// Minimum number of layers to execute.
    pub min_layers: usize,
    /// Maximum number of layers to skip.
    pub max_skip: usize,
}

impl Default for SpecEEConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            confidence_threshold: 0.9,
            min_layers: 4,
            max_skip: 8,
        }
    }
}

impl SpecEEConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), SpecEEError> {
        if self.temperature <= 0.0 {
            return Err(SpecEEError::InvalidConfig("temperature must be positive".into()));
        }
        if self.confidence_threshold <= 0.0 || self.confidence_threshold > 1.0 {
            return Err(SpecEEError::InvalidConfig("confidence_threshold must be in (0, 1]".into()));
        }
        if self.min_layers == 0 {
            return Err(SpecEEError::InvalidConfig("min_layers must be at least 1".into()));
        }
        Ok(())
    }
}

/// SpecEE/LayerSkip Metal kernel wrapper.
pub struct SpecEEKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_compute_conf_f32: ComputePipelineState,
    pipeline_compute_conf_f16: ComputePipelineState,
    pipeline_layer_skip_f32: ComputePipelineState,
    pipeline_layer_skip_f16: ComputePipelineState,
    pipeline_early_exit_f32: ComputePipelineState,
    pipeline_early_exit_f16: ComputePipelineState,
}

impl SpecEEKernel {
    /// Load SpecEE/LayerSkip kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, SpecEEError> {
        let library = load_library(device)?;

        let pipeline_compute_conf_f32 = build_pipeline(device, &library, KERNEL_COMPUTE_CONFIDENCE_F32)?;
        let pipeline_compute_conf_f16 = build_pipeline(device, &library, KERNEL_COMPUTE_CONFIDENCE_F16)?;
        let pipeline_layer_skip_f32 = build_pipeline(device, &library, KERNEL_LAYER_SKIP_DECISION_F32)?;
        let pipeline_layer_skip_f16 = build_pipeline(device, &library, KERNEL_LAYER_SKIP_DECISION_F16)?;
        let pipeline_early_exit_f32 = build_pipeline(device, &library, KERNEL_EARLY_EXIT_F32)?;
        let pipeline_early_exit_f16 = build_pipeline(device, &library, KERNEL_EARLY_EXIT_F16)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_compute_conf_f32,
            pipeline_compute_conf_f16,
            pipeline_layer_skip_f32,
            pipeline_layer_skip_f16,
            pipeline_early_exit_f32,
            pipeline_early_exit_f16,
        })
    }

    /// Compute layer-level confidence scores (f32).
    pub fn compute_confidence_f32(
        &self,
        hidden_states: &Buffer,
        classifier_weight: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
        temperature: f32,
    ) -> Result<Buffer, SpecEEError> {
        self.compute_confidence_impl(
            hidden_states,
            classifier_weight,
            batch_size,
            seq_len,
            hidden_dim,
            num_layers,
            temperature,
            &self.pipeline_compute_conf_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Compute layer-level confidence scores (f16).
    pub fn compute_confidence_f16(
        &self,
        hidden_states: &Buffer,
        classifier_weight: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
        temperature: f32,
    ) -> Result<Buffer, SpecEEError> {
        self.compute_confidence_impl(
            hidden_states,
            classifier_weight,
            batch_size,
            seq_len,
            hidden_dim,
            num_layers,
            temperature,
            &self.pipeline_compute_conf_f16,
            mem::size_of::<u16>(),
        )
    }

    fn compute_confidence_impl(
        &self,
        hidden_states: &Buffer,
        classifier_weight: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
        temperature: f32,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, SpecEEError> {
        let output_elements = batch_size * seq_len * num_layers;
        let output_bytes = (output_elements * element_size) as u64;

        let params = ComputeConfidenceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            num_layers: num_layers as u32,
            temperature,
            _pad: [0; 3],
        };

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(hidden_states), 0);
        encoder.set_buffer(1, Some(classifier_weight), 0);
        encoder.set_buffer(2, Some(&output), 0);

        let params_size = mem::size_of::<ComputeConfidenceParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(output_elements as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Determine which layers to skip based on confidence (f32).
    pub fn layer_skip_decision_f32(
        &self,
        confidence_scores: &Buffer,
        batch_size: usize,
        seq_len: usize,
        num_layers: usize,
        config: &SpecEEConfig,
    ) -> Result<Buffer, SpecEEError> {
        config.validate()?;
        self.layer_skip_decision_impl(
            confidence_scores,
            batch_size,
            seq_len,
            num_layers,
            config,
            &self.pipeline_layer_skip_f32,
        )
    }

    /// Determine which layers to skip based on confidence (f16).
    pub fn layer_skip_decision_f16(
        &self,
        confidence_scores: &Buffer,
        batch_size: usize,
        seq_len: usize,
        num_layers: usize,
        config: &SpecEEConfig,
    ) -> Result<Buffer, SpecEEError> {
        config.validate()?;
        self.layer_skip_decision_impl(
            confidence_scores,
            batch_size,
            seq_len,
            num_layers,
            config,
            &self.pipeline_layer_skip_f16,
        )
    }

    fn layer_skip_decision_impl(
        &self,
        confidence_scores: &Buffer,
        batch_size: usize,
        seq_len: usize,
        num_layers: usize,
        config: &SpecEEConfig,
        pipeline: &ComputePipelineState,
    ) -> Result<Buffer, SpecEEError> {
        // Output: skip mask for each layer [batch, num_layers] as u32
        let output_elements = batch_size * num_layers;
        let output_bytes = (output_elements * mem::size_of::<u32>()) as u64;

        let params = LayerSkipDecisionParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            num_layers: num_layers as u32,
            confidence_threshold: config.confidence_threshold,
            min_layers: config.min_layers as u32,
            max_skip: config.max_skip as u32,
            _pad: [0; 2],
        };

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(confidence_scores), 0);
        encoder.set_buffer(1, Some(&output), 0);

        let params_size = mem::size_of::<LayerSkipDecisionParams>() as u64;
        encoder.set_bytes(2, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Perform early exit from transformer layers (f32).
    pub fn early_exit_f32(
        &self,
        hidden_states: &Buffer,
        lm_head_weight: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
        current_layer: usize,
        total_layers: usize,
        confidence_threshold: f32,
    ) -> Result<(Buffer, Buffer), SpecEEError> {
        self.early_exit_impl(
            hidden_states,
            lm_head_weight,
            batch_size,
            seq_len,
            hidden_dim,
            vocab_size,
            current_layer,
            total_layers,
            confidence_threshold,
            &self.pipeline_early_exit_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Perform early exit from transformer layers (f16).
    pub fn early_exit_f16(
        &self,
        hidden_states: &Buffer,
        lm_head_weight: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
        current_layer: usize,
        total_layers: usize,
        confidence_threshold: f32,
    ) -> Result<(Buffer, Buffer), SpecEEError> {
        self.early_exit_impl(
            hidden_states,
            lm_head_weight,
            batch_size,
            seq_len,
            hidden_dim,
            vocab_size,
            current_layer,
            total_layers,
            confidence_threshold,
            &self.pipeline_early_exit_f16,
            mem::size_of::<u16>(),
        )
    }

    fn early_exit_impl(
        &self,
        hidden_states: &Buffer,
        lm_head_weight: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
        current_layer: usize,
        total_layers: usize,
        confidence_threshold: f32,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<(Buffer, Buffer), SpecEEError> {
        let params = EarlyExitParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            vocab_size: vocab_size as u32,
            current_layer: current_layer as u32,
            total_layers: total_layers as u32,
            confidence_threshold,
            _pad: 0,
        };

        // Output: logits [batch, seq_len, vocab_size] and exit flags [batch, seq_len]
        let logits_elements = batch_size * seq_len * vocab_size;
        let logits_bytes = (logits_elements * element_size) as u64;
        let exit_flags_elements = batch_size * seq_len;
        let exit_flags_bytes = (exit_flags_elements * mem::size_of::<u32>()) as u64;

        let logits = self.device.new_buffer(logits_bytes, MTLResourceOptions::StorageModeShared);
        let exit_flags = self.device.new_buffer(exit_flags_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(hidden_states), 0);
        encoder.set_buffer(1, Some(lm_head_weight), 0);
        encoder.set_buffer(2, Some(&logits), 0);
        encoder.set_buffer(3, Some(&exit_flags), 0);

        let params_size = mem::size_of::<EarlyExitParams>() as u64;
        encoder.set_bytes(4, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * seq_len) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((logits, exit_flags))
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, SpecEEError> {
    SPEC_EE_METALLIB.load(device).map_err(SpecEEError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, SpecEEError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| SpecEEError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(SpecEEError::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
