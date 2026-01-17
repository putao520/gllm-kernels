//! Metal EAGLE-3 adaptive draft length speculative decoding kernels.
//!
//! This module provides Metal GPU-accelerated kernels for EAGLE-3 speculative decoding:
//! - Multi-layer feature fusion
//! - Token-level confidence prediction
//! - Adaptive draft length scheduling
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

const KERNEL_FUSE_HIDDEN_F32: &str = "eagle3_fuse_hidden_f32";
const KERNEL_FUSE_HIDDEN_F16: &str = "eagle3_fuse_hidden_f16";
const KERNEL_PREDICT_CONFIDENCE_F32: &str = "eagle3_predict_confidence_f32";
const KERNEL_PREDICT_CONFIDENCE_F16: &str = "eagle3_predict_confidence_f16";
const KERNEL_GENERATE_DRAFT_F32: &str = "eagle3_generate_draft_f32";
const KERNEL_GENERATE_DRAFT_F16: &str = "eagle3_generate_draft_f16";

/// Metallib collection for EAGLE-3 kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static EAGLE3_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "eagle3",
    metallib_data: include_bytes!("kernels/eagle3.metallib"),
};

/// Parameters for hidden state fusion kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FuseHiddenParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    fusion_layers: u32,
    fused_dim: u32,
    _pad: [u32; 3],
}

/// Parameters for confidence prediction kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PredictConfidenceParams {
    batch_size: u32,
    seq_len: u32,
    fused_dim: u32,
    bias: f32,
}

/// Parameters for draft generation kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GenerateDraftParams {
    seq_len: u32,
    vocab_size: u32,
    fused_dim: u32,
    max_draft_len: u32,
    confidence_threshold: f32,
    _pad: [u32; 3],
}

/// Errors surfaced by the Metal EAGLE-3 kernels.
#[derive(Debug)]
pub enum Eagle3Error {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for Eagle3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for Eagle3Error {}

impl From<MetallibLoadError> for Eagle3Error {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// EAGLE-3 Metal kernel wrapper.
pub struct Eagle3Kernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_fuse_hidden_f32: ComputePipelineState,
    pipeline_fuse_hidden_f16: ComputePipelineState,
    pipeline_predict_conf_f32: ComputePipelineState,
    pipeline_predict_conf_f16: ComputePipelineState,
    pipeline_generate_draft_f32: ComputePipelineState,
    pipeline_generate_draft_f16: ComputePipelineState,
}

impl Eagle3Kernel {
    /// Load EAGLE-3 kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, Eagle3Error> {
        let library = load_library(device)?;

        let pipeline_fuse_hidden_f32 = build_pipeline(device, &library, KERNEL_FUSE_HIDDEN_F32)?;
        let pipeline_fuse_hidden_f16 = build_pipeline(device, &library, KERNEL_FUSE_HIDDEN_F16)?;
        let pipeline_predict_conf_f32 = build_pipeline(device, &library, KERNEL_PREDICT_CONFIDENCE_F32)?;
        let pipeline_predict_conf_f16 = build_pipeline(device, &library, KERNEL_PREDICT_CONFIDENCE_F16)?;
        let pipeline_generate_draft_f32 = build_pipeline(device, &library, KERNEL_GENERATE_DRAFT_F32)?;
        let pipeline_generate_draft_f16 = build_pipeline(device, &library, KERNEL_GENERATE_DRAFT_F16)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_fuse_hidden_f32,
            pipeline_fuse_hidden_f16,
            pipeline_predict_conf_f32,
            pipeline_predict_conf_f16,
            pipeline_generate_draft_f32,
            pipeline_generate_draft_f16,
        })
    }

    /// Fuse hidden states from multiple layers (f32).
    ///
    /// # Arguments
    /// * `layer_hidden` - Hidden states from multiple layers: [fusion_layers][batch, seq_len, hidden_dim]
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension per layer
    /// * `fusion_layers` - Number of layers to fuse
    ///
    /// # Returns
    /// Fused hidden states: [batch, seq_len, hidden_dim * fusion_layers]
    pub fn fuse_hidden_f32(
        &self,
        layer_hidden: &[&Buffer],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        fusion_layers: usize,
    ) -> Result<Buffer, Eagle3Error> {
        self.fuse_hidden_impl(
            layer_hidden,
            batch_size,
            seq_len,
            hidden_dim,
            fusion_layers,
            &self.pipeline_fuse_hidden_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Fuse hidden states from multiple layers (f16).
    pub fn fuse_hidden_f16(
        &self,
        layer_hidden: &[&Buffer],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        fusion_layers: usize,
    ) -> Result<Buffer, Eagle3Error> {
        self.fuse_hidden_impl(
            layer_hidden,
            batch_size,
            seq_len,
            hidden_dim,
            fusion_layers,
            &self.pipeline_fuse_hidden_f16,
            mem::size_of::<u16>(),
        )
    }

    fn fuse_hidden_impl(
        &self,
        layer_hidden: &[&Buffer],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        fusion_layers: usize,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, Eagle3Error> {
        if layer_hidden.len() != fusion_layers {
            return Err(Eagle3Error::InvalidConfig(format!(
                "expected {} layers, got {}",
                fusion_layers,
                layer_hidden.len()
            )));
        }

        let fused_dim = hidden_dim * fusion_layers;
        let output_elements = batch_size * seq_len * fused_dim;
        let output_bytes = (output_elements * element_size) as u64;

        let params = FuseHiddenParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            fusion_layers: fusion_layers as u32,
            fused_dim: fused_dim as u32,
            _pad: [0; 3],
        };

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        // Bind layer hidden buffers
        for (i, buf) in layer_hidden.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }

        // Output buffer (after input buffers)
        encoder.set_buffer(fusion_layers as u64, Some(&output), 0);

        // Parameters
        let params_size = mem::size_of::<FuseHiddenParams>() as u64;
        encoder.set_bytes((fusion_layers + 1) as u64, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * seq_len) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Predict token-level confidence from fused hidden states (f32).
    ///
    /// # Arguments
    /// * `fused_hidden` - Fused hidden states: [batch, seq_len, fused_dim]
    /// * `weight` - Predictor weight: [fused_dim, 1]
    /// * `bias` - Predictor bias
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `fused_dim` - Fused hidden dimension
    ///
    /// # Returns
    /// Confidence probabilities: [batch, seq_len]
    pub fn predict_confidence_f32(
        &self,
        fused_hidden: &Buffer,
        weight: &Buffer,
        bias: f32,
        batch_size: usize,
        seq_len: usize,
        fused_dim: usize,
    ) -> Result<Buffer, Eagle3Error> {
        self.predict_confidence_impl(
            fused_hidden,
            weight,
            bias,
            batch_size,
            seq_len,
            fused_dim,
            &self.pipeline_predict_conf_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Predict token-level confidence from fused hidden states (f16).
    pub fn predict_confidence_f16(
        &self,
        fused_hidden: &Buffer,
        weight: &Buffer,
        bias: f32,
        batch_size: usize,
        seq_len: usize,
        fused_dim: usize,
    ) -> Result<Buffer, Eagle3Error> {
        self.predict_confidence_impl(
            fused_hidden,
            weight,
            bias,
            batch_size,
            seq_len,
            fused_dim,
            &self.pipeline_predict_conf_f16,
            mem::size_of::<u16>(),
        )
    }

    fn predict_confidence_impl(
        &self,
        fused_hidden: &Buffer,
        weight: &Buffer,
        bias: f32,
        batch_size: usize,
        seq_len: usize,
        fused_dim: usize,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, Eagle3Error> {
        let output_elements = batch_size * seq_len;
        let output_bytes = (output_elements * element_size) as u64;

        let params = PredictConfidenceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            fused_dim: fused_dim as u32,
            bias,
        };

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(fused_hidden), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(&output), 0);

        let params_size = mem::size_of::<PredictConfidenceParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(output_elements as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Generate draft tokens with confidence-based early termination (f32).
    ///
    /// # Arguments
    /// * `draft_logits` - Logits from draft model: [seq_len, vocab_size]
    /// * `fused_hidden` - Fused hidden states: [seq_len, fused_dim]
    /// * `weight` - Confidence predictor weight: [fused_dim, 1]
    /// * `seq_len` - Sequence length
    /// * `vocab_size` - Vocabulary size
    /// * `fused_dim` - Fused hidden dimension
    /// * `max_draft_len` - Maximum draft length
    /// * `confidence_threshold` - Confidence threshold for early termination
    ///
    /// # Returns
    /// Draft token IDs and their count: (tokens: Buffer, count: Buffer)
    pub fn generate_draft_f32(
        &self,
        draft_logits: &Buffer,
        fused_hidden: &Buffer,
        weight: &Buffer,
        seq_len: usize,
        vocab_size: usize,
        fused_dim: usize,
        max_draft_len: usize,
        confidence_threshold: f32,
    ) -> Result<(Buffer, Buffer), Eagle3Error> {
        self.generate_draft_impl(
            draft_logits,
            fused_hidden,
            weight,
            seq_len,
            vocab_size,
            fused_dim,
            max_draft_len,
            confidence_threshold,
            &self.pipeline_generate_draft_f32,
        )
    }

    /// Generate draft tokens with confidence-based early termination (f16).
    pub fn generate_draft_f16(
        &self,
        draft_logits: &Buffer,
        fused_hidden: &Buffer,
        weight: &Buffer,
        seq_len: usize,
        vocab_size: usize,
        fused_dim: usize,
        max_draft_len: usize,
        confidence_threshold: f32,
    ) -> Result<(Buffer, Buffer), Eagle3Error> {
        self.generate_draft_impl(
            draft_logits,
            fused_hidden,
            weight,
            seq_len,
            vocab_size,
            fused_dim,
            max_draft_len,
            confidence_threshold,
            &self.pipeline_generate_draft_f16,
        )
    }

    fn generate_draft_impl(
        &self,
        draft_logits: &Buffer,
        fused_hidden: &Buffer,
        weight: &Buffer,
        seq_len: usize,
        vocab_size: usize,
        fused_dim: usize,
        max_draft_len: usize,
        confidence_threshold: f32,
        pipeline: &ComputePipelineState,
    ) -> Result<(Buffer, Buffer), Eagle3Error> {
        let params = GenerateDraftParams {
            seq_len: seq_len as u32,
            vocab_size: vocab_size as u32,
            fused_dim: fused_dim as u32,
            max_draft_len: max_draft_len as u32,
            confidence_threshold,
            _pad: [0; 3],
        };

        // Output buffers: token IDs (u32) and count (u32)
        let tokens_bytes = (max_draft_len * mem::size_of::<u32>()) as u64;
        let count_bytes = mem::size_of::<u32>() as u64;

        let tokens = self.device.new_buffer(tokens_bytes, MTLResourceOptions::StorageModeShared);
        let count = self.device.new_buffer(count_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(draft_logits), 0);
        encoder.set_buffer(1, Some(fused_hidden), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(&tokens), 0);
        encoder.set_buffer(4, Some(&count), 0);

        let params_size = mem::size_of::<GenerateDraftParams>() as u64;
        encoder.set_bytes(5, params_size, &params as *const _ as *const c_void);

        // Single thread for sequential draft generation
        let threads_per_grid = MTLSize::new(1, 1, 1);
        let threads_per_threadgroup = MTLSize::new(1, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((tokens, count))
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, Eagle3Error> {
    EAGLE3_METALLIB.load(device).map_err(Eagle3Error::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, Eagle3Error> {
    let function = library
        .get_function(name, None)
        .map_err(|_| Eagle3Error::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(Eagle3Error::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
