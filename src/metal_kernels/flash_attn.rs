//! Metal FlashAttention-style kernels.
//!
//! This module provides a naive, correctness-focused Metal kernel wrapper
//! that mirrors the CUDA/HIP tiled attention entry points.
//!
//! ## Precompiled metallib Support
//!
//! For production use, precompile the shader using:
//! ```bash
//! ./scripts/compile_metal_kernels.sh
//! ```
//!
//! The kernel loader will automatically use precompiled metallib if available,
//! falling back to runtime compilation from source.

use std::fmt;
use std::mem;
use std::os::raw::c_void;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};

use crate::metal_kernels::metallib_loader::{MetallibCollection, MetallibLoadError};

const KERNEL_F32: &str = "tiled_attention_forward_f32";
const KERNEL_F16: &str = "tiled_attention_forward_f16";
const MAX_HEAD_DIM: usize = 256;

/// Metallib collection for flash attention kernel.
static FLASH_ATTENTION_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "flash_attention",
    metallib_data: include_bytes!("kernels/flash_attention.metallib"),
    source: include_str!("kernels/flash_attention.metal"),
};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct AttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    is_causal: u32,
    position_offset: u32,
    _pad: u32,
}

/// Errors surfaced by the Metal FlashAttention kernels.
#[derive(Debug)]
pub enum FlashAttentionError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for FlashAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for FlashAttentionError {}

impl From<MetallibLoadError> for FlashAttentionError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// FlashAttention Metal kernel wrapper.
pub struct FlashAttentionKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_f32: ComputePipelineState,
    pipeline_f16: ComputePipelineState,
}

impl FlashAttentionKernel {
    /// Load FlashAttention kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, FlashAttentionError> {
        let library = load_library(device)?;
        let pipeline_f32 = build_pipeline(device, &library, KERNEL_F32)?;
        let pipeline_f16 = build_pipeline(device, &library, KERNEL_F16)?;
        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_f32,
            pipeline_f16,
        })
    }

    /// FlashAttention forward for f32 inputs.
    pub fn forward_f32(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<Buffer, FlashAttentionError> {
        self.forward_impl(
            q,
            k,
            v,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            is_causal,
            scale,
            position_offset,
            &self.pipeline_f32,
            mem::size_of::<f32>(),
        )
    }

    /// FlashAttention forward for f16 inputs.
    pub fn forward_f16(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<Buffer, FlashAttentionError> {
        self.forward_impl(
            q,
            k,
            v,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            is_causal,
            scale,
            position_offset,
            &self.pipeline_f16,
            mem::size_of::<u16>(),
        )
    }

    fn forward_impl(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, FlashAttentionError> {
        let (params, total_queries, output_len) = build_params(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            is_causal,
            scale,
            position_offset,
        )?;

        let output_len_u64 = u64::try_from(output_len)
            .map_err(|_| FlashAttentionError::InvalidConfig("output_len exceeds u64".into()))?;
        let output_bytes = output_len_u64
            .checked_mul(element_size as u64)
            .ok_or_else(|| FlashAttentionError::InvalidConfig("output size overflow".into()))?;

        validate_buffer(q, output_bytes, "q")?;
        validate_buffer(k, output_bytes, "k")?;
        validate_buffer(v, output_bytes, "v")?;

        let output = self
            .device
            .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k), 0);
        encoder.set_buffer(2, Some(v), 0);
        encoder.set_buffer(3, Some(&output), 0);

        let params_size = mem::size_of::<AttentionParams>() as u64;
        encoder.set_bytes(4, params_size, &params as *const _ as *const c_void);

        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        let threads_per_grid = MTLSize::new(total_queries as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }
}

/// Load Metal library using MetallibCollection with automatic fallback.
///
/// Priority:
/// 1. Embedded metallib - always tried first
/// 2. Runtime compilation from source (automatic fallback)
fn load_library(device: &Device) -> Result<Library, FlashAttentionError> {
    FLASH_ATTENTION_METALLIB.load(device).map_err(FlashAttentionError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, FlashAttentionError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| FlashAttentionError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(FlashAttentionError::Metal)
}

fn build_params(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    is_causal: bool,
    scale: f32,
    position_offset: usize,
) -> Result<(AttentionParams, usize, usize), FlashAttentionError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err(FlashAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(FlashAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        )));
    }

    let batch_size_u32 = u32::try_from(batch_size)
        .map_err(|_| FlashAttentionError::InvalidConfig("batch_size exceeds u32".into()))?;
    let num_heads_u32 = u32::try_from(num_heads)
        .map_err(|_| FlashAttentionError::InvalidConfig("num_heads exceeds u32".into()))?;
    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|_| FlashAttentionError::InvalidConfig("seq_len exceeds u32".into()))?;
    let head_dim_u32 = u32::try_from(head_dim)
        .map_err(|_| FlashAttentionError::InvalidConfig("head_dim exceeds u32".into()))?;
    let position_offset_u32 = u32::try_from(position_offset).map_err(|_| {
        FlashAttentionError::InvalidConfig("position_offset exceeds u32".into())
    })?;

    let total_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| FlashAttentionError::InvalidConfig("num_queries overflow".into()))?;
    if total_queries > u32::MAX as usize {
        return Err(FlashAttentionError::InvalidConfig(
            "num_queries exceeds u32".into(),
        ));
    }

    let output_len = total_queries
        .checked_mul(head_dim)
        .ok_or_else(|| FlashAttentionError::InvalidConfig("output_len overflow".into()))?;

    let params = AttentionParams {
        batch_size: batch_size_u32,
        num_heads: num_heads_u32,
        seq_len: seq_len_u32,
        head_dim: head_dim_u32,
        scale,
        is_causal: if is_causal { 1 } else { 0 },
        position_offset: position_offset_u32,
        _pad: 0,
    };

    Ok((params, total_queries, output_len))
}

fn validate_buffer(
    buffer: &Buffer,
    expected_bytes: u64,
    name: &str,
) -> Result<(), FlashAttentionError> {
    if buffer.length() < expected_bytes {
        return Err(FlashAttentionError::InvalidConfig(format!(
            "{name} buffer too small: {} < {}",
            buffer.length(),
            expected_bytes
        )));
    }
    Ok(())
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
