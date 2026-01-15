//! CUDA FlashAttention-style kernels using cudarc 0.18.
//!
//! This module provides a naive, correctness-focused GPU kernel that mirrors
//! flash-attention semantics. It favors clarity over performance and serves as
//! an integration point for precompiled PTX.
//!
//! # SM-Aware PTX Loading
//!
//! This module automatically selects the best PTX binary for the detected GPU:
//! - SM 61 (Pascal): GTX 1060/1070/1080
//! - SM 75 (Turing): RTX 2060/2070/2080
//! - SM 80 (Ampere): A100, RTX 30 series
//! - SM 86 (Ampere): RTX 3060/3070/3080/3090
//! - SM 89 (Ada): RTX 4060/4070/4080/4090
//! - SM 90 (Hopper): H100, H200
//!
//! If no matching PTX is found, NVRTC runtime compilation is used as fallback.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, DriverError, PushKernelArg};
use half::f16;

use crate::types::AttentionConfig;
use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_F32: &str = "tiled_attention_forward_f32";
const KERNEL_F16: &str = "tiled_attention_forward_f16";
const MAX_HEAD_DIM: usize = 256;
const DEFAULT_BLOCK: u32 = 128;

/// CUDA source for runtime compilation.
const KERNEL_SOURCE: &str = include_str!("kernels/tiled_attention.cu");

/// SM-aware PTX collection for tiled attention kernel.
/// PTX compiled for a lower SM version is forward-compatible with higher SM GPUs.
static TILED_ATTENTION_PTX: PtxCollection = PtxCollection {
    kernel_name: "tiled_attention",
    source: KERNEL_SOURCE,
    ptx_versions: &[
        // SM 61 (Pascal) - GTX 1060/1070/1080
        (61, include_str!("kernels/tiled_attention_sm61.ptx")),
        // SM 80 (Ampere) - default for A100/RTX 30 series and higher
        (80, include_str!("kernels/tiled_attention.ptx")),
    ],
};

/// Errors surfaced by the CUDA FlashAttention kernels.
#[derive(Debug)]
pub enum FlashAttentionError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
}

impl fmt::Display for FlashAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for FlashAttentionError {}

impl From<DriverError> for FlashAttentionError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for FlashAttentionError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// FlashAttention CUDA kernel wrapper.
pub struct FlashAttentionKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_f32: CudaFunction,
    kernel_f16: CudaFunction,
}

impl FlashAttentionKernel {
    /// Load a FlashAttention kernel module on the given device.
    ///
    /// This method automatically selects the best PTX binary for the detected GPU:
    /// - Uses precompiled PTX for matching SM version
    /// - Falls back to NVRTC runtime compilation if needed
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, FlashAttentionError> {
        let ptx = TILED_ATTENTION_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_f32 = module
            .load_function(KERNEL_F32)
            .map_err(|_| FlashAttentionError::KernelMissing(KERNEL_F32))?;
        let kernel_f16 = module
            .load_function(KERNEL_F16)
            .map_err(|_| FlashAttentionError::KernelMissing(KERNEL_F16))?;

        Ok(Self {
            module,
            kernel_f32,
            kernel_f16,
        })
    }

    /// FlashAttention forward for f16 inputs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<CudaSlice<f16>, FlashAttentionError> {
        self.forward_f16(
            stream,
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
        )
    }

    /// FlashAttention forward for f32 inputs.
    pub fn forward_f32(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<CudaSlice<f32>, FlashAttentionError> {
        self.forward_f32_with_block(
            stream,
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
            DEFAULT_BLOCK,
        )
    }

    /// FlashAttention forward for f16 inputs.
    pub fn forward_f16(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<CudaSlice<f16>, FlashAttentionError> {
        self.forward_f16_with_block(
            stream,
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
            DEFAULT_BLOCK,
        )
    }

    fn forward_f32_with_block(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
        block_size: u32,
    ) -> Result<CudaSlice<f32>, FlashAttentionError> {
        let (output_len, cfg) = build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;
        let is_causal_i32 = if is_causal { 1i32 } else { 0i32 };
        let position_offset_i32 = i32::try_from(position_offset)
            .map_err(|_| FlashAttentionError::InvalidConfig("position_offset too large".into()))?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let seq_len_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f32);
            builder.arg(q);
            builder.arg(k);
            builder.arg(v);
            builder.arg(&mut output);
            builder.arg(&batch_size_i32);
            builder.arg(&num_heads_i32);
            builder.arg(&seq_len_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&scale);
            builder.arg(&is_causal_i32);
            builder.arg(&position_offset_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    fn forward_f16_with_block(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
        block_size: u32,
    ) -> Result<CudaSlice<f16>, FlashAttentionError> {
        let (output_len, cfg) = build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_len)?;
        let is_causal_i32 = if is_causal { 1i32 } else { 0i32 };
        let position_offset_i32 = i32::try_from(position_offset)
            .map_err(|_| FlashAttentionError::InvalidConfig("position_offset too large".into()))?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let seq_len_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f16);
            builder.arg(q);
            builder.arg(k);
            builder.arg(v);
            builder.arg(&mut output);
            builder.arg(&batch_size_i32);
            builder.arg(&num_heads_i32);
            builder.arg(&seq_len_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&scale);
            builder.arg(&is_causal_i32);
            builder.arg(&position_offset_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }
}

/// CUDA attention wrapper that uses a naive tiled kernel.
pub struct OptimizedCudaAttention {
    tile_size: u32,
    kernel: FlashAttentionKernel,
}

impl OptimizedCudaAttention {
    /// Create a tiled CUDA attention wrapper for f32 inputs.
    pub fn new(ctx: &Arc<CudaContext>, tile_size: usize) -> Result<Self, FlashAttentionError> {
        let tile_size = tile_size.clamp(1, 1024) as u32;
        let kernel = FlashAttentionKernel::new(ctx)?;
        Ok(Self {
            tile_size,
            kernel,
        })
    }

    /// Forward pass using the tiled attention kernel.
    pub fn forward_tiled(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        config: &AttentionConfig,
        position_offset: usize,
    ) -> Result<CudaSlice<f32>, FlashAttentionError> {
        if config.query_len != config.kv_len {
            return Err(FlashAttentionError::InvalidConfig(
                "query_len must match kv_len for the tiled kernel".into(),
            ));
        }

        let output = self.kernel.forward_f32_with_block(
            stream,
            q,
            k,
            v,
            config.batch_size,
            config.num_heads,
            config.query_len,
            config.head_dim,
            config.causal,
            config.scale,
            position_offset,
            self.tile_size,
        )?;

        Ok(output)
    }
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: u32,
) -> Result<(usize, LaunchConfig), FlashAttentionError> {
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
    if batch_size > i32::MAX as usize
        || num_heads > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || head_dim > i32::MAX as usize
    {
        return Err(FlashAttentionError::InvalidConfig(
            "Dimensions exceed i32::MAX".into(),
        ));
    }

    let num_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| FlashAttentionError::InvalidConfig("num_queries overflow".into()))?;

    let output_len = num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| FlashAttentionError::InvalidConfig("output_len overflow".into()))?;

    let block_dim = block_size.clamp(1, 1024) as usize;
    let grid_dim = (num_queries + block_dim - 1) / block_dim;
    let grid_dim = u32::try_from(grid_dim).map_err(|_| {
        FlashAttentionError::InvalidConfig("grid_dim exceeds u32::MAX".into())
    })?;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    Ok((output_len, cfg))
}
