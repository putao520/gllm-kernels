//! CUDA fused QKV projection + attention kernel wrapper.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_F32: &str = "fused_qkv_attention_forward";
const KERNEL_F16: &str = "fused_qkv_attention_forward_f16";
const MAX_HEAD_DIM: usize = 256;
const DEFAULT_BLOCK: u32 = 128;
const BLOCK_M: usize = 16;
const BLOCK_N: usize = 16;

/// CUDA source for runtime compilation.
const KERNEL_SOURCE: &str = include_str!("kernels/fused_qkv_attention.cu");

/// SM-aware PTX collection for fused QKV attention kernel.
static FUSED_QKV_ATTENTION_PTX: PtxCollection = PtxCollection {
    kernel_name: "fused_qkv_attention",
    source: KERNEL_SOURCE,
    ptx_versions: &[
        // SM 61 (Pascal) - GTX 1060/1070/1080
        (61, include_str!("kernels/fused_qkv_attention_sm61.ptx")),
        // SM 80 (Ampere) - default for A100/RTX 30 series and higher
        (80, include_str!("kernels/fused_qkv_attention.ptx")),
    ],
};

/// Errors surfaced by the fused QKV attention kernel.
#[derive(Debug)]
pub enum FusedQKVAttentionError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
}

impl fmt::Display for FusedQKVAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for FusedQKVAttentionError {}

impl From<DriverError> for FusedQKVAttentionError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for FusedQKVAttentionError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// Fused QKV projection + attention CUDA kernel wrapper.
pub struct FusedQKVAttentionKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_f32: CudaFunction,
    kernel_f16: CudaFunction,
}

impl FusedQKVAttentionKernel {
    /// Load the fused attention kernel module on the given device.
    ///
    /// This method automatically selects the best PTX binary for the detected GPU:
    /// - Uses precompiled PTX for matching SM version
    /// - Falls back to NVRTC runtime compilation if needed
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, FusedQKVAttentionError> {
        let ptx = FUSED_QKV_ATTENTION_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_f32 = module
            .load_function(KERNEL_F32)
            .map_err(|_| FusedQKVAttentionError::KernelMissing(KERNEL_F32))?;
        let kernel_f16 = module
            .load_function(KERNEL_F16)
            .map_err(|_| FusedQKVAttentionError::KernelMissing(KERNEL_F16))?;

        Ok(Self {
            module,
            kernel_f32,
            kernel_f16,
        })
    }

    /// Fused attention forward for f16 inputs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f16>,
        w_qkv: &CudaSlice<f16>,
        b_qkv: &CudaSlice<f16>,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f16>, FusedQKVAttentionError> {
        self.forward_f16(
            stream,
            input,
            w_qkv,
            b_qkv,
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            head_dim,
        )
    }

    /// Fused attention forward for f32 inputs.
    pub fn forward_f32(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        w_qkv: &CudaSlice<f32>,
        b_qkv: &CudaSlice<f32>,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, FusedQKVAttentionError> {
        if hidden_dim == 0 || hidden_dim > i32::MAX as usize {
            return Err(FusedQKVAttentionError::InvalidConfig(
                "hidden_dim must be within (0, i32::MAX]".into(),
            ));
        }
        let (output_len, cfg) = build_launch(batch_size, num_heads, seq_len, head_dim)?;

        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let hidden_i32 = hidden_dim as i32;
        let heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f32);
            builder.arg(input);
            builder.arg(w_qkv);
            builder.arg(b_qkv);
            builder.arg(&mut output);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&hidden_i32);
            builder.arg(&heads_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Fused attention forward for f16 inputs.
    pub fn forward_f16(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f16>,
        w_qkv: &CudaSlice<f16>,
        b_qkv: &CudaSlice<f16>,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f16>, FusedQKVAttentionError> {
        if hidden_dim == 0 || hidden_dim > i32::MAX as usize {
            return Err(FusedQKVAttentionError::InvalidConfig(
                "hidden_dim must be within (0, i32::MAX]".into(),
            ));
        }
        let (output_len, cfg) = build_launch(batch_size, num_heads, seq_len, head_dim)?;

        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_len)?;

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let hidden_i32 = hidden_dim as i32;
        let heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f16);
            builder.arg(input);
            builder.arg(w_qkv);
            builder.arg(b_qkv);
            builder.arg(&mut output);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&hidden_i32);
            builder.arg(&heads_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(usize, LaunchConfig), FusedQKVAttentionError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err(FusedQKVAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }

    if head_dim > MAX_HEAD_DIM {
        return Err(FusedQKVAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        )));
    }

    if batch_size > i32::MAX as usize
        || num_heads > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || head_dim > i32::MAX as usize
    {
        return Err(FusedQKVAttentionError::InvalidConfig(
            "Dimensions exceed i32::MAX".into(),
        ));
    }

    let q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    let grid_dim = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(q_blocks))
        .ok_or_else(|| FusedQKVAttentionError::InvalidConfig("grid_dim overflow".into()))?;

    let grid_dim = u32::try_from(grid_dim).map_err(|_| {
        FusedQKVAttentionError::InvalidConfig("grid_dim exceeds u32::MAX".into())
    })?;

    let output_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| FusedQKVAttentionError::InvalidConfig("output_len overflow".into()))?;

    let block_dim = DEFAULT_BLOCK.max(BLOCK_M as u32).min(1024);

    let shared_floats = (BLOCK_M + 2 * BLOCK_N)
        .checked_mul(head_dim)
        .ok_or_else(|| FusedQKVAttentionError::InvalidConfig("shared_mem overflow".into()))?;
    let shared_bytes = shared_floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| FusedQKVAttentionError::InvalidConfig("shared_mem overflow".into()))?;
    let shared_bytes_u32 = u32::try_from(shared_bytes).map_err(|_| {
        FusedQKVAttentionError::InvalidConfig("shared_mem exceeds u32::MAX".into())
    })?;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: shared_bytes_u32,
    };

    Ok((output_len, cfg))
}
