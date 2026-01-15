//! CUDA online softmax kernel wrapper.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_F32: &str = "online_softmax_forward";
const MIN_BLOCK: usize = 32;
const MAX_BLOCK: usize = 256;

/// CUDA source for runtime compilation.
const KERNEL_SOURCE: &str = include_str!("kernels/online_softmax.cu");

/// SM-aware PTX collection for online softmax kernel.
static ONLINE_SOFTMAX_PTX: PtxCollection = PtxCollection {
    kernel_name: "online_softmax",
    source: KERNEL_SOURCE,
    ptx_versions: &[
        // SM 61 (Pascal) - GTX 1060/1070/1080
        (61, include_str!("kernels/online_softmax_sm61.ptx")),
        // SM 80 (Ampere) - default for A100/RTX 30 series and higher
        (80, include_str!("kernels/online_softmax.ptx")),
    ],
};

/// Errors surfaced by the CUDA online softmax kernel.
#[derive(Debug)]
pub enum OnlineSoftmaxError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
}

impl fmt::Display for OnlineSoftmaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for OnlineSoftmaxError {}

impl From<DriverError> for OnlineSoftmaxError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for OnlineSoftmaxError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// Outputs produced by the online softmax kernel.
pub struct OnlineSoftmaxOutput {
    pub output: CudaSlice<f32>,
    pub max_val: CudaSlice<f32>,
    pub sum_exp: CudaSlice<f32>,
}

/// Online softmax CUDA kernel wrapper.
pub struct OnlineSoftmaxKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_f32: CudaFunction,
}

impl OnlineSoftmaxKernel {
    /// Load an online softmax kernel module on the given device.
    ///
    /// This method automatically selects the best PTX binary for the detected GPU:
    /// - Uses precompiled PTX for matching SM version
    /// - Falls back to NVRTC runtime compilation if needed
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, OnlineSoftmaxError> {
        let ptx = ONLINE_SOFTMAX_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_f32 = module
            .load_function(KERNEL_F32)
            .map_err(|_| OnlineSoftmaxError::KernelMissing(KERNEL_F32))?;

        Ok(Self { module, kernel_f32 })
    }

    /// Online softmax forward for f32 inputs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
    ) -> Result<OnlineSoftmaxOutput, OnlineSoftmaxError> {
        let (output_len, stats_len, cfg) = build_launch(batch_size, num_heads, seq_len)?;

        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;
        let mut max_val: CudaSlice<f32> = stream.alloc_zeros(stats_len)?;
        let mut sum_exp: CudaSlice<f32> = stream.alloc_zeros(stats_len)?;

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let seq_i32 = seq_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f32);
            builder.arg(logits);
            builder.arg(&mut output);
            builder.arg(&mut max_val);
            builder.arg(&mut sum_exp);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&seq_i32);
            builder.launch(cfg)?;
        }

        Ok(OnlineSoftmaxOutput {
            output,
            max_val,
            sum_exp,
        })
    }
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
) -> Result<(usize, usize, LaunchConfig), OnlineSoftmaxError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 {
        return Err(OnlineSoftmaxError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if batch_size > i32::MAX as usize || num_heads > i32::MAX as usize || seq_len > i32::MAX as usize {
        return Err(OnlineSoftmaxError::InvalidConfig(
            "Dimensions exceed i32::MAX".into(),
        ));
    }

    let total_rows = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| OnlineSoftmaxError::InvalidConfig("row count overflow".into()))?;
    let output_len = total_rows
        .checked_mul(seq_len)
        .ok_or_else(|| OnlineSoftmaxError::InvalidConfig("output_len overflow".into()))?;

    let grid_dim = u32::try_from(total_rows)
        .map_err(|_| OnlineSoftmaxError::InvalidConfig("grid_dim exceeds u32::MAX".into()))?;

    let mut block_dim = seq_len
        .checked_next_power_of_two()
        .unwrap_or(MAX_BLOCK);
    block_dim = block_dim.clamp(MIN_BLOCK, MAX_BLOCK);

    let shared_bytes = block_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| OnlineSoftmaxError::InvalidConfig("shared_mem overflow".into()))?;
    let shared_bytes_u32 = u32::try_from(shared_bytes).map_err(|_| {
        OnlineSoftmaxError::InvalidConfig("shared_mem exceeds u32::MAX".into())
    })?;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: shared_bytes_u32,
    };

    Ok((output_len, total_rows, cfg))
}
