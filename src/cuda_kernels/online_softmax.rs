//! CUDA online softmax kernel wrapper.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

const KERNEL_F32: &str = "online_softmax_forward";
const PRECOMPILED_PTX: &str = include_str!("kernels/online_softmax.ptx");
const KERNEL_SOURCE: &str = include_str!("kernels/online_softmax.cu");
const MIN_BLOCK: usize = 32;
const MAX_BLOCK: usize = 256;

/// Errors surfaced by the CUDA online softmax kernel.
#[derive(Debug)]
pub enum OnlineSoftmaxError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
}

impl fmt::Display for OnlineSoftmaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
        }
    }
}

impl std::error::Error for OnlineSoftmaxError {}

impl From<DriverError> for OnlineSoftmaxError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
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
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, OnlineSoftmaxError> {
        let ptx = load_ptx()?;
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

fn load_ptx() -> Result<Ptx, OnlineSoftmaxError> {
    // Priority 1: Check if precompiled PTX is valid (not a placeholder)
    if !PRECOMPILED_PTX.contains("Placeholder") {
        log::debug!("Loading precompiled PTX from embedded data");
        return Ok(Ptx::from_src(PRECOMPILED_PTX));
    }

    // Priority 2: Try runtime compilation with NVRTC
    #[cfg(feature = "nvrtc")]
    {
        log::debug!("Compiling PTX from source at runtime");
        use cudarc::nvrtc::compile_ptx;
        return compile_ptx(KERNEL_SOURCE).map_err(|e| {
            OnlineSoftmaxError::InvalidConfig(format!("NVRTC compilation failed: {}", e))
        });
    }

    #[cfg(not(feature = "nvrtc"))]
    Err(OnlineSoftmaxError::InvalidConfig(
        "PTX is a placeholder. Either: \n\
         1. Compile with: nvcc -ptx -arch=sm_61 online_softmax.cu -o online_softmax.ptx\n\
         2. Enable 'nvrtc' feature and ensure CUDA toolkit is installed".into(),
    ))
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
