//! CUDA selective scan kernels for Mamba-style SSMs.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const KERNEL_F32: &str = "selective_scan_fwd";
const PRECOMPILED_PTX: &str = include_str!("kernels/selective_scan.ptx");
const KERNEL_SOURCE: &str = include_str!("kernels/selective_scan.cu");
const MIN_BLOCK: usize = 32;
const MAX_BLOCK: usize = 256;

/// Errors surfaced by the CUDA selective scan kernel.
#[derive(Debug)]
pub enum SelectiveScanError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
}

impl fmt::Display for SelectiveScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
        }
    }
}

impl std::error::Error for SelectiveScanError {}

impl From<DriverError> for SelectiveScanError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

/// Selective scan CUDA kernel wrapper.
pub struct SelectiveScanKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_f32: CudaFunction,
}

impl SelectiveScanKernel {
    /// Load a selective scan kernel module on the given device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, SelectiveScanError> {
        let ptx = load_ptx()?;
        let module = ctx.load_module(ptx)?;

        let kernel_f32 = module
            .load_function(KERNEL_F32)
            .map_err(|_| SelectiveScanError::KernelMissing(KERNEL_F32))?;

        Ok(Self { module, kernel_f32 })
    }

    /// Selective scan forward for f32 inputs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        u: &CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &CudaSlice<f32>,
        batch_size: usize,
        seq_len: usize,
        state_dim: usize,
        expanded_dim: usize,
    ) -> Result<CudaSlice<f32>, SelectiveScanError> {
        let (output_len, cfg) = build_launch(batch_size, seq_len, state_dim, expanded_dim)?;

        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let state_i32 = state_dim as i32;
        let expanded_i32 = expanded_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f32);
            builder.arg(u);
            builder.arg(delta);
            builder.arg(a);
            builder.arg(b);
            builder.arg(c);
            builder.arg(&mut output);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&state_i32);
            builder.arg(&expanded_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }
}

fn load_ptx() -> Result<Ptx, SelectiveScanError> {
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
            SelectiveScanError::InvalidConfig(format!("NVRTC compilation failed: {}", e))
        });
    }

    #[cfg(not(feature = "nvrtc"))]
    Err(SelectiveScanError::InvalidConfig(
        "PTX is a placeholder. Either: \n\
         1. Compile with: nvcc -ptx -arch=sm_61 selective_scan.cu -o selective_scan.ptx\n\
         2. Enable 'nvrtc' feature and ensure CUDA toolkit is installed".into(),
    ))
}

fn build_launch(
    batch_size: usize,
    seq_len: usize,
    state_dim: usize,
    expanded_dim: usize,
) -> Result<(usize, LaunchConfig), SelectiveScanError> {
    if batch_size == 0 || seq_len == 0 || state_dim == 0 || expanded_dim == 0 {
        return Err(SelectiveScanError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }

    if batch_size > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || state_dim > i32::MAX as usize
        || expanded_dim > i32::MAX as usize
    {
        return Err(SelectiveScanError::InvalidConfig(
            "Dimensions exceed i32::MAX".into(),
        ));
    }

    let output_len = batch_size
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(expanded_dim))
        .ok_or_else(|| SelectiveScanError::InvalidConfig("output_len overflow".into()))?;

    let grid_dim = batch_size
        .checked_mul(expanded_dim)
        .ok_or_else(|| SelectiveScanError::InvalidConfig("grid_dim overflow".into()))?;
    let grid_dim = u32::try_from(grid_dim)
        .map_err(|_| SelectiveScanError::InvalidConfig("grid_dim exceeds u32::MAX".into()))?;

    let mut block_dim = state_dim
        .checked_next_power_of_two()
        .unwrap_or(MAX_BLOCK);
    block_dim = block_dim.max(MIN_BLOCK).min(MAX_BLOCK);

    let shared_floats = state_dim
        .checked_mul(2)
        .and_then(|value| value.checked_add(block_dim))
        .ok_or_else(|| SelectiveScanError::InvalidConfig("shared_mem overflow".into()))?;
    let shared_bytes = shared_floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| SelectiveScanError::InvalidConfig("shared_mem overflow".into()))?;
    let shared_bytes_u32 = u32::try_from(shared_bytes).map_err(|_| {
        SelectiveScanError::InvalidConfig("shared_mem exceeds u32::MAX".into())
    })?;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: shared_bytes_u32,
    };

    Ok((output_len, cfg))
}
