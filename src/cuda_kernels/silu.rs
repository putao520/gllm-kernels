//! CUDA SiLU kernel wrapper.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_F32: &str = "silu_f32";
const KERNEL_INPLACE_F32: &str = "silu_inplace_f32";

/// SM-aware PTX collection for SiLU kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static SILU_PTX: PtxCollection = PtxCollection {
    kernel_name: "silu",
    ptx_versions: &[
        (61, include_str!("kernels/silu_sm61.ptx")),
        (80, include_str!("kernels/silu.ptx")),
    ],
};

/// Errors surfaced by the CUDA SiLU kernel.
#[derive(Debug)]
pub enum SiluError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
}

impl fmt::Display for SiluError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for SiluError {}

impl From<DriverError> for SiluError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for SiluError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// SiLU CUDA kernel wrapper.
pub struct CudaSilu {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_f32: CudaFunction,
    kernel_inplace_f32: CudaFunction,
}

impl CudaSilu {
    /// Load a SiLU kernel module on the given device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, SiluError> {
        let ptx = SILU_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_f32 = module
            .load_function(KERNEL_F32)
            .map_err(|_| SiluError::KernelMissing(KERNEL_F32))?;
        let kernel_inplace_f32 = module
            .load_function(KERNEL_INPLACE_F32)
            .map_err(|_| SiluError::KernelMissing(KERNEL_INPLACE_F32))?;

        Ok(Self {
            module,
            kernel_f32,
            kernel_inplace_f32,
        })
    }

    /// SiLU forward for f32 inputs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        len: usize,
    ) -> Result<(), SiluError> {
        if len == 0 {
            return Ok(());
        }

        let len_u32 = u32::try_from(len)
            .map_err(|_| SiluError::InvalidConfig("len exceeds u32".into()))?;
        let len_i32 = i32::try_from(len)
            .map_err(|_| SiluError::InvalidConfig("len exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(len_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f32);
            builder.arg(input);
            builder.arg(output);
            builder.arg(&len_i32);
            builder.launch(cfg)
        }
        .map_err(SiluError::Driver)?;

        Ok(())
    }

    /// SiLU forward in-place for f32 inputs.
    pub fn forward_inplace(
        &self,
        stream: &Arc<CudaStream>,
        data: &mut CudaSlice<f32>,
        len: usize,
    ) -> Result<(), SiluError> {
        if len == 0 {
            return Ok(());
        }

        let len_u32 = u32::try_from(len)
            .map_err(|_| SiluError::InvalidConfig("len exceeds u32".into()))?;
        let len_i32 = i32::try_from(len)
            .map_err(|_| SiluError::InvalidConfig("len exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(len_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_inplace_f32);
            builder.arg(data);
            builder.arg(&len_i32);
            builder.launch(cfg)
        }
        .map_err(SiluError::Driver)?;

        Ok(())
    }
}
