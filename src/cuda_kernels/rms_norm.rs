//! CUDA RMSNorm kernel wrapper.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_F32: &str = "rms_norm_f32";

/// SM-aware PTX collection for RMSNorm kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static RMS_NORM_PTX: PtxCollection = PtxCollection {
    kernel_name: "rms_norm",
    ptx_versions: &[
        (61, include_str!("kernels/rms_norm_sm61.ptx")),
        (80, include_str!("kernels/rms_norm.ptx")),
    ],
};

/// Errors surfaced by the CUDA RMSNorm kernel.
#[derive(Debug)]
pub enum RmsNormError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
}

impl fmt::Display for RmsNormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for RmsNormError {}

impl From<DriverError> for RmsNormError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for RmsNormError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// RMSNorm CUDA kernel wrapper.
pub struct RmsNormKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_f32: CudaFunction,
}

impl RmsNormKernel {
    /// Load an RMSNorm kernel module on the given device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, RmsNormError> {
        let ptx = RMS_NORM_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_f32 = module
            .load_function(KERNEL_F32)
            .map_err(|_| RmsNormError::KernelMissing(KERNEL_F32))?;

        Ok(Self { module, kernel_f32 })
    }

    /// RMSNorm forward for f32 inputs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u8>,
        weight: &CudaSlice<u8>,
        output: &CudaSlice<u8>,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), RmsNormError> {
        if rows == 0 || hidden == 0 {
            return Ok(());
        }

        let rows_u32 = u32::try_from(rows)
            .map_err(|_| RmsNormError::InvalidConfig("rows exceeds u32".into()))?;
        let hidden_i32 = i32::try_from(hidden)
            .map_err(|_| RmsNormError::InvalidConfig("hidden exceeds i32".into()))?;
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| RmsNormError::InvalidConfig("rows exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(rows_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_f32);
            builder.arg(input);
            builder.arg(weight);
            builder.arg(output);
            builder.arg(&hidden_i32);
            builder.arg(&rows_i32);
            builder.arg(&eps);
            builder.launch(cfg)
        }
        .map_err(|err| RmsNormError::Driver(err))?;

        Ok(())
    }
}
