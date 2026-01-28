//! CUDA elementwise and permutation kernels.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, CudaView, CudaViewMut, DriverError,
    LaunchConfig, PushKernelArg,
};

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_ADD_F32: &str = "elementwise_add_f32";
const KERNEL_PERMUTE_F32: &str = "permute_qkv_f32";
const KERNEL_PERMUTE_BACK_F32: &str = "permute_qkv_back_f32";

/// SM-aware PTX collection for elementwise kernels.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static ELEMENTWISE_PTX: PtxCollection = PtxCollection {
    kernel_name: "elementwise",
    ptx_versions: &[
        (61, include_str!("kernels/elementwise_sm61.ptx")),
        (80, include_str!("kernels/elementwise.ptx")),
    ],
};

/// Errors surfaced by the CUDA elementwise kernels.
#[derive(Debug)]
pub enum ElementwiseError {
    Driver(DriverError),
    InvalidConfig(String),
    KernelMissing(&'static str),
    PtxLoad(PtxLoadError),
}

impl fmt::Display for ElementwiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for ElementwiseError {}

impl From<DriverError> for ElementwiseError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for ElementwiseError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// CUDA elementwise kernel wrapper.
pub struct CudaElementwiseKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_add_f32: CudaFunction,
    kernel_permute_f32: CudaFunction,
    kernel_permute_back_f32: CudaFunction,
}

impl CudaElementwiseKernel {
    /// Load elementwise kernel module on the given device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, ElementwiseError> {
        let ptx = ELEMENTWISE_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_add_f32 = module
            .load_function(KERNEL_ADD_F32)
            .map_err(|_| ElementwiseError::KernelMissing(KERNEL_ADD_F32))?;
        let kernel_permute_f32 = module
            .load_function(KERNEL_PERMUTE_F32)
            .map_err(|_| ElementwiseError::KernelMissing(KERNEL_PERMUTE_F32))?;
        let kernel_permute_back_f32 = module
            .load_function(KERNEL_PERMUTE_BACK_F32)
            .map_err(|_| ElementwiseError::KernelMissing(KERNEL_PERMUTE_BACK_F32))?;

        Ok(Self {
            module,
            kernel_add_f32,
            kernel_permute_f32,
            kernel_permute_back_f32,
        })
    }

    pub fn add_view(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaView<'_, f32>,
        b: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        len: usize,
    ) -> Result<(), ElementwiseError> {
        if len == 0 {
            return Ok(());
        }

        let len_u32 = u32::try_from(len)
            .map_err(|_| ElementwiseError::InvalidConfig("len exceeds u32".into()))?;
        let len_i32 = i32::try_from(len)
            .map_err(|_| ElementwiseError::InvalidConfig("len exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(len_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_add_f32);
            builder.arg(a);
            builder.arg(b);
            builder.arg(output);
            builder.arg(&len_i32);
            builder.launch(cfg)
        }
        .map_err(ElementwiseError::Driver)?;

        Ok(())
    }

    pub fn permute_qkv_view(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), ElementwiseError> {
        let total = batch
            .checked_mul(seq)
            .and_then(|v| v.checked_mul(num_heads))
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| ElementwiseError::InvalidConfig("permute size overflow".into()))?;

        if total == 0 {
            return Ok(());
        }

        let total_u32 = u32::try_from(total)
            .map_err(|_| ElementwiseError::InvalidConfig("permute size exceeds u32".into()))?;
        let batch_i32 = i32::try_from(batch)
            .map_err(|_| ElementwiseError::InvalidConfig("batch exceeds i32".into()))?;
        let seq_i32 = i32::try_from(seq)
            .map_err(|_| ElementwiseError::InvalidConfig("seq exceeds i32".into()))?;
        let heads_i32 = i32::try_from(num_heads)
            .map_err(|_| ElementwiseError::InvalidConfig("num_heads exceeds i32".into()))?;
        let head_dim_i32 = i32::try_from(head_dim)
            .map_err(|_| ElementwiseError::InvalidConfig("head_dim exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(total_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_permute_f32);
            builder.arg(input);
            builder.arg(output);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&heads_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)
        }
        .map_err(ElementwiseError::Driver)?;

        Ok(())
    }

    pub fn permute_qkv_back_view(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), ElementwiseError> {
        let total = batch
            .checked_mul(seq)
            .and_then(|v| v.checked_mul(num_heads))
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| ElementwiseError::InvalidConfig("permute size overflow".into()))?;

        if total == 0 {
            return Ok(());
        }

        let total_u32 = u32::try_from(total)
            .map_err(|_| ElementwiseError::InvalidConfig("permute size exceeds u32".into()))?;
        let batch_i32 = i32::try_from(batch)
            .map_err(|_| ElementwiseError::InvalidConfig("batch exceeds i32".into()))?;
        let seq_i32 = i32::try_from(seq)
            .map_err(|_| ElementwiseError::InvalidConfig("seq exceeds i32".into()))?;
        let heads_i32 = i32::try_from(num_heads)
            .map_err(|_| ElementwiseError::InvalidConfig("num_heads exceeds i32".into()))?;
        let head_dim_i32 = i32::try_from(head_dim)
            .map_err(|_| ElementwiseError::InvalidConfig("head_dim exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(total_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_permute_back_f32);
            builder.arg(input);
            builder.arg(output);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&heads_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)
        }
        .map_err(ElementwiseError::Driver)?;

        Ok(())
    }
}
