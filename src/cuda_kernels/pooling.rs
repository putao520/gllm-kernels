//! CUDA embedding gather and pooling kernels.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, CudaView, CudaViewMut, DriverError,
    LaunchConfig, PushKernelArg,
};

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_EMBEDDING_GATHER_F32: &str = "embedding_gather_f32";
const KERNEL_MEAN_POOLING_F32: &str = "mean_pooling_f32";
const KERNEL_L2_NORMALIZE_F32: &str = "l2_normalize_f32";
const KERNEL_CLS_EXTRACT_F32: &str = "cls_extract_f32";

/// SM-aware PTX collection for pooling kernels.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static POOLING_PTX: PtxCollection = PtxCollection {
    kernel_name: "pooling",
    ptx_versions: &[
        (61, include_str!("kernels/pooling_sm61.ptx")),
        (80, include_str!("kernels/pooling.ptx")),
    ],
};

/// Errors surfaced by the CUDA pooling kernels.
#[derive(Debug)]
pub enum PoolingError {
    Driver(DriverError),
    InvalidConfig(String),
    KernelMissing(&'static str),
    PtxLoad(PtxLoadError),
}

impl fmt::Display for PoolingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for PoolingError {}

impl From<DriverError> for PoolingError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for PoolingError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// CUDA pooling kernel wrapper.
pub struct CudaPoolingKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_embedding_gather_f32: CudaFunction,
    kernel_mean_pooling_f32: CudaFunction,
    kernel_l2_normalize_f32: CudaFunction,
    kernel_cls_extract_f32: CudaFunction,
}

impl CudaPoolingKernel {
    /// Load pooling kernel module on the given device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, PoolingError> {
        let ptx = POOLING_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_embedding_gather_f32 = module
            .load_function(KERNEL_EMBEDDING_GATHER_F32)
            .map_err(|_| PoolingError::KernelMissing(KERNEL_EMBEDDING_GATHER_F32))?;
        let kernel_mean_pooling_f32 = module
            .load_function(KERNEL_MEAN_POOLING_F32)
            .map_err(|_| PoolingError::KernelMissing(KERNEL_MEAN_POOLING_F32))?;
        let kernel_l2_normalize_f32 = module
            .load_function(KERNEL_L2_NORMALIZE_F32)
            .map_err(|_| PoolingError::KernelMissing(KERNEL_L2_NORMALIZE_F32))?;
        let kernel_cls_extract_f32 = module
            .load_function(KERNEL_CLS_EXTRACT_F32)
            .map_err(|_| PoolingError::KernelMissing(KERNEL_CLS_EXTRACT_F32))?;

        Ok(Self {
            module,
            kernel_embedding_gather_f32,
            kernel_mean_pooling_f32,
            kernel_l2_normalize_f32,
            kernel_cls_extract_f32,
        })
    }

    pub fn embedding_gather_view(
        &self,
        stream: &Arc<CudaStream>,
        token_ids: &CudaView<'_, u32>,
        table: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        num_tokens: usize,
        hidden_dim: usize,
    ) -> Result<(), PoolingError> {
        if num_tokens == 0 || hidden_dim == 0 {
            return Ok(());
        }

        let total = num_tokens
            .checked_mul(hidden_dim)
            .ok_or_else(|| PoolingError::InvalidConfig("embedding gather size overflow".into()))?;
        let total_u32 = u32::try_from(total)
            .map_err(|_| PoolingError::InvalidConfig("embedding gather exceeds u32".into()))?;
        let tokens_i32 = i32::try_from(num_tokens)
            .map_err(|_| PoolingError::InvalidConfig("num_tokens exceeds i32".into()))?;
        let hidden_i32 = i32::try_from(hidden_dim)
            .map_err(|_| PoolingError::InvalidConfig("hidden_dim exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(total_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_embedding_gather_f32);
            builder.arg(token_ids);
            builder.arg(table);
            builder.arg(output);
            builder.arg(&tokens_i32);
            builder.arg(&hidden_i32);
            builder.launch(cfg)
        }
        .map_err(PoolingError::Driver)?;

        Ok(())
    }

    pub fn mean_pooling_view(
        &self,
        stream: &Arc<CudaStream>,
        hidden: &CudaView<'_, f32>,
        mask: Option<&CudaView<'_, f32>>,
        output: &mut CudaViewMut<'_, f32>,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> Result<(), PoolingError> {
        let total = batch
            .checked_mul(hidden_dim)
            .ok_or_else(|| PoolingError::InvalidConfig("mean_pooling size overflow".into()))?;
        if total == 0 {
            return Ok(());
        }

        let total_u32 = u32::try_from(total)
            .map_err(|_| PoolingError::InvalidConfig("mean_pooling exceeds u32".into()))?;
        let batch_i32 = i32::try_from(batch)
            .map_err(|_| PoolingError::InvalidConfig("batch exceeds i32".into()))?;
        let seq_i32 = i32::try_from(seq)
            .map_err(|_| PoolingError::InvalidConfig("seq exceeds i32".into()))?;
        let hidden_i32 = i32::try_from(hidden_dim)
            .map_err(|_| PoolingError::InvalidConfig("hidden_dim exceeds i32".into()))?;
        let use_mask_i32 = if mask.is_some() { 1 } else { 0 };

        let cfg = LaunchConfig::for_num_elems(total_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_mean_pooling_f32);
            builder.arg(hidden);
            match mask {
                Some(m) => builder.arg(m),
                None => builder.arg(&0u64),
            };
            builder.arg(output);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&hidden_i32);
            builder.arg(&use_mask_i32);
            builder.launch(cfg)
        }
        .map_err(PoolingError::Driver)?;

        Ok(())
    }

    pub fn l2_normalize_view(
        &self,
        stream: &Arc<CudaStream>,
        data: &mut CudaViewMut<'_, f32>,
        batch: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<(), PoolingError> {
        if batch == 0 || hidden_dim == 0 {
            return Ok(());
        }

        let batch_u32 = u32::try_from(batch)
            .map_err(|_| PoolingError::InvalidConfig("batch exceeds u32".into()))?;
        let batch_i32 = i32::try_from(batch)
            .map_err(|_| PoolingError::InvalidConfig("batch exceeds i32".into()))?;
        let hidden_i32 = i32::try_from(hidden_dim)
            .map_err(|_| PoolingError::InvalidConfig("hidden_dim exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(batch_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_l2_normalize_f32);
            builder.arg(data);
            builder.arg(&batch_i32);
            builder.arg(&hidden_i32);
            builder.arg(&eps);
            builder.launch(cfg)
        }
        .map_err(PoolingError::Driver)?;

        Ok(())
    }

    pub fn cls_extract_view(
        &self,
        stream: &Arc<CudaStream>,
        hidden: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
        cls_pos: usize,
    ) -> Result<(), PoolingError> {
        let total = batch
            .checked_mul(hidden_dim)
            .ok_or_else(|| PoolingError::InvalidConfig("cls_extract size overflow".into()))?;
        if total == 0 {
            return Ok(());
        }

        let total_u32 = u32::try_from(total)
            .map_err(|_| PoolingError::InvalidConfig("cls_extract exceeds u32".into()))?;
        let batch_i32 = i32::try_from(batch)
            .map_err(|_| PoolingError::InvalidConfig("batch exceeds i32".into()))?;
        let seq_i32 = i32::try_from(seq)
            .map_err(|_| PoolingError::InvalidConfig("seq exceeds i32".into()))?;
        let hidden_i32 = i32::try_from(hidden_dim)
            .map_err(|_| PoolingError::InvalidConfig("hidden_dim exceeds i32".into()))?;
        let cls_i32 = i32::try_from(cls_pos)
            .map_err(|_| PoolingError::InvalidConfig("cls_pos exceeds i32".into()))?;

        let cfg = LaunchConfig::for_num_elems(total_u32);

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_cls_extract_f32);
            builder.arg(hidden);
            builder.arg(output);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&hidden_i32);
            builder.arg(&cls_i32);
            builder.launch(cfg)
        }
        .map_err(PoolingError::Driver)?;

        Ok(())
    }
}
