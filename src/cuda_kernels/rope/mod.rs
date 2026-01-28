use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, CudaView, CudaViewMut, DriverError,
    LaunchConfig, PushKernelArg,
};
use half::f16;

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

const KERNEL_ROPE_Q_F32: &str = "rope_apply_f32";
const KERNEL_ROPE_K_F32: &str = "rope_apply_k_f32";
const KERNEL_ROPE_INPLACE_F32: &str = "rope_apply_inplace_f32";
const KERNEL_ROPE_Q_F16: &str = "rope_apply_f16";
const KERNEL_ROPE_K_F16: &str = "rope_apply_k_f16";
const KERNEL_ROPE_INPLACE_F16: &str = "rope_apply_inplace_f16";
const DEFAULT_BLOCK_SIZE: u32 = 256;

static ROPE_PTX: PtxCollection = PtxCollection {
    kernel_name: "rope",
    ptx_versions: &[
        (61, include_str!("../kernels/rope_sm61.ptx")),
        (80, include_str!("../kernels/rope.ptx")),
    ],
};

#[derive(Debug)]
pub enum RoPEKernelError {
    Driver(DriverError),
    InvalidConfig(String),
    KernelMissing(&'static str),
    PtxLoad(PtxLoadError),
}

impl fmt::Display for RoPEKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for RoPEKernelError {}

impl From<DriverError> for RoPEKernelError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for RoPEKernelError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

pub struct CudaRoPEKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_rope_q_f32: CudaFunction,
    kernel_rope_k_f32: CudaFunction,
    kernel_rope_inplace_f32: CudaFunction,
    kernel_rope_q_f16: CudaFunction,
    kernel_rope_k_f16: CudaFunction,
    kernel_rope_inplace_f16: CudaFunction,
}

impl CudaRoPEKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, RoPEKernelError> {
        let ptx = ROPE_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_rope_q_f32 = module
            .load_function(KERNEL_ROPE_Q_F32)
            .map_err(|_| RoPEKernelError::KernelMissing(KERNEL_ROPE_Q_F32))?;
        let kernel_rope_k_f32 = module
            .load_function(KERNEL_ROPE_K_F32)
            .map_err(|_| RoPEKernelError::KernelMissing(KERNEL_ROPE_K_F32))?;
        let kernel_rope_inplace_f32 = module
            .load_function(KERNEL_ROPE_INPLACE_F32)
            .map_err(|_| RoPEKernelError::KernelMissing(KERNEL_ROPE_INPLACE_F32))?;
        let kernel_rope_q_f16 = module
            .load_function(KERNEL_ROPE_Q_F16)
            .map_err(|_| RoPEKernelError::KernelMissing(KERNEL_ROPE_Q_F16))?;
        let kernel_rope_k_f16 = module
            .load_function(KERNEL_ROPE_K_F16)
            .map_err(|_| RoPEKernelError::KernelMissing(KERNEL_ROPE_K_F16))?;
        let kernel_rope_inplace_f16 = module
            .load_function(KERNEL_ROPE_INPLACE_F16)
            .map_err(|_| RoPEKernelError::KernelMissing(KERNEL_ROPE_INPLACE_F16))?;

        Ok(Self {
            module,
            kernel_rope_q_f32,
            kernel_rope_k_f32,
            kernel_rope_inplace_f32,
            kernel_rope_q_f16,
            kernel_rope_k_f16,
            kernel_rope_inplace_f16,
        })
    }

    /// Apply RoPE to Q tensor (f32)
    pub fn apply_q_f32(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaView<'_, f32>,
        k: &CudaView<'_, f32>,
        cos_cache: &CudaView<'_, f32>,
        sin_cache: &CudaView<'_, f32>,
        q_out: &mut CudaViewMut<'_, f32>,
        k_out: &mut CudaViewMut<'_, f32>,
        batch_size: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), RoPEKernelError> {
        if head_dim % 2 != 0 {
            return Err(RoPEKernelError::InvalidConfig(
                "head_dim must be even".into(),
            ));
        }
        let half_dim = head_dim / 2;

        // Launch kernel for Q
        let total_q_items = batch_size * seq_len * num_q_heads * half_dim;
        let cfg = LaunchConfig {
            grid_dim: ((total_q_items as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let num_q_heads_i32 = num_q_heads as i32;
        let num_kv_heads_i32 = num_kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let half_dim_i32 = half_dim as i32;
        let pos_offset_i32 = position_offset as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_rope_q_f32);
            builder.arg(q);
            builder.arg(k);
            builder.arg(cos_cache);
            builder.arg(sin_cache);
            builder.arg(q_out);
            builder.arg(k_out);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&num_q_heads_i32);
            builder.arg(&num_kv_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&half_dim_i32);
            builder.arg(&pos_offset_i32);
            builder.launch(cfg)?;
        }

        // Launch kernel for K (separate because different number of heads)
        let total_k_items = batch_size * seq_len * num_kv_heads * half_dim;
        let cfg_k = LaunchConfig {
            grid_dim: ((total_k_items as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_rope_k_f32);
            builder.arg(k);
            builder.arg(cos_cache);
            builder.arg(sin_cache);
            builder.arg(k_out);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&num_kv_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&half_dim_i32);
            builder.arg(&pos_offset_i32);
            builder.launch(cfg_k)?;
        }

        Ok(())
    }

    /// Apply RoPE in-place (f32)
    pub fn apply_inplace_f32(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaViewMut<'_, f32>,
        cos_cache: &CudaView<'_, f32>,
        sin_cache: &CudaView<'_, f32>,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), RoPEKernelError> {
        if head_dim % 2 != 0 {
            return Err(RoPEKernelError::InvalidConfig(
                "head_dim must be even".into(),
            ));
        }
        let half_dim = head_dim / 2;

        let total_items = batch_size * seq_len * num_heads * half_dim;
        let cfg = LaunchConfig {
            grid_dim: ((total_items as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let num_heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let half_dim_i32 = half_dim as i32;
        let pos_offset_i32 = position_offset as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_rope_inplace_f32);
            builder.arg(x);
            builder.arg(cos_cache);
            builder.arg(sin_cache);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&half_dim_i32);
            builder.arg(&pos_offset_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Apply RoPE to Q tensor (f16)
    pub fn apply_q_f16(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaView<'_, f16>,
        k: &CudaView<'_, f16>,
        cos_cache: &CudaView<'_, f32>,
        sin_cache: &CudaView<'_, f32>,
        q_out: &mut CudaViewMut<'_, f16>,
        k_out: &mut CudaViewMut<'_, f16>,
        batch_size: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), RoPEKernelError> {
        if head_dim % 2 != 0 {
            return Err(RoPEKernelError::InvalidConfig(
                "head_dim must be even".into(),
            ));
        }
        let half_dim = head_dim / 2;

        // Launch kernel for Q
        let total_q_items = batch_size * seq_len * num_q_heads * half_dim;
        let cfg = LaunchConfig {
            grid_dim: ((total_q_items as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let num_q_heads_i32 = num_q_heads as i32;
        let num_kv_heads_i32 = num_kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let half_dim_i32 = half_dim as i32;
        let pos_offset_i32 = position_offset as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_rope_q_f16);
            builder.arg(q);
            builder.arg(k);
            builder.arg(cos_cache);
            builder.arg(sin_cache);
            builder.arg(q_out);
            builder.arg(k_out);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&num_q_heads_i32);
            builder.arg(&num_kv_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&half_dim_i32);
            builder.arg(&pos_offset_i32);
            builder.launch(cfg)?;
        }

        // Launch kernel for K
        let total_k_items = batch_size * seq_len * num_kv_heads * half_dim;
        let cfg_k = LaunchConfig {
            grid_dim: ((total_k_items as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_rope_k_f16);
            builder.arg(k);
            builder.arg(cos_cache);
            builder.arg(sin_cache);
            builder.arg(k_out);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&num_kv_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&half_dim_i32);
            builder.arg(&pos_offset_i32);
            builder.launch(cfg_k)?;
        }

        Ok(())
    }

    /// Apply RoPE in-place (f16)
    pub fn apply_inplace_f16(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaViewMut<'_, f16>,
        cos_cache: &CudaView<'_, f32>,
        sin_cache: &CudaView<'_, f32>,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), RoPEKernelError> {
        if head_dim % 2 != 0 {
            return Err(RoPEKernelError::InvalidConfig(
                "head_dim must be even".into(),
            ));
        }
        let half_dim = head_dim / 2;

        let total_items = batch_size * seq_len * num_heads * half_dim;
        let cfg = LaunchConfig {
            grid_dim: ((total_items as u32 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let seq_i32 = seq_len as i32;
        let num_heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let half_dim_i32 = half_dim as i32;
        let pos_offset_i32 = position_offset as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_rope_inplace_f16);
            builder.arg(x);
            builder.arg(cos_cache);
            builder.arg(sin_cache);
            builder.arg(&batch_i32);
            builder.arg(&seq_i32);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&half_dim_i32);
            builder.arg(&pos_offset_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }
}
