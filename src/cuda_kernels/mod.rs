use std::ffi::CString;

use cudarc::driver::{result, sys};

use crate::backend_trait::{BackendError, BackendResult};

pub mod activation;
pub mod add;
pub mod embedding;
pub mod flash_attn;
mod launch;
pub mod linear;
pub mod quantization;
pub mod rms_norm;
pub mod rope;
pub mod sampling;

pub use activation::{
    SiluConfig, SiluKernel, SwiGluConfig, SwiGluKernel, FUSED_GATE_UP_SILU_KERNEL_NAME,
};
pub use add::{AddConfig, AddKernel, ADD_KERNEL_NAMES};
pub use embedding::{EmbeddingConfig, EmbeddingKernel, EMBEDDING_KERNEL_NAMES};
pub use flash_attn::{
    FlashAttnConfig, FlashAttnKernel, FlashAttnPagedConfig, FlashAttnPagedKernel,
    FLASH_ATTN_KERNEL_NAME, FLASH_ATTN_PAGED_KERNEL_NAME,
};
pub use linear::{LinearConfig, LinearKernel, LINEAR_KERNEL_NAMES};
pub use quantization::{
    Int1MmConfig, Int2MmConfig, Int4MmConfig, Int8MmConfig, QuantizedBits, QuantizedConfig,
    QuantizedMmKernel, QuantizedMmKernels, QUANTIZED_MM_KERNEL_NAME_INT1,
    QUANTIZED_MM_KERNEL_NAME_INT2, QUANTIZED_MM_KERNEL_NAME_INT4, QUANTIZED_MM_KERNEL_NAME_INT8,
};
pub use rms_norm::{RmsNormConfig, RmsNormKernel, RMS_NORM_KERNEL_NAME};
pub use rope::{
    FusedQkvRopeConfig, FusedQkvRopeKernel, RopeConfig, RopeKernel, FUSED_QKV_ROPE_KERNEL_NAME,
    ROPE_KERNEL_NAME,
};
pub use sampling::{SamplingKernel, SamplingKernelConfig, SAMPLING_KERNEL_NAMES};

pub use cudarc::driver::LaunchConfig;
pub(crate) use launch::KernelLaunch;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmVersion {
    Sm80,
    Sm86,
    Sm89,
    Sm90,
}

impl SmVersion {
    pub fn as_str(self) -> &'static str {
        match self {
            SmVersion::Sm80 => "sm_80",
            SmVersion::Sm86 => "sm_86",
            SmVersion::Sm89 => "sm_89",
            SmVersion::Sm90 => "sm_90",
        }
    }

    pub fn from_compute_capability(major: i32, minor: i32) -> Option<Self> {
        match (major, minor) {
            (8, 0) => Some(SmVersion::Sm80),
            (8, 6) => Some(SmVersion::Sm86),
            (8, 9) => Some(SmVersion::Sm89),
            (9, 0) => Some(SmVersion::Sm90),
            _ => None,
        }
    }
}

const KERNELS_SM80: &[u8] = include_bytes!("kernels/kernels_sm80.cubin");
const KERNELS_SM86: &[u8] = include_bytes!("kernels/kernels_sm86.cubin");
const KERNELS_SM89: &[u8] = include_bytes!("kernels/kernels_sm89.cubin");
const KERNELS_SM90: &[u8] = include_bytes!("kernels/kernels_sm90.cubin");

pub fn cubin_for(sm: SmVersion) -> &'static [u8] {
    match sm {
        SmVersion::Sm80 => KERNELS_SM80,
        SmVersion::Sm86 => KERNELS_SM86,
        SmVersion::Sm89 => KERNELS_SM89,
        SmVersion::Sm90 => KERNELS_SM90,
    }
}

#[derive(Debug)]
pub struct CudaKernels {
    pub flash_attn: FlashAttnKernel,
    pub flash_attn_paged: FlashAttnPagedKernel,
    pub rope: RopeKernel,
    pub fused_qkv_rope: FusedQkvRopeKernel,
    pub rms_norm: RmsNormKernel,
    pub silu: SiluKernel,
    pub swiglu: SwiGluKernel,
    pub add: AddKernel,
    pub embedding: EmbeddingKernel,
    pub linear: LinearKernel,
    pub quantized_mm: QuantizedMmKernels,
    pub sampling: SamplingKernel,
}

impl CudaKernels {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            flash_attn: FlashAttnKernel::load(module)?,
            flash_attn_paged: FlashAttnPagedKernel::load(module)?,
            rope: RopeKernel::load(module)?,
            fused_qkv_rope: FusedQkvRopeKernel::load(module)?,
            rms_norm: RmsNormKernel::load(module)?,
            silu: SiluKernel::load(module)?,
            swiglu: SwiGluKernel::load(module)?,
            add: AddKernel::load(module)?,
            embedding: EmbeddingKernel::load(module)?,
            linear: LinearKernel::load(module)?,
            quantized_mm: QuantizedMmKernels::load(module),
            sampling: SamplingKernel::load(module)?,
        })
    }
}

pub(crate) fn load_function(module: sys::CUmodule, name: &str) -> BackendResult<sys::CUfunction> {
    let name = CString::new(name)
        .map_err(|_| BackendError::Cuda(format!("invalid kernel name: {name}")))?;
    Ok(unsafe { result::module::get_function(module, name)? })
}
