use cudarc::driver::{sys, CudaSlice, CudaStream, DeviceRepr, LaunchConfig};

use crate::backend_trait::{BackendError, BackendResult};
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const EMBEDDING_KERNEL_NAMES: &[&str] = &["embedding_lookup", "embedding"];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingConfig {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub seq_len: u32,
    pub stride: u32,
}

unsafe impl DeviceRepr for EmbeddingConfig {}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingKernel {
    func: sys::CUfunction,
}

unsafe impl Send for EmbeddingKernel {}
unsafe impl Sync for EmbeddingKernel {}

impl EmbeddingKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        let mut last_err = None;
        for name in EMBEDDING_KERNEL_NAMES {
            match load_function(module, name) {
                Ok(func) => return Ok(Self { func }),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.unwrap_or_else(|| BackendError::Cuda("missing embedding kernel".into())))
    }

    pub fn launch<T: DeviceRepr>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        tokens: &CudaSlice<u32>,
        embedding: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        params: &EmbeddingConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 4);
        launch
            .arg_device(tokens)
            .arg_device(embedding)
            .arg_device_mut(output)
            .arg_scalar(params);
        unsafe { launch.launch() }
    }
}
