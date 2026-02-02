use cudarc::driver::{sys, CudaSlice, CudaStream, DeviceRepr, LaunchConfig};

use crate::backend_trait::{BackendError, BackendResult};
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const SAMPLING_KERNEL_NAMES: &[&str] = &["sample_from_logits", "sampling", "argmax_topk"];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SamplingKernelConfig {
    pub vocab_size: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub stride: u32,
    pub batch: u32,
}

unsafe impl DeviceRepr for SamplingKernelConfig {}

#[derive(Debug, Clone, Copy)]
pub struct SamplingKernel {
    func: sys::CUfunction,
}

unsafe impl Send for SamplingKernel {}
unsafe impl Sync for SamplingKernel {}

impl SamplingKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        let mut last_err = None;
        for name in SAMPLING_KERNEL_NAMES {
            match load_function(module, name) {
                Ok(func) => return Ok(Self { func }),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.unwrap_or_else(|| BackendError::Cuda("missing sampling kernel".into())))
    }

    pub fn launch(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        logits: &CudaSlice<f32>,
        output: &mut CudaSlice<u32>,
        params: &SamplingKernelConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 3);
        launch
            .arg_device(logits)
            .arg_device_mut(output)
            .arg_scalar(params);
        unsafe { launch.launch() }
    }
}
