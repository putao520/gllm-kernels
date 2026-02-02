use cudarc::driver::{sys, CudaSlice, CudaStream, DeviceRepr, LaunchConfig};

use crate::backend_trait::BackendResult;
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const RMS_NORM_KERNEL_NAME: &str = "rms_norm";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RmsNormConfig {
    pub hidden_size: u32,
    pub stride: u32,
    pub eps: f32,
    pub seq_len: u32,
}

unsafe impl DeviceRepr for RmsNormConfig {}

#[derive(Debug, Clone, Copy)]
pub struct RmsNormKernel {
    func: sys::CUfunction,
}

unsafe impl Send for RmsNormKernel {}
unsafe impl Sync for RmsNormKernel {}

impl RmsNormKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            func: load_function(module, RMS_NORM_KERNEL_NAME)?,
        })
    }

    pub fn launch<T: DeviceRepr>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<T>,
        weight: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        params: &RmsNormConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 4);
        launch
            .arg_device(input)
            .arg_device(weight)
            .arg_device_mut(output)
            .arg_scalar(params);
        unsafe { launch.launch() }
    }
}
