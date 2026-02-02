use cudarc::driver::{sys, CudaSlice, CudaStream, DeviceRepr, LaunchConfig};

use crate::backend_trait::{BackendError, BackendResult};
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const ADD_KERNEL_NAMES: &[&str] = &["add", "add_residual", "residual_add"];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AddConfig {
    pub numel: u32,
}

unsafe impl DeviceRepr for AddConfig {}

#[derive(Debug, Clone, Copy)]
pub struct AddKernel {
    func: sys::CUfunction,
}

unsafe impl Send for AddKernel {}
unsafe impl Sync for AddKernel {}

impl AddKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        let mut last_err = None;
        for name in ADD_KERNEL_NAMES {
            match load_function(module, name) {
                Ok(func) => return Ok(Self { func }),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.unwrap_or_else(|| BackendError::Cuda("missing add kernel".into())))
    }

    pub fn launch<T: DeviceRepr>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        lhs: &CudaSlice<T>,
        rhs: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        params: &AddConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 4);
        launch
            .arg_device(lhs)
            .arg_device(rhs)
            .arg_device_mut(output)
            .arg_scalar(params);
        unsafe { launch.launch() }
    }
}
