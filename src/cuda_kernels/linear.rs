use cudarc::driver::{sys, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig};

use crate::backend_trait::{BackendError, BackendResult};
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const LINEAR_KERNEL_NAMES: &[&str] = &["linear", "gemm", "matmul"];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub input_stride: u32,
    pub weight_stride: u32,
    pub output_stride: u32,
    pub use_bias: u32,
}

unsafe impl DeviceRepr for LinearConfig {}

#[derive(Debug, Clone, Copy)]
pub struct LinearKernel {
    func: sys::CUfunction,
}

unsafe impl Send for LinearKernel {}
unsafe impl Sync for LinearKernel {}

impl LinearKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        let mut last_err = None;
        for name in LINEAR_KERNEL_NAMES {
            match load_function(module, name) {
                Ok(func) => return Ok(Self { func }),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.unwrap_or_else(|| BackendError::Cuda("missing linear kernel".into())))
    }

    pub fn launch<
        T: DeviceRepr,
        I: DevicePtr<T>,
        W: DevicePtr<T>,
        B: DevicePtr<T>,
        O: DevicePtrMut<T>,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &I,
        weight: &W,
        bias: Option<&B>,
        output: &mut O,
        params: &LinearConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 5);
        launch
            .arg_device(input)
            .arg_device(weight)
            .arg_device_mut(output);

        match bias {
            Some(bias) => {
                launch.arg_device(bias);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        launch.arg_scalar(params);
        unsafe { launch.launch() }
    }
}
