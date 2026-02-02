use cudarc::driver::{sys, CudaSlice, CudaStream, DeviceRepr, LaunchConfig};

use crate::backend_trait::BackendResult;
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const SILU_KERNEL_NAME: &str = "silu";
pub const SWIGLU_KERNEL_NAME: &str = "fused_gate_up_silu";
pub const FUSED_GATE_UP_SILU_KERNEL_NAME: &str = SWIGLU_KERNEL_NAME;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SiluConfig {
    pub numel: u32,
}

unsafe impl DeviceRepr for SiluConfig {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwiGluConfig {
    pub numel: u32,
}

unsafe impl DeviceRepr for SwiGluConfig {}

#[derive(Debug, Clone, Copy)]
pub struct SiluKernel {
    func: sys::CUfunction,
}

unsafe impl Send for SiluKernel {}
unsafe impl Sync for SiluKernel {}

#[derive(Debug, Clone, Copy)]
pub struct SwiGluKernel {
    func: sys::CUfunction,
}

unsafe impl Send for SwiGluKernel {}
unsafe impl Sync for SwiGluKernel {}

impl SiluKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            func: load_function(module, SILU_KERNEL_NAME)?,
        })
    }

    pub fn launch<T: DeviceRepr>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        params: &SiluConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 3);
        launch
            .arg_device(input)
            .arg_device_mut(output)
            .arg_scalar(params);
        unsafe { launch.launch() }
    }
}

impl SwiGluKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            func: load_function(module, SWIGLU_KERNEL_NAME)?,
        })
    }

    pub fn launch<T: DeviceRepr>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        gate: &CudaSlice<T>,
        up: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        params: &SwiGluConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 4);
        launch
            .arg_device(gate)
            .arg_device(up)
            .arg_device_mut(output)
            .arg_scalar(params);
        unsafe { launch.launch() }
    }
}
