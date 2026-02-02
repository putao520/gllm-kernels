use cudarc::driver::{sys, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig};

use crate::backend_trait::BackendResult;
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const QUANTIZED_MM_KERNEL_NAME_INT1: &str = "_Z12quantized_mmILi1EEvPKfPKhPf15QuantizedConfig";
pub const QUANTIZED_MM_KERNEL_NAME_INT2: &str = "_Z12quantized_mmILi2EEvPKfPKhPf15QuantizedConfig";
pub const QUANTIZED_MM_KERNEL_NAME_INT4: &str = "_Z12quantized_mmILi4EEvPKfPKhPf15QuantizedConfig";
pub const QUANTIZED_MM_KERNEL_NAME_INT8: &str = "_Z12quantized_mmILi8EEvPKfPKhPf15QuantizedConfig";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QuantizedConfig {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub input_stride: u32,
    pub weight_stride: u32,
    pub output_stride: u32,
    pub scale: f32,
}

unsafe impl DeviceRepr for QuantizedConfig {}

pub type Int8MmConfig = QuantizedConfig;
pub type Int4MmConfig = QuantizedConfig;
pub type Int2MmConfig = QuantizedConfig;
pub type Int1MmConfig = QuantizedConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedBits {
    Int1,
    Int2,
    Int4,
    Int8,
}

impl QuantizedBits {
    pub const fn kernel_name(self) -> &'static str {
        match self {
            QuantizedBits::Int1 => QUANTIZED_MM_KERNEL_NAME_INT1,
            QuantizedBits::Int2 => QUANTIZED_MM_KERNEL_NAME_INT2,
            QuantizedBits::Int4 => QUANTIZED_MM_KERNEL_NAME_INT4,
            QuantizedBits::Int8 => QUANTIZED_MM_KERNEL_NAME_INT8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMmKernel {
    func: sys::CUfunction,
}

unsafe impl Send for QuantizedMmKernel {}
unsafe impl Sync for QuantizedMmKernel {}

impl QuantizedMmKernel {
    pub(crate) fn load(module: sys::CUmodule, name: &str) -> BackendResult<Self> {
        let func = load_function(module, name)?;
        Ok(Self { func })
    }

    pub fn launch<I, W, O, Weight>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &I,
        weight: &W,
        output: &mut O,
        params: &QuantizedConfig,
    ) -> BackendResult<()>
    where
        I: DevicePtr<f32>,
        W: DevicePtr<Weight>,
        O: DevicePtrMut<f32>,
    {
        let mut launch = KernelLaunch::new(self.func, stream, config, 4);
        launch
            .arg_device(input)
            .arg_device(weight)
            .arg_device_mut(output)
            .arg_scalar(params);
        unsafe { launch.launch() }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMmKernels {
    pub int1: Option<QuantizedMmKernel>,
    pub int2: Option<QuantizedMmKernel>,
    pub int4: Option<QuantizedMmKernel>,
    pub int8: Option<QuantizedMmKernel>,
}

impl QuantizedMmKernels {
    pub(crate) fn load(module: sys::CUmodule) -> Self {
        Self {
            int1: QuantizedMmKernel::load(module, QUANTIZED_MM_KERNEL_NAME_INT1).ok(),
            int2: QuantizedMmKernel::load(module, QUANTIZED_MM_KERNEL_NAME_INT2).ok(),
            int4: QuantizedMmKernel::load(module, QUANTIZED_MM_KERNEL_NAME_INT4).ok(),
            int8: QuantizedMmKernel::load(module, QUANTIZED_MM_KERNEL_NAME_INT8).ok(),
        }
    }
}
