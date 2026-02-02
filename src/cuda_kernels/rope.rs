use cudarc::driver::{sys, CudaSlice, CudaStream, DeviceRepr, LaunchConfig};

use crate::backend_trait::BackendResult;
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const ROPE_KERNEL_NAME: &str = "rope";
pub const FUSED_QKV_ROPE_KERNEL_NAME: &str = "fused_qkv_rope";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RopeConfig {
    pub seq_len: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub base: f32,
    pub scale: f32,
    pub interleaved: u32,
    pub position_stride: u32,
    pub precompute_max_seq_len: u32,
}

unsafe impl DeviceRepr for RopeConfig {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FusedQkvRopeConfig {
    pub batch: u32,
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub input_stride: u32,
    pub qkv_stride: u32,
    pub base: f32,
    pub scale: f32,
    pub interleaved: u32,
    pub precompute_max_seq_len: u32,
}

unsafe impl DeviceRepr for FusedQkvRopeConfig {}

#[derive(Debug, Clone, Copy)]
pub struct RopeKernel {
    func: sys::CUfunction,
}

unsafe impl Send for RopeKernel {}
unsafe impl Sync for RopeKernel {}

#[derive(Debug, Clone, Copy)]
pub struct FusedQkvRopeKernel {
    func: sys::CUfunction,
}

unsafe impl Send for FusedQkvRopeKernel {}
unsafe impl Sync for FusedQkvRopeKernel {}

impl RopeKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            func: load_function(module, ROPE_KERNEL_NAME)?,
        })
    }

    pub fn launch<T: DeviceRepr>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q: &mut CudaSlice<T>,
        k: &mut CudaSlice<T>,
        positions: Option<&CudaSlice<i32>>,
        cos_table: Option<&CudaSlice<f32>>,
        sin_table: Option<&CudaSlice<f32>>,
        params: &RopeConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 6);
        launch.arg_device_mut(q).arg_device_mut(k);

        match positions {
            Some(pos) => {
                launch.arg_device(pos);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        match cos_table {
            Some(cos) => {
                launch.arg_device(cos);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        match sin_table {
            Some(sin) => {
                launch.arg_device(sin);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        launch.arg_scalar(params);
        unsafe { launch.launch() }
    }
}

impl FusedQkvRopeKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            func: load_function(module, FUSED_QKV_ROPE_KERNEL_NAME)?,
        })
    }

    pub fn launch<T: DeviceRepr>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<T>,
        qkv_weight: &CudaSlice<T>,
        qkv_bias: Option<&CudaSlice<T>>,
        qkv_out: &mut CudaSlice<T>,
        positions: Option<&CudaSlice<i32>>,
        cos_table: Option<&CudaSlice<f32>>,
        sin_table: Option<&CudaSlice<f32>>,
        params: &FusedQkvRopeConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 8);
        launch
            .arg_device(input)
            .arg_device(qkv_weight)
            .arg_device_mut(qkv_out);

        match qkv_bias {
            Some(bias) => {
                launch.arg_device(bias);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        match positions {
            Some(pos) => {
                launch.arg_device(pos);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        match cos_table {
            Some(cos) => {
                launch.arg_device(cos);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        match sin_table {
            Some(sin) => {
                launch.arg_device(sin);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        launch.arg_scalar(params);
        unsafe { launch.launch() }
    }
}
