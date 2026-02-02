use cudarc::driver::{
    sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig,
};

use crate::backend_trait::BackendResult;
use crate::cuda_kernels::{load_function, KernelLaunch};

pub const FLASH_ATTN_KERNEL_NAME: &str = "flash_attention";
pub const FLASH_ATTN_PAGED_KERNEL_NAME: &str = "flash_attention_paged";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnConfig {
    pub batch: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub q_seq_len: u32,
    pub kv_seq_len: u32,
    pub q_stride: u32,
    pub kv_stride: u32,
    pub o_stride: u32,
    pub causal: u32,
    pub scale: f32,
    pub q_pos_offset: u32,
}

unsafe impl DeviceRepr for FlashAttnConfig {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnPagedConfig {
    pub batch: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub q_seq_len: u32,
    pub kv_seq_len: u32,
    pub q_stride: u32,
    pub kv_stride: u32,
    pub o_stride: u32,
    pub causal: u32,
    pub scale: f32,
    pub q_pos_offset: u32,
    pub page_size: u32,
    pub pages_per_layer: u32,
}

unsafe impl DeviceRepr for FlashAttnPagedConfig {}

#[derive(Debug, Clone, Copy)]
pub struct FlashAttnKernel {
    func: sys::CUfunction,
}

unsafe impl Send for FlashAttnKernel {}
unsafe impl Sync for FlashAttnKernel {}

impl FlashAttnKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            func: load_function(module, FLASH_ATTN_KERNEL_NAME)?,
        })
    }

    pub fn launch<
        T: DeviceRepr,
        Q: DevicePtr<T>,
        K: DevicePtr<T>,
        V: DevicePtr<T>,
        O: DevicePtrMut<T>,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q: &Q,
        k: &K,
        v: &V,
        output: &mut O,
        alibi_slopes: Option<&CudaSlice<f32>>,
        params: &FlashAttnConfig,
    ) -> BackendResult<()> {
        self.launch_with_lse::<T, Q, K, V, O, CudaSlice<f32>>(
            stream,
            config,
            q,
            k,
            v,
            output,
            None,
            alibi_slopes,
            params,
        )
    }

    pub fn launch_with_lse<
        T: DeviceRepr,
        Q: DevicePtr<T>,
        K: DevicePtr<T>,
        V: DevicePtr<T>,
        O: DevicePtrMut<T>,
        L: DevicePtrMut<f32>,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q: &Q,
        k: &K,
        v: &V,
        output: &mut O,
        lse: Option<&mut L>,
        alibi_slopes: Option<&CudaSlice<f32>>,
        params: &FlashAttnConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 7);
        launch
            .arg_device(q)
            .arg_device(k)
            .arg_device(v)
            .arg_device_mut(output);

        match lse {
            Some(buf) => {
                launch.arg_device_mut(buf);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        match alibi_slopes {
            Some(slopes) => {
                launch.arg_device(slopes);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        launch.arg_scalar(params);
        unsafe { launch.launch() }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlashAttnPagedKernel {
    func: sys::CUfunction,
}

unsafe impl Send for FlashAttnPagedKernel {}
unsafe impl Sync for FlashAttnPagedKernel {}

impl FlashAttnPagedKernel {
    pub(crate) fn load(module: sys::CUmodule) -> BackendResult<Self> {
        Ok(Self {
            func: load_function(module, FLASH_ATTN_PAGED_KERNEL_NAME)?,
        })
    }

    pub fn launch<
        T: DeviceRepr,
        Q: DevicePtr<T>,
        P: DevicePtr<u64>,
        O: DevicePtrMut<T>,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q: &Q,
        page_table: &P,
        output: &mut O,
        alibi_slopes: Option<&CudaSlice<f32>>,
        params: &FlashAttnPagedConfig,
    ) -> BackendResult<()> {
        self.launch_with_lse::<T, Q, P, O, CudaSlice<f32>>(
            stream,
            config,
            q,
            page_table,
            output,
            None,
            alibi_slopes,
            params,
        )
    }

    pub fn launch_with_lse<
        T: DeviceRepr,
        Q: DevicePtr<T>,
        P: DevicePtr<u64>,
        O: DevicePtrMut<T>,
        L: DevicePtrMut<f32>,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        q: &Q,
        page_table: &P,
        output: &mut O,
        lse: Option<&mut L>,
        alibi_slopes: Option<&CudaSlice<f32>>,
        params: &FlashAttnPagedConfig,
    ) -> BackendResult<()> {
        let mut launch = KernelLaunch::new(self.func, stream, config, 6);
        launch
            .arg_device(q)
            .arg_device(page_table)
            .arg_device_mut(output);

        match lse {
            Some(buf) => {
                launch.arg_device_mut(buf);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        match alibi_slopes {
            Some(slopes) => {
                launch.arg_device(slopes);
            }
            None => {
                launch.arg_device_ptr(0);
            }
        }

        launch.arg_scalar(params);
        unsafe { launch.launch() }
    }
}
