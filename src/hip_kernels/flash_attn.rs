//! HIP/ROCm FlashAttention kernel wrapper.
//!
//! This module provides FlashAttention-style kernels for AMD GPUs via HIP runtime.
//! The implementation uses raw FFI bindings to the HIP runtime library.

use std::ffi::CString;
use std::fmt;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;

use crate::types::AttentionConfig;

// HIP type definitions (FFI)
type HipDevice = c_int;
type HipStream = *mut c_void;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HipDeviceptr = *mut c_void;
type HipError = c_int;

const HIP_SUCCESS: HipError = 0;

// HIP FFI declarations
#[link(name = "amdhip64")]
extern "C" {
    fn hipInit(flags: c_uint) -> HipError;
    fn hipSetDevice(device: HipDevice) -> HipError;
    #[allow(dead_code)]
    fn hipGetDevice(device: *mut HipDevice) -> HipError;
    fn hipGetDeviceCount(count: *mut c_int) -> HipError;

    fn hipMalloc(ptr: *mut HipDeviceptr, size: usize) -> HipError;
    fn hipFree(ptr: HipDeviceptr) -> HipError;
    fn hipMemcpy(
        dst: HipDeviceptr,
        src: *const c_void,
        size: usize,
        kind: c_int,
    ) -> HipError;
    fn hipMemset(ptr: HipDeviceptr, value: c_int, size: usize) -> HipError;

    fn hipModuleLoad(module: *mut HipModule, fname: *const c_char) -> HipError;
    fn hipModuleLoadData(module: *mut HipModule, image: *const c_void) -> HipError;
    fn hipModuleGetFunction(
        func: *mut HipFunction,
        module: HipModule,
        name: *const c_char,
    ) -> HipError;
    fn hipModuleUnload(module: HipModule) -> HipError;

    fn hipModuleLaunchKernel(
        f: HipFunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: HipStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;

    fn hipStreamCreate(stream: *mut HipStream) -> HipError;
    fn hipStreamDestroy(stream: HipStream) -> HipError;
    fn hipStreamSynchronize(stream: HipStream) -> HipError;

    fn hipDeviceSynchronize() -> HipError;
    fn hipGetErrorString(error: HipError) -> *const c_char;
}

// Memory copy kinds
const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

const KERNEL_F32: &str = "tiled_attention_forward_f32\0";
const KERNEL_F16: &str = "tiled_attention_forward_f16\0";
const DEFAULT_BLOCK: u32 = 128;
const MAX_HEAD_DIM: usize = 256;

// Embedded HSACO (placeholder - replace with actual compiled binary)
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/flash_attention.hsaco");

/// Errors from HIP FlashAttention kernels.
#[derive(Debug)]
pub enum FlashAttentionError {
    /// HIP runtime error.
    Hip(i32, String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Module load failure.
    ModuleLoadFailed(String),
}

impl fmt::Display for FlashAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(code, msg) => write!(f, "HIP error {}: {}", code, msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {}", name),
            Self::ModuleLoadFailed(msg) => write!(f, "Module load failed: {}", msg),
        }
    }
}

impl std::error::Error for FlashAttentionError {}

fn check_hip(result: HipError) -> Result<(), FlashAttentionError> {
    if result == HIP_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = hipGetErrorString(result);
            if ptr.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(FlashAttentionError::Hip(result, msg))
    }
}

/// HIP device memory buffer.
pub struct HipBuffer<T> {
    ptr: HipDeviceptr,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> HipBuffer<T> {
    /// Allocate zeroed device memory.
    pub fn alloc_zeros(len: usize) -> Result<Self, FlashAttentionError> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr: HipDeviceptr = ptr::null_mut();

        unsafe {
            check_hip(hipMalloc(&mut ptr, size))?;
            check_hip(hipMemset(ptr, 0, size))?;
        }

        Ok(Self {
            ptr,
            len,
            _marker: std::marker::PhantomData,
        })
    }

    /// Copy from host slice to device.
    pub fn from_slice(data: &[T]) -> Result<Self, FlashAttentionError> {
        let size = data.len() * std::mem::size_of::<T>();
        let mut ptr: HipDeviceptr = ptr::null_mut();

        unsafe {
            check_hip(hipMalloc(&mut ptr, size))?;
            check_hip(hipMemcpy(
                ptr,
                data.as_ptr() as *const c_void,
                size,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ))?;
        }

        Ok(Self {
            ptr,
            len: data.len(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Copy device memory to host vector.
    pub fn to_vec(&self) -> Result<Vec<T>, FlashAttentionError>
    where
        T: Default + Clone,
    {
        let mut data = vec![T::default(); self.len];
        let size = self.len * std::mem::size_of::<T>();

        unsafe {
            check_hip(hipMemcpy(
                data.as_mut_ptr() as HipDeviceptr,
                self.ptr as *const c_void,
                size,
                HIP_MEMCPY_DEVICE_TO_HOST,
            ))?;
        }

        Ok(data)
    }

    /// Get raw device pointer.
    pub fn as_ptr(&self) -> HipDeviceptr {
        self.ptr
    }

    /// Get buffer length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for HipBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = hipFree(self.ptr);
            }
        }
    }
}

/// HIP stream wrapper.
pub struct HipStreamWrapper {
    stream: HipStream,
}

impl HipStreamWrapper {
    /// Create a new HIP stream.
    pub fn new() -> Result<Self, FlashAttentionError> {
        let mut stream: HipStream = ptr::null_mut();
        unsafe {
            check_hip(hipStreamCreate(&mut stream))?;
        }
        Ok(Self { stream })
    }

    /// Get null stream (default stream).
    pub fn null() -> Self {
        Self {
            stream: ptr::null_mut(),
        }
    }

    /// Synchronize the stream.
    pub fn synchronize(&self) -> Result<(), FlashAttentionError> {
        unsafe { check_hip(hipStreamSynchronize(self.stream)) }
    }

    /// Get raw stream handle.
    pub fn as_raw(&self) -> HipStream {
        self.stream
    }
}

impl Drop for HipStreamWrapper {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = hipStreamDestroy(self.stream);
            }
        }
    }
}

/// FlashAttention HIP kernel wrapper.
pub struct FlashAttentionKernel {
    module: HipModule,
    kernel_f32: HipFunction,
    kernel_f16: HipFunction,
}

impl FlashAttentionKernel {
    /// Initialize HIP runtime and load kernels.
    pub fn new(device: i32) -> Result<Self, FlashAttentionError> {
        unsafe {
            check_hip(hipInit(0))?;
            check_hip(hipSetDevice(device))?;
        }

        let module = Self::load_module()?;
        let kernel_f32 = Self::get_function(module, KERNEL_F32)?;
        let kernel_f16 = Self::get_function(module, KERNEL_F16)?;

        Ok(Self {
            module,
            kernel_f32,
            kernel_f16,
        })
    }

    fn load_module() -> Result<HipModule, FlashAttentionError> {
        // Try environment variable first
        if let Ok(path) = std::env::var("GLLM_HIP_FLASH_ATTN_HSACO") {
            let c_path = CString::new(path.clone())
                .map_err(|_| FlashAttentionError::ModuleLoadFailed("Invalid path".into()))?;

            let mut module: HipModule = ptr::null_mut();
            unsafe {
                check_hip(hipModuleLoad(&mut module, c_path.as_ptr()))?;
            }
            return Ok(module);
        }

        // Use embedded HSACO
        if PRECOMPILED_HSACO.len() > 1 {
            let mut module: HipModule = ptr::null_mut();
            unsafe {
                check_hip(hipModuleLoadData(
                    &mut module,
                    PRECOMPILED_HSACO.as_ptr() as *const c_void,
                ))?;
            }
            return Ok(module);
        }

        Err(FlashAttentionError::ModuleLoadFailed(
            "No HSACO binary available. Set GLLM_HIP_FLASH_ATTN_HSACO or compile kernels.".into(),
        ))
    }

    fn get_function(module: HipModule, name: &str) -> Result<HipFunction, FlashAttentionError> {
        let mut func: HipFunction = ptr::null_mut();
        unsafe {
            check_hip(hipModuleGetFunction(
                &mut func,
                module,
                name.as_ptr() as *const c_char,
            ))?;
        }

        if func.is_null() {
            return Err(FlashAttentionError::KernelMissing(
                if name == KERNEL_F32 {
                    "tiled_attention_forward_f32"
                } else {
                    "tiled_attention_forward_f16"
                },
            ));
        }

        Ok(func)
    }

    /// Forward pass for f32 inputs.
    pub fn forward_f32(
        &self,
        stream: &HipStreamWrapper,
        q: &HipBuffer<f32>,
        k: &HipBuffer<f32>,
        v: &HipBuffer<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<HipBuffer<f32>, FlashAttentionError> {
        self.forward_f32_impl(
            stream,
            q,
            k,
            v,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            is_causal,
            scale,
            position_offset,
            DEFAULT_BLOCK,
        )
    }

    /// Forward pass for f16 inputs.
    pub fn forward_f16(
        &self,
        stream: &HipStreamWrapper,
        q: &HipBuffer<u16>,
        k: &HipBuffer<u16>,
        v: &HipBuffer<u16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<HipBuffer<u16>, FlashAttentionError> {
        self.forward_f16_impl(
            stream,
            q,
            k,
            v,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            is_causal,
            scale,
            position_offset,
            DEFAULT_BLOCK,
        )
    }

    fn forward_f32_impl(
        &self,
        stream: &HipStreamWrapper,
        q: &HipBuffer<f32>,
        k: &HipBuffer<f32>,
        v: &HipBuffer<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
        block_size: u32,
    ) -> Result<HipBuffer<f32>, FlashAttentionError> {
        let (output_len, grid_dim, block_dim) =
            build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let output = HipBuffer::<f32>::alloc_zeros(output_len)?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let seq_len_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;
        let is_causal_i32 = if is_causal { 1i32 } else { 0i32 };
        let position_offset_i32 = position_offset as i32;

        let q_ptr = q.as_ptr();
        let k_ptr = k.as_ptr();
        let v_ptr = v.as_ptr();
        let o_ptr = output.ptr;

        let mut args: [*mut c_void; 11] = [
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &batch_size_i32 as *const _ as *mut c_void,
            &num_heads_i32 as *const _ as *mut c_void,
            &seq_len_i32 as *const _ as *mut c_void,
            &head_dim_i32 as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &is_causal_i32 as *const _ as *mut c_void,
            &position_offset_i32 as *const _ as *mut c_void,
        ];

        unsafe {
            check_hip(hipModuleLaunchKernel(
                self.kernel_f32,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream.as_raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }

        Ok(output)
    }

    fn forward_f16_impl(
        &self,
        stream: &HipStreamWrapper,
        q: &HipBuffer<u16>,
        k: &HipBuffer<u16>,
        v: &HipBuffer<u16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
        block_size: u32,
    ) -> Result<HipBuffer<u16>, FlashAttentionError> {
        let (output_len, grid_dim, block_dim) =
            build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let output = HipBuffer::<u16>::alloc_zeros(output_len)?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let seq_len_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;
        let is_causal_i32 = if is_causal { 1i32 } else { 0i32 };
        let position_offset_i32 = position_offset as i32;

        let q_ptr = q.as_ptr();
        let k_ptr = k.as_ptr();
        let v_ptr = v.as_ptr();
        let o_ptr = output.ptr;

        let mut args: [*mut c_void; 11] = [
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &batch_size_i32 as *const _ as *mut c_void,
            &num_heads_i32 as *const _ as *mut c_void,
            &seq_len_i32 as *const _ as *mut c_void,
            &head_dim_i32 as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &is_causal_i32 as *const _ as *mut c_void,
            &position_offset_i32 as *const _ as *mut c_void,
        ];

        unsafe {
            check_hip(hipModuleLaunchKernel(
                self.kernel_f16,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream.as_raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }

        Ok(output)
    }
}

impl Drop for FlashAttentionKernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = hipModuleUnload(self.module);
            }
        }
    }
}

/// Optimized HIP attention wrapper.
pub struct OptimizedHipAttention {
    tile_size: u32,
    kernel: FlashAttentionKernel,
}

impl OptimizedHipAttention {
    /// Create a new optimized attention wrapper.
    pub fn new(device: i32, tile_size: usize) -> Result<Self, FlashAttentionError> {
        let tile_size = tile_size.clamp(1, 1024) as u32;
        let kernel = FlashAttentionKernel::new(device)?;
        Ok(Self { tile_size, kernel })
    }

    /// Forward pass using tiled attention.
    pub fn forward_tiled(
        &self,
        stream: &HipStreamWrapper,
        q: &HipBuffer<f32>,
        k: &HipBuffer<f32>,
        v: &HipBuffer<f32>,
        config: &AttentionConfig,
        position_offset: usize,
    ) -> Result<HipBuffer<f32>, FlashAttentionError> {
        if config.query_len != config.kv_len {
            return Err(FlashAttentionError::InvalidConfig(
                "query_len must match kv_len for the tiled kernel".into(),
            ));
        }

        self.kernel.forward_f32_impl(
            stream,
            q,
            k,
            v,
            config.batch_size,
            config.num_heads,
            config.query_len,
            config.head_dim,
            config.causal,
            config.scale,
            position_offset,
            self.tile_size,
        )
    }
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: u32,
) -> Result<(usize, u32, u32), FlashAttentionError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err(FlashAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(FlashAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        )));
    }

    let num_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| FlashAttentionError::InvalidConfig("num_queries overflow".into()))?;

    let output_len = num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| FlashAttentionError::InvalidConfig("output_len overflow".into()))?;

    let block_dim = block_size.clamp(1, 1024);
    let grid_dim = ((num_queries + block_dim as usize - 1) / block_dim as usize) as u32;

    Ok((output_len, grid_dim, block_dim))
}

/// Get available HIP devices.
pub fn get_device_count() -> Result<i32, FlashAttentionError> {
    let mut count: c_int = 0;
    unsafe {
        check_hip(hipInit(0))?;
        check_hip(hipGetDeviceCount(&mut count))?;
    }
    Ok(count)
}

/// Synchronize all HIP devices.
pub fn device_synchronize() -> Result<(), FlashAttentionError> {
    unsafe { check_hip(hipDeviceSynchronize()) }
}
