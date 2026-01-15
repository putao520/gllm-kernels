//! HIP/ROCm paged attention kernel wrapper.
//!
//! Provides paged attention kernels for AMD GPUs via HIP runtime.
//! Uses dynamic loading - no ROCm installation required at compile time.

use std::ffi::CString;
use std::fmt;
use std::os::raw::{c_char, c_uint, c_void};
use std::ptr;

use super::hip_runtime::{
    get_hip_lib, HipDeviceptr, HipError, HipFunction, HipModule, HipStream,
    HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS,
};

const KERNEL_F32: &str = "paged_attention_forward_f32\0";
const KERNEL_F16: &str = "paged_attention_forward_f16\0";
const DEFAULT_BLOCK: u32 = 128;
const MAX_HEAD_DIM: usize = 256;

// Embedded HSACO (placeholder - replace with actual compiled binary)
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/paged_attention.hsaco");

/// Errors from HIP paged attention kernels.
#[derive(Debug)]
pub enum PagedAttentionError {
    /// HIP runtime error.
    Hip(i32, String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Module load failure.
    ModuleLoadFailed(String),
    /// HIP library not available.
    HipNotAvailable(String),
}

impl fmt::Display for PagedAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(code, msg) => write!(f, "HIP error {}: {}", code, msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {}", name),
            Self::ModuleLoadFailed(msg) => write!(f, "Module load failed: {}", msg),
            Self::HipNotAvailable(msg) => write!(f, "HIP not available: {}", msg),
        }
    }
}

impl std::error::Error for PagedAttentionError {}

fn check_hip(result: HipError) -> Result<(), PagedAttentionError> {
    if result == HIP_SUCCESS {
        Ok(())
    } else {
        let msg = super::hip_runtime::get_error_string(result);
        Err(PagedAttentionError::Hip(result, msg))
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
    pub fn alloc_zeros(len: usize) -> Result<Self, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        let size = len * std::mem::size_of::<T>();
        let mut ptr: HipDeviceptr = ptr::null_mut();

        unsafe {
            check_hip((lib.hip_malloc)(&mut ptr, size))?;
            check_hip((lib.hip_memset)(ptr, 0, size))?;
        }

        Ok(Self {
            ptr,
            len,
            _marker: std::marker::PhantomData,
        })
    }

    /// Copy from host slice to device.
    pub fn from_slice(data: &[T]) -> Result<Self, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        let size = data.len() * std::mem::size_of::<T>();
        let mut ptr: HipDeviceptr = ptr::null_mut();

        unsafe {
            check_hip((lib.hip_malloc)(&mut ptr, size))?;
            check_hip((lib.hip_memcpy)(
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
    pub fn to_vec(&self) -> Result<Vec<T>, PagedAttentionError>
    where
        T: Default + Clone,
    {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        let mut data = vec![T::default(); self.len];
        let size = self.len * std::mem::size_of::<T>();

        unsafe {
            check_hip((lib.hip_memcpy)(
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
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for HipBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(lib) = get_hip_lib() {
                unsafe {
                    let _ = (lib.hip_free)(self.ptr);
                }
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
    pub fn new() -> Result<Self, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        let mut stream: HipStream = ptr::null_mut();
        unsafe {
            check_hip((lib.hip_stream_create)(&mut stream))?;
        }
        Ok(Self { stream })
    }

    /// Synchronize the stream.
    pub fn synchronize(&self) -> Result<(), PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;
        unsafe { check_hip((lib.hip_stream_synchronize)(self.stream)) }
    }

    /// Get raw stream.
    pub fn as_ptr(&self) -> HipStream {
        self.stream
    }
}

impl Drop for HipStreamWrapper {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            if let Ok(lib) = get_hip_lib() {
                unsafe {
                    let _ = (lib.hip_stream_destroy)(self.stream);
                }
            }
        }
    }
}

/// Paged attention HIP kernel wrapper.
pub struct PagedAttentionKernel {
    module: HipModule,
    kernel_f32: HipFunction,
    kernel_f16: HipFunction,
}

impl PagedAttentionKernel {
    /// Initialize HIP runtime and load kernels.
    pub fn new(device: i32) -> Result<Self, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        unsafe {
            check_hip((lib.hip_init)(0))?;
            check_hip((lib.hip_set_device)(device))?;
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

    fn load_module() -> Result<HipModule, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        // Priority 1: Embedded precompiled HSACO
        if PRECOMPILED_HSACO.len() > 1 {
            log::debug!("Loading precompiled HSACO from embedded data");
            let mut module: HipModule = ptr::null_mut();
            unsafe {
                check_hip((lib.hip_module_load_data)(
                    &mut module,
                    PRECOMPILED_HSACO.as_ptr() as *const c_void,
                ))?;
            }
            log::info!("Loaded embedded HSACO successfully");
            return Ok(module);
        }

        Err(PagedAttentionError::ModuleLoadFailed(
            "No HSACO binary available. Compile with hipcc and embed kernels.".into(),
        ))
    }

    fn get_function(module: HipModule, name: &str) -> Result<HipFunction, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        let mut func: HipFunction = ptr::null_mut();
        unsafe {
            check_hip((lib.hip_module_get_function)(
                &mut func,
                module,
                name.as_ptr() as *const c_char,
            ))?;
        }

        if func.is_null() {
            return Err(PagedAttentionError::KernelMissing(
                if name == KERNEL_F32 {
                    "paged_attention_forward_f32"
                } else {
                    "paged_attention_forward_f16"
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
        k_cache: &HipBuffer<f32>,
        v_cache: &HipBuffer<f32>,
        block_tables: &HipBuffer<i32>,
        block_offsets: &HipBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
    ) -> Result<HipBuffer<f32>, PagedAttentionError> {
        self.forward_f32_impl(
            stream,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_offsets,
            batch_size,
            num_heads,
            head_dim,
            page_block_size,
            seq_len,
            DEFAULT_BLOCK,
        )
    }

    /// Forward pass for f16 inputs (u16 storage).
    pub fn forward_f16(
        &self,
        stream: &HipStreamWrapper,
        q: &HipBuffer<u16>,
        k_cache: &HipBuffer<u16>,
        v_cache: &HipBuffer<u16>,
        block_tables: &HipBuffer<i32>,
        block_offsets: &HipBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
    ) -> Result<HipBuffer<u16>, PagedAttentionError> {
        self.forward_f16_impl(
            stream,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_offsets,
            batch_size,
            num_heads,
            head_dim,
            page_block_size,
            seq_len,
            DEFAULT_BLOCK,
        )
    }

    fn forward_f32_impl(
        &self,
        stream: &HipStreamWrapper,
        q: &HipBuffer<f32>,
        k_cache: &HipBuffer<f32>,
        v_cache: &HipBuffer<f32>,
        block_tables: &HipBuffer<i32>,
        block_offsets: &HipBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
        block_size: u32,
    ) -> Result<HipBuffer<f32>, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        let (output_len, grid_dim) = build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;
        let output = HipBuffer::<f32>::alloc_zeros(output_len)?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let page_block_i32 = page_block_size as i32;
        let seq_len_i32 = seq_len as i32;

        unsafe {
            let mut args: [*mut c_void; 11] = [
                &q.ptr as *const _ as *mut c_void,
                &k_cache.ptr as *const _ as *mut c_void,
                &v_cache.ptr as *const _ as *mut c_void,
                &block_tables.ptr as *const _ as *mut c_void,
                &block_offsets.ptr as *const _ as *mut c_void,
                &output.ptr as *const _ as *mut c_void,
                &batch_size_i32 as *const _ as *mut c_void,
                &num_heads_i32 as *const _ as *mut c_void,
                &head_dim_i32 as *const _ as *mut c_void,
                &page_block_i32 as *const _ as *mut c_void,
                &seq_len_i32 as *const _ as *mut c_void,
            ];

            check_hip((lib.hip_module_launch_kernel)(
                self.kernel_f32,
                grid_dim,
                1,
                1,
                block_size,
                1,
                1,
                0,
                stream.as_ptr(),
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
        k_cache: &HipBuffer<u16>,
        v_cache: &HipBuffer<u16>,
        block_tables: &HipBuffer<i32>,
        block_offsets: &HipBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
        block_size: u32,
    ) -> Result<HipBuffer<u16>, PagedAttentionError> {
        let lib = get_hip_lib()
            .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

        let (output_len, grid_dim) = build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;
        let output = HipBuffer::<u16>::alloc_zeros(output_len)?;

        let batch_size_i32 = batch_size as i32;
        let num_heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let page_block_i32 = page_block_size as i32;
        let seq_len_i32 = seq_len as i32;

        unsafe {
            let mut args: [*mut c_void; 11] = [
                &q.ptr as *const _ as *mut c_void,
                &k_cache.ptr as *const _ as *mut c_void,
                &v_cache.ptr as *const _ as *mut c_void,
                &block_tables.ptr as *const _ as *mut c_void,
                &block_offsets.ptr as *const _ as *mut c_void,
                &output.ptr as *const _ as *mut c_void,
                &batch_size_i32 as *const _ as *mut c_void,
                &num_heads_i32 as *const _ as *mut c_void,
                &head_dim_i32 as *const _ as *mut c_void,
                &page_block_i32 as *const _ as *mut c_void,
                &seq_len_i32 as *const _ as *mut c_void,
            ];

            check_hip((lib.hip_module_launch_kernel)(
                self.kernel_f16,
                grid_dim,
                1,
                1,
                block_size,
                1,
                1,
                0,
                stream.as_ptr(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }

        Ok(output)
    }
}

impl Drop for PagedAttentionKernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            if let Ok(lib) = get_hip_lib() {
                unsafe {
                    let _ = (lib.hip_module_unload)(self.module);
                }
            }
        }
    }
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: u32,
) -> Result<(usize, c_uint), PagedAttentionError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err(PagedAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(PagedAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        )));
    }

    let num_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| PagedAttentionError::InvalidConfig("num_queries overflow".into()))?;
    let output_len = num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| PagedAttentionError::InvalidConfig("output_len overflow".into()))?;

    let block_dim = block_size.clamp(1, 1024) as usize;
    let grid_dim = (num_queries + block_dim - 1) / block_dim;
    let grid_dim = c_uint::try_from(grid_dim).map_err(|_| {
        PagedAttentionError::InvalidConfig("grid_dim exceeds u32::MAX".into())
    })?;

    Ok((output_len, grid_dim))
}

#[allow(dead_code)]
pub fn get_device_count() -> Result<i32, PagedAttentionError> {
    let lib = get_hip_lib()
        .map_err(|e| PagedAttentionError::HipNotAvailable(e.to_string()))?;

    let mut count = 0;
    unsafe {
        check_hip((lib.hip_init)(0))?;
        check_hip((lib.hip_get_device_count)(&mut count))?;
    }
    Ok(count)
}
