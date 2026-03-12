//! CudaDevice — implements `GpuDevice` trait for NVIDIA GPUs.
//!
//! Wraps a CUDA context + driver handle and provides memory allocation,
//! host↔device transfers, and stream management through the GpuDevice trait.

use std::sync::Arc;

use crate::gpu::{GpuDevice, GpuBuffer, GpuStream, GpuError};
use super::driver::*;

// ── CudaBuffer ──────────────────────────────────────────────────────

/// Device-side memory allocation backed by `cuMemAlloc_v2`.
pub struct CudaBuffer {
    ptr: CUdeviceptr,
    size: usize,
    driver: Arc<CudaDriver>,
}

impl GpuBuffer for CudaBuffer {
    fn as_device_ptr(&self) -> u64 {
        self.ptr
    }

    fn len(&self) -> usize {
        self.size
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { (self.driver.cuMemFree_v2)(self.ptr); }
        }
    }
}

// ── CudaStream ──────────────────────────────────────────────────────

/// Execution stream backed by `cuStreamCreate`.
pub struct CudaStream {
    handle: CUstream,
    driver: Arc<CudaDriver>,
}

impl GpuStream for CudaStream {
    fn synchronize(&self) -> Result<(), GpuError> {
        let res = unsafe { (self.driver.cuStreamSynchronize)(self.handle) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuStreamSynchronize failed with error {res}"
            )));
        }
        Ok(())
    }
}

impl CudaStream {
    /// Raw stream handle for kernel launches.
    pub fn handle(&self) -> CUstream {
        self.handle
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe { (self.driver.cuStreamDestroy_v2)(self.handle); }
        }
    }
}

// SAFETY: CUDA streams are thread-safe.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

// ── CudaDevice ──────────────────────────────────────────────────────

/// A CUDA device handle wrapping a `CUcontext`.
pub struct CudaDevice {
    driver: Arc<CudaDriver>,
    context: CUcontext,
    device_id: CUdevice,
    default_stream: CudaStream,
    name: String,
    total_memory: usize,
    sm_version: u32,
}

impl CudaDevice {
    /// Open device `ordinal` (0-based), create a context, and query capabilities.
    pub fn new(driver: Arc<CudaDriver>, ordinal: i32) -> Result<Self, GpuError> {
        // Get device handle
        let mut device_id: CUdevice = 0;
        let res = unsafe { (driver.cuDeviceGet)(&mut device_id, ordinal) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::DeviceNotFound(format!(
                "cuDeviceGet({ordinal}) failed with error {res}"
            )));
        }

        // Create context (flags=0 → default scheduling)
        let mut context: CUcontext = 0;
        let res = unsafe { (driver.cuCtxCreate_v2)(&mut context, 0, device_id) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuCtxCreate_v2 failed with error {res}"
            )));
        }

        // Query SM version
        let major = driver.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id)?;
        let minor = driver.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id)?;
        let sm_version = (major * 10 + minor) as u32;

        // Build device name from SM version (we avoid cuDeviceGetName for simplicity)
        let sm_count = driver.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_id)?;
        let name = format!("CUDA device {ordinal} (sm_{sm_version}, {sm_count} SMs)");

        // Create default stream (handle=0 is the CUDA default/legacy stream)
        let default_stream = CudaStream {
            handle: 0,
            driver: Arc::clone(&driver),
        };

        let total_memory = {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = unsafe { (driver.cuMemGetInfo_v2)(&mut free, &mut total) };
            if res == CUDA_SUCCESS { total } else { 0 }
        };

        Ok(Self {
            driver,
            context,
            device_id,
            default_stream,
            name,
            total_memory,
            sm_version,
        })
    }

    /// SM version (e.g. 80 for sm_80 / A100).
    pub fn sm_version(&self) -> u32 {
        self.sm_version
    }

    /// Raw CUDA context handle.
    pub fn context(&self) -> CUcontext {
        self.context
    }

    /// Reference to the underlying driver.
    pub fn driver(&self) -> &Arc<CudaDriver> {
        &self.driver
    }
}

impl GpuDevice for CudaDevice {
    type Buffer = CudaBuffer;
    type Stream = CudaStream;

    fn name(&self) -> &str {
        &self.name
    }

    fn total_memory(&self) -> usize {
        self.total_memory
    }

    fn free_memory(&self) -> usize {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let res = unsafe { (self.driver.cuMemGetInfo_v2)(&mut free, &mut total) };
        if res != CUDA_SUCCESS {
            return 0;
        }
        free
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Buffer, GpuError> {
        let mut ptr: CUdeviceptr = 0;
        let res = unsafe { (self.driver.cuMemAlloc_v2)(&mut ptr, bytes) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::OutOfMemory {
                requested: bytes,
                available: self.free_memory(),
            });
        }
        Ok(CudaBuffer {
            ptr,
            size: bytes,
            driver: Arc::clone(&self.driver),
        })
    }

    fn alloc_zeros(&self, bytes: usize) -> Result<Self::Buffer, GpuError> {
        let buf = self.alloc(bytes)?;
        // Zero-fill via host→device of zeroed buffer
        let zeros = vec![0u8; bytes];
        let res = unsafe {
            (self.driver.cuMemcpyHtoD_v2)(buf.ptr, zeros.as_ptr() as *const _, bytes)
        };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "cuMemcpyHtoD_v2 (zero-fill) failed with error {res}"
            )));
        }
        Ok(buf)
    }

    fn htod(
        &self,
        src: &[u8],
        dst: &mut Self::Buffer,
        _stream: &Self::Stream,
    ) -> Result<(), GpuError> {
        let res = unsafe {
            (self.driver.cuMemcpyHtoD_v2)(dst.ptr, src.as_ptr() as *const _, src.len())
        };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "cuMemcpyHtoD_v2 failed with error {res}"
            )));
        }
        Ok(())
    }

    fn dtoh(
        &self,
        src: &Self::Buffer,
        dst: &mut [u8],
        _stream: &Self::Stream,
    ) -> Result<(), GpuError> {
        let res = unsafe {
            (self.driver.cuMemcpyDtoH_v2)(dst.as_mut_ptr() as *mut _, src.ptr, dst.len())
        };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "cuMemcpyDtoH_v2 failed with error {res}"
            )));
        }
        Ok(())
    }

    fn dtod(
        &self,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        _stream: &Self::Stream,
    ) -> Result<(), GpuError> {
        let bytes = src.size.min(dst.size);
        let res = unsafe { (self.driver.cuMemcpyDtoD_v2)(dst.ptr, src.ptr, bytes) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "cuMemcpyDtoD_v2 failed with error {res}"
            )));
        }
        Ok(())
    }

    fn create_stream(&self) -> Result<Self::Stream, GpuError> {
        let mut handle: CUstream = 0;
        let res = unsafe { (self.driver.cuStreamCreate)(&mut handle, 0) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuStreamCreate failed with error {res}"
            )));
        }
        Ok(CudaStream {
            handle,
            driver: Arc::clone(&self.driver),
        })
    }

    fn default_stream(&self) -> &Self::Stream {
        &self.default_stream
    }

    fn sync(&self) -> Result<(), GpuError> {
        self.default_stream.synchronize()
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if self.context != 0 {
            unsafe { (self.driver.cuCtxDestroy_v2)(self.context); }
        }
    }
}
