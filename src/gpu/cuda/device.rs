//! CudaDevice — implements `GpuDevice` trait for NVIDIA GPUs.
//!
//! Wraps a CUDA context + driver handle and provides memory allocation,
//! host↔device transfers, and stream management through the GpuDevice trait.

use std::sync::Arc;

use crate::gpu::{GpuDevice, GpuBuffer, GpuStream, GpuError};
use super::driver::*;

pub use super::driver::CUevent;

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

    /// 查询 GPU 硬件能力，构建 GpuDeviceProfile。
    pub fn gpu_profile(&self) -> Result<crate::gpu::GpuDeviceProfile, GpuError> {
        use super::driver::*;
        use crate::compiler::codegen::emitter::Platform;

        let d = &self.driver;
        let id = self.device_id;

        let compute_units = d.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, id)? as u32;
        let shared_mem = d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, id)? as u32;
        let max_regs = d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, id)? as u32;
        let warp_size = d.device_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, id)? as u32;
        let max_threads = d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, id)? as u32;

        let max_block_dim = [
            d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, id)? as u32,
            d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, id)? as u32,
            d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, id)? as u32,
        ];
        let max_grid_dim = [
            d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, id)? as u32,
            d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, id)? as u32,
            d.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, id)? as u32,
        ];

        let clock_khz = d.device_attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, id)? as u32;
        let clock_mhz = clock_khz / 1000;

        // Memory bandwidth estimate: mem_clock (MHz) * bus_width (bits) * 2 (DDR) / 8 (bytes)
        let mem_clock_khz = d.device_attribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, id).unwrap_or(0) as f64;
        let bus_width = d.device_attribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, id).unwrap_or(0) as f64;
        let memory_bandwidth_gbs = (mem_clock_khz * 1e-6) * (bus_width / 8.0) * 2.0;

        // Peak GFLOPS estimate: SMs * cores/SM * 2 (FMA) * clock_GHz
        let cores_per_sm: f64 = if self.sm_version >= 80 { 128.0 } else if self.sm_version >= 70 { 64.0 } else { 128.0 };
        let peak_gflops_f32 = (compute_units as f64) * cores_per_sm * 2.0 * (clock_mhz as f64 / 1000.0);

        Ok(crate::gpu::GpuDeviceProfile {
            platform: Platform::Cuda { sm_version: self.sm_version },
            compute_units,
            shared_mem_per_block: shared_mem,
            max_registers_per_thread: max_regs / max_threads.max(1),
            warp_size,
            max_threads_per_block: max_threads,
            max_block_dim,
            max_grid_dim,
            total_memory: self.total_memory,
            memory_bandwidth_gbs,
            peak_gflops_f32,
            peak_gflops_f16: if self.sm_version >= 70 { peak_gflops_f32 * 2.0 } else { peak_gflops_f32 },
            has_matrix_unit: self.sm_version >= 70,
            clock_mhz,
            isv: crate::gpu::GpuIsvCapabilities {
                tensor_core_gen: if self.sm_version >= 90 { 3 }
                    else if self.sm_version >= 80 { 2 }
                    else if self.sm_version >= 70 { 1 }
                    else { 0 },
                ..Default::default()
            },
        })
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

impl CudaDevice {
    /// Create a CUDA event for profiling.
    pub fn create_event(&self) -> Result<CUevent, GpuError> {
        let mut event: CUevent = 0;
        let res = unsafe { (self.driver.cuEventCreate)(&mut event, 0) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuEventCreate failed with error {res}"
            )));
        }
        Ok(event)
    }

    /// Record an event on a stream.
    pub fn record_event(&self, event: CUevent, stream: &CudaStream) -> Result<(), GpuError> {
        let res = unsafe { (self.driver.cuEventRecord)(event, stream.handle()) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuEventRecord failed with error {res}"
            )));
        }
        Ok(())
    }

    /// Synchronize on an event (wait for completion).
    pub fn sync_event(&self, event: CUevent) -> Result<(), GpuError> {
        let res = unsafe { (self.driver.cuEventSynchronize)(event) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuEventSynchronize failed with error {res}"
            )));
        }
        Ok(())
    }

    /// Compute elapsed time between two events in milliseconds.
    pub fn event_elapsed_time(&self, start: CUevent, end: CUevent) -> Result<f32, GpuError> {
        let mut elapsed_ms = 0.0f32;
        let res = unsafe { (self.driver.cuEventElapsedTime)(&mut elapsed_ms, start, end) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuEventElapsedTime failed with error {res}"
            )));
        }
        Ok(elapsed_ms)
    }

    /// Destroy a CUDA event.
    pub fn destroy_event(&self, event: CUevent) -> Result<(), GpuError> {
        let res = unsafe { (self.driver.cuEventDestroy_v2)(event) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuEventDestroy_v2 failed with error {res}"
            )));
        }
        Ok(())
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if self.context != 0 {
            unsafe { (self.driver.cuCtxDestroy_v2)(self.context); }
        }
    }
}

// ── CudaModule ───────────────────────────────────────────────────────

/// A loaded CUDA module (PTX or cubin) backed by `cuModuleLoadData`.
pub struct CudaModule {
    module: CUmodule,
    driver: Arc<CudaDriver>,
}

impl CudaModule {
    /// Look up a kernel function by name.
    pub fn get_function(&self, name: &str) -> Result<CUfunction, GpuError> {
        use std::ffi::CString;
        let cname = CString::new(name).map_err(|e| {
            GpuError::Driver(format!("invalid function name '{name}': {e}"))
        })?;
        let mut func: CUfunction = 0;
        let res = unsafe {
            (self.driver.cuModuleGetFunction)(&mut func, self.module, cname.as_ptr())
        };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuModuleGetFunction('{name}') failed with error {res}"
            )));
        }
        Ok(func)
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if self.module != 0 {
            unsafe { (self.driver.cuModuleUnload)(self.module); }
        }
    }
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaDevice {
    /// Load a PTX or cubin image into a new `CudaModule`.
    ///
    /// `ptx_bytes` must be a null-terminated PTX string or a cubin blob.
    pub fn load_ptx(&self, ptx_bytes: &[u8]) -> Result<CudaModule, GpuError> {
        let mut module: CUmodule = 0;
        let res = unsafe {
            (self.driver.cuModuleLoadData)(&mut module, ptx_bytes.as_ptr() as *const _)
        };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuModuleLoadData failed with error {res}"
            )));
        }
        Ok(CudaModule {
            module,
            driver: Arc::clone(&self.driver),
        })
    }

    /// Launch a kernel function.
    ///
    /// `args` is a slice of pointers to each kernel argument (as required by
    /// `cuLaunchKernel`'s `kernelParams` convention).
    pub fn launch_kernel(
        &self,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*mut std::ffi::c_void],
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        let res = unsafe {
            (self.driver.cuLaunchKernel)(
                func,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                0,                              // shared mem bytes
                stream.handle(),
                args.as_ptr() as *mut *mut _,
                std::ptr::null_mut(),           // extra
            )
        };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuLaunchKernel failed with error {res}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuBuffer, GpuDevice, GpuStream};

    /// Create a stub CudaDriver Arc for testing.
    ///
    /// Allocates a zeroed block of memory of CudaDriver's size and wraps it
    /// in an Arc via Box::from_raw. All fields (including function pointers)
    /// are zero. This is technically UB to hold (null fn pointers), but safe
    /// in practice because:
    ///   - Tests only call accessor methods that never dereference fn pointers
    ///   - Every test calls `forget_all` to prevent Drop from running
    ///   - No CUDA driver calls are ever made through the stub
    fn stub_driver() -> Arc<CudaDriver> {
        use std::alloc::{alloc_zeroed, Layout};
        let layout = Layout::new::<CudaDriver>();
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "failed to allocate CudaDriver stub");
        let boxed: Box<CudaDriver> = unsafe { Box::from_raw(ptr as *mut CudaDriver) };
        Arc::from(boxed)
    }

    /// Prevent Drop from running on all stub-backed structs.
    /// We must not drop these because the stub driver has null function pointers.
    macro_rules! forget_all {
        ($($item:expr),* $(,)?) => {
            $(
                std::mem::forget($item);
            )*
        };
    }

    #[test]
    fn test_cuda_buffer_accessors_reflect_fields() {
        // Arrange
        let driver = stub_driver();
        let buffer = CudaBuffer {
            ptr: 0xDEADBEEFu64,
            size: 4096,
            driver: Arc::clone(&driver),
        };

        // Act
        let ptr = buffer.as_device_ptr();
        let len = buffer.len();

        // Assert
        assert_eq!(ptr, 0xDEADBEEFu64, "as_device_ptr must return the raw CUdeviceptr");
        assert_eq!(len, 4096, "len must return the allocated size in bytes");

        forget_all!(buffer, driver);
    }

    #[test]
    fn test_cuda_buffer_is_empty_when_size_is_zero() {
        // Arrange
        let driver = stub_driver();
        let buffer = CudaBuffer {
            ptr: 0,
            size: 0,
            driver: Arc::clone(&driver),
        };

        // Act
        let empty = buffer.is_empty();

        // Assert
        assert!(empty, "buffer with size=0 must report is_empty=true");

        forget_all!(buffer, driver);
    }

    #[test]
    fn test_cuda_buffer_is_not_empty_when_size_is_nonzero() {
        // Arrange
        let driver = stub_driver();
        let buffer = CudaBuffer {
            ptr: 0x1000,
            size: 1024,
            driver: Arc::clone(&driver),
        };

        // Act
        let empty = buffer.is_empty();

        // Assert
        assert!(!empty, "buffer with size>0 must report is_empty=false");

        forget_all!(buffer, driver);
    }

    #[test]
    fn test_cuda_buffer_drop_skips_free_when_ptr_is_null() {
        // Arrange — ptr=0: Drop guard `if self.ptr != 0` prevents cuMemFree_v2 call.
        // We allow this one to drop naturally since ptr=0 means no driver call.
        // But the driver Arc inside the buffer will still drop, so we must
        // ensure the driver Arc survives. Clone an extra reference first.
        let driver = stub_driver();
        let buffer = CudaBuffer {
            ptr: 0,
            size: 256,
            driver: Arc::clone(&driver),
        };

        // Act — drop buffer (ptr=0, so cuMemFree_v2 is skipped)
        drop(buffer);

        // Assert — driver Arc still alive with refcount 1
        assert_eq!(Arc::strong_count(&driver), 1);

        forget_all!(driver);
    }

    #[test]
    fn test_cuda_stream_handle_accessor() {
        // Arrange
        let driver = stub_driver();
        let stream = CudaStream {
            handle: 0xABCDu64,
            driver: Arc::clone(&driver),
        };

        // Act
        let handle = stream.handle();

        // Assert
        assert_eq!(handle, 0xABCDu64, "handle() must return the raw CUstream");

        forget_all!(stream, driver);
    }

    #[test]
    fn test_cuda_stream_drop_skips_destroy_when_handle_is_null() {
        // Arrange — handle=0: Drop guard `if self.handle != 0` prevents cuStreamDestroy_v2.
        let driver = stub_driver();
        let stream = CudaStream {
            handle: 0,
            driver: Arc::clone(&driver),
        };

        // Act — drop stream (handle=0, so cuStreamDestroy_v2 is skipped)
        drop(stream);

        // Assert — driver Arc still alive
        assert_eq!(Arc::strong_count(&driver), 1);

        forget_all!(driver);
    }

    #[test]
    fn test_cuda_device_accessors() {
        // Arrange
        let driver = stub_driver();
        let default_stream = CudaStream {
            handle: 0,
            driver: Arc::clone(&driver),
        };
        let device = CudaDevice {
            driver: Arc::clone(&driver),
            context: 0,
            device_id: 3,
            default_stream,
            name: "Test CUDA sm_80".to_string(),
            total_memory: 40 * 1024 * 1024 * 1024,
            sm_version: 80,
        };

        // Act
        let sm = device.sm_version();
        let ctx = device.context();
        let name = device.name();
        let total = device.total_memory();
        let drv = device.driver();

        // Assert
        assert_eq!(sm, 80, "sm_version() must return 80");
        assert_eq!(ctx, 0, "context() must return the raw context handle");
        assert_eq!(name, "Test CUDA sm_80", "name() must return the device name");
        assert_eq!(total, 40 * 1024 * 1024 * 1024, "total_memory must return 40 GB");
        assert!(Arc::ptr_eq(drv, &driver), "driver() must return the same Arc");

        forget_all!(device, driver);
    }

    #[test]
    fn test_cuda_device_default_stream_returns_embedded_stream() {
        // Arrange
        let driver = stub_driver();
        let default_stream = CudaStream {
            handle: 42,
            driver: Arc::clone(&driver),
        };
        let device = CudaDevice {
            driver: Arc::clone(&driver),
            context: 0,
            device_id: 0,
            default_stream,
            name: String::new(),
            total_memory: 0,
            sm_version: 90,
        };

        // Act
        let stream: &<CudaDevice as GpuDevice>::Stream = device.default_stream();

        // Assert — the returned stream is the same one embedded in the device
        assert_eq!(stream.handle(), 42, "default stream handle must match the embedded stream");

        forget_all!(device, driver);
    }

    #[test]
    fn test_cuda_device_drop_skips_destroy_when_context_is_null() {
        // Arrange — context=0: Drop guard `if self.context != 0` prevents cuCtxDestroy_v2.
        let driver = stub_driver();
        let default_stream = CudaStream {
            handle: 0,
            driver: Arc::clone(&driver),
        };
        let device = CudaDevice {
            driver: Arc::clone(&driver),
            context: 0,
            device_id: 0,
            default_stream,
            name: String::new(),
            total_memory: 0,
            sm_version: 0,
        };

        // Act — drop device (context=0, so cuCtxDestroy_v2 is skipped;
        // default_stream handle=0, so cuStreamDestroy_v2 is also skipped)
        drop(device);

        // Assert — driver Arc still alive
        assert_eq!(Arc::strong_count(&driver), 1);

        forget_all!(driver);
    }

    #[test]
    fn test_cuda_buffer_arc_sharing_between_device_and_buffer() {
        // Arrange — verify Arc reference counting: device + buffer share the same driver
        let driver = stub_driver();
        let default_stream = CudaStream {
            handle: 0,
            driver: Arc::clone(&driver),
        };
        let device = CudaDevice {
            driver: Arc::clone(&driver),
            context: 0,
            device_id: 0,
            default_stream,
            name: "Arc test".to_string(),
            total_memory: 0,
            sm_version: 70,
        };
        let buffer = CudaBuffer {
            ptr: 0x2000,
            size: 512,
            driver: Arc::clone(&driver),
        };

        // Act — check Arc strong count:
        //   1) original `driver`
        //   2) default_stream.driver (inside device)
        //   3) device.driver
        //   4) buffer.driver
        let count = Arc::strong_count(&driver);

        // Assert
        assert_eq!(count, 4, "driver Arc must be shared by stub + stream + device + buffer");
        assert!(Arc::ptr_eq(device.driver(), &driver));
        assert_eq!(buffer.as_device_ptr(), 0x2000);
        assert_eq!(buffer.len(), 512);

        forget_all!(device, buffer, driver);
    }
}
