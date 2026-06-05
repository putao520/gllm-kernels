//! HipDevice — implements `GpuDevice` trait for AMD GPUs.
//!
//! Wraps a HIP device handle + driver and provides memory allocation,
//! host↔device transfers, and stream management through the GpuDevice trait.

use std::ffi::c_void;
use std::sync::Arc;

use crate::gpu::{GpuDevice, GpuBuffer, GpuStream, GpuError};
use super::driver::*;

// ── HipBuffer ──────────────────────────────────────────────────────

/// Device-side memory allocation backed by `hipMalloc`.
pub struct HipBuffer {
    ptr: *mut c_void,
    size: usize,
    driver: Arc<HipDriver>,
}

impl GpuBuffer for HipBuffer {
    fn as_device_ptr(&self) -> u64 {
        self.ptr as u64
    }

    fn len(&self) -> usize {
        self.size
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.driver.hipFree)(self.ptr); }
        }
    }
}

// ── HipStream ──────────────────────────────────────────────────────

/// Execution stream backed by `hipStreamCreate`.
pub struct HipStream {
    handle: super::driver::HipStream,
    driver: Arc<HipDriver>,
}

impl GpuStream for HipStream {
    fn synchronize(&self) -> Result<(), GpuError> {
        let res = unsafe { (self.driver.hipStreamSynchronize)(self.handle) };
        if res != HIP_SUCCESS {
            return Err(GpuError::Driver(format!(
                "hipStreamSynchronize failed with error {res}"
            )));
        }
        Ok(())
    }
}

impl HipStream {
    /// Raw stream handle for kernel launches.
    pub fn handle(&self) -> super::driver::HipStream {
        self.handle
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.driver.hipStreamDestroy)(self.handle); }
        }
    }
}

// SAFETY: HIP streams are thread-safe.
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

// ── HipDevice ──────────────────────────────────────────────────────

/// A HIP device handle for AMD GPUs.
pub struct HipDevice {
    driver: Arc<HipDriver>,
    device_id: i32,
    default_stream: HipStream,
    name: String,
    total_memory: usize,
    gfx_arch: u32,
}

impl HipDevice {
    /// Open device `ordinal` (0-based), set it active, and query capabilities.
    pub fn new(driver: Arc<HipDriver>, ordinal: i32) -> Result<Self, GpuError> {
        // Set active device
        let res = unsafe { (driver.hipSetDevice)(ordinal) };
        if res != HIP_SUCCESS {
            return Err(GpuError::DeviceNotFound(format!(
                "hipSetDevice({ordinal}) failed with error {res}"
            )));
        }

        // Query gfx arch
        let gfx_arch = driver.gfx_arch(ordinal)?;

        // Build device name
        let raw_name = driver.device_name(ordinal).unwrap_or_default();
        let cu_count = driver.device_attribute(HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ordinal)
            .unwrap_or(0);
        let name = if raw_name.is_empty() {
            format!("HIP device {ordinal} (gfx{gfx_arch}, {cu_count} CUs)")
        } else {
            format!("{raw_name} (gfx{gfx_arch}, {cu_count} CUs)")
        };

        // Default stream (NULL = HIP default stream)
        let default_stream = HipStream {
            handle: std::ptr::null_mut(),
            driver: Arc::clone(&driver),
        };

        // Query total memory
        let total_memory = {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = unsafe { (driver.hipMemGetInfo)(&mut free, &mut total) };
            if res == HIP_SUCCESS { total } else { 0 }
        };

        Ok(Self {
            driver,
            device_id: ordinal,
            default_stream,
            name,
            total_memory,
            gfx_arch,
        })
    }

    /// GFX architecture number (e.g. 908 for gfx908 / MI100).
    pub fn gfx_arch(&self) -> u32 {
        self.gfx_arch
    }

    /// Reference to the underlying driver.
    pub fn driver(&self) -> &Arc<HipDriver> {
        &self.driver
    }

    /// 查询 GPU 硬件能力，构建 GpuDeviceProfile。
    pub fn gpu_profile(&self) -> Result<crate::gpu::GpuDeviceProfile, GpuError> {
        use crate::compiler::codegen::emitter::Platform;

        let d = &self.driver;
        let id = self.device_id;

        let compute_units = d.device_attribute(HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, id)? as u32;
        let shared_mem = d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, id)? as u32;
        let max_regs = d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, id)? as u32;
        let warp_size = d.device_attribute(HIP_DEVICE_ATTRIBUTE_WARP_SIZE, id)? as u32;
        let max_threads = d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, id)? as u32;

        let max_block_dim = [
            d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, id)? as u32,
            d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, id)? as u32,
            d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, id)? as u32,
        ];
        let max_grid_dim = [
            d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, id)? as u32,
            d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, id)? as u32,
            d.device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, id)? as u32,
        ];

        let clock_khz = d.device_attribute(HIP_DEVICE_ATTRIBUTE_CLOCK_RATE, id)? as u32;
        let clock_mhz = clock_khz / 1000;

        // Memory bandwidth estimate
        let mem_clock_khz = d.device_attribute(HIP_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, id).unwrap_or(0) as f64;
        let bus_width = d.device_attribute(HIP_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, id).unwrap_or(0) as f64;
        let memory_bandwidth_gbs = (mem_clock_khz * 1e-6) * (bus_width / 8.0) * 2.0;

        // Peak GFLOPS estimate for AMD GPUs
        // CDNA (gfx908/gfx90a/gfx940+): 64 stream processors per CU, 2 FMA ops
        // RDNA (gfx1010+): 64 stream processors per CU (dual-issue), 2 FMA ops
        let cores_per_cu: f64 = 64.0;
        let peak_gflops_f32 = (compute_units as f64) * cores_per_cu * 2.0 * (clock_mhz as f64 / 1000.0);

        // F16 throughput: CDNA with MFMA (gfx908+) gets ~2x, RDNA gets ~2x via packed math
        // gfx_arch is parsed as hex: gfx908=0x908=2312, gfx90a=0x90a=2314, gfx940=0x940=2368
        let peak_gflops_f16 = if self.gfx_arch >= 0x908 {
            peak_gflops_f32 * 2.0
        } else {
            peak_gflops_f32
        };

        // Hardware capability detection for JIT codegen decisions
        // gfx940+=CDNA3 (MI300), gfx908+=CDNA1/2 (MI100/MI200/MI210)
        let isv = crate::gpu::GpuIsvCapabilities {
            tensor_core_gen: if self.gfx_arch >= 0x940 { 3 }   // MI300 (MFMA v3)
                else if self.gfx_arch >= 0x908 { 2 }            // MI100/MI200 (MFMA v2)
                else { 0 },
            ..Default::default()
        };

        Ok(crate::gpu::GpuDeviceProfile {
            platform: Platform::Hip { gfx_arch: self.gfx_arch },
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
            peak_gflops_f16,
            has_matrix_unit: self.gfx_arch >= 0x908,
            clock_mhz,
            isv,
        })
    }

    /// Load HSACO/AMDGPU binary into a module.
    pub fn load_hsaco(&self, code: &[u8]) -> Result<HipModule, GpuError> {
        let mut module: super::driver::HipModule = std::ptr::null_mut();
        let res = unsafe {
            (self.driver.hipModuleLoadData)(&mut module, code.as_ptr() as *const _)
        };
        if res != 0 {
            return Err(GpuError::Driver(format!(
                "hipModuleLoadData failed with error {res}"
            )));
        }
        Ok(HipModule {
            module,
            driver: Arc::clone(&self.driver),
        })
    }

    /// Launch a kernel function.
    pub fn launch_kernel(
        &self,
        func: super::driver::HipFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*mut std::ffi::c_void],
        stream: &HipStream,
    ) -> Result<(), GpuError> {
        let res = unsafe {
            (self.driver.hipModuleLaunchKernel)(
                func,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                0,
                stream.handle(),
                args.as_ptr() as *mut *mut _,
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            return Err(GpuError::Driver(format!(
                "hipModuleLaunchKernel failed with error {res}"
            )));
        }
        Ok(())
    }
}

/// Loaded HIP module (owns the module handle, unloads on drop).
pub struct HipModule {
    module: super::driver::HipModule,
    driver: Arc<HipDriver>,
}

impl HipModule {
    pub fn get_function(&self, name: &str) -> Result<super::driver::HipFunction, GpuError> {
        let cname = std::ffi::CString::new(name).map_err(|_| {
            GpuError::Driver(format!("kernel name contains NUL: {name}"))
        })?;
        let mut func: super::driver::HipFunction = std::ptr::null_mut();
        let res = unsafe {
            (self.driver.hipModuleGetFunction)(&mut func, self.module, cname.as_ptr())
        };
        if res != 0 {
            return Err(GpuError::Driver(format!(
                "hipModuleGetFunction({name}) failed with error {res}"
            )));
        }
        Ok(func)
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe { (self.driver.hipModuleUnload)(self.module) };
        }
    }
}

impl GpuDevice for HipDevice {
    type Buffer = HipBuffer;
    type Stream = HipStream;

    fn name(&self) -> &str {
        &self.name
    }

    fn total_memory(&self) -> usize {
        self.total_memory
    }

    fn free_memory(&self) -> usize {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let res = unsafe { (self.driver.hipMemGetInfo)(&mut free, &mut total) };
        if res != HIP_SUCCESS {
            return 0;
        }
        free
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Buffer, GpuError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let res = unsafe { (self.driver.hipMalloc)(&mut ptr, bytes) };
        if res != HIP_SUCCESS {
            return Err(GpuError::OutOfMemory {
                requested: bytes,
                available: self.free_memory(),
            });
        }
        Ok(HipBuffer {
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
            (self.driver.hipMemcpyHtoD)(buf.ptr as u64, zeros.as_ptr() as *const _, bytes)
        };
        if res != HIP_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "hipMemcpyHtoD (zero-fill) failed with error {res}"
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
            (self.driver.hipMemcpyHtoD)(dst.ptr as u64, src.as_ptr() as *const _, src.len())
        };
        if res != HIP_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "hipMemcpyHtoD failed with error {res}"
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
            (self.driver.hipMemcpyDtoH)(dst.as_mut_ptr() as *mut _, src.ptr as u64, dst.len())
        };
        if res != HIP_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "hipMemcpyDtoH failed with error {res}"
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
        let res = unsafe {
            (self.driver.hipMemcpyDtoD)(dst.ptr as u64, src.ptr as u64, bytes)
        };
        if res != HIP_SUCCESS {
            return Err(GpuError::Transfer(format!(
                "hipMemcpyDtoD failed with error {res}"
            )));
        }
        Ok(())
    }

    fn create_stream(&self) -> Result<Self::Stream, GpuError> {
        let mut handle: super::driver::HipStream = std::ptr::null_mut();
        let res = unsafe { (self.driver.hipStreamCreate)(&mut handle) };
        if res != HIP_SUCCESS {
            return Err(GpuError::Driver(format!(
                "hipStreamCreate failed with error {res}"
            )));
        }
        Ok(HipStream {
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
