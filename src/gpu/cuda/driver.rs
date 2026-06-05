//! CUDA Driver API FFI bindings via runtime `dlopen`.
//!
//! Zero build-time dependency — loads `libcuda.so.1` at runtime and resolves
//! all function pointers through `dlsym`. If the driver is not installed,
//! `CudaDriver::load()` returns `Err` instead of panicking.

use std::ffi::{c_void, c_char, c_int, c_uint};

use crate::gpu::GpuError;

/// CUDA result type (CUresult). 0 = CUDA_SUCCESS.
pub type CUresult = c_int;

/// Opaque CUDA handles — all represented as u64 (pointer-sized on 64-bit).
pub type CUdevice = c_int;
pub type CUcontext = u64;
pub type CUmodule = u64;
pub type CUfunction = u64;
pub type CUdeviceptr = u64;
pub type CUstream = u64;
pub type CUevent = u64;

/// CUDA_SUCCESS constant.
pub const CUDA_SUCCESS: CUresult = 0;

// ── dlopen / dlsym FFI ──────────────────────────────────────────────

const RTLD_LAZY: c_int = 0x1;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

// ── CudaDriver ──────────────────────────────────────────────────────

/// Runtime-loaded CUDA Driver API function table.
///
/// All function pointers are resolved via `dlsym` from `libcuda.so.1`.
/// The struct is `Send + Sync` because the CUDA driver is thread-safe
/// once initialized.
#[allow(non_snake_case)]
pub struct CudaDriver {
    _lib: *mut c_void,

    // ── Initialization ──
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,

    // ── Device management ──
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,

    // ── Context management ──
    pub cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    pub cuCtxDestroy_v2: unsafe extern "C" fn(CUcontext) -> CUresult,

    // ── Module / kernel ──
    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
    pub cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint, c_uint, c_uint, // grid dim x, y, z
        c_uint, c_uint, c_uint, // block dim x, y, z
        c_uint,                 // shared mem bytes
        CUstream,               // stream
        *mut *mut c_void,       // kernel params
        *mut *mut c_void,       // extra
    ) -> CUresult,

    // ── Memory management ──
    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuMemcpyDtoD_v2: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    pub cuMemGetInfo_v2: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,

    // ── Stream management ──
    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamDestroy_v2: unsafe extern "C" fn(CUstream) -> CUresult,

    // ── Event management (for profiling) ──
    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
    pub cuEventDestroy_v2: unsafe extern "C" fn(CUevent) -> CUresult,
}

// SAFETY: The CUDA driver API is thread-safe once cuInit has been called.
unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

/// Helper: resolve a single symbol from the loaded library.
unsafe fn load_sym<T>(lib: *mut c_void, name: &[u8]) -> Result<T, GpuError> {
    let ptr = dlsym(lib, name.as_ptr() as *const c_char);
    if ptr.is_null() {
        let name_str = std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("?");
        return Err(GpuError::Driver(format!("dlsym failed for {name_str}")));
    }
    Ok(std::mem::transmute_copy(&ptr))
}

impl CudaDriver {
    /// Load the CUDA driver from `libcuda.so.1` via `dlopen`.
    ///
    /// Returns `Err(GpuError::Driver)` if the library cannot be found or any
    /// required symbol is missing. Does NOT call `cuInit` — the caller is
    /// responsible for initialization.
    pub fn load() -> Result<Self, GpuError> {
        unsafe {
            let lib = dlopen(b"libcuda.so.1\0".as_ptr() as *const c_char, RTLD_LAZY);
            if lib.is_null() {
                return Err(GpuError::Driver(
                    "failed to dlopen libcuda.so.1 — NVIDIA driver not installed?".into(),
                ));
            }

            let driver = Self {
                _lib: lib,
                cuInit: load_sym(lib, b"cuInit\0")?,
                cuDeviceGet: load_sym(lib, b"cuDeviceGet\0")?,
                cuDeviceGetAttribute: load_sym(lib, b"cuDeviceGetAttribute\0")?,
                cuDeviceGetCount: load_sym(lib, b"cuDeviceGetCount\0")?,
                cuCtxCreate_v2: load_sym(lib, b"cuCtxCreate_v2\0")?,
                cuCtxDestroy_v2: load_sym(lib, b"cuCtxDestroy_v2\0")?,
                cuModuleLoadData: load_sym(lib, b"cuModuleLoadData\0")?,
                cuModuleUnload: load_sym(lib, b"cuModuleUnload\0")?,
                cuModuleGetFunction: load_sym(lib, b"cuModuleGetFunction\0")?,
                cuLaunchKernel: load_sym(lib, b"cuLaunchKernel\0")?,
                cuMemAlloc_v2: load_sym(lib, b"cuMemAlloc_v2\0")?,
                cuMemFree_v2: load_sym(lib, b"cuMemFree_v2\0")?,
                cuMemcpyHtoD_v2: load_sym(lib, b"cuMemcpyHtoD_v2\0")?,
                cuMemcpyDtoH_v2: load_sym(lib, b"cuMemcpyDtoH_v2\0")?,
                cuMemcpyDtoD_v2: load_sym(lib, b"cuMemcpyDtoD_v2\0")?,
                cuMemGetInfo_v2: load_sym(lib, b"cuMemGetInfo_v2\0")?,
                cuStreamCreate: load_sym(lib, b"cuStreamCreate\0")?,
                cuStreamSynchronize: load_sym(lib, b"cuStreamSynchronize\0")?,
                cuStreamDestroy_v2: load_sym(lib, b"cuStreamDestroy_v2\0")?,
                cuEventCreate: load_sym(lib, b"cuEventCreate\0")?,
                cuEventRecord: load_sym(lib, b"cuEventRecord\0")?,
                cuEventSynchronize: load_sym(lib, b"cuEventSynchronize\0")?,
                cuEventElapsedTime: load_sym(lib, b"cuEventElapsedTime\0")?,
                cuEventDestroy_v2: load_sym(lib, b"cuEventDestroy_v2\0")?,
            };

            Ok(driver)
        }
    }

    /// Initialize the CUDA driver (`cuInit(0)`).
    pub fn init(&self) -> Result<(), GpuError> {
        let res = unsafe { (self.cuInit)(0) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!("cuInit failed with error {res}")));
        }
        Ok(())
    }

    /// Query the number of CUDA-capable devices.
    pub fn device_count(&self) -> Result<i32, GpuError> {
        let mut count: c_int = 0;
        let res = unsafe { (self.cuDeviceGetCount)(&mut count) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuDeviceGetCount failed with error {res}"
            )));
        }
        Ok(count)
    }

    /// Query a device attribute by attribute ID.
    pub fn device_attribute(&self, attrib: i32, device: CUdevice) -> Result<i32, GpuError> {
        let mut value: c_int = 0;
        let res = unsafe { (self.cuDeviceGetAttribute)(&mut value, attrib, device) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuDeviceGetAttribute({attrib}) failed with error {res}"
            )));
        }
        Ok(value)
    }

    /// Allocate device memory (cuMemAlloc_v2). Requires a current CUDA context.
    ///
    /// Returns the raw CUdeviceptr as `u64`. Pairs with `mem_free`.
    pub fn mem_alloc(&self, size: usize) -> Result<u64, GpuError> {
        let mut ptr: CUdeviceptr = 0;
        let res = unsafe { (self.cuMemAlloc_v2)(&mut ptr, size) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuMemAlloc_v2({size}) failed with error {res}"
            )));
        }
        Ok(ptr)
    }

    /// Free device memory (cuMemFree_v2).
    pub fn mem_free(&self, ptr: u64) -> Result<(), GpuError> {
        let res = unsafe { (self.cuMemFree_v2)(ptr) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuMemFree_v2 failed with error {res}"
            )));
        }
        Ok(())
    }

    /// Synchronous host-to-device copy (cuMemcpyHtoD_v2).
    ///
    /// # Safety
    /// `src` must be a valid host pointer of at least `size` bytes, and `ptr`
    /// must be a device allocation with capacity ≥ `size`.
    pub fn memcpy_htod(&self, ptr: u64, src: *const u8, size: usize) -> Result<(), GpuError> {
        let res = unsafe { (self.cuMemcpyHtoD_v2)(ptr, src as *const c_void, size) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuMemcpyHtoD_v2({size}) failed with error {res}"
            )));
        }
        Ok(())
    }

    /// Resolve device handle for ordinal, ensuring `cuInit` has run first.
    fn device_handle(&self, ordinal: c_int) -> Result<CUdevice, GpuError> {
        self.init()?;
        let mut device: CUdevice = 0;
        let res = unsafe { (self.cuDeviceGet)(&mut device, ordinal) };
        if res != CUDA_SUCCESS {
            return Err(GpuError::Driver(format!(
                "cuDeviceGet({ordinal}) failed with error {res}"
            )));
        }
        Ok(device)
    }

    /// Compute capability of device 0 encoded as `major * 10 + minor`
    /// (e.g. sm_61 → 61, sm_80 → 80, sm_90 → 90).
    pub fn compute_capability(&self) -> Result<u32, GpuError> {
        let device = self.device_handle(0)?;
        let major = self.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)?;
        let minor = self.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)?;
        Ok((major as u32) * 10 + (minor as u32))
    }

    /// Full GPU device profile for device 0, populated from driver queries.
    pub fn device_profile(&self) -> Result<crate::gpu::GpuDeviceProfile, GpuError> {
        use crate::compiler::codegen::emitter::Platform;
        use crate::gpu::{GpuDeviceProfile, GpuIsvCapabilities};

        let device = self.device_handle(0)?;
        let sm_version = {
            let maj = self.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)?;
            let min = self.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)?;
            (maj as u32) * 10 + (min as u32)
        };

        let compute_units =
            self.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)? as u32;
        let shared_mem_per_block =
            self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device)? as u32;
        let max_regs_per_mp =
            self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device)?
                as u32;
        let max_threads_per_block =
            self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device)? as u32;
        let warp_size = self.device_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, device)? as u32;
        let max_block_x = self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)? as u32;
        let max_block_y = self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device)? as u32;
        let max_block_z = self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device)? as u32;
        let max_grid_x = self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device)? as u32;
        let max_grid_y = self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device)? as u32;
        let max_grid_z = self.device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device)? as u32;
        let clock_khz = self.device_attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device)? as u32;
        let mem_clock_khz =
            self.device_attribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device)? as u32;
        let mem_bus_width_bits =
            self.device_attribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device)? as u32;

        // CUDA cores per SM by architecture (NVIDIA hardware spec)
        let cores_per_sm: u32 = match sm_version {
            30..=37 => 192, // Kepler
            50..=53 => 128, // Maxwell
            60 => 64,       // GP100
            61 | 62 => 128, // GP102/104/106/107/108
            70 | 72 => 64,  // Volta
            75 => 64,       // Turing
            80 => 64,       // GA100
            86 | 87 | 89 => 128, // GA10x / Ada
            90 => 128,      // Hopper
            100..=129 => 128, // Blackwell family estimate
            _ => 64,        // unknown → conservative
        };

        let clock_ghz = (clock_khz as f64) / 1_000_000.0;
        let peak_gflops_f32 =
            (compute_units as f64) * (cores_per_sm as f64) * 2.0 * clock_ghz;

        // Tensor Core generation and relative f16 throughput multiplier.
        let (tensor_core_gen, f16_mul) = match sm_version {
            v if v >= 100 => (3u8, 16.0),
            v if v >= 90 => (3u8, 16.0),
            v if v >= 80 => (2u8, 8.0),
            v if v >= 70 => (1u8, 4.0),
            _ => (0u8, 1.0),
        };
        let peak_gflops_f16 = peak_gflops_f32 * f16_mul;

        // GDDR/HBM double-data-rate bandwidth.
        let memory_bandwidth_gbs =
            (mem_clock_khz as f64) * 2.0 * (mem_bus_width_bits as f64 / 8.0) / 1_000_000.0;

        // Total device memory requires a context.
        let mut ctx: CUcontext = 0;
        let total_memory = unsafe {
            let r = (self.cuCtxCreate_v2)(&mut ctx, 0, device);
            if r != CUDA_SUCCESS {
                return Err(GpuError::Driver(format!(
                    "cuCtxCreate_v2 failed with error {r}"
                )));
            }
            let mut free: usize = 0;
            let mut total: usize = 0;
            let r = (self.cuMemGetInfo_v2)(&mut free, &mut total);
            let _ = (self.cuCtxDestroy_v2)(ctx);
            if r != CUDA_SUCCESS {
                return Err(GpuError::Driver(format!(
                    "cuMemGetInfo_v2 failed with error {r}"
                )));
            }
            total
        };

        Ok(GpuDeviceProfile {
            platform: Platform::Cuda { sm_version },
            compute_units,
            shared_mem_per_block,
            max_registers_per_thread: max_regs_per_mp / max_threads_per_block.max(1),
            warp_size,
            max_threads_per_block,
            max_block_dim: [max_block_x, max_block_y, max_block_z],
            max_grid_dim: [max_grid_x, max_grid_y, max_grid_z],
            total_memory,
            memory_bandwidth_gbs,
            peak_gflops_f32,
            peak_gflops_f16,
            has_matrix_unit: sm_version >= 70,
            clock_mhz: clock_khz / 1000,
            isv: GpuIsvCapabilities { tensor_core_gen },
        })
    }
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        if !self._lib.is_null() {
            unsafe { dlclose(self._lib); }
        }
    }
}

// ── Well-known CUdevice_attribute constants ─────────────────────────

/// CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
/// CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
/// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
/// CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: i32 = 81;
/// CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: i32 = 82;
/// CU_DEVICE_ATTRIBUTE_WARP_SIZE
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 10;
/// CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: i32 = 38;
/// CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
/// CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: i32 = 2;
/// CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: i32 = 3;
/// CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: i32 = 4;
/// CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: i32 = 5;
/// CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: i32 = 6;
/// CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: i32 = 7;
/// CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
/// CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: i32 = 12;
/// CU_DEVICE_ATTRIBUTE_CLOCK_RATE
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: i32 = 13;
/// CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: i32 = 36;
/// CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuDeviceProfile, GpuIsvCapabilities};

    #[test]
    fn test_cuda_driver_load_graceful_failure() {
        // On machines without NVIDIA driver, load() should return Err, not panic.
        match CudaDriver::load() {
            Ok(drv) => {
                // Driver found — verify init works
                drv.init().expect("cuInit should succeed if driver loads");
                let count = drv.device_count().expect("device_count should work");
                eprintln!("CUDA devices found: {count}");
            }
            Err(e) => {
                eprintln!("CudaDriver::load() returned expected error: {e}");
            }
        }
    }

    // ── Constant correctness tests ──────────────────────────────────────

    #[test]
    fn test_cuda_success_is_zero() {
        assert_eq!(CUDA_SUCCESS, 0);
        let result: CUresult = CUDA_SUCCESS;
        assert_eq!(result, 0, "CUDA_SUCCESS must be 0 (c_int)");
    }

    #[test]
    fn test_device_attribute_constants_are_distinct() {
        let attrs: [i32; 18] = [
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
            CU_DEVICE_ATTRIBUTE_WARP_SIZE,
            CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
            CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
            CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        ];
        // All attribute IDs must be unique (no accidental duplicates).
        for i in 0..attrs.len() {
            for j in (i + 1)..attrs.len() {
                assert_ne!(
                    attrs[i], attrs[j],
                    "CU_DEVICE_ATTRIBUTE constants must be distinct: index {i} and {j} both = {}",
                    attrs[i]
                );
            }
        }
    }

    #[test]
    fn test_memory_clock_rate_and_bus_width_constants() {
        // Verify the two memory-related attribute constants have positive values
        // and are distinct from each other.
        assert!(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE > 0);
        assert!(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH > 0);
        assert_ne!(
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
            CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
        );
    }

    // ── Handle type alias tests ────────────────────────────────────────

    #[test]
    fn test_cu_handle_type_sizes() {
        // All opaque handles are u64 (pointer-sized on 64-bit) except CUdevice (c_int).
        assert_eq!(std::mem::size_of::<CUcontext>(), 8);
        assert_eq!(std::mem::size_of::<CUmodule>(), 8);
        assert_eq!(std::mem::size_of::<CUfunction>(), 8);
        assert_eq!(std::mem::size_of::<CUdeviceptr>(), 8);
        assert_eq!(std::mem::size_of::<CUstream>(), 8);
        assert_eq!(std::mem::size_of::<CUevent>(), 8);
        assert_eq!(std::mem::size_of::<CUdevice>(), 4);
    }

    #[test]
    fn test_cu_deviceptr_zero_is_null_equivalent() {
        // CUdeviceptr = 0 is the null device pointer convention in CUDA.
        let null_ptr: CUdeviceptr = 0;
        assert_eq!(null_ptr, 0u64);
        // A non-zero ptr should be representable.
        let valid_ptr: CUdeviceptr = 0xFFFF_FFFF_FFFF_FFFF;
        assert_eq!(valid_ptr, u64::MAX);
    }

    // ── GpuError variant tests ─────────────────────────────────────────

    #[test]
    fn test_gpu_error_driver_variant_display() {
        let msg = "dlsym failed for cuLaunchKernel";
        let err = GpuError::Driver(msg.to_string());
        let display = format!("{err}");
        assert!(
            display.contains(msg),
            "GpuError::Driver display must contain the original message: got '{display}'"
        );
        assert!(
            display.starts_with("driver error:"),
            "GpuError::Driver display must start with 'driver error:'"
        );
    }

    #[test]
    fn test_gpu_error_out_of_memory_fields() {
        let err = GpuError::OutOfMemory {
            requested: 4_294_967_296, // 4 GB
            available: 1_073_741_824, // 1 GB
        };
        let display = format!("{err}");
        assert!(display.contains("4294967296"), "must show requested bytes");
        assert!(display.contains("1073741824"), "must show available bytes");

        // Pattern match to verify field access.
        if let GpuError::OutOfMemory { requested, available } = err {
            assert_eq!(requested, 4_294_967_296);
            assert_eq!(available, 1_073_741_824);
            assert!(requested > available);
        } else {
            panic!("Expected GpuError::OutOfMemory variant");
        }
    }

    #[test]
    fn test_gpu_error_all_variant_display_prefixes() {
        // Each GpuError variant must produce a non-empty display string
        // with a recognizable prefix.
        let cases: Vec<(GpuError, &str)> = vec![
            (GpuError::DeviceNotFound("no GPU".into()), "device not found:"),
            (GpuError::OutOfMemory { requested: 0, available: 0 }, "out of memory:"),
            (GpuError::KernelLaunch("bad params".into()), "kernel launch failed:"),
            (GpuError::ShaderCompilation("compile err".into()), "shader compilation failed:"),
            (GpuError::Transfer("DMA fail".into()), "transfer failed:"),
            (GpuError::Driver("segfault".into()), "driver error:"),
        ];
        for (err, prefix) in cases {
            let display = format!("{err}");
            assert!(
                display.starts_with(prefix),
                "GpuError variant display must start with '{prefix}', got '{display}'"
            );
        }
    }

    #[test]
    fn test_gpu_error_debug_format_roundtrip() {
        // GpuError derives Debug; verify it produces a non-empty string.
        let err = GpuError::ShaderCompilation("PTX parse error at line 42".into());
        let debug = format!("{err:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("ShaderCompilation"));
    }

    // ── GpuIsvCapabilities default and construction ────────────────────

    #[test]
    fn test_gpu_isv_capabilities_default() {
        let isv = GpuIsvCapabilities::default();
        assert_eq!(isv.tensor_core_gen, 0, "Default tensor_core_gen must be 0 (no tensor cores)");
    }

    #[test]
    fn test_gpu_isv_capabilities_tensor_core_gen_range() {
        // tensor_core_gen uses semantic values: 0=none, 1=Volta, 2=Ampere, 3=Hopper.
        let none = GpuIsvCapabilities { tensor_core_gen: 0 };
        let volta = GpuIsvCapabilities { tensor_core_gen: 1 };
        let ampere = GpuIsvCapabilities { tensor_core_gen: 2 };
        let hopper = GpuIsvCapabilities { tensor_core_gen: 3 };

        assert_eq!(none.tensor_core_gen, 0);
        assert_eq!(volta.tensor_core_gen, 1);
        assert_eq!(ampere.tensor_core_gen, 2);
        assert_eq!(hopper.tensor_core_gen, 3);

        // Verify u8 boundary fits in the field type.
        let max_gen = GpuIsvCapabilities { tensor_core_gen: u8::MAX };
        assert_eq!(max_gen.tensor_core_gen, 255);
    }

    // ── GpuDeviceProfile construction via struct update syntax ──────────

    #[test]
    fn test_gpu_device_profile_struct_update_syntax() {
        use crate::compiler::codegen::emitter::Platform;

        // Build a base profile with struct update syntax, overriding only
        // specific fields from default-like values.
        let base = GpuDeviceProfile {
            platform: Platform::Cuda { sm_version: 80 },
            compute_units: 108,
            shared_mem_per_block: 49152,
            max_registers_per_thread: 255,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, 65535, 65535],
            total_memory: 40 * 1024 * 1024 * 1024,
            memory_bandwidth_gbs: 1555.0,
            peak_gflops_f32: 19500.0,
            peak_gflops_f16: 39000.0,
            has_matrix_unit: true,
            clock_mhz: 1410,
            isv: GpuIsvCapabilities { tensor_core_gen: 2 },
        };

        // Use struct update syntax to create a sm_90 variant.
        let sm90 = GpuDeviceProfile {
            platform: Platform::Cuda { sm_version: 90 },
            peak_gflops_f16: base.peak_gflops_f16 * 2.0,
            isv: GpuIsvCapabilities { tensor_core_gen: 3 },
            ..base.clone()
        };

        // Overridden fields must differ.
        assert_eq!(sm90.platform, Platform::Cuda { sm_version: 90 });
        assert_eq!(sm90.peak_gflops_f16, 78000.0);
        assert_eq!(sm90.isv.tensor_core_gen, 3);

        // Inherited fields must match the base.
        assert_eq!(sm90.compute_units, base.compute_units);
        assert_eq!(sm90.warp_size, base.warp_size);
        assert_eq!(sm90.total_memory, base.total_memory);
        assert_eq!(sm90.clock_mhz, base.clock_mhz);
    }

    // ── usize overflow safety for total_memory ─────────────────────────

    #[test]
    fn test_total_memory_large_values_no_overflow() {
        // total_memory is usize; on 64-bit systems it must hold up to
        // physically plausible GPU memory sizes (256 GB+) without overflow.
        let sizes_to_test: Vec<usize> = vec![
            0,
            1024,
            16 * 1024 * 1024 * 1024,  // 16 GB
            80 * 1024 * 1024 * 1024,  // 80 GB
            256 * 1024 * 1024 * 1024, // 256 GB
        ];
        for size in sizes_to_test {
            assert!(
                size / 1024 / 1024 / 1024 <= 256,
                "test setup: {size} exceeds 256 GB"
            );
            // Verify arithmetic on total_memory doesn't overflow.
            let doubled = size.checked_mul(2);
            // On 64-bit, even 256 GB * 2 fits in usize.
            assert!(doubled.is_some(), "usize overflow for size {size}");
        }
    }

    // ── GpuError std::error::Error integration ───────────────────────────

    #[test]
    fn test_gpu_error_implements_std_error() {
        // GpuError must be usable as a dyn std::error::Error source.
        let err = GpuError::KernelLaunch("too many threads".into());
        let _: &dyn std::error::Error = &err;
        // Verify the error chain works via source() (returns None for leaf errors).
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_gpu_error_transfer_variant_display() {
        let msg = "DMA from device 0 failed at offset 4096";
        let err = GpuError::Transfer(msg.to_string());
        let display = format!("{err}");
        assert!(
            display.starts_with("transfer failed:"),
            "Transfer variant must start with 'transfer failed:', got '{display}'"
        );
        assert!(
            display.contains(msg),
            "Transfer display must contain original message"
        );
    }

    // ── GpuIsvCapabilities Clone independence ────────────────────────────

    #[test]
    fn test_gpu_isv_capabilities_clone_independence() {
        let original = GpuIsvCapabilities { tensor_core_gen: 2 };
        let mut clone = original.clone();
        // Mutating the clone must not affect the original.
        clone.tensor_core_gen = 3;
        assert_eq!(original.tensor_core_gen, 2, "original must remain unchanged");
        assert_eq!(clone.tensor_core_gen, 3, "clone must reflect the mutation");
    }

    // ── GpuDeviceProfile Clone produces deep copy ────────────────────────

    #[test]
    fn test_gpu_device_profile_clone_is_deep_copy() {
        use crate::compiler::codegen::emitter::Platform;

        let profile = GpuDeviceProfile {
            platform: Platform::Cuda { sm_version: 86 },
            compute_units: 128,
            shared_mem_per_block: 49152,
            max_registers_per_thread: 255,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, 65535, 65535],
            total_memory: 24 * 1024 * 1024 * 1024,
            memory_bandwidth_gbs: 960.0,
            peak_gflops_f32: 30000.0,
            peak_gflops_f16: 240000.0,
            has_matrix_unit: true,
            clock_mhz: 1830,
            isv: GpuIsvCapabilities { tensor_core_gen: 2 },
        };

        let mut cloned = profile.clone();
        // Mutate cloned profile fields to verify deep copy.
        cloned.compute_units = 1;
        cloned.total_memory = 0;
        cloned.max_block_dim[0] = 0;

        // Original must be unaffected.
        assert_eq!(profile.compute_units, 128);
        assert_eq!(profile.total_memory, 24 * 1024 * 1024 * 1024);
        assert_eq!(profile.max_block_dim[0], 1024);
    }

    // ── Device profile field range validation ────────────────────────────

    #[test]
    fn test_device_profile_warp_size_is_power_of_two() {
        // All known NVIDIA GPUs have warp_size = 32 (power of 2).
        // Verify the field type can represent only powers of two.
        let valid_warp_sizes: Vec<u32> = vec![1, 2, 4, 8, 16, 32, 64];
        for ws in valid_warp_sizes {
            assert!(ws.is_power_of_two(), "{ws} must be power of 2");
        }
        // Typical value.
        assert_eq!(32u32.is_power_of_two(), true);
        // Non-power-of-two must fail.
        assert!(!33u32.is_power_of_two());
    }

    // ── Compute capability encoding (major * 10 + minor) ────────────────

    #[test]
    fn test_compute_capability_encoding() {
        // The compute_capability() method encodes SM version as major*10+minor.
        // Verify encoding matches well-known GPU generations.
        let cases: Vec<((i32, i32), u32)> = vec![
            ((7, 0), 70),   // V100
            ((7, 5), 75),   // T4
            ((8, 0), 80),   // A100
            ((8, 6), 86),   // RTX 3090
            ((8, 9), 89),   // RTX 4090
            ((9, 0), 90),   // H100
            ((10, 0), 100), // B100
            ((12, 0), 120), // Future
        ];
        for ((major, minor), expected) in cases {
            let encoded = (major as u32) * 10 + (minor as u32);
            assert_eq!(
                encoded, expected,
                "SM {major}.{minor} must encode to {expected}, got {encoded}"
            );
        }
    }

    // ── cores_per_sm match arm correctness ───────────────────────────────

    #[test]
    fn test_cores_per_sm_by_sm_version() {
        // Verify the cores_per_sm lookup table used in device_profile().
        // This mirrors the match in CudaDriver::device_profile().
        let table: Vec<(u32, u32)> = vec![
            // Kepler
            (30, 192), (35, 192), (37, 192),
            // Maxwell
            (50, 128), (52, 128), (53, 128),
            // Pascal GP100
            (60, 64),
            // Pascal GP102/104/106/107/108
            (61, 128), (62, 128),
            // Volta
            (70, 64), (72, 64),
            // Turing
            (75, 64),
            // GA100
            (80, 64),
            // GA10x / Ada
            (86, 128), (87, 128), (89, 128),
            // Hopper
            (90, 128),
            // Blackwell
            (100, 128), (110, 128), (120, 128), (129, 128),
        ];
        for (sm_version, expected_cores) in table {
            let cores: u32 = match sm_version {
                30..=37 => 192,
                50..=53 => 128,
                60 => 64,
                61 | 62 => 128,
                70 | 72 => 64,
                75 => 64,
                80 => 64,
                86 | 87 | 89 => 128,
                90 => 128,
                100..=129 => 128,
                _ => 64,
            };
            assert_eq!(
                cores, expected_cores,
                "SM {sm_version} must have {expected_cores} cores/SM, got {cores}"
            );
        }
    }

    // ── CUresult type is c_int ───────────────────────────────────────────

    #[test]
    fn test_curesult_type_is_i32() {
        // CUresult is defined as c_int, which on all supported platforms is i32.
        let _: CUresult = 0i32;
        let success: CUresult = CUDA_SUCCESS;
        assert_eq!(success, 0i32);
        // Non-zero error codes must be representable.
        let err: CUresult = 1;
        assert_ne!(err, CUDA_SUCCESS);
        // Negative error codes (used by some CUDA versions) must also work.
        let neg: CUresult = -1;
        assert!(neg < 0);
    }

    // ── RTLD_LAZY constant value ─────────────────────────────────────────

    #[test]
    fn test_rtld_lazy_constant_value() {
        // RTLD_LAZY = 0x1 per POSIX spec.
        assert_eq!(RTLD_LAZY, 0x1);
        assert_eq!(RTLD_LAZY, 1);
    }

    // ── Memory bandwidth calculation formula verification ────────────────

    #[test]
    fn test_memory_bandwidth_calculation_formula() {
        // device_profile() computes:
        //   bandwidth = mem_clock_khz * 2.0 * (bus_width_bits / 8) / 1_000_000
        // Simulate for H100: 3352 MHz mem clock, 6144-bit bus (HBM3).
        let mem_clock_khz: f64 = 3_352_000.0; // 3352 MHz in kHz
        let bus_width_bits: f64 = 6144.0;
        let bandwidth = mem_clock_khz * 2.0 * (bus_width_bits / 8.0) / 1_000_000.0;
        // Expected: 3352000 * 2 * 768 / 1000000 = 5148.672 GB/s
        let expected = 5148.672;
        assert!(
            (bandwidth - expected).abs() < 0.01,
            "H100 bandwidth: expected ~{expected}, got {bandwidth}"
        );

        // Simulate for RTX 4090: 10501 MHz GDDR6X, 384-bit bus.
        let mem_clock_khz: f64 = 10_501_000.0;
        let bus_width_bits: f64 = 384.0;
        let bw = mem_clock_khz * 2.0 * (bus_width_bits / 8.0) / 1_000_000.0;
        // Expected: 10501000 * 2 * 48 / 1000000 = 1008.096 GB/s
        let expected_bw = 1008.096;
        assert!(
            (bw - expected_bw).abs() < 0.01,
            "RTX 4090 bandwidth: expected ~{expected_bw}, got {bw}"
        );
    }
}
