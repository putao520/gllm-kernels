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

#[cfg(test)]
mod tests {
    use super::*;

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
}
