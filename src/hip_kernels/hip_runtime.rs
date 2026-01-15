//! HIP runtime dynamic loading.
//!
//! This module provides dynamic loading of HIP/ROCm libraries similar to cudarc's
//! dynamic-loading feature. On systems without ROCm installed, the code will compile
//! but runtime HIP operations will return appropriate errors.

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use libloading::Library;

// HIP type definitions
pub type HipDevice = c_int;
pub type HipStream = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipDeviceptr = *mut c_void;
pub type HipError = c_int;

pub const HIP_SUCCESS: HipError = 0;

// Memory copy kinds
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// Function pointer types
type HipInitFn = unsafe extern "C" fn(c_uint) -> HipError;
type HipSetDeviceFn = unsafe extern "C" fn(HipDevice) -> HipError;
type HipGetDeviceFn = unsafe extern "C" fn(*mut HipDevice) -> HipError;
type HipGetDeviceCountFn = unsafe extern "C" fn(*mut c_int) -> HipError;

type HipMallocFn = unsafe extern "C" fn(*mut HipDeviceptr, usize) -> HipError;
type HipFreeFn = unsafe extern "C" fn(HipDeviceptr) -> HipError;
type HipMemcpyFn = unsafe extern "C" fn(HipDeviceptr, *const c_void, usize, c_int) -> HipError;
type HipMemsetFn = unsafe extern "C" fn(HipDeviceptr, c_int, usize) -> HipError;

type HipModuleLoadFn = unsafe extern "C" fn(*mut HipModule, *const c_char) -> HipError;
type HipModuleLoadDataFn = unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError;
type HipModuleGetFunctionFn =
    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError;
type HipModuleUnloadFn = unsafe extern "C" fn(HipModule) -> HipError;

type HipModuleLaunchKernelFn = unsafe extern "C" fn(
    HipFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipError;

type HipStreamCreateFn = unsafe extern "C" fn(*mut HipStream) -> HipError;
type HipStreamDestroyFn = unsafe extern "C" fn(HipStream) -> HipError;
type HipStreamSynchronizeFn = unsafe extern "C" fn(HipStream) -> HipError;

type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> HipError;
type HipGetErrorStringFn = unsafe extern "C" fn(HipError) -> *const c_char;

/// HIP library function table.
pub struct HipLib {
    #[allow(dead_code)]
    lib: Library,

    // Device management
    pub hip_init: HipInitFn,
    pub hip_set_device: HipSetDeviceFn,
    #[allow(dead_code)]
    pub hip_get_device: HipGetDeviceFn,
    pub hip_get_device_count: HipGetDeviceCountFn,

    // Memory management
    pub hip_malloc: HipMallocFn,
    pub hip_free: HipFreeFn,
    pub hip_memcpy: HipMemcpyFn,
    pub hip_memset: HipMemsetFn,

    // Module management
    pub hip_module_load: HipModuleLoadFn,
    pub hip_module_load_data: HipModuleLoadDataFn,
    pub hip_module_get_function: HipModuleGetFunctionFn,
    pub hip_module_unload: HipModuleUnloadFn,

    // Kernel launch
    pub hip_module_launch_kernel: HipModuleLaunchKernelFn,

    // Stream management
    pub hip_stream_create: HipStreamCreateFn,
    pub hip_stream_destroy: HipStreamDestroyFn,
    pub hip_stream_synchronize: HipStreamSynchronizeFn,

    // Synchronization
    pub hip_device_synchronize: HipDeviceSynchronizeFn,
    pub hip_get_error_string: HipGetErrorStringFn,
}

// Safety: HipLib contains function pointers from a loaded library.
// The library is loaded once and lives for the entire program lifetime.
// Function pointers are thread-safe as they're immutable after initialization.
unsafe impl Send for HipLib {}
unsafe impl Sync for HipLib {}

impl HipLib {
    /// Try to load the HIP library.
    fn load() -> Result<Self, String> {
        // Try common library paths for ROCm
        let lib_names = [
            "libamdhip64.so",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/lib64/libamdhip64.so",
        ];

        let lib = lib_names
            .iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| "Failed to load HIP library (libamdhip64.so)".to_string())?;

        // Get all function pointers first, then move lib into the struct.
        // We must dereference each Symbol before moving lib.
        let hip_init: HipInitFn = unsafe {
            *lib.get::<HipInitFn>(b"hipInit\0")
                .map_err(|e| format!("hipInit: {e}"))?
        };
        let hip_set_device: HipSetDeviceFn = unsafe {
            *lib.get::<HipSetDeviceFn>(b"hipSetDevice\0")
                .map_err(|e| format!("hipSetDevice: {e}"))?
        };
        let hip_get_device: HipGetDeviceFn = unsafe {
            *lib.get::<HipGetDeviceFn>(b"hipGetDevice\0")
                .map_err(|e| format!("hipGetDevice: {e}"))?
        };
        let hip_get_device_count: HipGetDeviceCountFn = unsafe {
            *lib.get::<HipGetDeviceCountFn>(b"hipGetDeviceCount\0")
                .map_err(|e| format!("hipGetDeviceCount: {e}"))?
        };

        let hip_malloc: HipMallocFn = unsafe {
            *lib.get::<HipMallocFn>(b"hipMalloc\0")
                .map_err(|e| format!("hipMalloc: {e}"))?
        };
        let hip_free: HipFreeFn = unsafe {
            *lib.get::<HipFreeFn>(b"hipFree\0")
                .map_err(|e| format!("hipFree: {e}"))?
        };
        let hip_memcpy: HipMemcpyFn = unsafe {
            *lib.get::<HipMemcpyFn>(b"hipMemcpy\0")
                .map_err(|e| format!("hipMemcpy: {e}"))?
        };
        let hip_memset: HipMemsetFn = unsafe {
            *lib.get::<HipMemsetFn>(b"hipMemset\0")
                .map_err(|e| format!("hipMemset: {e}"))?
        };

        let hip_module_load: HipModuleLoadFn = unsafe {
            *lib.get::<HipModuleLoadFn>(b"hipModuleLoad\0")
                .map_err(|e| format!("hipModuleLoad: {e}"))?
        };
        let hip_module_load_data: HipModuleLoadDataFn = unsafe {
            *lib.get::<HipModuleLoadDataFn>(b"hipModuleLoadData\0")
                .map_err(|e| format!("hipModuleLoadData: {e}"))?
        };
        let hip_module_get_function: HipModuleGetFunctionFn = unsafe {
            *lib.get::<HipModuleGetFunctionFn>(b"hipModuleGetFunction\0")
                .map_err(|e| format!("hipModuleGetFunction: {e}"))?
        };
        let hip_module_unload: HipModuleUnloadFn = unsafe {
            *lib.get::<HipModuleUnloadFn>(b"hipModuleUnload\0")
                .map_err(|e| format!("hipModuleUnload: {e}"))?
        };

        let hip_module_launch_kernel: HipModuleLaunchKernelFn = unsafe {
            *lib.get::<HipModuleLaunchKernelFn>(b"hipModuleLaunchKernel\0")
                .map_err(|e| format!("hipModuleLaunchKernel: {e}"))?
        };

        let hip_stream_create: HipStreamCreateFn = unsafe {
            *lib.get::<HipStreamCreateFn>(b"hipStreamCreate\0")
                .map_err(|e| format!("hipStreamCreate: {e}"))?
        };
        let hip_stream_destroy: HipStreamDestroyFn = unsafe {
            *lib.get::<HipStreamDestroyFn>(b"hipStreamDestroy\0")
                .map_err(|e| format!("hipStreamDestroy: {e}"))?
        };
        let hip_stream_synchronize: HipStreamSynchronizeFn = unsafe {
            *lib.get::<HipStreamSynchronizeFn>(b"hipStreamSynchronize\0")
                .map_err(|e| format!("hipStreamSynchronize: {e}"))?
        };

        let hip_device_synchronize: HipDeviceSynchronizeFn = unsafe {
            *lib.get::<HipDeviceSynchronizeFn>(b"hipDeviceSynchronize\0")
                .map_err(|e| format!("hipDeviceSynchronize: {e}"))?
        };
        let hip_get_error_string: HipGetErrorStringFn = unsafe {
            *lib.get::<HipGetErrorStringFn>(b"hipGetErrorString\0")
                .map_err(|e| format!("hipGetErrorString: {e}"))?
        };

        // Now lib is no longer borrowed, we can move it
        Ok(Self {
            lib,
            hip_init,
            hip_set_device,
            hip_get_device,
            hip_get_device_count,
            hip_malloc,
            hip_free,
            hip_memcpy,
            hip_memset,
            hip_module_load,
            hip_module_load_data,
            hip_module_get_function,
            hip_module_unload,
            hip_module_launch_kernel,
            hip_stream_create,
            hip_stream_destroy,
            hip_stream_synchronize,
            hip_device_synchronize,
            hip_get_error_string,
        })
    }
}

/// Global HIP library instance.
static HIP_LIB: OnceLock<Result<HipLib, String>> = OnceLock::new();

/// Get the global HIP library instance.
///
/// Returns Ok if HIP is available, Err with a message if not.
pub fn get_hip_lib() -> Result<&'static HipLib, &'static str> {
    HIP_LIB
        .get_or_init(HipLib::load)
        .as_ref()
        .map_err(|e| e.as_str())
}

/// Check if HIP is available on this system.
pub fn is_hip_available() -> bool {
    get_hip_lib().is_ok()
}

/// Get error string from HIP error code.
pub fn get_error_string(error: HipError) -> String {
    if let Ok(lib) = get_hip_lib() {
        unsafe {
            let ptr = (lib.hip_get_error_string)(error);
            if ptr.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    } else {
        format!("HIP error code: {error}")
    }
}
