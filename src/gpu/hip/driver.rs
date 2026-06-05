//! HIP Driver API FFI bindings via runtime `dlopen`.
//!
//! Zero build-time dependency — loads `libamdhip64.so` at runtime and resolves
//! all function pointers through `dlsym`. If the driver is not installed,
//! `HipDriver::load()` returns `Err` instead of panicking.

use std::ffi::{c_void, c_char, c_int, c_uint};

use crate::gpu::GpuError;

/// HIP result type (hipError_t). 0 = hipSuccess.
pub type HipResult = c_int;

/// Opaque HIP handles.
pub type HipDeviceptr = u64;
pub type HipStream = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;

/// hipSuccess constant.
pub const HIP_SUCCESS: HipResult = 0;

// ── dlopen / dlsym FFI ──────────────────────────────────────────────

const RTLD_LAZY: c_int = 0x1;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

// ── HipDriver ──────────────────────────────────────────────────────

/// Runtime-loaded HIP Driver API function table.
///
/// All function pointers are resolved via `dlsym` from `libamdhip64.so`.
#[allow(non_snake_case)]
pub struct HipDriver {
    _lib: *mut c_void,

    // ── Initialization ──
    pub hipInit: unsafe extern "C" fn(c_uint) -> HipResult,

    // ── Device management ──
    pub hipGetDeviceCount: unsafe extern "C" fn(*mut c_int) -> HipResult,
    pub hipSetDevice: unsafe extern "C" fn(c_int) -> HipResult,
    pub hipDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> HipResult,

    // ── Memory management ──
    pub hipMalloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipResult,
    pub hipFree: unsafe extern "C" fn(*mut c_void) -> HipResult,
    pub hipMemcpyHtoD: unsafe extern "C" fn(HipDeviceptr, *const c_void, usize) -> HipResult,
    pub hipMemcpyDtoH: unsafe extern "C" fn(*mut c_void, HipDeviceptr, usize) -> HipResult,
    pub hipMemcpyDtoD: unsafe extern "C" fn(HipDeviceptr, HipDeviceptr, usize) -> HipResult,
    pub hipMemGetInfo: unsafe extern "C" fn(*mut usize, *mut usize) -> HipResult,

    // ── Stream management ──
    pub hipStreamCreate: unsafe extern "C" fn(*mut HipStream) -> HipResult,
    pub hipStreamSynchronize: unsafe extern "C" fn(HipStream) -> HipResult,
    pub hipStreamDestroy: unsafe extern "C" fn(HipStream) -> HipResult,

    // ── Module / kernel ──
    pub hipModuleLoadData: unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipResult,
    pub hipModuleUnload: unsafe extern "C" fn(HipModule) -> HipResult,
    pub hipModuleGetFunction: unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipResult,
    pub hipModuleLaunchKernel: unsafe extern "C" fn(
        HipFunction,
        c_uint, c_uint, c_uint, // grid dim x, y, z
        c_uint, c_uint, c_uint, // block dim x, y, z
        c_uint,                 // shared mem bytes
        HipStream,              // stream
        *mut *mut c_void,       // kernel params
        *mut *mut c_void,       // extra
    ) -> HipResult,

    // ── GCN arch name (for gfx_arch detection) ──
    pub hipDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipResult,
    pub hipGetDeviceProperties: unsafe extern "C" fn(*mut c_void, c_int) -> HipResult,
}

// SAFETY: The HIP runtime API is thread-safe once hipInit has been called.
unsafe impl Send for HipDriver {}
unsafe impl Sync for HipDriver {}

/// Helper: resolve a single symbol from the loaded library.
unsafe fn load_sym<T>(lib: *mut c_void, name: &[u8]) -> Result<T, GpuError> {
    let ptr = dlsym(lib, name.as_ptr() as *const c_char);
    if ptr.is_null() {
        let name_str = std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("?");
        return Err(GpuError::Driver(format!("dlsym failed for {name_str}")));
    }
    Ok(std::mem::transmute_copy(&ptr))
}

impl HipDriver {
    /// Load the HIP runtime from `libamdhip64.so` via `dlopen`.
    ///
    /// Returns `Err(GpuError::Driver)` if the library cannot be found or any
    /// required symbol is missing. Does NOT call `hipInit` — the caller is
    /// responsible for initialization.
    pub fn load() -> Result<Self, GpuError> {
        unsafe {
            let lib = dlopen(b"libamdhip64.so\0".as_ptr() as *const c_char, RTLD_LAZY);
            if lib.is_null() {
                return Err(GpuError::Driver(
                    "failed to dlopen libamdhip64.so — ROCm not installed?".into(),
                ));
            }

            let driver = Self {
                _lib: lib,
                hipInit: load_sym(lib, b"hipInit\0")?,
                hipGetDeviceCount: load_sym(lib, b"hipGetDeviceCount\0")?,
                hipSetDevice: load_sym(lib, b"hipSetDevice\0")?,
                hipDeviceGetAttribute: load_sym(lib, b"hipDeviceGetAttribute\0")?,
                hipMalloc: load_sym(lib, b"hipMalloc\0")?,
                hipFree: load_sym(lib, b"hipFree\0")?,
                hipMemcpyHtoD: load_sym(lib, b"hipMemcpyHtoD\0")?,
                hipMemcpyDtoH: load_sym(lib, b"hipMemcpyDtoH\0")?,
                hipMemcpyDtoD: load_sym(lib, b"hipMemcpyDtoD\0")?,
                hipMemGetInfo: load_sym(lib, b"hipMemGetInfo\0")?,
                hipStreamCreate: load_sym(lib, b"hipStreamCreate\0")?,
                hipStreamSynchronize: load_sym(lib, b"hipStreamSynchronize\0")?,
                hipStreamDestroy: load_sym(lib, b"hipStreamDestroy\0")?,
                hipModuleLoadData: load_sym(lib, b"hipModuleLoadData\0")?,
                hipModuleUnload: load_sym(lib, b"hipModuleUnload\0")?,
                hipModuleGetFunction: load_sym(lib, b"hipModuleGetFunction\0")?,
                hipModuleLaunchKernel: load_sym(lib, b"hipModuleLaunchKernel\0")?,
                hipDeviceGetName: load_sym(lib, b"hipDeviceGetName\0")?,
                hipGetDeviceProperties: load_sym(lib, b"hipGetDeviceProperties\0")?,
            };

            Ok(driver)
        }
    }

    /// Initialize the HIP runtime (`hipInit(0)`).
    pub fn init(&self) -> Result<(), GpuError> {
        let res = unsafe { (self.hipInit)(0) };
        if res != HIP_SUCCESS {
            return Err(GpuError::Driver(format!("hipInit failed with error {res}")));
        }
        Ok(())
    }

    /// Query the number of HIP-capable devices.
    pub fn device_count(&self) -> Result<i32, GpuError> {
        let mut count: c_int = 0;
        let res = unsafe { (self.hipGetDeviceCount)(&mut count) };
        if res != HIP_SUCCESS {
            return Err(GpuError::Driver(format!(
                "hipGetDeviceCount failed with error {res}"
            )));
        }
        Ok(count)
    }

    /// Query a device attribute by attribute ID.
    pub fn device_attribute(&self, attrib: i32, device: i32) -> Result<i32, GpuError> {
        let mut value: c_int = 0;
        let res = unsafe { (self.hipDeviceGetAttribute)(&mut value, attrib, device) };
        if res != HIP_SUCCESS {
            return Err(GpuError::Driver(format!(
                "hipDeviceGetAttribute({attrib}) failed with error {res}"
            )));
        }
        Ok(value)
    }

    /// Query device name string.
    pub fn device_name(&self, device: i32) -> Result<String, GpuError> {
        let mut buf = [0u8; 256];
        let res = unsafe {
            (self.hipDeviceGetName)(buf.as_mut_ptr() as *mut c_char, 256, device)
        };
        if res != HIP_SUCCESS {
            return Err(GpuError::Driver(format!(
                "hipDeviceGetName failed with error {res}"
            )));
        }
        let name = unsafe {
            std::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char)
        };
        Ok(name.to_string_lossy().into_owned())
    }

    /// Extract gfx_arch number from device properties.
    ///
    /// Uses `hipGetDeviceProperties` to read `gcnArchName` (e.g. "gfx908",
    /// "gfx90a", "gfx1100") and parses the hex suffix.
    ///
    /// AMD gfx names use hex: gfx90a = 0x90a = 2314, gfx940 = 0x940 = 2368.
    /// We parse the suffix as hex to correctly handle all variants.
    pub fn gfx_arch(&self, device: i32) -> Result<u32, GpuError> {
        // hipDeviceProp_t is very large (3.5KB+ on ROCm 6.x). We allocate a
        // generous 8192-byte zeroed buffer to safely cover all ROCm versions.
        let mut buf = vec![0u8; 8192];
        let res = unsafe {
            (self.hipGetDeviceProperties)(buf.as_mut_ptr() as *mut c_void, device)
        };
        if res != HIP_SUCCESS {
            return Err(GpuError::Driver(format!(
                "hipGetDeviceProperties failed with error {res}"
            )));
        }

        // Scan the buffer for "gfx" pattern to find gcnArchName.
        // This is more robust than relying on a fixed offset.
        let buf_str = String::from_utf8_lossy(&buf);
        if let Some(arch) = parse_gfx_arch_from_str(&buf_str) {
            return Ok(arch);
        }

        // Fallback: try device name which often contains gfx arch.
        let name = self.device_name(device)?;
        if let Some(arch) = parse_gfx_arch_from_str(&name) {
            return Ok(arch);
        }

        Err(GpuError::Driver(
            "could not determine gfx_arch from device properties or name".into(),
        ))
    }
}

impl Drop for HipDriver {
    fn drop(&mut self) {
        if !self._lib.is_null() {
            unsafe { dlclose(self._lib); }
        }
    }
}

// ── gfx_arch parsing ──────────────────────────────────────────────────

/// Parse gfx arch number from a string containing "gfxNNN" pattern.
///
/// AMD gfx names use hex digits: "gfx90a" = 0x90a = 2314, "gfx940" = 0x940 = 2368,
/// "gfx1100" = 0x1100 = 4352. We parse the suffix as hex.
fn parse_gfx_arch_from_str(s: &str) -> Option<u32> {
    let pos = s.find("gfx")?;
    let after_gfx = &s[pos + 3..];
    let hex_str: String = after_gfx
        .chars()
        .take_while(|c| c.is_ascii_hexdigit())
        .collect();
    if hex_str.is_empty() {
        return None;
    }
    u32::from_str_radix(&hex_str, 16).ok()
}

// ── Well-known hipDeviceAttribute_t constants ─────────────────────────

/// hipDeviceAttributeMultiprocessorCount
pub const HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
/// hipDeviceAttributeMaxSharedMemoryPerBlock
pub const HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
/// hipDeviceAttributeMaxRegistersPerBlock
pub const HIP_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: i32 = 12;
/// hipDeviceAttributeWarpSize
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 10;
/// hipDeviceAttributeMaxThreadsPerBlock
pub const HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
/// hipDeviceAttributeMaxBlockDimX
pub const HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: i32 = 2;
/// hipDeviceAttributeMaxBlockDimY
pub const HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: i32 = 3;
/// hipDeviceAttributeMaxBlockDimZ
pub const HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: i32 = 4;
/// hipDeviceAttributeMaxGridDimX
pub const HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: i32 = 5;
/// hipDeviceAttributeMaxGridDimY
pub const HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: i32 = 6;
/// hipDeviceAttributeMaxGridDimZ
pub const HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: i32 = 7;
/// hipDeviceAttributeClockRate
pub const HIP_DEVICE_ATTRIBUTE_CLOCK_RATE: i32 = 13;
/// hipDeviceAttributeMemoryClockRate
pub const HIP_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: i32 = 36;
/// hipDeviceAttributeGlobalMemoryBusWidth
pub const HIP_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;
/// hipDeviceAttributeL2CacheSize
pub const HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: i32 = 38;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_driver_load_graceful_failure() {
        // On machines without ROCm, load() should return Err, not panic.
        match HipDriver::load() {
            Ok(drv) => {
                drv.init().expect("hipInit should succeed if driver loads");
                let count = drv.device_count().expect("device_count should work");
                eprintln!("HIP devices found: {count}");
            }
            Err(e) => {
                eprintln!("HipDriver::load() returned expected error: {e}");
            }
        }
    }

    #[test]
    fn test_parse_gfx_arch_hex() {
        // Pure decimal-looking names
        assert_eq!(parse_gfx_arch_from_str("gfx908"), Some(0x908));
        assert_eq!(parse_gfx_arch_from_str("gfx940"), Some(0x940));
        assert_eq!(parse_gfx_arch_from_str("gfx1100"), Some(0x1100));
        // Hex suffix: gfx90a = MI200
        assert_eq!(parse_gfx_arch_from_str("gfx90a"), Some(0x90a));
        // Embedded in longer string
        assert_eq!(parse_gfx_arch_from_str("AMD Instinct MI200 (gfx90a)"), Some(0x90a));
        assert_eq!(parse_gfx_arch_from_str("some\0garbage\0gfx942\0more"), Some(0x942));
        // No match
        assert_eq!(parse_gfx_arch_from_str("no arch here"), None);
        assert_eq!(parse_gfx_arch_from_str("gfx"), None);
    }

    #[test]
    fn test_gfx_arch_ordering() {
        // Verify hex ordering matches expected hardware generations
        let gfx908 = parse_gfx_arch_from_str("gfx908").unwrap(); // MI100
        let gfx90a = parse_gfx_arch_from_str("gfx90a").unwrap(); // MI200
        let gfx940 = parse_gfx_arch_from_str("gfx940").unwrap(); // MI300
        let gfx1100 = parse_gfx_arch_from_str("gfx1100").unwrap(); // RDNA3

        assert!(gfx908 < gfx90a, "MI100 < MI200");
        assert!(gfx90a < gfx940, "MI200 < MI300");
        assert!(gfx940 < gfx1100, "MI300 < RDNA3");

        // All CDNA should have matrix units (>= 0x908)
        assert!(gfx908 >= 0x908);
        assert!(gfx90a >= 0x908);
        assert!(gfx940 >= 0x940);
    }
}
