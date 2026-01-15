//! HSA Runtime (ROCr) dynamic loading.
//!
//! This module provides dynamic loading of the HSA Runtime library (libhsa-runtime64.so).
//! HSA Runtime is the low-level driver API for AMD GPUs - it only requires the AMD GPU
//! driver to be installed, NOT the full ROCm development toolkit.
//!
//! Architecture:
//! - Compile time: HIP kernel source → HSACO binary (via hipcc/offline compiler)
//! - Runtime: HSA Runtime loads HSACO → GPU executes
//!
//! This is analogous to CUDA's driver API (libcuda.so) vs runtime API (libcudart.so).

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::ptr;
use std::sync::OnceLock;

use libloading::Library;

// HSA type definitions
pub type HsaStatus = c_int;
pub type HsaAgent = u64;
pub type HsaSignal = u64; // hsa_signal_t is a struct with handle
pub type HsaQueue = *mut c_void;
pub type HsaExecutable = u64;
pub type HsaCodeObjectReader = u64;
pub type HsaRegion = u64;

// HSA status codes
pub const HSA_STATUS_SUCCESS: HsaStatus = 0;
pub const HSA_STATUS_INFO_BREAK: HsaStatus = 1;

// HSA device types
pub const HSA_DEVICE_TYPE_CPU: u32 = 0;
pub const HSA_DEVICE_TYPE_GPU: u32 = 1;

// HSA agent info attributes
pub const HSA_AGENT_INFO_DEVICE: u32 = 1;
pub const HSA_AGENT_INFO_NAME: u32 = 0;
pub const HSA_AGENT_INFO_QUEUE_MAX_SIZE: u32 = 4;

// HSA region info
pub const HSA_REGION_INFO_SEGMENT: u32 = 0;
pub const HSA_REGION_INFO_GLOBAL_FLAGS: u32 = 1;
pub const HSA_REGION_INFO_SIZE: u32 = 2;
pub const HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED: u32 = 5;

// HSA segment types
pub const HSA_REGION_SEGMENT_GLOBAL: u32 = 0;
pub const HSA_REGION_GLOBAL_FLAG_KERNARG: u32 = 1;
pub const HSA_REGION_GLOBAL_FLAG_FINE_GRAINED: u32 = 2;
pub const HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED: u32 = 4;

// HSA executable state
pub const HSA_EXECUTABLE_STATE_UNFROZEN: u32 = 0;
pub const HSA_EXECUTABLE_STATE_FROZEN: u32 = 1;

// HSA kernel dispatch packet
#[repr(C)]
pub struct HsaKernelDispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: *mut c_void,
    pub reserved2: u64,
    pub completion_signal: HsaSignal,
}

// Callback types
type HsaIterateAgentsCallback = unsafe extern "C" fn(HsaAgent, *mut c_void) -> HsaStatus;
type HsaIterateRegionsCallback = unsafe extern "C" fn(HsaRegion, *mut c_void) -> HsaStatus;

// Function pointer types
type HsaInitFn = unsafe extern "C" fn() -> HsaStatus;
type HsaShutDownFn = unsafe extern "C" fn() -> HsaStatus;
type HsaIterateAgentsFn =
    unsafe extern "C" fn(HsaIterateAgentsCallback, *mut c_void) -> HsaStatus;
type HsaAgentGetInfoFn =
    unsafe extern "C" fn(HsaAgent, u32, *mut c_void) -> HsaStatus;
type HsaAgentIterateRegionsFn =
    unsafe extern "C" fn(HsaAgent, HsaIterateRegionsCallback, *mut c_void) -> HsaStatus;
type HsaRegionGetInfoFn =
    unsafe extern "C" fn(HsaRegion, u32, *mut c_void) -> HsaStatus;

type HsaQueueCreateFn = unsafe extern "C" fn(
    HsaAgent,
    u32,        // size
    u32,        // type
    *mut c_void, // callback
    *mut c_void, // data
    u32,        // private_segment_size
    u32,        // group_segment_size
    *mut HsaQueue,
) -> HsaStatus;
type HsaQueueDestroyFn = unsafe extern "C" fn(HsaQueue) -> HsaStatus;
type HsaQueueLoadWriteIndexRelaxedFn = unsafe extern "C" fn(HsaQueue) -> u64;
type HsaQueueStoreWriteIndexRelaxedFn = unsafe extern "C" fn(HsaQueue, u64);
type HsaQueueAddWriteIndexRelaxedFn = unsafe extern "C" fn(HsaQueue, u64) -> u64;

type HsaSignalCreateFn =
    unsafe extern "C" fn(i64, u32, *const HsaAgent, *mut HsaSignal) -> HsaStatus;
type HsaSignalDestroyFn = unsafe extern "C" fn(HsaSignal) -> HsaStatus;
type HsaSignalStoreRelaxedFn = unsafe extern "C" fn(HsaSignal, i64);
type HsaSignalWaitAcquireFn =
    unsafe extern "C" fn(HsaSignal, u32, i64, u64, u32) -> i64;

type HsaMemoryAllocateFn =
    unsafe extern "C" fn(HsaRegion, usize, *mut *mut c_void) -> HsaStatus;
type HsaMemoryFreeFn = unsafe extern "C" fn(*mut c_void) -> HsaStatus;
type HsaMemoryCopyFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> HsaStatus;

type HsaCodeObjectReaderCreateFromMemoryFn =
    unsafe extern "C" fn(*const c_void, usize, *mut HsaCodeObjectReader) -> HsaStatus;
type HsaCodeObjectReaderDestroyFn = unsafe extern "C" fn(HsaCodeObjectReader) -> HsaStatus;

type HsaExecutableCreateAltFn = unsafe extern "C" fn(
    u32,        // profile
    u32,        // default_float_rounding_mode
    *const c_char, // options
    *mut HsaExecutable,
) -> HsaStatus;
type HsaExecutableDestroyFn = unsafe extern "C" fn(HsaExecutable) -> HsaStatus;
type HsaExecutableLoadAgentCodeObjectFn = unsafe extern "C" fn(
    HsaExecutable,
    HsaAgent,
    HsaCodeObjectReader,
    *const c_char,
    *mut u64,
) -> HsaStatus;
type HsaExecutableFreezeFn =
    unsafe extern "C" fn(HsaExecutable, *const c_char) -> HsaStatus;
type HsaExecutableGetSymbolByNameFn = unsafe extern "C" fn(
    HsaExecutable,
    *const c_char,
    *const HsaAgent,
    *mut u64,
) -> HsaStatus;
type HsaExecutableSymbolGetInfoFn =
    unsafe extern "C" fn(u64, u32, *mut c_void) -> HsaStatus;

type HsaStatusStringFn =
    unsafe extern "C" fn(HsaStatus, *mut *const c_char) -> HsaStatus;

/// HSA Runtime library function table.
pub struct HsaLib {
    #[allow(dead_code)]
    lib: Library,

    // Initialization
    pub hsa_init: HsaInitFn,
    pub hsa_shut_down: HsaShutDownFn,

    // Agent enumeration
    pub hsa_iterate_agents: HsaIterateAgentsFn,
    pub hsa_agent_get_info: HsaAgentGetInfoFn,
    pub hsa_agent_iterate_regions: HsaAgentIterateRegionsFn,
    pub hsa_region_get_info: HsaRegionGetInfoFn,

    // Queue management
    pub hsa_queue_create: HsaQueueCreateFn,
    pub hsa_queue_destroy: HsaQueueDestroyFn,
    pub hsa_queue_load_write_index_relaxed: HsaQueueLoadWriteIndexRelaxedFn,
    pub hsa_queue_store_write_index_relaxed: HsaQueueStoreWriteIndexRelaxedFn,
    pub hsa_queue_add_write_index_relaxed: HsaQueueAddWriteIndexRelaxedFn,

    // Signal management
    pub hsa_signal_create: HsaSignalCreateFn,
    pub hsa_signal_destroy: HsaSignalDestroyFn,
    pub hsa_signal_store_relaxed: HsaSignalStoreRelaxedFn,
    pub hsa_signal_wait_acquire: HsaSignalWaitAcquireFn,

    // Memory management
    pub hsa_memory_allocate: HsaMemoryAllocateFn,
    pub hsa_memory_free: HsaMemoryFreeFn,
    pub hsa_memory_copy: HsaMemoryCopyFn,

    // Code object loading
    pub hsa_code_object_reader_create_from_memory: HsaCodeObjectReaderCreateFromMemoryFn,
    pub hsa_code_object_reader_destroy: HsaCodeObjectReaderDestroyFn,

    // Executable management
    pub hsa_executable_create_alt: HsaExecutableCreateAltFn,
    pub hsa_executable_destroy: HsaExecutableDestroyFn,
    pub hsa_executable_load_agent_code_object: HsaExecutableLoadAgentCodeObjectFn,
    pub hsa_executable_freeze: HsaExecutableFreezeFn,
    pub hsa_executable_get_symbol_by_name: HsaExecutableGetSymbolByNameFn,
    pub hsa_executable_symbol_get_info: HsaExecutableSymbolGetInfoFn,

    // Error handling
    pub hsa_status_string: HsaStatusStringFn,
}

// Safety: HsaLib contains function pointers from a loaded library.
// The library is loaded once and lives for the entire program lifetime.
// Function pointers are thread-safe as they're immutable after initialization.
unsafe impl Send for HsaLib {}
unsafe impl Sync for HsaLib {}

impl HsaLib {
    /// Try to load the HSA Runtime library.
    fn load() -> Result<Self, String> {
        // Try common library paths for HSA Runtime
        // HSA Runtime is part of the AMD GPU driver, not ROCm toolkit
        let lib_names = [
            "libhsa-runtime64.so",
            "libhsa-runtime64.so.1",
            "/opt/rocm/lib/libhsa-runtime64.so",
            "/opt/rocm/lib64/libhsa-runtime64.so",
            // AMD driver installs HSA runtime here
            "/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so",
            "/usr/lib64/libhsa-runtime64.so",
        ];

        let lib = lib_names
            .iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| {
                "Failed to load HSA Runtime library (libhsa-runtime64.so). \
                 This library is part of the AMD GPU driver."
                    .to_string()
            })?;

        // Load all function pointers
        let hsa_init: HsaInitFn = unsafe {
            *lib.get::<HsaInitFn>(b"hsa_init\0")
                .map_err(|e| format!("hsa_init: {e}"))?
        };
        let hsa_shut_down: HsaShutDownFn = unsafe {
            *lib.get::<HsaShutDownFn>(b"hsa_shut_down\0")
                .map_err(|e| format!("hsa_shut_down: {e}"))?
        };
        let hsa_iterate_agents: HsaIterateAgentsFn = unsafe {
            *lib.get::<HsaIterateAgentsFn>(b"hsa_iterate_agents\0")
                .map_err(|e| format!("hsa_iterate_agents: {e}"))?
        };
        let hsa_agent_get_info: HsaAgentGetInfoFn = unsafe {
            *lib.get::<HsaAgentGetInfoFn>(b"hsa_agent_get_info\0")
                .map_err(|e| format!("hsa_agent_get_info: {e}"))?
        };
        let hsa_agent_iterate_regions: HsaAgentIterateRegionsFn = unsafe {
            *lib.get::<HsaAgentIterateRegionsFn>(b"hsa_agent_iterate_regions\0")
                .map_err(|e| format!("hsa_agent_iterate_regions: {e}"))?
        };
        let hsa_region_get_info: HsaRegionGetInfoFn = unsafe {
            *lib.get::<HsaRegionGetInfoFn>(b"hsa_region_get_info\0")
                .map_err(|e| format!("hsa_region_get_info: {e}"))?
        };

        let hsa_queue_create: HsaQueueCreateFn = unsafe {
            *lib.get::<HsaQueueCreateFn>(b"hsa_queue_create\0")
                .map_err(|e| format!("hsa_queue_create: {e}"))?
        };
        let hsa_queue_destroy: HsaQueueDestroyFn = unsafe {
            *lib.get::<HsaQueueDestroyFn>(b"hsa_queue_destroy\0")
                .map_err(|e| format!("hsa_queue_destroy: {e}"))?
        };
        let hsa_queue_load_write_index_relaxed: HsaQueueLoadWriteIndexRelaxedFn = unsafe {
            *lib.get::<HsaQueueLoadWriteIndexRelaxedFn>(b"hsa_queue_load_write_index_relaxed\0")
                .map_err(|e| format!("hsa_queue_load_write_index_relaxed: {e}"))?
        };
        let hsa_queue_store_write_index_relaxed: HsaQueueStoreWriteIndexRelaxedFn = unsafe {
            *lib.get::<HsaQueueStoreWriteIndexRelaxedFn>(b"hsa_queue_store_write_index_relaxed\0")
                .map_err(|e| format!("hsa_queue_store_write_index_relaxed: {e}"))?
        };
        let hsa_queue_add_write_index_relaxed: HsaQueueAddWriteIndexRelaxedFn = unsafe {
            *lib.get::<HsaQueueAddWriteIndexRelaxedFn>(b"hsa_queue_add_write_index_relaxed\0")
                .map_err(|e| format!("hsa_queue_add_write_index_relaxed: {e}"))?
        };

        let hsa_signal_create: HsaSignalCreateFn = unsafe {
            *lib.get::<HsaSignalCreateFn>(b"hsa_signal_create\0")
                .map_err(|e| format!("hsa_signal_create: {e}"))?
        };
        let hsa_signal_destroy: HsaSignalDestroyFn = unsafe {
            *lib.get::<HsaSignalDestroyFn>(b"hsa_signal_destroy\0")
                .map_err(|e| format!("hsa_signal_destroy: {e}"))?
        };
        let hsa_signal_store_relaxed: HsaSignalStoreRelaxedFn = unsafe {
            *lib.get::<HsaSignalStoreRelaxedFn>(b"hsa_signal_store_relaxed\0")
                .map_err(|e| format!("hsa_signal_store_relaxed: {e}"))?
        };
        let hsa_signal_wait_acquire: HsaSignalWaitAcquireFn = unsafe {
            *lib.get::<HsaSignalWaitAcquireFn>(b"hsa_signal_wait_acquire\0")
                .map_err(|e| format!("hsa_signal_wait_acquire: {e}"))?
        };

        let hsa_memory_allocate: HsaMemoryAllocateFn = unsafe {
            *lib.get::<HsaMemoryAllocateFn>(b"hsa_memory_allocate\0")
                .map_err(|e| format!("hsa_memory_allocate: {e}"))?
        };
        let hsa_memory_free: HsaMemoryFreeFn = unsafe {
            *lib.get::<HsaMemoryFreeFn>(b"hsa_memory_free\0")
                .map_err(|e| format!("hsa_memory_free: {e}"))?
        };
        let hsa_memory_copy: HsaMemoryCopyFn = unsafe {
            *lib.get::<HsaMemoryCopyFn>(b"hsa_memory_copy\0")
                .map_err(|e| format!("hsa_memory_copy: {e}"))?
        };

        let hsa_code_object_reader_create_from_memory: HsaCodeObjectReaderCreateFromMemoryFn =
            unsafe {
                *lib.get::<HsaCodeObjectReaderCreateFromMemoryFn>(
                    b"hsa_code_object_reader_create_from_memory\0",
                )
                .map_err(|e| format!("hsa_code_object_reader_create_from_memory: {e}"))?
            };
        let hsa_code_object_reader_destroy: HsaCodeObjectReaderDestroyFn = unsafe {
            *lib.get::<HsaCodeObjectReaderDestroyFn>(b"hsa_code_object_reader_destroy\0")
                .map_err(|e| format!("hsa_code_object_reader_destroy: {e}"))?
        };

        let hsa_executable_create_alt: HsaExecutableCreateAltFn = unsafe {
            *lib.get::<HsaExecutableCreateAltFn>(b"hsa_executable_create_alt\0")
                .map_err(|e| format!("hsa_executable_create_alt: {e}"))?
        };
        let hsa_executable_destroy: HsaExecutableDestroyFn = unsafe {
            *lib.get::<HsaExecutableDestroyFn>(b"hsa_executable_destroy\0")
                .map_err(|e| format!("hsa_executable_destroy: {e}"))?
        };
        let hsa_executable_load_agent_code_object: HsaExecutableLoadAgentCodeObjectFn = unsafe {
            *lib.get::<HsaExecutableLoadAgentCodeObjectFn>(
                b"hsa_executable_load_agent_code_object\0",
            )
            .map_err(|e| format!("hsa_executable_load_agent_code_object: {e}"))?
        };
        let hsa_executable_freeze: HsaExecutableFreezeFn = unsafe {
            *lib.get::<HsaExecutableFreezeFn>(b"hsa_executable_freeze\0")
                .map_err(|e| format!("hsa_executable_freeze: {e}"))?
        };
        let hsa_executable_get_symbol_by_name: HsaExecutableGetSymbolByNameFn = unsafe {
            *lib.get::<HsaExecutableGetSymbolByNameFn>(b"hsa_executable_get_symbol_by_name\0")
                .map_err(|e| format!("hsa_executable_get_symbol_by_name: {e}"))?
        };
        let hsa_executable_symbol_get_info: HsaExecutableSymbolGetInfoFn = unsafe {
            *lib.get::<HsaExecutableSymbolGetInfoFn>(b"hsa_executable_symbol_get_info\0")
                .map_err(|e| format!("hsa_executable_symbol_get_info: {e}"))?
        };

        let hsa_status_string: HsaStatusStringFn = unsafe {
            *lib.get::<HsaStatusStringFn>(b"hsa_status_string\0")
                .map_err(|e| format!("hsa_status_string: {e}"))?
        };

        Ok(Self {
            lib,
            hsa_init,
            hsa_shut_down,
            hsa_iterate_agents,
            hsa_agent_get_info,
            hsa_agent_iterate_regions,
            hsa_region_get_info,
            hsa_queue_create,
            hsa_queue_destroy,
            hsa_queue_load_write_index_relaxed,
            hsa_queue_store_write_index_relaxed,
            hsa_queue_add_write_index_relaxed,
            hsa_signal_create,
            hsa_signal_destroy,
            hsa_signal_store_relaxed,
            hsa_signal_wait_acquire,
            hsa_memory_allocate,
            hsa_memory_free,
            hsa_memory_copy,
            hsa_code_object_reader_create_from_memory,
            hsa_code_object_reader_destroy,
            hsa_executable_create_alt,
            hsa_executable_destroy,
            hsa_executable_load_agent_code_object,
            hsa_executable_freeze,
            hsa_executable_get_symbol_by_name,
            hsa_executable_symbol_get_info,
            hsa_status_string,
        })
    }
}

/// Global HSA library instance.
static HSA_LIB: OnceLock<Result<HsaLib, String>> = OnceLock::new();

/// HSA Runtime initialization state.
static HSA_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

/// Get the global HSA library instance.
pub fn get_hsa_lib() -> Result<&'static HsaLib, &'static str> {
    HSA_LIB
        .get_or_init(HsaLib::load)
        .as_ref()
        .map_err(|e| e.as_str())
}

/// Initialize HSA Runtime (call once at startup).
pub fn hsa_init() -> Result<(), &'static str> {
    HSA_INITIALIZED
        .get_or_init(|| {
            let lib = get_hsa_lib().map_err(|e| e.to_string())?;
            let status = unsafe { (lib.hsa_init)() };
            if status == HSA_STATUS_SUCCESS {
                Ok(())
            } else {
                Err(format!("hsa_init failed with status {}", status))
            }
        })
        .as_ref()
        .map(|_| ())
        .map_err(|e| e.as_str())
}

/// Check if HSA Runtime is available on this system.
pub fn is_hsa_available() -> bool {
    get_hsa_lib().is_ok()
}

/// Get error string from HSA status code.
pub fn get_error_string(status: HsaStatus) -> String {
    if let Ok(lib) = get_hsa_lib() {
        unsafe {
            let mut msg: *const c_char = ptr::null();
            if (lib.hsa_status_string)(status, &mut msg) == HSA_STATUS_SUCCESS && !msg.is_null() {
                CStr::from_ptr(msg).to_string_lossy().into_owned()
            } else {
                format!("HSA error code: {status}")
            }
        }
    } else {
        format!("HSA error code: {status}")
    }
}

/// GPU Agent information.
#[derive(Debug, Clone)]
pub struct GpuAgent {
    pub handle: HsaAgent,
    pub name: String,
    pub kernarg_region: HsaRegion,
    pub fine_grained_region: HsaRegion,
    pub coarse_grained_region: HsaRegion,
}

/// Find all GPU agents.
pub fn find_gpu_agents() -> Result<Vec<GpuAgent>, String> {
    hsa_init().map_err(|e| e.to_string())?;
    let lib = get_hsa_lib().map_err(|e| e.to_string())?;

    let mut agents: Vec<GpuAgent> = Vec::new();

    unsafe extern "C" fn agent_callback(agent: HsaAgent, data: *mut c_void) -> HsaStatus {
        let agents = &mut *(data as *mut Vec<HsaAgent>);

        // Check if this is a GPU
        let lib = match get_hsa_lib() {
            Ok(lib) => lib,
            Err(_) => return HSA_STATUS_SUCCESS,
        };

        let mut device_type: u32 = 0;
        let status = (lib.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_DEVICE,
            &mut device_type as *mut _ as *mut c_void,
        );

        if status == HSA_STATUS_SUCCESS && device_type == HSA_DEVICE_TYPE_GPU {
            agents.push(agent);
        }

        HSA_STATUS_SUCCESS
    }

    let mut raw_agents: Vec<HsaAgent> = Vec::new();
    let status = unsafe {
        (lib.hsa_iterate_agents)(agent_callback, &mut raw_agents as *mut _ as *mut c_void)
    };

    if status != HSA_STATUS_SUCCESS {
        return Err(format!("hsa_iterate_agents failed: {}", get_error_string(status)));
    }

    // Get detailed info for each GPU agent
    for handle in raw_agents {
        let mut name_buf = [0u8; 64];
        unsafe {
            (lib.hsa_agent_get_info)(
                handle,
                HSA_AGENT_INFO_NAME,
                name_buf.as_mut_ptr() as *mut c_void,
            );
        }
        let name = CStr::from_bytes_until_nul(&name_buf)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "Unknown GPU".to_string());

        // Find memory regions
        let mut kernarg_region: HsaRegion = 0;
        let mut fine_grained_region: HsaRegion = 0;
        let mut coarse_grained_region: HsaRegion = 0;

        #[repr(C)]
        struct RegionData {
            kernarg: HsaRegion,
            fine_grained: HsaRegion,
            coarse_grained: HsaRegion,
        }

        unsafe extern "C" fn region_callback(region: HsaRegion, data: *mut c_void) -> HsaStatus {
            let lib = match get_hsa_lib() {
                Ok(lib) => lib,
                Err(_) => return HSA_STATUS_SUCCESS,
            };
            let regions = &mut *(data as *mut RegionData);

            let mut segment: u32 = 0;
            (lib.hsa_region_get_info)(
                region,
                HSA_REGION_INFO_SEGMENT,
                &mut segment as *mut _ as *mut c_void,
            );

            if segment == HSA_REGION_SEGMENT_GLOBAL {
                let mut flags: u32 = 0;
                (lib.hsa_region_get_info)(
                    region,
                    HSA_REGION_INFO_GLOBAL_FLAGS,
                    &mut flags as *mut _ as *mut c_void,
                );

                let mut runtime_alloc: u32 = 0;
                (lib.hsa_region_get_info)(
                    region,
                    HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                    &mut runtime_alloc as *mut _ as *mut c_void,
                );

                if runtime_alloc != 0 {
                    if flags & HSA_REGION_GLOBAL_FLAG_KERNARG != 0 {
                        regions.kernarg = region;
                    }
                    if flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED != 0 {
                        regions.fine_grained = region;
                    }
                    if flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED != 0 {
                        regions.coarse_grained = region;
                    }
                }
            }

            HSA_STATUS_SUCCESS
        }

        let mut region_data = RegionData {
            kernarg: 0,
            fine_grained: 0,
            coarse_grained: 0,
        };

        unsafe {
            (lib.hsa_agent_iterate_regions)(
                handle,
                region_callback,
                &mut region_data as *mut _ as *mut c_void,
            );
        }

        kernarg_region = region_data.kernarg;
        fine_grained_region = region_data.fine_grained;
        coarse_grained_region = region_data.coarse_grained;

        agents.push(GpuAgent {
            handle,
            name,
            kernarg_region,
            fine_grained_region,
            coarse_grained_region,
        });
    }

    Ok(agents)
}

/// Get the number of available GPU agents.
pub fn get_gpu_count() -> Result<usize, String> {
    find_gpu_agents().map(|agents| agents.len())
}
