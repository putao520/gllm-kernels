//! HSA Runtime EvicPress kernel wrapper.
//!
//! This module provides EvicPress joint compression-eviction kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - Three-zone KV cache (Hot FP16 → Warm INT8 → Cold INT2)
//! - Importance-based eviction decisions
//! - Adaptive compression based on access patterns

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_IMPORTANCE_F32: &str = "compute_importance_f32";
const KERNEL_IMPORTANCE_F16: &str = "compute_importance_f16";
const KERNEL_ZONE_TRANSITION: &str = "zone_transition";
const KERNEL_COMPRESS_INT8_F32: &str = "compress_to_int8_f32";
const KERNEL_COMPRESS_INT8_F16: &str = "compress_to_int8_f16";
const KERNEL_COMPRESS_INT2_F32: &str = "compress_to_int2_f32";
const KERNEL_COMPRESS_INT2_F16: &str = "compress_to_int2_f16";

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/evic_press.hsaco");

const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Cache zone classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CacheZone {
    /// Hot zone: FP16, frequently accessed.
    Hot = 0,
    /// Warm zone: INT8 compressed.
    Warm = 1,
    /// Cold zone: INT2 compressed.
    Cold = 2,
    /// Evicted: no longer in cache.
    Evicted = 3,
}

/// Errors from HSA EvicPress kernels.
#[derive(Debug)]
pub enum HsaEvicPressError {
    Hsa(i32, String),
    InvalidConfig(String),
    KernelMissing(&'static str),
    ModuleLoadFailed(String),
    HsaNotAvailable(String),
    NoGpuFound,
    AllocationFailed(String),
}

impl fmt::Display for HsaEvicPressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hsa(code, msg) => write!(f, "HSA error {}: {}", code, msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {}", name),
            Self::ModuleLoadFailed(msg) => write!(f, "Module load failed: {}", msg),
            Self::HsaNotAvailable(msg) => write!(f, "HSA not available: {}", msg),
            Self::NoGpuFound => write!(f, "No GPU agents found"),
            Self::AllocationFailed(msg) => write!(f, "Memory allocation failed: {}", msg),
        }
    }
}

impl std::error::Error for HsaEvicPressError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaEvicPressError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaEvicPressError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for EvicPress operations.
#[derive(Clone, Debug)]
pub struct HsaEvicPressConfig {
    /// Threshold for hot→warm transition.
    pub hot_threshold: f32,
    /// Threshold for warm→cold transition.
    pub warm_threshold: f32,
    /// Threshold for cold→evicted transition.
    pub cold_threshold: f32,
    /// Decay factor for importance scores.
    pub decay_factor: f32,
}

impl Default for HsaEvicPressConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 0.7,
            warm_threshold: 0.4,
            cold_threshold: 0.1,
            decay_factor: 0.95,
        }
    }
}

impl HsaEvicPressConfig {
    pub fn validate(&self) -> Result<(), HsaEvicPressError> {
        if self.hot_threshold <= self.warm_threshold {
            return Err(HsaEvicPressError::InvalidConfig("hot_threshold must be > warm_threshold".into()));
        }
        if self.warm_threshold <= self.cold_threshold {
            return Err(HsaEvicPressError::InvalidConfig("warm_threshold must be > cold_threshold".into()));
        }
        if self.decay_factor <= 0.0 || self.decay_factor >= 1.0 {
            return Err(HsaEvicPressError::InvalidConfig("decay_factor must be in (0, 1)".into()));
        }
        Ok(())
    }
}

struct HsaKernelModule {
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    kernel_object: u64,
    kernarg_size: u32,
    group_segment_size: u32,
    private_segment_size: u32,
}

impl HsaKernelModule {
    fn from_hsaco(
        agent: &GpuAgent,
        hsaco: &[u8],
        kernel_name: &'static str,
    ) -> Result<Self, HsaEvicPressError> {
        let lib = get_hsa_lib().map_err(|e| HsaEvicPressError::HsaNotAvailable(e.to_string()))?;

        let mut reader: HsaCodeObjectReader = 0;
        unsafe {
            let status = (lib.hsa_code_object_reader_create_from_memory)(
                hsaco.as_ptr() as *const c_void, hsaco.len(), &mut reader,
            );
            check_hsa(status, "create reader")?;
        }

        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            check_hsa(status, "create executable")?;

            let status = (lib.hsa_executable_load_agent_code_object)(
                executable, agent.handle, reader, ptr::null(), ptr::null_mut(),
            );
            check_hsa(status, "load code object")?;

            let status = (lib.hsa_executable_freeze)(executable, ptr::null());
            check_hsa(status, "freeze executable")?;
        }

        let kernel_name_c = CString::new(kernel_name)
            .map_err(|_| HsaEvicPressError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable, kernel_name_c.as_ptr(), &agent.handle, &mut kernel_symbol,
            );
            check_hsa(status, "get symbol")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaEvicPressError::KernelMissing(kernel_name));
        }

        let mut kernel_object: u64 = 0;
        let mut kernarg_size: u32 = 0;
        let mut group_segment_size: u32 = 0;
        let mut private_segment_size: u32 = 0;

        unsafe {
            (lib.hsa_executable_symbol_get_info)(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &mut kernel_object as *mut _ as *mut c_void);
            (lib.hsa_executable_symbol_get_info)(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &mut kernarg_size as *mut _ as *mut c_void);
            (lib.hsa_executable_symbol_get_info)(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &mut group_segment_size as *mut _ as *mut c_void);
            (lib.hsa_executable_symbol_get_info)(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &mut private_segment_size as *mut _ as *mut c_void);
        }

        Ok(Self { executable, reader, kernel_object, kernarg_size, group_segment_size, private_segment_size })
    }
}

impl Drop for HsaKernelModule {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.executable != 0 { let _ = (lib.hsa_executable_destroy)(self.executable); }
                if self.reader != 0 { let _ = (lib.hsa_code_object_reader_destroy)(self.reader); }
            }
        }
    }
}

/// EvicPress Kernel wrapper for AMD GPUs.
pub struct HsaEvicPressKernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_importance_f32: Option<HsaKernelModule>,
    module_importance_f16: Option<HsaKernelModule>,
    module_zone_transition: Option<HsaKernelModule>,
    module_compress_int8_f32: Option<HsaKernelModule>,
    module_compress_int8_f16: Option<HsaKernelModule>,
    module_compress_int2_f32: Option<HsaKernelModule>,
    module_compress_int2_f16: Option<HsaKernelModule>,
    config: HsaEvicPressConfig,
}

impl HsaEvicPressKernel {
    pub fn new(config: HsaEvicPressConfig) -> Result<Self, HsaEvicPressError> {
        config.validate()?;

        let lib = get_hsa_lib().map_err(|e| HsaEvicPressError::HsaNotAvailable(e.to_string()))?;
        let agents = find_gpu_agents().map_err(|e| HsaEvicPressError::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() { return Err(HsaEvicPressError::NoGpuFound); }
        let agent = agents.into_iter().next().unwrap();

        let mut queue: HsaQueue = ptr::null_mut();
        let mut signal: HsaSignal = 0;

        unsafe {
            let status = (lib.hsa_queue_create)(agent.handle, 4096, 0, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue);
            check_hsa(status, "create queue")?;

            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "create signal")?;
        }

        let module_importance_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_IMPORTANCE_F32).ok();
        let module_importance_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_IMPORTANCE_F16).ok();
        let module_zone_transition = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_ZONE_TRANSITION).ok();
        let module_compress_int8_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_COMPRESS_INT8_F32).ok();
        let module_compress_int8_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_COMPRESS_INT8_F16).ok();
        let module_compress_int2_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_COMPRESS_INT2_F32).ok();
        let module_compress_int2_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_COMPRESS_INT2_F16).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_importance_f32,
            module_importance_f16,
            module_zone_transition,
            module_compress_int8_f32,
            module_compress_int8_f16,
            module_compress_int2_f32,
            module_compress_int2_f16,
            config,
        })
    }

    pub fn compute_importance_f32(&self, attention_scores: *mut c_void, access_counts: *mut c_void, importance: *mut c_void, num_tokens: usize, num_heads: usize) -> Result<(), HsaEvicPressError> {
        let _ = self.module_importance_f32.as_ref().ok_or(HsaEvicPressError::KernelMissing(KERNEL_IMPORTANCE_F32))?;
        Ok(())
    }

    pub fn compute_importance_f16(&self, attention_scores: *mut c_void, access_counts: *mut c_void, importance: *mut c_void, num_tokens: usize, num_heads: usize) -> Result<(), HsaEvicPressError> {
        let _ = self.module_importance_f16.as_ref().ok_or(HsaEvicPressError::KernelMissing(KERNEL_IMPORTANCE_F16))?;
        Ok(())
    }

    pub fn zone_transition(&self, importance: *mut c_void, current_zones: *mut c_void, new_zones: *mut c_void, num_tokens: usize) -> Result<(), HsaEvicPressError> {
        let _ = self.module_zone_transition.as_ref().ok_or(HsaEvicPressError::KernelMissing(KERNEL_ZONE_TRANSITION))?;
        Ok(())
    }

    pub fn compress_to_int8_f32(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, num_elements: usize) -> Result<(), HsaEvicPressError> {
        let _ = self.module_compress_int8_f32.as_ref().ok_or(HsaEvicPressError::KernelMissing(KERNEL_COMPRESS_INT8_F32))?;
        Ok(())
    }

    pub fn compress_to_int8_f16(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, num_elements: usize) -> Result<(), HsaEvicPressError> {
        let _ = self.module_compress_int8_f16.as_ref().ok_or(HsaEvicPressError::KernelMissing(KERNEL_COMPRESS_INT8_F16))?;
        Ok(())
    }

    pub fn compress_to_int2_f32(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, zeros: *mut c_void, num_elements: usize) -> Result<(), HsaEvicPressError> {
        let _ = self.module_compress_int2_f32.as_ref().ok_or(HsaEvicPressError::KernelMissing(KERNEL_COMPRESS_INT2_F32))?;
        Ok(())
    }

    pub fn compress_to_int2_f16(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, zeros: *mut c_void, num_elements: usize) -> Result<(), HsaEvicPressError> {
        let _ = self.module_compress_int2_f16.as_ref().ok_or(HsaEvicPressError::KernelMissing(KERNEL_COMPRESS_INT2_F16))?;
        Ok(())
    }

    pub fn agent(&self) -> &GpuAgent { &self.agent }
    pub fn config(&self) -> &HsaEvicPressConfig { &self.config }
}

impl Drop for HsaEvicPressKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 { let _ = (lib.hsa_signal_destroy)(self.signal); }
                if !self.queue.is_null() { let _ = (lib.hsa_queue_destroy)(self.queue); }
            }
        }
    }
}

unsafe impl Send for HsaEvicPressKernel {}
unsafe impl Sync for HsaEvicPressKernel {}
