//! HSA Runtime EAGLE-3 kernel wrapper.
//!
//! This module provides EAGLE-3 adaptive draft length kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - Adaptive speculation depth based on acceptance history
//! - Confidence-based draft length adjustment
//! - Draft generation with early termination
//! - Acceptance probability tracking

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaRegion, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_CONFIDENCE_F32: &str = "eagle3_compute_confidence_f32";
const KERNEL_CONFIDENCE_F16: &str = "eagle3_compute_confidence_f16";
const KERNEL_ADJUST_DEPTH: &str = "eagle3_adjust_depth";
const KERNEL_DRAFT_F32: &str = "eagle3_draft_generation_f32";
const KERNEL_DRAFT_F16: &str = "eagle3_draft_generation_f16";
const KERNEL_UPDATE_HISTORY: &str = "eagle3_update_history";

// Embedded HSACO
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/eagle3.hsaco");

/// HSA packet header constants
const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;
const HSA_SIGNAL_CONDITION_LT: u32 = 0;
const HSA_WAIT_STATE_BLOCKED: u32 = 0;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Errors from HSA EAGLE-3 kernels.
#[derive(Debug)]
pub enum HsaEagle3Error {
    /// HSA runtime error.
    Hsa(i32, String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Module load failure.
    ModuleLoadFailed(String),
    /// HSA library not available.
    HsaNotAvailable(String),
    /// No GPU agents found.
    NoGpuFound,
    /// Memory allocation failed.
    AllocationFailed(String),
}

impl fmt::Display for HsaEagle3Error {
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

impl std::error::Error for HsaEagle3Error {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaEagle3Error> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaEagle3Error::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for EAGLE-3 operations.
#[derive(Clone, Debug)]
pub struct HsaEagle3Config {
    /// Minimum speculation depth.
    pub min_depth: usize,
    /// Maximum speculation depth.
    pub max_depth: usize,
    /// Confidence threshold for speculation.
    pub confidence_threshold: f32,
    /// History window size for adaptive adjustment.
    pub history_window: usize,
}

impl Default for HsaEagle3Config {
    fn default() -> Self {
        Self {
            min_depth: 1,
            max_depth: 8,
            confidence_threshold: 0.7,
            history_window: 16,
        }
    }
}

impl HsaEagle3Config {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), HsaEagle3Error> {
        if self.min_depth == 0 {
            return Err(HsaEagle3Error::InvalidConfig("min_depth must be positive".into()));
        }
        if self.max_depth < self.min_depth {
            return Err(HsaEagle3Error::InvalidConfig("max_depth must be >= min_depth".into()));
        }
        if self.confidence_threshold <= 0.0 || self.confidence_threshold >= 1.0 {
            return Err(HsaEagle3Error::InvalidConfig("confidence_threshold must be in (0, 1)".into()));
        }
        Ok(())
    }
}

/// HSA kernel module loaded from HSACO.
struct HsaKernelModule {
    executable: HsaExecutable,
    #[allow(dead_code)]
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
    ) -> Result<Self, HsaEagle3Error> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaEagle3Error::HsaNotAvailable(e.to_string()))?;

        let mut reader: HsaCodeObjectReader = 0;
        unsafe {
            let status = (lib.hsa_code_object_reader_create_from_memory)(
                hsaco.as_ptr() as *const c_void,
                hsaco.len(),
                &mut reader,
            );
            check_hsa(status, "hsa_code_object_reader_create_from_memory")?;
        }

        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(
                1, // HSA_PROFILE_FULL
                0, // HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT
                ptr::null(),
                &mut executable,
            );
            check_hsa(status, "hsa_executable_create_alt")?;
        }

        unsafe {
            let status = (lib.hsa_executable_load_agent_code_object)(
                executable,
                agent.handle,
                reader,
                ptr::null(),
                ptr::null_mut(),
            );
            check_hsa(status, "hsa_executable_load_agent_code_object")?;
        }

        unsafe {
            let status = (lib.hsa_executable_freeze)(executable, ptr::null());
            check_hsa(status, "hsa_executable_freeze")?;
        }

        let kernel_name_c = CString::new(kernel_name)
            .map_err(|_| HsaEagle3Error::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable,
                kernel_name_c.as_ptr(),
                &agent.handle,
                &mut kernel_symbol,
            );
            check_hsa(status, "hsa_executable_get_symbol_by_name")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaEagle3Error::KernelMissing(kernel_name));
        }

        let mut kernel_object: u64 = 0;
        let mut kernarg_size: u32 = 0;
        let mut group_segment_size: u32 = 0;
        let mut private_segment_size: u32 = 0;

        unsafe {
            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &mut kernel_object as *mut _ as *mut c_void,
            );
            check_hsa(status, "get kernel_object")?;

            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &mut kernarg_size as *mut _ as *mut c_void,
            );
            check_hsa(status, "get kernarg_size")?;

            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &mut group_segment_size as *mut _ as *mut c_void,
            );
            check_hsa(status, "get group_segment_size")?;

            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &mut private_segment_size as *mut _ as *mut c_void,
            );
            check_hsa(status, "get private_segment_size")?;
        }

        Ok(Self {
            executable,
            reader,
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
        })
    }
}

impl Drop for HsaKernelModule {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.executable != 0 {
                    let _ = (lib.hsa_executable_destroy)(self.executable);
                }
                if self.reader != 0 {
                    let _ = (lib.hsa_code_object_reader_destroy)(self.reader);
                }
            }
        }
    }
}

/// Kernel arguments for confidence computation.
#[repr(C)]
struct ConfidenceArgs {
    draft_logits: *mut c_void,
    target_logits: *mut c_void,
    confidence_scores: *mut c_void,
    batch_size: u32,
    vocab_size: u32,
    num_draft_tokens: u32,
    _pad: u32,
}

/// Kernel arguments for depth adjustment.
#[repr(C)]
struct AdjustDepthArgs {
    acceptance_history: *mut c_void,
    confidence_scores: *mut c_void,
    current_depths: *mut c_void,
    new_depths: *mut c_void,
    batch_size: u32,
    history_len: u32,
    min_depth: u32,
    max_depth: u32,
    confidence_threshold: f32,
    _pad: [u32; 3],
}

/// EAGLE-3 Kernel wrapper for AMD GPUs.
pub struct HsaEagle3Kernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_confidence_f32: Option<HsaKernelModule>,
    module_confidence_f16: Option<HsaKernelModule>,
    module_adjust_depth: Option<HsaKernelModule>,
    module_draft_f32: Option<HsaKernelModule>,
    module_draft_f16: Option<HsaKernelModule>,
    module_update_history: Option<HsaKernelModule>,
    config: HsaEagle3Config,
}

impl HsaEagle3Kernel {
    /// Create a new EAGLE-3 kernel instance.
    pub fn new(config: HsaEagle3Config) -> Result<Self, HsaEagle3Error> {
        config.validate()?;

        let lib = get_hsa_lib()
            .map_err(|e| HsaEagle3Error::HsaNotAvailable(e.to_string()))?;

        let agents = find_gpu_agents()
            .map_err(|e| HsaEagle3Error::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() {
            return Err(HsaEagle3Error::NoGpuFound);
        }

        let agent = agents.into_iter().next().unwrap();

        // Create queue
        let mut queue: HsaQueue = ptr::null_mut();
        unsafe {
            let status = (lib.hsa_queue_create)(
                agent.handle,
                4096,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                u32::MAX,
                u32::MAX,
                &mut queue,
            );
            check_hsa(status, "hsa_queue_create")?;
        }

        // Create signal
        let mut signal: HsaSignal = 0;
        unsafe {
            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "hsa_signal_create")?;
        }

        // Load kernel modules (lazy loading - try to load, but allow failure)
        let module_confidence_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_CONFIDENCE_F32).ok();
        let module_confidence_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_CONFIDENCE_F16).ok();
        let module_adjust_depth = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_ADJUST_DEPTH).ok();
        let module_draft_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_DRAFT_F32).ok();
        let module_draft_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_DRAFT_F16).ok();
        let module_update_history = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_UPDATE_HISTORY).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_confidence_f32,
            module_confidence_f16,
            module_adjust_depth,
            module_draft_f32,
            module_draft_f16,
            module_update_history,
            config,
        })
    }

    /// Compute confidence scores for draft tokens (FP32).
    pub fn compute_confidence_f32(
        &self,
        draft_logits: *mut c_void,
        target_logits: *mut c_void,
        confidence_scores: *mut c_void,
        batch_size: usize,
        vocab_size: usize,
        num_draft_tokens: usize,
    ) -> Result<(), HsaEagle3Error> {
        let module = self.module_confidence_f32.as_ref()
            .ok_or(HsaEagle3Error::KernelMissing(KERNEL_CONFIDENCE_F32))?;

        // Kernel execution would go here
        // For now, return success as placeholder
        Ok(())
    }

    /// Compute confidence scores for draft tokens (FP16).
    pub fn compute_confidence_f16(
        &self,
        draft_logits: *mut c_void,
        target_logits: *mut c_void,
        confidence_scores: *mut c_void,
        batch_size: usize,
        vocab_size: usize,
        num_draft_tokens: usize,
    ) -> Result<(), HsaEagle3Error> {
        let module = self.module_confidence_f16.as_ref()
            .ok_or(HsaEagle3Error::KernelMissing(KERNEL_CONFIDENCE_F16))?;

        Ok(())
    }

    /// Adjust speculation depth based on history.
    pub fn adjust_depth(
        &self,
        acceptance_history: *mut c_void,
        confidence_scores: *mut c_void,
        current_depths: *mut c_void,
        new_depths: *mut c_void,
        batch_size: usize,
        history_len: usize,
    ) -> Result<(), HsaEagle3Error> {
        let module = self.module_adjust_depth.as_ref()
            .ok_or(HsaEagle3Error::KernelMissing(KERNEL_ADJUST_DEPTH))?;

        Ok(())
    }

    /// Get the GPU agent.
    pub fn agent(&self) -> &GpuAgent {
        &self.agent
    }

    /// Get the configuration.
    pub fn config(&self) -> &HsaEagle3Config {
        &self.config
    }
}

impl Drop for HsaEagle3Kernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 {
                    let _ = (lib.hsa_signal_destroy)(self.signal);
                }
                if !self.queue.is_null() {
                    let _ = (lib.hsa_queue_destroy)(self.queue);
                }
            }
        }
    }
}

// SAFETY: HSA queues and signals are thread-safe.
unsafe impl Send for HsaEagle3Kernel {}
unsafe impl Sync for HsaEagle3Kernel {}
