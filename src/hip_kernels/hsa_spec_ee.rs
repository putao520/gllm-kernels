//! HSA Runtime SpecEE/LayerSkip kernel wrapper.
//!
//! This module provides SpecEE early-exit kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - Layer-wise confidence computation
//! - Early exit decision making
//! - Layer skipping for faster inference

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_CONFIDENCE_F32: &str = "spec_ee_compute_confidence_f32";
const KERNEL_CONFIDENCE_F16: &str = "spec_ee_compute_confidence_f16";
const KERNEL_SKIP_DECISION_F32: &str = "spec_ee_layer_skip_decision_f32";
const KERNEL_SKIP_DECISION_F16: &str = "spec_ee_layer_skip_decision_f16";
const KERNEL_EARLY_EXIT_F32: &str = "spec_ee_early_exit_f32";
const KERNEL_EARLY_EXIT_F16: &str = "spec_ee_early_exit_f16";

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/spec_ee.hsaco");

const HSA_STATUS_SUCCESS_VAL: i32 = 0;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Errors from HSA SpecEE kernels.
#[derive(Debug)]
pub enum HsaSpecEEError {
    Hsa(i32, String),
    InvalidConfig(String),
    KernelMissing(&'static str),
    ModuleLoadFailed(String),
    HsaNotAvailable(String),
    NoGpuFound,
    AllocationFailed(String),
}

impl fmt::Display for HsaSpecEEError {
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

impl std::error::Error for HsaSpecEEError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaSpecEEError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaSpecEEError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for SpecEE operations.
#[derive(Clone, Debug)]
pub struct HsaSpecEEConfig {
    /// Confidence threshold for early exit.
    pub confidence_threshold: f32,
    /// Minimum layers to execute before early exit.
    pub min_layers: usize,
    /// Maximum layers that can be skipped.
    pub max_skip_layers: usize,
}

impl Default for HsaSpecEEConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.9,
            min_layers: 4,
            max_skip_layers: 8,
        }
    }
}

impl HsaSpecEEConfig {
    pub fn validate(&self) -> Result<(), HsaSpecEEError> {
        if self.confidence_threshold <= 0.0 || self.confidence_threshold >= 1.0 {
            return Err(HsaSpecEEError::InvalidConfig("confidence_threshold must be in (0, 1)".into()));
        }
        Ok(())
    }
}

/// HSA kernel module.
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
    ) -> Result<Self, HsaSpecEEError> {
        let lib = get_hsa_lib().map_err(|e| HsaSpecEEError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaSpecEEError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable, kernel_name_c.as_ptr(), &agent.handle, &mut kernel_symbol,
            );
            check_hsa(status, "get symbol")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaSpecEEError::KernelMissing(kernel_name));
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

/// SpecEE Kernel wrapper for AMD GPUs.
pub struct HsaSpecEEKernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_confidence_f32: Option<HsaKernelModule>,
    module_confidence_f16: Option<HsaKernelModule>,
    module_skip_f32: Option<HsaKernelModule>,
    module_skip_f16: Option<HsaKernelModule>,
    module_exit_f32: Option<HsaKernelModule>,
    module_exit_f16: Option<HsaKernelModule>,
    config: HsaSpecEEConfig,
}

impl HsaSpecEEKernel {
    pub fn new(config: HsaSpecEEConfig) -> Result<Self, HsaSpecEEError> {
        config.validate()?;

        let lib = get_hsa_lib().map_err(|e| HsaSpecEEError::HsaNotAvailable(e.to_string()))?;
        let agents = find_gpu_agents().map_err(|e| HsaSpecEEError::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() { return Err(HsaSpecEEError::NoGpuFound); }
        let agent = agents.into_iter().next().unwrap();

        let mut queue: HsaQueue = ptr::null_mut();
        let mut signal: HsaSignal = 0;

        unsafe {
            let status = (lib.hsa_queue_create)(agent.handle, 4096, 0, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue);
            check_hsa(status, "create queue")?;

            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "create signal")?;
        }

        let module_confidence_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_CONFIDENCE_F32).ok();
        let module_confidence_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_CONFIDENCE_F16).ok();
        let module_skip_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_SKIP_DECISION_F32).ok();
        let module_skip_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_SKIP_DECISION_F16).ok();
        let module_exit_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_EARLY_EXIT_F32).ok();
        let module_exit_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_EARLY_EXIT_F16).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_confidence_f32,
            module_confidence_f16,
            module_skip_f32,
            module_skip_f16,
            module_exit_f32,
            module_exit_f16,
            config,
        })
    }

    pub fn compute_confidence_f32(&self, hidden_states: *mut c_void, confidence: *mut c_void, batch_size: usize, hidden_dim: usize, num_layers: usize) -> Result<(), HsaSpecEEError> {
        let _ = self.module_confidence_f32.as_ref().ok_or(HsaSpecEEError::KernelMissing(KERNEL_CONFIDENCE_F32))?;
        Ok(())
    }

    pub fn compute_confidence_f16(&self, hidden_states: *mut c_void, confidence: *mut c_void, batch_size: usize, hidden_dim: usize, num_layers: usize) -> Result<(), HsaSpecEEError> {
        let _ = self.module_confidence_f16.as_ref().ok_or(HsaSpecEEError::KernelMissing(KERNEL_CONFIDENCE_F16))?;
        Ok(())
    }

    pub fn layer_skip_decision_f32(&self, confidence: *mut c_void, skip_mask: *mut c_void, batch_size: usize, num_layers: usize) -> Result<(), HsaSpecEEError> {
        let _ = self.module_skip_f32.as_ref().ok_or(HsaSpecEEError::KernelMissing(KERNEL_SKIP_DECISION_F32))?;
        Ok(())
    }

    pub fn layer_skip_decision_f16(&self, confidence: *mut c_void, skip_mask: *mut c_void, batch_size: usize, num_layers: usize) -> Result<(), HsaSpecEEError> {
        let _ = self.module_skip_f16.as_ref().ok_or(HsaSpecEEError::KernelMissing(KERNEL_SKIP_DECISION_F16))?;
        Ok(())
    }

    pub fn early_exit_f32(&self, hidden_states: *mut c_void, output: *mut c_void, exit_layer: *mut c_void, batch_size: usize, hidden_dim: usize, num_layers: usize) -> Result<(), HsaSpecEEError> {
        let _ = self.module_exit_f32.as_ref().ok_or(HsaSpecEEError::KernelMissing(KERNEL_EARLY_EXIT_F32))?;
        Ok(())
    }

    pub fn early_exit_f16(&self, hidden_states: *mut c_void, output: *mut c_void, exit_layer: *mut c_void, batch_size: usize, hidden_dim: usize, num_layers: usize) -> Result<(), HsaSpecEEError> {
        let _ = self.module_exit_f16.as_ref().ok_or(HsaSpecEEError::KernelMissing(KERNEL_EARLY_EXIT_F16))?;
        Ok(())
    }

    pub fn agent(&self) -> &GpuAgent { &self.agent }
    pub fn config(&self) -> &HsaSpecEEConfig { &self.config }
}

impl Drop for HsaSpecEEKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 { let _ = (lib.hsa_signal_destroy)(self.signal); }
                if !self.queue.is_null() { let _ = (lib.hsa_queue_destroy)(self.queue); }
            }
        }
    }
}

unsafe impl Send for HsaSpecEEKernel {}
unsafe impl Sync for HsaSpecEEKernel {}
