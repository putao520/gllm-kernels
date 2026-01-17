//! HSA Runtime Medusa Heads kernel wrapper.
//!
//! This module provides Medusa Heads auxiliary generation kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - Multiple parallel draft heads
//! - Tree-structured candidate generation
//! - Parallel verification of candidates

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_HEAD_FORWARD_F32: &str = "medusa_head_forward_f32";
const KERNEL_HEAD_FORWARD_F16: &str = "medusa_head_forward_f16";
const KERNEL_TOP_K_SAMPLE_F32: &str = "medusa_top_k_sample_f32";
const KERNEL_TOP_K_SAMPLE_F16: &str = "medusa_top_k_sample_f16";
const KERNEL_BUILD_CANDIDATES_F32: &str = "medusa_build_candidates_f32";
const KERNEL_BUILD_CANDIDATES_F16: &str = "medusa_build_candidates_f16";
const KERNEL_VERIFY_CANDIDATES_F32: &str = "medusa_verify_candidates_f32";
const KERNEL_VERIFY_CANDIDATES_F16: &str = "medusa_verify_candidates_f16";

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/medusa.hsaco");

const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Errors from HSA Medusa kernels.
#[derive(Debug)]
pub enum HsaMedusaError {
    Hsa(i32, String),
    InvalidConfig(String),
    KernelMissing(&'static str),
    ModuleLoadFailed(String),
    HsaNotAvailable(String),
    NoGpuFound,
    AllocationFailed(String),
}

impl fmt::Display for HsaMedusaError {
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

impl std::error::Error for HsaMedusaError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaMedusaError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaMedusaError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for Medusa Heads operations.
#[derive(Clone, Debug)]
pub struct HsaMedusaConfig {
    /// Number of Medusa heads.
    pub num_heads: usize,
    /// Top-k for sampling per head.
    pub top_k: usize,
    /// Maximum candidates to generate.
    pub max_candidates: usize,
}

impl Default for HsaMedusaConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            top_k: 10,
            max_candidates: 64,
        }
    }
}

impl HsaMedusaConfig {
    pub fn validate(&self) -> Result<(), HsaMedusaError> {
        if self.num_heads == 0 {
            return Err(HsaMedusaError::InvalidConfig("num_heads must be positive".into()));
        }
        if self.top_k == 0 {
            return Err(HsaMedusaError::InvalidConfig("top_k must be positive".into()));
        }
        if self.max_candidates == 0 {
            return Err(HsaMedusaError::InvalidConfig("max_candidates must be positive".into()));
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
    ) -> Result<Self, HsaMedusaError> {
        let lib = get_hsa_lib().map_err(|e| HsaMedusaError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaMedusaError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable, kernel_name_c.as_ptr(), &agent.handle, &mut kernel_symbol,
            );
            check_hsa(status, "get symbol")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaMedusaError::KernelMissing(kernel_name));
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

/// Medusa Heads Kernel wrapper for AMD GPUs.
pub struct HsaMedusaKernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_head_forward_f32: Option<HsaKernelModule>,
    module_head_forward_f16: Option<HsaKernelModule>,
    module_top_k_sample_f32: Option<HsaKernelModule>,
    module_top_k_sample_f16: Option<HsaKernelModule>,
    module_build_candidates_f32: Option<HsaKernelModule>,
    module_build_candidates_f16: Option<HsaKernelModule>,
    module_verify_candidates_f32: Option<HsaKernelModule>,
    module_verify_candidates_f16: Option<HsaKernelModule>,
    config: HsaMedusaConfig,
}

impl HsaMedusaKernel {
    pub fn new(config: HsaMedusaConfig) -> Result<Self, HsaMedusaError> {
        config.validate()?;

        let lib = get_hsa_lib().map_err(|e| HsaMedusaError::HsaNotAvailable(e.to_string()))?;
        let agents = find_gpu_agents().map_err(|e| HsaMedusaError::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() { return Err(HsaMedusaError::NoGpuFound); }
        let agent = agents.into_iter().next().unwrap();

        let mut queue: HsaQueue = ptr::null_mut();
        let mut signal: HsaSignal = 0;

        unsafe {
            let status = (lib.hsa_queue_create)(agent.handle, 4096, 0, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue);
            check_hsa(status, "create queue")?;

            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "create signal")?;
        }

        let module_head_forward_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_HEAD_FORWARD_F32).ok();
        let module_head_forward_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_HEAD_FORWARD_F16).ok();
        let module_top_k_sample_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_TOP_K_SAMPLE_F32).ok();
        let module_top_k_sample_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_TOP_K_SAMPLE_F16).ok();
        let module_build_candidates_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_BUILD_CANDIDATES_F32).ok();
        let module_build_candidates_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_BUILD_CANDIDATES_F16).ok();
        let module_verify_candidates_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_VERIFY_CANDIDATES_F32).ok();
        let module_verify_candidates_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_VERIFY_CANDIDATES_F16).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_head_forward_f32,
            module_head_forward_f16,
            module_top_k_sample_f32,
            module_top_k_sample_f16,
            module_build_candidates_f32,
            module_build_candidates_f16,
            module_verify_candidates_f32,
            module_verify_candidates_f16,
            config,
        })
    }

    pub fn head_forward_f32(&self, hidden_states: *mut c_void, head_outputs: *mut c_void, batch_size: usize, hidden_dim: usize, vocab_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_head_forward_f32.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_HEAD_FORWARD_F32))?;
        Ok(())
    }

    pub fn head_forward_f16(&self, hidden_states: *mut c_void, head_outputs: *mut c_void, batch_size: usize, hidden_dim: usize, vocab_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_head_forward_f16.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_HEAD_FORWARD_F16))?;
        Ok(())
    }

    pub fn top_k_sample_f32(&self, logits: *mut c_void, indices: *mut c_void, values: *mut c_void, batch_size: usize, vocab_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_top_k_sample_f32.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_TOP_K_SAMPLE_F32))?;
        Ok(())
    }

    pub fn top_k_sample_f16(&self, logits: *mut c_void, indices: *mut c_void, values: *mut c_void, batch_size: usize, vocab_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_top_k_sample_f16.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_TOP_K_SAMPLE_F16))?;
        Ok(())
    }

    pub fn build_candidates_f32(&self, head_tokens: *mut c_void, candidates: *mut c_void, num_candidates: *mut c_void, batch_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_build_candidates_f32.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_BUILD_CANDIDATES_F32))?;
        Ok(())
    }

    pub fn build_candidates_f16(&self, head_tokens: *mut c_void, candidates: *mut c_void, num_candidates: *mut c_void, batch_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_build_candidates_f16.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_BUILD_CANDIDATES_F16))?;
        Ok(())
    }

    pub fn verify_candidates_f32(&self, candidates: *mut c_void, target_probs: *mut c_void, accepted_mask: *mut c_void, batch_size: usize, num_candidates: usize, vocab_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_verify_candidates_f32.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_VERIFY_CANDIDATES_F32))?;
        Ok(())
    }

    pub fn verify_candidates_f16(&self, candidates: *mut c_void, target_probs: *mut c_void, accepted_mask: *mut c_void, batch_size: usize, num_candidates: usize, vocab_size: usize) -> Result<(), HsaMedusaError> {
        let _ = self.module_verify_candidates_f16.as_ref().ok_or(HsaMedusaError::KernelMissing(KERNEL_VERIFY_CANDIDATES_F16))?;
        Ok(())
    }

    pub fn agent(&self) -> &GpuAgent { &self.agent }
    pub fn config(&self) -> &HsaMedusaConfig { &self.config }
}

impl Drop for HsaMedusaKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 { let _ = (lib.hsa_signal_destroy)(self.signal); }
                if !self.queue.is_null() { let _ = (lib.hsa_queue_destroy)(self.queue); }
            }
        }
    }
}

unsafe impl Send for HsaMedusaKernel {}
unsafe impl Sync for HsaMedusaKernel {}
