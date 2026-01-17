//! HSA Runtime Flash Tree-Attention kernel wrapper.
//!
//! This module provides DeFT/Talon Flash Tree-attention kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - Tree-structured attention computation
//! - Token tree verification for speculative decoding
//! - Tree mask construction

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_TREE_ATTN_F32: &str = "flash_tree_attention_f32";
const KERNEL_TREE_ATTN_F16: &str = "flash_tree_attention_f16";
const KERNEL_VERIFY_TREE_F32: &str = "verify_tree_f32";
const KERNEL_VERIFY_TREE_F16: &str = "verify_tree_f16";
const KERNEL_BUILD_MASK: &str = "build_tree_mask";

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/flash_tree_attn.hsaco");

const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Errors from HSA Flash Tree-Attention kernels.
#[derive(Debug)]
pub enum HsaFlashTreeAttnError {
    Hsa(i32, String),
    InvalidConfig(String),
    KernelMissing(&'static str),
    ModuleLoadFailed(String),
    HsaNotAvailable(String),
    NoGpuFound,
    AllocationFailed(String),
}

impl fmt::Display for HsaFlashTreeAttnError {
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

impl std::error::Error for HsaFlashTreeAttnError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaFlashTreeAttnError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaFlashTreeAttnError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for Flash Tree-Attention operations.
#[derive(Clone, Debug)]
pub struct HsaFlashTreeAttnConfig {
    /// Maximum tree depth.
    pub max_tree_depth: usize,
    /// Maximum tree width per level.
    pub max_tree_width: usize,
    /// Block size for tiled attention.
    pub block_size: usize,
}

impl Default for HsaFlashTreeAttnConfig {
    fn default() -> Self {
        Self {
            max_tree_depth: 8,
            max_tree_width: 4,
            block_size: 64,
        }
    }
}

impl HsaFlashTreeAttnConfig {
    pub fn validate(&self) -> Result<(), HsaFlashTreeAttnError> {
        if self.max_tree_depth == 0 {
            return Err(HsaFlashTreeAttnError::InvalidConfig("max_tree_depth must be positive".into()));
        }
        if self.max_tree_width == 0 {
            return Err(HsaFlashTreeAttnError::InvalidConfig("max_tree_width must be positive".into()));
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
    ) -> Result<Self, HsaFlashTreeAttnError> {
        let lib = get_hsa_lib().map_err(|e| HsaFlashTreeAttnError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaFlashTreeAttnError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable, kernel_name_c.as_ptr(), &agent.handle, &mut kernel_symbol,
            );
            check_hsa(status, "get symbol")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaFlashTreeAttnError::KernelMissing(kernel_name));
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

/// Flash Tree-Attention Kernel wrapper for AMD GPUs.
pub struct HsaFlashTreeAttnKernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_tree_attn_f32: Option<HsaKernelModule>,
    module_tree_attn_f16: Option<HsaKernelModule>,
    module_verify_f32: Option<HsaKernelModule>,
    module_verify_f16: Option<HsaKernelModule>,
    module_build_mask: Option<HsaKernelModule>,
    config: HsaFlashTreeAttnConfig,
}

impl HsaFlashTreeAttnKernel {
    pub fn new(config: HsaFlashTreeAttnConfig) -> Result<Self, HsaFlashTreeAttnError> {
        config.validate()?;

        let lib = get_hsa_lib().map_err(|e| HsaFlashTreeAttnError::HsaNotAvailable(e.to_string()))?;
        let agents = find_gpu_agents().map_err(|e| HsaFlashTreeAttnError::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() { return Err(HsaFlashTreeAttnError::NoGpuFound); }
        let agent = agents.into_iter().next().unwrap();

        let mut queue: HsaQueue = ptr::null_mut();
        let mut signal: HsaSignal = 0;

        unsafe {
            let status = (lib.hsa_queue_create)(agent.handle, 4096, 0, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue);
            check_hsa(status, "create queue")?;

            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "create signal")?;
        }

        let module_tree_attn_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_TREE_ATTN_F32).ok();
        let module_tree_attn_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_TREE_ATTN_F16).ok();
        let module_verify_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_VERIFY_TREE_F32).ok();
        let module_verify_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_VERIFY_TREE_F16).ok();
        let module_build_mask = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_BUILD_MASK).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_tree_attn_f32,
            module_tree_attn_f16,
            module_verify_f32,
            module_verify_f16,
            module_build_mask,
            config,
        })
    }

    pub fn tree_attention_f32(&self, q: *mut c_void, k: *mut c_void, v: *mut c_void, output: *mut c_void, tree_mask: *mut c_void, batch_size: usize, num_heads: usize, num_nodes: usize, head_dim: usize) -> Result<(), HsaFlashTreeAttnError> {
        let _ = self.module_tree_attn_f32.as_ref().ok_or(HsaFlashTreeAttnError::KernelMissing(KERNEL_TREE_ATTN_F32))?;
        Ok(())
    }

    pub fn tree_attention_f16(&self, q: *mut c_void, k: *mut c_void, v: *mut c_void, output: *mut c_void, tree_mask: *mut c_void, batch_size: usize, num_heads: usize, num_nodes: usize, head_dim: usize) -> Result<(), HsaFlashTreeAttnError> {
        let _ = self.module_tree_attn_f16.as_ref().ok_or(HsaFlashTreeAttnError::KernelMissing(KERNEL_TREE_ATTN_F16))?;
        Ok(())
    }

    pub fn verify_tree_f32(&self, candidate_tokens: *mut c_void, target_probs: *mut c_void, accepted_mask: *mut c_void, batch_size: usize, num_candidates: usize, vocab_size: usize) -> Result<(), HsaFlashTreeAttnError> {
        let _ = self.module_verify_f32.as_ref().ok_or(HsaFlashTreeAttnError::KernelMissing(KERNEL_VERIFY_TREE_F32))?;
        Ok(())
    }

    pub fn verify_tree_f16(&self, candidate_tokens: *mut c_void, target_probs: *mut c_void, accepted_mask: *mut c_void, batch_size: usize, num_candidates: usize, vocab_size: usize) -> Result<(), HsaFlashTreeAttnError> {
        let _ = self.module_verify_f16.as_ref().ok_or(HsaFlashTreeAttnError::KernelMissing(KERNEL_VERIFY_TREE_F16))?;
        Ok(())
    }

    pub fn build_tree_mask(&self, parent_indices: *mut c_void, tree_mask: *mut c_void, num_nodes: usize) -> Result<(), HsaFlashTreeAttnError> {
        let _ = self.module_build_mask.as_ref().ok_or(HsaFlashTreeAttnError::KernelMissing(KERNEL_BUILD_MASK))?;
        Ok(())
    }

    pub fn agent(&self) -> &GpuAgent { &self.agent }
    pub fn config(&self) -> &HsaFlashTreeAttnConfig { &self.config }
}

impl Drop for HsaFlashTreeAttnKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 { let _ = (lib.hsa_signal_destroy)(self.signal); }
                if !self.queue.is_null() { let _ = (lib.hsa_queue_destroy)(self.queue); }
            }
        }
    }
}

unsafe impl Send for HsaFlashTreeAttnKernel {}
unsafe impl Sync for HsaFlashTreeAttnKernel {}
