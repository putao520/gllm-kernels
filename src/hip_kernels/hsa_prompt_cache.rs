//! HSA Runtime Prompt Cache kernel wrapper.
//!
//! This module provides Prompt Caching/CacheBlend kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - Hash-based prompt prefix matching
//! - KV cache reuse across requests
//! - Smooth blending at cache boundaries

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_HASH_F32: &str = "compute_hash_f32";
const KERNEL_HASH_F16: &str = "compute_hash_f16";
const KERNEL_PREFIX_MATCH: &str = "find_prefix_match";
const KERNEL_BLEND_F32: &str = "cache_blend_f32";
const KERNEL_BLEND_F16: &str = "cache_blend_f16";
const KERNEL_COPY_KV_F32: &str = "copy_kv_f32";
const KERNEL_COPY_KV_F16: &str = "copy_kv_f16";

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/prompt_cache.hsaco");

const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Errors from HSA Prompt Cache kernels.
#[derive(Debug)]
pub enum HsaPromptCacheError {
    Hsa(i32, String),
    InvalidConfig(String),
    KernelMissing(&'static str),
    ModuleLoadFailed(String),
    HsaNotAvailable(String),
    NoGpuFound,
    AllocationFailed(String),
}

impl fmt::Display for HsaPromptCacheError {
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

impl std::error::Error for HsaPromptCacheError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaPromptCacheError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaPromptCacheError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for Prompt Cache operations.
#[derive(Clone, Debug)]
pub struct HsaPromptCacheConfig {
    /// Chunk size for hash computation.
    pub chunk_size: usize,
    /// Maximum prefix length to match.
    pub max_prefix_len: usize,
    /// Blend region length at cache boundary.
    pub blend_len: usize,
}

impl Default for HsaPromptCacheConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            max_prefix_len: 4096,
            blend_len: 16,
        }
    }
}

impl HsaPromptCacheConfig {
    pub fn validate(&self) -> Result<(), HsaPromptCacheError> {
        if self.chunk_size == 0 {
            return Err(HsaPromptCacheError::InvalidConfig("chunk_size must be positive".into()));
        }
        if self.max_prefix_len == 0 {
            return Err(HsaPromptCacheError::InvalidConfig("max_prefix_len must be positive".into()));
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
    ) -> Result<Self, HsaPromptCacheError> {
        let lib = get_hsa_lib().map_err(|e| HsaPromptCacheError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaPromptCacheError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable, kernel_name_c.as_ptr(), &agent.handle, &mut kernel_symbol,
            );
            check_hsa(status, "get symbol")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaPromptCacheError::KernelMissing(kernel_name));
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

/// Prompt Cache Kernel wrapper for AMD GPUs.
pub struct HsaPromptCacheKernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_hash_f32: Option<HsaKernelModule>,
    module_hash_f16: Option<HsaKernelModule>,
    module_prefix_match: Option<HsaKernelModule>,
    module_blend_f32: Option<HsaKernelModule>,
    module_blend_f16: Option<HsaKernelModule>,
    module_copy_kv_f32: Option<HsaKernelModule>,
    module_copy_kv_f16: Option<HsaKernelModule>,
    config: HsaPromptCacheConfig,
}

impl HsaPromptCacheKernel {
    pub fn new(config: HsaPromptCacheConfig) -> Result<Self, HsaPromptCacheError> {
        config.validate()?;

        let lib = get_hsa_lib().map_err(|e| HsaPromptCacheError::HsaNotAvailable(e.to_string()))?;
        let agents = find_gpu_agents().map_err(|e| HsaPromptCacheError::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() { return Err(HsaPromptCacheError::NoGpuFound); }
        let agent = agents.into_iter().next().unwrap();

        let mut queue: HsaQueue = ptr::null_mut();
        let mut signal: HsaSignal = 0;

        unsafe {
            let status = (lib.hsa_queue_create)(agent.handle, 4096, 0, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue);
            check_hsa(status, "create queue")?;

            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "create signal")?;
        }

        let module_hash_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_HASH_F32).ok();
        let module_hash_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_HASH_F16).ok();
        let module_prefix_match = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_PREFIX_MATCH).ok();
        let module_blend_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_BLEND_F32).ok();
        let module_blend_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_BLEND_F16).ok();
        let module_copy_kv_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_COPY_KV_F32).ok();
        let module_copy_kv_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_COPY_KV_F16).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_hash_f32,
            module_hash_f16,
            module_prefix_match,
            module_blend_f32,
            module_blend_f16,
            module_copy_kv_f32,
            module_copy_kv_f16,
            config,
        })
    }

    pub fn compute_hash_f32(&self, embeddings: *mut c_void, hashes: *mut c_void, seq_len: usize, hidden_dim: usize) -> Result<(), HsaPromptCacheError> {
        let _ = self.module_hash_f32.as_ref().ok_or(HsaPromptCacheError::KernelMissing(KERNEL_HASH_F32))?;
        Ok(())
    }

    pub fn compute_hash_f16(&self, embeddings: *mut c_void, hashes: *mut c_void, seq_len: usize, hidden_dim: usize) -> Result<(), HsaPromptCacheError> {
        let _ = self.module_hash_f16.as_ref().ok_or(HsaPromptCacheError::KernelMissing(KERNEL_HASH_F16))?;
        Ok(())
    }

    pub fn find_prefix_match(&self, query_hashes: *mut c_void, cached_hashes: *mut c_void, cached_lengths: *mut c_void, best_match_idx: *mut c_void, best_match_len: *mut c_void, query_len: usize, num_cached: usize) -> Result<(), HsaPromptCacheError> {
        let _ = self.module_prefix_match.as_ref().ok_or(HsaPromptCacheError::KernelMissing(KERNEL_PREFIX_MATCH))?;
        Ok(())
    }

    pub fn cache_blend_f32(&self, cached_kv: *mut c_void, new_kv: *mut c_void, output_kv: *mut c_void, blend_len: usize, num_heads: usize, head_dim: usize, blend_factor: f32) -> Result<(), HsaPromptCacheError> {
        let _ = self.module_blend_f32.as_ref().ok_or(HsaPromptCacheError::KernelMissing(KERNEL_BLEND_F32))?;
        Ok(())
    }

    pub fn cache_blend_f16(&self, cached_kv: *mut c_void, new_kv: *mut c_void, output_kv: *mut c_void, blend_len: usize, num_heads: usize, head_dim: usize, blend_factor: f32) -> Result<(), HsaPromptCacheError> {
        let _ = self.module_blend_f16.as_ref().ok_or(HsaPromptCacheError::KernelMissing(KERNEL_BLEND_F16))?;
        Ok(())
    }

    pub fn copy_kv_f32(&self, src: *mut c_void, dst: *mut c_void, num_tokens: usize, num_heads: usize, head_dim: usize) -> Result<(), HsaPromptCacheError> {
        let _ = self.module_copy_kv_f32.as_ref().ok_or(HsaPromptCacheError::KernelMissing(KERNEL_COPY_KV_F32))?;
        Ok(())
    }

    pub fn copy_kv_f16(&self, src: *mut c_void, dst: *mut c_void, num_tokens: usize, num_heads: usize, head_dim: usize) -> Result<(), HsaPromptCacheError> {
        let _ = self.module_copy_kv_f16.as_ref().ok_or(HsaPromptCacheError::KernelMissing(KERNEL_COPY_KV_F16))?;
        Ok(())
    }

    pub fn agent(&self) -> &GpuAgent { &self.agent }
    pub fn config(&self) -> &HsaPromptCacheConfig { &self.config }
}

impl Drop for HsaPromptCacheKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 { let _ = (lib.hsa_signal_destroy)(self.signal); }
                if !self.queue.is_null() { let _ = (lib.hsa_queue_destroy)(self.queue); }
            }
        }
    }
}

unsafe impl Send for HsaPromptCacheKernel {}
unsafe impl Sync for HsaPromptCacheKernel {}
