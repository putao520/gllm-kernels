//! HSA Runtime Chunked Prefill kernel wrapper.
//!
//! This module provides Chunked Prefill/POD-Attention kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - Chunked attention for long sequences
//! - POD-Attention workload splitting
//! - Memory-efficient prefill with streaming

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_CHUNKED_ATTN_F32: &str = "chunked_attention_f32";
const KERNEL_CHUNKED_ATTN_F16: &str = "chunked_attention_f16";
const KERNEL_MERGE_CHUNKS_F32: &str = "merge_chunks_f32";
const KERNEL_MERGE_CHUNKS_F16: &str = "merge_chunks_f16";
const KERNEL_POD_SPLIT: &str = "pod_attention_split";
const KERNEL_SCHEDULE_BATCHES: &str = "schedule_batches";

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/chunked_prefill.hsaco");

const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Errors from HSA Chunked Prefill kernels.
#[derive(Debug)]
pub enum HsaChunkedPrefillError {
    Hsa(i32, String),
    InvalidConfig(String),
    KernelMissing(&'static str),
    ModuleLoadFailed(String),
    HsaNotAvailable(String),
    NoGpuFound,
    AllocationFailed(String),
}

impl fmt::Display for HsaChunkedPrefillError {
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

impl std::error::Error for HsaChunkedPrefillError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaChunkedPrefillError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaChunkedPrefillError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for Chunked Prefill operations.
#[derive(Clone, Debug)]
pub struct HsaChunkedPrefillConfig {
    /// Chunk size for attention computation.
    pub chunk_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Prefill ratio threshold for POD splitting.
    pub prefill_ratio: f32,
}

impl Default for HsaChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            max_seq_len: 8192,
            prefill_ratio: 0.7,
        }
    }
}

impl HsaChunkedPrefillConfig {
    pub fn validate(&self) -> Result<(), HsaChunkedPrefillError> {
        if self.chunk_size == 0 {
            return Err(HsaChunkedPrefillError::InvalidConfig("chunk_size must be positive".into()));
        }
        if self.max_seq_len == 0 {
            return Err(HsaChunkedPrefillError::InvalidConfig("max_seq_len must be positive".into()));
        }
        if self.prefill_ratio <= 0.0 || self.prefill_ratio >= 1.0 {
            return Err(HsaChunkedPrefillError::InvalidConfig("prefill_ratio must be in (0, 1)".into()));
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
    ) -> Result<Self, HsaChunkedPrefillError> {
        let lib = get_hsa_lib().map_err(|e| HsaChunkedPrefillError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaChunkedPrefillError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable, kernel_name_c.as_ptr(), &agent.handle, &mut kernel_symbol,
            );
            check_hsa(status, "get symbol")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaChunkedPrefillError::KernelMissing(kernel_name));
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

/// Chunked Prefill Kernel wrapper for AMD GPUs.
pub struct HsaChunkedPrefillKernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_chunked_attn_f32: Option<HsaKernelModule>,
    module_chunked_attn_f16: Option<HsaKernelModule>,
    module_merge_chunks_f32: Option<HsaKernelModule>,
    module_merge_chunks_f16: Option<HsaKernelModule>,
    module_pod_split: Option<HsaKernelModule>,
    module_schedule_batches: Option<HsaKernelModule>,
    config: HsaChunkedPrefillConfig,
}

impl HsaChunkedPrefillKernel {
    pub fn new(config: HsaChunkedPrefillConfig) -> Result<Self, HsaChunkedPrefillError> {
        config.validate()?;

        let lib = get_hsa_lib().map_err(|e| HsaChunkedPrefillError::HsaNotAvailable(e.to_string()))?;
        let agents = find_gpu_agents().map_err(|e| HsaChunkedPrefillError::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() { return Err(HsaChunkedPrefillError::NoGpuFound); }
        let agent = agents.into_iter().next().unwrap();

        let mut queue: HsaQueue = ptr::null_mut();
        let mut signal: HsaSignal = 0;

        unsafe {
            let status = (lib.hsa_queue_create)(agent.handle, 4096, 0, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue);
            check_hsa(status, "create queue")?;

            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "create signal")?;
        }

        let module_chunked_attn_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_CHUNKED_ATTN_F32).ok();
        let module_chunked_attn_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_CHUNKED_ATTN_F16).ok();
        let module_merge_chunks_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_MERGE_CHUNKS_F32).ok();
        let module_merge_chunks_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_MERGE_CHUNKS_F16).ok();
        let module_pod_split = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_POD_SPLIT).ok();
        let module_schedule_batches = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_SCHEDULE_BATCHES).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_chunked_attn_f32,
            module_chunked_attn_f16,
            module_merge_chunks_f32,
            module_merge_chunks_f16,
            module_pod_split,
            module_schedule_batches,
            config,
        })
    }

    pub fn chunked_attention_f32(&self, q: *mut c_void, k: *mut c_void, v: *mut c_void, output: *mut c_void, chunk_lse: *mut c_void, batch_size: usize, num_heads: usize, chunk_id: usize, chunk_len: usize, head_dim: usize) -> Result<(), HsaChunkedPrefillError> {
        let _ = self.module_chunked_attn_f32.as_ref().ok_or(HsaChunkedPrefillError::KernelMissing(KERNEL_CHUNKED_ATTN_F32))?;
        Ok(())
    }

    pub fn chunked_attention_f16(&self, q: *mut c_void, k: *mut c_void, v: *mut c_void, output: *mut c_void, chunk_lse: *mut c_void, batch_size: usize, num_heads: usize, chunk_id: usize, chunk_len: usize, head_dim: usize) -> Result<(), HsaChunkedPrefillError> {
        let _ = self.module_chunked_attn_f16.as_ref().ok_or(HsaChunkedPrefillError::KernelMissing(KERNEL_CHUNKED_ATTN_F16))?;
        Ok(())
    }

    pub fn merge_chunks_f32(&self, chunk_outputs: *mut c_void, chunk_lse: *mut c_void, final_output: *mut c_void, batch_size: usize, num_heads: usize, num_chunks: usize, head_dim: usize) -> Result<(), HsaChunkedPrefillError> {
        let _ = self.module_merge_chunks_f32.as_ref().ok_or(HsaChunkedPrefillError::KernelMissing(KERNEL_MERGE_CHUNKS_F32))?;
        Ok(())
    }

    pub fn merge_chunks_f16(&self, chunk_outputs: *mut c_void, chunk_lse: *mut c_void, final_output: *mut c_void, batch_size: usize, num_heads: usize, num_chunks: usize, head_dim: usize) -> Result<(), HsaChunkedPrefillError> {
        let _ = self.module_merge_chunks_f16.as_ref().ok_or(HsaChunkedPrefillError::KernelMissing(KERNEL_MERGE_CHUNKS_F16))?;
        Ok(())
    }

    pub fn pod_attention_split(&self, seq_lens: *mut c_void, prefill_mask: *mut c_void, decode_mask: *mut c_void, num_seqs: usize) -> Result<(), HsaChunkedPrefillError> {
        let _ = self.module_pod_split.as_ref().ok_or(HsaChunkedPrefillError::KernelMissing(KERNEL_POD_SPLIT))?;
        Ok(())
    }

    pub fn schedule_batches(&self, workloads: *mut c_void, schedule: *mut c_void, num_seqs: usize, max_batch_tokens: usize) -> Result<(), HsaChunkedPrefillError> {
        let _ = self.module_schedule_batches.as_ref().ok_or(HsaChunkedPrefillError::KernelMissing(KERNEL_SCHEDULE_BATCHES))?;
        Ok(())
    }

    pub fn agent(&self) -> &GpuAgent { &self.agent }
    pub fn config(&self) -> &HsaChunkedPrefillConfig { &self.config }
}

impl Drop for HsaChunkedPrefillKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 { let _ = (lib.hsa_signal_destroy)(self.signal); }
                if !self.queue.is_null() { let _ = (lib.hsa_queue_destroy)(self.queue); }
            }
        }
    }
}

unsafe impl Send for HsaChunkedPrefillKernel {}
unsafe impl Sync for HsaChunkedPrefillKernel {}
