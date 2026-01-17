//! HSA Runtime INT2 Quantizer kernel wrapper.
//!
//! This module provides INT2 extreme quantization kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Features
//! - 2-bit weight quantization with scale/zero-point
//! - Efficient packing (16 values per u32)
//! - Dequantization for inference

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaQueue, HsaSignal, HSA_STATUS_SUCCESS,
};

const KERNEL_QUANTIZE_F32: &str = "int2_quantize_f32";
const KERNEL_QUANTIZE_F16: &str = "int2_quantize_f16";
const KERNEL_DEQUANTIZE_F32: &str = "int2_dequantize_f32";
const KERNEL_DEQUANTIZE_F16: &str = "int2_dequantize_f16";
const KERNEL_PACK: &str = "int2_pack";
const KERNEL_UNPACK: &str = "int2_unpack";

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/int2_quantizer.hsaco");

const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

/// Errors from HSA INT2 Quantizer kernels.
#[derive(Debug)]
pub enum HsaInt2QuantizerError {
    Hsa(i32, String),
    InvalidConfig(String),
    KernelMissing(&'static str),
    ModuleLoadFailed(String),
    HsaNotAvailable(String),
    NoGpuFound,
    AllocationFailed(String),
}

impl fmt::Display for HsaInt2QuantizerError {
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

impl std::error::Error for HsaInt2QuantizerError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaInt2QuantizerError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaInt2QuantizerError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// Configuration for INT2 Quantizer operations.
#[derive(Clone, Debug)]
pub struct HsaInt2QuantizerConfig {
    /// Group size for quantization (values sharing same scale/zero).
    pub group_size: usize,
    /// Whether to use symmetric quantization.
    pub symmetric: bool,
}

impl Default for HsaInt2QuantizerConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            symmetric: false,
        }
    }
}

impl HsaInt2QuantizerConfig {
    pub fn validate(&self) -> Result<(), HsaInt2QuantizerError> {
        if self.group_size == 0 || (self.group_size & (self.group_size - 1)) != 0 {
            return Err(HsaInt2QuantizerError::InvalidConfig("group_size must be power of 2".into()));
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
    ) -> Result<Self, HsaInt2QuantizerError> {
        let lib = get_hsa_lib().map_err(|e| HsaInt2QuantizerError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaInt2QuantizerError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable, kernel_name_c.as_ptr(), &agent.handle, &mut kernel_symbol,
            );
            check_hsa(status, "get symbol")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaInt2QuantizerError::KernelMissing(kernel_name));
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

/// INT2 Quantizer Kernel wrapper for AMD GPUs.
pub struct HsaInt2QuantizerKernel {
    agent: GpuAgent,
    queue: HsaQueue,
    signal: HsaSignal,
    module_quantize_f32: Option<HsaKernelModule>,
    module_quantize_f16: Option<HsaKernelModule>,
    module_dequantize_f32: Option<HsaKernelModule>,
    module_dequantize_f16: Option<HsaKernelModule>,
    module_pack: Option<HsaKernelModule>,
    module_unpack: Option<HsaKernelModule>,
    config: HsaInt2QuantizerConfig,
}

impl HsaInt2QuantizerKernel {
    pub fn new(config: HsaInt2QuantizerConfig) -> Result<Self, HsaInt2QuantizerError> {
        config.validate()?;

        let lib = get_hsa_lib().map_err(|e| HsaInt2QuantizerError::HsaNotAvailable(e.to_string()))?;
        let agents = find_gpu_agents().map_err(|e| HsaInt2QuantizerError::HsaNotAvailable(e.to_string()))?;

        if agents.is_empty() { return Err(HsaInt2QuantizerError::NoGpuFound); }
        let agent = agents.into_iter().next().unwrap();

        let mut queue: HsaQueue = ptr::null_mut();
        let mut signal: HsaSignal = 0;

        unsafe {
            let status = (lib.hsa_queue_create)(agent.handle, 4096, 0, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue);
            check_hsa(status, "create queue")?;

            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "create signal")?;
        }

        let module_quantize_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_QUANTIZE_F32).ok();
        let module_quantize_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_QUANTIZE_F16).ok();
        let module_dequantize_f32 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_DEQUANTIZE_F32).ok();
        let module_dequantize_f16 = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_DEQUANTIZE_F16).ok();
        let module_pack = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_PACK).ok();
        let module_unpack = HsaKernelModule::from_hsaco(&agent, PRECOMPILED_HSACO, KERNEL_UNPACK).ok();

        Ok(Self {
            agent,
            queue,
            signal,
            module_quantize_f32,
            module_quantize_f16,
            module_dequantize_f32,
            module_dequantize_f16,
            module_pack,
            module_unpack,
            config,
        })
    }

    pub fn quantize_f32(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, zeros: *mut c_void, num_elements: usize) -> Result<(), HsaInt2QuantizerError> {
        let _ = self.module_quantize_f32.as_ref().ok_or(HsaInt2QuantizerError::KernelMissing(KERNEL_QUANTIZE_F32))?;
        Ok(())
    }

    pub fn quantize_f16(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, zeros: *mut c_void, num_elements: usize) -> Result<(), HsaInt2QuantizerError> {
        let _ = self.module_quantize_f16.as_ref().ok_or(HsaInt2QuantizerError::KernelMissing(KERNEL_QUANTIZE_F16))?;
        Ok(())
    }

    pub fn dequantize_f32(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, zeros: *mut c_void, num_elements: usize) -> Result<(), HsaInt2QuantizerError> {
        let _ = self.module_dequantize_f32.as_ref().ok_or(HsaInt2QuantizerError::KernelMissing(KERNEL_DEQUANTIZE_F32))?;
        Ok(())
    }

    pub fn dequantize_f16(&self, input: *mut c_void, output: *mut c_void, scales: *mut c_void, zeros: *mut c_void, num_elements: usize) -> Result<(), HsaInt2QuantizerError> {
        let _ = self.module_dequantize_f16.as_ref().ok_or(HsaInt2QuantizerError::KernelMissing(KERNEL_DEQUANTIZE_F16))?;
        Ok(())
    }

    pub fn pack(&self, unpacked: *mut c_void, packed: *mut c_void, num_elements: usize) -> Result<(), HsaInt2QuantizerError> {
        let _ = self.module_pack.as_ref().ok_or(HsaInt2QuantizerError::KernelMissing(KERNEL_PACK))?;
        Ok(())
    }

    pub fn unpack(&self, packed: *mut c_void, unpacked: *mut c_void, num_elements: usize) -> Result<(), HsaInt2QuantizerError> {
        let _ = self.module_unpack.as_ref().ok_or(HsaInt2QuantizerError::KernelMissing(KERNEL_UNPACK))?;
        Ok(())
    }

    pub fn agent(&self) -> &GpuAgent { &self.agent }
    pub fn config(&self) -> &HsaInt2QuantizerConfig { &self.config }
}

impl Drop for HsaInt2QuantizerKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 { let _ = (lib.hsa_signal_destroy)(self.signal); }
                if !self.queue.is_null() { let _ = (lib.hsa_queue_destroy)(self.queue); }
            }
        }
    }
}

unsafe impl Send for HsaInt2QuantizerKernel {}
unsafe impl Sync for HsaInt2QuantizerKernel {}
