//! HSA SiLU (Swish) activation kernel wrapper for AMD GPUs.
use std::ffi::{c_void, CString};
use std::ptr;
use super::hsa_runtime::{
    get_error_string, get_hsa_lib, GpuAgent, HsaCodeObjectReader, HsaExecutable,
    HsaKernelDispatchPacket, HSA_STATUS_SUCCESS,
};
use super::hsa_flash_attn::{HsaFlashAttentionError, HsaQueueWrapper};
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/silu.hsaco");
const KERNEL_F32: &str = "silu_f32";
const KERNEL_INPLACE_F32: &str = "silu_inplace_f32";
#[derive(Debug, Clone)]
pub enum HsaSiluError {
    HsaError(String),
    InvalidInput(String),
}

impl std::fmt::Display for HsaSiluError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HsaError(msg) => write!(f, "HSA error: {}", msg),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for HsaSiluError {}
#[repr(C)]
struct SiluArgs {
    input_ptr: *const c_void,
    output_ptr: *mut c_void,
    len: i32,
}
#[repr(C)]
struct SiluInplaceArgs {
    data_ptr: *mut c_void,
    len: i32,
}
pub struct HsaSiluKernel {
    agent: GpuAgent,
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    silu_f32_kernel: u64,
    silu_inplace_f32_kernel: u64,
    silu_kernarg_size: u32,
    silu_inplace_kernarg_size: u32,
}

impl HsaSiluKernel {
    pub fn new(agent: &GpuAgent) -> Result<Self, HsaSiluError> {
        let lib = get_hsa_lib().map_err(|e| HsaSiluError::HsaError(e.to_string()))?;
        if PRECOMPILED_HSACO.len() < 100 {
            return Err(HsaSiluError::HsaError(
                "SiLU HSACO not compiled (placeholder file)".to_string(),
            ));
        }
        let mut reader: HsaCodeObjectReader = 0;
        unsafe {
            let status = (lib.hsa_code_object_reader_create_from_memory)(
                PRECOMPILED_HSACO.as_ptr() as *const c_void,
                PRECOMPILED_HSACO.len(),
                &mut reader,
            );
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaSiluError::HsaError(format!(
                    "Failed to create HSA reader: {}",
                    get_error_string(status)
                )));
            }
        }
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaSiluError::HsaError(format!(
                    "Failed to create HSA executable: {}",
                    get_error_string(status)
                )));
            }
        }
        unsafe {
            let status = (lib.hsa_executable_load_agent_code_object)(
                executable,
                agent.handle,
                reader,
                ptr::null(),
                ptr::null_mut(),
            );
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaSiluError::HsaError(format!(
                    "Failed to load HSA code: {}",
                    get_error_string(status)
                )));
            }
        }
        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }
        let silu_name = CString::new(KERNEL_F32).map_err(|_| {
            HsaSiluError::HsaError("Invalid SiLU kernel name".to_string())
        })?;
        let silu_inplace_name = CString::new(KERNEL_INPLACE_F32).map_err(|_| {
            HsaSiluError::HsaError("Invalid SiLU inplace kernel name".to_string())
        })?;
        let mut silu_symbol: u64 = 0;
        let mut silu_inplace_symbol: u64 = 0;
        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(
                executable,
                silu_name.as_ptr(),
                &agent.handle,
                &mut silu_symbol,
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable,
                silu_inplace_name.as_ptr(),
                &agent.handle,
                &mut silu_inplace_symbol,
            );
        }
        if silu_symbol == 0 || silu_inplace_symbol == 0 {
            return Err(HsaSiluError::HsaError(
                "SiLU kernel symbols not found in HSACO".to_string(),
            ));
        }
        let mut silu_f32_kernel: u64 = 0;
        let mut silu_inplace_f32_kernel: u64 = 0;
        let mut silu_kernarg_size: u32 = 0;
        let mut silu_inplace_kernarg_size: u32 = 0;
        unsafe {
            (lib.hsa_executable_symbol_get_info)(
                silu_symbol,
                22,
                &mut silu_f32_kernel as *mut _ as *mut c_void,
            );
            (lib.hsa_executable_symbol_get_info)(
                silu_symbol,
                23,
                &mut silu_kernarg_size as *mut _ as *mut c_void,
            );
            (lib.hsa_executable_symbol_get_info)(
                silu_inplace_symbol,
                22,
                &mut silu_inplace_f32_kernel as *mut _ as *mut c_void,
            );
            (lib.hsa_executable_symbol_get_info)(
                silu_inplace_symbol,
                23,
                &mut silu_inplace_kernarg_size as *mut _ as *mut c_void,
            );
        }
        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            silu_f32_kernel,
            silu_inplace_f32_kernel,
            silu_kernarg_size,
            silu_inplace_kernarg_size,
        })
    }
    pub fn forward(
        &self,
        queue: &HsaQueueWrapper,
        input_ptr: *const c_void,
        output_ptr: *mut c_void,
        len: usize,
    ) -> Result<(), HsaSiluError> {
        if len == 0 {
            return Ok(());
        }
        let len_i32 = i32::try_from(len)
            .map_err(|_| HsaSiluError::InvalidInput("len exceeds i32".to_string()))?;
        let args = SiluArgs {
            input_ptr,
            output_ptr,
            len: len_i32,
        };
        self.dispatch_kernel(queue, self.silu_f32_kernel, self.silu_kernarg_size, &args, len)
    }
    pub fn forward_inplace(
        &self,
        queue: &HsaQueueWrapper,
        data_ptr: *mut c_void,
        len: usize,
    ) -> Result<(), HsaSiluError> {
        if len == 0 {
            return Ok(());
        }
        let len_i32 = i32::try_from(len)
            .map_err(|_| HsaSiluError::InvalidInput("len exceeds i32".to_string()))?;
        let args = SiluInplaceArgs {
            data_ptr,
            len: len_i32,
        };
        self.dispatch_kernel(
            queue,
            self.silu_inplace_f32_kernel,
            self.silu_inplace_kernarg_size,
            &args,
            len,
        )
    }
    fn dispatch_kernel<TArgs>(
        &self,
        queue: &HsaQueueWrapper,
        kernel_object: u64,
        kernarg_size: u32,
        args: &TArgs,
        len: usize,
    ) -> Result<(), HsaSiluError> {
        let lib = get_hsa_lib().map_err(|e| HsaSiluError::HsaError(e.to_string()))?;
        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                kernarg_size as usize,
                &mut kernarg_ptr,
            );
            ptr::copy_nonoverlapping(args, kernarg_ptr as *mut TArgs, 1);
        }
        let workgroup_size = 256u32;
        let grid_size = ((len + workgroup_size as usize - 1) / workgroup_size as usize)
            as u32
            * workgroup_size;
        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8)
                .add((write_index % 4096) as usize * 64)
                as *mut HsaKernelDispatchPacket;
            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1,
                workgroup_size_x: workgroup_size as u16,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid_size,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };
            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);
            queue
                .synchronize()
                .map_err(|e: HsaFlashAttentionError| HsaSiluError::HsaError(e.to_string()))?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }
        Ok(())
    }
}
impl Drop for HsaSiluKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_executable_destroy)(self.executable);
                (lib.hsa_code_object_reader_destroy)(self.reader);
            }
        }
    }
}
