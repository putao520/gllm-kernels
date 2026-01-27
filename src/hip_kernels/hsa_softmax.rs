//! HSA Softmax kernel wrapper for AMD GPUs.

use std::ffi::{c_void, CString};
use std::ptr;

use super::hsa_runtime::{
    get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HSA_STATUS_SUCCESS, HsaKernelDispatchPacket,
};
use super::hsa_flash_attn::{HsaQueueWrapper, HsaFlashAttentionError};

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/softmax.hsaco");

/// HSA Softmax kernel error type.
#[derive(Debug, Clone)]
pub enum HsaSoftmaxError {
    HsaError(String),
    InvalidInput(String),
}

impl std::fmt::Display for HsaSoftmaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HsaError(s) => write!(f, "HSA error: {}", s),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for HsaSoftmaxError {}

#[repr(C)]
struct SoftmaxArgs {
    input_ptr: *const c_void,
    output_ptr: *mut c_void,
    num_rows: i32,
    row_size: i32,
}

#[repr(C)]
struct SoftmaxInplaceArgs {
    data_ptr: *mut c_void,
    num_rows: i32,
    row_size: i32,
}

/// HSA Softmax kernel for AMD GPUs.
pub struct HsaSoftmaxKernel {
    agent: GpuAgent,
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    softmax_f32_kernel: u64,
    softmax_f16_kernel: u64,
    softmax_inplace_f32_kernel: u64,
    kernarg_size: u32,
    inplace_kernarg_size: u32,
}

impl HsaSoftmaxKernel {
    /// Create a new HSA softmax kernel.
    pub fn new(agent: &GpuAgent) -> Result<Self, HsaSoftmaxError> {
        let lib = get_hsa_lib().map_err(|e| HsaSoftmaxError::HsaError(e.to_string()))?;

        // Check if HSACO is valid (not a placeholder)
        if PRECOMPILED_HSACO.len() < 100 {
            return Err(HsaSoftmaxError::HsaError(
                "Softmax HSACO not compiled (placeholder file)".to_string()
            ));
        }

        // 1. Create reader
        let mut reader: HsaCodeObjectReader = 0;
        unsafe {
            let status = (lib.hsa_code_object_reader_create_from_memory)(
                PRECOMPILED_HSACO.as_ptr() as *const c_void,
                PRECOMPILED_HSACO.len(),
                &mut reader,
            );
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaSoftmaxError::HsaError(
                    format!("Failed to create HSA reader: {}", get_error_string(status))
                ));
            }
        }

        // 2. Create executable
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaSoftmaxError::HsaError(
                    format!("Failed to create HSA executable: {}", get_error_string(status))
                ));
            }
        }

        // 3. Load code
        unsafe {
            let status = (lib.hsa_executable_load_agent_code_object)(
                executable,
                agent.handle,
                reader,
                ptr::null(),
                ptr::null_mut(),
            );
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaSoftmaxError::HsaError(
                    format!("Failed to load HSA code: {}", get_error_string(status))
                ));
            }
        }

        // 4. Freeze
        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }

        // 5. Get symbols
        let softmax_f32_name = CString::new("softmax_f32").unwrap();
        let softmax_f16_name = CString::new("softmax_f16").unwrap();
        let softmax_inplace_f32_name = CString::new("softmax_inplace_f32").unwrap();

        let mut softmax_f32_symbol: u64 = 0;
        let mut softmax_f16_symbol: u64 = 0;
        let mut softmax_inplace_f32_symbol: u64 = 0;

        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(
                executable, softmax_f32_name.as_ptr(), &agent.handle, &mut softmax_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, softmax_f16_name.as_ptr(), &agent.handle, &mut softmax_f16_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, softmax_inplace_f32_name.as_ptr(), &agent.handle, &mut softmax_inplace_f32_symbol
            );
        }

        if softmax_f32_symbol == 0 {
            return Err(HsaSoftmaxError::HsaError(
                "Softmax kernel symbols not found in HSACO".to_string()
            ));
        }

        // 6. Get kernel objects
        let mut softmax_f32_kernel: u64 = 0;
        let mut softmax_f16_kernel: u64 = 0;
        let mut softmax_inplace_f32_kernel: u64 = 0;
        let mut kernarg_size: u32 = 0;
        let mut inplace_kernarg_size: u32 = 0;

        unsafe {
            (lib.hsa_executable_symbol_get_info)(
                softmax_f32_symbol, 22, &mut softmax_f32_kernel as *mut _ as *mut c_void
            );
            (lib.hsa_executable_symbol_get_info)(
                softmax_f32_symbol, 23, &mut kernarg_size as *mut _ as *mut c_void
            );

            if softmax_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    softmax_f16_symbol, 22, &mut softmax_f16_kernel as *mut _ as *mut c_void
                );
            }

            if softmax_inplace_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    softmax_inplace_f32_symbol, 22, &mut softmax_inplace_f32_kernel as *mut _ as *mut c_void
                );
                (lib.hsa_executable_symbol_get_info)(
                    softmax_inplace_f32_symbol, 23, &mut inplace_kernarg_size as *mut _ as *mut c_void
                );
            }
        }

        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            softmax_f32_kernel,
            softmax_f16_kernel,
            softmax_inplace_f32_kernel,
            kernarg_size,
            inplace_kernarg_size,
        })
    }

    /// Execute softmax on f32 data.
    pub fn softmax_f32(
        &self,
        queue: &HsaQueueWrapper,
        input_ptr: *const c_void,
        output_ptr: *mut c_void,
        num_rows: usize,
        row_size: usize,
    ) -> Result<(), HsaSoftmaxError> {
        let lib = get_hsa_lib().map_err(|e| HsaSoftmaxError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = SoftmaxArgs {
                input_ptr,
                output_ptr,
                num_rows: num_rows as i32,
                row_size: row_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut SoftmaxArgs, 1);
        }

        // Dispatch
        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64
            ) as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1 << 0, // 1 Dimension
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: num_rows as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 1024, // Shared memory for warp reduction
                kernel_object: self.softmax_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaSoftmaxError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute softmax on f16 data.
    pub fn softmax_f16(
        &self,
        queue: &HsaQueueWrapper,
        input_ptr: *const c_void,
        output_ptr: *mut c_void,
        num_rows: usize,
        row_size: usize,
    ) -> Result<(), HsaSoftmaxError> {
        if self.softmax_f16_kernel == 0 {
            return Err(HsaSoftmaxError::HsaError(
                "f16 softmax kernel not available".to_string(),
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaSoftmaxError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.kernarg_size as usize,
                &mut kernarg_ptr,
            );

            let args = SoftmaxArgs {
                input_ptr,
                output_ptr,
                num_rows: num_rows as i32,
                row_size: row_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut SoftmaxArgs, 1);
        }

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr =
                (queue.queue() as *mut u8).add((write_index % 4096) as usize * 64)
                    as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1 << 0,
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: num_rows as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 1024,
                kernel_object: self.softmax_f16_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue
                .synchronize()
                .map_err(|e: HsaFlashAttentionError| HsaSoftmaxError::HsaError(e.to_string()))?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute softmax in-place on f32 data.
    pub fn softmax_inplace_f32(
        &self,
        queue: &HsaQueueWrapper,
        data_ptr: *mut c_void,
        num_rows: usize,
        row_size: usize,
    ) -> Result<(), HsaSoftmaxError> {
        if self.softmax_inplace_f32_kernel == 0 {
            return Err(HsaSoftmaxError::HsaError(
                "Inplace softmax kernel not available".to_string()
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaSoftmaxError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.inplace_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = SoftmaxInplaceArgs {
                data_ptr,
                num_rows: num_rows as i32,
                row_size: row_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut SoftmaxInplaceArgs, 1);
        }

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64
            ) as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1 << 0,
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: num_rows as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 1024,
                kernel_object: self.softmax_inplace_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaSoftmaxError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Check if f16 kernel is available.
    pub fn has_f16(&self) -> bool {
        self.softmax_f16_kernel != 0
    }
}

impl Drop for HsaSoftmaxKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_executable_destroy)(self.executable);
                (lib.hsa_code_object_reader_destroy)(self.reader);
            }
        }
    }
}
