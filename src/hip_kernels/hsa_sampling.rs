//! HSA Sampling kernels (argmax, topk) wrapper for AMD GPUs.

use std::ffi::{c_void, CString};
use std::ptr;

use super::hsa_runtime::{
    get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HSA_STATUS_SUCCESS, HsaKernelDispatchPacket,
};
use super::hsa_flash_attn::{HsaQueueWrapper, HsaFlashAttentionError};

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/sampling.hsaco");

/// HSA Sampling kernel error type.
#[derive(Debug, Clone)]
pub enum HsaSamplingError {
    HsaError(String),
    InvalidInput(String),
}

impl std::fmt::Display for HsaSamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HsaError(s) => write!(f, "HSA error: {}", s),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for HsaSamplingError {}

#[repr(C)]
struct ArgmaxArgs {
    logits_ptr: *const c_void,
    indices_ptr: *mut c_void,
    batch_size: i32,
    vocab_size: i32,
}

#[repr(C)]
struct TopkArgs {
    logits_ptr: *const c_void,
    indices_ptr: *mut c_void,
    values_ptr: *mut c_void,
    batch_size: i32,
    vocab_size: i32,
    k: i32,
}

/// HSA Sampling kernel for AMD GPUs (argmax, topk).
pub struct HsaSamplingKernel {
    agent: GpuAgent,
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    argmax_f32_kernel: u64,
    argmax_f16_kernel: u64,
    topk_f32_kernel: u64,
    topk_f16_kernel: u64,
    argmax_kernarg_size: u32,
    topk_kernarg_size: u32,
}

impl HsaSamplingKernel {
    /// Create a new HSA sampling kernel.
    pub fn new(agent: &GpuAgent) -> Result<Self, HsaSamplingError> {
        let lib = get_hsa_lib().map_err(|e| HsaSamplingError::HsaError(e.to_string()))?;

        // Check if HSACO is valid
        if PRECOMPILED_HSACO.len() < 100 {
            return Err(HsaSamplingError::HsaError(
                "Sampling HSACO not compiled (placeholder file)".to_string()
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
                return Err(HsaSamplingError::HsaError(
                    format!("Failed to create HSA reader: {}", get_error_string(status))
                ));
            }
        }

        // 2. Create executable
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaSamplingError::HsaError(
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
                return Err(HsaSamplingError::HsaError(
                    format!("Failed to load HSA code: {}", get_error_string(status))
                ));
            }
        }

        // 4. Freeze
        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }

        // 5. Get symbols
        let argmax_f32_name = CString::new("argmax_f32").unwrap();
        let argmax_f16_name = CString::new("argmax_f16").unwrap();
        let topk_f32_name = CString::new("topk_f32").unwrap();
        let topk_f16_name = CString::new("topk_f16").unwrap();

        let mut argmax_f32_symbol: u64 = 0;
        let mut argmax_f16_symbol: u64 = 0;
        let mut topk_f32_symbol: u64 = 0;
        let mut topk_f16_symbol: u64 = 0;

        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(
                executable, argmax_f32_name.as_ptr(), &agent.handle, &mut argmax_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, argmax_f16_name.as_ptr(), &agent.handle, &mut argmax_f16_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, topk_f32_name.as_ptr(), &agent.handle, &mut topk_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, topk_f16_name.as_ptr(), &agent.handle, &mut topk_f16_symbol
            );
        }

        if argmax_f32_symbol == 0 {
            return Err(HsaSamplingError::HsaError(
                "Sampling kernel symbols not found in HSACO".to_string()
            ));
        }

        // 6. Get kernel objects
        let mut argmax_f32_kernel: u64 = 0;
        let mut argmax_f16_kernel: u64 = 0;
        let mut topk_f32_kernel: u64 = 0;
        let mut topk_f16_kernel: u64 = 0;
        let mut argmax_kernarg_size: u32 = 0;
        let mut topk_kernarg_size: u32 = 0;

        unsafe {
            (lib.hsa_executable_symbol_get_info)(
                argmax_f32_symbol, 22, &mut argmax_f32_kernel as *mut _ as *mut c_void
            );
            (lib.hsa_executable_symbol_get_info)(
                argmax_f32_symbol, 23, &mut argmax_kernarg_size as *mut _ as *mut c_void
            );

            if argmax_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    argmax_f16_symbol, 22, &mut argmax_f16_kernel as *mut _ as *mut c_void
                );
            }

            if topk_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    topk_f32_symbol, 22, &mut topk_f32_kernel as *mut _ as *mut c_void
                );
                (lib.hsa_executable_symbol_get_info)(
                    topk_f32_symbol, 23, &mut topk_kernarg_size as *mut _ as *mut c_void
                );
            }

            if topk_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    topk_f16_symbol, 22, &mut topk_f16_kernel as *mut _ as *mut c_void
                );
            }
        }

        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            argmax_f32_kernel,
            argmax_f16_kernel,
            topk_f32_kernel,
            topk_f16_kernel,
            argmax_kernarg_size,
            topk_kernarg_size,
        })
    }

    /// Execute argmax on f32 logits.
    pub fn argmax_f32(
        &self,
        queue: &HsaQueueWrapper,
        logits_ptr: *const c_void,
        indices_ptr: *mut c_void,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<(), HsaSamplingError> {
        let lib = get_hsa_lib().map_err(|e| HsaSamplingError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.argmax_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = ArgmaxArgs {
                logits_ptr,
                indices_ptr,
                batch_size: batch_size as i32,
                vocab_size: vocab_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut ArgmaxArgs, 1);
        }

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
                grid_size_x: batch_size as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 256, // Shared memory for warp reduction
                kernel_object: self.argmax_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaSamplingError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute argmax on f16 logits.
    pub fn argmax_f16(
        &self,
        queue: &HsaQueueWrapper,
        logits_ptr: *const c_void,
        indices_ptr: *mut c_void,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<(), HsaSamplingError> {
        if self.argmax_f16_kernel == 0 {
            return Err(HsaSamplingError::HsaError(
                "Argmax f16 kernel not available".to_string(),
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaSamplingError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.argmax_kernarg_size as usize,
                &mut kernarg_ptr,
            );

            let args = ArgmaxArgs {
                logits_ptr,
                indices_ptr,
                batch_size: batch_size as i32,
                vocab_size: vocab_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut ArgmaxArgs, 1);
        }

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr =
                (queue.queue() as *mut u8).add((write_index % 4096) as usize * 64)
                    as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1 << 0, // 1 Dimension
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: batch_size as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 256, // Shared memory for warp reduction
                kernel_object: self.argmax_f16_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue
                .synchronize()
                .map_err(|e: HsaFlashAttentionError| HsaSamplingError::HsaError(e.to_string()))?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute topk on f32 logits.
    pub fn topk_f32(
        &self,
        queue: &HsaQueueWrapper,
        logits_ptr: *const c_void,
        indices_ptr: *mut c_void,
        values_ptr: *mut c_void,
        batch_size: usize,
        vocab_size: usize,
        k: usize,
    ) -> Result<(), HsaSamplingError> {
        if self.topk_f32_kernel == 0 {
            return Err(HsaSamplingError::HsaError(
                "TopK kernel not available".to_string()
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaSamplingError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.topk_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = TopkArgs {
                logits_ptr,
                indices_ptr,
                values_ptr,
                batch_size: batch_size as i32,
                vocab_size: vocab_size as i32,
                k: k as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut TopkArgs, 1);
        }

        // Calculate shared memory size: each thread needs k floats + k ints
        let shared_mem_size = 256 * k * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>());

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
                grid_size_x: batch_size as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: shared_mem_size as u32,
                kernel_object: self.topk_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaSamplingError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute topk on f16 logits.
    pub fn topk_f16(
        &self,
        queue: &HsaQueueWrapper,
        logits_ptr: *const c_void,
        indices_ptr: *mut c_void,
        values_ptr: *mut c_void,
        batch_size: usize,
        vocab_size: usize,
        k: usize,
    ) -> Result<(), HsaSamplingError> {
        if self.topk_f16_kernel == 0 {
            return Err(HsaSamplingError::HsaError(
                "TopK f16 kernel not available".to_string(),
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaSamplingError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.topk_kernarg_size as usize,
                &mut kernarg_ptr,
            );

            let args = TopkArgs {
                logits_ptr,
                indices_ptr,
                values_ptr,
                batch_size: batch_size as i32,
                vocab_size: vocab_size as i32,
                k: k as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut TopkArgs, 1);
        }

        // Calculate shared memory size: each thread needs k floats + k ints
        let shared_mem_size =
            256 * k * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>());

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
                grid_size_x: batch_size as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: shared_mem_size as u32,
                kernel_object: self.topk_f16_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue
                .synchronize()
                .map_err(|e: HsaFlashAttentionError| HsaSamplingError::HsaError(e.to_string()))?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Check if f16 kernels are available.
    pub fn has_f16(&self) -> bool {
        self.argmax_f16_kernel != 0
    }

    /// Check if topk is available.
    pub fn has_topk(&self) -> bool {
        self.topk_f32_kernel != 0
    }
}

impl Drop for HsaSamplingKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_executable_destroy)(self.executable);
                (lib.hsa_code_object_reader_destroy)(self.reader);
            }
        }
    }
}
