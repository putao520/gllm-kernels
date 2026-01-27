//! HSA RoPE (Rotary Position Embedding) kernel wrapper for AMD GPUs.

use std::ffi::{c_void, CString};
use std::ptr;

use super::hsa_runtime::{
    get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HSA_STATUS_SUCCESS, HsaKernelDispatchPacket,
};
use super::hsa_flash_attn::{HsaQueueWrapper, HsaFlashAttentionError};

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/rope.hsaco");

/// HSA RoPE kernel error type.
#[derive(Debug, Clone)]
pub enum HsaRoPEError {
    HsaError(String),
    InvalidInput(String),
}

impl std::fmt::Display for HsaRoPEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HsaError(s) => write!(f, "HSA error: {}", s),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for HsaRoPEError {}

#[repr(C)]
struct RoPEApplyArgs {
    q_ptr: *const c_void,
    k_ptr: *const c_void,
    cos_cache_ptr: *const c_void,
    sin_cache_ptr: *const c_void,
    q_out_ptr: *mut c_void,
    k_out_ptr: *mut c_void,
    batch_size: i32,
    seq_len: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    position_offset: i32,
}

#[repr(C)]
struct RoPEInplaceArgs {
    x_ptr: *mut c_void,
    cos_cache_ptr: *const c_void,
    sin_cache_ptr: *const c_void,
    batch_size: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    position_offset: i32,
}

/// HSA RoPE kernel for AMD GPUs.
pub struct HsaRoPEKernel {
    agent: GpuAgent,
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    rope_apply_f32_kernel: u64,
    rope_apply_f16_kernel: u64,
    rope_inplace_f32_kernel: u64,
    rope_inplace_f16_kernel: u64,
    apply_kernarg_size: u32,
    inplace_kernarg_size: u32,
}

impl HsaRoPEKernel {
    /// Create a new HSA RoPE kernel.
    pub fn new(agent: &GpuAgent) -> Result<Self, HsaRoPEError> {
        let lib = get_hsa_lib().map_err(|e| HsaRoPEError::HsaError(e.to_string()))?;

        // Check if HSACO is valid
        if PRECOMPILED_HSACO.len() < 100 {
            return Err(HsaRoPEError::HsaError(
                "RoPE HSACO not compiled (placeholder file)".to_string()
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
                return Err(HsaRoPEError::HsaError(
                    format!("Failed to create HSA reader: {}", get_error_string(status))
                ));
            }
        }

        // 2. Create executable
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaRoPEError::HsaError(
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
                return Err(HsaRoPEError::HsaError(
                    format!("Failed to load HSA code: {}", get_error_string(status))
                ));
            }
        }

        // 4. Freeze
        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }

        // 5. Get symbols
        let rope_apply_f32_name = CString::new("rope_apply_f32").unwrap();
        let rope_apply_f16_name = CString::new("rope_apply_f16").unwrap();
        let rope_inplace_f32_name = CString::new("rope_apply_inplace_f32").unwrap();
        let rope_inplace_f16_name = CString::new("rope_apply_inplace_f16").unwrap();

        let mut rope_apply_f32_symbol: u64 = 0;
        let mut rope_apply_f16_symbol: u64 = 0;
        let mut rope_inplace_f32_symbol: u64 = 0;
        let mut rope_inplace_f16_symbol: u64 = 0;

        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(
                executable, rope_apply_f32_name.as_ptr(), &agent.handle, &mut rope_apply_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, rope_apply_f16_name.as_ptr(), &agent.handle, &mut rope_apply_f16_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, rope_inplace_f32_name.as_ptr(), &agent.handle, &mut rope_inplace_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, rope_inplace_f16_name.as_ptr(), &agent.handle, &mut rope_inplace_f16_symbol
            );
        }

        if rope_apply_f32_symbol == 0 {
            return Err(HsaRoPEError::HsaError(
                "RoPE kernel symbols not found in HSACO".to_string()
            ));
        }

        // 6. Get kernel objects
        let mut rope_apply_f32_kernel: u64 = 0;
        let mut rope_apply_f16_kernel: u64 = 0;
        let mut rope_inplace_f32_kernel: u64 = 0;
        let mut rope_inplace_f16_kernel: u64 = 0;
        let mut apply_kernarg_size: u32 = 0;
        let mut inplace_kernarg_size: u32 = 0;

        unsafe {
            (lib.hsa_executable_symbol_get_info)(
                rope_apply_f32_symbol, 22, &mut rope_apply_f32_kernel as *mut _ as *mut c_void
            );
            (lib.hsa_executable_symbol_get_info)(
                rope_apply_f32_symbol, 23, &mut apply_kernarg_size as *mut _ as *mut c_void
            );

            if rope_apply_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    rope_apply_f16_symbol, 22, &mut rope_apply_f16_kernel as *mut _ as *mut c_void
                );
            }

            if rope_inplace_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    rope_inplace_f32_symbol, 22, &mut rope_inplace_f32_kernel as *mut _ as *mut c_void
                );
                (lib.hsa_executable_symbol_get_info)(
                    rope_inplace_f32_symbol, 23, &mut inplace_kernarg_size as *mut _ as *mut c_void
                );
            }

            if rope_inplace_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    rope_inplace_f16_symbol, 22, &mut rope_inplace_f16_kernel as *mut _ as *mut c_void
                );
            }
        }

        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            rope_apply_f32_kernel,
            rope_apply_f16_kernel,
            rope_inplace_f32_kernel,
            rope_inplace_f16_kernel,
            apply_kernarg_size,
            inplace_kernarg_size,
        })
    }

    /// Apply RoPE to Q and K tensors (out-of-place).
    pub fn rope_apply_f32(
        &self,
        queue: &HsaQueueWrapper,
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        cos_cache_ptr: *const c_void,
        sin_cache_ptr: *const c_void,
        q_out_ptr: *mut c_void,
        k_out_ptr: *mut c_void,
        batch_size: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), HsaRoPEError> {
        let lib = get_hsa_lib().map_err(|e| HsaRoPEError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.apply_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = RoPEApplyArgs {
                q_ptr,
                k_ptr,
                cos_cache_ptr,
                sin_cache_ptr,
                q_out_ptr,
                k_out_ptr,
                batch_size: batch_size as i32,
                seq_len: seq_len as i32,
                num_q_heads: num_q_heads as i32,
                num_kv_heads: num_kv_heads as i32,
                head_dim: head_dim as i32,
                position_offset: position_offset as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut RoPEApplyArgs, 1);
        }

        // Calculate grid size
        let half_head_dim = head_dim / 2;
        let total_q_elements = batch_size * seq_len * num_q_heads * half_head_dim;
        let total_kv_elements = batch_size * seq_len * num_kv_heads * half_head_dim;
        let total_elements = total_q_elements.max(total_kv_elements);
        let num_blocks = (total_elements + 255) / 256;

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
                grid_size_x: (num_blocks * 256) as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.rope_apply_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaRoPEError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Apply RoPE to Q and K tensors (out-of-place) on f16 data.
    pub fn rope_apply_f16(
        &self,
        queue: &HsaQueueWrapper,
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        cos_cache_ptr: *const c_void,
        sin_cache_ptr: *const c_void,
        q_out_ptr: *mut c_void,
        k_out_ptr: *mut c_void,
        batch_size: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), HsaRoPEError> {
        if self.rope_apply_f16_kernel == 0 {
            return Err(HsaRoPEError::HsaError(
                "f16 RoPE apply kernel not available".to_string(),
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaRoPEError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.apply_kernarg_size as usize,
                &mut kernarg_ptr,
            );

            let args = RoPEApplyArgs {
                q_ptr,
                k_ptr,
                cos_cache_ptr,
                sin_cache_ptr,
                q_out_ptr,
                k_out_ptr,
                batch_size: batch_size as i32,
                seq_len: seq_len as i32,
                num_q_heads: num_q_heads as i32,
                num_kv_heads: num_kv_heads as i32,
                head_dim: head_dim as i32,
                position_offset: position_offset as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut RoPEApplyArgs, 1);
        }

        let half_head_dim = head_dim / 2;
        let total_q_elements = batch_size * seq_len * num_q_heads * half_head_dim;
        let total_kv_elements = batch_size * seq_len * num_kv_heads * half_head_dim;
        let total_elements = total_q_elements.max(total_kv_elements);
        let num_blocks = (total_elements + 255) / 256;

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
                grid_size_x: (num_blocks * 256) as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.rope_apply_f16_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue
                .synchronize()
                .map_err(|e: HsaFlashAttentionError| HsaRoPEError::HsaError(e.to_string()))?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Apply RoPE in-place to a single tensor.
    pub fn rope_apply_inplace_f32(
        &self,
        queue: &HsaQueueWrapper,
        x_ptr: *mut c_void,
        cos_cache_ptr: *const c_void,
        sin_cache_ptr: *const c_void,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), HsaRoPEError> {
        if self.rope_inplace_f32_kernel == 0 {
            return Err(HsaRoPEError::HsaError(
                "Inplace RoPE kernel not available".to_string()
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaRoPEError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.inplace_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = RoPEInplaceArgs {
                x_ptr,
                cos_cache_ptr,
                sin_cache_ptr,
                batch_size: batch_size as i32,
                seq_len: seq_len as i32,
                num_heads: num_heads as i32,
                head_dim: head_dim as i32,
                position_offset: position_offset as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut RoPEInplaceArgs, 1);
        }

        let half_head_dim = head_dim / 2;
        let total_elements = batch_size * seq_len * num_heads * half_head_dim;
        let num_blocks = (total_elements + 255) / 256;

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
                grid_size_x: (num_blocks * 256) as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.rope_inplace_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaRoPEError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Check if f16 kernels are available.
    pub fn has_f16(&self) -> bool {
        self.rope_apply_f16_kernel != 0
    }

    /// Check if inplace kernels are available.
    pub fn has_inplace(&self) -> bool {
        self.rope_inplace_f32_kernel != 0
    }
}

impl Drop for HsaRoPEKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_executable_destroy)(self.executable);
                (lib.hsa_code_object_reader_destroy)(self.reader);
            }
        }
    }
}
