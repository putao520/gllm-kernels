//! HSA MoE Routing kernel wrapper for AMD GPUs.

use std::ffi::{c_void, CString};
use std::ptr;

use super::hsa_runtime::{
    get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HSA_STATUS_SUCCESS, HsaKernelDispatchPacket,
};
use super::hsa_flash_attn::{HsaQueueWrapper, HsaFlashAttentionError};

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/moe_route.hsaco");

/// HSA MoE Routing kernel error type.
#[derive(Debug, Clone)]
pub enum HsaMoeRouteError {
    HsaError(String),
    InvalidInput(String),
}

impl std::fmt::Display for HsaMoeRouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HsaError(s) => write!(f, "HSA error: {}", s),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for HsaMoeRouteError {}

#[repr(C)]
struct MoeRouteArgs {
    hidden_states_ptr: *const c_void,
    gate_weights_ptr: *const c_void,
    expert_indices_ptr: *mut c_void,
    expert_weights_ptr: *mut c_void,
    num_tokens: i32,
    hidden_size: i32,
    num_experts: i32,
    top_k: i32,
}

#[repr(C)]
struct ComputeLogitsArgs {
    hidden_states_ptr: *const c_void,
    gate_weights_ptr: *const c_void,
    logits_ptr: *mut c_void,
    num_tokens: i32,
    hidden_size: i32,
    num_experts: i32,
}

/// HSA MoE Routing kernel result.
pub struct HsaMoeRouteResult {
    pub expert_indices: Vec<u32>,
    pub expert_weights: Vec<f32>,
}

/// HSA MoE Routing kernel for AMD GPUs.
pub struct HsaMoeRouteKernel {
    agent: GpuAgent,
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    moe_route_f32_kernel: u64,
    moe_route_f16_kernel: u64,
    compute_logits_f32_kernel: u64,
    compute_logits_f16_kernel: u64,
    route_kernarg_size: u32,
    logits_kernarg_size: u32,
}

impl HsaMoeRouteKernel {
    /// Create a new HSA MoE routing kernel.
    pub fn new(agent: &GpuAgent) -> Result<Self, HsaMoeRouteError> {
        let lib = get_hsa_lib().map_err(|e| HsaMoeRouteError::HsaError(e.to_string()))?;

        // Check if HSACO is valid
        if PRECOMPILED_HSACO.len() < 100 {
            return Err(HsaMoeRouteError::HsaError(
                "MoE route HSACO not compiled (placeholder file)".to_string()
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
                return Err(HsaMoeRouteError::HsaError(
                    format!("Failed to create HSA reader: {}", get_error_string(status))
                ));
            }
        }

        // 2. Create executable
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaMoeRouteError::HsaError(
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
                return Err(HsaMoeRouteError::HsaError(
                    format!("Failed to load HSA code: {}", get_error_string(status))
                ));
            }
        }

        // 4. Freeze
        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }

        // 5. Get symbols
        let moe_route_f32_name = CString::new("moe_route_f32").unwrap();
        let moe_route_f16_name = CString::new("moe_route_f16").unwrap();
        let compute_logits_f32_name = CString::new("compute_routing_logits_f32").unwrap();
        let compute_logits_f16_name = CString::new("compute_routing_logits_f16").unwrap();

        let mut moe_route_f32_symbol: u64 = 0;
        let mut moe_route_f16_symbol: u64 = 0;
        let mut compute_logits_f32_symbol: u64 = 0;
        let mut compute_logits_f16_symbol: u64 = 0;

        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(
                executable, moe_route_f32_name.as_ptr(), &agent.handle, &mut moe_route_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, moe_route_f16_name.as_ptr(), &agent.handle, &mut moe_route_f16_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, compute_logits_f32_name.as_ptr(), &agent.handle, &mut compute_logits_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, compute_logits_f16_name.as_ptr(), &agent.handle, &mut compute_logits_f16_symbol
            );
        }

        if moe_route_f32_symbol == 0 {
            return Err(HsaMoeRouteError::HsaError(
                "MoE route kernel symbols not found in HSACO".to_string()
            ));
        }

        // 6. Get kernel objects
        let mut moe_route_f32_kernel: u64 = 0;
        let mut moe_route_f16_kernel: u64 = 0;
        let mut compute_logits_f32_kernel: u64 = 0;
        let mut compute_logits_f16_kernel: u64 = 0;
        let mut route_kernarg_size: u32 = 0;
        let mut logits_kernarg_size: u32 = 0;

        unsafe {
            (lib.hsa_executable_symbol_get_info)(
                moe_route_f32_symbol, 22, &mut moe_route_f32_kernel as *mut _ as *mut c_void
            );
            (lib.hsa_executable_symbol_get_info)(
                moe_route_f32_symbol, 23, &mut route_kernarg_size as *mut _ as *mut c_void
            );

            if moe_route_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    moe_route_f16_symbol, 22, &mut moe_route_f16_kernel as *mut _ as *mut c_void
                );
            }

            if compute_logits_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    compute_logits_f32_symbol, 22, &mut compute_logits_f32_kernel as *mut _ as *mut c_void
                );
                (lib.hsa_executable_symbol_get_info)(
                    compute_logits_f32_symbol, 23, &mut logits_kernarg_size as *mut _ as *mut c_void
                );
            }

            if compute_logits_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    compute_logits_f16_symbol, 22, &mut compute_logits_f16_kernel as *mut _ as *mut c_void
                );
            }
        }

        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            moe_route_f32_kernel,
            moe_route_f16_kernel,
            compute_logits_f32_kernel,
            compute_logits_f16_kernel,
            route_kernarg_size,
            logits_kernarg_size,
        })
    }

    /// Execute MoE routing on f32 data.
    pub fn moe_route_f32(
        &self,
        queue: &HsaQueueWrapper,
        hidden_states_ptr: *const c_void,
        gate_weights_ptr: *const c_void,
        expert_indices_ptr: *mut c_void,
        expert_weights_ptr: *mut c_void,
        num_tokens: usize,
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), HsaMoeRouteError> {
        let lib = get_hsa_lib().map_err(|e| HsaMoeRouteError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.route_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = MoeRouteArgs {
                hidden_states_ptr,
                gate_weights_ptr,
                expert_indices_ptr,
                expert_weights_ptr,
                num_tokens: num_tokens as i32,
                hidden_size: hidden_size as i32,
                num_experts: num_experts as i32,
                top_k: top_k as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut MoeRouteArgs, 1);
        }

        // Shared memory: logits + topk_vals + topk_indices
        // MAX_EXPERTS=64, MAX_TOPK=8
        let shared_mem_size = 64 * std::mem::size_of::<f32>()  // s_logits
            + 8 * std::mem::size_of::<f32>()  // s_topk_vals
            + 8 * std::mem::size_of::<i32>(); // s_topk_indices

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
                grid_size_x: num_tokens as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: shared_mem_size as u32,
                kernel_object: self.moe_route_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaMoeRouteError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute MoE routing on f16 hidden states with f32 gate weights.
    pub fn moe_route_f16(
        &self,
        queue: &HsaQueueWrapper,
        hidden_states_ptr: *const c_void,
        gate_weights_ptr: *const c_void,
        expert_indices_ptr: *mut c_void,
        expert_weights_ptr: *mut c_void,
        num_tokens: usize,
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), HsaMoeRouteError> {
        if self.moe_route_f16_kernel == 0 {
            return Err(HsaMoeRouteError::HsaError(
                "MoE route f16 kernel not available".to_string(),
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaMoeRouteError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.route_kernarg_size as usize,
                &mut kernarg_ptr,
            );

            let args = MoeRouteArgs {
                hidden_states_ptr,
                gate_weights_ptr,
                expert_indices_ptr,
                expert_weights_ptr,
                num_tokens: num_tokens as i32,
                hidden_size: hidden_size as i32,
                num_experts: num_experts as i32,
                top_k: top_k as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut MoeRouteArgs, 1);
        }

        // Shared memory: logits + topk_vals + topk_indices
        // MAX_EXPERTS=64, MAX_TOPK=8
        let shared_mem_size = 64 * std::mem::size_of::<f32>()  // s_logits
            + 8 * std::mem::size_of::<f32>()  // s_topk_vals
            + 8 * std::mem::size_of::<i32>(); // s_topk_indices

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8)
                .add((write_index % 4096) as usize * 64)
                as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1 << 0, // 1 Dimension
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: num_tokens as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: shared_mem_size as u32,
                kernel_object: self.moe_route_f16_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue
                .synchronize()
                .map_err(|e: HsaFlashAttentionError| HsaMoeRouteError::HsaError(e.to_string()))?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Compute routing logits only (for custom routing).
    pub fn compute_logits_f32(
        &self,
        queue: &HsaQueueWrapper,
        hidden_states_ptr: *const c_void,
        gate_weights_ptr: *const c_void,
        logits_ptr: *mut c_void,
        num_tokens: usize,
        hidden_size: usize,
        num_experts: usize,
    ) -> Result<(), HsaMoeRouteError> {
        if self.compute_logits_f32_kernel == 0 {
            return Err(HsaMoeRouteError::HsaError(
                "Compute logits kernel not available".to_string()
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaMoeRouteError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.logits_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = ComputeLogitsArgs {
                hidden_states_ptr,
                gate_weights_ptr,
                logits_ptr,
                num_tokens: num_tokens as i32,
                hidden_size: hidden_size as i32,
                num_experts: num_experts as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut ComputeLogitsArgs, 1);
        }

        let shared_mem_size = 64 * std::mem::size_of::<f32>(); // MAX_EXPERTS

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
                grid_size_x: num_tokens as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: shared_mem_size as u32,
                kernel_object: self.compute_logits_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaMoeRouteError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Check if f16 kernels are available.
    pub fn has_f16(&self) -> bool {
        self.moe_route_f16_kernel != 0
    }
}

impl Drop for HsaMoeRouteKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_executable_destroy)(self.executable);
                (lib.hsa_code_object_reader_destroy)(self.reader);
            }
        }
    }
}
