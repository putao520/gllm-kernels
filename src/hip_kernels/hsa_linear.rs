use std::ffi::{c_void, CString};
use std::ptr;

use super::hsa_runtime::{
    get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HSA_STATUS_SUCCESS,
};
use super::hsa_flash_attn::{HsaQueueWrapper, HsaFlashAttentionError};
use crate::kernel_types::LinearParams;

const KERNEL_NAME: &str = "linear_forward_kernel";
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/linear.hsaco");

#[repr(C)]
struct LinearArgs {
    input_ptr: *const c_void,
    weight_ptr: *const c_void,
    bias_ptr: *const c_void,
    output_ptr: *mut c_void,
    in_features: i32,
    out_features: i32,
    has_bias: i32,
}

pub struct HsaLinear {
    agent: GpuAgent,
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    kernel_object: u64,
    fused_kernel_object: u64,
    kernarg_size: u32,
    fused_kernarg_size: u32,
}

impl HsaLinear {
    pub fn new(agent: &GpuAgent) -> Result<Self, String> {
        let lib = get_hsa_lib().map_err(|e| e.to_string())?;

        // 1. Create reader
        let mut reader: HsaCodeObjectReader = 0;
        unsafe {
            let status = (lib.hsa_code_object_reader_create_from_memory)(
                PRECOMPILED_HSACO.as_ptr() as *const c_void,
                PRECOMPILED_HSACO.len(),
                &mut reader,
            );
            if status != HSA_STATUS_SUCCESS {
                return Err(format!("Failed to create HSA reader: {}", get_error_string(status)));
            }
        }

        // 2. Create executable
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(format!("Failed to create HSA executable: {}", get_error_string(status)));
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
                return Err(format!("Failed to load HSA code: {}", get_error_string(status)));
            }
        }

        // 4. Freeze
        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }

        // 5. Get symbols
        let name_c = CString::new(KERNEL_NAME).unwrap();
        let fused_name_c = CString::new("fused_gate_up_silu_kernel").unwrap();
        
        let mut symbol: u64 = 0;
        let mut fused_symbol: u64 = 0;
        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(executable, name_c.as_ptr(), &agent.handle, &mut symbol);
            (lib.hsa_executable_get_symbol_by_name)(executable, fused_name_c.as_ptr(), &agent.handle, &mut fused_symbol);
        }

        if symbol == 0 || fused_symbol == 0 {
            return Err("Linear kernel symbols not found in HSACO".to_string());
        }

        // 6. Get info
        let mut kernel_object: u64 = 0;
        let mut fused_kernel_object: u64 = 0;
        let mut kernarg_size: u32 = 0;
        let mut fused_kernarg_size: u32 = 0;
        unsafe {
            (lib.hsa_executable_symbol_get_info)(symbol, 22, &mut kernel_object as *mut _ as *mut c_void);
            (lib.hsa_executable_symbol_get_info)(symbol, 23, &mut kernarg_size as *mut _ as *mut c_void);
            (lib.hsa_executable_symbol_get_info)(fused_symbol, 22, &mut fused_kernel_object as *mut _ as *mut c_void);
            (lib.hsa_executable_symbol_get_info)(fused_symbol, 23, &mut fused_kernarg_size as *mut _ as *mut c_void);
        }

        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            kernel_object,
            fused_kernel_object,
            kernarg_size,
            fused_kernarg_size,
        })
    }

    pub fn forward(
        &self,
        queue: &HsaQueueWrapper,
        params: LinearParams,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        bias_ptr: Option<*const c_void>,
        output_ptr: *mut c_void,
        batch_size: usize,
    ) -> Result<(), String> {
        let lib = get_hsa_lib().map_err(|e| e.to_string())?;

        // Allocate kernargs
        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(self.agent.kernarg_region, self.kernarg_size as usize, &mut kernarg_ptr);
            
            let args = LinearArgs {
                input_ptr,
                weight_ptr,
                bias_ptr: bias_ptr.unwrap_or(ptr::null()),
                output_ptr,
                in_features: params.in_features as i32,
                out_features: params.out_features as i32,
                has_bias: if bias_ptr.is_some() { 1 } else { 0 },
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut LinearArgs, 1);
        }

        // Dispatch
        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64 // HSA_PACKET_SIZE
            ) as *mut crate::hip_kernels::hsa_runtime::HsaKernelDispatchPacket;

            let mut packet = crate::hip_kernels::hsa_runtime::HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11), // TYPE_KERNEL_DISPATCH, BARRIER, SYSTEM_SCOPE
                setup: 2, // 2D grid? Actually setup is dimension count - 1 if using standard header
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: params.out_features as u32,
                grid_size_y: batch_size as u32,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.kernel_object,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };
            
            // Adjust setup for 2D dispatch
            packet.setup = 2 << 0; // 2 Dimensions

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError| e.to_string())?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    pub fn fused_gate_up_silu(
        &self,
        queue: &HsaQueueWrapper,
        params: LinearParams,
        input_ptr: *const c_void,
        weight_gate_ptr: *const c_void,
        weight_up_ptr: *const c_void,
        output_ptr: *mut c_void,
        batch_size: usize,
    ) -> Result<(), String> {
        let lib = get_hsa_lib().map_err(|e| e.to_string())?;

        #[repr(C)]
        struct FusedLinearArgs {
            input_ptr: *const c_void,
            weight_gate_ptr: *const c_void,
            weight_up_ptr: *const c_void,
            output_ptr: *mut c_void,
            in_features: i32,
            out_features: i32,
        }

        // Allocate kernargs
        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(self.agent.kernarg_region, self.fused_kernarg_size as usize, &mut kernarg_ptr);
            
            let args = FusedLinearArgs {
                input_ptr,
                weight_gate_ptr,
                weight_up_ptr,
                output_ptr,
                in_features: params.in_features as i32,
                out_features: params.out_features as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut FusedLinearArgs, 1);
        }

        // Dispatch
        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64 // HSA_PACKET_SIZE
            ) as *mut crate::hip_kernels::hsa_runtime::HsaKernelDispatchPacket;

            let mut packet = crate::hip_kernels::hsa_runtime::HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11), // TYPE_KERNEL_DISPATCH, BARRIER, SYSTEM_SCOPE
                setup: 2 << 0, // 2 Dimensions
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: params.out_features as u32,
                grid_size_y: batch_size as u32,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.fused_kernel_object,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError| e.to_string())?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }
}

impl Drop for HsaLinear {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_executable_destroy)(self.executable);
                (lib.hsa_code_object_reader_destroy)(self.reader);
            }
        }
    }
}

/// Backward-compatible alias that matches the kernel naming convention used by other modules.
pub type HsaLinearKernel = HsaLinear;
