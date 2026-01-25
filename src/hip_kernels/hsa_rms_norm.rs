use std::ffi::{c_void, CString};
use std::ptr;

use super::hsa_runtime::{
    get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader, HsaExecutable,
    HSA_STATUS_SUCCESS,
};
use super::hsa_flash_attn::HsaQueueWrapper;

const KERNEL_NAME: &str = "rms_norm_f32";
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/rms_norm.hsaco");

#[repr(C)]
struct RmsNormArgs {
    input_ptr: *const c_void,
    weight_ptr: *const c_void,
    output_ptr: *mut c_void,
    hidden: i32,
    rows: i32,
    eps: f32,
}

pub struct HsaRmsNormKernel {
    agent: GpuAgent,
    #[allow(dead_code)]
    executable: HsaExecutable,
    #[allow(dead_code)]
    reader: HsaCodeObjectReader,
    kernel_object: u64,
    kernarg_size: u32,
}

impl HsaRmsNormKernel {
    pub fn new(agent: &GpuAgent) -> Result<Self, String> {
        let lib = get_hsa_lib().map_err(|e| e.to_string())?;

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

        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(format!("Failed to create HSA executable: {}", get_error_string(status)));
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
                return Err(format!("Failed to load HSA code: {}", get_error_string(status)));
            }
        }

        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }

        let name_c = CString::new(KERNEL_NAME).map_err(|_| "Invalid kernel name".to_string())?;
        let mut symbol: u64 = 0;
        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(
                executable,
                name_c.as_ptr(),
                &agent.handle,
                &mut symbol,
            );
        }

        if symbol == 0 {
            return Err("RMSNorm kernel symbol not found in HSACO".to_string());
        }

        let mut kernel_object: u64 = 0;
        let mut kernarg_size: u32 = 0;
        unsafe {
            (lib.hsa_executable_symbol_get_info)(symbol, 22, &mut kernel_object as *mut _ as *mut c_void);
            (lib.hsa_executable_symbol_get_info)(symbol, 23, &mut kernarg_size as *mut _ as *mut c_void);
        }

        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            kernel_object,
            kernarg_size,
        })
    }

    pub fn forward(
        &self,
        queue: &HsaQueueWrapper,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        output_ptr: *mut c_void,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String> {
        if rows == 0 || hidden == 0 {
            return Ok(());
        }

        let rows_i32 = i32::try_from(rows).map_err(|_| "rows exceeds i32".to_string())?;
        let hidden_i32 = i32::try_from(hidden).map_err(|_| "hidden exceeds i32".to_string())?;

        let lib = get_hsa_lib().map_err(|e| e.to_string())?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(self.agent.kernarg_region, self.kernarg_size as usize, &mut kernarg_ptr);

            let args = RmsNormArgs {
                input_ptr,
                weight_ptr,
                output_ptr,
                hidden: hidden_i32,
                rows: rows_i32,
                eps,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut RmsNormArgs, 1);
        }

        let workgroup_size = 256u32;
        let grid_size = ((rows + workgroup_size as usize - 1) / workgroup_size as usize) as u32
            * workgroup_size;

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64
            ) as *mut crate::hip_kernels::hsa_runtime::HsaKernelDispatchPacket;

            let packet = crate::hip_kernels::hsa_runtime::HsaKernelDispatchPacket {
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
                kernel_object: self.kernel_object,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e| e.to_string())?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }
}
