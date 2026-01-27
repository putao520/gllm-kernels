//! HSA Quantized matmul kernels (Q4, Q8, AWQ) wrapper for AMD GPUs.

use std::ffi::{c_void, CString};
use std::ptr;

use super::hsa_runtime::{
    get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HSA_STATUS_SUCCESS, HsaKernelDispatchPacket,
};
use super::hsa_flash_attn::{HsaQueueWrapper, HsaFlashAttentionError};

const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/quantized.hsaco");

/// HSA Quantized kernel error type.
#[derive(Debug, Clone)]
pub enum HsaQuantizedError {
    HsaError(String),
    InvalidInput(String),
}

impl std::fmt::Display for HsaQuantizedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HsaError(s) => write!(f, "HSA error: {}", s),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for HsaQuantizedError {}

#[repr(C)]
struct Q4MatmulArgs {
    input_ptr: *const c_void,
    weight_ptr: *const c_void,
    scales_ptr: *const c_void,
    zeros_ptr: *const c_void,
    output_ptr: *mut c_void,
    m: i32,
    k: i32,
    n: i32,
    group_size: i32,
}

#[repr(C)]
struct Q8MatmulArgs {
    input_ptr: *const c_void,
    weight_ptr: *const c_void,
    scales_ptr: *const c_void,
    zeros_ptr: *const c_void,
    output_ptr: *mut c_void,
    m: i32,
    k: i32,
    n: i32,
    group_size: i32,
}

#[repr(C)]
struct AwqMatmulArgs {
    input_ptr: *const c_void,
    weight_ptr: *const c_void,
    scales_ptr: *const c_void,
    output_ptr: *mut c_void,
    m: i32,
    k: i32,
    n: i32,
    group_size: i32,
}

#[repr(C)]
struct Q4DequantizeArgs {
    qweight_ptr: *const c_void,
    scales_ptr: *const c_void,
    output_ptr: *mut c_void,
    num_blocks: i32,
}

#[repr(C)]
struct AwqDequantizeArgs {
    qweight_ptr: *const c_void,
    qzeros_ptr: *const c_void,
    scales_ptr: *const c_void,
    output_ptr: *mut c_void,
    n: i32,
    k: i32,
    group_size: i32,
    groups: i32,
}

/// HSA Quantized matmul kernel for AMD GPUs.
pub struct HsaQuantizedKernel {
    agent: GpuAgent,
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    q4_matmul_f32_kernel: u64,
    q4_matmul_f16_kernel: u64,
    q8_matmul_f32_kernel: u64,
    q8_matmul_f16_kernel: u64,
    awq_matmul_f32_kernel: u64,
    awq_matmul_f16_kernel: u64,
    q4_dequantize_f32_kernel: u64,
    awq_dequantize_f32_kernel: u64,
    q4_kernarg_size: u32,
    q8_kernarg_size: u32,
    awq_kernarg_size: u32,
    q4_dequant_kernarg_size: u32,
    awq_dequant_kernarg_size: u32,
}

impl HsaQuantizedKernel {
    /// Create a new HSA quantized matmul kernel.
    pub fn new(agent: &GpuAgent) -> Result<Self, HsaQuantizedError> {
        let lib = get_hsa_lib().map_err(|e| HsaQuantizedError::HsaError(e.to_string()))?;

        // Check if HSACO is valid
        if PRECOMPILED_HSACO.len() < 100 {
            return Err(HsaQuantizedError::HsaError(
                "Quantized HSACO not compiled (placeholder file)".to_string()
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
                return Err(HsaQuantizedError::HsaError(
                    format!("Failed to create HSA reader: {}", get_error_string(status))
                ));
            }
        }

        // 2. Create executable
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(1, 0, ptr::null(), &mut executable);
            if status != HSA_STATUS_SUCCESS {
                return Err(HsaQuantizedError::HsaError(
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
                return Err(HsaQuantizedError::HsaError(
                    format!("Failed to load HSA code: {}", get_error_string(status))
                ));
            }
        }

        // 4. Freeze
        unsafe {
            (lib.hsa_executable_freeze)(executable, ptr::null());
        }

        // 5. Get symbols
        let q4_f32_name = CString::new("q4_matmul_f32").unwrap();
        let q4_f16_name = CString::new("q4_matmul_f16").unwrap();
        let q8_f32_name = CString::new("q8_matmul_f32").unwrap();
        let q8_f16_name = CString::new("q8_matmul_f16").unwrap();
        let awq_f32_name = CString::new("awq_matmul_f32").unwrap();
        let awq_f16_name = CString::new("awq_matmul_f16").unwrap();
        let q4_dequant_f32_name = CString::new("q4_dequantize_f32").unwrap();
        let awq_dequant_f32_name = CString::new("awq_dequantize_f32").unwrap();

        let mut q4_f32_symbol: u64 = 0;
        let mut q4_f16_symbol: u64 = 0;
        let mut q8_f32_symbol: u64 = 0;
        let mut q8_f16_symbol: u64 = 0;
        let mut awq_f32_symbol: u64 = 0;
        let mut awq_f16_symbol: u64 = 0;
        let mut q4_dequant_f32_symbol: u64 = 0;
        let mut awq_dequant_f32_symbol: u64 = 0;

        unsafe {
            (lib.hsa_executable_get_symbol_by_name)(
                executable, q4_f32_name.as_ptr(), &agent.handle, &mut q4_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, q4_f16_name.as_ptr(), &agent.handle, &mut q4_f16_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, q8_f32_name.as_ptr(), &agent.handle, &mut q8_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, q8_f16_name.as_ptr(), &agent.handle, &mut q8_f16_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, awq_f32_name.as_ptr(), &agent.handle, &mut awq_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, awq_f16_name.as_ptr(), &agent.handle, &mut awq_f16_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable, q4_dequant_f32_name.as_ptr(), &agent.handle, &mut q4_dequant_f32_symbol
            );
            (lib.hsa_executable_get_symbol_by_name)(
                executable,
                awq_dequant_f32_name.as_ptr(),
                &agent.handle,
                &mut awq_dequant_f32_symbol,
            );
        }

        if q4_f32_symbol == 0 {
            return Err(HsaQuantizedError::HsaError(
                "Quantized kernel symbols not found in HSACO".to_string()
            ));
        }

        // 6. Get kernel objects
        let mut q4_matmul_f32_kernel: u64 = 0;
        let mut q4_matmul_f16_kernel: u64 = 0;
        let mut q8_matmul_f32_kernel: u64 = 0;
        let mut q8_matmul_f16_kernel: u64 = 0;
        let mut awq_matmul_f32_kernel: u64 = 0;
        let mut awq_matmul_f16_kernel: u64 = 0;
        let mut q4_dequantize_f32_kernel: u64 = 0;
        let mut awq_dequantize_f32_kernel: u64 = 0;
        let mut q4_kernarg_size: u32 = 0;
        let mut q8_kernarg_size: u32 = 0;
        let mut awq_kernarg_size: u32 = 0;
        let mut q4_dequant_kernarg_size: u32 = 0;
        let mut awq_dequant_kernarg_size: u32 = 0;

        unsafe {
            (lib.hsa_executable_symbol_get_info)(
                q4_f32_symbol, 22, &mut q4_matmul_f32_kernel as *mut _ as *mut c_void
            );
            (lib.hsa_executable_symbol_get_info)(
                q4_f32_symbol, 23, &mut q4_kernarg_size as *mut _ as *mut c_void
            );

            if q4_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    q4_f16_symbol, 22, &mut q4_matmul_f16_kernel as *mut _ as *mut c_void
                );
            }

            if q8_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    q8_f32_symbol, 22, &mut q8_matmul_f32_kernel as *mut _ as *mut c_void
                );
                (lib.hsa_executable_symbol_get_info)(
                    q8_f32_symbol, 23, &mut q8_kernarg_size as *mut _ as *mut c_void
                );
            }

            if q8_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    q8_f16_symbol, 22, &mut q8_matmul_f16_kernel as *mut _ as *mut c_void
                );
            }

            if awq_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    awq_f32_symbol, 22, &mut awq_matmul_f32_kernel as *mut _ as *mut c_void
                );
                (lib.hsa_executable_symbol_get_info)(
                    awq_f32_symbol, 23, &mut awq_kernarg_size as *mut _ as *mut c_void
                );
            }

            if awq_f16_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    awq_f16_symbol, 22, &mut awq_matmul_f16_kernel as *mut _ as *mut c_void
                );
            }

            if q4_dequant_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    q4_dequant_f32_symbol,
                    22,
                    &mut q4_dequantize_f32_kernel as *mut _ as *mut c_void,
                );
                (lib.hsa_executable_symbol_get_info)(
                    q4_dequant_f32_symbol,
                    23,
                    &mut q4_dequant_kernarg_size as *mut _ as *mut c_void,
                );
            }

            if awq_dequant_f32_symbol != 0 {
                (lib.hsa_executable_symbol_get_info)(
                    awq_dequant_f32_symbol,
                    22,
                    &mut awq_dequantize_f32_kernel as *mut _ as *mut c_void,
                );
                (lib.hsa_executable_symbol_get_info)(
                    awq_dequant_f32_symbol,
                    23,
                    &mut awq_dequant_kernarg_size as *mut _ as *mut c_void,
                );
            }
        }

        Ok(Self {
            agent: agent.clone(),
            executable,
            reader,
            q4_matmul_f32_kernel,
            q4_matmul_f16_kernel,
            q8_matmul_f32_kernel,
            q8_matmul_f16_kernel,
            awq_matmul_f32_kernel,
            awq_matmul_f16_kernel,
            q4_dequantize_f32_kernel,
            awq_dequantize_f32_kernel,
            q4_kernarg_size,
            q8_kernarg_size,
            awq_kernarg_size,
            q4_dequant_kernarg_size,
            awq_dequant_kernarg_size,
        })
    }

    /// Execute Q4 matmul: input [M, K] x weight_q4 [K, N] -> output [M, N]
    pub fn q4_matmul_f32(
        &self,
        queue: &HsaQueueWrapper,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        scales_ptr: *const c_void,
        zeros_ptr: *const c_void,
        output_ptr: *mut c_void,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> Result<(), HsaQuantizedError> {
        let lib = get_hsa_lib().map_err(|e| HsaQuantizedError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.q4_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = Q4MatmulArgs {
                input_ptr,
                weight_ptr,
                scales_ptr,
                zeros_ptr,
                output_ptr,
                m: m as i32,
                k: k as i32,
                n: n as i32,
                group_size: group_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut Q4MatmulArgs, 1);
        }

        // 2D grid: [N, M]
        let grid_x = ((n + 255) / 256) * 256;

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64
            ) as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 2 << 0, // 2 Dimensions
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid_x as u32,
                grid_size_y: m as u32,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.q4_matmul_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaQuantizedError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute Q8 matmul: input [M, K] x weight_q8 [K, N] -> output [M, N]
    pub fn q8_matmul_f32(
        &self,
        queue: &HsaQueueWrapper,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        scales_ptr: *const c_void,
        zeros_ptr: *const c_void,
        output_ptr: *mut c_void,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> Result<(), HsaQuantizedError> {
        if self.q8_matmul_f32_kernel == 0 {
            return Err(HsaQuantizedError::HsaError(
                "Q8 matmul kernel not available".to_string()
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaQuantizedError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.q8_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = Q8MatmulArgs {
                input_ptr,
                weight_ptr,
                scales_ptr,
                zeros_ptr,
                output_ptr,
                m: m as i32,
                k: k as i32,
                n: n as i32,
                group_size: group_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut Q8MatmulArgs, 1);
        }

        let grid_x = ((n + 255) / 256) * 256;

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64
            ) as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 2 << 0,
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid_x as u32,
                grid_size_y: m as u32,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.q8_matmul_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaQuantizedError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute AWQ matmul: input [M, K] x weight_awq [K, N] -> output [M, N]
    pub fn awq_matmul_f32(
        &self,
        queue: &HsaQueueWrapper,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        scales_ptr: *const c_void,
        output_ptr: *mut c_void,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> Result<(), HsaQuantizedError> {
        if self.awq_matmul_f32_kernel == 0 {
            return Err(HsaQuantizedError::HsaError(
                "AWQ matmul kernel not available".to_string()
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaQuantizedError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.awq_kernarg_size as usize,
                &mut kernarg_ptr
            );

            let args = AwqMatmulArgs {
                input_ptr,
                weight_ptr,
                scales_ptr,
                output_ptr,
                m: m as i32,
                k: k as i32,
                n: n as i32,
                group_size: group_size as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut AwqMatmulArgs, 1);
        }

        let grid_x = ((n + 255) / 256) * 256;

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr = (queue.queue() as *mut u8).add(
                (write_index % 4096) as usize * 64
            ) as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 2 << 0,
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid_x as u32,
                grid_size_y: m as u32,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.awq_matmul_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError|
                HsaQuantizedError::HsaError(e.to_string())
            )?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute Q4_0 dequantize: packed blocks -> f32 weights on GPU.
    pub fn q4_dequantize_f32(
        &self,
        queue: &HsaQueueWrapper,
        qweight_ptr: *const c_void,
        scales_ptr: *const c_void,
        output_ptr: *mut c_void,
        num_blocks: usize,
    ) -> Result<(), HsaQuantizedError> {
        if self.q4_dequantize_f32_kernel == 0 || self.q4_dequant_kernarg_size == 0 {
            return Err(HsaQuantizedError::HsaError(
                "Q4 dequantize kernel not available".to_string(),
            ));
        }
        if num_blocks == 0 || num_blocks > i32::MAX as usize {
            return Err(HsaQuantizedError::InvalidInput(
                "num_blocks must be in 1..=i32::MAX".to_string(),
            ));
        }
        let total_values = num_blocks
            .checked_mul(32)
            .ok_or_else(|| HsaQuantizedError::InvalidInput("output overflow".to_string()))?;
        if total_values > u32::MAX as usize {
            return Err(HsaQuantizedError::InvalidInput(
                "output exceeds addressable range".to_string(),
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaQuantizedError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.q4_dequant_kernarg_size as usize,
                &mut kernarg_ptr,
            );

            let args = Q4DequantizeArgs {
                qweight_ptr,
                scales_ptr,
                output_ptr,
                num_blocks: num_blocks as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut Q4DequantizeArgs, 1);
        }

        let grid_x = ((total_values + 255) / 256) * 256;

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr =
                (queue.queue() as *mut u8).add((write_index % 4096) as usize * 64)
                    as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1 << 0, // 1 dimension
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid_x as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.q4_dequantize_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError| {
                HsaQuantizedError::HsaError(e.to_string())
            })?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Execute AWQ INT4 dequantize: packed weights -> f32 weights on GPU.
    pub fn awq_dequantize_f32(
        &self,
        queue: &HsaQueueWrapper,
        qweight_ptr: *const c_void,
        qzeros_ptr: *const c_void,
        scales_ptr: *const c_void,
        output_ptr: *mut c_void,
        n: usize,
        k: usize,
        group_size: usize,
        groups: usize,
    ) -> Result<(), HsaQuantizedError> {
        if self.awq_dequantize_f32_kernel == 0 || self.awq_dequant_kernarg_size == 0 {
            return Err(HsaQuantizedError::HsaError(
                "AWQ dequantize kernel not available".to_string(),
            ));
        }
        if n == 0 || k == 0 || group_size == 0 || groups == 0 {
            return Err(HsaQuantizedError::InvalidInput(
                "n, k, group_size, and groups must be > 0".to_string(),
            ));
        }
        if n > i32::MAX as usize
            || k > i32::MAX as usize
            || group_size > i32::MAX as usize
            || groups > i32::MAX as usize
        {
            return Err(HsaQuantizedError::InvalidInput(
                "dimensions exceed addressable range".to_string(),
            ));
        }
        let total_values = n
            .checked_mul(k)
            .ok_or_else(|| HsaQuantizedError::InvalidInput("output overflow".to_string()))?;
        if total_values > u32::MAX as usize {
            return Err(HsaQuantizedError::InvalidInput(
                "output exceeds addressable range".to_string(),
            ));
        }

        let lib = get_hsa_lib().map_err(|e| HsaQuantizedError::HsaError(e.to_string()))?;

        let mut kernarg_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                self.awq_dequant_kernarg_size as usize,
                &mut kernarg_ptr,
            );

            let args = AwqDequantizeArgs {
                qweight_ptr,
                qzeros_ptr,
                scales_ptr,
                output_ptr,
                n: n as i32,
                k: k as i32,
                group_size: group_size as i32,
                groups: groups as i32,
            };

            ptr::copy_nonoverlapping(&args, kernarg_ptr as *mut AwqDequantizeArgs, 1);
        }

        let grid_x = ((total_values + 255) / 256) * 256;

        unsafe {
            queue.reset_signal();
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue.queue(), 1);
            let packet_ptr =
                (queue.queue() as *mut u8).add((write_index % 4096) as usize * 64)
                    as *mut HsaKernelDispatchPacket;

            let packet = HsaKernelDispatchPacket {
                header: (1 << 0) | (2 << 9) | (2 << 11),
                setup: 1 << 0, // 1 dimension
                workgroup_size_x: 256,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid_x as u32,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: self.awq_dequantize_f32_kernel,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: queue.signal(),
            };

            ptr::write_volatile(packet_ptr, packet);
            (lib.hsa_queue_store_write_index_relaxed)(queue.queue(), write_index + 1);
            (lib.hsa_signal_store_relaxed)(queue.signal(), 1);

            queue.synchronize().map_err(|e: HsaFlashAttentionError| {
                HsaQuantizedError::HsaError(e.to_string())
            })?;
            (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }

    /// Check if f16 kernels are available.
    pub fn has_f16(&self) -> bool {
        self.q4_matmul_f16_kernel != 0
    }

    /// Check if Q8 is available.
    pub fn has_q8(&self) -> bool {
        self.q8_matmul_f32_kernel != 0
    }

    /// Check if AWQ is available.
    pub fn has_awq(&self) -> bool {
        self.awq_matmul_f32_kernel != 0
    }

    /// Check if Q4 dequantize is available.
    pub fn has_q4_dequantize(&self) -> bool {
        self.q4_dequantize_f32_kernel != 0
    }

    /// Check if AWQ dequantize is available.
    pub fn has_awq_dequantize(&self) -> bool {
        self.awq_dequantize_f32_kernel != 0
    }
}

impl Drop for HsaQuantizedKernel {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_executable_destroy)(self.executable);
                (lib.hsa_code_object_reader_destroy)(self.reader);
            }
        }
    }
}
