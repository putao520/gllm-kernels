//! HSA Runtime paged attention kernel wrapper.
//!
//! This module provides paged attention kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Architecture
//!
//! ```text
//! Compile time: HIP kernel source → HSACO binary (via hipcc/offline compiler)
//! Runtime: HSA Runtime loads HSACO → GPU executes
//! ```

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaKernelDispatchPacket, HSA_STATUS_SUCCESS,
};
use super::hsa_flash_attn::{HsaBuffer, HsaQueueWrapper};

const KERNEL_F32: &str = "paged_attention_forward_f32";
const KERNEL_F16: &str = "paged_attention_forward_f16";
const DEFAULT_BLOCK: u32 = 128;
const MAX_HEAD_DIM: usize = 256;

// HSA packet header constants
const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;

// HSA executable symbol info
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

// Embedded HSACO (placeholder - replace with actual compiled binary)
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/paged_attention.hsaco");

/// Errors from HSA paged attention kernels.
#[derive(Debug)]
pub enum HsaPagedAttentionError {
    /// HSA runtime error.
    Hsa(i32, String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Module load failure.
    ModuleLoadFailed(String),
    /// HSA library not available.
    HsaNotAvailable(String),
    /// No GPU agents found.
    NoGpuFound,
    /// Memory allocation failed.
    AllocationFailed(String),
}

impl fmt::Display for HsaPagedAttentionError {
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

impl std::error::Error for HsaPagedAttentionError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaPagedAttentionError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaPagedAttentionError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// HSA kernel module loaded from HSACO.
struct HsaKernelModule {
    executable: HsaExecutable,
    #[allow(dead_code)]
    reader: HsaCodeObjectReader,
    kernel_object: u64,
    kernarg_size: u32,
    group_segment_size: u32,
    private_segment_size: u32,
}

impl HsaKernelModule {
    /// Load kernel from HSACO binary.
    fn from_hsaco(
        agent: &GpuAgent,
        hsaco: &[u8],
        kernel_name: &str,
    ) -> Result<Self, HsaPagedAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaPagedAttentionError::HsaNotAvailable(e.to_string()))?;

        // 1. Create code object reader from memory
        let mut reader: HsaCodeObjectReader = 0;
        unsafe {
            let status = (lib.hsa_code_object_reader_create_from_memory)(
                hsaco.as_ptr() as *const c_void,
                hsaco.len(),
                &mut reader,
            );
            check_hsa(status, "hsa_code_object_reader_create_from_memory")?;
        }

        // 2. Create executable
        let mut executable: HsaExecutable = 0;
        unsafe {
            let status = (lib.hsa_executable_create_alt)(
                1, // HSA_PROFILE_FULL
                0, // HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT
                ptr::null(),
                &mut executable,
            );
            check_hsa(status, "hsa_executable_create_alt")?;
        }

        // 3. Load code object to executable
        unsafe {
            let status = (lib.hsa_executable_load_agent_code_object)(
                executable,
                agent.handle,
                reader,
                ptr::null(),
                ptr::null_mut(),
            );
            check_hsa(status, "hsa_executable_load_agent_code_object")?;
        }

        // 4. Freeze executable
        unsafe {
            let status = (lib.hsa_executable_freeze)(executable, ptr::null());
            check_hsa(status, "hsa_executable_freeze")?;
        }

        // 5. Get kernel symbol
        let kernel_name_c = CString::new(kernel_name)
            .map_err(|_| HsaPagedAttentionError::InvalidConfig("Invalid kernel name".into()))?;

        let mut kernel_symbol: u64 = 0;
        unsafe {
            let status = (lib.hsa_executable_get_symbol_by_name)(
                executable,
                kernel_name_c.as_ptr(),
                &agent.handle,
                &mut kernel_symbol,
            );
            check_hsa(status, "hsa_executable_get_symbol_by_name")?;
        }

        if kernel_symbol == 0 {
            return Err(HsaPagedAttentionError::KernelMissing(
                if kernel_name.contains("f32") {
                    "paged_attention_forward_f32"
                } else {
                    "paged_attention_forward_f16"
                },
            ));
        }

        // 6. Get kernel object and metadata
        let mut kernel_object: u64 = 0;
        let mut kernarg_size: u32 = 0;
        let mut group_segment_size: u32 = 0;
        let mut private_segment_size: u32 = 0;

        unsafe {
            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &mut kernel_object as *mut _ as *mut c_void,
            );
            check_hsa(status, "get kernel_object")?;

            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &mut kernarg_size as *mut _ as *mut c_void,
            );
            check_hsa(status, "get kernarg_size")?;

            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &mut group_segment_size as *mut _ as *mut c_void,
            );
            check_hsa(status, "get group_segment_size")?;

            let status = (lib.hsa_executable_symbol_get_info)(
                kernel_symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &mut private_segment_size as *mut _ as *mut c_void,
            );
            check_hsa(status, "get private_segment_size")?;
        }

        Ok(Self {
            executable,
            reader,
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
        })
    }
}

impl Drop for HsaKernelModule {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.executable != 0 {
                    let _ = (lib.hsa_executable_destroy)(self.executable);
                }
                if self.reader != 0 {
                    let _ = (lib.hsa_code_object_reader_destroy)(self.reader);
                }
            }
        }
    }
}

/// Kernel arguments structure for paged attention.
#[repr(C)]
struct PagedAttentionArgs {
    q_ptr: *mut c_void,
    k_cache_ptr: *mut c_void,
    v_cache_ptr: *mut c_void,
    block_tables_ptr: *mut c_void,
    block_offsets_ptr: *mut c_void,
    o_ptr: *mut c_void,
    batch_size: i32,
    num_heads: i32,
    head_dim: i32,
    page_block_size: i32,
    seq_len: i32,
}

/// Paged attention HSA kernel wrapper.
pub struct HsaPagedAttentionKernel {
    agent: GpuAgent,
    module_f32: HsaKernelModule,
    module_f16: HsaKernelModule,
}

impl HsaPagedAttentionKernel {
    /// Initialize HSA runtime and load kernels.
    pub fn new(device: i32) -> Result<Self, HsaPagedAttentionError> {
        // Find GPU agents
        let agents = find_gpu_agents()
            .map_err(|e| HsaPagedAttentionError::HsaNotAvailable(e))?;

        let agent = agents
            .get(device as usize)
            .cloned()
            .ok_or(HsaPagedAttentionError::NoGpuFound)?;

        // Load HSACO
        let hsaco = Self::load_hsaco()?;

        // Load kernel modules
        let module_f32 = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_F32)?;
        let module_f16 = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_F16)?;

        Ok(Self {
            agent,
            module_f32,
            module_f16,
        })
    }

    fn load_hsaco() -> Result<Vec<u8>, HsaPagedAttentionError> {
        // Priority 1: Embedded precompiled HSACO
        if PRECOMPILED_HSACO.len() > 1 {
            log::debug!("Loading precompiled HSACO from embedded data");
            return Ok(PRECOMPILED_HSACO.to_vec());
        }

        Err(HsaPagedAttentionError::ModuleLoadFailed(
            "No HSACO binary available. Compile with hipcc and embed kernels.".into(),
        ))
    }

    /// Get the GPU agent.
    pub fn agent(&self) -> &GpuAgent {
        &self.agent
    }

    /// Forward pass for f32 inputs.
    pub fn forward_f32(
        &self,
        queue: &HsaQueueWrapper,
        q: &HsaBuffer<f32>,
        k_cache: &HsaBuffer<f32>,
        v_cache: &HsaBuffer<f32>,
        block_tables: &HsaBuffer<i32>,
        block_offsets: &HsaBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
    ) -> Result<HsaBuffer<f32>, HsaPagedAttentionError> {
        self.forward_f32_impl(
            queue,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_offsets,
            batch_size,
            num_heads,
            head_dim,
            page_block_size,
            seq_len,
            DEFAULT_BLOCK,
        )
    }

    /// Forward pass for f16 inputs (u16 storage).
    pub fn forward_f16(
        &self,
        queue: &HsaQueueWrapper,
        q: &HsaBuffer<u16>,
        k_cache: &HsaBuffer<u16>,
        v_cache: &HsaBuffer<u16>,
        block_tables: &HsaBuffer<i32>,
        block_offsets: &HsaBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
    ) -> Result<HsaBuffer<u16>, HsaPagedAttentionError> {
        self.forward_f16_impl(
            queue,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_offsets,
            batch_size,
            num_heads,
            head_dim,
            page_block_size,
            seq_len,
            DEFAULT_BLOCK,
        )
    }

    fn forward_f32_impl(
        &self,
        queue: &HsaQueueWrapper,
        q: &HsaBuffer<f32>,
        k_cache: &HsaBuffer<f32>,
        v_cache: &HsaBuffer<f32>,
        block_tables: &HsaBuffer<i32>,
        block_offsets: &HsaBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
        block_size: u32,
    ) -> Result<HsaBuffer<f32>, HsaPagedAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaPagedAttentionError::HsaNotAvailable(e.to_string()))?;

        let (output_len, grid_size, workgroup_size) =
            build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let output = HsaBuffer::<f32>::alloc_zeros(&self.agent, output_len)
            .map_err(|e| HsaPagedAttentionError::AllocationFailed(e.to_string()))?;

        // Allocate kernarg buffer
        let kernarg_size = self.module_f32.kernarg_size as usize;
        let mut kernarg_ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let status = (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                kernarg_size,
                &mut kernarg_ptr,
            );
            check_hsa(status, "allocate kernarg")?;

            // Fill kernel arguments
            let args = PagedAttentionArgs {
                q_ptr: q.as_ptr(),
                k_cache_ptr: k_cache.as_ptr(),
                v_cache_ptr: v_cache.as_ptr(),
                block_tables_ptr: block_tables.as_ptr(),
                block_offsets_ptr: block_offsets.as_ptr(),
                o_ptr: output.as_ptr(),
                batch_size: batch_size as i32,
                num_heads: num_heads as i32,
                head_dim: head_dim as i32,
                page_block_size: page_block_size as i32,
                seq_len: seq_len as i32,
            };

            ptr::copy_nonoverlapping(
                &args as *const PagedAttentionArgs as *const u8,
                kernarg_ptr as *mut u8,
                std::mem::size_of::<PagedAttentionArgs>().min(kernarg_size),
            );
        }

        // Reset signal
        queue.reset_signal();

        // Dispatch kernel
        self.dispatch_kernel(
            queue,
            &self.module_f32,
            kernarg_ptr,
            grid_size,
            workgroup_size,
        )?;

        // Wait for completion
        queue.synchronize()
            .map_err(|e| HsaPagedAttentionError::Hsa(0, e.to_string()))?;

        // Free kernarg buffer
        unsafe {
            let _ = (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(output)
    }

    fn forward_f16_impl(
        &self,
        queue: &HsaQueueWrapper,
        q: &HsaBuffer<u16>,
        k_cache: &HsaBuffer<u16>,
        v_cache: &HsaBuffer<u16>,
        block_tables: &HsaBuffer<i32>,
        block_offsets: &HsaBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        page_block_size: usize,
        seq_len: usize,
        block_size: u32,
    ) -> Result<HsaBuffer<u16>, HsaPagedAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaPagedAttentionError::HsaNotAvailable(e.to_string()))?;

        let (output_len, grid_size, workgroup_size) =
            build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let output = HsaBuffer::<u16>::alloc_zeros(&self.agent, output_len)
            .map_err(|e| HsaPagedAttentionError::AllocationFailed(e.to_string()))?;

        // Allocate kernarg buffer
        let kernarg_size = self.module_f16.kernarg_size as usize;
        let mut kernarg_ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let status = (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                kernarg_size,
                &mut kernarg_ptr,
            );
            check_hsa(status, "allocate kernarg")?;

            // Fill kernel arguments
            let args = PagedAttentionArgs {
                q_ptr: q.as_ptr(),
                k_cache_ptr: k_cache.as_ptr(),
                v_cache_ptr: v_cache.as_ptr(),
                block_tables_ptr: block_tables.as_ptr(),
                block_offsets_ptr: block_offsets.as_ptr(),
                o_ptr: output.as_ptr(),
                batch_size: batch_size as i32,
                num_heads: num_heads as i32,
                head_dim: head_dim as i32,
                page_block_size: page_block_size as i32,
                seq_len: seq_len as i32,
            };

            ptr::copy_nonoverlapping(
                &args as *const PagedAttentionArgs as *const u8,
                kernarg_ptr as *mut u8,
                std::mem::size_of::<PagedAttentionArgs>().min(kernarg_size),
            );
        }

        // Reset signal
        queue.reset_signal();

        // Dispatch kernel
        self.dispatch_kernel(
            queue,
            &self.module_f16,
            kernarg_ptr,
            grid_size,
            workgroup_size,
        )?;

        // Wait for completion
        queue.synchronize()
            .map_err(|e| HsaPagedAttentionError::Hsa(0, e.to_string()))?;

        // Free kernarg buffer
        unsafe {
            let _ = (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(output)
    }

    fn dispatch_kernel(
        &self,
        queue_wrapper: &HsaQueueWrapper,
        module: &HsaKernelModule,
        kernarg_ptr: *mut c_void,
        grid_size: u32,
        workgroup_size: u32,
    ) -> Result<(), HsaPagedAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaPagedAttentionError::HsaNotAvailable(e.to_string()))?;

        let queue = queue_wrapper.queue();
        let signal = queue_wrapper.signal();

        unsafe {
            // Get write index
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue, 1);

            // Calculate packet address
            // Queue base address is at offset 40 in the queue structure
            let queue_base = *(queue as *const u64).add(5); // base_address
            let queue_size = *(queue as *const u32).add(3); // size (mask + 1)
            let packet_index = (write_index & (queue_size as u64 - 1)) as usize;
            let packet_ptr = (queue_base as *mut HsaKernelDispatchPacket).add(packet_index);

            // Build dispatch packet
            let header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << 8)
                | (HSA_FENCE_SCOPE_SYSTEM << 4)
                | HSA_FENCE_SCOPE_SYSTEM;

            let packet = HsaKernelDispatchPacket {
                header,
                setup: 1, // 1D dispatch
                workgroup_size_x: workgroup_size as u16,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid_size,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: module.private_segment_size,
                group_segment_size: module.group_segment_size,
                kernel_object: module.kernel_object,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: signal,
            };

            // Write packet (atomically set header last)
            ptr::write_volatile(packet_ptr, packet);

            // Ring doorbell
            (lib.hsa_signal_store_relaxed)(
                *(queue as *const u64).add(4), // doorbell_signal
                write_index as i64,
            );
        }

        Ok(())
    }
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: u32,
) -> Result<(usize, u32, u32), HsaPagedAttentionError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err(HsaPagedAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(HsaPagedAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        )));
    }

    let num_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| HsaPagedAttentionError::InvalidConfig("num_queries overflow".into()))?;

    let output_len = num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| HsaPagedAttentionError::InvalidConfig("output_len overflow".into()))?;

    let workgroup_size = block_size.clamp(1, 1024);
    let grid_size = ((num_queries + workgroup_size as usize - 1) / workgroup_size as usize) as u32;

    Ok((output_len, grid_size, workgroup_size))
}

/// Get available GPU agents count.
pub fn get_gpu_count() -> Result<usize, HsaPagedAttentionError> {
    find_gpu_agents()
        .map(|agents| agents.len())
        .map_err(|e| HsaPagedAttentionError::HsaNotAvailable(e))
}
