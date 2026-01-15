//! HSA Runtime FlashAttention kernel wrapper.
//!
//! This module provides FlashAttention-style kernels for AMD GPUs via HSA Runtime.
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
use std::marker::PhantomData;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaKernelDispatchPacket, HsaQueue, HsaRegion, HsaSignal,
    HSA_STATUS_SUCCESS,
};
use crate::types::AttentionConfig;
use crate::validation::{
    validate_attention_dims, compute_num_queries, compute_output_len,
};

const KERNEL_F32: &str = "tiled_attention_forward_f32";
const KERNEL_F16: &str = "tiled_attention_forward_f16";
const DEFAULT_BLOCK: u32 = 128;

// HSA packet header constants
const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;

// HSA signal wait conditions
const HSA_SIGNAL_CONDITION_LT: u32 = 0;
const HSA_WAIT_STATE_BLOCKED: u32 = 0;

// HSA executable symbol info
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

// Embedded HSACO (placeholder - replace with actual compiled binary)
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/flash_attention.hsaco");

/// Errors from HSA FlashAttention kernels.
#[derive(Debug)]
pub enum HsaFlashAttentionError {
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

impl fmt::Display for HsaFlashAttentionError {
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

impl std::error::Error for HsaFlashAttentionError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaFlashAttentionError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaFlashAttentionError::Hsa(status, format!("{}: {}", context, msg)))
    }
}

/// HSA device memory buffer.
pub struct HsaBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    #[allow(dead_code)]
    region: HsaRegion,
    _marker: PhantomData<T>,
}

impl<T> HsaBuffer<T> {
    /// Allocate zeroed device memory from the specified region.
    pub fn alloc_zeros(agent: &GpuAgent, len: usize) -> Result<Self, HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

        // Prefer coarse-grained for device memory, fall back to fine-grained
        let region = if agent.coarse_grained_region != 0 {
            agent.coarse_grained_region
        } else if agent.fine_grained_region != 0 {
            agent.fine_grained_region
        } else {
            return Err(HsaFlashAttentionError::AllocationFailed(
                "No suitable memory region found".to_string(),
            ));
        };

        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let status = (lib.hsa_memory_allocate)(region, size, &mut ptr);
            check_hsa(status, "hsa_memory_allocate")?;

            // Zero the memory
            ptr::write_bytes(ptr as *mut u8, 0, size);
        }

        Ok(Self {
            ptr,
            len,
            region,
            _marker: PhantomData,
        })
    }

    /// Copy from host slice to device.
    pub fn from_slice(agent: &GpuAgent, data: &[T]) -> Result<Self, HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

        // Prefer coarse-grained for device memory
        let region = if agent.coarse_grained_region != 0 {
            agent.coarse_grained_region
        } else if agent.fine_grained_region != 0 {
            agent.fine_grained_region
        } else {
            return Err(HsaFlashAttentionError::AllocationFailed(
                "No suitable memory region found".to_string(),
            ));
        };

        let size = data.len() * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let status = (lib.hsa_memory_allocate)(region, size, &mut ptr);
            check_hsa(status, "hsa_memory_allocate")?;

            let status = (lib.hsa_memory_copy)(ptr, data.as_ptr() as *const c_void, size);
            check_hsa(status, "hsa_memory_copy")?;
        }

        Ok(Self {
            ptr,
            len: data.len(),
            region,
            _marker: PhantomData,
        })
    }

    /// Copy device memory to host vector.
    pub fn to_vec(&self) -> Result<Vec<T>, HsaFlashAttentionError>
    where
        T: Default + Clone,
    {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

        let mut data = vec![T::default(); self.len];
        let size = self.len * std::mem::size_of::<T>();

        unsafe {
            let status = (lib.hsa_memory_copy)(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
            );
            check_hsa(status, "hsa_memory_copy")?;
        }

        Ok(data)
    }

    /// Get raw device pointer.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get buffer length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for HsaBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(lib) = get_hsa_lib() {
                unsafe {
                    let _ = (lib.hsa_memory_free)(self.ptr);
                }
            }
        }
    }
}

/// HSA queue wrapper with signal management.
pub struct HsaQueueWrapper {
    queue: HsaQueue,
    signal: HsaSignal,
}

impl HsaQueueWrapper {
    /// Create a new HSA queue for the given agent.
    pub fn new(agent: &GpuAgent) -> Result<Self, HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

        // Create queue
        let mut queue: HsaQueue = ptr::null_mut();
        let queue_size = 4096u32; // Standard queue size

        unsafe {
            let status = (lib.hsa_queue_create)(
                agent.handle,
                queue_size,
                0, // HSA_QUEUE_TYPE_MULTI
                ptr::null_mut(),
                ptr::null_mut(),
                u32::MAX,
                u32::MAX,
                &mut queue,
            );
            check_hsa(status, "hsa_queue_create")?;
        }

        // Create completion signal
        let mut signal: HsaSignal = 0;
        unsafe {
            let status = (lib.hsa_signal_create)(1, 0, ptr::null(), &mut signal);
            check_hsa(status, "hsa_signal_create")?;
        }

        Ok(Self { queue, signal })
    }

    /// Wait for all operations to complete.
    pub fn synchronize(&self) -> Result<(), HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

        unsafe {
            // Wait until signal becomes less than 1 (i.e., 0)
            (lib.hsa_signal_wait_acquire)(
                self.signal,
                HSA_SIGNAL_CONDITION_LT, // less than
                1,                        // compare value
                u64::MAX,                 // timeout (infinite)
                HSA_WAIT_STATE_BLOCKED,
            );
        }

        Ok(())
    }

    /// Reset signal for next dispatch.
    pub fn reset_signal(&self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                (lib.hsa_signal_store_relaxed)(self.signal, 1);
            }
        }
    }

    /// Get the queue handle.
    pub fn queue(&self) -> HsaQueue {
        self.queue
    }

    /// Get the completion signal.
    pub fn signal(&self) -> HsaSignal {
        self.signal
    }
}

impl Drop for HsaQueueWrapper {
    fn drop(&mut self) {
        if let Ok(lib) = get_hsa_lib() {
            unsafe {
                if self.signal != 0 {
                    let _ = (lib.hsa_signal_destroy)(self.signal);
                }
                if !self.queue.is_null() {
                    let _ = (lib.hsa_queue_destroy)(self.queue);
                }
            }
        }
    }
}

// SAFETY: HSA queues are thread-safe for concurrent dispatch operations.
// The HSA programming model supports multi-threaded queue access with
// proper synchronization via signals. The queue contains a raw pointer
// (*mut c_void) but HSA guarantees thread-safe access to queue operations.
unsafe impl Send for HsaQueueWrapper {}
unsafe impl Sync for HsaQueueWrapper {}

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
    ) -> Result<Self, HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaFlashAttentionError::InvalidConfig("Invalid kernel name".into()))?;

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
            return Err(HsaFlashAttentionError::KernelMissing(
                if kernel_name.contains("f32") {
                    "tiled_attention_forward_f32"
                } else {
                    "tiled_attention_forward_f16"
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

/// Kernel arguments structure for flash attention.
#[repr(C)]
struct FlashAttentionArgs {
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    v_ptr: *mut c_void,
    o_ptr: *mut c_void,
    batch_size: i32,
    num_heads: i32,
    seq_len: i32,
    head_dim: i32,
    scale: f32,
    is_causal: i32,
    position_offset: i32,
}

/// FlashAttention HSA kernel wrapper.
pub struct HsaFlashAttentionKernel {
    agent: GpuAgent,
    module_f32: HsaKernelModule,
    module_f16: HsaKernelModule,
}

impl HsaFlashAttentionKernel {
    /// Initialize HSA runtime and load kernels.
    pub fn new(device: i32) -> Result<Self, HsaFlashAttentionError> {
        // Find GPU agents
        let agents = find_gpu_agents()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e))?;

        let agent = agents
            .get(device as usize)
            .cloned()
            .ok_or(HsaFlashAttentionError::NoGpuFound)?;

        // Try to load HSACO from environment variable or embedded binary
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

    fn load_hsaco() -> Result<Vec<u8>, HsaFlashAttentionError> {
        // Priority 1: Embedded precompiled HSACO
        if PRECOMPILED_HSACO.len() > 1 {
            log::debug!("Loading precompiled HSACO from embedded data");
            return Ok(PRECOMPILED_HSACO.to_vec());
        }

        Err(HsaFlashAttentionError::ModuleLoadFailed(
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
        k: &HsaBuffer<f32>,
        v: &HsaBuffer<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<HsaBuffer<f32>, HsaFlashAttentionError> {
        self.forward_f32_impl(
            queue,
            q,
            k,
            v,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            is_causal,
            scale,
            position_offset,
            DEFAULT_BLOCK,
        )
    }

    /// Forward pass for f16 inputs.
    pub fn forward_f16(
        &self,
        queue: &HsaQueueWrapper,
        q: &HsaBuffer<u16>,
        k: &HsaBuffer<u16>,
        v: &HsaBuffer<u16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
    ) -> Result<HsaBuffer<u16>, HsaFlashAttentionError> {
        self.forward_f16_impl(
            queue,
            q,
            k,
            v,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            is_causal,
            scale,
            position_offset,
            DEFAULT_BLOCK,
        )
    }

    fn forward_f32_impl(
        &self,
        queue: &HsaQueueWrapper,
        q: &HsaBuffer<f32>,
        k: &HsaBuffer<f32>,
        v: &HsaBuffer<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
        block_size: u32,
    ) -> Result<HsaBuffer<f32>, HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

        let (output_len, grid_size, workgroup_size) =
            build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let output = HsaBuffer::<f32>::alloc_zeros(&self.agent, output_len)?;

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
            let args = FlashAttentionArgs {
                q_ptr: q.as_ptr(),
                k_ptr: k.as_ptr(),
                v_ptr: v.as_ptr(),
                o_ptr: output.as_ptr(),
                batch_size: batch_size as i32,
                num_heads: num_heads as i32,
                seq_len: seq_len as i32,
                head_dim: head_dim as i32,
                scale,
                is_causal: if is_causal { 1 } else { 0 },
                position_offset: position_offset as i32,
            };

            ptr::copy_nonoverlapping(
                &args as *const FlashAttentionArgs as *const u8,
                kernarg_ptr as *mut u8,
                std::mem::size_of::<FlashAttentionArgs>().min(kernarg_size),
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
        queue.synchronize()?;

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
        k: &HsaBuffer<u16>,
        v: &HsaBuffer<u16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        scale: f32,
        position_offset: usize,
        block_size: u32,
    ) -> Result<HsaBuffer<u16>, HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

        let (output_len, grid_size, workgroup_size) =
            build_launch(batch_size, num_heads, seq_len, head_dim, block_size)?;

        let output = HsaBuffer::<u16>::alloc_zeros(&self.agent, output_len)?;

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
            let args = FlashAttentionArgs {
                q_ptr: q.as_ptr(),
                k_ptr: k.as_ptr(),
                v_ptr: v.as_ptr(),
                o_ptr: output.as_ptr(),
                batch_size: batch_size as i32,
                num_heads: num_heads as i32,
                seq_len: seq_len as i32,
                head_dim: head_dim as i32,
                scale,
                is_causal: if is_causal { 1 } else { 0 },
                position_offset: position_offset as i32,
            };

            ptr::copy_nonoverlapping(
                &args as *const FlashAttentionArgs as *const u8,
                kernarg_ptr as *mut u8,
                std::mem::size_of::<FlashAttentionArgs>().min(kernarg_size),
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
        queue.synchronize()?;

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
    ) -> Result<(), HsaFlashAttentionError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e.to_string()))?;

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

/// Optimized HSA attention wrapper.
pub struct OptimizedHsaAttention {
    tile_size: u32,
    kernel: HsaFlashAttentionKernel,
}

impl OptimizedHsaAttention {
    /// Create a new optimized attention wrapper.
    pub fn new(device: i32, tile_size: usize) -> Result<Self, HsaFlashAttentionError> {
        let tile_size = tile_size.clamp(1, 1024) as u32;
        let kernel = HsaFlashAttentionKernel::new(device)?;
        Ok(Self { tile_size, kernel })
    }

    /// Get the GPU agent.
    pub fn agent(&self) -> &GpuAgent {
        self.kernel.agent()
    }

    /// Forward pass using tiled attention.
    pub fn forward_tiled(
        &self,
        queue: &HsaQueueWrapper,
        q: &HsaBuffer<f32>,
        k: &HsaBuffer<f32>,
        v: &HsaBuffer<f32>,
        config: &AttentionConfig,
        position_offset: usize,
    ) -> Result<HsaBuffer<f32>, HsaFlashAttentionError> {
        if config.query_len != config.kv_len {
            return Err(HsaFlashAttentionError::InvalidConfig(
                "query_len must match kv_len for the tiled kernel".into(),
            ));
        }

        self.kernel.forward_f32_impl(
            queue,
            q,
            k,
            v,
            config.batch_size,
            config.num_heads,
            config.query_len,
            config.head_dim,
            config.causal,
            config.scale,
            position_offset,
            self.tile_size,
        )
    }
}

fn build_launch(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: u32,
) -> Result<(usize, u32, u32), HsaFlashAttentionError> {
    validate_attention_dims(batch_size, num_heads, seq_len, head_dim)
        .map_err(HsaFlashAttentionError::InvalidConfig)?;

    let num_queries = compute_num_queries(batch_size, num_heads, seq_len)
        .map_err(HsaFlashAttentionError::InvalidConfig)?;
    let output_len = compute_output_len(num_queries, head_dim)
        .map_err(HsaFlashAttentionError::InvalidConfig)?;

    let workgroup_size = block_size.clamp(1, 1024);
    let grid_size = ((num_queries + workgroup_size as usize - 1) / workgroup_size as usize) as u32;

    Ok((output_len, grid_size, workgroup_size))
}

/// Get available GPU agents count.
pub fn get_gpu_count() -> Result<usize, HsaFlashAttentionError> {
    find_gpu_agents()
        .map(|agents| agents.len())
        .map_err(|e| HsaFlashAttentionError::HsaNotAvailable(e))
}
