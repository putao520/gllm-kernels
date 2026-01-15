//! HSA Runtime embedding quantization kernel wrapper.
//!
//! This module provides embedding quantization kernels for AMD GPUs via HSA Runtime.
//! Uses the low-level HSA driver API - only requires AMD GPU driver, NOT ROCm toolkit.
//!
//! ## Supported Operations
//!
//! - Binary IP (Hamming distance for 1-bit quantized vectors)
//! - Int8 Dot Product (4x compression, near-lossless)
//! - Int4 Packed Dot Product (8x compression)
//! - Matryoshka Dimension Truncation

use std::ffi::{c_void, CString};
use std::fmt;
use std::ptr;

use super::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, get_error_string, GpuAgent, HsaCodeObjectReader,
    HsaExecutable, HsaKernelDispatchPacket, HSA_STATUS_SUCCESS,
};
use super::hsa_flash_attn::{HsaBuffer, HsaQueueWrapper};

use crate::validation::{validate_binary_dim, validate_int8_dim, validate_int4_dim, validate_input_len};

const KERNEL_BINARY_IP_HAMMING: &str = "binary_ip_hamming";
const KERNEL_BINARY_IP_ASYMMETRIC: &str = "binary_ip_asymmetric";
const KERNEL_INT8_DOT_PRODUCT: &str = "int8_dot_product";
const KERNEL_INT4_DOT_PRODUCT: &str = "int4_dot_product";
const KERNEL_MATRYOSHKA_TRUNCATE: &str = "matryoshka_truncate";
const KERNEL_MATRYOSHKA_NORMALIZE: &str = "matryoshka_normalize";

const DEFAULT_WORKGROUP_SIZE: u32 = 256;

// HSA packet header constants
const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;

// HSA executable symbol info
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 23;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 24;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 25;

// Embedded HSACO (placeholder - replace with actual compiled binary)
const PRECOMPILED_HSACO: &[u8] = include_bytes!("kernels/embedding_ops.hsaco");

/// Errors from HSA embedding operations kernels.
#[derive(Debug)]
pub enum HsaEmbeddingOpsError {
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

impl fmt::Display for HsaEmbeddingOpsError {
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

impl std::error::Error for HsaEmbeddingOpsError {}

fn check_hsa(status: i32, context: &str) -> Result<(), HsaEmbeddingOpsError> {
    if status == HSA_STATUS_SUCCESS {
        Ok(())
    } else {
        let msg = get_error_string(status);
        Err(HsaEmbeddingOpsError::Hsa(status, format!("{}: {}", context, msg)))
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
    ) -> Result<Self, HsaEmbeddingOpsError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaEmbeddingOpsError::HsaNotAvailable(e.to_string()))?;

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
            .map_err(|_| HsaEmbeddingOpsError::InvalidConfig("Invalid kernel name".into()))?;

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
            return Err(HsaEmbeddingOpsError::KernelMissing(
                match kernel_name {
                    n if n == KERNEL_BINARY_IP_HAMMING => KERNEL_BINARY_IP_HAMMING,
                    n if n == KERNEL_BINARY_IP_ASYMMETRIC => KERNEL_BINARY_IP_ASYMMETRIC,
                    n if n == KERNEL_INT8_DOT_PRODUCT => KERNEL_INT8_DOT_PRODUCT,
                    n if n == KERNEL_INT4_DOT_PRODUCT => KERNEL_INT4_DOT_PRODUCT,
                    n if n == KERNEL_MATRYOSHKA_TRUNCATE => KERNEL_MATRYOSHKA_TRUNCATE,
                    _ => KERNEL_MATRYOSHKA_NORMALIZE,
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

// Kernel arguments structures

/// Arguments for binary_ip_hamming kernel.
#[repr(C)]
struct BinaryIpHammingArgs {
    queries: *mut c_void,
    database: *mut c_void,
    scores: *mut c_void,
    dim: u32,
    num_queries: u32,
    num_vectors: u32,
}

/// Arguments for binary_ip_asymmetric kernel.
#[repr(C)]
struct BinaryIpAsymmetricArgs {
    queries: *mut c_void,
    database: *mut c_void,
    scores: *mut c_void,
    dim: u32,
    num_queries: u32,
    num_vectors: u32,
}

/// Arguments for int8_dot_product kernel.
#[repr(C)]
struct Int8DotProductArgs {
    queries: *mut c_void,
    database: *mut c_void,
    scores: *mut c_void,
    dim: u32,
    num_queries: u32,
    num_vectors: u32,
    scale: f32,
}

/// Arguments for int4_dot_product kernel.
#[repr(C)]
struct Int4DotProductArgs {
    queries: *mut c_void,
    database: *mut c_void,
    scores: *mut c_void,
    dim: u32,
    num_queries: u32,
    num_vectors: u32,
    scale: f32,
    zero_point: i32,
}

/// Arguments for matryoshka_truncate kernel.
#[repr(C)]
struct MatryoshkaTruncateArgs {
    input: *mut c_void,
    output: *mut c_void,
    full_dim: u32,
    target_dim: u32,
    num_vectors: u32,
}

/// Arguments for matryoshka_normalize kernel.
#[repr(C)]
struct MatryoshkaNormalizeArgs {
    vectors: *mut c_void,
    target_dim: u32,
    num_vectors: u32,
}

/// HSA embedding operations kernel wrapper.
pub struct HsaEmbeddingOpsKernel {
    agent: GpuAgent,
    module_binary_hamming: HsaKernelModule,
    module_binary_asymmetric: HsaKernelModule,
    module_int8_dot: HsaKernelModule,
    module_int4_dot: HsaKernelModule,
    module_matryoshka_truncate: HsaKernelModule,
    module_matryoshka_normalize: HsaKernelModule,
}

impl HsaEmbeddingOpsKernel {
    /// Initialize HSA runtime and load embedding kernels.
    pub fn new(device: i32) -> Result<Self, HsaEmbeddingOpsError> {
        // Find GPU agents
        let agents = find_gpu_agents()
            .map_err(|e| HsaEmbeddingOpsError::HsaNotAvailable(e))?;

        let agent = agents
            .get(device as usize)
            .cloned()
            .ok_or(HsaEmbeddingOpsError::NoGpuFound)?;

        // Load HSACO
        let hsaco = Self::load_hsaco()?;

        // Load kernel modules
        let module_binary_hamming = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_BINARY_IP_HAMMING)?;
        let module_binary_asymmetric = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_BINARY_IP_ASYMMETRIC)?;
        let module_int8_dot = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_INT8_DOT_PRODUCT)?;
        let module_int4_dot = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_INT4_DOT_PRODUCT)?;
        let module_matryoshka_truncate = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_MATRYOSHKA_TRUNCATE)?;
        let module_matryoshka_normalize = HsaKernelModule::from_hsaco(&agent, &hsaco, KERNEL_MATRYOSHKA_NORMALIZE)?;

        Ok(Self {
            agent,
            module_binary_hamming,
            module_binary_asymmetric,
            module_int8_dot,
            module_int4_dot,
            module_matryoshka_truncate,
            module_matryoshka_normalize,
        })
    }

    fn load_hsaco() -> Result<Vec<u8>, HsaEmbeddingOpsError> {
        // Priority 1: Embedded precompiled HSACO
        if PRECOMPILED_HSACO.len() > 1 {
            log::debug!("Loading precompiled HSACO from embedded data");
            return Ok(PRECOMPILED_HSACO.to_vec());
        }

        Err(HsaEmbeddingOpsError::ModuleLoadFailed(
            "No HSACO binary available. Compile with hipcc and embed kernels.".into(),
        ))
    }

    /// Get the GPU agent.
    pub fn agent(&self) -> &GpuAgent {
        &self.agent
    }

    /// Binary IP Hamming distance between binary-quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/32] packed u32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] Hamming distances (i32)
    pub fn binary_ip_hamming(
        &self,
        queue: &HsaQueueWrapper,
        queries: &HsaBuffer<u32>,
        database: &HsaBuffer<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<HsaBuffer<i32>, HsaEmbeddingOpsError> {
        validate_binary_dim(dim).map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 32;
        validate_input_len(queries.len(), num_queries * packed_dim, "queries")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let output = HsaBuffer::<i32>::alloc_zeros(&self.agent, output_len)
            .map_err(|e| HsaEmbeddingOpsError::AllocationFailed(e.to_string()))?;

        let args = BinaryIpHammingArgs {
            queries: queries.as_ptr(),
            database: database.as_ptr(),
            scores: output.as_ptr(),
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
        };

        self.dispatch_kernel(
            queue,
            &self.module_binary_hamming,
            &args as *const _ as *const c_void,
            std::mem::size_of::<BinaryIpHammingArgs>(),
            output_len,
        )?;

        Ok(output)
    }

    /// Asymmetric Binary IP: f32 query vs binary database.
    ///
    /// - `queries`: [num_queries, dim] f32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn binary_ip_asymmetric(
        &self,
        queue: &HsaQueueWrapper,
        queries: &HsaBuffer<f32>,
        database: &HsaBuffer<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<HsaBuffer<f32>, HsaEmbeddingOpsError> {
        validate_binary_dim(dim).map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 32;
        validate_input_len(queries.len(), num_queries * dim, "queries")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let output = HsaBuffer::<f32>::alloc_zeros(&self.agent, output_len)
            .map_err(|e| HsaEmbeddingOpsError::AllocationFailed(e.to_string()))?;

        let args = BinaryIpAsymmetricArgs {
            queries: queries.as_ptr(),
            database: database.as_ptr(),
            scores: output.as_ptr(),
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
        };

        self.dispatch_kernel(
            queue,
            &self.module_binary_asymmetric,
            &args as *const _ as *const c_void,
            std::mem::size_of::<BinaryIpAsymmetricArgs>(),
            output_len,
        )?;

        Ok(output)
    }

    /// Int8 dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/4] packed i8x4 as u32
    /// - `database`: [num_vectors, dim/4] packed i8x4 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int8_dot_product(
        &self,
        queue: &HsaQueueWrapper,
        queries: &HsaBuffer<u32>,
        database: &HsaBuffer<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
    ) -> Result<HsaBuffer<f32>, HsaEmbeddingOpsError> {
        validate_int8_dim(dim).map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 4;
        validate_input_len(queries.len(), num_queries * packed_dim, "queries")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let output = HsaBuffer::<f32>::alloc_zeros(&self.agent, output_len)
            .map_err(|e| HsaEmbeddingOpsError::AllocationFailed(e.to_string()))?;

        let args = Int8DotProductArgs {
            queries: queries.as_ptr(),
            database: database.as_ptr(),
            scores: output.as_ptr(),
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
            scale,
        };

        self.dispatch_kernel(
            queue,
            &self.module_int8_dot,
            &args as *const _ as *const c_void,
            std::mem::size_of::<Int8DotProductArgs>(),
            output_len,
        )?;

        Ok(output)
    }

    /// Int4 packed dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/8] packed i4x8 as u32
    /// - `database`: [num_vectors, dim/8] packed i4x8 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int4_dot_product(
        &self,
        queue: &HsaQueueWrapper,
        queries: &HsaBuffer<u32>,
        database: &HsaBuffer<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
        zero_point: i32,
    ) -> Result<HsaBuffer<f32>, HsaEmbeddingOpsError> {
        validate_int4_dim(dim).map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 8;
        validate_input_len(queries.len(), num_queries * packed_dim, "queries")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let output = HsaBuffer::<f32>::alloc_zeros(&self.agent, output_len)
            .map_err(|e| HsaEmbeddingOpsError::AllocationFailed(e.to_string()))?;

        let args = Int4DotProductArgs {
            queries: queries.as_ptr(),
            database: database.as_ptr(),
            scores: output.as_ptr(),
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
            scale,
            zero_point,
        };

        self.dispatch_kernel(
            queue,
            &self.module_int4_dot,
            &args as *const _ as *const c_void,
            std::mem::size_of::<Int4DotProductArgs>(),
            output_len,
        )?;

        Ok(output)
    }

    /// Truncate embeddings to a smaller dimension (Matryoshka).
    ///
    /// - `input`: [num_vectors, full_dim] f32
    /// - Returns: [num_vectors, target_dim] f32
    pub fn matryoshka_truncate(
        &self,
        queue: &HsaQueueWrapper,
        input: &HsaBuffer<f32>,
        full_dim: usize,
        target_dim: usize,
        num_vectors: usize,
        normalize: bool,
    ) -> Result<HsaBuffer<f32>, HsaEmbeddingOpsError> {
        if target_dim > full_dim {
            return Err(HsaEmbeddingOpsError::InvalidConfig(
                "target_dim > full_dim".into(),
            ));
        }
        validate_input_len(input.len(), num_vectors * full_dim, "input")
            .map_err(HsaEmbeddingOpsError::InvalidConfig)?;

        let output_len = num_vectors * target_dim;
        let output = HsaBuffer::<f32>::alloc_zeros(&self.agent, output_len)
            .map_err(|e| HsaEmbeddingOpsError::AllocationFailed(e.to_string()))?;

        // Truncate
        let truncate_args = MatryoshkaTruncateArgs {
            input: input.as_ptr(),
            output: output.as_ptr(),
            full_dim: full_dim as u32,
            target_dim: target_dim as u32,
            num_vectors: num_vectors as u32,
        };

        self.dispatch_kernel(
            queue,
            &self.module_matryoshka_truncate,
            &truncate_args as *const _ as *const c_void,
            std::mem::size_of::<MatryoshkaTruncateArgs>(),
            output_len,
        )?;

        // Normalize if requested
        if normalize {
            let normalize_args = MatryoshkaNormalizeArgs {
                vectors: output.as_ptr(),
                target_dim: target_dim as u32,
                num_vectors: num_vectors as u32,
            };

            self.dispatch_kernel(
                queue,
                &self.module_matryoshka_normalize,
                &normalize_args as *const _ as *const c_void,
                std::mem::size_of::<MatryoshkaNormalizeArgs>(),
                num_vectors,
            )?;
        }

        Ok(output)
    }

    fn dispatch_kernel(
        &self,
        queue_wrapper: &HsaQueueWrapper,
        module: &HsaKernelModule,
        args: *const c_void,
        args_size: usize,
        total_threads: usize,
    ) -> Result<(), HsaEmbeddingOpsError> {
        let lib = get_hsa_lib()
            .map_err(|e| HsaEmbeddingOpsError::HsaNotAvailable(e.to_string()))?;

        let queue = queue_wrapper.queue();
        let signal = queue_wrapper.signal();

        // Allocate kernarg buffer
        let kernarg_size = module.kernarg_size as usize;
        let mut kernarg_ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let status = (lib.hsa_memory_allocate)(
                self.agent.kernarg_region,
                kernarg_size,
                &mut kernarg_ptr,
            );
            check_hsa(status, "allocate kernarg")?;

            // Copy arguments
            ptr::copy_nonoverlapping(
                args as *const u8,
                kernarg_ptr as *mut u8,
                args_size.min(kernarg_size),
            );
        }

        // Reset signal
        queue_wrapper.reset_signal();

        // Calculate grid dimensions
        let workgroup_size = DEFAULT_WORKGROUP_SIZE;
        let grid_size = ((total_threads + workgroup_size as usize - 1) / workgroup_size as usize) as u32;

        unsafe {
            // Get write index
            let write_index = (lib.hsa_queue_add_write_index_relaxed)(queue, 1);

            // Calculate packet address
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
                grid_size_x: grid_size * workgroup_size,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: module.private_segment_size,
                group_segment_size: module.group_segment_size,
                kernel_object: module.kernel_object,
                kernarg_address: kernarg_ptr,
                reserved2: 0,
                completion_signal: signal,
            };

            // Write packet
            ptr::write_volatile(packet_ptr, packet);

            // Ring doorbell
            (lib.hsa_signal_store_relaxed)(
                *(queue as *const u64).add(4), // doorbell_signal
                write_index as i64,
            );
        }

        // Wait for completion
        queue_wrapper.synchronize()
            .map_err(|e| HsaEmbeddingOpsError::Hsa(-1, e.to_string()))?;

        // Free kernarg buffer
        unsafe {
            let _ = (lib.hsa_memory_free)(kernarg_ptr);
        }

        Ok(())
    }
}

