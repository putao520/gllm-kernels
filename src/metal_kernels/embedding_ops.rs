//! Metal embedding quantization and similarity kernels.
//!
//! GPU-accelerated operations for:
//! - Binary IP (Hamming distance for 1-bit quantized vectors)
//! - Int8 Dot Product (4x compression, near-lossless)
//! - Int4 Packed Dot Product (8x compression)
//! - Matryoshka Dimension Truncation
//!
//! ## Precompiled metallib (Required)
//!
//! metallib must be precompiled before use:
//! ```bash
//! ./scripts/compile_metal_kernels.sh
//! ```
//!
//! metallib is Metal's intermediate format (like PTX/HSACO).
//! NO runtime compilation fallback - metallib must be precompiled and embedded.

use std::fmt;
use std::mem;
use std::os::raw::c_void;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};

use crate::metal_kernels::metallib_loader::{MetallibCollection, MetallibLoadError};
use crate::validation::{validate_binary_dim, validate_int8_dim, validate_int4_dim};

// Kernel entry point names
const KERNEL_BINARY_IP_HAMMING: &str = "binary_ip_hamming";
const KERNEL_BINARY_IP_ASYMMETRIC: &str = "binary_ip_asymmetric";
const KERNEL_INT8_DOT_PRODUCT: &str = "int8_dot_product";
const KERNEL_INT4_DOT_PRODUCT: &str = "int4_dot_product";
const KERNEL_MATRYOSHKA_TRUNCATE: &str = "matryoshka_truncate";
const KERNEL_MATRYOSHKA_NORMALIZE: &str = "matryoshka_normalize";

/// Metallib collection for embedding ops kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static EMBEDDING_OPS_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "embedding_ops",
    metallib_data: include_bytes!("kernels/embedding_ops.metallib"),
};

/// Errors from Metal embedding kernels.
#[derive(Debug)]
pub enum EmbeddingOpsError {
    /// Metal framework error.
    Metal(String),
    /// Invalid configuration or shape.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for EmbeddingOpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for EmbeddingOpsError {}

impl From<MetallibLoadError> for EmbeddingOpsError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Metal embedding operations kernel wrapper.
pub struct EmbeddingOpsKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_binary_hamming: ComputePipelineState,
    pipeline_binary_asymmetric: ComputePipelineState,
    pipeline_int8_dot: ComputePipelineState,
    pipeline_int4_dot: ComputePipelineState,
    pipeline_matryoshka_truncate: ComputePipelineState,
    pipeline_matryoshka_normalize: ComputePipelineState,
}

impl EmbeddingOpsKernel {
    /// Load embedding ops kernel module on the given device.
    ///
    /// Loads precompiled metallib (Metal's intermediate format).
    /// NO runtime compilation fallback - metallib must be precompiled.
    pub fn new(device: &Device) -> Result<Self, EmbeddingOpsError> {
        let library = load_library(device)?;

        let pipeline_binary_hamming = build_pipeline(device, &library, KERNEL_BINARY_IP_HAMMING)?;
        let pipeline_binary_asymmetric = build_pipeline(device, &library, KERNEL_BINARY_IP_ASYMMETRIC)?;
        let pipeline_int8_dot = build_pipeline(device, &library, KERNEL_INT8_DOT_PRODUCT)?;
        let pipeline_int4_dot = build_pipeline(device, &library, KERNEL_INT4_DOT_PRODUCT)?;
        let pipeline_matryoshka_truncate = build_pipeline(device, &library, KERNEL_MATRYOSHKA_TRUNCATE)?;
        let pipeline_matryoshka_normalize = build_pipeline(device, &library, KERNEL_MATRYOSHKA_NORMALIZE)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_binary_hamming,
            pipeline_binary_asymmetric,
            pipeline_int8_dot,
            pipeline_int4_dot,
            pipeline_matryoshka_truncate,
            pipeline_matryoshka_normalize,
        })
    }

    /// Binary IP Hamming distance between binary-quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/32] packed u32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] Hamming distances (i32)
    pub fn binary_ip_hamming(
        &self,
        queries: &Buffer,
        database: &Buffer,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<Buffer, EmbeddingOpsError> {
        validate_binary_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 32;
        validate_buffer(queries, (num_queries * packed_dim * 4) as u64, "queries")?;
        validate_buffer(database, (num_vectors * packed_dim * 4) as u64, "database")?;

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<i32>()) as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let dim_u32 = dim as u32;
        let num_queries_u32 = num_queries as u32;
        let num_vectors_u32 = num_vectors as u32;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_binary_hamming);
        encoder.set_buffer(0, Some(queries), 0);
        encoder.set_buffer(1, Some(database), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, mem::size_of::<u32>() as u64, &dim_u32 as *const _ as *const c_void);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &num_queries_u32 as *const _ as *const c_void);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &num_vectors_u32 as *const _ as *const c_void);

        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_binary_hamming);
        let threads_per_grid = MTLSize::new(output_len as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Asymmetric Binary IP: f32 query vs binary database.
    ///
    /// - `queries`: [num_queries, dim] f32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn binary_ip_asymmetric(
        &self,
        queries: &Buffer,
        database: &Buffer,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<Buffer, EmbeddingOpsError> {
        validate_binary_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 32;
        validate_buffer(queries, (num_queries * dim * 4) as u64, "queries")?;
        validate_buffer(database, (num_vectors * packed_dim * 4) as u64, "database")?;

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let dim_u32 = dim as u32;
        let num_queries_u32 = num_queries as u32;
        let num_vectors_u32 = num_vectors as u32;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_binary_asymmetric);
        encoder.set_buffer(0, Some(queries), 0);
        encoder.set_buffer(1, Some(database), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, mem::size_of::<u32>() as u64, &dim_u32 as *const _ as *const c_void);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &num_queries_u32 as *const _ as *const c_void);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &num_vectors_u32 as *const _ as *const c_void);

        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_binary_asymmetric);
        let threads_per_grid = MTLSize::new(output_len as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Int8 dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/4] packed i8x4 as u32
    /// - `database`: [num_vectors, dim/4] packed i8x4 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int8_dot_product(
        &self,
        queries: &Buffer,
        database: &Buffer,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
    ) -> Result<Buffer, EmbeddingOpsError> {
        validate_int8_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 4;
        validate_buffer(queries, (num_queries * packed_dim * 4) as u64, "queries")?;
        validate_buffer(database, (num_vectors * packed_dim * 4) as u64, "database")?;

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let dim_u32 = dim as u32;
        let num_queries_u32 = num_queries as u32;
        let num_vectors_u32 = num_vectors as u32;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_int8_dot);
        encoder.set_buffer(0, Some(queries), 0);
        encoder.set_buffer(1, Some(database), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, mem::size_of::<u32>() as u64, &dim_u32 as *const _ as *const c_void);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &num_queries_u32 as *const _ as *const c_void);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &num_vectors_u32 as *const _ as *const c_void);
        encoder.set_bytes(6, mem::size_of::<f32>() as u64, &scale as *const _ as *const c_void);

        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_int8_dot);
        let threads_per_grid = MTLSize::new(output_len as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Int4 packed dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/8] packed i4x8 as u32
    /// - `database`: [num_vectors, dim/8] packed i4x8 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int4_dot_product(
        &self,
        queries: &Buffer,
        database: &Buffer,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
        zero_point: i32,
    ) -> Result<Buffer, EmbeddingOpsError> {
        validate_int4_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 8;
        validate_buffer(queries, (num_queries * packed_dim * 4) as u64, "queries")?;
        validate_buffer(database, (num_vectors * packed_dim * 4) as u64, "database")?;

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let dim_u32 = dim as u32;
        let num_queries_u32 = num_queries as u32;
        let num_vectors_u32 = num_vectors as u32;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_int4_dot);
        encoder.set_buffer(0, Some(queries), 0);
        encoder.set_buffer(1, Some(database), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, mem::size_of::<u32>() as u64, &dim_u32 as *const _ as *const c_void);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &num_queries_u32 as *const _ as *const c_void);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &num_vectors_u32 as *const _ as *const c_void);
        encoder.set_bytes(6, mem::size_of::<f32>() as u64, &scale as *const _ as *const c_void);
        encoder.set_bytes(7, mem::size_of::<i32>() as u64, &zero_point as *const _ as *const c_void);

        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_int4_dot);
        let threads_per_grid = MTLSize::new(output_len as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Truncate embeddings to a smaller dimension (Matryoshka).
    ///
    /// - `input`: [num_vectors, full_dim] f32
    /// - Returns: [num_vectors, target_dim] f32
    pub fn matryoshka_truncate(
        &self,
        input: &Buffer,
        full_dim: usize,
        target_dim: usize,
        num_vectors: usize,
        normalize: bool,
    ) -> Result<Buffer, EmbeddingOpsError> {
        if target_dim > full_dim {
            return Err(EmbeddingOpsError::InvalidConfig(
                "target_dim > full_dim".into(),
            ));
        }
        validate_buffer(input, (num_vectors * full_dim * 4) as u64, "input")?;

        let output_len = num_vectors * target_dim;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let full_dim_u32 = full_dim as u32;
        let target_dim_u32 = target_dim as u32;
        let num_vectors_u32 = num_vectors as u32;

        // Truncate pass
        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_matryoshka_truncate);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_bytes(2, mem::size_of::<u32>() as u64, &full_dim_u32 as *const _ as *const c_void);
            encoder.set_bytes(3, mem::size_of::<u32>() as u64, &target_dim_u32 as *const _ as *const c_void);
            encoder.set_bytes(4, mem::size_of::<u32>() as u64, &num_vectors_u32 as *const _ as *const c_void);

            let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_matryoshka_truncate);
            let threads_per_grid = MTLSize::new(output_len as u64, 1, 1);
            encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Normalize pass if requested
        if normalize {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_matryoshka_normalize);
            encoder.set_buffer(0, Some(&output), 0);
            encoder.set_bytes(1, mem::size_of::<u32>() as u64, &target_dim_u32 as *const _ as *const c_void);
            encoder.set_bytes(2, mem::size_of::<u32>() as u64, &num_vectors_u32 as *const _ as *const c_void);

            let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_matryoshka_normalize);
            let threads_per_grid = MTLSize::new(num_vectors as u64, 1, 1);
            encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        Ok(output)
    }
}

/// Load Metal library from embedded metallib.
///
/// metallib is Metal's intermediate format - NO runtime compilation fallback.
fn load_library(device: &Device) -> Result<Library, EmbeddingOpsError> {
    EMBEDDING_OPS_METALLIB.load(device).map_err(EmbeddingOpsError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, EmbeddingOpsError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| EmbeddingOpsError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(EmbeddingOpsError::Metal)
}

fn validate_buffer(
    buffer: &Buffer,
    expected_bytes: u64,
    name: &str,
) -> Result<(), EmbeddingOpsError> {
    if buffer.length() < expected_bytes {
        return Err(EmbeddingOpsError::InvalidConfig(format!(
            "{name} buffer too small: {} < {}",
            buffer.length(),
            expected_bytes
        )));
    }
    Ok(())
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
