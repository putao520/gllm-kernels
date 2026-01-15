//! CUDA embedding quantization and similarity kernels.
//!
//! GPU-accelerated operations for:
//! - Binary IP (Hamming distance for 1-bit quantized vectors)
//! - Int8 Dot Product (4x compression, near-lossless)
//! - Int4 Packed Dot Product (8x compression)
//! - Matryoshka Dimension Truncation
//!
//! # SM-Aware PTX Loading
//!
//! This module automatically selects the best PTX binary for the detected GPU.
//! 🚨 **Fat Binary Only**: NO runtime compilation fallback.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};
use crate::validation::{validate_binary_dim, validate_int8_dim, validate_int4_dim, validate_input_len};

// Kernel entry point names
const KERNEL_BINARY_IP_HAMMING: &str = "binary_ip_hamming";
const KERNEL_BINARY_IP_ASYMMETRIC: &str = "binary_ip_asymmetric";
const KERNEL_INT8_DOT_PRODUCT: &str = "int8_dot_product";
const KERNEL_INT4_DOT_PRODUCT: &str = "int4_dot_product";
const KERNEL_MATRYOSHKA_TRUNCATE: &str = "matryoshka_truncate";
const KERNEL_MATRYOSHKA_NORMALIZE: &str = "matryoshka_normalize";

const BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for embedding kernels.
/// PTX compiled for a lower SM version is forward-compatible with higher SM GPUs.
///
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
/// Precompile PTX with: ./scripts/compile_cuda_kernels.sh
static EMBEDDING_OPS_PTX: PtxCollection = PtxCollection {
    kernel_name: "embedding_ops",
    ptx_versions: &[
        // SM 61 (Pascal) - GTX 1060/1070/1080
        (61, include_str!("kernels/embedding_ops_sm61.ptx")),
        // SM 80 (Ampere) - default for A100/RTX 30 series and higher
        (80, include_str!("kernels/embedding_ops.ptx")),
    ],
};

/// Errors from CUDA embedding kernels.
#[derive(Debug)]
pub enum EmbeddingOpsError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or shape.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
}

impl fmt::Display for EmbeddingOpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
        }
    }
}

impl std::error::Error for EmbeddingOpsError {}

impl From<DriverError> for EmbeddingOpsError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for EmbeddingOpsError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// CUDA embedding operations kernel wrapper.
pub struct EmbeddingOpsKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernel_binary_hamming: CudaFunction,
    kernel_binary_asymmetric: CudaFunction,
    kernel_int8_dot: CudaFunction,
    kernel_int4_dot: CudaFunction,
    kernel_matryoshka_truncate: CudaFunction,
    kernel_matryoshka_normalize: CudaFunction,
}

impl EmbeddingOpsKernel {
    /// Load embedding ops kernel module on the given device.
    ///
    /// This method automatically selects the best PTX binary for the detected GPU,
    /// falling back to NVRTC runtime compilation if needed.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, EmbeddingOpsError> {
        let ptx = EMBEDDING_OPS_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_binary_hamming = module
            .load_function(KERNEL_BINARY_IP_HAMMING)
            .map_err(|_| EmbeddingOpsError::KernelMissing(KERNEL_BINARY_IP_HAMMING))?;
        let kernel_binary_asymmetric = module
            .load_function(KERNEL_BINARY_IP_ASYMMETRIC)
            .map_err(|_| EmbeddingOpsError::KernelMissing(KERNEL_BINARY_IP_ASYMMETRIC))?;
        let kernel_int8_dot = module
            .load_function(KERNEL_INT8_DOT_PRODUCT)
            .map_err(|_| EmbeddingOpsError::KernelMissing(KERNEL_INT8_DOT_PRODUCT))?;
        let kernel_int4_dot = module
            .load_function(KERNEL_INT4_DOT_PRODUCT)
            .map_err(|_| EmbeddingOpsError::KernelMissing(KERNEL_INT4_DOT_PRODUCT))?;
        let kernel_matryoshka_truncate = module
            .load_function(KERNEL_MATRYOSHKA_TRUNCATE)
            .map_err(|_| EmbeddingOpsError::KernelMissing(KERNEL_MATRYOSHKA_TRUNCATE))?;
        let kernel_matryoshka_normalize = module
            .load_function(KERNEL_MATRYOSHKA_NORMALIZE)
            .map_err(|_| EmbeddingOpsError::KernelMissing(KERNEL_MATRYOSHKA_NORMALIZE))?;

        Ok(Self {
            module,
            kernel_binary_hamming,
            kernel_binary_asymmetric,
            kernel_int8_dot,
            kernel_int4_dot,
            kernel_matryoshka_truncate,
            kernel_matryoshka_normalize,
        })
    }

    /// Binary IP Hamming distance between binary-quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/32] packed u32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] Hamming distances (i32)
    pub fn binary_ip_hamming(
        &self,
        stream: &Arc<CudaStream>,
        queries: &CudaSlice<u32>,
        database: &CudaSlice<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<CudaSlice<i32>, EmbeddingOpsError> {
        validate_binary_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 32;
        validate_input_len(queries.len(), num_queries * packed_dim, "queries")
            .map_err(EmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(EmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let mut output: CudaSlice<i32> = stream.alloc_zeros(output_len)?;

        let dim_i32 = dim as i32;
        let num_queries_i32 = num_queries as i32;
        let num_vectors_i32 = num_vectors as i32;

        let cfg = build_launch_config(output_len)?;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_binary_hamming);
            builder.arg(queries);
            builder.arg(database);
            builder.arg(&mut output);
            builder.arg(&dim_i32);
            builder.arg(&num_queries_i32);
            builder.arg(&num_vectors_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Asymmetric Binary IP: f32 query vs binary database.
    ///
    /// - `queries`: [num_queries, dim] f32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn binary_ip_asymmetric(
        &self,
        stream: &Arc<CudaStream>,
        queries: &CudaSlice<f32>,
        database: &CudaSlice<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<CudaSlice<f32>, EmbeddingOpsError> {
        validate_binary_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 32;
        validate_input_len(queries.len(), num_queries * dim, "queries")
            .map_err(EmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(EmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;

        let dim_i32 = dim as i32;
        let num_queries_i32 = num_queries as i32;
        let num_vectors_i32 = num_vectors as i32;

        let cfg = build_launch_config(output_len)?;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_binary_asymmetric);
            builder.arg(queries);
            builder.arg(database);
            builder.arg(&mut output);
            builder.arg(&dim_i32);
            builder.arg(&num_queries_i32);
            builder.arg(&num_vectors_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Int8 dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/4] packed i8x4 as u32
    /// - `database`: [num_vectors, dim/4] packed i8x4 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int8_dot_product(
        &self,
        stream: &Arc<CudaStream>,
        queries: &CudaSlice<u32>,
        database: &CudaSlice<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, EmbeddingOpsError> {
        validate_int8_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 4;
        validate_input_len(queries.len(), num_queries * packed_dim, "queries")
            .map_err(EmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(EmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;

        let dim_i32 = dim as i32;
        let num_queries_i32 = num_queries as i32;
        let num_vectors_i32 = num_vectors as i32;

        let cfg = build_launch_config(output_len)?;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_int8_dot);
            builder.arg(queries);
            builder.arg(database);
            builder.arg(&mut output);
            builder.arg(&dim_i32);
            builder.arg(&num_queries_i32);
            builder.arg(&num_vectors_i32);
            builder.arg(&scale);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Int4 packed dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/8] packed i4x8 as u32
    /// - `database`: [num_vectors, dim/8] packed i4x8 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int4_dot_product(
        &self,
        stream: &Arc<CudaStream>,
        queries: &CudaSlice<u32>,
        database: &CudaSlice<u32>,
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
        zero_point: i32,
    ) -> Result<CudaSlice<f32>, EmbeddingOpsError> {
        validate_int4_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 8;
        validate_input_len(queries.len(), num_queries * packed_dim, "queries")
            .map_err(EmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(EmbeddingOpsError::InvalidConfig)?;

        let output_len = num_queries * num_vectors;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;

        let dim_i32 = dim as i32;
        let num_queries_i32 = num_queries as i32;
        let num_vectors_i32 = num_vectors as i32;

        let cfg = build_launch_config(output_len)?;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_int4_dot);
            builder.arg(queries);
            builder.arg(database);
            builder.arg(&mut output);
            builder.arg(&dim_i32);
            builder.arg(&num_queries_i32);
            builder.arg(&num_vectors_i32);
            builder.arg(&scale);
            builder.arg(&zero_point);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Truncate embeddings to a smaller dimension (Matryoshka).
    ///
    /// - `input`: [num_vectors, full_dim] f32
    /// - Returns: [num_vectors, target_dim] f32
    pub fn matryoshka_truncate(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        full_dim: usize,
        target_dim: usize,
        num_vectors: usize,
        normalize: bool,
    ) -> Result<CudaSlice<f32>, EmbeddingOpsError> {
        if target_dim > full_dim {
            return Err(EmbeddingOpsError::InvalidConfig(
                "target_dim > full_dim".into(),
            ));
        }
        validate_input_len(input.len(), num_vectors * full_dim, "input")
            .map_err(EmbeddingOpsError::InvalidConfig)?;

        let output_len = num_vectors * target_dim;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_len)?;

        let full_dim_i32 = full_dim as i32;
        let target_dim_i32 = target_dim as i32;
        let num_vectors_i32 = num_vectors as i32;

        let cfg = build_launch_config(output_len)?;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_matryoshka_truncate);
            builder.arg(input);
            builder.arg(&mut output);
            builder.arg(&full_dim_i32);
            builder.arg(&target_dim_i32);
            builder.arg(&num_vectors_i32);
            builder.launch(cfg)?;
        }

        // Normalize if requested
        if normalize {
            let norm_cfg = build_launch_config(num_vectors)?;
            unsafe {
                let mut builder = stream.launch_builder(&self.kernel_matryoshka_normalize);
                builder.arg(&mut output);
                builder.arg(&target_dim_i32);
                builder.arg(&num_vectors_i32);
                builder.launch(norm_cfg)?;
            }
        }

        Ok(output)
    }
}

// Helper functions

fn build_launch_config(total_threads: usize) -> Result<LaunchConfig, EmbeddingOpsError> {
    let grid_dim = (total_threads as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Ok(LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}
