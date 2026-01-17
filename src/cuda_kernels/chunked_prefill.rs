//! CUDA Chunked Prefill / POD-Attention kernel.
//!
//! Implements memory-efficient prefill for long contexts:
//! - Chunked attention computation with online softmax
//! - POD-Attention workload splitting
//! - Batch scheduling for mixed prefill/decode
//!
//! # SM-Aware PTX Loading
//!
//! - SM 61 (Pascal): GTX 1060/1070/1080
//! - SM 80 (Ampere): A100, RTX 30 series and higher
//!
//! 🚨 **Fat Binary Only**: NO runtime compilation fallback.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

// Kernel function names
const KERNEL_CHUNK_ATTENTION_F32: &str = "chunked_prefill_attention_f32";
const KERNEL_CHUNK_ATTENTION_F16: &str = "chunked_prefill_attention_f16";
const KERNEL_MERGE_CHUNKS_F32: &str = "chunked_prefill_merge_f32";
const KERNEL_MERGE_CHUNKS_F16: &str = "chunked_prefill_merge_f16";
const KERNEL_POD_SPLIT_F32: &str = "pod_attention_split_f32";
const KERNEL_POD_SPLIT_F16: &str = "pod_attention_split_f16";
const KERNEL_SCHEDULE_F32: &str = "chunked_prefill_schedule_f32";
const KERNEL_SCHEDULE_F16: &str = "chunked_prefill_schedule_f16";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for Chunked Prefill kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static CHUNKED_PREFILL_PTX: PtxCollection = PtxCollection {
    kernel_name: "chunked_prefill",
    ptx_versions: &[
        (61, include_str!("kernels/chunked_prefill_sm61.ptx")),
        (80, include_str!("kernels/chunked_prefill.ptx")),
    ],
};

/// Errors surfaced by the CUDA Chunked Prefill kernels.
#[derive(Debug)]
pub enum ChunkedPrefillError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
    /// Chunking error.
    ChunkingError(String),
}

impl fmt::Display for ChunkedPrefillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
            Self::ChunkingError(msg) => write!(f, "Chunking error: {msg}"),
        }
    }
}

impl std::error::Error for ChunkedPrefillError {}

impl From<DriverError> for ChunkedPrefillError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for ChunkedPrefillError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// Chunked Prefill CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Chunked attention with online softmax
/// - Chunk output merging with log-sum-exp
/// - POD-Attention workload splitting
/// - Batch scheduling
pub struct ChunkedPrefillKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Chunk attention kernels
    kernel_chunk_attention_f32: CudaFunction,
    kernel_chunk_attention_f16: CudaFunction,
    // Chunk merging kernels
    kernel_merge_f32: CudaFunction,
    kernel_merge_f16: CudaFunction,
    // POD split kernels
    kernel_pod_split_f32: CudaFunction,
    kernel_pod_split_f16: CudaFunction,
    // Scheduling kernels
    kernel_schedule_f32: CudaFunction,
    kernel_schedule_f16: CudaFunction,
}

impl ChunkedPrefillKernel {
    /// Load Chunked Prefill kernel module on the given device.
    ///
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, ChunkedPrefillError> {
        let ptx = CHUNKED_PREFILL_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_chunk_attention_f32 = module
            .load_function(KERNEL_CHUNK_ATTENTION_F32)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_CHUNK_ATTENTION_F32))?;
        let kernel_chunk_attention_f16 = module
            .load_function(KERNEL_CHUNK_ATTENTION_F16)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_CHUNK_ATTENTION_F16))?;
        let kernel_merge_f32 = module
            .load_function(KERNEL_MERGE_CHUNKS_F32)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_MERGE_CHUNKS_F32))?;
        let kernel_merge_f16 = module
            .load_function(KERNEL_MERGE_CHUNKS_F16)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_MERGE_CHUNKS_F16))?;
        let kernel_pod_split_f32 = module
            .load_function(KERNEL_POD_SPLIT_F32)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_POD_SPLIT_F32))?;
        let kernel_pod_split_f16 = module
            .load_function(KERNEL_POD_SPLIT_F16)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_POD_SPLIT_F16))?;
        let kernel_schedule_f32 = module
            .load_function(KERNEL_SCHEDULE_F32)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_SCHEDULE_F32))?;
        let kernel_schedule_f16 = module
            .load_function(KERNEL_SCHEDULE_F16)
            .map_err(|_| ChunkedPrefillError::KernelMissing(KERNEL_SCHEDULE_F16))?;

        Ok(Self {
            module,
            kernel_chunk_attention_f32,
            kernel_chunk_attention_f16,
            kernel_merge_f32,
            kernel_merge_f16,
            kernel_pod_split_f32,
            kernel_pod_split_f16,
            kernel_schedule_f32,
            kernel_schedule_f16,
        })
    }

    /// Compute attention for a single chunk (f32).
    ///
    /// Uses online softmax for memory efficiency.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `query` - Query tensor [batch * num_heads * query_len * head_dim]
    /// * `key` - Key tensor [batch * num_heads * chunk_len * head_dim]
    /// * `value` - Value tensor [batch * num_heads * chunk_len * head_dim]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `query_len` - Query sequence length
    /// * `chunk_len` - Key/Value chunk length
    /// * `head_dim` - Dimension per head
    /// * `chunk_start` - Starting position of this chunk in full sequence
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Returns
    /// Tuple of (output [batch * num_heads * query_len * head_dim],
    ///           log_sum_exp [batch * num_heads * query_len])
    pub fn chunked_attention_f32(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        query_len: usize,
        chunk_len: usize,
        head_dim: usize,
        chunk_start: usize,
        causal: bool,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), ChunkedPrefillError> {
        let output_size = batch_size * num_heads * query_len * head_dim;
        let lse_size = batch_size * num_heads * query_len;

        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_size)?;
        let mut log_sum_exp: CudaSlice<f32> = stream.alloc_zeros(lse_size)?;

        let total_work = batch_size * num_heads * query_len;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (chunk_len * std::mem::size_of::<f32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let query_i32 = query_len as i32;
        let chunk_i32 = chunk_len as i32;
        let dim_i32 = head_dim as i32;
        let start_i32 = chunk_start as i32;
        let causal_i32: i32 = if causal { 1 } else { 0 };
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_chunk_attention_f32);
            builder.arg(query);
            builder.arg(key);
            builder.arg(value);
            builder.arg(&mut output);
            builder.arg(&mut log_sum_exp);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&query_i32);
            builder.arg(&chunk_i32);
            builder.arg(&dim_i32);
            builder.arg(&start_i32);
            builder.arg(&causal_i32);
            builder.arg(&scale);
            builder.launch(cfg)?;
        }

        Ok((output, log_sum_exp))
    }

    /// Compute attention for a single chunk (f16).
    pub fn chunked_attention_f16(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<f16>,
        key: &CudaSlice<f16>,
        value: &CudaSlice<f16>,
        batch_size: usize,
        num_heads: usize,
        query_len: usize,
        chunk_len: usize,
        head_dim: usize,
        chunk_start: usize,
        causal: bool,
    ) -> Result<(CudaSlice<f16>, CudaSlice<f32>), ChunkedPrefillError> {
        let output_size = batch_size * num_heads * query_len * head_dim;
        let lse_size = batch_size * num_heads * query_len;

        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_size)?;
        let mut log_sum_exp: CudaSlice<f32> = stream.alloc_zeros(lse_size)?;  // LSE always f32

        let total_work = batch_size * num_heads * query_len;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (chunk_len * std::mem::size_of::<f16>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let query_i32 = query_len as i32;
        let chunk_i32 = chunk_len as i32;
        let dim_i32 = head_dim as i32;
        let start_i32 = chunk_start as i32;
        let causal_i32: i32 = if causal { 1 } else { 0 };
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_chunk_attention_f16);
            builder.arg(query);
            builder.arg(key);
            builder.arg(value);
            builder.arg(&mut output);
            builder.arg(&mut log_sum_exp);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&query_i32);
            builder.arg(&chunk_i32);
            builder.arg(&dim_i32);
            builder.arg(&start_i32);
            builder.arg(&causal_i32);
            builder.arg(&scale);
            builder.launch(cfg)?;
        }

        Ok((output, log_sum_exp))
    }

    /// Merge chunk outputs using log-sum-exp (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `chunk_outputs` - Outputs from each chunk [num_chunks * batch * num_heads * query_len * head_dim]
    /// * `chunk_lse` - Log-sum-exp from each chunk [num_chunks * batch * num_heads * query_len]
    /// * `num_chunks` - Number of chunks
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `query_len` - Query sequence length
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    /// Merged output: [batch * num_heads * query_len * head_dim]
    pub fn merge_chunks_f32(
        &self,
        stream: &Arc<CudaStream>,
        chunk_outputs: &CudaSlice<f32>,
        chunk_lse: &CudaSlice<f32>,
        num_chunks: usize,
        batch_size: usize,
        num_heads: usize,
        query_len: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, ChunkedPrefillError> {
        let output_size = batch_size * num_heads * query_len * head_dim;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        let total_work = batch_size * num_heads * query_len;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (num_chunks * std::mem::size_of::<f32>()) as u32,
        };

        let chunks_i32 = num_chunks as i32;
        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let query_i32 = query_len as i32;
        let dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_merge_f32);
            builder.arg(chunk_outputs);
            builder.arg(chunk_lse);
            builder.arg(&mut output);
            builder.arg(&chunks_i32);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&query_i32);
            builder.arg(&dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Merge chunk outputs using log-sum-exp (f16).
    pub fn merge_chunks_f16(
        &self,
        stream: &Arc<CudaStream>,
        chunk_outputs: &CudaSlice<f16>,
        chunk_lse: &CudaSlice<f32>,  // LSE always f32
        num_chunks: usize,
        batch_size: usize,
        num_heads: usize,
        query_len: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f16>, ChunkedPrefillError> {
        let output_size = batch_size * num_heads * query_len * head_dim;
        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_size)?;

        let total_work = batch_size * num_heads * query_len;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (num_chunks * std::mem::size_of::<f32>()) as u32,
        };

        let chunks_i32 = num_chunks as i32;
        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let query_i32 = query_len as i32;
        let dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_merge_f16);
            builder.arg(chunk_outputs);
            builder.arg(chunk_lse);
            builder.arg(&mut output);
            builder.arg(&chunks_i32);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&query_i32);
            builder.arg(&dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Split workload for POD-Attention (f32/f16 agnostic).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `seq_lens` - Sequence lengths for each request [batch_size]
    /// * `batch_size` - Batch size
    /// * `target_chunk_size` - Target chunk size
    /// * `prefill_threshold` - Threshold to classify as prefill
    ///
    /// # Returns
    /// Tuple of (chunk_assignments [total_chunks], chunk_offsets [total_chunks],
    ///           prefill_mask [batch_size], num_prefill, num_decode)
    pub fn pod_attention_split(
        &self,
        stream: &Arc<CudaStream>,
        seq_lens: &CudaSlice<i32>,
        batch_size: usize,
        target_chunk_size: usize,
        prefill_threshold: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>), ChunkedPrefillError>
    {
        // Estimate max chunks (worst case: each request needs many chunks)
        let max_chunks = batch_size * 32;  // Conservative estimate

        let mut chunk_assignments: CudaSlice<i32> = stream.alloc_zeros(max_chunks)?;
        let mut chunk_offsets: CudaSlice<i32> = stream.alloc_zeros(max_chunks)?;
        let mut prefill_mask: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;
        let mut counts: CudaSlice<i32> = stream.alloc_zeros(3)?;  // [num_chunks, num_prefill, num_decode]

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (batch_size * std::mem::size_of::<i32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let chunk_size_i32 = target_chunk_size as i32;
        let threshold_i32 = prefill_threshold as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_pod_split_f32);
            builder.arg(seq_lens);
            builder.arg(&mut chunk_assignments);
            builder.arg(&mut chunk_offsets);
            builder.arg(&mut prefill_mask);
            builder.arg(&mut counts);
            builder.arg(&batch_i32);
            builder.arg(&chunk_size_i32);
            builder.arg(&threshold_i32);
            builder.launch(cfg)?;
        }

        Ok((chunk_assignments, chunk_offsets, prefill_mask, counts))
    }

    /// Schedule mixed prefill/decode batches.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `prefill_mask` - Prefill/decode classification [batch_size]
    /// * `seq_lens` - Sequence lengths [batch_size]
    /// * `batch_size` - Batch size
    /// * `max_prefill_batch` - Maximum prefill batch size
    /// * `max_decode_batch` - Maximum decode batch size
    ///
    /// # Returns
    /// Tuple of (schedule [max_batches * max_batch_size], batch_sizes [max_batches],
    ///           num_batches)
    pub fn schedule_batches(
        &self,
        stream: &Arc<CudaStream>,
        prefill_mask: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>,
        batch_size: usize,
        max_prefill_batch: usize,
        max_decode_batch: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>), ChunkedPrefillError> {
        let max_batch = max_prefill_batch.max(max_decode_batch);
        let max_batches = (batch_size + 1) / 2 + 1;  // Conservative estimate

        let mut schedule: CudaSlice<i32> = stream.alloc_zeros(max_batches * max_batch)?;
        let mut batch_sizes: CudaSlice<i32> = stream.alloc_zeros(max_batches)?;
        let mut num_batches: CudaSlice<i32> = stream.alloc_zeros(1)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE.min(batch_size as u32), 1, 1),
            shared_mem_bytes: (batch_size * 2 * std::mem::size_of::<i32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let max_prefill_i32 = max_prefill_batch as i32;
        let max_decode_i32 = max_decode_batch as i32;
        let max_batch_i32 = max_batch as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_schedule_f32);
            builder.arg(prefill_mask);
            builder.arg(seq_lens);
            builder.arg(&mut schedule);
            builder.arg(&mut batch_sizes);
            builder.arg(&mut num_batches);
            builder.arg(&batch_i32);
            builder.arg(&max_prefill_i32);
            builder.arg(&max_decode_i32);
            builder.arg(&max_batch_i32);
            builder.launch(cfg)?;
        }

        Ok((schedule, batch_sizes, num_batches))
    }
}

/// Configuration for Chunked Prefill CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct ChunkedPrefillCudaConfig {
    /// Target chunk size for attention.
    pub chunk_size: usize,
    /// Threshold sequence length to classify as prefill.
    pub prefill_threshold: usize,
    /// Maximum prefill batch size.
    pub max_prefill_batch: usize,
    /// Maximum decode batch size.
    pub max_decode_batch: usize,
    /// Whether to use causal masking.
    pub causal: bool,
}

impl Default for ChunkedPrefillCudaConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            prefill_threshold: 128,
            max_prefill_batch: 4,
            max_decode_batch: 256,
            causal: true,
        }
    }
}

impl ChunkedPrefillCudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), ChunkedPrefillError> {
        if self.chunk_size == 0 {
            return Err(ChunkedPrefillError::InvalidConfig(
                "chunk_size must be > 0".into(),
            ));
        }
        if self.prefill_threshold == 0 {
            return Err(ChunkedPrefillError::InvalidConfig(
                "prefill_threshold must be > 0".into(),
            ));
        }
        if self.max_prefill_batch == 0 {
            return Err(ChunkedPrefillError::InvalidConfig(
                "max_prefill_batch must be > 0".into(),
            ));
        }
        if self.max_decode_batch == 0 {
            return Err(ChunkedPrefillError::InvalidConfig(
                "max_decode_batch must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Compute number of chunks for a given sequence length.
    pub fn num_chunks(&self, seq_len: usize) -> usize {
        (seq_len + self.chunk_size - 1) / self.chunk_size
    }

    /// Check if a sequence should be treated as prefill.
    pub fn is_prefill(&self, seq_len: usize) -> bool {
        seq_len >= self.prefill_threshold
    }
}
