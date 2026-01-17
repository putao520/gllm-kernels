//! Metal Chunked Prefill/POD-Attention kernels.
//!
//! This module provides Metal GPU-accelerated kernels for chunked prefill:
//! - Memory-efficient chunked attention computation
//! - POD-Attention workload splitting for mixed prefill/decode
//! - Online softmax for chunk merging
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

const KERNEL_CHUNKED_ATTENTION_F32: &str = "chunked_prefill_attention_f32";
const KERNEL_CHUNKED_ATTENTION_F16: &str = "chunked_prefill_attention_f16";
const KERNEL_MERGE_CHUNKS_F32: &str = "chunked_prefill_merge_f32";
const KERNEL_MERGE_CHUNKS_F16: &str = "chunked_prefill_merge_f16";
const KERNEL_POD_ATTENTION_SPLIT: &str = "chunked_prefill_pod_split";
const KERNEL_SCHEDULE_BATCHES: &str = "chunked_prefill_schedule_batches";

/// Metallib collection for Chunked Prefill kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static CHUNKED_PREFILL_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "chunked_prefill",
    metallib_data: include_bytes!("kernels/chunked_prefill.metallib"),
};

/// Parameters for chunked attention kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ChunkedAttentionParams {
    batch_size: u32,
    num_heads: u32,
    chunk_size: u32,
    num_chunks: u32,
    head_dim: u32,
    scale: f32,
    is_causal: u32,
    _pad: u32,
}

/// Parameters for merge chunks kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct MergeChunksParams {
    batch_size: u32,
    num_heads: u32,
    num_chunks: u32,
    head_dim: u32,
}

/// Parameters for POD attention split kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PodSplitParams {
    batch_size: u32,
    total_tokens: u32,
    prefill_ratio: f32,
    _pad: u32,
}

/// Parameters for schedule batches kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ScheduleBatchesParams {
    num_requests: u32,
    max_batch_size: u32,
    max_tokens_per_batch: u32,
    _pad: u32,
}

/// Errors surfaced by the Metal Chunked Prefill kernels.
#[derive(Debug)]
pub enum ChunkedPrefillError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for ChunkedPrefillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for ChunkedPrefillError {}

impl From<MetallibLoadError> for ChunkedPrefillError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Configuration for Chunked Prefill operations.
#[derive(Clone, Debug)]
pub struct ChunkedPrefillConfig {
    /// Size of each chunk for attention computation.
    pub chunk_size: usize,
    /// Maximum batch size for scheduling.
    pub max_batch_size: usize,
    /// Maximum tokens per batch for scheduling.
    pub max_tokens_per_batch: usize,
    /// Whether to use causal masking.
    pub is_causal: bool,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            max_batch_size: 32,
            max_tokens_per_batch: 8192,
            is_causal: true,
        }
    }
}

impl ChunkedPrefillConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), ChunkedPrefillError> {
        if self.chunk_size == 0 {
            return Err(ChunkedPrefillError::InvalidConfig("chunk_size must be positive".into()));
        }
        if self.max_batch_size == 0 {
            return Err(ChunkedPrefillError::InvalidConfig("max_batch_size must be positive".into()));
        }
        if self.max_tokens_per_batch == 0 {
            return Err(ChunkedPrefillError::InvalidConfig("max_tokens_per_batch must be positive".into()));
        }
        Ok(())
    }
}

/// Chunked Prefill Metal kernel wrapper.
pub struct ChunkedPrefillKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_chunked_attn_f32: ComputePipelineState,
    pipeline_chunked_attn_f16: ComputePipelineState,
    pipeline_merge_chunks_f32: ComputePipelineState,
    pipeline_merge_chunks_f16: ComputePipelineState,
    pipeline_pod_split: ComputePipelineState,
    pipeline_schedule_batches: ComputePipelineState,
}

impl ChunkedPrefillKernel {
    /// Load Chunked Prefill kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, ChunkedPrefillError> {
        let library = load_library(device)?;

        let pipeline_chunked_attn_f32 = build_pipeline(device, &library, KERNEL_CHUNKED_ATTENTION_F32)?;
        let pipeline_chunked_attn_f16 = build_pipeline(device, &library, KERNEL_CHUNKED_ATTENTION_F16)?;
        let pipeline_merge_chunks_f32 = build_pipeline(device, &library, KERNEL_MERGE_CHUNKS_F32)?;
        let pipeline_merge_chunks_f16 = build_pipeline(device, &library, KERNEL_MERGE_CHUNKS_F16)?;
        let pipeline_pod_split = build_pipeline(device, &library, KERNEL_POD_ATTENTION_SPLIT)?;
        let pipeline_schedule_batches = build_pipeline(device, &library, KERNEL_SCHEDULE_BATCHES)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_chunked_attn_f32,
            pipeline_chunked_attn_f16,
            pipeline_merge_chunks_f32,
            pipeline_merge_chunks_f16,
            pipeline_pod_split,
            pipeline_schedule_batches,
        })
    }

    /// Compute chunked attention for a single chunk (f32).
    ///
    /// # Arguments
    /// * `q` - Query: [batch, num_heads, chunk_size, head_dim]
    /// * `k` - Key: [batch, num_heads, total_kv_len, head_dim]
    /// * `v` - Value: [batch, num_heads, total_kv_len, head_dim]
    /// * `chunk_idx` - Which chunk this is (for causal masking)
    ///
    /// # Returns
    /// (chunk_output, chunk_lse): Output and log-sum-exp for merging
    pub fn chunked_attention_f32(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        batch_size: usize,
        num_heads: usize,
        chunk_size: usize,
        total_kv_len: usize,
        head_dim: usize,
        chunk_idx: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<(Buffer, Buffer), ChunkedPrefillError> {
        self.chunked_attention_impl(
            q, k, v,
            batch_size, num_heads, chunk_size, total_kv_len, head_dim, chunk_idx, scale, is_causal,
            &self.pipeline_chunked_attn_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Compute chunked attention for a single chunk (f16).
    pub fn chunked_attention_f16(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        batch_size: usize,
        num_heads: usize,
        chunk_size: usize,
        total_kv_len: usize,
        head_dim: usize,
        chunk_idx: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<(Buffer, Buffer), ChunkedPrefillError> {
        self.chunked_attention_impl(
            q, k, v,
            batch_size, num_heads, chunk_size, total_kv_len, head_dim, chunk_idx, scale, is_causal,
            &self.pipeline_chunked_attn_f16,
            mem::size_of::<u16>(),
        )
    }

    fn chunked_attention_impl(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        batch_size: usize,
        num_heads: usize,
        chunk_size: usize,
        _total_kv_len: usize,
        head_dim: usize,
        chunk_idx: usize,
        scale: f32,
        is_causal: bool,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<(Buffer, Buffer), ChunkedPrefillError> {
        let num_chunks = chunk_idx + 1; // For this chunk

        let params = ChunkedAttentionParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            chunk_size: chunk_size as u32,
            num_chunks: num_chunks as u32,
            head_dim: head_dim as u32,
            scale,
            is_causal: if is_causal { 1 } else { 0 },
            _pad: 0,
        };

        // Output: chunk attention output and log-sum-exp
        let output_elements = batch_size * num_heads * chunk_size * head_dim;
        let output_bytes = (output_elements * element_size) as u64;
        let lse_elements = batch_size * num_heads * chunk_size;
        let lse_bytes = (lse_elements * element_size) as u64;

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);
        let lse = self.device.new_buffer(lse_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k), 0);
        encoder.set_buffer(2, Some(v), 0);
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_buffer(4, Some(&lse), 0);

        let params_size = mem::size_of::<ChunkedAttentionParams>() as u64;
        encoder.set_bytes(5, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * num_heads * chunk_size) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((output, lse))
    }

    /// Merge chunk outputs using online softmax (f32).
    ///
    /// # Arguments
    /// * `chunk_outputs` - Outputs from all chunks: [num_chunks][batch, num_heads, chunk_size, head_dim]
    /// * `chunk_lses` - Log-sum-exp from all chunks: [num_chunks][batch, num_heads, chunk_size]
    pub fn merge_chunks_f32(
        &self,
        chunk_outputs: &[&Buffer],
        chunk_lses: &[&Buffer],
        batch_size: usize,
        num_heads: usize,
        chunk_size: usize,
        head_dim: usize,
    ) -> Result<Buffer, ChunkedPrefillError> {
        self.merge_chunks_impl(
            chunk_outputs, chunk_lses,
            batch_size, num_heads, chunk_size, head_dim,
            &self.pipeline_merge_chunks_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Merge chunk outputs using online softmax (f16).
    pub fn merge_chunks_f16(
        &self,
        chunk_outputs: &[&Buffer],
        chunk_lses: &[&Buffer],
        batch_size: usize,
        num_heads: usize,
        chunk_size: usize,
        head_dim: usize,
    ) -> Result<Buffer, ChunkedPrefillError> {
        self.merge_chunks_impl(
            chunk_outputs, chunk_lses,
            batch_size, num_heads, chunk_size, head_dim,
            &self.pipeline_merge_chunks_f16,
            mem::size_of::<u16>(),
        )
    }

    fn merge_chunks_impl(
        &self,
        chunk_outputs: &[&Buffer],
        chunk_lses: &[&Buffer],
        batch_size: usize,
        num_heads: usize,
        chunk_size: usize,
        head_dim: usize,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, ChunkedPrefillError> {
        let num_chunks = chunk_outputs.len();
        if num_chunks != chunk_lses.len() {
            return Err(ChunkedPrefillError::InvalidConfig(
                "chunk_outputs and chunk_lses must have same length".into(),
            ));
        }

        let params = MergeChunksParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            num_chunks: num_chunks as u32,
            head_dim: head_dim as u32,
        };

        // For simplicity, assume chunks are stored contiguously (in reality would need to concatenate)
        // Output: merged attention [batch, num_heads, total_seq_len, head_dim]
        let total_seq_len = num_chunks * chunk_size;
        let output_elements = batch_size * num_heads * total_seq_len * head_dim;
        let output_bytes = (output_elements * element_size) as u64;

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        // Bind chunk buffers
        for (i, buf) in chunk_outputs.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }
        let outputs_end = num_chunks as u64;
        for (i, buf) in chunk_lses.iter().enumerate() {
            encoder.set_buffer(outputs_end + i as u64, Some(*buf), 0);
        }
        encoder.set_buffer(outputs_end + num_chunks as u64, Some(&output), 0);

        let params_size = mem::size_of::<MergeChunksParams>() as u64;
        encoder.set_bytes(outputs_end + num_chunks as u64 + 1, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * num_heads * total_seq_len) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Split workload for POD-Attention (Prefill On Decode).
    ///
    /// # Arguments
    /// * `token_types` - 0=prefill, 1=decode for each token: [batch, total_tokens]
    ///
    /// # Returns
    /// (prefill_indices, decode_indices, counts): Indices for each type and counts
    pub fn pod_attention_split(
        &self,
        token_types: &Buffer,
        batch_size: usize,
        total_tokens: usize,
    ) -> Result<(Buffer, Buffer, Buffer), ChunkedPrefillError> {
        let prefill_ratio = 0.5f32; // Will be computed from actual data

        let params = PodSplitParams {
            batch_size: batch_size as u32,
            total_tokens: total_tokens as u32,
            prefill_ratio,
            _pad: 0,
        };

        // Output: indices for prefill and decode tokens, plus counts
        let max_elements = batch_size * total_tokens;
        let indices_bytes = (max_elements * mem::size_of::<u32>()) as u64;
        let counts_bytes = (batch_size * 2 * mem::size_of::<u32>()) as u64; // prefill_count, decode_count per batch

        let prefill_indices = self.device.new_buffer(indices_bytes, MTLResourceOptions::StorageModeShared);
        let decode_indices = self.device.new_buffer(indices_bytes, MTLResourceOptions::StorageModeShared);
        let counts = self.device.new_buffer(counts_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_pod_split);
        encoder.set_buffer(0, Some(token_types), 0);
        encoder.set_buffer(1, Some(&prefill_indices), 0);
        encoder.set_buffer(2, Some(&decode_indices), 0);
        encoder.set_buffer(3, Some(&counts), 0);

        let params_size = mem::size_of::<PodSplitParams>() as u64;
        encoder.set_bytes(4, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_pod_split);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((prefill_indices, decode_indices, counts))
    }

    /// Schedule requests into batches for efficient processing.
    ///
    /// # Arguments
    /// * `request_lengths` - Token count for each request: [num_requests]
    ///
    /// # Returns
    /// (batch_assignments, batch_sizes): Which batch each request belongs to
    pub fn schedule_batches(
        &self,
        request_lengths: &Buffer,
        num_requests: usize,
        config: &ChunkedPrefillConfig,
    ) -> Result<(Buffer, Buffer), ChunkedPrefillError> {
        config.validate()?;

        let params = ScheduleBatchesParams {
            num_requests: num_requests as u32,
            max_batch_size: config.max_batch_size as u32,
            max_tokens_per_batch: config.max_tokens_per_batch as u32,
            _pad: 0,
        };

        // Output: batch assignment per request and batch sizes
        let assignments_bytes = (num_requests * mem::size_of::<u32>()) as u64;
        let max_batches = (num_requests + config.max_batch_size - 1) / config.max_batch_size;
        let batch_sizes_bytes = (max_batches * mem::size_of::<u32>()) as u64;

        let batch_assignments = self.device.new_buffer(assignments_bytes, MTLResourceOptions::StorageModeShared);
        let batch_sizes = self.device.new_buffer(batch_sizes_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_schedule_batches);
        encoder.set_buffer(0, Some(request_lengths), 0);
        encoder.set_buffer(1, Some(&batch_assignments), 0);
        encoder.set_buffer(2, Some(&batch_sizes), 0);

        let params_size = mem::size_of::<ScheduleBatchesParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        // Single thread for scheduling algorithm
        let threads_per_grid = MTLSize::new(1, 1, 1);
        let threads_per_threadgroup = MTLSize::new(1, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((batch_assignments, batch_sizes))
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, ChunkedPrefillError> {
    CHUNKED_PREFILL_METALLIB.load(device).map_err(ChunkedPrefillError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, ChunkedPrefillError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| ChunkedPrefillError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(ChunkedPrefillError::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
