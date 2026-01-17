//! Metal Prompt Caching/CacheBlend kernels.
//!
//! This module provides Metal GPU-accelerated kernels for prompt caching:
//! - Content-based prompt hashing (xxHash64-style)
//! - Prefix matching and cache lookup
//! - KV cache blending for partial matches
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

const KERNEL_COMPUTE_HASH_F32: &str = "prompt_cache_compute_hash_f32";
const KERNEL_COMPUTE_HASH_F16: &str = "prompt_cache_compute_hash_f16";
const KERNEL_FIND_PREFIX_MATCH: &str = "prompt_cache_find_prefix_match";
const KERNEL_CACHE_BLEND_F32: &str = "prompt_cache_blend_f32";
const KERNEL_CACHE_BLEND_F16: &str = "prompt_cache_blend_f16";
const KERNEL_COPY_KV_F32: &str = "prompt_cache_copy_kv_f32";
const KERNEL_COPY_KV_F16: &str = "prompt_cache_copy_kv_f16";

/// Metallib collection for Prompt Cache kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static PROMPT_CACHE_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "prompt_cache",
    metallib_data: include_bytes!("kernels/prompt_cache.metallib"),
};

/// Parameters for hash computation kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ComputeHashParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    hash_window: u32,
}

/// Parameters for prefix match kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FindPrefixMatchParams {
    num_queries: u32,
    num_cache_entries: u32,
    hash_len: u32,
    min_match_len: u32,
}

/// Parameters for cache blend kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct CacheBlendParams {
    batch_size: u32,
    match_len: u32,
    new_len: u32,
    num_heads: u32,
    head_dim: u32,
    blend_factor: f32,
    _pad: [u32; 2],
}

/// Parameters for KV copy kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct CopyKVParams {
    batch_size: u32,
    copy_len: u32,
    num_heads: u32,
    head_dim: u32,
}

/// Errors surfaced by the Metal Prompt Cache kernels.
#[derive(Debug)]
pub enum PromptCacheError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for PromptCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for PromptCacheError {}

impl From<MetallibLoadError> for PromptCacheError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Configuration for Prompt Cache operations.
#[derive(Clone, Debug)]
pub struct PromptCacheConfig {
    /// Window size for rolling hash computation.
    pub hash_window: usize,
    /// Minimum match length to consider a cache hit.
    pub min_match_len: usize,
    /// Blend factor for combining cached and new KV (0=all cached, 1=all new).
    pub blend_factor: f32,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            hash_window: 64,
            min_match_len: 32,
            blend_factor: 0.0,
        }
    }
}

impl PromptCacheConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), PromptCacheError> {
        if self.hash_window == 0 {
            return Err(PromptCacheError::InvalidConfig("hash_window must be positive".into()));
        }
        if self.min_match_len == 0 {
            return Err(PromptCacheError::InvalidConfig("min_match_len must be positive".into()));
        }
        if self.blend_factor < 0.0 || self.blend_factor > 1.0 {
            return Err(PromptCacheError::InvalidConfig("blend_factor must be in [0, 1]".into()));
        }
        Ok(())
    }
}

/// Prompt Cache Metal kernel wrapper.
pub struct PromptCacheKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_compute_hash_f32: ComputePipelineState,
    pipeline_compute_hash_f16: ComputePipelineState,
    pipeline_find_prefix_match: ComputePipelineState,
    pipeline_cache_blend_f32: ComputePipelineState,
    pipeline_cache_blend_f16: ComputePipelineState,
    pipeline_copy_kv_f32: ComputePipelineState,
    pipeline_copy_kv_f16: ComputePipelineState,
}

impl PromptCacheKernel {
    /// Load Prompt Cache kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, PromptCacheError> {
        let library = load_library(device)?;

        let pipeline_compute_hash_f32 = build_pipeline(device, &library, KERNEL_COMPUTE_HASH_F32)?;
        let pipeline_compute_hash_f16 = build_pipeline(device, &library, KERNEL_COMPUTE_HASH_F16)?;
        let pipeline_find_prefix_match = build_pipeline(device, &library, KERNEL_FIND_PREFIX_MATCH)?;
        let pipeline_cache_blend_f32 = build_pipeline(device, &library, KERNEL_CACHE_BLEND_F32)?;
        let pipeline_cache_blend_f16 = build_pipeline(device, &library, KERNEL_CACHE_BLEND_F16)?;
        let pipeline_copy_kv_f32 = build_pipeline(device, &library, KERNEL_COPY_KV_F32)?;
        let pipeline_copy_kv_f16 = build_pipeline(device, &library, KERNEL_COPY_KV_F16)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_compute_hash_f32,
            pipeline_compute_hash_f16,
            pipeline_find_prefix_match,
            pipeline_cache_blend_f32,
            pipeline_cache_blend_f16,
            pipeline_copy_kv_f32,
            pipeline_copy_kv_f16,
        })
    }

    /// Compute rolling hash for prompt tokens (f32).
    ///
    /// # Arguments
    /// * `token_embeddings` - Token embeddings: [batch, seq_len, hidden_dim]
    ///
    /// # Returns
    /// Rolling hashes: [batch, seq_len] as u64
    pub fn compute_hash_f32(
        &self,
        token_embeddings: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        hash_window: usize,
    ) -> Result<Buffer, PromptCacheError> {
        self.compute_hash_impl(
            token_embeddings,
            batch_size, seq_len, hidden_dim, hash_window,
            &self.pipeline_compute_hash_f32,
        )
    }

    /// Compute rolling hash for prompt tokens (f16).
    pub fn compute_hash_f16(
        &self,
        token_embeddings: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        hash_window: usize,
    ) -> Result<Buffer, PromptCacheError> {
        self.compute_hash_impl(
            token_embeddings,
            batch_size, seq_len, hidden_dim, hash_window,
            &self.pipeline_compute_hash_f16,
        )
    }

    fn compute_hash_impl(
        &self,
        token_embeddings: &Buffer,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        hash_window: usize,
        pipeline: &ComputePipelineState,
    ) -> Result<Buffer, PromptCacheError> {
        let params = ComputeHashParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            hash_window: hash_window as u32,
        };

        // Output: hash per position as u64
        let output_bytes = (batch_size * seq_len * mem::size_of::<u64>()) as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(token_embeddings), 0);
        encoder.set_buffer(1, Some(&output), 0);

        let params_size = mem::size_of::<ComputeHashParams>() as u64;
        encoder.set_bytes(2, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * seq_len) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Find prefix matches in cache.
    ///
    /// # Arguments
    /// * `query_hashes` - Hashes of query prompts: [num_queries, hash_len]
    /// * `cache_hashes` - Hashes of cached prompts: [num_cache_entries, hash_len]
    ///
    /// # Returns
    /// (match_indices, match_lengths): Index of best match and length matched
    pub fn find_prefix_match(
        &self,
        query_hashes: &Buffer,
        cache_hashes: &Buffer,
        num_queries: usize,
        num_cache_entries: usize,
        hash_len: usize,
        min_match_len: usize,
    ) -> Result<(Buffer, Buffer), PromptCacheError> {
        let params = FindPrefixMatchParams {
            num_queries: num_queries as u32,
            num_cache_entries: num_cache_entries as u32,
            hash_len: hash_len as u32,
            min_match_len: min_match_len as u32,
        };

        // Output: best match index [num_queries] and match length [num_queries]
        let indices_bytes = (num_queries * mem::size_of::<i32>()) as u64; // -1 for no match
        let lengths_bytes = (num_queries * mem::size_of::<u32>()) as u64;

        let match_indices = self.device.new_buffer(indices_bytes, MTLResourceOptions::StorageModeShared);
        let match_lengths = self.device.new_buffer(lengths_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_find_prefix_match);
        encoder.set_buffer(0, Some(query_hashes), 0);
        encoder.set_buffer(1, Some(cache_hashes), 0);
        encoder.set_buffer(2, Some(&match_indices), 0);
        encoder.set_buffer(3, Some(&match_lengths), 0);

        let params_size = mem::size_of::<FindPrefixMatchParams>() as u64;
        encoder.set_bytes(4, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(num_queries as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_find_prefix_match);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((match_indices, match_lengths))
    }

    /// Blend cached KV with newly computed KV (f32).
    ///
    /// # Arguments
    /// * `cached_kv` - Cached KV: [batch, match_len, num_heads, 2, head_dim]
    /// * `new_kv` - Newly computed KV: [batch, new_len, num_heads, 2, head_dim]
    /// * `blend_factor` - 0=all cached, 1=all new (for overlapping region)
    pub fn cache_blend_f32(
        &self,
        cached_kv: &Buffer,
        new_kv: &Buffer,
        batch_size: usize,
        match_len: usize,
        new_len: usize,
        num_heads: usize,
        head_dim: usize,
        blend_factor: f32,
    ) -> Result<Buffer, PromptCacheError> {
        self.cache_blend_impl(
            cached_kv, new_kv,
            batch_size, match_len, new_len, num_heads, head_dim, blend_factor,
            &self.pipeline_cache_blend_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Blend cached KV with newly computed KV (f16).
    pub fn cache_blend_f16(
        &self,
        cached_kv: &Buffer,
        new_kv: &Buffer,
        batch_size: usize,
        match_len: usize,
        new_len: usize,
        num_heads: usize,
        head_dim: usize,
        blend_factor: f32,
    ) -> Result<Buffer, PromptCacheError> {
        self.cache_blend_impl(
            cached_kv, new_kv,
            batch_size, match_len, new_len, num_heads, head_dim, blend_factor,
            &self.pipeline_cache_blend_f16,
            mem::size_of::<u16>(),
        )
    }

    fn cache_blend_impl(
        &self,
        cached_kv: &Buffer,
        new_kv: &Buffer,
        batch_size: usize,
        match_len: usize,
        new_len: usize,
        num_heads: usize,
        head_dim: usize,
        blend_factor: f32,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, PromptCacheError> {
        let params = CacheBlendParams {
            batch_size: batch_size as u32,
            match_len: match_len as u32,
            new_len: new_len as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            blend_factor,
            _pad: [0; 2],
        };

        // Total length is match_len + (new_len - overlap)
        // For simplicity, output is match_len + new_len (handles both cached prefix and new suffix)
        let total_len = match_len + new_len;
        let output_elements = batch_size * total_len * num_heads * 2 * head_dim;
        let output_bytes = (output_elements * element_size) as u64;

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(cached_kv), 0);
        encoder.set_buffer(1, Some(new_kv), 0);
        encoder.set_buffer(2, Some(&output), 0);

        let params_size = mem::size_of::<CacheBlendParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * total_len * num_heads) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Copy KV cache entries (f32).
    pub fn copy_kv_f32(
        &self,
        src_kv: &Buffer,
        batch_size: usize,
        copy_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Buffer, PromptCacheError> {
        self.copy_kv_impl(
            src_kv,
            batch_size, copy_len, num_heads, head_dim,
            &self.pipeline_copy_kv_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Copy KV cache entries (f16).
    pub fn copy_kv_f16(
        &self,
        src_kv: &Buffer,
        batch_size: usize,
        copy_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Buffer, PromptCacheError> {
        self.copy_kv_impl(
            src_kv,
            batch_size, copy_len, num_heads, head_dim,
            &self.pipeline_copy_kv_f16,
            mem::size_of::<u16>(),
        )
    }

    fn copy_kv_impl(
        &self,
        src_kv: &Buffer,
        batch_size: usize,
        copy_len: usize,
        num_heads: usize,
        head_dim: usize,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, PromptCacheError> {
        let params = CopyKVParams {
            batch_size: batch_size as u32,
            copy_len: copy_len as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
        };

        let output_elements = batch_size * copy_len * num_heads * 2 * head_dim;
        let output_bytes = (output_elements * element_size) as u64;

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(src_kv), 0);
        encoder.set_buffer(1, Some(&output), 0);

        let params_size = mem::size_of::<CopyKVParams>() as u64;
        encoder.set_bytes(2, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * copy_len * num_heads) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, PromptCacheError> {
    PROMPT_CACHE_METALLIB.load(device).map_err(PromptCacheError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, PromptCacheError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| PromptCacheError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(PromptCacheError::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
