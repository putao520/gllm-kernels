//! CUDA Prompt Caching / CacheBlend kernel.
//!
//! Implements efficient KV cache reuse for repeated prompts:
//! - Fast hash computation for prompt fingerprinting
//! - Prefix matching with trie-like lookup
//! - CacheBlend for partial cache hits
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
const KERNEL_COMPUTE_HASH_F32: &str = "prompt_cache_compute_hash_f32";
const KERNEL_COMPUTE_HASH_F16: &str = "prompt_cache_compute_hash_f16";
const KERNEL_PREFIX_MATCH_F32: &str = "prompt_cache_prefix_match_f32";
const KERNEL_PREFIX_MATCH_F16: &str = "prompt_cache_prefix_match_f16";
const KERNEL_CACHE_BLEND_F32: &str = "prompt_cache_blend_f32";
const KERNEL_CACHE_BLEND_F16: &str = "prompt_cache_blend_f16";
const KERNEL_COPY_KV_F32: &str = "prompt_cache_copy_kv_f32";
const KERNEL_COPY_KV_F16: &str = "prompt_cache_copy_kv_f16";
const KERNEL_ROLLING_HASH: &str = "prompt_cache_rolling_hash";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for Prompt Cache kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static PROMPT_CACHE_PTX: PtxCollection = PtxCollection {
    kernel_name: "prompt_cache",
    ptx_versions: &[
        (61, include_str!("kernels/prompt_cache_sm61.ptx")),
        (80, include_str!("kernels/prompt_cache.ptx")),
    ],
};

/// Errors surfaced by the CUDA Prompt Cache kernels.
#[derive(Debug)]
pub enum PromptCacheError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
    /// Cache miss.
    CacheMiss(String),
}

impl fmt::Display for PromptCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
            Self::CacheMiss(msg) => write!(f, "Cache miss: {msg}"),
        }
    }
}

impl std::error::Error for PromptCacheError {}

impl From<DriverError> for PromptCacheError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for PromptCacheError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// Prompt Cache CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Prompt hash computation (xxHash64-style)
/// - Prefix matching for cache lookup
/// - CacheBlend for partial hits
/// - KV cache copying
pub struct PromptCacheKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Hash computation kernels
    kernel_compute_hash_f32: CudaFunction,
    kernel_compute_hash_f16: CudaFunction,
    // Prefix matching kernels
    kernel_prefix_match_f32: CudaFunction,
    kernel_prefix_match_f16: CudaFunction,
    // CacheBlend kernels
    kernel_blend_f32: CudaFunction,
    kernel_blend_f16: CudaFunction,
    // KV copy kernels
    kernel_copy_kv_f32: CudaFunction,
    kernel_copy_kv_f16: CudaFunction,
    // Rolling hash kernel
    kernel_rolling_hash: CudaFunction,
}

impl PromptCacheKernel {
    /// Load Prompt Cache kernel module on the given device.
    ///
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, PromptCacheError> {
        let ptx = PROMPT_CACHE_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_compute_hash_f32 = module
            .load_function(KERNEL_COMPUTE_HASH_F32)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_COMPUTE_HASH_F32))?;
        let kernel_compute_hash_f16 = module
            .load_function(KERNEL_COMPUTE_HASH_F16)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_COMPUTE_HASH_F16))?;
        let kernel_prefix_match_f32 = module
            .load_function(KERNEL_PREFIX_MATCH_F32)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_PREFIX_MATCH_F32))?;
        let kernel_prefix_match_f16 = module
            .load_function(KERNEL_PREFIX_MATCH_F16)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_PREFIX_MATCH_F16))?;
        let kernel_blend_f32 = module
            .load_function(KERNEL_CACHE_BLEND_F32)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_CACHE_BLEND_F32))?;
        let kernel_blend_f16 = module
            .load_function(KERNEL_CACHE_BLEND_F16)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_CACHE_BLEND_F16))?;
        let kernel_copy_kv_f32 = module
            .load_function(KERNEL_COPY_KV_F32)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_COPY_KV_F32))?;
        let kernel_copy_kv_f16 = module
            .load_function(KERNEL_COPY_KV_F16)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_COPY_KV_F16))?;
        let kernel_rolling_hash = module
            .load_function(KERNEL_ROLLING_HASH)
            .map_err(|_| PromptCacheError::KernelMissing(KERNEL_ROLLING_HASH))?;

        Ok(Self {
            module,
            kernel_compute_hash_f32,
            kernel_compute_hash_f16,
            kernel_prefix_match_f32,
            kernel_prefix_match_f16,
            kernel_blend_f32,
            kernel_blend_f16,
            kernel_copy_kv_f32,
            kernel_copy_kv_f16,
            kernel_rolling_hash,
        })
    }

    /// Compute hash of input tokens (f32).
    ///
    /// Uses xxHash64-style algorithm for fast fingerprinting.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `tokens` - Input token IDs [seq_len]
    /// * `seq_len` - Sequence length
    /// * `seed` - Hash seed
    ///
    /// # Returns
    /// Hash values for each prefix length: [seq_len]
    pub fn compute_hash_f32(
        &self,
        stream: &Arc<CudaStream>,
        tokens: &CudaSlice<i32>,
        seq_len: usize,
        seed: u64,
    ) -> Result<CudaSlice<u64>, PromptCacheError> {
        let mut hashes: CudaSlice<u64> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let seq_i32 = seq_len as i32;
        let seed_lo = seed as u32;
        let seed_hi = (seed >> 32) as u32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_compute_hash_f32);
            builder.arg(tokens);
            builder.arg(&mut hashes);
            builder.arg(&seq_i32);
            builder.arg(&seed_lo);
            builder.arg(&seed_hi);
            builder.launch(cfg)?;
        }

        Ok(hashes)
    }

    /// Compute hash of input tokens (f16) - same as f32 since tokens are integers.
    pub fn compute_hash_f16(
        &self,
        stream: &Arc<CudaStream>,
        tokens: &CudaSlice<i32>,
        seq_len: usize,
        seed: u64,
    ) -> Result<CudaSlice<u64>, PromptCacheError> {
        self.compute_hash_f32(stream, tokens, seq_len, seed)
    }

    /// Find longest matching prefix in cache.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `query_hashes` - Hash values for current prompt [query_len]
    /// * `cache_hashes` - Hash values in cache [num_entries * max_cache_len]
    /// * `cache_lengths` - Actual lengths of cached prompts [num_entries]
    /// * `query_len` - Length of current prompt
    /// * `num_entries` - Number of cache entries
    /// * `max_cache_len` - Maximum cached prompt length
    ///
    /// # Returns
    /// Tuple of (best_entry_idx, match_length)
    pub fn find_prefix_match(
        &self,
        stream: &Arc<CudaStream>,
        query_hashes: &CudaSlice<u64>,
        cache_hashes: &CudaSlice<u64>,
        cache_lengths: &CudaSlice<i32>,
        query_len: usize,
        num_entries: usize,
        max_cache_len: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>), PromptCacheError> {
        let mut best_entry: CudaSlice<i32> = stream.alloc_zeros(1)?;
        let mut match_length: CudaSlice<i32> = stream.alloc_zeros(1)?;

        // Use one block for reduction across entries
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE.min(num_entries as u32), 1, 1),
            shared_mem_bytes: (num_entries * 2 * std::mem::size_of::<i32>()) as u32,
        };

        let query_i32 = query_len as i32;
        let entries_i32 = num_entries as i32;
        let max_len_i32 = max_cache_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_prefix_match_f32);
            builder.arg(query_hashes);
            builder.arg(cache_hashes);
            builder.arg(cache_lengths);
            builder.arg(&mut best_entry);
            builder.arg(&mut match_length);
            builder.arg(&query_i32);
            builder.arg(&entries_i32);
            builder.arg(&max_len_i32);
            builder.launch(cfg)?;
        }

        Ok((best_entry, match_length))
    }

    /// Blend cached KV with fresh computation (f32).
    ///
    /// CacheBlend algorithm for partial cache hits.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `cached_kv` - Cached KV tensor [match_len * num_heads * head_dim]
    /// * `fresh_kv` - Freshly computed KV [fresh_len * num_heads * head_dim]
    /// * `match_len` - Length of cached prefix
    /// * `fresh_len` - Length of fresh computation (may overlap)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `blend_window` - Window size for blending at boundary
    ///
    /// # Returns
    /// Blended KV tensor: [(match_len + fresh_len) * num_heads * head_dim]
    pub fn cache_blend_f32(
        &self,
        stream: &Arc<CudaStream>,
        cached_kv: &CudaSlice<f32>,
        fresh_kv: &CudaSlice<f32>,
        match_len: usize,
        fresh_len: usize,
        num_heads: usize,
        head_dim: usize,
        blend_window: usize,
    ) -> Result<CudaSlice<f32>, PromptCacheError> {
        let total_len = match_len + fresh_len;
        let output_size = total_len * num_heads * head_dim;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let total_work = total_len * num_heads;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let match_i32 = match_len as i32;
        let fresh_i32 = fresh_len as i32;
        let heads_i32 = num_heads as i32;
        let dim_i32 = head_dim as i32;
        let window_i32 = blend_window as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_blend_f32);
            builder.arg(cached_kv);
            builder.arg(fresh_kv);
            builder.arg(&mut output);
            builder.arg(&match_i32);
            builder.arg(&fresh_i32);
            builder.arg(&heads_i32);
            builder.arg(&dim_i32);
            builder.arg(&window_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Blend cached KV with fresh computation (f16).
    pub fn cache_blend_f16(
        &self,
        stream: &Arc<CudaStream>,
        cached_kv: &CudaSlice<f16>,
        fresh_kv: &CudaSlice<f16>,
        match_len: usize,
        fresh_len: usize,
        num_heads: usize,
        head_dim: usize,
        blend_window: usize,
    ) -> Result<CudaSlice<f16>, PromptCacheError> {
        let total_len = match_len + fresh_len;
        let output_size = total_len * num_heads * head_dim;
        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let total_work = total_len * num_heads;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let match_i32 = match_len as i32;
        let fresh_i32 = fresh_len as i32;
        let heads_i32 = num_heads as i32;
        let dim_i32 = head_dim as i32;
        let window_i32 = blend_window as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_blend_f16);
            builder.arg(cached_kv);
            builder.arg(fresh_kv);
            builder.arg(&mut output);
            builder.arg(&match_i32);
            builder.arg(&fresh_i32);
            builder.arg(&heads_i32);
            builder.arg(&dim_i32);
            builder.arg(&window_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Copy KV cache from source to destination (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `src` - Source KV cache [src_len * num_heads * head_dim]
    /// * `src_offset` - Starting position in source
    /// * `copy_len` - Number of positions to copy
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    /// Copied KV tensor: [copy_len * num_heads * head_dim]
    pub fn copy_kv_f32(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
        src_offset: usize,
        copy_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, PromptCacheError> {
        let output_size = copy_len * num_heads * head_dim;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (output_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let offset_i32 = src_offset as i32;
        let len_i32 = copy_len as i32;
        let heads_i32 = num_heads as i32;
        let dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_copy_kv_f32);
            builder.arg(src);
            builder.arg(&mut output);
            builder.arg(&offset_i32);
            builder.arg(&len_i32);
            builder.arg(&heads_i32);
            builder.arg(&dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Copy KV cache from source to destination (f16).
    pub fn copy_kv_f16(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f16>,
        src_offset: usize,
        copy_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f16>, PromptCacheError> {
        let output_size = copy_len * num_heads * head_dim;
        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (output_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let offset_i32 = src_offset as i32;
        let len_i32 = copy_len as i32;
        let heads_i32 = num_heads as i32;
        let dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_copy_kv_f16);
            builder.arg(src);
            builder.arg(&mut output);
            builder.arg(&offset_i32);
            builder.arg(&len_i32);
            builder.arg(&heads_i32);
            builder.arg(&dim_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Update rolling hash incrementally.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `current_hash` - Current hash value [1]
    /// * `new_token` - New token to add
    /// * `old_token` - Token to remove (if window is full, else -1)
    /// * `window_size` - Rolling window size
    ///
    /// # Returns
    /// Updated hash value: [1]
    pub fn update_rolling_hash(
        &self,
        stream: &Arc<CudaStream>,
        current_hash: &CudaSlice<u64>,
        new_token: i32,
        old_token: i32,
        window_size: usize,
    ) -> Result<CudaSlice<u64>, PromptCacheError> {
        let mut output: CudaSlice<u64> = stream.alloc_zeros(1)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let window_i32 = window_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_rolling_hash);
            builder.arg(current_hash);
            builder.arg(&mut output);
            builder.arg(&new_token);
            builder.arg(&old_token);
            builder.arg(&window_i32);
            builder.launch(cfg)?;
        }

        Ok(output)
    }
}

/// Configuration for Prompt Cache CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct PromptCacheCudaConfig {
    /// Maximum number of cache entries.
    pub max_entries: usize,
    /// Maximum cached prompt length per entry.
    pub max_prompt_len: usize,
    /// Blend window size for CacheBlend.
    pub blend_window: usize,
    /// Hash seed.
    pub hash_seed: u64,
    /// Minimum match length to use cache.
    pub min_match_len: usize,
}

impl Default for PromptCacheCudaConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_prompt_len: 4096,
            blend_window: 16,
            hash_seed: 0x9e3779b97f4a7c15,  // xxHash64 seed
            min_match_len: 32,
        }
    }
}

impl PromptCacheCudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), PromptCacheError> {
        if self.max_entries == 0 {
            return Err(PromptCacheError::InvalidConfig(
                "max_entries must be > 0".into(),
            ));
        }
        if self.max_prompt_len == 0 {
            return Err(PromptCacheError::InvalidConfig(
                "max_prompt_len must be > 0".into(),
            ));
        }
        if self.blend_window == 0 {
            return Err(PromptCacheError::InvalidConfig(
                "blend_window must be > 0".into(),
            ));
        }
        if self.min_match_len == 0 {
            return Err(PromptCacheError::InvalidConfig(
                "min_match_len must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Get total hash storage size.
    pub fn hash_storage_size(&self) -> usize {
        self.max_entries * self.max_prompt_len
    }
}
