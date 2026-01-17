//! CUDA EvicPress joint compression-eviction kernel.
//!
//! Implements three-zone KV cache management:
//! - Hot zone: FP16 full precision (recent tokens)
//! - Warm zone: INT8 quantized (moderately aged tokens)
//! - Cold zone: INT2 heavily compressed (oldest tokens)
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
const KERNEL_COMPUTE_IMPORTANCE_F32: &str = "evicpress_compute_importance_f32";
const KERNEL_COMPUTE_IMPORTANCE_F16: &str = "evicpress_compute_importance_f16";
const KERNEL_ZONE_TRANSITION_F32: &str = "evicpress_zone_transition_f32";
const KERNEL_ZONE_TRANSITION_F16: &str = "evicpress_zone_transition_f16";
const KERNEL_COMPRESS_TO_INT8_F32: &str = "evicpress_compress_int8_f32";
const KERNEL_COMPRESS_TO_INT8_F16: &str = "evicpress_compress_int8_f16";
const KERNEL_COMPRESS_TO_INT2_F32: &str = "evicpress_compress_int2_f32";
const KERNEL_COMPRESS_TO_INT2_F16: &str = "evicpress_compress_int2_f16";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for EvicPress kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static EVICPRESS_PTX: PtxCollection = PtxCollection {
    kernel_name: "evic_press",
    ptx_versions: &[
        (61, include_str!("kernels/evic_press_sm61.ptx")),
        (80, include_str!("kernels/evic_press.ptx")),
    ],
};

/// Errors surfaced by the CUDA EvicPress kernels.
#[derive(Debug)]
pub enum EvicPressError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
    /// Zone transition error.
    ZoneTransition(String),
}

impl fmt::Display for EvicPressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
            Self::ZoneTransition(msg) => write!(f, "Zone transition error: {msg}"),
        }
    }
}

impl std::error::Error for EvicPressError {}

impl From<DriverError> for EvicPressError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for EvicPressError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// Cache zone identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CacheZone {
    /// Hot zone: FP16 full precision.
    Hot = 0,
    /// Warm zone: INT8 quantized.
    Warm = 1,
    /// Cold zone: INT2 compressed.
    Cold = 2,
    /// Evicted: no longer in cache.
    Evicted = 3,
}

/// EvicPress CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Token importance computation
/// - Zone transition decisions
/// - FP16 → INT8 compression
/// - INT8 → INT2 compression
pub struct EvicPressKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Importance computation kernels
    kernel_importance_f32: CudaFunction,
    kernel_importance_f16: CudaFunction,
    // Zone transition kernels
    kernel_zone_transition_f32: CudaFunction,
    kernel_zone_transition_f16: CudaFunction,
    // Compression kernels
    kernel_compress_int8_f32: CudaFunction,
    kernel_compress_int8_f16: CudaFunction,
    kernel_compress_int2_f32: CudaFunction,
    kernel_compress_int2_f16: CudaFunction,
}

impl EvicPressKernel {
    /// Load EvicPress kernel module on the given device.
    ///
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, EvicPressError> {
        let ptx = EVICPRESS_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_importance_f32 = module
            .load_function(KERNEL_COMPUTE_IMPORTANCE_F32)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_COMPUTE_IMPORTANCE_F32))?;
        let kernel_importance_f16 = module
            .load_function(KERNEL_COMPUTE_IMPORTANCE_F16)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_COMPUTE_IMPORTANCE_F16))?;
        let kernel_zone_transition_f32 = module
            .load_function(KERNEL_ZONE_TRANSITION_F32)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_ZONE_TRANSITION_F32))?;
        let kernel_zone_transition_f16 = module
            .load_function(KERNEL_ZONE_TRANSITION_F16)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_ZONE_TRANSITION_F16))?;
        let kernel_compress_int8_f32 = module
            .load_function(KERNEL_COMPRESS_TO_INT8_F32)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_COMPRESS_TO_INT8_F32))?;
        let kernel_compress_int8_f16 = module
            .load_function(KERNEL_COMPRESS_TO_INT8_F16)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_COMPRESS_TO_INT8_F16))?;
        let kernel_compress_int2_f32 = module
            .load_function(KERNEL_COMPRESS_TO_INT2_F32)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_COMPRESS_TO_INT2_F32))?;
        let kernel_compress_int2_f16 = module
            .load_function(KERNEL_COMPRESS_TO_INT2_F16)
            .map_err(|_| EvicPressError::KernelMissing(KERNEL_COMPRESS_TO_INT2_F16))?;

        Ok(Self {
            module,
            kernel_importance_f32,
            kernel_importance_f16,
            kernel_zone_transition_f32,
            kernel_zone_transition_f16,
            kernel_compress_int8_f32,
            kernel_compress_int8_f16,
            kernel_compress_int2_f32,
            kernel_compress_int2_f16,
        })
    }

    /// Compute token importance scores (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `attention_weights` - Attention weights [batch * num_heads * seq_len]
    /// * `token_ages` - Token ages in steps [seq_len]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `seq_len` - Sequence length
    /// * `recency_weight` - Weight for recency factor
    /// * `attention_weight` - Weight for attention factor
    ///
    /// # Returns
    /// Importance scores: [seq_len]
    pub fn compute_importance_f32(
        &self,
        stream: &Arc<CudaStream>,
        attention_weights: &CudaSlice<f32>,
        token_ages: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        recency_weight: f32,
        attention_weight: f32,
    ) -> Result<CudaSlice<f32>, EvicPressError> {
        let mut importance: CudaSlice<f32> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let seq_i32 = seq_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_importance_f32);
            builder.arg(attention_weights);
            builder.arg(token_ages);
            builder.arg(&mut importance);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&seq_i32);
            builder.arg(&recency_weight);
            builder.arg(&attention_weight);
            builder.launch(cfg)?;
        }

        Ok(importance)
    }

    /// Compute token importance scores (f16).
    pub fn compute_importance_f16(
        &self,
        stream: &Arc<CudaStream>,
        attention_weights: &CudaSlice<f16>,
        token_ages: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        recency_weight: f32,
        attention_weight: f32,
    ) -> Result<CudaSlice<f32>, EvicPressError> {
        let mut importance: CudaSlice<f32> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let seq_i32 = seq_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_importance_f16);
            builder.arg(attention_weights);
            builder.arg(token_ages);
            builder.arg(&mut importance);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&seq_i32);
            builder.arg(&recency_weight);
            builder.arg(&attention_weight);
            builder.launch(cfg)?;
        }

        Ok(importance)
    }

    /// Determine zone transitions based on importance and cache pressure.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `current_zones` - Current zone assignments [seq_len]
    /// * `importance` - Importance scores [seq_len]
    /// * `seq_len` - Sequence length
    /// * `hot_threshold` - Threshold for staying in hot zone
    /// * `warm_threshold` - Threshold for staying in warm zone
    /// * `cache_pressure` - Current cache pressure (0.0-1.0)
    ///
    /// # Returns
    /// New zone assignments: [seq_len]
    pub fn zone_transition(
        &self,
        stream: &Arc<CudaStream>,
        current_zones: &CudaSlice<i32>,
        importance: &CudaSlice<f32>,
        seq_len: usize,
        hot_threshold: f32,
        warm_threshold: f32,
        cache_pressure: f32,
    ) -> Result<CudaSlice<i32>, EvicPressError> {
        let mut new_zones: CudaSlice<i32> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let seq_i32 = seq_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_zone_transition_f32);
            builder.arg(current_zones);
            builder.arg(importance);
            builder.arg(&mut new_zones);
            builder.arg(&seq_i32);
            builder.arg(&hot_threshold);
            builder.arg(&warm_threshold);
            builder.arg(&cache_pressure);
            builder.launch(cfg)?;
        }

        Ok(new_zones)
    }

    /// Compress KV cache from FP32 to INT8 (f32 input).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `kv_cache` - KV cache [seq_len * head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    ///
    /// # Returns
    /// Tuple of (compressed [seq_len * head_dim], scales [seq_len])
    pub fn compress_to_int8_f32(
        &self,
        stream: &Arc<CudaStream>,
        kv_cache: &CudaSlice<f32>,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(CudaSlice<i8>, CudaSlice<f32>), EvicPressError> {
        let total_elements = seq_len * head_dim;
        let mut compressed: CudaSlice<i8> = stream.alloc_zeros(total_elements)?;
        let mut scales: CudaSlice<f32> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (head_dim * std::mem::size_of::<f32>()) as u32,
        };

        let seq_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_compress_int8_f32);
            builder.arg(kv_cache);
            builder.arg(&mut compressed);
            builder.arg(&mut scales);
            builder.arg(&seq_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)?;
        }

        Ok((compressed, scales))
    }

    /// Compress KV cache from FP16 to INT8 (f16 input).
    pub fn compress_to_int8_f16(
        &self,
        stream: &Arc<CudaStream>,
        kv_cache: &CudaSlice<f16>,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(CudaSlice<i8>, CudaSlice<f16>), EvicPressError> {
        let total_elements = seq_len * head_dim;
        let mut compressed: CudaSlice<i8> = stream.alloc_zeros(total_elements)?;
        let mut scales: CudaSlice<f16> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (head_dim * std::mem::size_of::<f16>()) as u32,
        };

        let seq_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_compress_int8_f16);
            builder.arg(kv_cache);
            builder.arg(&mut compressed);
            builder.arg(&mut scales);
            builder.arg(&seq_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)?;
        }

        Ok((compressed, scales))
    }

    /// Compress KV cache from INT8 to INT2.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `int8_cache` - INT8 KV cache [seq_len * head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    ///
    /// # Returns
    /// Tuple of (compressed_packed [(seq_len * head_dim) / 4], scales [seq_len])
    pub fn compress_to_int2_f32(
        &self,
        stream: &Arc<CudaStream>,
        int8_cache: &CudaSlice<i8>,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<f32>), EvicPressError> {
        let total_elements = seq_len * head_dim;
        let packed_size = (total_elements + 3) / 4; // 4 INT2 values per byte
        let mut compressed: CudaSlice<u8> = stream.alloc_zeros(packed_size)?;
        let mut scales: CudaSlice<f32> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (head_dim * std::mem::size_of::<i8>()) as u32,
        };

        let seq_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_compress_int2_f32);
            builder.arg(int8_cache);
            builder.arg(&mut compressed);
            builder.arg(&mut scales);
            builder.arg(&seq_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)?;
        }

        Ok((compressed, scales))
    }

    /// Compress KV cache from INT8 to INT2 (f16 scales).
    pub fn compress_to_int2_f16(
        &self,
        stream: &Arc<CudaStream>,
        int8_cache: &CudaSlice<i8>,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<f16>), EvicPressError> {
        let total_elements = seq_len * head_dim;
        let packed_size = (total_elements + 3) / 4;
        let mut compressed: CudaSlice<u8> = stream.alloc_zeros(packed_size)?;
        let mut scales: CudaSlice<f16> = stream.alloc_zeros(seq_len)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (seq_len + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (head_dim * std::mem::size_of::<i8>()) as u32,
        };

        let seq_i32 = seq_len as i32;
        let head_dim_i32 = head_dim as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_compress_int2_f16);
            builder.arg(int8_cache);
            builder.arg(&mut compressed);
            builder.arg(&mut scales);
            builder.arg(&seq_i32);
            builder.arg(&head_dim_i32);
            builder.launch(cfg)?;
        }

        Ok((compressed, scales))
    }
}

/// Configuration for EvicPress CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct EvicPressCudaConfig {
    /// Hot zone size (number of tokens).
    pub hot_zone_size: usize,
    /// Warm zone size (number of tokens).
    pub warm_zone_size: usize,
    /// Cold zone size (number of tokens).
    pub cold_zone_size: usize,
    /// Importance threshold for hot zone.
    pub hot_threshold: f32,
    /// Importance threshold for warm zone.
    pub warm_threshold: f32,
    /// Weight for recency in importance calculation.
    pub recency_weight: f32,
    /// Weight for attention in importance calculation.
    pub attention_weight: f32,
}

impl Default for EvicPressCudaConfig {
    fn default() -> Self {
        Self {
            hot_zone_size: 1024,
            warm_zone_size: 2048,
            cold_zone_size: 4096,
            hot_threshold: 0.7,
            warm_threshold: 0.4,
            recency_weight: 0.3,
            attention_weight: 0.7,
        }
    }
}

impl EvicPressCudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), EvicPressError> {
        if self.hot_zone_size == 0 {
            return Err(EvicPressError::InvalidConfig(
                "hot_zone_size must be > 0".into(),
            ));
        }
        if self.hot_threshold <= 0.0 || self.hot_threshold > 1.0 {
            return Err(EvicPressError::InvalidConfig(
                "hot_threshold must be in (0, 1]".into(),
            ));
        }
        if self.warm_threshold <= 0.0 || self.warm_threshold >= self.hot_threshold {
            return Err(EvicPressError::InvalidConfig(
                "warm_threshold must be in (0, hot_threshold)".into(),
            ));
        }
        if self.recency_weight + self.attention_weight != 1.0 {
            return Err(EvicPressError::InvalidConfig(
                "recency_weight + attention_weight must equal 1.0".into(),
            ));
        }
        Ok(())
    }

    /// Get total cache capacity.
    pub fn total_capacity(&self) -> usize {
        self.hot_zone_size + self.warm_zone_size + self.cold_zone_size
    }
}
