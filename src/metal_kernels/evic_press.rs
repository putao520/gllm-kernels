//! Metal EvicPress joint compression-eviction kernels.
//!
//! This module provides Metal GPU-accelerated kernels for EvicPress KV cache management:
//! - Token importance scoring
//! - Three-zone cache management (Hot FP16 → Warm INT8 → Cold INT2)
//! - Adaptive compression and eviction
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

const KERNEL_COMPUTE_IMPORTANCE_F32: &str = "evic_press_compute_importance_f32";
const KERNEL_COMPUTE_IMPORTANCE_F16: &str = "evic_press_compute_importance_f16";
const KERNEL_ZONE_TRANSITION: &str = "evic_press_zone_transition";
const KERNEL_COMPRESS_INT8_F32: &str = "evic_press_compress_int8_f32";
const KERNEL_COMPRESS_INT8_F16: &str = "evic_press_compress_int8_f16";
const KERNEL_COMPRESS_INT2_F32: &str = "evic_press_compress_int2_f32";
const KERNEL_COMPRESS_INT2_F16: &str = "evic_press_compress_int2_f16";

/// Metallib collection for EvicPress kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static EVIC_PRESS_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "evic_press",
    metallib_data: include_bytes!("kernels/evic_press.metallib"),
};

/// Cache zone for KV cache entries.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheZone {
    /// Hot zone: FP16 full precision, frequently accessed.
    Hot = 0,
    /// Warm zone: INT8 quantized, moderately accessed.
    Warm = 1,
    /// Cold zone: INT2 compressed, rarely accessed.
    Cold = 2,
    /// Evicted: removed from cache.
    Evicted = 3,
}

/// Parameters for importance computation kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ComputeImportanceParams {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    recency_weight: f32,
    frequency_weight: f32,
    attention_weight: f32,
    _pad: u32,
}

/// Parameters for zone transition kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ZoneTransitionParams {
    num_entries: u32,
    hot_threshold: f32,
    warm_threshold: f32,
    cold_threshold: f32,
}

/// Parameters for compression kernels.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct CompressParams {
    num_elements: u32,
    head_dim: u32,
    _pad: [u32; 2],
}

/// Errors surfaced by the Metal EvicPress kernels.
#[derive(Debug)]
pub enum EvicPressError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for EvicPressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for EvicPressError {}

impl From<MetallibLoadError> for EvicPressError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Configuration for EvicPress operations.
#[derive(Clone, Debug)]
pub struct EvicPressConfig {
    /// Weight for recency in importance scoring.
    pub recency_weight: f32,
    /// Weight for access frequency in importance scoring.
    pub frequency_weight: f32,
    /// Weight for attention scores in importance scoring.
    pub attention_weight: f32,
    /// Importance threshold for hot zone (above this = hot).
    pub hot_threshold: f32,
    /// Importance threshold for warm zone (above this = warm).
    pub warm_threshold: f32,
    /// Importance threshold for cold zone (above this = cold, below = evicted).
    pub cold_threshold: f32,
}

impl Default for EvicPressConfig {
    fn default() -> Self {
        Self {
            recency_weight: 0.4,
            frequency_weight: 0.3,
            attention_weight: 0.3,
            hot_threshold: 0.7,
            warm_threshold: 0.4,
            cold_threshold: 0.1,
        }
    }
}

impl EvicPressConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), EvicPressError> {
        let sum = self.recency_weight + self.frequency_weight + self.attention_weight;
        if (sum - 1.0).abs() > 0.01 {
            return Err(EvicPressError::InvalidConfig("importance weights must sum to 1.0".into()));
        }
        if self.hot_threshold <= self.warm_threshold || self.warm_threshold <= self.cold_threshold {
            return Err(EvicPressError::InvalidConfig("thresholds must be hot > warm > cold".into()));
        }
        Ok(())
    }
}

/// EvicPress Metal kernel wrapper.
pub struct EvicPressKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_importance_f32: ComputePipelineState,
    pipeline_importance_f16: ComputePipelineState,
    pipeline_zone_transition: ComputePipelineState,
    pipeline_compress_int8_f32: ComputePipelineState,
    pipeline_compress_int8_f16: ComputePipelineState,
    pipeline_compress_int2_f32: ComputePipelineState,
    pipeline_compress_int2_f16: ComputePipelineState,
}

impl EvicPressKernel {
    /// Load EvicPress kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, EvicPressError> {
        let library = load_library(device)?;

        let pipeline_importance_f32 = build_pipeline(device, &library, KERNEL_COMPUTE_IMPORTANCE_F32)?;
        let pipeline_importance_f16 = build_pipeline(device, &library, KERNEL_COMPUTE_IMPORTANCE_F16)?;
        let pipeline_zone_transition = build_pipeline(device, &library, KERNEL_ZONE_TRANSITION)?;
        let pipeline_compress_int8_f32 = build_pipeline(device, &library, KERNEL_COMPRESS_INT8_F32)?;
        let pipeline_compress_int8_f16 = build_pipeline(device, &library, KERNEL_COMPRESS_INT8_F16)?;
        let pipeline_compress_int2_f32 = build_pipeline(device, &library, KERNEL_COMPRESS_INT2_F32)?;
        let pipeline_compress_int2_f16 = build_pipeline(device, &library, KERNEL_COMPRESS_INT2_F16)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_importance_f32,
            pipeline_importance_f16,
            pipeline_zone_transition,
            pipeline_compress_int8_f32,
            pipeline_compress_int8_f16,
            pipeline_compress_int2_f32,
            pipeline_compress_int2_f16,
        })
    }

    /// Compute importance scores for KV cache entries (f32).
    pub fn compute_importance_f32(
        &self,
        attention_scores: &Buffer,
        access_counts: &Buffer,
        last_access_times: &Buffer,
        current_step: u32,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        config: &EvicPressConfig,
    ) -> Result<Buffer, EvicPressError> {
        config.validate()?;
        self.compute_importance_impl(
            attention_scores, access_counts, last_access_times, current_step,
            batch_size, seq_len, num_heads, head_dim, config,
            &self.pipeline_importance_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Compute importance scores for KV cache entries (f16).
    pub fn compute_importance_f16(
        &self,
        attention_scores: &Buffer,
        access_counts: &Buffer,
        last_access_times: &Buffer,
        current_step: u32,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        config: &EvicPressConfig,
    ) -> Result<Buffer, EvicPressError> {
        config.validate()?;
        self.compute_importance_impl(
            attention_scores, access_counts, last_access_times, current_step,
            batch_size, seq_len, num_heads, head_dim, config,
            &self.pipeline_importance_f16,
            mem::size_of::<u16>(),
        )
    }

    fn compute_importance_impl(
        &self,
        attention_scores: &Buffer,
        access_counts: &Buffer,
        last_access_times: &Buffer,
        _current_step: u32,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        config: &EvicPressConfig,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, EvicPressError> {
        let num_entries = batch_size * seq_len * num_heads;
        let output_bytes = (num_entries * element_size) as u64;

        let params = ComputeImportanceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            recency_weight: config.recency_weight,
            frequency_weight: config.frequency_weight,
            attention_weight: config.attention_weight,
            _pad: 0,
        };

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(attention_scores), 0);
        encoder.set_buffer(1, Some(access_counts), 0);
        encoder.set_buffer(2, Some(last_access_times), 0);
        encoder.set_buffer(3, Some(&output), 0);

        let params_size = mem::size_of::<ComputeImportanceParams>() as u64;
        encoder.set_bytes(4, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(num_entries as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Determine zone transitions based on importance scores.
    pub fn zone_transition(
        &self,
        importance_scores: &Buffer,
        current_zones: &Buffer,
        num_entries: usize,
        config: &EvicPressConfig,
    ) -> Result<Buffer, EvicPressError> {
        config.validate()?;

        let params = ZoneTransitionParams {
            num_entries: num_entries as u32,
            hot_threshold: config.hot_threshold,
            warm_threshold: config.warm_threshold,
            cold_threshold: config.cold_threshold,
        };

        // Output: new zone assignments as u8
        let output_bytes = num_entries as u64;
        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_zone_transition);
        encoder.set_buffer(0, Some(importance_scores), 0);
        encoder.set_buffer(1, Some(current_zones), 0);
        encoder.set_buffer(2, Some(&output), 0);

        let params_size = mem::size_of::<ZoneTransitionParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(num_entries as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(&self.pipeline_zone_transition);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Compress KV cache entries to INT8 (f32 input).
    pub fn compress_to_int8_f32(
        &self,
        kv_data: &Buffer,
        num_elements: usize,
        head_dim: usize,
    ) -> Result<(Buffer, Buffer), EvicPressError> {
        self.compress_to_int8_impl(
            kv_data, num_elements, head_dim,
            &self.pipeline_compress_int8_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Compress KV cache entries to INT8 (f16 input).
    pub fn compress_to_int8_f16(
        &self,
        kv_data: &Buffer,
        num_elements: usize,
        head_dim: usize,
    ) -> Result<(Buffer, Buffer), EvicPressError> {
        self.compress_to_int8_impl(
            kv_data, num_elements, head_dim,
            &self.pipeline_compress_int8_f16,
            mem::size_of::<u16>(),
        )
    }

    fn compress_to_int8_impl(
        &self,
        kv_data: &Buffer,
        num_elements: usize,
        head_dim: usize,
        pipeline: &ComputePipelineState,
        scale_element_size: usize,
    ) -> Result<(Buffer, Buffer), EvicPressError> {
        let params = CompressParams {
            num_elements: num_elements as u32,
            head_dim: head_dim as u32,
            _pad: [0; 2],
        };

        let num_vectors = num_elements / head_dim;
        let compressed_bytes = num_elements as u64; // INT8
        let scales_bytes = (num_vectors * scale_element_size) as u64;

        let compressed = self.device.new_buffer(compressed_bytes, MTLResourceOptions::StorageModeShared);
        let scales = self.device.new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(kv_data), 0);
        encoder.set_buffer(1, Some(&compressed), 0);
        encoder.set_buffer(2, Some(&scales), 0);

        let params_size = mem::size_of::<CompressParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(num_vectors as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((compressed, scales))
    }

    /// Compress KV cache entries to INT2 (f32 input).
    pub fn compress_to_int2_f32(
        &self,
        kv_data: &Buffer,
        num_elements: usize,
        head_dim: usize,
    ) -> Result<(Buffer, Buffer), EvicPressError> {
        self.compress_to_int2_impl(
            kv_data, num_elements, head_dim,
            &self.pipeline_compress_int2_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Compress KV cache entries to INT2 (f16 input).
    pub fn compress_to_int2_f16(
        &self,
        kv_data: &Buffer,
        num_elements: usize,
        head_dim: usize,
    ) -> Result<(Buffer, Buffer), EvicPressError> {
        self.compress_to_int2_impl(
            kv_data, num_elements, head_dim,
            &self.pipeline_compress_int2_f16,
            mem::size_of::<u16>(),
        )
    }

    fn compress_to_int2_impl(
        &self,
        kv_data: &Buffer,
        num_elements: usize,
        head_dim: usize,
        pipeline: &ComputePipelineState,
        scale_element_size: usize,
    ) -> Result<(Buffer, Buffer), EvicPressError> {
        let params = CompressParams {
            num_elements: num_elements as u32,
            head_dim: head_dim as u32,
            _pad: [0; 2],
        };

        let num_vectors = num_elements / head_dim;
        let compressed_bytes = (num_elements / 4) as u64; // INT2 packed
        let scales_bytes = (num_vectors * scale_element_size) as u64;

        let compressed = self.device.new_buffer(compressed_bytes, MTLResourceOptions::StorageModeShared);
        let scales = self.device.new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(kv_data), 0);
        encoder.set_buffer(1, Some(&compressed), 0);
        encoder.set_buffer(2, Some(&scales), 0);

        let params_size = mem::size_of::<CompressParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(num_vectors as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((compressed, scales))
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, EvicPressError> {
    EVIC_PRESS_METALLIB.load(device).map_err(EvicPressError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, EvicPressError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| EvicPressError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(EvicPressError::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
