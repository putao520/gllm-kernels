//! WGPU paged attention kernels using WGSL.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use wgpu::{BindGroupLayout, ComputePipeline, Device, Queue};

use super::utils::{buffer_layout_entry, uniform_layout_entry};

const KERNEL_F32: &str = "paged_attention_forward_f32";
const KERNEL_F16: &str = "paged_attention_forward_f16";
const SHADER_SOURCE: &str = include_str!("../kernels/paged_attention.wgsl");
pub(super) const WORKGROUP_SIZE: u32 = 128;
pub(super) const MAX_HEAD_DIM: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PagedAttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub seq_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Errors surfaced by the WGPU paged attention kernels.
#[derive(Debug)]
pub enum PagedAttentionError {
    /// WGPU driver error or initialization failure.
    Wgpu(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Unsupported device capability.
    Unsupported(String),
}

impl fmt::Display for PagedAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu(msg) => write!(f, "WGPU error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
        }
    }
}

impl std::error::Error for PagedAttentionError {}

/// Buffer pool configuration for PagedAttention (P0-MEM-3).
const PA_POOL_MAX_PER_BUCKET: usize = 4;
const PA_POOL_MAX_BYTES: usize = 256 * 1024 * 1024; // 256 MB

/// Size bucket for buffer pooling (rounded to power of 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SizeBucket(u64);

impl SizeBucket {
    fn from_size(size: u64) -> Self {
        let min_size = 256u64;
        let bucket = size.max(min_size).next_power_of_two();
        SizeBucket(bucket)
    }

    fn size(&self) -> u64 {
        self.0
    }
}

/// Buffer pool for PagedAttention dispatch (P0-MEM-3).
pub(super) struct DispatchBufferPool {
    storage_buckets: HashMap<SizeBucket, Vec<wgpu::Buffer>>,
    staging_buckets: HashMap<SizeBucket, Vec<wgpu::Buffer>>,
    total_bytes: usize,
}

impl DispatchBufferPool {
    fn new() -> Self {
        Self {
            storage_buckets: HashMap::new(),
            staging_buckets: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Get or create a storage buffer.
    pub fn get_storage(
        &mut self,
        device: &Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> wgpu::Buffer {
        let bucket = SizeBucket::from_size(size);

        if let Some(buffers) = self.storage_buckets.get_mut(&bucket) {
            if let Some(buffer) = buffers.pop() {
                self.total_bytes = self.total_bytes.saturating_sub(bucket.size() as usize);
                log::trace!("DispatchBufferPool: reused storage buffer size={}", bucket.size());
                return buffer;
            }
        }

        log::trace!("DispatchBufferPool: created storage buffer size={} label={}", bucket.size(), label);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bucket.size(),
            usage,
            mapped_at_creation: false,
        })
    }

    /// Get or create a staging buffer for readback.
    pub fn get_staging(
        &mut self,
        device: &Device,
        size: u64,
        label: &'static str,
    ) -> wgpu::Buffer {
        let bucket = SizeBucket::from_size(size);

        if let Some(buffers) = self.staging_buckets.get_mut(&bucket) {
            if let Some(buffer) = buffers.pop() {
                self.total_bytes = self.total_bytes.saturating_sub(bucket.size() as usize);
                log::trace!("DispatchBufferPool: reused staging buffer size={}", bucket.size());
                return buffer;
            }
        }

        log::trace!("DispatchBufferPool: created staging buffer size={} label={}", bucket.size(), label);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bucket.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Release a storage buffer back to pool.
    pub fn release_storage(&mut self, buffer: wgpu::Buffer, original_size: u64) {
        let bucket = SizeBucket::from_size(original_size);
        let bucket_size = bucket.size() as usize;

        if self.total_bytes + bucket_size > PA_POOL_MAX_BYTES {
            return;
        }

        let buffers = self.storage_buckets.entry(bucket).or_insert_with(Vec::new);
        if buffers.len() >= PA_POOL_MAX_PER_BUCKET {
            return;
        }

        buffers.push(buffer);
        self.total_bytes += bucket_size;
    }

    /// Release a staging buffer back to pool.
    pub fn release_staging(&mut self, buffer: wgpu::Buffer, original_size: u64) {
        let bucket = SizeBucket::from_size(original_size);
        let bucket_size = bucket.size() as usize;

        if self.total_bytes + bucket_size > PA_POOL_MAX_BYTES / 4 {
            return;
        }

        let buffers = self.staging_buckets.entry(bucket).or_insert_with(Vec::new);
        if buffers.len() >= PA_POOL_MAX_PER_BUCKET {
            return;
        }

        buffers.push(buffer);
        self.total_bytes += bucket_size;
    }
}

/// Paged attention WGPU kernel wrapper.
pub struct PagedAttentionKernel {
    pub(super) device: Arc<Device>,
    pub(super) queue: Arc<Queue>,
    pub(super) bind_group_layout: BindGroupLayout,
    pub(super) pipeline_f32: ComputePipeline,
    pub(super) pipeline_f16: Option<ComputePipeline>,
    /// Buffer pool for dispatch operations (P0-MEM-3).
    pub(super) buffer_pool: Mutex<DispatchBufferPool>,
}

impl PagedAttentionKernel {
    /// Create a paged attention kernel wrapper for an existing WGPU device.
    pub fn new(device: &Device, queue: &Queue) -> Result<Self, PagedAttentionError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paged_attention.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paged_attention_bind_group_layout"),
            entries: &[
                buffer_layout_entry(0, true),
                buffer_layout_entry(1, true),
                buffer_layout_entry(2, true),
                buffer_layout_entry(3, true),
                buffer_layout_entry(4, true),
                buffer_layout_entry(5, false),
                uniform_layout_entry(6),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("paged_attention_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("paged_attention_f32"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(KERNEL_F32),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_f16 = if device.features().contains(wgpu::Features::SHADER_F16) {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("paged_attention_f16"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_F16),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        Ok(Self {
            device: Arc::new(device.clone()),
            queue: Arc::new(queue.clone()),
            bind_group_layout,
            pipeline_f32,
            pipeline_f16,
            buffer_pool: Mutex::new(DispatchBufferPool::new()),
        })
    }

    /// Create a paged attention kernel wrapper with a newly initialized device.
    pub fn create_default(require_f16: bool) -> Result<Self, PagedAttentionError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .map_err(|e| PagedAttentionError::Wgpu(format!("no compatible adapter found: {e}")))?;

        let mut features = wgpu::Features::empty();
        if require_f16 {
            if adapter.features().contains(wgpu::Features::SHADER_F16) {
                features |= wgpu::Features::SHADER_F16;
            } else {
                return Err(PagedAttentionError::Unsupported(
                    "adapter does not support shader f16".into(),
                ));
            }
        }

        let limits = wgpu::Limits::default();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gllm-wgpu-paged-attn"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|err| PagedAttentionError::Wgpu(format!("request_device failed: {err}")))?;

        Self::new(&device, &queue)
    }

}
