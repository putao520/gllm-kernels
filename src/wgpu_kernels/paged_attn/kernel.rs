//! WGPU paged attention kernels using WGSL.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use wgpu::{BindGroupLayout, ComputePipeline, Device, Queue};

use super::utils::{buffer_layout_entry, uniform_layout_entry};

const KERNEL_F32: &str = "paged_attention_forward_f32";
const KERNEL_F16: &str = "paged_attention_forward_f16";
const SHADER_SOURCE: &str = include_str!("../kernels/paged_attention.wgsl");
pub(super) const WORKGROUP_SIZE: u32 = 128;
pub(super) const MAX_HEAD_DIM: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(super) struct PagedAttentionParams {
    pub(super) batch_size: u32,
    pub(super) num_heads: u32,
    pub(super) head_dim: u32,
    pub(super) block_size: u32,
    pub(super) seq_len: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
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

/// Paged attention WGPU kernel wrapper.
pub struct PagedAttentionKernel {
    pub(super) device: Arc<Device>,
    pub(super) queue: Arc<Queue>,
    pub(super) bind_group_layout: BindGroupLayout,
    pub(super) pipeline_f32: ComputePipeline,
    pub(super) pipeline_f16: Option<ComputePipeline>,
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
