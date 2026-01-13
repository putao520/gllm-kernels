//! WGPU FlashAttention-style kernels using WGSL.

use std::borrow::Cow;
use std::fmt;
use std::mem;
use std::sync::Arc;

use half::f16;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device,
    Queue,
};

const KERNEL_F32: &str = "flash_attention_forward_f32";
const KERNEL_F16: &str = "flash_attention_forward_f16";
const SHADER_SOURCE: &str = include_str!("kernels/flash_attention.wgsl");
const WORKGROUP_SIZE: u32 = 128;
const MAX_HEAD_DIM: usize = 256;
const DEFAULT_BLOCK: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct AttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    block_size: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Errors surfaced by the WGPU FlashAttention kernels.
#[derive(Debug)]
pub enum FlashAttentionError {
    /// WGPU driver error or initialization failure.
    Wgpu(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Unsupported device capability.
    Unsupported(String),
}

impl fmt::Display for FlashAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu(msg) => write!(f, "WGPU error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
        }
    }
}

impl std::error::Error for FlashAttentionError {}

/// FlashAttention WGPU kernel wrapper.
pub struct FlashAttentionKernel {
    device: Arc<Device>,
    queue: Arc<Queue>,
    bind_group_layout: BindGroupLayout,
    pipeline_f32: ComputePipeline,
    pipeline_f16: Option<ComputePipeline>,
}

impl FlashAttentionKernel {
    /// Create a FlashAttention kernel wrapper for an existing WGPU device.
    pub fn new(device: &Device, queue: &Queue) -> Result<Self, FlashAttentionError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flash_attention.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flash_attention_bind_group_layout"),
            entries: &[
                buffer_layout_entry(0, true),
                buffer_layout_entry(1, true),
                buffer_layout_entry(2, true),
                buffer_layout_entry(3, false),
                uniform_layout_entry(4),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flash_attention_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flash_attention_f32"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(KERNEL_F32),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_f16 = if device.features().contains(wgpu::Features::SHADER_F16) {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flash_attention_f16"),
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

    /// Create a FlashAttention kernel wrapper with a newly initialized device.
    pub fn create_default(require_f16: bool) -> Result<Self, FlashAttentionError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok_or_else(|| FlashAttentionError::Wgpu("no compatible adapter found".into()))?;

        let mut features = wgpu::Features::empty();
        if require_f16 {
            if adapter.features().contains(wgpu::Features::SHADER_F16) {
                features |= wgpu::Features::SHADER_F16;
            } else {
                return Err(FlashAttentionError::Unsupported(
                    "adapter does not support shader f16".into(),
                ));
            }
        }

        let limits = wgpu::Limits::default();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gllm-wgpu-attn"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
            None,
        ))
        .map_err(|err| FlashAttentionError::Wgpu(format!("request_device failed: {err}")))?;

        Self::new(&device, &queue)
    }

    /// FlashAttention forward for f32 inputs.
    pub fn forward_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        block_size: Option<u32>,
        scale: f32,
    ) -> Result<Vec<f32>, FlashAttentionError> {
        let params = build_params(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            block_size,
            scale,
        )?;
        let expected = expected_elements(batch_size, num_heads, seq_len, head_dim)?;
        if q.len() != expected || k.len() != expected || v.len() != expected {
            return Err(FlashAttentionError::InvalidConfig(
                "Q/K/V length mismatch".into(),
            ));
        }

        let output_bytes = expected
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| FlashAttentionError::InvalidConfig("output size overflow".into()))?;

        let output = self.dispatch_bytes(
            slice_as_bytes(q),
            slice_as_bytes(k),
            slice_as_bytes(v),
            output_bytes as u64,
            params,
            &self.pipeline_f32,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// FlashAttention forward for f16 inputs.
    pub fn forward_f16(
        &self,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        block_size: Option<u32>,
        scale: f32,
    ) -> Result<Vec<f16>, FlashAttentionError> {
        let pipeline = self.pipeline_f16.as_ref().ok_or_else(|| {
            FlashAttentionError::Unsupported("device does not support f16 kernels".into())
        })?;
        let params = build_params(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            block_size,
            scale,
        )?;
        let expected = expected_elements(batch_size, num_heads, seq_len, head_dim)?;
        if q.len() != expected || k.len() != expected || v.len() != expected {
            return Err(FlashAttentionError::InvalidConfig(
                "Q/K/V length mismatch".into(),
            ));
        }

        let output_bytes = expected
            .checked_mul(mem::size_of::<f16>())
            .ok_or_else(|| FlashAttentionError::InvalidConfig("output size overflow".into()))?;

        let output = self.dispatch_bytes(
            slice_as_bytes(q),
            slice_as_bytes(k),
            slice_as_bytes(v),
            output_bytes as u64,
            params,
            pipeline,
        )?;

        Ok(bytes_to_vec(&output))
    }

    fn dispatch_bytes(
        &self,
        q_bytes: &[u8],
        k_bytes: &[u8],
        v_bytes: &[u8],
        output_bytes: u64,
        params: AttentionParams,
        pipeline: &ComputePipeline,
    ) -> Result<Vec<u8>, FlashAttentionError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let q_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flash_attention_q"),
            contents: q_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let k_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flash_attention_k"),
            contents: k_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let v_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flash_attention_v"),
            contents: v_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flash_attention_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flash_attention_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flash_attention_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                buffer_binding(0, &q_buffer),
                buffer_binding(1, &k_buffer),
                buffer_binding(2, &v_buffer),
                buffer_binding(3, &output_buffer),
                buffer_binding(4, &params_buffer),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flash_attention_encoder"),
            });

        self.encode_pass(
            &mut encoder,
            pipeline,
            &bind_group,
            params.seq_len,
            params.batch_size,
            params.num_heads,
        );

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flash_attention_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        let mut output = data;
        output.truncate(output_bytes as usize);
        Ok(output)
    }

    fn encode_pass(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        seq_len: u32,
        batch_size: u32,
        num_heads: u32,
    ) {
        let workgroups_x = (seq_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = batch_size.saturating_mul(num_heads);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flash_attention_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

fn buffer_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_binding(binding: u32, buffer: &Buffer) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn build_params(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: Option<u32>,
    scale: f32,
) -> Result<AttentionParams, FlashAttentionError> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err(FlashAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(FlashAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        )));
    }

    let batch_size_u32 = u32::try_from(batch_size).map_err(|_| {
        FlashAttentionError::InvalidConfig("batch_size exceeds u32".into())
    })?;
    let num_heads_u32 = u32::try_from(num_heads)
        .map_err(|_| FlashAttentionError::InvalidConfig("num_heads exceeds u32".into()))?;
    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|_| FlashAttentionError::InvalidConfig("seq_len exceeds u32".into()))?;
    let head_dim_u32 = u32::try_from(head_dim)
        .map_err(|_| FlashAttentionError::InvalidConfig("head_dim exceeds u32".into()))?;

    let block = block_size.unwrap_or(DEFAULT_BLOCK).max(1);

    Ok(AttentionParams {
        batch_size: batch_size_u32,
        num_heads: num_heads_u32,
        seq_len: seq_len_u32,
        head_dim: head_dim_u32,
        block_size: block,
        scale,
        _pad0: 0,
        _pad1: 0,
    })
}

fn expected_elements(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<usize, FlashAttentionError> {
    let num_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| FlashAttentionError::InvalidConfig("num_queries overflow".into()))?;

    num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| FlashAttentionError::InvalidConfig("output_len overflow".into()))
}

fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((value as *const T) as *const u8, mem::size_of::<T>())
    }
}

fn slice_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * mem::size_of::<T>())
    }
}

fn bytes_to_vec<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let len = bytes.len() / mem::size_of::<T>();
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

fn align_up(value: u64, align: u64) -> u64 {
    if align == 0 {
        return value;
    }
    (value + align - 1) / align * align
}

fn max_align() -> u64 {
    let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
    let map_align = wgpu::MAP_ALIGNMENT;
    if copy_align > map_align {
        copy_align
    } else {
        map_align
    }
}

fn read_buffer_sync(device: &Device, buffer: &Buffer, size: u64) -> Result<Vec<u8>, FlashAttentionError> {
    let slice = buffer.slice(0..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            return Err(FlashAttentionError::Wgpu(format!("map_async failed: {err}")))
        }
        Err(_) => {
            return Err(FlashAttentionError::Wgpu("map_async channel closed".into()));
        }
    }

    let data = slice.get_mapped_range();
    let bytes = data.to_vec();
    drop(data);
    buffer.unmap();
    Ok(bytes)
}
