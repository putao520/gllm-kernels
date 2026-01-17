//! WGPU EAGLE-3 adaptive draft length speculative decoding kernel.
//!
//! Based on EAGLE-3 (NeurIPS'25): Multi-layer feature fusion, token-level
//! confidence prediction, and adaptive draft length scheduling for 2-6x
//! inference acceleration.
//!
//! # Key Operations
//! - Multi-layer hidden state fusion (concatenation along hidden dim)
//! - Token-level confidence prediction via linear projection + sigmoid
//! - Draft generation with confidence-based early termination

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

const SHADER_SOURCE: &str = include_str!("kernels/eagle3.wgsl");
const WORKGROUP_SIZE: u32 = 256;

// Kernel entry points
const KERNEL_FUSE_F32: &str = "eagle3_fuse_layers_f32";
const KERNEL_FUSE_F16: &str = "eagle3_fuse_layers_f16";
const KERNEL_CONFIDENCE_F32: &str = "eagle3_predict_confidence_f32";
const KERNEL_CONFIDENCE_F16: &str = "eagle3_predict_confidence_f16";
const KERNEL_TERMINATION_F32: &str = "eagle3_check_termination_f32";

/// WGSL shader struct: FusionParams
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FusionParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    num_layers: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// WGSL shader struct: ConfidenceParams
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ConfidenceParams {
    batch_size: u32,
    seq_len: u32,
    fused_dim: u32,
    bias: f32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Errors surfaced by the WGPU EAGLE-3 kernels.
#[derive(Debug)]
pub enum Eagle3Error {
    /// WGPU driver error or initialization failure.
    Wgpu(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Unsupported device capability.
    Unsupported(String),
    /// Dimension mismatch.
    DimensionMismatch(String),
}

impl fmt::Display for Eagle3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu(msg) => write!(f, "WGPU error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
        }
    }
}

impl std::error::Error for Eagle3Error {}

/// EAGLE-3 WGPU kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Multi-layer hidden state fusion
/// - Token-level confidence prediction
/// - Early termination check
pub struct Eagle3Kernel {
    device: Arc<Device>,
    queue: Arc<Queue>,
    // Fusion pipelines
    fusion_layout: BindGroupLayout,
    pipeline_fuse_f32: ComputePipeline,
    pipeline_fuse_f16: Option<ComputePipeline>,
    // Confidence pipelines
    confidence_layout: BindGroupLayout,
    pipeline_confidence_f32: ComputePipeline,
    pipeline_confidence_f16: Option<ComputePipeline>,
    // Termination pipeline
    termination_layout: BindGroupLayout,
    pipeline_termination_f32: ComputePipeline,
}

impl Eagle3Kernel {
    /// Create an EAGLE-3 kernel wrapper for an existing WGPU device.
    pub fn new(device: &Device, queue: &Queue) -> Result<Self, Eagle3Error> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eagle3.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // Fusion bind group layout: input, output, params
        let fusion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("eagle3_fusion_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // input
                buffer_layout_entry(1, false), // output
                uniform_layout_entry(2),       // params
            ],
        });

        // Confidence bind group layout: fused, weights, output, params
        let confidence_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("eagle3_confidence_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // fused
                buffer_layout_entry(1, true),  // weights
                buffer_layout_entry(2, false), // output
                uniform_layout_entry(3),       // params
            ],
        });

        // Termination bind group layout: confidence, output, params
        let termination_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("eagle3_termination_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // confidence
                buffer_layout_entry(1, false), // output (u32)
                uniform_layout_entry(2),       // params
            ],
        });

        let fusion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("eagle3_fusion_pipeline_layout"),
                bind_group_layouts: &[&fusion_layout],
                push_constant_ranges: &[],
            });

        let confidence_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("eagle3_confidence_pipeline_layout"),
                bind_group_layouts: &[&confidence_layout],
                push_constant_ranges: &[],
            });

        let termination_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("eagle3_termination_pipeline_layout"),
                bind_group_layouts: &[&termination_layout],
                push_constant_ranges: &[],
            });

        // Create F32 pipelines
        let pipeline_fuse_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("eagle3_fuse_f32"),
            layout: Some(&fusion_pipeline_layout),
            module: &shader,
            entry_point: Some(KERNEL_FUSE_F32),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_confidence_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eagle3_confidence_f32"),
                layout: Some(&confidence_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_CONFIDENCE_F32),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline_termination_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eagle3_termination_f32"),
                layout: Some(&termination_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_TERMINATION_F32),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create F16 pipelines if supported
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        let pipeline_fuse_f16 = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eagle3_fuse_f16"),
                layout: Some(&fusion_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_FUSE_F16),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        let pipeline_confidence_f16 = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eagle3_confidence_f16"),
                layout: Some(&confidence_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_CONFIDENCE_F16),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        Ok(Self {
            device: Arc::new(device.clone()),
            queue: Arc::new(queue.clone()),
            fusion_layout,
            pipeline_fuse_f32,
            pipeline_fuse_f16,
            confidence_layout,
            pipeline_confidence_f32,
            pipeline_confidence_f16,
            termination_layout,
            pipeline_termination_f32,
        })
    }

    /// Create an EAGLE-3 kernel wrapper with a newly initialized device.
    pub fn create_default(require_f16: bool) -> Result<Self, Eagle3Error> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| Eagle3Error::Wgpu(format!("no compatible adapter found: {e}")))?;

        let mut features = wgpu::Features::empty();
        if require_f16 {
            if adapter.features().contains(wgpu::Features::SHADER_F16) {
                features |= wgpu::Features::SHADER_F16;
            } else {
                return Err(Eagle3Error::Unsupported(
                    "adapter does not support shader f16".into(),
                ));
            }
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gllm-wgpu-eagle3"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|err| Eagle3Error::Wgpu(format!("request_device failed: {err}")))?;

        Self::new(&device, &queue)
    }

    /// Fuse multi-layer hidden states (f32).
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [num_layers, batch_size, seq_len, hidden_dim]
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension per layer
    /// * `num_layers` - Number of layers to fuse
    ///
    /// # Returns
    /// Fused tensor [batch_size, seq_len, hidden_dim * num_layers]
    pub fn fuse_layers_f32(
        &self,
        hidden_states: &[f32],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<Vec<f32>, Eagle3Error> {
        let expected_input = num_layers * batch_size * seq_len * hidden_dim;
        if hidden_states.len() != expected_input {
            return Err(Eagle3Error::DimensionMismatch(format!(
                "expected {} elements, got {}",
                expected_input,
                hidden_states.len()
            )));
        }

        let params = FusionParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            num_layers: num_layers as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let output_len = batch_size * seq_len * hidden_dim * num_layers;
        let output_bytes = output_len * mem::size_of::<f32>();

        let output = self.dispatch_fusion(
            slice_as_bytes(hidden_states),
            output_bytes as u64,
            params,
            &self.pipeline_fuse_f32,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Fuse multi-layer hidden states (f16).
    pub fn fuse_layers_f16(
        &self,
        hidden_states: &[f16],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<Vec<f16>, Eagle3Error> {
        let pipeline = self.pipeline_fuse_f16.as_ref().ok_or_else(|| {
            Eagle3Error::Unsupported("device does not support f16 kernels".into())
        })?;

        let expected_input = num_layers * batch_size * seq_len * hidden_dim;
        if hidden_states.len() != expected_input {
            return Err(Eagle3Error::DimensionMismatch(format!(
                "expected {} elements, got {}",
                expected_input,
                hidden_states.len()
            )));
        }

        let params = FusionParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            num_layers: num_layers as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let output_len = batch_size * seq_len * hidden_dim * num_layers;
        let output_bytes = output_len * mem::size_of::<f16>();

        let output = self.dispatch_fusion(
            slice_as_bytes(hidden_states),
            output_bytes as u64,
            params,
            pipeline,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Predict token-level confidence scores (f32).
    ///
    /// # Arguments
    /// * `fused` - Fused hidden states [batch_size, seq_len, fused_dim]
    /// * `weights` - Projection weights [fused_dim]
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `fused_dim` - Fused hidden dimension
    /// * `bias` - Bias term for projection
    ///
    /// # Returns
    /// Confidence scores [batch_size, seq_len] in range [0, 1]
    pub fn predict_confidence_f32(
        &self,
        fused: &[f32],
        weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        fused_dim: usize,
        bias: f32,
    ) -> Result<Vec<f32>, Eagle3Error> {
        let expected_fused = batch_size * seq_len * fused_dim;
        if fused.len() != expected_fused {
            return Err(Eagle3Error::DimensionMismatch(format!(
                "fused: expected {} elements, got {}",
                expected_fused,
                fused.len()
            )));
        }
        if weights.len() != fused_dim {
            return Err(Eagle3Error::DimensionMismatch(format!(
                "weights: expected {} elements, got {}",
                fused_dim,
                weights.len()
            )));
        }

        let params = ConfidenceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            fused_dim: fused_dim as u32,
            bias,
            threshold: 0.0, // not used in confidence prediction
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = batch_size * seq_len;
        let output_bytes = output_len * mem::size_of::<f32>();

        let output = self.dispatch_confidence(
            slice_as_bytes(fused),
            slice_as_bytes(weights),
            output_bytes as u64,
            params,
            &self.pipeline_confidence_f32,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Predict token-level confidence scores (f16).
    pub fn predict_confidence_f16(
        &self,
        fused: &[f16],
        weights: &[f16],
        batch_size: usize,
        seq_len: usize,
        fused_dim: usize,
        bias: f32,
    ) -> Result<Vec<f16>, Eagle3Error> {
        let pipeline = self.pipeline_confidence_f16.as_ref().ok_or_else(|| {
            Eagle3Error::Unsupported("device does not support f16 kernels".into())
        })?;

        let expected_fused = batch_size * seq_len * fused_dim;
        if fused.len() != expected_fused {
            return Err(Eagle3Error::DimensionMismatch(format!(
                "fused: expected {} elements, got {}",
                expected_fused,
                fused.len()
            )));
        }
        if weights.len() != fused_dim {
            return Err(Eagle3Error::DimensionMismatch(format!(
                "weights: expected {} elements, got {}",
                fused_dim,
                weights.len()
            )));
        }

        let params = ConfidenceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            fused_dim: fused_dim as u32,
            bias,
            threshold: 0.0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = batch_size * seq_len;
        let output_bytes = output_len * mem::size_of::<f16>();

        let output = self.dispatch_confidence(
            slice_as_bytes(fused),
            slice_as_bytes(weights),
            output_bytes as u64,
            params,
            pipeline,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Check for early termination based on confidence threshold (f32).
    ///
    /// # Arguments
    /// * `confidence` - Confidence scores [batch_size, seq_len]
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `threshold` - Confidence threshold for termination
    ///
    /// # Returns
    /// Termination indices [batch_size] - index at which to terminate, or seq_len if none
    pub fn check_termination_f32(
        &self,
        confidence: &[f32],
        batch_size: usize,
        seq_len: usize,
        threshold: f32,
    ) -> Result<Vec<u32>, Eagle3Error> {
        let expected = batch_size * seq_len;
        if confidence.len() != expected {
            return Err(Eagle3Error::DimensionMismatch(format!(
                "confidence: expected {} elements, got {}",
                expected,
                confidence.len()
            )));
        }

        let params = ConfidenceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            fused_dim: 0, // not used
            bias: 0.0,    // not used
            threshold,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_bytes = batch_size * mem::size_of::<u32>();

        let output = self.dispatch_termination(
            slice_as_bytes(confidence),
            output_bytes as u64,
            params,
        )?;

        Ok(bytes_to_vec(&output))
    }

    // Internal dispatch methods

    fn dispatch_fusion(
        &self,
        input_bytes: &[u8],
        output_bytes: u64,
        params: FusionParams,
        pipeline: &ComputePipeline,
    ) -> Result<Vec<u8>, Eagle3Error> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eagle3_fusion_input"),
                contents: input_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eagle3_fusion_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eagle3_fusion_params"),
                contents: bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eagle3_fusion_bind_group"),
            layout: &self.fusion_layout,
            entries: &[
                buffer_binding(0, &input_buffer),
                buffer_binding(1, &output_buffer),
                buffer_binding(2, &params_buffer),
            ],
        });

        let total_elements =
            params.batch_size * params.seq_len * params.hidden_dim * params.num_layers;
        let workgroups = (total_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eagle3_fusion_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eagle3_fusion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eagle3_fusion_readback"),
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

    fn dispatch_confidence(
        &self,
        fused_bytes: &[u8],
        weights_bytes: &[u8],
        output_bytes: u64,
        params: ConfidenceParams,
        pipeline: &ComputePipeline,
    ) -> Result<Vec<u8>, Eagle3Error> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let fused_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eagle3_confidence_fused"),
                contents: fused_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let weights_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eagle3_confidence_weights"),
                contents: weights_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eagle3_confidence_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eagle3_confidence_params"),
                contents: bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eagle3_confidence_bind_group"),
            layout: &self.confidence_layout,
            entries: &[
                buffer_binding(0, &fused_buffer),
                buffer_binding(1, &weights_buffer),
                buffer_binding(2, &output_buffer),
                buffer_binding(3, &params_buffer),
            ],
        });

        let total_outputs = params.batch_size * params.seq_len;
        let workgroups = (total_outputs + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eagle3_confidence_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eagle3_confidence_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eagle3_confidence_readback"),
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

    fn dispatch_termination(
        &self,
        confidence_bytes: &[u8],
        output_bytes: u64,
        params: ConfidenceParams,
    ) -> Result<Vec<u8>, Eagle3Error> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let confidence_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eagle3_termination_confidence"),
                contents: confidence_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eagle3_termination_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("eagle3_termination_params"),
                contents: bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eagle3_termination_bind_group"),
            layout: &self.termination_layout,
            entries: &[
                buffer_binding(0, &confidence_buffer),
                buffer_binding(1, &output_buffer),
                buffer_binding(2, &params_buffer),
            ],
        });

        let workgroups = (params.batch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eagle3_termination_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eagle3_termination_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_termination_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eagle3_termination_readback"),
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
}

// Helper functions

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

fn buffer_binding(binding: u32, buffer: &Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T) as *const u8, mem::size_of::<T>()) }
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

fn read_buffer_sync(
    device: &Device,
    buffer: &Buffer,
    size: u64,
) -> Result<Vec<u8>, Eagle3Error> {
    let slice = buffer.slice(0..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(Eagle3Error::Wgpu(format!("map_async failed: {err}"))),
        Err(_) => return Err(Eagle3Error::Wgpu("map_async channel closed".into())),
    }

    let data = slice.get_mapped_range();
    let bytes = data.to_vec();
    drop(data);
    buffer.unmap();
    Ok(bytes)
}
