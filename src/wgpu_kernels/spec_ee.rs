//! WGPU SpecEE / LayerSkip early-exit speculative decoding kernel.
//!
//! Based on SpecEE (ISCA'25) and LayerSkip (ACL'24): Per-layer early exit
//! decisions based on hidden state confidence, enabling 1.5-3x inference
//! acceleration by skipping unnecessary layers.
//!
//! # Key Operations
//! - Per-layer confidence computation via linear projection + sigmoid
//! - Early exit decision based on confidence threshold
//! - LM head projection for early exit logits
//! - Argmax for greedy token selection

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

const SHADER_SOURCE: &str = include_str!("kernels/spec_ee.wgsl");
const WORKGROUP_SIZE: u32 = 256;

// Kernel entry points
const KERNEL_CONFIDENCE_F32: &str = "spec_ee_compute_confidence_f32";
const KERNEL_CONFIDENCE_F16: &str = "spec_ee_compute_confidence_f16";
const KERNEL_LM_HEAD_F32: &str = "spec_ee_lm_head_f32";
const KERNEL_LM_HEAD_F16: &str = "spec_ee_lm_head_f16";
const KERNEL_ARGMAX_F32: &str = "spec_ee_argmax_f32";

/// WGSL shader struct: ConfidenceParams
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ConfidenceParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    layer_idx: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// WGSL shader struct: LMHeadParams
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct LMHeadParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    vocab_size: u32,
    temperature: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Errors surfaced by the WGPU SpecEE kernels.
#[derive(Debug)]
pub enum SpecEEError {
    /// WGPU driver error or initialization failure.
    Wgpu(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Unsupported device capability.
    Unsupported(String),
    /// Dimension mismatch.
    DimensionMismatch(String),
}

impl fmt::Display for SpecEEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu(msg) => write!(f, "WGPU error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
        }
    }
}

impl std::error::Error for SpecEEError {}

/// Result of early exit confidence computation.
pub struct EarlyExitResult {
    /// Confidence scores [batch_size, seq_len] in range [0, 1].
    pub confidence: Vec<f32>,
    /// Exit decisions [batch_size, seq_len] - 1 if should exit, 0 otherwise.
    pub should_exit: Vec<u32>,
}

/// Result of early exit confidence computation (f16).
pub struct EarlyExitResultF16 {
    /// Confidence scores [batch_size, seq_len] in range [0, 1].
    pub confidence: Vec<f16>,
    /// Exit decisions [batch_size, seq_len] - 1 if should exit, 0 otherwise.
    pub should_exit: Vec<u32>,
}

/// SpecEE WGPU kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Per-layer early exit confidence
/// - LM head projection for early exit
/// - Argmax token selection
pub struct SpecEEKernel {
    device: Arc<Device>,
    queue: Arc<Queue>,
    // Confidence pipelines
    confidence_layout: BindGroupLayout,
    pipeline_confidence_f32: ComputePipeline,
    pipeline_confidence_f16: Option<ComputePipeline>,
    // LM head pipelines
    lm_head_layout: BindGroupLayout,
    pipeline_lm_head_f32: ComputePipeline,
    pipeline_lm_head_f16: Option<ComputePipeline>,
    // Argmax pipeline
    argmax_layout: BindGroupLayout,
    pipeline_argmax_f32: ComputePipeline,
}

impl SpecEEKernel {
    /// Create a SpecEE kernel wrapper for an existing WGPU device.
    pub fn new(device: &Device, queue: &Queue) -> Result<Self, SpecEEError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spec_ee.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // Confidence bind group layout: hidden, weights, confidence, should_exit, params
        let confidence_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spec_ee_confidence_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // hidden
                buffer_layout_entry(1, true),  // conf_weights
                buffer_layout_entry(2, false), // confidence
                buffer_layout_entry(3, false), // should_exit
                uniform_layout_entry(4),       // params
            ],
        });

        // LM head bind group layout: hidden, weights, logits, params
        let lm_head_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spec_ee_lm_head_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // hidden
                buffer_layout_entry(1, true),  // lm_weights
                buffer_layout_entry(2, false), // logits
                uniform_layout_entry(3),       // params
            ],
        });

        // Argmax bind group layout: logits, tokens, params
        let argmax_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spec_ee_argmax_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // logits
                buffer_layout_entry(1, false), // tokens
                uniform_layout_entry(2),       // params
            ],
        });

        let confidence_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("spec_ee_confidence_pipeline_layout"),
                bind_group_layouts: &[&confidence_layout],
                push_constant_ranges: &[],
            });

        let lm_head_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("spec_ee_lm_head_pipeline_layout"),
                bind_group_layouts: &[&lm_head_layout],
                push_constant_ranges: &[],
            });

        let argmax_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("spec_ee_argmax_pipeline_layout"),
                bind_group_layouts: &[&argmax_layout],
                push_constant_ranges: &[],
            });

        // Create F32 pipelines
        let pipeline_confidence_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spec_ee_confidence_f32"),
                layout: Some(&confidence_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_CONFIDENCE_F32),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline_lm_head_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spec_ee_lm_head_f32"),
                layout: Some(&lm_head_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_LM_HEAD_F32),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline_argmax_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spec_ee_argmax_f32"),
            layout: Some(&argmax_pipeline_layout),
            module: &shader,
            entry_point: Some(KERNEL_ARGMAX_F32),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create F16 pipelines if supported
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        let pipeline_confidence_f16 = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spec_ee_confidence_f16"),
                layout: Some(&confidence_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_CONFIDENCE_F16),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        let pipeline_lm_head_f16 = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spec_ee_lm_head_f16"),
                layout: Some(&lm_head_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_LM_HEAD_F16),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        Ok(Self {
            device: Arc::new(device.clone()),
            queue: Arc::new(queue.clone()),
            confidence_layout,
            pipeline_confidence_f32,
            pipeline_confidence_f16,
            lm_head_layout,
            pipeline_lm_head_f32,
            pipeline_lm_head_f16,
            argmax_layout,
            pipeline_argmax_f32,
        })
    }

    /// Create a SpecEE kernel wrapper with a newly initialized device.
    pub fn create_default(require_f16: bool) -> Result<Self, SpecEEError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| SpecEEError::Wgpu(format!("no compatible adapter found: {e}")))?;

        let mut features = wgpu::Features::empty();
        if require_f16 {
            if adapter.features().contains(wgpu::Features::SHADER_F16) {
                features |= wgpu::Features::SHADER_F16;
            } else {
                return Err(SpecEEError::Unsupported(
                    "adapter does not support shader f16".into(),
                ));
            }
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gllm-wgpu-spec-ee"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|err| SpecEEError::Wgpu(format!("request_device failed: {err}")))?;

        Self::new(&device, &queue)
    }

    /// Compute early exit confidence for a layer (f32).
    ///
    /// # Arguments
    /// * `hidden` - Hidden states [batch_size, seq_len, hidden_dim]
    /// * `conf_weights` - Confidence projection weights [hidden_dim]
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension
    /// * `layer_idx` - Current layer index
    /// * `threshold` - Confidence threshold for early exit
    ///
    /// # Returns
    /// EarlyExitResult containing confidence scores and exit decisions
    pub fn compute_confidence_f32(
        &self,
        hidden: &[f32],
        conf_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        layer_idx: usize,
        threshold: f32,
    ) -> Result<EarlyExitResult, SpecEEError> {
        let expected_hidden = batch_size * seq_len * hidden_dim;
        if hidden.len() != expected_hidden {
            return Err(SpecEEError::DimensionMismatch(format!(
                "hidden: expected {} elements, got {}",
                expected_hidden,
                hidden.len()
            )));
        }
        if conf_weights.len() != hidden_dim {
            return Err(SpecEEError::DimensionMismatch(format!(
                "conf_weights: expected {} elements, got {}",
                hidden_dim,
                conf_weights.len()
            )));
        }

        let params = ConfidenceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            layer_idx: layer_idx as u32,
            threshold,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = batch_size * seq_len;
        let confidence_bytes = output_len * mem::size_of::<f32>();
        let exit_bytes = output_len * mem::size_of::<u32>();

        let (confidence_data, exit_data) = self.dispatch_confidence(
            slice_as_bytes(hidden),
            slice_as_bytes(conf_weights),
            confidence_bytes as u64,
            exit_bytes as u64,
            params,
            &self.pipeline_confidence_f32,
        )?;

        Ok(EarlyExitResult {
            confidence: bytes_to_vec(&confidence_data),
            should_exit: bytes_to_vec(&exit_data),
        })
    }

    /// Compute early exit confidence for a layer (f16).
    pub fn compute_confidence_f16(
        &self,
        hidden: &[f16],
        conf_weights: &[f16],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        layer_idx: usize,
        threshold: f32,
    ) -> Result<EarlyExitResultF16, SpecEEError> {
        let pipeline = self.pipeline_confidence_f16.as_ref().ok_or_else(|| {
            SpecEEError::Unsupported("device does not support f16 kernels".into())
        })?;

        let expected_hidden = batch_size * seq_len * hidden_dim;
        if hidden.len() != expected_hidden {
            return Err(SpecEEError::DimensionMismatch(format!(
                "hidden: expected {} elements, got {}",
                expected_hidden,
                hidden.len()
            )));
        }
        if conf_weights.len() != hidden_dim {
            return Err(SpecEEError::DimensionMismatch(format!(
                "conf_weights: expected {} elements, got {}",
                hidden_dim,
                conf_weights.len()
            )));
        }

        let params = ConfidenceParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            layer_idx: layer_idx as u32,
            threshold,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = batch_size * seq_len;
        let confidence_bytes = output_len * mem::size_of::<f16>();
        let exit_bytes = output_len * mem::size_of::<u32>();

        let (confidence_data, exit_data) = self.dispatch_confidence(
            slice_as_bytes(hidden),
            slice_as_bytes(conf_weights),
            confidence_bytes as u64,
            exit_bytes as u64,
            params,
            pipeline,
        )?;

        Ok(EarlyExitResultF16 {
            confidence: bytes_to_vec(&confidence_data),
            should_exit: bytes_to_vec(&exit_data),
        })
    }

    /// Compute LM head projection for early exit (f32).
    ///
    /// # Arguments
    /// * `hidden` - Hidden states [batch_size, seq_len, hidden_dim]
    /// * `lm_weights` - LM head weights [hidden_dim, vocab_size]
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension
    /// * `vocab_size` - Vocabulary size
    /// * `temperature` - Temperature for scaling logits (0.0 for no scaling)
    ///
    /// # Returns
    /// Logits [batch_size, seq_len, vocab_size]
    pub fn lm_head_f32(
        &self,
        hidden: &[f32],
        lm_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
        temperature: f32,
    ) -> Result<Vec<f32>, SpecEEError> {
        let expected_hidden = batch_size * seq_len * hidden_dim;
        if hidden.len() != expected_hidden {
            return Err(SpecEEError::DimensionMismatch(format!(
                "hidden: expected {} elements, got {}",
                expected_hidden,
                hidden.len()
            )));
        }
        let expected_weights = hidden_dim * vocab_size;
        if lm_weights.len() != expected_weights {
            return Err(SpecEEError::DimensionMismatch(format!(
                "lm_weights: expected {} elements, got {}",
                expected_weights,
                lm_weights.len()
            )));
        }

        let params = LMHeadParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            vocab_size: vocab_size as u32,
            temperature,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = batch_size * seq_len * vocab_size;
        let output_bytes = output_len * mem::size_of::<f32>();

        let output = self.dispatch_lm_head(
            slice_as_bytes(hidden),
            slice_as_bytes(lm_weights),
            output_bytes as u64,
            params,
            &self.pipeline_lm_head_f32,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Compute LM head projection for early exit (f16).
    pub fn lm_head_f16(
        &self,
        hidden: &[f16],
        lm_weights: &[f16],
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
        temperature: f32,
    ) -> Result<Vec<f16>, SpecEEError> {
        let pipeline = self.pipeline_lm_head_f16.as_ref().ok_or_else(|| {
            SpecEEError::Unsupported("device does not support f16 kernels".into())
        })?;

        let expected_hidden = batch_size * seq_len * hidden_dim;
        if hidden.len() != expected_hidden {
            return Err(SpecEEError::DimensionMismatch(format!(
                "hidden: expected {} elements, got {}",
                expected_hidden,
                hidden.len()
            )));
        }
        let expected_weights = hidden_dim * vocab_size;
        if lm_weights.len() != expected_weights {
            return Err(SpecEEError::DimensionMismatch(format!(
                "lm_weights: expected {} elements, got {}",
                expected_weights,
                lm_weights.len()
            )));
        }

        let params = LMHeadParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: hidden_dim as u32,
            vocab_size: vocab_size as u32,
            temperature,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = batch_size * seq_len * vocab_size;
        let output_bytes = output_len * mem::size_of::<f16>();

        let output = self.dispatch_lm_head(
            slice_as_bytes(hidden),
            slice_as_bytes(lm_weights),
            output_bytes as u64,
            params,
            pipeline,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Compute argmax over vocab dimension (f32).
    ///
    /// # Arguments
    /// * `logits` - Logits [batch_size, seq_len, vocab_size]
    /// * `batch_size` - Batch dimension
    /// * `seq_len` - Sequence length
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Token indices [batch_size, seq_len]
    pub fn argmax_f32(
        &self,
        logits: &[f32],
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, SpecEEError> {
        let expected = batch_size * seq_len * vocab_size;
        if logits.len() != expected {
            return Err(SpecEEError::DimensionMismatch(format!(
                "logits: expected {} elements, got {}",
                expected,
                logits.len()
            )));
        }

        let params = LMHeadParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_dim: 0, // not used
            vocab_size: vocab_size as u32,
            temperature: 0.0, // not used
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = batch_size * seq_len;
        let output_bytes = output_len * mem::size_of::<u32>();

        let output = self.dispatch_argmax(slice_as_bytes(logits), output_bytes as u64, params)?;

        Ok(bytes_to_vec(&output))
    }

    // Internal dispatch methods

    fn dispatch_confidence(
        &self,
        hidden_bytes: &[u8],
        weights_bytes: &[u8],
        confidence_bytes: u64,
        exit_bytes: u64,
        params: ConfidenceParams,
        pipeline: &ComputePipeline,
    ) -> Result<(Vec<u8>, Vec<u8>), SpecEEError> {
        if confidence_bytes == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let hidden_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_hidden"),
                contents: hidden_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let weights_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_conf_weights"),
                contents: weights_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let conf_padded = align_up(confidence_bytes, max_align());
        let confidence_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_confidence"),
            size: conf_padded,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let exit_padded = align_up(exit_bytes, max_align());
        let exit_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_should_exit"),
            size: exit_padded,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_conf_params"),
                contents: bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spec_ee_confidence_bind_group"),
            layout: &self.confidence_layout,
            entries: &[
                buffer_binding(0, &hidden_buffer),
                buffer_binding(1, &weights_buffer),
                buffer_binding(2, &confidence_buffer),
                buffer_binding(3, &exit_buffer),
                buffer_binding(4, &params_buffer),
            ],
        });

        let total = params.batch_size * params.seq_len;
        let workgroups = (total + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spec_ee_confidence_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spec_ee_confidence_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let conf_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_conf_readback"),
            size: conf_padded,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let exit_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_exit_readback"),
            size: exit_padded,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&confidence_buffer, 0, &conf_readback, 0, conf_padded);
        encoder.copy_buffer_to_buffer(&exit_buffer, 0, &exit_readback, 0, exit_padded);
        self.queue.submit(Some(encoder.finish()));

        let mut conf_data = read_buffer_sync(&self.device, &conf_readback, conf_padded)?;
        conf_data.truncate(confidence_bytes as usize);

        let mut exit_data = read_buffer_sync(&self.device, &exit_readback, exit_padded)?;
        exit_data.truncate(exit_bytes as usize);

        Ok((conf_data, exit_data))
    }

    fn dispatch_lm_head(
        &self,
        hidden_bytes: &[u8],
        weights_bytes: &[u8],
        output_bytes: u64,
        params: LMHeadParams,
        pipeline: &ComputePipeline,
    ) -> Result<Vec<u8>, SpecEEError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let hidden_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_lm_hidden"),
                contents: hidden_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let weights_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_lm_weights"),
                contents: weights_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_lm_logits"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_lm_params"),
                contents: bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spec_ee_lm_head_bind_group"),
            layout: &self.lm_head_layout,
            entries: &[
                buffer_binding(0, &hidden_buffer),
                buffer_binding(1, &weights_buffer),
                buffer_binding(2, &output_buffer),
                buffer_binding(3, &params_buffer),
            ],
        });

        let total = params.batch_size * params.seq_len * params.vocab_size;
        let workgroups = (total + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spec_ee_lm_head_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spec_ee_lm_head_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_lm_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let mut data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        data.truncate(output_bytes as usize);
        Ok(data)
    }

    fn dispatch_argmax(
        &self,
        logits_bytes: &[u8],
        output_bytes: u64,
        params: LMHeadParams,
    ) -> Result<Vec<u8>, SpecEEError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let logits_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_argmax_logits"),
                contents: logits_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_argmax_tokens"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spec_ee_argmax_params"),
                contents: bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spec_ee_argmax_bind_group"),
            layout: &self.argmax_layout,
            entries: &[
                buffer_binding(0, &logits_buffer),
                buffer_binding(1, &output_buffer),
                buffer_binding(2, &params_buffer),
            ],
        });

        let total = params.batch_size * params.seq_len;
        let workgroups = (total + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spec_ee_argmax_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spec_ee_argmax_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_argmax_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spec_ee_argmax_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let mut data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        data.truncate(output_bytes as usize);
        Ok(data)
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

fn read_buffer_sync(device: &Device, buffer: &Buffer, size: u64) -> Result<Vec<u8>, SpecEEError> {
    let slice = buffer.slice(0..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(SpecEEError::Wgpu(format!("map_async failed: {err}"))),
        Err(_) => return Err(SpecEEError::Wgpu("map_async channel closed".into())),
    }

    let data = slice.get_mapped_range();
    let bytes = data.to_vec();
    drop(data);
    buffer.unmap();
    Ok(bytes)
}
