//! Medusa Heads Parallel Token Prediction WGPU Implementation
//!
//! Based on Medusa (ICML'24): Multiple auxiliary heads for speculative decoding
//! with parallel token prediction and tree-based verification.

use std::sync::mpsc;
use wgpu::util::DeviceExt;

/// WGSL shader source embedded at compile time (Fat Binary)
const SHADER_SOURCE: &str = include_str!("kernels/medusa.wgsl");

/// Workgroup size matching WGSL kernel
const WORKGROUP_SIZE: u32 = 256;

/// Parameters for Medusa head forward pass
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HeadForwardParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub hidden_dim: u32,
    pub vocab_size: u32,
}

/// Parameters for Top-K selection
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TopKParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub vocab_size: u32,
    pub k: u32,
}

/// Parameters for candidate tree building
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CandidateParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub k: u32,
    pub max_candidates: u32,
}

/// Result of Top-K selection
#[derive(Debug, Clone)]
pub struct TopKResult {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub k: u32,
}

/// Result of candidate tree building
#[derive(Debug, Clone)]
pub struct CandidateTree {
    pub candidates: Vec<u32>,
    pub counts: Vec<u32>,
    pub batch_size: u32,
}

/// Medusa heads implementation for WGPU
pub struct WgpuMedusa {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // F32 pipelines
    head_forward_pipeline_f32: wgpu::ComputePipeline,
    top_k_pipeline_f32: wgpu::ComputePipeline,
    log_softmax_pipeline_f32: wgpu::ComputePipeline,
    build_candidates_pipeline_f32: wgpu::ComputePipeline,
    // F16 pipelines (optional)
    head_forward_pipeline_f16: Option<wgpu::ComputePipeline>,
    top_k_pipeline_f16: Option<wgpu::ComputePipeline>,
    log_softmax_pipeline_f16: Option<wgpu::ComputePipeline>,
    // Bind group layouts
    head_forward_layout: wgpu::BindGroupLayout,
    top_k_layout: wgpu::BindGroupLayout,
    log_softmax_layout: wgpu::BindGroupLayout,
    candidates_layout: wgpu::BindGroupLayout,
}

impl WgpuMedusa {
    /// Create a new Medusa instance with the given WGPU device and queue
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("medusa_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        // Head forward layout
        let head_forward_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("head_forward_layout"),
                entries: &[
                    // Hidden states
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Weights
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bias
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output logits
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Top-K layout
        let top_k_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("top_k_layout"),
            entries: &[
                // Logits
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Top indices
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Top values
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Log-softmax layout
        let log_softmax_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("log_softmax_layout"),
                entries: &[
                    // Input logits
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output log probs
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Candidates layout
        let candidates_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("candidates_layout"),
                entries: &[
                    // Top indices from all heads
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output candidates
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Candidate counts
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layouts
        let head_forward_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("head_forward_pipeline_layout"),
                bind_group_layouts: &[&head_forward_layout],
                push_constant_ranges: &[],
            });

        let top_k_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("top_k_pipeline_layout"),
                bind_group_layouts: &[&top_k_layout],
                push_constant_ranges: &[],
            });

        let log_softmax_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("log_softmax_pipeline_layout"),
                bind_group_layouts: &[&log_softmax_layout],
                push_constant_ranges: &[],
            });

        let candidates_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("candidates_pipeline_layout"),
                bind_group_layouts: &[&candidates_layout],
                push_constant_ranges: &[],
            });

        // Create F32 pipelines
        let head_forward_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("head_forward_f32_pipeline"),
                layout: Some(&head_forward_pipeline_layout),
                module: &shader_module,
                entry_point: Some("medusa_head_forward_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let top_k_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("top_k_f32_pipeline"),
                layout: Some(&top_k_pipeline_layout),
                module: &shader_module,
                entry_point: Some("medusa_top_k_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let log_softmax_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("log_softmax_f32_pipeline"),
                layout: Some(&log_softmax_pipeline_layout),
                module: &shader_module,
                entry_point: Some("medusa_log_softmax_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let build_candidates_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build_candidates_f32_pipeline"),
                layout: Some(&candidates_pipeline_layout),
                module: &shader_module,
                entry_point: Some("medusa_build_candidates_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create F16 pipelines if supported
        let (head_forward_pipeline_f16, top_k_pipeline_f16, log_softmax_pipeline_f16) = if has_f16 {
            let head = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("head_forward_f16_pipeline"),
                layout: Some(&head_forward_pipeline_layout),
                module: &shader_module,
                entry_point: Some("medusa_head_forward_f16"),
                compilation_options: Default::default(),
                cache: None,
            });

            let topk = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("top_k_f16_pipeline"),
                layout: Some(&top_k_pipeline_layout),
                module: &shader_module,
                entry_point: Some("medusa_top_k_f16"),
                compilation_options: Default::default(),
                cache: None,
            });

            let logsoftmax = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("log_softmax_f16_pipeline"),
                layout: Some(&log_softmax_pipeline_layout),
                module: &shader_module,
                entry_point: Some("medusa_log_softmax_f16"),
                compilation_options: Default::default(),
                cache: None,
            });

            (Some(head), Some(topk), Some(logsoftmax))
        } else {
            (None, None, None)
        };

        Self {
            device,
            queue,
            head_forward_pipeline_f32,
            top_k_pipeline_f32,
            log_softmax_pipeline_f32,
            build_candidates_pipeline_f32,
            head_forward_pipeline_f16,
            top_k_pipeline_f16,
            log_softmax_pipeline_f16,
            head_forward_layout,
            top_k_layout,
            log_softmax_layout,
            candidates_layout,
        }
    }

    /// Forward pass through a Medusa head
    pub fn head_forward_f32(
        &self,
        hidden_states: &[f32],
        weights: &[f32],
        bias: &[f32],
        batch_size: u32,
        seq_len: u32,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> Vec<f32> {
        let params = HeadForwardParams {
            batch_size,
            seq_len,
            hidden_dim,
            vocab_size,
        };

        let hidden_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hidden_buffer"),
                contents: bytemuck::cast_slice(hidden_states),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let weights_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weights_buffer"),
                contents: bytemuck::cast_slice(weights),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let bias_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bias_buffer"),
                contents: bytemuck::cast_slice(bias),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (batch_size * seq_len * vocab_size) as usize;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("head_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("head_forward_bind_group"),
            layout: &self.head_forward_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("head_forward_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("head_forward_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.head_forward_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = batch_size * seq_len * vocab_size;
            let workgroups = (total + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_staging"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        self.read_buffer_f32(&staging, output_size)
    }

    /// Select top-K tokens from logits
    pub fn top_k_f32(
        &self,
        logits: &[f32],
        batch_size: u32,
        seq_len: u32,
        vocab_size: u32,
        k: u32,
    ) -> TopKResult {
        let params = TopKParams {
            batch_size,
            seq_len,
            vocab_size,
            k,
        };

        let logits_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("topk_logits"),
                contents: bytemuck::cast_slice(logits),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (batch_size * seq_len * k) as usize;
        let indices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_indices"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let values_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_values"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("topk_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("topk_bind_group"),
            layout: &self.top_k_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("topk_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("topk_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.top_k_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per position
            pass.dispatch_workgroups(batch_size * seq_len, 1, 1);
        }

        let indices_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indices_staging"),
            size: indices_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let values_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("values_staging"),
            size: values_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&indices_buffer, 0, &indices_staging, 0, indices_buffer.size());
        encoder.copy_buffer_to_buffer(&values_buffer, 0, &values_staging, 0, values_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let indices = self.read_buffer_u32(&indices_staging, output_size);
        let values = self.read_buffer_f32(&values_staging, output_size);

        TopKResult {
            indices,
            values,
            batch_size,
            seq_len,
            k,
        }
    }

    /// Build candidate tree from multiple head predictions
    pub fn build_candidates(
        &self,
        top_indices: &[u32], // [batch, num_heads, k]
        batch_size: u32,
        num_heads: u32,
        k: u32,
        max_candidates: u32,
    ) -> CandidateTree {
        let params = CandidateParams {
            batch_size,
            num_heads,
            k,
            max_candidates,
        };

        let indices_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cand_indices"),
                contents: bytemuck::cast_slice(top_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let candidates_size = (batch_size * max_candidates) as usize;
        let candidates_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candidates_output"),
            size: (candidates_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counts_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counts_output"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cand_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("candidates_bind_group"),
            layout: &self.candidates_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("candidates_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("candidates_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_candidates_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (batch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let cand_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cand_staging"),
            size: candidates_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counts_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counts_staging"),
            size: counts_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&candidates_buffer, 0, &cand_staging, 0, candidates_buffer.size());
        encoder.copy_buffer_to_buffer(&counts_buffer, 0, &counts_staging, 0, counts_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let candidates = self.read_buffer_u32(&cand_staging, candidates_size);
        let counts = self.read_buffer_u32(&counts_staging, batch_size as usize);

        CandidateTree {
            candidates,
            counts,
            batch_size,
        }
    }

    /// Check if F16 operations are available
    pub fn has_f16_support(&self) -> bool {
        self.head_forward_pipeline_f16.is_some()
    }

    // Helper methods
    fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let (tx, rx) = mpsc::channel();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        buffer.unmap();
        result
    }

    fn read_buffer_u32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
        let (tx, rx) = mpsc::channel();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        buffer.unmap();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_device_with_f16() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;

        // Check if f16 is supported
        if !adapter.features().contains(wgpu::Features::SHADER_F16) {
            return None;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SHADER_F16,
                ..Default::default()
            })
            .await
            .ok()?;

        Some((device, queue))
    }

    #[test]
    fn test_head_forward() {
        pollster::block_on(async {
            let Some((device, queue)) = create_device_with_f16().await else {
                println!("No GPU with f16 support available, skipping test");
                return;
            };

            let medusa = WgpuMedusa::new(device, queue);

            let batch_size = 2;
            let seq_len = 4;
            let hidden_dim = 64;
            let vocab_size = 100;

            let hidden_states: Vec<f32> = (0..batch_size * seq_len * hidden_dim)
                .map(|i| (i as f32) * 0.01)
                .collect();
            let weights: Vec<f32> = (0..hidden_dim * vocab_size)
                .map(|i| (i as f32) * 0.001)
                .collect();
            let bias: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.1).collect();

            let logits = medusa.head_forward_f32(
                &hidden_states,
                &weights,
                &bias,
                batch_size as u32,
                seq_len as u32,
                hidden_dim as u32,
                vocab_size as u32,
            );

            assert_eq!(logits.len(), batch_size * seq_len * vocab_size);
        });
    }
}
