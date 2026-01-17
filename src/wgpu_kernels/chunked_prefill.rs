// Chunked Prefill / POD-Attention WGPU Implementation
// Based on Sarathi-Serve, DeepSpeed-FastGen, POD-Attention
//
// Features:
// - Chunked attention computation with online softmax
// - Log-sum-exp based chunk merging
// - POD-Attention workload splitting for prefill/decode interleaving
// - Batch scheduling primitives

use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("kernels/chunked_prefill.wgsl");

/// Parameters for chunked attention computation
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkAttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub chunk_size: u32,
    pub total_seq_len: u32,
    pub chunk_idx: u32,
    pub scale: f32,
    pub _pad0: u32,
}

/// Parameters for chunk merging
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkMergeParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub num_chunks: u32,
    pub chunk_size: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Parameters for POD-Attention workload splitting
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PODSplitParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub prefill_len: u32,
    pub decode_len: u32,
    pub prefill_ratio: f32,  // 0.0-1.0, fraction of compute for prefill
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Parameters for batch scheduling
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScheduleParams {
    pub num_requests: u32,
    pub max_batch_size: u32,
    pub max_seq_len: u32,
    pub chunk_size: u32,
}

/// Result of chunked attention computation
#[derive(Debug, Clone)]
pub struct ChunkAttentionResult {
    /// Output tensor [batch, chunk_size, num_heads, head_dim]
    pub output: Vec<f32>,
    /// Log-sum-exp for merging [batch, chunk_size, num_heads]
    pub lse: Vec<f32>,
}

/// Result of POD-Attention split
#[derive(Debug, Clone)]
pub struct PODSplitResult {
    /// Work allocation for prefill per batch item
    pub prefill_allocation: Vec<u32>,
    /// Work allocation for decode per batch item
    pub decode_allocation: Vec<u32>,
}

/// Result of batch scheduling
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    /// Batch assignment for each request (0xFFFFFFFF if not scheduled)
    pub batch_assignments: Vec<u32>,
    /// Offsets within each batch
    pub batch_offsets: Vec<u32>,
    /// Number of batches created
    pub num_batches: u32,
}

/// WGPU Chunked Prefill implementation
pub struct WgpuChunkedPrefill {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Compute pipelines
    attention_f32_pipeline: wgpu::ComputePipeline,
    merge_f32_pipeline: wgpu::ComputePipeline,
    pod_split_pipeline: wgpu::ComputePipeline,
    schedule_pipeline: wgpu::ComputePipeline,

    // F16 pipelines (optional)
    attention_f16_pipeline: Option<wgpu::ComputePipeline>,
    merge_f16_pipeline: Option<wgpu::ComputePipeline>,

    // Bind group layouts
    attention_layout: wgpu::BindGroupLayout,
    merge_layout: wgpu::BindGroupLayout,
    pod_split_layout: wgpu::BindGroupLayout,
    schedule_layout: wgpu::BindGroupLayout,
}

impl WgpuChunkedPrefill {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chunked_prefill_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Attention layout: Q, K, V, O, LSE, params
        let attention_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention_layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        // Merge layout: chunks, lse, output, params
        let merge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("merge_layout"),
            entries: &[
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

        // POD split layout: prefill_lens, decode_lens, prefill_alloc, decode_alloc, params
        let pod_split_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pod_split_layout"),
            entries: &[
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

        // Schedule layout: request_lens, batch_assign, batch_offsets, num_batches, params
        let schedule_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("schedule_layout"),
            entries: &[
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

        // Create pipeline layouts
        let attention_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attention_pipeline_layout"),
            bind_group_layouts: &[&attention_layout],
            push_constant_ranges: &[],
        });

        let merge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("merge_pipeline_layout"),
            bind_group_layouts: &[&merge_layout],
            push_constant_ranges: &[],
        });

        let pod_split_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pod_split_pipeline_layout"),
            bind_group_layouts: &[&pod_split_layout],
            push_constant_ranges: &[],
        });

        let schedule_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("schedule_pipeline_layout"),
            bind_group_layouts: &[&schedule_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let attention_f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chunked_prefill_attention_f32"),
            layout: Some(&attention_pipeline_layout),
            module: &shader_module,
            entry_point: Some("chunked_prefill_attention_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let merge_f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chunked_prefill_merge_f32"),
            layout: Some(&merge_pipeline_layout),
            module: &shader_module,
            entry_point: Some("chunked_prefill_merge_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pod_split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pod_attention_split_f32"),
            layout: Some(&pod_split_pipeline_layout),
            module: &shader_module,
            entry_point: Some("pod_attention_split_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let schedule_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chunked_prefill_schedule_f32"),
            layout: Some(&schedule_pipeline_layout),
            module: &shader_module,
            entry_point: Some("chunked_prefill_schedule_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        // F16 pipelines (conditional)
        let attention_f16_pipeline = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chunked_prefill_attention_f16"),
                layout: Some(&attention_pipeline_layout),
                module: &shader_module,
                entry_point: Some("chunked_prefill_attention_f16"),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        let merge_f16_pipeline = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chunked_prefill_merge_f16"),
                layout: Some(&merge_pipeline_layout),
                module: &shader_module,
                entry_point: Some("chunked_prefill_merge_f16"),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        Self {
            device,
            queue,
            attention_f32_pipeline,
            merge_f32_pipeline,
            pod_split_pipeline,
            schedule_pipeline,
            attention_f16_pipeline,
            merge_f16_pipeline,
            attention_layout,
            merge_layout,
            pod_split_layout,
            schedule_layout,
        }
    }

    /// Check if F16 kernels are available
    pub fn has_f16_support(&self) -> bool {
        self.attention_f16_pipeline.is_some()
    }

    /// Compute chunked attention for a single chunk (F32)
    pub fn chunked_attention_f32(
        &self,
        q: &[f32],           // [batch, chunk_size, num_heads, head_dim]
        k: &[f32],           // [batch, total_seq_len, num_heads, head_dim]
        v: &[f32],           // [batch, total_seq_len, num_heads, head_dim]
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        chunk_size: u32,
        total_seq_len: u32,
        chunk_idx: u32,
    ) -> ChunkAttentionResult {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let output_size = (batch_size * chunk_size * num_heads * head_dim) as usize;
        let lse_size = (batch_size * chunk_size * num_heads) as usize;

        let params = ChunkAttentionParams {
            batch_size,
            num_heads,
            head_dim,
            chunk_size,
            total_seq_len,
            chunk_idx,
            scale,
            _pad0: 0,
        };

        let q_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q_buffer"),
            contents: bytemuck::cast_slice(q),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let k_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("k_buffer"),
            contents: bytemuck::cast_slice(k),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let v_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("v_buffer"),
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let o_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("o_buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let lse_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lse_buffer"),
            size: (lse_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attention_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attention_bind_group"),
            layout: &self.attention_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: o_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lse_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attention_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attention_f32_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch: one workgroup per (batch * num_heads, chunk_size)
            pass.dispatch_workgroups(batch_size * num_heads, chunk_size, 1);
        }

        // Read back output and lse
        let o_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("o_staging"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lse_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lse_staging"),
            size: (lse_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &o_buffer,
            0,
            &o_staging,
            0,
            (output_size * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &lse_buffer,
            0,
            &lse_staging,
            0,
            (lse_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        o_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        lse_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();

        let o_data = o_staging.slice(..).get_mapped_range();
        let output: Vec<f32> = bytemuck::cast_slice(&o_data).to_vec();
        drop(o_data);
        o_staging.unmap();

        let lse_data = lse_staging.slice(..).get_mapped_range();
        let lse: Vec<f32> = bytemuck::cast_slice(&lse_data).to_vec();
        drop(lse_data);
        lse_staging.unmap();

        ChunkAttentionResult { output, lse }
    }

    /// Merge chunk outputs (F32)
    pub fn merge_chunks_f32(
        &self,
        chunk_outputs: &[f32],  // [num_chunks, batch, chunk_size, num_heads, head_dim]
        chunk_lse: &[f32],      // [num_chunks, batch, chunk_size, num_heads]
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        num_chunks: u32,
        chunk_size: u32,
    ) -> Vec<f32> {
        let total_len = num_chunks * chunk_size;
        let output_size = (batch_size * total_len * num_heads * head_dim) as usize;

        let params = ChunkMergeParams {
            batch_size,
            num_heads,
            head_dim,
            num_chunks,
            chunk_size,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let chunks_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunks_buffer"),
            contents: bytemuck::cast_slice(chunk_outputs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let lse_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lse_buffer"),
            contents: bytemuck::cast_slice(chunk_lse),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge_output_buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("merge_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("merge_bind_group"),
            layout: &self.merge_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lse_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("merge_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.merge_f32_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((output_size as u32 + 255) / 256, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge_staging"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (output_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = staging_buffer.slice(..).get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Split workload between prefill and decode (POD-Attention)
    pub fn pod_attention_split(
        &self,
        prefill_lens: &[u32],
        decode_lens: &[u32],
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        prefill_len: u32,
        decode_len: u32,
        prefill_ratio: f32,
    ) -> PODSplitResult {
        let params = PODSplitParams {
            batch_size,
            num_heads,
            head_dim,
            prefill_len,
            decode_len,
            prefill_ratio,
            _pad0: 0,
            _pad1: 0,
        };

        let prefill_lens_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefill_lens_buffer"),
            contents: bytemuck::cast_slice(prefill_lens),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let decode_lens_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("decode_lens_buffer"),
            contents: bytemuck::cast_slice(decode_lens),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let prefill_alloc_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefill_alloc_buffer"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let decode_alloc_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decode_alloc_buffer"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pod_split_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pod_split_bind_group"),
            layout: &self.pod_split_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefill_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: decode_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prefill_alloc_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: decode_alloc_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pod_split_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pod_split_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pod_split_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((batch_size + 255) / 256, 1, 1);
        }

        let prefill_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefill_staging"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let decode_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decode_staging"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &prefill_alloc_buffer,
            0,
            &prefill_staging,
            0,
            (batch_size as usize * std::mem::size_of::<u32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &decode_alloc_buffer,
            0,
            &decode_staging,
            0,
            (batch_size as usize * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        prefill_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        decode_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();

        let prefill_data = prefill_staging.slice(..).get_mapped_range();
        let prefill_allocation: Vec<u32> = bytemuck::cast_slice(&prefill_data).to_vec();
        drop(prefill_data);
        prefill_staging.unmap();

        let decode_data = decode_staging.slice(..).get_mapped_range();
        let decode_allocation: Vec<u32> = bytemuck::cast_slice(&decode_data).to_vec();
        drop(decode_data);
        decode_staging.unmap();

        PODSplitResult {
            prefill_allocation,
            decode_allocation,
        }
    }

    /// Schedule requests into batches
    pub fn schedule_batches(
        &self,
        request_lens: &[u32],
        num_requests: u32,
        max_batch_size: u32,
        max_seq_len: u32,
        chunk_size: u32,
    ) -> ScheduleResult {
        let params = ScheduleParams {
            num_requests,
            max_batch_size,
            max_seq_len,
            chunk_size,
        };

        let request_lens_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("request_lens_buffer"),
            contents: bytemuck::cast_slice(request_lens),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let batch_assign_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_assign_buffer"),
            size: (num_requests as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let batch_offsets_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_offsets_buffer"),
            size: (max_batch_size as usize * num_requests as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let num_batches_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("num_batches_buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("schedule_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("schedule_bind_group"),
            layout: &self.schedule_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: request_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batch_assign_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: batch_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: num_batches_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("schedule_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("schedule_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.schedule_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);  // Single workgroup for sequential scheduling
        }

        let assign_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("assign_staging"),
            size: (num_requests as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let offsets_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offsets_staging"),
            size: (max_batch_size as usize * num_requests as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let num_batches_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("num_batches_staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &batch_assign_buffer,
            0,
            &assign_staging,
            0,
            (num_requests as usize * std::mem::size_of::<u32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &batch_offsets_buffer,
            0,
            &offsets_staging,
            0,
            (max_batch_size as usize * num_requests as usize * std::mem::size_of::<u32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &num_batches_buffer,
            0,
            &num_batches_staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        let tx3 = tx.clone();
        assign_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        offsets_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });
        num_batches_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx3.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();

        let assign_data = assign_staging.slice(..).get_mapped_range();
        let batch_assignments: Vec<u32> = bytemuck::cast_slice(&assign_data).to_vec();
        drop(assign_data);
        assign_staging.unmap();

        let offsets_data = offsets_staging.slice(..).get_mapped_range();
        let batch_offsets: Vec<u32> = bytemuck::cast_slice(&offsets_data).to_vec();
        drop(offsets_data);
        offsets_staging.unmap();

        let num_batches_data = num_batches_staging.slice(..).get_mapped_range();
        let num_batches: u32 = *bytemuck::from_bytes(&num_batches_data);
        drop(num_batches_data);
        num_batches_staging.unmap();

        ScheduleResult {
            batch_assignments,
            batch_offsets,
            num_batches,
        }
    }
}
