//! WGPU Fused MoE FFN Kernel
//!
//! Single kernel launch for Mixture of Experts FFN computation.
//! Computes: output = sum(weight[k] * FFN(input, expert[k])) for selected experts
//! FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

use wgpu::util::DeviceExt;

/// Parameters for MoE FFN kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MoEFfnParams {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_tokens: u32,
    pub top_k: u32,
    pub num_experts: u32,
    pub _padding0: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Expert weight references for MoE computation.
pub struct ExpertWeights<'a> {
    pub gate: &'a wgpu::Buffer,
    pub up: &'a wgpu::Buffer,
    pub down: &'a wgpu::Buffer,
}

/// Fused MoE FFN kernel for WGPU.
pub struct WgpuMoeFfn {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ffn_pipeline: wgpu::ComputePipeline,
    zero_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuMoeFfn {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader_src = include_str!("kernels/moe_ffn.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MoE FFN Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout for MoE FFN
        // 0: params (uniform)
        // 1: input (storage, read)
        // 2: expert_indices (storage, read)
        // 3: expert_weights (storage, read)
        // 4: output (storage, read_write)
        // 5: gate_weights (storage, read)
        // 6: up_weights (storage, read)
        // 7: down_weights (storage, read)
        // 8: scratch (storage, read_write)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MoE FFN Bind Group Layout"),
            entries: &[
                // params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // input storage (read)
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
                // expert_indices storage (read)
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
                // expert_weights storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // output storage (read_write)
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
                // gate_weights storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // up_weights storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // down_weights storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // scratch storage (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MoE FFN Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let ffn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE FFN Forward Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("moe_ffn_forward"),
            compilation_options: Default::default(),
            cache: None,
        });

        let zero_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Zero Output Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("moe_zero_output"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            ffn_pipeline,
            zero_pipeline,
            bind_group_layout,
        }
    }

    /// Execute fused MoE FFN forward pass.
    ///
    /// # Arguments
    /// * `input` - Input tensor [num_tokens, hidden_size]
    /// * `expert_indices` - Selected expert indices [num_tokens, top_k]
    /// * `expert_weights` - Routing weights [num_tokens, top_k]
    /// * `gate_weights` - All experts' gate weights [num_experts, intermediate, hidden]
    /// * `up_weights` - All experts' up weights [num_experts, intermediate, hidden]
    /// * `down_weights` - All experts' down weights [num_experts, hidden, intermediate]
    /// * `output` - Output tensor [num_tokens, hidden_size]
    /// * `scratch` - Scratch space [num_tokens * top_k * intermediate * 2]
    /// * `params` - MoE parameters
    pub fn forward(
        &self,
        input: &wgpu::Buffer,
        expert_indices: &wgpu::Buffer,
        expert_weights: &wgpu::Buffer,
        gate_weights: &wgpu::Buffer,
        up_weights: &wgpu::Buffer,
        down_weights: &wgpu::Buffer,
        output: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        params: MoEFfnParams,
    ) {
        // Create params buffer
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MoE FFN Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoE FFN Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: expert_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: expert_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gate_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: up_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: down_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: scratch.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MoE FFN Encoder"),
        });

        // First: zero output buffer
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MoE Zero Output Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.zero_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = params.num_tokens * params.hidden_size;
            let workgroups = (total + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Second: fused MoE FFN forward
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MoE FFN Forward Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ffn_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch: one workgroup per (token, expert_slot) pair
            pass.dispatch_workgroups(params.num_tokens, params.top_k, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
