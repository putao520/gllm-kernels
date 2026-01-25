use wgpu::util::DeviceExt;

#[derive(Debug)]
pub enum RmsNormError {
    Wgpu(String),
}

impl std::fmt::Display for RmsNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RmsNormError::Wgpu(msg) => write!(f, "WGPU error: {}", msg),
        }
    }
}

impl std::error::Error for RmsNormError {}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormParams {
    pub rows: u32,
    pub hidden: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub eps: f32,
    pub _pad2: [f32; 3],
}

pub struct WgpuRmsNorm {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pipeline_inplace: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_layout_inplace: wgpu::BindGroupLayout,
}

impl WgpuRmsNorm {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader_src = include_str!("kernels/rms_norm.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RMSNorm Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RMSNorm Bind Group Layout"),
            entries: &[
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
            ],
        });

        let bind_group_layout_inplace = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RMSNorm Bind Group Layout (Inplace)"),
            entries: &[
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RMSNorm Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_layout_inplace = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RMSNorm Pipeline Layout (Inplace)"),
            bind_group_layouts: &[&bind_group_layout_inplace],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMSNorm Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("rms_norm_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let pipeline_inplace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMSNorm Pipeline (Inplace)"),
            layout: Some(&pipeline_layout_inplace),
            module: &shader,
            entry_point: Some("rms_norm_inplace_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            device,
            queue,
            pipeline,
            pipeline_inplace,
            bind_group_layout,
            bind_group_layout_inplace,
        }
    }

    pub fn forward(
        &self,
        params: RmsNormParams,
        input: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        if params.rows == 0 || params.hidden == 0 {
            return;
        }

        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RMSNorm Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RMSNorm Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RMSNorm Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.rows, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
    }

    pub fn forward_inplace(
        &self,
        params: RmsNormParams,
        data: &wgpu::Buffer,
        weight: &wgpu::Buffer,
    ) {
        if params.rows == 0 || params.hidden == 0 {
            return;
        }

        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm Params (Inplace)"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RMSNorm Bind Group (Inplace)"),
            layout: &self.bind_group_layout_inplace,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RMSNorm Encoder (Inplace)"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RMSNorm Pass (Inplace)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_inplace);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.rows, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
    }
}
