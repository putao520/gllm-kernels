use wgpu::util::DeviceExt;

#[derive(Debug)]
pub enum MatmulError {
    Wgpu(String),
}

impl std::fmt::Display for MatmulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulError::Wgpu(msg) => write!(f, "WGPU error: {}", msg),
        }
    }
}

impl std::error::Error for MatmulError {}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatmulParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub trans_a: u32,
    pub trans_b: u32,
    pub _pad0: [u32; 3],
    pub alpha: f32,
    pub beta: f32,
    pub _pad1: [f32; 2],
}

pub struct WgpuMatmul {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    workgroup_x: u32,
    workgroup_y: u32,
}

impl WgpuMatmul {
    const DEFAULT_WORKGROUP_X: u32 = 16;
    const DEFAULT_WORKGROUP_Y: u32 = 16;

    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self::new_with_workgroup(
            device,
            queue,
            Self::DEFAULT_WORKGROUP_X,
            Self::DEFAULT_WORKGROUP_Y,
        )
    }

    pub fn new_with_workgroup(
        device: wgpu::Device,
        queue: wgpu::Queue,
        workgroup_x: u32,
        workgroup_y: u32,
    ) -> Self {
        let workgroup_x = workgroup_x.max(1);
        let workgroup_y = workgroup_y.max(1);

        let shader_src = include_str!("kernels/matmul.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matmul Bind Group Layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let constants = [
            ("WORKGROUP_X", workgroup_x as f64),
            ("WORKGROUP_Y", workgroup_y as f64),
        ];

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("matmul_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: true,
            },
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            workgroup_x,
            workgroup_y,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn forward(
        &self,
        params: MatmulParams,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
    ) {
        if params.m == 0 || params.n == 0 {
            return;
        }

        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matmul Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = params
                .m
                .saturating_add(self.workgroup_x - 1)
                / self.workgroup_x;
            let workgroups_y = params
                .n
                .saturating_add(self.workgroup_y - 1)
                / self.workgroup_y;
            if workgroups_x > 0 && workgroups_y > 0 {
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
    }
}
