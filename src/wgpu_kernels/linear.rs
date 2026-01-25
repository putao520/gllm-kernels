use crate::runtime_detection::BackendType;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub enum LinearError {
    Wgpu(String),
}

impl std::fmt::Display for LinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearError::Wgpu(msg) => write!(f, "WGPU error: {}", msg),
        }
    }
}

impl std::error::Error for LinearError {}

pub struct WgpuLinear {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LinearParams {
    pub in_features: u32,
    pub out_features: u32,
    pub has_bias: u32,
    pub padding: u32,
}

impl WgpuLinear {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader_src = include_str!("kernels/linear.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GEMV Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Linear Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Linear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gemv_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        }
    }

    pub fn forward(
        &self,
        params: LinearParams,
        input_buf: &wgpu::Buffer,
        weight_buf: &wgpu::Buffer,
        bias_buf: Option<&wgpu::Buffer>,
        output_buf: &wgpu::Buffer,
    ) {
        let dummy_bias = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Bias"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bias_to_use = bias_buf.unwrap_or(&dummy_bias);

        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Linear Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Linear Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias_to_use.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Linear Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Linear Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (params.out_features + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
    }

    /// Read data from GPU buffer back to host slice.
    pub fn readback_to_slice(&self, buffer: &wgpu::Buffer, output: &mut [f32]) {
        let size = (output.len() * 4) as wgpu::BufferAddress;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::PollType::Wait);
        if let Ok(Ok(_)) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: &[f32] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, output.len())
            };
            output.copy_from_slice(result);
            drop(data);
            staging.unmap();
        }
    }
}
