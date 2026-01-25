//! WGPU Tensor Operations for MoE - Pure GPU implementations

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TensorOpsParams {
    pub size: u32,
    pub offset: u32,
    pub scale: f32,
    pub _padding: u32,
}

pub struct WgpuTensorOps {
    device: wgpu::Device,
    queue: wgpu::Queue,
    zero_pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
    slice_pipeline: wgpu::ComputePipeline,
    scale_add_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuTensorOps {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader_src = include_str!("kernels/tensor_ops.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TensorOps Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TensorOps Bind Group Layout"),
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
                // output storage (read_write)
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
                // input storage (read_only)
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
            label: Some("TensorOps Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create all pipelines before moving device into Self
        let zero_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TensorZero Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("tensor_zero"),
            compilation_options: Default::default(),
            cache: None,
        });

        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TensorAdd Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("tensor_add"),
            compilation_options: Default::default(),
            cache: None,
        });

        let slice_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TensorSlice Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("tensor_slice"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scale_add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TensorScaleAdd Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("tensor_scale_add"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            zero_pipeline,
            add_pipeline,
            slice_pipeline,
            scale_add_pipeline,
            bind_group_layout,
        }
    }

    fn create_bind_group(&self, params_buf: &wgpu::Buffer, output_buf: &wgpu::Buffer, input_buf: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TensorOps BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: input_buf.as_entire_binding() },
            ],
        })
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, size: u32) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TensorOps Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TensorOps Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let workgroups = (size + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Zero tensor: output[i] = 0
    pub fn tensor_zero(&self, output_buf: &wgpu::Buffer, size: usize) {
        let params = TensorOpsParams {
            size: size as u32,
            offset: 0,
            scale: 0.0,
            _padding: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorZero Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // For zero, we don't need input, but binding requires it - use output as dummy
        let bind_group = self.create_bind_group(&params_buf, output_buf, output_buf);
        self.dispatch(&self.zero_pipeline, &bind_group, size as u32);
    }

    /// Add tensors: output[i] += input[i]
    pub fn tensor_add(&self, output_buf: &wgpu::Buffer, input_buf: &wgpu::Buffer, size: usize) {
        let params = TensorOpsParams {
            size: size as u32,
            offset: 0,
            scale: 0.0,
            _padding: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorAdd Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.create_bind_group(&params_buf, output_buf, input_buf);
        self.dispatch(&self.add_pipeline, &bind_group, size as u32);
    }

    /// Slice tensor: output[i] = input[offset + i]
    pub fn tensor_slice(&self, output_buf: &wgpu::Buffer, input_buf: &wgpu::Buffer, offset: usize, len: usize) {
        let params = TensorOpsParams {
            size: len as u32,
            offset: offset as u32,
            scale: 0.0,
            _padding: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorSlice Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.create_bind_group(&params_buf, output_buf, input_buf);
        self.dispatch(&self.slice_pipeline, &bind_group, len as u32);
    }

    /// Scale and add: output[offset + i] += input[i] * scale
    pub fn tensor_scale_add(&self, output_buf: &wgpu::Buffer, input_buf: &wgpu::Buffer, offset: usize, input_len: usize, scale: f32) {
        let params = TensorOpsParams {
            size: input_len as u32,
            offset: offset as u32,
            scale,
            _padding: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TensorScaleAdd Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.create_bind_group(&params_buf, output_buf, input_buf);
        self.dispatch(&self.scale_add_pipeline, &bind_group, input_len as u32);
    }
}
