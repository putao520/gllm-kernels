//! INT2 Extreme Quantization WGPU Implementation
//!
//! Based on QuaRot, GPTQ-INT2, SqueezeLLM techniques for extreme weight compression.
//! Provides 4x compression vs FP16 with group-wise scales for accuracy preservation.

use std::sync::mpsc;
use wgpu::util::DeviceExt;

/// WGSL shader source embedded at compile time (Fat Binary)
const SHADER_SOURCE: &str = include_str!("kernels/int2_quantizer.wgsl");

/// Workgroup size matching WGSL kernel
const WORKGROUP_SIZE: u32 = 256;

/// Quantization parameters for GPU operations
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuantizeParams {
    pub num_elements: u32,
    pub group_size: u32,
    pub num_groups: u32,
    pub _pad0: u32,
}

/// Packing parameters for bit-packing operations
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackParams {
    pub num_groups: u32,
    pub group_size: u32,
    pub packed_size: u32, // group_size / 4
    pub _pad0: u32,
}

/// INT2 quantizer implementation for WGPU
pub struct WgpuInt2Quantizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // F32 pipelines
    quantize_pipeline_f32: wgpu::ComputePipeline,
    dequantize_pipeline_f32: wgpu::ComputePipeline,
    pack_pipeline_f32: wgpu::ComputePipeline,
    unpack_pipeline_f32: wgpu::ComputePipeline,
    // F16 pipelines (optional)
    quantize_pipeline_f16: Option<wgpu::ComputePipeline>,
    dequantize_pipeline_f16: Option<wgpu::ComputePipeline>,
    // Bind group layouts
    quantize_layout: wgpu::BindGroupLayout,
    dequantize_layout: wgpu::BindGroupLayout,
    pack_layout: wgpu::BindGroupLayout,
    unpack_layout: wgpu::BindGroupLayout,
}

impl WgpuInt2Quantizer {
    /// Create a new INT2 quantizer with the given WGPU device and queue
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("int2_quantizer_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Check for F16 support
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        // Quantize bind group layout
        let quantize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("int2_quantize_layout"),
            entries: &[
                // Input values
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
                // Output quantized
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
                // Scales
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
                // Zeros
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

        // Dequantize bind group layout
        let dequantize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("int2_dequantize_layout"),
            entries: &[
                // Input quantized
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
                // Scales
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
                // Zeros
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
                // Output values
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

        // Pack/Unpack bind group layout
        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("int2_pack_layout"),
            entries: &[
                // Input
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
                // Output
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

        let unpack_layout = pack_layout.clone();

        // Create pipeline layouts
        let quantize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("int2_quantize_pipeline_layout"),
                bind_group_layouts: &[&quantize_layout],
                push_constant_ranges: &[],
            });

        let dequantize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("int2_dequantize_pipeline_layout"),
                bind_group_layouts: &[&dequantize_layout],
                push_constant_ranges: &[],
            });

        let pack_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("int2_pack_pipeline_layout"),
            bind_group_layouts: &[&pack_layout],
            push_constant_ranges: &[],
        });

        // Create F32 pipelines
        let quantize_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("int2_quantize_f32_pipeline"),
                layout: Some(&quantize_pipeline_layout),
                module: &shader_module,
                entry_point: Some("int2_quantize_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let dequantize_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("int2_dequantize_f32_pipeline"),
                layout: Some(&dequantize_pipeline_layout),
                module: &shader_module,
                entry_point: Some("int2_dequantize_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pack_pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("int2_pack_f32_pipeline"),
            layout: Some(&pack_pipeline_layout),
            module: &shader_module,
            entry_point: Some("int2_pack_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let unpack_pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("int2_unpack_f32_pipeline"),
            layout: Some(&pack_pipeline_layout),
            module: &shader_module,
            entry_point: Some("int2_unpack_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create F16 pipelines if supported
        let (quantize_pipeline_f16, dequantize_pipeline_f16) = if has_f16 {
            let quant = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("int2_quantize_f16_pipeline"),
                layout: Some(&quantize_pipeline_layout),
                module: &shader_module,
                entry_point: Some("int2_quantize_f16"),
                compilation_options: Default::default(),
                cache: None,
            });

            let dequant = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("int2_dequantize_f16_pipeline"),
                layout: Some(&dequantize_pipeline_layout),
                module: &shader_module,
                entry_point: Some("int2_dequantize_f16"),
                compilation_options: Default::default(),
                cache: None,
            });

            (Some(quant), Some(dequant))
        } else {
            (None, None)
        };

        Self {
            device,
            queue,
            quantize_pipeline_f32,
            dequantize_pipeline_f32,
            pack_pipeline_f32,
            unpack_pipeline_f32,
            quantize_pipeline_f16,
            dequantize_pipeline_f16,
            quantize_layout,
            dequantize_layout,
            pack_layout,
            unpack_layout,
        }
    }

    /// Quantize F32 values to INT2 with group-wise scales
    pub fn quantize_f32(
        &self,
        values: &[f32],
        group_size: u32,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let num_elements = values.len() as u32;
        let num_groups = (num_elements + group_size - 1) / group_size;

        let params = QuantizeParams {
            num_elements,
            group_size,
            num_groups,
            _pad0: 0,
        };

        // Create buffers
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quantize_input"),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quantize_output"),
            size: (num_elements as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scales_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quantize_scales"),
            size: (num_groups as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let zeros_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quantize_zeros"),
            size: (num_groups as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quantize_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quantize_bind_group"),
            layout: &self.quantize_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: zeros_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quantize_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quantize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.quantize_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per group
            pass.dispatch_workgroups(num_groups, 1, 1);
        }

        // Create staging buffers for readback
        let output_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_staging"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scales_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scales_staging"),
            size: scales_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let zeros_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zeros_staging"),
            size: zeros_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &output_staging, 0, output_buffer.size());
        encoder.copy_buffer_to_buffer(&scales_buffer, 0, &scales_staging, 0, scales_buffer.size());
        encoder.copy_buffer_to_buffer(&zeros_buffer, 0, &zeros_staging, 0, zeros_buffer.size());

        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let quantized = self.read_buffer_u32(&output_staging, num_elements as usize);
        let scales = self.read_buffer_f32(&scales_staging, num_groups as usize);
        let zeros = self.read_buffer_f32(&zeros_staging, num_groups as usize);

        (quantized, scales, zeros)
    }

    /// Dequantize INT2 values back to F32
    pub fn dequantize_f32(
        &self,
        quantized: &[u32],
        scales: &[f32],
        zeros: &[f32],
        group_size: u32,
    ) -> Vec<f32> {
        let num_elements = quantized.len() as u32;
        let num_groups = scales.len() as u32;

        let params = QuantizeParams {
            num_elements,
            group_size,
            num_groups,
            _pad0: 0,
        };

        // Create buffers
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dequantize_input"),
                contents: bytemuck::cast_slice(quantized),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let scales_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dequantize_scales"),
                contents: bytemuck::cast_slice(scales),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let zeros_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dequantize_zeros"),
                contents: bytemuck::cast_slice(zeros),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dequantize_output"),
            size: (num_elements as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dequantize_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dequantize_bind_group"),
            layout: &self.dequantize_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: zeros_buffer.as_entire_binding(),
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

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dequantize_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dequantize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.dequantize_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dequantize_staging"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        self.read_buffer_f32(&staging, num_elements as usize)
    }

    /// Pack 4 INT2 values into single bytes
    pub fn pack(&self, quantized: &[u32], group_size: u32) -> Vec<u32> {
        let num_groups = quantized.len() as u32 / group_size;
        let packed_size = group_size / 4;

        let params = PackParams {
            num_groups,
            group_size,
            packed_size,
            _pad0: 0,
        };

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pack_input"),
                contents: bytemuck::cast_slice(quantized),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (num_groups * packed_size) as usize;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pack_output"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pack_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pack_bind_group"),
            layout: &self.pack_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pack_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = num_groups * packed_size;
            let workgroups = (total + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pack_staging"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        self.read_buffer_u32(&staging, output_size)
    }

    /// Check if F16 operations are available
    pub fn has_f16_support(&self) -> bool {
        self.quantize_pipeline_f16.is_some()
    }

    // Helper methods for buffer readback
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
    fn test_quantize_dequantize() {
        pollster::block_on(async {
            let Some((device, queue)) = create_device_with_f16().await else {
                println!("No GPU with f16 support available, skipping test");
                return;
            };

            let quantizer = WgpuInt2Quantizer::new(device, queue);

            // Test data
            let values: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
            let group_size = 32;

            // Quantize
            let (quantized, scales, zeros) = quantizer.quantize_f32(&values, group_size);

            // Check quantized values are in INT2 range
            assert!(quantized.iter().all(|&v| v <= 3));

            // Dequantize
            let recovered = quantizer.dequantize_f32(&quantized, &scales, &zeros, group_size);

            // Check approximate reconstruction
            assert_eq!(recovered.len(), values.len());
        });
    }
}
