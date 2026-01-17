//! EvicPress Joint Compression and Eviction WGPU Implementation
//!
//! Based on KVPress (EMNLP'24): Three-zone progressive KV cache management
//! with importance-based eviction and progressive quantization.

use std::sync::mpsc;
use wgpu::util::DeviceExt;

/// WGSL shader source embedded at compile time (Fat Binary)
const SHADER_SOURCE: &str = include_str!("kernels/evic_press.wgsl");

/// Workgroup size matching WGSL kernel
const WORKGROUP_SIZE: u32 = 256;

/// Parameters for importance score computation
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImportanceParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub seq_len: u32,
    pub head_dim: u32,
    pub attention_weight: f32,
    pub semantic_weight: f32,
    pub recency_weight: f32,
    pub _pad0: u32,
}

/// Parameters for zone transition operations
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZoneTransitionParams {
    pub num_elements: u32,
    pub group_size: u32,
    pub num_groups: u32,
    pub _pad0: u32,
}

/// Token importance result
#[derive(Debug, Clone)]
pub struct TokenImportance {
    pub scores: Vec<f32>,
    pub batch_size: u32,
    pub seq_len: u32,
}

/// EvicPress implementation for WGPU
pub struct WgpuEvicPress {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // F32 pipelines
    importance_pipeline_f32: wgpu::ComputePipeline,
    quant_fp16_to_int8_pipeline_f32: wgpu::ComputePipeline,
    quant_int8_to_int2_pipeline_f32: wgpu::ComputePipeline,
    dequant_int8_pipeline_f32: wgpu::ComputePipeline,
    dequant_int2_pipeline_f32: wgpu::ComputePipeline,
    // F16 pipelines (optional)
    importance_pipeline_f16: Option<wgpu::ComputePipeline>,
    quant_fp16_to_int8_pipeline_f16: Option<wgpu::ComputePipeline>,
    // Bind group layouts
    importance_layout: wgpu::BindGroupLayout,
    zone_hot_to_warm_layout: wgpu::BindGroupLayout,
    zone_warm_to_cold_layout: wgpu::BindGroupLayout,
    dequant_layout: wgpu::BindGroupLayout,
}

impl WgpuEvicPress {
    /// Create a new EvicPress instance with the given WGPU device and queue
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("evic_press_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        // Importance layout
        let importance_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("importance_layout"),
            entries: &[
                // Attention scores
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
                // Positions
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
                // Output importance
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

        // Hot→Warm zone transition layout
        let zone_hot_to_warm_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("zone_hot_to_warm_layout"),
                entries: &[
                    // Input (FP16/FP32)
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
                    // Output (INT8)
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

        // Warm→Cold zone transition layout
        let zone_warm_to_cold_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("zone_warm_to_cold_layout"),
                entries: &[
                    // Input (INT8)
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
                    // Output (INT2)
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
                    // Scales (output)
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
                    // Warm scales (input)
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

        // Dequantize layout
        let dequant_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dequant_layout"),
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
                // Output
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
        let importance_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("importance_pipeline_layout"),
                bind_group_layouts: &[&importance_layout],
                push_constant_ranges: &[],
            });

        let hot_to_warm_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hot_to_warm_pipeline_layout"),
                bind_group_layouts: &[&zone_hot_to_warm_layout],
                push_constant_ranges: &[],
            });

        let warm_to_cold_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("warm_to_cold_pipeline_layout"),
                bind_group_layouts: &[&zone_warm_to_cold_layout],
                push_constant_ranges: &[],
            });

        let dequant_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dequant_pipeline_layout"),
                bind_group_layouts: &[&dequant_layout],
                push_constant_ranges: &[],
            });

        // Create F32 pipelines
        let importance_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("importance_f32_pipeline"),
                layout: Some(&importance_pipeline_layout),
                module: &shader_module,
                entry_point: Some("evicpress_compute_importance_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let quant_fp16_to_int8_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quant_fp16_to_int8_f32_pipeline"),
                layout: Some(&hot_to_warm_pipeline_layout),
                module: &shader_module,
                entry_point: Some("evicpress_quantize_fp16_to_int8_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let quant_int8_to_int2_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quant_int8_to_int2_f32_pipeline"),
                layout: Some(&warm_to_cold_pipeline_layout),
                module: &shader_module,
                entry_point: Some("evicpress_quantize_int8_to_int2_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let dequant_int8_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dequant_int8_f32_pipeline"),
                layout: Some(&dequant_pipeline_layout),
                module: &shader_module,
                entry_point: Some("evicpress_dequantize_int8_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        let dequant_int2_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dequant_int2_f32_pipeline"),
                layout: Some(&dequant_pipeline_layout),
                module: &shader_module,
                entry_point: Some("evicpress_dequantize_int2_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create F16 pipelines if supported
        let (importance_pipeline_f16, quant_fp16_to_int8_pipeline_f16) = if has_f16 {
            let imp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("importance_f16_pipeline"),
                layout: Some(&importance_pipeline_layout),
                module: &shader_module,
                entry_point: Some("evicpress_compute_importance_f16"),
                compilation_options: Default::default(),
                cache: None,
            });

            let quant = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quant_fp16_to_int8_f16_pipeline"),
                layout: Some(&hot_to_warm_pipeline_layout),
                module: &shader_module,
                entry_point: Some("evicpress_quantize_fp16_to_int8_f16"),
                compilation_options: Default::default(),
                cache: None,
            });

            (Some(imp), Some(quant))
        } else {
            (None, None)
        };

        Self {
            device,
            queue,
            importance_pipeline_f32,
            quant_fp16_to_int8_pipeline_f32,
            quant_int8_to_int2_pipeline_f32,
            dequant_int8_pipeline_f32,
            dequant_int2_pipeline_f32,
            importance_pipeline_f16,
            quant_fp16_to_int8_pipeline_f16,
            importance_layout,
            zone_hot_to_warm_layout,
            zone_warm_to_cold_layout,
            dequant_layout,
        }
    }

    /// Compute importance scores for all tokens
    pub fn compute_importance_f32(
        &self,
        attention_scores: &[f32],
        positions: &[u32],
        batch_size: u32,
        num_heads: u32,
        seq_len: u32,
        head_dim: u32,
        weights: (f32, f32, f32), // (attention, semantic, recency)
    ) -> TokenImportance {
        let params = ImportanceParams {
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            attention_weight: weights.0,
            semantic_weight: weights.1,
            recency_weight: weights.2,
            _pad0: 0,
        };

        let attention_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("attention_buffer"),
                contents: bytemuck::cast_slice(attention_scores),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let positions_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("positions_buffer"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (batch_size * seq_len) as usize;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("importance_output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("importance_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("importance_bind_group"),
            layout: &self.importance_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: attention_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("importance_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("importance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.importance_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = batch_size * seq_len;
            let workgroups = (total + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("importance_staging"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let scores = self.read_buffer_f32(&staging, output_size);

        TokenImportance {
            scores,
            batch_size,
            seq_len,
        }
    }

    /// Transition from Hot zone (FP16/FP32) to Warm zone (INT8)
    pub fn hot_to_warm_f32(&self, hot_data: &[f32], group_size: u32) -> (Vec<i32>, Vec<f32>) {
        let num_elements = hot_data.len() as u32;
        let num_groups = (num_elements + group_size - 1) / group_size;

        let params = ZoneTransitionParams {
            num_elements,
            group_size,
            num_groups,
            _pad0: 0,
        };

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hot_input"),
                contents: bytemuck::cast_slice(hot_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warm_output"),
            size: (num_elements as usize * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scales_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warm_scales"),
            size: (num_groups as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hot_to_warm_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hot_to_warm_bind_group"),
            layout: &self.zone_hot_to_warm_layout,
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
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hot_to_warm_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hot_to_warm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.quant_fp16_to_int8_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_groups, 1, 1);
        }

        let output_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warm_output_staging"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scales_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warm_scales_staging"),
            size: scales_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &output_staging, 0, output_buffer.size());
        encoder.copy_buffer_to_buffer(&scales_buffer, 0, &scales_staging, 0, scales_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let quantized = self.read_buffer_i32(&output_staging, num_elements as usize);
        let scales = self.read_buffer_f32(&scales_staging, num_groups as usize);

        (quantized, scales)
    }

    /// Check if F16 operations are available
    pub fn has_f16_support(&self) -> bool {
        self.importance_pipeline_f16.is_some()
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

    fn read_buffer_i32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<i32> {
        let (tx, rx) = mpsc::channel();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<i32> = bytemuck::cast_slice(&data)[..count].to_vec();
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
    fn test_compute_importance() {
        pollster::block_on(async {
            let Some((device, queue)) = create_device_with_f16().await else {
                println!("No GPU with f16 support available, skipping test");
                return;
            };

            let evic_press = WgpuEvicPress::new(device, queue);

            let batch_size = 2;
            let num_heads = 4;
            let seq_len = 8;
            let head_dim = 64;

            // Create test attention scores
            let attention_scores: Vec<f32> =
                (0..(batch_size * num_heads * seq_len)).map(|i| (i as f32) * 0.01).collect();
            let positions: Vec<u32> = (0..batch_size * seq_len)
                .map(|i| (i % seq_len) as u32)
                .collect();

            let importance = evic_press.compute_importance_f32(
                &attention_scores,
                &positions,
                batch_size as u32,
                num_heads as u32,
                seq_len as u32,
                head_dim as u32,
                (0.5, 0.3, 0.2),
            );

            assert_eq!(importance.scores.len(), (batch_size * seq_len) as usize);
        });
    }
}
