use std::sync::mpsc;

use half::f16;
use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("quantized.wgsl");
const WORKGROUP_SIZE: u32 = 256;
const Q4_BLOCK_SIZE: usize = 32;
const Q4_PACKED_BYTES: usize = 16;

#[derive(Debug)]
pub enum QuantizedDequantError {
    Wgpu(String),
    Invalid(String),
}

impl std::fmt::Display for QuantizedDequantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wgpu(msg) => write!(f, "WGPU error: {msg}"),
            Self::Invalid(msg) => write!(f, "Invalid config: {msg}"),
        }
    }
}

impl std::error::Error for QuantizedDequantError {}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Q4DequantParams {
    pub num_values: u32,
    pub num_blocks: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AwqDequantParams {
    pub n: u32,
    pub k: u32,
    pub group_size: u32,
    pub groups: u32,
    pub num_values: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub struct WgpuQuantizedDequantizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    q4_pipeline: wgpu::ComputePipeline,
    awq_pipeline: wgpu::ComputePipeline,
    q4_layout: wgpu::BindGroupLayout,
    awq_layout: wgpu::BindGroupLayout,
}

impl WgpuQuantizedDequantizer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quantized_dequant_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let q4_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("q4_dequant_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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

        let awq_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("awq_dequant_layout"),
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
                    binding: 8,
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

        let q4_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("q4_dequant_pipeline_layout"),
            bind_group_layouts: &[&q4_layout],
            push_constant_ranges: &[],
        });

        let awq_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("awq_dequant_pipeline_layout"),
            bind_group_layouts: &[&awq_layout],
            push_constant_ranges: &[],
        });

        let q4_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q4_dequant_pipeline"),
            layout: Some(&q4_pipeline_layout),
            module: &shader,
            entry_point: Some("q4_0_dequantize"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let awq_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("awq_dequant_pipeline"),
            layout: Some(&awq_pipeline_layout),
            module: &shader,
            entry_point: Some("awq_dequantize"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            device,
            queue,
            q4_pipeline,
            awq_pipeline,
            q4_layout,
            awq_layout,
        }
    }

    pub fn dequantize_q4(
        &self,
        q_weight: &[u8],
        scales: &[f16],
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, QuantizedDequantError> {
        let (num_blocks, total_values) = validate_q4_layout(q_weight, scales, n, k)?;
        let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();

        let q_weight_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4_qweight"),
            contents: q_weight,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let scales_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4_scales"),
            contents: bytemuck::cast_slice(&scales_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (total_values * std::mem::size_of::<f32>()) as u64;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q4_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Q4DequantParams {
            num_values: total_values as u32,
            num_blocks: num_blocks as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q4_dequant_bind_group"),
            layout: &self.q4_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: q_weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scales_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4_dequant_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q4_dequant_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.q4_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (params.num_values + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q4_dequant_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        Ok(self.read_buffer_f32(&staging, total_values))
    }

    pub fn dequantize_awq(
        &self,
        qweight: &[u32],
        qzeros: &[u32],
        scales: &[f16],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<Vec<f32>, QuantizedDequantError> {
        let (groups, total_values) = validate_awq_layout(qweight, qzeros, scales, n, k, group_size)?;
        let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();

        let qweight_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("awq_qweight"),
            contents: bytemuck::cast_slice(qweight),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let qzeros_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("awq_qzeros"),
            contents: bytemuck::cast_slice(qzeros),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let scales_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("awq_scales"),
            contents: bytemuck::cast_slice(&scales_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (total_values * std::mem::size_of::<f32>()) as u64;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("awq_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = AwqDequantParams {
            n: n as u32,
            k: k as u32,
            group_size: group_size as u32,
            groups: groups as u32,
            num_values: total_values as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("awq_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("awq_dequant_bind_group"),
            layout: &self.awq_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: qweight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qzeros_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scales_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("awq_dequant_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awq_dequant_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.awq_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (params.num_values + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("awq_dequant_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        Ok(self.read_buffer_f32(&staging, total_values))
    }

    fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let (tx, rx) = mpsc::channel();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        buffer.unmap();
        result
    }
}

fn checked_mul(a: usize, b: usize, name: &str) -> Result<usize, QuantizedDequantError> {
    a.checked_mul(b)
        .ok_or_else(|| QuantizedDequantError::Invalid(format!("{name} overflow")))
}

fn validate_q4_layout(
    q_weight: &[u8],
    scales: &[f16],
    n: usize,
    k: usize,
) -> Result<(usize, usize), QuantizedDequantError> {
    if n == 0 || k == 0 {
        return Err(QuantizedDequantError::Invalid("Dimensions must be > 0".into()));
    }
    if k % Q4_BLOCK_SIZE != 0 {
        return Err(QuantizedDequantError::Invalid(
            "k must be multiple of 32 for Q4".into(),
        ));
    }
    let blocks = k / Q4_BLOCK_SIZE;
    let num_blocks = checked_mul(n, blocks, "q4 blocks")?;
    let expected_q_weight = checked_mul(num_blocks, Q4_PACKED_BYTES, "q4 weights")?;
    if q_weight.len() != expected_q_weight {
        return Err(QuantizedDequantError::Invalid(format!(
            "q_weight length mismatch: expected {expected_q_weight}, got {}",
            q_weight.len()
        )));
    }
    if scales.len() != num_blocks {
        return Err(QuantizedDequantError::Invalid(format!(
            "scales length mismatch: expected {num_blocks}, got {}",
            scales.len()
        )));
    }
    if q_weight.len() % 4 != 0 {
        return Err(QuantizedDequantError::Invalid(
            "q_weight length must be multiple of 4".into(),
        ));
    }
    let total_values = checked_mul(n, k, "output")?;
    if num_blocks > u32::MAX as usize || total_values > u32::MAX as usize {
        return Err(QuantizedDequantError::Invalid(
            "Q4 dimensions exceed addressable range".into(),
        ));
    }
    Ok((num_blocks, total_values))
}

fn validate_awq_layout(
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[f16],
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<(usize, usize), QuantizedDequantError> {
    if n == 0 || k == 0 {
        return Err(QuantizedDequantError::Invalid("Dimensions must be > 0".into()));
    }
    if group_size == 0 {
        return Err(QuantizedDequantError::Invalid(
            "group_size must be > 0".into(),
        ));
    }
    if n % 8 != 0 {
        return Err(QuantizedDequantError::Invalid(
            "n must be multiple of 8 for AWQ packing".into(),
        ));
    }
    if k % group_size != 0 {
        return Err(QuantizedDequantError::Invalid(
            "k must be multiple of group_size for AWQ".into(),
        ));
    }
    let groups = k / group_size;
    let packed_out = n / 8;
    let expected_qweight = checked_mul(packed_out, k, "qweight")?;
    let expected_qzeros = checked_mul(packed_out, groups, "qzeros")?;
    let expected_scales = checked_mul(n, groups, "scales")?;
    if qweight.len() != expected_qweight {
        return Err(QuantizedDequantError::Invalid(format!(
            "qweight length mismatch: expected {expected_qweight}, got {}",
            qweight.len()
        )));
    }
    if qzeros.len() != expected_qzeros {
        return Err(QuantizedDequantError::Invalid(format!(
            "qzeros length mismatch: expected {expected_qzeros}, got {}",
            qzeros.len()
        )));
    }
    if scales.len() != expected_scales {
        return Err(QuantizedDequantError::Invalid(format!(
            "scales length mismatch: expected {expected_scales}, got {}",
            scales.len()
        )));
    }
    let total_values = checked_mul(n, k, "output")?;
    if groups > u32::MAX as usize || total_values > u32::MAX as usize {
        return Err(QuantizedDequantError::Invalid(
            "AWQ dimensions exceed addressable range".into(),
        ));
    }
    Ok((groups, total_values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quantized::{awq_dequantize_cpu, q4_dequantize_cpu};

    async fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;
        Some((device, queue))
    }

    #[test]
    fn q4_dequantize_matches_cpu() {
        pollster::block_on(async {
            let Some((device, queue)) = create_device().await else {
                println!("No WGPU device available, skipping Q4 test");
                return;
            };
            let kernel = WgpuQuantizedDequantizer::new(device, queue);
            let n = 1;
            let k = 32;
            let scales = vec![f16::from_f32(1.0); n * (k / 32)];
            let q_weight = vec![0x99u8; n * (k / 32) * Q4_PACKED_BYTES];
            let cpu = q4_dequantize_cpu(&q_weight, &scales, n, k).unwrap();
            let gpu = kernel.dequantize_q4(&q_weight, &scales, n, k).unwrap();
            assert_eq!(cpu.len(), gpu.len());
            for (a, b) in cpu.iter().zip(gpu.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        });
    }

    #[test]
    fn awq_dequantize_matches_cpu() {
        pollster::block_on(async {
            let Some((device, queue)) = create_device().await else {
                println!("No WGPU device available, skipping AWQ test");
                return;
            };
            let kernel = WgpuQuantizedDequantizer::new(device, queue);
            let n = 8;
            let k = 8;
            let group_size = 4;
            let qweight = vec![0x2222_2222u32; (n / 8) * k];
            let qzeros = vec![0x1111_1111u32; (n / 8) * (k / group_size)];
            let scales = vec![f16::from_f32(1.0); n * (k / group_size)];
            let cpu = awq_dequantize_cpu(&qweight, &qzeros, &scales, n, k, group_size).unwrap();
            let gpu = kernel
                .dequantize_awq(&qweight, &qzeros, &scales, n, k, group_size)
                .unwrap();
            assert_eq!(cpu.len(), gpu.len());
            for (a, b) in cpu.iter().zip(gpu.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        });
    }
}
