// Prompt Caching / CacheBlend WGPU Implementation
// Based on SGLang RadixAttention, vLLM Prefix Caching, CacheBlend
//
// Kernels:
// - Hash computation for token sequences (xxHash64-style)
// - Prefix match finding in cache
// - CacheBlend position reencoding
// - KV cache segment copying
// - Rolling hash for incremental updates

use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("kernels/prompt_cache.wgsl");

/// Parameters for hash computation
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HashParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub hash_dim: u32,  // Number of tokens per hash block
    pub _pad0: u32,
}

/// Parameters for prefix matching
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrefixMatchParams {
    pub batch_size: u32,
    pub query_len: u32,
    pub cache_entries: u32,
    pub max_prefix_len: u32,
}

/// Parameters for CacheBlend
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlendParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub cached_len: u32,
    pub new_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Parameters for KV cache copy
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CopyKVParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub src_seq_len: u32,
    pub dst_offset: u32,
    pub copy_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Parameters for rolling hash update
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RollingHashParams {
    pub batch_size: u32,
    pub window_size: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Result of prefix match operation
#[derive(Debug, Clone)]
pub struct PrefixMatchResult {
    /// Cache entry index for each batch item (0xFFFFFFFF if no match)
    pub match_entries: Vec<u32>,
    /// Match length in hash blocks for each batch item
    pub match_lengths: Vec<u32>,
}

/// WGPU Prompt Cache implementation
pub struct WgpuPromptCache {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Compute pipelines
    hash_pipeline: wgpu::ComputePipeline,
    prefix_match_pipeline: wgpu::ComputePipeline,
    blend_f32_pipeline: wgpu::ComputePipeline,
    copy_kv_f32_pipeline: wgpu::ComputePipeline,
    rolling_hash_pipeline: wgpu::ComputePipeline,

    // F16 pipelines (optional)
    blend_f16_pipeline: Option<wgpu::ComputePipeline>,
    copy_kv_f16_pipeline: Option<wgpu::ComputePipeline>,

    // Bind group layouts
    hash_layout: wgpu::BindGroupLayout,
    prefix_match_layout: wgpu::BindGroupLayout,
    blend_layout: wgpu::BindGroupLayout,
    copy_kv_layout: wgpu::BindGroupLayout,
    rolling_hash_layout: wgpu::BindGroupLayout,
}

impl WgpuPromptCache {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prompt_cache_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Hash layout: tokens (input), hashes (output), params
        let hash_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hash_layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Prefix match layout: query_hashes, cache_hashes, match_entry, match_len, params
        let prefix_match_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix_match_layout"),
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

        // Blend layout: cached_kv, new_kv, output, params
        let blend_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blend_layout"),
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

        // Copy KV layout: src, dst, params
        let copy_kv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy_kv_layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Rolling hash layout: old_hash, old_token, new_token, new_hash, params
        let rolling_hash_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rolling_hash_layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layouts
        let hash_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hash_pipeline_layout"),
            bind_group_layouts: &[&hash_layout],
            push_constant_ranges: &[],
        });

        let prefix_match_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefix_match_pipeline_layout"),
            bind_group_layouts: &[&prefix_match_layout],
            push_constant_ranges: &[],
        });

        let blend_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blend_pipeline_layout"),
            bind_group_layouts: &[&blend_layout],
            push_constant_ranges: &[],
        });

        let copy_kv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("copy_kv_pipeline_layout"),
            bind_group_layouts: &[&copy_kv_layout],
            push_constant_ranges: &[],
        });

        let rolling_hash_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rolling_hash_pipeline_layout"),
            bind_group_layouts: &[&rolling_hash_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let hash_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prompt_cache_hash_f32"),
            layout: Some(&hash_pipeline_layout),
            module: &shader_module,
            entry_point: Some("prompt_cache_hash_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let prefix_match_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prompt_cache_prefix_match_f32"),
            layout: Some(&prefix_match_pipeline_layout),
            module: &shader_module,
            entry_point: Some("prompt_cache_prefix_match_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blend_f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prompt_cache_blend_f32"),
            layout: Some(&blend_pipeline_layout),
            module: &shader_module,
            entry_point: Some("prompt_cache_blend_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let copy_kv_f32_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prompt_cache_copy_kv_f32"),
            layout: Some(&copy_kv_pipeline_layout),
            module: &shader_module,
            entry_point: Some("prompt_cache_copy_kv_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rolling_hash_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prompt_cache_rolling_hash_f32"),
            layout: Some(&rolling_hash_pipeline_layout),
            module: &shader_module,
            entry_point: Some("prompt_cache_rolling_hash_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        // F16 pipelines (conditional)
        let blend_f16_pipeline = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prompt_cache_blend_f16"),
                layout: Some(&blend_pipeline_layout),
                module: &shader_module,
                entry_point: Some("prompt_cache_blend_f16"),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        let copy_kv_f16_pipeline = if has_f16 {
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prompt_cache_copy_kv_f16"),
                layout: Some(&copy_kv_pipeline_layout),
                module: &shader_module,
                entry_point: Some("prompt_cache_copy_kv_f16"),
                compilation_options: Default::default(),
                cache: None,
            }))
        } else {
            None
        };

        Self {
            device,
            queue,
            hash_pipeline,
            prefix_match_pipeline,
            blend_f32_pipeline,
            copy_kv_f32_pipeline,
            rolling_hash_pipeline,
            blend_f16_pipeline,
            copy_kv_f16_pipeline,
            hash_layout,
            prefix_match_layout,
            blend_layout,
            copy_kv_layout,
            rolling_hash_layout,
        }
    }

    /// Check if F16 kernels are available
    pub fn has_f16_support(&self) -> bool {
        self.blend_f16_pipeline.is_some()
    }

    /// Compute hash for token sequences
    /// Returns hash values for each block of tokens
    pub fn compute_hash(
        &self,
        tokens: &[u32],
        batch_size: u32,
        seq_len: u32,
        hash_dim: u32,
    ) -> Vec<u32> {
        let num_blocks = seq_len / hash_dim;
        let output_size = (batch_size * num_blocks) as usize;

        let params = HashParams {
            batch_size,
            seq_len,
            hash_dim,
            _pad0: 0,
        };

        let tokens_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tokens_buffer"),
            contents: bytemuck::cast_slice(tokens),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash_output_buffer"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hash_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hash_bind_group"),
            layout: &self.hash_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tokens_buffer.as_entire_binding(),
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

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hash_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hash_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((output_size as u32 + 255) / 256, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash_staging"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (output_size * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = staging_buffer.slice(..).get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Find longest prefix match in cache
    pub fn find_prefix_match(
        &self,
        query_hashes: &[u32],
        cache_hashes: &[u32],
        batch_size: u32,
        query_len: u32,
        cache_entries: u32,
        max_prefix_len: u32,
    ) -> PrefixMatchResult {
        let params = PrefixMatchParams {
            batch_size,
            query_len,
            cache_entries,
            max_prefix_len,
        };

        let query_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("query_hashes_buffer"),
            contents: bytemuck::cast_slice(query_hashes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cache_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cache_hashes_buffer"),
            contents: bytemuck::cast_slice(cache_hashes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let entry_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_entry_buffer"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let len_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_len_buffer"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefix_match_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_match_bind_group"),
            layout: &self.prefix_match_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cache_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: entry_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: len_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prefix_match_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_match_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prefix_match_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((batch_size + 255) / 256, 1, 1);
        }

        let entry_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entry_staging"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let len_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("len_staging"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &entry_buffer,
            0,
            &entry_staging,
            0,
            (batch_size as usize * std::mem::size_of::<u32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &len_buffer,
            0,
            &len_staging,
            0,
            (batch_size as usize * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        entry_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        len_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();

        let entry_data = entry_staging.slice(..).get_mapped_range();
        let match_entries: Vec<u32> = bytemuck::cast_slice(&entry_data).to_vec();
        drop(entry_data);
        entry_staging.unmap();

        let len_data = len_staging.slice(..).get_mapped_range();
        let match_lengths: Vec<u32> = bytemuck::cast_slice(&len_data).to_vec();
        drop(len_data);
        len_staging.unmap();

        PrefixMatchResult {
            match_entries,
            match_lengths,
        }
    }

    /// Blend cached and new KV cache (F32)
    pub fn blend_kv_f32(
        &self,
        cached_kv: &[f32],
        new_kv: &[f32],
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        cached_len: u32,
        new_len: u32,
    ) -> Vec<f32> {
        let total_len = cached_len + new_len;
        let kv_size = num_heads * head_dim;
        let output_size = (batch_size * total_len * kv_size) as usize;

        let params = BlendParams {
            batch_size,
            num_heads,
            head_dim,
            cached_len,
            new_len,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let cached_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cached_kv_buffer"),
            contents: bytemuck::cast_slice(cached_kv),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let new_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("new_kv_buffer"),
            contents: bytemuck::cast_slice(new_kv),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blend_output_buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blend_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blend_bind_group"),
            layout: &self.blend_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cached_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: new_buffer.as_entire_binding(),
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
            label: Some("blend_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blend_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blend_f32_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((output_size as u32 + 255) / 256, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blend_staging"),
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

    /// Copy KV cache segment (F32)
    pub fn copy_kv_f32(
        &self,
        src_kv: &[f32],
        dst_kv: &mut [f32],
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        src_seq_len: u32,
        dst_offset: u32,
        copy_len: u32,
    ) {
        let kv_size = num_heads * head_dim;
        let copy_size = (batch_size * copy_len * kv_size) as usize;

        let params = CopyKVParams {
            batch_size,
            num_heads,
            head_dim,
            src_seq_len,
            dst_offset,
            copy_len,
            _pad0: 0,
            _pad1: 0,
        };

        let src_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("copy_src_buffer"),
            contents: bytemuck::cast_slice(src_kv),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dst_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("copy_dst_buffer"),
            contents: bytemuck::cast_slice(dst_kv),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("copy_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_bind_group"),
            layout: &self.copy_kv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("copy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_kv_f32_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((copy_size as u32 + 255) / 256, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("copy_staging"),
            size: (dst_kv.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &dst_buffer,
            0,
            &staging_buffer,
            0,
            (dst_kv.len() * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = staging_buffer.slice(..).get_mapped_range();
        dst_kv.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging_buffer.unmap();
    }

    /// Update rolling hash incrementally
    pub fn update_rolling_hash(
        &self,
        old_hashes: &[u32],
        old_tokens: &[u32],
        new_tokens: &[u32],
        batch_size: u32,
        window_size: u32,
    ) -> Vec<u32> {
        let params = RollingHashParams {
            batch_size,
            window_size,
            _pad0: 0,
            _pad1: 0,
        };

        let old_hash_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("old_hash_buffer"),
            contents: bytemuck::cast_slice(old_hashes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let old_token_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("old_token_buffer"),
            contents: bytemuck::cast_slice(old_tokens),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let new_token_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("new_token_buffer"),
            contents: bytemuck::cast_slice(new_tokens),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let new_hash_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("new_hash_buffer"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rolling_hash_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rolling_hash_bind_group"),
            layout: &self.rolling_hash_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: old_hash_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: old_token_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_token_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: new_hash_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rolling_hash_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rolling_hash_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rolling_hash_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((batch_size + 255) / 256, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rolling_hash_staging"),
            size: (batch_size as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &new_hash_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size as usize * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        let data = staging_buffer.slice(..).get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }
}
