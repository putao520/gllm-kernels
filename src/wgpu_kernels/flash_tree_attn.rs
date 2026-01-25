//! DeFT / Talon Flash Tree-Attention WGPU Implementation
//!
//! Based on DeFT (ICLR'25) and Talon (ICLR'26)
//! Tree-structured attention with O(n+m) complexity for speculative decoding
//!
//! Kernels:
//! - tree_attn_build_mask: Build tree mask from parent indices
//! - tree_attn_forward: Tree-structured attention forward pass
//! - tree_attn_verify: Verify draft tokens against ground truth

use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// WGSL shader source (embedded at compile time)
const SHADER_SOURCE: &str = include_str!("kernels/flash_tree_attn.wgsl");

/// Workgroup size (must match WGSL)
const WORKGROUP_SIZE: u32 = 128;

/// Tree attention parameters for forward pass
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeAttnParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub prompt_len: u32,
    pub tree_size: u32,
    pub head_dim: u32,
    pub scale: f32,
    _pad0: u32,
    _pad1: u32,
}

impl TreeAttnParams {
    pub fn new(
        batch_size: u32,
        num_heads: u32,
        prompt_len: u32,
        tree_size: u32,
        head_dim: u32,
        scale: f32,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            prompt_len,
            tree_size,
            head_dim,
            scale,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create with automatic scale (1/sqrt(head_dim))
    pub fn with_auto_scale(
        batch_size: u32,
        num_heads: u32,
        prompt_len: u32,
        tree_size: u32,
        head_dim: u32,
    ) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self::new(batch_size, num_heads, prompt_len, tree_size, head_dim, scale)
    }
}

/// Tree mask building parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeMaskParams {
    pub num_nodes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl TreeMaskParams {
    pub fn new(num_nodes: u32) -> Self {
        Self {
            num_nodes,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Verification parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VerifyParams {
    pub batch_size: u32,
    pub tree_size: u32,
    _pad0: u32,
    _pad1: u32,
}

impl VerifyParams {
    pub fn new(batch_size: u32, tree_size: u32) -> Self {
        Self {
            batch_size,
            tree_size,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Verification result
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Number of accepted tokens per batch
    pub accepted_lengths: Vec<u32>,
    /// Indices of accepted tokens per batch (flattened)
    pub accepted_tokens: Vec<u32>,
}

/// Flash Tree-Attention WGPU kernel wrapper
pub struct FlashTreeAttn {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Mask building pipeline
    mask_pipeline: wgpu::ComputePipeline,
    mask_bind_group_layout: wgpu::BindGroupLayout,
    // Forward attention pipelines
    forward_pipeline_f32: wgpu::ComputePipeline,
    forward_bind_group_layout_f32: wgpu::BindGroupLayout,
    forward_pipeline_f16: Option<wgpu::ComputePipeline>,
    forward_bind_group_layout_f16: Option<wgpu::BindGroupLayout>,
    // Verification pipeline
    verify_pipeline: wgpu::ComputePipeline,
    verify_bind_group_layout: wgpu::BindGroupLayout,
}

impl FlashTreeAttn {
    /// Create a new FlashTreeAttn instance
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find an appropriate adapter: {}", e))?;

        let features = adapter.features();
        let has_f16 = features.contains(wgpu::Features::SHADER_F16);

        let mut required_features = wgpu::Features::empty();
        if has_f16 {
            required_features |= wgpu::Features::SHADER_F16;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("FlashTreeAttn Device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        Self::from_device(device, queue)
    }

    fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, String> {
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FlashTreeAttn Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // Mask building bind group layout
        let mask_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Mask Build Bind Group Layout"),
                entries: &[
                    // parent_indices (input)
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
                    // tree_mask (output)
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
                    // params (uniform)
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

        let mask_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Mask Build Pipeline Layout"),
                bind_group_layouts: &[&mask_bind_group_layout],
                push_constant_ranges: &[],
            });

        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mask Build Pipeline"),
            layout: Some(&mask_pipeline_layout),
            module: &shader_module,
            entry_point: Some("tree_attn_build_mask_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Forward attention F32 bind group layout
        let forward_bind_group_layout_f32 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Forward F32 Bind Group Layout"),
                entries: &[
                    // Q (input)
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
                    // K (input)
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
                    // V (input)
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
                    // tree_mask (input)
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
                    // O (output)
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
                    // params (uniform)
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

        let forward_pipeline_layout_f32 =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Forward F32 Pipeline Layout"),
                bind_group_layouts: &[&forward_bind_group_layout_f32],
                push_constant_ranges: &[],
            });

        let forward_pipeline_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Forward F32 Pipeline"),
                layout: Some(&forward_pipeline_layout_f32),
                module: &shader_module,
                entry_point: Some("tree_attn_forward_f32"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Forward attention F16 (if supported)
        let (forward_pipeline_f16, forward_bind_group_layout_f16) = if has_f16 {
            let layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Forward F16 Bind Group Layout"),
                    entries: &[
                        // Q (input f16)
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
                        // K (input f16)
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
                        // V (input f16)
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
                        // tree_mask (input i32)
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
                        // O (output f16)
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
                        // params (uniform)
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

            let pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Forward F16 Pipeline Layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });

            let pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Forward F16 Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("tree_attn_forward_f16"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            (Some(pipeline), Some(layout))
        } else {
            (None, None)
        };

        // Verification bind group layout
        let verify_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Verify Bind Group Layout"),
                entries: &[
                    // draft_tokens (input)
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
                    // ground_truth (input)
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
                    // parent_indices (input)
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
                    // accepted_length (output)
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
                    // accepted_tokens (output)
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
                    // params (uniform)
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

        let verify_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Verify Pipeline Layout"),
                bind_group_layouts: &[&verify_bind_group_layout],
                push_constant_ranges: &[],
            });

        let verify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Verify Pipeline"),
            layout: Some(&verify_pipeline_layout),
            module: &shader_module,
            entry_point: Some("tree_attn_verify_f32"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            mask_pipeline,
            mask_bind_group_layout,
            forward_pipeline_f32,
            forward_bind_group_layout_f32,
            forward_pipeline_f16,
            forward_bind_group_layout_f16,
            verify_pipeline,
            verify_bind_group_layout,
        })
    }

    /// Create with an existing device/queue.
    pub fn new_with_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, String> {
        Self::from_device(device, queue)
    }

    /// Create synchronously using pollster
    pub fn new_sync() -> Result<Self, String> {
        pollster::block_on(Self::new())
    }

    /// Check if F16 support is available
    pub fn has_f16_support(&self) -> bool {
        self.forward_pipeline_f16.is_some()
    }

    /// Build tree mask from parent indices
    ///
    /// # Arguments
    /// * `parent_indices` - Parent index for each node (-1 for root)
    ///
    /// # Returns
    /// * Tree mask [num_nodes, num_nodes] where mask[j,i]=1 if i is ancestor of j
    pub fn build_mask(&self, parent_indices: &[i32]) -> Vec<i32> {
        let num_nodes = parent_indices.len() as u32;
        let total_mask = (num_nodes * num_nodes) as usize;

        let params = TreeMaskParams::new(num_nodes);

        // Create buffers
        let parents_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parent Indices Buffer"),
                contents: bytemuck::cast_slice(parent_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mask_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tree Mask Buffer"),
            size: (total_mask * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mask Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mask Build Bind Group"),
            layout: &self.mask_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: parents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Mask Build Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mask Build Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (total_mask as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mask Staging Buffer"),
            size: (total_mask * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &mask_buffer,
            0,
            &staging_buffer,
            0,
            (total_mask * std::mem::size_of::<i32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::PollType::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Tree attention forward pass (F32)
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, tree_size, num_heads, head_dim]
    /// * `k` - Key tensor [batch, prompt_len+tree_size, num_heads, head_dim]
    /// * `v` - Value tensor [batch, prompt_len+tree_size, num_heads, head_dim]
    /// * `tree_mask` - Tree attention mask [tree_size, tree_size] as i32
    /// * `params` - Attention parameters
    ///
    /// # Returns
    /// * Output tensor [batch, tree_size, num_heads, head_dim]
    pub fn forward_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tree_mask: &[i32],
        params: &TreeAttnParams,
    ) -> Vec<f32> {
        let output_size = (params.batch_size
            * params.tree_size
            * params.num_heads
            * params.head_dim) as usize;

        // Create buffers
        let q_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Q Buffer"),
                contents: bytemuck::cast_slice(q),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let k_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("K Buffer"),
                contents: bytemuck::cast_slice(k),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("V Buffer"),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mask_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tree Mask Buffer"),
                contents: bytemuck::cast_slice(tree_mask),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let o_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Forward F32 Bind Group"),
            layout: &self.forward_bind_group_layout_f32,
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
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: o_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Forward F32 Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Forward F32 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: x = ceil(tree_size / WORKGROUP_SIZE), y = batch_size * num_heads
            let workgroups_x = (params.tree_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let workgroups_y = params.batch_size * params.num_heads;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &o_buffer,
            0,
            &staging_buffer,
            0,
            (output_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::PollType::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Tree attention forward pass (F16)
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, tree_size, num_heads, head_dim] as f16 bytes
    /// * `k` - Key tensor [batch, prompt_len+tree_size, num_heads, head_dim] as f16 bytes
    /// * `v` - Value tensor [batch, prompt_len+tree_size, num_heads, head_dim] as f16 bytes
    /// * `tree_mask` - Tree attention mask [tree_size, tree_size] as i32
    /// * `params` - Attention parameters
    ///
    /// # Returns
    /// * Output tensor as f16 bytes
    pub fn forward_f16(
        &self,
        q: &[u8],
        k: &[u8],
        v: &[u8],
        tree_mask: &[i32],
        params: &TreeAttnParams,
    ) -> Result<Vec<u8>, String> {
        let pipeline = self
            .forward_pipeline_f16
            .as_ref()
            .ok_or("F16 not supported")?;
        let layout = self
            .forward_bind_group_layout_f16
            .as_ref()
            .ok_or("F16 not supported")?;

        let output_elements = (params.batch_size
            * params.tree_size
            * params.num_heads
            * params.head_dim) as usize;
        let output_size = output_elements * 2; // f16 = 2 bytes

        // Create buffers
        let q_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Q Buffer F16"),
                contents: q,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let k_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("K Buffer F16"),
                contents: k,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("V Buffer F16"),
                contents: v,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mask_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tree Mask Buffer"),
                contents: bytemuck::cast_slice(tree_mask),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let o_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer F16"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Forward F16 Bind Group"),
            layout,
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
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: o_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Forward F16 Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Forward F16 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = (params.tree_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let workgroups_y = params.batch_size * params.num_heads;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&o_buffer, 0, &staging_buffer, 0, output_size as u64);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::PollType::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// GPU-pure tree attention forward (no readback/upload).
    pub fn forward_gpu_pure(
        &self,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        tree_mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: TreeAttnParams,
        use_f16: bool,
    ) -> Result<(), String> {
        let (pipeline, layout) = if use_f16 {
            let pipeline = self
                .forward_pipeline_f16
                .as_ref()
                .ok_or("F16 not supported")?;
            let layout = self
                .forward_bind_group_layout_f16
                .as_ref()
                .ok_or("F16 not supported")?;
            (pipeline, layout)
        } else {
            (&self.forward_pipeline_f32, &self.forward_bind_group_layout_f32)
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tree Attn Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tree Attn Bind Group GPU Pure"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tree_mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tree Attn Encoder GPU Pure"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tree Attn Pass GPU Pure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = (params.tree_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let workgroups_y = params.batch_size * params.num_heads;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Verify draft tokens against ground truth
    ///
    /// # Arguments
    /// * `draft_tokens` - Draft token indices [batch, tree_size]
    /// * `ground_truth` - Ground truth token indices [batch, tree_size]
    /// * `parent_indices` - Parent index for each node
    /// * `params` - Verification parameters
    ///
    /// # Returns
    /// * VerifyResult with accepted lengths and token indices
    pub fn verify(
        &self,
        draft_tokens: &[u32],
        ground_truth: &[u32],
        parent_indices: &[i32],
        params: &VerifyParams,
    ) -> VerifyResult {
        let batch_size = params.batch_size as usize;
        let tree_size = params.tree_size as usize;

        // Create buffers
        let draft_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Draft Tokens Buffer"),
                contents: bytemuck::cast_slice(draft_tokens),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let truth_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ground Truth Buffer"),
                contents: bytemuck::cast_slice(ground_truth),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let parents_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parent Indices Buffer"),
                contents: bytemuck::cast_slice(parent_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let length_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accepted Length Buffer"),
            size: (batch_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accepted_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accepted Tokens Buffer"),
            size: (batch_size * tree_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Verify Params Buffer"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Verify Bind Group"),
            layout: &self.verify_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: draft_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: truth_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: parents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: length_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: accepted_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Verify Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Verify Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.verify_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (batch_size as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Read back results
        let length_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Length Staging Buffer"),
            size: (batch_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let accepted_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accepted Staging Buffer"),
            size: (batch_size * tree_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &length_buffer,
            0,
            &length_staging,
            0,
            (batch_size * std::mem::size_of::<u32>()) as u64,
        );

        encoder.copy_buffer_to_buffer(
            &accepted_buffer,
            0,
            &accepted_staging,
            0,
            (batch_size * tree_size * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read lengths
        let length_slice = length_staging.slice(..);
        let (sender1, receiver1) = std::sync::mpsc::channel();
        length_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender1.send(result).unwrap();
        });

        let accepted_slice = accepted_staging.slice(..);
        let (sender2, receiver2) = std::sync::mpsc::channel();
        accepted_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender2.send(result).unwrap();
        });

        self.device.poll(wgpu::PollType::Wait);
        receiver1.recv().unwrap().unwrap();
        receiver2.recv().unwrap().unwrap();

        let length_data = length_slice.get_mapped_range();
        let accepted_lengths: Vec<u32> = bytemuck::cast_slice(&length_data).to_vec();
        drop(length_data);
        length_staging.unmap();

        let accepted_data = accepted_slice.get_mapped_range();
        let accepted_tokens: Vec<u32> = bytemuck::cast_slice(&accepted_data).to_vec();
        drop(accepted_data);
        accepted_staging.unmap();

        VerifyResult {
            accepted_lengths,
            accepted_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_params() {
        let params = TreeMaskParams::new(8);
        assert_eq!(params.num_nodes, 8);
    }

    #[test]
    fn test_attn_params() {
        let params = TreeAttnParams::new(2, 4, 128, 16, 64, 0.125);
        assert_eq!(params.batch_size, 2);
        assert_eq!(params.num_heads, 4);
        assert_eq!(params.prompt_len, 128);
        assert_eq!(params.tree_size, 16);
        assert_eq!(params.head_dim, 64);
        assert!((params.scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_attn_params_auto_scale() {
        let params = TreeAttnParams::with_auto_scale(2, 4, 128, 16, 64);
        let expected_scale = 1.0 / (64.0_f32).sqrt();
        assert!((params.scale - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn test_verify_params() {
        let params = VerifyParams::new(4, 8);
        assert_eq!(params.batch_size, 4);
        assert_eq!(params.tree_size, 8);
    }
}
