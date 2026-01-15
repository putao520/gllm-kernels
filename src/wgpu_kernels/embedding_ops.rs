//! WGPU embedding quantization and similarity kernels.
//!
//! GPU-accelerated operations for:
//! - Binary IP (Hamming distance for 1-bit quantized vectors)
//! - Int8 Dot Product (4x compression, near-lossless)
//! - Int4 Packed Dot Product (8x compression)
//! - Matryoshka Dimension Truncation
//! - Top-K Selection for Rerank Pipeline

use std::borrow::Cow;
use std::fmt;
use std::mem;
use std::sync::Arc;

use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::validation::{validate_binary_dim, validate_input_len};

const SHADER_SOURCE: &str = include_str!("kernels/embedding_ops.wgsl");
const WORKGROUP_SIZE: u32 = 256;
const MAX_LOCAL_K: usize = 256; // Must match WGSL MAX_LOCAL_K

// Kernel entry point names
const KERNEL_BINARY_IP_HAMMING: &str = "binary_ip_hamming";
const KERNEL_BINARY_IP_ASYMMETRIC: &str = "binary_ip_asymmetric";
const KERNEL_INT8_DOT_PRODUCT: &str = "int8_dot_product";
const KERNEL_INT4_DOT_PRODUCT: &str = "int4_dot_product";
const KERNEL_MATRYOSHKA_TRUNCATE: &str = "matryoshka_truncate";
const KERNEL_MATRYOSHKA_NORMALIZE: &str = "matryoshka_normalize";
const KERNEL_TOP_K_SELECT_F32: &str = "top_k_select_f32";
const KERNEL_TOP_K_SELECT_I32: &str = "top_k_select_i32";

/// Parameters for binary IP operations (symmetric and asymmetric).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BinaryIpParams {
    dim: u32,
    num_queries: u32,
    num_vectors: u32,
    _pad: u32,
}

/// Parameters for Int8 dot product.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Int8DotParams {
    dim: u32,
    num_queries: u32,
    num_vectors: u32,
    scale: f32,
}

/// Parameters for Int4 packed dot product.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Int4DotParams {
    dim: u32,
    num_queries: u32,
    num_vectors: u32,
    scale: f32,
    zero_point: i32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Parameters for Matryoshka truncation.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct MatryoshkaParams {
    full_dim: u32,
    target_dim: u32,
    num_vectors: u32,
    normalize: u32,
}

/// Parameters for Top-K selection.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TopKParams {
    num_elements: u32,
    k: u32,
    ascending: u32, // 0 = descending (higher is better), 1 = ascending (lower is better)
    _pad: u32,
}

/// Result of GPU rerank pipeline stage.
#[derive(Debug, Clone)]
pub struct GpuRerankStageResult {
    /// Indices of selected candidates
    pub indices: Vec<u32>,
    /// Scores of selected candidates
    pub scores: Vec<f32>,
}

/// Configuration for GPU rerank pipeline.
#[derive(Debug, Clone)]
pub struct GpuRerankConfig {
    /// Number of candidates to select in binary stage (e.g., 10000)
    pub binary_k: usize,
    /// Number of candidates to select in int8 stage (e.g., 100)
    pub int8_k: usize,
    /// Vector dimension
    pub dim: usize,
}

/// Errors from embedding GPU kernels.
#[derive(Debug)]
pub enum EmbeddingOpsError {
    /// WGPU driver or initialization error.
    Wgpu(String),
    /// Invalid configuration or shape.
    InvalidConfig(String),
    /// Unsupported operation.
    Unsupported(String),
}

impl fmt::Display for EmbeddingOpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu(msg) => write!(f, "WGPU error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
        }
    }
}

impl std::error::Error for EmbeddingOpsError {}

/// WGPU embedding operations kernel wrapper.
pub struct EmbeddingOpsKernel {
    device: Arc<Device>,
    queue: Arc<Queue>,
    // Binary IP layouts and pipelines
    binary_ip_layout: BindGroupLayout,
    pipeline_binary_hamming: ComputePipeline,
    pipeline_binary_asymmetric: ComputePipeline,
    // Int8 layout and pipeline
    int8_layout: BindGroupLayout,
    pipeline_int8_dot: ComputePipeline,
    // Int4 layout and pipeline
    int4_layout: BindGroupLayout,
    pipeline_int4_dot: ComputePipeline,
    // Matryoshka layouts and pipelines
    matryoshka_truncate_layout: BindGroupLayout,
    pipeline_matryoshka_truncate: ComputePipeline,
    matryoshka_normalize_layout: BindGroupLayout,
    pipeline_matryoshka_normalize: ComputePipeline,
    // Top-K layouts and pipelines
    top_k_f32_layout: BindGroupLayout,
    pipeline_top_k_f32: ComputePipeline,
    top_k_i32_layout: BindGroupLayout,
    pipeline_top_k_i32: ComputePipeline,
}

impl EmbeddingOpsKernel {
    /// Create embedding ops kernel wrapper for an existing WGPU device.
    pub fn new(device: &Device, queue: &Queue) -> Result<Self, EmbeddingOpsError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embedding_ops.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // Binary IP layout: queries (r), database (r), scores (rw), params (u)
        let binary_ip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binary_ip_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // queries
                buffer_layout_entry(1, true),  // database
                buffer_layout_entry(2, false), // scores
                uniform_layout_entry(3),       // params
            ],
        });

        let binary_ip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("binary_ip_pipeline_layout"),
                bind_group_layouts: &[&binary_ip_layout],
                push_constant_ranges: &[],
            });

        let pipeline_binary_hamming =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("binary_ip_hamming"),
                layout: Some(&binary_ip_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_BINARY_IP_HAMMING),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline_binary_asymmetric =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("binary_ip_asymmetric"),
                layout: Some(&binary_ip_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_BINARY_IP_ASYMMETRIC),
                compilation_options: Default::default(),
                cache: None,
            });

        // Int8 layout: same as binary IP but output is f32
        let int8_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("int8_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // queries
                buffer_layout_entry(1, true),  // database
                buffer_layout_entry(2, false), // scores
                uniform_layout_entry(3),       // params
            ],
        });

        let int8_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("int8_pipeline_layout"),
            bind_group_layouts: &[&int8_layout],
            push_constant_ranges: &[],
        });

        let pipeline_int8_dot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("int8_dot_product"),
            layout: Some(&int8_pipeline_layout),
            module: &shader,
            entry_point: Some(KERNEL_INT8_DOT_PRODUCT),
            compilation_options: Default::default(),
            cache: None,
        });

        // Int4 layout
        let int4_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("int4_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // queries
                buffer_layout_entry(1, true),  // database
                buffer_layout_entry(2, false), // scores
                uniform_layout_entry(3),       // params
            ],
        });

        let int4_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("int4_pipeline_layout"),
            bind_group_layouts: &[&int4_layout],
            push_constant_ranges: &[],
        });

        let pipeline_int4_dot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("int4_dot_product"),
            layout: Some(&int4_pipeline_layout),
            module: &shader,
            entry_point: Some(KERNEL_INT4_DOT_PRODUCT),
            compilation_options: Default::default(),
            cache: None,
        });

        // Matryoshka truncate layout: input (r), output (rw), params (u)
        let matryoshka_truncate_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matryoshka_truncate_layout"),
                entries: &[
                    buffer_layout_entry(0, true),  // input
                    buffer_layout_entry(1, false), // output
                    uniform_layout_entry(2),       // params
                ],
            });

        let matryoshka_truncate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matryoshka_truncate_pipeline_layout"),
                bind_group_layouts: &[&matryoshka_truncate_layout],
                push_constant_ranges: &[],
            });

        let pipeline_matryoshka_truncate =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matryoshka_truncate"),
                layout: Some(&matryoshka_truncate_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_MATRYOSHKA_TRUNCATE),
                compilation_options: Default::default(),
                cache: None,
            });

        // Matryoshka normalize layout: vectors (rw), params (u)
        let matryoshka_normalize_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matryoshka_normalize_layout"),
                entries: &[
                    buffer_layout_entry(0, false), // vectors (read-write)
                    uniform_layout_entry(1),       // params
                ],
            });

        let matryoshka_normalize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matryoshka_normalize_pipeline_layout"),
                bind_group_layouts: &[&matryoshka_normalize_layout],
                push_constant_ranges: &[],
            });

        let pipeline_matryoshka_normalize =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matryoshka_normalize"),
                layout: Some(&matryoshka_normalize_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_MATRYOSHKA_NORMALIZE),
                compilation_options: Default::default(),
                cache: None,
            });

        // Top-K f32 layout: scores (r), out_indices (rw), out_scores (rw), params (u)
        let top_k_f32_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("top_k_f32_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // scores
                buffer_layout_entry(1, false), // out_indices
                buffer_layout_entry(2, false), // out_scores
                uniform_layout_entry(3),       // params
            ],
        });

        let top_k_f32_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("top_k_f32_pipeline_layout"),
                bind_group_layouts: &[&top_k_f32_layout],
                push_constant_ranges: &[],
            });

        let pipeline_top_k_f32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("top_k_select_f32"),
                layout: Some(&top_k_f32_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_TOP_K_SELECT_F32),
                compilation_options: Default::default(),
                cache: None,
            });

        // Top-K i32 layout: scores (r), out_indices (rw), out_scores (rw), params (u)
        let top_k_i32_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("top_k_i32_layout"),
            entries: &[
                buffer_layout_entry(0, true),  // scores
                buffer_layout_entry(1, false), // out_indices
                buffer_layout_entry(2, false), // out_scores
                uniform_layout_entry(3),       // params
            ],
        });

        let top_k_i32_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("top_k_i32_pipeline_layout"),
                bind_group_layouts: &[&top_k_i32_layout],
                push_constant_ranges: &[],
            });

        let pipeline_top_k_i32 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("top_k_select_i32"),
                layout: Some(&top_k_i32_pipeline_layout),
                module: &shader,
                entry_point: Some(KERNEL_TOP_K_SELECT_I32),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device: Arc::new(device.clone()),
            queue: Arc::new(queue.clone()),
            binary_ip_layout,
            pipeline_binary_hamming,
            pipeline_binary_asymmetric,
            int8_layout,
            pipeline_int8_dot,
            int4_layout,
            pipeline_int4_dot,
            matryoshka_truncate_layout,
            pipeline_matryoshka_truncate,
            matryoshka_normalize_layout,
            pipeline_matryoshka_normalize,
            top_k_f32_layout,
            pipeline_top_k_f32,
            top_k_i32_layout,
            pipeline_top_k_i32,
        })
    }

    /// Create with a newly initialized default WGPU device.
    pub fn create_default() -> Result<Self, EmbeddingOpsError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| EmbeddingOpsError::Wgpu(format!("no compatible adapter: {e}")))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gllm-wgpu-embedding"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|e| EmbeddingOpsError::Wgpu(format!("request_device failed: {e}")))?;

        Self::new(&device, &queue)
    }

    /// Binary IP Hamming distance between binary-quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/32] packed u32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] Hamming distances (i32)
    pub fn binary_ip_hamming(
        &self,
        queries: &[u32],
        database: &[u32],
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<Vec<i32>, EmbeddingOpsError> {
        validate_binary_dim(dim).map_err(EmbeddingOpsError::InvalidConfig)?;
        let packed_dim = dim / 32;
        validate_input_len(queries.len(), num_queries * packed_dim, "queries")
            .map_err(EmbeddingOpsError::InvalidConfig)?;
        validate_input_len(database.len(), num_vectors * packed_dim, "database")
            .map_err(EmbeddingOpsError::InvalidConfig)?;

        let params = BinaryIpParams {
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
            _pad: 0,
        };

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<i32>()) as u64;

        let output = self.dispatch_binary_op(
            slice_as_bytes(queries),
            slice_as_bytes(database),
            output_bytes,
            params,
            &self.pipeline_binary_hamming,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Asymmetric Binary IP: f32 query vs binary database.
    ///
    /// - `queries`: [num_queries, dim] f32
    /// - `database`: [num_vectors, dim/32] packed u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn binary_ip_asymmetric(
        &self,
        queries: &[f32],
        database: &[u32],
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
    ) -> Result<Vec<f32>, EmbeddingOpsError> {
        let expected_queries = num_queries * dim;
        let expected_database = num_vectors * (dim / 32);
        if queries.len() != expected_queries {
            return Err(EmbeddingOpsError::InvalidConfig(format!(
                "queries len {} != expected {}",
                queries.len(),
                expected_queries
            )));
        }
        if database.len() != expected_database {
            return Err(EmbeddingOpsError::InvalidConfig(format!(
                "database len {} != expected {}",
                database.len(),
                expected_database
            )));
        }

        let params = BinaryIpParams {
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
            _pad: 0,
        };

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;

        let output = self.dispatch_asymmetric_op(
            slice_as_bytes(queries),
            slice_as_bytes(database),
            output_bytes,
            params,
            &self.pipeline_binary_asymmetric,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Int8 dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/4] packed i8x4 as u32
    /// - `database`: [num_vectors, dim/4] packed i8x4 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int8_dot_product(
        &self,
        queries: &[u32],
        database: &[u32],
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
    ) -> Result<Vec<f32>, EmbeddingOpsError> {
        let packed_dim = dim / 4;
        let expected_queries = num_queries * packed_dim;
        let expected_database = num_vectors * packed_dim;
        if queries.len() != expected_queries || database.len() != expected_database {
            return Err(EmbeddingOpsError::InvalidConfig(
                "int8 input length mismatch".into(),
            ));
        }

        let params = Int8DotParams {
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
            scale,
        };

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;

        let output = self.dispatch_int8_op(
            slice_as_bytes(queries),
            slice_as_bytes(database),
            output_bytes,
            params,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Int4 packed dot product between quantized vectors.
    ///
    /// - `queries`: [num_queries, dim/8] packed i4x8 as u32
    /// - `database`: [num_vectors, dim/8] packed i4x8 as u32
    /// - Returns: [num_queries, num_vectors] similarity scores (f32)
    pub fn int4_dot_product(
        &self,
        queries: &[u32],
        database: &[u32],
        dim: usize,
        num_queries: usize,
        num_vectors: usize,
        scale: f32,
        zero_point: i32,
    ) -> Result<Vec<f32>, EmbeddingOpsError> {
        let packed_dim = dim / 8;
        let expected_queries = num_queries * packed_dim;
        let expected_database = num_vectors * packed_dim;
        if queries.len() != expected_queries || database.len() != expected_database {
            return Err(EmbeddingOpsError::InvalidConfig(
                "int4 input length mismatch".into(),
            ));
        }

        let params = Int4DotParams {
            dim: dim as u32,
            num_queries: num_queries as u32,
            num_vectors: num_vectors as u32,
            scale,
            zero_point,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let output_len = num_queries * num_vectors;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;

        let output = self.dispatch_int4_op(
            slice_as_bytes(queries),
            slice_as_bytes(database),
            output_bytes,
            params,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Truncate embeddings to a smaller dimension (Matryoshka).
    ///
    /// - `input`: [num_vectors, full_dim] f32
    /// - Returns: [num_vectors, target_dim] f32
    pub fn matryoshka_truncate(
        &self,
        input: &[f32],
        full_dim: usize,
        target_dim: usize,
        num_vectors: usize,
        normalize: bool,
    ) -> Result<Vec<f32>, EmbeddingOpsError> {
        if target_dim > full_dim {
            return Err(EmbeddingOpsError::InvalidConfig(
                "target_dim > full_dim".into(),
            ));
        }
        let expected_input = num_vectors * full_dim;
        if input.len() != expected_input {
            return Err(EmbeddingOpsError::InvalidConfig(
                "input length mismatch".into(),
            ));
        }

        let params = MatryoshkaParams {
            full_dim: full_dim as u32,
            target_dim: target_dim as u32,
            num_vectors: num_vectors as u32,
            normalize: if normalize { 1 } else { 0 },
        };

        let output_len = num_vectors * target_dim;
        let output_bytes = (output_len * mem::size_of::<f32>()) as u64;

        let mut output = self.dispatch_matryoshka_truncate(
            slice_as_bytes(input),
            output_bytes,
            params,
        )?;

        // If normalize requested, run normalize pass
        if normalize {
            output = self.dispatch_matryoshka_normalize(&output, params)?;
        }

        Ok(bytes_to_vec(&output))
    }

    /// Top-K selection for f32 scores.
    ///
    /// - `scores`: [num_elements] f32 scores
    /// - `k`: number of top elements to select
    /// - `ascending`: if true, select lowest scores (for Hamming distance)
    /// - Returns: (indices, scores) of top-K elements
    pub fn top_k_select_f32(
        &self,
        scores: &[f32],
        k: usize,
        ascending: bool,
    ) -> Result<(Vec<u32>, Vec<f32>), EmbeddingOpsError> {
        let num_elements = scores.len();
        if num_elements == 0 || k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let k = k.min(num_elements).min(MAX_LOCAL_K);
        let num_workgroups = (num_elements + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize;

        let (indices, out_scores) = self.dispatch_top_k_f32(
            slice_as_bytes(scores),
            num_elements,
            k,
            ascending,
            num_workgroups,
        )?;

        // Merge results from workgroups on CPU
        let merged = self.merge_top_k_f32(&indices, &out_scores, k, ascending, num_workgroups);
        Ok(merged)
    }

    /// Top-K selection for i32 scores (Hamming distance).
    ///
    /// - `scores`: [num_elements] i32 scores (Hamming distances)
    /// - `k`: number of top elements to select
    /// - `ascending`: if true, select lowest scores (default for Hamming)
    /// - Returns: (indices, scores as f32) of top-K elements
    pub fn top_k_select_i32(
        &self,
        scores: &[i32],
        k: usize,
        ascending: bool,
    ) -> Result<(Vec<u32>, Vec<f32>), EmbeddingOpsError> {
        let num_elements = scores.len();
        if num_elements == 0 || k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let k = k.min(num_elements).min(MAX_LOCAL_K);
        let num_workgroups = (num_elements + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize;

        let (indices, out_scores_i32) = self.dispatch_top_k_i32(
            slice_as_bytes(scores),
            num_elements,
            k,
            ascending,
            num_workgroups,
        )?;

        // Merge results from workgroups on CPU
        let merged = self.merge_top_k_i32(&indices, &out_scores_i32, k, ascending, num_workgroups);
        Ok(merged)
    }

    /// Binary stage of rerank pipeline using GPU.
    ///
    /// Computes Hamming distances and selects top-K candidates.
    /// - `query`: [dim/32] packed u32 binary query
    /// - `database`: [num_vectors, dim/32] packed u32 binary database
    /// - `k`: number of candidates to select
    /// - Returns: indices and converted f32 scores (negated for consistency with other stages)
    pub fn rerank_binary_stage(
        &self,
        query: &[u32],
        database: &[u32],
        dim: usize,
        num_vectors: usize,
        k: usize,
    ) -> Result<GpuRerankStageResult, EmbeddingOpsError> {
        // Compute Hamming distances
        let distances = self.binary_ip_hamming(query, database, dim, 1, num_vectors)?;

        // Select top-K (ascending = true for Hamming, lower is better)
        let (indices, scores) = self.top_k_select_i32(&distances, k, true)?;

        Ok(GpuRerankStageResult { indices, scores })
    }

    /// Int8 stage of rerank pipeline using GPU.
    ///
    /// Computes int8 dot products and selects top-K candidates.
    /// - `query`: [dim/4] packed i8x4 query
    /// - `database`: [num_vectors, dim/4] packed i8x4 database
    /// - `candidate_indices`: indices from previous stage
    /// - `k`: number of candidates to select
    /// - Returns: indices (into original database) and f32 scores
    pub fn rerank_int8_stage(
        &self,
        query: &[u32],
        database: &[u32],
        dim: usize,
        candidate_indices: &[u32],
        k: usize,
        scale: f32,
    ) -> Result<GpuRerankStageResult, EmbeddingOpsError> {
        let num_candidates = candidate_indices.len();
        if num_candidates == 0 {
            return Ok(GpuRerankStageResult {
                indices: Vec::new(),
                scores: Vec::new(),
            });
        }

        // Extract candidate vectors
        let packed_dim = dim / 4;
        let mut candidate_database = Vec::with_capacity(num_candidates * packed_dim);
        for &idx in candidate_indices {
            let start = idx as usize * packed_dim;
            let end = start + packed_dim;
            if end <= database.len() {
                candidate_database.extend_from_slice(&database[start..end]);
            }
        }

        // Compute int8 dot products
        let scores = self.int8_dot_product(query, &candidate_database, dim, 1, num_candidates, scale)?;

        // Select top-K (descending = false for dot product, higher is better)
        let (local_indices, out_scores) = self.top_k_select_f32(&scores, k, false)?;

        // Map back to original indices
        let indices: Vec<u32> = local_indices
            .iter()
            .filter_map(|&local_idx| {
                if (local_idx as usize) < candidate_indices.len() {
                    Some(candidate_indices[local_idx as usize])
                } else {
                    None
                }
            })
            .collect();

        Ok(GpuRerankStageResult {
            indices,
            scores: out_scores,
        })
    }

    /// Full rerank pipeline combining binary and int8 stages.
    ///
    /// Stage 1: Binary Hamming distance → select binary_k candidates
    /// Stage 2: Int8 dot product → select int8_k candidates
    ///
    /// - `binary_query`: [dim/32] packed u32 binary query
    /// - `binary_database`: [num_vectors, dim/32] packed u32 binary database
    /// - `int8_query`: [dim/4] packed i8x4 query
    /// - `int8_database`: [num_vectors, dim/4] packed i8x4 database
    /// - `config`: pipeline configuration
    /// - Returns: final candidates with scores
    pub fn rerank_pipeline(
        &self,
        binary_query: &[u32],
        binary_database: &[u32],
        int8_query: &[u32],
        int8_database: &[u32],
        num_vectors: usize,
        config: &GpuRerankConfig,
        int8_scale: f32,
    ) -> Result<GpuRerankStageResult, EmbeddingOpsError> {
        // Stage 1: Binary stage
        let binary_result = self.rerank_binary_stage(
            binary_query,
            binary_database,
            config.dim,
            num_vectors,
            config.binary_k,
        )?;

        if binary_result.indices.is_empty() {
            return Ok(binary_result);
        }

        // Stage 2: Int8 stage
        let int8_result = self.rerank_int8_stage(
            int8_query,
            int8_database,
            config.dim,
            &binary_result.indices,
            config.int8_k,
            int8_scale,
        )?;

        Ok(int8_result)
    }

    // Internal dispatch methods

    fn dispatch_binary_op(
        &self,
        queries_bytes: &[u8],
        database_bytes: &[u8],
        output_bytes: u64,
        params: BinaryIpParams,
        pipeline: &ComputePipeline,
    ) -> Result<Vec<u8>, EmbeddingOpsError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let queries_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("binary_queries"),
            contents: queries_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let database_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("binary_database"),
            contents: database_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("binary_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_bind_group"),
            layout: &self.binary_ip_layout,
            entries: &[
                buffer_binding(0, &queries_buffer),
                buffer_binding(1, &database_buffer),
                buffer_binding(2, &output_buffer),
                buffer_binding(3, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("binary_encoder"),
        });

        let total_pairs = params.num_queries * params.num_vectors;
        let workgroups = (total_pairs + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let mut data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        data.truncate(output_bytes as usize);
        Ok(data)
    }

    fn dispatch_asymmetric_op(
        &self,
        queries_bytes: &[u8],
        database_bytes: &[u8],
        output_bytes: u64,
        params: BinaryIpParams,
        pipeline: &ComputePipeline,
    ) -> Result<Vec<u8>, EmbeddingOpsError> {
        // Same as binary op but queries are f32
        self.dispatch_binary_op(queries_bytes, database_bytes, output_bytes, params, pipeline)
    }

    fn dispatch_int8_op(
        &self,
        queries_bytes: &[u8],
        database_bytes: &[u8],
        output_bytes: u64,
        params: Int8DotParams,
    ) -> Result<Vec<u8>, EmbeddingOpsError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let queries_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("int8_queries"),
            contents: queries_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let database_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("int8_database"),
            contents: database_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("int8_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("int8_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("int8_bind_group"),
            layout: &self.int8_layout,
            entries: &[
                buffer_binding(0, &queries_buffer),
                buffer_binding(1, &database_buffer),
                buffer_binding(2, &output_buffer),
                buffer_binding(3, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("int8_encoder"),
        });

        let total_pairs = params.num_queries * params.num_vectors;
        let workgroups = (total_pairs + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("int8_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_int8_dot);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("int8_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let mut data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        data.truncate(output_bytes as usize);
        Ok(data)
    }

    fn dispatch_int4_op(
        &self,
        queries_bytes: &[u8],
        database_bytes: &[u8],
        output_bytes: u64,
        params: Int4DotParams,
    ) -> Result<Vec<u8>, EmbeddingOpsError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let queries_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("int4_queries"),
            contents: queries_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let database_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("int4_database"),
            contents: database_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("int4_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("int4_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("int4_bind_group"),
            layout: &self.int4_layout,
            entries: &[
                buffer_binding(0, &queries_buffer),
                buffer_binding(1, &database_buffer),
                buffer_binding(2, &output_buffer),
                buffer_binding(3, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("int4_encoder"),
        });

        let total_pairs = params.num_queries * params.num_vectors;
        let workgroups = (total_pairs + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("int4_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_int4_dot);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("int4_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let mut data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        data.truncate(output_bytes as usize);
        Ok(data)
    }

    fn dispatch_matryoshka_truncate(
        &self,
        input_bytes: &[u8],
        output_bytes: u64,
        params: MatryoshkaParams,
    ) -> Result<Vec<u8>, EmbeddingOpsError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matryoshka_input"),
            contents: input_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let padded_bytes = align_up(output_bytes, max_align());
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matryoshka_output"),
            size: padded_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matryoshka_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matryoshka_truncate_bind_group"),
            layout: &self.matryoshka_truncate_layout,
            entries: &[
                buffer_binding(0, &input_buffer),
                buffer_binding(1, &output_buffer),
                buffer_binding(2, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matryoshka_encoder"),
        });

        let total_elements = params.num_vectors * params.target_dim;
        let workgroups = (total_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matryoshka_truncate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_matryoshka_truncate);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matryoshka_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let mut data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        data.truncate(output_bytes as usize);
        Ok(data)
    }

    fn dispatch_matryoshka_normalize(
        &self,
        input_bytes: &[u8],
        params: MatryoshkaParams,
    ) -> Result<Vec<u8>, EmbeddingOpsError> {
        if input_bytes.is_empty() {
            return Ok(Vec::new());
        }

        let padded_bytes = align_up(input_bytes.len() as u64, max_align());

        // Create read-write buffer with input data
        let vectors_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matryoshka_normalize_vectors"),
            contents: input_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matryoshka_normalize_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matryoshka_normalize_bind_group"),
            layout: &self.matryoshka_normalize_layout,
            entries: &[
                buffer_binding(0, &vectors_buffer),
                buffer_binding(1, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matryoshka_normalize_encoder"),
        });

        let workgroups = (params.num_vectors + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matryoshka_normalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_matryoshka_normalize);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matryoshka_normalize_readback"),
            size: padded_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&vectors_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let mut data = read_buffer_sync(&self.device, &readback, padded_bytes)?;
        data.truncate(input_bytes.len());
        Ok(data)
    }

    fn dispatch_top_k_f32(
        &self,
        scores_bytes: &[u8],
        num_elements: usize,
        k: usize,
        ascending: bool,
        num_workgroups: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), EmbeddingOpsError> {
        let params = TopKParams {
            num_elements: num_elements as u32,
            k: k as u32,
            ascending: if ascending { 1 } else { 0 },
            _pad: 0,
        };

        let scores_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("topk_scores_f32"),
            contents: scores_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // Output: [num_workgroups * k] indices and scores
        let output_len = num_workgroups * k;
        let indices_bytes = (output_len * mem::size_of::<u32>()) as u64;
        let scores_out_bytes = (output_len * mem::size_of::<f32>()) as u64;

        let padded_indices = align_up(indices_bytes, max_align());
        let padded_scores = align_up(scores_out_bytes, max_align());

        let out_indices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_out_indices"),
            size: padded_indices,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_scores_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_out_scores"),
            size: padded_scores,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("topk_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("topk_f32_bind_group"),
            layout: &self.top_k_f32_layout,
            entries: &[
                buffer_binding(0, &scores_buffer),
                buffer_binding(1, &out_indices_buffer),
                buffer_binding(2, &out_scores_buffer),
                buffer_binding(3, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("topk_f32_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("topk_f32_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_top_k_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        let indices_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_indices_readback"),
            size: padded_indices,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scores_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_scores_readback"),
            size: padded_scores,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&out_indices_buffer, 0, &indices_readback, 0, padded_indices);
        encoder.copy_buffer_to_buffer(&out_scores_buffer, 0, &scores_readback, 0, padded_scores);
        self.queue.submit(Some(encoder.finish()));

        let mut indices_data = read_buffer_sync(&self.device, &indices_readback, padded_indices)?;
        indices_data.truncate(indices_bytes as usize);

        let mut scores_data = read_buffer_sync(&self.device, &scores_readback, padded_scores)?;
        scores_data.truncate(scores_out_bytes as usize);

        let indices: Vec<u32> = bytes_to_vec(&indices_data);
        let scores: Vec<f32> = bytes_to_vec(&scores_data);

        Ok((indices, scores))
    }

    fn dispatch_top_k_i32(
        &self,
        scores_bytes: &[u8],
        num_elements: usize,
        k: usize,
        ascending: bool,
        num_workgroups: usize,
    ) -> Result<(Vec<u32>, Vec<i32>), EmbeddingOpsError> {
        let params = TopKParams {
            num_elements: num_elements as u32,
            k: k as u32,
            ascending: if ascending { 1 } else { 0 },
            _pad: 0,
        };

        let scores_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("topk_scores_i32"),
            contents: scores_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let output_len = num_workgroups * k;
        let indices_bytes = (output_len * mem::size_of::<u32>()) as u64;
        let scores_out_bytes = (output_len * mem::size_of::<i32>()) as u64;

        let padded_indices = align_up(indices_bytes, max_align());
        let padded_scores = align_up(scores_out_bytes, max_align());

        let out_indices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_i32_out_indices"),
            size: padded_indices,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_scores_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_i32_out_scores"),
            size: padded_scores,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("topk_i32_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("topk_i32_bind_group"),
            layout: &self.top_k_i32_layout,
            entries: &[
                buffer_binding(0, &scores_buffer),
                buffer_binding(1, &out_indices_buffer),
                buffer_binding(2, &out_scores_buffer),
                buffer_binding(3, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("topk_i32_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("topk_i32_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_top_k_i32);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        let indices_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_i32_indices_readback"),
            size: padded_indices,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scores_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_i32_scores_readback"),
            size: padded_scores,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&out_indices_buffer, 0, &indices_readback, 0, padded_indices);
        encoder.copy_buffer_to_buffer(&out_scores_buffer, 0, &scores_readback, 0, padded_scores);
        self.queue.submit(Some(encoder.finish()));

        let mut indices_data = read_buffer_sync(&self.device, &indices_readback, padded_indices)?;
        indices_data.truncate(indices_bytes as usize);

        let mut scores_data = read_buffer_sync(&self.device, &scores_readback, padded_scores)?;
        scores_data.truncate(scores_out_bytes as usize);

        let indices: Vec<u32> = bytes_to_vec(&indices_data);
        let scores: Vec<i32> = bytes_to_vec(&scores_data);

        Ok((indices, scores))
    }

    /// Merge top-K results from multiple workgroups (f32 version).
    fn merge_top_k_f32(
        &self,
        indices: &[u32],
        scores: &[f32],
        k: usize,
        ascending: bool,
        num_workgroups: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        if num_workgroups == 0 || k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Collect all (score, index) pairs from workgroups
        let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(num_workgroups * k);
        for wg in 0..num_workgroups {
            let offset = wg * k;
            for i in 0..k {
                if offset + i < indices.len() && offset + i < scores.len() {
                    let idx = indices[offset + i];
                    let score = scores[offset + i];
                    // Skip invalid entries (0xFFFFFFFF is sentinel)
                    if idx != 0xFFFFFFFF {
                        pairs.push((score, idx));
                    }
                }
            }
        }

        // Sort by score
        if ascending {
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Take top-K
        pairs.truncate(k);

        let result_indices: Vec<u32> = pairs.iter().map(|(_, idx)| *idx).collect();
        let result_scores: Vec<f32> = pairs.iter().map(|(score, _)| *score).collect();

        (result_indices, result_scores)
    }

    /// Merge top-K results from multiple workgroups (i32 version, converts to f32).
    fn merge_top_k_i32(
        &self,
        indices: &[u32],
        scores: &[i32],
        k: usize,
        ascending: bool,
        num_workgroups: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        if num_workgroups == 0 || k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Collect all (score, index) pairs from workgroups
        let mut pairs: Vec<(i32, u32)> = Vec::with_capacity(num_workgroups * k);
        for wg in 0..num_workgroups {
            let offset = wg * k;
            for i in 0..k {
                if offset + i < indices.len() && offset + i < scores.len() {
                    let idx = indices[offset + i];
                    let score = scores[offset + i];
                    // Skip invalid entries
                    if idx != 0xFFFFFFFF {
                        pairs.push((score, idx));
                    }
                }
            }
        }

        // Sort by score
        if ascending {
            pairs.sort_by_key(|(score, _)| *score);
        } else {
            pairs.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        }

        // Take top-K
        pairs.truncate(k);

        let result_indices: Vec<u32> = pairs.iter().map(|(_, idx)| *idx).collect();
        // Convert i32 scores to f32 (negated for Hamming so higher = better)
        let result_scores: Vec<f32> = pairs.iter().map(|(score, _)| -(*score as f32)).collect();

        (result_indices, result_scores)
    }
}

// Helper functions

fn buffer_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_binding(binding: u32, buffer: &Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T) as *const u8, mem::size_of::<T>()) }
}

fn slice_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * mem::size_of::<T>())
    }
}

fn bytes_to_vec<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let len = bytes.len() / mem::size_of::<T>();
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

fn align_up(value: u64, align: u64) -> u64 {
    if align == 0 {
        return value;
    }
    (value + align - 1) / align * align
}

fn max_align() -> u64 {
    let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
    let map_align = wgpu::MAP_ALIGNMENT;
    if copy_align > map_align {
        copy_align
    } else {
        map_align
    }
}

fn read_buffer_sync(
    device: &Device,
    buffer: &Buffer,
    size: u64,
) -> Result<Vec<u8>, EmbeddingOpsError> {
    let slice = buffer.slice(0..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            return Err(EmbeddingOpsError::Wgpu(format!("map_async failed: {err}")))
        }
        Err(_) => {
            return Err(EmbeddingOpsError::Wgpu("map_async channel closed".into()));
        }
    }

    let data = slice.get_mapped_range();
    let bytes = data.to_vec();
    drop(data);
    buffer.unmap();
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_ip_hamming_small() {
        // Skip if no WGPU device available
        let kernel = match EmbeddingOpsKernel::create_default() {
            Ok(k) => k,
            Err(_) => return, // No GPU available
        };

        // 64-bit vectors (2 x u32)
        let queries = vec![0xFFFF_FFFFu32, 0x0000_0000];  // all 1s, all 0s
        let database = vec![0x0000_0000u32, 0xFFFF_FFFF]; // all 0s, all 1s

        let result = kernel.binary_ip_hamming(&queries, &database, 64, 1, 1).unwrap();
        // XOR: 0xFFFF_FFFF ^ 0 = 0xFFFF_FFFF (32 bits), 0 ^ 0xFFFF_FFFF = 0xFFFF_FFFF (32 bits)
        // Total Hamming = 64
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 64);
    }

    #[test]
    fn test_matryoshka_truncate_small() {
        let kernel = match EmbeddingOpsKernel::create_default() {
            Ok(k) => k,
            Err(_) => return,
        };

        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 vectors of dim 4
        let result = kernel.matryoshka_truncate(&input, 4, 2, 2, false).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[3], 6.0);
    }
}
