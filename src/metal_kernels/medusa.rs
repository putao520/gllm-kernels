//! Metal Medusa Heads auxiliary generation kernels.
//!
//! This module provides Metal GPU-accelerated kernels for Medusa speculative decoding:
//! - Multi-head parallel draft generation
//! - Top-k sampling from Medusa heads
//! - Candidate tree building and verification
//!
//! ## Precompiled metallib (Required)
//!
//! metallib must be precompiled before use:
//! ```bash
//! ./scripts/compile_metal_kernels.sh
//! ```
//!
//! metallib is Metal's intermediate format (like PTX/HSACO).
//! NO runtime compilation fallback - metallib must be precompiled and embedded.

use std::fmt;
use std::mem;
use std::os::raw::c_void;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};

use crate::metal_kernels::metallib_loader::{MetallibCollection, MetallibLoadError};

const KERNEL_HEAD_FORWARD_F32: &str = "medusa_head_forward_f32";
const KERNEL_HEAD_FORWARD_F16: &str = "medusa_head_forward_f16";
const KERNEL_TOP_K_SAMPLE_F32: &str = "medusa_top_k_sample_f32";
const KERNEL_TOP_K_SAMPLE_F16: &str = "medusa_top_k_sample_f16";
const KERNEL_BUILD_CANDIDATES_F32: &str = "medusa_build_candidates_f32";
const KERNEL_BUILD_CANDIDATES_F16: &str = "medusa_build_candidates_f16";
const KERNEL_VERIFY_CANDIDATES_F32: &str = "medusa_verify_candidates_f32";
const KERNEL_VERIFY_CANDIDATES_F16: &str = "medusa_verify_candidates_f16";

/// Metallib collection for Medusa kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static MEDUSA_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "medusa",
    metallib_data: include_bytes!("kernels/medusa.metallib"),
};

/// Parameters for head forward kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct HeadForwardParams {
    batch_size: u32,
    hidden_dim: u32,
    vocab_size: u32,
    num_heads: u32,
}

/// Parameters for top-k sampling kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TopKSampleParams {
    batch_size: u32,
    vocab_size: u32,
    num_heads: u32,
    top_k: u32,
    temperature: f32,
    _pad: [u32; 3],
}

/// Parameters for build candidates kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BuildCandidatesParams {
    batch_size: u32,
    num_heads: u32,
    top_k: u32,
    max_candidates: u32,
}

/// Parameters for verify candidates kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct VerifyCandidatesParams {
    batch_size: u32,
    num_candidates: u32,
    seq_len: u32,
    vocab_size: u32,
}

/// Errors surfaced by the Metal Medusa kernels.
#[derive(Debug)]
pub enum MedusaError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for MedusaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for MedusaError {}

impl From<MetallibLoadError> for MedusaError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Configuration for Medusa operations.
#[derive(Clone, Debug)]
pub struct MedusaConfig {
    /// Number of Medusa heads.
    pub num_heads: usize,
    /// Top-k for each head's sampling.
    pub top_k: usize,
    /// Maximum number of candidates in the tree.
    pub max_candidates: usize,
    /// Temperature for sampling.
    pub temperature: f32,
}

impl Default for MedusaConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            top_k: 10,
            max_candidates: 64,
            temperature: 1.0,
        }
    }
}

impl MedusaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), MedusaError> {
        if self.num_heads == 0 {
            return Err(MedusaError::InvalidConfig("num_heads must be positive".into()));
        }
        if self.top_k == 0 {
            return Err(MedusaError::InvalidConfig("top_k must be positive".into()));
        }
        if self.temperature <= 0.0 {
            return Err(MedusaError::InvalidConfig("temperature must be positive".into()));
        }
        Ok(())
    }
}

/// Medusa Metal kernel wrapper.
pub struct MedusaKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_head_forward_f32: ComputePipelineState,
    pipeline_head_forward_f16: ComputePipelineState,
    pipeline_top_k_sample_f32: ComputePipelineState,
    pipeline_top_k_sample_f16: ComputePipelineState,
    pipeline_build_candidates_f32: ComputePipelineState,
    pipeline_build_candidates_f16: ComputePipelineState,
    pipeline_verify_candidates_f32: ComputePipelineState,
    pipeline_verify_candidates_f16: ComputePipelineState,
}

impl MedusaKernel {
    /// Load Medusa kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, MedusaError> {
        let library = load_library(device)?;

        let pipeline_head_forward_f32 = build_pipeline(device, &library, KERNEL_HEAD_FORWARD_F32)?;
        let pipeline_head_forward_f16 = build_pipeline(device, &library, KERNEL_HEAD_FORWARD_F16)?;
        let pipeline_top_k_sample_f32 = build_pipeline(device, &library, KERNEL_TOP_K_SAMPLE_F32)?;
        let pipeline_top_k_sample_f16 = build_pipeline(device, &library, KERNEL_TOP_K_SAMPLE_F16)?;
        let pipeline_build_candidates_f32 = build_pipeline(device, &library, KERNEL_BUILD_CANDIDATES_F32)?;
        let pipeline_build_candidates_f16 = build_pipeline(device, &library, KERNEL_BUILD_CANDIDATES_F16)?;
        let pipeline_verify_candidates_f32 = build_pipeline(device, &library, KERNEL_VERIFY_CANDIDATES_F32)?;
        let pipeline_verify_candidates_f16 = build_pipeline(device, &library, KERNEL_VERIFY_CANDIDATES_F16)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_head_forward_f32,
            pipeline_head_forward_f16,
            pipeline_top_k_sample_f32,
            pipeline_top_k_sample_f16,
            pipeline_build_candidates_f32,
            pipeline_build_candidates_f16,
            pipeline_verify_candidates_f32,
            pipeline_verify_candidates_f16,
        })
    }

    /// Forward pass through all Medusa heads (f32).
    ///
    /// # Arguments
    /// * `hidden_states` - Hidden states: [batch, hidden_dim]
    /// * `head_weights` - Medusa head weights: [num_heads, hidden_dim, vocab_size]
    ///
    /// # Returns
    /// Logits from all heads: [batch, num_heads, vocab_size]
    pub fn head_forward_f32(
        &self,
        hidden_states: &Buffer,
        head_weights: &Buffer,
        batch_size: usize,
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
    ) -> Result<Buffer, MedusaError> {
        self.head_forward_impl(
            hidden_states, head_weights,
            batch_size, hidden_dim, vocab_size, num_heads,
            &self.pipeline_head_forward_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Forward pass through all Medusa heads (f16).
    pub fn head_forward_f16(
        &self,
        hidden_states: &Buffer,
        head_weights: &Buffer,
        batch_size: usize,
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
    ) -> Result<Buffer, MedusaError> {
        self.head_forward_impl(
            hidden_states, head_weights,
            batch_size, hidden_dim, vocab_size, num_heads,
            &self.pipeline_head_forward_f16,
            mem::size_of::<u16>(),
        )
    }

    fn head_forward_impl(
        &self,
        hidden_states: &Buffer,
        head_weights: &Buffer,
        batch_size: usize,
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, MedusaError> {
        let output_elements = batch_size * num_heads * vocab_size;
        let output_bytes = (output_elements * element_size) as u64;

        let params = HeadForwardParams {
            batch_size: batch_size as u32,
            hidden_dim: hidden_dim as u32,
            vocab_size: vocab_size as u32,
            num_heads: num_heads as u32,
        };

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(hidden_states), 0);
        encoder.set_buffer(1, Some(head_weights), 0);
        encoder.set_buffer(2, Some(&output), 0);

        let params_size = mem::size_of::<HeadForwardParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * num_heads) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Top-k sampling from Medusa head logits (f32).
    ///
    /// # Returns
    /// (tokens, probs): Sampled token IDs and their probabilities
    pub fn top_k_sample_f32(
        &self,
        head_logits: &Buffer,
        batch_size: usize,
        vocab_size: usize,
        num_heads: usize,
        top_k: usize,
        temperature: f32,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        self.top_k_sample_impl(
            head_logits,
            batch_size, vocab_size, num_heads, top_k, temperature,
            &self.pipeline_top_k_sample_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Top-k sampling from Medusa head logits (f16).
    pub fn top_k_sample_f16(
        &self,
        head_logits: &Buffer,
        batch_size: usize,
        vocab_size: usize,
        num_heads: usize,
        top_k: usize,
        temperature: f32,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        self.top_k_sample_impl(
            head_logits,
            batch_size, vocab_size, num_heads, top_k, temperature,
            &self.pipeline_top_k_sample_f16,
            mem::size_of::<u16>(),
        )
    }

    fn top_k_sample_impl(
        &self,
        head_logits: &Buffer,
        batch_size: usize,
        vocab_size: usize,
        num_heads: usize,
        top_k: usize,
        temperature: f32,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        let params = TopKSampleParams {
            batch_size: batch_size as u32,
            vocab_size: vocab_size as u32,
            num_heads: num_heads as u32,
            top_k: top_k as u32,
            temperature,
            _pad: [0; 3],
        };

        let num_samples = batch_size * num_heads * top_k;
        let tokens_bytes = (num_samples * mem::size_of::<u32>()) as u64;
        let probs_bytes = (num_samples * element_size) as u64;

        let tokens = self.device.new_buffer(tokens_bytes, MTLResourceOptions::StorageModeShared);
        let probs = self.device.new_buffer(probs_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(head_logits), 0);
        encoder.set_buffer(1, Some(&tokens), 0);
        encoder.set_buffer(2, Some(&probs), 0);

        let params_size = mem::size_of::<TopKSampleParams>() as u64;
        encoder.set_bytes(3, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * num_heads) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((tokens, probs))
    }

    /// Build candidate tree from sampled tokens (f32).
    ///
    /// # Returns
    /// (candidates, tree_indices): Candidate sequences and their tree structure
    pub fn build_candidates_f32(
        &self,
        sampled_tokens: &Buffer,
        sampled_probs: &Buffer,
        batch_size: usize,
        num_heads: usize,
        top_k: usize,
        max_candidates: usize,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        self.build_candidates_impl(
            sampled_tokens, sampled_probs,
            batch_size, num_heads, top_k, max_candidates,
            &self.pipeline_build_candidates_f32,
        )
    }

    /// Build candidate tree from sampled tokens (f16).
    pub fn build_candidates_f16(
        &self,
        sampled_tokens: &Buffer,
        sampled_probs: &Buffer,
        batch_size: usize,
        num_heads: usize,
        top_k: usize,
        max_candidates: usize,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        self.build_candidates_impl(
            sampled_tokens, sampled_probs,
            batch_size, num_heads, top_k, max_candidates,
            &self.pipeline_build_candidates_f16,
        )
    }

    fn build_candidates_impl(
        &self,
        sampled_tokens: &Buffer,
        sampled_probs: &Buffer,
        batch_size: usize,
        num_heads: usize,
        top_k: usize,
        max_candidates: usize,
        pipeline: &ComputePipelineState,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        let params = BuildCandidatesParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            top_k: top_k as u32,
            max_candidates: max_candidates as u32,
        };

        // Output: candidates [batch, max_candidates, num_heads] and tree [batch, max_candidates]
        let candidates_bytes = (batch_size * max_candidates * num_heads * mem::size_of::<u32>()) as u64;
        let tree_bytes = (batch_size * max_candidates * mem::size_of::<i32>()) as u64;

        let candidates = self.device.new_buffer(candidates_bytes, MTLResourceOptions::StorageModeShared);
        let tree_indices = self.device.new_buffer(tree_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(sampled_tokens), 0);
        encoder.set_buffer(1, Some(sampled_probs), 0);
        encoder.set_buffer(2, Some(&candidates), 0);
        encoder.set_buffer(3, Some(&tree_indices), 0);

        let params_size = mem::size_of::<BuildCandidatesParams>() as u64;
        encoder.set_bytes(4, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((candidates, tree_indices))
    }

    /// Verify candidates against target model logits (f32).
    ///
    /// # Returns
    /// (accepted_mask, accepted_count): Which candidates are accepted
    pub fn verify_candidates_f32(
        &self,
        candidates: &Buffer,
        target_logits: &Buffer,
        batch_size: usize,
        num_candidates: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        self.verify_candidates_impl(
            candidates, target_logits,
            batch_size, num_candidates, seq_len, vocab_size,
            &self.pipeline_verify_candidates_f32,
        )
    }

    /// Verify candidates against target model logits (f16).
    pub fn verify_candidates_f16(
        &self,
        candidates: &Buffer,
        target_logits: &Buffer,
        batch_size: usize,
        num_candidates: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        self.verify_candidates_impl(
            candidates, target_logits,
            batch_size, num_candidates, seq_len, vocab_size,
            &self.pipeline_verify_candidates_f16,
        )
    }

    fn verify_candidates_impl(
        &self,
        candidates: &Buffer,
        target_logits: &Buffer,
        batch_size: usize,
        num_candidates: usize,
        seq_len: usize,
        vocab_size: usize,
        pipeline: &ComputePipelineState,
    ) -> Result<(Buffer, Buffer), MedusaError> {
        let params = VerifyCandidatesParams {
            batch_size: batch_size as u32,
            num_candidates: num_candidates as u32,
            seq_len: seq_len as u32,
            vocab_size: vocab_size as u32,
        };

        let accepted_mask_bytes = (batch_size * num_candidates * mem::size_of::<u32>()) as u64;
        let accepted_count_bytes = (batch_size * mem::size_of::<u32>()) as u64;

        let accepted_mask = self.device.new_buffer(accepted_mask_bytes, MTLResourceOptions::StorageModeShared);
        let accepted_count = self.device.new_buffer(accepted_count_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(candidates), 0);
        encoder.set_buffer(1, Some(target_logits), 0);
        encoder.set_buffer(2, Some(&accepted_mask), 0);
        encoder.set_buffer(3, Some(&accepted_count), 0);

        let params_size = mem::size_of::<VerifyCandidatesParams>() as u64;
        encoder.set_bytes(4, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((accepted_mask, accepted_count))
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, MedusaError> {
    MEDUSA_METALLIB.load(device).map_err(MedusaError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, MedusaError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| MedusaError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MedusaError::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
