//! Metal DeFT/Talon Flash Tree-attention kernels.
//!
//! This module provides Metal GPU-accelerated kernels for tree-structured attention:
//! - Tree attention computation for speculative decoding verification
//! - Multi-path attention with tree structure
//! - Efficient tree mask generation
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

const KERNEL_TREE_ATTENTION_F32: &str = "flash_tree_attn_forward_f32";
const KERNEL_TREE_ATTENTION_F16: &str = "flash_tree_attn_forward_f16";
const KERNEL_VERIFY_TREE_F32: &str = "flash_tree_attn_verify_f32";
const KERNEL_VERIFY_TREE_F16: &str = "flash_tree_attn_verify_f16";
const KERNEL_BUILD_TREE_MASK: &str = "flash_tree_attn_build_mask";

/// Metallib collection for Flash Tree-attention kernels.
/// metallib must be precompiled with: ./scripts/compile_metal_kernels.sh
static FLASH_TREE_ATTN_METALLIB: MetallibCollection = MetallibCollection {
    kernel_name: "flash_tree_attn",
    metallib_data: include_bytes!("kernels/flash_tree_attn.metallib"),
};

/// Parameters for tree attention kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TreeAttentionParams {
    batch_size: u32,
    num_heads: u32,
    tree_size: u32,
    context_len: u32,
    head_dim: u32,
    scale: f32,
    _pad: [u32; 2],
}

/// Parameters for verify tree kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct VerifyTreeParams {
    batch_size: u32,
    tree_size: u32,
    vocab_size: u32,
    _pad: u32,
}

/// Parameters for build tree mask kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BuildTreeMaskParams {
    tree_size: u32,
    max_depth: u32,
    _pad: [u32; 2],
}

/// Errors surfaced by the Metal Flash Tree-attention kernels.
#[derive(Debug)]
pub enum FlashTreeAttnError {
    /// Metal framework error.
    Metal(String),
    /// Invalid launch or shape configuration.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// Metallib loading error.
    MetallibLoad(MetallibLoadError),
}

impl fmt::Display for FlashTreeAttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal(msg) => write!(f, "Metal error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::MetallibLoad(err) => write!(f, "Metallib load error: {err}"),
        }
    }
}

impl std::error::Error for FlashTreeAttnError {}

impl From<MetallibLoadError> for FlashTreeAttnError {
    fn from(err: MetallibLoadError) -> Self {
        Self::MetallibLoad(err)
    }
}

/// Configuration for Flash Tree-attention operations.
#[derive(Clone, Debug)]
pub struct FlashTreeAttnConfig {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Block size for tiled computation.
    pub block_size: usize,
    /// Whether to use online softmax.
    pub use_online_softmax: bool,
}

impl Default for FlashTreeAttnConfig {
    fn default() -> Self {
        Self {
            max_depth: 16,
            block_size: 64,
            use_online_softmax: true,
        }
    }
}

impl FlashTreeAttnConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), FlashTreeAttnError> {
        if self.max_depth == 0 {
            return Err(FlashTreeAttnError::InvalidConfig("max_depth must be positive".into()));
        }
        if self.block_size == 0 || (self.block_size & (self.block_size - 1)) != 0 {
            return Err(FlashTreeAttnError::InvalidConfig("block_size must be power of 2".into()));
        }
        Ok(())
    }
}

/// Flash Tree-attention Metal kernel wrapper.
pub struct FlashTreeAttnKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_tree_attn_f32: ComputePipelineState,
    pipeline_tree_attn_f16: ComputePipelineState,
    pipeline_verify_f32: ComputePipelineState,
    pipeline_verify_f16: ComputePipelineState,
    pipeline_build_mask: ComputePipelineState,
}

impl FlashTreeAttnKernel {
    /// Load Flash Tree-attention kernels on the given device.
    pub fn new(device: &Device) -> Result<Self, FlashTreeAttnError> {
        let library = load_library(device)?;

        let pipeline_tree_attn_f32 = build_pipeline(device, &library, KERNEL_TREE_ATTENTION_F32)?;
        let pipeline_tree_attn_f16 = build_pipeline(device, &library, KERNEL_TREE_ATTENTION_F16)?;
        let pipeline_verify_f32 = build_pipeline(device, &library, KERNEL_VERIFY_TREE_F32)?;
        let pipeline_verify_f16 = build_pipeline(device, &library, KERNEL_VERIFY_TREE_F16)?;
        let pipeline_build_mask = build_pipeline(device, &library, KERNEL_BUILD_TREE_MASK)?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline_tree_attn_f32,
            pipeline_tree_attn_f16,
            pipeline_verify_f32,
            pipeline_verify_f16,
            pipeline_build_mask,
        })
    }

    /// Compute tree attention (f32).
    ///
    /// # Arguments
    /// * `q` - Query tensor: [batch, num_heads, tree_size, head_dim]
    /// * `k` - Key tensor: [batch, num_heads, context_len, head_dim]
    /// * `v` - Value tensor: [batch, num_heads, context_len, head_dim]
    /// * `tree_mask` - Tree attention mask: [tree_size, context_len]
    /// * `parent_indices` - Parent index for each tree node: [tree_size]
    pub fn tree_attention_f32(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        tree_mask: &Buffer,
        parent_indices: &Buffer,
        batch_size: usize,
        num_heads: usize,
        tree_size: usize,
        context_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Buffer, FlashTreeAttnError> {
        self.tree_attention_impl(
            q, k, v, tree_mask, parent_indices,
            batch_size, num_heads, tree_size, context_len, head_dim, scale,
            &self.pipeline_tree_attn_f32,
            mem::size_of::<f32>(),
        )
    }

    /// Compute tree attention (f16).
    pub fn tree_attention_f16(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        tree_mask: &Buffer,
        parent_indices: &Buffer,
        batch_size: usize,
        num_heads: usize,
        tree_size: usize,
        context_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Buffer, FlashTreeAttnError> {
        self.tree_attention_impl(
            q, k, v, tree_mask, parent_indices,
            batch_size, num_heads, tree_size, context_len, head_dim, scale,
            &self.pipeline_tree_attn_f16,
            mem::size_of::<u16>(),
        )
    }

    fn tree_attention_impl(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        tree_mask: &Buffer,
        parent_indices: &Buffer,
        batch_size: usize,
        num_heads: usize,
        tree_size: usize,
        context_len: usize,
        head_dim: usize,
        scale: f32,
        pipeline: &ComputePipelineState,
        element_size: usize,
    ) -> Result<Buffer, FlashTreeAttnError> {
        let output_elements = batch_size * num_heads * tree_size * head_dim;
        let output_bytes = (output_elements * element_size) as u64;

        let params = TreeAttentionParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            tree_size: tree_size as u32,
            context_len: context_len as u32,
            head_dim: head_dim as u32,
            scale,
            _pad: [0; 2],
        };

        let output = self.device.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k), 0);
        encoder.set_buffer(2, Some(v), 0);
        encoder.set_buffer(3, Some(tree_mask), 0);
        encoder.set_buffer(4, Some(parent_indices), 0);
        encoder.set_buffer(5, Some(&output), 0);

        let params_size = mem::size_of::<TreeAttentionParams>() as u64;
        encoder.set_bytes(6, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new((batch_size * num_heads * tree_size) as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Verify draft tokens against target logits using tree structure (f32).
    pub fn verify_tree_f32(
        &self,
        draft_tokens: &Buffer,
        target_logits: &Buffer,
        parent_indices: &Buffer,
        batch_size: usize,
        tree_size: usize,
        vocab_size: usize,
    ) -> Result<(Buffer, Buffer), FlashTreeAttnError> {
        self.verify_tree_impl(
            draft_tokens, target_logits, parent_indices,
            batch_size, tree_size, vocab_size,
            &self.pipeline_verify_f32,
        )
    }

    /// Verify draft tokens against target logits using tree structure (f16).
    pub fn verify_tree_f16(
        &self,
        draft_tokens: &Buffer,
        target_logits: &Buffer,
        parent_indices: &Buffer,
        batch_size: usize,
        tree_size: usize,
        vocab_size: usize,
    ) -> Result<(Buffer, Buffer), FlashTreeAttnError> {
        self.verify_tree_impl(
            draft_tokens, target_logits, parent_indices,
            batch_size, tree_size, vocab_size,
            &self.pipeline_verify_f16,
        )
    }

    fn verify_tree_impl(
        &self,
        draft_tokens: &Buffer,
        target_logits: &Buffer,
        parent_indices: &Buffer,
        batch_size: usize,
        tree_size: usize,
        vocab_size: usize,
        pipeline: &ComputePipelineState,
    ) -> Result<(Buffer, Buffer), FlashTreeAttnError> {
        let params = VerifyTreeParams {
            batch_size: batch_size as u32,
            tree_size: tree_size as u32,
            vocab_size: vocab_size as u32,
            _pad: 0,
        };

        // Output: accepted mask [batch, tree_size] and accepted count [batch]
        let accepted_mask_bytes = (batch_size * tree_size * mem::size_of::<u32>()) as u64;
        let accepted_count_bytes = (batch_size * mem::size_of::<u32>()) as u64;

        let accepted_mask = self.device.new_buffer(accepted_mask_bytes, MTLResourceOptions::StorageModeShared);
        let accepted_count = self.device.new_buffer(accepted_count_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(draft_tokens), 0);
        encoder.set_buffer(1, Some(target_logits), 0);
        encoder.set_buffer(2, Some(parent_indices), 0);
        encoder.set_buffer(3, Some(&accepted_mask), 0);
        encoder.set_buffer(4, Some(&accepted_count), 0);

        let params_size = mem::size_of::<VerifyTreeParams>() as u64;
        encoder.set_bytes(5, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_threadgroup = threads_per_threadgroup(pipeline);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((accepted_mask, accepted_count))
    }

    /// Build tree attention mask from parent indices.
    pub fn build_tree_mask(
        &self,
        parent_indices: &Buffer,
        tree_size: usize,
        max_depth: usize,
    ) -> Result<Buffer, FlashTreeAttnError> {
        let params = BuildTreeMaskParams {
            tree_size: tree_size as u32,
            max_depth: max_depth as u32,
            _pad: [0; 2],
        };

        // Output: tree mask [tree_size, tree_size] as u8
        let mask_bytes = (tree_size * tree_size * mem::size_of::<u8>()) as u64;

        let mask = self.device.new_buffer(mask_bytes, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_build_mask);
        encoder.set_buffer(0, Some(parent_indices), 0);
        encoder.set_buffer(1, Some(&mask), 0);

        let params_size = mem::size_of::<BuildTreeMaskParams>() as u64;
        encoder.set_bytes(2, params_size, &params as *const _ as *const c_void);

        let threads_per_grid = MTLSize::new(tree_size as u64, tree_size as u64, 1);
        let threads_per_threadgroup = MTLSize::new(16, 16, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(mask)
    }
}

/// Load Metal library from embedded metallib.
fn load_library(device: &Device) -> Result<Library, FlashTreeAttnError> {
    FLASH_TREE_ATTN_METALLIB.load(device).map_err(FlashTreeAttnError::from)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &'static str,
) -> Result<ComputePipelineState, FlashTreeAttnError> {
    let function = library
        .get_function(name, None)
        .map_err(|_| FlashTreeAttnError::KernelMissing(name))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(FlashTreeAttnError::Metal)
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let mut width = max_threads.min(256);
    if width == 0 {
        width = 1;
    }
    MTLSize::new(width, 1, 1)
}
