//! CUDA DeFT/Talon Flash Tree-attention kernel.
//!
//! Based on DeFT and Talon papers for tree-structured attention:
//! - Flash attention optimizations for tree structures
//! - Speculative decoding tree verification
//! - Cache-aware tree traversal
//!
//! # SM-Aware PTX Loading
//!
//! - SM 61 (Pascal): GTX 1060/1070/1080
//! - SM 80 (Ampere): A100, RTX 30 series and higher
//!
//! 🚨 **Fat Binary Only**: NO runtime compilation fallback.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use crate::cuda_kernels::ptx_loader::{PtxCollection, PtxLoadError};

// Kernel function names
const KERNEL_TREE_ATTENTION_F32: &str = "flash_tree_attention_f32";
const KERNEL_TREE_ATTENTION_F16: &str = "flash_tree_attention_f16";
const KERNEL_VERIFY_TREE_F32: &str = "flash_tree_verify_f32";
const KERNEL_VERIFY_TREE_F16: &str = "flash_tree_verify_f16";
const KERNEL_BUILD_TREE_MASK_F32: &str = "flash_tree_build_mask_f32";
const KERNEL_BUILD_TREE_MASK_F16: &str = "flash_tree_build_mask_f16";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for Flash Tree-attention kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static FLASH_TREE_ATTN_PTX: PtxCollection = PtxCollection {
    kernel_name: "flash_tree_attn",
    ptx_versions: &[
        (61, include_str!("kernels/flash_tree_attn_sm61.ptx")),
        (80, include_str!("kernels/flash_tree_attn.ptx")),
    ],
};

/// Errors surfaced by the CUDA Flash Tree-attention kernels.
#[derive(Debug)]
pub enum FlashTreeAttnError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
    /// Tree structure error.
    TreeStructure(String),
}

impl fmt::Display for FlashTreeAttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
            Self::TreeStructure(msg) => write!(f, "Tree structure error: {msg}"),
        }
    }
}

impl std::error::Error for FlashTreeAttnError {}

impl From<DriverError> for FlashTreeAttnError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for FlashTreeAttnError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// Flash Tree-attention CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Tree-structured attention computation
/// - Speculative decoding tree verification
/// - Tree mask generation
pub struct FlashTreeAttnKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Tree attention kernels
    kernel_tree_attention_f32: CudaFunction,
    kernel_tree_attention_f16: CudaFunction,
    // Tree verification kernels
    kernel_verify_tree_f32: CudaFunction,
    kernel_verify_tree_f16: CudaFunction,
    // Tree mask building kernels
    kernel_build_tree_mask_f32: CudaFunction,
    kernel_build_tree_mask_f16: CudaFunction,
}

impl FlashTreeAttnKernel {
    /// Load Flash Tree-attention kernel module on the given device.
    ///
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, FlashTreeAttnError> {
        let ptx = FLASH_TREE_ATTN_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_tree_attention_f32 = module
            .load_function(KERNEL_TREE_ATTENTION_F32)
            .map_err(|_| FlashTreeAttnError::KernelMissing(KERNEL_TREE_ATTENTION_F32))?;
        let kernel_tree_attention_f16 = module
            .load_function(KERNEL_TREE_ATTENTION_F16)
            .map_err(|_| FlashTreeAttnError::KernelMissing(KERNEL_TREE_ATTENTION_F16))?;
        let kernel_verify_tree_f32 = module
            .load_function(KERNEL_VERIFY_TREE_F32)
            .map_err(|_| FlashTreeAttnError::KernelMissing(KERNEL_VERIFY_TREE_F32))?;
        let kernel_verify_tree_f16 = module
            .load_function(KERNEL_VERIFY_TREE_F16)
            .map_err(|_| FlashTreeAttnError::KernelMissing(KERNEL_VERIFY_TREE_F16))?;
        let kernel_build_tree_mask_f32 = module
            .load_function(KERNEL_BUILD_TREE_MASK_F32)
            .map_err(|_| FlashTreeAttnError::KernelMissing(KERNEL_BUILD_TREE_MASK_F32))?;
        let kernel_build_tree_mask_f16 = module
            .load_function(KERNEL_BUILD_TREE_MASK_F16)
            .map_err(|_| FlashTreeAttnError::KernelMissing(KERNEL_BUILD_TREE_MASK_F16))?;

        Ok(Self {
            module,
            kernel_tree_attention_f32,
            kernel_tree_attention_f16,
            kernel_verify_tree_f32,
            kernel_verify_tree_f16,
            kernel_build_tree_mask_f32,
            kernel_build_tree_mask_f16,
        })
    }

    /// Compute tree-structured flash attention (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `query` - Query tensor [batch * num_heads * tree_size * head_dim]
    /// * `key` - Key tensor [batch * num_heads * (prefix_len + tree_size) * head_dim]
    /// * `value` - Value tensor [batch * num_heads * (prefix_len + tree_size) * head_dim]
    /// * `tree_mask` - Tree attention mask [tree_size * (prefix_len + tree_size)]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `tree_size` - Number of tree nodes
    /// * `prefix_len` - Length of prefix context
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    /// Output tensor: [batch * num_heads * tree_size * head_dim]
    pub fn tree_attention_f32(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        tree_mask: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        tree_size: usize,
        prefix_len: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, FlashTreeAttnError> {
        let output_size = batch_size * num_heads * tree_size * head_dim;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        let total_work = batch_size * num_heads * tree_size;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (head_dim * 2 * std::mem::size_of::<f32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let tree_i32 = tree_size as i32;
        let prefix_i32 = prefix_len as i32;
        let head_dim_i32 = head_dim as i32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_tree_attention_f32);
            builder.arg(query);
            builder.arg(key);
            builder.arg(value);
            builder.arg(tree_mask);
            builder.arg(&mut output);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&tree_i32);
            builder.arg(&prefix_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&scale);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Compute tree-structured flash attention (f16).
    pub fn tree_attention_f16(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<f16>,
        key: &CudaSlice<f16>,
        value: &CudaSlice<f16>,
        tree_mask: &CudaSlice<i32>,
        batch_size: usize,
        num_heads: usize,
        tree_size: usize,
        prefix_len: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f16>, FlashTreeAttnError> {
        let output_size = batch_size * num_heads * tree_size * head_dim;
        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_size)?;

        let total_work = batch_size * num_heads * tree_size;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (head_dim * 2 * std::mem::size_of::<f16>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let tree_i32 = tree_size as i32;
        let prefix_i32 = prefix_len as i32;
        let head_dim_i32 = head_dim as i32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_tree_attention_f16);
            builder.arg(query);
            builder.arg(key);
            builder.arg(value);
            builder.arg(tree_mask);
            builder.arg(&mut output);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&tree_i32);
            builder.arg(&prefix_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&scale);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Verify tree structure and find accepted tokens (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `draft_tokens` - Draft token IDs [batch * tree_size]
    /// * `target_probs` - Target model probabilities [batch * tree_size * vocab_size]
    /// * `draft_probs` - Draft model probabilities [batch * tree_size * vocab_size]
    /// * `parent_indices` - Parent node indices [tree_size]
    /// * `batch_size` - Batch size
    /// * `tree_size` - Tree size
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Tuple of (accepted_mask [batch * tree_size], accepted_count [batch])
    pub fn verify_tree_f32(
        &self,
        stream: &Arc<CudaStream>,
        draft_tokens: &CudaSlice<i32>,
        target_probs: &CudaSlice<f32>,
        draft_probs: &CudaSlice<f32>,
        parent_indices: &CudaSlice<i32>,
        batch_size: usize,
        tree_size: usize,
        vocab_size: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>), FlashTreeAttnError> {
        let mut accepted_mask: CudaSlice<i32> = stream.alloc_zeros(batch_size * tree_size)?;
        let mut accepted_count: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (tree_size * std::mem::size_of::<i32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let tree_i32 = tree_size as i32;
        let vocab_i32 = vocab_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_verify_tree_f32);
            builder.arg(draft_tokens);
            builder.arg(target_probs);
            builder.arg(draft_probs);
            builder.arg(parent_indices);
            builder.arg(&mut accepted_mask);
            builder.arg(&mut accepted_count);
            builder.arg(&batch_i32);
            builder.arg(&tree_i32);
            builder.arg(&vocab_i32);
            builder.launch(cfg)?;
        }

        Ok((accepted_mask, accepted_count))
    }

    /// Verify tree structure and find accepted tokens (f16).
    pub fn verify_tree_f16(
        &self,
        stream: &Arc<CudaStream>,
        draft_tokens: &CudaSlice<i32>,
        target_probs: &CudaSlice<f16>,
        draft_probs: &CudaSlice<f16>,
        parent_indices: &CudaSlice<i32>,
        batch_size: usize,
        tree_size: usize,
        vocab_size: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>), FlashTreeAttnError> {
        let mut accepted_mask: CudaSlice<i32> = stream.alloc_zeros(batch_size * tree_size)?;
        let mut accepted_count: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (tree_size * std::mem::size_of::<i32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let tree_i32 = tree_size as i32;
        let vocab_i32 = vocab_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_verify_tree_f16);
            builder.arg(draft_tokens);
            builder.arg(target_probs);
            builder.arg(draft_probs);
            builder.arg(parent_indices);
            builder.arg(&mut accepted_mask);
            builder.arg(&mut accepted_count);
            builder.arg(&batch_i32);
            builder.arg(&tree_i32);
            builder.arg(&vocab_i32);
            builder.launch(cfg)?;
        }

        Ok((accepted_mask, accepted_count))
    }

    /// Build tree attention mask from parent indices (f32/f16 agnostic).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `parent_indices` - Parent node indices [tree_size], -1 for root
    /// * `tree_size` - Tree size
    /// * `prefix_len` - Prefix context length
    ///
    /// # Returns
    /// Tree mask: [tree_size * (prefix_len + tree_size)]
    pub fn build_tree_mask(
        &self,
        stream: &Arc<CudaStream>,
        parent_indices: &CudaSlice<i32>,
        tree_size: usize,
        prefix_len: usize,
    ) -> Result<CudaSlice<i32>, FlashTreeAttnError> {
        let total_len = prefix_len + tree_size;
        let mask_size = tree_size * total_len;
        let mut mask: CudaSlice<i32> = stream.alloc_zeros(mask_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (tree_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let tree_i32 = tree_size as i32;
        let prefix_i32 = prefix_len as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_build_tree_mask_f32);
            builder.arg(parent_indices);
            builder.arg(&mut mask);
            builder.arg(&tree_i32);
            builder.arg(&prefix_i32);
            builder.launch(cfg)?;
        }

        Ok(mask)
    }
}

/// Configuration for Flash Tree-attention CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct FlashTreeAttnCudaConfig {
    /// Maximum tree depth.
    pub max_tree_depth: usize,
    /// Maximum tree width (branching factor).
    pub max_tree_width: usize,
    /// Maximum tree size (total nodes).
    pub max_tree_size: usize,
    /// Temperature for acceptance probability.
    pub temperature: f32,
}

impl Default for FlashTreeAttnCudaConfig {
    fn default() -> Self {
        Self {
            max_tree_depth: 6,
            max_tree_width: 4,
            max_tree_size: 64,
            temperature: 1.0,
        }
    }
}

impl FlashTreeAttnCudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), FlashTreeAttnError> {
        if self.max_tree_depth == 0 {
            return Err(FlashTreeAttnError::InvalidConfig(
                "max_tree_depth must be > 0".into(),
            ));
        }
        if self.max_tree_width == 0 {
            return Err(FlashTreeAttnError::InvalidConfig(
                "max_tree_width must be > 0".into(),
            ));
        }
        if self.max_tree_size == 0 {
            return Err(FlashTreeAttnError::InvalidConfig(
                "max_tree_size must be > 0".into(),
            ));
        }
        if self.temperature <= 0.0 {
            return Err(FlashTreeAttnError::InvalidConfig(
                "temperature must be > 0".into(),
            ));
        }
        Ok(())
    }
}
