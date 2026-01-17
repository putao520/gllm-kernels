//! CUDA Medusa Heads auxiliary generation kernel.
//!
//! Implements Medusa speculative decoding:
//! - Multiple auxiliary heads for parallel token prediction
//! - Tree-based candidate generation
//! - Efficient batch verification
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
const KERNEL_HEAD_FORWARD_F32: &str = "medusa_head_forward_f32";
const KERNEL_HEAD_FORWARD_F16: &str = "medusa_head_forward_f16";
const KERNEL_TOP_K_SAMPLE_F32: &str = "medusa_top_k_sample_f32";
const KERNEL_TOP_K_SAMPLE_F16: &str = "medusa_top_k_sample_f16";
const KERNEL_BUILD_CANDIDATES_F32: &str = "medusa_build_candidates_f32";
const KERNEL_BUILD_CANDIDATES_F16: &str = "medusa_build_candidates_f16";
const KERNEL_VERIFY_CANDIDATES_F32: &str = "medusa_verify_candidates_f32";
const KERNEL_VERIFY_CANDIDATES_F16: &str = "medusa_verify_candidates_f16";

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// SM-aware PTX collection for Medusa kernel.
/// 🚨 **Fat Binary Only**: All PTX precompiled and embedded, no runtime compilation.
static MEDUSA_PTX: PtxCollection = PtxCollection {
    kernel_name: "medusa",
    ptx_versions: &[
        (61, include_str!("kernels/medusa_sm61.ptx")),
        (80, include_str!("kernels/medusa.ptx")),
    ],
};

/// Errors surfaced by the CUDA Medusa kernels.
#[derive(Debug)]
pub enum MedusaError {
    /// CUDA driver error.
    Driver(DriverError),
    /// Invalid configuration or parameters.
    InvalidConfig(String),
    /// Missing kernel entry point.
    KernelMissing(&'static str),
    /// PTX loading error.
    PtxLoad(PtxLoadError),
    /// Candidate generation error.
    CandidateGeneration(String),
}

impl fmt::Display for MedusaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::KernelMissing(name) => write!(f, "Kernel not found: {name}"),
            Self::PtxLoad(err) => write!(f, "PTX loading error: {err}"),
            Self::CandidateGeneration(msg) => write!(f, "Candidate generation error: {msg}"),
        }
    }
}

impl std::error::Error for MedusaError {}

impl From<DriverError> for MedusaError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<PtxLoadError> for MedusaError {
    fn from(err: PtxLoadError) -> Self {
        Self::PtxLoad(err)
    }
}

/// Medusa CUDA kernel wrapper.
///
/// Provides GPU-accelerated operations for:
/// - Medusa head forward pass
/// - Top-K sampling from each head
/// - Candidate tree construction
/// - Candidate verification
pub struct MedusaKernel {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    // Head forward kernels
    kernel_head_forward_f32: CudaFunction,
    kernel_head_forward_f16: CudaFunction,
    // Top-K sampling kernels
    kernel_top_k_f32: CudaFunction,
    kernel_top_k_f16: CudaFunction,
    // Candidate building kernels
    kernel_build_candidates_f32: CudaFunction,
    kernel_build_candidates_f16: CudaFunction,
    // Verification kernels
    kernel_verify_candidates_f32: CudaFunction,
    kernel_verify_candidates_f16: CudaFunction,
}

impl MedusaKernel {
    /// Load Medusa kernel module on the given device.
    ///
    /// 🚨 **Fat Binary Only**: No runtime compilation fallback.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, MedusaError> {
        let ptx = MEDUSA_PTX.load(ctx)?;
        let module = ctx.load_module(ptx)?;

        let kernel_head_forward_f32 = module
            .load_function(KERNEL_HEAD_FORWARD_F32)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_HEAD_FORWARD_F32))?;
        let kernel_head_forward_f16 = module
            .load_function(KERNEL_HEAD_FORWARD_F16)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_HEAD_FORWARD_F16))?;
        let kernel_top_k_f32 = module
            .load_function(KERNEL_TOP_K_SAMPLE_F32)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_TOP_K_SAMPLE_F32))?;
        let kernel_top_k_f16 = module
            .load_function(KERNEL_TOP_K_SAMPLE_F16)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_TOP_K_SAMPLE_F16))?;
        let kernel_build_candidates_f32 = module
            .load_function(KERNEL_BUILD_CANDIDATES_F32)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_BUILD_CANDIDATES_F32))?;
        let kernel_build_candidates_f16 = module
            .load_function(KERNEL_BUILD_CANDIDATES_F16)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_BUILD_CANDIDATES_F16))?;
        let kernel_verify_candidates_f32 = module
            .load_function(KERNEL_VERIFY_CANDIDATES_F32)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_VERIFY_CANDIDATES_F32))?;
        let kernel_verify_candidates_f16 = module
            .load_function(KERNEL_VERIFY_CANDIDATES_F16)
            .map_err(|_| MedusaError::KernelMissing(KERNEL_VERIFY_CANDIDATES_F16))?;

        Ok(Self {
            module,
            kernel_head_forward_f32,
            kernel_head_forward_f16,
            kernel_top_k_f32,
            kernel_top_k_f16,
            kernel_build_candidates_f32,
            kernel_build_candidates_f16,
            kernel_verify_candidates_f32,
            kernel_verify_candidates_f16,
        })
    }

    /// Forward pass through Medusa heads (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `hidden_states` - Hidden states [batch * hidden_dim]
    /// * `head_weights` - Medusa head weights [num_heads * hidden_dim * vocab_size]
    /// * `head_biases` - Medusa head biases [num_heads * vocab_size] (optional, can be zero-sized)
    /// * `batch_size` - Batch size
    /// * `hidden_dim` - Hidden dimension
    /// * `vocab_size` - Vocabulary size
    /// * `num_heads` - Number of Medusa heads
    ///
    /// # Returns
    /// Logits for each head: [batch * num_heads * vocab_size]
    pub fn head_forward_f32(
        &self,
        stream: &Arc<CudaStream>,
        hidden_states: &CudaSlice<f32>,
        head_weights: &CudaSlice<f32>,
        head_biases: &CudaSlice<f32>,
        batch_size: usize,
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
    ) -> Result<CudaSlice<f32>, MedusaError> {
        let output_size = batch_size * num_heads * vocab_size;
        let mut output: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        // Launch one block per (batch, head) pair
        let total_work = batch_size * num_heads;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let hidden_i32 = hidden_dim as i32;
        let vocab_i32 = vocab_size as i32;
        let heads_i32 = num_heads as i32;
        let has_bias: i32 = if head_biases.len() > 0 { 1 } else { 0 };

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_head_forward_f32);
            builder.arg(hidden_states);
            builder.arg(head_weights);
            builder.arg(head_biases);
            builder.arg(&mut output);
            builder.arg(&batch_i32);
            builder.arg(&hidden_i32);
            builder.arg(&vocab_i32);
            builder.arg(&heads_i32);
            builder.arg(&has_bias);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Forward pass through Medusa heads (f16).
    pub fn head_forward_f16(
        &self,
        stream: &Arc<CudaStream>,
        hidden_states: &CudaSlice<f16>,
        head_weights: &CudaSlice<f16>,
        head_biases: &CudaSlice<f16>,
        batch_size: usize,
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
    ) -> Result<CudaSlice<f16>, MedusaError> {
        let output_size = batch_size * num_heads * vocab_size;
        let mut output: CudaSlice<f16> = stream.alloc_zeros(output_size)?;

        let total_work = batch_size * num_heads;
        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (total_work + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let batch_i32 = batch_size as i32;
        let hidden_i32 = hidden_dim as i32;
        let vocab_i32 = vocab_size as i32;
        let heads_i32 = num_heads as i32;
        let has_bias: i32 = if head_biases.len() > 0 { 1 } else { 0 };

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_head_forward_f16);
            builder.arg(hidden_states);
            builder.arg(head_weights);
            builder.arg(head_biases);
            builder.arg(&mut output);
            builder.arg(&batch_i32);
            builder.arg(&hidden_i32);
            builder.arg(&vocab_i32);
            builder.arg(&heads_i32);
            builder.arg(&has_bias);
            builder.launch(cfg)?;
        }

        Ok(output)
    }

    /// Top-K sampling from Medusa head logits (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `logits` - Head logits [batch * num_heads * vocab_size]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of Medusa heads
    /// * `vocab_size` - Vocabulary size
    /// * `top_k` - Number of top candidates per head
    /// * `temperature` - Sampling temperature
    ///
    /// # Returns
    /// Tuple of (top_k_tokens [batch * num_heads * top_k], top_k_probs [batch * num_heads * top_k])
    pub fn top_k_sample_f32(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        vocab_size: usize,
        top_k: usize,
        temperature: f32,
    ) -> Result<(CudaSlice<i32>, CudaSlice<f32>), MedusaError> {
        let output_size = batch_size * num_heads * top_k;
        let mut top_k_tokens: CudaSlice<i32> = stream.alloc_zeros(output_size)?;
        let mut top_k_probs: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        // One block per (batch, head) pair
        let cfg = LaunchConfig {
            grid_dim: ((batch_size * num_heads) as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE.min(vocab_size as u32), 1, 1),
            shared_mem_bytes: (top_k * 2 * std::mem::size_of::<f32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let vocab_i32 = vocab_size as i32;
        let topk_i32 = top_k as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_top_k_f32);
            builder.arg(logits);
            builder.arg(&mut top_k_tokens);
            builder.arg(&mut top_k_probs);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&vocab_i32);
            builder.arg(&topk_i32);
            builder.arg(&temperature);
            builder.launch(cfg)?;
        }

        Ok((top_k_tokens, top_k_probs))
    }

    /// Top-K sampling from Medusa head logits (f16).
    pub fn top_k_sample_f16(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f16>,
        batch_size: usize,
        num_heads: usize,
        vocab_size: usize,
        top_k: usize,
        temperature: f32,
    ) -> Result<(CudaSlice<i32>, CudaSlice<f32>), MedusaError> {
        let output_size = batch_size * num_heads * top_k;
        let mut top_k_tokens: CudaSlice<i32> = stream.alloc_zeros(output_size)?;
        let mut top_k_probs: CudaSlice<f32> = stream.alloc_zeros(output_size)?;

        let cfg = LaunchConfig {
            grid_dim: ((batch_size * num_heads) as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE.min(vocab_size as u32), 1, 1),
            shared_mem_bytes: (top_k * 2 * std::mem::size_of::<f32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let vocab_i32 = vocab_size as i32;
        let topk_i32 = top_k as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_top_k_f16);
            builder.arg(logits);
            builder.arg(&mut top_k_tokens);
            builder.arg(&mut top_k_probs);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&vocab_i32);
            builder.arg(&topk_i32);
            builder.arg(&temperature);
            builder.launch(cfg)?;
        }

        Ok((top_k_tokens, top_k_probs))
    }

    /// Build candidate sequences from top-K tokens.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `top_k_tokens` - Top-K tokens from each head [batch * num_heads * top_k]
    /// * `top_k_probs` - Top-K probabilities [batch * num_heads * top_k]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of Medusa heads
    /// * `top_k` - Number of top candidates per head
    /// * `max_candidates` - Maximum number of candidate sequences
    ///
    /// # Returns
    /// Tuple of (candidate_tokens [batch * max_candidates * num_heads],
    ///           candidate_probs [batch * max_candidates],
    ///           num_candidates [batch])
    pub fn build_candidates_f32(
        &self,
        stream: &Arc<CudaStream>,
        top_k_tokens: &CudaSlice<i32>,
        top_k_probs: &CudaSlice<f32>,
        batch_size: usize,
        num_heads: usize,
        top_k: usize,
        max_candidates: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<f32>, CudaSlice<i32>), MedusaError> {
        let candidate_tokens_size = batch_size * max_candidates * num_heads;
        let candidate_probs_size = batch_size * max_candidates;

        let mut candidate_tokens: CudaSlice<i32> = stream.alloc_zeros(candidate_tokens_size)?;
        let mut candidate_probs: CudaSlice<f32> = stream.alloc_zeros(candidate_probs_size)?;
        let mut num_candidates: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (max_candidates * num_heads * std::mem::size_of::<i32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let topk_i32 = top_k as i32;
        let max_cand_i32 = max_candidates as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_build_candidates_f32);
            builder.arg(top_k_tokens);
            builder.arg(top_k_probs);
            builder.arg(&mut candidate_tokens);
            builder.arg(&mut candidate_probs);
            builder.arg(&mut num_candidates);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&topk_i32);
            builder.arg(&max_cand_i32);
            builder.launch(cfg)?;
        }

        Ok((candidate_tokens, candidate_probs, num_candidates))
    }

    /// Build candidate sequences from top-K tokens (f16).
    pub fn build_candidates_f16(
        &self,
        stream: &Arc<CudaStream>,
        top_k_tokens: &CudaSlice<i32>,
        top_k_probs: &CudaSlice<f32>,  // Probs are always f32
        batch_size: usize,
        num_heads: usize,
        top_k: usize,
        max_candidates: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<f32>, CudaSlice<i32>), MedusaError> {
        // Same as f32 since tokens are i32 and probs are f32
        self.build_candidates_f32(
            stream, top_k_tokens, top_k_probs, batch_size, num_heads, top_k, max_candidates,
        )
    }

    /// Verify candidate sequences against target model (f32).
    ///
    /// # Arguments
    /// * `stream` - CUDA stream
    /// * `candidate_tokens` - Candidate tokens [batch * num_candidates * seq_len]
    /// * `target_probs` - Target model probabilities [batch * num_candidates * seq_len * vocab_size]
    /// * `batch_size` - Batch size
    /// * `num_candidates` - Number of candidates
    /// * `seq_len` - Sequence length (num_heads + 1)
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Tuple of (accepted_length [batch * num_candidates], best_candidate [batch])
    pub fn verify_candidates_f32(
        &self,
        stream: &Arc<CudaStream>,
        candidate_tokens: &CudaSlice<i32>,
        target_probs: &CudaSlice<f32>,
        batch_size: usize,
        num_candidates: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>), MedusaError> {
        let mut accepted_length: CudaSlice<i32> = stream.alloc_zeros(batch_size * num_candidates)?;
        let mut best_candidate: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (num_candidates * std::mem::size_of::<i32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let cand_i32 = num_candidates as i32;
        let seq_i32 = seq_len as i32;
        let vocab_i32 = vocab_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_verify_candidates_f32);
            builder.arg(candidate_tokens);
            builder.arg(target_probs);
            builder.arg(&mut accepted_length);
            builder.arg(&mut best_candidate);
            builder.arg(&batch_i32);
            builder.arg(&cand_i32);
            builder.arg(&seq_i32);
            builder.arg(&vocab_i32);
            builder.launch(cfg)?;
        }

        Ok((accepted_length, best_candidate))
    }

    /// Verify candidate sequences against target model (f16).
    pub fn verify_candidates_f16(
        &self,
        stream: &Arc<CudaStream>,
        candidate_tokens: &CudaSlice<i32>,
        target_probs: &CudaSlice<f16>,
        batch_size: usize,
        num_candidates: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<i32>), MedusaError> {
        let mut accepted_length: CudaSlice<i32> = stream.alloc_zeros(batch_size * num_candidates)?;
        let mut best_candidate: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        let num_threads = DEFAULT_BLOCK_SIZE as usize;
        let num_blocks = (batch_size + num_threads - 1) / num_threads;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (num_candidates * std::mem::size_of::<i32>()) as u32,
        };

        let batch_i32 = batch_size as i32;
        let cand_i32 = num_candidates as i32;
        let seq_i32 = seq_len as i32;
        let vocab_i32 = vocab_size as i32;

        unsafe {
            let mut builder = stream.launch_builder(&self.kernel_verify_candidates_f16);
            builder.arg(candidate_tokens);
            builder.arg(target_probs);
            builder.arg(&mut accepted_length);
            builder.arg(&mut best_candidate);
            builder.arg(&batch_i32);
            builder.arg(&cand_i32);
            builder.arg(&seq_i32);
            builder.arg(&vocab_i32);
            builder.launch(cfg)?;
        }

        Ok((accepted_length, best_candidate))
    }
}

/// Configuration for Medusa CUDA operations.
#[derive(Debug, Clone, Copy)]
pub struct MedusaCudaConfig {
    /// Number of Medusa heads.
    pub num_heads: usize,
    /// Top-K candidates per head.
    pub top_k: usize,
    /// Maximum number of candidate sequences.
    pub max_candidates: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Typical acceptance threshold.
    pub typical_acceptance_threshold: f32,
}

impl Default for MedusaCudaConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            top_k: 10,
            max_candidates: 64,
            temperature: 1.0,
            typical_acceptance_threshold: 0.3,
        }
    }
}

impl MedusaCudaConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), MedusaError> {
        if self.num_heads == 0 {
            return Err(MedusaError::InvalidConfig(
                "num_heads must be > 0".into(),
            ));
        }
        if self.top_k == 0 {
            return Err(MedusaError::InvalidConfig("top_k must be > 0".into()));
        }
        if self.max_candidates == 0 {
            return Err(MedusaError::InvalidConfig(
                "max_candidates must be > 0".into(),
            ));
        }
        if self.temperature <= 0.0 {
            return Err(MedusaError::InvalidConfig(
                "temperature must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Get the maximum sequence length from Medusa heads.
    pub fn max_draft_len(&self) -> usize {
        self.num_heads
    }

    /// Compute theoretical maximum candidates (top_k^num_heads).
    pub fn theoretical_max_candidates(&self) -> usize {
        self.top_k.pow(self.num_heads as u32)
    }
}
