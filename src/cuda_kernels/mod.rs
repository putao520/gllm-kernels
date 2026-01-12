//! CUDA kernel integrations for GPU-accelerated attention.

#[cfg(feature = "cuda-kernel")]
pub mod flash_attn;

#[cfg(feature = "cuda-kernel")]
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel, OptimizedCudaAttention};
