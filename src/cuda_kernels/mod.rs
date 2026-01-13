//! CUDA kernel integrations for GPU-accelerated attention.

#[cfg(feature = "cuda-kernel")]
pub mod flash_attn;
#[cfg(feature = "fused-kernel")]
pub mod fused_attention;
#[cfg(feature = "softmax-kernel")]
pub mod online_softmax;
#[cfg(feature = "cuda-kernel")]
pub mod paged_attn;
#[cfg(feature = "mamba-kernel")]
pub mod selective_scan;

#[cfg(feature = "cuda-kernel")]
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel, OptimizedCudaAttention};
#[cfg(feature = "fused-kernel")]
pub use fused_attention::{FusedQKVAttentionError, FusedQKVAttentionKernel};
#[cfg(feature = "softmax-kernel")]
pub use online_softmax::{OnlineSoftmaxError, OnlineSoftmaxKernel, OnlineSoftmaxOutput};
#[cfg(feature = "cuda-kernel")]
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};
#[cfg(feature = "mamba-kernel")]
pub use selective_scan::{SelectiveScanError, SelectiveScanKernel};
