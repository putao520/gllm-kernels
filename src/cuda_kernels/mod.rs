//! CUDA kernel integrations for GPU-accelerated attention.
//!
//! All kernels are always compiled. Runtime detection determines whether
//! CUDA is available. Kernels use cudarc's dynamic-loading feature to
//! load CUDA libraries at runtime.

pub mod flash_attn;
pub mod fused_attention;
pub mod online_softmax;
pub mod paged_attn;
pub mod selective_scan;

pub use flash_attn::{FlashAttentionError, FlashAttentionKernel, OptimizedCudaAttention};
pub use fused_attention::{FusedQKVAttentionError, FusedQKVAttentionKernel};
pub use online_softmax::{OnlineSoftmaxError, OnlineSoftmaxKernel, OnlineSoftmaxOutput};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};
pub use selective_scan::{SelectiveScanError, SelectiveScanKernel};
