//! CUDA kernel integrations for GPU-accelerated attention.
//!
//! All kernels are always compiled. Runtime detection determines whether
//! CUDA is available. Kernels use cudarc's dynamic-loading feature to
//! load CUDA libraries at runtime.
//!
//! # SM-Aware PTX Loading
//!
//! This module automatically selects the best PTX binary for the detected GPU:
//! - SM 61 (Pascal): GTX 1060/1070/1080
//! - SM 75 (Turing): RTX 2060/2070/2080
//! - SM 80/86 (Ampere): RTX 30 series, A100
//! - SM 89 (Ada): RTX 40 series
//! - SM 90 (Hopper): H100, H200
//!
//! If no matching PTX is found, NVRTC runtime compilation is used as fallback.

pub mod embedding_ops;
pub mod ptx_loader;
pub mod flash_attn;
pub mod fused_attention;
pub mod online_softmax;
pub mod paged_attn;
pub mod selective_scan;

pub use ptx_loader::{PtxCollection, PtxLoadError, detect_sm_version, find_best_sm_match};

pub use embedding_ops::{EmbeddingOpsError as CudaEmbeddingOpsError, EmbeddingOpsKernel as CudaEmbeddingOpsKernel};
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel, OptimizedCudaAttention};
pub use fused_attention::{FusedQKVAttentionError, FusedQKVAttentionKernel};
pub use online_softmax::{OnlineSoftmaxError, OnlineSoftmaxKernel, OnlineSoftmaxOutput};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};
pub use selective_scan::{SelectiveScanError, SelectiveScanKernel};
