//! CUDA kernel integrations for GPU-accelerated attention.
//!
//! All kernels are always compiled. Runtime detection determines whether
//! CUDA is available. Kernels use cudarc's dynamic-loading feature to
//! load CUDA libraries at runtime.
//!
//! # SM-Aware Binary Loading
//!
//! This module automatically selects the best precompiled CUBIN/PTX for the detected GPU:
//! - SM 61 (Pascal): GTX 1060/1070/1080
//! - SM 75 (Turing): RTX 2060/2070/2080
//! - SM 80/86 (Ampere): RTX 30 series, A100
//! - SM 89 (Ada): RTX 40 series
//! - SM 90 (Hopper): H100, H200
//!
//! Fat Binary Only: runtime compilation is disabled.

pub mod embedding_ops;
pub mod binary_loader;
// Backward-compatible module shim.
pub mod ptx_loader {
    pub use super::binary_loader::*;
}
pub mod flash_attn;
pub mod fused_attention;
pub mod online_softmax;
pub mod paged_attn;
pub mod rms_norm;
pub mod selective_scan;
pub mod silu;
pub mod linear;
pub mod elementwise;
pub mod pooling;
pub mod rope;
pub mod sampling;
pub mod moe_route;

// Inference optimization kernels (REQ-OP-008 ~ REQ-OP-015)
pub mod eagle3;
pub mod spec_ee;
pub mod flash_tree_attn;
pub mod int2_quantizer;
pub mod quantized;
pub mod evic_press;
pub mod medusa;
pub mod prompt_cache;
pub mod chunked_prefill;

pub use binary_loader::{PtxCollection, PtxLoadError, detect_sm_version, find_best_sm_match};

pub use embedding_ops::{EmbeddingOpsError as CudaEmbeddingOpsError, EmbeddingOpsKernel as CudaEmbeddingOpsKernel};
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel, OptimizedCudaAttention};
pub use fused_attention::{FusedQKVAttentionError, FusedQKVAttentionKernel};
pub use online_softmax::{OnlineSoftmaxError, OnlineSoftmaxKernel, OnlineSoftmaxOutput};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};
pub use rms_norm::{RmsNormError, RmsNormKernel};
pub use selective_scan::{SelectiveScanError, SelectiveScanKernel};
pub use silu::{CudaSilu, SiluError};
pub use linear::CudaLinear;
pub use elementwise::{CudaElementwiseKernel, ElementwiseError};
pub use pooling::{CudaPoolingKernel, PoolingError};
pub use rope::{CudaRoPEKernel, RoPEKernelError};
pub use sampling::{CudaSamplingKernel, SamplingKernelError};
pub use moe_route::{CudaMoeRouteKernel, MoeRouteKernelError, CudaMoeRouteResult};

// Inference optimization kernels exports
pub use eagle3::{Eagle3Error, Eagle3Kernel, Eagle3CudaConfig as Eagle3Config};
pub use spec_ee::{SpecEEError, SpecEEKernel, SpecEECudaConfig as SpecEEConfig};
pub use flash_tree_attn::{FlashTreeAttnError, FlashTreeAttnKernel, FlashTreeAttnCudaConfig as FlashTreeAttnConfig};
pub use int2_quantizer::{Int2QuantizerError, Int2QuantizerKernel, Int2QuantizerCudaConfig as Int2QuantizerConfig};
pub use quantized::{QuantizedDequantError, QuantizedDequantKernel};
pub use evic_press::{EvicPressError, EvicPressKernel, EvicPressCudaConfig as EvicPressConfig, CacheZone};
pub use medusa::{MedusaError, MedusaKernel, MedusaCudaConfig as MedusaConfig};
pub use prompt_cache::{PromptCacheError, PromptCacheKernel, PromptCacheCudaConfig as PromptCacheConfig};
pub use chunked_prefill::{ChunkedPrefillError, ChunkedPrefillKernel, ChunkedPrefillCudaConfig as ChunkedPrefillConfig};
