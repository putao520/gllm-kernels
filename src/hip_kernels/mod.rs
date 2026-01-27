//! ROCm/AMD GPU kernels via HSA Runtime.
//!
//! This module provides GPU kernels for AMD GPUs using the HSA Runtime (ROCr),
//! the low-level driver API that only requires AMD GPU drivers (not full ROCm).
//!
//! ## Architecture
//!
//! ```text
//! Compile time: HIP kernel source → HSACO binary (via hipcc/offline compiler)
//! Runtime: HSA Runtime loads HSACO → GPU executes
//! ```
//!
//! This is analogous to CUDA's driver API (libcuda.so) vs runtime API (libcudart.so).
//!
//! ## Runtime Detection
//!
//! HSA Runtime (libhsa-runtime64.so) is loaded dynamically. On systems without
//! AMD GPU drivers, the code compiles but operations return appropriate errors.
//!
//! ## Zero Configuration
//!
//! This module is fully automatic with no user configuration required.
//! The kernel loader automatically loads embedded HSACO binaries.
//!
//! ## Usage
//!
//! - `HsaFlashAttentionKernel` - Flash attention kernel
//! - `HsaPagedAttentionKernel` - Paged attention kernel
//! - `HsaBuffer` - HSA memory buffer
//! - `HsaQueueWrapper` - HSA command queue
//!
//! Only requires AMD GPU driver, NOT the full ROCm toolkit.

pub mod hsa_runtime;
pub mod hsa_flash_attn;
pub mod hsa_paged_attn;
pub mod hsa_embedding_ops;

// Core operator kernels
pub mod hsa_softmax;
pub mod hsa_sampling;
pub mod hsa_moe_route;
pub mod hsa_rope;
pub mod hsa_quantized;
pub mod hsa_rms_norm;
pub mod hsa_linear;
pub mod hsa_silu;

// Inference optimization kernels (REQ-OP-008 ~ REQ-OP-015)
pub mod hsa_eagle3;
pub mod hsa_spec_ee;
pub mod hsa_flash_tree_attn;
pub mod hsa_int2_quantizer;
pub mod hsa_evic_press;
pub mod hsa_medusa;
pub mod hsa_prompt_cache;
pub mod hsa_chunked_prefill;

// HSA Runtime
pub use hsa_runtime::{get_hsa_lib, is_hsa_available, HsaLib, GpuAgent, find_gpu_agents};

// HSA-based kernels
pub use hsa_flash_attn::{
    HsaFlashAttentionError, HsaFlashAttentionKernel, HsaBuffer, HsaQueueWrapper,
    OptimizedHsaAttention,
};
pub use hsa_paged_attn::{HsaPagedAttentionError, HsaPagedAttentionKernel};
pub use hsa_embedding_ops::{HsaEmbeddingOpsError, HsaEmbeddingOpsKernel};

// Core operator kernel exports
pub use hsa_softmax::{HsaSoftmaxError, HsaSoftmaxKernel};
pub use hsa_sampling::{HsaSamplingError, HsaSamplingKernel};
pub use hsa_moe_route::{HsaMoeRouteError, HsaMoeRouteKernel};
pub use hsa_rope::{HsaRoPEError, HsaRoPEKernel};
pub use hsa_quantized::{HsaQuantizedError, HsaQuantizedKernel};
pub use hsa_rms_norm::HsaRmsNormKernel;
pub use hsa_linear::HsaLinearKernel;
pub use hsa_silu::HsaSiluKernel;

// Inference optimization kernels exports
pub use hsa_eagle3::{HsaEagle3Error, HsaEagle3Kernel, HsaEagle3Config};
pub use hsa_spec_ee::{HsaSpecEEError, HsaSpecEEKernel, HsaSpecEEConfig};
pub use hsa_flash_tree_attn::{HsaFlashTreeAttnError, HsaFlashTreeAttnKernel, HsaFlashTreeAttnConfig};
pub use hsa_int2_quantizer::{HsaInt2QuantizerError, HsaInt2QuantizerKernel, HsaInt2QuantizerConfig};
pub use hsa_evic_press::{HsaEvicPressError, HsaEvicPressKernel, HsaEvicPressConfig, CacheZone};
pub use hsa_medusa::{HsaMedusaError, HsaMedusaKernel, HsaMedusaConfig};
pub use hsa_prompt_cache::{HsaPromptCacheError, HsaPromptCacheKernel, HsaPromptCacheConfig};
pub use hsa_chunked_prefill::{HsaChunkedPrefillError, HsaChunkedPrefillKernel, HsaChunkedPrefillConfig};

/// Check if AMD GPU is available (HSA runtime).
pub fn is_amd_gpu_available() -> bool {
    is_hsa_available()
}
