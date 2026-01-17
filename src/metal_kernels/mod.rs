//! Metal kernel integrations for GPU-accelerated attention.
//!
//! This module provides Metal compute kernels for Apple GPUs.
//! Uses precompiled metallib binaries embedded at compile time.
//!
//! ## Architecture
//!
//! ```text
//! Compile time: Metal shader source → metallib binary (via xcrun metallib) → embedded
//! Runtime: Metal Framework loads embedded metallib → GPU executes
//! Fallback: Runtime compilation from embedded source if metallib fails
//! ```
//!
//! ## Zero Configuration
//!
//! This module is fully automatic with no user configuration required.
//! The kernel loader automatically selects the best loading strategy.

pub mod metallib_loader;
pub mod metal_runtime;
pub mod embedding_ops;
pub mod flash_attn;
pub mod paged_attn;

// Inference optimization kernels (REQ-OP-008 ~ REQ-OP-015)
pub mod eagle3;
pub mod spec_ee;
pub mod flash_tree_attn;
pub mod int2_quantizer;
pub mod evic_press;
pub mod medusa;
pub mod prompt_cache;
pub mod chunked_prefill;

pub use metallib_loader::{MetallibCollection, MetallibLoadError};
pub use metal_runtime::{
    get_metal_device, is_metal_available, MetalDeviceInfo, MetalError,
    MetalKernelExecutor, MetalKernelLoader,
};
pub use embedding_ops::{EmbeddingOpsError as MetalEmbeddingOpsError, EmbeddingOpsKernel as MetalEmbeddingOpsKernel};
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};

// Inference optimization kernels exports
pub use eagle3::{Eagle3Error, Eagle3Kernel, Eagle3Config};
pub use spec_ee::{SpecEEError, SpecEEKernel, SpecEEConfig};
pub use flash_tree_attn::{FlashTreeAttnError, FlashTreeAttnKernel, FlashTreeAttnConfig};
pub use int2_quantizer::{Int2QuantizerError, Int2QuantizerKernel, Int2QuantizerConfig};
pub use evic_press::{EvicPressError, EvicPressKernel, EvicPressConfig, CacheZone};
pub use medusa::{MedusaError, MedusaKernel, MedusaConfig};
pub use prompt_cache::{PromptCacheError, PromptCacheKernel, PromptCacheConfig};
pub use chunked_prefill::{ChunkedPrefillError, ChunkedPrefillKernel, ChunkedPrefillConfig};
