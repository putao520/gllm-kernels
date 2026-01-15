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

pub mod metal_runtime;
pub mod flash_attn;
pub mod paged_attn;

pub use metal_runtime::{
    get_metal_device, is_metal_available, MetalDeviceInfo, MetalError,
    MetalKernelExecutor, MetalKernelLoader,
};
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};
