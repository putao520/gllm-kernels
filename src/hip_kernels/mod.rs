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

// HSA Runtime
pub use hsa_runtime::{get_hsa_lib, is_hsa_available, HsaLib, GpuAgent, find_gpu_agents};

// HSA-based kernels
pub use hsa_flash_attn::{
    HsaFlashAttentionError, HsaFlashAttentionKernel, HsaBuffer, HsaQueueWrapper,
    OptimizedHsaAttention,
};
pub use hsa_paged_attn::{HsaPagedAttentionError, HsaPagedAttentionKernel};
pub use hsa_embedding_ops::{HsaEmbeddingOpsError, HsaEmbeddingOpsKernel};

/// Check if AMD GPU is available (HSA runtime).
pub fn is_amd_gpu_available() -> bool {
    is_hsa_available()
}
