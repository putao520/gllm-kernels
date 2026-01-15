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
//! ## Recommended Usage
//!
//! For new code, use the HSA-based implementation:
//! - `hsa_flash_attn::HsaFlashAttentionKernel` - Only requires AMD GPU driver
//! - `hsa_flash_attn::HsaBuffer` - HSA memory buffer
//! - `hsa_flash_attn::HsaQueueWrapper` - HSA command queue
//!
//! The HIP-based implementation (`flash_attn`) requires the full ROCm toolkit.

pub mod hsa_runtime;
pub mod hip_runtime;
pub mod flash_attn;
pub mod hsa_flash_attn;
pub mod paged_attn;

// HSA Runtime (preferred - only needs AMD driver)
pub use hsa_runtime::{get_hsa_lib, is_hsa_available, HsaLib, GpuAgent, find_gpu_agents};

// HIP Runtime (fallback - needs ROCm)
pub use hip_runtime::{get_hip_lib, is_hip_available, HipLib};

// HSA-based kernels (preferred - only needs AMD driver)
pub use hsa_flash_attn::{
    HsaFlashAttentionError, HsaFlashAttentionKernel, HsaBuffer, HsaQueueWrapper,
    OptimizedHsaAttention,
};

// HIP-based kernels (legacy - needs ROCm)
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel, OptimizedHipAttention};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};

/// Check if AMD GPU is available (HSA or HIP runtime).
pub fn is_amd_gpu_available() -> bool {
    is_hsa_available() || is_hip_available()
}
