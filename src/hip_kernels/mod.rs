//! HIP/ROCm FlashAttention kernels for AMD GPUs.
//!
//! This module provides FlashAttention-style kernels compiled for AMD GPUs
//! using the HIP runtime (ROCm). The API mirrors the CUDA kernel module
//! for easy switching between backends.
//!
//! ## Feature Requirements
//!
//! Enable the `rocm-kernel` feature to use this module:
//! ```toml
//! gllm-kernels = { version = "0.1", features = ["rocm-kernel"] }
//! ```
//!
//! ## Environment Variables
//!
//! - `GLLM_HIP_FLASH_ATTN_HSACO`: Path to precompiled HSACO binary
//! - `GLLM_HIP_FLASH_ATTN_SOURCE`: Path to HIP source for runtime compilation

pub mod flash_attn;

pub use flash_attn::{FlashAttentionError, FlashAttentionKernel, OptimizedHipAttention};
