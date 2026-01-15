//! gllm-kernels: low-level attention kernels built on Burn.
//!
//! This crate provides high-performance attention operators with:
//! - **Runtime Backend Selection**: Automatically detects CUDA/ROCm/Metal/WGPU/CPU
//! - **Zero-Cost Abstraction**: Enum-based dispatch with `#[inline(always)]`
//! - **2M Context Stability**: LogSpaceSoftmax and KahanAccumulator for long sequences
//!
//! # Quick Start
//!
//! ```ignore
//! use gllm_kernels::{KernelDispatcher, FlashAttentionConfig};
//!
//! let dispatcher = KernelDispatcher::new(); // Auto-detect backend
//! dispatcher.flash_attention(q, k, v, &mut output, config);
//! ```

pub mod backend;
pub mod comm;
pub mod device;
pub mod ops;
pub mod types;

// GPU kernels - always compiled, runtime detection determines availability
// CUDA kernels use cudarc's dynamic-loading feature
pub mod cuda_kernels;
// ROCm/HIP kernels (Linux only)
#[cfg(target_os = "linux")]
pub mod hip_kernels;
// Metal kernels (macOS only)
#[cfg(target_os = "macos")]
pub mod metal_kernels;
// WGPU kernels (cross-platform)
pub mod wgpu_kernels;

// Runtime backend detection
pub mod runtime_detection;
pub mod backend_trait;
pub mod backend_selector;
pub mod kernel_cache;

// Zero-cost kernel dispatcher
pub mod kernel_dispatcher;

pub use backend::DefaultBackend;
pub use backend::select_device;
pub use runtime_detection::{
    BackendType, BackendDetectionResult, DeviceInfo, GpuCapabilities,
    detect_backend, redetect_backend, ensure_kernels,
};
pub use kernel_cache::{
    kernel_cache_dir, kernel_cache_path, load_cached_kernel, save_kernel_to_cache,
    clear_kernel_cache, kernel_cache_size,
};
pub use wgpu_kernels::FlashAttentionKernel;

// Zero-cost dispatcher exports
pub use kernel_dispatcher::{
    KernelDispatcher, KernelFloat, FlashAttentionConfig, PagedAttentionConfig, SoftmaxConfig,
};

// Engram conditional memory exports
pub use ops::engram::{Engram, EngramConfig, fuse_engram_attention, fuse_engram_attention_simd};
pub use ops::engram_hash::{EngramHasher, EngramHashConfig};
pub use ops::engram_lookup::{EngramEmbeddingTable, EngramLookupConfig, EngramModule, EngramError};

// Embedding operations exports (Binary/Int8/Int4 quantization, Matryoshka, Rerank)
pub use ops::embedding::{
    // Binary Quantization
    BinaryIpConfig, pack_binary_f32, binary_ip_hamming, binary_ip_hamming_simd, binary_ip_asymmetric,
    // Int8 Quantization
    Int8DotConfig, quantize_to_int8, int8_dot_product, int8_dot_product_unrolled,
    // Int4 Packed Quantization
    Int4PackedConfig, pack_int4, unpack_int4, quantize_to_int4_packed, int4_packed_dot_product,
    // Matryoshka Dimension Truncation
    MatryoshkaConfig, matryoshka_truncate, select_matryoshka_dim,
    // Three-Stage Rerank Pipeline
    RerankPipelineConfig, RerankResult, rerank_binary_stage, rerank_int8_stage,
};
