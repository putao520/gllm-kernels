//! gllm-kernels: low-level attention kernels for LLM inference.
//!
//! This crate provides high-performance attention operators with:
//! - **Runtime Backend Selection**: Automatically detects CUDA/ROCm/Metal/WGPU/CPU
//! - **Zero-Cost Abstraction**: Enum-based dispatch with `#[inline(always)]`
//! - **2M Context Stability**: LogSpaceSoftmax and KahanAccumulator for long sequences
//! - **Burn-Free Design**: Pure Rust with raw slice APIs (ADR-001)
//!
//! # Quick Start
//!
//! ```ignore
//! use gllm_kernels::{KernelDispatcher, FlashAttentionConfig};
//!
//! let dispatcher = KernelDispatcher::new(); // Auto-detect backend
//! dispatcher.flash_attention(q, k, v, &mut output, config);
//! ```

pub mod comm;
pub mod ops;
pub mod types;
pub mod validation;
pub mod weights;

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

// Fat Binary: embedded pre-compiled kernels
pub mod embedded_kernels;

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
    Eagle3Config, Eagle3DraftResult, Eagle3VerifyConfig, Eagle3VerifyResult,
    SpecEEConfig, SpecEEForwardResult,
    FlashTreeAttentionConfig,
    Int2QuantConfig, Int2QuantResult,
    EvicPressCompression, EvicPressCompressConfig, EvicPressCompressionResult,
    EvicPressEvictConfig, EvicPressEvictResult,
    MedusaConfig, MedusaForwardResult, MedusaVerifyConfig, MedusaVerifyResult,
    PromptCacheLookupConfig, PromptCacheLookupResult, PromptCacheBlendConfig,
    ChunkedPrefillConfig, ChunkedPrefillResult,
    GpuRerankConfig, GpuRerankStageResult,
    MatmulConfig,
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

// RoPE (Rotary Position Embedding) exports
pub use ops::rope::{RoPEConfig, rope_precompute, rope_apply, rope_apply_inplace};

// Sampling operations exports
pub use ops::sampling::{
    SamplingConfig, TopKResult,
    topk, apply_temperature, softmax_1d, apply_top_p, sample_tokens, argmax,
};

// MoE (Mixture-of-Experts) routing exports
pub use ops::moe_routing::{
    MoERoutingConfig, MoERoutingResult,
    moe_route, compute_routing_logits, compute_expert_load, compute_load_balance_loss,
};

// Zero-cost weight containers
pub use weights::{WeightMatrix, WeightVector, Weight3D, Weight4D};

// Zero-cost Linear layer exports
pub use ops::linear::{
    linear_forward, linear_forward_transposed, linear_forward_fused, add_bias,
};

// Zero-cost RMS Norm exports
pub use ops::rms_norm::{
    rms_norm_forward, rms_norm_inplace, rms_norm_forward_with_bias, compute_rms,
};

// Zero-cost Layer Norm exports
pub use ops::layer_norm::{
    layer_norm_forward, layer_norm_inplace, layer_norm_no_affine,
    welford_mean_var, compute_mean, compute_variance,
};

// Zero-cost Activation function exports
pub use ops::activations::{
    // SiLU/Swish
    silu, silu_inplace, silu_mul_inplace,
    // GELU
    gelu, gelu_inplace, gelu_exact, gelu_exact_inplace,
    // ReLU
    relu, relu_inplace, leaky_relu, leaky_relu_inplace,
    // Sigmoid
    sigmoid, sigmoid_inplace, sigmoid_scalar,
    // Tanh
    tanh_activation, tanh_inplace, fast_tanh,
    // Softplus
    softplus, softplus_inplace, softplus_scalar,
    // Element-wise ops
    mul_inplace, add_inplace,
};
