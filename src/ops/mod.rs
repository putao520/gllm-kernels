pub(crate) mod math;

// Pure Rust numerical stability modules (no external dependencies)
pub mod softmax;
pub mod stable_accumulator;

// Zero-cost NN layer operations (Phase 1 additions)
pub mod linear;
pub mod rms_norm;
pub mod layer_norm;
pub mod activations;

// RoPE (Rotary Position Embedding) - pure Rust implementation
pub mod rope;

// Sampling operations - pure Rust implementation
pub mod sampling;

// MoE (Mixture-of-Experts) routing - pure Rust implementation
pub mod moe_routing;

// Engram conditional memory module (O(1) static knowledge lookup)
pub mod engram;
pub mod engram_hash;
pub mod engram_lookup;

// Embedding operations for vector search and rerank
pub mod embedding;

// 2025-2026 Inference Optimization Modules (REQ-OP-008 to REQ-OP-015)
// All modules are Burn-free, using KernelFloat trait for zero-cost abstraction
pub mod eagle3;          // REQ-OP-008: EAGLE-3 Adaptive Draft Length
pub mod spec_ee;         // REQ-OP-009: SpecEE/LayerSkip Early Exit
pub mod medusa;          // REQ-OP-013: Assisted Generation (Medusa Heads)
pub mod int2_quantizer;  // REQ-OP-011: INT2 Extreme Quantization
pub mod evic_press;      // REQ-OP-011: EvicPress Joint Compression/Eviction
pub mod prompt_cache;    // REQ-OP-014: Prompt Caching / CacheBlend
pub mod flash_tree_attn; // REQ-OP-010: DeFT/Talon Flash Tree-attention
pub mod chunked_prefill; // REQ-OP-015: Chunked Prefill / POD-Attention

// NOTE: Burn-based attention modules (flash_attention, paged_attention, ring_attention,
// mamba, mla, etc.) have been removed per ADR-001.
// Use `KernelDispatcher` for GPU-accelerated attention operations.

pub use engram::{Engram, EngramConfig, fuse_engram_attention, fuse_engram_attention_simd};
pub use engram_hash::{EngramHasher, EngramHashConfig};
pub use engram_lookup::{EngramEmbeddingTable, EngramLookupConfig, EngramModule, EngramError};

// EAGLE-3 exports
pub use eagle3::{
    AdaptiveDraftConfig, ConfidencePredictor, Eagle3Decoder, Eagle3Stats,
    LengthScheduler,
};

// SpecEE/LayerSkip exports
pub use spec_ee::{
    EarlyExitHead, LayerDropoutSchedule, SharedActivations, SpecEEConfig, SpecEEEngine,
};

// Medusa exports
pub use medusa::{
    AssistedGenerationConfig, MedusaDraft, MedusaEngine, MedusaHead, NgramCache,
};

// INT2 Quantization exports
pub use int2_quantizer::{
    Int2PackedBuffer, Int2QuantConfig, Int2Quantizer, Int2Tensor,
};

// EvicPress exports
pub use evic_press::{
    EvicPressConfig, KVEntry, MemoryStats, ProgressiveKVCache, StorageZone, TokenImportance,
};

// Prompt Cache exports
pub use prompt_cache::{
    BlendedKVCache, CacheHit, CacheStats, EvictionPolicy, HashAlgorithm,
    PositionReencoding, PromptCacheConfig, PromptCacheEntry, PromptCacheManager,
    StorageTier, TierStats,
};

// Flash Tree-attention exports
pub use flash_tree_attn::{
    BatchTreeConfig, FlashTreeAttention, PartitionStrategy, TalonConfig,
    TalonController, TokenTree, TraversalResult, TreeAttentionOutput,
    TreeAttentionStats, TreeMask,
};

// Chunked Prefill exports
pub use chunked_prefill::{
    BatchOutput, ChunkConfig, ChunkedPrefillScheduler, DecodeOutput, DecodeRequest,
    PODAttentionConfig, PrefillChunk, PrefillOutput, PrefillRequest, ScheduledBatch,
    SchedulerStats,
};

// RoPE exports
pub use rope::{RoPEConfig, rope_precompute, rope_apply, rope_apply_inplace};

// Sampling exports
pub use sampling::{
    SamplingConfig, TopKResult,
    topk, apply_temperature, softmax_1d, apply_top_p, sample_tokens, argmax,
};

// MoE Routing exports
pub use moe_routing::{
    MoERoutingConfig, MoERoutingResult,
    moe_route, compute_routing_logits, compute_expert_load, compute_load_balance_loss,
};

// Linear layer exports
pub use linear::{
    linear_forward, linear_forward_transposed, linear_forward_fused, add_bias,
};

// RMS Norm exports
pub use rms_norm::{
    rms_norm_forward, rms_norm_inplace, rms_norm_forward_with_bias, compute_rms,
};

// Layer Norm exports
pub use layer_norm::{
    layer_norm_forward, layer_norm_inplace, layer_norm_no_affine,
    welford_mean_var, compute_mean, compute_variance,
};

// Activation function exports
pub use activations::{
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
