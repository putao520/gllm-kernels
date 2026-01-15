pub mod flash_attention;
pub mod flash_attention_v3;
pub mod fused_attention;
pub mod kv_compression;
pub mod mamba;
pub mod mamba_v3;
pub mod mla;
pub mod paged_attention;
pub mod ring_attention;
pub mod speculative_decoding;
pub mod sparse_attention;
pub mod softmax;
pub mod stable_accumulator;

// Engram conditional memory module (O(1) static knowledge lookup)
pub mod engram;
pub mod engram_hash;
pub mod engram_lookup;

// Embedding operations for vector search and rerank
pub mod embedding;

// 2025-2026 Inference Optimization Modules (REQ-OP-008 to REQ-OP-015)
pub mod eagle3;          // REQ-OP-008: EAGLE-3 Adaptive Draft Length
pub mod spec_ee;         // REQ-OP-009: SpecEE/LayerSkip Early Exit
pub mod medusa;          // REQ-OP-013: Assisted Generation (Medusa Heads)
pub mod int2_quantizer;  // REQ-OP-011: INT2 Extreme Quantization
pub mod evic_press;      // REQ-OP-011: EvicPress Joint Compression/Eviction
pub mod prompt_cache;    // REQ-OP-014: Prompt Caching / CacheBlend
pub mod flash_tree_attn; // REQ-OP-010: DeFT/Talon Flash Tree-attention
pub mod chunked_prefill; // REQ-OP-015: Chunked Prefill / POD-Attention

pub use ring_attention::{CommBackend, RingAttention, RingAttentionConfig};
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
