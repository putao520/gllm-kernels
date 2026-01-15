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

pub use ring_attention::{CommBackend, RingAttention, RingAttentionConfig};
pub use engram::{Engram, EngramConfig, fuse_engram_attention, fuse_engram_attention_simd};
pub use engram_hash::{EngramHasher, EngramHashConfig};
pub use engram_lookup::{EngramEmbeddingTable, EngramLookupConfig, EngramModule, EngramError};
