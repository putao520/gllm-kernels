//! Metal kernel integrations for GPU-accelerated attention.

#[cfg(feature = "metal-kernel")]
pub mod flash_attn;

#[cfg(feature = "metal-kernel")]
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel};
