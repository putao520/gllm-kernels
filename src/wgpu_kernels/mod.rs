//! WGPU kernel integrations for GPU-accelerated attention.

pub mod flash_attn;

pub use flash_attn::{FlashAttentionError, FlashAttentionKernel};
