//! WGPU kernel integrations for GPU-accelerated attention.

pub mod flash_attn;
pub mod paged_attn;

pub use flash_attn::{FlashAttentionError, FlashAttentionKernel};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel};
