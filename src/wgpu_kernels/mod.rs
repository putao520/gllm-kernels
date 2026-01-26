//! WGPU kernel integrations for GPU-accelerated operations.
//!
//! This module provides WGPU (WebGPU) implementations of inference optimization kernels.
//! All shaders use WGSL (WebGPU Shading Language) embedded via `include_str!`.

// Core attention kernels
pub mod flash_attn;
pub mod paged_attn;
pub mod flash_tree_attn;

// Speculative decoding kernels
pub mod eagle3;
pub mod spec_ee;
pub mod medusa;

// KV cache optimization kernels
pub mod prompt_cache;
pub mod chunked_prefill;
pub mod evic_press;

// Quantization kernels
pub mod int2_quantizer;

// Utility kernels
pub mod embedding_ops;
pub mod linear;
pub mod rms_norm;
pub mod silu;
pub mod tensor_ops;
pub mod moe_ffn;
pub mod moe_routing_gpu;

// Re-exports - Core attention
pub use flash_attn::{FlashAttentionError, FlashAttentionKernel};
pub use paged_attn::{PagedAttentionError, PagedAttentionKernel, PagedAttentionParams};
pub use flash_tree_attn::{
    TreeAttnParams, VerifyParams, TreeMaskParams, VerifyResult, FlashTreeAttn,
};
pub use flash_tree_attn::FlashTreeAttn as WgpuFlashTreeAttn;

// Re-exports - Speculative decoding
pub use eagle3::{Eagle3Error, Eagle3Kernel};
pub use eagle3::Eagle3Kernel as WgpuEagle3;
pub use spec_ee::{EarlyExitResult, EarlyExitResultF16, SpecEEError, SpecEEKernel};
pub use spec_ee::SpecEEKernel as WgpuSpecEE;
pub use medusa::{
    HeadForwardParams, TopKParams, CandidateParams,
    TopKResult, CandidateTree, WgpuMedusa,
};

// Re-exports - KV cache optimization
pub use prompt_cache::{
    HashParams, PrefixMatchParams, BlendParams, CopyKVParams, RollingHashParams,
    PrefixMatchResult, WgpuPromptCache,
};
pub use chunked_prefill::{
    ChunkAttentionParams, ChunkMergeParams, PODSplitParams, ScheduleParams,
    ChunkAttentionResult, PODSplitResult, ScheduleResult, WgpuChunkedPrefill,
};
pub use evic_press::{
    ImportanceParams, ZoneTransitionParams,
    TokenImportance, WgpuEvicPress,
};

// Re-exports - Quantization
pub use int2_quantizer::{
    QuantizeParams, PackParams,
    WgpuInt2Quantizer,
};

// Re-exports - Utility
pub use embedding_ops::{EmbeddingOpsError, EmbeddingOpsKernel};
// Note: GpuRerankConfig and GpuRerankStageResult are exported from crate::types
pub use linear::{LinearError, LinearParams, WgpuLinear};
pub use rms_norm::{RmsNormError, RmsNormParams, WgpuRmsNorm};
pub use silu::{SiluError, SiluParams, WgpuSilu};
pub use tensor_ops::{TensorOpsParams, WgpuTensorOps};
pub use moe_ffn::{MoEFfnParams, WgpuMoeFfn};
pub use moe_routing_gpu::{MoERoutingGpuParams, WgpuMoERouting};
