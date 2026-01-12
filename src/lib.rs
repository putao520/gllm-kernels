//! gllm-kernels: low-level attention kernels built on Burn.

pub mod backend;
pub mod comm;
#[cfg(feature = "cuda-kernel")]
pub mod cuda_kernels;
#[cfg(feature = "rocm-kernel")]
pub mod hip_kernels;
pub mod device;
pub mod ops;
pub mod types;

pub use backend::{select_device, DefaultBackend};
pub use comm::{CommError, CommResult, Communicator, SharedMemoryComm, SharedMemoryGroup, TcpComm};
#[cfg(feature = "cuda-kernel")]
pub use cuda_kernels::{FlashAttentionError, FlashAttentionKernel, OptimizedCudaAttention};
#[cfg(feature = "rocm-kernel")]
pub use hip_kernels::{
    FlashAttentionError as HipFlashAttentionError, FlashAttentionKernel as HipFlashAttentionKernel,
    OptimizedHipAttention,
};
pub use device::{default_device, DefaultDevice};
pub use ops::flash_attention::{AttentionWorkspace, FlashAttentionConfig, FusedPagedAttention, HierarchicalFlashAttention};
pub use ops::flash_attention_v3::{FlashAttention3, FlashAttention3Config};
pub use ops::kv_compression::{CompressedKV, CompressionMethod, KVCacheCompressor, KVLayout};
pub use ops::mamba::{
    HybridLayer, HybridStrategy, MambaBlock, MambaConfig, MambaParameters, MambaState,
};
pub use ops::mla::{CompressedKVCache, MultiHeadLatentAttention};
pub use ops::paged_attention::{
    BlockManager, BlockTable, KVBlock, KVBlockIterator, KVBlockRef, PagedAttention, PagedKVCache,
};
pub use ops::ring_attention::{CommBackend, RingAttention, RingAttentionConfig};
pub use ops::speculative_decoding::{
    PredictionConfig, PredictionHeadType, SpeculativeCandidates, SpeculativeDecoder,
    SpeculativeToken, SpeculativeTree, SpeculativeVerification, TreeConfig, VerificationStrategy,
};
pub use ops::sparse_attention::{
    SparseAttention, SparseAttentionConfig, SparseSelection, SparsityPattern,
};
pub use ops::softmax::{log_add_exp, log_sum_exp, log_sum_exp_kahan, LogSpaceSoftmax};
pub use ops::stable_accumulator::{
    AccumulatorConfig, HierarchicalAccumulator, KahanAccumulator, KahanSum, OutputAccumulator,
    StableAccumulator, StableRowState,
};
pub use types::{AttentionConfig, KernelPrecision, PagedAttentionConfig};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
