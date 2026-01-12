//! gllm-kernels: low-level attention kernels built on Burn.

pub mod backend;
pub mod comm;
pub mod device;
pub mod ops;
pub mod types;

pub use backend::{select_device, DefaultBackend};
pub use comm::{
    CommError, CommResult, Communicator, NcclComm, NcclCommConfig, SharedMemoryComm,
    SharedMemoryGroup, TcpComm, TcpCommConfig, TensorMessage,
};
pub use device::{default_device, DefaultDevice};
pub use ops::flash_attention::{FlashAttentionConfig, FusedPagedAttention, HierarchicalFlashAttention};
pub use ops::paged_attention::{
    BlockManager, BlockTable, KVBlock, KVBlockIterator, KVBlockRef, PagedAttention, PagedKVCache,
};
pub use ops::ring_attention::{CommBackend, RingAttention, RingAttentionConfig};
pub use ops::softmax::{log_add_exp, log_sum_exp, log_sum_exp_kahan, LogSpaceSoftmax};
pub use ops::stable_accumulator::{
    AccumulatorConfig, HierarchicalAccumulator, KahanAccumulator, KahanSum, OutputAccumulator,
    StableAccumulator, StableRowState,
};
pub use types::{AttentionConfig, KernelPrecision, PagedAttentionConfig};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
