//! Communication backends for ring attention.
//!
//! Zero-cost abstraction via the `Communicator` trait:
//! - NCCL: NVIDIA multi-GPU (feature = "nccl")
//! - RCCL: AMD/ROCm multi-GPU (feature = "rccl")
//! - SharedMemory: Single-node fallback
//! - TCP: Multi-node fallback

mod shared_memory;
mod tcp;
mod traits;
#[cfg(feature = "nccl")]
mod nccl;
#[cfg(feature = "rccl")]
mod rccl;

pub use shared_memory::{SharedMemoryComm, SharedMemoryGroup};
pub use tcp::TcpComm;
pub use traits::{CommError, CommResult, Communicator};
#[cfg(feature = "nccl")]
pub use nccl::NcclComm;
#[cfg(feature = "rccl")]
pub use rccl::{is_rccl_available, RcclCommunicator, RcclError, RcclUniqueId};
