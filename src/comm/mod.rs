//! Communication backends for ring attention.

mod shared_memory;
mod tcp;
mod traits;
#[cfg(feature = "nccl")]
mod nccl;

pub use shared_memory::{SharedMemoryComm, SharedMemoryGroup};
pub use tcp::TcpComm;
pub use traits::{CommError, CommResult, Communicator};
#[cfg(feature = "nccl")]
pub use nccl::NcclComm;
