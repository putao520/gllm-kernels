//! Communication backends for distributed computation.
//!
//! This module provides communication primitives for ring attention
//! and other distributed operations. Three backends are available:
//!
//! - `SharedMemory`: For single-node multi-GPU using crossbeam channels
//! - `Tcp`: For multi-node using TCP sockets
//! - `Nccl`: For high-performance GPU communication (requires `nccl` feature)

mod nccl;
mod shared_memory;
mod tcp;
mod traits;

pub use nccl::{NcclComm, NcclCommConfig};
pub use shared_memory::{run_ring, SharedMemoryComm, SharedMemoryGroup};
pub use tcp::{TcpComm, TcpCommConfig};
pub use traits::{CommError, CommResult, Communicator, TensorMessage};
