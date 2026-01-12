//! Communication traits for distributed computation.

use std::fmt;

/// Communication error types.
#[derive(Debug, Clone)]
pub enum CommError {
    /// Connection failed.
    ConnectionFailed(String),
    /// Send operation failed.
    SendFailed(String),
    /// Receive operation failed.
    RecvFailed(String),
    /// Timeout occurred.
    Timeout(String),
    /// Invalid rank or configuration.
    InvalidConfig(String),
    /// Channel disconnected.
    Disconnected,
    /// Serialization/deserialization error.
    SerdeError(String),
}

impl fmt::Display for CommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            CommError::SendFailed(msg) => write!(f, "Send failed: {}", msg),
            CommError::RecvFailed(msg) => write!(f, "Recv failed: {}", msg),
            CommError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            CommError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            CommError::Disconnected => write!(f, "Channel disconnected"),
            CommError::SerdeError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for CommError {}

/// Result type for communication operations.
pub type CommResult<T> = Result<T, CommError>;

/// Communicator trait for ring communication pattern.
///
/// This trait defines the interface for point-to-point communication
/// in a ring topology, where each node sends to its successor and
/// receives from its predecessor.
pub trait Communicator: Send + Sync {
    /// Data type for communication.
    type Data: Send + Clone;

    /// Get the rank of this communicator.
    fn rank(&self) -> usize;

    /// Get the total number of participants.
    fn world_size(&self) -> usize;

    /// Send data to the next rank in the ring.
    fn send(&self, data: &Self::Data) -> CommResult<()>;

    /// Receive data from the previous rank in the ring.
    fn recv(&self) -> CommResult<Self::Data>;

    /// Send to next and receive from previous simultaneously.
    /// This is the core operation for ring communication.
    fn send_recv(&self, send_data: &Self::Data) -> CommResult<Self::Data> {
        // Default implementation: sequential send then recv
        // Subclasses may override for true simultaneous operation
        self.send(send_data)?;
        self.recv()
    }

    /// Barrier synchronization across all ranks.
    fn barrier(&self) -> CommResult<()>;

    /// Get the next rank in the ring (rank + 1) % world_size.
    fn next_rank(&self) -> usize {
        (self.rank() + 1) % self.world_size()
    }

    /// Get the previous rank in the ring (rank + world_size - 1) % world_size.
    fn prev_rank(&self) -> usize {
        (self.rank() + self.world_size() - 1) % self.world_size()
    }
}

/// Message wrapper for serialized tensor data.
#[derive(Clone, Debug)]
pub struct TensorMessage {
    /// Flattened tensor data as f32 values.
    pub data: Vec<f32>,
    /// Tensor shape.
    pub shape: Vec<usize>,
}

impl TensorMessage {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}
