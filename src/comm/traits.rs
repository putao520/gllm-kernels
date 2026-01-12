//! Communication traits for distributed computation.

use std::fmt;

use burn::tensor::TensorData;

/// Communication error types.
#[derive(Debug)]
pub enum CommError {
    /// Invalid rank or configuration.
    InvalidConfig(String),
    /// Connection failed.
    ConnectionFailed(String),
    /// Send operation failed.
    SendFailed(String),
    /// Receive operation failed.
    RecvFailed(String),
    /// Channel disconnected.
    Disconnected,
    /// Serialization/deserialization error.
    Serialization(String),
}

impl fmt::Display for CommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            CommError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            CommError::SendFailed(msg) => write!(f, "Send failed: {}", msg),
            CommError::RecvFailed(msg) => write!(f, "Recv failed: {}", msg),
            CommError::Disconnected => write!(f, "Channel disconnected"),
            CommError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for CommError {}

/// Result type for communication operations.
pub type CommResult<T> = Result<T, CommError>;

/// Communicator trait for ring communication pattern.
pub trait Communicator: Send + Sync {
    /// Get the rank of this communicator.
    fn rank(&self) -> usize;

    /// Get the total number of participants.
    fn world_size(&self) -> usize;

    /// Send data to the next rank in the ring.
    fn send(&self, data: &TensorData) -> CommResult<()>;

    /// Receive data from the previous rank in the ring.
    fn recv(&self) -> CommResult<TensorData>;

    /// Send to next and receive from previous simultaneously.
    fn send_recv(&self, send_data: &TensorData) -> CommResult<TensorData> {
        self.send(send_data)?;
        self.recv()
    }

    /// Barrier synchronization across all ranks.
    fn barrier(&self) -> CommResult<()>;
}
