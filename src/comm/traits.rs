//! Communication traits for distributed inference.
//!
//! Zero-cost abstraction: uses raw slices + shape arrays, no wrapper types.

use std::fmt;

use crate::kernel_dispatcher::KernelFloat;

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
///
/// Zero-cost design: operates directly on raw slices with shape metadata.
pub trait Communicator: Send + Sync {
    /// Get the rank of this communicator.
    fn rank(&self) -> usize;

    /// Get the total number of participants.
    fn world_size(&self) -> usize;

    /// Send tensor data to the next rank in the ring.
    ///
    /// # Arguments
    /// * `data` - Raw tensor data as bytes
    /// * `shape` - Tensor shape
    /// * `dtype` - Data type identifier (from KernelFloat::TYPE_ID)
    fn send_raw(&self, data: &[u8], shape: &[usize], dtype: u8) -> CommResult<()>;

    /// Receive tensor data from the previous rank in the ring.
    ///
    /// # Returns
    /// Tuple of (data bytes, shape, dtype)
    fn recv_raw(&self) -> CommResult<(Vec<u8>, Vec<usize>, u8)>;

    /// Send typed tensor data (zero-cost generic dispatch).
    #[inline(always)]
    fn send<T: KernelFloat>(&self, data: &[T], shape: &[usize]) -> CommResult<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
        };
        self.send_raw(bytes, shape, T::TYPE_ID.as_u8())
    }

    /// Receive typed tensor data (zero-cost generic dispatch).
    #[inline(always)]
    fn recv<T: KernelFloat>(&self) -> CommResult<(Vec<T>, Vec<usize>)> {
        let (bytes, shape, dtype) = self.recv_raw()?;
        if dtype != T::TYPE_ID.as_u8() {
            return Err(CommError::Serialization(format!(
                "dtype mismatch: expected {}, got {}",
                T::TYPE_ID.as_u8(),
                dtype
            )));
        }
        let len = bytes.len() / std::mem::size_of::<T>();
        let mut data = vec![T::zero(); len];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.as_mut_ptr() as *mut u8, bytes.len());
        }
        Ok((data, shape))
    }

    /// Send to next and receive from previous simultaneously.
    fn send_recv<T: KernelFloat>(&self, send_data: &[T], shape: &[usize]) -> CommResult<(Vec<T>, Vec<usize>)> {
        self.send(send_data, shape)?;
        self.recv()
    }

    /// Barrier synchronization across all ranks.
    fn barrier(&self) -> CommResult<()>;
}
