//! NCCL communicator for high-performance multi-GPU communication.
//!
//! Uses NVIDIA NCCL for efficient GPU-to-GPU data transfer.

use std::sync::Arc;

use burn::tensor::TensorData;
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::nccl::{group_end, group_start, Comm, Id, ReduceOp};

use super::{CommError, CommResult, Communicator};

/// NCCL Communicator for multi-GPU ring communication.
pub struct NcclComm {
    /// NCCL communicator handle.
    comm: Comm,
    /// CUDA device.
    device: Arc<CudaDevice>,
    /// World size.
    world_size: usize,
    /// Current rank.
    rank: usize,
}

// SAFETY: NCCL library is designed to be thread-safe.
// The Comm handle can be safely used across threads.
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    /// Create a new NCCL communicator.
    ///
    /// For single-node multi-GPU, one process creates the ID and shares it.
    /// For multi-node, the ID must be broadcast from rank 0 to all nodes.
    pub fn new(id: Id, world_size: usize, rank: usize, device_id: usize) -> CommResult<Self> {
        if world_size == 0 {
            return Err(CommError::InvalidConfig(
                "world_size must be > 0".to_string(),
            ));
        }
        if rank >= world_size {
            return Err(CommError::InvalidConfig(format!(
                "rank {} >= world_size {}",
                rank, world_size
            )));
        }
        if world_size > i32::MAX as usize {
            return Err(CommError::InvalidConfig(
                "world_size exceeds NCCL limits".to_string(),
            ));
        }

        let device = CudaDevice::new(device_id)
            .map_err(|e| CommError::ConnectionFailed(format!("CUDA device {}: {}", device_id, e)))?;

        let comm = Comm::from_rank(device.clone(), rank, world_size, id)
            .map_err(|e| CommError::ConnectionFailed(format!("NCCL init: {:?}", e)))?;

        Ok(Self {
            comm,
            device,
            world_size,
            rank,
        })
    }

    /// Create a unique NCCL ID (call on rank 0, then broadcast to others).
    pub fn create_id() -> CommResult<Id> {
        Id::new().map_err(|e| CommError::ConnectionFailed(format!("Create NCCL ID: {:?}", e)))
    }

    /// Get the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    fn peer_rank(&self, rank: usize) -> CommResult<i32> {
        i32::try_from(rank).map_err(|_| {
            CommError::InvalidConfig(format!("rank {} exceeds NCCL limits", rank))
        })
    }

    /// Send f32 data to next rank using NCCL.
    fn nccl_send(&self, data: &[f32]) -> CommResult<()> {
        let next_rank = (self.rank + 1) % self.world_size;
        let next_rank = self.peer_rank(next_rank)?;

        let gpu_data = self
            .device
            .htod_sync_copy(data)
            .map_err(|e| CommError::SendFailed(format!("H2D copy: {}", e)))?;

        self.comm
            .send(&gpu_data, next_rank)
            .map_err(|e| CommError::SendFailed(format!("NCCL send: {:?}", e)))?;

        Ok(())
    }

    /// Receive f32 data from previous rank using NCCL.
    fn nccl_recv(&self, len: usize) -> CommResult<Vec<f32>> {
        let prev_rank = (self.rank + self.world_size - 1) % self.world_size;
        let prev_rank = self.peer_rank(prev_rank)?;

        let mut gpu_buf: CudaSlice<f32> = self
            .device
            .alloc_zeros(len)
            .map_err(|e| CommError::RecvFailed(format!("Alloc: {}", e)))?;

        self.comm
            .recv(&mut gpu_buf, prev_rank)
            .map_err(|e| CommError::RecvFailed(format!("NCCL recv: {:?}", e)))?;

        let data = self
            .device
            .dtoh_sync_copy(&gpu_buf)
            .map_err(|e| CommError::RecvFailed(format!("D2H copy: {}", e)))?;

        Ok(data)
    }

    /// Fused send and receive for ring pattern (more efficient).
    fn nccl_send_recv(&self, send_data: &[f32]) -> CommResult<Vec<f32>> {
        let next_rank = (self.rank + 1) % self.world_size;
        let prev_rank = (self.rank + self.world_size - 1) % self.world_size;
        let next_rank = self.peer_rank(next_rank)?;
        let prev_rank = self.peer_rank(prev_rank)?;
        let len = send_data.len();

        let gpu_send = self
            .device
            .htod_sync_copy(send_data)
            .map_err(|e| CommError::SendFailed(format!("H2D copy: {}", e)))?;

        let mut gpu_recv: CudaSlice<f32> = self
            .device
            .alloc_zeros(len)
            .map_err(|e| CommError::RecvFailed(format!("Alloc: {}", e)))?;

        group_start().map_err(|e| CommError::SendFailed(format!("Group start: {:?}", e)))?;

        self.comm
            .send(&gpu_send, next_rank)
            .map_err(|e| CommError::SendFailed(format!("NCCL send: {:?}", e)))?;

        self.comm
            .recv(&mut gpu_recv, prev_rank)
            .map_err(|e| CommError::RecvFailed(format!("NCCL recv: {:?}", e)))?;

        group_end().map_err(|e| CommError::SendFailed(format!("Group end: {:?}", e)))?;

        self.device
            .synchronize()
            .map_err(|e| CommError::RecvFailed(format!("Sync: {}", e)))?;

        let data = self
            .device
            .dtoh_sync_copy(&gpu_recv)
            .map_err(|e| CommError::RecvFailed(format!("D2H copy: {}", e)))?;

        Ok(data)
    }
}

impl Communicator for NcclComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, data: &TensorData) -> CommResult<()> {
        let floats: Vec<f32> = data
            .to_vec::<f32>()
            .map_err(|e| CommError::Serialization(format!("{:?}", e)))?;
        self.nccl_send(&floats)
    }

    fn recv(&self) -> CommResult<TensorData> {
        Err(CommError::RecvFailed(
            "Use recv_with_size for NCCL".into(),
        ))
    }

    fn send_recv(&self, data: &TensorData) -> CommResult<TensorData> {
        let floats: Vec<f32> = data
            .to_vec::<f32>()
            .map_err(|e| CommError::Serialization(format!("{:?}", e)))?;
        let shape = data.shape.clone();

        let recv_floats = self.nccl_send_recv(&floats)?;

        Ok(TensorData::new(recv_floats, shape))
    }

    fn barrier(&self) -> CommResult<()> {
        if self.world_size <= 1 {
            return Ok(());
        }

        let send_buf: CudaSlice<f32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| CommError::ConnectionFailed(format!("Alloc: {}", e)))?;
        let mut recv_buf: CudaSlice<f32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| CommError::ConnectionFailed(format!("Alloc: {}", e)))?;

        self.comm
            .all_reduce(&send_buf, &mut recv_buf, &ReduceOp::Sum)
            .map_err(|e| CommError::ConnectionFailed(format!("Barrier: {:?}", e)))?;

        self.device
            .synchronize()
            .map_err(|e| CommError::ConnectionFailed(format!("Sync: {}", e)))?;

        Ok(())
    }
}
