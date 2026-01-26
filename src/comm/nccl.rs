//! NCCL communicator for high-performance multi-GPU communication.
//!
//! Uses NVIDIA NCCL for efficient GPU-to-GPU data transfer.
//! Zero-cost abstraction: raw bytes + shape, no wrapper types.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::nccl::{group_end, group_start, Comm, Id, ReduceOp};

use super::traits::{CommError, CommResult, Communicator};
use crate::kernel_types::KernelFloat;

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
    fn nccl_send_f32(&self, data: &[f32]) -> CommResult<()> {
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
    fn nccl_recv_f32(&self, len: usize) -> CommResult<Vec<f32>> {
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
    fn nccl_send_recv_f32(&self, send_data: &[f32]) -> CommResult<Vec<f32>> {
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

    /// Optimized send_recv using NCCL group operations.
    pub fn send_recv_parallel<T: KernelFloat>(&self, data: &[T], shape: &[usize]) -> CommResult<(Vec<T>, Vec<usize>)> {
        // Convert to f32 for NCCL (NCCL natively supports f32)
        // For production, we'd add f16/bf16 support using NCCL's native types
        let f32_data: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();

        let recv_f32 = self.nccl_send_recv_f32(&f32_data)?;

        let recv_data: Vec<T> = recv_f32.iter().map(|&x| T::from_f32(x)).collect();

        Ok((recv_data, shape.to_vec()))
    }
}

impl Communicator for NcclComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send_raw(&self, data: &[u8], shape: &[usize], dtype: u8) -> CommResult<()> {
        // Protocol: send metadata first, then data
        // Metadata: [ndim: u64, shape[0]: u64, ..., shape[n]: u64, dtype: u8, data_len: u64]

        let ndim = shape.len();
        let metadata_f32_len = 2 + ndim + 1; // ndim + shape + dtype + data_len (packed as f32)
        let mut metadata = Vec::with_capacity(metadata_f32_len);

        metadata.push(ndim as f32);
        for &dim in shape {
            metadata.push(dim as f32);
        }
        metadata.push(dtype as f32);
        metadata.push(data.len() as f32);

        // Send metadata
        self.nccl_send_f32(&metadata)?;

        // Convert bytes to f32 for NCCL transfer
        // Pad to f32 alignment
        let f32_len = (data.len() + 3) / 4;
        let mut data_f32 = vec![0.0f32; f32_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                data_f32.as_mut_ptr() as *mut u8,
                data.len(),
            );
        }

        // Send data
        self.nccl_send_f32(&data_f32)?;

        Ok(())
    }

    fn recv_raw(&self) -> CommResult<(Vec<u8>, Vec<usize>, u8)> {
        // First, receive metadata (we need to know ndim to calculate metadata size)
        // Initial recv: just ndim (1 f32)
        let ndim_vec = self.nccl_recv_f32(1)?;
        let ndim = ndim_vec[0] as usize;

        // Now receive rest of metadata: shape + dtype + data_len
        let rest_len = ndim + 2; // shape + dtype + data_len
        let rest = self.nccl_recv_f32(rest_len)?;

        let mut shape = Vec::with_capacity(ndim);
        for i in 0..ndim {
            shape.push(rest[i] as usize);
        }
        let dtype = rest[ndim] as u8;
        let data_len = rest[ndim + 1] as usize;

        // Receive data
        let f32_len = (data_len + 3) / 4;
        let data_f32 = self.nccl_recv_f32(f32_len)?;

        // Convert back to bytes
        let mut data = vec![0u8; data_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data_f32.as_ptr() as *const u8,
                data.as_mut_ptr(),
                data_len,
            );
        }

        Ok((data, shape, dtype))
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
