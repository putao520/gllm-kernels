//! TCP communication for multi-node distributed inference.
//!
//! Zero-cost abstraction: raw bytes + shape, no wrapper types.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

use super::traits::{CommError, CommResult, Communicator};

/// TCP communicator for multi-node ring communication.
pub struct TcpComm {
    rank: usize,
    world_size: usize,
    send_stream: Mutex<TcpStream>,
    recv_stream: Mutex<TcpStream>,
}

impl TcpComm {
    /// Connect to peers in a ring topology.
    pub fn connect(addresses: Vec<String>, rank: usize) -> CommResult<Self> {
        if addresses.is_empty() {
            return Err(CommError::InvalidConfig("addresses must not be empty".into()));
        }

        let world_size = addresses.len();
        if rank >= world_size {
            return Err(CommError::InvalidConfig(format!(
                "rank {} >= world_size {}", rank, world_size
            )));
        }

        let bind_addr = &addresses[rank];
        let listener = TcpListener::bind(bind_addr)
            .map_err(|e| CommError::ConnectionFailed(format!("Failed to bind {}: {}", bind_addr, e)))?;

        let next_rank = (rank + 1) % world_size;
        let next_addr = &addresses[next_rank];
        let send_stream = TcpStream::connect(next_addr)
            .map_err(|e| CommError::ConnectionFailed(format!("Failed to connect to {}: {}", next_addr, e)))?;
        let _ = send_stream.set_nodelay(true);

        let (recv_stream, _) = listener.accept()
            .map_err(|e| CommError::ConnectionFailed(format!("Failed to accept on {}: {}", bind_addr, e)))?;
        let _ = recv_stream.set_nodelay(true);

        Ok(Self {
            rank,
            world_size,
            send_stream: Mutex::new(send_stream),
            recv_stream: Mutex::new(recv_stream),
        })
    }
}

impl Communicator for TcpComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send_raw(&self, data: &[u8], shape: &[usize], dtype: u8) -> CommResult<()> {
        let mut stream = self.send_stream.lock()
            .map_err(|_| CommError::SendFailed("Failed to acquire send lock".into()))?;

        // Header: dtype (1) + ndim (1) + shape (ndim * 8) + data_len (8)
        let ndim = shape.len() as u8;
        let header_size = 1 + 1 + (ndim as usize * 8) + 8;
        let mut header = Vec::with_capacity(header_size);

        header.push(dtype);
        header.push(ndim);
        for &dim in shape {
            header.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        header.extend_from_slice(&(data.len() as u64).to_le_bytes());

        stream.write_all(&header)
            .map_err(|e| CommError::SendFailed(format!("Failed to send header: {}", e)))?;
        stream.write_all(data)
            .map_err(|e| CommError::SendFailed(format!("Failed to send data: {}", e)))?;
        stream.flush()
            .map_err(|e| CommError::SendFailed(format!("Failed to flush: {}", e)))?;

        Ok(())
    }

    fn recv_raw(&self) -> CommResult<(Vec<u8>, Vec<usize>, u8)> {
        let mut stream = self.recv_stream.lock()
            .map_err(|_| CommError::RecvFailed("Failed to acquire recv lock".into()))?;

        // Read dtype and ndim
        let mut header = [0u8; 2];
        stream.read_exact(&mut header)
            .map_err(|e| CommError::RecvFailed(format!("Failed to read header: {}", e)))?;

        let dtype = header[0];
        let ndim = header[1] as usize;

        // Read shape
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_buf = [0u8; 8];
            stream.read_exact(&mut dim_buf)
                .map_err(|e| CommError::RecvFailed(format!("Failed to read shape: {}", e)))?;
            shape.push(u64::from_le_bytes(dim_buf) as usize);
        }

        // Read data length and data
        let mut len_buf = [0u8; 8];
        stream.read_exact(&mut len_buf)
            .map_err(|e| CommError::RecvFailed(format!("Failed to read data length: {}", e)))?;
        let data_len = u64::from_le_bytes(len_buf) as usize;

        let mut data = vec![0u8; data_len];
        stream.read_exact(&mut data)
            .map_err(|e| CommError::RecvFailed(format!("Failed to read data: {}", e)))?;

        Ok((data, shape, dtype))
    }

    fn barrier(&self) -> CommResult<()> {
        if self.world_size <= 1 {
            return Ok(());
        }

        // Simple barrier using empty tensor
        let empty: [u8; 0] = [];
        if self.rank == 0 {
            self.send_raw(&empty, &[0], 0)?;
            let _ = self.recv_raw()?;
            self.send_raw(&empty, &[0], 0)?;
            let _ = self.recv_raw()?;
        } else {
            let _ = self.recv_raw()?;
            self.send_raw(&empty, &[0], 0)?;
            let _ = self.recv_raw()?;
            self.send_raw(&empty, &[0], 0)?;
        }

        Ok(())
    }
}
