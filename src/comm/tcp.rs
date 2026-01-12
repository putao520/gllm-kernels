//! TCP communication for multi-node distributed ring attention.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::traits::{CommError, CommResult, Communicator, TensorMessage};

/// TCP communicator configuration.
#[derive(Clone, Debug)]
pub struct TcpCommConfig {
    /// List of all node addresses in the ring (host:port).
    pub addresses: Vec<String>,
    /// Rank of this node.
    pub rank: usize,
    /// Connection timeout in milliseconds.
    pub timeout_ms: u64,
}

impl TcpCommConfig {
    pub fn new(addresses: Vec<String>, rank: usize) -> Self {
        Self {
            addresses,
            rank,
            timeout_ms: 30000,
        }
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
}

/// TCP communicator for multi-node ring communication.
pub struct TcpComm {
    rank: usize,
    world_size: usize,
    /// Connection to send to next rank.
    send_conn: Arc<Mutex<TcpStream>>,
    /// Connection to receive from previous rank.
    recv_conn: Arc<Mutex<TcpStream>>,
    /// Listener for barrier synchronization.
    barrier_listener: Arc<Mutex<TcpListener>>,
    /// Addresses of all nodes.
    addresses: Vec<String>,
}

impl TcpComm {
    /// Create a new TCP communicator.
    ///
    /// This will:
    /// 1. Listen for incoming connection from previous rank
    /// 2. Connect to next rank
    /// 3. Set up barrier coordination
    pub fn new(config: TcpCommConfig) -> CommResult<Self> {
        let world_size = config.addresses.len();
        let rank = config.rank;

        if rank >= world_size {
            return Err(CommError::InvalidConfig(format!(
                "rank {} >= world_size {}",
                rank, world_size
            )));
        }

        let timeout = Duration::from_millis(config.timeout_ms);

        // Parse our own address to get the port for listening
        let my_addr = &config.addresses[rank];

        // Listen for connection from previous rank
        let listener = TcpListener::bind(my_addr).map_err(|e| {
            CommError::ConnectionFailed(format!("Failed to bind to {}: {}", my_addr, e))
        })?;

        listener.set_nonblocking(false).ok();

        // Connect to next rank
        let next_rank = (rank + 1) % world_size;
        let next_addr = &config.addresses[next_rank];

        // Retry connection with backoff
        let send_conn = Self::connect_with_retry(next_addr, timeout)?;

        // Accept connection from previous rank
        let (recv_conn, _) = listener.accept().map_err(|e| {
            CommError::ConnectionFailed(format!("Failed to accept connection: {}", e))
        })?;

        // Create barrier listener on a different port
        let barrier_port: u16 = my_addr
            .split(':')
            .last()
            .and_then(|p| p.parse().ok())
            .unwrap_or(9000)
            + 1000;
        let barrier_addr = format!(
            "{}:{}",
            my_addr.split(':').next().unwrap_or("0.0.0.0"),
            barrier_port
        );
        let barrier_listener = TcpListener::bind(&barrier_addr).map_err(|e| {
            CommError::ConnectionFailed(format!("Failed to bind barrier listener: {}", e))
        })?;

        Ok(Self {
            rank,
            world_size,
            send_conn: Arc::new(Mutex::new(send_conn)),
            recv_conn: Arc::new(Mutex::new(recv_conn)),
            barrier_listener: Arc::new(Mutex::new(barrier_listener)),
            addresses: config.addresses,
        })
    }

    fn connect_with_retry(addr: &str, timeout: Duration) -> CommResult<TcpStream> {
        let start = std::time::Instant::now();
        let mut last_error = None;

        while start.elapsed() < timeout {
            match TcpStream::connect(addr) {
                Ok(stream) => {
                    stream.set_nodelay(true).ok();
                    return Ok(stream);
                }
                Err(e) => {
                    last_error = Some(e);
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        Err(CommError::ConnectionFailed(format!(
            "Failed to connect to {}: {:?}",
            addr, last_error
        )))
    }

    fn serialize_message(msg: &TensorMessage) -> Vec<u8> {
        let mut buf = Vec::new();

        // Write shape length and shape
        let shape_len = msg.shape.len() as u32;
        buf.extend_from_slice(&shape_len.to_le_bytes());
        for &dim in &msg.shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        // Write data length and data
        let data_len = msg.data.len() as u64;
        buf.extend_from_slice(&data_len.to_le_bytes());
        for &val in &msg.data {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        buf
    }

    fn deserialize_message(buf: &[u8]) -> CommResult<TensorMessage> {
        let mut pos = 0;

        if buf.len() < 4 {
            return Err(CommError::SerdeError("Buffer too small".to_string()));
        }

        // Read shape
        let shape_len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            if pos + 8 > buf.len() {
                return Err(CommError::SerdeError("Buffer too small for shape".to_string()));
            }
            let dim = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
            shape.push(dim);
            pos += 8;
        }

        // Read data
        if pos + 8 > buf.len() {
            return Err(CommError::SerdeError(
                "Buffer too small for data length".to_string(),
            ));
        }
        let data_len = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let mut data = Vec::with_capacity(data_len);
        for _ in 0..data_len {
            if pos + 4 > buf.len() {
                return Err(CommError::SerdeError("Buffer too small for data".to_string()));
            }
            let val = f32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            data.push(val);
            pos += 4;
        }

        Ok(TensorMessage::new(data, shape))
    }
}

impl Communicator for TcpComm {
    type Data = TensorMessage;

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, data: &Self::Data) -> CommResult<()> {
        let buf = Self::serialize_message(data);
        let len = buf.len() as u64;

        let mut conn = self.send_conn.lock().map_err(|_| {
            CommError::SendFailed("Failed to acquire send lock".to_string())
        })?;

        // Send length prefix
        conn.write_all(&len.to_le_bytes())
            .map_err(|e| CommError::SendFailed(format!("Failed to send length: {}", e)))?;

        // Send data
        conn.write_all(&buf)
            .map_err(|e| CommError::SendFailed(format!("Failed to send data: {}", e)))?;

        conn.flush()
            .map_err(|e| CommError::SendFailed(format!("Failed to flush: {}", e)))?;

        Ok(())
    }

    fn recv(&self) -> CommResult<Self::Data> {
        let mut conn = self.recv_conn.lock().map_err(|_| {
            CommError::RecvFailed("Failed to acquire recv lock".to_string())
        })?;

        // Read length prefix
        let mut len_buf = [0u8; 8];
        conn.read_exact(&mut len_buf)
            .map_err(|e| CommError::RecvFailed(format!("Failed to read length: {}", e)))?;
        let len = u64::from_le_bytes(len_buf) as usize;

        // Read data
        let mut buf = vec![0u8; len];
        conn.read_exact(&mut buf)
            .map_err(|e| CommError::RecvFailed(format!("Failed to read data: {}", e)))?;

        Self::deserialize_message(&buf)
    }

    fn barrier(&self) -> CommResult<()> {
        // Simple barrier: each rank notifies all others and waits
        // This is not the most efficient but works for correctness

        let barrier_port_offset = 1000u16;

        // Send barrier signal to all other ranks
        for i in 0..self.world_size {
            if i != self.rank {
                let base_addr = &self.addresses[i];
                let port: u16 = base_addr
                    .split(':')
                    .last()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(9000)
                    + barrier_port_offset;
                let addr = format!(
                    "{}:{}",
                    base_addr.split(':').next().unwrap_or("127.0.0.1"),
                    port
                );

                if let Ok(mut stream) = TcpStream::connect(&addr) {
                    let _ = stream.write_all(&[1u8]);
                }
            }
        }

        // Wait for signals from all other ranks
        let listener = self.barrier_listener.lock().map_err(|_| {
            CommError::RecvFailed("Failed to acquire barrier lock".to_string())
        })?;

        for _ in 0..(self.world_size - 1) {
            let (mut stream, _) = listener
                .accept()
                .map_err(|e| CommError::RecvFailed(format!("Barrier accept failed: {}", e)))?;
            let mut buf = [0u8; 1];
            let _ = stream.read_exact(&mut buf);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_tcp_serialization() {
        let msg = TensorMessage::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let buf = TcpComm::serialize_message(&msg);
        let decoded = TcpComm::deserialize_message(&buf).unwrap();

        assert_eq!(decoded.shape, msg.shape);
        assert_eq!(decoded.data, msg.data);
    }

    // Note: Full TCP ring tests require multiple processes or careful port management
    // and are better suited for integration tests
}
