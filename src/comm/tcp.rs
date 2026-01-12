//! TCP communication for multi-node ring attention.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

use burn::tensor::{Bytes, DType, TensorData};

use super::traits::{CommError, CommResult, Communicator};

const DTYPE_F64: u8 = 1;
const DTYPE_F32: u8 = 2;
const DTYPE_F16: u8 = 3;
const DTYPE_BF16: u8 = 4;
const DTYPE_I64: u8 = 5;
const DTYPE_I32: u8 = 6;
const DTYPE_I16: u8 = 7;
const DTYPE_I8: u8 = 8;
const DTYPE_U64: u8 = 9;
const DTYPE_U32: u8 = 10;
const DTYPE_U16: u8 = 11;
const DTYPE_U8: u8 = 12;
const DTYPE_BOOL: u8 = 13;
const DTYPE_QFLOAT: u8 = 14;

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
            return Err(CommError::InvalidConfig(
                "addresses must not be empty".to_string(),
            ));
        }

        let world_size = addresses.len();
        if rank >= world_size {
            return Err(CommError::InvalidConfig(format!(
                "rank {} >= world_size {}",
                rank, world_size
            )));
        }

        let bind_addr = &addresses[rank];
        let listener = TcpListener::bind(bind_addr).map_err(|e| {
            CommError::ConnectionFailed(format!("Failed to bind {}: {}", bind_addr, e))
        })?;

        let next_rank = (rank + 1) % world_size;
        let next_addr = &addresses[next_rank];
        let send_stream = TcpStream::connect(next_addr).map_err(|e| {
            CommError::ConnectionFailed(format!("Failed to connect to {}: {}", next_addr, e))
        })?;
        let _ = send_stream.set_nodelay(true);

        let (recv_stream, _) = listener.accept().map_err(|e| {
            CommError::ConnectionFailed(format!("Failed to accept on {}: {}", bind_addr, e))
        })?;
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

    fn send(&self, data: &TensorData) -> CommResult<()> {
        let payload = serialize_tensor_data(data)?;
        let len = payload.len() as u64;

        let mut stream = self.send_stream.lock().map_err(|_| {
            CommError::SendFailed("Failed to acquire send lock".to_string())
        })?;

        stream
            .write_all(&len.to_le_bytes())
            .map_err(|e| CommError::SendFailed(format!("Failed to send length: {}", e)))?;
        stream
            .write_all(&payload)
            .map_err(|e| CommError::SendFailed(format!("Failed to send payload: {}", e)))?;
        stream
            .flush()
            .map_err(|e| CommError::SendFailed(format!("Failed to flush: {}", e)))?;

        Ok(())
    }

    fn recv(&self) -> CommResult<TensorData> {
        let mut stream = self.recv_stream.lock().map_err(|_| {
            CommError::RecvFailed("Failed to acquire recv lock".to_string())
        })?;

        let mut len_buf = [0u8; 8];
        stream
            .read_exact(&mut len_buf)
            .map_err(|e| CommError::RecvFailed(format!("Failed to read length: {}", e)))?;
        let len = u64::from_le_bytes(len_buf) as usize;

        let mut buf = vec![0u8; len];
        stream
            .read_exact(&mut buf)
            .map_err(|e| CommError::RecvFailed(format!("Failed to read payload: {}", e)))?;

        deserialize_tensor_data(&buf)
    }

    fn barrier(&self) -> CommResult<()> {
        if self.world_size <= 1 {
            return Ok(());
        }

        let token = TensorData::new(Vec::<u8>::new(), [0]);

        if self.rank == 0 {
            self.send(&token)?;
            let _ = self.recv()?;
            self.send(&token)?;
            let _ = self.recv()?;
        } else {
            let _ = self.recv()?;
            self.send(&token)?;
            let _ = self.recv()?;
            self.send(&token)?;
        }

        Ok(())
    }
}

fn serialize_tensor_data(data: &TensorData) -> CommResult<Vec<u8>> {
    let mut buf = Vec::new();
    encode_dtype(&mut buf, data.dtype)?;

    write_u32(&mut buf, data.shape.len() as u32);
    for dim in &data.shape {
        write_u64(&mut buf, *dim as u64);
    }

    let bytes = data.as_bytes();
    write_u64(&mut buf, bytes.len() as u64);
    buf.extend_from_slice(bytes);

    Ok(buf)
}

fn deserialize_tensor_data(buf: &[u8]) -> CommResult<TensorData> {
    let mut pos = 0usize;
    let dtype = decode_dtype(buf, &mut pos)?;

    let shape_len = read_u32(buf, &mut pos)? as usize;
    let mut shape = Vec::with_capacity(shape_len);
    for _ in 0..shape_len {
        shape.push(read_u64(buf, &mut pos)? as usize);
    }

    let bytes_len = read_u64(buf, &mut pos)? as usize;
    let bytes = read_slice(buf, &mut pos, bytes_len)?.to_vec();

    Ok(TensorData::from_bytes(Bytes::from_bytes_vec(bytes), shape, dtype))
}

fn encode_dtype(buf: &mut Vec<u8>, dtype: DType) -> CommResult<()> {
    match dtype {
        DType::F64 => buf.push(DTYPE_F64),
        DType::F32 => buf.push(DTYPE_F32),
        DType::F16 => buf.push(DTYPE_F16),
        DType::BF16 => buf.push(DTYPE_BF16),
        DType::I64 => buf.push(DTYPE_I64),
        DType::I32 => buf.push(DTYPE_I32),
        DType::I16 => buf.push(DTYPE_I16),
        DType::I8 => buf.push(DTYPE_I8),
        DType::U64 => buf.push(DTYPE_U64),
        DType::U32 => buf.push(DTYPE_U32),
        DType::U16 => buf.push(DTYPE_U16),
        DType::U8 => buf.push(DTYPE_U8),
        DType::Bool => buf.push(DTYPE_BOOL),
        // Quantized types not supported in TCP comm yet
        _ => {
            return Err(CommError::Serialization(
                "Quantized types not supported in TCP comm".to_string(),
            ));
        }
    }

    Ok(())
}

fn decode_dtype(buf: &[u8], pos: &mut usize) -> CommResult<DType> {
    let tag = read_u8(buf, pos)?;
    match tag {
        DTYPE_F64 => Ok(DType::F64),
        DTYPE_F32 => Ok(DType::F32),
        DTYPE_F16 => Ok(DType::F16),
        DTYPE_BF16 => Ok(DType::BF16),
        DTYPE_I64 => Ok(DType::I64),
        DTYPE_I32 => Ok(DType::I32),
        DTYPE_I16 => Ok(DType::I16),
        DTYPE_I8 => Ok(DType::I8),
        DTYPE_U64 => Ok(DType::U64),
        DTYPE_U32 => Ok(DType::U32),
        DTYPE_U16 => Ok(DType::U16),
        DTYPE_U8 => Ok(DType::U8),
        DTYPE_BOOL => Ok(DType::Bool),
        DTYPE_QFLOAT => Err(CommError::Serialization(
            "Quantized types not supported in TCP comm".to_string(),
        )),
        _ => Err(CommError::Serialization(format!(
            "Unknown dtype tag {}",
            tag
        ))),
    }
}

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn read_u8(buf: &[u8], pos: &mut usize) -> CommResult<u8> {
    let slice = read_slice(buf, pos, 1)?;
    Ok(slice[0])
}

fn read_u32(buf: &[u8], pos: &mut usize) -> CommResult<u32> {
    let slice = read_slice(buf, pos, 4)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(buf: &[u8], pos: &mut usize) -> CommResult<u64> {
    let slice = read_slice(buf, pos, 8)?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_slice<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> CommResult<&'a [u8]> {
    if *pos + len > buf.len() {
        return Err(CommError::Serialization(
            "Buffer too small for decode".to_string(),
        ));
    }
    let slice = &buf[*pos..*pos + len];
    *pos += len;
    Ok(slice)
}
