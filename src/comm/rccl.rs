//! RCCL (ROCm Communication Collectives Library) dynamic loading.
//!
//! RCCL is AMD's equivalent of NVIDIA's NCCL, providing high-performance
//! multi-GPU communication primitives. This module dynamically loads
//! librccl.so at runtime, requiring only the AMD GPU driver.
//!
//! RCCL API is intentionally compatible with NCCL for easy porting.
//!
//! Zero-cost abstraction: raw bytes + shape, no wrapper types.

use std::ffi::{c_char, c_int, c_void};
use std::fmt;
use std::ptr;
use std::sync::OnceLock;

use libloading::Library;

use super::traits::{CommError, CommResult, Communicator};
use crate::hip_kernels::hsa_runtime::{
    find_gpu_agents, get_hsa_lib, hsa_init, GpuAgent, HsaRegion, HSA_STATUS_SUCCESS,
};

// RCCL type definitions (compatible with NCCL)
pub type RcclResult = c_int;
pub type RcclComm = *mut c_void;
pub type RcclDataType = c_int;
pub type RcclRedOp = c_int;

// RCCL result codes
pub const RCCL_SUCCESS: RcclResult = 0;

// RCCL data types
pub const RCCL_FLOAT32: RcclDataType = 7;
pub const RCCL_FLOAT16: RcclDataType = 6;

// RCCL reduce operations
pub const RCCL_SUM: RcclRedOp = 0;

// RCCL unique ID size (same as NCCL)
pub const RCCL_UNIQUE_ID_BYTES: usize = 128;

/// RCCL unique ID for communicator creation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RcclUniqueId {
    pub internal: [u8; RCCL_UNIQUE_ID_BYTES],
}

impl Default for RcclUniqueId {
    fn default() -> Self {
        Self {
            internal: [0u8; RCCL_UNIQUE_ID_BYTES],
        }
    }
}

// Function pointer types
type RcclGetUniqueIdFn = unsafe extern "C" fn(*mut RcclUniqueId) -> RcclResult;
type RcclCommInitRankFn =
    unsafe extern "C" fn(*mut RcclComm, c_int, RcclUniqueId, c_int) -> RcclResult;
type RcclCommDestroyFn = unsafe extern "C" fn(RcclComm) -> RcclResult;
type RcclCommCountFn = unsafe extern "C" fn(RcclComm, *mut c_int) -> RcclResult;
type RcclCommUserRankFn = unsafe extern "C" fn(RcclComm, *mut c_int) -> RcclResult;

type RcclSendFn = unsafe extern "C" fn(
    *const c_void, // sendbuff
    usize,         // count
    RcclDataType,  // datatype
    c_int,         // peer
    RcclComm,      // comm
    *mut c_void,   // stream (hipStream_t)
) -> RcclResult;

type RcclRecvFn = unsafe extern "C" fn(
    *mut c_void,  // recvbuff
    usize,        // count
    RcclDataType, // datatype
    c_int,        // peer
    RcclComm,     // comm
    *mut c_void,  // stream
) -> RcclResult;

type RcclAllReduceFn = unsafe extern "C" fn(
    *const c_void, // sendbuff
    *mut c_void,   // recvbuff
    usize,         // count
    RcclDataType,  // datatype
    RcclRedOp,     // op
    RcclComm,      // comm
    *mut c_void,   // stream
) -> RcclResult;

type RcclGroupStartFn = unsafe extern "C" fn() -> RcclResult;
type RcclGroupEndFn = unsafe extern "C" fn() -> RcclResult;

type RcclGetErrorStringFn = unsafe extern "C" fn(RcclResult) -> *const c_char;

/// RCCL library function table.
pub struct RcclLib {
    #[allow(dead_code)]
    lib: Library,

    pub rccl_get_unique_id: RcclGetUniqueIdFn,
    pub rccl_comm_init_rank: RcclCommInitRankFn,
    pub rccl_comm_destroy: RcclCommDestroyFn,
    pub rccl_comm_count: RcclCommCountFn,
    pub rccl_comm_user_rank: RcclCommUserRankFn,
    pub rccl_send: RcclSendFn,
    pub rccl_recv: RcclRecvFn,
    pub rccl_all_reduce: RcclAllReduceFn,
    pub rccl_group_start: RcclGroupStartFn,
    pub rccl_group_end: RcclGroupEndFn,
    pub rccl_get_error_string: RcclGetErrorStringFn,
}

// Safety: RcclLib contains function pointers from a loaded library.
// The library is loaded once and lives for the entire program lifetime.
unsafe impl Send for RcclLib {}
unsafe impl Sync for RcclLib {}

impl RcclLib {
    fn load() -> Result<Self, String> {
        let lib_names = [
            "librccl.so",
            "librccl.so.1",
            "/opt/rocm/lib/librccl.so",
            "/opt/rocm/lib64/librccl.so",
            "/usr/lib/x86_64-linux-gnu/librccl.so",
            "/usr/lib64/librccl.so",
        ];

        let lib = lib_names
            .iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| {
                "Failed to load RCCL library (librccl.so). \
                 Install ROCm with RCCL support."
                    .to_string()
            })?;

        // Load all function pointers
        let rccl_get_unique_id: RcclGetUniqueIdFn = unsafe {
            *lib.get::<RcclGetUniqueIdFn>(b"ncclGetUniqueId\0")
                .map_err(|e| format!("ncclGetUniqueId: {e}"))?
        };
        let rccl_comm_init_rank: RcclCommInitRankFn = unsafe {
            *lib.get::<RcclCommInitRankFn>(b"ncclCommInitRank\0")
                .map_err(|e| format!("ncclCommInitRank: {e}"))?
        };
        let rccl_comm_destroy: RcclCommDestroyFn = unsafe {
            *lib.get::<RcclCommDestroyFn>(b"ncclCommDestroy\0")
                .map_err(|e| format!("ncclCommDestroy: {e}"))?
        };
        let rccl_comm_count: RcclCommCountFn = unsafe {
            *lib.get::<RcclCommCountFn>(b"ncclCommCount\0")
                .map_err(|e| format!("ncclCommCount: {e}"))?
        };
        let rccl_comm_user_rank: RcclCommUserRankFn = unsafe {
            *lib.get::<RcclCommUserRankFn>(b"ncclCommUserRank\0")
                .map_err(|e| format!("ncclCommUserRank: {e}"))?
        };
        let rccl_send: RcclSendFn = unsafe {
            *lib.get::<RcclSendFn>(b"ncclSend\0")
                .map_err(|e| format!("ncclSend: {e}"))?
        };
        let rccl_recv: RcclRecvFn = unsafe {
            *lib.get::<RcclRecvFn>(b"ncclRecv\0")
                .map_err(|e| format!("ncclRecv: {e}"))?
        };
        let rccl_all_reduce: RcclAllReduceFn = unsafe {
            *lib.get::<RcclAllReduceFn>(b"ncclAllReduce\0")
                .map_err(|e| format!("ncclAllReduce: {e}"))?
        };
        let rccl_group_start: RcclGroupStartFn = unsafe {
            *lib.get::<RcclGroupStartFn>(b"ncclGroupStart\0")
                .map_err(|e| format!("ncclGroupStart: {e}"))?
        };
        let rccl_group_end: RcclGroupEndFn = unsafe {
            *lib.get::<RcclGroupEndFn>(b"ncclGroupEnd\0")
                .map_err(|e| format!("ncclGroupEnd: {e}"))?
        };
        let rccl_get_error_string: RcclGetErrorStringFn = unsafe {
            *lib.get::<RcclGetErrorStringFn>(b"ncclGetErrorString\0")
                .map_err(|e| format!("ncclGetErrorString: {e}"))?
        };

        Ok(Self {
            lib,
            rccl_get_unique_id,
            rccl_comm_init_rank,
            rccl_comm_destroy,
            rccl_comm_count,
            rccl_comm_user_rank,
            rccl_send,
            rccl_recv,
            rccl_all_reduce,
            rccl_group_start,
            rccl_group_end,
            rccl_get_error_string,
        })
    }
}

/// Global RCCL library instance.
static RCCL_LIB: OnceLock<Result<RcclLib, String>> = OnceLock::new();

/// Get the global RCCL library instance.
pub fn get_rccl_lib() -> Result<&'static RcclLib, &'static str> {
    RCCL_LIB
        .get_or_init(RcclLib::load)
        .as_ref()
        .map_err(|e| e.as_str())
}

/// Check if RCCL is available.
pub fn is_rccl_available() -> bool {
    get_rccl_lib().is_ok()
}

/// RCCL error type.
#[derive(Debug)]
pub enum RcclError {
    /// RCCL library not available.
    LibraryNotFound(String),
    /// RCCL operation failed.
    OperationFailed(String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// HSA/HIP error.
    HsaError(String),
}

impl fmt::Display for RcclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound(msg) => write!(f, "RCCL library not found: {}", msg),
            Self::OperationFailed(msg) => write!(f, "RCCL operation failed: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::HsaError(msg) => write!(f, "HSA error: {}", msg),
        }
    }
}

impl std::error::Error for RcclError {}

/// Get RCCL error string.
fn get_rccl_error_string(result: RcclResult) -> String {
    if let Ok(lib) = get_rccl_lib() {
        unsafe {
            let ptr = (lib.rccl_get_error_string)(result);
            if !ptr.is_null() {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            } else {
                format!("RCCL error code: {}", result)
            }
        }
    } else {
        format!("RCCL error code: {}", result)
    }
}

/// GPU memory buffer for RCCL operations.
struct GpuBuffer {
    ptr: *mut c_void,
    size: usize,
    region: HsaRegion,
}

impl GpuBuffer {
    fn new(size: usize, region: HsaRegion) -> Result<Self, RcclError> {
        let hsa = get_hsa_lib().map_err(|e| RcclError::HsaError(e.to_string()))?;

        let mut ptr: *mut c_void = ptr::null_mut();
        let status = unsafe { (hsa.hsa_memory_allocate)(region, size, &mut ptr) };

        if status != HSA_STATUS_SUCCESS {
            return Err(RcclError::HsaError(format!(
                "hsa_memory_allocate failed: {}",
                status
            )));
        }

        Ok(Self { ptr, size, region })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) -> Result<(), RcclError> {
        if data.len() > self.size {
            return Err(RcclError::InvalidConfig("Data too large for buffer".into()));
        }

        let hsa = get_hsa_lib().map_err(|e| RcclError::HsaError(e.to_string()))?;
        let status =
            unsafe { (hsa.hsa_memory_copy)(self.ptr, data.as_ptr() as *const c_void, data.len()) };

        if status != HSA_STATUS_SUCCESS {
            return Err(RcclError::HsaError(format!(
                "hsa_memory_copy (H2D) failed: {}",
                status
            )));
        }

        Ok(())
    }

    fn copy_to_host(&self, data: &mut [u8]) -> Result<(), RcclError> {
        if data.len() > self.size {
            return Err(RcclError::InvalidConfig("Buffer too small".into()));
        }

        let hsa = get_hsa_lib().map_err(|e| RcclError::HsaError(e.to_string()))?;
        let status = unsafe {
            (hsa.hsa_memory_copy)(data.as_mut_ptr() as *mut c_void, self.ptr, data.len())
        };

        if status != HSA_STATUS_SUCCESS {
            return Err(RcclError::HsaError(format!(
                "hsa_memory_copy (D2H) failed: {}",
                status
            )));
        }

        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(hsa) = get_hsa_lib() {
                unsafe {
                    (hsa.hsa_memory_free)(self.ptr);
                }
            }
        }
    }
}

/// RCCL Communicator for multi-GPU ring communication on ROCm.
pub struct RcclCommunicator {
    comm: RcclComm,
    rank: usize,
    world_size: usize,
    gpu_agent: GpuAgent,
}

// Safety: RCCL library is designed to be thread-safe.
unsafe impl Send for RcclCommunicator {}
unsafe impl Sync for RcclCommunicator {}

impl RcclCommunicator {
    /// Create a unique RCCL ID (call on rank 0, then broadcast to others).
    pub fn create_unique_id() -> Result<RcclUniqueId, RcclError> {
        let lib = get_rccl_lib().map_err(|e| RcclError::LibraryNotFound(e.to_string()))?;

        let mut id = RcclUniqueId::default();
        let result = unsafe { (lib.rccl_get_unique_id)(&mut id) };

        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(get_rccl_error_string(result)));
        }

        Ok(id)
    }

    /// Create a new RCCL communicator.
    ///
    /// For single-node multi-GPU, one process creates the ID and shares it.
    /// For multi-node, the ID must be broadcast from rank 0 to all nodes.
    pub fn new(
        id: RcclUniqueId,
        world_size: usize,
        rank: usize,
        device_id: usize,
    ) -> Result<Self, RcclError> {
        if world_size == 0 {
            return Err(RcclError::InvalidConfig("world_size must be > 0".into()));
        }
        if rank >= world_size {
            return Err(RcclError::InvalidConfig(format!(
                "rank {} >= world_size {}",
                rank, world_size
            )));
        }

        // Initialize HSA runtime
        hsa_init().map_err(|e| RcclError::HsaError(e.to_string()))?;

        // Get GPU agents
        let agents = find_gpu_agents().map_err(|e| RcclError::HsaError(e))?;
        if device_id >= agents.len() {
            return Err(RcclError::InvalidConfig(format!(
                "device_id {} >= num_gpus {}",
                device_id,
                agents.len()
            )));
        }
        let gpu_agent = agents[device_id].clone();

        let lib = get_rccl_lib().map_err(|e| RcclError::LibraryNotFound(e.to_string()))?;

        let mut comm: RcclComm = ptr::null_mut();
        let result = unsafe {
            (lib.rccl_comm_init_rank)(
                &mut comm,
                world_size as c_int,
                id,
                rank as c_int,
            )
        };

        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclCommInitRank failed: {}",
                get_rccl_error_string(result)
            )));
        }

        Ok(Self {
            comm,
            rank,
            world_size,
            gpu_agent,
        })
    }

    /// Get the underlying GPU agent.
    pub fn gpu_agent(&self) -> &GpuAgent {
        &self.gpu_agent
    }

    fn peer_rank(&self, rank: usize) -> Result<c_int, RcclError> {
        c_int::try_from(rank).map_err(|_| {
            RcclError::InvalidConfig(format!("rank {} exceeds RCCL limits", rank))
        })
    }

    /// Send f32 data to next rank using RCCL.
    fn rccl_send(&self, data: &[f32]) -> Result<(), RcclError> {
        let lib = get_rccl_lib().map_err(|e| RcclError::LibraryNotFound(e.to_string()))?;

        let next_rank = (self.rank + 1) % self.world_size;
        let next_rank = self.peer_rank(next_rank)?;

        // Allocate GPU buffer and copy data
        let byte_size = data.len() * std::mem::size_of::<f32>();
        let gpu_buf = GpuBuffer::new(byte_size, self.gpu_agent.fine_grained_region)?;

        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_size) };
        gpu_buf.copy_from_host(bytes)?;

        // RCCL send (using null stream for synchronous operation)
        let result = unsafe {
            (lib.rccl_send)(
                gpu_buf.as_ptr(),
                data.len(),
                RCCL_FLOAT32,
                next_rank,
                self.comm,
                ptr::null_mut(),
            )
        };

        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclSend failed: {}",
                get_rccl_error_string(result)
            )));
        }

        Ok(())
    }

    /// Receive f32 data from previous rank using RCCL.
    fn rccl_recv(&self, len: usize) -> Result<Vec<f32>, RcclError> {
        let lib = get_rccl_lib().map_err(|e| RcclError::LibraryNotFound(e.to_string()))?;

        let prev_rank = (self.rank + self.world_size - 1) % self.world_size;
        let prev_rank = self.peer_rank(prev_rank)?;

        // Allocate GPU buffer
        let byte_size = len * std::mem::size_of::<f32>();
        let gpu_buf = GpuBuffer::new(byte_size, self.gpu_agent.fine_grained_region)?;

        // RCCL recv
        let result = unsafe {
            (lib.rccl_recv)(
                gpu_buf.as_ptr(),
                len,
                RCCL_FLOAT32,
                prev_rank,
                self.comm,
                ptr::null_mut(),
            )
        };

        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclRecv failed: {}",
                get_rccl_error_string(result)
            )));
        }

        // Copy back to host
        let mut bytes = vec![0u8; byte_size];
        gpu_buf.copy_to_host(&mut bytes)?;

        // Reinterpret as f32
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(data)
    }

    /// Fused send and receive for ring pattern (more efficient).
    fn rccl_send_recv(&self, send_data: &[f32]) -> Result<Vec<f32>, RcclError> {
        let lib = get_rccl_lib().map_err(|e| RcclError::LibraryNotFound(e.to_string()))?;

        let next_rank = (self.rank + 1) % self.world_size;
        let prev_rank = (self.rank + self.world_size - 1) % self.world_size;
        let next_rank = self.peer_rank(next_rank)?;
        let prev_rank = self.peer_rank(prev_rank)?;
        let len = send_data.len();

        // Allocate GPU buffers
        let byte_size = len * std::mem::size_of::<f32>();
        let send_buf = GpuBuffer::new(byte_size, self.gpu_agent.fine_grained_region)?;
        let recv_buf = GpuBuffer::new(byte_size, self.gpu_agent.fine_grained_region)?;

        // Copy send data to GPU
        let bytes = unsafe {
            std::slice::from_raw_parts(send_data.as_ptr() as *const u8, byte_size)
        };
        send_buf.copy_from_host(bytes)?;

        // Group start
        let result = unsafe { (lib.rccl_group_start)() };
        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclGroupStart failed: {}",
                get_rccl_error_string(result)
            )));
        }

        // Send
        let result = unsafe {
            (lib.rccl_send)(
                send_buf.as_ptr(),
                len,
                RCCL_FLOAT32,
                next_rank,
                self.comm,
                ptr::null_mut(),
            )
        };
        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclSend failed: {}",
                get_rccl_error_string(result)
            )));
        }

        // Recv
        let result = unsafe {
            (lib.rccl_recv)(
                recv_buf.as_ptr(),
                len,
                RCCL_FLOAT32,
                prev_rank,
                self.comm,
                ptr::null_mut(),
            )
        };
        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclRecv failed: {}",
                get_rccl_error_string(result)
            )));
        }

        // Group end
        let result = unsafe { (lib.rccl_group_end)() };
        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclGroupEnd failed: {}",
                get_rccl_error_string(result)
            )));
        }

        // Copy back to host
        let mut recv_bytes = vec![0u8; byte_size];
        recv_buf.copy_to_host(&mut recv_bytes)?;

        // Reinterpret as f32
        let data: Vec<f32> = recv_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(data)
    }

    /// Perform all-reduce operation.
    fn rccl_all_reduce(&self, data: &[f32]) -> Result<Vec<f32>, RcclError> {
        let lib = get_rccl_lib().map_err(|e| RcclError::LibraryNotFound(e.to_string()))?;

        let byte_size = data.len() * std::mem::size_of::<f32>();
        let send_buf = GpuBuffer::new(byte_size, self.gpu_agent.fine_grained_region)?;
        let recv_buf = GpuBuffer::new(byte_size, self.gpu_agent.fine_grained_region)?;

        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_size) };
        send_buf.copy_from_host(bytes)?;

        let result = unsafe {
            (lib.rccl_all_reduce)(
                send_buf.as_ptr(),
                recv_buf.as_ptr(),
                data.len(),
                RCCL_FLOAT32,
                RCCL_SUM,
                self.comm,
                ptr::null_mut(),
            )
        };

        if result != RCCL_SUCCESS {
            return Err(RcclError::OperationFailed(format!(
                "ncclAllReduce failed: {}",
                get_rccl_error_string(result)
            )));
        }

        let mut recv_bytes = vec![0u8; byte_size];
        recv_buf.copy_to_host(&mut recv_bytes)?;

        let data: Vec<f32> = recv_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(data)
    }
}

impl Drop for RcclCommunicator {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            if let Ok(lib) = get_rccl_lib() {
                unsafe {
                    (lib.rccl_comm_destroy)(self.comm);
                }
            }
        }
    }
}

impl Communicator for RcclCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send_raw(&self, data: &[u8], shape: &[usize], dtype: u8) -> CommResult<()> {
        // Protocol: send metadata first, then data
        // Metadata: [ndim, shape..., dtype, data_len] as f32

        let ndim = shape.len();
        let metadata_f32_len = 2 + ndim + 1; // ndim + shape + dtype + data_len
        let mut metadata = Vec::with_capacity(metadata_f32_len);

        metadata.push(ndim as f32);
        for &dim in shape {
            metadata.push(dim as f32);
        }
        metadata.push(dtype as f32);
        metadata.push(data.len() as f32);

        // Send metadata
        self.rccl_send(&metadata)
            .map_err(|e| CommError::SendFailed(e.to_string()))?;

        // Convert bytes to f32 for RCCL transfer
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
        self.rccl_send(&data_f32)
            .map_err(|e| CommError::SendFailed(e.to_string()))?;

        Ok(())
    }

    fn recv_raw(&self) -> CommResult<(Vec<u8>, Vec<usize>, u8)> {
        // First, receive metadata (ndim)
        let ndim_vec = self.rccl_recv(1)
            .map_err(|e| CommError::RecvFailed(e.to_string()))?;
        let ndim = ndim_vec[0] as usize;

        // Receive rest of metadata: shape + dtype + data_len
        let rest_len = ndim + 2;
        let rest = self.rccl_recv(rest_len)
            .map_err(|e| CommError::RecvFailed(e.to_string()))?;

        let mut shape = Vec::with_capacity(ndim);
        for i in 0..ndim {
            shape.push(rest[i] as usize);
        }
        let dtype = rest[ndim] as u8;
        let data_len = rest[ndim + 1] as usize;

        // Receive data
        let f32_len = (data_len + 3) / 4;
        let data_f32 = self.rccl_recv(f32_len)
            .map_err(|e| CommError::RecvFailed(e.to_string()))?;

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

        // Use all-reduce as barrier
        let send_data = vec![0.0f32; 1];
        self.rccl_all_reduce(&send_data)
            .map_err(|e| CommError::ConnectionFailed(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rccl_availability() {
        // This test just checks if the library loading works
        // It will pass if RCCL is installed, skip otherwise
        let available = is_rccl_available();
        println!("RCCL available: {}", available);
    }
}
