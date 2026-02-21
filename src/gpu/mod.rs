//! GPU device abstraction layer (SPEC/04 §3–§6).
//!
//! Platform-agnostic traits for GPU compute backends. Concrete implementations
//! live behind feature gates (`cuda`, `metal`) and are not compiled unless
//! the corresponding feature is enabled.
//!
//! This module is always available — it contains only trait definitions,
//! error types, and the device-resident `GpuTensor` wrapper.

use std::fmt;
use std::marker::PhantomData;

use crate::traits::Element;

// ── Error ────────────────────────────────────────────────────────────

/// Errors from GPU device operations.
#[derive(Debug)]
pub enum GpuError {
    /// No suitable device found.
    DeviceNotFound(String),
    /// Device memory exhausted.
    OutOfMemory { requested: usize, available: usize },
    /// Kernel launch failure.
    KernelLaunch(String),
    /// Shader / PTX / MSL compilation failure.
    ShaderCompilation(String),
    /// Host↔device transfer failure.
    Transfer(String),
    /// Low-level driver error.
    Driver(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound(s) => write!(f, "device not found: {s}"),
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: requested {requested} bytes, {available} available")
            }
            Self::KernelLaunch(s) => write!(f, "kernel launch failed: {s}"),
            Self::ShaderCompilation(s) => write!(f, "shader compilation failed: {s}"),
            Self::Transfer(s) => write!(f, "transfer failed: {s}"),
            Self::Driver(s) => write!(f, "driver error: {s}"),
        }
    }
}

impl std::error::Error for GpuError {}

// ── Device buffer ────────────────────────────────────────────────────

/// A contiguous device-side memory allocation.
pub trait GpuBuffer: Send + Sync {
    /// Raw device pointer (passed to kernel launches).
    fn as_device_ptr(&self) -> u64;

    /// Size in bytes.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Execution stream / command queue ─────────────────────────────────

/// An ordered sequence of GPU operations (CUDA stream / Metal command buffer).
pub trait GpuStream: Send + Sync {
    /// Block the host until all enqueued work on this stream completes.
    fn synchronize(&self) -> Result<(), GpuError>;
}

// ── Device ───────────────────────────────────────────────────────────

/// A GPU device handle with memory management and stream creation.
///
/// Implementations wrap platform-specific handles (`cudarc::CudaDevice`,
/// `metal::Device`) and expose a uniform interface.
pub trait GpuDevice: Send + Sync + 'static {
    type Buffer: GpuBuffer;
    type Stream: GpuStream;

    /// Human-readable device name (e.g. "NVIDIA A100", "Apple M2 Pro").
    fn name(&self) -> &str;

    /// Total device memory in bytes.
    fn total_memory(&self) -> usize;

    /// Currently free device memory in bytes (best-effort estimate).
    fn free_memory(&self) -> usize;

    // ── Allocation ───────────────────────────────────────────────

    /// Allocate `bytes` of uninitialized device memory.
    fn alloc(&self, bytes: usize) -> Result<Self::Buffer, GpuError>;

    /// Allocate `bytes` of zero-filled device memory.
    fn alloc_zeros(&self, bytes: usize) -> Result<Self::Buffer, GpuError>;

    // ── Transfers ────────────────────────────────────────────────

    /// Copy host → device (async on `stream`).
    fn htod(
        &self,
        src: &[u8],
        dst: &mut Self::Buffer,
        stream: &Self::Stream,
    ) -> Result<(), GpuError>;

    /// Copy device → host (async on `stream`).
    fn dtoh(
        &self,
        src: &Self::Buffer,
        dst: &mut [u8],
        stream: &Self::Stream,
    ) -> Result<(), GpuError>;

    /// Copy device → device on the same device (async on `stream`).
    fn dtod(
        &self,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        stream: &Self::Stream,
    ) -> Result<(), GpuError>;

    // ── Streams ──────────────────────────────────────────────────

    /// Create a new execution stream / command queue.
    fn create_stream(&self) -> Result<Self::Stream, GpuError>;

    /// The default stream (created at device init).
    fn default_stream(&self) -> &Self::Stream;

    /// Synchronize all pending work across all streams.
    fn sync(&self) -> Result<(), GpuError>;
}

// ── GpuTensor ────────────────────────────────────────────────────────

/// A device-resident tensor that avoids host↔device round-trips between
/// operator calls.
///
/// Phase 1 of the GPU backend uses implicit transfers through the
/// `Kernels<E>` trait (host slices). Phase 2 will add `GpuKernelsExt`
/// that operates directly on `GpuTensor` for zero-copy operator chaining.
pub struct GpuTensor<E: Element, D: GpuDevice> {
    buffer: D::Buffer,
    /// Number of elements (not bytes).
    len: usize,
    _elem: PhantomData<E>,
}

impl<E: Element, D: GpuDevice> GpuTensor<E, D> {
    /// Allocate an uninitialized tensor of `len` elements on `device`.
    pub fn alloc(device: &D, len: usize) -> Result<Self, GpuError> {
        let bytes = len * std::mem::size_of::<E>();
        let buffer = device.alloc(bytes)?;
        Ok(Self { buffer, len, _elem: PhantomData })
    }

    /// Allocate a zero-filled tensor of `len` elements.
    pub fn zeros(device: &D, len: usize) -> Result<Self, GpuError> {
        let bytes = len * std::mem::size_of::<E>();
        let buffer = device.alloc_zeros(bytes)?;
        Ok(Self { buffer, len, _elem: PhantomData })
    }

    /// Upload from a host slice.
    pub fn from_slice(
        device: &D,
        data: &[E],
        stream: &D::Stream,
    ) -> Result<Self, GpuError> {
        let bytes = data.len() * std::mem::size_of::<E>();
        let mut buffer = device.alloc(bytes)?;
        let src = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
        };
        device.htod(src, &mut buffer, stream)?;
        Ok(Self { buffer, len: data.len(), _elem: PhantomData })
    }

    /// Download to a host `Vec`.
    pub fn to_vec(
        &self,
        device: &D,
        stream: &D::Stream,
    ) -> Result<Vec<E>, GpuError> {
        let mut out = vec![E::ZERO; self.len];
        let bytes = self.len * std::mem::size_of::<E>();
        let dst = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, bytes)
        };
        device.dtoh(&self.buffer, dst, stream)?;
        stream.synchronize()?;
        Ok(out)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for kernel launches.
    pub fn device_ptr(&self) -> u64 {
        self.buffer.as_device_ptr()
    }

    /// Borrow the underlying device buffer.
    pub fn buffer(&self) -> &D::Buffer {
        &self.buffer
    }

    /// Mutably borrow the underlying device buffer.
    pub fn buffer_mut(&mut self) -> &mut D::Buffer {
        &mut self.buffer
    }
}

// ── Dispatch threshold ───────────────────────────────────────────────

/// Minimum number of elements before dispatching to GPU.
///
/// Below this threshold, kernel launch overhead (~5–10μs CUDA, ~2–5μs Metal)
/// dominates and CPU execution is faster.
pub const GPU_DISPATCH_THRESHOLD: usize = 1024;
