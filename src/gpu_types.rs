use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use crate::runtime_detection::BackendType;

/// Unified wrapper for backend-specific GPU buffers.
///
/// This enum holds the actual device memory handle.
/// Using Arc ensures cheap cloning and shared ownership, required for weight sharing.
///
/// Note: All backends are unconditionally compiled (Fat Binary principle).
/// OS-specific variants use target_os conditions for type availability.
#[derive(Clone, Debug)]
pub enum GpuBuffer {
    /// WGPU buffer handle
    Wgpu(Arc<wgpu::Buffer>),

    /// CUDA buffer handle (device pointer)
    /// Always compiled - Fat Binary principle, no feature flags
    Cuda(Arc<cudarc::driver::CudaSlice<u8>>),

    /// Metal buffer handle (macOS only - OS limitation, not feature flag)
    #[cfg(target_os = "macos")]
    Metal(Arc<metal::Buffer>),

    /// ROCm/HIP buffer handle via HSA Runtime (Fat Binary: always compiled)
    Rocm(Arc<crate::hip_kernels::HsaBuffer<u8>>),

    /// CPU fallback (if allocation fails or for testing)
    Cpu(Arc<Vec<u8>>),
}

/// A tensor residing in GPU memory.
#[derive(Clone, Debug)]
pub struct GpuTensor {
    /// The underlying GPU buffer
    pub buffer: GpuBuffer,
    /// Shape definitions (e.g. [rows, cols])
    pub shape: Vec<usize>,
    /// Data type (f32/f16/u32/i8)
    pub dtype: TensorDtype,
    /// Total size in bytes
    pub size_in_bytes: usize,
    /// Backend where this tensor resides
    pub backend: BackendType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    F16,
    U32,
    I8,
}

/// Simple GPU memory pool to recycle buffers and avoid allocation overhead.
#[derive(Default)]
pub struct GpuMemoryPool {
    // Key: (BackendType, size_in_bytes)
    buffers: Mutex<HashMap<(BackendType, usize), Vec<GpuBuffer>>>,
}

impl GpuMemoryPool {
    pub fn get() -> &'static Self {
        static POOL: OnceLock<GpuMemoryPool> = OnceLock::new();
        POOL.get_or_init(Self::default)
    }

    pub fn take(&self, backend: BackendType, size: usize) -> Option<GpuBuffer> {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.get_mut(&(backend, size))?.pop()
    }

    pub fn put(&self, buffer: GpuBuffer, backend: BackendType, size: usize) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.entry((backend, size)).or_default().push(buffer);
    }
}

impl GpuTensor {
    pub fn new(buffer: GpuBuffer, shape: Vec<usize>, dtype: TensorDtype, backend: BackendType) -> Self {
        let element_size = match dtype {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            TensorDtype::U32 => 4,
            TensorDtype::I8 => 1,
        };
        let num_elements: usize = shape.iter().product();
        Self {
            buffer,
            shape,
            dtype,
            size_in_bytes: num_elements * element_size,
            backend,
        }
    }

    /// Create a temporary tensor from the memory pool.
    pub fn new_temp(shape: Vec<usize>, dtype: TensorDtype, backend: BackendType) -> Result<Self, String> {
        let element_size = match dtype {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            TensorDtype::U32 => 4,
            TensorDtype::I8 => 1,
        };
        let size = shape.iter().product::<usize>() * element_size;
        
        if let Some(buffer) = GpuMemoryPool::get().take(backend, size) {
             Ok(Self::new(buffer, shape, dtype, backend))
        } else {
             // If not in pool, we must allocate a NEW one but it's still "temp" in logic.
             // This happens on first few tokens.
             // We can't easily allocate here without dispatcher/agent context for all backends.
             // For now, let's assume the pool is warmed or we fallback to CPU?
             // Actually, new_temp should probably just return Option and the caller handles allocation.
             // But to keep gllm code clean:
             Err(format!("No pooled buffer available for size {}", size))
        }
    }

    /// Create a temporary tensor from a host slice, using the pool if possible.
    pub fn from_slice_temp<T: DeviceRepr + crate::types::KernelFloat>(
        data: &[T],
        shape: Vec<usize>,
        dtype: TensorDtype,
        backend: BackendType,
    ) -> Result<Self, String> {
        let mut tensor = Self::new_temp(shape, dtype, backend)?;
        let dispatcher = crate::KernelDispatcher::new();
        dispatcher.upload_to_tensor(data, &mut tensor)?;
        Ok(tensor)
    }

    /// Return the tensor's buffer to the pool.
    pub fn release(self) {
        GpuMemoryPool::get().put(self.buffer, self.backend, self.size_in_bytes);
    }

    /// Get the total number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Persistent KV cache on GPU.
#[derive(Clone, Debug)]
pub struct GpuKVCache {
    pub keys: Vec<GpuBuffer>,   // One buffer per layer
    pub values: Vec<GpuBuffer>, // One buffer per layer
    pub num_layers: usize,
    pub max_len: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub current_len: usize,
    pub backend: BackendType,
}

impl GpuKVCache {
    pub fn new(
        num_layers: usize,
        max_len: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        backend: BackendType,
    ) -> Result<Self, String> {
        let element_size = 4; // Assume f32 for now
        let layer_bytes = batch_size * num_heads * max_len * head_dim * element_size;
        
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        
        // Using dispatcher to allocate persistent buffers
        let dispatcher = crate::KernelDispatcher::new();
        for _ in 0..num_layers {
            keys.push(dispatcher.allocate_raw_buffer(layer_bytes)?);
            values.push(dispatcher.allocate_raw_buffer(layer_bytes)?);
        }

        Ok(Self {
            keys,
            values,
            num_layers,
            max_len,
            batch_size,
            num_heads,
            head_dim,
            current_len: 0,
            backend,
        })
    }

    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &GpuTensor,
        new_v: &GpuTensor,
    ) -> Result<(), String> {
        let dispatcher = crate::KernelDispatcher::new();
        dispatcher.update_kv_cache_gpu(self, layer_idx, new_k, new_v)
    }

    pub fn layer_keys(&self, layer: usize) -> &GpuBuffer {
        &self.keys[layer]
    }

    pub fn layer_values(&self, layer: usize) -> &GpuBuffer {
        &self.values[layer]
    }
}

// Helper trait for CUDA/HSA copy
pub trait DeviceRepr: Copy + Send + Sync {}
impl DeviceRepr for f32 {}
impl DeviceRepr for u8 {}
impl DeviceRepr for i32 {}
