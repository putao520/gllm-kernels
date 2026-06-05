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

#[cfg(feature = "jit-metal")]
pub mod metal;


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
        let bytes = std::mem::size_of_val(data);
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

// ── Platform-specific backends ───────────────────────────────────────

#[cfg(feature = "jit-cuda")]
pub mod cuda;

#[cfg(feature = "jit-hip")]
pub mod hip;

use crate::compiler::codegen::emitter::Platform;

/// GPU hardware capability flags for JIT codegen decisions.
#[derive(Debug, Clone, Default)]
pub struct GpuIsvCapabilities {
    /// Tensor Core / Matrix Unit generation (0=none, 1=Volta/sm70, 2=Ampere/sm80/CDNA1-2, 3=Hopper/sm90/CDNA3)
    pub tensor_core_gen: u8,
}

/// GPU 硬件能力描述，用于 codegen 决策（tile size、block size、指令选择）。
#[derive(Debug, Clone)]
pub struct GpuDeviceProfile {
    /// 目标平台
    pub platform: Platform,
    /// SM/CU/GPU core 数量
    pub compute_units: u32,
    /// Shared memory per block/threadgroup (bytes)
    pub shared_mem_per_block: u32,
    /// Max registers per thread
    pub max_registers_per_thread: u32,
    /// Warp/wavefront/SIMD-group 大小
    pub warp_size: u32,
    /// Max threads per block/threadgroup
    pub max_threads_per_block: u32,
    /// Max block dimensions (x, y, z)
    pub max_block_dim: [u32; 3],
    /// Max grid dimensions (x, y, z)
    pub max_grid_dim: [u32; 3],
    /// Total device memory (bytes)
    pub total_memory: usize,
    /// Memory bandwidth (GB/s, best-effort estimate)
    pub memory_bandwidth_gbs: f64,
    /// Peak compute (GFLOPS f32, best-effort estimate)
    pub peak_gflops_f32: f64,
    /// Peak compute (GFLOPS f16/bf16, best-effort estimate)
    pub peak_gflops_f16: f64,
    /// Has matrix acceleration unit (Tensor Core / MFMA / simdgroup_matrix)
    pub has_matrix_unit: bool,
    /// Clock rate (MHz) for roofline estimation
    pub clock_mhz: u32,
    /// ISV library availability (runtime detected)
    pub isv: GpuIsvCapabilities,
}

/// Kernel launch 参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid_dim: [u32; 3],
    pub block_dim: [u32; 3],
    pub shared_mem_bytes: u32,
}

impl GpuDeviceProfile {
    /// Peak GFLOPS for a given DType on GPU.
    pub fn peak_gflops(&self, dtype: crate::types::DType) -> f64 {
        use crate::types::DType;
        match dtype {
            DType::F32 => self.peak_gflops_f32,
            DType::F16 | DType::BF16 => self.peak_gflops_f16,
            DType::F8E4M3 | DType::F8E5M2 => self.peak_gflops_f16 * 2.0,
            DType::F6E3M2 | DType::F6E2M3 | DType::F4E2M1 => self.peak_gflops_f16 * 2.0,
            DType::U8 => self.peak_gflops_f32,
        }
    }

    /// 根据元素数量计算 1D elementwise launch 参数。
    pub fn launch_config_1d(&self, n_elements: usize) -> LaunchConfig {
        let block_x = self.max_threads_per_block.min(256);
        let grid_x = ((n_elements as u32) + block_x - 1) / block_x;
        let grid_x = grid_x.min(self.max_grid_dim[0]);
        LaunchConfig {
            grid_dim: [grid_x, 1, 1],
            block_dim: [block_x, 1, 1],
            shared_mem_bytes: 0,
        }
    }

    /// 根据行数和 block_size 计算 row-wise kernel launch 参数（softmax、normlike）。
    pub fn launch_config_row_wise(&self, num_rows: usize, block_size: u32) -> LaunchConfig {
        let block_x = block_size.min(self.max_threads_per_block);
        let grid_x = (num_rows as u32).min(self.max_grid_dim[0]);
        let shared_bytes = block_x * std::mem::size_of::<f32>() as u32; // float per thread
        LaunchConfig {
            grid_dim: [grid_x, 1, 1],
            block_dim: [block_x, 1, 1],
            shared_mem_bytes: shared_bytes,
        }
    }

    /// Roofline model: 计算 arithmetic intensity 阈值。
    pub fn roofline_ridge_point(&self) -> f64 {
        if self.memory_bandwidth_gbs > 0.0 {
            self.peak_gflops_f32 / self.memory_bandwidth_gbs
        } else {
            0.0
        }
    }
}

#[cfg(all(test, feature = "jit-cuda"))]
mod tests {
    use super::*;

    fn mock_cuda_profile() -> GpuDeviceProfile {
        use crate::compiler::codegen::emitter::Platform;
        GpuDeviceProfile {
            platform: Platform::Cuda { sm_version: 80 },
            compute_units: 108,
            shared_mem_per_block: 49152,
            max_registers_per_thread: 255,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [2147483647, 65535, 65535],
            total_memory: 40 * 1024 * 1024 * 1024, // 40 GB
            memory_bandwidth_gbs: 1555.0,
            peak_gflops_f32: 19500.0,
            peak_gflops_f16: 39000.0,
            has_matrix_unit: true,
            clock_mhz: 1410,
            isv: GpuIsvCapabilities::default(),
        }
    }

    #[test]
    fn test_gpu_device_profile_construction() {
        let p = mock_cuda_profile();
        assert_eq!(p.compute_units, 108);
        assert_eq!(p.warp_size, 32);
        assert!(p.has_matrix_unit);
        assert_eq!(p.max_threads_per_block, 1024);
    }

    #[test]
    fn test_launch_config_1d_small() {
        let p = mock_cuda_profile();
        let lc = p.launch_config_1d(512);
        assert_eq!(lc.block_dim, [256, 1, 1]);
        assert_eq!(lc.grid_dim, [2, 1, 1]);
        assert_eq!(lc.shared_mem_bytes, 0);
    }

    #[test]
    fn test_launch_config_1d_large() {
        let p = mock_cuda_profile();
        let lc = p.launch_config_1d(1_000_000);
        assert_eq!(lc.block_dim, [256, 1, 1]);
        assert_eq!(lc.grid_dim[0], 3907); // ceil(1M / 256)
    }

    #[test]
    fn test_launch_config_row_wise() {
        let p = mock_cuda_profile();
        let lc = p.launch_config_row_wise(32, 256);
        assert_eq!(lc.block_dim, [256, 1, 1]);
        assert_eq!(lc.grid_dim, [32, 1, 1]);
        assert_eq!(lc.shared_mem_bytes, 1024); // 256 * 4
    }

    #[test]
    fn test_roofline_ridge_point() {
        let p = mock_cuda_profile();
        let ridge = p.roofline_ridge_point();
        // A100: ~19500 GFLOPS / ~1555 GB/s ≈ 12.5 FLOP/byte
        assert!(ridge > 10.0 && ridge < 15.0, "ridge = {ridge}");
    }

    #[test]
    fn test_launch_config_equality() {
        let a = LaunchConfig { grid_dim: [1, 1, 1], block_dim: [256, 1, 1], shared_mem_bytes: 0 };
        let b = LaunchConfig { grid_dim: [1, 1, 1], block_dim: [256, 1, 1], shared_mem_bytes: 0 };
        assert_eq!(a, b);
    }

    // ── 13 new tests ──────────────────────────────────────────────────

    #[test]
    fn test_gpu_error_display_device_not_found() {
        // Arrange
        let err = GpuError::DeviceNotFound("no CUDA devices".into());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("device not found"), "msg = {msg}");
        assert!(msg.contains("no CUDA devices"), "msg = {msg}");
    }

    #[test]
    fn test_gpu_error_display_out_of_memory() {
        // Arrange
        let err = GpuError::OutOfMemory { requested: 1_073_741_824, available: 536_870_912 };
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("out of memory"), "msg = {msg}");
        assert!(msg.contains("1073741824"), "msg = {msg}");
        assert!(msg.contains("536870912"), "msg = {msg}");
    }

    #[test]
    fn test_gpu_error_display_kernel_launch() {
        // Arrange
        let err = GpuError::KernelLaunch("invalid config".into());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("kernel launch failed"), "msg = {msg}");
        assert!(msg.contains("invalid config"), "msg = {msg}");
    }

    #[test]
    fn test_gpu_error_display_shader_compilation() {
        // Arrange
        let err = GpuError::ShaderCompilation("syntax error at line 42".into());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("shader compilation failed"), "msg = {msg}");
        assert!(msg.contains("syntax error"), "msg = {msg}");
    }

    #[test]
    fn test_gpu_error_display_transfer() {
        // Arrange
        let err = GpuError::Transfer("PCIe link down".into());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("transfer failed"), "msg = {msg}");
        assert!(msg.contains("PCIe link down"), "msg = {msg}");
    }

    #[test]
    fn test_gpu_error_display_driver() {
        // Arrange
        let err = GpuError::Driver("CUDA driver version mismatch".into());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("driver error"), "msg = {msg}");
        assert!(msg.contains("version mismatch"), "msg = {msg}");
    }

    #[test]
    fn test_gpu_isv_capabilities_default() {
        // Arrange — default construction
        let isv = GpuIsvCapabilities::default();
        // Assert — tensor_core_gen should be 0 (no tensor cores)
        assert_eq!(isv.tensor_core_gen, 0, "default tensor_core_gen must be 0");
    }

    #[test]
    fn test_peak_gflops_f32_and_f16() {
        // Arrange
        use crate::types::DType;
        let p = mock_cuda_profile(); // peak_gflops_f32=19500, peak_gflops_f16=39000
        // Act
        let f32_gflops = p.peak_gflops(DType::F32);
        let f16_gflops = p.peak_gflops(DType::F16);
        let bf16_gflops = p.peak_gflops(DType::BF16);
        // Assert
        assert!((f32_gflops - 19500.0).abs() < 1e-6, "f32 = {f32_gflops}");
        assert!((f16_gflops - 39000.0).abs() < 1e-6, "f16 = {f16_gflops}");
        assert!((bf16_gflops - 39000.0).abs() < 1e-6, "bf16 = {bf16_gflops}");
    }

    #[test]
    fn test_peak_gflops_sub_byte_dtypes() {
        // Arrange
        use crate::types::DType;
        let p = mock_cuda_profile(); // peak_gflops_f16=39000; sub-byte = 2x f16
        // Act
        let fp8_e4m3 = p.peak_gflops(DType::F8E4M3);
        let fp8_e5m2 = p.peak_gflops(DType::F8E5M2);
        let fp6_e3m2 = p.peak_gflops(DType::F6E3M2);
        let fp6_e2m3 = p.peak_gflops(DType::F6E2M3);
        let fp4 = p.peak_gflops(DType::F4E2M1);
        // Assert — all sub-byte types are 2x f16 peak
        let expected = 39000.0 * 2.0;
        for (name, val) in [("f8e4m3", fp8_e4m3), ("f8e5m2", fp8_e5m2),
                            ("f6e3m2", fp6_e3m2), ("f6e2m3", fp6_e2m3), ("fp4", fp4)] {
            assert!((val - expected).abs() < 1e-6, "{name} = {val}, expected {expected}");
        }
    }

    #[test]
    fn test_roofline_ridge_point_zero_bandwidth() {
        // Arrange
        let mut p = mock_cuda_profile();
        p.memory_bandwidth_gbs = 0.0;
        // Act
        let ridge = p.roofline_ridge_point();
        // Assert — should return 0.0 to avoid division by zero
        assert!((ridge - 0.0).abs() < 1e-9, "ridge = {ridge}");
    }

    #[test]
    fn test_launch_config_1d_zero_elements() {
        // Arrange
        let p = mock_cuda_profile();
        // Act
        let lc = p.launch_config_1d(0);
        // Assert — ceil(0/256) = 0 grid, block is still 256
        assert_eq!(lc.grid_dim, [0, 1, 1], "grid for 0 elements should be [0,1,1]");
        assert_eq!(lc.block_dim, [256, 1, 1]);
        assert_eq!(lc.shared_mem_bytes, 0);
    }

    #[test]
    fn test_launch_config_1d_exact_multiple() {
        // Arrange
        let p = mock_cuda_profile(); // block_x = 256
        let n = 256 * 4; // 1024 — exact multiple of block size
        // Act
        let lc = p.launch_config_1d(n);
        // Assert
        assert_eq!(lc.grid_dim, [4, 1, 1], "exact 4 blocks");
        assert_eq!(lc.block_dim, [256, 1, 1]);
    }

    #[test]
    fn test_launch_config_row_wise_grid_clamped() {
        // Arrange — profile with max_grid_dim[0] = 1024 for easy clamping
        use crate::compiler::codegen::emitter::Platform;
        let p = GpuDeviceProfile {
            platform: Platform::Cuda { sm_version: 80 },
            compute_units: 108,
            shared_mem_per_block: 49152,
            max_registers_per_thread: 255,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [1024, 65535, 65535], // low grid_x limit
            total_memory: 40 * 1024 * 1024 * 1024,
            memory_bandwidth_gbs: 1555.0,
            peak_gflops_f32: 19500.0,
            peak_gflops_f16: 39000.0,
            has_matrix_unit: true,
            clock_mhz: 1410,
            isv: GpuIsvCapabilities::default(),
        };
        // Act — request 2000 rows, but max_grid_dim[0] = 1024
        let lc = p.launch_config_row_wise(2000, 256);
        // Assert
        assert_eq!(lc.grid_dim[0], 1024, "grid should be clamped to max_grid_dim[0]");
        assert_eq!(lc.block_dim, [256, 1, 1]);
    }
}
