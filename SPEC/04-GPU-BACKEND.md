# SPEC/04 — GPU Backend Architecture

## §1 Overview

This document specifies the GPU compute backend for `gllm-kernels`, extending the existing `Backend` / `Kernels<E>` trait hierarchy (SPEC/03 §2.2–2.3) to CUDA (NVIDIA) and Metal (Apple Silicon) accelerators.

### §1.1 Design Goals

1. **Unified trait surface** — GPU backends implement the same `Kernels<E>` trait as `CpuKernels<E>`. User code is generic over `B: Backend`.
2. **Async command-buffer model** — All kernel launches return immediately; synchronization is explicit (`stream.sync()` / `command_buffer.wait_until_completed()`).
3. **Zero-copy where possible** — Host↔device transfers use pinned/page-locked memory. Device tensors stay on-device across operator chains.
4. **Feature-gated compilation** — `cuda` and `metal` are Cargo features; the crate compiles on any platform without GPU SDKs installed.
5. **Runtime kernel selection** — PTX/MSL kernels are embedded at compile time (`include_str!`) or compiled at first use via NVRTC / Metal shader compilation.

### §1.2 Non-Goals

- Multi-GPU (tensor parallelism, pipeline parallelism) — deferred to a future SPEC.
- Vulkan/OpenCL compute — out of scope.
- Custom autograd / backward pass — this crate is inference-only.

---

## §2 Cargo Feature Gates

```toml
[features]
default = []
cuda = ["dep:cudarc"]
metal = ["dep:metal", "dep:objc"]

[target.'cfg(target_os = "linux")'.dependencies]
cudarc = { version = "0.12", optional = true, features = ["driver", "nvrtc"] }

[target.'cfg(target_os = "macos")'.dependencies]
metal = { version = "0.29", optional = true }
objc = { version = "0.2", optional = true }
```

All GPU modules are gated:

```rust
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal_backend;
```

---

## §3 Device Abstraction Layer

### §3.1 `GpuDevice` Trait

A thin abstraction over device handles, shared by both backends:

```rust
pub trait GpuDevice: Send + Sync + 'static {
    type Buffer: GpuBuffer;
    type Stream: GpuStream;

    fn name(&self) -> &str;
    fn total_memory(&self) -> usize;
    fn free_memory(&self) -> usize;

    /// Allocate device memory (uninitialized).
    fn alloc(&self, bytes: usize) -> Result<Self::Buffer, GpuError>;

    /// Allocate and zero-fill.
    fn alloc_zeros(&self, bytes: usize) -> Result<Self::Buffer, GpuError>;

    /// Copy host → device.
    fn htod(&self, src: &[u8], dst: &mut Self::Buffer, stream: &Self::Stream) -> Result<(), GpuError>;

    /// Copy device → host.
    fn dtoh(&self, src: &Self::Buffer, dst: &mut [u8], stream: &Self::Stream) -> Result<(), GpuError>;

    /// Copy device → device (same device).
    fn dtod(&self, src: &Self::Buffer, dst: &mut Self::Buffer, stream: &Self::Stream) -> Result<(), GpuError>;

    /// Create an execution stream / command queue.
    fn create_stream(&self) -> Result<Self::Stream, GpuError>;

    /// Default stream.
    fn default_stream(&self) -> &Self::Stream;

    /// Synchronize all pending work on the device.
    fn sync(&self) -> Result<(), GpuError>;
}
```

### §3.2 `GpuBuffer` Trait

```rust
pub trait GpuBuffer: Send + Sync {
    /// Raw device pointer (for kernel launch).
    fn as_device_ptr(&self) -> u64;
    /// Size in bytes.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
```

### §3.3 `GpuStream` Trait

```rust
pub trait GpuStream: Send + Sync {
    /// Block host until all enqueued work completes.
    fn synchronize(&self) -> Result<(), GpuError>;
}
```

### §3.4 Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("device not found: {0}")]
    DeviceNotFound(String),
    #[error("out of memory: requested {requested} bytes, {available} available")]
    OutOfMemory { requested: usize, available: usize },
    #[error("kernel launch failed: {0}")]
    KernelLaunch(String),
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),
    #[error("transfer failed: {0}")]
    Transfer(String),
    #[error("driver error: {0}")]
    Driver(String),
}
```

---

## §4 CUDA Backend (`feature = "cuda"`)

### §4.1 Module Layout

```
src/cuda/
├── mod.rs              // CudaBackend, CudaDevice, re-exports
├── device.rs           // GpuDevice impl over cudarc::driver::CudaDevice
├── kernels.rs          // Kernels<E> impl: CudaKernels<E>
├── launch.rs           // Grid/block helpers, occupancy calculator
├── shaders/
│   ├── elementwise.cu  // vec_add, vec_mul, silu, gelu, relu, etc.
│   ├── reduce.cu       // vec_sum, vec_max, softmax
│   ├── norm.cu         // rms_norm, layer_norm
│   ├── gemm.cu         // tiled GEMM (fallback; prefer cuBLAS)
│   ├── gemv.cu         // GEMV kernels
│   ├── rope.cu         // RoPE positional encoding
│   ├── quant.cu        // dequantization (Q4_K, Q8_K, etc.)
│   └── quant_gemv.cu   // fused dequant+GEMV
└── ptx/                // Pre-compiled PTX (optional, for offline builds)
```

### §4.2 `CudaDevice` — `GpuDevice` Implementation

```rust
#[cfg(feature = "cuda")]
pub struct CudaDeviceWrapper {
    inner: Arc<cudarc::driver::CudaDevice>,
    default_stream: CudaStreamWrapper,
}

impl GpuDevice for CudaDeviceWrapper {
    type Buffer = CudaBufferWrapper;
    type Stream = CudaStreamWrapper;
    // ... delegates to cudarc
}
```

Key design decisions:
- Uses `cudarc` with `driver` + `nvrtc` features (runtime PTX compilation).
- `cudarc::driver::CudaDevice` is `Arc`-wrapped internally; our wrapper adds the `GpuDevice` trait surface.
- Kernel modules are loaded lazily via `OnceLock<CudaFunction>` per kernel.

### §4.3 `CudaKernels<E>` — `Kernels<E>` Implementation

```rust
pub struct CudaKernels<E: Element> {
    device: Arc<CudaDeviceWrapper>,
    _phantom: PhantomData<E>,
}

impl<E: Element> Kernels<E> for CudaKernels<E> {
    fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]) { ... }
    fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize) { ... }
    fn silu(&self, a: &[E], out: &mut [E]) { ... }
    fn dequant_q4_k(&self, block: &[u8], out: &mut [f32]) { ... }
    fn dequant_q8_k(&self, block: &[u8], out: &mut [f32]) { ... }
    // ... all 70+ operators
}
```

**Kernel launch pattern** (internal):

```rust
fn launch_elementwise<E: Element>(
    device: &CudaDeviceWrapper,
    kernel_name: &str,
    a: &[E], b: &[E], out: &mut [E],
) -> Result<(), GpuError> {
    let n = a.len();
    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let d_a = device.htod_sync(a)?;
    let d_b = device.htod_sync(b)?;
    let mut d_out = device.alloc(n * std::mem::size_of::<E>())?;

    let func = device.get_or_load_func(kernel_name, ELEMENTWISE_PTX)?;
    unsafe {
        func.launch(
            LaunchConfig::for_num_elems(n as u32),
            (&d_a, &d_b, &mut d_out, n as u32),
        )?;
    }

    device.dtoh_sync(&d_out, out)?;
    Ok(())
}
```

### §4.4 GEMM Strategy

| M×N×K range | Strategy |
|---|---|
| Any (default) | cuBLAS `sgemm` / `hgemm` via `cudarc::cublas` |
| M=1 (GEMV) | Custom CUDA GEMV kernel (better for single-token decode) |
| Quantized | Fused dequant+GEMV kernel (§4.6) |

cuBLAS is preferred for dense GEMM — no point reimplementing what NVIDIA already optimized. The custom kernels exist for:
- Quantized weight formats (no cuBLAS equivalent)
- Fused epilogues (bias + activation in one pass)
- GEMV with M=1 (cuBLAS overhead dominates at small M)

### §4.5 Kernel Compilation

Two modes:
1. **NVRTC (default)** — `.cu` sources compiled to PTX at first kernel launch, cached in `OnceLock`.
2. **Pre-compiled PTX** — `include_str!("ptx/kernel.ptx")` for offline/hermetic builds. Selected via `cuda-precompiled` feature flag.

```rust
static ELEMENTWISE_PTX: OnceLock<CudaModule> = OnceLock::new();

fn get_elementwise_module(device: &CudaDeviceWrapper) -> &CudaModule {
    ELEMENTWISE_PTX.get_or_init(|| {
        device.compile_ptx(include_str!("shaders/elementwise.cu"))
            .expect("elementwise.cu compilation failed")
    })
}
```

### §4.6 Quantized Kernels

Fused dequant+dot kernels for all GGML quant formats:

| Format | Block size | Kernel | Notes |
|---|---|---|---|
| Q4_K | 256 | `dequant_q4k_f32` / `fused_gemv_q4k` | Super-blocks, 4-bit + 6-bit scales |
| Q8_K | 256 | `dequant_q8k_f32` / `fused_gemv_q8k` | 8-bit quantized |
| Q2_K–Q6_K | 256 | `dequant_qNk_f32` | K-quant family |
| Q4_0–Q8_1 | 32 | `dequant_qN_M_f32` | Classic GGML |
| IQ1–IQ4 | varies | `dequant_iqN_f32` | Importance quant |

Each dequant kernel:
1. One thread-block per super-block (256 elements for K-quants, 32 for classic).
2. Shared memory for scale/min unpacking.
3. Output to global f32 buffer.

Fused GEMV kernels skip the intermediate f32 buffer — dequant + dot in registers.

---

## §5 Metal Backend (`feature = "metal"`)

### §5.1 Module Layout

```
src/metal_backend/
├── mod.rs              // MetalBackend, MetalDevice, re-exports
├── device.rs           // GpuDevice impl over metal::Device
├── kernels.rs          // Kernels<E> impl: MetalKernels<E>
├── pipeline.rs         // Pipeline state cache (MSL → PSO)
├── shaders/
│   ├── elementwise.metal
│   ├── reduce.metal
│   ├── norm.metal
│   ├── gemm.metal
│   ├── gemv.metal
│   ├── rope.metal
│   ├── quant.metal
│   └── quant_gemv.metal
└── metallib/           // Pre-compiled .metallib (optional)
```

### §5.2 `MetalDevice` — `GpuDevice` Implementation

```rust
#[cfg(feature = "metal")]
pub struct MetalDeviceWrapper {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    pipeline_cache: DashMap<String, metal::ComputePipelineState>,
}

impl GpuDevice for MetalDeviceWrapper {
    type Buffer = MetalBufferWrapper;
    type Stream = MetalCommandBuffer;
    // ...
}
```

Key design decisions:
- `metal::Device::system_default()` for device acquisition.
- Pipeline states (compiled shaders) cached in `DashMap` by kernel name.
- Command buffers are the "stream" equivalent — created per-batch, committed, waited.

### §5.3 `MetalKernels<E>` — `Kernels<E>` Implementation

Same structure as CUDA. Metal-specific considerations:

- **Threadgroup sizes**: Metal uses `threads_per_threadgroup` (≈ CUDA block) and `threadgroups_per_grid` (≈ CUDA grid). Max threadgroup size is typically 1024.
- **SIMD-groups**: Metal's equivalent of warps (32 threads on Apple GPU). Reductions use `simd_shuffle` / `simd_sum`.
- **Shared memory**: `threadgroup` address space in MSL, declared in kernel signature.

### §5.4 MSL Shader Compilation

```rust
fn get_or_create_pipeline(
    device: &MetalDeviceWrapper,
    kernel_name: &str,
    source: &str,
) -> Result<metal::ComputePipelineState, GpuError> {
    if let Some(pso) = device.pipeline_cache.get(kernel_name) {
        return Ok(pso.clone());
    }
    let library = device.device.new_library_with_source(source, &metal::CompileOptions::new())
        .map_err(|e| GpuError::ShaderCompilation(e.to_string()))?;
    let func = library.get_function(kernel_name, None)
        .map_err(|e| GpuError::KernelLaunch(e.to_string()))?;
    let pso = device.device.new_compute_pipeline_state_with_function(&func)
        .map_err(|e| GpuError::KernelLaunch(e.to_string()))?;
    device.pipeline_cache.insert(kernel_name.to_string(), pso.clone());
    Ok(pso)
}
```

### §5.5 GEMM Strategy (Metal)

| M×N×K range | Strategy |
|---|---|
| Large dense | MPS (Metal Performance Shaders) `MPSMatrixMultiplication` |
| M=1 (GEMV) | Custom MSL GEMV kernel |
| Quantized | Fused dequant+GEMV MSL kernel |

MPS is Apple's equivalent of cuBLAS — use it for dense matmul, custom kernels for everything else.

### §5.6 Memory Model

Metal uses a unified memory architecture (UMA) on Apple Silicon:
- `MTLResourceStorageModeShared` — CPU and GPU share the same physical memory. Zero-copy for host↔device.
- No explicit `htod`/`dtoh` needed for shared buffers — just `contents()` pointer.
- For `GpuDevice` trait compliance, `htod`/`dtoh` become `memcpy` into shared buffers.

```rust
impl GpuDevice for MetalDeviceWrapper {
    fn htod(&self, src: &[u8], dst: &mut Self::Buffer, _stream: &Self::Stream) -> Result<(), GpuError> {
        // UMA: direct memcpy into shared buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.buffer.contents() as *mut u8,
                src.len(),
            );
        }
        Ok(())
    }
}
```

---

## §6 `GpuTensor<E, D>` — Device-Resident Tensor

To avoid round-tripping through host slices on every operator call, GPU backends use a device-resident tensor wrapper:

```rust
pub struct GpuTensor<E: Element, D: GpuDevice> {
    buffer: D::Buffer,
    len: usize,       // number of elements (not bytes)
    _elem: PhantomData<E>,
}

impl<E: Element, D: GpuDevice> GpuTensor<E, D> {
    /// Allocate on device, uninitialized.
    pub fn alloc(device: &D, len: usize) -> Result<Self, GpuError> { ... }

    /// Upload from host slice.
    pub fn from_slice(device: &D, data: &[E], stream: &D::Stream) -> Result<Self, GpuError> { ... }

    /// Download to host Vec.
    pub fn to_vec(&self, device: &D, stream: &D::Stream) -> Result<Vec<E>, GpuError> { ... }

    pub fn len(&self) -> usize { self.len }
    pub fn device_ptr(&self) -> u64 { self.buffer.as_device_ptr() }
}
```

### §6.1 Kernels Trait Adaptation

The existing `Kernels<E>` trait uses host slices (`&[E]`, `&mut [E]`). For GPU, this means implicit htod→kernel→dtoh per call — correct but slow for operator chains.

Two-phase approach:
1. **Phase 1 (this SPEC)**: Implement `Kernels<E>` with implicit transfers. Functionally correct, enables all existing tests to pass.
2. **Phase 2 (future SPEC)**: Add `GpuKernels<E>` extension trait with `GpuTensor` parameters for zero-copy operator chaining.

```rust
/// Phase 2 extension (future):
pub trait GpuKernelsExt<E: Element>: Kernels<E> {
    type Device: GpuDevice;

    fn vec_add_gpu(&self, a: &GpuTensor<E, Self::Device>, b: &GpuTensor<E, Self::Device>,
                   out: &mut GpuTensor<E, Self::Device>) -> Result<(), GpuError>;
    fn gemm_gpu(&self, a: &GpuTensor<E, Self::Device>, b: &GpuTensor<E, Self::Device>,
                c: &mut GpuTensor<E, Self::Device>, m: usize, n: usize, k: usize) -> Result<(), GpuError>;
    // ...
}
```

---

## §7 Backend Registration

### §7.1 `CudaBackend` / `MetalBackend`

```rust
#[cfg(feature = "cuda")]
pub struct CudaBackend;

#[cfg(feature = "cuda")]
impl Backend for CudaBackend {
    const NAME: &'static str = "cuda";
    type Kernels<E: Element> = CudaKernels<E>;

    fn init<E: Element>() -> Self::Kernels<E> {
        CudaKernels::new(CudaDeviceWrapper::default_device()
            .expect("CUDA device initialization failed"))
    }
}

#[cfg(feature = "metal")]
pub struct MetalBackend;

#[cfg(feature = "metal")]
impl Backend for MetalBackend {
    const NAME: &'static str = "metal";
    type Kernels<E: Element> = MetalKernels<E>;

    fn init<E: Element>() -> Self::Kernels<E> {
        MetalKernels::new(MetalDeviceWrapper::system_default()
            .expect("Metal device initialization failed"))
    }
}
```

### §7.2 `CpuBackend` (existing, for reference)

```rust
pub struct CpuBackend;

impl Backend for CpuBackend {
    const NAME: &'static str = "cpu";
    type Kernels<E: Element> = CpuKernels<E>;

    fn init<E: Element>() -> Self::Kernels<E> {
        CpuKernels::new()
    }
}
```

---

## §8 Kernel Implementation Priority

Ordered by inference-critical-path impact:

| Priority | Kernel group | Operators | Notes |
|---|---|---|---|
| P0 | Dense GEMM | `gemm`, `gemm_bt`, `gemm_bias`, `gemm_bias_act` | Delegates to cuBLAS/MPS |
| P0 | Dequant | `dequant_q4_k`, `dequant_q8_k`, all K-quant/classic/IQ | Required for quantized models |
| P0 | Quant matmul | `kquant_matmul`, `classic_matmul`, `iq_matmul` | Fused dequant+GEMV |
| P1 | Normalization | `rms_norm`, `layer_norm` | Per-token, latency-sensitive |
| P1 | Activations | `silu`, `gelu`, `relu`, `swiglu`, `softmax` | Fused where possible |
| P1 | RoPE | `rope`, `rope_with_pos` | Positional encoding |
| P2 | BLAS-1 | `vec_add`, `vec_mul`, `vec_dot`, `vec_scale`, `vec_axpy` | Element-wise |
| P2 | Reductions | `vec_sum`, `vec_max`, `vec_sum_squares` | Single-pass reduce |
| P3 | Quantized GEMV | `gemv_q8`, `gemv_q4`, `gemv_q2`, `gemv_q1` | INT8/INT4 paths |
| P3 | AWQ/GPTQ | `awq_matmul`, `gptq_matmul`, `squeeze_matmul` | External quant formats |

---

## §9 Testing Strategy

### §9.1 Cross-Backend Correctness

All GPU kernel outputs are validated against CPU reference implementations:

```rust
#[cfg(test)]
fn test_kernel_cross_backend<F>(name: &str, cpu_fn: F, gpu_fn: F, tolerance: f32)
where F: Fn(&dyn Kernels<f32>, &[f32], &[f32], &mut [f32])
{
    let cpu_k = CpuKernels::<f32>::new();
    let gpu_k = GpuKernels::<f32>::new(...);

    let a = random_vec(1024);
    let b = random_vec(1024);
    let mut cpu_out = vec![0.0; 1024];
    let mut gpu_out = vec![0.0; 1024];

    cpu_fn(&cpu_k, &a, &b, &mut cpu_out);
    gpu_fn(&gpu_k, &a, &b, &mut gpu_out);

    assert_approx_eq(&cpu_out, &gpu_out, tolerance);
}
```

### §9.2 Tolerance Thresholds

| Precision | Element-wise ops | Reductions | GEMM |
|---|---|---|---|
| f32 | 1e-6 | 1e-5 | 1e-4 (accumulation error) |
| f16 | 1e-3 | 1e-2 | 1e-2 |

### §9.3 CI Configuration

- CUDA tests: Run on GPU-enabled CI runners (or skipped with `#[cfg(feature = "cuda")]`).
- Metal tests: Run on macOS CI runners (or skipped with `#[cfg(feature = "metal")]`).
- CPU tests: Always run, serve as ground truth.

---

## §10 Performance Considerations

### §10.1 Launch Overhead Amortization

GPU kernel launch overhead (~5–10μs CUDA, ~2–5μs Metal) dominates for small tensors. Rules:

- Tensors < 1024 elements: Fall back to CPU (configurable threshold).
- Tensors 1K–64K: Single threadblock, minimize grid overhead.
- Tensors > 64K: Full grid occupancy.

```rust
const GPU_DISPATCH_THRESHOLD: usize = 1024;

fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]) {
    if a.len() < GPU_DISPATCH_THRESHOLD {
        // Fall back to CPU kernel
        self.cpu_fallback.vec_add(a, b, out);
        return;
    }
    // GPU path
    self.launch_elementwise("vec_add", a, b, out);
}
```

### §10.2 Memory Pool

Repeated alloc/free is expensive. Both backends use a simple free-list allocator:

```rust
struct GpuMemoryPool<D: GpuDevice> {
    free_list: Mutex<BTreeMap<usize, Vec<D::Buffer>>>,  // size → available buffers
    device: Arc<D>,
}

impl<D: GpuDevice> GpuMemoryPool<D> {
    fn alloc(&self, bytes: usize) -> Result<D::Buffer, GpuError> { ... }
    fn free(&self, buf: D::Buffer) { ... }
}
```

### §10.3 Kernel Fusion Opportunities

Future optimization — not required for Phase 1:

| Fusion | Operators | Benefit |
|---|---|---|
| GEMM+bias+act | `gemm` → `vec_add` → `silu` | 1 kernel instead of 3, no intermediate buffer |
| RMSNorm+scale | `rms_norm` → `vec_scale` | Single pass over data |
| Dequant+GEMV | `dequant_q4k` → `gemv` | No intermediate f32 buffer |

---

## §11 File Inventory

| File | Purpose | Feature gate |
|---|---|---|
| `src/gpu/mod.rs` | `GpuDevice`, `GpuBuffer`, `GpuStream`, `GpuError`, `GpuTensor` | always (trait defs) |
| `src/cuda/mod.rs` | `CudaBackend`, `CudaDeviceWrapper`, re-exports | `cuda` |
| `src/cuda/device.rs` | `GpuDevice` impl for CUDA | `cuda` |
| `src/cuda/kernels.rs` | `Kernels<E>` impl for CUDA | `cuda` |
| `src/cuda/launch.rs` | Grid/block config helpers | `cuda` |
| `src/cuda/shaders/*.cu` | CUDA kernel sources | `cuda` |
| `src/metal_backend/mod.rs` | `MetalBackend`, `MetalDeviceWrapper`, re-exports | `metal` |
| `src/metal_backend/device.rs` | `GpuDevice` impl for Metal | `metal` |
| `src/metal_backend/kernels.rs` | `Kernels<E>` impl for Metal | `metal` |
| `src/metal_backend/pipeline.rs` | PSO cache | `metal` |
| `src/metal_backend/shaders/*.metal` | MSL kernel sources | `metal` |

---

## §12 Migration Checklist

1. [ ] Add `gpu` module with trait definitions (`GpuDevice`, `GpuBuffer`, `GpuStream`, `GpuError`)
2. [ ] Add `cuda` feature + `cudarc` dependency
3. [ ] Implement `CudaDeviceWrapper` + `GpuDevice` impl
4. [ ] Implement P0 CUDA kernels (GEMM via cuBLAS, dequant, quant_matmul)
5. [ ] Implement P1 CUDA kernels (norm, activations, RoPE)
6. [ ] Implement P2/P3 CUDA kernels (BLAS-1, reductions, quantized GEMV)
7. [ ] Add `metal` feature + `metal-rs` dependency
8. [ ] Implement `MetalDeviceWrapper` + `GpuDevice` impl
9. [ ] Port all kernel groups to MSL
10. [ ] Cross-backend test suite
11. [ ] Benchmark suite (GPU vs CPU, kernel launch overhead characterization)
12. [ ] `CpuBackend` impl in `src/backend.rs` (currently missing file)
