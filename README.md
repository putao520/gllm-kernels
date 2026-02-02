# gllm-kernels

**Core Calculation Backend for [gllm](https://github.com/putao520/gllm).**

Provides **L3 GPU-Pure** inference primitives using direct Driver APIs (CUDA/Metal/ROCm) without external runtime dependencies.

[![Crates.io](https://img.shields.io/crates/v/gllm-kernels.svg)](https://crates.io/crates/gllm-kernels)
[![Documentation](https://docs.rs/gllm-kernels/badge.svg)](https://docs.rs/gllm-kernels)
[![License](https://img.shields.io/crates/l/gllm-kernels.svg)](LICENSE)

## Architecture: Driver API & AOT

Unlike traditional bindings that rely on `libcudart` or `nvcc` at runtime, **gllm-kernels** uses a strictly **Ahead-of-Time (AOT)** and **Driver API** approach:

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| **Execution** | **Driver API** (`libcuda.so`, `libhsa.so`) | Zero runtime dependency (no CUDA Toolkit required). |
| **Compilation** | **AOT Only** (CUBIN/HSACO) | 100ms startup time, no JIT compilation overhead. |
| **Memory** | **L3 GPU-Pure** | Inputs & Outputs are GPU-resident (no PCIe traffic during generation). |
| **Dispatch** | **Static Dispatch** | Rust traits determine backend at compile/link time. |

## Supported Backends

The system automatically detects available hardware at runtime:

1.  **CUDA** (NVIDIA):
    -   Target: `libcuda.so` / `nvcuda.dll`
    -   Kernels: Precompiled `.cubin` for `sm_80` (A100), `sm_86` (RTX 30), `sm_89` (RTX 40), `sm_90` (H100).
    -   *Legacy GPUs (Pascal/Volta) are NOT supported.*
2.  **CPU** (Fallback):
    -   Target: Pure Rust (`faer`)
    -   Kernels: AVX-512 / NEON SIMD implementations.
3.  **Metal** (Apple Silicon) - *Planned*
4.  **ROCm** (AMD) - *Planned*

## Installation

```toml
[dependencies]
gllm-kernels = "0.2"
```

## Usage (L3 GPU-Pure API)

This crate provides low-level primitives. Most users should use the high-level `gllm` crate.

```rust
use gllm_kernels::backend::{Backend, CudaBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize Backend (Auto-detects GPU)
    let backend = CudaBackend::new(0)?; // Device 0

    // 2. Upload Weights (Once)
    let weights = backend.upload_generator_weights(&cpu_weights)?;

    // 3. Allocate KV Cache (GPU Resident)
    let mut kv_cache = backend.alloc_kv_cache(128, &config)?;

    // 4. Zero-Copy Forward (Loop)
    // Only Token IDs (4 bytes) transfer over PCIe
    let logits_handle = backend.generator_forward_gpu_pure(
        &[token_id],
        &topology,
        &weights,
        &mut kv_cache,
        &forward_config
    )?;

    // 5. Sample on GPU
    let next_token = backend.sample_from_tensor(&logits_handle, ...)?;

    Ok(())
}
```

## Developer Guide: Recompiling Kernels

Users do **not** need to compile kernels. Only developers modifying `src/cuda_kernels/kernels/kernels.cu`
need to run:

```bash
# Requires CUDA Toolkit (nvcc) or Docker with NVIDIA Container Toolkit
./scripts/compile_kernels.sh cuda
```

This generates `src/cuda_kernels/kernels/kernels_sm80.cubin` (and sm_86/89/90) which are embedded into
the Rust binary.

## License

MIT
