# gllm-kernels

**High-Performance Compute Backend** - The computational engine for `gllm`.

> **🚨 TABULA RASA (2026-02)**: This project has been reset. All legacy code has been removed to enforce strict architectural compliance.

## SPEC Location
- `./SPEC/` (Single Source of Truth)

## Technology Stack (Strict)

| Component | Technology | Constraint |
|-----------|------------|------------|
| **Language** | Rust (2021) | **Pure Rust Only** (No C/C++ build scripts) |
| **GPU API** | CUDA Driver API | Via `cudarc` (No Runtime API `libcudart.so`) |
| **CPU SIMD** | `faer` + `simba` | Runtime detection (AVX2/AVX-512/NEON) |
| **Kernel Dist** | AOT Binary | Embed `.cubin` (sm_80/86/89/90). **No PTX/JIT**. |

## Core Architecture (FROZEN)

### 1. Quantization Kernel Template (ARCH-QUANT-TEMPLATE)
**Unified Template Implementation**:
- **CUDA Kernels**: Use C++ templates `template<int BITS>` for quantized matmul
- **Code Reuse**: Single implementation covers 1/2/4/8-bit quantization
- **Zero Runtime Overhead**: Template instantiation at compile time
- **Rust Dispatch**: Enum matching calls appropriate template instance
- **Violation**: Implementing separate kernels for each bit width is forbidden

### 2. L3 GPU-Pure Architecture (ARCH-GPU-PURE)
**Zero-Copy Generation Loop**:
- **Weights**: Uploaded once to GPU memory.
- **KV Cache**: Permanently resident on GPU.
- **Logits**: Generated and sampled on GPU.
- **Data Transfer**: Only 8 bytes/step (TokenID in -> TokenID out).
- **Violation**: Any `Vec<f32>` transfer during generation loop is a critical bug.

### 3. High-Performance CPU Backend
**NOT just a fallback**:
- Must implement runtime ISA detection (AVX2 vs AVX-512 vs NEON).
- Uses `faer` for BLAS-equivalent performance in pure Rust.
- **Violation**: Linking against `OpenBLAS`, `MKL`, or `Accelerate`.

### 4. Build & Distribution
- **No `build.rs` compilation**: No `cc` crate, no `nvcc` invocation at build time for Rust code.
- **Pre-compiled Kernels**: `.cubin` files are checked into the repo (`src/cuda_kernels/kernels/`).
- **Template Instantiation**: Each sm_XX arch gets templates instantiated for BITS=1,2,4,8.

### 5. Fused-First Architecture / 融合优先原则
- **Constraint**: 调度/执行层必须优先选择融合算子 (Fused Kernels)。仅在无法匹配融合模式时，才降级使用原子算子 (Atomic Kernels)。
- **Constraint**: ONNX Loader 必须实现 Graph Pattern Matching，将子图映射为 Fused Kernels，严禁 naive 的 1:1 翻译。

## Directory Structure

```
src/
├── lib.rs              # Entry point
├── backend_trait.rs    # Backend trait definition (L3 API)
├── cpu_backend.rs      # CPU implementation (SIMD dispatch)
├── cuda_backend.rs     # CUDA implementation (Driver API)
├── ops/                # Reference implementations (Pure Rust)
│   └── mod.rs
├── cpu_kernels/        # Optimized CPU kernels (faer/SIMD)
│   └── mod.rs
└── cuda_kernels/       # CUDA kernel wrappers & CUBIN loader
    ├── mod.rs
    └── kernels/        # *.cubin files (Git tracked, sm_80/86/89/90)
```

## Common Commands

```bash
cargo check
cargo test
# Benchmarks will be added later
```
