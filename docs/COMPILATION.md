# gllm-kernels Compilation Guide

## Overview

**gllm-kernels** uses an **Ahead-of-Time (AOT)** compilation strategy. Instead of distributing PTX (which requires JIT at runtime), we distribute pre-compiled machine code (`.cubin` for NVIDIA, `.hsaco` for AMD).

## Requirements

### 1. CUDA (NVIDIA)
- **Tool**: `nvcc` (CUDA Toolkit 12.x)
- **Target**: `.cubin` (CUBIN)
- **Supported Architectures**:
  - `sm_80` (A100)
  - `sm_86` (RTX 3080/3090)
  - `sm_89` (RTX 4090, L40)
  - `sm_90` (H100)

### 2. ROCm (AMD)
- **Tool**: `hipcc` (ROCm 6.x)
- **Target**: `.hsaco` (Code Object)

## Compilation Instructions

The project provides a helper script to compile kernels for all supported architectures.

```bash
# Compile all kernels (requires Docker or local nvcc/hipcc)
./scripts/compile_kernels.sh
```

### Manual Compilation (CUDA Example)

If you need to compile manually:

```bash
cd src/cuda_kernels/kernels

# Compile for RTX 4090 (sm_89)
nvcc -cubin \
     -arch=sm_89 \
     --use_fast_math \
     -O3 \
     tiled_attention.cu \
     -o tiled_attention_sm89.cubin
```

## Output Location

Compiled binaries are stored in the source tree so they can be embedded into the Rust library:

```
src/
└── cuda_kernels/
    └── kernels/
        ├── flash_attn_sm80.cubin
        ├── flash_attn_sm86.cubin
        ├── flash_attn_sm89.cubin
        └── flash_attn_sm90.cubin
```

## CI/CD Pipeline

The GitHub Workflow `.github/workflows/compile-kernels.yml` automatically:
1. Runs compilation for all architectures.
2. Checks that the generated `.cubin` files match the source code.
3. Uploads artifacts for release.
