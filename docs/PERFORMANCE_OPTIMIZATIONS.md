# GLLM-Kernels Performance Optimizations

## Overview

This document describes the SIMD and algorithmic optimizations implemented in `gllm-kernels` to improve CPU inference performance.

## Dependencies Added

```toml
matrixmultiply = "0.3"  # High-performance pure Rust GEMM (AVX2/AVX-512)
wide = "0.7"            # Cross-platform SIMD (AVX2/SSE/NEON)
```

## Optimized Components

### 1. Linear Layer (GEMM) - `ops/linear.rs`

**Before**: Naive parallel dot product with `rayon`
**After**: High-performance GEMM using `matrixmultiply::sgemm`

- Uses optimized cache-blocking and SIMD (AVX2/AVX-512) internally
- ~2x speedup for matrix multiplications
- Pure Rust, no external C/Fortran dependencies

### 2. RMS Normalization - `ops/rms_norm.rs`

**Before**: Scalar loop with Kahan summation
**After**: SIMD-optimized using `wide::f32x8`

Key functions:
- `simd_sum_squares()`: Vectorized sum of squares
- `simd_scale_multiply()`: Vectorized normalize + scale

Benefits:
- Processes 8 elements per iteration
- ~3x speedup for hidden dimensions ≥ 64

### 3. Activation Functions - `ops/activations.rs`

**SiLU (Swish)** - Most important for LLaMA/Mistral

**Before**: `x * sigmoid(x)` with scalar operations
**After**: SIMD with fast exp approximation

Key functions:
- `simd_sigmoid()`: Vectorized sigmoid using fast exp
- `simd_fast_exp()`: Polynomial exp approximation

Optimized functions:
- `silu_inplace()` 
- `silu()`
- `silu_mul_inplace()` - Fused SiLU + gate multiplication

### 4. Element-wise Operations - `ops/activations.rs`

**Before**: Scalar zip iteration
**After**: SIMD using `wide::f32x8`

Optimized functions:
- `mul_inplace()` - Residual connections
- `add_inplace()` - Residual additions

## Quantization Support Added

### Q3_K_M Dequantization - `quantized.rs`

Added support for Q3_K_S/Q3_K_M/Q3_K_L 3-bit K-quantization:
- 256 elements per block
- 110 bytes per block layout: `hmask[32] + qs[64] + scales[12] + d[2]`
- Each element is 3 bits: 2 bits from qs + 1 bit from hmask

## Performance Results

Benchmarked on SmolLM2-135M-Instruct model:

| Optimization Stage | Speed (tokens/s) | Improvement |
|-------------------|------------------|-------------|
| Baseline (naive)  | 2.08             | -           |
| + matrixmultiply  | 3.89             | +87%        |
| + SIMD ops        | 4.20             | +102%       |

## Future Optimization Opportunities

1. **Flash Attention CPU Kernel** - Currently uses naive attention
2. **RoPE SIMD** - Position embeddings can be vectorized
3. **Layer Norm SIMD** - Similar to RMS Norm optimization
4. **Quantized Matmul SIMD** - Direct Q4/Q8 computation without dequantization

## Build Recommendations

For maximum performance, compile with native CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This enables AVX2/AVX-512 instructions on supported CPUs.
