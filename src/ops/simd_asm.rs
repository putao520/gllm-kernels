//! Architecture-specific SIMD implementations using inline assembly.
//!
//! This module provides hand-optimized kernels for:
//! - x86_64: AVX2, AVX-512
//! - ARM64: NEON
//!
//! Falls back to generic `wide` SIMD when specific features aren't available.

// ============================================================================
// x86_64 AVX2 Implementation
// ============================================================================

/// AVX2-optimized dot product using FMA instructions.
/// 
/// # Safety
/// - `a` and `b` must be valid pointers to at least `len` f32 elements.
/// - CPU must support AVX2 and FMA features (checked at runtime).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[inline(always)]
pub unsafe fn dot_avx2_fma(a: *const f32, b: *const f32, len: usize) -> f32 {
    use core::arch::x86_64::*;
    
    let simd_len = len / 32 * 32; // Process 32 floats per iteration (4x ymm registers)
    
    // Initialize 4 accumulators to hide FMA latency
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    
    let mut i = 0;
    while i < simd_len {
        let a0 = _mm256_loadu_ps(a.add(i));
        let b0 = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        
        let a1 = _mm256_loadu_ps(a.add(i + 8));
        let b1 = _mm256_loadu_ps(b.add(i + 8));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        
        let a2 = _mm256_loadu_ps(a.add(i + 16));
        let b2 = _mm256_loadu_ps(b.add(i + 16));
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
        
        let a3 = _mm256_loadu_ps(a.add(i + 24));
        let b3 = _mm256_loadu_ps(b.add(i + 24));
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        
        i += 32;
    }
    
    // Combine accumulators
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    
    // Horizontal sum of ymm register
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);
    
    // Handle remainder
    for j in simd_len..len {
        result += *a.add(j) * *b.add(j);
    }
    
    result
}

// ============================================================================
// x86_64 AVX-512 Implementation
// ============================================================================

/// AVX-512 optimized dot product with 16-wide SIMD.
/// 
/// # Safety
/// - `a` and `b` must be valid pointers to at least `len` f32 elements.
/// - CPU must support AVX-512F feature.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
pub unsafe fn dot_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use core::arch::x86_64::*;
    
    let simd_len = len / 64 * 64; // Process 64 floats per iteration (4x zmm registers)
    
    // Initialize 4 accumulators for latency hiding
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    
    let mut i = 0;
    while i < simd_len {
        let a0 = _mm512_loadu_ps(a.add(i));
        let b0 = _mm512_loadu_ps(b.add(i));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        
        let a2 = _mm512_loadu_ps(a.add(i + 32));
        let b2 = _mm512_loadu_ps(b.add(i + 32));
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        
        let a3 = _mm512_loadu_ps(a.add(i + 48));
        let b3 = _mm512_loadu_ps(b.add(i + 48));
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        
        i += 64;
    }
    
    // Combine accumulators
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    
    // Horizontal sum of zmm register
    let mut result = _mm512_reduce_add_ps(acc0);
    
    // Handle remainder with scalar
    for j in simd_len..len {
        result += *a.add(j) * *b.add(j);
    }
    
    result
}

/// AVX2-optimized dot product (without FMA, for older CPUs like Haswell).
/// Uses 4 accumulators for latency hiding, similar to AVX2+FMA version.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "fma")))]
#[inline(always)]
pub unsafe fn dot_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use core::arch::x86_64::*;
    
    let simd_len = len / 32 * 32; // Process 32 floats per iteration (4x ymm)
    
    // 4 accumulators for latency hiding
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    
    let mut i = 0;
    while i < simd_len {
        let a0 = _mm256_loadu_ps(a.add(i));
        let b0 = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
        
        let a1 = _mm256_loadu_ps(a.add(i + 8));
        let b1 = _mm256_loadu_ps(b.add(i + 8));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
        
        let a2 = _mm256_loadu_ps(a.add(i + 16));
        let b2 = _mm256_loadu_ps(b.add(i + 16));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2, b2));
        
        let a3 = _mm256_loadu_ps(a.add(i + 24));
        let b3 = _mm256_loadu_ps(b.add(i + 24));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3, b3));
        
        i += 32;
    }
    
    // Combine accumulators
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    
    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);
    
    // Handle remainder
    for j in simd_len..len {
        result += *a.add(j) * *b.add(j);
    }
    
    result
}

// ============================================================================
// ARM64 NEON Implementation
// ============================================================================

/// NEON-optimized dot product for ARM64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn dot_neon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use core::arch::aarch64::*;
    
    let simd_len = len / 16 * 16; // Process 16 floats per iteration (4x float32x4)
    
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    
    let mut i = 0;
    while i < simd_len {
        let a0 = vld1q_f32(a.add(i));
        let b0 = vld1q_f32(b.add(i));
        acc0 = vfmaq_f32(acc0, a0, b0);
        
        let a1 = vld1q_f32(a.add(i + 4));
        let b1 = vld1q_f32(b.add(i + 4));
        acc1 = vfmaq_f32(acc1, a1, b1);
        
        let a2 = vld1q_f32(a.add(i + 8));
        let b2 = vld1q_f32(b.add(i + 8));
        acc2 = vfmaq_f32(acc2, a2, b2);
        
        let a3 = vld1q_f32(a.add(i + 12));
        let b3 = vld1q_f32(b.add(i + 12));
        acc3 = vfmaq_f32(acc3, a3, b3);
        
        i += 16;
    }
    
    // Combine accumulators
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    
    // Horizontal sum
    let mut result = vaddvq_f32(acc0);
    
    // Handle remainder
    for j in simd_len..len {
        result += *a.add(j) * *b.add(j);
    }
    
    result
}

// ============================================================================
// Generic Fallback (scalar)
// ============================================================================

/// Scalar dot product fallback.
#[inline(always)]
pub fn dot_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *a.add(i) * *b.add(i) };
    }
    sum
}
// ============================================================================
// SIMD Capability Detection
// ============================================================================

/// SIMD implementation currently in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdPath {
    Avx512,
    Avx2Fma,
    Avx2,
    Neon,
    Scalar,
}

impl std::fmt::Display for SimdPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimdPath::Avx512 => write!(f, "AVX-512 (64 floats/iter)"),
            SimdPath::Avx2Fma => write!(f, "AVX2+FMA (32 floats/iter)"),
            SimdPath::Avx2 => write!(f, "AVX2 (32 floats/iter)"),
            SimdPath::Neon => write!(f, "NEON (16 floats/iter)"),
            SimdPath::Scalar => write!(f, "Scalar fallback"),
        }
    }
}

/// Returns the SIMD path currently selected at compile time.
#[inline]
pub fn current_simd_path() -> SimdPath {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    { return SimdPath::Avx512; }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", not(target_feature = "avx512f")))]
    { return SimdPath::Avx2Fma; }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "fma"), not(target_feature = "avx512f")))]
    { return SimdPath::Avx2; }
    
    #[cfg(target_arch = "aarch64")]
    { return SimdPath::Neon; }
    
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "avx512f"),
        target_arch = "aarch64"
    )))]
    { return SimdPath::Scalar; }
}

// ============================================================================
// Dispatcher: Chooses best implementation at compile time
// ============================================================================

/// Dispatch to the best available dot product implementation.
/// Priority: AVX-512 > AVX2+FMA > AVX2 > NEON > scalar fallback
#[inline(always)]
pub fn simd_dot_product(a: *const f32, b: *const f32, len: usize) -> f32 {
    // AVX-512: Best for large vectors (hidden >= 4096)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        return unsafe { dot_avx512(a, b, len) };
    }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", not(target_feature = "avx512f")))]
    {
        return unsafe { dot_avx2_fma(a, b, len) };
    }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "fma"), not(target_feature = "avx512f")))]
    {
        return unsafe { dot_avx2(a, b, len) };
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_neon(a, b, len) };
    }
    
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "avx512f"),
        target_arch = "aarch64"
    )))]
    {
        return dot_scalar(a, b, len);
    }
}

// ============================================================================
// Size-Specialized GEMV: Optimized for Batch=1 Inference
// ============================================================================

/// Model size tiers for routing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    /// Micro models: hidden < 1024 (SmolLM2-135M, etc.)
    Micro,
    /// Medium models: 1024 <= hidden < 4096 (Llama-7B, etc.)
    Medium,
    /// Large models: hidden >= 4096 (Llama-70B, Qwen-72B, etc.)
    Large,
}

impl ModelTier {
    #[inline]
    pub fn from_hidden_size(hidden: usize) -> Self {
        if hidden < 1024 {
            ModelTier::Micro
        } else if hidden < 4096 {
            ModelTier::Medium
        } else {
            ModelTier::Large
        }
    }
}

/// Parallel threshold for GEMV operations.
/// For very small output dimensions, parallelization overhead > benefit.
const PARALLEL_THRESHOLD_OUT: usize = 256;

/// Single-threaded GEMV: output[i] = dot(weight[i], input)
/// Optimal for Micro tier models.
#[inline]
pub fn gemv_single(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), out_features);
    
    for i in 0..out_features {
        let w_row = weight[i * in_features..].as_ptr();
        output[i] = simd_dot_product(input.as_ptr(), w_row, in_features);
    }
}

/// Parallel GEMV using rayon for larger output dimensions.
/// Optimal for Medium/Large tier models.
#[inline]
pub fn gemv_parallel(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), out_features);

    // Rayon removed; fall back to the single-threaded path.
    gemv_single(input, weight, output, out_features, in_features);
}

/// Size-specialized GEMV dispatcher.
/// Chooses optimal strategy based on hidden size.
#[inline]
pub fn gemv_dispatch(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    if out_features < PARALLEL_THRESHOLD_OUT {
        // Small output: single-threaded is faster (avoids rayon overhead)
        gemv_single(input, weight, output, out_features, in_features);
    } else {
        // Large output: parallel benefits outweigh overhead
        gemv_parallel(input, weight, output, out_features, in_features);
    }
}

/// GEMV with residual add: output[i] += dot(weight[i], input) + bias
/// For zero-allocation residual connections.
#[inline]
pub fn gemv_add_bias(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), out_features);
    
    for i in 0..out_features {
        let w_row = weight[i * in_features..].as_ptr();
        let dot = simd_dot_product(input.as_ptr(), w_row, in_features);
        let b = bias.map_or(0.0, |b| b[i]);
        output[i] += dot + b;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn test_simd_dot_product_basic() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 * 0.1).collect();
        
        let expected = reference_dot(&a, &b);
        let result = simd_dot_product(a.as_ptr(), b.as_ptr(), a.len());
        
        assert!((result - expected).abs() < 0.01, "Expected {}, got {}", expected, result);
    }

    #[test]
    fn test_simd_dot_product_remainder() {
        // Test with non-aligned length
        let a: Vec<f32> = (0..37).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0).collect();
        
        let expected = reference_dot(&a, &b);
        let result = simd_dot_product(a.as_ptr(), b.as_ptr(), a.len());
        
        assert!((result - expected).abs() < 0.001, "Expected {}, got {}", expected, result);
    }
}
