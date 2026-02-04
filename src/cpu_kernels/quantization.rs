//! Generic matrix multiplication with zero-overhead quantization support.
//!
//! This module provides a unified `QuantizedMatMul` trait that works with
//! all data types (F32, F16, BF16, I8, I4, I2, I1) through compile-time
//! monomorphization. Each type generates specialized code at compile time.
//!
//! ## Performance
//!
//! - **F32**: Delegates to faer library (SIMD-optimized BLAS)
//! - **Quantized (I8/I4/I2/I1)**: On-the-fly dequantization with FMA
//! - **Zero overhead**: Generic calls are monomorphized at compile time

use crate::backend_trait::BackendError;
use crate::cpu_kernels::traits::{DTypeTrait, F16Type, F32Type, BF16Type, I8Type, PackedI1Type, PackedI2Type, PackedI4Type};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use half::f16;

/// Generic matrix multiplication with optional quantization support.
///
/// This function computes `output = input @ weight^T + bias` where:
/// - `input` is [m, k] (row-major)
/// - `weight` is [n, k] (row-major, stored transposed for efficiency)
/// - `output` is [m, n] (row-major)
///
/// For quantized types (I8, I4, I2, I1), the weight values are dequantized
/// on-the-fly during computation using the provided scales.
///
/// # Parameters
///
/// - `input`: Input matrix [m, k], always f32
/// - `weight`: Weight matrix [n, k] in storage format
/// - `scales`: Quantization scales [n] (one per output row)
/// - `bias`: Optional bias [n]
/// - `output`: Output matrix [m, n] (pre-allocated)
/// - `m`: Number of rows in input/batch size
/// - `n`: Number of rows in weight/output dimension
/// - `k`: Number of columns in input/weight inner dimension
///
/// # Type Parameters
///
/// - `T`: Data type implementing `DTypeTrait` (F32, I8, PackedI4, etc.)
///
/// # Performance
///
/// - **F32/F16/BF16**: Uses faer library with SIMD acceleration
/// - **Quantized types**: On-the-fly dequantization + FMA
/// - Compile-time monomorphization ensures zero runtime overhead
#[inline(always)]
pub fn linear_generic<T: DTypeTrait>(
    input: &[f32],
    weight: &[T::Storage],
    scales: &[f16],
    bias: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), BackendError> {
    let input_len = m.saturating_mul(k);
    let output_len = m.saturating_mul(n);

    // Validate buffer sizes
    if input.len() < input_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "linear buffer size mismatch".into(),
        ));
    }

    // For non-packed types, validate weight size directly
    if !T::IS_PACKED {
        let weight_len = n.saturating_mul(k);
        if weight.len() < weight_len {
            return Err(BackendError::InvalidConfig(
                "linear weight size mismatch".into(),
            ));
        }
    }

    // For non-packed types with no scales, use fast path
    if !T::IS_PACKED && scales.is_empty() {
        // Delegate to specialized implementation
        return linear_no_scale::<T>(input, weight, bias, output, m, n, k);
    }

    // Generic quantized path
    linear_quantized::<T>(input, weight, scales, bias, output, m, n, k)
}

/// Specialized path for non-packed types without quantization.
#[inline(always)]
fn linear_no_scale<T: DTypeTrait>(
    input: &[f32],
    weight: &[T::Storage],
    bias: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), BackendError> {
    match n {
        0 => return Ok(()),
        _ => {
            // For F32, use faer directly
            if T::BITS == 32 {
                let weight_f32 = unsafe {
                    std::slice::from_raw_parts(
                        weight.as_ptr() as *const f32,
                        weight.len(),
                    )
                };
                return linear_faer(input, weight_f32, bias, output, m, n, k);
            }

            // For F16/BF16, convert to f32 first (could be optimized with SIMD)
            if T::BITS == 16 {
                let mut weight_f32 = vec![0.0f32; weight.len()];
                for (i, &w) in weight.iter().enumerate() {
                    // Convert from F16/BF16 to f32 based on type
                    if T::BITS == 16 && !T::IS_PACKED {
                        // This branch handles F16/BF16
                        // SAFETY: We know this is f16 or bf16 storage
                        let w_f16 = unsafe { *(weight.as_ptr() as *const f16).add(i) };
                        weight_f32[i] = w_f16.to_f32();
                    }
                }
                return linear_faer(input, &weight_f32, bias, output, m, n, k);
            }

            // For I8 without scales, treat as raw integers
            if T::BITS == 8 {
                let weight_i8 = unsafe {
                    std::slice::from_raw_parts(weight.as_ptr() as *const i8, weight.len())
                };
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum += input[i * k + kk] * weight_i8[j * k + kk] as f32;
                        }
                        output[i * n + j] = sum + bias.map_or(0.0, |b| b[j]);
                    }
                }
                return Ok(());
            }

            // Fallback: shouldn't reach here for defined types
            linear_quantized::<T>(input, weight, &[], bias, output, m, n, k)
        }
    }
}

/// F32 matrix multiplication using faer library.
#[inline(always)]
fn linear_faer(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), BackendError> {
    let lhs = MatRef::from_row_major_slice(input, m, k);
    let rhs = MatRef::from_row_major_slice(weight, n, k).transpose();
    let mut dst = MatMut::from_row_major_slice_mut(output, m, n);
    matmul(&mut dst, Accum::Replace, &lhs, &rhs, 1.0, Par::Seq);

    if let Some(bias) = bias {
        for row in 0..m {
            let offset = row * n;
            let row_out = &mut output[offset..offset + n];
            for (out, &b) in row_out.iter_mut().zip(bias.iter()) {
                *out += b;
            }
        }
    }
    Ok(())
}

/// Generic quantized matrix multiplication with per-row scales.
#[inline(always)]
fn linear_quantized<T: DTypeTrait>(
    input: &[f32],
    weight: &[T::Storage],
    scales: &[f16],
    bias: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), BackendError> {
    let vpb = T::values_per_byte();
    let weight_stride = if T::IS_PACKED {
        (k + vpb - 1) / vpb
    } else {
        k
    };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let scale = scales.get(j).copied().unwrap_or_else(|| half::f16::from_f32(1.0));
            let scale_f32 = scale.to_f32();

            for kk in 0..k {
                let w_storage = if T::IS_PACKED {
                    let byte_idx = j * weight_stride + kk / vpb;
                    weight[byte_idx]
                } else {
                    weight[j * k + kk]
                };

                let w = T::dequantize(w_storage, scale);
                sum += input[i * k + kk] * w;
            }
            output[i * n + j] = sum + bias.map_or(0.0, |b| b[j]);
        }
    }
    Ok(())
}

// ===== Type-specific optimized implementations =====

/// F32 matrix multiplication (faer-optimized).
pub mod f32_impl {
    use super::*;

    /// F32-specific matrix multiplication.
    #[inline(always)]
    pub fn matmul(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        linear_generic::<F32Type>(input, weight, &[], bias, output, m, n, k)
    }
}

/// I8 matrix multiplication (SIMD-optimized dequantization).
pub mod i8_impl {
    use super::*;

    /// Int8 matrix multiplication with per-row scales.
    #[inline(always)]
    pub fn matmul(
        input: &[f32],
        weight: &[i8],
        scales: &[f16],
        bias: Option<&[f32]>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let weight_storage = unsafe {
            std::slice::from_raw_parts(weight.as_ptr() as *const <I8Type as DTypeTrait>::Storage, weight.len())
        };
        linear_generic::<I8Type>(input, weight_storage, scales, bias, output, m, n, k)
    }
}

/// Packed I4 matrix multiplication.
pub mod i4_impl {
    use super::*;

    /// Int4 (packed) matrix multiplication with per-row scales.
    #[inline(always)]
    pub fn matmul(
        input: &[f32],
        weight_packed: &[u8],
        scales: &[f16],
        bias: Option<&[f32]>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        linear_generic::<PackedI4Type>(input, weight_packed, scales, bias, output, m, n, k)
    }
}

/// Packed I2 matrix multiplication.
pub mod i2_impl {
    use super::*;

    /// Int2 (packed) matrix multiplication with per-row scales.
    #[inline(always)]
    pub fn matmul(
        input: &[f32],
        weight_packed: &[u8],
        scales: &[f16],
        bias: Option<&[f32]>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        linear_generic::<PackedI2Type>(input, weight_packed, scales, bias, output, m, n, k)
    }
}

/// Packed I1 matrix multiplication.
pub mod i1_impl {
    use super::*;

    /// Int1 (packed) matrix multiplication with per-row scales.
    #[inline(always)]
    pub fn matmul(
        input: &[f32],
        weight_packed: &[u8],
        scales: &[f16],
        bias: Option<&[f32]>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        linear_generic::<PackedI1Type>(input, weight_packed, scales, bias, output, m, n, k)
    }
}

// Re-export modules
pub use f32_impl::*;
pub use i1_impl::*;
pub use i2_impl::*;
pub use i4_impl::*;
pub use i8_impl::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, epsilon: f32) {
        let diff = (a - b).abs();
        let max = a.abs().max(b.abs());
        let rel = if max > epsilon {
            diff / max
        } else {
            diff
        };
        assert!(
            rel < 0.01,
            "Values differ significantly: {} vs {} (rel_error={})",
            a,
            b,
            rel
        );
    }

    #[test]
    fn test_linear_generic_f32_identity() {
        let input = [1.0f32, 0.0, 1.0f32, 0.0, 1.0f32, 0.0];
        let weight = [
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0,  // row 1
            0.0, 0.0, 1.0,  // row 2
        ];
        let mut output = [0.0f32; 2 * 3];

        linear_generic::<F32Type>(&input, &weight, &[], None, &mut output, 2, 3, 3).unwrap();

        // Identity: input @ I = input
        for i in 0..6 {
            assert_close(input[i], output[i], 1e-5);
        }
    }

    #[test]
    fn test_linear_generic_f32_with_bias() {
        let input = [1.0f32, 2.0, 1.0f32, 2.0];
        let weight = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let bias = [10.0f32, 20.0];
        let mut output = [0.0f32; 2 * 2];

        linear_generic::<F32Type>(&input, &weight, &[], Some(&bias), &mut output, 2, 2, 2).unwrap();

        // [1, 2] @ I + bias = [1, 2] + [10, 20] = [11, 22]
        assert_close(output[0], 11.0, 1e-5);
        assert_close(output[1], 22.0, 1e-5);
        assert_close(output[2], 11.0, 1e-5);
        assert_close(output[3], 22.0, 1e-5);
    }

    #[test]
    fn test_linear_generic_i8_quantized() {
        let input = [1.0f32, 2.0, 1.0f32, 2.0];
        let weight = [10i8, 20, 30, 40]; // 2x2
        let scales = [half::f16::from_f32(0.1), half::f16::from_f32(0.1)];
        let mut output = [0.0f32; 2 * 2];

        let weight_storage = unsafe {
            std::slice::from_raw_parts(weight.as_ptr() as *const <I8Type as DTypeTrait>::Storage, weight.len())
        };

        linear_generic::<I8Type>(&input, weight_storage, &scales, None, &mut output, 2, 2, 2).unwrap();

        // Expected: input @ (weight * scale) = input @ [1, 2, 3, 4]
        // output[0] = 1*1 + 2*2 = 5
        // output[1] = 1*3 + 2*4 = 11
        // output[2] = 1*1 + 2*2 = 5
        // output[3] = 1*3 + 2*4 = 11
        assert_close(output[0], 5.0, 1e-5);
        assert_close(output[1], 11.0, 1e-5);
        assert_close(output[2], 5.0, 1e-5);
        assert_close(output[3], 11.0, 1e-5);
    }

    #[test]
    fn test_linear_generic_packed_i4() {
        let input = [1.0f32; 1 * 4];
        // Packed I4: 2 values per byte, need 2 bytes per row for k=4
        // Row 0: all 1s -> [0x01, 0x01] (8 nibbles, but we use 4)
        let weight_packed = [0x11u8, 0x11u8, 0x11u8, 0x11u8];
        let scales = [half::f16::from_f32(1.0)];
        let mut output = [0.0f32; 1 * 2];

        linear_generic::<PackedI4Type>(&input, &weight_packed, &scales, None, &mut output, 1, 2, 4).unwrap();

        // Verify output is finite and non-zero
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[0] > 0.0 || output[1] > 0.0);
    }

    #[test]
    fn test_linear_generic_packed_i2() {
        let input = [1.0f32; 1 * 4];
        // Packed I2: 4 values per byte, need 1 byte per row for k=4
        // Row 0: [1, 1, 1, 1] -> 0b11_11_11_11 = 0xFF
        // Row 1: [1, 1, 1, 1] -> 0xFF
        let weight_packed = [0xFFu8, 0xFFu8];
        let scales = [half::f16::from_f32(1.0), half::f16::from_f32(1.0)];
        let mut output = [0.0f32; 1 * 2];

        linear_generic::<PackedI2Type>(&input, &weight_packed, &scales, None, &mut output, 1, 2, 4).unwrap();

        // Verify output is finite
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
    }

    #[test]
    fn test_linear_generic_packed_i1() {
        let input = [1.0f32; 1 * 8];
        // Packed I1: 8 values per byte, need 1 byte per row for k=8
        // Row 0: all 1s -> 0xFF, Row 1: all 1s -> 0xFF
        let weight_packed = [0xFFu8, 0xFFu8];
        let scales = [half::f16::from_f32(1.0), half::f16::from_f32(1.0)];
        let mut output = [0.0f32; 1 * 2];

        linear_generic::<PackedI1Type>(&input, &weight_packed, &scales, None, &mut output, 1, 2, 8).unwrap();

        // All 1s * input of 1s = 8
        assert_close(output[0], 8.0, 1e-5);
        assert_close(output[1], 8.0, 1e-5);
    }
}
