//! Generic quantization traits for zero-overhead abstraction.
//!
//! This module defines the core type system for quantized computation,
//! emulating CUDA C++ templates using Rust generics with compile-time
//! monomorphization. This provides zero runtime overhead while supporting
//! multiple data types (F32, F16, BF16, I8, I4, I2, I1).
//!
//! ## Design Principles
//!
//! - **Compile-time monomorphization**: Each type generates specialized code
//! - **Zero runtime dispatch**: No `dyn Trait`, only generic impls
//! - **Unified storage**: All quantized types use `u8` as storage container
//! - **Const generics**: BITS and IS_PACKED are compile-time constants

pub use half::{bf16, f16};

/// Marker types for different data precisions.
///
/// Each type implements `DTypeTrait` with compile-time constants
/// for bit width, packing behavior, and storage type.
pub mod types {
    // F32 marker type
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct F32Type;

    // F16 marker type
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct F16Type;

    // BF16 marker type
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BF16Type;

    // Int8 marker type
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct I8Type;

    // Packed Int4 marker type (2 values per u8)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PackedI4Type;

    // Packed Int2 marker type (4 values per u8)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PackedI2Type;

    // Packed Int1 marker type (8 values per u8)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PackedI1Type;
}

pub use types::*;

/// Core trait for data types that can be used in quantized computation.
///
/// This trait provides compile-time type information and a dequantization
/// method. All const values are optimized away during compilation.
///
/// # Type Parameters
///
/// - `Storage`: The underlying storage type (f32, f16, bf16, i8, u8)
///
/// # Const Parameters
///
/// - `BITS`: Number of bits per value (32, 16, 8, 4, 2, 1)
/// - `IS_PACKED`: Whether multiple values are packed into a single storage unit
pub trait DTypeTrait: Sized + Copy + 'static {
    /// The underlying storage type.
    type Storage: Copy;

    /// Dequantize a single value from storage to f32.
    ///
    /// For non-packed types (F32, F16, BF16, I8), this is a direct conversion.
    /// For packed types (I4, I2, I1), this unpacks the value first.
    fn dequantize(scaled: Self::Storage, scale: f16) -> f32;

    /// Number of bits per value (compile-time constant).
    const BITS: u8;

    /// Whether multiple values are packed into one storage unit.
    const IS_PACKED: bool;

    /// Number of values per storage unit (derived from BITS).
    #[inline(always)]
    fn values_per_byte() -> usize {
        if Self::IS_PACKED {
            8 / Self::BITS as usize
        } else {
            1
        }
    }
}

// Implement DTypeTrait for all marker types

impl DTypeTrait for F32Type {
    type Storage = f32;

    #[inline(always)]
    fn dequantize(scaled: f32, _scale: f16) -> f32 {
        scaled
    }

    const BITS: u8 = 32;
    const IS_PACKED: bool = false;
}

impl DTypeTrait for F16Type {
    type Storage = f16;

    #[inline(always)]
    fn dequantize(scaled: f16, _scale: f16) -> f32 {
        scaled.to_f32()
    }

    const BITS: u8 = 16;
    const IS_PACKED: bool = false;
}

impl DTypeTrait for BF16Type {
    type Storage = bf16;

    #[inline(always)]
    fn dequantize(scaled: bf16, _scale: f16) -> f32 {
        scaled.to_f32()
    }

    const BITS: u8 = 16;
    const IS_PACKED: bool = false;
}

impl DTypeTrait for I8Type {
    type Storage = i8;

    #[inline(always)]
    fn dequantize(scaled: i8, scale: f16) -> f32 {
        scaled as f32 * scale.to_f32()
    }

    const BITS: u8 = 8;
    const IS_PACKED: bool = false;
}

impl DTypeTrait for PackedI4Type {
    type Storage = u8;

    #[inline(always)]
    fn dequantize(packed: u8, scale: f16) -> f32 {
        // Unpack 4-bit signed integer
        let value = if packed & 0x08 != 0 {
            (packed as i8) - 16
        } else {
            packed as i8
        };
        value as f32 * scale.to_f32()
    }

    const BITS: u8 = 4;
    const IS_PACKED: bool = true;
}

impl DTypeTrait for PackedI2Type {
    type Storage = u8;

    #[inline(always)]
    fn dequantize(packed: u8, scale: f16) -> f32 {
        // Unpack 2-bit signed integer (offset binary)
        let value = (packed & 0x03) as i8;
        let value = if value & 0x02 != 0 {
            value - 4
        } else {
            value
        };
        value as f32 * scale.to_f32()
    }

    const BITS: u8 = 2;
    const IS_PACKED: bool = true;
}

impl DTypeTrait for PackedI1Type {
    type Storage = u8;

    #[inline(always)]
    fn dequantize(packed: u8, scale: f16) -> f32 {
        // Unpack 1-bit signed integer (0 -> -1, 1 -> 1)
        let value = if packed & 0x01 != 0 { 1.0 } else { -1.0 };
        value * scale.to_f32()
    }

    const BITS: u8 = 1;
    const IS_PACKED: bool = true;
}

/// Weight format for QKV projection.
///
/// This enum allows the QKV projection function to automatically
/// select the optimal computation path based on the available weights.
///
/// # Variants
///
/// - `Separated`: Three independent weight matrices (optimal: 3× small matmul)
/// - `Fused`: Single combined weight matrix (fallback: 1× large matmul)
pub enum QkvWeightFormat<'a, T: DTypeTrait> {
    Separated {
        q_weight: &'a [T::Storage],
        k_weight: &'a [T::Storage],
        v_weight: &'a [T::Storage],
        q_scales: Option<&'a [f16]>,
        k_scales: Option<&'a [f16]>,
        v_scales: Option<&'a [f16]>,
    },
    Fused {
        qkv_weight: &'a [T::Storage],
        scales: Option<&'a [f16]>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_trait_bits() {
        assert_eq!(F32Type::BITS, 32);
        assert_eq!(F16Type::BITS, 16);
        assert_eq!(BF16Type::BITS, 16);
        assert_eq!(I8Type::BITS, 8);
        assert_eq!(PackedI4Type::BITS, 4);
        assert_eq!(PackedI2Type::BITS, 2);
        assert_eq!(PackedI1Type::BITS, 1);
    }

    #[test]
    fn test_dtype_trait_is_packed() {
        assert!(!F32Type::IS_PACKED);
        assert!(!F16Type::IS_PACKED);
        assert!(!I8Type::IS_PACKED);
        assert!(PackedI4Type::IS_PACKED);
        assert!(PackedI2Type::IS_PACKED);
        assert!(PackedI1Type::IS_PACKED);
    }

    #[test]
    fn test_dtype_trait_values_per_byte() {
        assert_eq!(F32Type::values_per_byte(), 1);
        assert_eq!(I8Type::values_per_byte(), 1);
        assert_eq!(PackedI4Type::values_per_byte(), 2);
        assert_eq!(PackedI2Type::values_per_byte(), 4);
        assert_eq!(PackedI1Type::values_per_byte(), 8);
    }

    #[test]
    fn test_dequantize_f32() {
        let val: <F32Type as DTypeTrait>::Storage = 3.14f32;
        assert_eq!(F32Type::dequantize(val, half::f16::from_f32(2.0)), 3.14);
    }

    #[test]
    fn test_dequantize_i8() {
        let val: <I8Type as DTypeTrait>::Storage = 42i8;
        let scale = half::f16::from_f32(0.1);
        let result = I8Type::dequantize(val, scale);
        let expected = 42i8 as f32 * scale.to_f32();
        assert!((result - expected).abs() < 0.001, "dequantize mismatch: {} vs {}", result, expected);
    }

    #[test]
    fn test_dequantize_packed_i4() {
        // Positive value (0x00 = 0)
        assert_eq!(PackedI4Type::dequantize(0x00, half::f16::from_f32(1.0)), 0.0);
        // Positive max (0x07 = 7)
        assert_eq!(PackedI4Type::dequantize(0x07, half::f16::from_f32(1.0)), 7.0);
        // Negative value (0x08 = -8)
        assert_eq!(PackedI4Type::dequantize(0x08, half::f16::from_f32(1.0)), -8.0);
        // Negative max (0x0F = -1)
        assert_eq!(PackedI4Type::dequantize(0x0F, half::f16::from_f32(1.0)), -1.0);
    }

    #[test]
    fn test_dequantize_packed_i2() {
        // -2 (0x10 = 0b10)
        assert_eq!(PackedI2Type::dequantize(0x02, half::f16::from_f32(1.0)), -2.0);
        // -1 (0x11 = 0b11)
        assert_eq!(PackedI2Type::dequantize(0x03, half::f16::from_f32(1.0)), -1.0);
        // 0 (0x00 = 0b00)
        assert_eq!(PackedI2Type::dequantize(0x00, half::f16::from_f32(1.0)), 0.0);
        // 1 (0x01 = 0b01)
        assert_eq!(PackedI2Type::dequantize(0x01, half::f16::from_f32(1.0)), 1.0);
    }

    #[test]
    fn test_dequantize_packed_i1() {
        // -1 (bit 0 = 0)
        assert_eq!(PackedI1Type::dequantize(0x00, half::f16::from_f32(1.0)), -1.0);
        // 1 (bit 0 = 1)
        assert_eq!(PackedI1Type::dequantize(0x01, half::f16::from_f32(1.0)), 1.0);
    }
}
