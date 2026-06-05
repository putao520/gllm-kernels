//! Symbolic Quantization IR — Unified JIT compilation for 28 quantization formats.
//!
//! REQ-JIT-QUANT-001: All quantization formats share a unified IR representation.
//! JIT codegen generates optimized kernels based on IR, avoiding per-format hand-written code.

use crate::quant::QuantType;

/// Unified quantization format IR (28 formats).
#[derive(Debug, Clone, PartialEq)]
pub enum QuantFormat {
    // ── Integer Quantization (6) ──
    Int4PerTensor,
    Int4PerChannel,
    Int4BlockScale { block_size: usize },
    Int8PerTensor,
    Int8PerChannel,
    Int8BlockScale { block_size: usize },

    // ── Floating-Point Quantization (6) ──
    Fp4E2M1,
    Fp6E3M2,
    Fp6E2M3,
    Fp8E4M3,
    Fp8E5M2,
    Fp8E4M3Fn,

    // ── Asymmetric Quantization (4) ──
    Uint4PerTensor,
    Uint4PerChannel,
    Uint8PerTensor,
    Uint8PerChannel,

    // ── Mixed Precision (2) ──
    Fp16Int8Mixed,
    Fp8Int4Mixed,

    // ── Special Formats (4) ──
    Kivi3Bit,
    Kivi4Bit,
    RaBitQ { bits: u8 },
    FwhtRotated { bits: u8 },

    // ── Other (6) ──
    Bf16,
    Tf32,
    Fp32,
    Fp16,
    Binary,
    Ternary,
}

/// Quantization granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGranularity {
    PerTensor,
    PerChannel,
    BlockScale { block_size: usize },
}

/// Quantization IR — symbolic representation for JIT compilation.
#[derive(Debug, Clone)]
pub struct QuantIR {
    pub format: QuantFormat,
    pub input_shape: Vec<usize>,
    pub scale_shape: Vec<usize>,
    pub zero_point: Option<Vec<usize>>,
    pub granularity: QuantGranularity,
}

impl QuantFormat {
    /// Effective bit width.
    pub const fn bits(&self) -> u8 {
        match self {
            Self::Binary => 1,
            Self::Ternary => 2,
            Self::Kivi3Bit => 3,
            Self::Int4PerTensor | Self::Int4PerChannel | Self::Int4BlockScale { .. }
            | Self::Uint4PerTensor | Self::Uint4PerChannel
            | Self::Kivi4Bit | Self::Fp4E2M1 => 4,
            Self::Fp6E3M2 | Self::Fp6E2M3 => 6,
            Self::Int8PerTensor | Self::Int8PerChannel | Self::Int8BlockScale { .. }
            | Self::Uint8PerTensor | Self::Uint8PerChannel
            | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Fp8E4M3Fn => 8,
            Self::Bf16 | Self::Fp16 => 16,
            Self::Tf32 | Self::Fp32 => 32,
            Self::Fp16Int8Mixed => 8,
            Self::Fp8Int4Mixed => 4,
            Self::RaBitQ { bits } | Self::FwhtRotated { bits } => *bits,
        }
    }

    /// Is symmetric quantization (no zero-point).
    pub const fn is_symmetric(&self) -> bool {
        matches!(self,
            Self::Int4PerTensor | Self::Int4PerChannel | Self::Int4BlockScale { .. }
            | Self::Int8PerTensor | Self::Int8PerChannel | Self::Int8BlockScale { .. }
            | Self::Fp4E2M1 | Self::Fp6E3M2 | Self::Fp6E2M3
            | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Fp8E4M3Fn
            | Self::Kivi3Bit | Self::Kivi4Bit | Self::RaBitQ { .. })
    }

    /// Requires scale parameter.
    pub const fn requires_scale(&self) -> bool {
        !matches!(self, Self::Fp32 | Self::Fp16 | Self::Bf16 | Self::Tf32)
    }

    /// Requires zero-point parameter.
    pub const fn requires_zero_point(&self) -> bool {
        matches!(self,
            Self::Uint4PerTensor | Self::Uint4PerChannel
            | Self::Uint8PerTensor | Self::Uint8PerChannel)
    }

    /// Map to existing QuantType (for compatibility).
    pub fn to_quant_type(&self) -> Option<QuantType> {
        match self {
            Self::Int4BlockScale { block_size: 32 } => Some(QuantType::Q4_0),
            Self::Int8BlockScale { block_size: 32 } => Some(QuantType::Q8_0),
            Self::Int4BlockScale { block_size: 256 } => Some(QuantType::Q4K),
            Self::Int8BlockScale { block_size: 256 } => Some(QuantType::Q8K),
            _ => None,
        }
    }
}

impl QuantIR {
    /// Create new quantization IR.
    pub fn new(format: QuantFormat, input_shape: Vec<usize>) -> Self {
        let granularity = match &format {
            QuantFormat::Int4PerTensor | QuantFormat::Int8PerTensor
            | QuantFormat::Uint4PerTensor | QuantFormat::Uint8PerTensor => QuantGranularity::PerTensor,
            QuantFormat::Int4PerChannel | QuantFormat::Int8PerChannel
            | QuantFormat::Uint4PerChannel | QuantFormat::Uint8PerChannel => QuantGranularity::PerChannel,
            QuantFormat::Int4BlockScale { block_size } | QuantFormat::Int8BlockScale { block_size } => {
                QuantGranularity::BlockScale { block_size: *block_size }
            }
            _ => QuantGranularity::PerTensor,
        };

        let scale_shape = match granularity {
            QuantGranularity::PerTensor => vec![1],
            QuantGranularity::PerChannel => vec![input_shape[0]],
            QuantGranularity::BlockScale { block_size } => {
                let num_blocks = input_shape.iter().product::<usize>() / block_size;
                vec![num_blocks]
            }
        };

        let zero_point = if format.requires_zero_point() {
            Some(scale_shape.clone())
        } else {
            None
        };

        Self { format, input_shape, scale_shape, zero_point, granularity }
    }

    /// Effective bit width.
    pub fn bits(&self) -> u8 {
        self.format.bits()
    }

    /// Is symmetric quantization.
    pub fn is_symmetric(&self) -> bool {
        self.format.is_symmetric()
    }

    /// Compute output size in bytes.
    pub fn output_bytes(&self) -> usize {
        let num_elements: usize = self.input_shape.iter().product();
        let bits = self.bits() as usize;
        (num_elements * bits + 7) / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_format_bits() {
        assert_eq!(QuantFormat::Int4PerTensor.bits(), 4);
        assert_eq!(QuantFormat::Int8PerTensor.bits(), 8);
        assert_eq!(QuantFormat::Fp8E4M3.bits(), 8);
        assert_eq!(QuantFormat::Kivi3Bit.bits(), 3);
        assert_eq!(QuantFormat::Binary.bits(), 1);
        assert_eq!(QuantFormat::Ternary.bits(), 2);
    }

    #[test]
    fn test_quant_format_symmetric() {
        assert!(QuantFormat::Int4PerTensor.is_symmetric());
        assert!(QuantFormat::Int8PerChannel.is_symmetric());
        assert!(!QuantFormat::Uint4PerTensor.is_symmetric());
        assert!(!QuantFormat::Uint8PerChannel.is_symmetric());
    }

    #[test]
    fn test_quant_format_requirements() {
        assert!(QuantFormat::Int4PerTensor.requires_scale());
        assert!(!QuantFormat::Fp32.requires_scale());
        assert!(QuantFormat::Uint4PerTensor.requires_zero_point());
        assert!(!QuantFormat::Int4PerTensor.requires_zero_point());
    }

    #[test]
    fn test_quant_ir_creation() {
        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![1024, 768]);
        assert_eq!(ir.bits(), 8);
        assert_eq!(ir.scale_shape, vec![1]);
        assert!(ir.zero_point.is_none());
    }

    #[test]
    fn test_quant_ir_per_channel() {
        let ir = QuantIR::new(QuantFormat::Int4PerChannel, vec![128, 512]);
        assert_eq!(ir.scale_shape, vec![128]);
        assert!(ir.zero_point.is_none());
    }

    #[test]
    fn test_quant_ir_block_scale() {
        let ir = QuantIR::new(
            QuantFormat::Int4BlockScale { block_size: 32 },
            vec![1024, 768]
        );
        let num_blocks = (1024 * 768) / 32;
        assert_eq!(ir.scale_shape, vec![num_blocks]);
    }

    #[test]
    fn test_quant_ir_output_bytes() {
        let ir = QuantIR::new(QuantFormat::Int4PerTensor, vec![1024]);
        assert_eq!(ir.output_bytes(), 512); // 1024 * 4 bits / 8

        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![1024]);
        assert_eq!(ir.output_bytes(), 1024); // 1024 * 8 bits / 8
    }

    #[test]
    fn test_format_count() {
        // Verify 28 formats are defined
        let formats = vec![
            QuantFormat::Int4PerTensor,
            QuantFormat::Int4PerChannel,
            QuantFormat::Int4BlockScale { block_size: 32 },
            QuantFormat::Int8PerTensor,
            QuantFormat::Int8PerChannel,
            QuantFormat::Int8BlockScale { block_size: 32 },
            QuantFormat::Fp4E2M1,
            QuantFormat::Fp6E3M2,
            QuantFormat::Fp6E2M3,
            QuantFormat::Fp8E4M3,
            QuantFormat::Fp8E5M2,
            QuantFormat::Fp8E4M3Fn,
            QuantFormat::Uint4PerTensor,
            QuantFormat::Uint4PerChannel,
            QuantFormat::Uint8PerTensor,
            QuantFormat::Uint8PerChannel,
            QuantFormat::Fp16Int8Mixed,
            QuantFormat::Fp8Int4Mixed,
            QuantFormat::Kivi3Bit,
            QuantFormat::Kivi4Bit,
            QuantFormat::RaBitQ { bits: 2 },
            QuantFormat::FwhtRotated { bits: 4 },
            QuantFormat::Bf16,
            QuantFormat::Tf32,
            QuantFormat::Fp32,
            QuantFormat::Fp16,
            QuantFormat::Binary,
            QuantFormat::Ternary,
        ];
        assert_eq!(formats.len(), 28, "Must support exactly 28 quantization formats");
    }

    // ── Extended coverage ──

    #[test]
    fn test_fp_format_bits() {
        assert_eq!(QuantFormat::Fp4E2M1.bits(), 4);
        assert_eq!(QuantFormat::Fp6E3M2.bits(), 6);
        assert_eq!(QuantFormat::Fp6E2M3.bits(), 6);
        assert_eq!(QuantFormat::Fp8E4M3.bits(), 8);
        assert_eq!(QuantFormat::Fp8E5M2.bits(), 8);
        assert_eq!(QuantFormat::Fp16.bits(), 16);
        assert_eq!(QuantFormat::Bf16.bits(), 16);
        assert_eq!(QuantFormat::Fp32.bits(), 32);
        assert_eq!(QuantFormat::Tf32.bits(), 32);
    }

    #[test]
    fn test_mixed_precision_bits() {
        assert_eq!(QuantFormat::Fp16Int8Mixed.bits(), 8);
        assert_eq!(QuantFormat::Fp8Int4Mixed.bits(), 4);
    }

    #[test]
    fn test_variable_bits_formats() {
        assert_eq!(QuantFormat::RaBitQ { bits: 1 }.bits(), 1);
        assert_eq!(QuantFormat::RaBitQ { bits: 4 }.bits(), 4);
        assert_eq!(QuantFormat::FwhtRotated { bits: 2 }.bits(), 2);
        assert_eq!(QuantFormat::FwhtRotated { bits: 8 }.bits(), 8);
    }

    #[test]
    fn test_all_fp_formats_do_not_require_zero_point() {
        for fmt in [
            QuantFormat::Fp4E2M1, QuantFormat::Fp6E3M2, QuantFormat::Fp6E2M3,
            QuantFormat::Fp8E4M3, QuantFormat::Fp8E5M2, QuantFormat::Fp8E4M3Fn,
        ] {
            assert!(!fmt.requires_zero_point(), "{:?} should not require zero_point", fmt);
        }
    }

    #[test]
    fn test_native_formats_no_scale_no_zp() {
        for fmt in [QuantFormat::Fp32, QuantFormat::Fp16, QuantFormat::Bf16, QuantFormat::Tf32] {
            assert!(!fmt.requires_scale(), "{:?} should not require scale", fmt);
            assert!(!fmt.requires_zero_point(), "{:?} should not require zero_point", fmt);
        }
    }

    #[test]
    fn test_uint_formats_asymmetric() {
        for fmt in [
            QuantFormat::Uint4PerTensor, QuantFormat::Uint4PerChannel,
            QuantFormat::Uint8PerTensor, QuantFormat::Uint8PerChannel,
        ] {
            assert!(!fmt.is_symmetric(), "{:?} should be asymmetric", fmt);
            assert!(fmt.requires_zero_point(), "{:?} should require zero_point", fmt);
        }
    }

    #[test]
    fn test_kivi_formats_symmetric() {
        assert!(QuantFormat::Kivi3Bit.is_symmetric());
        assert!(QuantFormat::Kivi4Bit.is_symmetric());
    }

    #[test]
    fn test_to_quant_type_mapping() {
        assert_eq!(QuantFormat::Int4BlockScale { block_size: 32 }.to_quant_type(), Some(QuantType::Q4_0));
        assert_eq!(QuantFormat::Int8BlockScale { block_size: 32 }.to_quant_type(), Some(QuantType::Q8_0));
        assert_eq!(QuantFormat::Int4BlockScale { block_size: 256 }.to_quant_type(), Some(QuantType::Q4K));
        assert_eq!(QuantFormat::Int8BlockScale { block_size: 256 }.to_quant_type(), Some(QuantType::Q8K));
    }

    #[test]
    fn test_to_quant_type_unmapped_returns_none() {
        assert_eq!(QuantFormat::Fp32.to_quant_type(), None);
        assert_eq!(QuantFormat::Fp8E4M3.to_quant_type(), None);
        assert_eq!(QuantFormat::Int4PerTensor.to_quant_type(), None);
        assert_eq!(QuantFormat::Binary.to_quant_type(), None);
    }

    #[test]
    fn test_quant_ir_uint_zero_point_populated() {
        let ir = QuantIR::new(QuantFormat::Uint4PerTensor, vec![128]);
        assert_eq!(ir.zero_point, Some(vec![1]));
        assert!(ir.zero_point.is_some());
    }

    #[test]
    fn test_quant_ir_granularity_per_tensor() {
        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![256]);
        assert_eq!(ir.granularity, QuantGranularity::PerTensor);
        assert_eq!(ir.scale_shape, vec![1]);
    }

    #[test]
    fn test_quant_ir_granularity_per_channel() {
        let ir = QuantIR::new(QuantFormat::Int8PerChannel, vec![64, 512]);
        assert_eq!(ir.granularity, QuantGranularity::PerChannel);
        assert_eq!(ir.scale_shape, vec![64]);
    }

    #[test]
    fn test_quant_ir_output_bytes_sub_byte() {
        let ir = QuantIR::new(QuantFormat::Binary, vec![16]);
        assert_eq!(ir.output_bytes(), 2); // 16 * 1 bit / 8 = 2 bytes
    }

    #[test]
    fn test_quant_ir_output_bytes_ternary() {
        let ir = QuantIR::new(QuantFormat::Ternary, vec![16]);
        assert_eq!(ir.output_bytes(), 4); // 16 * 2 bits / 8 = 4 bytes
    }

    #[test]
    fn test_quant_ir_output_bytes_fp16() {
        let ir = QuantIR::new(QuantFormat::Fp16, vec![4]);
        assert_eq!(ir.output_bytes(), 8); // 4 * 16 bits / 8 = 8 bytes
    }

    #[test]
    fn test_quant_format_equality() {
        assert_eq!(QuantFormat::Int4BlockScale { block_size: 32 }, QuantFormat::Int4BlockScale { block_size: 32 });
        assert_ne!(QuantFormat::Int4BlockScale { block_size: 32 }, QuantFormat::Int4BlockScale { block_size: 64 });
        assert_eq!(QuantFormat::RaBitQ { bits: 4 }, QuantFormat::RaBitQ { bits: 4 });
        assert_ne!(QuantFormat::RaBitQ { bits: 2 }, QuantFormat::RaBitQ { bits: 4 });
    }
}
