//! Quantization format conversion and graph integration.
//!
//! REQ-JIT-QUANT-001: Format conversion via FP32 intermediate + graph-level quantization.

use crate::compiler::quant_ir::{QuantFormat, QuantIR};
use crate::compiler::graph::{CompilerGraph, OpKind};
// QuantCodegen/PtxQuantCodegen 已迁移到 Register VM。
// 量化 codegen 将通过 VmInstr::Transcendental(Dequant) + IsaLower 实现。
use crate::types::CompilerError;

/// Convert between quantization formats via FP32 intermediate.

// ============================================================================
// Helper functions (defined first to avoid forward reference errors)
// ============================================================================

fn compute_scale_int8(input: &[f32]) -> f32 {
    let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    max_abs / 127.0
}

fn compute_scale_int4(input: &[f32]) -> f32 {
    let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    max_abs / 7.0
}

fn compute_scale_rabitq(input: &[f32], bits: u8) -> f32 {
    let max_val = (1 << (bits - 1)) - 1;
    let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    max_abs / max_val as f32
}

fn extract_bits(data: &[u8], index: usize, bits: u8) -> i32 {
    let bit_offset = index * bits as usize;
    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;
    let mask = (1u32 << bits) - 1;

    let mut value = 0u32;
    let mut bits_read = 0u8;
    let mut byte_idx = byte_offset;

    while bits_read < bits {
        let bits_in_byte = 8 - if bits_read == 0 { bit_shift } else { 0 };
        let bits_to_read = bits.saturating_sub(bits_read).min(bits_in_byte as u8);
        let shift = if bits_read == 0 { bit_shift } else { 0 };

        if byte_idx < data.len() {
            let byte_val = (data[byte_idx] >> shift) as u32;
            value |= (byte_val & ((1 << bits_to_read) - 1)) << bits_read;
        }

        bits_read += bits_to_read;
        byte_idx += 1;
    }

    value &= mask;
    let sign_bit = 1 << (bits - 1);
    if value & sign_bit != 0 {
        (value | !mask) as i32
    } else {
        value as i32
    }
}

fn pack_bits(data: &mut [u8], index: usize, bits: u8, value: i32) {
    let bit_offset = index * bits as usize;
    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;
    let mask = (1u32 << bits) - 1;
    let mut val = (value as u32) & mask;

    let mut bits_written = 0u8;
    let mut byte_idx = byte_offset;

    while bits_written < bits && byte_idx < data.len() {
        let bits_in_byte = 8 - if bits_written == 0 { bit_shift } else { 0 };
        let bits_to_write = bits.saturating_sub(bits_written).min(bits_in_byte as u8);
        let shift = if bits_written == 0 { bit_shift } else { 0 };

        let byte_mask = ((1u8 << bits_to_write) - 1) << shift;
        data[byte_idx] = (data[byte_idx] & !byte_mask) | ((val as u8) << shift);

        val >>= bits_to_write;
        bits_written += bits_to_write;
        byte_idx += 1;
    }
}

pub fn convert_quant_format(
    input: &[u8],
    from: &QuantIR,
    to: &QuantIR,
) -> Result<Vec<u8>, CompilerError> {
    // Step 1: Dequantize to FP32 (need scale from metadata, use placeholder)
    let fp32 = dequantize_to_fp32(input, from)?;

    // Step 2: Quantize to target format
    let (output, _scale) = quantize_from_fp32(&fp32, to)?;

    Ok(output)
}

/// Dequantize to FP32 (scalar reference implementation).
fn dequantize_to_fp32(input: &[u8], ir: &QuantIR) -> Result<Vec<f32>, CompilerError> {
    let num_elements: usize = ir.input_shape.iter().product();
    let mut output = vec![0.0f32; num_elements];

    match ir.format {
        QuantFormat::Int8PerTensor => {
            // Compute scale from input data (should match quantization scale)
            let max_q = input.iter().map(|&b| (b as i8).abs()).max().unwrap_or(1) as f32;
            let scale = max_q / 127.0;
            for i in 0..num_elements {
                let q = input[i] as i8;
                output[i] = q as f32 * scale;
            }
        }
        QuantFormat::Int4PerTensor => {
            // Compute scale from input data
            let mut max_q = 0i8;
            for i in 0..num_elements {
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    (input[byte_idx] & 0x0F) as i8
                } else {
                    ((input[byte_idx] >> 4) & 0x0F) as i8
                };
                let q = if nibble > 7 { nibble - 16 } else { nibble };
                max_q = max_q.max(q.abs());
            }
            let scale = max_q as f32 / 7.0;
            for i in 0..num_elements {
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    (input[byte_idx] & 0x0F) as i8
                } else {
                    ((input[byte_idx] >> 4) & 0x0F) as i8
                };
                let q = if nibble > 7 { nibble - 16 } else { nibble };
                output[i] = q as f32 * scale;
            }
        }
        QuantFormat::RaBitQ { bits } => {
            let max_val = (1 << (bits - 1)) - 1;
            let mut max_q = 0i32;
            for i in 0..num_elements {
                let q = extract_bits(input, i, bits);
                max_q = max_q.max(q.abs());
            }
            let scale = max_q as f32 / max_val as f32;
            for i in 0..num_elements {
                let q = extract_bits(input, i, bits);
                output[i] = q as f32 * scale;
            }
        }
        _ => return Err(format!("Dequantize not implemented for {:?}", ir.format).into()),
    }

    Ok(output)
}

/// Quantize from FP32 (scalar reference implementation).
/// Returns (quantized_data, scale) tuple.
pub fn quantize_from_fp32(input: &[f32], ir: &QuantIR) -> Result<(Vec<u8>, f32), CompilerError> {
    let output_bytes = ir.output_bytes();
    let mut output = vec![0u8; output_bytes];

    match ir.format {
        QuantFormat::Int8PerTensor => {
            let scale = compute_scale_int8(input);
            for (i, &val) in input.iter().enumerate() {
                let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
                output[i] = q as u8;
            }
            Ok((output, scale))
        }
        QuantFormat::Int4PerTensor => {
            let scale = compute_scale_int4(input);
            for (i, &val) in input.iter().enumerate() {
                let q = (val / scale).round().clamp(-8.0, 7.0) as i8;
                let nibble = (q & 0x0F) as u8;
                let byte_idx = i / 2;
                if i % 2 == 0 {
                    output[byte_idx] = nibble;
                } else {
                    output[byte_idx] |= nibble << 4;
                }
            }
            Ok((output, scale))
        }
        QuantFormat::RaBitQ { bits } => {
            let max_val = (1 << (bits - 1)) - 1;
            let scale = compute_scale_rabitq(input, bits);
            for (i, &val) in input.iter().enumerate() {
                let q = (val / scale).round().clamp(-(max_val as f32) - 1.0, max_val as f32) as i32;
                pack_bits(&mut output, i, bits, q);
            }
            Ok((output, scale))
        }
        _ => Err(format!("Quantize not implemented for {:?}", ir.format).into()),
    }
}

/// Dequantize to FP32 with explicit scale.
fn dequantize_to_fp32_with_scale(input: &[u8], ir: &QuantIR, scale: f32) -> Result<Vec<f32>, CompilerError> {
    let num_elements: usize = ir.input_shape.iter().product();
    let mut output = vec![0.0f32; num_elements];

    match ir.format {
        QuantFormat::Int8PerTensor => {
            for i in 0..num_elements {
                let q = input[i] as i8;
                output[i] = q as f32 * scale;
            }
        }
        QuantFormat::Int4PerTensor => {
            for i in 0..num_elements {
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    (input[byte_idx] & 0x0F) as i8
                } else {
                    ((input[byte_idx] >> 4) & 0x0F) as i8
                };
                let q = if nibble > 7 { nibble - 16 } else { nibble };
                output[i] = q as f32 * scale;
            }
        }
        QuantFormat::RaBitQ { bits } => {
            let max_val = (1 << (bits - 1)) - 1;
            for i in 0..num_elements {
                let q = extract_bits(input, i, bits);
                output[i] = q as f32 * scale;
            }
        }
        _ => return Err(format!("Dequantize not implemented for {:?}", ir.format).into()),
    }

    Ok(output)
}


/// Apply quantization to a CompilerGraph.
pub fn apply_quantization(
    graph: &mut CompilerGraph,
    format: QuantFormat,
) -> Result<(), CompilerError> {
    // Find all weight tensors that should be quantized
    for op in &mut graph.ops {
        if should_quantize_op(&op.kind) {
            let ir = QuantIR::new(format.clone(), vec![1024]); // Placeholder shape

            // 量化 kernel 生成已迁移到 Register VM (VmInstr 路径)
            let _ = ir;

            // Mark op as quantized (would store kernel in real implementation)
            // op.set_quantized(quant_kernel);
        }
    }

    Ok(())
}

fn should_quantize_op(kind: &OpKind) -> bool {
    matches!(kind, OpKind::Gemm { .. } | OpKind::GemmBias { .. })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_quantize_dequantize_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![4]);

        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();
        assert_eq!(quantized.len(), 4);

        let dequantized = dequantize_to_fp32_with_scale(&quantized, &ir, scale).unwrap();
        assert_eq!(dequantized.len(), 4);

        // Check approximate equality (quantization loses precision)
        for (orig, deq) in input.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_int4_quantize_dequantize_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ir = QuantIR::new(QuantFormat::Int4PerTensor, vec![4]);

        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();
        assert_eq!(quantized.len(), 2); // 4 elements * 4 bits / 8 = 2 bytes

        let dequantized = dequantize_to_fp32_with_scale(&quantized, &ir, scale).unwrap();
        assert_eq!(dequantized.len(), 4);
    }

    #[test]
    fn test_format_conversion() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ir_int8 = QuantIR::new(QuantFormat::Int8PerTensor, vec![4]);
        let ir_int4 = QuantIR::new(QuantFormat::Int4PerTensor, vec![4]);

        let (int8_data, _) = quantize_from_fp32(&input, &ir_int8).unwrap();
        let int4_data = convert_quant_format(&int8_data, &ir_int8, &ir_int4).unwrap();

        assert_eq!(int4_data.len(), 2); // 4 elements * 4 bits / 8
    }

    #[test]
    fn test_compute_scale_int8() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let scale = compute_scale_int8(&input);
        assert!((scale - 4.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_scale_int4() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let scale = compute_scale_int4(&input);
        assert!((scale - 4.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_rabitq_2bit_quantize_dequantize() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ir = QuantIR::new(QuantFormat::RaBitQ { bits: 2 }, vec![4]);

        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();
        assert_eq!(quantized.len(), 1); // 4 elements * 2 bits / 8 = 1 byte

        println!("Input: {:?}", input);
        println!("Scale: {}", scale);
        println!("Quantized bytes: {:08b}", quantized[0]);

        let dequantized = dequantize_to_fp32_with_scale(&quantized, &ir, scale).unwrap();
        assert_eq!(dequantized.len(), 4);

        println!("Dequantized: {:?}", dequantized);

        // 2-bit has only 4 levels: -2, -1, 0, 1 (signed), so precision is very low
        for (i, (orig, deq)) in input.iter().zip(dequantized.iter()).enumerate() {
            println!("  [{}] orig={}, deq={}, diff={}", i, orig, deq, (orig - deq).abs());
        }
    }

    #[test]
    fn test_rabitq_4bit_quantize_dequantize() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ir = QuantIR::new(QuantFormat::RaBitQ { bits: 4 }, vec![4]);

        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();
        assert_eq!(quantized.len(), 2); // 4 elements * 4 bits / 8 = 2 bytes

        let dequantized = dequantize_to_fp32_with_scale(&quantized, &ir, scale).unwrap();
        assert_eq!(dequantized.len(), 4);

        for (orig, deq) in input.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.5, "orig={}, deq={}", orig, deq);
        }
    }

    // ── Test 8: compute_scale_int8 with all zeros ──

    #[test]
    fn compute_scale_int8_all_zeros() {
        let input = vec![0.0f32; 8];
        let scale = compute_scale_int8(&input);
        assert_eq!(scale, 0.0);
    }

    // ── Test 9: compute_scale_int4 with negative values ──

    #[test]
    fn compute_scale_int4_negative_values() {
        let input = vec![-5.0f32, 3.0, -1.0, 2.0];
        let scale = compute_scale_int4(&input);
        assert!((scale - 5.0 / 7.0).abs() < 1e-6);
    }

    // ── Test 10: compute_scale_rabitq ──

    #[test]
    fn compute_scale_rabitq_3bit() {
        let input = vec![10.0f32, -6.0, 3.0];
        let scale = compute_scale_rabitq(&input, 3);
        // max_val = (1 << 2) - 1 = 3; max_abs = 10.0; scale = 10.0 / 3.0
        assert!((scale - 10.0 / 3.0).abs() < 1e-6);
    }

    // ── Test 11: extract_bits and pack_bits roundtrip ──

    #[test]
    fn extract_pack_bits_roundtrip_4bit() {
        let mut data = vec![0u8; 4]; // 8 elements * 4 bits / 8 = 4 bytes
        let values = [1i32, -2, 3, -4, 5, -6, 7, -7];

        for (i, &v) in values.iter().enumerate() {
            pack_bits(&mut data, i, 4, v);
        }
        for (i, &v) in values.iter().enumerate() {
            let extracted = extract_bits(&data, i, 4);
            assert_eq!(extracted, v, "mismatch at index {}", i);
        }
    }

    // ── Test 12: should_quantize_op ──

    #[test]
    fn should_quantize_op_gemm_only() {
        use crate::compiler::graph::SymDim;
        assert!(should_quantize_op(&OpKind::Gemm {
            m: SymDim::Concrete(1), n: 64, k: 64, dtype: crate::types::DType::F32, trans_b: false
        }));
        assert!(!should_quantize_op(&OpKind::Silu));
        assert!(!should_quantize_op(&OpKind::Add));
    }

    // ── Test 13: quantize_from_fp32 Int8 single element ──

    #[test]
    fn quantize_int8_single_element() {
        let input = vec![42.0f32];
        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![1]);
        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();
        assert_eq!(quantized.len(), 1);
        assert!(scale > 0.0);
    }

    // ── Test 14: dequantize_to_fp32 Int8 with explicit scale ──

    #[test]
    fn dequantize_int8_with_scale_roundtrip() {
        let input = vec![10.0f32, -20.0, 30.0, -40.0];
        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![4]);
        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();
        let dequantized = dequantize_to_fp32_with_scale(&quantized, &ir, scale).unwrap();
        // Check relative error is reasonable
        for (orig, deq) in input.iter().zip(dequantized.iter()) {
            let rel_err = (orig - deq).abs() / orig.abs().max(1e-6);
            assert!(rel_err < 0.15, "rel_err too high: orig={}, deq={}", orig, deq);
        }
    }

    // ── Test 15: apply_quantization does not error on empty graph ──

    #[test]
    fn apply_quantization_empty_graph() {
        let mut graph = CompilerGraph::new();
        let result = apply_quantization(&mut graph, QuantFormat::Int8PerTensor);
        assert!(result.is_ok());
    }

    // ── Additional tests ──

    #[test]
    fn convert_quant_format_int4_to_int8() {
        // Arrange: start from Int4, convert to Int8
        let input = vec![1.0f32, -2.0, 3.0, -4.0];
        let ir_int4 = QuantIR::new(QuantFormat::Int4PerTensor, vec![4]);
        let ir_int8 = QuantIR::new(QuantFormat::Int8PerTensor, vec![4]);

        // Act: quantize to Int4 first
        let (int4_data, _) = quantize_from_fp32(&input, &ir_int4).unwrap();
        // Then convert Int4 -> Int8
        let int8_data = convert_quant_format(&int4_data, &ir_int4, &ir_int8).unwrap();

        // Assert: Int8 output is 4 bytes (1 byte per element)
        assert_eq!(int8_data.len(), 4);
    }

    #[test]
    fn quantize_from_fp32_all_zeros_int8() {
        let input = vec![0.0f32; 8];
        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![8]);
        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();

        // All zeros should quantize to all zero bytes
        for &b in &quantized {
            assert_eq!(b, 0);
        }
        // Scale should be 0 when max_abs is 0
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn quantize_from_fp32_all_zeros_int4() {
        let input = vec![0.0f32; 4];
        let ir = QuantIR::new(QuantFormat::Int4PerTensor, vec![4]);
        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();

        assert_eq!(quantized.len(), 2);
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn dequantize_to_fp32_int8_single_element() {
        // Arrange: manually construct Int8 data with known scale
        let ir = QuantIR::new(QuantFormat::Int8PerTensor, vec![1]);
        let data = vec![100u8]; // 100 as i8 = 100

        // Act
        let result = dequantize_to_fp32(&data, &ir).unwrap();

        // Assert
        assert_eq!(result.len(), 1);
        // scale = max_q / 127.0 = 100 / 127.0
        // dequantized = 100 * scale ≈ 100 * (100/127) ≈ 78.74
        let expected = 100.0f32 * (100.0f32 / 127.0f32);
        assert!((result[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn should_quantize_op_gemm_bias() {
        use crate::compiler::graph::SymDim;
        assert!(should_quantize_op(&OpKind::GemmBias {
            m: SymDim::Concrete(1), n: 64, k: 64,
            dtype: crate::types::DType::F32, trans_b: false,
        }));
    }

    #[test]
    fn extract_bits_2bit_roundtrip() {
        // 2-bit values: range [-2, 1] (signed 2-bit)
        let mut data = vec![0u8; 2]; // 8 elements * 2 bits / 8 = 2 bytes
        let values = [0i32, 1, -1, -2, 0, 1, -1, -2];

        for (i, &v) in values.iter().enumerate() {
            pack_bits(&mut data, i, 2, v);
        }
        for (i, &v) in values.iter().enumerate() {
            let extracted = extract_bits(&data, i, 2);
            assert_eq!(extracted, v, "2-bit mismatch at index {}", i);
        }
    }

    #[test]
    fn compute_scale_rabitq_4bit() {
        let input = vec![1.0f32, -3.0, 5.0, -7.0];
        let scale = compute_scale_rabitq(&input, 4);
        // max_val = (1 << 3) - 1 = 7; max_abs = 7.0; scale = 7.0 / 7.0 = 1.0
        assert!((scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn quantize_int4_negative_values() {
        let input = vec![-7.0f32, 7.0, -3.0, 3.0];
        let ir = QuantIR::new(QuantFormat::Int4PerTensor, vec![4]);
        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();

        assert_eq!(quantized.len(), 2);
        // scale should be 7.0 / 7.0 = 1.0
        assert!((scale - 1.0).abs() < 1e-6);

        // Dequantize and check approximate roundtrip
        let dequantized = dequantize_to_fp32(&quantized, &ir).unwrap();
        for (orig, deq) in input.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 1.5, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn apply_quantization_with_gemm_ops() {
        use crate::compiler::graph::SymDim;
        use crate::types::DType;

        // Arrange: create a graph with a Gemm op
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[16], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[16, 16], DType::F32);
        let output = graph.add_tensor_concrete("output", &[16], DType::F32);
        graph.add_op(
            OpKind::Gemm { m: SymDim::Concrete(1), n: 16, k: 16, dtype: DType::F32, trans_b: false },
            vec![input, weight],
            vec![output],
            "gemm",
        );

        // Act: should not error even with Gemm ops
        let result = apply_quantization(&mut graph, QuantFormat::Int8PerTensor);
        assert!(result.is_ok());
    }

    #[test]
    fn dequantize_int4_roundtrip_with_scale() {
        let input = vec![0.5f32, -0.5, 1.0, -1.0];
        let ir = QuantIR::new(QuantFormat::Int4PerTensor, vec![4]);
        let (quantized, scale) = quantize_from_fp32(&input, &ir).unwrap();

        let dequantized = dequantize_to_fp32_with_scale(&quantized, &ir, scale).unwrap();
        // 4-bit has limited precision but should be reasonably close
        for (orig, deq) in input.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.3, "orig={}, deq={}", orig, deq);
        }
    }
}
