//! INT2 extreme quantization for KV cache compression.
//!
//! Based on PM-KVQ (Progressive Mixed-precision Quantization) and
//! MiniKV (2-bit extreme quantization) papers.
//!
//! # Key Features
//! - 2-bit quantization (16x compression vs FP16)
//! - Group-wise scale factors for accuracy
//! - Efficient bit-packing (4 INT2 values per byte)
//! - SIMD-friendly unpacking

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// INT2 quantization configuration.
#[derive(Debug, Clone, Copy)]
pub struct Int2QuantConfig {
    /// Group size for scale factors (default: 128).
    pub group_size: usize,
    /// Use symmetric quantization (default: true).
    pub symmetric: bool,
    /// Calibration samples for scale estimation (default: 128).
    pub calibration_samples: usize,
}

impl Default for Int2QuantConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            symmetric: true,
            calibration_samples: 128,
        }
    }
}

impl Int2QuantConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.group_size == 0 {
            return Err("group_size must be > 0");
        }
        if self.calibration_samples == 0 {
            return Err("calibration_samples must be > 0");
        }
        Ok(())
    }
}

/// INT2 quantizer for extreme compression.
#[derive(Debug, Clone, Copy)]
pub struct Int2Quantizer {
    /// Quantization scale factor.
    pub scale: f32,
    /// Zero point (0 for symmetric).
    pub zero_point: i8,
}

impl Int2Quantizer {
    /// Create a symmetric INT2 quantizer from max absolute value.
    pub fn from_absmax(absmax: f32) -> Self {
        // For symmetric INT2: [-1.5, -0.5, 0.5, 1.5] * scale
        // Range covers [-1.5*scale, 1.5*scale]
        let scale = if absmax > 0.0 { absmax / 1.5 } else { 1.0 };
        Self {
            scale,
            zero_point: 0,
        }
    }

    /// Create a quantizer from data statistics.
    pub fn from_data(data: &[f32]) -> Self {
        let absmax = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        Self::from_absmax(absmax)
    }

    /// Quantize a single f32 value to INT2 (0-3).
    #[inline]
    pub fn quantize(&self, value: f32) -> u8 {
        // Map value to [-1.5, 1.5] range then to [0, 3]
        let normalized = value / self.scale;
        let clamped = normalized.clamp(-1.5, 1.5);
        // Map [-1.5, -0.5] -> 0, [-0.5, 0.5] -> 1 or 2, [0.5, 1.5] -> 3
        // Using: round((clamped + 1.5) / 1.0) clamped to [0, 3]
        let quantized = ((clamped + 1.5) + 0.5) as u8;
        quantized.min(3)
    }

    /// Dequantize an INT2 value (0-3) to f32.
    #[inline]
    pub fn dequantize(&self, value: u8) -> f32 {
        // Map [0, 1, 2, 3] to [-1.5, -0.5, 0.5, 1.5] * scale
        let level = match value & 0x03 {
            0 => -1.5,
            1 => -0.5,
            2 => 0.5,
            3 => 1.5,
            _ => 0.0,
        };
        level * self.scale
    }

    /// Quantize a slice of f32 values to INT2.
    pub fn quantize_slice(&self, input: &[f32]) -> Vec<u8> {
        input.iter().map(|&v| self.quantize(v)).collect()
    }

    /// Dequantize a slice of INT2 values to f32.
    pub fn dequantize_slice(&self, input: &[u8]) -> Vec<f32> {
        input.iter().map(|&v| self.dequantize(v)).collect()
    }
}

/// Packed INT2 buffer (4 values per byte).
#[derive(Debug, Clone)]
pub struct Int2PackedBuffer {
    /// Packed data (4 INT2 values per byte).
    data: Vec<u8>,
    /// Per-group scale factors.
    scales: Vec<f32>,
    /// Group size.
    group_size: usize,
    /// Original number of elements.
    num_elements: usize,
}

impl Int2PackedBuffer {
    /// Create an empty buffer.
    pub fn new(group_size: usize) -> Self {
        Self {
            data: Vec::new(),
            scales: Vec::new(),
            group_size,
            num_elements: 0,
        }
    }

    /// Create from f32 data with group-wise quantization.
    pub fn from_f32(input: &[f32], group_size: usize) -> Self {
        let num_elements = input.len();
        let num_groups = (num_elements + group_size - 1) / group_size;
        let packed_bytes = (num_elements + 3) / 4;

        let mut scales = Vec::with_capacity(num_groups);
        let mut quantized = Vec::with_capacity(num_elements);

        // Quantize each group with its own scale
        for group_idx in 0..num_groups {
            let start = group_idx * group_size;
            let end = (start + group_size).min(num_elements);
            let group_data = &input[start..end];

            let quantizer = Int2Quantizer::from_data(group_data);
            scales.push(quantizer.scale);

            for &value in group_data {
                quantized.push(quantizer.quantize(value));
            }
        }

        // Pack 4 INT2 values per byte
        let mut data = vec![0u8; packed_bytes];
        for i in 0..num_elements {
            let byte_idx = i / 4;
            let bit_offset = (3 - (i % 4)) * 2; // MSB first
            data[byte_idx] |= (quantized[i] & 0x03) << bit_offset;
        }

        Self {
            data,
            scales,
            group_size,
            num_elements,
        }
    }

    /// Dequantize to f32.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.num_elements);

        for i in 0..self.num_elements {
            let group_idx = i / self.group_size;
            let scale = self.scales.get(group_idx).copied().unwrap_or(1.0);
            let quantizer = Int2Quantizer {
                scale,
                zero_point: 0,
            };

            let byte_idx = i / 4;
            let bit_offset = (3 - (i % 4)) * 2;
            let int2_value = (self.data[byte_idx] >> bit_offset) & 0x03;

            result.push(quantizer.dequantize(int2_value));
        }

        result
    }

    /// Get number of elements.
    pub fn len(&self) -> usize {
        self.num_elements
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Get packed data size in bytes.
    pub fn packed_size(&self) -> usize {
        self.data.len()
    }

    /// Get compression ratio vs FP32.
    pub fn compression_ratio_f32(&self) -> f32 {
        if self.num_elements == 0 {
            return 1.0;
        }
        let original_bytes = self.num_elements * 4; // FP32
        let packed_bytes = self.data.len() + self.scales.len() * 4;
        original_bytes as f32 / packed_bytes as f32
    }

    /// Get compression ratio vs FP16.
    pub fn compression_ratio_f16(&self) -> f32 {
        if self.num_elements == 0 {
            return 1.0;
        }
        let original_bytes = self.num_elements * 2; // FP16
        let packed_bytes = self.data.len() + self.scales.len() * 4;
        original_bytes as f32 / packed_bytes as f32
    }

    /// Access raw packed data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Access scale factors.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Get group size.
    pub fn group_size(&self) -> usize {
        self.group_size
    }
}

/// Pack 4 INT2 values into a single byte.
#[inline]
pub fn pack_4_int2(a: u8, b: u8, c: u8, d: u8) -> u8 {
    ((a & 0x03) << 6) | ((b & 0x03) << 4) | ((c & 0x03) << 2) | (d & 0x03)
}

/// Unpack a byte into 4 INT2 values.
#[inline]
pub fn unpack_4_int2(byte: u8) -> (u8, u8, u8, u8) {
    (
        (byte >> 6) & 0x03,
        (byte >> 4) & 0x03,
        (byte >> 2) & 0x03,
        byte & 0x03,
    )
}

/// INT2 tensor wrapper for burn integration.
#[derive(Debug, Clone)]
pub struct Int2Tensor<B: Backend> {
    /// Packed INT2 data.
    packed: Int2PackedBuffer,
    /// Original tensor shape.
    shape: Vec<usize>,
    /// Device reference.
    device: B::Device,
    /// Phantom marker.
    _marker: PhantomData<B>,
}

impl<B: Backend> Int2Tensor<B> {
    /// Create from a burn tensor.
    pub fn from_tensor(tensor: Tensor<B, 3>, group_size: usize) -> Result<Self, &'static str> {
        let shape: Vec<usize> = tensor.dims().to_vec();
        let device = tensor.device();

        let data = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to convert tensor to f32")?;

        let packed = Int2PackedBuffer::from_f32(&data, group_size);

        Ok(Self {
            packed,
            shape,
            device,
            _marker: PhantomData,
        })
    }

    /// Convert back to burn tensor.
    pub fn to_tensor(&self) -> Result<Tensor<B, 3>, &'static str> {
        if self.shape.len() != 3 {
            return Err("expected 3D shape");
        }

        let data = self.packed.to_f32();
        let shape = [self.shape[0], self.shape[1], self.shape[2]];

        Ok(Tensor::from_data(
            TensorData::new(data, shape),
            &self.device,
        ))
    }

    /// Get original shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get compression ratio vs FP16.
    pub fn compression_ratio(&self) -> f32 {
        self.packed.compression_ratio_f16()
    }

    /// Get packed buffer reference.
    pub fn packed(&self) -> &Int2PackedBuffer {
        &self.packed
    }
}

/// Batch INT2 quantization for multiple tensors.
pub fn batch_quantize_int2<B: Backend>(
    tensors: &[Tensor<B, 3>],
    group_size: usize,
) -> Result<Vec<Int2Tensor<B>>, &'static str> {
    tensors
        .iter()
        .map(|t| Int2Tensor::from_tensor(t.clone(), group_size))
        .collect()
}

/// Batch INT2 dequantization.
pub fn batch_dequantize_int2<B: Backend>(
    packed: &[Int2Tensor<B>],
) -> Result<Vec<Tensor<B, 3>>, &'static str> {
    packed.iter().map(|p| p.to_tensor()).collect()
}

/// Calculate mean squared error between original and quantized.
pub fn quantization_mse(original: &[f32], quantized: &[f32]) -> f32 {
    if original.len() != quantized.len() || original.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| (o - q).powi(2))
        .sum();

    sum_sq / original.len() as f32
}

/// Calculate signal-to-quantization-noise ratio (SQNR) in dB.
pub fn quantization_sqnr(original: &[f32], quantized: &[f32]) -> f32 {
    if original.len() != quantized.len() || original.is_empty() {
        return 0.0;
    }

    let signal_power: f32 = original.iter().map(|&x| x.powi(2)).sum();
    let noise_power: f32 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| (o - q).powi(2))
        .sum();

    if noise_power < 1e-10 {
        return f32::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int2_quantizer() {
        let quantizer = Int2Quantizer::from_absmax(1.5);

        // Test quantize/dequantize roundtrip
        for &level in &[-1.5, -0.5, 0.5, 1.5] {
            let q = quantizer.quantize(level * quantizer.scale);
            let dq = quantizer.dequantize(q);
            assert!((dq - level * quantizer.scale).abs() < 0.1 * quantizer.scale);
        }
    }

    #[test]
    fn test_pack_unpack() {
        let packed = pack_4_int2(0, 1, 2, 3);
        let (a, b, c, d) = unpack_4_int2(packed);
        assert_eq!((a, b, c, d), (0, 1, 2, 3));

        let packed2 = pack_4_int2(3, 2, 1, 0);
        let (a2, b2, c2, d2) = unpack_4_int2(packed2);
        assert_eq!((a2, b2, c2, d2), (3, 2, 1, 0));
    }

    #[test]
    fn test_int2_packed_buffer() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();

        let packed = Int2PackedBuffer::from_f32(&input, 64);
        assert_eq!(packed.len(), 256);
        assert_eq!(packed.packed_size(), 64); // 256 / 4

        let output = packed.to_f32();
        assert_eq!(output.len(), 256);

        // Check compression ratio (should be ~8x vs FP16)
        let ratio = packed.compression_ratio_f16();
        assert!(ratio > 4.0); // At least 4x compression
    }

    #[test]
    fn test_quantization_quality() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 400.0).collect();

        let packed = Int2PackedBuffer::from_f32(&input, 128);
        let output = packed.to_f32();

        let mse = quantization_mse(&input, &output);
        let sqnr = quantization_sqnr(&input, &output);

        // INT2 will have significant error, but should be usable
        assert!(mse < 1.0);
        assert!(sqnr > 5.0); // At least 5dB SQNR
    }

    #[test]
    fn test_edge_cases() {
        // Empty input
        let packed = Int2PackedBuffer::from_f32(&[], 64);
        assert!(packed.is_empty());
        assert!(packed.to_f32().is_empty());

        // Input smaller than group size
        let small_input = vec![1.0, 2.0, 3.0];
        let packed = Int2PackedBuffer::from_f32(&small_input, 64);
        assert_eq!(packed.len(), 3);
        assert_eq!(packed.scales.len(), 1);
    }

    #[test]
    fn test_compression_ratio() {
        let input: Vec<f32> = vec![0.0; 4096];
        let packed = Int2PackedBuffer::from_f32(&input, 128);

        // FP16 would be 8192 bytes
        // INT2 packed is 1024 bytes + 32*4 = 1152 bytes for scales
        // Ratio should be around 7-8x
        let ratio = packed.compression_ratio_f16();
        assert!(ratio > 6.0);
    }
}
