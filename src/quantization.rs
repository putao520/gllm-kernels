use half::f16;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::backend_trait::{BackendError, BackendResult};
use crate::kernel_types::PackedBits;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Block<const N: usize> {
    pub scale: f16,
    pub data: [u8; N],
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockwiseMatrix<const N: usize> {
    pub blocks: Vec<Block<N>>,
    pub rows: usize,
    pub cols: usize,
    pub block_values: usize,
    pub bits: u8,
}

impl<const N: usize> BlockwiseMatrix<N> {
    pub fn blocks_per_row(&self) -> usize {
        if self.block_values == 0 {
            0
        } else {
            (self.cols + self.block_values - 1) / self.block_values
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedU8 {
    pub bits: PackedBits,
    pub data: Vec<u8>,
    pub values: usize,
}

impl PackedU8 {
    pub fn from_i8(values: &[i8], bits: PackedBits) -> BackendResult<Self> {
        let values_len = values.len();
        let (min_q, max_q, _) = quant_bounds(bits.bits() as u8)?;
        let bytes = packed_storage_bytes(values_len, bits.bits() as u8)?;
        let mut data = vec![0u8; bytes];
        for (idx, value) in values.iter().enumerate() {
            let mut q = *value;
            if q < min_q {
                q = min_q;
            } else if q > max_q {
                q = max_q;
            }
            pack_value(bits.bits() as u8, idx, q, &mut data)?;
        }
        Ok(Self {
            bits,
            data,
            values: values_len,
        })
    }

    pub fn unpack_i8(&self, output: &mut [i8]) -> BackendResult<()> {
        if output.len() < self.values {
            return Err(BackendError::InvalidConfig(
                "packed output too small".into(),
            ));
        }
        for idx in 0..self.values {
            output[idx] = unpack_value(self.bits.bits() as u8, idx, &self.data)?;
        }
        Ok(())
    }

    pub fn values_per_byte(&self) -> usize {
        self.bits.values_per_byte()
    }
}

#[inline]
fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn packed_storage_bytes(values: usize, bits: u8) -> BackendResult<usize> {
    let per_byte = values_per_byte(bits)?;
    if values == 0 {
        return Ok(0);
    }
    values
        .checked_add(per_byte - 1)
        .map(|v| v / per_byte)
        .ok_or_else(|| BackendError::InvalidConfig("packed storage overflow".into()))
}

fn values_per_byte(bits: u8) -> BackendResult<usize> {
    match bits {
        1 | 2 | 4 | 8 => Ok(8 / bits as usize),
        _ => Err(BackendError::InvalidConfig(
            "unsupported packed bits".into(),
        )),
    }
}

fn quant_bounds(bits: u8) -> BackendResult<(i8, i8, f32)> {
    match bits {
        1 => Ok((-1, 1, 1.0)),
        2 | 4 | 8 => {
            let max = (1i16 << (bits - 1)) - 1;
            let min = -1i16 << (bits - 1);
            Ok((min as i8, max as i8, max as f32))
        }
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

fn encode_signed(bits: u8, value: i8) -> BackendResult<u8> {
    match bits {
        1 => Ok(if value >= 0 { 1 } else { 0 }),
        2 | 4 | 8 => {
            let mask = (1u16 << bits) - 1;
            Ok((value as i16 as u16 & mask) as u8)
        }
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

fn decode_signed(bits: u8, raw: u8) -> BackendResult<i8> {
    match bits {
        1 => Ok(if raw & 0x1 != 0 { 1 } else { -1 }),
        2 | 4 | 8 => {
            let mask = (1u16 << bits) - 1;
            let sign_bit = 1u16 << (bits - 1);
            let value = (raw as u16) & mask;
            if value & sign_bit != 0 {
                Ok((value as i16 - (1i16 << bits)) as i8)
            } else {
                Ok(value as i8)
            }
        }
        _ => Err(BackendError::InvalidConfig("unsupported quant bits".into())),
    }
}

fn pack_value(bits: u8, idx: usize, value: i8, data: &mut [u8]) -> BackendResult<()> {
    let per_byte = values_per_byte(bits)?;
    let byte_idx = idx / per_byte;
    if byte_idx >= data.len() {
        return Err(BackendError::InvalidConfig("packed write overflow".into()));
    }
    let shift = ((idx % per_byte) * bits as usize) as u32;
    let mask = ((1u16 << bits) - 1) as u8;
    let encoded = encode_signed(bits, value)? & mask;
    let slot = data[byte_idx] & !(mask << shift);
    data[byte_idx] = slot | (encoded << shift);
    Ok(())
}

fn unpack_value(bits: u8, idx: usize, data: &[u8]) -> BackendResult<i8> {
    let per_byte = values_per_byte(bits)?;
    let byte_idx = idx / per_byte;
    if byte_idx >= data.len() {
        return Err(BackendError::InvalidConfig("packed read overflow".into()));
    }
    let shift = ((idx % per_byte) * bits as usize) as u32;
    let mask = ((1u16 << bits) - 1) as u8;
    let raw = (data[byte_idx] >> shift) & mask;
    decode_signed(bits, raw)
}

fn block_values_for<const N: usize>(bits: u8) -> BackendResult<usize> {
    if N == 0 {
        return Err(BackendError::InvalidConfig("block size must be > 0".into()));
    }
    let per_byte = values_per_byte(bits)?;
    N.checked_mul(per_byte)
        .ok_or_else(|| BackendError::InvalidConfig("block size overflow".into()))
}

pub fn quantize_blockwise<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
) -> BackendResult<BlockwiseMatrix<N>> {
    let block_values = block_values_for::<N>(bits)?;
    let total = rows.saturating_mul(cols);
    if input.len() < total {
        return Err(BackendError::InvalidConfig(
            "blockwise quantize input too small".into(),
        ));
    }

    let (min_q, max_q, max_q_f) = quant_bounds(bits)?;
    let blocks_per_row = if block_values == 0 {
        0
    } else {
        (cols + block_values - 1) / block_values
    };
    let mut blocks = Vec::with_capacity(rows.saturating_mul(blocks_per_row));

    for row in 0..rows {
        let row_base = row * cols;
        for block_idx in 0..blocks_per_row {
            let block_start = row_base + block_idx * block_values;
            let mut max_abs = 0.0f32;
            for t in 0..block_values {
                let idx = block_start + t;
                if idx >= row_base + cols {
                    break;
                }
                let v = input[idx].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            let scale = if max_abs > 0.0 {
                max_abs / max_q_f
            } else {
                1.0
            };
            let inv_scale = 1.0 / scale;
            let mut data = [0u8; N];
            for t in 0..block_values {
                let idx = block_start + t;
                let v = if idx < row_base + cols {
                    input[idx]
                } else {
                    0.0
                };
                let mut q = (v * inv_scale).round() as i32;
                if q < min_q as i32 {
                    q = min_q as i32;
                } else if q > max_q as i32 {
                    q = max_q as i32;
                }
                pack_value(bits, t, q as i8, &mut data)?;
            }
            blocks.push(Block {
                scale: f16::from_f32(scale),
                data,
            });
        }
    }

    Ok(BlockwiseMatrix {
        blocks,
        rows,
        cols,
        block_values,
        bits,
    })
}

pub fn quantize_blockwise_int8<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise::<N>(input, rows, cols, 8)
}

pub fn quantize_blockwise_int4<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise::<N>(input, rows, cols, 4)
}

pub fn quantize_blockwise_int2<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise::<N>(input, rows, cols, 2)
}

pub fn quantize_blockwise_int1<const N: usize>(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<BlockwiseMatrix<N>> {
    quantize_blockwise::<N>(input, rows, cols, 1)
}

pub fn dequantize_blockwise<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    values_per_byte(matrix.bits)?;
    let total = matrix.rows.saturating_mul(matrix.cols);
    if output.len() < total {
        return Err(BackendError::InvalidConfig(
            "blockwise dequantize output too small".into(),
        ));
    }

    let blocks_per_row = matrix.blocks_per_row();
    if matrix.blocks.len() < matrix.rows.saturating_mul(blocks_per_row) {
        return Err(BackendError::InvalidConfig(
            "blockwise matrix storage too small".into(),
        ));
    }

    let expected_block_values = block_values_for::<N>(matrix.bits)?;
    if matrix.block_values != expected_block_values {
        return Err(BackendError::InvalidConfig(
            "blockwise matrix block size mismatch".into(),
        ));
    }

    for row in 0..matrix.rows {
        let row_base = row * matrix.cols;
        for block_idx in 0..blocks_per_row {
            let block = matrix.blocks[row * blocks_per_row + block_idx];
            let scale = block.scale.to_f32();
            let block_start = row_base + block_idx * matrix.block_values;
            let valid = (matrix.cols - block_idx * matrix.block_values).min(matrix.block_values);
            for t in 0..valid {
                let q = unpack_value(matrix.bits, t, &block.data)?;
                output[block_start + t] = q as f32 * scale;
            }
        }
    }
    Ok(())
}

pub fn dequantize_blockwise_int8<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 8 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 8-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise(matrix, output)
}

pub fn dequantize_blockwise_int4<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 4 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 4-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise(matrix, output)
}

pub fn dequantize_blockwise_int2<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 2 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 2-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise(matrix, output)
}

pub fn dequantize_blockwise_int1<const N: usize>(
    matrix: &BlockwiseMatrix<N>,
    output: &mut [f32],
) -> BackendResult<()> {
    if matrix.bits != 1 {
        return Err(BackendError::InvalidConfig(
            "dequantize expects 1-bit blockwise matrix".into(),
        ));
    }
    dequantize_blockwise(matrix, output)
}

pub fn matmul_blockwise<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    values_per_byte(weight.bits)?;
    let k = weight.cols;
    let n = weight.rows;
    let input_len = m.saturating_mul(k);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "blockwise matmul buffer size mismatch".into(),
        ));
    }

    let blocks_per_row = weight.blocks_per_row();
    if weight.blocks.len() < n.saturating_mul(blocks_per_row) {
        return Err(BackendError::InvalidConfig(
            "blockwise matmul weight storage too small".into(),
        ));
    }

    let expected_block_values = block_values_for::<N>(weight.bits)?;
    if weight.block_values != expected_block_values {
        return Err(BackendError::InvalidConfig(
            "blockwise matmul block size mismatch".into(),
        ));
    }

    let block_values = weight.block_values;
    let mut values = vec![0f32; block_values];
    for row in 0..m {
        let row_base = row * k;
        for col in 0..n {
            let mut sum = 0.0f32;
            let weight_base = col * blocks_per_row;
            for block_idx in 0..blocks_per_row {
                let block = weight.blocks[weight_base + block_idx];
                let block_start = block_idx * block_values;
                let valid = (k - block_start).min(block_values);
                decode_block_into(&block, weight.bits, valid, &mut values)?;
                let input_slice = &input[row_base + block_start..row_base + block_start + valid];
                sum += dot_f32(input_slice, &values[..valid]);
            }
            output[row * n + col] = sum;
        }
    }
    Ok(())
}

pub fn matmul_blockwise_int8<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 8 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 8-bit blockwise matrix".into(),
        ));
    }
    matmul_blockwise(input, weight, output, m)
}

pub fn matmul_blockwise_int4<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 4 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 4-bit blockwise matrix".into(),
        ));
    }
    matmul_blockwise(input, weight, output, m)
}

pub fn matmul_blockwise_int2<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 2 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 2-bit blockwise matrix".into(),
        ));
    }
    matmul_blockwise(input, weight, output, m)
}

pub fn matmul_blockwise_int1<const N: usize>(
    input: &[f32],
    weight: &BlockwiseMatrix<N>,
    output: &mut [f32],
    m: usize,
) -> BackendResult<()> {
    if weight.bits != 1 {
        return Err(BackendError::InvalidConfig(
            "matmul expects 1-bit blockwise matrix".into(),
        ));
    }
    matmul_blockwise(input, weight, output, m)
}

fn decode_block_into<const N: usize>(
    block: &Block<N>,
    bits: u8,
    valid: usize,
    out: &mut [f32],
) -> BackendResult<()> {
    let scale = block.scale.to_f32();
    for idx in 0..valid {
        let q = unpack_value(bits, idx, &block.data)?;
        out[idx] = q as f32 * scale;
    }
    Ok(())
}

fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    if is_avx2_available() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if std::arch::is_x86_feature_detected!("fma") {
                return dot_avx2_fma(lhs, rhs, len);
            }
            return dot_avx2(lhs, rhs, len);
        }
    }
    if is_neon_available() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return dot_neon(lhs, rhs, len);
        }
    }
    dot_scalar(lhs, rhs, len)
}

fn dot_scalar(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += lhs[i] * rhs[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
        i += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += lhs[i] * rhs[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
        acc = _mm256_fmadd_ps(a, b, acc);
        i += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += lhs[i] * rhs[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(lhs: &[f32], rhs: &[f32], len: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let a = vld1q_f32(lhs.as_ptr().add(i));
        let b = vld1q_f32(rhs.as_ptr().add(i));
        acc = vmlaq_f32(acc, a, b);
        i += 4;
    }
    let mut tmp = [0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += lhs[i] * rhs[i];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_u8_roundtrip() {
        let values = vec![-2i8, -1, 0, 1, 1, 0, -1, -2, 1];
        let packed = PackedU8::from_i8(&values, PackedBits::Int2).unwrap();
        let mut output = vec![0i8; values.len()];
        packed.unpack_i8(&mut output).unwrap();
        assert_eq!(values, output);

        let values = vec![-1i8, 1, -1, 1, 1, -1, 1, -1];
        let packed = PackedU8::from_i8(&values, PackedBits::Int1).unwrap();
        let mut output = vec![0i8; values.len()];
        packed.unpack_i8(&mut output).unwrap();
        assert_eq!(values, output);
    }

    #[test]
    fn blockwise_int4_roundtrip() {
        let rows = 2;
        let cols = 9;
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 10.0) * 0.25).collect();
        let matrix = quantize_blockwise_int4::<16>(&input, rows, cols).unwrap();
        let mut output = vec![0f32; rows * cols];
        dequantize_blockwise_int4(&matrix, &mut output).unwrap();
        for (orig, deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.5);
        }
    }

    #[test]
    fn blockwise_int4_matmul_matches_dequantized() {
        let rows = 3;
        let cols = 8;
        let input_rows = 2;
        let input: Vec<f32> = (0..input_rows * cols)
            .map(|i| (i as f32 - 5.0) * 0.1)
            .collect();
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.2) - 1.5).collect();
        let matrix = quantize_blockwise_int4::<16>(&weight, rows, cols).unwrap();

        let mut dequant = vec![0f32; rows * cols];
        dequantize_blockwise_int4(&matrix, &mut dequant).unwrap();

        let mut expected = vec![0f32; input_rows * rows];
        for i in 0..input_rows {
            for j in 0..rows {
                let mut sum = 0.0f32;
                for k in 0..cols {
                    sum += input[i * cols + k] * dequant[j * cols + k];
                }
                expected[i * rows + j] = sum;
            }
        }

        let mut output = vec![0f32; input_rows * rows];
        matmul_blockwise_int4(&input, &matrix, &mut output, input_rows).unwrap();

        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn blockwise_matmul_multiple_bits() {
        fn run<const N: usize>(bits: u8) {
            let rows = 4;
            let cols = 7;
            let input_rows = 3;
            let input: Vec<f32> = (0..input_rows * cols)
                .map(|i| (i as f32 - 4.0) * 0.2)
                .collect();
            let weight: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.15) - 1.0).collect();
            let matrix = quantize_blockwise::<N>(&weight, rows, cols, bits).unwrap();

            let mut dequant = vec![0f32; rows * cols];
            dequantize_blockwise(&matrix, &mut dequant).unwrap();

            let mut expected = vec![0f32; input_rows * rows];
            for i in 0..input_rows {
                for j in 0..rows {
                    let mut sum = 0.0f32;
                    for k in 0..cols {
                        sum += input[i * cols + k] * dequant[j * cols + k];
                    }
                    expected[i * rows + j] = sum;
                }
            }

            let mut output = vec![0f32; input_rows * rows];
            matmul_blockwise(&input, &matrix, &mut output, input_rows).unwrap();

            for (a, b) in output.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }

        run::<16>(8);
        run::<16>(4);
        run::<16>(2);
        run::<16>(1);
    }
}
