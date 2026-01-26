use half::f16;

use crate::validation::validate_input_len;

const Q4_BLOCK_SIZE: usize = 32;
const Q4_PACKED_BYTES: usize = 16;
const Q8_BLOCK_SIZE: usize = 32;
const Q8_PACKED_BYTES: usize = 32;

#[inline(always)]
fn checked_mul(a: usize, b: usize, name: &str) -> Result<usize, String> {
    a.checked_mul(b).ok_or_else(|| format!("{name} overflow"))
}

#[inline(always)]
fn validate_q4_layout(
    q_weight_len: usize,
    scales_len: usize,
    n: usize,
    k: usize,
) -> Result<usize, String> {
    if n == 0 || k == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if k % Q4_BLOCK_SIZE != 0 {
        return Err("k must be multiple of 32 for Q4".into());
    }
    let blocks = k / Q4_BLOCK_SIZE;
    let expected_blocks = checked_mul(n, blocks, "q4 blocks")?;
    let expected_q_weight = checked_mul(expected_blocks, Q4_PACKED_BYTES, "q4 weights")?;
    validate_input_len(q_weight_len, expected_q_weight, "q_weight")?;
    validate_input_len(scales_len, expected_blocks, "scales")?;
    Ok(blocks)
}

#[inline(always)]
fn validate_q8_params(
    input_len: usize,
    q_weight_len: usize,
    scales_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<usize, String> {
    if m == 0 || n == 0 || k == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if k % Q8_BLOCK_SIZE != 0 {
        return Err("k must be multiple of 32 for Q8".into());
    }
    let blocks = k / Q8_BLOCK_SIZE;
    let expected_input = checked_mul(m, k, "input")?;
    let expected_blocks = checked_mul(n, blocks, "q8 blocks")?;
    let expected_q_weight = checked_mul(expected_blocks, Q8_PACKED_BYTES, "q8 weights")?;
    validate_input_len(input_len, expected_input, "input")?;
    validate_input_len(q_weight_len, expected_q_weight, "q_weight")?;
    validate_input_len(scales_len, expected_blocks, "scales")?;
    Ok(blocks)
}

#[inline(always)]
fn validate_awq_layout(
    qweight_len: usize,
    qzeros_len: usize,
    scales_len: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<(usize, usize), String> {
    if n == 0 || k == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    if n % 8 != 0 {
        return Err("n must be multiple of 8 for AWQ packing".into());
    }
    if k % group_size != 0 {
        return Err("k must be multiple of group_size for AWQ".into());
    }
    let groups = k / group_size;
    let packed_out = n / 8;
    let expected_qweight = checked_mul(packed_out, k, "qweight")?;
    let expected_qzeros = checked_mul(packed_out, groups, "qzeros")?;
    let expected_scales = checked_mul(n, groups, "scales")?;
    validate_input_len(qweight_len, expected_qweight, "qweight")?;
    validate_input_len(qzeros_len, expected_qzeros, "qzeros")?;
    validate_input_len(scales_len, expected_scales, "scales")?;
    Ok((groups, packed_out))
}

#[inline(always)]
fn validate_q4_params(
    input_len: usize,
    q_weight_len: usize,
    scales_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<usize, String> {
    if m == 0 {
        return Err("Dimensions must be > 0".into());
    }
    let blocks = validate_q4_layout(q_weight_len, scales_len, n, k)?;
    let expected_input = checked_mul(m, k, "input")?;
    validate_input_len(input_len, expected_input, "input")?;
    Ok(blocks)
}

#[inline(always)]
fn validate_awq_params(
    input_len: usize,
    qweight_len: usize,
    qzeros_len: usize,
    scales_len: usize,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<(usize, usize), String> {
    if m == 0 {
        return Err("Dimensions must be > 0".into());
    }
    let (groups, packed_out) =
        validate_awq_layout(qweight_len, qzeros_len, scales_len, n, k, group_size)?;
    let expected_input = checked_mul(m, k, "input")?;
    validate_input_len(input_len, expected_input, "input")?;
    Ok((groups, packed_out))
}

#[inline(always)]
fn dot_q4_block(input: &[f32], q_weight: &[u8], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..Q4_BLOCK_SIZE {
        let byte = q_weight[i >> 1];
        let nibble = if (i & 1) == 0 { byte & 0x0F } else { byte >> 4 };
        let q = (nibble as i8) - 8;
        sum += input[i] * (q as f32) * scale;
    }
    sum
}

#[inline(always)]
fn dot_q8_block(input: &[f32], q_weight: &[i8], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..Q8_BLOCK_SIZE {
        sum += input[i] * (q_weight[i] as f32) * scale;
    }
    sum
}

#[inline(always)]
fn awq_nibble(word: u32, index: usize) -> u8 {
    ((word >> (index * 4)) & 0x0F) as u8
}

/// CPU reference implementation of Q4_0 dequantization.
pub fn q4_dequantize_cpu(
    q_weight: &[u8],
    scales: &[f16],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let blocks = validate_q4_layout(q_weight.len(), scales.len(), n, k)?;
    let output_len = checked_mul(n, k, "output")?;
    let mut output = vec![0.0f32; output_len];
    for out in 0..n {
        let block_offset = out * blocks;
        let out_row = &mut output[out * k..(out + 1) * k];
        for block in 0..blocks {
            let scale = scales[block_offset + block].to_f32();
            let q_offset = (block_offset + block) * Q4_PACKED_BYTES;
            let out_offset = block * Q4_BLOCK_SIZE;
            for i in 0..Q4_BLOCK_SIZE {
                let byte = q_weight[q_offset + (i >> 1)];
                let nibble = if (i & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                let q = (nibble as i8) - 8;
                out_row[out_offset + i] = (q as f32) * scale;
            }
        }
    }
    Ok(output)
}

/// CPU reference implementation of AWQ INT4 dequantization.
pub fn awq_dequantize_cpu(
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[f16],
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<Vec<f32>, String> {
    let (groups, _packed_out) =
        validate_awq_layout(qweight.len(), qzeros.len(), scales.len(), n, k, group_size)?;
    let output_len = checked_mul(n, k, "output")?;
    let mut output = vec![0.0f32; output_len];
    for out in 0..n {
        let packed_row = out / 8;
        let nibble = out % 8;
        let out_row = &mut output[out * k..(out + 1) * k];
        for idx in 0..k {
            let group = idx / group_size;
            let scale = scales[out * groups + group].to_f32();
            let zero_word = qzeros[packed_row * groups + group];
            let zero = awq_nibble(zero_word, nibble) as i32;
            let weight_word = qweight[packed_row * k + idx];
            let w = awq_nibble(weight_word, nibble) as i32 - zero;
            out_row[idx] = (w as f32) * scale;
        }
    }
    Ok(output)
}

/// CPU reference implementation of Q4_0/Q4_K quantized matrix multiplication.
pub fn q4_matmul_cpu(
    input: &[f32],
    q_weight: &[u8],
    scales: &[f16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let blocks = validate_q4_params(input.len(), q_weight.len(), scales.len(), m, n, k)?;
    let output_len = checked_mul(m, n, "output")?;
    let mut output = vec![0.0f32; output_len];
    for row in 0..m {
        let input_row = &input[row * k..(row + 1) * k];
        let out_row = &mut output[row * n..(row + 1) * n];
        for out in 0..n {
            let block_offset = out * blocks;
            let mut sum = 0.0f64;
            for block in 0..blocks {
                let scale = scales[block_offset + block].to_f32();
                let q_offset = (block_offset + block) * Q4_PACKED_BYTES;
                let in_offset = block * Q4_BLOCK_SIZE;
                let block_sum = dot_q4_block(
                    &input_row[in_offset..in_offset + Q4_BLOCK_SIZE],
                    &q_weight[q_offset..q_offset + Q4_PACKED_BYTES],
                    scale,
                );
                sum += block_sum as f64;
            }
            out_row[out] = sum as f32;
        }
    }
    Ok(output)
}

/// CPU reference implementation of Q8_0 quantized matrix multiplication.
pub fn q8_matmul_cpu(
    input: &[f32],
    q_weight: &[i8],
    scales: &[f16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let blocks = validate_q8_params(input.len(), q_weight.len(), scales.len(), m, n, k)?;
    let output_len = checked_mul(m, n, "output")?;
    let mut output = vec![0.0f32; output_len];
    for row in 0..m {
        let input_row = &input[row * k..(row + 1) * k];
        let out_row = &mut output[row * n..(row + 1) * n];
        for out in 0..n {
            let block_offset = out * blocks;
            let mut sum = 0.0f64;
            for block in 0..blocks {
                let scale = scales[block_offset + block].to_f32();
                let q_offset = (block_offset + block) * Q8_PACKED_BYTES;
                let in_offset = block * Q8_BLOCK_SIZE;
                let block_sum = dot_q8_block(
                    &input_row[in_offset..in_offset + Q8_BLOCK_SIZE],
                    &q_weight[q_offset..q_offset + Q8_PACKED_BYTES],
                    scale,
                );
                sum += block_sum as f64;
            }
            out_row[out] = sum as f32;
        }
    }
    Ok(output)
}

/// CPU reference implementation of AWQ INT4 quantized matrix multiplication.
pub fn awq_matmul_cpu(
    input: &[f32],
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[f16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<Vec<f32>, String> {
    let (groups, _packed_out) = validate_awq_params(
        input.len(),
        qweight.len(),
        qzeros.len(),
        scales.len(),
        m,
        n,
        k,
        group_size,
    )?;
    let output_len = checked_mul(m, n, "output")?;
    let mut output = vec![0.0f32; output_len];
    for row in 0..m {
        let input_row = &input[row * k..(row + 1) * k];
        let out_row = &mut output[row * n..(row + 1) * n];
        for out in 0..n {
            let packed_row = out / 8;
            let nibble = out % 8;
            let mut sum = 0.0f64;
            for group in 0..groups {
                let scale = scales[out * groups + group].to_f32();
                let zero_word = qzeros[packed_row * groups + group];
                let zero = awq_nibble(zero_word, nibble) as i32;
                let group_start = group * group_size;
                let group_end = group_start + group_size;
                for idx in group_start..group_end {
                    let weight_word = qweight[packed_row * k + idx];
                    let w = awq_nibble(weight_word, nibble) as i32 - zero;
                    sum += (w as f64) * (scale as f64) * (input_row[idx] as f64);
                }
            }
            out_row[out] = sum as f32;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{
        awq_dequantize_cpu, awq_matmul_cpu, q4_dequantize_cpu, q4_matmul_cpu, q8_matmul_cpu,
        Q4_PACKED_BYTES,
    };
    use half::f16;

    #[test]
    fn q4_matmul_cpu_basic() {
        let m = 1;
        let n = 1;
        let k = 32;
        let input = vec![1.0f32; k];
        let scales = vec![f16::from_f32(1.0); n * (k / 32)];
        let mut q_weight = vec![0u8; n * (k / 32) * Q4_PACKED_BYTES];
        for byte in &mut q_weight {
            *byte = 0x99;
        }
        let output = q4_matmul_cpu(&input, &q_weight, &scales, m, n, k).unwrap();
        assert_eq!(output.len(), 1);
        assert!((output[0] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn q4_dequantize_cpu_basic() {
        let n = 1;
        let k = 32;
        let scales = vec![f16::from_f32(1.0); n * (k / 32)];
        let mut q_weight = vec![0u8; n * (k / 32) * Q4_PACKED_BYTES];
        for byte in &mut q_weight {
            *byte = 0x99;
        }
        let output = q4_dequantize_cpu(&q_weight, &scales, n, k).unwrap();
        assert_eq!(output.len(), k);
        assert!(output.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn q8_matmul_cpu_basic() {
        let m = 1;
        let n = 1;
        let k = 32;
        let input = vec![1.0f32; k];
        let scales = vec![f16::from_f32(1.0); n * (k / 32)];
        let q_weight = vec![1i8; n * k];
        let output = q8_matmul_cpu(&input, &q_weight, &scales, m, n, k).unwrap();
        assert_eq!(output.len(), 1);
        assert!((output[0] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn awq_matmul_cpu_basic() {
        let m = 1;
        let n = 8;
        let k = 8;
        let group_size = 4;
        let input = vec![1.0f32; k];
        let qweight = vec![0x2222_2222u32; (n / 8) * k];
        let qzeros = vec![0x1111_1111u32; (n / 8) * (k / group_size)];
        let scales = vec![f16::from_f32(1.0); n * (k / group_size)];
        let output = awq_matmul_cpu(
            &input,
            &qweight,
            &qzeros,
            &scales,
            m,
            n,
            k,
            group_size,
        )
        .unwrap();
        assert_eq!(output.len(), 8);
        for value in output {
            assert!((value - 8.0).abs() < 1e-6);
        }
    }

    #[test]
    fn awq_dequantize_cpu_basic() {
        let n = 8;
        let k = 8;
        let group_size = 4;
        let qweight = vec![0x2222_2222u32; (n / 8) * k];
        let qzeros = vec![0x1111_1111u32; (n / 8) * (k / group_size)];
        let scales = vec![f16::from_f32(1.0); n * (k / group_size)];
        let output = awq_dequantize_cpu(&qweight, &qzeros, &scales, n, k, group_size).unwrap();
        assert_eq!(output.len(), n * k);
        assert!(output.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }
}
