use crate::backend_trait::{BackendError, BackendResult};
use crate::kernel_types::precompute_rope_tables;
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct RopeCache {
    cos: Vec<f32>,
    sin: Vec<f32>,
    max_seq_len: usize,
    rotary_dim: usize,
}

impl RopeCache {
    pub fn new(max_seq_len: usize, rotary_dim: usize, base: f32, scale: f32) -> Self {
        let (cos, sin) = precompute_rope_tables(max_seq_len, rotary_dim, base, scale);
        Self {
            cos,
            sin,
            max_seq_len,
            rotary_dim,
        }
    }

    fn lookup(&self, position: usize, pair_idx: usize) -> Option<(f32, f32)> {
        let half = self.rotary_dim / 2;
        if position >= self.max_seq_len || pair_idx >= half {
            return None;
        }
        let idx = position * half + pair_idx;
        Some((self.cos[idx], self.sin[idx]))
    }
}

pub fn embedding_lookup(
    tokens: &[u32],
    embedding: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    hidden_size: usize,
) -> BackendResult<()> {
    let seq_len = tokens.len();
    if output.len() != seq_len.saturating_mul(hidden_size) {
        return Err(BackendError::InvalidConfig(
            "embedding output size mismatch".into(),
        ));
    }
    if embedding.len() < vocab_size.saturating_mul(hidden_size) {
        return Err(BackendError::InvalidConfig(
            "embedding weight size mismatch".into(),
        ));
    }
    for (row, &token) in tokens.iter().enumerate() {
        let out_row = &mut output[row * hidden_size..(row + 1) * hidden_size];
        if (token as usize) < vocab_size {
            let base = token as usize * hidden_size;
            out_row.copy_from_slice(&embedding[base..base + hidden_size]);
        } else {
            out_row.fill(0.0);
        }
    }
    Ok(())
}

pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> BackendResult<()> {
    let input_len = m.saturating_mul(k);
    let weight_len = n.saturating_mul(k);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || weight.len() < weight_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "linear buffer size mismatch".into(),
        ));
    }
    if let Some(bias) = bias {
        if bias.len() < n {
            return Err(BackendError::InvalidConfig(
                "linear bias size mismatch".into(),
            ));
        }
    }

    let input = &input[..input_len];
    let weight = &weight[..weight_len];
    let output = &mut output[..output_len];

    let lhs = MatRef::from_row_major_slice(input, m, k);
    let rhs = MatRef::from_row_major_slice(weight, n, k).transpose();
    let mut dst = MatMut::from_row_major_slice_mut(output, m, n);
    matmul(&mut dst, Accum::Replace, &lhs, &rhs, 1.0, Par::Seq);

    if let Some(bias) = bias {
        for row in 0..m {
            let offset = row * n;
            let row_out = &mut output[offset..offset + n];
            if is_avx2_available() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    add_bias_avx2(row_out, bias);
                    continue;
                }
            }
            if is_neon_available() {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    add_bias_neon(row_out, bias);
                    continue;
                }
            }
            add_bias_scalar(row_out, bias);
        }
    }
    Ok(())
}

pub fn linear_f64(
    input: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    output: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) -> BackendResult<()> {
    let input_len = m.saturating_mul(k);
    let weight_len = n.saturating_mul(k);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || weight.len() < weight_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "linear_f64 buffer size mismatch".into(),
        ));
    }
    if let Some(bias) = bias {
        if bias.len() < n {
            return Err(BackendError::InvalidConfig(
                "linear_f64 bias size mismatch".into(),
            ));
        }
    }

    let input = &input[..input_len];
    let weight = &weight[..weight_len];
    let output = &mut output[..output_len];

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

fn add_bias_scalar(row: &mut [f32], bias: &[f32]) {
    for (out, &b) in row.iter_mut().zip(bias.iter()) {
        *out += b;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_bias_avx2(row: &mut [f32], bias: &[f32]) {
    let mut i = 0usize;
    let len = row.len();
    while i + 8 <= len {
        let out = _mm256_loadu_ps(row.as_ptr().add(i));
        let b = _mm256_loadu_ps(bias.as_ptr().add(i));
        let sum = _mm256_add_ps(out, b);
        _mm256_storeu_ps(row.as_mut_ptr().add(i), sum);
        i += 8;
    }
    while i < len {
        row[i] += bias[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_bias_neon(row: &mut [f32], bias: &[f32]) {
    let mut i = 0usize;
    let len = row.len();
    while i + 4 <= len {
        let out = vld1q_f32(row.as_ptr().add(i));
        let b = vld1q_f32(bias.as_ptr().add(i));
        let sum = vaddq_f32(out, b);
        vst1q_f32(row.as_mut_ptr().add(i), sum);
        i += 4;
    }
    while i < len {
        row[i] += bias[i];
        i += 1;
    }
}

pub fn quantize_f32_to_int8(input: &[f32]) -> (Vec<i8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in input {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;
    let mut output = Vec::with_capacity(input.len());
    for &v in input {
        let mut q = (v * inv_scale).round() as i32;
        if q > 127 {
            q = 127;
        } else if q < -127 {
            q = -127;
        }
        output.push(q as i8);
    }
    (output, scale)
}

pub fn dequantize_int8_to_f32(input: &[i8], scale: f32, output: &mut [f32]) -> BackendResult<()> {
    if output.len() < input.len() {
        return Err(BackendError::InvalidConfig(
            "int8 dequantize output size mismatch".into(),
        ));
    }
    for (out, &v) in output.iter_mut().zip(input.iter()) {
        *out = v as f32 * scale;
    }
    Ok(())
}

fn encode_int4(value: i8) -> u8 {
    let clamped = if value > 7 {
        7
    } else if value < -8 {
        -8
    } else {
        value
    };
    (clamped as i8 as u8) & 0x0f
}

fn decode_int4(nibble: u8) -> i8 {
    let value = (nibble & 0x0f) as i8;
    if value & 0x08 != 0 {
        value - 16
    } else {
        value
    }
}

fn quant_bounds(bits: u8) -> (i8, i8, f32) {
    match bits {
        1 => (-1, 1, 1.0),
        2 => {
            let max = 1i8;
            let min = -2i8;
            (min, max, max as f32)
        }
        _ => (0, 0, 1.0),
    }
}

fn encode_signed(bits: u8, value: i8) -> u8 {
    match bits {
        1 => {
            if value >= 0 {
                1
            } else {
                0
            }
        }
        2 => (value as i8 as u8) & 0x03,
        _ => 0,
    }
}

fn decode_signed(bits: u8, raw: u8) -> i8 {
    match bits {
        1 => {
            if raw & 0x01 != 0 {
                1
            } else {
                -1
            }
        }
        2 => {
            let value = (raw & 0x03) as i8;
            if value & 0x02 != 0 {
                value - 4
            } else {
                value
            }
        }
        _ => 0,
    }
}

fn values_per_byte(bits: u8) -> usize {
    8 / bits as usize
}

fn packed_len(bits: u8, values: usize) -> usize {
    let per_byte = values_per_byte(bits);
    if values == 0 {
        0
    } else {
        (values + per_byte - 1) / per_byte
    }
}

fn pack_value(bits: u8, idx: usize, value: i8, data: &mut [u8]) {
    let per_byte = values_per_byte(bits);
    let byte_idx = idx / per_byte;
    let shift = (idx % per_byte) * bits as usize;
    let mask = ((1u16 << bits) - 1) as u8;
    let encoded = encode_signed(bits, value) & mask;
    let slot = data[byte_idx] & !(mask << shift);
    data[byte_idx] = slot | (encoded << shift);
}

fn unpack_value(bits: u8, idx: usize, data: &[u8]) -> i8 {
    let per_byte = values_per_byte(bits);
    let byte_idx = idx / per_byte;
    let shift = (idx % per_byte) * bits as usize;
    let mask = ((1u16 << bits) - 1) as u8;
    let raw = (data[byte_idx] >> shift) & mask;
    decode_signed(bits, raw)
}

pub fn quantize_f32_to_int4(input: &[f32]) -> (Vec<u8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in input {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;
    let mut output = Vec::with_capacity((input.len() + 1) / 2);
    let mut idx = 0;
    while idx < input.len() {
        let q0 = (input[idx] * inv_scale).round() as i8;
        let low = encode_int4(q0);
        let high = if idx + 1 < input.len() {
            let q1 = (input[idx + 1] * inv_scale).round() as i8;
            encode_int4(q1) << 4
        } else {
            0
        };
        output.push(low | high);
        idx += 2;
    }
    (output, scale)
}

pub fn quantize_f32_to_int2(input: &[f32]) -> (Vec<u8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in input {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let (_, _, max_q) = quant_bounds(2);
    let scale = if max_abs > 0.0 { max_abs / max_q } else { 1.0 };
    let inv_scale = 1.0 / scale;
    let mut output = vec![0u8; packed_len(2, input.len())];
    for (idx, &v) in input.iter().enumerate() {
        let mut q = (v * inv_scale).round() as i32;
        if q > 1 {
            q = 1;
        } else if q < -2 {
            q = -2;
        }
        pack_value(2, idx, q as i8, &mut output);
    }
    (output, scale)
}

pub fn quantize_f32_to_int2_matrix(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<(Vec<u8>, f32)> {
    let total = rows.saturating_mul(cols);
    if input.len() < total {
        return Err(BackendError::InvalidConfig(
            "int2 matrix quantize input too small".into(),
        ));
    }
    let mut max_abs = 0.0f32;
    for &v in &input[..total] {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let (_, _, max_q) = quant_bounds(2);
    let scale = if max_abs > 0.0 { max_abs / max_q } else { 1.0 };
    let inv_scale = 1.0 / scale;
    let stride = (cols + 3) / 4;
    let mut output = vec![0u8; rows.saturating_mul(stride)];
    for row in 0..rows {
        let row_base = row * cols;
        let row_out = &mut output[row * stride..(row + 1) * stride];
        for col in 0..cols {
            let mut q = (input[row_base + col] * inv_scale).round() as i32;
            if q > 1 {
                q = 1;
            } else if q < -2 {
                q = -2;
            }
            pack_value(2, col, q as i8, row_out);
        }
    }
    Ok((output, scale))
}

pub fn quantize_f32_to_int1(input: &[f32]) -> (Vec<u8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in input {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let (_, _, max_q) = quant_bounds(1);
    let scale = if max_abs > 0.0 { max_abs / max_q } else { 1.0 };
    let inv_scale = 1.0 / scale;
    let mut output = vec![0u8; packed_len(1, input.len())];
    for (idx, &v) in input.iter().enumerate() {
        let mut q = (v * inv_scale).round() as i32;
        if q > 1 {
            q = 1;
        } else if q < -1 {
            q = -1;
        }
        pack_value(1, idx, q as i8, &mut output);
    }
    (output, scale)
}

pub fn quantize_f32_to_int1_matrix(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> BackendResult<(Vec<u8>, f32)> {
    let total = rows.saturating_mul(cols);
    if input.len() < total {
        return Err(BackendError::InvalidConfig(
            "int1 matrix quantize input too small".into(),
        ));
    }
    let mut max_abs = 0.0f32;
    for &v in &input[..total] {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let (_, _, max_q) = quant_bounds(1);
    let scale = if max_abs > 0.0 { max_abs / max_q } else { 1.0 };
    let inv_scale = 1.0 / scale;
    let stride = (cols + 7) / 8;
    let mut output = vec![0u8; rows.saturating_mul(stride)];
    for row in 0..rows {
        let row_base = row * cols;
        let row_out = &mut output[row * stride..(row + 1) * stride];
        for col in 0..cols {
            let mut q = (input[row_base + col] * inv_scale).round() as i32;
            if q > 1 {
                q = 1;
            } else if q < -1 {
                q = -1;
            }
            pack_value(1, col, q as i8, row_out);
        }
    }
    Ok((output, scale))
}

pub fn dequantize_int4_to_f32(
    packed: &[u8],
    len: usize,
    scale: f32,
    output: &mut [f32],
) -> BackendResult<()> {
    if output.len() < len {
        return Err(BackendError::InvalidConfig(
            "int4 dequantize output size mismatch".into(),
        ));
    }
    for i in 0..len {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        output[i] = decode_int4(nibble) as f32 * scale;
    }
    Ok(())
}

pub fn dequantize_int2_to_f32(
    packed: &[u8],
    len: usize,
    scale: f32,
    output: &mut [f32],
) -> BackendResult<()> {
    if output.len() < len {
        return Err(BackendError::InvalidConfig(
            "int2 dequantize output size mismatch".into(),
        ));
    }
    for i in 0..len {
        let q = unpack_value(2, i, packed);
        output[i] = q as f32 * scale;
    }
    Ok(())
}

pub fn dequantize_int1_to_f32(
    packed: &[u8],
    len: usize,
    scale: f32,
    output: &mut [f32],
) -> BackendResult<()> {
    if output.len() < len {
        return Err(BackendError::InvalidConfig(
            "int1 dequantize output size mismatch".into(),
        ));
    }
    for i in 0..len {
        let q = unpack_value(1, i, packed);
        output[i] = q as f32 * scale;
    }
    Ok(())
}

pub fn matmul_int8(
    input: &[f32],
    weight: &[i8],
    scale: f32,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> BackendResult<()> {
    let input_len = m.saturating_mul(k);
    let weight_len = n.saturating_mul(k);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || weight.len() < weight_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "int8 matmul buffer size mismatch".into(),
        ));
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let base = j * k;
            for kk in 0..k {
                let w = weight[base + kk] as f32 * scale;
                sum += input[i * k + kk] * w;
            }
            output[i * n + j] = sum;
        }
    }
    Ok(())
}

pub fn matmul_int4(
    input: &[f32],
    weight_packed: &[u8],
    scale: f32,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> BackendResult<()> {
    let input_len = m.saturating_mul(k);
    let weight_stride = (k + 1) / 2;
    let weight_len = n.saturating_mul(weight_stride);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || weight_packed.len() < weight_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "int4 matmul buffer size mismatch".into(),
        ));
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let row_base = j * weight_stride;
            for kk in 0..k {
                let byte = weight_packed[row_base + (kk / 2)];
                let nibble = if kk % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let w = decode_int4(nibble) as f32 * scale;
                sum += input[i * k + kk] * w;
            }
            output[i * n + j] = sum;
        }
    }
    Ok(())
}

pub fn matmul_int2(
    input: &[f32],
    weight_packed: &[u8],
    scale: f32,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> BackendResult<()> {
    let input_len = m.saturating_mul(k);
    let weight_stride = (k + 3) / 4;
    let weight_len = n.saturating_mul(weight_stride);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || weight_packed.len() < weight_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "int2 matmul buffer size mismatch".into(),
        ));
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let row_base = j * weight_stride;
            for kk in 0..k {
                let byte = weight_packed[row_base + (kk / 4)];
                let shift = (kk % 4) * 2;
                let raw = (byte >> shift) & 0x03;
                let w = decode_signed(2, raw) as f32 * scale;
                sum += input[i * k + kk] * w;
            }
            output[i * n + j] = sum;
        }
    }
    Ok(())
}

pub fn matmul_int1(
    input: &[f32],
    weight_packed: &[u8],
    scale: f32,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> BackendResult<()> {
    let input_len = m.saturating_mul(k);
    let weight_stride = (k + 7) / 8;
    let weight_len = n.saturating_mul(weight_stride);
    let output_len = m.saturating_mul(n);
    if input.len() < input_len || weight_packed.len() < weight_len || output.len() < output_len {
        return Err(BackendError::InvalidConfig(
            "int1 matmul buffer size mismatch".into(),
        ));
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let row_base = j * weight_stride;
            for kk in 0..k {
                let byte = weight_packed[row_base + (kk / 8)];
                let raw = (byte >> (kk % 8)) & 0x01;
                let w = decode_signed(1, raw) as f32 * scale;
                sum += input[i * k + kk] * w;
            }
            output[i * n + j] = sum;
        }
    }
    Ok(())
}

pub fn add(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BackendResult<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(BackendError::InvalidConfig(
            "add buffer size mismatch".into(),
        ));
    }
    if is_avx2_available() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            add_avx2(lhs, rhs, output);
            return Ok(());
        }
    }
    if is_neon_available() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            add_neon(lhs, rhs, output);
            return Ok(());
        }
    }
    add_scalar(lhs, rhs, output);
    Ok(())
}

fn add_scalar(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    for ((out, l), r) in output.iter_mut().zip(lhs).zip(rhs) {
        *out = *l + *r;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    let mut i = 0usize;
    let len = lhs.len();
    while i + 8 <= len {
        let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
        let sum = _mm256_add_ps(a, b);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), sum);
        i += 8;
    }
    while i < len {
        output[i] = lhs[i] + rhs[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_neon(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    let mut i = 0usize;
    let len = lhs.len();
    while i + 4 <= len {
        let a = vld1q_f32(lhs.as_ptr().add(i));
        let b = vld1q_f32(rhs.as_ptr().add(i));
        let sum = vaddq_f32(a, b);
        vst1q_f32(output.as_mut_ptr().add(i), sum);
        i += 4;
    }
    while i < len {
        output[i] = lhs[i] + rhs[i];
        i += 1;
    }
}

pub fn rms_norm(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    eps: f32,
) -> BackendResult<()> {
    let total = seq_len.saturating_mul(hidden_size);
    if input.len() < total || output.len() < total || weight.len() < hidden_size {
        return Err(BackendError::InvalidConfig(
            "rms_norm buffer size mismatch".into(),
        ));
    }
    let use_avx2 = is_avx2_available();
    let use_neon = is_neon_available();
    for row in 0..seq_len {
        let base = row * hidden_size;
        let slice = &input[base..base + hidden_size];
        let sum = if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                sum_squares_avx2(slice)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                sum_squares_scalar(slice)
            }
        } else if use_neon {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                sum_squares_neon(slice)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                sum_squares_scalar(slice)
            }
        } else {
            sum_squares_scalar(slice)
        };
        let inv = (sum / hidden_size as f32 + eps).sqrt().recip();
        let out_slice = &mut output[base..base + hidden_size];
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                scale_norm_avx2(slice, weight, out_slice, inv);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                scale_norm_scalar(slice, weight, out_slice, inv);
            }
        } else if use_neon {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                scale_norm_neon(slice, weight, out_slice, inv);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                scale_norm_scalar(slice, weight, out_slice, inv);
            }
        } else {
            scale_norm_scalar(slice, weight, out_slice, inv);
        }
    }
    Ok(())
}

fn sum_squares_scalar(slice: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in slice {
        sum += v * v;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_squares_avx2(slice: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    let len = slice.len();
    while i + 8 <= len {
        let v = _mm256_loadu_ps(slice.as_ptr().add(i));
        let mul = _mm256_mul_ps(v, v);
        acc = _mm256_add_ps(acc, mul);
        i += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += slice[i] * slice[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_squares_neon(slice: &[f32]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let len = slice.len();
    while i + 4 <= len {
        let v = vld1q_f32(slice.as_ptr().add(i));
        acc = vmlaq_f32(acc, v, v);
        i += 4;
    }
    let mut tmp = [0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < len {
        sum += slice[i] * slice[i];
        i += 1;
    }
    sum
}

fn scale_norm_scalar(input: &[f32], weight: &[f32], output: &mut [f32], inv: f32) {
    for i in 0..input.len() {
        output[i] = input[i] * inv * weight[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_norm_avx2(input: &[f32], weight: &[f32], output: &mut [f32], inv: f32) {
    let mut i = 0usize;
    let len = input.len();
    let inv_vec = _mm256_set1_ps(inv);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let w = _mm256_loadu_ps(weight.as_ptr().add(i));
        let y = _mm256_mul_ps(_mm256_mul_ps(x, inv_vec), w);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), y);
        i += 8;
    }
    while i < len {
        output[i] = input[i] * inv * weight[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_norm_neon(input: &[f32], weight: &[f32], output: &mut [f32], inv: f32) {
    let mut i = 0usize;
    let len = input.len();
    let inv_vec = vdupq_n_f32(inv);
    while i + 4 <= len {
        let x = vld1q_f32(input.as_ptr().add(i));
        let w = vld1q_f32(weight.as_ptr().add(i));
        let y = vmulq_f32(vmulq_f32(x, inv_vec), w);
        vst1q_f32(output.as_mut_ptr().add(i), y);
        i += 4;
    }
    while i < len {
        output[i] = input[i] * inv * weight[i];
        i += 1;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn fused_gate_up_silu(gate: &[f32], up: &[f32], output: &mut [f32]) -> BackendResult<()> {
    if gate.len() != up.len() || gate.len() != output.len() {
        return Err(BackendError::InvalidConfig(
            "swi-glu buffer size mismatch".into(),
        ));
    }
    for i in 0..gate.len() {
        output[i] = gate[i] * silu(up[i]);
    }
    Ok(())
}

pub fn apply_rope(
    qkv: &mut [f32],
    positions: &[i32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rope_scale: f32,
    interleaved: bool,
    cache: Option<&RopeCache>,
) -> BackendResult<()> {
    let q_out = num_heads.saturating_mul(head_dim);
    let kv_out = num_kv_heads.saturating_mul(head_dim);
    let qkv_stride = q_out
        .checked_add(2usize.saturating_mul(kv_out))
        .ok_or_else(|| BackendError::InvalidConfig("qkv stride overflow".into()))?;
    let total = seq_len.saturating_mul(qkv_stride);
    if qkv.len() < total || positions.len() < seq_len {
        return Err(BackendError::InvalidConfig(
            "rope buffer size mismatch".into(),
        ));
    }
    if head_dim == 0 {
        return Err(BackendError::InvalidConfig("invalid head_dim".into()));
    }
    let rotary = rotary_dim.min(head_dim);
    let half = rotary / 2;
    if half == 0 {
        return Ok(());
    }
    for token in 0..seq_len {
        let pos_i32 = positions[token];
        let pos = pos_i32 as f32;
        let pos_idx = if pos_i32 >= 0 {
            Some(pos_i32 as usize)
        } else {
            None
        };
        let base = token * qkv_stride;
        for head in 0..num_heads {
            let head_base = base + head * head_dim;
            for dim in 0..half {
                let (idx0, idx1, pair_idx) = if interleaved {
                    let even = dim * 2;
                    (even, even + 1, dim)
                } else {
                    (dim, dim + half, dim)
                };
                if idx1 >= rotary {
                    continue;
                }
                let cached =
                    pos_idx.and_then(|pos_idx| cache.and_then(|c| c.lookup(pos_idx, pair_idx)));
                let (c, s) = if let Some((c, s)) = cached {
                    (c, s)
                } else {
                    let freq = rope_theta.powf(-2.0 * pair_idx as f32 / rotary as f32);
                    let angle = pos * rope_scale * freq;
                    (angle.cos(), angle.sin())
                };
                let q0 = qkv[head_base + idx0];
                let q1 = qkv[head_base + idx1];
                qkv[head_base + idx0] = q0 * c - q1 * s;
                qkv[head_base + idx1] = q0 * s + q1 * c;
            }
        }
        for head in 0..num_kv_heads {
            let head_base = base + q_out + head * head_dim;
            for dim in 0..half {
                let (idx0, idx1, pair_idx) = if interleaved {
                    let even = dim * 2;
                    (even, even + 1, dim)
                } else {
                    (dim, dim + half, dim)
                };
                if idx1 >= rotary {
                    continue;
                }
                let cached =
                    pos_idx.and_then(|pos_idx| cache.and_then(|c| c.lookup(pos_idx, pair_idx)));
                let (c, s) = if let Some((c, s)) = cached {
                    (c, s)
                } else {
                    let freq = rope_theta.powf(-2.0 * pair_idx as f32 / rotary as f32);
                    let angle = pos * rope_scale * freq;
                    (angle.cos(), angle.sin())
                };
                let k0 = qkv[head_base + idx0];
                let k1 = qkv[head_base + idx1];
                qkv[head_base + idx0] = k0 * c - k1 * s;
                qkv[head_base + idx1] = k0 * s + k1 * c;
            }
        }
    }
    Ok(())
}

pub fn fused_qkv_rope(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rope_scale: f32,
    rope_interleaved: bool,
    rope_cache: Option<&RopeCache>,
    positions: &[i32],
) -> BackendResult<()> {
    let q_out = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| BackendError::InvalidConfig("q_out overflow".into()))?;
    let kv_out = num_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| BackendError::InvalidConfig("kv_out overflow".into()))?;
    let qkv_stride = q_out
        .checked_add(2usize.saturating_mul(kv_out))
        .ok_or_else(|| BackendError::InvalidConfig("qkv stride overflow".into()))?;
    linear(
        input,
        weight,
        bias,
        output,
        seq_len,
        qkv_stride,
        hidden_size,
    )?;
    apply_rope(
        output,
        positions,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        rope_theta,
        rope_scale,
        rope_interleaved,
        rope_cache,
    )
}

pub fn flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    kv_seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    scale: f32,
    alibi_slopes: Option<&[f32]>,
    q_pos_offset: usize,
) -> BackendResult<()> {
    let q_stride = num_heads.saturating_mul(head_dim);
    let kv_stride = num_kv_heads.saturating_mul(head_dim);
    let o_stride = q_stride;
    if q.len() < seq_len.saturating_mul(q_stride)
        || k.len() < kv_seq_len.saturating_mul(kv_stride)
        || v.len() < kv_seq_len.saturating_mul(kv_stride)
        || output.len() < seq_len.saturating_mul(o_stride)
    {
        return Err(BackendError::InvalidConfig(
            "flash attention buffer size mismatch".into(),
        ));
    }
    if num_kv_heads == 0 || head_dim == 0 || num_heads == 0 {
        return Err(BackendError::InvalidConfig(
            "invalid attention head config".into(),
        ));
    }

    let mut group = num_heads / num_kv_heads;
    if group == 0 {
        group = 1;
    }
    for q_pos in 0..seq_len {
        let q_pos_abs = q_pos_offset.saturating_add(q_pos) as i32;
        let q_base = q_pos * q_stride;
        for head in 0..num_heads {
            let kv_head = (head / group).min(num_kv_heads - 1);
            let q_offset = q_base + head * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];
            let slope = alibi_slopes
                .and_then(|slopes| slopes.get(head).copied())
                .unwrap_or(0.0);
            let mut kv_limit = kv_seq_len;
            if causal && kv_seq_len >= seq_len {
                let offset = kv_seq_len - seq_len;
                let max_pos = offset + q_pos;
                kv_limit = kv_limit.min(max_pos + 1);
            }
            let out_offset = q_base + head * head_dim;
            let out_slice = &mut output[out_offset..out_offset + head_dim];
            if kv_limit == 0 {
                out_slice.fill(0.0);
                continue;
            }

            let mut max_score = f32::NEG_INFINITY;
            for kv_pos in 0..kv_limit {
                let k_offset = kv_pos * kv_stride + kv_head * head_dim;
                let k_vec = &k[k_offset..k_offset + head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_vec[i] * k_vec[i];
                }
                let bias = slope * (kv_pos as i32 - q_pos_abs) as f32;
                score = score * scale + bias;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut denom = 0.0f32;
            out_slice.fill(0.0);
            for kv_pos in 0..kv_limit {
                let k_offset = kv_pos * kv_stride + kv_head * head_dim;
                let v_offset = kv_pos * kv_stride + kv_head * head_dim;
                let k_vec = &k[k_offset..k_offset + head_dim];
                let v_vec = &v[v_offset..v_offset + head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_vec[i] * k_vec[i];
                }
                let bias = slope * (kv_pos as i32 - q_pos_abs) as f32;
                score = (score * scale + bias - max_score).exp();
                denom += score;
                for i in 0..head_dim {
                    out_slice[i] += score * v_vec[i];
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for i in 0..head_dim {
                out_slice[i] *= inv;
            }
        }
    }
    Ok(())
}

pub fn flash_attention_paged(
    q: &[f32],
    pages: &[Vec<f32>],
    page_size: usize,
    output: &mut [f32],
    seq_len: usize,
    kv_seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    scale: f32,
    alibi_slopes: Option<&[f32]>,
    q_pos_offset: usize,
) -> BackendResult<()> {
    let q_stride = num_heads.saturating_mul(head_dim);
    let kv_stride = num_kv_heads.saturating_mul(head_dim);
    let o_stride = q_stride;
    if q.len() < seq_len.saturating_mul(q_stride) || output.len() < seq_len.saturating_mul(o_stride)
    {
        return Err(BackendError::InvalidConfig(
            "flash attention buffer size mismatch".into(),
        ));
    }
    if page_size == 0 || pages.is_empty() {
        return Err(BackendError::InvalidConfig(
            "paged attention requires non-empty pages".into(),
        ));
    }
    if num_kv_heads == 0 || head_dim == 0 || num_heads == 0 {
        return Err(BackendError::InvalidConfig(
            "invalid attention head config".into(),
        ));
    }

    let mut group = num_heads / num_kv_heads;
    if group == 0 {
        group = 1;
    }
    for q_pos in 0..seq_len {
        let q_pos_abs = q_pos_offset.saturating_add(q_pos) as i32;
        let q_base = q_pos * q_stride;
        for head in 0..num_heads {
            let kv_head = (head / group).min(num_kv_heads - 1);
            let q_offset = q_base + head * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];
            let slope = alibi_slopes
                .and_then(|slopes| slopes.get(head).copied())
                .unwrap_or(0.0);
            let mut kv_limit = kv_seq_len;
            if causal && kv_seq_len >= seq_len {
                let offset = kv_seq_len - seq_len;
                let max_pos = offset + q_pos;
                kv_limit = kv_limit.min(max_pos + 1);
            }
            let out_offset = q_base + head * head_dim;
            let out_slice = &mut output[out_offset..out_offset + head_dim];
            if kv_limit == 0 {
                out_slice.fill(0.0);
                continue;
            }

            let mut max_score = f32::NEG_INFINITY;
            for kv_pos in 0..kv_limit {
                let page = kv_pos / page_size;
                let offset = kv_pos - page * page_size;
                let page_buf = pages.get(page).ok_or_else(|| {
                    BackendError::InvalidConfig("paged attention page out of range".into())
                })?;
                let k_offset = offset * kv_stride + kv_head * head_dim;
                let k_vec = &page_buf[k_offset..k_offset + head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_vec[i] * k_vec[i];
                }
                let bias = slope * (kv_pos as i32 - q_pos_abs) as f32;
                score = score * scale + bias;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut denom = 0.0f32;
            out_slice.fill(0.0);
            for kv_pos in 0..kv_limit {
                let page = kv_pos / page_size;
                let offset = kv_pos - page * page_size;
                let page_buf = pages.get(page).ok_or_else(|| {
                    BackendError::InvalidConfig("paged attention page out of range".into())
                })?;
                let k_offset = offset * kv_stride + kv_head * head_dim;
                let v_offset = page_size * kv_stride + offset * kv_stride + kv_head * head_dim;
                let k_vec = &page_buf[k_offset..k_offset + head_dim];
                let v_vec = &page_buf[v_offset..v_offset + head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_vec[i] * k_vec[i];
                }
                let bias = slope * (kv_pos as i32 - q_pos_abs) as f32;
                score = (score * scale + bias - max_score).exp();
                denom += score;
                for i in 0..head_dim {
                    out_slice[i] += score * v_vec[i];
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for i in 0..head_dim {
                out_slice[i] *= inv;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_expected() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0f32; 6];
        linear(&input, &weight, None, &mut output, 2, 3, 2).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0]);
    }

    #[test]
    fn rms_norm_scales() {
        let input = vec![3.0f32, 4.0];
        let weight = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 2];
        rms_norm(&input, &weight, &mut output, 1, 2, 1e-5).unwrap();
        let norm = (3.0f32 * 3.0 + 4.0 * 4.0) / 2.0;
        let inv = (norm + 1e-5).sqrt().recip();
        assert!((output[0] - 3.0 * inv).abs() < 1e-4);
        assert!((output[1] - 4.0 * inv).abs() < 1e-4);
    }

    #[test]
    fn fused_gate_up_silu_matches() {
        let gate = vec![1.0f32, 2.0];
        let up = vec![0.5f32, -1.0];
        let mut output = vec![0.0f32; 2];
        fused_gate_up_silu(&gate, &up, &mut output).unwrap();
        let expected0 = gate[0] * (up[0] / (1.0 + (-up[0]).exp()));
        let expected1 = gate[1] * (up[1] / (1.0 + (-up[1]).exp()));
        assert!((output[0] - expected0).abs() < 1e-6);
        assert!((output[1] - expected1).abs() < 1e-6);
    }

    #[test]
    fn fused_qkv_rope_matches_linear_rope() {
        let seq_len = 2;
        let hidden_size = 4;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;
        let qkv_stride = q_out + 2 * kv_out;
        let input = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut weight = vec![0.0f32; qkv_stride * hidden_size];
        for out in 0..qkv_stride {
            weight[out * hidden_size + (out % hidden_size)] = 1.0;
        }
        let positions = vec![0i32, 1];
        let mut expected = vec![0.0f32; seq_len * qkv_stride];
        linear(
            &input,
            &weight,
            None,
            &mut expected,
            seq_len,
            qkv_stride,
            hidden_size,
        )
        .unwrap();
        apply_rope(
            &mut expected,
            &positions,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            10000.0,
            1.0,
            false,
            None,
        )
        .unwrap();

        let mut output = vec![0.0f32; seq_len * qkv_stride];
        fused_qkv_rope(
            &input,
            &weight,
            None,
            &mut output,
            seq_len,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            10000.0,
            1.0,
            false,
            None,
            &positions,
        )
        .unwrap();

        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn int8_quantize_roundtrip() {
        let input = vec![0.0f32, 1.0, -2.0, 127.0];
        let (quantized, scale) = quantize_f32_to_int8(&input);
        assert!((scale - 1.0).abs() < 1e-6);
        let mut output = vec![0.0f32; input.len()];
        dequantize_int8_to_f32(&quantized, scale, &mut output).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn int4_quantize_roundtrip() {
        let input = vec![0.0f32, 1.0, -2.0, 7.0, -7.0];
        let (packed, scale) = quantize_f32_to_int4(&input);
        assert!((scale - 1.0).abs() < 1e-6);
        let mut output = vec![0.0f32; input.len()];
        dequantize_int4_to_f32(&packed, input.len(), scale, &mut output).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn int2_quantize_roundtrip() {
        let input = vec![0.0f32, 1.0, -1.0, 0.0, 1.0];
        let (packed, scale) = quantize_f32_to_int2(&input);
        assert!((scale - 1.0).abs() < 1e-6);
        let mut output = vec![0.0f32; input.len()];
        dequantize_int2_to_f32(&packed, input.len(), scale, &mut output).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn int1_quantize_roundtrip() {
        let input = vec![1.0f32, -1.0, 1.0, -1.0];
        let (packed, scale) = quantize_f32_to_int1(&input);
        assert!((scale - 1.0).abs() < 1e-6);
        let mut output = vec![0.0f32; input.len()];
        dequantize_int1_to_f32(&packed, input.len(), scale, &mut output).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn int8_matmul_matches_f32() {
        let input = vec![1.0f32, -1.0, 0.5, 2.0, 0.0, -1.5];
        let weight = vec![127.0f32, -1.0, 3.0, 2.0, -4.0, 5.0];
        let (q_weight, scale) = quantize_f32_to_int8(&weight);
        let mut expected = vec![0.0f32; 4];
        linear(&input, &weight, None, &mut expected, 2, 2, 3).unwrap();
        let mut output = vec![0.0f32; 4];
        matmul_int8(&input, &q_weight, scale, &mut output, 2, 2, 3).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn int4_matmul_matches_f32() {
        let input = vec![1.0f32, 2.0, -1.0, 0.5, -2.0, 1.5, 0.0, -0.5];
        let weight = vec![
            7.0f32, -3.0, 2.0, 1.0, -7.0, 4.0, 0.0, 5.0, 6.0, -2.0, 3.0, -1.0,
        ];
        let (packed, scale) = quantize_f32_to_int4(&weight);
        let mut expected = vec![0.0f32; 6];
        linear(&input, &weight, None, &mut expected, 2, 3, 4).unwrap();
        let mut output = vec![0.0f32; 6];
        matmul_int4(&input, &packed, scale, &mut output, 2, 3, 4).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn int2_matmul_matches_f32() {
        let input = vec![1.0f32, -1.0, 0.5, -0.5, 1.0, 0.0];
        let weight = vec![1.0f32, -1.0, 0.0, 1.0, 0.0, -1.0];
        let (packed, scale) = quantize_f32_to_int2_matrix(&weight, 2, 3).unwrap();
        let mut expected = vec![0.0f32; 4];
        linear(&input, &weight, None, &mut expected, 2, 2, 3).unwrap();
        let mut output = vec![0.0f32; 4];
        matmul_int2(&input, &packed, scale, &mut output, 2, 2, 3).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn int1_matmul_matches_f32() {
        let input = vec![1.0f32, -1.0, 0.5, -0.5];
        let weight = vec![1.0f32, -1.0, -1.0, 1.0];
        let (packed, scale) = quantize_f32_to_int1_matrix(&weight, 2, 2).unwrap();
        let mut expected = vec![0.0f32; 4];
        linear(&input, &weight, None, &mut expected, 2, 2, 2).unwrap();
        let mut output = vec![0.0f32; 4];
        matmul_int1(&input, &packed, scale, &mut output, 2, 2, 2).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn rope_precompute_matches_online() {
        let seq_len = 2;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;
        let stride = q_out + 2 * kv_out;
        let mut qkv_online = (0..seq_len * stride)
            .map(|i| i as f32 * 0.1)
            .collect::<Vec<_>>();
        let mut qkv_pre = qkv_online.clone();
        let positions = vec![0i32, 1];
        let cache = RopeCache::new(seq_len, head_dim, 10000.0, 1.0);

        apply_rope(
            &mut qkv_online,
            &positions,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            10000.0,
            1.0,
            false,
            None,
        )
        .unwrap();
        apply_rope(
            &mut qkv_pre,
            &positions,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            10000.0,
            1.0,
            false,
            Some(&cache),
        )
        .unwrap();

        for (a, b) in qkv_online.iter().zip(qkv_pre.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn flash_attention_alibi_biases_toward_recent() {
        let q = vec![1.0f32, 1.0];
        let k = vec![1.0f32, 1.0];
        let v = vec![1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        flash_attention(
            &q,
            &k,
            &v,
            &mut output,
            2,
            2,
            1,
            1,
            1,
            true,
            1.0,
            Some(&[1.0]),
            0,
        )
        .unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
        let e = std::f32::consts::E;
        let expected = (1.0 + 2.0 * e) / (1.0 + e);
        assert!((output[1] - expected).abs() < 1e-5);
    }

    #[test]
    fn flash_attention_paged_matches_contiguous() {
        let seq_len = 2;
        let kv_seq_len = 3;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let q = vec![1.0f32; seq_len * num_heads * head_dim];
        let k = vec![0.5f32, 1.0, 0.5, 1.0, 0.5, 1.0];
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_contig = vec![0.0f32; seq_len * num_heads * head_dim];
        flash_attention(
            &q,
            &k,
            &v,
            &mut out_contig,
            seq_len,
            kv_seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            true,
            1.0,
            None,
            0,
        )
        .unwrap();

        let page_size = 2;
        let kv_stride = num_kv_heads * head_dim;
        let page_len = page_size * kv_stride * 2;
        let mut pages = vec![vec![0.0f32; page_len]; 2];
        pages[0][0..page_size * kv_stride].copy_from_slice(&k[0..page_size * kv_stride]);
        pages[0][page_size * kv_stride..page_len].copy_from_slice(&v[0..page_size * kv_stride]);
        let k_tail = &k[page_size * kv_stride..kv_seq_len * kv_stride];
        let v_tail = &v[page_size * kv_stride..kv_seq_len * kv_stride];
        pages[1][0..k_tail.len()].copy_from_slice(k_tail);
        pages[1][page_size * kv_stride..page_size * kv_stride + v_tail.len()]
            .copy_from_slice(v_tail);

        let mut out_paged = vec![0.0f32; seq_len * num_heads * head_dim];
        flash_attention_paged(
            &q,
            &pages,
            page_size,
            &mut out_paged,
            seq_len,
            kv_seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            true,
            1.0,
            None,
            0,
        )
        .unwrap();

        for (a, b) in out_contig.iter().zip(out_paged.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
