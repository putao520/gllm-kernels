//! KV cache compression utilities (low-rank, vector quantization, hybrid).

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::ops::mla::CompressedKVCache as MlaCompressedKVCache;
use crate::ops::paged_attention::PagedKVCache;

const ENERGY_FRACTION: f32 = 0.9;
const OUTLIER_STD_FACTOR: f32 = 3.0;
const OUTLIER_MAD_FACTOR: f32 = 6.0;

/// Compression method for KV caches.
#[derive(Debug, Clone)]
pub enum CompressionMethod {
    /// Low-rank projection (PALU-style).
    LowRank { rank: usize },
    /// Vector quantization (CommVQ-style).
    VectorQuantization { codebook_size: usize },
    /// Low-rank projection with quantization.
    Hybrid { rank: usize, quant_bits: u8 },
}

/// Layout metadata for compressed KV tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    /// Unbatched KV: [num_heads, seq_len, head_dim].
    Unbatched {
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    },
    /// Batched KV: [batch, num_heads, seq_len, head_dim].
    Batched {
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    },
}

/// Compressed KV pair with metadata.
#[derive(Debug, Clone)]
pub struct CompressedKV<B: Backend> {
    device: B::Device,
    layout: KVLayout,
    keys: CompressedTensor<B>,
    values: CompressedTensor<B>,
}

impl<B: Backend> CompressedKV<B> {
    /// Access the layout metadata.
    pub fn layout(&self) -> KVLayout {
        self.layout
    }

    /// Device used for reconstruction.
    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

/// KV cache compressor for low-rank and quantized representations.
#[derive(Debug, Clone)]
pub struct KVCacheCompressor<B: Backend> {
    /// Compression method selection.
    pub method: CompressionMethod,
    /// Default quantization bits (INT4/INT8).
    pub quant_bits: u8,
    /// Phantom marker for backend type.
    _marker: PhantomData<B>,
}

impl<B: Backend> KVCacheCompressor<B> {
    /// Create a new KV cache compressor.
    pub fn new(method: CompressionMethod, quant_bits: u8) -> Self {
        Self { method, quant_bits, _marker: PhantomData }
    }

    /// Access the compression method.
    pub fn method(&self) -> &CompressionMethod {
        &self.method
    }

    /// Default quantization bits.
    pub fn quant_bits(&self) -> u8 {
        self.quant_bits
    }

    /// Compress batched KV tensors.
    ///
    /// # Shapes
    /// * `k`, `v`: [batch, num_heads, seq_len, head_dim]
    pub fn compress_kv(
        &self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Result<CompressedKV<B>, &'static str> {
        let [batch, num_heads, seq_len, head_dim] = k.dims();
        if v.dims() != [batch, num_heads, seq_len, head_dim] {
            return Err("keys/values shape mismatch");
        }
        let combined_heads = batch * num_heads;
        let k = k.reshape([combined_heads, seq_len, head_dim]);
        let v = v.reshape([combined_heads, seq_len, head_dim]);
        let layout = KVLayout::Batched {
            batch,
            num_heads,
            seq_len,
            head_dim,
        };
        self.compress_with_layout(k, v, layout)
    }

    /// Compress KV tensors without batch dimension.
    ///
    /// # Shapes
    /// * `k`, `v`: [num_heads, seq_len, head_dim]
    pub fn compress_kv_3d(
        &self,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
    ) -> Result<CompressedKV<B>, &'static str> {
        let [num_heads, seq_len, head_dim] = k.dims();
        if v.dims() != [num_heads, seq_len, head_dim] {
            return Err("keys/values shape mismatch");
        }
        let layout = KVLayout::Unbatched {
            num_heads,
            seq_len,
            head_dim,
        };
        self.compress_with_layout(k, v, layout)
    }

    /// Decompress batched KV tensors.
    ///
    /// # Shapes
    /// * returns: [batch, num_heads, seq_len, head_dim]
    pub fn decompress_kv(
        &self,
        compressed: CompressedKV<B>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>), &'static str> {
        let (k, v, layout) = self.decompress_to_3d(compressed)?;
        match layout {
            KVLayout::Batched {
                batch,
                num_heads,
                seq_len,
                head_dim,
            } => {
                let k = k.reshape([batch, num_heads, seq_len, head_dim]);
                let v = v.reshape([batch, num_heads, seq_len, head_dim]);
                Ok((k, v))
            }
            KVLayout::Unbatched { .. } => Err("expected batched layout"),
        }
    }

    /// Decompress KV tensors without batch dimension.
    ///
    /// # Shapes
    /// * returns: [num_heads, seq_len, head_dim]
    pub fn decompress_kv_3d(
        &self,
        compressed: CompressedKV<B>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), &'static str> {
        let (k, v, layout) = self.decompress_to_3d(compressed)?;
        match layout {
            KVLayout::Unbatched { .. } => Ok((k, v)),
            KVLayout::Batched { .. } => Err("expected unbatched layout"),
        }
    }

    /// Compress a sequence from a paged KV cache.
    pub fn compress_paged_cache(
        &self,
        cache: &PagedKVCache<B>,
        layer: usize,
        seq_id: usize,
    ) -> Result<CompressedKV<B>, &'static str> {
        let (k, v) = cache.get_kv(layer, seq_id)?;
        self.compress_kv_3d(k, v)
    }

    /// Decompress into a paged KV cache by appending tokens.
    pub fn decompress_to_paged_cache(
        &self,
        compressed: CompressedKV<B>,
        cache: &mut PagedKVCache<B>,
        layer: usize,
        seq_id: usize,
    ) -> Result<(), &'static str> {
        let (k, v) = self.decompress_kv_3d(compressed)?;
        cache.append(layer, seq_id, k, v)
    }

    /// Compress a sequence from an MLA compressed cache.
    pub fn compress_mla_cache(
        &self,
        cache: &MlaCompressedKVCache<B>,
        layer: usize,
        seq_id: usize,
    ) -> Result<CompressedKV<B>, &'static str> {
        let (k, v) = cache.get_kv(layer, seq_id)?;
        self.compress_kv_3d(k, v)
    }

    /// Decompress into an MLA compressed cache by appending tokens.
    pub fn decompress_to_mla_cache(
        &self,
        compressed: CompressedKV<B>,
        cache: &mut MlaCompressedKVCache<B>,
        layer: usize,
        seq_id: usize,
    ) -> Result<(), &'static str> {
        let (k, v) = self.decompress_kv_3d(compressed)?;
        cache.append(layer, seq_id, k, v)
    }

    fn compress_with_layout(
        &self,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        layout: KVLayout,
    ) -> Result<CompressedKV<B>, &'static str> {
        if k.dims() != v.dims() {
            return Err("keys/values shape mismatch");
        }
        let device = k.device();
        let keys = self.compress_tensor(k)?;
        let values = self.compress_tensor(v)?;
        Ok(CompressedKV {
            device,
            layout,
            keys,
            values,
        })
    }

    fn decompress_to_3d(
        &self,
        compressed: CompressedKV<B>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>, KVLayout), &'static str> {
        let device = compressed.device.clone();
        let keys = decompress_tensor(compressed.keys, &device)?;
        let values = decompress_tensor(compressed.values, &device)?;
        Ok((keys, values, compressed.layout))
    }

    fn compress_tensor(&self, tensor: Tensor<B, 3>) -> Result<CompressedTensor<B>, &'static str> {
        match self.method {
            CompressionMethod::LowRank { rank } => {
                let low_rank = compress_low_rank(tensor, rank)?;
                Ok(CompressedTensor::LowRank(low_rank))
            }
            CompressionMethod::VectorQuantization { codebook_size } => {
                let bits = effective_vq_bits(self.quant_bits, codebook_size)?;
                let vq = compress_vector_quantization(tensor, codebook_size, bits)?;
                Ok(CompressedTensor::VectorQuantized(vq))
            }
            CompressionMethod::Hybrid { rank, quant_bits } => {
                let bits = if quant_bits == 0 { self.quant_bits } else { quant_bits };
                let hybrid = compress_hybrid(tensor, rank, bits)?;
                Ok(CompressedTensor::Hybrid(hybrid))
            }
        }
    }
}

#[derive(Debug, Clone)]
enum CompressedTensor<B: Backend> {
    LowRank(LowRankTensor<B>),
    VectorQuantized(VectorQuantizedTensor),
    Hybrid(HybridTensor),
}

#[derive(Debug, Clone)]
struct LowRankTensor<B: Backend> {
    projected: Tensor<B, 3>,
    basis_indices: Vec<usize>,
    original_head_dim: usize,
}

#[derive(Debug, Clone)]
struct VectorQuantizedTensor {
    codebook: Vec<f32>,
    codes: QuantizedCodes,
    vector_dim: usize,
    shape: [usize; 3],
    outliers: Vec<OutlierVector>,
}

#[derive(Debug, Clone)]
struct OutlierVector {
    index: usize,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct HybridTensor {
    quantized: QuantizedTensor,
    basis_indices: Vec<usize>,
    original_head_dim: usize,
}

#[derive(Debug, Clone)]
struct QuantizedTensor {
    data: QuantizedData,
    shape: [usize; 3],
    scale: f32,
    bits: u8,
    outliers: Vec<(usize, f32)>,
}

#[derive(Debug, Clone)]
enum QuantizedData {
    Int8(Vec<i8>),
    Int4(Vec<u8>),
}

#[derive(Debug, Clone)]
enum QuantizedCodes {
    Int4 { data: Vec<u8>, len: usize },
    Int8 { data: Vec<u8> },
}

fn compress_low_rank<B: Backend>(
    tensor: Tensor<B, 3>,
    rank: usize,
) -> Result<LowRankTensor<B>, &'static str> {
    let [combined_heads, seq_len, head_dim] = tensor.dims();
    if rank == 0 || head_dim == 0 {
        return Err("invalid rank or head_dim");
    }
    let device = tensor.device();
    let data = tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "low-rank compression expects f32 data")?;
    let tokens = combined_heads * seq_len;
    let mut energies = vec![0.0f32; head_dim];
    for token in 0..tokens {
        let base = token * head_dim;
        for dim in 0..head_dim {
            let value = data[base + dim];
            energies[dim] += value * value;
        }
    }

    let max_rank = rank.min(head_dim);
    let mut ranked: Vec<(usize, f32)> = energies.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_energy: f32 = ranked.iter().map(|(_, energy)| *energy).sum();
    let mut effective_rank = max_rank;
    if total_energy > 0.0 {
        let mut cumulative = 0.0f32;
        effective_rank = 0;
        for (_, energy) in ranked.iter() {
            cumulative += *energy;
            effective_rank += 1;
            if cumulative / total_energy >= ENERGY_FRACTION {
                break;
            }
        }
        effective_rank = effective_rank.min(max_rank).max(1);
    }

    let basis_indices: Vec<usize> = ranked
        .iter()
        .take(effective_rank)
        .map(|(idx, _)| *idx)
        .collect();

    let mut projected = vec![0.0f32; tokens * effective_rank];
    for token in 0..tokens {
        let in_base = token * head_dim;
        let out_base = token * effective_rank;
        for (r, &dim) in basis_indices.iter().enumerate() {
            projected[out_base + r] = data[in_base + dim];
        }
    }

    let projected = Tensor::<B, 3>::from_data(
        TensorData::new(projected, [combined_heads, seq_len, effective_rank]),
        &device,
    );

    Ok(LowRankTensor {
        projected,
        basis_indices,
        original_head_dim: head_dim,
    })
}

fn compress_hybrid<B: Backend>(
    tensor: Tensor<B, 3>,
    rank: usize,
    quant_bits: u8,
) -> Result<HybridTensor, &'static str> {
    let low_rank = compress_low_rank(tensor, rank)?;
    let [combined_heads, seq_len, effective_rank] = low_rank.projected.dims();
    let projected = low_rank.projected.clone();
    let data = projected
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "hybrid compression expects f32 data")?;

    let quantized = quantize_values(
        &data,
        [combined_heads, seq_len, effective_rank],
        quant_bits,
    )?;

    Ok(HybridTensor {
        quantized,
        basis_indices: low_rank.basis_indices,
        original_head_dim: low_rank.original_head_dim,
    })
}

fn compress_vector_quantization<B: Backend>(
    tensor: Tensor<B, 3>,
    codebook_size: usize,
    quant_bits: u8,
) -> Result<VectorQuantizedTensor, &'static str> {
    if codebook_size == 0 {
        return Err("codebook_size must be > 0");
    }
    let [combined_heads, seq_len, head_dim] = tensor.dims();
    let data = tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "vector quantization expects f32 data")?;
    let tokens = combined_heads * seq_len;

    if tokens == 0 {
        return Err("vector quantization expects non-empty tensor");
    }
    if codebook_size > 256 {
        return Err("codebook_size must be <= 256");
    }
    if quant_bits == 4 && codebook_size > 16 {
        return Err("codebook_size must be <= 16 for INT4");
    }

    let mut codebook = vec![0.0f32; codebook_size * head_dim];
    for c in 0..codebook_size {
        if c < tokens {
            let start = c * head_dim;
            let end = start + head_dim;
            codebook[c * head_dim..(c + 1) * head_dim].copy_from_slice(&data[start..end]);
        }
    }

    refine_codebook(&data, &mut codebook, codebook_size, head_dim, tokens);

    let (codes, distances) = assign_codes(&data, &codebook, codebook_size, head_dim, tokens);
    let outliers = detect_vq_outliers(&data, &distances, head_dim);
    let packed = pack_codes(&codes, quant_bits);

    Ok(VectorQuantizedTensor {
        codebook,
        codes: packed,
        vector_dim: head_dim,
        shape: [combined_heads, seq_len, head_dim],
        outliers,
    })
}

fn refine_codebook(
    data: &[f32],
    codebook: &mut [f32],
    codebook_size: usize,
    vector_dim: usize,
    tokens: usize,
) {
    const KMEANS_ITERS: usize = 2;
    for _ in 0..KMEANS_ITERS {
        let mut counts = vec![0usize; codebook_size];
        let mut sums = vec![0.0f32; codebook_size * vector_dim];

        for token in 0..tokens {
            let (idx, _) = nearest_centroid(
                data,
                codebook,
                codebook_size,
                vector_dim,
                token,
            );
            counts[idx] += 1;
            let base = token * vector_dim;
            let sum_base = idx * vector_dim;
            for d in 0..vector_dim {
                sums[sum_base + d] += data[base + d];
            }
        }

        for c in 0..codebook_size {
            if counts[c] > 0 {
                let base = c * vector_dim;
                for d in 0..vector_dim {
                    codebook[base + d] = sums[base + d] / counts[c] as f32;
                }
            }
        }
    }
}

fn assign_codes(
    data: &[f32],
    codebook: &[f32],
    codebook_size: usize,
    vector_dim: usize,
    tokens: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(tokens);
    let mut distances = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let (idx, dist) = nearest_centroid(data, codebook, codebook_size, vector_dim, token);
        codes.push(idx as u8);
        distances.push(dist);
    }
    (codes, distances)
}

fn detect_vq_outliers(
    data: &[f32],
    distances: &[f32],
    vector_dim: usize,
) -> Vec<OutlierVector> {
    if distances.is_empty() {
        return Vec::new();
    }
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    let mut var = 0.0f32;
    for &dist in distances {
        let diff = dist - mean;
        var += diff * diff;
    }
    let std = (var / distances.len() as f32).sqrt();
    let threshold = mean + OUTLIER_STD_FACTOR * std;

    let mut outliers = Vec::new();
    for (token, &dist) in distances.iter().enumerate() {
        if dist > threshold {
            let base = token * vector_dim;
            let values = data[base..base + vector_dim].to_vec();
            outliers.push(OutlierVector { index: token, values });
        }
    }
    outliers
}

fn nearest_centroid(
    data: &[f32],
    codebook: &[f32],
    codebook_size: usize,
    vector_dim: usize,
    token: usize,
) -> (usize, f32) {
    let base = token * vector_dim;
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;
    for c in 0..codebook_size {
        let mut dist = 0.0f32;
        let code_base = c * vector_dim;
        for d in 0..vector_dim {
            let diff = data[base + d] - codebook[code_base + d];
            dist += diff * diff;
        }
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
    }
    (best_idx, best_dist)
}

fn pack_codes(codes: &[u8], bits: u8) -> QuantizedCodes {
    match bits {
        4 => QuantizedCodes::Int4 {
            data: pack_nibbles(codes),
            len: codes.len(),
        },
        _ => QuantizedCodes::Int8 { data: codes.to_vec() },
    }
}

fn unpack_codes(codes: &QuantizedCodes) -> Vec<u8> {
    match codes {
        QuantizedCodes::Int4 { data, len } => unpack_nibbles(data, *len),
        QuantizedCodes::Int8 { data } => data.clone(),
    }
}

fn quantize_values(
    data: &[f32],
    shape: [usize; 3],
    bits: u8,
) -> Result<QuantizedTensor, &'static str> {
    if bits != 4 && bits != 8 {
        return Err("quant_bits must be 4 or 8");
    }
    if data.is_empty() {
        return Err("cannot quantize empty tensor");
    }

    let mut sum_abs = 0.0f32;
    let mut abs_values = Vec::with_capacity(data.len());
    for value in data {
        let abs = value.abs();
        sum_abs += abs;
        abs_values.push(abs);
    }
    let mean = sum_abs / data.len() as f32;
    let mut var = 0.0f32;
    for &abs in &abs_values {
        let diff = abs - mean;
        var += diff * diff;
    }
    let std = (var / data.len() as f32).sqrt();
    abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = abs_values[abs_values.len() / 2];
    let mut deviations: Vec<f32> = abs_values.iter().map(|v| (v - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = deviations[deviations.len() / 2];
    let threshold = if mad > 0.0 {
        median + OUTLIER_MAD_FACTOR * mad
    } else {
        mean + OUTLIER_STD_FACTOR * std
    };

    let mut clipped = Vec::with_capacity(data.len());
    let mut outliers = Vec::new();
    let mut max_abs = 0.0f32;

    for (idx, &value) in data.iter().enumerate() {
        let abs = value.abs();
        if abs > threshold {
            outliers.push((idx, value));
            let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
            let clipped_value = sign * threshold;
            max_abs = max_abs.max(clipped_value.abs());
            clipped.push(clipped_value);
        } else {
            max_abs = max_abs.max(abs);
            clipped.push(value);
        }
    }

    let max_level = if bits == 4 { 7.0 } else { 127.0 };
    let scale = if max_abs > 0.0 { max_abs / max_level } else { 1.0 };

    let quantized = match bits {
        4 => {
            let mut values = Vec::with_capacity(clipped.len());
            for value in &clipped {
                let q = (value / scale).round().clamp(-max_level, max_level) as i8;
                values.push(q);
            }
            QuantizedData::Int4(pack_int4(&values))
        }
        _ => {
            let mut values = Vec::with_capacity(clipped.len());
            for value in &clipped {
                let q = (value / scale).round().clamp(-max_level, max_level) as i8;
                values.push(q);
            }
            QuantizedData::Int8(values)
        }
    };

    Ok(QuantizedTensor {
        data: quantized,
        shape,
        scale,
        bits,
        outliers,
    })
}

fn dequantize_values<B: Backend>(
    quantized: &QuantizedTensor,
    device: &B::Device,
) -> Result<Tensor<B, 3>, &'static str> {
    let num_values = quantized.shape[0] * quantized.shape[1] * quantized.shape[2];
    let mut values: Vec<f32> = match &quantized.data {
        QuantizedData::Int8(data) => data.iter().map(|&q| q as f32 * quantized.scale).collect(),
        QuantizedData::Int4(data) => {
            let unpacked = unpack_int4(data, num_values);
            unpacked
                .into_iter()
                .map(|q| q as f32 * quantized.scale)
                .collect()
        }
    };

    for &(idx, value) in &quantized.outliers {
        if idx < values.len() {
            values[idx] = value;
        }
    }

    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(values, quantized.shape),
        device,
    ))
}

fn decompress_tensor<B: Backend>(
    compressed: CompressedTensor<B>,
    device: &B::Device,
) -> Result<Tensor<B, 3>, &'static str> {
    match compressed {
        CompressedTensor::LowRank(low_rank) => decompress_low_rank(low_rank, device),
        CompressedTensor::VectorQuantized(vq) => decompress_vector_quantized(vq, device),
        CompressedTensor::Hybrid(hybrid) => decompress_hybrid(hybrid, device),
    }
}

fn decompress_low_rank<B: Backend>(
    low_rank: LowRankTensor<B>,
    device: &B::Device,
) -> Result<Tensor<B, 3>, &'static str> {
    let [combined_heads, seq_len, rank] = low_rank.projected.dims();
    if rank == 0 {
        return Err("low-rank projection has rank 0");
    }
    let data = low_rank
        .projected
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "low-rank decompression expects f32 data")?;
    let tokens = combined_heads * seq_len;
    let head_dim = low_rank.original_head_dim;
    let mut full = vec![0.0f32; tokens * head_dim];

    for token in 0..tokens {
        let in_base = token * rank;
        let out_base = token * head_dim;
        for (r, &dim) in low_rank.basis_indices.iter().enumerate() {
            if dim < head_dim {
                full[out_base + dim] = data[in_base + r];
            }
        }
    }

    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(full, [combined_heads, seq_len, head_dim]),
        device,
    ))
}

fn decompress_hybrid<B: Backend>(
    hybrid: HybridTensor,
    device: &B::Device,
) -> Result<Tensor<B, 3>, &'static str> {
    let projected = dequantize_values::<B>(&hybrid.quantized, device)?;
    let low_rank = LowRankTensor {
        projected,
        basis_indices: hybrid.basis_indices,
        original_head_dim: hybrid.original_head_dim,
    };
    decompress_low_rank(low_rank, device)
}

fn decompress_vector_quantized<B: Backend>(
    vq: VectorQuantizedTensor,
    device: &B::Device,
) -> Result<Tensor<B, 3>, &'static str> {
    let tokens = vq.shape[0] * vq.shape[1];
    let vector_dim = vq.vector_dim;
    let codes = unpack_codes(&vq.codes);
    if codes.len() != tokens {
        return Err("vector quantization code length mismatch");
    }
    let mut data = vec![0.0f32; tokens * vector_dim];

    for token in 0..tokens {
        let code = codes[token] as usize;
        let base = token * vector_dim;
        let code_base = code * vector_dim;
        for d in 0..vector_dim {
            data[base + d] = vq.codebook[code_base + d];
        }
    }

    for outlier in &vq.outliers {
        let base = outlier.index * vector_dim;
        if base + vector_dim <= data.len() {
            data[base..base + vector_dim].copy_from_slice(&outlier.values);
        }
    }

    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(data, vq.shape),
        device,
    ))
}

fn effective_vq_bits(bits: u8, codebook_size: usize) -> Result<u8, &'static str> {
    match bits {
        4 => {
            if codebook_size > 16 {
                Err("codebook_size must be <= 16 for INT4")
            } else {
                Ok(4)
            }
        }
        8 => {
            if codebook_size > 256 {
                Err("codebook_size must be <= 256 for INT8")
            } else {
                Ok(8)
            }
        }
        _ => {
            if codebook_size <= 16 {
                Ok(4)
            } else if codebook_size <= 256 {
                Ok(8)
            } else {
                Err("codebook_size must be <= 256")
            }
        }
    }
}

fn pack_nibbles(values: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);
    let mut iter = values.iter();
    loop {
        let low = match iter.next() {
            Some(v) => v & 0x0F,
            None => break,
        };
        let high = match iter.next() {
            Some(v) => (v & 0x0F) << 4,
            None => 0,
        };
        packed.push(low | high);
    }
    packed
}

fn unpack_nibbles(values: &[u8], len: usize) -> Vec<u8> {
    let mut unpacked = Vec::with_capacity(len);
    for &byte in values {
        if unpacked.len() < len {
            unpacked.push(byte & 0x0F);
        }
        if unpacked.len() < len {
            unpacked.push((byte >> 4) & 0x0F);
        }
    }
    unpacked
}

fn pack_int4(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);
    let mut iter = values.iter();
    loop {
        let low = match iter.next() {
            Some(v) => (*v as i16 + 8).clamp(0, 15) as u8,
            None => break,
        };
        let high = match iter.next() {
            Some(v) => ((*v as i16 + 8).clamp(0, 15) as u8) << 4,
            None => 0,
        };
        packed.push(low | high);
    }
    packed
}

fn unpack_int4(values: &[u8], len: usize) -> Vec<i8> {
    let mut unpacked = Vec::with_capacity(len);
    for &byte in values {
        if unpacked.len() < len {
            unpacked.push(((byte & 0x0F) as i8) - 8);
        }
        if unpacked.len() < len {
            unpacked.push(((byte >> 4) as i8) - 8);
        }
    }
    unpacked
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::NdArray;

    #[test]
    fn test_low_rank_roundtrip_preserves_top_dims() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let num_heads = 2;
        let seq_len = 2;
        let head_dim = 4;
        let mut data = Vec::new();
        for _ in 0..(num_heads * seq_len) {
            data.extend_from_slice(&[1.0, 0.1, 1.0, 0.1]);
        }

        let k = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(data.clone(), [num_heads, seq_len, head_dim]),
            &device,
        );
        let v = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(data, [num_heads, seq_len, head_dim]),
            &device,
        );

        let compressor =
            KVCacheCompressor::<NdArray<f32>>::new(CompressionMethod::LowRank { rank: 3 }, 8);
        let compressed = compressor.compress_kv_3d(k, v).expect("compress");
        let (k_full, _v_full) = compressor.decompress_kv_3d(compressed).expect("decompress");

        let k_data = k_full.into_data().into_vec::<f32>().expect("data");
        for token in 0..(num_heads * seq_len) {
            let base = token * head_dim;
            assert!((k_data[base + 1]).abs() < 1e-3);
            assert!((k_data[base + 3]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_hybrid_quantization_outlier_preserved() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let num_heads = 1;
        let seq_len = 4;
        let head_dim = 1;
        let data = vec![0.1, 0.2, 0.15, 10.0];

        let k = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(data.clone(), [num_heads, seq_len, head_dim]),
            &device,
        );
        let v = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(data, [num_heads, seq_len, head_dim]),
            &device,
        );

        let compressor = KVCacheCompressor::<NdArray<f32>>::new(
            CompressionMethod::Hybrid {
                rank: 1,
                quant_bits: 4,
            },
            4,
        );
        let compressed = compressor.compress_kv_3d(k, v).expect("compress");
        let (k_full, _) = compressor.decompress_kv_3d(compressed).expect("decompress");
        let k_data = k_full.into_data().into_vec::<f32>().expect("data");

        assert!((k_data[3] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_vector_quantization_roundtrip() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let num_heads = 1;
        let seq_len = 2;
        let head_dim = 2;
        let data = vec![1.0, 0.0, -1.0, 0.0];
        let original_data = data.clone();

        let k = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(data.clone(), [num_heads, seq_len, head_dim]),
            &device,
        );
        let v = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(data, [num_heads, seq_len, head_dim]),
            &device,
        );

        let compressor = KVCacheCompressor::<NdArray<f32>>::new(
            CompressionMethod::VectorQuantization { codebook_size: 2 },
            8,
        );
        let compressed = compressor.compress_kv_3d(k, v).expect("compress");
        let (k_full, _) = compressor.decompress_kv_3d(compressed).expect("decompress");
        let k_data = k_full.into_data().into_vec::<f32>().expect("data");

        for (orig, round) in original_data.iter().zip(k_data.iter()) {
            assert!((orig - round).abs() < 1e-4);
        }
    }

    #[test]
    fn test_paged_cache_compatibility() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(4, 1, 1, 2, &device);
        let seq_id = cache.allocate_sequence();
        let keys = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 3, 2]),
            &device,
        );
        let values = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![0.6, 0.5, 0.4, 0.3, 0.2, 0.1], [1, 3, 2]),
            &device,
        );
        cache.append(0, seq_id, keys, values).expect("append");

        let compressor =
            KVCacheCompressor::<NdArray<f32>>::new(CompressionMethod::LowRank { rank: 2 }, 8);
        let compressed = compressor
            .compress_paged_cache(&cache, 0, seq_id)
            .expect("compress paged");

        let seq_id2 = cache.allocate_sequence();
        compressor
            .decompress_to_paged_cache(compressed, &mut cache, 0, seq_id2)
            .expect("decompress paged");
        assert_eq!(cache.seq_len(0, seq_id2).expect("seq len"), 3);
    }

    fn identity_matrix(
        dim: usize,
        device: &<NdArray<f32> as Backend>::Device,
    ) -> Tensor<NdArray<f32>, 2> {
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Tensor::from_data(TensorData::new(data, [dim, dim]), device)
    }

    fn zero_matrix(
        rows: usize,
        cols: usize,
        device: &<NdArray<f32> as Backend>::Device,
    ) -> Tensor<NdArray<f32>, 2> {
        Tensor::from_data(TensorData::new(vec![0.0f32; rows * cols], [rows, cols]), device)
    }

    #[test]
    fn test_mla_cache_compatibility() {
        use crate::ops::mla::MultiHeadLatentAttention;

        let device = <NdArray<f32> as Backend>::Device::default();
        let head_dim = 2;
        let latent_dim = 2;
        let down = identity_matrix(head_dim, &device);
        let up = identity_matrix(head_dim, &device);
        let rope = zero_matrix(latent_dim, head_dim, &device);

        let mla = MultiHeadLatentAttention::new(1, latent_dim, down, up, rope);
        let mut cache = MlaCompressedKVCache::new(4, 1, 1, mla, &device);
        let seq_id = cache.allocate_sequence();

        let keys = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 3, 2]),
            &device,
        );
        let values = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![0.6, 0.5, 0.4, 0.3, 0.2, 0.1], [1, 3, 2]),
            &device,
        );
        cache.append(0, seq_id, keys, values).expect("append");

        let compressor =
            KVCacheCompressor::<NdArray<f32>>::new(CompressionMethod::LowRank { rank: 2 }, 8);
        let compressed = compressor
            .compress_mla_cache(&cache, 0, seq_id)
            .expect("compress mla");

        let seq_id2 = cache.allocate_sequence();
        compressor
            .decompress_to_mla_cache(compressed, &mut cache, 0, seq_id2)
            .expect("decompress mla");
        assert_eq!(cache.seq_len(0, seq_id2).expect("seq len"), 3);
    }
}
