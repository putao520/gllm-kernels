//! Multi-Head Latent Attention (MLA) compression utilities.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::ops::paged_attention::PagedKVCache;

/// Multi-head latent attention module for KV compression.
#[derive(Debug, Clone)]
pub struct MultiHeadLatentAttention<B: Backend> {
    /// Compression ratio (typical: 8-16).
    compression_ratio: usize,
    /// Latent dimension after compression.
    latent_dim: usize,
    /// Down projection matrix W_dkv: [head_dim, latent_dim].
    down_proj: Tensor<B, 2>,
    /// Up projection matrix W_ukv: [latent_dim, head_dim].
    up_proj: Tensor<B, 2>,
    /// Decoupled RoPE projection: [latent_dim, head_dim].
    rope_key: Tensor<B, 2>,
}

impl<B: Backend> MultiHeadLatentAttention<B> {
    /// Create a new MLA module with explicit projections.
    pub fn new(
        compression_ratio: usize,
        latent_dim: usize,
        down_proj: Tensor<B, 2>,
        up_proj: Tensor<B, 2>,
        rope_key: Tensor<B, 2>,
    ) -> Self {
        Self {
            compression_ratio,
            latent_dim,
            down_proj,
            up_proj,
            rope_key,
        }
    }

    /// Compression ratio configured for this module.
    pub fn compression_ratio(&self) -> usize {
        self.compression_ratio
    }

    /// Latent dimension configured for this module.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Compress KV tensors to latent space.
    ///
    /// # Shapes
    /// * `k`, `v`: [batch, num_heads, seq_len, head_dim]
    /// * returns: [batch, num_heads, seq_len, latent_dim]
    pub fn compress_kv(
        &self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>), &'static str> {
        let [batch, num_heads, seq_len, head_dim] = k.dims();
        if v.dims() != [batch, num_heads, seq_len, head_dim] {
            return Err("keys/values shape mismatch");
        }
        self.validate_projections(head_dim)?;

        let tokens = batch * num_heads * seq_len;
        let k_flat = k.reshape([tokens, head_dim]);
        let v_flat = v.reshape([tokens, head_dim]);

        let k_latent = k_flat.matmul(self.down_proj.clone());
        let v_latent = v_flat.matmul(self.down_proj.clone());

        let k_latent = k_latent.reshape([batch, num_heads, seq_len, self.latent_dim]);
        let v_latent = v_latent.reshape([batch, num_heads, seq_len, self.latent_dim]);

        Ok((k_latent, v_latent))
    }

    /// Decompress KV tensors from latent space.
    ///
    /// # Shapes
    /// * `k_latent`, `v_latent`: [batch, num_heads, seq_len, latent_dim]
    /// * returns: [batch, num_heads, seq_len, head_dim]
    pub fn decompress_kv(
        &self,
        k_latent: Tensor<B, 4>,
        v_latent: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>), &'static str> {
        let [batch, num_heads, seq_len, latent_dim] = k_latent.dims();
        if v_latent.dims() != [batch, num_heads, seq_len, latent_dim] {
            return Err("latent keys/values shape mismatch");
        }
        if latent_dim != self.latent_dim {
            return Err("latent dimension mismatch");
        }

        let head_dim = self.up_proj.dims()[1];
        self.validate_projections(head_dim)?;

        let tokens = batch * num_heads * seq_len;
        let k_flat = k_latent.reshape([tokens, latent_dim]);
        let v_flat = v_latent.reshape([tokens, latent_dim]);

        let mut k_full = k_flat.clone().matmul(self.up_proj.clone());
        let v_full = v_flat.matmul(self.up_proj.clone());

        if self.rope_key.dims() == [latent_dim, head_dim] {
            let rope = k_flat.clone().matmul(self.rope_key.clone());
            k_full = k_full + rope;
        }

        let k_full = k_full.reshape([batch, num_heads, seq_len, head_dim]);
        let v_full = v_full.reshape([batch, num_heads, seq_len, head_dim]);

        Ok((k_full, v_full))
    }

    /// Compress KV tensors with 3D shapes.
    ///
    /// # Shapes
    /// * `k`, `v`: [num_heads, seq_len, head_dim]
    /// * returns: [num_heads, seq_len, latent_dim]
    pub fn compress_kv_3d(
        &self,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), &'static str> {
        let [num_heads, seq_len, head_dim] = k.dims();
        if v.dims() != [num_heads, seq_len, head_dim] {
            return Err("keys/values shape mismatch");
        }
        self.validate_projections(head_dim)?;

        let k = k.reshape([1, num_heads, seq_len, head_dim]);
        let v = v.reshape([1, num_heads, seq_len, head_dim]);
        let (k_latent, v_latent) = self.compress_kv(k, v)?;

        Ok((
            k_latent.reshape([num_heads, seq_len, self.latent_dim]),
            v_latent.reshape([num_heads, seq_len, self.latent_dim]),
        ))
    }

    /// Decompress KV tensors with 3D shapes.
    ///
    /// # Shapes
    /// * `k_latent`, `v_latent`: [num_heads, seq_len, latent_dim]
    /// * returns: [num_heads, seq_len, head_dim]
    pub fn decompress_kv_3d(
        &self,
        k_latent: Tensor<B, 3>,
        v_latent: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), &'static str> {
        let [num_heads, seq_len, latent_dim] = k_latent.dims();
        if v_latent.dims() != [num_heads, seq_len, latent_dim] {
            return Err("latent keys/values shape mismatch");
        }
        if latent_dim != self.latent_dim {
            return Err("latent dimension mismatch");
        }

        let k = k_latent.reshape([1, num_heads, seq_len, latent_dim]);
        let v = v_latent.reshape([1, num_heads, seq_len, latent_dim]);
        let (k_full, v_full) = self.decompress_kv(k, v)?;
        let head_dim = k_full.dims()[3];
        let value_dim = v_full.dims()[3];

        Ok((
            k_full.reshape([num_heads, seq_len, head_dim]),
            v_full.reshape([num_heads, seq_len, value_dim]),
        ))
    }

    fn validate_projections(&self, head_dim: usize) -> Result<(), &'static str> {
        let down_dims = self.down_proj.dims();
        if down_dims != [head_dim, self.latent_dim] {
            return Err("down projection shape mismatch");
        }
        let up_dims = self.up_proj.dims();
        if up_dims != [self.latent_dim, head_dim] {
            return Err("up projection shape mismatch");
        }
        let rope_dims = self.rope_key.dims();
        if rope_dims != [self.latent_dim, head_dim] {
            return Err("rope key shape mismatch");
        }
        if self.compression_ratio == 0 || self.latent_dim == 0 {
            return Err("invalid compression configuration");
        }
        Ok(())
    }
}

/// Compressed KV cache compatible with paged attention APIs.
#[derive(Debug, Clone)]
pub struct CompressedKVCache<B: Backend> {
    inner: PagedKVCache<B>,
    mla: MultiHeadLatentAttention<B>,
}

impl<B: Backend> CompressedKVCache<B> {
    /// Create a compressed KV cache for a given MLA configuration.
    pub fn new(
        max_blocks: usize,
        num_layers: usize,
        num_heads: usize,
        mla: MultiHeadLatentAttention<B>,
        device: &B::Device,
    ) -> Self {
        let inner = PagedKVCache::new(max_blocks, num_layers, num_heads, mla.latent_dim(), device);
        Self { inner, mla }
    }

    /// Allocate a new sequence and return its id.
    pub fn allocate_sequence(&mut self) -> usize {
        self.inner.allocate_sequence()
    }

    /// Append uncompressed KV tensors (3D) into the compressed cache.
    pub fn append(
        &mut self,
        layer: usize,
        seq_id: usize,
        keys: Tensor<B, 3>,
        values: Tensor<B, 3>,
    ) -> Result<(), &'static str> {
        let (k_latent, v_latent) = self.mla.compress_kv_3d(keys, values)?;
        self.inner.append(layer, seq_id, k_latent, v_latent)
    }

    /// Append uncompressed KV tensors (4D, batch=1) into the compressed cache.
    pub fn append_batched(
        &mut self,
        layer: usize,
        seq_id: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> Result<(), &'static str> {
        let [batch, num_heads, seq_len, head_dim] = keys.dims();
        if batch != 1 {
            return Err("compressed cache expects batch=1");
        }
        if values.dims() != [batch, num_heads, seq_len, head_dim] {
            return Err("keys/values shape mismatch");
        }
        let keys = keys.reshape([num_heads, seq_len, head_dim]);
        let values = values.reshape([num_heads, seq_len, head_dim]);
        self.append(layer, seq_id, keys, values)
    }

    /// Append pre-compressed KV tensors directly.
    pub fn append_compressed(
        &mut self,
        layer: usize,
        seq_id: usize,
        keys_latent: Tensor<B, 3>,
        values_latent: Tensor<B, 3>,
    ) -> Result<(), &'static str> {
        self.inner.append(layer, seq_id, keys_latent, values_latent)
    }

    /// Get decompressed KV tensors for a layer/sequence.
    pub fn get_kv(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), &'static str> {
        let (k_latent, v_latent) = self.inner.get_kv(layer, seq_id)?;
        self.mla.decompress_kv_3d(k_latent, v_latent)
    }

    /// Get compressed KV tensors for a layer/sequence.
    pub fn get_compressed_kv(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), &'static str> {
        self.inner.get_kv(layer, seq_id)
    }

    /// Iterate over decompressed KV blocks for a sequence.
    pub fn iter_kv_blocks(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<Vec<(Tensor<B, 3>, Tensor<B, 3>)>, &'static str> {
        let kv_iter = self.inner.iter_kv_blocks(layer, seq_id)?;
        let mut blocks = Vec::new();
        for block in kv_iter {
            let [num_heads, _, latent_dim] = block.keys.dims();
            let k_latent = block
                .keys
                .clone()
                .slice([0..num_heads, 0..block.num_tokens, 0..latent_dim]);
            let v_latent = block
                .values
                .clone()
                .slice([0..num_heads, 0..block.num_tokens, 0..latent_dim]);
            let (k_full, v_full) = self.mla.decompress_kv_3d(k_latent, v_latent)?;
            blocks.push((k_full, v_full));
        }
        Ok(blocks)
    }

    /// Iterate over compressed KV blocks for a sequence.
    pub fn iter_compressed_blocks(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<Vec<(Tensor<B, 3>, Tensor<B, 3>)>, &'static str> {
        let kv_iter = self.inner.iter_kv_blocks(layer, seq_id)?;
        let mut blocks = Vec::new();
        for block in kv_iter {
            let [num_heads, _, latent_dim] = block.keys.dims();
            let k_latent = block
                .keys
                .clone()
                .slice([0..num_heads, 0..block.num_tokens, 0..latent_dim]);
            let v_latent = block
                .values
                .clone()
                .slice([0..num_heads, 0..block.num_tokens, 0..latent_dim]);
            blocks.push((k_latent, v_latent));
        }
        Ok(blocks)
    }

    /// Get sequence length for a layer/sequence.
    pub fn seq_len(&self, layer: usize, seq_id: usize) -> Result<usize, &'static str> {
        self.inner.seq_len(layer, seq_id)
    }

    /// Get the number of free blocks in the cache.
    pub fn num_free_blocks(&self) -> usize {
        self.inner.num_free_blocks()
    }

    /// Get block manager configuration.
    pub fn num_heads(&self) -> usize {
        self.inner.num_heads()
    }

    pub fn latent_dim(&self) -> usize {
        self.inner.head_dim()
    }

    pub fn device(&self) -> &B::Device {
        self.inner.device()
    }

    /// Release all blocks associated with a sequence id.
    pub fn free_sequence(&mut self, seq_id: usize) -> Result<(), &'static str> {
        self.inner.free_sequence(seq_id)
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn identity_matrix(dim: usize, device: &<TestBackend as Backend>::Device) -> Tensor<TestBackend, 2> {
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Tensor::from_data(TensorData::new(data, [dim, dim]), device)
    }

    fn zero_matrix(rows: usize, cols: usize, device: &<TestBackend as Backend>::Device) -> Tensor<TestBackend, 2> {
        let data = vec![0.0f32; rows * cols];
        Tensor::from_data(TensorData::new(data, [rows, cols]), device)
    }

    #[test]
    fn test_mla_compress_decompress_roundtrip_identity() {
        let device = <TestBackend as Backend>::Device::default();
        let head_dim = 4;
        let latent_dim = 4;
        let down = identity_matrix(head_dim, &device);
        let up = identity_matrix(head_dim, &device);
        let rope = zero_matrix(latent_dim, head_dim, &device);

        let mla = MultiHeadLatentAttention::new(1, latent_dim, down, up, rope);

        let q = Tensor::<TestBackend, 4>::random([1, 2, 4, head_dim], Distribution::Normal(0.0, 0.5), &device);
        let v = Tensor::<TestBackend, 4>::random([1, 2, 4, head_dim], Distribution::Normal(0.0, 0.5), &device);

        let (k_latent, v_latent) = mla.compress_kv(q.clone(), v.clone()).expect("compress");
        let (k_full, v_full) = mla.decompress_kv(k_latent, v_latent).expect("decompress");

        let k_data = q.into_data().into_vec::<f32>().expect("k data");
        let k_roundtrip = k_full.into_data().into_vec::<f32>().expect("k roundtrip");
        for (idx, (orig, round)) in k_data.iter().zip(k_roundtrip.iter()).enumerate() {
            let diff = (orig - round).abs();
            assert!(diff < 1e-4, "k mismatch at {}: {} vs {}", idx, orig, round);
        }

        let v_data = v.into_data().into_vec::<f32>().expect("v data");
        let v_roundtrip = v_full.into_data().into_vec::<f32>().expect("v roundtrip");
        for (idx, (orig, round)) in v_data.iter().zip(v_roundtrip.iter()).enumerate() {
            let diff = (orig - round).abs();
            assert!(diff < 1e-4, "v mismatch at {}: {} vs {}", idx, orig, round);
        }
    }

    #[test]
    fn test_compressed_kv_cache_roundtrip() {
        let device = <TestBackend as Backend>::Device::default();
        let head_dim = 4;
        let latent_dim = 4;
        let down = identity_matrix(head_dim, &device);
        let up = identity_matrix(head_dim, &device);
        let rope = zero_matrix(latent_dim, head_dim, &device);

        let mla = MultiHeadLatentAttention::new(1, latent_dim, down, up, rope);
        let mut cache = CompressedKVCache::new(4, 1, 2, mla, &device);

        let seq_id = cache.allocate_sequence();
        let keys = Tensor::<TestBackend, 3>::random(
            [2, 5, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );
        let values = Tensor::<TestBackend, 3>::random(
            [2, 5, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );

        cache.append(0, seq_id, keys.clone(), values.clone()).expect("append");

        let (k_full, v_full) = cache.get_kv(0, seq_id).expect("get kv");
        assert_eq!(k_full.dims(), [2, 5, head_dim]);
        assert_eq!(v_full.dims(), [2, 5, head_dim]);

        let k_data = keys.into_data().into_vec::<f32>().expect("keys data");
        let k_round = k_full.into_data().into_vec::<f32>().expect("keys roundtrip");
        for (idx, (orig, round)) in k_data.iter().zip(k_round.iter()).enumerate() {
            let diff = (orig - round).abs();
            assert!(diff < 1e-4, "k mismatch at {}: {} vs {}", idx, orig, round);
        }
    }
}
