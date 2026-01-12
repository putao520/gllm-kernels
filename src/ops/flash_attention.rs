//! Hierarchical FlashAttention with direct paged KV access.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::ops::stable_accumulator::AccumulatorConfig;

/// Configuration for deterministic computation.
#[derive(Clone, Debug)]
pub struct DeterministicConfig {
    /// Enable deterministic mode.
    pub enabled: bool,
    /// Force fixed tile processing order for reproducibility.
    pub fixed_tile_order: bool,
    /// Fixed random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for DeterministicConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fixed_tile_order: false,
            seed: None,
        }
    }
}

impl DeterministicConfig {
    /// Create a configuration for maximum reproducibility.
    pub fn strict() -> Self {
        Self {
            enabled: true,
            fixed_tile_order: true,
            seed: Some(42),
        }
    }

    /// Create a configuration that allows some non-determinism for speed.
    pub fn relaxed() -> Self {
        Self {
            enabled: false,
            fixed_tile_order: false,
            seed: None,
        }
    }

    /// Create a configuration for 2M context (strict by default).
    pub fn ultra_long_context() -> Self {
        Self::strict()
    }

    /// Check if any deterministic guarantees are enabled.
    pub fn is_deterministic(&self) -> bool {
        self.enabled || self.fixed_tile_order || self.seed.is_some()
    }
}

/// Strict ordering iterator for deterministic processing.
pub struct StrictOrderIterator<I> {
    inner: I,
    index: usize,
}

impl<I: Iterator> StrictOrderIterator<I> {
    pub fn new(iter: I) -> Self {
        Self { inner: iter, index: 0 }
    }

    /// Get the current index (for verification).
    pub fn current_index(&self) -> usize {
        self.index
    }
}

impl<I: Iterator> Iterator for StrictOrderIterator<I> {
    type Item = (usize, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        let index = self.index;
        self.index += 1;

        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        Some((index, item))
    }
}

/// Extension trait for creating strict order iterators.
pub trait StrictOrderExt: Iterator + Sized {
    fn strict_order(self) -> StrictOrderIterator<Self> {
        StrictOrderIterator::new(self)
    }
}

impl<I: Iterator> StrictOrderExt for I {}

/// Configuration for hierarchical FlashAttention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Query block size for tiling.
    pub block_q: usize,
    /// KV block size for tiling (should match PagedKVCache block size).
    pub block_kv: usize,
    /// Accumulator configuration for numerical stability.
    pub accumulator: AccumulatorConfig,
    /// Determinism configuration.
    pub determinism: DeterministicConfig,
    /// Use log-space accumulation (more stable but slightly slower).
    pub use_log_space: bool,
    /// Maximum sequence length to expect (for pre-allocation).
    pub max_seq_len: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_q: 64,
            block_kv: 16,
            accumulator: AccumulatorConfig::max_precision(),
            determinism: DeterministicConfig::strict(),
            use_log_space: true,
            max_seq_len: 2_000_000,
        }
    }
}

impl FlashAttentionConfig {
    /// Configuration optimized for 2M context.
    pub fn ultra_long_context() -> Self {
        Self {
            block_q: 64,
            block_kv: 16,
            accumulator: AccumulatorConfig::max_precision(),
            determinism: DeterministicConfig::ultra_long_context(),
            use_log_space: true,
            max_seq_len: 2_000_000,
        }
    }

    /// Configuration for shorter contexts (< 100K).
    pub fn short_context() -> Self {
        Self {
            block_q: 128,
            block_kv: 64,
            accumulator: AccumulatorConfig::short_context(),
            determinism: DeterministicConfig::relaxed(),
            use_log_space: false,
            max_seq_len: 100_000,
        }
    }
}

/// Backward-compatible alias.
pub type HierarchicalFlashConfig = FlashAttentionConfig;

/// Trait for fused paged attention computation.
pub trait FusedPagedAttention<B: Backend> {
    /// Compute attention with direct access to paged KV blocks.
    fn forward_fused<'a, I>(
        &self,
        q: Tensor<B, 4>,
        kv_blocks: I,
        config: &FlashAttentionConfig,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4>
    where
        I: Iterator<Item = (Tensor<B, 3>, Tensor<B, 3>)> + 'a;
}

/// Hierarchical FlashAttention implementation.
#[derive(Debug, Clone)]
pub struct HierarchicalFlashAttention {
    config: FlashAttentionConfig,
}

impl HierarchicalFlashAttention {
    /// Create a new HierarchicalFlashAttention with the given configuration.
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(FlashAttentionConfig::default())
    }

    /// Create optimized for 2M context.
    pub fn ultra_long_context() -> Self {
        Self::new(FlashAttentionConfig::ultra_long_context())
    }

    /// Get the configuration.
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Standard FlashAttention forward pass (non-fused, for reference/testing).
    pub fn forward<B: Backend>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let device = q.device();
        let [batch_size, num_heads, query_len, head_dim] = q.dims();
        let key_len = k.dims()[2];

        if query_len == 0 || key_len == 0 {
            return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
        }

        let block_q = self.config.block_q.max(1);
        let block_kv = self.config.block_kv.max(1);
        let inv_scale = 1.0 / (head_dim as f32).sqrt();

        let num_q_blocks = (query_len + block_q - 1) / block_q;
        let mut outputs = Vec::with_capacity(num_q_blocks);
        let fixed_tile_order = self.config.determinism.fixed_tile_order;

        let process_q_block = |q_start: usize, outputs: &mut Vec<Tensor<B, 4>>| {
            let q_end = (q_start + block_q).min(query_len);
            let q_block_len = q_end - q_start;
            let q_block = q.clone().slice([
                0..batch_size,
                0..num_heads,
                q_start..q_end,
                0..head_dim,
            ]);

            let mut m_i = Tensor::<B, 4>::full(
                [batch_size, num_heads, q_block_len, 1],
                f32::NEG_INFINITY,
                &device,
            );
            let mut l_i = Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
            let mut o_i = Tensor::<B, 4>::zeros(
                [batch_size, num_heads, q_block_len, head_dim],
                &device,
            );

            if fixed_tile_order {
                let mut kv_start = 0usize;
                while kv_start < key_len {
                    let kv_end = (kv_start + block_kv).min(key_len);
                    let kv_block_len = kv_end - kv_start;

                    let k_block = k.clone().slice([
                        0..batch_size,
                        0..num_heads,
                        kv_start..kv_end,
                        0..head_dim,
                    ]);
                    let v_block = v.clone().slice([
                        0..batch_size,
                        0..num_heads,
                        kv_start..kv_end,
                        0..head_dim,
                    ]);

                    let mut scores = q_block.clone().matmul(k_block.transpose()) * inv_scale;

                    if causal {
                        let mask = self.build_causal_mask::<B>(
                            &device,
                            q_block_len,
                            kv_block_len,
                            q_start,
                            kv_start,
                            position_offset,
                        );
                        scores = scores + mask;
                    }

                    let m_ij = scores.clone().max_dim(3);
                    let m_new = m_i.clone().max_pair(m_ij);

                    let m_scale = (m_i - m_new.clone()).exp();
                    let p_ij = (scores - m_new.clone()).exp();
                    let p_sum = p_ij.clone().sum_dim(3);

                    l_i = m_scale.clone() * l_i + p_sum;
                    o_i = m_scale * o_i + p_ij.matmul(v_block);
                    m_i = m_new;

                    kv_start = kv_end;
                }
            } else {
                for kv_start in (0..key_len).step_by(block_kv) {
                    let kv_end = (kv_start + block_kv).min(key_len);
                    let kv_block_len = kv_end - kv_start;

                    let k_block = k.clone().slice([
                        0..batch_size,
                        0..num_heads,
                        kv_start..kv_end,
                        0..head_dim,
                    ]);
                    let v_block = v.clone().slice([
                        0..batch_size,
                        0..num_heads,
                        kv_start..kv_end,
                        0..head_dim,
                    ]);

                    let mut scores = q_block.clone().matmul(k_block.transpose()) * inv_scale;

                    if causal {
                        let mask = self.build_causal_mask::<B>(
                            &device,
                            q_block_len,
                            kv_block_len,
                            q_start,
                            kv_start,
                            position_offset,
                        );
                        scores = scores + mask;
                    }

                    let m_ij = scores.clone().max_dim(3);
                    let m_new = m_i.clone().max_pair(m_ij);

                    let m_scale = (m_i - m_new.clone()).exp();
                    let p_ij = (scores - m_new.clone()).exp();
                    let p_sum = p_ij.clone().sum_dim(3);

                    l_i = m_scale.clone() * l_i + p_sum;
                    o_i = m_scale * o_i + p_ij.matmul(v_block);
                    m_i = m_new;
                }
            }

            outputs.push(o_i / l_i);
        };

        if fixed_tile_order {
            let mut q_start = 0usize;
            while q_start < query_len {
                process_q_block(q_start, &mut outputs);
                q_start += block_q;
            }
        } else {
            for q_start in (0..query_len).step_by(block_q) {
                process_q_block(q_start, &mut outputs);
            }
        }

        let output = Tensor::cat(outputs, 2);
        B::sync(&output.device());
        output
    }

    /// Fused forward pass that directly iterates over KV blocks.
    pub fn forward_fused_iter<'a, B, I>(
        &self,
        q: Tensor<B, 4>,
        kv_blocks: I,
        causal: bool,
        position_offset: usize,
        total_kv_len: usize,
    ) -> Tensor<B, 4>
    where
        B: Backend,
        I: Iterator<Item = (Tensor<B, 3>, Tensor<B, 3>)> + 'a,
    {
        let device = q.device();
        let [batch_size, num_heads, query_len, head_dim] = q.dims();

        if query_len == 0 || total_kv_len == 0 {
            return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
        }

        let block_q = self.config.block_q.max(1);
        let inv_scale = 1.0 / (head_dim as f32).sqrt();

        let kv_blocks: Vec<_> = if self.config.determinism.fixed_tile_order {
            kv_blocks.strict_order().map(|(_, kv)| kv).collect()
        } else {
            kv_blocks.collect()
        };

        let num_q_blocks = (query_len + block_q - 1) / block_q;
        let mut outputs = Vec::with_capacity(num_q_blocks);
        let mut q_start = 0usize;

        while q_start < query_len {
            let q_end = (q_start + block_q).min(query_len);
            let q_block = q.clone().slice([
                0..batch_size,
                0..num_heads,
                q_start..q_end,
                0..head_dim,
            ]);

            let output = if self.config.use_log_space {
                self.process_q_block_log_space(
                    q_block,
                    &kv_blocks,
                    causal,
                    q_start,
                    position_offset,
                    inv_scale,
                )
            } else {
                self.process_q_block_standard(
                    q_block,
                    &kv_blocks,
                    causal,
                    q_start,
                    position_offset,
                    inv_scale,
                )
            };

            outputs.push(output);
            q_start = q_end;
        }

        Tensor::cat(outputs, 2)
    }

    fn process_q_block_standard<B: Backend>(
        &self,
        q_block: Tensor<B, 4>,
        kv_blocks: &[(Tensor<B, 3>, Tensor<B, 3>)],
        causal: bool,
        q_start: usize,
        position_offset: usize,
        inv_scale: f32,
    ) -> Tensor<B, 4> {
        let device = q_block.device();
        let [batch_size, num_heads, q_block_len, head_dim] = q_block.dims();

        let mut m_i = Tensor::<B, 4>::full(
            [batch_size, num_heads, q_block_len, 1],
            f32::NEG_INFINITY,
            &device,
        );
        let mut l_i = Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
        let mut o_i = Tensor::<B, 4>::zeros(
            [batch_size, num_heads, q_block_len, head_dim],
            &device,
        );

        let mut kv_start = 0usize;

        for (k_block, v_block) in kv_blocks {
            let kv_block_len = k_block.dims()[1];
            let k_block_4d = k_block.clone().unsqueeze_dim(0);
            let v_block_4d = v_block.clone().unsqueeze_dim(0);

            let mut scores = q_block.clone().matmul(k_block_4d.transpose()) * inv_scale;

            if causal {
                let mask = self.build_causal_mask::<B>(
                    &device,
                    q_block_len,
                    kv_block_len,
                    q_start,
                    kv_start,
                    position_offset,
                );
                scores = scores + mask;
            }

            let m_ij = scores.clone().max_dim(3);
            let m_new = m_i.clone().max_pair(m_ij);

            let m_scale = (m_i - m_new.clone()).exp();
            let p_ij = (scores - m_new.clone()).exp();
            let p_sum = p_ij.clone().sum_dim(3);

            l_i = m_scale.clone() * l_i + p_sum;
            o_i = m_scale * o_i + p_ij.matmul(v_block_4d);
            m_i = m_new;

            kv_start += kv_block_len;
        }

        o_i / l_i
    }

    fn process_q_block_log_space<B: Backend>(
        &self,
        q_block: Tensor<B, 4>,
        kv_blocks: &[(Tensor<B, 3>, Tensor<B, 3>)],
        causal: bool,
        q_start: usize,
        position_offset: usize,
        inv_scale: f32,
    ) -> Tensor<B, 4> {
        let device = q_block.device();
        let [batch_size, num_heads, q_block_len, head_dim] = q_block.dims();

        let mut m_i = Tensor::<B, 4>::full(
            [batch_size, num_heads, q_block_len, 1],
            f32::NEG_INFINITY,
            &device,
        );
        let mut log_l_i = Tensor::<B, 4>::full(
            [batch_size, num_heads, q_block_len, 1],
            f32::NEG_INFINITY,
            &device,
        );
        let mut o_i = Tensor::<B, 4>::zeros(
            [batch_size, num_heads, q_block_len, head_dim],
            &device,
        );

        let mut kv_start = 0usize;

        for (k_block, v_block) in kv_blocks {
            let kv_block_len = k_block.dims()[1];
            let k_block_4d = k_block.clone().unsqueeze_dim(0);
            let v_block_4d = v_block.clone().unsqueeze_dim(0);

            let mut scores = q_block.clone().matmul(k_block_4d.transpose()) * inv_scale;

            if causal {
                let mask = self.build_causal_mask::<B>(
                    &device,
                    q_block_len,
                    kv_block_len,
                    q_start,
                    kv_start,
                    position_offset,
                );
                scores = scores + mask;
            }

            let m_ij = scores.clone().max_dim(3);
            let m_new = m_i.clone().max_pair(m_ij.clone());

            let scores_shifted = scores - m_ij.clone();
            let p_ij = scores_shifted.exp();
            let sum_p = p_ij.clone().sum_dim(3);
            let log_sum_p = sum_p.log();

            let m_diff = m_i - m_new.clone();
            let log_prev = m_diff.clone() + log_l_i;
            let log_curr = (m_ij - m_new.clone()) + log_sum_p;

            let log_l_new = Self::tensor_log_add_exp(log_prev, log_curr);

            let m_scale = m_diff.exp();
            o_i = m_scale * o_i + p_ij.matmul(v_block_4d);

            m_i = m_new;
            log_l_i = log_l_new;
            kv_start += kv_block_len;
        }

        let l_i = log_l_i.exp();
        o_i / l_i
    }

    fn tensor_log_add_exp<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> Tensor<B, 4> {
        let max = a.clone().max_pair(b.clone());
        let diff_a = a - max.clone();
        let diff_b = b - max.clone();
        max + (diff_a.exp() + diff_b.exp()).log()
    }

    fn build_causal_mask<B: Backend>(
        &self,
        device: &B::Device,
        query_len: usize,
        key_len: usize,
        q_start: usize,
        kv_start: usize,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(query_len * key_len);
        let mask_value = -1.0e4_f32;

        for i in 0..query_len {
            let absolute_pos = position_offset + q_start + i;
            for j in 0..key_len {
                let absolute_key = kv_start + j;
                let allowed = absolute_key <= absolute_pos;
                data.push(if allowed { 0.0 } else { mask_value });
            }
        }

        Tensor::<B, 2>::from_data(TensorData::new(data, [query_len, key_len]), device)
            .reshape([1, 1, query_len, key_len])
    }
}

impl<B: Backend> FusedPagedAttention<B> for HierarchicalFlashAttention {
    fn forward_fused<'a, I>(
        &self,
        q: Tensor<B, 4>,
        kv_blocks: I,
        config: &FlashAttentionConfig,
        causal: bool,
        position_offset: usize,
    ) -> Tensor<B, 4>
    where
        I: Iterator<Item = (Tensor<B, 3>, Tensor<B, 3>)> + 'a,
    {
        let kv_blocks: Vec<_> = kv_blocks.collect();
        let total_kv_len: usize = kv_blocks.iter().map(|(k, _)| k.dims()[1]).sum();

        let attention = Self::new(config.clone());

        attention.forward_fused_iter(q, kv_blocks.into_iter(), causal, position_offset, total_kv_len)
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::activation::softmax;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_hierarchical_flash_basic() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::default_config();

        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 16;
        let head_dim = 8;

        let q = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = attention.forward(q, k, v, false, 0);
        assert_eq!(output.dims(), [batch_size, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_hierarchical_flash_matches_standard() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::new(FlashAttentionConfig {
            block_q: 4,
            block_kv: 4,
            use_log_space: false,
            ..Default::default()
        });

        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 4;

        let q = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let output_hier = attention.forward(q.clone(), k.clone(), v.clone(), false, 0);

        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.transpose()) / scale;
        let attn = softmax(scores, 3);
        let output_std = attn.matmul(v);

        let hier_data = output_hier
            .into_data()
            .into_vec::<f32>()
            .expect("output data");
        let std_data = output_std
            .into_data()
            .into_vec::<f32>()
            .expect("output data");

        for (i, (h, s)) in hier_data.iter().zip(std_data.iter()).enumerate() {
            let diff = (h - s).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at {}: hier={}, std={}, diff={}",
                i,
                h,
                s,
                diff
            );
        }
    }

    #[test]
    fn test_fused_iter_matches_standard() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::new(FlashAttentionConfig {
            block_q: 4,
            block_kv: 4,
            use_log_space: false,
            ..Default::default()
        });

        let num_heads = 2;
        let seq_len = 16;
        let head_dim = 4;
        let block_size = 4;

        let q = Tensor::<TestBackend, 4>::random(
            [1, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let num_blocks = seq_len / block_size;
        let kv_blocks: Vec<_> = (0..num_blocks)
            .map(|_| {
                let k = Tensor::<TestBackend, 3>::random(
                    [num_heads, block_size, head_dim],
                    burn::tensor::Distribution::Normal(0.0, 0.5),
                    &device,
                );
                let v = Tensor::<TestBackend, 3>::random(
                    [num_heads, block_size, head_dim],
                    burn::tensor::Distribution::Normal(0.0, 0.5),
                    &device,
                );
                (k, v)
            })
            .collect();

        let output_fused = attention.forward_fused_iter(
            q.clone(),
            kv_blocks.clone().into_iter(),
            false,
            0,
            seq_len,
        );

        let k_cat: Vec<_> = kv_blocks.iter().map(|(k, _)| k.clone()).collect();
        let v_cat: Vec<_> = kv_blocks.iter().map(|(_, v)| v.clone()).collect();

        let k_full = Tensor::cat(k_cat, 1).reshape([1, num_heads, seq_len, head_dim]);
        let v_full = Tensor::cat(v_cat, 1).reshape([1, num_heads, seq_len, head_dim]);

        let output_std = attention.forward(q, k_full, v_full, false, 0);

        let fused_data = output_fused
            .into_data()
            .into_vec::<f32>()
            .expect("output data");
        let std_data = output_std
            .into_data()
            .into_vec::<f32>()
            .expect("output data");

        for (i, (f, s)) in fused_data.iter().zip(std_data.iter()).enumerate() {
            let diff = (f - s).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at {}: fused={}, std={}, diff={}",
                i,
                f,
                s,
                diff
            );
        }
    }

    #[test]
    fn test_causal_mask() {
        let device = <TestBackend as Backend>::Device::default();
        let attention = HierarchicalFlashAttention::default_config();

        let mask = attention.build_causal_mask::<TestBackend>(&device, 4, 4, 0, 0, 0);

        let data = mask.into_data().into_vec::<f32>().expect("mask data");

        assert!(data[0].abs() < 1e-5);
        assert!(data[1] < -1000.0);
        assert!(data[4].abs() < 1e-5);
        assert!(data[5].abs() < 1e-5);
        assert!(data[6] < -1000.0);
    }
}
