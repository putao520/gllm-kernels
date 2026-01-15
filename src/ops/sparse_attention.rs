//! Sparse attention utilities with Lightning Indexer-style selection.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Configuration for sparse attention selection.
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    /// Number of KV tokens selected per query (typical: 2048).
    pub selected_kv_count: usize,
    /// Lightning Indexer block size.
    pub block_size: usize,
    /// Sparsity pattern to apply.
    pub sparsity_pattern: SparsityPattern,
}

/// Sparse attention patterns.
#[derive(Debug, Clone)]
pub enum SparsityPattern {
    /// Sliding window plus global tokens.
    SlidingWindowGlobal { window: usize, global_tokens: usize },
    /// Dynamic selection with Lightning Indexer.
    Dynamic,
    /// Block-sparse pattern.
    BlockSparse { block_size: usize },
}

/// Selected indices for sparse attention.
#[derive(Debug, Clone)]
pub struct SparseSelection {
    #[allow(dead_code)]
    batch: usize,
    num_heads: usize,
    query_len: usize,
    selected_kv_count: usize,
    indices: Vec<usize>,
}

impl SparseSelection {
    /// Create a new sparse selection result.
    pub fn new(
        batch: usize,
        num_heads: usize,
        query_len: usize,
        selected_kv_count: usize,
        indices: Vec<usize>,
    ) -> Self {
        Self {
            batch,
            num_heads,
            query_len,
            selected_kv_count,
            indices,
        }
    }

    /// Number of KV tokens selected per query.
    pub fn selected_kv_count(&self) -> usize {
        self.selected_kv_count
    }

    /// Slice of indices for a (batch, head, query) triplet.
    pub fn indices_for(&self, batch: usize, head: usize, query: usize) -> &[usize] {
        let stride = self.selected_kv_count;
        let idx = ((batch * self.num_heads + head) * self.query_len + query) * stride;
        &self.indices[idx..idx + stride]
    }

    /// Flat view of all indices.
    pub fn flat_indices(&self) -> &[usize] {
        &self.indices
    }
}

/// Sparse attention selector with Lightning Indexer.
#[derive(Debug, Clone)]
pub struct SparseAttention {
    config: SparseAttentionConfig,
}

impl SparseAttention {
    /// Create a new sparse attention selector.
    pub fn new(config: SparseAttentionConfig) -> Self {
        Self { config }
    }

    /// Access the sparse attention configuration.
    pub fn config(&self) -> &SparseAttentionConfig {
        &self.config
    }

    /// Select sparse indices for attention scores.
    ///
    /// # Shapes
    /// * `scores`: [batch, num_heads, query_len, kv_len]
    pub fn select_indices<B: Backend>(
        &self,
        scores: Tensor<B, 4>,
    ) -> Result<SparseSelection, &'static str> {
        let dims = scores.dims();
        let data = scores
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "sparse attention expects f32 scores")?;
        self.select_indices_from_data(&data, dims)
    }

    /// Apply sparse selection by masking unselected scores.
    ///
    /// # Shapes
    /// * `scores`: [batch, num_heads, query_len, kv_len]
    /// * returns: masked scores tensor + selection indices
    pub fn sparsify_scores<B: Backend>(
        &self,
        scores: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 4>, SparseSelection), &'static str> {
        let device = scores.device();
        let dims = scores.dims();
        let mut data = scores
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "sparse attention expects f32 scores")?;
        let selection = self.select_indices_from_data(&data, dims)?;

        let [batch, num_heads, query_len, kv_len] = dims;
        let stride_query = kv_len;
        let stride_head = query_len * stride_query;
        let stride_batch = num_heads * stride_head;

        for b in 0..batch {
            for h in 0..num_heads {
                for q in 0..query_len {
                    let offset = b * stride_batch + h * stride_head + q * stride_query;
                    let selected = selection.indices_for(b, h, q);
                    let mut keep = vec![false; kv_len];
                    for &idx in selected {
                        if idx < kv_len {
                            keep[idx] = true;
                        }
                    }
                    for idx in 0..kv_len {
                        if !keep[idx] {
                            data[offset + idx] = MASK_VALUE;
                        }
                    }
                }
            }
        }

        let masked = Tensor::<B, 4>::from_data(TensorData::new(data, dims), &device);
        Ok((masked, selection))
    }

    fn select_indices_from_data(
        &self,
        data: &[f32],
        dims: [usize; 4],
    ) -> Result<SparseSelection, &'static str> {
        self.validate_config()?;
        let [batch, num_heads, query_len, kv_len] = dims;
        if kv_len == 0 {
            return Err("kv_len must be > 0");
        }

        let target = self.config.selected_kv_count.min(kv_len);
        if target == 0 {
            return Err("selected_kv_count must be > 0");
        }

        let mut indices = Vec::with_capacity(batch * num_heads * query_len * target);
        let stride_query = kv_len;
        let stride_head = query_len * stride_query;
        let stride_batch = num_heads * stride_head;

        for b in 0..batch {
            for h in 0..num_heads {
                for q in 0..query_len {
                    let offset = b * stride_batch + h * stride_head + q * stride_query;
                    let scores = &data[offset..offset + kv_len];
                    let selected = self.select_for_query(scores, q, kv_len, target);
                    indices.extend(selected);
                }
            }
        }

        Ok(SparseSelection::new(
            batch,
            num_heads,
            query_len,
            target,
            indices,
        ))
    }

    fn select_for_query(
        &self,
        scores: &[f32],
        query_idx: usize,
        kv_len: usize,
        target: usize,
    ) -> Vec<usize> {
        let mut forced = Vec::new();
        let mut forced_mask = vec![false; kv_len];

        match self.config.sparsity_pattern {
            SparsityPattern::SlidingWindowGlobal {
                window,
                global_tokens,
            } => {
                let start = query_idx.saturating_sub(window);
                let end = (query_idx + window + 1).min(kv_len);
                for idx in start..end {
                    push_unique(idx, &mut forced, &mut forced_mask);
                }
                let global = global_tokens.min(kv_len);
                for idx in 0..global {
                    push_unique(idx, &mut forced, &mut forced_mask);
                }
            }
            SparsityPattern::Dynamic | SparsityPattern::BlockSparse { .. } => {}
        }

        if forced.len() >= target {
            return top_k_indices(scores, &forced, target);
        }

        let remaining = target - forced.len();
        let block_size = self.block_size();
        let block_count = (remaining + block_size - 1) / block_size;
        let blocks = select_blocks(scores, kv_len, block_size, block_count, Some(&forced_mask));

        let mut candidates = Vec::new();
        for block in blocks {
            let start = block * block_size;
            let end = (start + block_size).min(kv_len);
            for idx in start..end {
                if !forced_mask[idx] {
                    candidates.push(idx);
                }
            }
        }

        if candidates.len() < remaining {
            for idx in 0..kv_len {
                if !forced_mask[idx] {
                    candidates.push(idx);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();

        let mut selected = forced;
        if remaining > 0 {
            let mut extra = top_k_indices(scores, &candidates, remaining);
            selected.append(&mut extra);
        }
        selected.sort_unstable();
        selected.truncate(target);
        selected
    }

    fn validate_config(&self) -> Result<(), &'static str> {
        if self.config.selected_kv_count == 0 {
            return Err("selected_kv_count must be > 0");
        }
        if self.config.block_size == 0 {
            return Err("block_size must be > 0");
        }
        if let SparsityPattern::BlockSparse { block_size } = self.config.sparsity_pattern {
            if block_size == 0 {
                return Err("block sparse block_size must be > 0");
            }
        }
        Ok(())
    }

    fn block_size(&self) -> usize {
        match self.config.sparsity_pattern {
            SparsityPattern::BlockSparse { block_size } => block_size.max(1),
            _ => self.config.block_size.max(1),
        }
    }
}

const MASK_VALUE: f32 = -1.0e4_f32;

fn push_unique(idx: usize, list: &mut Vec<usize>, mask: &mut [bool]) {
    if !mask[idx] {
        mask[idx] = true;
        list.push(idx);
    }
}

fn select_blocks(
    scores: &[f32],
    kv_len: usize,
    block_size: usize,
    block_count: usize,
    skip_mask: Option<&[bool]>,
) -> Vec<usize> {
    if block_count == 0 || kv_len == 0 {
        return Vec::new();
    }
    let num_blocks = (kv_len + block_size - 1) / block_size;
    let mut block_scores = Vec::with_capacity(num_blocks);

    for block in 0..num_blocks {
        let start = block * block_size;
        let end = (start + block_size).min(kv_len);
        let mut max_score = f32::NEG_INFINITY;
        for idx in start..end {
            if skip_mask.map_or(false, |mask| mask[idx]) {
                continue;
            }
            let score = scores[idx];
            let score = if score.is_nan() { f32::NEG_INFINITY } else { score };
            if score > max_score {
                max_score = score;
            }
        }
        if max_score > f32::NEG_INFINITY {
            block_scores.push((block, max_score));
        }
    }

    block_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    block_scores.truncate(block_count.min(block_scores.len()));
    block_scores.into_iter().map(|(block, _)| block).collect()
}

fn top_k_indices(scores: &[f32], candidates: &[usize], k: usize) -> Vec<usize> {
    if k == 0 || candidates.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&idx| {
            let score = scores[idx];
            let score = if score.is_nan() { f32::NEG_INFINITY } else { score };
            (idx, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k.min(scored.len()));
    let mut indices: Vec<usize> = scored.into_iter().map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    indices
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor, TensorData};
    use burn_ndarray::NdArray;

    #[test]
    fn test_sliding_window_global_forced_tokens() {
        let config = SparseAttentionConfig {
            selected_kv_count: 4,
            block_size: 2,
            sparsity_pattern: SparsityPattern::SlidingWindowGlobal {
                window: 1,
                global_tokens: 1,
            },
        };
        let selector = SparseAttention::new(config);
        let device = <NdArray<f32> as Backend>::Device::default();
        let scores =
            Tensor::<NdArray<f32>, 4>::random([1, 1, 3, 6], Distribution::Uniform(0.0, 1.0), &device);

        let selection = selector.select_indices(scores).expect("selection");
        let indices = selection.indices_for(0, 0, 2);
        assert_eq!(indices, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_block_sparse_selection() {
        let config = SparseAttentionConfig {
            selected_kv_count: 3,
            block_size: 4,
            sparsity_pattern: SparsityPattern::BlockSparse { block_size: 4 },
        };
        let selector = SparseAttention::new(config);
        let device = <NdArray<f32> as Backend>::Device::default();
        let data = vec![0.1, 0.2, 0.3, 0.4, 5.0, 4.0, 3.0, 2.0];
        let scores = Tensor::<NdArray<f32>, 4>::from_data(TensorData::new(data, [1, 1, 1, 8]), &device);

        let selection = selector.select_indices(scores).expect("selection");
        let indices = selection.indices_for(0, 0, 0);
        assert_eq!(indices.len(), 3);
        assert!(indices.iter().all(|&idx| idx >= 4));
    }

    #[test]
    fn test_sparsify_scores_masks_unselected() {
        let config = SparseAttentionConfig {
            selected_kv_count: 1,
            block_size: 2,
            sparsity_pattern: SparsityPattern::Dynamic,
        };
        let selector = SparseAttention::new(config);
        let device = <NdArray<f32> as Backend>::Device::default();
        let data = vec![0.1, 0.2, 5.0, 0.3, 0.4];
        let scores = Tensor::<NdArray<f32>, 4>::from_data(TensorData::new(data, [1, 1, 1, 5]), &device);

        let (masked, selection) = selector.sparsify_scores(scores).expect("sparsify");
        let masked_data = masked.into_data().into_vec::<f32>().expect("masked data");
        let indices = selection.indices_for(0, 0, 0);
        assert_eq!(indices, &[2]);
        for (idx, value) in masked_data.iter().enumerate() {
            if idx == 2 {
                assert!((value - 5.0).abs() < 1e-4);
            } else {
                assert!((*value - MASK_VALUE).abs() < 1e-4);
            }
        }
    }
}
