//! Paged Attention CPU reference implementation.

use crate::kernel_types::KernelFloat;
use crate::types::PagedAttentionConfig;
use crate::ops::stable_accumulator::{AccumulatorConfig, KahanAccumulator, StableAccumulator};

#[derive(Clone, Copy, Debug)]
pub struct PagedAttentionLayout {
    pub batch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub max_kv_len: usize,
    pub page_size: usize,
    pub num_blocks: usize,
    pub scale: f32,
}

/// CPU reference implementation of Paged Attention.
pub fn paged_attention<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output: &mut [T],
    config: PagedAttentionConfig,
) {
    let layout = match build_paged_layout(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        output.len(),
        &config,
    ) {
        Some(layout) => layout,
        None => {
            output.iter_mut().for_each(|v| *v = T::zero());
            return;
        }
    };

    let acc_config = if config.use_log_space_softmax || config.use_kahan_accumulator {
        AccumulatorConfig::max_precision()
    } else {
        AccumulatorConfig::short_context()
    };

    let batch_stride = layout.num_heads * layout.seq_len * layout.head_dim;
    let mut invalid_block_logged = false;

    for b in 0..layout.batch_size {
        let kv_len = seq_lens[b] as usize;
        if kv_len == 0 {
            let base = b * batch_stride;
            output[base..base + batch_stride].iter_mut().for_each(|v| *v = T::zero());
            continue;
        }
        if kv_len > layout.max_kv_len || kv_len < layout.seq_len {
            if !invalid_block_logged {
                log::warn!("Paged attention: invalid kv length for batch {}", b);
                invalid_block_logged = true;
            }
            let base = b * batch_stride;
            output[base..base + batch_stride].iter_mut().for_each(|v| *v = T::zero());
            continue;
        }

        let position_offset = kv_len - layout.seq_len;
        let table_base = b * layout.max_kv_len;
        let num_blocks = (kv_len + layout.page_size - 1) / layout.page_size;

        for h in 0..layout.num_heads {
            for q_pos in 0..layout.seq_len {
                let q_base = ((b * layout.num_heads + h) * layout.seq_len + q_pos) * layout.head_dim;
                let mut q_local = vec![0.0f32; layout.head_dim];
                for d in 0..layout.head_dim {
                    q_local[d] = q[q_base + d].to_f32();
                }

                let q_abs = position_offset + q_pos;
                let mut stable = StableAccumulator::new(acc_config.clone());
                let mut block_scores = Vec::with_capacity(layout.page_size);

                for block_idx in 0..num_blocks {
                    let token_base = block_idx * layout.page_size;
                    if token_base >= kv_len {
                        break;
                    }
                    let block_id = page_table[table_base + token_base] as usize;
                    if block_id >= layout.num_blocks {
                        if !invalid_block_logged {
                            log::warn!("Paged attention: page_table references invalid block id");
                            invalid_block_logged = true;
                        }
                        continue;
                    }

                    block_scores.clear();
                    let tokens_in_block = (kv_len - token_base).min(layout.page_size);
                    for t in 0..tokens_in_block {
                        let k_idx = token_base + t;
                        if k_idx > q_abs {
                            continue;
                        }
                        let kv_base =
                            ((block_id * layout.page_size + t) * layout.num_heads + h) * layout.head_dim;
                        let mut score = 0.0f32;
                        for d in 0..layout.head_dim {
                            score += q_local[d] * k_cache[kv_base + d].to_f32();
                        }
                        score *= layout.scale;
                        block_scores.push(score);
                    }

                    if block_scores.is_empty() {
                        continue;
                    }
                    let mut block_max = f64::NEG_INFINITY;
                    for &score in &block_scores {
                        block_max = block_max.max(score as f64);
                    }
                    let mut block_sum_exp = 0.0f64;
                    for &score in &block_scores {
                        block_sum_exp += ((score as f64) - block_max).exp();
                    }
                    stable.update(block_max, block_sum_exp);
                }

                let m = stable.max();
                let l = stable.sum();
                let mut weighted_sum: Vec<KahanAccumulator<f32>> =
                    vec![KahanAccumulator::new(); layout.head_dim];

                for block_idx in 0..num_blocks {
                    let token_base = block_idx * layout.page_size;
                    if token_base >= kv_len {
                        break;
                    }
                    let block_id = page_table[table_base + token_base] as usize;
                    if block_id >= layout.num_blocks {
                        continue;
                    }

                    let tokens_in_block = (kv_len - token_base).min(layout.page_size);
                    for t in 0..tokens_in_block {
                        let k_idx = token_base + t;
                        if k_idx > q_abs {
                            continue;
                        }
                        let kv_base =
                            ((block_id * layout.page_size + t) * layout.num_heads + h) * layout.head_dim;
                        let mut score = 0.0f32;
                        for d in 0..layout.head_dim {
                            score += q_local[d] * k_cache[kv_base + d].to_f32();
                        }
                        score *= layout.scale;

                        let attn_weight = if l > 0.0 {
                            (((score as f64) - m).exp() / l) as f32
                        } else {
                            0.0
                        };
                        for d in 0..layout.head_dim {
                            weighted_sum[d].add(attn_weight * v_cache[kv_base + d].to_f32());
                        }
                    }
                }

                for d in 0..layout.head_dim {
                    output[q_base + d] = T::from_f32(weighted_sum[d].value());
                }
            }
        }
    }
}

pub fn build_paged_layout<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output_len: usize,
    config: &PagedAttentionConfig,
) -> Option<PagedAttentionLayout> {
    if output_len == 0 || q.is_empty() || q.len() != output_len {
        return None;
    }
    let batch_size = seq_lens.len();
    if batch_size == 0 {
        return None;
    }
    let num_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let page_size = config.page_size;
    if num_heads == 0 || head_dim == 0 || page_size == 0 {
        return None;
    }
    let tokens_per_query = num_heads * head_dim * batch_size;
    if q.len() % tokens_per_query != 0 {
        return None;
    }
    let seq_len = q.len() / tokens_per_query;
    if seq_len == 0 || k_cache.len() != v_cache.len() {
        return None;
    }
    let block_stride = page_size * num_heads * head_dim;
    if block_stride == 0 || k_cache.len() % block_stride != 0 {
        return None;
    }
    let num_blocks = k_cache.len() / block_stride;
    if num_blocks == 0 || page_table.len() % batch_size != 0 {
        return None;
    }
    let max_kv_len = page_table.len() / batch_size;
    if max_kv_len == 0 {
        return None;
    }

    Some(PagedAttentionLayout {
        batch_size,
        num_heads,
        head_dim,
        seq_len,
        max_kv_len,
        page_size,
        num_blocks,
        scale: 1.0 / (head_dim as f32).sqrt(),
    })
}

pub(crate) struct PagedGpuInputs {
    pub(crate) layout: PagedAttentionLayout,
    pub(crate) q_f32: Vec<f32>,
    pub(crate) k_f32: Vec<f32>,
    pub(crate) v_f32: Vec<f32>,
    pub(crate) block_tables: Vec<i32>,
    pub(crate) block_offsets: Vec<i32>,
}

pub(crate) fn build_paged_gpu_inputs<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output_len: usize,
    config: &PagedAttentionConfig,
) -> Option<PagedGpuInputs> {
    let layout = build_paged_layout(q, k_cache, v_cache, page_table, seq_lens, output_len, config)?;
    let kv_len = seq_lens[0] as usize;
    if seq_lens.iter().any(|&len| len as usize != kv_len) {
        log::debug!("Paged attention: GPU kernels require uniform seq_lens");
        return None;
    }
    if kv_len != layout.max_kv_len {
        log::debug!("Paged attention: GPU kernels require packed page_table");
        return None;
    }
    if kv_len < layout.seq_len {
        log::warn!("Paged attention: kv_len shorter than seq_len");
        return None;
    }
    let offset = kv_len - layout.seq_len;
    let offset_i32 = match i32::try_from(offset) {
        Ok(value) => value,
        Err(_) => {
            log::warn!("Paged attention: block offset exceeds i32");
            return None;
        }
    };
    let block_offsets = vec![offset_i32; layout.batch_size];

    let max_block_id = page_table.iter().copied().max().unwrap_or(0) as usize;
    if max_block_id >= layout.num_blocks {
        log::warn!("Paged attention: page_table references invalid block id");
        return None;
    }

    let block_tables: Vec<i32> = match page_table
        .iter()
        .map(|&value| i32::try_from(value).ok())
        .collect::<Option<Vec<_>>>()
    {
        Some(values) => values,
        None => {
            log::warn!("Paged attention: page_table value exceeds i32");
            return None;
        }
    };

    let q_f32: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_cache.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_cache.iter().map(|x| x.to_f32()).collect();

    Some(PagedGpuInputs {
        layout,
        q_f32,
        k_f32,
        v_f32,
        block_tables,
        block_offsets,
    })
}
