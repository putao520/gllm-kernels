//! Flash Attention CPU reference implementation.

use crate::kernel_types::{FloatType, KernelFloat};
use crate::types::FlashAttentionConfig;
use crate::ops::stable_accumulator::{StableAccumulator, AccumulatorConfig};
use crate::ops::simd_asm::simd_dot_product;

/// CPU reference implementation of Flash Attention.
pub fn flash_attention<T: KernelFloat>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: FlashAttentionConfig,
) {
    let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
    let heads = config.num_heads;
    let seq_q = config.seq_len_q;
    let seq_kv = config.seq_len_kv;
    let head_dim = config.head_dim;

    if seq_q == 1 {
        flash_attention_decode(q, k, v, output, heads, seq_kv, head_dim, scale);
        return;
    }

    let acc_config = if config.use_log_space_softmax || config.use_kahan_accumulator {
        AccumulatorConfig::max_precision()
    } else {
        AccumulatorConfig::short_context()
    };

    let chunk_size = seq_q * head_dim;
    output.chunks_exact_mut(chunk_size).enumerate().for_each(|(idx, out_chunk)| {
        process_head_attention(idx, out_chunk, heads, seq_q, seq_kv, head_dim, q, k, v, scale, &acc_config, config.causal);
    });
}

#[inline(always)]
pub fn cpu_flash_attention<T: KernelFloat>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: FlashAttentionConfig,
) {
    flash_attention(q, k, v, output, config);
}

#[inline(always)]
pub fn cpu_paged_attention<T: KernelFloat>(
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    page_table: &[u32],
    seq_lens: &[u32],
    output: &mut [T],
    config: crate::types::PagedAttentionConfig,
) {
    crate::ops::paged_attn::paged_attention(q, k_cache, v_cache, page_table, seq_lens, output, config);
}

/// Ultra-fast decode path (seq_q=1).
pub fn flash_attention_decode<T: KernelFloat>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    num_heads: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
) {
    for h in 0..num_heads {
        let q_offset = h * head_dim;
        let q_row = &q[q_offset..q_offset + head_dim];
        
        let mut max_score = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        let mut acc = [0.0f32; 256];
        let hd = head_dim.min(256);
        
        for j in 0..seq_kv {
            let k_offset = h * seq_kv * head_dim + j * head_dim;
            let k_row = &k[k_offset..k_offset + head_dim];
            
            let score = dot_product(q_row, k_row) * scale;
            
            max_score = max_score.max(score);
        }
        
        for j in 0..seq_kv {
            let k_offset = h * seq_kv * head_dim + j * head_dim;
            let k_row = &k[k_offset..k_offset + head_dim];
            let v_offset = h * seq_kv * head_dim + j * head_dim;
            let v_row = &v[v_offset..v_offset + head_dim];
            
            let score = dot_product(q_row, k_row) * scale;
            
            let exp_score = (score - max_score).exp();
            sum_exp += exp_score;
            
            for d in 0..hd {
                acc[d] += exp_score * v_row[d].to_f32();
            }
        }
        
        let out_offset = h * head_dim;
        let out_row = &mut output[out_offset..out_offset + head_dim];
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        
        for d in 0..head_dim {
            out_row[d] = T::from_f32(acc[d] * inv_sum);
        }
    }
}

fn process_head_attention<T: KernelFloat>(
    idx: usize,
    out_chunk: &mut [T],
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    q: &[T],
    k: &[T],
    v: &[T],
    scale: f32,
    acc_config: &AccumulatorConfig,
    causal: bool,
) {
    let b = idx / heads;
    let h = idx % heads;

    for i in 0..seq_q {
        let mut stable = StableAccumulator::new(acc_config.clone());
        let max_j = if causal { i + 1 } else { seq_kv };
        let mut scores = vec![0.0f32; max_j.min(seq_kv)];
        let mut block_max = -f32::INFINITY;

        let q_base = b * heads * seq_q * head_dim + h * seq_q * head_dim + i * head_dim;
        let q_row = &q[q_base..q_base + head_dim];

        for j in 0..max_j.min(seq_kv) {
            let k_base = b * heads * seq_kv * head_dim + h * seq_kv * head_dim + j * head_dim;
            let k_row = &k[k_base..k_base + head_dim];
            
            let score = dot_product(q_row, k_row) * scale;
            block_max = block_max.max(score);
            scores[j] = score;
        }

        if !scores.is_empty() {
            let block_sum_exp: f64 = scores.iter().map(|&s| ((s as f64) - (block_max as f64)).exp()).sum();
            stable.update(block_max as f64, block_sum_exp);
        }

        let m = stable.max();
        let l = stable.sum();

        let out_row = &mut out_chunk[i * head_dim..(i + 1) * head_dim];
        let mut weighted_sum = [0.0f32; 256];

        for (j, &score) in scores.iter().enumerate() {
            let attn_weight = if l > 0.0 { (((score as f64) - m).exp() / l) as f32 } else { 0.0 };
            if attn_weight == 0.0 { continue; }
            
            let v_base = b * heads * seq_kv * head_dim + h * seq_kv * head_dim + j * head_dim;
            let v_row = &v[v_base..v_base + head_dim];
            
            for d in 0..head_dim {
                weighted_sum[d] += attn_weight * v_row[d].to_f32();
            }
        }

        for d in 0..head_dim {
            out_row[d] = T::from_f32(weighted_sum[d]);
        }
    }
}

#[inline(always)]
fn dot_product<T: KernelFloat>(a: &[T], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match T::TYPE_ID {
        FloatType::F32 => {
            // T is f32 in this monomorphized branch.
            simd_dot_product(a.as_ptr() as *const f32, b.as_ptr() as *const f32, a.len())
        }
        _ => a.iter().zip(b.iter()).map(|(x, y)| x.to_f32() * y.to_f32()).sum(),
    }
}
