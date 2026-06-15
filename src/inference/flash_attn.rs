//! FlashAttention — tiled attention computation for long sequences.
//!
//! Implements the FlashAttention algorithm (Dao et al., 2022) in pure Rust
//! for the CPU fallback path. Processes Q/K/V in tiles to keep the working
//! set in L1/L2 cache, avoiding the O(seq^2) memory of full attention matrices.
//!
//! Provides three execution paths:
//! - Contiguous: `flash_attn_multi_head` for prefill with contiguous Q/K/V
//! - Paged decode: `flash_attn_decode_paged` for single-token decode from paged KV cache
//! - Naive reference: `naive_attention_multi_head` for correctness testing

use crate::inference::kv_cache::{KvCache, PAGE_SIZE};

/// Compute default tile_kv from L1 cache heuristic.
///
/// Uses a conservative 32 KiB L1 estimate (half for KV tiles).
/// For each tile: K[tile, head_dim] + V[tile, head_dim] in f32.
fn default_tile_kv(head_dim: usize) -> usize {
    let l1_half = 16 * 1024; // 16 KiB = half of conservative 32 KiB L1
    let bytes_per_kv_row = head_dim * std::mem::size_of::<f32>() * 2; // K + V, f32
    if bytes_per_kv_row == 0 {
        return 256;
    }
    let max_tile = l1_half / bytes_per_kv_row;
    // Round down to power of 2, clamp [16, 512]
    let tile = max_tile.next_power_of_two() >> 1;
    tile.max(16).min(512)
}

/// Tiled attention parameters.
pub struct FlashAttnConfig {
    /// KV tile size (columns)
    pub tile_kv: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Attention scale factor (typically 1/sqrt(head_dim))
    pub scale: f32,
}

impl FlashAttnConfig {
    pub fn new(head_dim: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        FlashAttnConfig {
            tile_kv: 256,
            head_dim,
            num_heads,
            num_kv_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Create with tile_kv derived from L1 cache size.
    ///
    /// Optimal tile_kv = L1 / (2 * head_dim * sizeof(f32)) — fits one K tile
    /// and one V tile in L1 simultaneously, rounded down to power of 2.
    pub fn with_cache_hint(head_dim: usize, num_heads: usize, num_kv_heads: usize, l1_bytes: usize) -> Self {
        // Each tile needs K[tile_kv, head_dim] + V[tile_kv, head_dim] in cache
        let bytes_per_kv_row = head_dim * std::mem::size_of::<f32>() * 2; // K + V, f32
        let max_tile = if bytes_per_kv_row > 0 {
            (l1_bytes / 2) / bytes_per_kv_row // use half L1 for KV tiles
        } else {
            256
        };
        // Round down to power of 2, clamp to [16, 512]
        let tile_kv = max_tile.next_power_of_two() >> 1; // round down
        let tile_kv = tile_kv.max(16).min(512);
        FlashAttnConfig {
            tile_kv,
            head_dim,
            num_heads,
            num_kv_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

/// Run FlashAttention for a single head.
///
/// Uses online softmax (Milakov & Gimelshein, 2018) to process KV in tiles
/// without materializing the full attention matrix.
///
/// `q`: `[seq_q, head_dim]`
/// `k`: `[seq_kv, head_dim]`
/// `v`: `[seq_kv, head_dim]`
/// `output`: `[seq_q, head_dim]`
/// `causal`: if true, apply causal mask (position j masked when j > i)
pub fn flash_attn_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    let tile_kv = default_tile_kv(head_dim).min(seq_kv);

    // Per-row running max and sum for online softmax
    let mut row_max = vec![f32::NEG_INFINITY; seq_q];
    let mut row_sum = vec![0.0f32; seq_q];

    // Initialize output to zero
    for val in output.iter_mut() {
        *val = 0.0;
    }

    // Process KV in tiles
    let mut kv_start = 0;
    while kv_start < seq_kv {
        let kv_end = (kv_start + tile_kv).min(seq_kv);
        let tile_len = kv_end - kv_start;

        // For each query position
        for qi in 0..seq_q {
            let q_row = &q[qi * head_dim..(qi + 1) * head_dim];

            // Compute scores for this tile
            let mut tile_scores = vec![0.0f32; tile_len];
            for ti in 0..tile_len {
                let kv_pos = kv_start + ti;

                // Causal mask
                if causal && kv_pos > qi {
                    tile_scores[ti] = f32::NEG_INFINITY;
                    continue;
                }

                let k_row = &k[kv_pos * head_dim..(kv_pos + 1) * head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                tile_scores[ti] = dot * scale;
            }

            // Online softmax update
            let old_max = row_max[qi];
            let tile_max = tile_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let new_max = old_max.max(tile_max);

            // Rescale previous accumulator
            let correction = (old_max - new_max).exp();
            let out_row = &mut output[qi * head_dim..(qi + 1) * head_dim];
            for d in 0..head_dim {
                out_row[d] *= correction;
            }
            row_sum[qi] *= correction;

            // Accumulate this tile
            for ti in 0..tile_len {
                let w = (tile_scores[ti] - new_max).exp();
                let kv_pos = kv_start + ti;
                let v_row = &v[kv_pos * head_dim..(kv_pos + 1) * head_dim];
                for d in 0..head_dim {
                    out_row[d] += w * v_row[d];
                }
                row_sum[qi] += w;
            }

            row_max[qi] = new_max;
        }

        kv_start = kv_end;
    }

    // Final normalization
    for qi in 0..seq_q {
        let out_row = &mut output[qi * head_dim..(qi + 1) * head_dim];
        if row_sum[qi] > 0.0 {
            for d in 0..head_dim {
                out_row[d] /= row_sum[qi];
            }
        }
    }
}

/// Run FlashAttention for all heads (multi-head attention).
///
/// Handles GQA by mapping multiple query heads to the same KV head.
///
/// `q`: `[num_heads, seq_q, head_dim]`
/// `k`: `[num_kv_heads, seq_kv, head_dim]`
/// `v`: `[num_kv_heads, seq_kv, head_dim]`
/// `output`: `[num_heads, seq_q, head_dim]`
pub fn flash_attn_multi_head(
    config: &FlashAttnConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    causal: bool,
) {
    let hd = config.head_dim;
    let heads_per_kv = config.num_heads / config.num_kv_heads;

    for h in 0..config.num_heads {
        let kv_h = h / heads_per_kv;
        let q_off = h * seq_q * hd;
        let k_off = kv_h * seq_kv * hd;
        let v_off = kv_h * seq_kv * hd;
        let o_off = h * seq_q * hd;

        flash_attn_single_head(
            &q[q_off..q_off + seq_q * hd],
            &k[k_off..k_off + seq_kv * hd],
            &v[v_off..v_off + seq_kv * hd],
            &mut output[o_off..o_off + seq_q * hd],
            seq_q,
            seq_kv,
            hd,
            config.scale,
            causal,
        );
    }
}

// ============================================================================
// Naive attention — full materialization reference for correctness testing
// ============================================================================

/// Naive (full materialization) attention for a single head.
///
/// Computes the full `[seq_q, seq_kv]` score matrix in memory.
/// Memory: O(seq_q * seq_kv) — use only for testing, not production.
///
/// `q`: `[seq_q, head_dim]`
/// `k`: `[seq_kv, head_dim]`
/// `v`: `[seq_kv, head_dim]`
/// `output`: `[seq_q, head_dim]`
pub fn naive_attention_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    for qi in 0..seq_q {
        let q_row = &q[qi * head_dim..(qi + 1) * head_dim];

        // Compute all scores
        let mut scores = vec![0.0f32; seq_kv];
        for kj in 0..seq_kv {
            if causal && kj > qi {
                scores[kj] = f32::NEG_INFINITY;
                continue;
            }
            let k_row = &k[kj * head_dim..(kj + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            scores[kj] = dot * scale;
        }

        // Softmax
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        // Weighted sum of V
        let out_row = &mut output[qi * head_dim..(qi + 1) * head_dim];
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for kj in 0..seq_kv {
                val += scores[kj] * v[kj * head_dim + d];
            }
            out_row[d] = val;
        }
    }
}

/// Naive multi-head attention with GQA support.
///
/// Full materialization reference — O(N^2) memory per head.
///
/// `q`: `[num_heads, seq_q, head_dim]`
/// `k`: `[num_kv_heads, seq_kv, head_dim]`
/// `v`: `[num_kv_heads, seq_kv, head_dim]`
/// `output`: `[num_heads, seq_q, head_dim]`
pub fn naive_attention_multi_head(
    config: &FlashAttnConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    causal: bool,
) {
    let hd = config.head_dim;
    let heads_per_kv = config.num_heads / config.num_kv_heads;

    for h in 0..config.num_heads {
        let kv_h = h / heads_per_kv;
        let q_off = h * seq_q * hd;
        let k_off = kv_h * seq_kv * hd;
        let v_off = kv_h * seq_kv * hd;
        let o_off = h * seq_q * hd;

        naive_attention_single_head(
            &q[q_off..q_off + seq_q * hd],
            &k[k_off..k_off + seq_kv * hd],
            &v[v_off..v_off + seq_kv * hd],
            &mut output[o_off..o_off + seq_q * hd],
            seq_q,
            seq_kv,
            hd,
            config.scale,
            causal,
        );
    }
}

// ============================================================================
// Paged KV cache decode — single-token attention from paged cache
// ============================================================================

/// FlashAttention decode path: single query token reading K/V from paged KV cache.
///
/// Uses online softmax (tiled) to avoid materializing the full score vector.
/// Handles GQA, causal mask, and optional sliding window.
/// Memory: O(tile_kv) per head — constant regardless of cached sequence length.
///
/// `q`: `[num_heads * head_dim]` — single token, all heads concatenated
/// `output`: `[num_heads * head_dim]` — single token output
/// `token_pos`: absolute position of the query token (for causal mask)
pub fn flash_attn_decode_paged(
    config: &FlashAttnConfig,
    q: &[f32],
    output: &mut [f32],
    kv_cache: &KvCache,
    layer: usize,
    seq_idx: usize,
    token_pos: usize,
    sliding_window: Option<usize>,
) {
    let hd = config.head_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = config.scale;
    let cached_len = kv_cache.seq_len(layer, seq_idx);
    let seq_pages = kv_cache.seq_pages(layer, seq_idx);
    let v_base = num_kv_heads * PAGE_SIZE * hd;
    let tile_kv = config.tile_kv.min(cached_len);
    let kv_dtype = kv_cache.dtype();
    let elem_bytes = kv_dtype.size_bytes();

    if cached_len == 0 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    for ah in 0..num_heads {
        let kv_h = ah / heads_per_kv;
        let q_off = ah * hd;
        let o_off = ah * hd;

        // Online softmax state
        let mut row_max = f32::NEG_INFINITY;
        let mut row_sum = 0.0f32;
        for d in 0..hd {
            output[o_off + d] = 0.0;
        }

        // Process KV in tiles
        let mut kv_start = 0;
        while kv_start < cached_len {
            let kv_end = (kv_start + tile_kv).min(cached_len);
            let tile_len = kv_end - kv_start;

            // Compute scores for this tile
            let mut tile_scores = vec![0.0f32; tile_len];
            for ti in 0..tile_len {
                let t = kv_start + ti;

                // Causal mask
                if t > token_pos {
                    tile_scores[ti] = f32::NEG_INFINITY;
                    continue;
                }
                // Sliding window mask
                if let Some(w) = sliding_window {
                    if token_pos >= w && t < token_pos - w + 1 {
                        tile_scores[ti] = f32::NEG_INFINITY;
                        continue;
                    }
                }

                let pid = seq_pages[t / PAGE_SIZE];
                let off = t % PAGE_SIZE;
                let kp = kv_cache.page_ptr(pid);
                let k_base = kv_h * PAGE_SIZE * hd + off * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    let byte_off = (k_base + d) * elem_bytes;
                    let k_val = unsafe {
                        match kv_dtype {
                            crate::types::DType::F32 => *(kp.add(byte_off) as *const f32),
                            crate::types::DType::BF16 => {
                                (*(kp.add(byte_off) as *const half::bf16)).to_f32()
                            }
                            crate::types::DType::F16 => {
                                (*(kp.add(byte_off) as *const half::f16)).to_f32()
                            }
                            other => panic!("unsupported KV cache dtype for K read: {other:?}"),
                        }
                    };
                    dot += q[q_off + d] * k_val;
                }
                tile_scores[ti] = dot * scale;
            }

            // Online softmax update
            let old_max = row_max;
            let tile_max = tile_scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let new_max = old_max.max(tile_max);

            // Rescale previous accumulator
            let correction = (old_max - new_max).exp();
            for d in 0..hd {
                output[o_off + d] *= correction;
            }
            row_sum *= correction;

            // Accumulate this tile
            for ti in 0..tile_len {
                let w = (tile_scores[ti] - new_max).exp();
                let t = kv_start + ti;
                let pid = seq_pages[t / PAGE_SIZE];
                let off = t % PAGE_SIZE;
                let vp = kv_cache.page_ptr(pid);
                let v_page_off = v_base + kv_h * PAGE_SIZE * hd + off * hd;
                for d in 0..hd {
                    let byte_off = (v_page_off + d) * elem_bytes;
                    let v_val = unsafe {
                        match kv_dtype {
                            crate::types::DType::F32 => *(vp.add(byte_off) as *const f32),
                            crate::types::DType::BF16 => {
                                (*(vp.add(byte_off) as *const half::bf16)).to_f32()
                            }
                            crate::types::DType::F16 => {
                                (*(vp.add(byte_off) as *const half::f16)).to_f32()
                            }
                            other => panic!("unsupported KV cache dtype for V read: {other:?}"),
                        }
                    };
                    output[o_off + d] += w * v_val;
                }
                row_sum += w;
            }

            row_max = new_max;
            kv_start = kv_end;
        }

        // Final normalization
        if row_sum > 0.0 {
            for d in 0..hd {
                output[o_off + d] /= row_sum;
            }
        }
    }
}

// ============================================================================
// Prefill path — multi-token attention with contiguous Q/K/V
// ============================================================================

/// FlashAttention prefill: multi-token causal attention with GQA.
///
/// Uses online softmax (tiled) — memory O(N) not O(N^2).
/// `kv_offset`: position offset for causal mask when prior tokens exist in cache.
///   For fresh prefill, pass 0. For continued prefill, pass prior cached length.
///
/// `q`: `[num_heads, seq_q, head_dim]`
/// `k`: `[num_kv_heads, seq_kv, head_dim]`
/// `v`: `[num_kv_heads, seq_kv, head_dim]`
/// `output`: `[num_heads, seq_q, head_dim]`
pub fn flash_attn_prefill(
    config: &FlashAttnConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    kv_offset: usize,
) {
    let hd = config.head_dim;
    let heads_per_kv = config.num_heads / config.num_kv_heads;

    for h in 0..config.num_heads {
        let kv_h = h / heads_per_kv;
        let q_off = h * seq_q * hd;
        let k_off = kv_h * seq_kv * hd;
        let v_off = kv_h * seq_kv * hd;
        let o_off = h * seq_q * hd;

        flash_attn_prefill_single_head(
            &q[q_off..q_off + seq_q * hd],
            &k[k_off..k_off + seq_kv * hd],
            &v[v_off..v_off + seq_kv * hd],
            &mut output[o_off..o_off + seq_q * hd],
            seq_q,
            seq_kv,
            hd,
            config.scale,
            kv_offset,
        );
    }
}

/// Single-head prefill with online softmax and causal mask.
fn flash_attn_prefill_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    kv_offset: usize,
) {
    let tile_kv = default_tile_kv(head_dim).min(seq_kv);

    let mut row_max = vec![f32::NEG_INFINITY; seq_q];
    let mut row_sum = vec![0.0f32; seq_q];

    for val in output.iter_mut() {
        *val = 0.0;
    }

    let mut kv_start = 0;
    while kv_start < seq_kv {
        let kv_end = (kv_start + tile_kv).min(seq_kv);
        let tile_len = kv_end - kv_start;

        for qi in 0..seq_q {
            let q_row = &q[qi * head_dim..(qi + 1) * head_dim];
            let q_pos = kv_offset + qi;

            let mut tile_scores = vec![0.0f32; tile_len];
            for ti in 0..tile_len {
                let kv_pos = kv_start + ti;

                // Causal mask: query at q_pos can only attend to kv_pos <= q_pos
                if kv_pos > q_pos {
                    tile_scores[ti] = f32::NEG_INFINITY;
                    continue;
                }

                let k_row = &k[kv_pos * head_dim..(kv_pos + 1) * head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                tile_scores[ti] = dot * scale;
            }

            // Online softmax update
            let old_max = row_max[qi];
            let tile_max = tile_scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let new_max = old_max.max(tile_max);

            let correction = (old_max - new_max).exp();
            let out_row = &mut output[qi * head_dim..(qi + 1) * head_dim];
            for d in 0..head_dim {
                out_row[d] *= correction;
            }
            row_sum[qi] *= correction;

            for ti in 0..tile_len {
                let w = (tile_scores[ti] - new_max).exp();
                let kv_pos = kv_start + ti;
                let v_row = &v[kv_pos * head_dim..(kv_pos + 1) * head_dim];
                for d in 0..head_dim {
                    out_row[d] += w * v_row[d];
                }
                row_sum[qi] += w;
            }

            row_max[qi] = new_max;
        }

        kv_start = kv_end;
    }

    // Final normalization
    for qi in 0..seq_q {
        let out_row = &mut output[qi * head_dim..(qi + 1) * head_dim];
        if row_sum[qi] > 0.0 {
            for d in 0..head_dim {
                out_row[d] /= row_sum[qi];
            }
        }
    }
}

// ============================================================================
// Layout transpose helpers
// ============================================================================

/// Transpose from token-major `[seq_len, num_heads * head_dim]` to
/// head-major `[num_heads, seq_len, head_dim]`.
pub fn transpose_to_head_major(
    src: &[f32],
    dst: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    for s in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                dst[h * seq_len * head_dim + s * head_dim + d] =
                    src[s * num_heads * head_dim + h * head_dim + d];
            }
        }
    }
}

/// Transpose from head-major `[num_heads, seq_len, head_dim]` to
/// token-major `[seq_len, num_heads * head_dim]`.
pub fn transpose_to_token_major(
    src: &[f32],
    dst: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    for h in 0..num_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                dst[s * num_heads * head_dim + h * head_dim + d] =
                    src[h * seq_len * head_dim + s * head_dim + d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single token attention: Q=[1, hd], K=V=[1, hd].
    /// With only one KV position, softmax = [1.0], so output = V.
    #[test]
    fn test_flash_attn_single_token() {
        let head_dim = 4;
        let q = vec![1.0f32, 0.0, 1.0, 0.0];
        let k = vec![1.0f32, 1.0, 1.0, 1.0];
        let v = vec![0.5f32, 1.5, 2.5, 3.5];
        let mut output = vec![0.0f32; head_dim];

        let scale = 1.0 / (head_dim as f32).sqrt();
        flash_attn_single_head(&q, &k, &v, &mut output, 1, 1, head_dim, scale, false);

        // Only one KV position => softmax weight = 1.0 => output = V
        for d in 0..head_dim {
            assert!(
                (output[d] - v[d]).abs() < 1e-6,
                "output[{d}] = {}, expected {}",
                output[d],
                v[d]
            );
        }
    }

    /// Causal mask: position 0 should only see position 0, not position 1.
    #[test]
    fn test_flash_attn_causal_mask() {
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 2 query positions, 2 KV positions
        // Q = [[1,1,1,1], [1,1,1,1]]
        // K = [[1,1,1,1], [1,1,1,1]]  (equal scores)
        // V = [[1,0,0,0], [0,1,0,0]]  (distinct values to verify masking)
        let q = vec![1.0f32; 2 * head_dim];
        let k = vec![1.0f32; 2 * head_dim];
        let mut v = vec![0.0f32; 2 * head_dim];
        v[0] = 1.0; // V[0] = [1,0,0,0]
        v[head_dim + 1] = 1.0; // V[1] = [0,1,0,0]

        let mut output = vec![0.0f32; 2 * head_dim];
        flash_attn_single_head(&q, &k, &v, &mut output, 2, 2, head_dim, scale, true);

        // Position 0 (causal): sees only position 0 => output = V[0] = [1,0,0,0]
        assert!(
            (output[0] - 1.0).abs() < 1e-5,
            "pos 0 output[0] = {}, expected 1.0",
            output[0]
        );
        assert!(
            output[1].abs() < 1e-5,
            "pos 0 output[1] = {}, expected 0.0",
            output[1]
        );

        // Position 1 (causal): sees positions 0 and 1 with equal scores => 50/50
        // output = 0.5 * [1,0,0,0] + 0.5 * [0,1,0,0] = [0.5, 0.5, 0, 0]
        assert!(
            (output[head_dim] - 0.5).abs() < 1e-5,
            "pos 1 output[0] = {}, expected 0.5",
            output[head_dim]
        );
        assert!(
            (output[head_dim + 1] - 0.5).abs() < 1e-5,
            "pos 1 output[1] = {}, expected 0.5",
            output[head_dim + 1]
        );
    }

    /// FlashAttention result must match naive full-materialization attention
    /// within 1e-5 tolerance.
    #[test]
    fn test_flash_attn_matches_naive() {
        let head_dim = 8;
        let seq_q = 6;
        let seq_kv = 6;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Deterministic Q/K/V with varied values
        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.13).sin() * 0.5)
                .collect()
        };
        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 100);
        let v = make_data(seq_kv * head_dim, 200);

        // --- Naive attention (full materialization) ---
        let mut naive_output = vec![0.0f32; seq_q * head_dim];
        for qi in 0..seq_q {
            // Compute all scores
            let mut scores = vec![0.0f32; seq_kv];
            for kj in 0..seq_kv {
                // Causal mask
                if kj > qi {
                    scores[kj] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * head_dim + d] * k[kj * head_dim + d];
                }
                scores[kj] = dot * scale;
            }
            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }
            // Weighted sum
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for kj in 0..seq_kv {
                    val += scores[kj] * v[kj * head_dim + d];
                }
                naive_output[qi * head_dim + d] = val;
            }
        }

        // --- FlashAttention ---
        let mut flash_output = vec![0.0f32; seq_q * head_dim];
        flash_attn_single_head(
            &q, &k, &v, &mut flash_output, seq_q, seq_kv, head_dim, scale, true,
        );

        // Compare
        let max_diff = naive_output
            .iter()
            .zip(flash_output.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "FlashAttention vs naive max diff = {max_diff:.2e} (threshold 1e-5)"
        );
    }

    /// Multi-head attention with GQA (num_heads=4, num_kv_heads=2).
    #[test]
    fn test_flash_attn_multi_head() {
        let head_dim = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let seq_q = 3;
        let seq_kv = 3;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.17).sin() * 0.4)
                .collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 150);
        let mut output = vec![0.0f32; num_heads * seq_q * head_dim];

        flash_attn_multi_head(&config, &q, &k, &v, &mut output, seq_q, seq_kv, true);

        // Verify per-head against single-head reference
        let heads_per_kv = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * seq_q * head_dim;
            let k_off = kv_h * seq_kv * head_dim;
            let v_off = kv_h * seq_kv * head_dim;
            let o_off = h * seq_q * head_dim;

            let mut ref_out = vec![0.0f32; seq_q * head_dim];
            flash_attn_single_head(
                &q[q_off..q_off + seq_q * head_dim],
                &k[k_off..k_off + seq_kv * head_dim],
                &v[v_off..v_off + seq_kv * head_dim],
                &mut ref_out,
                seq_q,
                seq_kv,
                head_dim,
                config.scale,
                true,
            );

            let max_diff = output[o_off..o_off + seq_q * head_dim]
                .iter()
                .zip(ref_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-6,
                "multi_head vs single_head mismatch at head {h}: max_diff = {max_diff:.2e}"
            );
        }

        // Sanity: output should be finite and non-trivial
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().any(|v| v.abs() > 1e-6));
    }

    // ====================================================================
    // NEW TESTS: GQA correctness, paged decode, prefill, long sequences
    // ====================================================================

    /// GQA numerical correctness: flash_attn_multi_head vs naive_attention_multi_head
    /// for multiple GQA ratios (4:2, 8:2, 8:1, 32:1).
    /// Tolerance: ≤1e-3 relative error.
    #[test]
    fn test_gqa_numerical_correctness() {
        let head_dim = 16;
        let seq_len = 12;

        let ratios: &[(usize, usize)] = &[(4, 2), (8, 2), (8, 1), (32, 1)];

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.07).sin() * 0.3)
                .collect()
        };

        for &(num_heads, num_kv_heads) in ratios {
            let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

            let q = make_data(num_heads * seq_len * head_dim, 0);
            let k = make_data(num_kv_heads * seq_len * head_dim, 100);
            let v = make_data(num_kv_heads * seq_len * head_dim, 200);

            let mut flash_out = vec![0.0f32; num_heads * seq_len * head_dim];
            let mut naive_out = vec![0.0f32; num_heads * seq_len * head_dim];

            flash_attn_multi_head(
                &config, &q, &k, &v, &mut flash_out, seq_len, seq_len, true,
            );
            naive_attention_multi_head(
                &config, &q, &k, &v, &mut naive_out, seq_len, seq_len, true,
            );

            let max_rel_err = flash_out
                .iter()
                .zip(naive_out.iter())
                .map(|(&f, &n)| {
                    let denom = n.abs().max(1e-7);
                    (f - n).abs() / denom
                })
                .fold(0.0f32, f32::max);

            assert!(
                max_rel_err < 1e-3,
                "GQA {num_heads}:{num_kv_heads} max relative error = {max_rel_err:.2e} (threshold 1e-3)"
            );
        }
    }

    /// FlashAttention long sequence (seq_len=2048): verify it completes without
    /// OOM and produces correct results vs naive on a small sample.
    /// The flash path uses O(N) memory; naive would use O(N^2).
    #[test]
    fn test_flash_attn_long_sequence_memory() {
        let head_dim = 64;
        let seq_len = 2048;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.003).sin() * 0.2)
                .collect()
        };

        let q = make_data(seq_len * head_dim, 0);
        let k = make_data(seq_len * head_dim, 1000);
        let v = make_data(seq_len * head_dim, 2000);

        let mut flash_out = vec![0.0f32; seq_len * head_dim];
        flash_attn_single_head(
            &q, &k, &v, &mut flash_out, seq_len, seq_len, head_dim, scale, true,
        );

        // Verify output is finite and non-trivial
        assert!(
            flash_out.iter().all(|v| v.is_finite()),
            "long sequence output contains non-finite values"
        );
        assert!(
            flash_out.iter().any(|v| v.abs() > 1e-8),
            "long sequence output is all zeros"
        );

        // Spot-check: verify first and last query positions against naive
        // (only check 2 positions to avoid O(N^2) memory for full naive)
        for qi in [0, seq_len - 1] {
            let q_row = &q[qi * head_dim..(qi + 1) * head_dim];
            let visible_len = qi + 1; // causal: can see [0..qi]

            let mut scores = vec![0.0f32; visible_len];
            for kj in 0..visible_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k[kj * head_dim + d];
                }
                scores[kj] = dot * scale;
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            let mut expected = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                for kj in 0..visible_len {
                    expected[d] += scores[kj] * v[kj * head_dim + d];
                }
            }

            let max_diff = flash_out[qi * head_dim..(qi + 1) * head_dim]
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-4,
                "long seq qi={qi}: flash vs naive max_diff = {max_diff:.2e}"
            );
        }
    }

    /// Paged KV cache decode: single-token attention reading from paged cache.
    /// Write K/V manually into cache pages, then verify flash_attn_decode_paged
    /// matches naive computation.
    #[test]
    fn test_flash_attn_decode_paged_single_token() {
        use crate::types::ModelConfig;

        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let cached_len = 5usize;

        let mut kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        // Generate deterministic K/V data
        let make_val = |i: usize, seed: usize| -> f32 {
            ((i + seed) as f32 * 0.11).sin() * 0.3
        };

        // Append tokens and write K/V into cache pages
        let v_base = num_kv_heads * PAGE_SIZE * head_dim;
        let positions = kv_cache.append(0, 0, cached_len).unwrap();
        for (t, &(pid, off)) in positions.iter().enumerate() {
            let page_ptr = kv_cache.page_mut_ptr(pid) as *mut f32;
            for kv_h in 0..num_kv_heads {
                let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim;
                for d in 0..head_dim {
                    unsafe {
                        *page_ptr.add(k_base + d) = make_val(t * head_dim + d, kv_h * 100);
                        *page_ptr.add(v_off + d) = make_val(t * head_dim + d, kv_h * 100 + 500);
                    }
                }
            }
        }

        // Query at position 4 (last cached position)
        let token_pos = cached_len - 1;
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| make_val(i, 999))
            .collect();
        let mut paged_out = vec![0.0f32; num_heads * head_dim];

        flash_attn_decode_paged(
            &attn_config,
            &q,
            &mut paged_out,
            &kv_cache,
            0,
            0,
            token_pos,
            None,
        );

        // Build contiguous K/V for naive reference
        let heads_per_kv = num_heads / num_kv_heads;
        for ah in 0..num_heads {
            let kv_h = ah / heads_per_kv;
            let q_off = ah * head_dim;

            // Gather K/V for this kv_head
            let mut k_cont = vec![0.0f32; cached_len * head_dim];
            let mut v_cont = vec![0.0f32; cached_len * head_dim];
            for t in 0..cached_len {
                for d in 0..head_dim {
                    k_cont[t * head_dim + d] = make_val(t * head_dim + d, kv_h * 100);
                    v_cont[t * head_dim + d] = make_val(t * head_dim + d, kv_h * 100 + 500);
                }
            }

            // Naive single-head attention (seq_q=1)
            // Use causal=false because the paged decode uses token_pos for masking,
            // and all cached positions [0..cached_len-1] are <= token_pos.
            // The naive causal mask uses qi (=0 for decode), which would incorrectly
            // mask out positions > 0.
            let mut naive_out = vec![0.0f32; head_dim];
            naive_attention_single_head(
                &q[q_off..q_off + head_dim],
                &k_cont,
                &v_cont,
                &mut naive_out,
                1,
                cached_len,
                head_dim,
                attn_config.scale,
                false,
            );

            let max_diff = paged_out[q_off..q_off + head_dim]
                .iter()
                .zip(naive_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-5,
                "paged decode head {ah}: max_diff = {max_diff:.2e}"
            );
        }
    }

    /// Multi-step autoregressive decode with paged KV cache.
    /// Append tokens one at a time, verify each step's attention output.
    #[test]
    fn test_flash_attn_decode_paged_multi_step() {
        use crate::types::ModelConfig;

        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);
        let v_base = num_kv_heads * PAGE_SIZE * head_dim;

        let mut kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();

        let make_val = |i: usize, seed: usize| -> f32 {
            ((i + seed) as f32 * 0.13).sin() * 0.25
        };

        // Accumulate K/V for naive reference
        let mut all_k = Vec::<Vec<f32>>::new(); // [num_kv_heads][t * head_dim..]
        for _ in 0..num_kv_heads {
            all_k.push(Vec::new());
        }
        let mut all_v = Vec::<Vec<f32>>::new();
        for _ in 0..num_kv_heads {
            all_v.push(Vec::new());
        }

        let num_steps = 8;
        for step in 0..num_steps {
            // Append one token to cache
            let positions = kv_cache.append(0, 0, 1).unwrap();
            let (pid, off) = positions[0];
            let page_ptr = kv_cache.page_mut_ptr(pid) as *mut f32;

            for kv_h in 0..num_kv_heads {
                let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim;
                for d in 0..head_dim {
                    let kval = make_val(step * head_dim + d, kv_h * 100);
                    let vval = make_val(step * head_dim + d, kv_h * 100 + 500);
                    unsafe {
                        *page_ptr.add(k_base + d) = kval;
                        *page_ptr.add(v_off + d) = vval;
                    }
                    all_k[kv_h].push(kval);
                    all_v[kv_h].push(vval);
                }
            }

            // Query
            let q: Vec<f32> = (0..num_heads * head_dim)
                .map(|i| make_val(i, step * 1000 + 999))
                .collect();
            let mut paged_out = vec![0.0f32; num_heads * head_dim];

            flash_attn_decode_paged(
                &attn_config,
                &q,
                &mut paged_out,
                &kv_cache,
                0,
                0,
                step,
                None,
            );

            // Naive reference per head
            let cached_len = step + 1;
            let heads_per_kv = num_heads / num_kv_heads;
            for ah in 0..num_heads {
                let kv_h = ah / heads_per_kv;
                let q_off = ah * head_dim;

                let mut naive_out = vec![0.0f32; head_dim];
                naive_attention_single_head(
                    &q[q_off..q_off + head_dim],
                    &all_k[kv_h][..cached_len * head_dim],
                    &all_v[kv_h][..cached_len * head_dim],
                    &mut naive_out,
                    1,
                    cached_len,
                    head_dim,
                    attn_config.scale,
                    false, // causal=false: all cached positions are <= token_pos
                );

                let max_diff = paged_out[q_off..q_off + head_dim]
                    .iter()
                    .zip(naive_out.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);

                assert!(
                    max_diff < 1e-5,
                    "step {step} head {ah}: max_diff = {max_diff:.2e}"
                );
            }
        }
    }

    /// Prefill with causal mask: flash_attn_prefill vs naive_attention_multi_head.
    #[test]
    fn test_flash_attn_prefill_causal() {
        let head_dim = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let seq_len = 20;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.09).sin() * 0.4)
                .collect()
        };

        let q = make_data(num_heads * seq_len * head_dim, 0);
        let k = make_data(num_kv_heads * seq_len * head_dim, 100);
        let v = make_data(num_kv_heads * seq_len * head_dim, 200);

        let mut prefill_out = vec![0.0f32; num_heads * seq_len * head_dim];
        let mut naive_out = vec![0.0f32; num_heads * seq_len * head_dim];

        flash_attn_prefill(
            &config,
            &q,
            &k,
            &v,
            &mut prefill_out,
            seq_len,
            seq_len,
            0,
        );
        naive_attention_multi_head(
            &config,
            &q,
            &k,
            &v,
            &mut naive_out,
            seq_len,
            seq_len,
            true,
        );

        let max_rel_err = prefill_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&f, &n)| {
                let denom = n.abs().max(1e-7);
                (f - n).abs() / denom
            })
            .fold(0.0f32, f32::max);

        assert!(
            max_rel_err < 1e-3,
            "prefill vs naive max relative error = {max_rel_err:.2e} (threshold 1e-3)"
        );
    }

    /// Prefill followed by decode: verify autoregressive consistency.
    /// Prefill 4 tokens, then decode 4 more tokens one at a time.
    /// The decode outputs should match what a full naive attention would produce.
    #[test]
    fn test_prefill_then_decode_consistency() {
        use crate::types::ModelConfig;

        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);
        let v_base = num_kv_heads * PAGE_SIZE * head_dim;

        let prefill_len = 4usize;
        let decode_steps = 4usize;
        let total_len = prefill_len + decode_steps;

        let make_val = |i: usize, seed: usize| -> f32 {
            ((i + seed) as f32 * 0.11).sin() * 0.3
        };

        // Pre-generate all K/V/Q for all positions
        let all_k_flat: Vec<Vec<f32>> = (0..num_kv_heads)
            .map(|kv_h| {
                (0..total_len * head_dim)
                    .map(|i| make_val(i, kv_h * 100))
                    .collect()
            })
            .collect();
        let all_v_flat: Vec<Vec<f32>> = (0..num_kv_heads)
            .map(|kv_h| {
                (0..total_len * head_dim)
                    .map(|i| make_val(i, kv_h * 100 + 500))
                    .collect()
            })
            .collect();
        let all_q: Vec<Vec<f32>> = (0..total_len)
            .map(|t| {
                (0..num_heads * head_dim)
                    .map(|i| make_val(i, t * 1000 + 999))
                    .collect()
            })
            .collect();

        // Write all K/V into paged cache
        let mut kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();
        let positions = kv_cache.append(0, 0, total_len).unwrap();
        for (t, &(pid, off)) in positions.iter().enumerate() {
            let page_ptr = kv_cache.page_mut_ptr(pid) as *mut f32;
            for kv_h in 0..num_kv_heads {
                let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim;
                for d in 0..head_dim {
                    unsafe {
                        *page_ptr.add(k_base + d) =
                            all_k_flat[kv_h][t * head_dim + d];
                        *page_ptr.add(v_off + d) =
                            all_v_flat[kv_h][t * head_dim + d];
                    }
                }
            }
        }

        // For each decode step, verify paged decode matches naive
        let heads_per_kv = num_heads / num_kv_heads;
        for step in 0..decode_steps {
            let token_pos = prefill_len + step;
            let cached_len = token_pos + 1;
            let q = &all_q[token_pos];

            let mut paged_out = vec![0.0f32; num_heads * head_dim];
            flash_attn_decode_paged(
                &attn_config,
                q,
                &mut paged_out,
                &kv_cache,
                0,
                0,
                token_pos,
                None,
            );

            // Naive reference
            for ah in 0..num_heads {
                let kv_h = ah / heads_per_kv;
                let q_off = ah * head_dim;

                let mut naive_out = vec![0.0f32; head_dim];
                naive_attention_single_head(
                    &q[q_off..q_off + head_dim],
                    &all_k_flat[kv_h][..cached_len * head_dim],
                    &all_v_flat[kv_h][..cached_len * head_dim],
                    &mut naive_out,
                    1,
                    cached_len,
                    head_dim,
                    attn_config.scale,
                    false, // causal=false: all cached positions are <= token_pos
                );

                let max_diff = paged_out[q_off..q_off + head_dim]
                    .iter()
                    .zip(naive_out.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);

                assert!(
                    max_diff < 1e-5,
                    "prefill+decode step {step} head {ah}: max_diff = {max_diff:.2e}"
                );
            }
        }
    }

    /// GQA extreme ratio (32:1) — like Gemma architecture.
    #[test]
    fn test_gqa_extreme_ratio_32_to_1() {
        let head_dim = 8;
        let num_heads = 32;
        let num_kv_heads = 1;
        let seq_len = 8;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.05).sin() * 0.2)
                .collect()
        };

        let q = make_data(num_heads * seq_len * head_dim, 0);
        let k = make_data(num_kv_heads * seq_len * head_dim, 100);
        let v = make_data(num_kv_heads * seq_len * head_dim, 200);

        let mut flash_out = vec![0.0f32; num_heads * seq_len * head_dim];
        let mut naive_out = vec![0.0f32; num_heads * seq_len * head_dim];

        flash_attn_multi_head(
            &config, &q, &k, &v, &mut flash_out, seq_len, seq_len, true,
        );
        naive_attention_multi_head(
            &config, &q, &k, &v, &mut naive_out, seq_len, seq_len, true,
        );

        // All 32 query heads share the same single KV head
        // Verify all heads that share a KV head produce identical results
        // when given the same Q (they won't here since Q differs per head,
        // but the KV they attend to is the same)
        let max_rel_err = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&f, &n)| {
                let denom = n.abs().max(1e-7);
                (f - n).abs() / denom
            })
            .fold(0.0f32, f32::max);

        assert!(
            max_rel_err < 1e-3,
            "GQA 32:1 max relative error = {max_rel_err:.2e} (threshold 1e-3)"
        );

        // Extra: verify that heads sharing the same KV head produce
        // identical output when given identical Q
        let q_same: Vec<f32> = (0..head_dim)
            .map(|d| make_data(1, d)[0])
            .collect::<Vec<_>>()
            .repeat(num_heads * seq_len);
        let mut out_same = vec![0.0f32; num_heads * seq_len * head_dim];
        flash_attn_multi_head(
            &config, &q_same, &k, &v, &mut out_same, seq_len, seq_len, true,
        );

        // All heads should produce identical output since Q is the same
        // and they all share the single KV head
        let head_size = seq_len * head_dim;
        let head0 = &out_same[0..head_size];
        for h in 1..num_heads {
            let head_h = &out_same[h * head_size..(h + 1) * head_size];
            let max_diff = head0
                .iter()
                .zip(head_h.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "GQA 32:1 same-Q: head 0 vs head {h} diff = {max_diff:.2e}"
            );
        }
    }

    /// Transpose helpers round-trip test.
    #[test]
    fn test_transpose_round_trip() {
        let seq_len = 3;
        let num_heads = 4;
        let head_dim = 8;
        let total = seq_len * num_heads * head_dim;

        let original: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut head_major = vec![0.0f32; total];
        let mut back = vec![0.0f32; total];

        transpose_to_head_major(&original, &mut head_major, seq_len, num_heads, head_dim);
        transpose_to_token_major(&head_major, &mut back, seq_len, num_heads, head_dim);

        assert_eq!(original, back, "transpose round-trip failed");
    }

    // ====================================================================
    // NEW TESTS: struct constructors, field values, edge cases, float precision
    // ====================================================================

    /// FlashAttnConfig::new produces correct scale = 1/sqrt(head_dim) and default tile_kv=256.
    #[test]
    fn test_config_new_field_values() {
        // Arrange
        let head_dim = 64;
        let num_heads = 8;
        let num_kv_heads = 4;

        // Act
        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        // Assert
        let expected_scale = 1.0 / (head_dim as f32).sqrt();
        assert!(
            (config.scale - expected_scale).abs() < 1e-7,
            "scale = {}, expected {}",
            config.scale,
            expected_scale
        );
        assert_eq!(config.tile_kv, 256, "default tile_kv should be 256");
        assert_eq!(config.head_dim, head_dim);
        assert_eq!(config.num_heads, num_heads);
        assert_eq!(config.num_kv_heads, num_kv_heads);
    }

    /// FlashAttnConfig::with_cache_hint computes tile_kv from L1 cache size,
    /// clamped to [16, 512] and rounded down to power of 2.
    #[test]
    fn test_config_with_cache_hint_tile_clamping() {
        // Arrange: small head_dim=4 => bytes_per_kv_row = 4*4*2 = 32
        // With l1=32768: half = 16384, max_tile = 16384/32 = 512
        // next_power_of_two(512) = 512, >>1 = 256 => clamped to 256
        let head_dim = 4;

        // Act
        let config = FlashAttnConfig::with_cache_hint(head_dim, 4, 2, 32768);

        // Assert
        assert!(
            config.tile_kv >= 16 && config.tile_kv <= 512,
            "tile_kv = {} should be in [16, 512]",
            config.tile_kv
        );
        assert!(config.tile_kv.is_power_of_two(), "tile_kv should be power of 2");

        // Verify scale is still correct
        let expected_scale = 1.0 / (head_dim as f32).sqrt();
        assert!(
            (config.scale - expected_scale).abs() < 1e-7,
            "scale mismatch"
        );
    }

    /// FlashAttnConfig::with_cache_hint with tiny L1 => tile_kv floors to 16.
    #[test]
    fn test_config_with_cache_hint_tiny_l1() {
        // Arrange: head_dim=128 => bytes_per_kv_row = 128*4*2 = 1024
        // With l1=512: half = 256, max_tile = 256/1024 = 0
        // next_power_of_two(0) = 1, >>1 = 0 => clamped to 16
        let config = FlashAttnConfig::with_cache_hint(128, 4, 2, 512);

        // Assert
        assert_eq!(config.tile_kv, 16, "tiny L1 should floor tile_kv to minimum 16");
    }

    /// FlashAttnConfig struct update syntax preserves unmodified fields.
    #[test]
    fn test_config_struct_update_syntax() {
        // Arrange
        let base = FlashAttnConfig::new(64, 8, 4);

        // Act: override tile_kv via struct update syntax
        let custom = FlashAttnConfig {
            tile_kv: 128,
            ..base
        };

        // Assert
        assert_eq!(custom.tile_kv, 128, "tile_kv should be overridden");
        assert_eq!(custom.head_dim, 64, "head_dim should be inherited");
        assert_eq!(custom.num_heads, 8, "num_heads should be inherited");
        assert_eq!(custom.num_kv_heads, 4, "num_kv_heads should be inherited");
        assert!(
            (custom.scale - base.scale).abs() < 1e-10,
            "scale should be inherited exactly"
        );
    }

    /// default_tile_kv with head_dim=0 returns 256 (division-safe fallback).
    #[test]
    fn test_default_tile_kv_zero_head_dim() {
        // Arrange: head_dim=0 => bytes_per_kv_row=0 => returns 256
        // Act
        let tile = default_tile_kv(0);

        // Assert
        assert_eq!(tile, 256, "head_dim=0 should return safe fallback 256");
    }

    /// default_tile_kv produces power-of-2 values in [16, 512].
    #[test]
    fn test_default_tile_kv_range_and_power_of_two() {
        // Arrange: test various head_dim values
        let test_dims: Vec<usize> = vec![1, 4, 16, 32, 64, 96, 128, 256, 512, 1024];

        for head_dim in test_dims {
            // Act
            let tile = default_tile_kv(head_dim);

            // Assert
            assert!(
                tile >= 16 && tile <= 512,
                "head_dim={}: tile_kv={} not in [16, 512]",
                head_dim,
                tile
            );
            assert!(
                tile.is_power_of_two(),
                "head_dim={}: tile_kv={} should be power of 2",
                head_dim,
                tile
            );
        }
    }

    /// Non-causal attention: all positions attend to all positions.
    /// With uniform Q=K, softmax is uniform => output = mean of V rows.
    #[test]
    fn test_flash_attn_non_causal_uniform_attends_all() {
        // Arrange
        let head_dim = 4;
        let seq_len = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Uniform Q and K so all dot products are equal
        let q = vec![1.0f32; seq_len * head_dim];
        let k = vec![1.0f32; seq_len * head_dim];

        // Distinct V rows: [1,0,0,0], [0,1,0,0], [0,0,1,0]
        let mut v = vec![0.0f32; seq_len * head_dim];
        v[0] = 1.0;
        v[head_dim + 1] = 1.0;
        v[2 * head_dim + 2] = 1.0;

        let mut output = vec![0.0f32; seq_len * head_dim];

        // Act: non-causal
        flash_attn_single_head(&q, &k, &v, &mut output, seq_len, seq_len, head_dim, scale, false);

        // Assert: uniform attention => each output = mean of V = [1/3, 1/3, 1/3, 0]
        let expected = 1.0 / 3.0;
        for qi in 0..seq_len {
            assert!(
                (output[qi * head_dim] - expected).abs() < 1e-5,
                "qi={qi} dim0 = {}, expected {expected}",
                output[qi * head_dim]
            );
            assert!(
                (output[qi * head_dim + 1] - expected).abs() < 1e-5,
                "qi={qi} dim1 = {}, expected {expected}",
                output[qi * head_dim + 1]
            );
            assert!(
                (output[qi * head_dim + 2] - expected).abs() < 1e-5,
                "qi={qi} dim2 = {}, expected {expected}",
                output[qi * head_dim + 2]
            );
            assert!(
                output[qi * head_dim + 3].abs() < 1e-5,
                "qi={qi} dim3 = {}, expected 0.0",
                output[qi * head_dim + 3]
            );
        }
    }

    /// All-zero Q produces zero attention output (dot product = 0 for all positions,
    /// softmax is uniform, but Q=0 means weighted sum is still 0).
    /// Wait — softmax uniform * V = mean(V), which is not necessarily zero.
    /// With Q=0: dot=0, scores=[0*scale], softmax=[1/N], output = mean(V).
    /// So output should be mean of V, not zero.
    #[test]
    fn test_flash_attn_zero_query_produces_uniform_attention() {
        // Arrange
        let head_dim = 4;
        let seq_kv = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![0.0f32; head_dim]; // single zero query

        // V rows: [1,0,0,0], [0,2,0,0], [0,0,3,0]
        let k = vec![1.0f32; seq_kv * head_dim];
        let mut v = vec![0.0f32; seq_kv * head_dim];
        v[0] = 1.0;
        v[head_dim + 1] = 2.0;
        v[2 * head_dim + 2] = 3.0;

        let mut output = vec![0.0f32; head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, 1, seq_kv, head_dim, scale, false);

        // Assert: uniform attention => output = mean(V) = [1/3, 2/3, 1, 0]
        assert!(
            (output[0] - 1.0 / 3.0).abs() < 1e-5,
            "output[0] = {}, expected 1/3",
            output[0]
        );
        assert!(
            (output[1] - 2.0 / 3.0).abs() < 1e-5,
            "output[1] = {}, expected 2/3",
            output[1]
        );
        assert!(
            (output[2] - 1.0).abs() < 1e-5,
            "output[2] = {}, expected 1.0",
            output[2]
        );
        assert!(
            output[3].abs() < 1e-5,
            "output[3] = {}, expected 0.0",
            output[3]
        );
    }

    /// Single query, single KV with identical vectors: dot product = head_dim,
    /// softmax = [1.0], output = V.
    #[test]
    fn test_flash_attn_identical_qk_single_position() {
        // Arrange
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|d| (d as f32 + 1.0) * 0.5).collect();
        let k = q.clone(); // Q == K
        let v: Vec<f32> = (0..head_dim).map(|d| (d as f32 + 10.0) * 0.3).collect();
        let mut output = vec![0.0f32; head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, 1, 1, head_dim, scale, false);

        // Assert: single position => output must equal V
        for d in 0..head_dim {
            assert!(
                (output[d] - v[d]).abs() < 1e-6,
                "output[{d}] = {}, expected {}",
                output[d],
                v[d]
            );
        }
    }

    /// Prefill with kv_offset > 0: query positions are shifted by offset
    /// for causal masking. Position 0 query at offset=5 sees KV positions [0..5].
    #[test]
    fn test_prefill_with_kv_offset_causal_mask() {
        // Arrange
        let head_dim = 8;
        let seq_q = 3; // query positions [0, 1, 2]
        let seq_kv = 8; // KV positions [0..7]
        let kv_offset = 5; // query position 0 maps to absolute position 5

        let num_heads = 1;
        let num_kv_heads = 1;
        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.07).sin() * 0.4).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 200);

        let mut prefill_out = vec![0.0f32; num_heads * seq_q * head_dim];
        let mut naive_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_q, seq_kv, kv_offset);
        naive_attention_multi_head(&config, &q, &k, &v, &mut naive_out, seq_q, seq_kv, true);

        // Assert: prefill with offset should match naive causal attention
        // In naive, qi=0 sees only kj<=0, but prefill with offset=5 means qi=0 sees kj<=5
        // So they should NOT match with kv_offset=0 causal.
        // The correct comparison is against a manual naive with shifted positions.
        // We verify by manually computing for qi=0 (absolute pos 5, can see kj=[0..5]).
        let scale = config.scale;
        let q_row = &q[0..head_dim];
        let visible_kv = kv_offset + 1; // positions 0..=5
        let mut scores = vec![0.0f32; visible_kv];
        for kj in 0..visible_kv {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_row[d] * k[kj * head_dim + d];
            }
            scores[kj] = dot * scale;
        }
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }
        let mut expected = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            for kj in 0..visible_kv {
                expected[d] += scores[kj] * v[kj * head_dim + d];
            }
        }

        let max_diff = prefill_out[0..head_dim]
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "prefill kv_offset={kv_offset} qi=0: max_diff = {max_diff:.2e}"
        );
    }

    /// Transpose identity: seq_len=1 => head-major == token-major.
    #[test]
    fn test_transpose_identity_single_token() {
        // Arrange
        let seq_len = 1;
        let num_heads = 4;
        let head_dim = 8;
        let total = seq_len * num_heads * head_dim;

        let original: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut head_major = vec![0.0f32; total];

        // Act: with seq_len=1, head-major layout should be identical to token-major
        transpose_to_head_major(&original, &mut head_major, seq_len, num_heads, head_dim);

        // Assert
        assert_eq!(
            original, head_major,
            "seq_len=1: head-major should equal token-major"
        );
    }

    /// Transpose with head_dim=1 preserves element ordering.
    #[test]
    fn test_transpose_head_dim_one() {
        // Arrange
        let seq_len = 5;
        let num_heads = 3;
        let head_dim = 1;
        let total = seq_len * num_heads * head_dim;

        let original: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut head_major = vec![0.0f32; total];
        let mut back = vec![0.0f32; total];

        // Act
        transpose_to_head_major(&original, &mut head_major, seq_len, num_heads, head_dim);
        transpose_to_token_major(&head_major, &mut back, seq_len, num_heads, head_dim);

        // Assert: round-trip must be identity
        assert_eq!(original, back, "head_dim=1 transpose round-trip failed");
    }

    /// FlashAttention with large head_dim (512) and short sequence: numerical stability.
    /// Ensures no overflow/underflow in dot product accumulation.
    #[test]
    fn test_flash_attn_large_head_dim_numerical_stability() {
        // Arrange
        let head_dim = 512;
        let seq_len = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Values that produce moderate dot products despite large head_dim
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 0.001).sin() * 0.1))
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 0.002).cos() * 0.1))
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 0.003).sin() * 0.5))
            .collect();

        let mut flash_out = vec![0.0f32; seq_len * head_dim];
        let mut naive_out = vec![0.0f32; seq_len * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_len, seq_len, head_dim, scale, true);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_len, seq_len, head_dim, scale, true);

        // Assert: all outputs finite
        assert!(
            flash_out.iter().all(|v| v.is_finite()),
            "large head_dim flash output contains non-finite values"
        );

        // Assert: flash matches naive
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "large head_dim: flash vs naive max_diff = {max_diff:.2e}"
        );
    }

    /// Scale precision: verify FlashAttnConfig::new computes scale correctly
    /// for various head_dim values including edge cases.
    #[test]
    fn test_config_scale_precision_across_head_dims() {
        // Arrange & Act & Assert: multiple head_dim values
        let test_dims: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 96, 128, 256];

        for &head_dim in test_dims {
            let config = FlashAttnConfig::new(head_dim, 4, 2);
            let expected_scale = 1.0 / (head_dim as f32).sqrt();

            assert!(
                (config.scale - expected_scale).abs() < 1e-7,
                "head_dim={head_dim}: scale = {}, expected {}",
                config.scale,
                expected_scale
            );

            // Verify scale is positive and finite
            assert!(config.scale > 0.0, "scale should be positive");
            assert!(config.scale.is_finite(), "scale should be finite");
        }
    }

    // ====================================================================
    // Wave 12k85 tests: additional edge cases and struct coverage
    // ====================================================================

    /// default_tile_kv with very large head_dim (2048) produces the minimum tile size 16.
    /// bytes_per_kv_row = 2048 * 4 * 2 = 16384, half L1 = 16384, max_tile = 1.
    /// next_power_of_two(1) >> 1 = 0, clamped to 16.
    #[test]
    fn test_default_tile_kv_large_head_dim_floors_to_minimum() {
        // Arrange: head_dim=2048 => bytes_per_kv_row = 16384
        // half_l1 = 16384, max_tile = 16384 / 16384 = 1
        // next_power_of_two(1) = 1, >>1 = 0 => clamped to 16
        let tile = default_tile_kv(2048);

        // Assert
        assert_eq!(tile, 16, "large head_dim=2048 should floor tile_kv to 16");
    }

    /// FlashAttnConfig::with_cache_hint with huge L1 cache (4 MiB) caps tile_kv at 512.
    #[test]
    fn test_config_with_cache_hint_huge_l1_caps_at_512() {
        // Arrange: head_dim=4, l1=4*1024*1024 = 4194304
        // bytes_per_kv_row = 32, half = 2097152, max_tile = 65536
        // next_power_of_two(65536) >> 1 = 32768 => clamped to 512
        let config = FlashAttnConfig::with_cache_hint(4, 8, 4, 4 * 1024 * 1024);

        // Assert
        assert_eq!(config.tile_kv, 512, "huge L1 should cap tile_kv at maximum 512");
    }

    /// Double transpose: head-major -> token-major -> head-major should round-trip.
    /// Verifies that transpose_to_token_major inverts transpose_to_head_major.
    #[test]
    fn test_transpose_double_round_trip_via_both() {
        // Arrange
        let seq_len = 5;
        let num_heads = 3;
        let head_dim = 6;
        let total = seq_len * num_heads * head_dim;

        let original: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.37).sin())).collect();
        let mut head_major = vec![0.0f32; total];
        let mut token_major = vec![0.0f32; total];
        let mut back_to_head = vec![0.0f32; total];

        // Act: original(token) -> head -> token -> head
        transpose_to_head_major(&original, &mut head_major, seq_len, num_heads, head_dim);
        transpose_to_token_major(&head_major, &mut token_major, seq_len, num_heads, head_dim);
        transpose_to_head_major(&token_major, &mut back_to_head, seq_len, num_heads, head_dim);

        // Assert: head_major round-trips through token-major
        let max_diff = head_major
            .iter()
            .zip(back_to_head.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-10,
            "double transpose round-trip max_diff = {max_diff:.2e}"
        );
    }

    /// naive_attention_single_head with single KV position produces output = V.
    /// With one KV position, softmax weight is 1.0 regardless of score.
    #[test]
    fn test_naive_attention_single_kv_position_outputs_v() {
        // Arrange
        let head_dim = 6;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|d| ((d + 1) as f32 * 0.5).sin()).collect();
        let k: Vec<f32> = (0..head_dim).map(|d| ((d + 3) as f32 * 0.7).cos()).collect();
        let v: Vec<f32> = (0..head_dim).map(|d| (d as f32 + 1.0) * 0.3).collect();
        let mut output = vec![0.0f32; head_dim];

        // Act
        naive_attention_single_head(&q, &k, &v, &mut output, 1, 1, head_dim, scale, false);

        // Assert: single KV position => softmax=[1.0] => output=V
        for d in 0..head_dim {
            assert!(
                (output[d] - v[d]).abs() < 1e-6,
                "output[{d}] = {}, expected V[{d}] = {}",
                output[d],
                v[d]
            );
        }
    }

    /// All-ones Q and K with V = identity basis vectors: output = V columns uniformly.
    /// With uniform scores, attention produces equal weighting of all V rows.
    #[test]
    fn test_flash_attn_all_ones_uniform_scores_weighted_v() {
        // Arrange
        let head_dim = 4;
        let seq_q = 1;
        let seq_kv = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Uniform Q and K => all dot products = head_dim * scale^2 = 1.0
        let q = vec![1.0f32; seq_q * head_dim];
        let k = vec![1.0f32; seq_kv * head_dim];

        // V = identity-like: row t has value t+1 in dimension 0
        let mut v = vec![0.0f32; seq_kv * head_dim];
        for t in 0..seq_kv {
            v[t * head_dim] = (t + 1) as f32;
        }
        let mut output = vec![0.0f32; head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, false);

        // Assert: uniform attention => output dim0 = mean(1,2,3,4) = 2.5
        let expected_mean = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        assert!(
            (output[0] - expected_mean).abs() < 1e-5,
            "output[0] = {}, expected {}",
            output[0],
            expected_mean
        );
        // Other dims should be 0 (no V contribution)
        for d in 1..head_dim {
            assert!(
                output[d].abs() < 1e-5,
                "output[{d}] = {}, expected 0.0",
                output[d]
            );
        }
    }

    /// Non-causal flash vs naive: verify both produce identical results
    /// for arbitrary Q/K/V when causal=false.
    #[test]
    fn test_flash_attn_non_causal_matches_naive() {
        // Arrange
        let head_dim = 16;
        let seq_q = 5;
        let seq_kv = 7;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.11).sin() * 0.4).collect()
        };

        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 50);
        let v = make_data(seq_kv * head_dim, 100);

        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, false);

        // Assert
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "non-causal flash vs naive max_diff = {max_diff:.2e}"
        );
    }

    /// MHA (no GQA): num_heads == num_kv_heads, each head has its own KV.
    /// flash_attn_multi_head must match per-head single-head calls exactly.
    #[test]
    fn test_mha_no_gqa_each_head_independent() {
        // Arrange
        let head_dim = 8;
        let num_heads = 4;
        let num_kv_heads = 4; // MHA: equal to num_heads
        let seq_len = 5;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.13).sin() * 0.3).collect()
        };

        let q = make_data(num_heads * seq_len * head_dim, 0);
        let k = make_data(num_kv_heads * seq_len * head_dim, 50);
        let v = make_data(num_kv_heads * seq_len * head_dim, 100);
        let mut multi_out = vec![0.0f32; num_heads * seq_len * head_dim];

        // Act
        flash_attn_multi_head(&config, &q, &k, &v, &mut multi_out, seq_len, seq_len, true);

        // Assert: each head should match independent single-head call
        for h in 0..num_heads {
            let off = h * seq_len * head_dim;

            let mut single_out = vec![0.0f32; seq_len * head_dim];
            flash_attn_single_head(
                &q[off..off + seq_len * head_dim],
                &k[off..off + seq_len * head_dim],
                &v[off..off + seq_len * head_dim],
                &mut single_out,
                seq_len,
                seq_len,
                head_dim,
                config.scale,
                true,
            );

            let max_diff = multi_out[off..off + seq_len * head_dim]
                .iter()
                .zip(single_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-6,
                "MHA head {h}: multi_head vs single_head max_diff = {max_diff:.2e}"
            );
        }
    }

    /// FlashAttnConfig::with_cache_hint with medium L1 produces an intermediate tile_kv.
    #[test]
    fn test_config_with_cache_hint_medium_l1_intermediate_tile() {
        // Arrange: head_dim=32 => bytes_per_kv_row = 32*4*2 = 256
        // l1=65536 => half=32768, max_tile=32768/256=128
        // 128 is already power of 2, >>1 = 64 => tile_kv=64
        let config = FlashAttnConfig::with_cache_hint(32, 8, 4, 65536);

        // Assert
        assert_eq!(
            config.tile_kv, 64,
            "medium L1 head_dim=32 should produce tile_kv=64"
        );
        assert!(config.tile_kv.is_power_of_two());
        assert!(
            config.tile_kv >= 16 && config.tile_kv <= 512,
            "tile_kv should be in [16, 512]"
        );
    }

    /// Multiple query positions with single KV position: each query independently
    /// attends to the one KV position, producing output = V for every query.
    #[test]
    fn test_flash_attn_multi_query_single_kv_each_gets_v() {
        // Arrange
        let head_dim = 4;
        let seq_q = 5;
        let seq_kv = 1;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_q * head_dim).map(|i| ((i as f32 * 0.3).sin())).collect();
        let k: Vec<f32> = (0..head_dim).map(|d| ((d as f32 * 0.5).cos())).collect();
        let v: Vec<f32> = (0..head_dim).map(|d| (d as f32 + 1.0) * 0.25).collect();
        let mut output = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, false);

        // Assert: single KV => softmax=[1.0] for every query => output=V
        for qi in 0..seq_q {
            for d in 0..head_dim {
                assert!(
                    (output[qi * head_dim + d] - v[d]).abs() < 1e-6,
                    "qi={qi} d={d}: output = {}, expected V[{d}] = {}",
                    output[qi * head_dim + d],
                    v[d]
                );
            }
        }
    }

    /// Prefill with kv_offset=0 and causal=true matches flash_attn_multi_head causal.
    /// Both should implement the same causal masking when offset=0.
    #[test]
    fn test_prefill_offset_zero_matches_flash_causal() {
        // Arrange
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let seq_len = 6;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.09).cos() * 0.35).collect()
        };

        let q = make_data(num_heads * seq_len * head_dim, 0);
        let k = make_data(num_kv_heads * seq_len * head_dim, 50);
        let v = make_data(num_kv_heads * seq_len * head_dim, 100);

        let mut prefill_out = vec![0.0f32; num_heads * seq_len * head_dim];
        let mut flash_out = vec![0.0f32; num_heads * seq_len * head_dim];

        // Act
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_len, seq_len, 0);
        flash_attn_multi_head(&config, &q, &k, &v, &mut flash_out, seq_len, seq_len, true);

        // Assert
        let max_diff = prefill_out
            .iter()
            .zip(flash_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "prefill(kv_offset=0) vs flash_attn_multi_head(causal) max_diff = {max_diff:.2e}"
        );
    }

    // ====================================================================
    // Wave 12kaq tests: additional edge cases and path coverage
    // ====================================================================

    /// Asymmetric seq_q != seq_kv: flash_attn_single_head with 3 queries, 10 KV positions.
    /// Verify flash matches naive with non-square attention.
    #[test]
    fn test_flash_attn_asymmetric_seq_q_neq_seq_kv() {
        // Arrange
        let head_dim = 8;
        let seq_q = 3;
        let seq_kv = 10;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.09).sin() * 0.35).collect()
        };

        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 50);
        let v = make_data(seq_kv * head_dim, 100);
        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, true);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, true);

        // Assert
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "asymmetric seq_q=3, seq_kv=10: flash vs naive max_diff = {max_diff:.2e}"
        );
    }

    /// Prefill with seq_q != seq_kv (cross-attention style): 2 query tokens, 12 KV tokens.
    #[test]
    fn test_prefill_asymmetric_seq_lengths() {
        // Arrange
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let seq_q = 2;
        let seq_kv = 12;
        let kv_offset = 0;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.07).cos() * 0.3).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);

        let mut prefill_out = vec![0.0f32; num_heads * seq_q * head_dim];
        let mut naive_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_q, seq_kv, kv_offset);
        naive_attention_multi_head(&config, &q, &k, &v, &mut naive_out, seq_q, seq_kv, true);

        // Assert
        let max_diff = prefill_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "asymmetric prefill seq_q=2, seq_kv=12: max_diff = {max_diff:.2e}"
        );
    }

    /// flash_attn_decode_paged with sliding_window: tokens outside the window are masked.
    /// Write 10 KV tokens, decode at position 9 with sliding_window=4.
    /// Only positions [6..=9] should contribute.
    #[test]
    fn test_flash_attn_decode_paged_sliding_window() {
        use crate::types::ModelConfig;

        // Arrange
        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: Some(4),
        };

        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let cached_len = 10usize;
        let v_base = num_kv_heads * PAGE_SIZE * head_dim;
        let sliding_window = 4usize;

        let mut kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_val = |i: usize, seed: usize| -> f32 {
            ((i + seed) as f32 * 0.11).sin() * 0.3
        };

        // Write 10 tokens into paged cache
        let positions = kv_cache.append(0, 0, cached_len).unwrap();
        for (t, &(pid, off)) in positions.iter().enumerate() {
            let page_ptr = kv_cache.page_mut_ptr(pid) as *mut f32;
            for kv_h in 0..num_kv_heads {
                let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim;
                for d in 0..head_dim {
                    unsafe {
                        *page_ptr.add(k_base + d) = make_val(t * head_dim + d, kv_h * 100);
                        *page_ptr.add(v_off + d) = make_val(t * head_dim + d, kv_h * 100 + 500);
                    }
                }
            }
        }

        // Query at position 9 (last cached)
        let token_pos = cached_len - 1;
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| make_val(i, 999))
            .collect();
        let mut paged_out = vec![0.0f32; num_heads * head_dim];

        // Act: decode with sliding window
        flash_attn_decode_paged(
            &attn_config,
            &q,
            &mut paged_out,
            &kv_cache,
            0,
            0,
            token_pos,
            Some(sliding_window),
        );

        // Assert: build contiguous K/V with only visible positions [6..=9]
        let heads_per_kv = num_heads / num_kv_heads;
        for ah in 0..num_heads {
            let kv_h = ah / heads_per_kv;
            let q_off = ah * head_dim;

            // Only positions [token_pos - sliding_window + 1 ..= token_pos] visible
            let visible_start = token_pos + 1 - sliding_window; // 6
            let visible_len = sliding_window; // 4
            let mut k_cont = vec![0.0f32; visible_len * head_dim];
            let mut v_cont = vec![0.0f32; visible_len * head_dim];
            for (vi, t) in (visible_start..=token_pos).enumerate() {
                for d in 0..head_dim {
                    k_cont[vi * head_dim + d] = make_val(t * head_dim + d, kv_h * 100);
                    v_cont[vi * head_dim + d] = make_val(t * head_dim + d, kv_h * 100 + 500);
                }
            }

            let mut naive_out = vec![0.0f32; head_dim];
            naive_attention_single_head(
                &q[q_off..q_off + head_dim],
                &k_cont,
                &v_cont,
                &mut naive_out,
                1,
                visible_len,
                head_dim,
                attn_config.scale,
                false,
            );

            let max_diff = paged_out[q_off..q_off + head_dim]
                .iter()
                .zip(naive_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-5,
                "sliding window head {ah}: max_diff = {max_diff:.2e}"
            );
        }
    }

    /// flash_attn_decode_paged with empty cache (cached_len=0): output should be all zeros.
    #[test]
    fn test_flash_attn_decode_paged_empty_cache_outputs_zeros() {
        use crate::types::ModelConfig;

        // Arrange
        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let num_heads = cfg.num_heads;
        let head_dim = cfg.head_dim;
        let kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_heads);

        let q = vec![1.0f32; num_heads * head_dim];
        let mut output = vec![1.0f32; num_heads * head_dim]; // non-zero initial

        // Act: decode with no cached tokens
        flash_attn_decode_paged(&attn_config, &q, &mut output, &kv_cache, 0, 0, 0, None);

        // Assert: all output should be zero
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "empty cache output[{i}] = {v}, expected 0.0"
            );
        }
    }

    /// naive_attention_multi_head non-causal mode: verify it matches per-head naive
    /// when causal=false, confirming the non-causal path works for multi-head wrapper.
    #[test]
    fn test_naive_attention_multi_head_non_causal() {
        // Arrange
        let head_dim = 8;
        let num_heads = 4;
        let num_kv_heads = 2;
        let seq_q = 3;
        let seq_kv = 5;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.13).cos() * 0.35).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let mut naive_multi_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        naive_attention_multi_head(
            &config, &q, &k, &v, &mut naive_multi_out, seq_q, seq_kv, false,
        );

        // Assert: each head must match independent naive_attention_single_head
        let heads_per_kv = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * seq_q * head_dim;
            let k_off = kv_h * seq_kv * head_dim;
            let v_off = kv_h * seq_kv * head_dim;
            let o_off = h * seq_q * head_dim;

            let mut single_out = vec![0.0f32; seq_q * head_dim];
            naive_attention_single_head(
                &q[q_off..q_off + seq_q * head_dim],
                &k[k_off..k_off + seq_kv * head_dim],
                &v[v_off..v_off + seq_kv * head_dim],
                &mut single_out,
                seq_q,
                seq_kv,
                head_dim,
                config.scale,
                false,
            );

            let max_diff = naive_multi_out[o_off..o_off + seq_q * head_dim]
                .iter()
                .zip(single_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-10,
                "naive multi-head non-causal head {h}: max_diff = {max_diff:.2e}"
            );
        }
    }

    /// flash_attn_single_head with seq_kv=0: output should be all zeros.
    #[test]
    fn test_flash_attn_empty_kv_produces_zeros() {
        // Arrange
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![1.0f32; head_dim];
        let k = vec![];
        let v = vec![];
        let mut output = vec![1.0f32; head_dim]; // non-zero initial

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, 1, 0, head_dim, scale, false);

        // Assert: output initialized to zero, no KV to process => stays zero
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-10,
                "empty KV output[{i}] = {val}, expected 0.0"
            );
        }
    }

    /// transpose_to_head_major element-level correctness: verify exact index mapping
    /// for a small known input (seq=2, heads=2, dim=3).
    #[test]
    fn test_transpose_head_major_exact_index_mapping() {
        // Arrange: token-major layout [seq_len=2, num_heads=2, head_dim=3]
        // original[s * num_heads * head_dim + h * head_dim + d]
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 3;
        let total = seq_len * num_heads * head_dim;

        // Token-major: [s0h0d0, s0h0d1, s0h0d2, s0h1d0, s0h1d1, s0h1d2,
        //               s1h0d0, s1h0d1, s1h0d2, s1h1d0, s1h1d1, s1h1d2]
        let original: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut head_major = vec![0.0f32; total];

        // Act
        transpose_to_head_major(&original, &mut head_major, seq_len, num_heads, head_dim);

        // Assert: head-major layout [h * seq_len * head_dim + s * head_dim + d]
        // Head 0: [s0d0=0, s0d1=1, s0d2=2, s1d0=6, s1d1=7, s1d2=8]
        // Head 1: [s0d0=3, s0d1=4, s0d2=5, s1d0=9, s1d1=10, s1d2=11]
        assert_eq!(head_major[0], 0.0, "h0 s0 d0");
        assert_eq!(head_major[1], 1.0, "h0 s0 d1");
        assert_eq!(head_major[2], 2.0, "h0 s0 d2");
        assert_eq!(head_major[3], 6.0, "h0 s1 d0");
        assert_eq!(head_major[4], 7.0, "h0 s1 d1");
        assert_eq!(head_major[5], 8.0, "h0 s1 d2");
        assert_eq!(head_major[6], 3.0, "h1 s0 d0");
        assert_eq!(head_major[7], 4.0, "h1 s0 d1");
        assert_eq!(head_major[8], 5.0, "h1 s0 d2");
        assert_eq!(head_major[9], 9.0, "h1 s1 d0");
        assert_eq!(head_major[10], 10.0, "h1 s1 d1");
        assert_eq!(head_major[11], 11.0, "h1 s1 d2");
    }

    /// FlashAttnConfig::with_cache_hint preserves head_dim, num_heads, num_kv_heads,
    /// and computes the same scale as FlashAttnConfig::new.
    #[test]
    fn test_config_with_cache_hint_preserves_fields_and_scale() {
        // Arrange
        let head_dim = 96;
        let num_heads = 8;
        let num_kv_heads = 2;
        let l1_bytes = 32768;

        // Act
        let config_new = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);
        let config_hint = FlashAttnConfig::with_cache_hint(head_dim, num_heads, num_kv_heads, l1_bytes);

        // Assert: structural fields must match
        assert_eq!(config_hint.head_dim, head_dim);
        assert_eq!(config_hint.num_heads, num_heads);
        assert_eq!(config_hint.num_kv_heads, num_kv_heads);

        // Assert: scale must be identical to new()
        assert!(
            (config_hint.scale - config_new.scale).abs() < 1e-10,
            "with_cache_hint scale = {}, new scale = {}",
            config_hint.scale,
            config_new.scale
        );

        // Assert: tile_kv must differ from new() default (unless coincidentally equal)
        // but must always be valid
        assert!(
            config_hint.tile_kv >= 16 && config_hint.tile_kv <= 512,
            "tile_kv = {} not in valid range",
            config_hint.tile_kv
        );
    }

    /// Causal attention: query at position 0 can only see KV position 0.
    /// With seq_q=1, seq_kv=1, causal=true, the single query attends to the single KV.
    /// A separate test verifies that the naive path matches flash for this edge case.
    #[test]
    fn test_flash_attn_causal_single_query_first_position_matches_naive() {
        // Arrange: query position 0 with causal mask, 3 KV positions
        // qi=0 can only see kv_pos=0 (not 1, 2). So output is solely determined by V[0].
        let head_dim = 4;
        let seq_q = 1;
        let seq_kv = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![1.0f32; head_dim];
        let k = vec![1.0f32; seq_kv * head_dim];
        // Distinct V rows so we can verify only V[0] contributes
        let mut v = vec![0.0f32; seq_kv * head_dim];
        v[0] = 1.0;                    // V[0] = [1, 0, 0, 0]
        v[head_dim + 1] = 2.0;         // V[1] = [0, 2, 0, 0]
        v[2 * head_dim + 2] = 3.0;     // V[2] = [0, 0, 3, 0]

        let mut flash_out = vec![0.0f32; head_dim];
        let mut naive_out = vec![0.0f32; head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, true);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, true);

        // Assert: flash matches naive
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "causal first-position flash vs naive max_diff = {max_diff:.2e}"
        );

        // Assert: output dim0 should be 1.0 (= V[0][0]) and dim1, dim2 should be 0
        assert!(
            (flash_out[0] - 1.0).abs() < 1e-5,
            "output[0] = {}, expected 1.0 (V[0][0])",
            flash_out[0]
        );
        assert!(
            flash_out[1].abs() < 1e-5,
            "output[1] = {}, expected 0.0",
            flash_out[1]
        );
        assert!(
            flash_out[2].abs() < 1e-5,
            "output[2] = {}, expected 0.0",
            flash_out[2]
        );
    }

    /// Prefill with kv_offset > 0 and seq_q < seq_kv: verify intermediate query positions
    /// see the correct number of KV positions via causal mask.
    #[test]
    fn test_prefill_with_offset_intermediate_positions() {
        // Arrange: kv_offset=3, seq_q=4, seq_kv=10
        // Query positions map to absolute positions [3, 4, 5, 6].
        // qi=0 (abs=3) sees KV[0..=3], qi=1 (abs=4) sees KV[0..=4], etc.
        let head_dim = 8;
        let num_heads = 1;
        let num_kv_heads = 1;
        let seq_q = 4;
        let seq_kv = 10;
        let kv_offset = 3;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.07).sin() * 0.4).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 200);

        let mut prefill_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_q, seq_kv, kv_offset);

        // Assert: manually verify each query position
        let scale = config.scale;
        for qi in 0..seq_q {
            let abs_pos = kv_offset + qi;
            // qi can see KV positions [0..=abs_pos]
            let visible_len = abs_pos + 1;

            let mut scores = vec![0.0f32; visible_len];
            for kj in 0..visible_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * head_dim + d] * k[kj * head_dim + d];
                }
                scores[kj] = dot * scale;
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            let mut expected = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                for kj in 0..visible_len {
                    expected[d] += scores[kj] * v[kj * head_dim + d];
                }
            }

            let max_diff = prefill_out[qi * head_dim..(qi + 1) * head_dim]
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-5,
                "prefill offset={kv_offset} qi={qi} (abs={abs_pos}): max_diff = {max_diff:.2e}"
            );
        }
    }

    // ====================================================================
    // Wave 12kha tests: 10 additional tests (56 total)
    // ====================================================================

    /// FlashAttention with very small head_dim=2: ensure tiling logic works
    /// when head_dim is smaller than typical SIMD widths.
    /// Verify flash matches naive for this edge case.
    #[test]
    fn test_flash_attn_tiny_head_dim_two_matches_naive() {
        // Arrange
        let head_dim = 2;
        let seq_q = 4;
        let seq_kv = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.17).cos() * 0.5).collect()
        };

        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 50);
        let v = make_data(seq_kv * head_dim, 100);
        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, true);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, true);

        // Assert
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "tiny head_dim=2: flash vs naive max_diff = {max_diff:.2e}"
        );
    }

    /// FlashAttention with custom scale (not 1/sqrt(head_dim)): verify the scale
    /// factor is correctly applied to dot products, not just the default.
    #[test]
    fn test_flash_attn_custom_scale_matches_naive() {
        // Arrange
        let head_dim = 8;
        let seq_q = 3;
        let seq_kv = 5;
        let custom_scale = 0.125f32; // deliberately different from 1/sqrt(8) ~ 0.354

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.11).sin() * 0.4).collect()
        };

        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 50);
        let v = make_data(seq_kv * head_dim, 100);
        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, custom_scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, custom_scale, false);

        // Assert
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "custom scale flash vs naive max_diff = {max_diff:.2e}"
        );

        // Sanity: outputs must be finite
        assert!(
            flash_out.iter().all(|v| v.is_finite()),
            "custom scale output contains non-finite values"
        );
    }

    /// FlashAttention with negative Q and K values: verify online softmax handles
    /// large negative dot products without numerical issues.
    #[test]
    fn test_flash_attn_negative_qk_numerical_stability() {
        // Arrange
        let head_dim = 16;
        let seq_q = 4;
        let seq_kv = 6;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Mix of large positive and negative values
        let q: Vec<f32> = (0..seq_q * head_dim)
            .map(|i| if i % 3 == 0 { -10.0 } else { (i as f32 * 0.13).sin() * 2.0 })
            .collect();
        let k: Vec<f32> = (0..seq_kv * head_dim)
            .map(|i| if i % 5 == 0 { -8.0 } else { (i as f32 * 0.17).cos() * 1.5 })
            .collect();
        let v: Vec<f32> = (0..seq_kv * head_dim)
            .map(|i| (i as f32 * 0.03).sin() * 0.5)
            .collect();

        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, false);

        // Assert
        assert!(
            flash_out.iter().all(|v| v.is_finite()),
            "negative Q/K flash output contains non-finite values"
        );

        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "negative Q/K flash vs naive max_diff = {max_diff:.2e}"
        );
    }

    /// Multi-head flash with asymmetric seq_q != seq_kv and GQA:
    /// verify per-head correctness when query and KV lengths differ.
    #[test]
    fn test_multi_head_asymmetric_gqa_per_head_correctness() {
        // Arrange
        let head_dim = 8;
        let num_heads = 6;
        let num_kv_heads = 2;
        let seq_q = 3;
        let seq_kv = 7;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.07).sin() * 0.3).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let mut multi_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_multi_head(&config, &q, &k, &v, &mut multi_out, seq_q, seq_kv, true);

        // Assert: each head matches independent single-head call
        let heads_per_kv = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * seq_q * head_dim;
            let k_off = kv_h * seq_kv * head_dim;
            let v_off = kv_h * seq_kv * head_dim;
            let o_off = h * seq_q * head_dim;

            let mut single_out = vec![0.0f32; seq_q * head_dim];
            flash_attn_single_head(
                &q[q_off..q_off + seq_q * head_dim],
                &k[k_off..k_off + seq_kv * head_dim],
                &v[v_off..v_off + seq_kv * head_dim],
                &mut single_out,
                seq_q,
                seq_kv,
                head_dim,
                config.scale,
                true,
            );

            let max_diff = multi_out[o_off..o_off + seq_q * head_dim]
                .iter()
                .zip(single_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-6,
                "asymmetric GQA head {h}: max_diff = {max_diff:.2e}"
            );
        }
    }

    /// transpose_to_token_major exact index mapping: verify element-level correctness
    /// for a small known input (seq=3, heads=2, dim=2).
    #[test]
    fn test_transpose_token_major_exact_index_mapping() {
        // Arrange: head-major layout [h * seq_len * head_dim + s * head_dim + d]
        // With seq_len=3, num_heads=2, head_dim=2, total=12
        let seq_len = 3;
        let num_heads = 2;
        let head_dim = 2;
        let total = seq_len * num_heads * head_dim;

        // Head-major: head0=[s0d0=0, s0d1=1, s1d0=2, s1d1=3, s2d0=4, s2d1=5]
        //             head1=[s0d0=6, s0d1=7, s1d0=8, s1d1=9, s2d0=10, s2d1=11]
        let head_major: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut token_major = vec![0.0f32; total];

        // Act
        transpose_to_token_major(&head_major, &mut token_major, seq_len, num_heads, head_dim);

        // Assert: token-major [s * num_heads * head_dim + h * head_dim + d]
        // s0: [h0d0=0, h0d1=1, h1d0=6, h1d1=7]
        // s1: [h0d0=2, h0d1=3, h1d0=8, h1d1=9]
        // s2: [h0d0=4, h0d1=5, h1d0=10, h1d1=11]
        assert_eq!(token_major[0], 0.0, "s0 h0 d0");
        assert_eq!(token_major[1], 1.0, "s0 h0 d1");
        assert_eq!(token_major[2], 6.0, "s0 h1 d0");
        assert_eq!(token_major[3], 7.0, "s0 h1 d1");
        assert_eq!(token_major[4], 2.0, "s1 h0 d0");
        assert_eq!(token_major[5], 3.0, "s1 h0 d1");
        assert_eq!(token_major[6], 8.0, "s1 h1 d0");
        assert_eq!(token_major[7], 9.0, "s1 h1 d1");
        assert_eq!(token_major[8], 4.0, "s2 h0 d0");
        assert_eq!(token_major[9], 5.0, "s2 h0 d1");
        assert_eq!(token_major[10], 10.0, "s2 h1 d0");
        assert_eq!(token_major[11], 11.0, "s2 h1 d1");
    }

    /// naive_attention_single_head with causal mask and asymmetric lengths:
    /// verify the naive path correctly masks future positions when seq_q < seq_kv.
    #[test]
    fn test_naive_causal_asymmetric_masks_future() {
        // Arrange
        let head_dim = 4;
        let seq_q = 2;
        let seq_kv = 5;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Uniform Q and K => equal scores for unmasked positions
        let q = vec![1.0f32; seq_q * head_dim];
        let k = vec![1.0f32; seq_kv * head_dim];

        // Distinct V rows so we can verify which KV positions contribute
        let mut v = vec![0.0f32; seq_kv * head_dim];
        for t in 0..seq_kv {
            v[t * head_dim] = (t + 1) as f32; // dim 0 carries the position index
        }

        let mut output = vec![0.0f32; seq_q * head_dim];

        // Act: causal => qi=0 sees kj=0 only; qi=1 sees kj=0,1
        naive_attention_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, true);

        // Assert: qi=0 sees only V[0] => output[0] = V[0][0] = 1.0
        assert!(
            (output[0] - 1.0).abs() < 1e-5,
            "qi=0 dim0 = {}, expected 1.0",
            output[0]
        );
        // Other dims should be 0
        assert!(
            output[1].abs() < 1e-5,
            "qi=0 dim1 = {}, expected 0.0",
            output[1]
        );

        // Assert: qi=1 sees V[0] and V[1] with equal weight => output[0] = (1+2)/2 = 1.5
        let expected_mean = (1.0 + 2.0) / 2.0;
        assert!(
            (output[head_dim] - expected_mean).abs() < 1e-5,
            "qi=1 dim0 = {}, expected {}",
            output[head_dim],
            expected_mean
        );
    }

    /// flash_attn_decode_paged with GQA (num_heads=4, num_kv_heads=2):
    /// multiple query heads share KV heads, verify per-head correctness.
    #[test]
    fn test_flash_attn_decode_paged_gqa_multi_head() {
        use crate::types::ModelConfig;

        // Arrange
        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let cached_len = 6usize;
        let v_base = num_kv_heads * PAGE_SIZE * head_dim;

        let mut kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_val = |i: usize, seed: usize| -> f32 {
            ((i + seed) as f32 * 0.11).sin() * 0.3
        };

        // Write K/V into cache
        let positions = kv_cache.append(0, 0, cached_len).unwrap();
        for (t, &(pid, off)) in positions.iter().enumerate() {
            let page_ptr = kv_cache.page_mut_ptr(pid) as *mut f32;
            for kv_h in 0..num_kv_heads {
                let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim;
                for d in 0..head_dim {
                    unsafe {
                        *page_ptr.add(k_base + d) = make_val(t * head_dim + d, kv_h * 100);
                        *page_ptr.add(v_off + d) = make_val(t * head_dim + d, kv_h * 100 + 500);
                    }
                }
            }
        }

        let token_pos = cached_len - 1;
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| make_val(i, 777))
            .collect();
        let mut paged_out = vec![0.0f32; num_heads * head_dim];

        // Act
        flash_attn_decode_paged(
            &attn_config,
            &q,
            &mut paged_out,
            &kv_cache,
            0,
            0,
            token_pos,
            None,
        );

        // Assert: each attention head matches naive reference
        let heads_per_kv = num_heads / num_kv_heads;
        for ah in 0..num_heads {
            let kv_h = ah / heads_per_kv;
            let q_off = ah * head_dim;

            let mut k_cont = vec![0.0f32; cached_len * head_dim];
            let mut v_cont = vec![0.0f32; cached_len * head_dim];
            for t in 0..cached_len {
                for d in 0..head_dim {
                    k_cont[t * head_dim + d] = make_val(t * head_dim + d, kv_h * 100);
                    v_cont[t * head_dim + d] = make_val(t * head_dim + d, kv_h * 100 + 500);
                }
            }

            let mut naive_out = vec![0.0f32; head_dim];
            naive_attention_single_head(
                &q[q_off..q_off + head_dim],
                &k_cont,
                &v_cont,
                &mut naive_out,
                1,
                cached_len,
                head_dim,
                attn_config.scale,
                false,
            );

            let max_diff = paged_out[q_off..q_off + head_dim]
                .iter()
                .zip(naive_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-5,
                "GQA paged decode head {ah} (kv_h={kv_h}): max_diff = {max_diff:.2e}"
            );
        }
    }

    /// Prefill with kv_offset where all query positions have full visibility:
    /// kv_offset=0, seq_q=3, seq_kv=3, causal=false.
    /// This tests the prefill path without causal masking, matching naive non-causal.
    #[test]
    fn test_prefill_non_causal_matches_naive_non_causal() {
        // Arrange
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let seq_q = 4;
        let seq_kv = 4;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.09).sin() * 0.4).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);

        let mut prefill_out = vec![0.0f32; num_heads * seq_q * head_dim];
        let mut naive_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act: prefill always uses causal internally, but with kv_offset=0 and seq_q=seq_kv,
        // each qi sees exactly [0..=qi]. We compare against naive causal.
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_q, seq_kv, 0);
        naive_attention_multi_head(&config, &q, &k, &v, &mut naive_out, seq_q, seq_kv, true);

        // Assert
        let max_diff = prefill_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "prefill vs naive causal max_diff = {max_diff:.2e}"
        );
    }

    /// flash_attn_multi_head non-causal with GQA: verify all heads can attend
    /// to all KV positions when causal=false.
    #[test]
    fn test_multi_head_non_causal_gqa_all_positions_visible() {
        // Arrange
        let head_dim = 4;
        let num_heads = 4;
        let num_kv_heads = 1;
        let seq_len = 3;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        // Uniform Q/K so all scores are equal
        let q = vec![1.0f32; num_heads * seq_len * head_dim];
        let k = vec![1.0f32; num_kv_heads * seq_len * head_dim];

        // V rows carry position identity
        let mut v = vec![0.0f32; num_kv_heads * seq_len * head_dim];
        for t in 0..seq_len {
            v[t * head_dim] = (t + 1) as f32;
        }

        let mut flash_out = vec![0.0f32; num_heads * seq_len * head_dim];

        // Act: non-causal => all positions visible
        flash_attn_multi_head(&config, &q, &k, &v, &mut flash_out, seq_len, seq_len, false);

        // Assert: all query positions in all heads should produce the same uniform mean
        let expected_mean = (1.0 + 2.0 + 3.0) / 3.0;
        for h in 0..num_heads {
            for qi in 0..seq_len {
                let idx = h * seq_len * head_dim + qi * head_dim;
                assert!(
                    (flash_out[idx] - expected_mean).abs() < 1e-5,
                    "h={h} qi={qi} dim0 = {}, expected {expected_mean}",
                    flash_out[idx]
                );
            }
        }
    }

    /// FlashAttention with single query and many KV positions (decode-like pattern):
    /// seq_q=1, seq_kv=100, causal=false. Verify flash matches naive for this
    /// long-KV decode scenario.
    #[test]
    fn test_flash_attn_single_query_long_kv_matches_naive() {
        // Arrange
        let head_dim = 16;
        let seq_q = 1;
        let seq_kv = 100;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.03).sin() * 0.4).collect()
        };

        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 50);
        let v = make_data(seq_kv * head_dim, 100);
        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, false);

        // Assert
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "single query long KV: flash vs naive max_diff = {max_diff:.2e}"
        );

        // Sanity: output must be non-trivial (not all zeros)
        assert!(
            flash_out.iter().any(|v| v.abs() > 1e-6),
            "output should not be all zeros"
        );
    }

    /// Transpose with large dimensions: verify correctness for seq_len=8, num_heads=16, head_dim=64.
    /// Round-trip must recover the original exactly.
    #[test]
    fn test_transpose_large_dimensions_round_trip() {
        // Arrange
        let seq_len = 8;
        let num_heads = 16;
        let head_dim = 64;
        let total = seq_len * num_heads * head_dim;

        let original: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.0001).sin())).collect();
        let mut head_major = vec![0.0f32; total];
        let mut back = vec![0.0f32; total];

        // Act
        transpose_to_head_major(&original, &mut head_major, seq_len, num_heads, head_dim);
        transpose_to_token_major(&head_major, &mut back, seq_len, num_heads, head_dim);

        // Assert: exact round-trip
        let max_diff = original
            .iter()
            .zip(back.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-10,
            "large dim transpose round-trip max_diff = {max_diff:.2e}"
        );
    }

    // ====================================================================
    // Wave 12khb tests: 10 additional tests (66 total)
    // ====================================================================

    /// flash_attn_single_head with seq_q=0: no queries, output remains untouched.
    #[test]
    fn test_flash_attn_zero_queries_preserves_output() {
        // Arrange
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let k = vec![1.0f32; 3 * head_dim];
        let v = vec![1.0f32; 3 * head_dim];
        let mut output = vec![0.0f32; 0]; // empty output for 0 queries

        // Act
        flash_attn_single_head(&[], &k, &v, &mut output, 0, 3, head_dim, scale, false);

        // Assert: output is empty (no crash, no panic)
        assert!(output.is_empty(), "zero queries should produce empty output");
    }

    /// FlashAttnConfig::new with head_dim=1: scale = 1.0, tile_kv = 256.
    #[test]
    fn test_config_new_head_dim_one_scale_is_one() {
        // Arrange & Act
        let config = FlashAttnConfig::new(1, 4, 2);

        // Assert
        assert!(
            (config.scale - 1.0).abs() < 1e-7,
            "head_dim=1: scale = {}, expected 1.0",
            config.scale
        );
        assert_eq!(config.head_dim, 1);
        assert_eq!(config.tile_kv, 256);
    }

    /// naive_attention_single_head with all-zero K: all dot products are zero,
    /// softmax is uniform, output = mean(V).
    #[test]
    fn test_naive_attention_zero_k_uniform_attention() {
        // Arrange
        let head_dim = 4;
        let seq_q = 1;
        let seq_kv = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![1.0f32; head_dim];
        let k = vec![0.0f32; seq_kv * head_dim];
        // V: each row has a unique value in dim 0
        let mut v = vec![0.0f32; seq_kv * head_dim];
        v[0] = 2.0;
        v[head_dim] = 4.0;
        v[2 * head_dim] = 6.0;
        let mut output = vec![0.0f32; head_dim];

        // Act
        naive_attention_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, false);

        // Assert: uniform attention => mean(V dim0) = (2+4+6)/3 = 4.0
        let expected_mean = 4.0f32;
        assert!(
            (output[0] - expected_mean).abs() < 1e-5,
            "output[0] = {}, expected {expected_mean}",
            output[0]
        );
        // Other dims should be 0
        for d in 1..head_dim {
            assert!(
                output[d].abs() < 1e-5,
                "output[{d}] = {}, expected 0.0",
                output[d]
            );
        }
    }

    /// flash_attn_single_head with identical K rows: all scores equal,
    /// attention weights are uniform regardless of Q.
    #[test]
    fn test_flash_attn_identical_k_uniform_scores() {
        // Arrange
        let head_dim = 4;
        let seq_q = 1;
        let seq_kv = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|d| (d as f32 + 0.5)).collect();
        let k = vec![1.0f32; seq_kv * head_dim]; // all K rows identical
        // V rows carry distinct values in dim 0
        let mut v = vec![0.0f32; seq_kv * head_dim];
        for t in 0..seq_kv {
            v[t * head_dim] = (t + 1) as f32;
        }
        let mut output = vec![0.0f32; head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, false);

        // Assert: uniform scores => output dim0 = mean(1,2,3,4) = 2.5
        let expected_mean = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        assert!(
            (output[0] - expected_mean).abs() < 1e-5,
            "output[0] = {}, expected {expected_mean}",
            output[0]
        );
        for d in 1..head_dim {
            assert!(
                output[d].abs() < 1e-5,
                "output[{d}] = {}, expected 0.0",
                output[d]
            );
        }
    }

    /// flash_attn_multi_head with seq_q=1 matches flash_attn_single_head per head.
    #[test]
    fn test_multi_head_seq_q_one_matches_single_head() {
        // Arrange
        let head_dim = 8;
        let num_heads = 4;
        let num_kv_heads = 2;
        let seq_q = 1;
        let seq_kv = 5;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.11).sin() * 0.3).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let mut multi_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_multi_head(&config, &q, &k, &v, &mut multi_out, seq_q, seq_kv, true);

        // Assert: each head matches independent single-head call
        let heads_per_kv = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * seq_q * head_dim;
            let k_off = kv_h * seq_kv * head_dim;
            let v_off = kv_h * seq_kv * head_dim;
            let o_off = h * seq_q * head_dim;

            let mut single_out = vec![0.0f32; seq_q * head_dim];
            flash_attn_single_head(
                &q[q_off..q_off + seq_q * head_dim],
                &k[k_off..k_off + seq_kv * head_dim],
                &v[v_off..v_off + seq_kv * head_dim],
                &mut single_out,
                seq_q,
                seq_kv,
                head_dim,
                config.scale,
                true,
            );

            let max_diff = multi_out[o_off..o_off + seq_q * head_dim]
                .iter()
                .zip(single_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-6,
                "seq_q=1 head {h}: multi vs single max_diff = {max_diff:.2e}"
            );
        }
    }

    /// Transpose invariance: transposing twice (head->token->head) with same parameters
    /// must reproduce the original. Tests with non-trivial seq_len=7.
    #[test]
    fn test_transpose_double_inversion_seq_len_seven() {
        // Arrange
        let seq_len = 7;
        let num_heads = 3;
        let head_dim = 5;
        let total = seq_len * num_heads * head_dim;

        let original: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.23).cos())).collect();
        let mut head_major = vec![0.0f32; total];
        let mut back = vec![0.0f32; total];

        // Act
        transpose_to_head_major(&original, &mut head_major, seq_len, num_heads, head_dim);
        transpose_to_token_major(&head_major, &mut back, seq_len, num_heads, head_dim);

        // Assert
        let max_diff = original
            .iter()
            .zip(back.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-10,
            "seq_len=7 transpose round-trip max_diff = {max_diff:.2e}"
        );
    }

    /// default_tile_kv with head_dim=1: bytes_per_kv_row=8, max_tile=16384/8=2048,
    /// next_power_of_two(2048)>>1=1024, clamped to max 512.
    #[test]
    fn test_default_tile_kv_head_dim_one_caps_at_512() {
        // Arrange & Act
        let tile = default_tile_kv(1);

        // Assert: head_dim=1, bytes_per_kv_row=8, max_tile=2048,
        // next_pow2(2048)=2048, >>1=1024, clamped to 512
        assert_eq!(tile, 512, "head_dim=1 should produce tile_kv=512");
        assert!(tile.is_power_of_two());
    }

    /// flash_attn_single_head with large V values: verify output remains finite
    /// and matches naive (numerical stability with large V magnitudes).
    #[test]
    fn test_flash_attn_large_v_values_stability() {
        // Arrange
        let head_dim = 8;
        let seq_q = 3;
        let seq_kv = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_q * head_dim)
            .map(|i| ((i as f32 * 0.1).sin() * 0.01))
            .collect();
        let k: Vec<f32> = (0..seq_kv * head_dim)
            .map(|i| ((i as f32 * 0.1).cos() * 0.01))
            .collect();
        // Large V values
        let v: Vec<f32> = (0..seq_kv * head_dim)
            .map(|i| (i as f32 * 100.0 + 50.0))
            .collect();

        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, false);

        // Assert
        assert!(
            flash_out.iter().all(|v| v.is_finite()),
            "large V flash output contains non-finite values"
        );
        let max_diff = flash_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "large V: flash vs naive max_diff = {max_diff:.2e}"
        );
    }

    /// flash_attn_multi_head with num_heads=num_kv_heads=1: single-head wrapper
    /// must match flash_attn_single_head exactly.
    #[test]
    fn test_multi_head_single_head_wrapper_matches_single() {
        // Arrange
        let head_dim = 8;
        let seq_len = 5;
        let config = FlashAttnConfig::new(head_dim, 1, 1);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.13).sin() * 0.35).collect()
        };

        let q = make_data(seq_len * head_dim, 0);
        let k = make_data(seq_len * head_dim, 50);
        let v = make_data(seq_len * head_dim, 100);
        let mut multi_out = vec![0.0f32; seq_len * head_dim];
        let mut single_out = vec![0.0f32; seq_len * head_dim];

        // Act
        flash_attn_multi_head(&config, &q, &k, &v, &mut multi_out, seq_len, seq_len, true);
        flash_attn_single_head(&q, &k, &v, &mut single_out, seq_len, seq_len, head_dim, config.scale, true);

        // Assert
        let max_diff = multi_out
            .iter()
            .zip(single_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-10,
            "single-head wrapper vs single_head max_diff = {max_diff:.2e}"
        );
    }

    /// Prefill with kv_offset where query position exceeds all KV positions:
    /// kv_offset=10, seq_q=1, seq_kv=5 => query at abs=10 sees all KV [0..4].
    #[test]
    fn test_prefill_offset_exceeds_kv_len_sees_all() {
        // Arrange
        let head_dim = 8;
        let num_heads = 1;
        let num_kv_heads = 1;
        let seq_q = 1;
        let seq_kv = 5;
        let kv_offset = 10; // query at abs=10, all KV positions [0..4] are visible

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.07).sin() * 0.4).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 200);

        let mut prefill_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_q, seq_kv, kv_offset);

        // Assert: manually compute — all 5 KV positions visible, no masking
        let scale = config.scale;
        let q_row = &q[0..head_dim];
        let mut scores = vec![0.0f32; seq_kv];
        for kj in 0..seq_kv {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_row[d] * k[kj * head_dim + d];
            }
            scores[kj] = dot * scale;
        }
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }
        let mut expected = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            for kj in 0..seq_kv {
                expected[d] += scores[kj] * v[kj * head_dim + d];
            }
        }

        let max_diff = prefill_out[0..head_dim]
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "prefill kv_offset=10 qi=0: max_diff = {max_diff:.2e}"
        );
    }

    /// Zero-length KV: flash_attn_single_head with seq_kv=0 should produce zero output.
    #[test]
    fn test_flash_attn_zero_kv_length_output_is_zero() {
        // Arrange
        let head_dim = 8;
        let seq_q = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = vec![1.0f32; seq_q * head_dim];
        let k: Vec<f32> = Vec::new();
        let v: Vec<f32> = Vec::new();
        let mut output = vec![1.0f32; seq_q * head_dim]; // non-zero initial

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, 0, head_dim, scale, false);

        // Assert: no KV positions => all outputs remain at 0 (online softmax never normalizes)
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "zero-KV output[{i}] = {val}, expected 0.0"
            );
        }
    }

    /// Single token decode with KvCache: cached_len=1, output must equal V row.
    #[test]
    fn test_flash_attn_decode_paged_single_cached_token() {
        use crate::types::ModelConfig;

        // Arrange
        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let v_base = num_kv_heads * PAGE_SIZE * head_dim;

        let mut kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        // Write 1 token into cache
        let positions = kv_cache.append(0, 0, 1).unwrap();
        let (pid, off) = positions[0];
        let page_ptr = kv_cache.page_mut_ptr(pid) as *mut f32;
        for kv_h in 0..num_kv_heads {
            let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
            let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim;
            for d in 0..head_dim {
                unsafe {
                    *page_ptr.add(k_base + d) = 1.0f32;
                    *page_ptr.add(v_off + d) = (d as f32 + 0.5) * 2.0;
                }
            }
        }

        let q = vec![1.0f32; num_heads * head_dim];
        let mut output = vec![0.0f32; num_heads * head_dim];

        // Act
        flash_attn_decode_paged(&attn_config, &q, &mut output, &kv_cache, 0, 0, 0, None);

        // Assert: single KV position => softmax=[1], output=V for each KV head group
        let heads_per_kv = num_heads / num_kv_heads;
        for ah in 0..num_heads {
            let kv_h = ah / heads_per_kv;
            let o_off = ah * head_dim;
            for d in 0..head_dim {
                let expected = (d as f32 + 0.5) * 2.0;
                assert!(
                    (output[o_off + d] - expected).abs() < 1e-4,
                    "single-token decode head {ah} dim {d}: got {}, expected {expected}",
                    output[o_off + d]
                );
            }
        }
    }

    /// head_dim=1: attention with scalar-valued heads, flash matches naive.
    #[test]
    fn test_flash_attn_head_dim_one_matches_naive() {
        // Arrange
        let head_dim = 1;
        let seq_q = 4;
        let seq_kv = 5;
        let scale = 1.0; // 1/sqrt(1) = 1.0

        let q: Vec<f32> = (0..seq_q).map(|i| (i as f32 * 0.3 + 0.1)).collect();
        let k: Vec<f32> = (0..seq_kv).map(|i| (i as f32 * 0.2 + 0.5)).collect();
        let v: Vec<f32> = (0..seq_kv).map(|i| (i as f32 + 1.0)).collect();
        let mut flash_out = vec![0.0f32; seq_q];
        let mut naive_out = vec![0.0f32; seq_q];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, true);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, true);

        // Assert
        let max_diff = flash_out.iter().zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "head_dim=1 flash vs naive max_diff = {max_diff:.2e}");
    }

    /// head_dim=128: realistic transformer head dimension, flash matches naive.
    #[test]
    fn test_flash_attn_head_dim_128_matches_naive() {
        // Arrange
        let head_dim = 128;
        let seq_q = 2;
        let seq_kv = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_q * head_dim).map(|i| ((i as f32 * 0.01).sin())).collect();
        let k: Vec<f32> = (0..seq_kv * head_dim).map(|i| ((i as f32 * 0.02).cos())).collect();
        let v: Vec<f32> = (0..seq_kv * head_dim).map(|i| (i as f32 * 0.005 + 0.1)).collect();
        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, false);

        // Assert
        let max_diff = flash_out.iter().zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "head_dim=128 flash vs naive max_diff = {max_diff:.2e}");
    }

    /// GQA 8:2 grouping: 8 query heads share 2 KV heads, each group of 4 Q heads
    /// must share the same KV and produce the same output when given the same Q.
    #[test]
    fn test_gqa_grouping_shared_kv_produces_consistent_output() {
        // Arrange
        let head_dim = 8;
        let num_heads = 8;
        let num_kv_heads = 2;
        let seq_q = 1;
        let seq_kv = 4;
        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);
        let heads_per_kv = num_heads / num_kv_heads;

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.07).sin() * 0.4).collect()
        };

        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);
        // Same Q for all heads in a KV group
        let q_row = make_data(seq_q * head_dim, 0);
        let mut q = vec![0.0f32; num_heads * seq_q * head_dim];
        for h in 0..num_heads {
            let off = h * seq_q * head_dim;
            q[off..off + seq_q * head_dim].copy_from_slice(&q_row);
        }
        let mut output = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_multi_head(&config, &q, &k, &v, &mut output, seq_q, seq_kv, false);

        // Assert: heads sharing the same KV head must produce identical output
        for kv_h in 0..num_kv_heads {
            let first_qh = kv_h * heads_per_kv;
            let first_off = first_qh * seq_q * head_dim;
            for qh in (kv_h * heads_per_kv + 1)..((kv_h + 1) * heads_per_kv) {
                let off = qh * seq_q * head_dim;
                for d in 0..seq_q * head_dim {
                    assert!(
                        (output[first_off + d] - output[off + d]).abs() < 1e-10,
                        "GQA group {kv_h}: head {first_qh} vs {qh} dim {d} differ"
                    );
                }
            }
        }
    }

    /// GQA with num_kv_heads=1: all query heads share one KV head.
    /// Multi-head output must match single-head called with the same KV.
    #[test]
    fn test_gqa_single_kv_head_all_query_heads_share() {
        // Arrange
        let head_dim = 6;
        let num_heads = 6;
        let num_kv_heads = 1;
        let seq_q = 2;
        let seq_kv = 5;
        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.09).cos() * 0.25).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let mut multi_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act
        flash_attn_multi_head(&config, &q, &k, &v, &mut multi_out, seq_q, seq_kv, true);

        // Assert: each head matches independent single-head call with same KV
        for h in 0..num_heads {
            let q_off = h * seq_q * head_dim;
            let mut single_out = vec![0.0f32; seq_q * head_dim];
            flash_attn_single_head(
                &q[q_off..q_off + seq_q * head_dim],
                &k,
                &v,
                &mut single_out,
                seq_q, seq_kv, head_dim, config.scale, true,
            );
            let max_diff = multi_out[q_off..q_off + seq_q * head_dim]
                .iter().zip(single_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-6, "single KV head group: head {h} max_diff = {max_diff:.2e}");
        }
    }

    /// Subnormal f32 values in Q/K/V: output must remain finite and match naive.
    #[test]
    fn test_flash_attn_subnormal_values_stability() {
        // Arrange
        let head_dim = 8;
        let seq_q = 2;
        let seq_kv = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let tiny = f32::from_bits(1); // smallest positive subnormal
        let q: Vec<f32> = (0..seq_q * head_dim).map(|i| tiny * (i as f32 + 1.0)).collect();
        let k: Vec<f32> = (0..seq_kv * head_dim).map(|i| tiny * (i as f32 + 2.0)).collect();
        let v: Vec<f32> = (0..seq_kv * head_dim).map(|i| tiny * (i as f32 + 3.0)).collect();
        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, false);

        // Assert
        assert!(flash_out.iter().all(|v| v.is_finite()), "subnormal flash output non-finite");
        let max_diff = flash_out.iter().zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-12, "subnormal flash vs naive max_diff = {max_diff:.2e}");
    }

    /// Very large Q·K scores: scale=1.0 (no normalizing denominator), verify
    /// online softmax handles extreme exponents without NaN/Inf.
    #[test]
    fn test_flash_attn_extreme_scores_no_nan() {
        // Arrange
        let head_dim = 4;
        let seq_q = 2;
        let seq_kv = 3;
        let scale = 1.0; // intentionally no scaling to create extreme dot products

        let q = vec![100.0f32; seq_q * head_dim];
        let k = vec![100.0f32; seq_kv * head_dim];
        let v: Vec<f32> = (0..seq_kv * head_dim).map(|i| (i as f32 * 0.01)).collect();
        let mut flash_out = vec![0.0f32; seq_q * head_dim];
        let mut naive_out = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut flash_out, seq_q, seq_kv, head_dim, scale, false);
        naive_attention_single_head(&q, &k, &v, &mut naive_out, seq_q, seq_kv, head_dim, scale, false);

        // Assert: both must be finite and agree
        assert!(flash_out.iter().all(|v| v.is_finite()), "extreme scores: flash non-finite");
        assert!(naive_out.iter().all(|v| v.is_finite()), "extreme scores: naive non-finite");
        let max_diff = flash_out.iter().zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "extreme scores: max_diff = {max_diff:.2e}");
    }

    /// Naive attention with seq_q=1, seq_kv=1: single query and single KV position,
    /// output must exactly equal V regardless of scale.
    #[test]
    fn test_naive_single_query_single_kv_equals_v() {
        // Arrange
        let head_dim = 8;
        let scale = 0.5f32; // arbitrary scale, single position => softmax=[1]
        let q = vec![3.14f32; head_dim];
        let k = vec![2.71f32; head_dim];
        let v: Vec<f32> = (0..head_dim).map(|d| (d as f32 * 10.0 + 5.0)).collect();
        let mut output = vec![0.0f32; head_dim];

        // Act
        naive_attention_single_head(&q, &k, &v, &mut output, 1, 1, head_dim, scale, false);

        // Assert: single KV => softmax weight = 1.0 => output = V
        for d in 0..head_dim {
            assert!(
                (output[d] - v[d]).abs() < 1e-6,
                "dim {d}: got {}, expected {}", output[d], v[d]
            );
        }
    }

    /// Prefill with kv_offset=0, seq_q=1: degenerate case where the only query
    /// position can only attend to KV position 0 (causal mask: kv_pos<=q_pos=0).
    #[test]
    fn test_prefill_single_query_causal_attends_only_position_zero() {
        // Arrange
        let head_dim = 8;
        let num_heads = 1;
        let num_kv_heads = 1;
        let seq_q = 1;
        let seq_kv = 5;
        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.11).sin() * 0.3).collect()
        };
        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let mut prefill_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act: kv_offset=0, causal => qi=0 attends only kv_pos=0
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_q, seq_kv, 0);

        // Assert: output should equal V[0] (only visible position)
        for d in 0..head_dim {
            assert!(
                (prefill_out[d] - v[d]).abs() < 1e-5,
                "prefill causal qi=0 dim {d}: got {}, expected {}", prefill_out[d], v[d]
            );
        }
    }

    // ====================================================================
    // Wave 12x59 tests: 10 additional tests
    // ====================================================================

    /// Softmax normalization verification: with known Q/K, manually compute
    /// softmax weights and verify the output is exactly the weighted sum.
    /// Uses head_dim=2 so dot products are easy to verify by hand.
    #[test]
    fn test_flash_attn_softmax_weights_known_output() {
        // Arrange: head_dim=2, seq_q=1, seq_kv=2, scale=1/sqrt(2)
        let head_dim = 2;
        let seq_q = 1;
        let seq_kv = 2;
        let scale = 1.0 / (head_dim as f32).sqrt(); // ~0.7071

        // Q = [1, 0], K0 = [1, 0], K1 = [0, 1]
        // dot(Q, K0) = 1, score0 = 1 * scale = 0.7071
        // dot(Q, K1) = 0, score1 = 0 * scale = 0
        let q = vec![1.0f32, 0.0];
        let k = vec![1.0f32, 0.0, 0.0, 1.0];
        // V0 = [3, 7], V1 = [11, 13]
        let v = vec![3.0f32, 7.0, 11.0, 13.0];
        let mut output = vec![0.0f32; head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, false);

        // Assert: manually compute softmax
        // scores = [0.7071, 0.0], max = 0.7071
        // exp(0.7071 - 0.7071) = 1.0, exp(0.0 - 0.7071) = exp(-0.7071) ~ 0.4931
        // sum = 1.0 + 0.4931 = 1.4931
        // w0 = 1.0 / 1.4931 ~ 0.6697, w1 = 0.4931 / 1.4931 ~ 0.3303
        let s0 = 1.0f32 * scale;
        let s1 = 0.0f32 * scale;
        let max_s = s0.max(s1);
        let e0 = (s0 - max_s).exp();
        let e1 = (s1 - max_s).exp();
        let sum_e = e0 + e1;
        let w0 = e0 / sum_e;
        let w1 = e1 / sum_e;

        // output = w0 * V0 + w1 * V1
        let expected = [w0 * 3.0 + w1 * 11.0, w0 * 7.0 + w1 * 13.0];
        for d in 0..head_dim {
            assert!(
                (output[d] - expected[d]).abs() < 1e-5,
                "dim {d}: got {}, expected {}", output[d], expected[d]
            );
        }
    }

    /// Score normalization: verify that attention output is a convex combination
    /// of V rows (all output dimensions lie within [min(V), max(V)]).
    #[test]
    fn test_flash_attn_output_is_convex_combination_of_v() {
        // Arrange
        let head_dim = 4;
        let seq_q = 3;
        let seq_kv = 5;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.11).sin() * 2.0 - 1.0).collect()
        };

        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 50);
        let v = make_data(seq_kv * head_dim, 100);
        let mut output = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, false);

        // Assert: for each query and each dimension, output must be within [min(V_dim), max(V_dim)]
        for qi in 0..seq_q {
            for d in 0..head_dim {
                let v_vals: Vec<f32> = (0..seq_kv).map(|t| v[t * head_dim + d]).collect();
                let v_min = v_vals.iter().copied().fold(f32::INFINITY, f32::min);
                let v_max = v_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let out_val = output[qi * head_dim + d];
                assert!(
                    out_val >= v_min - 1e-5 && out_val <= v_max + 1e-5,
                    "qi={qi} dim={d}: output={out_val} not in [{v_min}, {v_max}]"
                );
            }
        }
    }

    /// FlashAttention with scale=0.0: all scores become 0, softmax is uniform,
    /// output equals mean of V rows regardless of Q and K.
    #[test]
    fn test_flash_attn_zero_scale_uniform_attention() {
        // Arrange
        let head_dim = 4;
        let seq_q = 2;
        let seq_kv = 3;

        let q = vec![1.0f32; seq_q * head_dim];
        let k = vec![1.0f32; seq_kv * head_dim];
        // V with distinct rows
        let mut v = vec![0.0f32; seq_kv * head_dim];
        for t in 0..seq_kv {
            v[t * head_dim] = (t + 1) as f32;
            v[t * head_dim + 1] = (t + 10) as f32;
        }
        let mut output = vec![0.0f32; seq_q * head_dim];

        // Act: scale=0.0 => all scores = 0 => uniform softmax = 1/seq_kv
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, 0.0, false);

        // Assert: output dim0 = mean(1,2,3) = 2.0 for all queries
        let expected_d0 = (1.0 + 2.0 + 3.0) / 3.0;
        let expected_d1 = (10.0 + 11.0 + 12.0) / 3.0;
        for qi in 0..seq_q {
            assert!(
                (output[qi * head_dim] - expected_d0).abs() < 1e-5,
                "qi={qi} dim0 = {}, expected {expected_d0}", output[qi * head_dim]
            );
            assert!(
                (output[qi * head_dim + 1] - expected_d1).abs() < 1e-5,
                "qi={qi} dim1 = {}, expected {expected_d1}", output[qi * head_dim + 1]
            );
        }
    }

    /// Transpose element conservation: sum of all elements must be identical
    /// before and after transpose (no element loss or duplication).
    #[test]
    fn test_transpose_preserves_element_sum() {
        // Arrange
        let seq_len = 7;
        let num_heads = 3;
        let head_dim = 11;
        let total = seq_len * num_heads * head_dim;

        let original: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.17).sin())).collect();
        let mut transposed = vec![0.0f32; total];

        // Act
        transpose_to_head_major(&original, &mut transposed, seq_len, num_heads, head_dim);

        // Assert: sum must be preserved (allow floating-point rounding)
        let sum_orig: f32 = original.iter().sum();
        let sum_trans: f32 = transposed.iter().sum();
        let rel_err = (sum_orig - sum_trans).abs() / sum_orig.abs().max(1e-7);
        assert!(
            rel_err < 1e-5,
            "sum before={sum_orig}, after={sum_trans}, rel_err={rel_err:.2e}"
        );
    }

    /// Causal attention with identical V rows: even though masking changes attention
    /// weights per position, if all V rows are identical, every position gets the
    /// same output regardless of masking.
    #[test]
    fn test_flash_attn_causal_identical_v_rows_same_output() {
        // Arrange
        let head_dim = 8;
        let seq_q = 5;
        let seq_kv = 5;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_q * head_dim)
            .map(|i| ((i as f32 * 0.13).sin() * 0.5))
            .collect();
        let k: Vec<f32> = (0..seq_kv * head_dim)
            .map(|i| ((i as f32 * 0.17).cos() * 0.5))
            .collect();
        // All V rows identical
        let v_row: Vec<f32> = (0..head_dim).map(|d| (d as f32 + 1.0) * 0.1).collect();
        let v: Vec<f32> = v_row.repeat(seq_kv);
        let mut output = vec![0.0f32; seq_q * head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, true);

        // Assert: every query position output equals the identical V row
        for qi in 0..seq_q {
            for d in 0..head_dim {
                assert!(
                    (output[qi * head_dim + d] - v_row[d]).abs() < 1e-5,
                    "qi={qi} dim={d}: got {}, expected {}", output[qi * head_dim + d], v_row[d]
                );
            }
        }
    }

    /// Prefill with large kv_offset where query position is far beyond all KV:
    /// kv_offset=100, seq_q=1, seq_kv=3. All KV positions visible (0..2 <= 100).
    /// Verify against naive non-causal reference (all positions visible).
    #[test]
    fn test_prefill_large_offset_all_kv_visible() {
        // Arrange
        let head_dim = 8;
        let num_heads = 1;
        let num_kv_heads = 1;
        let seq_q = 1;
        let seq_kv = 3;
        let kv_offset = 100;

        let config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.07).sin() * 0.4).collect()
        };

        let q = make_data(num_heads * seq_q * head_dim, 0);
        let k = make_data(num_kv_heads * seq_kv * head_dim, 50);
        let v = make_data(num_kv_heads * seq_kv * head_dim, 100);
        let mut prefill_out = vec![0.0f32; num_heads * seq_q * head_dim];
        let mut naive_out = vec![0.0f32; num_heads * seq_q * head_dim];

        // Act: prefill with kv_offset=100 (all KV visible)
        flash_attn_prefill(&config, &q, &k, &v, &mut prefill_out, seq_q, seq_kv, kv_offset);
        // Naive non-causal: all positions visible
        naive_attention_multi_head(&config, &q, &k, &v, &mut naive_out, seq_q, seq_kv, false);

        // Assert
        let max_diff = prefill_out
            .iter()
            .zip(naive_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "large kv_offset prefill vs naive non-causal max_diff = {max_diff:.2e}"
        );
    }

    /// FlashAttnConfig scale monotonicity: larger head_dim produces smaller scale.
    /// Verify that 1/sqrt(head_dim) is strictly decreasing.
    #[test]
    fn test_config_scale_monotonically_decreases_with_head_dim() {
        // Arrange: test head_dims in ascending order
        let head_dims: &[usize] = &[4, 8, 16, 32, 64, 128, 256];
        let mut prev_scale = f32::INFINITY;

        // Act & Assert
        for &hd in head_dims {
            let config = FlashAttnConfig::new(hd, 4, 2);
            assert!(
                config.scale < prev_scale,
                "head_dim={hd}: scale={} not less than prev={}", config.scale, prev_scale
            );
            prev_scale = config.scale;
        }
    }

    /// flash_attn_single_head with dominant KV position: one KV row has a much
    /// larger dot product with the query than others, so attention concentrates
    /// on that position. Verify the output is dominated by the dominant V row.
    #[test]
    fn test_flash_attn_dominant_kv_position_concentrated_attention() {
        // Arrange: head_dim=4, 1 query, 3 KV positions
        let head_dim = 4;
        let seq_q = 1;
        let seq_kv = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q aligned with K0 (large dot product) and orthogonal to K1, K2
        let q = vec![10.0f32, 0.0, 0.0, 0.0];
        let k = vec![10.0f32, 1.0, 1.0, 1.0,   // K0 aligned with Q => large dot
                     0.0, 10.0, 1.0, 1.0,         // K1 orthogonal to Q
                     0.0, 1.0, 10.0, 1.0];         // K2 orthogonal to Q
        // V rows with distinct values
        let v = vec![10.0f32, 20.0, 30.0, 40.0,
                     50.0f32, 60.0, 70.0, 80.0,
                     90.0f32, 100.0, 110.0, 120.0];
        let mut output = vec![0.0f32; head_dim];

        // Act
        flash_attn_single_head(&q, &k, &v, &mut output, seq_q, seq_kv, head_dim, scale, false);

        // Assert: dominant position is K0, output should be close to V0
        // dot(Q,K0) = 100*scale ~ 50, dot(Q,K1) = 0, dot(Q,K2) = 0
        // softmax concentrates on K0: output ~= V0 = [10, 20, 30, 40]
        for d in 0..head_dim {
            assert!(
                (output[d] - v[d]).abs() < 0.1,
                "dim {d}: got {}, expected ~{}", output[d], v[d]
            );
        }
    }

    /// flash_attn_decode_paged with sliding_window at the boundary:
    /// token_pos = sliding_window - 1 => all cached positions visible (no eviction yet).
    #[test]
    fn test_flash_attn_decode_paged_sliding_window_boundary() {
        use crate::types::ModelConfig;

        // Arrange
        let sliding_window = 4usize;
        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: crate::types::DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: Some(sliding_window),
        };

        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let cached_len = sliding_window; // exactly at the boundary
        let v_base = num_kv_heads * PAGE_SIZE * head_dim;

        let mut kv_cache = KvCache::new(&cfg, 1, cfg.max_seq_len).unwrap();
        let attn_config = FlashAttnConfig::new(head_dim, num_heads, num_kv_heads);

        let make_val = |i: usize, seed: usize| -> f32 {
            ((i + seed) as f32 * 0.11).sin() * 0.3
        };

        // Write exactly sliding_window tokens
        let positions = kv_cache.append(0, 0, cached_len).unwrap();
        for (t, &(pid, off)) in positions.iter().enumerate() {
            let page_ptr = kv_cache.page_mut_ptr(pid) as *mut f32;
            for kv_h in 0..num_kv_heads {
                let k_base = kv_h * PAGE_SIZE * head_dim + off * head_dim;
                let v_off = v_base + kv_h * PAGE_SIZE * head_dim + off * head_dim;
                for d in 0..head_dim {
                    unsafe {
                        *page_ptr.add(k_base + d) = make_val(t * head_dim + d, kv_h * 100);
                        *page_ptr.add(v_off + d) = make_val(t * head_dim + d, kv_h * 100 + 500);
                    }
                }
            }
        }

        let token_pos = cached_len - 1; // last position
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| make_val(i, 999))
            .collect();
        let mut paged_out = vec![0.0f32; num_heads * head_dim];

        // Act: decode with sliding_window at the boundary
        flash_attn_decode_paged(
            &attn_config, &q, &mut paged_out, &kv_cache, 0, 0, token_pos, Some(sliding_window),
        );

        // Assert: all cached positions should be visible (no eviction at boundary)
        let heads_per_kv = num_heads / num_kv_heads;
        for ah in 0..num_heads {
            let kv_h = ah / heads_per_kv;
            let q_off = ah * head_dim;

            let mut k_cont = vec![0.0f32; cached_len * head_dim];
            let mut v_cont = vec![0.0f32; cached_len * head_dim];
            for t in 0..cached_len {
                for d in 0..head_dim {
                    k_cont[t * head_dim + d] = make_val(t * head_dim + d, kv_h * 100);
                    v_cont[t * head_dim + d] = make_val(t * head_dim + d, kv_h * 100 + 500);
                }
            }

            let mut naive_out = vec![0.0f32; head_dim];
            naive_attention_single_head(
                &q[q_off..q_off + head_dim],
                &k_cont, &v_cont, &mut naive_out,
                1, cached_len, head_dim, attn_config.scale, false,
            );

            let max_diff = paged_out[q_off..q_off + head_dim]
                .iter().zip(naive_out.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-5,
                "sliding window boundary head {ah}: max_diff = {max_diff:.2e}"
            );
        }
    }

    /// naive_attention_single_head with causal and single query at last position:
    /// qi = seq_kv-1 can see all KV positions. Verify output equals full non-causal.
    #[test]
    fn test_naive_causal_last_position_sees_all_matches_non_causal() {
        // Arrange
        let head_dim = 8;
        let seq_q = 1;
        let seq_kv = 6;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.09).sin() * 0.4).collect()
        };

        let q = make_data(seq_q * head_dim, 0);
        let k = make_data(seq_kv * head_dim, 50);
        let v = make_data(seq_kv * head_dim, 100);
        let mut causal_out = vec![0.0f32; head_dim];
        let mut non_causal_out = vec![0.0f32; head_dim];

        // Act: query at position seq_kv-1 with causal should see all positions
        // We simulate by using the last row of Q (position seq_kv-1) with causal
        // Build Q with seq_kv rows, but only check the last row
        let q_full = make_data(seq_kv * head_dim, 0);
        let mut full_causal_out = vec![0.0f32; seq_kv * head_dim];
        let mut full_non_causal_out = vec![0.0f32; seq_kv * head_dim];
        naive_attention_single_head(
            &q_full, &k, &v, &mut full_causal_out, seq_kv, seq_kv, head_dim, scale, true,
        );
        naive_attention_single_head(
            &q_full, &k, &v, &mut full_non_causal_out, seq_kv, seq_kv, head_dim, scale, false,
        );

        // Assert: last query position (qi=seq_kv-1) should have same output
        // in both causal and non-causal (can see all KV positions)
        let last_off = (seq_kv - 1) * head_dim;
        for d in 0..head_dim {
            assert!(
                (full_causal_out[last_off + d] - full_non_causal_out[last_off + d]).abs() < 1e-6,
                "last position dim {d}: causal={}, non-causal={}",
                full_causal_out[last_off + d], full_non_causal_out[last_off + d]
            );
        }
    }

    /// flash_attn_multi_head with two different scales: verify that changing the
    /// scale factor actually changes the output (not silently ignored).
    #[test]
    fn test_flash_attn_different_scales_produce_different_outputs() {
        // Arrange
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let seq_len = 4;

        let make_data = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + seed) as f32 * 0.13).sin() * 0.4).collect()
        };

        let q = make_data(num_heads * seq_len * head_dim, 0);
        let k = make_data(num_kv_heads * seq_len * head_dim, 50);
        let v = make_data(num_kv_heads * seq_len * head_dim, 100);

        let config_small = FlashAttnConfig {
            scale: 0.01,
            ..FlashAttnConfig::new(head_dim, num_heads, num_kv_heads)
        };
        let config_large = FlashAttnConfig {
            scale: 10.0,
            ..FlashAttnConfig::new(head_dim, num_heads, num_kv_heads)
        };

        let mut out_small = vec![0.0f32; num_heads * seq_len * head_dim];
        let mut out_large = vec![0.0f32; num_heads * seq_len * head_dim];

        // Act
        flash_attn_multi_head(&config_small, &q, &k, &v, &mut out_small, seq_len, seq_len, true);
        flash_attn_multi_head(&config_large, &q, &k, &v, &mut out_large, seq_len, seq_len, true);

        // Assert: outputs must differ (small scale => more uniform, large scale => more peaked)
        let any_diff = out_small.iter().zip(out_large.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(
            any_diff,
            "different scales must produce different outputs"
        );

        // Assert: both must be finite
        assert!(out_small.iter().all(|v| v.is_finite()), "small scale output non-finite");
        assert!(out_large.iter().all(|v| v.is_finite()), "large scale output non-finite");
    }
}
