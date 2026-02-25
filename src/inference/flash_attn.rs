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
    let tile_kv = 256.min(seq_kv);

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
                let kp = kv_cache.page_ptr(pid) as *const f32;
                let k_base = kv_h * PAGE_SIZE * hd + off * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[q_off + d] * unsafe { *kp.add(k_base + d) };
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
                let vp = kv_cache.page_ptr(pid) as *const f32;
                let v_page_off = v_base + kv_h * PAGE_SIZE * hd + off * hd;
                for d in 0..hd {
                    output[o_off + d] += w * unsafe { *vp.add(v_page_off + d) };
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
    let tile_kv = 256.min(seq_kv);

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
        use crate::inference::types::ModelConfig;

        let cfg = ModelConfig {
            arch: crate::inference::types::ModelArch::Llama,
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
            dtype: crate::inference::types::DType::F32,
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
        use crate::inference::types::ModelConfig;

        let cfg = ModelConfig {
            arch: crate::inference::types::ModelArch::Llama,
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
            dtype: crate::inference::types::DType::F32,
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
        use crate::inference::types::ModelConfig;

        let cfg = ModelConfig {
            arch: crate::inference::types::ModelArch::Llama,
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
            dtype: crate::inference::types::DType::F32,
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
        let mut all_k_flat: Vec<Vec<f32>> = (0..num_kv_heads)
            .map(|kv_h| {
                (0..total_len * head_dim)
                    .map(|i| make_val(i, kv_h * 100))
                    .collect()
            })
            .collect();
        let mut all_v_flat: Vec<Vec<f32>> = (0..num_kv_heads)
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
}
