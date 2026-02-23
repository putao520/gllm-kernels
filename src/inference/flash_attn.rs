//! FlashAttention — tiled attention computation for long sequences.
//!
//! Implements the FlashAttention algorithm (Dao et al., 2022) in pure Rust
//! for the CPU fallback path. Processes Q/K/V in tiles to keep the working
//! set in L1/L2 cache, avoiding the O(seq^2) memory of full attention matrices.

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
}
