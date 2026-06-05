//! Scalar multi-head attention helper for JIT extern calls.

/// C-callable multi-head attention.
///
/// Q, K, V are `[seq_len, num_heads * head_dim]` row-major (interleaved heads).
/// Output is `[seq_len, num_heads * head_dim]` row-major.
///
/// Internally reshapes to `[num_heads, seq_len, head_dim]`, runs per-head
/// scaled dot-product attention (non-causal, for BERT-style models),
/// then reshapes back.
///
/// ABI: extern "C" so JIT can call it directly.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_multi_head_attention(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    output: *mut f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let hidden = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_slice = unsafe { std::slice::from_raw_parts(q, seq_len * hidden) };
    let k_slice = unsafe { std::slice::from_raw_parts(k, seq_len * hidden) };
    let v_slice = unsafe { std::slice::from_raw_parts(v, seq_len * hidden) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(output, seq_len * hidden) };

    // Per-head buffers
    let mut q_head = vec![0.0f32; seq_len * head_dim];
    let mut k_head = vec![0.0f32; seq_len * head_dim];
    let mut v_head = vec![0.0f32; seq_len * head_dim];
    // Attention scores: [seq_len, seq_len]
    let mut scores = vec![0.0f32; seq_len * seq_len];
    let mut out_head = vec![0.0f32; seq_len * head_dim];

    for h in 0..num_heads {
        // Extract head h from [seq, hidden] -> [seq, head_dim]
        for s in 0..seq_len {
            let src_off = s * hidden + h * head_dim;
            let dst_off = s * head_dim;
            q_head[dst_off..dst_off + head_dim]
                .copy_from_slice(&q_slice[src_off..src_off + head_dim]);
            k_head[dst_off..dst_off + head_dim]
                .copy_from_slice(&k_slice[src_off..src_off + head_dim]);
            v_head[dst_off..dst_off + head_dim]
                .copy_from_slice(&v_slice[src_off..src_off + head_dim]);
        }

        // Compute scaled dot-product attention for this head:
        // scores = Q @ K^T * scale, then softmax, then @ V

        // Step 1: scores[i][j] = dot(Q[i], K[j]) * scale
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // Step 2: row-wise softmax
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..seq_len {
                if row[j] > max_val {
                    max_val = row[j];
                }
            }
            // Exp and sum
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                row[j] = (row[j] - max_val).exp();
                sum += row[j];
            }
            // Normalize
            let inv_sum = 1.0 / sum;
            for j in 0..seq_len {
                row[j] *= inv_sum;
            }
        }

        // Step 3: out_head = scores @ V
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += scores[i * seq_len + j] * v_head[j * head_dim + d];
                }
                out_head[i * head_dim + d] = acc;
            }
        }

        // Write back head h to [seq, hidden] layout
        for s in 0..seq_len {
            let src_off = s * head_dim;
            let dst_off = s * hidden + h * head_dim;
            out_slice[dst_off..dst_off + head_dim]
                .copy_from_slice(&out_head[src_off..src_off + head_dim]);
        }
    }
}

/// C-callable multi-head attention with learnable per-head **attention sinks**
/// (OpenAI gpt-oss-20b / StreamingLLM style).
///
/// Same layout as [`scalar_multi_head_attention`], but the softmax denominator
/// includes a virtual `ki = -1` position scoring `sinks[h]` that absorbs probability
/// mass without contributing to the output. For each head `h`:
///
/// ```text
///   running_max_init = sinks[h]
///   running_sum_init = exp(sinks[h] - sinks[h]) = 1.0
///   // Then standard online softmax over ki ∈ [0, seq_len).
///   o[qi, h, d] = Σ_{ki} (exp(s(qi, ki, h) - m_h) / l_h) * V[ki, h, d]
///   where m_h = max(sinks[h], max_{ki} s(qi, ki, h))
///         l_h = exp(sinks[h] - m_h) + Σ_{ki} exp(s(qi, ki, h) - m_h).
/// ```
///
/// The sink term does **not** contribute to the output accumulator (virtual
/// ki = -1 has no V row) — it only inflates the softmax denominator.
///
/// ABI: extern "C" so JIT can call it directly.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_multi_head_attention_with_sinks(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    sinks: *const f32,
    output: *mut f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    causal: usize, // 0 = bidirectional, non-zero = causal mask (ki ≤ qi)
) {
    let hidden = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_slice = unsafe { std::slice::from_raw_parts(q, seq_len * hidden) };
    let k_slice = unsafe { std::slice::from_raw_parts(k, seq_len * hidden) };
    let v_slice = unsafe { std::slice::from_raw_parts(v, seq_len * hidden) };
    let sinks_slice = unsafe { std::slice::from_raw_parts(sinks, num_heads) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(output, seq_len * hidden) };

    let causal_mask = causal != 0;

    for h in 0..num_heads {
        let sink_h = sinks_slice[h];
        for qi in 0..seq_len {
            let q_base = qi * hidden + h * head_dim;

            // Phase 1: compute raw scores s[ki] = scale * dot(Q[qi,h,:], K[ki,h,:])
            let ki_end = if causal_mask { qi + 1 } else { seq_len };
            let mut scores = vec![0.0f32; ki_end];
            for ki in 0..ki_end {
                let k_base = ki * hidden + h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_slice[q_base + d] * k_slice[k_base + d];
                }
                scores[ki] = dot * scale;
            }

            // Phase 2: stable softmax with sink term in denominator.
            let mut m = sink_h;
            for &s in &scores {
                if s > m {
                    m = s;
                }
            }
            // l = exp(sink_h - m) + Σ exp(s - m); sink does not go into numerator.
            let mut l = (sink_h - m).exp();
            let mut weights = vec![0.0f32; ki_end];
            for (ki, &s) in scores.iter().enumerate() {
                let w = (s - m).exp();
                weights[ki] = w;
                l += w;
            }
            let inv_l = 1.0 / l;

            // Phase 3: output = Σ_{ki} (weights[ki] / l) * V[ki,h,:]
            let out_base = qi * hidden + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for ki in 0..ki_end {
                    let v_val = v_slice[ki * hidden + h * head_dim + d];
                    acc += weights[ki] * v_val;
                }
                out_slice[out_base + d] = acc * inv_l;
            }
        }
    }
}

/// C-callable cached GQA attention (scalar reference).
///
/// Q is `[seq_len, num_heads * head_dim]` (current query).
/// K_cache is `[total_seq, num_kv_heads * head_dim]` (full KV cache).
/// V_cache is `[total_seq, num_kv_heads * head_dim]`.
/// Output is `[seq_len, num_heads * head_dim]`.
///
/// Supports GQA: `num_heads` may differ from `num_kv_heads` (grouped query attention).
/// Applies causal mask: for each query position `i`, only attends to positions `0..=i+(total_seq-seq_len)`.
/// Writes attention sparsity (fraction of near-zero weights) as the last f32 in output buffer.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_cached_gqa_attention(
    q: *const f32,
    k_cache: *const f32,
    v_cache: *const f32,
    output: *mut f32,
    seq_len: usize,
    total_seq: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_group = num_heads / num_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let prefix_len = total_seq.saturating_sub(seq_len);

    let q_slice = unsafe { std::slice::from_raw_parts(q, seq_len * q_dim) };
    let k_slice = unsafe { std::slice::from_raw_parts(k_cache, total_seq * kv_dim) };
    let v_slice = unsafe { std::slice::from_raw_parts(v_cache, total_seq * kv_dim) };
    let out_size = seq_len * q_dim + 1; // +1 for sparsity stat
    let out_slice = unsafe { std::slice::from_raw_parts_mut(output, out_size) };

    let mut scores = vec![0.0f32; total_seq];
    let mut near_zero_count: u64 = 0;
    let mut total_weights: u64 = 0;

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;

        for s in 0..seq_len {
            let cur_pos = prefix_len + s;

            // Compute scores: dot(Q[s,h], K_cache[t,kv_h]) * scale
            for t in 0..total_seq {
                if t > cur_pos {
                    scores[t] = f32::NEG_INFINITY; // causal mask
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_slice[s * q_dim + h * head_dim + d]
                            * k_slice[t * kv_dim + kv_h * head_dim + d];
                    }
                    scores[t] = dot * scale;
                }
            }

            // Row-wise softmax
            let valid_len = cur_pos + 1;
            let mut max_val = f32::NEG_INFINITY;
            for t in 0..valid_len {
                if scores[t] > max_val { max_val = scores[t]; }
            }
            let mut sum = 0.0f32;
            for t in 0..valid_len {
                scores[t] = (scores[t] - max_val).exp();
                sum += scores[t];
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for t in 0..valid_len {
                scores[t] *= inv_sum;
            }
            for t in valid_len..total_seq {
                scores[t] = 0.0;
            }

            // Sparsity stats
            for t in 0..valid_len {
                total_weights += 1;
                if scores[t] < 0.01 { near_zero_count += 1; }
            }

            // Weighted sum: out[s,h] = Σ_t scores[t] * V_cache[t,kv_h]
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..valid_len {
                    acc += scores[t] * v_slice[t * kv_dim + kv_h * head_dim + d];
                }
                out_slice[s * q_dim + h * head_dim + d] = acc;
            }
        }
    }

    // Write sparsity ratio at the end
    let sparsity = if total_weights > 0 {
        near_zero_count as f32 / total_weights as f32
    } else {
        0.0
    };
    out_slice[seq_len * q_dim] = sparsity;
}
