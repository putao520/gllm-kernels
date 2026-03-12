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
pub extern "C" fn scalar_multi_head_attention(
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
