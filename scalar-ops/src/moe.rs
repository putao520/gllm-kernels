//! Scalar MoE (Mixture of Experts) operator references for JIT extern calls.

/// MoE gate: hidden[seq_len, hidden] @ router_w[hidden, num_experts] → softmax → gate_probs[seq_len, num_experts]
///
/// Equivalent to Gemm + row-wise Softmax.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_moe_gate(
    hidden_input: *const f32,
    router_w: *const f32,
    output: *mut f32,
    seq_len: usize,
    num_experts: usize,
    hidden: usize,
) {
    unsafe {
        // Step 1: GEMM — output[s, e] = Σ_h hidden[s,h] * router_w[h,e]
        for s in 0..seq_len {
            for e in 0..num_experts {
                let mut acc = 0.0f32;
                for h in 0..hidden {
                    acc += *hidden_input.add(s * hidden + h)
                        * *router_w.add(h * num_experts + e);
                }
                *output.add(s * num_experts + e) = acc;
            }
        }

        // Step 2: Row-wise softmax
        for s in 0..seq_len {
            let row = s * num_experts;
            let mut max_val = f32::NEG_INFINITY;
            for e in 0..num_experts {
                let v = *output.add(row + e);
                if v > max_val { max_val = v; }
            }
            let mut sum = 0.0f32;
            for e in 0..num_experts {
                let v = (*output.add(row + e) - max_val).exp();
                *output.add(row + e) = v;
                sum += v;
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for e in 0..num_experts {
                *output.add(row + e) *= inv_sum;
            }
        }
    }
}

/// Top-K selection: for each row in gate_probs[seq_len, num_experts],
/// select top_k largest values and their indices, renormalize weights.
///
/// Output layout: indices[seq_len * top_k] as u32 (reinterpreted as f32 bits),
/// followed by weights[seq_len * top_k] as f32.
/// Total output size: seq_len * top_k * 2 floats.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_topk(
    gate_probs: *const f32,
    output: *mut f32,
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
) {
    unsafe {
        let idx_base = output; // indices section: seq_len * top_k
        let wt_base = output.add(seq_len * top_k); // weights section

        for s in 0..seq_len {
            let row = gate_probs.add(s * num_experts);

            // Insertion sort to find top_k
            let mut top_indices = vec![0u32; top_k];
            let mut top_values = vec![f32::NEG_INFINITY; top_k];

            for e in 0..num_experts {
                let val = *row.add(e);
                // Find insertion point
                let mut insert_pos = top_k;
                for ki in 0..top_k {
                    if val > top_values[ki] {
                        insert_pos = ki;
                        break;
                    }
                }
                if insert_pos < top_k {
                    // Shift down
                    for ki in (insert_pos + 1..top_k).rev() {
                        top_indices[ki] = top_indices[ki - 1];
                        top_values[ki] = top_values[ki - 1];
                    }
                    top_indices[insert_pos] = e as u32;
                    top_values[insert_pos] = val;
                }
            }

            // Renormalize weights
            let mut sum = 0.0f32;
            for ki in 0..top_k {
                sum += top_values[ki];
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

            for ki in 0..top_k {
                // Store index as f32 bits
                let idx_f32 = f32::from_bits(top_indices[ki]);
                *idx_base.add(s * top_k + ki) = idx_f32;
                *wt_base.add(s * top_k + ki) = top_values[ki] * inv_sum;
            }
        }
    }
}

/// Weighted sum of expert outputs:
/// output[s, d] = Σ_k weights[s,k] * expert_outputs[indices[s,k]][s, d]
///
/// expert_outputs_base points to a flat buffer of all expert outputs concatenated:
/// expert_outputs_base[expert_idx * seq_len * hidden + s * hidden + d]
///
/// indices: [seq_len * top_k] as u32 (stored as f32 bits)
/// weights: [seq_len * top_k] as f32
/// output: [seq_len * hidden]
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_weighted_sum(
    expert_outputs: *const f32,
    indices: *const f32,
    weights: *const f32,
    output: *mut f32,
    seq_len: usize,
    hidden: usize,
    top_k: usize,
) {
    unsafe {
        // Zero output
        for i in 0..seq_len * hidden {
            *output.add(i) = 0.0;
        }

        for s in 0..seq_len {
            for ki in 0..top_k {
                let expert_idx = f32::to_bits(*indices.add(s * top_k + ki)) as usize;
                let w = *weights.add(s * top_k + ki);

                let expert_row = expert_outputs
                    .add(expert_idx * seq_len * hidden + s * hidden);

                for d in 0..hidden {
                    *output.add(s * hidden + d) += w * *expert_row.add(d);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_gate_softmax_sums_to_one() {
        let hidden = 4;
        let num_experts = 3;
        let seq_len = 2;
        let input = vec![1.0f32; seq_len * hidden];
        let router = vec![0.1f32; hidden * num_experts];
        let mut output = vec![0.0f32; seq_len * num_experts];
        scalar_moe_gate(
            input.as_ptr(), router.as_ptr(), output.as_mut_ptr(),
            seq_len, num_experts, hidden,
        );
        for s in 0..seq_len {
            let sum: f32 = output[s * num_experts..(s + 1) * num_experts].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {s} sum = {sum}");
        }
    }

    #[test]
    fn test_topk_selects_largest() {
        let probs = vec![0.1f32, 0.5, 0.3, 0.1]; // 1 row, 4 experts
        let top_k = 2;
        let mut output = vec![0.0f32; top_k * 2]; // indices + weights
        scalar_topk(probs.as_ptr(), output.as_mut_ptr(), 1, 4, top_k);
        let idx0 = f32::to_bits(output[0]) as usize;
        let idx1 = f32::to_bits(output[1]) as usize;
        assert_eq!(idx0, 1, "top-1 should be expert 1");
        assert_eq!(idx1, 2, "top-2 should be expert 2");
        let w_sum: f32 = output[top_k..].iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-5, "weights sum = {w_sum}");
    }

    #[test]
    fn test_weighted_sum_identity() {
        let hidden = 3;
        let top_k = 1;
        let seq_len = 1;
        // Single expert, weight=1.0
        let expert_out = vec![1.0f32, 2.0, 3.0];
        let indices = vec![f32::from_bits(0u32)]; // expert 0
        let weights = vec![1.0f32];
        let mut output = vec![0.0f32; hidden];
        scalar_weighted_sum(
            expert_out.as_ptr(), indices.as_ptr(), weights.as_ptr(),
            output.as_mut_ptr(), seq_len, hidden, top_k,
        );
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }
}
