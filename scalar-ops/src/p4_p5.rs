//! Scalar reference implementations for P4/P5 experimental operators.
//!
//! Phase 0 reference only. NOT runtime callable (CLAUDE.md NO_SCALAR).
//! These functions serve as ground truth for JIT codegen correctness testing.

/// GateMask: `output[i] = gate[i] > threshold ? activation[i] : 0.0`
///
/// - `activation`: `[hidden]` f32 input activations
/// - `gate`: `[hidden]` f32 gate logits
/// - `output`: `[hidden]` f32 masked output
/// - `hidden`: dimension size
/// - `threshold`: gate decision threshold
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_gate_mask(
    activation: *const f32,
    gate: *const f32,
    output: *mut f32,
    hidden: usize,
    threshold: f32,
) {
    if activation.is_null() || gate.is_null() || output.is_null() || hidden == 0 {
        return;
    }
    unsafe {
        for i in 0..hidden {
            *output.add(i) = if *gate.add(i) > threshold {
                *activation.add(i)
            } else {
                0.0
            };
        }
    }
}

/// AttentionSkipMask: `output[i] = tokens[i] != skip_token_id ? 1.0 : 0.0`
///
/// Produces a binary mask where positions matching `skip_token_id` (e.g. padding)
/// are masked out (0.0) and all other positions are kept (1.0).
///
/// - `tokens`: `[seq_len]` f32 values representing integer token IDs
/// - `output`: `[seq_len]` f32 binary mask
/// - `seq_len`: sequence length
/// - `skip_token_id`: token ID to skip (e.g. pad_token_id), stored as f32
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_attention_skip_mask(
    tokens: *const f32,
    output: *mut f32,
    seq_len: usize,
    skip_token_id: f32,
) {
    if tokens.is_null() || output.is_null() || seq_len == 0 {
        return;
    }
    unsafe {
        for i in 0..seq_len {
            *output.add(i) = if *tokens.add(i) != skip_token_id {
                1.0
            } else {
                0.0
            };
        }
    }
}

/// LayerBypass: `output[i] = input[i]` (pass-through / identity).
///
/// In the real JIT, this op conditionally skips the current layer when the
/// early-exit confidence exceeds `threshold`. The scalar reference models
/// the default "apply layer" path (identity copy of hidden state).
///
/// - `input`: `[hidden]` f32 hidden state
/// - `output`: `[hidden]` f32 output
/// - `hidden`: dimension size
/// - `_threshold`: bypass threshold (unused in scalar; JIT uses it for conditional skip)
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_layer_bypass(
    input: *const f32,
    output: *mut f32,
    hidden: usize,
    _threshold: f32,
) {
    if input.is_null() || output.is_null() || hidden == 0 {
        return;
    }
    unsafe {
        for i in 0..hidden {
            *output.add(i) = *input.add(i);
        }
    }
}

/// SoftmaxWithEntropy: standard softmax over `vocab_size` logits + entropy.
///
/// Computes `output[i] = softmax(logits)[i]` for `i in [0, vocab_size-1]`,
/// then writes `entropy = -sum(p[i] * ln(p[i]))` into `output[vocab_size]`.
///
/// Output layout: `[vocab_size + 1]` — first `vocab_size` elements are softmax
/// probabilities, last element is the entropy.
///
/// - `logits`: `[vocab_size]` f32 raw logits
/// - `output`: `[vocab_size + 1]` f32 softmax probabilities + entropy
/// - `vocab_size`: vocabulary size
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_softmax_with_entropy(
    logits: *const f32,
    output: *mut f32,
    vocab_size: usize,
) {
    if logits.is_null() || output.is_null() || vocab_size == 0 {
        return;
    }
    unsafe {
        // Pass 1: find max for numerical stability
        let mut max_val = *logits;
        for i in 1..vocab_size {
            let v = *logits.add(i);
            if v > max_val {
                max_val = v;
            }
        }

        // Pass 2: exp(x - max) and accumulate sum
        let mut sum = 0.0_f32;
        for i in 0..vocab_size {
            let e = (*logits.add(i) - max_val).exp();
            *output.add(i) = e;
            sum += e;
        }

        // Pass 3: normalize and compute entropy
        let inv_sum = 1.0 / sum;
        let mut entropy = 0.0_f32;
        for i in 0..vocab_size {
            let p = *output.add(i) * inv_sum;
            *output.add(i) = p;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        *output.add(vocab_size) = entropy;
    }
}

/// EntropyGate: `output[0] = entropy < threshold ? 1.0 : 0.0`
///
/// Reads a scalar entropy value and produces a binary gate decision.
/// When entropy is low (confident prediction), the gate is 1.0 (write KV cache).
///
/// - `entropy`: single f32 entropy value
/// - `output`: single f32 gate decision (0.0 or 1.0)
/// - `_vocab_size`: unused (for ABI compatibility)
/// - `threshold`: entropy decision threshold
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_entropy_gate(
    entropy: *const f32,
    output: *mut f32,
    _vocab_size: usize,
    threshold: f32,
) {
    if entropy.is_null() || output.is_null() {
        return;
    }
    unsafe {
        *output = if *entropy < threshold { 1.0 } else { 0.0 };
    }
}

/// VRangeQuant: value-range quantization and dequantization (round-trip).
///
/// Per-block quantization:
///   1. Find (min, max) in each block of `block_size` elements
///   2. Quantize: `q = clamp(round((x - min) / (max - min) * (2^bits - 1)), 0, 2^bits - 1)`
///   3. Dequantize: `x' = q / (2^bits - 1) * (max - min) + min`
///
/// The output is the dequantized (reconstructed) values.
///
/// - `input`: `[seq_len * kv_dim]` f32 input values
/// - `output`: `[seq_len * kv_dim]` f32 dequantized output values
/// - `seq_len`: sequence length
/// - `kv_dim`: per-head KV dimension
/// - `block_size`: quantization block size
/// - `bits`: bits per quantized value (must be <= 8 for practical purposes)
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_vrange_quant(
    input: *const f32,
    output: *mut f32,
    seq_len: usize,
    kv_dim: usize,
    block_size: usize,
    bits: usize,
) {
    if input.is_null() || output.is_null() || seq_len == 0 || kv_dim == 0 || block_size == 0 {
        return;
    }
    if bits == 0 || bits > 32 {
        return;
    }
    let total = seq_len * kv_dim;
    let levels = ((1u32 << bits) - 1) as f32;
    if levels == 0.0 {
        return;
    }
    unsafe {
        let mut block_start = 0usize;
        while block_start < total {
            let block_end = (block_start + block_size).min(total);
            let block_len = block_end - block_start;

            // Find block min and max
            let mut b_min = *input.add(block_start);
            let mut b_max = *input.add(block_start);
            for i in 1..block_len {
                let v = *input.add(block_start + i);
                if v < b_min {
                    b_min = v;
                }
                if v > b_max {
                    b_max = v;
                }
            }

            let range = b_max - b_min;
            let inv_range = if range > 0.0 { 1.0 / range } else { 0.0 };

            // Quantize + dequantize each element
            for i in 0..block_len {
                let x = *input.add(block_start + i);
                let q = ((x - b_min) * inv_range * levels).round().clamp(0.0, levels);
                *output.add(block_start + i) = q / levels * range + b_min;
            }

            block_start = block_end;
        }
    }
}

/// KvCentroidPrefetch: `output[i] = distances[i] < threshold ? 1.0 : 0.0`
///
/// Produces a binary mask indicating which KV cache centroids are within
/// `threshold` distance and should be prefetched.
///
/// - `distances`: `[seq_len * num_heads]` f32 centroid distances
/// - `output`: `[seq_len * num_heads]` f32 binary prefetch mask
/// - `total`: seq_len * num_heads total elements
/// - `threshold`: distance threshold for prefetch decision
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_kv_centroid_prefetch(
    distances: *const f32,
    output: *mut f32,
    total: usize,
    threshold: f32,
) {
    if distances.is_null() || output.is_null() || total == 0 {
        return;
    }
    unsafe {
        for i in 0..total {
            *output.add(i) = if *distances.add(i) < threshold {
                1.0
            } else {
                0.0
            };
        }
    }
}

/// VariableLengthBatch: pack ragged sequences into a padded contiguous tensor.
///
/// Copies `lengths[i]` elements from each sequence in `input` to `output`,
/// zero-padding the remainder of each row to `max_len`.
///
/// - `input`: ragged input data (all sequences concatenated)
/// - `output`: `[num_seqs, max_len]` row-major padded output
/// - `lengths`: `[num_seqs]` f32 values representing integer sequence lengths
/// - `num_seqs`: number of sequences
/// - `max_len`: maximum sequence length (row stride in output)
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_variable_length_batch(
    input: *const f32,
    output: *mut f32,
    lengths: *const f32,
    num_seqs: usize,
    max_len: usize,
) {
    if input.is_null() || output.is_null() || lengths.is_null() || num_seqs == 0 || max_len == 0 {
        return;
    }
    unsafe {
        let mut input_offset = 0usize;
        for s in 0..num_seqs {
            let len = *lengths.add(s) as usize;
            let row_base = s * max_len;

            // Copy actual data
            for j in 0..len.min(max_len) {
                *output.add(row_base + j) = *input.add(input_offset + j);
            }

            // Zero-pad remainder
            for j in len.min(max_len)..max_len {
                *output.add(row_base + j) = 0.0;
            }

            input_offset += len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_gate_mask_basic() {
        let activation = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let gate = vec![0.5_f32, -0.1, 1.0, 0.0, 2.0];
        let mut output = vec![0.0_f32; 5];
        scalar_gate_mask(activation.as_ptr(), gate.as_ptr(), output.as_mut_ptr(), 5, 0.0);
        assert_eq!(output, vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    }

    #[test]
    fn test_scalar_gate_mask_threshold() {
        let activation = vec![10.0_f32, 20.0, 30.0];
        let gate = vec![0.3_f32, 0.5, 0.7];
        let mut output = vec![0.0_f32; 3];
        scalar_gate_mask(activation.as_ptr(), gate.as_ptr(), output.as_mut_ptr(), 3, 0.5);
        assert_eq!(output, vec![0.0, 0.0, 30.0]);
    }

    #[test]
    fn test_scalar_attention_skip_mask() {
        let tokens = vec![1.0_f32, 2.0, 0.0, 3.0, 0.0]; // 0 = pad
        let mut output = vec![0.0_f32; 5];
        scalar_attention_skip_mask(tokens.as_ptr(), output.as_mut_ptr(), 5, 0.0);
        assert_eq!(output, vec![1.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_scalar_layer_bypass() {
        let input = vec![1.0_f32, 2.0, 3.0];
        let mut output = vec![0.0_f32; 3];
        scalar_layer_bypass(input.as_ptr(), output.as_mut_ptr(), 3, 0.001);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scalar_softmax_with_entropy() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0_f32; 5]; // 4 softmax + 1 entropy
        scalar_softmax_with_entropy(logits.as_ptr(), output.as_mut_ptr(), 4);

        // Verify softmax sums to 1
        let sum: f32 = output[..4].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");

        // Verify entropy is positive (4 distinct values)
        let entropy = output[4];
        assert!(entropy > 0.0, "entropy should be positive, got {entropy}");

        // Verify monotonic (input is increasing)
        for i in 1..4 {
            assert!(output[i] > output[i - 1], "softmax not monotonic at {i}");
        }
    }

    #[test]
    fn test_scalar_softmax_with_entropy_uniform() {
        // Uniform logits -> uniform softmax -> maximum entropy
        let logits = vec![2.0_f32, 2.0, 2.0, 2.0];
        let mut output = vec![0.0_f32; 5];
        scalar_softmax_with_entropy(logits.as_ptr(), output.as_mut_ptr(), 4);

        // All softmax values should be 0.25
        for i in 0..4 {
            assert!((output[i] - 0.25).abs() < 1e-5, "uniform softmax[{i}] = {}", output[i]);
        }

        // Entropy should be ln(4) ≈ 1.386
        let expected_entropy = 4.0_f32.ln();
        assert!(
            (output[4] - expected_entropy).abs() < 1e-4,
            "entropy = {}, expected = {}",
            output[4], expected_entropy
        );
    }

    #[test]
    fn test_scalar_entropy_gate_low_entropy() {
        let entropy = vec![0.5_f32];
        let mut output = vec![0.0_f32; 1];
        scalar_entropy_gate(entropy.as_ptr(), output.as_mut_ptr(), 100, 1.0);
        assert_eq!(output[0], 1.0); // entropy < threshold -> write KV
    }

    #[test]
    fn test_scalar_entropy_gate_high_entropy() {
        let entropy = vec![5.0_f32];
        let mut output = vec![0.0_f32; 1];
        scalar_entropy_gate(entropy.as_ptr(), output.as_mut_ptr(), 100, 1.0);
        assert_eq!(output[0], 0.0); // entropy >= threshold -> skip KV
    }

    #[test]
    fn test_scalar_vrange_quant_roundtrip() {
        // 2-bit quantization, block_size=4
        let input = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = vec![0.0_f32; 8];
        scalar_vrange_quant(input.as_ptr(), output.as_mut_ptr(), 2, 4, 4, 2);

        // 2-bit -> 3 levels (0,1,2,3)
        // Block 0 [0,1,2,3]: min=0, max=3, range=3
        //   0 -> q=0 -> dq=0
        //   1 -> q=1 -> dq=1
        //   2 -> q=2 -> dq=2
        //   3 -> q=3 -> dq=3
        for i in 0..4 {
            assert!(
                (output[i] - input[i]).abs() < 0.01,
                "block0[{i}]: got {}, expected {}",
                output[i], input[i]
            );
        }

        // Block 1 [4,5,6,7]: min=4, max=7, range=3
        //   4 -> q=0 -> dq=4
        //   5 -> q=1 -> dq=5
        //   6 -> q=2 -> dq=6
        //   7 -> q=3 -> dq=7
        for i in 4..8 {
            assert!(
                (output[i] - input[i]).abs() < 0.01,
                "block1[{i}]: got {}, expected {}",
                output[i], input[i]
            );
        }
    }

    #[test]
    fn test_scalar_vrange_quant_quantization_error() {
        // 1-bit quantization of [0, 10] -> only levels 0 and 1
        let input = vec![0.0_f32, 10.0];
        let mut output = vec![0.0_f32; 2];
        scalar_vrange_quant(input.as_ptr(), output.as_mut_ptr(), 1, 2, 2, 1);

        // 1-bit -> 1 level -> q can be 0 or 1
        // 0 -> q=0 -> dq=0
        // 10 -> q=1 -> dq=10
        assert!((output[0] - 0.0).abs() < 0.01);
        assert!((output[1] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_scalar_kv_centroid_prefetch() {
        let distances = vec![0.1_f32, 0.5, 1.0, 0.01, 2.0];
        let mut output = vec![0.0_f32; 5];
        scalar_kv_centroid_prefetch(distances.as_ptr(), output.as_mut_ptr(), 5, 0.5);
        assert_eq!(output, vec![1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_scalar_variable_length_batch() {
        // Two sequences: lengths [2, 3], max_len=4
        let input = vec![1.0_f32, 2.0, 10.0, 20.0, 30.0]; // concatenated
        let lengths = vec![2.0_f32, 3.0_f32];
        let mut output = vec![0.0_f32; 8]; // 2 x 4
        scalar_variable_length_batch(
            input.as_ptr(), output.as_mut_ptr(), lengths.as_ptr(), 2, 4,
        );

        // Row 0: [1, 2, 0, 0]
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);

        // Row 1: [10, 20, 30, 0]
        assert_eq!(output[4], 10.0);
        assert_eq!(output[5], 20.0);
        assert_eq!(output[6], 30.0);
        assert_eq!(output[7], 0.0);
    }

    #[test]
    fn test_scalar_variable_length_batch_truncation() {
        // Sequence longer than max_len should be truncated
        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let lengths = vec![5.0_f32];
        let mut output = vec![0.0_f32; 3]; // max_len=3
        scalar_variable_length_batch(
            input.as_ptr(), output.as_mut_ptr(), lengths.as_ptr(), 1, 3,
        );

        // Row 0: [1, 2, 3] (truncated from 5 to 3)
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 3.0);
    }
}
