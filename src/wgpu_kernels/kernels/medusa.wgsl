// Medusa Heads Parallel Token Prediction Kernels
// Based on Medusa (ICML'24): Multiple auxiliary heads for speculative decoding
//
// Kernels:
// - medusa_head_forward_f32: Forward pass through Medusa head (matmul + bias)
// - medusa_top_k_f32: Top-K selection for candidate tokens
// - medusa_log_softmax_f32: Log-softmax for confidence scores
// - medusa_build_candidates_f32: Build candidate tree from predictions
// - F16 variants for all kernels

enable f16;

const WORKGROUP_SIZE: u32 = 256u;
const TILE_SIZE: u32 = 16u;
const MAX_K: u32 = 64u;

// ============================================================================
// Parameters
// ============================================================================

struct HeadForwardParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    vocab_size: u32,
};

struct TopKParams {
    batch_size: u32,
    seq_len: u32,
    vocab_size: u32,
    k: u32,
};

struct CandidateParams {
    batch_size: u32,
    num_heads: u32,
    k: u32,
    max_candidates: u32,
};

// ============================================================================
// Medusa Head Forward Pass (F32)
// ============================================================================
// Computes logits = hidden @ weights + bias
// Input: hidden[batch, seq_len, hidden_dim], weights[hidden_dim, vocab_size], bias[vocab_size]
// Output: logits[batch, seq_len, vocab_size]

@group(0) @binding(0) var<storage, read> head_hidden_f32: array<f32>;
@group(0) @binding(1) var<storage, read> head_weights_f32: array<f32>;
@group(0) @binding(2) var<storage, read> head_bias_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> head_logits_f32: array<f32>;
@group(0) @binding(4) var<uniform> head_params_f32: HeadForwardParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn medusa_head_forward_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = head_params_f32.batch_size * head_params_f32.seq_len * head_params_f32.vocab_size;

    if (idx >= total) {
        return;
    }

    let vocab_idx = idx % head_params_f32.vocab_size;
    let batch_seq_idx = idx / head_params_f32.vocab_size;

    let hidden_base = batch_seq_idx * head_params_f32.hidden_dim;

    // Compute dot product: hidden[b,s,:] @ weights[:, v]
    var logit = head_bias_f32[vocab_idx];
    for (var d: u32 = 0u; d < head_params_f32.hidden_dim; d++) {
        let weight_idx = d * head_params_f32.vocab_size + vocab_idx;
        logit += head_hidden_f32[hidden_base + d] * head_weights_f32[weight_idx];
    }

    head_logits_f32[idx] = logit;
}

// ============================================================================
// Top-K Selection (F32)
// ============================================================================
// Finds top-K tokens from logits for each position
// Input: logits[batch, seq_len, vocab_size]
// Output: top_indices[batch, seq_len, k], top_values[batch, seq_len, k]

@group(0) @binding(0) var<storage, read> topk_logits_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> topk_indices_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_values_f32: array<f32>;
@group(0) @binding(3) var<uniform> topk_params_f32: TopKParams;

// Shared memory for partial results
var<workgroup> shared_indices: array<u32, 256>;
var<workgroup> shared_values: array<f32, 256>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn medusa_top_k_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_seq_idx = wg_id.x;
    let local_idx = local_id.x;

    let total_positions = topk_params_f32.batch_size * topk_params_f32.seq_len;
    if (batch_seq_idx >= total_positions) {
        return;
    }

    let logits_base = batch_seq_idx * topk_params_f32.vocab_size;
    let k = min(topk_params_f32.k, MAX_K);

    // Phase 1: Each thread finds its local max
    var local_max_val = -1e38f;
    var local_max_idx = 0u;

    var v = local_idx;
    while (v < topk_params_f32.vocab_size) {
        let val = topk_logits_f32[logits_base + v];
        if (val > local_max_val) {
            local_max_val = val;
            local_max_idx = v;
        }
        v += WORKGROUP_SIZE;
    }

    shared_indices[local_idx] = local_max_idx;
    shared_values[local_idx] = local_max_val;

    workgroupBarrier();

    // Phase 2: First thread extracts top-K using partial sort
    if (local_idx == 0u) {
        let output_base = batch_seq_idx * topk_params_f32.k;

        for (var i: u32 = 0u; i < k; i++) {
            var best_val = -1e38f;
            var best_idx = 0u;
            var best_slot = 0u;

            // Find max among all slots
            for (var j: u32 = 0u; j < WORKGROUP_SIZE; j++) {
                if (shared_values[j] > best_val) {
                    best_val = shared_values[j];
                    best_idx = shared_indices[j];
                    best_slot = j;
                }
            }

            topk_indices_f32[output_base + i] = best_idx;
            topk_values_f32[output_base + i] = best_val;

            // Mark as used
            shared_values[best_slot] = -1e38f;

            // For that slot, find next max from its range
            var next_max_val = -1e38f;
            var next_max_idx = 0u;
            var vv = best_slot;
            while (vv < topk_params_f32.vocab_size) {
                let val = topk_logits_f32[logits_base + vv];
                var already_selected = false;
                for (var check: u32 = 0u; check <= i; check++) {
                    if (topk_indices_f32[output_base + check] == vv) {
                        already_selected = true;
                        break;
                    }
                }
                if (!already_selected && val > next_max_val) {
                    next_max_val = val;
                    next_max_idx = vv;
                }
                vv += WORKGROUP_SIZE;
            }
            shared_indices[best_slot] = next_max_idx;
            shared_values[best_slot] = next_max_val;
        }
    }
}

// ============================================================================
// Log-Softmax (F32)
// ============================================================================
// Computes log-softmax for confidence scores
// Input: logits[batch, seq_len, vocab_size]
// Output: log_probs[batch, seq_len, vocab_size]

@group(0) @binding(0) var<storage, read> logsoftmax_logits_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> logsoftmax_output_f32: array<f32>;
@group(0) @binding(2) var<uniform> logsoftmax_params_f32: HeadForwardParams;

var<workgroup> softmax_max: f32;
var<workgroup> softmax_sum: f32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn medusa_log_softmax_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_seq_idx = wg_id.x;
    let local_idx = local_id.x;

    let total_positions = logsoftmax_params_f32.batch_size * logsoftmax_params_f32.seq_len;
    if (batch_seq_idx >= total_positions) {
        return;
    }

    let base = batch_seq_idx * logsoftmax_params_f32.vocab_size;

    // Phase 1: Find max
    var local_max = -1e38f;
    var v = local_idx;
    while (v < logsoftmax_params_f32.vocab_size) {
        local_max = max(local_max, logsoftmax_logits_f32[base + v]);
        v += WORKGROUP_SIZE;
    }

    // Reduce max (simplified)
    if (local_idx == 0u) {
        var g_max = -1e38f;
        for (var i: u32 = 0u; i < logsoftmax_params_f32.vocab_size; i++) {
            g_max = max(g_max, logsoftmax_logits_f32[base + i]);
        }
        softmax_max = g_max;
    }

    workgroupBarrier();

    // Phase 2: Compute sum of exp
    if (local_idx == 0u) {
        var g_sum = 0.0f;
        for (var i: u32 = 0u; i < logsoftmax_params_f32.vocab_size; i++) {
            g_sum += exp(logsoftmax_logits_f32[base + i] - softmax_max);
        }
        softmax_sum = g_sum;
    }

    workgroupBarrier();

    // Phase 3: Compute log-softmax
    let log_sum = log(softmax_sum);
    v = local_idx;
    while (v < logsoftmax_params_f32.vocab_size) {
        let logit = logsoftmax_logits_f32[base + v];
        logsoftmax_output_f32[base + v] = logit - softmax_max - log_sum;
        v += WORKGROUP_SIZE;
    }
}

// ============================================================================
// Build Candidate Tree (F32)
// ============================================================================
// Builds candidate tree from multi-head predictions
// Input: top_indices[batch, num_heads, k] from each head
// Output: candidates[batch, max_candidates], num_candidates[batch]

@group(0) @binding(0) var<storage, read> cand_top_indices_f32: array<u32>;
@group(0) @binding(1) var<storage, read_write> cand_output_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> cand_count_f32: array<u32>;
@group(0) @binding(3) var<uniform> cand_params_f32: CandidateParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn medusa_build_candidates_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let batch_idx = global_id.x;

    if (batch_idx >= cand_params_f32.batch_size) {
        return;
    }

    let k = cand_params_f32.k;
    let num_heads = cand_params_f32.num_heads;
    let max_cand = cand_params_f32.max_candidates;

    let input_stride = num_heads * k;
    let input_base = batch_idx * input_stride;
    let output_base = batch_idx * max_cand;

    // Simple greedy tree building: take top-1 from each head as primary path
    // Then add alternates
    var count = 0u;

    // Primary path: top-1 from each head
    for (var h: u32 = 0u; h < num_heads && count < max_cand; h++) {
        let idx = input_base + h * k;
        cand_output_f32[output_base + count] = cand_top_indices_f32[idx];
        count++;
    }

    // Secondary paths: top-2 to top-k from first head
    for (var i: u32 = 1u; i < k && count < max_cand; i++) {
        cand_output_f32[output_base + count] = cand_top_indices_f32[input_base + i];
        count++;
    }

    // Fill remaining with top candidates from other heads
    for (var h: u32 = 1u; h < num_heads; h++) {
        for (var i: u32 = 1u; i < k && count < max_cand; i++) {
            let idx = input_base + h * k + i;
            cand_output_f32[output_base + count] = cand_top_indices_f32[idx];
            count++;
        }
    }

    cand_count_f32[batch_idx] = count;
}

// ============================================================================
// F16 Medusa Head Forward Pass
// ============================================================================

@group(0) @binding(0) var<storage, read> head_hidden_f16: array<f16>;
@group(0) @binding(1) var<storage, read> head_weights_f16: array<f16>;
@group(0) @binding(2) var<storage, read> head_bias_f16: array<f16>;
@group(0) @binding(3) var<storage, read_write> head_logits_f16: array<f16>;
@group(0) @binding(4) var<uniform> head_params_f16: HeadForwardParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn medusa_head_forward_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = head_params_f16.batch_size * head_params_f16.seq_len * head_params_f16.vocab_size;

    if (idx >= total) {
        return;
    }

    let vocab_idx = idx % head_params_f16.vocab_size;
    let batch_seq_idx = idx / head_params_f16.vocab_size;

    let hidden_base = batch_seq_idx * head_params_f16.hidden_dim;

    // Compute in f32 for precision
    var logit = f32(head_bias_f16[vocab_idx]);
    for (var d: u32 = 0u; d < head_params_f16.hidden_dim; d++) {
        let weight_idx = d * head_params_f16.vocab_size + vocab_idx;
        logit += f32(head_hidden_f16[hidden_base + d]) * f32(head_weights_f16[weight_idx]);
    }

    head_logits_f16[idx] = f16(logit);
}

// ============================================================================
// F16 Top-K Selection
// ============================================================================

@group(0) @binding(0) var<storage, read> topk_logits_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write> topk_indices_f16: array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_values_f16: array<f16>;
@group(0) @binding(3) var<uniform> topk_params_f16: TopKParams;

var<workgroup> shared_indices_f16: array<u32, 256>;
var<workgroup> shared_values_f16: array<f32, 256>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn medusa_top_k_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_seq_idx = wg_id.x;
    let local_idx = local_id.x;

    let total_positions = topk_params_f16.batch_size * topk_params_f16.seq_len;
    if (batch_seq_idx >= total_positions) {
        return;
    }

    let logits_base = batch_seq_idx * topk_params_f16.vocab_size;
    let k = min(topk_params_f16.k, MAX_K);

    // Find local max
    var local_max_val = -1e38f;
    var local_max_idx = 0u;

    var v = local_idx;
    while (v < topk_params_f16.vocab_size) {
        let val = f32(topk_logits_f16[logits_base + v]);
        if (val > local_max_val) {
            local_max_val = val;
            local_max_idx = v;
        }
        v += WORKGROUP_SIZE;
    }

    shared_indices_f16[local_idx] = local_max_idx;
    shared_values_f16[local_idx] = local_max_val;

    workgroupBarrier();

    if (local_idx == 0u) {
        let output_base = batch_seq_idx * topk_params_f16.k;

        for (var i: u32 = 0u; i < k; i++) {
            var best_val = -1e38f;
            var best_idx = 0u;
            var best_slot = 0u;

            for (var j: u32 = 0u; j < WORKGROUP_SIZE; j++) {
                if (shared_values_f16[j] > best_val) {
                    best_val = shared_values_f16[j];
                    best_idx = shared_indices_f16[j];
                    best_slot = j;
                }
            }

            topk_indices_f16[output_base + i] = best_idx;
            topk_values_f16[output_base + i] = f16(best_val);

            shared_values_f16[best_slot] = -1e38f;

            var next_max_val = -1e38f;
            var next_max_idx = 0u;
            var vv = best_slot;
            while (vv < topk_params_f16.vocab_size) {
                let val = f32(topk_logits_f16[logits_base + vv]);
                var already_selected = false;
                for (var check: u32 = 0u; check <= i; check++) {
                    if (topk_indices_f16[output_base + check] == vv) {
                        already_selected = true;
                        break;
                    }
                }
                if (!already_selected && val > next_max_val) {
                    next_max_val = val;
                    next_max_idx = vv;
                }
                vv += WORKGROUP_SIZE;
            }
            shared_indices_f16[best_slot] = next_max_idx;
            shared_values_f16[best_slot] = next_max_val;
        }
    }
}

// ============================================================================
// F16 Log-Softmax
// ============================================================================

@group(0) @binding(0) var<storage, read> logsoftmax_logits_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write> logsoftmax_output_f16: array<f16>;
@group(0) @binding(2) var<uniform> logsoftmax_params_f16: HeadForwardParams;

var<workgroup> softmax_max_f16: f32;
var<workgroup> softmax_sum_f16: f32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn medusa_log_softmax_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_seq_idx = wg_id.x;
    let local_idx = local_id.x;

    let total_positions = logsoftmax_params_f16.batch_size * logsoftmax_params_f16.seq_len;
    if (batch_seq_idx >= total_positions) {
        return;
    }

    let base = batch_seq_idx * logsoftmax_params_f16.vocab_size;

    if (local_idx == 0u) {
        var g_max = -1e38f;
        for (var i: u32 = 0u; i < logsoftmax_params_f16.vocab_size; i++) {
            g_max = max(g_max, f32(logsoftmax_logits_f16[base + i]));
        }
        softmax_max_f16 = g_max;
    }

    workgroupBarrier();

    if (local_idx == 0u) {
        var g_sum = 0.0f;
        for (var i: u32 = 0u; i < logsoftmax_params_f16.vocab_size; i++) {
            g_sum += exp(f32(logsoftmax_logits_f16[base + i]) - softmax_max_f16);
        }
        softmax_sum_f16 = g_sum;
    }

    workgroupBarrier();

    let log_sum = log(softmax_sum_f16);
    var v = local_idx;
    while (v < logsoftmax_params_f16.vocab_size) {
        let logit = f32(logsoftmax_logits_f16[base + v]);
        logsoftmax_output_f16[base + v] = f16(logit - softmax_max_f16 - log_sum);
        v += WORKGROUP_SIZE;
    }
}
