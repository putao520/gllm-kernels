// Chunked Prefill / POD-Attention Kernels
// Based on Sarathi-Serve, DeepSpeed-FastGen, POD-Attention
//
// Kernels:
// - chunked_prefill_attention_f32: Chunked attention computation
// - chunked_prefill_merge_f32: Merge chunk outputs
// - chunked_prefill_schedule_f32: Batch scheduling primitives
// - pod_attention_split_f32: POD-Attention workload splitting
// - F16 variants

enable f16;

const WORKGROUP_SIZE: u32 = 256u;
const TILE_K: u32 = 32u;
const TILE_V: u32 = 32u;

// ============================================================================
// Parameters
// ============================================================================

struct ChunkAttentionParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    chunk_size: u32,
    total_seq_len: u32,
    chunk_idx: u32,
    scale: f32,
    _pad0: u32,
};

struct ChunkMergeParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    num_chunks: u32,
    chunk_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct PODSplitParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    prefill_len: u32,
    decode_len: u32,
    prefill_ratio: f32,  // 0.0-1.0, fraction of compute for prefill
    _pad0: u32,
    _pad1: u32,
};

struct ScheduleParams {
    num_requests: u32,
    max_batch_size: u32,
    max_seq_len: u32,
    chunk_size: u32,
};

// ============================================================================
// Chunked Prefill Attention (F32)
// ============================================================================
// Computes attention for a single chunk of the sequence
// Input: Q[batch, chunk_size, heads, dim], K[batch, kv_len, heads, dim], V[batch, kv_len, heads, dim]
// Output: O[batch, chunk_size, heads, dim], lse[batch, chunk_size, heads] (log-sum-exp for merging)

@group(0) @binding(0) var<storage, read> chunk_q_f32: array<f32>;
@group(0) @binding(1) var<storage, read> chunk_k_f32: array<f32>;
@group(0) @binding(2) var<storage, read> chunk_v_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> chunk_o_f32: array<f32>;
@group(0) @binding(4) var<storage, read_write> chunk_lse_f32: array<f32>;
@group(0) @binding(5) var<uniform> chunk_params_f32: ChunkAttentionParams;

var<workgroup> shared_k: array<f32, 1024>;  // TILE_K * head_dim
var<workgroup> shared_v: array<f32, 1024>;  // TILE_V * head_dim

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn chunked_prefill_attention_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_head_idx = wg_id.x;
    let q_idx = wg_id.y;  // Query position within chunk
    let local_idx = local_id.x;

    let batch_idx = batch_head_idx / chunk_params_f32.num_heads;
    let head_idx = batch_head_idx % chunk_params_f32.num_heads;

    if (batch_idx >= chunk_params_f32.batch_size || q_idx >= chunk_params_f32.chunk_size) {
        return;
    }

    let head_dim = chunk_params_f32.head_dim;
    let scale = chunk_params_f32.scale;

    // Global query position
    let global_q_pos = chunk_params_f32.chunk_idx * chunk_params_f32.chunk_size + q_idx;
    if (global_q_pos >= chunk_params_f32.total_seq_len) {
        return;
    }

    // KV length for this chunk (causal: only attend to positions <= current)
    let kv_len = min(global_q_pos + 1u, chunk_params_f32.total_seq_len);

    // Load Q for this position
    let q_base = batch_idx * chunk_params_f32.chunk_size * chunk_params_f32.num_heads * head_dim +
                 q_idx * chunk_params_f32.num_heads * head_dim +
                 head_idx * head_dim;

    // Online softmax accumulators
    var max_val = -1e38f;
    var sum_exp = 0.0f;
    var acc = array<f32, 128>();  // Assume max head_dim = 128

    // Initialize accumulator
    for (var d: u32 = 0u; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Process KV in tiles
    var kv_start = 0u;
    while (kv_start < kv_len) {
        let tile_len = min(TILE_K, kv_len - kv_start);

        // Each thread loads part of K tile
        if (local_idx < tile_len * head_dim) {
            let k_pos = kv_start + local_idx / head_dim;
            let d = local_idx % head_dim;
            let k_idx = batch_idx * chunk_params_f32.total_seq_len * chunk_params_f32.num_heads * head_dim +
                        k_pos * chunk_params_f32.num_heads * head_dim +
                        head_idx * head_dim + d;
            shared_k[local_idx] = chunk_k_f32[k_idx];
        }

        workgroupBarrier();

        // Compute attention scores for this tile
        if (local_idx == 0u) {
            for (var t: u32 = 0u; t < tile_len; t++) {
                // Dot product Q @ K^T
                var score = 0.0f;
                for (var d: u32 = 0u; d < head_dim; d++) {
                    score += chunk_q_f32[q_base + d] * shared_k[t * head_dim + d];
                }
                score *= scale;

                // Online softmax update
                let new_max = max(max_val, score);
                let exp_diff = exp(max_val - new_max);
                let exp_score = exp(score - new_max);

                // Update accumulators
                sum_exp = sum_exp * exp_diff + exp_score;

                // Load V and accumulate
                let v_pos = kv_start + t;
                let v_base = batch_idx * chunk_params_f32.total_seq_len * chunk_params_f32.num_heads * head_dim +
                             v_pos * chunk_params_f32.num_heads * head_dim +
                             head_idx * head_dim;

                for (var d: u32 = 0u; d < head_dim; d++) {
                    acc[d] = acc[d] * exp_diff + exp_score * chunk_v_f32[v_base + d];
                }

                max_val = new_max;
            }
        }

        workgroupBarrier();
        kv_start += TILE_K;
    }

    // Write output
    if (local_idx == 0u) {
        let o_base = batch_idx * chunk_params_f32.chunk_size * chunk_params_f32.num_heads * head_dim +
                     q_idx * chunk_params_f32.num_heads * head_dim +
                     head_idx * head_dim;

        let inv_sum = 1.0 / sum_exp;
        for (var d: u32 = 0u; d < head_dim; d++) {
            chunk_o_f32[o_base + d] = acc[d] * inv_sum;
        }

        // Store log-sum-exp for chunk merging
        let lse_idx = batch_idx * chunk_params_f32.chunk_size * chunk_params_f32.num_heads +
                      q_idx * chunk_params_f32.num_heads +
                      head_idx;
        chunk_lse_f32[lse_idx] = max_val + log(sum_exp);
    }
}

// ============================================================================
// Chunk Merge (F32)
// ============================================================================
// Merges outputs from multiple chunks using log-sum-exp
// Input: chunk_outputs[num_chunks, batch, chunk_size, heads, dim]
//        chunk_lse[num_chunks, batch, chunk_size, heads]
// Output: merged_output[batch, total_len, heads, dim]

@group(0) @binding(0) var<storage, read> merge_chunks_f32: array<f32>;
@group(0) @binding(1) var<storage, read> merge_lse_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> merge_output_f32: array<f32>;
@group(0) @binding(3) var<uniform> merge_params_f32: ChunkMergeParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn chunked_prefill_merge_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_len = merge_params_f32.num_chunks * merge_params_f32.chunk_size;
    let total = merge_params_f32.batch_size * total_len * merge_params_f32.num_heads * merge_params_f32.head_dim;

    if (idx >= total) {
        return;
    }

    let d = idx % merge_params_f32.head_dim;
    let head_idx = (idx / merge_params_f32.head_dim) % merge_params_f32.num_heads;
    let seq_idx = (idx / (merge_params_f32.head_dim * merge_params_f32.num_heads)) % total_len;
    let batch_idx = idx / (merge_params_f32.head_dim * merge_params_f32.num_heads * total_len);

    let chunk_idx = seq_idx / merge_params_f32.chunk_size;
    let pos_in_chunk = seq_idx % merge_params_f32.chunk_size;

    // For single chunk, just copy
    // For overlapping chunks, would need weighted merge
    let chunk_stride = merge_params_f32.batch_size * merge_params_f32.chunk_size *
                       merge_params_f32.num_heads * merge_params_f32.head_dim;

    let src_idx = chunk_idx * chunk_stride +
                  batch_idx * merge_params_f32.chunk_size * merge_params_f32.num_heads * merge_params_f32.head_dim +
                  pos_in_chunk * merge_params_f32.num_heads * merge_params_f32.head_dim +
                  head_idx * merge_params_f32.head_dim + d;

    merge_output_f32[idx] = merge_chunks_f32[src_idx];
}

// ============================================================================
// POD-Attention Workload Split (F32)
// ============================================================================
// Splits workload between prefill and decode for interleaved execution
// Input: prefill_mask[batch, prefill_len], decode_indices[batch, decode_len]
// Output: sm_allocation[num_sms], prefill_work[...], decode_work[...]

@group(0) @binding(0) var<storage, read> pod_prefill_lens_f32: array<u32>;
@group(0) @binding(1) var<storage, read> pod_decode_lens_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> pod_prefill_allocation_f32: array<u32>;
@group(0) @binding(3) var<storage, read_write> pod_decode_allocation_f32: array<u32>;
@group(0) @binding(4) var<uniform> pod_params_f32: PODSplitParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn pod_attention_split_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let batch_idx = global_id.x;

    if (batch_idx >= pod_params_f32.batch_size) {
        return;
    }

    let prefill_len = pod_prefill_lens_f32[batch_idx];
    let decode_len = pod_decode_lens_f32[batch_idx];

    // Estimate compute cost (quadratic for prefill, linear for decode)
    let prefill_cost = prefill_len * prefill_len;
    let decode_cost = decode_len * pod_params_f32.prefill_len;  // decode attends to all prefill

    let total_cost = prefill_cost + decode_cost;

    // Allocate work based on ratio
    let prefill_work = u32(f32(total_cost) * pod_params_f32.prefill_ratio);
    let decode_work = total_cost - prefill_work;

    pod_prefill_allocation_f32[batch_idx] = prefill_work;
    pod_decode_allocation_f32[batch_idx] = decode_work;
}

// ============================================================================
// Batch Scheduling (F32)
// ============================================================================
// Schedules requests into batches based on chunk sizes
// Input: request_lens[num_requests]
// Output: batch_assignments[num_requests], batch_offsets[max_batch_size]

@group(0) @binding(0) var<storage, read> sched_request_lens_f32: array<u32>;
@group(0) @binding(1) var<storage, read_write> sched_batch_assign_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> sched_batch_offsets_f32: array<u32>;
@group(0) @binding(3) var<storage, read_write> sched_num_batches_f32: array<u32>;
@group(0) @binding(4) var<uniform> sched_params_f32: ScheduleParams;

@compute @workgroup_size(1, 1, 1)
fn chunked_prefill_schedule_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Simple first-fit scheduling
    var current_batch = 0u;
    var current_batch_len = 0u;

    for (var r: u32 = 0u; r < sched_params_f32.num_requests; r++) {
        let req_len = sched_request_lens_f32[r];
        let chunks_needed = (req_len + sched_params_f32.chunk_size - 1u) / sched_params_f32.chunk_size;

        // Check if fits in current batch
        if (current_batch_len + chunks_needed > sched_params_f32.max_seq_len / sched_params_f32.chunk_size) {
            // Start new batch
            current_batch++;
            current_batch_len = 0u;
            if (current_batch >= sched_params_f32.max_batch_size) {
                // Can't schedule more
                sched_batch_assign_f32[r] = 0xFFFFFFFFu;
                continue;
            }
        }

        sched_batch_assign_f32[r] = current_batch;
        sched_batch_offsets_f32[current_batch * sched_params_f32.max_batch_size + r] = current_batch_len;
        current_batch_len += chunks_needed;
    }

    sched_num_batches_f32[0] = current_batch + 1u;
}

// ============================================================================
// F16 Chunked Prefill Attention
// ============================================================================

@group(0) @binding(0) var<storage, read> chunk_q_f16: array<f16>;
@group(0) @binding(1) var<storage, read> chunk_k_f16: array<f16>;
@group(0) @binding(2) var<storage, read> chunk_v_f16: array<f16>;
@group(0) @binding(3) var<storage, read_write> chunk_o_f16: array<f16>;
@group(0) @binding(4) var<storage, read_write> chunk_lse_f16: array<f16>;
@group(0) @binding(5) var<uniform> chunk_params_f16: ChunkAttentionParams;

var<workgroup> shared_k_f16: array<f32, 1024>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn chunked_prefill_attention_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_head_idx = wg_id.x;
    let q_idx = wg_id.y;
    let local_idx = local_id.x;

    let batch_idx = batch_head_idx / chunk_params_f16.num_heads;
    let head_idx = batch_head_idx % chunk_params_f16.num_heads;

    if (batch_idx >= chunk_params_f16.batch_size || q_idx >= chunk_params_f16.chunk_size) {
        return;
    }

    let head_dim = chunk_params_f16.head_dim;
    let scale = chunk_params_f16.scale;

    let global_q_pos = chunk_params_f16.chunk_idx * chunk_params_f16.chunk_size + q_idx;
    if (global_q_pos >= chunk_params_f16.total_seq_len) {
        return;
    }

    let kv_len = min(global_q_pos + 1u, chunk_params_f16.total_seq_len);

    let q_base = batch_idx * chunk_params_f16.chunk_size * chunk_params_f16.num_heads * head_dim +
                 q_idx * chunk_params_f16.num_heads * head_dim +
                 head_idx * head_dim;

    var max_val = -1e38f;
    var sum_exp = 0.0f;
    var acc = array<f32, 128>();

    for (var d: u32 = 0u; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    var kv_start = 0u;
    while (kv_start < kv_len) {
        let tile_len = min(TILE_K, kv_len - kv_start);

        if (local_idx < tile_len * head_dim) {
            let k_pos = kv_start + local_idx / head_dim;
            let d = local_idx % head_dim;
            let k_idx = batch_idx * chunk_params_f16.total_seq_len * chunk_params_f16.num_heads * head_dim +
                        k_pos * chunk_params_f16.num_heads * head_dim +
                        head_idx * head_dim + d;
            shared_k_f16[local_idx] = f32(chunk_k_f16[k_idx]);
        }

        workgroupBarrier();

        if (local_idx == 0u) {
            for (var t: u32 = 0u; t < tile_len; t++) {
                var score = 0.0f;
                for (var d: u32 = 0u; d < head_dim; d++) {
                    score += f32(chunk_q_f16[q_base + d]) * shared_k_f16[t * head_dim + d];
                }
                score *= scale;

                let new_max = max(max_val, score);
                let exp_diff = exp(max_val - new_max);
                let exp_score = exp(score - new_max);

                sum_exp = sum_exp * exp_diff + exp_score;

                let v_pos = kv_start + t;
                let v_base = batch_idx * chunk_params_f16.total_seq_len * chunk_params_f16.num_heads * head_dim +
                             v_pos * chunk_params_f16.num_heads * head_dim +
                             head_idx * head_dim;

                for (var d: u32 = 0u; d < head_dim; d++) {
                    acc[d] = acc[d] * exp_diff + exp_score * f32(chunk_v_f16[v_base + d]);
                }

                max_val = new_max;
            }
        }

        workgroupBarrier();
        kv_start += TILE_K;
    }

    if (local_idx == 0u) {
        let o_base = batch_idx * chunk_params_f16.chunk_size * chunk_params_f16.num_heads * head_dim +
                     q_idx * chunk_params_f16.num_heads * head_dim +
                     head_idx * head_dim;

        let inv_sum = 1.0 / sum_exp;
        for (var d: u32 = 0u; d < head_dim; d++) {
            chunk_o_f16[o_base + d] = f16(acc[d] * inv_sum);
        }

        let lse_idx = batch_idx * chunk_params_f16.chunk_size * chunk_params_f16.num_heads +
                      q_idx * chunk_params_f16.num_heads +
                      head_idx;
        chunk_lse_f16[lse_idx] = f16(max_val + log(sum_exp));
    }
}

// ============================================================================
// F16 Chunk Merge
// ============================================================================

@group(0) @binding(0) var<storage, read> merge_chunks_f16: array<f16>;
@group(0) @binding(1) var<storage, read> merge_lse_f16: array<f16>;
@group(0) @binding(2) var<storage, read_write> merge_output_f16: array<f16>;
@group(0) @binding(3) var<uniform> merge_params_f16: ChunkMergeParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn chunked_prefill_merge_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_len = merge_params_f16.num_chunks * merge_params_f16.chunk_size;
    let total = merge_params_f16.batch_size * total_len * merge_params_f16.num_heads * merge_params_f16.head_dim;

    if (idx >= total) {
        return;
    }

    let d = idx % merge_params_f16.head_dim;
    let head_idx = (idx / merge_params_f16.head_dim) % merge_params_f16.num_heads;
    let seq_idx = (idx / (merge_params_f16.head_dim * merge_params_f16.num_heads)) % total_len;
    let batch_idx = idx / (merge_params_f16.head_dim * merge_params_f16.num_heads * total_len);

    let chunk_idx = seq_idx / merge_params_f16.chunk_size;
    let pos_in_chunk = seq_idx % merge_params_f16.chunk_size;

    let chunk_stride = merge_params_f16.batch_size * merge_params_f16.chunk_size *
                       merge_params_f16.num_heads * merge_params_f16.head_dim;

    let src_idx = chunk_idx * chunk_stride +
                  batch_idx * merge_params_f16.chunk_size * merge_params_f16.num_heads * merge_params_f16.head_dim +
                  pos_in_chunk * merge_params_f16.num_heads * merge_params_f16.head_dim +
                  head_idx * merge_params_f16.head_dim + d;

    merge_output_f16[idx] = merge_chunks_f16[src_idx];
}
