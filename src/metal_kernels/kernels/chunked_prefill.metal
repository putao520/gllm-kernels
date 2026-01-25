// Chunked Prefill / POD-Attention - Metal Shaders
// Based on Sarathi-Serve, DeepSpeed-FastGen, POD-Attention

#include <metal_stdlib>
using namespace metal;

#define MAX_HEAD_DIM 128
#define TILE_K 32

// Parameter structures
struct ChunkAttentionParams {
    uint batch_size;
    uint num_heads;
    uint head_dim;
    uint chunk_size;
    uint total_seq_len;
    uint chunk_idx;
    float scale;
    uint _pad0;
};

struct ChunkMergeParams {
    uint batch_size;
    uint num_heads;
    uint head_dim;
    uint num_chunks;
    uint chunk_size;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct PODSplitParams {
    uint batch_size;
    uint num_heads;
    uint head_dim;
    uint prefill_len;
    uint decode_len;
    float prefill_ratio;
    uint _pad0;
    uint _pad1;
};

// Chunked prefill attention forward (F32)
kernel void chunked_prefill_attention_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* lse [[buffer(4)]],
    constant ChunkAttentionParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.chunk_size;
    if (tid >= total) return;
    
    uint q_idx = tid % params.chunk_size;
    uint bh_idx = tid / params.chunk_size;
    uint batch_idx = bh_idx / params.num_heads;
    uint head_idx = bh_idx % params.num_heads;
    
    // Global query position
    uint global_q_pos = params.chunk_idx * params.chunk_size + q_idx;
    if (global_q_pos >= params.total_seq_len) return;
    
    // KV length (causal)
    uint kv_len = min(global_q_pos + 1, params.total_seq_len);
    
    // Q base offset
    uint q_base = batch_idx * params.chunk_size * params.num_heads * params.head_dim +
                  q_idx * params.num_heads * params.head_dim +
                  head_idx * params.head_dim;
    
    // KV base offset
    uint kv_batch_stride = params.total_seq_len * params.num_heads * params.head_dim;
    uint kv_head_stride = params.head_dim;
    uint kv_base = batch_idx * kv_batch_stride + head_idx * kv_head_stride;
    
    // Online softmax
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[MAX_HEAD_DIM];
    
    for (uint d = 0; d < params.head_dim && d < MAX_HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }
    
    // Process KV
    for (uint kv_pos = 0; kv_pos < kv_len; kv_pos++) {
        // Compute Q @ K^T
        float dot = 0.0f;
        uint k_offset = kv_base + kv_pos * params.num_heads * params.head_dim;
        
        for (uint d = 0; d < params.head_dim; d++) {
            dot += Q[q_base + d] * K[k_offset + d];
        }
        float score = dot * params.scale;
        
        // Online softmax update
        float new_max = max(max_val, score);
        float exp_diff = exp(max_val - new_max);
        float exp_score = exp(score - new_max);
        
        sum_exp = sum_exp * exp_diff + exp_score;
        
        // Update accumulator
        for (uint d = 0; d < params.head_dim; d++) {
            acc[d] = acc[d] * exp_diff + exp_score * V[k_offset + d];
        }
        
        max_val = new_max;
    }
    
    // Write output
    uint o_base = batch_idx * params.chunk_size * params.num_heads * params.head_dim +
                  q_idx * params.num_heads * params.head_dim +
                  head_idx * params.head_dim;
    
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint d = 0; d < params.head_dim; d++) {
        O[o_base + d] = acc[d] * inv_sum;
    }
    
    // Store log-sum-exp for chunk merging
    uint lse_idx = batch_idx * params.chunk_size * params.num_heads +
                   q_idx * params.num_heads +
                   head_idx;
    lse[lse_idx] = max_val + log(sum_exp + 1e-10f);
}

// Chunk merge (F32)
kernel void chunked_prefill_merge_f32(
    device const float* chunk_outputs [[buffer(0)]],
    device const float* chunk_lse [[buffer(1)]],
    device float* merged_output [[buffer(2)]],
    constant ChunkMergeParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total_len = params.num_chunks * params.chunk_size;
    uint total = params.batch_size * total_len * params.num_heads * params.head_dim;
    if (tid >= total) return;
    
    uint d = tid % params.head_dim;
    uint head_idx = (tid / params.head_dim) % params.num_heads;
    uint seq_idx = (tid / (params.head_dim * params.num_heads)) % total_len;
    uint batch_idx = tid / (params.head_dim * params.num_heads * total_len);
    
    uint chunk_idx = seq_idx / params.chunk_size;
    uint pos_in_chunk = seq_idx % params.chunk_size;
    
    // Copy from chunk output
    uint chunk_stride = params.batch_size * params.chunk_size * params.num_heads * params.head_dim;
    uint src_idx = chunk_idx * chunk_stride +
                   batch_idx * params.chunk_size * params.num_heads * params.head_dim +
                   pos_in_chunk * params.num_heads * params.head_dim +
                   head_idx * params.head_dim + d;
    
    merged_output[tid] = chunk_outputs[src_idx];
}

// POD-Attention workload split (F32)
kernel void pod_attention_split_f32(
    device const uint* prefill_lens [[buffer(0)]],
    device const uint* decode_lens [[buffer(1)]],
    device uint* prefill_allocation [[buffer(2)]],
    device uint* decode_allocation [[buffer(3)]],
    constant PODSplitParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    uint prefill_len = prefill_lens[tid];
    uint decode_len = decode_lens[tid];
    
    // Estimate compute cost
    uint prefill_cost = prefill_len * prefill_len;
    uint decode_cost = decode_len * params.prefill_len;
    uint total_cost = prefill_cost + decode_cost;
    
    // Allocate based on ratio
    uint prefill_work = uint(float(total_cost) * params.prefill_ratio);
    uint decode_work = total_cost - prefill_work;
    
    prefill_allocation[tid] = prefill_work;
    decode_allocation[tid] = decode_work;
}

// F16 variants
kernel void chunked_prefill_attention_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device half* lse [[buffer(4)]],
    constant ChunkAttentionParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.chunk_size;
    if (tid >= total) return;
    
    uint q_idx = tid % params.chunk_size;
    uint bh_idx = tid / params.chunk_size;
    uint batch_idx = bh_idx / params.num_heads;
    uint head_idx = bh_idx % params.num_heads;
    
    uint global_q_pos = params.chunk_idx * params.chunk_size + q_idx;
    if (global_q_pos >= params.total_seq_len) return;
    
    uint kv_len = min(global_q_pos + 1, params.total_seq_len);
    
    uint q_base = batch_idx * params.chunk_size * params.num_heads * params.head_dim +
                  q_idx * params.num_heads * params.head_dim +
                  head_idx * params.head_dim;
    
    uint kv_batch_stride = params.total_seq_len * params.num_heads * params.head_dim;
    uint kv_head_stride = params.head_dim;
    uint kv_base = batch_idx * kv_batch_stride + head_idx * kv_head_stride;
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[MAX_HEAD_DIM];
    
    for (uint d = 0; d < params.head_dim && d < MAX_HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }
    
    for (uint kv_pos = 0; kv_pos < kv_len; kv_pos++) {
        float dot = 0.0f;
        uint k_offset = kv_base + kv_pos * params.num_heads * params.head_dim;
        
        for (uint d = 0; d < params.head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * params.scale;
        
        float new_max = max(max_val, score);
        float exp_diff = exp(max_val - new_max);
        float exp_score = exp(score - new_max);
        
        sum_exp = sum_exp * exp_diff + exp_score;
        
        for (uint d = 0; d < params.head_dim; d++) {
            acc[d] = acc[d] * exp_diff + exp_score * float(V[k_offset + d]);
        }
        
        max_val = new_max;
    }
    
    uint o_base = batch_idx * params.chunk_size * params.num_heads * params.head_dim +
                  q_idx * params.num_heads * params.head_dim +
                  head_idx * params.head_dim;
    
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint d = 0; d < params.head_dim; d++) {
        O[o_base + d] = half(acc[d] * inv_sum);
    }
    
    uint lse_idx = batch_idx * params.chunk_size * params.num_heads +
                   q_idx * params.num_heads +
                   head_idx;
    lse[lse_idx] = half(max_val + log(sum_exp + 1e-10f));
}

kernel void chunked_prefill_merge_f16(
    device const half* chunk_outputs [[buffer(0)]],
    device const half* chunk_lse [[buffer(1)]],
    device half* merged_output [[buffer(2)]],
    constant ChunkMergeParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total_len = params.num_chunks * params.chunk_size;
    uint total = params.batch_size * total_len * params.num_heads * params.head_dim;
    if (tid >= total) return;
    
    uint d = tid % params.head_dim;
    uint head_idx = (tid / params.head_dim) % params.num_heads;
    uint seq_idx = (tid / (params.head_dim * params.num_heads)) % total_len;
    uint batch_idx = tid / (params.head_dim * params.num_heads * total_len);
    
    uint chunk_idx = seq_idx / params.chunk_size;
    uint pos_in_chunk = seq_idx % params.chunk_size;
    
    uint chunk_stride = params.batch_size * params.chunk_size * params.num_heads * params.head_dim;
    uint src_idx = chunk_idx * chunk_stride +
                   batch_idx * params.chunk_size * params.num_heads * params.head_dim +
                   pos_in_chunk * params.num_heads * params.head_dim +
                   head_idx * params.head_dim + d;
    
    merged_output[tid] = chunk_outputs[src_idx];
}
