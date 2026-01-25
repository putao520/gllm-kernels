// Chunked Prefill/POD-Attention CUDA kernels.
//
// This file contains CUDA kernels for chunked prefill:
// - Chunked attention computation for long sequences
// - Online softmax with log-sum-exp for chunk merging
// - POD-Attention workload splitting (prefill vs decode)
// - Batch scheduling for mixed workloads
//
// Ported from chunked_prefill.hip

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>  // for FLT_MAX
#include <cmath>

// Parameters for chunked attention
struct ChunkedAttnParams {
    unsigned int batch_size;
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int seq_len;
    unsigned int chunk_size;
    unsigned int num_chunks;
    float scale;
    unsigned int _pad;
};

// Parameters for chunk merging
struct MergeParams {
    unsigned int batch_size;
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int num_chunks;
};

// Parameters for POD splitting
struct PODSplitParams {
    unsigned int batch_size;
    unsigned int num_prefill;
    unsigned int num_decode;
    unsigned int max_prefill_len;
    unsigned int _pad;
};

// Parameters for batch scheduling
struct ScheduleParams {
    unsigned int num_requests;
    unsigned int max_batch_size;
    unsigned int prefill_budget;
    unsigned int decode_budget;
};

// Chunked attention (FP32) - process one chunk at a time
// Returns partial output and log-sum-exp for later merging
extern "C" __global__ void chunked_attention_f32(
    const float* __restrict__ query,       // [batch, num_heads, head_dim]
    const float* __restrict__ key,         // [batch, seq_len, num_heads, head_dim]
    const float* __restrict__ value,       // [batch, seq_len, num_heads, head_dim]
    float* __restrict__ chunk_output,      // [batch, num_chunks, num_heads, head_dim]
    float* __restrict__ chunk_lse,         // [batch, num_chunks, num_heads]
    ChunkedAttnParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int chunk_id = blockIdx.z;
    unsigned int dim_id = threadIdx.x;

    if (batch_id >= params.batch_size || head_id >= params.num_heads || chunk_id >= params.num_chunks) return;

    unsigned int chunk_start = chunk_id * params.chunk_size;
    unsigned int chunk_end = min(chunk_start + params.chunk_size, params.seq_len);

    // Load query for this head
    __shared__ float s_query[128];  // Assume head_dim <= 128
    if (dim_id < params.head_dim) {
        s_query[dim_id] = query[batch_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id];
    }
    __syncthreads();

    // Compute attention scores for this chunk
    __shared__ float s_scores[256];  // Max chunk size
    __shared__ float s_max;
    __shared__ float s_sum;

    if (threadIdx.x == 0) {
        s_max = -INFINITY;
        s_sum = 0.0f;
    }
    __syncthreads();

    // Compute scores: Q @ K^T * scale
    for (unsigned int k = chunk_start + threadIdx.x; k < chunk_end; k += blockDim.x) {
        float score = 0.0f;
        for (unsigned int d = 0; d < params.head_dim; d++) {
            float q = s_query[d];
            float kv = key[batch_id * params.seq_len * params.num_heads * params.head_dim
                         + k * params.num_heads * params.head_dim + head_id * params.head_dim + d];
            score += q * kv;
        }
        score *= params.scale;
        s_scores[k - chunk_start] = score;
        atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(score));
    }
    __syncthreads();

    // Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (unsigned int k = chunk_start + threadIdx.x; k < chunk_end; k += blockDim.x) {
        float score = s_scores[k - chunk_start];
        float exp_score = expf(score - s_max);
        s_scores[k - chunk_start] = exp_score;
        local_sum += exp_score;
    }
    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    // Compute weighted sum of values
    float output_val = 0.0f;
    if (dim_id < params.head_dim) {
        for (unsigned int k = chunk_start; k < chunk_end; k++) {
            float weight = s_scores[k - chunk_start] / s_sum;
            float v = value[batch_id * params.seq_len * params.num_heads * params.head_dim
                          + k * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id];
            output_val += weight * v;
        }

        // Store chunk output
        unsigned int out_idx = batch_id * params.num_chunks * params.num_heads * params.head_dim
                             + chunk_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
        chunk_output[out_idx] = output_val;
    }

    // Store log-sum-exp for this chunk
    if (threadIdx.x == 0) {
        float lse = s_max + logf(s_sum);
        unsigned int lse_idx = batch_id * params.num_chunks * params.num_heads + chunk_id * params.num_heads + head_id;
        chunk_lse[lse_idx] = lse;
    }
}

// Chunked attention (FP16)
extern "C" __global__ void chunked_attention_f16(
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    __half* __restrict__ chunk_output,
    float* __restrict__ chunk_lse,
    ChunkedAttnParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int chunk_id = blockIdx.z;
    unsigned int dim_id = threadIdx.x;

    if (batch_id >= params.batch_size || head_id >= params.num_heads || chunk_id >= params.num_chunks) return;

    unsigned int chunk_start = chunk_id * params.chunk_size;
    unsigned int chunk_end = min(chunk_start + params.chunk_size, params.seq_len);

    __shared__ float s_query[128];
    if (dim_id < params.head_dim) {
        s_query[dim_id] = __half2float(query[batch_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id]);
    }
    __syncthreads();

    __shared__ float s_scores[256];
    __shared__ float s_max;
    __shared__ float s_sum;

    if (threadIdx.x == 0) {
        s_max = -INFINITY;
        s_sum = 0.0f;
    }
    __syncthreads();

    for (unsigned int k = chunk_start + threadIdx.x; k < chunk_end; k += blockDim.x) {
        float score = 0.0f;
        for (unsigned int d = 0; d < params.head_dim; d++) {
            float q = s_query[d];
            float kv = __half2float(key[batch_id * params.seq_len * params.num_heads * params.head_dim
                         + k * params.num_heads * params.head_dim + head_id * params.head_dim + d]);
            score += q * kv;
        }
        score *= params.scale;
        s_scores[k - chunk_start] = score;
        atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(score));
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned int k = chunk_start + threadIdx.x; k < chunk_end; k += blockDim.x) {
        float score = s_scores[k - chunk_start];
        float exp_score = expf(score - s_max);
        s_scores[k - chunk_start] = exp_score;
        local_sum += exp_score;
    }
    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    float output_val = 0.0f;
    if (dim_id < params.head_dim) {
        for (unsigned int k = chunk_start; k < chunk_end; k++) {
            float weight = s_scores[k - chunk_start] / s_sum;
            float v = __half2float(value[batch_id * params.seq_len * params.num_heads * params.head_dim
                          + k * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id]);
            output_val += weight * v;
        }

        unsigned int out_idx = batch_id * params.num_chunks * params.num_heads * params.head_dim
                             + chunk_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
        chunk_output[out_idx] = __float2half(output_val);
    }

    if (threadIdx.x == 0) {
        float lse = s_max + logf(s_sum);
        unsigned int lse_idx = batch_id * params.num_chunks * params.num_heads + chunk_id * params.num_heads + head_id;
        chunk_lse[lse_idx] = lse;
    }
}

// Merge chunk outputs using log-sum-exp (FP32)
extern "C" __global__ void merge_chunks_f32(
    const float* __restrict__ chunk_outputs,  // [batch, num_chunks, num_heads, head_dim]
    const float* __restrict__ chunk_lse,      // [batch, num_chunks, num_heads]
    float* __restrict__ output,               // [batch, num_heads, head_dim]
    MergeParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int dim_id = threadIdx.x;

    if (batch_id >= params.batch_size || head_id >= params.num_heads || dim_id >= params.head_dim) return;

    // Load LSE values for all chunks
    __shared__ float s_lse[32];  // Max 32 chunks
    __shared__ float s_max_lse;
    __shared__ float s_sum_exp;

    if (dim_id < params.num_chunks) {
        s_lse[dim_id] = chunk_lse[batch_id * params.num_chunks * params.num_heads + dim_id * params.num_heads + head_id];
    }
    if (threadIdx.x == 0) {
        s_max_lse = -INFINITY;
        s_sum_exp = 0.0f;
    }
    __syncthreads();

    // Find max LSE
    for (unsigned int c = threadIdx.x; c < params.num_chunks; c += blockDim.x) {
        atomicMax(reinterpret_cast<int*>(&s_max_lse), __float_as_int(s_lse[c]));
    }
    __syncthreads();

    // Compute sum of exp(lse - max_lse)
    float local_sum = 0.0f;
    for (unsigned int c = threadIdx.x; c < params.num_chunks; c += blockDim.x) {
        local_sum += expf(s_lse[c] - s_max_lse);
    }
    atomicAdd(&s_sum_exp, local_sum);
    __syncthreads();

    // Compute weighted sum of chunk outputs
    if (dim_id < params.head_dim) {
        float result = 0.0f;
        for (unsigned int c = 0; c < params.num_chunks; c++) {
            float weight = expf(s_lse[c] - s_max_lse) / s_sum_exp;
            unsigned int chunk_idx = batch_id * params.num_chunks * params.num_heads * params.head_dim
                                   + c * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
            result += weight * chunk_outputs[chunk_idx];
        }

        unsigned int out_idx = batch_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
        output[out_idx] = result;
    }
}

// Merge chunks (FP16)
extern "C" __global__ void merge_chunks_f16(
    const __half* __restrict__ chunk_outputs,
    const float* __restrict__ chunk_lse,
    __half* __restrict__ output,
    MergeParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int dim_id = threadIdx.x;

    if (batch_id >= params.batch_size || head_id >= params.num_heads || dim_id >= params.head_dim) return;

    __shared__ float s_lse[32];
    __shared__ float s_max_lse;
    __shared__ float s_sum_exp;

    if (dim_id < params.num_chunks) {
        s_lse[dim_id] = chunk_lse[batch_id * params.num_chunks * params.num_heads + dim_id * params.num_heads + head_id];
    }
    if (threadIdx.x == 0) {
        s_max_lse = -INFINITY;
        s_sum_exp = 0.0f;
    }
    __syncthreads();

    for (unsigned int c = threadIdx.x; c < params.num_chunks; c += blockDim.x) {
        atomicMax(reinterpret_cast<int*>(&s_max_lse), __float_as_int(s_lse[c]));
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned int c = threadIdx.x; c < params.num_chunks; c += blockDim.x) {
        local_sum += expf(s_lse[c] - s_max_lse);
    }
    atomicAdd(&s_sum_exp, local_sum);
    __syncthreads();

    if (dim_id < params.head_dim) {
        float result = 0.0f;
        for (unsigned int c = 0; c < params.num_chunks; c++) {
            float weight = expf(s_lse[c] - s_max_lse) / s_sum_exp;
            unsigned int chunk_idx = batch_id * params.num_chunks * params.num_heads * params.head_dim
                                   + c * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
            result += weight * __half2float(chunk_outputs[chunk_idx]);
        }

        unsigned int out_idx = batch_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
        output[out_idx] = __float2half(result);
    }
}

// POD-Attention workload splitting
// Separate prefill and decode requests for optimal scheduling
extern "C" __global__ void pod_attention_split(
    const unsigned int* __restrict__ request_types,    // [batch_size] 0=prefill, 1=decode
    const unsigned int* __restrict__ request_lengths,  // [batch_size]
    unsigned int* __restrict__ prefill_indices,        // [max_prefill]
    unsigned int* __restrict__ decode_indices,         // [max_decode]
    unsigned int* __restrict__ num_prefill,            // [1]
    unsigned int* __restrict__ num_decode,             // [1]
    PODSplitParams params
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int s_prefill_count;
    __shared__ unsigned int s_decode_count;

    if (threadIdx.x == 0) {
        s_prefill_count = 0;
        s_decode_count = 0;
    }
    __syncthreads();

    if (idx < params.batch_size) {
        unsigned int req_type = request_types[idx];

        if (req_type == 0) {  // Prefill
            unsigned int slot = atomicAdd(&s_prefill_count, 1);
            if (slot < params.num_prefill) {
                prefill_indices[slot] = idx;
            }
        } else {  // Decode
            unsigned int slot = atomicAdd(&s_decode_count, 1);
            if (slot < params.num_decode) {
                decode_indices[slot] = idx;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *num_prefill = min(s_prefill_count, params.num_prefill);
        *num_decode = min(s_decode_count, params.num_decode);
    }
}

// Schedule batches for mixed prefill/decode workloads
extern "C" __global__ void schedule_batches(
    const unsigned int* __restrict__ request_lengths,  // [num_requests]
    const unsigned int* __restrict__ request_types,    // [num_requests]
    unsigned int* __restrict__ batch_assignments,      // [num_requests]
    unsigned int* __restrict__ batch_sizes,            // [max_batches]
    ScheduleParams params
) {
    unsigned int req_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (req_id >= params.num_requests) return;

    unsigned int len = request_lengths[req_id];
    unsigned int req_type = request_types[req_id];

    // Simple greedy scheduling: assign to first batch with capacity
    // Prefill requests consume more budget than decode
    unsigned int cost = (req_type == 0) ? len : 1;  // Prefill cost = length, decode cost = 1

    // For simplicity, assign based on request ID modulo batch size
    // Real implementation would use more sophisticated scheduling
    unsigned int batch_id = req_id / params.max_batch_size;
    batch_assignments[req_id] = batch_id;

    atomicAdd(&batch_sizes[batch_id], 1);
}
