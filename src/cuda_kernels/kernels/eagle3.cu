// EAGLE-3 Adaptive Draft Length Speculative Decoding CUDA Kernel
// Based on EAGLE-3 (NeurIPS'25): Multi-layer feature fusion with token-level confidence prediction
//
// Key operations:
// - Fused hidden state aggregation from multiple layers
// - Token-level acceptance probability prediction
// - Adaptive draft length scheduling
//
// Ported from eagle3.hip

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>  // for FLT_MAX
#include <cmath>

#define WARP_SIZE 32
#define MAX_FUSION_LAYERS 8
#define MAX_HIDDEN_DIM 1024

// Helper: warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// =============================================================================
// Multi-layer Feature Fusion Kernel
// Fuses hidden states from multiple transformer layers for confidence prediction
// =============================================================================
extern "C" __global__ void eagle3_fuse_layers_f32(
    const float* __restrict__ layer_hidden,    // [num_layers, batch, seq_len, hidden_dim]
    float* __restrict__ fused_hidden,          // [batch, seq_len, hidden_dim * fusion_layers]
    const int* __restrict__ layer_indices,     // [fusion_layers] - which layers to fuse
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int num_layers,
    const int fusion_layers
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * hidden_dim * fusion_layers;
    if (idx >= total) return;

    int fused_dim = hidden_dim * fusion_layers;
    int b = idx / (seq_len * fused_dim);
    int rem = idx % (seq_len * fused_dim);
    int s = rem / fused_dim;
    int fd = rem % fused_dim;

    int layer_idx = fd / hidden_dim;
    int d = fd % hidden_dim;

    if (layer_idx < fusion_layers) {
        int src_layer = layer_indices[layer_idx];
        if (src_layer < num_layers) {
            int src_offset = src_layer * (batch_size * seq_len * hidden_dim) +
                             b * (seq_len * hidden_dim) +
                             s * hidden_dim + d;
            fused_hidden[idx] = layer_hidden[src_offset];
        } else {
            fused_hidden[idx] = 0.0f;
        }
    }
}

extern "C" __global__ void eagle3_fuse_layers_f16(
    const __half* __restrict__ layer_hidden,
    __half* __restrict__ fused_hidden,
    const int* __restrict__ layer_indices,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int num_layers,
    const int fusion_layers
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * hidden_dim * fusion_layers;
    if (idx >= total) return;

    int fused_dim = hidden_dim * fusion_layers;
    int b = idx / (seq_len * fused_dim);
    int rem = idx % (seq_len * fused_dim);
    int s = rem / fused_dim;
    int fd = rem % fused_dim;

    int layer_idx = fd / hidden_dim;
    int d = fd % hidden_dim;

    if (layer_idx < fusion_layers) {
        int src_layer = layer_indices[layer_idx];
        if (src_layer < num_layers) {
            int src_offset = src_layer * (batch_size * seq_len * hidden_dim) +
                             b * (seq_len * hidden_dim) +
                             s * hidden_dim + d;
            fused_hidden[idx] = layer_hidden[src_offset];
        } else {
            fused_hidden[idx] = __float2half(0.0f);
        }
    }
}

// =============================================================================
// Token-level Confidence Prediction Kernel
// Computes acceptance probability for each token position
// =============================================================================
extern "C" __global__ void eagle3_predict_confidence_f32(
    const float* __restrict__ fused_hidden,  // [batch, seq_len, fused_dim]
    const float* __restrict__ weight,        // [fused_dim]
    float* __restrict__ confidence,          // [batch, seq_len]
    const float bias,
    const int batch_size,
    const int seq_len,
    const int fused_dim
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (b >= batch_size || s >= seq_len) return;

    // Each thread computes partial dot product
    float partial_sum = 0.0f;
    int base = (b * seq_len + s) * fused_dim;

    for (int d = tid; d < fused_dim; d += block_size) {
        partial_sum += fused_hidden[base + d] * weight[d];
    }

    // Reduce within block
    smem[tid] = partial_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes result with sigmoid activation
    if (tid == 0) {
        float logit = smem[0] + bias;
        float prob = 1.0f / (1.0f + expf(-logit));
        confidence[b * seq_len + s] = prob;
    }
}

extern "C" __global__ void eagle3_predict_confidence_f16(
    const __half* __restrict__ fused_hidden,
    const __half* __restrict__ weight,
    __half* __restrict__ confidence,
    const float bias,
    const int batch_size,
    const int seq_len,
    const int fused_dim
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (b >= batch_size || s >= seq_len) return;

    float partial_sum = 0.0f;
    int base = (b * seq_len + s) * fused_dim;

    for (int d = tid; d < fused_dim; d += block_size) {
        partial_sum += __half2float(fused_hidden[base + d]) * __half2float(weight[d]);
    }

    smem[tid] = partial_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = smem[0] + bias;
        float prob = 1.0f / (1.0f + expf(-logit));
        confidence[b * seq_len + s] = __float2half(prob);
    }
}

// =============================================================================
// Adaptive Draft Length Selection Kernel
// Determines optimal draft length based on confidence threshold
// =============================================================================
extern "C" __global__ void eagle3_select_draft_length_f32(
    const float* __restrict__ confidence,     // [batch, seq_len]
    int* __restrict__ draft_lengths,          // [batch]
    const float threshold,
    const int batch_size,
    const int seq_len,
    const int min_length,
    const int max_length
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int draft_len = max_length;

    // Find first position where confidence drops below threshold
    for (int s = 0; s < seq_len && s < max_length; ++s) {
        if (confidence[b * seq_len + s] < threshold) {
            draft_len = s > 0 ? s : min_length;
            break;
        }
    }

    // Clamp to valid range
    draft_len = max(min_length, min(draft_len, max_length));
    draft_lengths[b] = draft_len;
}

// =============================================================================
// Verification and Acceptance Kernel
// Verifies draft tokens against target model and computes acceptance
// =============================================================================
extern "C" __global__ void eagle3_verify_draft_f32(
    const float* __restrict__ draft_logits,   // [batch, draft_len, vocab_size]
    const float* __restrict__ target_logits,  // [batch, draft_len, vocab_size]
    const int* __restrict__ draft_tokens,     // [batch, draft_len]
    int* __restrict__ accepted_lengths,       // [batch]
    float* __restrict__ acceptance_probs,     // [batch, draft_len]
    const int batch_size,
    const int draft_len,
    const int vocab_size
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int pos = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (b >= batch_size || pos >= draft_len) return;

    int token = draft_tokens[b * draft_len + pos];
    int base_draft = (b * draft_len + pos) * vocab_size;
    int base_target = (b * draft_len + pos) * vocab_size;

    // Compute softmax denominators in parallel
    float draft_max = -FLT_MAX;
    float target_max = -FLT_MAX;

    for (int v = tid; v < vocab_size; v += block_size) {
        draft_max = fmaxf(draft_max, draft_logits[base_draft + v]);
        target_max = fmaxf(target_max, target_logits[base_target + v]);
    }

    // Reduce to find global max
    smem[tid] = draft_max;
    smem[tid + block_size] = target_max;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
            smem[tid + block_size] = fmaxf(smem[tid + block_size], smem[tid + block_size + stride]);
        }
        __syncthreads();
    }

    draft_max = smem[0];
    target_max = smem[block_size];

    // Compute exp sums and token probabilities
    float draft_sum = 0.0f;
    float target_sum = 0.0f;
    float draft_prob_token = 0.0f;
    float target_prob_token = 0.0f;

    for (int v = tid; v < vocab_size; v += block_size) {
        float d_exp = expf(draft_logits[base_draft + v] - draft_max);
        float t_exp = expf(target_logits[base_target + v] - target_max);
        draft_sum += d_exp;
        target_sum += t_exp;
        if (v == token) {
            draft_prob_token = d_exp;
            target_prob_token = t_exp;
        }
    }

    // Reduce sums and token probs
    smem[tid] = draft_sum;
    smem[tid + block_size] = target_sum;
    smem[tid + 2 * block_size] = draft_prob_token;
    smem[tid + 3 * block_size] = target_prob_token;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
            smem[tid + block_size] += smem[tid + block_size + stride];
            smem[tid + 2 * block_size] += smem[tid + 2 * block_size + stride];
            smem[tid + 3 * block_size] += smem[tid + 3 * block_size + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float p_draft = smem[2 * block_size] / (smem[0] + 1e-10f);
        float p_target = smem[3 * block_size] / (smem[block_size] + 1e-10f);

        // Acceptance probability: min(1, p_target / p_draft)
        float accept_prob = (p_draft > 0.0f) ? fminf(1.0f, p_target / p_draft) : 0.0f;
        acceptance_probs[b * draft_len + pos] = accept_prob;
    }
}
