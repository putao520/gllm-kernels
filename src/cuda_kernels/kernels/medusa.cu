// Medusa Heads auxiliary generation CUDA kernels.
//
// This file contains CUDA kernels for Medusa-style speculative decoding:
// - Medusa head forward pass (auxiliary LM heads)
// - Top-k sampling from Medusa outputs
// - Candidate tree construction
// - Parallel candidate verification
//
// Ported from medusa.hip

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>  // for FLT_MAX
#include <cmath>

// Parameters for head forward pass
struct HeadForwardParams {
    unsigned int batch_size;
    unsigned int hidden_dim;
    unsigned int vocab_size;
    unsigned int num_heads;  // Number of Medusa heads
};

// Parameters for top-k sampling
struct TopKParams {
    unsigned int batch_size;
    unsigned int vocab_size;
    unsigned int num_heads;
    unsigned int top_k;
};

// Parameters for candidate building
struct CandidateParams {
    unsigned int batch_size;
    unsigned int num_heads;
    unsigned int top_k;
    unsigned int max_candidates;
};

// Parameters for verification
struct VerifyParams {
    unsigned int batch_size;
    unsigned int num_candidates;
    unsigned int vocab_size;
    unsigned int _pad;
};

// Medusa head forward pass (FP32)
// hidden_states: [batch, hidden_dim]
// weights: [num_heads, hidden_dim, vocab_size]
// output: [batch, num_heads, vocab_size]
extern "C" __global__ void medusa_head_forward_f32(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weights,
    float* __restrict__ output,
    HeadForwardParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int vocab_id = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch_id >= params.batch_size || head_id >= params.num_heads || vocab_id >= params.vocab_size) return;

    // Compute logit: sum(hidden[batch, :] * weight[head, :, vocab])
    float sum = 0.0f;
    for (unsigned int d = 0; d < params.hidden_dim; d++) {
        float h = hidden_states[batch_id * params.hidden_dim + d];
        float w = weights[head_id * params.hidden_dim * params.vocab_size + d * params.vocab_size + vocab_id];
        sum += h * w;
    }

    // Store logit
    unsigned int out_idx = batch_id * params.num_heads * params.vocab_size + head_id * params.vocab_size + vocab_id;
    output[out_idx] = sum;
}

// Medusa head forward pass (FP16)
extern "C" __global__ void medusa_head_forward_f16(
    const __half* __restrict__ hidden_states,
    const __half* __restrict__ weights,
    __half* __restrict__ output,
    HeadForwardParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int vocab_id = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch_id >= params.batch_size || head_id >= params.num_heads || vocab_id >= params.vocab_size) return;

    float sum = 0.0f;
    for (unsigned int d = 0; d < params.hidden_dim; d++) {
        float h = __half2float(hidden_states[batch_id * params.hidden_dim + d]);
        float w = __half2float(weights[head_id * params.hidden_dim * params.vocab_size + d * params.vocab_size + vocab_id]);
        sum += h * w;
    }

    unsigned int out_idx = batch_id * params.num_heads * params.vocab_size + head_id * params.vocab_size + vocab_id;
    output[out_idx] = __float2half(sum);
}

// Top-k sampling from logits (FP32)
// logits: [batch, num_heads, vocab_size]
// top_k_indices: [batch, num_heads, top_k]
// top_k_probs: [batch, num_heads, top_k]
extern "C" __global__ void medusa_top_k_sample_f32(
    const float* __restrict__ logits,
    unsigned int* __restrict__ top_k_indices,
    float* __restrict__ top_k_probs,
    TopKParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;

    if (batch_id >= params.batch_size || head_id >= params.num_heads) return;

    // Thread-local top-k using insertion sort (small k, simple implementation)
    extern __shared__ char smem[];
    float* s_vals = (float*)smem;
    unsigned int* s_indices = (unsigned int*)(smem + params.top_k * sizeof(float));

    unsigned int lane_id = threadIdx.x;

    // Initialize with -inf
    if (lane_id < params.top_k) {
        s_vals[lane_id] = -INFINITY;
        s_indices[lane_id] = 0;
    }
    __syncthreads();

    unsigned int base_idx = batch_id * params.num_heads * params.vocab_size + head_id * params.vocab_size;

    // Find top-k using parallel reduction + serial update
    for (unsigned int v = lane_id; v < params.vocab_size; v += blockDim.x) {
        float val = logits[base_idx + v];

        // Check if this value should be in top-k
        if (val > s_vals[params.top_k - 1]) {
            // Find insertion position
            for (unsigned int k = 0; k < params.top_k; k++) {
                if (val > s_vals[k]) {
                    // Shift down and insert
                    for (unsigned int j = params.top_k - 1; j > k; j--) {
                        s_vals[j] = s_vals[j - 1];
                        s_indices[j] = s_indices[j - 1];
                    }
                    s_vals[k] = val;
                    s_indices[k] = v;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // Compute softmax over top-k
    if (lane_id < params.top_k) {
        float max_val = s_vals[0];
        float sum = 0.0f;

        for (unsigned int k = 0; k < params.top_k; k++) {
            sum += expf(s_vals[k] - max_val);
        }

        float prob = expf(s_vals[lane_id] - max_val) / sum;

        unsigned int out_base = batch_id * params.num_heads * params.top_k + head_id * params.top_k;
        top_k_indices[out_base + lane_id] = s_indices[lane_id];
        top_k_probs[out_base + lane_id] = prob;
    }
}

// Top-k sampling from logits (FP16)
extern "C" __global__ void medusa_top_k_sample_f16(
    const __half* __restrict__ logits,
    unsigned int* __restrict__ top_k_indices,
    __half* __restrict__ top_k_probs,
    TopKParams params
) {
    unsigned int batch_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;

    if (batch_id >= params.batch_size || head_id >= params.num_heads) return;

    extern __shared__ char smem[];
    float* s_vals = (float*)smem;
    unsigned int* s_indices = (unsigned int*)(smem + params.top_k * sizeof(float));

    unsigned int lane_id = threadIdx.x;

    if (lane_id < params.top_k) {
        s_vals[lane_id] = -INFINITY;
        s_indices[lane_id] = 0;
    }
    __syncthreads();

    unsigned int base_idx = batch_id * params.num_heads * params.vocab_size + head_id * params.vocab_size;

    for (unsigned int v = lane_id; v < params.vocab_size; v += blockDim.x) {
        float val = __half2float(logits[base_idx + v]);

        if (val > s_vals[params.top_k - 1]) {
            for (unsigned int k = 0; k < params.top_k; k++) {
                if (val > s_vals[k]) {
                    for (unsigned int j = params.top_k - 1; j > k; j--) {
                        s_vals[j] = s_vals[j - 1];
                        s_indices[j] = s_indices[j - 1];
                    }
                    s_vals[k] = val;
                    s_indices[k] = v;
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (lane_id < params.top_k) {
        float max_val = s_vals[0];
        float sum = 0.0f;

        for (unsigned int k = 0; k < params.top_k; k++) {
            sum += expf(s_vals[k] - max_val);
        }

        float prob = expf(s_vals[lane_id] - max_val) / sum;

        unsigned int out_base = batch_id * params.num_heads * params.top_k + head_id * params.top_k;
        top_k_indices[out_base + lane_id] = s_indices[lane_id];
        top_k_probs[out_base + lane_id] = __float2half(prob);
    }
}

// Build candidate token sequences from top-k selections (FP32)
// Cartesian product across heads, pruned by probability threshold
extern "C" __global__ void medusa_build_candidates_f32(
    const unsigned int* __restrict__ top_k_indices,  // [batch, num_heads, top_k]
    const float* __restrict__ top_k_probs,           // [batch, num_heads, top_k]
    unsigned int* __restrict__ candidates,           // [batch, max_candidates, num_heads]
    float* __restrict__ candidate_probs,             // [batch, max_candidates]
    unsigned int* __restrict__ num_candidates,       // [batch]
    CandidateParams params
) {
    unsigned int batch_id = blockIdx.x;
    if (batch_id >= params.batch_size) return;

    unsigned int lane_id = threadIdx.x;

    // Simple enumeration for small top_k (e.g., k=4, 3 heads = 64 candidates max)
    // Each thread handles some candidates
    unsigned int total_possible = 1;
    for (unsigned int h = 0; h < params.num_heads; h++) {
        total_possible *= params.top_k;
    }

    __shared__ unsigned int s_count;
    if (lane_id == 0) s_count = 0;
    __syncthreads();

    for (unsigned int c = lane_id; c < total_possible && c < params.max_candidates; c += blockDim.x) {
        // Decode candidate index to per-head selections
        unsigned int indices[8];  // Max 8 heads
        float prob = 1.0f;
        unsigned int temp = c;

        for (unsigned int h = 0; h < params.num_heads; h++) {
            unsigned int k_idx = temp % params.top_k;
            temp /= params.top_k;

            unsigned int idx_base = batch_id * params.num_heads * params.top_k + h * params.top_k;
            indices[h] = top_k_indices[idx_base + k_idx];
            prob *= top_k_probs[idx_base + k_idx];
        }

        // Store candidate if probability is reasonable
        if (prob > 0.001f) {  // Threshold
            unsigned int slot = atomicAdd(&s_count, 1);
            if (slot < params.max_candidates) {
                candidate_probs[batch_id * params.max_candidates + slot] = prob;
                for (unsigned int h = 0; h < params.num_heads; h++) {
                    candidates[batch_id * params.max_candidates * params.num_heads + slot * params.num_heads + h] = indices[h];
                }
            }
        }
    }
    __syncthreads();

    if (lane_id == 0) {
        num_candidates[batch_id] = min(s_count, params.max_candidates);
    }
}

// Build candidate token sequences (FP16)
extern "C" __global__ void medusa_build_candidates_f16(
    const unsigned int* __restrict__ top_k_indices,
    const __half* __restrict__ top_k_probs,
    unsigned int* __restrict__ candidates,
    __half* __restrict__ candidate_probs,
    unsigned int* __restrict__ num_candidates,
    CandidateParams params
) {
    unsigned int batch_id = blockIdx.x;
    if (batch_id >= params.batch_size) return;

    unsigned int lane_id = threadIdx.x;

    unsigned int total_possible = 1;
    for (unsigned int h = 0; h < params.num_heads; h++) {
        total_possible *= params.top_k;
    }

    __shared__ unsigned int s_count;
    if (lane_id == 0) s_count = 0;
    __syncthreads();

    for (unsigned int c = lane_id; c < total_possible && c < params.max_candidates; c += blockDim.x) {
        unsigned int indices[8];
        float prob = 1.0f;
        unsigned int temp = c;

        for (unsigned int h = 0; h < params.num_heads; h++) {
            unsigned int k_idx = temp % params.top_k;
            temp /= params.top_k;

            unsigned int idx_base = batch_id * params.num_heads * params.top_k + h * params.top_k;
            indices[h] = top_k_indices[idx_base + k_idx];
            prob *= __half2float(top_k_probs[idx_base + k_idx]);
        }

        if (prob > 0.001f) {
            unsigned int slot = atomicAdd(&s_count, 1);
            if (slot < params.max_candidates) {
                candidate_probs[batch_id * params.max_candidates + slot] = __float2half(prob);
                for (unsigned int h = 0; h < params.num_heads; h++) {
                    candidates[batch_id * params.max_candidates * params.num_heads + slot * params.num_heads + h] = indices[h];
                }
            }
        }
    }
    __syncthreads();

    if (lane_id == 0) {
        num_candidates[batch_id] = min(s_count, params.max_candidates);
    }
}

// Verify candidates against target model logits (FP32)
extern "C" __global__ void medusa_verify_candidates_f32(
    const unsigned int* __restrict__ candidates,     // [batch, num_candidates, depth]
    const float* __restrict__ target_logits,         // [batch, num_candidates, vocab_size]
    unsigned int* __restrict__ accepted_lengths,     // [batch]
    unsigned int* __restrict__ accepted_tokens,      // [batch, max_depth]
    VerifyParams params
) {
    unsigned int batch_id = blockIdx.x;
    if (batch_id >= params.batch_size) return;

    unsigned int lane_id = threadIdx.x;

    // Each candidate is verified by checking if argmax(logits) matches the candidate token
    __shared__ unsigned int s_best_length;
    __shared__ unsigned int s_best_candidate;

    if (lane_id == 0) {
        s_best_length = 0;
        s_best_candidate = 0;
    }
    __syncthreads();

    for (unsigned int c = lane_id; c < params.num_candidates; c += blockDim.x) {
        unsigned int match_length = 0;

        // Check each position in the candidate sequence
        // Simplified: check first token match
        unsigned int cand_base = batch_id * params.num_candidates * params.vocab_size + c * params.vocab_size;
        unsigned int cand_token = candidates[batch_id * params.num_candidates + c];

        float cand_logit = target_logits[cand_base + cand_token];

        // Find argmax
        float max_logit = -INFINITY;
        unsigned int max_idx = 0;
        for (unsigned int v = 0; v < params.vocab_size; v++) {
            float l = target_logits[cand_base + v];
            if (l > max_logit) {
                max_logit = l;
                max_idx = v;
            }
        }

        if (max_idx == cand_token) {
            match_length = 1;  // Simplified: full verification would check all positions
        }

        atomicMax(&s_best_length, match_length);
    }
    __syncthreads();

    if (lane_id == 0) {
        accepted_lengths[batch_id] = s_best_length;
    }
}

// Verify candidates (FP16)
extern "C" __global__ void medusa_verify_candidates_f16(
    const unsigned int* __restrict__ candidates,
    const __half* __restrict__ target_logits,
    unsigned int* __restrict__ accepted_lengths,
    unsigned int* __restrict__ accepted_tokens,
    VerifyParams params
) {
    unsigned int batch_id = blockIdx.x;
    if (batch_id >= params.batch_size) return;

    unsigned int lane_id = threadIdx.x;

    __shared__ unsigned int s_best_length;
    __shared__ unsigned int s_best_candidate;

    if (lane_id == 0) {
        s_best_length = 0;
        s_best_candidate = 0;
    }
    __syncthreads();

    for (unsigned int c = lane_id; c < params.num_candidates; c += blockDim.x) {
        unsigned int cand_base = batch_id * params.num_candidates * params.vocab_size + c * params.vocab_size;
        unsigned int cand_token = candidates[batch_id * params.num_candidates + c];

        float max_logit = -INFINITY;
        unsigned int max_idx = 0;
        for (unsigned int v = 0; v < params.vocab_size; v++) {
            float l = __half2float(target_logits[cand_base + v]);
            if (l > max_logit) {
                max_logit = l;
                max_idx = v;
            }
        }

        unsigned int match_length = (max_idx == cand_token) ? 1 : 0;
        atomicMax(&s_best_length, match_length);
    }
    __syncthreads();

    if (lane_id == 0) {
        accepted_lengths[batch_id] = s_best_length;
    }
}
