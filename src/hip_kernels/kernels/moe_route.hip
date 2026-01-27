#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>
#include <stdint.h>

// MoE Routing kernel: computes routing logits, selects top-k experts, applies softmax
// Input: hidden_states [num_tokens, hidden_size], gate_weights [hidden_size, num_experts]
// Output: expert_indices [num_tokens, top_k], expert_weights [num_tokens, top_k]

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 64;
constexpr int MAX_EXPERTS = 64;
constexpr int MAX_TOPK = 8;

// Device function: softmax over top-k values (in-place)
__device__ void softmax_topk(float* values, int k) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < k; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }

    if (sum > 0.0f) {
        for (int i = 0; i < k; i++) {
            values[i] /= sum;
        }
    }
}

// Device function: insert into sorted top-k list (descending order)
__device__ void insert_topk_sorted(
    float* topk_vals,
    int* topk_indices,
    int k,
    float new_val,
    int new_idx
) {
    if (new_val <= topk_vals[k-1]) return;

    int pos = k - 1;
    while (pos > 0 && new_val > topk_vals[pos-1]) {
        topk_vals[pos] = topk_vals[pos-1];
        topk_indices[pos] = topk_indices[pos-1];
        pos--;
    }
    topk_vals[pos] = new_val;
    topk_indices[pos] = new_idx;
}

// Fused MoE routing kernel for f32
// Each block handles one token
extern "C" __global__ void moe_route_f32(
    const float* __restrict__ hidden_states,  // [num_tokens, hidden_size]
    const float* __restrict__ gate_weights,   // [hidden_size, num_experts]
    uint32_t* __restrict__ expert_indices,    // [num_tokens, top_k]
    float* __restrict__ expert_weights,       // [num_tokens, top_k]
    int num_tokens,
    int hidden_size,
    int num_experts,
    int top_k
) {
    __shared__ float s_logits[MAX_EXPERTS];
    __shared__ float s_topk_vals[MAX_TOPK];
    __shared__ int s_topk_indices[MAX_TOPK];

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const float* token_hidden = hidden_states + token_idx * hidden_size;

    // Step 1: Compute routing logits
    // Initialize logits to 0
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        s_logits[e] = 0.0f;
    }
    __syncthreads();

    // Accumulate: each thread handles a subset of hidden dimensions
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float h_val = token_hidden[h];
        for (int e = 0; e < num_experts; e++) {
            atomicAdd(&s_logits[e], h_val * gate_weights[h * num_experts + e]);
        }
    }
    __syncthreads();

    // Step 2: Find top-k experts (single thread for simplicity with small expert count)
    if (threadIdx.x == 0) {
        for (int i = 0; i < top_k; i++) {
            s_topk_vals[i] = -FLT_MAX;
            s_topk_indices[i] = 0;
        }

        for (int e = 0; e < num_experts; e++) {
            insert_topk_sorted(s_topk_vals, s_topk_indices, top_k, s_logits[e], e);
        }

        // Step 3: Apply softmax to top-k values
        softmax_topk(s_topk_vals, top_k);

        // Write output
        uint32_t* out_indices = expert_indices + token_idx * top_k;
        float* out_weights = expert_weights + token_idx * top_k;
        for (int i = 0; i < top_k; i++) {
            out_indices[i] = (uint32_t)s_topk_indices[i];
            out_weights[i] = s_topk_vals[i];
        }
    }
}

// Fused MoE routing kernel for f16 (hidden states in f16, gate weights in f32)
extern "C" __global__ void moe_route_f16(
    const __half* __restrict__ hidden_states,   // [num_tokens, hidden_size]
    const float* __restrict__ gate_weights,     // [hidden_size, num_experts]
    uint32_t* __restrict__ expert_indices,      // [num_tokens, top_k]
    float* __restrict__ expert_weights,         // [num_tokens, top_k]
    int num_tokens,
    int hidden_size,
    int num_experts,
    int top_k
) {
    __shared__ float s_logits[MAX_EXPERTS];
    __shared__ float s_topk_vals[MAX_TOPK];
    __shared__ int s_topk_indices[MAX_TOPK];

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const __half* token_hidden = hidden_states + token_idx * hidden_size;

    // Initialize logits
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        s_logits[e] = 0.0f;
    }
    __syncthreads();

    // Accumulate
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float h_val = __half2float(token_hidden[h]);
        for (int e = 0; e < num_experts; e++) {
            atomicAdd(&s_logits[e], h_val * gate_weights[h * num_experts + e]);
        }
    }
    __syncthreads();

    // Find top-k and softmax (single thread)
    if (threadIdx.x == 0) {
        for (int i = 0; i < top_k; i++) {
            s_topk_vals[i] = -FLT_MAX;
            s_topk_indices[i] = 0;
        }

        for (int e = 0; e < num_experts; e++) {
            insert_topk_sorted(s_topk_vals, s_topk_indices, top_k, s_logits[e], e);
        }

        softmax_topk(s_topk_vals, top_k);

        uint32_t* out_indices = expert_indices + token_idx * top_k;
        float* out_weights = expert_weights + token_idx * top_k;
        for (int i = 0; i < top_k; i++) {
            out_indices[i] = (uint32_t)s_topk_indices[i];
            out_weights[i] = s_topk_vals[i];
        }
    }
}

// Compute routing logits only (for analysis/custom routing)
extern "C" __global__ void compute_routing_logits_f32(
    const float* __restrict__ hidden_states,  // [num_tokens, hidden_size]
    const float* __restrict__ gate_weights,   // [hidden_size, num_experts]
    float* __restrict__ logits,               // [num_tokens, num_experts]
    int num_tokens,
    int hidden_size,
    int num_experts
) {
    __shared__ float s_logits[MAX_EXPERTS];

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const float* token_hidden = hidden_states + token_idx * hidden_size;

    // Initialize
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        s_logits[e] = 0.0f;
    }
    __syncthreads();

    // Accumulate
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float h_val = token_hidden[h];
        for (int e = 0; e < num_experts; e++) {
            atomicAdd(&s_logits[e], h_val * gate_weights[h * num_experts + e]);
        }
    }
    __syncthreads();

    // Write output
    float* out_logits = logits + token_idx * num_experts;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        out_logits[e] = s_logits[e];
    }
}

extern "C" __global__ void compute_routing_logits_f16(
    const __half* __restrict__ hidden_states,
    const float* __restrict__ gate_weights,
    float* __restrict__ logits,
    int num_tokens,
    int hidden_size,
    int num_experts
) {
    __shared__ float s_logits[MAX_EXPERTS];

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const __half* token_hidden = hidden_states + token_idx * hidden_size;

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        s_logits[e] = 0.0f;
    }
    __syncthreads();

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float h_val = __half2float(token_hidden[h]);
        for (int e = 0; e < num_experts; e++) {
            atomicAdd(&s_logits[e], h_val * gate_weights[h * num_experts + e]);
        }
    }
    __syncthreads();

    float* out_logits = logits + token_idx * num_experts;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        out_logits[e] = s_logits[e];
    }
}
