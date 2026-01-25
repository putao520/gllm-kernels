// SpecEE / LayerSkip Early-Exit Speculative Decoding CUDA Kernel
// Based on SpecEE (ISCA'25) and LayerSkip (ACL'24)
//
// Key operations:
// - Early exit confidence computation per layer
// - LM head projection for early exit predictions
// - Shared activation management between draft and verify phases
//
// Ported from spec_ee.hip

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

#define WARP_SIZE 32
#define MAX_EXIT_LAYERS 12
#define MAX_HIDDEN_DIM 1024

// =============================================================================
// Early Exit Confidence Prediction
// Computes exit confidence for each layer using lightweight classifier
// =============================================================================
extern "C" __global__ void spec_ee_compute_confidence_f32(
    const float* __restrict__ hidden_states,  // [batch, seq_len, hidden_dim]
    const float* __restrict__ confidence_weight,  // [hidden_dim]
    float* __restrict__ confidence,           // [batch, seq_len]
    const float confidence_bias,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (b >= batch_size || s >= seq_len) return;

    // Compute dot product with confidence head
    float partial_sum = 0.0f;
    int base = (b * seq_len + s) * hidden_dim;

    for (int d = tid; d < hidden_dim; d += block_size) {
        partial_sum += hidden_states[base + d] * confidence_weight[d];
    }

    smem[tid] = partial_sum;
    __syncthreads();

    // Parallel reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = smem[0] + confidence_bias;
        float prob = 1.0f / (1.0f + expf(-logit));
        confidence[b * seq_len + s] = prob;
    }
}

extern "C" __global__ void spec_ee_compute_confidence_f16(
    const __half* __restrict__ hidden_states,
    const __half* __restrict__ confidence_weight,
    __half* __restrict__ confidence,
    const float confidence_bias,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (b >= batch_size || s >= seq_len) return;

    float partial_sum = 0.0f;
    int base = (b * seq_len + s) * hidden_dim;

    for (int d = tid; d < hidden_dim; d += block_size) {
        partial_sum += __half2float(hidden_states[base + d]) * __half2float(confidence_weight[d]);
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
        float logit = smem[0] + confidence_bias;
        float prob = 1.0f / (1.0f + expf(-logit));
        confidence[b * seq_len + s] = __float2half(prob);
    }
}

// =============================================================================
// Early Exit LM Head Projection
// Projects hidden states to vocabulary logits at exit layer
// =============================================================================
extern "C" __global__ void spec_ee_lm_head_f32(
    const float* __restrict__ hidden_states,  // [batch, seq_len, hidden_dim]
    const float* __restrict__ lm_weight,      // [hidden_dim, vocab_size]
    float* __restrict__ logits,               // [batch, seq_len, vocab_size]
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * vocab_size;
    if (idx >= total) return;

    int b = idx / (seq_len * vocab_size);
    int rem = idx % (seq_len * vocab_size);
    int s = rem / vocab_size;
    int v = rem % vocab_size;

    float sum = 0.0f;
    int hidden_base = (b * seq_len + s) * hidden_dim;

    for (int d = 0; d < hidden_dim; ++d) {
        sum += hidden_states[hidden_base + d] * lm_weight[d * vocab_size + v];
    }

    logits[idx] = sum;
}

extern "C" __global__ void spec_ee_lm_head_f16(
    const __half* __restrict__ hidden_states,
    const __half* __restrict__ lm_weight,
    __half* __restrict__ logits,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * vocab_size;
    if (idx >= total) return;

    int b = idx / (seq_len * vocab_size);
    int rem = idx % (seq_len * vocab_size);
    int s = rem / vocab_size;
    int v = rem % vocab_size;

    float sum = 0.0f;
    int hidden_base = (b * seq_len + s) * hidden_dim;

    for (int d = 0; d < hidden_dim; ++d) {
        sum += __half2float(hidden_states[hidden_base + d]) * __half2float(lm_weight[d * vocab_size + v]);
    }

    logits[idx] = __float2half(sum);
}

// =============================================================================
// Exit Layer Selection
// Determines which layer to exit from based on confidence thresholds
// =============================================================================
extern "C" __global__ void spec_ee_select_exit_layer_f32(
    const float* __restrict__ layer_confidences,  // [num_exit_layers, batch, seq_len]
    int* __restrict__ exit_layers,                // [batch, seq_len]
    const float* __restrict__ thresholds,         // [num_exit_layers]
    const int* __restrict__ exit_layer_indices,   // [num_exit_layers]
    const int batch_size,
    const int seq_len,
    const int num_exit_layers,
    const int min_exit_layer,
    const int full_layer_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len;
    if (idx >= total) return;

    int b = idx / seq_len;
    int s = idx % seq_len;

    int selected_layer = full_layer_idx;  // Default to full model

    // Check exit layers in order (early to late)
    for (int i = 0; i < num_exit_layers; ++i) {
        int layer_idx = exit_layer_indices[i];
        if (layer_idx < min_exit_layer) continue;

        float conf = layer_confidences[i * batch_size * seq_len + b * seq_len + s];
        if (conf >= thresholds[i]) {
            selected_layer = layer_idx;
            break;  // Exit at first confident layer
        }
    }

    exit_layers[idx] = selected_layer;
}

// =============================================================================
// Shared Activation Copy (for draft->verify reuse)
// Efficiently copies activations between draft and verify phases
// =============================================================================
extern "C" __global__ void spec_ee_copy_activations_f32(
    const float* __restrict__ src,    // [batch, seq_len, hidden_dim]
    float* __restrict__ dst,          // [batch, seq_len, hidden_dim]
    const int* __restrict__ copy_mask,  // [batch, seq_len] - 1 to copy, 0 to skip
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * hidden_dim;
    if (idx >= total) return;

    int bs_idx = idx / hidden_dim;
    int b = bs_idx / seq_len;
    int s = bs_idx % seq_len;

    if (copy_mask[b * seq_len + s] != 0) {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// Layer Dropout Mask Generation (for training)
// Generates per-layer dropout masks with increasing rates for higher layers
// =============================================================================
extern "C" __global__ void spec_ee_generate_dropout_mask(
    float* __restrict__ random_values,  // [num_layers]
    int* __restrict__ dropout_mask,     // [num_layers]
    const float min_rate,
    const float max_rate,
    const int num_layers
) {
    int layer = blockIdx.x * blockDim.x + threadIdx.x;
    if (layer >= num_layers) return;

    // Linear dropout schedule: rate increases with layer index
    float t = (num_layers > 1) ? (float)layer / (float)(num_layers - 1) : 0.0f;
    float rate = min_rate + t * (max_rate - min_rate);

    dropout_mask[layer] = (random_values[layer] < rate) ? 0 : 1;
}
