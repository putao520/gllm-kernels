#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void embedding_gather_f32(
    const uint32_t* __restrict__ token_ids,
    const float* __restrict__ table,
    float* __restrict__ output,
    int num_tokens,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_dim;
    if (idx >= total) {
        return;
    }

    int token_idx = idx / hidden_dim;
    int h = idx - token_idx * hidden_dim;
    uint32_t token = token_ids[token_idx];
    output[idx] = table[(int)token * hidden_dim + h];
}

extern "C" __global__ void mean_pooling_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ mask,
    float* __restrict__ output,
    int batch,
    int seq,
    int hidden_dim,
    int use_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_dim;
    if (idx >= total) {
        return;
    }

    int b = idx / hidden_dim;
    int h = idx - b * hidden_dim;
    float sum = 0.0f;
    float count = 0.0f;

    int base = b * seq * hidden_dim;
    int mask_base = b * seq;
    for (int s = 0; s < seq; ++s) {
        float mask_val = use_mask ? mask[mask_base + s] : 1.0f;
        if (mask_val > 0.0f) {
            sum += hidden[base + s * hidden_dim + h] * mask_val;
            count += mask_val;
        }
    }

    output[idx] = (count > 0.0f) ? (sum / count) : 0.0f;
}

extern "C" __global__ void l2_normalize_f32(
    float* __restrict__ data,
    int batch,
    int hidden_dim,
    float eps
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }

    int base = b * hidden_dim;
    float sum = 0.0f;
    for (int d = 0; d < hidden_dim; ++d) {
        float v = data[base + d];
        sum += v * v;
    }
    float norm = sqrtf(sum + eps);
    float inv = norm > 0.0f ? (1.0f / norm) : 0.0f;

    for (int d = 0; d < hidden_dim; ++d) {
        data[base + d] *= inv;
    }
}

extern "C" __global__ void cls_extract_f32(
    const float* __restrict__ hidden,
    float* __restrict__ output,
    int batch,
    int seq,
    int hidden_dim,
    int cls_pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_dim;
    if (idx >= total) {
        return;
    }

    int b = idx / hidden_dim;
    int h = idx - b * hidden_dim;
    int src = (b * seq + cls_pos) * hidden_dim + h;
    output[idx] = hidden[src];
}
