// Naive tiled attention kernel for testing correctness.
// This is intentionally simple and not optimized for performance.

#include <cuda_fp16.h>
#include <math.h>

#define MAX_HEAD_DIM 256

extern "C" __global__ void tiled_attention_forward_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int is_causal,
    int position_offset
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = batch_size * num_heads * seq_len;
    if (idx >= total) {
        return;
    }

    int head_stride = seq_len * head_dim;
    int batch_stride = num_heads * head_stride;

    int b = idx / (num_heads * seq_len);
    int rem = idx - b * num_heads * seq_len;
    int h = rem / seq_len;
    int q_pos = rem - h * seq_len;

    int base = b * batch_stride + h * head_stride;
    const float* q_ptr = Q + base + q_pos * head_dim;

    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
        if (is_causal && j > q_pos + position_offset) {
            continue;
        }
        const float* k_ptr = K + base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        float score = dot * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float out[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        out[d] = 0.0f;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        if (is_causal && j > q_pos + position_offset) {
            continue;
        }
        const float* k_ptr = K + base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        float score = dot * scale;
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;

        const float* v_ptr = V + base + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out[d] += exp_score * v_ptr[d];
        }
    }

    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    float* o_ptr = O + base + q_pos * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = out[d] * inv_sum;
    }
}

extern "C" __global__ void tiled_attention_forward_f16(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int is_causal,
    int position_offset
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = batch_size * num_heads * seq_len;
    if (idx >= total) {
        return;
    }

    int head_stride = seq_len * head_dim;
    int batch_stride = num_heads * head_stride;

    int b = idx / (num_heads * seq_len);
    int rem = idx - b * num_heads * seq_len;
    int h = rem / seq_len;
    int q_pos = rem - h * seq_len;

    int base = b * batch_stride + h * head_stride;
    const half* q_ptr = Q + base + q_pos * head_dim;

    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
        if (is_causal && j > q_pos + position_offset) {
            continue;
        }
        const half* k_ptr = K + base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        float score = dot * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float out[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        out[d] = 0.0f;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        if (is_causal && j > q_pos + position_offset) {
            continue;
        }
        const half* k_ptr = K + base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        float score = dot * scale;
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;

        const half* v_ptr = V + base + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out[d] += exp_score * __half2float(v_ptr[d]);
        }
    }

    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    half* o_ptr = O + base + q_pos * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = __float2half_rn(out[d] * inv_sum);
    }
}
