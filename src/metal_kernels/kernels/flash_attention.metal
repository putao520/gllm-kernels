#include <metal_stdlib>

using namespace metal;

// Metal requires constant address space for program-scope variables
// Use macro for array size to ensure compile-time constant
#define MAX_HEAD_DIM 256

struct AttentionParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    float scale;
    uint is_causal;
    uint position_offset;
    uint _pad;
};

kernel void tiled_attention_forward_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *o [[buffer(3)]],
    constant AttentionParams &params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.seq_len;
    if (tid >= total || params.head_dim == 0 || params.head_dim > MAX_HEAD_DIM) {
        return;
    }

    uint head_stride = params.seq_len * params.head_dim;
    uint batch_stride = params.num_heads * head_stride;

    uint b = tid / (params.num_heads * params.seq_len);
    uint rem = tid - b * params.num_heads * params.seq_len;
    uint h = rem / params.seq_len;
    uint q_pos = rem - h * params.seq_len;

    uint base = b * batch_stride + h * head_stride;
    uint q_base = base + q_pos * params.head_dim;

    float max_score = -INFINITY;
    for (uint j = 0; j < params.seq_len; ++j) {
        if (params.is_causal != 0 && j > q_pos + params.position_offset) {
            continue;
        }
        uint kv_base = base + j * params.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            dot += q[q_base + d] * k[kv_base + d];
        }
        float score = dot * params.scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float out[MAX_HEAD_DIM];
    for (uint d = 0; d < params.head_dim; ++d) {
        out[d] = 0.0f;
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j < params.seq_len; ++j) {
        if (params.is_causal != 0 && j > q_pos + params.position_offset) {
            continue;
        }
        uint kv_base = base + j * params.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            dot += q[q_base + d] * k[kv_base + d];
        }
        float score = dot * params.scale;
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;

        for (uint d = 0; d < params.head_dim; ++d) {
            out[d] += exp_score * v[kv_base + d];
        }
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    uint o_base = base + q_pos * params.head_dim;
    for (uint d = 0; d < params.head_dim; ++d) {
        o[o_base + d] = out[d] * inv_sum;
    }
}

kernel void tiled_attention_forward_f16(
    device const half *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device half *o [[buffer(3)]],
    constant AttentionParams &params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.seq_len;
    if (tid >= total || params.head_dim == 0 || params.head_dim > MAX_HEAD_DIM) {
        return;
    }

    uint head_stride = params.seq_len * params.head_dim;
    uint batch_stride = params.num_heads * head_stride;

    uint b = tid / (params.num_heads * params.seq_len);
    uint rem = tid - b * params.num_heads * params.seq_len;
    uint h = rem / params.seq_len;
    uint q_pos = rem - h * params.seq_len;

    uint base = b * batch_stride + h * head_stride;
    uint q_base = base + q_pos * params.head_dim;

    float max_score = -INFINITY;
    for (uint j = 0; j < params.seq_len; ++j) {
        if (params.is_causal != 0 && j > q_pos + params.position_offset) {
            continue;
        }
        uint kv_base = base + j * params.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            dot += float(q[q_base + d]) * float(k[kv_base + d]);
        }
        float score = dot * params.scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float out[MAX_HEAD_DIM];
    for (uint d = 0; d < params.head_dim; ++d) {
        out[d] = 0.0f;
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j < params.seq_len; ++j) {
        if (params.is_causal != 0 && j > q_pos + params.position_offset) {
            continue;
        }
        uint kv_base = base + j * params.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            dot += float(q[q_base + d]) * float(k[kv_base + d]);
        }
        float score = dot * params.scale;
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;

        for (uint d = 0; d < params.head_dim; ++d) {
            out[d] += exp_score * float(v[kv_base + d]);
        }
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    uint o_base = base + q_pos * params.head_dim;
    for (uint d = 0; d < params.head_dim; ++d) {
        o[o_base + d] = half(out[d] * inv_sum);
    }
}
