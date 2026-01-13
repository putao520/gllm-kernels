#include <metal_stdlib>

using namespace metal;

constexpr uint MAX_HEAD_DIM = 256;

struct PagedAttentionParams {
    uint batch_size;
    uint num_heads;
    uint head_dim;
    uint block_size;
    uint seq_len;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

kernel void paged_attention_forward_f32(
    device const float *q [[buffer(0)]],
    device const float *k_cache [[buffer(1)]],
    device const float *v_cache [[buffer(2)]],
    device const int *block_tables [[buffer(3)]],
    device const int *block_offsets [[buffer(4)]],
    device float *o [[buffer(5)]],
    constant PagedAttentionParams &params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.seq_len;
    if (tid >= total || params.head_dim == 0 || params.head_dim > MAX_HEAD_DIM || params.block_size == 0) {
        return;
    }

    uint q_pos = tid % params.seq_len;
    uint rem = tid / params.seq_len;
    uint h = rem % params.num_heads;
    uint b = rem / params.num_heads;

    int position_offset = block_offsets[b];
    int kv_len_i = position_offset + int(params.seq_len);
    if (kv_len_i <= 0) {
        return;
    }
    uint kv_len = uint(kv_len_i);

    float q_local[MAX_HEAD_DIM];
    uint q_base = ((b * params.num_heads + h) * params.seq_len + q_pos) * params.head_dim;
    for (uint d = 0; d < params.head_dim; ++d) {
        q_local[d] = q[q_base + d];
    }

    float acc[MAX_HEAD_DIM];
    for (uint d = 0; d < params.head_dim; ++d) {
        acc[d] = 0.0f;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float scale = rsqrt(float(params.head_dim));

    uint table_base = b * kv_len;
    uint num_blocks = (kv_len + params.block_size - 1) / params.block_size;
    int q_abs = position_offset + int(q_pos);

    for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint token_base = block_idx * params.block_size;
        if (token_base >= kv_len) {
            break;
        }

        int block_id = block_tables[table_base + token_base];
        uint tokens_in_block = kv_len - token_base;
        if (tokens_in_block > params.block_size) {
            tokens_in_block = params.block_size;
        }

        for (uint t = 0; t < tokens_in_block; ++t) {
            uint k_idx = token_base + t;
            if (int(k_idx) > q_abs) {
                continue;
            }

            uint kv_base = ((uint(block_id) * params.block_size + t) * params.num_heads + h) * params.head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < params.head_dim; ++d) {
                dot += q_local[d] * k_cache[kv_base + d];
            }
            float score = dot * scale;

            float m_new = max(m_i, score);
            float exp_scale = exp(m_i - m_new);
            float exp_score = exp(score - m_new);
            l_i = l_i * exp_scale + exp_score;

            for (uint d = 0; d < params.head_dim; ++d) {
                acc[d] = acc[d] * exp_scale + exp_score * v_cache[kv_base + d];
            }
            m_i = m_new;
        }
    }

    float inv_l = l_i > 0.0f ? (1.0f / l_i) : 0.0f;
    uint o_base = ((b * params.num_heads + h) * params.seq_len + q_pos) * params.head_dim;
    for (uint d = 0; d < params.head_dim; ++d) {
        o[o_base + d] = acc[d] * inv_l;
    }
}

kernel void paged_attention_forward_f16(
    device const half *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device const int *block_tables [[buffer(3)]],
    device const int *block_offsets [[buffer(4)]],
    device half *o [[buffer(5)]],
    constant PagedAttentionParams &params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.seq_len;
    if (tid >= total || params.head_dim == 0 || params.head_dim > MAX_HEAD_DIM || params.block_size == 0) {
        return;
    }

    uint q_pos = tid % params.seq_len;
    uint rem = tid / params.seq_len;
    uint h = rem % params.num_heads;
    uint b = rem / params.num_heads;

    int position_offset = block_offsets[b];
    int kv_len_i = position_offset + int(params.seq_len);
    if (kv_len_i <= 0) {
        return;
    }
    uint kv_len = uint(kv_len_i);

    float q_local[MAX_HEAD_DIM];
    uint q_base = ((b * params.num_heads + h) * params.seq_len + q_pos) * params.head_dim;
    for (uint d = 0; d < params.head_dim; ++d) {
        q_local[d] = float(q[q_base + d]);
    }

    float acc[MAX_HEAD_DIM];
    for (uint d = 0; d < params.head_dim; ++d) {
        acc[d] = 0.0f;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float scale = rsqrt(float(params.head_dim));

    uint table_base = b * kv_len;
    uint num_blocks = (kv_len + params.block_size - 1) / params.block_size;
    int q_abs = position_offset + int(q_pos);

    for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint token_base = block_idx * params.block_size;
        if (token_base >= kv_len) {
            break;
        }

        int block_id = block_tables[table_base + token_base];
        uint tokens_in_block = kv_len - token_base;
        if (tokens_in_block > params.block_size) {
            tokens_in_block = params.block_size;
        }

        for (uint t = 0; t < tokens_in_block; ++t) {
            uint k_idx = token_base + t;
            if (int(k_idx) > q_abs) {
                continue;
            }

            uint kv_base = ((uint(block_id) * params.block_size + t) * params.num_heads + h) * params.head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < params.head_dim; ++d) {
                dot += q_local[d] * float(k_cache[kv_base + d]);
            }
            float score = dot * scale;

            float m_new = max(m_i, score);
            float exp_scale = exp(m_i - m_new);
            float exp_score = exp(score - m_new);
            l_i = l_i * exp_scale + exp_score;

            for (uint d = 0; d < params.head_dim; ++d) {
                acc[d] = acc[d] * exp_scale + exp_score * float(v_cache[kv_base + d]);
            }
            m_i = m_new;
        }
    }

    float inv_l = l_i > 0.0f ? (1.0f / l_i) : 0.0f;
    uint o_base = ((b * params.num_heads + h) * params.seq_len + q_pos) * params.head_dim;
    for (uint d = 0; d < params.head_dim; ++d) {
        o[o_base + d] = half(acc[d] * inv_l);
    }
}
