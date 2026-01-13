// Paged attention kernel for AMD GPUs using HIP.
// Implements online softmax accumulation with paged KV cache indexing.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>

#define MAX_HEAD_DIM 256

extern "C" __global__ void paged_attention_forward_f32(
    const float* __restrict__ Q,           // [B, H, Q, D]
    const float* __restrict__ K_CACHE,     // [num_blocks, block_size, H, D]
    const float* __restrict__ V_CACHE,     // [num_blocks, block_size, H, D]
    const int* __restrict__ BLOCK_TABLES,  // [B, KV] block id per token
    const int* __restrict__ BLOCK_OFFSETS, // [B] position offset
    float* __restrict__ O,                 // [B, H, Q, D]
    int batch_size,
    int num_heads,
    int head_dim,
    int block_size,
    int seq_len
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = batch_size * num_heads * seq_len;
    if (idx >= total || head_dim <= 0 || head_dim > MAX_HEAD_DIM || block_size <= 0) {
        return;
    }

    int q_pos = idx % seq_len;
    int rem = idx / seq_len;
    int h = rem % num_heads;
    int b = rem / num_heads;

    int position_offset = BLOCK_OFFSETS[b];
    int kv_len = position_offset + seq_len;
    if (kv_len <= 0) {
        return;
    }

    float q_local[MAX_HEAD_DIM];
    int q_base = ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        q_local[d] = Q[q_base + d];
    }

    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float scale = rsqrtf((float)head_dim);

    int table_base = b * kv_len;
    int num_blocks = (kv_len + block_size - 1) / block_size;
    int q_abs = position_offset + q_pos;

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int token_base = block_idx * block_size;
        if (token_base >= kv_len) {
            break;
        }

        int block_id = BLOCK_TABLES[table_base + token_base];
        int tokens_in_block = kv_len - token_base;
        if (tokens_in_block > block_size) {
            tokens_in_block = block_size;
        }

        for (int t = 0; t < tokens_in_block; ++t) {
            int k_idx = token_base + t;
            if (k_idx > q_abs) {
                continue;
            }

            int kv_base = ((block_id * block_size + t) * num_heads + h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_local[d] * K_CACHE[kv_base + d];
            }
            float score = dot * scale;

            float m_new = fmaxf(m_i, score);
            float exp_scale = expf(m_i - m_new);
            float exp_score = expf(score - m_new);
            l_i = l_i * exp_scale + exp_score;

            for (int d = 0; d < head_dim; ++d) {
                acc[d] = acc[d] * exp_scale + exp_score * V_CACHE[kv_base + d];
            }
            m_i = m_new;
        }
    }

    float inv_l = l_i > 0.0f ? (1.0f / l_i) : 0.0f;
    int out_base = ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        O[out_base + d] = acc[d] * inv_l;
    }
}

extern "C" __global__ void paged_attention_forward_f16(
    const half* __restrict__ Q,
    const half* __restrict__ K_CACHE,
    const half* __restrict__ V_CACHE,
    const int* __restrict__ BLOCK_TABLES,
    const int* __restrict__ BLOCK_OFFSETS,
    half* __restrict__ O,
    int batch_size,
    int num_heads,
    int head_dim,
    int block_size,
    int seq_len
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = batch_size * num_heads * seq_len;
    if (idx >= total || head_dim <= 0 || head_dim > MAX_HEAD_DIM || block_size <= 0) {
        return;
    }

    int q_pos = idx % seq_len;
    int rem = idx / seq_len;
    int h = rem % num_heads;
    int b = rem / num_heads;

    int position_offset = BLOCK_OFFSETS[b];
    int kv_len = position_offset + seq_len;
    if (kv_len <= 0) {
        return;
    }

    float q_local[MAX_HEAD_DIM];
    int q_base = ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        q_local[d] = __half2float(Q[q_base + d]);
    }

    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float scale = rsqrtf((float)head_dim);

    int table_base = b * kv_len;
    int num_blocks = (kv_len + block_size - 1) / block_size;
    int q_abs = position_offset + q_pos;

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int token_base = block_idx * block_size;
        if (token_base >= kv_len) {
            break;
        }

        int block_id = BLOCK_TABLES[table_base + token_base];
        int tokens_in_block = kv_len - token_base;
        if (tokens_in_block > block_size) {
            tokens_in_block = block_size;
        }

        for (int t = 0; t < tokens_in_block; ++t) {
            int k_idx = token_base + t;
            if (k_idx > q_abs) {
                continue;
            }

            int kv_base = ((block_id * block_size + t) * num_heads + h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_local[d] * __half2float(K_CACHE[kv_base + d]);
            }
            float score = dot * scale;

            float m_new = fmaxf(m_i, score);
            float exp_scale = expf(m_i - m_new);
            float exp_score = expf(score - m_new);
            l_i = l_i * exp_scale + exp_score;

            for (int d = 0; d < head_dim; ++d) {
                acc[d] = acc[d] * exp_scale + exp_score * __half2float(V_CACHE[kv_base + d]);
            }
            m_i = m_new;
        }
    }

    float inv_l = l_i > 0.0f ? (1.0f / l_i) : 0.0f;
    int out_base = ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        O[out_base + d] = __float2half_rn(acc[d] * inv_l);
    }
}
