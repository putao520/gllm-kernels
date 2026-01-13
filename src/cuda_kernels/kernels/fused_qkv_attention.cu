// Fused QKV projection + attention kernel.
//
// This kernel computes Q, K, V projections on-the-fly and performs
// block-wise attention without writing intermediate Q/K/V to global memory.
// The implementation favors correctness and fusion while keeping the
// shared-memory footprint bounded. Tile sizes are tunable via BLOCK_M/BLOCK_N.

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#ifndef BLOCK_M
#define BLOCK_M 16
#endif

#ifndef BLOCK_N
#define BLOCK_N 16
#endif

#define MAX_HEAD_DIM 256

extern "C" __global__ void fused_qkv_attention_forward(
    const float* __restrict__ input,       // [B, S, H]
    const float* __restrict__ W_qkv,        // [3, num_heads, head_dim, hidden_dim]
    const float* __restrict__ b_qkv,        // [3, num_heads, head_dim] (can be NULL)
    float* __restrict__ output,             // [B, num_heads, S, head_dim]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim
) {
    extern __shared__ float smem[];

    const int tid = (int)threadIdx.x;
    const int q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int total_blocks = batch_size * num_heads * q_blocks;
    const int block_idx = (int)blockIdx.x;

    if (block_idx >= total_blocks) {
        return;
    }

    const int bh_idx = block_idx / q_blocks;
    const int q_block = block_idx - bh_idx * q_blocks;
    const int b = bh_idx / num_heads;
    const int h = bh_idx - b * num_heads;

    const int q_start = q_block * BLOCK_M;
    const int input_batch_stride = seq_len * hidden_dim;
    const float* input_base = input + b * input_batch_stride;

    const int weight_head_stride = head_dim * hidden_dim;
    const int weight_qkv_stride = num_heads * weight_head_stride;

    const float* w_q = W_qkv + 0 * weight_qkv_stride + h * weight_head_stride;
    const float* w_k = W_qkv + 1 * weight_qkv_stride + h * weight_head_stride;
    const float* w_v = W_qkv + 2 * weight_qkv_stride + h * weight_head_stride;

    const float* b_q = b_qkv ? (b_qkv + 0 * num_heads * head_dim + h * head_dim) : nullptr;
    const float* b_k = b_qkv ? (b_qkv + 1 * num_heads * head_dim + h * head_dim) : nullptr;
    const float* b_v = b_qkv ? (b_qkv + 2 * num_heads * head_dim + h * head_dim) : nullptr;

    float* q_smem = smem;
    float* k_smem = q_smem + BLOCK_M * head_dim;
    float* v_smem = k_smem + BLOCK_N * head_dim;

    // Compute Q tile [BLOCK_M, head_dim]
    for (int i = tid; i < BLOCK_M * head_dim; i += (int)blockDim.x) {
        const int m = i / head_dim;
        const int d = i - m * head_dim;
        const int pos = q_start + m;

        float acc = 0.0f;
        if (b_q) {
            acc = b_q[d];
        }
        if (pos < seq_len) {
            const float* in_ptr = input_base + pos * hidden_dim;
            const float* w_ptr = w_q + d * hidden_dim;
            for (int k = 0; k < hidden_dim; ++k) {
                acc += in_ptr[k] * w_ptr[k];
            }
        }
        q_smem[m * head_dim + d] = acc;
    }

    __syncthreads();

    const int q_pos = q_start + tid;
    const bool active = (tid < BLOCK_M) && (q_pos < seq_len);
    float acc[MAX_HEAD_DIM];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    const float scale = rsqrtf((float)head_dim);
    const float* q_ptr = active ? (q_smem + tid * head_dim) : nullptr;

    if (active) {
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = 0.0f;
        }
    }

    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
        // Compute K/V tile [BLOCK_N, head_dim]
        for (int i = tid; i < BLOCK_N * head_dim; i += (int)blockDim.x) {
            const int n = i / head_dim;
            const int d = i - n * head_dim;
            const int pos = kv_start + n;

            float k_acc = 0.0f;
            float v_acc = 0.0f;
            if (b_k) {
                k_acc = b_k[d];
            }
            if (b_v) {
                v_acc = b_v[d];
            }
            if (pos < seq_len) {
                const float* in_ptr = input_base + pos * hidden_dim;
                const float* wk_ptr = w_k + d * hidden_dim;
                const float* wv_ptr = w_v + d * hidden_dim;
                for (int k = 0; k < hidden_dim; ++k) {
                    const float in_val = in_ptr[k];
                    k_acc += in_val * wk_ptr[k];
                    v_acc += in_val * wv_ptr[k];
                }
            }
            k_smem[n * head_dim + d] = k_acc;
            v_smem[n * head_dim + d] = v_acc;
        }

        __syncthreads();

        if (active) {
            const int kv_limit = min(BLOCK_N, seq_len - kv_start);
            for (int n = 0; n < kv_limit; ++n) {
                const float* k_ptr = k_smem + n * head_dim;
                const float* v_ptr = v_smem + n * head_dim;

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }

                const float score = dot * scale;
                const float m_new = fmaxf(m_i, score);
                const float exp_old = expf(m_i - m_new);
                const float exp_new = expf(score - m_new);

                l_i = l_i * exp_old + exp_new;
                for (int d = 0; d < head_dim; ++d) {
                    acc[d] = acc[d] * exp_old + exp_new * v_ptr[d];
                }
                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (active) {
        const float inv_l = l_i > 0.0f ? (1.0f / l_i) : 0.0f;
        const int head_stride = seq_len * head_dim;
        const int batch_stride = num_heads * head_stride;
        float* out_ptr = output + b * batch_stride + h * head_stride + q_pos * head_dim;

        for (int d = 0; d < head_dim; ++d) {
            out_ptr[d] = acc[d] * inv_l;
        }
    }
}

extern "C" __global__ void fused_qkv_attention_forward_f16(
    const half* __restrict__ input,        // [B, S, H]
    const half* __restrict__ W_qkv,         // [3, num_heads, head_dim, hidden_dim]
    const half* __restrict__ b_qkv,         // [3, num_heads, head_dim] (can be NULL)
    half* __restrict__ output,              // [B, num_heads, S, head_dim]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim
) {
    extern __shared__ float smem[];

    const int tid = (int)threadIdx.x;
    const int q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int total_blocks = batch_size * num_heads * q_blocks;
    const int block_idx = (int)blockIdx.x;

    if (block_idx >= total_blocks) {
        return;
    }

    const int bh_idx = block_idx / q_blocks;
    const int q_block = block_idx - bh_idx * q_blocks;
    const int b = bh_idx / num_heads;
    const int h = bh_idx - b * num_heads;

    const int q_start = q_block * BLOCK_M;
    const int input_batch_stride = seq_len * hidden_dim;
    const half* input_base = input + b * input_batch_stride;

    const int weight_head_stride = head_dim * hidden_dim;
    const int weight_qkv_stride = num_heads * weight_head_stride;

    const half* w_q = W_qkv + 0 * weight_qkv_stride + h * weight_head_stride;
    const half* w_k = W_qkv + 1 * weight_qkv_stride + h * weight_head_stride;
    const half* w_v = W_qkv + 2 * weight_qkv_stride + h * weight_head_stride;

    const half* b_q = b_qkv ? (b_qkv + 0 * num_heads * head_dim + h * head_dim) : nullptr;
    const half* b_k = b_qkv ? (b_qkv + 1 * num_heads * head_dim + h * head_dim) : nullptr;
    const half* b_v = b_qkv ? (b_qkv + 2 * num_heads * head_dim + h * head_dim) : nullptr;

    float* q_smem = smem;
    float* k_smem = q_smem + BLOCK_M * head_dim;
    float* v_smem = k_smem + BLOCK_N * head_dim;

    // Compute Q tile [BLOCK_M, head_dim]
    for (int i = tid; i < BLOCK_M * head_dim; i += (int)blockDim.x) {
        const int m = i / head_dim;
        const int d = i - m * head_dim;
        const int pos = q_start + m;

        float acc = 0.0f;
        if (b_q) {
            acc = __half2float(b_q[d]);
        }
        if (pos < seq_len) {
            const half* in_ptr = input_base + pos * hidden_dim;
            const half* w_ptr = w_q + d * hidden_dim;
            for (int k = 0; k < hidden_dim; ++k) {
                acc += __half2float(in_ptr[k]) * __half2float(w_ptr[k]);
            }
        }
        q_smem[m * head_dim + d] = acc;
    }

    __syncthreads();

    const int q_pos = q_start + tid;
    const bool active = (tid < BLOCK_M) && (q_pos < seq_len);
    float acc[MAX_HEAD_DIM];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    const float scale = rsqrtf((float)head_dim);
    const float* q_ptr = active ? (q_smem + tid * head_dim) : nullptr;

    if (active) {
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = 0.0f;
        }
    }

    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
        // Compute K/V tile [BLOCK_N, head_dim]
        for (int i = tid; i < BLOCK_N * head_dim; i += (int)blockDim.x) {
            const int n = i / head_dim;
            const int d = i - n * head_dim;
            const int pos = kv_start + n;

            float k_acc = 0.0f;
            float v_acc = 0.0f;
            if (b_k) {
                k_acc = __half2float(b_k[d]);
            }
            if (b_v) {
                v_acc = __half2float(b_v[d]);
            }
            if (pos < seq_len) {
                const half* in_ptr = input_base + pos * hidden_dim;
                const half* wk_ptr = w_k + d * hidden_dim;
                const half* wv_ptr = w_v + d * hidden_dim;
                for (int k = 0; k < hidden_dim; ++k) {
                    const float in_val = __half2float(in_ptr[k]);
                    k_acc += in_val * __half2float(wk_ptr[k]);
                    v_acc += in_val * __half2float(wv_ptr[k]);
                }
            }
            k_smem[n * head_dim + d] = k_acc;
            v_smem[n * head_dim + d] = v_acc;
        }

        __syncthreads();

        if (active) {
            const int kv_limit = min(BLOCK_N, seq_len - kv_start);
            for (int n = 0; n < kv_limit; ++n) {
                const float* k_ptr = k_smem + n * head_dim;
                const float* v_ptr = v_smem + n * head_dim;

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }

                const float score = dot * scale;
                const float m_new = fmaxf(m_i, score);
                const float exp_old = expf(m_i - m_new);
                const float exp_new = expf(score - m_new);

                l_i = l_i * exp_old + exp_new;
                for (int d = 0; d < head_dim; ++d) {
                    acc[d] = acc[d] * exp_old + exp_new * v_ptr[d];
                }
                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (active) {
        const float inv_l = l_i > 0.0f ? (1.0f / l_i) : 0.0f;
        const int head_stride = seq_len * head_dim;
        const int batch_stride = num_heads * head_stride;
        half* out_ptr = output + b * batch_stride + h * head_stride + q_pos * head_dim;

        for (int d = 0; d < head_dim; ++d) {
            out_ptr[d] = __float2half_rn(acc[d] * inv_l);
        }
    }
}
