// FlashAttention-style HIP kernel for AMD GPUs (ROCm)
// Implements online softmax to reduce memory bandwidth.
//
// HIP is largely compatible with CUDA, with minor API differences.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>
#include <math.h>

// Configurable tile sizes
#ifndef BLOCK_M
#define BLOCK_M 64    // Query block size
#endif

#ifndef BLOCK_N
#define BLOCK_N 64    // KV block size
#endif

#ifndef BLOCK_K
#define BLOCK_K 64    // Head dimension block
#endif

#define WARP_SIZE 64  // AMD uses 64-wide wavefronts
#define MAX_HEAD_DIM 256

// Helper: warp-level reduction for max (AMD wavefront)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Helper: warp-level reduction for sum (AMD wavefront)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Helper: block-level reduction for max
__device__ float block_reduce_max(float val, float* shared, int tid, int block_size) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (tid < num_warps) {
        val = shared[tid];
    } else {
        val = -FLT_MAX;
    }

    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// Helper: block-level reduction for sum
__device__ float block_reduce_sum(float val, float* shared, int tid, int block_size) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (tid < num_warps) {
        val = shared[tid];
    } else {
        val = 0.0f;
    }

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// =============================================================================
// FlashAttention Forward - F32 version with shared memory tiling (HIP)
// =============================================================================
extern "C" __global__ void flash_attention_forward_f32(
    const float* __restrict__ Q,      // [B, H, N, D]
    const float* __restrict__ K,      // [B, H, N, D]
    const float* __restrict__ V,      // [B, H, N, D]
    float* __restrict__ O,            // [B, H, N, D]
    float* __restrict__ L,            // [B, H, N] - logsumexp (optional, can be NULL)
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int is_causal,
    const int position_offset
) {
    // Shared memory for Q, K, V tiles and reduction
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Determine which query position this block handles
    const int block_idx = blockIdx.x;
    const int total_blocks = batch_size * num_heads * ((seq_len + BLOCK_M - 1) / BLOCK_M);
    if (block_idx >= total_blocks) return;

    // Decode block index
    const int queries_per_head = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int bh_idx = block_idx / queries_per_head;
    const int q_block = block_idx % queries_per_head;
    const int b = bh_idx / num_heads;
    const int h = bh_idx % num_heads;

    // Starting positions
    const int q_start = q_block * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len);

    // Base pointers for this batch and head
    const int head_stride = seq_len * head_dim;
    const int batch_stride = num_heads * head_stride;
    const int base_offset = b * batch_stride + h * head_stride;

    const float* Q_block = Q + base_offset + q_start * head_dim;
    const float* K_base = K + base_offset;
    const float* V_base = V + base_offset;
    float* O_block = O + base_offset + q_start * head_dim;
    float* L_block = (L != NULL) ? L + b * num_heads * seq_len + h * seq_len + q_start : NULL;

    // Partition shared memory
    float* Q_smem = smem;
    float* K_smem = Q_smem + BLOCK_M * head_dim;
    float* V_smem = K_smem + BLOCK_N * head_dim;

    // Load Q tile to shared memory
    for (int i = tid; i < BLOCK_M * head_dim; i += block_size) {
        int m = i / head_dim;
        int d = i % head_dim;
        if (q_start + m < seq_len && d < head_dim) {
            Q_smem[i] = Q_block[m * head_dim + d];
        } else {
            Q_smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Initialize online softmax accumulators
    float acc[MAX_HEAD_DIM];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    // Process KV in blocks
    const int kv_end = is_causal ? min(q_start + BLOCK_M + position_offset, seq_len) : seq_len;

    for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_N) {
        const int kv_block_end = min(kv_start + BLOCK_N, seq_len);

        // Load K tile
        for (int i = tid; i < BLOCK_N * head_dim; i += block_size) {
            int n = i / head_dim;
            int d = i % head_dim;
            if (kv_start + n < seq_len && d < head_dim) {
                K_smem[i] = K_base[(kv_start + n) * head_dim + d];
            } else {
                K_smem[i] = 0.0f;
            }
        }

        // Load V tile
        for (int i = tid; i < BLOCK_N * head_dim; i += block_size) {
            int n = i / head_dim;
            int d = i % head_dim;
            if (kv_start + n < seq_len && d < head_dim) {
                V_smem[i] = V_base[(kv_start + n) * head_dim + d];
            } else {
                V_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // Each thread processes one query position in the block
        const int local_q = tid % BLOCK_M;
        const int global_q = q_start + local_q;

        if (global_q < seq_len && tid < BLOCK_M) {
            float row_max = -FLT_MAX;
            float scores[BLOCK_N];

            for (int j = 0; j < BLOCK_N && kv_start + j < kv_block_end; ++j) {
                const int kv_pos = kv_start + j;

                if (is_causal && kv_pos > global_q + position_offset) {
                    scores[j] = -FLT_MAX;
                    continue;
                }

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += Q_smem[local_q * head_dim + d] * K_smem[j * head_dim + d];
                }
                scores[j] = dot * scale;
                row_max = fmaxf(row_max, scores[j]);
            }

            // Online softmax update
            float m_new = fmaxf(m_i, row_max);
            float l_new = 0.0f;

            if (m_i > -FLT_MAX) {
                float scale_old = expf(m_i - m_new);
                l_new = l_i * scale_old;
                for (int d = 0; d < head_dim; ++d) {
                    acc[d] *= scale_old;
                }
            }

            for (int j = 0; j < BLOCK_N && kv_start + j < kv_block_end; ++j) {
                if (scores[j] > -FLT_MAX) {
                    float p = expf(scores[j] - m_new);
                    l_new += p;
                    for (int d = 0; d < head_dim; ++d) {
                        acc[d] += p * V_smem[j * head_dim + d];
                    }
                }
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // Write output
    if (tid < BLOCK_M && q_start + tid < seq_len) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            O_block[tid * head_dim + d] = acc[d] * inv_l;
        }

        if (L_block != NULL) {
            L_block[tid] = m_i + logf(l_i + 1e-10f);
        }
    }
}

// =============================================================================
// FlashAttention Forward - F16 version with shared memory tiling (HIP)
// =============================================================================
extern "C" __global__ void flash_attention_forward_f16(
    const _Float16* __restrict__ Q,
    const _Float16* __restrict__ K,
    const _Float16* __restrict__ V,
    _Float16* __restrict__ O,
    _Float16* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int is_causal,
    const int position_offset
) {
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int block_idx = blockIdx.x;

    const int queries_per_head = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int total_blocks = batch_size * num_heads * queries_per_head;
    if (block_idx >= total_blocks) return;

    const int bh_idx = block_idx / queries_per_head;
    const int q_block = block_idx % queries_per_head;
    const int b = bh_idx / num_heads;
    const int h = bh_idx % num_heads;

    const int q_start = q_block * BLOCK_M;

    const int head_stride = seq_len * head_dim;
    const int batch_stride = num_heads * head_stride;
    const int base_offset = b * batch_stride + h * head_stride;

    const _Float16* Q_block = Q + base_offset + q_start * head_dim;
    const _Float16* K_base = K + base_offset;
    const _Float16* V_base = V + base_offset;
    _Float16* O_block = O + base_offset + q_start * head_dim;

    float* Q_smem = smem;
    float* K_smem = Q_smem + BLOCK_M * head_dim;
    float* V_smem = K_smem + BLOCK_N * head_dim;

    // Load Q tile (convert f16 -> f32)
    for (int i = tid; i < BLOCK_M * head_dim; i += block_size) {
        int m = i / head_dim;
        int d = i % head_dim;
        if (q_start + m < seq_len && d < head_dim) {
            Q_smem[i] = (float)Q_block[m * head_dim + d];
        } else {
            Q_smem[i] = 0.0f;
        }
    }
    __syncthreads();

    float acc[MAX_HEAD_DIM];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    const int kv_end = is_causal ? min(q_start + BLOCK_M + position_offset, seq_len) : seq_len;

    for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_N) {
        const int kv_block_end = min(kv_start + BLOCK_N, seq_len);

        // Load K tile (f16 -> f32)
        for (int i = tid; i < BLOCK_N * head_dim; i += block_size) {
            int n = i / head_dim;
            int d = i % head_dim;
            if (kv_start + n < seq_len && d < head_dim) {
                K_smem[i] = (float)K_base[(kv_start + n) * head_dim + d];
            } else {
                K_smem[i] = 0.0f;
            }
        }

        // Load V tile (f16 -> f32)
        for (int i = tid; i < BLOCK_N * head_dim; i += block_size) {
            int n = i / head_dim;
            int d = i % head_dim;
            if (kv_start + n < seq_len && d < head_dim) {
                V_smem[i] = (float)V_base[(kv_start + n) * head_dim + d];
            } else {
                V_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        const int local_q = tid % BLOCK_M;
        const int global_q = q_start + local_q;

        if (global_q < seq_len && tid < BLOCK_M) {
            float row_max = -FLT_MAX;
            float scores[BLOCK_N];

            for (int j = 0; j < BLOCK_N && kv_start + j < kv_block_end; ++j) {
                const int kv_pos = kv_start + j;

                if (is_causal && kv_pos > global_q + position_offset) {
                    scores[j] = -FLT_MAX;
                    continue;
                }

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += Q_smem[local_q * head_dim + d] * K_smem[j * head_dim + d];
                }
                scores[j] = dot * scale;
                row_max = fmaxf(row_max, scores[j]);
            }

            float m_new = fmaxf(m_i, row_max);
            float l_new = 0.0f;

            if (m_i > -FLT_MAX) {
                float scale_old = expf(m_i - m_new);
                l_new = l_i * scale_old;
                for (int d = 0; d < head_dim; ++d) {
                    acc[d] *= scale_old;
                }
            }

            for (int j = 0; j < BLOCK_N && kv_start + j < kv_block_end; ++j) {
                if (scores[j] > -FLT_MAX) {
                    float p = expf(scores[j] - m_new);
                    l_new += p;
                    for (int d = 0; d < head_dim; ++d) {
                        acc[d] += p * V_smem[j * head_dim + d];
                    }
                }
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // Write output (f32 -> f16)
    if (tid < BLOCK_M && q_start + tid < seq_len) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            O_block[tid * head_dim + d] = (_Float16)(acc[d] * inv_l);
        }
    }
}

// =============================================================================
// Simple tiled attention (backward compatible interface)
// =============================================================================
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

    float max_score = -FLT_MAX;
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
    const _Float16* Q,
    const _Float16* K,
    const _Float16* V,
    _Float16* O,
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
    const _Float16* q_ptr = Q + base + q_pos * head_dim;

    float max_score = -FLT_MAX;
    for (int j = 0; j < seq_len; ++j) {
        if (is_causal && j > q_pos + position_offset) {
            continue;
        }
        const _Float16* k_ptr = K + base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += (float)q_ptr[d] * (float)k_ptr[d];
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
        const _Float16* k_ptr = K + base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += (float)q_ptr[d] * (float)k_ptr[d];
        }
        float score = dot * scale;
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;

        const _Float16* v_ptr = V + base + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out[d] += exp_score * (float)v_ptr[d];
        }
    }

    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    _Float16* o_ptr = O + base + q_pos * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = (_Float16)(out[d] * inv_sum);
    }
}
