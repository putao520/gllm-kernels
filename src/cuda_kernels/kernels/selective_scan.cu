// Selective scan CUDA kernel for Mamba-style SSMs.
// Uses block-parallel reduction over state_dim with online state updates.

#include <math.h>

#define EXP_CLAMP_MIN -50.0f
#define EXP_CLAMP_MAX 20.0f

__device__ __forceinline__ float clamp_exp_arg(float value) {
    return fminf(fmaxf(value, EXP_CLAMP_MIN), EXP_CLAMP_MAX);
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared, int tid, int block_size) {
    shared[tid] = val;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

extern "C" __global__ void selective_scan_fwd(
    const float* __restrict__ u,       // [B, T, D]
    const float* __restrict__ delta,   // [B, T, D]
    const float* __restrict__ A,       // [N, D]
    const float* __restrict__ B,       // [B, T, N]
    const float* __restrict__ C,       // [B, T, N]
    float* __restrict__ output,        // [B, T, D]
    int batch_size,
    int seq_len,
    int state_dim,
    int expanded_dim
) {
    if (batch_size <= 0 || seq_len <= 0 || state_dim <= 0 || expanded_dim <= 0) {
        return;
    }

    int block_idx = (int)blockIdx.x;
    int b = block_idx / expanded_dim;
    int d = block_idx - b * expanded_dim;
    if (b >= batch_size || d >= expanded_dim) {
        return;
    }

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float smem[];
    float* state = smem;                       // [state_dim]
    float* a_col = state + state_dim;          // [state_dim]
    float* reduce = a_col + state_dim;         // [block_size]

    for (int s = tid; s < state_dim; s += block_size) {
        a_col[s] = A[s * expanded_dim + d];
        state[s] = 0.0f;
    }
    __syncthreads();

    int base_ud = (b * seq_len * expanded_dim) + d;
    int base_bc = b * seq_len * state_dim;

    for (int t = 0; t < seq_len; ++t) {
        float dt = delta[base_ud + t * expanded_dim];
        float u_val = u[base_ud + t * expanded_dim];
        float input = dt * u_val;

        int bc_offset = base_bc + t * state_dim;
        float partial = 0.0f;

        for (int s = tid; s < state_dim; s += block_size) {
            float decay_arg = clamp_exp_arg(dt * a_col[s]);
            float decay = __expf(decay_arg);
            float x = state[s] * decay + B[bc_offset + s] * input;
            state[s] = x;
            partial += C[bc_offset + s] * x;
        }

        float sum = block_reduce_sum(partial, reduce, tid, block_size);
        if (tid == 0) {
            output[base_ud + t * expanded_dim] = sum;
        }
        __syncthreads();
    }
}
