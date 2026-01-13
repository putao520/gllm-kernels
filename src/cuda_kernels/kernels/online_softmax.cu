// Online softmax kernel with two-pass reduction (max, sum).
// The kernel processes one row per block in [batch, heads, seq, seq].

#include <float.h>
#include <math.h>
#include <stddef.h>

extern "C" __global__ void online_softmax_forward(
    const float* __restrict__ logits,     // [B, H, S, S]
    float* __restrict__ output,          // [B, H, S, S]
    float* __restrict__ max_val,         // [B, H, S]
    float* __restrict__ sum_exp,         // [B, H, S]
    int batch_size,
    int num_heads,
    int seq_len
) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const size_t total_rows = (size_t)batch_size * (size_t)num_heads * (size_t)seq_len;
    const size_t row_idx = (size_t)blockIdx.x;
    if (row_idx >= total_rows) {
        return;
    }

    const size_t row_offset = row_idx * (size_t)seq_len;

    // Pass 1: reduce max.
    float local_max = -INFINITY;
    for (int col = tid; col < seq_len; col += block_size) {
        const float val = logits[row_offset + (size_t)col];
        local_max = fmaxf(local_max, val);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = shared[0];
    if (tid == 0) {
        max_val[row_idx] = row_max;
    }
    __syncthreads();

    // Pass 2: reduce sum of exp and write unnormalized output.
    float local_sum = 0.0f;
    for (int col = tid; col < seq_len; col += block_size) {
        const float val = expf(logits[row_offset + (size_t)col] - row_max);
        output[row_offset + (size_t)col] = val;
        local_sum += val;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    const float row_sum = shared[0];
    if (tid == 0) {
        sum_exp[row_idx] = row_sum;
    }
    __syncthreads();

    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int col = tid; col < seq_len; col += block_size) {
        output[row_offset + (size_t)col] *= inv_sum;
    }
}
