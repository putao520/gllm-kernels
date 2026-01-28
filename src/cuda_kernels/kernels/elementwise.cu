#include <cuda_runtime.h>

extern "C" __global__ void elementwise_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

// Permute [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
extern "C" __global__ void permute_qkv_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int seq,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * num_heads * head_dim;
    if (idx >= total) {
        return;
    }

    int d = idx % head_dim;
    int t = idx / head_dim;
    int s = t % seq;
    int h = (t / seq) % num_heads;
    int b = t / (seq * num_heads);

    int input_idx = ((b * seq + s) * num_heads + h) * head_dim + d;
    output[idx] = input[input_idx];
}

// Permute [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
extern "C" __global__ void permute_qkv_back_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int seq,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * num_heads * head_dim;
    if (idx >= total) {
        return;
    }

    int d = idx % head_dim;
    int t = idx / head_dim;
    int s = t % seq;
    int h = (t / seq) % num_heads;
    int b = t / (seq * num_heads);

    int input_idx = ((b * num_heads + h) * seq + s) * head_dim + d;
    output[idx] = input[input_idx];
}
