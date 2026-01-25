#include <hip/hip_runtime.h>

extern "C" __global__ void linear_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int in_features,
    const int out_features,
    const int has_bias
) {
    // Grid: [out_features, batch_size]
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx >= out_features) return;
    
    // For now, handle y as batch index
    // Note: In real production, we might use a 1D grid and calculate indices
    
    float sum = 0.0f;
    int input_offset = batch_idx * in_features;
    int weight_offset = out_idx * in_features;

    for (int i = 0; i < in_features; i++) {
        sum += input[input_offset + i] * weight[weight_offset + i];
    }

    if (has_bias) {
        sum += bias[out_idx];
    }

    output[batch_idx * out_features + out_idx] = sum;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

extern "C" __global__ void fused_gate_up_silu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight_gate,
    const float* __restrict__ weight_up,
    float* __restrict__ output,
    const int in_features,
    const int out_features
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx >= out_features) return;

    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    int input_offset = batch_idx * in_features;
    int weight_offset = out_idx * in_features;

    for (int i = 0; i < in_features; i++) {
        float in_val = input[input_offset + i];
        sum_gate += in_val * weight_gate[weight_offset + i];
        sum_up += in_val * weight_up[weight_offset + i];
    }

    output[batch_idx * out_features + out_idx] = silu(sum_gate) * sum_up;
}
