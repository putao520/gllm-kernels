#include <metal_stdlib>
using namespace metal;

kernel void linear_forward_kernel(
    const device float* input [[ buffer(0) ]],
    const device float* weight [[ buffer(1) ]],
    const device float* bias [[ buffer(2) ]],
    device float* output [[ buffer(3) ]],
    constant int& in_features [[ buffer(4) ]],
    constant int& out_features [[ buffer(5) ]],
    constant int& has_bias [[ buffer(6) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    // 1D Linear: Input [1, in], Weight [out, in], Output [1, out]
    // Or Batch Linear: Input [batch, in], Weight [out, in], Output [batch, out]
    // gid.x corresponds to output feature index? Or batch index?
    // Dispatch strategy:
    // If we assume input is [batch, in] and we want [batch, out].
    // Grid: [out_features, batch_size]
    // gid.x = output feature index (0..out)
    // gid.y = batch index (0..batch)

    int out_idx = gid.x;
    int batch_idx = gid.y;

    if (out_idx >= out_features) return;
    
    // For large batch logic, we need to pass batch size or valid range?
    // We assume grid covers valid range perfectly or checks bounds if passed.
    // For now simple 1D grid if batch=1?
    // Let's implement generic Grid [out_features, batch_size].
    
    // Input offset: batch_idx * in_features
    float sum = 0.0f;
    int input_offset = batch_idx * in_features;
    int weight_offset = out_idx * in_features;

    for (int i = 0; i < in_features; i++) {
        sum += input[input_offset + i] * weight[weight_offset + i];
    }

    if (has_bias) {
        sum += bias[out_idx];
    }

    // Output offset: batch_idx * out_features + out_idx
    output[batch_idx * out_features + out_idx] = sum;
}

float silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void fused_gate_up_silu(
    device const float* input [[buffer(0)]],
    device const float* weight_gate [[buffer(1)]],
    device const float* weight_up [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& in_features [[buffer(4)]],
    constant int& out_features [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int out_idx = gid.x;
    int batch_idx = gid.y;

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
