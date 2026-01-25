#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f32(
    const device float* input [[ buffer(0) ]],
    const device float* weight [[ buffer(1) ]],
    device float* output [[ buffer(2) ]],
    constant int& hidden [[ buffer(3) ]],
    constant int& rows [[ buffer(4) ]],
    constant float& eps [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint row = gid.x;
    if (row >= static_cast<uint>(rows)) {
        return;
    }

    int base = static_cast<int>(row) * hidden;
    float sum = 0.0f;

    for (int i = 0; i < hidden; i++) {
        float v = input[base + i];
        sum += v * v;
    }

    float mean = sum / static_cast<float>(hidden);
    float inv_rms = rsqrt(mean + eps);

    for (int i = 0; i < hidden; i++) {
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }
}
