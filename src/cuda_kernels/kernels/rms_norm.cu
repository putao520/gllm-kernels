#include <cuda_runtime.h>

extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int hidden,
    const int rows,
    const float eps
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int base = row * hidden;
    float sum = 0.0f;

    for (int i = 0; i < hidden; ++i) {
        float v = input[base + i];
        sum += v * v;
    }

    float mean = sum / static_cast<float>(hidden);
    float inv_rms = rsqrtf(mean + eps);

    for (int i = 0; i < hidden; ++i) {
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }
}
