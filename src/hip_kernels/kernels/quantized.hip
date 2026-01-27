#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Quantized matrix multiplication kernels for AMD GPUs
// Supports Q4, Q8, and AWQ formats

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 64;  // AMD wavefront size

// Q4 dequantization helper (4-bit quantization)
__device__ __forceinline__ float dequantize_q4(uint8_t packed, int idx, float scale, float zero_point) {
    // Extract 4-bit value (2 values per byte)
    uint8_t val = (idx & 1) ? (packed >> 4) : (packed & 0x0F);
    return (float(val) - zero_point) * scale;
}

// Q8 dequantization helper (8-bit quantization)
__device__ __forceinline__ float dequantize_q8(int8_t val, float scale, float zero_point) {
    return (float(val) - zero_point) * scale;
}

// Q4 matmul kernel: input [M, K] x weight_q4 [K, N] -> output [M, N]
// weight_q4 is packed (2 values per byte), scales/zeros per group
extern "C" __global__ void q4_matmul_f32(
    const float* __restrict__ input,       // [M, K]
    const uint8_t* __restrict__ weight,    // [K/2, N] packed Q4
    const float* __restrict__ scales,      // [num_groups, N]
    const float* __restrict__ zeros,       // [num_groups, N]
    float* __restrict__ output,            // [M, N]
    int M,
    int K,
    int N,
    int group_size                         // typically 128
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int num_groups = (K + group_size - 1) / group_size;

    for (int k = 0; k < K; k++) {
        // Determine which group this k belongs to
        int group_idx = k / group_size;
        float scale = scales[group_idx * N + col];
        float zero = zeros[group_idx * N + col];

        // Get packed weight (2 values per byte)
        int weight_idx = (k / 2) * N + col;
        uint8_t packed = weight[weight_idx];

        // Dequantize
        float w = dequantize_q4(packed, k, scale, zero);
        float x = input[row * K + k];

        sum += x * w;
    }

    output[row * N + col] = sum;
}

// Q4 matmul with f16 input
extern "C" __global__ void q4_matmul_f16(
    const __half* __restrict__ input,      // [M, K]
    const uint8_t* __restrict__ weight,    // [K/2, N] packed Q4
    const float* __restrict__ scales,      // [num_groups, N]
    const float* __restrict__ zeros,       // [num_groups, N]
    __half* __restrict__ output,           // [M, N]
    int M,
    int K,
    int N,
    int group_size
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        int group_idx = k / group_size;
        float scale = scales[group_idx * N + col];
        float zero = zeros[group_idx * N + col];

        int weight_idx = (k / 2) * N + col;
        uint8_t packed = weight[weight_idx];

        float w = dequantize_q4(packed, k, scale, zero);
        float x = __half2float(input[row * K + k]);

        sum += x * w;
    }

    output[row * N + col] = __float2half(sum);
}

// Q8 matmul kernel: input [M, K] x weight_q8 [K, N] -> output [M, N]
extern "C" __global__ void q8_matmul_f32(
    const float* __restrict__ input,       // [M, K]
    const int8_t* __restrict__ weight,     // [K, N] Q8
    const float* __restrict__ scales,      // [num_groups, N]
    const float* __restrict__ zeros,       // [num_groups, N]
    float* __restrict__ output,            // [M, N]
    int M,
    int K,
    int N,
    int group_size
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        int group_idx = k / group_size;
        float scale = scales[group_idx * N + col];
        float zero = zeros[group_idx * N + col];

        int8_t w_q8 = weight[k * N + col];
        float w = dequantize_q8(w_q8, scale, zero);
        float x = input[row * K + k];

        sum += x * w;
    }

    output[row * N + col] = sum;
}

// Q8 matmul with f16 input
extern "C" __global__ void q8_matmul_f16(
    const __half* __restrict__ input,      // [M, K]
    const int8_t* __restrict__ weight,     // [K, N] Q8
    const float* __restrict__ scales,      // [num_groups, N]
    const float* __restrict__ zeros,       // [num_groups, N]
    __half* __restrict__ output,           // [M, N]
    int M,
    int K,
    int N,
    int group_size
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        int group_idx = k / group_size;
        float scale = scales[group_idx * N + col];
        float zero = zeros[group_idx * N + col];

        int8_t w_q8 = weight[k * N + col];
        float w = dequantize_q8(w_q8, scale, zero);
        float x = __half2float(input[row * K + k]);

        sum += x * w;
    }

    output[row * N + col] = __float2half(sum);
}

// AWQ (Activation-aware Weight Quantization) dequantization
// AWQ uses per-channel scaling with packed 4-bit weights
__device__ __forceinline__ float dequantize_awq(
    uint32_t packed,  // 8 x 4-bit values
    int idx,          // which of the 8 values (0-7)
    float scale
) {
    // Extract 4-bit value at position idx
    uint8_t val = (packed >> (idx * 4)) & 0x0F;
    // AWQ uses symmetric quantization (no zero point, signed)
    int8_t signed_val = (int8_t)val - 8;  // Convert to signed [-8, 7]
    return float(signed_val) * scale;
}

// AWQ matmul kernel
// weight is packed: 8 x 4-bit values per uint32_t
extern "C" __global__ void awq_matmul_f32(
    const float* __restrict__ input,       // [M, K]
    const uint32_t* __restrict__ weight,   // [K/8, N] packed AWQ
    const float* __restrict__ scales,      // [K/group_size, N]
    float* __restrict__ output,            // [M, N]
    int M,
    int K,
    int N,
    int group_size
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int num_packed = K / 8;  // Number of uint32_t per column

    for (int pk = 0; pk < num_packed; pk++) {
        uint32_t packed = weight[pk * N + col];

        // Unpack and process 8 values
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k = pk * 8 + i;
            if (k >= K) break;

            int group_idx = k / group_size;
            float scale = scales[group_idx * N + col];

            float w = dequantize_awq(packed, i, scale);
            float x = input[row * K + k];

            sum += x * w;
        }
    }

    // Handle remainder if K is not divisible by 8
    int remainder_start = num_packed * 8;
    if (remainder_start < K) {
        uint32_t packed = weight[num_packed * N + col];
        for (int i = 0; remainder_start + i < K; i++) {
            int k = remainder_start + i;
            int group_idx = k / group_size;
            float scale = scales[group_idx * N + col];

            float w = dequantize_awq(packed, i, scale);
            float x = input[row * K + k];

            sum += x * w;
        }
    }

    output[row * N + col] = sum;
}

// AWQ matmul with f16 input
extern "C" __global__ void awq_matmul_f16(
    const __half* __restrict__ input,      // [M, K]
    const uint32_t* __restrict__ weight,   // [K/8, N] packed AWQ
    const float* __restrict__ scales,      // [K/group_size, N]
    __half* __restrict__ output,           // [M, N]
    int M,
    int K,
    int N,
    int group_size
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int num_packed = K / 8;

    for (int pk = 0; pk < num_packed; pk++) {
        uint32_t packed = weight[pk * N + col];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k = pk * 8 + i;
            if (k >= K) break;

            int group_idx = k / group_size;
            float scale = scales[group_idx * N + col];

            float w = dequantize_awq(packed, i, scale);
            float x = __half2float(input[row * K + k]);

            sum += x * w;
        }
    }

    int remainder_start = num_packed * 8;
    if (remainder_start < K) {
        uint32_t packed = weight[num_packed * N + col];
        for (int i = 0; remainder_start + i < K; i++) {
            int k = remainder_start + i;
            int group_idx = k / group_size;
            float scale = scales[group_idx * N + col];

            float w = dequantize_awq(packed, i, scale);
            float x = __half2float(input[row * K + k]);

            sum += x * w;
        }
    }

    output[row * N + col] = __float2half(sum);
}

// Fused dequantize kernel (for cases where we want to dequantize to fp32/fp16 first)
extern "C" __global__ void dequantize_q4_f32(
    const uint8_t* __restrict__ weight,    // [K/2, N] packed Q4
    const float* __restrict__ scales,      // [num_groups, N]
    const float* __restrict__ zeros,       // [num_groups, N]
    float* __restrict__ output,            // [K, N]
    int K,
    int N,
    int group_size
) {
    int k = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K || col >= N) return;

    int group_idx = k / group_size;
    float scale = scales[group_idx * N + col];
    float zero = zeros[group_idx * N + col];

    int weight_idx = (k / 2) * N + col;
    uint8_t packed = weight[weight_idx];

    output[k * N + col] = dequantize_q4(packed, k, scale, zero);
}

extern "C" __global__ void dequantize_q8_f32(
    const int8_t* __restrict__ weight,     // [K, N] Q8
    const float* __restrict__ scales,      // [num_groups, N]
    const float* __restrict__ zeros,       // [num_groups, N]
    float* __restrict__ output,            // [K, N]
    int K,
    int N,
    int group_size
) {
    int k = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K || col >= N) return;

    int group_idx = k / group_size;
    float scale = scales[group_idx * N + col];
    float zero = zeros[group_idx * N + col];

    int8_t w_q8 = weight[k * N + col];
    output[k * N + col] = dequantize_q8(w_q8, scale, zero);
}

// Q4_0 dequantize kernel aligned with CPU/CUDA layout.
// - qweight: [num_blocks, 16 bytes], each block = 32 values packed as nibbles
// - scales:  [num_blocks], one scale per block
// - output:  [num_blocks * 32] laid out as [n, k] row-major by block order
extern "C" __global__ void q4_dequantize_f32(
    const uint8_t* __restrict__ qweight,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int num_blocks
) {
    const int global_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_values = num_blocks * 32;
    if (global_idx >= total_values) return;

    const int block_idx = global_idx / 32;
    const int in_block = global_idx - block_idx * 32;

    const int byte_idx = in_block >> 1;
    const uint8_t packed = qweight[block_idx * 16 + byte_idx];
    const uint8_t nibble = (in_block & 1) ? (packed >> 4) : (packed & 0x0F);
    const int q = static_cast<int>(nibble) - 8;
    const float scale = __half2float(scales[block_idx]);

    output[global_idx] = static_cast<float>(q) * scale;
}

// AWQ INT4 dequantize kernel aligned with CPU/CUDA layout.
// Layout matches ops::quantized::awq_dequantize_cpu:
// - qweight: [n/8, k]
// - qzeros:  [n/8, groups]
// - scales:  [n, groups]
// - output:  [n, k] row-major
extern "C" __global__ void awq_dequantize_f32(
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int n,
    int k,
    int group_size,
    int groups
) {
    const int global_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_values = n * k;
    if (global_idx >= total_values) return;

    const int out = global_idx / k;
    const int k_idx = global_idx - out * k;
    const int group = k_idx / group_size;
    const int packed_row = out >> 3;
    const int nibble = out & 7;

    const uint32_t weight_word = qweight[packed_row * k + k_idx];
    const uint32_t zero_word = qzeros[packed_row * groups + group];
    const int w = static_cast<int>((weight_word >> (nibble * 4)) & 0x0F);
    const int z = static_cast<int>((zero_word >> (nibble * 4)) & 0x0F);
    const float scale = __half2float(scales[out * groups + group]);

    output[global_idx] = static_cast<float>(w - z) * scale;
}
