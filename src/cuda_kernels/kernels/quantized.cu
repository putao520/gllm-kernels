#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int kQ4BlockSize = 32;

__device__ __forceinline__ uint32_t get_byte(uint32_t word, uint32_t index) {
    return (word >> (index * 8u)) & 0xFFu;
}

extern "C" __global__ void q4_0_dequantize_f32(
    const uint8_t* __restrict__ qweight,
    const half* __restrict__ scales,
    float* __restrict__ output,
    int num_blocks
) {
    __shared__ float shared_scales[8];
    __shared__ uint4 shared_qs[8];

    const int local_idx = threadIdx.x;
    const int block_base = blockIdx.x * 8;
    if (local_idx < 8) {
        const int block_idx = block_base + local_idx;
        if (block_idx < num_blocks) {
            shared_scales[local_idx] = __half2float(scales[block_idx]);
            const float4* q4_words = reinterpret_cast<const float4*>(qweight);
            const float4 packed_f = q4_words[block_idx];
            shared_qs[local_idx] = make_uint4(
                __float_as_uint(packed_f.x),
                __float_as_uint(packed_f.y),
                __float_as_uint(packed_f.z),
                __float_as_uint(packed_f.w)
            );
        } else {
            shared_scales[local_idx] = 0.0f;
            shared_qs[local_idx] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int global_idx = blockIdx.x * blockDim.x + local_idx;
    const int total_values = num_blocks * kQ4BlockSize;
    if (global_idx >= total_values) {
        return;
    }

    const int local_block = local_idx >> 5;
    const int in_block = local_idx & 31;
    const uint4 packed = shared_qs[local_block];
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    const uint32_t byte_idx = static_cast<uint32_t>(in_block >> 1);
    const uint32_t word = words[byte_idx >> 2];
    const uint32_t byte = get_byte(word, byte_idx & 3u);
    const uint32_t nibble = (in_block & 1) ? (byte >> 4) : (byte & 0x0Fu);
    const int q = static_cast<int>(nibble) - 8;
    float scale = shared_scales[local_block];
    scale = __shfl_sync(0xffffffff, scale, 0);
    output[global_idx] = static_cast<float>(q) * scale;
}

extern "C" __global__ void awq_dequantize_f32(
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const half* __restrict__ scales,
    float* __restrict__ output,
    int n,
    int k,
    int group_size,
    int groups
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_values = n * k;
    if (global_idx >= total_values) {
        return;
    }

    const int out = global_idx / k;
    const int k_idx = global_idx - out * k;
    const int group = k_idx / group_size;
    const int packed_row = out >> 3;
    const int nibble = out & 7;

    const uint32_t weight_word = qweight[packed_row * k + k_idx];
    const uint32_t zero_word = qzeros[packed_row * groups + group];
    const int w = static_cast<int>((weight_word >> (nibble * 4)) & 0x0Fu);
    const int z = static_cast<int>((zero_word >> (nibble * 4)) & 0x0Fu);
    const float scale = __half2float(scales[out * groups + group]);
    output[global_idx] = static_cast<float>(w - z) * scale;
}
