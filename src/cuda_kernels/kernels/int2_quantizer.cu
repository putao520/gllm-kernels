// INT2 extreme quantization CUDA kernels.
//
// This file contains CUDA kernels for INT2 quantization:
// - Group-wise INT2 quantization with scale factors
// - INT2 dequantization back to floating point
// - Efficient bit packing/unpacking
//
// Ported from int2_quantizer.hip

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>  // for FLT_MAX

// Parameters for quantization kernel
struct QuantizeParams {
    unsigned int num_elements;
    unsigned int group_size;
    unsigned int num_groups;
    unsigned int _pad;
};

// Parameters for dequantization kernel
struct DequantizeParams {
    unsigned int num_elements;
    unsigned int group_size;
    unsigned int num_groups;
    unsigned int _pad;
};

// Parameters for pack/unpack kernels
struct PackParams {
    unsigned int num_int2_values;
    unsigned int _pad[3];
};

// INT2 quantization: find group min/max, scale, and quantize to 2-bit
extern "C" __global__ void int2_quantize_f32(
    const float* __restrict__ input,
    unsigned char* __restrict__ quantized,
    float* __restrict__ scales,
    QuantizeParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;

    // Find group min/max using reduction
    __shared__ float s_min[256];
    __shared__ float s_max[256];

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = input[group_start + i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    s_min[lane_id] = local_min;
    s_max[lane_id] = local_max;
    __syncthreads();

    // Reduction to find group min/max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_min[lane_id] = fminf(s_min[lane_id], s_min[lane_id + s]);
            s_max[lane_id] = fmaxf(s_max[lane_id], s_max[lane_id + s]);
        }
        __syncthreads();
    }

    float group_min = s_min[0];
    float group_max = s_max[0];

    // Compute scale for symmetric quantization around zero
    float abs_max = fmaxf(fabsf(group_min), fabsf(group_max));
    float scale = abs_max / 1.5f;  // INT2 range: -1.5 to 1.5 (maps to -2,-1,0,1)

    if (lane_id == 0) {
        scales[group_id] = scale;
    }
    __syncthreads();

    // Quantize values to INT2 (-2, -1, 0, 1) and pack 4 per byte
    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

    unsigned int packed_per_group = params.group_size / 4;
    unsigned int packed_start = group_id * packed_per_group;

    for (unsigned int i = lane_id; i < packed_per_group; i += blockDim.x) {
        unsigned char packed = 0;
        for (int j = 0; j < 4; j++) {
            unsigned int idx = group_start + i * 4 + j;
            float val = input[idx] * inv_scale;
            // Clamp and round to INT2: -2, -1, 0, 1
            int q = __float2int_rn(val);
            q = max(-2, min(1, q));
            // Map to 0-3: -2->0, -1->1, 0->2, 1->3
            unsigned char uq = (unsigned char)(q + 2);
            packed |= (uq & 0x3) << (j * 2);
        }
        quantized[packed_start + i] = packed;
    }
}

// INT2 quantization for FP16
extern "C" __global__ void int2_quantize_f16(
    const __half* __restrict__ input,
    unsigned char* __restrict__ quantized,
    __half* __restrict__ scales,
    QuantizeParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;

    __shared__ float s_min[256];
    __shared__ float s_max[256];

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = __half2float(input[group_start + i]);
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    s_min[lane_id] = local_min;
    s_max[lane_id] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_min[lane_id] = fminf(s_min[lane_id], s_min[lane_id + s]);
            s_max[lane_id] = fmaxf(s_max[lane_id], s_max[lane_id + s]);
        }
        __syncthreads();
    }

    float group_min = s_min[0];
    float group_max = s_max[0];
    float abs_max = fmaxf(fabsf(group_min), fabsf(group_max));
    float scale = abs_max / 1.5f;

    if (lane_id == 0) {
        scales[group_id] = __float2half(scale);
    }
    __syncthreads();

    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

    unsigned int packed_per_group = params.group_size / 4;
    unsigned int packed_start = group_id * packed_per_group;

    for (unsigned int i = lane_id; i < packed_per_group; i += blockDim.x) {
        unsigned char packed = 0;
        for (int j = 0; j < 4; j++) {
            unsigned int idx = group_start + i * 4 + j;
            float val = __half2float(input[idx]) * inv_scale;
            int q = __float2int_rn(val);
            q = max(-2, min(1, q));
            unsigned char uq = (unsigned char)(q + 2);
            packed |= (uq & 0x3) << (j * 2);
        }
        quantized[packed_start + i] = packed;
    }
}

// INT2 dequantization to FP32
extern "C" __global__ void int2_dequantize_f32(
    const unsigned char* __restrict__ quantized,
    const float* __restrict__ scales,
    float* __restrict__ output,
    DequantizeParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;
    float scale = scales[group_id];

    unsigned int packed_per_group = params.group_size / 4;
    unsigned int packed_start = group_id * packed_per_group;

    for (unsigned int i = lane_id; i < packed_per_group; i += blockDim.x) {
        unsigned char packed = quantized[packed_start + i];
        for (int j = 0; j < 4; j++) {
            unsigned char uq = (packed >> (j * 2)) & 0x3;
            // Map back: 0->-2, 1->-1, 2->0, 3->1
            int q = (int)uq - 2;
            float val = (float)q * scale;
            output[group_start + i * 4 + j] = val;
        }
    }
}

// INT2 dequantization to FP16
extern "C" __global__ void int2_dequantize_f16(
    const unsigned char* __restrict__ quantized,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    DequantizeParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;
    float scale = __half2float(scales[group_id]);

    unsigned int packed_per_group = params.group_size / 4;
    unsigned int packed_start = group_id * packed_per_group;

    for (unsigned int i = lane_id; i < packed_per_group; i += blockDim.x) {
        unsigned char packed = quantized[packed_start + i];
        for (int j = 0; j < 4; j++) {
            unsigned char uq = (packed >> (j * 2)) & 0x3;
            int q = (int)uq - 2;
            float val = (float)q * scale;
            output[group_start + i * 4 + j] = __float2half(val);
        }
    }
}

// Pack INT2 values (4 values per byte)
extern "C" __global__ void int2_pack(
    const unsigned char* __restrict__ unpacked,
    unsigned char* __restrict__ packed,
    PackParams params
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int packed_idx = idx;

    if (packed_idx >= params.num_int2_values / 4) return;

    unsigned int base = packed_idx * 4;
    unsigned char result = 0;
    result |= (unpacked[base + 0] & 0x3) << 0;
    result |= (unpacked[base + 1] & 0x3) << 2;
    result |= (unpacked[base + 2] & 0x3) << 4;
    result |= (unpacked[base + 3] & 0x3) << 6;
    packed[packed_idx] = result;
}

// Unpack INT2 values (4 values per byte to individual bytes)
extern "C" __global__ void int2_unpack(
    const unsigned char* __restrict__ packed,
    unsigned char* __restrict__ unpacked,
    PackParams params
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int packed_idx = idx;

    if (packed_idx >= params.num_int2_values / 4) return;

    unsigned char p = packed[packed_idx];
    unsigned int base = packed_idx * 4;
    unpacked[base + 0] = (p >> 0) & 0x3;
    unpacked[base + 1] = (p >> 2) & 0x3;
    unpacked[base + 2] = (p >> 4) & 0x3;
    unpacked[base + 3] = (p >> 6) & 0x3;
}
