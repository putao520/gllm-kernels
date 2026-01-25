// INT2 Extreme Quantization - Metal Shaders
// Based on QuaRot, GPTQ-INT2, SqueezeLLM techniques

#include <metal_stdlib>
using namespace metal;

#define INT2_MIN 0.0f
#define INT2_MAX 3.0f

// Parameter structures
struct QuantizeParams {
    uint num_elements;
    uint group_size;
    uint num_groups;
    uint _pad0;
};

struct PackParams {
    uint num_groups;
    uint group_size;
    uint packed_size;
    uint _pad0;
};

// INT2 Quantize (F32)
// Input: values[num_elements]
// Output: quantized[num_elements], scales[num_groups], zeros[num_groups]
kernel void int2_quantize_f32(
    device const float* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device float* scales [[buffer(2)]],
    device float* zeros [[buffer(3)]],
    constant QuantizeParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint group_idx = tid;
    if (group_idx >= params.num_groups) return;
    
    uint group_start = group_idx * params.group_size;
    uint group_end = min(group_start + params.group_size, params.num_elements);
    
    // Find min/max in group
    float g_min = INFINITY;
    float g_max = -INFINITY;
    
    for (uint i = group_start; i < group_end; i++) {
        float val = input[i];
        g_min = min(g_min, val);
        g_max = max(g_max, val);
    }
    
    // Compute scale and zero point
    float range = g_max - g_min;
    float scale = (range < 1e-8f) ? 1.0f : (range / INT2_MAX);
    float zero = g_min;
    
    scales[group_idx] = scale;
    zeros[group_idx] = zero;
    
    // Quantize values
    float inv_scale = (scale < 1e-8f) ? 0.0f : (1.0f / scale);
    
    for (uint i = group_start; i < group_end; i++) {
        float val = input[i];
        float normalized = (val - zero) * inv_scale;
        float clamped = clamp(normalized, INT2_MIN, INT2_MAX);
        output[i] = uint(round(clamped));
    }
}

// INT2 Dequantize (F32)
kernel void int2_dequantize_f32(
    device const uint* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* zeros [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant QuantizeParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;
    
    uint group_idx = tid / params.group_size;
    float scale = scales[group_idx];
    float zero = zeros[group_idx];
    
    float quantized = float(input[tid]);
    output[tid] = quantized * scale + zero;
}

// INT2 Pack (F32) - Pack 4 INT2 values into single byte
kernel void int2_pack_f32(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant PackParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total_packed = params.num_groups * params.packed_size;
    if (tid >= total_packed) return;
    
    uint group_idx = tid / params.packed_size;
    uint pack_idx = tid % params.packed_size;
    uint base_idx = group_idx * params.group_size + pack_idx * 4;
    
    // Pack 4 INT2 values (MSB-first)
    uint v0 = input[base_idx] & 0x3;
    uint v1 = input[base_idx + 1] & 0x3;
    uint v2 = input[base_idx + 2] & 0x3;
    uint v3 = input[base_idx + 3] & 0x3;
    
    output[tid] = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3;
}

// INT2 Unpack (F32) - Unpack byte to 4 INT2 values
kernel void int2_unpack_f32(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant PackParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total_packed = params.num_groups * params.packed_size;
    if (tid >= total_packed) return;
    
    uint group_idx = tid / params.packed_size;
    uint pack_idx = tid % params.packed_size;
    uint packed = input[tid];
    uint base_idx = group_idx * params.group_size + pack_idx * 4;
    
    output[base_idx] = (packed >> 6) & 0x3;
    output[base_idx + 1] = (packed >> 4) & 0x3;
    output[base_idx + 2] = (packed >> 2) & 0x3;
    output[base_idx + 3] = packed & 0x3;
}

// INT2 Quantize (F16)
kernel void int2_quantize_f16(
    device const half* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device half* scales [[buffer(2)]],
    device half* zeros [[buffer(3)]],
    constant QuantizeParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint group_idx = tid;
    if (group_idx >= params.num_groups) return;
    
    uint group_start = group_idx * params.group_size;
    uint group_end = min(group_start + params.group_size, params.num_elements);
    
    float g_min = INFINITY;
    float g_max = -INFINITY;
    
    for (uint i = group_start; i < group_end; i++) {
        float val = float(input[i]);
        g_min = min(g_min, val);
        g_max = max(g_max, val);
    }
    
    float range = g_max - g_min;
    float scale = (range < 1e-8f) ? 1.0f : (range / INT2_MAX);
    float zero = g_min;
    
    scales[group_idx] = half(scale);
    zeros[group_idx] = half(zero);
    
    float inv_scale = (scale < 1e-8f) ? 0.0f : (1.0f / scale);
    
    for (uint i = group_start; i < group_end; i++) {
        float val = float(input[i]);
        float normalized = (val - zero) * inv_scale;
        float clamped = clamp(normalized, INT2_MIN, INT2_MAX);
        output[i] = uint(round(clamped));
    }
}

// INT2 Dequantize (F16)
kernel void int2_dequantize_f16(
    device const uint* input [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* zeros [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantizeParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;
    
    uint group_idx = tid / params.group_size;
    float scale = float(scales[group_idx]);
    float zero = float(zeros[group_idx]);
    
    float quantized = float(input[tid]);
    output[tid] = half(quantized * scale + zero);
}
