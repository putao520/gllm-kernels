// EvicPress Joint Compression and Eviction - Metal Shaders
// Based on KVPress (EMNLP'24): Three-zone progressive KV cache management

#include <metal_stdlib>
using namespace metal;

#define INT8_MIN -128.0f
#define INT8_MAX 127.0f
#define INT2_MIN 0.0f
#define INT2_MAX 3.0f

// Parameter structures
struct ImportanceParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    float attention_weight;
    float semantic_weight;
    float recency_weight;
    uint _pad0;
};

struct ZoneTransitionParams {
    uint num_elements;
    uint group_size;
    uint num_groups;
    uint _pad0;
};

// Compute importance scores (F32)
kernel void evicpress_compute_importance_f32(
    device const float* attention [[buffer(0)]],
    device const uint* positions [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ImportanceParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len;
    if (tid >= total) return;
    
    uint batch_idx = tid / params.seq_len;
    uint seq_idx = tid % params.seq_len;
    
    // Average attention score across heads
    float attention_sum = 0.0f;
    uint head_stride = params.seq_len;
    uint batch_stride = params.num_heads * head_stride;
    
    for (uint h = 0; h < params.num_heads; h++) {
        uint attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        attention_sum += attention[attn_idx];
    }
    float attention_score = attention_sum / float(params.num_heads);
    
    // Position-based recency
    uint position = positions[tid];
    float recency_score = float(position) / float(params.seq_len);
    
    // Attention variance as semantic proxy
    float attn_var = 0.0f;
    for (uint h = 0; h < params.num_heads; h++) {
        uint attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        float diff = attention[attn_idx] - attention_score;
        attn_var += diff * diff;
    }
    float semantic_score = 1.0f - sqrt(attn_var / float(params.num_heads));
    
    // Combined importance
    float combined = params.attention_weight * attention_score +
                     params.semantic_weight * semantic_score +
                     params.recency_weight * recency_score;
    
    output[tid] = combined;
}

// FP16 -> INT8 Quantization (Hot -> Warm zone)
kernel void evicpress_quantize_fp16_to_int8_f32(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant ZoneTransitionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint group_idx = tid;
    if (group_idx >= params.num_groups) return;
    
    uint group_start = group_idx * params.group_size;
    uint group_end = min(group_start + params.group_size, params.num_elements);
    
    // Find abs max
    float abs_max = 0.0f;
    for (uint i = group_start; i < group_end; i++) {
        abs_max = max(abs_max, abs(input[i]));
    }
    
    // Symmetric quantization
    float scale = (abs_max < 1e-8f) ? 1.0f : (abs_max / 127.0f);
    scales[group_idx] = scale;
    
    float inv_scale = (scale < 1e-8f) ? 0.0f : (1.0f / scale);
    
    for (uint i = group_start; i < group_end; i++) {
        float val = input[i] * inv_scale;
        val = clamp(val, INT8_MIN, INT8_MAX);
        output[i] = int(round(val));
    }
}

// INT8 -> INT2 Quantization (Warm -> Cold zone)
kernel void evicpress_quantize_int8_to_int2_f32(
    device const int* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device float* scales [[buffer(2)]],
    device const float* warm_scales [[buffer(3)]],
    constant ZoneTransitionParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint group_idx = tid;
    if (group_idx >= params.num_groups) return;
    
    uint group_start = group_idx * params.group_size;
    uint group_end = min(group_start + params.group_size, params.num_elements);
    
    // Find min/max
    float g_min = 127.0f;
    float g_max = -128.0f;
    
    for (uint i = group_start; i < group_end; i++) {
        float val = float(input[i]);
        g_min = min(g_min, val);
        g_max = max(g_max, val);
    }
    
    float range = g_max - g_min;
    float scale = (range < 1e-8f) ? 1.0f : (range / INT2_MAX);
    scales[group_idx] = warm_scales[group_idx] * scale;
    
    for (uint i = group_start; i < group_end; i++) {
        float val = float(input[i]);
        float normalized = (val + 128.0f) / 255.0f * 3.0f;
        normalized = clamp(normalized, INT2_MIN, INT2_MAX);
        output[i] = uint(round(normalized));
    }
}

// INT8 Dequantization (Warm zone read)
kernel void evicpress_dequantize_int8_f32(
    device const int* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ZoneTransitionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;
    
    uint group_idx = tid / params.group_size;
    float scale = scales[group_idx];
    
    output[tid] = float(input[tid]) * scale;
}

// INT2 Dequantization (Cold zone read)
kernel void evicpress_dequantize_int2_f32(
    device const uint* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ZoneTransitionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;
    
    uint group_idx = tid / params.group_size;
    float scale = scales[group_idx];
    
    // Map 0..3 back to -128..127
    float quantized = float(input[tid]);
    float denormalized = (quantized / 3.0f * 255.0f) - 128.0f;
    output[tid] = denormalized * scale;
}

// F16 variants
kernel void evicpress_compute_importance_f16(
    device const half* attention [[buffer(0)]],
    device const uint* positions [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant ImportanceParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len;
    if (tid >= total) return;
    
    uint batch_idx = tid / params.seq_len;
    uint seq_idx = tid % params.seq_len;
    
    float attention_sum = 0.0f;
    uint head_stride = params.seq_len;
    uint batch_stride = params.num_heads * head_stride;
    
    for (uint h = 0; h < params.num_heads; h++) {
        uint attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        attention_sum += float(attention[attn_idx]);
    }
    float attention_score = attention_sum / float(params.num_heads);
    
    uint position = positions[tid];
    float recency_score = float(position) / float(params.seq_len);
    
    float attn_var = 0.0f;
    for (uint h = 0; h < params.num_heads; h++) {
        uint attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        float diff = float(attention[attn_idx]) - attention_score;
        attn_var += diff * diff;
    }
    float semantic_score = 1.0f - sqrt(attn_var / float(params.num_heads));
    
    float combined = params.attention_weight * attention_score +
                     params.semantic_weight * semantic_score +
                     params.recency_weight * recency_score;
    
    output[tid] = half(combined);
}

kernel void evicpress_quantize_fp16_to_int8_f16(
    device const half* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    device half* scales [[buffer(2)]],
    constant ZoneTransitionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint group_idx = tid;
    if (group_idx >= params.num_groups) return;
    
    uint group_start = group_idx * params.group_size;
    uint group_end = min(group_start + params.group_size, params.num_elements);
    
    float abs_max = 0.0f;
    for (uint i = group_start; i < group_end; i++) {
        abs_max = max(abs_max, abs(float(input[i])));
    }
    
    float scale = (abs_max < 1e-8f) ? 1.0f : (abs_max / 127.0f);
    scales[group_idx] = half(scale);
    
    float inv_scale = (scale < 1e-8f) ? 0.0f : (1.0f / scale);
    
    for (uint i = group_start; i < group_end; i++) {
        float val = float(input[i]) * inv_scale;
        val = clamp(val, INT8_MIN, INT8_MAX);
        output[i] = int(round(val));
    }
}
