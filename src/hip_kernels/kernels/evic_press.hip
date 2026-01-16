// EvicPress joint compression-eviction HIP kernels.
//
// This file contains HIP kernels for EvicPress KV cache management on AMD GPUs:
// - Importance score computation using attention entropy
// - Three-zone management: Hot (FP16) → Warm (INT8) → Cold (INT2)
// - Zone transitions based on importance scores
// - Compression operations for zone demotion
//
// Compile with: hipcc -c -fgpu-rdc --offload-arch=gfx90a,gfx1030,gfx1100 -o evic_press.hsaco evic_press.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Cache zone enum
enum CacheZone : unsigned int {
    HOT = 0,      // Full precision FP16
    WARM = 1,     // Compressed INT8
    COLD = 2,     // Extreme compression INT2
    EVICTED = 3   // Removed from cache
};

// Parameters for importance computation
struct ImportanceParams {
    unsigned int num_tokens;
    unsigned int num_heads;
    unsigned int head_dim;
    float temperature;
};

// Parameters for zone transition
struct ZoneTransitionParams {
    unsigned int num_tokens;
    float hot_threshold;
    float warm_threshold;
    float cold_threshold;
    unsigned int _pad;
};

// Parameters for compression
struct CompressParams {
    unsigned int num_elements;
    unsigned int group_size;
    unsigned int num_groups;
    unsigned int _pad;
};

// Compute importance scores based on attention entropy (FP32)
extern "C" __global__ void compute_importance_f32(
    const float* __restrict__ attention_probs,  // [num_heads, num_tokens, num_tokens]
    float* __restrict__ importance_scores,       // [num_tokens]
    ImportanceParams params
) {
    unsigned int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= params.num_tokens) return;

    float total_importance = 0.0f;

    // Accumulate attention received from all heads and all query positions
    for (unsigned int h = 0; h < params.num_heads; h++) {
        for (unsigned int q = 0; q < params.num_tokens; q++) {
            // attention_probs[h, q, token_id]
            unsigned int idx = h * params.num_tokens * params.num_tokens
                             + q * params.num_tokens + token_id;
            float attn = attention_probs[idx];

            // Entropy contribution: -p * log(p + eps)
            float eps = 1e-10f;
            float entropy = -attn * logf(attn + eps);
            total_importance += entropy;
        }
    }

    // Normalize by number of heads and tokens
    float norm = (float)(params.num_heads * params.num_tokens);
    importance_scores[token_id] = total_importance / norm;
}

// Compute importance scores (FP16)
extern "C" __global__ void compute_importance_f16(
    const __half* __restrict__ attention_probs,
    float* __restrict__ importance_scores,
    ImportanceParams params
) {
    unsigned int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= params.num_tokens) return;

    float total_importance = 0.0f;

    for (unsigned int h = 0; h < params.num_heads; h++) {
        for (unsigned int q = 0; q < params.num_tokens; q++) {
            unsigned int idx = h * params.num_tokens * params.num_tokens
                             + q * params.num_tokens + token_id;
            float attn = __half2float(attention_probs[idx]);
            float eps = 1e-10f;
            float entropy = -attn * logf(attn + eps);
            total_importance += entropy;
        }
    }

    float norm = (float)(params.num_heads * params.num_tokens);
    importance_scores[token_id] = total_importance / norm;
}

// Determine zone transitions based on importance scores
extern "C" __global__ void zone_transition(
    const float* __restrict__ importance_scores,
    const unsigned int* __restrict__ current_zones,
    unsigned int* __restrict__ new_zones,
    ZoneTransitionParams params
) {
    unsigned int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= params.num_tokens) return;

    float importance = importance_scores[token_id];
    unsigned int current = current_zones[token_id];
    unsigned int new_zone = current;

    // Zone demotion based on importance thresholds
    // Higher importance = stay in hotter zone
    if (importance >= params.hot_threshold) {
        new_zone = HOT;
    } else if (importance >= params.warm_threshold) {
        new_zone = (current == HOT) ? WARM : current;  // Can demote from HOT
    } else if (importance >= params.cold_threshold) {
        new_zone = (current <= WARM) ? COLD : current;  // Can demote from HOT or WARM
    } else {
        new_zone = EVICTED;  // Below all thresholds
    }

    // Never promote (one-way demotion)
    if (new_zone < current) {
        new_zone = current;
    }

    new_zones[token_id] = new_zone;
}

// Compress FP16 KV cache to INT8 (HOT → WARM)
extern "C" __global__ void compress_to_int8_f16(
    const __half* __restrict__ input,
    signed char* __restrict__ output,
    __half* __restrict__ scales,
    CompressParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;

    // Find group max absolute value
    __shared__ float s_max[256];
    float local_max = 0.0f;

    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = fabsf(__half2float(input[group_start + i]));
        local_max = fmaxf(local_max, val);
    }

    s_max[lane_id] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_max[lane_id] = fmaxf(s_max[lane_id], s_max[lane_id + s]);
        }
        __syncthreads();
    }

    float abs_max = s_max[0];
    float scale = abs_max / 127.0f;

    if (lane_id == 0) {
        scales[group_id] = __float2half(scale);
    }
    __syncthreads();

    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

    // Quantize to INT8
    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = __half2float(input[group_start + i]) * inv_scale;
        int q = __float2int_rn(val);
        q = max(-127, min(127, q));
        output[group_start + i] = (signed char)q;
    }
}

// Compress FP32 KV cache to INT8 (HOT → WARM)
extern "C" __global__ void compress_to_int8_f32(
    const float* __restrict__ input,
    signed char* __restrict__ output,
    float* __restrict__ scales,
    CompressParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;

    __shared__ float s_max[256];
    float local_max = 0.0f;

    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = fabsf(input[group_start + i]);
        local_max = fmaxf(local_max, val);
    }

    s_max[lane_id] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_max[lane_id] = fmaxf(s_max[lane_id], s_max[lane_id + s]);
        }
        __syncthreads();
    }

    float abs_max = s_max[0];
    float scale = abs_max / 127.0f;

    if (lane_id == 0) {
        scales[group_id] = scale;
    }
    __syncthreads();

    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = input[group_start + i] * inv_scale;
        int q = __float2int_rn(val);
        q = max(-127, min(127, q));
        output[group_start + i] = (signed char)q;
    }
}

// Compress INT8 KV cache to INT2 (WARM → COLD)
extern "C" __global__ void compress_to_int2_f16(
    const signed char* __restrict__ input,
    unsigned char* __restrict__ output,
    __half* __restrict__ scales,
    CompressParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;

    // Find group max for re-scaling
    __shared__ float s_max[256];
    float local_max = 0.0f;

    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = fabsf((float)input[group_start + i]);
        local_max = fmaxf(local_max, val);
    }

    s_max[lane_id] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_max[lane_id] = fmaxf(s_max[lane_id], s_max[lane_id + s]);
        }
        __syncthreads();
    }

    float abs_max = s_max[0];
    float scale = abs_max / 1.5f;  // INT2 range

    if (lane_id == 0) {
        scales[group_id] = __float2half(scale);
    }
    __syncthreads();

    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

    // Pack 4 INT2 values per byte
    unsigned int packed_per_group = params.group_size / 4;
    unsigned int packed_start = group_id * packed_per_group;

    for (unsigned int i = lane_id; i < packed_per_group; i += blockDim.x) {
        unsigned char packed = 0;
        for (int j = 0; j < 4; j++) {
            float val = (float)input[group_start + i * 4 + j] * inv_scale;
            int q = __float2int_rn(val);
            q = max(-2, min(1, q));
            unsigned char uq = (unsigned char)(q + 2);
            packed |= (uq & 0x3) << (j * 2);
        }
        output[packed_start + i] = packed;
    }
}

// Compress INT8 KV cache to INT2 (WARM → COLD) with FP32 scales
extern "C" __global__ void compress_to_int2_f32(
    const signed char* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ scales,
    CompressParams params
) {
    unsigned int group_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (group_id >= params.num_groups) return;

    unsigned int group_start = group_id * params.group_size;

    __shared__ float s_max[256];
    float local_max = 0.0f;

    for (unsigned int i = lane_id; i < params.group_size; i += blockDim.x) {
        float val = fabsf((float)input[group_start + i]);
        local_max = fmaxf(local_max, val);
    }

    s_max[lane_id] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_max[lane_id] = fmaxf(s_max[lane_id], s_max[lane_id + s]);
        }
        __syncthreads();
    }

    float abs_max = s_max[0];
    float scale = abs_max / 1.5f;

    if (lane_id == 0) {
        scales[group_id] = scale;
    }
    __syncthreads();

    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

    unsigned int packed_per_group = params.group_size / 4;
    unsigned int packed_start = group_id * packed_per_group;

    for (unsigned int i = lane_id; i < packed_per_group; i += blockDim.x) {
        unsigned char packed = 0;
        for (int j = 0; j < 4; j++) {
            float val = (float)input[group_start + i * 4 + j] * inv_scale;
            int q = __float2int_rn(val);
            q = max(-2, min(1, q));
            unsigned char uq = (unsigned char)(q + 2);
            packed |= (uq & 0x3) << (j * 2);
        }
        output[packed_start + i] = packed;
    }
}
