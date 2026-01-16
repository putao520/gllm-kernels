// Prompt Caching/CacheBlend HIP kernels.
//
// This file contains HIP kernels for prompt caching on AMD GPUs:
// - Hash computation for prompt prefix matching
// - Prefix matching and lookup
// - KV cache blending for partial matches
// - KV cache copy operations
//
// Compile with: hipcc -c -fgpu-rdc --offload-arch=gfx90a,gfx1030,gfx1100 -o prompt_cache.hsaco prompt_cache.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Parameters for hash computation
struct HashParams {
    unsigned int seq_len;
    unsigned int hidden_dim;
    unsigned int num_chunks;
    unsigned int chunk_size;
};

// Parameters for prefix matching
struct PrefixMatchParams {
    unsigned int query_len;
    unsigned int num_cached;
    unsigned int max_prefix_len;
    unsigned int _pad;
};

// Parameters for cache blending
struct BlendParams {
    unsigned int blend_start;
    unsigned int blend_len;
    unsigned int num_heads;
    unsigned int head_dim;
    float blend_factor;
    unsigned int _pad[3];
};

// Parameters for KV copy
struct CopyParams {
    unsigned int num_tokens;
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int _pad;
};

// xxHash-style mixing function
__device__ __forceinline__ unsigned long long xxhash_mix(unsigned long long h, unsigned long long val) {
    const unsigned long long PRIME1 = 11400714785074694791ULL;
    const unsigned long long PRIME2 = 14029467366897019727ULL;
    h ^= val * PRIME2;
    h = ((h << 31) | (h >> 33)) * PRIME1;
    return h;
}

// Compute hash of token embeddings for prefix matching (FP32)
extern "C" __global__ void compute_hash_f32(
    const float* __restrict__ embeddings,  // [seq_len, hidden_dim]
    unsigned long long* __restrict__ hashes,  // [num_chunks]
    HashParams params
) {
    unsigned int chunk_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (chunk_id >= params.num_chunks) return;

    unsigned int chunk_start = chunk_id * params.chunk_size;
    unsigned int chunk_end = min(chunk_start + params.chunk_size, params.seq_len);

    // Compute hash for this chunk using parallel reduction
    __shared__ unsigned long long s_hash[256];
    unsigned long long local_hash = 0xcbf29ce484222325ULL;  // FNV offset basis

    for (unsigned int t = chunk_start; t < chunk_end; t++) {
        for (unsigned int d = lane_id; d < params.hidden_dim; d += blockDim.x) {
            float val = embeddings[t * params.hidden_dim + d];
            // Convert float to bits for hashing
            unsigned int bits = __float_as_uint(val);
            local_hash = xxhash_mix(local_hash, bits);
        }
    }

    s_hash[lane_id] = local_hash;
    __syncthreads();

    // XOR reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_hash[lane_id] ^= s_hash[lane_id + s];
        }
        __syncthreads();
    }

    if (lane_id == 0) {
        hashes[chunk_id] = s_hash[0];
    }
}

// Compute hash (FP16)
extern "C" __global__ void compute_hash_f16(
    const __half* __restrict__ embeddings,
    unsigned long long* __restrict__ hashes,
    HashParams params
) {
    unsigned int chunk_id = blockIdx.x;
    unsigned int lane_id = threadIdx.x;

    if (chunk_id >= params.num_chunks) return;

    unsigned int chunk_start = chunk_id * params.chunk_size;
    unsigned int chunk_end = min(chunk_start + params.chunk_size, params.seq_len);

    __shared__ unsigned long long s_hash[256];
    unsigned long long local_hash = 0xcbf29ce484222325ULL;

    for (unsigned int t = chunk_start; t < chunk_end; t++) {
        for (unsigned int d = lane_id; d < params.hidden_dim; d += blockDim.x) {
            __half val = embeddings[t * params.hidden_dim + d];
            unsigned short bits = *reinterpret_cast<const unsigned short*>(&val);
            local_hash = xxhash_mix(local_hash, bits);
        }
    }

    s_hash[lane_id] = local_hash;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane_id < s) {
            s_hash[lane_id] ^= s_hash[lane_id + s];
        }
        __syncthreads();
    }

    if (lane_id == 0) {
        hashes[chunk_id] = s_hash[0];
    }
}

// Find longest matching prefix from cache
extern "C" __global__ void find_prefix_match(
    const unsigned long long* __restrict__ query_hashes,   // [query_chunks]
    const unsigned long long* __restrict__ cached_hashes,  // [num_cached, max_chunks]
    const unsigned int* __restrict__ cached_lengths,       // [num_cached]
    unsigned int* __restrict__ best_match_idx,             // [1]
    unsigned int* __restrict__ best_match_len,             // [1]
    PrefixMatchParams params
) {
    unsigned int cache_id = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int s_best_idx;
    __shared__ unsigned int s_best_len;

    if (threadIdx.x == 0) {
        s_best_idx = 0;
        s_best_len = 0;
    }
    __syncthreads();

    if (cache_id < params.num_cached) {
        unsigned int cached_len = cached_lengths[cache_id];
        unsigned int match_len = 0;

        // Compare hashes chunk by chunk
        unsigned int max_compare = min(params.query_len, cached_len);
        max_compare = min(max_compare, params.max_prefix_len);

        for (unsigned int c = 0; c < max_compare; c++) {
            unsigned long long q_hash = query_hashes[c];
            unsigned long long c_hash = cached_hashes[cache_id * params.max_prefix_len + c];

            if (q_hash == c_hash) {
                match_len = c + 1;
            } else {
                break;  // Prefix must be contiguous
            }
        }

        // Atomic max to find best match
        if (match_len > 0) {
            atomicMax(&s_best_len, match_len);
        }
    }
    __syncthreads();

    // Second pass to find the index with best length
    if (cache_id < params.num_cached) {
        unsigned int cached_len = cached_lengths[cache_id];
        unsigned int match_len = 0;

        unsigned int max_compare = min(params.query_len, cached_len);
        max_compare = min(max_compare, params.max_prefix_len);

        for (unsigned int c = 0; c < max_compare; c++) {
            unsigned long long q_hash = query_hashes[c];
            unsigned long long c_hash = cached_hashes[cache_id * params.max_prefix_len + c];

            if (q_hash == c_hash) {
                match_len = c + 1;
            } else {
                break;
            }
        }

        if (match_len == s_best_len && match_len > 0) {
            atomicMin(&s_best_idx, cache_id);  // Take first match
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *best_match_idx = s_best_idx;
        *best_match_len = s_best_len;
    }
}

// Blend KV cache at boundary (FP32)
// Smooth transition between cached and new KV values
extern "C" __global__ void cache_blend_f32(
    const float* __restrict__ cached_kv,   // [blend_len, num_heads, head_dim]
    const float* __restrict__ new_kv,      // [blend_len, num_heads, head_dim]
    float* __restrict__ output_kv,         // [blend_len, num_heads, head_dim]
    BlendParams params
) {
    unsigned int pos = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int dim_id = blockIdx.z * blockDim.x + threadIdx.x;

    if (pos >= params.blend_len || head_id >= params.num_heads || dim_id >= params.head_dim) return;

    unsigned int idx = pos * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;

    // Linear blend: output = (1 - alpha) * cached + alpha * new
    // alpha increases from 0 to 1 across blend region
    float alpha = (float)(pos + 1) / (float)(params.blend_len + 1);
    alpha *= params.blend_factor;  // Scale by blend factor

    float cached_val = cached_kv[idx];
    float new_val = new_kv[idx];

    output_kv[idx] = (1.0f - alpha) * cached_val + alpha * new_val;
}

// Blend KV cache (FP16)
extern "C" __global__ void cache_blend_f16(
    const __half* __restrict__ cached_kv,
    const __half* __restrict__ new_kv,
    __half* __restrict__ output_kv,
    BlendParams params
) {
    unsigned int pos = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int dim_id = blockIdx.z * blockDim.x + threadIdx.x;

    if (pos >= params.blend_len || head_id >= params.num_heads || dim_id >= params.head_dim) return;

    unsigned int idx = pos * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;

    float alpha = (float)(pos + 1) / (float)(params.blend_len + 1);
    alpha *= params.blend_factor;

    float cached_val = __half2float(cached_kv[idx]);
    float new_val = __half2float(new_kv[idx]);

    float result = (1.0f - alpha) * cached_val + alpha * new_val;
    output_kv[idx] = __float2half(result);
}

// Copy KV cache from source to destination (FP32)
extern "C" __global__ void copy_kv_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    CopyParams params
) {
    unsigned int token_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int dim_id = blockIdx.z * blockDim.x + threadIdx.x;

    if (token_id >= params.num_tokens || head_id >= params.num_heads || dim_id >= params.head_dim) return;

    unsigned int idx = token_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
    dst[idx] = src[idx];
}

// Copy KV cache (FP16)
extern "C" __global__ void copy_kv_f16(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    CopyParams params
) {
    unsigned int token_id = blockIdx.x;
    unsigned int head_id = blockIdx.y;
    unsigned int dim_id = blockIdx.z * blockDim.x + threadIdx.x;

    if (token_id >= params.num_tokens || head_id >= params.num_heads || dim_id >= params.head_dim) return;

    unsigned int idx = token_id * params.num_heads * params.head_dim + head_id * params.head_dim + dim_id;
    dst[idx] = src[idx];
}
