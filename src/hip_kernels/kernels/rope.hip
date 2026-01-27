#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RoPE (Rotary Position Embedding) kernels
// Applies rotary position embedding to Q and K tensors

constexpr int BLOCK_SIZE = 256;

// RoPE apply kernel for f32
// q, k: [batch, seq, num_heads, head_dim]
// cos_cache, sin_cache: [max_seq_len, head_dim/2]
extern "C" __global__ void rope_apply_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset
) {
    int half_head_dim = head_dim / 2;
    int total_q_elements = batch_size * seq_len * num_q_heads * half_head_dim;
    int total_kv_elements = batch_size * seq_len * num_kv_heads * half_head_dim;

    // Process Q
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_q_elements;
         idx += gridDim.x * blockDim.x) {

        int pair_idx = idx % half_head_dim;
        int head_idx = (idx / half_head_dim) % num_q_heads;
        int seq_idx = (idx / (half_head_dim * num_q_heads)) % seq_len;
        int batch_idx = idx / (half_head_dim * num_q_heads * seq_len);

        int pos = position_offset + seq_idx;
        float cos_val = cos_cache[pos * half_head_dim + pair_idx];
        float sin_val = sin_cache[pos * half_head_dim + pair_idx];

        int base_offset = batch_idx * seq_len * num_q_heads * head_dim +
                         seq_idx * num_q_heads * head_dim +
                         head_idx * head_dim;

        float q0 = q[base_offset + pair_idx];
        float q1 = q[base_offset + half_head_dim + pair_idx];

        q_out[base_offset + pair_idx] = q0 * cos_val - q1 * sin_val;
        q_out[base_offset + half_head_dim + pair_idx] = q0 * sin_val + q1 * cos_val;
    }

    // Process K
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_kv_elements;
         idx += gridDim.x * blockDim.x) {

        int pair_idx = idx % half_head_dim;
        int head_idx = (idx / half_head_dim) % num_kv_heads;
        int seq_idx = (idx / (half_head_dim * num_kv_heads)) % seq_len;
        int batch_idx = idx / (half_head_dim * num_kv_heads * seq_len);

        int pos = position_offset + seq_idx;
        float cos_val = cos_cache[pos * half_head_dim + pair_idx];
        float sin_val = sin_cache[pos * half_head_dim + pair_idx];

        int base_offset = batch_idx * seq_len * num_kv_heads * head_dim +
                         seq_idx * num_kv_heads * head_dim +
                         head_idx * head_dim;

        float k0 = k[base_offset + pair_idx];
        float k1 = k[base_offset + half_head_dim + pair_idx];

        k_out[base_offset + pair_idx] = k0 * cos_val - k1 * sin_val;
        k_out[base_offset + half_head_dim + pair_idx] = k0 * sin_val + k1 * cos_val;
    }
}

// RoPE apply kernel for f16
extern "C" __global__ void rope_apply_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset
) {
    int half_head_dim = head_dim / 2;
    int total_q_elements = batch_size * seq_len * num_q_heads * half_head_dim;
    int total_kv_elements = batch_size * seq_len * num_kv_heads * half_head_dim;

    // Process Q
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_q_elements;
         idx += gridDim.x * blockDim.x) {

        int pair_idx = idx % half_head_dim;
        int head_idx = (idx / half_head_dim) % num_q_heads;
        int seq_idx = (idx / (half_head_dim * num_q_heads)) % seq_len;
        int batch_idx = idx / (half_head_dim * num_q_heads * seq_len);

        int pos = position_offset + seq_idx;
        float cos_val = cos_cache[pos * half_head_dim + pair_idx];
        float sin_val = sin_cache[pos * half_head_dim + pair_idx];

        int base_offset = batch_idx * seq_len * num_q_heads * head_dim +
                         seq_idx * num_q_heads * head_dim +
                         head_idx * head_dim;

        float q0 = __half2float(q[base_offset + pair_idx]);
        float q1 = __half2float(q[base_offset + half_head_dim + pair_idx]);

        q_out[base_offset + pair_idx] = __float2half(q0 * cos_val - q1 * sin_val);
        q_out[base_offset + half_head_dim + pair_idx] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    // Process K
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_kv_elements;
         idx += gridDim.x * blockDim.x) {

        int pair_idx = idx % half_head_dim;
        int head_idx = (idx / half_head_dim) % num_kv_heads;
        int seq_idx = (idx / (half_head_dim * num_kv_heads)) % seq_len;
        int batch_idx = idx / (half_head_dim * num_kv_heads * seq_len);

        int pos = position_offset + seq_idx;
        float cos_val = cos_cache[pos * half_head_dim + pair_idx];
        float sin_val = sin_cache[pos * half_head_dim + pair_idx];

        int base_offset = batch_idx * seq_len * num_kv_heads * head_dim +
                         seq_idx * num_kv_heads * head_dim +
                         head_idx * head_dim;

        float k0 = __half2float(k[base_offset + pair_idx]);
        float k1 = __half2float(k[base_offset + half_head_dim + pair_idx]);

        k_out[base_offset + pair_idx] = __float2half(k0 * cos_val - k1 * sin_val);
        k_out[base_offset + half_head_dim + pair_idx] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

// RoPE inplace kernel for f32
// x: [batch, seq, num_heads, head_dim]
extern "C" __global__ void rope_apply_inplace_f32(
    float* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int position_offset
) {
    int half_head_dim = head_dim / 2;
    int total_elements = batch_size * seq_len * num_heads * half_head_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {

        int pair_idx = idx % half_head_dim;
        int head_idx = (idx / half_head_dim) % num_heads;
        int seq_idx = (idx / (half_head_dim * num_heads)) % seq_len;
        int batch_idx = idx / (half_head_dim * num_heads * seq_len);

        int pos = position_offset + seq_idx;
        float cos_val = cos_cache[pos * half_head_dim + pair_idx];
        float sin_val = sin_cache[pos * half_head_dim + pair_idx];

        int base_offset = batch_idx * seq_len * num_heads * head_dim +
                         seq_idx * num_heads * head_dim +
                         head_idx * head_dim;

        float x0 = x[base_offset + pair_idx];
        float x1 = x[base_offset + half_head_dim + pair_idx];

        x[base_offset + pair_idx] = x0 * cos_val - x1 * sin_val;
        x[base_offset + half_head_dim + pair_idx] = x0 * sin_val + x1 * cos_val;
    }
}

// RoPE inplace kernel for f16
extern "C" __global__ void rope_apply_inplace_f16(
    __half* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int position_offset
) {
    int half_head_dim = head_dim / 2;
    int total_elements = batch_size * seq_len * num_heads * half_head_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {

        int pair_idx = idx % half_head_dim;
        int head_idx = (idx / half_head_dim) % num_heads;
        int seq_idx = (idx / (half_head_dim * num_heads)) % seq_len;
        int batch_idx = idx / (half_head_dim * num_heads * seq_len);

        int pos = position_offset + seq_idx;
        float cos_val = cos_cache[pos * half_head_dim + pair_idx];
        float sin_val = sin_cache[pos * half_head_dim + pair_idx];

        int base_offset = batch_idx * seq_len * num_heads * head_dim +
                         seq_idx * num_heads * head_dim +
                         head_idx * head_dim;

        float x0 = __half2float(x[base_offset + pair_idx]);
        float x1 = __half2float(x[base_offset + half_head_dim + pair_idx]);

        x[base_offset + pair_idx] = __float2half(x0 * cos_val - x1 * sin_val);
        x[base_offset + half_head_dim + pair_idx] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}
