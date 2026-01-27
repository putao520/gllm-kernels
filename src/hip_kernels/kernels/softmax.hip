#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

// Softmax kernel with online (numerically stable) algorithm
// Each block handles one row of the input

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 64;  // AMD wavefront size

// Warp reduction for maximum
__device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Warp reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

extern "C" __global__ void softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rows,
    int row_size
) {
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const float* row_input = input + row_idx * row_size;
    float* row_output = output + row_idx * row_size;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Step 1: Find max across the row
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    // Warp reduce
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        local_max = (lane_id < num_warps) ? s_max[lane_id] : -FLT_MAX;
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) {
            s_max[0] = local_max;
        }
    }
    __syncthreads();
    float row_max = s_max[0];

    // Step 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_sum += expf(row_input[i] - row_max);
    }

    // Warp reduce
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        local_sum = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) {
            s_sum[0] = local_sum;
        }
    }
    __syncthreads();
    float row_sum = s_sum[0];

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - row_max) * inv_sum;
    }
}

extern "C" __global__ void softmax_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int num_rows,
    int row_size
) {
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const __half* row_input = input + row_idx * row_size;
    __half* row_output = output + row_idx * row_size;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Step 1: Find max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(row_input[i]));
    }

    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane_id < num_warps) ? s_max[lane_id] : -FLT_MAX;
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) {
            s_max[0] = local_max;
        }
    }
    __syncthreads();
    float row_max = s_max[0];

    // Step 2: Compute sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_sum += expf(__half2float(row_input[i]) - row_max);
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) {
            s_sum[0] = local_sum;
        }
    }
    __syncthreads();
    float row_sum = s_sum[0];

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float val = expf(__half2float(row_input[i]) - row_max) * inv_sum;
        row_output[i] = __float2half(val);
    }
}

// Softmax inplace variant
extern "C" __global__ void softmax_inplace_f32(
    float* __restrict__ data,
    int num_rows,
    int row_size
) {
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

    int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    float* row_data = data + row_idx * row_size;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Step 1: Find max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_data[i]);
    }

    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane_id < num_warps) ? s_max[lane_id] : -FLT_MAX;
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) s_max[0] = local_max;
    }
    __syncthreads();
    float row_max = s_max[0];

    // Step 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_sum += expf(row_data[i] - row_max);
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) s_sum[0] = local_sum;
    }
    __syncthreads();
    float row_sum = s_sum[0];

    // Step 3: Normalize in place
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_data[i] = expf(row_data[i] - row_max) * inv_sum;
    }
}
