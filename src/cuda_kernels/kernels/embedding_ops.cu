/**
 * Embedding quantization and similarity kernels for vector search.
 *
 * Operations:
 * - Binary IP Hamming: XOR + POPCNT for 1-bit quantized vectors
 * - Int8 Dot Product: Packed i8x4 multiply-accumulate
 * - Int4 Packed Dot Product: Nibble-packed quantization
 * - Matryoshka Dimension Truncation
 */

#define BLOCK_SIZE 256

// ============================================================================
// Binary Inner Product (Hamming Distance)
// ============================================================================

/**
 * Binary IP Hamming: Compute Hamming distance between binary-quantized vectors.
 * Lower score = more similar.
 *
 * @param queries [num_queries, dim/32] as packed u32
 * @param database [num_vectors, dim/32] as packed u32
 * @param scores [num_queries, num_vectors] as i32 (Hamming distance)
 * @param dim Original dimension (must be multiple of 32)
 * @param num_queries Number of query vectors
 * @param num_vectors Number of database vectors
 */
extern "C" __global__ void binary_ip_hamming(
    const unsigned int* __restrict__ queries,
    const unsigned int* __restrict__ database,
    int* __restrict__ scores,
    int dim,
    int num_queries,
    int num_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_queries * num_vectors;

    if (idx >= total_pairs) return;

    int q_idx = idx / num_vectors;
    int v_idx = idx % num_vectors;
    int packed_dim = dim / 32;

    int q_offset = q_idx * packed_dim;
    int v_offset = v_idx * packed_dim;

    int hamming_dist = 0;

    // XOR + POPCNT for each packed u32
    for (int i = 0; i < packed_dim; i++) {
        unsigned int xor_result = queries[q_offset + i] ^ database[v_offset + i];
        hamming_dist += __popc(xor_result);
    }

    scores[idx] = hamming_dist;
}

/**
 * Asymmetric Binary IP: f32 query vs binary database.
 * Higher score = more similar.
 *
 * @param queries [num_queries, dim] as f32
 * @param database [num_vectors, dim/32] as packed u32
 * @param scores [num_queries, num_vectors] as f32
 */
extern "C" __global__ void binary_ip_asymmetric(
    const float* __restrict__ queries,
    const unsigned int* __restrict__ database,
    float* __restrict__ scores,
    int dim,
    int num_queries,
    int num_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_queries * num_vectors;

    if (idx >= total_pairs) return;

    int q_idx = idx / num_vectors;
    int v_idx = idx % num_vectors;
    int packed_dim = dim / 32;

    int q_offset = q_idx * dim;
    int v_offset = v_idx * packed_dim;

    float score = 0.0f;

    // For each packed u32 in database
    for (int i = 0; i < packed_dim; i++) {
        unsigned int packed = database[v_offset + i];

        // Unpack 32 bits and compute dot product with f32 query
        #pragma unroll 32
        for (int bit = 0; bit < 32; bit++) {
            int dim_idx = i * 32 + bit;
            if (dim_idx < dim) {
                float sign = ((packed >> bit) & 1u) ? 1.0f : -1.0f;
                score += queries[q_offset + dim_idx] * sign;
            }
        }
    }

    scores[idx] = score;
}

// ============================================================================
// Int8 Dot Product
// ============================================================================

/**
 * Unpack signed int8 from packed u32.
 */
__device__ __forceinline__ int unpack_i8(unsigned int packed, int byte_idx) {
    unsigned int byte_val = (packed >> (byte_idx * 8)) & 0xFFu;
    // Sign-extend from 8-bit to 32-bit
    if ((byte_val & 0x80u) != 0) {
        return (int)(byte_val | 0xFFFFFF00u);
    }
    return (int)byte_val;
}

/**
 * Int8 Dot Product: Compute dot product of int8-quantized vectors.
 * Packed as i8x4 in u32 for efficient memory access.
 *
 * @param queries [num_queries, dim/4] as packed i8x4 (u32)
 * @param database [num_vectors, dim/4] as packed i8x4 (u32)
 * @param scores [num_queries, num_vectors] as f32
 * @param dim Original dimension (must be multiple of 4)
 * @param scale Quantization scale factor
 */
extern "C" __global__ void int8_dot_product(
    const unsigned int* __restrict__ queries,
    const unsigned int* __restrict__ database,
    float* __restrict__ scores,
    int dim,
    int num_queries,
    int num_vectors,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_queries * num_vectors;

    if (idx >= total_pairs) return;

    int q_idx = idx / num_vectors;
    int v_idx = idx % num_vectors;
    int packed_dim = dim / 4;

    int q_offset = q_idx * packed_dim;
    int v_offset = v_idx * packed_dim;

    int acc = 0;

    // Process 4 int8 values at a time
    for (int i = 0; i < packed_dim; i++) {
        unsigned int q_packed = queries[q_offset + i];
        unsigned int v_packed = database[v_offset + i];

        // Unpack i8x4 and compute dot product
        int q0 = unpack_i8(q_packed, 0);
        int q1 = unpack_i8(q_packed, 1);
        int q2 = unpack_i8(q_packed, 2);
        int q3 = unpack_i8(q_packed, 3);

        int v0 = unpack_i8(v_packed, 0);
        int v1 = unpack_i8(v_packed, 1);
        int v2 = unpack_i8(v_packed, 2);
        int v3 = unpack_i8(v_packed, 3);

        acc += q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3;
    }

    // Scale back to float
    scores[idx] = (float)acc * scale * scale;
}

// ============================================================================
// Int4 Packed Dot Product
// ============================================================================

/**
 * Unpack signed int4 from packed u32 (nibble index 0-7).
 */
__device__ __forceinline__ int unpack_i4(unsigned int packed, int nibble_idx) {
    unsigned int nibble = (packed >> (nibble_idx * 4)) & 0xFu;
    // Sign-extend from 4-bit to 32-bit (-8 to 7)
    if ((nibble & 0x8u) != 0) {
        return (int)(nibble | 0xFFFFFFF0u);
    }
    return (int)nibble;
}

/**
 * Int4 Packed Dot Product: 2 values per byte, 8 values per u32.
 *
 * @param queries [num_queries, dim/8] as packed i4x8 (u32)
 * @param database [num_vectors, dim/8] as packed i4x8 (u32)
 * @param scores [num_queries, num_vectors] as f32
 * @param dim Original dimension (must be multiple of 8)
 * @param scale Quantization scale factor
 * @param zero_point Quantization zero point
 */
extern "C" __global__ void int4_dot_product(
    const unsigned int* __restrict__ queries,
    const unsigned int* __restrict__ database,
    float* __restrict__ scores,
    int dim,
    int num_queries,
    int num_vectors,
    float scale,
    int zero_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_queries * num_vectors;

    if (idx >= total_pairs) return;

    int q_idx = idx / num_vectors;
    int v_idx = idx % num_vectors;
    int packed_dim = dim / 8;

    int q_offset = q_idx * packed_dim;
    int v_offset = v_idx * packed_dim;

    int acc = 0;

    // Process 8 int4 values at a time (packed in u32)
    for (int i = 0; i < packed_dim; i++) {
        unsigned int q_packed = queries[q_offset + i];
        unsigned int v_packed = database[v_offset + i];

        // Unpack 8 nibbles and compute dot product
        #pragma unroll 8
        for (int nibble = 0; nibble < 8; nibble++) {
            int q_val = unpack_i4(q_packed, nibble) - zero_point;
            int v_val = unpack_i4(v_packed, nibble) - zero_point;
            acc += q_val * v_val;
        }
    }

    // Scale back to float
    scores[idx] = (float)acc * scale * scale;
}

// ============================================================================
// Matryoshka Dimension Truncation
// ============================================================================

/**
 * Truncate embeddings to target dimension with optional L2 normalization.
 *
 * @param input [num_vectors, full_dim] as f32
 * @param output [num_vectors, target_dim] as f32
 * @param full_dim Full embedding dimension
 * @param target_dim Target dimension (must be <= full_dim)
 * @param num_vectors Number of vectors
 */
extern "C" __global__ void matryoshka_truncate(
    const float* __restrict__ input,
    float* __restrict__ output,
    int full_dim,
    int target_dim,
    int num_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_vectors * target_dim;

    if (idx >= total_elements) return;

    int vec_idx = idx / target_dim;
    int dim_idx = idx % target_dim;

    int in_offset = vec_idx * full_dim + dim_idx;
    int out_offset = vec_idx * target_dim + dim_idx;

    output[out_offset] = input[in_offset];
}

/**
 * L2 normalize truncated vectors (separate pass for efficiency).
 *
 * @param vectors [num_vectors, target_dim] as f32 (in-place)
 * @param target_dim Dimension of each vector
 * @param num_vectors Number of vectors
 */
extern "C" __global__ void matryoshka_normalize(
    float* __restrict__ vectors,
    int target_dim,
    int num_vectors
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx >= num_vectors) return;

    int offset = vec_idx * target_dim;

    // Compute L2 norm
    float norm_sq = 0.0f;
    for (int i = 0; i < target_dim; i++) {
        float val = vectors[offset + i];
        norm_sq += val * val;
    }

    float norm = sqrtf(norm_sq);
    if (norm > 0.0f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < target_dim; i++) {
            vectors[offset + i] *= inv_norm;
        }
    }
}
