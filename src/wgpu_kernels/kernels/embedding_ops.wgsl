//! Embedding quantization and similarity kernels for vector search.
//!
//! Operations:
//! - Binary IP Hamming: XOR + POPCNT for 1-bit quantized vectors
//! - Int8 Dot Product: Packed i8x4 multiply-accumulate
//! - Int4 Packed Dot Product: Nibble-packed quantization
//!
//! Note: WGSL uses u32 for binary (no native u64), i32 for int ops.

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Binary Inner Product (Hamming Distance)
// ============================================================================

struct BinaryIpParams {
    dim: u32,           // Original dimension (must be multiple of 32)
    num_queries: u32,
    num_vectors: u32,
    _pad: u32,
};

// Module-scope bindings for binary_ip_hamming
@group(0) @binding(0) var<storage, read> binary_queries: array<u32>;
@group(0) @binding(1) var<storage, read> binary_database: array<u32>;
@group(0) @binding(2) var<storage, read_write> binary_scores_i32: array<i32>;
@group(0) @binding(3) var<uniform> binary_params: BinaryIpParams;

/// Binary IP Hamming: Compute Hamming distance between binary-quantized vectors.
/// Lower score = more similar.
///
/// Input: queries [num_queries, dim/32] as packed u32
/// Input: database [num_vectors, dim/32] as packed u32
/// Output: scores [num_queries, num_vectors] as i32 (Hamming distance)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn binary_ip_hamming(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_pairs = binary_params.num_queries * binary_params.num_vectors;

    if (idx >= total_pairs) {
        return;
    }

    let q_idx = idx / binary_params.num_vectors;
    let v_idx = idx % binary_params.num_vectors;
    let packed_dim = binary_params.dim / 32u;

    let q_offset = q_idx * packed_dim;
    let v_offset = v_idx * packed_dim;

    var hamming_dist: i32 = 0;

    // XOR + POPCNT for each packed u32
    for (var i: u32 = 0u; i < packed_dim; i = i + 1u) {
        let xor_result = binary_queries[q_offset + i] ^ binary_database[v_offset + i];
        hamming_dist = hamming_dist + i32(countOneBits(xor_result));
    }

    binary_scores_i32[idx] = hamming_dist;
}

// ============================================================================
// Binary IP Asymmetric (f32 query vs binary database)
// ============================================================================

// Module-scope bindings for binary_ip_asymmetric
@group(0) @binding(0) var<storage, read> asym_queries: array<f32>;
@group(0) @binding(1) var<storage, read> asym_database: array<u32>;
@group(0) @binding(2) var<storage, read_write> asym_scores: array<f32>;
@group(0) @binding(3) var<uniform> asym_params: BinaryIpParams;

/// Asymmetric Binary IP: f32 query vs binary database.
/// Higher score = more similar.
///
/// Input: queries [num_queries, dim] as f32
/// Input: database [num_vectors, dim/32] as packed u32
/// Output: scores [num_queries, num_vectors] as f32
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn binary_ip_asymmetric(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_pairs = asym_params.num_queries * asym_params.num_vectors;

    if (idx >= total_pairs) {
        return;
    }

    let q_idx = idx / asym_params.num_vectors;
    let v_idx = idx % asym_params.num_vectors;
    let packed_dim = asym_params.dim / 32u;

    let q_offset = q_idx * asym_params.dim;
    let v_offset = v_idx * packed_dim;

    var score: f32 = 0.0;

    // For each packed u32 in database
    for (var i: u32 = 0u; i < packed_dim; i = i + 1u) {
        let packed = asym_database[v_offset + i];

        // Unpack 32 bits and compute dot product with f32 query
        for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
            let dim_idx = i * 32u + bit;
            if (dim_idx < asym_params.dim) {
                let sign = select(-1.0, 1.0, ((packed >> bit) & 1u) == 1u);
                score = score + asym_queries[q_offset + dim_idx] * sign;
            }
        }
    }

    asym_scores[idx] = score;
}

// ============================================================================
// Int8 Dot Product
// ============================================================================

struct Int8DotParams {
    dim: u32,           // Original dimension (must be multiple of 4)
    num_queries: u32,
    num_vectors: u32,
    scale: f32,         // Quantization scale factor
};

// Module-scope bindings for int8_dot_product
@group(0) @binding(0) var<storage, read> int8_queries: array<u32>;
@group(0) @binding(1) var<storage, read> int8_database: array<u32>;
@group(0) @binding(2) var<storage, read_write> int8_scores: array<f32>;
@group(0) @binding(3) var<uniform> int8_params: Int8DotParams;

/// Unpack signed int8 from packed u32.
fn unpack_i8(packed: u32, byte_idx: u32) -> i32 {
    let byte = (packed >> (byte_idx * 8u)) & 0xFFu;
    // Sign-extend from 8-bit to 32-bit
    if ((byte & 0x80u) != 0u) {
        return i32(byte | 0xFFFFFF00u);
    }
    return i32(byte);
}

/// Int8 Dot Product: Compute dot product of int8-quantized vectors.
/// Packed as i8x4 in u32 for efficient memory access.
///
/// Input: queries [num_queries, dim/4] as packed i8x4 (u32)
/// Input: database [num_vectors, dim/4] as packed i8x4 (u32)
/// Output: scores [num_queries, num_vectors] as f32
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int8_dot_product(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_pairs = int8_params.num_queries * int8_params.num_vectors;

    if (idx >= total_pairs) {
        return;
    }

    let q_idx = idx / int8_params.num_vectors;
    let v_idx = idx % int8_params.num_vectors;
    let packed_dim = int8_params.dim / 4u;

    let q_offset = q_idx * packed_dim;
    let v_offset = v_idx * packed_dim;

    var acc: i32 = 0;

    // Process 4 int8 values at a time
    for (var i: u32 = 0u; i < packed_dim; i = i + 1u) {
        let q_packed = int8_queries[q_offset + i];
        let v_packed = int8_database[v_offset + i];

        // Unpack i8x4 and compute dot product
        // Each byte is a signed int8 (-128 to 127)
        let q0 = unpack_i8(q_packed, 0u);
        let q1 = unpack_i8(q_packed, 1u);
        let q2 = unpack_i8(q_packed, 2u);
        let q3 = unpack_i8(q_packed, 3u);

        let v0 = unpack_i8(v_packed, 0u);
        let v1 = unpack_i8(v_packed, 1u);
        let v2 = unpack_i8(v_packed, 2u);
        let v3 = unpack_i8(v_packed, 3u);

        acc = acc + q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3;
    }

    // Scale back to float
    int8_scores[idx] = f32(acc) * int8_params.scale * int8_params.scale;
}

// ============================================================================
// Int4 Packed Dot Product
// ============================================================================

struct Int4DotParams {
    dim: u32,           // Original dimension (must be multiple of 8)
    num_queries: u32,
    num_vectors: u32,
    scale: f32,         // Quantization scale factor
    zero_point: i32,    // Quantization zero point
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Module-scope bindings for int4_dot_product
@group(0) @binding(0) var<storage, read> int4_queries: array<u32>;
@group(0) @binding(1) var<storage, read> int4_database: array<u32>;
@group(0) @binding(2) var<storage, read_write> int4_scores: array<f32>;
@group(0) @binding(3) var<uniform> int4_params: Int4DotParams;

/// Unpack signed int4 from packed u32 (nibble index 0-7).
fn unpack_i4(packed: u32, nibble_idx: u32) -> i32 {
    let nibble = (packed >> (nibble_idx * 4u)) & 0xFu;
    // Sign-extend from 4-bit to 32-bit (-8 to 7)
    if ((nibble & 0x8u) != 0u) {
        return i32(nibble | 0xFFFFFFF0u);
    }
    return i32(nibble);
}

/// Int4 Packed Dot Product: 2 values per byte, 8 values per u32.
///
/// Input: queries [num_queries, dim/8] as packed i4x8 (u32)
/// Input: database [num_vectors, dim/8] as packed i4x8 (u32)
/// Output: scores [num_queries, num_vectors] as f32
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int4_dot_product(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_pairs = int4_params.num_queries * int4_params.num_vectors;

    if (idx >= total_pairs) {
        return;
    }

    let q_idx = idx / int4_params.num_vectors;
    let v_idx = idx % int4_params.num_vectors;
    let packed_dim = int4_params.dim / 8u;

    let q_offset = q_idx * packed_dim;
    let v_offset = v_idx * packed_dim;

    var acc: i32 = 0;

    // Process 8 int4 values at a time (packed in u32)
    for (var i: u32 = 0u; i < packed_dim; i = i + 1u) {
        let q_packed = int4_queries[q_offset + i];
        let v_packed = int4_database[v_offset + i];

        // Unpack 8 nibbles and compute dot product
        for (var nibble: u32 = 0u; nibble < 8u; nibble = nibble + 1u) {
            let q_val = unpack_i4(q_packed, nibble) - int4_params.zero_point;
            let v_val = unpack_i4(v_packed, nibble) - int4_params.zero_point;
            acc = acc + q_val * v_val;
        }
    }

    // Scale back to float
    int4_scores[idx] = f32(acc) * int4_params.scale * int4_params.scale;
}

// ============================================================================
// Matryoshka Dimension Truncation
// ============================================================================

struct MatryoshkaParams {
    full_dim: u32,
    target_dim: u32,
    num_vectors: u32,
    normalize: u32,     // 0 = no, 1 = yes
};

// Module-scope bindings for matryoshka_truncate
@group(0) @binding(0) var<storage, read> mat_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> mat_output: array<f32>;
@group(0) @binding(2) var<uniform> mat_params: MatryoshkaParams;

/// Truncate embeddings to target dimension with optional L2 normalization.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn matryoshka_truncate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_elements = mat_params.num_vectors * mat_params.target_dim;

    if (idx >= total_elements) {
        return;
    }

    let vec_idx = idx / mat_params.target_dim;
    let dim_idx = idx % mat_params.target_dim;

    let in_offset = vec_idx * mat_params.full_dim + dim_idx;
    let out_offset = vec_idx * mat_params.target_dim + dim_idx;

    mat_output[out_offset] = mat_input[in_offset];
}

// Module-scope bindings for matryoshka_normalize
@group(0) @binding(0) var<storage, read_write> norm_vectors: array<f32>;
@group(0) @binding(1) var<uniform> norm_params: MatryoshkaParams;

/// L2 normalize truncated vectors (separate pass for efficiency).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn matryoshka_normalize(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let vec_idx = global_id.x;

    if (vec_idx >= norm_params.num_vectors) {
        return;
    }

    let offset = vec_idx * norm_params.target_dim;

    // Compute L2 norm
    var norm_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < norm_params.target_dim; i = i + 1u) {
        let val = norm_vectors[offset + i];
        norm_sq = norm_sq + val * val;
    }

    let norm = sqrt(norm_sq);
    if (norm > 0.0) {
        let inv_norm = 1.0 / norm;
        for (var i: u32 = 0u; i < norm_params.target_dim; i = i + 1u) {
            norm_vectors[offset + i] = norm_vectors[offset + i] * inv_norm;
        }
    }
}

// ============================================================================
// Top-K Selection for Rerank Pipeline
// ============================================================================

// Maximum K we can handle in workgroup local memory
const MAX_LOCAL_K: u32 = 256u;

struct TopKParams {
    num_elements: u32,      // Total number of elements to select from
    k: u32,                 // Number of top elements to select
    ascending: u32,         // 0 = descending (higher is better), 1 = ascending (lower is better)
    _pad: u32,
};

// Module-scope bindings for top_k_select_f32
@group(0) @binding(0) var<storage, read> topk_scores_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> topk_out_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_out_scores: array<f32>;
@group(0) @binding(3) var<uniform> topk_params: TopKParams;

// Workgroup shared memory for local top-K
var<workgroup> local_scores: array<f32, MAX_LOCAL_K>;
var<workgroup> local_indices: array<u32, MAX_LOCAL_K>;
var<workgroup> local_count: atomic<u32>;

/// Insert into sorted local array (insertion sort for small K).
/// Returns true if inserted.
fn try_insert_local(score: f32, index: u32, k: u32, ascending: bool) -> bool {
    // Find insertion position
    var pos: u32 = k;
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        let cmp = select(score > local_scores[i], score < local_scores[i], ascending);
        if (cmp) {
            pos = i;
            break;
        }
    }

    if (pos >= k) {
        return false;
    }

    // Shift elements down
    for (var i: u32 = k - 1u; i > pos; i = i - 1u) {
        local_scores[i] = local_scores[i - 1u];
        local_indices[i] = local_indices[i - 1u];
    }

    // Insert new element
    local_scores[pos] = score;
    local_indices[pos] = index;
    return true;
}

/// Top-K selection using workgroup-parallel processing.
/// Each workgroup processes a chunk and maintains local top-K.
/// Final merge happens on CPU for simplicity.
///
/// For rerank pipeline:
/// - Binary stage: selects top-K from Hamming distances (ascending = true, lower = better)
/// - Int8 stage: selects top-K from dot products (ascending = false, higher = better)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn top_k_select_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let k = min(topk_params.k, MAX_LOCAL_K);
    let ascending = topk_params.ascending == 1u;

    // Initialize local arrays (first k threads)
    if (local_idx < k) {
        local_scores[local_idx] = select(-3.402823e38, 3.402823e38, ascending);
        local_indices[local_idx] = 0xFFFFFFFFu;
    }

    // Initialize counter
    if (local_idx == 0u) {
        atomicStore(&local_count, 0u);
    }

    workgroupBarrier();

    // Each thread processes elements strided by workgroup size
    let num_elements = topk_params.num_elements;
    let elements_per_group = (num_elements + 255u) / 256u;  // Ceiling division
    let group_start = group_id.x * elements_per_group;
    let group_end = min(group_start + elements_per_group, num_elements);

    // Process elements assigned to this workgroup
    var elem_idx = group_start + local_idx;
    while (elem_idx < group_end) {
        let score = topk_scores_f32[elem_idx];

        // Check if this score should be in top-K
        // Compare with worst score in current top-K
        let worst_idx = k - 1u;
        let dominated = select(
            score > local_scores[worst_idx],  // descending: need higher score
            score < local_scores[worst_idx],  // ascending: need lower score
            ascending
        );

        if (dominated) {
            // Try to insert (need synchronization)
            // Use atomic counter as a simple lock
            let slot = atomicAdd(&local_count, 1u) % k;

            // Simple replacement strategy (not perfectly sorted, but fast)
            // Final sort will happen on CPU
            let replace = select(
                score > local_scores[slot],
                score < local_scores[slot],
                ascending
            );
            if (replace) {
                local_scores[slot] = score;
                local_indices[slot] = elem_idx;
            }
        }

        elem_idx = elem_idx + WORKGROUP_SIZE;
    }

    workgroupBarrier();

    // Write output (first k threads)
    let out_offset = group_id.x * k;
    if (local_idx < k) {
        topk_out_indices[out_offset + local_idx] = local_indices[local_idx];
        topk_out_scores[out_offset + local_idx] = local_scores[local_idx];
    }
}

// ============================================================================
// Top-K Selection for i32 scores (Hamming distance)
// ============================================================================

// Module-scope bindings for top_k_select_i32
@group(0) @binding(0) var<storage, read> topk_scores_i32: array<i32>;
@group(0) @binding(1) var<storage, read_write> topk_i32_out_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_i32_out_scores: array<i32>;
@group(0) @binding(3) var<uniform> topk_i32_params: TopKParams;

var<workgroup> local_scores_i32: array<i32, MAX_LOCAL_K>;
var<workgroup> local_indices_i32: array<u32, MAX_LOCAL_K>;
var<workgroup> local_count_i32: atomic<u32>;

/// Top-K selection for i32 scores (used for Hamming distance in binary stage).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn top_k_select_i32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let k = min(topk_i32_params.k, MAX_LOCAL_K);
    let ascending = topk_i32_params.ascending == 1u;

    // Initialize local arrays (first k threads)
    if (local_idx < k) {
        local_scores_i32[local_idx] = select(-2147483647, 2147483647, ascending);
        local_indices_i32[local_idx] = 0xFFFFFFFFu;
    }

    if (local_idx == 0u) {
        atomicStore(&local_count_i32, 0u);
    }

    workgroupBarrier();

    let num_elements = topk_i32_params.num_elements;
    let elements_per_group = (num_elements + 255u) / 256u;
    let group_start = group_id.x * elements_per_group;
    let group_end = min(group_start + elements_per_group, num_elements);

    var elem_idx = group_start + local_idx;
    while (elem_idx < group_end) {
        let score = topk_scores_i32[elem_idx];

        let worst_idx = k - 1u;
        let dominated = select(
            score > local_scores_i32[worst_idx],
            score < local_scores_i32[worst_idx],
            ascending
        );

        if (dominated) {
            let slot = atomicAdd(&local_count_i32, 1u) % k;
            let replace = select(
                score > local_scores_i32[slot],
                score < local_scores_i32[slot],
                ascending
            );
            if (replace) {
                local_scores_i32[slot] = score;
                local_indices_i32[slot] = elem_idx;
            }
        }

        elem_idx = elem_idx + WORKGROUP_SIZE;
    }

    workgroupBarrier();

    let out_offset = group_id.x * k;
    if (local_idx < k) {
        topk_i32_out_indices[out_offset + local_idx] = local_indices_i32[local_idx];
        topk_i32_out_scores[out_offset + local_idx] = local_scores_i32[local_idx];
    }
}
