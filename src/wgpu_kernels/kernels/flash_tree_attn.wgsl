// DeFT / Talon Flash Tree-Attention
// Based on DeFT (ICLR'25) and Talon (ICLR'26)
//
// Kernels:
// - tree_attn_forward_f32: Tree-structured attention O(n+m) complexity
// - tree_attn_build_mask_f32: Build tree mask from parent indices
// - tree_attn_forward_f16: F16 variant
// - tree_attn_verify_f32: Verify draft tokens against ground truth

enable f16;

const WORKGROUP_SIZE: u32 = 128u;
const MAX_HEAD_DIM: u32 = 256u;
const BLOCK_K: u32 = 8u;

// ============================================================================
// Parameters
// ============================================================================

struct TreeAttnParams {
    batch_size: u32,
    num_heads: u32,
    prompt_len: u32,        // n: prompt sequence length
    tree_size: u32,         // m: number of tree nodes
    head_dim: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
};

struct TreeMaskParams {
    num_nodes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct VerifyParams {
    batch_size: u32,
    tree_size: u32,
    _pad0: u32,
    _pad1: u32,
};

// ============================================================================
// Workgroup shared memory
// ============================================================================

var<workgroup> k_tile_tree: array<f32, BLOCK_K * MAX_HEAD_DIM>;
var<workgroup> v_tile_tree: array<f32, BLOCK_K * MAX_HEAD_DIM>;

// ============================================================================
// F32 Tree Mask Building Kernel
// ============================================================================
// Builds attention mask from parent indices (j can attend to i iff is_ancestor(i,j))
// Input: parent_indices[tree_size] (-1 for root)
// Output: tree_mask[tree_size, tree_size] (1 if can attend, 0 otherwise)

@group(0) @binding(0) var<storage, read> mask_parents_f32: array<i32>;
@group(0) @binding(1) var<storage, read_write> mask_output_i32: array<i32>;
@group(0) @binding(2) var<uniform> mask_params_f32: TreeMaskParams;

fn is_ancestor(i: u32, j: u32, parents: ptr<storage, array<i32>, read>, num_nodes: u32) -> bool {
    if (i == j) {
        return true;
    }

    var current = j;
    loop {
        if (current >= num_nodes) {
            return false;
        }
        let parent = (*parents)[current];
        if (parent < 0) {
            return false;
        }
        if (u32(parent) == i) {
            return true;
        }
        current = u32(parent);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn tree_attn_build_mask_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = mask_params_f32.num_nodes * mask_params_f32.num_nodes;

    if (idx >= total) {
        return;
    }

    let j = idx / mask_params_f32.num_nodes; // query node
    let i = idx % mask_params_f32.num_nodes; // key node

    // j can attend to i if i is an ancestor of j
    if (is_ancestor(i, j, &mask_parents_f32, mask_params_f32.num_nodes)) {
        mask_output_i32[idx] = 1;
    } else {
        mask_output_i32[idx] = 0;
    }
}

// ============================================================================
// F32 Tree Attention Forward Kernel
// ============================================================================
// Computes tree-structured attention with O(n+m) complexity using DeFT-Flatten
// Input: Q[batch, tree_size, num_heads, head_dim], K/V[batch, prompt_len+tree_size, num_heads, head_dim]
//        tree_mask[tree_size, tree_size]
// Output: O[batch, tree_size, num_heads, head_dim]

@group(0) @binding(0) var<storage, read> tree_q_f32: array<f32>;
@group(0) @binding(1) var<storage, read> tree_k_f32: array<f32>;
@group(0) @binding(2) var<storage, read> tree_v_f32: array<f32>;
@group(0) @binding(3) var<storage, read> tree_mask_i32: array<i32>;
@group(0) @binding(4) var<storage, read_write> tree_o_f32: array<f32>;
@group(0) @binding(5) var<uniform> tree_params_f32: TreeAttnParams;

fn load_kv_tile_tree_f32(
    local_id: u32,
    tile_len: u32,
    head_dim: u32,
    k_base: u32,
    v_base: u32,
) {
    let elements = tile_len * head_dim;
    var idx = local_id;
    loop {
        if (idx >= elements) {
            break;
        }
        let tile_k = idx / head_dim;
        let d = idx - tile_k * head_dim;
        k_tile_tree[tile_k * MAX_HEAD_DIM + d] = tree_k_f32[k_base + tile_k * head_dim + d];
        v_tile_tree[tile_k * MAX_HEAD_DIM + d] = tree_v_f32[v_base + tile_k * head_dim + d];
        idx = idx + WORKGROUP_SIZE;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn tree_attn_forward_f32(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if (tree_params_f32.tree_size == 0u || tree_params_f32.head_dim == 0u) {
        return;
    }
    if (tree_params_f32.head_dim > MAX_HEAD_DIM) {
        return;
    }

    let q_index = group_id.x * WORKGROUP_SIZE + local_id.x;
    if (q_index >= tree_params_f32.tree_size) {
        return;
    }

    let bh = group_id.y;
    let batch = bh / tree_params_f32.num_heads;
    if (batch >= tree_params_f32.batch_size) {
        return;
    }
    let head = bh - batch * tree_params_f32.num_heads;

    let head_dim = tree_params_f32.head_dim;
    let total_kv_len = tree_params_f32.prompt_len + tree_params_f32.tree_size;

    // Q base: [batch, tree_size, num_heads, head_dim]
    let q_base = ((batch * tree_params_f32.tree_size + q_index) * tree_params_f32.num_heads + head) * head_dim;

    // Load Q into local memory
    var q_local: array<f32, MAX_HEAD_DIM>;
    var o_local: array<f32, MAX_HEAD_DIM>;

    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = tree_q_f32[q_base + d];
        o_local[d] = 0.0;
    }

    var m = -3.402823e38f;
    var l = 0.0f;

    // Process prompt tokens (all visible to tree nodes)
    let tile_k = max(1u, min(BLOCK_K, tree_params_f32.prompt_len));
    var key_base: u32 = 0u;
    loop {
        if (key_base >= tree_params_f32.prompt_len) {
            break;
        }
        let remaining = tree_params_f32.prompt_len - key_base;
        let tile_len = min(tile_k, remaining);

        // K/V base: [batch, kv_len, num_heads, head_dim]
        let kv_offset = ((batch * total_kv_len + key_base) * tree_params_f32.num_heads + head) * head_dim;
        load_kv_tile_tree_f32(local_id.x, tile_len, head_dim, kv_offset, kv_offset);
        workgroupBarrier();

        for (var t: u32 = 0u; t < tile_len; t = t + 1u) {
            var sum = 0.0f;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                sum += q_local[d] * k_tile_tree[t * MAX_HEAD_DIM + d];
            }

            let score = sum * tree_params_f32.scale;
            let m_new = max(m, score);
            let exp_m = exp(m - m_new);
            let exp_score = exp(score - m_new);
            let l_new = l * exp_m + exp_score;

            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                o_local[d] = o_local[d] * exp_m + exp_score * v_tile_tree[t * MAX_HEAD_DIM + d];
            }

            m = m_new;
            l = l_new;
        }

        workgroupBarrier();
        key_base += tile_len;
    }

    // Process tree tokens with tree mask
    key_base = 0u;
    loop {
        if (key_base >= tree_params_f32.tree_size) {
            break;
        }
        let remaining = tree_params_f32.tree_size - key_base;
        let tile_len = min(tile_k, remaining);

        let tree_kv_start = tree_params_f32.prompt_len;
        let kv_offset = ((batch * total_kv_len + tree_kv_start + key_base) * tree_params_f32.num_heads + head) * head_dim;
        load_kv_tile_tree_f32(local_id.x, tile_len, head_dim, kv_offset, kv_offset);
        workgroupBarrier();

        for (var t: u32 = 0u; t < tile_len; t = t + 1u) {
            let key_idx = key_base + t;
            // Check tree mask: can q_index attend to key_idx?
            let mask_idx = q_index * tree_params_f32.tree_size + key_idx;
            let mask_val = tree_mask_i32[mask_idx];

            if (mask_val != 0) {
                var sum = 0.0f;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    sum += q_local[d] * k_tile_tree[t * MAX_HEAD_DIM + d];
                }

                let score = sum * tree_params_f32.scale;
                let m_new = max(m, score);
                let exp_m = exp(m - m_new);
                let exp_score = exp(score - m_new);
                let l_new = l * exp_m + exp_score;

                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    o_local[d] = o_local[d] * exp_m + exp_score * v_tile_tree[t * MAX_HEAD_DIM + d];
                }

                m = m_new;
                l = l_new;
            }
        }

        workgroupBarrier();
        key_base += tile_len;
    }

    // Write output
    let inv_l = 1.0 / l;
    let o_base = ((batch * tree_params_f32.tree_size + q_index) * tree_params_f32.num_heads + head) * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        tree_o_f32[o_base + d] = o_local[d] * inv_l;
    }
}

// ============================================================================
// F32 Verification Kernel
// ============================================================================
// Verifies draft tokens against ground truth, returns longest accepted prefix
// Input: draft_tokens[batch, tree_size], ground_truth[batch, tree_size]
//        parent_indices[tree_size]
// Output: accepted_length[batch], accepted_tokens[batch, tree_size]

@group(0) @binding(0) var<storage, read> verify_draft_f32: array<u32>;
@group(0) @binding(1) var<storage, read> verify_truth_f32: array<u32>;
@group(0) @binding(2) var<storage, read> verify_parents_f32: array<i32>;
@group(0) @binding(3) var<storage, read_write> verify_length_f32: array<u32>;
@group(0) @binding(4) var<storage, read_write> verify_accepted_f32: array<u32>;
@group(0) @binding(5) var<uniform> verify_params_f32: VerifyParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn tree_attn_verify_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let batch_idx = global_id.x;

    if (batch_idx >= verify_params_f32.batch_size) {
        return;
    }

    let base = batch_idx * verify_params_f32.tree_size;
    var accepted_count = 0u;

    // Greedy path traversal: start from root, follow matching children
    var current = 0u;
    loop {
        if (current >= verify_params_f32.tree_size) {
            break;
        }

        let draft = verify_draft_f32[base + current];
        let truth = verify_truth_f32[base + current];

        if (draft == truth) {
            verify_accepted_f32[base + accepted_count] = current;
            accepted_count += 1u;

            // Find next child (simple linear search for first child)
            var found_child = false;
            for (var i: u32 = 0u; i < verify_params_f32.tree_size; i = i + 1u) {
                if (verify_parents_f32[i] == i32(current)) {
                    current = i;
                    found_child = true;
                    break;
                }
            }
            if (!found_child) {
                break;
            }
        } else {
            break;
        }
    }

    verify_length_f32[batch_idx] = accepted_count;
}

// ============================================================================
// F16 Tree Attention Forward Kernel
// ============================================================================

@group(0) @binding(0) var<storage, read> tree_q_f16: array<f16>;
@group(0) @binding(1) var<storage, read> tree_k_f16: array<f16>;
@group(0) @binding(2) var<storage, read> tree_v_f16: array<f16>;
@group(0) @binding(3) var<storage, read> tree_mask_i32_f16: array<i32>;
@group(0) @binding(4) var<storage, read_write> tree_o_f16: array<f16>;
@group(0) @binding(5) var<uniform> tree_params_f16: TreeAttnParams;

fn load_kv_tile_tree_f16(
    local_id: u32,
    tile_len: u32,
    head_dim: u32,
    k_base: u32,
    v_base: u32,
) {
    let elements = tile_len * head_dim;
    var idx = local_id;
    loop {
        if (idx >= elements) {
            break;
        }
        let tile_k = idx / head_dim;
        let d = idx - tile_k * head_dim;
        k_tile_tree[tile_k * MAX_HEAD_DIM + d] = f32(tree_k_f16[k_base + tile_k * head_dim + d]);
        v_tile_tree[tile_k * MAX_HEAD_DIM + d] = f32(tree_v_f16[v_base + tile_k * head_dim + d]);
        idx = idx + WORKGROUP_SIZE;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn tree_attn_forward_f16(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if (tree_params_f16.tree_size == 0u || tree_params_f16.head_dim == 0u) {
        return;
    }
    if (tree_params_f16.head_dim > MAX_HEAD_DIM) {
        return;
    }

    let q_index = group_id.x * WORKGROUP_SIZE + local_id.x;
    if (q_index >= tree_params_f16.tree_size) {
        return;
    }

    let bh = group_id.y;
    let batch = bh / tree_params_f16.num_heads;
    if (batch >= tree_params_f16.batch_size) {
        return;
    }
    let head = bh - batch * tree_params_f16.num_heads;

    let head_dim = tree_params_f16.head_dim;
    let total_kv_len = tree_params_f16.prompt_len + tree_params_f16.tree_size;

    let q_base = ((batch * tree_params_f16.tree_size + q_index) * tree_params_f16.num_heads + head) * head_dim;

    var q_local: array<f32, MAX_HEAD_DIM>;
    var o_local: array<f32, MAX_HEAD_DIM>;

    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = f32(tree_q_f16[q_base + d]);
        o_local[d] = 0.0;
    }

    var m = -3.402823e38f;
    var l = 0.0f;

    // Process prompt tokens
    let tile_k = max(1u, min(BLOCK_K, tree_params_f16.prompt_len));
    var key_base: u32 = 0u;
    loop {
        if (key_base >= tree_params_f16.prompt_len) {
            break;
        }
        let remaining = tree_params_f16.prompt_len - key_base;
        let tile_len = min(tile_k, remaining);

        let kv_offset = ((batch * total_kv_len + key_base) * tree_params_f16.num_heads + head) * head_dim;
        load_kv_tile_tree_f16(local_id.x, tile_len, head_dim, kv_offset, kv_offset);
        workgroupBarrier();

        for (var t: u32 = 0u; t < tile_len; t = t + 1u) {
            var sum = 0.0f;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                sum += q_local[d] * k_tile_tree[t * MAX_HEAD_DIM + d];
            }

            let score = sum * tree_params_f16.scale;
            let m_new = max(m, score);
            let exp_m = exp(m - m_new);
            let exp_score = exp(score - m_new);
            let l_new = l * exp_m + exp_score;

            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                o_local[d] = o_local[d] * exp_m + exp_score * v_tile_tree[t * MAX_HEAD_DIM + d];
            }

            m = m_new;
            l = l_new;
        }

        workgroupBarrier();
        key_base += tile_len;
    }

    // Process tree tokens with mask
    key_base = 0u;
    loop {
        if (key_base >= tree_params_f16.tree_size) {
            break;
        }
        let remaining = tree_params_f16.tree_size - key_base;
        let tile_len = min(tile_k, remaining);

        let tree_kv_start = tree_params_f16.prompt_len;
        let kv_offset = ((batch * total_kv_len + tree_kv_start + key_base) * tree_params_f16.num_heads + head) * head_dim;
        load_kv_tile_tree_f16(local_id.x, tile_len, head_dim, kv_offset, kv_offset);
        workgroupBarrier();

        for (var t: u32 = 0u; t < tile_len; t = t + 1u) {
            let key_idx = key_base + t;
            let mask_idx = q_index * tree_params_f16.tree_size + key_idx;
            let mask_val = tree_mask_i32_f16[mask_idx];

            if (mask_val != 0) {
                var sum = 0.0f;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    sum += q_local[d] * k_tile_tree[t * MAX_HEAD_DIM + d];
                }

                let score = sum * tree_params_f16.scale;
                let m_new = max(m, score);
                let exp_m = exp(m - m_new);
                let exp_score = exp(score - m_new);
                let l_new = l * exp_m + exp_score;

                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    o_local[d] = o_local[d] * exp_m + exp_score * v_tile_tree[t * MAX_HEAD_DIM + d];
                }

                m = m_new;
                l = l_new;
            }
        }

        workgroupBarrier();
        key_base += tile_len;
    }

    let inv_l = 1.0 / l;
    let o_base = ((batch * tree_params_f16.tree_size + q_index) * tree_params_f16.num_heads + head) * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        tree_o_f16[o_base + d] = f16(o_local[d] * inv_l);
    }
}
