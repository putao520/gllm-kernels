// DeFT / Talon Flash Tree-attention HIP Kernel
// Based on DeFT (ICLR'25), Talon (ICLR'26), and SEQUOIA
//
// Key operations:
// - Tree structure verification with O(n+m) complexity
// - DeFT-Flatten: uniform tree distribution to GPU SMs
// - DeFT-Node: node-level parallelism without causal mask
// - Traversal verification: sequence-level path verification

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 64
#define MAX_TREE_DEPTH 32
#define MAX_HEAD_DIM 256

// Helper: warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Helper: warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// =============================================================================
// Tree Mask Generation
// Generates attention mask from tree parent indices
// =============================================================================
extern "C" __global__ void flash_tree_generate_mask(
    const int* __restrict__ parent_indices,  // [num_nodes]
    int* __restrict__ tree_mask,             // [num_nodes, num_nodes]
    const int num_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_nodes || j >= num_nodes) return;

    // Check if j is an ancestor of i (or i == j)
    int mask_val = 0;

    if (i == j) {
        mask_val = 1;  // Self-attention always allowed
    } else {
        // Traverse from i to root, check if we hit j
        int current = i;
        while (current >= 0 && current < num_nodes) {
            int parent = parent_indices[current];
            if (parent == j) {
                mask_val = 1;
                break;
            }
            if (parent < 0 || parent >= num_nodes) break;
            current = parent;
        }
    }

    tree_mask[i * num_nodes + j] = mask_val;
}

// =============================================================================
// Flash Tree Attention Forward (DeFT-Flatten style)
// Processes tree-structured attention with tiled computation
// =============================================================================
extern "C" __global__ void flash_tree_attention_forward_f32(
    const float* __restrict__ Q,              // [batch, num_heads, prompt_len, head_dim]
    const float* __restrict__ K,              // [batch, num_heads, prompt_len + tree_len, head_dim]
    const float* __restrict__ V,              // [batch, num_heads, prompt_len + tree_len, head_dim]
    const int* __restrict__ tree_mask,        // [tree_len, tree_len] or NULL for full attention
    const int* __restrict__ parent_indices,   // [tree_len] parent index for each tree node
    float* __restrict__ O,                    // [batch, num_heads, prompt_len + tree_len, head_dim]
    const int batch_size,
    const int num_heads,
    const int prompt_len,
    const int tree_len,
    const int head_dim,
    const float scale
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int block_idx = blockIdx.x;

    int total_len = prompt_len + tree_len;
    int queries_per_head = total_len;
    int total_blocks = batch_size * num_heads * queries_per_head;

    if (block_idx >= total_blocks) return;

    int bh_idx = block_idx / queries_per_head;
    int q_pos = block_idx % queries_per_head;
    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;

    int head_stride = total_len * head_dim;
    int batch_stride = num_heads * head_stride;
    int base = b * batch_stride + h * head_stride;

    const float* q_ptr = Q + base + q_pos * head_dim;
    float* o_ptr = O + base + q_pos * head_dim;

    // Initialize accumulator
    float out[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim && d < MAX_HEAD_DIM; ++d) {
        out[d] = 0.0f;
    }

    // Compute max score (first pass)
    float max_score = -FLT_MAX;

    for (int kv_pos = 0; kv_pos < total_len; ++kv_pos) {
        // Check tree mask if in tree region
        if (q_pos >= prompt_len && kv_pos >= prompt_len) {
            int tree_i = q_pos - prompt_len;
            int tree_j = kv_pos - prompt_len;

            // For tree nodes: allow attention only to ancestors
            if (tree_mask != NULL) {
                if (tree_mask[tree_i * tree_len + tree_j] == 0) {
                    continue;
                }
            } else if (parent_indices != NULL) {
                // Check ancestry inline if no precomputed mask
                int valid = (tree_i == tree_j) ? 1 : 0;
                if (!valid) {
                    int current = tree_i;
                    while (current >= 0 && current < tree_len && !valid) {
                        int parent = parent_indices[current];
                        if (parent == tree_j) {
                            valid = 1;
                        } else if (parent < 0) {
                            break;
                        }
                        current = parent;
                    }
                }
                if (!valid) continue;
            }
        }

        // Prompt tokens: causal attention for tree queries
        if (q_pos >= prompt_len && kv_pos >= prompt_len) {
            // Tree-to-tree: already handled above
        } else if (q_pos >= prompt_len) {
            // Tree-to-prompt: always allowed
        } else if (kv_pos > q_pos) {
            // Prompt-to-prompt: causal
            continue;
        }

        const float* k_ptr = K + base + kv_pos * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        float score = dot * scale;
        max_score = fmaxf(max_score, score);
    }

    // Second pass: compute softmax and weighted sum
    float sum_exp = 0.0f;

    for (int kv_pos = 0; kv_pos < total_len; ++kv_pos) {
        // Same masking logic as first pass
        if (q_pos >= prompt_len && kv_pos >= prompt_len) {
            int tree_i = q_pos - prompt_len;
            int tree_j = kv_pos - prompt_len;

            if (tree_mask != NULL) {
                if (tree_mask[tree_i * tree_len + tree_j] == 0) {
                    continue;
                }
            } else if (parent_indices != NULL) {
                int valid = (tree_i == tree_j) ? 1 : 0;
                if (!valid) {
                    int current = tree_i;
                    while (current >= 0 && current < tree_len && !valid) {
                        int parent = parent_indices[current];
                        if (parent == tree_j) {
                            valid = 1;
                        } else if (parent < 0) {
                            break;
                        }
                        current = parent;
                    }
                }
                if (!valid) continue;
            }
        } else if (q_pos >= prompt_len) {
            // Tree-to-prompt: allowed
        } else if (kv_pos > q_pos) {
            continue;
        }

        const float* k_ptr = K + base + kv_pos * head_dim;
        const float* v_ptr = V + base + kv_pos * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        float score = dot * scale;
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;

        for (int d = 0; d < head_dim; ++d) {
            out[d] += exp_score * v_ptr[d];
        }
    }

    // Normalize and write output
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = out[d] * inv_sum;
    }
}

extern "C" __global__ void flash_tree_attention_forward_f16(
    const _Float16* __restrict__ Q,
    const _Float16* __restrict__ K,
    const _Float16* __restrict__ V,
    const int* __restrict__ tree_mask,
    const int* __restrict__ parent_indices,
    _Float16* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int prompt_len,
    const int tree_len,
    const int head_dim,
    const float scale
) {
    int block_idx = blockIdx.x;

    int total_len = prompt_len + tree_len;
    int queries_per_head = total_len;
    int total_blocks = batch_size * num_heads * queries_per_head;

    if (block_idx >= total_blocks) return;

    int bh_idx = block_idx / queries_per_head;
    int q_pos = block_idx % queries_per_head;
    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;

    int head_stride = total_len * head_dim;
    int batch_stride = num_heads * head_stride;
    int base = b * batch_stride + h * head_stride;

    const _Float16* q_ptr = Q + base + q_pos * head_dim;
    _Float16* o_ptr = O + base + q_pos * head_dim;

    float out[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim && d < MAX_HEAD_DIM; ++d) {
        out[d] = 0.0f;
    }

    float max_score = -FLT_MAX;

    // First pass: find max
    for (int kv_pos = 0; kv_pos < total_len; ++kv_pos) {
        if (q_pos >= prompt_len && kv_pos >= prompt_len) {
            int tree_i = q_pos - prompt_len;
            int tree_j = kv_pos - prompt_len;
            if (tree_mask != NULL && tree_mask[tree_i * tree_len + tree_j] == 0) continue;
        } else if (q_pos < prompt_len && kv_pos > q_pos) {
            continue;
        }

        const _Float16* k_ptr = K + base + kv_pos * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += (float)q_ptr[d] * (float)k_ptr[d];
        }
        max_score = fmaxf(max_score, dot * scale);
    }

    // Second pass: softmax + accumulate
    float sum_exp = 0.0f;
    for (int kv_pos = 0; kv_pos < total_len; ++kv_pos) {
        if (q_pos >= prompt_len && kv_pos >= prompt_len) {
            int tree_i = q_pos - prompt_len;
            int tree_j = kv_pos - prompt_len;
            if (tree_mask != NULL && tree_mask[tree_i * tree_len + tree_j] == 0) continue;
        } else if (q_pos < prompt_len && kv_pos > q_pos) {
            continue;
        }

        const _Float16* k_ptr = K + base + kv_pos * head_dim;
        const _Float16* v_ptr = V + base + kv_pos * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += (float)q_ptr[d] * (float)k_ptr[d];
        }
        float exp_score = expf(dot * scale - max_score);
        sum_exp += exp_score;

        for (int d = 0; d < head_dim; ++d) {
            out[d] += exp_score * (float)v_ptr[d];
        }
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = (_Float16)(out[d] * inv_sum);
    }
}

// =============================================================================
// Traversal Verification
// Verifies complete paths through the token tree
// =============================================================================
extern "C" __global__ void flash_tree_verify_path(
    const int* __restrict__ path_tokens,      // [batch, max_path_len]
    const int* __restrict__ path_lengths,     // [batch]
    const float* __restrict__ target_logits,  // [batch, max_path_len, vocab_size]
    int* __restrict__ accepted_lengths,       // [batch]
    const int batch_size,
    const int max_path_len,
    const int vocab_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int path_len = path_lengths[b];
    int accepted = 0;

    for (int pos = 0; pos < path_len && pos < max_path_len; ++pos) {
        int token = path_tokens[b * max_path_len + pos];
        int base = (b * max_path_len + pos) * vocab_size;

        // Find argmax of target logits
        float max_logit = -FLT_MAX;
        int max_token = 0;
        for (int v = 0; v < vocab_size; ++v) {
            if (target_logits[base + v] > max_logit) {
                max_logit = target_logits[base + v];
                max_token = v;
            }
        }

        if (max_token == token) {
            accepted = pos + 1;
        } else {
            break;  // First mismatch ends acceptance
        }
    }

    accepted_lengths[b] = accepted;
}
