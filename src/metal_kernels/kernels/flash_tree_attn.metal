// DeFT / Talon Flash Tree-Attention - Metal Shaders
// Based on DeFT (ICLR'25), Talon (ICLR'26), and SEQUOIA

#include <metal_stdlib>
using namespace metal;

#define MAX_HEAD_DIM 256
#define MAX_TREE_DEPTH 32

// Parameter structures (must match Rust side)
struct TreeAttentionParams {
    uint batch_size;
    uint num_heads;
    uint tree_size;
    uint context_len;
    uint head_dim;
    float scale;
    uint _pad[2];
};

struct VerifyTreeParams {
    uint batch_size;
    uint tree_size;
    uint vocab_size;
    uint _pad;
};

struct BuildTreeMaskParams {
    uint tree_size;
    uint max_depth;
    uint _pad[2];
};

// Tree attention forward (F32)
// Q: [batch, num_heads, tree_size, head_dim]
// K: [batch, num_heads, context_len, head_dim]
// V: [batch, num_heads, context_len, head_dim]
// tree_mask: [tree_size, context_len] (1 = attend, 0 = mask)
// parent_indices: [tree_size]
// O: [batch, num_heads, tree_size, head_dim]
kernel void flash_tree_attn_forward_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const uint* tree_mask [[buffer(3)]],
    device const int* parent_indices [[buffer(4)]],
    device float* O [[buffer(5)]],
    constant TreeAttentionParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.tree_size;
    if (tid >= total) return;
    
    uint tree_idx = tid % params.tree_size;
    uint bh_idx = tid / params.tree_size;
    uint batch_idx = bh_idx / params.num_heads;
    uint head_idx = bh_idx % params.num_heads;
    
    uint head_stride = params.tree_size * params.head_dim;
    uint batch_stride = params.num_heads * head_stride;
    uint q_base = batch_idx * batch_stride + head_idx * head_stride + tree_idx * params.head_dim;
    
    uint kv_head_stride = params.context_len * params.head_dim;
    uint kv_batch_stride = params.num_heads * kv_head_stride;
    uint kv_base = batch_idx * kv_batch_stride + head_idx * kv_head_stride;
    
    // First pass: find max score
    float max_score = -INFINITY;
    
    for (uint kv_pos = 0; kv_pos < params.context_len; kv_pos++) {
        // Check tree mask
        uint mask_idx = tree_idx * params.context_len + kv_pos;
        if (tree_mask[mask_idx] == 0) continue;
        
        // Compute Q @ K^T
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            dot += Q[q_base + d] * K[kv_base + kv_pos * params.head_dim + d];
        }
        float score = dot * params.scale;
        max_score = max(max_score, score);
    }
    
    // Second pass: compute softmax and weighted sum
    float sum_exp = 0.0f;
    float out[MAX_HEAD_DIM];
    for (uint d = 0; d < params.head_dim && d < MAX_HEAD_DIM; d++) {
        out[d] = 0.0f;
    }
    
    for (uint kv_pos = 0; kv_pos < params.context_len; kv_pos++) {
        uint mask_idx = tree_idx * params.context_len + kv_pos;
        if (tree_mask[mask_idx] == 0) continue;
        
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            dot += Q[q_base + d] * K[kv_base + kv_pos * params.head_dim + d];
        }
        float score = dot * params.scale;
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;
        
        for (uint d = 0; d < params.head_dim; d++) {
            out[d] += exp_score * V[kv_base + kv_pos * params.head_dim + d];
        }
    }
    
    // Normalize and write output
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    uint o_base = batch_idx * batch_stride + head_idx * head_stride + tree_idx * params.head_dim;
    for (uint d = 0; d < params.head_dim; d++) {
        O[o_base + d] = out[d] * inv_sum;
    }
}

// Tree attention forward (F16)
kernel void flash_tree_attn_forward_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const uint* tree_mask [[buffer(3)]],
    device const int* parent_indices [[buffer(4)]],
    device half* O [[buffer(5)]],
    constant TreeAttentionParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.num_heads * params.tree_size;
    if (tid >= total) return;
    
    uint tree_idx = tid % params.tree_size;
    uint bh_idx = tid / params.tree_size;
    uint batch_idx = bh_idx / params.num_heads;
    uint head_idx = bh_idx % params.num_heads;
    
    uint head_stride = params.tree_size * params.head_dim;
    uint batch_stride = params.num_heads * head_stride;
    uint q_base = batch_idx * batch_stride + head_idx * head_stride + tree_idx * params.head_dim;
    
    uint kv_head_stride = params.context_len * params.head_dim;
    uint kv_batch_stride = params.num_heads * kv_head_stride;
    uint kv_base = batch_idx * kv_batch_stride + head_idx * kv_head_stride;
    
    float max_score = -INFINITY;
    
    for (uint kv_pos = 0; kv_pos < params.context_len; kv_pos++) {
        uint mask_idx = tree_idx * params.context_len + kv_pos;
        if (tree_mask[mask_idx] == 0) continue;
        
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[kv_base + kv_pos * params.head_dim + d]);
        }
        float score = dot * params.scale;
        max_score = max(max_score, score);
    }
    
    float sum_exp = 0.0f;
    float out[MAX_HEAD_DIM];
    for (uint d = 0; d < params.head_dim && d < MAX_HEAD_DIM; d++) {
        out[d] = 0.0f;
    }
    
    for (uint kv_pos = 0; kv_pos < params.context_len; kv_pos++) {
        uint mask_idx = tree_idx * params.context_len + kv_pos;
        if (tree_mask[mask_idx] == 0) continue;
        
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[kv_base + kv_pos * params.head_dim + d]);
        }
        float score = dot * params.scale;
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;
        
        for (uint d = 0; d < params.head_dim; d++) {
            out[d] += exp_score * float(V[kv_base + kv_pos * params.head_dim + d]);
        }
    }
    
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    uint o_base = batch_idx * batch_stride + head_idx * head_stride + tree_idx * params.head_dim;
    for (uint d = 0; d < params.head_dim; d++) {
        O[o_base + d] = half(out[d] * inv_sum);
    }
}

// Verify tree tokens (F32)
// draft_tokens: [batch, tree_size]
// target_logits: [batch, tree_size, vocab_size]
// parent_indices: [tree_size]
// accepted_mask: [batch, tree_size]
// accepted_count: [batch]
kernel void flash_tree_attn_verify_f32(
    device const uint* draft_tokens [[buffer(0)]],
    device const float* target_logits [[buffer(1)]],
    device const int* parent_indices [[buffer(2)]],
    device uint* accepted_mask [[buffer(3)]],
    device uint* accepted_count [[buffer(4)]],
    constant VerifyTreeParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    uint batch_idx = tid;
    uint count = 0;
    
    // Initialize all to rejected
    for (uint t = 0; t < params.tree_size; t++) {
        accepted_mask[batch_idx * params.tree_size + t] = 0;
    }
    
    // BFS verification from root
    bool path_valid[MAX_TREE_DEPTH];
    for (uint i = 0; i < MAX_TREE_DEPTH; i++) {
        path_valid[i] = true;
    }
    
    for (uint t = 0; t < params.tree_size; t++) {
        int parent = parent_indices[t];
        
        // Check if parent is valid (root has parent -1)
        bool parent_ok = (parent < 0) || 
                         (uint(parent) < params.tree_size && 
                          accepted_mask[batch_idx * params.tree_size + uint(parent)] == 1);
        
        if (!parent_ok && parent >= 0) {
            continue;
        }
        
        // Find argmax of target logits at this position
        uint logits_base = (batch_idx * params.tree_size + t) * params.vocab_size;
        float max_logit = target_logits[logits_base];
        uint max_token = 0;
        
        for (uint v = 1; v < params.vocab_size; v++) {
            float logit = target_logits[logits_base + v];
            if (logit > max_logit) {
                max_logit = logit;
                max_token = v;
            }
        }
        
        // Check if draft matches target
        uint draft_token = draft_tokens[batch_idx * params.tree_size + t];
        if (draft_token == max_token) {
            accepted_mask[batch_idx * params.tree_size + t] = 1;
            count++;
        }
    }
    
    accepted_count[batch_idx] = count;
}

// Verify tree tokens (F16) - same logic, different input type
kernel void flash_tree_attn_verify_f16(
    device const uint* draft_tokens [[buffer(0)]],
    device const half* target_logits [[buffer(1)]],
    device const int* parent_indices [[buffer(2)]],
    device uint* accepted_mask [[buffer(3)]],
    device uint* accepted_count [[buffer(4)]],
    constant VerifyTreeParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    uint batch_idx = tid;
    uint count = 0;
    
    for (uint t = 0; t < params.tree_size; t++) {
        accepted_mask[batch_idx * params.tree_size + t] = 0;
    }
    
    for (uint t = 0; t < params.tree_size; t++) {
        int parent = parent_indices[t];
        
        bool parent_ok = (parent < 0) || 
                         (uint(parent) < params.tree_size && 
                          accepted_mask[batch_idx * params.tree_size + uint(parent)] == 1);
        
        if (!parent_ok && parent >= 0) {
            continue;
        }
        
        uint logits_base = (batch_idx * params.tree_size + t) * params.vocab_size;
        float max_logit = float(target_logits[logits_base]);
        uint max_token = 0;
        
        for (uint v = 1; v < params.vocab_size; v++) {
            float logit = float(target_logits[logits_base + v]);
            if (logit > max_logit) {
                max_logit = logit;
                max_token = v;
            }
        }
        
        uint draft_token = draft_tokens[batch_idx * params.tree_size + t];
        if (draft_token == max_token) {
            accepted_mask[batch_idx * params.tree_size + t] = 1;
            count++;
        }
    }
    
    accepted_count[batch_idx] = count;
}

// Build tree attention mask from parent indices
// parent_indices: [tree_size]
// mask: [tree_size, tree_size] as u8
kernel void flash_tree_attn_build_mask(
    device const int* parent_indices [[buffer(0)]],
    device uchar* mask [[buffer(1)]],
    constant BuildTreeMaskParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;  // Query position
    uint j = tid.y;  // Key position
    
    if (i >= params.tree_size || j >= params.tree_size) return;
    
    uint mask_idx = i * params.tree_size + j;
    
    // Self-attention always allowed
    if (i == j) {
        mask[mask_idx] = 1;
        return;
    }
    
    // Check if j is an ancestor of i
    uchar is_ancestor = 0;
    int current = int(i);
    uint depth = 0;
    
    while (current >= 0 && depth < params.max_depth) {
        int parent = parent_indices[current];
        if (parent == int(j)) {
            is_ancestor = 1;
            break;
        }
        if (parent < 0 || uint(parent) >= params.tree_size) break;
        current = parent;
        depth++;
    }
    
    mask[mask_idx] = is_ancestor;
}
