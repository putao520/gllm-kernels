// Medusa Multi-Head Speculative Decoding - Metal Shaders
// Based on Medusa (ICML'24)

#include <metal_stdlib>
using namespace metal;

#define MAX_HEADS 8
#define MAX_TOPK 64
#define MAX_VOCAB_SIZE 65536

// Parameter structures
struct LMHeadParams {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint vocab_size;
    uint num_heads;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct TopKParams {
    uint batch_size;
    uint seq_len;
    uint vocab_size;
    uint top_k;
    uint num_heads;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct CandidateParams {
    uint batch_size;
    uint num_heads;
    uint top_k;
    uint max_candidates;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

struct VerifyParams {
    uint batch_size;
    uint num_candidates;
    uint vocab_size;
    uint _pad0;
};

// Medusa LM head forward (F32)
// Projects hidden states through multiple Medusa heads to get logits
kernel void medusa_lm_head_forward_f32(
    device const float* hidden [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant LMHeadParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.num_heads * params.vocab_size;
    if (tid >= total) return;
    
    uint vocab_idx = tid % params.vocab_size;
    uint head_idx = (tid / params.vocab_size) % params.num_heads;
    uint seq_idx = (tid / (params.vocab_size * params.num_heads)) % params.seq_len;
    uint batch_idx = tid / (params.vocab_size * params.num_heads * params.seq_len);
    
    // hidden: [batch, seq, hidden_dim]
    // weights: [num_heads, hidden_dim, vocab_size]
    uint hidden_base = batch_idx * params.seq_len * params.hidden_dim +
                       seq_idx * params.hidden_dim;
    
    uint weight_base = head_idx * params.hidden_dim * params.vocab_size;
    
    float sum = 0.0f;
    for (uint d = 0; d < params.hidden_dim; d++) {
        sum += hidden[hidden_base + d] * weights[weight_base + d * params.vocab_size + vocab_idx];
    }
    
    logits[tid] = sum;
}

// Medusa top-k sampling (F32)
// Finds top-k tokens for each Medusa head
kernel void medusa_topk_sample_f32(
    device const float* logits [[buffer(0)]],
    device uint* top_tokens [[buffer(1)]],
    device float* top_probs [[buffer(2)]],
    constant TopKParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.num_heads;
    if (tid >= total) return;
    
    uint head_idx = tid % params.num_heads;
    uint seq_idx = (tid / params.num_heads) % params.seq_len;
    uint batch_idx = tid / (params.num_heads * params.seq_len);
    
    uint logits_base = batch_idx * params.seq_len * params.num_heads * params.vocab_size +
                       seq_idx * params.num_heads * params.vocab_size +
                       head_idx * params.vocab_size;
    
    uint out_base = batch_idx * params.seq_len * params.num_heads * params.top_k +
                    seq_idx * params.num_heads * params.top_k +
                    head_idx * params.top_k;
    
    // Simple top-k selection (O(n*k))
    for (uint k = 0; k < params.top_k && k < MAX_TOPK; k++) {
        float max_val = -INFINITY;
        uint max_idx = 0;
        
        for (uint v = 0; v < params.vocab_size; v++) {
            float val = logits[logits_base + v];
            
            // Check if already selected
            bool already_selected = false;
            for (uint j = 0; j < k; j++) {
                if (top_tokens[out_base + j] == v) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && val > max_val) {
                max_val = val;
                max_idx = v;
            }
        }
        
        top_tokens[out_base + k] = max_idx;
        top_probs[out_base + k] = max_val;
    }
    
    // Convert logits to probabilities (softmax among top-k)
    float max_prob = top_probs[out_base];
    for (uint k = 1; k < params.top_k; k++) {
        max_prob = max(max_prob, top_probs[out_base + k]);
    }
    
    float sum_exp = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        top_probs[out_base + k] = exp(top_probs[out_base + k] - max_prob);
        sum_exp += top_probs[out_base + k];
    }
    
    for (uint k = 0; k < params.top_k; k++) {
        top_probs[out_base + k] /= sum_exp;
    }
}

// Medusa candidate tree building
kernel void medusa_build_candidates_f32(
    device const uint* top_tokens [[buffer(0)]],
    device const float* top_probs [[buffer(1)]],
    device uint* candidates [[buffer(2)]],
    device float* candidate_probs [[buffer(3)]],
    device uint* num_candidates [[buffer(4)]],
    constant CandidateParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    uint batch_idx = tid;
    uint tokens_base = batch_idx * params.num_heads * params.top_k;
    uint probs_base = tokens_base;
    uint cand_base = batch_idx * params.max_candidates * (params.num_heads + 1);
    
    // Build candidate sequences (Cartesian product of top-k from each head)
    // For simplicity, use greedy tree: just top-1 from each head
    uint count = 0;
    
    // Add the greedy path
    if (count < params.max_candidates) {
        for (uint h = 0; h < params.num_heads && h < MAX_HEADS; h++) {
            candidates[cand_base + count * (params.num_heads + 1) + h] = top_tokens[tokens_base + h * params.top_k];
        }
        candidates[cand_base + count * (params.num_heads + 1) + params.num_heads] = 0; // length marker
        
        float prob = 1.0f;
        for (uint h = 0; h < params.num_heads; h++) {
            prob *= top_probs[probs_base + h * params.top_k];
        }
        candidate_probs[batch_idx * params.max_candidates + count] = prob;
        count++;
    }
    
    // Add alternative paths (vary each head position)
    for (uint vary_h = 0; vary_h < params.num_heads && count < params.max_candidates; vary_h++) {
        for (uint k = 1; k < params.top_k && count < params.max_candidates; k++) {
            for (uint h = 0; h < params.num_heads; h++) {
                uint token_idx = (h == vary_h) ? k : 0;
                candidates[cand_base + count * (params.num_heads + 1) + h] = 
                    top_tokens[tokens_base + h * params.top_k + token_idx];
            }
            candidates[cand_base + count * (params.num_heads + 1) + params.num_heads] = vary_h;
            
            float prob = 1.0f;
            for (uint h = 0; h < params.num_heads; h++) {
                uint prob_idx = (h == vary_h) ? k : 0;
                prob *= top_probs[probs_base + h * params.top_k + prob_idx];
            }
            candidate_probs[batch_idx * params.max_candidates + count] = prob;
            count++;
        }
    }
    
    num_candidates[batch_idx] = count;
}

// Medusa verification
kernel void medusa_verify_candidates_f32(
    device const uint* candidates [[buffer(0)]],
    device const float* target_logits [[buffer(1)]],
    device uint* accepted_mask [[buffer(2)]],
    device uint* accepted_count [[buffer(3)]],
    constant VerifyParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    uint batch_idx = tid;
    uint count = 0;
    
    // For each candidate, check if tokens match target argmax
    for (uint c = 0; c < params.num_candidates; c++) {
        uint mask_idx = batch_idx * params.num_candidates + c;
        
        // Find argmax of target logits for this candidate position
        uint logits_base = (batch_idx * params.num_candidates + c) * params.vocab_size;
        float max_val = target_logits[logits_base];
        uint max_token = 0;
        
        for (uint v = 1; v < params.vocab_size; v++) {
            float val = target_logits[logits_base + v];
            if (val > max_val) {
                max_val = val;
                max_token = v;
            }
        }
        
        // Check if candidate matches
        uint cand_token = candidates[batch_idx * params.num_candidates + c];
        if (cand_token == max_token) {
            accepted_mask[mask_idx] = 1;
            count++;
        } else {
            accepted_mask[mask_idx] = 0;
        }
    }
    
    accepted_count[batch_idx] = count;
}

// F16 variants
kernel void medusa_lm_head_forward_f16(
    device const half* hidden [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device half* logits [[buffer(2)]],
    constant LMHeadParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.num_heads * params.vocab_size;
    if (tid >= total) return;
    
    uint vocab_idx = tid % params.vocab_size;
    uint head_idx = (tid / params.vocab_size) % params.num_heads;
    uint seq_idx = (tid / (params.vocab_size * params.num_heads)) % params.seq_len;
    uint batch_idx = tid / (params.vocab_size * params.num_heads * params.seq_len);
    
    uint hidden_base = batch_idx * params.seq_len * params.hidden_dim +
                       seq_idx * params.hidden_dim;
    
    uint weight_base = head_idx * params.hidden_dim * params.vocab_size;
    
    float sum = 0.0f;
    for (uint d = 0; d < params.hidden_dim; d++) {
        sum += float(hidden[hidden_base + d]) * float(weights[weight_base + d * params.vocab_size + vocab_idx]);
    }
    
    logits[tid] = half(sum);
}

kernel void medusa_topk_sample_f16(
    device const half* logits [[buffer(0)]],
    device uint* top_tokens [[buffer(1)]],
    device half* top_probs [[buffer(2)]],
    constant TopKParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.num_heads;
    if (tid >= total) return;
    
    uint head_idx = tid % params.num_heads;
    uint seq_idx = (tid / params.num_heads) % params.seq_len;
    uint batch_idx = tid / (params.num_heads * params.seq_len);
    
    uint logits_base = batch_idx * params.seq_len * params.num_heads * params.vocab_size +
                       seq_idx * params.num_heads * params.vocab_size +
                       head_idx * params.vocab_size;
    
    uint out_base = batch_idx * params.seq_len * params.num_heads * params.top_k +
                    seq_idx * params.num_heads * params.top_k +
                    head_idx * params.top_k;
    
    float top_vals[MAX_TOPK];
    uint top_idxs[MAX_TOPK];
    
    for (uint k = 0; k < params.top_k && k < MAX_TOPK; k++) {
        float max_val = -INFINITY;
        uint max_idx = 0;
        
        for (uint v = 0; v < params.vocab_size; v++) {
            float val = float(logits[logits_base + v]);
            
            bool already_selected = false;
            for (uint j = 0; j < k; j++) {
                if (top_idxs[j] == v) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && val > max_val) {
                max_val = val;
                max_idx = v;
            }
        }
        
        top_idxs[k] = max_idx;
        top_vals[k] = max_val;
        top_tokens[out_base + k] = max_idx;
    }
    
    // Softmax
    float max_prob = top_vals[0];
    for (uint k = 1; k < params.top_k; k++) {
        max_prob = max(max_prob, top_vals[k]);
    }
    
    float sum_exp = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        top_vals[k] = exp(top_vals[k] - max_prob);
        sum_exp += top_vals[k];
    }
    
    for (uint k = 0; k < params.top_k; k++) {
        top_probs[out_base + k] = half(top_vals[k] / sum_exp);
    }
}
