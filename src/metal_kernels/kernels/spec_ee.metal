// SpecEE / LayerSkip Early-Exit Speculative Decoding - Metal Shaders
// Based on SpecEE (ISCA'25) and LayerSkip (ACL'24)

#include <metal_stdlib>
using namespace metal;

#define MAX_HIDDEN_DIM 4096

// Parameter structures (must match Rust side)
struct ComputeConfidenceParams {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint num_layers;
    float temperature;
    uint _pad[3];
};

struct LayerSkipDecisionParams {
    uint batch_size;
    uint seq_len;
    uint num_layers;
    float confidence_threshold;
    uint min_layers;
    uint max_skip;
    uint _pad[2];
};

struct EarlyExitParams {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint vocab_size;
    uint current_layer;
    uint total_layers;
    float confidence_threshold;
    uint _pad;
};

// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Compute confidence scores (F32)
// Input: hidden_states[batch * seq_len * hidden_dim], classifier_weight[hidden_dim]
// Output: confidence[batch * seq_len * num_layers]
kernel void spec_ee_compute_confidence_f32(
    device const float* hidden_states [[buffer(0)]],
    device const float* classifier_weight [[buffer(1)]],
    device float* confidence [[buffer(2)]],
    constant ComputeConfidenceParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.num_layers;
    if (tid >= total) return;
    
    uint layer_stride = params.batch_size * params.seq_len;
    uint layer_idx = tid / layer_stride;
    uint bs_idx = tid % layer_stride;
    
    uint hidden_base = bs_idx * params.hidden_dim;
    uint weight_base = layer_idx * params.hidden_dim;
    
    float sum = 0.0f;
    for (uint d = 0; d < params.hidden_dim; d++) {
        sum += hidden_states[hidden_base + d] * classifier_weight[weight_base + d];
    }
    
    float scaled = sum / max(params.temperature, 0.01f);
    confidence[tid] = sigmoid(scaled);
}

// Compute confidence scores (F16)
kernel void spec_ee_compute_confidence_f16(
    device const half* hidden_states [[buffer(0)]],
    device const half* classifier_weight [[buffer(1)]],
    device half* confidence [[buffer(2)]],
    constant ComputeConfidenceParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.num_layers;
    if (tid >= total) return;
    
    uint layer_stride = params.batch_size * params.seq_len;
    uint layer_idx = tid / layer_stride;
    uint bs_idx = tid % layer_stride;
    
    uint hidden_base = bs_idx * params.hidden_dim;
    uint weight_base = layer_idx * params.hidden_dim;
    
    float sum = 0.0f;
    for (uint d = 0; d < params.hidden_dim; d++) {
        sum += float(hidden_states[hidden_base + d]) * float(classifier_weight[weight_base + d]);
    }
    
    float scaled = sum / max(params.temperature, 0.01f);
    confidence[tid] = half(sigmoid(scaled));
}

// Layer skip decision (F32)
// Input: confidence_scores[batch * seq_len * num_layers]
// Output: skip_mask[batch * num_layers] (1 = skip, 0 = execute)
kernel void spec_ee_layer_skip_decision_f32(
    device const float* confidence_scores [[buffer(0)]],
    device uint* skip_mask [[buffer(1)]],
    constant LayerSkipDecisionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    uint batch_idx = tid;
    uint skipped = 0;
    
    for (uint layer = 0; layer < params.num_layers; layer++) {
        uint out_idx = batch_idx * params.num_layers + layer;
        
        if (layer < params.min_layers) {
            skip_mask[out_idx] = 0;
            continue;
        }
        
        if (skipped >= params.max_skip) {
            skip_mask[out_idx] = 0;
            continue;
        }
        
        float avg_conf = 0.0f;
        for (uint s = 0; s < params.seq_len; s++) {
            uint conf_idx = layer * params.batch_size * params.seq_len + batch_idx * params.seq_len + s;
            avg_conf += confidence_scores[conf_idx];
        }
        avg_conf /= float(max(params.seq_len, 1u));
        
        if (avg_conf >= params.confidence_threshold) {
            skip_mask[out_idx] = 1;
            skipped++;
        } else {
            skip_mask[out_idx] = 0;
        }
    }
}

// Layer skip decision (F16)
kernel void spec_ee_layer_skip_decision_f16(
    device const half* confidence_scores [[buffer(0)]],
    device uint* skip_mask [[buffer(1)]],
    constant LayerSkipDecisionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    uint batch_idx = tid;
    uint skipped = 0;
    
    for (uint layer = 0; layer < params.num_layers; layer++) {
        uint out_idx = batch_idx * params.num_layers + layer;
        
        if (layer < params.min_layers) {
            skip_mask[out_idx] = 0;
            continue;
        }
        
        if (skipped >= params.max_skip) {
            skip_mask[out_idx] = 0;
            continue;
        }
        
        float avg_conf = 0.0f;
        for (uint s = 0; s < params.seq_len; s++) {
            uint conf_idx = layer * params.batch_size * params.seq_len + batch_idx * params.seq_len + s;
            avg_conf += float(confidence_scores[conf_idx]);
        }
        avg_conf /= float(max(params.seq_len, 1u));
        
        if (avg_conf >= params.confidence_threshold) {
            skip_mask[out_idx] = 1;
            skipped++;
        } else {
            skip_mask[out_idx] = 0;
        }
    }
}

// Early exit LM head projection (F32)
// Input: hidden_states[batch * seq_len * hidden_dim], lm_weight[hidden_dim * vocab_size]
// Output: logits[batch * seq_len * vocab_size], exit_flags[batch * seq_len]
kernel void spec_ee_early_exit_f32(
    device const float* hidden_states [[buffer(0)]],
    device const float* lm_weight [[buffer(1)]],
    device float* logits [[buffer(2)]],
    device uint* exit_flags [[buffer(3)]],
    constant EarlyExitParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.vocab_size;
    if (tid >= total) return;
    
    uint vocab_stride = params.vocab_size;
    uint bs_idx = tid / vocab_stride;
    uint v = tid % vocab_stride;
    
    uint hidden_base = bs_idx * params.hidden_dim;
    
    float sum = 0.0f;
    for (uint d = 0; d < params.hidden_dim; d++) {
        sum += hidden_states[hidden_base + d] * lm_weight[d * params.vocab_size + v];
    }
    
    logits[tid] = sum;
    
    if (v == 0) {
        exit_flags[bs_idx] = (params.current_layer >= params.total_layers - 1) ? 1 : 0;
    }
}

// Early exit LM head projection (F16)
kernel void spec_ee_early_exit_f16(
    device const half* hidden_states [[buffer(0)]],
    device const half* lm_weight [[buffer(1)]],
    device half* logits [[buffer(2)]],
    device uint* exit_flags [[buffer(3)]],
    constant EarlyExitParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.batch_size * params.seq_len * params.vocab_size;
    if (tid >= total) return;
    
    uint vocab_stride = params.vocab_size;
    uint bs_idx = tid / vocab_stride;
    uint v = tid % vocab_stride;
    
    uint hidden_base = bs_idx * params.hidden_dim;
    
    float sum = 0.0f;
    for (uint d = 0; d < params.hidden_dim; d++) {
        sum += float(hidden_states[hidden_base + d]) * float(lm_weight[d * params.vocab_size + v]);
    }
    
    logits[tid] = half(sum);
    
    if (v == 0) {
        exit_flags[bs_idx] = (params.current_layer >= params.total_layers - 1) ? 1 : 0;
    }
}
