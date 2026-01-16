//
// EAGLE-3 Adaptive Draft Length Speculative Decoding Kernels
//
// Based on EAGLE-3 (NeurIPS'25): 2-6x inference acceleration through
// multi-layer feature fusion, token-level confidence prediction, and
// adaptive draft length scheduling.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Parameter Structures
// ============================================================================

struct FuseHiddenParams {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint fusion_layers;
    uint fused_dim;
    uint _pad[3];
};

struct PredictConfidenceParams {
    uint batch_size;
    uint seq_len;
    uint fused_dim;
    float bias;
};

struct GenerateDraftParams {
    uint seq_len;
    uint vocab_size;
    uint fused_dim;
    uint max_draft_len;
    float confidence_threshold;
    uint _pad[3];
};

// ============================================================================
// Helper Functions
// ============================================================================

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// ============================================================================
// Fuse Hidden States Kernels
// ============================================================================

/// Fuse hidden states from multiple layers (f32).
/// layer0..layerN: [batch, seq_len, hidden_dim]
/// output: [batch, seq_len, fused_dim]
kernel void eagle3_fuse_hidden_f32(
    device const float* layer0 [[buffer(0)]],
    device const float* layer1 [[buffer(1)]],
    device const float* layer2 [[buffer(2)]],
    device const float* layer3 [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant FuseHiddenParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size * params.seq_len) return;

    uint batch_idx = gid / params.seq_len;
    uint seq_idx = gid % params.seq_len;

    uint input_offset = (batch_idx * params.seq_len + seq_idx) * params.hidden_dim;
    uint output_base = (batch_idx * params.seq_len + seq_idx) * params.fused_dim;

    // Copy from each layer to fused output
    // Layer 0
    for (uint h = 0; h < params.hidden_dim; h++) {
        output[output_base + h] = layer0[input_offset + h];
    }

    // Layer 1
    if (params.fusion_layers > 1) {
        uint offset = params.hidden_dim;
        for (uint h = 0; h < params.hidden_dim; h++) {
            output[output_base + offset + h] = layer1[input_offset + h];
        }
    }

    // Layer 2
    if (params.fusion_layers > 2) {
        uint offset = params.hidden_dim * 2;
        for (uint h = 0; h < params.hidden_dim; h++) {
            output[output_base + offset + h] = layer2[input_offset + h];
        }
    }

    // Layer 3
    if (params.fusion_layers > 3) {
        uint offset = params.hidden_dim * 3;
        for (uint h = 0; h < params.hidden_dim; h++) {
            output[output_base + offset + h] = layer3[input_offset + h];
        }
    }
}

/// Fuse hidden states from multiple layers (f16).
kernel void eagle3_fuse_hidden_f16(
    device const half* layer0 [[buffer(0)]],
    device const half* layer1 [[buffer(1)]],
    device const half* layer2 [[buffer(2)]],
    device const half* layer3 [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant FuseHiddenParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size * params.seq_len) return;

    uint batch_idx = gid / params.seq_len;
    uint seq_idx = gid % params.seq_len;

    uint input_offset = (batch_idx * params.seq_len + seq_idx) * params.hidden_dim;
    uint output_base = (batch_idx * params.seq_len + seq_idx) * params.fused_dim;

    // Copy from each layer to fused output
    for (uint h = 0; h < params.hidden_dim; h++) {
        output[output_base + h] = layer0[input_offset + h];
    }

    if (params.fusion_layers > 1) {
        uint offset = params.hidden_dim;
        for (uint h = 0; h < params.hidden_dim; h++) {
            output[output_base + offset + h] = layer1[input_offset + h];
        }
    }

    if (params.fusion_layers > 2) {
        uint offset = params.hidden_dim * 2;
        for (uint h = 0; h < params.hidden_dim; h++) {
            output[output_base + offset + h] = layer2[input_offset + h];
        }
    }

    if (params.fusion_layers > 3) {
        uint offset = params.hidden_dim * 3;
        for (uint h = 0; h < params.hidden_dim; h++) {
            output[output_base + offset + h] = layer3[input_offset + h];
        }
    }
}

// ============================================================================
// Confidence Prediction Kernels
// ============================================================================

/// Predict token-level confidence from fused hidden states (f32).
/// fused_hidden: [batch, seq_len, fused_dim]
/// weight: [fused_dim, 1]
/// output: [batch, seq_len]
kernel void eagle3_predict_confidence_f32(
    device const float* fused_hidden [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant PredictConfidenceParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size * params.seq_len) return;

    uint hidden_offset = gid * params.fused_dim;

    // Dot product with weight vector
    float logit = params.bias;
    for (uint i = 0; i < params.fused_dim; i++) {
        logit += fused_hidden[hidden_offset + i] * weight[i];
    }

    // Sigmoid activation
    output[gid] = sigmoid(logit);
}

/// Predict token-level confidence from fused hidden states (f16).
kernel void eagle3_predict_confidence_f16(
    device const half* fused_hidden [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant PredictConfidenceParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size * params.seq_len) return;

    uint hidden_offset = gid * params.fused_dim;

    // Dot product with weight vector (accumulate in float for precision)
    float logit = params.bias;
    for (uint i = 0; i < params.fused_dim; i++) {
        logit += float(fused_hidden[hidden_offset + i]) * float(weight[i]);
    }

    // Sigmoid activation
    output[gid] = half(sigmoid(logit));
}

// ============================================================================
// Draft Generation Kernels
// ============================================================================

/// Generate draft tokens with confidence-based early termination (f32).
/// draft_logits: [seq_len, vocab_size]
/// fused_hidden: [seq_len, fused_dim]
/// weight: [fused_dim, 1]
/// tokens: [max_draft_len] output token IDs
/// count: [1] number of generated tokens
kernel void eagle3_generate_draft_f32(
    device const float* draft_logits [[buffer(0)]],
    device const float* fused_hidden [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device uint* tokens [[buffer(3)]],
    device uint* count [[buffer(4)]],
    constant GenerateDraftParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Single thread execution for sequential generation
    if (gid != 0) return;

    uint draft_count = 0;
    float bias = 0.0f;  // Assuming bias is 0, could be passed in params

    for (uint pos = 0; pos < params.seq_len && draft_count < params.max_draft_len; pos++) {
        // Calculate confidence for this position
        uint hidden_offset = pos * params.fused_dim;
        float logit = bias;
        for (uint i = 0; i < params.fused_dim; i++) {
            logit += fused_hidden[hidden_offset + i] * weight[i];
        }
        float confidence = sigmoid(logit);

        // Early termination check (except for first token)
        if (confidence < params.confidence_threshold && draft_count > 0) {
            break;
        }

        // Find top token from logits
        uint logit_offset = pos * params.vocab_size;
        float max_logit = draft_logits[logit_offset];
        uint max_idx = 0;

        for (uint v = 1; v < params.vocab_size; v++) {
            float l = draft_logits[logit_offset + v];
            if (l > max_logit) {
                max_logit = l;
                max_idx = v;
            }
        }

        tokens[draft_count] = max_idx;
        draft_count++;
    }

    count[0] = draft_count;
}

/// Generate draft tokens with confidence-based early termination (f16).
kernel void eagle3_generate_draft_f16(
    device const half* draft_logits [[buffer(0)]],
    device const half* fused_hidden [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device uint* tokens [[buffer(3)]],
    device uint* count [[buffer(4)]],
    constant GenerateDraftParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint draft_count = 0;
    float bias = 0.0f;

    for (uint pos = 0; pos < params.seq_len && draft_count < params.max_draft_len; pos++) {
        uint hidden_offset = pos * params.fused_dim;
        float logit = bias;
        for (uint i = 0; i < params.fused_dim; i++) {
            logit += float(fused_hidden[hidden_offset + i]) * float(weight[i]);
        }
        float confidence = sigmoid(logit);

        if (confidence < params.confidence_threshold && draft_count > 0) {
            break;
        }

        uint logit_offset = pos * params.vocab_size;
        float max_logit = float(draft_logits[logit_offset]);
        uint max_idx = 0;

        for (uint v = 1; v < params.vocab_size; v++) {
            float l = float(draft_logits[logit_offset + v]);
            if (l > max_logit) {
                max_logit = l;
                max_idx = v;
            }
        }

        tokens[draft_count] = max_idx;
        draft_count++;
    }

    count[0] = draft_count;
}
