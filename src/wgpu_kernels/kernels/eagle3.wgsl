// EAGLE-3 Adaptive Draft Length Speculative Decoding
// Based on EAGLE-3 (NeurIPS'25): Multi-layer feature fusion + token-level confidence
//
// Kernels:
// - eagle3_fuse_layers_f32: Fuse multiple layer hidden states
// - eagle3_predict_confidence_f32: Compute token-level acceptance probability
// - eagle3_fuse_layers_f16: F16 variant for layer fusion
// - eagle3_predict_confidence_f16: F16 variant for confidence prediction

enable f16;

const WORKGROUP_SIZE: u32 = 256u;
const MAX_FUSION_LAYERS: u32 = 8u;

// ============================================================================
// Parameters
// ============================================================================

struct FusionParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    num_layers: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

struct ConfidenceParams {
    batch_size: u32,
    seq_len: u32,
    fused_dim: u32,
    bias: f32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// ============================================================================
// F32 Layer Fusion Kernel
// ============================================================================
// Fuses hidden states from multiple layers into a single representation
// Input: hidden_states[batch, seq_len, hidden_dim] for each layer
// Output: fused[batch, seq_len, hidden_dim * num_layers]

@group(0) @binding(0) var<storage, read> fusion_input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> fusion_output_f32: array<f32>;
@group(0) @binding(2) var<uniform> fusion_params_f32: FusionParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn eagle3_fuse_layers_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_elements = fusion_params_f32.batch_size * fusion_params_f32.seq_len * fusion_params_f32.hidden_dim * fusion_params_f32.num_layers;

    if (idx >= total_elements) {
        return;
    }

    let fused_dim = fusion_params_f32.hidden_dim * fusion_params_f32.num_layers;
    let batch_seq_idx = idx / fused_dim;
    let fused_offset = idx % fused_dim;
    let layer_idx = fused_offset / fusion_params_f32.hidden_dim;
    let dim_idx = fused_offset % fusion_params_f32.hidden_dim;

    // Input layout: [num_layers, batch, seq_len, hidden_dim]
    let input_layer_stride = fusion_params_f32.batch_size * fusion_params_f32.seq_len * fusion_params_f32.hidden_dim;
    let input_idx = layer_idx * input_layer_stride + batch_seq_idx * fusion_params_f32.hidden_dim + dim_idx;

    fusion_output_f32[idx] = fusion_input_f32[input_idx];
}

// ============================================================================
// F32 Confidence Prediction Kernel
// ============================================================================
// Computes token-level acceptance probability using fused hidden states
// Input: fused[batch, seq_len, fused_dim], weights[fused_dim]
// Output: confidence[batch, seq_len]

@group(0) @binding(0) var<storage, read> conf_fused_f32: array<f32>;
@group(0) @binding(1) var<storage, read> conf_weights_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> conf_output_f32: array<f32>;
@group(0) @binding(3) var<uniform> conf_params_f32: ConfidenceParams;

fn sigmoid_f32(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn eagle3_predict_confidence_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_outputs = conf_params_f32.batch_size * conf_params_f32.seq_len;

    if (idx >= total_outputs) {
        return;
    }

    // Compute dot product of fused hidden state with weights
    let fused_base = idx * conf_params_f32.fused_dim;
    var logit = conf_params_f32.bias;

    for (var d: u32 = 0u; d < conf_params_f32.fused_dim; d = d + 1u) {
        logit += conf_fused_f32[fused_base + d] * conf_weights_f32[d];
    }

    // Apply sigmoid to get probability
    let prob = sigmoid_f32(logit);
    conf_output_f32[idx] = prob;
}

// ============================================================================
// F32 Early Termination Check Kernel
// ============================================================================
// Checks if confidence drops below threshold for early termination
// Input: confidence[batch, seq_len], threshold
// Output: terminate_idx[batch] - index where to terminate, or seq_len if none

@group(0) @binding(0) var<storage, read> term_confidence_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> term_output_f32: array<u32>;
@group(0) @binding(2) var<uniform> term_params_f32: ConfidenceParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn eagle3_check_termination_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let batch_idx = global_id.x;

    if (batch_idx >= term_params_f32.batch_size) {
        return;
    }

    let base_idx = batch_idx * term_params_f32.seq_len;
    var terminate_at = term_params_f32.seq_len;

    for (var s: u32 = 0u; s < term_params_f32.seq_len; s = s + 1u) {
        let conf = term_confidence_f32[base_idx + s];
        if (conf < term_params_f32.threshold) {
            terminate_at = s;
            break;
        }
    }

    term_output_f32[batch_idx] = terminate_at;
}

// ============================================================================
// F16 Layer Fusion Kernel
// ============================================================================

@group(0) @binding(0) var<storage, read> fusion_input_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write> fusion_output_f16: array<f16>;
@group(0) @binding(2) var<uniform> fusion_params_f16: FusionParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn eagle3_fuse_layers_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_elements = fusion_params_f16.batch_size * fusion_params_f16.seq_len * fusion_params_f16.hidden_dim * fusion_params_f16.num_layers;

    if (idx >= total_elements) {
        return;
    }

    let fused_dim = fusion_params_f16.hidden_dim * fusion_params_f16.num_layers;
    let batch_seq_idx = idx / fused_dim;
    let fused_offset = idx % fused_dim;
    let layer_idx = fused_offset / fusion_params_f16.hidden_dim;
    let dim_idx = fused_offset % fusion_params_f16.hidden_dim;

    let input_layer_stride = fusion_params_f16.batch_size * fusion_params_f16.seq_len * fusion_params_f16.hidden_dim;
    let input_idx = layer_idx * input_layer_stride + batch_seq_idx * fusion_params_f16.hidden_dim + dim_idx;

    fusion_output_f16[idx] = fusion_input_f16[input_idx];
}

// ============================================================================
// F16 Confidence Prediction Kernel
// ============================================================================

@group(0) @binding(0) var<storage, read> conf_fused_f16: array<f16>;
@group(0) @binding(1) var<storage, read> conf_weights_f16: array<f16>;
@group(0) @binding(2) var<storage, read_write> conf_output_f16: array<f16>;
@group(0) @binding(3) var<uniform> conf_params_f16: ConfidenceParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn eagle3_predict_confidence_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_outputs = conf_params_f16.batch_size * conf_params_f16.seq_len;

    if (idx >= total_outputs) {
        return;
    }

    let fused_base = idx * conf_params_f16.fused_dim;
    var logit = conf_params_f16.bias;

    // Accumulate in f32 for precision
    for (var d: u32 = 0u; d < conf_params_f16.fused_dim; d = d + 1u) {
        logit += f32(conf_fused_f16[fused_base + d]) * f32(conf_weights_f16[d]);
    }

    let prob = sigmoid_f32(logit);
    conf_output_f16[idx] = f16(prob);
}
