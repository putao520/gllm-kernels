// SpecEE / LayerSkip Early-Exit Speculative Decoding
// Based on SpecEE (ISCA'25) and LayerSkip (ACL'24)
//
// Kernels:
// - spec_ee_compute_confidence_f32: Per-layer early exit confidence
// - spec_ee_lm_head_f32: Early exit LM head projection
// - spec_ee_compute_confidence_f16: F16 variant
// - spec_ee_lm_head_f16: F16 variant

enable f16;

const WORKGROUP_SIZE: u32 = 256u;
const MAX_VOCAB_TILE: u32 = 1024u;

// ============================================================================
// Parameters
// ============================================================================

struct ConfidenceParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    layer_idx: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct LMHeadParams {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    vocab_size: u32,
    temperature: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// ============================================================================
// F32 Early Exit Confidence Kernel
// ============================================================================
// Computes layer-level confidence for early exit decision
// Input: hidden[batch, seq_len, hidden_dim], conf_weights[hidden_dim]
// Output: confidence[batch, seq_len], should_exit[batch, seq_len]

@group(0) @binding(0) var<storage, read> ee_hidden_f32: array<f32>;
@group(0) @binding(1) var<storage, read> ee_conf_weights_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> ee_confidence_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> ee_should_exit_f32: array<u32>;
@group(0) @binding(4) var<uniform> ee_params_f32: ConfidenceParams;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn spec_ee_compute_confidence_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = ee_params_f32.batch_size * ee_params_f32.seq_len;

    if (idx >= total) {
        return;
    }

    // Compute confidence via dot product
    let hidden_base = idx * ee_params_f32.hidden_dim;
    var logit = 0.0f;

    for (var d: u32 = 0u; d < ee_params_f32.hidden_dim; d = d + 1u) {
        logit += ee_hidden_f32[hidden_base + d] * ee_conf_weights_f32[d];
    }

    let confidence = sigmoid(logit);
    ee_confidence_f32[idx] = confidence;

    // Check if should exit
    if (confidence >= ee_params_f32.threshold) {
        ee_should_exit_f32[idx] = 1u;
    } else {
        ee_should_exit_f32[idx] = 0u;
    }
}

// ============================================================================
// F32 LM Head Projection Kernel
// ============================================================================
// Computes logits from hidden states for early exit
// Input: hidden[batch, seq_len, hidden_dim], lm_weights[hidden_dim, vocab_size]
// Output: logits[batch, seq_len, vocab_size]

@group(0) @binding(0) var<storage, read> lm_hidden_f32: array<f32>;
@group(0) @binding(1) var<storage, read> lm_weights_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> lm_logits_f32: array<f32>;
@group(0) @binding(3) var<uniform> lm_params_f32: LMHeadParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn spec_ee_lm_head_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = lm_params_f32.batch_size * lm_params_f32.seq_len * lm_params_f32.vocab_size;

    if (idx >= total) {
        return;
    }

    let vocab_stride = lm_params_f32.vocab_size;
    let batch_seq_idx = idx / vocab_stride;
    let vocab_idx = idx % vocab_stride;

    let hidden_base = batch_seq_idx * lm_params_f32.hidden_dim;

    // Compute dot product: hidden[b,s,:] @ weights[:,v]
    var logit = 0.0f;
    for (var d: u32 = 0u; d < lm_params_f32.hidden_dim; d = d + 1u) {
        let weight_idx = d * lm_params_f32.vocab_size + vocab_idx;
        logit += lm_hidden_f32[hidden_base + d] * lm_weights_f32[weight_idx];
    }

    // Apply temperature if non-zero
    if (lm_params_f32.temperature > 0.0) {
        logit = logit / lm_params_f32.temperature;
    }

    lm_logits_f32[idx] = logit;
}

// ============================================================================
// F32 Argmax Kernel (for greedy decoding)
// ============================================================================
// Finds argmax of logits along vocab dimension
// Input: logits[batch, seq_len, vocab_size]
// Output: tokens[batch, seq_len]

@group(0) @binding(0) var<storage, read> argmax_logits_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> argmax_tokens_f32: array<u32>;
@group(0) @binding(2) var<uniform> argmax_params_f32: LMHeadParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn spec_ee_argmax_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = argmax_params_f32.batch_size * argmax_params_f32.seq_len;

    if (idx >= total) {
        return;
    }

    let logits_base = idx * argmax_params_f32.vocab_size;

    var max_val = argmax_logits_f32[logits_base];
    var max_idx = 0u;

    for (var v: u32 = 1u; v < argmax_params_f32.vocab_size; v = v + 1u) {
        let val = argmax_logits_f32[logits_base + v];
        if (val > max_val) {
            max_val = val;
            max_idx = v;
        }
    }

    argmax_tokens_f32[idx] = max_idx;
}

// ============================================================================
// F16 Early Exit Confidence Kernel
// ============================================================================

@group(0) @binding(0) var<storage, read> ee_hidden_f16: array<f16>;
@group(0) @binding(1) var<storage, read> ee_conf_weights_f16: array<f16>;
@group(0) @binding(2) var<storage, read_write> ee_confidence_f16: array<f16>;
@group(0) @binding(3) var<storage, read_write> ee_should_exit_f16: array<u32>;
@group(0) @binding(4) var<uniform> ee_params_f16: ConfidenceParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn spec_ee_compute_confidence_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = ee_params_f16.batch_size * ee_params_f16.seq_len;

    if (idx >= total) {
        return;
    }

    let hidden_base = idx * ee_params_f16.hidden_dim;
    var logit = 0.0f;

    // Accumulate in f32 for precision
    for (var d: u32 = 0u; d < ee_params_f16.hidden_dim; d = d + 1u) {
        logit += f32(ee_hidden_f16[hidden_base + d]) * f32(ee_conf_weights_f16[d]);
    }

    let confidence = sigmoid(logit);
    ee_confidence_f16[idx] = f16(confidence);

    if (confidence >= ee_params_f16.threshold) {
        ee_should_exit_f16[idx] = 1u;
    } else {
        ee_should_exit_f16[idx] = 0u;
    }
}

// ============================================================================
// F16 LM Head Projection Kernel
// ============================================================================

@group(0) @binding(0) var<storage, read> lm_hidden_f16: array<f16>;
@group(0) @binding(1) var<storage, read> lm_weights_f16: array<f16>;
@group(0) @binding(2) var<storage, read_write> lm_logits_f16: array<f16>;
@group(0) @binding(3) var<uniform> lm_params_f16: LMHeadParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn spec_ee_lm_head_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = lm_params_f16.batch_size * lm_params_f16.seq_len * lm_params_f16.vocab_size;

    if (idx >= total) {
        return;
    }

    let vocab_stride = lm_params_f16.vocab_size;
    let batch_seq_idx = idx / vocab_stride;
    let vocab_idx = idx % vocab_stride;

    let hidden_base = batch_seq_idx * lm_params_f16.hidden_dim;

    var logit = 0.0f;
    for (var d: u32 = 0u; d < lm_params_f16.hidden_dim; d = d + 1u) {
        let weight_idx = d * lm_params_f16.vocab_size + vocab_idx;
        logit += f32(lm_hidden_f16[hidden_base + d]) * f32(lm_weights_f16[weight_idx]);
    }

    if (lm_params_f16.temperature > 0.0) {
        logit = logit / lm_params_f16.temperature;
    }

    lm_logits_f16[idx] = f16(logit);
}
