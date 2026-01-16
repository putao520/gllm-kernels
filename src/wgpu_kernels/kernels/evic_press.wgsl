// EvicPress Joint Compression and Eviction Kernels
// Based on KVPress (EMNLP'24): Three-zone progressive KV cache management
//
// Kernels:
// - evicpress_compute_importance_f32: Calculate token importance scores
// - evicpress_quantize_fp16_to_int8_f32: Zone transition Hot→Warm
// - evicpress_quantize_int8_to_int2_f32: Zone transition Warm→Cold
// - evicpress_dequantize_int8_f32: Warm zone dequantization
// - evicpress_dequantize_int2_f32: Cold zone dequantization
// - F16 variants for all kernels

enable f16;

const WORKGROUP_SIZE: u32 = 256u;
const INT8_MIN: f32 = -128.0;
const INT8_MAX: f32 = 127.0;
const INT2_MIN: f32 = 0.0;
const INT2_MAX: f32 = 3.0;

// ============================================================================
// Parameters
// ============================================================================

struct ImportanceParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    attention_weight: f32,
    semantic_weight: f32,
    recency_weight: f32,
    _pad0: u32,
};

struct ZoneTransitionParams {
    num_elements: u32,
    group_size: u32,
    num_groups: u32,
    _pad0: u32,
};

// ============================================================================
// Importance Score Computation (F32)
// ============================================================================
// Computes combined importance score for each token
// Input: attention_scores[batch, heads, seq], positions[batch, seq]
// Output: importance[batch, seq]

@group(0) @binding(0) var<storage, read> imp_attention_f32: array<f32>;
@group(0) @binding(1) var<storage, read> imp_positions_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> imp_output_f32: array<f32>;
@group(0) @binding(3) var<uniform> imp_params_f32: ImportanceParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn evicpress_compute_importance_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = imp_params_f32.batch_size * imp_params_f32.seq_len;

    if (idx >= total) {
        return;
    }

    let batch_idx = idx / imp_params_f32.seq_len;
    let seq_idx = idx % imp_params_f32.seq_len;

    // Average attention score across heads
    var attention_sum = 0.0f;
    let head_stride = imp_params_f32.seq_len;
    let batch_stride = imp_params_f32.num_heads * head_stride;

    for (var h: u32 = 0u; h < imp_params_f32.num_heads; h++) {
        let attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        attention_sum += imp_attention_f32[attn_idx];
    }
    let attention_score = attention_sum / f32(imp_params_f32.num_heads);

    // Position-based recency (newer = higher score)
    let position = imp_positions_f32[idx];
    let max_pos = imp_params_f32.seq_len;
    let recency_score = f32(position) / f32(max_pos);

    // Semantic importance (could be enhanced with actual semantic features)
    // For now, use attention variance as proxy
    var attn_var = 0.0f;
    for (var h: u32 = 0u; h < imp_params_f32.num_heads; h++) {
        let attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        let diff = imp_attention_f32[attn_idx] - attention_score;
        attn_var += diff * diff;
    }
    let semantic_score = 1.0 - sqrt(attn_var / f32(imp_params_f32.num_heads));

    // Combined importance score
    let combined = imp_params_f32.attention_weight * attention_score +
                   imp_params_f32.semantic_weight * semantic_score +
                   imp_params_f32.recency_weight * recency_score;

    imp_output_f32[idx] = combined;
}

// ============================================================================
// FP16 → INT8 Quantization (Hot → Warm zone transition)
// ============================================================================
// Quantizes FP16 values to INT8 with group-wise scales

@group(0) @binding(0) var<storage, read> zone_hot_input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> zone_warm_output_f32: array<i32>;
@group(0) @binding(2) var<storage, read_write> zone_warm_scales_f32: array<f32>;
@group(0) @binding(3) var<uniform> zone_trans_params_f32: ZoneTransitionParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn evicpress_quantize_fp16_to_int8_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let group_idx = wg_id.x;
    let local_idx = local_id.x;

    if (group_idx >= zone_trans_params_f32.num_groups) {
        return;
    }

    let group_start = group_idx * zone_trans_params_f32.group_size;
    let group_end = min(group_start + zone_trans_params_f32.group_size, zone_trans_params_f32.num_elements);

    // Phase 1: Find abs max in group
    if (local_idx == 0u) {
        var abs_max = 0.0f;
        for (var j = group_start; j < group_end; j++) {
            abs_max = max(abs_max, abs(zone_hot_input_f32[j]));
        }

        // Compute scale (symmetric quantization)
        let scale = select(abs_max / 127.0, 1.0, abs_max < 1e-8);
        zone_warm_scales_f32[group_idx] = scale;
    }

    workgroupBarrier();

    // Phase 2: Quantize values
    let scale = zone_warm_scales_f32[group_idx];
    let inv_scale = select(1.0 / scale, 0.0, scale < 1e-8);

    var i = group_start + local_idx;
    while (i < group_end) {
        let val = zone_hot_input_f32[i];
        let scaled = val * inv_scale;
        let clamped = clamp(scaled, INT8_MIN, INT8_MAX);
        zone_warm_output_f32[i] = i32(round(clamped));
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// INT8 → INT2 Quantization (Warm → Cold zone transition)
// ============================================================================
// Further quantizes INT8 to INT2 with group-wise scales

@group(0) @binding(0) var<storage, read> zone_warm_input_f32: array<i32>;
@group(0) @binding(1) var<storage, read_write> zone_cold_output_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> zone_cold_scales_f32: array<f32>;
@group(0) @binding(3) var<storage, read> zone_warm_scales_read_f32: array<f32>;
@group(0) @binding(4) var<uniform> zone_cold_params_f32: ZoneTransitionParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn evicpress_quantize_int8_to_int2_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let group_idx = wg_id.x;
    let local_idx = local_id.x;

    if (group_idx >= zone_cold_params_f32.num_groups) {
        return;
    }

    let group_start = group_idx * zone_cold_params_f32.group_size;
    let group_end = min(group_start + zone_cold_params_f32.group_size, zone_cold_params_f32.num_elements);

    // Phase 1: Find min/max in group
    if (local_idx == 0u) {
        var g_min = 127.0f;
        var g_max = -128.0f;
        for (var j = group_start; j < group_end; j++) {
            let val = f32(zone_warm_input_f32[j]);
            g_min = min(g_min, val);
            g_max = max(g_max, val);
        }

        // Compute scale for INT2 (asymmetric)
        let range = g_max - g_min;
        let scale = select(range / INT2_MAX, 1.0, range < 1e-8);
        // Store combined scale (warm_scale * cold_scale)
        let warm_scale = zone_warm_scales_read_f32[group_idx];
        zone_cold_scales_f32[group_idx] = warm_scale * scale;
    }

    workgroupBarrier();

    // Phase 2: Quantize to INT2
    var i = group_start + local_idx;
    while (i < group_end) {
        let val = f32(zone_warm_input_f32[i]);
        // Map -128..127 to 0..3
        let normalized = (val + 128.0) / 255.0 * 3.0;
        let clamped = clamp(normalized, INT2_MIN, INT2_MAX);
        zone_cold_output_f32[i] = u32(round(clamped));
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// INT8 Dequantization (Warm zone read)
// ============================================================================

@group(0) @binding(0) var<storage, read> dequant_warm_input_f32: array<i32>;
@group(0) @binding(1) var<storage, read> dequant_warm_scales_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> dequant_warm_output_f32: array<f32>;
@group(0) @binding(3) var<uniform> dequant_warm_params_f32: ZoneTransitionParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn evicpress_dequantize_int8_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;

    if (idx >= dequant_warm_params_f32.num_elements) {
        return;
    }

    let group_idx = idx / dequant_warm_params_f32.group_size;
    let scale = dequant_warm_scales_f32[group_idx];

    let quantized = f32(dequant_warm_input_f32[idx]);
    dequant_warm_output_f32[idx] = quantized * scale;
}

// ============================================================================
// INT2 Dequantization (Cold zone read)
// ============================================================================

@group(0) @binding(0) var<storage, read> dequant_cold_input_f32: array<u32>;
@group(0) @binding(1) var<storage, read> dequant_cold_scales_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> dequant_cold_output_f32: array<f32>;
@group(0) @binding(3) var<uniform> dequant_cold_params_f32: ZoneTransitionParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn evicpress_dequantize_int2_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;

    if (idx >= dequant_cold_params_f32.num_elements) {
        return;
    }

    let group_idx = idx / dequant_cold_params_f32.group_size;
    let scale = dequant_cold_scales_f32[group_idx];

    // Map 0..3 back to -128..127, then apply scale
    let quantized = f32(dequant_cold_input_f32[idx]);
    let denormalized = (quantized / 3.0 * 255.0) - 128.0;
    dequant_cold_output_f32[idx] = denormalized * scale;
}

// ============================================================================
// F16 Importance Score Computation
// ============================================================================

@group(0) @binding(0) var<storage, read> imp_attention_f16: array<f16>;
@group(0) @binding(1) var<storage, read> imp_positions_f16: array<u32>;
@group(0) @binding(2) var<storage, read_write> imp_output_f16: array<f16>;
@group(0) @binding(3) var<uniform> imp_params_f16: ImportanceParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn evicpress_compute_importance_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total = imp_params_f16.batch_size * imp_params_f16.seq_len;

    if (idx >= total) {
        return;
    }

    let batch_idx = idx / imp_params_f16.seq_len;
    let seq_idx = idx % imp_params_f16.seq_len;

    // Compute in f32 for precision
    var attention_sum = 0.0f;
    let head_stride = imp_params_f16.seq_len;
    let batch_stride = imp_params_f16.num_heads * head_stride;

    for (var h: u32 = 0u; h < imp_params_f16.num_heads; h++) {
        let attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        attention_sum += f32(imp_attention_f16[attn_idx]);
    }
    let attention_score = attention_sum / f32(imp_params_f16.num_heads);

    let position = imp_positions_f16[idx];
    let recency_score = f32(position) / f32(imp_params_f16.seq_len);

    var attn_var = 0.0f;
    for (var h: u32 = 0u; h < imp_params_f16.num_heads; h++) {
        let attn_idx = batch_idx * batch_stride + h * head_stride + seq_idx;
        let diff = f32(imp_attention_f16[attn_idx]) - attention_score;
        attn_var += diff * diff;
    }
    let semantic_score = 1.0 - sqrt(attn_var / f32(imp_params_f16.num_heads));

    let combined = imp_params_f16.attention_weight * attention_score +
                   imp_params_f16.semantic_weight * semantic_score +
                   imp_params_f16.recency_weight * recency_score;

    imp_output_f16[idx] = f16(combined);
}

// ============================================================================
// F16 Zone Transitions
// ============================================================================

@group(0) @binding(0) var<storage, read> zone_hot_input_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write> zone_warm_output_f16: array<i32>;
@group(0) @binding(2) var<storage, read_write> zone_warm_scales_f16: array<f16>;
@group(0) @binding(3) var<uniform> zone_trans_params_f16: ZoneTransitionParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn evicpress_quantize_fp16_to_int8_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let group_idx = wg_id.x;
    let local_idx = local_id.x;

    if (group_idx >= zone_trans_params_f16.num_groups) {
        return;
    }

    let group_start = group_idx * zone_trans_params_f16.group_size;
    let group_end = min(group_start + zone_trans_params_f16.group_size, zone_trans_params_f16.num_elements);

    if (local_idx == 0u) {
        var abs_max = 0.0f;
        for (var j = group_start; j < group_end; j++) {
            abs_max = max(abs_max, abs(f32(zone_hot_input_f16[j])));
        }

        let scale = select(abs_max / 127.0, 1.0, abs_max < 1e-8);
        zone_warm_scales_f16[group_idx] = f16(scale);
    }

    workgroupBarrier();

    let scale = f32(zone_warm_scales_f16[group_idx]);
    let inv_scale = select(1.0 / scale, 0.0, scale < 1e-8);

    var i = group_start + local_idx;
    while (i < group_end) {
        let val = f32(zone_hot_input_f16[i]);
        let scaled = val * inv_scale;
        let clamped = clamp(scaled, INT8_MIN, INT8_MAX);
        zone_warm_output_f16[i] = i32(round(clamped));
        i += WORKGROUP_SIZE;
    }
}
