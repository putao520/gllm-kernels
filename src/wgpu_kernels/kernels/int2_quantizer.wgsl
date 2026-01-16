// INT2 Extreme Quantization Kernels
// Based on QuaRot, GPTQ-INT2, SqueezeLLM techniques
//
// Kernels:
// - int2_quantize_f32: Quantize F32 to INT2 (0-3) with group-wise scales
// - int2_dequantize_f32: Dequantize INT2 back to F32
// - int2_pack_f32: Pack 4 INT2 values into single byte
// - int2_unpack_f32: Unpack byte to 4 INT2 values
// - int2_quantize_f16/dequantize_f16: F16 variants

enable f16;

const WORKGROUP_SIZE: u32 = 256u;
const INT2_MIN: f32 = 0.0;
const INT2_MAX: f32 = 3.0;

// ============================================================================
// Parameters
// ============================================================================

struct QuantizeParams {
    num_elements: u32,
    group_size: u32,
    num_groups: u32,
    _pad0: u32,
};

struct PackParams {
    num_groups: u32,
    group_size: u32,
    packed_size: u32,  // group_size / 4
    _pad0: u32,
};

// ============================================================================
// F32 Quantize Kernel
// ============================================================================
// Quantizes F32 values to INT2 (0-3) with group-wise scales
// Input: values[num_elements]
// Output: quantized[num_elements] (as u32 for INT2 values 0-3)
//         scales[num_groups], zeros[num_groups]

@group(0) @binding(0) var<storage, read> quant_input_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> quant_output_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> quant_scales_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> quant_zeros_f32: array<f32>;
@group(0) @binding(4) var<uniform> quant_params_f32: QuantizeParams;

var<workgroup> group_min: f32;
var<workgroup> group_max: f32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int2_quantize_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let group_idx = wg_id.x;
    let local_idx = local_id.x;

    if (group_idx >= quant_params_f32.num_groups) {
        return;
    }

    let group_start = group_idx * quant_params_f32.group_size;
    let group_end = min(group_start + quant_params_f32.group_size, quant_params_f32.num_elements);

    // Phase 1: Find min/max in group (parallel reduction)
    var local_min = 1e38f;
    var local_max = -1e38f;

    var i = group_start + local_idx;
    while (i < group_end) {
        let val = quant_input_f32[i];
        local_min = min(local_min, val);
        local_max = max(local_max, val);
        i += WORKGROUP_SIZE;
    }

    // Workgroup reduction for min/max
    // Simplified: first thread collects (for correctness, not optimal)
    if (local_idx == 0u) {
        var g_min = 1e38f;
        var g_max = -1e38f;
        for (var j = group_start; j < group_end; j++) {
            let val = quant_input_f32[j];
            g_min = min(g_min, val);
            g_max = max(g_max, val);
        }
        group_min = g_min;
        group_max = g_max;

        // Compute scale and zero point
        let range = g_max - g_min;
        let scale = select(range / INT2_MAX, 1.0, range < 1e-8);
        let zero = g_min;

        quant_scales_f32[group_idx] = scale;
        quant_zeros_f32[group_idx] = zero;
    }

    workgroupBarrier();

    // Phase 2: Quantize values
    let scale = quant_scales_f32[group_idx];
    let zero = quant_zeros_f32[group_idx];
    let inv_scale = select(1.0 / scale, 0.0, scale < 1e-8);

    i = group_start + local_idx;
    while (i < group_end) {
        let val = quant_input_f32[i];
        let normalized = (val - zero) * inv_scale;
        let clamped = clamp(normalized, INT2_MIN, INT2_MAX);
        let quantized = u32(round(clamped));
        quant_output_f32[i] = quantized;
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// F32 Dequantize Kernel
// ============================================================================
// Dequantizes INT2 values back to F32
// Input: quantized[num_elements], scales[num_groups], zeros[num_groups]
// Output: values[num_elements]

@group(0) @binding(0) var<storage, read> dequant_input_f32: array<u32>;
@group(0) @binding(1) var<storage, read> dequant_scales_f32: array<f32>;
@group(0) @binding(2) var<storage, read> dequant_zeros_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> dequant_output_f32: array<f32>;
@group(0) @binding(4) var<uniform> dequant_params_f32: QuantizeParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int2_dequantize_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;

    if (idx >= dequant_params_f32.num_elements) {
        return;
    }

    let group_idx = idx / dequant_params_f32.group_size;
    let scale = dequant_scales_f32[group_idx];
    let zero = dequant_zeros_f32[group_idx];

    let quantized = f32(dequant_input_f32[idx]);
    let dequantized = quantized * scale + zero;

    dequant_output_f32[idx] = dequantized;
}

// ============================================================================
// F32 Pack Kernel
// ============================================================================
// Packs 4 INT2 values (0-3) into a single byte, MSB-first
// Input: quantized[group_size] per group
// Output: packed[packed_size] per group (packed_size = group_size / 4)

@group(0) @binding(0) var<storage, read> pack_input_f32: array<u32>;
@group(0) @binding(1) var<storage, read_write> pack_output_f32: array<u32>;
@group(0) @binding(2) var<uniform> pack_params_f32: PackParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int2_pack_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_packed = pack_params_f32.num_groups * pack_params_f32.packed_size;

    if (idx >= total_packed) {
        return;
    }

    let group_idx = idx / pack_params_f32.packed_size;
    let pack_idx = idx % pack_params_f32.packed_size;

    let base_idx = group_idx * pack_params_f32.group_size + pack_idx * 4u;

    // Pack 4 INT2 values into one byte (MSB-first)
    // Format: [v0:2bits][v1:2bits][v2:2bits][v3:2bits]
    let v0 = pack_input_f32[base_idx] & 0x3u;
    let v1 = pack_input_f32[base_idx + 1u] & 0x3u;
    let v2 = pack_input_f32[base_idx + 2u] & 0x3u;
    let v3 = pack_input_f32[base_idx + 3u] & 0x3u;

    let packed = (v0 << 6u) | (v1 << 4u) | (v2 << 2u) | v3;
    pack_output_f32[idx] = packed;
}

// ============================================================================
// F32 Unpack Kernel
// ============================================================================
// Unpacks byte to 4 INT2 values
// Input: packed[packed_size] per group
// Output: quantized[group_size] per group

@group(0) @binding(0) var<storage, read> unpack_input_f32: array<u32>;
@group(0) @binding(1) var<storage, read_write> unpack_output_f32: array<u32>;
@group(0) @binding(2) var<uniform> unpack_params_f32: PackParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int2_unpack_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_packed = unpack_params_f32.num_groups * unpack_params_f32.packed_size;

    if (idx >= total_packed) {
        return;
    }

    let group_idx = idx / unpack_params_f32.packed_size;
    let pack_idx = idx % unpack_params_f32.packed_size;

    let packed = unpack_input_f32[idx];
    let base_idx = group_idx * unpack_params_f32.group_size + pack_idx * 4u;

    // Unpack MSB-first
    unpack_output_f32[base_idx] = (packed >> 6u) & 0x3u;
    unpack_output_f32[base_idx + 1u] = (packed >> 4u) & 0x3u;
    unpack_output_f32[base_idx + 2u] = (packed >> 2u) & 0x3u;
    unpack_output_f32[base_idx + 3u] = packed & 0x3u;
}

// ============================================================================
// F16 Quantize Kernel
// ============================================================================

@group(0) @binding(0) var<storage, read> quant_input_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write> quant_output_f16: array<u32>;
@group(0) @binding(2) var<storage, read_write> quant_scales_f16: array<f16>;
@group(0) @binding(3) var<storage, read_write> quant_zeros_f16: array<f16>;
@group(0) @binding(4) var<uniform> quant_params_f16: QuantizeParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int2_quantize_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let group_idx = wg_id.x;
    let local_idx = local_id.x;

    if (group_idx >= quant_params_f16.num_groups) {
        return;
    }

    let group_start = group_idx * quant_params_f16.group_size;
    let group_end = min(group_start + quant_params_f16.group_size, quant_params_f16.num_elements);

    // Phase 1: Find min/max (compute in f32 for precision)
    if (local_idx == 0u) {
        var g_min = 1e38f;
        var g_max = -1e38f;
        for (var j = group_start; j < group_end; j++) {
            let val = f32(quant_input_f16[j]);
            g_min = min(g_min, val);
            g_max = max(g_max, val);
        }

        let range = g_max - g_min;
        let scale = select(range / INT2_MAX, 1.0, range < 1e-8);
        let zero = g_min;

        quant_scales_f16[group_idx] = f16(scale);
        quant_zeros_f16[group_idx] = f16(zero);
    }

    workgroupBarrier();

    // Phase 2: Quantize values
    let scale = f32(quant_scales_f16[group_idx]);
    let zero = f32(quant_zeros_f16[group_idx]);
    let inv_scale = select(1.0 / scale, 0.0, scale < 1e-8);

    var i = group_start + local_idx;
    while (i < group_end) {
        let val = f32(quant_input_f16[i]);
        let normalized = (val - zero) * inv_scale;
        let clamped = clamp(normalized, INT2_MIN, INT2_MAX);
        let quantized = u32(round(clamped));
        quant_output_f16[i] = quantized;
        i += WORKGROUP_SIZE;
    }
}

// ============================================================================
// F16 Dequantize Kernel
// ============================================================================

@group(0) @binding(0) var<storage, read> dequant_input_f16: array<u32>;
@group(0) @binding(1) var<storage, read> dequant_scales_f16_read: array<f16>;
@group(0) @binding(2) var<storage, read> dequant_zeros_f16_read: array<f16>;
@group(0) @binding(3) var<storage, read_write> dequant_output_f16: array<f16>;
@group(0) @binding(4) var<uniform> dequant_params_f16: QuantizeParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn int2_dequantize_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;

    if (idx >= dequant_params_f16.num_elements) {
        return;
    }

    let group_idx = idx / dequant_params_f16.group_size;
    let scale = f32(dequant_scales_f16_read[group_idx]);
    let zero = f32(dequant_zeros_f16_read[group_idx]);

    let quantized = f32(dequant_input_f16[idx]);
    let dequantized = quantized * scale + zero;

    dequant_output_f16[idx] = f16(dequantized);
}
