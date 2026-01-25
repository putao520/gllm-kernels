// RMSNorm kernel (f32 only).
// output = (input / rms) * weight, rms = sqrt(mean(x^2) + eps)

struct Params {
    rows: u32,
    hidden: u32,
    _pad0: u32,
    _pad1: u32,
    eps: f32,
    _pad2: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn rms_norm_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.rows) {
        return;
    }

    let hidden = params.hidden;
    let base = row * hidden;
    var sum: f32 = 0.0;

    for (var i: u32 = 0u; i < hidden; i = i + 1u) {
        let v = input[base + i];
        sum = sum + v * v;
    }

    let mean = sum / f32(hidden);
    let inv_rms = inverseSqrt(mean + params.eps);

    for (var i: u32 = 0u; i < hidden; i = i + 1u) {
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }
}

@group(0) @binding(0) var<uniform> params_inplace: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> weight_inplace: array<f32>;

@compute @workgroup_size(1)
fn rms_norm_inplace_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params_inplace.rows) {
        return;
    }

    let hidden = params_inplace.hidden;
    let base = row * hidden;
    var sum: f32 = 0.0;

    for (var i: u32 = 0u; i < hidden; i = i + 1u) {
        let v = data[base + i];
        sum = sum + v * v;
    }

    let mean = sum / f32(hidden);
    let inv_rms = inverseSqrt(mean + params_inplace.eps);

    for (var i: u32 = 0u; i < hidden; i = i + 1u) {
        data[base + i] = data[base + i] * inv_rms * weight_inplace[i];
    }
}
