// MoE Tensor Operations - Pure GPU implementations

// Uniform buffer for operation parameters
struct TensorOpsParams {
    size: u32,
    offset: u32,
    scale: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: TensorOpsParams;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;

// Zero kernel: output[i] = 0
@compute @workgroup_size(256)
fn tensor_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        output[idx] = 0.0;
    }
}

// Add kernel: output[i] += input[i]
@compute @workgroup_size(256)
fn tensor_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        output[idx] = output[idx] + input[idx];
    }
}

// Slice kernel: output[i] = input[offset + i]
@compute @workgroup_size(256)
fn tensor_slice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        output[idx] = input[params.offset + idx];
    }
}

// Scale-add kernel: output[offset + i] += input[i] * scale
@compute @workgroup_size(256)
fn tensor_scale_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        output[params.offset + idx] = output[params.offset + idx] + input[idx] * params.scale;
    }
}
