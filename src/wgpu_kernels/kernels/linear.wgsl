// GEMV (Matrix-Vector Multiplication) Kernel for Decode (batch=1)
// output[i] = dot(weight[i], input) + bias[i]

struct Params {
    in_features: u32,
    out_features: u32,
    has_bias: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read> weight_mat: array<f32>;
@group(0) @binding(3) var<storage, read> bias_vec: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_vec: array<f32>;

// Using a workgroup size of 256 for good occupancy on most GPUs
@compute @workgroup_size(256)
fn gemv_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.out_features) {
        return;
    }

    let in_features = params.in_features;
    let weight_start = row * in_features;

    var sum: f32 = 0.0;
    
    // Simple loop for now. In Phase 2, we can optimize with shared memory 
    // and vector types (f32x4) if performance requires it.
    for (var i: u32 = 0u; i < in_features; i = i + 1u) {
        sum = sum + weight_mat[weight_start + i] * input_vec[i];
    }

    if (params.has_bias != 0u) {
        sum = sum + bias_vec[row];
    }

    output_vec[row] = sum;
}
