// GEMM (Matrix-Matrix Multiplication) Kernel (f32 only).
// C = alpha * A * B + beta * C

override WORKGROUP_X: u32 = 16u;
override WORKGROUP_Y: u32 = 16u;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    trans_a: u32,
    trans_b: u32,
    _pad0: array<u32, 3>,
    alpha: f32,
    beta: f32,
    _pad1: array<f32, 2>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn matmul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.m || col >= params.n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var l: u32 = 0u; l < params.k; l = l + 1u) {
        let a_idx = select(row * params.k + l, l * params.m + row, params.trans_a != 0u);
        let b_idx = select(l * params.n + col, col * params.k + l, params.trans_b != 0u);
        sum = sum + a[a_idx] * b[b_idx];
    }

    let out_idx = row * params.n + col;
    var out_val = params.alpha * sum;
    if (params.beta != 0.0) {
        out_val = out_val + params.beta * c[out_idx];
    }
    c[out_idx] = out_val;
}
