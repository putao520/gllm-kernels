struct Q4DequantParams {
    num_values: u32,
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
};

struct AwqDequantParams {
    n: u32,
    k: u32,
    group_size: u32,
    groups: u32,
    num_values: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> q4_qweight: array<u32>;
@group(0) @binding(1) var<storage, read> q4_scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> q4_output: array<f32>;
@group(0) @binding(3) var<uniform> q4_params: Q4DequantParams;

var<workgroup> q4_shared_scales: array<f32, 8>;

fn q4_get_byte(word: u32, byte_idx: u32) -> u32 {
    return (word >> (byte_idx * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn q4_0_dequantize(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_idx = lid.x;
    if (local_idx < 8u) {
        let block_idx = wid.x * 8u + local_idx;
        if (block_idx < q4_params.num_blocks) {
            q4_shared_scales[local_idx] = q4_scales[block_idx];
        } else {
            q4_shared_scales[local_idx] = 0.0;
        }
    }
    workgroupBarrier();

    let idx = gid.x;
    if (idx >= q4_params.num_values) {
        return;
    }
    let block_idx = idx >> 5u;
    let in_block = idx & 31u;
    let local_block = lid.x >> 5u;
    let byte_idx = in_block >> 1u;
    let word_idx = (block_idx << 2u) + (byte_idx >> 2u);
    let word = q4_qweight[word_idx];
    let byte = q4_get_byte(word, byte_idx & 3u);
    let nibble = select(byte & 0xFu, byte >> 4u, (in_block & 1u) == 1u);
    let q = i32(nibble) - 8;
    let scale = q4_shared_scales[local_block];
    q4_output[idx] = f32(q) * scale;
}

@group(0) @binding(4) var<storage, read> awq_qweight: array<u32>;
@group(0) @binding(5) var<storage, read> awq_qzeros: array<u32>;
@group(0) @binding(6) var<storage, read> awq_scales: array<f32>;
@group(0) @binding(7) var<storage, read_write> awq_output: array<f32>;
@group(0) @binding(8) var<uniform> awq_params: AwqDequantParams;

fn awq_nibble(word: u32, index: u32) -> u32 {
    return (word >> (index * 4u)) & 0xFu;
}

@compute @workgroup_size(256)
fn awq_dequantize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= awq_params.num_values) {
        return;
    }
    let out = idx / awq_params.k;
    let k_idx = idx - out * awq_params.k;
    let group = k_idx / awq_params.group_size;
    let packed_row = out >> 3u;
    let nibble = out & 7u;

    let weight_word = awq_qweight[packed_row * awq_params.k + k_idx];
    let zero_word = awq_qzeros[packed_row * awq_params.groups + group];
    let w = i32(awq_nibble(weight_word, nibble));
    let z = i32(awq_nibble(zero_word, nibble));
    let scale = awq_scales[out * awq_params.groups + group];
    awq_output[idx] = f32(w - z) * scale;
}
