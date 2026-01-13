enable f16;

const WORKGROUP_SIZE: u32 = 128u;
const MAX_HEAD_DIM: u32 = 256u;
const BLOCK_K: u32 = 8u;

struct AttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    block_size: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
};

var<workgroup> k_tile: array<f32, BLOCK_K * MAX_HEAD_DIM>;
var<workgroup> v_tile: array<f32, BLOCK_K * MAX_HEAD_DIM>;

fn load_tile_f32(
    local_id: u32,
    tile_len: u32,
    head_dim: u32,
    base_offset: u32,
    qkv: ptr<storage, array<f32>>,
    tile: ptr<workgroup, array<f32, BLOCK_K * MAX_HEAD_DIM>>,
) {
    let elements = tile_len * head_dim;
    var idx = local_id;
    loop {
        if (idx >= elements) {
            break;
        }
        let tile_k = idx / head_dim;
        let d = idx - tile_k * head_dim;
        let pos = base_offset + tile_k * head_dim + d;
        (*tile)[tile_k * MAX_HEAD_DIM + d] = (*qkv)[pos];
        idx = idx + WORKGROUP_SIZE;
    }
}

fn load_tile_f16(
    local_id: u32,
    tile_len: u32,
    head_dim: u32,
    base_offset: u32,
    qkv: ptr<storage, array<f16>>,
    tile: ptr<workgroup, array<f32, BLOCK_K * MAX_HEAD_DIM>>,
) {
    let elements = tile_len * head_dim;
    var idx = local_id;
    loop {
        if (idx >= elements) {
            break;
        }
        let tile_k = idx / head_dim;
        let d = idx - tile_k * head_dim;
        let pos = base_offset + tile_k * head_dim + d;
        (*tile)[tile_k * MAX_HEAD_DIM + d] = f32((*qkv)[pos]);
        idx = idx + WORKGROUP_SIZE;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn flash_attention_forward_f32(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @group(0) @binding(0) var<storage, read> q: array<f32>,
    @group(0) @binding(1) var<storage, read> k: array<f32>,
    @group(0) @binding(2) var<storage, read> v: array<f32>,
    @group(0) @binding(3) var<storage, read_write> o: array<f32>,
    @group(0) @binding(4) var<uniform> params: AttentionParams,
) {
    if (params.seq_len == 0u || params.head_dim == 0u) {
        return;
    }
    if (params.head_dim > MAX_HEAD_DIM) {
        return;
    }

    let q_index = group_id.x * WORKGROUP_SIZE + local_id.x;
    if (q_index >= params.seq_len) {
        return;
    }

    let bh = group_id.y;
    let batch = bh / params.num_heads;
    if (batch >= params.batch_size) {
        return;
    }
    let head = bh - batch * params.num_heads;

    let head_dim = params.head_dim;
    let base = ((batch * params.num_heads + head) * params.seq_len + q_index) * head_dim;

    var q_local: array<f32, MAX_HEAD_DIM>;
    var o_local: array<f32, MAX_HEAD_DIM>;

    var d = 0u;
    loop {
        if (d >= head_dim) {
            break;
        }
        let idx = base + d;
        q_local[d] = q[idx];
        o_local[d] = 0.0;
        d = d + 1u;
    }

    var m = -3.402823e38f;
    var l = 0.0f;

    let tile_k = max(1u, min(params.block_size, BLOCK_K));
    var key_base = 0u;
    loop {
        if (key_base >= params.seq_len) {
            break;
        }
        let remaining = params.seq_len - key_base;
        let tile_len = min(tile_k, remaining);
        let k_base = ((batch * params.num_heads + head) * params.seq_len + key_base) * head_dim;

        load_tile_f32(local_id.x, tile_len, head_dim, k_base, &k, &k_tile);
        load_tile_f32(local_id.x, tile_len, head_dim, k_base, &v, &v_tile);
        workgroupBarrier();

        var tile_idx = 0u;
        loop {
            if (tile_idx >= tile_len) {
                break;
            }
            var sum = 0.0f;
            var dd = 0u;
            loop {
                if (dd >= head_dim) {
                    break;
                }
                sum = sum + q_local[dd] * k_tile[tile_idx * MAX_HEAD_DIM + dd];
                dd = dd + 1u;
            }

            let score = sum * params.scale;
            let m_new = max(m, score);
            let exp_m = exp(m - m_new);
            let exp_score = exp(score - m_new);
            let l_new = l * exp_m + exp_score;

            let rescale = exp_m;
            var od = 0u;
            loop {
                if (od >= head_dim) {
                    break;
                }
                let v_val = v_tile[tile_idx * MAX_HEAD_DIM + od];
                o_local[od] = o_local[od] * rescale + exp_score * v_val;
                od = od + 1u;
            }

            m = m_new;
            l = l_new;
            tile_idx = tile_idx + 1u;
        }

        workgroupBarrier();
        key_base = key_base + tile_len;
    }

    let inv_l = 1.0f / l;
    var out_d = 0u;
    loop {
        if (out_d >= head_dim) {
            break;
        }
        o[base + out_d] = o_local[out_d] * inv_l;
        out_d = out_d + 1u;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn flash_attention_forward_f16(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @group(0) @binding(0) var<storage, read> q: array<f16>,
    @group(0) @binding(1) var<storage, read> k: array<f16>,
    @group(0) @binding(2) var<storage, read> v: array<f16>,
    @group(0) @binding(3) var<storage, read_write> o: array<f16>,
    @group(0) @binding(4) var<uniform> params: AttentionParams,
) {
    if (params.seq_len == 0u || params.head_dim == 0u) {
        return;
    }
    if (params.head_dim > MAX_HEAD_DIM) {
        return;
    }

    let q_index = group_id.x * WORKGROUP_SIZE + local_id.x;
    if (q_index >= params.seq_len) {
        return;
    }

    let bh = group_id.y;
    let batch = bh / params.num_heads;
    if (batch >= params.batch_size) {
        return;
    }
    let head = bh - batch * params.num_heads;

    let head_dim = params.head_dim;
    let base = ((batch * params.num_heads + head) * params.seq_len + q_index) * head_dim;

    var q_local: array<f32, MAX_HEAD_DIM>;
    var o_local: array<f32, MAX_HEAD_DIM>;

    var d = 0u;
    loop {
        if (d >= head_dim) {
            break;
        }
        let idx = base + d;
        q_local[d] = f32(q[idx]);
        o_local[d] = 0.0;
        d = d + 1u;
    }

    var m = -3.402823e38f;
    var l = 0.0f;

    let tile_k = max(1u, min(params.block_size, BLOCK_K));
    var key_base = 0u;
    loop {
        if (key_base >= params.seq_len) {
            break;
        }
        let remaining = params.seq_len - key_base;
        let tile_len = min(tile_k, remaining);
        let k_base = ((batch * params.num_heads + head) * params.seq_len + key_base) * head_dim;

        load_tile_f16(local_id.x, tile_len, head_dim, k_base, &k, &k_tile);
        load_tile_f16(local_id.x, tile_len, head_dim, k_base, &v, &v_tile);
        workgroupBarrier();

        var tile_idx = 0u;
        loop {
            if (tile_idx >= tile_len) {
                break;
            }
            var sum = 0.0f;
            var dd = 0u;
            loop {
                if (dd >= head_dim) {
                    break;
                }
                sum = sum + q_local[dd] * k_tile[tile_idx * MAX_HEAD_DIM + dd];
                dd = dd + 1u;
            }

            let score = sum * params.scale;
            let m_new = max(m, score);
            let exp_m = exp(m - m_new);
            let exp_score = exp(score - m_new);
            let l_new = l * exp_m + exp_score;

            let rescale = exp_m;
            var od = 0u;
            loop {
                if (od >= head_dim) {
                    break;
                }
                let v_val = v_tile[tile_idx * MAX_HEAD_DIM + od];
                o_local[od] = o_local[od] * rescale + exp_score * v_val;
                od = od + 1u;
            }

            m = m_new;
            l = l_new;
            tile_idx = tile_idx + 1u;
        }

        workgroupBarrier();
        key_base = key_base + tile_len;
    }

    let inv_l = 1.0f / l;
    var out_d = 0u;
    loop {
        if (out_d >= head_dim) {
            break;
        }
        o[base + out_d] = f16(o_local[out_d] * inv_l);
        out_d = out_d + 1u;
    }
}
