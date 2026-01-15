enable f16;

const WORKGROUP_SIZE: u32 = 128u;
const MAX_HEAD_DIM: u32 = 256u;

struct PagedAttentionParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    block_size: u32,
    seq_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn paged_attention_forward_f32(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @group(0) @binding(0) var<storage, read> q: array<f32>,
    @group(0) @binding(1) var<storage, read> k_cache: array<f32>,
    @group(0) @binding(2) var<storage, read> v_cache: array<f32>,
    @group(0) @binding(3) var<storage, read> block_tables: array<i32>,
    @group(0) @binding(4) var<storage, read> block_offsets: array<i32>,
    @group(0) @binding(5) var<storage, read_write> o: array<f32>,
    @group(0) @binding(6) var<uniform> params: PagedAttentionParams,
) {
    if (params.seq_len == 0u || params.head_dim == 0u || params.block_size == 0u) {
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

    let position_offset = block_offsets[batch];
    let kv_len_i = position_offset + i32(params.seq_len);
    if (kv_len_i <= 0) {
        return;
    }
    let kv_len = u32(kv_len_i);

    let head_dim = params.head_dim;
    let base = ((batch * params.num_heads + head) * params.seq_len + q_index) * head_dim;

    var q_local: array<f32, MAX_HEAD_DIM>;
    var acc: array<f32, MAX_HEAD_DIM>;

    var d = 0u;
    loop {
        if (d >= head_dim) {
            break;
        }
        q_local[d] = q[base + d];
        acc[d] = 0.0;
        d = d + 1u;
    }

    var m = -3.402823e38f;
    var l = 0.0f;
    let scale = inverseSqrt(f32(head_dim));

    let table_base = batch * kv_len;
    let num_blocks = (kv_len + params.block_size - 1u) / params.block_size;
    let q_abs = position_offset + i32(q_index);

    var block_idx = 0u;
    loop {
        if (block_idx >= num_blocks) {
            break;
        }
        let token_base = block_idx * params.block_size;
        if (token_base >= kv_len) {
            break;
        }
        let block_id = block_tables[table_base + token_base];
        if (block_id < 0) {
            block_idx = block_idx + 1u;
            continue;
        }

        var tokens_in_block = kv_len - token_base;
        if (tokens_in_block > params.block_size) {
            tokens_in_block = params.block_size;
        }

        var t = 0u;
        loop {
            if (t >= tokens_in_block) {
                break;
            }
            let k_idx = token_base + t;
            if (i32(k_idx) > q_abs) {
                t = t + 1u;
                continue;
            }

            let kv_base = ((u32(block_id) * params.block_size + t) * params.num_heads + head) * head_dim;
            var dot = 0.0f;
            var dd = 0u;
            loop {
                if (dd >= head_dim) {
                    break;
                }
                dot = dot + q_local[dd] * k_cache[kv_base + dd];
                dd = dd + 1u;
            }
            let score = dot * scale;
            let m_new = max(m, score);
            let exp_scale = exp(m - m_new);
            let exp_score = exp(score - m_new);
            l = l * exp_scale + exp_score;

            var od = 0u;
            loop {
                if (od >= head_dim) {
                    break;
                }
                acc[od] = acc[od] * exp_scale + exp_score * v_cache[kv_base + od];
                od = od + 1u;
            }
            m = m_new;
            t = t + 1u;
        }

        block_idx = block_idx + 1u;
    }

    let inv_l = select(0.0, 1.0 / l, l > 0.0);
    var out_d = 0u;
    loop {
        if (out_d >= head_dim) {
            break;
        }
        o[base + out_d] = acc[out_d] * inv_l;
        out_d = out_d + 1u;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn paged_attention_forward_f16(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @group(0) @binding(0) var<storage, read> q: array<f16>,
    @group(0) @binding(1) var<storage, read> k_cache: array<f16>,
    @group(0) @binding(2) var<storage, read> v_cache: array<f16>,
    @group(0) @binding(3) var<storage, read> block_tables: array<i32>,
    @group(0) @binding(4) var<storage, read> block_offsets: array<i32>,
    @group(0) @binding(5) var<storage, read_write> o: array<f16>,
    @group(0) @binding(6) var<uniform> params: PagedAttentionParams,
) {
    if (params.seq_len == 0u || params.head_dim == 0u || params.block_size == 0u) {
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

    let position_offset = block_offsets[batch];
    let kv_len_i = position_offset + i32(params.seq_len);
    if (kv_len_i <= 0) {
        return;
    }
    let kv_len = u32(kv_len_i);

    let head_dim = params.head_dim;
    let base = ((batch * params.num_heads + head) * params.seq_len + q_index) * head_dim;

    var q_local: array<f32, MAX_HEAD_DIM>;
    var acc: array<f32, MAX_HEAD_DIM>;

    var d = 0u;
    loop {
        if (d >= head_dim) {
            break;
        }
        q_local[d] = f32(q[base + d]);
        acc[d] = 0.0;
        d = d + 1u;
    }

    var m = -3.402823e38f;
    var l = 0.0f;
    let scale = inverseSqrt(f32(head_dim));

    let table_base = batch * kv_len;
    let num_blocks = (kv_len + params.block_size - 1u) / params.block_size;
    let q_abs = position_offset + i32(q_index);

    var block_idx = 0u;
    loop {
        if (block_idx >= num_blocks) {
            break;
        }
        let token_base = block_idx * params.block_size;
        if (token_base >= kv_len) {
            break;
        }
        let block_id = block_tables[table_base + token_base];
        if (block_id < 0) {
            block_idx = block_idx + 1u;
            continue;
        }

        var tokens_in_block = kv_len - token_base;
        if (tokens_in_block > params.block_size) {
            tokens_in_block = params.block_size;
        }

        var t = 0u;
        loop {
            if (t >= tokens_in_block) {
                break;
            }
            let k_idx = token_base + t;
            if (i32(k_idx) > q_abs) {
                t = t + 1u;
                continue;
            }

            let kv_base = ((u32(block_id) * params.block_size + t) * params.num_heads + head) * head_dim;
            var dot = 0.0f;
            var dd = 0u;
            loop {
                if (dd >= head_dim) {
                    break;
                }
                dot = dot + q_local[dd] * f32(k_cache[kv_base + dd]);
                dd = dd + 1u;
            }
            let score = dot * scale;
            let m_new = max(m, score);
            let exp_scale = exp(m - m_new);
            let exp_score = exp(score - m_new);
            l = l * exp_scale + exp_score;

            var od = 0u;
            loop {
                if (od >= head_dim) {
                    break;
                }
                acc[od] = acc[od] * exp_scale + exp_score * f32(v_cache[kv_base + od]);
                od = od + 1u;
            }
            m = m_new;
            t = t + 1u;
        }

        block_idx = block_idx + 1u;
    }

    let inv_l = select(0.0, 1.0 / l, l > 0.0);
    var out_d = 0u;
    loop {
        if (out_d >= head_dim) {
            break;
        }
        o[base + out_d] = f16(acc[out_d] * inv_l);
        out_d = out_d + 1u;
    }
}
