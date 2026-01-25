// MoE routing - fused matmul + softmax + top-k selection.

const MAX_TOPK: u32 = 64u;
const NEG_INF: f32 = -3.402823e38;

struct MoERoutingParams {
    num_tokens: u32,
    hidden_size: u32,
    num_experts: u32,
    top_k: u32,
}

@group(0) @binding(0) var<uniform> params: MoERoutingParams;
@group(0) @binding(1) var<storage, read> hidden_states: array<f32>;  // [num_tokens, hidden_size]
@group(0) @binding(2) var<storage, read> gate_weights: array<f32>;   // [hidden_size, num_experts]
@group(0) @binding(3) var<storage, read_write> expert_indices: array<u32>; // [num_tokens, top_k]
@group(0) @binding(4) var<storage, read_write> expert_weights: array<f32>; // [num_tokens, top_k]

@compute @workgroup_size(256)
fn moe_routing(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.num_tokens) {
        return;
    }
    if (params.top_k == 0u || params.num_experts == 0u) {
        return;
    }
    if (params.top_k > MAX_TOPK) {
        return;
    }

    var top_values: array<f32, MAX_TOPK>;
    var top_indices: array<u32, MAX_TOPK>;
    for (var i: u32 = 0u; i < MAX_TOPK; i = i + 1u) {
        top_values[i] = NEG_INF;
        top_indices[i] = 0u;
    }

    var filled: u32 = 0u;
    let hidden_offset = token_idx * params.hidden_size;

    for (var e: u32 = 0u; e < params.num_experts; e = e + 1u) {
        var logit: f32 = 0.0;
        for (var h: u32 = 0u; h < params.hidden_size; h = h + 1u) {
            logit = logit + hidden_states[hidden_offset + h] * gate_weights[h * params.num_experts + e];
        }

        var insert_pos: i32 = -1;
        var i: u32 = 0u;
        loop {
            if (i >= filled) {
                break;
            }
            if (logit > top_values[i]) {
                insert_pos = i32(i);
                break;
            }
            i = i + 1u;
        }

        if (insert_pos < 0 && filled < params.top_k) {
            insert_pos = i32(filled);
        }

        if (insert_pos >= 0) {
            if (filled < params.top_k) {
                filled = filled + 1u;
            }

            var j: i32 = i32(filled) - 1;
            loop {
                if (j <= insert_pos) {
                    break;
                }
                let ju = u32(j);
                let jm1 = u32(j - 1);
                top_values[ju] = top_values[jm1];
                top_indices[ju] = top_indices[jm1];
                j = j - 1;
            }

            let insert_u = u32(insert_pos);
            top_values[insert_u] = logit;
            top_indices[insert_u] = e;
        }
    }

    let out_offset = token_idx * params.top_k;
    let max_val = top_values[0];
    var denom: f32 = 0.0;
    for (var i: u32 = 0u; i < params.top_k; i = i + 1u) {
        denom = denom + exp(top_values[i] - max_val);
    }

    if (denom == 0.0) {
        for (var i: u32 = 0u; i < params.top_k; i = i + 1u) {
            expert_indices[out_offset + i] = top_indices[i];
            expert_weights[out_offset + i] = 0.0;
        }
        return;
    }

    for (var i: u32 = 0u; i < params.top_k; i = i + 1u) {
        expert_indices[out_offset + i] = top_indices[i];
        expert_weights[out_offset + i] = exp(top_values[i] - max_val) / denom;
    }
}
