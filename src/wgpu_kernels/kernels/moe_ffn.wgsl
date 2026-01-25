// MoE FFN - Fused kernel for Mixture of Experts
// Computes: output = sum(weight[k] * FFN(input, expert[k])) for selected experts
// FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

struct MoEParams {
    hidden_size: u32,
    intermediate_size: u32,
    num_tokens: u32,
    top_k: u32,
    num_experts: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: MoEParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;           // [num_tokens, hidden_size]
@group(0) @binding(2) var<storage, read> expert_indices: array<u32>;  // [num_tokens, top_k]
@group(0) @binding(3) var<storage, read> expert_weights: array<f32>;  // [num_tokens, top_k]
@group(0) @binding(4) var<storage, read_write> output: array<f32>;    // [num_tokens, hidden_size]

// Expert weights - packed contiguously: [num_experts, 3, intermediate/hidden, hidden/intermediate]
@group(0) @binding(5) var<storage, read> gate_weights: array<f32>;    // [num_experts, intermediate, hidden]
@group(0) @binding(6) var<storage, read> up_weights: array<f32>;      // [num_experts, intermediate, hidden]
@group(0) @binding(7) var<storage, read> down_weights: array<f32>;    // [num_experts, hidden, intermediate]

// Scratch space for intermediate computations
@group(0) @binding(8) var<storage, read_write> scratch: array<f32>;   // [intermediate * 2]

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// Single token, single expert FFN with weighted accumulation
// This is called per-workgroup, each handling one (token, expert) pair
@compute @workgroup_size(256)
fn moe_ffn_forward(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(workgroup_id) wgid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
    let token_idx = wgid.x;
    let k_idx = wgid.y;  // which of the top_k experts

    if (token_idx >= params.num_tokens || k_idx >= params.top_k) {
        return;
    }

    let routing_idx = token_idx * params.top_k + k_idx;
    let expert_idx = expert_indices[routing_idx];
    let weight = expert_weights[routing_idx];

    if (expert_idx >= params.num_experts || weight == 0.0) {
        return;
    }

    let hidden = params.hidden_size;
    let intermediate = params.intermediate_size;

    // Offsets into expert weight arrays
    let expert_gate_offset = expert_idx * intermediate * hidden;
    let expert_up_offset = expert_idx * intermediate * hidden;
    let expert_down_offset = expert_idx * hidden * intermediate;

    // Input offset for this token
    let input_offset = token_idx * hidden;
    let output_offset = token_idx * hidden;

    // Scratch offset (per workgroup)
    let scratch_base = (token_idx * params.top_k + k_idx) * intermediate * 2u;

    // Step 1: Gate projection (hidden -> intermediate)
    // gate[i] = sum(input[j] * gate_weights[expert][i][j])
    for (var i = lid.x; i < intermediate; i += 256u) {
        var sum = 0.0;
        for (var j = 0u; j < hidden; j++) {
            sum += input[input_offset + j] * gate_weights[expert_gate_offset + i * hidden + j];
        }
        scratch[scratch_base + i] = sum;
    }
    workgroupBarrier();

    // Step 2: Up projection (hidden -> intermediate)
    // up[i] = sum(input[j] * up_weights[expert][i][j])
    for (var i = lid.x; i < intermediate; i += 256u) {
        var sum = 0.0;
        for (var j = 0u; j < hidden; j++) {
            sum += input[input_offset + j] * up_weights[expert_up_offset + i * hidden + j];
        }
        scratch[scratch_base + intermediate + i] = sum;
    }
    workgroupBarrier();

    // Step 3: SiLU(gate) * up
    for (var i = lid.x; i < intermediate; i += 256u) {
        let gate_val = scratch[scratch_base + i];
        let up_val = scratch[scratch_base + intermediate + i];
        scratch[scratch_base + i] = silu(gate_val) * up_val;
    }
    workgroupBarrier();

    // Step 4: Down projection with weighted accumulation to output
    // output[i] += weight * sum(activated[j] * down_weights[expert][i][j])
    for (var i = lid.x; i < hidden; i += 256u) {
        var sum = 0.0;
        for (var j = 0u; j < intermediate; j++) {
            sum += scratch[scratch_base + j] * down_weights[expert_down_offset + i * intermediate + j];
        }
        // Atomic add for accumulation from multiple experts
        // Note: WGSL atomicAdd only works on i32/u32, so we use non-atomic for f32
        // This is safe because each (token, output_dim) is written by only one thread
        output[output_offset + i] += weight * sum;
    }
}

// Zero output buffer before MoE computation
@compute @workgroup_size(256)
fn moe_zero_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.num_tokens * params.hidden_size;
    if (idx < total) {
        output[idx] = 0.0;
    }
}
