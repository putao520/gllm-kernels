//! Scalar reference for MoEDispatchPacked (OpenAI gpt-oss-20b).

/// Packed-expert MoE dispatch with mxfp4 dequant + clipped SwiGLU.
///
/// # Layout
/// - `hidden_input`: [seq_len, hidden] f32
/// - `router_weights`: [seq_len, top_k] f32 (normalized weights)
/// - `router_indices`: [seq_len, top_k] u32 (stored as f32 bits)
/// - `gate_up_blocks`: [num_experts, 2*intermediate_size / block_size, block_bytes] u8 (mxfp4 packed)
/// - `gate_up_scales`: [num_experts, 2*intermediate_size / block_size] u8 (e8m0)
/// - `gate_up_bias`: [num_experts, 2*intermediate_size] f32
/// - `down_blocks`: [num_experts, hidden / block_size, intermediate_size * block_bytes] u8 (mxfp4 packed)
/// - `down_scales`: [num_experts, hidden / block_size] u8 (e8m0)
/// - `down_bias`: [num_experts, hidden] f32
/// - `output`: [seq_len, hidden] f32 (accumulates weighted expert outputs)
///
/// # Algorithm (per token s)
/// ```text
/// for k in 0..top_k:
///     e = router_indices[s, k]  (u32 from f32 bits)
///     w = router_weights[s, k]
///
///     // Dequant gate_up for expert e
///     gu = mxfp4_dequant(gate_up_blocks[e], gate_up_scales[e]) + gate_up_bias[e]
///     gate = gu[0..intermediate_size]
///     up   = gu[intermediate_size..2*intermediate_size]
///
///     // Clipped SwiGLU
///     for i in 0..intermediate_size:
///         gate_clamped = clamp(gate[i], -limit, +limit)
///         up_clamped   = clamp(up[i], -limit, +limit)
///         activ[i]     = silu(gate_clamped) * up_clamped
///
///     // Dequant down for expert e + GEMV
///     down_w = mxfp4_dequant(down_blocks[e], down_scales[e])  // [hidden, intermediate_size]
///     for h in 0..hidden:
///         acc = down_bias[e][h]
///         for i in 0..intermediate_size:
///             acc += down_w[h * intermediate_size + i] * activ[i]
///         output[s, h] += w * acc
/// ```
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_moe_dispatch_packed(
    _hidden_input: *const f32,
    router_weights: *const f32,
    router_indices: *const f32,
    gate_up_blocks: *const u8,
    gate_up_scales: *const u8,
    gate_up_bias: *const f32,
    down_blocks: *const u8,
    down_scales: *const u8,
    down_bias: *const f32,
    output: *mut f32,
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
    mxfp4_block_size: usize,
    swiglu_limit: f32,
    intermediate_size: usize,
    hidden: usize,
) {
    // Helper: decode e8m0 scale
    fn decode_e8m0(byte: u8) -> f32 {
        match byte {
            255 => f32::NAN,
            0 => f32::from_bits(0x0040_0000),
            v => f32::from_bits((v as u32) << 23),
        }
    }

    // Helper: decode e2m1 nibble
    const E2M1_LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    // Helper: silu
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // Helper: clamp
    fn clamp(x: f32, min: f32, max: f32) -> f32 {
        if x < min { min } else if x > max { max } else { x }
    }

    // Temp buffers (stack allocation for small sizes, heap for large)
    let gu_size = 2 * intermediate_size;
    let mut gu_buf = vec![0.0f32; gu_size];
    let mut activ_buf = vec![0.0f32; intermediate_size];
    let mut down_w_buf = vec![0.0f32; hidden * intermediate_size];

    let gu_blocks_per_expert = gu_size / mxfp4_block_size;
    let gu_block_bytes = mxfp4_block_size / 2;
    let down_blocks_per_expert = hidden / mxfp4_block_size;
    let down_block_bytes = intermediate_size * mxfp4_block_size / 2;

    for s in 0..seq_len {
        for k in 0..top_k {
            // Extract expert index (u32 stored in f32 bits)
            let idx_bits = *router_indices.add(s * top_k + k);
            let expert_id = f32::to_bits(idx_bits) as usize;
            if expert_id >= num_experts {
                continue; // Invalid expert, skip
            }

            let weight = *router_weights.add(s * top_k + k);

            // === Dequant gate_up ===
            let gu_blocks_base = gate_up_blocks.add(expert_id * gu_blocks_per_expert * gu_block_bytes);
            let gu_scales_base = gate_up_scales.add(expert_id * gu_blocks_per_expert);
            let gu_bias_base = gate_up_bias.add(expert_id * gu_size);

            for blk in 0..gu_blocks_per_expert {
                let scale = decode_e8m0(*gu_scales_base.add(blk));
                let block_ptr = gu_blocks_base.add(blk * gu_block_bytes);
                for i in 0..mxfp4_block_size {
                    let byte_idx = i / 2;
                    let nibble = if i % 2 == 0 {
                        *block_ptr.add(byte_idx) & 0x0F
                    } else {
                        (*block_ptr.add(byte_idx) >> 4) & 0x0F
                    };
                    let val = E2M1_LUT[nibble as usize] * scale;
                    let out_idx = blk * mxfp4_block_size + i;
                    gu_buf[out_idx] = val + *gu_bias_base.add(out_idx);
                }
            }

            // === Clipped SwiGLU ===
            for i in 0..intermediate_size {
                let gate_val = clamp(gu_buf[i], -swiglu_limit, swiglu_limit);
                let up_val = clamp(gu_buf[intermediate_size + i], -swiglu_limit, swiglu_limit);
                activ_buf[i] = silu(gate_val) * up_val;
            }

            // === Dequant down ===
            let down_blocks_base = down_blocks.add(expert_id * down_blocks_per_expert * down_block_bytes);
            let down_scales_base = down_scales.add(expert_id * down_blocks_per_expert);

            for blk in 0..down_blocks_per_expert {
                let scale = decode_e8m0(*down_scales_base.add(blk));
                let block_ptr = down_blocks_base.add(blk * down_block_bytes);
                // Each block: [mxfp4_block_size rows, intermediate_size cols]
                for row in 0..mxfp4_block_size {
                    for col in 0..intermediate_size {
                        let flat_idx = row * intermediate_size + col;
                        let byte_idx = flat_idx / 2;
                        let nibble = if flat_idx % 2 == 0 {
                            *block_ptr.add(byte_idx) & 0x0F
                        } else {
                            (*block_ptr.add(byte_idx) >> 4) & 0x0F
                        };
                        let val = E2M1_LUT[nibble as usize] * scale;
                        let out_row = blk * mxfp4_block_size + row;
                        down_w_buf[out_row * intermediate_size + col] = val;
                    }
                }
            }

            // === GEMV + bias + accumulate ===
            let down_bias_base = down_bias.add(expert_id * hidden);
            for h in 0..hidden {
                let mut acc = *down_bias_base.add(h);
                for i in 0..intermediate_size {
                    acc += down_w_buf[h * intermediate_size + i] * activ_buf[i];
                }
                let out_ptr = output.add(s * hidden + h);
                *out_ptr += weight * acc;
            }
        }
    }
}
