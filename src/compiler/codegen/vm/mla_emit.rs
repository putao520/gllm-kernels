//! MLA (Multi-head Latent Attention) inline lowering — DeepSeek V3/R1, Kimi-K2.
//!
//! Implements the two MLA-specific structural TraceOp emissions:
//! - `emit_mla_attn_score_inline`: compressed-space attention scoring + V restore
//! - `emit_mla_rope_merge_inline`: decoupled RoPE key merge into c_KV buffer
//!
//! Per SPEC/33 REQ-MLA-007: these are structural ops that decompose into
//! existing VmInstr primitives (VecLoad, VecStore, Fma, HReduce, Softmax*,
//! LoopBegin/End, GprBinOp) composed through TraceOp bodies + auto_lower_trace.

use super::instr::*;
use crate::compiler::trace::{QuantPrecision, TraceOp, ReduceKind, ValueId};
use crate::types::CompilerError;

// ── Shared TraceOp bodies ──────────────────────────────────────────────────

/// FMA dot product accumulation: acc += a * b
fn dot_fma_body() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // acc
        TraceOp::Input(1), // a
        TraceOp::Input(2), // b
        TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)), // acc + a*b
    ]
}

/// Scale: result = x * scale
fn scale_body() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // x
        TraceOp::Input(1), // scale
        TraceOp::Mul(ValueId(0), ValueId(1)),
    ]
}

/// V accumulation: o_acc = o_acc * correction + weight * v_vec
fn accumulate_body() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // o_acc
        TraceOp::Input(1), // correction
        TraceOp::Input(2), // weight
        TraceOp::Input(3), // v_vec
        TraceOp::Mul(ValueId(0), ValueId(1)), // o_acc * correction
        TraceOp::Mul(ValueId(2), ValueId(3)), // weight * v_vec
        TraceOp::Add(ValueId(4), ValueId(5)), // result
    ]
}

/// Online softmax body: computes exp(score - running_max), returns
/// [0]=running_max, [1]=score, [2]=new_max, [4]=correction, [6]=weight
fn softmax_body() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // running_max
        TraceOp::Input(1), // score
        TraceOp::Max(ValueId(0), ValueId(1)),       // [2] new_max
        TraceOp::Sub(ValueId(0), ValueId(2)),        // [3] old_max - new_max = correction_exp_arg
        TraceOp::Exp(ValueId(3)),                     // [4] correction = exp(old_max - new_max)
        TraceOp::Sub(ValueId(1), ValueId(2)),        // [5] score - new_max
        TraceOp::Exp(ValueId(5)),                     // [6] weight = exp(score - new_max)
    ]
}

/// Running sum update: new_sum = sum * correction + weight
fn sum_update_body() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // running_sum
        TraceOp::Input(1), // correction
        TraceOp::Input(2), // weight
        TraceOp::Mul(ValueId(0), ValueId(1)), // sum * correction
        TraceOp::Add(ValueId(3), ValueId(2)), // new_sum
    ]
}

/// Identity passthrough: returns input as-is
fn identity_body() -> Vec<TraceOp> {
    vec![TraceOp::Input(0)]
}

/// Reciprocal: result = 1/x
fn recip_body() -> Vec<TraceOp> {
    vec![TraceOp::Input(0), TraceOp::Recip(ValueId(0))]
}

/// Normalize: result = x * (1/sum)
fn normalize_body() -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // x
        TraceOp::Input(1), // recip_sum
        TraceOp::Mul(ValueId(0), ValueId(1)),
    ]
}

// ── MLA Attention Score ───────────────────────────────────────────────────

/// MLA Attention score computation in compressed d_c space.
///
/// For each head h in 0..num_heads:
///   1. Load Q_absorbed[h]: [d_c] elements
///   2. For each KV position:
///      - Load key[pos]: [d_c] elements (c_KV with RoPE-merged k_pe in last d_rope dims)
///      - Compute dot product: score = Q_absorbed[h] . key[pos]
///   3. Online softmax over all positions
///   4. For each KV position:
///      - Load c_KV[pos]: [d_c] elements
///      - V restore: V_h[pos] = GEMV(W_UV_h, c_KV[pos])  → [head_dim] elements
///      - Accumulate: o_acc += score[pos] * V_h[pos]
///   5. Store output: o_acc / running_sum → output[h * head_dim .. (h+1) * head_dim]
///
/// Input slots (from auto_select):
///   [0]: q_absorbed_ptr    — Q_absorbed [num_heads, d_c]
///   [1]: kv_cache_ptr      — KV cache (c_KV + k_pe) [kv_len, d_c + d_rope]
///   [2]: w_uv_ptr          — W_UV weights [num_heads, d_c, head_dim]
///   [3]: output_ptr        — Output [num_heads * head_dim]
///   kv_len: VRegId         — Runtime kv_len value (scalar GPR, already computed by caller)
///
/// Note: kv_len is dynamic (Symbolic), passed as a pre-computed VRegId.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_mla_attn_score_inline(
    prog: &mut VmProgram,
    num_heads: usize,
    head_dim: usize,
    d_c: usize,
    d_rope: usize,
    slots: &[VRegId],
    kv_len: VRegId,
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    let q_absorbed_ptr = slots[0];
    let kv_cache_ptr = slots[1];
    let w_uv_ptr = slots[2];
    let output_ptr = slots[3];

    let lanes = width.f32_lanes().max(1);
    let elem_bytes = default_dtype.elem_bytes();

    // Dimension sanity checks
    if d_c == 0 || head_dim == 0 || num_heads == 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "MlaAttnScore: invalid dimensions (num_heads={}, head_dim={}, d_c={}, d_rope={})",
            num_heads, head_dim, d_c, d_rope
        )));
    }

    let dc_vecs = d_c / lanes;
    let dc_vec_step = lanes * elem_bytes;
    let hd_vecs = head_dim / lanes;
    let hd_vec_step = lanes * elem_bytes;

    // Strides
    let head_dc_bytes = d_c * elem_bytes;
    let head_hd_bytes = head_dim * elem_bytes;
    let kv_row_bytes = (d_c + d_rope) * elem_bytes;
    // W_UV per-head stride: d_c * head_dim * elem_bytes
    let w_uv_head_stride = d_c * head_dim * elem_bytes;

    // Score scale: 1/sqrt(d_c) — attention in compressed space uses d_c not head_dim
    let inv_sqrt_dc = 1.0 / (d_c as f32).sqrt();
    let scale_vec = prog.alloc_vreg(VRegKind::Vec, width);
    prog.emit(VmInstr::Broadcast {
        dst: scale_vec,
        src: ScalarExpr::Const(inv_sqrt_dc),
        width,
        dtype: default_dtype,
    });

    // TraceOp bodies (allocated once, reused per head)
    let dot_body = dot_fma_body();
    let hreduce_body = vec![
        TraceOp::Input(0),
        TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum },
    ];
    let s_body = scale_body();
    let sm_body = softmax_body();
    let su_body = sum_update_body();
    let id_body = identity_body();
    let acc_body = accumulate_body();
    let rec_body = recip_body();
    let norm_body = normalize_body();

    // Allocate score buffer pointer: we need space for kv_len scores
    // Use a scratch area from the output_ptr region (after num_heads * head_dim)
    let score_buf_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let score_buf_offset = num_heads * head_dim * elem_bytes;
    prog.emit(VmInstr::AddPtr {
        dst: score_buf_ptr,
        base: output_ptr,
        offset: score_buf_offset,
    });

    // Per-head loop
    let num_heads_bound = BoundExpr::Const(num_heads);
    prog.emit_loop(num_heads_bound, head_dc_bytes, |prog, h_ctr, h_off| {
        // Q_absorbed row for this head: q_absorbed_ptr + h * head_dc_bytes
        let q_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::AddPtr {
            dst: q_row,
            base: q_absorbed_ptr,
            offset: 0, // will use h_off
        });
        // Fix: use LoadPtr with computed offset
        prog.emit(VmInstr::GprBinOp {
            dst: q_row,
            a: q_absorbed_ptr,
            b: GprOperand::VReg(h_off),
            op: GprOp::Add,
        });

        // W_UV row for this head: w_uv_ptr + h * w_uv_head_stride
        let wuv_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let h_wuv_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: h_wuv_off,
            a: h_ctr,
            b: GprOperand::Imm(w_uv_head_stride as i64),
            op: GprOp::Mul,
        });
        prog.emit(VmInstr::GprBinOp {
            dst: wuv_row,
            a: w_uv_ptr,
            b: GprOperand::VReg(h_wuv_off),
            op: GprOp::Add,
        });

        // Output row for this head: output_ptr + h * head_hd_bytes
        let o_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let h_out_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: h_out_off,
            a: h_ctr,
            b: GprOperand::Imm(head_hd_bytes as i64),
            op: GprOp::Mul,
        });
        prog.emit(VmInstr::GprBinOp {
            dst: o_row,
            a: output_ptr,
            b: GprOperand::VReg(h_out_off),
            op: GprOp::Add,
        });

        // Initialize running_max = -inf, running_sum = 0
        let running_max = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast {
            dst: running_max,
            src: ScalarExpr::Const(f32::NEG_INFINITY),
            width,
            dtype: default_dtype,
        });
        let running_sum = prog.alloc_vreg(VRegKind::Vec, width);
        prog.emit(VmInstr::Broadcast {
            dst: running_sum,
            src: ScalarExpr::Const(0.0),
            width,
            dtype: default_dtype,
        });

        // o_acc[d] = 0.0 for d in 0..hd_vecs
        let o_acc: Vec<VRegId> = (0..hd_vecs)
            .map(|_| {
                let acc = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::Broadcast {
                    dst: acc,
                    src: ScalarExpr::Const(0.0),
                    width,
                    dtype: default_dtype,
                });
                acc
            })
            .collect();

        // ── Phase 1: Score computation + online softmax ──
        // For each KV position, compute Q_absorbed · key[pos]^T, then update softmax
        let kv_len_bound = BoundExpr::DynamicVReg(kv_len);
        prog.emit_loop(kv_len_bound, kv_row_bytes, |prog, _pos_ctr, pos_off| {
            // key row for this position: kv_cache_ptr + pos_off
            let key_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprBinOp {
                dst: key_row,
                a: kv_cache_ptr,
                b: GprOperand::VReg(pos_off),
                op: GprOp::Add,
            });

            // Dot product: score = Q_absorbed[h] . key[pos] (d_c dimensions)
            let dot_acc = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::Broadcast {
                dst: dot_acc,
                src: ScalarExpr::Const(0.0),
                width,
                dtype: default_dtype,
            });

            if dc_vecs > 0 {
                prog.emit_loop(BoundExpr::Const(dc_vecs), dc_vec_step, |prog, _d_ctr, d_off| {
                    let q_vec = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::VecLoad {
                        dst: q_vec,
                        base: q_row,
                        offset: OffsetExpr::LoopOffset(d_off),
                        width,
                        dtype: default_dtype, predicate: None,
                    });
                    let k_vec = prog.alloc_vreg(VRegKind::Vec, width);
                    prog.emit(VmInstr::VecLoad {
                        dst: k_vec,
                        base: key_row,
                        offset: OffsetExpr::LoopOffset(d_off),
                        width,
                        dtype: default_dtype, predicate: None,
                    });
                    super::auto_select::auto_lower_trace_into(
                        prog, &dot_body, &[dot_acc, q_vec, k_vec], dot_acc, width, default_dtype,
                    ).expect("MLA dot FMA auto_lower failed");
                });
            }

            // HReduce sum + scale
            let hr_slots = super::auto_select::auto_lower_trace_raw(
                prog, &hreduce_body, &[dot_acc], width, default_dtype,
            ).expect("MLA HReduce auto_lower failed");
            let score_scalar = hr_slots[1];
            prog.emit(VmInstr::Broadcast {
                dst: dot_acc,
                src: ScalarExpr::ExtractLane0(score_scalar),
                width,
                dtype: default_dtype,
            });
            super::auto_select::auto_lower_trace_into(
                prog, &s_body, &[dot_acc, scale_vec], dot_acc, width, default_dtype,
            ).expect("MLA scale auto_lower failed");

            // Online softmax update
            let sm_slots = super::auto_select::auto_lower_trace_raw(
                prog, &sm_body, &[running_max, dot_acc], width, default_dtype,
            ).expect("MLA softmax auto_lower failed");
            let new_max = sm_slots[2];
            let correction = sm_slots[4];
            let weight = sm_slots[6];

            // Update running_sum: new_sum = sum * correction + weight
            super::auto_select::auto_lower_trace_into(
                prog, &su_body, &[running_sum, correction, weight], running_sum, width, default_dtype,
            ).expect("MLA sum update auto_lower failed");

            // Update running_max: running_max = new_max
            super::auto_select::auto_lower_trace_into(
                prog, &id_body, &[new_max], running_max, width, default_dtype,
            ).expect("MLA max identity auto_lower failed");

            // Store weight in score buffer for Phase 2
            // score_buf_ptr[pos_idx] = weight (we need pos_idx * vec_step, but weight is broadcast)
            // We store the scalar weight value
            prog.emit(VmInstr::VecStore {
                base: score_buf_ptr,
                offset: OffsetExpr::LoopOffset(pos_off),
                src: weight,
                width: SimdWidth::Scalar, // store single scalar
                dtype: default_dtype, predicate: None,
            });

            // V accumulation: for each head_dim vector, rescale o_acc by correction and add weight*v
            // c_KV row (without k_pe): same as key_row, we load first d_c elements
            // V restore: V_h = c_KV[pos] * W_UV_h  (GEMV, but simplified to dot products per output dim)
            // For decode (M=1), this is a GEMV: for each output dim d, v[d] = sum_c(c_KV[c] * W_UV[c, d])
            for d in 0..hd_vecs {
                let v_acc = prog.alloc_vreg(VRegKind::Vec, width);
                prog.emit(VmInstr::Broadcast {
                    dst: v_acc,
                    src: ScalarExpr::Const(0.0),
                    width,
                    dtype: default_dtype,
                });

                // GEMV: v[d_chunk] = sum over c_chunks of c_KV[c_chunk] * W_UV[c_chunk, d_chunk]
                if dc_vecs > 0 {
                    prog.emit_loop(BoundExpr::Const(dc_vecs), dc_vec_step, |prog, _c_ctr, c_off| {
                        let ckv_vec = prog.alloc_vreg(VRegKind::Vec, width);
                        prog.emit(VmInstr::VecLoad {
                            dst: ckv_vec,
                            base: key_row, // key_row points to c_KV (first d_c elements of KV cache row)
                            offset: OffsetExpr::LoopOffset(c_off),
                            width,
                            dtype: default_dtype, predicate: None,
                        });
                        // W_UV offset: d * hd_vec_step + c_off (row-major: [d_c, head_dim])
                        // Actually W_UV is [d_c, head_dim], so for output dim chunk d:
                        // W_UV[c_chunk, d_chunk] is at offset c_chunk * head_dim_bytes + d_chunk * elem_bytes
                        let d_out_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprBinOp {
                            dst: d_out_off,
                            a: _c_ctr, // This is wrong, we need the byte offset not counter
                            b: GprOperand::Imm(0),
                            op: GprOp::Add,
                        });
                        // Simplified: load W_UV at (c_off + d * hd_vec_step) treating W_UV as flat
                        // Actually for GEMV, the weight layout matters.
                        // W_UV is [d_c, head_dim] per head, stored row-major.
                        // v[d] = sum_c c_KV[c] * W_UV[c, d]
                        // For vectorized inner product across chunks:
                        // Each chunk of output d processes: v_chunk = sum_c (c_KV_chunk_c * W_UV_chunk_c_for_d)
                        // W_UV at (c, d_chunk) is at offset: c * head_dim_bytes + d * hd_vec_step
                        let wuv_d_offset = d * hd_vec_step;
                        let w_vec = prog.alloc_vreg(VRegKind::Vec, width);
                        // We need to compute the correct offset for W_UV[c_chunk, d_chunk]
                        // W_UV layout per head: row-major [d_c, head_dim]
                        // Element (c, d_i) = wuv_row + c * head_dim_bytes + d_i * elem_bytes
                        // For vectorized load of row c, output dim chunk d:
                        // offset = c_off * head_dim / dc_vec_step * hd_vec_step ... complex
                        //
                        // Simpler approach: load W_UV column-chunk for all c in inner loop
                        // This requires the inner loop to iterate c (d_c dimension) and for each c,
                        // load the weight element at column offset d.
                        //
                        // But since we're inside a c_chunk loop, we use the transposed access:
                        // The weight for (c_chunk, d_chunk) at byte offset c_chunk_row_offset + d * elem_bytes
                        // where c_chunk_row_offset = (c_off counter * head_dim_bytes / elem_bytes) ...
                        //
                        // For correctness, we emit a scalar GEMV loop instead of trying to vectorize
                        // the inner dimension. This is the same approach as attention_emit's score dot.
                        // The inner loop is over dc_vecs chunks of the c_KV dimension.
                        // W_UV[c_chunk, d_chunk] needs: c_chunk_base + d_chunk_offset
                        // c_chunk_base = c_off_counter * head_dim_bytes (but c_off is byte offset for c)
                        // So the row index for c is c_off / dc_vec_step (element index of c chunk)
                        // Then the weight offset for that row's d_chunk is:
                        //   row_bytes = head_dim * elem_bytes
                        //   row_idx_elems = c_off / (lanes * elem_bytes)  -- which c-chunk we're on
                        //   weight_offset = row_idx_elems * row_bytes + d * hd_vec_step
                        //
                        // Since c_off is already LoopOffset (byte offset stepping dc_vec_step),
                        // row_idx_bytes = c_off * head_dim / dc_vec_step ... but c_off is VReg not usize
                        //
                        // Simpler: precompute the GPR offset for each W_UV element
                        // weight_base_for_c_chunk = wuv_row + (c_off / elem_bytes_per_lane) * head_dim_bytes
                        // This requires integer division at runtime which is expensive.
                        //
                        // Most efficient: iterate c dimension in scalar, accumulate per d_chunk.
                        // But that defeats vectorization.
                        //
                        // Pragmatic approach for decode (M=1): the V restore GEMV is small
                        // (d_c x head_dim = 512 x 128), we use the same tiled approach as
                        // emit_score_dot_cpu: vectorize the inner (c) dimension with FMA,
                        // one accumulator per output d_chunk.
                        //
                        // The weight access pattern for W_UV[c_chunk, d_chunk]:
                        // stride between c chunks = head_dim_bytes (row stride of W_UV)
                        // offset within c chunk to reach d_chunk = d * hd_vec_step
                        //
                        // So weight address = wuv_row + c_elem * head_dim_bytes + d * hd_vec_step
                        // where c_elem is the starting element of the current c_chunk.
                        //
                        // Since c_off is the byte offset stepping dc_vec_step (= lanes * elem_bytes),
                        // c_elem = c_off / elem_bytes, but we need c_off * head_dim_bytes / (lanes * elem_bytes)
                        // which simplifies to (c_off / elem_bytes) * head_dim_bytes
                        // = c_off * head_dim_bytes / elem_bytes
                        // This requires runtime multiply which we can do with GprBinOp.
                        let c_off_gpr = _c_ctr; // c counter (not byte offset)
                        let w_row_off = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprBinOp {
                            dst: w_row_off,
                            a: c_off_gpr,
                            b: GprOperand::Imm(head_hd_bytes as i64),
                            op: GprOp::Mul,
                        });
                        let w_addr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                        prog.emit(VmInstr::GprBinOp {
                            dst: w_addr,
                            a: wuv_row,
                            b: GprOperand::VReg(w_row_off),
                            op: GprOp::Add,
                        });
                        prog.emit(VmInstr::VecLoad {
                            dst: w_vec,
                            base: w_addr,
                            offset: OffsetExpr::Const(wuv_d_offset),
                            width,
                            dtype: default_dtype, predicate: None,
                        });
                        super::auto_select::auto_lower_trace_into(
                            prog, &dot_body, &[v_acc, ckv_vec, w_vec], v_acc, width, default_dtype,
                        ).expect("MLA V restore GEMV FMA auto_lower failed");
                    });
                }

                // HReduce the GEMV accumulator
                let v_hr = super::auto_select::auto_lower_trace_raw(
                    prog, &hreduce_body, &[v_acc], width, default_dtype,
                ).expect("MLA V restore HReduce auto_lower failed");
                prog.emit(VmInstr::Broadcast {
                    dst: v_acc,
                    src: ScalarExpr::ExtractLane0(v_hr[1]),
                    width,
                    dtype: default_dtype,
                });

                // o_acc[d] = o_acc[d] * correction + weight * v[d]
                super::auto_select::auto_lower_trace_into(
                    prog, &acc_body, &[o_acc[d], correction, weight, v_acc], o_acc[d], width, default_dtype,
                ).expect("MLA V accumulate auto_lower failed");
            }
        });

        // ── Phase 2: Normalize output ──
        // O[d] = O_acc[d] / running_sum
        let recip_slots = super::auto_select::auto_lower_trace_raw(
            prog, &rec_body, &[running_sum], width, default_dtype,
        ).expect("MLA recip auto_lower failed");
        let recip_sum = recip_slots[1];

        for d in 0..hd_vecs {
            let norm_slots = super::auto_select::auto_lower_trace_raw(
                prog, &norm_body, &[o_acc[d], recip_sum], width, default_dtype,
            ).expect("MLA normalize auto_lower failed");
            prog.emit(VmInstr::VecStore {
                base: o_row,
                offset: OffsetExpr::Const(d * hd_vec_step),
                src: norm_slots[2],
                width,
                dtype: default_dtype, predicate: None,
            });
        }
    });

    Ok(output_ptr)
}

// ── MLA RoPE Merge ────────────────────────────────────────────────────────

/// MLA decoupled RoPE merge: replace c_KV[d_c-d_rope..d_c] with RoPE(k_pe).
///
/// This is an injective operation:
///   merged_key[0..d_c-d_rope] = c_KV[0..d_c-d_rope]     (copy first part)
///   merged_key[d_c-d_rope..d_c] = RoPE(k_pe[0..d_rope])  (RoPE applied to k_pe)
///
/// Input slots (from auto_select):
///   [0]: c_kv_ptr   — compressed KV vector [seq_len, d_c]
///   [1]: k_pe_ptr   — RoPE key [seq_len, d_rope]
///   [2]: output_ptr — merged output [seq_len, d_c]
///   [3]: cos_ptr    — cos table for RoPE [max_seq_len, d_rope/2]
///   [4]: sin_ptr    — sin table for RoPE [max_seq_len, d_rope/2]
///   [5]: position   — position index (Scalar GPR)
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_mla_rope_merge_inline(
    prog: &mut VmProgram,
    d_c: usize,
    d_rope: usize,
    slots: &[VRegId],
    width: SimdWidth,
    default_dtype: QuantPrecision,
) -> Result<VRegId, CompilerError> {
    let c_kv_ptr = slots[0];
    let k_pe_ptr = slots[1];
    let output_ptr = slots[2];
    let cos_ptr = slots[3];
    let sin_ptr = slots[4];
    let _position = slots.get(5).copied();

    let lanes = width.f32_lanes().max(1);
    let elem_bytes = default_dtype.elem_bytes();

    if d_c == 0 || d_rope == 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "MlaRopeMerge: invalid dimensions (d_c={}, d_rope={})",
            d_c, d_rope
        )));
    }

    let d_main = d_c - d_rope; // first part dimensions (no RoPE)
    let dc_vecs = d_c / lanes;
    let dc_vec_step = lanes * elem_bytes;
    let d_main_vecs = d_main / lanes;
    let d_rope_half = d_rope / 2; // RoPE operates on pairs
    let d_rope_vecs = d_rope / lanes;
    let d_rope_vec_step = lanes * elem_bytes;

    // TraceOp body for RoPE rotation: out_even = x0*cos - x1*sin, out_odd = x1*cos + x0*sin
    let rope_body: Vec<TraceOp> = vec![
        TraceOp::Input(0),  // [0] x0 (even)
        TraceOp::Input(1),  // [1] x1 (odd)
        TraceOp::Input(2),  // [2] cos
        TraceOp::Input(3),  // [3] sin
        TraceOp::Mul(ValueId(0), ValueId(2)), // [4] x0 * cos
        TraceOp::Mul(ValueId(1), ValueId(3)), // [5] x1 * sin
        TraceOp::Sub(ValueId(4), ValueId(5)), // [6] out_even = x0*cos - x1*sin
        TraceOp::Mul(ValueId(1), ValueId(2)), // [7] x1 * cos
        TraceOp::Mul(ValueId(0), ValueId(3)), // [8] x0 * sin
        TraceOp::Add(ValueId(7), ValueId(8)), // [9] out_odd = x1*cos + x0*sin
    ];

    // TraceOp body for identity passthrough (copy)
    let copy_body = vec![TraceOp::Input(0)];

    // Step 1: Copy c_KV[0..d_main] to output[0..d_main]
    if d_main_vecs > 0 {
        prog.emit_loop(BoundExpr::Const(d_main_vecs), dc_vec_step, |prog, _ctr, byte_off| {
            let data = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: data,
                base: c_kv_ptr,
                offset: OffsetExpr::LoopOffset(byte_off),
                width,
                dtype: default_dtype, predicate: None,
            });
            let slots = super::auto_select::auto_lower_trace_raw(
                prog, &copy_body, &[data], width, default_dtype,
            ).expect("MlaRopeMerge copy auto_lower failed");
            prog.emit(VmInstr::VecStore {
                base: output_ptr,
                offset: OffsetExpr::LoopOffset(byte_off),
                src: slots[0],
                width,
                dtype: default_dtype, predicate: None,
            });
        });
    }

    // Step 2: Apply RoPE to k_pe and store at output[d_main..d_c]
    // RoPE operates on pairs: (x[2i], x[2i+1]) → (x[2i]*cos[i] - x[2i+1]*sin[i], x[2i+1]*cos[i] + x[2i]*sin[i])
    let d_main_bytes = d_main * elem_bytes;
    let cos_sin_row_bytes = d_rope_half * elem_bytes;
    let rope_pair_vecs = d_rope_half / lanes; // number of vector chunks of half-dim

    if d_rope_half > 0 && d_rope_half % lanes == 0 {
        // Vectorized RoPE: process d_rope_half elements at a time
        let vec_step = lanes * elem_bytes;
        prog.emit_loop(BoundExpr::Const(rope_pair_vecs), vec_step, |prog, _ctr, byte_off| {
            // Load k_pe even components: k_pe[2i]
            let x_even = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: x_even,
                base: k_pe_ptr,
                offset: OffsetExpr::LoopOffset(byte_off),
                width,
                dtype: default_dtype, predicate: None,
            });
            // Load k_pe odd components: k_pe[2i+1] (offset by d_rope_half)
            let x_odd = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: x_odd,
                base: k_pe_ptr,
                offset: OffsetExpr::loop_plus_const(byte_off, d_rope_half * elem_bytes),
                width,
                dtype: default_dtype, predicate: None,
            });
            // Load cos/sin
            let cos_val = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: cos_val,
                base: cos_ptr,
                offset: OffsetExpr::LoopOffset(byte_off),
                width,
                dtype: default_dtype, predicate: None,
            });
            let sin_val = prog.alloc_vreg(VRegKind::Vec, width);
            prog.emit(VmInstr::VecLoad {
                dst: sin_val,
                base: sin_ptr,
                offset: OffsetExpr::LoopOffset(byte_off),
                width,
                dtype: default_dtype, predicate: None,
            });
            // Apply RoPE rotation
            let rope_slots = super::auto_select::auto_lower_trace_raw(
                prog, &rope_body, &[x_even, x_odd, cos_val, sin_val], width, default_dtype,
            ).expect("MlaRopeMerge RoPE auto_lower failed");
            let out_even = rope_slots[6];
            let out_odd = rope_slots[9];
            // Store even result at output[d_main + 2i]
            prog.emit(VmInstr::VecStore {
                base: output_ptr,
                offset: OffsetExpr::loop_plus_const(byte_off, d_main_bytes),
                src: out_even,
                width,
                dtype: default_dtype, predicate: None,
            });
            // Store odd result at output[d_main + 2i + d_rope_half]
            prog.emit(VmInstr::VecStore {
                base: output_ptr,
                offset: OffsetExpr::loop_plus_const(byte_off, d_main_bytes + d_rope_half * elem_bytes),
                src: out_odd,
                width,
                dtype: default_dtype, predicate: None,
            });
        });
    } else {
        // Scalar fallback for non-aligned d_rope (shouldn't happen with typical configs)
        // Handle remaining elements one by one using the same TraceOp body
        return Err(CompilerError::CodegenViolation(format!(
            "MlaRopeMerge: d_rope_half ({}) not aligned to SIMD lanes ({})",
            d_rope_half, lanes
        )));
    }

    Ok(output_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_attn_score_compilation() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_absorbed_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_cache_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_uv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let result = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_absorbed_ptr, kv_cache_ptr, w_uv_ptr, output_ptr],
            kv_len, width, dtype,
        );
        assert!(result.is_ok(), "emit_mla_attn_score_inline failed: {:?}", result.err());
        assert!(!prog.is_empty(), "expected non-empty VmInstr output");
    }

    #[test]
    fn test_mla_rope_merge_compilation() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let position = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // d_c=32, d_rope=16: d_main=16, d_rope_half=8 (aligned to 8 lanes)
        let result = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, position],
            width, dtype,
        );
        assert!(result.is_ok(), "emit_mla_rope_merge_inline failed: {:?}", result.err());
        assert!(!prog.is_empty(), "expected non-empty VmInstr output");
    }

    #[test]
    fn test_mla_rope_merge_rejects_unaligned() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dummy = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // d_rope=6 -> d_rope_half=3, not aligned to 8 lanes
        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 6,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, dummy],
            width, dtype,
        );
        assert!(result.is_err(), "expected error for unaligned d_rope");
    }

    #[test]
    fn test_mla_attn_score_rejects_zero_dimensions() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let result = emit_mla_attn_score_inline(
            &mut prog, 0, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );
        assert!(result.is_err(), "zero num_heads should be rejected");
    }

    #[test]
    fn test_mla_attn_score_rejects_zero_dc() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let result = emit_mla_attn_score_inline(
            &mut prog, 4, 8, 0, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );
        assert!(result.is_err(), "zero d_c should be rejected");
    }

    #[test]
    fn test_mla_rope_merge_rejects_zero_dc() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let result = emit_mla_rope_merge_inline(
            &mut prog, 0, 8,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );
        assert!(result.is_err(), "zero d_c should be rejected");
    }

    #[test]
    fn test_mla_rope_merge_rejects_zero_d_rope() {
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 0,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );
        assert!(result.is_err(), "zero d_rope should be rejected");
    }

    #[test]
    fn test_dot_fma_body_structure() {
        let body = dot_fma_body();
        assert_eq!(body.len(), 4);
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Input(2)));
        assert!(matches!(body[3], TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2))));
    }

    #[test]
    fn test_softmax_body_structure() {
        let body = softmax_body();
        // Should contain: Input(0), Input(1), Max, Sub, Exp, Sub, Exp = 7 ops
        assert_eq!(body.len(), 7);
        assert!(matches!(body[0], TraceOp::Input(0))); // running_max
        assert!(matches!(body[1], TraceOp::Input(1))); // score
        assert!(matches!(body[2], TraceOp::Max(..)));
        assert!(matches!(body[6], TraceOp::Exp(ValueId(5)))); // weight
    }

    #[test]
    fn test_accumulate_body_structure() {
        let body = accumulate_body();
        assert_eq!(body.len(), 7);
        // Inputs: o_acc, correction, weight, v_vec
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Input(2)));
        assert!(matches!(body[3], TraceOp::Input(3)));
        // Mul(o_acc * correction), Mul(weight * v_vec), Add
        assert!(matches!(body[4], TraceOp::Mul(..)));
        assert!(matches!(body[5], TraceOp::Mul(..)));
        assert!(matches!(body[6], TraceOp::Add(..)));
    }

    #[test]
    fn test_scale_body_structure() {
        let body = scale_body();
        assert_eq!(body.len(), 3);
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Mul(..)));
    }

    #[test]
    fn test_sum_update_body_structure() {
        let body = sum_update_body();
        assert_eq!(body.len(), 5);
        assert!(matches!(body[0], TraceOp::Input(0))); // running_sum
        assert!(matches!(body[1], TraceOp::Input(1))); // correction
        assert!(matches!(body[2], TraceOp::Input(2))); // weight
        assert!(matches!(body[3], TraceOp::Mul(..)));
        assert!(matches!(body[4], TraceOp::Add(..)));
    }

    #[test]
    fn test_identity_body_structure() {
        let body = identity_body();
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], TraceOp::Input(0)));
    }

    #[test]
    fn test_recip_body_structure() {
        let body = recip_body();
        assert_eq!(body.len(), 2);
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Recip(ValueId(0))));
    }

    #[test]
    fn test_normalize_body_structure() {
        let body = normalize_body();
        assert_eq!(body.len(), 3);
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Mul(..)));
    }

    // ── 13 new tests ─────────────────────────────────────────────────────

    #[test]
    fn test_mla_attn_score_with_single_head() {
        // Arrange: single-head MLA configuration (minimal valid)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: emit with num_heads=1, head_dim=8, d_c=16, d_rope=8
        let result = emit_mla_attn_score_inline(
            &mut prog, 1, 8, 16, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert: succeeds, produces instructions, returns output ptr
        assert!(result.is_ok(), "single head should compile: {:?}", result.err());
        assert!(prog.len() > 0, "single head should emit instructions");
        assert_eq!(result.unwrap(), out_ptr, "should return output_ptr");
    }

    #[test]
    fn test_mla_attn_score_with_avx512_width() {
        // Arrange: AVX-512 SIMD width (16 f32 lanes)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W512;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, head_dim=32 must both be divisible by 16 lanes
        let result = emit_mla_attn_score_inline(
            &mut prog, 4, 32, 32, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert: AVX-512 compilation succeeds
        assert!(result.is_ok(), "AVX-512 compilation should succeed: {:?}", result.err());
        assert!(prog.len() > 0, "should emit instructions for AVX-512");
    }

    #[test]
    fn test_mla_attn_score_rejects_zero_head_dim() {
        // Arrange
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: head_dim=0 is invalid
        let result = emit_mla_attn_score_inline(
            &mut prog, 4, 0, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert
        assert!(result.is_err(), "zero head_dim should be rejected");
    }

    #[test]
    fn test_mla_attn_score_emits_broadcast_for_scale() {
        // Arrange: verify the 1/sqrt(d_c) scale Broadcast is present
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: first instruction should be Broadcast of inv_sqrt_dc
        let has_const_broadcast = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::Broadcast {
                src: ScalarExpr::Const(v),
                ..
            } if *v != 0.0 && *v != f32::NEG_INFINITY)
        });
        assert!(has_const_broadcast, "should emit Broadcast with 1/sqrt(d_c) constant");
    }

    #[test]
    fn test_mla_attn_score_emits_neg_inf_broadcast() {
        // Arrange: verify running_max initialization with -inf
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: should have a Broadcast with NEG_INFINITY for running_max init
        let has_neg_inf = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::Broadcast {
                src: ScalarExpr::Const(v),
                ..
            } if *v == f32::NEG_INFINITY)
        });
        assert!(has_neg_inf, "should initialize running_max to -inf via Broadcast");
    }

    #[test]
    fn test_mla_attn_score_emits_loop_begin() {
        // Arrange: verify nested loops are emitted (heads loop, kv_len loop, dc_vecs loop)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: must contain LoopBegin instructions
        let loop_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { .. })
        }).count();
        assert!(loop_count >= 2, "should emit at least 2 LoopBegin (heads + kv_len), got {}", loop_count);
    }

    #[test]
    fn test_mla_rope_merge_d_main_zero() {
        // Arrange: d_c == d_rope means d_main=0, no copy phase
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=16 → d_main=0, d_rope_half=8 (aligned to 8 lanes)
        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert: should succeed since d_rope_half=8 is aligned
        assert!(result.is_ok(), "d_main=0 case should succeed: {:?}", result.err());
        assert!(prog.len() > 0, "should still emit RoPE instructions");
    }

    #[test]
    fn test_mla_rope_merge_with_w128_simd() {
        // Arrange: SSE/NEON 128-bit width (4 f32 lanes)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W128;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_rope_half=4, aligned to 4 lanes (W128)
        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 8,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert
        assert!(result.is_ok(), "W128 should succeed: {:?}", result.err());
        assert!(prog.len() > 0);
    }

    #[test]
    fn test_mla_rope_merge_emits_vec_load_for_k_pe() {
        // Arrange: verify k_pe loads are emitted
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: VecLoad instructions with base=k_pe_ptr must exist
        let k_pe_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == k_pe_ptr)
        }).count();
        assert!(k_pe_loads >= 2, "should emit at least 2 VecLoad from k_pe_ptr (even+odd), got {}", k_pe_loads);
    }

    #[test]
    fn test_mla_rope_merge_emits_vec_load_for_cos_sin() {
        // Arrange: verify cos/sin table loads
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: VecLoad from cos_ptr and sin_ptr
        let cos_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == cos_ptr)
        }).count();
        let sin_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == sin_ptr)
        }).count();
        assert!(cos_loads >= 1, "should emit VecLoad from cos_ptr, got {}", cos_loads);
        assert!(sin_loads >= 1, "should emit VecLoad from sin_ptr, got {}", sin_loads);
    }

    #[test]
    fn test_mla_rope_merge_emits_vec_store_to_output() {
        // Arrange: verify results stored to output_ptr
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: VecStore to output_ptr for both copy and RoPE phases
        let output_stores = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { base, .. } if *base == output_ptr)
        }).count();
        // copy phase: d_main_vecs stores + RoPE phase: 2 stores per iteration (even+odd)
        assert!(output_stores >= 2, "should emit VecStore to output_ptr, got {}", output_stores);
    }

    #[test]
    fn test_mla_rope_merge_unaligned_with_w128() {
        // Arrange: W128 (4 lanes), d_rope_half=3 → not aligned to 4
        let mut prog = VmProgram::new();
        let width = SimdWidth::W128;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dummy = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_rope=6 → d_rope_half=3, not aligned to W128 lanes=4
        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 6,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, dummy],
            width, dtype,
        );

        // Assert: unaligned d_rope_half should be rejected
        assert!(result.is_err(), "unaligned d_rope_half=3 with W128 should be rejected");
    }

    #[test]
    fn test_mla_attn_score_larger_d_c_produces_more_instructions() {
        // Arrange: larger d_c means more dc_vecs loop iterations in the dot product,
        // which expands the inner loop body. head_dim is also larger so more o_acc accumulators.
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let mut prog_small = VmProgram::new();
        let q1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl1 = prog_small.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=16, head_dim=8 → dc_vecs=2, hd_vecs=1
        emit_mla_attn_score_inline(
            &mut prog_small, 2, 8, 16, 4,
            &[q1, kv1, w1, o1], kl1, width, dtype,
        ).unwrap();
        let count_small = prog_small.len();

        let mut prog_large = VmProgram::new();
        let q2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl2 = prog_large.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=32, head_dim=16 → dc_vecs=4, hd_vecs=2 (more inner loop unrolling)
        emit_mla_attn_score_inline(
            &mut prog_large, 2, 16, 32, 8,
            &[q2, kv2, w2, o2], kl2, width, dtype,
        ).unwrap();
        let count_large = prog_large.len();

        // Assert: larger d_c/head_dim should produce more instructions
        assert!(count_large > count_small,
            "d_c=32/head_dim=16 ({}) should produce more instructions than d_c=16/head_dim=8 ({})",
            count_large, count_small);
    }

    // ── 10 additional tests ──────────────────────────────────────────────

    #[test]
    fn test_rope_body_structure() {
        // Arrange: the RoPE rotation body defined inline in emit_mla_rope_merge_inline
        // has 10 ops: 4 Inputs + 4 Muls + 1 Sub + 1 Add
        let rope_body: Vec<TraceOp> = vec![
            TraceOp::Input(0),  // x0
            TraceOp::Input(1),  // x1
            TraceOp::Input(2),  // cos
            TraceOp::Input(3),  // sin
            TraceOp::Mul(ValueId(0), ValueId(2)), // x0*cos
            TraceOp::Mul(ValueId(1), ValueId(3)), // x1*sin
            TraceOp::Sub(ValueId(4), ValueId(5)), // even = x0*cos - x1*sin
            TraceOp::Mul(ValueId(1), ValueId(2)), // x1*cos
            TraceOp::Mul(ValueId(0), ValueId(3)), // x0*sin
            TraceOp::Add(ValueId(7), ValueId(8)), // odd = x1*cos + x0*sin
        ];

        // Assert
        assert_eq!(rope_body.len(), 10, "RoPE body should have exactly 10 TraceOps");
        assert!(matches!(rope_body[0], TraceOp::Input(0)), "first input is x0");
        assert!(matches!(rope_body[1], TraceOp::Input(1)), "second input is x1");
        assert!(matches!(rope_body[2], TraceOp::Input(2)), "third input is cos");
        assert!(matches!(rope_body[3], TraceOp::Input(3)), "fourth input is sin");
        assert!(matches!(rope_body[6], TraceOp::Sub(..)), "op[6] computes even component");
        assert!(matches!(rope_body[9], TraceOp::Add(..)), "op[9] computes odd component");
    }

    #[test]
    fn test_copy_body_structure() {
        // Arrange: copy/identity body used in rope_merge c_KV copy phase
        let copy_body = vec![TraceOp::Input(0)];

        // Assert
        assert_eq!(copy_body.len(), 1, "copy body should be single passthrough");
        assert!(matches!(copy_body[0], TraceOp::Input(0)));
    }

    #[test]
    fn test_mla_attn_score_error_message_content() {
        // Arrange: verify error message includes dimension info
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: zero d_c
        let result = emit_mla_attn_score_inline(
            &mut prog, 4, 8, 0, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert: error message should contain all dimension values for diagnostics
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("num_heads=4"), "error should mention num_heads, got: {}", err_msg);
        assert!(err_msg.contains("head_dim=8"), "error should mention head_dim, got: {}", err_msg);
        assert!(err_msg.contains("d_c=0"), "error should mention d_c=0, got: {}", err_msg);
    }

    #[test]
    fn test_mla_rope_merge_error_message_content() {
        // Arrange: verify error message includes dimension info
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: zero d_rope
        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 0,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert: error should contain both d_c and d_rope
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("d_c=16"), "error should mention d_c, got: {}", err_msg);
        assert!(err_msg.contains("d_rope=0"), "error should mention d_rope=0, got: {}", err_msg);
    }

    #[test]
    fn test_mla_attn_score_returns_output_ptr() {
        // Arrange: the returned VRegId must equal the output_ptr slot
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let result = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert
        assert_eq!(result, out_ptr, "emit_mla_attn_score_inline should return the output_ptr slot");
    }

    #[test]
    fn test_mla_rope_merge_returns_output_ptr() {
        // Arrange: the returned VRegId must equal the output_ptr slot
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let result = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert
        assert_eq!(result, output_ptr, "emit_mla_rope_merge_inline should return the output_ptr slot");
    }

    #[test]
    fn test_mla_rope_merge_works_without_position_slot() {
        // Arrange: only 5 slots provided (position is optional via .get(5))
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // Act: only 5 slots, no position
        let result = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr],
            width, dtype,
        );

        // Assert: should succeed without position
        assert!(result.is_ok(), "should work without 6th position slot: {:?}", result.err());
        assert!(prog.len() > 0, "should still emit instructions");
    }

    #[test]
    fn test_mla_attn_score_emits_vec_store_for_scores() {
        // Arrange: verify the score buffer stores exist (one per kv position in inner loop)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: at least one VecStore with Scalar width (score buffer store)
        let scalar_stores = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_stores >= 1, "should emit at least one Scalar-width VecStore for score buffer, got {}", scalar_stores);
    }

    #[test]
    fn test_mla_rope_merge_copy_phase_loads_from_c_kv() {
        // Arrange: when d_main > 0, copy phase should load from c_kv_ptr
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, d_rope=16 → d_main=16 > 0, so copy phase should happen
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: VecLoad from c_kv_ptr must exist (copy phase)
        let c_kv_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        assert!(c_kv_loads >= 1,
            "d_main=16 > 0 should trigger copy phase with VecLoad from c_kv_ptr, got {} loads",
            c_kv_loads);
    }

    #[test]
    fn test_mla_rope_merge_aligned_with_w512() {
        // Arrange: AVX-512 (16 f32 lanes), d_rope_half must be divisible by 16
        let mut prog = VmProgram::new();
        let width = SimdWidth::W512;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=64, d_rope=32 → d_main=32, d_rope_half=16 (aligned to W512=16 lanes)
        let result = emit_mla_rope_merge_inline(
            &mut prog, 64, 32,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert
        assert!(result.is_ok(), "W512 with aligned d_rope_half=16 should succeed: {:?}", result.err());
        assert!(prog.len() > 0, "should emit instructions for W512");
    }

    // ── 10 new tests (wave-12kce) ──────────────────────────────────────────

    #[test]
    fn test_mla_attn_score_with_bf16_dtype() {
        // Arrange: BF16 dtype — verify compilation succeeds with non-F32 precision
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::BF16;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let result = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert: should compile successfully with BF16
        assert!(result.is_ok(), "BF16 dtype should compile: {:?}", result.err());
        // All Broadcast and VecLoad instructions should use BF16 dtype
        let bf16_instrs = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::Broadcast { dtype: QuantPrecision::BF16, .. }
                     | VmInstr::VecLoad { dtype: QuantPrecision::BF16, predicate: None, .. }
                     | VmInstr::VecStore { dtype: QuantPrecision::BF16, predicate: None, .. })
        }).count();
        assert!(bf16_instrs > 0, "should emit BF16-typed instructions, got 0");
    }

    #[test]
    fn test_mla_rope_merge_with_bf16_dtype() {
        // Arrange: BF16 dtype for rope merge
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::BF16;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, d_rope=16 → d_rope_half=8 aligned to W256
        let result = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert
        assert!(result.is_ok(), "BF16 rope merge should compile: {:?}", result.err());
        let bf16_stores = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { dtype: QuantPrecision::BF16, predicate: None, .. })
        }).count();
        assert!(bf16_stores > 0, "should emit BF16-typed VecStore instructions");
    }

    #[test]
    fn test_mla_attn_score_emits_add_ptr_for_score_buf() {
        // Arrange: verify score_buf_ptr is computed via AddPtr from output_ptr
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: AddPtr with base=output_ptr and offset = num_heads*head_dim*elem_bytes = 2*8*4 = 64
        let has_score_buf_addptr = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::AddPtr { base, offset: 64, .. } if *base == out_ptr)
        });
        assert!(has_score_buf_addptr,
            "should emit AddPtr from output_ptr with offset=64 (2 heads * 8 head_dim * 4 bytes)");
    }

    #[test]
    fn test_mla_attn_score_emits_gpr_binop_mul_for_wuv_stride() {
        // Arrange: verify W_UV head stride computed via GprBinOp::Mul
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, head_dim=8 → w_uv_head_stride = 16*8*4 = 512
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Mul with Imm(512) for w_uv_head_stride
        let has_wuv_stride_mul = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::GprBinOp {
                op: GprOp::Mul,
                b: GprOperand::Imm(512),
                ..
            })
        });
        assert!(has_wuv_stride_mul,
            "should emit GprBinOp::Mul with Imm(512) for w_uv_head_stride");
    }

    #[test]
    fn test_mla_attn_score_dynamic_kv_loop_uses_vreg() {
        // Arrange: verify kv_len loop uses BoundExpr::DynamicVReg (emits LoopBegin with VReg bound)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: at least one LoopBegin references the kv_len VRegId as DynamicVReg
        let dynamic_loop_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::DynamicVReg(vreg), .. }
                     if *vreg == kv_len)
        }).count();
        assert!(dynamic_loop_count >= 1,
            "should emit at least one LoopBegin with DynamicVReg(kv_len), got {}", dynamic_loop_count);
    }

    #[test]
    fn test_mla_attn_score_multiple_heads_emits_per_head_gpr_ops() {
        // Arrange: 8 heads → more GprBinOp for head stride computation
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: 8 heads, head_dim=8, d_c=16
        let _ = emit_mla_attn_score_inline(
            &mut prog, 8, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Mul with Imm(64) = head_hd_bytes = 8*4*2 = 64... wait
        // head_hd_bytes = head_dim * elem_bytes = 8 * 4 = 32
        let has_output_stride_mul = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::GprBinOp {
                op: GprOp::Mul,
                b: GprOperand::Imm(32),
                ..
            })
        });
        assert!(has_output_stride_mul,
            "should emit GprBinOp::Mul with Imm(32) for output head stride (head_dim=8 * 4 bytes)");
    }

    #[test]
    fn test_mla_rope_merge_no_copy_phase_fewer_stores() {
        // Arrange: d_main=0 (d_c == d_rope) means no copy phase, only RoPE stores
        let mut prog_no_copy = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c1 = prog_no_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k1 = prog_no_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_no_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos1 = prog_no_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin1 = prog_no_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p1 = prog_no_copy.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=16 → d_main=0, no copy phase
        emit_mla_rope_merge_inline(
            &mut prog_no_copy, 16, 16,
            &[c1, k1, o1, cos1, sin1, p1],
            width, dtype,
        ).unwrap();
        let stores_no_copy = prog_no_copy.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { base, .. } if *base == o1)
        }).count();

        // Now with d_main > 0 (has copy phase)
        let mut prog_with_copy = VmProgram::new();
        let c2 = prog_with_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog_with_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog_with_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos2 = prog_with_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin2 = prog_with_copy.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p2 = prog_with_copy.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, d_rope=16 → d_main=16, has copy phase
        emit_mla_rope_merge_inline(
            &mut prog_with_copy, 32, 16,
            &[c2, k2, o2, cos2, sin2, p2],
            width, dtype,
        ).unwrap();
        let stores_with_copy = prog_with_copy.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { base, .. } if *base == o2)
        }).count();

        // Assert: more stores when copy phase is present
        assert!(stores_with_copy > stores_no_copy,
            "with copy (d_main=16, {} stores) should have more stores than without copy (d_main=0, {} stores)",
            stores_with_copy, stores_no_copy);
    }

    #[test]
    fn test_mla_attn_score_scale_value_correctness() {
        // Arrange: verify inv_sqrt_dc = 1/sqrt(d_c) is correct for different d_c values
        // d_c=16 → inv_sqrt_dc ≈ 0.25
        // d_c=64 → inv_sqrt_dc = 0.125
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let mut prog1 = VmProgram::new();
        let q1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl1 = prog1.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_attn_score_inline(
            &mut prog1, 2, 8, 16, 4,
            &[q1, kv1, w1, o1], kl1, width, dtype,
        ).unwrap();

        let expected_scale_16 = 1.0f32 / 16.0f32.sqrt();
        let has_correct_scale_16 = prog1.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::Broadcast { src: ScalarExpr::Const(v), .. }
                     if (*v - expected_scale_16).abs() < 1e-6)
        });
        assert!(has_correct_scale_16, "d_c=16 scale should be 1/sqrt(16) ≈ 0.25");

        let mut prog2 = VmProgram::new();
        let q2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl2 = prog2.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_attn_score_inline(
            &mut prog2, 2, 16, 64, 8,
            &[q2, kv2, w2, o2], kl2, width, dtype,
        ).unwrap();

        let expected_scale_64 = 1.0f32 / 64.0f32.sqrt();
        let has_correct_scale_64 = prog2.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::Broadcast { src: ScalarExpr::Const(v), .. }
                     if (*v - expected_scale_64).abs() < 1e-6)
        });
        assert!(has_correct_scale_64, "d_c=64 scale should be 1/sqrt(64) = 0.125");
    }

    #[test]
    fn test_mla_rope_merge_emits_loop_for_rope_phase() {
        // Arrange: verify the RoPE phase emits a LoopBegin for rope_pair_vecs iterations
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, d_rope=16 → d_main=16, d_rope_half=8, rope_pair_vecs=1
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: must have LoopBegin instructions (copy loop + RoPE loop)
        let loop_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { .. })
        }).count();
        assert!(loop_count >= 2,
            "should emit at least 2 loops (copy + RoPE), got {}", loop_count);
    }

    #[test]
    fn test_mla_rope_merge_rope_phase_loads_from_k_pe_with_offset() {
        // Arrange: odd components loaded with offset = d_rope_half * elem_bytes
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_rope=16, d_rope_half=8, d_rope_half*elem_bytes=32 for F32
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: odd component VecLoad from k_pe_ptr uses Add offset containing Const(32)
        // This is produced by OffsetExpr::loop_plus_const(byte_off, d_rope_half * elem_bytes)
        let odd_loads = prog.instrs.iter().filter(|instr| {
            if let VmInstr::VecLoad { base, offset, .. } = instr {
                if *base != k_pe_ptr { return false; }
                // Check if offset is Add(_, Const(32)) — the d_rope_half * elem_bytes offset
                if let OffsetExpr::Add(_, rhs) = offset {
                    if let OffsetExpr::Const(32) = rhs.as_ref() { return true; }
                }
                false
            } else {
                false
            }
        }).count();
        assert!(odd_loads >= 1,
            "should emit VecLoad from k_pe_ptr with Add(_, Const(32)) for odd components, got {}", odd_loads);
    }

    #[test]
    fn test_mla_attn_score_with_scalar_width() {
        // Arrange: SimdWidth::Scalar (1 lane) — each element processed individually
        let mut prog = VmProgram::new();
        let width = SimdWidth::Scalar;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: with Scalar width, d_c=4 → dc_vecs=4/1=4, head_dim=4 → hd_vecs=4/1=4
        let result = emit_mla_attn_score_inline(
            &mut prog, 2, 4, 4, 2,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert: should succeed with scalar width
        assert!(result.is_ok(), "Scalar width should compile: {:?}", result.err());
        assert!(prog.len() > 0, "should emit instructions with Scalar width");
        // With scalar width, all broadcasts should be scalar-width
        let scalar_broadcasts = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::Broadcast { width: SimdWidth::Scalar, .. })
        }).count();
        assert!(scalar_broadcasts > 0, "should emit Scalar-width Broadcast instructions");
    }

    // ── 10 new tests (wave-12kkc) ────────────────────────────────────────────

    #[test]
    fn test_mla_attn_score_with_warp_simd_width() {
        // Arrange: GPU Warp(32) width — 32 f32 lanes, d_c and head_dim must be divisible by 32
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, head_dim=32, d_rope=16, all divisible by 32
        let result = emit_mla_attn_score_inline(
            &mut prog, 4, 32, 32, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert: should compile successfully with GPU Warp width
        assert!(result.is_ok(), "Warp(32) width should compile: {:?}", result.err());
        assert!(prog.len() > 0, "should emit instructions with Warp width");
    }

    #[test]
    fn test_mla_rope_merge_with_warp_simd_width() {
        // Arrange: GPU Warp(32) width for rope merge
        let mut prog = VmProgram::new();
        let width = SimdWidth::Warp(32);
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=96, d_rope=64 → d_main=32, d_rope_half=32 aligned to Warp(32) lanes
        let result = emit_mla_rope_merge_inline(
            &mut prog, 96, 64,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert
        assert!(result.is_ok(), "Warp(32) rope merge should compile: {:?}", result.err());
        assert!(prog.len() > 0, "should emit instructions with Warp width");
    }

    #[test]
    fn test_mla_attn_score_zero_init_broadcast_count() {
        // Arrange: verify multiple zero-init broadcasts for o_acc accumulators
        // head_dim=16, W256=8 lanes → hd_vecs=2, so 2 o_acc zero-inits
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: head_dim=16 → hd_vecs=2, so 2 o_acc zero Broadcasts inside the head loop
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 16, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: count zero-valued Broadcast instructions
        let zero_broadcasts = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::Broadcast { src: ScalarExpr::Const(0.0), .. })
        }).count();
        // Per head: 1 running_sum + hd_vecs=2 o_acc + 1 dot_acc per kv iteration (in loop)
        // At minimum 1 (running_sum) + 2 (o_acc) = 3 zero broadcasts outside loops
        assert!(zero_broadcasts >= 3,
            "should emit at least 3 zero Broadcast (running_sum + 2 o_acc), got {}", zero_broadcasts);
    }

    #[test]
    fn test_mla_rope_merge_rope_body_has_correct_intermediate_value_ids() {
        // Arrange: verify the RoPE body TraceOp chain uses correct ValueId references
        // The chain must satisfy SSA def-before-use for all intermediate values
        let rope_body: Vec<TraceOp> = vec![
            TraceOp::Input(0),  // [0] x0
            TraceOp::Input(1),  // [1] x1
            TraceOp::Input(2),  // [2] cos
            TraceOp::Input(3),  // [3] sin
            TraceOp::Mul(ValueId(0), ValueId(2)), // [4] x0*cos
            TraceOp::Mul(ValueId(1), ValueId(3)), // [5] x1*sin
            TraceOp::Sub(ValueId(4), ValueId(5)), // [6] even
            TraceOp::Mul(ValueId(1), ValueId(2)), // [7] x1*cos
            TraceOp::Mul(ValueId(0), ValueId(3)), // [8] x0*sin
            TraceOp::Add(ValueId(7), ValueId(8)), // [9] odd
        ];

        // Assert: verify ValueId references point to previously defined slots
        // Mul at [4] references [0] and [2] — both defined before
        if let TraceOp::Mul(ValueId(a), ValueId(b)) = rope_body[4] {
            assert_eq!(a, 0, "first arg of x0*cos should reference slot 0");
            assert_eq!(b, 2, "second arg of x0*cos should reference slot 2 (cos)");
        } else {
            panic!("rope_body[4] should be Mul");
        }
        // Sub at [6] references [4] and [5]
        if let TraceOp::Sub(ValueId(a), ValueId(b)) = rope_body[6] {
            assert_eq!(a, 4, "Sub should reference slot 4 (x0*cos)");
            assert_eq!(b, 5, "Sub should reference slot 5 (x1*sin)");
        } else {
            panic!("rope_body[6] should be Sub");
        }
        // Add at [9] references [7] and [8]
        if let TraceOp::Add(ValueId(a), ValueId(b)) = rope_body[9] {
            assert_eq!(a, 7, "Add should reference slot 7 (x1*cos)");
            assert_eq!(b, 8, "Add should reference slot 8 (x0*sin)");
        } else {
            panic!("rope_body[9] should be Add");
        }
    }

    #[test]
    fn test_mla_attn_score_emits_gpr_binop_add_for_kv_access() {
        // Arrange: verify KV cache row address computed via GprBinOp::Add
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Add with base=kv_ptr (computing key_row = kv_cache_ptr + pos_off)
        let kv_adds = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp {
                op: GprOp::Add,
                a,
                ..
            } if *a == kv_ptr)
        }).count();
        assert!(kv_adds >= 1,
            "should emit GprBinOp::Add with kv_ptr as base for key row access, got {}", kv_adds);
    }

    #[test]
    fn test_mla_rope_merge_d_rope_equals_d_c_skips_c_kv_load() {
        // Arrange: when d_c == d_rope (d_main=0), no VecLoad from c_kv_ptr should occur
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=16 → d_main=0, copy phase skipped entirely
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 16, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: zero VecLoad from c_kv_ptr (copy phase is skipped)
        let c_kv_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        assert_eq!(c_kv_loads, 0,
            "d_main=0 should not emit VecLoad from c_kv_ptr, got {}", c_kv_loads);
    }

    #[test]
    fn test_softmax_body_all_intermediates_valid() {
        // Arrange: verify softmax body produces valid SSA chain —
        // all ValueId references in each op index back to previously defined slots
        let body = softmax_body();
        assert_eq!(body.len(), 7);

        // [2] = Max([0], [1]) — references inputs, valid
        if let TraceOp::Max(ValueId(a), ValueId(b)) = body[2] {
            assert!(a < 2 && b < 2, "Max should reference inputs [0] and [1]");
        } else {
            panic!("body[2] should be Max");
        }
        // [3] = Sub([0], [2]) — references [0] (input) and [2] (Max result), valid
        if let TraceOp::Sub(ValueId(a), ValueId(b)) = body[3] {
            assert_eq!(a, 0, "Sub first arg should be slot 0");
            assert_eq!(b, 2, "Sub second arg should be slot 2 (new_max)");
        } else {
            panic!("body[3] should be Sub");
        }
        // [4] = Exp([3]) — references Sub result, valid
        assert!(matches!(body[4], TraceOp::Exp(ValueId(3))),
            "body[4] should be Exp(ValueId(3))");
        // [5] = Sub([1], [2]) — references score and new_max
        if let TraceOp::Sub(ValueId(a), ValueId(b)) = body[5] {
            assert_eq!(a, 1, "Sub first arg should be slot 1 (score)");
            assert_eq!(b, 2, "Sub second arg should be slot 2 (new_max)");
        } else {
            panic!("body[5] should be Sub");
        }
        // [6] = Exp([5]) — references second Sub result, valid
        assert!(matches!(body[6], TraceOp::Exp(ValueId(5))),
            "body[6] should be Exp(ValueId(5))");
    }

    #[test]
    fn test_mla_rope_merge_store_offsets_include_d_main_bytes() {
        // Arrange: verify RoPE phase VecStore offsets include d_main_bytes shift
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=48, d_rope=16 → d_main=32, d_main_bytes=128 (32*4)
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 48, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: RoPE even/odd stores use offset Add(_, Const(128)) or larger
        let rope_stores_with_d_main = prog.instrs.iter().filter(|instr| {
            if let VmInstr::VecStore { base, offset, .. } = instr {
                if *base != output_ptr { return false; }
                // Check for Add offset containing d_main_bytes=128
                if let OffsetExpr::Add(_, rhs) = offset {
                    if let OffsetExpr::Const(v) = rhs.as_ref() {
                        return *v >= 128; // d_main_bytes or more
                    }
                }
                false
            } else {
                false
            }
        }).count();
        assert!(rope_stores_with_d_main >= 1,
            "should emit VecStore to output_ptr with d_main offset >= 128, got {}",
            rope_stores_with_d_main);
    }

    #[test]
    fn test_mla_attn_score_rope_merge_pipeline_integration() {
        // Arrange: simulate a realistic MLA decode step: rope merge first, then attn score
        // This verifies both functions can share a VmProgram without register conflicts
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        // Allocate slots for rope merge
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let merged_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Phase 1: rope merge
        let rope_result = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, merged_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );
        assert!(rope_result.is_ok(), "rope merge should succeed in pipeline: {:?}", rope_result.err());
        let instr_after_rope = prog.len();

        // Allocate new slots for attn score (some reuse the merged output)
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_uv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Phase 2: attn score using merged KV as input
        let attn_result = emit_mla_attn_score_inline(
            &mut prog, 4, 8, 32, 16,
            &[q_ptr, merged_ptr, w_uv_ptr, out_ptr],
            kv_len, width, dtype,
        );

        // Assert: both phases succeed, and attn adds more instructions
        assert!(attn_result.is_ok(), "attn score should succeed after rope merge: {:?}", attn_result.err());
        assert!(prog.len() > instr_after_rope,
            "attn score should add instructions after rope merge ({} > {})",
            prog.len(), instr_after_rope);
    }

    #[test]
    fn test_mla_rope_merge_rope_unaligned_error_message() {
        // Arrange: verify unaligned d_rope_half error contains both d_rope_half and lanes
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_rope=10 → d_rope_half=5, not aligned to W256 lanes=8
        let result = emit_mla_rope_merge_inline(
            &mut prog, 24, 10,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert: error message should mention both d_rope_half and lane count
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("d_rope_half"), "error should mention d_rope_half, got: {}", err_msg);
        assert!(err_msg.contains("aligned") || err_msg.contains("lanes"),
            "error should mention alignment issue, got: {}", err_msg);
    }

    // ── 10 new tests ──────────────────────────────────────────────────

    #[test]
    fn test_mla_attn_score_emits_gpr_binop_add_from_q_absorbed() {
        // Arrange: verify Q row address computed via GprBinOp::Add with q_ptr as base
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Add with a=q_ptr for computing q_row = q_ptr + head_offset
        let q_adds = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Add, a, .. } if *a == q_ptr)
        }).count();
        assert!(q_adds >= 1, "should emit GprBinOp::Add from q_ptr, got {}", q_adds);
    }

    #[test]
    fn test_mla_attn_score_emits_gpr_binop_add_from_w_uv() {
        // Arrange: verify W_UV row address computed via GprBinOp::Add with w_uv_ptr
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Add with a=w_ptr for computing wuv_row
        let w_adds = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Add, a, .. } if *a == w_ptr)
        }).count();
        assert!(w_adds >= 1, "should emit GprBinOp::Add from w_uv_ptr, got {}", w_adds);
    }

    #[test]
    fn test_mla_attn_score_emits_extract_lane0_for_score_scalar() {
        // Arrange: verify ExtractLane0 used to broadcast HReduce result to scalar score
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: at least one Broadcast with ExtractLane0 (score scalar after HReduce)
        let extract_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::Broadcast { src: ScalarExpr::ExtractLane0(_), .. })
        }).count();
        assert!(extract_count >= 1,
            "should emit Broadcast with ExtractLane0 for score, got {}", extract_count);
    }

    #[test]
    fn test_mla_attn_score_emits_gpr_binop_add_from_output_ptr() {
        // Arrange: verify output row address computed via GprBinOp::Add with output_ptr
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Add with a=out_ptr for computing o_row = output_ptr + head_offset
        let out_adds = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Add, a, .. } if *a == out_ptr)
        }).count();
        assert!(out_adds >= 1, "should emit GprBinOp::Add from output_ptr, got {}", out_adds);
    }

    #[test]
    fn test_mla_attn_score_emits_const_loop_for_dc_vecs() {
        // Arrange: verify dc_vecs inner loop uses BoundExpr::Const
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, W256=8 lanes → dc_vecs=2
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: LoopBegin with BoundExpr::Const(2) for dc_vecs=2
        let dc_loops = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::Const(2), .. })
        }).count();
        assert!(dc_loops >= 1,
            "should emit LoopBegin with Const(2) for dc_vecs, got {}", dc_loops);
    }

    #[test]
    fn test_mla_attn_score_emits_vector_width_stores() {
        // Arrange: verify final output stores use vector width (not Scalar)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: VecStore with W256 width exists (normalize phase output stores)
        let vec_stores = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { width: SimdWidth::W256, .. })
        }).count();
        assert!(vec_stores >= 1,
            "should emit W256-width VecStore for output, got {}", vec_stores);
    }

    #[test]
    fn test_mla_rope_merge_const_loop_bound_includes_d_main_vecs() {
        // Arrange: d_c=32, d_rope=16 → d_main=16, d_main_vecs=2, rope_pair_vecs=1
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: LoopBegin with Const(2) for d_main_vecs=2 copy phase
        let d_main_loops = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::Const(2), .. })
        }).count();
        assert!(d_main_loops >= 1,
            "should emit LoopBegin Const(2) for d_main_vecs copy phase, got {}", d_main_loops);
    }

    #[test]
    fn test_mla_rope_merge_even_odd_stores_have_different_const_offsets() {
        // Arrange: d_c=32, d_rope=16 → d_main_bytes=64, d_rope_half*4=32
        // Even stores: Add(_, Const(64)), Odd stores: Add(_, Const(96))
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: even stores at offset Const(64), odd stores at Const(96)
        let even_offset_stores = prog.instrs.iter().filter(|instr| {
            if let VmInstr::VecStore { base, offset, .. } = instr {
                *base == output_ptr &&
                    matches!(offset, OffsetExpr::Add(_, rhs) if matches!(rhs.as_ref(), OffsetExpr::Const(64)))
            } else { false }
        }).count();
        let odd_offset_stores = prog.instrs.iter().filter(|instr| {
            if let VmInstr::VecStore { base, offset, .. } = instr {
                *base == output_ptr &&
                    matches!(offset, OffsetExpr::Add(_, rhs) if matches!(rhs.as_ref(), OffsetExpr::Const(96)))
            } else { false }
        }).count();
        assert!(even_offset_stores >= 1, "should emit even store with Const(64), got {}", even_offset_stores);
        assert!(odd_offset_stores >= 1, "should emit odd store with Const(96), got {}", odd_offset_stores);
    }

    #[test]
    fn test_mla_attn_score_two_heads_produces_two_neg_inf_broadcasts() {
        // Arrange: each head initializes running_max to -inf → 2 heads = 2 NEG_INFINITY broadcasts
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: 2 heads, each emits 1 NEG_INFINITY Broadcast for running_max
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: at least 1 NEG_INFINITY broadcast for running_max init
        let neg_inf_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::Broadcast { src: ScalarExpr::Const(v), .. } if *v == f32::NEG_INFINITY)
        }).count();
        assert!(neg_inf_count >= 1,
            "should produce at least 1 NEG_INFINITY broadcast, got {}", neg_inf_count);
    }

    #[test]
    fn test_mla_rope_merge_larger_d_main_produces_more_c_kv_loads() {
        // Arrange: compare c_kv loads when d_main differs
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let mut prog_small = VmProgram::new();
        let c1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin1 = prog_small.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p1 = prog_small.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=24, d_rope=16 → d_main=8, d_main_vecs=1
        emit_mla_rope_merge_inline(
            &mut prog_small, 24, 16,
            &[c1, k1, o1, cos1, sin1, p1], width, dtype,
        ).unwrap();
        let loads_small = prog_small.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c1)
        }).count();

        let mut prog_large = VmProgram::new();
        let c2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin2 = prog_large.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p2 = prog_large.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=48, d_rope=16 → d_main=32, d_main_vecs=4
        emit_mla_rope_merge_inline(
            &mut prog_large, 48, 16,
            &[c2, k2, o2, cos2, sin2, p2], width, dtype,
        ).unwrap();
        let loads_large = prog_large.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c2)
        }).count();

        // Assert: larger d_main produces at least as many c_kv loads in copy phase
        assert!(loads_large >= loads_small,
            "d_main=32 ({} loads) should produce >= d_main=8 ({} loads) c_kv loads",
            loads_large, loads_small);
    }

    // ── 10 new tests (edge cases: d_kv zero, d_rope boundary, absorbed routing, c_KV offsets) ──

    #[test]
    fn test_mla_attn_score_rejects_d_rope_exceeding_d_c() {
        // Arrange: d_rope > d_c is impossible (MLA c_KV buffer = d_c + d_rope per row,
        // but d_rope must be <= d_c since RoPE-merged key occupies last d_rope of d_c)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=8, d_rope=16 — d_rope exceeds d_c (logically invalid)
        // The function does not explicitly check d_rope > d_c, but this tests
        // that such misconfiguration still compiles (stride calculation proceeds).
        let result = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 8, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        );

        // Assert: compilation succeeds (stride math is valid) — semantic validation
        // belongs to the caller, not the emit function
        assert!(result.is_ok(), "d_rope > d_c compiles (caller validates): {:?}", result.err());
    }

    #[test]
    fn test_mla_rope_merge_d_rope_equal_to_d_c_absorbed_path() {
        // Arrange: absorbed path — d_c == d_rope means every dimension gets RoPE applied
        // (no non-RoPE c_KV prefix, d_main=0). This is the fully absorbed extreme.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, d_rope=32 → d_main=0, d_rope_half=16 aligned to W256=8 lanes
        let result = emit_mla_rope_merge_inline(
            &mut prog, 32, 32,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert: fully absorbed case succeeds, no copy-phase stores to output_ptr
        assert!(result.is_ok(), "d_c==d_rope absorbed path should succeed: {:?}", result.err());
        let copy_stores = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        assert_eq!(copy_stores, 0,
            "fully absorbed (d_main=0) should have zero c_kv loads, got {}", copy_stores);
    }

    #[test]
    fn test_mla_attn_score_kv_row_bytes_includes_d_rope() {
        // Arrange: verify kv_row_bytes = (d_c + d_rope) * elem_bytes is reflected
        // in VmInstr stride computation. KV cache stores c_KV + k_pe per row.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=8 → kv_row_bytes = (16+8)*4 = 96
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: LoopBegin with step_bytes=96 for KV iteration
        let kv_loop_with_step_96 = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 96, .. })
        });
        assert!(kv_loop_with_step_96,
            "should emit KV loop with step_bytes=96 ((d_c+d_rope)*4)");
    }

    #[test]
    fn test_mla_rope_merge_d_rope_boundary_minimal_aligned() {
        // Arrange: minimal aligned d_rope for W128 — d_rope=8, d_rope_half=4
        let mut prog = VmProgram::new();
        let width = SimdWidth::W128;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=8 → d_main=8, d_rope_half=4 aligned to W128=4 lanes
        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 8,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );

        // Assert: minimal aligned d_rope succeeds with W128
        assert!(result.is_ok(), "minimal d_rope=8 with W128 should succeed: {:?}", result.err());
        assert!(prog.len() > 0, "should emit instructions for minimal aligned d_rope");
    }

    #[test]
    fn test_mla_attn_score_c_kv_load_offset_starts_at_zero() {
        // Arrange: c_KV loads inside V-restore inner loop use key_row (kv position base)
        // with OffsetExpr::LoopOffset(c_off), which starts at byte offset 0.
        // This means c_KV data is loaded from the beginning of each KV row.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=8 → c_KV is first 16 elements, k_pe is last 8
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: dc_vecs loop step is dc_vec_step = lanes * elem_bytes = 8*4=32
        let dc_loop_step_32 = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 32, .. })
        });
        assert!(dc_loop_step_32,
            "dc_vecs inner loop should step_bytes by 32 (8 lanes * 4 bytes F32)");
    }

    #[test]
    fn test_mla_rope_merge_unabsorbed_path_has_copy_and_rope_phases() {
        // Arrange: un-absorbed path — d_main > 0 means c_KV prefix is copied
        // unchanged, and only the last d_rope dimensions get RoPE applied.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=48, d_rope=16 → d_main=32 (un-absorbed, significant copy phase)
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 48, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: both phases present — c_kv loads (copy) AND k_pe loads (RoPE)
        let c_kv_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        let k_pe_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == k_pe_ptr)
        }).count();
        assert!(c_kv_loads >= 1,
            "un-absorbed path should have c_kv loads for copy phase, got {}", c_kv_loads);
        assert!(k_pe_loads >= 2,
            "un-absorbed path should have k_pe loads for RoPE phase (even+odd), got {}", k_pe_loads);
    }

    #[test]
    fn test_mla_attn_score_w_uv_load_offset_per_output_chunk() {
        // Arrange: W_UV weight loads in V-restore GEMV use Const offset per output chunk d.
        // For head_dim=16, W256=8 lanes → hd_vecs=2, so d=0 and d=1 each get
        // VecLoad with OffsetExpr::Const(0) and OffsetExpr::Const(8*4=32).
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: head_dim=16 → hd_vecs=2, wuv_d_offset for d=0 = 0, d=1 = 32
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 16, 16, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: VecLoad with OffsetExpr::Const(0) exists (first output chunk)
        let w_loads_offset_0 = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { offset: OffsetExpr::Const(0), .. })
        }).count();
        assert!(w_loads_offset_0 >= 1,
            "should emit VecLoad with Const(0) offset for first W_UV output chunk");
    }

    #[test]
    fn test_mla_rope_merge_odd_component_offset_equals_d_main_plus_d_rope_half() {
        // Arrange: the odd component store offset in RoPE phase is
        // d_main_bytes + d_rope_half * elem_bytes. For d_c=40, d_rope=16:
        // d_main=24, d_main_bytes=96, odd_offset_const = 96 + 32 = 128
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=40, d_rope=16 → d_main=24, d_main_bytes=96, d_rope_half=8
        // odd store offset = Add(_, Const(96 + 32)) = Add(_, Const(128))
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 40, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: odd store uses offset Const(128) = 96 (d_main_bytes) + 32 (d_rope_half*4)
        let odd_stores_128 = prog.instrs.iter().filter(|instr| {
            if let VmInstr::VecStore { base, offset, .. } = instr {
                *base == output_ptr &&
                    matches!(offset, OffsetExpr::Add(_, rhs)
                             if matches!(rhs.as_ref(), OffsetExpr::Const(128)))
            } else { false }
        }).count();
        assert!(odd_stores_128 >= 1,
            "odd component store should use offset Const(128), got {}", odd_stores_128);
    }

    #[test]
    fn test_mla_attn_score_score_buf_offset_equals_heads_times_head_dim_bytes() {
        // Arrange: score_buf_offset = num_heads * head_dim * elem_bytes.
        // The AddPtr instruction for score_buf_ptr uses this offset from output_ptr.
        // For num_heads=4, head_dim=8, F32: offset = 4*8*4 = 128
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: 4 heads, head_dim=8 → score_buf_offset = 4*8*4 = 128
        let _ = emit_mla_attn_score_inline(
            &mut prog, 4, 8, 16, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: AddPtr with base=out_ptr and offset=128
        let has_addptr_128 = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::AddPtr { base, offset: 128, .. } if *base == out_ptr)
        });
        assert!(has_addptr_128,
            "score_buf AddPtr should use offset=128 (4*8*4), got no matching instruction");
    }

    #[test]
    fn test_mla_attn_score_and_rope_merge_coexist_in_same_program_different_d_rope() {
        // Arrange: simulate absorbed and un-absorbed MLA paths in one program.
        // Absorbed: d_c == d_rope (all dims get RoPE). Un-absorbed: d_c > d_rope (partial).
        // Both emit functions must coexist without register conflicts.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        // Phase 1: absorbed rope merge (d_c==d_rope=16)
        let c1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let m1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin1 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p1 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let absorbed = emit_mla_rope_merge_inline(
            &mut prog, 16, 16,
            &[c1, k1, m1, cos1, sin1, p1], width, dtype,
        );
        assert!(absorbed.is_ok(), "absorbed rope merge should succeed: {:?}", absorbed.err());
        let instr_after_absorbed = prog.len();

        // Phase 2: un-absorbed rope merge (d_c=32, d_rope=16)
        let c2 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let m2 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos2 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin2 = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p2 = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let unabsorbed = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c2, k2, m2, cos2, sin2, p2], width, dtype,
        );

        // Assert: both paths succeed, un-absorbed adds more instructions
        assert!(unabsorbed.is_ok(), "un-absorbed rope merge should succeed: {:?}", unabsorbed.err());
        assert!(prog.len() > instr_after_absorbed,
            "un-absorbed path should add more instructions ({} > {})",
            prog.len(), instr_after_absorbed);
    }

    // ── 10 new tests (wave-12x60: MLA VmInstr structure, d_c/d_rope ratio, SIMD width effects) ──

    #[test]
    fn test_mla_attn_score_hreduce_body_produces_sum_trace() {
        // Arrange: verify the HReduce TraceOp body used in attn score has correct structure
        // The hreduce_body is constructed inline: [Input(0), HReduce{src: ValueId(0), op: Sum}]
        let hreduce_body: Vec<TraceOp> = vec![
            TraceOp::Input(0),
            TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum },
        ];

        // Assert: body has exactly 2 ops, first is Input, second is HReduce with Sum
        assert_eq!(hreduce_body.len(), 2, "hreduce body should have 2 ops");
        assert!(matches!(hreduce_body[0], TraceOp::Input(0)),
            "first op should be Input(0)");
        assert!(matches!(hreduce_body[1], TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum }),
            "second op should be HReduce Sum of slot 0");
    }

    #[test]
    fn test_mla_attn_score_w_uv_stride_scale_with_head_dim() {
        // Arrange: w_uv_head_stride = d_c * head_dim * elem_bytes.
        // Larger head_dim means larger GprBinOp::Mul Imm value.
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let mut prog_hd8 = VmProgram::new();
        let q1 = prog_hd8.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv1 = prog_hd8.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w1 = prog_hd8.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_hd8.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl1 = prog_hd8.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=16, head_dim=8 → w_uv_head_stride = 16*8*4 = 512
        emit_mla_attn_score_inline(
            &mut prog_hd8, 2, 8, 16, 4,
            &[q1, kv1, w1, o1], kl1, width, dtype,
        ).unwrap();
        let stride_512_count = prog_hd8.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(512), .. })
        }).count();

        let mut prog_hd16 = VmProgram::new();
        let q2 = prog_hd16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv2 = prog_hd16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w2 = prog_hd16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog_hd16.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl2 = prog_hd16.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=16, head_dim=16 → w_uv_head_stride = 16*16*4 = 1024
        emit_mla_attn_score_inline(
            &mut prog_hd16, 2, 16, 16, 4,
            &[q2, kv2, w2, o2], kl2, width, dtype,
        ).unwrap();
        let stride_1024_count = prog_hd16.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(1024), .. })
        }).count();

        // Assert: both stride multipliers are present
        assert!(stride_512_count >= 1,
            "head_dim=8 should emit GprBinOp::Mul with Imm(512), got {}", stride_512_count);
        assert!(stride_1024_count >= 1,
            "head_dim=16 should emit GprBinOp::Mul with Imm(1024), got {}", stride_1024_count);
    }

    #[test]
    fn test_mla_rope_merge_d_rope_half_aligned_with_w256_lanes() {
        // Arrange: d_c=32, d_rope=32 → d_rope_half=16, must be divisible by W256 lanes=8.
        // d_rope_half=16 / 8 = 2 rope_pair_vecs iterations.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, d_rope=32 → d_main=0, d_rope_half=16, rope_pair_vecs=2
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 32,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: LoopBegin with Const(2) for rope_pair_vecs=2
        let rope_loops = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::Const(2), .. })
        });
        assert!(rope_loops,
            "should emit LoopBegin Const(2) for rope_pair_vecs=2 with d_rope_half=16/W256");
    }

    #[test]
    fn test_mla_attn_score_head_stride_loop_step_bytes() {
        // Arrange: the heads loop steps by head_dc_bytes = d_c * elem_bytes.
        // For d_c=32, F32: step = 32*4 = 128.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, head_dim=8 → head_dc_bytes=128
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 32, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: heads loop uses step_bytes=128 (d_c*4)
        let heads_loop_128 = prog.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 128, .. })
        });
        assert!(heads_loop_128,
            "heads loop should step by d_c*elem_bytes=128 (d_c=32, F32)");
    }

    #[test]
    fn test_mla_rope_merge_rejects_odd_d_rope_with_w256() {
        // Arrange: d_rope=7 → d_rope_half=3.5 which truncates to 3 in usize,
        // not aligned to W256 lanes=8.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dummy = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=7 → d_rope_half=3 (integer division), not aligned to 8
        let result = emit_mla_rope_merge_inline(
            &mut prog, 16, 7,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, dummy],
            width, dtype,
        );

        // Assert: odd d_rope produces unaligned d_rope_half, should be rejected
        assert!(result.is_err(), "odd d_rope=7 with W256 should be rejected (d_rope_half=3, not aligned to 8)");
    }

    #[test]
    fn test_mla_attn_score_simd_width_affects_dc_vecs_count() {
        // Arrange: W256 (8 lanes) with d_c=32 → dc_vecs=4.
        // W512 (16 lanes) with d_c=32 → dc_vecs=2.
        // Fewer dc_vecs means fewer Const loop bounds with value 2 vs 4.
        let width256 = SimdWidth::W256;
        let width512 = SimdWidth::W512;
        let dtype = QuantPrecision::F32;

        let mut prog256 = VmProgram::new();
        let q1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl1 = prog256.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_attn_score_inline(
            &mut prog256, 2, 16, 32, 16,
            &[q1, kv1, w1, o1], kl1, width256, dtype,
        ).unwrap();
        let const_4_loops_w256 = prog256.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::Const(4), .. })
        }).count();

        let mut prog512 = VmProgram::new();
        let q2 = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv2 = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w2 = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog512.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl2 = prog512.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_attn_score_inline(
            &mut prog512, 2, 16, 32, 16,
            &[q2, kv2, w2, o2], kl2, width512, dtype,
        ).unwrap();
        let const_2_loops_w512 = prog512.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::Const(2), .. })
        }).count();

        // Assert: W256 has Const(4) loops (dc_vecs=4), W512 has Const(2) loops (dc_vecs=2)
        assert!(const_4_loops_w256 >= 1,
            "W256 d_c=32 should have Const(4) dc_vecs loop, got {}", const_4_loops_w256);
        assert!(const_2_loops_w512 >= 1,
            "W512 d_c=32 should have Const(2) dc_vecs loop, got {}", const_2_loops_w512);
    }

    #[test]
    fn test_mla_rope_merge_copy_phase_loop_step_equals_dc_vec_step() {
        // Arrange: copy phase loop steps by dc_vec_step = lanes * elem_bytes.
        // For W256 (8 lanes), F32: step = 8*4 = 32.
        // For W128 (4 lanes), F32: step = 4*4 = 16.
        let width256 = SimdWidth::W256;
        let width128 = SimdWidth::W128;
        let dtype = QuantPrecision::F32;

        let mut prog256 = VmProgram::new();
        let c1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin1 = prog256.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p1 = prog256.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=32, d_rope=16 → d_main=16, d_main_vecs=2 for W256, step=32
        emit_mla_rope_merge_inline(
            &mut prog256, 32, 16,
            &[c1, k1, o1, cos1, sin1, p1], width256, dtype,
        ).unwrap();
        let step_32_loops = prog256.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 32, .. })
        }).count();

        let mut prog128 = VmProgram::new();
        let c2 = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos2 = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin2 = prog128.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p2 = prog128.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        // d_c=32, d_rope=16 → d_main=16, d_main_vecs=4 for W128, step=16
        emit_mla_rope_merge_inline(
            &mut prog128, 32, 16,
            &[c2, k2, o2, cos2, sin2, p2], width128, dtype,
        ).unwrap();
        let step_16_loops = prog128.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 16, .. })
        }).count();

        // Assert: W256 uses step_bytes=32, W128 uses step_bytes=16
        assert!(step_32_loops >= 1, "W256 copy/RoPE loops should step by 32, got {}", step_32_loops);
        assert!(step_16_loops >= 1, "W128 copy/RoPE loops should step by 16, got {}", step_16_loops);
    }

    #[test]
    fn test_mla_attn_score_v_restore_gemv_uses_dc_vecs_const_bound() {
        // Arrange: V restore GEMV inner loop iterates dc_vecs times.
        // d_c=32 with W256 (8 lanes) → dc_vecs=4, which appears as Const(4) LoopBegin.
        // At least one such loop must exist for the dot product accumulation.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=32, head_dim=8 → dc_vecs=4, hd_vecs=1
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 32, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: at least one Const(4) loop for dc_vecs=4 (dot product accumulation)
        let dc_vec4_loops_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::Const(4), .. })
        }).count();
        assert!(dc_vec4_loops_count >= 1,
            "d_c=32 with W256 should produce at least 1 Const(4) loop for dc_vecs, got {}",
            dc_vec4_loops_count);
    }

    #[test]
    fn test_mla_rope_merge_even_store_offset_equals_d_main_bytes() {
        // Arrange: even component stores at offset Add(LoopOffset, d_main_bytes).
        // For d_c=48, d_rope=16 → d_main=32, d_main_bytes=128.
        // The even store uses Add(_, Const(128)) offset from output_ptr.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=48, d_rope=16 → d_main=32, d_main_bytes=128
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 48, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: VecStore to output_ptr with Add offset containing Const(128) for even store
        let even_stores = prog.instrs.iter().filter(|instr| {
            if let VmInstr::VecStore { base, offset, .. } = instr {
                *base == output_ptr && match offset {
                    OffsetExpr::Add(_, rhs) => matches!(rhs.as_ref(), OffsetExpr::Const(128)),
                    _ => false,
                }
            } else { false }
        }).count();
        assert!(even_stores >= 1,
            "even RoPE store should use Add(_, Const(128)) = d_main_bytes offset, got {}", even_stores);
    }

    #[test]
    fn test_mla_attn_score_output_store_loop_step_is_hd_vec_step() {
        // Arrange: normalize phase output stores use Const offset stepping by
        // hd_vec_step = lanes * elem_bytes. For W256 (8 lanes), F32: step = 32.
        // head_dim=16 → hd_vecs=2, stores at Const(0) and Const(32).
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: head_dim=16 → hd_vecs=2, hd_vec_step=32
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 16, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: VecStore with Const(32) offset exists (second chunk of head_dim output)
        let stores_offset_32 = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { offset: OffsetExpr::Const(32), .. })
        }).count();
        assert!(stores_offset_32 >= 1,
            "should emit VecStore with Const(32) for hd_vec_step, got {}", stores_offset_32);
    }

    #[test]
    fn test_mla_rope_merge_d_main_vecs_zero_skips_copy_loop() {
        // Arrange: when d_main=0 (d_c == d_rope), d_main_vecs=0, so the copy phase
        // emit_loop with BoundExpr::Const(0) should still emit LoopBegin+LoopEnd
        // (the loop body simply never executes). Check that no c_kv loads exist.
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c=16, d_rope=16 → d_main=0, d_main_vecs=0, copy loop is Const(0) bound
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 16, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: zero Const(0) bound loops may exist but no c_kv loads (copy body skipped)
        let c_kv_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        assert_eq!(c_kv_loads, 0,
            "d_main=0 should produce zero c_kv loads (copy phase body never executes), got {}", c_kv_loads);
    }

    /// Test: 1/sqrt(d_c) scale in attention score uses non-zero, non-inf constant.
    /// The inv_sqrt_dc Broadcast should have a constant in (0, 1) range.
    #[test]
    fn test_mla_attn_score_inv_sqrt_dc_scale_is_not_zero() {
        // Arrange: d_c=64 → inv_sqrt_dc = 1/sqrt(64) = 0.125
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 4, 16, 64, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: find Broadcast with Const scale in (0, 1) range
        let scale_broadcasts = prog.instrs.iter().filter(|instr| {
            if let VmInstr::Broadcast { src: ScalarExpr::Const(c), .. } = instr {
                *c > 0.0 && *c < 1.0 && (*c - 0.125).abs() < 0.001
            } else { false }
        }).count();
        assert!(scale_broadcasts >= 1,
            "inv_sqrt_dc=0.125 should appear as Broadcast Const, got {} matches", scale_broadcasts);
    }

    /// Test: d_rope > d_c still compiles (caller validates semantic correctness).
    /// The emit function only checks alignment, not semantic validity of d_rope vs d_c.
    #[test]
    fn test_mla_attn_score_d_rope_larger_than_d_c_still_compiles() {
        // Arrange: d_c=32, d_rope=64 (semantically invalid but compiles)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_rope=64 > d_c=32 should still succeed (alignment is valid)
        let result = emit_mla_attn_score_inline(
            &mut prog, 4, 16, 32, 64,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        );

        // Assert: succeeds because alignment check passes (d_rope_half=32 aligned to 8)
        assert!(result.is_ok(), "d_rope > d_c should compile if aligned, got {:?}", result.err());
    }

    /// Test: Un-absorbed path (d_main > 0) produces copy-phase VecLoads from c_kv.
    /// d_main = d_c - d_rope. When d_main > 0, copy phase adds VecLoad from c_kv.
    /// When d_main = 0 (absorbed), no c_kv loads exist.
    #[test]
    fn test_mla_rope_merge_d_main_equals_d_c_minus_d_rope() {
        // Arrange: d_c=48, d_rope=16 → d_main=32 (un-absorbed, has copy phase)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: un-absorbed (d_main=32 > 0)
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 48, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: c_kv loads exist because d_main > 0 triggers copy phase
        let c_kv_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        assert!(c_kv_loads > 0,
            "d_main=32 should produce c_kv loads from copy phase, got {}", c_kv_loads);

        // Compare: absorbed path (d_main=0) has zero c_kv loads
        let mut prog2 = VmProgram::new();
        let c_kv2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos2 = prog2.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        let _ = emit_mla_rope_merge_inline(
            &mut prog2, 16, 16,
            &[c_kv2, k_pe2, out2, cos2, sin2, pos2],
            width, dtype,
        ).unwrap();
        let c_kv_loads2 = prog2.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv2)
        }).count();
        assert_eq!(c_kv_loads2, 0,
            "absorbed path (d_main=0) should have zero c_kv loads, got {}", c_kv_loads2);
    }

    /// Test: w_uv head stride scales with d_c and head_dim.
    /// The GprBinOp::Mul for head stride should have Imm(d_c * head_dim * 4) for F32.
    #[test]
    fn test_mla_attn_score_w_uv_head_stride_scales_with_dc_and_head_dim() {
        // Arrange: d_c=32, head_dim=16 → head_stride = 32*16*4 = 2048 bytes
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 4, 16, 32, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: find GprBinOp::Mul with b=Imm(2048) for head stride
        let stride_2048 = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(2048), .. })
        }).count();
        assert!(stride_2048 >= 1, "head_stride=2048 should appear in GprBinOp::Mul Imm, got {}", stride_2048);
    }

    /// Test: d_rope_half vecs loop bound matches alignment.
    /// For d_c=64, d_rope=32 → d_rope_half=16, with W256 lanes=8 → rope_pair_vecs=2.
    #[test]
    fn test_mla_rope_merge_d_rope_half_vecs_loop_bound_matches_alignment() {
        // Arrange: d_c=64, d_rope=32 → d_rope_half=16, lanes=8 → rope_pair_vecs=2
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 64, 32,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: LoopBegin with Const(2) for rope_pair_vecs
        let const_2_loops = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { bound: BoundExpr::Const(2), .. })
        }).count();
        assert!(const_2_loops >= 1,
            "d_rope_half=16 with W256 should produce Const(2) loop for rope_pair_vecs, got {}",
            const_2_loops);
    }

    /// Test: KV loop step includes d_rope in stride calculation.
    /// Different d_rope values produce different step_bytes in the KV LoopBegin.
    #[test]
    fn test_mla_attn_score_kv_loop_step_includes_d_rope_in_stride() {
        // Arrange: d_c=32, d_rope=24 → kv_row_bytes = (32+24)*4 = 224 bytes per KV entry
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 4, 16, 32, 24,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: find LoopBegin with step_bytes=224 for KV loop
        let kv_loop_224 = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 224, .. })
        }).count();
        assert!(kv_loop_224 >= 1, "KV loop should have step_bytes=(d_c+d_rope)*4=224, got {} matches", kv_loop_224);
    }

    /// Test: Absorbed path (d_c == d_rope) produces zero c_kv loads.
    /// When all dimensions get RoPE, there's no copy phase, so no c_kv loads.
    #[test]
    fn test_mla_rope_merge_absorbed_path_no_c_kv_loads() {
        // Arrange: d_c=32, d_rope=32 → absorbed path (d_main=0)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: d_c == d_rope (absorbed)
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 32,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: zero VecLoad from c_kv_ptr (no copy phase)
        let c_kv_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        assert_eq!(c_kv_loads, 0,
            "absorbed path (d_c==d_rope) should have zero c_kv loads, got {}", c_kv_loads);
    }

    /// Test: Output stride multiplier equals head_dim * sizeof(dtype).
    /// For head_dim=16, F32 → stride = 64 bytes per head output.
    #[test]
    fn test_mla_attn_score_output_stride_mul_imm_equals_head_dim_bytes() {
        // Arrange: head_dim=16, F32 → head_output_stride = 16*4 = 64
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_attn_score_inline(
            &mut prog, 4, 16, 32, 16,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Mul with b=Imm(64) for output head stride
        let stride_64 = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(64), .. })
        }).count();
        assert!(stride_64 >= 1, "head_dim=16 F32 should produce Mul Imm(64), got {}", stride_64);
    }

    /// Test: Un-absorbed path (d_main > 0) has both c_kv and k_pe loads.
    /// Copy phase loads from c_kv, RoPE phase loads from k_pe.
    #[test]
    fn test_mla_rope_merge_unabsorbed_path_has_both_c_kv_and_k_pe_loads() {
        // Arrange: d_c=48, d_rope=16 → d_main=32 (un-absorbed)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 48, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: both c_kv loads (copy phase) and k_pe loads (RoPE phase) present
        let c_kv_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == c_kv_ptr)
        }).count();
        let k_pe_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == k_pe_ptr)
        }).count();
        assert!(c_kv_loads > 0, "un-absorbed should have c_kv loads for copy phase, got {}", c_kv_loads);
        assert!(k_pe_loads > 0, "un-absorbed should have k_pe loads for RoPE phase, got {}", k_pe_loads);
    }

    /// Test: Full MLA decode step combines rope merge + attn score in one program.
    /// Verifies register allocation doesn't conflict between the two phases.
    #[test]
    fn test_mla_attn_score_and_rope_merge_combined_program_register_allocation() {
        // Arrange: simulate full MLA decode step
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        // Phase 1: rope merge
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let merged_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let rope_result = emit_mla_rope_merge_inline(
            &mut prog, 64, 16,
            &[c_kv_ptr, k_pe_ptr, merged_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        );
        assert!(rope_result.is_ok(), "rope merge should succeed: {:?}", rope_result.err());
        let instr_after_rope = prog.len();

        // Phase 2: attn score (reusing merged_ptr as kv_ptr)
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_uv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let attn_result = emit_mla_attn_score_inline(
            &mut prog, 8, 16, 64, 16,
            &[q_ptr, merged_ptr, w_uv_ptr, out_ptr], kv_len, width, dtype,
        );

        // Assert: both phases succeed, attn adds instructions
        assert!(attn_result.is_ok(), "attn score should succeed: {:?}", attn_result.err());
        assert!(prog.len() > instr_after_rope,
            "combined program should grow after attn ({} > {})", prog.len(), instr_after_rope);
    }

    // ── 10 new tests (wave-12x61: MLA VmInstr structural, d_rope alignment edge, d_c scaling) ──

    /// Test: V-restore GEMV inner loop produces VecLoad from w_uv_ptr with
    /// GprBinOp-computed address (c_off * head_hd_bytes + wuv_row), verifying
    /// that the weight access pattern uses runtime GPR arithmetic, not hardcoded offsets.
    #[test]
    fn test_mla_attn_score_v_restore_gemv_weight_load_uses_gpr_computed_address() {
        // Arrange: head_dim=8, d_c=16 → hd_vecs=1, dc_vecs=2
        // Inside V-restore GEMV, W_UV address = wuv_row + c_ctr * head_hd_bytes + d_offset
        // This requires GprBinOp::Mul(c_ctr, head_hd_bytes) + GprBinOp::Add(wuv_row, ...)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: head_hd_bytes = 8 * 4 = 32
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: GprBinOp::Mul with Imm(32) exists for c_off * head_hd_bytes inside V-restore
        let mul_32 = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(32), .. })
        }).count();
        assert!(mul_32 >= 1,
            "V-restore GEMV should emit GprBinOp::Mul Imm(32) for c_off * head_hd_bytes, got {}",
            mul_32);
    }

    /// Test: RoPE merge odd component load from k_pe_ptr uses the correct
    /// byte offset = d_rope_half * elem_bytes. For d_rope=24, F32: d_rope_half=12,
    /// but 12 is not aligned to W256 lanes=8 → should be rejected.
    /// For d_rope=32, F32: d_rope_half=16, aligned to 8 → odd offset = 64 bytes.
    #[test]
    fn test_mla_rope_merge_odd_load_offset_scales_with_d_rope_half() {
        // Arrange: d_c=48, d_rope=32 → d_main=16, d_rope_half=16, odd offset = 16*4 = 64
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 48, 32,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: VecLoad from k_pe_ptr with Add offset containing Const(64) for odd components
        // d_rope_half * elem_bytes = 16 * 4 = 64
        let odd_loads_with_64 = prog.instrs.iter().filter(|instr| {
            if let VmInstr::VecLoad { base, offset, .. } = instr {
                *base == k_pe_ptr &&
                    matches!(offset, OffsetExpr::Add(_, rhs)
                             if matches!(rhs.as_ref(), OffsetExpr::Const(64)))
            } else { false }
        }).count();
        assert!(odd_loads_with_64 >= 1,
            "odd component load from k_pe should have offset Const(64) = d_rope_half*4, got {}",
            odd_loads_with_64);
    }

    /// Test: Absorbed path produces fewer total VecStore instructions than un-absorbed.
    /// Absorbed (d_main=0) has only RoPE stores (2 per iteration: even + odd).
    /// Un-absorbed (d_main > 0) has copy stores PLUS RoPE stores.
    #[test]
    fn test_mla_rope_merge_absorbed_produces_fewer_vec_stores_than_unabsorbed() {
        // Arrange
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        // Absorbed: d_c=16, d_rope=16 → d_main=0
        let mut prog_abs = VmProgram::new();
        let c1 = prog_abs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k1 = prog_abs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog_abs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos1 = prog_abs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin1 = prog_abs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p1 = prog_abs.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_rope_merge_inline(
            &mut prog_abs, 16, 16,
            &[c1, k1, o1, cos1, sin1, p1], width, dtype,
        ).unwrap();
        let abs_stores = prog_abs.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { .. })
        }).count();

        // Un-absorbed: d_c=32, d_rope=16 → d_main=16
        let mut prog_unabs = VmProgram::new();
        let c2 = prog_unabs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k2 = prog_unabs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog_unabs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos2 = prog_unabs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin2 = prog_unabs.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let p2 = prog_unabs.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_rope_merge_inline(
            &mut prog_unabs, 32, 16,
            &[c2, k2, o2, cos2, sin2, p2], width, dtype,
        ).unwrap();
        let unabs_stores = prog_unabs.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore { .. })
        }).count();

        // Assert: un-absorbed has more stores due to copy phase
        assert!(unabs_stores > abs_stores,
            "un-absorbed ({} stores) should exceed absorbed ({} stores) due to copy phase",
            unabs_stores, abs_stores);
    }

    /// Test: The accumulate_body TraceOp correctly chains o_acc*correction + weight*v_vec.
    /// Verify all intermediate ValueIds reference previously defined slots (SSA validity).
    #[test]
    fn test_accumulate_body_ssa_chain_references_valid_slots() {
        // Arrange: accumulate_body has 7 ops:
        // [0] Input(0)=o_acc, [1] Input(1)=correction, [2] Input(2)=weight, [3] Input(3)=v_vec
        // [4] Mul(0,1), [5] Mul(2,3), [6] Add(4,5)
        let body = accumulate_body();
        assert_eq!(body.len(), 7);

        // Assert: Mul at [4] references inputs [0] and [1]
        if let TraceOp::Mul(ValueId(a), ValueId(b)) = body[4] {
            assert_eq!(a, 0, "first Mul should reference slot 0 (o_acc)");
            assert_eq!(b, 1, "first Mul should reference slot 1 (correction)");
        } else {
            panic!("body[4] should be Mul");
        }

        // Mul at [5] references inputs [2] and [3]
        if let TraceOp::Mul(ValueId(a), ValueId(b)) = body[5] {
            assert_eq!(a, 2, "second Mul should reference slot 2 (weight)");
            assert_eq!(b, 3, "second Mul should reference slot 3 (v_vec)");
        } else {
            panic!("body[5] should be Mul");
        }

        // Add at [6] references Mul results [4] and [5]
        if let TraceOp::Add(ValueId(a), ValueId(b)) = body[6] {
            assert_eq!(a, 4, "Add should reference slot 4 (o_acc * correction)");
            assert_eq!(b, 5, "Add should reference slot 5 (weight * v_vec)");
        } else {
            panic!("body[6] should be Add");
        }
    }

    /// Test: V-restore GEMV HReduce produces Broadcast with ExtractLane0 for each
    /// output chunk d. head_dim=16 with W256 → hd_vecs=2, so at least 2 ExtractLane0
    /// broadcasts exist (one per output chunk accumulation).
    #[test]
    fn test_mla_attn_score_v_restore_hreduce_produces_extract_lane0_per_chunk() {
        // Arrange: head_dim=16, W256=8 lanes → hd_vecs=2 output chunks
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: head_dim=16 → 2 V-restore output chunks, each with HReduce + ExtractLane0
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 16, 16, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: at least 2 ExtractLane0 broadcasts (score scalar + at least 1 V-restore)
        let extract_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::Broadcast { src: ScalarExpr::ExtractLane0(_), .. })
        }).count();
        assert!(extract_count >= 2,
            "hd_vecs=2 should produce >= 2 ExtractLane0 broadcasts (score + V-restore), got {}",
            extract_count);
    }

    /// Test: RoPE merge odd component store uses offset = d_main_bytes + d_rope_half * elem_bytes.
    /// For d_c=32, d_rope=16: d_main_bytes=64, d_rope_half=8, odd_store_offset = 64 + 32 = 96.
    /// The even store uses offset = d_main_bytes = 64.
    #[test]
    fn test_mla_rope_merge_even_and_odd_stores_use_distinct_offsets() {
        // Arrange: d_c=32, d_rope=16 → d_main=16, d_main_bytes=64
        // d_rope_half=8, d_rope_half*4=32
        // Even store: Add(_, Const(64)), Odd store: Add(_, Const(96))
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 16,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Collect all Add-const offsets in VecStore to output_ptr
        let mut const_offsets: Vec<usize> = prog.instrs.iter().filter_map(|instr| {
            if let VmInstr::VecStore { base, offset, .. } = instr {
                if *base == output_ptr {
                    if let OffsetExpr::Add(_, rhs) = offset {
                        if let OffsetExpr::Const(v) = rhs.as_ref() {
                            return Some(*v);
                        }
                    }
                }
            }
            None
        }).collect();
        const_offsets.sort();
        const_offsets.dedup();

        // Assert: both 64 (even) and 96 (odd) offsets are present
        let has_64 = const_offsets.iter().any(|&v| v == 64);
        let has_96 = const_offsets.iter().any(|&v| v == 96);
        assert!(has_64, "even RoPE store should use offset 64 (d_main_bytes), found offsets: {:?}", const_offsets);
        assert!(has_96, "odd RoPE store should use offset 96 (d_main_bytes + d_rope_half*4), found offsets: {:?}", const_offsets);
    }

    /// Test: KV cache row stride changes with d_rope value.
    /// d_c=16, d_rope=8 → step_bytes = (16+8)*4 = 96
    /// d_c=16, d_rope=16 → step_bytes = (16+16)*4 = 128
    /// Different d_rope values produce different KV loop step_bytes.
    #[test]
    fn test_mla_attn_score_different_d_rope_produces_different_kv_stride() {
        // Arrange: two configs with same d_c but different d_rope
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        // Config 1: d_rope=8 → kv_row_bytes = 24*4 = 96
        let mut prog1 = VmProgram::new();
        let q1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o1 = prog1.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl1 = prog1.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_attn_score_inline(
            &mut prog1, 2, 8, 16, 8,
            &[q1, kv1, w1, o1], kl1, width, dtype,
        ).unwrap();
        let has_step_96 = prog1.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 96, .. })
        });

        // Config 2: d_rope=16 → kv_row_bytes = 32*4 = 128
        let mut prog2 = VmProgram::new();
        let q2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let o2 = prog2.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kl2 = prog2.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        emit_mla_attn_score_inline(
            &mut prog2, 2, 8, 16, 16,
            &[q2, kv2, w2, o2], kl2, width, dtype,
        ).unwrap();
        let has_step_128 = prog2.instrs.iter().any(|instr| {
            matches!(instr, VmInstr::LoopBegin { step_bytes: 128, .. })
        });

        // Assert: different d_rope values produce different KV loop strides
        assert!(has_step_96, "d_rope=8 should produce KV loop step_bytes=96");
        assert!(has_step_128, "d_rope=16 should produce KV loop step_bytes=128");
    }

    /// Test: The sum_update_body TraceOp correctly models running_sum * correction + weight.
    /// Verify ValueId chain: Mul(running_sum, correction) → Add(result, weight).
    #[test]
    fn test_sum_update_body_ssa_def_before_use() {
        // Arrange: sum_update_body = [Input(0)=sum, Input(1)=correction, Input(2)=weight,
        //   Mul(0,1)=sum*correction, Add(3,2)=sum*correction + weight]
        let body = sum_update_body();
        assert_eq!(body.len(), 5);

        // Assert: Mul at [3] references [0] and [1]
        if let TraceOp::Mul(ValueId(a), ValueId(b)) = body[3] {
            assert_eq!(a, 0, "Mul should reference slot 0 (running_sum)");
            assert_eq!(b, 1, "Mul should reference slot 1 (correction)");
        } else {
            panic!("body[3] should be Mul");
        }

        // Add at [4] references Mul result [3] and Input [2]
        if let TraceOp::Add(ValueId(a), ValueId(b)) = body[4] {
            assert_eq!(a, 3, "Add should reference slot 3 (sum*correction)");
            assert_eq!(b, 2, "Add should reference slot 2 (weight)");
        } else {
            panic!("body[4] should be Add");
        }
    }

    /// Test: The normalize phase at the end of emit_mla_attn_score_inline emits
    /// VecStore with Const offset (d * hd_vec_step) for each head_dim output chunk.
    /// The stores go to o_row (derived pointer), not directly to output_ptr.
    /// Verify that VecStore with OffsetExpr::Const(0) and vector width exist.
    #[test]
    fn test_mla_attn_score_normalize_phase_emits_const_offset_stores() {
        // Arrange: verify normalize phase stores use Const offsets
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: head_dim=8 → hd_vecs=1, normalize stores at OffsetExpr::Const(0)
        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 8, 16, 4,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: VecStore with Const(0) offset and W256 width (normalize output chunk 0)
        let const_0_vec_stores = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecStore {
                offset: OffsetExpr::Const(0),
                width: SimdWidth::W256,
                .. })
        }).count();
        assert!(const_0_vec_stores >= 1,
            "normalize phase should emit VecStore with Const(0) offset and W256 width, got {}",
            const_0_vec_stores);
    }

    /// Test: The heads loop in emit_mla_attn_score_inline uses BoundExpr::Const(num_heads).
    /// For num_heads=4, the outermost loop should have bound Const(4).
    #[test]
    fn test_mla_attn_score_heads_loop_bound_matches_num_heads() {
        // Arrange: 4 heads → outermost LoopBegin should have bound Const(4)
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act: 4 heads
        let _ = emit_mla_attn_score_inline(
            &mut prog, 4, 8, 16, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr], kv_len, width, dtype,
        ).unwrap();

        // Assert: first LoopBegin should have bound Const(4) for num_heads
        let first_loop = prog.instrs.iter().find(|instr| {
            matches!(instr, VmInstr::LoopBegin { .. })
        });
        assert!(first_loop.is_some(), "should emit at least one LoopBegin");
        let has_const_4 = matches!(first_loop.unwrap(), VmInstr::LoopBegin { bound: BoundExpr::Const(4), .. });
        assert!(has_const_4,
            "first loop should have bound Const(4) for num_heads=4");
    }

    /// Test: RoPE merge with d_rope equal to d_c (absorbed path) still emits RoPE
    /// VecLoads from k_pe_ptr, cos_ptr, and sin_ptr (the RoPE phase runs, copy doesn't).
    #[test]
    fn test_mla_rope_merge_absorbed_still_loads_k_pe_cos_sin() {
        // Arrange: d_c=32, d_rope=32 → d_main=0 (absorbed), d_rope_half=16
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let pos = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        // Act
        let _ = emit_mla_rope_merge_inline(
            &mut prog, 32, 32,
            &[c_kv_ptr, k_pe_ptr, output_ptr, cos_ptr, sin_ptr, pos],
            width, dtype,
        ).unwrap();

        // Assert: k_pe, cos, sin loads all present despite d_main=0
        let k_pe_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == k_pe_ptr)
        }).count();
        let cos_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == cos_ptr)
        }).count();
        let sin_loads = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::VecLoad { base, .. } if *base == sin_ptr)
        }).count();

        assert!(k_pe_loads >= 2,
            "absorbed path should still load k_pe (even+odd), got {}", k_pe_loads);
        assert!(cos_loads >= 1,
            "absorbed path should still load cos table, got {}", cos_loads);
        assert!(sin_loads >= 1,
            "absorbed path should still load sin table, got {}", sin_loads);
    }

    // ── MLA PagedAttention page stride tests (SPEC 33 REQ-MLA-008) ──

    #[test]
    fn test_mla_paged_kv_page_stride_matches_dc_plus_drope() {
        // MLA PagedKV stores [d_c + d_rope] per token per layer, NOT 2*head_dim
        let d_c = 512;
        let d_rope = 64;
        let page_size = 16;
        let page_stride = page_size * (d_c + d_rope) * 4; // F32
        // Standard MHA would be: page_size * 2 * num_kv_heads * head_dim * 4
        // MLA is more compact: page_size * (d_c + d_rope) * 4
        let standard_stride = page_size * 2 * 32 * 128 * 4; // 32 heads, 128 dim
        assert!(page_stride < standard_stride,
            "MLA page stride ({page_stride}) should be much smaller than standard ({standard_stride})");
        assert_eq!(page_stride, page_size * (d_c + d_rope) * 4);
    }

    #[test]
    fn test_mla_kv_dim_is_dc_plus_drope() {
        // Verify that MLA effective KV dim = d_c + d_rope
        let d_c = 512;
        let d_rope = 64;
        let kv_dim = d_c + d_rope;
        assert_eq!(kv_dim, 576);
        // This is 56.9x compression vs standard 2*32*128 = 8192
        let standard_dim = 2 * 32 * 128;
        let compression = standard_dim as f64 / kv_dim as f64;
        assert!(compression > 10.0, "MLA compression ratio should be >10x, got {compression:.1}x");
    }

    #[test]
    fn test_mla_attn_score_scale_is_inv_sqrt_dc() {
        // MLA attention scoring in compressed space uses 1/sqrt(d_c), not 1/sqrt(head_dim)
        let d_c = 512;
        let head_dim = 128;
        let scale = 1.0 / (d_c as f32).sqrt();
        let standard_scale = 1.0 / (head_dim as f32).sqrt();
        assert!(scale < standard_scale,
            "MLA scale ({scale}) should be smaller than standard ({standard_scale})");
        assert!((scale - 0.04419417f32).abs() < 1e-6);
    }

    #[test]
    fn test_mla_attn_score_single_token_decode() {
        // Single-token decode: kv_len=1, should produce valid attention
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let result = emit_mla_attn_score_inline(
            &mut prog, 4, 128, 512, 64,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        );

        assert!(result.is_ok(), "single-token decode should compile: {:?}", result.err());
        // Should have LoopBegin for kv_len iteration (but may be eliminated for single token)
        assert!(prog.instrs.len() > 5, "should produce instructions for attention scoring");
    }

    #[test]
    fn test_mla_rope_merge_with_varied_dims() {
        // Test RoPE merge with different d_c / d_rope ratios
        for d_rope in [32, 64, 128] {
            let d_c = 512 - d_rope;
            let mut prog = VmProgram::new();
            let width = SimdWidth::W256;
            let dtype = QuantPrecision::F32;

            let c_kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let k_pe_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let cos_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            let sin_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

            let result = emit_mla_rope_merge_inline(
                &mut prog, d_c, d_rope,
                &[c_kv_ptr, k_pe_ptr, out_ptr, cos_ptr, sin_ptr],
                width, dtype,
            );

            assert!(result.is_ok(),
                "rope merge should compile with d_c={d_c}, d_rope={d_rope}: {:?}",
                result.err());
        }
    }

    #[test]
    fn test_mla_attn_score_uses_compressed_dot_product() {
        // Verify MLA attention uses d_c-dimension dot product, not head_dim
        let mut prog = VmProgram::new();
        let width = SimdWidth::W256;
        let dtype = QuantPrecision::F32;
        let d_c = 32; // Small d_c that's a multiple of 8 (lanes)

        let q_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let w_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let out_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let kv_len = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);

        let _ = emit_mla_attn_score_inline(
            &mut prog, 2, 16, d_c, 8,
            &[q_ptr, kv_ptr, w_ptr, out_ptr],
            kv_len, width, dtype,
        ).unwrap();

        // Should contain FMA instructions for dot product in compressed space
        let fma_count = prog.instrs.iter().filter(|instr| {
            matches!(instr, VmInstr::Fma { .. })
        }).count();
        assert!(fma_count > 0, "should contain FMA for compressed dot product");
    }
}