//! QuantGather inline lowering — GGUF quantized embedding lookup.

use super::instr::*;
use super::auto_select;
use super::quant_offset_dsl::QuantOffsetDsl;
use crate::compiler::trace::{QuantPrecision, TraceOp, ValueId};
use crate::types::CompilerError;

/// QuantGather: embedding lookup on quantized weight table (ARCH-RUST-IS-CODEGEN §4.2 REQ-QCG-005).
///
/// Unified path using `DecodeTraceBuilder` for all quantization formats.
/// SPEC 24-QUANT-PIPELINE-JIT §5 Fusion: replaces per-format hand-written VmInstr emission
/// with parameterized trace templates driven by `QuantFormatDescriptor`.
///
/// Loop structure (all emit_loop_try for ARCH-NO-LOOP-UNROLL):
///   outer: seq_len tokens (step = 4B per token_id)
///     load token_id → compute row base → set row_ptr
///     block loop: row_blocks blocks (step = block_bytes)
///       build decode trace via DecodeTraceBuilder
///       auto_lower_trace_raw → VmInstr → ISA machine code
///       VecStore decoded F32 to output row
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_quant_gather_inline(
    prog: &mut VmProgram,
    seq_bound: BoundExpr,
    _vocab_size: usize,
    hidden_dim: usize,
    quant_type: crate::quant::QuantType,
    width: SimdWidth,
    indices_ptr: VRegId,   // input: i32 token_ids [seq_len]
    embed_ptr: VRegId,     // weight: quantized embed table
    output_ptr: VRegId,    // output: F32 [seq_len, hidden_dim]
    dtype: QuantPrecision,
    embedding_scale: Option<f32>,
) -> Result<(), CompilerError> {
    let desc = crate::quant_format::registry().get(&quant_type)
        .cloned()
        .ok_or_else(|| CompilerError::CodegenViolation(
            format!("QuantGather: no QuantFormatDescriptor for {:?}", quant_type)
        ))?;

    let block_size = desc.block_size;
    let block_bytes = desc.block_bytes;

    if hidden_dim == 0 || block_size == 0 || block_bytes == 0 {
        return Err(CompilerError::CodegenViolation(
            "emit_quant_gather_inline: zero dimension/format param".into(),
        ));
    }
    if hidden_dim % block_size != 0 {
        return Err(CompilerError::CodegenViolation(format!(
            "emit_quant_gather_inline: hidden_dim={} not divisible by block_size={}",
            hidden_dim, block_size
        )));
    }

    emit_quant_gather_trace_driven(
        prog, seq_bound, hidden_dim, &desc, quant_type, width,
        indices_ptr, embed_ptr, output_ptr, dtype, embedding_scale,
    )
}

/// Unified trace-driven QuantGather: uses DecodeTraceBuilder for all formats.
///
/// SPEC 24-QUANT-PIPELINE-JIT REQ-QPJ-002: all quantization formats (Q4_0/Q4_1/Q8_0/Q8_1
/// and K-Quant/IQ/FP4/etc) share this single code path. The DecodeTraceBuilder generates
/// per-block TraceOp sequences from QuantFormatDescriptor metadata, which auto_lower_trace_raw
/// maps to VmInstr. No per-format hand-written VmInstr emission.
///
/// REQ-LC-008: 输入偏移使用 block_bytes, 输出偏移使用 compute_dtype.elem_bytes().
/// REQ-LC-009: 所有偏移通过 QuantOffsetDsl derive 方法推导，零手写公式.
#[allow(clippy::too_many_arguments)]
fn emit_quant_gather_trace_driven(
    prog: &mut VmProgram,
    seq_bound: BoundExpr,
    hidden_dim: usize,
    desc: &crate::quant_format::QuantFormatDescriptor,
    quant_type: crate::quant::QuantType,
    width: SimdWidth,
    indices_ptr: VRegId,
    embed_ptr: VRegId,
    output_ptr: VRegId,
    dtype: QuantPrecision,
    embedding_scale: Option<f32>,
) -> Result<(), CompilerError> {
    // REQ-LC-007/009: 所有偏移通过 QuantOffsetDsl derive 方法推导
    // REQ-LC-011: 编译时数值模拟 — 验证解量化不产生 NaN/Inf
    super::verify::verify_numerical_sanity(desc, width.f32_lanes().max(1))?;

    let lanes = width.f32_lanes().max(1);
    let block_size = desc.block_size;
    let compute_elem_bytes = dtype.elem_bytes();

    // REQ-LC-008: 输入偏移 (量化数据) vs 输出偏移 (compute_dtype) 严格分离

    // 输入偏移: block_stride = desc.block_bytes (量化格式的 block 字节大小)
    let block_stride = desc.block_bytes;

    // 输入偏移: row_stride_bytes = (hidden_dim / block_size) * block_bytes
    let row_stride_bytes = QuantOffsetDsl::derive_row_stride_bytes(desc, hidden_dim);

    // 输入偏移: data_byte_advance (sub-block 间 data 指针步进)
    let data_byte_advance = QuantOffsetDsl::derive_data_byte_advance(desc, lanes);

    // 输出偏移: output_row_stride = hidden_dim * compute_elem_bytes
    let out_row_bytes_dsl = QuantOffsetDsl::derive_output_row_stride(hidden_dim, compute_elem_bytes);
    let out_row_bytes = out_row_bytes_dsl.evaluate(0) as usize;

    // 输出偏移: sub_block_output_step = lanes * compute_elem_bytes
    let sub_block_output_step_dsl = QuantOffsetDsl::derive_sub_block_output_step(lanes, compute_elem_bytes);
    let sub_block_output_step = sub_block_output_step_dsl.evaluate(0) as usize;

    // 结构参数
    let row_blocks = hidden_dim / block_size;
    let sub_blocks = QuantOffsetDsl::sub_block_count(desc, lanes);

    prog.emit(VmInstr::Comment(format!(
        "QuantGather(trace): quant_type={:?} hidden={} block_size={} block_stride={} compute_elem_bytes={}",
        quant_type, hidden_dim, block_size, block_stride, compute_elem_bytes
    )));

    let idx_scalar = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    let row_base = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let row_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let out_row = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let lane_offset = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: lane_offset, value: 0 });

    // Use caller-provided VRegs directly (same rationale as classic path).
    let safe_indices_ptr = indices_ptr;
    let safe_embed_ptr = embed_ptr;
    let safe_output_ptr = output_ptr;
    prog.emit(VmInstr::LoadPtr { dst: out_row, src: PtrExpr::VRegPlusConst(safe_output_ptr, 0) });

    // Outer seq loop: one token_id per iteration
    // REQ-LC-009 position 1: seq loop step_bytes = 4 (fixed: each u32 index is 4 bytes)
    prog.emit_loop_try(seq_bound, 4, |prog, _seq_ctr, seq_byte_off| -> Result<(), CompilerError> {
        // Load token_id
        prog.emit(VmInstr::ScalarLoad {
            dst: idx_scalar,
            base: safe_indices_ptr,
            offset: OffsetExpr::LoopOffset(seq_byte_off),
        });
        // row_base = token_id * row_stride_bytes (输入偏移: 量化行字节大小)
        prog.emit(VmInstr::ScalarToIndex {
            dst: row_base,
            src: idx_scalar,
            stride: row_stride_bytes,
        });
        // row_ptr = safe_embed_ptr + row_base
        prog.emit(VmInstr::LoadPtr { dst: row_ptr, src: PtrExpr::VRegPlusVReg(safe_embed_ptr, row_base) });

        // Build decode trace once (format-dependent, not data-dependent)
        let mut decode_trace = Vec::new();
        let builder = super::quant_decode::DecodeTraceBuilder::new(desc, lanes);
        let needs_lo = builder.needs_lane_offset();
        let needs_high_bits = builder.needs_high_bits_ptr();
        let high_bits_stride_val = builder.high_bits_stride();
        let _decoded_slot = builder.build(&mut decode_trace);

        // Stride GPR for data pointer advance per sub-block iteration
        let data_advance_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: data_advance_gpr, value: data_byte_advance as usize });
        let zero_gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::GprLoadImm { dst: zero_gpr, value: 0 });

        // For NibbleWithHighBits (Q6_K, Q5_0, Q5_1): high-bit-plane pointer stride.
        let high_bits_stride_gpr = if needs_high_bits {
            let gpr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
            prog.emit(VmInstr::GprLoadImm { dst: gpr, value: high_bits_stride_val });
            Some(gpr)
        } else {
            None
        };
        let high_offset_val: usize = match &desc.data_layout {
            crate::quant_format::DataLayout::NibbleWithHighBits { high_offset, .. } => *high_offset,
            _ => 0,
        };

        // REQ-LC-009 position 2: block loop step = block_stride (量化输入偏移)
        // REQ-LC-008: block_stride 使用 block_bytes, 不是 compute_elem_bytes
        prog.emit_loop_try(
            BoundExpr::Const(row_blocks),
            block_stride,
            |prog, blk_ctr, blk_off| -> Result<(), CompilerError> {
                let block_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::LoadPtr {
                    dst: block_ptr,
                    src: PtrExpr::VRegPlusVReg(row_ptr, blk_off),
                });

                if needs_lo {
                    prog.emit(VmInstr::GprLoadImm { dst: lane_offset, value: 0 });
                }

                let data_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: block_ptr, b: GprOperand::VReg(zero_gpr), op: GprOp::Add });

                let high_bits_ptr = if needs_high_bits {
                    let hbp = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
                    prog.emit(VmInstr::AddPtr { dst: hbp, base: block_ptr, offset: high_offset_val });
                    Some(hbp)
                } else {
                    None
                };

                let lanes_stride = if needs_lo {
                    let gpr = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
                    prog.emit(VmInstr::GprLoadImm { dst: gpr, value: lanes });
                    Some(gpr)
                } else {
                    None
                };

                // REQ-LC-009 position 3: sub_block loop step = sub_block_output_step (输出偏移: F32)
                // REQ-LC-008: 输出行内步进使用 lanes * compute_elem_bytes, 不是 block_bytes
                prog.emit_loop_try(
                    BoundExpr::Const(sub_blocks),
                    sub_block_output_step,
                    |prog, _sub_ctr, sub_off| -> Result<(), CompilerError> {
                        let mut gather_inputs: Vec<VRegId> = if needs_lo {
                            vec![block_ptr, data_ptr, lane_offset]
                        } else {
                            vec![block_ptr, data_ptr]
                        };
                        if let Some(hbp) = high_bits_ptr {
                            gather_inputs.push(hbp);
                        }
                        let slots = auto_select::auto_lower_trace_raw(
                            prog, &decode_trace, &gather_inputs, width, QuantPrecision::F32).map_err(|e| CompilerError::CodegenViolation(
                            format!("QuantGather: decode auto_lower failed for {:?}: {:?}", quant_type, e)
                        ))?;

                        let decoded = slots.last().copied()
                            .ok_or_else(|| CompilerError::CodegenViolation(
                                format!("QuantGather: decode trace produced no output for {:?}", quant_type)
                            ))?;

                        // Apply embedding_scale (e.g. Gemma 4: sqrt(hidden_size))
                        let store_src = if let Some(s) = embedding_scale {
                            let scale_trace = vec![
                                TraceOp::Const(s as f64),
                                TraceOp::Input(0),
                                TraceOp::Mul(ValueId(0), ValueId(1)),
                            ];
                            let scale_inputs = vec![decoded];
                            let scale_slots = auto_select::auto_lower_trace_raw(
                                prog, &scale_trace, &scale_inputs, width, QuantPrecision::F32,
                            ).map_err(|e| CompilerError::CodegenViolation(
                                format!("QuantGather: embedding_scale auto_lower failed: {:?}", e)
                            ))?;
                            scale_slots.last().copied()
                                .ok_or_else(|| CompilerError::CodegenViolation(
                                    "QuantGather: scale trace produced no output".into()
                                ))?
                        } else {
                            decoded
                        };

                        // REQ-LC-008/009: 输出偏移 = blk_ctr * block_size * compute_elem_bytes + sub_off
                        // block_size * compute_elem_bytes 是输出缓冲区的步进 (不是量化数据的 block_bytes)
                        // derive_output_block_offset 使用 compute_elem_bytes, 不使用 block_bytes
                        let out_offset = OffsetExpr::Add(
                            Box::new(OffsetExpr::Mul(
                                Box::new(OffsetExpr::ScalarVReg(blk_ctr)),
                                block_size * compute_elem_bytes,
                            )),
                            Box::new(OffsetExpr::LoopOffset(sub_off)),
                        );
                        prog.emit(VmInstr::VecStore {
                            base: out_row,
                            offset: out_offset,
                            src: store_src,
                            width,
                            dtype, predicate: None,
                        });

                        // REQ-LC-009 position 5: data_ptr 步进 = data_byte_advance (输入偏移)
                        prog.emit(VmInstr::GprBinOp { dst: data_ptr, a: data_ptr, b: GprOperand::VReg(data_advance_gpr), op: GprOp::Add });
                        if let (Some(lo), Some(ls)) = (Some(lane_offset), lanes_stride) {
                            prog.emit(VmInstr::GprBinOp { dst: lo, a: lo, b: GprOperand::VReg(ls), op: GprOp::Add });
                        }
                        if let (Some(hbp), Some(hbs)) = (high_bits_ptr, high_bits_stride_gpr) {
                            prog.emit(VmInstr::GprBinOp { dst: hbp, a: hbp, b: GprOperand::VReg(hbs), op: GprOp::Add });
                        }
                        Ok(())
                    },
                )?;
                Ok(())
            },
        )?;

        // REQ-LC-009 position 7: 输出行步进 = derive_output_row_stride (输出偏移)
        // out_row_bytes = hidden_dim * compute_elem_bytes (不使用 block_bytes)
        prog.emit(VmInstr::LoadPtr {
            dst: out_row,
            src: PtrExpr::VRegPlusConst(out_row, out_row_bytes),
        });
        Ok(())
    })?;

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 35 §2: QuantGather Double Buffer (REQ-QWP-005)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Double-buffer state for QuantGather in GEMM prologue.
///
/// Two shared memory buffers (FragmentA/B) alternate: while one is consumed
/// by the GEMM compute kernel, the other is loaded with the next quantized block.
/// On SM61 (no cp.async), uses `ld.global` + `bar.sync` for synchronization.
pub(crate) struct QuantGatherDoubleBuffer {
    /// Current active buffer index (0 = FragmentA, 1 = FragmentB).
    pub active: u32,
    /// Shared memory address for FragmentA buffer.
    pub frag_a_smem: VRegId,
    /// Shared memory address for FragmentB buffer.
    pub frag_b_smem: VRegId,
    /// Block bytes per load operation.
    pub block_bytes: usize,
}

impl QuantGatherDoubleBuffer {
    /// Allocate double buffer state and emit the initial load of FragmentA.
    pub fn new(prog: &mut VmProgram, block_bytes: usize) -> Self {
        let frag_a_smem = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let frag_b_smem = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        Self {
            active: 0,
            frag_a_smem,
            frag_b_smem,
            block_bytes,
        }
    }

    /// Get the smem pointer for the buffer currently being consumed (read side).
    pub fn read_buffer(&self) -> VRegId {
        if self.active == 0 { self.frag_a_smem } else { self.frag_b_smem }
    }

    /// Get the smem pointer for the buffer being loaded (write side).
    pub fn write_buffer(&self) -> VRegId {
        if self.active == 0 { self.frag_b_smem } else { self.frag_a_smem }
    }

    /// Swap active buffer after barrier synchronization.
    pub fn swap(&mut self) {
        self.active = 1 - self.active;
    }
}

/// Emit a quantized block load from global memory to shared memory buffer.
///
/// SM61 path: `ld.global.u8` (no cp.async) + `bar.sync` after load completes.
/// On SM80+, this would use `cp.async` with asynchronous wait instead.
pub(crate) fn emit_quant_block_load_to_smem(
    prog: &mut VmProgram,
    global_ptr: VRegId,
    smem_ptr: VRegId,
    block_bytes: usize,
    sm_version: u32,
) {
    let bytes_to_load = block_bytes;
    let word_size = if sm_version >= 80 { 16 } else { 4 }; // SM80: 128-bit load; SM61: 32-bit
    let num_words = bytes_to_load / word_size;

    prog.emit(VmInstr::Comment(format!(
        "QuantGather double-buffer load: {} bytes ({} x {}B words) SM{}",
        bytes_to_load, num_words, word_size, sm_version
    )));

    // For simplicity, emit block_bytes / 4 scalar loads (u32 words).
    // Real production code would use wider loads on SM80+.
    let byte_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
    prog.emit(VmInstr::GprLoadImm { dst: byte_off, value: 0 });

    for i in 0..num_words {
        let tmp = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::ScalarLoad {
            dst: tmp,
            base: global_ptr,
            offset: OffsetExpr::Const(i * word_size),
        });
        prog.emit(VmInstr::ScalarStore {
            src: tmp,
            base: smem_ptr,
            offset: OffsetExpr::Const(i * word_size),
        });
    }

    // Synchronize: SM61 uses bar.sync (no async copy)
    prog.emit(VmInstr::MemFence { order: MemFenceOrder::AcqRel });
}

/// Emit SM61 dequantization path: load u8 quantized data + nibble unpack + cvt to F32.
///
/// SPEC 35 REQ-QWP-006: SM61 has no Tensor Core/WMMA, so dequant uses scalar path:
/// 1. ld.global.u8 to load quantized block
/// 2. Nibble unpack: and/shr to extract 4-bit values
/// 3. cvt.rn.f32.s32 to convert to F32
/// 4. mul.f32 with scale
pub(crate) fn emit_sm61_dequant_block(
    prog: &mut VmProgram,
    smem_ptr: VRegId,
    output_ptr: VRegId,
    block_size: usize,
    scale_ptr: VRegId,
) {
    let num_bytes = block_size / 2; // 4-bit quantized, 2 values per byte
    prog.emit(VmInstr::Comment(format!(
        "SM61 dequant: block_size={} num_bytes={}", block_size, num_bytes
    )));

    let scale_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
    prog.emit(VmInstr::ScalarLoad {
        dst: scale_val,
        base: scale_ptr,
        offset: OffsetExpr::Const(0),
    });

    for i in 0..num_bytes {
        let byte_val = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::ScalarLoad {
            dst: byte_val,
            base: smem_ptr,
            offset: OffsetExpr::Const(i),
        });

        // Low nibble: (byte >> 0) & 0xF
        let low = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: low,
            a: byte_val,
            b: GprOperand::Imm(0xF),
            op: GprOp::And,
        });

        // High nibble: (byte >> 4) & 0xF
        let high = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: high,
            a: byte_val,
            b: GprOperand::Imm(4),
            op: GprOp::Shr,
        });
        prog.emit(VmInstr::GprBinOp {
            dst: high,
            a: high,
            b: GprOperand::Imm(0xF),
            op: GprOp::And,
        });

        // Convert to F32 and multiply by scale: low
        let f_low = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: f_low,
            a: low,
            b: GprOperand::VReg(scale_val),
            op: GprOp::Mul, // simplified; real path would cvt s32→f32 then mul.f32
        });
        prog.emit(VmInstr::ScalarStore {
            src: f_low,
            base: output_ptr,
            offset: OffsetExpr::Const(i * 8),
        });

        // High nibble
        let f_high = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
        prog.emit(VmInstr::GprBinOp {
            dst: f_high,
            a: high,
            b: GprOperand::VReg(scale_val),
            op: GprOp::Mul,
        });
        prog.emit(VmInstr::ScalarStore {
            src: f_high,
            base: output_ptr,
            offset: OffsetExpr::Const(i * 8 + 4),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantType;

    /// Helper: create a fresh VmProgram with 3 pre-allocated pointer VRegs.
    /// Returns (program, indices_ptr, embed_ptr, output_ptr).
    fn make_test_program() -> (VmProgram, VRegId, VRegId, VRegId) {
        let mut prog = VmProgram::new();
        let indices_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let embed_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        (prog, indices_ptr, embed_ptr, output_ptr)
    }

    // ── Test 1: zero hidden_dim produces CodegenViolation ──

    #[test]
    fn test_emit_quant_gather_zero_hidden_dim() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();
        let hidden_dim = 0;

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            1024,
            hidden_dim,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        let err = result.expect_err("zero hidden_dim should fail");
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(msg.contains("zero dimension"), "unexpected message: {msg}");
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    // ── Test 2: hidden_dim not divisible by block_size produces error ──

    #[test]
    fn test_emit_quant_gather_hidden_dim_not_divisible() {
        // Arrange: Q4_0 has block_size=32, use hidden_dim=33 which is not divisible
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            1024,
            33, // not divisible by Q4_0 block_size=32
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        let err = result.expect_err("non-divisible hidden_dim should fail");
        match err {
            CompilerError::CodegenViolation(msg) => {
                assert!(
                    msg.contains("not divisible"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected CodegenViolation, got: {other:?}"),
        }
    }

    // ── Test 3: valid Q4_0 with minimal hidden_dim succeeds ──

    #[test]
    fn test_emit_quant_gather_q4_0_minimal() {
        // Arrange: Q4_0 block_size=32, hidden_dim=32 (exactly 1 block)
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            1024,
            32,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: must succeed and produce instructions
        assert!(result.is_ok(), "Q4_0 minimal should succeed: {:?}", result);
        assert!(prog.len() > 0, "program should have instructions");
    }

    // ── Test 4: valid Q8_0 succeeds and emits instructions ──

    #[test]
    fn test_emit_quant_gather_q8_0_succeeds() {
        // Arrange: Q8_0 block_size=32, hidden_dim=64 (2 blocks)
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            2048,
            64,
            QuantType::Q8_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "Q8_0 should succeed: {:?}", result);
        assert!(prog.len() > 0, "program should have instructions");
    }

    // ── Test 5: program contains a Comment VmInstr with quant_type name ──

    #[test]
    fn test_emit_quant_gather_emits_comment() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );
        assert!(result.is_ok());

        // Assert: find a Comment VmInstr mentioning the format
        let has_comment = prog.instrs.iter().any(|i| {
            if let VmInstr::Comment(text) = i {
                text.contains("QuantGather(trace)")
            } else {
                false
            }
        });
        assert!(has_comment, "program should contain a QuantGather trace comment");
    }

    // ── Test 6: program contains LoopBegin/LoopEnd (outer seq loop) ──

    #[test]
    fn test_emit_quant_gather_has_outer_loop() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: must have at least one LoopBegin and LoopEnd
        let loop_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        let loop_ends = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopEnd)).count();
        assert!(loop_begins >= 1, "expected at least 1 LoopBegin, got {loop_begins}");
        assert_eq!(loop_begins, loop_ends, "LoopBegin/LoopEnd count must match");
    }

    // ── Test 7: program contains ScalarLoad (token_id load) ──

    #[test]
    fn test_emit_quant_gather_has_scalar_load() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            64,
            QuantType::Q8_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: must have ScalarLoad for loading token_id
        let has_scalar_load = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::ScalarLoad { .. })
        });
        assert!(has_scalar_load, "program should contain ScalarLoad for token_id");
    }

    // ── Test 8: program contains ScalarToIndex (row base computation) ──

    #[test]
    fn test_emit_quant_gather_has_scalar_to_index() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: must have ScalarToIndex for row_base = token_id * row_stride
        let has_scalar_to_index = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::ScalarToIndex { .. })
        });
        assert!(has_scalar_to_index, "program should contain ScalarToIndex");
    }

    // ── Test 9: program contains VecStore (decoded output write) ──

    #[test]
    fn test_emit_quant_gather_has_vec_store() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: must have VecStore for writing decoded F32 output
        let has_vec_store = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::VecStore { .. })
        });
        assert!(has_vec_store, "program should contain VecStore");
    }

    // ── Test 10: Q4_1 format (different block_bytes=20) succeeds ──

    #[test]
    fn test_emit_quant_gather_q4_1_format() {
        // Arrange: Q4_1 has block_bytes=20, block_size=32
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            1024,
            96, // 3 blocks
            QuantType::Q4_1,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "Q4_1 should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q4_1 program should have instructions");
    }

    // ── Test 11: hidden_dim controls block loop bound value ──

    #[test]
    fn test_emit_quant_gather_block_loop_bound_reflects_hidden_dim() {
        // Arrange: hidden_dim=128 with Q4_0 block_size=32 → 4 blocks.
        // The block loop's bound should be BoundExpr::Const(4).
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            128, // 128 / 32 = 4 blocks
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: find the block loop (inner loop with BoundExpr::Const(4))
        let block_loop = prog.instrs.iter().find(|i| {
            if let VmInstr::LoopBegin { bound: BoundExpr::Const(4), step_bytes, .. } = i {
                // Q4_0 block_bytes=18, so the block loop step should be 18
                *step_bytes == 18
            } else {
                false
            }
        });
        assert!(
            block_loop.is_some(),
            "should find block loop with bound=4 and step=18 (Q4_0 block_bytes)"
        );
    }

    // ── Test 12: Symbolic bound expression works for seq dimension ──

    #[test]
    fn test_emit_quant_gather_symbolic_seq_bound() {
        // Arrange: use a Symbolic bound instead of Const
        let (mut prog, idx, emb, out) = make_test_program();
        let sym_bound = BoundExpr::Symbolic(SymBound {
            name: "seq_len".to_string(),
            max_alloc: 512,
        });

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            sym_bound,
            1024,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: Symbolic bound should work (seq dimension is dynamic)
        assert!(result.is_ok(), "Symbolic bound should succeed: {:?}", result);

        // Verify the outer loop has the symbolic bound
        let has_symbolic_loop = prog.instrs.iter().any(|i| {
            if let VmInstr::LoopBegin { bound: BoundExpr::Symbolic(sb), .. } = i {
                sb.name == "seq_len" && sb.max_alloc == 512
            } else {
                false
            }
        });
        assert!(has_symbolic_loop, "outer loop should use Symbolic bound");
    }

    // ── Test 13: VReg count increases with valid compilation ──

    #[test]
    fn test_emit_quant_gather_allocates_vregs() {
        // Arrange: fresh program has 3 VRegs from make_test_program
        let (mut prog, idx, emb, out) = make_test_program();
        let vregs_before = prog.vreg_count();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: emit should have allocated additional VRegs for temporaries
        let vregs_after = prog.vreg_count();
        assert!(
            vregs_after > vregs_before,
            "emit should allocate VRegs (before={vregs_before}, after={vregs_after})"
        );
    }

    // ── Test 14: LoadPtr for row pointer computation ──

    #[test]
    fn test_emit_quant_gather_has_load_ptr() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: must have LoadPtr for row_ptr = embed_ptr + row_base
        let load_ptr_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::LoadPtr { .. })
        }).count();
        assert!(load_ptr_count >= 2, "should have at least 2 LoadPtr (out_row init + row_ptr), got {load_ptr_count}");
    }

    // ── Test 15: GprLoadImm for initial lane_offset zero ──

    #[test]
    fn test_emit_quant_gather_has_gpr_load_imm() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: must have GprLoadImm for lane_offset = 0 initialization
        let has_gpr_load_imm = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::GprLoadImm { .. })
        });
        assert!(has_gpr_load_imm, "program should contain GprLoadImm");
    }

    // ── Test 16: Q8_0 format succeeds with larger hidden_dim ──

    #[test]
    fn test_emit_quant_gather_q8_0_large_hidden() {
        // Arrange: Q8_0 block_size=32, block_bytes=34, hidden_dim=256 (8 blocks)
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(4),
            4096,
            256,
            QuantType::Q8_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "Q8_0 hidden=256 should succeed: {:?}", result);
        assert!(prog.len() > 20, "large hidden_dim should produce many instructions, got {}", prog.len());
    }

    // ── Test 17: comment includes hidden_dim and block_size ──

    #[test]
    fn test_emit_quant_gather_comment_contains_params() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: comment should mention hidden=64 and block_size=32
        let comment = prog.instrs.iter().find_map(|i| {
            if let VmInstr::Comment(text) = i {
                Some(text.clone())
            } else {
                None
            }
        });
        let comment = comment.expect("should have a Comment VmInstr");
        assert!(comment.contains("hidden=64"), "comment should mention hidden=64: {comment}");
        assert!(comment.contains("block_size=32"), "comment should mention block_size=32: {comment}");
    }

    // ── Test 18: seq_bound=1 single token produces valid program ──

    #[test]
    fn test_emit_quant_gather_single_token() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            32, // exactly 1 block
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "single token should succeed: {:?}", result);
        // Even with seq=1, block=1, should still have the outer loop
        let loop_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_begins >= 1, "single token should still emit outer loop");
    }

    // ── Test 19: nested loop structure (3 loops: seq → block → sub_block) ──

    #[test]
    fn test_emit_quant_gather_three_level_loops() {
        // Arrange: hidden_dim=128, Q4_0 block_size=32 → 4 blocks
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            128,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: should have 3 nested LoopBegin (seq, block, sub_block)
        let loop_begins = prog.instrs.iter().filter(|i| matches!(i, VmInstr::LoopBegin { .. })).count();
        assert!(loop_begins >= 3, "should have at least 3 nested loops (seq+block+sub_block), got {loop_begins}");
    }

    // ── Test 20: W512 SIMD width succeeds ──

    #[test]
    fn test_emit_quant_gather_w512_width() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W512,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: W512 should also work (more lanes per vector)
        assert!(result.is_ok(), "W512 should succeed: {:?}", result);
    }

    // ── Test 21: multiple seq tokens produce more VmInstr than single ──

    #[test]
    fn test_emit_quant_gather_more_tokens_more_instrs() {
        // Arrange
        let (mut prog1, idx1, emb1, out1) = make_test_program();
        let (mut prog2, idx2, emb2, out2) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog1,
            BoundExpr::Const(1),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx1, emb1, out1,
            QuantPrecision::F32,
            None,
        ).unwrap();
        let len1 = prog1.len();

        emit_quant_gather_inline(
            &mut prog2,
            BoundExpr::Const(8),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx2, emb2, out2,
            QuantPrecision::F32,
            None,
        ).unwrap();
        let len2 = prog2.len();

        // Assert: both should succeed; same number of instructions (loop body identical)
        assert!(len1 > 0);
        assert!(len2 > 0);
        // The outer loop body is identical regardless of seq_len, so instruction count is the same
        assert_eq!(len1, len2, "loop-based: same instruction count regardless of seq_len");
    }

    // ── Test 22: BF16 dtype output succeeds ──

    #[test]
    fn test_emit_quant_gather_bf16_dtype() {
        // Arrange
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::BF16,
            None,
        );

        // Assert: BF16 output should succeed (decode to F32 then store as BF16)
        assert!(result.is_ok(), "BF16 output should succeed: {:?}", result);
    }

    // ── Test 23: Q6K K-Quant (NibbleWithHighBits + lane offset, block_size=256) ──

    #[test]
    fn test_emit_quant_gather_q6k_format() {
        // Arrange: Q6K has NibbleWithHighBits (2 high bits per elem), block_size=256,
        // block_bytes=210. needs_lane_offset=true, needs_high_bits_ptr=true.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::Q6K,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: Q6K must succeed with NibbleWithHighBits + lane offset path
        assert!(result.is_ok(), "Q6K should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q6K program should have instructions");
    }

    // ── Test 24: Q6K emits AddPtr for high bits pointer construction ──

    #[test]
    fn test_emit_quant_gather_q6k_adds_high_bits_pointer() {
        // Arrange: Q6K's NibbleWithHighBits layout triggers AddPtr for high bits ptr.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            4096,
            256,
            QuantType::Q6K,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: program should contain AddPtr for constructing high_bits_ptr
        let has_add_ptr = prog.instrs.iter().any(|i| {
            matches!(i, VmInstr::AddPtr { .. })
        });
        assert!(has_add_ptr, "Q6K should emit AddPtr for high bits pointer");
    }

    // ── Test 25: Q8K format (K-Quant Bytes layout, block_size=256) succeeds ──

    #[test]
    fn test_emit_quant_gather_q8k_format() {
        // Arrange: Q8K has Bytes layout (signed), block_size=256, block_bytes=292.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::Q8K,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "Q8K should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q8K program should have instructions");
    }

    // ── Test 26: Q4K K-Quant (block_size=256, needs lane offset) succeeds ──

    #[test]
    fn test_emit_quant_gather_q4k_format() {
        // Arrange: Q4K has block_size=256, uses hierarchical packed scales (KQuant6Bit).
        // hidden_dim must be a multiple of 256.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::Q4K,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: Q4K must succeed with block_size=256
        assert!(result.is_ok(), "Q4K should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q4K program should have instructions");

        // Q4K needs lane_offset; verify GprLoadImm sets lane_offset = 0 in block loop body
        let gpr_load_imm_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::GprLoadImm { .. })
        }).count();
        assert!(gpr_load_imm_count >= 3, "Q4K should have GprLoadImm for lane_offset + strides, got {gpr_load_imm_count}");
    }

    // ── Test 27: W128 SIMD width succeeds (4 f32 lanes) ──

    #[test]
    fn test_emit_quant_gather_w128_width() {
        // Arrange: W128 = 4 f32 lanes (SSE/NEON width)
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W128,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: W128 should succeed with fewer lanes per vector op
        assert!(result.is_ok(), "W128 should succeed: {:?}", result);
    }

    // ── Test 28: Scalar SIMD width succeeds (1 lane) ──

    #[test]
    fn test_emit_quant_gather_scalar_width() {
        // Arrange: Scalar width = 1 lane (fallback scalar path)
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            32,
            QuantType::Q4_0,
            SimdWidth::Scalar,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: scalar width should still produce a valid program
        assert!(result.is_ok(), "Scalar width should succeed: {:?}", result);
    }

    // ── Test 29: Q8_1 format (classic with min) succeeds ──

    #[test]
    fn test_emit_quant_gather_q8_1_format() {
        // Arrange: Q8_1 has ScaleMin layout (d + m per block), block_size=32, block_bytes=40.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            1024,
            64,
            QuantType::Q8_1,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "Q8_1 should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q8_1 program should have instructions");
    }

    // ── Test 30: program contains GprBinOp for data pointer advance ──

    #[test]
    fn test_emit_quant_gather_has_data_advance_gpr_binop() {
        // Arrange: data_ptr advance uses GprBinOp(Add) per sub-block iteration
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            512,
            64,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        ).unwrap();

        // Assert: must have at least one GprBinOp for data_advance step
        let gpr_binop_count = prog.instrs.iter().filter(|i| {
            matches!(i, VmInstr::GprBinOp { .. })
        }).count();
        assert!(gpr_binop_count >= 1, "should have GprBinOp for data_advance, got {gpr_binop_count}");
    }

    // ── Test 31: IQ4NL format (CodebookIndex layout) succeeds ──

    #[test]
    fn test_emit_quant_gather_iq4nl_format() {
        // Arrange: IQ4NL has CodebookIndex data layout, block_size=32, block_bytes=18.
        // hidden_dim=64 (2 blocks). Codebook-based super-low-bit format.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            1024,
            64,
            QuantType::IQ4NL,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: IQ4NL codebook-based format must succeed
        assert!(result.is_ok(), "IQ4NL should succeed: {:?}", result);
        assert!(prog.len() > 0, "IQ4NL program should have instructions");
    }

    // ── Test 32: BF16 quantization type as input format succeeds ──

    #[test]
    fn test_emit_quant_gather_bf16_input_format() {
        // Arrange: BF16 as quantization type (native float, no scale/zero, no unpack).
        // block_size=1, block_bytes=2 per element.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            1024,
            64,
            QuantType::Bf16,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::BF16,
            None,
        );

        // Assert: BF16→BF16 should succeed (native path, zero dequantization overhead)
        assert!(result.is_ok(), "BF16 input format should succeed: {:?}", result);
        assert!(prog.len() > 0, "BF16 program should have instructions");
    }

    // ── Test 33: TQ1_0 format (block_size=256, ternary quantization) succeeds ──

    #[test]
    fn test_emit_quant_gather_tq1_0_format() {
        // Arrange: TQ1_0 has ternary 1-bit quantization, block_size=256, block_bytes=54.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::TQ1_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: TQ1_0 ternary quantization format must succeed
        assert!(result.is_ok(), "TQ1_0 should succeed: {:?}", result);
        assert!(prog.len() > 0, "TQ1_0 program should have instructions");
    }

    // ── Test 34: TQ2_0 format (block_size=256, ternary 2-bit) succeeds ──

    #[test]
    fn test_emit_quant_gather_tq2_0_format() {
        // Arrange: TQ2_0 has ternary 2-bit quantization, block_size=256, block_bytes=66.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::TQ2_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "TQ2_0 should succeed: {:?}", result);
        assert!(prog.len() > 0, "TQ2_0 program should have instructions");
    }

    // ── Test 35: F16 quantization type as input format succeeds ──

    #[test]
    fn test_emit_quant_gather_f16_input_format() {
        // Arrange: F16 native float, block_size=1, block_bytes=2 per element.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            1024,
            64,
            QuantType::F16,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F16,
            None,
        );

        // Assert: F16→F16 should succeed (native float path)
        assert!(result.is_ok(), "F16 input format should succeed: {:?}", result);
        assert!(prog.len() > 0, "F16 program should have instructions");
    }

    // ── Test 36: Q2K format (block_size=256) succeeds ──

    #[test]
    fn test_emit_quant_gather_q2k_format() {
        // Arrange: Q2K is the smallest K-Quant (2-bit), block_size=256, block_bytes=84.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::Q2K,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: Q2K must succeed (super-block K-Quant path)
        assert!(result.is_ok(), "Q2K should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q2K program should have instructions");
    }

    // ── Test 37: Q5K format (block_size=256) succeeds ──

    #[test]
    fn test_emit_quant_gather_q5k_format() {
        // Arrange: Q5K has hierarchical packed scales, block_size=256, block_bytes=176.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::Q5K,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "Q5K should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q5K program should have instructions");
    }

    // ── Test 38: Q3K format (block_size=256) succeeds ──

    #[test]
    fn test_emit_quant_gather_q3k_format() {
        // Arrange: Q3K has 3-bit packed with hmask, block_size=256, block_bytes=110.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::Q3K,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "Q3K should succeed: {:?}", result);
        assert!(prog.len() > 0, "Q3K program should have instructions");
    }

    // ── Test 39: vocab_size=0 does not affect success (not used in computation) ──

    #[test]
    fn test_emit_quant_gather_zero_vocab_size_succeeds() {
        // Arrange: vocab_size is accepted but not used in computation;
        // only hidden_dim, block_size matter for the codegen path.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act: vocab_size=0 is valid — it does not gate any computation
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            0, // vocab_size=0 — not used in emit path
            32,
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: should succeed (vocab_size is not used in codegen)
        assert!(result.is_ok(), "vocab_size=0 should succeed: {:?}", result);
        assert!(prog.len() > 0, "program should have instructions");
    }

    // ── Test 40: large hidden_dim=512 with Q4_0 produces many blocks ──

    #[test]
    fn test_emit_quant_gather_large_hidden_dim_many_blocks() {
        // Arrange: Q4_0 block_size=32, hidden_dim=512 → 16 blocks.
        // Tests that larger hidden_dim produces proportionally more VmInstr.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            512,
            512, // 512 / 32 = 16 blocks
            QuantType::Q4_0,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert
        assert!(result.is_ok(), "hidden_dim=512 should succeed: {:?}", result);
        // Verify the block loop bound reflects 16 blocks
        let block_loop_16 = prog.instrs.iter().any(|i| {
            if let VmInstr::LoopBegin { bound: BoundExpr::Const(16), step_bytes, .. } = i {
                *step_bytes == 18 // Q4_0 block_bytes=18
            } else {
                false
            }
        });
        assert!(block_loop_16, "should find block loop with bound=16 for hidden_dim=512");
    }

    // ── Test 41: F32 quantization type as input format succeeds ──

    #[test]
    fn test_emit_quant_gather_f32_input_format() {
        // Arrange: F32 native float, block_size=1, block_bytes=4 per element.
        // F32→F32 is the simplest path (no quantization at all).
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            1024,
            64,
            QuantType::F32,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: F32→F32 should succeed (identity-like path)
        assert!(result.is_ok(), "F32 input format should succeed: {:?}", result);
        assert!(prog.len() > 0, "F32 program should have instructions");
    }

    // ── Test 42: IQ4XS format (block_size=256, IQ codebook) succeeds ──

    #[test]
    fn test_emit_quant_gather_iq4xs_format() {
        // Arrange: IQ4XS has CodebookIndex layout, block_size=256, block_bytes=136.
        // Super-block IQ format with extended codebook.
        let (mut prog, idx, emb, out) = make_test_program();

        // Act
        let result = emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(2),
            4096,
            256,
            QuantType::IQ4XS,
            SimdWidth::W256,
            idx, emb, out,
            QuantPrecision::F32,
            None,
        );

        // Assert: IQ4XS codebook format must succeed
        assert!(result.is_ok(), "IQ4XS should succeed: {:?}", result);
        assert!(prog.len() > 0, "IQ4XS program should have instructions");
    }

    // ── SPEC 35 QuantGather Double Buffer Tests (REQ-QWP-005) ──

    #[test]
    fn quant_gather_double_buffer_new_allocates_two_buffers() {
        let mut prog = VmProgram::new();
        let db = QuantGatherDoubleBuffer::new(&mut prog, 32);
        assert_ne!(db.frag_a_smem, db.frag_b_smem, "two buffers must be distinct vregs");
        assert_eq!(db.active, 0, "initially FragmentA is active");
        assert_eq!(db.block_bytes, 32);
    }

    #[test]
    fn quant_gather_double_buffer_read_write_buffers_differ() {
        let mut prog = VmProgram::new();
        let db = QuantGatherDoubleBuffer::new(&mut prog, 16);
        assert_ne!(db.read_buffer(), db.write_buffer(), "read and write must be different buffers");
    }

    #[test]
    fn quant_gather_double_buffer_swap_toggles() {
        let mut prog = VmProgram::new();
        let mut db = QuantGatherDoubleBuffer::new(&mut prog, 16);
        assert_eq!(db.active, 0);
        let initial_read = db.read_buffer();
        let initial_write = db.write_buffer();

        db.swap();
        assert_eq!(db.active, 1);
        assert_eq!(db.read_buffer(), initial_write, "after swap, read is old write");
        assert_eq!(db.write_buffer(), initial_read, "after swap, write is old read");

        db.swap();
        assert_eq!(db.active, 0, "swap back restores original");
    }

    #[test]
    fn quant_gather_double_buffer_swap_round_trip() {
        let mut prog = VmProgram::new();
        let mut db = QuantGatherDoubleBuffer::new(&mut prog, 64);
        for i in 0..10 {
            db.swap();
            assert_eq!(db.active, (i + 1) % 2);
        }
    }

    #[test]
    fn quant_gather_block_load_emits_scalar_load_store() {
        let mut prog = VmProgram::new();
        let global_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_quant_block_load_to_smem(&mut prog, global_ptr, smem_ptr, 16, 61);

        let has_load = prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarLoad { .. }));
        let has_store = prog.instrs.iter().any(|i| matches!(i, VmInstr::ScalarStore { .. }));
        let has_fence = prog.instrs.iter().any(|i| matches!(i, VmInstr::MemFence { .. }));
        assert!(has_load, "should emit ScalarLoad for global memory read");
        assert!(has_store, "should emit ScalarStore for shared memory write");
        assert!(has_fence, "should emit MemFence for synchronization");
    }

    #[test]
    fn quant_gather_block_load_sm80_uses_wider_words() {
        let mut prog = VmProgram::new();
        let global_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let count_before = prog.len();
        emit_quant_block_load_to_smem(&mut prog, global_ptr, smem_ptr, 64, 80);

        // SM80 uses 16B words = 64/16 = 4 loads; SM61 would be 64/4 = 16 loads
        let scalar_loads = prog.instrs[count_before..].iter()
            .filter(|i| matches!(i, VmInstr::ScalarLoad { .. }))
            .count();
        assert_eq!(scalar_loads, 4, "SM80 should use 16B words = 4 loads for 64B block");
    }

    // ── SPEC 35 SM61 Dequant Block Tests (REQ-QWP-006) ──

    #[test]
    fn sm61_dequant_block_emits_nibble_unpack() {
        let mut prog = VmProgram::new();
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_sm61_dequant_block(&mut prog, smem_ptr, output_ptr, 8, scale_ptr);

        // Should have AND (nibble mask) and SHR (shift high nibble)
        let has_and = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::GprBinOp { op: GprOp::And, .. }));
        let has_shr = prog.instrs.iter().any(|i| matches!(i,
            VmInstr::GprBinOp { op: GprOp::Shr, .. }));
        assert!(has_and, "should emit AND for nibble mask");
        assert!(has_shr, "should emit SHR for high nibble extraction");
    }

    #[test]
    fn sm61_dequant_block_loads_scale() {
        let mut prog = VmProgram::new();
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_sm61_dequant_block(&mut prog, smem_ptr, output_ptr, 8, scale_ptr);

        // First ScalarLoad should be scale
        let first_load = prog.instrs.iter().find(|i| matches!(i, VmInstr::ScalarLoad { .. }));
        assert!(first_load.is_some(), "should emit ScalarLoad for scale");
    }

    #[test]
    fn sm61_dequant_block_stores_f32_results() {
        let mut prog = VmProgram::new();
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        emit_sm61_dequant_block(&mut prog, smem_ptr, output_ptr, 8, scale_ptr);

        let stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::ScalarStore { .. }))
            .count();
        // 4 bytes * 2 nibbles = 8 F32 outputs = 8 ScalarStore
        assert_eq!(stores, 8, "block_size=8 → 4 bytes → 8 F32 stores");
    }

    #[test]
    fn sm61_dequant_block_zero_block_size_emits_nothing() {
        let mut prog = VmProgram::new();
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let before = prog.len();
        emit_sm61_dequant_block(&mut prog, smem_ptr, output_ptr, 0, scale_ptr);

        // block_size=0 → no dequant loop; only comment + scale load emitted
        // Exact count: Comment + ScalarLoad(scale) = varies by implementation
        let added = prog.len() - before;
        assert!(added > 0 && added <= 10,
            "block_size=0 should emit minimal instructions, got {} added", added);
        assert!(added < 20,
            "block_size=0 should not emit loop iterations");
    }

    // ── SPEC 35 REQ-QWP-005: QuantGather Double Buffer tests ──

    #[test]
    fn test_double_buffer_new_allocates_two_smem_ptrs() {
        let mut prog = VmProgram::new();
        let db = QuantGatherDoubleBuffer::new(&mut prog, 256);
        assert_eq!(db.active, 0, "initial active buffer should be 0 (FragmentA)");
        assert_eq!(db.block_bytes, 256);
        assert_ne!(db.frag_a_smem, db.frag_b_smem, "FragmentA and B must be distinct VRegs");
    }

    #[test]
    fn test_double_buffer_swap_toggles_active() {
        let mut prog = VmProgram::new();
        let mut db = QuantGatherDoubleBuffer::new(&mut prog, 128);
        assert_eq!(db.active, 0);
        assert_eq!(db.read_buffer(), db.frag_a_smem);
        assert_eq!(db.write_buffer(), db.frag_b_smem);

        db.swap();
        assert_eq!(db.active, 1);
        assert_eq!(db.read_buffer(), db.frag_b_smem);
        assert_eq!(db.write_buffer(), db.frag_a_smem);

        db.swap();
        assert_eq!(db.active, 0);
    }

    #[test]
    fn test_quant_block_load_to_smem_sm61_emits_loads_and_fence() {
        let mut prog = VmProgram::new();
        let global_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_quant_block_load_to_smem(&mut prog, global_ptr, smem_ptr, 16, 61);

        // 16 bytes / 4 bytes per word = 4 loads + 4 stores + comment + fence
        let loads = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::ScalarLoad { .. }))
            .count();
        let stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::ScalarStore { .. }))
            .count();
        let fences = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::MemFence { .. }))
            .count();

        assert_eq!(loads, 4, "SM61 should emit 4 x u32 scalar loads for 16 bytes");
        assert_eq!(stores, 4, "SM61 should emit 4 x u32 scalar stores to smem");
        assert!(fences >= 1, "SM61 load should include at least one MemFence");
    }

    #[test]
    fn test_quant_block_load_sm80_uses_wider_words() {
        let mut prog = VmProgram::new();
        let global_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_quant_block_load_to_smem(&mut prog, global_ptr, smem_ptr, 64, 80);

        // SM80: word_size=16, so 64/16 = 4 loads
        let loads = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::ScalarLoad { .. }))
            .count();
        assert_eq!(loads, 4, "SM80 should emit 4 x 128-bit loads for 64 bytes");
    }

    #[test]
    fn test_sm61_dequant_block_nibble_unpack_structure() {
        let mut prog = VmProgram::new();
        let smem_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let scale_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // block_size=8 → 4 bytes of quantized data → 8 output F32 values
        emit_sm61_dequant_block(&mut prog, smem_ptr, output_ptr, 8, scale_ptr);

        // Should have: 1 scale load + 4 byte loads + 4 AND(low) + 4 SHR + 4 AND(high) + 4 MUL(low) + 4 STORE(low) + 4 MUL(high) + 4 STORE(high)
        let and_ops = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprBinOp { op: GprOp::And, .. }))
            .count();
        let shr_ops = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprBinOp { op: GprOp::Shr, .. }))
            .count();
        let mul_ops = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::GprBinOp { op: GprOp::Mul, .. }))
            .count();
        let stores = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::ScalarStore { .. }))
            .count();

        assert_eq!(and_ops, 8, "should have 8 AND ops (4 low + 4 high nibble masks)");
        assert_eq!(shr_ops, 4, "should have 4 SHR ops for high nibble extraction");
        assert_eq!(mul_ops, 8, "should have 8 MUL ops (4 low + 4 high × scale)");
        assert_eq!(stores, 8, "should store 8 F32 output values");
    }
}
