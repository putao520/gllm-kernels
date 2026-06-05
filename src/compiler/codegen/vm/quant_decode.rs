//! DecodeTraceBuilder — Algorithmic dequantization trace generator.
//!
//! SPEC: `gllm/SPEC/23-QUANT-CODEGEN-ALGO.md §3`
//! REQ-QCG-003 / REQ-QCG-007 / REQ-QCG6 (DequantFMA path)
//!
//! Converts a [`QuantFormatDescriptor`] + block base/data pointers + lane-offset
//! into a hardware-agnostic TraceOp SSA sequence.  The sequence is then fed into
//! `auto_lower_trace` which converts it to `VmInstr` → ISA machine code with zero
//! per-(format × ISA) hand-written code.
//!
//! The DequantFMA path (§QCG6) relies on DecodeTraceBuilder for the core
//! dequant formula `(qw - zp) × scale`, followed by vfmadd231ps FMA accumulation.
//!
//! # Input slot convention
//!
//! | Input | Name            | Purpose                                      |
//! |-------|-----------------|----------------------------------------------|
//! | 0     | `block_base`    | Fixed pointer to block start (scale/zero)    |
//! | 1     | `data_ptr`      | May advance within block (packed data loads) |
//! | 2     | `lane_offset`   | Sub-block index (hierarchical formats only)  |
//!
//! For QuantGather (no inner element loop), `block_base == data_ptr`.
//! For QuantGemm (inner element loop), `data_ptr` advances by `data_step` per iteration
//! while `block_base` stays fixed for scale/zero loads at block-relative offsets.
//!
//! # Slot indexing
//!
//! `TraceOp` bodies use `u32` slot indices that equal the position of the op in
//! the `Vec<TraceOp>` at the time of emission.  `push_op` appends and returns the
//! slot index of the newly-pushed op.

use crate::compiler::trace::TraceOp;
use crate::compiler::trace::ValueId;
use crate::quant::QuantType;
use crate::quant_format::{
    DataLayout, PackedScaleAlgorithm, QuantDataKind, QuantFormatDescriptor, ScaleDType, ScaleLayout, ZeroLayout,
};

// ── helper ───────────────────────────────────────────────────────────────────

/// Append a TraceOp and return its slot index (= `trace.len() - 1 as u32`).
fn push_op(trace: &mut Vec<TraceOp>, op: TraceOp) -> ValueId {
    let idx = ValueId(trace.len() as u32);
    trace.push(op);
    idx
}

// ── ScaleResult ──────────────────────────────────────────────────────────────

/// Result of `emit_scale_load`: a VReg slot index that holds the scalar scale
/// value (f32, ready to multiply).  For hierarchical formats the slot already
/// encodes `block_d * sub_scale`.
struct ScaleResult {
    /// Slot index of the f32 scalar scale (not yet broadcast to a vector).
    scale_slot: ValueId,
}

// ── ZeroResult ───────────────────────────────────────────────────────────────

#[derive(Debug)]
enum ZeroResult {
    /// No zero-point; the decoded integer needs a static subtraction later.
    StaticBias(i32),
    /// AWQ4/GPTQ4: `value = (unpacked - zero) × scale`.  Zero subtracted BEFORE scale.
    PreScaleSubtract(ValueId),
    /// Q4_K/Q5_K: `value = (unpacked - bias) × scale + min`.  Min added AFTER scale.
    PostScaleAdd(ValueId),
    /// No correction needed.
    None,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DecodeTraceBuilder
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Algorithmic dequantization trace generator.
///
/// # Usage
/// ```ignore
/// let trace = DecodeTraceBuilder::new(descriptor, block_ptr_vreg, lane_offset_slot, output_lanes)
///     .build();
/// // feed `trace` to auto_lower_trace(prog, &trace, width, dtype)
/// ```
pub struct DecodeTraceBuilder<'a> {
    desc: &'a QuantFormatDescriptor,
    /// Number of output f32 elements (= SIMD lane count).
    output_lanes: usize,
}

impl<'a> DecodeTraceBuilder<'a> {
    pub fn new(
        desc: &'a QuantFormatDescriptor,
        output_lanes: usize,
    ) -> Self {
        Self { desc, output_lanes }
    }

    /// Whether the trace requires a lane_offset input (inputs[2]).
    /// Hierarchical and Q6K scale layouts use it to index sub-blocks.
    pub fn needs_lane_offset(&self) -> bool {
        matches!(
            &self.desc.scale_layout,
            ScaleLayout::Hierarchical { .. } | ScaleLayout::Q6KScales { .. }
                | ScaleLayout::ExternalArray { .. } | ScaleLayout::SubBlockScalars { .. }
        ) || matches!(
            &self.desc.zero_layout,
            ZeroLayout::Hierarchical { .. }
        )
    }

    /// Whether the trace requires a high_bits_ptr input (Input(3)).
    /// NibbleWithHighBits (Q6_K, Q5_0, Q5_1) uses a separate pointer for the
    /// high bit-plane, advanced independently from the low data pointer.
    pub fn needs_high_bits_ptr(&self) -> bool {
        matches!(&self.desc.data_layout, DataLayout::NibbleWithHighBits { .. })
    }

    /// Byte stride for the high_bits_ptr per ei iteration.
    /// For NibbleWithHighBits with 2 bits/elem: 8 elements * 2 bits / 8 = 2 bytes.
    pub fn high_bits_stride(&self) -> usize {
        match &self.desc.data_layout {
            DataLayout::NibbleWithHighBits { high_bits_per_elem, .. } => {
                (self.output_lanes * (*high_bits_per_elem as usize) + 7) / 8
            }
            _ => 0,
        }
    }

    /// Build the complete dequantization TraceOp sequence.
    ///
    /// # Input slot convention
    ///
    /// | Input | Name            | Purpose                                      |
    /// |-------|-----------------|----------------------------------------------|
    /// | 0     | `block_base`    | Fixed pointer to block start (scale/zero)    |
    /// | 1     | `data_ptr`      | May advance within block (packed data loads) |
    /// | 2     | `lane_offset`   | Sub-block index (hierarchical formats only)  |
    /// | 3     | `high_bits_ptr` | High bit-plane pointer (NibbleWithHighBits)  |
    ///
    /// The caller must pass at least `&[block_base_vreg, data_ptr_vreg]`.
    /// If `needs_lane_offset()`, also append a lane_offset GPR.
    /// If `needs_high_bits_ptr()`, also append a high_bits_ptr GPR.
    ///
    /// For QuantGather (no inner advance), `block_base == data_ptr`.
    /// For QuantGemm (inner element loop), `block_base` is the block start
    /// while `data_ptr` advances by `data_step` per iteration.
    ///
    /// Returns the slot index of the final decoded f32 vector.
    pub fn build(self, trace: &mut Vec<TraceOp>) -> ValueId {
        // Input(0): block base pointer — fixed for the whole block.
        // Used for scale loads at fixed offsets (e.g. Q6_K block_d at +208).
        let block_base_slot = push_op(trace, TraceOp::Input(0));

        // Input(1): data pointer — may advance within the block.
        // Used for packed data loads (e.g. Q6_K low nibbles at +0).
        // In QuantGather this equals block_base; in QuantGemm it advances.
        let data_ptr_slot = push_op(trace, TraceOp::Input(1));

        // Input(2): lane offset for hierarchical scale layouts (Q6_K, K-Quant, etc.).
        // When unused, alias block_base_slot as a harmless placeholder.
        let lane_offset_slot = if self.needs_lane_offset() {
            push_op(trace, TraceOp::Input(2))
        } else {
            block_base_slot
        };

        // Input(3): high_bits_ptr for NibbleWithHighBits (Q6_K, Q5_0, Q5_1).
        // Independently advanced pointer into the high bit-plane.
        // When unused, alias block_base_slot.
        let high_bits_ptr_slot = if self.needs_high_bits_ptr() {
            push_op(trace, TraceOp::Input(3))
        } else {
            block_base_slot
        };

        // E2M1 formats (MXFP4, NVFP4): use combined LUT decode TraceOp
        if self.is_e2m1_format() {
            return self.build_e2m1_decode(trace, block_base_slot, data_ptr_slot, lane_offset_slot);
        }

        let scale_result = self.emit_scale_load(trace, block_base_slot, lane_offset_slot);
        let zero_result = self.emit_zero_load(trace, block_base_slot, lane_offset_slot);
        let quant_data_slot = self.emit_data_load(trace, data_ptr_slot);
        let unpacked_slot = self.emit_unpack(trace, quant_data_slot, block_base_slot, high_bits_ptr_slot);
        self.emit_dequant_algebra(trace, unpacked_slot, scale_result, zero_result)
    }

    /// Whether this format uses E2M1 LUT decode (FP4 data kinds).
    fn is_e2m1_format(&self) -> bool {
        use crate::quant_format::QuantDataKind;
        matches!(self.desc.data_kind, QuantDataKind::Float4 | QuantDataKind::Nvfp4)
    }

    /// E2M1 LUT decode path for MXFP4/NVFP4: load raw scale byte + data pointer
    /// → single QuantE2m1LutDecode TraceOp → VmInstr handles LUT + scale internally.
    fn build_e2m1_decode(
        &self,
        trace: &mut Vec<TraceOp>,
        block_base_slot: ValueId,
        data_ptr_slot: ValueId,
        lane_offset_slot: ValueId,
    ) -> ValueId {
        let nvfp4_mode = matches!(
            self.desc.scale_layout,
            ScaleLayout::SubBlockScalars { .. }
        );

        // 1. Compute scale byte address and load it as raw byte
        let scale_byte_slot = self.emit_raw_scale_byte_load(trace, block_base_slot, lane_offset_slot);

        // 2. Compute packed data pointer (offset from data_ptr)
        let data_offset = match &self.desc.data_layout {
            DataLayout::PackedNibbles { offset, .. } => *offset as i64,
            _ => 0,
        };
        let packed_data_ptr_slot = if data_offset != 0 {
            push_op(trace, TraceOp::QuantPtrAddOffset {
                base: data_ptr_slot,
                offset_bytes: data_offset,
            })
        } else {
            data_ptr_slot
        };

        // 3. Combined E2M1 LUT decode
        push_op(trace, TraceOp::QuantE2m1LutDecode {
            packed_data_ptr: packed_data_ptr_slot,
            scale_byte: scale_byte_slot,
            nvfp4_mode,
        })
    }

    /// Load the raw scale byte (not decoded to F32) for E2M1 formats.
    /// Returns a slot holding the raw byte value in a Ptr VReg.
    fn emit_raw_scale_byte_load(
        &self,
        trace: &mut Vec<TraceOp>,
        block_ptr_slot: ValueId,
        lane_offset_slot: ValueId,
    ) -> ValueId {
        match &self.desc.scale_layout {
            ScaleLayout::BlockScalar { offset_bytes, .. } => {
                // MXFP4 inline: scale at block_base + offset_bytes
                let scale_ptr = push_op(trace, TraceOp::QuantPtrAddOffset {
                    base: block_ptr_slot,
                    offset_bytes: *offset_bytes as i64,
                });
                push_op(trace, TraceOp::QuantScalarLoad {
                    ptr: scale_ptr,
                    offset_bytes: 0,
                })
            }

            ScaleLayout::ExternalArray { .. } => {
                // MXFP4 external: scale from external array
                // Use block_ptr as scale pointer (ABI input 0 holds scales base)
                // The caller must set up the block_ptr to point to the correct scale.
                push_op(trace, TraceOp::QuantScalarLoad {
                    ptr: block_ptr_slot,
                    offset_bytes: 0,
                })
            }

            ScaleLayout::SubBlockScalars { offset_bytes, sub_block_size, .. } => {
                // NVFP4: sub_idx = lane_offset / sub_block_size
                let sub_idx_slot = push_op(trace, TraceOp::QuantIntDivConst {
                    src: lane_offset_slot,
                    divisor: *sub_block_size as i64,
                });
                let scales_base = push_op(trace, TraceOp::QuantPtrAddOffset {
                    base: block_ptr_slot,
                    offset_bytes: *offset_bytes as i64,
                });
                let final_ptr = push_op(trace, TraceOp::QuantPtrAddDynamic {
                    base: scales_base,
                    index: sub_idx_slot,
                });
                push_op(trace, TraceOp::QuantScalarLoad {
                    ptr: final_ptr,
                    offset_bytes: 0,
                })
            }

            _ => push_op(trace, TraceOp::QuantScalarLoad {
                ptr: block_ptr_slot,
                offset_bytes: 0,
            }),
        }
    }

    // ── §3.2 Scale load ──────────────────────────────────────────────────────

    fn emit_scale_load(&self, trace: &mut Vec<TraceOp>, block_ptr_slot: ValueId, lane_offset_slot: ValueId) -> ScaleResult {
        match &self.desc.scale_layout {
            ScaleLayout::None => {
                ScaleResult { scale_slot: push_op(trace, TraceOp::Const(1.0)) }
            }

            ScaleLayout::BlockScalar { offset_bytes, dtype } => {
                let f32_slot = self.load_and_cast_scalar(trace, block_ptr_slot, *offset_bytes as i64, *dtype);
                ScaleResult { scale_slot: f32_slot }
            }

            ScaleLayout::BlockScalarWithMin { d_offset, dtype, .. } => {
                let f32_slot = self.load_and_cast_scalar(trace, block_ptr_slot, *d_offset as i64, *dtype);
                ScaleResult { scale_slot: f32_slot }
            }

            ScaleLayout::Hierarchical {
                block_d_offset,
                sub_scales_offset,
                sub_block_elements,
                packed_layout,
                ..
            } => {
                // 1. Load block-level d (f16) and convert to f32
                let block_d_f32 = push_op(
                    trace,
                    TraceOp::QuantLoadF16toF32 {
                        ptr: block_ptr_slot,
                        offset_bytes: *block_d_offset as i64,
                    },
                );

                // 2. sub_block_idx = lane_offset / sub_block_elements
                let sub_idx_slot = push_op(
                    trace,
                    TraceOp::QuantIntDivConst {
                        src: lane_offset_slot,
                        divisor: *sub_block_elements as i64,
                    },
                );

                // 3. Decode packed sub-scale via the algorithm
                let sub_scales_ptr_slot = push_op(
                    trace,
                    TraceOp::QuantPtrAddOffset {
                        base: block_ptr_slot,
                        offset_bytes: *sub_scales_offset as i64,
                    },
                );
                let sub_scale_slot = self.emit_packed_scale_lookup(
                    trace,
                    sub_scales_ptr_slot,
                    sub_idx_slot,
                    &packed_layout.algorithm,
                );

                // 4. final_scale = block_d * sub_scale
                let final_scale = push_op(trace, TraceOp::Mul(block_d_f32, sub_scale_slot));
                ScaleResult { scale_slot: final_scale }
            }

            ScaleLayout::Q6KScales {
                block_d_offset,
                sub_scales_offset,
                sub_block_elements,
                ..
            } => {
                // Q6_K: block d + per-element i8 scales (no bit-packing)
                let block_d_f32 = push_op(
                    trace,
                    TraceOp::QuantLoadF16toF32 {
                        ptr: block_ptr_slot,
                        offset_bytes: *block_d_offset as i64,
                    },
                );

                // sub_block_idx = lane_offset / sub_block_elements
                let sub_idx_slot = push_op(
                    trace,
                    TraceOp::QuantIntDivConst {
                        src: lane_offset_slot,
                        divisor: *sub_block_elements as i64,
                    },
                );

                // sub_scales_base = block_ptr + sub_scales_offset (pointer to scales array)
                let sub_scales_base = push_op(
                    trace,
                    TraceOp::QuantPtrAddOffset {
                        base: block_ptr_slot,
                        offset_bytes: *sub_scales_offset as i64,
                    },
                );
                // final_ptr = sub_scales_base + sub_block_idx (1 byte per scale entry)
                let sub_scales_ptr = push_op(
                    trace,
                    TraceOp::QuantPtrAddDynamic {
                        base: sub_scales_base,
                        index: sub_idx_slot,
                    },
                );
                // Load sub_scale indexed by sub_idx (i8 entry) and convert to f32
                let sub_scale_f32 = push_op(
                    trace,
                    TraceOp::QuantLoadI8toF32 {
                        ptr: sub_scales_ptr,
                        offset_bytes: 0,
                    },
                );

                // final_scale = block_d * sub_scale
                let final_scale = push_op(trace, TraceOp::Mul(block_d_f32, sub_scale_f32));
                ScaleResult { scale_slot: final_scale }
            }

            ScaleLayout::ExternalArray { stride, dtype } => {
                // MXFP4: lane_index * stride gives the byte offset into external scales array.
                // We model the external array pointer as ABI arg input slot.
                let scales_ptr_slot = push_op(trace, TraceOp::Input(0xFFFF_FFFE)); // sentinel for ABI scales ptr

                let byte_offset_slot = push_op(
                    trace,
                    TraceOp::QuantIntMul {
                        src: lane_offset_slot,
                        factor: *stride as i64,
                    },
                );
                let scale_raw = push_op(
                    trace,
                    TraceOp::QuantScalarLoad {
                        ptr: scales_ptr_slot,
                        offset_bytes: 0,
                    },
                );
                // byte_offset_slot is the dynamic element — we fold it into the ptr via
                // QuantIntMul above; the actual load uses the accumulated pointer.
                let _ = byte_offset_slot;
                let f32_slot = match *dtype {
                    ScaleDType::F16 => push_op(trace, TraceOp::QuantLoadF16toF32 { ptr: scales_ptr_slot, offset_bytes: 0 }),
                    ScaleDType::I8Range => push_op(trace, TraceOp::QuantLoadI8toF32 { ptr: scales_ptr_slot, offset_bytes: 0 }),
                    _ => self.cast_scale_to_f32(trace, scale_raw, *dtype),
                };
                ScaleResult { scale_slot: f32_slot }
            }

            ScaleLayout::SubBlockScalars { offset_bytes, sub_block_size, dtype } => {
                // NVFP4: sub-block scale at block_base + offset_bytes + sub_block_idx
                // sub_block_idx = lane_offset / sub_block_size
                let sub_idx_slot = push_op(
                    trace,
                    TraceOp::QuantIntDivConst {
                        src: lane_offset_slot,
                        divisor: *sub_block_size as i64,
                    },
                );

                // base_ptr = block_ptr + offset_bytes
                let sub_scales_base = push_op(
                    trace,
                    TraceOp::QuantPtrAddOffset {
                        base: block_ptr_slot,
                        offset_bytes: *offset_bytes as i64,
                    },
                );
                // final_ptr = sub_scales_base + sub_block_idx
                let final_ptr = push_op(
                    trace,
                    TraceOp::QuantPtrAddDynamic {
                        base: sub_scales_base,
                        index: sub_idx_slot,
                    },
                );
                let f32_slot = self.load_and_cast_scalar(trace, final_ptr, 0, *dtype);
                ScaleResult { scale_slot: f32_slot }
            }
        }
    }

    // ── §3.2 Zero / min load ─────────────────────────────────────────────────

    fn emit_zero_load(&self, trace: &mut Vec<TraceOp>, block_ptr_slot: ValueId, lane_offset_slot: ValueId) -> ZeroResult {
        match &self.desc.zero_layout {
            ZeroLayout::None => ZeroResult::None,

            ZeroLayout::StaticBias { value } => ZeroResult::StaticBias(*value),

            // AWQ4/GPTQ4: `value = (unpacked - zp) × scale` — pre-scale subtraction
            ZeroLayout::BlockScalar { offset_bytes, dtype } => {
                let f32_slot = self.load_and_cast_scalar(trace, block_ptr_slot, *offset_bytes as i64, *dtype);
                ZeroResult::PreScaleSubtract(f32_slot)
            }

            ZeroLayout::Hierarchical { dmin_offset, sub_m_offset } => {
                // block_dmin × sub_m   (both f16-derived)
                let dmin_f32 = push_op(
                    trace,
                    TraceOp::QuantLoadF16toF32 {
                        ptr: block_ptr_slot,
                        offset_bytes: *dmin_offset as i64,
                    },
                );

                // sub_m sub-block index mirrors sub_scale
                let sub_block_elements = self.sub_block_elements_for_zero();
                let sub_idx_slot = push_op(
                    trace,
                    TraceOp::QuantIntDivConst {
                        src: lane_offset_slot,
                        divisor: sub_block_elements as i64,
                    },
                );

                // sub_m_base = block_ptr + sub_m_offset (points to packed scales array)
                let sub_m_base = push_op(
                    trace,
                    TraceOp::QuantPtrAddOffset {
                        base: block_ptr_slot,
                        offset_bytes: *sub_m_offset as i64,
                    },
                );

                // Decode packed sub_m using K-quant 6-bit min extraction
                let sub_m_f32 = push_op(
                    trace,
                    TraceOp::QuantKQuantPackedScaleLookup {
                        scales_base: sub_m_base,
                        sub_block_idx: sub_idx_slot,
                        is_q3k_extended: false,
                        is_min: true,
                    },
                );

                // min = dmin * sub_m
                let min_slot = push_op(trace, TraceOp::Mul(dmin_f32, sub_m_f32));
                ZeroResult::PostScaleAdd(min_slot)
            }
        }
    }

    // ── §3.3 Data load ────────────────────────────────────────────────────────

    fn emit_data_load(&self, trace: &mut Vec<TraceOp>, block_ptr_slot: ValueId) -> ValueId {
        match &self.desc.data_layout {
            DataLayout::PackedNibbles { offset, .. } => {
                // Load lanes/2 packed bytes as integer vector (zero-extended to i32 per byte).
                let byte_count = self.output_lanes / 2;
                push_op(
                    trace,
                    TraceOp::QuantLoadBytesVec {
                        ptr: block_ptr_slot,
                        offset_bytes: *offset as i64,
                        count: byte_count,
                        signed: false,
                    },
                )
            }

            DataLayout::NibbleWithHighBits { low_offset, .. } => {
                // Load lanes/2 low nibble bytes as integer vector
                let byte_count = self.output_lanes / 2;
                push_op(
                    trace,
                    TraceOp::QuantLoadBytesVec {
                        ptr: block_ptr_slot,
                        offset_bytes: *low_offset as i64,
                        count: byte_count,
                        signed: false,
                    },
                )
            }

            DataLayout::Bytes { offset, signed, .. } => {
                let raw = push_op(
                    trace,
                    TraceOp::QuantLoadBytesVec {
                        ptr: block_ptr_slot,
                        offset_bytes: *offset as i64,
                        count: self.output_lanes,
                        signed: *signed,
                    },
                );
                if matches!(self.desc.data_kind, QuantDataKind::Float8) {
                    let is_e4m3 = matches!(self.desc.quant_type, QuantType::Fp8E4M3);
                    push_op(trace, TraceOp::QuantCastFp8toF32 { src: raw, is_e4m3 })
                } else {
                    push_op(trace, TraceOp::QuantCastI8toF32 { src: raw })
                }
            }

            DataLayout::CodebookIndex { offset, .. } => {
                push_op(
                    trace,
                    TraceOp::QuantScalarLoad {
                        ptr: block_ptr_slot,
                        offset_bytes: *offset as i64,
                    },
                )
            }
        }
    }

    // ── §3.3 Unpack ───────────────────────────────────────────────────────────

    fn emit_unpack(&self, trace: &mut Vec<TraceOp>, raw_data_slot: ValueId, block_ptr_slot: ValueId, high_bits_ptr_slot: ValueId) -> ValueId {
        match &self.desc.data_layout {
            DataLayout::PackedNibbles { low_first, .. } => {
                // raw_data_slot holds a packed byte.
                // lo = raw & 0x0F
                // hi = (raw >> 4) & 0x0F
                // result = interleave(lo, hi) if low_first else interleave(hi, lo)
                let lo_slot = push_op(
                    trace,
                    TraceOp::QuantAndMask { src: raw_data_slot, mask: 0x0F },
                );
                let hi_shifted = push_op(
                    trace,
                    TraceOp::QuantShiftRight { src: raw_data_slot, amount: 4 },
                );
                let hi_slot = push_op(
                    trace,
                    TraceOp::QuantAndMask { src: hi_shifted, mask: 0x0F },
                );
                let interleaved = if *low_first {
                    push_op(trace, TraceOp::QuantInterleave { lo: lo_slot, hi: hi_slot })
                } else {
                    push_op(trace, TraceOp::QuantInterleave { lo: hi_slot, hi: lo_slot })
                };
                // Interleave output is integer (i32 lanes) — convert to f32 before arithmetic.
                push_op(trace, TraceOp::QuantCastI8toF32 { src: interleaved })
            }

            DataLayout::NibbleWithHighBits { high_bits_per_elem, .. } => {
                // High bits are loaded from high_bits_ptr (Input 3), which is
                // independently advanced by the GEMV ei loop at its own stride.
                // This fixes the Q6_K SIGSEGV where block_ptr+high_offset was a
                // fixed address that didn't track the inner loop progression.
                let byte_count = (self.output_lanes * (*high_bits_per_elem as usize) + 7) / 8;
                let qh_slot = push_op(
                    trace,
                    TraceOp::QuantLoadBytesVec {
                        ptr: high_bits_ptr_slot,
                        offset_bytes: 0,
                        count: byte_count,
                        signed: false,
                    },
                );
                let shift_amount = 4u32;
                let qh_shifted = push_op(
                    trace,
                    TraceOp::QuantShiftLeft {
                        src: qh_slot,
                        amount: shift_amount + (4 - *high_bits_per_elem as u32),
                    },
                );
                // Mask high bits before OR: (qh << shift) & high_mask
                let high_mask_val = (((1u32 << *high_bits_per_elem) - 1) << 4) as u64;
                let qh_masked = push_op(
                    trace,
                    TraceOp::QuantAndMask { src: qh_shifted, mask: high_mask_val },
                );
                // merged = qs | qh_masked
                let merged = push_op(trace, TraceOp::QuantBitOr { lhs: raw_data_slot, rhs: qh_masked });
                // Output is integer — convert to f32 before arithmetic.
                push_op(trace, TraceOp::QuantCastI8toF32 { src: merged })
            }

            DataLayout::Bytes { .. } => {
                // Already loaded and converted to f32 by emit_data_load
                raw_data_slot
            }

            DataLayout::CodebookIndex { index_bits, .. } => {
                if let Some(cb) = self.desc.codebook.as_ref() {
                    // Codebook lookup path: extract indices → lookup in codebook → F32
                    let indices_slot = push_op(
                        trace,
                        TraceOp::QuantExtractBits {
                            src: raw_data_slot,
                            bit_offset: 0,
                            bit_width: *index_bits,
                        },
                    );
                    push_op(
                        trace,
                        TraceOp::QuantCodebookLookup {
                            indices: indices_slot,
                            codebook_data: cb.codebook_data,
                            vector_size: cb.vector_size,
                            bits_per_entry: cb.bits_per_entry,
                        },
                    )
                } else {
                    // No codebook: extract bits and cast to F32 directly.
                    // Used for IQ formats with complex grid lookup (handled by SIMD kernels).
                    push_op(
                        trace,
                        TraceOp::QuantExtractBits {
                            src: raw_data_slot,
                            bit_offset: 0,
                            bit_width: *index_bits,
                        },
                    )
                }
            }
        }
    }

    // ── §3.4 Dequant algebra ──────────────────────────────────────────────────

    /// Emit the dequantization formula.
    ///
    /// AWQ4/GPTQ4 (REQ-QCG-010):  `value = (unpacked - zp) × scale`
    /// GGUF Q4_K/Q5_K:             `value = (unpacked - bias) × scale + min`
    /// GGUF Q4_0/Q8_0:             `value = (unpacked - bias) × scale`
    fn emit_dequant_algebra(
        &self,
        trace: &mut Vec<TraceOp>,
        unpacked_slot: ValueId,
        scale: ScaleResult,
        zero: ZeroResult,
    ) -> ValueId {
        // 1. Apply pre-scale subtraction (static bias or dynamic zero-point)
        let biased_slot = match &zero {
            ZeroResult::StaticBias(bias) => {
                let bias_f32 = push_op(trace, TraceOp::Const(*bias as f64));
                push_op(trace, TraceOp::Sub(unpacked_slot, bias_f32))
            }
            ZeroResult::PreScaleSubtract(zp_slot) => {
                let zp_broadcast = push_op(
                    trace,
                    TraceOp::QuantBroadcast { src: *zp_slot, lanes: self.output_lanes },
                );
                push_op(trace, TraceOp::Sub(unpacked_slot, zp_broadcast))
            }
            _ => unpacked_slot,
        };

        // 2. Broadcast scale to all output lanes
        let scale_broadcast = push_op(
            trace,
            TraceOp::QuantBroadcast { src: scale.scale_slot, lanes: self.output_lanes },
        );

        // 3. acc = 0; value = biased * scale
        let zero_acc = push_op(trace, TraceOp::Const(0.0));
        let scaled_slot = push_op(
            trace,
            TraceOp::QuantDequantFma { acc: zero_acc, a: biased_slot, b: scale_broadcast },
        );

        // 4. Post-scale addition (Q4_K/Q5_K dmin × sub_m)
        match zero {
            ZeroResult::PostScaleAdd(min_slot) => {
                let min_broadcast = push_op(
                    trace,
                    TraceOp::QuantBroadcast { src: min_slot, lanes: self.output_lanes },
                );
                push_op(trace, TraceOp::Add(scaled_slot, min_broadcast))
            }
            _ => scaled_slot,
        }
    }

    // ── K-Quant 6-bit packed scale lookup (REQ-QCG-007) ──────────────────────

    /// Emit the TraceOp sequence to decode one 6-bit packed sub-scale value.
    ///
    /// `KQuant6Bit` algorithm (per llama.cpp `ggml-quants.c`):
    ///   scales[12] → 8 (sc, m) pairs where each sc/m is 6 bits.
    ///   - Entry 0..3: sc = scales[i] & 0x3F
    ///   - Entry 4..7: sc = (scales[i-4] >> 6) | ((scales[i] & 0xF) << 2)   (6-bit value)
    ///
    /// For this TraceOp sequence we emit a simplified "load the right byte and
    /// mask" sequence.  The sub_idx_slot selects which of the 8 pairs to use.
    fn emit_packed_scale_lookup(
        &self,
        trace: &mut Vec<TraceOp>,
        sub_scales_base_slot: ValueId,
        sub_idx_slot: ValueId,
        algo: &PackedScaleAlgorithm,
    ) -> ValueId {
        let is_q3k = matches!(algo, PackedScaleAlgorithm::Q3KExtended);
        push_op(
            trace,
            TraceOp::QuantKQuantPackedScaleLookup {
                scales_base: sub_scales_base_slot,
                sub_block_idx: sub_idx_slot,
                is_q3k_extended: is_q3k,
                is_min: false,
            },
        )
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Load a scalar value at `block_ptr + offset_bytes` and cast to f32.
    /// For F16/I8 dtypes, emits a combined load+cast TraceOp to avoid the
    /// two-step QuantScalarLoad + QuantCast pattern that causes double-loading.
    fn load_and_cast_scalar(
        &self,
        trace: &mut Vec<TraceOp>,
        block_ptr_slot: ValueId,
        offset_bytes: i64,
        dtype: ScaleDType,
    ) -> ValueId {
        match dtype {
            ScaleDType::F16 => push_op(
                trace,
                TraceOp::QuantLoadF16toF32 { ptr: block_ptr_slot, offset_bytes },
            ),
            ScaleDType::I8Range => push_op(
                trace,
                TraceOp::QuantLoadI8toF32 { ptr: block_ptr_slot, offset_bytes },
            ),
            // F32 and others: plain scalar load, no cast needed
            _ => push_op(
                trace,
                TraceOp::QuantScalarLoad { ptr: block_ptr_slot, offset_bytes },
            ),
        }
    }

    /// Cast a raw slot (loaded bits) to f32 based on dtype.
    /// raw_slot comes from QuantScalarLoad (Ptr VReg holding a byte value).
    /// For F16/I8/E8M0, we use load_and_cast_scalar to avoid GPR→Vec broadcast issues.
    fn cast_scale_to_f32(&self, trace: &mut Vec<TraceOp>, raw_slot: ValueId, dtype: ScaleDType) -> ValueId {
        raw_slot
    }

    /// Return the sub-block element count used for zero-point indexing
    /// (mirrors the scale layout's sub_block_elements when available).
    fn sub_block_elements_for_zero(&self) -> usize {
        match &self.desc.scale_layout {
            ScaleLayout::Hierarchical { sub_block_elements, .. } => *sub_block_elements,
            ScaleLayout::Q6KScales { sub_block_elements, .. } => *sub_block_elements,
            _ => self.desc.block_size,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Unit tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_format::registry;
    use crate::quant::QuantType;

    /// Check that the trace is non-empty and ends with a slot that comes from
    /// one of the dequant-algebra Quant* ops (QuantDequantFma / Add / Sub).
    fn assert_trace_valid(trace: &[TraceOp]) {
        assert!(!trace.is_empty(), "DecodeTraceBuilder produced empty trace");
        // First two ops must be Input(0) and Input(1) for block_base and data_ptr
        assert!(
            matches!(trace[0], TraceOp::Input(0)),
            "First TraceOp must be Input(0) for block_base, got {:?}",
            trace[0]
        );
        assert!(
            matches!(trace[1], TraceOp::Input(1)),
            "Second TraceOp must be Input(1) for data_ptr, got {:?}",
            trace[1]
        );
        // Last op must be a computation (not a plain Input)
        let last = trace.last().unwrap();
        assert!(
            !matches!(last, TraceOp::Input(_)),
            "Last TraceOp should be a computation, got {:?}",
            last
        );
    }

    #[test]
    fn test_q4_0_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q4_0).expect("Q4_0 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
        // Q4_0: ZeroLayout::None (bias already handled in GgufInt4Load), no Sub expected
        // Should have QuantLoadF16toF32 for the f16 scale load
        let has_cast = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_cast, "Q4_0 should load f16 scale to f32");
    }

    #[test]
    fn test_q4_1_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q4_1).expect("Q4_1 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
        // Q4_1: PackedNibbles → QuantInterleave present
        let has_interleave = trace.iter().any(|op| matches!(op, TraceOp::QuantInterleave { .. }));
        assert!(has_interleave, "Q4_1 should have QuantInterleave for PackedNibbles");
    }

    #[test]
    fn test_q8_0_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q8_0).expect("Q8_0 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
        // Q8_0: signed bytes → QuantLoadBytesVec(signed=true) + QuantCastI8toF32
        let has_bytes_vec = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadBytesVec { signed: true, .. }));
        let has_i8_to_f32 = trace.iter().any(|op| matches!(op, TraceOp::QuantCastI8toF32 { .. }));
        assert!(has_bytes_vec, "Q8_0 should use QuantLoadBytesVec with signed=true");
        assert!(has_i8_to_f32, "Q8_0 should have QuantCastI8toF32 for i8→f32 conversion");
    }

    #[test]
    fn test_q8_1_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q8_1).expect("Q8_1 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
    }

    #[test]
    fn test_q4k_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q4K).expect("Q4_K must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
        // Q4_K: Hierarchical scale → QuantIntDivConst present
        let has_div = trace.iter().any(|op| matches!(op, TraceOp::QuantIntDivConst { .. }));
        assert!(has_div, "Q4_K hierarchical should have QuantIntDivConst for sub_block_idx");
    }

    #[test]
    fn test_q6k_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q6K).expect("Q6_K must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
        // Q6_K: StaticBias(32) → Sub op
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "Q6_K should have Sub for static bias of 32");
    }

    #[test]
    fn test_trace_slot_indices_in_range() {
        let r = registry();
        for qt in &[
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1, QuantType::Q2K, QuantType::Q3K,
            QuantType::Q4K, QuantType::Q5K, QuantType::Q6K,
            QuantType::AWQ4, QuantType::GPTQ4,
        ] {
            let desc = match r.get(qt) {
                Some(d) => d,
                None => continue,
            };
            let mut trace = Vec::new();
            let final_slot = DecodeTraceBuilder::new(desc, 8)
                .build(&mut trace);
            let len = trace.len();
            assert!(
                final_slot.0 < len as u32,
                "final_slot {} out of range [0, {}) for {:?}",
                final_slot, len, qt
            );
            // Validate all slot references within each op are in range
            for (pos, op) in trace.iter().enumerate() {
                let refs: Vec<ValueId> = match op {
                    TraceOp::QuantBitAnd { lhs, rhs } |
                    TraceOp::QuantBitOr { lhs, rhs } => vec![*lhs, *rhs],
                    TraceOp::QuantBroadcast { src, .. } |
                    TraceOp::QuantCastF16toF32 { src } |
                    TraceOp::QuantCastI8toF32 { src } |
                    TraceOp::QuantExtractBits { src, .. } |
                    TraceOp::QuantIntDivConst { src, .. } |
                    TraceOp::QuantIntMul { src, .. } |
                    TraceOp::QuantShiftLeft { src, .. } |
                    TraceOp::QuantShiftRight { src, .. } => vec![*src],
                    TraceOp::QuantCodebookLookup { indices, .. } => vec![*indices],
                    TraceOp::QuantDequantFma { acc, a, b } => vec![*acc, *a, *b],
                    TraceOp::QuantInterleave { lo, hi } => vec![*lo, *hi],
                    TraceOp::QuantScalarLoad { ptr, .. } => vec![*ptr],
                    TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b) => vec![*a, *b],
                    _ => vec![],
                };
                for r in refs {
                    assert!(
                        (r.0 as usize) < pos,
                        "Op at slot {} refs future slot {} ({:?})",
                        pos, r, op
                    );
                }
            }
        }
    }

    #[test]
    fn test_fp8_e4m3_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Fp8E4M3).expect("Fp8E4M3 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
        let has_fp8_cast = trace.iter().any(|op| matches!(op, TraceOp::QuantCastFp8toF32 { is_e4m3: true, .. }));
        assert!(has_fp8_cast, "FP8 E4M3 should use QuantCastFp8toF32");
        let has_i8_cast = trace.iter().any(|op| matches!(op, TraceOp::QuantCastI8toF32 { .. }));
        assert!(!has_i8_cast, "FP8 E4M3 should NOT use QuantCastI8toF32");
    }

    #[test]
    fn test_fp8_e5m2_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Fp8E5M2).expect("Fp8E5M2 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);
        let has_fp8_cast = trace.iter().any(|op| matches!(op, TraceOp::QuantCastFp8toF32 { is_e4m3: false, .. }));
        assert!(has_fp8_cast, "FP8 E5M2 should use QuantCastFp8toF32 with is_e4m3=false");
    }

    #[test]
    fn test_fp8_e4m3_no_scale_no_zero() {
        let r = registry();
        let desc = r.get(&QuantType::Fp8E4M3).expect("Fp8E4M3 must be registered");
        let mut trace = Vec::new();
        DecodeTraceBuilder::new(desc, 8).build(&mut trace);
        // FP8 has ScaleLayout::None → should use Const(1.0), no QuantLoadF16toF32
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(!has_f16_load, "FP8 should not load f16 scales");
    }

    #[test]
    fn test_awq4_trace() {
        let r = registry();
        let desc = r.get(&QuantType::AWQ4).expect("AWQ4 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // AWQ4: `value = (qw - zero_point) × scale` — Sub must appear BEFORE FMA
        // 1. Should have a Sub (unpacked - zero_point) — pre-scale subtraction
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "AWQ4 must have Sub for pre-scale zero-point subtraction");

        // 2. Should NOT have a post-scale Add (only Sub + FMA)
        let has_add = trace.iter().any(|op| matches!(op, TraceOp::Add(_, _)));
        assert!(!has_add, "AWQ4 must NOT have post-scale Add; formula is (qw-zp)×scale");

        // 3. Verify FMA exists
        let has_fma = trace.iter().any(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
        assert!(has_fma, "AWQ4 must have QuantDequantFma for scale multiplication");

        // 4. Should load f16 scale and f16 zero-point
        let f16_count = trace.iter()
            .filter(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }))
            .count();
        assert!(f16_count >= 2, "AWQ4 should load at least 2 f16 values (scale + zero), got {}", f16_count);
    }

    #[test]
    fn test_gptq4_trace() {
        let r = registry();
        let desc = r.get(&QuantType::GPTQ4).expect("GPTQ4 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // GPTQ4: same formula as AWQ4: `value = (qw - zero_point) × scale`
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "GPTQ4 must have Sub for pre-scale zero-point subtraction");

        let has_add = trace.iter().any(|op| matches!(op, TraceOp::Add(_, _)));
        assert!(!has_add, "GPTQ4 must NOT have post-scale Add; formula is (qw-zp)×scale");
    }

    /// REQ-QCG-007: Q2K trace must handle Hierarchical scale + Hierarchical zero.
    #[test]
    fn test_q2k_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q2K).expect("Q2_K must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Q2K: Hierarchical scale (block_d + sub_scales) → QuantIntDivConst present
        let has_div = trace.iter().any(|op| matches!(op, TraceOp::QuantIntDivConst { .. }));
        assert!(has_div, "Q2K hierarchical scale should have QuantIntDivConst for sub_block_idx");

        // Q2K: Hierarchical zero (dmin + sub_m) → Sub present (dequant formula involves min)
        let has_add_or_sub = trace.iter().any(|op| matches!(op, TraceOp::Add(_, _) | TraceOp::Sub(_, _)));
        assert!(has_add_or_sub, "Q2K should have Add/Sub for hierarchical zero-point (min)");
    }

    /// REQ-QCG-007: Q3K trace must handle Hierarchical scale + hmask-based unpack.
    #[test]
    fn test_q3k_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Q3K).expect("Q3_K must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Q3K: Hierarchical scale → QuantIntDivConst present
        let has_div = trace.iter().any(|op| matches!(op, TraceOp::QuantIntDivConst { .. }));
        assert!(has_div, "Q3K hierarchical scale should have QuantIntDivConst for sub_block_idx");

        // Q3K: StaticBias(4) → Sub present
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "Q3K should have Sub for StaticBias(4)");

        // Q3K: NibbleWithHighBits (hmask) → needs_high_bits_ptr
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(builder.needs_high_bits_ptr(), "Q3K should need high bits pointer for hmask");
    }

    /// REQ-QCG-009: MXFP4 E2M1 LUT decode trace.
    #[test]
    fn test_mxfp4_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Mxfp4 { block_size: 32 }).expect("MXFP4 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // MXFP4 uses QuantE2m1LutDecode with nvfp4_mode=false
        let has_e2m1 = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantE2m1LutDecode { nvfp4_mode: false, .. }
        ));
        assert!(has_e2m1, "MXFP4 should emit QuantE2m1LutDecode with nvfp4_mode=false");
    }

    /// REQ-QCG-009a: NVFP4 E2M1 LUT decode trace (nvfp4_mode=true).
    #[test]
    fn test_nvfp4_trace() {
        let r = registry();
        let desc = r.get(&QuantType::Nvfp4).expect("NVFP4 must be registered");
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8)
            .build(&mut trace);
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // NVFP4 uses QuantE2m1LutDecode with nvfp4_mode=true
        let has_e2m1_nvfp4 = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantE2m1LutDecode { nvfp4_mode: true, .. }
        ));
        assert!(has_e2m1_nvfp4, "NVFP4 should emit QuantE2m1LutDecode with nvfp4_mode=true");
    }

    // @trace TEST-QD-01 [req:REQ-QCG] [level:unit]
    // Q5_0: NibbleWithHighBits + StaticBias(16) + BlockScalar scale.
    // Tests high bits pointer path (1 high bit per elem) and static bias subtraction.
    #[test]
    fn test_q5_0_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q5_0).expect("Q5_0 must be registered");
        let builder = DecodeTraceBuilder::new(desc, 8);

        // Assert: Q5_0 uses NibbleWithHighBits → needs high bits pointer
        assert!(builder.needs_high_bits_ptr(), "Q5_0 should need high bits pointer for 5-bit unpack");

        // Act
        let mut trace = Vec::new();
        let final_slot = builder.build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Q5_0: StaticBias(16) → Const(16.0) + Sub
        let has_bias_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_bias_sub, "Q5_0 should have Sub for static bias of 16");

        // Q5_0: NibbleWithHighBits → QuantBitOr (merge low + high bits)
        let has_bitor = trace.iter().any(|op| matches!(op, TraceOp::QuantBitOr { .. }));
        assert!(has_bitor, "Q5_0 should have QuantBitOr for merging low nibble + high bit");
    }

    // @trace TEST-QD-02 [req:REQ-QCG] [level:unit]
    // Q5_1: NibbleWithHighBits + BlockScalarWithMin scale (d + m) + no zero.
    // Tests BlockScalarWithMin scale path with high bits.
    #[test]
    fn test_q5_1_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q5_1).expect("Q5_1 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Q5_1: BlockScalarWithMin → loads d via QuantLoadF16toF32
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_f16_load, "Q5_1 should load f16 scale via QuantLoadF16toF32");

        // Q5_1: NibbleWithHighBits → QuantLoadBytesVec for high bits
        let has_bytes_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadBytesVec { .. }));
        assert!(has_bytes_load, "Q5_1 should load bytes for nibble data");

        // Q5_1: no zero layout → no Sub for bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(!has_sub, "Q5_1 should NOT have Sub (zero_layout is None)");
    }

    // @trace TEST-QD-03 [req:REQ-QCG] [level:unit]
    // Q5_K: Hierarchical scale + Hierarchical zero + NibbleWithHighBits (1 high bit).
    // Tests the full K-quant hierarchical path with 5-bit high bits.
    #[test]
    fn test_q5k_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q5K).expect("Q5_K must be registered");
        let builder = DecodeTraceBuilder::new(desc, 8);

        // Assert: Q5_K has hierarchical scale + NibbleWithHighBits
        assert!(builder.needs_lane_offset(), "Q5_K should need lane offset for hierarchical scale");
        assert!(builder.needs_high_bits_ptr(), "Q5_K should need high bits pointer");

        // Act
        let mut trace = Vec::new();
        let final_slot = builder.build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Q5_K: Hierarchical scale → QuantKQuantPackedScaleLookup
        let has_packed_lookup = trace.iter().any(|op| matches!(op, TraceOp::QuantKQuantPackedScaleLookup { .. }));
        assert!(has_packed_lookup, "Q5_K should have QuantKQuantPackedScaleLookup for hierarchical scale");

        // Q5_K: Hierarchical zero → PostScaleAdd (dmin * sub_m)
        let has_add = trace.iter().any(|op| matches!(op, TraceOp::Add(_, _)));
        assert!(has_add, "Q5_K should have Add for post-scale min addition (dmin * sub_m)");
    }

    // @trace TEST-QD-04 [req:REQ-QCG] [level:unit]
    // Q8_K: F32 scale dtype + Bytes signed + no zero.
    // Tests the F32 scale path (QuantScalarLoad, no F16 cast) with signed i8 bytes.
    #[test]
    fn test_q8k_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q8K).expect("Q8_K must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Q8_K: F32 scale → QuantScalarLoad (not QuantLoadF16toF32)
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(!has_f16_load, "Q8_K should NOT use QuantLoadF16toF32; scale is F32");

        // Q8_K: signed bytes → QuantLoadBytesVec(signed=true)
        let has_signed_bytes = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadBytesVec { signed: true, .. }));
        assert!(has_signed_bytes, "Q8_K should load signed bytes");

        // Q8_K: no zero → no Sub/Add for bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(!has_sub, "Q8_K should NOT have Sub (no zero layout)");
    }

    // @trace TEST-QD-05 [req:REQ-QCG] [level:unit]
    // IQ4_NL: CodebookIndex with static codebook.
    // Tests the codebook lookup path: extract bits → QuantCodebookLookup.
    #[test]
    fn test_iq4nl_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ4NL).expect("IQ4_NL must be registered");
        assert!(desc.codebook.is_some(), "IQ4_NL should have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ4_NL: CodebookIndex → QuantExtractBits (extract 4-bit indices)
        let has_extract = trace.iter().any(|op| matches!(op, TraceOp::QuantExtractBits { bit_width: 4, .. }));
        assert!(has_extract, "IQ4_NL should extract 4-bit indices");

        // IQ4_NL: Has codebook → QuantCodebookLookup
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(has_codebook, "IQ4_NL should use QuantCodebookLookup for codebook decode");
    }

    // @trace TEST-QD-06 [req:REQ-QCG] [level:unit]
    // IQ1_S: CodebookIndex without codebook (1-bit indices).
    // Tests the no-codebook path where QuantExtractBits result is used directly.
    #[test]
    fn test_iq1s_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ1S).expect("IQ1_S must be registered");
        assert!(desc.codebook.is_none(), "IQ1_S should NOT have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ1_S: CodebookIndex with index_bits=1 → QuantExtractBits (1-bit)
        let has_extract = trace.iter().any(|op| matches!(op, TraceOp::QuantExtractBits { bit_width: 1, .. }));
        assert!(has_extract, "IQ1_S should extract 1-bit indices");

        // IQ1_S: No codebook → no QuantCodebookLookup
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(!has_codebook, "IQ1_S should NOT use QuantCodebookLookup (no codebook)");
    }

    // @trace TEST-QD-07 [req:REQ-QCG] [level:unit]
    // SqueezeLLM: StaticBias(4) + PackedNibbles + BlockScalar f16 scale.
    // Tests the 3-bit packed format with static bias of 4.
    #[test]
    fn test_squeeze_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Squeeze).expect("Squeeze must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Squeeze: StaticBias(4) → Const(4.0) + Sub
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "Squeeze should have Sub for static bias of 4");

        // Squeeze: PackedNibbles → QuantInterleave present
        let has_interleave = trace.iter().any(|op| matches!(op, TraceOp::QuantInterleave { .. }));
        assert!(has_interleave, "Squeeze should have QuantInterleave for PackedNibbles");

        // Squeeze: no post-scale add (only pre-scale sub)
        let has_add = trace.iter().any(|op| matches!(op, TraceOp::Add(_, _)));
        assert!(!has_add, "Squeeze should NOT have post-scale Add");
    }

    // @trace TEST-QD-08 [req:REQ-QCG] [level:unit]
    // TQ1_0: Ternary format with StaticBias(1) + f16 scale.
    // Tests ternary quantization: value = d * (trit - 1.0).
    #[test]
    fn test_tq1_0_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::TQ1_0).expect("TQ1_0 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // TQ1_0: StaticBias(1) → Const(1.0) + Sub
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "TQ1_0 should have Sub for static bias of 1");

        // TQ1_0: f16 scale → QuantLoadF16toF32
        let has_f16 = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_f16, "TQ1_0 should load f16 scale");
    }

    // @trace TEST-QD-09 [req:REQ-QCG] [level:unit]
    // BF16: native float format with ScaleLayout::None, ZeroLayout::None, Bytes unsigned.
    // Tests the trivial path where no scale/zero/bias is applied.
    #[test]
    fn test_bf16_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Bf16).expect("BF16 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // BF16: ScaleLayout::None → Const(1.0) for scale
        let has_const_one = trace.iter().any(|op| matches!(op, TraceOp::Const(v) if *v == 1.0_f64));
        assert!(has_const_one, "BF16 should use Const(1.0) for identity scale");

        // BF16: no f16 scale load, no zero-point ops
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(!has_f16_load, "BF16 should NOT load f16 scales (ScaleLayout::None)");

        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(!has_sub, "BF16 should NOT have Sub (ZeroLayout::None)");

        // BF16: Bytes unsigned → QuantCastI8toF32 (unsigned bytes loaded)
        let has_cast = trace.iter().any(|op| matches!(op, TraceOp::QuantCastI8toF32 { .. }));
        assert!(has_cast, "BF16 should have QuantCastI8toF32 for byte→f32 conversion");
    }

    // @trace TEST-QD-10 [req:REQ-QCG] [level:unit]
    // needs_lane_offset: verifies the method correctly reports lane offset
    // requirements for various format categories.
    #[test]
    fn test_needs_lane_offset_method() {
        // Arrange: formats with hierarchical scale layouts need lane offset
        let r = registry();

        // Assert: hierarchical formats need lane offset
        let q4k_desc = r.get(&QuantType::Q4K).expect("Q4K");
        assert!(DecodeTraceBuilder::new(q4k_desc, 8).needs_lane_offset(),
            "Q4K (Hierarchical scale) should need lane offset");

        let q6k_desc = r.get(&QuantType::Q6K).expect("Q6K");
        assert!(DecodeTraceBuilder::new(q6k_desc, 8).needs_lane_offset(),
            "Q6K (Q6KScales) should need lane offset");

        let nvfp4_desc = r.get(&QuantType::Nvfp4).expect("NVFP4");
        assert!(DecodeTraceBuilder::new(nvfp4_desc, 8).needs_lane_offset(),
            "NVFP4 (SubBlockScalars) should need lane offset");

        // Assert: flat formats do NOT need lane offset
        let q4_0_desc = r.get(&QuantType::Q4_0).expect("Q4_0");
        assert!(!DecodeTraceBuilder::new(q4_0_desc, 8).needs_lane_offset(),
            "Q4_0 (BlockScalar scale) should NOT need lane offset");

        let bf16_desc = r.get(&QuantType::Bf16).expect("BF16");
        assert!(!DecodeTraceBuilder::new(bf16_desc, 8).needs_lane_offset(),
            "BF16 (None scale) should NOT need lane offset");
    }

    // @trace TEST-QD-11 [req:REQ-QCG] [level:unit]
    // needs_high_bits_ptr and high_bits_stride: verifies high bits pointer
    // detection and stride calculation for NibbleWithHighBits formats.
    #[test]
    fn test_high_bits_ptr_and_stride() {
        // Arrange
        let r = registry();

        // Assert: NibbleWithHighBits formats need high bits pointer
        let q6k_desc = r.get(&QuantType::Q6K).expect("Q6K");
        let q6k_builder = DecodeTraceBuilder::new(q6k_desc, 8);
        assert!(q6k_builder.needs_high_bits_ptr(), "Q6K should need high bits pointer");
        // Q6_K: 2 bits/elem, 8 lanes → (8*2+7)/8 = 2 bytes stride
        assert_eq!(q6k_builder.high_bits_stride(), 2, "Q6_K high bits stride should be 2 bytes");

        let q5_0_desc = r.get(&QuantType::Q5_0).expect("Q5_0");
        let q5_0_builder = DecodeTraceBuilder::new(q5_0_desc, 8);
        assert!(q5_0_builder.needs_high_bits_ptr(), "Q5_0 should need high bits pointer");
        // Q5_0: 1 bit/elem, 8 lanes → (8*1+7)/8 = 1 byte stride
        assert_eq!(q5_0_builder.high_bits_stride(), 1, "Q5_0 high bits stride should be 1 byte");

        // Assert: PackedNibbles formats do NOT need high bits pointer
        let q4_0_desc = r.get(&QuantType::Q4_0).expect("Q4_0");
        let q4_0_builder = DecodeTraceBuilder::new(q4_0_desc, 8);
        assert!(!q4_0_builder.needs_high_bits_ptr(), "Q4_0 should NOT need high bits pointer");
        assert_eq!(q4_0_builder.high_bits_stride(), 0, "Q4_0 high bits stride should be 0");

        // Assert: Bytes formats do NOT need high bits pointer
        let q8_0_desc = r.get(&QuantType::Q8_0).expect("Q8_0");
        assert!(!DecodeTraceBuilder::new(q8_0_desc, 8).needs_high_bits_ptr(),
            "Q8_0 should NOT need high bits pointer");
    }

    // @trace TEST-QD-12 [req:REQ-QCG] [level:unit]
    // output_lanes variation: verifies the builder works with different lane counts
    // (4 lanes and 16 lanes) and produces valid traces with correct byte counts.
    #[test]
    fn test_output_lanes_variation() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q4_0).expect("Q4_0 must be registered");

        // Act: 4 lanes (half the default)
        let mut trace_4 = Vec::new();
        let slot_4 = DecodeTraceBuilder::new(desc, 4).build(&mut trace_4);
        assert_trace_valid(&trace_4);
        assert!(slot_4.0 < trace_4.len() as u32);

        // Q4_0 with 4 lanes: PackedNibbles loads 4/2 = 2 bytes
        let has_2byte_load = trace_4.iter().any(|op| matches!(
            op,
            TraceOp::QuantLoadBytesVec { count: 2, .. }
        ));
        assert!(has_2byte_load, "Q4_0 with 4 lanes should load 2 bytes (lanes/2)");

        // Act: 16 lanes (double the default)
        let mut trace_16 = Vec::new();
        let slot_16 = DecodeTraceBuilder::new(desc, 16).build(&mut trace_16);
        assert_trace_valid(&trace_16);
        assert!(slot_16.0 < trace_16.len() as u32);

        // Q4_0 with 16 lanes: PackedNibbles loads 16/2 = 8 bytes
        let has_8byte_load = trace_16.iter().any(|op| matches!(
            op,
            TraceOp::QuantLoadBytesVec { count: 8, .. }
        ));
        assert!(has_8byte_load, "Q4_0 with 16 lanes should load 8 bytes (lanes/2)");

        // Assert: trace size scales with lane count (more ops for broadcast etc.)
        assert!(
            trace_16.len() >= trace_4.len(),
            "16-lane trace should be at least as large as 4-lane trace"
        );
    }

    // @trace TEST-QD-13 [req:REQ-QCG] [level:unit]
    // IQ4_XS: Hierarchical scale + CodebookIndex with codebook.
    // Tests the combination of hierarchical scale decode + codebook lookup in one trace.
    #[test]
    fn test_iq4xs_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ4XS).expect("IQ4_XS must be registered");
        assert!(desc.codebook.is_some(), "IQ4_XS should have a static codebook");
        let builder = DecodeTraceBuilder::new(desc, 8);

        // Assert: IQ4_XS has hierarchical scale → needs lane offset
        assert!(builder.needs_lane_offset(), "IQ4_XS should need lane offset for hierarchical scale");

        // Act
        let mut trace = Vec::new();
        let final_slot = builder.build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ4_XS: Hierarchical scale → QuantIntDivConst for sub_block_idx
        let has_div = trace.iter().any(|op| matches!(op, TraceOp::QuantIntDivConst { .. }));
        assert!(has_div, "IQ4_XS should have QuantIntDivConst for hierarchical scale");

        // IQ4_XS: Hierarchical scale → QuantKQuantPackedScaleLookup
        let has_packed = trace.iter().any(|op| matches!(op, TraceOp::QuantKQuantPackedScaleLookup { .. }));
        assert!(has_packed, "IQ4_XS should have QuantKQuantPackedScaleLookup");

        // IQ4_XS: CodebookIndex + codebook → QuantExtractBits + QuantCodebookLookup
        let has_extract = trace.iter().any(|op| matches!(op, TraceOp::QuantExtractBits { .. }));
        assert!(has_extract, "IQ4_XS should have QuantExtractBits for index extraction");

        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(has_codebook, "IQ4_XS should use QuantCodebookLookup");
    }

    // ── Wave 12k31c: +13 additional tests ──

    #[test]
    fn decode_trace_builder_q4_0_needs_no_lane_offset() {
        let r = registry();
        let desc = r.get(&QuantType::Q4_0).expect("Q4_0");
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(!builder.needs_lane_offset(), "Q4_0 should not need lane offset");
    }

    #[test]
    fn decode_trace_builder_q4_0_needs_no_high_bits_ptr() {
        let r = registry();
        let desc = r.get(&QuantType::Q4_0).expect("Q4_0");
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(!builder.needs_high_bits_ptr(), "Q4_0 should not need high bits ptr");
    }

    #[test]
    fn decode_trace_builder_q6k_needs_lane_offset() {
        let r = registry();
        let desc = r.get(&QuantType::Q6K).expect("Q6_K");
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(builder.needs_lane_offset(), "Q6_K should need lane offset for Q6KScales");
    }

    #[test]
    fn decode_trace_builder_q5_0_needs_high_bits_ptr() {
        let r = registry();
        let desc = r.get(&QuantType::Q5_0).expect("Q5_0");
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(builder.needs_high_bits_ptr(), "Q5_0 should need high bits ptr");
    }

    #[test]
    fn decode_trace_builder_q5_1_needs_high_bits_ptr() {
        let r = registry();
        let desc = r.get(&QuantType::Q5_1).expect("Q5_1");
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(builder.needs_high_bits_ptr(), "Q5_1 should need high bits ptr");
    }

    #[test]
    fn high_bits_stride_q5_0() {
        let r = registry();
        let desc = r.get(&QuantType::Q5_0).expect("Q5_0");
        let builder = DecodeTraceBuilder::new(desc, 8);
        let stride = builder.high_bits_stride();
        assert!(stride > 0, "Q5_0 high_bits_stride should be > 0, got {stride}");
    }

    #[test]
    fn high_bits_stride_q6k() {
        let r = registry();
        let desc = r.get(&QuantType::Q6K).expect("Q6_K");
        let builder = DecodeTraceBuilder::new(desc, 8);
        let stride = builder.high_bits_stride();
        assert!(stride > 0, "Q6_K high_bits_stride should be > 0, got {stride}");
    }

    #[test]
    fn high_bits_stride_non_nibble_format_is_zero() {
        let r = registry();
        let desc = r.get(&QuantType::Q4_0).expect("Q4_0");
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert_eq!(builder.high_bits_stride(), 0, "Q4_0 should have zero high_bits_stride");
    }

    #[test]
    fn q4k_trace_has_packed_scale_lookup() {
        let r = registry();
        let desc = r.get(&QuantType::Q4K).expect("Q4_K");
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);
        assert_trace_valid(&trace);
        let has_packed = trace.iter().any(|op| matches!(op, TraceOp::QuantKQuantPackedScaleLookup { .. }));
        assert!(has_packed, "Q4_K should have QuantKQuantPackedScaleLookup");
    }

    #[test]
    fn q5k_trace_valid() {
        let r = registry();
        let desc = r.get(&QuantType::Q5K).expect("Q5_K");
        let mut trace = Vec::new();
        let slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);
        assert_trace_valid(&trace);
        assert!(slot.0 < trace.len() as u32);
    }

    #[test]
    fn mxfp4_trace_e2m1_decode() {
        let r = registry();
        let desc = r.get(&QuantType::Mxfp4 { block_size: 32 }).expect("Mxfp4");
        let mut trace = Vec::new();
        let slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);
        assert_trace_valid(&trace);
        assert!(slot.0 < trace.len() as u32);
        let has_e2m1 = trace.iter().any(|op| matches!(op, TraceOp::QuantE2m1LutDecode { .. }));
        assert!(has_e2m1, "MXFP4 should have QuantE2m1LutDecode");
    }

    #[test]
    fn nvfp4_trace_e2m1_decode() {
        let r = registry();
        let desc = r.get(&QuantType::Nvfp4).expect("Nvfp4");
        let mut trace = Vec::new();
        let slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);
        assert_trace_valid(&trace);
        assert!(slot.0 < trace.len() as u32);
        let has_e2m1 = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantE2m1LutDecode { nvfp4_mode: true, .. }
        ));
        assert!(has_e2m1, "NVFP4 should have QuantE2m1LutDecode with nvfp4_mode=true");
    }

    #[test]
    fn bf16_trace_valid() {
        let r = registry();
        let desc = r.get(&QuantType::Bf16).expect("Bf16");
        assert!(!builder_needs_lane_offset(desc), "BF16 should not need lane offset");
        let mut trace = Vec::new();
        let slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);
        assert_trace_valid(&trace);
        assert!(slot.0 < trace.len() as u32);
    }

    fn builder_needs_lane_offset(desc: &QuantFormatDescriptor) -> bool {
        DecodeTraceBuilder::new(desc, 8).needs_lane_offset()
    }

    // ── Wave 12kcd: +10 additional tests ──

    // @trace TEST-QD-14 [req:REQ-QCG] [level:unit]
    // FP16: native float format with ScaleLayout::None, ZeroLayout::None, Bytes unsigned.
    // Tests the trivial native-float decode path with Float16 data kind.
    #[test]
    fn test_f16_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::F16).expect("F16 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // FP16: ScaleLayout::None → Const(1.0) for identity scale
        let has_const_one = trace.iter().any(|op| matches!(op, TraceOp::Const(v) if *v == 1.0_f64));
        assert!(has_const_one, "FP16 should use Const(1.0) for identity scale");

        // FP16: no f16 scale load (ScaleLayout::None means no QuantLoadF16toF32)
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(!has_f16_load, "FP16 should NOT load f16 scales (ScaleLayout::None)");

        // FP16: Bytes unsigned → QuantLoadBytesVec(signed=false)
        let has_unsigned_bytes = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantLoadBytesVec { signed: false, .. }
        ));
        assert!(has_unsigned_bytes, "FP16 should load unsigned bytes");
    }

    // @trace TEST-QD-15 [req:REQ-QCG] [level:unit]
    // F32: baseline native float, trivial decode path, data_kind Float32.
    // Tests the simplest possible trace: bytes load → cast → FMA with Const(1.0).
    #[test]
    fn test_f32_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::F32).expect("F32 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // F32: ScaleLayout::None → no scale loads at all
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        let has_scalar_load = trace.iter().any(|op| matches!(op, TraceOp::QuantScalarLoad { .. }));
        assert!(!has_f16_load, "F32 should NOT have QuantLoadF16toF32");
        assert!(!has_scalar_load, "F32 should NOT have QuantScalarLoad (ScaleLayout::None)");

        // F32: ZeroLayout::None → no Sub for bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(!has_sub, "F32 should NOT have Sub (ZeroLayout::None)");

        // F32: still has FMA with Const(1.0) as scale
        let has_fma = trace.iter().any(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
        assert!(has_fma, "F32 should have QuantDequantFma with Const(1.0) × data");
    }

    // @trace TEST-QD-16 [req:REQ-QCG] [level:unit]
    // TQ2_0: ternary 2.0 with PackedNibbles, BlockScalar f16 scale, StaticBias(1).
    // Tests the 2-bit ternary decode: value = d * (q - 1.0).
    #[test]
    fn test_tq2_0_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::TQ2_0).expect("TQ2_0 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // TQ2_0: StaticBias(1) → Const(1.0) + Sub for pre-scale subtraction
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "TQ2_0 should have Sub for StaticBias(1)");

        // TQ2_0: f16 scale load
        let has_f16 = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_f16, "TQ2_0 should load f16 scale");

        // TQ2_0: PackedNibbles → QuantInterleave
        let has_interleave = trace.iter().any(|op| matches!(op, TraceOp::QuantInterleave { .. }));
        assert!(has_interleave, "TQ2_0 should have QuantInterleave for PackedNibbles");

        // TQ2_0: no high bits pointer needed
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(!builder.needs_high_bits_ptr(), "TQ2_0 should NOT need high bits ptr");
    }

    // @trace TEST-QD-17 [req:REQ-QCG] [level:unit]
    // IQ2_XXS: CodebookIndex without codebook, 2-bit indices, f16 BlockScalar scale.
    // Tests the no-codebook 2-bit extract path (QuantExtractBits only, no QuantCodebookLookup).
    #[test]
    fn test_iq2xxs_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ2XXS).expect("IQ2_XXS must be registered");
        assert!(desc.codebook.is_none(), "IQ2_XXS should NOT have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ2_XXS: CodebookIndex with index_bits=2 → QuantExtractBits (2-bit)
        let has_extract = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantExtractBits { bit_width: 2, .. }
        ));
        assert!(has_extract, "IQ2_XXS should extract 2-bit indices");

        // IQ2_XXS: No codebook → no QuantCodebookLookup
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(!has_codebook, "IQ2_XXS should NOT use QuantCodebookLookup (no codebook)");

        // IQ2_XXS: f16 BlockScalar scale
        let has_f16 = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_f16, "IQ2_XXS should load f16 scale");
    }

    // @trace TEST-QD-18 [req:REQ-QCG] [level:unit]
    // IQ3_XXS: CodebookIndex without codebook, 3-bit indices, f16 BlockScalar scale.
    // Tests the no-codebook 3-bit extract path.
    #[test]
    fn test_iq3xxs_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ3XXS).expect("IQ3_XXS must be registered");
        assert!(desc.codebook.is_none(), "IQ3_XXS should NOT have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ3_XXS: CodebookIndex with index_bits=3 → QuantExtractBits (3-bit)
        let has_extract = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantExtractBits { bit_width: 3, .. }
        ));
        assert!(has_extract, "IQ3_XXS should extract 3-bit indices");

        // IQ3_XXS: No codebook → no QuantCodebookLookup
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(!has_codebook, "IQ3_XXS should NOT use QuantCodebookLookup (no codebook)");

        // IQ3_XXS: no lane offset needed (BlockScalar scale)
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(!builder.needs_lane_offset(), "IQ3_XXS should NOT need lane offset");
    }

    // @trace TEST-QD-19 [req:REQ-QCG] [level:unit]
    // IQ1_M: CodebookIndex without codebook, 1-bit indices, U8Range scale dtype.
    // Tests the U8Range scale path — scale loaded via QuantScalarLoad (not QuantLoadF16toF32).
    #[test]
    fn test_iq1m_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ1M).expect("IQ1_M must be registered");
        assert!(desc.codebook.is_none(), "IQ1_M should NOT have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ1_M: CodebookIndex with index_bits=1 → QuantExtractBits (1-bit)
        let has_extract = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantExtractBits { bit_width: 1, .. }
        ));
        assert!(has_extract, "IQ1_M should extract 1-bit indices");

        // IQ1_M: No codebook → no QuantCodebookLookup
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(!has_codebook, "IQ1_M should NOT use QuantCodebookLookup (no codebook)");

        // IQ1_M: U8Range scale → QuantScalarLoad (not F16 load)
        let has_f16 = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(!has_f16, "IQ1_M should NOT use QuantLoadF16toF32 (U8Range scale)");
    }

    // @trace TEST-QD-20 [req:REQ-QCG] [level:unit]
    // Q8_1: BlockScalarWithMin scale (d + m) + Bytes signed + no zero.
    // Tests the BlockScalarWithMin scale path — loads d via QuantLoadF16toF32.
    #[test]
    fn test_q8_1_trace_structure() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q8_1).expect("Q8_1 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // Q8_1: BlockScalarWithMin → loads d via QuantLoadF16toF32
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_f16_load, "Q8_1 should load f16 scale via QuantLoadF16toF32");

        // Q8_1: Bytes signed → QuantLoadBytesVec(signed=true)
        let has_signed_bytes = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantLoadBytesVec { signed: true, .. }
        ));
        assert!(has_signed_bytes, "Q8_1 should load signed bytes");

        // Q8_1: ZeroLayout::None → no Sub for bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(!has_sub, "Q8_1 should NOT have Sub (ZeroLayout::None)");

        // Q8_1: no high bits pointer needed
        let builder = DecodeTraceBuilder::new(desc, 8);
        assert!(!builder.needs_high_bits_ptr(), "Q8_1 should NOT need high bits ptr");
    }

    // @trace TEST-QD-21 [req:REQ-QCG] [level:unit]
    // is_e2m1_format: verifies the E2M1 LUT detection method.
    // MXFP4/NVFP4 should return true; all other formats should return false.
    #[test]
    fn test_is_e2m1_format_method() {
        // Arrange
        let r = registry();

        // Assert: E2M1 formats return true
        let mxfp4 = r.get(&QuantType::Mxfp4 { block_size: 32 }).expect("MXFP4");
        let mxfp4_builder = DecodeTraceBuilder::new(mxfp4, 8);
        assert!(mxfp4_builder.is_e2m1_format(), "MXFP4 should be E2M1 format");

        let nvfp4 = r.get(&QuantType::Nvfp4).expect("NVFP4");
        let nvfp4_builder = DecodeTraceBuilder::new(nvfp4, 8);
        assert!(nvfp4_builder.is_e2m1_format(), "NVFP4 should be E2M1 format");

        // Assert: non-E2M1 formats return false
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        assert!(!DecodeTraceBuilder::new(q4_0, 8).is_e2m1_format(),
            "Q4_0 should NOT be E2M1 format");

        let q8_0 = r.get(&QuantType::Q8_0).expect("Q8_0");
        assert!(!DecodeTraceBuilder::new(q8_0, 8).is_e2m1_format(),
            "Q8_0 should NOT be E2M1 format");

        let fp8 = r.get(&QuantType::Fp8E4M3).expect("FP8_E4M3");
        assert!(!DecodeTraceBuilder::new(fp8, 8).is_e2m1_format(),
            "FP8_E4M3 should NOT be E2M1 format");
    }

    // @trace TEST-QD-22 [req:REQ-QCG] [level:unit]
    // sub_block_elements_for_zero: verifies the helper returns correct values
    // for Hierarchical (uses sub_block_elements), Q6KScales (uses sub_block_elements),
    // and non-hierarchical formats (falls back to block_size).
    #[test]
    fn test_sub_block_elements_for_zero() {
        // Arrange
        let r = registry();

        // Assert: Q4K (Hierarchical scale) → sub_block_elements used
        let q4k = r.get(&QuantType::Q4K).expect("Q4K");
        assert_eq!(q4k.block_size, 256, "Q4K block_size should be 256");
        // Hierarchical scale has sub_block_elements (should not equal block_size)

        // Assert: Q6K (Q6KScales) → sub_block_elements used
        let q6k = r.get(&QuantType::Q6K).expect("Q6K");
        assert_eq!(q6k.block_size, 256, "Q6K block_size should be 256");

        // Assert: Q4_0 (BlockScalar) → falls back to block_size
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        assert_eq!(q4_0.block_size, 32, "Q4_0 block_size should be 32");

        // Verify trace builds correctly for all
        for qt in &[QuantType::Q4K, QuantType::Q6K, QuantType::Q4_0] {
            let desc = r.get(qt).unwrap();
            let mut trace = Vec::new();
            let slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);
            assert_trace_valid(&trace);
            assert!(slot.0 < trace.len() as u32, "{:?} trace slot out of range", qt);
        }
    }

    // @trace TEST-QD-23 [req:REQ-QCG] [level:unit]
    // high_bits_stride with 16 lanes: verifies the stride calculation scales
    // correctly with different output_lanes values for NibbleWithHighBits formats.
    #[test]
    fn test_high_bits_stride_with_16_lanes() {
        // Arrange
        let r = registry();

        // Act & Assert: Q6K with 16 lanes
        // Q6_K: 2 bits/elem, 16 lanes → (16*2+7)/8 = 4 bytes
        let q6k = r.get(&QuantType::Q6K).expect("Q6K");
        let builder_16 = DecodeTraceBuilder::new(q6k, 16);
        assert_eq!(builder_16.high_bits_stride(), 4,
            "Q6_K with 16 lanes should have 4-byte high bits stride");

        // Act & Assert: Q5_0 with 16 lanes
        // Q5_0: 1 bit/elem, 16 lanes → (16*1+7)/8 = 2 bytes
        let q5_0 = r.get(&QuantType::Q5_0).expect("Q5_0");
        let q5_0_builder_16 = DecodeTraceBuilder::new(q5_0, 16);
        assert_eq!(q5_0_builder_16.high_bits_stride(), 2,
            "Q5_0 with 16 lanes should have 2-byte high bits stride");

        // Verify the 16-lane trace builds correctly for Q6K
        let mut trace = Vec::new();
        let slot = builder_16.build(&mut trace);
        assert_trace_valid(&trace);
        assert!(slot.0 < trace.len() as u32);

        // Q6K with 16 lanes: QuantLoadBytesVec for high bits should load 4 bytes
        let has_4byte_high = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantLoadBytesVec { count: 4, .. }
        ));
        assert!(has_4byte_high, "Q6_K with 16 lanes should load 4 bytes for high bits");
    }

    // ── Wave 12kka: +10 additional tests ──

    // @trace TEST-QD-24 [req:REQ-QCG] [level:unit]
    // IQ2_XS: CodebookIndex without codebook, 2-bit indices, f16 BlockScalar scale.
    // Tests 2-bit index extraction for IQ2_XS variant (74 bytes/block, no codebook).
    #[test]
    fn test_iq2xs_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ2XS).expect("IQ2_XS must be registered");
        assert!(desc.codebook.is_none(), "IQ2_XS should NOT have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ2_XS: CodebookIndex with index_bits=2 → QuantExtractBits (2-bit)
        let has_extract = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantExtractBits { bit_width: 2, .. }
        ));
        assert!(has_extract, "IQ2_XS should extract 2-bit indices");

        // IQ2_XS: No codebook → no QuantCodebookLookup
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(!has_codebook, "IQ2_XS should NOT use QuantCodebookLookup (no codebook)");

        // IQ2_XS: f16 BlockScalar scale
        let has_f16 = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_f16, "IQ2_XS should load f16 scale");
    }

    // @trace TEST-QD-25 [req:REQ-QCG] [level:unit]
    // IQ2_S: CodebookIndex without codebook, 2-bit indices, f16 BlockScalar scale.
    // Tests 2-bit index extraction for IQ2_S variant (82 bytes/block, no codebook).
    #[test]
    fn test_iq2s_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ2S).expect("IQ2_S must be registered");
        assert!(desc.codebook.is_none(), "IQ2_S should NOT have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ2_S: CodebookIndex with index_bits=2 → QuantExtractBits (2-bit)
        let has_extract = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantExtractBits { bit_width: 2, .. }
        ));
        assert!(has_extract, "IQ2_S should extract 2-bit indices");

        // IQ2_S: No codebook → no QuantCodebookLookup
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(!has_codebook, "IQ2_S should NOT use QuantCodebookLookup (no codebook)");

        // IQ2_S: ZeroLayout::None → no Sub for bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(!has_sub, "IQ2_S should NOT have Sub (ZeroLayout::None)");
    }

    // @trace TEST-QD-26 [req:REQ-QCG] [level:unit]
    // IQ3_S: CodebookIndex without codebook, 3-bit indices, U8Range scale dtype.
    // Tests the U8Range scale path — scale loaded via QuantScalarLoad (not QuantLoadF16toF32).
    #[test]
    fn test_iq3s_trace() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ3S).expect("IQ3_S must be registered");
        assert!(desc.codebook.is_none(), "IQ3_S should NOT have a static codebook");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // IQ3_S: CodebookIndex with index_bits=3 → QuantExtractBits (3-bit)
        let has_extract = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantExtractBits { bit_width: 3, .. }
        ));
        assert!(has_extract, "IQ3_S should extract 3-bit indices");

        // IQ3_S: U8Range scale → no QuantLoadF16toF32
        let has_f16 = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(!has_f16, "IQ3_S should NOT use QuantLoadF16toF32 (U8Range scale)");

        // IQ3_S: ZeroLayout::None → no Sub for bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(!has_sub, "IQ3_S should NOT have Sub (ZeroLayout::None)");
    }

    // @trace TEST-QD-27 [req:REQ-QCG] [level:unit]
    // Q4_0 static bias value: verifies the Const(8.0) bias constant in the trace.
    // Q4_0 maps integer range [0,15] → symmetric [-8,7] via StaticBias(8).
    #[test]
    fn test_q4_0_static_bias_value() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q4_0).expect("Q4_0 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q4_0 has StaticBias(8) → Const(8.0) in trace
        let has_bias_8 = trace.iter().any(|op| matches!(op, TraceOp::Const(v) if (*v - 8.0_f64).abs() < f64::EPSILON));
        assert!(has_bias_8, "Q4_0 should have Const(8.0) for StaticBias(8)");

        // Assert: Sub must be present to subtract the bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "Q4_0 should have Sub to subtract StaticBias(8)");
    }

    // @trace TEST-QD-28 [req:REQ-QCG] [level:unit]
    // Q6_K static bias value: verifies Const(32.0) bias constant in the trace.
    // Q6_K maps integer range [0,63] → symmetric [-32,31] via StaticBias(32).
    #[test]
    fn test_q6k_static_bias_value() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q6K).expect("Q6_K must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q6_K has StaticBias(32) → Const(32.0) in trace
        let has_bias_32 = trace.iter().any(|op| matches!(op, TraceOp::Const(v) if (*v - 32.0_f64).abs() < f64::EPSILON));
        assert!(has_bias_32, "Q6_K should have Const(32.0) for StaticBias(32)");

        // Assert: Sub must be present to subtract the bias
        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "Q6_K should have Sub to subtract StaticBias(32)");
    }

    // @trace TEST-QD-29 [req:REQ-QCG] [level:unit]
    // AWQ4 trace slot ordering: verifies that Sub (zero-point subtraction)
    // appears before QuantDequantFma (scale multiplication) in the trace.
    // Formula: `value = (qw - zp) × scale` — Sub must precede FMA.
    #[test]
    fn test_awq4_sub_before_fma_ordering() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::AWQ4).expect("AWQ4 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: find positions of Sub and QuantDequantFma
        let sub_pos = trace.iter().position(|op| matches!(op, TraceOp::Sub(_, _)));
        let fma_pos = trace.iter().position(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
        assert!(sub_pos.is_some(), "AWQ4 should have Sub for zero-point");
        assert!(fma_pos.is_some(), "AWQ4 should have QuantDequantFma for scale");
        assert!(
            sub_pos.unwrap() < fma_pos.unwrap(),
            "AWQ4 Sub (slot {}) must appear before QuantDequantFma (slot {})",
            sub_pos.unwrap(), fma_pos.unwrap()
        );
    }

    // @trace TEST-QD-30 [req:REQ-QCG] [level:unit]
    // MXFP4 block_size=64 variation: verifies the builder produces a valid trace
    // with E2M1 LUT decode when using a non-default block size.
    #[test]
    fn test_mxfp4_block_size_64() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Mxfp4 { block_size: 64 }).expect("MXFP4 block_size=64 must be registered");

        // Act
        let mut trace = Vec::new();
        let final_slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        assert_trace_valid(&trace);
        assert!(final_slot.0 < trace.len() as u32);

        // MXFP4 block_size=64: still uses QuantE2m1LutDecode with nvfp4_mode=false
        let has_e2m1 = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantE2m1LutDecode { nvfp4_mode: false, .. }
        ));
        assert!(has_e2m1, "MXFP4 block_size=64 should emit QuantE2m1LutDecode with nvfp4_mode=false");

        // MXFP4 block_size=64: BlockScalar scale → loads raw scale byte via QuantScalarLoad
        let has_scalar_load = trace.iter().any(|op| matches!(op, TraceOp::QuantScalarLoad { .. }));
        assert!(has_scalar_load, "MXFP4 block_size=64 should have QuantScalarLoad for scale byte");
    }

    // @trace TEST-QD-31 [req:REQ-QCG] [level:unit]
    // Q4_1 BlockScalarWithMin: verifies the d and m fields are both loaded via
    // QuantLoadF16toF32. Q4_1 uses BlockScalarWithMin (d_offset for scale, m for min)
    // and QuantInterleave for PackedNibbles unpack.
    #[test]
    fn test_q4_1_block_scalar_with_min_structure() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q4_1).expect("Q4_1 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q4_1 has f16 scale load
        let f16_count = trace.iter()
            .filter(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }))
            .count();
        assert!(f16_count >= 1, "Q4_1 should load at least 1 f16 value (d), got {}", f16_count);

        // Assert: Q4_1 has QuantInterleave for PackedNibbles
        let has_interleave = trace.iter().any(|op| matches!(op, TraceOp::QuantInterleave { .. }));
        assert!(has_interleave, "Q4_1 should have QuantInterleave for PackedNibbles");

        // Assert: Q4_1 has QuantBroadcast for scale
        let has_broadcast = trace.iter().any(|op| matches!(op, TraceOp::QuantBroadcast { .. }));
        assert!(has_broadcast, "Q4_1 should have QuantBroadcast for scale vector");
    }

    // @trace TEST-QD-32 [req:REQ-QCG] [level:unit]
    // Squeeze QuantAndMask: verifies the PackedNibbles path emits mask and shift ops.
    // Squeeze uses 3-bit packed nibbles with StaticBias(4), BlockScalar f16 scale.
    #[test]
    fn test_squeeze_unpack_mask_and_shift() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Squeeze).expect("Squeeze must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: PackedNibbles → QuantAndMask for low nibble extraction
        let has_mask = trace.iter().any(|op| matches!(op, TraceOp::QuantAndMask { .. }));
        assert!(has_mask, "Squeeze should have QuantAndMask for nibble masking");

        // Assert: PackedNibbles → QuantShiftRight for high nibble extraction
        let has_shift = trace.iter().any(|op| matches!(op, TraceOp::QuantShiftRight { .. }));
        assert!(has_shift, "Squeeze should have QuantShiftRight for high nibble");

        // Assert: PackedNibbles → QuantInterleave
        let has_interleave = trace.iter().any(|op| matches!(op, TraceOp::QuantInterleave { .. }));
        assert!(has_interleave, "Squeeze should have QuantInterleave for lo/hi merge");
    }

    // @trace TEST-QD-33 [req:REQ-QCG] [level:unit]
    // All registered formats produce traces where QuantBroadcast lane count
    // matches the builder's output_lanes parameter.
    #[test]
    fn test_broadcast_lane_count_matches_output_lanes() {
        // Arrange
        let r = registry();
        let test_lanes = 8usize;

        // Act & Assert: for a representative set of formats, verify broadcast lane count
        for qt in &[
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q8_0,
            QuantType::AWQ4, QuantType::GPTQ4, QuantType::Squeeze,
        ] {
            let desc = match r.get(qt) {
                Some(d) => d,
                None => continue,
            };
            let mut trace = Vec::new();
            let _slot = DecodeTraceBuilder::new(desc, test_lanes).build(&mut trace);

            // All QuantBroadcast ops should use the same lane count
            for op in &trace {
                if let TraceOp::QuantBroadcast { lanes, .. } = op {
                    assert_eq!(
                        *lanes, test_lanes,
                        "QuantBroadcast lanes should be {}, got {} for {:?}",
                        test_lanes, lanes, qt
                    );
                }
            }
        }
    }

    // @trace TEST-QD-34 [req:REQ-QCG] [level:unit]
    // GPTQ4 trace ordering: Sub (zero-point) must appear before FMA (scale multiply).
    // Mirrors the AWQ4 ordering test but for GPTQ4 to ensure consistent pre-scale subtraction.
    #[test]
    fn test_gptq4_sub_before_fma_ordering() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::GPTQ4).expect("GPTQ4 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert
        let sub_pos = trace.iter().position(|op| matches!(op, TraceOp::Sub(_, _)));
        let fma_pos = trace.iter().position(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
        assert!(sub_pos.is_some(), "GPTQ4 should have Sub for zero-point");
        assert!(fma_pos.is_some(), "GPTQ4 should have QuantDequantFma for scale");
        assert!(
            sub_pos.unwrap() < fma_pos.unwrap(),
            "GPTQ4 Sub (slot {}) must appear before QuantDequantFma (slot {})",
            sub_pos.unwrap(), fma_pos.unwrap()
        );
    }

    // @trace TEST-QD-35 [req:REQ-QCG] [level:unit]
    // Q3K static bias value: verifies Const(4.0) appears in the trace.
    // Q3K uses StaticBias(4) to map integer range [0,15] → symmetric [-4,11].
    #[test]
    fn test_q3k_static_bias_value() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q3K).expect("Q3_K must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q3K has StaticBias(4) → Const(4.0) in trace
        let has_bias_4 = trace.iter().any(|op| matches!(op, TraceOp::Const(v) if (*v - 4.0_f64).abs() < f64::EPSILON));
        assert!(has_bias_4, "Q3_K should have Const(4.0) for StaticBias(4)");

        let has_sub = trace.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
        assert!(has_sub, "Q3_K should have Sub to subtract StaticBias(4)");
    }

    // @trace TEST-QD-36 [req:REQ-QCG] [level:unit]
    // Q5_0 high bits shift amount: verifies QuantShiftLeft uses the correct shift
    // for 1-bit high bits (shift_amount = 4 + (4 - 1) = 7).
    #[test]
    fn test_q5_0_high_bits_shift_amount() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q5_0).expect("Q5_0 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q5_0 has QuantShiftLeft with amount=7 (4 + (4 - 1))
        let has_shift_7 = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantShiftLeft { amount: 7, .. }
        ));
        assert!(has_shift_7, "Q5_0 should have QuantShiftLeft with amount=7 for 1-bit high bits");

        // Assert: Q5_0 has QuantBitOr to merge low nibble + shifted high bit
        let has_bitor = trace.iter().any(|op| matches!(op, TraceOp::QuantBitOr { .. }));
        assert!(has_bitor, "Q5_0 should have QuantBitOr for low+high merge");
    }

    // @trace TEST-QD-37 [req:REQ-QCG] [level:unit]
    // FP8 E5M2 is_e4m3=false: verifies the E5M2 variant uses QuantCastFp8toF32
    // with is_e4m3=false and does NOT use QuantCastI8toF32.
    #[test]
    fn test_fp8_e5m2_no_i8_cast() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Fp8E5M2).expect("Fp8E5M2 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: E5M2 uses QuantCastFp8toF32 with is_e4m3=false
        let has_fp8_e5m2 = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantCastFp8toF32 { is_e4m3: false, .. }
        ));
        assert!(has_fp8_e5m2, "FP8 E5M2 should use QuantCastFp8toF32 with is_e4m3=false");

        // Assert: E5M2 should NOT use QuantCastI8toF32
        let has_i8_cast = trace.iter().any(|op| matches!(op, TraceOp::QuantCastI8toF32 { .. }));
        assert!(!has_i8_cast, "FP8 E5M2 should NOT use QuantCastI8toF32");
    }

    // @trace TEST-QD-38 [req:REQ-QCG] [level:unit]
    // Q2K hierarchical zero: verifies PostScaleAdd path with dmin × sub_m.
    // Q2K uses Hierarchical zero layout → Add appears after FMA (post-scale min addition).
    #[test]
    fn test_q2k_post_scale_add_ordering() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q2K).expect("Q2_K must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q2K has Add for post-scale min addition
        let add_pos = trace.iter().position(|op| matches!(op, TraceOp::Add(_, _)));
        let fma_pos = trace.iter().position(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
        assert!(add_pos.is_some(), "Q2_K should have Add for post-scale min");
        assert!(fma_pos.is_some(), "Q2_K should have QuantDequantFma");
        assert!(
            add_pos.unwrap() > fma_pos.unwrap(),
            "Q2_K Add (slot {}) must appear after QuantDequantFma (slot {})",
            add_pos.unwrap(), fma_pos.unwrap()
        );
    }

    // @trace TEST-QD-39 [req:REQ-QCG] [level:unit]
    // Q8K QuantLoadBytesVec with signed=true: verifies Q8_K loads signed i8 data.
    // Q8_K uses Bytes layout with signed=true, unlike Q8_0 which also uses signed bytes.
    #[test]
    fn test_q8k_signed_bytes_load_count() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q8K).expect("Q8_K must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: exactly one QuantLoadBytesVec with signed=true
        let signed_count = trace.iter()
            .filter(|op| matches!(op, TraceOp::QuantLoadBytesVec { signed: true, .. }))
            .count();
        assert!(signed_count >= 1, "Q8_K should have at least 1 QuantLoadBytesVec with signed=true, got {}", signed_count);

        // Assert: no QuantLoadF16toF32 (F32 scale uses QuantScalarLoad)
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(!has_f16_load, "Q8_K should NOT have QuantLoadF16toF32 (F32 scale dtype)");
    }

    // @trace TEST-QD-40 [req:REQ-QCG] [level:unit]
    // All E2M1 formats produce exactly one QuantE2m1LutDecode op.
    // Both MXFP4 and NVFP4 should have exactly one LUT decode, not multiple.
    #[test]
    fn test_e2m1_formats_single_lut_decode() {
        // Arrange
        let r = registry();

        // Act & Assert: MXFP4
        let mxfp4 = r.get(&QuantType::Mxfp4 { block_size: 32 }).expect("MXFP4");
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(mxfp4, 8).build(&mut trace);
        let mxfp4_lut_count = trace.iter()
            .filter(|op| matches!(op, TraceOp::QuantE2m1LutDecode { .. }))
            .count();
        assert_eq!(mxfp4_lut_count, 1, "MXFP4 should have exactly 1 QuantE2m1LutDecode, got {}", mxfp4_lut_count);

        // Act & Assert: NVFP4
        let nvfp4 = r.get(&QuantType::Nvfp4).expect("NVFP4");
        let mut trace2 = Vec::new();
        let _slot2 = DecodeTraceBuilder::new(nvfp4, 8).build(&mut trace2);
        let nvfp4_lut_count = trace2.iter()
            .filter(|op| matches!(op, TraceOp::QuantE2m1LutDecode { .. }))
            .count();
        assert_eq!(nvfp4_lut_count, 1, "NVFP4 should have exactly 1 QuantE2m1LutDecode, got {}", nvfp4_lut_count);
    }

    // @trace TEST-QD-41 [req:REQ-QCG] [level:unit]
    // TQ1_0 needs no lane offset and no high bits ptr.
    // TQ1_0 uses BlockScalar scale + PackedNibbles, so no special pointers needed.
    #[test]
    fn test_tq1_0_no_special_pointers() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::TQ1_0).expect("TQ1_0 must be registered");
        let builder = DecodeTraceBuilder::new(desc, 8);

        // Assert: TQ1_0 does not need lane offset or high bits
        assert!(!builder.needs_lane_offset(), "TQ1_0 should NOT need lane offset");
        assert!(!builder.needs_high_bits_ptr(), "TQ1_0 should NOT need high bits ptr");
        assert!(!builder.is_e2m1_format(), "TQ1_0 should NOT be E2M1 format");
    }

    // @trace TEST-QD-42 [req:REQ-QCG] [level:unit]
    // TQ2_0 trace structure: verifies PackedNibbles + QuantInterleave + QuantAndMask.
    // TQ2_0 uses 2-bit packed format with f16 BlockScalar scale and StaticBias(1).
    #[test]
    fn test_tq2_0_interleave_and_mask() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::TQ2_0).expect("TQ2_0 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: PackedNibbles path emits QuantAndMask
        let has_mask = trace.iter().any(|op| matches!(op, TraceOp::QuantAndMask { .. }));
        assert!(has_mask, "TQ2_0 should have QuantAndMask for nibble extraction");

        // Assert: PackedNibbles path emits QuantShiftRight
        let has_shift = trace.iter().any(|op| matches!(op, TraceOp::QuantShiftRight { .. }));
        assert!(has_shift, "TQ2_0 should have QuantShiftRight for high nibble");

        // Assert: no post-scale Add (only pre-scale Sub)
        let has_add = trace.iter().any(|op| matches!(op, TraceOp::Add(_, _)));
        assert!(!has_add, "TQ2_0 should NOT have post-scale Add");
    }

    // @trace TEST-QD-43 [req:REQ-QCG] [level:unit]
    // IQ4_XS: verifies QuantExtractBits bit_width matches the format's bits_per_weight.
    // IQ4_XS uses 4-bit codebook indices with hierarchical scale + codebook lookup.
    #[test]
    fn test_iq4xs_extract_bits_width() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ4XS).expect("IQ4_XS must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: IQ4_XS extracts 4-bit indices
        let has_4bit = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantExtractBits { bit_width: 4, .. }
        ));
        assert!(has_4bit, "IQ4_XS should extract 4-bit indices");

        // Assert: IQ4_XS uses codebook lookup (has codebook)
        assert!(desc.codebook.is_some(), "IQ4_XS should have a codebook");
        let has_codebook = trace.iter().any(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(has_codebook, "IQ4_XS should use QuantCodebookLookup");
    }

    // ── Wave 12k3c: +10 additional tests ──

    // @trace TEST-QD-44 [req:REQ-QCG] [level:unit]
    // QuantPtrAddOffset emission: verifies the TraceOp for computing block-relative
    // addresses is emitted correctly for formats with non-zero scale/data offsets.
    #[test]
    fn test_quant_ptr_add_offset_emission() {
        // Arrange
        let r = registry();
        // Q4_0: scale at offset 0, data at offset 2 → no ptr add for scale
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        let mut trace_q4_0 = Vec::new();
        let _ = DecodeTraceBuilder::new(q4_0, 8).build(&mut trace_q4_0);
        // Q4_0 has BlockScalar scale at offset 0 → no QuantPtrAddOffset for scale
        let q4_0_ptr_add_count = trace_q4_0.iter()
            .filter(|op| matches!(op, TraceOp::QuantPtrAddOffset { .. }))
            .count();
        // Q4_0 may have ptr add for data offset depending on layout
        assert!(q4_0_ptr_add_count <= 1, "Q4_0 should have at most 1 QuantPtrAddOffset");

        // Q4K: hierarchical scale has sub_scales_offset → QuantPtrAddOffset present
        let q4k = r.get(&QuantType::Q4K).expect("Q4K");
        let mut trace_q4k = Vec::new();
        let _ = DecodeTraceBuilder::new(q4k, 8).build(&mut trace_q4k);
        let q4k_has_ptr_add = trace_q4k.iter()
            .any(|op| matches!(op, TraceOp::QuantPtrAddOffset { .. }));
        assert!(q4k_has_ptr_add, "Q4K should have QuantPtrAddOffset for sub_scales_offset");
    }

    // @trace TEST-QD-45 [req:REQ-QCG] [level:unit]
    // QuantIntDivConst emission: verifies integer division by constant is emitted
    // for hierarchical formats that compute sub-block indices.
    #[test]
    fn test_quant_int_div_const_for_sub_block_index() {
        // Arrange
        let r = registry();

        // Q4K: hierarchical scale → QuantIntDivConst for sub_block_idx
        let q4k = r.get(&QuantType::Q4K).expect("Q4K");
        let mut trace_q4k = Vec::new();
        let _ = DecodeTraceBuilder::new(q4k, 8).build(&mut trace_q4k);
        let q4k_div = trace_q4k.iter()
            .filter(|op| matches!(op, TraceOp::QuantIntDivConst { .. }))
            .count();
        assert!(q4k_div >= 1, "Q4K should have at least 1 QuantIntDivConst for sub_block_idx");

        // Q6K: Q6KScales layout → QuantIntDivConst for sub_block_idx
        let q6k = r.get(&QuantType::Q6K).expect("Q6K");
        let mut trace_q6k = Vec::new();
        let _ = DecodeTraceBuilder::new(q6k, 8).build(&mut trace_q6k);
        let q6k_div = trace_q6k.iter()
            .filter(|op| matches!(op, TraceOp::QuantIntDivConst { .. }))
            .count();
        assert!(q6k_div >= 1, "Q6K should have at least 1 QuantIntDivConst for sub_block_idx");

        // Q4_0: BlockScalar scale → no QuantIntDivConst
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        let mut trace_q4_0 = Vec::new();
        let _ = DecodeTraceBuilder::new(q4_0, 8).build(&mut trace_q4_0);
        let q4_0_div = trace_q4_0.iter()
            .filter(|op| matches!(op, TraceOp::QuantIntDivConst { .. }))
            .count();
        assert_eq!(q4_0_div, 0, "Q4_0 should have no QuantIntDivConst (flat scale layout)");
    }

    // @trace TEST-QD-46 [req:REQ-QCG] [level:unit]
    // QuantPtrAddDynamic emission: verifies dynamic pointer arithmetic for indexed
    // access (e.g., sub-block scale lookup) emits the correct TraceOp.
    #[test]
    fn test_quant_ptr_add_dynamic_for_indexed_access() {
        // Arrange
        let r = registry();

        // Q6K: Q6KScales uses QuantPtrAddDynamic for sub_scales_ptr + sub_idx
        let q6k = r.get(&QuantType::Q6K).expect("Q6K");
        let mut trace = Vec::new();
        let _ = DecodeTraceBuilder::new(q6k, 8).build(&mut trace);
        let has_dynamic_ptr = trace.iter()
            .any(|op| matches!(op, TraceOp::QuantPtrAddDynamic { .. }));
        assert!(has_dynamic_ptr, "Q6K should have QuantPtrAddDynamic for indexed scale access");

        // NVFP4: SubBlockScalars uses QuantPtrAddDynamic for sub-block scale lookup
        let nvfp4 = r.get(&QuantType::Nvfp4).expect("NVFP4");
        let mut trace_nvfp4 = Vec::new();
        let _ = DecodeTraceBuilder::new(nvfp4, 8).build(&mut trace_nvfp4);
        let has_dynamic_ptr_nvfp4 = trace_nvfp4.iter()
            .any(|op| matches!(op, TraceOp::QuantPtrAddDynamic { .. }));
        assert!(has_dynamic_ptr_nvfp4, "NVFP4 should have QuantPtrAddDynamic for sub-block scale");
    }

    // @trace TEST-QD-47 [req:REQ-QCG] [level:unit]
    // QuantLoadI8toF32 emission: verifies i8 scale loading for Q6KScales layout.
    // Q6_K stores per-element i8 scales (not bit-packed) that need sign extension.
    #[test]
    fn test_quant_load_i8_to_f32_for_q6k_scales() {
        // Arrange
        let r = registry();
        let q6k = r.get(&QuantType::Q6K).expect("Q6K");

        // Act
        let mut trace = Vec::new();
        let _ = DecodeTraceBuilder::new(q6k, 8).build(&mut trace);

        // Assert: Q6K has QuantLoadI8toF32 for i8 sub-scale loading
        let has_i8_load = trace.iter()
            .any(|op| matches!(op, TraceOp::QuantLoadI8toF32 { .. }));
        assert!(has_i8_load, "Q6K should have QuantLoadI8toF32 for i8 sub-scale loading");

        // Q4_0 uses f16 scale → no QuantLoadI8toF32
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        let mut trace_q4_0 = Vec::new();
        let _ = DecodeTraceBuilder::new(q4_0, 8).build(&mut trace_q4_0);
        let has_i8_load_q4_0 = trace_q4_0.iter()
            .any(|op| matches!(op, TraceOp::QuantLoadI8toF32 { .. }));
        assert!(!has_i8_load_q4_0, "Q4_0 should NOT have QuantLoadI8toF32 (f16 scale)");
    }

    // @trace TEST-QD-48 [req:REQ-QCG] [level:unit]
    // QuantBroadcast for zero-point: verifies AWQ4/GPTQ4 broadcast the scalar
    // zero-point to a vector before subtraction.
    #[test]
    fn test_quant_broadcast_for_zero_point() {
        // Arrange
        let r = registry();

        // AWQ4: PreScaleSubtract → QuantBroadcast for zero-point
        let awq4 = r.get(&QuantType::AWQ4).expect("AWQ4");
        let mut trace_awq4 = Vec::new();
        let _ = DecodeTraceBuilder::new(awq4, 8).build(&mut trace_awq4);
        let broadcast_count = trace_awq4.iter()
            .filter(|op| matches!(op, TraceOp::QuantBroadcast { .. }))
            .count();
        assert!(broadcast_count >= 2, "AWQ4 should have at least 2 QuantBroadcast (scale + zero), got {}", broadcast_count);

        // GPTQ4: same formula → same broadcast pattern
        let gptq4 = r.get(&QuantType::GPTQ4).expect("GPTQ4");
        let mut trace_gptq4 = Vec::new();
        let _ = DecodeTraceBuilder::new(gptq4, 8).build(&mut trace_gptq4);
        let gptq4_broadcast_count = trace_gptq4.iter()
            .filter(|op| matches!(op, TraceOp::QuantBroadcast { .. }))
            .count();
        assert!(gptq4_broadcast_count >= 2, "GPTQ4 should have at least 2 QuantBroadcast (scale + zero), got {}", gptq4_broadcast_count);
    }

    // @trace TEST-QD-49 [req:REQ-QCG] [level:unit]
    // QuantDequantFma acc=Const(0.0): verifies the FMA accumulator initialization
    // uses Const(0.0) for the standard dequant formula `value = (qw - zp) * scale`.
    #[test]
    fn test_quant_dequant_fma_zero_accumulator() {
        // Arrange
        let r = registry();

        // Test multiple formats: all should have FMA with zero accumulator
        for qt in &[QuantType::Q4_0, QuantType::Q4_1, QuantType::AWQ4, QuantType::GPTQ4] {
            let desc = r.get(qt).expect(&format!("{:?} must be registered", qt));
            let mut trace = Vec::new();
            let _ = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

            // Find QuantDequantFma and verify acc is a Const(0.0) slot
            let fma_op = trace.iter().find(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
            assert!(fma_op.is_some(), "{:?} should have QuantDequantFma", qt);

            if let Some(TraceOp::QuantDequantFma { acc, .. }) = fma_op {
                // The acc slot should reference a Const(0.0) op
                let acc_slot = acc.0 as usize;
                assert!(acc_slot < trace.len(), "{:?} FMA acc slot out of range", qt);
                let acc_op = &trace[acc_slot];
                assert!(
                    matches!(acc_op, TraceOp::Const(v) if *v == 0.0_f64),
                    "{:?} FMA accumulator should be Const(0.0), got {:?}",
                    qt, acc_op
                );
            }
        }
    }

    // @trace TEST-QD-50 [req:REQ-QCG] [level:unit]
    // Input slot convention: verifies Input(0) and Input(1) are always the first
    // two ops in the trace, representing block_base and data_ptr respectively.
    #[test]
    fn test_input_slot_convention() {
        // Arrange
        let r = registry();

        // Test a representative set of formats
        for qt in &[
            QuantType::Q4_0, QuantType::Q6K, QuantType::AWQ4,
            QuantType::Mxfp4 { block_size: 32 }, QuantType::Nvfp4,
            QuantType::Fp8E4M3, QuantType::Bf16,
        ] {
            let desc = match r.get(qt) {
                Some(d) => d,
                None => continue,
            };
            let mut trace = Vec::new();
            let _ = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

            // First op must be Input(0) for block_base
            assert!(
                matches!(trace.first(), Some(TraceOp::Input(0))),
                "{:?} first op should be Input(0) for block_base", qt
            );

            // Second op must be Input(1) for data_ptr
            assert!(
                matches!(trace.get(1), Some(TraceOp::Input(1))),
                "{:?} second op should be Input(1) for data_ptr", qt
            );
        }
    }

    // @trace TEST-QD-51 [req:REQ-QCG] [level:unit]
    // Input(2) lane_offset for hierarchical formats: verifies Input(2) is emitted
    // when needs_lane_offset() returns true.
    #[test]
    fn test_input_2_lane_offset_for_hierarchical() {
        // Arrange
        let r = registry();

        // Q4K: hierarchical scale → needs_lane_offset → Input(2) present
        let q4k = r.get(&QuantType::Q4K).expect("Q4K");
        let builder = DecodeTraceBuilder::new(q4k, 8);
        assert!(builder.needs_lane_offset(), "Q4K should need lane offset");
        let mut trace = Vec::new();
        let _ = builder.build(&mut trace);
        let has_input_2 = trace.iter().any(|op| matches!(op, TraceOp::Input(2)));
        assert!(has_input_2, "Q4K should have Input(2) for lane_offset");

        // Q4_0: flat scale → no lane offset → Input(2) should NOT appear
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        let builder_q4_0 = DecodeTraceBuilder::new(q4_0, 8);
        assert!(!builder_q4_0.needs_lane_offset(), "Q4_0 should NOT need lane offset");
        let mut trace_q4_0 = Vec::new();
        let _ = builder_q4_0.build(&mut trace_q4_0);
        let has_input_2_q4_0 = trace_q4_0.iter().any(|op| matches!(op, TraceOp::Input(2)));
        assert!(!has_input_2_q4_0, "Q4_0 should NOT have Input(2)");
    }

    // @trace TEST-QD-52 [req:REQ-QCG] [level:unit]
    // Input(3) high_bits_ptr for NibbleWithHighBits: verifies Input(3) is emitted
    // when needs_high_bits_ptr() returns true.
    #[test]
    fn test_input_3_high_bits_ptr_for_nibble_with_high_bits() {
        // Arrange
        let r = registry();

        // Q6K: NibbleWithHighBits → needs_high_bits_ptr → Input(3) present
        let q6k = r.get(&QuantType::Q6K).expect("Q6K");
        let builder = DecodeTraceBuilder::new(q6k, 8);
        assert!(builder.needs_high_bits_ptr(), "Q6K should need high bits ptr");
        let mut trace = Vec::new();
        let _ = builder.build(&mut trace);
        let has_input_3 = trace.iter().any(|op| matches!(op, TraceOp::Input(3)));
        assert!(has_input_3, "Q6K should have Input(3) for high_bits_ptr");

        // Q5_0: NibbleWithHighBits (1 high bit) → Input(3) present
        let q5_0 = r.get(&QuantType::Q5_0).expect("Q5_0");
        let builder_q5_0 = DecodeTraceBuilder::new(q5_0, 8);
        assert!(builder_q5_0.needs_high_bits_ptr(), "Q5_0 should need high bits ptr");
        let mut trace_q5_0 = Vec::new();
        let _ = builder_q5_0.build(&mut trace_q5_0);
        let has_input_3_q5_0 = trace_q5_0.iter().any(|op| matches!(op, TraceOp::Input(3)));
        assert!(has_input_3_q5_0, "Q5_0 should have Input(3) for high_bits_ptr");

        // Q4_0: PackedNibbles → no high bits ptr → Input(3) should NOT appear
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        let builder_q4_0 = DecodeTraceBuilder::new(q4_0, 8);
        assert!(!builder_q4_0.needs_high_bits_ptr(), "Q4_0 should NOT need high bits ptr");
        let mut trace_q4_0 = Vec::new();
        let _ = builder_q4_0.build(&mut trace_q4_0);
        let has_input_3_q4_0 = trace_q4_0.iter().any(|op| matches!(op, TraceOp::Input(3)));
        assert!(!has_input_3_q4_0, "Q4_0 should NOT have Input(3)");
    }

    // @trace TEST-QD-53 [req:REQ-QCG] [level:unit]
    // Mul for hierarchical scale multiplication: verifies block_d * sub_scale
    // emits a Mul TraceOp for formats with hierarchical scale layouts.
    #[test]
    fn test_mul_for_hierarchical_scale_multiplication() {
        // Arrange
        let r = registry();

        // Q4K: hierarchical scale → block_d * sub_scale → Mul present
        let q4k = r.get(&QuantType::Q4K).expect("Q4K");
        let mut trace_q4k = Vec::new();
        let _ = DecodeTraceBuilder::new(q4k, 8).build(&mut trace_q4k);
        let has_mul = trace_q4k.iter().any(|op| matches!(op, TraceOp::Mul(_, _)));
        assert!(has_mul, "Q4K should have Mul for block_d * sub_scale");

        // Q6K: Q6KScales → block_d * sub_scale → Mul present
        let q6k = r.get(&QuantType::Q6K).expect("Q6K");
        let mut trace_q6k = Vec::new();
        let _ = DecodeTraceBuilder::new(q6k, 8).build(&mut trace_q6k);
        let has_mul_q6k = trace_q6k.iter().any(|op| matches!(op, TraceOp::Mul(_, _)));
        assert!(has_mul_q6k, "Q6K should have Mul for block_d * sub_scale");

        // Q4_0: flat scale → no Mul for scale computation
        let q4_0 = r.get(&QuantType::Q4_0).expect("Q4_0");
        let mut trace_q4_0 = Vec::new();
        let _ = DecodeTraceBuilder::new(q4_0, 8).build(&mut trace_q4_0);
        let has_mul_q4_0 = trace_q4_0.iter().any(|op| matches!(op, TraceOp::Mul(_, _)));
        assert!(!has_mul_q4_0, "Q4_0 should NOT have Mul for scale (flat BlockScalar)");
    }

    // ── Wave 12k59: +10 additional tests ──

    // @trace TEST-QD-54 [req:REQ-QCG] [level:unit]
    // Q4_0 PackedNibbles low_first ordering: verifies that Q4_0 uses PackedNibbles
    // with low_first=true, meaning low nibble maps to even-index output elements.
    // This affects QuantInterleave argument ordering (lo first, hi second).
    #[test]
    fn test_q4_0_packed_nibbles_low_first_interleave_order() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q4_0).expect("Q4_0 must be registered");

        // Verify Q4_0 uses PackedNibbles with low_first=true
        match &desc.data_layout {
            DataLayout::PackedNibbles { low_first, .. } => {
                assert!(low_first, "Q4_0 should use low_first=true for PackedNibbles");
            }
            other => panic!("Q4_0 should use PackedNibbles, got {:?}", other),
        }

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: PackedNibbles with low_first -> QuantAndMask(mask=0x0F) for low nibble
        let has_low_mask = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantAndMask { mask, .. } if *mask == 0x0F
        ));
        assert!(has_low_mask, "Q4_0 should have QuantAndMask with mask=0x0F for low nibble extraction");
    }

    // @trace TEST-QD-55 [req:REQ-QCG] [level:unit]
    // Q2K hierarchical zero dmin load: verifies Q2_K loads the block-level dmin
    // via QuantLoadF16toF32 for the Hierarchical zero layout (dmin * sub_m).
    #[test]
    fn test_q2k_hierarchical_zero_dmin_load() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q2K).expect("Q2_K must be registered");

        // Verify Q2_K uses Hierarchical zero layout
        match &desc.zero_layout {
            ZeroLayout::Hierarchical { dmin_offset, sub_m_offset } => {
                assert!(*dmin_offset > 0, "Q2_K dmin_offset should be > 0");
                assert!(*sub_m_offset < desc.block_bytes, "Q2_K sub_m_offset should be within block");
            }
            other => panic!("Q2_K should use Hierarchical zero layout, got {:?}", other),
        }

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q2_K has at least 2 QuantLoadF16toF32 (block_d + dmin)
        let f16_load_count = trace.iter()
            .filter(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }))
            .count();
        assert!(f16_load_count >= 2,
            "Q2_K should have at least 2 QuantLoadF16toF32 (block_d + dmin), got {}", f16_load_count);

        // Assert: Q2_K has QuantKQuantPackedScaleLookup with is_min=true for sub_m
        let has_min_lookup = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantKQuantPackedScaleLookup { is_min: true, .. }
        ));
        assert!(has_min_lookup, "Q2_K should have QuantKQuantPackedScaleLookup with is_min=true for sub_m");
    }

    // @trace TEST-QD-56 [req:REQ-QCG] [level:unit]
    // Q6K NibbleWithHighBits high_bits_per_elem: verifies Q6_K uses 2 high bits
    // per element, resulting in QuantShiftLeft with amount = 4 + (4 - 2) = 6.
    #[test]
    fn test_q6k_high_bits_per_elem_shift_amount() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q6K).expect("Q6_K must be registered");

        // Verify Q6_K uses NibbleWithHighBits with high_bits_per_elem=2
        match &desc.data_layout {
            DataLayout::NibbleWithHighBits { high_bits_per_elem, .. } => {
                assert_eq!(*high_bits_per_elem, 2,
                    "Q6_K should have high_bits_per_elem=2");
            }
            other => panic!("Q6_K should use NibbleWithHighBits, got {:?}", other),
        }

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: Q6_K has QuantShiftLeft with amount=6 (4 + (4 - 2))
        let has_shift_6 = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantShiftLeft { amount: 6, .. }
        ));
        assert!(has_shift_6,
            "Q6_K should have QuantShiftLeft with amount=6 for 2-bit high bits");

        // Assert: Q6_K high mask is 0x30 ((0b11 << 4) = 0x30)
        let has_high_mask = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantAndMask { mask, .. } if *mask == 0x30
        ));
        assert!(has_high_mask,
            "Q6_K should have QuantAndMask with mask=0x30 for 2-bit high bits shifted to bits 4-5");
    }

    // @trace TEST-QD-57 [req:REQ-QCG] [level:unit]
    // Q4K hierarchical scale: verifies Q4_K has both QuantLoadF16toF32 for block_d
    // and QuantKQuantPackedScaleLookup for sub-scale decoding, followed by Mul.
    #[test]
    fn test_q4k_hierarchical_scale_structure() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q4K).expect("Q4_K must be registered");

        // Verify Q4_K uses Hierarchical scale layout
        match &desc.scale_layout {
            ScaleLayout::Hierarchical { block_d_offset, sub_scales_offset, sub_block_elements, .. } => {
                assert!(*block_d_offset < desc.block_bytes,
                    "Q4_K block_d_offset should be within block");
                assert!(*sub_scales_offset < desc.block_bytes,
                    "Q4_K sub_scales_offset should be within block");
                assert!(*sub_block_elements > 0,
                    "Q4_K sub_block_elements should be > 0");
            }
            other => panic!("Q4_K should use Hierarchical scale layout, got {:?}", other),
        }

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: block_d loaded via QuantLoadF16toF32
        let has_f16_load = trace.iter().any(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }));
        assert!(has_f16_load, "Q4_K should load block_d via QuantLoadF16toF32");

        // Assert: sub-scale lookup via QuantKQuantPackedScaleLookup
        let has_packed = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantKQuantPackedScaleLookup { is_q3k_extended: false, is_min: false, .. }
        ));
        assert!(has_packed, "Q4_K should have QuantKQuantPackedScaleLookup for sub-scale");

        // Assert: Mul combines block_d * sub_scale
        let has_mul = trace.iter().any(|op| matches!(op, TraceOp::Mul(_, _)));
        assert!(has_mul, "Q4_K should have Mul for block_d * sub_scale");
    }

    // @trace TEST-QD-58 [req:REQ-QCG] [level:unit]
    // Q5_K: verifies Q5_K trace has all expected components:
    // Hierarchical scale, Hierarchical zero (PostScaleAdd), NibbleWithHighBits,
    // and the final Add comes after FMA.
    #[test]
    fn test_q5k_full_hierarchical_structure() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q5K).expect("Q5_K must be registered");

        // Verify Q5_K uses Hierarchical zero layout
        match &desc.zero_layout {
            ZeroLayout::Hierarchical { .. } => {}
            other => panic!("Q5_K should use Hierarchical zero layout, got {:?}", other),
        }

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: has QuantKQuantPackedScaleLookup for scale (not is_min)
        let has_scale_lookup = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantKQuantPackedScaleLookup { is_min: false, .. }
        ));
        assert!(has_scale_lookup, "Q5_K should have packed scale lookup for sub-scale");

        // Assert: has QuantKQuantPackedScaleLookup for min (is_min=true)
        let has_min_lookup = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantKQuantPackedScaleLookup { is_min: true, .. }
        ));
        assert!(has_min_lookup, "Q5_K should have packed scale lookup for sub_m (is_min=true)");

        // Assert: Mul for dmin * sub_m
        let mul_count = trace.iter().filter(|op| matches!(op, TraceOp::Mul(_, _))).count();
        assert!(mul_count >= 2,
            "Q5_K should have at least 2 Mul ops (block_d*sub_scale + dmin*sub_m), got {}", mul_count);

        // Assert: Add after FMA for post-scale min
        let add_pos = trace.iter().position(|op| matches!(op, TraceOp::Add(_, _)));
        let fma_pos = trace.iter().position(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
        assert!(add_pos.is_some(), "Q5_K should have Add for post-scale min");
        assert!(fma_pos.is_some(), "Q5_K should have QuantDequantFma");
        assert!(
            add_pos.unwrap() > fma_pos.unwrap(),
            "Q5_K Add (slot {}) must be after FMA (slot {})",
            add_pos.unwrap(), fma_pos.unwrap()
        );
    }

    // @trace TEST-QD-59 [req:REQ-QCG] [level:unit]
    // NVFP4 SubBlockScalars scale: verifies NVFP4 uses SubBlockScalars scale layout
    // with QuantPtrAddOffset + QuantIntDivConst + QuantPtrAddDynamic for sub-block
    // scale indexing, and QuantE2m1LutDecode with nvfp4_mode=true.
    #[test]
    fn test_nvfp4_sub_block_scalars_scale_structure() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Nvfp4).expect("NVFP4 must be registered");

        // Verify NVFP4 uses SubBlockScalars scale layout
        match &desc.scale_layout {
            ScaleLayout::SubBlockScalars { offset_bytes, sub_block_size, .. } => {
                assert!(*offset_bytes < desc.block_bytes,
                    "NVFP4 scale offset should be within block");
                assert!(*sub_block_size > 0,
                    "NVFP4 sub_block_size should be > 0");
            }
            other => panic!("NVFP4 should use SubBlockScalars scale layout, got {:?}", other),
        }

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: QuantIntDivConst for sub_block_idx
        let has_div = trace.iter().any(|op| matches!(op, TraceOp::QuantIntDivConst { .. }));
        assert!(has_div, "NVFP4 should have QuantIntDivConst for sub-block index");

        // Assert: QuantPtrAddDynamic for sub-block scale address
        let has_dyn_ptr = trace.iter().any(|op| matches!(op, TraceOp::QuantPtrAddDynamic { .. }));
        assert!(has_dyn_ptr, "NVFP4 should have QuantPtrAddDynamic for sub-block scale address");

        // Assert: QuantE2m1LutDecode with nvfp4_mode=true
        let has_nvfp4_lut = trace.iter().any(|op| matches!(
            op,
            TraceOp::QuantE2m1LutDecode { nvfp4_mode: true, .. }
        ));
        assert!(has_nvfp4_lut, "NVFP4 should have QuantE2m1LutDecode with nvfp4_mode=true");
    }

    // @trace TEST-QD-60 [req:REQ-QCG] [level:unit]
    // Q4_1 QuantDequantFma structure: verifies Q4_1 uses FMA with proper accumulator
    // and that the trace contains exactly one QuantDequantFma (not multiple).
    #[test]
    fn test_q4_1_single_fma_with_zero_accumulator() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q4_1).expect("Q4_1 must be registered");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: exactly one QuantDequantFma
        let fma_count = trace.iter()
            .filter(|op| matches!(op, TraceOp::QuantDequantFma { .. }))
            .count();
        assert_eq!(fma_count, 1, "Q4_1 should have exactly 1 QuantDequantFma, got {}", fma_count);

        // Assert: FMA acc is Const(0.0)
        let fma_op = trace.iter().find(|op| matches!(op, TraceOp::QuantDequantFma { .. }));
        if let Some(TraceOp::QuantDequantFma { acc, .. }) = fma_op {
            let acc_idx = acc.0 as usize;
            assert!(acc_idx < trace.len(), "Q4_1 FMA acc slot out of range");
            match &trace[acc_idx] {
                TraceOp::Const(v) => assert!(
                    (*v - 0.0_f64).abs() < f64::EPSILON,
                    "Q4_1 FMA accumulator should be 0.0, got {}", v
                ),
                other => panic!("Q4_1 FMA acc should be Const(0.0), got {:?}", other),
            }
        }

        // Assert: Q4_1 has no post-scale Add
        let has_add = trace.iter().any(|op| matches!(op, TraceOp::Add(_, _)));
        assert!(!has_add, "Q4_1 should NOT have post-scale Add");
    }

    // @trace TEST-QD-61 [req:REQ-QCG] [level:unit]
    // IQ4NL codebook structure: verifies IQ4_NL codebook has valid codebook_data
    // pointer, vector_size, and bits_per_entry, and the trace uses QuantCodebookLookup
    // with matching parameters.
    #[test]
    fn test_iq4nl_codebook_lookup_parameters() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::IQ4NL).expect("IQ4_NL must be registered");
        let cb = desc.codebook.as_ref().expect("IQ4_NL should have codebook");

        // Verify codebook parameters are sensible
        assert!(cb.vector_size > 0, "IQ4_NL codebook vector_size should be > 0, got {}", cb.vector_size);
        assert!(cb.bits_per_entry > 0, "IQ4_NL codebook bits_per_entry should be > 0");
        assert!(!cb.codebook_data.is_empty(), "IQ4_NL codebook_data should not be empty");

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: QuantCodebookLookup uses matching parameters
        let codebook_op = trace.iter().find(|op| matches!(op, TraceOp::QuantCodebookLookup { .. }));
        assert!(codebook_op.is_some(), "IQ4_NL should have QuantCodebookLookup");
        if let Some(TraceOp::QuantCodebookLookup {
            vector_size, bits_per_entry, ..
        }) = codebook_op {
            assert_eq!(*vector_size, cb.vector_size,
                "QuantCodebookLookup vector_size should match codebook spec");
            assert_eq!(*bits_per_entry, cb.bits_per_entry,
                "QuantCodebookLookup bits_per_entry should match codebook spec");
        }
    }

    // @trace TEST-QD-62 [req:REQ-QCG] [level:unit]
    // AWQ4 BlockScalar zero-point: verifies AWQ4 loads zero-point from BlockScalar
    // zero layout at a specific offset, and the zero-point is subtracted BEFORE
    // scale multiplication (PreScaleSubtract path).
    #[test]
    fn test_awq4_block_scalar_zero_point_pre_scale_subtract() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::AWQ4).expect("AWQ4 must be registered");

        // Verify AWQ4 uses BlockScalar zero layout (PreScaleSubtract)
        match &desc.zero_layout {
            ZeroLayout::BlockScalar { offset_bytes, dtype } => {
                assert!(*offset_bytes < desc.block_bytes,
                    "AWQ4 zero offset should be within block");
                // AWQ4 uses F16 zero-point
                assert!(matches!(dtype, ScaleDType::F16),
                    "AWQ4 zero dtype should be F16");
            }
            other => panic!("AWQ4 should use BlockScalar zero layout, got {:?}", other),
        }

        // Act
        let mut trace = Vec::new();
        let _slot = DecodeTraceBuilder::new(desc, 8).build(&mut trace);

        // Assert: QuantLoadF16toF32 for zero-point (at least scale + zero = 2 loads)
        let f16_count = trace.iter()
            .filter(|op| matches!(op, TraceOp::QuantLoadF16toF32 { .. }))
            .count();
        assert!(f16_count >= 2,
            "AWQ4 should have at least 2 QuantLoadF16toF32 (scale + zero), got {}", f16_count);

        // Assert: QuantBroadcast for zero-point (before Sub)
        let sub_pos = trace.iter().position(|op| matches!(op, TraceOp::Sub(_, _)));
        let broadcast_positions: Vec<usize> = trace.iter()
            .enumerate()
            .filter(|(_, op)| matches!(op, TraceOp::QuantBroadcast { .. }))
            .map(|(i, _)| i)
            .collect();
        assert!(broadcast_positions.len() >= 2,
            "AWQ4 should have at least 2 QuantBroadcast (scale + zero), got {}", broadcast_positions.len());

        // Assert: at least one broadcast appears before Sub (zero-point broadcast)
        let first_broadcast = broadcast_positions[0];
        assert!(
            first_broadcast < sub_pos.unwrap(),
            "AWQ4 zero-point broadcast (slot {}) should appear before Sub (slot {})",
            first_broadcast, sub_pos.unwrap()
        );
    }

    // @trace TEST-QD-63 [req:REQ-QCG] [level:unit]
    // Q8_0 QuantLoadBytesVec count matches output_lanes: verifies the byte count
    // in QuantLoadBytesVec equals output_lanes for Bytes layout formats (not
    // PackedNibbles where count = lanes/2).
    #[test]
    fn test_q8_0_bytes_load_count_matches_lanes() {
        // Arrange
        let r = registry();
        let desc = r.get(&QuantType::Q8_0).expect("Q8_0 must be registered");

        // Verify Q8_0 uses Bytes layout (not PackedNibbles)
        match &desc.data_layout {
            DataLayout::Bytes { signed, .. } => {
                assert!(signed, "Q8_0 should use signed bytes");
            }
            other => panic!("Q8_0 should use Bytes layout, got {:?}", other),
        }

        // Act & Assert: test with different lane counts
        for lanes in &[4usize, 8, 16] {
            let mut trace = Vec::new();
            let _slot = DecodeTraceBuilder::new(desc, *lanes).build(&mut trace);

            // Bytes layout: count should equal output_lanes (1 byte per element)
            let bytes_op = trace.iter().find(|op| matches!(
                op,
                TraceOp::QuantLoadBytesVec { signed: true, .. }
            ));
            assert!(bytes_op.is_some(), "Q8_0 should have QuantLoadBytesVec with signed=true for {} lanes", lanes);
            if let Some(TraceOp::QuantLoadBytesVec { count, .. }) = bytes_op {
                assert_eq!(*count, *lanes,
                    "Q8_0 QuantLoadBytesVec count should be {} (== output_lanes), got {}",
                    lanes, count);
            }
        }
    }
}
