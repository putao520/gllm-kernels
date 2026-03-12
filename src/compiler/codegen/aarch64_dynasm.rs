//! AArch64 JIT code generator backed by dynasm-rs.
//!
//! Feature-gated behind `jit-aarch64`. Provides the `MachineCodeEmitter` /
//! `PlatformBackend` interface. Uses `dynasm!` macros for instruction emission
//! and automatic label resolution — eliminating manual branch-offset patching.
//!
//! NEON SIMD instructions that dynasm-rs may not directly support are emitted
//! via `DynasmApi::push_u32` using proven encoding helpers.


/// Emit a minimal aarch64 stub (`ret` = 0xD65F03C0).
pub fn emit_stub() -> super::CodegenOutput {
    let mut code = Vec::with_capacity(4);
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes()); // ret
    super::CodegenOutput { code, scratchpad_bytes: 0 }
}
#[cfg(feature = "jit-aarch64")]
pub mod jit {
    use dynasm::dynasm;
    use dynasmrt::{aarch64::Assembler, DynasmApi, DynasmLabelApi};

    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::compiler::codegen::CodegenOutput;
    use crate::compiler::codegen::simd_ops::{SimdOps, VReg, BaseReg, MemOperand, Label};
    use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
    use crate::compiler::graph::{CompilerGraph, OpId, OpKind};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::trace::{ComputePattern, TraceOp};
    use crate::dispatch::DeviceProfile;
    use crate::dispatch::device_profile::IsaLevel;
    use crate::compiler::codegen::apple_amx::{emit_apple_amx_f32_gemm, apple_amx_gemm_eligible};

    /// NEON register width: 4 x f32 = 128 bits.
    const NEON_WIDTH_F32: usize = 4;
    /// Total NEON/ASIMD registers available; used by vreg_for_index() for round-robin allocation.
    const NUM_NEON_REGS: usize = 32;

    /// NEON microkernel dimensions (must match asm::aarch64::MR/NR).
    const MR: usize = 8;
    const NR: usize = 12;

    /// AArch64 JIT code generator using dynasm-rs.
    ///
    /// Uses `dynasm!` macros for instruction encoding and automatic label
    /// resolution. Register convention matches the manual backend (AAPCS64).
    pub struct DynasmAArch64CodeGen {
        ops: Assembler,
        simd_width: usize,
        blis_scratchpad_bytes: usize,
        blis_scratchpad_offset: usize,
        blis_base_offset: usize,
        label_counter: u32,
        labels: Vec<Option<usize>>,
        /// Width stack for push_width/pop_width (NEON is fixed W128).
        width_stack: Vec<crate::compiler::codegen::simd_ops::SimdWidth>,
    }

    // ── NEON encoding helpers (ported from manual backend) ─────────────

    fn encode_f32x4_binop(base: u32, rd: u8, rn: u8, rm: u8) -> u32 {
        base | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_f32x4_unary(base: u32, rd: u8, rn: u8) -> u32 {
        base | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    fn encode_mov_v(rd: u8, rn: u8) -> u32 {
        0x4EA01C00
            | ((rn as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_eor_v(rd: u8, rn: u8, rm: u8) -> u32 {
        0x6E201C00
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_fmla(rd: u8, rn: u8, rm: u8) -> u32 {
        0x4E20CC00
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_fmla_lane(rd: u8, rn: u8, rm: u8, index: u8) -> u32 {
        let h = ((index >> 1) & 1) as u32;
        let l = (index & 1) as u32;
        0x4F801000
            | (l << 21)
            | ((rm as u32 & 0xF) << 16)
            | (h << 11)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_frsqrts(rd: u8, rn: u8, rm: u8) -> u32 {
        0x4EA0FC00
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_frecps(rd: u8, rn: u8, rm: u8) -> u32 {
        0x4E20FC00
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_ld1_post(vt: u8, xn: u8) -> u32 {
        0x4CDF7800 | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
    }

    fn encode_st1_post(vt: u8, xn: u8) -> u32 {
        0x4C9F7800 | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
    }

    fn encode_fmov_one(rd: u8) -> u32 {
        0x4F03F600 | (rd as u32 & 0x1F)
    }

    fn encode_fmov_half(rd: u8) -> u32 {
        0x4F03F400 | (rd as u32 & 0x1F)
    }

    fn encode_dup_4s_gp(vd: u8, xn: u8) -> u32 {
        0x4E040C00 | ((xn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    fn encode_ushr_4s(rd: u8, rn: u8, shift: u8) -> u32 {
        let immh_immb = (64 - shift as u32) & 0x7F;
        0x6F000400
            | (immh_immb << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_and_v(rd: u8, rn: u8, rm: u8) -> u32 {
        0x4E201C00
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_orr_v(rd: u8, rn: u8, rm: u8) -> u32 {
        0x4EA01C00
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    fn encode_ucvtf_4s(rd: u8, rn: u8) -> u32 {
        0x6E21D800 | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    fn encode_str_q(vt: u8, xn: u8, byte_offset: u32) -> u32 {
        let imm12 = (byte_offset / 16) & 0xFFF;
        0x3D800000 | (imm12 << 10) | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
    }

    fn encode_ldr_q(vt: u8, xn: u8, byte_offset: u32) -> u32 {
        let imm12 = (byte_offset / 16) & 0xFFF;
        0x3DC00000 | (imm12 << 10) | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
    }

    /// ZIP1 Vd.4S, Vn.4S, Vm.4S — interleave even elements
    fn encode_zip1_4s(rd: u8, rn: u8, rm: u8) -> u32 {
        0x4E803800
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    /// ZIP2 Vd.4S, Vn.4S, Vm.4S — interleave odd elements
    fn encode_zip2_4s(rd: u8, rn: u8, rm: u8) -> u32 {
        0x4E807800
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    /// REV64 Vd.4S, Vn.4S — reverse elements within 64-bit lanes (swaps adjacent pairs)
    fn encode_rev64_4s(rd: u8, rn: u8) -> u32 {
        0x4EA00800
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    // ── Scalar float helpers ────────────────────────────────────────

    /// FMUL Sd, Sn, Sm — scalar single-precision multiply
    fn encode_fmul_s(rd: u8, rn: u8, rm: u8) -> u32 {
        0x1E200800
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    /// FSUB Sd, Sn, Sm — scalar single-precision subtract
    fn encode_fsub_s(rd: u8, rn: u8, rm: u8) -> u32 {
        0x1E203800
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    /// FMAXNM Sd, Sn, Sm — scalar single-precision max (NaN-propagating)
    fn encode_fmaxnm_s(rd: u8, rn: u8, rm: u8) -> u32 {
        0x1E206800
            | ((rm as u32 & 0x1F) << 16)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    /// FSQRT Sd, Sn — scalar single-precision square root
    fn encode_fsqrt_s(rd: u8, rn: u8) -> u32 {
        0x1E21C000
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    /// FMADD Sd, Sn, Sm, Sa — scalar fused multiply-add: Sd = Sa + Sn * Sm
    fn encode_fmadd_s(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
        0x1F000000
            | ((rm as u32 & 0x1F) << 16)
            | ((ra as u32 & 0x1F) << 10)
            | ((rn as u32 & 0x1F) << 5)
            | (rd as u32 & 0x1F)
    }

    // ── Helpers: push a raw u32 instruction ────────────────────────────

    /// Push a raw 32-bit instruction word into the assembler.
    fn emit_raw(ops: &mut Assembler, insn: u32) {
        // dynasmrt stores aarch64 instructions as little-endian u32
        let bytes = insn.to_le_bytes();
        ops.push(bytes[0]);
        ops.push(bytes[1]);
        ops.push(bytes[2]);
        ops.push(bytes[3]);
    }

    impl DynasmAArch64CodeGen {
        pub fn new(_profile: &DeviceProfile) -> Self {
            DynasmAArch64CodeGen {
                ops: Assembler::new().expect("failed to create aarch64 assembler"),
                simd_width: NEON_WIDTH_F32,
                blis_scratchpad_bytes: 0,
                blis_scratchpad_offset: 0,
                blis_base_offset: 0,
                label_counter: 0,
                labels: Vec::new(),
                width_stack: Vec::new(),
            }
        }

        pub fn simd_width(&self) -> usize {
            self.simd_width
        }

        pub fn code_len(&self) -> usize {
            self.ops.offset().0
        }

        // ── Prologue / Epilogue ────────────────────────────────────────

        fn emit_prologue(&mut self) {
            dynasm!(self.ops
                ; .arch aarch64
                ; stp x29, x30, [sp, #-0x10]!
                ; mov x29, sp
            );
        }

        fn emit_epilogue(&mut self) {
            dynasm!(self.ops
                ; .arch aarch64
                ; ldp x29, x30, [sp], #0x10
                ; ret
            );
        }

        fn emit_save_callee_saved(&mut self) {
            dynasm!(self.ops
                ; .arch aarch64
                ; stp x19, x20, [sp, #-0x50]!
                ; stp x21, x22, [sp, #0x10]
                ; stp x23, x24, [sp, #0x20]
                ; stp x25, x26, [sp, #0x30]
                ; stp x27, x28, [sp, #0x40]
            );
        }

        fn emit_restore_callee_saved(&mut self) {
            dynasm!(self.ops
                ; .arch aarch64
                ; ldp x27, x28, [sp, #0x40]
                ; ldp x25, x26, [sp, #0x30]
                ; ldp x23, x24, [sp, #0x20]
                ; ldp x21, x22, [sp, #0x10]
                ; ldp x19, x20, [sp], #0x50
            );
        }

        // ── GP helpers using dynasm! ───────────────────────────────────

        fn emit_mov_imm_gp(&mut self, rd: u32, val: usize) {
            let v = val as u64;
            let imm0 = (v & 0xFFFF) as u32;
            // MOVZ Xd, #imm16
            emit_raw(&mut self.ops, 0xD2800000 | (imm0 << 5) | rd);
            if v > 0xFFFF {
                let imm1 = ((v >> 16) & 0xFFFF) as u32;
                // MOVK Xd, #imm16, LSL #16
                emit_raw(&mut self.ops, 0xF2A00000 | (imm1 << 5) | rd);
            }
            if v > 0xFFFF_FFFF {
                let imm2 = ((v >> 32) & 0xFFFF) as u32;
                // MOVK Xd, #imm16, LSL #32
                emit_raw(&mut self.ops, 0xF2C00000 | (imm2 << 5) | rd);
            }
            if v > 0xFFFF_FFFF_FFFF {
                let imm3 = ((v >> 48) & 0xFFFF) as u32;
                // MOVK Xd, #imm16, LSL #48
                emit_raw(&mut self.ops, 0xF2E00000 | (imm3 << 5) | rd);
            }
        }

        fn emit_add_offset_gp(&mut self, rd: u32, rn: u32, offset: usize) {
            if offset == 0 {
                if rd != rn {
                    // MOV Xd, Xn  (alias for ORR Xd, XZR, Xn)
                    emit_raw(&mut self.ops, 0xAA0003E0 | (rn << 16) | rd);
                }
            } else if offset < 4096 {
                let imm12 = offset as u32;
                // ADD Xd, Xn, #imm12
                emit_raw(&mut self.ops, 0x91000000 | (imm12 << 10) | (rn << 5) | rd);
            } else {
                self.emit_mov_imm_gp(15, offset);
                // ADD Xd, Xn, X15
                emit_raw(&mut self.ops, 0x8B0F0000 | (rn << 5) | rd);
            }
        }

        // ── NEON helpers (push_u32 for SIMD instructions) ─────────────

        fn emit_load_f32_const_neon(&mut self, vd: u8, val: f32) {
            let bits = val.to_bits();
            let lo = bits as u16;
            let hi = (bits >> 16) as u16;
            if hi == 0 {
                dynasm!(self.ops ; .arch aarch64 ; movz w9, lo as u32);
            } else if lo == 0 {
                dynasm!(self.ops ; .arch aarch64 ; movz w9, hi as u32, lsl #16);
            } else {
                dynasm!(self.ops ; .arch aarch64 ; movz w9, lo as u32);
                dynasm!(self.ops ; .arch aarch64 ; movk w9, hi as u32, lsl #16);
            }
            emit_raw(&mut self.ops, encode_dup_4s_gp(vd, 9));
        }

        fn emit_exp_neon(&mut self, rd: u8, ra: u8) {
            let scratch = [VReg(28), VReg(29), VReg(30)];
            crate::compiler::codegen::math_approx::emit_exp(self, VReg(rd), VReg(ra), scratch).unwrap();
        }

        fn emit_rsqrt_refined_neon(&mut self, rd: u8, ra: u8) {
            let (s0, s1) = (28u8, 29u8);
            emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA1D800, rd, ra)); // frsqrte
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, s0, rd, rd));
            emit_raw(&mut self.ops, encode_frsqrts(s1, ra, s0));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, rd, s1));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, s0, rd, rd));
            emit_raw(&mut self.ops, encode_frsqrts(s1, ra, s0));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, rd, s1));
        }

        fn emit_recip_refined_neon(&mut self, rd: u8, ra: u8) {
            let s0 = 28u8;
            emit_raw(&mut self.ops, encode_f32x4_unary(0x4EA1D800, rd, ra)); // frecpe
            emit_raw(&mut self.ops, encode_frecps(s0, ra, rd));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, rd, s0));
        }

        fn emit_tanh_neon(&mut self, rd: u8, ra: u8) {
            let scratch = [VReg(28), VReg(29), VReg(30)];
            crate::compiler::codegen::math_approx::emit_tanh(self, VReg(rd), VReg(ra), scratch).unwrap();
        }

        fn emit_log_neon(&mut self, rd: u8, ra: u8) {
            let scratch = [VReg(28), VReg(29), VReg(30)];
            crate::compiler::codegen::math_approx::emit_log(self, VReg(rd), VReg(ra), scratch).unwrap();
        }

        // ── Scratch-based NEON helpers (for epilogue on accumulators) ──

        fn emit_exp_scratch_neon(&mut self) {
            let scratch = [VReg(1), VReg(2), VReg(3)];
            crate::compiler::codegen::math_approx::emit_exp(self, VReg(0), VReg(0), scratch).unwrap();
        }

        fn emit_rsqrt_scratch_neon(&mut self) {
            emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA1D800, 1, 0));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 2, 1, 1));
            emit_raw(&mut self.ops, encode_frsqrts(3, 0, 2));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 1, 1, 3));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 2, 1, 1));
            emit_raw(&mut self.ops, encode_frsqrts(3, 0, 2));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 1, 1, 3));
            emit_raw(&mut self.ops, encode_mov_v(0, 1));
        }

        fn emit_recip_scratch_neon(&mut self) {
            emit_raw(&mut self.ops, encode_f32x4_unary(0x4EA1D800, 1, 0));
            emit_raw(&mut self.ops, encode_frecps(2, 0, 1));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 1, 1, 2));
            emit_raw(&mut self.ops, encode_mov_v(0, 1));
        }

        fn emit_tanh_scratch_neon(&mut self) {
            let scratch = [VReg(1), VReg(2), VReg(3)];
            crate::compiler::codegen::math_approx::emit_tanh(self, VReg(0), VReg(0), scratch).unwrap();
        }

        fn emit_log_scratch_neon(&mut self) {
            let scratch = [VReg(1), VReg(2), VReg(3)];
            crate::compiler::codegen::math_approx::emit_log(self, VReg(0), VReg(0), scratch).unwrap();
        }

        // ── TraceOp → NEON instruction emission ───────────────────────

        fn vreg_for_index(index: usize) -> u8 {
            (index % NUM_NEON_REGS) as u8
        }

        /// Emit NEON instructions for a sequence of TraceOps.
        /// Returns the NEON register number assigned to each TraceOp result.
        pub fn emit_trace_ops_neon(
            &mut self,
            ops_trace: &[TraceOp],
        ) -> Result<Vec<u8>, String> {
            let scratch = [VReg(28), VReg(29), VReg(30)];
            let mut reg_map: Vec<VReg> = Vec::with_capacity(ops_trace.len());

            for (i, op) in ops_trace.iter().enumerate() {
                let rd = Self::vreg_for_index(i);
                let dst = VReg(rd);
                match op {
                    TraceOp::Input(n) => {
                        // Input(n) is pre-loaded into v_n by caller, no instruction needed
                        debug_assert_eq!(rd, *n as u8);
                    }
                    _ => {
                        crate::compiler::codegen::algorithm::emit_trace_op(self, op, dst, &reg_map, scratch)?;
                    }
                }
                reg_map.push(dst);
            }

            Ok(reg_map.iter().map(|v| v.0).collect())
        }

        // ── Elementwise loop (dynasm! labels for branch management) ───

        /// Emit a vectorized elementwise loop over `elem_count` f32 elements.
        ///
        /// Uses dynasm! labels for automatic branch offset resolution —
        /// the key advantage over the manual backend's `patch_bcond_ge`.
        pub fn emit_elementwise_loop_neon(
            &mut self,
            body: &[TraceOp],
            elem_count: usize,
            is_binary: bool,
        ) -> Result<(), String> {
            // x10 = input ptr, x11 = output ptr, x12 = second input (if binary)
            dynasm!(self.ops ; .arch aarch64 ; mov x10, x0);
            dynasm!(self.ops ; .arch aarch64 ; mov x11, x7);
            if is_binary {
                dynasm!(self.ops ; .arch aarch64 ; mov x12, x1);
            }

            let vec_iters = elem_count / NEON_WIDTH_F32;
            let remainder = elem_count % NEON_WIDTH_F32;

            if vec_iters > 0 {
                self.emit_mov_imm_gp(9, vec_iters);

                let loop_top = self.ops.new_dynamic_label();
                let loop_exit = self.ops.new_dynamic_label();

                dynasm!(self.ops
                    ; .arch aarch64
                    ; =>loop_top
                    ; cbz x9, =>loop_exit
                );

                emit_raw(&mut self.ops, encode_ld1_post(0, 10));
                if is_binary {
                    emit_raw(&mut self.ops, encode_ld1_post(1, 12));
                }

                let regs = self.emit_trace_ops_neon(body)?;
                let result_reg = *regs.last().ok_or("empty trace body")?;

                emit_raw(&mut self.ops, encode_st1_post(result_reg, 11));

                dynasm!(self.ops
                    ; .arch aarch64
                    ; sub x9, x9, #1
                    ; b =>loop_top
                    ; =>loop_exit
                );
            }

            if remainder > 0 {
                self.emit_mov_imm_gp(9, remainder);

                let scalar_top = self.ops.new_dynamic_label();
                let scalar_exit = self.ops.new_dynamic_label();

                dynasm!(self.ops
                    ; .arch aarch64
                    ; =>scalar_top
                    ; cbz x9, =>scalar_exit
                );

                // ldr s0, [x10], #4
                emit_raw(&mut self.ops, 0xBC404400 | (10 << 5));
                if is_binary {
                    emit_raw(&mut self.ops, 0xBC404400 | 1 | (12 << 5));
                }

                let regs = self.emit_trace_ops_neon(body)?;
                let result_reg = *regs.last().ok_or("empty trace body")?;

                // str s<result>, [x11], #4
                emit_raw(&mut self.ops, 0xBC004400 | (result_reg as u32) | (11 << 5));

                dynasm!(self.ops
                    ; .arch aarch64
                    ; sub x9, x9, #1
                    ; b =>scalar_top
                    ; =>scalar_exit
                );
            }

            Ok(())
        }

        /// Emit a fused elementwise loop chaining multiple TraceOp bodies.
        fn emit_fused_elementwise_loop_neon(
            &mut self,
            bodies: &[&[TraceOp]],
            elem_count: usize,
            is_binary: bool,
        ) -> Result<(), String> {
            if bodies.is_empty() {
                return Ok(());
            }
            if bodies.len() == 1 {
                return self.emit_elementwise_loop_neon(bodies[0], elem_count, is_binary);
            }

            dynasm!(self.ops ; .arch aarch64 ; mov x10, x0);
            dynasm!(self.ops ; .arch aarch64 ; mov x11, x7);
            if is_binary {
                dynasm!(self.ops ; .arch aarch64 ; mov x12, x1);
            }

            let vec_iters = elem_count / NEON_WIDTH_F32;
            let remainder = elem_count % NEON_WIDTH_F32;

            if vec_iters > 0 {
                self.emit_mov_imm_gp(9, vec_iters);
                let loop_top = self.ops.new_dynamic_label();
                let loop_exit = self.ops.new_dynamic_label();

                dynasm!(self.ops ; .arch aarch64 ; =>loop_top ; cbz x9, =>loop_exit);

                emit_raw(&mut self.ops, encode_ld1_post(0, 10));
                if is_binary {
                    emit_raw(&mut self.ops, encode_ld1_post(1, 12));
                }

                let mut last_result_reg = 0u8;
                for body in bodies {
                    let regs = self.emit_trace_ops_neon(body)?;
                    last_result_reg = *regs.last().ok_or("empty trace body")?;
                    if last_result_reg != 0 {
                        emit_raw(&mut self.ops, encode_mov_v(0, last_result_reg));
                    }
                }

                emit_raw(&mut self.ops, encode_st1_post(last_result_reg, 11));

                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>loop_top ; =>loop_exit);
            }

            if remainder > 0 {
                self.emit_mov_imm_gp(9, remainder);
                let scalar_top = self.ops.new_dynamic_label();
                let scalar_exit = self.ops.new_dynamic_label();

                dynasm!(self.ops ; .arch aarch64 ; =>scalar_top ; cbz x9, =>scalar_exit);

                emit_raw(&mut self.ops, 0xBC404400 | (10 << 5));
                if is_binary {
                    emit_raw(&mut self.ops, 0xBC404400 | 1 | (12 << 5));
                }

                let mut last_result_reg = 0u8;
                for body in bodies {
                    let regs = self.emit_trace_ops_neon(body)?;
                    last_result_reg = *regs.last().ok_or("empty trace body")?;
                    if last_result_reg != 0 {
                        emit_raw(&mut self.ops, encode_mov_v(0, last_result_reg));
                    }
                }

                emit_raw(&mut self.ops, 0xBC004400 | (last_result_reg as u32) | (11 << 5));

                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>scalar_top ; =>scalar_exit);
            }

            Ok(())
        }

        // ── Norm / Reduce standalone ──────────────────────────────────

        fn emit_norm_standalone_neon(
            &mut self,
            reduce: &[TraceOp],
            _finalize: &[TraceOp],
            transform: &[TraceOp],
            elem_count: usize,
            eps: f32,
        ) -> Result<(), String> {
            dynasm!(self.ops ; .arch aarch64 ; mov x10, x0 ; mov x11, x7);

            let vec_iters = elem_count / NEON_WIDTH_F32;
            let remainder = elem_count % NEON_WIDTH_F32;

            // v16 = accumulator (zeroed)
            emit_raw(&mut self.ops, encode_eor_v(16, 16, 16));
            dynasm!(self.ops ; .arch aarch64 ; mov x13, x10);

            // Pass 1: reduce
            if vec_iters > 0 {
                self.emit_mov_imm_gp(9, vec_iters);
                let lp = self.ops.new_dynamic_label();
                let le = self.ops.new_dynamic_label();
                dynasm!(self.ops ; .arch aarch64 ; =>lp ; cbz x9, =>le);
                emit_raw(&mut self.ops, encode_ld1_post(0, 13));
                let regs = self.emit_trace_ops_neon(reduce)?;
                let rr = *regs.last().ok_or("empty reduce body")?;
                emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 16, 16, rr));
                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>lp ; =>le);
            }
            if remainder > 0 {
                self.emit_mov_imm_gp(9, remainder);
                let lp = self.ops.new_dynamic_label();
                let le = self.ops.new_dynamic_label();
                dynasm!(self.ops ; .arch aarch64 ; =>lp ; cbz x9, =>le);
                emit_raw(&mut self.ops, 0xBC404400 | (13 << 5));
                let regs = self.emit_trace_ops_neon(reduce)?;
                let rr = *regs.last().ok_or("empty reduce body")?;
                emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 16, 16, rr));
                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>lp ; =>le);
            }

            // Horizontal add v16
            emit_raw(&mut self.ops, 0x6E30D600 | (16 << 5) | 16); // faddp
            emit_raw(&mut self.ops, 0x6E30D600 | (16 << 5) | 16); // faddp

            // Multiply by 1/N
            self.emit_load_f32_const_neon(17, 1.0 / elem_count as f32);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 16, 16, 17));

            // Add eps
            self.emit_load_f32_const_neon(17, eps);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 16, 16, 17));

            // rsqrt
            self.emit_rsqrt_refined_neon(16, 16);

            // Broadcast scalar to all lanes
            emit_raw(&mut self.ops, 0x4E040400 | (16 << 5) | 16);

            // Pass 2: transform (multiply by rsqrt)
            dynasm!(self.ops ; .arch aarch64 ; mov x13, x10);
            if vec_iters > 0 {
                self.emit_mov_imm_gp(9, vec_iters);
                let lp = self.ops.new_dynamic_label();
                let le = self.ops.new_dynamic_label();
                dynasm!(self.ops ; .arch aarch64 ; =>lp ; cbz x9, =>le);
                emit_raw(&mut self.ops, encode_ld1_post(0, 13));
                emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 17, 0, 16));
                emit_raw(&mut self.ops, encode_st1_post(17, 11));
                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>lp ; =>le);
            }
            if remainder > 0 {
                self.emit_mov_imm_gp(9, remainder);
                let lp = self.ops.new_dynamic_label();
                let le = self.ops.new_dynamic_label();
                dynasm!(self.ops ; .arch aarch64 ; =>lp ; cbz x9, =>le);
                emit_raw(&mut self.ops, 0xBC404400 | (13 << 5));
                // scalar fmul s17, s0, s16
                emit_raw(&mut self.ops, 0x1E300800 | (16 << 16) | (0 << 5) | 17);
                emit_raw(&mut self.ops, 0xBC004400 | 17 | (11 << 5));
                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>lp ; =>le);
            }

            Ok(())
        }

        fn emit_reduce_standalone_neon(
            &mut self,
            combine: &[TraceOp],
            elem_count: usize,
            identity: f64,
        ) -> Result<(), String> {
            dynasm!(self.ops ; .arch aarch64 ; mov x10, x0);

            let vec_iters = elem_count / NEON_WIDTH_F32;
            let remainder = elem_count % NEON_WIDTH_F32;

            self.emit_load_f32_const_neon(16, identity as f32);

            if vec_iters > 0 {
                self.emit_mov_imm_gp(9, vec_iters);
                let lp = self.ops.new_dynamic_label();
                let le = self.ops.new_dynamic_label();
                dynasm!(self.ops ; .arch aarch64 ; =>lp ; cbz x9, =>le);
                emit_raw(&mut self.ops, encode_ld1_post(0, 10));
                let regs = self.emit_trace_ops_neon(combine)?;
                let cr = *regs.last().ok_or("empty combine body")?;
                emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 16, 16, cr));
                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>lp ; =>le);
            }
            if remainder > 0 {
                self.emit_mov_imm_gp(9, remainder);
                let lp = self.ops.new_dynamic_label();
                let le = self.ops.new_dynamic_label();
                dynasm!(self.ops ; .arch aarch64 ; =>lp ; cbz x9, =>le);
                emit_raw(&mut self.ops, 0xBC404400 | (10 << 5));
                let regs = self.emit_trace_ops_neon(combine)?;
                let cr = *regs.last().ok_or("empty combine body")?;
                emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 16, 16, cr));
                dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, #1 ; b =>lp ; =>le);
            }

            // Horizontal add
            emit_raw(&mut self.ops, 0x6E30D600 | (16 << 5) | 16);
            emit_raw(&mut self.ops, 0x6E30D600 | (16 << 5) | 16);

            // str s16, [x7]
            emit_raw(&mut self.ops, 0xBD000000 | (7 << 5) | 16);

            Ok(())
        }


        // ── RoPE (Rotary Position Embedding) ─────────────────────────

        /// Emit NEON RoPE: non-interleaved rotary position embedding.
        ///
        /// ABI: x0 = input ptr [n_heads * head_dim f32],
        ///      x1 = cos table ptr [head_dim/2 f32],
        ///      x7 = output ptr [n_heads * head_dim f32].
        ///
        /// For each head, pairs (x[2i], x[2i+1]) are rotated by (cos[i], sin[i]):
        ///   out[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
        ///   out[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]
        /// where sin[i] = sqrt(max(0, 1 - cos[i]²)).
        fn emit_rope_standalone_neon(
            &mut self,
            head_dim: usize,
            n_heads: usize,
        ) -> Result<(), String> {
            let half = head_dim / 2;
            // NEON processes 4 cos values → 8 x floats (4 pairs) per iteration
            let simd_pairs = NEON_WIDTH_F32; // 4 pairs per v register
            let vec_iters = half / simd_pairs;

            for head in 0..n_heads {
                let head_byte_off = (head * head_dim * 4) as u64;

                for chunk in 0..vec_iters {
                    let cos_off = (chunk * simd_pairs * 4) as u64;
                    let x_off = head_byte_off + (chunk * simd_pairs * 2 * 4) as u64;

                    // Load 4 cos values into v0: ld1 {v0.4s}, [x1, #cos_off]
                    // x10 = x1 + cos_off
                    self.emit_add_offset_gp(10, 1, cos_off as usize);
                    emit_raw(&mut self.ops, 0x4C407800 | (10 << 5) | 0); // ld1 {v0.4s}, [x10]

                    // Duplicate cos to pairs: [c0,c0,c1,c1,c2,c2,c3,c3]
                    // v2 = zip1 v0.4s, v0.4s → [c0,c0,c1,c1]
                    emit_raw(&mut self.ops, encode_zip1_4s(2, 0, 0));
                    // v3 = zip2 v0.4s, v0.4s → [c2,c2,c3,c3]
                    emit_raw(&mut self.ops, encode_zip2_4s(3, 0, 0));

                    // sin = sqrt(max(0, 1 - cos²))
                    // v4 = v2 * v2 (cos² low), v5 = v3 * v3 (cos² high)
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 2, 2));
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 5, 3, 3));
                    // v6 = 1.0 broadcast
                    self.emit_load_f32_const_neon(6, 1.0);
                    // v4 = 1.0 - cos² low
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 4, 6, 4)); // fsub
                    // v5 = 1.0 - cos² high (v6 still holds 1.0)
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 5, 6, 5)); // fsub
                    // clamp negative to zero
                    emit_raw(&mut self.ops, encode_eor_v(6, 6, 6)); // v6 = 0
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20F400, 4, 4, 6)); // fmax v4, v4, v6
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20F400, 5, 5, 6)); // fmax v5, v5, v6
                    // sqrt → sin duplicated
                    emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA1F800, 4, 4)); // fsqrt v4
                    emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA1F800, 5, 5)); // fsqrt v5

                    // Build sin_signed: [-s,s,-s,s,...] by negating even lanes
                    // v6 = [-1,-1,-1,-1], v7 = [1,1,1,1]
                    self.emit_load_f32_const_neon(6, -1.0);
                    self.emit_load_f32_const_neon(7, 1.0);
                    // zip1 → v6 = [-1,1,-1,1]
                    emit_raw(&mut self.ops, encode_zip1_4s(6, 6, 7));
                    // v4 = sin_low * sign = [-s0,s0,-s1,s1]
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 4, 6)); // fmul
                    // v5 = sin_high * sign = [-s2,s2,-s3,s3]
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 5, 5, 6)); // fmul

                    // Load 8 x floats (two v regs): x[2i..2i+7]
                    // x10 = x0 + x_off
                    self.emit_add_offset_gp(10, 0, x_off as usize);
                    emit_raw(&mut self.ops, 0x4C407800 | (10 << 5) | 16); // ld1 {v16.4s}, [x10]
                    self.emit_add_offset_gp(10, 0, (x_off + 16) as usize);
                    emit_raw(&mut self.ops, 0x4C407800 | (10 << 5) | 17); // ld1 {v17.4s}, [x10]

                    // Swap adjacent pairs: [x1,x0,x3,x2] via rev64
                    // rev64 v18.4s, v16.4s → swaps pairs within 64-bit lanes
                    emit_raw(&mut self.ops, encode_rev64_4s(18, 16));
                    emit_raw(&mut self.ops, encode_rev64_4s(19, 17));

                    // result = x * cos_dup + x_swapped * sin_signed
                    // v16 = v16 * v2 (x_low * cos_low)
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 16, 16, 2)); // fmul
                    // v16 = v16 + v18 * v4 (fmla: v16 += x_swapped_low * sin_signed_low)
                    emit_raw(&mut self.ops, encode_fmla(16, 18, 4));
                    // v17 = v17 * v3 (x_high * cos_high)
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 17, 17, 3)); // fmul
                    // v17 = v17 + v19 * v5 (fmla: v17 += x_swapped_high * sin_signed_high)
                    emit_raw(&mut self.ops, encode_fmla(17, 19, 5));

                    // Store results
                    self.emit_add_offset_gp(10, 7, x_off as usize);
                    emit_raw(&mut self.ops, encode_st1_post(16, 10));
                    emit_raw(&mut self.ops, encode_st1_post(17, 10));
                }

                // Scalar tail for remaining pairs
                for i in (vec_iters * simd_pairs)..half {
                    let cos_off = (i * 4) as usize;
                    let x_off = (head_byte_off as usize) + i * 2 * 4;

                    // Load cos[i] into s0
                    self.emit_add_offset_gp(10, 1, cos_off);
                    emit_raw(&mut self.ops, 0xBD400000 | (10 << 5) | 0); // ldr s0, [x10]

                    // Load x[2i] into s16, x[2i+1] into s17
                    self.emit_add_offset_gp(10, 0, x_off);
                    emit_raw(&mut self.ops, 0xBD400000 | (10 << 5) | 16); // ldr s16, [x10]
                    self.emit_add_offset_gp(10, 0, x_off + 4);
                    emit_raw(&mut self.ops, 0xBD400000 | (10 << 5) | 17); // ldr s17, [x10]

                    // sin = sqrt(max(0, 1 - cos²))
                    // s1 = s0 * s0  (cos²)
                    emit_raw(&mut self.ops, encode_fmul_s(1, 0, 0));
                    // s2 = 1.0
                    self.emit_load_f32_const_neon(2, 1.0);
                    // s1 = s2 - s1  (1 - cos²)
                    emit_raw(&mut self.ops, encode_fsub_s(1, 2, 1));
                    // clamp: s3 = 0, s1 = fmaxnm(s1, s3)
                    emit_raw(&mut self.ops, encode_eor_v(3, 3, 3)); // v3 = 0
                    emit_raw(&mut self.ops, encode_fmaxnm_s(1, 1, 3));
                    // s1 = sqrt(s1)
                    emit_raw(&mut self.ops, encode_fsqrt_s(1, 1));

                    // out[2i] = x0*cos - x1*sin
                    emit_raw(&mut self.ops, encode_fmul_s(4, 16, 0));  // s4 = s16 * s0
                    emit_raw(&mut self.ops, encode_fmul_s(5, 17, 1));  // s5 = s17 * s1
                    emit_raw(&mut self.ops, encode_fsub_s(4, 4, 5));   // s4 = s4 - s5

                    // out[2i+1] = x1*cos + x0*sin
                    emit_raw(&mut self.ops, encode_fmul_s(5, 17, 0));  // s5 = s17 * s0
                    emit_raw(&mut self.ops, encode_fmadd_s(5, 16, 1, 5)); // s5 += s16 * s1

                    // Store
                    self.emit_add_offset_gp(10, 7, x_off);
                    emit_raw(&mut self.ops, 0xBD000000 | (10 << 5) | 4); // str s4, [x10]
                    self.emit_add_offset_gp(10, 7, x_off + 4);
                    emit_raw(&mut self.ops, 0xBD000000 | (10 << 5) | 5); // str s5, [x10]
                }
            }

            Ok(())
        }

        // ── Mean pooling ──────────────────────────────────────────────

        /// Emit NEON mean-pool: average `seq_len` rows of `hidden` f32 elements.
        ///
        /// ABI: x0 = input ptr [seq_len * hidden], x7 = output ptr [hidden].
        fn emit_mean_pool_neon(
            &mut self,
            seq_len: usize,
            hidden: usize,
        ) -> Result<(), String> {
            if seq_len == 0 || hidden == 0 {
                return Err("MeanPool: seq_len and hidden must be > 0".into());
            }

            let inv_n = 1.0f32 / seq_len as f32;
            let row_stride = hidden * 4; // bytes per row

            let vec_count = hidden / NEON_WIDTH_F32;
            let tail = hidden % NEON_WIDTH_F32;

            // ── NEON 4-wide columns ──
            // Use a runtime column loop: x10 = input col ptr, x11 = output col ptr
            if vec_count > 0 {
                dynasm!(self.ops ; .arch aarch64 ; mov x10, x0);
                dynasm!(self.ops ; .arch aarch64 ; mov x11, x7);
                self.emit_mov_imm_gp(13, vec_count); // x13 = column counter

                let col_top = self.ops.new_dynamic_label();
                let col_exit = self.ops.new_dynamic_label();

                dynasm!(self.ops
                    ; .arch aarch64
                    ; =>col_top
                    ; cbz x13, =>col_exit
                );

                // Load first row into v0: ld1 {v0.4s}, [x10]
                // (don't post-increment x10 yet — we need it as column base)
                emit_raw(&mut self.ops, 0x4C407800 | (10 << 5) | 0); // ld1 {v0.4s}, [x10]

                // Sum remaining rows: for each row r in 1..seq_len,
                // compute address = x10 + r * row_stride, load into v1, fadd into v0
                for row in 1..seq_len {
                    let off = row * row_stride;
                    // x14 = x10 + off
                    self.emit_add_offset_gp(14, 10, off);
                    // ld1 {v1.4s}, [x14]
                    emit_raw(&mut self.ops, 0x4C407800 | (14 << 5) | 1);
                    // fadd v0.4s, v0.4s, v1.4s
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 0, 0, 1));
                }

                // Multiply by 1/seq_len: load inv_n into v1, fmul v0 by v1
                self.emit_load_f32_const_neon(1, inv_n);
                emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 0, 0, 1)); // fmul v0.4s, v0.4s, v1.4s

                // Store result: st1 {v0.4s}, [x11], #16
                emit_raw(&mut self.ops, encode_st1_post(0, 11));

                // Advance input column pointer by 16 bytes
                dynasm!(self.ops ; .arch aarch64 ; add x10, x10, #16);

                // Decrement column counter and loop
                dynasm!(self.ops
                    ; .arch aarch64
                    ; sub x13, x13, #1
                    ; b =>col_top
                    ; =>col_exit
                );
            }

            // ── Scalar tail ──
            if tail > 0 {
                let base_off = vec_count * NEON_WIDTH_F32 * 4; // byte offset to tail start
                // x10 = input base + tail offset, x11 = output base + tail offset
                self.emit_add_offset_gp(10, 0, base_off);
                self.emit_add_offset_gp(11, 7, base_off);
                self.emit_mov_imm_gp(13, tail); // x13 = scalar counter

                let sc_top = self.ops.new_dynamic_label();
                let sc_exit = self.ops.new_dynamic_label();

                dynasm!(self.ops
                    ; .arch aarch64
                    ; =>sc_top
                    ; cbz x13, =>sc_exit
                );

                // ldr s0, [x10] (load first row scalar)
                emit_raw(&mut self.ops, 0xBD400000 | (10 << 5) | 0); // ldr s0, [x10]

                // Sum remaining rows
                for row in 1..seq_len {
                    let off = row * row_stride;
                    self.emit_add_offset_gp(14, 10, off);
                    // ldr s1, [x14]
                    emit_raw(&mut self.ops, 0xBD400000 | (14 << 5) | 1);
                    // fadd s0, s0, s1
                    emit_raw(&mut self.ops, 0x1E212800 | (1 << 5) | 0); // fadd s0, s0, s1
                }

                // fmul s0, s0, inv_n
                // Load inv_n into w9, fmov s1, w9, then fmul s0, s0, s1
                let bits = inv_n.to_bits();
                let lo = bits as u16;
                let hi = (bits >> 16) as u16;
                if hi == 0 {
                    dynasm!(self.ops ; .arch aarch64 ; movz w9, lo as u32);
                } else if lo == 0 {
                    dynasm!(self.ops ; .arch aarch64 ; movz w9, hi as u32, lsl #16);
                } else {
                    dynasm!(self.ops ; .arch aarch64 ; movz w9, lo as u32);
                    dynasm!(self.ops ; .arch aarch64 ; movk w9, hi as u32, lsl #16);
                }
                // fmov s1, w9
                emit_raw(&mut self.ops, 0x1E270000 | (9 << 5) | 1);
                // fmul s0, s0, s1
                emit_raw(&mut self.ops, 0x1E210800 | (1 << 5) | 0); // fmul s0, s0, s1

                // str s0, [x11], #4 (post-index store)
                emit_raw(&mut self.ops, 0xBC004400 | 0 | (11 << 5));

                // Advance input column pointer by 4 bytes
                dynasm!(self.ops ; .arch aarch64 ; add x10, x10, #4);

                dynasm!(self.ops
                    ; .arch aarch64
                    ; sub x13, x13, #1
                    ; b =>sc_top
                    ; =>sc_exit
                );
            }

            Ok(())
        }

        // ── Multi-head attention ────────────────────────────────────

        /// Emit NEON multi-head attention.
        ///
        /// ABI: x0=Q [seq,hidden], x1=K [seq,hidden], x2=V [seq,hidden],
        ///      x7=output [seq,hidden]. Scratchpad at [x29, #16].
        fn emit_multi_head_attention_neon(
            &mut self,
            seq_len: usize,
            num_heads: usize,
            head_dim: usize,
        ) -> Result<(), String> {
            let hidden = num_heads * head_dim;
            let hd_bytes = head_dim * 4;
            let hidden_bytes = hidden * 4;

            let head_mat_size = seq_len * head_dim * 4;
            let scores_size = seq_len * seq_len * 4;
            let total_scratch = head_mat_size * 4 + scores_size;

            let q_off = self.blis_scratchpad_offset;
            let k_off = q_off + head_mat_size;
            let v_off = k_off + head_mat_size;
            let sc_off = v_off + head_mat_size;
            let out_off = sc_off + scores_size;

            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset + total_scratch - self.blis_base_offset
            );

            // Save pointers to stack
            dynasm!(self.ops ; .arch aarch64 ; sub sp, sp, #48);
            dynasm!(self.ops ; .arch aarch64 ; str x0, [sp]);
            dynasm!(self.ops ; .arch aarch64 ; str x1, [sp, #8]);
            dynasm!(self.ops ; .arch aarch64 ; str x2, [sp, #16]);
            dynasm!(self.ops ; .arch aarch64 ; str x7, [sp, #24]);

            // x14 = scratchpad base: ldr x14, [x29, #16]
            emit_raw(&mut self.ops, 0xF9400000 | (16 / 8 << 10) | (29 << 5) | 14);

            let h_loop = self.ops.new_dynamic_label();
            let h_done = self.ops.new_dynamic_label();

            self.emit_mov_imm_gp(15, 0); // x15 = head counter
            dynasm!(self.ops ; .arch aarch64 ; =>h_loop);
            self.emit_mov_imm_gp(9, num_heads);
            dynasm!(self.ops ; .arch aarch64 ; cmp x15, x9 ; b.ge =>h_done);

            // x13 = h * hd_bytes
            self.emit_mov_imm_gp(9, hd_bytes);
            // mul x13, x15, x9 => madd x13, x15, x9, xzr
            emit_raw(&mut self.ops, 0x9B097C00 | (31 << 10) | (15 << 5) | 13);

            self.emit_mha_extract_heads_neon(seq_len, head_dim, hidden_bytes, q_off, k_off, v_off)?;
            self.emit_mha_compute_scores_neon(seq_len, head_dim, q_off, k_off, sc_off)?;
            self.emit_mha_softmax_inplace_neon(seq_len, sc_off)?;
            self.emit_mha_scores_times_v_neon(seq_len, head_dim, sc_off, v_off, out_off)?;
            self.emit_mha_scatter_head_neon(seq_len, head_dim, hidden_bytes, out_off)?;

            dynasm!(self.ops ; .arch aarch64 ; add x15, x15, #1 ; b =>h_loop ; =>h_done);
            dynasm!(self.ops ; .arch aarch64 ; add sp, sp, #48);
            Ok(())
        }

        /// MHA: extract Q/K/V heads from [seq, hidden] to [seq, head_dim].
        /// Invariants: x14=scratchpad, x13=h*hd_bytes, [sp]=Q,[sp+8]=K,[sp+16]=V.
        fn emit_mha_extract_heads_neon(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            hidden_bytes: usize,
            q_off: usize,
            k_off: usize,
            v_off: usize,
        ) -> Result<(), String> {
            let hd_bytes = head_dim * 4;
            let vec_count = head_dim / NEON_WIDTH_F32;
            let tail = head_dim % NEON_WIDTH_F32;

            for s in 0..seq_len {
                let src_row_off = s * hidden_bytes; // byte offset of row s in [seq, hidden]
                let dst_row_off = s * hd_bytes;     // byte offset of row s in [seq, head_dim]

                // For each of Q, K, V:
                for (sp_off, scratch_off) in [(0i32, q_off), (8, k_off), (16, v_off)] {
                    // x10 = input_base + src_row_off + x13 (head offset)
                    dynasm!(self.ops ; .arch aarch64 ; ldr x10, [sp, sp_off]);
                    self.emit_add_offset_gp(10, 10, src_row_off);
                    dynasm!(self.ops ; .arch aarch64 ; add x10, x10, x13);

                    // x11 = scratchpad + scratch_off + dst_row_off
                    self.emit_add_offset_gp(11, 14, scratch_off + dst_row_off);

                    // Copy head_dim floats
                    for v in 0..vec_count {
                        let off = (v * 16) as u32;
                        emit_raw(&mut self.ops, encode_ldr_q(0, 10, off));
                        emit_raw(&mut self.ops, encode_str_q(0, 11, off));
                    }
                    for t in 0..tail {
                        let off = ((vec_count * 4 + t) * 4) as u32;
                        // ldr s0, [x10, #off]
                        emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (10 << 5) | 0);
                        // str s0, [x11, #off]
                        emit_raw(&mut self.ops, 0xBD000000 | ((off / 4) << 10) | (11 << 5) | 0);
                    }
                }
            }
            Ok(())
        }

        /// MHA: scores[i][j] = dot(Q_head[i], K_head[j]) * scale.
        /// Invariants: x14=scratchpad.
        fn emit_mha_compute_scores_neon(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            q_off: usize,
            k_off: usize,
            sc_off: usize,
        ) -> Result<(), String> {
            let hd_bytes = head_dim * 4;
            let vec_count = head_dim / NEON_WIDTH_F32;
            let tail = head_dim % NEON_WIDTH_F32;
            let scale = 1.0 / (head_dim as f32).sqrt();

            for i in 0..seq_len {
                let q_row = q_off + i * hd_bytes;
                let sc_row = sc_off + i * seq_len * 4;

                for j in 0..seq_len {
                    let k_row = k_off + j * hd_bytes;

                    // x10 = &Q_head[i]
                    self.emit_add_offset_gp(10, 14, q_row);
                    // x11 = &K_head[j]
                    self.emit_add_offset_gp(11, 14, k_row);

                    // Zero accumulator v1
                    emit_raw(&mut self.ops, encode_eor_v(1, 1, 1));

                    for v in 0..vec_count {
                        let off = (v * 16) as u32;
                        emit_raw(&mut self.ops, encode_ldr_q(2, 10, off));
                        emit_raw(&mut self.ops, encode_ldr_q(3, 11, off));
                        emit_raw(&mut self.ops, encode_fmla(1, 2, 3));
                    }
                    for t in 0..tail {
                        let off = ((vec_count * 4 + t) * 4) as u32;
                        emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (10 << 5) | 2);
                        emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (11 << 5) | 3);
                        // fmul s4, s2, s3
                        emit_raw(&mut self.ops, 0x1E230844);
                        // fadd s1, s1, s4
                        emit_raw(&mut self.ops, 0x1E242821);
                    }

                    // Horizontal sum: faddp v1.4s twice
                    emit_raw(&mut self.ops, 0x6E21D421); // faddp v1.4s, v1.4s, v1.4s
                    emit_raw(&mut self.ops, 0x6E21D421);

                    // Multiply by scale
                    self.emit_load_f32_const_neon(2, scale);
                    // fmul s1, s1, s2 (scalar)
                    emit_raw(&mut self.ops, 0x1E220821);

                    // Store: str s1, [x14 + sc_row + j*4]
                    self.emit_add_offset_gp(12, 14, sc_row + j * 4);
                    // str s1, [x12]
                    emit_raw(&mut self.ops, 0xBD000000 | (12 << 5) | 1);
                }
            }
            Ok(())
        }

        /// MHA: row-wise softmax on scores[seq, seq] in-place.
        /// Invariants: x14=scratchpad.
        fn emit_mha_softmax_inplace_neon(
            &mut self,
            seq_len: usize,
            sc_off: usize,
        ) -> Result<(), String> {
            let vec_count = seq_len / NEON_WIDTH_F32;
            let tail = seq_len % NEON_WIDTH_F32;

            for i in 0..seq_len {
                let row_off = sc_off + i * seq_len * 4;

                // x10 = &scores[i, 0]
                self.emit_add_offset_gp(10, 14, row_off);

                // Pass 1: find max in v4
                self.emit_load_f32_const_neon(4, f32::NEG_INFINITY);
                for v in 0..vec_count {
                    let off = (v * 16) as u32;
                    emit_raw(&mut self.ops, encode_ldr_q(5, 10, off));
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20F400, 4, 4, 5)); // fmax
                }
                for t in 0..tail {
                    let off = ((vec_count * 4 + t) * 4) as u32;
                    emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (10 << 5) | 5);
                    emit_raw(&mut self.ops, 0x4E040400 | (5 << 5) | 5); // dup v5.4s, v5.s[0]
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20F400, 4, 4, 5));
                }
                // Horizontal max
                emit_raw(&mut self.ops, 0x6E24F484); // fmaxp v4.4s, v4.4s, v4.4s
                emit_raw(&mut self.ops, 0x6E24F484);
                emit_raw(&mut self.ops, 0x4E040484); // dup v4.4s, v4.s[0]

                // Pass 2: exp(x - max), accumulate sum in v5
                emit_raw(&mut self.ops, encode_eor_v(5, 5, 5));
                for v in 0..vec_count {
                    let off = (v * 16) as u32;
                    emit_raw(&mut self.ops, encode_ldr_q(6, 10, off));
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 6, 6, 4)); // fsub
                    self.emit_exp_neon(6, 6);
                    emit_raw(&mut self.ops, encode_str_q(6, 10, off));
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 5, 5, 6)); // fadd
                }
                for t in 0..tail {
                    let off = ((vec_count * 4 + t) * 4) as u32;
                    emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (10 << 5) | 6);
                    // dup, sub, exp, store scalar, add to sum
                    emit_raw(&mut self.ops, 0x4E040400 | (6 << 5) | 6); // dup
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 6, 6, 4));
                    self.emit_exp_neon(6, 6);
                    emit_raw(&mut self.ops, 0xBD000000 | ((off / 4) << 10) | (10 << 5) | 6);
                    // scalar fadd s5, s5, s6
                    emit_raw(&mut self.ops, 0x1E2628A5);
                }

                // Horizontal sum v5
                emit_raw(&mut self.ops, 0x6E25D4A5); // faddp v5.4s, v5.4s, v5.4s
                emit_raw(&mut self.ops, 0x6E25D4A5);
                // inv_sum: fdiv s6, 1.0, s5
                self.emit_load_f32_const_neon(6, 1.0);
                emit_raw(&mut self.ops, 0x1E2518C6); // fdiv s6, s6, s5
                emit_raw(&mut self.ops, 0x4E040400 | (6 << 5) | 5); // dup v5.4s, v6.s[0]

                // Pass 3: normalize
                for v in 0..vec_count {
                    let off = (v * 16) as u32;
                    emit_raw(&mut self.ops, encode_ldr_q(6, 10, off));
                    emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 6, 6, 5)); // fmul
                    emit_raw(&mut self.ops, encode_str_q(6, 10, off));
                }
                for t in 0..tail {
                    let off = ((vec_count * 4 + t) * 4) as u32;
                    emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (10 << 5) | 6);
                    emit_raw(&mut self.ops, 0x1E2508C6); // fmul s6, s6, s5
                    emit_raw(&mut self.ops, 0xBD000000 | ((off / 4) << 10) | (10 << 5) | 6);
                }
            }
            Ok(())
        }

        /// MHA: out_head = scores @ V_head.
        /// out_head[i][d] = sum_j scores[i][j] * V_head[j][d]
        /// Invariants: x14=scratchpad.
        fn emit_mha_scores_times_v_neon(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            sc_off: usize,
            v_off: usize,
            out_off: usize,
        ) -> Result<(), String> {
            let hd_bytes = head_dim * 4;
            let hd_vec_count = head_dim / NEON_WIDTH_F32;
            let hd_tail = head_dim % NEON_WIDTH_F32;

            for i in 0..seq_len {
                let out_row = out_off + i * hd_bytes;
                let sc_row = sc_off + i * seq_len * 4;

                // Zero out_head[i]
                self.emit_add_offset_gp(10, 14, out_row);
                emit_raw(&mut self.ops, encode_eor_v(0, 0, 0));
                for v in 0..hd_vec_count {
                    emit_raw(&mut self.ops, encode_str_q(0, 10, (v * 16) as u32));
                }
                for t in 0..hd_tail {
                    let off = ((hd_vec_count * 4 + t) * 4) as u32;
                    emit_raw(&mut self.ops, 0xBD000000 | ((off / 4) << 10) | (10 << 5) | 0);
                }

                for j in 0..seq_len {
                    // Load scores[i][j] into v1 (broadcast)
                    self.emit_add_offset_gp(11, 14, sc_row + j * 4);
                    // ldr s1, [x11]
                    emit_raw(&mut self.ops, 0xBD400000 | (11 << 5) | 1);
                    // dup v1.4s, v1.s[0]
                    emit_raw(&mut self.ops, 0x4E040421);

                    // x12 = &V_head[j]
                    self.emit_add_offset_gp(12, 14, v_off + j * hd_bytes);

                    // Accumulate: out[d] += scores[i][j] * V[j][d]
                    for v in 0..hd_vec_count {
                        let off = (v * 16) as u32;
                        emit_raw(&mut self.ops, encode_ldr_q(2, 10, off));  // out
                        emit_raw(&mut self.ops, encode_ldr_q(3, 12, off));  // V
                        emit_raw(&mut self.ops, encode_fmla(2, 1, 3));      // out += score * V
                        emit_raw(&mut self.ops, encode_str_q(2, 10, off));
                    }
                    for t in 0..hd_tail {
                        let off = ((hd_vec_count * 4 + t) * 4) as u32;
                        // scalar: ldr s2 (out), ldr s3 (V), fmadd s2, s1, s3, s2, str s2
                        emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (10 << 5) | 2);
                        emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (12 << 5) | 3);
                        // fmadd s2, s1, s3, s2: 0x1F030822
                        emit_raw(&mut self.ops, 0x1F030822);
                        emit_raw(&mut self.ops, 0xBD000000 | ((off / 4) << 10) | (10 << 5) | 2);
                    }
                }
            }
            Ok(())
        }

        /// MHA: scatter out_head[seq, head_dim] back to output[seq, hidden].
        /// Invariants: x14=scratchpad, x13=h*hd_bytes, [sp+24]=output.
        fn emit_mha_scatter_head_neon(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            hidden_bytes: usize,
            out_off: usize,
        ) -> Result<(), String> {
            let hd_bytes = head_dim * 4;
            let vec_count = head_dim / NEON_WIDTH_F32;
            let tail = head_dim % NEON_WIDTH_F32;

            for s in 0..seq_len {
                let src_off = out_off + s * hd_bytes;
                let dst_row_off = s * hidden_bytes;

                // x10 = &out_head[s]
                self.emit_add_offset_gp(10, 14, src_off);
                // x11 = &output[s, h*hd] = output_base + s*hidden_bytes + h*hd_bytes
                dynasm!(self.ops ; .arch aarch64 ; ldr x11, [sp, #24]);
                self.emit_add_offset_gp(11, 11, dst_row_off);
                dynasm!(self.ops ; .arch aarch64 ; add x11, x11, x13);

                for v in 0..vec_count {
                    let off = (v * 16) as u32;
                    emit_raw(&mut self.ops, encode_ldr_q(0, 10, off));
                    emit_raw(&mut self.ops, encode_str_q(0, 11, off));
                }
                for t in 0..tail {
                    let off = ((vec_count * 4 + t) * 4) as u32;
                    emit_raw(&mut self.ops, 0xBD400000 | ((off / 4) << 10) | (10 << 5) | 0);
                    emit_raw(&mut self.ops, 0xBD000000 | ((off / 4) << 10) | (11 << 5) | 0);
                }
            }
            Ok(())
        }


        // ── Epilogue on accumulators ──────────────────────────────────

        pub fn emit_trace_on_accumulator_neon(
            &mut self,
            epilogue_bodies: &[&[TraceOp]],
            accum_start: u8,
            accum_count: u8,
        ) -> Result<(), String> {
            for body in epilogue_bodies {
                if body.is_empty() { continue; }
                for a in 0..accum_count {
                    let acc = accum_start + a;
                    self.emit_single_acc_epilogue(acc, body)?;
                }
            }
            Ok(())
        }

        pub fn emit_single_acc_epilogue(
            &mut self,
            acc: u8,
            body: &[TraceOp],
        ) -> Result<(), String> {
            let n = body.len();
            if n == 0 { return Ok(()); }

            let frame_size = (n as u32) * 16;
            dynasm!(self.ops ; .arch aarch64 ; sub sp, sp, frame_size);

            for (i, op) in body.iter().enumerate() {
                let slot = (i as u32) * 16;
                match op {
                    TraceOp::Input(0) => {
                        emit_raw(&mut self.ops, encode_str_q(acc, 31, slot));
                    }
                    TraceOp::Input(_) => {
                        emit_raw(&mut self.ops, encode_eor_v(0, 0, 0));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Const(v) => {
                        self.emit_load_f32_const_neon(0, *v as f32);
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Add(a, b) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(1, 31, (*b as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 0, 0, 1));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Sub(a, b) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(1, 31, (*b as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 0, 0, 1));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Mul(a, b) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(1, 31, (*b as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 0, 0, 1));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Div(a, b) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(1, 31, (*b as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20FC00, 0, 0, 1));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Fma(a, b, c) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(1, 31, (*b as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(2, 31, (*c as u32) * 16));
                        emit_raw(&mut self.ops, encode_fmla(2, 0, 1));
                        emit_raw(&mut self.ops, encode_str_q(2, 31, slot));
                    }
                    TraceOp::Neg(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA0F800, 0, 0));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Abs(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_unary(0x4EA0F800, 0, 0));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Sqrt(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA1F800, 0, 0));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Rsqrt(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        self.emit_rsqrt_scratch_neon();
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Recip(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        self.emit_recip_scratch_neon();
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Exp(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        self.emit_exp_scratch_neon();
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Tanh(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        self.emit_tanh_scratch_neon();
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Log(a) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        self.emit_log_scratch_neon();
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Max(a, b) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(1, 31, (*b as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20F400, 0, 0, 1));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                    TraceOp::Min(a, b) => {
                        emit_raw(&mut self.ops, encode_ldr_q(0, 31, (*a as u32) * 16));
                        emit_raw(&mut self.ops, encode_ldr_q(1, 31, (*b as u32) * 16));
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0F400, 0, 0, 1));
                        emit_raw(&mut self.ops, encode_str_q(0, 31, slot));
                    }
                }
            }

            emit_raw(&mut self.ops, encode_ldr_q(acc, 31, ((n - 1) as u32) * 16));
            dynasm!(self.ops ; .arch aarch64 ; add sp, sp, frame_size);

            Ok(())
        }

        // ── GEMM 8x12 microkernel ────────────────────────────────────

        /// Emit the 8x12 NEON GEMM microkernel with optional epilogue.
        ///
        /// Uses dynasm! labels for the K-loop — no manual branch patching.
        pub fn emit_gemm_8x12_core(
            &mut self,
            k: usize,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            // Zero accumulators v8..v31
            for acc in 8u8..32u8 {
                emit_raw(&mut self.ops, encode_eor_v(acc, acc, acc));
            }

            // K-loop counter
            let k16 = (k & 0xFFFF) as u32;
            dynasm!(self.ops ; .arch aarch64 ; movz x9, k16 as u32);

            let k_loop = self.ops.new_dynamic_label();
            let k_done = self.ops.new_dynamic_label();

            dynasm!(self.ops
                ; .arch aarch64
                ; =>k_loop
                ; cbz x9, =>k_done
            );

            // Load A panel: 2 x v (8 floats = MR)
            emit_raw(&mut self.ops, encode_ld1_post(0, 0));
            emit_raw(&mut self.ops, encode_ld1_post(1, 0));

            // Load B panel: 3 x v (12 floats = NR)
            emit_raw(&mut self.ops, encode_ld1_post(2, 1));
            emit_raw(&mut self.ops, encode_ld1_post(3, 1));
            emit_raw(&mut self.ops, encode_ld1_post(4, 1));

            // FMA: 8 rows x 3 col-groups
            for row in 0u8..8 {
                let a_reg = row / 4;
                let lane = row % 4;
                for col in 0u8..3 {
                    let acc = 8 + row * 3 + col;
                    let b_reg = 2 + col;
                    emit_raw(&mut self.ops, encode_fmla_lane(acc, b_reg, a_reg, lane));
                }
            }

            dynasm!(self.ops
                ; .arch aarch64
                ; sub x9, x9, #1
                ; b =>k_loop
                ; =>k_done
            );

            // Epilogue injection on accumulators
            if !epilogue_bodies.is_empty() {
                self.emit_trace_on_accumulator_neon(epilogue_bodies, 8, 24)?;
            }

            // Store accumulators
            for acc in 8u8..32u8 {
                emit_raw(&mut self.ops, encode_st1_post(acc, 2));
            }

            Ok(())
        }

        pub fn emit_gemm_8x12_neon(&mut self, k: usize) -> Result<(), String> {
            self.emit_gemm_8x12_core(k, &[])
        }

        // ── BLIS 5-level loop nesting ────────────────────────────────

        fn emit_gemm_blis_core(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let blocking = profile.gemm_blocking(m, n, k);
            let (kc, mc, nc) = (blocking.kc, blocking.mc, blocking.nc);

            let pack_a_bytes = mc * kc * 4;
            let pack_b_bytes = kc * nc * 4;
            let pack_a_off = self.blis_scratchpad_offset;
            let pack_b_off = pack_a_off + pack_a_bytes;
            let total_extra = pack_a_bytes + pack_b_bytes;
            self.blis_scratchpad_offset += total_extra;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(total_extra);

            #[cfg(target_arch = "aarch64")]
            let pack_a_fn = crate::asm::aarch64::gllm_pack_a_f32_neon as *const () as u64;
            #[cfg(not(target_arch = "aarch64"))]
            let pack_a_fn = 0u64;

            #[cfg(target_arch = "aarch64")]
            let pack_b_fn = crate::asm::aarch64::gllm_pack_b_f32_neon as *const () as u64;
            #[cfg(not(target_arch = "aarch64"))]
            let pack_b_fn = 0u64;

            self.emit_save_callee_saved();

            // Save argument registers into callee-saved
            dynasm!(self.ops
                ; .arch aarch64
                ; mov x23, x0   // A base
                ; mov x24, x1   // B base
                ; mov x25, x7   // C base
            );
            // x22 = scratchpad base (loaded from stack frame)
            // ldr x22, [sp, #96]
            emit_raw(&mut self.ops, 0xF9400000 | (96 / 8 << 10) | (31 << 5) | 22);

            // ── JC loop (over N) ──
            let jc_loop = self.ops.new_dynamic_label();
            let jc_done = self.ops.new_dynamic_label();
            dynasm!(self.ops ; .arch aarch64 ; movz x19, 0);
            dynasm!(self.ops ; .arch aarch64 ; =>jc_loop);
            self.emit_mov_imm_gp(9, n);
            dynasm!(self.ops ; .arch aarch64 ; cmp x19, x9 ; b.ge =>jc_done);

            // nc_cur = min(n - jc, nc)
            self.emit_mov_imm_gp(9, n);
            dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, x19);
            self.emit_mov_imm_gp(26, nc);
            dynasm!(self.ops ; .arch aarch64 ; cmp x9, x26 ; csel x26, x9, x26, lo);

            // ── PC loop (over K) ──
            let pc_loop = self.ops.new_dynamic_label();
            let pc_done = self.ops.new_dynamic_label();
            dynasm!(self.ops ; .arch aarch64 ; movz x20, 0 ; =>pc_loop);
            self.emit_mov_imm_gp(9, k);
            dynasm!(self.ops ; .arch aarch64 ; cmp x20, x9 ; b.ge =>pc_done);

            // kc_cur = min(k - pc, kc)
            self.emit_mov_imm_gp(9, k);
            dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, x20);
            self.emit_mov_imm_gp(27, kc);
            dynasm!(self.ops ; .arch aarch64 ; cmp x9, x27 ; csel x27, x9, x27, lo);

            // pack_b(B + (pc*n + jc)*4, n, scratch+pack_b_off, kc_cur, nc_cur, NR)
            self.emit_mov_imm_gp(9, n);
            dynasm!(self.ops ; .arch aarch64 ; mul x0, x20, x9 ; add x0, x0, x19 ; lsl x0, x0, #2 ; add x0, x24, x0);
            self.emit_mov_imm_gp(1, n);
            self.emit_mov_imm_gp(9, pack_b_off);
            dynasm!(self.ops ; .arch aarch64 ; add x2, x22, x9 ; mov x3, x27 ; mov x4, x26);
            self.emit_mov_imm_gp(5, NR);
            self.emit_mov_imm_gp(10, pack_b_fn as usize);
            dynasm!(self.ops ; .arch aarch64 ; blr x10);

            // ── IC loop (over M) ──
            let ic_loop = self.ops.new_dynamic_label();
            let ic_done = self.ops.new_dynamic_label();
            dynasm!(self.ops ; .arch aarch64 ; movz x21, 0 ; =>ic_loop);
            self.emit_mov_imm_gp(9, m);
            dynasm!(self.ops ; .arch aarch64 ; cmp x21, x9 ; b.ge =>ic_done);

            // mc_cur = min(m - ic, mc)
            self.emit_mov_imm_gp(9, m);
            dynasm!(self.ops ; .arch aarch64 ; sub x9, x9, x21);
            self.emit_mov_imm_gp(28, mc);
            dynasm!(self.ops ; .arch aarch64 ; cmp x9, x28 ; csel x28, x9, x28, lo);

            // pack_a(A + (ic*k + pc)*4, k, scratch+pack_a_off, mc_cur, kc_cur, MR)
            self.emit_mov_imm_gp(9, k);
            dynasm!(self.ops ; .arch aarch64 ; mul x0, x21, x9 ; add x0, x0, x20 ; lsl x0, x0, #2 ; add x0, x23, x0);
            self.emit_mov_imm_gp(1, k);
            self.emit_mov_imm_gp(9, pack_a_off);
            dynasm!(self.ops ; .arch aarch64 ; add x2, x22, x9 ; mov x3, x28 ; mov x4, x27);
            self.emit_mov_imm_gp(5, MR);
            self.emit_mov_imm_gp(10, pack_a_fn as usize);
            dynasm!(self.ops ; .arch aarch64 ; blr x10);

            // ── JR loop (over nc_cur in steps of NR) ──
            let jr_loop = self.ops.new_dynamic_label();
            let jr_done = self.ops.new_dynamic_label();
            dynasm!(self.ops ; .arch aarch64 ; movz x14, 0 ; =>jr_loop);
            dynasm!(self.ops ; .arch aarch64 ; cmp x14, x26 ; b.ge =>jr_done);

            // ── IR loop (over mc_cur in steps of MR) ──
            let ir_loop = self.ops.new_dynamic_label();
            let ir_done = self.ops.new_dynamic_label();
            dynasm!(self.ops ; .arch aarch64 ; movz x15, 0 ; =>ir_loop);
            dynasm!(self.ops ; .arch aarch64 ; cmp x15, x28 ; b.ge =>ir_done);

            // x0 = pack_a + ir * kc_cur * 4
            dynasm!(self.ops ; .arch aarch64 ; mul x9, x15, x27 ; lsl x9, x9, #2);
            self.emit_mov_imm_gp(10, pack_a_off);
            dynasm!(self.ops ; .arch aarch64 ; add x0, x22, x10 ; add x0, x0, x9);

            // x1 = pack_b + jr * kc_cur * 4
            dynasm!(self.ops ; .arch aarch64 ; mul x9, x14, x27 ; lsl x9, x9, #2);
            self.emit_mov_imm_gp(10, pack_b_off);
            dynasm!(self.ops ; .arch aarch64 ; add x1, x22, x10 ; add x1, x1, x9);

            // x2 = C + ((ic+ir)*n + jc + jr) * 4
            dynasm!(self.ops ; .arch aarch64 ; add x9, x21, x15);
            self.emit_mov_imm_gp(10, n);
            dynasm!(self.ops
                ; .arch aarch64
                ; mul x9, x9, x10
                ; add x9, x9, x19
                ; add x9, x9, x14
                ; lsl x9, x9, #2
                ; add x2, x25, x9
                ; mov x3, x27
            );

            self.emit_gemm_8x12_core(kc, epilogue_bodies)?;

            // IR += MR
            dynasm!(self.ops ; .arch aarch64 ; add x15, x15, MR as u32 ; b =>ir_loop ; =>ir_done);
            // JR += NR
            dynasm!(self.ops ; .arch aarch64 ; add x14, x14, NR as u32 ; b =>jr_loop ; =>jr_done);
            // IC += mc
            dynasm!(self.ops ; .arch aarch64 ; add x21, x21, x28 ; b =>ic_loop ; =>ic_done);
            // PC += kc
            dynasm!(self.ops ; .arch aarch64 ; add x20, x20, x27 ; b =>pc_loop ; =>pc_done);
            // JC += nc
            dynasm!(self.ops ; .arch aarch64 ; add x19, x19, x26 ; b =>jc_loop ; =>jc_done);

            self.emit_restore_callee_saved();

            Ok(())
        }

        fn emit_gemm_microkernel(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
        ) -> Result<(), String> {
            // Dispatch to Apple AMX path when hardware supports it and dimensions fit.
            if profile.isa == IsaLevel::NeonAmx && apple_amx_gemm_eligible(m, n, k) {
                let words = emit_apple_amx_f32_gemm(m, n, k);
                for w in words {
                    self.ops.push_u32(w);
                }
                return Ok(());
            }
            let blocking = profile.gemm_blocking(m, n, k);
            if k <= blocking.kc && m <= blocking.mc && n <= blocking.nc {
                return self.emit_gemm_8x12_neon(k);
            }
            self.emit_gemm_blis_core(m, n, k, profile, &[])
        }

        /// Add bias vector to each row of the output matrix (post-GEMM epilogue).
        ///
        /// ABI: x2 = bias ptr (restored after GEMM), x7 = output ptr.
        /// Output is [m, n] row-major f32. Bias is [n] f32, broadcast across rows.
        ///
        /// Uses NEON 4xf32 vectors for the bulk, scalar tail for remainder.
        fn emit_bias_add_neon(&mut self, m: usize, n: usize) -> Result<(), String> {
            let vec_count = n / NEON_WIDTH_F32;
            let tail = n % NEON_WIDTH_F32;
            let row_bytes = n * 4;

            // x10 = row counter (0..m)
            dynasm!(self.ops ; .arch aarch64 ; movz x10, 0);

            let row_loop = self.ops.new_dynamic_label();
            let row_done = self.ops.new_dynamic_label();

            dynasm!(self.ops ; .arch aarch64 ; =>row_loop);
            self.emit_mov_imm_gp(9, m);
            dynasm!(self.ops ; .arch aarch64 ; cmp x10, x9 ; b.ge =>row_done);

            // x11 = row byte offset = x10 * row_bytes
            self.emit_mov_imm_gp(9, row_bytes);
            dynasm!(self.ops ; .arch aarch64 ; mul x11, x10, x9);

            // Vectorized: process NEON_WIDTH_F32 (4) floats at a time
            for v in 0..vec_count {
                let col_byte_off = (v * 16) as u32; // 4 floats * 4 bytes = 16

                // x12 = &output[row][v*4] = x7 + x11 + col_byte_off
                dynasm!(self.ops ; .arch aarch64 ; add x12, x7, x11);
                if col_byte_off > 0 {
                    self.emit_add_offset_gp(12, 12, col_byte_off as usize);
                }

                // x13 = &bias[v*4] = x2 + col_byte_off
                if col_byte_off == 0 {
                    dynasm!(self.ops ; .arch aarch64 ; mov x13, x2);
                } else {
                    self.emit_add_offset_gp(13, 2, col_byte_off as usize);
                }

                // ldr q0, [x12]  -- output chunk
                emit_raw(&mut self.ops, encode_ldr_q(0, 12, 0));
                // ldr q1, [x13]  -- bias chunk
                emit_raw(&mut self.ops, encode_ldr_q(1, 13, 0));
                // fadd v0.4s, v0.4s, v1.4s
                emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 0, 0, 1));
                // str q0, [x12]  -- store back
                emit_raw(&mut self.ops, encode_str_q(0, 12, 0));
            }

            // Scalar tail: remaining elements
            for t in 0..tail {
                let off = (vec_count * 16 + t * 4) as u32;

                // x12 = &output[row][vec_count*4 + t] = x7 + x11 + off
                dynasm!(self.ops ; .arch aarch64 ; add x12, x7, x11);
                self.emit_add_offset_gp(12, 12, off as usize);

                // x13 = &bias[vec_count*4 + t] = x2 + off
                self.emit_add_offset_gp(13, 2, off as usize);

                // ldr s0, [x12]  -- scalar load output
                emit_raw(&mut self.ops, 0xBD400000 | (12 << 5) | 0);
                // ldr s1, [x13]  -- scalar load bias
                emit_raw(&mut self.ops, 0xBD400000 | (13 << 5) | 1);
                // fadd s0, s0, s1
                emit_raw(&mut self.ops, 0x1E212800 | (1 << 5) | 0);
                // str s0, [x12]  -- scalar store
                emit_raw(&mut self.ops, 0xBD000000 | (12 << 5) | 0);
            }

            // row++
            dynasm!(self.ops ; .arch aarch64 ; add x10, x10, 1 ; b =>row_loop ; =>row_done);

            Ok(())
        }

        fn emit_gemm_microkernel_core(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let blocking = profile.gemm_blocking(m, n, k);
            if k <= blocking.kc && m <= blocking.mc && n <= blocking.nc {
                return self.emit_gemm_8x12_core(k, epilogue_bodies);
            }
            self.emit_gemm_blis_core(m, n, k, profile, epilogue_bodies)
        }

        // ── Dispatch: emit_group / emit_standalone / emit_plan ───────

        fn emit_inline_elementwise(
            &mut self,
            op_kind: &OpKind,
            elem_count: usize,
        ) -> Result<(), String> {
            let body: Vec<TraceOp> = match op_kind {
                OpKind::Silu => vec![
                    TraceOp::Input(0), TraceOp::Neg(0), TraceOp::Exp(1),
                    TraceOp::Const(1.0), TraceOp::Add(3, 2), TraceOp::Recip(4),
                    TraceOp::Mul(0, 5),
                ],
                OpKind::Gelu => vec![
                    TraceOp::Input(0), TraceOp::Const(0.7978845608028654),
                    TraceOp::Const(0.044715), TraceOp::Mul(0, 0),
                    TraceOp::Mul(3, 2), TraceOp::Const(1.0), TraceOp::Add(5, 4),
                    TraceOp::Mul(0, 6), TraceOp::Mul(7, 1), TraceOp::Tanh(8),
                    TraceOp::Const(1.0), TraceOp::Add(10, 9), TraceOp::Const(0.5),
                    TraceOp::Mul(0, 12), TraceOp::Mul(13, 11),
                ],
                OpKind::Add | OpKind::Residual => vec![
                    TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(0, 1),
                ],
                OpKind::Mul => vec![
                    TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(0, 1),
                ],
                OpKind::SwiGlu => vec![
                    TraceOp::Input(0), TraceOp::Input(1), TraceOp::Neg(0),
                    TraceOp::Exp(2), TraceOp::Const(1.0), TraceOp::Add(4, 3),
                    TraceOp::Recip(5), TraceOp::Mul(0, 6), TraceOp::Mul(7, 1),
                ],
                _ => return Err(format!("no inline fallback for {:?} on aarch64-dynasm", op_kind)),
            };
            let is_binary = body.iter().any(|op| matches!(op, TraceOp::Input(n) if *n >= 1));
            self.emit_elementwise_loop_neon(&body, elem_count, is_binary)
        }

        fn emit_traced_standalone(
            &mut self,
            pattern: &ComputePattern,
            op_kind: &OpKind,
            elem_count: usize,
        ) -> Result<(), String> {
            match pattern {
                ComputePattern::Elementwise { body } => {
                    self.emit_elementwise_loop_neon(body, elem_count, false)
                }
                ComputePattern::BinaryElementwise { body } => {
                    self.emit_elementwise_loop_neon(body, elem_count, true)
                }
                ComputePattern::Injective { body, .. } => {
                    let is_binary = body.iter().any(|op| matches!(op, TraceOp::Input(n) if *n >= 1));
                    self.emit_elementwise_loop_neon(body, elem_count, is_binary)
                }
                ComputePattern::QuantDecode { decode, .. } => {
                    self.emit_elementwise_loop_neon(decode, elem_count, false)
                }
                ComputePattern::NormLike { reduce, finalize, transform } => {
                    let eps = match op_kind {
                        OpKind::RmsNorm { eps } | OpKind::LayerNorm { eps } => *eps,
                        OpKind::L2Normalize { .. } => 1e-12,
                        _ => 1e-5,
                    };
                    self.emit_norm_standalone_neon(reduce, finalize, transform, elem_count, eps)
                }
                ComputePattern::Reduction { identity, combine, .. } => {
                    self.emit_reduce_standalone_neon(combine, elem_count, *identity)
                }
                ComputePattern::Gemm => {
                    Err("unexpected Gemm pattern in traced standalone".into())
                }
            }
        }

        fn emit_standalone(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing op")?;
            match &op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile)
                }
                OpKind::GemmBias { m, n, k } => {
                    // Save x2 (bias ptr) to x16 before GEMM -- the microkernel
                    // repurposes x2 as the C (output) pointer.
                    dynasm!(self.ops ; .arch aarch64 ; mov x16, x2);
                    self.emit_gemm_microkernel(*m, *n, *k, profile)?;
                    // Restore bias ptr and apply bias addition to output
                    dynasm!(self.ops ; .arch aarch64 ; mov x2, x16);
                    self.emit_bias_add_neon(*m, *n)
                }
                OpKind::Reshape { .. } | OpKind::Transpose { .. } => Ok(()),
                OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                    self.emit_multi_head_attention_neon(*seq_len, *num_heads, *head_dim)
                }
                OpKind::MeanPool { seq_len, hidden } => {
                    self.emit_mean_pool_neon(*seq_len, *hidden)
                }
                OpKind::RoPE { head_dim, .. } => {
                    let total_elems = op.outputs.first()
                        .and_then(|&out_id| graph.tensor_numel(out_id))
                        .unwrap_or(0);
                    if total_elems == 0 || *head_dim == 0 {
                        return Err("RoPE: zero-element tensor or head_dim".into());
                    }
                    let n_heads = total_elems / head_dim;
                    self.emit_rope_standalone_neon(*head_dim, n_heads)
                }
                _ => {
                    let elem_count = op.outputs.first()
                        .and_then(|&out_id| graph.tensor_numel(out_id))
                        .unwrap_or(0);
                    if elem_count == 0 { return Err("zero-element tensor in standalone op".into()); }

                    let key = ScalarOpRegistry::key_from_op_kind(&op.kind);
                    let trace = registry.and_then(|r| r.get_trace(&key));

                    match trace {
                        Some(t) => self.emit_traced_standalone(&t.pattern, &op.kind, elem_count),
                        None => self.emit_inline_elementwise(&op.kind, elem_count),
                    }
                }
            }
        }

        fn emit_loop_fusion(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            _profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            // Determine elem_count from the anchor op (like x86_64 emit_elementwise_chain)
            let anchor_op = graph.op(group.anchor).ok_or("missing anchor op")?;
            let elem_count = if let Some(&out_id) = anchor_op.outputs.first() {
                graph.tensor_numel(out_id).unwrap_or(0)
            } else {
                0
            };
            if elem_count == 0 { return Err("zero-element tensor in loop fusion".into()); }

            // Collect trace bodies from anchor + epilogue ops
            let mut trace_bodies: Vec<(Vec<TraceOp>, bool)> = Vec::new();
            if let Some(reg) = registry {
                // Anchor op trace
                let key = ScalarOpRegistry::key_from_op_kind(&anchor_op.kind);
                if let Some(trace) = reg.get_trace(&key) {
                    if let Some(body) = trace.pattern.body() {
                        let is_binary = matches!(
                            trace.pattern,
                            ComputePattern::BinaryElementwise { .. }
                        );
                        trace_bodies.push((body.to_vec(), is_binary));
                    }
                }

                // Epilogue op traces
                for &epi_id in &group.epilogue {
                    let epi_op = graph.op(epi_id).ok_or("missing epilogue op")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            let is_binary = matches!(
                                trace.pattern,
                                ComputePattern::BinaryElementwise { .. }
                            );
                            trace_bodies.push((body.to_vec(), is_binary));
                        }
                    }
                }
            }

            if trace_bodies.is_empty() {
                return Err(format!(
                    "loop fusion: no trace info found for group (anchor={:?}), registry lookup required",
                    group.anchor
                ));
            }

            let has_binary = trace_bodies.iter().any(|(_, b)| *b);

            // Set up pointers from scratchpad allocation
            let first_op = graph.op(*group.ops.first().unwrap_or(&group.anchor))
                .ok_or("missing first op")?;
            if let Some(&input_tid) = first_op.inputs.first() {
                if let Some(slot) = alloc.slots.iter().find(|s| s.tensor_id == input_tid) {
                    // ldr x14, [x29, #16]  — scratchpad base
                    emit_raw(&mut self.ops, 0xF9400000 | (16 / 8 << 10) | (29 << 5) | 14);
                    self.emit_add_offset_gp(0, 14, slot.offset);
                }
            }
            if has_binary {
                if let Some(&input_tid) = first_op.inputs.get(1) {
                    if let Some(slot) = alloc.slots.iter().find(|s| s.tensor_id == input_tid) {
                        emit_raw(&mut self.ops, 0xF9400000 | (16 / 8 << 10) | (29 << 5) | 14);
                        self.emit_add_offset_gp(1, 14, slot.offset);
                    }
                }
            }
            // Output pointer: use the last epilogue op's output, or anchor if no epilogue
            let final_op_id = group.epilogue.last().copied().unwrap_or(group.anchor);
            let final_op = graph.op(final_op_id).unwrap_or(anchor_op);
            if let Some(&output_tid) = final_op.outputs.first() {
                if let Some(slot) = alloc.slots.iter().find(|s| s.tensor_id == output_tid) {
                    emit_raw(&mut self.ops, 0xF9400000 | (16 / 8 << 10) | (29 << 5) | 14);
                    self.emit_add_offset_gp(7, 14, slot.offset);
                }
            }

            let all_bodies: Vec<&[TraceOp]> = trace_bodies.iter().map(|(b, _)| b.as_slice()).collect();
            self.emit_fused_elementwise_loop_neon(&all_bodies, elem_count, has_binary)
        }

        fn emit_gemm_with_epilogue(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let anchor_op = graph.op(group.anchor).ok_or("missing anchor op")?;
            let (m, n, k) = match &anchor_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => return Err(format!("EpilogueInjection anchor must be GEMM, got {:?}", other)),
            };

            let mut epilogue_owned: Vec<Vec<TraceOp>> = Vec::new();
            if let Some(reg) = registry {
                for &epi_id in &group.epilogue {
                    let epi_op = graph.op(epi_id).ok_or("missing epilogue op")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            epilogue_owned.push(body.to_vec());
                        }
                    }
                }
            }

            let body_refs: Vec<&[TraceOp]> = epilogue_owned.iter().map(|b| b.as_slice()).collect();
            self.emit_gemm_microkernel_core(m, n, k, profile, &body_refs)
        }

        fn emit_tile_level_fusion(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
            _tile_rows: usize,
            _registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            if let Some(pred_op) = graph.op(predecessor) {
                match &pred_op.kind {
                    OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                        self.emit_gemm_microkernel(*m, *n, *k, profile)?;
                    }
                    other => { return Err(format!("aarch64_dynasm: TileLevelFusion predecessor not implemented for {:?}", other)); }
                }
            }
            let anchor_op = graph.op(group.anchor).ok_or("missing anchor op")?;
            match &anchor_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile)
                }
                other => { Err(format!("aarch64_dynasm: TileLevelFusion anchor must be GEMM, got {:?}", other)) }
            }
        }

        fn emit_compute_root(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let norm_op = graph.op(predecessor).ok_or("ComputeRoot: predecessor not in graph")?;
            let gemm_op = graph.op(group.anchor).ok_or("ComputeRoot: GEMM not in graph")?;

            let _eps = match &norm_op.kind {
                OpKind::RmsNorm { eps } => *eps,
                other => return Err(format!("ComputeRoot: expected norm, got {:?}", other)),
            };
            let (m, n, k) = match &gemm_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => return Err(format!("ComputeRoot: expected GEMM, got {:?}", other)),
            };

            // Fallback: emit predecessor standalone, then GEMM
            let pred_group = FusionGroup {
                id: group.id,
                anchor: predecessor,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![predecessor],
            };
            self.emit_standalone(&pred_group, graph, profile, registry)?;
            self.emit_gemm_microkernel(m, n, k, profile)
        }

        fn emit_group(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            match group.mode {
                FusionMode::Standalone => self.emit_standalone(group, graph, profile, registry),
                FusionMode::LoopFusion => self.emit_loop_fusion(group, graph, alloc, profile, registry),
                FusionMode::EpilogueInjection => self.emit_gemm_with_epilogue(group, graph, profile, registry),
                FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                    self.emit_tile_level_fusion(group, graph, profile, predecessor, tile_rows, registry)
                }
                FusionMode::ComputeRoot { predecessor } => {
                    self.emit_compute_root(group, graph, profile, predecessor, registry)
                }
                FusionMode::QkvSharedInput | FusionMode::NormIntoGemm => {
                    for &op_id in &group.ops {
                        let single = FusionGroup {
                            id: group.id,
                            anchor: op_id,
                            epilogue: vec![],
                            mode: FusionMode::Standalone,
                            ops: vec![op_id],
                        };
                        self.emit_standalone(&single, graph, profile, registry)?;
                    }
                    Ok(())
                }
            }
        }

        // ── Public entry point ───────────────────────────────────────

        pub fn emit_plan(
            &mut self,
            plan: &FusionPlan,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<CodegenOutput, String> {
            // Reset state with a fresh assembler
            self.ops = Assembler::new().map_err(|e| format!("{:?}", e))?;
            self.blis_scratchpad_bytes = 0;
            self.blis_scratchpad_offset = alloc.total_bytes;
            self.blis_base_offset = alloc.total_bytes;

            self.emit_prologue();

            for group in &plan.groups {
                self.emit_group(group, graph, alloc, profile, registry)?;
            }

            self.emit_epilogue();

            // Finalize: extract code bytes from the assembler
            let old_ops = std::mem::replace(
                &mut self.ops,
                Assembler::new().map_err(|e| format!("{:?}", e))?,
            );
            let buf = old_ops.finalize().map_err(|e| format!("{:?}", e))?;

            Ok(CodegenOutput {
                code: buf.to_vec(),
                scratchpad_bytes: alloc.total_bytes + self.blis_scratchpad_bytes,
            })
        }
    }

    // ── GPR number mapping (matches aarch64.rs convention) ──────────────

    fn gpr_num(base: &BaseReg) -> u8 {
        match base {
            BaseReg::Arg(n) => *n,
            BaseReg::StackPtr => 31,
            BaseReg::ScratchpadBase => 22,
            BaseReg::OutputPtr => 7,
            BaseReg::LoopVar(n) => 19 + *n,
            BaseReg::Scratch(n) => 9 + *n,
        }
    }

    // ── SimdOps implementation ──────────────────────────────────────────

    impl SimdOps for DynasmAArch64CodeGen {
        // ── Vector arithmetic ───────────────────────────────────────

        fn vadd(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, dst.0, a.0, b.0));
            Ok(())
        }

        fn vsub(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, dst.0, a.0, b.0));
            Ok(())
        }

        fn vmul(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, dst.0, a.0, b.0));
            Ok(())
        }

        fn vdiv(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20FC00, dst.0, a.0, b.0));
            Ok(())
        }

        fn vfma(&mut self, dst: VReg, a: VReg, b: VReg, c: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_mov_v(dst.0, c.0));
            emit_raw(&mut self.ops, encode_fmla(dst.0, a.0, b.0));
            Ok(())
        }

        fn vneg(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA0F800, dst.0, a.0));
            Ok(())
        }

        fn vabs(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_unary(0x4EA0F800, dst.0, a.0));
            Ok(())
        }

        fn vsqrt(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA1F800, dst.0, a.0));
            Ok(())
        }

        fn vmax(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20F400, dst.0, a.0, b.0));
            Ok(())
        }

        fn vmin(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0F400, dst.0, a.0, b.0));
            Ok(())
        }

        // ── Approximate reciprocals ─────────────────────────────────

        fn vrecip(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            self.emit_recip_refined_neon(dst.0, a.0);
            Ok(())
        }

        fn vrsqrt(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            self.emit_rsqrt_refined_neon(dst.0, a.0);
            Ok(())
        }

        // ── Memory operations ───────────────────────────────────────

        fn vload(&mut self, _dst: VReg, _mem: MemOperand) -> Result<(), String> {
            Err("vload not yet implemented for dynasm backend".into())
        }

        fn vstore(&mut self, _mem: MemOperand, _src: VReg) -> Result<(), String> {
            Err("vstore not yet implemented for dynasm backend".into())
        }

        fn vbroadcast(&mut self, _dst: VReg, _mem: MemOperand) -> Result<(), String> {
            Err("vbroadcast not yet implemented for dynasm backend".into())
        }

        fn vbroadcast_const(&mut self, dst: VReg, val: f32) -> Result<(), String> {
            self.emit_load_f32_const_neon(dst.0, val);
            Ok(())
        }

        fn vzero(&mut self, dst: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_eor_v(dst.0, dst.0, dst.0));
            Ok(())
        }

        fn vmov(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_mov_v(dst.0, src.0));
            Ok(())
        }

        // ── Bitwise / integer operations ────────────────────────────

        fn vand(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_and_v(dst.0, a.0, b.0));
            Ok(())
        }

        fn vor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_orr_v(dst.0, a.0, b.0));
            Ok(())
        }

        fn vxor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_eor_v(dst.0, a.0, b.0));
            Ok(())
        }

        fn vshr_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String> {
            emit_raw(&mut self.ops, encode_ushr_4s(dst.0, a.0, imm));
            Ok(())
        }

        fn vshl_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String> {
            // SHL Vd.4S, Vn.4S, #shift — immh:immb = shift + 32
            let immh_immb = (imm as u32) + 32;
            emit_raw(&mut self.ops, 0x4F005400
                | ((immh_immb & 0x7F) << 16)
                | ((a.0 as u32) << 5)
                | (dst.0 as u32));
            Ok(())
        }

        fn vcvt_i32_f32(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            // SCVTF Vd.4S, Vn.4S (signed i32 -> f32)
            emit_raw(&mut self.ops, 0x4E21D800 | ((a.0 as u32) << 5) | (dst.0 as u32));
            Ok(())
        }

        fn vcvt_f32_i32(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            // FCVTZS Vd.4S, Vn.4S (f32 -> signed i32, truncate toward zero)
            emit_raw(&mut self.ops, 0x4EA1B800 | ((a.0 as u32) << 5) | (dst.0 as u32));
            Ok(())
        }

        fn vround(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            // FRINTN Vd.4S, Vn.4S (round to nearest, ties to even)
            emit_raw(&mut self.ops, 0x4E218800 | ((a.0 as u32) << 5) | (dst.0 as u32));
            Ok(())
        }

        fn vadd_i32(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            // ADD Vd.4S, Vn.4S, Vm.4S (integer add)
            emit_raw(&mut self.ops, 0x4EA08400
                | ((b.0 as u32) << 16)
                | ((a.0 as u32) << 5)
                | (dst.0 as u32));
            Ok(())
        }

        // ── FMA variants ────────────────────────────────────────────

        fn vfmadd213(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            // dst = dst * a + b
            // Use v31 as scratch: mov v31, b; fmla v31, dst, a; mov dst, v31
            emit_raw(&mut self.ops, encode_mov_v(31, b.0));
            emit_raw(&mut self.ops, encode_fmla(31, dst.0, a.0));
            emit_raw(&mut self.ops, encode_mov_v(dst.0, 31));
            Ok(())
        }

        fn vfmadd231(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            // dst = a * b + dst -> FMLA dst, a, b
            emit_raw(&mut self.ops, encode_fmla(dst.0, a.0, b.0));
            Ok(())
        }

        // ── Loop control ────────────────────────────────────────────

        fn alloc_label(&mut self) -> Label {
            let id = self.label_counter;
            self.label_counter += 1;
            self.labels.push(None);
            Label(id)
        }

        fn define_label(&mut self, _label: Label) -> Result<(), String> {
            Err("define_label not yet implemented for dynasm backend (use dynasm! labels)".into())
        }

        fn jump(&mut self, _label: Label) -> Result<(), String> {
            Err("jump not yet implemented for dynasm backend (use dynasm! labels)".into())
        }

        fn dec_and_branch_nz(&mut self, _counter: BaseReg, _label: Label) -> Result<(), String> {
            Err("dec_and_branch_nz not yet implemented for dynasm backend".into())
        }

        fn cmp_and_branch_lt(&mut self, _reg: BaseReg, _imm: i64, _label: Label) -> Result<(), String> {
            Err("cmp_and_branch_lt not yet implemented for dynasm backend".into())
        }

        fn cmp_and_branch_ge(&mut self, _reg: BaseReg, _imm: i64, _label: Label) -> Result<(), String> {
            Err("cmp_and_branch_ge not yet implemented for dynasm backend".into())
        }

        // ── GPR operations ──────────────────────────────────────────

        fn gpr_load_imm(&mut self, dst: BaseReg, imm: i64) -> Result<(), String> {
            let rd = gpr_num(&dst) as u32;
            self.emit_mov_imm_gp(rd, imm as usize);
            Ok(())
        }

        fn gpr_add_imm(&mut self, dst: BaseReg, imm: i32) -> Result<(), String> {
            let rd = gpr_num(&dst) as u32;
            if imm >= 0 && (imm as u32) < 4096 {
                // ADD Xd, Xd, #imm12
                emit_raw(&mut self.ops, 0x91000000
                    | (((imm as u32) & 0xFFF) << 10)
                    | (rd << 5)
                    | rd);
            } else if imm < 0 && ((-imm) as u32) < 4096 {
                // SUB Xd, Xd, #imm12
                let abs_imm = (-imm) as u32;
                emit_raw(&mut self.ops, 0xD1000000
                    | ((abs_imm & 0xFFF) << 10)
                    | (rd << 5)
                    | rd);
            } else {
                // Large immediate: load into x15, then add
                self.emit_mov_imm_gp(15, imm.unsigned_abs() as usize);
                if imm >= 0 {
                    // ADD Xd, Xd, X15
                    emit_raw(&mut self.ops, 0x8B0F0000 | (rd << 5) | rd);
                } else {
                    // SUB Xd, Xd, X15
                    emit_raw(&mut self.ops, 0xCB0F0000 | (rd << 5) | rd);
                }
            }
            Ok(())
        }

        fn gpr_mov(&mut self, dst: BaseReg, src: BaseReg) -> Result<(), String> {
            let rd = gpr_num(&dst) as u32;
            let rn = gpr_num(&src) as u32;
            // MOV Xd, Xn (alias for ORR Xd, XZR, Xn)
            emit_raw(&mut self.ops, 0xAA0003E0 | (rn << 16) | rd);
            Ok(())
        }

        // ── Function frame ──────────────────────────────────────────

        fn emit_prologue(&mut self) -> Result<(), String> {
            DynasmAArch64CodeGen::emit_prologue(self);
            Ok(())
        }

        fn emit_epilogue(&mut self) -> Result<(), String> {
            DynasmAArch64CodeGen::emit_epilogue(self);
            Ok(())
        }

        fn finalize(&mut self) -> Result<CodegenOutput, String> {
            let old_ops = std::mem::replace(
                &mut self.ops,
                Assembler::new().map_err(|e| format!("{:?}", e))?,
            );
            let buf = old_ops.finalize().map_err(|e| format!("{:?}", e))?;
            Ok(CodegenOutput {
                code: buf.to_vec(),
                scratchpad_bytes: self.blis_scratchpad_bytes,
            })
        }

        // ── Prefetch ────────────────────────────────────────────────

        fn prefetch_l1(&mut self, _mem: MemOperand) -> Result<(), String> {
            Err("prefetch_l1 not yet implemented for dynasm backend".into())
        }

        // ── Non-temporal store ──────────────────────────────────────

        fn vstore_nt(&mut self, _mem: MemOperand, _src: VReg) -> Result<(), String> {
            Err("vstore_nt not yet implemented for dynasm backend".into())
        }

        // ── Memory fence ────────────────────────────────────────────

        fn sfence(&mut self) -> Result<(), String> {
            // DMB ISH (data memory barrier, inner shareable)
            emit_raw(&mut self.ops, 0xD5033BBF);
            Ok(())
        }

        // ── Scalar operations ───────────────────────────────────────

        fn scalar_load(&mut self, _dst: VReg, _mem: MemOperand) -> Result<(), String> {
            Err("scalar_load not yet implemented for dynasm backend".into())
        }

        fn scalar_store(&mut self, _mem: MemOperand, _src: VReg) -> Result<(), String> {
            Err("scalar_store not yet implemented for dynasm backend".into())
        }

        // ── External function calls ─────────────────────────────────

        fn call_fn_ptr(&mut self, addr: u64) -> Result<(), String> {
            self.emit_mov_imm_gp(10, addr as usize);
            dynasm!(self.ops ; .arch aarch64 ; blr x10);
            Ok(())
        }

        // ── Horizontal reductions ────────────────────────────────────

        fn vhsum(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            let s = src.0 as u32;
            let d = dst.0 as u32;
            // faddp v30.4s, v_src.4s, v_src.4s
            emit_raw(&mut self.ops, 0x6E20D400 | (s << 16) | (s << 5) | 30);
            // faddp v_dst.4s, v30.4s, v30.4s
            emit_raw(&mut self.ops, 0x6E20D400 | (30 << 16) | (30 << 5) | d);
            // dup v_dst.4s, v_dst.s[0]
            emit_raw(&mut self.ops, 0x4E040400 | (d << 5) | d);
            Ok(())
        }

        fn vhmax(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            let s = src.0 as u32;
            let d = dst.0 as u32;
            // fmaxp v30.4s, v_src.4s, v_src.4s
            emit_raw(&mut self.ops, 0x6E20F400 | (s << 16) | (s << 5) | 30);
            // fmaxp v_dst.4s, v30.4s, v30.4s
            emit_raw(&mut self.ops, 0x6E20F400 | (30 << 16) | (30 << 5) | d);
            // dup v_dst.4s, v_dst.s[0]
            emit_raw(&mut self.ops, 0x4E040400 | (d << 5) | d);
            Ok(())
        }

        // ── Masked operations (NEON: no-op / scalar fallback) ────────

        fn set_tail_mask(&mut self, _remainder: usize) -> Result<(), String> {
            Ok(())
        }

        fn vload_masked(&mut self, _dst: VReg, _mem: MemOperand) -> Result<(), String> {
            Err("vload_masked not yet implemented for dynasm backend".into())
        }

        fn vstore_masked(&mut self, _mem: MemOperand, _src: VReg) -> Result<(), String> {
            Err("vstore_masked not yet implemented for dynasm backend".into())
        }

        // ── Additional FMA variant ───────────────────────────────────

        fn vfnmadd231(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            // dst = -(a * b) + dst -> FMLS Vd.4S, Va.4S, Vb.4S
            emit_raw(&mut self.ops, 0x4EA0CC00
                | ((b.0 as u32) << 16)
                | ((a.0 as u32) << 5)
                | (dst.0 as u32));
            Ok(())
        }

        // ── NOP ─────────────────────────────────────────────────────

        fn emit_metadata_nop(&mut self) -> Result<(), String> {
            dynasm!(self.ops ; .arch aarch64 ; nop);
            Ok(())
        }

        // ── Mixed-width support ─────────────────────────────────────────

        fn current_simd_width(&self) -> crate::compiler::codegen::simd_ops::SimdWidth {
            crate::compiler::codegen::simd_ops::SimdWidth::W128
        }

        fn width_f32_lanes(&self) -> usize {
            self.simd_width
        }

        fn push_width(&mut self, w: crate::compiler::codegen::simd_ops::SimdWidth) -> Result<crate::compiler::codegen::simd_ops::SimdWidth, String> {
            use crate::compiler::codegen::simd_ops::SimdWidth;
            let prev = SimdWidth::W128;
            match w {
                SimdWidth::W128 => {
                    self.width_stack.push(prev);
                    Ok(prev)
                }
                _ => Err(format!("AArch64 NEON only supports W128, requested {:?}", w)),
            }
        }

        fn pop_width(&mut self, _prev: crate::compiler::codegen::simd_ops::SimdWidth) -> Result<(), String> {
            if self.width_stack.pop().is_none() {
                return Err("pop_width called without matching push_width".into());
            }
            Ok(())
        }
    }

    // ── TileOps stub (no tile accelerator on aarch64/NEON) ──────────────

    impl crate::compiler::codegen::tile_ops::TileOps for DynasmAArch64CodeGen {
        fn tile_configure(&mut self, _configs: &[(crate::compiler::codegen::tile_ops::TReg, crate::compiler::codegen::tile_ops::TileConfig)]) -> Result<(), String> {
            Err("Tile operations not available on aarch64/NEON".into())
        }
        fn tile_release(&mut self) -> Result<(), String> {
            Err("Tile operations not available on aarch64/NEON".into())
        }
        fn tile_zero(&mut self, _dst: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            Err("Tile operations not available on aarch64/NEON".into())
        }
        fn tile_load(&mut self, _dst: crate::compiler::codegen::tile_ops::TReg, _base: BaseReg, _stride: BaseReg) -> Result<(), String> {
            Err("Tile operations not available on aarch64/NEON".into())
        }
        fn tile_store(&mut self, _base: BaseReg, _stride: BaseReg, _src: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            Err("Tile operations not available on aarch64/NEON".into())
        }
        fn tile_dpbf16(&mut self, _dst: crate::compiler::codegen::tile_ops::TReg, _a: crate::compiler::codegen::tile_ops::TReg, _b: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            Err("Tile operations not available on aarch64/NEON".into())
        }
        fn tile_dpbssd(&mut self, _dst: crate::compiler::codegen::tile_ops::TReg, _a: crate::compiler::codegen::tile_ops::TReg, _b: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            Err("Tile operations not available on aarch64/NEON".into())
        }
        fn tile_max_rows(&self) -> u8 { 0 }
        fn tile_max_cols_bytes(&self) -> u16 { 0 }
        fn tile_count(&self) -> u8 { 0 }
        fn has_tile_ops(&self) -> bool { false }
        fn tile_accel_kind(&self) -> Option<crate::compiler::codegen::tile_ops::TileAccelKind> { None }
        fn tile_supported_dtypes(&self) -> Vec<crate::compiler::codegen::tile_ops::TileDtype> { vec![] }
    }

    // ── HwCapabilityMatrix ──────────────────────────────────────────────

    impl DynasmAArch64CodeGen {
        /// Build HwCapabilityMatrix from the current codegen state.
        pub fn hw_capability_matrix(&self) -> crate::compiler::codegen::isa_scheduler::HwCapabilityMatrix {
            use crate::compiler::codegen::simd_ops::SimdWidth;
            crate::compiler::codegen::isa_scheduler::HwCapabilityMatrix {
                simd_widths: vec![SimdWidth::W128],
                has_tile_accel: false,
                tile_accel: None,
                peak_gflops_f32: 0.0,
                peak_bandwidth_gbs: 0.0,
                roofline_crossover: 0.0,
            }
        }
    }
}

// ── MachineCodeEmitter trait impl ────────────────────────────────────

#[cfg(feature = "jit-aarch64")]
impl crate::compiler::codegen::emitter::MachineCodeEmitter for jit::DynasmAArch64CodeGen {
    fn emit_plan(
        &mut self,
        plan: &crate::compiler::fusion::FusionPlan,
        graph: &crate::compiler::graph::CompilerGraph,
        alloc: &crate::compiler::buffer_alloc::BufferAllocation,
        profile: &crate::dispatch::DeviceProfile,
        registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<super::CodegenOutput, String> {
        self.emit_plan(plan, graph, alloc, profile, registry)
    }

    fn simd_width(&self) -> usize {
        self.simd_width()
    }
}

// ── PlatformBackend impl ─────────────────────────────────────────────

#[cfg(feature = "jit-aarch64")]
pub struct DynasmArm64Backend;

#[cfg(feature = "jit-aarch64")]
impl crate::compiler::codegen::emitter::PlatformBackend for DynasmArm64Backend {
    type Emitter = jit::DynasmAArch64CodeGen;

    fn new_emitter(&self, profile: &crate::dispatch::DeviceProfile) -> Self::Emitter {
        jit::DynasmAArch64CodeGen::new(profile)
    }

    fn platform(&self) -> crate::compiler::codegen::emitter::Platform {
        crate::compiler::codegen::emitter::Platform::Aarch64 { sve: false, amx: false }
    }

    fn num_simd_regs(&self) -> usize {
        32
    }
}

#[cfg(feature = "jit-aarch64")]
pub use jit::DynasmAArch64CodeGen;

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "jit-aarch64")]
mod tests {
    use super::*;
    use crate::compiler::trace::TraceOp;

    #[test]
    fn test_dynasm_codegen_creates() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let codegen = jit::DynasmAArch64CodeGen::new(&profile);
        assert_eq!(codegen.simd_width(), 4);
    }

    #[test]
    fn test_dynasm_trace_ops_silu() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::DynasmAArch64CodeGen::new(&profile);

        let ops = vec![
            TraceOp::Input(0),
            TraceOp::Neg(0),
            TraceOp::Exp(1),
            TraceOp::Const(1.0),
            TraceOp::Add(2, 3),
            TraceOp::Div(0, 4),
        ];

        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 6);
    }

    #[test]
    fn test_dynasm_elementwise_loop() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::DynasmAArch64CodeGen::new(&profile);

        let ops = vec![TraceOp::Input(0), TraceOp::Neg(0)];
        codegen.emit_elementwise_loop_neon(&ops, 16, false).unwrap();

        assert!(codegen.code_len() > 0);
        assert_eq!(codegen.code_len() % 4, 0);
    }

    #[test]
    fn test_dynasm_gemm_8x12() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::DynasmAArch64CodeGen::new(&profile);

        codegen.emit_gemm_8x12_neon(4).unwrap();

        assert!(codegen.code_len() > 0);
        assert_eq!(codegen.code_len() % 4, 0);

        let insn_count = codegen.code_len() / 4;
        assert!(insn_count >= 70, "GEMM 8x12 should emit >= 70 insns, got {insn_count}");
    }

    #[test]
    fn test_dynasm_gemm_with_epilogue() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen_plain = jit::DynasmAArch64CodeGen::new(&profile);
        let mut codegen_epi = jit::DynasmAArch64CodeGen::new(&profile);

        codegen_plain.emit_gemm_8x12_neon(4).unwrap();
        let plain_len = codegen_plain.code_len();

        let relu_body = vec![
            TraceOp::Input(0),
            TraceOp::Const(0.0),
            TraceOp::Max(0, 1),
        ];
        let bodies: Vec<&[TraceOp]> = vec![relu_body.as_slice()];
        codegen_epi.emit_gemm_8x12_core(4, &bodies).unwrap();
        let epi_len = codegen_epi.code_len();

        assert!(epi_len > plain_len,
            "GEMM+epilogue ({epi_len}) must be longer than plain GEMM ({plain_len})");
    }

    #[test]
    fn test_dynasm_emit_plan_standalone_silu() {
        use crate::compiler::graph::{CompilerGraph, OpKind};
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::codegen::emitter::MachineCodeEmitter;
        use crate::inference::types::DType;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![16], DType::F32);
        let output = g.add_tensor("output", vec![16], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(OpKind::Silu, vec![input], vec![output], "silu");

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        };
        let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };
        let profile = crate::dispatch::DeviceProfile::detect();

        let mut emitter = jit::DynasmAArch64CodeGen::new(&profile);
        let output = emitter.emit_plan(&plan, &g, &alloc, &profile, None).unwrap();
        assert!(!output.code.is_empty(), "emit_plan should produce code");
        assert_eq!(output.code.len() % 4, 0, "code must be 4-byte aligned");
    }

    #[test]
    fn test_dynasm_backend_trait() {
        use crate::compiler::codegen::emitter::PlatformBackend;

        let backend = DynasmArm64Backend;
        let profile = crate::dispatch::DeviceProfile::detect();
        let emitter = backend.new_emitter(&profile);

        assert!(matches!(backend.platform(),
            crate::compiler::codegen::emitter::Platform::Aarch64 { .. }));
        assert_eq!(backend.num_simd_regs(), 32);
        assert_eq!(emitter.simd_width(), 4);
    }
}
