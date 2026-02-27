//! AArch64 JIT code generator backed by dynasm-rs.
//!
//! Feature-gated behind `jit-aarch64-dynasm`. Provides the same
//! `MachineCodeEmitter` / `PlatformBackend` interface as the manual-encoding
//! backend in `aarch64.rs`, but uses `dynasm!` macros for instruction emission
//! and automatic label resolution — eliminating manual branch-offset patching.
//!
//! NEON SIMD instructions that dynasm-rs may not directly support are emitted
//! via `DynasmApi::push_u32` using the same proven encoding helpers ported from
//! the manual backend.

#[cfg(feature = "jit-aarch64-dynasm")]
pub mod jit {
    use dynasm::dynasm;
    use dynasmrt::{aarch64::Assembler, DynasmApi, DynasmLabelApi};

    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::compiler::codegen::CodegenOutput;
    use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
    use crate::compiler::graph::{CompilerGraph, OpId, OpKind};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::trace::{ComputePattern, TraceOp};
    use crate::dispatch::DeviceProfile;

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
            let s0 = 28u8;
            let s_half = 29u8;
            emit_raw(&mut self.ops, encode_fmov_half(s_half));
            emit_raw(&mut self.ops, encode_fmov_one(rd));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, rd, rd, ra)); // fadd
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, s0, ra, ra)); // fmul
            emit_raw(&mut self.ops, encode_fmla(rd, s0, s_half));
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
            self.emit_load_f32_const_neon(30, -2.0);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 30, ra, 30));
            self.emit_exp_neon(rd, 30);
            emit_raw(&mut self.ops, encode_fmov_one(28));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, rd, rd, 28));
            emit_raw(&mut self.ops, encode_f32x4_unary(0x4EA1D800, 29, rd));
            emit_raw(&mut self.ops, encode_frecps(30, rd, 29));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 29, 29, 30));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, rd, 29, 29));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, rd, rd, 28));
        }

        fn emit_log_neon(&mut self, rd: u8, ra: u8) {
            emit_raw(&mut self.ops, encode_ushr_4s(28, ra, 23));
            emit_raw(&mut self.ops, encode_ucvtf_4s(28, 28));
            self.emit_load_f32_const_neon(30, 127.0);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 28, 28, 30));
            self.emit_load_f32_const_neon(30, f32::from_bits(0x007F_FFFF));
            emit_raw(&mut self.ops, encode_and_v(29, ra, 30));
            self.emit_load_f32_const_neon(30, 1.0);
            emit_raw(&mut self.ops, encode_orr_v(29, 29, 30));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 29, 29, 30));
            self.emit_load_f32_const_neon(rd, -0.24073381);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, rd, 29));
            self.emit_load_f32_const_neon(30, 0.33179903);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, rd, rd, 30));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, rd, 29));
            self.emit_load_f32_const_neon(30, -0.49987412);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, rd, rd, 30));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, rd, 29));
            self.emit_load_f32_const_neon(30, 0.99999934);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, rd, rd, 30));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, rd, 29));
            self.emit_load_f32_const_neon(30, 0.6931471805599453_f32);
            emit_raw(&mut self.ops, encode_fmla(rd, 28, 30));
        }

        // ── Scratch-based NEON helpers (for epilogue on accumulators) ──

        fn emit_exp_scratch_neon(&mut self) {
            self.emit_load_f32_const_neon(1, 0.5);
            emit_raw(&mut self.ops, encode_fmov_one(2));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 2, 2, 0));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 3, 0, 0));
            emit_raw(&mut self.ops, encode_fmla(2, 3, 1));
            emit_raw(&mut self.ops, encode_mov_v(0, 2));
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
            self.emit_load_f32_const_neon(1, -2.0);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 1, 0, 1));
            self.emit_load_f32_const_neon(2, 0.5);
            emit_raw(&mut self.ops, encode_fmov_one(3));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 3, 3, 1));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 1, 1));
            emit_raw(&mut self.ops, encode_fmla(3, 4, 2));
            emit_raw(&mut self.ops, encode_fmov_one(2));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 3, 3, 2));
            emit_raw(&mut self.ops, encode_f32x4_unary(0x4EA1D800, 4, 3));
            emit_raw(&mut self.ops, encode_frecps(5, 3, 4));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 4, 5));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 0, 4, 4));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 0, 0, 2));
        }

        fn emit_log_scratch_neon(&mut self) {
            emit_raw(&mut self.ops, encode_ushr_4s(1, 0, 23));
            emit_raw(&mut self.ops, encode_ucvtf_4s(1, 1));
            self.emit_load_f32_const_neon(2, 127.0);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 1, 1, 2));
            self.emit_load_f32_const_neon(2, f32::from_bits(0x007F_FFFF));
            emit_raw(&mut self.ops, encode_and_v(3, 0, 2));
            self.emit_load_f32_const_neon(2, 1.0);
            emit_raw(&mut self.ops, encode_orr_v(3, 3, 2));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, 3, 3, 2));
            self.emit_load_f32_const_neon(4, -0.24073381);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 4, 3));
            self.emit_load_f32_const_neon(5, 0.33179903);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 4, 4, 5));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 4, 3));
            self.emit_load_f32_const_neon(5, -0.49987412);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 4, 4, 5));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 4, 3));
            self.emit_load_f32_const_neon(5, 0.99999934);
            emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, 4, 4, 5));
            emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, 4, 4, 3));
            self.emit_load_f32_const_neon(5, 0.6931471805599453_f32);
            emit_raw(&mut self.ops, encode_fmla(4, 1, 5));
            emit_raw(&mut self.ops, encode_mov_v(0, 4));
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
            let mut reg_map: Vec<u8> = Vec::with_capacity(ops_trace.len());

            for (i, op) in ops_trace.iter().enumerate() {
                let rd = Self::vreg_for_index(i);
                match op {
                    TraceOp::Input(n) => {
                        // Convention: callers pre-load Input(n) into v_n before
                        // calling emit_trace_ops_neon.  Since Input(n) always
                        // appears at trace index n, vreg_for_index(n) == n == rd,
                        // so the data is already in the correct register — no
                        // instruction emitted.
                        debug_assert_eq!(rd, *n as u8,
                            "Input(n) must appear at trace index n");
                    }
                    TraceOp::Const(v) => {
                        // Load the f32 constant into all 4 lanes of v_rd.
                        // Uses emit_load_f32_const_neon: movz+movk+dup (2-3 insns).
                        self.emit_load_f32_const_neon(rd, *v as f32);
                    }
                    TraceOp::Add(a, b) => {
                        let (ra, rb) = (reg_map[*a as usize], reg_map[*b as usize]);
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20D400, rd, ra, rb));
                    }
                    TraceOp::Sub(a, b) => {
                        let (ra, rb) = (reg_map[*a as usize], reg_map[*b as usize]);
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0D400, rd, ra, rb));
                    }
                    TraceOp::Mul(a, b) => {
                        let (ra, rb) = (reg_map[*a as usize], reg_map[*b as usize]);
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20DC00, rd, ra, rb));
                    }
                    TraceOp::Div(a, b) => {
                        let (ra, rb) = (reg_map[*a as usize], reg_map[*b as usize]);
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x6E20FC00, rd, ra, rb));
                    }
                    TraceOp::Fma(a, b, c) => {
                        let (ra, rb, rc) = (
                            reg_map[*a as usize],
                            reg_map[*b as usize],
                            reg_map[*c as usize],
                        );
                        emit_raw(&mut self.ops, encode_mov_v(rd, rc));
                        emit_raw(&mut self.ops, encode_fmla(rd, ra, rb));
                    }
                    TraceOp::Neg(a) => {
                        let ra = reg_map[*a as usize];
                        emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA0F800, rd, ra));
                    }
                    TraceOp::Abs(a) => {
                        let ra = reg_map[*a as usize];
                        emit_raw(&mut self.ops, encode_f32x4_unary(0x4EA0F800, rd, ra));
                    }
                    TraceOp::Sqrt(a) => {
                        let ra = reg_map[*a as usize];
                        emit_raw(&mut self.ops, encode_f32x4_unary(0x6EA1F800, rd, ra));
                    }
                    TraceOp::Rsqrt(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_rsqrt_refined_neon(rd, ra);
                    }
                    TraceOp::Recip(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_recip_refined_neon(rd, ra);
                    }
                    TraceOp::Exp(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_exp_neon(rd, ra);
                    }
                    TraceOp::Tanh(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_tanh_neon(rd, ra);
                    }
                    TraceOp::Log(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_log_neon(rd, ra);
                    }
                    TraceOp::Max(a, b) => {
                        let (ra, rb) = (reg_map[*a as usize], reg_map[*b as usize]);
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4E20F400, rd, ra, rb));
                    }
                    TraceOp::Min(a, b) => {
                        let (ra, rb) = (reg_map[*a as usize], reg_map[*b as usize]);
                        emit_raw(&mut self.ops, encode_f32x4_binop(0x4EA0F400, rd, ra, rb));
                    }
                }
                reg_map.push(rd);
            }

            Ok(reg_map)
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
            let blocking = profile.gemm_blocking(m, n, k);
            if k <= blocking.kc && m <= blocking.mc && n <= blocking.nc {
                return self.emit_gemm_8x12_neon(k);
            }
            self.emit_gemm_blis_core(m, n, k, profile, &[])
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
                        _ => 1e-5,
                    };
                    self.emit_norm_standalone_neon(reduce, finalize, transform, elem_count, eps)
                }
                ComputePattern::Reduction { identity, combine } => {
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
                OpKind::Reshape { .. } | OpKind::Transpose { .. } => Ok(()),
                _ => {
                    let elem_count = op.outputs.first()
                        .and_then(|&out_id| graph.tensor_numel(out_id))
                        .unwrap_or(0);
                    if elem_count == 0 { return Ok(()); }

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
            let last_op = graph.op(*group.ops.last().ok_or("empty fusion group")?)
                .ok_or("missing last op")?;
            let elem_count = last_op.outputs.first()
                .and_then(|&out_id| graph.tensor_numel(out_id))
                .unwrap_or(0);
            if elem_count == 0 { return Ok(()); }

            let mut trace_bodies: Vec<(Vec<TraceOp>, bool)> = Vec::new();
            if let Some(reg) = registry {
                for &op_id in &group.ops {
                    let op = graph.op(op_id).ok_or("missing op in fusion group")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            let is_binary = matches!(trace.pattern, ComputePattern::BinaryElementwise { .. });
                            trace_bodies.push((body.to_vec(), is_binary));
                        }
                    }
                }
            }

            if trace_bodies.is_empty() {
                let anchor_op = graph.op(group.anchor).ok_or("missing anchor")?;
                return self.emit_inline_elementwise(&anchor_op.kind, elem_count);
            }

            let has_binary = trace_bodies.iter().any(|(_, b)| *b);

            let first_op = graph.op(*group.ops.first().unwrap()).ok_or("missing first op")?;
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
            if let Some(&output_tid) = last_op.outputs.first() {
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
                    _ => { dynasm!(self.ops ; .arch aarch64 ; nop); }
                }
            }
            let anchor_op = graph.op(group.anchor).ok_or("missing anchor op")?;
            match &anchor_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile)
                }
                _ => { dynasm!(self.ops ; .arch aarch64 ; nop); Ok(()) }
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
}

// ── MachineCodeEmitter trait impl ────────────────────────────────────

#[cfg(feature = "jit-aarch64-dynasm")]
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

#[cfg(feature = "jit-aarch64-dynasm")]
pub struct DynasmArm64Backend;

#[cfg(feature = "jit-aarch64-dynasm")]
impl crate::compiler::codegen::emitter::PlatformBackend for DynasmArm64Backend {
    type Emitter = jit::DynasmAArch64CodeGen;

    fn new_emitter(&self, profile: &crate::dispatch::DeviceProfile) -> Self::Emitter {
        jit::DynasmAArch64CodeGen::new(profile)
    }

    fn platform(&self) -> crate::compiler::codegen::emitter::Platform {
        crate::compiler::codegen::emitter::Platform::Aarch64 { sve: false }
    }

    fn num_simd_regs(&self) -> usize {
        32
    }
}

#[cfg(feature = "jit-aarch64-dynasm")]
pub use jit::DynasmAArch64CodeGen;

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "jit-aarch64-dynasm")]
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
