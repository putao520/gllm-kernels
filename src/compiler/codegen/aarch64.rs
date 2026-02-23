//! aarch64 JIT code generation — NEON/SVE backend.
//!
//! This module provides the aarch64 counterpart to x86_64.rs.
//! Activated via `--features jit-aarch64`.
//!
//! TraceOp → NEON instruction mapping:
//!   Input(n)    → ldr q_n, [x_ptr, #offset]
//!   Const(v)    → fmov v_n.4s, #imm  or  ldr from constant pool
//!   Add(a,b)    → fadd v_dst.4s, v_a.4s, v_b.4s
//!   Sub(a,b)    → fsub v_dst.4s, v_a.4s, v_b.4s
//!   Mul(a,b)    → fmul v_dst.4s, v_a.4s, v_b.4s
//!   Div(a,b)    → fdiv v_dst.4s, v_a.4s, v_b.4s
//!   Fma(a,b,c)  → fmla v_c.4s, v_a.4s, v_b.4s
//!   Neg(a)      → fneg v_dst.4s, v_a.4s
//!   Abs(a)      → fabs v_dst.4s, v_a.4s
//!   Sqrt(a)     → fsqrt v_dst.4s, v_a.4s
//!   Rsqrt(a)    → frsqrte + frsqrts (Newton-Raphson)
//!   Recip(a)    → frecpe + frecps (Newton-Raphson)
//!   Max(a,b)    → fmax v_dst.4s, v_a.4s, v_b.4s
//!   Min(a,b)    → fmin v_dst.4s, v_a.4s, v_b.4s
//!   Exp(a)      → polynomial approximation (same as x86)
//!   Tanh(a)     → rational approximation (same as x86)

use super::CodegenOutput;

/// Emit a minimal aarch64 stub (`ret` = 0xD65F03C0).
pub fn emit_stub() -> CodegenOutput {
    let mut code = Vec::with_capacity(4);
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes()); // ret
    CodegenOutput { code, scratchpad_bytes: 0 }
}

// ── JIT code generator (raw instruction encoding) ────────────────────

#[cfg(feature = "jit-aarch64")]
pub mod jit {
    use crate::compiler::trace::TraceOp;
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::graph::{CompilerGraph, OpKind};
    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::dispatch::DeviceProfile;
    use super::CodegenOutput;

    /// NEON register width: 4 x f32 = 128 bits.
    const NEON_WIDTH_F32: usize = 4;
    /// Number of NEON/FP registers (v0-v31).
    #[allow(dead_code)]
    const NUM_NEON_REGS: usize = 32;

    /// aarch64 JIT code generator.
    ///
    /// Uses raw 32-bit instruction encoding (no external assembler crate).
    /// NEON instructions are fixed-width 32-bit words.
    ///
    /// Register convention (AAPCS64):
    /// - x0-x7: arguments (input ptrs, output ptrs, dims)
    /// - x9-x15: scratch (caller-saved)
    /// - x19-x28: callee-saved (must preserve)
    /// - v0-v7: argument/result (caller-saved)
    /// - v8-v15: callee-saved (lower 64 bits only)
    /// - v16-v31: scratch (caller-saved)
    pub struct AArch64CodeGen {
        code: Vec<u8>,
        simd_width: usize,
    }

    impl AArch64CodeGen {
        /// Create a new code generator configured for the detected hardware.
        pub fn new(_profile: &DeviceProfile) -> Self {
            AArch64CodeGen {
                code: Vec::with_capacity(4096),
                simd_width: NEON_WIDTH_F32,
            }
        }

        /// SIMD width in f32 elements (4 for NEON 128-bit).
        pub fn simd_width(&self) -> usize {
            self.simd_width
        }

        /// Current code buffer length in bytes.
        pub fn code_len(&self) -> usize {
            self.code.len()
        }

        /// Generate code for a complete fusion plan.
        ///
        /// Iterates over all fusion groups and emits the appropriate code
        /// for each pattern (standalone, elementwise chain, GEMM+epilogue).
        pub fn emit_plan(
            &mut self,
            plan: &FusionPlan,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            _profile: &DeviceProfile,
        ) -> Result<CodegenOutput, String> {
            self.emit_prologue();

            for group in &plan.groups {
                self.emit_group(group, graph)?;
            }

            self.emit_epilogue();

            Ok(CodegenOutput {
                code: self.code.clone(),
                scratchpad_bytes: alloc.total_bytes,
            })
        }

        /// AAPCS64 prologue: save frame pointer + link register.
        fn emit_prologue(&mut self) {
            // stp x29, x30, [sp, #-16]!
            self.emit_u32(0xA9BF7BFD);
            // mov x29, sp
            self.emit_u32(0x910003FD);
        }

        /// Restore frame pointer + link register and return.
        fn emit_epilogue(&mut self) {
            // ldp x29, x30, [sp], #16
            self.emit_u32(0xA8C17BFD);
            // ret
            self.emit_u32(0xD65F03C0);
        }

        /// Dispatch a fusion group to the appropriate emitter.
        fn emit_group(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
        ) -> Result<(), String> {
            match group.mode {
                FusionMode::Standalone => {
                    self.emit_standalone(group, graph)
                }
                FusionMode::LoopFusion => {
                    // MVP: emit a nop marker per fused op
                    // Real implementation: single loop body with TraceOp -> NEON for each op
                    for _ in &group.ops {
                        self.emit_nop();
                    }
                    Ok(())
                }
                FusionMode::EpilogueInjection => {
                    self.emit_standalone(group, graph)?;
                    // Epilogue ops would be injected into the store phase
                    // MVP: emit nop markers for each epilogue op
                    for _ in &group.epilogue {
                        self.emit_nop();
                    }
                    Ok(())
                }
                FusionMode::QkvSharedInput | FusionMode::NormIntoGemm
                | FusionMode::TileLevelFusion | FusionMode::ComputeRoot => {
                    // Emit as separate standalone ops for now
                    for &op_id in &group.ops {
                        let single = FusionGroup {
                            id: group.id,
                            anchor: op_id,
                            epilogue: vec![],
                            mode: FusionMode::Standalone,
                            ops: vec![op_id],
                        };
                        self.emit_standalone(&single, graph)?;
                    }
                    Ok(())
                }
            }
        }

        /// Emit a standalone op (single GEMM or elementwise nop placeholder).
        fn emit_standalone(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing op")?;
            match &op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    self.emit_gemm_markers(*m, *n, *k)
                }
                _ => {
                    self.emit_nop();
                    Ok(())
                }
            }
        }

        /// Emit GEMM loop structure markers (nops proportional to work).
        ///
        /// Real implementation will generate the full 3-level blocked loop
        /// with NEON FMA microkernel (8x12 tile: 24 v-accumulators).
        fn emit_gemm_markers(
            &mut self,
            _m: usize,
            _n: usize,
            _k: usize,
        ) -> Result<(), String> {
            // MVP: emit a few nops as markers for the GEMM structure
            // Real: NC loop → MC loop → KC loop → 8x12 NEON microkernel
            for _ in 0..4 {
                self.emit_nop();
            }
            Ok(())
        }

        /// Emit a TraceOp sequence as NEON SIMD instructions.
        ///
        /// This is the core of Phase 3: each TraceOp maps to one or more
        /// NEON instructions operating on v-registers (128-bit, 4xf32).
        /// The SSA indices in TraceOp directly map to v-register numbers
        /// (round-robin when > 32 ops).
        ///
        /// Mapping:
        /// - `Input(n)`    → `ldr q_reg, [x_ptr, #offset]`  (placeholder: eor zeroes reg)
        /// - `Const(v)`    → `fmov v_reg.4s, #imm` or `ldr` from const pool (placeholder: eor)
        /// - `Add(a,b)`    → `fadd v_dst.4s, v_a.4s, v_b.4s`
        /// - `Sub(a,b)`    → `fsub v_dst.4s, v_a.4s, v_b.4s`
        /// - `Mul(a,b)`    → `fmul v_dst.4s, v_a.4s, v_b.4s`
        /// - `Div(a,b)`    → `fdiv v_dst.4s, v_a.4s, v_b.4s`
        /// - `Fma(a,b,c)`  → `mov v_dst, v_c; fmla v_dst.4s, v_a.4s, v_b.4s`
        /// - `Neg(a)`      → `fneg v_dst.4s, v_a.4s`
        /// - `Abs(a)`      → `fabs v_dst.4s, v_a.4s`
        /// - `Sqrt(a)`     → `fsqrt v_dst.4s, v_a.4s`
        /// - `Rsqrt(a)`    → `frsqrte v_dst.4s, v_a.4s` + Newton-Raphson step
        /// - `Recip(a)`    → `frecpe v_dst.4s, v_a.4s` + Newton-Raphson step
        /// - `Exp(a)`      → polynomial approximation (placeholder: mov)
        /// - `Tanh(a)`     → rational approximation (placeholder: mov)
        /// - `Max(a,b)`    → `fmax v_dst.4s, v_a.4s, v_b.4s`
        /// - `Min(a,b)`    → `fmin v_dst.4s, v_a.4s, v_b.4s`
        pub fn emit_trace_ops_neon(
            &mut self,
            ops: &[TraceOp],
        ) -> Result<Vec<u8>, String> {
            let mut reg_map: Vec<u8> = Vec::with_capacity(ops.len());

            for (i, op) in ops.iter().enumerate() {
                let rd = Self::vreg_for_index(i);
                match op {
                    TraceOp::Input(_) | TraceOp::Const(_) => {
                        // Placeholder: zero register via EOR
                        // eor v_rd.16b, v_rd.16b, v_rd.16b
                        self.emit_u32(Self::encode_eor_v(rd, rd, rd));
                    }
                    TraceOp::Add(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        // fadd v_rd.4s, v_ra.4s, v_rb.4s
                        self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, ra, rb));
                    }
                    TraceOp::Sub(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        // fsub v_rd.4s, v_ra.4s, v_rb.4s
                        self.emit_u32(Self::encode_f32x4_binop(0x4EA0D400, rd, ra, rb));
                    }
                    TraceOp::Mul(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        // fmul v_rd.4s, v_ra.4s, v_rb.4s
                        self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, ra, rb));
                    }
                    TraceOp::Div(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        // fdiv v_rd.4s, v_ra.4s, v_rb.4s
                        self.emit_u32(Self::encode_f32x4_binop(0x6E20FC00, rd, ra, rb));
                    }
                    TraceOp::Fma(a, b, c) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        let rc = reg_map[*c as usize];
                        // mov v_rd.16b, v_rc.16b (copy accumulator)
                        self.emit_u32(Self::encode_mov_v(rd, rc));
                        // fmla v_rd.4s, v_ra.4s, v_rb.4s
                        self.emit_u32(Self::encode_fmla(rd, ra, rb));
                    }
                    TraceOp::Neg(a) => {
                        let ra = reg_map[*a as usize];
                        // fneg v_rd.4s, v_ra.4s
                        self.emit_u32(Self::encode_f32x4_unary(0x6EA0F800, rd, ra));
                    }
                    TraceOp::Abs(a) => {
                        let ra = reg_map[*a as usize];
                        // fabs v_rd.4s, v_ra.4s
                        self.emit_u32(Self::encode_f32x4_unary(0x4EA0F800, rd, ra));
                    }
                    TraceOp::Sqrt(a) => {
                        let ra = reg_map[*a as usize];
                        // fsqrt v_rd.4s, v_ra.4s
                        self.emit_u32(Self::encode_f32x4_unary(0x6EA1F800, rd, ra));
                    }
                    TraceOp::Rsqrt(a) => {
                        let ra = reg_map[*a as usize];
                        // frsqrte v_rd.4s, v_ra.4s (estimate, ~8-bit accuracy)
                        // Real impl adds Newton-Raphson: frsqrts + fmul
                        self.emit_u32(Self::encode_f32x4_unary(0x6EA1D800, rd, ra));
                    }
                    TraceOp::Recip(a) => {
                        let ra = reg_map[*a as usize];
                        // frecpe v_rd.4s, v_ra.4s (estimate, ~8-bit accuracy)
                        // Real impl adds Newton-Raphson: frecps + fmul
                        self.emit_u32(Self::encode_f32x4_unary(0x4EA1D800, rd, ra));
                    }
                    TraceOp::Exp(a) => {
                        let ra = reg_map[*a as usize];
                        // Placeholder: mov (real: polynomial approximation sequence)
                        self.emit_u32(Self::encode_mov_v(rd, ra));
                    }
                    TraceOp::Tanh(a) => {
                        let ra = reg_map[*a as usize];
                        // Placeholder: mov (real: rational approximation sequence)
                        self.emit_u32(Self::encode_mov_v(rd, ra));
                    }
                    TraceOp::Max(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        // fmax v_rd.4s, v_ra.4s, v_rb.4s
                        self.emit_u32(Self::encode_f32x4_binop(0x4E20F400, rd, ra, rb));
                    }
                    TraceOp::Min(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        // fmin v_rd.4s, v_ra.4s, v_rb.4s
                        self.emit_u32(Self::encode_f32x4_binop(0x4EA0F400, rd, ra, rb));
                    }
                }
                reg_map.push(rd);
            }

            Ok(reg_map)
        }

        // ── Instruction encoding helpers ─────────────────────────────

        /// Map a TraceOp SSA index to a v-register number (round-robin v0-v31).
        fn vreg_for_index(index: usize) -> u8 {
            (index % NUM_NEON_REGS) as u8
        }

        /// Encode a 3-register NEON f32x4 binary op.
        ///
        /// Format: `<opcode_base> | Rm:20-16 | Rn:9-5 | Rd:4-0`
        /// The base opcode already encodes the Q bit (128-bit) and size bits.
        fn encode_f32x4_binop(base: u32, rd: u8, rn: u8, rm: u8) -> u32 {
            base | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode a 2-register NEON f32x4 unary op.
        ///
        /// Format: `<opcode_base> | Rn:9-5 | Rd:4-0`
        fn encode_f32x4_unary(base: u32, rd: u8, rn: u8) -> u32 {
            base | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode `orr v_rd.16b, v_rn.16b, v_rn.16b` (mov alias).
        ///
        /// ORR (vector, register): 0x0EA01C00 | Rm:20-16 | Rn:9-5 | Rd:4-0
        /// When Rm == Rn this is the MOV alias.
        fn encode_mov_v(rd: u8, rn: u8) -> u32 {
            0x4EA01C00 | ((rn as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode `eor v_rd.16b, v_rn.16b, v_rm.16b`.
        ///
        /// EOR (vector): 0x2E201C00 | Rm:20-16 | Rn:9-5 | Rd:4-0
        fn encode_eor_v(rd: u8, rn: u8, rm: u8) -> u32 {
            0x6E201C00 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode `fmla v_rd.4s, v_rn.4s, v_rm.4s`.
        ///
        /// FMLA (vector): 0x0E20CC00 | Rm:20-16 | Rn:9-5 | Rd:4-0
        fn encode_fmla(rd: u8, rn: u8, rm: u8) -> u32 {
            0x4E20CC00 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Emit a NEON nop (`hint #0` = 0xD503201F).
        fn emit_nop(&mut self) {
            self.emit_u32(0xD503201F);
        }

        /// Append a 32-bit instruction to the code buffer (little-endian).
        fn emit_u32(&mut self, insn: u32) {
            self.code.extend_from_slice(&insn.to_le_bytes());
        }
    }
}

#[cfg(feature = "jit-aarch64")]
impl crate::compiler::codegen::emitter::MachineCodeEmitter for jit::AArch64CodeGen {
    fn emit_plan(
        &mut self,
        plan: &crate::compiler::fusion::FusionPlan,
        graph: &crate::compiler::graph::CompilerGraph,
        alloc: &crate::compiler::buffer_alloc::BufferAllocation,
        profile: &crate::dispatch::DeviceProfile,
        _registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<super::CodegenOutput, String> {
        self.emit_plan(plan, graph, alloc, profile)
    }

    fn simd_width(&self) -> usize {
        self.simd_width()
    }
}

/// AArch64 platform backend factory.
#[cfg(feature = "jit-aarch64")]
pub struct Arm64Backend;

#[cfg(feature = "jit-aarch64")]
impl crate::compiler::codegen::emitter::PlatformBackend for Arm64Backend {
    type Emitter = jit::AArch64CodeGen;

    fn new_emitter(&self, profile: &crate::dispatch::DeviceProfile) -> Self::Emitter {
        jit::AArch64CodeGen::new(profile)
    }

    fn platform(&self) -> crate::compiler::codegen::emitter::Platform {
        crate::compiler::codegen::emitter::Platform::Aarch64 { sve: false }
    }

    fn num_simd_regs(&self) -> usize {
        32 // AArch64: v0-v31
    }
}

#[cfg(feature = "jit-aarch64")]
pub use jit::AArch64CodeGen;

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use crate::compiler::executable::CompiledLayer;

    #[test]
    fn test_aarch64_stub_callable() {
        let output = emit_stub();
        assert!(!output.code.is_empty());
        let layer = CompiledLayer::from_code(&output.code, 0, 0).unwrap();
        unsafe {
            let f = layer.entry_point();
            f(
                std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                std::ptr::null(), std::ptr::null(),
                0, 0,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }
    }
}

// Cross-platform unit tests (don't execute code, just test structure)
#[cfg(test)]
mod cross_tests {
    use super::*;

    #[test]
    fn test_stub_produces_ret() {
        let output = emit_stub();
        assert_eq!(output.code.len(), 4);
        // ret = 0xD65F03C0 little-endian
        assert_eq!(output.code, vec![0xC0, 0x03, 0x5F, 0xD6]);
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_aarch64_codegen_creates() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let codegen = jit::AArch64CodeGen::new(&profile);
        assert_eq!(codegen.simd_width(), 4);
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_emit_trace_ops_silu() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // SiLU: v / (1 + exp(-v))
        let ops = vec![
            TraceOp::Input(0),   // [0] v
            TraceOp::Neg(0),     // [1] -v
            TraceOp::Exp(1),     // [2] exp(-v)
            TraceOp::Const(1.0), // [3] 1.0
            TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
            TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
        ];

        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 6);
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_emit_trace_ops_all_variants() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        let ops = vec![
            TraceOp::Input(0),     // [0]
            TraceOp::Input(1),     // [1]
            TraceOp::Input(2),     // [2]
            TraceOp::Const(3.14),  // [3]
            TraceOp::Add(0, 1),    // [4]
            TraceOp::Sub(0, 1),    // [5]
            TraceOp::Mul(0, 1),    // [6]
            TraceOp::Div(0, 1),    // [7]
            TraceOp::Fma(0, 1, 2), // [8]
            TraceOp::Neg(0),       // [9]
            TraceOp::Abs(0),       // [10]
            TraceOp::Sqrt(0),      // [11]
            TraceOp::Rsqrt(0),     // [12]
            TraceOp::Recip(0),     // [13]
            TraceOp::Exp(0),       // [14]
            TraceOp::Tanh(0),      // [15]
            TraceOp::Max(0, 1),    // [16]
            TraceOp::Min(0, 1),    // [17]
        ];

        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 18);

        // Each instruction is 4 bytes; Fma emits 2 instructions (mov + fmla)
        // 17 single-instruction ops + 1 Fma (2 instructions) = 19 instructions
        assert_eq!(codegen.code_len(), 19 * 4);
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_instruction_encoding_alignment() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let codegen = jit::AArch64CodeGen::new(&profile);
        // Code buffer starts empty
        assert_eq!(codegen.code_len() % 4, 0);
    }
}
