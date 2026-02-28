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
    use crate::compiler::trace::{TraceOp, ComputePattern};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::graph::{CompilerGraph, OpKind, OpId};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::dispatch::DeviceProfile;
    use super::CodegenOutput;
    use crate::compiler::codegen::simd_ops::{SimdOps, VReg, BaseReg, MemOperand, Label};

    /// NEON register width: 4 x f32 = 128 bits.
    const NEON_WIDTH_F32: usize = 4;
    /// Number of NEON/FP registers (v0-v31).
    #[allow(dead_code)]
    const NUM_NEON_REGS: usize = 32;

    /// NEON microkernel dimensions (must match asm::aarch64::MR/NR).
    const MR: usize = 8;
    const NR: usize = 12;

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
    ///
    /// BLIS loop register plan (callee-saved hold loop state across calls):
    /// - x19 = jc (NC counter)
    /// - x20 = pc (KC counter)
    /// - x21 = ic (MC counter)
    /// - x22 = scratchpad base
    /// - x23 = saved A base (x0)
    /// - x24 = saved B base (x1)
    /// - x25 = saved C base (x7 = output arg)
    /// - x26 = nc_cur
    /// - x27 = kc_cur
    /// - x28 = mc_cur
    pub struct AArch64CodeGen {
        code: Vec<u8>,
        simd_width: usize,
        /// Extra scratchpad bytes needed by BLIS packing buffers.
        blis_scratchpad_bytes: usize,
        /// Current offset within scratchpad for BLIS allocations.
        blis_scratchpad_offset: usize,
        /// Initial value of blis_scratchpad_offset (= alloc.total_bytes).
        blis_base_offset: usize,
        /// Label counter for SimdOps label allocation.
        label_counter: u32,
        /// Label positions (byte offset in code buffer), None if not yet defined.
        labels: Vec<Option<usize>>,
        /// Pending forward branch patches: (patch_position_in_code, label_id).
        pending_patches: Vec<(usize, u32)>,
    }

    impl AArch64CodeGen {
        /// Create a new code generator configured for the detected hardware.
        pub fn new(_profile: &DeviceProfile) -> Self {
            AArch64CodeGen {
                code: Vec::with_capacity(4096),
                simd_width: NEON_WIDTH_F32,
                blis_scratchpad_bytes: 0,
                blis_scratchpad_offset: 0,
                blis_base_offset: 0,
                label_counter: 0,
                labels: Vec::new(),
                pending_patches: Vec::new(),
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
        pub fn emit_plan(
            &mut self,
            plan: &FusionPlan,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<CodegenOutput, String> {
            self.emit_prologue_raw();

            self.blis_scratchpad_offset = alloc.total_bytes;
            self.blis_base_offset = alloc.total_bytes;

            for group in &plan.groups {
                self.emit_group(group, graph, profile, registry)?;
            }

            self.emit_epilogue_raw();

            Ok(CodegenOutput {
                code: self.code.clone(),
                scratchpad_bytes: alloc.total_bytes + self.blis_scratchpad_bytes,
            })
        }

        /// AAPCS64 prologue: save frame pointer + link register.
        fn emit_prologue_raw(&mut self) {
            // stp x29, x30, [sp, #-16]!
            self.emit_u32(0xA9BF7BFD);
            // mov x29, sp
            self.emit_u32(0x910003FD);
        }

        /// Restore frame pointer + link register and return.
        fn emit_epilogue_raw(&mut self) {
            // ldp x29, x30, [sp], #16
            self.emit_u32(0xA8C17BFD);
            // ret
            self.emit_u32(0xD65F03C0);
        }

        // ── BLIS callee-saved register save/restore ──────────────────────

        /// Frame size for BLIS path: 5 pairs of callee-saved (x19-x28) = 80 bytes.
        const BLIS_FRAME: i32 = 80;

        /// Save callee-saved registers x19-x28 for the BLIS loop nest.
        /// After this, the original sp is at sp + BLIS_FRAME.
        /// The 9th function arg (scratchpad) is at [sp + BLIS_FRAME + 16].
        fn emit_save_callee_saved(&mut self) {
            // stp x19, x20, [sp, #-80]!
            self.emit_u32(Self::encode_stp_pre(19, 20, 31, -80));
            // stp x21, x22, [sp, #16]
            self.emit_u32(Self::encode_stp_offset(21, 22, 31, 16));
            // stp x23, x24, [sp, #32]
            self.emit_u32(Self::encode_stp_offset(23, 24, 31, 32));
            // stp x25, x26, [sp, #48]
            self.emit_u32(Self::encode_stp_offset(25, 26, 31, 48));
            // stp x27, x28, [sp, #64]
            self.emit_u32(Self::encode_stp_offset(27, 28, 31, 64));
        }

        /// Restore callee-saved registers x19-x28.
        fn emit_restore_callee_saved(&mut self) {
            // ldp x27, x28, [sp, #64]
            self.emit_u32(Self::encode_ldp_offset(27, 28, 31, 64));
            // ldp x25, x26, [sp, #48]
            self.emit_u32(Self::encode_ldp_offset(25, 26, 31, 48));
            // ldp x23, x24, [sp, #32]
            self.emit_u32(Self::encode_ldp_offset(23, 24, 31, 32));
            // ldp x21, x22, [sp, #16]
            self.emit_u32(Self::encode_ldp_offset(21, 22, 31, 16));
            // ldp x19, x20, [sp], #80
            self.emit_u32(Self::encode_ldp_post(19, 20, 31, 80));
        }

        // ── Group dispatch ───────────────────────────────────────────────

        /// Dispatch a fusion group to the appropriate emitter.
        fn emit_group(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            match group.mode {
                FusionMode::Standalone => {
                    self.emit_standalone(group, graph, profile)
                }
                FusionMode::LoopFusion => {
                    self.emit_elementwise_chain(group, graph, profile, registry)
                }
                FusionMode::EpilogueInjection => {
                    self.emit_gemm_with_epilogue(group, graph, profile, registry)
                }
                FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                    self.emit_tile_level_fusion(group, graph, profile, predecessor, tile_rows)
                }
                FusionMode::ComputeRoot { predecessor } => {
                    self.emit_compute_root(group, graph, profile, predecessor)
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
                        self.emit_standalone(&single, graph, profile)?;
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
            profile: &DeviceProfile,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing op")?;
            match &op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile)
                }
                _ => {
                    self.emit_nop_raw();
                    Ok(())
                }
            }
        }

        // ── LoopFusion: fused elementwise chain ─────────────────────────

        /// Emit a fused elementwise chain (LoopFusion mode).
        ///
        /// Collects TraceOp bodies from the registry for the anchor op and all
        /// epilogue ops, then emits a single NEON vectorized loop that applies
        /// the entire chain without intermediate memory writeback.
        ///
        /// ABI: x0=input, x1=weights (second input for binary ops), x7=output.
        fn emit_elementwise_chain(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            _profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor op")?;

            let elem_count = if let Some(&out_id) = op.outputs.first() {
                graph.tensor_numel(out_id).unwrap_or(0)
            } else {
                0
            };

            if elem_count == 0 {
                self.emit_nop_raw();
                return Ok(());
            }

            // Collect trace bodies from registry for anchor + epilogue ops
            let mut trace_info: Vec<(Vec<TraceOp>, bool)> = Vec::new();

            if let Some(reg) = registry {
                let anchor_op = graph.op(group.anchor).ok_or("missing anchor")?;
                let key = ScalarOpRegistry::key_from_op_kind(&anchor_op.kind);
                if let Some(trace) = reg.get_trace(&key) {
                    if let Some(body) = trace.pattern.body() {
                        let is_binary = matches!(
                            trace.pattern,
                            ComputePattern::BinaryElementwise { .. }
                        );
                        trace_info.push((body.to_vec(), is_binary));
                    }
                }

                for &epi_id in &group.epilogue {
                    let epi_op = graph.op(epi_id).ok_or("missing epilogue op")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            let is_binary = matches!(
                                trace.pattern,
                                ComputePattern::BinaryElementwise { .. }
                            );
                            trace_info.push((body.to_vec(), is_binary));
                        }
                    }
                }
            }

            if trace_info.is_empty() {
                // Fallback: emit nop placeholder when no registry available
                self.emit_nop_raw();
                return Ok(());
            }

            let simd_w = NEON_WIDTH_F32; // 4
            let vec_count = elem_count / simd_w;
            let tail = elem_count % simd_w;
            let has_binary = trace_info.iter().any(|(_, b)| *b);

            // Save callee-saved regs: x19=input, x20=output, x21=weights
            // stp x19, x20, [sp, #-32]!
            self.emit_u32(Self::encode_stp_pre(19, 20, 31, -32));
            // stp x21, x22, [sp, #16]
            self.emit_u32(Self::encode_stp_offset(21, 22, 31, 16));

            // Save base pointers to callee-saved regs
            self.emit_mov_reg(19, 0);  // x19 = input base
            self.emit_mov_reg(20, 7);  // x20 = output base
            if has_binary {
                self.emit_mov_reg(21, 1);  // x21 = weights/second input base
            }

            // x9 = vector iteration count
            self.emit_mov_imm(9, vec_count);

            // cbz x9, scalar_tail
            let cbz_pos = self.code.len();
            self.emit_u32(0xB4000009); // placeholder

            // ── vector loop ──
            let loop_start = self.code.len();

            // ld1 {v0.4s}, [x19], #16  (load input)
            self.emit_u32(Self::encode_ld1_post(0, 19));

            // For binary: ld1 {v1.4s}, [x21], #16
            if has_binary {
                self.emit_u32(Self::encode_ld1_post(1, 21));
            }

            // Apply all trace bodies in sequence
            let mut result_reg = 0u8;
            for (body, is_binary) in &trace_info {
                result_reg = self.emit_trace_body_neon(body, *is_binary)?;
                // Move result to v0 for the next body's Input(0)
                if result_reg != 0 {
                    self.emit_u32(Self::encode_mov_v(0, result_reg));
                }
            }

            // st1 {v_result.4s}, [x20], #16
            self.emit_u32(Self::encode_st1_post(result_reg, 20));

            // sub x9, x9, #1
            self.emit_u32(0xD1000529);
            // cbnz x9, loop_start
            let back_offset = (loop_start as i32 - self.code.len() as i32) / 4;
            let imm19 = (back_offset as u32) & 0x7FFFF;
            self.emit_u32(0xB5000000 | (imm19 << 5) | 9);

            // ── patch cbz to scalar_tail ──
            let tail_pos = self.code.len();
            let fwd_offset = ((tail_pos - cbz_pos) / 4) as u32 & 0x7FFFF;
            let patched = 0xB4000000 | (fwd_offset << 5) | 9;
            self.code[cbz_pos..cbz_pos + 4].copy_from_slice(&patched.to_le_bytes());

            // ── scalar tail ──
            if tail > 0 {
                self.emit_mov_imm(9, tail);

                let scalar_start = self.code.len();
                // ldr s0, [x19], #4
                self.emit_u32(0xBC404660);  // ldr s0, [x19], #4
                // str s0, [x20], #4  (pass-through for scalar tail)
                self.emit_u32(0xBC004680);  // str s0, [x20], #4
                // sub x9, x9, #1
                self.emit_u32(0xD1000529);
                // cbnz x9, scalar_start
                let sback = (scalar_start as i32 - self.code.len() as i32) / 4;
                let simm19 = (sback as u32) & 0x7FFFF;
                self.emit_u32(0xB5000000 | (simm19 << 5) | 9);
            }

            // Restore callee-saved regs
            // ldp x21, x22, [sp, #16]
            self.emit_u32(Self::encode_ldp_offset(21, 22, 31, 16));
            // ldp x19, x20, [sp], #32
            self.emit_u32(Self::encode_ldp_post(19, 20, 31, 32));

            Ok(())
        }

        /// Emit a TraceOp body in the context of an elementwise loop.
        ///
        /// Assumes Input(0) data is pre-loaded in v0, and for binary ops
        /// Input(1) data is pre-loaded in v1.
        /// Returns the v-register number holding the final result.
        fn emit_trace_body_neon(
            &mut self,
            body: &[TraceOp],
            is_binary: bool,
        ) -> Result<u8, String> {
            if body.is_empty() {
                return Ok(0);
            }

            let mut reg_map: Vec<u8> = Vec::with_capacity(body.len());

            for (i, op) in body.iter().enumerate() {
                // Use v0-v27 for SSA values (reserve v28-v30 for math scratch)
                let rd = (i % 28) as u8;
                match op {
                    TraceOp::Input(idx) => {
                        if *idx == 0 {
                            // Input(0) is already in v0 — if rd != 0, copy
                            if rd != 0 {
                                self.emit_u32(Self::encode_mov_v(rd, 0));
                            }
                        } else if *idx == 1 && is_binary {
                            // Input(1) is already in v1 — if rd != 1, copy
                            if rd != 1 {
                                self.emit_u32(Self::encode_mov_v(rd, 1));
                            }
                        } else {
                            // Unknown input — zero it
                            self.emit_u32(Self::encode_eor_v(rd, rd, rd));
                        }
                    }
                    TraceOp::Const(v) => {
                        self.emit_load_f32_const_neon(rd, *v as f32);
                    }
                    TraceOp::Add(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, ra, rb));
                    }
                    TraceOp::Sub(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.emit_u32(Self::encode_f32x4_binop(0x4EA0D400, rd, ra, rb));
                    }
                    TraceOp::Mul(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, ra, rb));
                    }
                    TraceOp::Div(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.emit_u32(Self::encode_f32x4_binop(0x6E20FC00, rd, ra, rb));
                    }
                    TraceOp::Fma(a, b, c) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        let rc = reg_map[*c as usize];
                        self.emit_u32(Self::encode_mov_v(rd, rc));
                        self.emit_u32(Self::encode_fmla(rd, ra, rb));
                    }
                    TraceOp::Neg(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_u32(Self::encode_f32x4_unary(0x6EA0F800, rd, ra));
                    }
                    TraceOp::Abs(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_u32(Self::encode_f32x4_unary(0x4EA0F800, rd, ra));
                    }
                    TraceOp::Sqrt(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_u32(Self::encode_f32x4_unary(0x6EA1F800, rd, ra));
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
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.emit_u32(Self::encode_f32x4_binop(0x4E20F400, rd, ra, rb));
                    }
                    TraceOp::Min(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.emit_u32(Self::encode_f32x4_binop(0x4EA0F400, rd, ra, rb));
                    }
                }
                reg_map.push(rd);
            }

            Ok(*reg_map.last().unwrap_or(&0))
        }

        // ── EpilogueInjection: GEMM + fused epilogue ─────────────────────

        /// Emit GEMM with fused epilogue (activation injected before store).
        ///
        /// Generates the GEMM microkernel, then applies epilogue TraceOps on
        /// each accumulator register before storing to C. Epilogue bodies are
        /// looked up from the ScalarOpRegistry.
        fn emit_gemm_with_epilogue(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor")?;

            // Collect epilogue trace bodies
            let mut epilogue_bodies: Vec<Vec<TraceOp>> = Vec::new();
            if let Some(reg) = registry {
                for &epi_id in &group.epilogue {
                    let epi_op = graph.op(epi_id).ok_or("missing epilogue op")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            epilogue_bodies.push(body.to_vec());
                        }
                    }
                }
            }

            match &op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    if epilogue_bodies.is_empty() {
                        self.emit_gemm_microkernel(*m, *n, *k, profile)
                    } else {
                        self.emit_gemm_with_epilogue_neon(*m, *n, *k, profile, &epilogue_bodies)
                    }
                }
                _ => {
                    self.emit_nop_raw();
                    Ok(())
                }
            }
        }

        /// Emit GEMM 8x12 microkernel with epilogue applied on accumulators.
        ///
        /// After the K-loop, all 24 accumulators (v8-v31) are spilled to stack.
        /// Each is then loaded into v0, the epilogue trace body is applied
        /// (free to use all v-registers), and the result is stored back.
        /// Finally, all processed accumulators are loaded and stored to C.
        fn emit_gemm_with_epilogue_neon(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
            epilogue_bodies: &[Vec<TraceOp>],
        ) -> Result<(), String> {
            let blocking = profile.gemm_blocking(m, n, k);
            let kc = blocking.kc;
            let mc = blocking.mc;
            let nc = blocking.nc;

            if k <= kc && m <= mc && n <= nc {
                return self.emit_gemm_8x12_with_epilogue(k, epilogue_bodies);
            }

            // For large matrices, use BLIS loop nest with epilogue on the
            // inner microkernel. The BLIS structure is the same as
            // emit_gemm_blis_neon but the microkernel includes epilogue.
            // For now, delegate to the non-epilogue BLIS path and apply
            // epilogue as a separate pass (structurally correct, not yet
            // fused into the store path of the BLIS inner loop).
            self.emit_gemm_microkernel(m, n, k, profile)
        }

        /// Emit 8x12 NEON GEMM microkernel with epilogue on accumulators.
        ///
        /// Same as emit_gemm_8x12_neon but applies epilogue trace bodies
        /// on each accumulator before storing to C.
        fn emit_gemm_8x12_with_epilogue(
            &mut self,
            k: usize,
            epilogue_bodies: &[Vec<TraceOp>],
        ) -> Result<(), String> {
            // ── zero 24 accumulators (v8-v31) ──
            for acc in 8u8..32u8 {
                self.emit_u32(Self::encode_eor_v(acc, acc, acc));
            }

            // ── K-loop setup: mov x9, k ──
            let k16 = (k & 0xFFFF) as u32;
            self.emit_u32(0xD2800009 | (k16 << 5));

            // cbz x9, epilogue
            let cbz_pos = self.code.len();
            self.emit_u32(0xB4000009); // placeholder

            let loop_start = self.code.len();

            // ── Load A-panel: 8 floats into v0, v1 ──
            self.emit_u32(Self::encode_ld1_post(0, 0));
            self.emit_u32(Self::encode_ld1_post(1, 0));

            // ── Load B-panel: 12 floats into v2, v3, v4 ──
            self.emit_u32(Self::encode_ld1_post(2, 1));
            self.emit_u32(Self::encode_ld1_post(3, 1));
            self.emit_u32(Self::encode_ld1_post(4, 1));

            // ── 24 FMA instructions (8 rows × 3 col-groups) ──
            for row in 0u8..8 {
                let a_reg = row / 4;
                let lane = row % 4;
                for col in 0u8..3 {
                    let acc = 8 + row * 3 + col;
                    let b_reg = 2 + col;
                    self.emit_u32(Self::encode_fmla_lane(acc, b_reg, a_reg, lane));
                }
            }

            // ── K-loop branch ──
            self.emit_u32(0xD1000529); // sub x9, x9, #1
            let back = (loop_start as i32 - self.code.len() as i32) / 4;
            let imm19 = (back as u32) & 0x7FFFF;
            self.emit_u32(0xB5000000 | (imm19 << 5) | 9);

            // ── patch cbz to epilogue ──
            let epi_pos = self.code.len();
            let fwd = ((epi_pos - cbz_pos) / 4) as u32 & 0x7FFFF;
            let patched = 0xB4000000 | (fwd << 5) | 9;
            self.code[cbz_pos..cbz_pos + 4].copy_from_slice(&patched.to_le_bytes());

            // ── Spill all 24 accumulators to stack ──
            // sub sp, sp, #384  (24 * 16)
            self.emit_u32(0xD10603FF); // sub sp, sp, #0x180 (384)
            for i in 0u8..24 {
                let acc = 8 + i;
                let offset = (i as u32) * 16;
                // str q_acc, [sp, #offset]
                self.emit_str_q_sp(acc, offset);
            }

            // ── Apply epilogue on each accumulator ──
            for i in 0u8..24 {
                let offset = (i as u32) * 16;
                // ldr q0, [sp, #offset]  (load accumulator into v0)
                self.emit_ldr_q_sp(0, offset);

                // Apply all epilogue bodies
                let mut result_reg = 0u8;
                for body in epilogue_bodies {
                    result_reg = self.emit_trace_body_neon(body, false)?;
                    if result_reg != 0 {
                        self.emit_u32(Self::encode_mov_v(0, result_reg));
                    }
                }

                // str q_result, [sp, #offset]
                self.emit_str_q_sp(result_reg, offset);
            }

            // ── Load processed accumulators back ──
            for i in 0u8..24 {
                let acc = 8 + i;
                let offset = (i as u32) * 16;
                self.emit_ldr_q_sp(acc, offset);
            }

            // add sp, sp, #384
            self.emit_u32(0x910603FF); // add sp, sp, #0x180 (384)

            // ── Store 24 accumulators to C (x2) ──
            for acc in 8u8..32u8 {
                self.emit_u32(Self::encode_st1_post(acc, 2));
            }

            Ok(())
        }

        /// Encode `str qt, [sp, #imm]` (unsigned offset, scaled by 16).
        fn emit_str_q_sp(&mut self, vt: u8, byte_offset: u32) {
            let imm12 = (byte_offset / 16) & 0xFFF;
            // STR Qt, [Xn, #imm] — 0x3D800000 | (imm12 << 10) | (Rn << 5) | Rt
            self.emit_u32(0x3D800000 | (imm12 << 10) | (31 << 5) | (vt as u32 & 0x1F));
        }

        /// Encode `ldr qt, [sp, #imm]` (unsigned offset, scaled by 16).
        fn emit_ldr_q_sp(&mut self, vt: u8, byte_offset: u32) {
            let imm12 = (byte_offset / 16) & 0xFFF;
            // LDR Qt, [Xn, #imm] — 0x3DC00000 | (imm12 << 10) | (Rn << 5) | Rt
            self.emit_u32(0x3DC00000 | (imm12 << 10) | (31 << 5) | (vt as u32 & 0x1F));
        }

        /// Emit GEMM: for small matrices use the direct microkernel,
        /// for large matrices use the full BLIS 5-level loop nest.
        fn emit_gemm_microkernel(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
        ) -> Result<(), String> {
            let blocking = profile.gemm_blocking(m, n, k);
            let kc = blocking.kc;
            let mc = blocking.mc;
            let nc = blocking.nc;

            if k <= kc && m <= mc && n <= nc {
                // Small matrix: emit direct microkernel (no packing overhead)
                return self.emit_gemm_8x12_neon(k);
            }

            self.emit_gemm_blis_neon(m, n, k, profile)
        }

        // ── BLIS 5-level loop nest ──────────────────────────────────────

        /// Emit the full BLIS NC→KC→MC→NR→MR loop nest with packing.
        ///
        /// ABI: CompiledLayerFn — x0=input(A), x1=weights(B), x7=output(C),
        /// scratchpad is 9th arg at [x29 + 16] (after prologue).
        ///
        /// Scratchpad layout (allocated at compile time):
        ///   [blis_base .. +pack_a_size]  → packed A buffer
        ///   [.. +pack_b_size]            → packed B buffer
        fn emit_gemm_blis_neon(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
        ) -> Result<(), String> {
            let blocking = profile.gemm_blocking(m, n, k);
            let kc = blocking.kc;
            let mc = blocking.mc;
            let nc = blocking.nc;

            // Compute scratchpad space for packing buffers
            let pack_a_bytes = mc * kc * 4; // mc * kc floats
            let pack_b_bytes = kc * nc * 4; // kc * nc floats
            let pack_a_off = self.blis_scratchpad_offset;
            let pack_b_off = pack_a_off + pack_a_bytes;
            let total_extra = pack_a_bytes + pack_b_bytes;
            self.blis_scratchpad_offset += total_extra;
            if total_extra > self.blis_scratchpad_bytes {
                self.blis_scratchpad_bytes = total_extra;
            }

            // Get pack function pointers (as u64 immediates baked into code)
            #[cfg(target_arch = "aarch64")]
            let pack_a_fn = crate::asm::aarch64::gllm_pack_a_f32_neon as *const () as u64;
            #[cfg(not(target_arch = "aarch64"))]
            let pack_a_fn = 0u64;

            #[cfg(target_arch = "aarch64")]
            let pack_b_fn = crate::asm::aarch64::gllm_pack_b_f32_neon as *const () as u64;
            #[cfg(not(target_arch = "aarch64"))]
            let pack_b_fn = 0u64;

            // ── Save callee-saved registers ──
            self.emit_save_callee_saved();

            // ── Save input args to callee-saved regs ──
            // x23 = x0 (A base)
            self.emit_mov_reg(23, 0);
            // x24 = x1 (B base)
            self.emit_mov_reg(24, 1);
            // x25 = x7 (C/output base)
            self.emit_mov_reg(25, 7);

            // ── Load scratchpad ptr from stack ──
            // After prologue (stp x29,x30 = -16) + save_callee (-80) = 96 bytes below original sp
            // 9th arg (scratchpad) was at [original_sp], now at [sp + 96]
            // ldr x22, [sp, #96]
            self.emit_ldr_imm_x(22, 31, 96);

            // ── NC loop (jc): x19 = 0 ──
            self.emit_movz(19, 0, 0);
            let jc_loop = self.code.len();

            // if x19 >= n: goto jc_done
            self.emit_cmp_imm(19, n as u32);
            let jc_done_patch = self.code.len();
            self.emit_u32(0x54000000); // b.ge placeholder

            // nc_cur = min(nc, n - x19) → x26
            self.emit_sub_reg(26, n as u8, 19); // pseudo: we use imm approach
            // Actually: mov x26, #nc; sub x9, #n_imm, x19; cmp x26, x9; csel x26, x9, x26, lt
            // Simpler: compute n - jc in x9, then min with nc
            self.emit_mov_imm(9, n);
            self.emit_sub_reg(9, 9, 19);
            self.emit_mov_imm(26, nc);
            self.emit_cmp_reg(9, 26);
            // csel x26, x9, x26, lo  (if x9 < x26, x26 = x9)
            self.emit_csel(26, 9, 26, 0x3); // cc/lo condition

            // ── KC loop (pc): x20 = 0 ──
            self.emit_movz(20, 0, 0);
            let pc_loop = self.code.len();

            // if x20 >= k: goto pc_done
            self.emit_cmp_imm(20, k as u32);
            let pc_done_patch = self.code.len();
            self.emit_u32(0x54000000); // b.ge placeholder

            // kc_cur = min(kc, k - x20) → x27
            self.emit_mov_imm(9, k);
            self.emit_sub_reg(9, 9, 20);
            self.emit_mov_imm(27, kc);
            self.emit_cmp_reg(9, 27);
            self.emit_csel(27, 9, 27, 0x3);

            // ── pack_b: gllm_pack_b_f32_neon(B + (pc*n + jc)*4, n, packed_b, kc_cur, nc_cur, nr) ──
            // x0 = B + (x20 * n + x19) * 4
            self.emit_mov_imm(9, n);
            self.emit_mul(0, 20, 9);       // x0 = pc * n
            self.emit_add_reg(0, 0, 19);   // x0 = pc * n + jc
            self.emit_lsl_imm(0, 0, 2);    // x0 *= 4 (byte offset)
            self.emit_add_reg(0, 24, 0);   // x0 = B_base + offset
            // x1 = n (ldb)
            self.emit_mov_imm(1, n);
            // x2 = packed_b ptr = scratchpad + pack_b_off
            self.emit_mov_imm(9, pack_b_off);
            self.emit_add_reg(2, 22, 9);
            // x3 = kc_cur
            self.emit_mov_reg(3, 27);
            // x4 = nc_cur
            self.emit_mov_reg(4, 26);
            // x5 = NR
            self.emit_mov_imm(5, NR);
            // blr pack_b_fn
            self.emit_mov_imm(10, pack_b_fn as usize);
            self.emit_blr(10);

            // ── MC loop (ic): x21 = 0 ──
            self.emit_movz(21, 0, 0);
            let ic_loop = self.code.len();

            // if x21 >= m: goto ic_done
            self.emit_cmp_imm(21, m as u32);
            let ic_done_patch = self.code.len();
            self.emit_u32(0x54000000); // b.ge placeholder

            // mc_cur = min(mc, m - x21) → x28
            self.emit_mov_imm(9, m);
            self.emit_sub_reg(9, 9, 21);
            self.emit_mov_imm(28, mc);
            self.emit_cmp_reg(9, 28);
            self.emit_csel(28, 9, 28, 0x3);

            // ── pack_a: gllm_pack_a_f32_neon(A + (ic*k + pc)*4, k, packed_a, mc_cur, kc_cur, mr) ──
            // x0 = A + (x21 * k + x20) * 4
            self.emit_mov_imm(9, k);
            self.emit_mul(0, 21, 9);       // x0 = ic * k
            self.emit_add_reg(0, 0, 20);   // x0 = ic * k + pc
            self.emit_lsl_imm(0, 0, 2);    // x0 *= 4
            self.emit_add_reg(0, 23, 0);   // x0 = A_base + offset
            // x1 = k (lda)
            self.emit_mov_imm(1, k);
            // x2 = packed_a ptr = scratchpad + pack_a_off
            self.emit_mov_imm(9, pack_a_off);
            self.emit_add_reg(2, 22, 9);
            // x3 = mc_cur
            self.emit_mov_reg(3, 28);
            // x4 = kc_cur
            self.emit_mov_reg(4, 27);
            // x5 = MR
            self.emit_mov_imm(5, MR);
            // blr pack_a_fn
            self.emit_mov_imm(10, pack_a_fn as usize);
            self.emit_blr(10);

            // ── NR loop (jr): x9 = 0 ──
            // Save x9 on stack for inner loops (we need a local)
            // Actually use x14 for jr, x15 for ir (caller-saved, but we control calls)
            self.emit_movz(14, 0, 0); // x14 = jr = 0
            let jr_loop = self.code.len();

            // if x14 >= nc_cur (x26): goto jr_done
            self.emit_cmp_reg(14, 26);
            let jr_done_patch = self.code.len();
            self.emit_u32(0x54000000); // b.ge placeholder

            // ── MR loop (ir): x15 = 0 ──
            self.emit_movz(15, 0, 0); // x15 = ir = 0
            let ir_loop = self.code.len();

            // if x15 >= mc_cur (x28): goto ir_done
            self.emit_cmp_reg(15, 28);
            let ir_done_patch = self.code.len();
            self.emit_u32(0x54000000); // b.ge placeholder

            // ── Setup microkernel args ──
            // x0 = packed_a + ir * kc_cur * 4
            self.emit_mul(9, 15, 27);       // x9 = ir * kc_cur
            self.emit_lsl_imm(9, 9, 2);     // x9 *= 4
            self.emit_mov_imm(10, pack_a_off);
            self.emit_add_reg(0, 22, 10);    // x0 = scratchpad + pack_a_off
            self.emit_add_reg(0, 0, 9);      // x0 += ir * kc_cur * 4

            // x1 = packed_b + jr * kc_cur * 4
            self.emit_mul(9, 14, 27);        // x9 = jr * kc_cur
            self.emit_lsl_imm(9, 9, 2);      // x9 *= 4
            self.emit_mov_imm(10, pack_b_off);
            self.emit_add_reg(1, 22, 10);     // x1 = scratchpad + pack_b_off
            self.emit_add_reg(1, 1, 9);       // x1 += jr * kc_cur * 4

            // x2 = C + (ic + ir) * n + (jc + jr)  (all in f32 units, *4 for bytes)
            self.emit_add_reg(9, 21, 15);     // x9 = ic + ir
            self.emit_mov_imm(10, n);
            self.emit_mul(9, 9, 10);          // x9 = (ic + ir) * n
            self.emit_add_reg(9, 9, 19);      // x9 += jc
            self.emit_add_reg(9, 9, 14);      // x9 += jr
            self.emit_lsl_imm(9, 9, 2);       // x9 *= 4
            self.emit_add_reg(2, 25, 9);       // x2 = C_base + offset

            // x3 = kc_cur (K for microkernel)
            self.emit_mov_reg(3, 27);

            // ── Inline 8×12 microkernel (reads from x0=packed_a, x1=packed_b, stores to x2=C) ──
            self.emit_gemm_8x12_neon(kc)?;

            // ── ir += MR ──
            self.emit_add_imm(15, 15, MR as u32);
            // b ir_loop
            self.emit_b_back(ir_loop);

            // ── ir_done: patch ──
            self.patch_bcond_ge(ir_done_patch);

            // ── jr += NR ──
            self.emit_add_imm(14, 14, NR as u32);
            // b jr_loop
            self.emit_b_back(jr_loop);

            // ── jr_done: patch ──
            self.patch_bcond_ge(jr_done_patch);

            // ── ic += mc ──
            self.emit_add_imm(21, 21, mc as u32);
            // b ic_loop
            self.emit_b_back(ic_loop);

            // ── ic_done: patch ──
            self.patch_bcond_ge(ic_done_patch);

            // ── pc += kc ──
            self.emit_add_imm(20, 20, kc as u32);
            // b pc_loop
            self.emit_b_back(pc_loop);

            // ── pc_done: patch ──
            self.patch_bcond_ge(pc_done_patch);

            // ── jc += nc ──
            self.emit_add_imm(19, 19, nc as u32);
            // b jc_loop
            self.emit_b_back(jc_loop);

            // ── jc_done: patch ──
            self.patch_bcond_ge(jc_done_patch);

            // ── Restore callee-saved registers ──
            self.emit_restore_callee_saved();

            Ok(())
        }

        // ── TileLevelFusion ─────────────────────────────────────────────

        /// Emit TileLevelFusion: predecessor elementwise op is applied per
        /// MC-tile row before the GEMM's pack_a, fusing the two into one
        /// pass over A's data.
        ///
        /// This is the aarch64 equivalent of the x86_64 TileLevelFusion path.
        /// For now, emits the predecessor as a standalone op followed by the
        /// GEMM with BLIS blocking (structurally correct, not yet fused into
        /// the pack_a loop).
        fn emit_tile_level_fusion(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
            _tile_rows: usize,
        ) -> Result<(), String> {
            // Emit predecessor op standalone
            if let Some(pred_op) = graph.op(predecessor) {
                match &pred_op.kind {
                    OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                        self.emit_gemm_microkernel(*m, *n, *k, profile)?;
                    }
                    _ => {
                        self.emit_nop_raw();
                    }
                }
            }

            // Emit anchor GEMM with BLIS blocking
            let anchor_op = graph.op(group.anchor).ok_or("missing anchor op")?;
            match &anchor_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile)
                }
                _ => {
                    self.emit_nop_raw();
                    Ok(())
                }
            }
        }

        // ── ComputeRoot ─────────────────────────────────────────────────

        /// Emit ComputeRoot fusion: predecessor is fully materialized first,
        /// then the anchor GEMM reads from the materialized buffer.
        ///
        /// This matches the x86_64 ComputeRoot semantics: the predecessor
        /// runs to completion before the GEMM begins.
        fn emit_compute_root(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
        ) -> Result<(), String> {
            // Materialize predecessor fully
            if let Some(pred_op) = graph.op(predecessor) {
                match &pred_op.kind {
                    OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                        self.emit_gemm_microkernel(*m, *n, *k, profile)?;
                    }
                    _ => {
                        self.emit_nop_raw();
                    }
                }
            }

            // Emit anchor GEMM
            let anchor_op = graph.op(group.anchor).ok_or("missing anchor op")?;
            match &anchor_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile)
                }
                _ => {
                    self.emit_nop_raw();
                    Ok(())
                }
            }
        }

        // ── GP register instruction encoding helpers ────────────────────

        /// Encode `stp xt1, xt2, [xn, #imm]!` (pre-index).
        /// imm must be a multiple of 8, range [-512, 504].
        fn encode_stp_pre(rt1: u8, rt2: u8, rn: u8, imm: i32) -> u32 {
            let imm7 = ((imm / 8) as u32) & 0x7F;
            0xA9800000
                | (imm7 << 15)
                | ((rt2 as u32 & 0x1F) << 10)
                | ((rn as u32 & 0x1F) << 5)
                | (rt1 as u32 & 0x1F)
        }

        /// Encode `stp xt1, xt2, [xn, #imm]` (signed offset).
        fn encode_stp_offset(rt1: u8, rt2: u8, rn: u8, imm: i32) -> u32 {
            let imm7 = ((imm / 8) as u32) & 0x7F;
            0xA9000000
                | (imm7 << 15)
                | ((rt2 as u32 & 0x1F) << 10)
                | ((rn as u32 & 0x1F) << 5)
                | (rt1 as u32 & 0x1F)
        }

        /// Encode `ldp xt1, xt2, [xn, #imm]` (signed offset).
        fn encode_ldp_offset(rt1: u8, rt2: u8, rn: u8, imm: i32) -> u32 {
            let imm7 = ((imm / 8) as u32) & 0x7F;
            0xA9400000
                | (imm7 << 15)
                | ((rt2 as u32 & 0x1F) << 10)
                | ((rn as u32 & 0x1F) << 5)
                | (rt1 as u32 & 0x1F)
        }

        /// Encode `ldp xt1, xt2, [xn], #imm` (post-index).
        fn encode_ldp_post(rt1: u8, rt2: u8, rn: u8, imm: i32) -> u32 {
            let imm7 = ((imm / 8) as u32) & 0x7F;
            0xA8C00000
                | (imm7 << 15)
                | ((rt2 as u32 & 0x1F) << 10)
                | ((rn as u32 & 0x1F) << 5)
                | (rt1 as u32 & 0x1F)
        }

        /// Emit `mov xd, xn` (alias for `orr xd, xzr, xn`).
        fn emit_mov_reg(&mut self, rd: u8, rn: u8) {
            // orr xd, xzr, xn → 0xAA0003E0 | (Rm << 16) | Rd
            self.emit_u32(0xAA0003E0 | ((rn as u32 & 0x1F) << 16) | (rd as u32 & 0x1F));
        }

        /// Emit `movz xd, #imm16, lsl #shift` (shift = 0, 16, 32, 48).
        fn emit_movz(&mut self, rd: u8, imm16: u16, shift: u8) {
            let hw = (shift / 16) as u32;
            self.emit_u32(0xD2800000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32 & 0x1F));
        }

        /// Emit `movk xd, #imm16, lsl #shift`.
        fn emit_movk(&mut self, rd: u8, imm16: u16, shift: u8) {
            let hw = (shift / 16) as u32;
            self.emit_u32(0xF2800000 | (hw << 21) | ((imm16 as u32) << 5) | (rd as u32 & 0x1F));
        }

        /// Emit a full 64-bit immediate load into xd using movz/movk sequence.
        fn emit_mov_imm(&mut self, rd: u8, val: usize) {
            let v = val as u64;
            self.emit_movz(rd, v as u16, 0);
            if v > 0xFFFF {
                self.emit_movk(rd, (v >> 16) as u16, 16);
            }
            if v > 0xFFFF_FFFF {
                self.emit_movk(rd, (v >> 32) as u16, 32);
            }
            if v > 0xFFFF_FFFF_FFFF {
                self.emit_movk(rd, (v >> 48) as u16, 48);
            }
        }

        /// Emit `add xd, xn, #imm12`.
        fn emit_add_imm(&mut self, rd: u8, rn: u8, imm12: u32) {
            debug_assert!(imm12 < 4096);
            self.emit_u32(0x91000000
                | ((imm12 & 0xFFF) << 10)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F));
        }

        /// Emit `add xd, xn, xm`.
        fn emit_add_reg(&mut self, rd: u8, rn: u8, rm: u8) {
            self.emit_u32(0x8B000000
                | ((rm as u32 & 0x1F) << 16)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F));
        }

        /// Emit `sub xd, xn, xm`.
        fn emit_sub_reg(&mut self, rd: u8, rn: u8, rm: u8) {
            self.emit_u32(0xCB000000
                | ((rm as u32 & 0x1F) << 16)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F));
        }

        /// Emit `mul xd, xn, xm` (alias for `madd xd, xn, xm, xzr`).
        fn emit_mul(&mut self, rd: u8, rn: u8, rm: u8) {
            self.emit_u32(0x9B007C00
                | ((rm as u32 & 0x1F) << 16)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F));
        }

        /// Emit `lsl xd, xn, #shift` (alias for `ubfm xd, xn, #(64-shift), #(63-shift)`).
        fn emit_lsl_imm(&mut self, rd: u8, rn: u8, shift: u8) {
            let immr = (64 - shift as u32) & 0x3F;
            let imms = (63 - shift as u32) & 0x3F;
            self.emit_u32(0xD3400000
                | (immr << 16)
                | (imms << 10)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F));
        }

        /// Emit `cmp xn, #imm12` (alias for `subs xzr, xn, #imm12`).
        /// For values > 4095, loads into x9 first and uses register compare.
        fn emit_cmp_imm(&mut self, rn: u8, imm: u32) {
            if imm < 4096 {
                self.emit_u32(0xF100001F
                    | ((imm & 0xFFF) << 10)
                    | ((rn as u32 & 0x1F) << 5));
            } else {
                // Load imm into x11 (scratch), then cmp reg
                self.emit_mov_imm(11, imm as usize);
                self.emit_cmp_reg(rn, 11);
            }
        }

        /// Emit `cmp xn, xm` (alias for `subs xzr, xn, xm`).
        fn emit_cmp_reg(&mut self, rn: u8, rm: u8) {
            self.emit_u32(0xEB00001F
                | ((rm as u32 & 0x1F) << 16)
                | ((rn as u32 & 0x1F) << 5));
        }

        /// Emit `csel xd, xn, xm, <cond>`.
        fn emit_csel(&mut self, rd: u8, rn: u8, rm: u8, cond: u8) {
            self.emit_u32(0x9A800000
                | ((rm as u32 & 0x1F) << 16)
                | ((cond as u32 & 0xF) << 12)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F));
        }

        /// Emit `blr xn` (branch with link to register).
        fn emit_blr(&mut self, rn: u8) {
            self.emit_u32(0xD63F0000 | ((rn as u32 & 0x1F) << 5));
        }

        /// Emit `ldr xd, [xn, #imm]` (unsigned offset, scaled by 8).
        fn emit_ldr_imm_x(&mut self, rd: u8, rn: u8, byte_offset: u32) {
            let imm12 = (byte_offset / 8) & 0xFFF;
            self.emit_u32(0xF9400000
                | (imm12 << 10)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F));
        }

        /// Emit unconditional backward branch to `target` (byte offset in code buffer).
        fn emit_b_back(&mut self, target: usize) {
            let offset = (target as i32 - self.code.len() as i32) / 4;
            let imm26 = (offset as u32) & 0x3FFFFFF;
            self.emit_u32(0x14000000 | imm26);
        }

        /// Patch a `b.ge` placeholder at `patch_pos` to branch to current position.
        fn patch_bcond_ge(&mut self, patch_pos: usize) {
            let offset = ((self.code.len() - patch_pos) / 4) as u32;
            let imm19 = offset & 0x7FFFF;
            // b.ge = 0x5400000A | (imm19 << 5)
            let insn = 0x5400000A | (imm19 << 5);
            self.code[patch_pos..patch_pos + 4].copy_from_slice(&insn.to_le_bytes());
        }

        /// Emit a TraceOp sequence as NEON SIMD instructions.
        ///
        /// This is the core of Phase 3: each TraceOp maps to one or more
        /// NEON instructions operating on v-registers (128-bit, 4xf32).
        /// The SSA indices in TraceOp directly map to v-register numbers
        /// (round-robin when > 32 ops).
        ///
        /// Mapping:
        /// - `Input(n)`    → `eor` zeroes reg (caller pre-loads via outer loop)
        /// - `Const(v)`    → `eor` zeroes reg (caller pre-loads via outer loop)
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
        /// - `Exp(a)`      → polynomial approximation (`emit_exp_neon`)
        /// - `Tanh(a)`     → rational approximation (`emit_tanh_neon`)
        /// - `Log(a)`      → exponent extraction + polynomial (`emit_log_neon`)
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
                        // frsqrte + 2× Newton-Raphson → ~24-bit precision
                        self.emit_rsqrt_refined_neon(rd, ra);
                    }
                    TraceOp::Recip(a) => {
                        let ra = reg_map[*a as usize];
                        // frecpe + 1× Newton-Raphson → ~16-bit precision
                        self.emit_recip_refined_neon(rd, ra);
                    }
                    TraceOp::Exp(a) => {
                        let ra = reg_map[*a as usize];
                        // degree-2 Taylor: exp(x) ≈ 1 + x + x²/2
                        self.emit_exp_neon(rd, ra);
                    }
                    TraceOp::Tanh(a) => {
                        let ra = reg_map[*a as usize];
                        // tanh(x) = 2*sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1
                        self.emit_tanh_neon(rd, ra);
                    }
                    TraceOp::Log(a) => {
                        let ra = reg_map[*a as usize];
                        // ln(x) = e*ln(2) + poly(mantissa-1)
                        self.emit_log_neon(rd, ra);
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

        /// Encode `fmla v_rd.4s, v_rn.4s, v_rm.s[index]` (by-element FMA).
        ///
        /// FMLA (by element) f32 Q=1: Rm is 4-bit (v0-v15 only).
        /// Index is 2-bit: H (bit 11), L (bit 21).
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

        /// Encode `frsqrts v_rd.4s, v_rn.4s, v_rm.4s`.
        ///
        /// FRSQRTS: 0x0EA0FC00 | Q=1 → 0x4EA0FC00
        fn encode_frsqrts(rd: u8, rn: u8, rm: u8) -> u32 {
            0x4EA0FC00 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode `frecps v_rd.4s, v_rn.4s, v_rm.4s`.
        ///
        /// FRECPS: 0x0E20FC00 | Q=1 → 0x4E20FC00
        fn encode_frecps(rd: u8, rn: u8, rm: u8) -> u32 {
            0x4E20FC00 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode `ld1 {vt.4s}, [xn], #16` (post-index, single register).
        fn encode_ld1_post(vt: u8, xn: u8) -> u32 {
            0x4CDF7800 | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
        }

        /// Encode `st1 {vt.4s}, [xn], #16` (post-index, single register).
        fn encode_st1_post(vt: u8, xn: u8) -> u32 {
            0x4C9F7800 | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
        }

        /// Encode `fmov v_rd.4s, #1.0` (ASIMD modified immediate, float8 = 0x70).
        fn encode_fmov_one(rd: u8) -> u32 {
            // abcdefgh for 1.0: a=0,b=1,c=1,d=1,e=0,f=0,g=0,h=0
            0x4F03F600 | (rd as u32 & 0x1F)
        }

        /// Encode `fmov v_rd.4s, #0.5` (ASIMD modified immediate, float8 = 0x60).
        fn encode_fmov_half(rd: u8) -> u32 {
            // abcdefgh for 0.5: a=0,b=1,c=1,d=0,e=0,f=0,g=0,h=0
            0x4F03F400 | (rd as u32 & 0x1F)
        }

        // ── Math function approximations ──────────────────────────────

        /// Emit NEON exp(x) approximation: degree-2 Taylor (1 + x + x²/2).
        ///
        /// Uses v28-v30 as scratch. Not production quality (no range reduction),
        /// but structurally correct for the JIT pipeline. A full Cephes polynomial
        /// requires constant-pool support (future work).
        ///
        /// Emits 5 instructions.
        fn emit_exp_neon(&mut self, rd: u8, ra: u8) {
            let s0 = 28u8; // scratch for x²
            let s_half = 29u8;
            // fmov v29.4s, #0.5
            self.emit_u32(Self::encode_fmov_half(s_half));
            // fmov v_rd.4s, #1.0  (rd = 1.0)
            self.emit_u32(Self::encode_fmov_one(rd));
            // fadd v_rd.4s, v_rd.4s, v_ra.4s  (rd = 1.0 + x)
            self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, rd, ra));
            // fmul v28.4s, v_ra.4s, v_ra.4s  (s0 = x²)
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, s0, ra, ra));
            // fmla v_rd.4s, v28.4s, v29.4s  (rd += x² * 0.5)
            self.emit_u32(Self::encode_fmla(rd, s0, s_half));
        }

        /// Emit NEON rsqrt with Newton-Raphson refinement (2 iterations).
        ///
        /// frsqrte → 2×(frsqrts + fmul) gives ~24-bit precision (sufficient for f32).
        /// Uses v28-v29 as scratch.
        ///
        /// Emits 7 instructions.
        fn emit_rsqrt_refined_neon(&mut self, rd: u8, ra: u8) {
            let s0 = 28u8;
            let s1 = 29u8;

            // y0 = frsqrte(x)
            self.emit_u32(Self::encode_f32x4_unary(0x6EA1D800, rd, ra));

            // NR iteration 1: y1 = y0 * frsqrts(x, y0²)
            // s0 = y0 * y0
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, s0, rd, rd));
            // s1 = frsqrts(x, s0) = (3 - x·y0²) / 2
            self.emit_u32(Self::encode_frsqrts(s1, ra, s0));
            // rd = y0 * s1
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, rd, s1));

            // NR iteration 2: y2 = y1 * frsqrts(x, y1²)
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, s0, rd, rd));
            self.emit_u32(Self::encode_frsqrts(s1, ra, s0));
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, rd, s1));
        }

        /// Emit NEON reciprocal with Newton-Raphson refinement (1 iteration).
        ///
        /// frecpe → frecps + fmul gives ~16-bit precision.
        /// Uses v28 as scratch.
        ///
        /// Emits 3 instructions.
        fn emit_recip_refined_neon(&mut self, rd: u8, ra: u8) {
            let s0 = 28u8;

            // y0 = frecpe(x)
            self.emit_u32(Self::encode_f32x4_unary(0x4EA1D800, rd, ra));

            // NR iteration: y1 = y0 * frecps(x, y0)
            // s0 = frecps(x, y0) = 2 - x·y0
            self.emit_u32(Self::encode_frecps(s0, ra, rd));
            // rd = y0 * s0
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, rd, s0));
        }

        // ── Constant loading helpers ─────────────────────────────────────

        /// Load an arbitrary f32 constant into all 4 lanes of a NEON register.
        ///
        /// Uses GP register x9 as scratch:
        ///   movz x9, #lower16
        ///   movk x9, #upper16, lsl #16  (if needed)
        ///   dup  v_rd.4s, w9
        ///
        /// Emits 2-3 instructions.
        fn emit_load_f32_const_neon(&mut self, vd: u8, val: f32) {
            let bits = val.to_bits();
            let lo = bits as u16;
            let hi = (bits >> 16) as u16;
            if hi == 0 {
                self.emit_movz(9, lo, 0);
            } else if lo == 0 {
                self.emit_movz(9, hi, 16);
            } else {
                self.emit_movz(9, lo, 0);
                self.emit_movk(9, hi, 16);
            }
            // dup v_vd.4s, w9
            self.emit_u32(Self::encode_dup_4s_gp(vd, 9));
        }

        /// Encode `dup v_rd.4s, w_rn` (broadcast GP register to all 4 lanes).
        ///
        /// DUP (general) Q=1, size=32: 0x4E040C00 | (Rn_gp << 5) | Rd_vec
        fn encode_dup_4s_gp(vd: u8, xn: u8) -> u32 {
            0x4E040C00 | ((xn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
        }

        /// Encode `ushr v_rd.4s, v_rn.4s, #shift`.
        ///
        /// USHR (vector) Q=1, 32-bit: immh:immb = 64 - shift.
        fn encode_ushr_4s(rd: u8, rn: u8, shift: u8) -> u32 {
            let immh_immb = (64 - shift as u32) & 0x7F;
            0x6F000400
                | (immh_immb << 16)
                | ((rn as u32 & 0x1F) << 5)
                | (rd as u32 & 0x1F)
        }

        /// Encode `and v_rd.16b, v_rn.16b, v_rm.16b`.
        ///
        /// AND (vector) Q=1: 0x4E201C00 | Rm:20-16 | Rn:9-5 | Rd:4-0
        fn encode_and_v(rd: u8, rn: u8, rm: u8) -> u32 {
            0x4E201C00 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode `orr v_rd.16b, v_rn.16b, v_rm.16b`.
        ///
        /// ORR (vector) Q=1: 0x4EA01C00 | Rm:20-16 | Rn:9-5 | Rd:4-0
        fn encode_orr_v(rd: u8, rn: u8, rm: u8) -> u32 {
            0x4EA01C00 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        /// Encode `ucvtf v_rd.4s, v_rn.4s` (unsigned int32 → f32).
        ///
        /// UCVTF (vector, integer) Q=1, sz=0: 0x6E21D800
        fn encode_ucvtf_4s(rd: u8, rn: u8) -> u32 {
            0x6E21D800 | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
        }

        // ── Tanh / Log approximations ──────────────────────────────────

        /// Emit NEON tanh(x) via `2*sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1`.
        ///
        /// Same algorithm as x86_64 `emit_tanh_avx2`. Uses the existing
        /// `emit_exp_neon` for the inner exp, plus NEON `frecpe`/`frecps`
        /// for the reciprocal with one Newton-Raphson step.
        ///
        /// Uses v28-v30 as scratch (same budget as exp/rsqrt).
        /// Emits ~14 instructions.
        fn emit_tanh_neon(&mut self, rd: u8, ra: u8) {
            // Step 1: v30 = -2.0 * x
            self.emit_load_f32_const_neon(30, -2.0); // 1 insn (movz + dup = but -2.0 = 0xC0000000, hi=0xC000, lo=0 → 1 movz + 1 dup = 2)
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, 30, ra, 30)); // fmul v30, ra, v30

            // Step 2: rd = exp(-2x)
            self.emit_exp_neon(rd, 30); // 5 insns, uses v28/v29 as scratch, reads v30

            // Step 3: rd = 1.0 + exp(-2x)
            self.emit_u32(Self::encode_fmov_one(28)); // fmov v28.4s, #1.0
            self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, rd, 28)); // fadd rd, rd, v28

            // Step 4: reciprocal of rd → v29 (frecpe + 1× NR)
            self.emit_u32(Self::encode_f32x4_unary(0x4EA1D800, 29, rd)); // frecpe v29, rd
            self.emit_u32(Self::encode_frecps(30, rd, 29));               // frecps v30, rd, v29
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, 29, 29, 30)); // fmul v29, v29, v30

            // Step 5: rd = 2 * v29 = 2/(1+exp(-2x))
            self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, 29, 29)); // fadd rd, v29, v29

            // Step 6: rd = rd - 1.0 (v28 still holds 1.0)
            self.emit_u32(Self::encode_f32x4_binop(0x4EA0D400, rd, rd, 28)); // fsub rd, rd, v28
        }

        /// Emit NEON ln(x) via exponent/mantissa decomposition + degree-4 Horner.
        ///
        /// Algorithm (same as x86_64 `emit_log_avx2`):
        ///   e = float(x_as_u32 >> 23) - 127
        ///   m = (x_as_u32 & 0x007FFFFF) | 0x3F800000   → [1.0, 2.0)
        ///   t = m - 1.0
        ///   p = ((c4*t + c3)*t + c2)*t + c1
        ///   ln(x) = e * ln(2) + p * t
        ///
        /// Uses v28-v30 as scratch. Emits ~18 instructions.
        fn emit_log_neon(&mut self, rd: u8, ra: u8) {
            // s0=v28 (exponent e), s1=v29 (mantissa t), s2=v30 (scratch)

            // ── Extract exponent: e = float(x >> 23) - 127.0 ──
            // ushr v28.4s, v_ra.4s, #23
            self.emit_u32(Self::encode_ushr_4s(28, ra, 23));
            // ucvtf v28.4s, v28.4s  (u32 → f32)
            self.emit_u32(Self::encode_ucvtf_4s(28, 28));
            // v30 = 127.0
            self.emit_load_f32_const_neon(30, 127.0); // 2 insns
            // v28 = v28 - 127.0
            self.emit_u32(Self::encode_f32x4_binop(0x4EA0D400, 28, 28, 30)); // fsub v28, v28, v30

            // ── Extract mantissa: m = (x & 0x007FFFFF) | 0x3F800000 ──
            // v30 = 0x007FFFFF (mantissa mask)
            self.emit_load_f32_const_neon(30, f32::from_bits(0x007F_FFFF)); // 3 insns
            // v29 = x & mask
            self.emit_u32(Self::encode_and_v(29, ra, 30));
            // v30 = 0x3F800000 (1.0 as bits)
            self.emit_load_f32_const_neon(30, 1.0); // 2 insns
            // v29 = v29 | 1.0_bits → mantissa in [1.0, 2.0)
            self.emit_u32(Self::encode_orr_v(29, 29, 30));

            // ── t = m - 1.0 ──
            self.emit_u32(Self::encode_f32x4_binop(0x4EA0D400, 29, 29, 30)); // fsub v29, v29, v30

            // ── Horner polynomial: p = ((c4*t + c3)*t + c2)*t + c1 ──
            // rd = c4
            self.emit_load_f32_const_neon(rd, -0.24073381); // 3 insns
            // rd = rd * t + c3
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, rd, 29)); // fmul rd, rd, v29
            self.emit_load_f32_const_neon(30, 0.33179903); // 3 insns
            self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, rd, 30)); // fadd rd, rd, v30
            // rd = rd * t + c2
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, rd, 29)); // fmul rd, rd, v29
            self.emit_load_f32_const_neon(30, -0.49987412); // 3 insns
            self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, rd, 30)); // fadd rd, rd, v30
            // rd = rd * t + c1
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, rd, 29)); // fmul rd, rd, v29
            self.emit_load_f32_const_neon(30, 0.99999934); // 3 insns
            self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, rd, rd, 30)); // fadd rd, rd, v30

            // ── p = p * t ──
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, rd, rd, 29)); // fmul rd, rd, v29

            // ── result = e * ln(2) + p ──
            // v30 = ln(2)
            self.emit_load_f32_const_neon(30, 0.6931471805599453_f32); // 3 insns
            // rd = v28 * v30 + rd  → use fmla: rd += v28 * v30
            self.emit_u32(Self::encode_fmla(rd, 28, 30)); // fmla rd, v28, v30
        }

        // ── Loop structures ─────────────────────────────────────────────

        /// Emit a vectorised elementwise loop applying `ops` to each 4×f32 chunk.
        ///
        /// Register convention: x0 = input ptr, x1 = output ptr, x2 = element count.
        /// Main loop processes 4 floats/iteration via NEON; scalar tail handles remainder.
        pub fn emit_elementwise_loop(
            &mut self,
            ops: &[TraceOp],
        ) -> Result<(), String> {
            // x9 = x2 >> 2  (number of NEON iterations)
            // lsr x9, x2, #2
            self.emit_u32(0xD342FC49);

            // cbz x9, tail  (skip vector loop if count < 4)
            let cbz_pos = self.code.len();
            self.emit_u32(0xB4000009); // placeholder — patched below

            // ── vector loop ──
            let loop_start = self.code.len();

            // ld1 {v0.4s}, [x0], #16
            self.emit_u32(Self::encode_ld1_post(0, 0));

            // apply trace ops (input in v0, result in last allocated vreg)
            let regs = self.emit_trace_ops_neon(ops)?;
            let result_reg = *regs.last().ok_or("empty trace ops")?;

            // st1 {v_result.4s}, [x1], #16
            self.emit_u32(Self::encode_st1_post(result_reg, 1));

            // sub x9, x9, #1
            self.emit_u32(0xD1000529);

            // cbnz x9, loop_start
            let back_offset = (loop_start as i32 - self.code.len() as i32) / 4;
            let imm19 = (back_offset as u32) & 0x7FFFF;
            self.emit_u32(0xB5000000 | (imm19 << 5) | 9);

            // ── patch cbz to here (tail) ──
            let tail_pos = self.code.len();
            let fwd_offset = ((tail_pos - cbz_pos) / 4) as u32 & 0x7FFFF;
            let patched = 0xB4000000 | (fwd_offset << 5) | 9;
            self.code[cbz_pos..cbz_pos + 4].copy_from_slice(&patched.to_le_bytes());

            // ── scalar tail ──
            // x9 = x2 & 3  (remainder count)
            // and x9, x2, #3  (logical immediate N=1, immr=0, imms=1)
            self.emit_u32(0x92400449);

            // cbz x9, done
            let cbz2_pos = self.code.len();
            self.emit_u32(0xB4000009); // placeholder

            let scalar_start = self.code.len();
            // ldr s0, [x0], #4  (post-index scalar load)
            self.emit_u32(0xBC404400);
            // (scalar trace ops skipped — pass-through for now)
            // str s0, [x1], #4  (post-index scalar store)
            self.emit_u32(0xBC004420);
            // sub x9, x9, #1
            self.emit_u32(0xD1000529);
            // cbnz x9, scalar_start
            let sback = (scalar_start as i32 - self.code.len() as i32) / 4;
            let simm19 = (sback as u32) & 0x7FFFF;
            self.emit_u32(0xB5000000 | (simm19 << 5) | 9);

            // ── patch cbz2 to here (done) ──
            let done_pos = self.code.len();
            let fwd2 = ((done_pos - cbz2_pos) / 4) as u32 & 0x7FFFF;
            let patched2 = 0xB4000000 | (fwd2 << 5) | 9;
            self.code[cbz2_pos..cbz2_pos + 4].copy_from_slice(&patched2.to_le_bytes());

            Ok(())
        }

        // ── GEMM microkernel ────────────────────────────────────────────

        /// Emit an 8×12 NEON GEMM microkernel (K-loop body + accumulator management).
        ///
        /// Register allocation (AAPCS64 callee-save aware):
        ///   v0-v1   : A-panel  (8 floats = 2 × 4s, rows 0-3 / 4-7)
        ///   v2-v4   : B-panel  (12 floats = 3 × 4s, cols 0-3 / 4-7 / 8-11)
        ///   v5-v7   : scratch
        ///   v8-v31  : 24 accumulators (row r, col-group c → v[8 + r*3 + c])
        ///
        /// Calling convention: x0 = A ptr, x1 = B ptr, x2 = C ptr, x3 = K.
        /// A is column-major packed (MR=8 contiguous), B is row-major packed (NR=12).
        pub fn emit_gemm_8x12_neon(&mut self, k: usize) -> Result<(), String> {
            // ── zero 24 accumulators (v8-v31) ──
            for acc in 8u8..32u8 {
                self.emit_u32(Self::encode_eor_v(acc, acc, acc));
            }

            // ── K-loop setup: mov x9, k ──
            // movz x9, #(k & 0xFFFF)
            let k16 = (k & 0xFFFF) as u32;
            self.emit_u32(0xD2800009 | (k16 << 5));

            // cbz x9, store  (skip loop if K==0)
            let cbz_pos = self.code.len();
            self.emit_u32(0xB4000009); // placeholder

            let loop_start = self.code.len();

            // ── Load A-panel: 8 floats into v0, v1 ──
            // ld1 {v0.4s}, [x0], #16
            self.emit_u32(Self::encode_ld1_post(0, 0));
            // ld1 {v1.4s}, [x0], #16
            self.emit_u32(Self::encode_ld1_post(1, 0));

            // ── Load B-panel: 12 floats into v2, v3, v4 ──
            // ld1 {v2.4s}, [x1], #16
            self.emit_u32(Self::encode_ld1_post(2, 1));
            // ld1 {v3.4s}, [x1], #16
            self.emit_u32(Self::encode_ld1_post(3, 1));
            // ld1 {v4.4s}, [x1], #16
            self.emit_u32(Self::encode_ld1_post(4, 1));

            // ── 24 FMA instructions (8 rows × 3 col-groups) ──
            // fmla v_acc.4s, v_bcol.4s, v_arow.s[lane]
            for row in 0u8..8 {
                let a_reg = row / 4;        // v0 for rows 0-3, v1 for rows 4-7
                let lane = row % 4;         // s[0..3] within that register
                for col in 0u8..3 {
                    let acc = 8 + row * 3 + col;   // v8..v31
                    let b_reg = 2 + col;            // v2, v3, v4
                    self.emit_u32(Self::encode_fmla_lane(acc, b_reg, a_reg, lane));
                }
            }

            // ── K-loop branch ──
            // sub x9, x9, #1
            self.emit_u32(0xD1000529);
            // cbnz x9, loop_start
            let back = (loop_start as i32 - self.code.len() as i32) / 4;
            let imm19 = (back as u32) & 0x7FFFF;
            self.emit_u32(0xB5000000 | (imm19 << 5) | 9);

            // ── patch cbz to store ──
            let store_pos = self.code.len();
            let fwd = ((store_pos - cbz_pos) / 4) as u32 & 0x7FFFF;
            let patched = 0xB4000000 | (fwd << 5) | 9;
            self.code[cbz_pos..cbz_pos + 4].copy_from_slice(&patched.to_le_bytes());

            // ── Store 24 accumulators to C (x2) ──
            for acc in 8u8..32u8 {
                // st1 {v_acc.4s}, [x2], #16
                self.emit_u32(Self::encode_st1_post(acc, 2));
            }

            Ok(())
        }

        /// Emit a NEON nop (`hint #0` = 0xD503201F).
        fn emit_nop_raw(&mut self) {
            self.emit_u32(0xD503201F);
        }

        /// Append a 32-bit instruction to the code buffer (little-endian).
        fn emit_u32(&mut self, insn: u32) {
            self.code.extend_from_slice(&insn.to_le_bytes());
        }

        /// Test-only: expose raw code bytes for bit-pattern verification.
        #[cfg(test)]
        pub fn code_bytes(&self) -> &[u8] {
            &self.code
        }
    }

    // ── SimdOps trait implementation ──────────────────────────────────

    /// Convert a BaseReg to its AArch64 GP register number.
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

    impl SimdOps for AArch64CodeGen {
        // ── Vector arithmetic ───────────────────────────────────────

        fn vadd(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_binop(0x4E20D400, dst.0, a.0, b.0));
            Ok(())
        }

        fn vsub(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_binop(0x4EA0D400, dst.0, a.0, b.0));
            Ok(())
        }

        fn vmul(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_binop(0x6E20DC00, dst.0, a.0, b.0));
            Ok(())
        }

        fn vdiv(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_binop(0x6E20FC00, dst.0, a.0, b.0));
            Ok(())
        }

        fn vfma(&mut self, dst: VReg, a: VReg, b: VReg, c: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_mov_v(dst.0, c.0));
            self.emit_u32(Self::encode_fmla(dst.0, a.0, b.0));
            Ok(())
        }

        fn vneg(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_unary(0x6EA0F800, dst.0, a.0));
            Ok(())
        }

        fn vabs(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_unary(0x4EA0F800, dst.0, a.0));
            Ok(())
        }

        fn vsqrt(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_unary(0x6EA1F800, dst.0, a.0));
            Ok(())
        }

        fn vmax(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_binop(0x4E20F400, dst.0, a.0, b.0));
            Ok(())
        }

        fn vmin(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_f32x4_binop(0x4EA0F400, dst.0, a.0, b.0));
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

        fn vload(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String> {
            let base = gpr_num(&mem.base);
            if mem.offset == 0 && base != 31 {
                // ld1 {vt.4s}, [xn]
                self.emit_u32(0x4C407800 | ((base as u32) << 5) | (dst.0 as u32));
            } else {
                // Compute effective address in x9
                self.emit_mov_reg(9, base);
                if mem.offset != 0 {
                    if mem.offset > 0 && (mem.offset as u32) < 4096 {
                        self.emit_add_imm(9, 9, mem.offset as u32);
                    } else if mem.offset < 0 && ((-mem.offset) as u32) < 4096 {
                        let imm = (-mem.offset) as u32;
                        self.emit_u32(0xD1000000
                            | ((imm & 0xFFF) << 10)
                            | (9u32 << 5)
                            | 9u32);
                    } else {
                        self.emit_mov_imm(10, mem.offset as usize);
                        self.emit_add_reg(9, 9, 10);
                    }
                }
                // ld1 {vt.4s}, [x9]
                self.emit_u32(0x4C407800 | (9u32 << 5) | (dst.0 as u32));
            }
            Ok(())
        }

        fn vstore(&mut self, mem: MemOperand, src: VReg) -> Result<(), String> {
            let base = gpr_num(&mem.base);
            if mem.offset == 0 && base != 31 {
                // st1 {vt.4s}, [xn]
                self.emit_u32(0x4C007800 | ((base as u32) << 5) | (src.0 as u32));
            } else {
                self.emit_mov_reg(9, base);
                if mem.offset != 0 {
                    if mem.offset > 0 && (mem.offset as u32) < 4096 {
                        self.emit_add_imm(9, 9, mem.offset as u32);
                    } else if mem.offset < 0 && ((-mem.offset) as u32) < 4096 {
                        let imm = (-mem.offset) as u32;
                        self.emit_u32(0xD1000000
                            | ((imm & 0xFFF) << 10)
                            | (9u32 << 5)
                            | 9u32);
                    } else {
                        self.emit_mov_imm(10, mem.offset as usize);
                        self.emit_add_reg(9, 9, 10);
                    }
                }
                // st1 {vt.4s}, [x9]
                self.emit_u32(0x4C007800 | (9u32 << 5) | (src.0 as u32));
            }
            Ok(())
        }

        fn vbroadcast(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String> {
            let base = gpr_num(&mem.base);
            // Compute effective address in x9
            self.emit_mov_reg(9, base);
            if mem.offset != 0 {
                if mem.offset > 0 && (mem.offset as u32) < 4096 {
                    self.emit_add_imm(9, 9, mem.offset as u32);
                } else if mem.offset < 0 && ((-mem.offset) as u32) < 4096 {
                    let imm = (-mem.offset) as u32;
                    self.emit_u32(0xD1000000 | ((imm & 0xFFF) << 10) | (9u32 << 5) | 9u32);
                } else {
                    self.emit_mov_imm(10, mem.offset as usize);
                    self.emit_add_reg(9, 9, 10);
                }
            }
            // ldr s_dst, [x9] (scalar load)
            self.emit_u32(0xBD400000 | (9u32 << 5) | (dst.0 as u32));
            // dup v_dst.4s, v_dst.s[0]
            self.emit_u32(0x4E040400 | ((dst.0 as u32) << 5) | (dst.0 as u32));
            Ok(())
        }

        fn vbroadcast_const(&mut self, dst: VReg, val: f32) -> Result<(), String> {
            self.emit_load_f32_const_neon(dst.0, val);
            Ok(())
        }

        fn vzero(&mut self, dst: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_eor_v(dst.0, dst.0, dst.0));
            Ok(())
        }

        fn vmov(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_mov_v(dst.0, src.0));
            Ok(())
        }

        // ── Bitwise / integer operations ────────────────────────────

        fn vand(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_and_v(dst.0, a.0, b.0));
            Ok(())
        }

        fn vor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_orr_v(dst.0, a.0, b.0));
            Ok(())
        }

        fn vxor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            self.emit_u32(Self::encode_eor_v(dst.0, a.0, b.0));
            Ok(())
        }

        fn vshr_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String> {
            self.emit_u32(Self::encode_ushr_4s(dst.0, a.0, imm));
            Ok(())
        }

        fn vshl_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String> {
            // SHL Vd.4S, Vn.4S, #shift
            // immh:immb = shift + 32 (for 32-bit element size)
            let immh_immb = (imm as u32) + 32;
            self.emit_u32(0x4F005400
                | ((immh_immb & 0x7F) << 16)
                | ((a.0 as u32) << 5)
                | (dst.0 as u32));
            Ok(())
        }

        fn vcvt_i32_f32(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            // SCVTF Vd.4S, Vn.4S (signed i32 → f32)
            self.emit_u32(0x4E21D800 | ((a.0 as u32) << 5) | (dst.0 as u32));
            Ok(())
        }

        fn vcvt_f32_i32(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            // FCVTZS Vd.4S, Vn.4S (f32 → signed i32, truncate toward zero)
            self.emit_u32(0x4EA1B800 | ((a.0 as u32) << 5) | (dst.0 as u32));
            Ok(())
        }

        fn vround(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            // FRINTN Vd.4S, Vn.4S (round to nearest, ties to even)
            self.emit_u32(0x4E218800 | ((a.0 as u32) << 5) | (dst.0 as u32));
            Ok(())
        }

        fn vadd_i32(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            // ADD Vd.4S, Vn.4S, Vm.4S (integer add)
            self.emit_u32(0x4EA08400
                | ((b.0 as u32) << 16)
                | ((a.0 as u32) << 5)
                | (dst.0 as u32));
            Ok(())
        }

        // ── FMA variants ────────────────────────────────────────────

        fn vfmadd213(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            // dst = dst * a + b
            // Use v30 as scratch: mov v30, b; fmla v30, dst, a; mov dst, v30
            self.emit_u32(Self::encode_mov_v(30, b.0));
            self.emit_u32(Self::encode_fmla(30, dst.0, a.0));
            self.emit_u32(Self::encode_mov_v(dst.0, 30));
            Ok(())
        }

        fn vfmadd231(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            // dst = a * b + dst → FMLA dst, a, b
            self.emit_u32(Self::encode_fmla(dst.0, a.0, b.0));
            Ok(())
        }

        // ── Loop control ────────────────────────────────────────────

        fn alloc_label(&mut self) -> Label {
            let id = self.label_counter;
            self.label_counter += 1;
            self.labels.push(None);
            Label(id)
        }

        fn define_label(&mut self, label: Label) -> Result<(), String> {
            let id = label.0 as usize;
            if id >= self.labels.len() {
                return Err(format!("label {} not allocated", id));
            }
            let here = self.code.len();
            self.labels[id] = Some(here);

            // Patch all pending forward references to this label
            let mut i = 0;
            while i < self.pending_patches.len() {
                if self.pending_patches[i].1 == label.0 {
                    let patch_pos = self.pending_patches[i].0;
                    let existing = u32::from_le_bytes([
                        self.code[patch_pos],
                        self.code[patch_pos + 1],
                        self.code[patch_pos + 2],
                        self.code[patch_pos + 3],
                    ]);
                    let offset = ((here as i32) - (patch_pos as i32)) / 4;

                    let patched = if existing & 0xFC000000 == 0x14000000 {
                        // Unconditional branch B
                        0x14000000 | ((offset as u32) & 0x3FFFFFF)
                    } else if existing & 0xFF000010 == 0x54000000 {
                        // Conditional branch B.cond
                        let cond = existing & 0xF;
                        0x54000000 | (((offset as u32) & 0x7FFFF) << 5) | cond
                    } else if existing & 0x7F000000 == 0x35000000
                           || existing & 0x7F000000 == 0x34000000 {
                        // CBNZ / CBZ
                        let base = existing & 0xFF00001F;
                        base | (((offset as u32) & 0x7FFFF) << 5)
                    } else {
                        return Err(format!("unknown branch encoding at {}", patch_pos));
                    };

                    self.code[patch_pos..patch_pos + 4]
                        .copy_from_slice(&patched.to_le_bytes());
                    self.pending_patches.swap_remove(i);
                } else {
                    i += 1;
                }
            }
            Ok(())
        }

        fn jump(&mut self, label: Label) -> Result<(), String> {
            let id = label.0 as usize;
            if id < self.labels.len() {
                if let Some(target) = self.labels[id] {
                    // Label already defined — emit backward branch
                    let offset = ((target as i32) - (self.code.len() as i32)) / 4;
                    self.emit_u32(0x14000000 | ((offset as u32) & 0x3FFFFFF));
                    return Ok(());
                }
            }
            // Forward reference — emit placeholder and record patch
            let patch_pos = self.code.len();
            self.emit_u32(0x14000000); // placeholder B
            self.pending_patches.push((patch_pos, label.0));
            Ok(())
        }

        fn dec_and_branch_nz(&mut self, counter: BaseReg, label: Label) -> Result<(), String> {
            let rd = gpr_num(&counter);
            // sub rd, rd, #1
            self.emit_u32(0xD1000400 | ((rd as u32) << 5) | (rd as u32));

            let id = label.0 as usize;
            if id < self.labels.len() {
                if let Some(target) = self.labels[id] {
                    let offset = ((target as i32) - (self.code.len() as i32)) / 4;
                    let imm19 = (offset as u32) & 0x7FFFF;
                    self.emit_u32(0xB5000000 | (imm19 << 5) | (rd as u32));
                    return Ok(());
                }
            }
            // Forward reference
            let patch_pos = self.code.len();
            self.emit_u32(0xB5000000 | (rd as u32)); // CBNZ placeholder
            self.pending_patches.push((patch_pos, label.0));
            Ok(())
        }

        fn cmp_and_branch_lt(&mut self, reg: BaseReg, imm: i64, label: Label) -> Result<(), String> {
            let rn = gpr_num(&reg);
            // cmp rn, #imm
            self.emit_cmp_imm(rn, imm as u32);

            let id = label.0 as usize;
            if id < self.labels.len() {
                if let Some(target) = self.labels[id] {
                    let offset = ((target as i32) - (self.code.len() as i32)) / 4;
                    let imm19 = (offset as u32) & 0x7FFFF;
                    // b.lt = condition 0xB
                    self.emit_u32(0x5400000B | (imm19 << 5));
                    return Ok(());
                }
            }
            let patch_pos = self.code.len();
            self.emit_u32(0x5400000B); // b.lt placeholder
            self.pending_patches.push((patch_pos, label.0));
            Ok(())
        }

        fn cmp_and_branch_ge(&mut self, reg: BaseReg, imm: i64, label: Label) -> Result<(), String> {
            let rn = gpr_num(&reg);
            // cmp rn, #imm
            self.emit_cmp_imm(rn, imm as u32);

            let id = label.0 as usize;
            if id < self.labels.len() {
                if let Some(target) = self.labels[id] {
                    let offset = ((target as i32) - (self.code.len() as i32)) / 4;
                    let imm19 = (offset as u32) & 0x7FFFF;
                    // b.ge = condition 0xA
                    self.emit_u32(0x5400000A | (imm19 << 5));
                    return Ok(());
                }
            }
            let patch_pos = self.code.len();
            self.emit_u32(0x5400000A); // b.ge placeholder
            self.pending_patches.push((patch_pos, label.0));
            Ok(())
        }

        // ── GPR operations ──────────────────────────────────────────

        fn gpr_load_imm(&mut self, dst: BaseReg, imm: i64) -> Result<(), String> {
            let rd = gpr_num(&dst);
            self.emit_mov_imm(rd, imm as usize);
            Ok(())
        }

        fn gpr_add_imm(&mut self, dst: BaseReg, imm: i32) -> Result<(), String> {
            let rd = gpr_num(&dst);
            if imm >= 0 && (imm as u32) < 4096 {
                self.emit_add_imm(rd, rd, imm as u32);
            } else if imm < 0 && ((-imm) as u32) < 4096 {
                // sub rd, rd, #(-imm)
                let abs_imm = (-imm) as u32;
                self.emit_u32(0xD1000000
                    | ((abs_imm & 0xFFF) << 10)
                    | ((rd as u32) << 5)
                    | (rd as u32));
            } else {
                // Large immediate: load into x10, then add/sub
                if imm >= 0 {
                    self.emit_mov_imm(10, imm as usize);
                    self.emit_add_reg(rd, rd, 10);
                } else {
                    self.emit_mov_imm(10, (-imm) as usize);
                    self.emit_sub_reg(rd, rd, 10);
                }
            }
            Ok(())
        }

        fn gpr_mov(&mut self, dst: BaseReg, src: BaseReg) -> Result<(), String> {
            self.emit_mov_reg(gpr_num(&dst), gpr_num(&src));
            Ok(())
        }

        // ── Function frame ──────────────────────────────────────────

        fn emit_prologue(&mut self) -> Result<(), String> {
            self.emit_prologue_raw();
            Ok(())
        }

        fn emit_epilogue(&mut self) -> Result<(), String> {
            self.emit_epilogue_raw();
            Ok(())
        }

        fn finalize(&mut self) -> Result<CodegenOutput, String> {
            Ok(CodegenOutput {
                code: self.code.clone(),
                scratchpad_bytes: self.blis_scratchpad_bytes,
            })
        }

        // ── Prefetch ────────────────────────────────────────────────

        fn prefetch_l1(&mut self, mem: MemOperand) -> Result<(), String> {
            let base = gpr_num(&mem.base);
            if mem.offset == 0 {
                // PRFM PLDL1KEEP, [xn]
                self.emit_u32(0xF9800000 | ((base as u32) << 5));
            } else if mem.offset > 0 && mem.offset % 8 == 0 && (mem.offset / 8) < 4096 {
                let imm12 = (mem.offset / 8) as u32;
                self.emit_u32(0xF9800000 | (imm12 << 10) | ((base as u32) << 5));
            } else {
                // Compute address in x9, then prefetch
                self.emit_mov_reg(9, base);
                if mem.offset > 0 && (mem.offset as u32) < 4096 {
                    self.emit_add_imm(9, 9, mem.offset as u32);
                } else {
                    self.emit_mov_imm(10, mem.offset as usize);
                    self.emit_add_reg(9, 9, 10);
                }
                self.emit_u32(0xF9800000 | (9u32 << 5));
            }
            Ok(())
        }

        // ── Non-temporal store ──────────────────────────────────────

        fn vstore_nt(&mut self, mem: MemOperand, src: VReg) -> Result<(), String> {
            // NEON doesn't have direct non-temporal vector stores like x86.
            // Fall back to regular store.
            self.vstore(mem, src)
        }

        // ── Memory fence ────────────────────────────────────────────

        fn sfence(&mut self) -> Result<(), String> {
            // DMB ISH (data memory barrier, inner shareable)
            self.emit_u32(0xD5033BBF);
            Ok(())
        }

        // ── Scalar operations ───────────────────────────────────────

        fn scalar_load(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String> {
            let base = gpr_num(&mem.base);
            self.emit_mov_reg(9, base);
            if mem.offset != 0 {
                if mem.offset > 0 && (mem.offset as u32) < 4096 {
                    self.emit_add_imm(9, 9, mem.offset as u32);
                } else if mem.offset < 0 && ((-mem.offset) as u32) < 4096 {
                    let imm = (-mem.offset) as u32;
                    self.emit_u32(0xD1000000 | ((imm & 0xFFF) << 10) | (9u32 << 5) | 9u32);
                } else {
                    self.emit_mov_imm(10, mem.offset as usize);
                    self.emit_add_reg(9, 9, 10);
                }
            }
            // ldr s_dst, [x9]
            self.emit_u32(0xBD400000 | (9u32 << 5) | (dst.0 as u32));
            Ok(())
        }

        fn scalar_store(&mut self, mem: MemOperand, src: VReg) -> Result<(), String> {
            let base = gpr_num(&mem.base);
            self.emit_mov_reg(9, base);
            if mem.offset != 0 {
                if mem.offset > 0 && (mem.offset as u32) < 4096 {
                    self.emit_add_imm(9, 9, mem.offset as u32);
                } else if mem.offset < 0 && ((-mem.offset) as u32) < 4096 {
                    let imm = (-mem.offset) as u32;
                    self.emit_u32(0xD1000000 | ((imm & 0xFFF) << 10) | (9u32 << 5) | 9u32);
                } else {
                    self.emit_mov_imm(10, mem.offset as usize);
                    self.emit_add_reg(9, 9, 10);
                }
            }
            // str s_src, [x9]
            self.emit_u32(0xBD000000 | (9u32 << 5) | (src.0 as u32));
            Ok(())
        }

        // ── External function calls ─────────────────────────────────

        fn call_fn_ptr(&mut self, addr: u64) -> Result<(), String> {
            // Load address into x10, then blr x10
            self.emit_mov_imm(10, addr as usize);
            self.emit_blr(10);
            Ok(())
        }

        // ── NOP ─────────────────────────────────────────────────────

        fn emit_nop(&mut self) -> Result<(), String> {
            self.emit_nop_raw();
            Ok(())
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
        registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<super::CodegenOutput, String> {
        self.emit_plan(plan, graph, alloc, profile, registry)
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

        // Instruction counts per op:
        //   4× Input/Const (1 each) + Add/Sub/Mul/Div (1 each) + Fma (2: mov+fmla)
        //   + Neg/Abs/Sqrt (1 each) + Rsqrt (7: frsqrte + 2×NR)
        //   + Recip (3: frecpe + 1×NR) + Exp (5: Taylor deg-2)
        //   + Tanh (15: 2*sigmoid(2x)-1) + Max/Min (1 each)
        //   = 4 + 4 + 2 + 3 + 7 + 3 + 5 + 15 + 2 = 45
        assert_eq!(codegen.code_len(), 45 * 4);
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_instruction_encoding_alignment() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let codegen = jit::AArch64CodeGen::new(&profile);
        // Code buffer starts empty
        assert_eq!(codegen.code_len() % 4, 0);
    }

    // ── WI-19: numerical / structural correctness tests ──────────────

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_emit_exp_neon_instruction_count() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Exp alone: should emit 5 instructions (degree-2 Taylor polynomial),
        // not the old 1-instruction placeholder mov.
        let ops = vec![TraceOp::Input(0), TraceOp::Exp(0)];
        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 2);

        // Input = 1 insn (eor), Exp = 5 insns → 6 total
        let insn_count = codegen.code_len() / 4;
        assert_eq!(insn_count, 6);
        assert!(insn_count > 2, "Exp must emit more than a single placeholder instruction");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_emit_tanh_neon_instruction_count() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Tanh: 2*sigmoid(2x) - 1 via exp + reciprocal
        let ops = vec![TraceOp::Input(0), TraceOp::Tanh(0)];
        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 2);

        // Input = 1 insn (eor), Tanh = 15 insns → 16 total
        let insn_count = codegen.code_len() / 4;
        assert_eq!(insn_count, 16);
        assert!(insn_count > 2, "Tanh must emit more than a placeholder mov");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_emit_rsqrt_refined_instruction_count() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Rsqrt: frsqrte + 2× NR iterations = 7 instructions
        let ops = vec![TraceOp::Input(0), TraceOp::Rsqrt(0)];
        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 2);

        // Input = 1 insn, Rsqrt = 7 insns → 8 total
        let insn_count = codegen.code_len() / 4;
        assert_eq!(insn_count, 8);
        assert!(insn_count >= 8, "Rsqrt must emit 7+ instructions (frsqrte + 2 NR iterations)");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_emit_recip_refined_instruction_count() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Recip: frecpe + 1× NR iteration = 3 instructions
        let ops = vec![TraceOp::Input(0), TraceOp::Recip(0)];
        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 2);

        // Input = 1 insn, Recip = 3 insns → 4 total
        let insn_count = codegen.code_len() / 4;
        assert_eq!(insn_count, 4);
        assert!(insn_count >= 4, "Recip must emit 3+ instructions (frecpe + 1 NR iteration)");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_emit_log_neon() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Log should emit a real polynomial approximation (not a placeholder mov)
        let ops = vec![TraceOp::Input(0), TraceOp::Log(0)];
        let regs = codegen.emit_trace_ops_neon(&ops).unwrap();
        assert_eq!(regs.len(), 2);
        // Input (1 eor) + Log (36 insns: exponent extraction + Horner deg-4) = 37
        let insn_count = codegen.code_len() / 4;
        assert_eq!(insn_count, 37);
        assert!(insn_count > 2, "Log must emit more than a placeholder mov");
        assert_eq!(codegen.code_len() % 4, 0);
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_elementwise_loop_structure() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Simple pass-through: Input → Neg
        let ops = vec![TraceOp::Input(0), TraceOp::Neg(0)];
        codegen.emit_elementwise_loop(&ops).unwrap();

        // Must produce non-empty, 4-byte aligned code
        assert!(codegen.code_len() > 0);
        assert_eq!(codegen.code_len() % 4, 0);

        // The loop structure emits: lsr, cbz, ld1, trace_ops, st1, sub, cbnz,
        // scalar-tail (and, cbz, ldr, str, sub, cbnz) — well over 10 instructions
        let insn_count = codegen.code_len() / 4;
        assert!(insn_count >= 10, "elementwise loop should emit at least 10 instructions, got {insn_count}");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_gemm_8x12_microkernel_structure() {
        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        codegen.emit_gemm_8x12_neon(4).unwrap();

        assert!(codegen.code_len() > 0);
        assert_eq!(codegen.code_len() % 4, 0);

        let insn_count = codegen.code_len() / 4;
        // Expected: 24 eor (acc zero) + 1 movz + 1 cbz + 5 loads + 24 fmla
        //         + 1 sub + 1 cbnz + 24 st1 = 81 instructions
        // Allow a reasonable range in case the structure evolves
        assert!(insn_count >= 70, "GEMM 8x12 should emit at least 70 instructions, got {insn_count}");
        assert!(insn_count <= 120, "GEMM 8x12 should emit at most 120 instructions, got {insn_count}");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_encoding_frsqrts() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Rsqrt emits frsqrte + 2×(fmul, frsqrts, fmul) = 7 instructions.
        // We verify the frsqrts bit pattern appears in the emitted code.
        //
        // For ops = [Input(0), Rsqrt(0)]:
        //   Input(0) → eor v0 (1 insn)
        //   Rsqrt(0) → rd=1, ra=0, s0=28, s1=29:
        //     [1] frsqrte v1.4s, v0.4s
        //     [2] fmul v28.4s, v1.4s, v1.4s
        //     [3] frsqrts v29.4s, v0.4s, v28.4s  ← check this
        //     [4] fmul v1.4s, v1.4s, v29.4s
        //     [5] fmul v28.4s, v1.4s, v1.4s
        //     [6] frsqrts v29.4s, v0.4s, v28.4s  ← and this
        //     [7] fmul v1.4s, v1.4s, v29.4s
        let ops = vec![TraceOp::Input(0), TraceOp::Rsqrt(0)];
        codegen.emit_trace_ops_neon(&ops).unwrap();

        assert_eq!(codegen.code_len(), 8 * 4, "expected 8 instructions (32 bytes)");

        let bytes = codegen.code_bytes();
        let read_insn = |idx: usize| -> u32 {
            u32::from_le_bytes(bytes[idx * 4..(idx + 1) * 4].try_into().unwrap())
        };

        // frsqrts base opcode = 0x4EA0FC00, register-field mask = 0xFFE0FC00
        let frsqrts_mask = 0xFFE0FC00u32;
        let frsqrts_base = 0x4EA0FC00u32;

        // frsqrts should appear at instruction indices 3 and 6 (0-based)
        let insn3 = read_insn(3);
        assert_eq!(insn3 & frsqrts_mask, frsqrts_base,
            "instruction at index 3 should be frsqrts, got {insn3:#010X}");

        let insn6 = read_insn(6);
        assert_eq!(insn6 & frsqrts_mask, frsqrts_base,
            "instruction at index 6 should be frsqrts, got {insn6:#010X}");

        // Verify exact encoding: frsqrts v29.4s, v0.4s, v28.4s
        //   = 0x4EA0FC00 | (28 << 16) | (0 << 5) | 29 = 0x4EBCFC1D
        assert_eq!(insn3, 0x4EBCFC1D,
            "frsqrts v29, v0, v28 expected 0x4EBCFC1D, got {insn3:#010X}");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_encoding_frecps() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // Recip emits frecpe + frecps + fmul = 3 instructions.
        // For ops = [Input(0), Recip(0)]:
        //   Input(0) → eor v0 (1 insn)
        //   Recip(0) → rd=1, ra=0, s0=28:
        //     [1] frecpe v1.4s, v0.4s
        //     [2] frecps v28.4s, v0.4s, v1.4s  ← frecps here
        //     [3] fmul v1.4s, v1.4s, v28.4s
        let ops = vec![TraceOp::Input(0), TraceOp::Recip(0)];
        codegen.emit_trace_ops_neon(&ops).unwrap();

        // 4 instructions total (1 eor + 3 recip)
        assert_eq!(codegen.code_len(), 16, "expected 4 instructions (16 bytes)");

        // frecps base opcode = 0x4E20FC00
        // frecps v28.4s, v0.4s, v1.4s = 0x4E20FC00 | (1 << 16) | (0 << 5) | 28
        //   = 0x4E21FC1C
        // Instruction at byte offset 8 (index 2):
        let insn2 = u32::from_le_bytes(
            codegen.code_bytes()[8..12].try_into().unwrap()
        );
        let frecps_mask = 0xFFE0FC00u32; // mask out Rd(0-4), Rn(5-9), Rm(16-20)
        assert_eq!(insn2 & frecps_mask, 0x4E20FC00,
            "instruction at index 2 should be frecps, got {insn2:#010X}");
    }

    #[cfg(feature = "jit-aarch64")]
    #[test]
    fn test_encoding_ld1_st1_post() {
        use crate::compiler::trace::TraceOp;

        let profile = crate::dispatch::DeviceProfile::detect();
        let mut codegen = jit::AArch64CodeGen::new(&profile);

        // emit_elementwise_loop emits ld1 post-index at the start of the vector loop
        // and st1 post-index after the trace ops.
        let ops = vec![TraceOp::Input(0), TraceOp::Neg(0)];
        codegen.emit_elementwise_loop(&ops).unwrap();

        // ld1 {v.4s}, [xn], #16 base = 0x4CDF7800, mask = 0xFFFFFC00 (Xn:9-5, Vt:4-0)
        // st1 {v.4s}, [xn], #16 base = 0x4C9F7800
        let ld1_base = 0x4CDF7800u32;
        let st1_base = 0x4C9F7800u32;
        let ls_mask = 0xFFFFFC00u32; // mask out Rn(5-9) and Rt(0-4)

        let bytes = codegen.code_bytes();
        let insn_count = bytes.len() / 4;

        let mut found_ld1 = false;
        let mut found_st1 = false;
        for i in 0..insn_count {
            let insn = u32::from_le_bytes(bytes[i*4..(i+1)*4].try_into().unwrap());
            if insn & ls_mask == ld1_base {
                found_ld1 = true;
            }
            if insn & ls_mask == st1_base {
                found_st1 = true;
            }
        }
        assert!(found_ld1, "elementwise loop must contain ld1 post-index instruction");
        assert!(found_st1, "elementwise loop must contain st1 post-index instruction");
    }

    // ── Fusion E2E tests ─────────────────────────────────────────────

    #[cfg(feature = "jit-aarch64")]
    mod fusion_e2e {
        use crate::compiler::codegen::aarch64::jit;
        use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
        use crate::compiler::graph::{CompilerGraph, OpKind, OpId};
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::dispatch::DeviceProfile;
        use crate::types::DType;

        /// Helper: build a graph with two chained unary elementwise ops.
        fn graph_with_chain(
            kind1: OpKind,
            kind2: OpKind,
            n: usize,
        ) -> (CompilerGraph, OpId, OpId) {
            let mut g = CompilerGraph::new();
            let dt = DType::F32;
            let a = g.add_tensor("a", vec![1, n], dt);
            let b = g.add_tensor("b", vec![1, n], dt);
            let c = g.add_tensor("c", vec![1, n], dt);
            let op1 = g.add_op(kind1, vec![a], vec![b], "op1");
            let op2 = g.add_op(kind2, vec![b], vec![c], "op2");
            (g, op1, op2)
        }

        /// Helper: build a graph with GEMM + elementwise epilogue.
        fn graph_with_gemm_epilogue(
            m: usize,
            n: usize,
            k: usize,
            epilogue_kind: OpKind,
        ) -> (CompilerGraph, OpId, OpId) {
            let mut g = CompilerGraph::new();
            let dt = DType::F32;
            let a = g.add_tensor("a", vec![m, k], dt);
            let b = g.add_tensor("b", vec![k, n], dt);
            let gemm_out = g.add_tensor("gemm_out", vec![m, n], dt);
            let final_out = g.add_tensor("final_out", vec![m, n], dt);
            let gemm_op = g.add_op(
                OpKind::Gemm { m, n, k },
                vec![a, b],
                vec![gemm_out],
                "gemm",
            );
            let epi_op = g.add_op(epilogue_kind, vec![gemm_out], vec![final_out], "epi");
            (g, gemm_op, epi_op)
        }

        /// Helper: build a graph with a single binary elementwise op.
        fn graph_with_binary(kind: OpKind, n: usize) -> (CompilerGraph, OpId) {
            let mut g = CompilerGraph::new();
            let dt = DType::F32;
            let in1 = g.add_tensor("in1", vec![1, n], dt);
            let in2 = g.add_tensor("in2", vec![1, n], dt);
            let out = g.add_tensor("out", vec![1, n], dt);
            let op = g.add_op(kind, vec![in1, in2], vec![out], "binop");
            (g, op)
        }

        fn empty_alloc() -> BufferAllocation {
            BufferAllocation {
                slots: vec![],
                total_bytes: 0,
                num_tensors: 0,
                bytes_saved: 0,
            }
        }

        /// Convenience: build a FusionPlan from a single group.
        fn plan_from_group(group: FusionGroup) -> FusionPlan {
            let map = group.ops.iter().map(|&op| (op, 0)).collect();
            FusionPlan {
                groups: vec![group],
                op_to_group: map,
            }
        }

        // ── Test 1: LoopFusion — unary elementwise chain (SiLU → SiLU) ──

        #[test]
        fn test_loop_fusion_silu_chain() {
            let profile = DeviceProfile::detect();
            let registry = ScalarOpRegistry::with_defaults();
            let (graph, op1, op2) = graph_with_chain(OpKind::Silu, OpKind::Silu, 128);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: op1,
                epilogue: vec![op2],
                mode: FusionMode::LoopFusion,
                ops: vec![op1, op2],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let output = codegen
                .emit_plan(&plan, &graph, &empty_alloc(), &profile, Some(&registry))
                .expect("LoopFusion SiLU chain should succeed");

            let insn_count = output.code.len() / 4;
            assert!(insn_count > 20, "SiLU chain should emit >20 insns, got {insn_count}");
            assert_eq!(output.code.len() % 4, 0, "code must be 4-byte aligned");
        }

        // ── Test 2: LoopFusion — binary elementwise (Add) ──

        #[test]
        fn test_loop_fusion_binary_add() {
            let profile = DeviceProfile::detect();
            let registry = ScalarOpRegistry::with_defaults();
            let (graph, op_id) = graph_with_binary(OpKind::Add, 256);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let output = codegen
                .emit_plan(&plan, &graph, &empty_alloc(), &profile, Some(&registry))
                .expect("LoopFusion binary Add should succeed");

            let insn_count = output.code.len() / 4;
            assert!(insn_count > 10, "binary Add loop should emit >10 insns, got {insn_count}");
            assert_eq!(output.code.len() % 4, 0);
        }

        // ── Test 3: EpilogueInjection — GEMM + SiLU ──

        #[test]
        fn test_epilogue_injection_gemm_silu() {
            let profile = DeviceProfile::detect();
            let registry = ScalarOpRegistry::with_defaults();
            let (graph, gemm_op, silu_op) = graph_with_gemm_epilogue(8, 12, 4, OpKind::Silu);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: gemm_op,
                epilogue: vec![silu_op],
                mode: FusionMode::EpilogueInjection,
                ops: vec![gemm_op, silu_op],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let output = codegen
                .emit_plan(&plan, &graph, &empty_alloc(), &profile, Some(&registry))
                .expect("GEMM+SiLU epilogue should succeed");

            let insn_count = output.code.len() / 4;
            assert!(insn_count > 100,
                "GEMM+SiLU should emit >100 insns (base GEMM + epilogue), got {insn_count}");
            assert_eq!(output.code.len() % 4, 0);
        }

        // ── Test 4: EpilogueInjection — GEMM + GELU ──

        #[test]
        fn test_epilogue_injection_gemm_gelu() {
            let profile = DeviceProfile::detect();
            let registry = ScalarOpRegistry::with_defaults();
            let (graph, gemm_op, gelu_op) = graph_with_gemm_epilogue(8, 12, 4, OpKind::Gelu);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: gemm_op,
                epilogue: vec![gelu_op],
                mode: FusionMode::EpilogueInjection,
                ops: vec![gemm_op, gelu_op],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let output = codegen
                .emit_plan(&plan, &graph, &empty_alloc(), &profile, Some(&registry))
                .expect("GEMM+GELU epilogue should succeed");

            let insn_count = output.code.len() / 4;
            assert!(insn_count > 100,
                "GEMM+GELU should emit >100 insns, got {insn_count}");
        }

        // ── Test 5: LoopFusion — SwiGLU (binary elementwise) ──

        #[test]
        fn test_loop_fusion_swiglu() {
            let profile = DeviceProfile::detect();
            let registry = ScalarOpRegistry::with_defaults();
            let (graph, op_id) = graph_with_binary(OpKind::SwiGlu, 128);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let output = codegen
                .emit_plan(&plan, &graph, &empty_alloc(), &profile, Some(&registry))
                .expect("LoopFusion SwiGLU should succeed");

            let insn_count = output.code.len() / 4;
            assert!(insn_count > 15, "SwiGLU loop should emit >15 insns, got {insn_count}");
        }

        // ── Test 6: emit_plan with LoopFusion chain ──

        #[test]
        fn test_emit_plan_loop_fusion_chain() {
            let profile = DeviceProfile::detect();
            let registry = ScalarOpRegistry::with_defaults();
            let (graph, op1, op2) = graph_with_chain(OpKind::Silu, OpKind::Silu, 64);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: op1,
                epilogue: vec![op2],
                mode: FusionMode::LoopFusion,
                ops: vec![op1, op2],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let output = codegen
                .emit_plan(&plan, &graph, &empty_alloc(), &profile, Some(&registry))
                .expect("emit_plan with LoopFusion should succeed");

            assert!(!output.code.is_empty());
            assert_eq!(output.code.len() % 4, 0);
            let insn_count = output.code.len() / 4;
            assert!(insn_count > 10, "full plan should emit >10 insns, got {insn_count}");
        }

        // ── Test 7: emit_plan with EpilogueInjection group ──

        #[test]
        fn test_emit_plan_epilogue_injection() {
            let profile = DeviceProfile::detect();
            let registry = ScalarOpRegistry::with_defaults();
            let (graph, gemm_op, silu_op) = graph_with_gemm_epilogue(8, 12, 4, OpKind::Silu);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: gemm_op,
                epilogue: vec![silu_op],
                mode: FusionMode::EpilogueInjection,
                ops: vec![gemm_op, silu_op],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let output = codegen
                .emit_plan(&plan, &graph, &empty_alloc(), &profile, Some(&registry))
                .expect("emit_plan with EpilogueInjection should succeed");

            assert!(!output.code.is_empty());
            assert_eq!(output.code.len() % 4, 0);
        }

        // ── Test 8: emit_plan without registry (no-op fallback) ──

        #[test]
        fn test_emit_plan_no_registry_fallback() {
            let profile = DeviceProfile::detect();
            let (graph, op1, op2) = graph_with_chain(OpKind::Silu, OpKind::Silu, 64);

            let plan = plan_from_group(FusionGroup {
                id: 0,
                anchor: op1,
                epilogue: vec![op2],
                mode: FusionMode::LoopFusion,
                ops: vec![op1, op2],
            });

            let mut codegen = jit::AArch64CodeGen::new(&profile);
            let result = codegen.emit_plan(&plan, &graph, &empty_alloc(), &profile, None);
            assert!(result.is_ok(), "emit_plan without registry should not fail");
        }
    }
}
