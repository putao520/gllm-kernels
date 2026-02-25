//! x86_64 JIT code generation via iced-x86 CodeAssembler.
//!
//! Phase 3 of the JIT compiler pipeline: generates native x86_64 machine code
//! from FusionPlan + CompilerGraph + BufferAllocation.
//!
//! MVP implemented under the `jit-x86` feature flag (`jit::X86CodeGen`):
//! - TraceOp → AVX2 SIMD instruction mapping (elementwise kernel generation)
//! - GEMM microkernel generation (3-level blocked loop with nop body markers)
//! - Fused elementwise loops (nop markers per fused op; real SIMD loop is next)
//! - GEMM epilogue injection (nop markers; real epilogue on accumulators is next)
//! - Full `emit_plan()` pipeline: prologue → groups → epilogue → assemble
//!
//! Without the feature flag, only `emit_stub()` is available (minimal valid function).

use super::CodegenOutput;

/// Emit a minimal x86_64 stub (push rbp; mov rbp,rsp; nops; pop rbp; ret).
///
/// Used as fallback until the real Phase 3 code generator is implemented.
pub fn emit_stub() -> CodegenOutput {
    let mut code = Vec::with_capacity(16);
    code.push(0x55);                          // push rbp
    code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
    for _ in 0..8 {
        code.push(0x90);                      // nop
    }
    code.push(0x5D);                          // pop rbp
    code.push(0xC3);                          // ret
    CodegenOutput { code, scratchpad_bytes: 0 }
}

#[cfg(feature = "jit-x86")]
pub mod jit {
    use iced_x86::code_asm::*;
    use crate::compiler::trace::{TraceOp, ComputePattern};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::graph::{CompilerGraph, OpKind, OpId};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::dispatch::DeviceProfile;
    use super::CodegenOutput;

    /// x86_64 JIT code generator.
    ///
    /// Translates FusionPlan groups into native AVX2/AVX-512 machine code
    /// via iced-x86's `CodeAssembler`.
    pub struct X86CodeGen {
        asm: CodeAssembler,
        use_avx512: bool,
        /// Number of f32 elements per SIMD register (8 for AVX2, 16 for AVX-512).
        simd_width: usize,
        /// Constant pool: f32 values to be emitted after code section.
        /// Each entry is an 8×f32 (256-bit) broadcast-ready constant.
        const_pool: Vec<[f32; 8]>,
        /// Labels for each constant pool entry (RIP-relative addressing).
        const_labels: Vec<CodeLabel>,
        /// Extra scratchpad bytes needed by BLIS packing buffers.
        blis_scratchpad_bytes: usize,
        /// Offset within scratchpad where BLIS packing buffers start.
        blis_scratchpad_offset: usize,
        /// Initial value of blis_scratchpad_offset (= alloc.total_bytes).
        blis_base_offset: usize,
        /// Whether rdx (external tensor pointer) was saved to [rbp-48] for
        /// epilogue Input(1) access (e.g. bias vector in GEMM+AddBias).
        bias_saved: bool,
    }

    impl X86CodeGen {
        /// Create a new code generator configured for the detected hardware.
        pub fn new(profile: &DeviceProfile) -> Self {
            let use_avx512 = profile.kernel_config.use_avx512;
            X86CodeGen {
                asm: CodeAssembler::new(64).unwrap(),
                use_avx512,
                simd_width: if use_avx512 { 16 } else { 8 },
                const_pool: Vec::new(),
                const_labels: Vec::new(),
                blis_scratchpad_bytes: 0,
                blis_scratchpad_offset: 0,
                blis_base_offset: 0,
                bias_saved: false,
            }
        }

        /// Number of f32 elements per SIMD register (8 for AVX2, 16 for AVX-512).
        pub fn simd_width(&self) -> usize {
            self.simd_width
        }


        /// Deduplicates: returns existing label if value already present.
        fn const_f32(&mut self, val: f32) -> CodeLabel {
            let bits = val.to_bits();
            for (i, entry) in self.const_pool.iter().enumerate() {
                if entry[0].to_bits() == bits {
                    return self.const_labels[i];
                }
            }
            let label = self.asm.create_label();
            self.const_pool.push([val; 8]);
            self.const_labels.push(label);
            label
        }

        /// Emit the constant pool data after all code.
        /// Must be called before `assemble()`.
        fn emit_const_pool(&mut self) -> Result<(), String> {
            for i in 0..self.const_pool.len() {
                self.asm.set_label(&mut self.const_labels[i]).map_err(|e| e.to_string())?;
                for chunk in self.const_pool[i].chunks(2) {
                    self.asm.dd_f32(chunk).map_err(|e| e.to_string())?;
                }
            }
            Ok(())
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
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<CodegenOutput, String> {
            self.emit_prologue()?;

            self.blis_scratchpad_offset = alloc.total_bytes;
            self.blis_base_offset = alloc.total_bytes;

            for group in &plan.groups {
                self.emit_group(group, graph, alloc, profile, registry)?;
            }

            self.emit_epilogue()?;

            self.emit_const_pool()?;

            let code = self.asm.assemble(0x0)
                .map_err(|e| format!("asm error: {e}"))?;

            Ok(CodegenOutput {
                code,
                scratchpad_bytes: alloc.total_bytes + self.blis_scratchpad_bytes,
            })
        }

        /// System V AMD64 ABI prologue: save frame pointer + callee-saved registers.
        fn emit_prologue(&mut self) -> Result<(), String> {
            self.asm.push(rbp).map_err(|e| e.to_string())?;
            self.asm.mov(rbp, rsp).map_err(|e| e.to_string())?;
            self.asm.push(rbx).map_err(|e| e.to_string())?;
            self.asm.push(r12).map_err(|e| e.to_string())?;
            self.asm.push(r13).map_err(|e| e.to_string())?;
            self.asm.push(r14).map_err(|e| e.to_string())?;
            self.asm.push(r15).map_err(|e| e.to_string())?;
            self.asm.sub(rsp, 8i32).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Restore callee-saved registers and return.
        fn emit_epilogue(&mut self) -> Result<(), String> {
            self.asm.add(rsp, 8i32).map_err(|e| e.to_string())?;
            self.asm.pop(r15).map_err(|e| e.to_string())?;
            self.asm.pop(r14).map_err(|e| e.to_string())?;
            self.asm.pop(r13).map_err(|e| e.to_string())?;
            self.asm.pop(r12).map_err(|e| e.to_string())?;
            self.asm.pop(rbx).map_err(|e| e.to_string())?;
            self.asm.pop(rbp).map_err(|e| e.to_string())?;
            self.asm.ret().map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Dispatch a fusion group to the appropriate emitter.
        fn emit_group(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            match group.mode {
                FusionMode::Standalone => {
                    self.emit_standalone(group, graph, profile)
                }
                FusionMode::LoopFusion => {
                    self.emit_elementwise_chain(group, graph, alloc, profile, registry)
                }
                FusionMode::EpilogueInjection => {
                    self.emit_gemm_with_epilogue(group, graph, alloc, profile, registry)
                }
                FusionMode::QkvSharedInput => {
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
                FusionMode::NormIntoGemm => {
                    self.emit_norm_into_gemm(group, graph, profile)
                }
                FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                    self.emit_tile_level_fusion(group, graph, profile, predecessor, tile_rows)
                }
                FusionMode::ComputeRoot { predecessor } => {
                    self.emit_compute_root(group, graph, profile, predecessor)
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
                OpKind::Gemm { m, n, k } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile, &[])
                }
                OpKind::QuantGemm { m, n, k, block_size, bits } => {
                    self.emit_quant_gemm(*m, *n, *k, *block_size, *bits, profile, &[])
                }
                _ => self.emit_nop_placeholder(),

            }
        }

        /// Emit fused RmsNorm → GEMM: normalize A in-place (via scratchpad),
        /// then run the GEMM with the normalized data as input.
        ///
        /// ABI contract for NormIntoGemm groups:
        ///   rdi = A input ptr (pre-norm)
        ///   rsi = B weight matrix ptr
        ///   rdx = norm weight ptr
        ///   r8  = C output ptr
        ///   [rbp+32] = scratchpad ptr
        ///
        /// The normalized A is written to the scratchpad at offset 0,
        /// consuming m*k*4 bytes. BLIS packing buffers (if used) start
        /// after this region.
        fn emit_norm_into_gemm(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
        ) -> Result<(), String> {
            let norm_op_id = group.ops.first().copied()
                .ok_or("NormIntoGemm: missing norm op")?;
            let norm_op = graph.op(norm_op_id)
                .ok_or("NormIntoGemm: norm op not in graph")?;
            let gemm_op = graph.op(group.anchor)
                .ok_or("NormIntoGemm: GEMM op not in graph")?;

            let eps = match &norm_op.kind {
                OpKind::RmsNorm { eps } => *eps,
                OpKind::LayerNorm { eps } => {
                    return Err("NormIntoGemm: LayerNorm not yet supported".into());
                }
                other => {
                    return Err(format!("NormIntoGemm: expected norm op, got {:?}", other));
                }
            };

            let (m, n, k) = match &gemm_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => {
                    return Err(format!("NormIntoGemm: expected GEMM op, got {:?}", other));
                }
            };

            let norm_buf_bytes = m * k * 4;

            let norm_scratch_offset = self.blis_scratchpad_offset;
            self.blis_scratchpad_offset += norm_buf_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset - self.blis_base_offset
            );

            let norm_fn_ptr = crate::scalar_ops::norms::scalar_rms_norm as *const () as u64;

            let row_bytes = (k * 4) as i32;

            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            self.asm.sub(rsp, 16i32).map_err(|e| e.to_string())?;  // 16-byte aligned slot
            self.asm.mov(qword_ptr(rsp), r8).map_err(|e| e.to_string())?;  // save C output ptr
            self.asm.mov(r13, rdi).map_err(|e| e.to_string())?;  // A input base
            self.asm.mov(r14, rdx).map_err(|e| e.to_string())?;  // norm weight ptr
            self.asm.mov(r15, rsi).map_err(|e| e.to_string())?;  // B matrix ptr

            let mut row_loop = self.asm.create_label();
            let mut row_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;  // r12 = 0 (row counter)
            self.asm.set_label(&mut row_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(row_done).map_err(|e| e.to_string())?;

            self.asm.mov(rax, r12).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, row_bytes).map_err(|e| e.to_string())?;

            self.asm.lea(rdi, qword_ptr(r13 + rax)).map_err(|e| e.to_string())?;
            self.asm.mov(rsi, r14).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(rbx + rax + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;
            self.asm.mov(rcx, k as u64).map_err(|e| e.to_string())?;

            let eps_label = self.const_f32(eps);
            self.asm.movss(xmm0, dword_ptr(eps_label)).map_err(|e| e.to_string())?;

            self.emit_call_rax(norm_fn_ptr)?;

            self.asm.inc(r12).map_err(|e| e.to_string())?;
            self.asm.jmp(row_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut row_done).map_err(|e| e.to_string())?;

            self.asm.mov(r8, qword_ptr(rsp)).map_err(|e| e.to_string())?;  // restore C output ptr
            self.asm.add(rsp, 16i32).map_err(|e| e.to_string())?;
            self.asm.mov(rsi, r15).map_err(|e| e.to_string())?;  // restore B matrix ptr
            self.asm.lea(rdi, qword_ptr(rbx + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;  // A = normalized data

            self.emit_gemm_microkernel(m, n, k, profile, &[])
        }

        /// Emit TileLevelFusion: tile the predecessor norm into the GEMM's
        /// MC loop so each MC strip computes norm, pack_a, microkernel
        /// before moving to the next strip.
        ///
        /// Used when the full norm output exceeds 75% L1. By computing norm
        /// for only `tile_rows` (= MC) rows at a time, the norm output stays
        /// cache-hot when pack_a reads it immediately after.
        ///
        /// Loop structure (ic moved outside pc):
        /// ```text
        /// jc loop (NC):
        ///   ic loop (MC):
        ///     compute norm for mc_cur rows into norm_scratch
        ///     pc loop (KC):
        ///       pack_b: B[pc..pc+kc, jc..jc+nc] into packed_b
        ///       pack_a: norm_scratch[0..mc_cur, pc..pc+kc] into packed_a
        ///       jr loop (NR):
        ///         ir loop (MR):
        ///           microkernel tile
        /// ```
        ///
        /// ABI contract (same as NormIntoGemm):
        ///   rdi = A input ptr (pre-norm)
        ///   rsi = B weight matrix ptr
        ///   rdx = norm weight ptr
        ///   r8  = C output ptr
        ///   [rbp+32] = scratchpad ptr
        fn emit_tile_level_fusion(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
            tile_rows: usize,
        ) -> Result<(), String> {
            let norm_op = graph.op(predecessor)
                .ok_or("TileLevelFusion: predecessor norm op not in graph")?;
            let gemm_op = graph.op(group.anchor)
                .ok_or("TileLevelFusion: GEMM op not in graph")?;

            let eps = match &norm_op.kind {
                OpKind::RmsNorm { eps } => *eps,
                OpKind::LayerNorm { eps: _ } => {
                    return Err("TileLevelFusion: LayerNorm not yet supported".into());
                }
                other => {
                    return Err(format!("TileLevelFusion: expected norm op, got {:?}", other));
                }
            };

            let (m, n, k) = match &gemm_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => {
                    return Err(format!("TileLevelFusion: expected GEMM op, got {:?}", other));
                }
            };

            let blocking = profile.gemm_blocking(m, n, k);
            let mr = blocking.mr;
            let nr = blocking.nr;
            let nr_vecs = nr / 8; // AVX2: 8 f32 per ymm
            let kc = blocking.kc;
            let mc = tile_rows;
            let nc = blocking.nc;

            let c_row_bytes = (n * 4) as i32;
            let lda = k; // norm output is row-major [mc, k], stride = k
            let ldb = n; // B is row-major [k, n], stride = n

            // -- Scratchpad allocation --
            // norm_scratch: mc * k * 4 (norm output for one MC strip)
            let norm_scratch_bytes = mc * k * 4;
            let norm_scratch_off = self.blis_scratchpad_offset as i32;

            // pack_a / pack_b buffers (same layout as emit_gemm_blis)
            let pack_a_panels = (mc + mr - 1) / mr;
            let pack_a_bytes = pack_a_panels * mr * kc * 4;
            let pack_b_panels = (nc + nr - 1) / nr;
            let pack_b_bytes = pack_b_panels * nr * kc * 4;

            let pack_a_off = (self.blis_scratchpad_offset + norm_scratch_bytes) as i32;
            let pack_b_off = (self.blis_scratchpad_offset + norm_scratch_bytes + pack_a_bytes) as i32;

            let total_extra = self.blis_scratchpad_offset + norm_scratch_bytes
                + pack_a_bytes + pack_b_bytes - self.blis_base_offset;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(total_extra);

            let norm_fn_ptr = crate::scalar_ops::norms::scalar_rms_norm as *const () as u64;
            let pack_a_fn = crate::asm::x86_64::gemm_driver::gllm_pack_a_f32 as *const () as u64;
            let pack_b_fn = crate::asm::x86_64::gemm_driver::gllm_pack_b_f32 as *const () as u64;

            // Load scratchpad base.
            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            // Allocate stack locals (48 bytes, 16-byte aligned):
            //   [rsp+0]  = nc_cur
            //   [rsp+8]  = kc_cur
            //   [rsp+16] = mc_cur
            //   [rsp+24] = ir (reused by inner loops)
            //   [rsp+32] = saved norm weight ptr (rdx)
            //   [rsp+40] = saved original A ptr (rdi)
            self.asm.sub(rsp, 48i32).map_err(|e| e.to_string())?;

            // Save norm weight ptr and original A ptr -- clobbered by inner loops.
            self.asm.mov(qword_ptr(rsp + 32), rdx).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp + 40), rdi).map_err(|e| e.to_string())?;

            let loc_nc: i32 = 0;
            let loc_kc: i32 = 8;
            let loc_mc: i32 = 16;
            let loc_ir: i32 = 24;
            let sloc_nc = loc_nc + Self::CALL_SAVE_SIZE;
            let sloc_kc = loc_kc + Self::CALL_SAVE_SIZE;
            let sloc_mc = loc_mc + Self::CALL_SAVE_SIZE;

            // -- jc loop (NC) --
            let mut jc_loop = self.asm.create_label();
            let mut jc_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, n as i32).map_err(|e| e.to_string())?;
            self.asm.jge(jc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, n, nc, r12)?;
            self.asm.mov(qword_ptr(rsp + loc_nc), rax).map_err(|e| e.to_string())?;

            // -- ic loop (MC) -- moved outside pc for tile-level fusion
            let mut ic_loop = self.asm.create_label();
            let mut ic_done = self.asm.create_label();

            self.asm.xor(r14, r14).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r14, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(ic_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, m, mc, r14)?;
            self.asm.mov(qword_ptr(rsp + loc_mc), rax).map_err(|e| e.to_string())?;

            // -- Compute norm for mc_cur rows into norm_scratch --
            {
                let row_bytes_k = (k * 4) as i32;
                let mut norm_loop = self.asm.create_label();
                let mut norm_done = self.asm.create_label();

                // Save caller-saved regs before norm calls.
                self.emit_save_caller_regs()?;

                // rcx = row counter within MC strip (0..mc_cur)
                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut norm_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + sloc_mc)).map_err(|e| e.to_string())?;
                self.asm.jge(norm_done).map_err(|e| e.to_string())?;

                // Save row counter (rcx is caller-saved, clobbered by call).
                self.asm.push(rcx).map_err(|e| e.to_string())?;

                // Byte offset in original A: (ic + row) * k * 4
                self.asm.mov(rax, r14).map_err(|e| e.to_string())?;
                self.asm.add(rax, rcx).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, row_bytes_k).map_err(|e| e.to_string())?;

                // Byte offset in norm_scratch: row * k * 4
                self.asm.mov(r9, rcx).map_err(|e| e.to_string())?;
                self.asm.imul_3(r9, r9, row_bytes_k).map_err(|e| e.to_string())?;

                // scalar_rms_norm(x, weight, out, n, eps)
                // +8 accounts for the push rcx above
                let saved_a_off = 40 + Self::CALL_SAVE_SIZE + 8;
                let saved_wt_off = 32 + Self::CALL_SAVE_SIZE + 8;
                self.asm.mov(rdi, qword_ptr(rsp + saved_a_off)).map_err(|e| e.to_string())?;
                self.asm.add(rdi, rax).map_err(|e| e.to_string())?;
                self.asm.mov(rsi, qword_ptr(rsp + saved_wt_off)).map_err(|e| e.to_string())?;
                self.asm.lea(rdx, qword_ptr(rbx + norm_scratch_off)).map_err(|e| e.to_string())?;
                self.asm.add(rdx, r9).map_err(|e| e.to_string())?;
                self.asm.mov(rcx, k as u64).map_err(|e| e.to_string())?;

                let eps_label = self.const_f32(eps);
                self.asm.movss(xmm0, dword_ptr(eps_label)).map_err(|e| e.to_string())?;

                self.emit_call_rax(norm_fn_ptr)?;

                // Restore row counter and continue.
                self.asm.pop(rcx).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(norm_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut norm_done).map_err(|e| e.to_string())?;

                self.emit_restore_caller_regs()?;
            }

            // -- pc loop (KC) --
            let mut pc_loop = self.asm.create_label();
            let mut pc_done = self.asm.create_label();

            self.asm.xor(r13, r13).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13, k as i32).map_err(|e| e.to_string())?;
            self.asm.jge(pc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, k, kc, r13)?;
            self.asm.mov(qword_ptr(rsp + loc_kc), rax).map_err(|e| e.to_string())?;

            // -- pack_b: B[pc..pc+kc, jc..jc+nc] into packed_b --
            {
                // B source ptr: &B[pc, jc] = rsi + (pc * ldb + jc) * 4
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, ldb as i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, r12).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, rsi).map_err(|e| e.to_string())?;
                self.asm.mov(r10, rax).map_err(|e| e.to_string())?;

                self.emit_save_caller_regs()?;

                self.asm.mov(rdi, r10).map_err(|e| e.to_string())?;
                self.asm.mov(rsi, ldb as u64).map_err(|e| e.to_string())?;
                self.asm.lea(rdx, qword_ptr(rbx + pack_b_off)).map_err(|e| e.to_string())?;
                self.asm.mov(rcx, qword_ptr(rsp + sloc_kc)).map_err(|e| e.to_string())?;
                self.asm.mov(r8, qword_ptr(rsp + sloc_nc)).map_err(|e| e.to_string())?;
                self.asm.mov(r9, nr as u64).map_err(|e| e.to_string())?;

                self.emit_call_rax(pack_b_fn)?;
                self.emit_restore_caller_regs()?;
            }

            // -- pack_a: norm_scratch[0..mc_cur, pc..pc+kc] into packed_a --
            {
                // Source: &norm_scratch[0, pc] = rbx + norm_scratch_off + pc * 4
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.lea(r15, qword_ptr(rbx + norm_scratch_off)).map_err(|e| e.to_string())?;
                self.asm.add(r15, rax).map_err(|e| e.to_string())?;

                self.emit_save_caller_regs()?;

                self.asm.mov(rdi, r15).map_err(|e| e.to_string())?;
                self.asm.mov(rsi, lda as u64).map_err(|e| e.to_string())?;
                self.asm.lea(rdx, qword_ptr(rbx + pack_a_off)).map_err(|e| e.to_string())?;
                self.asm.mov(rcx, qword_ptr(rsp + sloc_mc)).map_err(|e| e.to_string())?;
                self.asm.mov(r8, qword_ptr(rsp + sloc_kc)).map_err(|e| e.to_string())?;
                self.asm.mov(r9, mr as u64).map_err(|e| e.to_string())?;

                self.emit_call_rax(pack_a_fn)?;
                self.emit_restore_caller_regs()?;
            }

            // -- jr / ir loops (NR / MR) -- reuse existing BLIS inner loops
            {
                let m_rem = m % mr;
                let n_rem = n % nr;
                let simd_w = 8usize;
                let n_rem_vecs = n_rem / simd_w;

                let mut jr_loop = self.asm.create_label();
                let mut jr_done = self.asm.create_label();
                let mut jr_rem_start = self.asm.create_label();

                self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_loop).map_err(|e| e.to_string())?;

                self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
                self.asm.add(rax, nr as i32).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                self.asm.jg(jr_rem_start).map_err(|e| e.to_string())?;

                // r11 = packed_b ptr for this jr strip
                self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                if pack_b_off != 0 {
                    self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                }

                self.emit_blis_ir_loop(
                    mr, nr_vecs, m_rem, loc_kc, n, nr_vecs, nr,
                    loc_ir, loc_mc, pack_a_off, c_row_bytes,
                    &[], k,
                )?;

                self.asm.add(r9, nr as i32).map_err(|e| e.to_string())?;
                self.asm.jmp(jr_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut jr_rem_start).map_err(|e| e.to_string())?;

                if n_rem_vecs > 0 {
                    self.asm.cmp(r9, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                    self.asm.jge(jr_done).map_err(|e| e.to_string())?;

                    self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                    self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                    self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                    self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                    if pack_b_off != 0 {
                        self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                    }

                    self.emit_blis_ir_loop(
                        mr, nr_vecs, m_rem, loc_kc, n, n_rem_vecs, nr,
                        loc_ir, loc_mc, pack_a_off, c_row_bytes,
                        &[], k,
                    )?;
                }

                self.asm.nop().map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_done).map_err(|e| e.to_string())?;
            }

            // -- Close pc loop --
            self.asm.add(r13, kc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(pc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_done).map_err(|e| e.to_string())?;

            // -- Close ic loop --
            self.asm.add(r14, mc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(ic_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_done).map_err(|e| e.to_string())?;

            // -- Close jc loop --
            self.asm.add(r12, nc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(jc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_done).map_err(|e| e.to_string())?;

            self.asm.add(rsp, 48i32).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit ComputeRoot fusion: compute predecessor norm fully into a
        /// scratchpad buffer, then run the GEMM reading from that buffer.
        ///
        /// Used when the norm output fits in L1 (le 75% L1), so no tiling
        /// is needed -- the entire norm result stays cache-hot for the GEMM.
        ///
        /// ABI contract (same as NormIntoGemm):
        ///   rdi = A input ptr (pre-norm)
        ///   rsi = B weight matrix ptr
        ///   rdx = norm weight ptr
        ///   r8  = C output ptr
        ///   [rbp+32] = scratchpad ptr
        fn emit_compute_root(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
        ) -> Result<(), String> {
            let norm_op = graph.op(predecessor)
                .ok_or("ComputeRoot: predecessor norm op not in graph")?;
            let gemm_op = graph.op(group.anchor)
                .ok_or("ComputeRoot: GEMM op not in graph")?;

            let eps = match &norm_op.kind {
                OpKind::RmsNorm { eps } => *eps,
                OpKind::LayerNorm { eps: _ } => {
                    return Err("ComputeRoot: LayerNorm not yet supported".into());
                }
                other => {
                    return Err(format!("ComputeRoot: expected norm op, got {:?}", other));
                }
            };

            let (m, n, k) = match &gemm_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => {
                    return Err(format!("ComputeRoot: expected GEMM op, got {:?}", other));
                }
            };

            // Allocate scratchpad for the full norm output (m * k * 4 bytes).
            let norm_buf_bytes = m * k * 4;
            let norm_scratch_offset = self.blis_scratchpad_offset;
            self.blis_scratchpad_offset += norm_buf_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset - self.blis_base_offset
            );

            let norm_fn_ptr = crate::scalar_ops::norms::scalar_rms_norm as *const () as u64;
            let row_bytes = (k * 4) as i32;

            // Load scratchpad base into rbx.
            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            // Save registers that the norm loop clobbers.
            self.asm.sub(rsp, 16i32).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp), r8).map_err(|e| e.to_string())?;   // save C output ptr
            self.asm.mov(r13, rdi).map_err(|e| e.to_string())?;  // A input base
            self.asm.mov(r14, rdx).map_err(|e| e.to_string())?;  // norm weight ptr
            self.asm.mov(r15, rsi).map_err(|e| e.to_string())?;  // B matrix ptr

            // Row loop: call scalar_rms_norm for each row.
            let mut row_loop = self.asm.create_label();
            let mut row_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;  // r12 = row counter
            self.asm.set_label(&mut row_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(row_done).map_err(|e| e.to_string())?;

            // Compute byte offset for current row.
            self.asm.mov(rax, r12).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, row_bytes).map_err(|e| e.to_string())?;

            // scalar_rms_norm(x, weight, out, n, eps)
            self.asm.lea(rdi, qword_ptr(r13 + rax)).map_err(|e| e.to_string())?;
            self.asm.mov(rsi, r14).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(rbx + rax + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;
            self.asm.mov(rcx, k as u64).map_err(|e| e.to_string())?;

            let eps_label = self.const_f32(eps);
            self.asm.movss(xmm0, dword_ptr(eps_label)).map_err(|e| e.to_string())?;

            self.emit_call_rax(norm_fn_ptr)?;

            self.asm.inc(r12).map_err(|e| e.to_string())?;
            self.asm.jmp(row_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut row_done).map_err(|e| e.to_string())?;

            // Restore registers and redirect A to the norm output buffer.
            self.asm.mov(r8, qword_ptr(rsp)).map_err(|e| e.to_string())?;   // restore C
            self.asm.add(rsp, 16i32).map_err(|e| e.to_string())?;
            self.asm.mov(rsi, r15).map_err(|e| e.to_string())?;  // restore B
            self.asm.lea(rdi, qword_ptr(rbx + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;  // A = norm output

            // Run the GEMM with A pointing to the normalized data.
            self.emit_gemm_microkernel(m, n, k, profile, &[])
        }

        /// Emit a fused elementwise chain as a single SIMD loop.
        ///
        /// All ops in the chain execute on the same data in registers,
        /// no intermediate memory traffic.
        ///
        /// Generated code layout (System V AMD64 ABI):
        /// ```text
        ///   rdi = input ptr (arg 0)
        ///   r8  = scratchpad ptr (arg 4, used as output)
        ///   element_count baked as immediate
        ///
        ///   xor rbx, rbx              ; i = 0
        /// .loop:
        ///   cmp rbx, count
        ///   jge .done
        ///   vmovups ymm0, [rdi + rbx*4]
        ///   ; --- trace ops ---
        ///   vmovups [r8 + rbx*4], ymm_result
        ///   add rbx, 8
        ///   jmp .loop
        /// .done:
        /// ```
        /// Emit a fused elementwise chain as a single SIMD loop.
        ///
        /// All ops in the chain execute on the same data in registers,
        /// no intermediate memory traffic. Trace bodies are looked up
        /// from the `ScalarOpRegistry` when available; falls back to
        /// the hardcoded `emit_chain_body` path otherwise.
        ///
        /// Generated code layout (System V AMD64 ABI):
        /// ```text
        ///   rdi = input ptr (arg 0)
        ///   rsi = second input ptr (arg 1, for binary ops)
        ///   r8  = output ptr (arg 4)
        ///   rbx = byte offset counter
        ///
        ///   xor rbx, rbx              ; i = 0
        /// .loop:
        ///   cmp rbx, total_vec_bytes
        ///   jge .done
        ///   vmovups ymm0, [rdi + rbx]
        ///   ; --- trace ops (from registry) ---
        ///   vmovups [r8 + rbx], ymm0
        ///   add rbx, 32
        ///   jmp .loop
        /// .done:
        ///   ; scalar tail for remaining elements
        /// ```
        fn emit_elementwise_chain(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            _alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor op")?;

            let elem_count = if let Some(&out_id) = op.outputs.first() {
                graph.tensor_numel(out_id).unwrap_or(0)
            } else {
                0
            };

            if elem_count == 0 {
                return self.emit_nop_placeholder();
            }

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
                return self.emit_elementwise_chain_hardcoded(group, graph, elem_count);
            }

            if self.use_avx512 {
                return self.emit_elementwise_chain_avx512(
                    &trace_info, elem_count, profile,
                );
            }

            let simd_w = self.simd_width; // 8 for AVX2
            let vec_count = elem_count / simd_w;
            let tail = elem_count % simd_w;

            let output_bytes = elem_count * 4;
            let (_, l2_size, _) = profile.cache_sizes();
            let use_nt_store = output_bytes > l2_size;

            let has_binary = trace_info.iter().any(|(_, b)| *b);

            let prefetch_dist = 512i32;

            let unroll = 4usize;
            let unrolled_vecs = (vec_count / unroll) * unroll;
            let remainder_vecs = vec_count - unrolled_vecs;

            let unrolled_bytes = (unrolled_vecs * simd_w * 4) as i32;
            let total_vec_bytes = (vec_count * simd_w * 4) as i32;

            let mut unrolled_loop = self.asm.create_label();
            let mut remainder_loop = self.asm.create_label();
            let mut done_label = self.asm.create_label();

            self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;

            if unrolled_vecs > 0 {
                self.asm.set_label(&mut unrolled_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, unrolled_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(remainder_loop).map_err(|e| e.to_string())?;

                self.asm.prefetcht0(byte_ptr(rdi + rbx + prefetch_dist))
                    .map_err(|e| e.to_string())?;
                if has_binary {
                    self.asm.prefetcht0(byte_ptr(rsi + rbx + prefetch_dist))
                        .map_err(|e| e.to_string())?;
                }

                for _u in 0..unroll {
                    self.asm.vmovups(ymm0, ymmword_ptr(rdi + rbx))
                        .map_err(|e| e.to_string())?;

                    for (body, is_binary) in &trace_info {
                        self.emit_elementwise_trace_body(ymm0, body, *is_binary, false)?;
                    }

                    if use_nt_store {
                        self.asm.vmovntps(ymmword_ptr(r8 + rbx), ymm0)
                            .map_err(|e| e.to_string())?;
                    } else {
                        self.asm.vmovups(ymmword_ptr(r8 + rbx), ymm0)
                            .map_err(|e| e.to_string())?;
                    }

                    self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
                }

                self.asm.jmp(unrolled_loop).map_err(|e| e.to_string())?;
            }

            self.asm.set_label(&mut remainder_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
            self.asm.jge(done_label).map_err(|e| e.to_string())?;

            self.asm.prefetcht0(byte_ptr(rdi + rbx + prefetch_dist))
                .map_err(|e| e.to_string())?;

            self.asm.vmovups(ymm0, ymmword_ptr(rdi + rbx))
                .map_err(|e| e.to_string())?;

            for (body, is_binary) in &trace_info {
                self.emit_elementwise_trace_body(ymm0, body, *is_binary, false)?;
            }

            if use_nt_store {
                self.asm.vmovntps(ymmword_ptr(r8 + rbx), ymm0)
                    .map_err(|e| e.to_string())?;
            } else {
                self.asm.vmovups(ymmword_ptr(r8 + rbx), ymm0)
                    .map_err(|e| e.to_string())?;
            }

            self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
            self.asm.jmp(remainder_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut done_label).map_err(|e| e.to_string())?;

            if use_nt_store {
                self.asm.sfence().map_err(|e| e.to_string())?;
            }

            if tail > 0 {
                let base_bytes = (vec_count * simd_w * 4) as i32;
                for t in 0..tail as i32 {
                    let off = base_bytes + t * 4;
                    self.asm.mov(rbx, off as i64).map_err(|e| e.to_string())?;
                    self.asm.vmovss(xmm0, dword_ptr(rdi + off))
                        .map_err(|e| e.to_string())?;
                    for (body, is_binary) in &trace_info {
                        self.emit_elementwise_trace_body(ymm0, body, *is_binary, true)?;
                    }
                    self.asm.vmovss(dword_ptr(r8 + off), xmm0)
                        .map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Emit a fused elementwise chain as a single AVX-512 SIMD loop.
        ///
        /// AVX-512 counterpart of `emit_elementwise_chain`'s TraceOp path.
        /// Uses zmm0 as the live value register and masked tail load/store
        /// (`k1`) for the final partial vector.
        fn emit_elementwise_chain_avx512(
            &mut self,
            trace_info: &[(Vec<TraceOp>, bool)],
            elem_count: usize,
            profile: &DeviceProfile,
        ) -> Result<(), String> {
            let simd_w = 16usize; // 16 f32 lanes per zmm
            let vec_count = elem_count / simd_w;
            let tail = elem_count % simd_w;

            let output_bytes = elem_count * 4;
            let (_, l2_size, _) = profile.cache_sizes();
            let use_nt_store = output_bytes > l2_size;

            let has_binary = trace_info.iter().any(|(_, b)| *b);
            let prefetch_dist = 512i32;

            let unroll = 4usize;
            let unrolled_vecs = (vec_count / unroll) * unroll;
            let _remainder_vecs = vec_count - unrolled_vecs;

            let unrolled_bytes = (unrolled_vecs * simd_w * 4) as i32;
            let total_vec_bytes = (vec_count * simd_w * 4) as i32;

            let mut unrolled_loop = self.asm.create_label();
            let mut remainder_loop = self.asm.create_label();
            let mut done_label = self.asm.create_label();

            self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;

            if unrolled_vecs > 0 {
                self.asm.set_label(&mut unrolled_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, unrolled_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(remainder_loop).map_err(|e| e.to_string())?;

                self.asm.prefetcht0(byte_ptr(rdi + rbx + prefetch_dist))
                    .map_err(|e| e.to_string())?;
                if has_binary {
                    self.asm.prefetcht0(byte_ptr(rsi + rbx + prefetch_dist))
                        .map_err(|e| e.to_string())?;
                }

                for _u in 0..unroll {
                    self.asm.vmovups(zmm0, zmmword_ptr(rdi + rbx))
                        .map_err(|e| e.to_string())?;

                    for (body, is_binary) in trace_info {
                        self.emit_elementwise_trace_body_avx512(zmm0, body, *is_binary, false)?;
                    }

                    if use_nt_store {
                        self.asm.vmovntps(zmmword_ptr(r8 + rbx), zmm0)
                            .map_err(|e| e.to_string())?;
                    } else {
                        self.asm.vmovups(zmmword_ptr(r8 + rbx), zmm0)
                            .map_err(|e| e.to_string())?;
                    }

                    self.asm.add(rbx, 64i32).map_err(|e| e.to_string())?;
                }

                self.asm.jmp(unrolled_loop).map_err(|e| e.to_string())?;
            }

            self.asm.set_label(&mut remainder_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
            self.asm.jge(done_label).map_err(|e| e.to_string())?;

            self.asm.prefetcht0(byte_ptr(rdi + rbx + prefetch_dist))
                .map_err(|e| e.to_string())?;

            self.asm.vmovups(zmm0, zmmword_ptr(rdi + rbx))
                .map_err(|e| e.to_string())?;

            for (body, is_binary) in trace_info {
                self.emit_elementwise_trace_body_avx512(zmm0, body, *is_binary, false)?;
            }

            if use_nt_store {
                self.asm.vmovntps(zmmword_ptr(r8 + rbx), zmm0)
                    .map_err(|e| e.to_string())?;
            } else {
                self.asm.vmovups(zmmword_ptr(r8 + rbx), zmm0)
                    .map_err(|e| e.to_string())?;
            }

            self.asm.add(rbx, 64i32).map_err(|e| e.to_string())?;
            self.asm.jmp(remainder_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut done_label).map_err(|e| e.to_string())?;

            if use_nt_store {
                self.asm.sfence().map_err(|e| e.to_string())?;
            }

            if tail > 0 {
                let base_bytes = (vec_count * simd_w * 4) as i32;
                self.emit_set_kmask(tail)?;
                self.asm.mov(rbx, base_bytes as i64).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmm0.k1().z(), zmmword_ptr(rdi + base_bytes))
                    .map_err(|e| e.to_string())?;
                for (body, is_binary) in trace_info {
                    self.emit_elementwise_trace_body_avx512(zmm0, body, *is_binary, true)?;
                }
                self.asm.vmovups(zmmword_ptr(r8 + base_bytes).k1(), zmm0)
                    .map_err(|e| e.to_string())?;
            }

            Ok(())
        }

        /// Hardcoded fallback for elementwise chain (no registry available).
        ///
        /// Uses the original OpKind-matching approach for SiLU, GELU, Add, Mul.
        fn emit_elementwise_chain_hardcoded(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            elem_count: usize,
        ) -> Result<(), String> {
            let simd_w = self.simd_width;
            let vec_count = elem_count / simd_w;
            let tail = elem_count % simd_w;

            let mut loop_label = self.asm.create_label();
            let mut done_label = self.asm.create_label();

            let total_vec_bytes = (vec_count * simd_w * 4) as i32;
            self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut loop_label).map_err(|e| e.to_string())?;
            self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
            self.asm.jge(done_label).map_err(|e| e.to_string())?;

            self.asm.vmovups(ymm0, ymmword_ptr(rdi + rbx))
                .map_err(|e| e.to_string())?;

            let result_reg = self.emit_chain_body(group, graph)?;

            self.asm.vmovups(ymmword_ptr(r8 + rbx), result_reg)
                .map_err(|e| e.to_string())?;

            self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
            self.asm.jmp(loop_label).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut done_label).map_err(|e| e.to_string())?;

            if tail > 0 {
                let base_offset = (vec_count * simd_w) as i32;
                for t in 0..tail as i32 {
                    let off = (base_offset + t) * 4;
                    self.asm.vmovss(xmm0, dword_ptr(rdi + off))
                        .map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r8 + off), xmm0)
                        .map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Emit the compute body for an elementwise chain (hardcoded fallback).
        ///
        /// Input is pre-loaded in ymm0. Returns the register holding the result.
        /// Uses ymm0-ymm5 for data, ymm13-ymm15 as scratch for exp/tanh.
        fn emit_chain_body(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
        ) -> Result<AsmRegisterYmm, String> {
            let op = graph.op(group.anchor).ok_or("missing anchor")?;
            match &op.kind {
                OpKind::Silu => {
                    self.asm.vxorps(ymm1, ymm1, ymm1).map_err(|e| e.to_string())?;
                    self.asm.vsubps(ymm1, ymm1, ymm0).map_err(|e| e.to_string())?;
                    self.emit_exp_avx2(ymm2, ymm1, [ymm13, ymm14, ymm15])?;
                    let one_label = self.const_f32(1.0);
                    self.asm.vbroadcastss(ymm3, dword_ptr(one_label)).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm2, ymm2, ymm3).map_err(|e| e.to_string())?;
                    self.asm.vdivps(ymm0, ymm0, ymm2).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                OpKind::Gelu => {
                    self.asm.vmulps(ymm1, ymm0, ymm0).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm1, ymm1, ymm0).map_err(|e| e.to_string())?;
                    let coeff_label = self.const_f32(0.044715);
                    self.asm.vbroadcastss(ymm2, dword_ptr(coeff_label)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm1, ymm2, ymm1).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm1, ymm0, ymm1).map_err(|e| e.to_string())?;
                    let sqrt2pi_label = self.const_f32(0.7978845608);
                    self.asm.vbroadcastss(ymm2, dword_ptr(sqrt2pi_label)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm1, ymm2, ymm1).map_err(|e| e.to_string())?;
                    self.emit_tanh_avx2(ymm1, ymm1, [ymm13, ymm14, ymm15])?;
                    let one_label = self.const_f32(1.0);
                    self.asm.vbroadcastss(ymm2, dword_ptr(one_label)).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm1, ymm1, ymm2).map_err(|e| e.to_string())?;
                    let half_label = self.const_f32(0.5);
                    self.asm.vbroadcastss(ymm2, dword_ptr(half_label)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm0, ymm2, ymm0).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm0, ymm0, ymm1).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                OpKind::Add => {
                    self.asm.vmovups(ymm1, ymmword_ptr(rsi + rbx)).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm0, ymm0, ymm1).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                OpKind::Mul => {
                    self.asm.vmovups(ymm1, ymmword_ptr(rsi + rbx)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm0, ymm0, ymm1).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                _ => {
                    Ok(ymm0)
                }
            }
        }
        /// Emit GEMM with fused epilogue (activation injected before store).
        ///
        /// Generates the full GEMM microkernel, then applies epilogue ops
        /// (bias add, activation) on each accumulator register before storing
        /// to C. Epilogue bodies are looked up from the `ScalarOpRegistry`
        /// as `TraceOp` SSA traces, eliminating hand-coded per-OpKind logic.
        fn emit_gemm_with_epilogue(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            _alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor")?;

            let mut epilogue_bodies: Vec<&[TraceOp]> = Vec::new();
            if let Some(reg) = registry {
                for &epi_id in &group.epilogue {
                    let epi_op = graph.op(epi_id).ok_or("missing epilogue op")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            epilogue_bodies.push(body);
                        }
                    }
                }
            }
            let epi_refs: Vec<&[TraceOp]> = epilogue_bodies.iter().map(|s| *s).collect();

            match &op.kind {
                OpKind::Gemm { m, n, k } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile, &epi_refs)?;
                }
                OpKind::QuantGemm { m, n, k, block_size, bits } => {
                    self.emit_quant_gemm(*m, *n, *k, *block_size, *bits, profile, &epi_refs)?;
                }
                _ => {}
            }
            Ok(())
        }

        /// GPRs that must be saved/restored around extern "C" calls.
        ///
        /// The BLIS loop uses rdi/rsi/r8 as A/B/C base pointers, but
        /// System V calls clobber rdi/rsi/rdx/rcx/r8/r9/r10/r11 + all xmm/ymm.
        /// Callee-saved regs (rbx, r12-r15, rbp) are preserved by the callee.
        ///
        /// We save the caller-saved GPRs we care about. Note: after pushing
        /// 6 regs (48 bytes), rsp shifts by 48. Stack-local offsets must
        /// account for this via `CALL_SAVE_SIZE`.
        const CALL_SAVE_GPRS: [iced_x86::code_asm::AsmRegister64; 6] = [
            rdi, rsi, r8, r9, r10, r11,
        ];
        /// Stack displacement from the 6 pushes in save_caller_regs.
        const CALL_SAVE_SIZE: i32 = 48;

        /// Push caller-saved GPRs before an extern call.
        ///
        /// After this, stack locals that were at `[rsp + X]` are now at
        /// `[rsp + X + CALL_SAVE_SIZE]`.
        fn emit_save_caller_regs(&mut self) -> Result<(), String> {
            for &reg in Self::CALL_SAVE_GPRS.iter() {
                self.asm.push(reg).map_err(|e| e.to_string())?;
            }
            Ok(())
        }

        /// Pop caller-saved GPRs after an extern call (reverse order).
        fn emit_restore_caller_regs(&mut self) -> Result<(), String> {
            for &reg in Self::CALL_SAVE_GPRS.iter().rev() {
                self.asm.pop(reg).map_err(|e| e.to_string())?;
            }
            Ok(())
        }

        /// Emit `call rax` with the given function pointer loaded into rax.
        fn emit_call_rax(&mut self, fn_ptr: u64) -> Result<(), String> {
            self.asm.mov(rax, fn_ptr).map_err(|e| e.to_string())?;
            self.asm.call(rax).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit `reg = min(cap, total - counter)`.
        ///
        /// Computes `reg = total - counter_reg`, then clamps to `cap`.
        /// Used for runtime tail handling: `kc_cur = min(KC, k - pc)`.
        fn emit_runtime_min(
            &mut self,
            dst: iced_x86::code_asm::AsmRegister64,
            total: usize,
            cap: usize,
            counter: iced_x86::code_asm::AsmRegister64,
        ) -> Result<(), String> {
            self.asm.mov(dst, total as u64).map_err(|e| e.to_string())?;
            self.asm.sub(dst, counter).map_err(|e| e.to_string())?;
            self.asm.cmp(dst, cap as i32).map_err(|e| e.to_string())?;
            let mut skip = self.asm.create_label();
            self.asm.jle(skip).map_err(|e| e.to_string())?;
            self.asm.mov(dst, cap as u64).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut skip).map_err(|e| e.to_string())?;
            Ok(())
        }

        // ── JIT-inlined pack_b ──────────────────────────────────────
        //
        // Replaces the external `gllm_pack_b_f32` call with inline JIT code.
        //
        // Pack B from row-major [kc_cur, nc_cur] into panel-major layout:
        //   For each NR-wide column strip j (0..nc_cur step NR):
        //     For each row k (0..kc_cur):
        //       Copy NR floats from B[k, j..j+NR] to packed_b contiguously.
        //   Remainder strip (nc_cur % NR): zero-padded to NR width.
        //
        // On entry:
        //   r10 = B source base ptr (B + pc*ldb + jc, byte address)
        //   r11 = packed_b destination base ptr
        //   [rsp + loc_kc] = kc_cur
        //   [rsp + loc_nc] = nc_cur
        //
        // Clobbers: rax, rcx, rdx, r9, r10, r11, r15, ymm0, ymm1
        // Preserves: rdi, rsi, r8, r12, r13, r14, rbx, rbp
        fn emit_jit_pack_b_avx2(
            &mut self,
            ldb: usize,
            nr: usize,
            pack_b_off: i32,
            loc_kc: i32,
            loc_nc: i32,
        ) -> Result<(), String> {
            use iced_x86::code_asm::*;

            let ldb_bytes = (ldb * 4) as i32;
            let nr_bytes = (nr * 4) as i32;

            // ── j loop: iterate over full NR-wide strips ──
            let mut j_loop = self.asm.create_label();
            let mut j_rem = self.asm.create_label();
            let mut j_end = self.asm.create_label();

            // r9 = j (column offset in elements)
            self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
            // r15 = dst pointer (advances through packed_b)
            self.asm.mov(r15, r11).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut j_loop).map_err(|e| e.to_string())?;

            // if j + NR > nc_cur, go to remainder
            self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
            self.asm.add(rax, nr as i32).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
            self.asm.jg(j_rem).map_err(|e| e.to_string())?;

            // ── k loop for full NR panel: vectorized copy ──
            {
                let mut k_loop = self.asm.create_label();
                let mut k_done = self.asm.create_label();

                // rdx = B source for this strip = r10 + j * 4
                self.asm.lea(rdx, qword_ptr(r10 + r9 * 4)).map_err(|e| e.to_string())?;
                // rcx = k counter
                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(k_done).map_err(|e| e.to_string())?;

                // Load NR=16 floats (2 ymm) from B[k, j..j+16]
                self.asm.vmovups(ymm0, ymmword_ptr(rdx)).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymm1, ymmword_ptr(rdx + 32)).map_err(|e| e.to_string())?;

                // Store to packed_b
                self.asm.vmovups(ymmword_ptr(r15), ymm0).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(r15 + 32), ymm1).map_err(|e| e.to_string())?;

                // Advance: src += ldb * 4, dst += NR * 4
                self.asm.add(rdx, ldb_bytes).map_err(|e| e.to_string())?;
                self.asm.add(r15, nr_bytes).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;
            }

            self.asm.add(r9, nr as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(j_loop).map_err(|e| e.to_string())?;

            // ── Remainder strip: nc_cur % NR columns, zero-padded ──
            self.asm.set_label(&mut j_rem).map_err(|e| e.to_string())?;

            // rax = nc_rem = nc_cur - j
            self.asm.mov(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, 0i32).map_err(|e| e.to_string())?;
            self.asm.jle(j_end).map_err(|e| e.to_string())?;

            // Save nc_rem in r9 (reuse, no longer need j)
            self.asm.mov(r9, rax).map_err(|e| e.to_string())?;

            // rdx = B source for remainder strip = r10 + (nc_cur - nc_rem) * 4
            self.asm.mov(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(r10 + rax * 4)).map_err(|e| e.to_string())?;

            {
                let mut rk_loop = self.asm.create_label();
                let mut rk_done = self.asm.create_label();

                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(rk_done).map_err(|e| e.to_string())?;

                // Zero the destination NR floats
                self.asm.vxorps(ymm0, ymm0, ymm0).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(r15), ymm0).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(r15 + 32), ymm0).map_err(|e| e.to_string())?;

                // Scalar copy of nc_rem floats
                {
                    let mut sc_loop = self.asm.create_label();
                    let mut sc_done = self.asm.create_label();

                    self.asm.xor(rax, rax).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_loop).map_err(|e| e.to_string())?;
                    self.asm.cmp(rax, r9).map_err(|e| e.to_string())?;
                    self.asm.jge(sc_done).map_err(|e| e.to_string())?;

                    self.asm.vmovss(xmm0, dword_ptr(rdx + rax * 4)).map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r15 + rax * 4), xmm0).map_err(|e| e.to_string())?;

                    self.asm.inc(rax).map_err(|e| e.to_string())?;
                    self.asm.jmp(sc_loop).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_done).map_err(|e| e.to_string())?;
                }

                self.asm.add(rdx, ldb_bytes).map_err(|e| e.to_string())?;
                self.asm.add(r15, nr_bytes).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(rk_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_done).map_err(|e| e.to_string())?;
            }

            // nop separates rk_done from j_end so iced-x86 doesn't see two labels on one insn
            self.asm.nop().map_err(|e| e.to_string())?;
            self.asm.set_label(&mut j_end).map_err(|e| e.to_string())?;

            Ok(())
        }

        // ── JIT-inlined pack_a ──────────────────────────────────────
        //
        // Replaces the external `gllm_pack_a_f32` call with inline JIT code.
        //
        // Pack A from row-major [mc_cur, kc_cur] into panel-major layout:
        //   For each MR-tall row strip i (0..mc_cur step MR):
        //     For each column k (0..kc_cur):
        //       Copy MR floats from A[i..i+MR, k] (column-gather, stride=lda)
        //       to packed_a contiguously.
        //   Remainder strip (mc_cur % MR): zero-padded to MR height.
        //
        // On entry:
        //   r15 = A source base ptr (A + ic*lda + pc, byte address)
        //   rbx = scratchpad base
        //   [rsp + loc_kc] = kc_cur
        //   [rsp + loc_mc] = mc_cur
        //
        // Clobbers: rax, rcx, rdx, r9, r10, r11, r15, ymm0..ymm5
        // Preserves: rdi, rsi, r8, r12, r13, r14, rbx, rbp
        fn emit_jit_pack_a_avx2(
            &mut self,
            lda: usize,
            mr: usize,
            pack_a_off: i32,
            loc_kc: i32,
            loc_mc: i32,
        ) -> Result<(), String> {
            use iced_x86::code_asm::*;

            let lda_bytes = (lda * 4) as i32;

            // ── i loop: iterate over full MR-tall strips ──
            let mut i_loop = self.asm.create_label();
            let mut i_rem = self.asm.create_label();
            let mut i_end = self.asm.create_label();

            // r9 = i (row offset in elements)
            self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
            // r11 = dst pointer (advances through packed_a)
            self.asm.lea(r11, qword_ptr(rbx + pack_a_off)).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut i_loop).map_err(|e| e.to_string())?;

            // if i + MR > mc_cur, go to remainder
            self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
            self.asm.add(rax, mr as i32).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.jg(i_rem).map_err(|e| e.to_string())?;

            // ── k loop for full MR panel: gather MR rows ──
            {
                let mut k_loop = self.asm.create_label();
                let mut k_done = self.asm.create_label();

                // r10 = A source for this strip = r15 + i * lda * 4
                self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, lda_bytes).map_err(|e| e.to_string())?;
                self.asm.lea(r10, qword_ptr(r15 + rax)).map_err(|e| e.to_string())?;

                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(k_done).map_err(|e| e.to_string())?;

                // Gather MR=6 floats from column k of rows i..i+MR
                // A[i+0, k], A[i+1, k], ..., A[i+5, k]
                // Each row is lda_bytes apart
                // Use scalar loads since rows are non-contiguous
                for row in 0..mr {
                    let row_off = (row as i32) * lda_bytes;
                    self.asm.vmovss(xmm0, dword_ptr(r10 + row_off)).map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r11 + (row * 4) as i32), xmm0).map_err(|e| e.to_string())?;
                }

                // Advance: src += 4 (next column), dst += MR * 4
                self.asm.add(r10, 4i32).map_err(|e| e.to_string())?;
                self.asm.add(r11, (mr * 4) as i32).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;
            }

            self.asm.add(r9, mr as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(i_loop).map_err(|e| e.to_string())?;

            // ── Remainder strip: mc_cur % MR rows, zero-padded ──
            self.asm.set_label(&mut i_rem).map_err(|e| e.to_string())?;

            // rax = mc_rem = mc_cur - i
            self.asm.mov(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, 0i32).map_err(|e| e.to_string())?;
            self.asm.jle(i_end).map_err(|e| e.to_string())?;

            // Save mc_rem in r9 (reuse)
            self.asm.mov(r9, rax).map_err(|e| e.to_string())?;

            // r10 = A source for remainder = r15 + (mc_cur - mc_rem) * lda * 4
            self.asm.mov(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, lda_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(r10, qword_ptr(r15 + rax)).map_err(|e| e.to_string())?;

            {
                let mut rk_loop = self.asm.create_label();
                let mut rk_done = self.asm.create_label();

                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(rk_done).map_err(|e| e.to_string())?;

                // Zero the destination MR floats
                self.asm.vxorps(ymm0, ymm0, ymm0).map_err(|e| e.to_string())?;
                // Zero MR floats (MR=6 → 24 bytes, use 1 ymm + partial)
                // Just zero all MR slots individually for simplicity
                for row in 0..mr {
                    self.asm.vmovss(dword_ptr(r11 + (row * 4) as i32), xmm0).map_err(|e| e.to_string())?;
                }

                // Scalar copy of mc_rem rows
                {
                    let mut sc_loop = self.asm.create_label();
                    let mut sc_done = self.asm.create_label();

                    self.asm.xor(rax, rax).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_loop).map_err(|e| e.to_string())?;
                    self.asm.cmp(rax, r9).map_err(|e| e.to_string())?;
                    self.asm.jge(sc_done).map_err(|e| e.to_string())?;

                    // Load A[rem_row, k] = r10 + rax * lda_bytes
                    self.asm.mov(rdx, rax).map_err(|e| e.to_string())?;
                    self.asm.imul_3(rdx, rdx, lda_bytes).map_err(|e| e.to_string())?;
                    self.asm.vmovss(xmm1, dword_ptr(r10 + rdx)).map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r11 + rax * 4), xmm1).map_err(|e| e.to_string())?;

                    self.asm.inc(rax).map_err(|e| e.to_string())?;
                    self.asm.jmp(sc_loop).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_done).map_err(|e| e.to_string())?;
                }

                self.asm.add(r10, 4i32).map_err(|e| e.to_string())?;
                self.asm.add(r11, (mr * 4) as i32).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(rk_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_done).map_err(|e| e.to_string())?;
            }

            // nop separates rk_done from i_end so iced-x86 doesn't see two labels on one insn
            self.asm.nop().map_err(|e| e.to_string())?;
            self.asm.set_label(&mut i_end).map_err(|e| e.to_string())?;

            Ok(())
        }


        /// Emit a GEMM microkernel using the BLIS 5-level blocking scheme.
        ///
        /// For small matrices that fit within a single block (k ≤ kc, m ≤ mc,
        /// n ≤ nc), falls back to the direct unpacked tile loop (no packing
        /// overhead). Otherwise emits the full BLIS loop nest:
        ///
        /// ```text
        /// Loop 5 (jc over N, step NC):
        ///   Loop 4 (pc over K, step KC):
        ///     pack_b: B[pc..pc+kc, jc..jc+nc] → packed_b
        ///     Loop 3 (ic over M, step MC):
        ///       pack_a: A[ic..ic+mc, pc..pc+kc] → packed_a
        ///       Loop 2 (jr over NC, step NR):
        ///         Loop 1 (ir over MC, step MR):
        ///           emit_gemm_tile_packed(MR×NR, kc)
        /// ```
        ///
        /// Scratchpad layout (from [rbp+32]):
        ///   [0 .. mc*kc*4)           = packed_a buffer
        ///   [mc*kc*4 .. +kc*nc*4)    = packed_b buffer
        ///
        /// GPR plan (BLIS path):
        /// - rdi=A base, rsi=B base, r8=C base (ABI args, saved across calls)
        /// - rbx=scratchpad base (from [rbp+32])
        /// - r12=jc (NC counter), r13=pc (KC counter), r14=ic (MC counter)
        /// - r15=packed_a ptr, r11=packed_b ptr, r10=C tile ptr (inner loops)
        /// - r9=jr (NR counter), rcx=K counter (tile), rdx=packed_b ptr (tile)
        ///
        /// Stack locals (allocated via sub rsp, 32):
        /// - [rsp+0]  = nc_cur (runtime min(nc, n-jc))
        /// - [rsp+8]  = kc_cur (runtime min(kc, k-pc))
        /// - [rsp+16] = mc_cur (runtime min(mc, m-ic))
        /// - [rsp+24] = (reserved / alignment)
        /// Check whether any epilogue body references `Input(idx)` with idx > 0,
        /// indicating an external tensor (e.g. bias vector) that must be loaded
        /// from the rdx pointer saved at `[rbp-48]`.
        fn epilogue_has_external_input(bodies: &[&[TraceOp]]) -> bool {
            bodies.iter().any(|body| {
                body.iter().any(|op| matches!(op, TraceOp::Input(i) if *i > 0))
            })
        }

        pub fn emit_gemm_microkernel(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let blocking = profile.gemm_blocking(m, n, k);
            let mr = blocking.mr;
            let nr = blocking.nr;
            let kc = blocking.kc;
            let mc = blocking.mc;
            let nc = blocking.nc;
            let simd_w = 8usize; // AVX2: 8 f32 per ymm
            let nr_vecs = nr / simd_w;
            let num_acc = mr * nr_vecs;

            if num_acc + 1 > 13 {
                return Err(format!(
                    "GEMM {}x{} microkernel needs {} accumulators + 1 scratch, \
                     exceeds 13 usable ymm (ymm13-15 reserved)",
                    mr, nr, num_acc
                ));
            }

            // If epilogue references Input(1) (external tensor like bias),
            // save rdx (3rd ABI arg) to [rbp-48] so the epilogue can load from it.
            if Self::epilogue_has_external_input(epilogue_bodies) {
                self.asm.mov(qword_ptr(rbp - 48), rdx).map_err(|e| e.to_string())?;
                self.bias_saved = true;
            }

            if m == 1 {
                eprintln!("GEMV 1x{}x{}: direct path (mr=1, nr={})", n, k, nr);
                return self.emit_gemm_microkernel_direct(1, n, k, 1, nr, nr_vecs, simd_w, epilogue_bodies);
            }

            eprintln!("GEMM {}x{}x{}: mr={} nr={} kc={} mc={} nc={} → {}", m, n, k, mr, nr, kc, mc, nc,
                if k <= kc && m <= mc && n <= nc { "DIRECT" } else { "BLIS" });
            if k <= kc && m <= mc && n <= nc {
                return self.emit_gemm_microkernel_direct(m, n, k, mr, nr, nr_vecs, simd_w, epilogue_bodies);
            }

            self.emit_gemm_blis(m, n, k, mr, nr, nr_vecs, kc, mc, nc, epilogue_bodies)?;

            Ok(())
        }

        /// Emit the BLIS 5-level loop nest for large GEMMs.
        ///
        /// Separated from `emit_gemm_microkernel` for clarity. All blocking
        /// parameters are compile-time constants; runtime min() is emitted
        /// for tail handling.
        ///
        /// Stack locals layout (32 bytes, allocated via `sub rsp, 32`):
        /// - `[rsp+0]`  = nc_cur
        /// - `[rsp+8]`  = kc_cur
        /// - `[rsp+16]` = mc_cur
        /// - `[rsp+24]` = first_kc (1 if pc==0, 0 otherwise)
        ///
        /// After `emit_save_caller_regs` (6 pushes = 48 bytes), these shift:
        /// - `[rsp+48]` = nc_cur, `[rsp+56]` = kc_cur, etc.
        fn emit_gemm_blis(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            mr: usize,
            nr: usize,
            nr_vecs: usize,
            kc: usize,
            mc: usize,
            nc: usize,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let c_row_bytes = (n * 4) as i32;
            let lda = k;  // A is row-major [m, k], stride = k
            let ldb = n;  // B is row-major [k, n], stride = n

            let pack_a_panels = (mc + mr - 1) / mr;
            let pack_a_bytes = pack_a_panels * mr * kc * 4;
            let pack_b_panels = (nc + nr - 1) / nr;
            let pack_b_bytes = pack_b_panels * nr * kc * 4;
            let blis_bytes = pack_a_bytes + pack_b_bytes;
            let total_extra = (self.blis_scratchpad_offset - self.blis_base_offset) + blis_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(total_extra);

            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            let pack_a_off = self.blis_scratchpad_offset as i32;
            let pack_b_off = (self.blis_scratchpad_offset + pack_a_bytes) as i32;

            self.asm.sub(rsp, 32i32).map_err(|e| e.to_string())?;

            let loc_nc: i32 = 0;
            let loc_kc: i32 = 8;
            let loc_mc: i32 = 16;

            let mut jc_loop = self.asm.create_label();
            let mut jc_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, n as i32).map_err(|e| e.to_string())?;
            self.asm.jge(jc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, n, nc, r12)?;
            self.asm.mov(qword_ptr(rsp + loc_nc), rax).map_err(|e| e.to_string())?;

            let mut pc_loop = self.asm.create_label();
            let mut pc_done = self.asm.create_label();

            self.asm.xor(r13, r13).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13, k as i32).map_err(|e| e.to_string())?;
            self.asm.jge(pc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, k, kc, r13)?;
            self.asm.mov(qword_ptr(rsp + loc_kc), rax).map_err(|e| e.to_string())?;

            // ── JIT pack_b: B[pc..pc+kc, jc..jc+nc] → packed_b ──
            {
                // r10 = &B[pc, jc] = rsi + (r13 * ldb + r12) * 4
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, ldb as i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, r12).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, rsi).map_err(|e| e.to_string())?;
                self.asm.mov(r10, rax).map_err(|e| e.to_string())?;

                // r11 = packed_b destination
                self.asm.lea(r11, qword_ptr(rbx + pack_b_off)).map_err(|e| e.to_string())?;

                self.emit_jit_pack_b_avx2(ldb, nr, pack_b_off, loc_kc, loc_nc)?;
            }

            let mut ic_loop = self.asm.create_label();
            let mut ic_done = self.asm.create_label();

            self.asm.xor(r14, r14).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r14, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(ic_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, m, mc, r14)?;
            self.asm.mov(qword_ptr(rsp + loc_mc), rax).map_err(|e| e.to_string())?;

            // ── JIT pack_a: A[ic..ic+mc, pc..pc+kc] → packed_a ──
            {
                // r15 = &A[ic, pc] = rdi + (r14 * lda + r13) * 4
                self.asm.mov(rax, r14).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, lda as i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, r13).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, rdi).map_err(|e| e.to_string())?;
                self.asm.mov(r15, rax).map_err(|e| e.to_string())?;

                self.emit_jit_pack_a_avx2(lda, mr, pack_a_off, loc_kc, loc_mc)?;
            }

            {
                let m_rem = m % mr;
                let n_rem = n % nr;
                let simd_w = nr / nr_vecs;  // 8 for AVX2
                let n_rem_vecs = n_rem / simd_w;  // 0 if n % nr == 0

                let loc_ir: i32 = 24;

                let mut jr_loop = self.asm.create_label();
                let mut jr_done = self.asm.create_label();
                let mut jr_rem_start = self.asm.create_label();

                self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_loop).map_err(|e| e.to_string())?;

                self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
                self.asm.add(rax, nr as i32).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                self.asm.jg(jr_rem_start).map_err(|e| e.to_string())?;

                self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                if pack_b_off != 0 {
                    self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                }

                self.emit_blis_ir_loop(
                    mr, nr_vecs, m_rem, loc_kc, n, nr_vecs, nr,
                    loc_ir, loc_mc, pack_a_off, c_row_bytes,
                    epilogue_bodies, k,
                )?;

                self.asm.add(r9, nr as i32).map_err(|e| e.to_string())?;
                self.asm.jmp(jr_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut jr_rem_start).map_err(|e| e.to_string())?;

                if n_rem_vecs > 0 {
                    self.asm.cmp(r9, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                    self.asm.jge(jr_done).map_err(|e| e.to_string())?;

                    self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                    self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                    self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                    self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                    if pack_b_off != 0 {
                        self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                    }

                    self.emit_blis_ir_loop(
                        mr, nr_vecs, m_rem, loc_kc, n, n_rem_vecs, nr,
                        loc_ir, loc_mc, pack_a_off, c_row_bytes,
                        epilogue_bodies, k,
                    )?;
                }

                self.asm.nop().map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_done).map_err(|e| e.to_string())?;
            }

            self.asm.add(r14, mc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(ic_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_done).map_err(|e| e.to_string())?;

            self.asm.add(r13, kc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(pc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_done).map_err(|e| e.to_string())?;

            self.asm.add(r12, nc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(jc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_done).map_err(|e| e.to_string())?;

            self.asm.add(rsp, 32i32).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit the ir (M-dimension) loop for one jr strip in the BLIS path.
        ///
        /// Iterates ir = 0..mc_cur (step MR), emitting full MR tiles, then
        /// one M-remainder tile if `m_rem != 0`.
        ///
        /// Expects on entry:
        ///   r11 = packed_b ptr for this jr strip
        ///   r9  = jr (current N offset within nc_cur)
        ///   r12 = jc, r13 = pc, r14 = ic
        ///   r8  = C base ptr
        ///   rbx = scratchpad base
        ///   [rsp + loc_mc] = mc_cur
        ///   [rsp + loc_nc] = nc_cur (unused here but on stack)
        ///
        /// `tile_nr_vecs`: number of ymm vectors per row for this jr strip
        ///   (nr_vecs for full tiles, n_rem_vecs for the N remainder strip)
        fn emit_blis_ir_loop(
            &mut self,
            mr: usize,
            nr_vecs: usize,       // full NR in vectors (for packed_a panel width)
            m_rem: usize,         // m % mr, 0 if no M remainder
            loc_kc: i32,
            n: usize,             // full N dimension (for C row stride)
            tile_nr_vecs: usize,  // actual nr_vecs for this strip's tiles
            nr: usize,            // full NR (packed_b stride per K step)
            loc_ir: i32,
            loc_mc: i32,
            pack_a_off: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
            k_total: usize,
        ) -> Result<(), String> {
            let mut ir_loop = self.asm.create_label();
            let mut ir_done = self.asm.create_label();
            let mut ir_rem_start = self.asm.create_label();

            self.asm.mov(qword_ptr(rsp + loc_ir), 0i32).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ir_loop).map_err(|e| e.to_string())?;

            self.asm.mov(rax, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
            self.asm.mov(rcx, rax).map_err(|e| e.to_string())?;
            self.asm.add(rcx, mr as i32).map_err(|e| e.to_string())?;
            self.asm.cmp(rcx, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.jg(ir_rem_start).map_err(|e| e.to_string())?;

            self.emit_blis_tile_setup(mr, loc_kc, n, pack_a_off)?;

            self.emit_blis_tile_with_first_kc_branch(mr, tile_nr_vecs, loc_kc, c_row_bytes, mr, nr, epilogue_bodies, k_total)?;

            self.asm.mov(rax, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
            self.asm.add(rax, mr as i32).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp + loc_ir), rax).map_err(|e| e.to_string())?;
            self.asm.jmp(ir_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut ir_rem_start).map_err(|e| e.to_string())?;

            if m_rem > 0 {
                self.asm.mov(rax, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
                self.asm.jge(ir_done).map_err(|e| e.to_string())?;

                self.emit_blis_tile_setup(mr, loc_kc, n, pack_a_off)?;

                self.emit_blis_tile_with_first_kc_branch(m_rem, tile_nr_vecs, loc_kc, c_row_bytes, mr, nr, epilogue_bodies, k_total)?;
            }

            self.asm.nop().map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ir_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Set up r15 (packed_a ptr) and r10 (C tile ptr) for a BLIS tile.
        ///
        /// Expects rax = ir, r14 = ic, r12 = jc, r9 = jr, r8 = C base,
        /// rbx = scratchpad base.
        fn emit_blis_tile_setup(
            &mut self,
            _mr: usize,
            loc_kc: i32,
            n: usize,
            pack_a_off: i32,
        ) -> Result<(), String> {
            self.asm.mov(r15, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
            self.asm.shl(r15, 2i32).map_err(|e| e.to_string())?;
            self.asm.imul_2(r15, rax).map_err(|e| e.to_string())?;
            self.asm.add(r15, rbx).map_err(|e| e.to_string())?;
            if pack_a_off != 0 {
                self.asm.add(r15, pack_a_off).map_err(|e| e.to_string())?;
            }

            self.asm.mov(r10, r14).map_err(|e| e.to_string())?;
            self.asm.add(r10, rax).map_err(|e| e.to_string())?;  // ic + ir
            self.asm.imul_3(r10, r10, (n * 4) as i32).map_err(|e| e.to_string())?;
            self.asm.mov(rcx, r12).map_err(|e| e.to_string())?;
            self.asm.add(rcx, r9).map_err(|e| e.to_string())?;
            self.asm.shl(rcx, 2i32).map_err(|e| e.to_string())?;
            self.asm.add(r10, rcx).map_err(|e| e.to_string())?;
            self.asm.add(r10, r8).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit a BLIS tile with runtime first_kc_block branching.
        ///
        /// Branches on r13 (pc): if r13 == 0, zero accumulators; else load from C.
        fn emit_blis_tile_with_first_kc_branch(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            loc_kc: i32,
            c_row_bytes: i32,
            packed_mr: usize,
            packed_nr: usize,
            epilogue_bodies: &[&[TraceOp]],
            k_total: usize,
        ) -> Result<(), String> {
            let mut not_first = self.asm.create_label();
            let mut tile_done = self.asm.create_label();

            self.asm.test(r13, r13).map_err(|e| e.to_string())?;
            self.asm.jnz(not_first).map_err(|e| e.to_string())?;

            self.emit_gemm_tile_packed(tile_mr, tile_nr_vecs, loc_kc, c_row_bytes, true, packed_mr, packed_nr, epilogue_bodies, k_total)?;
            self.asm.jmp(tile_done).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut not_first).map_err(|e| e.to_string())?;
            self.emit_gemm_tile_packed(tile_mr, tile_nr_vecs, loc_kc, c_row_bytes, false, packed_mr, packed_nr, epilogue_bodies, k_total)?;

            self.asm.set_label(&mut tile_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit the direct (unpacked) GEMM microkernel for small matrices.
        ///
        /// Used when the entire problem fits in a single MC×KC×NC block,
        /// so packing overhead is not worthwhile. Generates a two-level
        /// tile loop (N-outer, M-inner) reading A and B directly.
        ///
        /// GPR plan (same as original microkernel):
        /// - rdi=A, rsi=B, r8=C (from calling convention)
        /// - r15=A_row_ptr, r11=B_col_ptr, r10=C_tile_ptr
        /// - r14=j (N counter), r13=i (M counter), r12=p (K counter)
        /// - r9=B_k_ptr (advances each K step)
        fn emit_gemm_microkernel_direct(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            mr: usize,
            nr: usize,
            nr_vecs: usize,
            simd_w: usize,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let a_row_bytes = (k * 4) as i32;
            let b_row_bytes = (n * 4) as i32;
            let c_row_bytes = (n * 4) as i32;

            let m_full = (m / mr) * mr;
            let m_rem = m % mr;
            let n_full = (n / nr) * nr;
            let n_rem = n % nr;
            let n_rem_vecs = n_rem / simd_w;

            if n_full > 0 {
                let mut n_loop = self.asm.create_label();
                let mut n_done = self.asm.create_label();

                self.asm.xor(r14, r14).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut n_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(r14, n_full as i32).map_err(|e| e.to_string())?;
                self.asm.jge(n_done).map_err(|e| e.to_string())?;

                self.asm.mov(r11, r14).map_err(|e| e.to_string())?;
                self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(r11, rsi).map_err(|e| e.to_string())?;

                if m_full > 0 {
                    self.emit_gemm_m_loop(
                        mr, nr_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }
                if m_rem > 0 {
                    self.emit_gemm_m_remainder(
                        m_rem, nr_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }

                self.asm.add(r14, nr as i32).map_err(|e| e.to_string())?;
                self.asm.jmp(n_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut n_done).map_err(|e| e.to_string())?;
            }

            if n_rem_vecs > 0 {
                self.asm.mov(r11, rsi).map_err(|e| e.to_string())?;
                if n_full > 0 {
                    self.asm.add(r11, (n_full * 4) as i32).map_err(|e| e.to_string())?;
                }
                self.asm.mov(r14, n_full as i64).map_err(|e| e.to_string())?;

                if m_full > 0 {
                    self.emit_gemm_m_loop(
                        mr, n_rem_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }
                if m_rem > 0 {
                    self.emit_gemm_m_remainder(
                        m_rem, n_rem_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }
            }

            Ok(())
        }

        /// Emit the M-dimension loop over full MR-sized tiles.
        ///
        /// Expects r11 = B_col_ptr, r14 = j (N offset in elements).
        /// Uses r15 for A_row_ptr, r10 for C_tile_ptr, r13 for M counter.
        fn emit_gemm_m_loop(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            k: usize,
            m_limit: usize,
            a_row_bytes: i32,
            b_row_bytes: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let mut m_loop = self.asm.create_label();
            let mut m_done = self.asm.create_label();

            self.asm.mov(r15, rdi).map_err(|e| e.to_string())?;
            self.asm.mov(r10, r14).map_err(|e| e.to_string())?;
            self.asm.shl(r10, 2i32).map_err(|e| e.to_string())?;
            self.asm.add(r10, r8).map_err(|e| e.to_string())?;
            self.asm.xor(r13, r13).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut m_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13, m_limit as i32).map_err(|e| e.to_string())?;
            self.asm.jge(m_done).map_err(|e| e.to_string())?;

            self.emit_gemm_tile(tile_mr, tile_nr_vecs, k, a_row_bytes, b_row_bytes, c_row_bytes, epilogue_bodies)?;

            let mr_a_advance = (tile_mr as i32) * a_row_bytes;
            let mr_c_advance = (tile_mr as i32) * c_row_bytes;
            self.asm.add(r15, mr_a_advance).map_err(|e| e.to_string())?;
            self.asm.add(r10, mr_c_advance).map_err(|e| e.to_string())?;
            self.asm.add(r13, tile_mr as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(m_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut m_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit a single M-remainder tile (fewer than MR rows).
        ///
        /// Sets up r15/r10 for the remainder starting row and emits one tile.
        fn emit_gemm_m_remainder(
            &mut self,
            rem_mr: usize,
            tile_nr_vecs: usize,
            k: usize,
            m_full: usize,
            a_row_bytes: i32,
            b_row_bytes: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            self.asm.mov(r15, rdi).map_err(|e| e.to_string())?;
            if m_full > 0 {
                self.asm.add(r15, (m_full as i32) * a_row_bytes).map_err(|e| e.to_string())?;
            }
            self.asm.mov(r10, r14).map_err(|e| e.to_string())?;
            self.asm.shl(r10, 2i32).map_err(|e| e.to_string())?;
            self.asm.add(r10, r8).map_err(|e| e.to_string())?;
            if m_full > 0 {
                self.asm.add(r10, (m_full as i32) * c_row_bytes).map_err(|e| e.to_string())?;
            }

            self.emit_gemm_tile(rem_mr, tile_nr_vecs, k, a_row_bytes, b_row_bytes, c_row_bytes, epilogue_bodies)
        }

        /// Emit a single MR×NR GEMM tile: zero accumulators, K-loop with
        /// broadcast-FMA, store results.
        ///
        /// Inputs (set by caller):
        /// - r15 = &A[tile_row, 0]
        /// - r11 = &B[0, tile_col]
        /// - r10 = &C[tile_row, tile_col]
        ///
        /// Clobbers: r9 (B_k_ptr), r12 (K counter), ymm0..ymm(num_acc),
        ///           two ymm scratch registers at indices `num_acc` and `num_acc+1`
        ///           (A broadcast interleaving).
        fn emit_gemm_tile(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            k: usize,
            a_row_bytes: i32,
            b_row_bytes: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let num_acc = tile_mr * tile_nr_vecs;
            if num_acc + 2 > 16 {
                return Err(format!(
                    "GEMM tile {}x{} needs {} accumulators + 2 scratch, exceeds 16 ymm registers",
                    tile_mr, tile_nr_vecs, num_acc
                ));
            }
            let a_scratch_0 = Self::ymm_any(num_acc)?;
            let a_scratch_1 = Self::ymm_any(num_acc + 1)?;

            for a in 0..num_acc {
                let reg = Self::ymm_for_index(a)?;
                self.asm.vxorps(reg, reg, reg).map_err(|e| e.to_string())?;
            }

            let mut k_loop = self.asm.create_label();
            let mut k_done = self.asm.create_label();

            self.asm.mov(r9, r11).map_err(|e| e.to_string())?;
            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, k as i32).map_err(|e| e.to_string())?;
            self.asm.jge(k_done).map_err(|e| e.to_string())?;

            self.asm.prefetcht0(byte_ptr(r9 + b_row_bytes)).map_err(|e| e.to_string())?;
            if b_row_bytes > 64 {
                self.asm.prefetcht0(byte_ptr(r9 + b_row_bytes + 64)).map_err(|e| e.to_string())?;
            }

            let a_disp_0 = 0i32 * a_row_bytes;
            self.asm.vbroadcastss(
                a_scratch_0,
                dword_ptr(r15 + r12 * 4 + a_disp_0),
            ).map_err(|e| e.to_string())?;

            for ii in 0..tile_mr {
                let cur_scratch = if ii % 2 == 0 { a_scratch_0 } else { a_scratch_1 };
                let nxt_scratch = if ii % 2 == 0 { a_scratch_1 } else { a_scratch_0 };

                let acc0 = Self::ymm_for_index(ii * tile_nr_vecs)?;
                self.asm.vfmadd231ps(
                    acc0,
                    cur_scratch,
                    ymmword_ptr(r9),
                ).map_err(|e| e.to_string())?;

                if ii + 1 < tile_mr {
                    let a_disp_next = ((ii + 1) as i32) * a_row_bytes;
                    self.asm.vbroadcastss(
                        nxt_scratch,
                        dword_ptr(r15 + r12 * 4 + a_disp_next),
                    ).map_err(|e| e.to_string())?;
                }

                for v in 1..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let b_disp = (v as i32) * 32;
                    self.asm.vfmadd231ps(
                        acc,
                        cur_scratch,
                        ymmword_ptr(r9 + b_disp),
                    ).map_err(|e| e.to_string())?;
                }
            }

            self.asm.add(r9, b_row_bytes).map_err(|e| e.to_string())?;
            self.asm.inc(r12).map_err(|e| e.to_string())?;
            self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;

            // If epilogue has external input (bias), compute bias tile base
            // pointer into rax: rax = bias_ptr + r14 * 4
            // (r14 = current N offset in elements, r12/r9 are free after K-loop)
            if self.bias_saved && !epilogue_bodies.is_empty() {
                self.asm.mov(rax, qword_ptr(rbp - 48)).map_err(|e| e.to_string())?;
                self.asm.lea(rax, qword_ptr(rax + r14 * 4)).map_err(|e| e.to_string())?;
            }
            self.emit_epilogue_on_accumulators_inner(epilogue_bodies, num_acc, tile_nr_vecs)?;

            for ii in 0..tile_mr {
                for v in 0..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let c_disp = (ii as i32) * c_row_bytes + (v as i32) * 32;
                    self.asm.vmovups(
                        ymmword_ptr(r10 + c_disp),
                        acc,
                    ).map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Emit a single MR×NR GEMM tile reading from packed A/B buffers.
        ///
        /// This is the inner microkernel for the BLIS-style blocked GEMM.
        /// Packed layout:
        /// - packed_a: `[k * MR + ii]` (column-major MR-wide strips)
        /// - packed_b: `[k * NR + v*8]` (row-major NR-wide strips)
        ///
        /// Inputs (set by caller):
        /// - r15 = packed_a base for this MR strip
        /// - r11 = packed_b base for this NR strip
        /// - r10 = &C[tile_row, tile_col]
        ///
        /// Parameters:
        /// - `first_kc_block`: if true, zero accumulators; else load from C
        /// - `kc`: number of K iterations for this KC block
        ///
        /// Clobbers: r9 (B_k_ptr), r12 (K counter), ymm0..ymm(num_acc),
        ///           two ymm scratch registers at indices `num_acc` and `num_acc+1`
        ///           (A broadcast interleaving).
        fn emit_gemm_tile_packed(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            loc_kc: i32,
            c_row_bytes: i32,
            first_kc_block: bool,
            packed_mr: usize,     // MR used in packing (for A stride per K step)
            packed_nr: usize,     // NR used in packing (for B stride per K step)
            epilogue_bodies: &[&[TraceOp]],
            k_total: usize,
        ) -> Result<(), String> {
            let num_acc = tile_mr * tile_nr_vecs;
            if num_acc + 2 > 16 {
                return Err(format!(
                    "GEMM tile {}x{} needs {} accumulators + 2 scratch, exceeds 16 ymm registers",
                    tile_mr, tile_nr_vecs, num_acc
                ));
            }
            let a_scratch_0 = Self::ymm_any(num_acc)?;
            let a_scratch_1 = Self::ymm_any(num_acc + 1)?;
            let mr_stride_bytes = (packed_mr as i32) * 4; // packed_a advance per K step
            let nr_stride_bytes = (packed_nr as i32) * 4; // packed_b advance per K step

            if first_kc_block {
                for a in 0..num_acc {
                    let reg = Self::ymm_for_index(a)?;
                    self.asm.vxorps(reg, reg, reg).map_err(|e| e.to_string())?;
                }
            } else {
                for ii in 0..tile_mr {
                    for v in 0..tile_nr_vecs {
                        let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                        let c_disp = (ii as i32) * c_row_bytes + (v as i32) * 32;
                        self.asm.vmovups(acc, ymmword_ptr(r10 + c_disp))
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            let mut k_loop = self.asm.create_label();
            let mut k_done = self.asm.create_label();

            self.asm.mov(rdx, r11).map_err(|e| e.to_string())?;
            self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
            self.asm.jge(k_done).map_err(|e| e.to_string())?;

            self.asm.prefetcht0(byte_ptr(r15 + mr_stride_bytes)).map_err(|e| e.to_string())?;
            self.asm.prefetcht0(byte_ptr(rdx + nr_stride_bytes)).map_err(|e| e.to_string())?;
            if nr_stride_bytes > 64 {
                self.asm.prefetcht0(byte_ptr(rdx + nr_stride_bytes + 64)).map_err(|e| e.to_string())?;
            }

            self.asm.vbroadcastss(
                a_scratch_0,
                dword_ptr(r15),
            ).map_err(|e| e.to_string())?;

            for ii in 0..tile_mr {
                let cur_scratch = if ii % 2 == 0 { a_scratch_0 } else { a_scratch_1 };
                let nxt_scratch = if ii % 2 == 0 { a_scratch_1 } else { a_scratch_0 };

                let acc0 = Self::ymm_for_index(ii * tile_nr_vecs)?;
                self.asm.vfmadd231ps(
                    acc0,
                    cur_scratch,
                    ymmword_ptr(rdx),
                ).map_err(|e| e.to_string())?;

                if ii + 1 < tile_mr {
                    let a_disp_next = ((ii + 1) as i32) * 4;
                    self.asm.vbroadcastss(
                        nxt_scratch,
                        dword_ptr(r15 + a_disp_next),
                    ).map_err(|e| e.to_string())?;
                }

                for v in 1..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let b_disp = (v as i32) * 32;
                    self.asm.vfmadd231ps(
                        acc,
                        cur_scratch,
                        ymmword_ptr(rdx + b_disp),
                    ).map_err(|e| e.to_string())?;
                }
            }

            self.asm.add(r15, mr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rdx, nr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.inc(rcx).map_err(|e| e.to_string())?;
            self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;

            self.asm.mov(rax, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, mr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.sub(r15, rax).map_err(|e| e.to_string())?;

            if !epilogue_bodies.is_empty() {
                let mut skip_epi = self.asm.create_label();
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.add(rax, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, k_total as i32).map_err(|e| e.to_string())?;
                self.asm.jl(skip_epi).map_err(|e| e.to_string())?;
                // If epilogue has external input (bias), compute bias tile base
                // pointer into rax: rax = bias_ptr + (jc + jr) * 4
                // (rcx is free after K-loop; r12 = jc, r9 = jr are outer-loop regs)
                if self.bias_saved {
                    self.asm.mov(rax, qword_ptr(rbp - 48)).map_err(|e| e.to_string())?;
                    self.asm.mov(rcx, r12).map_err(|e| e.to_string())?;
                    self.asm.add(rcx, r9).map_err(|e| e.to_string())?;
                    self.asm.lea(rax, qword_ptr(rax + rcx * 4)).map_err(|e| e.to_string())?;
                }
                self.emit_epilogue_on_accumulators_inner(epilogue_bodies, num_acc, tile_nr_vecs)?;
                self.asm.set_label(&mut skip_epi).map_err(|e| e.to_string())?;
            }

            for ii in 0..tile_mr {
                for v in 0..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let c_disp = (ii as i32) * c_row_bytes + (v as i32) * 32;
                    self.asm.vmovups(
                        ymmword_ptr(r10 + c_disp),
                        acc,
                    ).map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Apply epilogue operations in-place on live accumulator registers
        /// ymm0..ymm(num_acc-1), driven entirely by `TraceOp` bodies.
        ///
        /// Each epilogue body is an SSA trace (from `OpTrace.pattern`).
        /// `Input(0)` in the body refers to the accumulator's current value.
        ///
        /// Strategy: for each accumulator, allocate `body.len() * 32` bytes
        /// of stack for SSA intermediates. Each SSA slot is a 32-byte ymm
        /// spill. We use ymm13-15 as scratch for compute, storing results
        /// back to stack slots. The final SSA value overwrites the accumulator.
        ///
        /// Must be called while accumulators hold GEMM results, BEFORE the
        /// store to C.
        fn emit_epilogue_on_accumulators(
            &mut self,
            epilogue_bodies: &[&[TraceOp]],
            num_acc: usize,
        ) -> Result<(), String> {
            self.emit_epilogue_on_accumulators_inner(epilogue_bodies, num_acc, 1)
        }

        /// Inner implementation that accepts `nr_vecs` so it can compute the
        /// per-vector bias byte offset for `Input(1)` loads.
        ///
        /// When `self.bias_saved` is true and a body contains `Input(1)`,
        /// the bias tile base pointer is expected in `rax` (set by the caller
        /// before invoking this method).  For accumulator index `a`, the
        /// vector index is `a % nr_vecs`, and the bias displacement is
        /// `(a % nr_vecs) * 32` bytes from `rax`.
        fn emit_epilogue_on_accumulators_inner(
            &mut self,
            epilogue_bodies: &[&[TraceOp]],
            num_acc: usize,
            nr_vecs: usize,
        ) -> Result<(), String> {
            if epilogue_bodies.is_empty() || num_acc == 0 {
                return Ok(());
            }

            for body in epilogue_bodies {
                if body.is_empty() {
                    continue;
                }
                let has_ext = body.iter().any(|op| matches!(op, TraceOp::Input(i) if *i > 0));
                if self.use_avx512 {
                    let acc_count_512 = num_acc.min(28); // zmm0-zmm27
                    for a in 0..acc_count_512 {
                        let acc = Self::zmm_for_index(a)?;
                        if has_ext && self.bias_saved {
                            let v = a % nr_vecs;
                            let bias_disp = (v as i32) * 64; // 64 bytes per zmm
                            self.emit_trace_on_accumulator_with_bias_avx512(acc, body, bias_disp)?;
                        } else {
                            self.emit_trace_on_accumulator_avx512(acc, body)?;
                        }
                    }
                } else {
                    let acc_count = num_acc.min(13); // ymm0-ymm12 max
                    for a in 0..acc_count {
                        let acc = Self::ymm_for_index(a)?;
                        if has_ext && self.bias_saved {
                            let v = a % nr_vecs;
                            let bias_disp = (v as i32) * 32;
                            self.emit_trace_on_accumulator_with_bias(acc, body, bias_disp)?;
                        } else {
                            self.emit_trace_on_accumulator(acc, body)?;
                        }
                    }
                }
            }

            Ok(())
        }

        /// Execute a TraceOp body in-place on a single accumulator register.
        ///
        /// `Input(0)` maps to the accumulator's current value.
        /// All SSA intermediates are spilled to stack slots at `[rsp + i*32]`.
        /// Uses ymm13-15 as scratch. The final SSA result is written back
        /// to `acc`.
        ///
        /// Stack layout (grows downward):
        /// ```text
        ///   [rsp + 0*32]  = SSA slot 0 (Input(0) = accumulator value)
        ///   [rsp + 1*32]  = SSA slot 1
        ///   ...
        ///   [rsp + (n-1)*32] = SSA slot n-1 (final result)
        /// ```
        fn emit_trace_on_accumulator(
            &mut self,
            acc: AsmRegisterYmm,
            body: &[TraceOp],
        ) -> Result<(), String> {
            let n = body.len();
            if n == 0 {
                return Ok(());
            }

            let frame_size = (n * 32) as i32;
            self.asm.sub(rsp, frame_size).map_err(|e| e.to_string())?;

            let slot_off = |i: u32| -> i32 { (i as i32) * 32 };

            for (i, op) in body.iter().enumerate() {
                let i32_idx = i as u32;
                match op {
                    TraceOp::Input(_) => {
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), acc)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(ymm13, dword_ptr(label))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vaddps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmulps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vdivps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*c)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymm14, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(ymm13, ymm14, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        self.asm.vxorps(ymm13, ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Abs(a) => {
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vandps(ymm13, ymm13, ymmword_ptr(abs_mask))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_exp_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sqrt(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsqrtps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrsqrtps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_tanh_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Recip(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrcpps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_log_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmaxps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vminps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            self.asm.vmovups(acc, ymmword_ptr(rsp + slot_off((n - 1) as u32)))
                .map_err(|e| e.to_string())?;

            self.asm.add(rsp, frame_size).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Like `emit_trace_on_accumulator` but handles `Input(1)` by loading
        /// from the external bias tensor.
        ///
        /// Expects `rax` to hold the bias tile base pointer (i.e.
        /// `bias_ptr + tile_col_start * 4`).  `bias_disp` is the byte
        /// displacement for this accumulator's vector within the tile
        /// (`(acc_idx % nr_vecs) * 32`).
        ///
        /// - `Input(0)` → accumulator's current value
        /// - `Input(1)` → `vmovups ymm, [rax + bias_disp]`
        /// - All other ops → identical to `emit_trace_on_accumulator`
        fn emit_trace_on_accumulator_with_bias(
            &mut self,
            acc: AsmRegisterYmm,
            body: &[TraceOp],
            bias_disp: i32,
        ) -> Result<(), String> {
            let n = body.len();
            if n == 0 {
                return Ok(());
            }

            let frame_size = (n * 32) as i32;
            self.asm.sub(rsp, frame_size).map_err(|e| e.to_string())?;

            let slot_off = |i: u32| -> i32 { (i as i32) * 32 };

            for (i, op) in body.iter().enumerate() {
                let i32_idx = i as u32;
                match op {
                    TraceOp::Input(idx) => {
                        if *idx == 0 {
                            self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), acc)
                                .map_err(|e| e.to_string())?;
                        } else {
                            // Input(1+): load from external tensor via rax + bias_disp
                            self.asm.vmovups(ymm13, ymmword_ptr(rax + bias_disp))
                                .map_err(|e| e.to_string())?;
                            self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(ymm13, dword_ptr(label))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vaddps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmulps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vdivps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*c)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymm14, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(ymm13, ymm14, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        self.asm.vxorps(ymm13, ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Abs(a) => {
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vandps(ymm13, ymm13, ymmword_ptr(abs_mask))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_exp_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sqrt(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsqrtps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrsqrtps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_tanh_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Recip(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrcpps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_log_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmaxps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vminps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            self.asm.vmovups(acc, ymmword_ptr(rsp + slot_off((n - 1) as u32)))
                .map_err(|e| e.to_string())?;

            self.asm.add(rsp, frame_size).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Apply a TraceOp body in-place on an accumulator register for
        /// elementwise chain codegen.
        ///
        /// Similar to `emit_trace_on_accumulator` but handles `Input(1)`
        /// by loading from `[rsi + rbx]` (the second input pointer at the
        /// current byte offset), enabling binary elementwise ops.
        ///
        /// - `Input(0)` → accumulator's current value
        /// - `Input(1)` → `[rsi + rbx]` (vector) or `vmovss` (scalar tail)
        /// - All compute ops → same as `emit_trace_on_accumulator`
        ///
        /// When `scalar_tail` is true, `Input(1)` uses `vmovss` (scalar load)
        /// instead of `vmovups` (vector load) to avoid reading past the buffer.
        fn emit_elementwise_trace_body(
            &mut self,
            acc: AsmRegisterYmm,
            body: &[TraceOp],
            is_binary: bool,
            scalar_tail: bool,
        ) -> Result<(), String> {
            let n = body.len();
            if n == 0 {
                return Ok(());
            }

            let frame_size = (n * 32) as i32;
            self.asm.sub(rsp, frame_size).map_err(|e| e.to_string())?;

            let slot_off = |i: u32| -> i32 { (i as i32) * 32 };

            for (i, op) in body.iter().enumerate() {
                let i32_idx = i as u32;
                match op {
                    TraceOp::Input(idx) => {
                        if *idx == 0 {
                            self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), acc)
                                .map_err(|e| e.to_string())?;
                        } else if *idx == 1 && is_binary {
                            if scalar_tail {
                                self.asm.vxorps(ymm13, ymm13, ymm13)
                                    .map_err(|e| e.to_string())?;
                                self.asm.vmovss(xmm13, dword_ptr(rsi + rbx))
                                    .map_err(|e| e.to_string())?;
                            } else {
                                self.asm.vmovups(ymm13, ymmword_ptr(rsi + rbx))
                                    .map_err(|e| e.to_string())?;
                            }
                            self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                                .map_err(|e| e.to_string())?;
                        } else {
                            self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), acc)
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(ymm13, dword_ptr(label))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vaddps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmulps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vdivps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*c)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymm14, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(ymm13, ymm14, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        self.asm.vxorps(ymm13, ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Abs(a) => {
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vandps(ymm13, ymm13, ymmword_ptr(abs_mask))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_exp_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sqrt(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsqrtps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrsqrtps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_tanh_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Recip(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrcpps(ymm13, ymm13)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_log_avx2(ymm14, ymm13, [ymm13, ymm15, acc])?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm14)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmaxps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        self.asm.vmovups(ymm13, ymmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vminps(ymm13, ymm13, ymmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(ymmword_ptr(rsp + slot_off(i32_idx)), ymm13)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            self.asm.vmovups(acc, ymmword_ptr(rsp + slot_off((n - 1) as u32)))
                .map_err(|e| e.to_string())?;

            self.asm.add(rsp, frame_size).map_err(|e| e.to_string())?;

            Ok(())
        }

        // ── AVX-512 epilogue on accumulators ──────────────────────────────

        /// AVX-512 version of `emit_epilogue_on_accumulators_inner`.
        ///
        /// Applies epilogue TraceOp bodies to zmm0..zmm(num_acc-1).
        /// Uses zmm29-31 as scratch. When `bias_saved` is true and a body
        /// contains `Input(1)`, the bias base pointer is expected in `rax`.
        fn emit_epilogue_on_accumulators_inner_avx512(
            &mut self,
            epilogue_bodies: &[&[TraceOp]],
            num_acc: usize,
            nr_vecs: usize,
        ) -> Result<(), String> {
            if epilogue_bodies.is_empty() || num_acc == 0 {
                return Ok(());
            }

            let acc_count = num_acc.min(28); // zmm0-zmm27 max

            for body in epilogue_bodies {
                if body.is_empty() {
                    continue;
                }
                let has_ext = body.iter().any(|op| matches!(op, TraceOp::Input(i) if *i > 0));
                for a in 0..acc_count {
                    let acc = Self::zmm_for_index(a)?;
                    if has_ext && self.bias_saved {
                        let v = a % nr_vecs;
                        let bias_disp = (v as i32) * 64; // 64 bytes per zmm
                        self.emit_trace_on_accumulator_with_bias_avx512(acc, body, bias_disp)?;
                    } else {
                        self.emit_trace_on_accumulator_avx512(acc, body)?;
                    }
                }
            }

            Ok(())
        }

        /// AVX-512 version of `emit_trace_on_accumulator_with_bias`.
        ///
        /// Same as `emit_trace_on_accumulator_avx512` but `Input(1)` loads
        /// from `[rax + bias_disp]` using `vmovups zmm`.
        fn emit_trace_on_accumulator_with_bias_avx512(
            &mut self,
            acc: AsmRegisterZmm,
            body: &[TraceOp],
            bias_disp: i32,
        ) -> Result<(), String> {
            let n = body.len();
            if n == 0 {
                return Ok(());
            }

            let frame_size = (n * 64) as i32;
            self.asm.sub(rsp, frame_size).map_err(|e| e.to_string())?;

            let slot_off = |i: u32| -> i32 { (i as i32) * 64 };

            for (i, op) in body.iter().enumerate() {
                let i32_idx = i as u32;
                match op {
                    TraceOp::Input(idx) => {
                        if *idx == 0 {
                            self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), acc)
                                .map_err(|e| e.to_string())?;
                        } else {
                            // Input(1+): load from external tensor via rax + bias_disp
                            self.asm.vmovups(zmm29, zmmword_ptr(rax + bias_disp))
                                .map_err(|e| e.to_string())?;
                            self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(zmm29, dword_ptr(label))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vaddps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmulps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vdivps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*c)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmm30, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(zmm29, zmm30, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        self.asm.vpxord(zmm29, zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Abs(a) => {
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vbroadcastss(zmm30, dword_ptr(abs_mask))
                            .map_err(|e| e.to_string())?;
                        self.asm.vandps(zmm29, zmm29, zmm30)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_exp_avx512(zmm30, zmm29, [zmm29, zmm31, acc])?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm30)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sqrt(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsqrtps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrsqrt14ps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_tanh_avx512(zmm30, zmm29, [zmm29, zmm31, acc])?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm30)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Recip(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrcp14ps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmaxps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vminps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_log_avx512(zmm30, zmm29, [zmm29, zmm31, acc])?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm30)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            self.asm.vmovups(acc, zmmword_ptr(rsp + slot_off((n - 1) as u32)))
                .map_err(|e| e.to_string())?;

            self.asm.add(rsp, frame_size).map_err(|e| e.to_string())?;

            Ok(())
        }

        // ── AVX-512 trace-on-accumulator methods ────────────────────────────

        /// Execute a TraceOp body in-place on a single zmm accumulator register.
        ///
        /// AVX-512 version of `emit_trace_on_accumulator`. Uses zmm29-31 as
        /// scratch and 64-byte stack slots.
        fn emit_trace_on_accumulator_avx512(
            &mut self,
            acc: AsmRegisterZmm,
            body: &[TraceOp],
        ) -> Result<(), String> {
            let n = body.len();
            if n == 0 {
                return Ok(());
            }

            let frame_size = (n * 64) as i32;
            self.asm.sub(rsp, frame_size).map_err(|e| e.to_string())?;

            let slot_off = |i: u32| -> i32 { (i as i32) * 64 };

            for (i, op) in body.iter().enumerate() {
                let i32_idx = i as u32;
                match op {
                    TraceOp::Input(_) => {
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), acc)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(zmm29, dword_ptr(label))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vaddps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmulps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vdivps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*c)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmm30, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(zmm29, zmm30, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        self.asm.vpxord(zmm29, zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Abs(a) => {
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vbroadcastss(zmm30, dword_ptr(abs_mask))
                            .map_err(|e| e.to_string())?;
                        self.asm.vandps(zmm29, zmm29, zmm30)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_exp_avx512(zmm30, zmm29, [zmm29, zmm31, acc])?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm30)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sqrt(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsqrtps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrsqrt14ps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_tanh_avx512(zmm30, zmm29, [zmm29, zmm31, acc])?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm30)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Recip(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrcp14ps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmaxps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vminps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        // Placeholder: copy input (real: polynomial approximation)
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            self.asm.vmovups(acc, zmmword_ptr(rsp + slot_off((n - 1) as u32)))
                .map_err(|e| e.to_string())?;

            self.asm.add(rsp, frame_size).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Apply a TraceOp body in-place on a zmm accumulator for elementwise
        /// chain codegen (AVX-512 version).
        ///
        /// Same as `emit_elementwise_trace_body` but uses zmm registers and
        /// 64-byte stack slots. `Input(1)` loads from `[rsi + rbx]` using
        /// `vmovups zmm` (vector) or masked `vmovups zmm{k1}{z}` (tail).
        fn emit_elementwise_trace_body_avx512(
            &mut self,
            acc: AsmRegisterZmm,
            body: &[TraceOp],
            is_binary: bool,
            scalar_tail: bool,
        ) -> Result<(), String> {
            let n = body.len();
            if n == 0 {
                return Ok(());
            }

            let frame_size = (n * 64) as i32;
            self.asm.sub(rsp, frame_size).map_err(|e| e.to_string())?;

            let slot_off = |i: u32| -> i32 { (i as i32) * 64 };

            for (i, op) in body.iter().enumerate() {
                let i32_idx = i as u32;
                match op {
                    TraceOp::Input(idx) => {
                        if *idx == 0 {
                            self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), acc)
                                .map_err(|e| e.to_string())?;
                        } else if *idx == 1 && is_binary {
                            if scalar_tail {
                                self.asm.vmovups(zmm29.k1().z(), zmmword_ptr(rsi + rbx))
                                    .map_err(|e| e.to_string())?;
                            } else {
                                self.asm.vmovups(zmm29, zmmword_ptr(rsi + rbx))
                                    .map_err(|e| e.to_string())?;
                            }
                            self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                                .map_err(|e| e.to_string())?;
                        } else {
                            self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), acc)
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(zmm29, dword_ptr(label))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vaddps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmulps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vdivps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*c)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmm30, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(zmm29, zmm30, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        self.asm.vpxord(zmm29, zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vsubps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Abs(a) => {
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vbroadcastss(zmm30, dword_ptr(abs_mask))
                            .map_err(|e| e.to_string())?;
                        self.asm.vandps(zmm29, zmm29, zmm30)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_exp_avx512(zmm30, zmm29, [zmm29, zmm31, acc])?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm30)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sqrt(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vsqrtps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrsqrt14ps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.emit_tanh_avx512(zmm30, zmm29, [zmm29, zmm31, acc])?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm30)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Recip(a) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vrcp14ps(zmm29, zmm29)
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmaxps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vminps(zmm29, zmm29, zmmword_ptr(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        // Placeholder: copy input (real: polynomial approximation)
                        self.asm.vmovups(zmm29, zmmword_ptr(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        self.asm.vmovups(zmmword_ptr(rsp + slot_off(i32_idx)), zmm29)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            self.asm.vmovups(acc, zmmword_ptr(rsp + slot_off((n - 1) as u32)))
                .map_err(|e| e.to_string())?;

            self.asm.add(rsp, frame_size).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit AVX2 exp(x) approximation (Cephes degree-5 polynomial).
        ///
        /// Computes `dst = exp(src)` using the Cody-Waite range reduction
        /// and a degree-5 Horner polynomial. ~20 AVX2 instructions, ≤1 ULP error.
        ///
        /// Clobbers: `dst`, `s[0]`, `s[1]`, `s[2]`.
        /// Preserves: `src` (if src != dst and src not in scratch).
        fn emit_exp_avx2(
            &mut self,
            dst: AsmRegisterYmm,
            src: AsmRegisterYmm,
            s: [AsmRegisterYmm; 3],
        ) -> Result<(), String> {
            let clamp_lo = self.const_f32(-88.376);
            let clamp_hi = self.const_f32(88.376);
            let log2e    = self.const_f32(1.4426950408889634);
            let c1       = self.const_f32(-0.693359375);
            let c2       = self.const_f32(2.12194440e-4);
            let p0       = self.const_f32(1.9875691500e-4);
            let p1       = self.const_f32(1.3981999507e-3);
            let p2       = self.const_f32(8.3334519073e-3);
            let p3       = self.const_f32(4.1665795894e-2);
            let p4       = self.const_f32(1.6666665459e-1);
            let p5       = self.const_f32(5.0000001201e-1);
            let one      = self.const_f32(1.0);
            let magic127 = self.const_f32(127.0); // for 2^k construction

            self.asm.vmaxps(s[0], src, ymmword_ptr(clamp_lo)).map_err(|e| e.to_string())?;
            self.asm.vminps(s[0], s[0], ymmword_ptr(clamp_hi)).map_err(|e| e.to_string())?;

            self.asm.vmulps(s[1], s[0], ymmword_ptr(log2e)).map_err(|e| e.to_string())?;

            self.asm.vroundps(s[2], s[1], 0i32).map_err(|e| e.to_string())?; // 0 = round to nearest

            self.asm.vfmadd231ps(s[0], s[2], ymmword_ptr(c1)).map_err(|e| e.to_string())?;
            self.asm.vfmadd231ps(s[0], s[2], ymmword_ptr(c2)).map_err(|e| e.to_string())?;

            self.asm.vbroadcastss(dst, dword_ptr(p0)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], ymmword_ptr(p1)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], ymmword_ptr(p2)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], ymmword_ptr(p3)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], ymmword_ptr(p4)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], ymmword_ptr(p5)).map_err(|e| e.to_string())?;
            self.asm.vmulps(dst, dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vmulps(dst, dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vaddps(dst, dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vaddps(dst, dst, ymmword_ptr(one)).map_err(|e| e.to_string())?;

            self.asm.vcvtps2dq(s[1], s[2]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[2], dword_ptr(magic127)).map_err(|e| e.to_string())?;
            self.asm.vcvtps2dq(s[2], s[2]).map_err(|e| e.to_string())?;
            self.asm.vpaddd(s[1], s[1], s[2]).map_err(|e| e.to_string())?;
            self.asm.vpslld(s[1], s[1], 23i32).map_err(|e| e.to_string())?;

            self.asm.vmulps(dst, dst, s[1]).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit AVX2 tanh(x) approximation via `2*sigmoid(2x) - 1`.
        ///
        /// Uses `tanh(x) = 2/(1+exp(-2x)) - 1` which reuses the exp approximation.
        ///
        /// Clobbers: `dst`, `s[0]`, `s[1]`, `s[2]`.
        fn emit_tanh_avx2(
            &mut self,
            dst: AsmRegisterYmm,
            src: AsmRegisterYmm,
            s: [AsmRegisterYmm; 3],
        ) -> Result<(), String> {
            let neg2 = self.const_f32(-2.0);
            let two  = self.const_f32(2.0);
            let one  = self.const_f32(1.0);

            self.asm.vmulps(s[0], src, ymmword_ptr(neg2)).map_err(|e| e.to_string())?;

            self.emit_exp_avx2(dst, s[0], s)?;

            self.asm.vaddps(dst, dst, ymmword_ptr(one)).map_err(|e| e.to_string())?;

            self.asm.vrcpps(s[0], dst).map_err(|e| e.to_string())?;
            self.asm.vmulps(s[1], dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[2], dword_ptr(two)).map_err(|e| e.to_string())?;
            self.asm.vsubps(s[1], s[2], s[1]).map_err(|e| e.to_string())?;
            self.asm.vmulps(s[0], s[0], s[1]).map_err(|e| e.to_string())?;
            self.asm.vmulps(dst, s[0], s[2]).map_err(|e| e.to_string())?;

            self.asm.vsubps(dst, dst, ymmword_ptr(one)).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit AVX2 ln(x) approximation (degree-4 minimax polynomial).
        ///
        /// Computes `dst = ln(src)` by decomposing x = 2^e * m (m in [1,2)),
        /// then `ln(x) = e * ln(2) + ln(m)` where `ln(m)` is approximated
        /// via a Horner polynomial on `t = m - 1`.
        ///
        /// Clobbers: `dst`, `s[0]`, `s[1]`, `s[2]`.
        /// Preserves: `src` (if src != dst and src not in scratch).
        fn emit_log_avx2(
            &mut self,
            dst: AsmRegisterYmm,
            src: AsmRegisterYmm,
            s: [AsmRegisterYmm; 3],
        ) -> Result<(), String> {
            // Constants
            let mantissa_mask = self.const_f32(f32::from_bits(0x007F_FFFF));
            let one           = self.const_f32(1.0);
            let magic127      = self.const_f32(127.0);
            let ln2           = self.const_f32(0.6931471805599453);
            // Minimax coefficients for ln(1+t), t in [0, 1)
            let c1            = self.const_f32(0.99999934);
            let c2            = self.const_f32(-0.49987412);
            let c3            = self.const_f32(0.33179903);
            let c4            = self.const_f32(-0.24073381);

            // Extract exponent: e = float(x >> 23) - 127.0
            self.asm.vpsrld(s[0], src, 23i32).map_err(|e| e.to_string())?;
            self.asm.vcvtdq2ps(s[0], s[0]).map_err(|e| e.to_string())?;
            self.asm.vsubps(s[0], s[0], ymmword_ptr(magic127)).map_err(|e| e.to_string())?;

            // Extract mantissa: m = (x & 0x007FFFFF) | 0x3F800000 → [1.0, 2.0)
            self.asm.vandps(s[1], src, ymmword_ptr(mantissa_mask)).map_err(|e| e.to_string())?;
            self.asm.vorps(s[1], s[1], ymmword_ptr(one)).map_err(|e| e.to_string())?;

            // t = m - 1.0
            self.asm.vsubps(s[1], s[1], ymmword_ptr(one)).map_err(|e| e.to_string())?;

            // Horner: p = ((c4 * t + c3) * t + c2) * t + c1
            self.asm.vbroadcastss(dst, dword_ptr(c4)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[1], ymmword_ptr(c3)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[1], ymmword_ptr(c2)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[1], ymmword_ptr(c1)).map_err(|e| e.to_string())?;
            // p = p * t → ln(1+t) ≈ p
            self.asm.vmulps(dst, dst, s[1]).map_err(|e| e.to_string())?;

            // result = e * ln(2) + p
            self.asm.vfmadd231ps(dst, s[0], ymmword_ptr(ln2)).map_err(|e| e.to_string())?;

            Ok(())
        }

        // ── AVX-512 transcendental approximations ───────────────────────────

        /// Emit AVX-512 exp(x) approximation (Cephes degree-5 polynomial).
        ///
        /// Same algorithm as `emit_exp_avx2` but uses zmm registers and
        /// AVX-512 specific instructions (`vrndscaleps` instead of `vroundps`).
        /// Constants are loaded via `vbroadcastss zmm, dword_ptr(label)` since
        /// the const pool stores 32-byte entries.
        ///
        /// Clobbers: `dst`, `s[0]`, `s[1]`, `s[2]`.
        fn emit_exp_avx512(
            &mut self,
            dst: AsmRegisterZmm,
            src: AsmRegisterZmm,
            s: [AsmRegisterZmm; 3],
        ) -> Result<(), String> {
            let clamp_lo = self.const_f32(-88.376);
            let clamp_hi = self.const_f32(88.376);
            let log2e    = self.const_f32(1.4426950408889634);
            let c1       = self.const_f32(-0.693359375);
            let c2       = self.const_f32(2.12194440e-4);
            let p0       = self.const_f32(1.9875691500e-4);
            let p1       = self.const_f32(1.3981999507e-3);
            let p2       = self.const_f32(8.3334519073e-3);
            let p3       = self.const_f32(4.1665795894e-2);
            let p4       = self.const_f32(1.6666665459e-1);
            let p5       = self.const_f32(5.0000001201e-1);
            let one      = self.const_f32(1.0);
            let magic127 = self.const_f32(127.0);

            // Clamp input
            self.asm.vbroadcastss(s[1], dword_ptr(clamp_lo)).map_err(|e| e.to_string())?;
            self.asm.vmaxps(s[0], src, s[1]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(clamp_hi)).map_err(|e| e.to_string())?;
            self.asm.vminps(s[0], s[0], s[1]).map_err(|e| e.to_string())?;

            // x * log2(e)
            self.asm.vbroadcastss(s[1], dword_ptr(log2e)).map_err(|e| e.to_string())?;
            self.asm.vmulps(s[1], s[0], s[1]).map_err(|e| e.to_string())?;

            // round to nearest (vrndscaleps replaces vroundps in AVX-512)
            self.asm.vrndscaleps(s[2], s[1], 0i32).map_err(|e| e.to_string())?;

            // Cody-Waite range reduction
            self.asm.vbroadcastss(s[1], dword_ptr(c1)).map_err(|e| e.to_string())?;
            self.asm.vfmadd231ps(s[0], s[2], s[1]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(c2)).map_err(|e| e.to_string())?;
            self.asm.vfmadd231ps(s[0], s[2], s[1]).map_err(|e| e.to_string())?;

            // Horner polynomial
            self.asm.vbroadcastss(dst, dword_ptr(p0)).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(p1)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], s[1]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(p2)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], s[1]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(p3)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], s[1]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(p4)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], s[1]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(p5)).map_err(|e| e.to_string())?;
            self.asm.vfmadd213ps(dst, s[0], s[1]).map_err(|e| e.to_string())?;
            self.asm.vmulps(dst, dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vmulps(dst, dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vaddps(dst, dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[1], dword_ptr(one)).map_err(|e| e.to_string())?;
            self.asm.vaddps(dst, dst, s[1]).map_err(|e| e.to_string())?;

            // Construct 2^k via integer shift
            self.asm.vcvtps2dq(s[1], s[2]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[2], dword_ptr(magic127)).map_err(|e| e.to_string())?;
            self.asm.vcvtps2dq(s[2], s[2]).map_err(|e| e.to_string())?;
            self.asm.vpaddd(s[1], s[1], s[2]).map_err(|e| e.to_string())?;
            self.asm.vpslld(s[1], s[1], 23i32).map_err(|e| e.to_string())?;

            self.asm.vmulps(dst, dst, s[1]).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit AVX-512 tanh(x) approximation via `2*sigmoid(2x) - 1`.
        ///
        /// Same algorithm as `emit_tanh_avx2` but uses zmm registers and
        /// `vrcp14ps` instead of `vrcpps`.
        ///
        /// Clobbers: `dst`, `s[0]`, `s[1]`, `s[2]`.
        fn emit_tanh_avx512(
            &mut self,
            dst: AsmRegisterZmm,
            src: AsmRegisterZmm,
            s: [AsmRegisterZmm; 3],
        ) -> Result<(), String> {
            let neg2 = self.const_f32(-2.0);
            let two  = self.const_f32(2.0);
            let one  = self.const_f32(1.0);

            // s[0] = -2 * src
            self.asm.vbroadcastss(s[1], dword_ptr(neg2)).map_err(|e| e.to_string())?;
            self.asm.vmulps(s[0], src, s[1]).map_err(|e| e.to_string())?;

            // dst = exp(-2x)
            self.emit_exp_avx512(dst, s[0], s)?;

            // dst = 1 + exp(-2x)
            self.asm.vbroadcastss(s[1], dword_ptr(one)).map_err(|e| e.to_string())?;
            self.asm.vaddps(dst, dst, s[1]).map_err(|e| e.to_string())?;

            // Newton-Raphson reciprocal: vrcp14ps + one NR step
            self.asm.vrcp14ps(s[0], dst).map_err(|e| e.to_string())?;
            self.asm.vmulps(s[1], dst, s[0]).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(s[2], dword_ptr(two)).map_err(|e| e.to_string())?;
            self.asm.vsubps(s[1], s[2], s[1]).map_err(|e| e.to_string())?;
            self.asm.vmulps(s[0], s[0], s[1]).map_err(|e| e.to_string())?;
            self.asm.vmulps(dst, s[0], s[2]).map_err(|e| e.to_string())?;

            // dst = 2*sigmoid(2x) - 1
            self.asm.vbroadcastss(s[1], dword_ptr(one)).map_err(|e| e.to_string())?;
            self.asm.vsubps(dst, dst, s[1]).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit a TraceOp sequence as AVX2 SIMD instructions.
        ///
        /// This is the core of Phase 3: each TraceOp maps to one or more
        /// SIMD instructions operating on ymm registers. The SSA indices
        /// in TraceOp directly map to ymm register numbers (max 16 SSA
        /// values; returns Err if exceeded).
        ///
        /// Mapping:
        /// - `Input(n)` → `vxorps` (placeholder; real: `vmovups` from memory)
        /// - `Const(v)` → `vxorps` (placeholder; real: `vbroadcastss` from const pool)
        /// - `Add(a,b)` → `vaddps`
        /// - `Sub(a,b)` → `vsubps`
        /// - `Mul(a,b)` → `vmulps`
        /// - `Div(a,b)` → `vdivps`
        /// - `Fma(a,b,c)` → `vmovaps` + `vfmadd231ps`
        /// - `Neg(a)` → `vxorps` + `vsubps` (0 - x)
        /// - `Exp(a)` → `vmovaps` (placeholder; real: polynomial approximation)
        /// - `Sqrt(a)` → `vsqrtps`
        /// - `Rsqrt(a)` → `vrsqrtps`
        /// - `Tanh(a)` → `vmovaps` (placeholder; real: rational approximation)
        /// - `Recip(a)` → `vrcpps`
        /// - `Abs(a)` → `vmovaps` (placeholder; real: `vandps` with 0x7FFFFFFF)
        /// - `Max(a,b)` → `vmaxps`
        /// - `Min(a,b)` → `vminps`
        pub fn emit_trace_ops_avx2(
            &mut self,
            ops: &[TraceOp],
        ) -> Result<Vec<AsmRegisterYmm>, String> {
            let mut reg_map: Vec<AsmRegisterYmm> = Vec::with_capacity(ops.len());

            for (i, op) in ops.iter().enumerate() {
                let reg = Self::ymm_for_index(i)?;
                match op {
                    TraceOp::Input(_) => {
                        self.asm.vxorps(reg, reg, reg).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(reg, dword_ptr(label)).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vaddps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vsubps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vmulps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vdivps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        let rc = reg_map[*c as usize];
                        self.asm.vmovaps(reg, rc).map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vxorps(reg, reg, reg).map_err(|e| e.to_string())?;
                        self.asm.vsubps(reg, reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_exp_avx2(reg, ra, [ymm13, ymm14, ymm15])?;
                    }
                    TraceOp::Sqrt(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vsqrtps(reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vrsqrtps(reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_tanh_avx2(reg, ra, [ymm13, ymm14, ymm15])?;
                    }
                    TraceOp::Recip(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vrcpps(reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_log_avx2(reg, ra, [ymm13, ymm14, ymm15])?;
                    }
                    TraceOp::Abs(a) => {
                        let ra = reg_map[*a as usize];
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vandps(reg, ra, ymmword_ptr(abs_mask)).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vmaxps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vminps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                }
                reg_map.push(reg);
            }

            Ok(reg_map)
        }

        /// Emit a TraceOp sequence as AVX-512 SIMD instructions.
        ///
        /// AVX-512 counterpart of `emit_trace_ops_avx2`. Maps TraceOp SSA
        /// indices to zmm0-zmm27 (zmm28-zmm31 reserved as scratch). Uses
        /// AVX-512 specific instructions (`vpxord`, `vrsqrt14ps`, `vrcp14ps`).
        ///
        /// Returns `Err` if SSA index >= 28.
        pub fn emit_trace_ops_avx512(
            &mut self,
            ops: &[TraceOp],
        ) -> Result<Vec<AsmRegisterZmm>, String> {
            let mut reg_map: Vec<AsmRegisterZmm> = Vec::with_capacity(ops.len());

            for (i, op) in ops.iter().enumerate() {
                let reg = Self::zmm_for_index(i)?;
                match op {
                    TraceOp::Input(_) => {
                        self.asm.vpxord(reg, reg, reg).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Const(v) => {
                        let label = self.const_f32(*v as f32);
                        self.asm.vbroadcastss(reg, dword_ptr(label)).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vaddps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vsubps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vmulps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vdivps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        let rc = reg_map[*c as usize];
                        self.asm.vmovaps(reg, rc).map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vpxord(reg, reg, reg).map_err(|e| e.to_string())?;
                        self.asm.vsubps(reg, reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_exp_avx512(reg, ra, [zmm29, zmm30, zmm31])?;
                    }
                    TraceOp::Sqrt(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vsqrtps(reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vrsqrt14ps(reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_tanh_avx512(reg, ra, [zmm29, zmm30, zmm31])?;
                    }
                    TraceOp::Recip(a) => {
                        let ra = reg_map[*a as usize];
                        self.asm.vrcp14ps(reg, ra).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        let ra = reg_map[*a as usize];
                        self.emit_log_avx512(reg, ra, [zmm29, zmm30, zmm31])?;
                    }
                    TraceOp::Abs(a) => {
                        let ra = reg_map[*a as usize];
                        let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
                        self.asm.vbroadcastss(zmm29, dword_ptr(abs_mask)).map_err(|e| e.to_string())?;
                        self.asm.vandps(reg, ra, zmm29).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vmaxps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        let ra = reg_map[*a as usize];
                        let rb = reg_map[*b as usize];
                        self.asm.vminps(reg, ra, rb).map_err(|e| e.to_string())?;
                    }
                }
                reg_map.push(reg);
            }

            Ok(reg_map)
        }

        /// Map a TraceOp SSA index to a ymm register (ymm0-ymm12).
        ///
        /// Returns `Err` if index >= 13 since ymm13-ymm15 are reserved as
        /// scratch registers for exp/tanh approximations.
        fn ymm_for_index(index: usize) -> Result<AsmRegisterYmm, String> {
            let regs = [
                ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                ymm8, ymm9, ymm10, ymm11, ymm12,
            ];
            if index >= regs.len() {
                return Err(format!(
                    "SSA index {} exceeds available ymm registers (13; ymm13-15 reserved for scratch)",
                    index
                ));
            }
            Ok(regs[index])
        }

        /// Map index 0-15 to ymm0-ymm15 (unrestricted, for scratch registers).
        fn ymm_any(index: usize) -> Result<AsmRegisterYmm, String> {
            let regs = [
                ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
            ];
            if index >= 16 {
                return Err(format!("ymm index {} out of range (0-15)", index));
            }
            Ok(regs[index])
        }

        // ── AVX-512 zmm register helpers ────────────────────────────────────

        /// Map a TraceOp SSA index to a zmm accumulator register (zmm0-zmm27).
        ///
        /// AVX-512 has 32 zmm registers. zmm28-zmm31 are reserved as scratch
        /// for A-panel broadcast and exp/tanh approximations.
        fn zmm_for_index(index: usize) -> Result<AsmRegisterZmm, String> {
            let regs: [AsmRegisterZmm; 28] = [
                zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7,
                zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15,
                zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23,
                zmm24, zmm25, zmm26, zmm27,
            ];
            if index >= regs.len() {
                return Err(format!(
                    "SSA index {} exceeds available zmm accumulators (28; zmm28-31 reserved)",
                    index
                ));
            }
            Ok(regs[index])
        }

        /// Map index 0-31 to zmm0-zmm31 (unrestricted, for scratch registers).
        fn zmm_any(index: usize) -> Result<AsmRegisterZmm, String> {
            let regs: [AsmRegisterZmm; 32] = [
                zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7,
                zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15,
                zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23,
                zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31,
            ];
            if index >= 32 {
                return Err(format!("zmm index {} out of range (0-31)", index));
            }
            Ok(regs[index])
        }

        /// SIMD register width in bytes for the current ISA (32 for AVX2, 64 for AVX-512).
        #[inline]
        fn simd_bytes(&self) -> i32 {
            if self.use_avx512 { 64 } else { 32 }
        }

        /// Set mask register k1 from a compile-time remainder count.
        ///
        /// Computes `(1 << remainder) - 1` and loads into k1 for masked
        /// load/store of tail elements.
        fn emit_set_kmask(&mut self, remainder: usize) -> Result<(), String> {
            let mask_val = (1u32 << remainder) - 1;
            self.asm.mov(eax, mask_val as i32).map_err(|e| e.to_string())?;
            self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit quantized GEMM: dequantize B into scratchpad, then regular GEMM.
        ///
        /// ABI: rdi=A (f32), rsi=B (quantized blocks), r8=C (f32), [rbp+32]=scratchpad
        ///
        /// Strategy: dequantize entire B matrix to f32 in scratchpad,
        /// redirect rsi to the dequantized buffer, then run regular GEMM.
        fn emit_quant_gemm(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            block_size: usize,
            bits: usize,
            profile: &DeviceProfile,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let num_elements = k * n;
            if num_elements % block_size != 0 {
                return Err(format!(
                    "QuantGemm: k*n={} not divisible by block_size={}",
                    num_elements, block_size
                ));
            }
            let num_blocks = num_elements / block_size;
            let dequant_bytes = num_elements * 4; // f32 output

            let dequant_offset = self.blis_scratchpad_offset;
            self.blis_scratchpad_offset += dequant_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset - self.blis_base_offset
            );

            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            eprintln!(
                "QuantGemm {}x{}x{}: bits={} blocks={} dequant_offset={} dequant_bytes={}",
                m, n, k, bits, num_blocks, dequant_offset, dequant_bytes
            );

            match bits {
                8 => self.emit_dequant_q8_0_loop(num_blocks, block_size, dequant_offset as i32)?,
                4 => self.emit_dequant_q4_0_loop(num_blocks, block_size, dequant_offset as i32)?,
                _ => return Err(format!("QuantGemm: unsupported bits={}", bits)),
            }

            self.asm.lea(rsi, qword_ptr(rbx + dequant_offset as i32))
                .map_err(|e| e.to_string())?;

            self.emit_gemm_microkernel(m, n, k, profile, epilogue_bodies)
        }

        /// Emit Q8_0 dequantize loop using AVX2 + F16C.
        ///
        /// BlockQ8_0: `d` (f16, 2 bytes) + `qs` (block_size × i8 bytes)
        /// Dequantize: `out[i] = qs[i] * f16_to_f32(d)`
        ///
        /// Processes 8 elements at a time using vpmovsxbd + vcvtdq2ps.
        ///
        /// Register plan:
        ///   rax = source ptr (quantized blocks, starts at rsi)
        ///   rdx = dest ptr (f32 output in scratchpad)
        ///   rcx = block counter (counts down)
        ///   ymm15 = broadcast f32 scale
        ///   ymm0 = temp for dequantized group
        fn emit_dequant_q8_0_loop(
            &mut self,
            num_blocks: usize,
            block_size: usize,
            dequant_offset: i32,
        ) -> Result<(), String> {
            let block_bytes = (2 + block_size) as i32; // f16 scale + i8 * block_size
            let groups_per_block = block_size / 8;

            self.asm.mov(rax, rsi).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(rbx + dequant_offset))
                .map_err(|e| e.to_string())?;
            self.asm.mov(rcx, num_blocks as u64).map_err(|e| e.to_string())?;

            let mut block_loop = self.asm.create_label();
            let mut block_done = self.asm.create_label();

            self.asm.set_label(&mut block_loop).map_err(|e| e.to_string())?;
            self.asm.test(rcx, rcx).map_err(|e| e.to_string())?;
            self.asm.jz(block_done).map_err(|e| e.to_string())?;

            self.asm.movzx(r9d, word_ptr(rax)).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm15, r9d).map_err(|e| e.to_string())?;
            self.asm.vcvtph2ps(xmm15, xmm15).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(ymm15, xmm15).map_err(|e| e.to_string())?;

            for g in 0..groups_per_block {
                let src_off = 2 + (g * 8) as i32; // skip f16 header
                let dst_off = (g * 8 * 4) as i32;  // 8 f32 = 32 bytes

                self.asm.vpmovsxbd(ymm0, qword_ptr(rax + src_off))
                    .map_err(|e| e.to_string())?;
                self.asm.vcvtdq2ps(ymm0, ymm0).map_err(|e| e.to_string())?;
                self.asm.vmulps(ymm0, ymm0, ymm15).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rdx + dst_off), ymm0)
                    .map_err(|e| e.to_string())?;
            }

            self.asm.add(rax, block_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rdx, (block_size * 4) as i32).map_err(|e| e.to_string())?;
            self.asm.dec(rcx).map_err(|e| e.to_string())?;
            self.asm.jmp(block_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut block_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit Q4_0 dequantize loop using AVX2 + F16C.
        ///
        /// BlockQ4_0: `d` (f16, 2 bytes) + `qs` (block_size/2 × u8, packed nibbles)
        /// Element layout (GGML convention):
        ///   elements 0..15  = low nibbles of qs[0..15]
        ///   elements 16..31 = high nibbles of qs[0..15]
        /// Dequantize: `out[i] = (nibble - 8) * f16_to_f32(d)`
        ///
        /// Register plan:
        ///   rax = source ptr, rdx = dest ptr, rcx = block counter
        ///   ymm15 = broadcast f32 scale
        ///   ymm14 = broadcast i32(8) for subtraction
        ///   xmm13 = 0x0F byte mask
        ///   ymm0-ymm1 = temps
        fn emit_dequant_q4_0_loop(
            &mut self,
            num_blocks: usize,
            block_size: usize,
            dequant_offset: i32,
        ) -> Result<(), String> {
            if block_size != 32 {
                return Err(format!("Q4_0: only block_size=32 supported, got {}", block_size));
            }
            let block_bytes = 2 + (block_size / 2) as i32; // 18 bytes for block_size=32

            self.asm.mov(rax, rsi).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(rbx + dequant_offset))
                .map_err(|e| e.to_string())?;
            self.asm.mov(rcx, num_blocks as u64).map_err(|e| e.to_string())?;

            self.asm.mov(r9d, 8i32).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm14, r9d).map_err(|e| e.to_string())?;
            self.asm.vpbroadcastd(ymm14, xmm14).map_err(|e| e.to_string())?;

            self.asm.mov(r9d, 0x0F0F0F0Fi32).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm13, r9d).map_err(|e| e.to_string())?;
            self.asm.vpbroadcastd(xmm13, xmm13).map_err(|e| e.to_string())?;

            let mut block_loop = self.asm.create_label();
            let mut block_done = self.asm.create_label();

            self.asm.set_label(&mut block_loop).map_err(|e| e.to_string())?;
            self.asm.test(rcx, rcx).map_err(|e| e.to_string())?;
            self.asm.jz(block_done).map_err(|e| e.to_string())?;

            self.asm.movzx(r9d, word_ptr(rax)).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm15, r9d).map_err(|e| e.to_string())?;
            self.asm.vcvtph2ps(xmm15, xmm15).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(ymm15, xmm15).map_err(|e| e.to_string())?;

            self.asm.vmovdqu(xmm12, xmmword_ptr(rax + 2))
                .map_err(|e| e.to_string())?;

            self.asm.vpand(xmm10, xmm12, xmm13).map_err(|e| e.to_string())?;
            self.asm.vpsrlw(xmm11, xmm12, 4i32).map_err(|e| e.to_string())?;
            self.asm.vpand(xmm11, xmm11, xmm13).map_err(|e| e.to_string())?;

            self.emit_q4_dequant_group(xmm10, 0)?;    // → [rdx + 0]
            self.asm.vpsrldq(xmm0, xmm10, 8i32).map_err(|e| e.to_string())?;
            self.emit_q4_dequant_group(xmm0, 32)?;     // → [rdx + 32]

            self.emit_q4_dequant_group(xmm11, 64)?;    // → [rdx + 64]
            self.asm.vpsrldq(xmm0, xmm11, 8i32).map_err(|e| e.to_string())?;
            self.emit_q4_dequant_group(xmm0, 96)?;     // → [rdx + 96]

            self.asm.add(rax, block_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rdx, (block_size * 4) as i32).map_err(|e| e.to_string())?;
            self.asm.dec(rcx).map_err(|e| e.to_string())?;
            self.asm.jmp(block_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut block_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit dequant for one group of 8 Q4_0 nibbles.
        ///
        /// Input: `src_xmm` lower 8 bytes contain nibble values (0-15).
        /// Output: 8 × f32 stored at `[rdx + dst_off]`.
        /// Uses ymm15 (scale), ymm14 (i32 broadcast 8), ymm0 as temp.
        fn emit_q4_dequant_group(
            &mut self,
            src_xmm: iced_x86::code_asm::AsmRegisterXmm,
            dst_off: i32,
        ) -> Result<(), String> {
            self.asm.vpmovzxbd(ymm0, src_xmm).map_err(|e| e.to_string())?;
            self.asm.vpsubd(ymm0, ymm0, ymm14).map_err(|e| e.to_string())?;
            self.asm.vcvtdq2ps(ymm0, ymm0).map_err(|e| e.to_string())?;
            self.asm.vmulps(ymm0, ymm0, ymm15).map_err(|e| e.to_string())?;
            self.asm.vmovups(ymmword_ptr(rdx + dst_off), ymm0)
                .map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit a single nop as placeholder.
        fn emit_nop_placeholder(&mut self) -> Result<(), String> {
            self.asm.nop().map_err(|e| e.to_string())
        }
    }

    // ── AVX-512 codegen unit tests ──────────────────────────────────────

    #[cfg(test)]
    #[cfg(target_arch = "x86_64")]
    mod avx512_tests {
        use super::*;
        use crate::compiler::trace::TraceOp;
        use crate::dispatch::DeviceProfile;
        use iced_x86::code_asm::*;

        /// Helper: create a DeviceProfile with use_avx512 forced on.
        fn avx512_profile() -> DeviceProfile {
            let mut profile = DeviceProfile::detect();
            profile.kernel_config.use_avx512 = true;
            profile
        }

        #[test]
        fn test_zmm_for_index_valid() {
            for i in 0..28 {
                let result = X86CodeGen::zmm_for_index(i);
                assert!(result.is_ok(), "zmm_for_index({i}) should succeed");
            }
        }

        #[test]
        fn test_zmm_for_index_overflow() {
            let result = X86CodeGen::zmm_for_index(28);
            assert!(result.is_err(), "zmm_for_index(28) should fail");
            let msg = result.unwrap_err();
            assert!(
                msg.contains("exceeds available zmm accumulators"),
                "unexpected error: {msg}"
            );
        }

        #[test]
        fn test_zmm_any_valid() {
            for i in 0..32 {
                let result = X86CodeGen::zmm_any(i);
                assert!(result.is_ok(), "zmm_any({i}) should succeed");
            }
        }

        #[test]
        fn test_zmm_any_overflow() {
            let result = X86CodeGen::zmm_any(32);
            assert!(result.is_err(), "zmm_any(32) should fail");
            let msg = result.unwrap_err();
            assert!(
                msg.contains("zmm index 32 out of range"),
                "unexpected error: {msg}"
            );
        }

        #[test]
        fn test_avx512_exp_compiles() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let result = codegen.emit_exp_avx512(zmm0, zmm1, [zmm29, zmm30, zmm31]);
            assert!(result.is_ok(), "emit_exp_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_tanh_compiles() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let result = codegen.emit_tanh_avx512(zmm0, zmm1, [zmm29, zmm30, zmm31]);
            assert!(result.is_ok(), "emit_tanh_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_log_compiles() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let result = codegen.emit_log_avx512(zmm0, zmm1, [zmm29, zmm30, zmm31]);
            assert!(result.is_ok(), "emit_log_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_trace_on_accumulator() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let body = vec![
                TraceOp::Input(0),
                TraceOp::Neg(0),
            ];
            let result = codegen.emit_trace_on_accumulator_avx512(zmm0, &body);
            assert!(result.is_ok(), "emit_trace_on_accumulator_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_elementwise_trace_body() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let body = vec![
                TraceOp::Input(0),
                TraceOp::Neg(0),
            ];
            let result = codegen.emit_elementwise_trace_body_avx512(zmm0, &body, false, false);
            assert!(result.is_ok(), "emit_elementwise_trace_body_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_trace_ops_silu() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let ops = vec![
                TraceOp::Input(0),   // [0] v
                TraceOp::Neg(0),     // [1] -v
                TraceOp::Exp(1),     // [2] exp(-v)
                TraceOp::Const(1.0), // [3] 1.0
                TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
                TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
            ];
            let result = codegen.emit_trace_ops_avx512(&ops);
            assert!(result.is_ok(), "emit_trace_ops_avx512 SiLU failed: {:?}", result.err());
            assert_eq!(result.unwrap().len(), 6);
        }

        #[test]
        fn test_avx512_trace_ops_all_ops() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let ops = vec![
                TraceOp::Input(0),     // [0]
                TraceOp::Input(1),     // [1]
                TraceOp::Neg(0),       // [2]
                TraceOp::Abs(0),       // [3]
                TraceOp::Exp(0),       // [4]
                TraceOp::Sqrt(0),      // [5]
                TraceOp::Rsqrt(0),     // [6]
                TraceOp::Tanh(0),      // [7]
                TraceOp::Recip(0),     // [8]
                TraceOp::Log(0),       // [9]
                TraceOp::Add(0, 1),    // [10]
                TraceOp::Sub(0, 1),    // [11]
                TraceOp::Mul(0, 1),    // [12]
                TraceOp::Div(0, 1),    // [13]
                TraceOp::Max(0, 1),    // [14]
                TraceOp::Min(0, 1),    // [15]
                TraceOp::Fma(0, 1, 2), // [16]
                TraceOp::Const(2.0),   // [17]
            ];
            let result = codegen.emit_trace_ops_avx512(&ops);
            assert!(result.is_ok(), "emit_trace_ops_avx512 all ops failed: {:?}", result.err());
            assert_eq!(result.unwrap().len(), 18);
        }

        #[test]
        fn test_avx512_trace_on_accumulator_with_bias() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let body = vec![
                TraceOp::Input(0),   // accumulator
                TraceOp::Input(1),   // bias from external
                TraceOp::Add(0, 1),  // acc + bias
            ];
            let result = codegen.emit_trace_on_accumulator_with_bias_avx512(zmm0, &body, 0i32);
            assert!(
                result.is_ok(),
                "emit_trace_on_accumulator_with_bias_avx512 failed: {:?}",
                result.err()
            );
        }

        #[test]
        fn test_avx512_trace_ops_overflow() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            // 28 inputs fill zmm0-zmm27, index 28 should fail
            let mut ops: Vec<TraceOp> = (0..28).map(|i| TraceOp::Input(i as u32)).collect();
            ops.push(TraceOp::Add(0, 1)); // SSA index 28
            let result = codegen.emit_trace_ops_avx512(&ops);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("exceeds available zmm accumulators"));
        }

        #[test]
        fn test_simd_bytes_avx2_vs_avx512() {
            // AVX2 mode
            let mut p = DeviceProfile::detect();
            p.kernel_config.use_avx512 = false;
            let cg = X86CodeGen::new(&p);
            assert_eq!(cg.simd_bytes(), 32, "AVX2 simd_bytes should be 32");

            // AVX-512 mode
            let profile = avx512_profile();
            let cg512 = X86CodeGen::new(&profile);
            assert_eq!(cg512.simd_bytes(), 64, "AVX-512 simd_bytes should be 64");
        }
    }
}

#[cfg(feature = "jit-x86")]
impl crate::compiler::codegen::emitter::MachineCodeEmitter for jit::X86CodeGen {
    fn emit_plan(
        &mut self,
        plan: &crate::compiler::fusion::FusionPlan,
        graph: &crate::compiler::graph::CompilerGraph,
        alloc: &crate::compiler::buffer_alloc::BufferAllocation,
        profile: &crate::dispatch::DeviceProfile,
        registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String> {
        self.emit_plan(plan, graph, alloc, profile, registry)
    }

    fn simd_width(&self) -> usize {
        self.simd_width()
    }
}

/// x86_64 platform backend factory.
#[cfg(feature = "jit-x86")]
pub struct X86Backend;

#[cfg(feature = "jit-x86")]
impl crate::compiler::codegen::emitter::PlatformBackend for X86Backend {
    type Emitter = jit::X86CodeGen;

    fn new_emitter(&self, profile: &crate::dispatch::DeviceProfile) -> Self::Emitter {
        jit::X86CodeGen::new(profile)
    }

    fn platform(&self) -> crate::compiler::codegen::emitter::Platform {
        #[cfg(target_arch = "x86_64")]
        let avx512 = std::is_x86_feature_detected!("avx512f");
        #[cfg(not(target_arch = "x86_64"))]
        let avx512 = false;
        crate::compiler::codegen::emitter::Platform::X86_64 { avx512 }
    }

    fn num_simd_regs(&self) -> usize {
        16 // x86_64: ymm0-ymm15 (AVX2) or zmm0-zmm15 (AVX-512)
    }
}

#[cfg(feature = "jit-x86")]
pub use jit::X86CodeGen;

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::compiler::executable::CompiledLayer;

    #[test]
    fn test_x86_stub_callable() {
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

    #[cfg(feature = "jit-x86")]
    mod jit_tests {
        use crate::compiler::trace::{TraceOp, ComputePattern};
        use crate::dispatch::DeviceProfile;
        use super::super::jit;

        #[test]
        fn test_emit_trace_ops_silu() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),   // [0] v
                TraceOp::Neg(0),     // [1] -v
                TraceOp::Exp(1),     // [2] exp(-v)
                TraceOp::Const(1.0), // [3] 1.0
                TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
                TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 6);
        }

        #[test]
        fn test_emit_trace_ops_add() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),
                TraceOp::Input(1),
                TraceOp::Add(0, 1),
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 3);
        }

        #[test]
        fn test_emit_trace_ops_gelu() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),              // [0] x
                TraceOp::Mul(0, 0),             // [1] x^2
                TraceOp::Mul(1, 0),             // [2] x^3
                TraceOp::Const(0.044715),       // [3] 0.044715
                TraceOp::Mul(3, 2),             // [4] 0.044715 * x^3
                TraceOp::Add(0, 4),             // [5] x + 0.044715 * x^3
                TraceOp::Const(0.7978845608),   // [6] sqrt(2/pi)
                TraceOp::Mul(6, 5),             // [7] sqrt(2/pi) * (...)
                TraceOp::Tanh(7),               // [8] tanh(...)
                TraceOp::Const(1.0),            // [9] 1.0
                TraceOp::Add(9, 8),             // [10] 1 + tanh(...)
                TraceOp::Mul(0, 10),            // [11] x * (1 + tanh(...))
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 12);
        }

        #[test]
        fn test_emit_trace_ops_all_unary() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),  // [0]
                TraceOp::Neg(0),    // [1]
                TraceOp::Abs(0),    // [2]
                TraceOp::Exp(0),    // [3]
                TraceOp::Sqrt(0),   // [4]
                TraceOp::Rsqrt(0),  // [5]
                TraceOp::Tanh(0),   // [6]
                TraceOp::Recip(0),  // [7]
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 8);
        }

        #[test]
        fn test_emit_trace_ops_all_binary() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),   // [0]
                TraceOp::Input(1),   // [1]
                TraceOp::Add(0, 1),  // [2]
                TraceOp::Sub(0, 1),  // [3]
                TraceOp::Mul(0, 1),  // [4]
                TraceOp::Div(0, 1),  // [5]
                TraceOp::Max(0, 1),  // [6]
                TraceOp::Min(0, 1),  // [7]
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 8);
        }

        #[test]
        fn test_emit_trace_ops_fma() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),    // [0] a
                TraceOp::Input(1),    // [1] b
                TraceOp::Input(2),    // [2] c
                TraceOp::Fma(0, 1, 2), // [3] a*b + c
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 4);
        }

        #[test]
        fn test_ymm_for_index_overflow() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let mut ops: Vec<TraceOp> = (0..13).map(|i| TraceOp::Input(i as u32)).collect();
            ops.push(TraceOp::Add(0, 1)); // SSA index 13

            let result = codegen.emit_trace_ops_avx2(&ops);
            assert!(result.is_err());
            let msg = result.unwrap_err();
            assert!(
                msg.contains("SSA index 13 exceeds available ymm registers"),
                "unexpected error: {msg}"
            );
        }

        #[test]
        fn test_emit_gemm_microkernel() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);
            codegen.emit_gemm_microkernel(1, 4096, 4096, &profile, &[]).unwrap();
        }

        #[test]
        fn test_emit_gemm_microkernel_large() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);
            codegen.emit_gemm_microkernel(512, 4096, 4096, &profile, &[]).unwrap();
        }

        #[test]
        fn test_emit_full_plan() {
            use crate::compiler::graph::CompilerGraph;
            use crate::compiler::ir::LayerIR;
            use crate::compiler::fusion::fuse_with_dag;
            use crate::compiler::registry::ScalarOpRegistry;
            use crate::compiler::buffer_alloc::{analyze_lifetimes, allocate_buffers};
            use crate::inference::types::ModelConfig;

            let config = ModelConfig::llama_7b();
            let ir = LayerIR::from_model_config(&config, 1);
            let profile = DeviceProfile::detect();
            let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
            let registry = ScalarOpRegistry::with_defaults();
            let plan = fuse_with_dag(&graph, &registry, &profile);
            let lifetimes = analyze_lifetimes(&graph, &plan);
            let alloc = allocate_buffers(&lifetimes);

            let mut codegen = jit::X86CodeGen::new(&profile);
            let output = codegen.emit_plan(&plan, &graph, &alloc, &profile, Some(&registry)).unwrap();

            assert!(!output.code.is_empty());
            assert!(output.scratchpad_bytes > 0);

            eprintln!("Generated {} bytes of x86_64 code", output.code.len());
        }
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "jit-x86")]
mod e2e_tests {
    use crate::compiler::codegen::x86_64::jit::X86CodeGen;
    use crate::compiler::executable::CompiledLayer;
    use crate::compiler::graph::{CompilerGraph, OpKind, OpId, TensorId};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::DType;
    use std::collections::HashMap;
    use crate::compiler::registry::{ScalarOpRegistry, OpKindKey};
    use crate::compiler::trace::{TraceOp, OpTrace, ComputePattern, ScalarFnSignature, ScalarParam};

    /// Number of f32 elements for e2e tests (must be multiple of 8 for AVX2).
    const N: usize = 32;

    /// Build a minimal graph with one unary elementwise op.
    /// Returns (graph, op_id, output_tensor_id).
    fn build_unary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![input], vec![output], "op");
        (g, op_id, output)
    }

    /// Build a minimal graph with one binary elementwise op.
    /// Returns (graph, op_id, output_tensor_id).
    fn build_binary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![N], DType::F32);
        let b = g.add_tensor("b", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![a, b], vec![output], "op");
        (g, op_id, output)
    }

    /// Build a FusionPlan with a single ElementwiseChain group.
    fn build_plan(op_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        }
    }

    /// Build a minimal BufferAllocation (no intermediate tensors needed).
    fn build_alloc(total_bytes: usize) -> BufferAllocation {
        BufferAllocation {
            slots: vec![],
            total_bytes,
            num_tensors: 0,
            bytes_saved: 0,
        }
    }

    /// Compile a graph+plan into executable code and return the CompiledLayer.
    fn compile(
        graph: &CompilerGraph,
        plan: &FusionPlan,
        alloc: &BufferAllocation,
    ) -> CompiledLayer {
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        let output = codegen.emit_plan(plan, graph, alloc, &profile, None).unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
    }

    /// Execute a compiled unary op.
    ///
    /// Codegen ABI: rdi=input (arg0), r8=output (arg4=seq_lens slot).
    unsafe fn exec_unary(layer: &CompiledLayer, input: &[f32], output: &mut [f32]) {
        let f = layer.entry_point();
        f(
            input.as_ptr() as *const u8,       // rdi = input
            std::ptr::null(),                   // rsi = weights (unused)
            std::ptr::null_mut(),               // rdx = kv_cache (unused)
            std::ptr::null(),                   // rcx = positions (unused)
            output.as_mut_ptr() as *const usize, // r8 = output ptr
            0,                                  // r9 = batch_size (unused)
            0,                                  // stack = seq_len (unused)
            std::ptr::null_mut(),               // stack = output (unused)
            std::ptr::null_mut(),               // stack = scratchpad (unused)
        );
    }

    /// Execute a compiled binary op.
    ///
    /// Codegen ABI: rdi=input_a (arg0), rsi=input_b (arg1=weights slot),
    ///              r8=output (arg4=seq_lens slot).
    unsafe fn exec_binary(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) {
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,            // rdi = input a
            b.as_ptr() as *const u8,            // rsi = input b (weights slot)
            std::ptr::null_mut(),                // rdx = kv_cache (unused)
            std::ptr::null(),                    // rcx = positions (unused)
            output.as_mut_ptr() as *const usize, // r8 = output ptr
            0,                                   // r9 = batch_size (unused)
            0,                                   // stack = seq_len (unused)
            std::ptr::null_mut(),                // stack = output (unused)
            std::ptr::null_mut(),                // stack = scratchpad (unused)
        );
    }

    /// Test data: a spread of values including negatives, zero, and positives.
    fn test_input() -> Vec<f32> {
        let base = [
            -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        ];
        (0..N).map(|i| base[i % base.len()]).collect()
    }

    fn test_input_b() -> Vec<f32> {
        (0..N).map(|i| 0.1 * (i as f32) + 0.5).collect()
    }

    /// Scalar SiLU reference: x / (1 + exp(-x))
    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Scalar GELU reference: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn ref_gelu(x: f32) -> f32 {
        let inner = (2.0_f32 / std::f32::consts::PI).sqrt()
            * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn test_e2e_silu_correctness() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Silu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("SiLU e2e: all {} elements within 1e-3 tolerance", N);
    }

    #[test]
    fn test_e2e_gelu_correctness() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Gelu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_gelu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "GELU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("GELU e2e: all {} elements within 1e-3 tolerance", N);
    }

    #[test]
    fn test_e2e_add_correctness() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Add);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] + b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Add mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Add e2e: all {} elements within 1e-5 tolerance", N);
    }

    #[test]
    fn test_e2e_mul_correctness() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Mul);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] * b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Mul mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Mul e2e: all {} elements within 1e-5 tolerance", N);
    }

    /// Build a minimal graph with one Gemm op (M×K @ K×N → M×N).
    fn build_gemm_graph(m: usize, n: usize, k: usize) -> (CompilerGraph, OpId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let c = g.add_tensor("C", vec![m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op_id = g.add_op(OpKind::Gemm { m, n, k }, vec![a, b], vec![c], "gemm");
        (g, op_id)
    }

    /// Build a FusionPlan with a single Standalone GEMM group.
    fn build_gemm_plan(op_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![op_id],
            }],
            op_to_group,
        }
    }

    /// Execute a compiled GEMM.
    ///
    /// Codegen ABI: rdi=A (arg0), rsi=B (arg1), r8=C (arg4=seq_lens slot).
    unsafe fn exec_gemm(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) {
        let mut scratch = vec![0u8; layer.scratchpad_bytes];
        let scratch_ptr = if layer.scratchpad_bytes > 0 {
            scratch.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,             // rdi = A
            b.as_ptr() as *const u8,             // rsi = B
            std::ptr::null_mut(),                 // rdx = kv_cache (unused)
            std::ptr::null(),                     // rcx = positions (unused)
            c.as_mut_ptr() as *const usize,       // r8 = C output
            0,                                    // r9 = batch_size (unused)
            0,                                    // stack = seq_len (unused)
            std::ptr::null_mut(),                 // stack = output (unused)
            scratch_ptr,                          // stack = scratchpad
        );
    }

    /// Scalar reference matmul: C[i,j] = sum_p A[i,p] * B[p,j]
    fn ref_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Fill a matrix with deterministic pseudo-random values in [-1, 1].
    fn fill_matrix(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
        let len = rows * cols;
        let mut v = Vec::with_capacity(len);
        let mut s = seed;
        for _ in 0..len {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((s as f32) / (u32::MAX as f32) * 2.0 - 1.0);
        }
        v
    }

    /// Assert two matrices are element-wise close.
    fn assert_matrix_close(
        got: &[f32], expected: &[f32], m: usize, n: usize, tol: f32, label: &str,
    ) {
        assert_eq!(got.len(), expected.len());
        let mut max_diff = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let diff = (got[idx] - expected[idx]).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < tol,
                    "{} mismatch at [{},{}]: got={}, expected={}, diff={}",
                    label, i, j, got[idx], expected[idx], diff,
                );
            }
        }
        eprintln!("{} e2e: {}×{} max_diff={:.2e} (tol={:.0e})", label, m, n, max_diff, tol);
    }

    #[test]
    fn test_e2e_gemm_single_tile() {
        let (m, n, k) = (4, 8, 16);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);
        eprintln!("scratchpad_bytes = {}", layer.scratchpad_bytes);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 137);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-5;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-single-tile");
    }

    #[test]
    fn test_e2e_gemm_full_tiles() {
        let (m, n, k) = (12, 16, 32);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 99);
        let b = fill_matrix(k, n, 200);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-5;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-full-tiles");
    }

    #[test]
    fn test_e2e_gemm_with_remainders() {
        let (m, n, k) = (10, 24, 20);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 7);
        let b = fill_matrix(k, n, 13);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-5;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-remainders");
    }

    #[test]
    fn test_e2e_gemm_larger() {
        let (m, n, k) = (25, 40, 64);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);
        eprintln!("scratchpad_bytes = {}", layer.scratchpad_bytes);

        let a = fill_matrix(m, k, 31415);
        let b = fill_matrix(k, n, 27182);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        for i in 0..8.min(m) {
            eprint!("row {}: ", i);
            for j in 0..8.min(n) {
                eprint!("{:.4} ", c_jit[i * n + j]);
            }
            eprintln!();
        }
        eprintln!("--- ref ---");
        for i in 0..8.min(m) {
            eprint!("row {}: ", i);
            for j in 0..8.min(n) {
                eprint!("{:.4} ", c_ref[i * n + j]);
            }
            eprintln!();
        }

        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-larger");
    }

    #[test]
    fn test_e2e_gemm_large_k() {
        let (m, n, k) = (12, 16, 1024);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 54321);
        let b = fill_matrix(k, n, 12345);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-large-k");
    }

    #[test]
    fn test_e2e_gemm_blis_diag() {
        for &(m, n, k) in &[
            (6usize, 16usize, 297usize),    // single tile, 2 KC blocks
            (12, 32, 297),                    // 2x2 tiles, 2 KC blocks
            (24, 48, 297),                    // multi-tile
            (48, 64, 512),
            (72, 256, 297),
            (128, 16, 297),
            (128, 256, 297),
            (128, 256, 512),
        ] {
            println!("DIAG: testing {}x{}x{}", m, n, k);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            let (graph, op_id) = build_gemm_graph(m, n, k);
            let plan = build_gemm_plan(op_id);
            let alloc = build_alloc(m * n * 4);
            let layer = compile(&graph, &plan, &alloc);

            let a = fill_matrix(m, k, 271828);
            let b = fill_matrix(k, n, 314159);
            let mut c_jit = vec![0.0f32; m * n];
            let mut c_ref = vec![0.0f32; m * n];

            ref_matmul(&a, &b, &mut c_ref, m, n, k);
            println!("DIAG: calling JIT...");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };
            println!("DIAG: JIT returned");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let tol = k as f32 * 1e-3;
            let mut mismatches = 0;
            let mut max_diff = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let idx = i * n + j;
                    let diff = (c_jit[idx] - c_ref[idx]).abs();
                    max_diff = max_diff.max(diff);
                    if diff >= tol && mismatches < 5 {
                        println!(
                            "  MISMATCH [{},{}]: jit={:.6}, ref={:.6}, diff={:.6}",
                            i, j, c_jit[idx], c_ref[idx], diff,
                        );
                        mismatches += 1;
                    }
                }
            }
            println!(
                "DIAG {}x{}x{}: max_diff={:.2e}, tol={:.2e}, mismatches={}",
                m, n, k, max_diff, tol, mismatches,
            );
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            assert_eq!(mismatches, 0, "GEMM-blis-diag {}x{}x{} failed", m, n, k);
        }
    }

    #[test]
    fn test_e2e_gemm_full_blis() {
        let (m, n, k) = (128, 256, 512);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 271828);
        let b = fill_matrix(k, n, 314159);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-full-blis");
    }

    /// Build a graph with GEMM followed by a unary epilogue op.
    /// Returns (graph, gemm_op_id, epilogue_op_id).
    fn build_gemm_epilogue_graph(
        m: usize, n: usize, k: usize, epi_kind: OpKind,
    ) -> (CompilerGraph, OpId, OpId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let gemm_out = g.add_tensor("gemm_out", vec![m, n], DType::F32);
        let epi_out = g.add_tensor("epi_out", vec![m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![epi_out];
        let gemm_id = g.add_op(
            OpKind::Gemm { m, n, k }, vec![a, b], vec![gemm_out], "gemm",
        );
        let epi_id = g.add_op(epi_kind, vec![gemm_out], vec![epi_out], "epilogue");
        (g, gemm_id, epi_id)
    }

    /// Build a FusionPlan with GemmEpilogue pattern.
    fn build_gemm_epilogue_plan(gemm_id: OpId, epi_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(gemm_id, 0);
        op_to_group.insert(epi_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_id,
                epilogue: vec![epi_id],
                mode: FusionMode::EpilogueInjection,
                ops: vec![gemm_id, epi_id],
            }],
            op_to_group,
        }
    }

    /// Compile with a ScalarOpRegistry (needed for epilogue trace lookup).
    fn compile_with_registry(
        graph: &CompilerGraph,
        plan: &FusionPlan,
        alloc: &BufferAllocation,
        registry: &ScalarOpRegistry,
    ) -> CompiledLayer {
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        let output = codegen
            .emit_plan(plan, graph, alloc, &profile, Some(registry))
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
    }

    /// Build a registry with a ReLU trace: max(0, x).
    fn registry_with_relu() -> ScalarOpRegistry {
        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(
            OpKindKey::Silu,
            OpTrace {
                op_kind: OpKind::Silu,
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),   // [0] x
                        TraceOp::Const(0.0),  // [1] 0.0
                        TraceOp::Max(0, 1),   // [2] max(x, 0)
                    ],
                },
                signature: ScalarFnSignature {
                    fn_ptr: std::ptr::null(),
                    params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr],
                },
            },
        );
        reg
    }

    /// Build a registry with the real SiLU trace: x / (1 + exp(-x)).
    fn registry_with_silu() -> ScalarOpRegistry {
        ScalarOpRegistry::with_defaults()
    }

    /// Scalar reference ReLU.
    fn apply_relu_inplace(v: &mut [f32]) {
        for x in v.iter_mut() {
            *x = x.max(0.0);
        }
    }

    /// Scalar reference SiLU: x / (1 + exp(-x)).
    fn apply_silu_inplace(v: &mut [f32]) {
        for x in v.iter_mut() {
            *x = *x / (1.0 + (-*x).exp());
        }
    }

    #[test]
    fn gemm_relu_epilogue_small() {
        let m = 4;
        let n = 8;
        let k = 4;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_relu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 99);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_relu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        assert_matrix_close(&c_jit, &c_ref, m, n, 1e-4, "GEMM+ReLU-small");
    }

    #[test]
    fn gemm_relu_epilogue_medium() {
        let m = 16;
        let n = 16;
        let k = 8;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_relu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 7);
        let b = fill_matrix(k, n, 13);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_relu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+ReLU-medium");
    }

    #[test]
    fn gemm_silu_epilogue_small() {
        let m = 4;
        let n = 8;
        let k = 4;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_silu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 99);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_silu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        assert_matrix_close(&c_jit, &c_ref, m, n, 5e-3, "GEMM+SiLU-small");
    }

    #[test]
    fn gemm_silu_epilogue_medium() {
        let m = 16;
        let n = 16;
        let k = 8;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_silu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 7);
        let b = fill_matrix(k, n, 13);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_silu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 5e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+SiLU-medium");
    }

    /// Large GEMM+SiLU that exercises the BLIS blocking path.
    #[test]
    fn gemm_silu_epilogue_blis() {
        let m = 64;
        let n = 64;
        let k = 64;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_silu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 31);
        let b = fill_matrix(k, n, 59);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_silu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 5e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+SiLU-blis");
    }

    /// Build a graph with RmsNorm → Gemm.
    fn build_norm_gemm_graph(
        m: usize, n: usize, k: usize, eps: f32,
    ) -> (CompilerGraph, OpId, OpId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let norm_w = g.add_tensor("norm_w", vec![k], DType::F32);
        let normed = g.add_tensor("normed", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let c = g.add_tensor("C", vec![m, n], DType::F32);
        g.inputs = vec![a, b, norm_w];
        g.outputs = vec![c];
        let norm_id = g.add_op(
            OpKind::RmsNorm { eps }, vec![a, norm_w], vec![normed], "rms_norm",
        );
        let gemm_id = g.add_op(
            OpKind::Gemm { m, n, k }, vec![normed, b], vec![c], "gemm",
        );
        (g, norm_id, gemm_id)
    }

    /// Build a FusionPlan with NormIntoGemm pattern.
    fn build_norm_gemm_plan(norm_id: OpId, gemm_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(norm_id, 0);
        op_to_group.insert(gemm_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_id,
                epilogue: vec![],
                mode: FusionMode::NormIntoGemm,
                ops: vec![norm_id, gemm_id],
            }],
            op_to_group,
        }
    }

    /// Execute a compiled NormIntoGemm kernel.
    ///
    /// ABI: rdi=A, rsi=B, rdx=norm_weight, r8=C, scratchpad on stack.
    unsafe fn exec_norm_gemm(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        norm_w: &[f32],
        c: &mut [f32],
    ) {
        let mut scratch = vec![0u8; layer.scratchpad_bytes];
        let scratch_ptr = if layer.scratchpad_bytes > 0 {
            scratch.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,              // rdi = A input
            b.as_ptr() as *const u8,              // rsi = B weight matrix
            norm_w.as_ptr() as *mut u8,           // rdx = norm weight ptr
            std::ptr::null(),                     // rcx = positions (unused)
            c.as_mut_ptr() as *const usize,       // r8 = C output
            0,                                    // r9 = batch_size (unused)
            0,                                    // stack = seq_len (unused)
            std::ptr::null_mut(),                 // stack = output (unused)
            scratch_ptr,                          // stack = scratchpad
        );
    }

    /// Reference: row-wise RmsNorm then matmul.
    fn ref_norm_gemm(
        a: &[f32], b: &[f32], norm_w: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize, eps: f32,
    ) {
        let mut normed = vec![0.0f32; m * k];
        for row in 0..m {
            let off = row * k;
            let mut sum_sq = 0.0f32;
            for j in 0..k {
                sum_sq += a[off + j] * a[off + j];
            }
            let scale = 1.0 / (sum_sq / k as f32 + eps).sqrt();
            for j in 0..k {
                normed[off + j] = a[off + j] * scale * norm_w[j];
            }
        }
        ref_matmul(&normed, b, c, m, n, k);
    }

    #[test]
    fn test_e2e_norm_into_gemm_small() {
        let (m, n, k) = (4, 8, 16);
        let eps = 1e-5;
        let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
        let plan = build_norm_gemm_plan(norm_id, gemm_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 137);
        let norm_w: Vec<f32> = (0..k).map(|i| 0.8 + 0.4 * (i as f32) / k as f32).collect();
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
        unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

        let tol = k as f32 * 1e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormIntoGemm-small");
    }

    #[test]
    fn test_e2e_norm_into_gemm_medium() {
        let (m, n, k) = (32, 64, 128);
        let eps = 1e-5;
        let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
        let plan = build_norm_gemm_plan(norm_id, gemm_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 271828);
        let b = fill_matrix(k, n, 314159);
        let norm_w: Vec<f32> = (0..k).map(|i| 0.5 + 1.0 * (i as f32) / k as f32).collect();
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
        unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

        let tol = k as f32 * 1e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormIntoGemm-medium");
    }

}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "jit-x86")]
mod registry_elementwise_tests {
    use crate::compiler::codegen::x86_64::jit::X86CodeGen;
    use crate::compiler::executable::CompiledLayer;
    use crate::compiler::graph::{CompilerGraph, OpKind, OpId, TensorId};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::buffer_alloc::BufferAllocation;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::DType;
    use crate::compiler::registry::ScalarOpRegistry;
    use std::collections::HashMap;

    const N: usize = 32;

    fn build_unary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![input], vec![output], "op");
        (g, op_id, output)
    }

    fn build_binary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![N], DType::F32);
        let b = g.add_tensor("b", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![a, b], vec![output], "op");
        (g, op_id, output)
    }

    /// Build a graph with two chained ops: unary anchor → binary epilogue.
    /// E.g. SiLU(x) + bias.
    fn build_chain_graph(
        anchor_kind: OpKind,
        epilogue_kind: OpKind,
    ) -> (CompilerGraph, OpId, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![N], DType::F32);
        let bias = g.add_tensor("bias", vec![N], DType::F32);
        let mid = g.add_tensor("mid", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![input, bias];
        g.outputs = vec![output];
        let anchor_id = g.add_op(anchor_kind, vec![input], vec![mid], "anchor");
        let epi_id = g.add_op(epilogue_kind, vec![mid, bias], vec![output], "epilogue");
        (g, anchor_id, epi_id, output)
    }

    fn build_plan(op_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        }
    }

    fn build_chain_plan(anchor_id: OpId, epi_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(anchor_id, 0);
        op_to_group.insert(epi_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: anchor_id,
                epilogue: vec![epi_id],
                mode: FusionMode::LoopFusion,
                ops: vec![anchor_id, epi_id],
            }],
            op_to_group,
        }
    }

    fn build_alloc(total_bytes: usize) -> BufferAllocation {
        BufferAllocation {
            slots: vec![],
            total_bytes,
            num_tensors: 0,
            bytes_saved: 0,
        }
    }

    fn compile_with_registry(
        graph: &CompilerGraph,
        plan: &FusionPlan,
        alloc: &BufferAllocation,
        registry: &ScalarOpRegistry,
    ) -> CompiledLayer {
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        let output = codegen
            .emit_plan(plan, graph, alloc, &profile, Some(registry))
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
    }

    unsafe fn exec_unary(layer: &CompiledLayer, input: &[f32], output: &mut [f32]) {
        let f = layer.entry_point();
        f(
            input.as_ptr() as *const u8,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            output.as_mut_ptr() as *const usize,
            0, 0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }

    unsafe fn exec_binary(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) {
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,
            b.as_ptr() as *const u8,
            std::ptr::null_mut(),
            std::ptr::null(),
            output.as_mut_ptr() as *const usize,
            0, 0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }

    fn test_input() -> Vec<f32> {
        let base = [
            -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        ];
        (0..N).map(|i| base[i % base.len()]).collect()
    }

    fn test_input_b() -> Vec<f32> {
        (0..N).map(|i| 0.1 * (i as f32) + 0.5).collect()
    }

    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn ref_gelu(x: f32) -> f32 {
        let inner = (2.0_f32 / std::f32::consts::PI).sqrt()
            * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn test_registry_silu() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Silu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "Registry SiLU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_gelu() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Gelu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_gelu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "Registry GELU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry GELU: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_add() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Add);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] + b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Registry Add mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry Add: all {} elements within 1e-5", N);
    }

    #[test]
    fn test_registry_mul() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Mul);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] * b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Registry Mul mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry Mul: all {} elements within 1e-5", N);
    }

    #[test]
    fn test_registry_silu_add_chain() {
        let (graph, anchor_id, epi_id, _) =
            build_chain_graph(OpKind::Silu, OpKind::Add);
        let plan = build_chain_plan(anchor_id, epi_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let bias = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &input, &bias, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]) + bias[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU+Add chain mismatch at [{}]: input={}, bias={}, got={}, expected={}, diff={}",
                i, input[i], bias[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU+Add chain: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_silu_mul_chain() {
        let (graph, anchor_id, epi_id, _) =
            build_chain_graph(OpKind::Silu, OpKind::Mul);
        let plan = build_chain_plan(anchor_id, epi_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let scale = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &input, &scale, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]) * scale[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU+Mul chain mismatch at [{}]: input={}, scale={}, got={}, expected={}, diff={}",
                i, input[i], scale[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU+Mul chain: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_silu_with_tail() {
        const TAIL_N: usize = 19;
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![TAIL_N], DType::F32);
        let output = g.add_tensor("output", vec![TAIL_N], DType::F32);
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
        let alloc = build_alloc(TAIL_N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&g, &plan, &alloc, &registry);

        let base = [
            -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
            -0.3, 0.7, 2.2,
        ];
        let inp: Vec<f32> = base.to_vec();
        let mut out = vec![0.0f32; TAIL_N];

        unsafe { exec_unary(&layer, &inp, &mut out) };

        for i in 0..TAIL_N {
            let expected = ref_silu(inp[i]);
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU tail mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, inp[i], out[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU with tail: all {} elements within 1e-3", TAIL_N);
    }
}
