pub struct PtxDialect {
    sm_version: u32,
}

impl PtxDialect {
    pub fn new(sm_version: u32) -> Self {
        Self { sm_version }
    }
}

impl GpuBackendDialect for PtxDialect {
    fn name(&self) -> &'static str { "PTX" }
    fn sm_version(&self) -> Option<u32> { Some(self.sm_version) }
    fn gfx_arch(&self) -> Option<u32> { None }
    fn wave_size(&self) -> Option<u32> { None }

    fn gpr32_prefix(&self) -> &'static str { "%r" }
    fn gpr64_prefix(&self) -> &'static str { "%rd" }
    fn vec_prefix(&self) -> &'static str { "%f" }
    fn mask_prefix(&self) -> &'static str { "%p" }
    fn tile_prefix(&self) -> &'static str { "%t" }

    fn emit_shared_mem_decl(&self, ctx: &mut GpuLowerContext, _name: &str, size_bytes: usize) {
        ctx.emit_line(&format!(".shared .align 16 .b8 smem[{size_bytes}];"));
    }

    fn mov_f32(&self, dst: &str, src: &str) -> String {
        format!("mov.f32 {dst}, {src};")
    }

    fn fma_f32(&self, dst: &str, a: &str, b: &str, acc: &str) -> String {
        format!("fma.rn.f32 {dst}, {a}, {b}, {acc};")
    }

    fn accumulate(&self, acc: &str, src: &str) -> String {
        format!("add.f32 {acc}, {acc}, {src};")
    }

    fn add_ptr(&self, dst: &str, base: &str, offset: &str) -> String {
        format!("add.u64 {dst}, {base}, {offset};")
    }

    fn add_gpr(&self, dst: &str, a: &str, b: &str) -> String {
        format!("add.u32 {dst}, {a}, {b};")
    }

    fn emit_global_load_f32(&self, ctx: &mut GpuLowerContext, dst: &str, base: &str, offset: &str) {
        // ARCH-GPU-PTX-ADDR: PTX `[reg+reg]` 语法非法，必须先算出 64-bit 地址
        ctx.emit_line(&format!("cvt.u64.u32 %rd_addr, {offset};"));
        ctx.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
        ctx.emit_line(&format!("ld.global.f32 {dst}, [%rd_addr];"));
    }

    fn emit_global_store_f32(&self, ctx: &mut GpuLowerContext, src: &str, base: &str, offset: &str) {
        ctx.emit_line(&format!("cvt.u64.u32 %rd_addr, {offset};"));
        ctx.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
        ctx.emit_line(&format!("st.global.f32 [%rd_addr], {src};"));
    }

    fn emit_load_abi_ptr(&self, ctx: &mut GpuLowerContext, dst: &str, param_name: &str) {
        ctx.emit_line(&format!("ld.param.u64 {dst}, [{param_name}];"));
    }

    fn emit_shared_load(&self, ctx: &mut GpuLowerContext, dst: &str, name: &str, offset: &str, dtype_str: &str) {
        ctx.emit_line(&format!("ld.shared.{dtype_str} {dst}, [{name} + {offset}];"));
    }

    fn emit_shared_store(&self, ctx: &mut GpuLowerContext, src: &str, name: &str, offset: &str, dtype_str: &str) {
        ctx.emit_line(&format!("st.shared.{dtype_str} [{name} + {offset}], {src};"));
    }

    fn emit_vec_binop(&self, ctx: &mut GpuLowerContext, dst: &str, a: &str, b: &str, op: &VecOp) {
        let line = match op {
            VecOp::Add => format!("add.f32 {dst}, {a}, {b};"),
            VecOp::Sub => format!("sub.f32 {dst}, {a}, {b};"),
            VecOp::Mul => format!("mul.f32 {dst}, {a}, {b};"),
            VecOp::Div => format!("div.approx.f32 {dst}, {a}, {b};"),
            VecOp::Max => format!("max.f32 {dst}, {a}, {b};"),
            VecOp::Min => format!("min.f32 {dst}, {a}, {b};"),
            VecOp::And => format!("and.b32 {dst}, {a}, {b};"),
            VecOp::Or  => format!("or.b32 {dst}, {a}, {b};"),
            VecOp::Xor => format!("xor.b32 {dst}, {a}, {b};"),
            VecOp::AndNot => {
                ctx.emit_line(&format!("not.b32 {dst}, {b};"));
                format!("and.b32 {dst}, {a}, {dst};")
            }
            VecOp::Not => format!("not.b32 {dst}, {a};"),
            VecOp::Shl => format!("shl.b32 {dst}, {a}, {b};"),
            VecOp::Shr => format!("shr.b32 {dst}, {a}, {b};"),
        };
        ctx.emit_line(&line);
    }

    fn emit_vec_shift_imm(&self, ctx: &mut GpuLowerContext, dst: &str, a: &str, amount: u32, dir: &VecShiftDir) {
        let line = match dir {
            VecShiftDir::Left => format!("shl.b32 {dst}, {a}, {amount};"),
            VecShiftDir::Right => format!("shr.b32 {dst}, {a}, {amount};"),
        };
        ctx.emit_line(&line);
    }

    fn emit_vec_unary_op(&self, dst: &str, a: &str, op: &GpuVecUnaryOp) -> String {
        match op {
            GpuVecUnaryOp::Neg => format!("neg.f32 {dst}, {a};"),
            GpuVecUnaryOp::Abs => format!("abs.f32 {dst}, {a};"),
            GpuVecUnaryOp::Sqrt => format!("sqrt.approx.f32 {dst}, {a};"),
            GpuVecUnaryOp::Rsqrt => format!("rsqrt.approx.f32 {dst}, {a};"),
            GpuVecUnaryOp::Recip => format!("rcp.approx.f32 {dst}, {a};"),
            GpuVecUnaryOp::Round => format!("cvt.rni.f32.f32 {dst}, {a};"),
            GpuVecUnaryOp::Floor => format!("cvt.rmi.f32.f32 {dst}, {a};"),
            GpuVecUnaryOp::Ceil  => format!("cvt.rpi.f32.f32 {dst}, {a};"),
            GpuVecUnaryOp::IntToFloat => format!("cvt.rn.f32.s32 {dst}, {a};"),
        }
    }

    fn emit_transcendental(&self, dst: &str, src: &str, func: &TranscendentalFn) -> String {
        let fn_str = match func {
            TranscendentalFn::Exp => "ex2.approx",
            TranscendentalFn::Log => "lg2.approx",
            TranscendentalFn::Tanh => "tanh.approx",
            _ => "mov",
        };
        format!("{fn_str}.f32 {dst}, {src};")
    }

    fn emit_broadcast_const(&self, ctx: &mut GpuLowerContext, dst: &str, val: f32) {
        // ARCH-GPU-PTX-LITERAL: PTX f32 立即数必须是 IEEE 754 hex: 0f<8位十六进制>
        ctx.emit_line(&format!("mov.f32 {dst}, 0f{:08X};", val.to_bits()));
    }

    fn emit_broadcast_mem_load(&self, ctx: &mut GpuLowerContext, dst: &str, base: &str, offset: &str) {
        ctx.emit_line(&format!("cvt.u64.u32 %rd_addr, {offset};"));
        ctx.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
        ctx.emit_line(&format!("ld.global.f32 {dst}, [%rd_addr];"));
    }

    fn emit_broadcast_vreg(&self, dst: &str, src: &str) -> String {
        format!("mov.f32 {dst}, {src};")
    }

    fn emit_sync(&self, ctx: &mut GpuLowerContext, kind: BarrierKind) {
        match kind {
            BarrierKind::WarpSync | BarrierKind::BlockSync => {
                ctx.emit_line("bar.sync 0;");
            }
            BarrierKind::MemFence => {
                ctx.emit_line("membar.cta;");
            }
        }
    }

    fn emit_loop_begin(
        &self,
        ctx: &mut GpuLowerContext,
        counter: &str,
        offset: &str,
        bound_expr: &str,
        label_id: u32,
        _step_bytes: usize,
        scratch_pred: &str,
        _scratch_bound: &str,
    ) {
        ctx.emit_line(&format!("mov.u32 {counter}, 0;"));
        ctx.emit_line(&format!("mov.u32 {offset}, 0;"));
        ctx.emit_line(&format!("LOOP_{label_id}:"));
        ctx.emit_line(&format!("setp.ge.u32 {scratch_pred}, {counter}, {bound_expr};"));
        ctx.emit_line(&format!("@{scratch_pred} bra LOOP_END_{label_id};"));
    }

    fn emit_loop_end(
        &self,
        ctx: &mut GpuLowerContext,
        counter: &str,
        offset: &str,
        label_id: u32,
        step_bytes: usize,
        _indent: &mut usize,
    ) {
        ctx.emit_line(&format!("add.u32 {counter}, {counter}, 1;"));
        ctx.emit_line(&format!("add.u32 {offset}, {offset}, {step_bytes};"));
        ctx.emit_line(&format!("bra LOOP_{label_id};"));
        ctx.emit_line(&format!("LOOP_END_{label_id}:"));
    }

    fn emit_conditional_skip(
        &self,
        ctx: &mut GpuLowerContext,
        mask: &str,
        skip_id: u32,
        scratch_pred: &str,
    ) {
        ctx.emit_line(&format!("setp.eq.f32 {scratch_pred}, {mask}, 0.0;"));
        ctx.emit_line(&format!("@{scratch_pred} bra SKIP_{skip_id};"));
    }

    fn emit_conditional_skip_end(&self, ctx: &mut GpuLowerContext, skip_id: u32) {
        ctx.emit_line(&format!("SKIP_{skip_id}:"));
    }

    fn async_copy_support(&self) -> AsyncCopyLevel {
        if self.sm_version >= 90 {
            AsyncCopyLevel::TmaBulk
        } else if self.sm_version >= 80 {
            AsyncCopyLevel::CpAsync
        } else {
            AsyncCopyLevel::None
        }
    }

    fn emit_async_copy(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        src: &str,
        size: usize,
    ) -> Result<(), CompilerError> {
        if self.sm_version >= 90 {
            ctx.emit_line(&format!("// §SM90 TMA async bulk copy ({size} bytes)"));
            ctx.emit_line(&format!("cp.async.bulk.shared::cluster.global [{dst}], [{src}], {size}, [mbar];"));
            ctx.emit_line(&format!("mbarrier.arrive.expect_tx.shared::cta.b64 [mbar], {size};"));
            Ok(())
        } else if self.sm_version >= 80 {
            ctx.emit_line(&format!("cp.async.ca.shared.global [{dst}], [{src}], {size};"));
            Ok(())
        } else {
            Err(CompilerError::CodegenViolation(
                format!("AsyncCopy: SM{} does not support async copy (requires SM80+)", self.sm_version),
            ))
        }
    }

    fn emit_async_wait(
        &self,
        ctx: &mut GpuLowerContext,
        scratch_pred: &str,
    ) -> Result<(), CompilerError> {
        if self.sm_version >= 90 {
            ctx.emit_line("// §SM90 mbarrier wait");
            ctx.emit_line(&format!("mbarrier.try_wait.parity.shared::cta.b64 {scratch_pred}, [mbar], 0;"));
            Ok(())
        } else if self.sm_version >= 80 {
            ctx.emit_line("cp.async.wait_all;");
            Ok(())
        } else {
            Err(CompilerError::CodegenViolation(
                format!("AsyncWait: SM{} does not support async wait (requires SM80+)", self.sm_version),
            ))
        }
    }

    fn emit_warp_reduce(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        src: &str,
        op: &ReduceOp,
    ) -> Result<(), CompilerError> {
        let op_str = match op {
            ReduceOp::Sum => "add",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Prod => "mul",
            ReduceOp::LogSum => "add",
        };
        let fs0 = ctx.scratch_vec_names[0];
        ctx.emit_line(&format!("mov.f32 {dst}, {src};"));
        for delta in [16, 8, 4, 2, 1] {
            ctx.emit_line(&format!("shfl.sync.bfly.b32 {fs0}, {dst}, {delta}, 0x1f, 0xffffffff;"));
            ctx.emit_line(&format!("{op_str}.f32 {dst}, {dst}, {fs0};"));
        }
        Ok(())
    }

    fn has_tensor_cores(&self) -> bool {
        self.sm_version >= 70
    }

    fn emit_tile_mma(
        &self,
        ctx: &mut GpuLowerContext,
        c: &str,
        a: &str,
        b: &str,
    ) -> Result<(), CompilerError> {
        if self.sm_version >= 100 {
            ctx.emit_line("// §SM100 tcgen05.mma (block-scaled, TMEM-backed)");
            ctx.emit_line(&format!("// A-desc in {a}, B-desc in {b}, C-tmem at %tmem_addr"));
            ctx.emit_line("tcgen05.mma.cta_group::1.kind::f16");
            ctx.emit_line(&format!("  [%tmem_addr], {a}, {b}, 0x0,"));
            ctx.emit_line("  0, 1;");
            ctx.emit_line("tcgen05.wait::ld.sync.aligned;");
        } else if self.sm_version >= 90 {
            ctx.emit_line("// §SM90 WGMMA async MMA (warpgroup 128 threads)");
            ctx.emit_line("wgmma.fence.sync.aligned;");
            ctx.emit_line(&format!("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {{{c}}}, {a}, {b};"));
            ctx.emit_line("wgmma.commit_group.sync.aligned;");
            ctx.emit_line("wgmma.wait_group.sync.aligned 0;");
        } else if self.sm_version >= 80 {
            ctx.emit_line(&format!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {c}, {a}, {b}, {c};"));
        } else if self.sm_version >= 70 {
            ctx.emit_line(&format!("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32 {c}, {a}, {b}, {c};"));
        } else {
            return Err(CompilerError::CodegenViolation(
                format!("TileMma: SM{} does not have tensor cores (requires SM70+)", self.sm_version),
            ));
        }
        Ok(())
    }

    fn emit_shared_mem_alloc(&self, ctx: &mut GpuLowerContext, name: &str, bytes: usize) {
        ctx.emit_line(&format!(".shared .align 4 .b8 {name}[{bytes}];"));
    }

    fn emit_gpr_binop(&self, dst: &str, a: &str, b: &str, op: &GpuGprBinOp) -> String {
        match op {
            GpuGprBinOp::Add => format!("add.u32 {dst}, {a}, {b};"),
            GpuGprBinOp::Sub => format!("sub.u32 {dst}, {a}, {b};"),
            GpuGprBinOp::Mul => format!("mul.lo.u32 {dst}, {a}, {b};"),
            GpuGprBinOp::Div => format!("div.u32 {dst}, {a}, {b};"),
            GpuGprBinOp::And => format!("and.b32 {dst}, {a}, {b};"),
            GpuGprBinOp::Or  => format!("or.b32 {dst}, {a}, {b};"),
            GpuGprBinOp::Xor => format!("xor.b32 {dst}, {a}, {b};"),
            GpuGprBinOp::Shl => format!("shl.b32 {dst}, {a}, {b};"),
            GpuGprBinOp::Shr => format!("shr.b32 {dst}, {a}, {b};"),
        }
    }

    fn emit_gpr_load_imm(&self, ctx: &mut GpuLowerContext, dst: &str, value: u32) {
        ctx.emit_line(&format!("mov.u32 {dst}, {value};"));
    }

    fn emit_prefetch(&self, ctx: &mut GpuLowerContext, addr: &str) -> Result<(), CompilerError> {
        ctx.emit_line(&format!("prefetch.global.L1 [{addr}];"));
        Ok(())
    }

    fn emit_mem_fence(&self, ctx: &mut GpuLowerContext) {
        ctx.emit_line("membar.cta;");
    }

    fn emit_warp_prng(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        seed: &str,
    ) -> Result<(), CompilerError> {
        ctx.emit_line(&format!("// PTX PRNG: xorwow from seed {seed}"));
        ctx.emit_line(&format!("cvt.rn.f32.u32 {dst}, %tid.x;"));
        ctx.emit_line(&format!("mul.f32 {dst}, {dst}, {seed};"));
        Ok(())
    }

    fn ptx_version_string(&self) -> Option<&'static str> {
        Some(if self.sm_version >= 100 {
            "8.7"  // Blackwell
        } else if self.sm_version >= 90 {
            "8.3"  // Hopper
        } else if self.sm_version >= 70 {
            "8.0"  // Volta/Turing/Ampere
        } else if self.sm_version >= 60 {
            "6.5"  // Pascal (SM 6.0-6.9)
        } else {
            "5.0"  // Maxwell/Kepler
        })
    }

    fn code_format(&self) -> crate::compiler::codegen::CodeFormat {
        crate::compiler::codegen::CodeFormat::Ptx
    }

    fn emit_shared_mem_addr(&self, ctx: &mut GpuLowerContext, dst: &str) {
        ctx.emit_line(&format!("cvta.shared.u64 {dst}, smem;"));
    }

    // ── 控制流分支 ──

    fn emit_conditional_exit(&self, ctx: &mut GpuLowerContext, cond: &str, scratch_pred: &str) {
        ctx.emit_line(&format!("setp.ne.u32 {scratch_pred}, {cond}, 0;"));
        ctx.emit_line(&format!("@{scratch_pred} ret;"));
    }

    fn emit_branch_if_ptr_nonnull(&self, ctx: &mut GpuLowerContext, ptr: &str, target_label: &str, scratch_pred: &str) {
        ctx.emit_line(&format!("setp.ne.u64 {scratch_pred}, {ptr}, 0;"));
        ctx.emit_line(&format!("@{scratch_pred} bra BATCH_MODE_{target_label};"));
    }

    fn emit_branch_if_gpr_zero(&self, ctx: &mut GpuLowerContext, value: &str, label: &str, scratch_pred: &str) {
        ctx.emit_line(&format!("setp.eq.u32 {scratch_pred}, {value}, 0;"));
        ctx.emit_line(&format!("@{scratch_pred} bra {label};"));
    }

    fn emit_branch_if_gpr_lt_u(&self, ctx: &mut GpuLowerContext, a: &str, b: &str, label: &str, scratch_pred: &str) {
        ctx.emit_line(&format!("setp.lt.u32 {scratch_pred}, {a}, {b};"));
        ctx.emit_line(&format!("@{scratch_pred} bra {label};"));
    }

    fn emit_unconditional_branch(&self, ctx: &mut GpuLowerContext, label: &str) {
        ctx.emit_line(&format!("bra {label};"));
    }

    fn emit_conditional_skip_end_with_indent(&self, ctx: &mut GpuLowerContext, skip_id: u32, _indent: &mut usize) {
        ctx.emit_line(&format!("SKIP_{skip_id}:"));
    }

    // ── GPR 条件操作 ──

    fn emit_gpr_cond_is_null(
        &self, ctx: &mut GpuLowerContext, ptr: &str, action: &GprBranchAction,
        scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        ctx.emit_line(&format!("setp.eq.u64 {scratch_pred}, {ptr}, 0;"));
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_is_nonnull(
        &self, ctx: &mut GpuLowerContext, ptr: &str, action: &GprBranchAction,
        scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        ctx.emit_line(&format!("setp.ne.u64 {scratch_pred}, {ptr}, 0;"));
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_cmp_eq(
        &self, ctx: &mut GpuLowerContext, a: &str, imm: u64, action: &GprBranchAction,
        scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        ctx.emit_line(&format!("setp.eq.u32 {scratch_pred}, {a}, {imm};"));
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_cmp_lt_u(
        &self, ctx: &mut GpuLowerContext, a: &str, imm: u64, action: &GprBranchAction,
        scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        ctx.emit_line(&format!("setp.lt.u32 {scratch_pred}, {a}, {imm};"));
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_bit_clear(
        &self, ctx: &mut GpuLowerContext, bitmap: &str, mask: u32, action: &GprBranchAction,
        scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        ctx.emit_line(&format!("and.b32 {bitmap}, {bitmap}, {mask};"));
        ctx.emit_line(&format!("setp.eq.u32 {scratch_pred}, {bitmap}, 0;"));
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_bit_set(
        &self, ctx: &mut GpuLowerContext, bitmap: &str, mask: u32, action: &GprBranchAction,
        scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        ctx.emit_line(&format!("and.b32 {bitmap}, {bitmap}, {mask};"));
        ctx.emit_line(&format!("setp.ne.u32 {scratch_pred}, {bitmap}, 0;"));
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("@{scratch_pred} bra {epilogue_label};"));
            }
        }
    }

    // ── 共享内存异步操作 ──

    fn emit_shared_mem_async_store(
        &self, ctx: &mut GpuLowerContext, name: &str, offset_str: &str,
        src_reg: &str, copy_bytes: usize,
    ) -> Result<(), CompilerError> {
        if self.sm_version >= 80 {
            ctx.emit_line(&format!("cp.async.ca.shared.global [{name} + {offset_str}], [{src_reg}], {copy_bytes};"));
            Ok(())
        } else {
            Err(CompilerError::CodegenViolation(
                format!("SharedMemAsyncStore: SM{} does not support cp.async (requires SM80+)", self.sm_version),
            ))
        }
    }

    fn emit_shared_mem_async_wait_group(
        &self, ctx: &mut GpuLowerContext, n: u32,
    ) -> Result<(), CompilerError> {
        if self.sm_version >= 80 {
            ctx.emit_line(&format!("cp.async.wait_group {n};"));
            Ok(())
        } else {
            Err(CompilerError::CodegenViolation(
                format!("SharedMemAsyncWaitGroup: SM{} does not support cp.async.wait_group (requires SM80+)", self.sm_version),
            ))
        }
    }

    // ── Warp 屏障 ──

    fn emit_warp_barrier_arrive(
        &self, ctx: &mut GpuLowerContext, barrier_symbol: &str, tx_bytes: usize,
    ) {
        if self.sm_version >= 90 {
            ctx.emit_line(&format!(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 [{barrier_symbol}], {tx_bytes};"
            ));
        }
    }

    fn emit_warp_barrier_wait(
        &self, ctx: &mut GpuLowerContext, barrier_symbol: &str, parity: u32,
        scratch_pred: &str,
    ) {
        if self.sm_version >= 90 {
            ctx.emit_line(&format!(
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 {scratch_pred}, [{barrier_symbol}], {parity};"
            ));
        }
    }

    // ── Warp Role ──

    fn emit_warp_role_declare(
        &self, ctx: &mut GpuLowerContext, role: u32,
    ) -> Result<(), CompilerError> {
        if self.sm_version >= 90 {
            match role {
                0 => {
                    ctx.emit_line("// §SM90 Warp Specialization: Producer warp (TMA load)");
                    ctx.emit_line("setmaxnreg.inc.sync.allocating_group.u32 32;");
                }
                1 => {
                    ctx.emit_line("// §SM90 Warp Specialization: Consumer warp (WGMMA compute)");
                    ctx.emit_line("setmaxnreg.dec.sync.allocating_group.u32 32;");
                }
                _ => {
                    return Err(CompilerError::CodegenViolation(
                        format!("WarpRoleDeclare: unknown role {role} (expected 0=Producer, 1=Consumer)"),
                    ));
                }
            }
        }
        Ok(())
    }
}

// ── C-style 方言共享实现 (HIP/Metal) ──

/// C-style 向量二元操作（HIP/Metal 共用）
