fn emit_c_style_vec_binop(ctx: &mut GpuLowerContext, dst: &str, a: &str, b: &str, op: &VecOp) {
    if matches!(op, VecOp::Max) {
        ctx.emit_line(&format!("{dst} = max({a}, {b});"));
    } else if matches!(op, VecOp::Min) {
        ctx.emit_line(&format!("{dst} = min({a}, {b});"));
    } else if matches!(op, VecOp::Not) {
        ctx.emit_line(&format!("{dst} = ~{a};"));
    } else {
        let op_str = match op {
            VecOp::Add => "+", VecOp::Sub => "-",
            VecOp::Mul => "*", VecOp::Div => "/",
            VecOp::And => "&", VecOp::Or => "|",
            VecOp::Xor => "^", VecOp::AndNot => "&~",
            VecOp::Shl => "<<", VecOp::Shr => ">>",
            _ => "+",
        };
        ctx.emit_line(&format!("{dst} = {a} {op_str} {b};"));
    }
}

// ── HIP 方言实现 ──

/// AMD HIP 方言
pub struct HipDialect {
    gfx_arch: u32,
    wave_size: u32,
}

impl HipDialect {
    pub fn new(gfx_arch: u32, wave_size: u32) -> Self {
        Self { gfx_arch, wave_size }
    }
}

impl GpuBackendDialect for HipDialect {
    fn name(&self) -> &'static str { "HIP" }
    fn sm_version(&self) -> Option<u32> { None }
    fn gfx_arch(&self) -> Option<u32> { Some(self.gfx_arch) }
    fn wave_size(&self) -> Option<u32> { Some(self.wave_size) }

    fn gpr32_prefix(&self) -> &'static str { "r_" }
    fn gpr64_prefix(&self) -> &'static str { "rd_" }
    fn vec_prefix(&self) -> &'static str { "f_" }
    fn mask_prefix(&self) -> &'static str { "p_" }
    fn tile_prefix(&self) -> &'static str { "t_" }

    fn emit_shared_mem_decl(&self, ctx: &mut GpuLowerContext, _name: &str, size_bytes: usize) {
        ctx.emit_line(&format!("__shared__ float smem[{}];", size_bytes / 4));
    }

    fn mov_f32(&self, dst: &str, src: &str) -> String {
        format!("s_mov_b32 {dst}, {src};")
    }

    fn fma_f32(&self, dst: &str, a: &str, b: &str, acc: &str) -> String {
        format!("{dst} = fma({a}, {b}, {acc});")
    }

    fn accumulate(&self, acc: &str, src: &str) -> String {
        format!("{acc} += {src};")
    }

    fn add_ptr(&self, dst: &str, base: &str, offset: &str) -> String {
        format!("{dst} = {base} + {offset};")
    }

    fn add_gpr(&self, dst: &str, a: &str, b: &str) -> String {
        format!("{dst} = {a} + {b};")
    }

    fn emit_global_load_f32(&self, ctx: &mut GpuLowerContext, dst: &str, base: &str, offset: &str) {
        ctx.emit_line(&format!("{dst} = *(({base}) + ({offset})/4);"));
    }

    fn emit_global_store_f32(&self, ctx: &mut GpuLowerContext, src: &str, base: &str, offset: &str) {
        ctx.emit_line(&format!("*(({base}) + ({offset})/4) = {src};"));
    }

    fn emit_load_abi_ptr(&self, ctx: &mut GpuLowerContext, dst: &str, param_name: &str) {
        ctx.emit_line(&format!("{dst} = {param_name};"));
    }

    fn emit_shared_load(&self, ctx: &mut GpuLowerContext, dst: &str, name: &str, offset: &str, _dtype_str: &str) {
        ctx.emit_line(&format!("{dst} = {name}[({offset}) / 4];"));
    }

    fn emit_shared_store(&self, ctx: &mut GpuLowerContext, src: &str, name: &str, offset: &str, _dtype_str: &str) {
        ctx.emit_line(&format!("{name}[({offset}) / 4] = {src};"));
    }

    fn emit_vec_binop(&self, ctx: &mut GpuLowerContext, dst: &str, a: &str, b: &str, op: &VecOp) {
        emit_c_style_vec_binop(ctx, dst, a, b, op);
    }

    fn emit_vec_shift_imm(&self, ctx: &mut GpuLowerContext, dst: &str, a: &str, amount: u32, dir: &VecShiftDir) {
        let op_str = match dir {
            VecShiftDir::Left => "<<",
            VecShiftDir::Right => ">>",
        };
        ctx.emit_line(&format!("{dst} = {a} {op_str} {amount};"));
    }

    fn emit_vec_unary_op(&self, dst: &str, a: &str, op: &GpuVecUnaryOp) -> String {
        match op {
            GpuVecUnaryOp::Neg => format!("{dst} = -{a};"),
            GpuVecUnaryOp::Abs => format!("{dst} = fabs({a});"),
            GpuVecUnaryOp::Sqrt => format!("{dst} = sqrt({a});"),
            GpuVecUnaryOp::Rsqrt => format!("{dst} = rsqrt({a});"),
            GpuVecUnaryOp::Recip => format!("{dst} = 1.0/{a};"),
            GpuVecUnaryOp::Round => format!("{dst} = round({a});"),
            GpuVecUnaryOp::Floor => format!("{dst} = floor({a});"),
            GpuVecUnaryOp::Ceil  => format!("{dst} = ceil({a});"),
            GpuVecUnaryOp::IntToFloat => format!("{dst} = int_as_float({a});"),
        }
    }

    fn emit_transcendental(&self, dst: &str, src: &str, func: &TranscendentalFn) -> String {
        let fn_str = match func {
            TranscendentalFn::Exp => "exp2",
            TranscendentalFn::Log => "log2",
            TranscendentalFn::Tanh => "tanh",
            _ => "mov",
        };
        format!("{dst} = {fn_str}({src});")
    }

    fn emit_broadcast_const(&self, ctx: &mut GpuLowerContext, dst: &str, val: f32) {
        ctx.emit_line(&format!("{dst} = {val}f;"));
    }

    fn emit_broadcast_mem_load(&self, ctx: &mut GpuLowerContext, dst: &str, base: &str, offset: &str) {
        ctx.emit_line(&format!("{dst} = *(({base}) + ({offset})/4);"));
    }

    fn emit_broadcast_vreg(&self, dst: &str, src: &str) -> String {
        format!("{dst} = {src};")
    }

    fn emit_sync(&self, ctx: &mut GpuLowerContext, kind: BarrierKind) {
        match kind {
            BarrierKind::WarpSync | BarrierKind::BlockSync => {
                ctx.emit_line("__syncthreads();");
            }
            BarrierKind::MemFence => {
                ctx.emit_line("__threadfence_block();");
            }
        }
    }

    fn emit_loop_begin(
        &self,
        ctx: &mut GpuLowerContext,
        counter: &str,
        offset: &str,
        bound_expr: &str,
        _label_id: u32,
        step_bytes: usize,
        _scratch_pred: &str,
        _scratch_bound: &str,
    ) {
        ctx.emit_line(&format!("for (int {counter} = 0, {offset} = 0; {counter} < {bound_expr}; {counter}++, {offset} += {step_bytes}) {{"));
        *ctx.indent += 1;
    }

    fn emit_loop_end(
        &self,
        ctx: &mut GpuLowerContext,
        _counter: &str,
        _offset: &str,
        _label_id: u32,
        _step_bytes: usize,
        indent: &mut usize,
    ) {
        *indent = indent.saturating_sub(1);
        ctx.emit_line("}");
    }

    fn emit_conditional_skip(
        &self,
        ctx: &mut GpuLowerContext,
        mask: &str,
        _skip_id: u32,
        _scratch_pred: &str,
    ) {
        ctx.emit_line(&format!("if ({mask} != 0.0) {{"));
        *ctx.indent += 1;
    }

    fn emit_conditional_skip_end(&self, ctx: &mut GpuLowerContext, _skip_id: u32) {
        *ctx.indent = ctx.indent.saturating_sub(1);
        ctx.emit_line("}");
    }

    fn async_copy_support(&self) -> AsyncCopyLevel {
        if self.gfx_arch >= 950 {
            AsyncCopyLevel::GlobalLoadLds
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
        if self.gfx_arch >= 950 {
            ctx.emit_line(&format!("// gfx950 GLOBAL_LOAD_LDS: {src} -> {dst} ({size} bytes)"));
            ctx.emit_line(&format!("global_load_lds {dst}, {src}, off;"));
            Ok(())
        } else {
            Err(CompilerError::CodegenViolation(
                format!("AsyncCopy: HIP gfx{} does not support async copy (requires gfx950+)", self.gfx_arch),
            ))
        }
    }

    fn emit_async_wait(
        &self,
        ctx: &mut GpuLowerContext,
        _scratch_pred: &str,
    ) -> Result<(), CompilerError> {
        ctx.emit_line("s_waitcnt lgkmcnt(0);");
        Ok(())
    }

    fn emit_warp_reduce(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        src: &str,
        op: &ReduceOp,
    ) -> Result<(), CompilerError> {
        let op_str = match op {
            ReduceOp::Sum => "add", ReduceOp::Max => "max",
            ReduceOp::Min => "min", ReduceOp::Prod => "mul",
            ReduceOp::LogSum => "add",
        };
        let ws = self.wave_size;
        ctx.emit_line(&format!("float hip_shfl_tmp; {dst} = {src};"));
        let max_delta = ws / 2;
        let mut delta = max_delta;
        while delta >= 1 {
            ctx.emit_line(&format!("hip_shfl_tmp = __shfl_xor({dst}, {delta});"));
            ctx.emit_line(&format!("{dst} = {op_str}({dst}, hip_shfl_tmp);"));
            delta /= 2;
        }
        Ok(())
    }

    fn has_tensor_cores(&self) -> bool {
        self.gfx_arch >= 908
    }

    fn emit_tile_mma(
        &self,
        ctx: &mut GpuLowerContext,
        c: &str,
        a: &str,
        b: &str,
    ) -> Result<(), CompilerError> {
        if self.gfx_arch >= 950 {
            ctx.emit_line(&format!("v_mfma_f32_32x32x16_f16 {c}, {a}, {b}, {c};"));
        } else if self.gfx_arch >= 908 {
            ctx.emit_line(&format!("v_mfma_f32_16x16x16_f16 {c}, {a}, {b}, {c};"));
        } else {
            return Err(CompilerError::CodegenViolation(
                format!("TileMma: HIP gfx{} lacks MFMA; RDNA GPUs require gfx908+", self.gfx_arch),
            ));
        }
        Ok(())
    }

    fn emit_shared_mem_alloc(&self, ctx: &mut GpuLowerContext, name: &str, bytes: usize) {
        ctx.emit_line(&format!("__shared__ float {name}[{}];", bytes / 4));
    }

    fn emit_gpr_binop(&self, dst: &str, a: &str, b: &str, op: &GpuGprBinOp) -> String {
        let op_str = match op {
            GpuGprBinOp::Add => "+", GpuGprBinOp::Sub => "-",
            GpuGprBinOp::Mul => "*", GpuGprBinOp::Div => "/",
            GpuGprBinOp::And => "&", GpuGprBinOp::Or => "|",
            GpuGprBinOp::Xor => "^", GpuGprBinOp::Shl => "<<",
            GpuGprBinOp::Shr => ">>",
        };
        format!("{dst} = {a} {op_str} {b};")
    }

    fn emit_gpr_load_imm(&self, ctx: &mut GpuLowerContext, dst: &str, value: u32) {
        ctx.emit_line(&format!("{dst} = {value};"));
    }

    fn emit_prefetch(&self, ctx: &mut GpuLowerContext, addr: &str) -> Result<(), CompilerError> {
        ctx.emit_line(&format!("// HIP prefetch: {addr}"));
        Ok(())
    }

    fn emit_mem_fence(&self, ctx: &mut GpuLowerContext) {
        ctx.emit_line("__threadfence_block();");
    }

    fn emit_warp_prng(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        seed: &str,
    ) -> Result<(), CompilerError> {
        ctx.emit_line(&format!("// HIP PRNG: xorwow from seed {seed}"));
        ctx.emit_line(&format!("{{ unsigned int tid = threadIdx.x; {dst} = (float)(tid ^ {seed}); }}"));
        Ok(())
    }

    fn ptx_version_string(&self) -> Option<&'static str> { None }

    fn code_format(&self) -> crate::compiler::codegen::CodeFormat {
        crate::compiler::codegen::CodeFormat::Hip
    }

    fn emit_shared_mem_addr(&self, ctx: &mut GpuLowerContext, dst: &str) {
        ctx.emit_line(&format!("{dst} = (float*)smem;"));
    }

    // ── 控制流分支 ──

    fn emit_conditional_exit(&self, ctx: &mut GpuLowerContext, cond: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ({cond}) return;"));
    }

    fn emit_branch_if_ptr_nonnull(&self, ctx: &mut GpuLowerContext, ptr: &str, target_label: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ({ptr} != 0) goto batch_mode_{target_label};"));
    }

    fn emit_branch_if_gpr_zero(&self, ctx: &mut GpuLowerContext, value: &str, label: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ({value} == 0) goto {label};"));
    }

    fn emit_branch_if_gpr_lt_u(&self, ctx: &mut GpuLowerContext, a: &str, b: &str, label: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ((unsigned){a} < (unsigned){b}) goto {label};"));
    }

    fn emit_unconditional_branch(&self, ctx: &mut GpuLowerContext, label: &str) {
        ctx.emit_line(&format!("goto {label};"));
    }

    fn emit_conditional_skip_end_with_indent(&self, ctx: &mut GpuLowerContext, skip_id: u32, indent: &mut usize) {
        *indent = indent.saturating_sub(1);
        ctx.emit_line(&format!("SKIP_{skip_id}:;"));
    }

    // ── GPR 条件操作 ──

    fn emit_gpr_cond_is_null(
        &self, ctx: &mut GpuLowerContext, ptr: &str, _action: &GprBranchAction,
        _scratch_pred: &str, _epilogue_label: &str, _skip_id: u32,
    ) {
        ctx.emit_line(&format!("if ({ptr} != 0UL) {{"));
    }

    fn emit_gpr_cond_is_nonnull(
        &self, ctx: &mut GpuLowerContext, ptr: &str, _action: &GprBranchAction,
        _scratch_pred: &str, _epilogue_label: &str, _skip_id: u32,
    ) {
        ctx.emit_line(&format!("if ({ptr} == 0UL) {{"));
    }

    fn emit_gpr_cond_cmp_eq(
        &self, ctx: &mut GpuLowerContext, a: &str, imm: u64, action: &GprBranchAction,
        _scratch_pred: &str, epilogue_label: &str, _skip_id: u32,
    ) {
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("if ({a} == {imm}u) {{"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("if ({a} == {imm}u) goto {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_cmp_lt_u(
        &self, ctx: &mut GpuLowerContext, a: &str, imm: u64, action: &GprBranchAction,
        _scratch_pred: &str, epilogue_label: &str, _skip_id: u32,
    ) {
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("if ({a} < {imm}u) {{"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("if ({a} < {imm}u) goto {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_bit_clear(
        &self, ctx: &mut GpuLowerContext, bitmap: &str, mask: u32, action: &GprBranchAction,
        _scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("if (({bitmap} & {mask}u) == 0u) goto SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("if (({bitmap} & {mask}u) == 0u) goto {epilogue_label};"));
            }
        }
    }

    fn emit_gpr_cond_bit_set(
        &self, ctx: &mut GpuLowerContext, bitmap: &str, mask: u32, action: &GprBranchAction,
        _scratch_pred: &str, epilogue_label: &str, skip_id: u32,
    ) {
        match action {
            GprBranchAction::Skip(_) => {
                ctx.emit_line(&format!("if (({bitmap} & {mask}u) != 0u) goto SKIP_{skip_id};"));
            }
            GprBranchAction::Exit(_) => {
                ctx.emit_line(&format!("if (({bitmap} & {mask}u) != 0u) goto {epilogue_label};"));
            }
        }
    }

    // ── 共享内存异步操作 ──

    fn emit_shared_mem_async_store(
        &self, _ctx: &mut GpuLowerContext, _name: &str, _offset_str: &str,
        _src_reg: &str, _copy_bytes: usize,
    ) -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation(
            "SharedMemAsyncStore: HIP does not support cp.async equivalent".into(),
        ))
    }

    fn emit_shared_mem_async_wait_group(
        &self, _ctx: &mut GpuLowerContext, _n: u32,
    ) -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation(
            "SharedMemAsyncWaitGroup: HIP does not support async wait group".into(),
        ))
    }

    // ── Warp 屏障 ──

    fn emit_warp_barrier_arrive(
        &self, _ctx: &mut GpuLowerContext, _barrier_symbol: &str, _tx_bytes: usize,
    ) {
        // HIP: NOP (no warp specialization)
    }

    fn emit_warp_barrier_wait(
        &self, _ctx: &mut GpuLowerContext, _barrier_symbol: &str, _parity: u32,
        _scratch_pred: &str,
    ) {
        // HIP: NOP
    }

    // ── Warp Role ──

    fn emit_warp_role_declare(
        &self, _ctx: &mut GpuLowerContext, _role: u32,
    ) -> Result<(), CompilerError> {
        // HIP: no warp specialization support
        Ok(())
    }
}

// ── Metal 方言实现 ──

/// Apple Metal 方言
