pub struct MetalDialect {
    gpu_family: u32,
}

impl MetalDialect {
    pub fn new(gpu_family: u32) -> Self {
        Self { gpu_family }
    }
}

impl GpuBackendDialect for MetalDialect {
    fn name(&self) -> &'static str { "Metal" }
    fn sm_version(&self) -> Option<u32> { None }
    fn gfx_arch(&self) -> Option<u32> { None }
    fn wave_size(&self) -> Option<u32> { None }

    fn gpr32_prefix(&self) -> &'static str { "r_" }
    fn gpr64_prefix(&self) -> &'static str { "rd_" }
    fn vec_prefix(&self) -> &'static str { "f_" }
    fn mask_prefix(&self) -> &'static str { "p_" }
    fn tile_prefix(&self) -> &'static str { "t_" }

    fn emit_shared_mem_decl(&self, ctx: &mut GpuLowerContext, _name: &str, size_bytes: usize) {
        ctx.emit_line(&format!("threadgroup float smem[{}];", size_bytes / 4));
    }

    fn mov_f32(&self, dst: &str, src: &str) -> String {
        format!("{dst} = {src};")
    }

    fn fma_f32(&self, dst: &str, a: &str, b: &str, acc: &str) -> String {
        format!("{dst} = fma({a}, {b}, {acc});")
    }

    fn accumulate(&self, acc: &str, src: &str) -> String {
        format!("{acc} += {src};")
    }

    fn add_ptr(&self, dst: &str, base: &str, offset: &str) -> String {
        format!("{dst} = {base} + ({offset});")
    }

    fn add_gpr(&self, dst: &str, a: &str, b: &str) -> String {
        format!("{dst} = {a} + {b};")
    }

    fn emit_global_load_f32(&self, ctx: &mut GpuLowerContext, dst: &str, base: &str, offset: &str) {
        ctx.emit_line(&format!("{dst} = {base}[({offset})/4];"));
    }

    fn emit_global_store_f32(&self, ctx: &mut GpuLowerContext, src: &str, base: &str, offset: &str) {
        ctx.emit_line(&format!("{base}[({offset})/4] = {src};"));
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
        ctx.emit_line(&format!("{dst} = {base}[({offset})/4];"));
    }

    fn emit_broadcast_vreg(&self, dst: &str, src: &str) -> String {
        format!("{dst} = {src};")
    }

    fn emit_sync(&self, ctx: &mut GpuLowerContext, kind: BarrierKind) {
        match kind {
            BarrierKind::WarpSync | BarrierKind::BlockSync => {
                ctx.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            BarrierKind::MemFence => {
                ctx.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
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
        AsyncCopyLevel::None
    }

    fn emit_async_copy(
        &self,
        _ctx: &mut GpuLowerContext,
        _dst: &str,
        _src: &str,
        _size: usize,
    ) -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation(
            "AsyncCopy: Metal does not support async copy".into(),
        ))
    }

    fn emit_async_wait(
        &self,
        _ctx: &mut GpuLowerContext,
        _scratch_pred: &str,
    ) -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation(
            "AsyncWait: Metal does not support async wait".into(),
        ))
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
        ctx.emit_line(&format!("{dst} = warp_reduce_{op_str}({src});"));
        Ok(())
    }

    fn has_tensor_cores(&self) -> bool {
        // Metal simdgroup_matrix_multiply available on Apple7+ GPUs
        self.gpu_family >= 7
    }

    fn emit_tile_mma(
        &self,
        _ctx: &mut GpuLowerContext,
        _c: &str,
        _a: &str,
        _b: &str,
    ) -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation(
            "TileMma: Metal simdgroup_matrix_multiply not yet implemented".into(),
        ))
    }

    fn emit_shared_mem_alloc(&self, ctx: &mut GpuLowerContext, name: &str, bytes: usize) {
        ctx.emit_line(&format!("threadgroup float {name}[{}];", bytes / 4));
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
        ctx.emit_line(&format!("// Metal prefetch: {addr}"));
        Ok(())
    }

    fn emit_mem_fence(&self, ctx: &mut GpuLowerContext) {
        ctx.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    }

    fn emit_warp_prng(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        seed: &str,
    ) -> Result<(), CompilerError> {
        ctx.emit_line(&format!("// Metal PRNG: from seed {seed}"));
        ctx.emit_line(&format!("{{ uint tid = threadIdx.x; {dst} = (float)(tid ^ {seed}); }}"));
        Ok(())
    }

    fn ptx_version_string(&self) -> Option<&'static str> { None }

    fn code_format(&self) -> crate::compiler::codegen::CodeFormat {
        crate::compiler::codegen::CodeFormat::Msl
    }

    fn emit_shared_mem_addr(&self, ctx: &mut GpuLowerContext, dst: &str) {
        ctx.emit_line(&format!("{dst} = (device float*)smem;"));
    }

    // ── 控制流分支 ──

    fn emit_conditional_exit(&self, ctx: &mut GpuLowerContext, cond: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ({cond}) return;"));
    }

    fn emit_branch_if_ptr_nonnull(&self, ctx: &mut GpuLowerContext, ptr: &str, target_label: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ({ptr} != 0) {{ goto batch_mode_{target_label}; }}"));
    }

    fn emit_branch_if_gpr_zero(&self, ctx: &mut GpuLowerContext, value: &str, label: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ({value} == 0) {{ goto {label}; }}"));
    }

    fn emit_branch_if_gpr_lt_u(&self, ctx: &mut GpuLowerContext, a: &str, b: &str, label: &str, _scratch_pred: &str) {
        ctx.emit_line(&format!("if ({a} < {b}) {{ goto {label}; }}"));
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
            "SharedMemAsyncStore: Metal does not support cp.async equivalent".into(),
        ))
    }

    fn emit_shared_mem_async_wait_group(
        &self, _ctx: &mut GpuLowerContext, _n: u32,
    ) -> Result<(), CompilerError> {
        Err(CompilerError::CodegenViolation(
            "SharedMemAsyncWaitGroup: Metal does not support async wait group".into(),
        ))
    }

    // ── Warp 屏障 ──

    fn emit_warp_barrier_arrive(
        &self, _ctx: &mut GpuLowerContext, _barrier_symbol: &str, _tx_bytes: usize,
    ) {
        // Metal: NOP
    }

    fn emit_warp_barrier_wait(
        &self, _ctx: &mut GpuLowerContext, _barrier_symbol: &str, _parity: u32,
        _scratch_pred: &str,
    ) {
        // Metal: NOP
    }

    // ── Warp Role ──

    fn emit_warp_role_declare(
        &self, _ctx: &mut GpuLowerContext, _role: u32,
    ) -> Result<(), CompilerError> {
        // Metal: no warp specialization support
        Ok(())
    }
}

// ── GpuDialect → Box<dyn GpuBackendDialect> 转换 ──

use super::gpu_lower::GpuDialect;

/// 从 GpuDialect enum 构造对应的 trait 对象。
///
/// 保持向后兼容：所有外部调用者仍使用 `GpuDialect` enum，
/// `GpuLower` 内部通过此函数获取 trait 对象。
pub fn make_gpu_dialect(dialect: &GpuDialect) -> Box<dyn GpuBackendDialect> {
    match dialect {
        GpuDialect::Ptx { sm_version } => Box::new(PtxDialect::new(*sm_version)),
        GpuDialect::Hip { gfx_arch, wave_size } => Box::new(HipDialect::new(*gfx_arch, *wave_size)),
        GpuDialect::Metal { gpu_family } => Box::new(MetalDialect::new(*gpu_family)),
    }
}
