/// （避免与 `self.dialect` 的 borrow 冲突）。
pub struct GpuLowerContext<'a> {
    /// IR 输出 buffer
    pub ir: &'a mut String,
    /// 当前缩进层级
    pub indent: &'a mut usize,
    /// scratch 向量寄存器名（如 ["%fs0", "%fs1", "%fs2"]）
    pub scratch_vec_names: &'a [&'static str],
    /// scratch GPR 名（如 ["%rs0", "%rs1", "%rs_bound"]）
    pub scratch_gpr_names: &'a [&'static str],
    /// scratch 谓词名（如 ["%ps0", "%ps1"]）
    pub scratch_pred_names: &'a [&'static str],
}

impl<'a> GpuLowerContext<'a> {
    /// 向 IR buffer 追加一行（带缩进）。
    pub fn emit_line(&mut self, line: &str) {
        for _ in 0..*self.indent {
            self.ir.push_str("  ");
        }
        self.ir.push_str(line);
        self.ir.push('\n');
    }
}

// ── 方言能力查询 ──

/// 异步拷贝支持级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncCopyLevel {
    /// 不支持异步拷贝
    None,
    /// cp.async 128B 级别 (SM80+)
    CpAsync,
    /// TMA bulk copy (SM90+)
    TmaBulk,
    /// GLOBAL_LOAD_LDS (gfx950+)
    GlobalLoadLds,
}

/// 同步屏障类型
#[derive(Debug, Clone, Copy)]
pub enum BarrierKind {
    /// warp 级同步
    WarpSync,
    /// block 级同步
    BlockSync,
    /// 内存栅栏
    MemFence,
}

/// GPR 二元操作（从 VecOp 子集映射）
#[derive(Debug, Clone, Copy)]
pub enum GpuGprBinOp {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

/// 向量一元操作类型
pub enum GpuVecUnaryOp {
    Neg,
    Abs,
    Sqrt,
    Rsqrt,
    Recip,
    Round,
    Floor,
    Ceil,
    IntToFloat,
}

// ── GPU Backend Dialect Trait ──

/// GPU 后端方言 trait — 每种 GPU ISA 实现此 trait。
///
/// 所有方法接收 `&mut GpuLowerContext` 以写入 IR buffer。
/// 方言本身不持有可变状态。
pub trait GpuBackendDialect {
    /// 方言名称 ("PTX" / "HIP" / "Metal")
    fn name(&self) -> &'static str;

    /// NVIDIA SM 版本（仅 PTX）
    fn sm_version(&self) -> Option<u32>;

    /// AMD gfx 架构版本（仅 HIP）
    fn gfx_arch(&self) -> Option<u32>;

    /// AMD wave 大小（仅 HIP）
    fn wave_size(&self) -> Option<u32>;

    // ── 寄存器命名 ──

    /// GPR 32-bit 前缀 (PTX: "%r", HIP/Metal: "r_")
    fn gpr32_prefix(&self) -> &'static str;
    /// GPR 64-bit 前缀 (PTX: "%rd", HIP/Metal: "rd_")
    fn gpr64_prefix(&self) -> &'static str;
    /// 向量寄存器前缀 (PTX: "%f", HIP/Metal: "f_")
    fn vec_prefix(&self) -> &'static str;
    /// 谓词/掩码寄存器前缀 (PTX: "%p", HIP/Metal: "p_")
    fn mask_prefix(&self) -> &'static str;
    /// Tile 寄存器前缀 (PTX: "%t", HIP/Metal: "t_")
    fn tile_prefix(&self) -> &'static str;

    // ── 共享内存声明 ──

    /// 声明共享内存
    fn emit_shared_mem_decl(&self, ctx: &mut GpuLowerContext, name: &str, size_bytes: usize);

    // ── 类型系统 ──

    /// 是否为 PTX 方言（ISA 语法为 PTX 汇编）
    fn is_ptx(&self) -> bool {
        self.sm_version().is_some()
    }

    /// 是否为 C 风格方言（HIP/Metal）
    fn is_c_style(&self) -> bool {
        !self.is_ptx()
    }

    // ── 简单算术（返回代码行字符串） ──

    /// Mov 指令
    fn mov_f32(&self, dst: &str, src: &str) -> String;

    /// FMA 指令
    fn fma_f32(&self, dst: &str, a: &str, b: &str, acc: &str) -> String;

    /// 累加（+=）
    fn accumulate(&self, acc: &str, src: &str) -> String;

    /// Add ptr (64-bit)
    fn add_ptr(&self, dst: &str, base: &str, offset: &str) -> String;

    /// Add GPR (32-bit)
    fn add_gpr(&self, dst: &str, a: &str, b: &str) -> String;

    // ── 向量内存操作 ──

    /// 全局 load (从 base + offset 加载 f32)
    fn emit_global_load_f32(&self, ctx: &mut GpuLowerContext, dst: &str, base: &str, offset: &str);

    /// 全局 store (向 base + offset 存储 f32)
    fn emit_global_store_f32(&self, ctx: &mut GpuLowerContext, src: &str, base: &str, offset: &str);

    /// 从 ABI 参数加载指针
    fn emit_load_abi_ptr(&self, ctx: &mut GpuLowerContext, dst: &str, param_name: &str);

    /// 共享内存 load
    fn emit_shared_load(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        name: &str,
        offset: &str,
        dtype_str: &str,
    );

    /// 共享内存 store
    fn emit_shared_store(
        &self,
        ctx: &mut GpuLowerContext,
        src: &str,
        name: &str,
        offset: &str,
        dtype_str: &str,
    );

    // ── 向量算术 ──

    /// 向量二元操作
    fn emit_vec_binop(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        a: &str,
        b: &str,
        op: &VecOp,
    );

    /// 向量立即数移位
    fn emit_vec_shift_imm(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        a: &str,
        amount: u32,
        dir: &VecShiftDir,
    );

    /// 向量一元操作
    fn emit_vec_unary_op(&self, dst: &str, a: &str, op: &GpuVecUnaryOp) -> String;

    /// 超越函数
    fn emit_transcendental(&self, dst: &str, src: &str, func: &TranscendentalFn) -> String;

    // ── 标量/广播操作 ──

    /// 广播常量到寄存器
    fn emit_broadcast_const(&self, ctx: &mut GpuLowerContext, dst: &str, val: f32);

    /// 广播内存加载
    fn emit_broadcast_mem_load(&self, ctx: &mut GpuLowerContext, dst: &str, base: &str, offset: &str);

    /// 广播 VReg (mov)
    fn emit_broadcast_vreg(&self, dst: &str, src: &str) -> String;

    // ── 同步/屏障 ──

    /// 发射同步指令
    fn emit_sync(&self, ctx: &mut GpuLowerContext, kind: BarrierKind);

    // ── 循环 ──

    /// 循环头：初始化 counter/offset 并生成循环入口标签
    fn emit_loop_begin(
        &self,
        ctx: &mut GpuLowerContext,
        counter: &str,
        offset: &str,
        bound_expr: &str,
        label_id: u32,
        step_bytes: usize,
        scratch_pred: &str,
        scratch_bound: &str,
    );

    /// 循环尾：递增 counter/offset 并跳回
    fn emit_loop_end(
        &self,
        ctx: &mut GpuLowerContext,
        counter: &str,
        offset: &str,
        label_id: u32,
        step_bytes: usize,
        indent: &mut usize,
    );

    // ── 条件控制流 ──

    /// 条件跳过（mask == 0 时跳过）
    fn emit_conditional_skip(
        &self,
        ctx: &mut GpuLowerContext,
        mask: &str,
        skip_id: u32,
        scratch_pred: &str,
    );

    /// 条件跳过结束
    fn emit_conditional_skip_end(
        &self,
        ctx: &mut GpuLowerContext,
        skip_id: u32,
    );

    // ── 异步拷贝 ──

    /// 异步拷贝支持级别
    fn async_copy_support(&self) -> AsyncCopyLevel;

    /// 异步拷贝（SM80+ cp.async / SM90+ TMA / gfx950 GLOBAL_LOAD_LDS）
    fn emit_async_copy(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        src: &str,
        size: usize,
    ) -> Result<(), CompilerError>;

    /// 异步等待
    fn emit_async_wait(
        &self,
        ctx: &mut GpuLowerContext,
        scratch_pred: &str,
    ) -> Result<(), CompilerError>;

    // ── 归约 ──

    /// Warp 级归约
    fn emit_warp_reduce(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        src: &str,
        op: &ReduceOp,
    ) -> Result<(), CompilerError>;

    // ── Tile/Tensor Core ──

    /// 是否有 Tensor Core
    fn has_tensor_cores(&self) -> bool;

    /// TileMma 指令
    fn emit_tile_mma(
        &self,
        ctx: &mut GpuLowerContext,
        c: &str,
        a: &str,
        b: &str,
    ) -> Result<(), CompilerError>;

    // ── SharedMem 操作（命名 shared memory） ──

    /// 命名共享内存分配
    fn emit_shared_mem_alloc(&self, ctx: &mut GpuLowerContext, name: &str, bytes: usize);

    // ── GPR 操作 ──

    /// GPR 二元操作
    fn emit_gpr_binop(&self, dst: &str, a: &str, b: &str, op: &GpuGprBinOp) -> String;

    /// GPR 加载立即数
    fn emit_gpr_load_imm(&self, ctx: &mut GpuLowerContext, dst: &str, value: u32);

    // ── 特殊操作 ──

    /// Prefetch
    fn emit_prefetch(&self, ctx: &mut GpuLowerContext, addr: &str) -> Result<(), CompilerError>;

    /// 内存栅栏
    fn emit_mem_fence(&self, ctx: &mut GpuLowerContext);

    /// Warp PRNG
    fn emit_warp_prng(
        &self,
        ctx: &mut GpuLowerContext,
        dst: &str,
        seed: &str,
    ) -> Result<(), CompilerError>;

    // ── Prologue/Epilogue 查询 ──

    /// PTX 版本字符串
    fn ptx_version_string(&self) -> Option<&'static str>;

    /// 生成 prologue 后更新 CodeFormat
    fn code_format(&self) -> crate::compiler::codegen::CodeFormat;

    // ── LoadPtr shared memory 符号 ──

    /// 获取共享内存符号地址
    fn emit_shared_mem_addr(&self, ctx: &mut GpuLowerContext, dst: &str);

    // ── 控制流分支 ──

    /// 条件退出（非零则 ret/return）
    fn emit_conditional_exit(
        &self,
        ctx: &mut GpuLowerContext,
        cond: &str,
        scratch_pred: &str,
    );

    /// 指针非空则跳转
    fn emit_branch_if_ptr_nonnull(
        &self,
        ctx: &mut GpuLowerContext,
        ptr: &str,
        target_label: &str,
        scratch_pred: &str,
    );

    /// GPR 为零则跳转
    fn emit_branch_if_gpr_zero(
        &self,
        ctx: &mut GpuLowerContext,
        value: &str,
        label: &str,
        scratch_pred: &str,
    );

    /// GPR 无符号小于则跳转
    fn emit_branch_if_gpr_lt_u(
        &self,
        ctx: &mut GpuLowerContext,
        a: &str,
        b: &str,
        label: &str,
        scratch_pred: &str,
    );

    /// 无条件跳转
    fn emit_unconditional_branch(
        &self,
        ctx: &mut GpuLowerContext,
        label: &str,
    );

    /// ConditionalSkipEnd: 恢复缩进 + 发射跳过标签
    fn emit_conditional_skip_end_with_indent(
        &self,
        ctx: &mut GpuLowerContext,
        skip_id: u32,
        indent: &mut usize,
    );

    // ── GPR 条件操作 (GprCondAction) ──

    /// GPR IsNull 条件分支
    fn emit_gpr_cond_is_null(
        &self,
        ctx: &mut GpuLowerContext,
        ptr: &str,
        action: &GprBranchAction,
        scratch_pred: &str,
        epilogue_label: &str,
        skip_id: u32,
    );

    /// GPR IsNonNull 条件分支
    fn emit_gpr_cond_is_nonnull(
        &self,
        ctx: &mut GpuLowerContext,
        ptr: &str,
        action: &GprBranchAction,
        scratch_pred: &str,
        epilogue_label: &str,
        skip_id: u32,
    );

    /// GPR CmpEq 条件分支
    fn emit_gpr_cond_cmp_eq(
        &self,
        ctx: &mut GpuLowerContext,
        a: &str,
        imm: u64,
        action: &GprBranchAction,
        scratch_pred: &str,
        epilogue_label: &str,
        skip_id: u32,
    );

    /// GPR CmpLtU 条件分支
    fn emit_gpr_cond_cmp_lt_u(
        &self,
        ctx: &mut GpuLowerContext,
        a: &str,
        imm: u64,
        action: &GprBranchAction,
        scratch_pred: &str,
        epilogue_label: &str,
        skip_id: u32,
    );

    /// GPR BitClear 条件分支
    fn emit_gpr_cond_bit_clear(
        &self,
        ctx: &mut GpuLowerContext,
        bitmap: &str,
        mask: u32,
        action: &GprBranchAction,
        scratch_pred: &str,
        epilogue_label: &str,
        skip_id: u32,
    );

    /// GPR BitSet 条件分支
    fn emit_gpr_cond_bit_set(
        &self,
        ctx: &mut GpuLowerContext,
        bitmap: &str,
        mask: u32,
        action: &GprBranchAction,
        scratch_pred: &str,
        epilogue_label: &str,
        skip_id: u32,
    );

    // ── 共享内存异步操作 ──

    /// 异步共享内存 store (SM80+ cp.async)
    fn emit_shared_mem_async_store(
        &self,
        ctx: &mut GpuLowerContext,
        name: &str,
        offset_str: &str,
        src_reg: &str,
        copy_bytes: usize,
    ) -> Result<(), CompilerError>;

    /// 异步等待组 (SM80+ cp.async.wait_group)
    fn emit_shared_mem_async_wait_group(
        &self,
        ctx: &mut GpuLowerContext,
        n: u32,
    ) -> Result<(), CompilerError>;

    // ── Warp 屏障 ──

    /// Warp barrier arrive (SM90+ mbarrier.arrive.expect_tx)
    fn emit_warp_barrier_arrive(
        &self,
        ctx: &mut GpuLowerContext,
        barrier_symbol: &str,
        tx_bytes: usize,
    );

    /// Warp barrier wait (SM90+ mbarrier.try_wait.parity)
    fn emit_warp_barrier_wait(
        &self,
        ctx: &mut GpuLowerContext,
        barrier_symbol: &str,
        parity: u32,
        scratch_pred: &str,
    );

    // ── Warp Role ──

    /// Warp 角色声明 (SM90+ setmaxnreg)
    fn emit_warp_role_declare(
        &self,
        ctx: &mut GpuLowerContext,
        role: u32,
    ) -> Result<(), CompilerError>;
}

// ── PTX 方言实现 ──

/// NVIDIA PTX 方言
