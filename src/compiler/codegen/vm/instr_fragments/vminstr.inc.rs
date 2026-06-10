
#[derive(Debug, Clone)]
pub enum VmInstr {
    // ── 内存操作 ──

    /// SIMD 向量加载: dst = [base + offset]
    VecLoad {
        dst: VRegId,
        base: VRegId,
        offset: OffsetExpr,
        width: SimdWidth,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// SIMD 向量存储: [base + offset] = src
    VecStore {
        base: VRegId,
        offset: OffsetExpr,
        src: VRegId,
        width: SimdWidth,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// 累加器窄化: dst = narrow(src) (REQ-DTYPE-006)
    /// 将 acc_dtype 的向量窄化为 dtype（如 F32→BF16）。
    VecNarrow {
        dst: VRegId,
        src: VRegId,
        dst_dtype: crate::compiler::trace::QuantPrecision,
        src_dtype: crate::compiler::trace::QuantPrecision,
        width: SimdWidth,
    },
    /// 向量宽化: dst = widen(src) (REQ-DTYPE-003)
    ///将窄 dtype 的向量宽化为宽 dtype（如 BF16→F32）。
    VecWiden {
        dst: VRegId,
        src: VRegId,
        dst_dtype: crate::compiler::trace::QuantPrecision,
        src_dtype: crate::compiler::trace::QuantPrecision,
        width: SimdWidth,
    },
    /// 标量广播到 SIMD 向量: dst = broadcast(scalar)
    Broadcast {
        dst: VRegId,
        src: ScalarExpr,
        width: SimdWidth,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// 寄存器拷贝: dst = src (mov-elimination 友好)
    /// x86: vmovaps; ARM: mov; GPU: mov.f32
    Mov {
        dst: VRegId,
        src: VRegId,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// 指针加载: dst = load_ptr(src)
    LoadPtr {
        dst: VRegId,
        src: PtrExpr,
    },

    // ── 标量操作 (Gather 索引) ──

    /// 标量 f32 加载: dst(xmm) = *(f32*)(base + offset)
    /// 用于从 indices 数组读取 token ID。dst 声明为 VRegKind::Scalar。
    ScalarLoad {
        dst: VRegId,
        base: VRegId,
        offset: OffsetExpr,
    },
    /// 标量 f32 存储: *(f32*)(base + offset) = src(xmm)
    /// 用于 WriteLogits 写单个 logit。src 声明为 VRegKind::Scalar。
    ScalarStore {
        base: VRegId,
        src: VRegId,
        offset: OffsetExpr,
    },
    /// 从 Vec 寄存器 lane 0 存储单个 f32: *(f32*)(base + offset) = src.lane[0]
    /// src 声明为 VRegKind::Vec。用于 QuantGemm GEMV 逐元素输出。
    VecScalarStore {
        base: VRegId,
        src: VRegId,
        offset: OffsetExpr,
    },
    /// 标量 f32→int 转换 + stride 乘法: dst(gpr) = (int)src(xmm) * stride
    /// 用于将 token ID 转换为 embedding table 字节偏移。
    /// dst 声明为 VRegKind::Ptr，src 声明为 VRegKind::Scalar。
    ScalarToIndex {
        dst: VRegId,
        src: VRegId,
        stride: usize,
    },
    /// GPR 整数→标量 f32 转换: dst(xmm) = (float)src(gpr)
    /// 用于将运行时整数维度值（如 seq_len loop counter）转换为浮点数，
    /// 以便计算 1/N 缩放因子（MeanPool 等归约操作需要）。
    /// dst 声明为 VRegKind::Scalar，src 声明为 VRegKind::Ptr / VRegKind::Counter。
    IndexToScalar {
        dst: VRegId,
        src: VRegId,
    },

    /// GPR 整数 × stride: dst(gpr) = src(gpr) * stride
    /// 与 ScalarToIndex 不同，src 已经是 u32 整数（在 GPR 中），不需要 float→int 转换。
    /// 用于 Gather 中 token ID → embedding table 字节偏移。
    IntMulStride {
        dst: VRegId,
        src: VRegId,
        stride: usize,
    },

    /// 向量索引加载: 从 base + indices[i]*stride 加载 lanes 个元素到 dst 向量。
    /// indices 是一个 VReg 指向 u32 索引数组。
    /// 后端映射: x86 vgatherdps (AVX2/AVX-512) / ARM scalar loop
    GatherLoad {
        dst: VRegId,
        base: VRegId,
        indices: VRegId,
        stride: usize,
        width: SimdWidth,
    },

    /// 向量索引存储: 将 src 向量的 lanes 个元素按 indices 写入 base + indices[i]*stride。
    /// indices 是一个 VReg 指向 u32 索引数组。
    /// 后端映射: x86 vscatterdps (AVX-512) / ARM scalar loop
    ScatterStore {
        base: VRegId,
        indices: VRegId,
        src: VRegId,
        stride: usize,
        width: SimdWidth,
    },

    /// 行查表: 从 base + row_index * row_bytes 加载一行 SIMD 向量。
    /// 等价于 IntMulStride(row_index, row_bytes) + PtrAdd(base, offset) + VecLoad。
    /// row_index 是一个 GPR VReg (VRegKind::Ptr 或 Scalar)，包含行号。
    /// 后端映射: 组合 imul + add + vmovups
    TableLookup {
        dst: VRegId,
        base: VRegId,
        row_index: VRegId,
        row_bytes: usize,
        width: SimdWidth,
    },

    // ── 量化操作 (mxfp4 dequant) ──

    /// 单字节加载: dst(gpr) = zero_extend(*(u8*)(base + offset))
    /// 用于 mxfp4 e8m0 scale 字节和 packed data 字节加载。
    /// dst 声明为 VRegKind::Ptr (映射到 GPR)。
    ScalarByteLoad {
        dst: VRegId,
        base: VRegId,
        offset: OffsetExpr,
    },
    // ── 算术操作 (SIMD) ──

    /// 二元 SIMD: dst = a op b
    VecBinOp {
        dst: VRegId,
        a: VRegId,
        b: VRegId,
        op: VecOp,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// 立即数向量移位: dst = a >> amount (或 a << amount)
    /// 使用 ISA 原生 imm8 形式 (vpsrld/vpslld imm8)，避免通过广播寄存器
    /// 传递移位计数导致 vpsrld 误读低 64 位为巨大计数的 bug。
    VecShiftImm {
        dst: VRegId,
        a: VRegId,
        amount: u8,
        op: VecShiftDir,
        width: SimdWidth,
    },
    /// 一元 SIMD: dst = op(a)
    VecUnaryOp {
        dst: VRegId,
        a: VRegId,
        op: VecUnaryOp,
    },
    /// 逐元素向量比较: dst = (a pred b) ? all_ones : all_zeros
    /// 结果为位掩码向量，可被 ConditionalSkip / VecBinOp::And 等消费。
    VecCmp {
        dst: VRegId,
        a: VRegId,
        b: VRegId,
        pred: CmpPredicate,
    },
    /// 向量类型转换: dst = convert(src, from → to)
    /// 后端: F16C vcvtph2ps / ARM fcvtl / PTX cvt
    VecCast {
        dst: VRegId,
        src: VRegId,
        from_bits: u8,
        to_bits: u8,
    },
    /// Conditional select: dst[i] = (mask[i] != 0) ? true_val[i] : false_val[i], per-lane.
    /// x86_64: vblendvps (AVX2) / vblendmps{k} (AVX-512)
    /// AArch64: bsl (NEON) / sel (SVE)
    ConditionalSelect {
        dst: VRegId,
        mask: VRegId,
        true_val: VRegId,
        false_val: VRegId,
    },
    /// FMA: dst = acc + a × b
    Fma {
        dst: VRegId,
        acc: VRegId,
        a: VRegId,
        b: VRegId,
        dtype: crate::compiler::trace::QuantPrecision,
    },

    // ── 归约 ──

    /// 水平归约: dst(scalar broadcast) = reduce(src_vec, op)
    HReduce {
        dst: VRegId,
        src: VRegId,
        op: ReduceOp,
    },
    /// 累加: acc += src
    Accumulate {
        acc: VRegId,
        src: VRegId,
    },

    // ── 控制流 ──

    /// 循环开始
    LoopBegin {
        /// 循环计数器 VReg
        counter: VRegId,
        /// 关联的字节偏移 VReg (= counter × step_bytes, 自动维护)
        byte_offset: VRegId,
        /// 循环上界
        bound: BoundExpr,
        /// 每次迭代的字节步进
        step_bytes: usize,
    },
    /// 循环结束 (自动步进 counter 和 byte_offset)
    LoopEnd,

    /// 作用域开始 (Compound group 内 sub-op 隔离)
    /// scope_id 由 emit_scope 自动分配，用于 ScopedSpillAllocator 的 scope-based 回收
    ScopeBegin { scope_id: usize },
    /// 作用域结束 (恢复 ScopeBegin 时保存的寄存器)
    /// scope_id 必须与匹配的 ScopeBegin 一致
    ScopeEnd { scope_id: usize },

    /// §13 FP1: 条件跳过 (Gate-First 死神经元掩码)
    ConditionalSkip {
        /// 条件掩码 VReg (全零 = 跳过)
        mask: VRegId,
        /// 跳过的指令数
        skip_count: usize,
    },

    /// GPR NULL 指针检查 + 条件跳过: 如果 ptr == 0 则跳过后面 skip_count 条指令。
    ///
    /// 用于 mega-kernel 中运行时可选功能（如 SG detect/inject）：
    /// hook_ctx_ptr 为 NULL → 跳过整个 SG 代码块（零开销）。
    /// hook_ctx_ptr 非 NULL → 执行 SG 代码块。
    ///
    /// x86 lower: resolve ptr to GPR → TEST reg, reg → JZ skip_label。
    /// AArch64 lower: CBZ reg, skip_label。
    /// GPR 条件操作 — 统一条件跳过/退出 (REQ-VR-003)。
    /// 替代旧 GprSkipIfNull / GprBitTest / GprCmpExit。
    GprCondAction {
        cond: GprCondition,
        action: GprBranchAction,
    },

    // ── Tile/Matrix 操作 ──

    /// 配置 tile 寄存器 (AMX TILECFG / SME SMSTART)
    TileConfig {
        rows: usize,
        cols: usize,
        dtype: DType,
    },
    /// Tile MMA: c += a × b
    TileMma {
        c: VRegId,
        a: VRegId,
        b: VRegId,
    },
    /// 释放 tile 资源
    TileRelease,

    /// VP2INTERSECT: 稀疏掩码硬件交集 (AVX-512 / Granite Rapids+)
    /// dst_k0/dst_k1 = VP2INTERSECTD/Q(a, b)
    /// 输出两个 k-mask: 匹配 a 中元素的掩码 + 匹配 b 中元素的掩码
    Vp2Intersect {
        dst_k0: VRegId,
        dst_k1: VRegId,
        a: VRegId,
        b: VRegId,
    },

    // ── GPU 并行 ──

    /// Warp/Wave 级同步屏障
    WarpSync,
    /// 异步内存拷贝
    AsyncCopy {
        dst: VRegId,
        src: VRegId,
        size: usize,
    },
    /// 等待异步操作
    AsyncWait {
        handle: u32,
    },
    /// Shared memory 声明: .shared .align 4 .b8 name[bytes]
    SharedMemAlloc {
        name: String,
        bytes: usize,
    },
    /// Register → Shared memory store: st.shared.{dtype} [name + offset], %r
    SharedMemStore {
        name: String,
        dst_offset: OffsetExpr,
        src: VRegId,
        width: SimdWidth,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// Shared memory → Register load: ld.shared.{dtype} %r, [name + offset]
    SharedMemLoad {
        dst: VRegId,
        name: String,
        src_offset: OffsetExpr,
        width: SimdWidth,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// 异步 Register → Shared memory store (cp.async on SM80+, TMA on SM90+).
    /// Fallback on SM70/Metal/HIP: emit synchronous SharedMemStore semantics.
    SharedMemAsyncStore {
        name: String,
        dst_offset: OffsetExpr,
        src: VRegId,
        width: SimdWidth,
        dtype: crate::compiler::trace::QuantPrecision,
    },
    /// Wait for N most recent async shared memory operations to complete.
    /// PTX: `cp.async.wait_group N` (SM80+); `bar.sync 0` (SM70).
    SharedMemAsyncWaitGroup {
        n: u32,
    },
    /// §0.2.8 WeightPrefetchAsync: 异步预取下一层权重到共享内存。
    ///
    /// 在层循环中，计算层 N 时同时发出层 N+1 权重的异步加载。
    /// GPU: SM90+ 使用 cp.async.bulk (TMA), SM80 使用 cp.async。
    /// CPU/ARM: 无异步预取能力，同步加载（与权重内联加载等效）。
    ///
    /// 该指令与 AsyncCopy 不同：AsyncCopy 是通用的异步内存拷贝，
    /// WeightPrefetchAsync 是专门为层循环权重预取设计的，使用
    /// 层循环的 byte_offset 计算源地址，目标为命名的共享内存 buffer。
    WeightPrefetchAsync {
        /// 共享内存 buffer 名称 (接收预取数据)
        smem_name: String,
        /// 权重基址 GPR (通常为层循环的 weight_ptr)
        weight_base: VRegId,
        /// 权重字节偏移 (编译时常量，由 PackMap 或图偏移决定)
        weight_offset: usize,
        /// 预取字节数
        size: usize,
    },
    /// §0.2.8 WeightPrefetchWait: 等待最近 N 个 WeightPrefetchAsync 完成。
    ///
    /// GPU: SM80+ 使用 cp.async.wait_group N; SM90+ 使用 mbarrier try_wait。
    /// CPU/ARM: 无操作 (同步加载已在 WeightPrefetchAsync 中完成)。
    WeightPrefetchWait {
        /// 等待组深度 (0 = 等待全部完成)
        group: u32,
    },
    /// Warp 角色声明: Producer (TMA 加载) 或 Consumer (WGMMA 计算)。
    ///
    /// SM90+ Cluster 中，warp 0-1 专职 TMA 加载 (Producer)，warp 2-3 专职 WGMMA (Consumer)。
    /// Producer 通过 mbarrier.arrive 通知 Consumer 数据就绪。
    ///
    /// PTX (SM90+):
    ///   Producer: setmaxnreg.inc.sync.allocating_group.u32 32 (增加 Producer 寄存器配额)
    ///   Consumer: setmaxnreg.dec.sync.allocating_group.u32 32 (减少 Consumer 寄存器配额)
    /// SM80-/CPU/ARM: NOP
    WarpRoleDeclare {
        /// 角色: 0 = Producer, 1 = Consumer
        role: u32,
    },

    /// mbarrier arrive: Producer 通知 barrier 数据已写入 shared memory。
    ///
    /// PTX (SM90+): mbarrier.arrive.expect_tx.release.cta.shared::cluster.shared::cta.b64 [%barrier], NumBytes
    /// SM80: cp.async.commit_group (implicit barrier)
    /// CPU/ARM: NOP
    WarpBarrierArrive {
        /// barrier 名称 (编译时常量，映射到 PTX barrier 编号)
        barrier_name: String,
        /// 预期写入字节数 (TMA 事务大小)
        tx_bytes: usize,
    },

    /// mbarrier wait: Consumer 等待 Producer 数据就绪。
    ///
    /// PTX (SM90+): mbarrier.try_wait.parity.acquire.cta.shared::cluster.shared::cta.b64 [%barrier], parity
    /// SM80: cp.async.wait_group 0 (等待所有异步操作)
    /// CPU/ARM: NOP
    WarpBarrierWait {
        /// barrier 名称
        barrier_name: String,
        /// Parity (0 or 1, for ping-pong buffer tracking)
        parity: u32,
    },

    /// TMA 2D tensor descriptor 初始化 (SM90+ Host 端 cuTensorMapEncodeTiled)。
    ///
    /// 编译时声明描述符参数，Host 端 glue 代码负责调用 cuTensorMapEncodeTiled。
    /// 描述符通过 ABI 参数传递给 GPU kernel。
    /// CPU/ARM: 返回 Error (TMA 仅 GPU SM90+)
    TmaDescriptorInit {
        /// 描述符名称 (编译时常量标识符)
        desc_name: String,
        /// 全局内存维度 [dim0, dim1] (行数, 列数)
        global_dim: [usize; 2],
        /// 全局内存步长 [stride0, stride1] (字节)
        global_stride: [usize; 2],
        /// Tile 大小 [box_dim0, box_dim1] (元素数)
        box_dim: [usize; 2],
        /// Swizzle 模式 (消除 shared memory bank conflict)
        swizzle: TmaSwizzle,
        /// 数据类型
        dtype: QuantPrecision,
    },

    /// TMA 2D tensor 异步拷贝 (SM90+ cp.async.bulk.tensor.2d.shared.global)。
    ///
    /// 单线程发起整个 tile 搬运，硬件自动完成，零其他线程参与。
    /// SM80 fallback: 使用 1D cp.async.ca 逐行加载。
    /// CPU/ARM: 返回 Error (TMA 仅 GPU SM90+)
    Tma2DCopy {
        /// TMA 描述符名称 (对应 TmaDescriptorInit 的 desc_name)
        desc_name: String,
        /// 目标 shared memory buffer 名称
        smem_name: String,
        /// Tile 起始行坐标 (运行时 GPR)
        coord_x: VRegId,
        /// Tile 起始列坐标 (运行时 GPR)
        coord_y: VRegId,
        /// mbarrier 名称 (用于完成信号)
        barrier_name: String,
    },

    /// mbarrier 初始化 (SM90+)。
    ///
    /// PTX: mbarrier.init.shared::cta.b64 [%name], %thread_count
    /// SM80: 无需显式初始化 (使用 cp.async.wait_group)
    /// CPU/ARM: NOP
    BarrierInit {
        /// barrier 名称
        name: String,
        /// 参与线程数
        thread_count: u32,
    },

    /// Block 同步 (bar.sync)
    BlockSync,
    /// Warp-level reduction: shfl.sync.down.* tree reduction pattern
    WarpReduce {
        op: ReduceOp,
        src: VRegId,
        dst: VRegId,
        width: SimdWidth,
    },

    // ── 超越函数 ──

    /// 内联超越函数 (后端选择最优实现)
    Transcendental {
        dst: VRegId,
        src: VRegId,
        func: TranscendentalFn,
    },

    // ── 运行时机制 ──

    /// §9.2 热修补: 5-byte NOP 占位符
    HotpatchSlot {
        slot_id: u32,
        initial_target: HotpatchTarget,
        alternatives: Vec<HotpatchTarget>,
    },
    /// §15 MoE 核内分发: 间接跳转
    IndirectJump {
        index: VRegId,
        targets: Vec<JumpTarget>,
    },
    /// §16 Early-Exit: 条件退出
    ConditionalExit {
        condition: VRegId,
        output: VRegId,
    },

    /// §20 BCI: 如果 ptr 寄存器非 NULL，跳转到 target_label。
    /// 用于 ABI prologue → outer loop setup Legacy / Batch 分支。
    /// x86 lower: test ptr, ptr; jnz target_label
    BranchIfPtrNonNull {
        ptr: VRegId,
        target_label: usize,
    },

    /// GPR 条件分支：若 scalar 值 == 0 则跳转到 target_label。
    /// 用于采样管线中 temperature==0 时跳过 stochastic sampling 走 argmax。
    BranchIfGprZero {
        /// 待检测的 GPR 寄存器 (Scalar)
        value: VRegId,
        /// 跳转目标 label ID
        target_label: usize,
    },

    /// GPR 无符号比较分支：若 a < b (unsigned) 则跳转到 target_label。
    /// 用于 mega-kernel prefill/generate 统一循环中，跳过 prefill 阶段的采样。
    BranchIfGprLtU {
        /// 左操作数 (Scalar)
        a: VRegId,
        /// 右操作数 (Scalar)
        b: VRegId,
        /// 跳转目标 label ID
        target_label: usize,
    },

    /// 无条件跳转到 target_label。
    /// 用于采样管线中 argmax 路径跳过 stochastic sampling 部分。
    UnconditionalBranch {
        target_label: usize,
    },

    /// §20 BCI-004: Batch per-token seq_id 查找。
    ///
    /// 给定 token_index (0..total_prefill_tokens)，通过 cumsum(prompt_lens) 二分搜索
    /// 确定该 token 属于哪个 sequence，同时输出该 sequence 的 page_table_offset。
    ///
    /// Layout:
    ///   batch_ctx + 0  = num_seqs (u64)
    ///   batch_ctx + 88 + seq_idx * 64 + 8  = prompt_len (u32) (BCI6)
    ///   batch_ctx + 88 + seq_idx * 64 + 16 = page_table_offset (u32) (BCI6)
    ///
    /// x86 lower: 线性扫描 cumsum 累加，对比 token_index，找到 seq_id 后读取 pt_offset。
    BatchSeqIdLookup {
        /// 输出: seq_id (u32, VRegKind::Scalar)
        dst: VRegId,
        /// 输出: page_table_offset for this sequence (u32, VRegKind::Scalar)
        pt_offset_out: VRegId,
        /// 输入: token_index (0-based in total_prefill_tokens, VRegKind::Scalar)
        token_index: VRegId,
        /// 输入: batch_ctx_ptr (VRegKind::Ptr)
        batch_ctx_ptr: VRegId,
    },

    /// §20 BCI-006: Per-sequence argmax on a specific logit row.
    ///
    /// logits_flat_ptr 指向 [num_seqs][vocab_size] 的连续 logits buffer。
    /// 读取 seq_id 对应的 logit 行，执行 argmax，输出 sampled token_id。
    BatchPerSeqArgmax {
        /// 输出: sampled token_id (u32, VRegKind::Scalar)
        dst: VRegId,
        /// 输入: seq_id (VRegKind::Scalar)
        seq_id: VRegId,
        /// 输入: logits_flat_ptr (VRegKind::Ptr), layout [num_seqs][vocab_size]
        logits_flat_ptr: VRegId,
        /// vocab_size (编译时常量，用于行偏移计算)
        vocab_size: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// §20 BCI-006: Per-sequence 停止条件检查 + active_flag 更新。
    ///
    /// 检查: token_id == eos_token_id 或 gen_count >= max_new_tokens。
    /// 如果满足，写入 active_flag = 0 到 batch_ctx 中对应 seq 的 metadata。
    ///
    /// Layout:
    ///   batch_ctx + 88 + seq_id * 64 + 8  = gen_count (u32) (BCI6)
    ///   batch_ctx + 88 + seq_id * 64 + 12 = max_new_tokens (u32) (BCI6)
    ///   batch_ctx + 88 + seq_id * 64 + 16 = eos_token_id (u32) (BCI6)
    ///   batch_ctx + 88 + seq_id * 64 + 24 = active_flag (u32) (BCI6)
    BatchPerSeqStopCheck {
        /// 输入: seq_id (VRegKind::Scalar)
        seq_id: VRegId,
        /// 输入: sampled token_id (VRegKind::Scalar)
        token_id: VRegId,
        /// 输入: batch_ctx_ptr (VRegKind::Ptr)
        batch_ctx_ptr: VRegId,
    },

    // ── 原子操作 ──

    /// 原子加: [base + offset] += value。
    /// elem_width=4 → u32 (`lock add dword`), elem_width=8 → u64 (`lock add qword`).
    /// elem_width=8 时自动附带 Acquire-Release 语义（ARCH-SG-QTAP step_index bump）。
    /// Replaces AtomicAddU32 + AtomicAddU64 (dtype in param, not in name).
    AtomicAdd {
        base: VRegId,
        offset: OffsetExpr,
        value: u64,
        /// 元素宽度: 4 (u32) 或 8 (u64).
        elem_width: usize,
    },

    /// 内存屏障 (ARCH-SG-QTAP)。
    ///
    /// 保证屏障前的所有内存写入对其他线程/CPU 可见，屏障后的读/写不会重排到前面。
    /// 语义接近 `std::sync::atomic::fence(Release)` / `Acquire` / `AcqRel` / `SeqCst`,
    /// 主要供 Q-Tap 等 Callback 边界写入使用。
    ///
    /// x86_64 lower: 对 SeqCst 发射 `mfence`; 对 Release/Acquire 在 x86 TSO 下可退化
    ///               为编译屏障 (此处仍发射 `mfence` 以避免前后 relaxed 指令重排)。
    /// AArch64 lower: `DMB ISH` (或 `DMB ISHST` for Release-only)。
    MemFence {
        order: MemFenceOrder,
    },

    // ── 采样指令 (Mega-Kernel generate 循环内使用) ──

    /// Argmax: 在 logits 向量中找最大值的索引。
    ///
    /// 从 logits_ptr[0..vocab_bytes] 扫描，将最大值的索引写入 dst (GPR)。
    /// 用于 greedy 采样 (temperature=0)。
    ///
    /// x86 lower: 向量化 VMAXPS + VPCMPEQD 水平归约 + 标量索引追踪。
    Argmax {
        /// 输出: 最大值索引 (GPR, VRegKind::Scalar)
        dst: VRegId,
        /// logits 向量基址 (Ptr)
        logits_ptr: VRegId,
        /// logits 向量总字节数 (vocab_size * elem_bytes)
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Temperature scaling: logits /= temperature。
    ///
    /// 当 temperature > 0 时，将 logits 向量每个元素除以 temperature。
    /// temperature=0 时不应使用此指令（直接走 Argmax）。
    ///
    /// x86 lower: 广播 temperature → VDIVPS。
    TemperatureScale {
        /// logits 向量基址 (in-place 操作)
        logits_ptr: VRegId,
        /// temperature 值的内存地址 (Ptr → f32)
        temp_ptr: VRegId,
        /// logits 向量总字节数
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// 存储 token 到输出缓冲 + 回写到 input_ids。
    ///
    /// 将 token_id (GPR 中的 u32) 写入 output_buf[token_counter]。
    /// 同时写入 input_ids_ptr[prompt_len_bytes/4 + token_counter + 1]，为下一次迭代提供输入。
    StoreToken {
        /// token ID (GPR, VRegKind::Scalar)
        token_id: VRegId,
        /// 输出缓冲基址 (Ptr)
        output_buf: VRegId,
        /// token 计数器 (Counter, 用于计算写入偏移)
        counter: VRegId,
        /// input_ids 基址 (Ptr) — 用于回写生成的新 token
        input_ids_ptr: VRegId,
        /// prompt_len * 4 字节偏移 (Scalar) — 回写位置 = input_ids_ptr + prompt_len_bytes + (counter + 1) * 4
        prompt_len_bytes: VRegId,
    },

    /// 停止条件检查: token_id == eos_token_id 或 counter >= max_new_tokens。
    ///
    /// 当条件满足时跳转到 done_label (LoopEnd 之后)。
    /// 用于 generate 循环的提前退出。
    ///
    /// x86 lower: CMP + JE/CMP + JGE → 条件跳转。
    CheckStopCondition {
        /// 当前 token ID (GPR, VRegKind::Scalar)
        token_id: VRegId,
        /// token 计数器 (Counter)
        counter: VRegId,
        /// EOS token ID 的内存地址 (Ptr → u32)
        eos_ptr: VRegId,
        /// max_new_tokens 的内存地址 (Ptr → usize)
        max_tokens_ptr: VRegId,
    },

    /// Softmax 阶段 1: 在 logits 向量中找最大值。
    ///
    /// 扫描 logits_ptr[0..vocab_bytes]，将最大值写入 dst (Scalar f32)。
    /// x86 lower: VMAXPS 水平归约（现有 Argmax 模式扩展）。
    /// GPU lower: warp shuffle reduction (`shfl.sync.down.b32` + `fmax.rn.f32`)。
    SoftmaxReduceMax {
        /// 输出: 最大值 (Scalar f32)
        dst: VRegId,
        /// logits 向量基址 (Ptr)
        logits_ptr: VRegId,
        /// logits 向量总字节数
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Softmax 阶段 2: exp(logits - max) 并求和。
    ///
    /// 对 logits_ptr[0..vocab_bytes] 每个元素: exp(x - max_val)，
    /// 同时累加所有 exp 值到 sum_dst (Scalar f32)。
    /// in-place 修改 logits_ptr 指向的内存（exp 结果覆盖原始 logits）。
    ///
    /// x86 lower: VSUBPS(max) + VEXPPS(近似) + VADDPS(horizontal sum)。
    /// GPU lower: 向量化 sub + ex2.approx.ftz.f32 + warp shuffle sum。
    SoftmaxExpSum {
        /// 输出: sum(exp(x-max)) (Scalar f32)
        sum_dst: VRegId,
        /// logits 向量基址 (Ptr, in-place 修改为 exp(x-max))
        logits_ptr: VRegId,
        /// 最大值 (Scalar f32, 来自 SoftmaxReduceMax)
        max_val: VRegId,
        /// logits 向量总字节数
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Softmax 阶段 3: 归一化 exp(x-max) / sum → 概率分布。
    ///
    /// 对 logits_ptr[0..vocab_bytes] 每个元素除以 sum_val。
    /// in-place 修改，结果为概率分布。
    ///
    /// x86 lower: 广播 sum → VDIVPS。
    /// GPU lower: broadcast(sum) + 向量化 div。
    SoftmaxNormalize {
        /// logits/exp 向量基址 (Ptr, in-place 修改为概率)
        logits_ptr: VRegId,
        /// sum 值 (Scalar f32, 来自 SoftmaxExpSum)
        sum_val: VRegId,
        /// 向量总字节数
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Top-K 过滤: 保留概率最高的 K 个候选，其余置零。
    ///
    /// 对 probs_ptr[0..vocab_size] 执行 bitonic sort，保留前 K 个，
    /// 其余位置清零。输出：保留的候选索引写入 indices_ptr，候选概率保留在 probs_ptr。
    /// k 值通常 ≤50，bitonic sort 只需 6 轮比较。
    ///
    /// x86 lower: 标量循环部分排序（vocab 通常 >32K，全部排序不现实）。
    ///   实际用 "top-K selection" 算法: 维护 K 大小的小顶堆，O(N log K)。
    /// GPU lower: warp 级 bitonic sort + warp 合作 top-K merge。
    SampleTopKFilter {
        /// 概率向量基址 (Ptr, in-place: 非 top-K 置零)
        probs_ptr: VRegId,
        /// 输出: 保留的候选索引 (Ptr → u32[], vocab_size 大小)
        indices_ptr: VRegId,
        /// K 值的内存地址 (Ptr → u32)
        k_ptr: VRegId,
        /// 向量总字节数
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Top-P (Nucleus) 过滤: 累积概率 ≥ P 后截断。
    ///
    /// 对 probs_ptr[0..vocab_size]（已 top-K 过滤，非零概率按降序排列），
    /// 计算累积概率，当累积 ≥ p_val 时将后续位置清零。
    ///
    /// x86 lower: 标量累加 + 比较清零。
    /// GPU lower: warp scan (inclusive prefix sum) + 比较 + 条件清零。
    SampleTopPFilter {
        /// 概率向量基址 (Ptr, in-place: 超出 top-p 的置零)
        probs_ptr: VRegId,
        /// P 值的内存地址 (Ptr → f32, 如 0.9)
        p_ptr: VRegId,
        /// 向量总字节数
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Multinomial 采样: 从概率分布中按权重随机采样一个 token。
    ///
    /// 生成一个 [0, 1) 均匀随机数，在累积概率分布中二分搜索，
    /// 返回采样到的 token ID。
    ///
    /// x86 lower: PRNG 生成 + 标量线性扫描累积分布（概率通常已过滤到 <50 个候选）。
    /// GPU lower: Philox PRNG + warp 合作二分搜索。
    SampleMultinomial {
        /// 输出: sampled token_id (Scalar u32)
        dst: VRegId,
        /// 概率向量基址 (Ptr, 已经过 top-k/top-p 过滤)
        probs_ptr: VRegId,
        /// PRNG 状态基址 (Ptr → u64[2], seed + counter)
        rng_state_ptr: VRegId,
        /// 向量总字节数
        vocab_bytes: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Warp PRNG: 使用 Philox 4×32 counter-based PRNG 生成一个 [0, 1) f32 随机数。
    ///
    /// Philox 算法: 纯整数运算（vadd + vmul + XOR），无共享状态，无需全局同步。
    /// 每个 warp lane 基于全局 counter + lane_id 生成独立随机数。
    /// 写入 rng_state 供 SampleMultinomial 消费。
    ///
    /// x86 lower: 3 轮 AES-round 替代 (模拟 S-box: multiply + XOR + rotate)。
    /// GPU lower: 同上，全部 PTX 整数指令 (add.u32 + mul.hi.u32 + xor.b32 + shl.b32)。
    WarpPRNG {
        /// 输出: [0, 1) 均匀随机数 (Scalar f32)
        dst: VRegId,
        /// PRNG 状态基址 (Ptr → u64[2]: {global_counter, thread_id})
        rng_state_ptr: VRegId,
    },

    /// 将编译时常量写入栈上位置 [rbp+offset]。
    /// elem_width=4 → u32 (`mov dword`), elem_width=8 → u64 (`mov qword`).
    /// Replaces StoreU32ToStack (dtype in param, not in name).
    StoreConstToStack {
        /// 相对 rbp 的偏移（正数，与 StackArg 相同约定）
        rbp_offset: i32,
        /// 要写入的值
        value: u64,
        /// 元素宽度: 4 (u32) 或 8 (u64).
        elem_width: usize,
    },
    /// Memory copy: copies `bytes` bytes from src to dst pointer.
    /// Used for KV cache writes: copy current K/V row to persistent KV cache.
    MemCopy {
        /// Destination pointer (VRegKind::Ptr).
        dst: VRegId,
        /// Source pointer (VRegKind::Ptr).
        src: VRegId,
        /// Number of bytes to copy (compile-time constant).
        bytes: usize,
    },

    /// 跳出 generate loop 到函数 epilogue。
    ///
    /// 用于所有输出模式的退出路径:
    /// - Generate: 返回 gen_counter VReg (实际生成 token 数)
    /// - Classify/Encode: 返回常量 0
    ///
    /// x86 lower: 设置 rax = return_value + JMP epilogue_label。
    BreakLoop {
        /// rax 返回值来源。
        return_value: ReturnValue,
    },

    /// 指针算术: dst = base + offset_bytes。
    ///
    /// 用于层循环中递增权重指针 (weight_ptr += layer_stride)。
    /// dst 和 base 都是 VRegKind::Ptr。
    AddPtr {
        /// 结果指针 (Ptr)
        dst: VRegId,
        /// 基址指针 (Ptr)
        base: VRegId,
        /// 偏移字节数 (编译时常量)
        offset: usize,
    },

    /// GPR 二元操作: dst = a op b (REQ-VR-003)。
    /// 替代旧 GprAdd/GprSub/GprMulConst/GprSubConst/GprShl/GprShr。
    GprBinOp {
        dst: VRegId,
        a: VRegId,
        b: GprOperand,
        op: GprOp,
    },

    /// GPR 一元操作: dst = op(src) (REQ-VR-010)。
    GprUnaryOp {
        dst: VRegId,
        src: VRegId,
        op: GprUnaryOpKind,
    },

    /// GPR 加载立即数: dst = value. VRegKind::Scalar.
    GprLoadImm {
        dst: VRegId,
        value: usize,
    },

    // ── 内存提示 ──

    /// 预取: hint 硬件将 [base + offset] 附近数据预加载到缓存
    Prefetch {
        base: VRegId,
        offset: OffsetExpr,
        /// 预取距离（从 IsaHook::prefetch_hint 获取）
        distance: usize,
        /// 缓存层级提示
        hint: super::isa_hook::PrefetchHint,
    },

    // ── 元操作 ──

    /// 声明虚拟寄存器 (帮助分配器)
    DeclareVReg {
        id: VRegId,
        kind: VRegKind,
        width: SimdWidth,
    },
    /// 标记 VReg 生命周期结束 (可回收物理寄存器)
    ReleaseVReg {
        id: VRegId,
    },
    /// 标记代码位置 (供 JMP table 的目标 label)。
    ///
    /// compile_mega_kernel_vm output redirect 在每条输出模式路径起始处插入 MarkLabel。
    /// x86 lower: create_label() + set_label() — 下一条指令即路径入口。
    MarkLabel {
        label_id: usize,
    },

    // ── 外部回调 (Callback Table) ──

    /// 从 callback table 加载一个 entry: fn_ptr = table[slot].fn_ptr, ctx = table[slot].ctx。
    ///
    /// Callback table 通过 ABI arg 20 传入（`*const MegaKernelCallbackTable`）。
    /// 每个 entry 为 16 字节：[fn_ptr: u64, ctx: u64]。
    /// `table_ptr` 是从栈加载的 table 基址，`slot_id` 是编译时常量。
    ///
    /// x86 lower: mov fn_ptr_out, [table_ptr + slot_id*16] ; mov ctx_out, [table_ptr + slot_id*16 + 8]
    /// AArch64 lower: ldr fn_ptr_out, [table_ptr, #slot_id*16] ; ldr ctx_out, [table_ptr, #slot_id*16+8]
    LoadCallbackEntry {
        /// table 基址 GPR (VRegKind::Ptr)
        table_ptr: VRegId,
        /// callback slot 索引 (编译时常量)
        slot_id: usize,
        /// 输出: fn_ptr GPR (VRegKind::Ptr)
        fn_ptr_out: VRegId,
        /// 输出: ctx GPR (VRegKind::Ptr)
        ctx_out: VRegId,
    },

    /// 调用外部函数指针: ret_val = fn_ptr(ctx_ptr)。
    ///
    /// 用于 mega-kernel 在 SgDetect 后回调外部函数（如 KnowledgeProvider）。
    /// fn_ptr 和 ctx 从 `LoadCallbackEntry` 获得。
    ///
    /// 调用约定: 仅使用 rdi (ctx) 作为输入，返回值在 eax。
    /// caller-saved 寄存器在调用前保存、调用后恢复。
    ///
    /// x86 lower: push caller-saved → mov rdi, ctx → call rax → pop → mov ret_val, eax
    /// AArch64 lower: save caller-saved → mov x0, ctx → blr fn_ptr → restore → str w0, ret_val
    NativeCall {
        /// 返回值 GPR (VRegKind::Scalar)
        ret_val: VRegId,
        /// 函数指针 GPR (VRegKind::Ptr)
        fn_ptr: VRegId,
        /// 上下文指针 GPR (VRegKind::Ptr)
        ctx_ptr: VRegId,
    },

    /// §3.7 ActivationSwap: 交换 ping-pong buffer 指针 (零数据拷贝)
    ///
    /// 层循环末尾调用：交换 ptr_a 和 ptr_b 寄存器中的指针值。
    /// 下一层迭代读 ptr_a 时实际读到的是上一层写入的 ptr_b buffer。
    ///
    /// x86 lower: xchg reg_a, reg_b (经过 RegAlloc 映射到物理寄存器)
    /// AArch64 lower: mov + mov pair swap
    ActivationSwap {
        /// ping buffer 指针寄存器
        ptr_a: VRegId,
        /// pong buffer 指针寄存器
        ptr_b: VRegId,
    },

    /// PagedAttention: 从 page table 计算物理地址。
    ///
    /// 计算: page_id = page_table[token_idx / page_size]
    ///        addr = pool_base + page_id * page_stride + (token_idx % page_size) * row_bytes + base_offset
    ///
    /// x86: mov eax, [page_table + token_idx*4]; imul rax, page_stride; add rax, pool_base; add rax, offset
    PageTableAddr {
        /// 目标 GPR (存放计算出的物理地址)
        dst: VRegId,
        /// KV pool 基地址 GPR
        pool_base: VRegId,
        /// page table 指针 GPR (u32[])
        page_table_ptr: VRegId,
        /// token 索引 (字节偏移, 通常来自 loop counter)
        ki_byte_off: OffsetExpr,
        /// 每个 token 的字节数 (head_dim * elem_bytes)
        row_bytes: usize,
        /// page 大小 (tokens per page)
        page_size: usize,
        /// page 步长 (bytes per page)
        page_stride: usize,
        /// 层/头偏移基础偏移量
        base_offset: usize,
        /// §20 BCI-005: per-sequence page_table 偏移寄存器 (batch 模式)
        /// None = 单序列 (当前行为)
        /// Some(vreg) = 从寄存器读取 seq 的 pt_offset, 加到 page table 索引上
        seq_pt_offset: Option<VRegId>,
    },

    /// PagedAttention: 将 KV 行写入 page pool。
    ///
    /// 从 src 向量寄存器写入 page pool 中对应 page 的位置。
    /// 写入位置 = pool_base + page_table[seq_index / page_size] * page_stride + (seq_index % page_size) * row_bytes + base_offset
    ///
    /// §20 BCI-005: `seq_index` 为运行时 GPR，batch 模式下不同序列使用不同索引。
    PageTableKVWrite {
        /// 源向量寄存器 (要写入的 KV 数据)
        src: VRegId,
        /// KV pool 基地址 GPR
        pool_base: VRegId,
        /// page table 指针 GPR (u32[])
        page_table_ptr: VRegId,
        /// 序列位置索引 (§20 BCI-005: 运行时 GPR, batch 模式 per-sequence)
        seq_index: VRegId,
        /// 每个 token 的字节数
        row_bytes: usize,
        /// page 大小
        page_size: usize,
        /// page 步长
        page_stride: usize,
        /// 层/头偏移基础偏移量
        base_offset: usize,
        /// SIMD 宽度
        width: SimdWidth,
        /// 数据类型
        dtype: DType,
    },

    /// PagedAttention: 将 KV 行量化后写入 page pool (KIVI 4-bit/2-bit)。
    ///
    /// 在 PageTableKVWrite 基础上增加 in-register 量化:
    /// 1. 从 src 向量寄存器读取 FP32 KV 数据
    /// 2. 计算 per-channel (K) 或 per-token (V) scale
    /// 3. 量化为 4-bit packed data
    /// 4. 写 packed data 到 page data 区域
    /// 5. 写 scale 到 page quant_meta 区域
    ///
    /// 写入位置计算同 PageTableKVWrite:
    /// packed_addr = pool_base + page_table[seq_index / page_size] * page_stride + (seq_index % page_size) * quant_row_bytes + base_offset
    /// scale_addr = packed_addr + scale_offset
    PageTableKVWriteQuant {
        /// 源向量寄存器 (要写入的 FP32 KV 数据)
        src: VRegId,
        /// KV pool 基地址 GPR
        pool_base: VRegId,
        /// page table 指针 GPR (u32[])
        page_table_ptr: VRegId,
        /// 序列位置索引 (运行时 GPR)
        seq_index: VRegId,
        /// 量化后每行字节数 (packed data)
        quant_row_bytes: usize,
        /// 原 FP32 每行字节数
        fp32_row_bytes: usize,
        /// page 大小
        page_size: usize,
        /// page 步长
        page_stride: usize,
        /// 层/头偏移基础偏移量
        base_offset: usize,
        /// scale 写入偏移 (相对于 packed data 起始)
        scale_offset: usize,
        /// SIMD 宽度
        width: SimdWidth,
        /// 量化模式: Kivi4 或 Kivi2
        kivi_mode: KvLoadMode,
        /// 元素数 (head_dim)
        num_elems: usize,
    },

    /// KIVI per-channel 4-bit 量化: 从 f32 向量计算 per-channel max 并量化为 4-bit packed。
    ///
    /// 计算: scale[i] = max(|src[i]|)
    ///        packed[j] = quantize_4bit(src[j], scale[j/2])
    /// 输出: packed data (VecStore) + scales (VecStore)
    KiviQuantChannel {
        /// 源向量寄存器 (f32 数据, head_dim 个元素)
        src: VRegId,
        /// 输出 packed data 地址 GPR
        dst_ptr: VRegId,
        /// 输出 scale 地址 GPR
        scale_ptr: VRegId,
        /// 通道数 (head_dim)
        num_channels: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// KIVI per-token 4-bit 量化: 从 f32 向量计算 per-token max 并量化为 4-bit packed。
    ///
    /// 用于 V cache 量化 (每个 token 一组 scale)。
    KiviQuantToken {
        /// 源向量寄存器 (f32 数据)
        src: VRegId,
        /// 输出 packed data 地址 GPR
        dst_ptr: VRegId,
        /// 输出 scale 地址 GPR
        scale_ptr: VRegId,
        /// token 内元素数 (head_dim)
        num_elems: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// KIVI 4-bit 反量化 load: 从 packed 4-bit + scale 恢复为 f32 向量。
    ///
    /// Attention decode KV load 路径使用。
    KiviDequantLoad {
        /// 目标向量寄存器 (f32)
        dst: VRegId,
        /// packed 4-bit data 地址 GPR
        src_ptr: VRegId,
        /// scale 地址 GPR
        scale_ptr: VRegId,
        /// 通道/token 元素数
        num_elems: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// Q6_K per-sub-block i8 scale load: dst(YMM) = (f32)(i8) * (base + index_reg).
    /// Loads a single i8 from `scales_base + sub_block_idx`, sign-extends to i32,
    /// converts to f32, and broadcasts to all YMM lanes.
    /// x86 lower: movsx rax,byte[scales_base + sub_idx_reg] → cvtsi2ss → vbroadcastss
    GgufSubScaleLoad {
        dst: VRegId,
        scales_base: VRegId,
        sub_block_idx: VRegId,
        width: SimdWidth,
    },

    /// K-Quant (Q3_K/Q4_K/Q5_K) packed 6-bit scale/min decode for one sub-block.
    /// Decodes the packed scale or min from `scales[12]` array indexed by sub_block_idx.
    ///
    /// When is_min=false (scale):
    ///   if j<4: sc = scales[j] & 0x3F
    ///   if j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    /// When is_min=true (min/zero):
    ///   if j<4: m = scales[j+4] & 0x3F
    ///   if j>=4: m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    /// Result: dst = (f32)sc_or_m (broadcast to all lanes)
    GgufKQuantScaleLoad {
        dst: VRegId,
        scales_base: VRegId,
        sub_block_idx: VRegId,
        /// Number of packed scales (typically 8 for Q3_K, 8 for Q4_K/Q5_K)
        scales_count: usize,
        /// Whether to use Q3KExtended layout (same bit extraction but different offset calc)
        is_q3k_extended: bool,
        /// Whether to extract min/zero instead of scale
        is_min: bool,
        width: SimdWidth,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SPEC 23-QUANT-CODEGEN-ALGO §3: Quant* 解码 VmInstr
    // VR-004 去重后保留 7 个变体 (18→7):
    //   保留: QuantBroadcastInt, QuantLoadBytesVec, QuantCodebookLookup,
    //         QuantExtractBits, QuantDequantFma, QuantInterleave, QuantScalarCvtLoad
    //   删除 (用通用指令替代): QuantVecBitAnd, QuantVecBitOr, QuantBroadcast,
    //         QuantFma, QuantScalarLoad, QuantIntDivConst, QuantIntMul,
    //         QuantVecShiftLeft, QuantVecShiftRight
    //   合并: QuantLoadF16toF32 + QuantLoadI8toF32 → QuantScalarCvtLoad
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// 整数常量广播到 YMM: dst[0..lanes] = value (i32). 用于 AND 掩码等整数操作。
    QuantBroadcastInt { dst: VRegId, value: u64, width: SimdWidth },

    /// 标量加载 + 类型转换: dst = (f32) load(src_dtype, base + offset)。
    /// Replaces QuantLoadF16toF32 + QuantLoadI8toF32 (REQ-VR-004 §1.4.2)。
    QuantScalarCvtLoad {
        dst: VRegId,
        base: VRegId,
        offset: i64,
        src_dtype: ScalarCvtSource,
        width: SimdWidth,
    },

    /// GGUF 量化块加载 — 单平面 (REQ-VR-001 §1.1.1)。
    QuantBlockLoad {
        dst: VRegId,
        base: VRegId,
        offset: OffsetExpr,
        unpack: BlockUnpackMode,
        width: SimdWidth,
    },

    /// GGUF 量化块加载 — 多平面位合并 (REQ-VR-001 §1.1.2)。
    QuantBiPlaneLoad {
        dst: VRegId,
        qs_base: VRegId,
        extra_base: VRegId,
        bias: f32,
        mode: BiPlaneMode,
        width: SimdWidth,
    },

    /// 多字节向量加载 (零扩展为 i32，不转 float):
    /// 从 base + offset 加载 count 字节，零扩展为 count 个 i32 存入 YMM。
    /// lanes 0..count 有效，count..lanes 为 0。后续需 AND/Shift 提取 nibble。
    QuantLoadBytesVec { dst: VRegId, base: VRegId, offset: i64, count: usize, signed: bool, width: SimdWidth },

    /// Codebook 查表: f32 向量 = [codebook[indices[i]] for i in 0..lanes]。
    QuantCodebookLookup {
        dst: VRegId,
        indices: VRegId,
        codebook_data: &'static [i8],
        vector_size: usize,
        bits_per_entry: u8,
        width: SimdWidth,
    },

    /// 位域提取: dst = (src >> bit_offset) & ((1 << bit_width) - 1)。
    QuantExtractBits { dst: VRegId, src: VRegId, bit_offset: u32, bit_width: u8, width: SimdWidth },

    /// Register-inline dequant + FMA fusion: packed weight → unpack → dequant → FMA.
    QuantDequantFma {
        /// Accumulator register (in/out): dst += result
        dst: VRegId,
        /// Packed weight register (already loaded)
        weight: VRegId,
        /// Activation register (already loaded)
        activation: VRegId,
        /// Scale register (per-channel or per-block)
        scale: VRegId,
        /// Zero-point register (optional, VRegId(0) for none)
        zero_point: VRegId,
        /// Quantization format — drives unpack/dequant strategy
        quant_kind: crate::quant_format::QuantDataKind,
        /// Accumulator dtype
        dtype: crate::compiler::trace::QuantPrecision,
        /// SIMD width
        width: SimdWidth,
    },

    /// 整数向量交叉合并 (lo nibbles + hi nibbles → lanes 个元素)。
    QuantInterleave { dst: VRegId, lo: VRegId, hi: VRegId, width: SimdWidth },

    /// 整数向量顺序拼接: dst = [lo[0..N/2], hi[0..N/2]]。
    /// 用于 QuantGather 路径，元素必须保持原始顺序。
    QuantConcatSeq { dst: VRegId, lo: VRegId, hi: VRegId, width: SimdWidth },

    /// Q3_K combined decode: extracts 2-bit values from qs[] at variable shift,
    /// applies conditional hmask bias, multiplies by per-sub-block scale.
    /// dst = f32 vector of decoded values.
    /// block_base: pointer to Q3_K block start (hmask + qs + scales + d)
    /// lane_offset: GPR holding iteration counter (0..31 for lanes=8)
    /// d_vreg: Vec VReg holding f32 super-block scale d
    /// qs_offset: byte offset from block_base to qs[] array (32 for Q3_K)
    /// hmask_offset: byte offset from block_base to hmask[] array (0 for Q3_K)
    /// lanes: number of output elements per iteration (8 for AVX2)
    Q3KDecodeStep {
        dst: VRegId,
        block_base: VRegId,
        lane_offset: VRegId,
        d_vreg: VRegId,
        qs_offset: usize,
        hmask_offset: usize,
        lanes: usize,
        width: SimdWidth,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SPEC 23-QUANT-CODEGEN-ALGO §4.3: 原生 Dot-Product VmInstr
    // 硬件无关语义，ISA lowering 决定具体指令
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Dot-product: acc += a · b (per-element), dtype determined by input_dtype (REQ-VR-002).
    DotProduct {
        acc: VRegId,
        a: VRegId,
        b: VRegId,
        input_dtype: DotDtype,
        width: SimdWidth,
    },

    /// 累加器缩放: dst = (f32)acc * scale + zero。
    /// input_dtype 决定累加器类型转换指令: I32 → vcvtdq2ps, 其他 → 按 dtype 选择。
    /// Replaces ScaleApplyInt (dtype in param, not in name).
    ScaleApply {
        /// FP32 输出
        dst: VRegId,
        /// 累加器
        acc: VRegId,
        /// FP32 scale (broadcast)
        scale: VRegId,
        /// FP32 zero-point (broadcast, or NONE_VREG for no offset)
        zero: VRegId,
        /// 累加器输入数据类型
        input_dtype: QuantPrecision,
        width: SimdWidth,
    },

    /// Shared Memory Swizzle: 将线性地址转换为 XOR swizzle 地址以消除 bank conflict。
    ///
    /// 32-bank shared memory: 连续线程访问同一 bank 导致序列化。
    /// XOR swizzle: swizzled = raw_addr ^ ((raw_addr >> (log2_banks + log2_bank_width)) & ((1 << log2_banks) - 1))
    ///
    /// PTX (SM80+):
    ///   shr.u32 %tmp, %raw_addr, (log2_banks + log2_bank_width);
    ///   and.b32 %tmp, %tmp, ((1 << log2_banks) - 1);
    ///   xor.b32 %dst, %raw_addr, %tmp;
    ///
    /// x86_64: 无 shared memory，NOP（直接复制 raw_addr → dst）
    /// AArch64: 同 x86，NOP
    SharedMemSwizzle {
        /// 输出: swizzled 地址 (GPR, Ptr)
        dst: VRegId,
        /// 输入: 原始 shared memory 偏移 (GPR, Scalar or ByteOffset)
        raw_addr: VRegId,
        /// Shared memory bank 数量的 log2 (通常为 5 = 32 banks)
        log2_banks: u8,
        /// Bank 宽度字节 log2 (通常为 2 = 4 bytes per bank)
        log2_bank_width: u8,
    },

    /// Bitwise GEMM: TQ1_0 三值化 XOR + POPCNT batch dot product。
    ///
    /// 将 32 个 ±1 sign bits pack 为 u32，与 32 个 input sign bits XOR，
    /// POPCNT 得汉明距离，dot_product = (32 - 2 * hamming_distance) * scale。
    ///
    /// 1 条 POPCNT 指令完成 32 个乘法累加，32x 计算密度提升。
    ///
    /// GPU PTX: popc.b32 %dst, %xor_result
    /// x86: popcnt r32, r32 (SSE4.2+)
    /// AArch64: CNT (NEON) per-byte, then horizontal add
    BitwiseGemm {
        /// 输出: dot product 结果 (Scalar f32)
        dst: VRegId,
        /// Packed sign bits 寄存器 (Scalar u32 — 32 个 ±1 编码为 0/1 bits)
        sign_bits: VRegId,
        /// Packed input sign bits 寄存器 (Scalar u32)
        input_sign_bits: VRegId,
        /// Scale 值 (Scalar f32)
        scale: VRegId,
        /// SIMD width (determines lane count)
        width: SimdWidth,
    },

    /// GPU Sparse Tensor Core 2:4 结构化稀疏 GEMM (SM80+)。
    ///
    /// 权重矩阵经过 2:4 剪枝 (每 4 个连续元素中 2 个为零)，
    /// 硬件 Sparse Tensor Core 自动利用零元素实现 2x 吞吐。
    /// PTX: mma.sparse.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
    /// 需要 2-bit sparse mask (每对元素 1 bit: 0=第一元素非零, 1=第二元素非零)
    /// SM100+ 扩展: 支持 FP8 sparse GEMM
    /// CPU/ARM: 返回 Error (Sparse Tensor Core 仅 GPU SM80+)
    SparseGemm {
        /// 累加器寄存器 (F32, 输入/输出)
        acc: VRegId,
        /// 稀疏矩阵 A (已剪枝, F16/BF16, 2:4 格式)
        a_sparse: VRegId,
        /// 密集矩阵 B (F16/BF16)
        b_dense: VRegId,
        /// 2:4 稀疏掩码指针 (2-bit per element pair)
        sparse_mask_ptr: VRegId,
        /// M 维度 (行数)
        m: usize,
        /// N 维度 (列数)
        n: usize,
        /// K 维度 (内积维度, 必须是 32 的倍数)
        k: usize,
        /// SIMD 宽度 (Warp)
        width: SimdWidth,
        /// 数据类型
        dtype: QuantPrecision,
    },

    /// SM100 原生 FP4 矩阵乘法 (tcgen05.mma .f4)
    ///
    /// 硬件级 E2M1 4-bit 计算，零解量化，结果 F32 累加。
    /// 仅 SM100+ (DeviceProfile tensor_cores_gen >= 100)。
    /// CPU/ARM: 返回 Error
    /// SM80/SM90: 返回 Error (仅 SM100+)
    NativeFp4Gemm {
        /// 累加器 (F32, 输入/输出)
        acc: VRegId,
        /// A 矩阵 (FP4 E2M1 packed, 2 elements per byte)
        a: VRegId,
        /// B 矩阵 (FP4 E2M1 packed, 2 elements per byte)
        b: VRegId,
        /// A 块缩放因子 (E8M0 per 32-element block)
        scale_a: VRegId,
        /// B 块缩放因子 (E8M0 per 32-element block)
        scale_b: VRegId,
        /// M 维度
        m: usize,
        /// N 维度
        n: usize,
        /// K 维度 (必须是 32 的倍数，FP4 天然 2x throughput)
        k: usize,
        /// SIMD 宽度
        width: SimdWidth,
    },

    /// SM100+ Sparse FP8 Tensor Core GEMM (2:4 结构化稀疏 + FP8 同时生效)。
    ///
    /// 2:4 稀疏 + FP8 (E4M3/E5M2) 双重加速，相比 FP16 dense 4x 吞吐量。
    /// PTX: mma.sparse.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%acc}, {%a_sparse}, {%b_dense}, %meta
    /// 仅 SM100+ (DeviceProfile tensor_cores_gen >= 100)。
    /// CPU/ARM/SM80/SM90: 返回 Error
    SparseFp8Gemm {
        /// 累加器 (F32, 输入/输出)
        acc: VRegId,
        /// 稀疏矩阵 A (FP8 2:4 格式)
        a_sparse: VRegId,
        /// 密集矩阵 B (FP8)
        b_dense: VRegId,
        /// 2:4 稀疏掩码指针
        sparse_mask_ptr: VRegId,
        /// M 维度
        m: usize,
        /// N 维度
        n: usize,
        /// K 维度
        k: usize,
        /// SIMD 宽度
        width: SimdWidth,
        /// FP8 格式 (E4M3 或 E5M2)
        fp8_kind: Fp8Kind,
    },

    /// SM90+ 原生 FP8 Tensor Core GEMM。
    ///
    /// 硬件级 FP8 (E4M3/E5M2) 计算，F32 累加。吞吐量是 FP16 的 2x。
    /// PTX: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 (E4M3)
    /// PTX: mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 (E5M2)
    /// WGMMA: wgmma.mma_async::sync.m64nNk16.f32.e4m3.e4m3 (SM90+ warp-group MMA)
    /// 仅 SM90+ (DeviceProfile tensor_cores_gen >= 90)。
    /// CPU/ARM: 返回 Error
    /// SM80: 返回 Error (仅 SM90+)
    NativeFp8Gemm {
        /// 累加器 (F32, 输入/输出)
        acc: VRegId,
        /// A 矩阵 (FP8 E4M3 或 E5M2)
        a: VRegId,
        /// B 矩阵 (FP8 E4M3 或 E5M2)
        b: VRegId,
        /// M 维度
        m: usize,
        /// N 维度
        n: usize,
        /// K 维度 (必须是 16 的倍数 for FP8 mma.sync)
        k: usize,
        /// SIMD 宽度
        width: SimdWidth,
        /// FP8 格式 (E4M3 或 E5M2)
        fp8_kind: Fp8Kind,
    },

    /// SM100+ 硬件量化解量化。
    ///
    /// Tensor Core 直接消费量化权重（NVFP4 E2M1），在硬件内完成
    /// 解量化 x GEMM 累加，零额外指令开销。
    /// 仅 SM100+。
    /// CPU/ARM: 返回 Error
    HwQuantDequant {
        /// 目标寄存器 (F32 解量化结果)
        dst: VRegId,
        /// 量化权重 (4-bit packed, 2 elements per byte)
        packed_weight: VRegId,
        /// 块缩放因子 (E8M0 per block 或 UE4M3 per sub-block)
        block_scale: VRegId,
        /// 全局缩放因子 (可选, F32)
        global_scale: VRegId,
        /// 量化格式
        quant_kind: QuantPrecision,
        /// 元素数量
        count: usize,
        /// 宽度
        width: SimdWidth,
    },

    /// SM100+ TMEM 分配 (tcgen05.alloc.tmem)。
    ///
    /// 在 Tensor Memory 中分配指定大小的空间。仅 SM100+。
    /// ~1MB/SM, 延迟 <1ns, 用于 attention score 中间结果暂存、
    /// MoE expert 权重缓存、WGMMA 累加器溢出暂存。
    /// CPU/ARM: 返回 Error
    TmemAlloc {
        /// TMEM buffer 名称
        name: String,
        /// 分配大小 (字节)
        bytes: usize,
    },

    /// SM100+ TMEM 加载 (tcgen05.ld.tmem)。
    ///
    /// 从 Tensor Memory 加载数据到寄存器。延迟 <1ns。
    /// CPU/ARM: 返回 Error
    TmemLoad {
        /// 目标寄存器
        dst: VRegId,
        /// TMEM buffer 名称
        name: String,
        /// TMEM 内偏移
        offset: OffsetExpr,
        /// 宽度
        width: SimdWidth,
        /// 数据类型
        dtype: QuantPrecision,
    },

    /// SM100+ TMEM 存储 (tcgen05.st.tmem)。
    ///
    /// 将寄存器数据存储到 Tensor Memory。延迟 <1ns。
    /// CPU/ARM: 返回 Error
    TmemStore {
        /// TMEM buffer 名称
        name: String,
        /// TMEM 内偏移
        offset: OffsetExpr,
        /// 源寄存器
        src: VRegId,
        /// 宽度
        width: SimdWidth,
        /// 数据类型
        dtype: QuantPrecision,
    },

    /// SM100+ TMEM 释放 (tcgen05.dealloc_tmem)。
    ///
    /// 释放 Tensor Memory 中指定 buffer 的空间。仅 SM100+。
    /// CPU/ARM: 返回 Error
    TmemDealloc {
        /// TMEM buffer 名称
        name: String,
    },

    /// SM90+ Cluster barrier 初始化。
    /// PTX: barrier.cluster.init [name], thread_count
    ClusterBarrierInit {
        /// barrier 名称
        name: String,
        /// 参与的线程数
        thread_count: u32,
    },

    /// SM90+ Distributed Shared Memory store (跨 CTA 可见)。
    /// PTX: st.shared::cluster [name + offset], src
    ClusterStore {
        /// DSMEM buffer 名称
        name: String,
        /// 偏移
        offset: OffsetExpr,
        /// 源寄存器
        src: VRegId,
        /// 宽度
        width: SimdWidth,
        /// 数据类型
        dtype: QuantPrecision,
    },

    /// SM90+ Distributed Shared Memory load (跨 CTA 可读)。
    /// PTX: ld.shared::cluster dst, [name + offset]
    ClusterLoad {
        /// 目标寄存器
        dst: VRegId,
        /// DSMEM buffer 名称
        name: String,
        /// 偏移
        offset: OffsetExpr,
        /// 宽度
        width: SimdWidth,
        /// 数据类型
        dtype: QuantPrecision,
    },

    /// 注释 (调试/审计)
    Comment(String),

    // ── 页压缩解码 (SPEC 22-PAGE-COMPRESSION §3.3) ──

    /// LZ4 流解压: 将 [src_ptr, src_ptr+compressed_size) 解压到 dst_ptr。
    ///
    /// 格式: LZ4 token 流 — [token: u8][literal_len?][literal_bytes][match_offset: u16][match_len?]
    /// - token 高 4 位: literal_len 编码 (==15 → 读后续 0xFF 字节直到非 0xFF)
    /// - token 低 4 位: match_len 编码 (+4)
    ///
    /// x86_64: AVX2 16-byte 块复制加速 literal/match copy，主路径标量解析。
    /// AArch64: NEON 16-byte copy。
    /// GPU: 1 thread/token，warp 协作 match copy + 4KB shared memory 滑动窗口。
    ///
    /// REQ-COMP-003
    Lz4Decode {
        /// 压缩字节流起点 GPR
        src_ptr: VRegId,
        /// 解压目标 GPR (page 物理位置)
        dst_ptr: VRegId,
        /// 压缩字节数 GPR (来自 header.compressed_size)
        compressed_size: VRegId,
        /// 编译时已知解压后字节数 (page_size * elem_bytes)
        decompressed_size: usize,
    },

    /// Bit-pack + RLE 解码: nibble 流 (KIVI4/KIVI2) 展开到 dst_ptr。
    ///
    /// 格式: 每 byte 一个 run — [low nibble = run_value][high nibble = run_len]
    /// run_len==15 → escape: 后续 byte 是真实 run_len (直到 < 255)。
    ///
    /// x86_64: 标量解析 run header + AVX2 vdupq/vmovups 填充。
    /// AArch64: NEON vdupq_n_u8 填充。
    /// GPU: 1 thread/run，warp prefix-sum 计算 dst offset 并行展开。
    ///
    /// REQ-COMP-004
    BitPackRleDecode {
        /// 压缩字节流起点 GPR
        src_ptr: VRegId,
        /// 解压目标 GPR
        dst_ptr: VRegId,
        /// 压缩字节数 GPR
        compressed_size: VRegId,
        /// nibble 位宽: 4 (KIVI4) 或 2 (KIVI2)
        nibble_bits: u8,
        /// 解压后元素数 (page_size)
        element_count: usize,
    },

    // ── Layer 6: JIT Debug Instrumentation ──

    /// 软件断点 — 生成 INT3 (0xCC)，DAP attach 后在此暂停。
    /// 仅当 CompileConfig.debug_jit=true 时插入。
    DebugBreakpoint {
        /// 断点标签（如 "embed_done", "L0_qkv_start"）
        label: String,
    },

    /// 调试标记 — 生成 NOP，仅影响 source map（无运行时开销）。
    DebugMarker {
        /// 标记内容（如 "layer=0 op=q_proj phase=decode"）
        message: String,
    },

    /// 调试探针 — 将 VReg 值写入共享内存环形缓冲区。
    DebugProbe {
        /// 被观测的 VReg
        vreg: VRegId,
        /// 探针显示名称
        probe_name: String,
        /// 数据宽度
        width: SimdWidth,
    },

    /// 条件断点 — 仅当 GPR 非零时触发 INT3。
    DebugBreakIf {
        label: String,
        /// 条件 GPR（非零则断）
        cond_gpr: VRegId,
    },

    // ── REQ-VR-005~010: 缺失指令补全 ──

    /// 向量 lane 重排 (REQ-VR-005)。
    VecShuffle {
        dst: VRegId,
        src: VRegId,
        mask: VecShuffleMask,
        width: SimdWidth,
    },

    /// 向量 lane 提取到标量 (REQ-VR-006)。
    VecExtractLane {
        dst: VRegId,
        src: VRegId,
        lane: u8,
        dtype: crate::compiler::trace::QuantPrecision,
    },

    /// 标量插入到向量指定 lane (REQ-VR-006)。
    VecInsertLane {
        dst: VRegId,
        src_vec: VRegId,
        src_scalar: VRegId,
        lane: u8,
        dtype: crate::compiler::trace::QuantPrecision,
    },

    /// 向量常量加载 (REQ-VR-008)。
    VecLoadConst {
        dst: VRegId,
        values: Vec<u32>,
        dtype: crate::compiler::trace::QuantPrecision,
        width: SimdWidth,
    },

    /// 原子比较并交换 (REQ-VR-009)。
    AtomicCAS {
        dst: VRegId,
        ptr: VRegId,
        expected: VRegId,
        desired: VRegId,
        elem_width: usize,
        success_order: MemOrdering,
        failure_order: MemOrdering,
    },

    /// §20 BCI-004: Prefill per-token seq_id 查找 — cumsum(prompt_lens) 线性搜索。
    ///
    /// 给定 token_index（0..total_prefill_tokens），通过累加 prompt_lens 找到
    /// token_index 所属的序列 ID：
    ///   seq_id = min s where cumsum(prompt_lens)[s] > token_index
    ///
    /// 实现: 从 seq_meta_base 顺序读取 prompt_lens[0..num_seqs]，
    ///        累加直到 cumulative > token_index, 返回对应的 seq_id。
    ///
    /// x86: xor ecx; xor edx; .loop: cmp eax,edx; jl .found;
    ///      add edx,[seq_meta_base+seq*STRIDE+0]; inc ecx; jmp .loop; .found:
    SeqIdLookup {
        /// 输出: seq_id (GPR scalar)
        dst: VRegId,
        /// 输入: token_index (0..M_prefill) — GPR scalar (通常是 qi 循环 counter)
        token_index: VRegId,
        /// BatchContext.seq_meta_base 指针 (GPR ptr)
        seq_meta_base: VRegId,
        /// num_seqs — 搜索上界 (GPR scalar, 从 batch_ctx+0 加载)
        num_seqs: VRegId,
        /// SEQ_META_STRIDE (编译时常量, = 64)
        seq_meta_stride: usize,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // §1.6 分布式通信 VmInstr (REQ-VR-014, feature = "nccl")
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Chunk 级 AllReduce: 对 sendbuf 中 count 个元素执行归约操作，结果写入 recvbuf。
    ///
    /// x86_lower/aarch64_lower: 生成 `call [gllm_nccl_all_reduce_chunk_stub]` 桩调用。
    /// gpu_lower: 生成 PTX `call {all_reduce_chunk, ...}` 或 HIP 等价调用。
    #[cfg(feature = "nccl")]
    AllReduceChunk {
        /// GPU HBM 发送缓冲区指针
        sendbuf: VRegId,
        /// GPU HBM 接收缓冲区指针
        recvbuf: VRegId,
        /// 元素数量
        count: VRegId,
        /// 数据类型
        dtype: CommDType,
        /// 归约操作
        op: ReduceOp,
        /// 当前 rank ID
        rank: VRegId,
        /// 总 rank 数
        world_size: VRegId,
        /// chunk 索引（用于 Ring 算法定位）
        chunk_idx: VRegId,
    },

    /// 通信同步屏障: 通信 thread block 与计算 thread block 之间的同步。
    ///
    /// GPU 语义: `bar.sync barrier_id, thread_count`
    /// CPU 语义: NOP (通信由 GPU 驱动执行)
    #[cfg(feature = "nccl")]
    CommBarrier {
        /// 屏障标识（PTX named barrier: 0-15）
        barrier_id: u8,
        /// 参与同步的线程数
        thread_count: VRegId,
    },

    /// NVLink 异步块拷贝: 通过 NVLink 进行跨 GPU 异步数据传输。
    ///
    /// GPU 语义 (PTX):
    ///   cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    ///     [dst], [src], len, [barrier_addr]
    /// CPU 语义: NOP (通信由 GPU 驱动执行)
    #[cfg(feature = "nccl")]
    NvlinkAsyncCopy {
        /// 目标地址（GPU HBM，本端或远端 rank 映射地址）
        dst: VRegId,
        /// 源地址（GPU HBM）
        src: VRegId,
        /// 拷贝字节数（必须 16-byte 对齐，16 的倍数）
        len: VRegId,
        /// NVLink lane（0-17，H100 最多 18 lanes）
        lane: u8,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // §6.1 分布式分页 VmInstr (SPEC/08 DP-010, feature = "nccl")
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// 查分布式页表获取页物理位置。
    ///
    /// 输入: page_id (sequence_id, page_index) → 输出: location struct
    /// GPU 语义: 从 global memory 页表做 hash lookup
    #[cfg(feature = "nccl")]
    RemotePageLookup {
        /// 输出: 页位置结构体指针 (PageLocation enum)
        dst: VRegId,
        /// 输入: sequence_id
        seq_id: VRegId,
        /// 输入: page_index
        page_index: VRegId,
        /// 输入: 页路由表基地址
        routing_table_base: VRegId,
    },

    /// NVLink/XGMI 跨 GPU 页获取 (intra-node P2P)。
    ///
    /// GPU 语义 (PTX): cp.async.bulk.global.global [local], [peer], size, [barrier]
    #[cfg(feature = "nccl")]
    P2pPageFetch {
        /// 本地缓冲区 (HBM)
        local_buf: VRegId,
        /// 远端 GPU 缓冲区 (P2P mapped HBM)
        peer_buf: VRegId,
        /// 页大小 (bytes)
        page_size: VRegId,
        /// 完成屏障地址
        barrier: VRegId,
    },

    /// RDMA 跨节点页获取 (inter-node, uncompressed)。
    ///
    /// GPU 语义 (PTX): GPU-initiated RDMA READ via GPUDirect RDMA
    #[cfg(feature = "nccl")]
    RdmaPageFetch {
        /// 本地缓冲区 (HBM)
        local_buf: VRegId,
        /// 远端内存地址
        remote_addr: VRegId,
        /// 远端内存 key
        rkey: VRegId,
        /// 页大小 (bytes)
        page_size: VRegId,
        /// SQ 描述符地址
        sq_desc: VRegId,
        /// Doorbell 地址
        doorbell: VRegId,
        /// CQ 地址
        cq_addr: VRegId,
    },

    /// 带量化压缩的 RDMA 跨节点页获取。
    ///
    /// Pipeline: Quantize → Compress → RDMA → Decompress → Dequantize
    #[cfg(feature = "nccl")]
    RdmaPageFetchCompressed {
        /// 本地缓冲区 (HBM)
        local_buf: VRegId,
        /// Scratch 缓冲区 (用于中间量化/压缩结果)
        scratch_buf: VRegId,
        /// 页大小 (bytes, original)
        page_size: VRegId,
        /// 远端内存地址
        remote_addr: VRegId,
        /// 远端内存 key
        rkey: VRegId,
        /// SQ 描述符地址
        sq_desc: VRegId,
        /// Doorbell 地址
        doorbell: VRegId,
        /// CQ 地址
        cq_addr: VRegId,
        /// 量化方案 (0=None, 1=FP8E4M3, 2=FP8E5M2, 3=INT8)
        quant_scheme: u8,
        /// 压缩算法 (0=None, 1=BitPack, 2=LZ4)
        compress_algorithm: u8,
    },

    /// 远程 KV 页直接 Attention (不迁移，就地远程读取)。
    ///
    /// NVLink 场景: cp.async.bulk 从 peer GPU 读 K/V tile
    #[cfg(feature = "nccl")]
    RemotePageAttn {
        /// Q 缓冲区 (本地 HBM)
        q_buf: VRegId,
        /// K 远端缓冲区 (P2P mapped 或 RDMA mapped)
        k_remote_buf: VRegId,
        /// V 远端缓冲区
        v_remote_buf: VRegId,
        /// 输出缓冲区 (本地 HBM)
        output_buf: VRegId,
        /// Shared memory 中间缓冲区
        shared_buf: VRegId,
        /// 完成屏障地址
        barrier: VRegId,
        /// Attention tile 大小 (bytes)
        tile_bytes: VRegId,
    },

    /// 获取页迁移锁 (原子 CAS)。
    ///
    /// 防止并发迁移同一页。成功 → dst=true, 失败 → dst=false
    #[cfg(feature = "nccl")]
    PageMigrationLock {
        /// 输出: 是否成功获取锁
        dst: VRegId,
        /// 页路由表条目地址
        entry_addr: VRegId,
    },

    /// 释放页迁移锁。
    #[cfg(feature = "nccl")]
    PageMigrationUnlock {
        /// 页路由表条目地址
        entry_addr: VRegId,
    },

    /// 更新页路由表条目 (metadata write)。
    ///
    /// 迁移完成后更新 location 和 state 字段
    #[cfg(feature = "nccl")]
    PageLocationUpdate {
        /// 页路由表条目地址
        entry_addr: VRegId,
        /// 新 location 数据地址
        new_location: VRegId,
        /// 新 state (0=Ready, 1=InTransit, 2=Invalid)
        new_state: u8,
    },
}

impl VmInstr {
    /// 元操作不需要 TraceOp provenance。
    pub fn is_meta(&self) -> bool {
        matches!(self,
            Self::DeclareVReg { .. }
            | Self::ReleaseVReg { .. }
            | Self::MarkLabel { .. }
            | Self::Comment(_)
            | Self::ScopeBegin { .. }
            | Self::ScopeEnd { .. }
            | Self::DebugBreakpoint { .. }
            | Self::DebugMarker { .. }
        )
    }
}
