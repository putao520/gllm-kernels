use crate::compiler::trace::QuantPrecision;
use crate::types::DType;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §0 FP8 格式变体
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// FP8 格式变体 (OCP standard, NVIDIA/AMD hardware-native 8-bit float)。
///
/// SM90+ (Hopper H100/H200) Tensor Core 原生支持两种 FP8 格式，
/// 吞吐量是 FP16 的 2x。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Kind {
    /// E4M3: 指数 4-bit, 尾数 3-bit (范围 ±448, 精度高, 适合前向传播 weight/activation)。
    E4M3,
    /// E5M2: 指数 5-bit, 尾数 2-bit (范围 ±57344, 精度低, 适合梯度/反向传播)。
    E5M2,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 虚拟寄存器
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 虚拟寄存器 ID——无限供应，由 RegAllocator 映射到物理寄存器。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VRegId(pub u32);

/// 虚拟寄存器用途——影响物理分配策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VRegKind {
    /// 标量指针 (→ GPR64)
    Ptr,
    /// SIMD 向量 (→ YMM/ZMM/V.4S/...)
    Vec,
    /// 标量值 (→ GPR64 或 XMM lane 0)
    Scalar,
    /// 循环计数器 (→ GPR64, 生命周期 = 循环体)
    Counter,
    /// 循环字节偏移 (→ GPR64, 与 Counter 关联, = counter × step_bytes)
    ByteOffset,
    /// Tile 寄存器 (→ AMX tile / SME ZA / Tensor Core fragment)
    Tile,
    /// 掩码 (→ k-mask / SVE predicate / GPU predicate)
    Mask,
}

/// SIMD 向量宽度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdWidth {
    Scalar,
    W128,        // SSE/NEON: 4×f32
    W256,        // AVX2: 8×f32
    W512,        // AVX-512: 16×f32
    Warp(u32),   // GPU: 32 (NVIDIA) / 64 (AMD)
    Scalable,    // SVE: runtime VL
}

impl SimdWidth {
    pub fn f32_lanes(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::W128 => 4,
            Self::W256 => 8,
            Self::W512 => 16,
            Self::Warp(n) => n as usize,
            Self::Scalable => 0, // runtime
        }
    }

    pub fn bytes(self) -> usize {
        self.f32_lanes() * 4
    }
}

/// TMA shared memory swizzle 模式 (SM90+)。
/// 控制 TMA 硬件对 shared memory 的 bank conflict 消除策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmaSwizzle {
    /// 无 swizzle
    None,
    /// 32B swizzle (4 个 128-bit banks)
    Swizzle32,
    /// 64B swizzle
    Swizzle64,
    /// 128B swizzle (最大，推荐用于大 tile)
    Swizzle128,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 表达式类型 (编译时推导，不允许硬编码)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 偏移表达式——从布局参数推导，ISA Lower 时计算。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OffsetExpr {
    /// 编译时常量 (从 Layout 推导)
    Const(usize),
    /// 循环的当前字节偏移 (= loop_counter × step_bytes)
    LoopOffset(VRegId),
    /// 两个偏移的和
    Add(Box<OffsetExpr>, Box<OffsetExpr>),
    /// 偏移乘以常量
    Mul(Box<OffsetExpr>, usize),
    /// 运行时标量 VReg 作为字节偏移 (Gather 间接寻址)
    ScalarVReg(VRegId),
}

impl OffsetExpr {
    /// 将 LoopOffset(vreg) 替换为 Const(value)——用于循环展开。
    pub fn substitute_loop_offset(&self, vreg: VRegId, value: usize) -> Self {
        match self {
            Self::LoopOffset(v) if *v == vreg => Self::Const(value),
            Self::Add(a, b) => Self::Add(
                Box::new(a.substitute_loop_offset(vreg, value)),
                Box::new(b.substitute_loop_offset(vreg, value)),
            ),
            Self::Mul(inner, scale) => Self::Mul(
                Box::new(inner.substitute_loop_offset(vreg, value)),
                *scale,
            ),
            // ScalarVReg is a runtime value, not a loop offset — pass through
            other => other.clone(),
        }
    }

    /// 快捷构造: const + loop_offset
    pub fn loop_plus_const(loop_vreg: VRegId, c: usize) -> Self {
        if c == 0 {
            Self::LoopOffset(loop_vreg)
        } else {
            Self::Add(
                Box::new(Self::LoopOffset(loop_vreg)),
                Box::new(Self::Const(c)),
            )
        }
    }
}

/// 指针来源表达式——从 ABI 或栈参数加载地址。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtrExpr {
    /// ABI 入参寄存器 (x86 SysV: rdi=0,rsi=1,rdx=2,rcx=3,r8=4,r9=5)
    AbiArg(u8),
    /// 栈上参数 ([rbp+offset] / [sp+offset])
    StackArg(i32),
    /// 已加载的 VReg + 常量偏移
    VRegPlusConst(VRegId, usize),
    /// 两个 VReg 相加 (base + offset_vreg)——用于嵌套循环的行基地址计算
    VRegPlusVReg(VRegId, VRegId),
    /// VReg + 动态偏移 (base + offset_expr)——用于 tile 内位置寻址
    VRegPlusOff(VRegId, OffsetExpr),
    /// 命名参数 (ARCH-VM-QUERY-NOT-ASSUME): ISA Lower 时通过 SymDimSlotMap 解析。
    /// 用途: opt_pass 等不持有 VmState 的上下文，用符号名引用 ABI 参数。
    /// 例如 `NamedArg("telemetry")` 在 x86 SysV 下解析为 `StackArg(40)`。
    NamedArg(String),
    /// 片上共享内存 scratchpad 基址 (ARCH-GPU-SHARED-SCRATCH)。
    /// GPU kernel 里 scratchpad 位于 `.shared`/`__shared__`/`threadgroup` 声明,
    /// 无法通过 `.param` 传入 — 需要符号引用 (如 PTX `mov.u64 rd, smem`)。
    /// 在 CPU ABI 下非法 (CPU scratchpad 是堆指针 `StackArg`),使用应在 cpu lower 层报错。
    SharedMem,
    /// 绝对地址立即数 (ARCH-SG-QTAP)。
    ///
    /// 编译期已知的 64-bit 主机虚拟地址（或设备指针），由 codegen 直接嵌入机器码。
    /// 用于 Semantic Gatekeeper 的 ring buffer / step_index 指针 — 这类地址在
    /// JIT 编译时已确定，且不由 ABI 传入；因此需要 `mov reg, imm64` 式加载。
    ///
    /// x86_64 lower: `mov rd, imm64` (10 字节 REX.W + B8+rd + imm64)。
    /// AArch64 lower: `MOVZ rd, imm16 #0 ; MOVK rd, imm16 #16 ; MOVK ... #32/#48`。
    AbsAddr(u64),
}

/// 标量值来源——用于 broadcast。
#[derive(Debug, Clone)]
pub enum ScalarExpr {
    /// 编译时常量
    Const(f32),
    /// 从内存加载标量: [base_vreg + offset]
    MemLoad(VRegId, OffsetExpr),
    /// 从 SIMD 向量提取 lane 0
    ExtractLane0(VRegId),
    /// 从标量 VReg 读取 (VRegKind::Scalar, 即 xmm/S 寄存器中的 float)
    /// 用于 IndexToScalar 结果的广播
    VReg(VRegId),
}

/// 符号化循环上界 (ARCH-SYMDIM-THREADING §3.1)。
///
/// 携带符号维度的名称和编译时分配上界。
/// lower 函数生成 `BoundExpr::Symbolic(SymBound)`，
/// ISA Lower 阶段通过 `SymDimSlotMap` 解析为物理位置。
#[derive(Debug, Clone, PartialEq)]
pub struct SymBound {
    /// 符号维度名称（与 SymDim::Symbolic.name 一致）
    pub name: String,
    /// 编译时分配上界（buffer 按此值分配）
    pub max_alloc: usize,
}

/// 循环上界表达式。
#[derive(Debug, Clone, PartialEq)]
pub enum BoundExpr {
    /// 编译时常量 — 硬编码到机器码（展开/优化目标）
    Const(usize),
    /// 运行时值 — 从栈参数/ABI 寄存器读取（低级，显式位置）
    Runtime(PtrExpr),
    /// 符号化运行时上界 — 通过 SymDimSlotMap 在 ISA Lower 时绑定到物理位置
    /// (ARCH-SYMDIM-THREADING §3.2)
    Symbolic(SymBound),
    /// 动态 VReg 上界: counter < vreg_value (ARCH-CAUSAL)。
    /// 绑定到外层循环计数器，语义与其他变体一致 (strict less-than)。
    /// ISA Lower 生成 `cmp counter_reg, outer_counter_reg; jge done`。
    DynamicVReg(VRegId),
    /// 动态 VReg+1 上界: counter < vreg_value + 1, 即 counter ≤ vreg_value (ARCH-CAUSAL)。
    /// 用途: causal attention 中 ki ≤ qi（ki 可以等于 qi）。
    /// ISA Lower 生成 `lea tmp, [outer_counter + 1]; cmp counter_reg, tmp; jge done`。
    DynamicVRegPlusOne(VRegId),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 算术/归约/超越函数 操作枚举
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// SIMD 二元操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    And,
    Or,
    Xor,
    AndNot,
    Shl,
    Shr,
    Not,
}

/// 立即数向量移位方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecShiftDir {
    Left,
    Right,
}

/// SIMD 一元操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecUnaryOp {
    Neg,
    Abs,
    Sqrt,
    Rsqrt,
    Recip,
    Floor,
    Ceil,
    Round,
    /// i32 → f32 vector convert (vcvtdq2ps / scvtf)
    IntToFloat,
    /// FP8 E4M3 → F32 vector convert (software bit-manipulation on x86, hardware on SM100+)
    Fp8E4M3ToFloat,
    /// FP8 E5M2 → F32 vector convert (software bit-manipulation on x86, hardware on SM100+)
    Fp8E5M2ToFloat,
}

/// 归约操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Prod,
    /// log(sum(exp(x))) — softmax 分母 (数值稳定版: max + log1p(sum(exp(x-max))))
    LogSum,
}

/// 通信数据类型 — AllReduceChunk 等分布式通信指令的数据类型 (REQ-VR-014, feature = "nccl").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "nccl")]
pub enum CommDType {
    Fp32,
    Fp16,
    Bf16,
    Fp8,
    Int8,
}

/// 比较谓词 — 逐元素向量比较。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpPredicate {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// 超越函数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscendentalFn {
    Exp,
    Log,
    Tanh,
    Sigmoid,
    /// §11 TurboQuant: Fast Walsh-Hadamard Transform
    Fwht,
}

/// 内存屏障强度 (ARCH-SG-QTAP)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemFenceOrder {
    /// Release: 屏障前的写对后续 Acquire-load 可见 (x86 TSO 下无需真实屏障,
    /// AArch64 发射 `DMB ISHST`)。
    Release,
    /// Acquire: 屏障后的读看到前面 Release-store 的值。
    Acquire,
    /// Acquire + Release。
    AcqRel,
    /// Sequentially consistent (强于 AcqRel; x86 `mfence`, AArch64 `DMB ISH`).
    SeqCst,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 热修补 / MoE / EarlyExit 关联类型
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// §9.2 热修补目标。
#[derive(Debug, Clone)]
pub enum HotpatchTarget {
    /// 跳转到指定 VmInstr 索引
    InstrIndex(usize),
    /// 外部函数地址 (运行时填充)
    ExternalAddr(u64),
}

/// §15 MoE 跳转目标。
#[derive(Debug, Clone)]
pub struct JumpTarget {
    /// 专家 ID
    pub expert_id: usize,
    /// 跳转到的 VmInstr 索引
    pub instr_index: usize,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 26: VmInstr 去类型化辅助枚举
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// GPR 二元操作类型 (REQ-VR-003)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GprOp {
    Add, Sub, Mul, Div, Shl, Shr, And, Or, Xor, BitTest,
}

/// GPR 操作数 — VReg 或 立即数 (REQ-VR-003)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GprOperand {
    VReg(VRegId),
    Imm(i64),
}

impl GprOperand {
    pub fn remap<F: Fn(VRegId) -> VRegId>(self, f: F) -> Self {
        match self {
            GprOperand::VReg(v) => GprOperand::VReg(f(v)),
            GprOperand::Imm(v) => GprOperand::Imm(v),
        }
    }

    pub fn vreg(&self) -> Option<VRegId> {
        match self { GprOperand::VReg(v) => Some(*v), GprOperand::Imm(_) => None }
    }
}

/// GPR 一元操作类型 (REQ-VR-010)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GprUnaryOpKind {
    Not, Popcount, Clz, Bswap, Neg,
}

/// GPR 条件判断 (REQ-VR-003)。
#[derive(Debug, Clone)]
pub enum GprCondition {
    IsNull(VRegId),
    IsNonNull(VRegId),
    CmpEq(VRegId, u64),
    CmpLtU(VRegId, u64),
    CmpGeU(VRegId, u64),
    BitClear(VRegId, u8),
    BitSet(VRegId, u8),
}

impl GprCondition {
    pub fn remap<F: Fn(VRegId) -> VRegId>(self, f: F) -> Self {
        match self {
            GprCondition::IsNull(v) => GprCondition::IsNull(f(v)),
            GprCondition::IsNonNull(v) => GprCondition::IsNonNull(f(v)),
            GprCondition::CmpEq(v, c) => GprCondition::CmpEq(f(v), c),
            GprCondition::CmpLtU(v, c) => GprCondition::CmpLtU(f(v), c),
            GprCondition::CmpGeU(v, c) => GprCondition::CmpGeU(f(v), c),
            GprCondition::BitClear(v, b) => GprCondition::BitClear(f(v), b),
            GprCondition::BitSet(v, b) => GprCondition::BitSet(f(v), b),
        }
    }

    pub fn vregs(&self) -> Vec<VRegId> {
        match self {
            GprCondition::IsNull(v) | GprCondition::IsNonNull(v) => vec![*v],
            GprCondition::CmpEq(v, _) | GprCondition::CmpLtU(v, _) | GprCondition::CmpGeU(v, _) => vec![*v],
            GprCondition::BitClear(v, _) | GprCondition::BitSet(v, _) => vec![*v],
        }
    }
}

/// GPR 条件行为 (REQ-VR-003)。
#[derive(Debug, Clone)]
pub enum GprBranchAction {
    Skip(usize),
    Exit(VRegId),
    /// Jump to a MarkLabel with the given label_id (SPEC 32 ForwardPhaseDispatch).
    JumpToLabel(usize),
}

impl GprBranchAction {
    pub fn remap<F: Fn(VRegId) -> VRegId>(self, f: F) -> Self {
        match self {
            GprBranchAction::Skip(n) => GprBranchAction::Skip(n),
            GprBranchAction::Exit(v) => GprBranchAction::Exit(f(v)),
            GprBranchAction::JumpToLabel(id) => GprBranchAction::JumpToLabel(id),
        }
    }

    pub fn vregs(&self) -> Vec<VRegId> {
        match self {
            GprBranchAction::Skip(_) => vec![],
            GprBranchAction::Exit(v) => vec![*v],
            GprBranchAction::JumpToLabel(_) => vec![],
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §Predicate + MemEffect (FAT-OPCODE-ARCHITECTURE-V2 §3 — VmInstr 自描述)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 内存指令条件谓词 (REQ-VR Phase 8)。
///
/// 复合 VmInstr (MemCopy/VecLoad/VecStore/GatherLoad/ScatterStore) 内化
/// 条件谓词，使每条内存指令自描述执行条件。
/// - None: 无条件执行（默认）
/// - Some(cond): 仅当条件成立时执行（x86: TEST+JZ skip; ARM: CBZ skip）
///
/// 替代旧"前置 GprCondAction + 内存指令"模式，简化 lowering 路径。
// @trace REQ-FATOP-011 [entity:Predicate] Predicate 类型（基于 GprCondition）
#[derive(Debug, Clone)]
pub struct Predicate(pub GprCondition);

impl Predicate {
    pub fn new(cond: GprCondition) -> Self {
        Self(cond)
    }

    pub fn inner(&self) -> &GprCondition {
        &self.0
    }

    pub fn remap<F: Fn(VRegId) -> VRegId>(&self, f: F) -> Self {
        Self(self.0.clone().remap(f))
    }

    pub fn vregs(&self) -> Vec<VRegId> {
        self.0.vregs()
    }
}

/// 内存效应类型 (REQ-VR Phase 8)。
///
/// 描述内存指令对内存的访问模式，用于：
/// 1. 寄存器分配的内存依赖分析
/// 2. 指令调度的重排序边界
/// 3. alias analysis（多指针别名推理）
// @trace REQ-FATOP-012 [entity:MemEffect] MemEffect 类型（Read/Write/ReadWrite）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemEffect {
    /// 纯读操作（如 GatherLoad）
    Read,
    /// 纯写操作（如 ScatterStore）
    Write,
    /// 读写操作（如 MemCopy 既是读 src 又是写 dst）
    ReadWrite,
}

impl MemEffect {
    pub fn reads(self) -> bool {
        matches!(self, MemEffect::Read | MemEffect::ReadWrite)
    }

    pub fn writes(self) -> bool {
        matches!(self, MemEffect::Write | MemEffect::ReadWrite)
    }
}

/// Dot-product 输入 dtype (REQ-VR-002)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotDtype {
    Bf16, Fp16, Int8, Int4x8, Fp4,
}

/// GGUF 量化块解包模式 (REQ-VR-001)。
#[derive(Debug, Clone)]
pub enum BlockUnpackMode {
    Int8,
    F16Broadcast,
    SignedNibbleLow,
    UnsignedNibbleLow,
    SignedNibbleHigh,
    UnsignedNibbleHigh,
    Bitpack2 { bias: f32 },
    Mxfp4 { scale_src: VRegId },
    Nvfp4 { scale_src: VRegId },
    /// Q5_0/Q5_1 high-bit plane expand: load 1 byte from base, expand each of 8 bits
    /// to f32 (0.0 or `bit_value`). Used for INT5/INT6 qh plane in two-phase GEMV.
    QhBitExpand { bit_value: f32 },
}

impl BlockUnpackMode {
    pub fn remap_vregs(&self, mut f: impl FnMut(VRegId) -> VRegId) -> Self {
        match self {
            Self::Mxfp4 { scale_src } => Self::Mxfp4 { scale_src: f(*scale_src) },
            Self::Nvfp4 { scale_src } => Self::Nvfp4 { scale_src: f(*scale_src) },
            other => other.clone(),
        }
    }

    pub fn vregs(&self) -> Vec<VRegId> {
        match self {
            Self::Mxfp4 { scale_src } => vec![*scale_src],
            Self::Nvfp4 { scale_src } => vec![*scale_src],
            _ => vec![],
        }
    }

    /// 量化块的字节大小（数据部分，不含 scale/zp metadata）。
    /// 用于 verify.rs 规则 8 的编译时偏移对齐检查。
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::Int8 => 32,
            Self::F16Broadcast => 64,
            Self::SignedNibbleLow | Self::UnsignedNibbleLow
            | Self::SignedNibbleHigh | Self::UnsignedNibbleHigh => 18,
            Self::Bitpack2 { .. } => 8,
            Self::Mxfp4 { .. } => 16,
            Self::Nvfp4 { .. } => 16,
            Self::QhBitExpand { .. } => 1,
        }
    }
}

/// GGUF 多平面位合并模式 (REQ-VR-001)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiPlaneMode {
    Low5, Low6, Q3Merge,
}

/// 标量转换源类型 (REQ-VR-004)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarCvtSource {
    F16, I8, U8,
}

/// 向量 shuffle 掩码 (REQ-VR-005)。
#[derive(Debug, Clone)]
pub enum VecShuffleMask {
    Const(Vec<u8>),
    Dynamic { ctrl: VRegId },
}

impl VecShuffleMask {
    pub fn remap<F: Fn(VRegId) -> VRegId>(&self, f: &F) -> Self {
        match self {
            VecShuffleMask::Const(v) => VecShuffleMask::Const(v.clone()),
            VecShuffleMask::Dynamic { ctrl } => VecShuffleMask::Dynamic { ctrl: f(*ctrl) },
        }
    }
}

/// 内存排序语义 (REQ-VR-009)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5 VmInstr — 半结构化状态追踪记录
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// BreakLoop 返回值来源。
#[derive(Debug, Clone, Copy)]
pub enum ReturnValue {
    /// 返回编译时常量 (classify/encode 模式返回 0)。
    Const(u32),
    /// 返回 VReg 的当前值 (generate 模式返回 gen_counter)。
    VReg(VRegId),
}

/// 编译时状态追踪记录。
///
/// 每条记录描述一次寄存器/内存/栈状态转移，基于虚拟寄存器 (VRegId)。
/// RegAllocator 读取全序列计算活跃区间 → 物理寄存器映射。
/// IsaLower 遍历全序列，将每条状态记录翻译为物理机器码。

/// KV cache load mode for attention variant compilation (KV-OPT-009).
///
/// Different PrecisionTier pages require different load+dequant sequences
/// in the Attention KV load micro-kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvLoadMode {
    /// Standard f32/f16 load — no dequantization needed
    #[default]
    Direct,
    /// Load 4-bit KIVI packed data, dequantize per-channel
    Kivi4,
    /// Load 2-bit KIVI packed data, dequantize per-token
    Kivi2,
    /// Load with bitmap skip — zero out channels marked in bitmap
    Sparse,
    /// Runtime tier dispatch — read precision_tier from KvPageHeader and
    /// select the appropriate load path at runtime (FP16→Direct, FP8→Direct,
    /// KIVI4→KiviDequantLoad, KIVI2→KiviDequantLoad, Sparse→sparse_masked_load).
    Auto,
}
