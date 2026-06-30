use crate::quant::QuantType;
use crate::quant_format::{PackedScaleAlgorithm, ZeroLayout};

/// Scalar + SymExec output: an operator's complete computational structure.
#[derive(Debug, Clone)]
pub struct OpTrace {
    /// Structural classification + SSA body.
    pub pattern: ComputePattern,
    /// Original scalar function pointer and parameter layout.
    pub signature: ScalarFnSignature,
}

/// Second-pass reduction descriptor (e.g. exp-sum in Softmax).
#[derive(Debug, Clone)]
pub struct ReductionSecondPass {
    /// Identity element for the second reduction (e.g. 0.0 for sum).
    pub identity: f64,
    /// Per-element transform before combining.
    /// Input(0) = current element, Input(1) = broadcast scalar from first pass.
    pub element_transform: Vec<TraceOp>,
    /// Combine step for the second reduction (e.g. add for sum).
    /// Input(0) = accumulator, Input(1) = transformed element.
    pub combine: Vec<TraceOp>,
}

/// Computational pattern — determines how ISA Lowering vectorizes the operator.
#[derive(Debug, Clone)]
// @trace REQ-AIS-002 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/compute-pattern]
pub enum ComputePattern {
    /// `out[i] = f(in[i])` — single-input elementwise.
    Elementwise { body: Vec<TraceOp> },
    /// `out[i] = f(a[i], b[i])` — dual-input elementwise.
    BinaryElementwise { body: Vec<TraceOp> },
    /// Multi-input/multi-output elementwise (e.g. RoPE).
    Injective {
        body: Vec<TraceOp>,
        num_inputs: usize,
        num_outputs: usize,
    },
    /// Reduction with identity element and combine step.
    ///
    /// For multi-pass reductions (e.g. Softmax = max → exp-sum → normalize):
    /// - `combine`: first-pass reduction (e.g. max)
    /// - `second_pass`: optional second-pass with per-element transform + reduction
    /// - `normalize`: optional final per-element normalization
    Reduction {
        identity: f64,
        combine: Vec<TraceOp>,
        /// Second reduction pass (e.g. exp-sum in Softmax).
        second_pass: Option<Box<ReductionSecondPass>>,
        /// Final per-element normalization (e.g. multiply by inv_sum).
        /// Input(0) = element from second_pass, Input(1) = broadcast scalar from second_pass reduction.
        normalize: Option<Vec<TraceOp>>,
    },
    /// Two-pass normalize: reduce → finalize → per-element transform.
    NormLike {
        reduce: Vec<TraceOp>,
        finalize: Vec<TraceOp>,
        transform: Vec<TraceOp>,
    },
    /// Triple-loop matrix multiply (GEMM).
    Gemm,
    /// Quantization decode with fixed block size.
    QuantDecode {
        block_size: usize,
        decode: Vec<TraceOp>,
    },
}

impl ComputePattern {
    /// Return the primary computation body, if this pattern has one.
    ///
    /// For elementwise patterns this is the single body; for multi-phase
    /// patterns (NormLike, Reduction) this returns `None` — those require
    /// specialized codegen paths.
    pub fn body(&self) -> Option<&[TraceOp]> {
        match self {
            ComputePattern::Elementwise { body } => Some(body),
            ComputePattern::BinaryElementwise { body } => Some(body),
            ComputePattern::Injective { body, .. } => Some(body),
            ComputePattern::QuantDecode { decode, .. } => Some(decode),
            ComputePattern::Gemm
            | ComputePattern::Reduction { .. }
            | ComputePattern::NormLike { .. } => None,
        }
    }
}

/// Analyze a TraceOp sequence and classify its ComputePattern.
///
/// Rules:
/// - Empty body → `Injective` (layout-only op)
/// - Only `Input(0)` + unary/const ops → `Elementwise`
/// - `Input(0)` + `Input(1)` present → `BinaryElementwise`
/// - 3+ distinct inputs → `Injective`
// @trace REQ-AIS-007 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/classify-pattern]
pub fn classify_pattern(body: &[TraceOp]) -> ComputePattern {
    if body.is_empty() {
        return ComputePattern::Injective {
            body: vec![],
            num_inputs: 0,
            num_outputs: 1,
        };
    }

    let max_input = body.iter().filter_map(|op| {
        if let TraceOp::Input(idx) = op { Some(*idx) } else { None }
    }).max();

    let num_inputs = match max_input {
        Some(idx) => (idx + 1) as usize,
        None => 0,
    };

    match num_inputs {
        0 | 1 => ComputePattern::Elementwise { body: body.to_vec() },
        2 => ComputePattern::BinaryElementwise { body: body.to_vec() },
        _ => ComputePattern::Injective {
            body: body.to_vec(),
            num_inputs,
            num_outputs: 1,
        },
    }
}

/// FP8 format variant — determines exponent bias and mantissa width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Format {
    /// E4M3: bias=7, 3 mantissa bits, no Inf/NaN.
    E4M3,
    /// E5M2: bias=15, 2 mantissa bits, supports Inf/NaN.
    E5M2,
}

/// K-Quant scale selector — determines which value is extracted from the packed `scales[]` array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleSelector {
    /// Extract the sub-block scale (d × scale factor).
    Scale,
    /// Extract the sub-block min/zero-point value.
    Min,
}

/// SSA value handle — stable, name-based reference to a TraceOp result.
///
/// Unlike raw `u32` array indices, ValueId is a typed handle that:
/// - Survives reordering/insertion/deletion of TraceOps
/// - Enables O(1) lookup via SecondaryMap (no HashMap overhead)
/// - Makes references self-documenting (`Fma(a, b, acc)` vs `Fma(3, 7, 12)`)
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ValueId(pub u32);

impl ValueId {
    pub const NONE: ValueId = ValueId(u32::MAX);
    pub fn is_some(self) -> bool { self.0 != u32::MAX }
    pub fn saturating_sub(self, n: u32) -> ValueId { ValueId(self.0.saturating_sub(n)) }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "v{}", self.0) }
}

impl std::ops::Sub<u32> for ValueId {
    type Output = ValueId;
    fn sub(self, rhs: u32) -> ValueId { ValueId(self.0 - rhs) }
}

/// SSA-form computation operation.
///
/// Each variant's `ValueId` fields are stable handles referencing a prior
/// operation's output. Unlike positional `u32` indices, `ValueId` survives
/// reordering/insertion/deletion of TraceOps in the body.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceOp {
    /// Load the i-th input element.
    Input(u32),
    /// Floating-point constant.
    Const(f64),

    // ── Arithmetic ──
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    /// Power: base ^ exp.
    Pow(ValueId, ValueId),
    /// Fused multiply-add: a * b + c
    Fma(ValueId, ValueId, ValueId),

    // ── Unary ──
    Neg(ValueId),
    Abs(ValueId),
    Exp(ValueId),
    Sqrt(ValueId),
    Rsqrt(ValueId),
    Tanh(ValueId),
    Recip(ValueId),
    /// Natural logarithm: ln(x).
    Log(ValueId),
    /// Sigmoid: 1 / (1 + exp(-a)).
    /// 后端映射: TranscendentalFn::Sigmoid (x86 Exp+Add+Div 软件 / GPU 原生)
    Sigmoid(ValueId),
    Max(ValueId, ValueId),
    Min(ValueId, ValueId),
    /// Conditional select: (mask != 0.0) ? true_val : false_val, per-lane.
    // @trace REQ-AIS-006 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/conditional-select]
    ConditionalBranch(ValueId, ValueId, ValueId),

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ARCH-JIT-GENERATOR §12+§14: 扩展 TraceOp
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ── 量化混合精度 (§11 TurboQuant / §13.12 硬件拓扑) ──

    /// 混合精度 FMA: 不同位宽的 act × weight 累加到 acc。
    /// 后端映射: gfx950 mfma_scale / SM100 tcgen05 / AMX-FP8 TDPFP8PS
    QuantFma {
        acc: ValueId,
        act: ValueId,
        weight: ValueId,
        act_dtype: QuantPrecision,
        weight_dtype: QuantPrecision,
    },

    /// Block exponent scaling (CDNA4 gfx950 mfma_scale 原生支持)。
    /// 对 data 的每个 block_size 元素组应用共享的 scale 因子。
    BlockScale {
        data: ValueId,
        scale: ValueId,
        block_size: usize,
    },

    /// 类型转换: F16C vcvtph2ps / ARM fcvtl / PTX cvt.f32.f16
    // @trace REQ-AIS-003 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/cast]
    Cast {
        src: ValueId,
        from: QuantPrecision,
        to: QuantPrecision,
    },

    // ── 水平归约 (§13 Epilogue 白嫖) ──

    /// 水平归约: 将向量寄存器归约为标量。
    /// 后端映射: x86 shuffle+hadd / ARM faddp+addv / GPU shfl.sync warp reduce
    // @trace REQ-AIS-004 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/hreduce]
    HReduce {
        src: ValueId,
        op: ReduceKind,
    },

    // ── 内存层级控制 (§13.2 质心预取 / §11 TurboQuant) ──

    /// Prefetch hint: 预取到指定缓存层级。
    /// 后端映射: prefetcht0/t1/nta / prfm pldl1keep / prefetch.global.L2
    Prefetch {
        level: CacheLevel,
    },

    /// Non-temporal store: 绕过缓存写入（KV 大量连续写入场景）。
    /// 后端映射: vmovntps / stnp / st.global.cs
    NonTemporalStore,

    // ── 位操作 (量化解包) ──

    BitExtract {
        src: ValueId,
        offset: u32,
        width: u32,
    },

    /// 向量排列/洗牌: 按 indices 重排 src 的元素。
    /// 后端映射: vpshufb/vpermd / tbl/tbx / prmt
    Permute {
        src: ValueId,
        indices: ValueId,
    },

    // ── 比较和掩码 (§13.1 Gate-First / §13.3 残差旁路) ──

    /// 比较生成掩码: 逐元素比较 a 和 b。
    /// 后端自动选择: AVX-512 k-mask / SVE predicate / GPU predicate
    // @trace REQ-AIS-003 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/compare]
    Compare {
        a: ValueId,
        b: ValueId,
        op: CmpOp,
    },

    /// 掩码操作: 仅对 mask 为 true 的 lane 执行 op。
    MaskedOp {
        op: Box<TraceOp>,
        mask: ValueId,
    },

    // ── 原子操作 (§13.6 MoE 命中计数) ──

    /// 原子加: addr[0] += val。用于 MoE expert 命中计数等。
    /// 后端映射: lock xadd / ldadd / atomicAdd
    AtomicAdd {
        addr: ValueId,
        val: ValueId,
    },

    // ── 信号处理 (§11.1 TurboQuant FWHT) ──

    /// Fast Walsh-Hadamard Transform: 在线旋转变换。
    /// 就地变换 dim 个元素。O(d log d) 复杂度。
    /// 后端映射: 展开的 butterfly 加减指令序列
    FWHT {
        src: ValueId,
        dim: usize,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 结构型内存操作 (ARCH-AUTO-INSTR-SELECT structural)
    // Gather/Attention/MoE 等结构型算子的索引内存访问语义。
    // 不再绕过 auto_select 走手写 lower_*，全部纳入自动指令选择。
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// 标量内存加载: 从 base_ptr + byte_offset 读取一个值。
    /// 用于 Gather 中从 input_ids 读取 token index。
    /// 后端映射: vmovss + vmovd (x86) / ldr w-reg (ARM)
    ScalarLoad {
        base: ValueId,
        offset: ValueId,
    },

    /// 整数步幅乘法: result = value * stride（整数运算）。
    /// 用于地址计算: row_offset = index * row_bytes。
    /// 后端映射: imul (x86) / madd (ARM)
    StrideMul {
        value: ValueId,
        stride: usize,
    },

    /// 指针加法: result = base + offset（计算内存地址）。
    /// 用于 Gather 中 table_row = weight_ptr + row_offset。
    /// 后端映射: lea / add (x86) / add (ARM)
    PtrAdd {
        base: ValueId,
        offset: ValueId,
    },

    /// 向量化索引加载: 从 base_ptr + byte_offset 加载 SIMD 向量。
    /// 用于 Gather 中从 embedding table 加载一行数据。
    /// 后端映射: vmovups (x86) / ldr q-reg (ARM)
    VecLoadIndexed {
        base: ValueId,
        offset: ValueId,
    },

    /// 向量化索引存储: 将 SIMD 向量写入 base_ptr + byte_offset。
    /// 用于 Gather 中将 embedding 行写入 output buffer。
    /// 后端映射: vmovups (x86) / str q-reg (ARM)
    VecStoreIndexed {
        base: ValueId,
        offset: ValueId,
        value: ValueId,
    },

    // ── 向量广播 + 逐元素条件 (GateMask / EntropyGate) ──

    /// 标量广播到向量: 将 src 的 lane 0 值复制到所有 lane。
    /// 用于 GateMask（threshold 广播到所有 hidden dim）。
    /// 后端映射: x86 vbroadcastss / ARM dup
    BroadcastScalar { src: ValueId },

    /// 从内存加载标量并广播到向量: 从 base[offset] 加载 1 个元素,
    /// 复制到向量所有 lane。用于 GEMM A 矩阵行加载 (broadcast a scalar to all lanes)。
    /// 后端映射: x86 vbroadcastss (m32) / ARM ld1r
    BroadcastLoad { base: ValueId, offset: ValueId },

    // ── 向量索引内存操作 (Gather / Scatter) ──

    /// 向量索引加载: 从 base + indices[i]*stride 加载元素到向量。
    /// 后端映射: x86 vgatherdps / ARM scalar loop (ld1 per element)
    GatherLoad {
        base: ValueId,
        indices: ValueId,
        stride: usize,
    },

    /// 向量索引存储: 将 value 的元素按 indices 写入 base + indices[i]*stride。
    /// 后端映射: x86 vscatterdps (AVX-512) / ARM scalar loop (st1 per element)
    ScatterStore {
        base: ValueId,
        indices: ValueId,
        value: ValueId,
        stride: usize,
    },

    // ── 查表操作 (embedding lookup 通用表达) ──

    /// 行查表: 从 base_ptr + row_index * row_bytes 加载一行（SIMD 向量宽度）。
    /// 等价于 StrideMul(row_index, row_bytes) + PtrAdd(base, offset) + VecLoadIndexed(base, offset)
    /// 后端映射: 组合 IntMulStride + LoadPtr(VRegPlusVReg) + VecLoad
    TableLookup {
        base: ValueId,
        row_index: ValueId,
        row_bytes: usize,
    },

    // ── 量化解量化 (MoE expert FFN) ──

    /// MXFP4 解量化: packed 4-bit blocks → f32 values × per-block scale。
    /// 用于 MoE DispatchPacked 中 gate_up/down 权重解量化。
    /// 后端映射: VmInstr::QuantBlockLoad { unpack: Mxfp4 } (x86 AVX2 LUT + vcvtdq2ps + vmulps)
    ///
    /// 字节偏移公式: slots[off_a] * stride_a + slots[off_b] * stride_b + slots[off_c] + const_off
    /// 未使用的分量设 slot = u32::MAX（视为 0）
    Mxfp4Dequant {
        data: ValueId,
        scales: ValueId,
        off_a: Option<ValueId>,
        stride_a: usize,
        off_b: Option<ValueId>,
        stride_b: usize,
        off_c: Option<ValueId>,
        const_off: usize,
        block_size: usize,
    },

    /// 位与运算: result = a & b（逐位，非浮点算术）。
    /// 用于 INT4 量化 GEMM 中的低位掩码（如 0x0F0F0F0F）。
    /// 后端映射: VmInstr::VecBinOp { op: VecOp::And }
    BitAnd(ValueId, ValueId),

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SPEC 23-QUANT-CODEGEN-ALGO §3: Quant* 解码 TraceOp
    // 供 DecodeTraceBuilder 生成 GGUF 量化格式的硬件无关 SSA 解码序列。
    // 全部 14 个变体按字母序排列。
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// 位与: result = lhs & rhs (整数逐位，用于 nibble 掩码)。
    /// 后端映射: VmInstr::VecBinOp { op: VecOp::And, dtype: I32 } (REQ-VR-004 §1.4.1)
    QuantBitAnd { lhs: ValueId, rhs: ValueId },

    /// 位或: result = lhs | rhs (整数逐位，用于合并 high-bit 平面)。
    /// 后端映射: VmInstr::VecBinOp { op: VecOp::Or, dtype: I32 } (REQ-VR-004 §1.4.1)
    QuantBitOr { lhs: ValueId, rhs: ValueId },

    /// 标量广播: 将 src 的 lane 0 值广播到 lanes 个 lane。
    /// 用于将单个 f32 scale 广播到整个解码向量。
    /// 后端映射: VmInstr::Broadcast { dtype: F32 } (REQ-VR-004 §1.4.1)
    QuantBroadcast { src: ValueId, lanes: usize },

    /// f16 → f32 类型转换: result = (f32) src。
    /// 用于将 GGUF block 头的 f16 scale 转换为 f32。
    /// 后端映射: VmInstr::VecCast { from_bits: 16, to_bits: 32 } (REQ-VR-004 §1.4.1)
    QuantCastF16toF32 { src: ValueId },

    /// i8 → f32 类型转换: result = (f32) src (K-quant sub-block scale)。
    /// 后端映射: VmInstr::VecUnaryOp { op: VecUnaryOp::IntToFloat } (REQ-VR-004 §1.4.1)
    QuantCastI8toF32 { src: ValueId },

    /// FP8 → F32 float-to-float conversion: result = fp8_to_f32(src).
    /// Unlike QuantCastI8toF32 (integer→float), this performs proper FP8 IEEE decode.
    /// `format`: E4M3 (bias=7, 3 mantissa bits) or E5M2 (bias=15, 2 mantissa bits).
    /// 后端映射: VmInstr::VecUnaryOp { op: Fp8E4M3ToFloat / Fp8E5M2ToFloat }
    QuantCastFp8toF32 { src: ValueId, format: Fp8Format },

    /// Codebook 查表: indices → f32 values via static codebook。
    /// 用于 IQ 系列 (IQ4_NL 等) 的 codebook 反量化。
    /// codebook_data 在 JIT 机器码中作为 .rodata 嵌入。
    /// 后端映射: VmInstr::QuantCodebookLookup
    QuantCodebookLookup {
        indices: ValueId,
        codebook_data: &'static [i8],
        vector_size: usize,
        bits_per_entry: u8,
    },

    /// 位域提取: result = (src >> bit_offset) & ((1 << bit_width) - 1)。
    /// 用于从 packed 字节流中提取 index_bits 宽的 codebook 索引。
    /// 后端映射: VmInstr::QuantExtractBits
    QuantExtractBits { src: ValueId, bit_offset: u32, bit_width: u8 },

    /// 解码 FMA: acc += a * b (全 f32)。
    /// 用于最终反量化的 unpacked * scale + zero 计算。
    /// 与已有 QuantFma (混合精度 TurboQuant) 不同，这里全部是 f32 dequant 代数。
    /// 后端映射: VmInstr::Fma (REQ-VR-004 §1.4.1)
    QuantDequantFma { acc: ValueId, a: ValueId, b: ValueId },

    /// 整数除以编译时常量: result = src / divisor (截断)。
    /// 用于计算 sub_block_idx = lane_offset / sub_block_elements。
    /// 后端映射: VmInstr::GprBinOp { op: GprOp::Div, b: GprOperand::Imm(divisor) }
    QuantIntDivConst { src: ValueId, divisor: i64 },

    /// 整数乘以编译时常量: result = src * factor。
    /// 用于计算字节偏移 (sub_block_idx * scale_entry_bytes)。
    /// 后端映射: VmInstr::GprBinOp { op: GprOp::Mul, b: GprOperand::Imm(factor) }
    QuantIntMul { src: ValueId, factor: i64 },

    /// 交叉合并两个半宽向量: result = interleave(lo, hi)。
    /// 用于 PackedNibbles: 将 lo nibbles 和 hi nibbles 合并为 lanes 个元素。
    /// 后端映射: VmInstr::QuantInterleave
    QuantInterleave { lo: ValueId, hi: ValueId },

    /// 顺序拼接两个半宽向量: result = [lo[0..N/2], hi[0..N/2]]。
    /// 用于 PackedNibbles 的 QuantGather 路径: 元素顺序必须为 [0,1,...,15,16,17,...,31]
    /// 而非交错 [0,16,1,17,...] (QuantInterleave)。
    /// 后端映射: VmInstr::QuantConcatSeq
    QuantConcatSeq { lo: ValueId, hi: ValueId },

    /// 指针算术 (不读内存): result = base_ptr + offset_bytes。
    /// 用于计算 block 内子数组的起始地址。
    /// 后端映射: VmInstr::AddPtr
    QuantPtrAddOffset { base: ValueId, offset_bytes: i64 },

    /// 指针算术 (不读内存): result = base_ptr + index_slot。
    /// 用于计算动态偏移地址 (base + sub_block_idx)。
    /// 后端映射: VmInstr::GprBinOp { op: GprOp::Add }
    QuantPtrAddDynamic { base: ValueId, index: ValueId },

    /// 整数按位 AND 常量掩码: result = lhs_i32 & mask。
    /// 在整数域操作 (vpand with integer-broadcasted mask), 不经过 f32 broadcast。
    QuantAndMask { src: ValueId, mask: u64 },

    /// K-Quant (Q3_K/Q4_K/Q5_K) packed 6-bit scale/min lookup for one sub-block.
    /// Decodes the packed scale from `scales[12]` array indexed by sub_block_idx.
    /// scale_algo=KQuant6Bit, selector=Scale:
    ///   if j<4: sc = scales[j] & 0x3F
    ///   if j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    /// scale_algo=KQuant6Bit, selector=Min:
    ///   if j<4: m = scales[j+4] & 0x3F
    ///   if j>=4: m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    /// Result: (f32)sc/m broadcast to all SIMD lanes.
    /// 后端映射: VmInstr::GgufKQuantScaleLoad
    QuantKQuantPackedScaleLookup {
        scales_base: ValueId,
        sub_block_idx: ValueId,
        scale_algo: PackedScaleAlgorithm,
        selector: ScaleSelector,
    },

    /// 标量内存加载 (带字节偏移): result = *(f16_or_i8*)(base_ptr + offset_bytes)。
    /// 用于读取 block 内固定偏移处的 scale/min/zero 字段。
    /// 后端映射: VmInstr::ScalarLoad + VmInstr::Broadcast (REQ-VR-004 §1.4.1)
    QuantScalarLoad { ptr: ValueId, offset_bytes: i64 },

    /// f16 标量内存加载 + 转 f32: result = (f32)*(f16*)(base_ptr + offset_bytes)。
    /// 单指令完成加载和类型转换，避免 QuantScalarLoad + QuantCastF16toF32 的双重加载。
    /// 后端映射: VmInstr::QuantScalarCvtLoad { src_dtype: ScalarCvtSource::F16 } (REQ-VR-004 §1.4.2)
    QuantLoadF16toF32 { ptr: ValueId, offset_bytes: i64 },

    /// i8 标量内存加载 + 转 f32: result = (f32)*(i8*)(base_ptr + offset_bytes)。
    /// 单指令完成加载和类型转换，避免 QuantScalarLoad + QuantCastI8toF32 的双重加载。
    /// 后端映射: VmInstr::QuantScalarCvtLoad { src_dtype: ScalarCvtSource::I8 } (REQ-VR-004 §1.4.2)
    QuantLoadI8toF32 { ptr: ValueId, offset_bytes: i64 },

    /// 多字节内存加载为 i32 向量 (零扩展，不转 float):
    /// 从 `base_ptr + offset_bytes` 加载 `count` 字节，零扩展为 `count` 个 i32。
    /// lanes 0..count 为实际值，count..lanes 为 0。
    /// 后端映射: VmInstr::QuantLoadBytesVec
    /// 加载多个字节到向量寄存器 (每个字节零扩展/符号扩展为 i32)。
    /// signed=true 时使用 vpmovsxbd (符号扩展); signed=false 时使用 vpmovzxbd (零扩展)。
    QuantLoadBytesVec { ptr: ValueId, offset_bytes: i64, count: usize, signed: bool },

    /// 整数向量左移: result = src << amount (逐元素)。
    /// 用于将 high-bit 平面左移后与低 nibbles 合并 (NibbleWithHighBits)。
    /// 后端映射: VmInstr::VecShiftImm { op: VecShiftDir::Left } (REQ-VR-004 §1.4.1)
    QuantShiftLeft { src: ValueId, amount: u32 },

    /// 整数向量右移: result = src >> amount (逐元素，逻辑右移)。
    /// 用于从 packed byte 中提取高 nibble。
    /// 后端映射: VmInstr::VecShiftImm { op: VecShiftDir::Right } (REQ-VR-004 §1.4.1)
    QuantShiftRight { src: ValueId, amount: u32 },

    /// E2M1 LUT decode: loads packed nibbles from data_ptr, decodes via E2M1 lookup table,
    /// multiplies by scale, outputs F32 SIMD vector.
    /// Used for MXFP4 (E8M0 scale) and NVFP4 (UE4M3 scale) — hardware-specialized decode
    /// not expressible as generic integer × float algebra.
    /// `packed_data_ptr`: slot of Ptr VReg pointing to packed nibble data
    /// `scale_byte`: slot of Ptr VReg holding raw scale byte (E8M0 or UE4M3)
    /// `nvfp4_mode`: true = UE4M3 scale (NVFP4), false = E8M0 scale (MXFP4)
    /// 后端映射: VmInstr::QuantBlockLoad { unpack: Mxfp4 } (E8M0) / VmInstr::QuantBlockLoad { unpack: Nvfp4 } (UE4M3)
    QuantE2m1LutDecode {
        packed_data_ptr: ValueId,
        scale_byte: ValueId,
        nvfp4_mode: bool,
    },

    /// Q3_K combined decode step: extracts 2-bit values from qs[] at variable shift,
    /// applies conditional bias from hmask[], multiplies by scale (dl = d * (scale - 32)).
    ///
    /// This is a monolithic op because Q3_K's non-linear data access pattern
    /// (qs bytes read with variable shifts, hmask bit positions accumulate across segments)
    /// cannot be decomposed into the standard load→unpack→dequant pipeline.
    ///
    /// Inputs:
    /// - `block_base`: pointer to the start of the Q3_K block
    /// - `lane_offset`: iteration counter within the block (0..31 for lanes=8)
    /// - `d_slot`: f32 scalar holding the block's super-block scale d (loaded from f16)
    ///
    /// Output: f32 vector of `lanes` decoded values.
    ///
    /// Internal logic (per element i in 0..lanes):
    ///   global_elem = lane_offset * lanes + i
    ///   seg = global_elem / 128
    ///   group_in_seg = (global_elem % 128) / 16
    ///   j = group_in_seg % 4  (shift index)
    ///   run = group_in_seg / 4  (run index, 0 or 1)
    ///   l = global_elem % 16
    ///   qs_val = (qs[seg*32 + run*16 + l] >> (j*2)) & 3
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SPEC 24-QUANT-PIPELINE-JIT §1.3: QuantGather/QuantGemm trace pipeline
    // 量化块级加载操作，驱动 auto_select 生成正确的 VmInstr 序列。
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// 加载量化 block 的 scale（f16 → broadcast f32）。
    /// 从 source + offset 读取 f16 scale 并广播为 f32 向量。
    /// 后端映射: x86 GgufF16ScaleLoad / ARM scalar f16→f32 / GPU ld.global.b16
    QuantScaleLoad {
        source: ValueId,
        offset: usize,
        dtype: QuantType,
    },

    /// 加载量化 block 的 packed data 并解包为 F32 向量。
    /// 根据 quant_type 和 block_size 查找 QuantFormatDescriptor 决定解包方式。
    /// 后端映射: x86 GgufInt4Load/GgufInt4HighLoad/GgufInt8Load /
    ///           ARM scalar byte unpack / GPU ld.global + bit manipulation
    QuantDataLoad {
        source: ValueId,
        offset: usize,
        quant_type: QuantType,
        block_size: usize,
    },

    /// 加载量化 block 的 zero-point / min / bias。
    /// zp_type 决定格式：StaticBias(Q4_0) / BlockScalar(Q4_1) / None(Q8_0)。
    /// 后端映射: f16→f32 load + broadcast (BlockScalar) / immediate sub (StaticBias)
    QuantZeroLoad {
        source: ValueId,
        offset: usize,
        zp_type: ZeroLayout,
    },

    /// 加载子块 scale（K-Quant 层级化 scale）。
    /// 用于 Q4_K/Q5_K 等格式的 sub-block 缩放。
    /// 后端映射: packed 6-bit unpack + f16 d × f32 sub_scale
    QuantSubScaleLoad {
        block_ptr: ValueId,
        byte_offset: usize,
        bits: usize,
        sub_block_size: usize,
    },

    /// 加载 high bits（INT5/INT6 的额外位平面）。
    /// Q5_0/Q5_1/Q6_K 用 1-2 个额外位扩展 4-bit 到 5/6-bit。
    /// 后端映射: byte load + bit interleave
    QuantHighBitsLoad {
        block_ptr: ValueId,
        byte_offset: usize,
        bits_per_elem: usize,
    },

    /// Codebook 查找反量化（SqueezeLLM / IQ 系列）。
    /// 与 QuantCodebookLookup 不同，本变体接受动态 codebook_ptr（运行时），
    /// 而非编译时静态 codebook_data。
    QuantCodebookDequant {
        indices: ValueId,
        codebook_ptr: ValueId,
        vector_size: usize,
        bits_per_entry: usize,
    },

    /// Q3_K combined decode: super-block d (f16→f32) + qs packed bits + hmask
    /// → outputs 256 f32 values in-place via QuantGather/GEMM lane decode.
    QuantQ3KDecode {
        block_base: ValueId,
        lane_offset: ValueId,
        d_slot: ValueId,
        qs_offset: usize,
        hmask_offset: usize,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SPEC 24-QUANT-PIPELINE-JIT §1.3: QuantGather/QuantGemm structural trace ops
    // Marker ops in the trace — auto_select expands to full emit_*_inline logic.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Quantized embedding lookup: decode quant blocks on-the-fly for indexed rows.
    /// The 3 preceding Input ops provide: indices_ptr, embed_ptr, output_ptr.
    /// auto_select mapping: emit_quant_gather_inline → VmInstr (seq loop → block decode → store)
    QuantGather {
        quant_type: QuantType,
        vocab_size: usize,
        hidden_dim: usize,
    },

    /// Quantized GEMM with on-the-fly dequantization.
    /// The 3 preceding Input ops provide: input_ptr, weight_ptr, output_ptr.
    /// auto_select mapping: emit_quant_gemm_inline → VmInstr (tiled M×N×K loop → decode → FMA)
    QuantGemm {
        quant_type: QuantType,
        m: usize,
        n: usize,
        k: usize,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SPEC 27 REQ-AT-002: TraceOp structural extensions
    // Loop / panel / softmax / GPU / tile — produced by template interpreter
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Structured loop: iterate `bound` times with `step_bytes` stride.
    /// auto_select maps: LoopBegin + body + LoopEnd VmInstr sequence.
    Loop {
        bound: crate::compiler::codegen::vm::instr::BoundExpr,
        step_bytes: usize,
        body: Vec<TraceOp>,
    },

    /// Matrix panel load: load rows×cols elements from base + offset into
    /// contiguous registers.
    /// auto_select maps: multiple VecLoad / VecLoadIndexed VmInstr.
    PanelLoad {
        base: ValueId,
        offset: ValueId,
        rows: usize,
        cols: usize,
    },

    /// Matrix panel store: write contiguous registers to base + offset.
    /// auto_select maps: multiple VecStore VmInstr.
    PanelStore {
        base: ValueId,
        offset: ValueId,
        rows: usize,
        cols: usize,
    },

    /// Buffer pack: rearrange B panel to contiguous scratch buffer.
    /// auto_select maps: VecLoad + VecStore sequence.
    PackBuffer {
        src: ValueId,
        dst: ValueId,
        rows: usize,
        cols: usize,
        layout: crate::compiler::codegen::vm::algo_template::PackLayout,
    },

    /// Shared memory declaration (GPU).
    SharedMemDeclare {
        name: String,
        bytes: usize,
    },

    /// Async copy to shared memory (GPU).
    AsyncCopyToShared {
        name: String,
        src_offset: ValueId,
        bytes: usize,
    },

    /// TMA 2D tensor 异步拷贝 (SM90+ cp.async.bulk.tensor.2d)。
    /// 需要 TMA 描述符 (Host 端 cuTensorMapEncodeTiled 创建)。
    Tma2DCopy {
        /// 描述符名称
        desc: String,
        /// 行坐标 ValueId
        coord_x: ValueId,
        /// 列坐标 ValueId
        coord_y: ValueId,
        /// 搬运字节大小
        bytes: usize,
    },

    /// Wait for async operation group (GPU).
    AsyncWaitGroup { n: u32 },

    /// Synchronization barrier (GPU).
    SyncBarrier { name: String },

    /// Configure hardware tile register (AMX/SME2).
    TileConfig { rows: usize, cols: usize },

    /// Tile matrix multiply-accumulate: c += a × b (AMX/SME2/GPU).
    /// shape: a=m×k, b=k×n, c=m×n (CR-TIER-SOVEREIGNTY-004, 透传给 VmInstr::TileMma)。
    /// dtype 由 auto_select 从 graph tensor 推断注入 VmInstr (同 TileConfig 模式)。
    TileMma { c: ValueId, a: ValueId, b: ValueId, m: usize, n: usize, k: usize },

    /// Release hardware tile resources.
    TileRelease,

    /// Softmax: reduce_max → exp(x-max) → sum → normalize.
    /// auto_select expands: HReduce(Max) → Sub → Exp → HReduce(Sum) → Div.
    Softmax { src: ValueId, dst: ValueId },

    /// Epilogue chain: apply a sequence of post-GEMM operations.
    EpilogueChain { ops: Vec<crate::compiler::codegen::vm::algo_template::EpilogueOp> },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // MTP Draft (MTP-001): Multi-Token Prediction structural trace op.
    // auto_select expands to depth × (GEMV + argmax + store) loop nest.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Multi-Token Prediction draft candidate generation.
    /// Expands to: depth iterations of hidden→vocab GEMV + argmax + store.
    /// Preceding Input ops: [hidden_ptr, weight_ptr, output_tokens_ptr].
    /// auto_select mapping: emit_mtp_draft_inline → VmInstr loop nest.
    MtpDraft {
        depth: usize,
        hidden_size: usize,
        vocab_size: usize,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // MLA (Multi-head Latent Attention) — DeepSeek V3/R1, Kimi-K2
    // Structural TraceOp for MlaAttention and MlaRopeMerge.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// MLA Attention score computation in compressed d_c space.
    /// Per-head: Q_absorbed[h] × key^T → softmax → weighted V restore.
    /// auto_select mapping: emit_mla_attn_score_inline → VmInstr loop nest.
    MlaAttnScore {
        num_heads: usize,
        head_dim: usize,
        d_c: usize,
        d_rope: usize,
    },

    /// MLA decoupled RoPE merge: replace c_KV[d_c-d_rope..d_c] with RoPE(k_pe).
    /// Injective: concat(c_KV[:, :d_c-d_rope], RoPE(k_pe)) → merged key.
    /// auto_select mapping: emit_mla_rope_merge_inline → VecLoad/VecStore/VecFma.
    MlaRopeMerge {
        d_c: usize,
        d_rope: usize,
    },

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SPEC 37 REQ-HWACC-007: Dynamic Precision Hot-Switch
    // GEMM prologue 分析 weight/activation 统计量 → 运行时选择 FP16/FP8/NVFP4 kernel
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// 动态精度选择：分析 tensor 统计量选择最优计算精度。
    /// GEMM prologue 使用：测量 weight/activation 的 max 值 → 选择 FP16/FP8/NVFP4。
    ///
    /// 语义:
    ///   1. HReduce(tensor, Max) → max_abs_value
    ///   2. 对每个候选精度（从高到低），检查 max_abs_value < threshold[i]
    ///   3. 选择第一个满足条件的精度；全部不满足则使用 candidates[0]（最高精度）
    ///   4. 输出 = 精度索引（i32 标量），后续 GEMM 根据此索引选择对应 kernel 变体
    ///
    /// 结果 dtype = candidates[0]（最高精度，作为默认计算精度）。
    /// has_side_effects = false（纯计算，不修改内存）。
    DynamicPrecisionSelect {
        /// 要分析的 tensor（weight 或 activation）的 ValueId。
        /// auto_select 会对该 tensor 发射 HReduce(Max) 提取最大绝对值。
        tensor: ValueId,
        /// 候选精度列表（从高到低排序，如 [FP16, FP8E4M3, FP4E2M1]）。
        /// candidates[0] 是最高精度（默认回退），candidates[last] 是最低精度。
        /// len 必须与 thresholds.len() 相同。
        candidates: Vec<QuantPrecision>,
        /// 每个精度对应的阈值（max_abs_value < threshold → 选用该精度）。
        /// thresholds[i] 对应 candidates[i]：当 max_abs < thresholds[i] 时可安全使用该精度。
        /// thresholds 必须严格单调递减（高精度阈值大，低精度阈值小）。
        thresholds: Vec<f64>,
    },
}

/// 4-bit 加载 pass（GGUF split layout）。
/// Q4_0/Q4_1 的 32 个 4-bit 值打包在 16 字节中，需要分两次 pass 加载。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantLoadPass {
    /// 低 nibble: (byte & 0x0F) - 8.0（有符号 Q4_0）或 (byte & 0x0F)（无符号 Q4_1）
    LowNibble,
    /// 高 nibble: ((byte >> 4) & 0x0F) - 8.0（有符号 Q4_0）或 ((byte >> 4) & 0x0F)（无符号 Q4_1）
    HighNibble,
    /// 单 pass（8-bit / 5-bit / 6-bit 等非 split 格式）
    Single,
}

/// 量化精度类型 (ARCH-JIT-GENERATOR §12)
///
/// 覆盖三代硬件的所有精度格式:
/// - NVIDIA: FP8(E4M3/E5M2), FP4, FP6 (Blackwell+)
/// - AMD: FP8(E4M3/E5M2), FP6(E2M3/E3M2), FP4(E2M1) (CDNA4+)
/// - Intel: TF32, FP8 (Diamond Rapids AMX-FP8)

/// 基础精度类型 — 量化无关的元素精度 (REQ-DTYPE-001)。
/// 新增精度只需添加变体，不影响 QuantPrecision 结构体布局。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    F32,
    F16,
    BF16,
    TF32,                // Intel AMX-TF32
    FP8E4M3,             // NVIDIA/AMD standard FP8
    FP8E5M2,             // NVIDIA/AMD alternate FP8
    FP6E2M3,             // AMD CDNA4
    FP6E3M2,             // AMD CDNA4
    FP4E2M1,             // AMD CDNA4 / NVIDIA Blackwell
    INT8,
    INT4,
    INT2,
    INT1,                // QJL 1-bit (§11.5 双轨池)
}

/// 存储打包方式 — 量化格式的打包策略 (REQ-DTYPE-001)。
/// 决定了权重在内存中的排列和解量化方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackingFormat {
    /// 标准: 1 elem = N bytes，无打包
    Plain,
    /// MX 格式: block_size elems 共享 scale (Microscaling)
    MXBlock,
    /// GGML 量化类型 (GGUF 格式)
    GGML(GgmlType),
    /// GPTQ 分组量化: group_size elems 共享 scale/zero_point
    GPTQ,
    /// AWQ 激活感知量化
    AWQ,
    /// BitNet 1.58-bit ternary (-1, 0, +1)
    Bitnet,
}

/// GGML 量化子类型 — GGUF 格式支持的全部量化变体。
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    Q2_K, Q3_K_S, Q3_K_M, Q3_K_L,
    Q4_K_S, Q4_K_M,
    Q5_K_S, Q5_K_M,
    Q6_K,
    IQ2_XXS, IQ2_XS, IQ3_XXS,
}

impl GgmlType {
    /// 该 GGML 类型对应的基础精度。
    pub fn dtype_kind(&self) -> DTypeKind {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q4_K_S | Self::Q4_K_M => DTypeKind::INT4,
            Self::Q5_0 | Self::Q5_1 | Self::Q5_K_S | Self::Q5_K_M => DTypeKind::INT4,
            Self::Q6_K => DTypeKind::INT4,
            Self::Q8_0 => DTypeKind::INT8,
            Self::Q2_K | Self::IQ2_XXS | Self::IQ2_XS => DTypeKind::INT2,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L | Self::IQ3_XXS => DTypeKind::INT2,
        }
    }

    /// 该 GGML 类型的量化块大小 (elements per block)。
    pub fn block_size(&self) -> u32 {
        match self {
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q2_K => 256,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => 256,
            Self::Q4_K_S | Self::Q4_K_M => 256,
            Self::Q5_K_S | Self::Q5_K_M => 256,
            Self::Q6_K => 256,
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ3_XXS => 256,
        }
    }
}

/// 参数化精度类型 — 覆盖所有已知和未来量化格式 (REQ-DTYPE-001)。
///
/// 设计原则:
/// - kind: 元素精度 (F32/BF16/INT8/...)
/// - packing: 存储打包方式 (Plain/MXBlock/GGML/GPTQ/AWQ/Bitnet)
/// - block_size: 量化块大小 (0 = 不适用)
/// - group_size: 分组量化的组大小 (0 = 不适用)
///
/// 便利常量 (QuantPrecision::F32 等) 保持向后兼容。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantPrecision {
    pub kind: DTypeKind,
    pub packing: PackingFormat,
    pub block_size: u32,
    pub group_size: u32,
}

/// AArch64 元素计算策略 (ARCH-DTYPE-JIT-TYPED)。
///
/// 驱动 AArch64 ISA lowering 层的指令选择，禁止 dtype 身份匹配。
/// - `Native`: NEON/SVE 原生支持 (F32/BF16/F16/INT8)。
/// - `WidenCompute`: 需要 widen 到 F32 计算 (sub-byte FP)。
/// - `DequantCompute`: 需要反量化到 F32 (GGML/GPTQ/AWQ/Bitnet/MXBlock)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AArch64ElemStrategy {
    /// NEON/SVE 原生支持该 dtype。
    Native,
    /// 需要 widen 到 F32 后计算。
    WidenCompute,
    /// 需要反量化后 F32 计算。
    DequantCompute,
}

/// GPU 元素计算策略 (ARCH-DTYPE-JIT-TYPED)。
///
/// 驱动 GPU ISA lowering 层的指令选择 (PTX/HIP/Metal)。
/// - `Native`: GPU 原生支持 (所有标准 FP/INT 类型)。
/// - `WidenCompute`: 需要 widen (sub-byte FP 类型)。
/// - `DequantCompute`: 需要反量化 (GGML/GPTQ/AWQ/Bitnet/MXBlock)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuElemStrategy {
    /// GPU 原生支持该 dtype。
    Native,
    /// 需要 widen 到更宽类型后计算。
    WidenCompute,
    /// 需要反量化后 F32 计算。
    DequantCompute,
}

/// x86 元素计算策略 (ARCH-DTYPE-JIT-TYPED)。
///
/// 驱动 ISA lowering 层的指令选择，禁止 dtype 身份匹配。
/// - `Native`: 当前硬件对该 dtype 有原生 SIMD 指令（如 F32 → vaddps）。
/// - `WidenCompute`: 需 widen 到 F32 计算（如 BF16 → vaddps on widened data）。
/// - `DequantCompute`: 需反量化到 F32 计算（如 INT8 → VNNI dp4a → F32）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X86ElemStrategy {
    /// 硬件原生支持该 dtype 的 SIMD 指令。
    Native,
    /// 需要 widen 到 F32 后计算。
    WidenCompute,
    /// 需要反量化后 F32 计算，附带反量化方法。
    DequantCompute(DequantMethod),
}

/// 反量化方法 — 量化类型到 F32 的硬件加速路径。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantMethod {
    /// AVX-512 VNNI: dp4a for INT8
    VNNI,
    /// AMX tile: TDPBSSD for INT8, TDPBF16PS for BF16
    AMX,
    /// 通用标量 LUT (GGML 量化类型)
    ScalarLUT,
    /// MX 格式: block shared scale × elem
    BlockScale,
}

/// 类型化寄存器 slot — 携带自动推断的 dtype (SPEC 00-PHILOSOPHY §4.1)。
#[derive(Debug, Clone, Copy)]
pub struct TypedSlot {
    pub vreg: crate::compiler::codegen::vm::instr::VRegId,
    pub dtype: QuantPrecision,
}

impl QuantPrecision {
    // ── 便利常量 (向后兼容) ──

    pub const F32: Self = Self { kind: DTypeKind::F32, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const F16: Self = Self { kind: DTypeKind::F16, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const BF16: Self = Self { kind: DTypeKind::BF16, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const TF32: Self = Self { kind: DTypeKind::TF32, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const FP8E4M3: Self = Self { kind: DTypeKind::FP8E4M3, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const FP8E5M2: Self = Self { kind: DTypeKind::FP8E5M2, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const FP6E2M3: Self = Self { kind: DTypeKind::FP6E2M3, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const FP6E3M2: Self = Self { kind: DTypeKind::FP6E3M2, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const FP4E2M1: Self = Self { kind: DTypeKind::FP4E2M1, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const INT8: Self = Self { kind: DTypeKind::INT8, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const INT4: Self = Self { kind: DTypeKind::INT4, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const INT2: Self = Self { kind: DTypeKind::INT2, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
    pub const INT1: Self = Self { kind: DTypeKind::INT1, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };

    /// MXFP4: FP4E2M1 + MXBlock packing, block_size=32
    pub const MXFP4_32: Self = Self { kind: DTypeKind::FP4E2M1, packing: PackingFormat::MXBlock, block_size: 32, group_size: 0 };

    /// 从 GgmlType 构建 QuantPrecision
    pub const fn ggml(ty: GgmlType) -> Self {
        Self { kind: DTypeKind::INT4, packing: PackingFormat::GGML(ty), block_size: 0, group_size: 0 }
    }

    /// 是否为打包量化格式 (非 Plain)
    pub fn is_packed(&self) -> bool {
        !matches!(self.packing, PackingFormat::Plain)
    }

    /// 元素字节数。sub-byte 类型按打包后平均每元素字节返回。
    pub const fn elem_bytes(self) -> usize {
        match self.kind {
            DTypeKind::F32 | DTypeKind::TF32 => 4,
            DTypeKind::BF16 | DTypeKind::F16 => 2,
            DTypeKind::FP8E4M3 | DTypeKind::FP8E5M2 | DTypeKind::INT8 => 1,
            DTypeKind::FP6E2M3 | DTypeKind::FP6E3M2 | DTypeKind::FP4E2M1
            | DTypeKind::INT4 | DTypeKind::INT2 | DTypeKind::INT1 => 0,
        }
    }

    /// 从 elem_bytes 反推 QuantPrecision (dtype 传播链辅助)。
    /// 仅支持标准字节对齐类型 (4/2/1)，sub-byte 类型无反向映射。
    pub const fn from_elem_bytes(bytes: usize) -> Self {
        match bytes {
            4 => Self::F32,
            2 => Self::BF16,
            1 => Self::FP8E4M3,
            _ => Self::F32,
        }
    }

    /// GEMM 累加器 dtype (REQ-DTYPE-005)。
    /// 由 (kind × packing × 硬件能力) 联合决定。
    /// x86 路径: BF16→F32 累加, F32→F32 累加
    pub fn accumulator_dtype(&self) -> QuantPrecision {
        match self.x86_elem_strategy() {
            X86ElemStrategy::Native => *self,
            X86ElemStrategy::WidenCompute | X86ElemStrategy::DequantCompute(_) => QuantPrecision::F32,
        }
    }

    /// GPU GEMM 累加器 dtype (REQ-DTYPE-005)。
    /// 由 (kind × packing × GPU 能力) 联合决定。
    /// GPU: BF16→F32 tensor core 累加, F16→F16 累加, 量化→F32
    pub fn gpu_accumulator_dtype(&self) -> QuantPrecision {
        match self.gpu_elem_strategy() {
            GpuElemStrategy::Native => match self.kind {
                // BF16 uses F32 tensor core accumulation (HMMA.16816.F32)
                DTypeKind::BF16 | DTypeKind::TF32 => QuantPrecision::F32,
                // F16 can use F16 accumulation (HMMA.16816.F16) or F32
                DTypeKind::F16 => QuantPrecision::F32,
                // FP8 uses F32 accumulation
                DTypeKind::FP8E4M3 | DTypeKind::FP8E5M2 | DTypeKind::FP4E2M1 => QuantPrecision::F32,
                // INT8 uses F32 accumulation
                DTypeKind::INT8 => QuantPrecision::F32,
                // Other native types keep their dtype
                _ => *self,
            },
            GpuElemStrategy::WidenCompute | GpuElemStrategy::DequantCompute => QuantPrecision::F32,
        }
    }

    /// 是否需要从累加器 dtype 窄化到本 dtype (REQ-DTYPE-006)。
    pub fn needs_narrowing_from(&self, acc: QuantPrecision) -> bool {
        acc != *self && self.elem_bytes() < acc.elem_bytes()
    }

    /// x86 元素计算策略 (ARCH-DTYPE-JIT-TYPED)。
    ///
    /// 属性驱动指令选择，禁止 dtype 身份匹配。
    pub fn x86_elem_strategy(&self) -> X86ElemStrategy {
        match self.packing {
            PackingFormat::Plain => match self.kind {
                DTypeKind::F32 | DTypeKind::TF32 => X86ElemStrategy::Native,
                DTypeKind::BF16 | DTypeKind::F16 => X86ElemStrategy::WidenCompute,
                DTypeKind::INT8 => X86ElemStrategy::DequantCompute(DequantMethod::VNNI),
                DTypeKind::INT4 | DTypeKind::INT2 | DTypeKind::INT1
                | DTypeKind::FP8E4M3 | DTypeKind::FP8E5M2
                | DTypeKind::FP6E2M3 | DTypeKind::FP6E3M2 | DTypeKind::FP4E2M1
                => X86ElemStrategy::DequantCompute(DequantMethod::ScalarLUT),
            },
            PackingFormat::MXBlock => X86ElemStrategy::DequantCompute(DequantMethod::BlockScale),
            PackingFormat::GGML(_) => X86ElemStrategy::DequantCompute(DequantMethod::ScalarLUT),
            PackingFormat::GPTQ | PackingFormat::AWQ => X86ElemStrategy::DequantCompute(DequantMethod::ScalarLUT),
            PackingFormat::Bitnet => X86ElemStrategy::DequantCompute(DequantMethod::ScalarLUT),
        }
    }

    /// AArch64 元素计算策略 (ARCH-DTYPE-JIT-TYPED)。
    ///
    /// 驱动 AArch64 ISA lowering 层的指令选择。
    /// - `Native`: NEON/SVE 原生支持 (F32→fadd, BF16→BFDOT, F16→FMLA, INT8→SDOT)。
    /// - `WidenCompute`: 需要 widen 到 F32 计算 (sub-byte FP)。
    /// - `DequantCompute`: 需要反量化到 F32 (GGML/GPTQ/AWQ/Bitnet/MXBlock)。
    pub fn aarch64_elem_strategy(&self) -> AArch64ElemStrategy {
        match self.packing {
            PackingFormat::Plain => match self.kind {
                DTypeKind::F32 | DTypeKind::TF32 => AArch64ElemStrategy::Native,
                DTypeKind::BF16 => AArch64ElemStrategy::Native,
                DTypeKind::F16 => AArch64ElemStrategy::Native,
                DTypeKind::INT8 => AArch64ElemStrategy::Native,
                DTypeKind::INT4 | DTypeKind::INT2 | DTypeKind::INT1
                | DTypeKind::FP8E4M3 | DTypeKind::FP8E5M2
                | DTypeKind::FP6E2M3 | DTypeKind::FP6E3M2 | DTypeKind::FP4E2M1
                => AArch64ElemStrategy::WidenCompute,
            },
            PackingFormat::MXBlock => AArch64ElemStrategy::DequantCompute,
            PackingFormat::GGML(_) => AArch64ElemStrategy::DequantCompute,
            PackingFormat::GPTQ | PackingFormat::AWQ => AArch64ElemStrategy::DequantCompute,
            PackingFormat::Bitnet => AArch64ElemStrategy::DequantCompute,
        }
    }

    /// GPU 元素计算策略 (ARCH-DTYPE-JIT-TYPED)。
    ///
    /// 驱动 GPU ISA lowering 层的指令选择 (PTX/HIP/Metal)。
    /// - `Native`: GPU 原生支持 (所有标准 FP/INT 类型)。
    /// - `WidenCompute`: 需要 widen (sub-byte FP 类型)。
    /// - `DequantCompute`: 需要反量化 (GGML/GPTQ/AWQ/Bitnet/MXBlock)。
    pub fn gpu_elem_strategy(&self) -> GpuElemStrategy {
        match self.packing {
            PackingFormat::Plain => match self.kind {
                DTypeKind::F32 | DTypeKind::TF32 => GpuElemStrategy::Native,
                DTypeKind::BF16 | DTypeKind::F16 => GpuElemStrategy::Native,
                DTypeKind::INT8 => GpuElemStrategy::Native,
                DTypeKind::FP8E4M3 | DTypeKind::FP8E5M2 => GpuElemStrategy::Native,
                DTypeKind::FP4E2M1 => GpuElemStrategy::Native,
                DTypeKind::INT4 | DTypeKind::INT2 | DTypeKind::INT1
                | DTypeKind::FP6E2M3 | DTypeKind::FP6E3M2
                => GpuElemStrategy::WidenCompute,
            },
            PackingFormat::MXBlock => GpuElemStrategy::DequantCompute,
            PackingFormat::GGML(_) => GpuElemStrategy::DequantCompute,
            PackingFormat::GPTQ | PackingFormat::AWQ => GpuElemStrategy::DequantCompute,
            PackingFormat::Bitnet => GpuElemStrategy::DequantCompute,
        }
    }

    /// GPU PTX/HIP/Metal 类型名 (仅 Native 策略)。
    ///
    /// 用于生成 .f32/.f16/.bf16 等 PTX 类型后缀。
    /// 非 Native 策略返回 Err，禁止静默回退。
    pub fn gpu_native_type_name(&self) -> Result<&'static str, ()> {
        match self.gpu_elem_strategy() {
            GpuElemStrategy::Native => match self.kind {
                DTypeKind::F32 | DTypeKind::TF32 => Ok("f32"),
                DTypeKind::F16 => Ok("f16"),
                DTypeKind::BF16 => Ok("bf16"),
                DTypeKind::INT8 => Ok("s8"),
                DTypeKind::FP8E4M3 => Ok("e4m3"),
                DTypeKind::FP8E5M2 => Ok("e5m2"),
                DTypeKind::FP4E2M1 => Ok("e2m1"),
                _ => Err(()),
            },
            GpuElemStrategy::WidenCompute | GpuElemStrategy::DequantCompute => Err(()),
        }
    }

    /// 计算字节数 (GPU lowering 用)。
    ///
    /// 与 elem_bytes() 相同但对 WidenCompute 策略类型返回 widen 后的字节数。
    pub fn gpu_compute_bytes(&self) -> usize {
        match self.gpu_elem_strategy() {
            GpuElemStrategy::Native => self.elem_bytes().max(1),
            GpuElemStrategy::WidenCompute => 4,
            GpuElemStrategy::DequantCompute => 4,
        }
    }

    /// Convert to DType for ISA strategy selection.
    /// Sub-byte types map to F32 (they widen for computation).
    pub fn to_dtype(self) -> crate::types::DType {
        match self.kind {
            DTypeKind::F32 | DTypeKind::TF32 => crate::types::DType::F32,
            DTypeKind::BF16 => crate::types::DType::BF16,
            DTypeKind::F16 => crate::types::DType::F16,
            _ => crate::types::DType::F32,
        }
    }

    /// 累加器精度 (FIX-15: 替代硬编码 F32)。
    ///
    /// 所有 dtype 的累加操作都使用 F32 精度（数值稳定性保证）。
    /// 替代硬编码 `QuantPrecision::F32`，遵循 ARCH-DTYPE-JIT-TYPED 铁律。
    pub fn accumulator_precision(&self) -> Self {
        QuantPrecision::F32
    }
}

/// 类型提升规则 (SPEC 00-PHILOSOPHY §4.2)。
pub fn promote(a: QuantPrecision, b: QuantPrecision) -> QuantPrecision {
    if a == b { return a; }
    // 打包格式不同 → 提升 F32
    if a.packing != b.packing || a.kind != b.kind {
        return QuantPrecision::F32;
    }
    match (a.kind, b.kind) {
        _ if a == QuantPrecision::F32 || b == QuantPrecision::F32 => QuantPrecision::F32,
        (DTypeKind::BF16, DTypeKind::F16) | (DTypeKind::F16, DTypeKind::BF16) => QuantPrecision::F32,
        _ => QuantPrecision::F32,
    }
}

/// 根据 TraceOp 语义自动推断结果 dtype (SPEC 00-PHILOSOPHY §4.1)。
pub fn infer_result_dtype(op: &TraceOp, slots: &[TypedSlot]) -> QuantPrecision {
    fn slot_dtype(slots: &[TypedSlot], idx: ValueId) -> QuantPrecision {
        slots.get(idx.0 as usize).map(|s| s.dtype).unwrap_or(QuantPrecision::F32)
    }
    match op {
        TraceOp::Input(n) => slots.get(*n as usize).map(|s| s.dtype).unwrap_or(QuantPrecision::F32),
        TraceOp::Const(_) => QuantPrecision::F32,
        TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b) | TraceOp::Div(a, b)
        | TraceOp::Pow(a, b)
        | TraceOp::Max(a, b) | TraceOp::Min(a, b) => promote(slot_dtype(slots, *a), slot_dtype(slots, *b)),
        TraceOp::Fma(a, b, c) => promote(promote(slot_dtype(slots, *a), slot_dtype(slots, *b)), slot_dtype(slots, *c)),
        TraceOp::Neg(a) | TraceOp::Abs(a) | TraceOp::Sqrt(a) | TraceOp::Rsqrt(a) | TraceOp::Recip(a) => slot_dtype(slots, *a),
        TraceOp::Exp(_) | TraceOp::Log(_) | TraceOp::Sigmoid(_) | TraceOp::Tanh(_) => QuantPrecision::F32,
        TraceOp::Cast { to, .. } => *to,
        TraceOp::HReduce { .. } | TraceOp::Compare { .. } => QuantPrecision::F32,
        TraceOp::QuantFma { .. } | TraceOp::Mxfp4Dequant { .. } => QuantPrecision::F32,
        TraceOp::BroadcastScalar { .. } | TraceOp::BroadcastLoad { .. } | TraceOp::ConditionalBranch(..) => QuantPrecision::F32,
        TraceOp::BlockScale { data, .. } => slot_dtype(slots, *data),
        TraceOp::Prefetch { .. } | TraceOp::NonTemporalStore => QuantPrecision::F32,
        TraceOp::BitExtract { src, .. } => slot_dtype(slots, *src),
        TraceOp::Permute { src, .. } => slot_dtype(slots, *src),
        TraceOp::MaskedOp { op, .. } => infer_result_dtype(op, slots),
        TraceOp::AtomicAdd { .. } => QuantPrecision::F32,
        TraceOp::FWHT { src, .. } => slot_dtype(slots, *src),
        TraceOp::ScalarLoad { .. } => QuantPrecision::F32,
        TraceOp::StrideMul { .. } => QuantPrecision::F32,
        TraceOp::PtrAdd { .. } => QuantPrecision::F32,
        TraceOp::VecLoadIndexed { .. } => QuantPrecision::F32,
        TraceOp::VecStoreIndexed { .. } => QuantPrecision::F32,
        TraceOp::GatherLoad { .. } | TraceOp::ScatterStore { .. } => QuantPrecision::F32,
        TraceOp::TableLookup { .. } => QuantPrecision::F32,
        TraceOp::BitAnd(_, _) => QuantPrecision::F32,
        // ── Quant* 解码 TraceOp (SPEC 23-QUANT-CODEGEN-ALGO §3) ──
        TraceOp::QuantBitAnd { .. } => QuantPrecision::F32,
        TraceOp::QuantBitOr { .. } => QuantPrecision::F32,
        TraceOp::QuantBroadcast { .. } => QuantPrecision::F32,
        TraceOp::QuantCastF16toF32 { .. } => QuantPrecision::F32,
        TraceOp::QuantCastI8toF32 { .. } => QuantPrecision::F32,
        TraceOp::QuantCastFp8toF32 { .. } => QuantPrecision::F32,
        TraceOp::QuantCodebookLookup { .. } => QuantPrecision::F32,
        TraceOp::QuantExtractBits { .. } => QuantPrecision::F32,
        TraceOp::QuantDequantFma { acc, .. } => slot_dtype(slots, *acc),
        TraceOp::QuantIntDivConst { .. } => QuantPrecision::F32,
        TraceOp::QuantIntMul { .. } => QuantPrecision::F32,
        TraceOp::QuantInterleave { lo, .. } | TraceOp::QuantConcatSeq { lo, .. } => slot_dtype(slots, *lo),
        TraceOp::QuantPtrAddOffset { .. } | TraceOp::QuantPtrAddDynamic { .. } | TraceOp::QuantScalarLoad { .. } | TraceOp::QuantLoadF16toF32 { .. } | TraceOp::QuantLoadI8toF32 { .. } | TraceOp::QuantLoadBytesVec { .. } | TraceOp::QuantAndMask { .. } | TraceOp::QuantKQuantPackedScaleLookup { .. } => QuantPrecision::F32,
        TraceOp::QuantShiftLeft { src, .. } => slot_dtype(slots, *src),
        TraceOp::QuantShiftRight { src, .. } => slot_dtype(slots, *src),
        TraceOp::QuantE2m1LutDecode { .. } => QuantPrecision::F32,
        // ── SPEC 24-QUANT-PIPELINE-JIT §1.3: QuantGather trace pipeline ──
        TraceOp::QuantScaleLoad { .. } => QuantPrecision::F32,
        TraceOp::QuantDataLoad { .. } => QuantPrecision::F32,
        TraceOp::QuantZeroLoad { .. } => QuantPrecision::F32,
        TraceOp::QuantSubScaleLoad { .. } => QuantPrecision::F32,
        TraceOp::QuantHighBitsLoad { .. } => QuantPrecision::F32,
        TraceOp::QuantCodebookDequant { .. } => QuantPrecision::F32,
        TraceOp::QuantQ3KDecode { .. } => QuantPrecision::F32,
        // ── SPEC 24-QUANT-PIPELINE-JIT §1.3: QuantGather/QuantGemm structural ──
        TraceOp::QuantGather { .. } => QuantPrecision::F32,
        TraceOp::QuantGemm { .. } => QuantPrecision::F32,
        // ── SPEC 27 AT-002: 结构型扩展 — 暂返回 F32（模板解释器后续处理） ──
        TraceOp::Loop { .. } => QuantPrecision::F32,
        TraceOp::PanelLoad { .. } => QuantPrecision::F32,
        TraceOp::PanelStore { .. } => QuantPrecision::F32,
        TraceOp::PackBuffer { .. } => QuantPrecision::F32,
        TraceOp::SharedMemDeclare { .. } => QuantPrecision::F32,
        TraceOp::AsyncCopyToShared { .. } => QuantPrecision::F32,
        TraceOp::Tma2DCopy { .. } => QuantPrecision::F32,
        TraceOp::AsyncWaitGroup { .. } => QuantPrecision::F32,
        TraceOp::SyncBarrier { .. } => QuantPrecision::F32,
        TraceOp::TileConfig { .. } => QuantPrecision::F32,
        TraceOp::TileMma { .. } => QuantPrecision::F32,
        TraceOp::TileRelease => QuantPrecision::F32,
        TraceOp::Softmax { .. } => QuantPrecision::F32,
        TraceOp::EpilogueChain { .. } => QuantPrecision::F32,
        TraceOp::MtpDraft { .. }
        | TraceOp::MlaAttnScore { .. }
        | TraceOp::MlaRopeMerge { .. } => QuantPrecision::F32,
        // ── SPEC 37 REQ-HWACC-007: DynamicPrecisionSelect ──
        // 返回第一个候选精度（最高精度），作为默认计算精度。
        // 后续 GEMM 根据实际选择的精度索引调整。
        TraceOp::DynamicPrecisionSelect { candidates, .. } => {
            candidates.first().copied().unwrap_or(QuantPrecision::F32)
        }
    }
}

impl TraceOp {
    /// Visit all ValueId references in this op (read-only).
    pub fn visit_value_ids<F: FnMut(ValueId)>(&self, mut f: F) {
        self.visit_value_ids_inner(&mut f);
    }

    /// Inner implementation that takes `&mut F` to avoid recursive closure
    /// type instantiation (which would exceed the compiler recursion limit).
    fn visit_value_ids_inner<F: FnMut(ValueId)>(&self, f: &mut F) {
        match self {
            // No ValueId fields
            TraceOp::Input(_) | TraceOp::Const(_) | TraceOp::Prefetch { .. }
            | TraceOp::NonTemporalStore | TraceOp::SharedMemDeclare { .. }
            | TraceOp::AsyncWaitGroup { .. } | TraceOp::SyncBarrier { .. }
            | TraceOp::TileConfig { .. } | TraceOp::TileRelease
            | TraceOp::EpilogueChain { .. }
            | TraceOp::QuantGather { .. } | TraceOp::QuantGemm { .. }
            | TraceOp::MtpDraft { .. }
            | TraceOp::MlaAttnScore { .. }
            | TraceOp::MlaRopeMerge { .. } => {}

            // Binary (2 ValueId)
            TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b)
            | TraceOp::Div(a, b) | TraceOp::Pow(a, b)
            | TraceOp::Max(a, b) | TraceOp::Min(a, b)
            | TraceOp::BitAnd(a, b)
            | TraceOp::QuantBitAnd { lhs: a, rhs: b }
            | TraceOp::QuantBitOr { lhs: a, rhs: b }
            | TraceOp::QuantInterleave { lo: a, hi: b }
            | TraceOp::QuantConcatSeq { lo: a, hi: b }
            | TraceOp::Permute { src: a, indices: b }
            | TraceOp::AtomicAdd { addr: a, val: b }
            | TraceOp::Compare { a, b, .. }
            | TraceOp::BlockScale { data: a, scale: b, .. }
            | TraceOp::QuantPtrAddDynamic { base: a, index: b }
            | TraceOp::PackBuffer { src: a, dst: b, .. }
            | TraceOp::Softmax { src: a, dst: b }
            | TraceOp::GatherLoad { base: a, indices: b, .. }
            | TraceOp::ScatterStore { base: a, indices: b, .. }
            | TraceOp::TableLookup { base: a, row_index: b, .. }
            | TraceOp::QuantKQuantPackedScaleLookup { scales_base: a, sub_block_idx: b, .. }
            | TraceOp::QuantCodebookDequant { indices: a, codebook_ptr: b, .. } => {
                f(*a); f(*b);
            }

            TraceOp::QuantQ3KDecode { block_base: a, lane_offset: b, d_slot: c, .. } => {
                f(*a); f(*b); f(*c);
            }

            // Ternary (3 ValueId)
            TraceOp::Fma(a, b, c)
            | TraceOp::ConditionalBranch(a, b, c)
            | TraceOp::QuantDequantFma { acc: a, a: b, b: c }
            | TraceOp::TileMma { c: a, a: b, b: c, m: _, n: _, k: _ } => {
                f(*a); f(*b); f(*c);
            }

            // Unary (1 ValueId)
            TraceOp::Neg(a) | TraceOp::Abs(a) | TraceOp::Exp(a)
            | TraceOp::Sqrt(a) | TraceOp::Rsqrt(a) | TraceOp::Recip(a)
            | TraceOp::Log(a) | TraceOp::Sigmoid(a) | TraceOp::Tanh(a)
            | TraceOp::HReduce { src: a, .. }
            | TraceOp::BitExtract { src: a, .. }
            | TraceOp::FWHT { src: a, .. }
            | TraceOp::BroadcastScalar { src: a }
            | TraceOp::Cast { src: a, .. }
            | TraceOp::StrideMul { value: a, .. }
            | TraceOp::QuantFma { acc: a, .. } => { f(*a); }

            TraceOp::QuantBroadcast { src: a, .. }
            | TraceOp::QuantCastF16toF32 { src: a }
            | TraceOp::QuantCastI8toF32 { src: a }
            | TraceOp::QuantCastFp8toF32 { src: a, .. }
            | TraceOp::QuantIntDivConst { src: a, .. }
            | TraceOp::QuantIntMul { src: a, .. }
            | TraceOp::QuantShiftLeft { src: a, .. }
            | TraceOp::QuantShiftRight { src: a, .. }
            | TraceOp::QuantScalarLoad { ptr: a, .. }
            | TraceOp::QuantLoadF16toF32 { ptr: a, .. }
            | TraceOp::QuantLoadI8toF32 { ptr: a, .. }
            | TraceOp::QuantLoadBytesVec { ptr: a, .. }
            | TraceOp::QuantPtrAddOffset { base: a, .. }
            | TraceOp::QuantExtractBits { src: a, .. }
            | TraceOp::QuantCodebookLookup { indices: a, .. }
            | TraceOp::QuantAndMask { src: a, .. } => { f(*a); }

            // QuantXxxLoad (1 ValueId: source)
            TraceOp::QuantScaleLoad { source: a, .. }
            | TraceOp::QuantDataLoad { source: a, .. }
            | TraceOp::QuantZeroLoad { source: a, .. }
            | TraceOp::QuantSubScaleLoad { block_ptr: a, .. }
            | TraceOp::QuantHighBitsLoad { block_ptr: a, .. } => { f(*a); }

            // Dual ValueId (base + offset / base + value)
            TraceOp::ScalarLoad { base: a, offset: b }
            | TraceOp::PtrAdd { base: a, offset: b }
            | TraceOp::VecLoadIndexed { base: a, offset: b }
            | TraceOp::BroadcastLoad { base: a, offset: b }
            | TraceOp::PanelLoad { base: a, offset: b, .. }
            | TraceOp::PanelStore { base: a, offset: b, .. } => { f(*a); f(*b); }

            TraceOp::VecStoreIndexed { base: a, offset: b, value: c } => {
                f(*a); f(*b); f(*c);
            }

            TraceOp::QuantE2m1LutDecode { packed_data_ptr: a, scale_byte: b, .. } => {
                f(*a); f(*b);
            }

            TraceOp::Mxfp4Dequant { data, scales, off_a, off_b, off_c, .. } => {
                f(*data); f(*scales);
                if let Some(a) = off_a { f(*a); }
                if let Some(b) = off_b { f(*b); }
                if let Some(c) = off_c { f(*c); }
            }

            TraceOp::MaskedOp { op, mask } => {
                op.visit_value_ids_inner(f);
                f(*mask);
            }

            TraceOp::AsyncCopyToShared { src_offset, .. } => { f(*src_offset); }

            TraceOp::Tma2DCopy { coord_x, coord_y, .. } => { f(*coord_x); f(*coord_y); }

            TraceOp::Loop { body: inner, .. } => {
                for op in inner { op.visit_value_ids_inner(f); }
            }

            TraceOp::DynamicPrecisionSelect { tensor, .. } => { f(*tensor); }
        }
    }

    /// Remap all ValueId references through a mapping function.
    pub fn map_value_ids(self, map_fn: &impl Fn(ValueId) -> ValueId) -> Self {
        let m = |vid: ValueId| -> ValueId { map_fn(vid) };
        match self {
            // No ValueId → identity
            TraceOp::Input(n) => TraceOp::Input(n),
            TraceOp::Const(v) => TraceOp::Const(v),
            TraceOp::Prefetch { level } => TraceOp::Prefetch { level },
            TraceOp::NonTemporalStore => TraceOp::NonTemporalStore,
            TraceOp::SharedMemDeclare { name, bytes } => TraceOp::SharedMemDeclare { name, bytes },
            TraceOp::AsyncWaitGroup { n } => TraceOp::AsyncWaitGroup { n },
            TraceOp::SyncBarrier { name } => TraceOp::SyncBarrier { name },
            TraceOp::TileConfig { rows, cols } => TraceOp::TileConfig { rows, cols },
            TraceOp::TileRelease => TraceOp::TileRelease,
            TraceOp::EpilogueChain { ops } => TraceOp::EpilogueChain { ops },

            // Binary
            TraceOp::Add(a, b) => TraceOp::Add(m(a), m(b)),
            TraceOp::Sub(a, b) => TraceOp::Sub(m(a), m(b)),
            TraceOp::Mul(a, b) => TraceOp::Mul(m(a), m(b)),
            TraceOp::Div(a, b) => TraceOp::Div(m(a), m(b)),
            TraceOp::Pow(a, b) => TraceOp::Pow(m(a), m(b)),
            TraceOp::Max(a, b) => TraceOp::Max(m(a), m(b)),
            TraceOp::Min(a, b) => TraceOp::Min(m(a), m(b)),
            TraceOp::BitAnd(a, b) => TraceOp::BitAnd(m(a), m(b)),

            // Ternary
            TraceOp::Fma(a, b, c) => TraceOp::Fma(m(a), m(b), m(c)),
            TraceOp::ConditionalBranch(a, b, c) => TraceOp::ConditionalBranch(m(a), m(b), m(c)),

            // Unary
            TraceOp::Neg(a) => TraceOp::Neg(m(a)),
            TraceOp::Abs(a) => TraceOp::Abs(m(a)),
            TraceOp::Exp(a) => TraceOp::Exp(m(a)),
            TraceOp::Sqrt(a) => TraceOp::Sqrt(m(a)),
            TraceOp::Rsqrt(a) => TraceOp::Rsqrt(m(a)),
            TraceOp::Recip(a) => TraceOp::Recip(m(a)),
            TraceOp::Log(a) => TraceOp::Log(m(a)),
            TraceOp::Sigmoid(a) => TraceOp::Sigmoid(m(a)),
            TraceOp::Tanh(a) => TraceOp::Tanh(m(a)),

            // Extended §12+§14
            TraceOp::QuantFma { acc, act, weight, act_dtype, weight_dtype } =>
                TraceOp::QuantFma { acc: m(acc), act: m(act), weight: m(weight), act_dtype, weight_dtype },
            TraceOp::BlockScale { data, scale, block_size } =>
                TraceOp::BlockScale { data: m(data), scale: m(scale), block_size },
            TraceOp::Cast { src, from, to } => TraceOp::Cast { src: m(src), from, to },
            TraceOp::HReduce { src, op } => TraceOp::HReduce { src: m(src), op },
            TraceOp::Compare { a, b, op } => TraceOp::Compare { a: m(a), b: m(b), op },
            TraceOp::MaskedOp { op, mask } => TraceOp::MaskedOp { op: Box::new(op.map_value_ids(map_fn)), mask: m(mask) },
            TraceOp::BitExtract { src, offset, width } => TraceOp::BitExtract { src: m(src), offset, width },
            TraceOp::Permute { src, indices } => TraceOp::Permute { src: m(src), indices: m(indices) },
            TraceOp::AtomicAdd { addr, val } => TraceOp::AtomicAdd { addr: m(addr), val: m(val) },
            TraceOp::FWHT { src, dim } => TraceOp::FWHT { src: m(src), dim },
            TraceOp::ScalarLoad { base, offset } => TraceOp::ScalarLoad { base: m(base), offset: m(offset) },
            TraceOp::StrideMul { value, stride } => TraceOp::StrideMul { value: m(value), stride },
            TraceOp::PtrAdd { base, offset } => TraceOp::PtrAdd { base: m(base), offset: m(offset) },
            TraceOp::VecLoadIndexed { base, offset } => TraceOp::VecLoadIndexed { base: m(base), offset: m(offset) },
            TraceOp::VecStoreIndexed { base, offset, value } =>
                TraceOp::VecStoreIndexed { base: m(base), offset: m(offset), value: m(value) },
            TraceOp::BroadcastScalar { src } => TraceOp::BroadcastScalar { src: m(src) },
            TraceOp::BroadcastLoad { base, offset } => TraceOp::BroadcastLoad { base: m(base), offset: m(offset) },
            TraceOp::GatherLoad { base, indices, stride } => TraceOp::GatherLoad { base: m(base), indices: m(indices), stride },
            TraceOp::ScatterStore { base, indices, value, stride } =>
                TraceOp::ScatterStore { base: m(base), indices: m(indices), value: m(value), stride },
            TraceOp::TableLookup { base, row_index, row_bytes } =>
                TraceOp::TableLookup { base: m(base), row_index: m(row_index), row_bytes },
            TraceOp::Mxfp4Dequant { data, scales, off_a, stride_a, off_b, stride_b, off_c, const_off, block_size } =>
                TraceOp::Mxfp4Dequant {
                    data: m(data), scales: m(scales),
                    off_a: off_a.map(&m), stride_a, off_b: off_b.map(&m), stride_b,
                    off_c: off_c.map(&m), const_off, block_size,
                },

            // SPEC 23 Quant* decode
            TraceOp::QuantBitAnd { lhs, rhs } => TraceOp::QuantBitAnd { lhs: m(lhs), rhs: m(rhs) },
            TraceOp::QuantBitOr { lhs, rhs } => TraceOp::QuantBitOr { lhs: m(lhs), rhs: m(rhs) },
            TraceOp::QuantBroadcast { src, lanes } => TraceOp::QuantBroadcast { src: m(src), lanes },
            TraceOp::QuantCastF16toF32 { src } => TraceOp::QuantCastF16toF32 { src: m(src) },
            TraceOp::QuantCastI8toF32 { src } => TraceOp::QuantCastI8toF32 { src: m(src) },
            TraceOp::QuantCastFp8toF32 { src, format } => TraceOp::QuantCastFp8toF32 { src: m(src), format },
            TraceOp::QuantCodebookLookup { indices, codebook_data, vector_size, bits_per_entry } =>
                TraceOp::QuantCodebookLookup { indices: m(indices), codebook_data, vector_size, bits_per_entry },
            TraceOp::QuantExtractBits { src, bit_offset, bit_width } =>
                TraceOp::QuantExtractBits { src: m(src), bit_offset, bit_width },
            TraceOp::QuantDequantFma { acc, a, b } => TraceOp::QuantDequantFma { acc: m(acc), a: m(a), b: m(b) },
            TraceOp::QuantIntDivConst { src, divisor } => TraceOp::QuantIntDivConst { src: m(src), divisor },
            TraceOp::QuantIntMul { src, factor } => TraceOp::QuantIntMul { src: m(src), factor },
            TraceOp::QuantInterleave { lo, hi } => TraceOp::QuantInterleave { lo: m(lo), hi: m(hi) },
            TraceOp::QuantConcatSeq { lo, hi } => TraceOp::QuantConcatSeq { lo: m(lo), hi: m(hi) },
            TraceOp::QuantPtrAddOffset { base, offset_bytes } => TraceOp::QuantPtrAddOffset { base: m(base), offset_bytes },
            TraceOp::QuantPtrAddDynamic { base, index } => TraceOp::QuantPtrAddDynamic { base: m(base), index: m(index) },
            TraceOp::QuantAndMask { src, mask } => TraceOp::QuantAndMask { src: m(src), mask },
            TraceOp::QuantKQuantPackedScaleLookup { scales_base, sub_block_idx, scale_algo, selector } =>
                TraceOp::QuantKQuantPackedScaleLookup { scales_base: m(scales_base), sub_block_idx: m(sub_block_idx), scale_algo, selector },
            TraceOp::QuantScalarLoad { ptr, offset_bytes } => TraceOp::QuantScalarLoad { ptr: m(ptr), offset_bytes },
            TraceOp::QuantLoadF16toF32 { ptr, offset_bytes } => TraceOp::QuantLoadF16toF32 { ptr: m(ptr), offset_bytes },
            TraceOp::QuantLoadI8toF32 { ptr, offset_bytes } => TraceOp::QuantLoadI8toF32 { ptr: m(ptr), offset_bytes },
            TraceOp::QuantLoadBytesVec { ptr, offset_bytes, count, signed } =>
                TraceOp::QuantLoadBytesVec { ptr: m(ptr), offset_bytes, count, signed },
            TraceOp::QuantShiftLeft { src, amount } => TraceOp::QuantShiftLeft { src: m(src), amount },
            TraceOp::QuantShiftRight { src, amount } => TraceOp::QuantShiftRight { src: m(src), amount },
            TraceOp::QuantE2m1LutDecode { packed_data_ptr, scale_byte, nvfp4_mode } =>
                TraceOp::QuantE2m1LutDecode { packed_data_ptr: m(packed_data_ptr), scale_byte: m(scale_byte), nvfp4_mode },
            TraceOp::QuantQ3KDecode { block_base, lane_offset, d_slot, qs_offset, hmask_offset } =>
                TraceOp::QuantQ3KDecode { block_base: m(block_base), lane_offset: m(lane_offset), d_slot: m(d_slot), qs_offset, hmask_offset },
            // SPEC 24 QuantGather/QuantGemm pipeline
            TraceOp::QuantScaleLoad { source, offset, dtype } =>
                TraceOp::QuantScaleLoad { source: m(source), offset, dtype },
            TraceOp::QuantDataLoad { source, offset, quant_type, block_size } =>
                TraceOp::QuantDataLoad { source: m(source), offset, quant_type, block_size },
            TraceOp::QuantZeroLoad { source, offset, zp_type } =>
                TraceOp::QuantZeroLoad { source: m(source), offset, zp_type },
            TraceOp::QuantSubScaleLoad { block_ptr, byte_offset, bits, sub_block_size } =>
                TraceOp::QuantSubScaleLoad { block_ptr: m(block_ptr), byte_offset, bits, sub_block_size },
            TraceOp::QuantHighBitsLoad { block_ptr, byte_offset, bits_per_elem } =>
                TraceOp::QuantHighBitsLoad { block_ptr: m(block_ptr), byte_offset, bits_per_elem },
            TraceOp::QuantCodebookDequant { indices, codebook_ptr, vector_size, bits_per_entry } =>
                TraceOp::QuantCodebookDequant { indices: m(indices), codebook_ptr: m(codebook_ptr), vector_size, bits_per_entry },

            // SPEC 24 QuantGather/QuantGemm structural (no ValueId fields)
            TraceOp::QuantGather { quant_type, vocab_size, hidden_dim } =>
                TraceOp::QuantGather { quant_type, vocab_size, hidden_dim },
            TraceOp::QuantGemm { quant_type, m: m_val, n, k } =>
                TraceOp::QuantGemm { quant_type, m: m_val, n, k },

            // SPEC 27 Structural
            TraceOp::Loop { bound, step_bytes, body } =>
                TraceOp::Loop { bound, step_bytes, body: body.into_iter().map(|op| op.map_value_ids(map_fn)).collect() },
            TraceOp::PanelLoad { base, offset, rows, cols } =>
                TraceOp::PanelLoad { base: m(base), offset: m(offset), rows, cols },
            TraceOp::PanelStore { base, offset, rows, cols } =>
                TraceOp::PanelStore { base: m(base), offset: m(offset), rows, cols },
            TraceOp::PackBuffer { src, dst, rows, cols, layout } =>
                TraceOp::PackBuffer { src: m(src), dst: m(dst), rows, cols, layout },
            TraceOp::AsyncCopyToShared { name, src_offset, bytes } =>
                TraceOp::AsyncCopyToShared { name, src_offset: m(src_offset), bytes },
            TraceOp::Tma2DCopy { desc, coord_x, coord_y, bytes } =>
                TraceOp::Tma2DCopy { desc, coord_x: m(coord_x), coord_y: m(coord_y), bytes },
            TraceOp::TileMma { c, a, b, m: mm, n: nn, k: kk } =>
                TraceOp::TileMma { c: m(c), a: m(a), b: m(b), m: mm, n: nn, k: kk },
            TraceOp::Softmax { src, dst } => TraceOp::Softmax { src: m(src), dst: m(dst) },

            // MTP Draft structural (no ValueId fields)
            TraceOp::MtpDraft { depth, hidden_size, vocab_size } =>
                TraceOp::MtpDraft { depth, hidden_size, vocab_size },
            TraceOp::MlaAttnScore { num_heads, head_dim, d_c, d_rope } =>
                TraceOp::MlaAttnScore { num_heads, head_dim, d_c, d_rope },
            TraceOp::MlaRopeMerge { d_c, d_rope } =>
                TraceOp::MlaRopeMerge { d_c, d_rope },

            // SPEC 37 REQ-HWACC-007: DynamicPrecisionSelect
            TraceOp::DynamicPrecisionSelect { tensor, candidates, thresholds } =>
                TraceOp::DynamicPrecisionSelect { tensor: m(tensor), candidates, thresholds },
        }
    }

    /// Whether this op has observable side effects (cannot be DCE'd).
    pub fn has_side_effects(&self) -> bool {
        matches!(self,
            TraceOp::VecStoreIndexed { .. }
            | TraceOp::ScatterStore { .. }
            | TraceOp::AtomicAdd { .. }
            | TraceOp::NonTemporalStore
            | TraceOp::PanelStore { .. }
            | TraceOp::Softmax { .. }
            | TraceOp::Loop { .. }
            | TraceOp::SyncBarrier { .. }
            | TraceOp::AsyncCopyToShared { .. }
            | TraceOp::Tma2DCopy { .. }
            | TraceOp::EpilogueChain { .. }
            | TraceOp::QuantGather { .. }
            | TraceOp::QuantGemm { .. }
            | TraceOp::MtpDraft { .. }
        )
    }

    /// Whether this op is pure (no side effects, safe for CSE).
    pub fn is_pure(&self) -> bool {
        !self.has_side_effects()
            && !matches!(self, TraceOp::Input(_) | TraceOp::Const(_))
    }
}

/// 水平归约类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
    Prod,
    Count,    // count(predicate == true)
    ArgMax,   // index of max element (§13.2 centroid)
    LogSum,   // log(sum(exp(x))) — softmax 分母 (数值稳定: max + log(sum(exp(x-max))))
}

/// 缓存层级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
    NonTemporal,
}

/// 比较操作
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Scalar function signature — pointer + parameter layout.
#[derive(Debug, Clone)]
pub struct ScalarFnSignature {
    /// Address of the `extern "C"` scalar function.
    pub fn_ptr: *const u8,
    /// Ordered parameter descriptors.
    pub params: Vec<ScalarParam>,
}

// SAFETY: fn_ptr points to a static extern "C" function in the binary's text segment.
// Static function pointers are inherently thread-safe (read-only, never deallocated).
unsafe impl Send for ScalarFnSignature {}
unsafe impl Sync for ScalarFnSignature {}

/// Describes one parameter of a scalar function.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarParam {
    InputPtr,
    OutputPtr,
    WeightPtr,
    Dim(usize),
    Scalar(f32),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPEC 24-QUANT-PIPELINE-JIT §1.4: 参数化 trace 模板
// QuantGather / QuantGemm 的 trace 生成函数。
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Build a QuantGather trace template for quantized embedding lookup.
///
/// Produces a trace of the form:
///   Input(0) — indices_ptr (token IDs)
///   Input(1) — embed_ptr (quantized embed table)
///   Input(2) — output_ptr (F32 embeddings output)
///   QuantGather { quant_type, vocab_size, hidden_dim }
///
/// The QuantGather marker is expanded by auto_select into the full
/// emit_quant_gather_inline loop nest with DecodeTraceBuilder block decode.
pub fn build_quant_gather_trace(
    quant_type: QuantType,
    vocab_size: usize,
    hidden_dim: usize,
) -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // indices_ptr
        TraceOp::Input(1), // embed_ptr
        TraceOp::Input(2), // output_ptr
        TraceOp::QuantGather {
            quant_type,
            vocab_size,
            hidden_dim,
        },
    ]
}

/// Build a QuantGemm trace template for quantized matrix multiplication.
///
/// Produces a trace of the form:
///   Input(0) — input_ptr (activation)
///   Input(1) — weight_ptr (quantized weight matrix)
///   Input(2) — output_ptr (F32 output)
///   QuantGemm { quant_type, m, n, k }
///
/// The QuantGemm marker is expanded by auto_select into the full
/// emit_quant_gemm_inline tiled loop nest with block decode + FMA.
pub fn build_quant_gemm_trace(
    quant_type: QuantType,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<TraceOp> {
    vec![
        TraceOp::Input(0), // input_ptr
        TraceOp::Input(1), // weight_ptr
        TraceOp::Input(2), // output_ptr
        TraceOp::QuantGemm {
            quant_type,
            m,
            n,
            k,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_silu_body_is_valid_ssa() {
        let body = vec![
            TraceOp::Input(0),              // [0] v
            TraceOp::Neg(ValueId(0)),       // [1] -v
            TraceOp::Exp(ValueId(1)),       // [2] exp(-v)
            TraceOp::Const(1.0),            // [3] 1.0
            TraceOp::Add(ValueId(2), ValueId(3)), // [4] 1 + exp(-v)
            TraceOp::Div(ValueId(0), ValueId(4)), // [5] v / (1 + exp(-v))
        ];

        let trace = OpTrace {
            pattern: ComputePattern::Elementwise { body: body.clone() },
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
            },
        };

        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::Neg(a) | TraceOp::Abs(a) | TraceOp::Exp(a)
                | TraceOp::Sqrt(a) | TraceOp::Rsqrt(a) | TraceOp::Tanh(a)
                | TraceOp::Recip(a) | TraceOp::Log(a) | TraceOp::Sigmoid(a) => {
                    assert!(a.0 < i as u32, "SSA violation at index {i}: operand {a}");
                }
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b)
                | TraceOp::Div(a, b) | TraceOp::Max(a, b) | TraceOp::Min(a, b) => {
                    assert!(a.0 < i as u32, "SSA violation at index {i}: operand {a}");
                    assert!(b.0 < i as u32, "SSA violation at index {i}: operand {b}");
                }
                TraceOp::Fma(a, b, c) => {
                    assert!(a.0 < i as u32, "SSA violation at index {i}: operand {a}");
                    assert!(b.0 < i as u32, "SSA violation at index {i}: operand {b}");
                    assert!(c.0 < i as u32, "SSA violation at index {i}: operand {c}");
                }
                TraceOp::ConditionalBranch(mask, t_val, f_val) => {
                    assert!(mask.0 < i as u32, "SSA violation at index {i}: operand {mask}");
                    assert!(t_val.0 < i as u32, "SSA violation at index {i}: operand {t_val}");
                    assert!(f_val.0 < i as u32, "SSA violation at index {i}: operand {f_val}");
                }
                // Extended §12+§14 variants: SSA validation for these is handled
                // per-variant in their own tests; skip here to keep this test focused.
                _ => {}
            }
        }

        assert!(matches!(trace.pattern, ComputePattern::Elementwise { .. }));
    }

    #[test]
    fn trace_gelu_body_is_valid_ssa() {
        let body = vec![
            TraceOp::Input(0),                              // [0] x
            TraceOp::Mul(ValueId(0), ValueId(0)),           // [1] x^2
            TraceOp::Mul(ValueId(1), ValueId(0)),           // [2] x^3
            TraceOp::Const(0.044715),                       // [3] 0.044715
            TraceOp::Mul(ValueId(3), ValueId(2)),           // [4] 0.044715 * x^3
            TraceOp::Add(ValueId(0), ValueId(4)),           // [5] x + 0.044715 * x^3
            TraceOp::Const(0.7978845608),                   // [6] sqrt(2/pi)
            TraceOp::Mul(ValueId(6), ValueId(5)),           // [7] sqrt(2/pi) * (x + 0.044715*x^3)
            TraceOp::Tanh(ValueId(7)),                      // [8] tanh(...)
            TraceOp::Const(1.0),                            // [9] 1.0
            TraceOp::Add(ValueId(9), ValueId(8)),           // [10] 1 + tanh(...)
            TraceOp::Const(0.5),                            // [11] 0.5
            TraceOp::Mul(ValueId(11), ValueId(0)),          // [12] 0.5 * x
            TraceOp::Mul(ValueId(12), ValueId(10)),         // [13] 0.5 * x * (1 + tanh(...))
        ];

        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::Mul(a, b) | TraceOp::Add(a, b) => {
                    assert!(a.0 < i as u32);
                    assert!(b.0 < i as u32);
                }
                TraceOp::Tanh(a) => {
                    assert!(a.0 < i as u32);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn trace_rms_norm_pattern() {
        let reduce = vec![
            TraceOp::Input(0),                          // [0] x
            TraceOp::Mul(ValueId(0), ValueId(0)),       // [1] x^2
        ];
        let finalize = vec![
            TraceOp::Input(0),                          // [0] sum_sq (reduction result)
            TraceOp::Input(1),                          // [1] n (dimension)
            TraceOp::Div(ValueId(0), ValueId(1)),       // [2] mean = sum_sq / n
            TraceOp::Const(1e-5),                       // [3] eps
            TraceOp::Add(ValueId(2), ValueId(3)),       // [4] mean + eps
            TraceOp::Rsqrt(ValueId(4)),                 // [5] rsqrt(mean + eps)
        ];
        let transform = vec![
            TraceOp::Input(0),                          // [0] x
            TraceOp::Input(1),                          // [1] scale (from finalize)
            TraceOp::Input(2),                          // [2] weight
            TraceOp::Mul(ValueId(0), ValueId(1)),       // [3] x * scale
            TraceOp::Mul(ValueId(3), ValueId(2)),       // [4] x * scale * weight
        ];

        let pattern = ComputePattern::NormLike { reduce, finalize, transform };
        assert!(matches!(pattern, ComputePattern::NormLike { .. }));
    }

    #[test]
    fn trace_binary_elementwise_add() {
        let body = vec![
            TraceOp::Input(0),                          // [0] a
            TraceOp::Input(1),                          // [1] b
            TraceOp::Add(ValueId(0), ValueId(1)),       // [2] a + b
        ];
        let pattern = ComputePattern::BinaryElementwise { body };
        assert!(matches!(pattern, ComputePattern::BinaryElementwise { .. }));
    }

    #[test]
    fn trace_gemm_pattern() {
        let pattern = ComputePattern::Gemm;
        assert!(matches!(pattern, ComputePattern::Gemm));
    }

    #[test]
    fn trace_scalar_fn_signature_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ScalarFnSignature>();
    }

    #[test]
    fn test_trace_op_log_display() {
        // Verify TraceOp::Log Debug representation — ValueId uses Debug derive (ValueId(N))
        let op = TraceOp::Log(ValueId(0));
        assert_eq!(format!("{op:?}"), "Log(ValueId(0))");

        let op5 = TraceOp::Log(ValueId(5));
        assert_eq!(format!("{op5:?}"), "Log(ValueId(5))");
    }

    // ── ValueId ──

    #[test]
    fn value_id_none_is_not_some() {
        assert!(!ValueId::NONE.is_some());
    }

    #[test]
    fn value_id_regular_is_some() {
        assert!(ValueId(0).is_some());
        assert!(ValueId(42).is_some());
    }

    #[test]
    fn value_id_display() {
        assert_eq!(ValueId(7).to_string(), "v7");
        assert_eq!(ValueId(0).to_string(), "v0");
    }

    #[test]
    fn value_id_sub() {
        assert_eq!(ValueId(10) - 3, ValueId(7));
    }

    #[test]
    fn value_id_saturating_sub() {
        assert_eq!(ValueId(2).saturating_sub(5), ValueId(0));
    }

    #[test]
    fn value_id_equality() {
        assert_eq!(ValueId(3), ValueId(3));
        assert_ne!(ValueId(3), ValueId(4));
    }

    // ── classify_pattern ──

    #[test]
    fn classify_empty_body_is_injective() {
        let body: Vec<TraceOp> = vec![];
        let pattern = classify_pattern(&body);
        assert!(matches!(pattern, ComputePattern::Injective { num_inputs: 0, .. }));
    }

    #[test]
    fn classify_single_input_is_elementwise() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Exp(ValueId(0)),
        ];
        let pattern = classify_pattern(&body);
        assert!(matches!(pattern, ComputePattern::Elementwise { .. }));
    }

    #[test]
    fn classify_two_inputs_is_binary() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];
        let pattern = classify_pattern(&body);
        assert!(matches!(pattern, ComputePattern::BinaryElementwise { .. }));
    }

    #[test]
    fn classify_three_inputs_is_injective() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Input(2),
            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),
        ];
        let pattern = classify_pattern(&body);
        assert!(matches!(pattern, ComputePattern::Injective { num_inputs: 3, .. }));
    }

    // ── ComputePattern::body ──

    #[test]
    fn compute_pattern_gemm_body_is_none() {
        assert!(ComputePattern::Gemm.body().is_none());
    }

    #[test]
    fn compute_pattern_elementwise_body_is_some() {
        let pattern = ComputePattern::Elementwise { body: vec![TraceOp::Input(0)] };
        assert!(pattern.body().is_some());
    }

    // ── GgmlType ──

    #[test]
    fn ggml_type_q4_0_dtype_and_block() {
        assert_eq!(GgmlType::Q4_0.dtype_kind(), DTypeKind::INT4);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
    }

    #[test]
    fn ggml_type_q8_0_dtype_and_block() {
        assert_eq!(GgmlType::Q8_0.dtype_kind(), DTypeKind::INT8);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
    }

    #[test]
    fn ggml_type_q2_k_is_int2_and_256_block() {
        assert_eq!(GgmlType::Q2_K.dtype_kind(), DTypeKind::INT2);
        assert_eq!(GgmlType::Q2_K.block_size(), 256);
    }

    #[test]
    fn ggml_type_q6_k_is_int4_and_256_block() {
        assert_eq!(GgmlType::Q6_K.dtype_kind(), DTypeKind::INT4);
        assert_eq!(GgmlType::Q6_K.block_size(), 256);
    }

    // ── PackingFormat ──

    #[test]
    fn packing_format_equality() {
        assert_eq!(PackingFormat::Plain, PackingFormat::Plain);
        assert_ne!(PackingFormat::Plain, PackingFormat::MXBlock);
        assert_eq!(PackingFormat::GGML(GgmlType::Q4_0), PackingFormat::GGML(GgmlType::Q4_0));
        assert_ne!(PackingFormat::GGML(GgmlType::Q4_0), PackingFormat::GGML(GgmlType::Q8_0));
    }

    // ── ReduceKind / CacheLevel / CmpOp ──

    #[test]
    fn reduce_kind_variants() {
        let kinds = [ReduceKind::Sum, ReduceKind::Max, ReduceKind::Min,
                     ReduceKind::Prod, ReduceKind::Count, ReduceKind::ArgMax, ReduceKind::LogSum];
        assert_eq!(kinds.len(), 7);
    }

    #[test]
    fn cache_level_variants() {
        assert_ne!(CacheLevel::L1, CacheLevel::L2);
        assert_ne!(CacheLevel::L2, CacheLevel::L3);
        assert_ne!(CacheLevel::L3, CacheLevel::NonTemporal);
    }

    #[test]
    fn cmp_op_ordering() {
        assert_ne!(CmpOp::Lt, CmpOp::Le);
        assert_ne!(CmpOp::Gt, CmpOp::Ge);
        assert_ne!(CmpOp::Eq, CmpOp::Ne);
    }

    // ── ScalarParam ──

    #[test]
    fn scalar_param_equality() {
        assert_eq!(ScalarParam::InputPtr, ScalarParam::InputPtr);
        assert_ne!(ScalarParam::InputPtr, ScalarParam::OutputPtr);
        assert_eq!(ScalarParam::Dim(128), ScalarParam::Dim(128));
        assert_ne!(ScalarParam::Dim(64), ScalarParam::Dim(128));
    }

    // ── TraceOp has_side_effects / is_pure ──

    #[test]
    fn trace_op_add_is_pure() {
        let op = TraceOp::Add(ValueId(0), ValueId(1));
        assert!(op.is_pure());
        assert!(!op.has_side_effects());
    }

    #[test]
    fn trace_op_store_has_side_effects() {
        let op = TraceOp::VecStoreIndexed {
            base: ValueId(0),
            offset: ValueId(1),
            value: ValueId(2),
        };
        assert!(op.has_side_effects());
        assert!(!op.is_pure());
    }

    #[test]
    fn trace_op_const_is_leaf_not_pure() {
        let op = TraceOp::Const(3.14);
        assert!(!op.is_pure()); // Input/Const are leaves, not computation
        assert!(!op.has_side_effects());
    }

    // ── TraceOp visit_value_ids ──

    #[test]
    fn trace_op_fma_visits_three_ids() {
        let op = TraceOp::Fma(ValueId(1), ValueId(2), ValueId(3));
        let mut ids = vec![];
        op.visit_value_ids(|id| ids.push(id));
        assert_eq!(ids, vec![ValueId(1), ValueId(2), ValueId(3)]);
    }

    #[test]
    fn trace_op_input_visits_none() {
        let op = TraceOp::Input(0);
        let mut count = 0;
        op.visit_value_ids(|_| count += 1);
        assert_eq!(count, 0);
    }

    // ── TraceOp map_value_ids ──

    #[test]
    fn trace_op_map_add_offsets_ids() {
        let op = TraceOp::Add(ValueId(0), ValueId(1));
        let mapped = op.map_value_ids(&|id| ValueId(id.0 + 10));
        assert_eq!(mapped, TraceOp::Add(ValueId(10), ValueId(11)));
    }

    // ── QuantLoadPass ──

    #[test]
    fn quant_load_pass_variants() {
        assert_ne!(QuantLoadPass::LowNibble, QuantLoadPass::HighNibble);
        assert_ne!(QuantLoadPass::Single, QuantLoadPass::LowNibble);
    }

    // ── DequantMethod / X86ElemStrategy ──

    #[test]
    fn dequant_method_variants() {
        assert_ne!(DequantMethod::VNNI, DequantMethod::AMX);
        assert_ne!(DequantMethod::ScalarLUT, DequantMethod::BlockScale);
    }

    #[test]
    fn x86_strategy_native_vs_dequant() {
        assert_ne!(X86ElemStrategy::Native, X86ElemStrategy::WidenCompute);
        assert_ne!(
            X86ElemStrategy::DequantCompute(DequantMethod::VNNI),
            X86ElemStrategy::DequantCompute(DequantMethod::AMX),
        );
    }

    // ── Wave 12k31: +13 additional tests ──

    #[test]
    fn value_id_sub_wraps_correctly() {
        let v = ValueId(10);
        assert_eq!(v - 3, ValueId(7));
    }

    #[test]
    fn value_id_saturating_sub_at_zero() {
        let v = ValueId(0);
        assert_eq!(v.saturating_sub(5), ValueId(0));
    }

    #[test]
    fn classify_four_inputs_is_injective() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Input(2),
            TraceOp::Input(3),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];
        let pat = classify_pattern(&body);
        match pat {
            ComputePattern::Injective { num_inputs, .. } => assert_eq!(num_inputs, 4),
            other => panic!("expected Injective, got {:?}", other),
        }
    }

    #[test]
    fn trace_op_mul_visits_two_ids() {
        let op = TraceOp::Mul(ValueId(5), ValueId(7));
        let count = std::sync::atomic::AtomicU32::new(0);
        op.visit_value_ids(|_| { count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn trace_op_neg_visits_one_id() {
        let op = TraceOp::Neg(ValueId(3));
        let count = std::sync::atomic::AtomicU32::new(0);
        op.visit_value_ids(|_| { count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn trace_op_exp_is_pure() {
        assert!(TraceOp::Exp(ValueId(0)).is_pure());
    }

    #[test]
    fn trace_op_sqrt_is_pure() {
        assert!(TraceOp::Sqrt(ValueId(0)).is_pure());
    }

    #[test]
    fn trace_op_recip_is_pure() {
        assert!(TraceOp::Recip(ValueId(0)).is_pure());
    }

    #[test]
    fn trace_op_tanh_is_pure() {
        assert!(TraceOp::Tanh(ValueId(0)).is_pure());
    }

    #[test]
    fn trace_op_log_is_pure() {
        assert!(TraceOp::Log(ValueId(0)).is_pure());
    }

    #[test]
    fn trace_op_sigmoid_is_pure() {
        assert!(TraceOp::Sigmoid(ValueId(0)).is_pure());
    }

    #[test]
    fn trace_op_max_is_pure() {
        assert!(TraceOp::Max(ValueId(0), ValueId(1)).is_pure());
    }

    #[test]
    fn trace_op_min_is_pure() {
        assert!(TraceOp::Min(ValueId(0), ValueId(1)).is_pure());
    }

    #[test]
    fn trace_op_abs_is_pure() {
        assert!(TraceOp::Abs(ValueId(0)).is_pure());
    }

    #[test]
    fn trace_op_conditional_branch_is_pure() {
        assert!(TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2)).is_pure());
    }

    #[test]
    fn trace_op_non_temporal_store_not_pure() {
        assert!(!TraceOp::NonTemporalStore.is_pure());
    }

    #[test]
    fn trace_op_const_has_no_visited_ids() {
        let op = TraceOp::Const(3.14);
        let count = std::sync::atomic::AtomicU32::new(0);
        op.visit_value_ids(|_| { count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn trace_op_div_visits_two_ids() {
        let op = TraceOp::Div(ValueId(10), ValueId(20));
        let count = std::sync::atomic::AtomicU32::new(0);
        op.visit_value_ids(|_| { count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn trace_op_sub_visits_two_ids() {
        let op = TraceOp::Sub(ValueId(1), ValueId(2));
        let count = std::sync::atomic::AtomicU32::new(0);
        op.visit_value_ids(|_| { count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    // ── Wave 12k33: +10 additional tests ──

    #[test]
    fn compute_pattern_reduction_body_is_none() {
        let pattern = ComputePattern::Reduction {
            identity: 0.0,
            combine: vec![TraceOp::Add(ValueId(0), ValueId(1))],
            second_pass: None,
            normalize: None,
        };
        assert!(pattern.body().is_none());
    }

    #[test]
    fn compute_pattern_normlike_body_is_none() {
        let pattern = ComputePattern::NormLike {
            reduce: vec![TraceOp::Input(0)],
            finalize: vec![TraceOp::Input(0)],
            transform: vec![TraceOp::Input(0)],
        };
        assert!(pattern.body().is_none());
    }

    #[test]
    fn compute_pattern_injective_body_is_some() {
        let pattern = ComputePattern::Injective {
            body: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(ValueId(0), ValueId(1))],
            num_inputs: 2,
            num_outputs: 1,
        };
        assert!(pattern.body().is_some());
        assert_eq!(pattern.body().unwrap().len(), 3);
    }

    #[test]
    fn quant_precision_f32_is_plain() {
        assert_eq!(QuantPrecision::F32.kind, DTypeKind::F32);
        assert_eq!(QuantPrecision::F32.packing, PackingFormat::Plain);
        assert_eq!(QuantPrecision::F32.elem_bytes(), 4);
        assert!(!QuantPrecision::F32.is_packed());
    }

    #[test]
    fn quant_precision_bf16_elem_bytes_and_strategy() {
        assert_eq!(QuantPrecision::BF16.elem_bytes(), 2);
        assert_eq!(QuantPrecision::BF16.x86_elem_strategy(), X86ElemStrategy::WidenCompute);
    }

    #[test]
    fn promote_same_returns_identity() {
        let a = QuantPrecision::F32;
        let b = QuantPrecision::F32;
        assert_eq!(promote(a, b), QuantPrecision::F32);
    }

    #[test]
    fn promote_bf16_f16_yields_f32() {
        let result = promote(QuantPrecision::BF16, QuantPrecision::F16);
        assert_eq!(result, QuantPrecision::F32);
    }

    #[test]
    fn trace_op_map_neg_preserves_variant() {
        let op = TraceOp::Neg(ValueId(5));
        let mapped = op.map_value_ids(&|id| ValueId(id.0 + 100));
        assert_eq!(mapped, TraceOp::Neg(ValueId(105)));
    }

    #[test]
    fn trace_op_map_fma_offsets_all_three_ids() {
        let op = TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2));
        let mapped = op.map_value_ids(&|id| ValueId(id.0 + 20));
        assert_eq!(mapped, TraceOp::Fma(ValueId(20), ValueId(21), ValueId(22)));
    }

    #[test]
    fn ggml_type_q4_k_is_int4_and_256_block() {
        assert_eq!(GgmlType::Q4_K_S.dtype_kind(), DTypeKind::INT4);
        assert_eq!(GgmlType::Q4_K_S.block_size(), 256);
        assert_eq!(GgmlType::Q4_K_M.dtype_kind(), DTypeKind::INT4);
        assert_eq!(GgmlType::Q4_K_M.block_size(), 256);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Wave 12k59: +10 edge-case tests
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn op_trace_empty_body_constructs() {
        let trace = OpTrace {
            pattern: ComputePattern::Injective {
                body: vec![],
                num_inputs: 0,
                num_outputs: 1,
            },
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr],
            },
        };
        assert!(matches!(trace.pattern, ComputePattern::Injective { num_inputs: 0, .. }));
        assert_eq!(trace.signature.params.len(), 2);
    }

    #[test]
    fn classify_const_only_trace_is_elementwise() {
        let body = vec![
            TraceOp::Const(42.0),
            TraceOp::Const(1.0),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];
        let pattern = classify_pattern(&body);
        assert!(matches!(pattern, ComputePattern::Elementwise { .. }));
    }

    #[test]
    fn single_op_trace_fma_ssa_valid() {
        let body = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Input(2),
            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),
        ];
        for (i, op) in body.iter().enumerate() {
            if let TraceOp::Fma(a, b, c) = op {
                assert!(a.0 < i as u32);
                assert!(b.0 < i as u32);
                assert!(c.0 < i as u32);
            }
        }
    }

    #[test]
    fn compute_pattern_quant_decode_body_is_some() {
        let pattern = ComputePattern::QuantDecode {
            block_size: 32,
            decode: vec![TraceOp::Input(0), TraceOp::Mul(ValueId(0), ValueId(0))],
        };
        assert!(pattern.body().is_some());
        assert_eq!(pattern.body().unwrap().len(), 2);
    }

    #[test]
    fn infer_result_dtype_const_returns_f32() {
        let slots: Vec<TypedSlot> = vec![];
        let op = TraceOp::Const(3.14);
        let dtype = infer_result_dtype(&op, &slots);
        assert_eq!(dtype, QuantPrecision::F32);
    }

    #[test]
    fn promote_different_packings_yields_f32() {
        let a = QuantPrecision { kind: DTypeKind::F32, packing: PackingFormat::Plain, block_size: 0, group_size: 0 };
        let b = QuantPrecision { kind: DTypeKind::F32, packing: PackingFormat::MXBlock, block_size: 32, group_size: 0 };
        assert_eq!(promote(a, b), QuantPrecision::F32);
    }

    #[test]
    fn quant_precision_from_elem_bytes_unknown_fallback() {
        let result = QuantPrecision::from_elem_bytes(8);
        assert_eq!(result, QuantPrecision::F32);
    }

    #[test]
    fn trace_op_map_conditional_branch_offsets_ids() {
        let op = TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2));
        let mapped = op.map_value_ids(&|id| ValueId(id.0 + 5));
        assert_eq!(mapped, TraceOp::ConditionalBranch(ValueId(5), ValueId(6), ValueId(7)));
    }

    #[test]
    fn build_quant_gather_trace_structure() {
        let trace = build_quant_gather_trace(QuantType::Q4_0, 1000, 512);
        assert_eq!(trace.len(), 4);
        assert!(matches!(trace[0], TraceOp::Input(0)));
        assert!(matches!(trace[1], TraceOp::Input(1)));
        assert!(matches!(trace[2], TraceOp::Input(2)));
        assert!(matches!(trace[3], TraceOp::QuantGather { .. }));
    }

    #[test]
    fn build_quant_gemm_trace_structure() {
        let trace = build_quant_gemm_trace(QuantType::Q4_0, 1, 512, 4096);
        assert_eq!(trace.len(), 4);
        assert!(matches!(trace[0], TraceOp::Input(0)));
        assert!(matches!(trace[1], TraceOp::Input(1)));
        assert!(matches!(trace[2], TraceOp::Input(2)));
        assert!(matches!(trace[3], TraceOp::QuantGemm { .. }));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Wave 12k60: +10 additional tests
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn trace_op_equality_same_variant() {
        // Arrange: create two identical TraceOp::Add instances
        let op1 = TraceOp::Add(ValueId(3), ValueId(7));
        let op2 = TraceOp::Add(ValueId(3), ValueId(7));
        // Act & Assert: PartialEq should confirm equality
        assert_eq!(op1, op2);
    }

    #[test]
    fn trace_op_equality_different_const_values() {
        // Arrange: two Const ops with different values
        let op1 = TraceOp::Const(1.0);
        let op2 = TraceOp::Const(2.0);
        // Act & Assert: different constants must not be equal
        assert_ne!(op1, op2);
    }

    #[test]
    fn trace_op_equality_different_input_indices() {
        // Arrange: Input(0) vs Input(1)
        let op0 = TraceOp::Input(0);
        let op1 = TraceOp::Input(1);
        // Act & Assert: different input indices are not equal
        assert_ne!(op0, op1);
    }

    #[test]
    fn reduction_second_pass_fields_match() {
        // Arrange: construct a ReductionSecondPass with specific identity and ops
        let second_pass = ReductionSecondPass {
            identity: 0.0,
            element_transform: vec![TraceOp::Exp(ValueId(0))],
            combine: vec![TraceOp::Add(ValueId(0), ValueId(1))],
        };
        // Act & Assert: identity is 0.0 (sum), transform has 1 op, combine has 1 op
        assert_eq!(second_pass.identity, 0.0);
        assert_eq!(second_pass.element_transform.len(), 1);
        assert_eq!(second_pass.combine.len(), 1);
    }

    #[test]
    fn compute_pattern_reduction_with_second_pass_and_normalize() {
        // Arrange: full Reduction pattern with second_pass (softmax-like)
        let pattern = ComputePattern::Reduction {
            identity: f64::NEG_INFINITY,
            combine: vec![TraceOp::Max(ValueId(0), ValueId(1))],
            second_pass: Some(Box::new(ReductionSecondPass {
                identity: 0.0,
                element_transform: vec![TraceOp::Sub(ValueId(0), ValueId(1)), TraceOp::Exp(ValueId(2))],
                combine: vec![TraceOp::Add(ValueId(0), ValueId(1))],
            })),
            normalize: Some(vec![TraceOp::Div(ValueId(0), ValueId(1))]),
        };
        // Act & Assert: Reduction pattern body() returns None (multi-phase pattern)
        assert!(pattern.body().is_none());
    }

    #[test]
    fn op_trace_pattern_and_signature_correspond() {
        // Arrange: construct OpTrace with pattern and signature
        let body = vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Mul(ValueId(0), ValueId(1))];
        let trace = OpTrace {
            pattern: ComputePattern::BinaryElementwise { body: body.clone() },
            signature: ScalarFnSignature {
                fn_ptr: std::ptr::null(),
                params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(4)],
            },
        };
        // Act & Assert: pattern has 3 ops, signature has 3 params
        assert_eq!(trace.pattern.body().unwrap().len(), 3);
        assert_eq!(trace.signature.params.len(), 3);
    }

    #[test]
    fn typed_slot_dtype_preserved() {
        // Arrange: construct TypedSlot with BF16 dtype
        use crate::compiler::codegen::vm::instr::VRegId;
        let slot = TypedSlot { vreg: VRegId(5), dtype: QuantPrecision::BF16 };
        // Act & Assert: dtype field is BF16, elem_bytes is 2
        assert_eq!(slot.dtype, QuantPrecision::BF16);
        assert_eq!(slot.dtype.elem_bytes(), 2);
    }

    #[test]
    fn infer_result_dtype_cast_returns_target() {
        // Arrange: Cast from BF16 to F32 with empty slot list
        let op = TraceOp::Cast { src: ValueId(0), from: QuantPrecision::BF16, to: QuantPrecision::F32 };
        let slots: Vec<TypedSlot> = vec![];
        // Act: infer result dtype from Cast
        let result = infer_result_dtype(&op, &slots);
        // Assert: Cast result dtype is the target precision (F32)
        assert_eq!(result, QuantPrecision::F32);
    }

    #[test]
    fn infer_result_dtype_add_promotes_types() {
        // Arrange: Add with BF16 slot at index 0 and F32 slot at index 1
        use crate::compiler::codegen::vm::instr::VRegId;
        let slots = vec![
            TypedSlot { vreg: VRegId(0), dtype: QuantPrecision::BF16 },
            TypedSlot { vreg: VRegId(1), dtype: QuantPrecision::F32 },
        ];
        let op = TraceOp::Add(ValueId(0), ValueId(1));
        // Act: infer result dtype for Add(BF16, F32)
        let result = infer_result_dtype(&op, &slots);
        // Assert: promote(BF16, F32) should yield F32
        assert_eq!(result, QuantPrecision::F32);
    }

    #[test]
    fn classify_pattern_no_input_ops_is_elementwise() {
        // Arrange: body with only Const and unary ops (no Input ops)
        let body = vec![
            TraceOp::Const(1.0),
            TraceOp::Const(2.0),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];
        // Act: classify this body
        let pattern = classify_pattern(&body);
        // Assert: no Input ops means max_input is None, num_inputs=0, classified as Elementwise
        match pattern {
            ComputePattern::Elementwise { body: b } => assert_eq!(b.len(), 3),
            other => panic!("expected Elementwise, got {:?}", other),
        }
    }

    // ── REQ-DTYPE-005: gpu_accumulator_dtype tests ──

    /// @trace TEST-DTYPE-005-1 [req:REQ-DTYPE-005] [level:unit]
    /// GPU: BF16 → F32 tensor core accumulation
    #[test]
    fn gpu_accumulator_dtype_bf16_is_f32() {
        assert_eq!(QuantPrecision::BF16.gpu_accumulator_dtype(), QuantPrecision::F32);
    }

    /// @trace TEST-DTYPE-005-2 [req:REQ-DTYPE-005] [level:unit]
    /// GPU: F16 → F32 accumulation (HMMA.16816.F32)
    #[test]
    fn gpu_accumulator_dtype_f16_is_f32() {
        assert_eq!(QuantPrecision::F16.gpu_accumulator_dtype(), QuantPrecision::F32);
    }

    /// @trace TEST-DTYPE-005-3 [req:REQ-DTYPE-005] [level:unit]
    /// GPU: F32 → F32 accumulation (identity)
    #[test]
    fn gpu_accumulator_dtype_f32_is_f32() {
        assert_eq!(QuantPrecision::F32.gpu_accumulator_dtype(), QuantPrecision::F32);
    }

    /// @trace TEST-DTYPE-005-4 [req:REQ-DTYPE-005] [level:unit]
    /// GPU: x86 accumulator_dtype for BF16 is F32 (WidenCompute)
    #[test]
    fn x86_accumulator_dtype_bf16_is_f32() {
        assert_eq!(QuantPrecision::BF16.accumulator_dtype(), QuantPrecision::F32);
    }

    /// @trace TEST-DTYPE-005-5 [req:REQ-DTYPE-005] [level:unit]
    /// GPU: x86 accumulator_dtype for F32 is F32 (Native)
    #[test]
    fn x86_accumulator_dtype_f32_is_f32() {
        assert_eq!(QuantPrecision::F32.accumulator_dtype(), QuantPrecision::F32);
    }

    // ── REQ-DTYPE-006: needs_narrowing_from tests ──

    /// @trace TEST-DTYPE-006-1 [req:REQ-DTYPE-006] [level:unit]
    /// F32→BF16 narrowing: needs_narrowing_from returns true
    #[test]
    fn needs_narrowing_f32_to_bf16() {
        assert!(QuantPrecision::BF16.needs_narrowing_from(QuantPrecision::F32));
    }

    /// @trace TEST-DTYPE-006-2 [req:REQ-DTYPE-006] [level:unit]
    /// F32→F32: no narrowing needed
    #[test]
    fn needs_no_narrowing_f32_to_f32() {
        assert!(!QuantPrecision::F32.needs_narrowing_from(QuantPrecision::F32));
    }

    /// @trace TEST-DTYPE-006-3 [req:REQ-DTYPE-006] [level:unit]
    /// BF16→BF16: no narrowing needed (same dtype)
    #[test]
    fn needs_no_narrowing_bf16_to_bf16() {
        assert!(!QuantPrecision::BF16.needs_narrowing_from(QuantPrecision::BF16));
    }

    /// @trace TEST-DTYPE-006-4 [req:REQ-DTYPE-006] [level:unit]
    /// BF16 GEMM accumulator chain: BF16 → accumulator_dtype() = F32 → needs narrowing
    #[test]
    fn bf16_gemm_accumulator_needs_narrowing() {
        let acc = QuantPrecision::BF16.accumulator_dtype();
        assert!(QuantPrecision::BF16.needs_narrowing_from(acc));
    }

    /// @trace TEST-DTYPE-006-5 [req:REQ-DTYPE-006] [level:unit]
    /// F32 GEMM accumulator chain: F32 → accumulator_dtype() = F32 → no narrowing
    #[test]
    fn f32_gemm_accumulator_no_narrowing() {
        let acc = QuantPrecision::F32.accumulator_dtype();
        assert!(!QuantPrecision::F32.needs_narrowing_from(acc));
    }

    /// @trace TEST-DTYPE-006-6 [req:REQ-DTYPE-006] [level:unit]
    /// GPU BF16 GEMM: gpu_accumulator_dtype() = F32 → needs narrowing
    #[test]
    fn gpu_bf16_gemm_accumulator_needs_narrowing() {
        let acc = QuantPrecision::BF16.gpu_accumulator_dtype();
        assert!(QuantPrecision::BF16.needs_narrowing_from(acc));
    }
}
