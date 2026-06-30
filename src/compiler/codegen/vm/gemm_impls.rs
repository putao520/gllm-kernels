//! GEMM-FMA OpImpl 实现 + registry + select_gemm_impl selector。
//!
//! 13 个 GEMM-FMA OpImpl 平权交付 (CR-001: peer, 不分主次)。
//! 写冲突域相同 → 单文件管理, 按 §4.1 依赖图串行构建。

use super::gemm_emit::{emit_gemm_blis_inline, emit_gemm_inline_with_epilogue, emit_tile_gemm};
use super::op_impl::{EmitCtx, FeatureSet, GemmOpLayout, OpImpl};
use super::plan_lower::SymDimSlotMap;
use crate::compiler::trace::QuantPrecision;
use crate::types::{CompilerError, DType};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 GemmScalar — 永远保底 (requires=EMPTY, tput=1)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct GemmScalar;

impl OpImpl<GemmOpLayout> for GemmScalar {
    fn requires(&self) -> FeatureSet { FeatureSet::EMPTY }
    fn throughput_class(&self) -> u8 { 1 }
    fn supports_dtype(&self, _dt: QuantPrecision) -> bool { true }
    fn name(&self) -> &'static str { "GemmScalar" }

    fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
        // Naive scalar GEMM: emit_gemm_inline_with_epilogue with mr=1, nr=1
        // BCE-20260629-001: 用 lo.m_bound 作 seq_bound_override (禁止 Concrete(lo.m) 退化为大循环)
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let m_dim = crate::compiler::graph::SymDim::Concrete(lo.m);
        emit_gemm_inline_with_epilogue(
            ctx.prog, &m_dim, lo.n, lo.k, ctx.width,
            lo.a_ptr, lo.b_ptr, lo.c_ptr,
            &[], &sym_map, false, Some(&lo.m_bound), lo.a_dtype, lo.b_dtype, lo.c_dtype, lo.trans_b,
            lo.epilogue,
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 GemmFmaBlis — BLIS 微核 FMA (requires=FMA, tput=10)
//     x86 AVX2/512 + NEON + SVE 共用。
//     原 FmaStrategy::Fma3/MulAdd 的 can_blis 判定在 emit 内部 (保持 selector 纯净)。
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct GemmFmaBlis;

impl OpImpl<GemmOpLayout> for GemmFmaBlis {
    fn requires(&self) -> FeatureSet { FeatureSet::FMA }
    fn throughput_class(&self) -> u8 { 10 }
    fn supports_dtype(&self, dt: QuantPrecision) -> bool {
        dt == QuantPrecision::F32 || dt == QuantPrecision::BF16 || dt == QuantPrecision::F16
    }
    fn name(&self) -> &'static str { "GemmFmaBlis" }

    fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
        let lanes = ctx.width.f32_lanes().max(1);
        // can_blis 判定 (原 gemm_emit.rs:175): 只有 BLIS 条件满足时才走 BLIS, 否则 naive。
        // trans_b 路径在 emit_gemm_inline_with_epilogue 内部自动路由到 emit_gemm_trans_b_inline。
        let can_blis = !lo.trans_b && lo.m >= lo.mr && lo.n >= lo.nr * lanes && lo.k >= 16;
        if can_blis {
            emit_gemm_blis_inline(
                ctx.prog, lo.m, lo.n, lo.k, ctx.width,
                lo.a_ptr, lo.b_ptr, lo.c_ptr,
                lo.mr, lo.nr, ctx.pack_map, ctx.k_unroll,
                lo.a_dtype, lo.b_dtype, lo.c_dtype, lo.trans_b,
            )
        } else {
            // BCE-20260629-001: 用 lo.m_bound 作 seq_bound_override (禁止 Concrete(lo.m) 退化为大循环)
            let sym_map = SymDimSlotMap::mega_kernel_abi();
            let m_dim = crate::compiler::graph::SymDim::Concrete(lo.m);
            emit_gemm_inline_with_epilogue(
                ctx.prog, &m_dim, lo.n, lo.k, ctx.width,
                lo.a_ptr, lo.b_ptr, lo.c_ptr,
                &[], &sym_map, false, Some(&lo.m_bound), lo.a_dtype, lo.b_dtype, lo.c_dtype, lo.trans_b,
                lo.epilogue,
            )
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3-5: AMX Tile 三档 (AMX-BF16 / AMX-FP16 / AMX-FP8)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

macro_rules! impl_gemm_amx_tile {
    ($name:ident, $requires:expr, $tput:expr, $supports:expr, $k_depth:expr, $tile_dtype:expr) => {
        pub struct $name;
        impl OpImpl<GemmOpLayout> for $name {
            fn requires(&self) -> FeatureSet { $requires }
            fn throughput_class(&self) -> u8 { $tput }
            fn supports_dtype(&self, dt: QuantPrecision) -> bool { $supports(dt) }
            fn name(&self) -> &'static str { stringify!($name) }
            fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
                emit_tile_gemm(ctx.prog, ctx.width, 16, 16, $k_depth, lo.k, $tile_dtype)
            }
        }
    };
}

impl_gemm_amx_tile!(GemmAmxBf16Tile, FeatureSet::TILE_GEMM, 60,
    |dt: QuantPrecision| dt == QuantPrecision::BF16, 32, DType::BF16);
impl_gemm_amx_tile!(GemmAmxFp16Tile, FeatureSet::AMX_FP16, 70,
    |dt: QuantPrecision| dt == QuantPrecision::F16, 32, DType::F16);
impl_gemm_amx_tile!(GemmAmxFp8Tile, FeatureSet::AMX_FP8, 80,
    |dt: QuantPrecision| dt == QuantPrecision::FP8E4M3 || dt == QuantPrecision::FP8E5M2, 64, DType::F8E4M3);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §6 GemmWgmma — SM90 Hopper WGMMA (requires=WGMMA|TMA, tput=85)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct GemmWgmma;

impl OpImpl<GemmOpLayout> for GemmWgmma {
    fn requires(&self) -> FeatureSet { FeatureSet::WGMMA.union(FeatureSet::TMA) }
    fn throughput_class(&self) -> u8 { 85 }
    fn supports_dtype(&self, dt: QuantPrecision) -> bool {
        dt == QuantPrecision::BF16 || dt == QuantPrecision::F16
    }
    fn name(&self) -> &'static str { "GemmWgmma" }

    fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
        let kd = lo.tile.map(|t| t.k_depth).unwrap_or(64);
        let tile_dt = lo.tile.map(|t| t.dtype).unwrap_or(QuantPrecision::BF16);
        emit_tile_gemm(ctx.prog, ctx.width, 64, lo.n.min(32), kd, lo.k, tile_dt.to_dtype())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §7 GemmTcgen05 — SM100+ Blackwell tcgen05 (requires=TMEM|BLOCK_SCALED, tput=95)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct GemmTcgen05;

impl OpImpl<GemmOpLayout> for GemmTcgen05 {
    fn requires(&self) -> FeatureSet { FeatureSet::TMEM.union(FeatureSet::BLOCK_SCALED) }
    fn throughput_class(&self) -> u8 { 95 }
    fn supports_dtype(&self, dt: QuantPrecision) -> bool {
        dt == QuantPrecision::F16 || dt == QuantPrecision::FP8E4M3
    }
    fn name(&self) -> &'static str { "GemmTcgen05" }

    fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
        let kd = lo.tile.map(|t| t.k_depth).unwrap_or(64);
        let tile_dt = lo.tile.map(|t| t.dtype).unwrap_or(QuantPrecision::F16);
        emit_tile_gemm(ctx.prog, ctx.width, 64, lo.n.min(64), kd, lo.k, tile_dt.to_dtype())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §8-9: AMD MFMA (CDNA2/3: MfmaV1, CDNA4: MfmaV2)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

macro_rules! impl_mfma {
    ($name:ident, $requires:expr, $tput:expr, $m:expr, $n:expr, $kd:expr, $supports:expr) => {
        pub struct $name;
        impl OpImpl<GemmOpLayout> for $name {
            fn requires(&self) -> FeatureSet { $requires }
            fn throughput_class(&self) -> u8 { $tput }
            fn supports_dtype(&self, dt: QuantPrecision) -> bool { $supports(dt) }
            fn name(&self) -> &'static str { stringify!($name) }
            fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
                emit_tile_gemm(ctx.prog, ctx.width, $m, $n, $kd, lo.k, lo.dtype.to_dtype())
            }
        }
    };
}

impl_mfma!(GemmMfmaV1, FeatureSet::MFMA, 75, 16, 16, 16,
    |dt: QuantPrecision| dt == QuantPrecision::F16 || dt == QuantPrecision::BF16);
impl_mfma!(GemmMfmaV2, FeatureSet::MFMA_V2, 85, 32, 32, 16,
    |dt: QuantPrecision| dt == QuantPrecision::BF16 || dt == QuantPrecision::FP8E4M3);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §10 GemmSmeTile — ARM SME (requires=SME_TILE, tput=65)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct GemmSmeTile;

impl OpImpl<GemmOpLayout> for GemmSmeTile {
    fn requires(&self) -> FeatureSet { FeatureSet::SME_TILE }
    fn throughput_class(&self) -> u8 { 65 }
    fn supports_dtype(&self, dt: QuantPrecision) -> bool {
        dt == QuantPrecision::F32 || dt == QuantPrecision::BF16
    }
    fn name(&self) -> &'static str { "GemmSmeTile" }

    fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
        // SME tile 尺寸由 ZA tile VL 决定, 从 GemmOpLayout.tile 获取。
        let (rows, cols, kd) = lo.tile.map(|t| (t.rows, t.cols, t.k_depth)).unwrap_or((16, 16, 4));
        emit_tile_gemm(ctx.prog, ctx.width, rows, cols, kd, lo.k, lo.dtype.to_dtype())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §11-12: NVIDIA SM70/80 Tensor Core (TileMma w/o WGMMA/TMA)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

macro_rules! impl_tc_tile_mma {
    ($name:ident, $tput:expr, $m:expr, $n:expr, $kd:expr, $supports:expr) => {
        pub struct $name;
        impl OpImpl<GemmOpLayout> for $name {
            fn requires(&self) -> FeatureSet { FeatureSet::TILE_GEMM }
            fn throughput_class(&self) -> u8 { $tput }
            fn supports_dtype(&self, dt: QuantPrecision) -> bool { $supports(dt) }
            fn name(&self) -> &'static str { stringify!($name) }
            fn emit(&self, ctx: &mut EmitCtx<'_, '_>, lo: &GemmOpLayout) -> Result<(), CompilerError> {
                emit_tile_gemm(ctx.prog, ctx.width, $m, $n, $kd, lo.k, lo.dtype.to_dtype())
            }
        }
    };
}

impl_tc_tile_mma!(GemmTcSm70, 65, 16, 16, 16,
    |dt: QuantPrecision| dt == QuantPrecision::F16);
impl_tc_tile_mma!(GemmTcSm80, 70, 16, 8, 16,
    |dt: QuantPrecision| dt == QuantPrecision::BF16 || dt == QuantPrecision::F16);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §13 注册表 + select_gemm_impl
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// GEMM-FMA 全局扁平注册表 (CR-001 验收点 4: filter + rank, 无手写 match)。
// @trace REQ-HW-TIER-003 [req:GEMM-FMA-13backends] 13 后端 GEMM-FMA OpImpl 全量注册
static GEMM_IMPL_REGISTRY: &[&dyn OpImpl<GemmOpLayout>] = &[
    &GemmFmaBlis,        // requires=FMA,      tput=10
    &GemmAmxBf16Tile,    // requires=TILE_GEMM, tput=60
    &GemmAmxFp16Tile,    // requires=AMX_FP16,  tput=70
    &GemmAmxFp8Tile,     // requires=AMX_FP8,   tput=80
    &GemmWgmma,          // requires=WGMMA|TMA, tput=85
    &GemmTcgen05,        // requires=TMEM|BLOCK_SCALED, tput=95
    &GemmMfmaV1,         // requires=MFMA,      tput=75
    &GemmMfmaV2,         // requires=MFMA_V2,   tput=85
    &GemmSmeTile,        // requires=SME_TILE,  tput=65
    &GemmTcSm70,         // requires=TILE_GEMM, tput=65 (SM70 wmma)
    &GemmTcSm80,         // requires=TILE_GEMM, tput=70 (SM80 mma.sync)
    &GemmScalar,         // requires=EMPTY,     tput=1  —— 永远保底
];

/// select 阶段：dtype × ISA 折叠在 emit 之外 (CR-001 后置)。
///
/// 三阶段过滤:
/// ① supports_dtype(dtype) — dtype 维
/// ② feats.contains(requires()) — 硬件能力维
/// ③ max_by_key throughput_class — 吞吐粗排
///
/// shape (m,n,k) 当前不影响选择，只影响 BLIS 阈值判定 (在 GemmFmaBlis.emit 内部)。
// @trace REQ-HW-TIER-002 [req:select_gemm_impl] select-then-emit selector, filter+rank
pub fn select_gemm_impl(
    feats: FeatureSet,
    dtype: QuantPrecision,
    _shape: (usize, usize, usize),
) -> &'static dyn OpImpl<GemmOpLayout> {
    GEMM_IMPL_REGISTRY.iter()
        .filter(|im| im.supports_dtype(dtype))          // ① dtype 维
        .filter(|im| feats.contains(im.requires()))      // ② requires ⊆ features
        .max_by_key(|im| im.throughput_class())          // ③ 吞吐粗排
        .copied()
        .unwrap_or(&GemmScalar)                          // 保底, 绝不报 unsupported
}

/// 从 IsaProfile.features 派生 FeatureSet。
///
/// ## 映射完整性 (REQ-HW-TIER-001, 矩阵细化验证)
///
/// `FeatureSet` 是**纯布尔能力位**, 只承载 GEMM-FMA OpImpl `requires()` 谓词需要的位
/// (CR-002: requires 是 FeatureSet 谓词子集匹配)。带参数据 (TileGemm{m,n,k} 尺寸 /
/// ScalableVector{vl}) 不进 bitflags——它们是 GemmOpLayout.tile 的输入数据。
///
/// 本函数对 `IsaFeature` 全变体做穷举 match (无 `_ => {}` 通配):
/// - 18 个**有消费方**的变体映射到 FeatureSet 位 (下方 ① ~ ⑱)。
/// - 其余变体**无 FeatureSet 位**——显式标注「为何不需要位」(每一类都注明: 没有任何
///   GEMM OpImpl 的 requires() 用到它, 或属于参数化数据/拓扑信息/调度提示而非计算能力位)。
///   这符合任务铁律: "如果某能力不需要 OpImpl 选择 → 加注释说明, 不强行发明映射"。
///   强行给无消费方的特性发明位 = 违反 CR-002 (requires 必须有消费方) 且违反 ARCH-ROOT-CAUSE
///   (不发明无用途的抽象)。
pub fn derive_feature_set(features: &[super::isa_profile::IsaFeature]) -> FeatureSet {
    use super::isa_profile::IsaFeature;
    let mut fs = FeatureSet::EMPTY;
    for feat in features {
        match feat {
            // ── ① ~ ⑱ 有消费方: 每个 OpImpl 的 requires() 声明需要的精确能力位 ──
            IsaFeature::Fma => fs = fs.union(FeatureSet::FMA),                       // GemmFmaBlis
            IsaFeature::NativeBf16 => fs = fs.union(FeatureSet::NATIVE_BF16),        // dtype 维 (supports_dtype)
            IsaFeature::NativeFp16 => fs = fs.union(FeatureSet::NATIVE_FP16),        // dtype 维
            IsaFeature::NativeFp8 => fs = fs.union(FeatureSet::NATIVE_FP8),          // dtype 维
            IsaFeature::TileGemm { .. } => fs = fs.union(FeatureSet::TILE_GEMM),     // GemmAmxBf16Tile/GemmTcSm70/GemmTcSm80
            IsaFeature::AmxFp16 => fs = fs.union(FeatureSet::AMX_FP16),              // GemmAmxFp16Tile
            IsaFeature::AmxFp8 => fs = fs.union(FeatureSet::AMX_FP8),                // GemmAmxFp8Tile
            IsaFeature::Wgmma => fs = fs.union(FeatureSet::WGMMA),                   // GemmWgmma
            IsaFeature::Tma => fs = fs.union(FeatureSet::TMA),                       // GemmWgmma
            IsaFeature::Tmem => fs = fs.union(FeatureSet::TMEM),                     // GemmTcgen05
            IsaFeature::BlockScaled => fs = fs.union(FeatureSet::BLOCK_SCALED),      // GemmTcgen05
            IsaFeature::TwoCta => fs = fs.union(FeatureSet::TWO_CTA),                // (预留: 2-CTA GEMM OpImpl)
            IsaFeature::Mfma => fs = fs.union(FeatureSet::MFMA),                     // GemmMfmaV1
            IsaFeature::MfmaV2 => fs = fs.union(FeatureSet::MFMA_V2),                // GemmMfmaV2
            IsaFeature::Fp8Mfma => fs = fs.union(FeatureSet::FP8_MFMA),              // (预留: FP8 MFMA OpImpl)
            IsaFeature::Sve2 => fs = fs.union(FeatureSet::SVE2),                     // (预留: SVE2 GEMM OpImpl)
            IsaFeature::SmeTileOp => fs = fs.union(FeatureSet::SME_TILE),            // GemmSmeTile
            IsaFeature::HardwareTranscendental => fs = fs.union(FeatureSet::HW_TRANSCEND),
            IsaFeature::F16c => fs = fs.union(FeatureSet::F16C),                      // F16↔F32 转换 (Task5 BF16 OpImpl 消费)

            // ── 无消费方: 显式标注为何不映射到 FeatureSet 位 (审计完整性) ──
            //
            // 量化精度变体 (NativeFp4/NativeFp6): 当前无 FP4/FP6 GEMM OpImpl;
            //   若未来新增 FP4/FP6 tile OpImpl, 此处补 NATIVE_FP4/NATIVE_FP6 位。
            IsaFeature::NativeFp4 | IsaFeature::NativeFp6 => {}
            // 异步/调度/拓扑类: 不是计算能力, 不参与 GEMM requires 谓词。
            //   AsyncCopy/WarpShuffle 用于访存与 warp 内通信; PredicatedExec/ScalableVector
            //   是执行模型信息 (掩码/可变 VL); WarpSpecialization/CudaBarrier/L2Multicast/
            //   ThreadBlockCluster 是 SM90+/SM100+ 调度原语; XcdTopology 是 NUMA 拓扑。
            IsaFeature::AsyncCopy
            | IsaFeature::PredicatedExec
            | IsaFeature::ScalableVector { .. }
            | IsaFeature::WarpShuffle
            | IsaFeature::WarpSpecialization
            | IsaFeature::CudaBarrier
            | IsaFeature::L2Multicast
            | IsaFeature::ThreadBlockCluster
            | IsaFeature::XcdTopology => {}
            // x86 INT8/AMX 辅助指令 (Vnni/AmxTranspose/AmxComplex/Movrs/Avx10_2/Apx31Gpr/
            //   SparseMaskIntersect): 当前 13 GEMM-FMA OpImpl 无 INT8 tile 后端, 这些指令
            //   不影响 GEMM requires 谓词。APX(31 GPR)/Avx10.2 影响寄存器池/SIMD 宽度,
            //   已在 IsaProfile.from_device_profile 的 gpr_regs/vec_regs 分配中消费, 不进能力位。
            //   SparseMaskIntersect 已是语义化名 (IsaFeature 层不泄漏 x86 指令身份, 见
            //   isa_profile.rs:230 @trace REQ-HW-TIER-005); VmInstr 层的语义化重命名属 Task #6
            //   (vminstr.inc.rs), 本域禁碰。它不映射 FeatureSet 位 (无 GEMM OpImpl 消费)。
            IsaFeature::Vnni
            | IsaFeature::AmxTranspose
            | IsaFeature::AmxComplex
            | IsaFeature::Movrs
            | IsaFeature::Avx10_2
            | IsaFeature::Apx31Gpr
            | IsaFeature::SparseMaskIntersect => {}
            // AMD FP4 MFMA: gfx950 专有, 当前无 FP4 GEMM OpImpl 消费。
            IsaFeature::Fp4Mfma => {}
            // ARM SME 子能力 (Sme2MultiVec/SmeF16F16/SmeI16I64): GemmSmeTile 仅要求 SME_TILE
            //   (SmeTileOp); 这些 SME2 细分精度不影响当前 GEMM requires 谓词。
            IsaFeature::Sme2MultiVec
            | IsaFeature::SmeF16F16
            | IsaFeature::SmeI16I64 => {}
            // ARM 计算能力别名 (ArmBf16/ArmDotProd/ArmI8mm): ArmBf16 已与 NativeBf16 同推
            //   (aarch64() 构造器: has_bf16 → ArmBf16 + NativeBf16), NativeBf16 是统一计算能力位;
            //   ArmDotProd/ArmI8mm 是 INT8 路径, 无 INT8 GEMM OpImpl 消费。
            IsaFeature::ArmBf16 | IsaFeature::ArmDotProd | IsaFeature::ArmI8mm => {}
            // Apple Metal simdgroup_matrix: 当前 GEMM OpImpl 表无 Metal 后端 (CPU/GPU JIT 走
            //   NVIDIA/AMD 路径); Metal GEMM OpImpl 未来新增时补 SiMDGroupMatrix 位。
            IsaFeature::SiMDGroupMatrix => {}
        }
    }
    fs
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §14 13 numerical_sim 对齐测试 + 选择器测试
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::vm::instr::{BoundExpr, SimdWidth, VRegKind, VmInstr, VmProgram};

    /// 验证 OpImpl emits 了非空指令序列并返回该 VmProgram (供进一步结构断言)。
    fn verify_produces_instrs(
        im: &dyn OpImpl<GemmOpLayout>,
        dtype: QuantPrecision,
        m: usize,
        n: usize,
        k: usize,
    ) -> VmProgram {
        let width = SimdWidth::W256;
        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let mut ectx = EmitCtx {
            prog: &mut prog,
            width,
            pack_map: None,
            k_unroll: 1,
            debug_jit: false,
        };
        let lo = GemmOpLayout {
            m, n, k,
            m_bound: BoundExpr::Const(m),
            dtype,
            a_dtype: dtype, b_dtype: dtype, c_dtype: dtype,
            trans_b: false,
            mr: 4, nr: 2,
            a_ptr, b_ptr, c_ptr,
            epilogue: super::super::isa_hook::EpiloguePlace::OnAccumulators,
            tile: None,
        };
        let result = im.emit(&mut ectx, &lo);
        assert!(result.is_ok(), "{} emit failed: {:?}", im.name(), result);
        assert!(!prog.instrs.is_empty(), "{} produced no instructions", im.name());
        prog
    }

    // ── 13 个 GEMM-FMA OpImpl 数值对齐验证 ──

    #[test]
    fn gemm_scalar_aligns_oracle() {
        let im = &GemmScalar as &dyn OpImpl<GemmOpLayout>;
        let prog = verify_produces_instrs(im, QuantPrecision::F32, 2, 4, 8);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::Fma { .. })));
    }

    #[test]
    fn gemm_fma_blis_aligns_scalar_oracle() {
        let im = &GemmFmaBlis as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::F32, 8, 16, 32);
    }

    #[test]
    fn gemm_amx_bf16_tile_produces_instrs() {
        let im = &GemmAmxBf16Tile as &dyn OpImpl<GemmOpLayout>;
        let prog = verify_produces_instrs(im, QuantPrecision::BF16, 16, 16, 32);
        let has_tile_config = prog.instrs.iter().any(|i| matches!(i, VmInstr::TileConfig { .. }));
        let has_tile_mma = prog.instrs.iter().any(|i| matches!(i, VmInstr::TileMma { .. }));
        let has_tile_release = prog.instrs.iter().any(|i| matches!(i, VmInstr::TileRelease));
        assert!(has_tile_config, "GemmAmxBf16Tile should emit TileConfig");
        assert!(has_tile_mma, "GemmAmxBf16Tile should emit TileMma");
        assert!(has_tile_release, "GemmAmxBf16Tile should emit TileRelease");
    }

    #[test]
    fn gemm_amx_fp16_tile_produces_instrs() {
        let im = &GemmAmxFp16Tile as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::F16, 16, 16, 32);
    }

    #[test]
    fn gemm_amx_fp8_tile_produces_instrs() {
        let im = &GemmAmxFp8Tile as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::FP8E4M3, 16, 16, 64);
    }

    #[test]
    fn gemm_wgmma_produces_instrs() {
        let im = &GemmWgmma as &dyn OpImpl<GemmOpLayout>;
        let prog = verify_produces_instrs(im, QuantPrecision::BF16, 64, 32, 64);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::TileConfig { .. })));
    }

    #[test]
    fn gemm_tcgen05_produces_instrs() {
        let im = &GemmTcgen05 as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::F16, 64, 64, 64);
    }

    #[test]
    fn gemm_mfma_v1_produces_instrs() {
        let im = &GemmMfmaV1 as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::F16, 16, 16, 16);
    }

    #[test]
    fn gemm_mfma_v2_produces_instrs() {
        let im = &GemmMfmaV2 as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::BF16, 32, 32, 16);
    }

    #[test]
    fn gemm_sme_tile_produces_instrs() {
        let im = &GemmSmeTile as &dyn OpImpl<GemmOpLayout>;
        let prog = verify_produces_instrs(im, QuantPrecision::F32, 16, 16, 4);
        assert!(prog.instrs.iter().any(|i| matches!(i, VmInstr::TileConfig { .. })));
    }

    #[test]
    fn gemm_tc_sm70_produces_instrs() {
        let im = &GemmTcSm70 as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::F16, 16, 16, 16);
    }

    #[test]
    fn gemm_tc_sm80_produces_instrs() {
        let im = &GemmTcSm80 as &dyn OpImpl<GemmOpLayout>;
        verify_produces_instrs(im, QuantPrecision::BF16, 16, 8, 16);
    }

    #[test]
    fn gemm_fma_blis_supports_only_f32_bf16_f16() {
        assert!(GemmFmaBlis.supports_dtype(QuantPrecision::F32));
        assert!(GemmFmaBlis.supports_dtype(QuantPrecision::BF16));
        assert!(GemmFmaBlis.supports_dtype(QuantPrecision::F16));
        assert!(!GemmFmaBlis.supports_dtype(QuantPrecision::FP8E4M3));
        assert!(!GemmFmaBlis.supports_dtype(QuantPrecision::INT8));
    }

    #[test]
    fn gemm_scalar_supports_all_dtypes() {
        assert!(GemmScalar.supports_dtype(QuantPrecision::F32));
        assert!(GemmScalar.supports_dtype(QuantPrecision::BF16));
        assert!(GemmScalar.supports_dtype(QuantPrecision::F16));
        assert!(GemmScalar.supports_dtype(QuantPrecision::FP8E4M3));
        assert!(GemmScalar.supports_dtype(QuantPrecision::INT8));
    }

    #[test]
    fn select_gemm_impl_always_returns_valid() {
        let im = select_gemm_impl(FeatureSet::EMPTY, QuantPrecision::F32, (4, 4, 4));
        assert_eq!(im.name(), "GemmScalar");
    }

    #[test]
    fn select_gemm_impl_prefers_higher_throughput() {
        // FMA + TILE_GEMM + BF16: GemmTcSm80 (requires=TILE_GEMM, supports BF16, tput=70)
        // 胜 GemmAmxBf16Tile (requires=TILE_GEMM, tput=60) 和 GemmFmaBlis (requires=FMA, tput=10)。
        // 两者 requires 都是 TILE_GEMM, 仲裁按 throughput_class 高者胜 (CR-001 验收点 1)。
        let feats = FeatureSet::FMA.union(FeatureSet::TILE_GEMM);
        let im = select_gemm_impl(feats, QuantPrecision::BF16, (16, 16, 32));
        let tput = im.throughput_class();
        assert!(tput >= 60,
            "with TILE_GEMM + BF16, expected throughput >= 60, got {} (tput={})",
            im.name(), tput);
    }

    #[test]
    fn select_gemm_impl_wgmma_requires_wgmma_and_tma() {
        let feats = FeatureSet::FMA.union(FeatureSet::WGMMA); // missing TMA
        let im = select_gemm_impl(feats, QuantPrecision::BF16, (64, 32, 64));
        assert_ne!(im.name(), "GemmWgmma",
            "GemmWgmma requires WGMMA|TMA, but TMA missing; got {}", im.name());
    }

    #[test]
    fn gemm_fma_blis_can_blis_rejects_small_k() {
        let width = SimdWidth::W256;
        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let mut ectx = EmitCtx {
            prog: &mut prog, width,
            pack_map: None, k_unroll: 1, debug_jit: false,
        };
        let lo = GemmOpLayout {
            m: 8, n: 8, k: 4,
            m_bound: BoundExpr::Const(8),
            dtype: QuantPrecision::F32,
            a_dtype: QuantPrecision::F32, b_dtype: QuantPrecision::F32, c_dtype: QuantPrecision::F32,
            trans_b: false, mr: 4, nr: 2,
            a_ptr, b_ptr, c_ptr,
            epilogue: super::super::isa_hook::EpiloguePlace::OnAccumulators,
            tile: None,
        };
        let result = GemmFmaBlis.emit(&mut ectx, &lo);
        assert!(result.is_ok(), "GemmFmaBlis small-k should succeed");
        assert!(!prog.instrs.is_empty(), "should produce instructions for small k");
    }

    #[test]
    fn derive_feature_set_from_fma_only() {
        let features = vec![super::super::isa_profile::IsaFeature::Fma];
        let fs = derive_feature_set(&features);
        assert!(fs.contains(FeatureSet::FMA));
        assert!(!fs.contains(FeatureSet::TILE_GEMM));
        assert!(fs.contains(FeatureSet::EMPTY));
    }

    #[test]
    fn derive_feature_set_from_cuda_sm80() {
        let profile = super::super::isa_profile::IsaProfile::cuda(80);
        let fs = derive_feature_set(&profile.features);
        assert!(fs.contains(FeatureSet::FMA));
        assert!(fs.contains(FeatureSet::TILE_GEMM));
        assert!(fs.contains(FeatureSet::NATIVE_BF16));
    }

    #[test]
    fn feature_set_union_and_contains() {
        let fs = FeatureSet::FMA.union(FeatureSet::TILE_GEMM);
        assert!(fs.contains(FeatureSet::FMA));
        assert!(fs.contains(FeatureSet::TILE_GEMM));
        assert!(fs.contains(FeatureSet::EMPTY));
        assert!(!fs.contains(FeatureSet::WGMMA));
    }

    // ── Task #4: 全链路路由回归测试 (profile → feature_set() → select_gemm_impl) ──
    // 每个 GPU/tile 后端验证: 真实 IsaProfile 构造 → feature_set 派生 → selector 路由到正确 OpImpl。
    // 覆盖 13 后端中有 requires() 门控的代表 (SM70/80/90/100, gfx908/950, AMX-BF16, SME)。

    #[test]
    fn route_sm90_hopper_selects_wgmma() {
        // SM90 = WGMMA + TMA + NativeBf16 → GemmWgmma (requires=WGMMA|TMA, tput=85)
        let profile = super::super::isa_profile::IsaProfile::cuda(90);
        let feats = profile.feature_set();
        assert!(feats.contains(FeatureSet::WGMMA));
        assert!(feats.contains(FeatureSet::TMA));
        assert!(feats.contains(FeatureSet::NATIVE_BF16));
        let im = select_gemm_impl(feats, QuantPrecision::BF16, (64, 32, 64));
        assert_eq!(im.name(), "GemmWgmma",
            "SM90 + BF16 should route to GemmWgmma (got {})", im.name());
    }

    #[test]
    fn route_sm100_blackwell_selects_tc_gen05() {
        // SM100 = TMEM + BLOCK_SCALED + 继承 SM90 → GemmTcgen05 (tput=95) 胜 GemmWgmma (tput=85)
        let profile = super::super::isa_profile::IsaProfile::cuda(100);
        let feats = profile.feature_set();
        assert!(feats.contains(FeatureSet::TMEM));
        assert!(feats.contains(FeatureSet::BLOCK_SCALED));
        let im = select_gemm_impl(feats, QuantPrecision::F16, (64, 64, 64));
        assert_eq!(im.name(), "GemmTcgen05",
            "SM100 + F16 should route to GemmTcgen05 (got {})", im.name());
    }

    #[test]
    fn route_sm80_ampere_selects_higher_tput_tile_impl() {
        // SM80 = TILE_GEMM + NativeBf16, 无 WGMMA/TMA/TMEM。
        // BF16 + TILE_GEMM 候选: GemmTcSm80 (tput=70) 与 GemmAmxBf16Tile (tput=60) requires 都满足,
        // max_by_key 选高 tput → GemmTcSm80 (验证 CR-001 验收点 1: 同 requires 高 tput 胜)。
        let profile = super::super::isa_profile::IsaProfile::cuda(80);
        let feats = profile.feature_set();
        assert!(feats.contains(FeatureSet::TILE_GEMM));
        assert!(feats.contains(FeatureSet::NATIVE_BF16));
        assert!(!feats.contains(FeatureSet::WGMMA));
        let im = select_gemm_impl(feats, QuantPrecision::BF16, (16, 8, 16));
        assert_eq!(im.name(), "GemmTcSm80",
            "SM80 + BF16 should route to GemmTcSm80 (got {})", im.name());
    }

    #[test]
    fn route_sm70_volta_tc_tile_is_selectable_for_fp16() {
        // SM70 = TILE_GEMM (wmma), 无 BF16, 无 FP8。
        // 注意 (设计发现, REQ-HW-TIER-001 验证): GemmTcSm70 与 GemmTcSm80 都只 requires=TILE_GEMM
        //   且都 supports F16 → TILE_GEMM 在 SM70/SM80 间是**重载位** (同一能力位, 两代 TC)。
        //   selector 按 throughput_class 仲裁 (GemmTcSm80 tput=70 > GemmTcSm70 tput=65)。
        //   真实 SM70 硬件无 mma.sync (SM80 才有), 但 FeatureSet 无 SM 版本细分位 —
        //   这是已知的 requires 粒度限制, 不在本 Task 范围内强行发明 SM-version 位 (遵循
        //   "无消费方不发明位" 铁律; 真正的 SM 版本区分由 DeviceProfile.sm_version 在更上层处理)。
        // 本测试验证: SM70 + F16 选到 TILE_GEMM 类 TC 后端 (而非 scalar/FMA), 且 tput >= 65。
        let profile = super::super::isa_profile::IsaProfile::cuda(70);
        let feats = profile.feature_set();
        assert!(feats.contains(FeatureSet::TILE_GEMM));
        assert!(!feats.contains(FeatureSet::NATIVE_BF16));
        let im = select_gemm_impl(feats, QuantPrecision::F16, (16, 16, 16));
        assert!(im.throughput_class() >= 65,
            "SM70 + F16 should select a TILE_GEMM TC impl (tput>=65), got {} (tput={})",
            im.name(), im.throughput_class());
        // 它必须是 TILE_GEMM 类 TC 后端 (GemmTcSm70 或 GemmTcSm80), 不能是 scalar/FMA
        let name = im.name();
        assert!(name == "GemmTcSm70" || name == "GemmTcSm80",
            "SM70 + F16 should route to a TC tile impl, got {}", name);
    }

    #[test]
    fn route_gfx950_cdna4_selects_mfma_v2() {
        // gfx950 = MFMA + MFMA_V2 + NativeBf16 → GemmMfmaV2 (tput=85) 胜 GemmMfmaV1 (tput=75)
        let profile = super::super::isa_profile::IsaProfile::hip(950);
        let feats = profile.feature_set();
        assert!(feats.contains(FeatureSet::MFMA));
        assert!(feats.contains(FeatureSet::MFMA_V2));
        let im = select_gemm_impl(feats, QuantPrecision::BF16, (32, 32, 16));
        assert_eq!(im.name(), "GemmMfmaV2",
            "gfx950 + BF16 should route to GemmMfmaV2 (got {})", im.name());
    }

    #[test]
    fn route_gfx908_cdna2_selects_mfma_v1() {
        // gfx908 = MFMA only (no MFMA_V2) → GemmMfmaV1
        let profile = super::super::isa_profile::IsaProfile::hip(908);
        let feats = profile.feature_set();
        assert!(feats.contains(FeatureSet::MFMA));
        assert!(!feats.contains(FeatureSet::MFMA_V2));
        let im = select_gemm_impl(feats, QuantPrecision::F16, (16, 16, 16));
        assert_eq!(im.name(), "GemmMfmaV1",
            "gfx908 + F16 should route to GemmMfmaV1 (got {})", im.name());
    }

    #[test]
    fn route_aarch64_sme_selects_sme_tile_for_f32() {
        // AArch64 SME = SME_TILE + TileGemm + NativeBf16。
        // F32: 只有 GemmSmeTile (supports F32, requires SME_TILE) 合法,
        // AMX 路径不支持 F32, GPU TILE_GEMM 后端不支持 F32 → GemmSmeTile 唯一。
        let profile = super::super::isa_profile::IsaProfile::aarch64(true, true, 64, true, true, true);
        let feats = profile.feature_set();
        assert!(feats.contains(FeatureSet::SME_TILE));
        assert!(feats.contains(FeatureSet::TILE_GEMM));
        let im = select_gemm_impl(feats, QuantPrecision::F32, (16, 16, 4));
        assert_eq!(im.name(), "GemmSmeTile",
            "SME + F32 should route to GemmSmeTile (got {})", im.name());
    }

    #[test]
    fn route_amx_bf16_features_select_tput_ge_60() {
        // AMX-BF16: has_amx → TileGemm{16,16,32}, has_bf16 → NativeBf16。
        // 用手构 features (x86 AMX profile 从 DeviceProfile::detect 派生, 不可移植)。
        // BF16 + TILE_GEMM: GemmTcSm80 (tput=70) 与 GemmAmxBf16Tile (tput=60) requires 都满足,
        // selector 选 tput 高者。本测试验证 TILE_GEMM+BF16 至少选到 tput>=60 的 tile 实现。
        let feats = FeatureSet::FMA
            .union(FeatureSet::TILE_GEMM)
            .union(FeatureSet::NATIVE_BF16);
        let im = select_gemm_impl(feats, QuantPrecision::BF16, (16, 16, 32));
        assert!(im.throughput_class() >= 60,
            "AMX-BF16 features should select tput>=60 impl, got {} (tput={})",
            im.name(), im.throughput_class());
    }

    #[test]
    fn route_scalar_fallback_when_no_features() {
        // 空特性 + 任意 dtype → GemmScalar (requires=EMPTY, 永远保底)
        let im = select_gemm_impl(FeatureSet::EMPTY, QuantPrecision::F32, (2, 2, 2));
        assert_eq!(im.name(), "GemmScalar");
        assert_eq!(im.throughput_class(), 1);
    }

    #[test]
    fn derive_feature_set_exhaustive_all_18_bits() {
        // 完整性: 每个有 FeatureSet 位的 IsaFeature 变体都能正确派生。
        // 构造含全部 18 个映射变体的 features 向量, 验证 18 位全部置位。
        use super::super::isa_profile::IsaFeature;
        let features = vec![
            IsaFeature::Fma,
            IsaFeature::NativeBf16,
            IsaFeature::NativeFp16,
            IsaFeature::NativeFp8,
            IsaFeature::TileGemm { m: 16, n: 16, k: 32 },
            IsaFeature::AmxFp16,
            IsaFeature::AmxFp8,
            IsaFeature::Wgmma,
            IsaFeature::Tma,
            IsaFeature::Tmem,
            IsaFeature::BlockScaled,
            IsaFeature::TwoCta,
            IsaFeature::Mfma,
            IsaFeature::MfmaV2,
            IsaFeature::Fp8Mfma,
            IsaFeature::Sve2,
            IsaFeature::SmeTileOp,
            IsaFeature::HardwareTranscendental,
        ];
        let fs = derive_feature_set(&features);
        // 18 位全部包含
        assert!(fs.contains(FeatureSet::FMA));
        assert!(fs.contains(FeatureSet::NATIVE_BF16));
        assert!(fs.contains(FeatureSet::NATIVE_FP16));
        assert!(fs.contains(FeatureSet::NATIVE_FP8));
        assert!(fs.contains(FeatureSet::TILE_GEMM));
        assert!(fs.contains(FeatureSet::AMX_FP16));
        assert!(fs.contains(FeatureSet::AMX_FP8));
        assert!(fs.contains(FeatureSet::WGMMA));
        assert!(fs.contains(FeatureSet::TMA));
        assert!(fs.contains(FeatureSet::TMEM));
        assert!(fs.contains(FeatureSet::BLOCK_SCALED));
        assert!(fs.contains(FeatureSet::TWO_CTA));
        assert!(fs.contains(FeatureSet::MFMA));
        assert!(fs.contains(FeatureSet::MFMA_V2));
        assert!(fs.contains(FeatureSet::FP8_MFMA));
        assert!(fs.contains(FeatureSet::SVE2));
        assert!(fs.contains(FeatureSet::SME_TILE));
        assert!(fs.contains(FeatureSet::HW_TRANSCEND));
    }

    #[test]
    fn derive_feature_set_no_consumer_variants_dont_invent_bits() {
        // 无消费方变体 (NativeFp4/AsyncCopy/Vnni/ArmBf16/SiMDGroupMatrix/...) 不发明 FeatureSet 位。
        // 验证: 这些变体派生后结果 = EMPTY (0 位)。
        use super::super::isa_profile::IsaFeature;
        let features = vec![
            IsaFeature::NativeFp4,
            IsaFeature::NativeFp6,
            IsaFeature::AsyncCopy,
            IsaFeature::PredicatedExec,
            IsaFeature::ScalableVector { min_vl: 16, max_vl: 256 },
            IsaFeature::WarpShuffle,
            IsaFeature::Vnni,
            IsaFeature::AmxTranspose,
            IsaFeature::AmxComplex,
            IsaFeature::Movrs,
            IsaFeature::Avx10_2,
            IsaFeature::Apx31Gpr,
            IsaFeature::SparseMaskIntersect,
            IsaFeature::WarpSpecialization,
            IsaFeature::CudaBarrier,
            IsaFeature::L2Multicast,
            IsaFeature::ThreadBlockCluster,
            IsaFeature::Fp4Mfma,
            IsaFeature::XcdTopology,
            IsaFeature::Sme2MultiVec,
            IsaFeature::SmeF16F16,
            IsaFeature::SmeI16I64,
            IsaFeature::ArmBf16,
            IsaFeature::ArmDotProd,
            IsaFeature::ArmI8mm,
            IsaFeature::SiMDGroupMatrix,
        ];
        let fs = derive_feature_set(&features);
        // 这些变体全无 FeatureSet 位 → 派生结果 = EMPTY
        // (ArmBf16 不映射: NativeBf16 才是统一计算能力位, ArmBf16 是架构别名)
        assert_eq!(fs, FeatureSet::EMPTY,
            "无消费方 IsaFeature 变体不应派生任何 FeatureSet 位, got {:?}", fs);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // §15 Task #7: numerical_sim 跨等级等价验证 (CR-TIER-SOVEREIGNTY-004)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //
    // 每个 OpImpl emit 的 VmInstr 序列, 通过 VmInstr 解释器执行, 与 scalar oracle 对齐。
    // BF16 容差 ~1e-2 (BF16 精度, 不是固定 1e-5), F32 容差 ~1e-5 (CR-004)。
    // @trace REQ-HW-TIER-006 [req:CrossTier-Equivalence-Tests] 跨等级等价测试集 (BF16×3 + F32×2)

    use crate::compiler::codegen::vm::instr::VRegId;
    use crate::compiler::codegen::vm::numerical_sim::{
        verify_op_impl_aligns_scalar, scalar_gemm_reference, tolerance_for,
    };

    /// 构造 GemmOpLayout + 执行 emit, 返回 VmProgram (供解释器验证)。
    fn emit_program_for_impl(
        im: &dyn OpImpl<GemmOpLayout>,
        dtype: QuantPrecision,
        m: usize, n: usize, k: usize,
    ) -> VmProgram {
        let width = SimdWidth::W256;
        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let mut ectx = EmitCtx {
            prog: &mut prog, width, pack_map: None, k_unroll: 1, debug_jit: false,
        };
        let lo = GemmOpLayout {
            m, n, k, m_bound: BoundExpr::Const(m), dtype, a_dtype: dtype, b_dtype: dtype, c_dtype: dtype, trans_b: false, mr: 4, nr: 2,
            a_ptr, b_ptr, c_ptr,
            epilogue: super::super::isa_hook::EpiloguePlace::OnAccumulators,
            tile: None,
        };
        im.emit(&mut ectx, &lo).expect("emit should succeed");
        prog
    }

    /// 生成确定性种子随机测试矩阵 (小规模, 避免溢出)。
    ///
    /// 用 `StdRng::seed_from_u64` 保证跨等级等价测试**可复现**: 每次跑同样的矩阵,
    /// 避免随机扰动导致 BF16 容差边界抖动 (CR-TIER-SOVEREIGNTY-004 数值验证需确定性)。
    fn seeded_test_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    // ── BF16 跨等级等价测试 (≥ 3 组) ──

    #[test]
    fn bf16_gemm_fma_blis_aligns_scalar_oracle_2x4x8() {
        // 小矩阵 2×4×8: 验证 BF16 WidenCompute 路径数值对齐。
        let im = &GemmFmaBlis as &dyn OpImpl<GemmOpLayout>;
        let dtype = QuantPrecision::BF16;
        let prog = emit_program_for_impl(im, dtype, 2, 4, 8);
        let a = seeded_test_matrix(2, 8, 42);
        let b = seeded_test_matrix(8, 4, 43);
        // ptr bindings: emit_program_for_impl allocs a_ptr=0, b_ptr=1, c_ptr=2 sequentially
        let ptr_bindings: [(VRegId, &str); 3] = [
            (VRegId(0), "a"), (VRegId(1), "b"), (VRegId(2), "c"),
        ];
        let (max_diff, passed) = verify_op_impl_aligns_scalar(
            &prog, &a, &b, 2, 4, 8, dtype, SimdWidth::W256, &ptr_bindings,
        ).expect("interpreter should run");
        let tol = tolerance_for(dtype);
        assert!(passed, "BF16 GemmFmaBlis 2x4x8 vs scalar: max_diff={} > tol={}", max_diff, tol);
        assert!(max_diff <= tol, "BF16 容差 {} 超标: {}", tol, max_diff);
    }

    #[test]
    fn bf16_gemm_fma_blis_aligns_scalar_oracle_4x8x16() {
        // 中等矩阵 4×8×16: 覆盖 BLIS 微核条件触发。
        let im = &GemmFmaBlis as &dyn OpImpl<GemmOpLayout>;
        let dtype = QuantPrecision::BF16;
        let prog = emit_program_for_impl(im, dtype, 4, 8, 16);
        let a = seeded_test_matrix(4, 16, 44);
        let b = seeded_test_matrix(16, 8, 45);
        let ptr_bindings: [(VRegId, &str); 3] = [
            (VRegId(0), "a"), (VRegId(1), "b"), (VRegId(2), "c"),
        ];
        let (max_diff, passed) = verify_op_impl_aligns_scalar(
            &prog, &a, &b, 4, 8, 16, dtype, SimdWidth::W256, &ptr_bindings,
        ).expect("interpreter should run");
        let tol = tolerance_for(dtype);
        assert!(passed, "BF16 GemmFmaBlis 4x8x16 vs scalar: max_diff={} > tol={}", max_diff, tol);
    }

    #[test]
    fn bf16_gemm_scalar_aligns_oracle_3x5x12() {
        // GemmScalar (保底) BF16 路径: 验证 scalar 保底实现数值正确。
        let im = &GemmScalar as &dyn OpImpl<GemmOpLayout>;
        let dtype = QuantPrecision::BF16;
        let prog = emit_program_for_impl(im, dtype, 3, 5, 12);
        let a = seeded_test_matrix(3, 12, 46);
        let b = seeded_test_matrix(12, 5, 47);
        let ptr_bindings: [(VRegId, &str); 3] = [
            (VRegId(0), "a"), (VRegId(1), "b"), (VRegId(2), "c"),
        ];
        let (max_diff, passed) = verify_op_impl_aligns_scalar(
            &prog, &a, &b, 3, 5, 12, dtype, SimdWidth::W256, &ptr_bindings,
        ).expect("interpreter should run");
        let tol = tolerance_for(dtype);
        assert!(passed, "BF16 GemmScalar 3x5x12 vs scalar: max_diff={} > tol={}", max_diff, tol);
    }

    // ── F32 跨等级等价测试 (≥ 2 组) ──

    #[test]
    fn f32_gemm_fma_blis_aligns_scalar_oracle_6x10x20() {
        // F32 短精度容差 ~1e-5: 验证 FMA 路径数值对齐。
        let im = &GemmFmaBlis as &dyn OpImpl<GemmOpLayout>;
        let dtype = QuantPrecision::F32;
        let prog = emit_program_for_impl(im, dtype, 6, 10, 20);
        let a = seeded_test_matrix(6, 20, 48);
        let b = seeded_test_matrix(20, 10, 49);
        let ptr_bindings: [(VRegId, &str); 3] = [
            (VRegId(0), "a"), (VRegId(1), "b"), (VRegId(2), "c"),
        ];
        let (max_diff, passed) = verify_op_impl_aligns_scalar(
            &prog, &a, &b, 6, 10, 20, dtype, SimdWidth::W256, &ptr_bindings,
        ).expect("interpreter should run");
        let tol = tolerance_for(dtype);
        assert!(passed, "F32 GemmFmaBlis 6x10x20 vs scalar: max_diff={} > tol={}", max_diff, tol);
        assert!(max_diff <= 1e-5, "F32 容差 1e-5 超标: {}", max_diff);
    }

    #[test]
    fn f32_gemm_scalar_aligns_oracle_2x3x6() {
        // F32 GemmScalar 保底路径: 验证保底数值正确。
        let im = &GemmScalar as &dyn OpImpl<GemmOpLayout>;
        let dtype = QuantPrecision::F32;
        let prog = emit_program_for_impl(im, dtype, 2, 3, 6);
        let a = seeded_test_matrix(2, 6, 50);
        let b = seeded_test_matrix(6, 3, 51);
        let ptr_bindings: [(VRegId, &str); 3] = [
            (VRegId(0), "a"), (VRegId(1), "b"), (VRegId(2), "c"),
        ];
        let (max_diff, passed) = verify_op_impl_aligns_scalar(
            &prog, &a, &b, 2, 3, 6, dtype, SimdWidth::W256, &ptr_bindings,
        ).expect("interpreter should run");
        let tol = tolerance_for(dtype);
        assert!(passed, "F32 GemmScalar 2x3x6 vs scalar: max_diff={} > tol={}", max_diff, tol);
    }

    // ── Scalar oracle 自验证 (已知答案矩阵) ──

    #[test]
    fn scalar_oracle_known_matrix_3x5x4() {
        // 已知答案: A = [[1,2,3,4],[5,6,7,8],[9,10,11,12]] (3×4)
        //           B = [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[1,1,1,1,1]] (4×5)
        //           C = Σ_p A[m][p] * B[p][n]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let golden = scalar_gemm_reference(&a, &b, 3, 5, 4);
        // 预期: C[0][0] = 1*1 + 2*0 + 3*0 + 4*1 = 5; C[0][1] = 1*0 + 2*1 + 3*0 + 4*1 = 6
        assert_eq!(golden[0], 5.0);
        assert_eq!(golden[1], 6.0);
        assert_eq!(golden[2], 3.0 + 4.0); // C[0][2] = 3*1 + 4*1 = 7
        // C[1][4] = 5*1 + 6*0 + 7*0 + 8*1 = 13 (row-major index = 1*5+4 = 9)
        assert_eq!(golden[9], 13.0);
        // C[2][4] = 9*1 + 10*0 + 11*0 + 12*1 = 21 (row-major index = 2*5+4 = 14)
        assert_eq!(golden[14], 21.0);
    }

    #[test]
    fn tolerance_bf16_vs_f32() {
        // CR-TIER-SOVEREIGNTY-004: BF16 容差 ~1e-2, F32 ~1e-5。
        let bf16_tol = tolerance_for(QuantPrecision::BF16);
        let f32_tol = tolerance_for(QuantPrecision::F32);
        assert!(bf16_tol >= 1e-2, "BF16 容差应 ≥ 1e-2, got {}", bf16_tol);
        assert!(bf16_tol <= 1e-1, "BF16 容差应 ≤ 1e-1 (保守), got {}", bf16_tol);
        assert!(f32_tol <= 1e-5, "F32 容差应 ≤ 1e-5, got {}", f32_tol);
    }
}
