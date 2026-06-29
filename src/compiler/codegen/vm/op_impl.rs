//! OpImpl — 算子族主权实现框架 (SPEC ARCH-HW-TIER-SOVEREIGNTY CR-TIER-SOVEREIGNTY-001..004)
//!
//! 每个 OpImpl<L> 是一个算子族 (L = Layout 类型) 的一个**主权实现**——peer,
//! 不分主次 (CR-001)。selector (select_gemm_impl) 按 `requires ⊆ features` +
//! `supports_dtype` + `throughput_class` 三维折叠选出当前 (硬件, dtype) 下最优实现,
//! 然后 emit 自包含 codegen, emit 体内零 dtype/ISA 分支 (CR-001 后置条件)。
//!
//! ## 三阶段: select → then → emit
//!
//! ```text
//! select_gemm_impl(feats, dtype, shape)
//!   ① filter supports_dtype(dtype)
//!   ② filter feats.contains(requires())
//!   ③ max_by_key throughput_class()
//!   → &'static dyn OpImpl<GemmOpLayout>
//! im.emit(&mut ectx, &layout)
//! ```
//!
//! ## GEMM-FMA 族 (本模块首发)
//!
//! 13 个 OpImpl 平权交付: GemmFmaBlis / GemmAmx{Bf16,Fp16,Fp8}Tile /
//! GemmWgmma / GemmTcgen05 / GemmMfmaV1 / GemmMfmaV2 / GemmSmeTile /
//! GemmTcSm70 / GemmTcSm80 / GemmScalar 保底。

use super::instr::{SimdWidth, VRegId};
use super::isa_hook::EpiloguePlace;
use crate::compiler::pack_map::PackMap;
use crate::compiler::trace::QuantPrecision;
use crate::types::CompilerError;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 FeatureSet — 从 IsaProfile.features 派生的布尔能力位
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 硬件能力位集合 — OpImpl.requires() 的返回类型 (CR-002: requires 是 FeatureSet 谓词子集匹配)。
///
/// 纯布尔能力位。带参数的数据 (TileGemm{m,n,k} 的尺寸 / ScalableVector{vl}) **不进**
/// bitflags——它们是 GemmOpLayout 的输入数据, 由 selector 从 IsaProfile 读出后填进
/// Layout, 不参与 requires 谓词匹配。
// @trace REQ-HW-TIER-002 [req:FeatureSet] 布尔能力位 bitflags, OpImpl.requires() 谓词源
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(pub u64);

impl FeatureSet {
    /// 空集 — GemmScalar 的 requires (永远满足)。
    pub const EMPTY: Self = Self(0);

    pub const FMA: Self = Self(1 << 0);
    pub const NATIVE_BF16: Self = Self(1 << 1);
    pub const NATIVE_FP16: Self = Self(1 << 2);
    pub const NATIVE_FP8: Self = Self(1 << 3);
    /// AMX/SME/Tensor Core 通用 tile 能力 (参数不在此, 在 GemmOpLayout.tile)。
    pub const TILE_GEMM: Self = Self(1 << 4);
    pub const AMX_FP16: Self = Self(1 << 5);
    pub const AMX_FP8: Self = Self(1 << 6);
    pub const WGMMA: Self = Self(1 << 7);
    pub const TMA: Self = Self(1 << 8);
    pub const TMEM: Self = Self(1 << 9);
    pub const BLOCK_SCALED: Self = Self(1 << 10);
    pub const TWO_CTA: Self = Self(1 << 11);
    pub const MFMA: Self = Self(1 << 12);
    pub const MFMA_V2: Self = Self(1 << 13);
    pub const FP8_MFMA: Self = Self(1 << 14);
    pub const SVE2: Self = Self(1 << 15);
    pub const SME_TILE: Self = Self(1 << 16);
    pub const HW_TRANSCEND: Self = Self(1 << 17);
    /// F16C (vcvtph2ps/vcvtps2ph) — F16↔F32 转换 (Ivy Bridge+ 基线)。
    /// 与 NATIVE_FP16 (AVX-512 FP16 计算) 不同: F16C 只转换, 不计算 F16。
    /// REQ-HW-TIER-001: 细粒度 flag, 区分"能转换 F16"和"能计算 F16"。
    pub const F16C: Self = Self(1 << 18);

    /// 并集。
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// 是否包含 other 的所有位 (CR-002 谓词子集匹配)。
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for FeatureSet {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 EmitCtx — emit 的可变副作用通道 (复用 LoweringContext 字段切片)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// emit 期需要的最小可变切片——OpImpl 不反向依赖整个 plan_lower。
///
/// 字段来源与 LoweringContext/CompileSession 一致, 避免重复构造。
pub struct EmitCtx<'a, 'p> {
    pub prog: &'a mut super::instr::VmProgram,
    pub width: SimdWidth,
    pub pack_map: Option<&'p PackMap>,
    pub k_unroll: usize,
    pub debug_jit: bool,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 GemmOpLayout — GEMM 族的形状契约 (VM/IR 层, 非机器码 layout)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Tile 形状参数——承载 OpImpl 需要但 requires() 不表达的参数化数据 (CR-002)。
///
/// selector 从 IsaProfile/硬件 hook 解析后填进 GemmOpLayout.tile, emit 零推导。
#[derive(Debug, Clone, Copy)]
pub struct TileShape {
    pub rows: usize,
    pub cols: usize,
    pub k_depth: usize,
    pub dtype: QuantPrecision,
}

/// GEMM 族的"形状契约"。承载 selector 已解析完毕的全部尺寸/指针/dtype,
/// emit 零推导、零 dtype 分支 (CR-001 后置条件)。
///
/// 命名注记: 项目已有 `codegen/x86_64/jit/layout.rs::GemmLayout` (机器码偏移层)。
/// 本结构是 VM/IR 层, 命名 `GemmOpLayout` 规避同名混淆。
pub struct GemmOpLayout {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// M 维循环边界表达式 (符号维必须用运行时 bound, 禁止退化为 max_value=8192)。
    /// @trace ARCH-SYMDIM-THREADING: symbolic M 的运行时边界穿透 (BCE-20260629-001 SIGSEGV 根治)。
    /// OpImpl emit 必须把 lo.m_bound 作 seq_bound_override 传给 emit_gemm_*,
    /// 不得传 None 让 m_dim 退化为 Concrete(lo.m) 大循环 (SIGSEGV)。
    pub m_bound: super::instr::BoundExpr,
    /// emit 不据此分支; 仅作 store/narrow 与 OpImpl 自身 tile dtype 推断用。
    pub dtype: QuantPrecision,
    /// BCE-20260629-003 (Pattern c): per-matrix dtype. a_dtype/c_dtype 通常 F32 (激活计算),
    /// b_dtype 可能是 BF16/F16 (权重)。VecLoad 按 dtype 走 WidenCompute 自动 widen。
    pub a_dtype: QuantPrecision,
    pub b_dtype: QuantPrecision,
    pub c_dtype: QuantPrecision,
    pub trans_b: bool,
    /// selector 从 hook.gemm_microkernel_shape() 预填。
    pub mr: usize,
    pub nr: usize,
    pub a_ptr: VRegId,
    pub b_ptr: VRegId,
    pub c_ptr: VRegId,
    pub epilogue: EpiloguePlace,
    /// Tile 路径专用 (TileMma/Wgmma/Tcgen05/Mfma/SME)。BLIS/naive 路径为 None。
    pub tile: Option<TileShape>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 OpImpl<L> trait — 算子族的一个主权实现
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 一个算子族的一个 sovereign 实现 —— peer, 不分主次 (CR-001)。
///
/// `L` 是该族的 Layout 结构体 (首发 `GemmOpLayout`)。不做单一巨型 Layout enum,
/// 避免 emit 内 match 复活 (违反 CR-001 后置条件)。
// @trace REQ-HW-TIER-002 [req:OpImpl] select-then-emit 两阶段主权 trait 框架
pub trait OpImpl<L>: Send + Sync {
    /// 该实现需要的精确硬件特性集 (CR-002: 谓词子集匹配)。纯 FeatureSet, 无参数化数据。
    fn requires(&self) -> FeatureSet;
    /// 该 ISA 上的相对吞吐 (选择用粗排, u8, 非地位高低)。CR-001 验收点 1。
    fn throughput_class(&self) -> u8;
    /// 自包含 codegen, 零 dtype/isa 分支 (CR-001 后置条件)。
    fn emit(&self, ctx: &mut EmitCtx<'_, '_>, layout: &L) -> Result<(), CompilerError>;
    /// 该实现支持的 dtype 谓词 (dtype 维过滤)。
    fn supports_dtype(&self, dt: QuantPrecision) -> bool;
    /// 诊断名 (按"它是什么"命名: GemmFmaBlis / GemmAmxFp8Tile)。
    fn name(&self) -> &'static str;
}
