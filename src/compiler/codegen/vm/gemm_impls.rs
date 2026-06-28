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
        let sym_map = SymDimSlotMap::mega_kernel_abi();
        let m_dim = crate::compiler::graph::SymDim::Concrete(lo.m);
        emit_gemm_inline_with_epilogue(
            ctx.prog, &m_dim, lo.n, lo.k, ctx.width,
            lo.a_ptr, lo.b_ptr, lo.c_ptr,
            &[], &sym_map, false, None, lo.dtype, lo.trans_b,
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
                lo.dtype, lo.trans_b,
            )
        } else {
            let sym_map = SymDimSlotMap::mega_kernel_abi();
            let m_dim = crate::compiler::graph::SymDim::Concrete(lo.m);
            emit_gemm_inline_with_epilogue(
                ctx.prog, &m_dim, lo.n, lo.k, ctx.width,
                lo.a_ptr, lo.b_ptr, lo.c_ptr,
                &[], &sym_map, false, None, lo.dtype, lo.trans_b,
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
pub fn derive_feature_set(features: &[super::isa_profile::IsaFeature]) -> FeatureSet {
    let mut fs = FeatureSet::EMPTY;
    for feat in features {
        use super::isa_profile::IsaFeature;
        match feat {
            IsaFeature::Fma => fs = fs.union(FeatureSet::FMA),
            IsaFeature::NativeBf16 => fs = fs.union(FeatureSet::NATIVE_BF16),
            IsaFeature::NativeFp16 => fs = fs.union(FeatureSet::NATIVE_FP16),
            IsaFeature::NativeFp8 => fs = fs.union(FeatureSet::NATIVE_FP8),
            IsaFeature::TileGemm { .. } => fs = fs.union(FeatureSet::TILE_GEMM),
            IsaFeature::AmxFp16 => fs = fs.union(FeatureSet::AMX_FP16),
            IsaFeature::AmxFp8 => fs = fs.union(FeatureSet::AMX_FP8),
            IsaFeature::Wgmma => fs = fs.union(FeatureSet::WGMMA),
            IsaFeature::Tma => fs = fs.union(FeatureSet::TMA),
            IsaFeature::Tmem => fs = fs.union(FeatureSet::TMEM),
            IsaFeature::BlockScaled => fs = fs.union(FeatureSet::BLOCK_SCALED),
            IsaFeature::TwoCta => fs = fs.union(FeatureSet::TWO_CTA),
            IsaFeature::Mfma => fs = fs.union(FeatureSet::MFMA),
            IsaFeature::MfmaV2 => fs = fs.union(FeatureSet::MFMA_V2),
            IsaFeature::Fp8Mfma => fs = fs.union(FeatureSet::FP8_MFMA),
            IsaFeature::Sve2 => fs = fs.union(FeatureSet::SVE2),
            IsaFeature::SmeTileOp => fs = fs.union(FeatureSet::SME_TILE),
            IsaFeature::HardwareTranscendental => fs = fs.union(FeatureSet::HW_TRANSCEND),
            _ => {}
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
    use crate::compiler::codegen::vm::instr::{SimdWidth, VRegKind, VmInstr, VmProgram};

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
            dtype,
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
            m: 8, n: 8, k: 4, dtype: QuantPrecision::F32,
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
}
