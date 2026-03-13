//! ISA Scheduler — decides which hardware features to use for each fusion group.
//!
//! Given a `HwCapabilityMatrix` (what the target can do) and a `FusionPlan`
//! (what needs to be computed), the scheduler produces an `IsaExecPlan` per
//! group that tells the code generator *how* to emit each kernel.

use super::simd_ops::SimdWidth;
use super::tile_ops::TileAccelKind;
use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};

// ── HwCapabilityMatrix ──────────────────────────────────────────────────────

/// Hardware capability matrix — summarizes what the target can do.
#[derive(Debug, Clone)]
pub struct HwCapabilityMatrix {
    /// Available SIMD widths (e.g., [W128, W256] for AVX2, [W128, W256, W512] for AVX-512).
    pub simd_widths: Vec<SimdWidth>,
    /// Whether tile accelerator is available (AMX/SME).
    pub has_tile_accel: bool,
    /// Tile accelerator kind (if available).
    pub tile_accel: Option<TileAccelKind>,
}

impl HwCapabilityMatrix {
    /// Build from a `KernelConfig`.
    pub fn from_kernel_config(cfg: &crate::microarch::KernelConfig) -> Self {
        let mut simd_widths = vec![SimdWidth::W128];
        // x86: always has SSE (W128); AVX2 adds W256; AVX-512 adds W512.
        // aarch64: NEON is W128; SVE adds Wvl (variable-length).
        if cfg.has_sve || cfg.has_sve2 {
            simd_widths.push(SimdWidth::Wvl);
        } else if cfg.simd_width >= 8 {
            simd_widths.push(SimdWidth::W256);
        }
        if cfg.use_avx512 {
            simd_widths.push(SimdWidth::W512);
        }

        let tile_accel = if cfg.has_amx {
            Some(TileAccelKind::Amx)
        } else {
            None
        };

        HwCapabilityMatrix {
            simd_widths,
            has_tile_accel: cfg.has_amx,
            tile_accel,
        }
    }
}

// ── Execution plan types ────────────────────────────────────────────────────

/// How to execute the GEMM core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmEngine {
    /// Use tile accelerator (AMX TDPBF16PS / SME FMOPA).
    Tile(TileAccelKind),
    /// Use SIMD FMA micro-kernel at the given width.
    Simd(SimdWidth),
}

/// How to handle tail elements (N % simd_lanes != 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TailStrategy {
    /// AVX-512 k-mask or SVE predicate — no extra code path.
    Masked,
    /// Fall back to narrower SIMD (e.g., zmm main -> ymm tail).
    NarrowSimd(SimdWidth),
    /// Scalar loop for remaining elements.
    Scalar,
}

/// Execution plan for a single fusion group.
#[derive(Debug, Clone)]
pub struct IsaExecPlan {
    /// GEMM engine selection (only for GEMM-containing groups).
    pub gemm_engine: Option<GemmEngine>,
    /// Primary SIMD width for the main loop body.
    pub simd_primary: SimdWidth,
    /// Tail handling strategy.
    pub tail: TailStrategy,
    /// Whether to use non-temporal stores for large outputs.
    pub use_nt_stores: bool,
}

// ── IsaScheduler ────────────────────────────────────────────────────────────

/// ISA scheduler — builds `IsaExecPlan` for each `FusionGroup`.
pub struct IsaScheduler {
    pub hw: HwCapabilityMatrix,
}

impl IsaScheduler {
    pub fn new(hw: HwCapabilityMatrix) -> Self {
        Self { hw }
    }

    /// Plan execution for an entire `FusionPlan`.
    pub fn plan(&self, fusion_plan: &FusionPlan) -> Vec<IsaExecPlan> {
        fusion_plan.groups.iter().map(|g| self.plan_group(g)).collect()
    }

    /// Plan execution for a single fusion group.
    pub fn plan_group(&self, group: &FusionGroup) -> IsaExecPlan {
        match &group.mode {
            FusionMode::EpilogueInjection
            | FusionMode::TileLevelFusion { .. }
            | FusionMode::ComputeRoot { .. }
            | FusionMode::QkvSharedInput
            | FusionMode::NormIntoGemm => self.plan_gemm(group),
            FusionMode::LoopFusion | FusionMode::Standalone => self.plan_elementwise(group),
        }
    }

    /// Plan a GEMM-containing group.
    ///
    /// Decision tree: AMX > AVX-512 > SVE > AVX2 > NEON.
    fn plan_gemm(&self, _group: &FusionGroup) -> IsaExecPlan {
        let gemm_engine = if self.hw.has_tile_accel {
            // Tile accelerator available — use it for the GEMM core.
            // unwrap is safe: has_tile_accel implies tile_accel.is_some().
            Some(GemmEngine::Tile(self.hw.tile_accel.unwrap()))
        } else {
            // No tile accel — GEMM uses SIMD FMA at the widest available width.
            None
        };

        let simd_primary = self.widest_simd();
        let tail = self.tail_strategy(simd_primary);

        IsaExecPlan {
            gemm_engine,
            simd_primary,
            tail,
            use_nt_stores: false,
        }
    }

    /// Plan an elementwise / standalone group.
    fn plan_elementwise(&self, _group: &FusionGroup) -> IsaExecPlan {
        let simd_primary = self.widest_simd();
        let tail = self.tail_strategy(simd_primary);

        IsaExecPlan {
            gemm_engine: None,
            simd_primary,
            tail,
            // Caller can override based on tensor size vs L2.
            use_nt_stores: false,
        }
    }

    /// Widest available SIMD width (last element of sorted simd_widths).
    fn widest_simd(&self) -> SimdWidth {
        *self.hw.simd_widths.last().unwrap_or(&SimdWidth::W128)
    }

    /// Choose tail strategy based on the primary SIMD width.
    fn tail_strategy(&self, primary: SimdWidth) -> TailStrategy {
        match primary {
            // AVX-512 has k-masks, SVE has predicates — masked tail is free.
            SimdWidth::W512 | SimdWidth::Wvl => TailStrategy::Masked,
            // AVX2 (W256): fall back to W128 for tail if available.
            SimdWidth::W256 if self.hw.simd_widths.contains(&SimdWidth::W128) => {
                TailStrategy::NarrowSimd(SimdWidth::W128)
            }
            // Everything else: scalar tail loop.
            _ => TailStrategy::Scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan};
    use crate::compiler::graph::OpId;
    use std::collections::HashMap;

    fn make_group(mode: FusionMode) -> FusionGroup {
        FusionGroup {
            id: 0,
            anchor: OpId(0),
            epilogue: vec![],
            mode,
            ops: vec![OpId(0)],
        }
    }

    fn avx2_hw() -> HwCapabilityMatrix {
        HwCapabilityMatrix {
            simd_widths: vec![SimdWidth::W128, SimdWidth::W256],
            has_tile_accel: false,
            tile_accel: None,
        }
    }

    fn avx512_amx_hw() -> HwCapabilityMatrix {
        HwCapabilityMatrix {
            simd_widths: vec![SimdWidth::W128, SimdWidth::W256, SimdWidth::W512],
            has_tile_accel: true,
            tile_accel: Some(TileAccelKind::Amx),
        }
    }

    fn neon_hw() -> HwCapabilityMatrix {
        HwCapabilityMatrix {
            simd_widths: vec![SimdWidth::W128],
            has_tile_accel: false,
            tile_accel: None,
        }
    }

    fn sve_hw() -> HwCapabilityMatrix {
        HwCapabilityMatrix {
            simd_widths: vec![SimdWidth::W128, SimdWidth::Wvl],
            has_tile_accel: false,
            tile_accel: None,
        }
    }

    #[test]
    fn avx2_gemm_uses_simd_w256() {
        let sched = IsaScheduler::new(avx2_hw());
        let plan = sched.plan_group(&make_group(FusionMode::EpilogueInjection));
        assert_eq!(plan.gemm_engine, None);
        assert_eq!(plan.simd_primary, SimdWidth::W256);
        assert_eq!(plan.tail, TailStrategy::NarrowSimd(SimdWidth::W128));
    }

    #[test]
    fn avx512_amx_gemm_uses_tile() {
        let sched = IsaScheduler::new(avx512_amx_hw());
        let plan = sched.plan_group(&make_group(FusionMode::EpilogueInjection));
        assert_eq!(plan.gemm_engine, Some(GemmEngine::Tile(TileAccelKind::Amx)));
        assert_eq!(plan.simd_primary, SimdWidth::W512);
        assert_eq!(plan.tail, TailStrategy::Masked);
    }

    #[test]
    fn neon_elementwise_scalar_tail() {
        let sched = IsaScheduler::new(neon_hw());
        let plan = sched.plan_group(&make_group(FusionMode::LoopFusion));
        assert_eq!(plan.gemm_engine, None);
        assert_eq!(plan.simd_primary, SimdWidth::W128);
        assert_eq!(plan.tail, TailStrategy::Scalar);
    }

    #[test]
    fn sve_elementwise_masked_tail() {
        let sched = IsaScheduler::new(sve_hw());
        let plan = sched.plan_group(&make_group(FusionMode::LoopFusion));
        assert_eq!(plan.gemm_engine, None);
        assert_eq!(plan.simd_primary, SimdWidth::Wvl);
        assert_eq!(plan.tail, TailStrategy::Masked);
    }

    #[test]
    fn plan_whole_fusion_plan() {
        let sched = IsaScheduler::new(avx512_amx_hw());
        let fusion_plan = FusionPlan {
            groups: vec![
                make_group(FusionMode::EpilogueInjection),
                make_group(FusionMode::LoopFusion),
                make_group(FusionMode::Standalone),
            ],
            op_to_group: HashMap::new(),
        };
        let plans = sched.plan(&fusion_plan);
        assert_eq!(plans.len(), 3);
        // GEMM group gets tile engine
        assert_eq!(plans[0].gemm_engine, Some(GemmEngine::Tile(TileAccelKind::Amx)));
        // Elementwise groups get no tile engine
        assert_eq!(plans[1].gemm_engine, None);
        assert_eq!(plans[2].gemm_engine, None);
    }
}
