//! Algorithm Template Declarative Data Model (SPEC 27 REQ-AT-001)
//!
//! Pure static data describing algorithm loop nesting, steps, and parameters.
//! Zero methods, zero code generation logic.
//! The template interpreter traverses AlgoStep trees and produces Vec<TraceOp>,
//! which then flows through the unified auto_lower_trace pipeline.

use crate::compiler::trace::QuantPrecision;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Core template
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Algorithm template — pure static data describing an algorithm's loop nesting
/// and computation steps.
///
/// The template interpreter walks the step tree and outputs `Vec<TraceOp>`,
/// then routes through the unified `auto_lower_trace` pipeline.
///
/// File location: `algo_templates/*.rs` — only `static` data definitions.
pub struct AlgoTemplate {
    pub name: &'static str,
    pub strategy: AlgoStrategy,
    pub device_req: DeviceReq,
    pub steps: &'static [AlgoStep],
    pub params: &'static [(&'static str, AlgoParam)],
    pub micro_kernel: Option<&'static MicroKernelDef>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AlgoStrategy — algorithm family META
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Algorithm family META — analogous to SPEC 26's `BlockUnpackMode`,
/// but describing algorithm-level strategy variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgoStrategy {
    // GEMM
    GemmNaive,
    GemmBlis,
    GemmGpuTiled,
    GemmGpuPipelined,
    GemmHardwareTile,

    // Attention
    AttnMha,
    AttnGqa,
    AttnMla,

    // MoE
    MoeRouterTopk,
    MoePackedDispatch,

    // Norm
    NormRms,
    NormLayer,

    // RoPE
    RopeStandard,
    RopePartial,

    // Sampling
    SamplingArgmax,
    SamplingTemperature,
    SamplingSoftmax,
    SamplingTopK,
    SamplingTopP,
    SamplingMultinomial,

    // Embedding
    EmbeddingGather,

    // Quant
    QuantGather,
}

/// Strategy family — used by the template registry to group related templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyFamily {
    Gemm,
    Attention,
    Moe,
    Norm,
    Rope,
    Sampling,
    Embedding,
    Quant,
}

impl AlgoStrategy {
    pub fn family(&self) -> StrategyFamily {
        match self {
            Self::GemmNaive | Self::GemmBlis | Self::GemmGpuTiled
            | Self::GemmGpuPipelined | Self::GemmHardwareTile => StrategyFamily::Gemm,

            Self::AttnMha | Self::AttnGqa | Self::AttnMla => StrategyFamily::Attention,

            Self::MoeRouterTopk | Self::MoePackedDispatch => StrategyFamily::Moe,

            Self::NormRms | Self::NormLayer => StrategyFamily::Norm,

            Self::RopeStandard | Self::RopePartial => StrategyFamily::Rope,

            Self::SamplingArgmax | Self::SamplingTemperature | Self::SamplingSoftmax
            | Self::SamplingTopK | Self::SamplingTopP | Self::SamplingMultinomial => StrategyFamily::Sampling,

            Self::EmbeddingGather => StrategyFamily::Embedding,

            Self::QuantGather => StrategyFamily::Quant,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AlgoStep — algorithm step tree
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Algorithm step — the body of a template, describing loop nesting and
/// computation steps. The interpreter walks the step tree and translates
/// each step into structured TraceOp sequences.
#[derive(Debug, Clone)]
pub enum AlgoStep {
    // Control structure

    /// Sequential execution of a group of steps.
    Seq(&'static [AlgoStep]),

    /// Loop: bound/step resolved from the parameter table.
    /// Interpreter produces TraceOp::Loop { bound, step, body }.
    Loop {
        bound: &'static str,
        step: &'static str,
        body: &'static [AlgoStep],
    },

    /// Conditional execution: only when device satisfies the requirement.
    Conditional {
        requirement: DeviceReq,
        body: &'static [AlgoStep],
    },

    // Memory operations (map to TraceOp variants)

    /// Load matrix panel → TraceOp::PanelLoad.
    LoadPanel {
        matrix: MatrixRole,
        rows_param: &'static str,
        cols_param: &'static str,
    },

    /// Pack buffer → TraceOp::PackBuffer.
    PackBuffer {
        buffer_name: &'static str,
        rows_param: &'static str,
        cols_param: &'static str,
    },

    /// Micro-kernel computation: mr × nr × kc MAC loop.
    /// Interpreter expands MicroKernelDef into TraceOp sequences.
    MicroKernel,

    /// Store result → TraceOp::PanelStore.
    StoreResult {
        rows_param: &'static str,
        cols_param: &'static str,
    },

    // GPU-specific (map to TraceOp extensions)

    /// TraceOp::SharedMemDeclare
    SharedMemDeclare {
        name: &'static str,
        size_param: &'static str,
    },

    /// TraceOp::AsyncCopyToShared
    AsyncCopyToSmem {
        buffer_name: &'static str,
        size_param: &'static str,
    },

    /// TraceOp::AsyncWaitGroup
    AsyncWait { group: u32 },

    /// TraceOp::SyncBarrier
    Barrier { barrier_name: &'static str },

    // Tile/Matrix specific

    /// TraceOp::TileConfig
    TileConfig { rows: &'static str, cols: &'static str },

    /// TraceOp::TileMma
    TileMma,

    /// TraceOp::TileRelease
    TileRelease,

    // Computation steps (directly produce TraceOp)

    /// Embed existing TraceOp sequence (from SymExec trace or manual definition).
    TraceBody(&'static [AlgoTraceStep]),

    /// Reduction: TraceOp::HReduce
    Reduce { op: ReduceOp },

    /// Activation function: TraceOp lookup (Silu→Sigmoid+Mul, Gelu→Exp+...)
    Activation { kind: ActivationKind },

    /// Softmax three-phase: reduce_max → exp_sum → normalize
    Softmax,

    /// Quantized dequantize: TraceOp + BlockUnpackMode
    Dequantize { mode: BlockUnpackMode },

    /// Embedding lookup: TraceOp::ScalarLoad + StrideMul + PtrAdd + VecLoadIndexed
    EmbeddingGather,

    /// MoE Router
    MoeRouterGemv {
        num_experts: &'static str,
        hidden: &'static str,
    },
    MoeTopK {
        num_experts: &'static str,
        top_k: &'static str,
    },

    /// Epilogue injection
    Epilogue { ops: &'static [EpilogueOp] },

    /// Zero-fill a region.
    ZeroFill { bytes_param: &'static str },

    /// Row copy from source to destination.
    RowCopy {
        rows_param: &'static str,
        cols_param: &'static str,
    },
}

/// A trace step embedded within an AlgoStep::TraceBody.
/// These map 1:1 to TraceOp variants but use parameter references
/// instead of concrete slot IDs (resolved by the interpreter).
#[derive(Debug, Clone)]
pub enum AlgoTraceStep {
    /// Load input by name → TraceOp::Input
    LoadInput { name: &'static str },
    /// Load constant → TraceOp::Const
    LoadConst { value: f64 },
    /// Load parameter by name → TraceOp::Const(value from ParamTable::resolve_f64).
    /// Unlike LoadConst which bakes a hardcoded value, LoadParam resolves the value
    /// at template-instantiation time from the ParamTable, allowing graph metadata
    /// (e.g. NormSpec.eps) to propagate through the template system.
    /// Falls back to 0.0 if the parameter is not set in the ParamTable.
    LoadParam { name: &'static str },
    /// Arithmetic → TraceOp::Add/Sub/Mul/Div/Fma/etc.
    BinOp { op: TraceBinOp, dst: &'static str, a: &'static str, b: &'static str },
    /// Unary → TraceOp::Exp/Sqrt/Rsqrt/Tanh/Sigmoid/etc.
    UnaryOp { op: TraceUnaryOp, dst: &'static str, src: &'static str },
    /// Multiply-accumulate: dst += a * b
    Fma { acc: &'static str, a: &'static str, b: &'static str },
    /// Horizontal reduce → TraceOp::HReduce
    HReduce { src: &'static str, op: ReduceKind },
    /// Broadcast scalar → TraceOp::BroadcastScalar
    Broadcast { src: &'static str, dst: &'static str },
    /// Vec load indexed → TraceOp::VecLoadIndexed
    VecLoadIndexed { base: &'static str, offset: &'static str },
    /// Vec store indexed → TraceOp::VecStoreIndexed
    VecStoreIndexed { base: &'static str, offset: &'static str, src: &'static str },
    /// Cast → TraceOp::Cast
    Cast { src: &'static str, from: QuantPrecision, to: QuantPrecision },
}

#[derive(Debug, Clone, Copy)]
pub enum TraceBinOp {
    Add, Sub, Mul, Div, Max, Min,
}

#[derive(Debug, Clone, Copy)]
pub enum TraceUnaryOp {
    Exp, Sqrt, Rsqrt, Tanh, Sigmoid, Neg, Abs, Recip, Log,
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceKind {
    Sum, Max, Min,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AlgoParam — parameter sources
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Parameter source — how a template parameter gets its value.
#[derive(Debug, Clone, Copy)]
pub enum AlgoParam {
    /// Compile-time constant.
    Const(usize),
    /// From PressureModel's blocking strategy (mc/nc/kc/mr/nr).
    FromPressureModel(&'static str),
    /// From DeviceProfile (simd_lanes, gemm_mr, gemm_nr, cache_sizes).
    FromDeviceProfile(&'static str),
    /// From CompilerGraph (m, n, k, hidden_dim, num_heads).
    FromGraph(&'static str),
    /// Derived from another parameter via arithmetic.
    Derived {
        base: &'static str,
        op: ParamArith,
        operand: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum ParamArith {
    CeilDiv,
    Mul,
    Div,
    Max,
    Min,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Auxiliary types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MatrixRole { A, B, C }

/// Device requirement — minimum device capability for a template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceReq {
    CpuAny,
    CpuAvx2,
    CpuAvx512,
    CpuAmx,
    CpuSme2,
    GpuSm70,
    GpuSm80,
    GpuSm90,
    GpuSm100,
}

impl DeviceReq {
    /// Priority for template selection: higher = more specialized = preferred.
    pub fn priority(&self) -> u32 {
        match self {
            Self::CpuAny => 0,
            Self::CpuAvx2 => 10,
            Self::CpuAvx512 => 20,
            Self::CpuAmx => 30,
            Self::CpuSme2 => 30,
            Self::GpuSm70 => 40,
            Self::GpuSm80 => 50,
            Self::GpuSm90 => 60,
            Self::GpuSm100 => 70,
        }
    }

    /// Whether a given profile satisfies this requirement.
    /// The profile must meet or exceed the requirement's capability level.
    pub fn is_satisfied_by(&self, profile_isa_level: u32) -> bool {
        profile_isa_level >= self.priority()
    }
}

/// Quantization block unpack mode — references SPEC 26's unified mode.
#[derive(Debug, Clone, Copy)]
pub enum BlockUnpackMode {
    Q4_0,
    Q4_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Mxfp4,
    Nvfp4,
    Awq4,
    Gptq4,
    IqSqueeze,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpilogueOp {
    BiasAdd,
    ResidualAdd,
    Relu,
    Silu,
    Gelu,
    RmsNorm { eps_param: &'static str },
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationKind { Relu, Silu, Gelu, Tanh, Sigmoid }

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp { Sum, Max, Min }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PackLayout { RowMajor, ColMajor }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MicroKernelDef
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Micro-kernel definition — mr × nr × kc inner loop.
/// The interpreter expands this into TraceOp sequences
/// (LoadARow/LoadBCol/Fma repeated mr×nr×k_step times).
pub struct MicroKernelDef {
    pub mr: &'static str,
    pub nr: &'static str,
    pub k_step: &'static str,
    pub steps: &'static [MicroKernelStep],
}

#[derive(Debug, Clone, Copy)]
pub enum MicroKernelStep {
    /// TraceOp::VecLoadIndexed (A panel row)
    LoadARow,
    /// TraceOp::VecLoadIndexed (B panel column)
    LoadBCol,
    /// TraceOp::Fma
    Fma,
    /// TraceOp::VecStoreIndexed (accumulator to output)
    StoreAccumulator,
    /// TraceOp::TileMma (GPU warp-level MMA)
    WarpMma,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_req_priority_ordering() {
        assert!(DeviceReq::CpuAny.priority() < DeviceReq::CpuAvx2.priority());
        assert!(DeviceReq::CpuAvx2.priority() < DeviceReq::CpuAvx512.priority());
        assert!(DeviceReq::CpuAvx512.priority() < DeviceReq::CpuAmx.priority());
        assert!(DeviceReq::CpuAmx.priority() == DeviceReq::CpuSme2.priority());
        assert!(DeviceReq::GpuSm70.priority() < DeviceReq::GpuSm80.priority());
        assert!(DeviceReq::GpuSm80.priority() < DeviceReq::GpuSm90.priority());
        assert!(DeviceReq::GpuSm90.priority() < DeviceReq::GpuSm100.priority());
    }

    #[test]
    fn device_req_satisfied_by_equal_or_higher() {
        assert!(DeviceReq::CpuAvx2.is_satisfied_by(DeviceReq::CpuAvx2.priority()));
        assert!(DeviceReq::CpuAvx2.is_satisfied_by(DeviceReq::CpuAvx512.priority()));
        assert!(!DeviceReq::CpuAvx512.is_satisfied_by(DeviceReq::CpuAvx2.priority()));
    }

    #[test]
    fn strategy_family_classification() {
        assert_eq!(AlgoStrategy::GemmBlis.family(), StrategyFamily::Gemm);
        assert_eq!(AlgoStrategy::AttnMha.family(), StrategyFamily::Attention);
        assert_eq!(AlgoStrategy::NormRms.family(), StrategyFamily::Norm);
    }

    #[test]
    fn algo_step_debug_format() {
        let step = AlgoStep::Loop {
            bound: "M",
            step: "1",
            body: &[AlgoStep::ZeroFill { bytes_param: "N" }],
        };
        let s = format!("{:?}", step);
        assert!(s.contains("Loop"));
    }

    // ── 5 new tests below ──────────────────────────────────────────────

    #[test]
    fn algo_strategy_all_families() {
        // Arrange: pick one representative from each family
        let cases: &[(AlgoStrategy, StrategyFamily)] = &[
            (AlgoStrategy::GemmNaive, StrategyFamily::Gemm),
            (AlgoStrategy::GemmGpuTiled, StrategyFamily::Gemm),
            (AlgoStrategy::AttnMha, StrategyFamily::Attention),
            (AlgoStrategy::AttnGqa, StrategyFamily::Attention),
            (AlgoStrategy::AttnMla, StrategyFamily::Attention),
            (AlgoStrategy::MoeRouterTopk, StrategyFamily::Moe),
            (AlgoStrategy::MoePackedDispatch, StrategyFamily::Moe),
            (AlgoStrategy::NormRms, StrategyFamily::Norm),
            (AlgoStrategy::NormLayer, StrategyFamily::Norm),
            (AlgoStrategy::RopeStandard, StrategyFamily::Rope),
            (AlgoStrategy::RopePartial, StrategyFamily::Rope),
            (AlgoStrategy::SamplingArgmax, StrategyFamily::Sampling),
            (AlgoStrategy::SamplingTemperature, StrategyFamily::Sampling),
            (AlgoStrategy::SamplingSoftmax, StrategyFamily::Sampling),
            (AlgoStrategy::SamplingTopK, StrategyFamily::Sampling),
            (AlgoStrategy::SamplingTopP, StrategyFamily::Sampling),
            (AlgoStrategy::SamplingMultinomial, StrategyFamily::Sampling),
            (AlgoStrategy::EmbeddingGather, StrategyFamily::Embedding),
            (AlgoStrategy::QuantGather, StrategyFamily::Quant),
        ];
        // Act & Assert
        for (strategy, expected_family) in cases {
            assert_eq!(strategy.family(), *expected_family,
                "{:?} should belong to {:?} family", strategy, expected_family);
        }
    }

    #[test]
    fn algo_param_derived_and_const_boundary() {
        // Arrange
        let const_max = AlgoParam::Const(usize::MAX);
        let const_zero = AlgoParam::Const(0);
        let derived = AlgoParam::Derived {
            base: "MC",
            op: ParamArith::CeilDiv,
            operand: 1,
        };

        // Act: verify Debug output contains variant names
        let max_dbg = format!("{:?}", const_max);
        let zero_dbg = format!("{:?}", const_zero);
        let derived_dbg = format!("{:?}", derived);

        // Assert
        assert!(max_dbg.contains("Const"), "Const variant debug must contain 'Const'");
        assert!(zero_dbg.contains("Const"), "zero-value Const must still be Const");
        assert!(derived_dbg.contains("Derived"), "Derived variant debug must contain 'Derived'");
        assert!(derived_dbg.contains("CeilDiv"), "Derived debug must include ParamArith");
    }

    #[test]
    fn param_arith_all_variants_clone_copy() {
        // Arrange
        let variants: Vec<ParamArith> = vec![
            ParamArith::CeilDiv,
            ParamArith::Mul,
            ParamArith::Div,
            ParamArith::Max,
            ParamArith::Min,
        ];
        // Act: clone each variant
        let cloned: Vec<ParamArith> = variants.iter().copied().collect();
        // Assert: Debug output round-trips
        for (original, copy) in variants.iter().zip(cloned.iter()) {
            assert_eq!(format!("{:?}", original), format!("{:?}", copy));
        }
    }

    #[test]
    fn matrix_role_ordering_and_equality() {
        // Arrange
        let a = MatrixRole::A;
        let b = MatrixRole::B;
        let c = MatrixRole::C;
        let a2 = MatrixRole::A;

        // Assert: equality
        assert_eq!(a, a2, "same MatrixRole variants must be equal");
        assert_ne!(a, b, "different MatrixRole variants must not be equal");

        // Assert: total ordering A < B < C
        assert!(a < b, "MatrixRole::A < B");
        assert!(b < c, "MatrixRole::B < C");
        assert!(a < c, "MatrixRole::A < C (transitivity)");
    }

    #[test]
    fn block_unpack_mode_all_variants_clone() {
        // Arrange
        let modes: Vec<BlockUnpackMode> = vec![
            BlockUnpackMode::Q4_0,
            BlockUnpackMode::Q4_1,
            BlockUnpackMode::Q8_0,
            BlockUnpackMode::Q8_1,
            BlockUnpackMode::Q2K,
            BlockUnpackMode::Q3K,
            BlockUnpackMode::Q4K,
            BlockUnpackMode::Q5K,
            BlockUnpackMode::Q6K,
            BlockUnpackMode::Mxfp4,
            BlockUnpackMode::Nvfp4,
            BlockUnpackMode::Awq4,
            BlockUnpackMode::Gptq4,
            BlockUnpackMode::IqSqueeze,
        ];
        // Act: clone
        let cloned: Vec<BlockUnpackMode> = modes.iter().cloned().collect();
        // Assert: each variant round-trips through Debug
        for (i, mode) in modes.iter().enumerate() {
            let orig_dbg = format!("{:?}", mode);
            let clone_dbg = format!("{:?}", cloned[i]);
            assert_eq!(orig_dbg, clone_dbg, "cloned BlockUnpackMode {:?} must match", mode);
        }
        assert_eq!(modes.len(), 14, "all 14 BlockUnpackMode variants covered");
    }

    #[test]
    fn epilogue_op_equality_and_field_access() {
        // Arrange
        let bias = EpilogueOp::BiasAdd;
        let residual = EpilogueOp::ResidualAdd;
        let relu = EpilogueOp::Relu;
        let silu = EpilogueOp::Silu;
        let gelu = EpilogueOp::Gelu;
        let rms = EpilogueOp::RmsNorm { eps_param: "EPS" };

        // Assert: PartialEq for simple variants
        assert_eq!(bias, EpilogueOp::BiasAdd);
        assert_ne!(bias, residual);
        assert_ne!(relu, silu);
        assert_ne!(silu, gelu);

        // Assert: RmsNorm with field
        match rms {
            EpilogueOp::RmsNorm { eps_param } => assert_eq!(eps_param, "EPS"),
            _ => panic!("expected RmsNorm variant"),
        }
    }

    #[test]
    fn activation_kind_all_variants_debug() {
        // Arrange
        let kinds: Vec<ActivationKind> = vec![
            ActivationKind::Relu,
            ActivationKind::Silu,
            ActivationKind::Gelu,
            ActivationKind::Tanh,
            ActivationKind::Sigmoid,
        ];
        // Act
        let dbg_strs: Vec<String> = kinds.iter().map(|k| format!("{:?}", k)).collect();
        // Assert: each debug string is non-empty and distinct
        for s in &dbg_strs {
            assert!(!s.is_empty(), "ActivationKind debug must not be empty");
        }
        assert_eq!(dbg_strs.len(), 5, "5 ActivationKind variants expected");
    }

    #[test]
    fn pack_layout_equality() {
        // Arrange
        let row = PackLayout::RowMajor;
        let col = PackLayout::ColMajor;

        // Assert
        assert_eq!(row, PackLayout::RowMajor);
        assert_eq!(col, PackLayout::ColMajor);
        assert_ne!(row, col, "RowMajor and ColMajor must differ");
    }

    #[test]
    fn micro_kernel_def_construction_and_field_access() {
        // Arrange
        let mk = MicroKernelDef {
            mr: "MR",
            nr: "NR",
            k_step: "KC",
            steps: &[
                MicroKernelStep::LoadARow,
                MicroKernelStep::LoadBCol,
                MicroKernelStep::Fma,
                MicroKernelStep::StoreAccumulator,
                MicroKernelStep::WarpMma,
            ],
        };

        // Act: field access
        let mr = mk.mr;
        let nr = mk.nr;
        let k_step = mk.k_step;
        let step_count = mk.steps.len();

        // Assert
        assert_eq!(mr, "MR");
        assert_eq!(nr, "NR");
        assert_eq!(k_step, "KC");
        assert_eq!(step_count, 5, "all 5 MicroKernelStep variants present");
    }

    #[test]
    fn algo_trace_step_float_precision_load_const() {
        // Arrange: f64 precision boundary values
        let zero = AlgoTraceStep::LoadConst { value: 0.0_f64 };
        let tiny = AlgoTraceStep::LoadConst { value: 1e-300_f64 };
        let large = AlgoTraceStep::LoadConst { value: 1e300_f64 };
        let neg = AlgoTraceStep::LoadConst { value: -3.141592653589793_f64 };
        let subnormal = AlgoTraceStep::LoadConst { value: f64::MIN_POSITIVE };

        // Act & Assert: extract values back
        if let AlgoTraceStep::LoadConst { value } = zero {
            assert_eq!(value, 0.0);
        }
        if let AlgoTraceStep::LoadConst { value } = tiny {
            assert!(value > 0.0, "tiny must be positive");
            assert!(value.is_normal() || value > 0.0, "subnormal or tiny");
        }
        if let AlgoTraceStep::LoadConst { value } = large {
            assert!(value.is_finite(), "1e300 must be finite");
        }
        if let AlgoTraceStep::LoadConst { value } = neg {
            assert!((value - (-std::f64::consts::PI)).abs() < 1e-10, "negative PI must roundtrip");
        }
        if let AlgoTraceStep::LoadConst { value } = subnormal {
            assert!(value > 0.0);
            assert!(value < 1e-300, "MIN_POSITIVE must be very small");
        }
    }

    #[test]
    fn algo_trace_step_binop_unaryop_variants() {
        // Arrange: all TraceBinOp variants
        let bin_ops: Vec<TraceBinOp> = vec![
            TraceBinOp::Add, TraceBinOp::Sub, TraceBinOp::Mul,
            TraceBinOp::Div, TraceBinOp::Max, TraceBinOp::Min,
        ];
        // Arrange: all TraceUnaryOp variants
        let unary_ops: Vec<TraceUnaryOp> = vec![
            TraceUnaryOp::Exp, TraceUnaryOp::Sqrt, TraceUnaryOp::Rsqrt,
            TraceUnaryOp::Tanh, TraceUnaryOp::Sigmoid, TraceUnaryOp::Neg,
            TraceUnaryOp::Abs, TraceUnaryOp::Recip, TraceUnaryOp::Log,
        ];
        // Assert: Debug clone round-trip
        for op in &bin_ops {
            let copy = *op;
            assert_eq!(format!("{:?}", op), format!("{:?}", copy));
        }
        for op in &unary_ops {
            let copy = *op;
            assert_eq!(format!("{:?}", op), format!("{:?}", copy));
        }
        assert_eq!(bin_ops.len(), 6, "6 TraceBinOp variants");
        assert_eq!(unary_ops.len(), 9, "9 TraceUnaryOp variants");
    }

    #[test]
    fn algo_template_struct_update_syntax() {
        // Arrange: base template
        let base = AlgoTemplate {
            name: "gemm_naive",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::MicroKernel],
            params: &[("MR", AlgoParam::Const(4)), ("NR", AlgoParam::Const(8))],
            micro_kernel: None,
        };

        // Act: struct update syntax overriding one field
        let updated = AlgoTemplate {
            strategy: AlgoStrategy::GemmBlis,
            ..base
        };

        // Assert: overridden field changed, others preserved
        assert_eq!(updated.strategy, AlgoStrategy::GemmBlis);
        assert_eq!(updated.name, "gemm_naive", "unchanged fields preserved");
        assert_eq!(updated.device_req, DeviceReq::CpuAny);
        assert_eq!(updated.steps.len(), 1);
        assert_eq!(updated.params.len(), 2);
        assert!(updated.micro_kernel.is_none());
    }

    #[test]
    fn device_req_all_variants_priority_monotonic() {
        // Arrange: all variants in declaration order
        let reqs: Vec<DeviceReq> = vec![
            DeviceReq::CpuAny,
            DeviceReq::CpuAvx2,
            DeviceReq::CpuAvx512,
            DeviceReq::CpuAmx,
            DeviceReq::CpuSme2,
            DeviceReq::GpuSm70,
            DeviceReq::GpuSm80,
            DeviceReq::GpuSm90,
            DeviceReq::GpuSm100,
        ];
        // Act: collect priorities
        let priorities: Vec<u32> = reqs.iter().map(|r| r.priority()).collect();
        // Assert: all non-negative
        for &p in &priorities {
            assert!(p < 100, "priority {} must be reasonable (< 100)", p);
        }
        // Assert: CpuAmx and CpuSme2 share same priority
        assert_eq!(priorities[3], priorities[4], "AMX and SME2 same priority");
        // Assert: strictly increasing (excluding AMX/SME2 tie)
        assert!(priorities[0] < priorities[1], "CpuAny < CpuAvx2");
        assert!(priorities[1] < priorities[2], "CpuAvx2 < CpuAvx512");
        assert!(priorities[5] < priorities[6], "GpuSm70 < GpuSm80");
        assert!(priorities[6] < priorities[7], "GpuSm80 < GpuSm90");
        assert!(priorities[7] < priorities[8], "GpuSm90 < GpuSm100");
    }

    #[test]
    fn reduce_kind_and_reduce_op_variants_debug() {
        // Arrange
        let reduce_kinds: Vec<ReduceKind> = vec![ReduceKind::Sum, ReduceKind::Max, ReduceKind::Min];
        let reduce_ops: Vec<ReduceOp> = vec![ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min];

        // Act & Assert: Debug clone round-trip for ReduceKind
        for k in &reduce_kinds {
            let copy = *k;
            assert_eq!(format!("{:?}", k), format!("{:?}", copy));
        }
        // Act & Assert: Debug clone round-trip for ReduceOp
        for op in &reduce_ops {
            let copy = *op;
            assert_eq!(format!("{:?}", op), format!("{:?}", copy));
        }
        assert_eq!(reduce_kinds.len(), 3);
        assert_eq!(reduce_ops.len(), 3);
    }

    // ── 10 additional tests ──────────────────────────────────────────

    #[test]
    fn algo_step_memory_and_gpu_variants_construction() {
        // Arrange: construct memory-operation AlgoStep variants
        let load_panel = AlgoStep::LoadPanel {
            matrix: MatrixRole::A,
            rows_param: "MC",
            cols_param: "KC",
        };
        let pack_buffer = AlgoStep::PackBuffer {
            buffer_name: "buf_a",
            rows_param: "MR",
            cols_param: "KC",
        };
        let store = AlgoStep::StoreResult {
            rows_param: "MR",
            cols_param: "NR",
        };
        let smem_decl = AlgoStep::SharedMemDeclare {
            name: "tile_a",
            size_param: "TILE_BYTES",
        };
        let async_copy = AlgoStep::AsyncCopyToSmem {
            buffer_name: "tile_b",
            size_param: "COPY_BYTES",
        };
        let async_wait = AlgoStep::AsyncWait { group: 2 };
        let barrier = AlgoStep::Barrier { barrier_name: "sync1" };
        let tile_cfg = AlgoStep::TileConfig { rows: "16", cols: "16" };
        let tile_mma = AlgoStep::TileMma;
        let tile_release = AlgoStep::TileRelease;

        // Act: Debug format each variant
        let cases: Vec<(&str, String)> = vec![
            ("LoadPanel", format!("{:?}", load_panel)),
            ("PackBuffer", format!("{:?}", pack_buffer)),
            ("StoreResult", format!("{:?}", store)),
            ("SharedMemDeclare", format!("{:?}", smem_decl)),
            ("AsyncCopyToSmem", format!("{:?}", async_copy)),
            ("AsyncWait", format!("{:?}", async_wait)),
            ("Barrier", format!("{:?}", barrier)),
            ("TileConfig", format!("{:?}", tile_cfg)),
            ("TileMma", format!("{:?}", tile_mma)),
            ("TileRelease", format!("{:?}", tile_release)),
        ];

        // Assert: each Debug output contains its variant name
        for (name, dbg_str) in &cases {
            assert!(dbg_str.contains(name), "{} debug must contain variant name, got: {}", name, dbg_str);
        }
    }

    #[test]
    fn algo_step_computation_variants_construction() {
        // Arrange: construct computation AlgoStep variants
        let cases: Vec<(&str, String)> = vec![
            ("Reduce", format!("{:?}", AlgoStep::Reduce { op: ReduceOp::Sum })),
            ("Activation", format!("{:?}", AlgoStep::Activation { kind: ActivationKind::Silu })),
            ("Softmax", format!("{:?}", AlgoStep::Softmax)),
            ("Dequantize", format!("{:?}", AlgoStep::Dequantize { mode: BlockUnpackMode::Awq4 })),
            ("EmbeddingGather", format!("{:?}", AlgoStep::EmbeddingGather)),
            ("MoeRouterGemv", format!("{:?}", AlgoStep::MoeRouterGemv { num_experts: "E", hidden: "H" })),
            ("MoeTopK", format!("{:?}", AlgoStep::MoeTopK { num_experts: "E", top_k: "K" })),
            ("Epilogue", format!("{:?}", AlgoStep::Epilogue { ops: &[EpilogueOp::BiasAdd, EpilogueOp::Silu] })),
            ("ZeroFill", format!("{:?}", AlgoStep::ZeroFill { bytes_param: "N_BYTES" })),
            ("RowCopy", format!("{:?}", AlgoStep::RowCopy { rows_param: "M", cols_param: "N" })),
        ];

        // Assert: each Debug output contains its variant name
        for (name, dbg_str) in &cases {
            assert!(dbg_str.contains(name), "{} debug must contain variant name, got: {}", name, dbg_str);
        }
    }

    #[test]
    fn algo_step_control_flow_seq_and_conditional() {
        // Arrange: &'static slices from promoted constants
        let seq = AlgoStep::Seq(&[
            AlgoStep::ZeroFill { bytes_param: "SIZE" },
            AlgoStep::MicroKernel,
        ]);
        let conditional = AlgoStep::Conditional {
            requirement: DeviceReq::GpuSm80,
            body: &[AlgoStep::TileMma, AlgoStep::TileRelease],
        };

        // Act: Debug format
        let seq_dbg = format!("{:?}", seq);
        let cond_dbg = format!("{:?}", conditional);

        // Assert: variant names present
        assert!(seq_dbg.contains("Seq"), "Seq debug must contain 'Seq'");
        assert!(seq_dbg.contains("ZeroFill"), "Seq body must contain 'ZeroFill'");
        assert!(seq_dbg.contains("MicroKernel"), "Seq body must contain 'MicroKernel'");
        assert!(cond_dbg.contains("Conditional"), "Conditional debug must contain 'Conditional'");
        assert!(cond_dbg.contains("GpuSm80"), "Conditional must reference requirement");
        assert!(cond_dbg.contains("TileMma"), "Conditional body must contain 'TileMma'");
    }

    #[test]
    fn algo_param_source_variants_debug() {
        // Arrange: all AlgoParam source variants
        let const_param = AlgoParam::Const(42);
        let pressure = AlgoParam::FromPressureModel("mc");
        let device = AlgoParam::FromDeviceProfile("simd_lanes");
        let graph = AlgoParam::FromGraph("m");
        let derived = AlgoParam::Derived {
            base: "MC",
            op: ParamArith::Mul,
            operand: 2,
        };

        // Act: Debug format each
        let const_dbg = format!("{:?}", const_param);
        let pressure_dbg = format!("{:?}", pressure);
        let device_dbg = format!("{:?}", device);
        let graph_dbg = format!("{:?}", graph);
        let derived_dbg = format!("{:?}", derived);

        // Assert: each contains its variant identifier
        assert!(const_dbg.contains("42"), "Const debug must include value");
        assert!(pressure_dbg.contains("FromPressureModel"), "must contain variant name");
        assert!(pressure_dbg.contains("mc"), "must include param name");
        assert!(device_dbg.contains("FromDeviceProfile"), "must contain variant name");
        assert!(device_dbg.contains("simd_lanes"), "must include param name");
        assert!(graph_dbg.contains("FromGraph"), "must contain variant name");
        assert!(graph_dbg.contains("\"m\""), "must include param name");
        assert!(derived_dbg.contains("Derived"), "must contain variant name");
        assert!(derived_dbg.contains("Mul"), "must include ParamArith variant");
    }

    #[test]
    fn algo_trace_step_all_variants_construction() {
        // Arrange: construct one of each AlgoTraceStep variant
        let cases: Vec<(&str, String)> = vec![
            ("LoadInput", format!("{:?}", AlgoTraceStep::LoadInput { name: "x" })),
            ("BinOp", format!("{:?}", AlgoTraceStep::BinOp {
                op: TraceBinOp::Add, dst: "out", a: "x", b: "y",
            })),
            ("UnaryOp", format!("{:?}", AlgoTraceStep::UnaryOp {
                op: TraceUnaryOp::Exp, dst: "exp_x", src: "x",
            })),
            ("Fma", format!("{:?}", AlgoTraceStep::Fma { acc: "acc", a: "a_val", b: "b_val" })),
            ("HReduce", format!("{:?}", AlgoTraceStep::HReduce { src: "vec", op: ReduceKind::Sum })),
            ("Broadcast", format!("{:?}", AlgoTraceStep::Broadcast { src: "scalar", dst: "vec" })),
            ("VecLoadIndexed", format!("{:?}", AlgoTraceStep::VecLoadIndexed { base: "table", offset: "idx" })),
            ("VecStoreIndexed", format!("{:?}", AlgoTraceStep::VecStoreIndexed {
                base: "out_buf", offset: "idx", src: "val",
            })),
        ];

        // Assert: each debug string contains its variant name
        for (name, dbg_str) in &cases {
            assert!(dbg_str.contains(name), "{} debug must contain variant name, got: {}", name, dbg_str);
        }
    }

    #[test]
    fn algo_trace_step_cast_variant_preserves_dtypes() {
        // Arrange
        let cast_step = AlgoTraceStep::Cast {
            src: "bf16_vec",
            from: QuantPrecision::BF16,
            to: QuantPrecision::F32,
        };

        // Act
        let dbg_str = format!("{:?}", cast_step);

        // Assert: debug contains variant and source name
        assert!(dbg_str.contains("Cast"), "must contain variant name");
        assert!(dbg_str.contains("bf16_vec"), "must contain source name");

        // Assert: extract fields back via pattern match
        if let AlgoTraceStep::Cast { from, to, .. } = cast_step {
            assert_eq!(from, QuantPrecision::BF16, "from QuantPrecision::F32 must be BF16");
            assert_eq!(to, QuantPrecision::F32, "to QuantPrecision::F32 must be F32");
        } else {
            panic!("expected Cast variant");
        }
    }

    #[test]
    fn strategy_family_all_variants_equality() {
        // Arrange: all 8 StrategyFamily variants
        let families = [
            StrategyFamily::Gemm,
            StrategyFamily::Attention,
            StrategyFamily::Moe,
            StrategyFamily::Norm,
            StrategyFamily::Rope,
            StrategyFamily::Sampling,
            StrategyFamily::Embedding,
            StrategyFamily::Quant,
        ];

        // Act & Assert: each variant equals itself, differs from all others
        for (i, fam) in families.iter().enumerate() {
            assert_eq!(*fam, families[i], "same index must be equal");
            for (j, other) in families.iter().enumerate() {
                if i != j {
                    assert_ne!(*fam, *other, "different indices must not be equal: {:?} vs {:?}", fam, other);
                }
            }
        }
        assert_eq!(families.len(), 8, "8 StrategyFamily variants expected");
    }

    #[test]
    fn device_req_satisfied_by_boundary_exact() {
        // Arrange: each DeviceReq variant and its exact priority
        let cases: Vec<(DeviceReq, u32)> = vec![
            (DeviceReq::CpuAny, DeviceReq::CpuAny.priority()),
            (DeviceReq::CpuAvx2, DeviceReq::CpuAvx2.priority()),
            (DeviceReq::CpuAvx512, DeviceReq::CpuAvx512.priority()),
            (DeviceReq::GpuSm80, DeviceReq::GpuSm80.priority()),
            (DeviceReq::GpuSm100, DeviceReq::GpuSm100.priority()),
        ];

        // Act & Assert: exact boundary satisfies, one below does not
        for (req, priority) in &cases {
            assert!(req.is_satisfied_by(*priority),
                "{:?} must be satisfied by its own priority {}", req, priority);
            if *priority > 0 {
                assert!(!req.is_satisfied_by(priority - 1),
                    "{:?} must NOT be satisfied by priority one below ({})", req, priority - 1);
            }
        }
    }

    #[test]
    fn algo_step_trace_body_with_embedded_steps() {
        // Arrange: TraceBody containing a sequence of AlgoTraceStep
        let trace_body = AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "hidden" },
            AlgoTraceStep::BinOp {
                op: TraceBinOp::Mul,
                dst: "scaled",
                a: "hidden",
                b: "weight",
            },
            AlgoTraceStep::UnaryOp {
                op: TraceUnaryOp::Sigmoid,
                dst: "activated",
                src: "scaled",
            },
        ]);

        // Act: Debug format
        let dbg_str = format!("{:?}", trace_body);

        // Assert: TraceBody wrapper present
        assert!(dbg_str.contains("TraceBody"), "must contain TraceBody");
        assert!(dbg_str.contains("LoadInput"), "must contain LoadInput step");
        assert!(dbg_str.contains("BinOp"), "must contain BinOp step");
        assert!(dbg_str.contains("UnaryOp"), "must contain UnaryOp step");
        assert!(dbg_str.contains("Sigmoid"), "must contain Sigmoid op");

        // Assert: clone round-trip
        let cloned = trace_body.clone();
        assert_eq!(format!("{:?}", trace_body), format!("{:?}", cloned),
            "cloned TraceBody must produce identical Debug output");
    }

    #[test]
    fn algo_step_nested_loop_tree() {
        // Arrange: nested loop with Seq body containing multiple steps
        let kc_loop = AlgoStep::Loop {
            bound: "KC",
            step: "K_STEP",
            body: &[
                AlgoStep::LoadPanel {
                    matrix: MatrixRole::A,
                    rows_param: "MR",
                    cols_param: "KC",
                },
                AlgoStep::MicroKernel,
                AlgoStep::StoreResult {
                    rows_param: "MR",
                    cols_param: "NR",
                },
            ],
        };

        // Act: Debug format the loop
        let dbg_str = format!("{:?}", kc_loop);

        // Assert: nesting structure visible in debug output
        assert!(dbg_str.contains("Loop"), "must contain Loop");
        assert!(dbg_str.contains("LoadPanel"), "must contain inner LoadPanel");
        assert!(dbg_str.contains("MicroKernel"), "must contain MicroKernel");
        assert!(dbg_str.contains("StoreResult"), "must contain StoreResult");

        // Assert: clone round-trip preserves entire tree
        let cloned = kc_loop.clone();
        assert_eq!(dbg_str, format!("{:?}", cloned),
            "nested loop clone must preserve full tree structure");
    }

    // ── 10 additional edge-case tests ────────────────────────────────

    #[test]
    fn empty_template_zero_steps_zero_params() {
        // Arrange: minimal template with no steps and no parameters
        let empty = AlgoTemplate {
            name: "empty",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[],
            params: &[],
            micro_kernel: None,
        };

        // Assert: empty collections are valid
        assert_eq!(empty.steps.len(), 0, "empty template must have zero steps");
        assert_eq!(empty.params.len(), 0, "empty template must have zero params");
        assert!(empty.micro_kernel.is_none(), "empty template has no micro-kernel");
        assert_eq!(empty.name, "empty");
    }

    #[test]
    fn deeply_nested_loop_three_levels() {
        // Arrange: 3-level loop nesting (M > N > K) using promoted statics
        const K_BODY: &[AlgoStep] = &[AlgoStep::MicroKernel];
        const K_LOOP: AlgoStep = AlgoStep::Loop {
            bound: "K",
            step: "K_STEP",
            body: K_BODY,
        };
        const N_BODY: &[AlgoStep] = &[K_LOOP];
        const N_LOOP: AlgoStep = AlgoStep::Loop {
            bound: "N",
            step: "NR",
            body: N_BODY,
        };
        const M_BODY: &[AlgoStep] = &[N_LOOP];
        const M_LOOP: AlgoStep = AlgoStep::Loop {
            bound: "M",
            step: "MR",
            body: M_BODY,
        };

        // Act
        let dbg_str = format!("{:?}", M_LOOP);

        // Assert: all three loop levels appear in debug output
        assert!(dbg_str.contains("Loop"), "must contain Loop variants");
        // Count occurrences of "bound" to verify nesting depth
        let bound_count = dbg_str.matches("bound").count();
        assert!(bound_count >= 3, "3-level nesting must have at least 3 bound fields, got {}", bound_count);
    }

    #[test]
    fn multi_step_seq_with_mixed_variants() {
        // Arrange: Seq containing diverse AlgoStep variants
        let seq = AlgoStep::Seq(&[
            AlgoStep::LoadPanel { matrix: MatrixRole::B, rows_param: "NC", cols_param: "KC" },
            AlgoStep::Activation { kind: ActivationKind::Gelu },
            AlgoStep::Reduce { op: ReduceOp::Max },
            AlgoStep::Softmax,
            AlgoStep::ZeroFill { bytes_param: "PAD" },
        ]);

        // Act
        let dbg_str = format!("{:?}", seq);

        // Assert: all five step types appear
        assert!(dbg_str.contains("LoadPanel"), "Seq must contain LoadPanel");
        assert!(dbg_str.contains("Activation"), "Seq must contain Activation");
        assert!(dbg_str.contains("Reduce"), "Seq must contain Reduce");
        assert!(dbg_str.contains("Softmax"), "Seq must contain Softmax");
        assert!(dbg_str.contains("ZeroFill"), "Seq must contain ZeroFill");
        assert_eq!(dbg_str.matches("Gelu").count(), 1, "Gelu appears exactly once");
    }

    #[test]
    fn param_derived_all_arith_ops() {
        // Arrange: one Derived param per ParamArith variant
        let cases: Vec<(ParamArith, &str)> = vec![
            (ParamArith::CeilDiv, "CeilDiv"),
            (ParamArith::Mul, "Mul"),
            (ParamArith::Div, "Div"),
            (ParamArith::Max, "Max"),
            (ParamArith::Min, "Min"),
        ];

        // Act & Assert: each arith variant produces debug containing its name
        for (arith, expected_name) in &cases {
            let param = AlgoParam::Derived { base: "X", op: *arith, operand: 4 };
            let dbg_str = format!("{:?}", param);
            assert!(dbg_str.contains(expected_name),
                "Derived with {:?} must contain '{}' in debug, got: {}", arith, expected_name, dbg_str);
        }
    }

    #[test]
    fn param_const_boundary_values() {
        // Arrange: boundary usize values
        let zero = AlgoParam::Const(0);
        let one = AlgoParam::Const(1);
        let max_val = AlgoParam::Const(usize::MAX);

        // Act & Assert: debug output contains numeric value
        assert!(format!("{:?}", zero).contains("0"), "Const(0) must contain '0'");
        assert!(format!("{:?}", one).contains("1"), "Const(1) must contain '1'");
        let max_dbg = format!("{:?}", max_val);
        assert!(max_dbg.contains(&usize::MAX.to_string()),
            "Const(MAX) must contain max value, got: {}", max_dbg);
    }

    #[test]
    fn algo_strategy_debug_trait_all_variants() {
        // Arrange: all AlgoStrategy variants
        let strategies: Vec<AlgoStrategy> = vec![
            AlgoStrategy::GemmNaive, AlgoStrategy::GemmBlis,
            AlgoStrategy::GemmGpuTiled, AlgoStrategy::GemmGpuPipelined,
            AlgoStrategy::GemmHardwareTile,
            AlgoStrategy::AttnMha, AlgoStrategy::AttnGqa, AlgoStrategy::AttnMla,
            AlgoStrategy::MoeRouterTopk, AlgoStrategy::MoePackedDispatch,
            AlgoStrategy::NormRms, AlgoStrategy::NormLayer,
            AlgoStrategy::RopeStandard, AlgoStrategy::RopePartial,
            AlgoStrategy::SamplingArgmax, AlgoStrategy::SamplingTemperature,
            AlgoStrategy::SamplingSoftmax, AlgoStrategy::SamplingTopK,
            AlgoStrategy::SamplingTopP, AlgoStrategy::SamplingMultinomial,
            AlgoStrategy::EmbeddingGather, AlgoStrategy::QuantGather,
        ];

        // Act: format each with Debug
        let debug_strs: Vec<String> = strategies.iter().map(|s| format!("{:?}", s)).collect();

        // Assert: all non-empty and pairwise distinct
        for s in &debug_strs {
            assert!(!s.is_empty(), "AlgoStrategy debug must not be empty");
        }
        assert_eq!(debug_strs.len(), 22, "all 22 AlgoStrategy variants covered");
        // Verify pairwise distinctness
        for i in 0..debug_strs.len() {
            for j in (i + 1)..debug_strs.len() {
                assert_ne!(debug_strs[i], debug_strs[j],
                    "AlgoStrategy variants {} and {} must have distinct debug output", i, j);
            }
        }
    }

    #[test]
    fn micro_kernel_step_all_variants_clone_copy() {
        // Arrange: all MicroKernelStep variants
        let steps: Vec<MicroKernelStep> = vec![
            MicroKernelStep::LoadARow,
            MicroKernelStep::LoadBCol,
            MicroKernelStep::Fma,
            MicroKernelStep::StoreAccumulator,
            MicroKernelStep::WarpMma,
        ];

        // Act: copy (Copy trait) and format
        let copies: Vec<(MicroKernelStep, String)> = steps.iter()
            .map(|&s| (s, format!("{:?}", s)))
            .collect();

        // Assert: debug output contains variant name and copies match
        assert_eq!(copies.len(), 5, "5 MicroKernelStep variants");
        for (i, (original, dbg_str)) in copies.iter().enumerate() {
            let copy_dbg = format!("{:?}", steps[i]);
            assert_eq!(*dbg_str, copy_dbg, "copy must match original debug");
            assert!(!dbg_str.is_empty(), "debug must not be empty");
        }
    }

    #[test]
    fn epilogue_op_rms_norm_with_different_eps_params() {
        // Arrange: two RmsNorm with different eps_param values
        let eps_small = EpilogueOp::RmsNorm { eps_param: "EPS_1e5" };
        let eps_large = EpilogueOp::RmsNorm { eps_param: "EPS_1e3" };

        // Act: extract params
        let small_param = match eps_small {
            EpilogueOp::RmsNorm { eps_param } => eps_param,
            _ => panic!("expected RmsNorm"),
        };
        let large_param = match eps_large {
            EpilogueOp::RmsNorm { eps_param } => eps_param,
            _ => panic!("expected RmsNorm"),
        };

        // Assert: params are distinct
        assert_ne!(small_param, large_param,
            "different eps_param values must not be equal");
        assert_eq!(small_param, "EPS_1e5");
        assert_eq!(large_param, "EPS_1e3");
    }

    #[test]
    fn conditional_step_device_requirement_check() {
        // Arrange: conditional step requiring GPU SM80
        let cond = AlgoStep::Conditional {
            requirement: DeviceReq::GpuSm80,
            body: &[AlgoStep::TileMma, AlgoStep::TileRelease],
        };

        // Act
        let dbg_str = format!("{:?}", cond);

        // Assert: debug references both the requirement and body steps
        assert!(dbg_str.contains("Conditional"), "must contain Conditional");
        assert!(dbg_str.contains("GpuSm80"), "must contain device requirement");
        assert!(dbg_str.contains("TileMma"), "body must contain TileMma");
        assert!(dbg_str.contains("TileRelease"), "body must contain TileRelease");

        // Assert: is_satisfied_by works for the requirement
        assert!(!DeviceReq::GpuSm80.is_satisfied_by(DeviceReq::GpuSm70.priority()),
            "SM70 must not satisfy SM80 requirement");
        assert!(DeviceReq::GpuSm80.is_satisfied_by(DeviceReq::GpuSm90.priority()),
            "SM90 must satisfy SM80 requirement");
    }

    #[test]
    fn algo_step_row_copy_and_embedding_gather_clone() {
        // Arrange
        let row_copy = AlgoStep::RowCopy { rows_param: "M", cols_param: "N" };
        let gather = AlgoStep::EmbeddingGather;

        // Act: clone both
        let row_copy_cloned = row_copy.clone();
        let gather_cloned = gather.clone();

        // Assert: clone preserves debug output
        assert_eq!(
            format!("{:?}", row_copy),
            format!("{:?}", row_copy_cloned),
            "RowCopy clone must preserve debug output",
        );
        assert_eq!(
            format!("{:?}", gather),
            format!("{:?}", gather_cloned),
            "EmbeddingGather clone must preserve debug output",
        );

        // Assert: debug contains variant names
        assert!(format!("{:?}", row_copy).contains("RowCopy"));
        assert!(format!("{:?}", gather).contains("EmbeddingGather"));
    }

    // ── 10 more tests ────────────────────────────────────────────────

    #[test]
    fn algo_template_with_micro_kernel_some_and_none() {
        // Arrange: const MicroKernelDef (required for 'static reference)
        const MK_STEPS: &[MicroKernelStep] = &[MicroKernelStep::LoadARow, MicroKernelStep::Fma];
        const MK: MicroKernelDef = MicroKernelDef {
            mr: "MR",
            nr: "NR",
            k_step: "KC",
            steps: MK_STEPS,
        };
        let with_mk = AlgoTemplate {
            name: "gemm_blis_mk",
            strategy: AlgoStrategy::GemmBlis,
            device_req: DeviceReq::CpuAvx512,
            steps: &[AlgoStep::MicroKernel],
            params: &[("MR", AlgoParam::FromPressureModel("mr"))],
            micro_kernel: Some(&MK),
        };
        // Arrange: template without a micro-kernel
        let without_mk = AlgoTemplate {
            name: "softmax_only",
            strategy: AlgoStrategy::SamplingSoftmax,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Softmax],
            params: &[],
            micro_kernel: None,
        };

        // Assert: Some variant is present and fields are accessible
        assert!(with_mk.micro_kernel.is_some(), "template with micro-kernel must be Some");
        let mk_ref = with_mk.micro_kernel.unwrap();
        assert_eq!(mk_ref.mr, "MR");
        assert_eq!(mk_ref.nr, "NR");
        assert_eq!(mk_ref.steps.len(), 2, "micro-kernel must have 2 steps");

        // Assert: None variant
        assert!(without_mk.micro_kernel.is_none(), "template without micro-kernel must be None");
        assert_eq!(without_mk.params.len(), 0, "no-params template");
    }

    #[test]
    fn algo_param_from_pressure_model_device_and_graph_debug() {
        // Arrange: non-Const AlgoParam variants carry string references
        let params: Vec<AlgoParam> = vec![
            AlgoParam::FromPressureModel("mc"),
            AlgoParam::FromPressureModel("nr"),
            AlgoParam::FromDeviceProfile("gemm_mr"),
            AlgoParam::FromDeviceProfile("cache_l1"),
            AlgoParam::FromGraph("m"),
            AlgoParam::FromGraph("hidden_dim"),
        ];

        // Act: format each as debug
        let debug_strs: Vec<String> = params.iter().map(|p| format!("{:?}", p)).collect();

        // Assert: each debug string contains its string parameter name
        assert!(debug_strs[0].contains("mc"), "FromPressureModel(mc) must contain 'mc'");
        assert!(debug_strs[1].contains("nr"), "FromPressureModel(nr) must contain 'nr'");
        assert!(debug_strs[2].contains("gemm_mr"), "FromDeviceProfile must contain param name");
        assert!(debug_strs[3].contains("cache_l1"), "FromDeviceProfile must contain param name");
        assert!(debug_strs[4].contains("m"), "FromGraph must contain param name");
        assert!(debug_strs[5].contains("hidden_dim"), "FromGraph must contain param name");

        // Assert: all debug strings are pairwise distinct (different param names)
        for i in 0..debug_strs.len() {
            for j in (i + 1)..debug_strs.len() {
                assert_ne!(debug_strs[i], debug_strs[j],
                    "params with different names must have different debug output: {} vs {}", i, j);
            }
        }
    }

    #[test]
    fn algo_step_loop_bound_and_step_parameter_names() {
        // Arrange: loop referencing named parameters
        let loop_step = AlgoStep::Loop {
            bound: "TOTAL_SEQ",
            step: "CHUNK_SIZE",
            body: &[AlgoStep::Softmax, AlgoStep::Activation { kind: ActivationKind::Relu }],
        };

        // Act
        let dbg_str = format!("{:?}", loop_step);

        // Assert: parameter names appear in debug output
        assert!(dbg_str.contains("TOTAL_SEQ"), "bound parameter name must appear in debug");
        assert!(dbg_str.contains("CHUNK_SIZE"), "step parameter name must appear in debug");
        assert!(dbg_str.contains("Softmax"), "body step must appear in debug");
        assert!(dbg_str.contains("Activation"), "body step must appear in debug");
        assert!(dbg_str.contains("Relu"), "activation kind must appear in debug");
    }

    #[test]
    fn algo_step_moe_router_and_topk_field_access() {
        // Arrange: MoE-related AlgoStep variants
        let gemv = AlgoStep::MoeRouterGemv {
            num_experts: "NUM_EXPERTS",
            hidden: "HIDDEN_DIM",
        };
        let topk = AlgoStep::MoeTopK {
            num_experts: "NUM_EXPERTS",
            top_k: "TOP_K",
        };

        // Act: extract fields via pattern match
        let (ne_gemv, h) = match &gemv {
            AlgoStep::MoeRouterGemv { num_experts, hidden } => (*num_experts, *hidden),
            _ => panic!("expected MoeRouterGemv"),
        };
        let (ne_topk, tk) = match &topk {
            AlgoStep::MoeTopK { num_experts, top_k } => (*num_experts, *top_k),
            _ => panic!("expected MoeTopK"),
        };

        // Assert: fields correctly stored and retrieved
        assert_eq!(ne_gemv, "NUM_EXPERTS");
        assert_eq!(h, "HIDDEN_DIM");
        assert_eq!(ne_topk, "NUM_EXPERTS");
        assert_eq!(tk, "TOP_K");
    }

    #[test]
    fn algo_trace_step_load_const_negative_and_special_floats() {
        // Arrange: LoadConst with negative, infinity, and NaN-edge values
        let neg_one = AlgoTraceStep::LoadConst { value: -1.0 };
        let large_neg = AlgoTraceStep::LoadConst { value: -f64::MAX };
        let inf = AlgoTraceStep::LoadConst { value: f64::INFINITY };
        let neg_inf = AlgoTraceStep::LoadConst { value: f64::NEG_INFINITY };

        // Act & Assert: extract and verify values
        if let AlgoTraceStep::LoadConst { value } = neg_one {
            assert!(value < 0.0, "negative one must be negative");
            assert!((value - (-1.0)).abs() < 1e-15);
        }
        if let AlgoTraceStep::LoadConst { value } = large_neg {
            assert!(value < 0.0, "large negative must be negative");
            assert!(value.is_finite(), "f64::MAX must be finite");
        }
        if let AlgoTraceStep::LoadConst { value } = inf {
            assert!(value.is_infinite() && value > 0.0, "must be positive infinity");
        }
        if let AlgoTraceStep::LoadConst { value } = neg_inf {
            assert!(value.is_infinite() && value < 0.0, "must be negative infinity");
        }
    }

    #[test]
    fn algo_trace_step_hreduce_and_broadcast_field_access() {
        // Arrange: AlgoTraceStep with HReduce and Broadcast
        let hreduce = AlgoTraceStep::HReduce { src: "attention_scores", op: ReduceKind::Max };
        let broadcast = AlgoTraceStep::Broadcast { src: "scalar_max", dst: "broadcast_max" };

        // Act: extract fields
        let (src, op) = match &hreduce {
            AlgoTraceStep::HReduce { src, op } => (*src, *op),
            _ => panic!("expected HReduce"),
        };
        let (b_src, b_dst) = match &broadcast {
            AlgoTraceStep::Broadcast { src, dst } => (*src, *dst),
            _ => panic!("expected Broadcast"),
        };

        // Assert: fields match
        assert_eq!(src, "attention_scores");
        assert!(matches!(op, ReduceKind::Max));
        assert_eq!(b_src, "scalar_max");
        assert_eq!(b_dst, "broadcast_max");
    }

    #[test]
    fn device_req_cpu_families_lower_priority_than_gpu() {
        // Arrange: collect CPU and GPU priorities
        let cpu_reqs = [DeviceReq::CpuAny, DeviceReq::CpuAvx2, DeviceReq::CpuAvx512, DeviceReq::CpuAmx, DeviceReq::CpuSme2];
        let gpu_reqs = [DeviceReq::GpuSm70, DeviceReq::GpuSm80, DeviceReq::GpuSm90, DeviceReq::GpuSm100];

        // Act: get max CPU priority and min GPU priority
        let max_cpu = cpu_reqs.iter().map(|r| r.priority()).max().unwrap();
        let min_gpu = gpu_reqs.iter().map(|r| r.priority()).min().unwrap();

        // Assert: all CPU priorities are strictly below all GPU priorities
        assert!(max_cpu < min_gpu,
            "max CPU priority ({}) must be below min GPU priority ({})", max_cpu, min_gpu);
    }

    #[test]
    fn algo_step_epilogue_with_multiple_ops() {
        // Arrange: Epilogue with a chain of operations
        let epilogue = AlgoStep::Epilogue {
            ops: &[
                EpilogueOp::BiasAdd,
                EpilogueOp::ResidualAdd,
                EpilogueOp::RmsNorm { eps_param: "EPS_1e6" },
                EpilogueOp::Silu,
            ],
        };

        // Act
        let dbg_str = format!("{:?}", epilogue);

        // Assert: all epilogue ops appear in debug
        assert!(dbg_str.contains("Epilogue"), "must contain Epilogue wrapper");
        assert!(dbg_str.contains("BiasAdd"), "must contain BiasAdd");
        assert!(dbg_str.contains("ResidualAdd"), "must contain ResidualAdd");
        assert!(dbg_str.contains("RmsNorm"), "must contain RmsNorm");
        assert!(dbg_str.contains("EPS_1e6"), "must contain eps_param name");
        assert!(dbg_str.contains("Silu"), "must contain Silu");

        // Assert: extract ops length via match
        if let AlgoStep::Epilogue { ops } = epilogue {
            assert_eq!(ops.len(), 4, "epilogue must have 4 operations");
        }
    }

    #[test]
    fn algo_step_dequantize_all_block_unpack_modes() {
        // Arrange: construct Dequantize step for each BlockUnpackMode variant
        let modes: Vec<BlockUnpackMode> = vec![
            BlockUnpackMode::Q4_0, BlockUnpackMode::Q4_1,
            BlockUnpackMode::Q8_0, BlockUnpackMode::Q8_1,
            BlockUnpackMode::Q2K, BlockUnpackMode::Q3K,
            BlockUnpackMode::Q4K, BlockUnpackMode::Q5K,
            BlockUnpackMode::Q6K, BlockUnpackMode::Mxfp4,
            BlockUnpackMode::Nvfp4, BlockUnpackMode::Awq4,
            BlockUnpackMode::Gptq4, BlockUnpackMode::IqSqueeze,
        ];

        // Act & Assert: each mode can be used in a Dequantize step and produces valid debug
        for mode in &modes {
            let step = AlgoStep::Dequantize { mode: *mode };
            let dbg_str = format!("{:?}", step);
            assert!(dbg_str.contains("Dequantize"),
                "Dequantize with {:?} must contain 'Dequantize' in debug", mode);
            assert!(!dbg_str.is_empty(),
                "debug for {:?} must not be empty", mode);
        }
        assert_eq!(modes.len(), 14, "all 14 BlockUnpackMode variants tested");
    }

    #[test]
    fn algo_template_params_lookup_by_name() {
        // Arrange: template with several named parameters
        let params: &[(&str, AlgoParam)] = &[
            ("M", AlgoParam::FromGraph("m")),
            ("N", AlgoParam::FromGraph("n")),
            ("K", AlgoParam::FromGraph("k")),
            ("MR", AlgoParam::FromPressureModel("mr")),
            ("NR", AlgoParam::FromPressureModel("nr")),
            ("KC", AlgoParam::Derived { base: "K", op: ParamArith::CeilDiv, operand: 4 }),
            ("TILE_SIZE", AlgoParam::Const(256)),
        ];
        let template = AlgoTemplate {
            name: "gemm_param_test",
            strategy: AlgoStrategy::GemmBlis,
            device_req: DeviceReq::CpuAvx2,
            steps: &[AlgoStep::MicroKernel],
            params,
            micro_kernel: None,
        };

        // Act: look up parameters by name
        let m_param = template.params.iter().find(|(name, _)| *name == "M");
        let kc_param = template.params.iter().find(|(name, _)| *name == "KC");
        let tile_param = template.params.iter().find(|(name, _)| *name == "TILE_SIZE");
        let missing = template.params.iter().find(|(name, _)| *name == "NONEXISTENT");

        // Assert: found parameters have correct values
        assert!(m_param.is_some(), "M parameter must exist");
        assert_eq!(m_param.unwrap().0, "M");
        if let AlgoParam::FromGraph(graph_name) = m_param.unwrap().1 {
            assert_eq!(graph_name, "m");
        } else {
            panic!("M must be FromGraph");
        }

        assert!(kc_param.is_some(), "KC parameter must exist");
        if let AlgoParam::Derived { base, op, operand } = kc_param.unwrap().1 {
            assert_eq!(base, "K");
            assert!(matches!(op, ParamArith::CeilDiv));
            assert_eq!(operand, 4);
        } else {
            panic!("KC must be Derived");
        }

        assert!(tile_param.is_some(), "TILE_SIZE parameter must exist");
        if let AlgoParam::Const(val) = tile_param.unwrap().1 {
            assert_eq!(val, 256);
        } else {
            panic!("TILE_SIZE must be Const");
        }

        assert!(missing.is_none(), "NONEXISTENT must not be found");
        assert_eq!(template.params.len(), 7, "template must have 7 parameters");
    }
}
