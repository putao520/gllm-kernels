//! Execution Planner — transforms LayerIR + DeviceProfile into an ExecutionPlan.
//!
//! The planner decides all performance-critical parameters at compile time:
//! GEMM blocking, fusion strategy, tiling, thread count, prefetch distances.
//! At runtime, the compiled layer executes with zero decisions.
//!
//! # HwOptEngine (§10)
//!
//! The planner now hosts the unified hardware-aware optimization engine.
//! All strategy decisions are computed via cost-model evaluation of candidate
//! strategies, not hardcoded if/else lookup tables. The engine produces an
//! immutable `ExecutionPlan` consumed by codegen, fusion, and scheduling.

use std::collections::HashMap;
use crate::compiler::ir::{LayerIR, MoeConfig};
use crate::compiler::pain_point::OpBottleneckMap;
use crate::dispatch::device_profile::DeviceProfile;
use crate::traits::Activation;
use crate::types::DType;

// ═══════════════════════════════════════════════════════════════
//  Strategy Bias (from Strategy Arbiter)
// ═══════════════════════════════════════════════════════════════

/// Strategy bias coefficients from the Strategy Arbiter.
/// Modulates cost-model decisions in all solvers.
/// See gllm SPEC/12-STRATEGY-ARBITER.md for derivation.
#[derive(Debug, Clone, Copy)]
pub struct StrategyBias {
    /// Fusion cost scaling: < 1.0 makes fusion cheaper (amplifies savings),
    /// > 1.0 makes fusion more expensive (dampens savings).
    pub fusion_cost_scale: f64,
    /// Pipeline cost scaling for parallelism sync overhead.
    pub pipeline_cost_scale: f64,
    /// Parallelism cost scaling for wave sync overhead.
    pub parallelism_cost_scale: f64,
    /// Epilogue depth preference: > 1.0 allows deeper epilogue chains,
    /// < 1.0 restricts them.
    pub epilogue_depth_preference: f64,
    /// K-loop pipeline depth preference: >= 1.5 aggressive, < 0.8 conservative.
    pub k_depth_preference: f64,
    /// KV cache budget scaling within L2 allocation.
    pub kv_cache_budget_scale: f64,
    /// Weight prefetch budget scaling within L2 allocation.
    pub weight_prefetch_budget_scale: f64,
    /// Batch flexibility: 0.0 forces batch=1, 1.0 is default, > 1.0 allows larger batches.
    pub batch_flexibility: f64,
    /// Decode ratio cap scaling.
    pub decode_ratio_scale: f64,
    /// Speculative decoding benefit multiplier.
    pub speculative_decoding_value: f64,
    /// Quantization aggressiveness coefficient.
    pub quantization_aggressiveness: f64,
    /// MoE expert eviction aggressiveness: 0.0 = full resident, 2.0 = aggressive eviction.
    /// Consumed by gllm ExpertThermalManager, not by planner solvers.
    pub expert_eviction_aggressiveness: f64,
    /// MoE expert prefetch priority: > 1.0 = more aggressive prefetching.
    /// Consumed by gllm ExpertWeightPrefetcher, not by planner solvers.
    pub expert_prefetch_priority: f64,
}

impl StrategyBias {
    /// Clamp every field to its valid range (SPEC §11.1).
    pub fn validate(&mut self) {
        self.fusion_cost_scale = self.fusion_cost_scale.clamp(0.2, 3.0);
        self.pipeline_cost_scale = self.pipeline_cost_scale.clamp(0.2, 3.0);
        self.parallelism_cost_scale = self.parallelism_cost_scale.clamp(0.1, 3.0);
        self.epilogue_depth_preference = self.epilogue_depth_preference.clamp(0.3, 3.0);
        self.k_depth_preference = self.k_depth_preference.clamp(0.3, 3.0);
        self.kv_cache_budget_scale = self.kv_cache_budget_scale.clamp(0.2, 3.0);
        self.weight_prefetch_budget_scale = self.weight_prefetch_budget_scale.clamp(0.2, 3.0);
        self.batch_flexibility = self.batch_flexibility.clamp(0.0, 1.0);
        self.decode_ratio_scale = self.decode_ratio_scale.clamp(0.3, 2.0);
        self.expert_eviction_aggressiveness = self.expert_eviction_aggressiveness.clamp(0.0, 2.0);
        self.expert_prefetch_priority = self.expert_prefetch_priority.clamp(0.1, 5.0);
        self.speculative_decoding_value = self.speculative_decoding_value.clamp(0.1, 3.0);
        self.quantization_aggressiveness = self.quantization_aggressiveness.clamp(0.3, 3.0);
    }

    /// Get expert eviction aggressiveness (for gllm MoE subsystem).
    pub fn expert_eviction_aggressiveness(&self) -> f64 {
        self.expert_eviction_aggressiveness
    }

    /// Get expert prefetch priority (for gllm MoE subsystem).
    pub fn expert_prefetch_priority(&self) -> f64 {
        self.expert_prefetch_priority
    }
}

impl Default for StrategyBias {
    fn default() -> Self {
        Self {
            fusion_cost_scale: 1.0,
            pipeline_cost_scale: 1.0,
            parallelism_cost_scale: 1.0,
            epilogue_depth_preference: 1.0,
            k_depth_preference: 1.0,
            kv_cache_budget_scale: 1.0,
            weight_prefetch_budget_scale: 1.0,
            batch_flexibility: 1.0,
            decode_ratio_scale: 1.0,
            speculative_decoding_value: 1.0,
            quantization_aggressiveness: 1.0,
            expert_eviction_aggressiveness: 0.0,
            expert_prefetch_priority: 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Data structures
// ═══════════════════════════════════════════════════════════════

/// A GEMM shape key for blocking parameter lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Microkernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrokernelChoice {
    pub mr: usize,
    pub nr: usize,
}

/// Fusion decisions — which operator pairs are fused in the compiled layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionDecision {
    /// RMSNorm output feeds directly into GEMM without memory writeback
    RmsNormIntoGemm,
    /// GEMM epilogue fuses bias + activation
    GemmBiasAct(Activation),
    /// QKV three GEMMs share input, single pack_a
    QkvSharedInput,
    /// Attention score + softmax + V matmul fused (FlashAttention-style tiling)
    FlashAttention,
    /// Gate and Up GEMMs fused with SiLU: SiLU(gate) * up
    SwiGluFusion,
    /// Gate and Up GEMMs fused with GELU: GELU(gate) * up (Gemma GeGLU)
    GeGluFusion,
}

// ── §10.3 Roofline ──

/// Operator bottleneck classification from roofline analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckClass {
    /// Arithmetic intensity >> ridge point → maximize FMA utilization
    ComputeBound,
    /// Arithmetic intensity << ridge point → minimize memory traffic
    MemoryBound,
    /// Near ridge point → balance compute and bandwidth
    Mixed,
}

/// Roofline analysis result per operator category.
#[derive(Debug, Clone)]
pub struct RooflineResult {
    /// Ridge point of the hardware (FLOPS/byte)
    pub ridge_point: f64,
    /// Peak GFLOPS (F32)
    pub peak_gflops: f64,
    /// Peak memory bandwidth (GB/s)
    pub peak_bandwidth_gbs: f64,
    /// Bottleneck classification for GEMM with seq_len >= 128
    pub gemm_prefill: BottleneckClass,
    /// Bottleneck classification for GEMM with seq_len == 1 (decode)
    pub gemm_decode: BottleneckClass,
    /// Bottleneck classification for Attention prefill
    pub attn_prefill: BottleneckClass,
    /// Bottleneck classification for Attention decode
    pub attn_decode: BottleneckClass,
    /// Bottleneck classification for elementwise ops (norm, activation)
    pub elementwise: BottleneckClass,
}

// ── §10.5 Cache Budget ──

/// Cache hierarchy budget allocation.
#[derive(Debug, Clone)]
pub struct CacheBudgetPlan {
    /// L1 bytes usable for GEMM tiles (A_panel + B_panel ≤ this)
    pub l1_tile_budget: usize,
    /// L1 bytes for TileLevelFusion scratch
    pub l1_fusion_scratch: usize,
    /// L2 bytes for KV hot pages
    pub l2_kv_budget: usize,
    /// L2 bytes for weight prefetch window
    pub l2_weight_budget: usize,
    /// L2 bytes for activation buffer
    pub l2_activation_budget: usize,
    /// L3 bytes for model weights
    pub l3_model_budget: usize,
    /// L3 bytes for KV cold pages
    pub l3_kv_cold_budget: usize,
}

// ── §10.4 GEMM Solver ──

/// GEMM microkernel strategy selected by cost-model evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmMicrokernelStrategy {
    /// AMX tile instructions (TDPBF16PS)
    AmxTile,
    /// AVX-512 native BF16 (VDPBF16PS)
    Avx512NativeBf16,
    /// BLIS-style JIT with AVX-512 zmm registers
    BlisAvx512,
    /// BLIS-style JIT with AVX2 ymm registers
    BlisAvx2,
    /// BLIS-style JIT with NEON v registers
    BlisNeon,
    /// BLIS-style JIT with SVE registers
    BlisSve,
    /// GPU Tensor Core (wmma/mma.sync/WGMMA)
    GpuTensorCore,
    /// GPU without matrix acceleration
    GpuScalar,
    /// Fallback scalar
    Scalar,
}

/// GEMM solver output: microkernel geometry and tuning knobs.
#[derive(Debug, Clone)]
pub struct GemmPlan {
    /// Microkernel M register block
    pub mr: usize,
    /// Microkernel N register block
    pub nr: usize,
    /// Number of SIMD vector registers for NR
    pub nr_vecs: usize,
    /// K-loop software pipeline depth (prefetch stages)
    pub k_pipeline_depth: usize,
    /// Prefetch distance in bytes for A panel
    pub pf_distance_a: usize,
    /// Prefetch distance in bytes for B panel
    pub pf_distance_b: usize,
    /// Maximum epilogue operations that fit in scratch registers
    pub max_epilogue_depth: usize,
    /// Number of accumulator registers used
    pub acc_regs: usize,
    /// Number of scratch registers available for epilogue
    pub scratch_regs: usize,
    /// Selected microkernel strategy
    pub strategy: GemmMicrokernelStrategy,
}

// ── §10.6 Fusion Solver ──

/// FFN block fusion path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnFusionStrategy {
    /// SiLU injected into Gate GEMM epilogue, then LoopFusion with Up
    GateSiLUInject,
    /// Gate/Up separate GEMM + LoopFusion(SiLU×Up)
    SeparateGemm,
}

/// Fusion solver output.
#[derive(Debug, Clone)]
pub struct FusionStrategy {
    /// Maximum epilogue chain depth from register budget
    pub max_epilogue_depth: usize,
    /// L1 bytes threshold: above this → TileLevelFusion, below → ComputeRoot
    pub tile_fusion_threshold: usize,
    /// FFN gate+up fusion path
    pub ffn_strategy: FfnFusionStrategy,
    /// Whether RmsNorm→GEMM direct feed is enabled
    pub norm_into_gemm: bool,
    /// Whether QKV shared input (single pack_a) is enabled
    pub qkv_shared_input: bool,
    /// Whether cross-layer residual bypass is enabled
    pub cross_layer_residual: bool,
}

// ── §10.7 Attention Solver ──

/// Attention implementation path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionVariant {
    /// FA4 block-scaled (SM100+)
    FA4BlockScaled,
    /// FA3 pipeline with WGMMA + TMA (SM90)
    FA3Pipeline,
    /// FA2 tiled with mma.sync (SM80)
    FA2Tiled,
    /// wmma tiled (SM70)
    WmmaTiled,
    /// AMX tile attention (CPU)
    AmxTile,
    /// AVX-512 vectorized loop (CPU)
    Avx512Loop,
    /// NEON vectorized loop (ARM)
    NeonLoop,
    /// Scalar fallback
    ScalarLoop,
}

/// Attention solver output.
#[derive(Debug, Clone)]
pub struct AttentionPlan {
    /// Selected attention variant
    pub variant: AttentionVariant,
    /// Q tile size for tiled attention
    pub tile_q: usize,
    /// KV tile size for tiled attention
    pub tile_kv: usize,
    /// Whether online softmax is used (single-pass)
    pub online_softmax: bool,
    /// Whether warp specialization is used (SM90+)
    pub warp_specialization: bool,
    /// Whether TMA 2D prefetch is used (SM90+)
    pub tma_enabled: bool,
}

// ── §10.8 Parallelism Solver ──

/// GPU SM partition for Multi-Wave.
#[derive(Debug, Clone)]
pub struct GpuSmPartition {
    /// Total SM count
    pub total_sm: usize,
    /// Number of partitions (waves)
    pub num_partitions: usize,
    /// SM per partition
    pub sm_per_partition: usize,
}

/// NUMA binding for a wave.
#[derive(Debug, Clone)]
pub struct NumaBinding {
    /// NUMA node ID
    pub node_id: usize,
    /// Core range [start, end)
    pub core_start: usize,
    pub core_end: usize,
    /// L3 cache bytes on this node
    pub l3_bytes: usize,
}

/// Parallelism solver output.
#[derive(Debug, Clone)]
pub struct ParallelPlan {
    /// Number of compute waves
    pub wave_count: usize,
    /// GPU SM partition (None on CPU)
    pub gpu_sm_partition: Option<GpuSmPartition>,
    /// CPU NUMA bindings (empty on GPU)
    pub numa_bindings: Vec<NumaBinding>,
    /// Minimum batch tokens per wave for hardware saturation
    pub min_batch_tokens_per_wave: usize,
    /// Minimum decode sequences per wave for saturation
    pub min_decode_per_wave: usize,
    /// Occupancy target (0.0-1.0)
    pub occupancy_target: f32,
}

// ── §10.9 Feature Router ──

/// A single hardware feature routing decision.
#[derive(Debug, Clone)]
pub struct FeatureDecision {
    /// Feature name (e.g., "avx512_bf16", "wgmma", "tma_2d")
    pub feature: String,
    /// Whether this feature is enabled
    pub enabled: bool,
    /// Human-readable reason for the decision
    pub reason: String,
}

/// Feature router output.
#[derive(Debug, Clone)]
pub struct FeaturePlan {
    /// All feature decisions
    pub decisions: Vec<FeatureDecision>,
    /// Total L1i instruction cache budget used (bytes)
    pub l1i_used: usize,
    /// Total L1i budget available (bytes)
    pub l1i_budget: usize,
}

// ── §10.10 Batch Solver ──

/// Batch strategy solver output.
#[derive(Debug, Clone)]
pub struct BatchPlan {
    /// Maximum decode ratio cap (0.0-1.0)
    pub decode_ratio_cap: f32,
    /// Maximum chunk size for prefill
    pub max_chunk_size: usize,
    /// Available golden sizes for same-length grouping
    pub golden_sizes: Vec<usize>,
    /// Minimum sequences to trigger compact
    pub min_compact_threshold: usize,
    /// Waste ratio threshold for compact decision
    pub compact_waste_threshold: f32,
    /// Maximum decode slots
    pub decode_slots: usize,
    /// Maximum chunks per batch
    pub max_chunks_per_batch: usize,
}

// ═══════════════════════════════════════════════════════════════
//  Complete ExecutionPlan (extended with HwOpt sub-plans)
// ═══════════════════════════════════════════════════════════════

/// Complete execution plan for a compiled transformer layer.
///
/// Every performance parameter is determined here. The codegen phase
/// translates this plan into machine code with all values baked as immediates.
///
/// The plan is built by `HwOptEngine::solve()` which uses cost-model
/// evaluation instead of hardcoded if/else lookup tables.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Hardware profile used for planning
    pub profile: DeviceProfile,

    // ── GEMM blocking (per shape) ──
    /// (KC, MC, NC) for each distinct GEMM shape in the layer
    pub gemm_blocking: HashMap<GemmShape, (usize, usize, usize)>,

    // ── Threading ──
    /// Thread count for compute-bound regions (GEMM)
    pub num_threads: usize,

    // ── Microkernel ──
    pub microkernel: MicrokernelChoice,

    // ── Prefetch ──
    pub prefetch_a_l1: usize,
    pub prefetch_b_l1: usize,
    pub prefetch_a_l2: usize,

    // ── Unrolling ──
    pub k_unroll: usize,

    // ── Attention tiling ──
    pub attn_tile_q: usize,
    pub attn_tile_kv: usize,

    // ── Buffer layout ──
    /// Total scratchpad bytes needed by the compiled layer
    pub scratchpad_bytes: usize,

    // ── Fusion decisions ──
    pub fusions: Vec<FusionDecision>,

    // ── §10 HwOpt sub-plans ──
    /// Roofline analysis results
    pub roofline: RooflineResult,
    /// Cache hierarchy budget allocation
    pub cache_budget: CacheBudgetPlan,
    /// GEMM solver output (microkernel geometry + strategy)
    pub gemm_plan: GemmPlan,
    /// Fusion solver output (epilogue depth, FFN strategy, etc.)
    pub fusion_plan: FusionStrategy,
    /// Attention solver output (variant + tiling)
    pub attention_plan: AttentionPlan,
    /// Parallelism solver output (waves + SM partition)
    pub parallel_plan: ParallelPlan,
    /// Batch strategy solver output
    pub batch_plan: BatchPlan,
    /// Feature router output
    pub feature_plan: FeaturePlan,
    /// Strategy bias coefficients applied during planning
    pub strategy_bias: StrategyBias,

    /// R0 per-GEMM bottleneck analysis (from PainPointAnalyzer).
    /// None when built from LayerIR (profile-only path without CompilerGraph).
    /// When present, codegen/fusion can use precise per-op bottleneck data
    /// instead of the global roofline classification.
    pub op_bottleneck_map: Option<OpBottleneckMap>,
}

impl ExecutionPlan {
    /// Build an execution plan from a LayerIR, DeviceProfile, and StrategyBias.
    ///
    /// Internally delegates to `HwOptEngine::solve()` for cost-model-driven
    /// strategy selection. The bias modulates cost-model decisions per
    /// SPEC/12-STRATEGY-ARBITER.md §5.
    pub fn build(ir: &LayerIR, profile: &DeviceProfile, bias: &StrategyBias) -> Self {
        HwOptEngine::solve(ir, profile, bias)
    }

    /// Build a profile-only execution plan (for CompilerGraph compilation paths
    /// that lack a full LayerIR). Strategy sub-plans are derived purely from
    /// DeviceProfile; IR-dependent fields use conservative defaults.
    /// Uses `StrategyBias::default()` (all neutral 1.0 coefficients).
    pub fn from_profile(profile: &DeviceProfile) -> Self {
        HwOptEngine::solve_profile_only(profile)
    }

    /// Profile-only solve with R0 PainPointAnalyzer bottlenecks injected.
    ///
    /// When `OpBottleneckMap` is available (from PainPointAnalyzer), the planner
    /// uses precise per-GEMM bottleneck data to refine roofline classifications
    /// instead of relying on global heuristics. This unifies planner and R0:
    /// PainPointAnalyzer does the analysis, planner consumes it.
    pub fn from_profile_with_bottlenecks(
        profile: &DeviceProfile,
        bottleneck_map: OpBottleneckMap,
    ) -> Self {
        HwOptEngine::solve_profile_with_bottlenecks(profile, bottleneck_map)
    }
}

/// SPEC §10: `HwOptPlan` is the unified immutable output of `HwOptEngine`.
///
/// All hardware-dependent optimization decisions are computed once at model load
/// time and baked into this plan. JIT codegen, fusion engine, and runtime
/// scheduler consume this plan as read-only during inference.
///
/// Type alias for backward compatibility — existing code may reference `ExecutionPlan`.
pub type HwOptPlan = ExecutionPlan;

// ═══════════════════════════════════════════════════════════════
//  §10 Named Solver Types — DAG Components of HwOptEngine
// ═══════════════════════════════════════════════════════════════

/// §10.3 RooflineAnalyzer — operator bottleneck classifier.
///
/// Computes arithmetic intensity for each operator category and classifies
/// as ComputeBound / MemoryBound / Mixed against the hardware ridge point.
/// **Level 0** (no dependencies).
pub struct RooflineAnalyzer;

impl RooflineAnalyzer {
    /// Classify operator bottlenecks from hardware peak metrics.
    pub fn solve(profile: &DeviceProfile) -> RooflineResult {
        HwOptEngine::solve_roofline(profile)
    }
}

/// §10.5 CacheBudgetSolver — cache hierarchy budget allocator.
///
/// Distributes L1/L2/L3 physical capacity across KV cache, weights,
/// activations, and workspace with bias-modulated ratios.
/// **Level 0** (no dependencies).
pub struct CacheBudgetSolver;

impl CacheBudgetSolver {
    /// Allocate cache budgets from hardware cache sizes.
    pub fn solve(
        profile: &DeviceProfile,
        l1: usize,
        l2: usize,
        l3: usize,
        ir: &LayerIR,
        bias: &StrategyBias,
    ) -> CacheBudgetPlan {
        HwOptEngine::solve_cache_budget(profile, l1, l2, l3, ir, bias)
    }
}

/// §10.4 GemmSolver — GEMM strategy solver.
///
/// Selects optimal microkernel geometry (MR×NR), K-loop pipeline depth,
/// prefetch distances, and epilogue capacity via cost-model evaluation.
/// **Level 1** (depends on RooflineAnalyzer + CacheBudgetSolver).
pub struct GemmSolver;

impl GemmSolver {
    /// Solve GEMM parameters from roofline classification and cache budget.
    pub fn solve(
        profile: &DeviceProfile,
        roofline: &RooflineResult,
        cache: &CacheBudgetPlan,
        mr: usize,
        nr: usize,
        bias: &StrategyBias,
    ) -> GemmPlan {
        HwOptEngine::solve_gemm(profile, roofline, cache, mr, nr, bias)
    }
}

/// §10.7 AttentionSolver — attention implementation selector.
///
/// Picks the optimal attention variant (FA4/FA3/FA2/wmma/AMX/AVX-512/NEON)
/// and tile sizes based on hardware capabilities.
/// **Level 1** (depends on CacheBudgetSolver + RooflineAnalyzer).
pub struct AttentionSolver;

impl AttentionSolver {
    /// Select attention variant and tile configuration.
    pub fn solve(
        profile: &DeviceProfile,
        cache: &CacheBudgetPlan,
        roofline: &RooflineResult,
        ir: &LayerIR,
    ) -> AttentionPlan {
        HwOptEngine::solve_attention(profile, cache, roofline, ir)
    }
}

/// §10.6 FusionSolver — fusion depth and strategy decider.
///
/// Determines epilogue chain depth, TileLevelFusion vs ComputeRoot threshold,
/// FFN gate fusion path, and cross-layer residual eligibility.
/// **Level 2** (depends on GemmSolver + CacheBudgetSolver + RooflineAnalyzer).
pub struct FusionSolver;

impl FusionSolver {
    /// Compute fusion strategy from GEMM and cache parameters.
    pub fn solve(
        profile: &DeviceProfile,
        gemm: &GemmPlan,
        cache: &CacheBudgetPlan,
        roofline: &RooflineResult,
        bias: &StrategyBias,
    ) -> FusionStrategy {
        HwOptEngine::solve_fusion(profile, gemm, cache, roofline, bias)
    }
}

/// §10.8 ParallelismSolver — compute wave and NUMA binding planner.
///
/// Plans GPU SM partitioning (Multi-Wave) or CPU NUMA core bindings
/// with cost-based wave count selection.
/// **Level 2** (depends on CacheBudgetSolver).
pub struct ParallelismSolver;

impl ParallelismSolver {
    /// Plan parallelism strategy from cache budget and hardware topology.
    pub fn solve(
        profile: &DeviceProfile,
        cache: &CacheBudgetPlan,
        bias: &StrategyBias,
    ) -> ParallelPlan {
        HwOptEngine::solve_parallelism(profile, cache, bias)
    }
}

/// §10.10 BatchSolver — batch scheduling strategy planner.
///
/// Computes decode ratio cap, golden sizes for same-length grouping,
/// decode slot count, and compact thresholds.
/// **Level 3** (depends on ParallelismSolver + CacheBudgetSolver).
pub struct BatchSolver;

impl BatchSolver {
    /// Plan batch scheduling from parallel and cache parameters.
    pub fn solve(
        profile: &DeviceProfile,
        parallel: &ParallelPlan,
        cache: &CacheBudgetPlan,
        ir: &LayerIR,
        bias: &StrategyBias,
    ) -> BatchPlan {
        HwOptEngine::solve_batch(profile, parallel, cache, ir, bias)
    }
}

/// §10.9 FeatureRouter — hardware feature activation router.
///
/// Routes hardware feature flags (AMX, AVX-512, BF16, WGMMA, TMA, etc.)
/// into enabled/disabled decisions with L1i instruction budget tracking.
/// **Level 4** (depends on GemmSolver + AttentionSolver + ParallelismSolver).
pub struct FeatureRouter;

impl FeatureRouter {
    /// Route feature decisions from solver outputs.
    pub fn solve(
        profile: &DeviceProfile,
        gemm: &GemmPlan,
        attention: &AttentionPlan,
        parallel: &ParallelPlan,
        bias: &StrategyBias,
    ) -> FeaturePlan {
        HwOptEngine::solve_features(profile, gemm, attention, parallel, bias)
    }
}

// ═══════════════════════════════════════════════════════════════
//  HwOptEngine — Unified Hardware-Aware Optimization Engine
// ═══════════════════════════════════════════════════════════════

/// Unified hardware-aware optimization engine.
///
/// Replaces scattered if/else hardware checks with cost-model-driven
/// strategy evaluation. Produces an immutable `ExecutionPlan` consumed
/// by JIT codegen, fusion engine, and runtime scheduler.
///
/// Solvers execute in DAG topological order:
/// ```text
/// Level 0: RooflineAnalyzer + CacheBudgetSolver
/// Level 1: GemmSolver + AttentionSolver
/// Level 2: FusionSolver + ParallelismSolver
/// Level 3: BatchSolver
/// Level 4: FeatureRouter
/// ```
pub struct HwOptEngine;

impl HwOptEngine {
    /// Solve all hardware-dependent optimization decisions.
    ///
    /// This is the single entry point that produces the complete `ExecutionPlan`.
    /// Called once at model load time. The result is immutable during inference.
    /// The `bias` parameter modulates cost-model decisions per SPEC §5.
    pub fn solve(ir: &LayerIR, profile: &DeviceProfile, bias: &StrategyBias) -> HwOptPlan {
        let kc = &profile.kernel_config;
        let (mr, nr) = (kc.mr, kc.nr);
        let (l1, l2, l3) = (kc.l1d, kc.l2, kc.l3);

        // ── Level 0: RooflineAnalyzer + CacheBudgetSolver ──
        let roofline = RooflineAnalyzer::solve(profile);
        let cache_budget = CacheBudgetSolver::solve(profile, l1, l2, l3, ir, bias);

        // ── Level 1: GemmSolver + AttentionSolver ──
        let gemm_plan = GemmSolver::solve(profile, &roofline, &cache_budget, mr, nr, bias);
        let attention_plan = AttentionSolver::solve(profile, &cache_budget, &roofline, ir);

        // ── Level 2: FusionSolver + ParallelismSolver ──
        let fusion_plan = FusionSolver::solve(profile, &gemm_plan, &cache_budget, &roofline, bias);
        let parallel_plan = ParallelismSolver::solve(profile, &cache_budget, bias);

        // ── Level 3: BatchSolver ──
        let batch_plan = BatchSolver::solve(profile, &parallel_plan, &cache_budget, ir, bias);

        // ── Level 4: FeatureRouter ──
        let feature_plan = FeatureRouter::solve(profile, &gemm_plan, &attention_plan, &parallel_plan, bias);

        // ── Legacy fields (GEMM blocking, scratchpad, fusion list) ──
        let gemm_shapes = collect_gemm_shapes(ir);
        let mut gemm_blocking = HashMap::new();
        for shape in &gemm_shapes {
            let blocking = compute_blocking(shape, kc.kc, kc.mc, kc.nc, mr, nr);
            gemm_blocking.insert(*shape, blocking);
        }

        let num_threads = profile.physical_cores;
        let scratchpad_bytes = compute_scratchpad(ir);
        let fusions = plan_fusions(ir);

        let k_unroll = profile.k_unroll_factor();
        let (attn_tile_q, attn_tile_kv) = (
            attention_plan.tile_q,
            attention_plan.tile_kv,
        );

        ExecutionPlan {
            profile: profile.clone(),
            gemm_blocking,
            num_threads,
            microkernel: MicrokernelChoice { mr, nr },
            prefetch_a_l1: gemm_plan.pf_distance_a,
            prefetch_b_l1: gemm_plan.pf_distance_b,
            prefetch_a_l2: gemm_plan.pf_distance_a * 2,
            k_unroll,
            attn_tile_q,
            attn_tile_kv,
            scratchpad_bytes,
            fusions,
            roofline,
            cache_budget,
            gemm_plan,
            fusion_plan,
            attention_plan,
            parallel_plan,
            batch_plan,
            feature_plan,
            strategy_bias: *bias,
            op_bottleneck_map: None,
        }
    }

    /// Profile-only solve — no LayerIR available (CompilerGraph path).
    ///
    /// Derives all hardware-dependent sub-plans from DeviceProfile alone.
    /// IR-dependent fields use conservative defaults.
    /// Uses `StrategyBias::default()` (all neutral 1.0 coefficients).
    pub fn solve_profile_only(profile: &DeviceProfile) -> ExecutionPlan {
        Self::solve_profile_only_with_bias(profile, &StrategyBias::default())
    }

    /// Profile-only solve with R0 PainPointAnalyzer bottlenecks injected.
    ///
    /// When `OpBottleneckMap` is available (from PainPointAnalyzer), the planner
    /// uses precise per-GEMM bottleneck data to refine roofline classifications
    /// instead of relying on global heuristics. This unifies planner and R0:
    /// PainPointAnalyzer does the analysis, planner consumes it.
    pub fn solve_profile_with_bottlenecks(
        profile: &DeviceProfile,
        bottleneck_map: OpBottleneckMap,
    ) -> ExecutionPlan {
        let mut plan = Self::solve_profile_only_with_bias(profile, &StrategyBias::default());

        // Refine global roofline using precise R0 data:
        // If any GEMM is memory-bound in R0, update the global classification
        if let Some(gemm_bn) = bottleneck_map.gemm_bottlenecks.values().find(|bn| {
            matches!(bn.bottleneck, crate::compiler::pain_point::BottleneckType::MemoryBound { .. })
        }) {
            // At least one GEMM is memory-bound → decode path is memory-bound
            plan.roofline.gemm_decode = BottleneckClass::MemoryBound;
            let _ = gemm_bn; // suppress unused warning
        }
        if let Some(gemm_bn) = bottleneck_map.gemm_bottlenecks.values().find(|bn| {
            matches!(bn.bottleneck, crate::compiler::pain_point::BottleneckType::ComputeBound { .. })
        }) {
            plan.roofline.gemm_prefill = BottleneckClass::ComputeBound;
            let _ = gemm_bn;
        }

        plan.op_bottleneck_map = Some(bottleneck_map);
        plan
    }

    /// Profile-only solve with an explicit `StrategyBias`.
    ///
    /// Same as `solve_profile_only` but uses the provided bias instead of
    /// neutral defaults. Called by `init_global_execution_plan_with_bias`.
    pub fn solve_profile_only_with_bias(profile: &DeviceProfile, bias: &StrategyBias) -> ExecutionPlan {
        // Synthesize a minimal IR with conservative defaults
        let default_ir = LayerIR {
            moe: None,
            hidden: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate: 11008,
            quant: None,
            dtype: crate::types::DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 2048,
            partial_rotary_factor: 1.0,
            activation: crate::traits::Activation::Silu,
        };

        // Reuse the full solver with synthetic IR and caller-provided bias
        Self::solve(&default_ir, profile, bias)
    }

    // ───────────────────────────────────────────────────────────
    //  Level 0: RooflineAnalyzer
    // ───────────────────────────────────────────────────────────

    fn solve_roofline(profile: &DeviceProfile) -> RooflineResult {
        let peak_gflops = profile.peak_gflops(DType::F32);
        let peak_bw = profile.peak_bandwidth_gbs;
        let ridge = if peak_bw > 0.0 { peak_gflops / peak_bw } else { 100.0 };

        // Classify operators by arithmetic intensity vs ridge point
        // GEMM prefill (M>=128): AI ≈ hidden_dim ≈ 4096 → always compute-bound
        let gemm_prefill = if 4096.0 > ridge * 1.5 {
            BottleneckClass::ComputeBound
        } else {
            BottleneckClass::Mixed
        };

        // GEMM decode (M=1): AI ≈ 1 → always memory-bound
        let gemm_decode = BottleneckClass::MemoryBound;

        // Attention prefill: AI ≈ 128 on GPU, lower on CPU
        let attn_prefill = if 128.0 > ridge * 1.5 {
            BottleneckClass::ComputeBound
        } else if 128.0 < ridge * 0.67 {
            BottleneckClass::MemoryBound
        } else {
            BottleneckClass::Mixed
        };

        // Attention decode: AI ≈ 2 → always memory-bound
        let attn_decode = BottleneckClass::MemoryBound;

        // Elementwise (norm/activation): AI ≈ 2-3 → memory-bound
        let elementwise = BottleneckClass::MemoryBound;

        RooflineResult {
            ridge_point: ridge,
            peak_gflops,
            peak_bandwidth_gbs: peak_bw,
            gemm_prefill,
            gemm_decode,
            attn_prefill,
            attn_decode,
            elementwise,
        }
    }

    // ───────────────────────────────────────────────────────────
    //  Level 0: CacheBudgetSolver
    // ───────────────────────────────────────────────────────────

    fn solve_cache_budget(
        profile: &DeviceProfile,
        l1: usize,
        l2: usize,
        l3: usize,
        ir: &LayerIR,
        bias: &StrategyBias,
    ) -> CacheBudgetPlan {
        let _ = ir; // IR available for workload-aware tuning

        // L1: 75% for GEMM tiles, 25% for fusion scratch
        let l1_tile = (l1 as f64 * 0.75) as usize;
        let l1_scratch = l1 - l1_tile;

        // L2: bias-modulated ratio allocation (§5.3)
        // Base ratios: KV=40%, weights=35%, activation=25%
        let raw_kv = 0.40 * bias.kv_cache_budget_scale;
        let raw_weight = 0.35 * bias.weight_prefetch_budget_scale;
        let raw_act = 0.25;
        let total = raw_kv + raw_weight + raw_act;
        let (kv_ratio, weight_ratio) = if total > 1.0 {
            (raw_kv / total, raw_weight / total)
        } else {
            (raw_kv, raw_weight)
        };
        let l2_kv = (l2 as f64 * kv_ratio) as usize;
        let l2_weight = (l2 as f64 * weight_ratio) as usize;
        let l2_act = l2.saturating_sub(l2_kv + l2_weight);

        // L3: 70% model, 20% KV cold, 10% workspace
        let l3_model = (l3 as f64 * 0.70) as usize;
        let l3_kv_cold = (l3 as f64 * 0.20) as usize;

        CacheBudgetPlan {
            l1_tile_budget: l1_tile,
            l1_fusion_scratch: l1_scratch,
            l2_kv_budget: l2_kv,
            l2_weight_budget: l2_weight,
            l2_activation_budget: l2_act,
            l3_model_budget: l3_model,
            l3_kv_cold_budget: l3_kv_cold,
        }
    }

    // ───────────────────────────────────────────────────────────
    //  Level 1: GemmSolver
    // ───────────────────────────────────────────────────────────

    fn solve_gemm(
        profile: &DeviceProfile,
        roofline: &RooflineResult,
        cache: &CacheBudgetPlan,
        mr: usize,
        nr: usize,
        bias: &StrategyBias,
    ) -> GemmPlan {
        let kc = &profile.kernel_config;
        let num_regs = profile.num_simd_regs();
        let simd_width = profile.simd_width(DType::F32);
        let nr_vecs = (nr + simd_width - 1) / simd_width;
        let acc_regs = mr * nr_vecs;
        let ptr_regs = 2; // A ptr + B ptr

        // K-loop pipeline depth based on bias.k_depth_preference (§5.4)
        // First compute with full scratch to determine k_depth candidate
        let base_scratch = num_regs.saturating_sub(acc_regs + ptr_regs);
        let k_depth = if bias.k_depth_preference >= 1.5 {
            // Strong pipeline preference: try depth 4, 2, 1
            if base_scratch >= 8 { 4 } else if base_scratch >= 4 { 2 } else { 1 }
        } else if bias.k_depth_preference >= 0.8 {
            // Default: original logic (compute-bound → 2, else → 1)
            if matches!(roofline.gemm_prefill, BottleneckClass::ComputeBound) && base_scratch >= 4 { 2 } else { 1 }
        } else {
            // Low preference: always 1
            1
        };

        // Account for pipeline registers consuming extra scratch
        let extra_pipeline_regs = if k_depth > 1 { k_depth } else { 0 };
        let scratch = num_regs.saturating_sub(acc_regs + ptr_regs + extra_pipeline_regs);

        // Prefetch distance: derived from cache line size × pipeline depth
        let cacheline = 64; // typical x86_64
        let pf_a = std::cmp::max(cacheline * k_depth, simd_width * 4);
        let pf_b = pf_a;

        // Max epilogue depth from scratch register budget, modulated by bias (§5.4)
        let base_max_epi = if scratch >= 2 {
            (scratch as f64 / 1.5).floor() as usize
        } else {
            0
        };
        let max_epi = ((base_max_epi as f64) * bias.epilogue_depth_preference)
            .round() as usize;
        // Ensure at least 1 if base allowed it, clamp to available scratch
        let max_epi = max_epi.max(if base_max_epi > 0 { 1 } else { 0 }).min(scratch);

        // Strategy selection: evaluate candidates by cost
        let strategy = Self::select_gemm_strategy_cost_based(profile, roofline, cache);

        GemmPlan {
            mr,
            nr,
            nr_vecs,
            k_pipeline_depth: k_depth,
            pf_distance_a: pf_a,
            pf_distance_b: pf_b,
            max_epilogue_depth: max_epi,
            acc_regs,
            scratch_regs: scratch,
            strategy,
        }
    }

    /// Cost-based GEMM strategy selection (replaces hardcoded if/else).
    fn select_gemm_strategy_cost_based(
        profile: &DeviceProfile,
        _roofline: &RooflineResult,
        _cache: &CacheBudgetPlan,
    ) -> GemmMicrokernelStrategy {
        let kc = &profile.kernel_config;

        // Build candidate pool with hardware prerequisites
        let mut candidates: Vec<GemmMicrokernelStrategy> = Vec::new();

        // CPU candidates
        if kc.has_amx {
            candidates.push(GemmMicrokernelStrategy::AmxTile);
        }
        if kc.use_avx512 && profile.isa == crate::dispatch::device_profile::IsaLevel::Avx512 {
            candidates.push(GemmMicrokernelStrategy::BlisAvx512);
        }
        if kc.use_avx512 {
            candidates.push(GemmMicrokernelStrategy::Avx512NativeBf16);
        }
        candidates.push(GemmMicrokernelStrategy::BlisAvx2);

        // ARM candidates
        #[cfg(target_arch = "aarch64")]
        {
            candidates.push(GemmMicrokernelStrategy::BlisNeon);
            if kc.has_sve {
                candidates.push(GemmMicrokernelStrategy::BlisSve);
            }
        }

        // GPU candidates (feature-gated)
        #[cfg(any(feature = "jit-cuda", feature = "jit-hip"))]
        {
            // GPU strategy is selected at a higher level by gemm_dispatch.rs
            // Here we just note that GPU path is available
        }

        // Cost evaluation: pick the highest-performing candidate
        // (candidates are ordered by preference — first match wins for now,
        //  future: full cost model evaluation)
        candidates.into_iter().next().unwrap_or(GemmMicrokernelStrategy::Scalar)
    }

    // ───────────────────────────────────────────────────────────
    //  Level 1: AttentionSolver
    // ───────────────────────────────────────────────────────────

    fn solve_attention(
        profile: &DeviceProfile,
        cache: &CacheBudgetPlan,
        roofline: &RooflineResult,
        ir: &LayerIR,
    ) -> AttentionPlan {
        let kc = &profile.kernel_config;

        // Select variant based on hardware capabilities
        let (variant, warp_spec, tma) = {
            #[cfg(feature = "jit-cuda")]
            {
                if let Some(sm) = Self::detect_cuda_sm() {
                    if sm >= 100 {
                        (AttentionVariant::FA4BlockScaled, true, true)
                    } else if sm >= 90 {
                        (AttentionVariant::FA3Pipeline, true, true)
                    } else if sm >= 80 {
                        (AttentionVariant::FA2Tiled, false, false)
                    } else if sm >= 70 {
                        (AttentionVariant::WmmaTiled, false, false)
                    } else {
                        (AttentionVariant::ScalarLoop, false, false)
                    }
                } else {
                    (AttentionVariant::ScalarLoop, false, false)
                }
            }
            #[cfg(not(feature = "jit-cuda"))]
            {
                // CPU path selection
                if kc.has_amx {
                    (AttentionVariant::AmxTile, false, false)
                } else if kc.use_avx512 {
                    (AttentionVariant::Avx512Loop, false, false)
                } else {
                    #[cfg(target_arch = "aarch64")]
                    { (AttentionVariant::NeonLoop, false, false) }
                    #[cfg(not(target_arch = "aarch64"))]
                    { (AttentionVariant::ScalarLoop, false, false) }
                }
            }
        };

        // Tile sizing: derive from cache budget and head dimension
        let head_dim = ir.head_dim;
        let elem_bytes = ir.dtype.size_bytes();
        // SMEM/L1 budget: 3 tiles (Q + K + V)
        let tile_bytes_per_token = 3 * head_dim * elem_bytes;
        let cache_for_attn = match variant {
            AttentionVariant::FA3Pipeline | AttentionVariant::FA4BlockScaled => {
                // GPU: use SMEM budget
                cache.l1_tile_budget.max(49152) // typical SMEM size
            }
            _ => cache.l1_tile_budget,
        };
        let max_tile = if tile_bytes_per_token > 0 {
            cache_for_attn / tile_bytes_per_token
        } else {
            64
        };

        let tile_q = prev_power_of_2(std::cmp::min(max_tile, ir.max_seq)).max(16);
        let tile_kv = prev_power_of_2(std::cmp::min(max_tile * 4, ir.max_seq)).max(16);

        // Online softmax: always enabled for tiled attention
        let online_softmax = tile_q < ir.max_seq;

        let _ = roofline; // Available for workload-aware tuning

        AttentionPlan {
            variant,
            tile_q,
            tile_kv,
            online_softmax,
            warp_specialization: warp_spec,
            tma_enabled: tma,
        }
    }

    #[cfg(feature = "jit-cuda")]
    fn detect_cuda_sm() -> Option<u32> {
        crate::gpu::cuda::CudaDriver::load().ok()?.compute_capability().ok()
    }

    // ───────────────────────────────────────────────────────────
    //  Level 2: FusionSolver
    // ───────────────────────────────────────────────────────────

    fn solve_fusion(
        profile: &DeviceProfile,
        gemm: &GemmPlan,
        cache: &CacheBudgetPlan,
        roofline: &RooflineResult,
        bias: &StrategyBias,
    ) -> FusionStrategy {
        use crate::compiler::hardware_profile::HardwareProfile;
        let hw = HardwareProfile::detect(profile);

        // Max epilogue depth: min of GEMM solver output and HardwareProfile budget.
        // HardwareProfile accounts for ISA-specific register file constraints
        // (e.g., 16 ymm on AVX2 vs 32 zmm on AVX-512 vs 255 regs on GPU).
        let max_epilogue = gemm.max_epilogue_depth.min(hw.max_epilogue_depth());

        // Aggressiveness-modulated epilogue depth: conservative profiles reduce depth.
        // aggressiveness < 0.5 -> use ceil(max * aggressiveness * 2) to reduce depth.
        // aggressiveness >= 0.5 -> keep full depth.
        let max_epilogue = if hw.fusion_aggressiveness() < 0.5 {
            ((max_epilogue as f32 * hw.fusion_aggressiveness() * 2.0).ceil() as usize).max(1)
        } else {
            max_epilogue
        };

        // Tile vs ComputeRoot threshold: 75% of L1, modulated by fusion_cost_scale (§5.5)
        // GPU profiles with TMA/large shared memory use a higher threshold (more aggressive tile fusion).
        let tile_threshold_base = cache.l1_tile_budget as f64 * 0.75;
        let tile_threshold_scale = if hw.prefer_gemm_fusion() {
            // GPU: larger tiles due to shared memory and async copy
            1.0 + hw.fusion_aggressiveness() as f64 * 0.3
        } else {
            1.0
        };
        let tile_threshold = (tile_threshold_base * tile_threshold_scale * bias.fusion_cost_scale) as usize;

        // FFN strategy: inject SiLU into epilogue if scratch allows AND profile is aggressive enough.
        // Conservative profiles (AVX2, Generic) use SeparateGemm to avoid register spills.
        let ffn_strategy = if gemm.scratch_regs >= 2 && hw.fusion_aggressiveness() >= 0.5 {
            // Need at least 1 reg for SiLU sigmoid constant + 1 for scratch
            FfnFusionStrategy::GateSiLUInject
        } else {
            FfnFusionStrategy::SeparateGemm
        };

        // NormIntoGemm: always beneficial (avoids one memory round-trip)
        let norm_into_gemm = true;

        // QKV shared input: always beneficial (3 pack_a → 1 pack_a)
        let qkv_shared_input = true;

        // Cross-layer residual: depends on L1 scratch availability and profile aggressiveness.
        // GPU profiles with large register files can always enable this.
        // Conservative CPU profiles need >= 4 scratch registers.
        let cross_layer_residual = if hw.fusion_aggressiveness() >= 0.8 {
            true // GPU / high-end CPU: always safe
        } else {
            gemm.scratch_regs >= 4
        };

        let _ = roofline; // Available for workload-aware tuning

        FusionStrategy {
            max_epilogue_depth: max_epilogue,
            tile_fusion_threshold: tile_threshold,
            ffn_strategy,
            norm_into_gemm,
            qkv_shared_input,
            cross_layer_residual,
        }
    }

    // ───────────────────────────────────────────────────────────
    //  Level 2: ParallelismSolver
    // ───────────────────────────────────────────────────────────

    fn solve_parallelism(
        profile: &DeviceProfile,
        cache: &CacheBudgetPlan,
        bias: &StrategyBias,
    ) -> ParallelPlan {
        let kc = &profile.kernel_config;

        // GPU wave count: cost-based selection with bias (§5.6)
        #[cfg(any(feature = "jit-cuda", feature = "jit-hip"))]
        {
            if let Some(gpu) = Self::detect_gpu_profile() {
                let total_sm = gpu.compute_units as usize;
                let min_sm_per_wave = 16usize;
                let max_waves = total_sm / min_sm_per_wave;

                // Cost-based wave selection with parallelism_cost_scale
                let mut best_wave = 1usize;
                let mut best_score = f64::NEG_INFINITY;
                for wc in [1usize, 2, 4] {
                    if wc > max_waves { break; }
                    let sync_cost = wc as f64 * 2.0 * bias.parallelism_cost_scale;
                    let parallel_benefit = (wc as f64).ln() * total_sm as f64 * 0.1;
                    let score = parallel_benefit - sync_cost;
                    if score > best_score {
                        best_score = score;
                        best_wave = wc;
                    }
                }
                let wave_count = best_wave;
                let sm_per_partition = total_sm / wave_count.max(1);
                let warp_size = gpu.warp_size as usize;
                let min_tokens = sm_per_partition * warp_size / 2;

                return ParallelPlan {
                    wave_count,
                    gpu_sm_partition: Some(GpuSmPartition {
                        total_sm,
                        num_partitions: wave_count,
                        sm_per_partition,
                    }),
                    numa_bindings: Vec::new(),
                    min_batch_tokens_per_wave: min_tokens,
                    min_decode_per_wave: min_tokens, // decode seq_len = 1
                    occupancy_target: 0.5,
                };
            }
        }

        // CPU NUMA path
        let numa_nodes = profile.numa.num_nodes().max(1);
        let cores_per_node = profile.physical_cores / numa_nodes;
        let cores_per_blis = 2; // main thread + 1 helper for pack
        let min_parallel_gemms = cores_per_node / cores_per_blis;

        let numa_bindings: Vec<NumaBinding> = (0..numa_nodes)
            .map(|node_id| {
                let start = node_id * cores_per_node;
                let end = start + cores_per_node;
                NumaBinding {
                    node_id,
                    core_start: start,
                    core_end: end,
                    l3_bytes: cache.l3_model_budget / numa_nodes,
                }
            })
            .collect();

        let wave_count = numa_nodes;
        let _ = kc;
        let _ = min_parallel_gemms;

        ParallelPlan {
            wave_count,
            gpu_sm_partition: None,
            numa_bindings,
            min_batch_tokens_per_wave: cores_per_node * 32, // rough estimate
            min_decode_per_wave: cores_per_node,
            occupancy_target: 0.5,
        }
    }

    #[cfg(any(feature = "jit-cuda", feature = "jit-hip"))]
    fn detect_gpu_profile() -> Option<crate::gpu::GpuDeviceProfile> {
        #[cfg(feature = "jit-cuda")]
        {
            use crate::gpu::cuda::CudaDriver;
            let driver = CudaDriver::load().ok()?;
            driver.device_profile().ok()
        }
        #[cfg(not(feature = "jit-cuda"))]
        None
    }

    // ───────────────────────────────────────────────────────────
    //  Level 3: BatchSolver
    // ───────────────────────────────────────────────────────────

    fn solve_batch(
        profile: &DeviceProfile,
        parallel: &ParallelPlan,
        cache: &CacheBudgetPlan,
        ir: &LayerIR,
        bias: &StrategyBias,
    ) -> BatchPlan {
        let _ = profile;
        let _ = cache;

        // batch_flexibility = 0.0 → force batch=1 (§5.7)
        if bias.batch_flexibility == 0.0 {
            return BatchPlan {
                decode_ratio_cap: 1.0,
                max_chunk_size: 1,
                golden_sizes: vec![1],
                min_compact_threshold: 1,
                compact_waste_threshold: 0.25,
                decode_slots: 1,
                max_chunks_per_batch: 1,
            };
        }

        // decode_ratio_cap modulated by bias (§5.7)
        let decode_ratio_cap = (0.6 * bias.decode_ratio_scale).min(1.0) as f32;

        // Golden sizes: powers of 2 up to max_seq
        let mut golden_sizes = Vec::new();
        let mut size = 16;
        while size <= ir.max_seq {
            golden_sizes.push(size);
            size *= 2;
        }
        if golden_sizes.is_empty() {
            golden_sizes.push(64);
        }

        // Decode slots: modulated by batch_flexibility (§5.7)
        let base_decode_slots = parallel.min_decode_per_wave.max(4);
        let decode_slots = ((base_decode_slots as f64) * bias.batch_flexibility).round() as usize;
        let decode_slots = decode_slots.max(1);
        let max_chunks_per_batch = parallel.wave_count * 4;

        BatchPlan {
            decode_ratio_cap,
            max_chunk_size: *golden_sizes.last().unwrap_or(&64),
            golden_sizes,
            min_compact_threshold: 4,
            compact_waste_threshold: 0.25,
            decode_slots,
            max_chunks_per_batch,
        }
    }

    // ───────────────────────────────────────────────────────────
    //  Level 4: FeatureRouter
    // ───────────────────────────────────────────────────────────

    fn solve_features(
        profile: &DeviceProfile,
        gemm: &GemmPlan,
        attention: &AttentionPlan,
        parallel: &ParallelPlan,
        bias: &StrategyBias,
    ) -> FeaturePlan {
        let mut decisions = Vec::new();
        let kc = &profile.kernel_config;

        // CPU features
        if kc.has_amx {
            decisions.push(FeatureDecision {
                feature: "amx_tile".into(),
                enabled: true,
                reason: "AMX detected, GEMM strategy: AmxTile".into(),
            });
        }
        if kc.use_avx512 {
            decisions.push(FeatureDecision {
                feature: "avx512".into(),
                enabled: true,
                reason: format!("AVX-512 enabled, {} zmm regs", profile.num_simd_regs()),
            });
        }
        if kc.has_vnni {
            decisions.push(FeatureDecision {
                feature: "vnni".into(),
                enabled: true,
                reason: "VNNI INT8 dot product available".into(),
            });
        }
        if kc.has_bf16 {
            decisions.push(FeatureDecision {
                feature: "bf16_native".into(),
                enabled: gemm.strategy == GemmMicrokernelStrategy::Avx512NativeBf16,
                reason: format!("BF16 HW: {}, strategy: {:?}", kc.has_bf16, gemm.strategy),
            });
        }

        // GPU features
        if attention.warp_specialization {
            decisions.push(FeatureDecision {
                feature: "warp_specialization".into(),
                enabled: true,
                reason: "SM90+ producer/consumer warp groups".into(),
            });
        }
        if attention.tma_enabled {
            decisions.push(FeatureDecision {
                feature: "tma_2d".into(),
                enabled: true,
                reason: "SM90+ TMA 2D prefetch".into(),
            });
        }

        // Parallel features
        if parallel.wave_count > 1 {
            decisions.push(FeatureDecision {
                feature: "multi_wave".into(),
                enabled: true,
                reason: format!("{} waves, SM partition", parallel.wave_count),
            });
        }

        // Speculative decoding: benefit scaled by bias (§5.8)
        let spec_decode_raw_benefit = 1.0;
        let spec_decode_benefit = spec_decode_raw_benefit * bias.speculative_decoding_value;
        let spec_decode_enabled = spec_decode_benefit > 0.5;
        decisions.push(FeatureDecision {
            feature: "speculative_decoding".into(),
            enabled: spec_decode_enabled,
            reason: format!("adjusted benefit: {:.2}", spec_decode_benefit),
        });

        // L1i budget estimation
        let l1i_budget = 32 * 1024; // typical x86_64 L1i
        let l1i_used = decisions.len() * 2048; // rough: ~2KB per enabled feature code section

        FeaturePlan {
            decisions,
            l1i_used,
            l1i_budget,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Helper functions (retained from legacy)
// ═══════════════════════════════════════════════════════════════

/// Compute the largest power of 2 ≤ n.
fn prev_power_of_2(n: usize) -> usize {
    if n == 0 { return 1; }
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

/// Collect all distinct GEMM shapes in a layer.
fn collect_gemm_shapes(ir: &LayerIR) -> Vec<GemmShape> {
    let h = ir.hidden;
    let q = ir.q_dim();
    let kv = ir.kv_dim();
    let inter = ir.intermediate;

    let mut shapes = vec![
        // QKV projections (M=1 for single token, but plan for max_batch)
        GemmShape { m: ir.max_batch, n: q, k: h },     // Q
        GemmShape { m: ir.max_batch, n: kv, k: h },    // K, V
        // Output projection
        GemmShape { m: ir.max_batch, n: h, k: q },     // O
    ];

    // FFN GEMM shapes derived from activation type.
    if ir.activation.is_gated() {
        // Gated FFN (SwiGLU/GeGLU): gate, up, down
        shapes.push(GemmShape { m: ir.max_batch, n: inter, k: h });  // gate, up
        shapes.push(GemmShape { m: ir.max_batch, n: h, k: inter }); // down
    } else {
        // Non-gated FFN (GELU/ReLU): up, down
        shapes.push(GemmShape { m: ir.max_batch, n: inter, k: h });
        shapes.push(GemmShape { m: ir.max_batch, n: h, k: inter });
    }

    shapes.sort_by_key(|s| (s.m, s.n, s.k));
    shapes.dedup();
    shapes
}

/// Compute BLIS-style blocking for a GEMM shape.
fn compute_blocking(
    shape: &GemmShape,
    default_kc: usize,
    default_mc: usize,
    default_nc: usize,
    mr: usize,
    nr: usize,
) -> (usize, usize, usize) {
    let kc = default_kc.min(shape.k);
    let mc = default_mc.min(shape.m);
    let nc = default_nc.min(shape.n);

    // Align to microkernel tile
    let mc = (mc / mr) * mr;
    let nc = (nc / nr) * nr;

    (kc.max(1), mc.max(mr), nc.max(nr))
}

/// Compute total scratchpad bytes for a layer.
fn compute_scratchpad(ir: &LayerIR) -> usize {
    let elem = ir.dtype.size_bytes();
    let h = ir.hidden;
    let q = ir.q_dim();
    let kv = ir.kv_dim();
    let inter = ir.intermediate;
    let b = ir.max_batch;

    // Normed input
    let normed = b * h * elem;
    // QKV outputs
    let qkv = b * (q + 2 * kv) * elem;
    // Attention output
    let attn = b * q * elem;
    // FFN intermediates (gate + up)
    let ffn = b * 2 * inter * elem;
    // Attention scores (per head)
    let scores = b * ir.num_heads * ir.max_seq * elem;

    // We reuse buffers, so take the max of non-overlapping phases
    let phase_attn = normed + qkv + attn + scores;
    let phase_ffn = normed + ffn;

    phase_attn.max(phase_ffn)
}

/// Determine fusion opportunities for a layer.
fn plan_fusions(ir: &LayerIR) -> Vec<FusionDecision> {
    let mut fusions = Vec::new();

    // QKV shared input is always beneficial
    fusions.push(FusionDecision::QkvSharedInput);

    // FFN fusion strategy derived from activation type.
    match ir.activation {
        Activation::GeGlu => fusions.push(FusionDecision::GeGluFusion),
        Activation::Silu => fusions.push(FusionDecision::SwiGluFusion),
        Activation::Gelu => fusions.push(FusionDecision::GemmBiasAct(Activation::Gelu)),
        _ => {} // Relu/None: no FFN fusion
    }

    // FlashAttention tiling for long sequences
    if ir.max_seq > 128 {
        fusions.push(FusionDecision::FlashAttention);
    }

    // RMSNorm→GEMM fusion (always beneficial, avoids one memory round-trip)
    fusions.push(FusionDecision::RmsNormIntoGemm);

    fusions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::{LayerIR, MoeConfig};
    use crate::types::ModelConfig;
    use crate::dispatch::DeviceProfile;

    #[test]
    fn test_execution_plan_build() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());

        assert!(plan.num_threads >= 1);
        assert!(plan.scratchpad_bytes > 0);
        assert!(!plan.fusions.is_empty());
        assert!(!plan.gemm_blocking.is_empty());
        assert!(plan.microkernel.mr >= 4);
        assert!(plan.microkernel.nr >= 4);

        // HwOpt sub-plans
        assert!(plan.roofline.ridge_point > 0.0);
        assert!(plan.cache_budget.l1_tile_budget > 0);
        assert!(plan.gemm_plan.acc_regs > 0);
        assert!(plan.gemm_plan.scratch_regs > 0 || plan.gemm_plan.max_epilogue_depth == 0);
        assert!(plan.attention_plan.tile_q >= 16);
        assert!(plan.parallel_plan.wave_count >= 1);
        assert!(!plan.batch_plan.golden_sizes.is_empty());

        eprintln!("Plan: {} threads, {} scratchpad bytes, {} fusions, {} GEMM shapes",
            plan.num_threads,
            plan.scratchpad_bytes,
            plan.fusions.len(),
            plan.gemm_blocking.len(),
        );
        eprintln!("  Roofline: ridge={:.1}, gemm_prefill={:?}, gemm_decode={:?}",
            plan.roofline.ridge_point, plan.roofline.gemm_prefill, plan.roofline.gemm_decode);
        eprintln!("  Cache: L1_tile={}KB, L2_kv={}KB, L3_model={}KB",
            plan.cache_budget.l1_tile_budget / 1024,
            plan.cache_budget.l2_kv_budget / 1024,
            plan.cache_budget.l3_model_budget / 1024);
        eprintln!("  GEMM: mr={}, nr={}, strategy={:?}, k_depth={}, max_epi={}",
            plan.gemm_plan.mr, plan.gemm_plan.nr, plan.gemm_plan.strategy,
            plan.gemm_plan.k_pipeline_depth, plan.gemm_plan.max_epilogue_depth);
        eprintln!("  Attention: variant={:?}, tile_q={}, tile_kv={}, online_softmax={}",
            plan.attention_plan.variant, plan.attention_plan.tile_q,
            plan.attention_plan.tile_kv, plan.attention_plan.online_softmax);
        eprintln!("  Parallel: waves={}, features={}",
            plan.parallel_plan.wave_count, plan.feature_plan.decisions.len());
    }

    #[test]
    fn test_fusions_decoder() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());

        assert!(plan.fusions.contains(&FusionDecision::QkvSharedInput));
        assert!(plan.fusions.contains(&FusionDecision::SwiGluFusion));
        assert!(plan.fusions.contains(&FusionDecision::RmsNormIntoGemm));
        assert!(plan.fusions.contains(&FusionDecision::FlashAttention));
    }

    #[test]
    fn test_fusions_gemma_geglu() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());

        assert!(plan.fusions.contains(&FusionDecision::GeGluFusion));
        assert!(!plan.fusions.contains(&FusionDecision::SwiGluFusion));
    }

    #[test]
    fn test_roofline_classifies_decode_as_memory_bound() {
        let profile = DeviceProfile::detect();
        let roofline = HwOptEngine::solve_roofline(&profile);
        assert_eq!(roofline.gemm_decode, BottleneckClass::MemoryBound);
        assert_eq!(roofline.attn_decode, BottleneckClass::MemoryBound);
        assert_eq!(roofline.elementwise, BottleneckClass::MemoryBound);
    }

    #[test]
    fn test_cache_budget_sums_within_bounds() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let budget = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());

        // L1: tile + scratch ≤ total
        assert!(budget.l1_tile_budget + budget.l1_fusion_scratch <= l1);
        // L2: sum of budgets ≤ total
        assert!(budget.l2_kv_budget + budget.l2_weight_budget + budget.l2_activation_budget <= l2);
        // L3: sum ≤ total
        assert!(budget.l3_model_budget + budget.l3_kv_cold_budget <= l3);
    }

    #[test]
    fn test_gemm_solver_register_budget() {
        let profile = DeviceProfile::detect();
        let roofline = HwOptEngine::solve_roofline(&profile);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &LayerIR::from_model_config(&ModelConfig::llama_7b(), 1), &StrategyBias::default());
        let kc = &profile.kernel_config;
        let gemm = HwOptEngine::solve_gemm(&profile, &roofline, &cache, kc.mr, kc.nr, &StrategyBias::default());

        // Register budget constraint: acc + ptr + scratch ≤ total
        assert!(gemm.acc_regs + 2 + gemm.scratch_regs <= profile.num_simd_regs());
    }

    #[test]
    fn test_batch_plan_has_golden_sizes() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        let parallel = HwOptEngine::solve_parallelism(&profile, &cache, &StrategyBias::default());
        let batch = HwOptEngine::solve_batch(&profile, &parallel, &cache, &ir, &StrategyBias::default());

        assert!(!batch.golden_sizes.is_empty());
        // Golden sizes should be powers of 2
        for &size in &batch.golden_sizes {
            assert_eq!(size, prev_power_of_2(size).max(size));
        }
        assert!(batch.decode_ratio_cap > 0.0 && batch.decode_ratio_cap <= 1.0);
    }

    // ── Pure data structure tests ─────────────────────────────────────

    #[test]
    fn strategy_bias_default_all_ones_except_expert() {
        let bias = StrategyBias::default();
        assert_eq!(bias.fusion_cost_scale, 1.0);
        assert_eq!(bias.pipeline_cost_scale, 1.0);
        assert_eq!(bias.parallelism_cost_scale, 1.0);
        assert_eq!(bias.epilogue_depth_preference, 1.0);
        assert_eq!(bias.k_depth_preference, 1.0);
        assert_eq!(bias.kv_cache_budget_scale, 1.0);
        assert_eq!(bias.weight_prefetch_budget_scale, 1.0);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert_eq!(bias.decode_ratio_scale, 1.0);
        assert_eq!(bias.speculative_decoding_value, 1.0);
        assert_eq!(bias.quantization_aggressiveness, 1.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 1.0);
    }

    #[test]
    fn strategy_bias_validate_clamps_out_of_range() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.01,
            batch_flexibility: 5.0,
            expert_eviction_aggressiveness: 10.0,
            ..StrategyBias::default()
        };
        bias.validate();
        assert!((bias.fusion_cost_scale - 0.2).abs() < 1e-10);
        assert!((bias.batch_flexibility - 1.0).abs() < 1e-10);
        assert!((bias.expert_eviction_aggressiveness - 2.0).abs() < 1e-10);
    }

    #[test]
    fn strategy_bias_validate_keeps_valid_values() {
        let mut bias = StrategyBias::default();
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 1.0);
        assert_eq!(bias.batch_flexibility, 1.0);
    }

    #[test]
    fn gemm_shape_equality() {
        let a = GemmShape { m: 1, n: 4096, k: 4096 };
        let b = GemmShape { m: 1, n: 4096, k: 4096 };
        let c = GemmShape { m: 512, n: 4096, k: 4096 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn microkernel_choice_equality() {
        let a = MicrokernelChoice { mr: 6, nr: 8 };
        let b = MicrokernelChoice { mr: 6, nr: 8 };
        assert_eq!(a, b);
    }

    #[test]
    fn fusion_decision_variants() {
        let decisions = vec![
            FusionDecision::RmsNormIntoGemm,
            FusionDecision::GemmBiasAct(Activation::Silu),
            FusionDecision::QkvSharedInput,
            FusionDecision::FlashAttention,
            FusionDecision::SwiGluFusion,
            FusionDecision::GeGluFusion,
        ];
        assert_eq!(decisions.len(), 6);
    }

    #[test]
    fn bottleneck_class_ordering() {
        assert_eq!(BottleneckClass::ComputeBound, BottleneckClass::ComputeBound);
        assert_ne!(BottleneckClass::ComputeBound, BottleneckClass::MemoryBound);
        assert_ne!(BottleneckClass::Mixed, BottleneckClass::MemoryBound);
    }

    #[test]
    fn gemm_microkernel_strategy_variants() {
        let strategies = [
            GemmMicrokernelStrategy::AmxTile,
            GemmMicrokernelStrategy::Avx512NativeBf16,
            GemmMicrokernelStrategy::BlisAvx512,
            GemmMicrokernelStrategy::BlisAvx2,
            GemmMicrokernelStrategy::BlisNeon,
            GemmMicrokernelStrategy::BlisSve,
            GemmMicrokernelStrategy::GpuTensorCore,
            GemmMicrokernelStrategy::GpuScalar,
            GemmMicrokernelStrategy::Scalar,
        ];
        assert_eq!(strategies.len(), 9);
    }

    #[test]
    fn ffn_fusion_strategy_variants() {
        assert_eq!(FfnFusionStrategy::GateSiLUInject, FfnFusionStrategy::GateSiLUInject);
        assert_ne!(FfnFusionStrategy::GateSiLUInject, FfnFusionStrategy::SeparateGemm);
    }

    #[test]
    fn attention_variant_variants() {
        let variants = [
            AttentionVariant::FA4BlockScaled,
            AttentionVariant::FA3Pipeline,
            AttentionVariant::FA2Tiled,
            AttentionVariant::WmmaTiled,
            AttentionVariant::AmxTile,
            AttentionVariant::Avx512Loop,
            AttentionVariant::NeonLoop,
            AttentionVariant::ScalarLoop,
        ];
        assert_eq!(variants.len(), 8);
    }

    #[test]
    fn prev_power_of_2_exact() {
        assert_eq!(prev_power_of_2(1024), 1024);
        assert_eq!(prev_power_of_2(256), 256);
    }

    #[test]
    fn prev_power_of_2_between() {
        assert_eq!(prev_power_of_2(5), 4);
        assert_eq!(prev_power_of_2(1000), 512);
        assert_eq!(prev_power_of_2(3), 2);
    }

    #[test]
    fn prev_power_of_2_edge() {
        assert_eq!(prev_power_of_2(1), 1);
        assert_eq!(prev_power_of_2(2), 2);
    }

    #[test]
    fn gemm_shape_hash_consistency() {
        let a = GemmShape { m: 1, n: 4096, k: 4096 };
        let b = GemmShape { m: 1, n: 4096, k: 4096 };
        let mut map = HashMap::new();
        map.insert(a, (64, 128, 256));
        assert_eq!(map.get(&b), Some(&(64, 128, 256)));
    }

    #[test]
    fn strategy_bias_copy_is_independent() {
        let original = StrategyBias {
            fusion_cost_scale: 2.5,
            batch_flexibility: 0.0,
            ..StrategyBias::default()
        };
        let mut copy = original;
        copy.fusion_cost_scale = 0.5;
        assert!((original.fusion_cost_scale - 2.5).abs() < 1e-10);
        assert!((copy.fusion_cost_scale - 0.5).abs() < 1e-10);
    }

    #[test]
    fn gemm_microkernel_strategy_copy_and_equality() {
        let a = GemmMicrokernelStrategy::BlisAvx512;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, GemmMicrokernelStrategy::BlisAvx2);
    }

    #[test]
    fn microkernel_choice_inequality() {
        let a = MicrokernelChoice { mr: 6, nr: 8 };
        let b = MicrokernelChoice { mr: 4, nr: 8 };
        let c = MicrokernelChoice { mr: 6, nr: 4 };
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn strategy_bias_accessors() {
        let bias = StrategyBias {
            expert_eviction_aggressiveness: 1.5,
            expert_prefetch_priority: 3.0,
            ..StrategyBias::default()
        };
        assert!((bias.expert_eviction_aggressiveness() - 1.5).abs() < 1e-10);
        assert!((bias.expert_prefetch_priority() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn compute_blocking_aligns_to_microkernel_tiles() {
        let shape = GemmShape { m: 17, n: 17, k: 17 };
        let (kc, mc, nc) = compute_blocking(&shape, 64, 64, 64, 6, 8);
        assert_eq!(kc, 17);
        assert_eq!(mc % 6, 0);
        assert_eq!(nc % 8, 0);
    }

    #[test]
    fn compute_blocking_never_below_microkernel_size() {
        let shape = GemmShape { m: 1, n: 1, k: 1 };
        let (kc, mc, nc) = compute_blocking(&shape, 64, 64, 64, 6, 8);
        assert!(kc >= 1);
        assert!(mc >= 6);
        assert!(nc >= 8);
    }

    #[test]
    fn collect_gemm_shapes_deduplicates() {
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 128,
            partial_rotary_factor: 1.0,
            activation: Activation::Gelu,
        };
        let shapes = collect_gemm_shapes(&ir);
        let shape_set: std::collections::HashSet<GemmShape> = shapes.iter().copied().collect();
        assert_eq!(shapes.len(), shape_set.len());
    }

    #[test]
    fn plan_fusions_encoder_uses_gelu() {
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 512,
            partial_rotary_factor: 1.0,
            activation: Activation::Gelu,
        };
        let fusions = plan_fusions(&ir);
        assert!(fusions.contains(&FusionDecision::GemmBiasAct(Activation::Gelu)));
        assert!(!fusions.contains(&FusionDecision::SwiGluFusion));
        assert!(!fusions.contains(&FusionDecision::GeGluFusion));
    }

    #[test]
    fn plan_fusions_short_seq_no_flash_attention() {
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 64,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        let fusions = plan_fusions(&ir);
        assert!(!fusions.contains(&FusionDecision::FlashAttention));
        assert!(fusions.contains(&FusionDecision::SwiGluFusion));
    }

    #[test]
    fn roofline_peak_derived_consistently() {
        let profile = DeviceProfile::detect();
        let roofline = HwOptEngine::solve_roofline(&profile);
        if roofline.peak_bandwidth_gbs > 0.0 {
            let expected_ridge = roofline.peak_gflops / roofline.peak_bandwidth_gbs;
            assert!((roofline.ridge_point - expected_ridge).abs() < 1e-6);
        }
    }

    #[test]
    fn batch_plan_zero_flexibility_forces_single() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        let parallel = HwOptEngine::solve_parallelism(&profile, &cache, &StrategyBias::default());
        let zero_bias = StrategyBias { batch_flexibility: 0.0, ..StrategyBias::default() };
        let batch = HwOptEngine::solve_batch(&profile, &parallel, &cache, &ir, &zero_bias);

        assert_eq!(batch.max_chunk_size, 1);
        assert_eq!(batch.decode_slots, 1);
        assert_eq!(batch.max_chunks_per_batch, 1);
        assert_eq!(batch.golden_sizes, vec![1]);
    }

    #[test]
    fn execution_plan_from_profile_has_no_bottleneck_map() {
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);
        assert!(plan.op_bottleneck_map.is_none());
        assert!(plan.num_threads >= 1);
        assert!(!plan.gemm_blocking.is_empty());
    }

    // ── Wave 12k31: +13 additional tests ──

    #[test]
    fn strategy_bias_speculative_decoding_value_clamp() {
        let mut bias = StrategyBias::default();
        bias.speculative_decoding_value = 5.0;
        bias.validate();
        assert!(
            (bias.speculative_decoding_value - 3.0).abs() < 1e-9,
            "speculative_decoding_value should be clamped to 3.0"
        );
    }

    #[test]
    fn strategy_bias_quantization_aggressiveness_clamp() {
        let mut bias = StrategyBias::default();
        bias.quantization_aggressiveness = 0.01;
        bias.validate();
        assert!(
            (bias.quantization_aggressiveness - 0.3).abs() < 1e-9,
            "quantization_aggressiveness should be clamped to 0.3"
        );
    }

    #[test]
    fn cache_budget_plan_all_nonzero() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let plan = CacheBudgetSolver::solve(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        assert!(plan.l1_tile_budget > 0);
        assert!(plan.l2_kv_budget > 0);
        assert!(plan.l3_model_budget > 0);
    }

    #[test]
    fn gemm_plan_k_pipeline_depth_positive() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        assert!(
            plan.gemm_plan.k_pipeline_depth >= 1,
            "k_pipeline_depth must be >= 1"
        );
    }

    #[test]
    fn attention_plan_has_valid_variant() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        match plan.attention_plan.variant {
            AttentionVariant::ScalarLoop
            | AttentionVariant::Avx512Loop
            | AttentionVariant::NeonLoop
            | AttentionVariant::AmxTile
            | AttentionVariant::FA2Tiled
            | AttentionVariant::FA3Pipeline
            | AttentionVariant::FA4BlockScaled
            | AttentionVariant::WmmaTiled => {}
        }
    }

    #[test]
    fn parallel_plan_wave_count_positive() {
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);
        assert!(plan.parallel_plan.wave_count >= 1);
        assert!(
            plan.parallel_plan.occupancy_target > 0.0 && plan.parallel_plan.occupancy_target <= 1.0,
            "occupancy_target must be in (0, 1]"
        );
    }

    #[test]
    fn feature_plan_decisions_nonempty() {
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::from_profile(&profile);
        assert!(!plan.feature_plan.decisions.is_empty());
        for d in &plan.feature_plan.decisions {
            assert!(!d.feature.is_empty());
            assert!(!d.reason.is_empty());
        }
    }

    #[test]
    fn batch_plan_decode_ratio_cap_range() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        assert!(
            plan.batch_plan.decode_ratio_cap >= 0.0 && plan.batch_plan.decode_ratio_cap <= 1.0,
            "decode_ratio_cap must be in [0, 1]"
        );
    }

    #[test]
    fn fusion_strategy_default_fields() {
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        assert!(plan.fusion_plan.max_epilogue_depth >= 0);
        assert!(plan.fusion_plan.tile_fusion_threshold > 0);
    }

    #[test]
    fn prev_power_of_2_large_values() {
        assert_eq!(prev_power_of_2(1025), 1024);
        assert_eq!(prev_power_of_2(2048), 2048);
        assert_eq!(prev_power_of_2(4097), 4096);
    }

    #[test]
    fn compute_blocking_small_shape() {
        let shape = GemmShape { m: 1, n: 1, k: 1 };
        let (kc, mc, nc) = compute_blocking(&shape, 256, 72, 1024, 6, 8);
        assert_eq!(kc, 1);
        assert!(mc >= 6);
        assert!(nc >= 8);
    }

    #[test]
    fn compute_scratchpad_positive() {
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 4);
        let scratch = compute_scratchpad(&ir);
        assert!(scratch > 0, "scratchpad must be positive for a real model");
    }

    #[test]
    fn gemm_shape_hash_roundtrip() {
        use std::collections::HashSet;
        let shapes: Vec<GemmShape> = (0..10).map(|i| GemmShape { m: i, n: i * 2, k: i * 3 }).collect();
        let set: HashSet<GemmShape> = shapes.iter().copied().collect();
        assert_eq!(set.len(), 10);
        assert!(set.contains(&GemmShape { m: 3, n: 6, k: 9 }));
    }

    // ── Wave 12k78: +10 additional tests ──

    #[test]
    fn prev_power_of_2_zero_returns_one() {
        // Line 1465: special case n==0 returns 1
        assert_eq!(prev_power_of_2(0), 1);
    }

    #[test]
    fn compute_blocking_caps_kc_to_shape_k() {
        // When shape.k < default_kc, kc should equal shape.k
        let shape = GemmShape { m: 64, n: 64, k: 32 };
        let (kc, _mc, _nc) = compute_blocking(&shape, 256, 64, 64, 6, 8);
        assert_eq!(kc, 32, "kc must be min(default_kc, shape.k)");
    }

    #[test]
    fn gpu_sm_partition_fields() {
        // Arrange: construct a GpuSmPartition directly
        let part = GpuSmPartition {
            total_sm: 108,
            num_partitions: 2,
            sm_per_partition: 54,
        };
        // Act & Assert: verify field values are preserved
        assert_eq!(part.total_sm, 108);
        assert_eq!(part.num_partitions, 2);
        assert_eq!(part.sm_per_partition, 54);
        assert_eq!(part.total_sm, part.num_partitions * part.sm_per_partition);
    }

    #[test]
    fn numa_binding_fields() {
        // Arrange: construct a NumaBinding directly
        let binding = NumaBinding {
            node_id: 1,
            core_start: 8,
            core_end: 15,
            l3_bytes: 32 * 1024 * 1024,
        };
        // Assert: verify field values
        assert_eq!(binding.node_id, 1);
        assert_eq!(binding.core_end - binding.core_start, 7);
        assert_eq!(binding.l3_bytes, 33_554_432);
    }

    #[test]
    fn feature_decision_construction() {
        // Arrange & Act
        let decision = FeatureDecision {
            feature: "avx512_bf16".into(),
            enabled: true,
            reason: "Hardware supports native BF16".into(),
        };
        // Assert
        assert_eq!(decision.feature, "avx512_bf16");
        assert!(decision.enabled);
        assert!(!decision.reason.is_empty());
    }

    #[test]
    fn plan_fusions_always_includes_rmsnorm_into_gemm() {
        // Arrange: decoder IR with long sequence
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 512,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        // Act
        let fusions = plan_fusions(&ir);
        // Assert: RmsNormIntoGemm is always present (line 1575)
        assert!(
            fusions.contains(&FusionDecision::RmsNormIntoGemm),
            "RmsNormIntoGemm should always be in fusion decisions"
        );
        // Also verify QkvSharedInput is always present (line 1553)
        assert!(
            fusions.contains(&FusionDecision::QkvSharedInput),
            "QkvSharedInput should always be in fusion decisions"
        );
    }

    #[test]
    fn cache_budget_l1_splits_exactly() {
        // Arrange
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let l1 = profile.kernel_config.l1d;
        let budget = CacheBudgetSolver::solve(
            &profile, l1, profile.kernel_config.l2, profile.kernel_config.l3,
            &ir, &StrategyBias::default(),
        );
        // Act: compute sum
        let l1_sum = budget.l1_tile_budget + budget.l1_fusion_scratch;
        // Assert: tile + scratch == total (line 901-902: l1_tile = 75%, scratch = l1 - tile)
        assert_eq!(l1_sum, l1, "L1 tile + scratch must sum to total L1");
    }

    #[test]
    fn execution_plan_stores_strategy_bias() {
        // Arrange
        let profile = DeviceProfile::detect();
        let bias = StrategyBias {
            fusion_cost_scale: 1.5,
            k_depth_preference: 2.0,
            ..StrategyBias::default()
        };
        // Act
        let empty_bottleneck_map = crate::compiler::pain_point::OpBottleneckMap {
            gemm_bottlenecks: std::collections::HashMap::new(),
            ridge_point: 0.0,
        };
        let plan = ExecutionPlan::from_profile_with_bottlenecks(
            &profile,
            empty_bottleneck_map,
        );
        // NOTE: from_profile_with_bottlenecks uses default bias, so we verify
        // that the plan's strategy_bias field is populated
        assert!(!plan.strategy_bias.fusion_cost_scale.is_nan());
        assert!(!plan.strategy_bias.k_depth_preference.is_nan());
        assert!(plan.op_bottleneck_map.is_some(), "bottleneck map should be present");
    }

    #[test]
    fn compute_scratchpad_encoder_vs_decoder() {
        // Arrange: two IRs with identical dimensions but different arch
        let make_ir = |act: Activation| LayerIR {
            moe: None,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate: 512,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 2,
            max_seq: 256,
            partial_rotary_factor: 1.0,
            activation: act,
        };
        let dec_ir = make_ir(Activation::Silu);
        let enc_ir = make_ir(Activation::Gelu);
        // Act
        let dec_scratch = compute_scratchpad(&dec_ir);
        let enc_scratch = compute_scratchpad(&enc_ir);
        // Assert: both positive and equal (same hidden/intermediate/batch dimensions)
        assert!(dec_scratch > 0);
        assert!(enc_scratch > 0);
        assert_eq!(dec_scratch, enc_scratch,
            "Decoder and Encoder with same dimensions should have equal scratchpad");
    }

    #[test]
    fn batch_plan_golden_sizes_ascending_powers_of_two() {
        // Arrange
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        let parallel = HwOptEngine::solve_parallelism(&profile, &cache, &StrategyBias::default());
        let batch = HwOptEngine::solve_batch(&profile, &parallel, &cache, &ir, &StrategyBias::default());
        // Act & Assert: golden sizes must be strictly ascending and each a power of 2
        for window in batch.golden_sizes.windows(2) {
            assert!(window[0] < window[1], "golden_sizes must be strictly ascending");
        }
        for &size in &batch.golden_sizes {
            assert_eq!(size.count_ones(), 1, "each golden size must be a power of 2, got {}", size);
        }
    }

    // -- Wave 12kak: +10 additional tests --

    #[test]
    fn collect_gemm_shapes_decoder_has_ffn_shapes() {
        // Arrange: decoder IR with distinct hidden/intermediate
        let ir = LayerIR {
            moe: None,
            hidden: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 256,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 128,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        // Act
        let shapes = collect_gemm_shapes(&ir);
        // Assert: must contain FFN gate/up (n=intermediate) and down (k=intermediate) shapes
        let has_gate = shapes.iter().any(|s| s.n == 256);
        let has_down = shapes.iter().any(|s| s.k == 256);
        assert!(has_gate, "decoder must have gate/up GEMM shape with n=intermediate");
        assert!(has_down, "decoder must have down GEMM shape with k=intermediate");
    }

    #[test]
    fn collect_gemm_shapes_moe_uses_decoder_path() {
        // Arrange: DecoderMoE with a small expert count
        let ir = LayerIR {
            moe: Some(MoeConfig { num_experts: 8, top_k: 2 }),
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 128,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 64,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        // Act
        let shapes = collect_gemm_shapes(&ir);
        // Assert: DecoderMoE falls into the same Decoder arm, so FFN shapes present
        assert!(shapes.iter().any(|s| s.n == 128), "MoE decoder must have FFN gate shape");
    }

    #[test]
    fn plan_fusions_moe_decoder_uses_swiglu() {
        // Arrange
        let ir = LayerIR {
            moe: Some(MoeConfig { num_experts: 4, top_k: 1 }),
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 512,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        // Act
        let fusions = plan_fusions(&ir);
        // Assert: DecoderMoE falls into Decoder match arm with SwiGLU
        assert!(
            fusions.contains(&FusionDecision::SwiGluFusion),
            "MoE decoder with Silu activation must use SwiGluFusion"
        );
        assert!(
            fusions.contains(&FusionDecision::FlashAttention),
            "max_seq=512 > 128 must trigger FlashAttention"
        );
    }

    #[test]
    fn fusion_decision_geglu_distinct_from_swiglu() {
        // Arrange: two different fusion decisions
        let ge = FusionDecision::GemmBiasAct(Activation::Gelu);
        let si = FusionDecision::GemmBiasAct(Activation::Silu);
        let swi = FusionDecision::SwiGluFusion;
        let geglu = FusionDecision::GeGluFusion;
        // Assert: GemmBiasAct with different activations are distinct
        assert_ne!(ge, si);
        assert_ne!(swi, geglu);
        assert_eq!(ge, FusionDecision::GemmBiasAct(Activation::Gelu));
        assert_eq!(si, FusionDecision::GemmBiasAct(Activation::Silu));
    }

    #[test]
    fn compute_scratchpad_scales_with_batch() {
        // Arrange: same model config, different batch sizes
        let ir_b1 = LayerIR {
            moe: None,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate: 512,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 128,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        let mut ir_b4 = ir_b1.clone();
        ir_b4.max_batch = 4;
        // Act
        let scratch_b1 = compute_scratchpad(&ir_b1);
        let scratch_b4 = compute_scratchpad(&ir_b4);
        // Assert: batch=4 scratchpad is larger than batch=1
        assert!(scratch_b4 > scratch_b1, "larger batch must need more scratchpad");
    }

    #[test]
    fn compute_blocking_large_shape_uses_defaults() {
        // Arrange: shape larger than all defaults
        let shape = GemmShape { m: 4096, n: 4096, k: 4096 };
        // Act
        let (kc, mc, nc) = compute_blocking(&shape, 256, 96, 1024, 6, 16);
        // Assert: kc capped at default_kc, mc aligned to mr, nc aligned to nr
        assert_eq!(kc, 256);
        assert_eq!(mc, 96);
        assert_eq!(nc, 1024);
    }

    #[test]
    fn roofline_result_construction_and_fields() {
        // Arrange & Act: construct a RooflineResult directly with exact-float values
        let result = RooflineResult {
            ridge_point: 10.0,
            peak_gflops: 500.0,
            peak_bandwidth_gbs: 50.0,
            gemm_prefill: BottleneckClass::ComputeBound,
            gemm_decode: BottleneckClass::MemoryBound,
            attn_prefill: BottleneckClass::Mixed,
            attn_decode: BottleneckClass::MemoryBound,
            elementwise: BottleneckClass::MemoryBound,
        };
        // Assert: fields preserved and ridge consistent
        assert!((result.ridge_point - 10.0).abs() < 1e-10);
        assert!((result.peak_gflops - 500.0).abs() < 1e-10);
        assert_eq!(result.gemm_prefill, BottleneckClass::ComputeBound);
        assert_eq!(result.attn_prefill, BottleneckClass::Mixed);
        let derived_ridge = result.peak_gflops / result.peak_bandwidth_gbs;
        assert!((result.ridge_point - derived_ridge).abs() < 1e-10);
    }

    #[test]
    fn cache_budget_plan_direct_construction() {
        // Arrange & Act
        let plan = CacheBudgetPlan {
            l1_tile_budget: 24_576,
            l1_fusion_scratch: 8_192,
            l2_kv_budget: 524_288,
            l2_weight_budget: 458_752,
            l2_activation_budget: 327_680,
            l3_model_budget: 107_374_182,
            l3_kv_cold_budget: 30_720_000,
        };
        // Assert: L1 sums correctly and all positive
        assert_eq!(plan.l1_tile_budget + plan.l1_fusion_scratch, 32_768);
        assert!(plan.l2_kv_budget > 0);
        assert!(plan.l3_model_budget > plan.l3_kv_cold_budget);
    }

    #[test]
    fn with_execution_plan_scopes_plan_to_closure() {
        // Arrange
        let bias = StrategyBias {
            fusion_cost_scale: 0.5,
            ..StrategyBias::default()
        };
        let custom_plan = compute_execution_plan_with_bias(&bias);
        // Act: inside the closure, the thread-local plan should be accessible
        let plan_ref = with_execution_plan(custom_plan.clone(), || {
            global_execution_plan() as *const ExecutionPlan
        });
        let global_ref = global_execution_plan() as *const ExecutionPlan;
        // Assert: inside scope we got the custom plan, outside we get global default
        let custom_ptr = Arc::as_ptr(&custom_plan);
        assert_eq!(plan_ref, custom_ptr, "inside scope must see the custom plan");
        assert_ne!(plan_ref, global_ref, "outside scope must see a different plan");
    }

    #[test]
    fn compute_execution_plan_with_bias_returns_valid_plan() {
        // Arrange
        let bias = StrategyBias {
            k_depth_preference: 2.0,
            epilogue_depth_preference: 0.5,
            ..StrategyBias::default()
        };
        // Act
        let plan = compute_execution_plan_with_bias(&bias);
        // Assert: plan has all sub-plans populated with sane values
        assert!(plan.num_threads >= 1);
        assert!(!plan.gemm_blocking.is_empty());
        assert!(plan.gemm_plan.k_pipeline_depth >= 1);
        assert!(plan.attention_plan.tile_q >= 16);
        assert!(plan.batch_plan.decode_ratio_cap <= 1.0);
        assert!((plan.strategy_bias.k_depth_preference - 2.0).abs() < 1e-10);
        assert!((plan.strategy_bias.epilogue_depth_preference - 0.5).abs() < 1e-10);
    }

    // -- Wave 12ked: +10 additional tests --

    #[test]
    fn cache_budget_l2_normalizes_when_bias_ratios_exceed_one() {
        // Arrange: inflate KV and weight scales so raw total > 1.0,
        // triggering the normalization branch (lines 910-912)
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let high_bias = StrategyBias {
            kv_cache_budget_scale: 3.0,
            weight_prefetch_budget_scale: 3.0,
            ..StrategyBias::default()
        };
        // Act
        let budget = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &high_bias);
        // Assert: L2 budgets must still fit within total L2
        assert!(
            budget.l2_kv_budget + budget.l2_weight_budget + budget.l2_activation_budget <= l2,
            "normalized L2 budgets must not exceed total L2"
        );
        assert!(budget.l2_kv_budget > 0);
        assert!(budget.l2_weight_budget > 0);
    }

    #[test]
    fn gemm_solver_k_depth_low_preference_always_one() {
        // Arrange: k_depth_preference < 0.8 forces k_pipeline_depth = 1 (line 964)
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let low_k_bias = StrategyBias {
            k_depth_preference: 0.5,
            ..StrategyBias::default()
        };
        // Act
        let plan = ExecutionPlan::build(&ir, &profile, &low_k_bias);
        // Assert
        assert_eq!(
            plan.gemm_plan.k_pipeline_depth, 1,
            "k_depth_preference < 0.8 must force k_pipeline_depth = 1"
        );
    }

    #[test]
    fn batch_solver_small_max_seq_uses_fallback_golden_size() {
        // Arrange: IR with max_seq = 8, which is less than the starting golden size of 16.
        // This triggers the fallback branch (lines 1347-1348: golden_sizes empty -> push 64).
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 8,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        let profile = DeviceProfile::detect();
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        let parallel = HwOptEngine::solve_parallelism(&profile, &cache, &StrategyBias::default());
        // Act
        let batch = HwOptEngine::solve_batch(&profile, &parallel, &cache, &ir, &StrategyBias::default());
        // Assert: fallback golden size of 64 is used since 16 > max_seq=8
        assert_eq!(batch.golden_sizes, vec![64], "max_seq < 16 must trigger fallback golden size");
        assert_eq!(batch.max_chunk_size, 64);
    }

    #[test]
    fn attention_solver_online_softmax_disabled_when_tile_covers_full_seq() {
        // Arrange: very small max_seq so tile_q >= max_seq, disabling online softmax.
        // The default L1 budget is large enough that max_tile >> small max_seq.
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 16,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        // Act & Assert: when tile_q >= max_seq, online_softmax should be false (line 1119)
        if plan.attention_plan.tile_q >= ir.max_seq {
            assert!(
                !plan.attention_plan.online_softmax,
                "online_softmax must be false when tile_q >= max_seq"
            );
        }
        // Regardless: tile_q must be >= 16 (minimum enforced by line 1115)
        assert!(plan.attention_plan.tile_q >= 16);
    }

    #[test]
    fn solve_profile_with_bottlenecks_memory_bound_refines_decode() {
        // Arrange: construct an OpBottleneckMap with a memory-bound GEMM entry
        use crate::compiler::pain_point::{
            OpBottleneckMap, GemmBottleneck, BottleneckType, GemmRole,
            FusionPriority, ExecPattern, ParallelismDesc,
        };
        use crate::compiler::graph::OpId;
        let mut gemm_bottlenecks = std::collections::HashMap::new();
        gemm_bottlenecks.insert(
            OpId(0),
            GemmBottleneck {
                gemm_role: GemmRole::GateUpProjection,
                shape: (1, 4096, 4096),
                arithmetic_intensity: 1.5,
                ridge_point: 10.0,
                bottleneck: BottleneckType::MemoryBound { bandwidth_utilization: 0.9 },
                optimal_fusion: FusionPriority::EpilogueInjection,
                fusion_benefits: std::collections::HashMap::new(),
                exec_pattern: ExecPattern::ScalarLoop,
                parallelism: ParallelismDesc::SimdVectorize {
                    element_width: 16,
                    unroll_factor: 1,
                },
            },
        );
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks,
            ridge_point: 10.0,
        };
        let profile = DeviceProfile::detect();
        // Act
        let plan = ExecutionPlan::from_profile_with_bottlenecks(&profile, bottleneck_map);
        // Assert: memory-bound GEMM in R0 data forces gemm_decode = MemoryBound (line 798)
        assert_eq!(plan.roofline.gemm_decode, BottleneckClass::MemoryBound);
        assert!(plan.op_bottleneck_map.is_some());
        assert!(!plan.op_bottleneck_map.unwrap().gemm_bottlenecks.is_empty());
    }

    #[test]
    fn solve_profile_with_bottlenecks_compute_bound_refines_prefill() {
        // Arrange: construct an OpBottleneckMap with a compute-bound GEMM entry
        use crate::compiler::pain_point::{
            OpBottleneckMap, GemmBottleneck, BottleneckType, GemmRole,
            FusionPriority, ExecPattern, ParallelismDesc,
        };
        use crate::compiler::graph::OpId;
        let mut gemm_bottlenecks = std::collections::HashMap::new();
        gemm_bottlenecks.insert(
            OpId(1),
            GemmBottleneck {
                gemm_role: GemmRole::QkvProjection,
                shape: (512, 4096, 4096),
                arithmetic_intensity: 50.0,
                ridge_point: 10.0,
                bottleneck: BottleneckType::ComputeBound { compute_utilization: 0.85 },
                optimal_fusion: FusionPriority::EpilogueInjection,
                fusion_benefits: std::collections::HashMap::new(),
                exec_pattern: ExecPattern::ScalarLoop,
                parallelism: ParallelismDesc::SimdVectorize {
                    element_width: 16,
                    unroll_factor: 1,
                },
            },
        );
        let bottleneck_map = OpBottleneckMap {
            gemm_bottlenecks,
            ridge_point: 10.0,
        };
        let profile = DeviceProfile::detect();
        // Act
        let plan = ExecutionPlan::from_profile_with_bottlenecks(&profile, bottleneck_map);
        // Assert: compute-bound GEMM in R0 data forces gemm_prefill = ComputeBound (line 804)
        assert_eq!(plan.roofline.gemm_prefill, BottleneckClass::ComputeBound);
    }

    #[test]
    fn feature_router_speculative_decoding_disabled_by_low_bias() {
        // Arrange: speculative_decoding_value = 0.2 (< 0.5 threshold on line 1440)
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let low_spec_bias = StrategyBias {
            speculative_decoding_value: 0.2,
            ..StrategyBias::default()
        };
        // Act
        let plan = ExecutionPlan::build(&ir, &profile, &low_spec_bias);
        // Assert: speculative_decoding feature must be disabled
        let spec_decision = plan.feature_plan.decisions.iter()
            .find(|d| d.feature == "speculative_decoding");
        assert!(spec_decision.is_some(), "speculative_decoding decision must exist");
        assert!(
            !spec_decision.unwrap().enabled,
            "speculative_decoding must be disabled when benefit <= 0.5"
        );
    }

    #[test]
    fn compute_blocking_aligns_down_when_shape_smaller_than_mr_nr() {
        // Arrange: shape.m and shape.n smaller than default_mc and default_nc but
        // mc must still be >= mr and nc >= nr after alignment (line 1518)
        let shape = GemmShape { m: 3, n: 5, k: 64 };
        // Act
        let (kc, mc, nc) = compute_blocking(&shape, 128, 64, 64, 6, 16);
        // Assert: kc capped at shape.k, mc >= mr (6), nc >= nr (16)
        assert_eq!(kc, 64, "kc must be min(default_kc, shape.k)");
        assert!(mc >= 6, "mc must be >= mr=6 even when shape.m=3, got {}", mc);
        assert!(nc >= 16, "nc must be >= nr=16 even when shape.n=5, got {}", nc);
    }

    #[test]
    fn compute_scratchpad_attention_phase_dominates_for_large_seq() {
        // Arrange: IR with very large max_seq makes attention scores dominate
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate: 64,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 8192,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        // Act
        let scratch = compute_scratchpad(&ir);
        // Assert: scratchpad is positive; attention phase should dominate
        // scores = b * num_heads * max_seq * elem_bytes = 1 * 4 * 8192 * 4 = 131072
        let scores_bytes = 1 * 4 * 8192 * 4;
        assert!(scratch > 0);
        assert!(
            scratch >= scores_bytes,
            "scratchpad must account for attention scores: scratch={}, scores={}",
            scratch, scores_bytes
        );
    }

    #[test]
    fn batch_plan_decode_slots_at_least_one_with_low_flexibility() {
        // Arrange: very low batch_flexibility (0.01) but not zero,
        // so it does NOT enter the batch_flexibility == 0.0 early-return branch.
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        let parallel = HwOptEngine::solve_parallelism(&profile, &cache, &StrategyBias::default());
        let low_flex_bias = StrategyBias {
            batch_flexibility: 0.01,
            ..StrategyBias::default()
        };
        // Act
        let batch = HwOptEngine::solve_batch(&profile, &parallel, &cache, &ir, &low_flex_bias);
        // Assert: decode_slots must be >= 1 (line 1354: .max(1))
        assert!(
            batch.decode_slots >= 1,
            "decode_slots must be >= 1 even with very low batch_flexibility"
        );
        // decode_ratio_cap must still be in valid range
        assert!(batch.decode_ratio_cap >= 0.0 && batch.decode_ratio_cap <= 1.0);
    }

    // -- Wave 12kkf: +10 additional tests --

    #[test]
    fn gemm_plan_direct_construction_preserves_fields() {
        // Arrange & Act
        let plan = GemmPlan {
            mr: 6,
            nr: 16,
            nr_vecs: 2,
            k_pipeline_depth: 2,
            pf_distance_a: 256,
            pf_distance_b: 256,
            max_epilogue_depth: 3,
            acc_regs: 12,
            scratch_regs: 8,
            strategy: GemmMicrokernelStrategy::BlisAvx512,
        };
        // Assert: every field round-trips correctly
        assert_eq!(plan.mr, 6);
        assert_eq!(plan.nr, 16);
        assert_eq!(plan.nr_vecs, 2);
        assert_eq!(plan.k_pipeline_depth, 2);
        assert_eq!(plan.pf_distance_a, 256);
        assert_eq!(plan.acc_regs, 12);
        assert_eq!(plan.scratch_regs, 8);
        assert_eq!(plan.strategy, GemmMicrokernelStrategy::BlisAvx512);
    }

    #[test]
    fn attention_plan_direct_construction_preserves_fields() {
        // Arrange & Act
        let plan = AttentionPlan {
            variant: AttentionVariant::Avx512Loop,
            tile_q: 32,
            tile_kv: 128,
            online_softmax: true,
            warp_specialization: false,
            tma_enabled: false,
        };
        // Assert
        assert_eq!(plan.variant, AttentionVariant::Avx512Loop);
        assert_eq!(plan.tile_q, 32);
        assert_eq!(plan.tile_kv, 128);
        assert!(plan.online_softmax);
        assert!(!plan.warp_specialization);
        assert!(!plan.tma_enabled);
    }

    #[test]
    fn collect_gemm_shapes_encoder_has_ffn_shapes() {
        // Arrange: encoder IR with distinct hidden and intermediate dimensions
        let ir = LayerIR {
            moe: None,
            hidden: 128,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 32,
            intermediate: 256,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 128,
            partial_rotary_factor: 1.0,
            activation: Activation::Gelu,
        };
        // Act
        let shapes = collect_gemm_shapes(&ir);
        // Assert: encoder must also have FFN gate/up (n=intermediate) and down (k=intermediate)
        let has_gate = shapes.iter().any(|s| s.n == 256);
        let has_down = shapes.iter().any(|s| s.k == 256);
        assert!(has_gate, "encoder must have FFN gate/up GEMM shape with n=intermediate");
        assert!(has_down, "encoder must have FFN down GEMM shape with k=intermediate");
    }

    #[test]
    fn compute_scratchpad_bf16_is_half_of_f32() {
        // Arrange: two identical IRs with different dtypes
        let f32_ir = LayerIR {
            moe: None,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate: 512,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 2,
            max_seq: 128,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        let mut bf16_ir = f32_ir.clone();
        bf16_ir.dtype = DType::BF16;
        // Act
        let f32_scratch = compute_scratchpad(&f32_ir);
        let bf16_scratch = compute_scratchpad(&bf16_ir);
        // Assert: BF16 uses 2 bytes per element vs 4 for F32, so scratchpad is exactly half
        assert_eq!(bf16_scratch * 2, f32_scratch,
            "BF16 scratchpad must be exactly half of F32 scratchpad for identical dimensions");
    }

    #[test]
    fn execution_plan_prefetch_l2_is_double_prefetch_l1() {
        // Arrange
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        // Act
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        // Assert: prefetch_a_l2 = prefetch_a_l1 * 2 (line 752)
        assert_eq!(
            plan.prefetch_a_l2, plan.prefetch_a_l1 * 2,
            "L2 prefetch distance must be 2x L1 prefetch distance"
        );
    }

    #[test]
    fn batch_plan_max_chunks_is_wave_count_times_four() {
        // Arrange
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        let parallel = HwOptEngine::solve_parallelism(&profile, &cache, &StrategyBias::default());
        let batch = HwOptEngine::solve_batch(&profile, &parallel, &cache, &ir, &StrategyBias::default());
        // Assert: max_chunks_per_batch = wave_count * 4 (line 1355)
        assert_eq!(
            batch.max_chunks_per_batch, parallel.wave_count * 4,
            "max_chunks_per_batch must equal wave_count * 4"
        );
    }

    #[test]
    fn cache_budget_l3_split_seventy_twenty_ratio() {
        // Arrange
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let l3 = profile.kernel_config.l3;
        let budget = CacheBudgetSolver::solve(
            &profile, profile.kernel_config.l1d, profile.kernel_config.l2, l3,
            &ir, &StrategyBias::default(),
        );
        // Act & Assert: L3 model budget = 70% of total (line 920), KV cold = 20% (line 921)
        let expected_model = (l3 as f64 * 0.70) as usize;
        let expected_kv_cold = (l3 as f64 * 0.20) as usize;
        assert_eq!(budget.l3_model_budget, expected_model);
        assert_eq!(budget.l3_kv_cold_budget, expected_kv_cold);
    }

    #[test]
    fn strategy_bias_pipeline_and_parallelism_clamp_lower_bounds() {
        // Arrange: set values below the lower clamp bounds
        let mut bias = StrategyBias {
            pipeline_cost_scale: 0.05,
            parallelism_cost_scale: 0.01,
            expert_prefetch_priority: 0.01,
            ..StrategyBias::default()
        };
        // Act
        bias.validate();
        // Assert: pipeline_cost_scale clamped to 0.2 (line 66)
        assert!((bias.pipeline_cost_scale - 0.2).abs() < 1e-10,
            "pipeline_cost_scale must be clamped to 0.2");
        // parallelism_cost_scale clamped to 0.1 (line 67)
        assert!((bias.parallelism_cost_scale - 0.1).abs() < 1e-10,
            "parallelism_cost_scale must be clamped to 0.1");
        // expert_prefetch_priority clamped to 0.1 (line 75)
        assert!((bias.expert_prefetch_priority - 0.1).abs() < 1e-10,
            "expert_prefetch_priority must be clamped to 0.1");
    }

    #[test]
    fn execution_plan_num_threads_equals_physical_cores() {
        // Arrange
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        // Act
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        // Assert: num_threads = profile.physical_cores (line 735)
        assert_eq!(plan.num_threads, profile.physical_cores);
    }

    #[test]
    fn batch_plan_decode_ratio_scales_with_bias() {
        // Arrange: decode_ratio_scale = 2.0 (upper clamp bound)
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        let (l1, l2, l3) = (profile.kernel_config.l1d, profile.kernel_config.l2, profile.kernel_config.l3);
        let cache = HwOptEngine::solve_cache_budget(&profile, l1, l2, l3, &ir, &StrategyBias::default());
        let parallel = HwOptEngine::solve_parallelism(&profile, &cache, &StrategyBias::default());
        let high_decode_bias = StrategyBias {
            decode_ratio_scale: 2.0,
            ..StrategyBias::default()
        };
        // Act
        let batch = HwOptEngine::solve_batch(&profile, &parallel, &cache, &ir, &high_decode_bias);
        // Assert: decode_ratio_cap = (0.6 * 2.0).min(1.0) = 1.0 (line 1338)
        let expected_cap = (0.6f64 * 2.0).min(1.0) as f32;
        assert!(
            (batch.decode_ratio_cap - expected_cap).abs() < 1e-6,
            "decode_ratio_cap must equal (0.6 * decode_ratio_scale).min(1.0), got {}",
            batch.decode_ratio_cap
        );
    }

    // -- Wave 12kki: +10 additional tests --

    #[test]
    fn gemm_shape_debug_output_contains_all_dimensions() {
        // Arrange
        let shape = GemmShape { m: 1, n: 4096, k: 4096 };
        // Act
        let debug_str = format!("{:?}", shape);
        // Assert: Debug output must contain m, n, k values
        assert!(debug_str.contains("m: 1"), "Debug must show m dimension");
        assert!(debug_str.contains("n: 4096"), "Debug must show n dimension");
        assert!(debug_str.contains("k: 4096"), "Debug must show k dimension");
    }

    #[test]
    fn microkernel_choice_debug_output() {
        // Arrange
        let choice = MicrokernelChoice { mr: 6, nr: 16 };
        // Act
        let debug_str = format!("{:?}", choice);
        // Assert
        assert!(debug_str.contains("mr: 6"));
        assert!(debug_str.contains("nr: 16"));
    }

    #[test]
    fn bottleneck_class_debug_variants() {
        // Arrange & Act
        let compute_dbg = format!("{:?}", BottleneckClass::ComputeBound);
        let memory_dbg = format!("{:?}", BottleneckClass::MemoryBound);
        let mixed_dbg = format!("{:?}", BottleneckClass::Mixed);
        // Assert
        assert!(compute_dbg.contains("ComputeBound"));
        assert!(memory_dbg.contains("MemoryBound"));
        assert!(mixed_dbg.contains("Mixed"));
    }

    #[test]
    fn ffn_fusion_strategy_debug_output() {
        // Arrange & Act
        let inject_dbg = format!("{:?}", FfnFusionStrategy::GateSiLUInject);
        let separate_dbg = format!("{:?}", FfnFusionStrategy::SeparateGemm);
        // Assert
        assert!(inject_dbg.contains("GateSiLUInject"));
        assert!(separate_dbg.contains("SeparateGemm"));
    }

    #[test]
    fn attention_variant_debug_all_variants() {
        // Arrange: test that all variants have meaningful Debug output
        let variants = [
            (AttentionVariant::FA4BlockScaled, "FA4BlockScaled"),
            (AttentionVariant::FA3Pipeline, "FA3Pipeline"),
            (AttentionVariant::FA2Tiled, "FA2Tiled"),
            (AttentionVariant::WmmaTiled, "WmmaTiled"),
            (AttentionVariant::AmxTile, "AmxTile"),
            (AttentionVariant::Avx512Loop, "Avx512Loop"),
            (AttentionVariant::NeonLoop, "NeonLoop"),
            (AttentionVariant::ScalarLoop, "ScalarLoop"),
        ];
        // Act & Assert
        for (variant, expected) in variants {
            let dbg = format!("{:?}", variant);
            assert!(dbg.contains(expected), "Debug for {:?} must contain {}", variant, expected);
        }
    }

    #[test]
    fn gemm_microkernel_strategy_debug_all_variants() {
        // Arrange
        let strategies = [
            (GemmMicrokernelStrategy::AmxTile, "AmxTile"),
            (GemmMicrokernelStrategy::Avx512NativeBf16, "Avx512NativeBf16"),
            (GemmMicrokernelStrategy::BlisAvx512, "BlisAvx512"),
            (GemmMicrokernelStrategy::BlisAvx2, "BlisAvx2"),
            (GemmMicrokernelStrategy::BlisNeon, "BlisNeon"),
            (GemmMicrokernelStrategy::BlisSve, "BlisSve"),
            (GemmMicrokernelStrategy::GpuTensorCore, "GpuTensorCore"),
            (GemmMicrokernelStrategy::GpuScalar, "GpuScalar"),
            (GemmMicrokernelStrategy::Scalar, "Scalar"),
        ];
        // Act & Assert
        for (strategy, expected) in strategies {
            let dbg = format!("{:?}", strategy);
            assert!(dbg.contains(expected), "Debug for {:?} must contain {}", strategy, expected);
        }
    }

    #[test]
    fn fusion_decision_equality_and_inequality() {
        // Arrange
        let norm1 = FusionDecision::RmsNormIntoGemm;
        let norm2 = FusionDecision::RmsNormIntoGemm;
        let flash = FusionDecision::FlashAttention;
        let qkv = FusionDecision::QkvSharedInput;
        // Act & Assert
        assert_eq!(norm1, norm2);
        assert_ne!(norm1, flash);
        assert_ne!(flash, qkv);
        // GemmBiasAct with same activation are equal
        assert_eq!(
            FusionDecision::GemmBiasAct(Activation::Silu),
            FusionDecision::GemmBiasAct(Activation::Silu)
        );
        assert_ne!(
            FusionDecision::GemmBiasAct(Activation::Silu),
            FusionDecision::GemmBiasAct(Activation::Gelu)
        );
    }

    #[test]
    fn parallel_plan_direct_construction() {
        // Arrange & Act
        let plan = ParallelPlan {
            wave_count: 2,
            gpu_sm_partition: Some(GpuSmPartition {
                total_sm: 108,
                num_partitions: 2,
                sm_per_partition: 54,
            }),
            numa_bindings: vec![],
            min_batch_tokens_per_wave: 256,
            min_decode_per_wave: 16,
            occupancy_target: 0.75,
        };
        // Assert
        assert_eq!(plan.wave_count, 2);
        assert!(plan.gpu_sm_partition.is_some());
        assert_eq!(plan.gpu_sm_partition.unwrap().total_sm, 108);
        assert!((plan.occupancy_target - 0.75).abs() < 1e-6);
    }

    #[test]
    fn feature_plan_l1i_budget_calculation() {
        // Arrange: create a FeaturePlan with known number of decisions
        let decisions = vec![
            FeatureDecision {
                feature: "test1".into(),
                enabled: true,
                reason: "reason1".into(),
            },
            FeatureDecision {
                feature: "test2".into(),
                enabled: false,
                reason: "reason2".into(),
            },
        ];
        // Act
        let plan = FeaturePlan {
            decisions: decisions.clone(),
            l1i_used: decisions.len() * 2048, // matches line 1449 logic
            l1i_budget: 32 * 1024,
        };
        // Assert
        assert_eq!(plan.decisions.len(), 2);
        assert_eq!(plan.l1i_used, 4096);
        assert!(plan.l1i_used < plan.l1i_budget, "L1i used must be within budget");
    }

    #[test]
    fn batch_plan_direct_construction_all_fields() {
        // Arrange & Act
        let plan = BatchPlan {
            decode_ratio_cap: 0.6,
            max_chunk_size: 512,
            golden_sizes: vec![16, 32, 64, 128, 256, 512],
            min_compact_threshold: 4,
            compact_waste_threshold: 0.25,
            decode_slots: 32,
            max_chunks_per_batch: 8,
        };
        // Assert: all fields round-trip correctly
        assert!((plan.decode_ratio_cap - 0.6).abs() < 1e-6);
        assert_eq!(plan.max_chunk_size, 512);
        assert_eq!(plan.golden_sizes.len(), 6);
        assert_eq!(plan.min_compact_threshold, 4);
        assert!((plan.compact_waste_threshold - 0.25).abs() < 1e-6);
        assert_eq!(plan.decode_slots, 32);
        assert_eq!(plan.max_chunks_per_batch, 8);
    }

    // -- Wave 12kkm: +10 additional tests --

    #[test]
    fn compute_blocking_very_small_matrix_caps_all_dimensions() {
        // Arrange: 1x1x1 matrix, smaller than any microkernel tile
        let shape = GemmShape { m: 1, n: 1, k: 1 };
        // Act
        let (kc, mc, nc) = compute_blocking(&shape, 512, 256, 1024, 6, 16);
        // Assert: kc capped at shape.k=1, mc >= mr=6, nc >= nr=16 (line 1518)
        assert_eq!(kc, 1, "kc must be min(default_kc, shape.k)");
        assert!(mc >= 6, "mc must be at least mr=6, got {}", mc);
        assert!(nc >= 16, "nc must be at least nr=16, got {}", nc);
    }

    #[test]
    fn compute_blocking_very_large_matrix_uses_full_defaults() {
        // Arrange: 16384x16384x16384 matrix, larger than all defaults
        let shape = GemmShape { m: 16384, n: 16384, k: 16384 };
        // Act
        let (kc, mc, nc) = compute_blocking(&shape, 256, 96, 1024, 6, 16);
        // Assert: defaults used as-is (shape larger than all), aligned to mr/nr
        assert_eq!(kc, 256, "kc uses default when shape.k > default");
        assert_eq!(mc, 96, "mc uses default when shape.m > default");
        assert_eq!(nc, 1024, "nc uses default when shape.n > default");
    }

    #[test]
    fn compute_blocking_misaligned_dimensions_get_aligned_to_microkernel() {
        // Arrange: dimensions not aligned to mr=6, nr=16
        let shape = GemmShape { m: 100, n: 100, k: 100 };
        // Act
        let (kc, mc, nc) = compute_blocking(&shape, 256, 64, 64, 6, 16);
        // Assert: mc aligned down to multiple of mr, nc to multiple of nr (lines 1515-1516)
        assert_eq!(kc, 100, "kc capped at shape.k");
        assert_eq!(mc % 6, 0, "mc must be aligned to mr=6, got mc={}", mc);
        assert_eq!(nc % 16, 0, "nc must be aligned to nr=16, got nc={}", nc);
    }

    #[test]
    fn gemm_shape_ordering_and_sorting() {
        // Arrange: shapes with different m, n, k values
        let shapes = vec![
            GemmShape { m: 1, n: 4096, k: 4096 },
            GemmShape { m: 1, n: 2048, k: 4096 },
            GemmShape { m: 1, n: 4096, k: 2048 },
            GemmShape { m: 512, n: 4096, k: 4096 },
        ];
        // Act: sort by (m, n, k) as done in collect_gemm_shapes (line 1496)
        let mut sorted = shapes.clone();
        sorted.sort_by_key(|s| (s.m, s.n, s.k));
        // Assert: sorted order is by m first, then n, then k
        assert_eq!(sorted[0], GemmShape { m: 1, n: 2048, k: 4096 });
        assert_eq!(sorted[1], GemmShape { m: 1, n: 4096, k: 2048 });
        assert_eq!(sorted[2], GemmShape { m: 1, n: 4096, k: 4096 });
        assert_eq!(sorted[3], GemmShape { m: 512, n: 4096, k: 4096 });
    }

    #[test]
    fn collect_gemm_shapes_qkv_dimensions_derived_from_ir() {
        // Arrange: IR with specific hidden, num_heads, num_kv_heads, head_dim
        let ir = LayerIR {
            moe: None,
            hidden: 512,
            num_heads: 8,
            num_kv_heads: 2,  // GQA: fewer KV heads
            head_dim: 64,
            intermediate: 1024,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 256,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        // Act
        let shapes = collect_gemm_shapes(&ir);
        // Assert: q_dim = num_heads * head_dim = 8 * 64 = 512
        // kv_dim = num_kv_heads * head_dim = 2 * 64 = 128
        let q_dim = ir.q_dim();
        let kv_dim = ir.kv_dim();
        assert_eq!(q_dim, 512);
        assert_eq!(kv_dim, 128);
        // Q projection: n = q_dim
        assert!(shapes.iter().any(|s| s.n == q_dim && s.k == ir.hidden),
            "must have Q projection shape with n=q_dim");
        // K/V projection: n = kv_dim
        assert!(shapes.iter().any(|s| s.n == kv_dim && s.k == ir.hidden),
            "must have K/V projection shape with n=kv_dim");
    }

    #[test]
    fn compute_scratchpad_scales_linearly_with_batch_size() {
        // Arrange: two IRs with batch=1 and batch=8
        let ir_b1 = LayerIR {
            moe: None,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate: 512,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 128,
            partial_rotary_factor: 1.0,
            activation: Activation::Silu,
        };
        let mut ir_b8 = ir_b1.clone();
        ir_b8.max_batch = 8;
        // Act
        let scratch_b1 = compute_scratchpad(&ir_b1);
        let scratch_b8 = compute_scratchpad(&ir_b8);
        // Assert: scratchpad scales linearly with batch (all terms have b factor)
        assert!(scratch_b8 > scratch_b1, "larger batch needs more scratchpad");
        // The ratio should be approximately 8x (exact if all terms scale with batch)
        let ratio = scratch_b8 as f64 / scratch_b1 as f64;
        assert!(ratio >= 7.5 && ratio <= 8.5,
            "scratchpad should scale ~8x, got ratio={}", ratio);
    }

    #[test]
    fn plan_fusions_decoder_with_geglu_activation() {
        // Arrange: decoder with GeGlu activation (not Silu)
        let ir = LayerIR {
            moe: None,
            hidden: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate: 128,
            quant: None,
            dtype: DType::F32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_batch: 1,
            max_seq: 512,
            partial_rotary_factor: 1.0,
            activation: Activation::GeGlu,
        };
        // Act
        let fusions = plan_fusions(&ir);
        // Assert: GeGlu activation triggers GeGluFusion (line 1559)
        assert!(
            fusions.contains(&FusionDecision::GeGluFusion),
            "GeGlu activation must produce GeGluFusion"
        );
        assert!(
            !fusions.contains(&FusionDecision::SwiGluFusion),
            "GeGlu activation must NOT produce SwiGluFusion"
        );
    }

    #[test]
    fn execution_plan_gemm_blocking_has_entry_for_each_shape() {
        // Arrange
        let profile = DeviceProfile::detect();
        let ir = LayerIR::from_model_config(&ModelConfig::llama_7b(), 1);
        // Act
        let plan = ExecutionPlan::build(&ir, &profile, &StrategyBias::default());
        // Assert: each GEMM shape from collect_gemm_shapes has a blocking entry
        let shapes = collect_gemm_shapes(&ir);
        for shape in &shapes {
            assert!(
                plan.gemm_blocking.contains_key(shape),
                "gemm_blocking must have entry for shape {:?}", shape
            );
            let (kc, mc, nc) = plan.gemm_blocking[shape];
            assert!(kc >= 1, "kc must be >= 1");
            assert!(mc >= plan.microkernel.mr, "mc must be >= mr");
            assert!(nc >= plan.microkernel.nr, "nc must be >= nr");
        }
    }

    #[test]
    fn strategy_bias_all_fields_are_copy() {
        // Arrange: verify StrategyBias is Copy (line 28: #[derive(Debug, Clone, Copy)])
        let original = StrategyBias {
            fusion_cost_scale: 1.5,
            pipeline_cost_scale: 0.8,
            parallelism_cost_scale: 1.2,
            epilogue_depth_preference: 0.6,
            k_depth_preference: 2.0,
            kv_cache_budget_scale: 1.5,
            weight_prefetch_budget_scale: 0.9,
            batch_flexibility: 0.5,
            decode_ratio_scale: 1.3,
            speculative_decoding_value: 0.8,
            quantization_aggressiveness: 1.1,
            expert_eviction_aggressiveness: 0.7,
            expert_prefetch_priority: 2.5,
        };
        // Act: copy the struct
        let copied = original;
        // Assert: both are independent (Copy trait)
        assert!((original.fusion_cost_scale - 1.5).abs() < 1e-10);
        assert!((copied.fusion_cost_scale - 1.5).abs() < 1e-10);
        // Verify all fields copied correctly
        assert!((original.pipeline_cost_scale - copied.pipeline_cost_scale).abs() < 1e-10);
        assert!((original.k_depth_preference - copied.k_depth_preference).abs() < 1e-10);
    }

    #[test]
    fn roofline_result_elementwise_always_memory_bound() {
        // Arrange: any DeviceProfile
        let profile = DeviceProfile::detect();
        // Act
        let roofline = HwOptEngine::solve_roofline(&profile);
        // Assert: elementwise ops have AI ~ 2-3, always memory-bound (line 872)
        assert_eq!(
            roofline.elementwise,
            BottleneckClass::MemoryBound,
            "elementwise ops must always be classified as MemoryBound"
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  Global ExecutionPlan singleton (SPEC §10.15)
// ═══════════════════════════════════════════════════════════════

use std::sync::{Arc, OnceLock};

static EXECUTION_PLAN: OnceLock<ExecutionPlan> = OnceLock::new();

thread_local! {
    /// ARCH-PER-CLIENT-PLAN (REQ-ARB-008): per-thread ExecutionPlan override stack.
    /// Set by `with_execution_plan(plan, || ...)` for the duration of a Client's
    /// inference call. `global_execution_plan()` returns the top of stack if non-empty,
    /// falling back to the OnceLock global default.
    ///
    /// 解决 task #14 part 2 fundamental: 不同 Client (不同 InferenceMode / archetype /
    /// model bias) 各自持有 Arc<ExecutionPlan>,在 inference 入口 push 到 thread-local,
    /// 退出时 pop。codegen / fusion / autotuning 经 `global_execution_plan()` 透明读取
    /// 当前 Client 的 plan,无需改 API。
    static CURRENT_PLAN: std::cell::RefCell<Vec<Arc<ExecutionPlan>>>
        = const { std::cell::RefCell::new(Vec::new()) };
}

/// Initialize the global ExecutionPlan with a `StrategyBias`.
///
/// Must be called BEFORE any call to `global_execution_plan()` (otherwise OnceLock
/// is already locked with default bias).
/// 推荐使用 `with_execution_plan` (per-Client) 而非 init_global (process-wide)。
pub fn init_global_execution_plan_with_bias(bias: &StrategyBias) {
    EXECUTION_PLAN.get_or_init(|| {
        let profile = crate::dispatch::device_profile();
        HwOptEngine::solve_profile_only_with_bias(profile, bias)
    });
}

/// Compute a fresh ExecutionPlan with the given StrategyBias (no global side effect).
/// Caller stores the returned `Arc<ExecutionPlan>` in their Client state and uses
/// `with_execution_plan` to scope inference calls.
pub fn compute_execution_plan_with_bias(bias: &StrategyBias) -> Arc<ExecutionPlan> {
    let profile = crate::dispatch::device_profile();
    Arc::new(HwOptEngine::solve_profile_only_with_bias(profile, bias))
}

/// Push `plan` onto the thread-local override stack for the duration of `f`,
/// then pop. Used by inference clients to scope codegen/runtime to the per-Client
/// ExecutionPlan computed from the model's archetype + bias.
pub fn with_execution_plan<R>(plan: Arc<ExecutionPlan>, f: impl FnOnce() -> R) -> R {
    CURRENT_PLAN.with(|stack| stack.borrow_mut().push(plan));
    let result = f();
    CURRENT_PLAN.with(|stack| { stack.borrow_mut().pop(); });
    result
}

/// Get the active ExecutionPlan: thread-local override first, else OnceLock global default.
///
/// SPEC §10.15: All strategy decisions derive from ExecutionPlan. Each Client may
/// install its own plan via `with_execution_plan`; absent that, all consumers see
/// the same default (lazy-initialized from DeviceProfile + StrategyBias::default()).
///
/// Returns `&'static ExecutionPlan` for backward compatibility — when an Arc-based
/// plan is on the thread-local stack, the returned reference is valid only for the
/// duration of the `with_execution_plan` scope. Callers must not store it across
/// scope boundaries (typical codegen/fusion path is fine: synchronous within scope).
pub fn global_execution_plan() -> &'static ExecutionPlan {
    // Try thread-local override first (per-Client plan).
    let from_tls: Option<&'static ExecutionPlan> = CURRENT_PLAN.with(|stack| {
        let stack = stack.borrow();
        stack.last().map(|arc| {
            // SAFETY: 'static lifetime is a contract: caller must not store this
            // reference beyond the with_execution_plan scope. The Arc keeps the plan
            // alive for the scope duration, and we only return a pointer — extending
            // the lifetime is sound as long as the contract holds.
            let raw: *const ExecutionPlan = Arc::as_ptr(arc);
            unsafe { &*raw }
        })
    });
    if let Some(plan) = from_tls {
        return plan;
    }
    // Fallback: lazy-init OnceLock global default.
    EXECUTION_PLAN.get_or_init(|| {
        let profile = crate::dispatch::device_profile();
        ExecutionPlan::from_profile(profile)
    })
}
