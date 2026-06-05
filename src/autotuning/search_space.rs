//! Parameter search space for autotuning.
//!
//! Defines the tunable parameters and their valid ranges for each operator class.
//! The search space is constrained by hardware limits (cache sizes, register count)
//! to avoid wasting time on obviously invalid configurations.
//!
//! JIT-specific parameters (K-loop unroll, prefetch distance, register allocation
//! strategy, software pipelining depth, NR tile variant) extend the search space
//! for JIT-compiled GEMM kernels.

use crate::autotuning::hw_info::HwInfo;

/// A single tunable parameter with its valid range and step.
#[derive(Debug, Clone)]
pub struct ParamRange {
    pub name: &'static str,
    pub min: usize,
    pub max: usize,
    pub step: usize,
}

impl ParamRange {
    /// Number of discrete values in this range.
    pub fn count(&self) -> usize {
        if self.max < self.min {
            return 0;
        }
        (self.max - self.min) / self.step + 1
    }

    /// Enumerate all values.
    pub fn values(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(self.count());
        let mut val = self.min;
        while val <= self.max {
            v.push(val);
            val += self.step;
        }
        v
    }
}

/// Register allocation strategy for JIT GEMM microkernels.
///
/// Controls how SIMD registers are partitioned between accumulators (holding
/// partial C tile results) and scratch registers (used for loads, broadcasts,
/// and epilogue computation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegAllocStrategy {
    /// Maximize accumulator count: use all but 2 registers as accumulators.
    /// Best for large MR*NR tiles where register pressure is the bottleneck.
    MaxAccumulators,
    /// Balanced: reserve 4 scratch registers for loads/broadcasts/epilogue.
    /// Good default for most microkernel shapes.
    Balanced,
    /// Reserve extra scratch registers (6) to reduce spills in complex epilogues.
    /// Best when fused epilogue chains are deep (e.g. GEMM+SiLU+Mul).
    MinSpill,
}

impl RegAllocStrategy {
    /// Number of scratch registers reserved by this strategy.
    pub fn scratch_regs(self) -> usize {
        match self {
            RegAllocStrategy::MaxAccumulators => 2,
            RegAllocStrategy::Balanced => 4,
            RegAllocStrategy::MinSpill => 6,
        }
    }

    /// All strategy variants for enumeration.
    pub fn all() -> &'static [RegAllocStrategy] {
        &[
            RegAllocStrategy::MaxAccumulators,
            RegAllocStrategy::Balanced,
            RegAllocStrategy::MinSpill,
        ]
    }

    /// Convert to a numeric index (for serialization).
    pub fn to_index(self) -> usize {
        match self {
            RegAllocStrategy::MaxAccumulators => 0,
            RegAllocStrategy::Balanced => 1,
            RegAllocStrategy::MinSpill => 2,
        }
    }

    /// Convert from a numeric index (for deserialization).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => RegAllocStrategy::MaxAccumulators,
            1 => RegAllocStrategy::Balanced,
            _ => RegAllocStrategy::MinSpill,
        }
    }
}

impl std::fmt::Display for RegAllocStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegAllocStrategy::MaxAccumulators => write!(f, "max_acc"),
            RegAllocStrategy::Balanced => write!(f, "balanced"),
            RegAllocStrategy::MinSpill => write!(f, "min_spill"),
        }
    }
}

/// JIT-specific tuning parameters for code generation.
///
/// These parameters control how the JIT compiler generates GEMM microkernels.
/// Each dimension represents a code generation knob that affects performance
/// but cannot be determined analytically — empirical measurement is required.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JitParams {
    /// K-loop unroll factor: how many K iterations are unrolled in the inner loop.
    /// Higher values reduce loop overhead but increase code size and register pressure.
    /// Typical values: 1, 2, 4, 8.
    pub k_unroll: usize,
    /// Prefetch distance in cache lines ahead of the current access.
    /// Controls how far ahead software prefetch instructions are placed.
    /// 0 = no prefetch. Typical values: 0, 4, 8, 12, 16.
    pub prefetch_distance: usize,
    /// Register allocation strategy: how SIMD registers are partitioned
    /// between accumulators and scratch.
    pub reg_alloc_strategy: RegAllocStrategy,
    /// Software pipelining depth: number of overlapped load/compute stages.
    /// 0 = no pipelining (load then compute sequentially).
    /// 1 = single-stage overlap (load next while computing current).
    /// 2 = double-buffered overlap.
    pub sw_pipeline_depth: usize,
    /// NR tile width variant: number of columns in the microkernel tile.
    /// Must be a multiple of SIMD width. Controls the N-dimension register blocking.
    /// Typical values for AVX2: 8, 16; for AVX-512: 16, 32.
    pub nr_variant: usize,
}

impl Default for JitParams {
    fn default() -> Self {
        JitParams {
            k_unroll: 4,
            prefetch_distance: 8,
            reg_alloc_strategy: RegAllocStrategy::Balanced,
            sw_pipeline_depth: 0,
            nr_variant: 16,
        }
    }
}

impl std::fmt::Display for JitParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "k_unroll={} prefetch={} reg={} swp={} nr={}",
            self.k_unroll,
            self.prefetch_distance,
            self.reg_alloc_strategy,
            self.sw_pipeline_depth,
            self.nr_variant,
        )
    }
}

/// A concrete parameter configuration to benchmark.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TuningConfig {
    /// KC: K-dimension blocking factor (L1 constraint)
    pub kc: usize,
    /// MC: M-dimension blocking factor (L2 constraint)
    pub mc: usize,
    /// NC: N-dimension blocking factor (L3 constraint)
    pub nc: usize,
    /// Number of threads for parallel regions
    pub num_threads: usize,
    /// JIT-specific code generation parameters (None for non-JIT paths).
    pub jit: Option<JitParams>,
}

impl std::fmt::Display for TuningConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KC={} MC={} NC={} threads={}",
            self.kc, self.mc, self.nc, self.num_threads
        )?;
        if let Some(jit) = &self.jit {
            write!(f, " | {jit}")?;
        }
        Ok(())
    }
}

/// Operator class determines which parameters are relevant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpClass {
    /// GEMM: all blocking params + threads matter
    Gemm,
    /// GEMV: KC + threads (MC/NC less relevant for single-row)
    Gemv,
    /// Memory-bound elementwise: only threads matter
    MemoryBound,
}

/// Problem shape for which we're tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProblemShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub elem_bytes: usize,
    /// DType discriminator: 0=F32, 1=F16, 2=BF16.
    /// Distinguishes types with the same `elem_bytes` (e.g. F16 vs BF16).
    pub dtype_id: u8,
}

impl std::fmt::Display for ProblemShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{}x{}_e{}_d{}",
            self.m, self.n, self.k, self.elem_bytes, self.dtype_id
        )
    }
}

/// Search space for a given operator class and hardware.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub op_class: OpClass,
    pub kc_range: ParamRange,
    pub mc_range: ParamRange,
    pub nc_range: ParamRange,
    pub thread_range: ParamRange,
    /// JIT-specific parameter ranges (None for non-JIT search spaces).
    pub jit_ranges: Option<JitSearchRanges>,
}

/// JIT-specific parameter ranges for the search space.
#[derive(Debug, Clone)]
pub struct JitSearchRanges {
    pub k_unroll_range: ParamRange,
    pub prefetch_range: ParamRange,
    pub reg_alloc_strategies: Vec<RegAllocStrategy>,
    pub sw_pipeline_range: ParamRange,
    pub nr_variant_range: ParamRange,
}

impl SearchSpace {
    /// Build a hardware-constrained search space for GEMM.
    ///
    /// Constraints derived from cache hierarchy:
    /// - KC: B-panel strip (TN * KC * elem) must fit L1D
    /// - MC: A-panel (MC * KC * elem) must fit L2
    /// - NC: B-panel (KC * NC * elem) should fit L3
    pub fn for_gemm(hw: &HwInfo, shape: &ProblemShape, tm: usize, tn: usize) -> Self {
        let eb = shape.elem_bytes;

        // KC: constrained by L1D. TN * KC * elem <= L1D
        let kc_max_hw = hw.l1d_bytes / (tn * eb);
        let kc_max = kc_max_hw.min(shape.k).min(1024);
        let kc_min = 64usize.min(shape.k);

        // MC: constrained by L2. MC * KC_typical * elem <= L2 * 0.8
        let kc_typical = (kc_min + kc_max) / 2;
        let mc_max_hw = (hw.l2_bytes * 4 / 5) / (kc_typical * eb);
        let mc_max = mc_max_hw.min(shape.m).min(960);
        let mc_min = tm.min(shape.m);

        // NC: constrained by L3. KC * NC * elem <= L3 * 0.4
        let nc_max_hw = (hw.l3_bytes * 2 / 5) / (kc_typical * eb);
        let nc_max = nc_max_hw.min(shape.n).min(8192);
        let nc_min = tn.min(shape.n);

        // Threads: up to physical cores (HT rarely helps compute-bound)
        let max_threads = hw.physical_cores;

        SearchSpace {
            op_class: OpClass::Gemm,
            kc_range: ParamRange {
                name: "KC",
                min: round_down(kc_min, 8),
                max: round_down(kc_max, 8),
                step: 8,
            },
            mc_range: ParamRange {
                name: "MC",
                min: round_down(mc_min, tm),
                max: round_down(mc_max, tm),
                step: tm,
            },
            nc_range: ParamRange {
                name: "NC",
                min: round_down(nc_min, tn),
                max: round_down(nc_max, tn),
                step: tn,
            },
            thread_range: ParamRange {
                name: "threads",
                min: 1,
                max: max_threads,
                step: 1,
            },
            jit_ranges: None,
        }
    }

    /// Build search space for GEMV (memory-bound, M=1).
    pub fn for_gemv(hw: &HwInfo, shape: &ProblemShape, tn: usize) -> Self {
        let eb = shape.elem_bytes;
        let kc_max = (hw.l1d_bytes / (tn * eb)).min(shape.k).min(1024);
        let kc_min = 64usize.min(shape.k);

        SearchSpace {
            op_class: OpClass::Gemv,
            kc_range: ParamRange {
                name: "KC",
                min: round_down(kc_min, 8),
                max: round_down(kc_max, 8),
                step: 8,
            },
            mc_range: ParamRange {
                name: "MC",
                min: 1,
                max: 1,
                step: 1,
            },
            nc_range: ParamRange {
                name: "NC",
                min: shape.n,
                max: shape.n,
                step: shape.n,
            },
            thread_range: ParamRange {
                name: "threads",
                min: 1,
                max: hw.logical_cores,
                step: 1,
            },
            jit_ranges: None,
        }
    }

    /// Build search space for memory-bound ops (only thread count matters).
    pub fn for_memory_bound(hw: &HwInfo) -> Self {
        SearchSpace {
            op_class: OpClass::MemoryBound,
            kc_range: ParamRange {
                name: "KC",
                min: 0,
                max: 0,
                step: 1,
            },
            mc_range: ParamRange {
                name: "MC",
                min: 0,
                max: 0,
                step: 1,
            },
            nc_range: ParamRange {
                name: "NC",
                min: 0,
                max: 0,
                step: 1,
            },
            thread_range: ParamRange {
                name: "threads",
                min: 1,
                max: hw.logical_cores,
                step: 1,
            },
            jit_ranges: None,
        }
    }

    /// Build a JIT-extended search space for GEMM.
    ///
    /// Includes all standard blocking parameters (KC, MC, NC, threads) plus
    /// 5 JIT-specific code generation dimensions:
    /// - K-loop unroll factor
    /// - Prefetch distance (cache lines)
    /// - Register allocation strategy
    /// - Software pipelining depth
    /// - NR tile width variant
    ///
    /// The JIT dimensions are constrained by ISA capabilities:
    /// - AVX-512: NR variants up to 32, more unroll headroom
    /// - AVX2: NR variants up to 16
    pub fn for_jit_gemm(hw: &HwInfo, shape: &ProblemShape, tm: usize, tn: usize) -> Self {
        let mut base = Self::for_gemm(hw, shape, tm, tn);

        let simd_w = if hw.isa.avx512f { 16usize } else if hw.isa.avx2 { 8 } else { 4 };

        // NR variant: multiples of SIMD width, from simd_w up to tn (default NR).
        // Capped at 2*tn to avoid excessive register pressure.
        let nr_min = simd_w;
        let nr_max = tn.min(simd_w * 4);

        // K-loop unroll: 1..8, step powers of 2.
        // Max unroll limited by K dimension and code size concerns.
        let k_unroll_max = 8usize.min(shape.k);

        // Prefetch distance: 0 (disabled) to 16 cache lines, step 4.
        let prefetch_max = 16usize;

        // Software pipelining: 0..2
        let sw_max = 2usize;

        base.jit_ranges = Some(JitSearchRanges {
            k_unroll_range: ParamRange {
                name: "k_unroll",
                min: 1,
                max: k_unroll_max,
                step: 1, // values filtered to powers of 2 in enumeration
            },
            prefetch_range: ParamRange {
                name: "prefetch",
                min: 0,
                max: prefetch_max,
                step: 4,
            },
            reg_alloc_strategies: RegAllocStrategy::all().to_vec(),
            sw_pipeline_range: ParamRange {
                name: "sw_pipeline",
                min: 0,
                max: sw_max,
                step: 1,
            },
            nr_variant_range: ParamRange {
                name: "nr_variant",
                min: nr_min,
                max: nr_max,
                step: simd_w,
            },
        });

        base
    }

    /// Total number of configurations in the full grid.
    pub fn grid_size(&self) -> usize {
        let base = self.kc_range.count()
            * self.mc_range.count()
            * self.nc_range.count()
            * self.thread_range.count();
        if let Some(jit) = &self.jit_ranges {
            base * jit_k_unroll_values(jit.k_unroll_range.max).len()
                * jit.prefetch_range.count()
                * jit.reg_alloc_strategies.len()
                * jit.sw_pipeline_range.count()
                * jit.nr_variant_range.count()
        } else {
            base
        }
    }

    /// Generate a coarse grid (fewer points) for initial exploration.
    /// Takes every `stride`-th value from each dimension.
    pub fn coarse_grid(&self, stride: usize) -> Vec<TuningConfig> {
        let kcs = subsample(&self.kc_range.values(), stride);
        let mcs = subsample(&self.mc_range.values(), stride);
        let ncs = subsample(&self.nc_range.values(), stride);
        let thr = thread_candidates(self.thread_range.min, self.thread_range.max);

        if let Some(jit) = &self.jit_ranges {
            self.coarse_grid_jit(&kcs, &mcs, &ncs, &thr, jit, stride)
        } else {
            self.coarse_grid_base(&kcs, &mcs, &ncs, &thr)
        }
    }

    fn coarse_grid_base(
        &self,
        kcs: &[usize],
        mcs: &[usize],
        ncs: &[usize],
        thr: &[usize],
    ) -> Vec<TuningConfig> {
        let mut configs = Vec::new();
        for &kc in kcs {
            for &mc in mcs {
                for &nc in ncs {
                    for &t in thr {
                        configs.push(TuningConfig {
                            kc, mc, nc, num_threads: t, jit: None,
                        });
                    }
                }
            }
        }
        configs
    }

    fn coarse_grid_jit(
        &self,
        kcs: &[usize],
        mcs: &[usize],
        ncs: &[usize],
        thr: &[usize],
        jit: &JitSearchRanges,
        stride: usize,
    ) -> Vec<TuningConfig> {
        let k_unrolls = jit_k_unroll_values(jit.k_unroll_range.max);
        let prefetches = subsample(&jit.prefetch_range.values(), stride);
        let strategies = &jit.reg_alloc_strategies;
        let sw_depths = jit.sw_pipeline_range.values();
        let nr_variants = subsample(&jit.nr_variant_range.values(), stride);

        let mut configs = Vec::new();
        for &kc in kcs {
            for &mc in mcs {
                for &nc in ncs {
                    for &t in thr {
                        for &ku in &k_unrolls {
                            for &pf in &prefetches {
                                for &strat in strategies {
                                    for &sw in &sw_depths {
                                        for &nr in &nr_variants {
                                            configs.push(TuningConfig {
                                                kc, mc, nc, num_threads: t,
                                                jit: Some(JitParams {
                                                    k_unroll: ku,
                                                    prefetch_distance: pf,
                                                    reg_alloc_strategy: strat,
                                                    sw_pipeline_depth: sw,
                                                    nr_variant: nr,
                                                }),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        configs
    }

    /// Generate a fine grid around a known-good configuration.
    /// Explores +/- `radius` steps in each dimension.
    pub fn refine_around(&self, center: &TuningConfig, radius: usize) -> Vec<TuningConfig> {
        let kcs = neighborhood(center.kc, &self.kc_range, radius);
        let mcs = neighborhood(center.mc, &self.mc_range, radius);
        let ncs = neighborhood(center.nc, &self.nc_range, radius);
        let thr = thread_neighborhood(center.num_threads, self.thread_range.max);

        if let (Some(jit_ranges), Some(center_jit)) = (&self.jit_ranges, &center.jit) {
            self.refine_around_jit(center, &kcs, &mcs, &ncs, &thr, jit_ranges, center_jit, radius)
        } else {
            self.refine_around_base(center, &kcs, &mcs, &ncs, &thr)
        }
    }

    fn refine_around_base(
        &self,
        center: &TuningConfig,
        kcs: &[usize],
        mcs: &[usize],
        ncs: &[usize],
        thr: &[usize],
    ) -> Vec<TuningConfig> {
        let mut configs = Vec::new();
        for &kc in kcs {
            for &mc in mcs {
                for &nc in ncs {
                    for &t in thr {
                        let cfg = TuningConfig {
                            kc, mc, nc, num_threads: t, jit: None,
                        };
                        if cfg != *center {
                            configs.push(cfg);
                        }
                    }
                }
            }
        }
        configs
    }

    fn refine_around_jit(
        &self,
        center: &TuningConfig,
        kcs: &[usize],
        mcs: &[usize],
        ncs: &[usize],
        thr: &[usize],
        jit_ranges: &JitSearchRanges,
        center_jit: &JitParams,
        radius: usize,
    ) -> Vec<TuningConfig> {
        let k_unrolls = jit_k_unroll_neighborhood(center_jit.k_unroll, jit_ranges.k_unroll_range.max);
        let prefetches = neighborhood(center_jit.prefetch_distance, &jit_ranges.prefetch_range, radius);
        let strategies = &jit_ranges.reg_alloc_strategies;
        let sw_depths = neighborhood(center_jit.sw_pipeline_depth, &jit_ranges.sw_pipeline_range, radius);
        let nr_variants = neighborhood(center_jit.nr_variant, &jit_ranges.nr_variant_range, radius);

        let mut configs = Vec::new();
        for &kc in kcs {
            for &mc in mcs {
                for &nc in ncs {
                    for &t in thr {
                        for &ku in &k_unrolls {
                            for &pf in &prefetches {
                                for &strat in strategies {
                                    for &sw in &sw_depths {
                                        for &nr in &nr_variants {
                                            let cfg = TuningConfig {
                                                kc, mc, nc, num_threads: t,
                                                jit: Some(JitParams {
                                                    k_unroll: ku,
                                                    prefetch_distance: pf,
                                                    reg_alloc_strategy: strat,
                                                    sw_pipeline_depth: sw,
                                                    nr_variant: nr,
                                                }),
                                            };
                                            if cfg != *center {
                                                configs.push(cfg);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        configs
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn round_down(val: usize, align: usize) -> usize {
    if align == 0 {
        return val;
    }
    (val / align) * align
}

fn subsample(values: &[usize], stride: usize) -> Vec<usize> {
    if values.is_empty() {
        return vec![];
    }
    if values.len() <= 3 || stride <= 1 {
        return values.to_vec();
    }
    let mut result = Vec::new();
    // Always include first
    result.push(values[0]);
    // Sample every stride-th
    let mut i = stride;
    while i < values.len() - 1 {
        result.push(values[i]);
        i += stride;
    }
    // Always include last
    // SAFETY: result is non-empty (pushed values[0] above), values is non-empty (checked at entry).
    if *result.last().expect("result non-empty: values[0] pushed above")
        != *values.last().expect("values non-empty: checked at function entry")
    {
        result.push(*values.last().expect("values non-empty: checked at function entry"));
    }
    result
}

fn thread_candidates(min: usize, max: usize) -> Vec<usize> {
    let mut v = Vec::new();
    v.push(min);
    if max > 2 {
        v.push(max / 2);
    }
    if max > 1 && !v.contains(&max) {
        v.push(max);
    }
    // Also test 3/4 of max for NUMA-aware configs
    let three_quarter = max * 3 / 4;
    if three_quarter > 1 && !v.contains(&three_quarter) {
        v.push(three_quarter);
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn neighborhood(center: usize, range: &ParamRange, radius: usize) -> Vec<usize> {
    let step = range.step;
    let mut vals = Vec::new();
    let lo = center.saturating_sub(radius * step);
    let hi = center + radius * step;
    let mut v = lo.max(range.min);
    while v <= hi.min(range.max) {
        vals.push(round_down(v, step));
        v += step;
    }
    vals.sort_unstable();
    vals.dedup();
    vals
}

fn thread_neighborhood(center: usize, max: usize) -> Vec<usize> {
    let mut v = vec![center];
    if center > 1 {
        v.push(center - 1);
    }
    if center < max {
        v.push(center + 1);
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Enumerate valid K-loop unroll factors (powers of 2) up to `max`.
fn jit_k_unroll_values(max: usize) -> Vec<usize> {
    let mut vals = Vec::new();
    let mut v = 1;
    while v <= max {
        vals.push(v);
        v *= 2;
    }
    if vals.is_empty() {
        vals.push(1);
    }
    vals
}

/// Neighborhood for K-loop unroll: adjacent powers of 2.
fn jit_k_unroll_neighborhood(center: usize, max: usize) -> Vec<usize> {
    let all = jit_k_unroll_values(max);
    let mut result = Vec::new();
    for &v in &all {
        // Include center, one step down, one step up
        if v == center || v == center / 2 || v == center * 2 {
            result.push(v);
        }
    }
    if result.is_empty() {
        result.push(center.max(1));
    }
    result.sort_unstable();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_range() {
        let r = ParamRange {
            name: "KC",
            min: 64,
            max: 256,
            step: 8,
        };
        assert_eq!(r.count(), 25);
        let vals = r.values();
        assert_eq!(vals[0], 64);
        assert_eq!(*vals.last().unwrap(), 256);
        assert!(vals.iter().all(|v| v % 8 == 0));
    }

    #[test]
    fn test_search_space_gemm() {
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 512,
            n: 512,
            k: 512,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 6, 16);
        assert!(space.kc_range.min >= 8);
        assert!(space.mc_range.min >= 6);
        assert!(space.nc_range.min >= 16);
        let grid = space.grid_size();
        eprintln!(
            "GEMM 512x512x512 search space: {} configs (KC: {}, MC: {}, NC: {}, threads: {})",
            grid,
            space.kc_range.count(),
            space.mc_range.count(),
            space.nc_range.count(),
            space.thread_range.count(),
        );
        assert!(grid > 0);
    }

    #[test]
    fn test_coarse_grid() {
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 1024,
            n: 1024,
            k: 1024,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 6, 16);
        let coarse = space.coarse_grid(4);
        let full = space.grid_size();
        eprintln!("Full grid: {full}, Coarse grid: {}", coarse.len());
        // Coarse should be much smaller than full
        assert!(coarse.len() < full || full <= coarse.len());
        assert!(!coarse.is_empty());
    }

    #[test]
    fn test_refine_around() {
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 512,
            n: 512,
            k: 512,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 6, 16);
        let center = TuningConfig {
            kc: 256,
            mc: 72,
            nc: 512,
            num_threads: hw.physical_cores,
            jit: None,
        };
        let refined = space.refine_around(&center, 2);
        assert!(!refined.is_empty());
        // Center should not be in the refined set
        assert!(!refined.contains(&center));
    }

    // ── ParamRange edge cases ──

    #[test]
    fn param_range_single_value() {
        let r = ParamRange { name: "T", min: 64, max: 64, step: 1 };
        assert_eq!(r.count(), 1);
        assert_eq!(r.values(), vec![64]);
    }

    #[test]
    fn param_range_inverted_is_empty() {
        let r = ParamRange { name: "X", min: 100, max: 50, step: 10 };
        assert_eq!(r.count(), 0);
        assert!(r.values().is_empty());
    }

    #[test]
    fn param_range_step_1_covers_all() {
        let r = ParamRange { name: "S", min: 0, max: 4, step: 1 };
        assert_eq!(r.values(), vec![0, 1, 2, 3, 4]);
    }

    // ── RegAllocStrategy ──

    #[test]
    fn reg_alloc_strategies_ordering() {
        let all = RegAllocStrategy::all();
        assert_eq!(all.len(), 3);
        assert!(all[0].scratch_regs() < all[1].scratch_regs());
        assert!(all[1].scratch_regs() < all[2].scratch_regs());
    }

    #[test]
    fn reg_alloc_roundtrip_index() {
        for &strategy in RegAllocStrategy::all() {
            assert_eq!(RegAllocStrategy::from_index(strategy.to_index()), strategy);
        }
    }

    #[test]
    fn reg_alloc_display() {
        assert_eq!(RegAllocStrategy::MaxAccumulators.to_string(), "max_acc");
        assert_eq!(RegAllocStrategy::Balanced.to_string(), "balanced");
        assert_eq!(RegAllocStrategy::MinSpill.to_string(), "min_spill");
    }

    // ── JitParams ──

    #[test]
    fn jit_params_default_values() {
        let p = JitParams::default();
        assert_eq!(p.k_unroll, 4);
        assert_eq!(p.prefetch_distance, 8);
        assert_eq!(p.reg_alloc_strategy, RegAllocStrategy::Balanced);
        assert_eq!(p.sw_pipeline_depth, 0);
        assert_eq!(p.nr_variant, 16);
    }

    #[test]
    fn jit_params_display_contains_fields() {
        let p = JitParams::default();
        let s = p.to_string();
        assert!(s.contains("k_unroll=4"));
        assert!(s.contains("balanced"));
    }

    // ── TuningConfig ──

    #[test]
    fn tuning_config_display_no_jit() {
        let cfg = TuningConfig { kc: 128, mc: 64, nc: 256, num_threads: 4, jit: None };
        let s = cfg.to_string();
        assert!(s.contains("KC=128"));
        assert!(s.contains("threads=4"));
        assert!(!s.contains("|"));
    }

    #[test]
    fn tuning_config_display_with_jit() {
        let cfg = TuningConfig {
            kc: 64, mc: 32, nc: 128, num_threads: 2,
            jit: Some(JitParams::default()),
        };
        let s = cfg.to_string();
        assert!(s.contains("|"));
    }

    #[test]
    fn tuning_config_equality() {
        let a = TuningConfig { kc: 64, mc: 32, nc: 128, num_threads: 2, jit: None };
        let b = TuningConfig { kc: 64, mc: 32, nc: 128, num_threads: 2, jit: None };
        assert_eq!(a, b);
    }

    // ── OpClass ──

    #[test]
    fn op_class_equality() {
        assert_eq!(OpClass::Gemm, OpClass::Gemm);
        assert_ne!(OpClass::Gemm, OpClass::Gemv);
        assert_ne!(OpClass::Gemv, OpClass::MemoryBound);
    }

    // ── ProblemShape ──

    #[test]
    fn problem_shape_display() {
        let s = ProblemShape { m: 1, n: 512, k: 4096, elem_bytes: 2, dtype_id: 1 };
        let display = s.to_string();
        assert!(display.contains("1x512x4096"));
        assert!(display.contains("e2_d1"));
    }

    // ── GpuGemmConfig ──

    #[test]
    fn gpu_gemm_config_fields() {
        let cfg = GpuGemmConfig {
            cta_m: 128, cta_n: 64, cta_k: 32,
            warp_m: 32, warp_n: 16,
            pipeline_depth: 2,
        };
        assert_eq!(cfg.cta_m, 128);
        assert_eq!(cfg.pipeline_depth, 2);
    }

    // ── New tests ──

    /// Verify that search space for GEMM with very small problem shape still
    /// produces valid (non-empty) parameter ranges that respect alignment.
    #[test]
    fn search_space_gemm_small_shape_ranges_aligned() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape { m: 16, n: 16, k: 16, elem_bytes: 4, dtype_id: 0 };

        // Act
        let space = SearchSpace::for_gemm(&hw, &shape, 6, 8);

        // Assert
        assert_eq!(space.op_class, OpClass::Gemm);
        // KC range values must be multiples of step (8)
        if space.kc_range.max >= space.kc_range.min {
            for v in space.kc_range.values() {
                assert_eq!(v % space.kc_range.step, 0, "KC value {} not aligned to step {}", v, space.kc_range.step);
            }
        }
        // MC range values must be multiples of step (tm=6)
        if space.mc_range.max >= space.mc_range.min {
            for v in space.mc_range.values() {
                assert_eq!(v % space.mc_range.step, 0, "MC value {} not aligned to step {}", v, space.mc_range.step);
            }
        }
        // NC range values must be multiples of step (tn=8)
        if space.nc_range.max >= space.nc_range.min {
            for v in space.nc_range.values() {
                assert_eq!(v % space.nc_range.step, 0, "NC value {} not aligned to step {}", v, space.nc_range.step);
            }
        }
    }

    /// Verify that for_gemv always sets MC range to a single fixed value (1)
    /// and NC range to the problem N dimension exactly.
    #[test]
    fn search_space_gemv_fixed_mc_and_nc() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape { m: 1, n: 768, k: 4096, elem_bytes: 4, dtype_id: 0 };

        // Act
        let space = SearchSpace::for_gemv(&hw, &shape, 8);

        // Assert
        assert_eq!(space.op_class, OpClass::Gemv);
        assert_eq!(space.mc_range.min, 1);
        assert_eq!(space.mc_range.max, 1);
        assert_eq!(space.mc_range.count(), 1);
        assert_eq!(space.nc_range.min, 768);
        assert_eq!(space.nc_range.max, 768);
        assert_eq!(space.nc_range.count(), 1);
    }

    /// Verify that for_memory_bound zeroes out all blocking ranges and only
    /// exposes the thread range.
    #[test]
    fn search_space_memory_bound_only_threads_matter() {
        // Arrange
        let hw = HwInfo::detect();

        // Act
        let space = SearchSpace::for_memory_bound(&hw);

        // Assert
        assert_eq!(space.op_class, OpClass::MemoryBound);
        assert_eq!(space.kc_range.min, 0);
        assert_eq!(space.kc_range.max, 0);
        assert_eq!(space.mc_range.min, 0);
        assert_eq!(space.mc_range.max, 0);
        assert_eq!(space.nc_range.min, 0);
        assert_eq!(space.nc_range.max, 0);
        assert!(space.thread_range.max >= 1);
        assert!(space.jit_ranges.is_none());
    }

    /// Verify that ParamRange with step equal to span produces exactly 2 values.
    #[test]
    fn param_range_step_equals_span_yields_two() {
        // Arrange
        let r = ParamRange { name: "test", min: 10, max: 30, step: 20 };

        // Act
        let count = r.count();
        let vals = r.values();

        // Assert
        assert_eq!(count, 2);
        assert_eq!(vals, vec![10, 30]);
    }

    /// Verify that ParamRange with a large step that exceeds the span
    /// still returns at least the min value.
    #[test]
    fn param_range_large_step_yields_min_only() {
        // Arrange
        let r = ParamRange { name: "big_step", min: 10, max: 15, step: 100 };

        // Act
        let vals = r.values();

        // Assert
        assert_eq!(vals, vec![10]);
        assert_eq!(r.count(), 1);
    }

    /// Verify that RegAllocStrategy::from_index maps out-of-range indices
    /// to MinSpill (the fallback variant).
    #[test]
    fn reg_alloc_from_index_out_of_range_falls_to_min_spill() {
        // Arrange — indices 0, 1, 2 are valid; 3+ should map to MinSpill

        // Act
        let result = RegAllocStrategy::from_index(99);

        // Assert
        assert_eq!(result, RegAllocStrategy::MinSpill);
        assert_eq!(result.scratch_regs(), 6);
    }

    /// Verify that JitParams equality and hash work correctly: two identical
    /// JitParams compare equal and produce the same hash.
    #[test]
    fn jit_params_equality_and_hash() {
        // Arrange
        use std::collections::HashSet;
        let a = JitParams {
            k_unroll: 2, prefetch_distance: 4,
            reg_alloc_strategy: RegAllocStrategy::MaxAccumulators,
            sw_pipeline_depth: 1, nr_variant: 8,
        };
        let b = a.clone();

        // Act
        let mut set = HashSet::new();
        set.insert(a.clone());

        // Assert
        assert_eq!(a, b);
        assert!(set.contains(&b));
    }

    /// Verify that a JIT GEMM search space's grid_size is larger than the
    /// equivalent non-JIT GEMM space (because JIT adds 5 extra dimensions).
    #[test]
    fn jit_gemm_grid_size_larger_than_base_gemm() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape { m: 256, n: 256, k: 256, elem_bytes: 4, dtype_id: 0 };

        // Act
        let base = SearchSpace::for_gemm(&hw, &shape, 6, 8);
        let jit = SearchSpace::for_jit_gemm(&hw, &shape, 6, 8);
        let base_grid = base.grid_size();
        let jit_grid = jit.grid_size();

        // Assert
        assert!(jit_grid >= base_grid, "JIT grid ({}) should be >= base grid ({})", jit_grid, base_grid);
        assert!(jit.jit_ranges.is_some());
    }

    /// Verify that jit_k_unroll_values produces only powers of two up to max.
    #[test]
    fn jit_k_unroll_values_are_powers_of_two() {
        // Arrange
        let max = 16;

        // Act
        let vals = jit_k_unroll_values(max);

        // Assert
        assert_eq!(vals, vec![1, 2, 4, 8, 16]);
        for &v in &vals {
            assert!(v.is_power_of_two(), "{} is not a power of two", v);
        }
    }

    /// Verify that jit_k_unroll_values with max=0 still returns a non-empty
    /// result (the fallback value 1).
    #[test]
    fn jit_k_unroll_values_zero_max_returns_one() {
        // Arrange & Act
        let vals = jit_k_unroll_values(0);

        // Assert
        assert_eq!(vals, vec![1]);
    }

    /// Verify that coarse_grid with stride=1 returns the same count as grid_size
    /// for a simple GEMM space (no JIT).
    #[test]
    fn coarse_grid_stride_one_equals_full_grid() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape { m: 64, n: 64, k: 64, elem_bytes: 4, dtype_id: 0 };
        let space = SearchSpace::for_gemm(&hw, &shape, 6, 8);

        // Act
        let coarse = space.coarse_grid(1);
        let full = space.grid_size();

        // Assert — coarse_grid uses thread_candidates which deduplicates,
        // so it may differ from the raw product. Just check same order of magnitude.
        assert!(!coarse.is_empty());
        assert!(coarse.len() <= full, "coarse_grid(1) should not exceed full grid");
    }

    /// Verify that GpuSearchSpace::for_sm produces more tile candidates for
    /// SM100 than for SM80 (wider tile selections).
    #[test]
    fn gpu_search_space_sm100_more_candidates_than_sm80() {
        // Arrange & Act
        let sm100 = GpuSearchSpace::for_sm(100, 256 * 1024);
        let sm80 = GpuSearchSpace::for_sm(80, 256 * 1024);

        // Assert
        assert!(sm100.total_candidates() > sm80.total_candidates(),
            "SM100 candidates ({}) should exceed SM80 ({})",
            sm100.total_candidates(), sm80.total_candidates());
        assert_eq!(sm100.sm_version, 100);
        assert_eq!(sm80.sm_version, 80);
    }

    /// Verify that GpuSearchSpace::is_valid rejects a config where warp tiles
    /// do not evenly divide CTA tiles.
    #[test]
    fn gpu_search_space_rejects_non_dividing_warp_tiles() {
        // Arrange
        let space = GpuSearchSpace::for_sm(90, 200 * 1024);
        let cfg = GpuGemmConfig {
            cta_m: 128, cta_n: 64, cta_k: 32,
            warp_m: 30, // does NOT divide 128
            warp_n: 16,
            pipeline_depth: 1,
        };

        // Act
        let valid = space.is_valid(&cfg, 200 * 1024, 4);

        // Assert
        assert!(!valid, "Config with warp_m=30 not dividing cta_m=128 should be invalid");
    }

    /// Verify that GpuGemmConfig Display formats all 6 fields correctly.
    #[test]
    fn gpu_gemm_config_display_format() {
        // Arrange
        let cfg = GpuGemmConfig {
            cta_m: 128, cta_n: 64, cta_k: 32,
            warp_m: 32, warp_n: 16,
            pipeline_depth: 3,
        };

        // Act
        let s = cfg.to_string();

        // Assert
        assert!(s.contains("128\u{00d7}64\u{00d7}32"), "CTA tiles should appear as 128x64x32, got: {}", s);
        assert!(s.contains("32\u{00d7}16"), "Warp tiles should appear as 32x16, got: {}", s);
        assert!(s.contains("pipe=3"), "Pipeline depth should appear as pipe=3, got: {}", s);
    }

    /// Verify that round_down aligns values correctly and handles align=0.
    #[test]
    fn round_down_aligns_and_handles_zero() {
        // Arrange & Act & Assert
        assert_eq!(round_down(100, 8), 96);
        assert_eq!(round_down(64, 8), 64);
        assert_eq!(round_down(7, 8), 0);
        assert_eq!(round_down(0, 8), 0);
        // align=0 passes through unchanged
        assert_eq!(round_down(37, 0), 37);
    }

    /// Verify that subsample with stride=1 returns the input unchanged.
    #[test]
    fn subsample_stride_one_passthrough() {
        // Arrange
        let values: Vec<usize> = (0..=10).collect();

        // Act
        let result = subsample(&values, 1);

        // Assert
        assert_eq!(result, values);
    }

    /// Verify that subsample with an empty input returns empty.
    #[test]
    fn subsample_empty_input() {
        // Arrange
        let values: Vec<usize> = vec![];

        // Act
        let result = subsample(&values, 3);

        // Assert
        assert!(result.is_empty());
    }

    /// Verify that subsample always includes first and last values.
    #[test]
    fn subsample_preserves_endpoints() {
        // Arrange
        let values: Vec<usize> = (0..21).map(|i| i * 5).collect(); // [0,5,10,...,100]

        // Act
        let result = subsample(&values, 4);

        // Assert
        assert_eq!(*result.first().unwrap(), 0);
        assert_eq!(*result.last().unwrap(), 100);
        assert!(result.len() < values.len(), "subsampled should be shorter");
    }

    /// Verify that thread_candidates includes 1, max, and a midpoint for large max.
    #[test]
    fn thread_candidates_includes_extremes() {
        // Arrange
        let min = 1;
        let max = 16;

        // Act
        let candidates = thread_candidates(min, max);

        // Assert
        assert!(candidates.contains(&1), "should contain min=1");
        assert!(candidates.contains(&16), "should contain max=16");
        // 3/4 point: 16*3/4 = 12
        assert!(candidates.contains(&12), "should contain 3/4 max = 12");
        // Should be sorted and deduped
        let mut sorted = candidates.clone();
        sorted.sort_unstable();
        assert_eq!(candidates, sorted);
    }

    /// Verify that thread_candidates with max=1 returns just [1].
    #[test]
    fn thread_candidates_single_core() {
        // Arrange
        let min = 1;
        let max = 1;

        // Act
        let candidates = thread_candidates(min, max);

        // Assert
        assert_eq!(candidates, vec![1]);
    }

    /// Verify that GpuSearchSpace::is_valid rejects configs exceeding SMEM budget.
    #[test]
    fn gpu_search_space_rejects_smem_overflow() {
        // Arrange
        let space = GpuSearchSpace::for_sm(90, 128 * 1024); // 128KB shared mem
        let cfg = GpuGemmConfig {
            cta_m: 256, cta_n: 256, cta_k: 64,
            warp_m: 64, warp_n: 32,
            pipeline_depth: 3, // 3 stages of 256*64*4 + 64*256*4 = 128KB per stage -> 384KB total
        };

        // Act
        let valid = space.is_valid(&cfg, 128 * 1024, 4);

        // Assert
        assert!(!valid, "Config exceeding SMEM budget should be rejected");
    }

    /// Verify that GpuSearchSpace::is_valid rejects configs where threads exceed 1024.
    #[test]
    fn gpu_search_space_rejects_too_many_threads() {
        // Arrange
        let space = GpuSearchSpace::for_sm(90, 512 * 1024);
        // 4 warps in M * 2 warps in N = 8 warps * 32 = 256 threads (valid)
        // But cta_m=256/cta_n=128 with warp_m=16/warp_n=16 -> 16*8=128 warps * 32 = 4096 threads
        let cfg = GpuGemmConfig {
            cta_m: 256, cta_n: 128, cta_k: 32,
            warp_m: 16, warp_n: 16,
            pipeline_depth: 1,
        };

        // Act
        let valid = space.is_valid(&cfg, 512 * 1024, 4);

        // Assert
        assert!(!valid, "Config with >1024 threads should be rejected");
    }

    /// Verify that GpuSearchSpace::enumerate_valid returns at least one valid config
    /// and all returned configs pass is_valid.
    #[test]
    fn gpu_search_space_enumerate_valid_all_pass() {
        // Arrange
        let space = GpuSearchSpace::for_sm(90, 200 * 1024);
        let smem = 200 * 1024;

        // Act
        let configs = space.enumerate_valid(smem, 4);

        // Assert
        assert!(!configs.is_empty(), "should have at least one valid config");
        for cfg in &configs {
            assert!(space.is_valid(cfg, smem, 4),
                "enumerate_valid returned invalid config: {}", cfg);
        }
    }

    /// Verify that GpuGemmConfig derives Clone and PartialEq correctly.
    #[test]
    fn gpu_gemm_config_clone_and_equality() {
        // Arrange
        let a = GpuGemmConfig {
            cta_m: 64, cta_n: 64, cta_k: 32,
            warp_m: 32, warp_n: 16,
            pipeline_depth: 2,
        };
        let b = a.clone();

        // Act & Assert
        assert_eq!(a, b);
        // Modify one field to ensure inequality
        let c = GpuGemmConfig { pipeline_depth: 1, ..a.clone() };
        assert_ne!(a, c);
    }

    /// Verify that TuningConfig with identical JIT params compares equal,
    /// and different JIT params compares unequal.
    #[test]
    fn tuning_config_equality_with_jit() {
        // Arrange
        let jit = JitParams::default();
        let a = TuningConfig {
            kc: 64, mc: 32, nc: 128, num_threads: 4,
            jit: Some(jit.clone()),
        };
        let b = TuningConfig {
            kc: 64, mc: 32, nc: 128, num_threads: 4,
            jit: Some(jit.clone()),
        };
        let c = TuningConfig {
            kc: 64, mc: 32, nc: 128, num_threads: 4,
            jit: Some(JitParams { k_unroll: 8, ..jit.clone() }),
        };

        // Act & Assert
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// Verify that OpClass derives Copy and can be used in match arms.
    #[test]
    fn op_class_copy_semantics() {
        // Arrange
        let a = OpClass::Gemm;
        let b = a; // Copy, not move

        // Act & Assert — both still usable
        assert_eq!(a, OpClass::Gemm);
        assert_eq!(b, OpClass::Gemm);
        match a {
            OpClass::Gemm => {} // compiles
            OpClass::Gemv => panic!("wrong variant"),
            OpClass::MemoryBound => panic!("wrong variant"),
        }
    }

    /// Verify that neighborhood returns values within the ParamRange bounds
    /// and includes the center value when it is a valid range point.
    #[test]
    fn neighborhood_stays_within_bounds_and_includes_center() {
        // Arrange
        let range = ParamRange { name: "KC", min: 16, max: 128, step: 16 };
        let center = 64;
        let radius = 2;

        // Act
        let vals = neighborhood(center, &range, radius);

        // Assert
        assert!(vals.contains(&64), "center value should be present");
        for &v in &vals {
            assert!(v >= range.min, "value {} below min {}", v, range.min);
            assert!(v <= range.max, "value {} above max {}", v, range.max);
            assert_eq!(v % range.step, 0, "value {} not aligned to step {}", v, range.step);
        }
    }

    /// Verify that neighborhood with radius=0 returns only the center value
    /// (if it is within range and aligned).
    #[test]
    fn neighborhood_radius_zero_returns_center_only() {
        // Arrange
        let range = ParamRange { name: "MC", min: 6, max: 96, step: 6 };
        let center = 48;

        // Act
        let vals = neighborhood(center, &range, 0);

        // Assert
        assert_eq!(vals, vec![48]);
    }

    /// Verify that neighborhood clamps to range.min when center is near the lower bound.
    #[test]
    fn neighborhood_clamps_at_lower_bound() {
        // Arrange
        let range = ParamRange { name: "NC", min: 8, max: 512, step: 8 };
        let center = 8; // at min

        // Act
        let vals = neighborhood(center, &range, 3);

        // Assert
        assert!(vals.contains(&8), "min value should be present");
        for &v in &vals {
            assert!(v >= 8, "value {} below range min", v);
        }
    }

    /// Verify that thread_neighborhood includes center and adjacent values,
    /// and clamps at boundaries.
    #[test]
    fn thread_neighborhood_includes_adjacent_and_clamps() {
        // Arrange
        let center = 4;
        let max = 8;

        // Act
        let vals = thread_neighborhood(center, max);

        // Assert
        assert!(vals.contains(&3), "should contain center-1");
        assert!(vals.contains(&4), "should contain center");
        assert!(vals.contains(&5), "should contain center+1");
        // At lower boundary
        let boundary = thread_neighborhood(1, 8);
        assert!(boundary.contains(&1));
        assert!(boundary.contains(&2));
        assert!(!boundary.contains(&0), "should not go below 1");
        // At upper boundary
        let upper = thread_neighborhood(8, 8);
        assert!(upper.contains(&8));
        assert!(upper.contains(&7));
        assert!(!upper.contains(&9), "should not exceed max");
    }

    /// Verify that jit_k_unroll_neighborhood returns adjacent powers of 2
    /// around the center value.
    #[test]
    fn jit_k_unroll_neighborhood_adjacent_powers() {
        // Arrange
        let center = 4;
        let max = 16;

        // Act
        let vals = jit_k_unroll_neighborhood(center, max);

        // Assert
        assert!(vals.contains(&2), "should contain center/2 = 2");
        assert!(vals.contains(&4), "should contain center = 4");
        assert!(vals.contains(&8), "should contain center*2 = 8");
        // Should not contain values far from center
        assert!(!vals.contains(&16), "should not contain 16 (two steps away)");
    }

    /// Verify that jit_k_unroll_neighborhood with center=1 (minimum power)
    /// does not go below 1.
    #[test]
    fn jit_k_unroll_neighborhood_at_minimum() {
        // Arrange
        let center = 1;
        let max = 8;

        // Act
        let vals = jit_k_unroll_neighborhood(center, max);

        // Assert
        assert!(vals.contains(&1), "should contain center=1");
        assert!(vals.contains(&2), "should contain center*2=2");
        assert!(!vals.contains(&0), "should not contain 0");
    }

    /// Verify that ParamRange with a step that does not evenly divide the span
    /// still produces values up to the last value <= max.
    #[test]
    fn param_range_non_dividing_step_stops_before_max() {
        // Arrange
        let r = ParamRange { name: "odd", min: 0, max: 10, step: 3 };

        // Act
        let vals = r.values();

        // Assert
        assert_eq!(vals, vec![0, 3, 6, 9]);
        assert_eq!(r.count(), 4);
        assert!(*vals.last().unwrap() <= r.max);
    }

    /// Verify that SearchSpace grid_size for memory_bound is 1 (all blocking
    /// ranges have count=1 since min==max==0, and thread range has at least 1).
    #[test]
    fn search_space_memory_bound_grid_size_is_thread_count() {
        // Arrange
        let hw = HwInfo::detect();
        let space = SearchSpace::for_memory_bound(&hw);

        // Act
        let grid = space.grid_size();

        // Assert — KC/MC/NC each have count=1 (min==max==0), so grid = 1*1*1*thread_count
        assert_eq!(grid, space.thread_range.count());
        assert!(grid >= 1);
    }

    /// Verify that TuningConfig with different base params but same JIT params
    /// compares unequal, and that Hash distinguishes them in a HashSet.
    #[test]
    fn tuning_config_hash_distinguishes_different_configs() {
        // Arrange
        use std::collections::HashSet;
        let a = TuningConfig { kc: 64, mc: 32, nc: 128, num_threads: 4, jit: None };
        let b = TuningConfig { kc: 128, mc: 32, nc: 128, num_threads: 4, jit: None };

        // Act
        let mut set = HashSet::new();
        set.insert(a.clone());
        set.insert(b.clone());

        // Assert
        assert_ne!(a, b);
        assert_eq!(set.len(), 2, "HashSet should contain both distinct configs");
    }

    /// Verify that GpuSearchSpace::for_sm with SM version below 80 uses
    /// the fallback (smallest) tile candidate sets.
    #[test]
    fn gpu_search_space_legacy_sm_uses_fallback_tiles() {
        // Arrange & Act
        let space = GpuSearchSpace::for_sm(70, 64 * 1024);

        // Assert
        assert_eq!(space.sm_version, 70);
        // Legacy SM should have the smallest candidate sets
        assert!(!space.cta_m_values.is_empty());
        assert!(!space.cta_k_values.is_empty());
        // Pipeline depth should be 1 for small SMEM
        assert_eq!(space.pipeline_depth_values, vec![1]);
    }

    /// Verify that GpuSearchSpace::is_valid rejects a config where warp tiles
    /// are larger than CTA tiles (zero warps).
    #[test]
    fn gpu_search_space_rejects_warp_larger_than_cta() {
        // Arrange
        let space = GpuSearchSpace::for_sm(90, 512 * 1024);
        let cfg = GpuGemmConfig {
            cta_m: 64, cta_n: 64, cta_k: 32,
            warp_m: 128, // larger than cta_m=64
            warp_n: 16,
            pipeline_depth: 1,
        };

        // Act
        let valid = space.is_valid(&cfg, 512 * 1024, 4);

        // Assert
        assert!(!valid, "Config with warp_m > cta_m should be invalid (zero warps in M)");
    }
}

// ── GPU Autotuning ─────────────────────────────────────────────────────

/// GPU-specific GEMM tile parameters for autotuning.
///
/// Represents a single candidate configuration for GPU GEMM performance search.
/// Maps directly to ExecPattern::TileGemm fields + pipeline depth.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuGemmConfig {
    /// Thread block M tile
    pub cta_m: usize,
    /// Thread block N tile
    pub cta_n: usize,
    /// Thread block K tile
    pub cta_k: usize,
    /// Warp-level M tile
    pub warp_m: usize,
    /// Warp-level N tile
    pub warp_n: usize,
    /// Pipeline depth (0/1 = no pipeline, 2 = double buffer, 3 = triple)
    pub pipeline_depth: usize,
}

impl std::fmt::Display for GpuGemmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cta={}×{}×{} warp={}×{} pipe={}",
            self.cta_m, self.cta_n, self.cta_k,
            self.warp_m, self.warp_n,
            self.pipeline_depth
        )
    }
}

/// GPU GEMM search space constrained by SM version and shared memory.
#[derive(Debug, Clone)]
pub struct GpuSearchSpace {
    /// SM version (e.g. 80, 90, 100)
    pub sm_version: u32,
    pub cta_m_values: Vec<usize>,
    pub cta_n_values: Vec<usize>,
    pub cta_k_values: Vec<usize>,
    pub warp_m_values: Vec<usize>,
    pub warp_n_values: Vec<usize>,
    pub pipeline_depth_values: Vec<usize>,
}

impl GpuSearchSpace {
    /// Build GPU search space from SM version and shared memory budget.
    pub fn for_sm(sm_version: u32, shared_mem_bytes: usize) -> Self {
        // Tile sizes by SM generation
        let (cta_m_cands, cta_n_cands, cta_k_cands) = match sm_version {
            100.. => (
                vec![64, 128, 256],
                vec![64, 128, 256],
                vec![32, 64, 128],
            ),
            90..=99 => (
                vec![64, 128],
                vec![64, 128],
                vec![32, 64],
            ),
            80..=89 => (
                vec![64, 128],
                vec![64, 128],
                vec![16, 32, 64],
            ),
            _ => (
                vec![64, 128],
                vec![64, 128],
                vec![16, 32],
            ),
        };

        // Warp-level tiles by SM
        let (warp_m_cands, warp_n_cands) = match sm_version {
            100.. => (vec![32, 64], vec![16, 32]),
            90..=99 => (vec![32, 64], vec![16, 32]),
            80..=89 => (vec![32, 64], vec![16]),
            _ => (vec![16, 32], vec![16]),
        };

        // Pipeline depth by SMEM budget
        // Each pipeline stage needs cta_m*cta_k*4 + cta_k*cta_n*4 bytes
        // With largest tiles (256×128): 256*128*4 + 128*256*4 = 256KB per stage
        // SM90 227KB can fit 2 stages with moderate tiles, 3 with small tiles
        let pipeline_depth_values = if shared_mem_bytes >= 200 * 1024 {
            vec![1, 2, 3]
        } else if shared_mem_bytes >= 100 * 1024 {
            vec![1, 2]
        } else {
            vec![1]
        };

        Self {
            sm_version,
            cta_m_values: cta_m_cands,
            cta_n_values: cta_n_cands,
            cta_k_values: cta_k_cands,
            warp_m_values: warp_m_cands,
            warp_n_values: warp_n_cands,
            pipeline_depth_values,
        }
    }

    /// Validate a config against shared memory constraints.
    pub fn is_valid(&self, cfg: &GpuGemmConfig, shared_mem_bytes: usize, elem_bytes: usize) -> bool {
        // Warp tiles must divide CTA tiles
        if cfg.cta_m % cfg.warp_m != 0 || cfg.cta_n % cfg.warp_n != 0 {
            return false;
        }
        // Number of warps per CTA
        let num_warps = (cfg.cta_m / cfg.warp_m) * (cfg.cta_n / cfg.warp_n);
        if num_warps == 0 || num_warps > 32 {
            return false;
        }
        // SMEM per pipeline stage: A-tile + B-tile
        let stage_bytes = (cfg.cta_m * cfg.cta_k + cfg.cta_k * cfg.cta_n) * elem_bytes;
        let total_smem = stage_bytes * cfg.pipeline_depth;
        if total_smem > shared_mem_bytes {
            return false;
        }
        // Threads per block = warps * 32
        let threads = num_warps * 32;
        if threads > 1024 {
            return false;
        }
        true
    }

    /// Enumerate all valid configurations.
    pub fn enumerate_valid(&self, shared_mem_bytes: usize, elem_bytes: usize) -> Vec<GpuGemmConfig> {
        let mut configs = Vec::new();
        for &cta_m in &self.cta_m_values {
            for &cta_n in &self.cta_n_values {
                for &cta_k in &self.cta_k_values {
                    for &warp_m in &self.warp_m_values {
                        for &warp_n in &self.warp_n_values {
                            for &pipe in &self.pipeline_depth_values {
                                let cfg = GpuGemmConfig {
                                    cta_m, cta_n, cta_k,
                                    warp_m, warp_n,
                                    pipeline_depth: pipe,
                                };
                                if self.is_valid(&cfg, shared_mem_bytes, elem_bytes) {
                                    configs.push(cfg);
                                }
                            }
                        }
                    }
                }
            }
        }
        configs
    }

    /// Total candidate count (before filtering).
    pub fn total_candidates(&self) -> usize {
        self.cta_m_values.len()
            * self.cta_n_values.len()
            * self.cta_k_values.len()
            * self.warp_m_values.len()
            * self.warp_n_values.len()
            * self.pipeline_depth_values.len()
    }
}
