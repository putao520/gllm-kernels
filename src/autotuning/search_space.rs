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
}

impl std::fmt::Display for ProblemShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{}x{}_e{}",
            self.m, self.n, self.k, self.elem_bytes
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
    // NOTE: result is non-empty (pushed values[0] above), values is non-empty (checked at entry).
    if *result.last().unwrap() != *values.last().unwrap() {
        result.push(*values.last().unwrap());
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
}
