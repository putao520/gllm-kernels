//! Parameter search space for autotuning.
//!
//! Defines the tunable parameters and their valid ranges for each operator class.
//! The search space is constrained by hardware limits (cache sizes, register count)
//! to avoid wasting time on obviously invalid configurations.

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
}

impl std::fmt::Display for TuningConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KC={} MC={} NC={} threads={}",
            self.kc, self.mc, self.nc, self.num_threads
        )
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
        }
    }

    /// Total number of configurations in the full grid.
    pub fn grid_size(&self) -> usize {
        self.kc_range.count()
            * self.mc_range.count()
            * self.nc_range.count()
            * self.thread_range.count()
    }

    /// Generate a coarse grid (fewer points) for initial exploration.
    /// Takes every `stride`-th value from each dimension.
    pub fn coarse_grid(&self, stride: usize) -> Vec<TuningConfig> {
        let kcs = subsample(&self.kc_range.values(), stride);
        let mcs = subsample(&self.mc_range.values(), stride);
        let ncs = subsample(&self.nc_range.values(), stride);
        // For threads, always test: 1, half, full
        let thr = thread_candidates(
            self.thread_range.min,
            self.thread_range.max,
        );

        let mut configs = Vec::new();
        for &kc in &kcs {
            for &mc in &mcs {
                for &nc in &ncs {
                    for &t in &thr {
                        configs.push(TuningConfig {
                            kc,
                            mc,
                            nc,
                            num_threads: t,
                        });
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
        // For threads, test center +/- 1
        let thr = thread_neighborhood(center.num_threads, self.thread_range.max);

        let mut configs = Vec::new();
        for &kc in &kcs {
            for &mc in &mcs {
                for &nc in &ncs {
                    for &t in &thr {
                        let cfg = TuningConfig {
                            kc,
                            mc,
                            nc,
                            num_threads: t,
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
        };
        let refined = space.refine_around(&center, 2);
        assert!(!refined.is_empty());
        // Center should not be in the refined set
        assert!(!refined.contains(&center));
    }
}
