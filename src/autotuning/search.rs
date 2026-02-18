//! Search engine: two-phase grid search with refinement.
//!
//! Phase 1: Coarse grid search over the full parameter space (subsampled).
//! Phase 2: Fine-grained refinement around the best configuration found.
//!
//! This is simpler and more predictable than Bayesian optimization,
//! and sufficient for the ~4D parameter space of GEMM blocking.

use crate::autotuning::measure::{BenchConfig, BenchResult};
use crate::autotuning::search_space::{SearchSpace, TuningConfig};

/// Result of a tuning search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Best configuration found
    pub best_config: TuningConfig,
    /// Benchmark result for the best configuration
    pub best_result: BenchResult,
    /// Total configurations evaluated
    pub configs_evaluated: usize,
    /// Total time spent searching (nanoseconds)
    pub search_time_ns: u64,
    /// All evaluated configurations with their results (sorted best-first)
    pub all_results: Vec<(TuningConfig, BenchResult)>,
}

impl std::fmt::Display for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Best: {} | {} | evaluated {} configs in {:.1}ms",
            self.best_config,
            self.best_result,
            self.configs_evaluated,
            self.search_time_ns as f64 / 1_000_000.0,
        )
    }
}

/// Configuration for the search process.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Coarse grid stride (sample every N-th value)
    pub coarse_stride: usize,
    /// Refinement radius (steps around best in each dimension)
    pub refine_radius: usize,
    /// Number of refinement iterations
    pub refine_iters: usize,
    /// Benchmark config for coarse phase
    pub coarse_bench: BenchConfig,
    /// Benchmark config for refinement phase
    pub refine_bench: BenchConfig,
    /// Early termination: skip configs that are unlikely to beat current best.
    /// If a config's first few samples are > threshold * best_median, skip it.
    pub early_reject_ratio: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            coarse_stride: 4,
            refine_radius: 2,
            refine_iters: 2,
            coarse_bench: BenchConfig::fast(),
            refine_bench: BenchConfig::default(),
            early_reject_ratio: 1.5,
        }
    }
}

impl SearchConfig {
    /// Fast search: fewer points, less precision.
    pub fn fast() -> Self {
        SearchConfig {
            coarse_stride: 6,
            refine_radius: 1,
            refine_iters: 1,
            coarse_bench: BenchConfig::fast(),
            refine_bench: BenchConfig::fast(),
            early_reject_ratio: 1.3,
        }
    }

    /// Thorough search: more points, higher precision.
    pub fn thorough() -> Self {
        SearchConfig {
            coarse_stride: 2,
            refine_radius: 3,
            refine_iters: 3,
            coarse_bench: BenchConfig::default(),
            refine_bench: BenchConfig::precise(),
            early_reject_ratio: 2.0,
        }
    }
}

/// Run a two-phase search over the given search space.
///
/// `bench_one` is a closure that benchmarks a single configuration.
/// It receives the config and bench settings, and returns a BenchResult.
pub fn run_search<F>(
    space: &SearchSpace,
    search_config: &SearchConfig,
    mut bench_one: F,
) -> SearchResult
where
    F: FnMut(&TuningConfig, &BenchConfig) -> BenchResult,
{
    let t0 = std::time::Instant::now();
    let mut all_results: Vec<(TuningConfig, BenchResult)> = Vec::new();

    // ── Phase 1: Coarse grid ────────────────────────────────────────────
    let coarse_configs = space.coarse_grid(search_config.coarse_stride);
    let mut best_median = f64::MAX;

    for cfg in &coarse_configs {
        let result = bench_one(cfg, &search_config.coarse_bench);

        if result.median_ns < best_median {
            best_median = result.median_ns;
        }
        all_results.push((cfg.clone(), result));
    }

    // Sort by median time (ascending = fastest first)
    all_results.sort_by(|a, b| {
        a.1.median_ns
            .partial_cmp(&b.1.median_ns)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── Phase 2: Refinement around top candidates ───────────────────────
    for _iter in 0..search_config.refine_iters {
        if all_results.is_empty() {
            break;
        }
        let current_best = all_results[0].0.clone();
        let refine_configs = space.refine_around(&current_best, search_config.refine_radius);

        best_median = all_results[0].1.median_ns;

        for cfg in &refine_configs {
            // Skip if already evaluated
            if all_results.iter().any(|(c, _)| c == cfg) {
                continue;
            }

            let result = bench_one(cfg, &search_config.refine_bench);

            // Early rejection: if this config is much worse, don't bother
            if result.median_ns > best_median * search_config.early_reject_ratio {
                all_results.push((cfg.clone(), result));
                continue;
            }

            if result.median_ns < best_median {
                best_median = result.median_ns;
            }
            all_results.push((cfg.clone(), result));
        }

        // Re-sort after refinement
        all_results.sort_by(|a, b| {
            a.1.median_ns
                .partial_cmp(&b.1.median_ns)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let search_time_ns = t0.elapsed().as_nanos() as u64;
    let configs_evaluated = all_results.len();

    let (best_config, best_result) = all_results
        .first()
        .cloned()
        .unwrap_or_else(|| {
            (
                TuningConfig {
                    kc: 256,
                    mc: 72,
                    nc: 1024,
                    num_threads: 1,
                },
                BenchResult {
                    median_ns: f64::MAX,
                    iqr_ns: 0.0,
                    min_ns: f64::MAX,
                    samples: 0,
                    gflops: None,
                    bandwidth_gbs: None,
                },
            )
        });

    SearchResult {
        best_config,
        best_result,
        configs_evaluated,
        search_time_ns,
        all_results,
    }
}

/// Generate a tuning report as a formatted string.
pub fn format_report(result: &SearchResult, op_name: &str, shape_desc: &str) -> String {
    let mut report = String::new();
    report.push_str(&format!(
        "=== Autotuning Report: {} ({}) ===\n",
        op_name, shape_desc
    ));
    report.push_str(&format!(
        "Search: {} configs in {:.1}ms\n",
        result.configs_evaluated,
        result.search_time_ns as f64 / 1_000_000.0,
    ));
    report.push_str(&format!("Best: {}\n", result.best_config));
    report.push_str(&format!("Perf: {}\n", result.best_result));

    // Top 5
    let top_n = result.all_results.len().min(5);
    if top_n > 1 {
        report.push_str("\nTop configurations:\n");
        for (i, (cfg, res)) in result.all_results.iter().take(top_n).enumerate() {
            let speedup = if i == 0 {
                1.0
            } else {
                res.median_ns / result.best_result.median_ns
            };
            report.push_str(&format!(
                "  #{}: {} | {} | {:.2}x vs best\n",
                i + 1,
                cfg,
                res,
                speedup,
            ));
        }
    }

    // Worst vs best ratio
    if let Some((_, worst)) = result.all_results.last() {
        let ratio = worst.median_ns / result.best_result.median_ns;
        report.push_str(&format!(
            "\nParameter sensitivity: worst/best = {:.1}x\n",
            ratio
        ));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotuning::measure;
    use crate::autotuning::search_space::*;
    use crate::autotuning::hw_info::HwInfo;

    #[test]
    fn test_search_synthetic() {
        // Synthetic benchmark: optimal at kc=128, mc=48
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 256,
            n: 256,
            k: 256,
            elem_bytes: 4,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 6, 16);
        let search_cfg = SearchConfig::fast();

        let result = run_search(&space, &search_cfg, |cfg, bench_cfg| {
            measure::bench_fn(bench_cfg, |_| {
                // Simulate: closer to kc=128 is faster
                let kc_penalty = ((cfg.kc as f64 - 128.0) / 64.0).powi(2);
                let mc_penalty = ((cfg.mc as f64 - 48.0) / 24.0).powi(2);
                let work = (1000.0 + kc_penalty * 500.0 + mc_penalty * 300.0) as u64;
                let mut sum = 0u64;
                for i in 0..work {
                    sum = sum.wrapping_add(i);
                }
                measure::black_box(sum);
            })
        });

        assert!(result.configs_evaluated > 0);
        assert!(result.best_result.median_ns > 0.0);
        eprintln!("{}", format_report(&result, "synthetic_gemm", "256x256x256"));
    }
}
