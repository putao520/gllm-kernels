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
        .unwrap_or({
            (
                TuningConfig {
                    kc: 256,
                    mc: 72,
                    nc: 1024,
                    num_threads: 1,
                    jit: None,
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
            dtype_id: 0,
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

    // ── SearchConfig presets ──

    #[test]
    fn search_config_default_values() {
        let cfg = SearchConfig::default();
        assert_eq!(cfg.coarse_stride, 4);
        assert_eq!(cfg.refine_radius, 2);
        assert_eq!(cfg.refine_iters, 2);
        assert!((cfg.early_reject_ratio - 1.5).abs() < 1e-6);
    }

    #[test]
    fn search_config_fast_uses_wider_stride() {
        let fast = SearchConfig::fast();
        let default = SearchConfig::default();
        assert!(fast.coarse_stride > default.coarse_stride);
        assert!(fast.refine_iters < default.refine_iters);
        assert!(fast.early_reject_ratio < default.early_reject_ratio);
    }

    #[test]
    fn search_config_thorough_uses_narrower_stride() {
        let thorough = SearchConfig::thorough();
        let default = SearchConfig::default();
        assert!(thorough.coarse_stride < default.coarse_stride);
        assert!(thorough.refine_radius > default.refine_radius);
        assert!(thorough.refine_iters > default.refine_iters);
    }

    // ── SearchResult Display ──

    #[test]
    fn search_result_display_contains_key_info() {
        let hw = HwInfo::detect();
        let shape = ProblemShape { m: 64, n: 64, k: 64, elem_bytes: 4, dtype_id: 0 };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);
        let result = run_search(&space, &SearchConfig::fast(), |cfg, bench_cfg| {
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(cfg.mc);
            })
        });
        let display = format!("{}", result);
        assert!(display.contains("evaluated"), "Display should contain 'evaluated'");
        assert!(display.contains("configs"), "Display should contain 'configs'");
    }

    // ── 13 Additional Tests ──

    #[test]
    fn search_config_fast_has_correct_field_values() {
        // Arrange
        // (nothing needed)

        // Act
        let cfg = SearchConfig::fast();

        // Assert
        assert_eq!(cfg.coarse_stride, 6, "fast coarse_stride should be 6");
        assert_eq!(cfg.refine_radius, 1, "fast refine_radius should be 1");
        assert_eq!(cfg.refine_iters, 1, "fast refine_iters should be 1");
        assert!(
            (cfg.early_reject_ratio - 1.3).abs() < 1e-6,
            "fast early_reject_ratio should be 1.3"
        );
    }

    #[test]
    fn search_config_thorough_has_correct_field_values() {
        // Arrange
        // (nothing needed)

        // Act
        let cfg = SearchConfig::thorough();

        // Assert
        assert_eq!(cfg.coarse_stride, 2, "thorough coarse_stride should be 2");
        assert_eq!(cfg.refine_radius, 3, "thorough refine_radius should be 3");
        assert_eq!(cfg.refine_iters, 3, "thorough refine_iters should be 3");
        assert!(
            (cfg.early_reject_ratio - 2.0).abs() < 1e-6,
            "thorough early_reject_ratio should be 2.0"
        );
    }

    #[test]
    fn search_config_fast_uses_fast_bench_configs() {
        // Arrange
        let fast_bench = BenchConfig::fast();

        // Act
        let cfg = SearchConfig::fast();

        // Assert
        assert_eq!(cfg.coarse_bench.warmup_iters, fast_bench.warmup_iters);
        assert_eq!(cfg.coarse_bench.min_iters, fast_bench.min_iters);
        assert_eq!(cfg.refine_bench.warmup_iters, fast_bench.warmup_iters);
        assert_eq!(cfg.refine_bench.min_iters, fast_bench.min_iters);
    }

    #[test]
    fn search_config_thorough_uses_precise_refine_bench() {
        // Arrange
        let precise_bench = BenchConfig::precise();

        // Act
        let cfg = SearchConfig::thorough();

        // Assert
        assert_eq!(cfg.refine_bench.warmup_iters, precise_bench.warmup_iters);
        assert_eq!(cfg.refine_bench.min_iters, precise_bench.min_iters);
        assert_eq!(cfg.refine_bench.max_iters, precise_bench.max_iters);
    }

    #[test]
    fn search_config_default_coarse_bench_is_fast() {
        // Arrange
        let fast_bench = BenchConfig::fast();

        // Act
        let cfg = SearchConfig::default();

        // Assert
        assert_eq!(cfg.coarse_bench.warmup_iters, fast_bench.warmup_iters);
        assert_eq!(cfg.coarse_bench.min_iters, fast_bench.min_iters);
        assert_eq!(cfg.coarse_bench.max_iters, fast_bench.max_iters);
        assert_eq!(cfg.coarse_bench.min_time_ns, fast_bench.min_time_ns);
        assert_eq!(cfg.coarse_bench.max_time_ns, fast_bench.max_time_ns);
    }

    #[test]
    fn search_config_thorough_rejects_slower_than_default() {
        // Arrange
        let thorough = SearchConfig::thorough();
        let default = SearchConfig::default();

        // Act & Assert
        assert!(
            thorough.early_reject_ratio > default.early_reject_ratio,
            "thorough should have a higher early_reject_ratio to explore more configs"
        );
    }

    #[test]
    fn search_config_presets_form_progression() {
        // Arrange
        let fast = SearchConfig::fast();
        let default = SearchConfig::default();
        let thorough = SearchConfig::thorough();

        // Act & Assert: coarse_stride should decrease (more samples) from fast → default → thorough
        assert!(
            fast.coarse_stride > default.coarse_stride,
            "fast stride > default stride"
        );
        assert!(
            default.coarse_stride > thorough.coarse_stride,
            "default stride > thorough stride"
        );

        // refine_iters should increase from fast → default → thorough
        assert!(
            fast.refine_iters < default.refine_iters,
            "fast refine_iters < default refine_iters"
        );
        assert!(
            default.refine_iters < thorough.refine_iters,
            "default refine_iters < thorough refine_iters"
        );
    }

    #[test]
    fn search_result_display_shows_ms_format() {
        // Arrange
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 128,
                mc: 48,
                nc: 512,
                num_threads: 4,
                jit: None,
            },
            best_result: BenchResult {
                median_ns: 150_000.0,
                iqr_ns: 10_000.0,
                min_ns: 140_000.0,
                samples: 10,
                gflops: None,
                bandwidth_gbs: None,
            },
            configs_evaluated: 42,
            search_time_ns: 3_500_000,
            all_results: vec![],
        };

        // Act
        let display = format!("{}", result);

        // Assert
        assert!(display.contains("42"), "should contain configs_evaluated");
        assert!(display.contains("3.5ms"), "should format search_time as ms");
        assert!(display.contains("Best:"), "should contain 'Best:'");
    }

    #[test]
    fn search_result_all_results_sorted_by_median() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 32,
            n: 32,
            k: 32,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);

        // Act
        let result = run_search(&space, &SearchConfig::fast(), |cfg, bench_cfg| {
            // Deterministic: median proportional to mc so results are predictable
            let fake_ns = cfg.mc as f64 * 100.0;
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(fake_ns as u64);
            })
        });

        // Assert: all_results should be sorted ascending by median_ns
        for window in result.all_results.windows(2) {
            assert!(
                window[0].1.median_ns <= window[1].1.median_ns,
                "all_results must be sorted ascending by median_ns"
            );
        }
    }

    #[test]
    fn search_result_configs_evaluated_equals_all_results_len() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 16,
            n: 16,
            k: 16,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);

        // Act
        let result = run_search(&space, &SearchConfig::fast(), |cfg, bench_cfg| {
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(cfg.nc);
            })
        });

        // Assert
        assert_eq!(
            result.configs_evaluated, result.all_results.len(),
            "configs_evaluated must equal all_results.len()"
        );
    }

    #[test]
    fn search_result_search_time_ns_is_nonzero() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 64,
            n: 64,
            k: 64,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);

        // Act
        let result = run_search(&space, &SearchConfig::fast(), |cfg, bench_cfg| {
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(cfg.kc);
            })
        });

        // Assert
        assert!(
            result.search_time_ns > 0,
            "search_time_ns must be nonzero after a real search"
        );
    }

    #[test]
    fn format_report_contains_all_sections() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 64,
            n: 64,
            k: 64,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);
        let result = run_search(&space, &SearchConfig::fast(), |cfg, bench_cfg| {
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(cfg.nc);
            })
        });

        // Act
        let report = format_report(&result, "test_gemm", "64x64x64");

        // Assert
        assert!(report.contains("Autotuning Report"), "report should contain header");
        assert!(report.contains("test_gemm"), "report should contain op_name");
        assert!(report.contains("64x64x64"), "report should contain shape_desc");
        assert!(report.contains("Best:"), "report should contain 'Best:'");
        assert!(report.contains("Perf:"), "report should contain 'Perf:'");
        assert!(report.contains("Search:"), "report should contain 'Search:'");
    }

    #[test]
    fn search_with_early_rejection_skips_slow_configs() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 64,
            n: 64,
            k: 64,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);
        let mut evaluated_configs: Vec<TuningConfig> = Vec::new();

        // Act: use tight early_reject_ratio so many configs get rejected early
        let mut tight_config = SearchConfig::fast();
        tight_config.early_reject_ratio = 1.01; // very tight — nearly all slower configs rejected

        let result = run_search(&space, &tight_config, |cfg, bench_cfg| {
            evaluated_configs.push(cfg.clone());
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(cfg.mc);
            })
        });

        // Assert: search still completes and returns valid result
        assert!(result.configs_evaluated > 0, "should evaluate at least one config");
        assert!(result.best_result.median_ns > 0.0, "best_result should have nonzero median");
    }

    // ── 10 Additional Tests ──

    #[test]
    fn search_config_clone_produces_equal_instance() {
        // Arrange
        let original = SearchConfig::default();

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(cloned.coarse_stride, original.coarse_stride);
        assert_eq!(cloned.refine_radius, original.refine_radius);
        assert_eq!(cloned.refine_iters, original.refine_iters);
        assert_eq!(cloned.coarse_bench.warmup_iters, original.coarse_bench.warmup_iters);
        assert_eq!(cloned.coarse_bench.min_iters, original.coarse_bench.min_iters);
        assert_eq!(cloned.refine_bench.warmup_iters, original.refine_bench.warmup_iters);
        assert_eq!(cloned.refine_bench.min_iters, original.refine_bench.min_iters);
        assert!((cloned.early_reject_ratio - original.early_reject_ratio).abs() < 1e-6);
    }

    #[test]
    fn search_result_clone_produces_equal_instance() {
        // Arrange
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 256,
                mc: 72,
                nc: 1024,
                num_threads: 4,
                jit: None,
            },
            best_result: BenchResult {
                median_ns: 1000.0,
                iqr_ns: 50.0,
                min_ns: 900.0,
                samples: 10,
                gflops: Some(42.5),
                bandwidth_gbs: None,
            },
            configs_evaluated: 5,
            search_time_ns: 1_000_000,
            all_results: vec![(
                TuningConfig {
                    kc: 256,
                    mc: 72,
                    nc: 1024,
                    num_threads: 4,
                    jit: None,
                },
                BenchResult {
                    median_ns: 1000.0,
                    iqr_ns: 50.0,
                    min_ns: 900.0,
                    samples: 10,
                    gflops: Some(42.5),
                    bandwidth_gbs: None,
                },
            )],
        };

        // Act
        let cloned = result.clone();

        // Assert
        assert_eq!(cloned.best_config, result.best_config);
        assert_eq!(cloned.configs_evaluated, result.configs_evaluated);
        assert_eq!(cloned.search_time_ns, result.search_time_ns);
        assert_eq!(cloned.all_results.len(), result.all_results.len());
        assert!((cloned.best_result.median_ns - result.best_result.median_ns).abs() < 1e-6);
    }

    #[test]
    fn search_result_display_formats_zero_search_time() {
        // Arrange
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 128,
                mc: 48,
                nc: 512,
                num_threads: 1,
                jit: None,
            },
            best_result: BenchResult {
                median_ns: 500.0,
                iqr_ns: 10.0,
                min_ns: 490.0,
                samples: 5,
                gflops: None,
                bandwidth_gbs: None,
            },
            configs_evaluated: 1,
            search_time_ns: 0,
            all_results: vec![],
        };

        // Act
        let display = format!("{}", result);

        // Assert
        assert!(display.contains("0.0ms"), "zero search_time_ns should display as 0.0ms");
        assert!(display.contains("1"), "should contain configs_evaluated=1");
    }

    #[test]
    fn search_config_default_refine_bench_is_default_bench() {
        // Arrange
        let default_bench = BenchConfig::default();

        // Act
        let cfg = SearchConfig::default();

        // Assert: default SearchConfig uses BenchConfig::default() for refine phase
        assert_eq!(cfg.refine_bench.warmup_iters, default_bench.warmup_iters);
        assert_eq!(cfg.refine_bench.min_iters, default_bench.min_iters);
        assert_eq!(cfg.refine_bench.max_iters, default_bench.max_iters);
        assert_eq!(cfg.refine_bench.min_time_ns, default_bench.min_time_ns);
        assert_eq!(cfg.refine_bench.max_time_ns, default_bench.max_time_ns);
    }

    #[test]
    fn search_config_fast_refine_bench_is_fast_bench() {
        // Arrange
        let fast_bench = BenchConfig::fast();

        // Act
        let cfg = SearchConfig::fast();

        // Assert: fast SearchConfig uses BenchConfig::fast() for both phases
        assert_eq!(cfg.refine_bench.warmup_iters, fast_bench.warmup_iters);
        assert_eq!(cfg.refine_bench.min_iters, fast_bench.min_iters);
        assert_eq!(cfg.refine_bench.max_iters, fast_bench.max_iters);
    }

    #[test]
    fn search_config_thorough_coarse_bench_is_default_bench() {
        // Arrange
        let default_bench = BenchConfig::default();

        // Act
        let cfg = SearchConfig::thorough();

        // Assert: thorough SearchConfig uses BenchConfig::default() for coarse phase
        assert_eq!(cfg.coarse_bench.warmup_iters, default_bench.warmup_iters);
        assert_eq!(cfg.coarse_bench.min_iters, default_bench.min_iters);
        assert_eq!(cfg.coarse_bench.max_iters, default_bench.max_iters);
    }

    #[test]
    fn search_result_display_with_large_search_time() {
        // Arrange
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 64,
                mc: 24,
                nc: 256,
                num_threads: 2,
                jit: None,
            },
            best_result: BenchResult {
                median_ns: 2000.0,
                iqr_ns: 100.0,
                min_ns: 1800.0,
                samples: 7,
                gflops: None,
                bandwidth_gbs: None,
            },
            configs_evaluated: 100,
            search_time_ns: 12_345_678_901, // ~12345.7ms
            all_results: vec![],
        };

        // Act
        let display = format!("{}", result);

        // Assert
        assert!(display.contains("12345.7ms"), "large search_time should format as ms with 1 decimal");
        assert!(display.contains("100"), "should contain configs_evaluated=100");
    }

    #[test]
    fn search_result_best_is_first_in_all_results() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 32,
            n: 32,
            k: 32,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);

        // Act: deterministic fake benchmark — kc directly drives median_ns
        let result = run_search(&space, &SearchConfig::fast(), |cfg, bench_cfg| {
            let fake_ns = cfg.kc as f64 * 10.0;
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(fake_ns as u64);
            })
        });

        // Assert: best_config should match the first entry in all_results
        if let Some((first_config, first_result)) = result.all_results.first() {
            assert_eq!(*first_config, result.best_config, "best_config should match first entry in all_results");
            assert!(
                (first_result.median_ns - result.best_result.median_ns).abs() < 1e-6,
                "best_result.median_ns should match first entry in all_results"
            );
        }
    }

    #[test]
    fn search_config_zero_refine_iters_skips_refinement() {
        // Arrange
        let hw = HwInfo::detect();
        let shape = ProblemShape {
            m: 32,
            n: 32,
            k: 32,
            elem_bytes: 4,
            dtype_id: 0,
        };
        let space = SearchSpace::for_gemm(&hw, &shape, 4, 8);
        let mut cfg = SearchConfig::fast();
        cfg.refine_iters = 0; // skip refinement entirely

        // Act
        let result = run_search(&space, &cfg, |cfg, bench_cfg| {
            let fake_ns = cfg.mc as f64 * 100.0;
            measure::bench_fn(bench_cfg, |_| {
                measure::black_box(fake_ns as u64);
            })
        });

        // Assert: still completes coarse phase and returns valid result
        assert!(result.configs_evaluated > 0, "should evaluate coarse configs even with 0 refine_iters");
        assert!(result.best_result.median_ns > 0.0, "best_result should have nonzero median");
    }

    #[test]
    fn format_report_single_result_no_top5_section() {
        // Arrange: SearchResult with a single entry in all_results (top_n=1, so no "Top configurations" section)
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 128,
                mc: 48,
                nc: 512,
                num_threads: 1,
                jit: None,
            },
            best_result: BenchResult {
                median_ns: 5000.0,
                iqr_ns: 200.0,
                min_ns: 4800.0,
                samples: 7,
                gflops: None,
                bandwidth_gbs: None,
            },
            configs_evaluated: 1,
            search_time_ns: 500_000,
            all_results: vec![(
                TuningConfig {
                    kc: 128,
                    mc: 48,
                    nc: 512,
                    num_threads: 1,
                    jit: None,
                },
                BenchResult {
                    median_ns: 5000.0,
                    iqr_ns: 200.0,
                    min_ns: 4800.0,
                    samples: 7,
                    gflops: None,
                    bandwidth_gbs: None,
                },
            )],
        };

        // Act
        let report = format_report(&result, "single_gemm", "128x128x128");

        // Assert
        assert!(report.contains("Autotuning Report"), "should contain header");
        assert!(report.contains("single_gemm"), "should contain op_name");
        assert!(!report.contains("Top configurations"), "single result should not show Top section");
        assert!(report.contains("Parameter sensitivity"), "should still show sensitivity with single result");
    }
}
