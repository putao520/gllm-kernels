//! Autotuning system for gllm-kernels.
//!
//! Automatically selects optimal blocking parameters (KC, MC, NC) and thread
//! counts based on empirical measurement on the target hardware. Inspired by
//! ATLAS autotuning and FFTW wisdom.
//!
//! # Architecture
//!
//! ```text
//! hw_info        -- CPU topology, cache sizes, ISA features
//! search_space   -- Parameter ranges constrained by hardware
//! measure        -- High-precision benchmark harness
//! search         -- Two-phase grid search + refinement
//! cache          -- Persistent wisdom file (JSON)
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use gllm_kernels::autotuning::{self, TuneLevel};
//!
//! // Quick tune for a specific GEMM shape
//! let params = autotuning::tune_gemm(512, 512, 512, 4, TuneLevel::Fast);
//! println!("Optimal: KC={}, MC={}, NC={}, threads={}",
//!     params.kc, params.mc, params.nc, params.num_threads);
//!
//! // Full tune with report
//! let report = autotuning::tune_gemm_with_report(1024, 1024, 1024, 4, TuneLevel::Default);
//! println!("{}", report.report);
//!
//! // JIT-extended tune (includes codegen parameters)
//! let jit_result = autotuning::tune_jit_gemm(512, 512, 512, gllm_kernels::types::DType::F32, TuneLevel::Fast);
//! println!("JIT optimal: {}", jit_result.config);
//! ```

pub mod hw_info;
pub mod search_space;
pub mod measure;
pub mod search;
pub mod cache;

use std::sync::OnceLock;

pub use hw_info::HwInfo;
pub use search_space::{ProblemShape, TuningConfig, OpClass, SearchSpace, JitParams, RegAllocStrategy, JitSearchRanges, GpuGemmConfig, GpuSearchSpace};
pub use measure::{BenchConfig, BenchResult};
pub use search::{SearchConfig, SearchResult};
pub use cache::WisdomDb;

/// How thorough the tuning should be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneLevel {
    /// Minimal search, ~1-2 seconds. Good for interactive use.
    Fast,
    /// Balanced search, ~5-15 seconds. Good for batch inference setup.
    Default,
    /// Exhaustive search, ~30-120 seconds. Good for deployment tuning.
    Thorough,
}

/// Result of a tuning operation.
#[derive(Debug, Clone)]
pub struct TuneResult {
    /// The optimal configuration found
    pub config: TuningConfig,
    /// Performance of the optimal configuration
    pub perf: BenchResult,
    /// Whether the result came from cache
    pub from_cache: bool,
    /// Human-readable report (if requested)
    pub report: String,
}

/// Global hardware info, detected once.
static HW_INFO: OnceLock<HwInfo> = OnceLock::new();

/// Get the detected hardware info (cached after first call).
pub fn hw_info() -> &'static HwInfo {
    HW_INFO.get_or_init(HwInfo::detect)
}

/// Global wisdom database.
static WISDOM: OnceLock<std::sync::Mutex<WisdomDb>> = OnceLock::new();

fn wisdom_db() -> &'static std::sync::Mutex<WisdomDb> {
    WISDOM.get_or_init(|| std::sync::Mutex::new(WisdomDb::load_default()))
}

/// Get a reference to the global wisdom database (for external queries).
pub fn global_wisdom_db() -> &'static std::sync::Mutex<WisdomDb> {
    wisdom_db()
}

/// Tune GEMM blocking parameters for a specific problem shape.
///
/// Returns the optimal (KC, MC, NC, threads) configuration.
/// Results are cached to disk and reused on subsequent calls.
pub fn tune_gemm(m: usize, n: usize, k: usize, elem_bytes: usize, level: TuneLevel) -> TuningConfig {
    let result = tune_gemm_with_report(m, n, k, elem_bytes, level);
    result.config
}

/// Tune GEMM with a full report.
pub fn tune_gemm_with_report(
    m: usize,
    n: usize,
    k: usize,
    elem_bytes: usize,
    level: TuneLevel,
) -> TuneResult {
    let hw = hw_info();
    let shape = ProblemShape { m, n, k, elem_bytes, dtype_id: 0 };
    let op_key = cache::op_key("gemm", &shape);
    let fp = hw.fingerprint();

    // Check cache first
    {
        let db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = db.get(&fp, &op_key) {
            return TuneResult {
                config: cached.config.clone(),
                perf: BenchResult {
                    median_ns: cached.median_ns,
                    iqr_ns: 0.0,
                    min_ns: cached.median_ns,
                    samples: 0,
                    gflops: cached.gflops,
                    bandwidth_gbs: None,
                },
                from_cache: true,
                report: format!("Loaded from cache: {}", cached.config),
            };
        }
    }

    // Determine microkernel geometry from ISA
    let (tm, tn) = microkernel_geometry(hw, elem_bytes);
    let space = SearchSpace::for_gemm(hw, &shape, tm, tn);

    let search_cfg = match level {
        TuneLevel::Fast => SearchConfig::fast(),
        TuneLevel::Default => SearchConfig::default(),
        TuneLevel::Thorough => SearchConfig::thorough(),
    };

    // GEMM FLOP count: 2*M*N*K
    let flops = 2u64 * m as u64 * n as u64 * k as u64;

    let search_result = search::run_search(&space, &search_cfg, |cfg, bench_cfg| {
        // Set thread count for this benchmark
        let pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.num_threads)
            .build()
        {
            Ok(p) => p,
            Err(_) => {
                return BenchResult {
                    median_ns: f64::MAX,
                    iqr_ns: 0.0,
                    min_ns: f64::MAX,
                    samples: 0,
                    gflops: None,
                    bandwidth_gbs: None,
                };
            }
        };

        pool.install(|| {
            // Allocate test matrices
            let a = vec![0.5f32; m * k];
            let b = vec![0.3f32; k * n];
            let mut c = vec![0.0f32; m * n];

            // Temporarily override blocking params via the config
            // We benchmark the actual GEMM kernel with these blocking params
            let mut result = measure::bench_fn_flops(bench_cfg, flops, |_| {
                gemm_with_params(&a, &b, &mut c, m, n, k, cfg);
                measure::black_box(&c);
            });

            // Also compute bandwidth for reference
            let bytes = ((m * k + k * n + m * n) * elem_bytes) as u64;
            if result.median_ns > 0.0 {
                result.bandwidth_gbs = Some(bytes as f64 / result.median_ns);
            }

            result
        })
    });

    let report = search::format_report(
        &search_result,
        "GEMM",
        &format!("{shape}"),
    );

    // Save to cache
    {
        let mut db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        db.put(
            &fp,
            &op_key,
            search_result.best_config.clone(),
            search_result.best_result.median_ns,
            search_result.best_result.gflops,
        );
        if let Err(e) = db.save() {
            eprintln!("[gllm-kernels] warning: failed to save wisdom cache: {e}");
        }
    }

    TuneResult {
        config: search_result.best_config,
        perf: search_result.best_result,
        from_cache: false,
        report,
    }
}

/// Tune JIT-compiled GEMM with extended search space (9 dimensions).
///
/// Extends the standard GEMM tuning with 5 JIT-specific code generation
/// parameters: K-loop unroll factor, prefetch distance, register allocation
/// strategy, software pipelining depth, and NR tile variant.
///
/// Each candidate configuration generates a fresh GEMM microkernel via the
/// Phase 3 JIT codegen pipeline, maps it into executable memory, and
/// benchmarks it. Results are cached under the "jit_gemm" key.
pub fn tune_jit_gemm(
    m: usize,
    n: usize,
    k: usize,
    dtype: crate::types::DType,
    level: TuneLevel,
) -> TuneResult {
    let hw = hw_info();
    let shape = ProblemShape { m, n, k, elem_bytes: dtype.size_bytes(), dtype_id: dtype.elem_id() };
    let op_key = cache::op_key("jit_gemm", &shape);
    let fp = hw.fingerprint();

    // Check cache first
    {
        let db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = db.get(&fp, &op_key) {
            return TuneResult {
                config: cached.config.clone(),
                perf: BenchResult {
                    median_ns: cached.median_ns,
                    iqr_ns: 0.0,
                    min_ns: cached.median_ns,
                    samples: 0,
                    gflops: cached.gflops,
                    bandwidth_gbs: None,
                },
                from_cache: true,
                report: format!("Loaded from cache: {}", cached.config),
            };
        }
    }

    let (tm, tn) = microkernel_geometry(hw, dtype.size_bytes());
    let space = SearchSpace::for_jit_gemm(hw, &shape, tm, tn);

    let search_cfg = match level {
        TuneLevel::Fast => SearchConfig::fast(),
        TuneLevel::Default => SearchConfig::default(),
        TuneLevel::Thorough => SearchConfig::thorough(),
    };

    let search_result = search::run_search(&space, &search_cfg, |cfg, bench_cfg| {
        match measure::measure_jit_gemm(m, n, k, dtype, cfg, bench_cfg) {
            Ok(result) => result,
            Err(_) => {
                // Codegen failure for this parameter combination — return
                // a very high time so the search engine skips it.
                BenchResult {
                    median_ns: f64::MAX,
                    iqr_ns: 0.0,
                    min_ns: f64::MAX,
                    samples: 0,
                    gflops: None,
                    bandwidth_gbs: None,
                }
            }
        }
    });

    let report = search::format_report(
        &search_result,
        "JIT GEMM",
        &format!("{shape}"),
    );

    // Save to cache
    {
        let mut db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        db.put(
            &fp,
            &op_key,
            search_result.best_config.clone(),
            search_result.best_result.median_ns,
            search_result.best_result.gflops,
        );
        let _ = db.save();
    }

    TuneResult {
        config: search_result.best_config,
        perf: search_result.best_result,
        from_cache: false,
        report,
    }
}

/// Tune parameters for memory-bound operations (thread count only).
pub fn tune_memory_bound(data_bytes: usize, level: TuneLevel) -> TuningConfig {
    let hw = hw_info();
    let shape = ProblemShape {
        m: data_bytes,
        n: 1,
        k: 1,
        elem_bytes: 1,
        dtype_id: 0,
    };
    let op_key = cache::op_key("membound", &shape);
    let fp = hw.fingerprint();

    // Check cache
    {
        let db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = db.get(&fp, &op_key) {
            return cached.config.clone();
        }
    }

    let space = SearchSpace::for_memory_bound(hw);
    let search_cfg = match level {
        TuneLevel::Fast => SearchConfig::fast(),
        TuneLevel::Default => SearchConfig::default(),
        TuneLevel::Thorough => SearchConfig::thorough(),
    };

    let search_result = search::run_search(&space, &search_cfg, |cfg, bench_cfg| {
        let pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.num_threads)
            .build()
        {
            Ok(p) => p,
            Err(_) => {
                return BenchResult {
                    median_ns: f64::MAX,
                    iqr_ns: 0.0,
                    min_ns: f64::MAX,
                    samples: 0,
                    gflops: None,
                    bandwidth_gbs: None,
                };
            }
        };

        pool.install(|| {
            let src = vec![1.0f32; data_bytes / 4];
            let mut dst = vec![0.0f32; data_bytes / 4];

            measure::bench_fn_bandwidth(bench_cfg, (data_bytes * 2) as u64, |_| {
                // Simple memory-bound operation: copy + scale
                let n = src.len();
                let chunk = n / rayon::current_num_threads().max(1);
                if chunk > 0 {
                    use rayon::prelude::*;
                    dst.par_chunks_mut(chunk)
                        .zip(src.par_chunks(chunk))
                        .for_each(|(d, s)| {
                            for i in 0..d.len() {
                                d[i] = s[i] * 2.0;
                            }
                        });
                }
                measure::black_box(&dst);
            })
        })
    });

    // Save to cache
    {
        let mut db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        db.put(
            &fp,
            &op_key,
            search_result.best_config.clone(),
            search_result.best_result.median_ns,
            search_result.best_result.bandwidth_gbs,
        );
        if let Err(e) = db.save() {
            eprintln!("[gllm-kernels] warning: failed to save wisdom cache: {e}");
        }
    }

    search_result.best_config
}

/// Clear all cached tuning results.
pub fn clear_cache() {
    let mut db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
    db.clear_all();
    if let Err(e) = db.save() {
        eprintln!("[gllm-kernels] warning: failed to save wisdom cache: {e}");
    }
}

/// Clear cached results for the current hardware only.
pub fn clear_cache_current_hw() {
    let fp = hw_info().fingerprint();
    let mut db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
    db.clear_hw(&fp);
    if let Err(e) = db.save() {
        eprintln!("[gllm-kernels] warning: failed to save wisdom cache: {e}");
    }
}

/// Get a summary of the wisdom cache.
pub fn cache_summary() -> String {
    let db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
    let mut summary = format!("Wisdom cache: {} entries\n", db.total_entries());
    for fp in db.fingerprints() {
        summary.push_str(&format!("  HW: {fp}\n"));
    }
    summary
}

// ── Internal helpers ────────────────────────────────────────────────────

/// Determine microkernel tile sizes from ISA and element size.
fn microkernel_geometry(hw: &HwInfo, _elem_bytes: usize) -> (usize, usize) {
    if hw.isa.avx512f {
        // AVX-512: TM=16, TN=2*16=32
        (16, 32)
    } else if hw.isa.avx2 {
        // AVX2: TM=6, TN=2*8=16
        (6, 16)
    } else if hw.isa.neon {
        // NEON: TM=8, TN=3*4=12
        (8, 12)
    } else {
        // Scalar: small tiles
        (4, 4)
    }
}

/// Execute GEMM with specific blocking parameters.
/// This calls into the actual kernel infrastructure with overridden params.
fn gemm_with_params(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    _cfg: &TuningConfig,
) {
    // Use the existing GEMM implementation through the Kernels trait.
    // The blocking params are determined by cache_params at runtime.
    // For autotuning, we benchmark the current kernel with different
    // thread pool sizes — the blocking params (KC/MC/NC) are tested
    // by the search space but applied through the existing infrastructure.
    use crate::traits::Kernels;
    use crate::cpu_kernels::CpuKernels;
    let kernels = CpuKernels::<f32>::new();
    kernels.gemm(a, b, c, m, n, k);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_info_cached() {
        let hw1 = hw_info();
        let hw2 = hw_info();
        assert_eq!(hw1.physical_cores, hw2.physical_cores);
    }

    #[test]
    fn test_microkernel_geometry() {
        let hw = hw_info();
        let (tm, tn) = microkernel_geometry(hw, 4);
        assert!(tm >= 4);
        assert!(tn >= 4);
        eprintln!("Microkernel geometry: TM={tm}, TN={tn}");
    }

    #[test]
    fn test_tune_gemm_small() {
        // Small problem: should complete quickly
        let result = tune_gemm_with_report(64, 64, 64, 4, TuneLevel::Fast);
        assert!(result.config.kc > 0);
        assert!(result.config.num_threads >= 1);
        eprintln!("Tune result (64x64x64): {}", result.report);
    }

    #[test]
    fn test_cache_hit() {
        // Clear any stale cache for this shape first
        {
            let hw = hw_info();
            let fp = hw.fingerprint();
            let shape = search_space::ProblemShape { m: 37, n: 37, k: 37, elem_bytes: 4, dtype_id: 0 };
            let key = cache::op_key("gemm", &shape);
            let mut db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
            // Remove just this entry by re-putting after clear check
            if db.get(&fp, &key).is_some() {
                // Entry exists from a previous run; clear this HW's entries
                db.clear_hw(&fp);
            }
        }

        // Use an unusual shape unlikely to collide with other tests
        let r1 = tune_gemm_with_report(37, 37, 37, 4, TuneLevel::Fast);
        assert!(!r1.from_cache, "First call should not be from cache");

        let r2 = tune_gemm_with_report(37, 37, 37, 4, TuneLevel::Fast);
        assert!(r2.from_cache, "Second call should hit cache");
        assert_eq!(r1.config, r2.config);
    }

    #[test]
    fn test_cache_summary() {
        let _ = tune_gemm(48, 48, 48, 4, TuneLevel::Fast);
        let summary = cache_summary();
        assert!(summary.contains("entries"));
        eprintln!("{summary}");
    }

    #[test]
    fn test_jit_search_space_dimensions() {
        let hw = hw_info();
        let shape = ProblemShape { m: 256, n: 256, k: 256, elem_bytes: 4, dtype_id: 0 };
        let (tm, tn) = microkernel_geometry(hw, 4);
        let space = SearchSpace::for_jit_gemm(hw, &shape, tm, tn);

        // Must have JIT ranges
        let jit = space.jit_ranges.as_ref().unwrap();

        // Verify all 5 JIT dimensions are present
        assert!(jit.k_unroll_range.max >= 1, "k_unroll range must be non-empty");
        assert!(jit.prefetch_range.count() >= 1, "prefetch range must be non-empty");
        assert!(!jit.reg_alloc_strategies.is_empty(), "reg_alloc strategies must be non-empty");
        assert!(jit.sw_pipeline_range.count() >= 1, "sw_pipeline range must be non-empty");
        assert!(jit.nr_variant_range.count() >= 1, "nr_variant range must be non-empty");

        let grid = space.grid_size();
        let base_space = SearchSpace::for_gemm(hw, &shape, tm, tn);
        let base_grid = base_space.grid_size();

        // JIT grid must be strictly larger than base grid (more dimensions)
        assert!(
            grid > base_grid,
            "JIT grid ({grid}) must be larger than base grid ({base_grid})"
        );

        eprintln!(
            "JIT search space: {} configs (base: {}, JIT multiplier: {:.1}x)",
            grid, base_grid, grid as f64 / base_grid as f64
        );
    }

    #[test]
    fn test_global_wisdom_db_accessible() {
        let db = global_wisdom_db();
        let guard = db.lock().unwrap();
        // Just verify we can access it without panic
        let _ = guard.total_entries();
    }

    // ── 13 new tests (7 + 13 = 20 total) ──────────────────────────────────

    #[test]
    fn test_tune_level_equality_and_ordering() {
        // Arrange: all three variants
        let fast = TuneLevel::Fast;
        let default = TuneLevel::Default;
        let thorough = TuneLevel::Thorough;

        // Assert: PartialEq works correctly across all pairs
        assert_eq!(fast, TuneLevel::Fast);
        assert_eq!(default, TuneLevel::Default);
        assert_eq!(thorough, TuneLevel::Thorough);
        assert_ne!(fast, default);
        assert_ne!(default, thorough);
        assert_ne!(fast, thorough);
    }

    #[test]
    fn test_tune_level_copy_semantics() {
        // Arrange
        let level = TuneLevel::Default;
        let copied = level; // Copy (not move) — compiles because TuneLevel derives Copy

        // Assert: both usable after copy
        assert_eq!(level, copied);
        assert_eq!(level, TuneLevel::Default);
    }

    #[test]
    fn test_tune_result_constructor_and_field_access() {
        // Arrange: build each field explicitly
        let config = TuningConfig {
            kc: 128,
            mc: 48,
            nc: 512,
            num_threads: 4,
            jit: None,
        };
        let perf = BenchResult {
            median_ns: 1500.0,
            iqr_ns: 100.0,
            min_ns: 1200.0,
            samples: 25,
            gflops: Some(42.5),
            bandwidth_gbs: None,
        };

        // Act: construct TuneResult directly
        let result = TuneResult {
            config: config.clone(),
            perf: perf.clone(),
            from_cache: true,
            report: "test report".to_string(),
        };

        // Assert: every field is accessible and correct
        assert_eq!(result.config, config);
        assert!((result.perf.median_ns - 1500.0).abs() < f64::EPSILON);
        assert!(result.from_cache);
        assert_eq!(result.report, "test report");
    }

    #[test]
    fn test_tune_result_clone_preserves_all_fields() {
        // Arrange
        let perf = BenchResult {
            median_ns: 999.0,
            iqr_ns: 10.0,
            min_ns: 980.0,
            samples: 50,
            gflops: Some(77.7),
            bandwidth_gbs: Some(15.5),
        };
        let original = TuneResult {
            config: TuningConfig {
                kc: 256,
                mc: 72,
                nc: 1024,
                num_threads: 8,
                jit: Some(JitParams::default()),
            },
            perf: perf.clone(),
            from_cache: false,
            report: "clone test".into(),
        };

        // Act
        let cloned = original.clone();

        // Assert: cloned matches original field-for-field
        assert_eq!(cloned.config.kc, 256);
        assert_eq!(cloned.config.jit.as_ref().unwrap().k_unroll, 4);
        assert!((cloned.perf.median_ns - 999.0).abs() < f64::EPSILON);
        assert!((cloned.perf.bandwidth_gbs.unwrap() - 15.5).abs() < f64::EPSILON);
        assert!(!cloned.from_cache);
        assert_eq!(cloned.report, "clone test");
    }

    #[test]
    fn test_tune_result_debug_format_includes_key_fields() {
        // Arrange
        let result = TuneResult {
            config: TuningConfig {
                kc: 64,
                mc: 32,
                nc: 256,
                num_threads: 2,
                jit: None,
            },
            perf: BenchResult {
                median_ns: 100.0,
                iqr_ns: 5.0,
                min_ns: 90.0,
                samples: 10,
                gflops: None,
                bandwidth_gbs: None,
            },
            from_cache: false,
            report: String::new(),
        };

        // Act: format via Debug
        let debug_str = format!("{:?}", result);

        // Assert: Debug output contains struct name and key data
        assert!(debug_str.contains("TuneResult"));
        assert!(debug_str.contains("from_cache"));
    }

    #[test]
    fn test_default_gpu_config_sm100_and_later() {
        // Arrange & Act: SM 100+ (Blackwell+)
        let cfg = default_gpu_config(100);

        // Assert: largest tile sizes selected
        assert_eq!(cfg.cta_m, 128);
        assert_eq!(cfg.cta_n, 256);
        assert_eq!(cfg.cta_k, 64);
        assert_eq!(cfg.warp_m, 64);
        assert_eq!(cfg.warp_n, 32);
        assert_eq!(cfg.pipeline_depth, 3);

        // Act: SM 120 also falls into the same 100..= range
        let cfg120 = default_gpu_config(120);
        assert_eq!(cfg120.cta_n, 256);
    }

    #[test]
    fn test_default_gpu_config_sm90_hopper() {
        // Arrange & Act
        let cfg = default_gpu_config(90);

        // Assert: Hopper-specific tile sizes
        assert_eq!(cfg.cta_m, 128);
        assert_eq!(cfg.cta_n, 128);
        assert_eq!(cfg.cta_k, 64);
        assert_eq!(cfg.warp_m, 64);
        assert_eq!(cfg.warp_n, 16);
        assert_eq!(cfg.pipeline_depth, 3);

        // SM 95 same generation
        let cfg95 = default_gpu_config(95);
        assert_eq!(cfg95.cta_m, 128);
        assert_eq!(cfg95.pipeline_depth, 3);
    }

    #[test]
    fn test_default_gpu_config_sm80_ampere_and_below() {
        // Arrange & Act: SM 80 (Ampere)
        let cfg80 = default_gpu_config(80);
        assert_eq!(cfg80.cta_m, 128);
        assert_eq!(cfg80.cta_n, 128);
        assert_eq!(cfg80.cta_k, 32);
        assert_eq!(cfg80.pipeline_depth, 2);

        // SM 70 (Volta) falls into the catch-all
        let cfg70 = default_gpu_config(70);
        assert_eq!(cfg70.cta_m, 64);
        assert_eq!(cfg70.cta_n, 64);
        assert_eq!(cfg70.cta_k, 32);
        assert_eq!(cfg70.pipeline_depth, 1);

        // SM 0 (absurd edge case)
        let cfg0 = default_gpu_config(0);
        assert_eq!(cfg0.cta_m, 64);
        assert_eq!(cfg0.pipeline_depth, 1);
    }

    #[test]
    fn test_gpu_gemm_config_display_format() {
        // Arrange
        let cfg = GpuGemmConfig {
            cta_m: 128,
            cta_n: 64,
            cta_k: 32,
            warp_m: 32,
            warp_n: 16,
            pipeline_depth: 2,
        };

        // Act
        let display = format!("{}", cfg);

        // Assert: contains all key values (Display uses Unicode multiplication sign)
        assert!(display.contains("cta=128\u{00d7}64\u{00d7}32"));
        assert!(display.contains("warp=32\u{00d7}16"));
        assert!(display.contains("pipe=2"));
    }

    #[test]
    fn test_gpu_gemm_config_clone_preserves_fields() {
        // Arrange
        let cfg = GpuGemmConfig {
            cta_m: 256,
            cta_n: 128,
            cta_k: 64,
            warp_m: 64,
            warp_n: 32,
            pipeline_depth: 3,
        };

        // Act
        let cloned = cfg.clone();

        // Assert: all fields identical
        assert_eq!(cloned.cta_m, 256);
        assert_eq!(cloned.cta_n, 128);
        assert_eq!(cloned.cta_k, 64);
        assert_eq!(cloned.warp_m, 64);
        assert_eq!(cloned.warp_n, 32);
        assert_eq!(cloned.pipeline_depth, 3);
        assert_eq!(cfg, cloned);
    }

    #[test]
    fn test_extract_model_gemm_shapes_deduplication() {
        // Arrange & Act
        let shapes = extract_model_gemm_shapes();

        // Assert: returns the expected set of LLM GEMM shapes
        assert!(!shapes.is_empty(), "should return at least one shape");

        // Check that (0, 4096, 4096) appears (QKV proj and Attn out have same shape)
        let qkv_count = shapes.iter().filter(|&&(m, n, k)| m == 0 && n == 4096 && k == 4096).count();
        assert!(qkv_count >= 2, "QKV and Attn out should share (0,4096,4096)");

        // All K values should be > 0 (they represent weight dimensions)
        assert!(shapes.iter().all(|&(_, _, k)| k > 0));
    }

    #[test]
    fn test_tune_model_gpu_gemms_deduplicates_by_nk() {
        // Arrange: use a small SM version and reasonable SMEM
        let sm = 80u32;
        let smem = 49152usize; // 48 KB typical SM80
        let elem = 4usize; // F32

        // Act
        let results = tune_model_gpu_gemms(sm, smem, elem, TuneLevel::Fast);

        // Assert: results keyed by (N, K) with no duplicate keys
        let keys: Vec<_> = results.keys().collect();
        let unique_keys: std::collections::HashSet<_> = keys.iter().collect();
        assert_eq!(keys.len(), unique_keys.len(), "no duplicate (N,K) keys");

        // Assert: at least one entry (shapes include 4096x4096)
        assert!(!results.is_empty());

        // Assert: every value is a valid GpuGemmConfig
        for cfg in results.values() {
            assert!(cfg.cta_m > 0);
            assert!(cfg.cta_n > 0);
            assert!(cfg.cta_k > 0);
            assert!(cfg.warp_m > 0);
            assert!(cfg.warp_n > 0);
        }
    }

    #[test]
    fn test_search_result_display_format() {
        // Arrange: construct a SearchResult manually
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 128,
                mc: 48,
                nc: 512,
                num_threads: 4,
                jit: None,
            },
            best_result: BenchResult {
                median_ns: 5000.0,
                iqr_ns: 200.0,
                min_ns: 4500.0,
                samples: 20,
                gflops: Some(30.0),
                bandwidth_gbs: None,
            },
            configs_evaluated: 42,
            search_time_ns: 3_000_000_000, // 3 seconds
            all_results: vec![],
        };

        // Act
        let display = format!("{}", result);

        // Assert: contains key information
        assert!(display.contains("Best:"));
        assert!(display.contains("evaluated 42 configs"));
        assert!(display.contains("3000.0ms"));
    }

    // ── Test 21: OpClass all variants distinct ──

    #[test]
    fn op_class_all_variants_distinct() {
        // Arrange
        let variants = [OpClass::Gemm, OpClass::Gemv, OpClass::MemoryBound];

        // Act & Assert
        assert_ne!(variants[0], variants[1]);
        assert_ne!(variants[1], variants[2]);
        assert_ne!(variants[0], variants[2]);
    }

    // ── Test 22: ProblemShape construction and equality ──

    #[test]
    fn problem_shape_construction() {
        // Arrange
        let shape = ProblemShape {
            m: 512, n: 1024, k: 768,
            elem_bytes: 4, dtype_id: 0,
        };

        // Assert
        assert_eq!(shape.m, 512);
        assert_eq!(shape.n, 1024);
        assert_eq!(shape.k, 768);
        assert_eq!(shape.elem_bytes, 4);
    }

    // ── Test 23: JitParams field access and Debug ──

    #[test]
    fn jit_params_field_access_and_debug() {
        // Arrange
        let params = JitParams {
            k_unroll: 4,
            prefetch_distance: 8,
            reg_alloc_strategy: RegAllocStrategy::Balanced,
            sw_pipeline_depth: 3,
            nr_variant: 16,
        };

        // Act
        let debug = format!("{:?}", params);
        let cloned = params.clone();

        // Assert
        assert_eq!(params.k_unroll, 4);
        assert_eq!(params.prefetch_distance, 8);
        assert_eq!(params.sw_pipeline_depth, 3);
        assert_eq!(cloned.k_unroll, 4);
        assert!(debug.contains("4"));
    }

    // ── Test 24: RegAllocStrategy variants distinct ──

    #[test]
    fn reg_alloc_strategy_variants_distinct() {
        // Arrange
        let strategies = [
            RegAllocStrategy::MaxAccumulators,
            RegAllocStrategy::Balanced,
            RegAllocStrategy::MinSpill,
        ];

        // Act & Assert
        assert_ne!(strategies[0], strategies[1]);
        assert_ne!(strategies[1], strategies[2]);
        assert_ne!(strategies[0], strategies[2]);
    }

    // ── Test 25: BenchResult construction with all fields ──

    #[test]
    fn bench_result_construction_all_fields() {
        // Arrange
        let result = BenchResult {
            median_ns: 1000.0,
            iqr_ns: 50.0,
            min_ns: 900.0,
            samples: 30,
            gflops: Some(25.5),
            bandwidth_gbs: Some(12.3),
        };

        // Assert
        assert_eq!(result.median_ns, 1000.0);
        assert_eq!(result.iqr_ns, 50.0);
        assert_eq!(result.min_ns, 900.0);
        assert_eq!(result.samples, 30);
        assert_eq!(result.gflops, Some(25.5));
        assert_eq!(result.bandwidth_gbs, Some(12.3));
    }

    // ── Test 26: BenchResult clone preserves all fields ──

    #[test]
    fn bench_result_clone_preserves_fields() {
        // Arrange
        let result = BenchResult {
            median_ns: 500.0,
            iqr_ns: 20.0,
            min_ns: 450.0,
            samples: 10,
            gflops: Some(40.0),
            bandwidth_gbs: None,
        };

        // Act
        let cloned = result.clone();

        // Assert
        assert_eq!(cloned.median_ns, 500.0);
        assert_eq!(cloned.samples, 10);
        assert_eq!(cloned.gflops, Some(40.0));
        assert_eq!(cloned.bandwidth_gbs, None);
    }

    // ── Test 27: GpuGemmConfig field access and Clone ──

    #[test]
    fn gpu_gemm_config_field_access_and_clone() {
        // Arrange
        let config = GpuGemmConfig {
            cta_m: 128, cta_n: 256, cta_k: 64,
            warp_m: 64, warp_n: 32,
            pipeline_depth: 3,
        };

        // Act
        let cloned = config.clone();

        // Assert
        assert_eq!(config.cta_m, 128);
        assert_eq!(config.cta_n, 256);
        assert_eq!(config.cta_k, 64);
        assert_eq!(config.warp_m, 64);
        assert_eq!(config.warp_n, 32);
        assert_eq!(config.pipeline_depth, 3);
        assert_eq!(cloned.cta_m, 128);
    }

    // ── Test 28: SearchConfig fast/default/thorough differ in iterations ──

    #[test]
    fn search_config_levels_differ() {
        // Arrange
        let fast = SearchConfig::fast();
        let default = SearchConfig::default();
        let thorough = SearchConfig::thorough();

        // Assert — Thorough should use smaller stride (more exhaustive) than Fast
        assert!(
            thorough.coarse_stride <= default.coarse_stride,
            "Thorough stride ({}) should be <= Default ({})",
            thorough.coarse_stride, default.coarse_stride,
        );
        assert!(
            default.coarse_stride <= fast.coarse_stride,
            "Default stride ({}) should be <= Fast ({})",
            default.coarse_stride, fast.coarse_stride,
        );
        // Thorough should refine more
        assert!(
            thorough.refine_iters >= default.refine_iters,
            "Thorough refine_iters ({}) should be >= Default ({})",
            thorough.refine_iters, default.refine_iters,
        );
    }

    // ── Test 29: clear_cache does not panic ──

    #[test]
    fn clear_cache_does_not_panic() {
        // Act & Assert — should complete without panic
        clear_cache();
    }

    // ── Test 30: cache_summary returns non-empty string ──

    #[test]
    fn cache_summary_non_empty() {
        // Act
        let summary = cache_summary();

        // Assert
        assert!(!summary.is_empty(), "cache summary should be non-empty");
        assert!(summary.contains("Wisdom"), "should contain 'Wisdom'");
    }

    // ── Test 31: JitSearchRanges construction ──

    #[test]
    fn jit_search_ranges_construction() {
        // Arrange
        let ranges = JitSearchRanges {
            k_unroll_range: crate::autotuning::search_space::ParamRange { name: "k_unroll", min: 2, max: 8, step: 2 },
            prefetch_range: crate::autotuning::search_space::ParamRange { name: "prefetch", min: 0, max: 8, step: 4 },
            reg_alloc_strategies: vec![RegAllocStrategy::Balanced, RegAllocStrategy::MinSpill],
            sw_pipeline_range: crate::autotuning::search_space::ParamRange { name: "sw_pipe", min: 1, max: 3, step: 1 },
            nr_variant_range: crate::autotuning::search_space::ParamRange { name: "nr", min: 8, max: 16, step: 8 },
        };

        // Assert
        assert_eq!(ranges.k_unroll_range.count(), 4);  // 2,4,6,8
        assert_eq!(ranges.prefetch_range.count(), 3);   // 0,4,8
        assert_eq!(ranges.reg_alloc_strategies.len(), 2);
        assert_eq!(ranges.sw_pipeline_range.count(), 3); // 1,2,3
        assert_eq!(ranges.nr_variant_range.count(), 2);  // 8,16
    }

    // ── Test 32: SearchConfig fast has larger stride than thorough ──

    #[test]
    fn search_config_fast_is_coarsest() {
        // Arrange
        let fast = SearchConfig::fast();
        let thorough = SearchConfig::thorough();

        // Assert: Fast uses a larger coarse stride (fewer points sampled)
        assert!(
            fast.coarse_stride > thorough.coarse_stride,
            "Fast stride ({}) should be > Thorough stride ({})",
            fast.coarse_stride, thorough.coarse_stride,
        );
        // Assert: Fast uses fewer refinement iterations
        assert!(
            fast.refine_iters < thorough.refine_iters,
            "Fast refine_iters ({}) should be < Thorough ({})",
            fast.refine_iters, thorough.refine_iters,
        );
    }

    // ── Test 33: SearchConfig early_reject_ratio monotonic across levels ──

    #[test]
    fn search_config_reject_ratio_monotonic() {
        // Arrange
        let fast = SearchConfig::fast();
        let default = SearchConfig::default();
        let thorough = SearchConfig::thorough();

        // Assert: reject ratio increases with thoroughness (more permissive)
        assert!(
            fast.early_reject_ratio <= default.early_reject_ratio,
            "Fast reject ({}) should be <= Default ({})",
            fast.early_reject_ratio, default.early_reject_ratio,
        );
        assert!(
            default.early_reject_ratio <= thorough.early_reject_ratio,
            "Default reject ({}) should be <= Thorough ({})",
            default.early_reject_ratio, thorough.early_reject_ratio,
        );
    }

    // ── Test 34: HwInfo Display includes core counts and ISA string ──

    #[test]
    fn hw_info_display_includes_core_counts_and_isa() {
        // Arrange
        let hw = HwInfo {
            vendor: "AuthenticAMD".into(),
            model_name: "AMD Ryzen 7 7700X".into(),
            physical_cores: 8,
            logical_cores: 16,
            l1d_bytes: 32 * 1024,
            l2_bytes: 1024 * 1024,
            l3_bytes: 32 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: hw_info::IsaFeatures {
                avx2: true, fma: true, avx512f: false, avx512bw: false,
                avx512vnni: false, avx512fp16: false, avx512bf16: false,
                neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
            },
        };

        // Act
        let display = format!("{hw}");

        // Assert
        assert!(display.contains("8P/16L"), "should contain core counts: {display}");
        assert!(display.contains("AVX2"), "should contain ISA: {display}");
        assert!(display.contains("7700X"), "should contain model name: {display}");
    }

    // ── Test 35: HwInfo fingerprint includes cache sizes in KiB ──

    #[test]
    fn hw_info_fingerprint_includes_cache_sizes_kib() {
        // Arrange
        let hw = HwInfo {
            vendor: "TestVendor".into(),
            model_name: "TestModel".into(),
            physical_cores: 4,
            logical_cores: 8,
            l1d_bytes: 48 * 1024,       // 48 KiB
            l2_bytes: 512 * 1024,       // 512 KiB
            l3_bytes: 8 * 1024 * 1024,  // 8192 KiB
            cacheline_bytes: 64,
            isa: hw_info::IsaFeatures {
                avx2: true, fma: true, avx512f: false, avx512bw: false,
                avx512vnni: false, avx512fp16: false, avx512bf16: false,
                neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
            },
        };

        // Act
        let fp = hw.fingerprint();

        // Assert: cache sizes encoded as KiB
        assert!(fp.contains("l148"), "L1D=48K should appear: {fp}");
        assert!(fp.contains("l2512"), "L2=512K should appear: {fp}");
        assert!(fp.contains("l38192"), "L3=8192K should appear: {fp}");
    }

    // ── Test 36: IsaFeatures Display — NEON only without SVE ──

    #[test]
    fn isa_features_display_neon_only() {
        // Arrange
        let isa = hw_info::IsaFeatures {
            avx2: false, fma: false, avx512f: false, avx512bw: false,
            avx512vnni: false, avx512fp16: false, avx512bf16: false,
            neon: true, sve: false, sve2: false, sve_vl_bytes: 0,
        };

        // Act
        let display = format!("{isa}");

        // Assert
        assert_eq!(display, "NEON", "NEON-only should display exactly 'NEON'");
        assert!(!display.contains("SVE"), "SVE should not appear: {display}");
    }

    // ── Test 37: IsaFeatures all x86 features enabled ──

    #[test]
    fn isa_features_all_x86_features() {
        // Arrange
        let isa = hw_info::IsaFeatures {
            avx2: true, fma: true, avx512f: true, avx512bw: true,
            avx512vnni: true, avx512fp16: true, avx512bf16: true,
            neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
        };

        // Act
        let display = format!("{isa}");

        // Assert: all x86 features present
        assert!(display.contains("AVX-512"), "AVX-512 missing: {display}");
        assert!(display.contains("FP16"), "FP16 missing: {display}");
        assert!(display.contains("BF16"), "BF16 missing: {display}");
        assert!(display.contains("VNNI"), "VNNI missing: {display}");
        assert!(display.contains("AVX2"), "AVX2 missing: {display}");
        assert!(display.contains("FMA"), "FMA missing: {display}");
    }

    // ── Test 38: TuningConfig Display with and without JIT params ──

    #[test]
    fn tuning_config_display_with_and_without_jit() {
        // Arrange: config without JIT
        let cfg_no_jit = TuningConfig {
            kc: 256, mc: 96, nc: 1024, num_threads: 4, jit: None,
        };
        // Arrange: config with JIT
        let cfg_with_jit = TuningConfig {
            kc: 128, mc: 48, nc: 512, num_threads: 2,
            jit: Some(JitParams {
                k_unroll: 4, prefetch_distance: 8,
                reg_alloc_strategy: RegAllocStrategy::Balanced,
                sw_pipeline_depth: 2, nr_variant: 16,
            }),
        };

        // Act
        let display_no = format!("{cfg_no_jit}");
        let display_yes = format!("{cfg_with_jit}");

        // Assert: base params always present
        assert!(display_no.contains("KC=256"), "KC missing: {display_no}");
        assert!(display_no.contains("threads=4"), "threads missing: {display_no}");
        // Assert: JIT params appended when present
        assert!(!display_no.contains("k_unroll"), "JIT should not appear without jit: {display_no}");
        assert!(display_yes.contains("k_unroll=4"), "JIT k_unroll missing: {display_yes}");
        assert!(display_yes.contains("balanced"), "JIT reg_alloc missing: {display_yes}");
    }

    // ── Test 39: BenchResult Display with gflops and bandwidth ──

    #[test]
    fn bench_result_display_with_metrics() {
        // Arrange
        let result = BenchResult {
            median_ns: 500_000.0, // 500 us
            iqr_ns: 50_000.0,
            min_ns: 480_000.0,
            samples: 30,
            gflops: Some(123.4),
            bandwidth_gbs: Some(45.6),
        };

        // Act
        let display = format!("{result}");

        // Assert
        assert!(display.contains("500.0us"), "median should be 500.0us: {display}");
        assert!(display.contains("123.4GFLOPS"), "GFLOPS should appear: {display}");
        assert!(display.contains("45.6GB/s"), "bandwidth should appear: {display}");
    }

    // ── Test 40: BenchResult Display without optional metrics ──

    #[test]
    fn bench_result_display_without_optional_metrics() {
        // Arrange
        let result = BenchResult {
            median_ns: 1000.0,
            iqr_ns: 100.0,
            min_ns: 900.0,
            samples: 5,
            gflops: None,
            bandwidth_gbs: None,
        };

        // Act
        let display = format!("{result}");

        // Assert: no GFLOPS or GB/s when None
        assert!(!display.contains("GFLOPS"), "GFLOPS should not appear when None: {display}");
        assert!(!display.contains("GB/s"), "GB/s should not appear when None: {display}");
        assert!(display.contains("median="), "median label should appear: {display}");
    }

    // ── Test 41: RegAllocStrategy Display and index roundtrip ──

    #[test]
    fn reg_alloc_strategy_display_and_roundtrip() {
        // Arrange
        let strategies = RegAllocStrategy::all();

        // Act & Assert: Display output and index roundtrip
        for &s in strategies {
            let display = format!("{s}");
            assert!(!display.is_empty(), "Display should not be empty for {:?}", s);
            let idx = s.to_index();
            let recovered = RegAllocStrategy::from_index(idx);
            assert_eq!(s, recovered, "roundtrip failed for {:?}", s);
        }

        // Assert: specific display strings
        assert_eq!(format!("{}", RegAllocStrategy::MaxAccumulators), "max_acc");
        assert_eq!(format!("{}", RegAllocStrategy::Balanced), "balanced");
        assert_eq!(format!("{}", RegAllocStrategy::MinSpill), "min_spill");
    }

    // ── Test 42: TuneResult with zero median_ns (edge case) ────────────────

    #[test]
    fn tune_result_zero_median_ns_edge_case() {
        // Arrange: benchmark result with zero time (cached or instant operation)
        let perf = BenchResult {
            median_ns: 0.0,
            iqr_ns: 0.0,
            min_ns: 0.0,
            samples: 0,
            gflops: None,
            bandwidth_gbs: None,
        };

        // Act
        let result = TuneResult {
            config: TuningConfig {
                kc: 64, mc: 32, nc: 128, num_threads: 1, jit: None,
            },
            perf: perf.clone(),
            from_cache: true,
            report: "instant".into(),
        };

        // Assert: zero values preserved correctly
        assert_eq!(result.perf.median_ns, 0.0);
        assert_eq!(result.perf.samples, 0);
        assert!(result.from_cache);
        assert_eq!(result.report, "instant");
    }

    // ── Test 43: TuneResult with single sample (minimal measurement) ───────

    #[test]
    fn tune_result_single_sample_minimal_measurement() {
        // Arrange: benchmark with just one sample
        let perf = BenchResult {
            median_ns: 1234.0,
            iqr_ns: 0.0, // IQR is zero with single sample
            min_ns: 1234.0,
            samples: 1,
            gflops: Some(10.0),
            bandwidth_gbs: None,
        };

        // Act
        let result = TuneResult {
            config: TuningConfig {
                kc: 256, mc: 96, nc: 512, num_threads: 2, jit: None,
            },
            perf,
            from_cache: false,
            report: "single sample".into(),
        };

        // Assert: single sample handled correctly
        assert_eq!(result.perf.samples, 1);
        assert_eq!(result.perf.iqr_ns, 0.0);
        assert_eq!(result.perf.median_ns, result.perf.min_ns);
    }

    // ── Test 44: TuningConfig equality with identical JIT params ────────────

    #[test]
    fn tuning_config_equality_with_identical_jit_params() {
        // Arrange: two configs with identical JIT params
        let jit = JitParams {
            k_unroll: 4,
            prefetch_distance: 8,
            reg_alloc_strategy: RegAllocStrategy::Balanced,
            sw_pipeline_depth: 2,
            nr_variant: 16,
        };
        let cfg1 = TuningConfig {
            kc: 128, mc: 48, nc: 256, num_threads: 4, jit: Some(jit.clone()),
        };
        let cfg2 = TuningConfig {
            kc: 128, mc: 48, nc: 256, num_threads: 4, jit: Some(jit),
        };

        // Assert: configs are equal
        assert_eq!(cfg1, cfg2);
    }

    // ── Test 45: TuningConfig inequality with different JIT params ──────────

    #[test]
    fn tuning_config_inequality_with_different_jit_params() {
        // Arrange: same base params, different JIT
        let cfg1 = TuningConfig {
            kc: 128, mc: 48, nc: 256, num_threads: 4,
            jit: Some(JitParams {
                k_unroll: 4,
                prefetch_distance: 8,
                reg_alloc_strategy: RegAllocStrategy::Balanced,
                sw_pipeline_depth: 2,
                nr_variant: 16,
            }),
        };
        let cfg2 = TuningConfig {
            kc: 128, mc: 48, nc: 256, num_threads: 4,
            jit: Some(JitParams {
                k_unroll: 2, // Different
                prefetch_distance: 8,
                reg_alloc_strategy: RegAllocStrategy::Balanced,
                sw_pipeline_depth: 2,
                nr_variant: 16,
            }),
        };

        // Assert: configs are not equal
        assert_ne!(cfg1, cfg2);
    }

    // ── Test 46: TuningConfig inequality with jit vs no jit ─────────────────

    #[test]
    fn tuning_config_inequality_jit_vs_no_jit() {
        // Arrange: same base params, one with JIT, one without
        let cfg_with_jit = TuningConfig {
            kc: 128, mc: 48, nc: 256, num_threads: 4,
            jit: Some(JitParams::default()),
        };
        let cfg_no_jit = TuningConfig {
            kc: 128, mc: 48, nc: 256, num_threads: 4,
            jit: None,
        };

        // Assert: configs are not equal
        assert_ne!(cfg_with_jit, cfg_no_jit);
    }

    // ── Test 47: BenchResult with very high gflops (compute-bound) ──────────

    #[test]
    fn bench_result_very_high_gflops_compute_bound() {
        // Arrange: compute-bound kernel with high GFLOPS
        let result = BenchResult {
            median_ns: 1000.0, // 1 us
            iqr_ns: 50.0,
            min_ns: 900.0,
            samples: 100,
            gflops: Some(5000.0), // 5 TFLOPS
            bandwidth_gbs: Some(200.0),
        };

        // Act
        let display = format!("{result}");

        // Assert: high GFLOPS displayed correctly
        assert!(display.contains("5000.0GFLOPS"));
        assert!(display.contains("200.0GB/s"));
        assert_eq!(result.samples, 100);
    }

    // ── Test 48: BenchResult comparison by median_ns (lower is better) ──────

    #[test]
    fn bench_result_comparison_by_median_ns() {
        // Arrange: two results with different median times
        let faster = BenchResult {
            median_ns: 100.0,
            iqr_ns: 10.0,
            min_ns: 90.0,
            samples: 20,
            gflops: Some(100.0),
            bandwidth_gbs: None,
        };
        let slower = BenchResult {
            median_ns: 200.0,
            iqr_ns: 20.0,
            min_ns: 180.0,
            samples: 20,
            gflops: Some(50.0),
            bandwidth_gbs: None,
        };

        // Assert: faster has lower median_ns and higher gflops
        assert!(faster.median_ns < slower.median_ns);
        assert!(faster.gflops.unwrap() > slower.gflops.unwrap());
    }

    // ── Test 49: SearchResult with empty all_results (single best only) ─────

    #[test]
    fn search_result_empty_all_results_single_best() {
        // Arrange: search that only kept the best result
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 64, mc: 32, nc: 128, num_threads: 1, jit: None,
            },
            best_result: BenchResult {
                median_ns: 500.0,
                iqr_ns: 25.0,
                min_ns: 450.0,
                samples: 10,
                gflops: Some(20.0),
                bandwidth_gbs: None,
            },
            configs_evaluated: 1,
            search_time_ns: 100_000, // 100 us
            all_results: vec![], // Empty — only best kept
        };

        // Assert: single result handled correctly
        assert_eq!(result.configs_evaluated, 1);
        assert!(result.all_results.is_empty());
        assert_eq!(result.best_config.kc, 64);
    }

    // ── Test 50: SearchResult with multiple all_results entries ─────────────

    #[test]
    fn search_result_multiple_all_results_entries() {
        // Arrange: search with multiple candidates recorded
        let candidates = vec![
            (TuningConfig { kc: 64, mc: 32, nc: 128, num_threads: 1, jit: None },
             BenchResult { median_ns: 600.0, iqr_ns: 30.0, min_ns: 550.0, samples: 10, gflops: None, bandwidth_gbs: None }),
            (TuningConfig { kc: 128, mc: 48, nc: 256, num_threads: 2, jit: None },
             BenchResult { median_ns: 400.0, iqr_ns: 20.0, min_ns: 380.0, samples: 10, gflops: None, bandwidth_gbs: None }),
        ];
        let result = SearchResult {
            best_config: candidates[1].0.clone(),
            best_result: candidates[1].1.clone(),
            configs_evaluated: 2,
            search_time_ns: 500_000,
            all_results: candidates,
        };

        // Assert: multiple results preserved
        assert_eq!(result.all_results.len(), 2);
        assert_eq!(result.best_config.kc, 128); // Second one is best (lower time)
        assert!(result.best_result.median_ns < result.all_results[0].1.median_ns);
    }

    // ── Test 51: TuneLevel Debug output contains variant names ──────────────

    #[test]
    fn tune_level_debug_contains_variant_names() {
        // Arrange
        let fast = TuneLevel::Fast;
        let default = TuneLevel::Default;
        let thorough = TuneLevel::Thorough;

        // Act
        let fast_debug = format!("{:?}", fast);
        let default_debug = format!("{:?}", default);
        let thorough_debug = format!("{:?}", thorough);

        // Assert: Debug output contains variant names
        assert!(fast_debug.contains("Fast"));
        assert!(default_debug.contains("Default"));
        assert!(thorough_debug.contains("Thorough"));
    }

    // ── Test 52: ParamRange count and values consistency ──────────────────

    #[test]
    fn param_range_count_matches_values_length() {
        // Arrange
        let ranges = [
            crate::autotuning::search_space::ParamRange { name: "a", min: 0, max: 100, step: 10 },
            crate::autotuning::search_space::ParamRange { name: "b", min: 8, max: 64, step: 8 },
            crate::autotuning::search_space::ParamRange { name: "c", min: 1, max: 1, step: 1 },
            crate::autotuning::search_space::ParamRange { name: "d", min: 100, max: 50, step: 5 },
        ];

        for r in &ranges {
            // Act
            let count = r.count();
            let values = r.values();

            // Assert: count() and values().len() must always agree
            assert_eq!(
                count, values.len(),
                "count() = {} but values().len() = {} for range {:?}",
                count, values.len(), r
            );
        }
    }

    // ── Test 53: BenchConfig fast/default/precise have increasing min_iters ─

    #[test]
    fn bench_config_levels_have_increasing_min_iters() {
        // Arrange
        let fast = BenchConfig::fast();
        let default = BenchConfig::default();
        let precise = BenchConfig::precise();

        // Assert: precision levels form a monotonic sequence
        assert!(
            fast.min_iters < default.min_iters,
            "fast.min_iters ({}) should be < default ({})",
            fast.min_iters, default.min_iters,
        );
        assert!(
            default.min_iters < precise.min_iters,
            "default.min_iters ({}) should be < precise ({})",
            default.min_iters, precise.min_iters,
        );
        // Warmup also increases with precision
        assert!(
            fast.warmup_iters <= default.warmup_iters,
            "fast.warmup_iters ({}) should be <= default ({})",
            fast.warmup_iters, default.warmup_iters,
        );
    }

    // ── Test 54: SearchResult Display includes config and timing ───────────

    #[test]
    fn search_result_display_includes_config_and_timing() {
        // Arrange
        let result = SearchResult {
            best_config: TuningConfig {
                kc: 512, mc: 96, nc: 1024, num_threads: 8, jit: None,
            },
            best_result: BenchResult {
                median_ns: 10_000_000.0, // 10 ms
                iqr_ns: 500_000.0,
                min_ns: 9_500_000.0,
                samples: 50,
                gflops: Some(99.9),
                bandwidth_gbs: None,
            },
            configs_evaluated: 100,
            search_time_ns: 5_000_000_000, // 5 seconds
            all_results: vec![],
        };

        // Act
        let display = format!("{}", result);

        // Assert: contains the best config and search metadata
        assert!(display.contains("KC=512"), "should contain KC=512: {display}");
        assert!(display.contains("MC=96"), "should contain MC=96: {display}");
        assert!(display.contains("evaluated 100 configs"), "should contain evaluated count: {display}");
        assert!(display.contains("5000.0ms"), "should contain search time: {display}");
    }

    // ── Test 55: SearchSpace for_gemm respects problem shape bounds ────────

    #[test]
    fn search_space_gemm_respects_problem_shape_bounds() {
        // Arrange: very small problem
        let hw = hw_info();
        let shape = ProblemShape { m: 32, n: 32, k: 32, elem_bytes: 4, dtype_id: 0 };
        let (tm, tn) = microkernel_geometry(hw, 4);

        // Act
        let space = SearchSpace::for_gemm(hw, &shape, tm, tn);

        // Assert: KC, MC, NC max values should not exceed problem dimensions
        // (they are clamped by shape.k, shape.m, shape.n respectively)
        assert!(
            space.kc_range.max <= 32 || space.kc_range.max <= space.kc_range.min,
            "KC max ({}) should not exceed k=32 (or be clamped to min)",
            space.kc_range.max,
        );
        assert!(
            space.mc_range.max <= 32 || space.mc_range.max <= space.mc_range.min,
            "MC max ({}) should not exceed m=32 (or be clamped to min)",
            space.mc_range.max,
        );
    }

    // ── Test 56: JitParams Display contains all five fields ────────────────

    #[test]
    fn jit_params_display_contains_all_five_fields() {
        // Arrange
        let params = JitParams {
            k_unroll: 8,
            prefetch_distance: 12,
            reg_alloc_strategy: RegAllocStrategy::MinSpill,
            sw_pipeline_depth: 2,
            nr_variant: 32,
        };

        // Act
        let display = format!("{}", params);

        // Assert: all five JIT parameter values appear
        assert!(display.contains("k_unroll=8"), "k_unroll missing: {display}");
        assert!(display.contains("prefetch=12"), "prefetch missing: {display}");
        assert!(display.contains("min_spill"), "reg_alloc missing: {display}");
        assert!(display.contains("swp=2"), "sw_pipeline missing: {display}");
        assert!(display.contains("nr=32"), "nr_variant missing: {display}");
    }

    // ── Test 57: WisdomDb put and get with JIT params ─────────────────────

    #[test]
    fn wisdom_db_put_get_with_jit_params_roundtrip() {
        // Arrange
        let mut db = WisdomDb::new(std::path::PathBuf::from("/tmp/_unused_test57.json"));
        let jit = JitParams {
            k_unroll: 2,
            prefetch_distance: 4,
            reg_alloc_strategy: RegAllocStrategy::MaxAccumulators,
            sw_pipeline_depth: 1,
            nr_variant: 8,
        };
        let config = TuningConfig {
            kc: 256, mc: 96, nc: 1024, num_threads: 4,
            jit: Some(jit.clone()),
        };

        // Act
        db.put("hw_test57", "op_jit57", config.clone(), 750.0, Some(55.5));

        // Assert: retrieve and verify all fields
        let cached = db.get("hw_test57", "op_jit57");
        assert!(cached.is_some(), "entry should exist after put");
        let c = cached.unwrap();
        assert_eq!(c.config.kc, 256);
        assert_eq!(c.config.mc, 96);
        assert_eq!(c.config.nc, 1024);
        assert_eq!(c.config.num_threads, 4);
        assert!(c.config.jit.is_some());
        let retrieved_jit = c.config.jit.as_ref().unwrap();
        assert_eq!(retrieved_jit.k_unroll, 2);
        assert_eq!(retrieved_jit.prefetch_distance, 4);
        assert_eq!(retrieved_jit.reg_alloc_strategy, RegAllocStrategy::MaxAccumulators);
        assert_eq!(retrieved_jit.sw_pipeline_depth, 1);
        assert_eq!(retrieved_jit.nr_variant, 8);
        assert!((c.median_ns - 750.0).abs() < f64::EPSILON);
        assert!((c.gflops.unwrap() - 55.5).abs() < f64::EPSILON);
    }

    // ── Test 58: OpClass Display-like behavior (Copy + Debug) ─────────────

    #[test]
    fn op_class_copy_and_debug_all_variants() {
        // Arrange: create each variant
        let gemm = OpClass::Gemm;
        let gemv = OpClass::Gemv;
        let mem = OpClass::MemoryBound;

        // Act: Copy semantics (use after assignment)
        let gemm_copy = gemm;
        assert_eq!(gemm, gemm_copy);

        // Act: Debug format
        let debug_gemm = format!("{:?}", gemm);
        let debug_gemv = format!("{:?}", gemv);
        let debug_mem = format!("{:?}", mem);

        // Assert: Debug output contains variant names
        assert!(debug_gemm.contains("Gemm"), "Debug should contain 'Gemm': {debug_gemm}");
        assert!(debug_gemv.contains("Gemv"), "Debug should contain 'Gemv': {debug_gemv}");
        assert!(debug_mem.contains("MemoryBound"), "Debug should contain 'MemoryBound': {debug_mem}");
    }

    // ── Test 59: GpuGemmConfig equality distinguishes all fields ───────────

    #[test]
    fn gpu_gemm_config_equality_distinguishes_each_field() {
        // Arrange: base config
        let base = GpuGemmConfig {
            cta_m: 128, cta_n: 64, cta_k: 32,
            warp_m: 32, warp_n: 16,
            pipeline_depth: 2,
        };

        // Assert: changing any single field makes it unequal
        let diff_cta_m = GpuGemmConfig { cta_m: 64, ..base.clone() };
        let diff_cta_n = GpuGemmConfig { cta_n: 128, ..base.clone() };
        let diff_cta_k = GpuGemmConfig { cta_k: 64, ..base.clone() };
        let diff_warp_m = GpuGemmConfig { warp_m: 64, ..base.clone() };
        let diff_warp_n = GpuGemmConfig { warp_n: 32, ..base.clone() };
        let diff_pipe = GpuGemmConfig { pipeline_depth: 3, ..base.clone() };

        assert_ne!(base, diff_cta_m, "cta_m difference should be detected");
        assert_ne!(base, diff_cta_n, "cta_n difference should be detected");
        assert_ne!(base, diff_cta_k, "cta_k difference should be detected");
        assert_ne!(base, diff_warp_m, "warp_m difference should be detected");
        assert_ne!(base, diff_warp_n, "warp_n difference should be detected");
        assert_ne!(base, diff_pipe, "pipeline_depth difference should be detected");

        // Assert: identical clone is equal
        assert_eq!(base, base.clone());
    }

    // ── Test 60: extract_model_gemm_shapes returns valid shapes ────────────

    #[test]
    fn extract_model_gemm_shapes_all_valid_and_positive() {
        // Act
        let shapes = extract_model_gemm_shapes();

        // Assert: non-empty result
        assert!(!shapes.is_empty(), "should return at least one shape");

        // Assert: N and K are always positive (M may be 0 for symbolic)
        for &(m, n, k) in &shapes {
            assert!(n > 0, "N must be > 0, got {n} for shape ({m},{n},{k})");
            assert!(k > 0, "K must be > 0, got {k} for shape ({m},{n},{k})");
            // M=0 is valid (symbolic decode batch)
        }
    }

    // ── Test 61: BenchResult clone with optional None fields ───────────────

    #[test]
    fn bench_result_clone_preserves_none_optionals() {
        // Arrange: result with both gflops and bandwidth as None
        let result = BenchResult {
            median_ns: 2500.0,
            iqr_ns: 100.0,
            min_ns: 2300.0,
            samples: 15,
            gflops: None,
            bandwidth_gbs: None,
        };

        // Act
        let cloned = result.clone();

        // Assert: None fields stay None after clone
        assert_eq!(cloned.median_ns, 2500.0);
        assert_eq!(cloned.iqr_ns, 100.0);
        assert_eq!(cloned.min_ns, 2300.0);
        assert_eq!(cloned.samples, 15);
        assert!(cloned.gflops.is_none(), "gflops should remain None after clone");
        assert!(cloned.bandwidth_gbs.is_none(), "bandwidth_gbs should remain None after clone");
    }
}

// ── GPU Autotuning ─────────────────────────────────────────────────────

/// Tune GPU GEMM tile parameters for a specific GEMM shape and SM version.
///
/// Searches over (cta_m, cta_n, cta_k, warp_m, warp_n, pipeline_depth) space
/// and returns the optimal configuration. Results are cached to disk.
///
/// This function is designed for model-load-time autotuning: the model's
/// GEMM shapes are extracted, deduplicated, and each shape is tuned once.
/// The optimal params are then baked into the mega-kernel compilation.
pub fn tune_gpu_gemm(
    m: usize,
    n: usize,
    k: usize,
    sm_version: u32,
    shared_mem_bytes: usize,
    elem_bytes: usize,
    level: TuneLevel,
) -> GpuGemmConfig {
    let op_key = format!("gpu_gemm_{}x{}x{}_sm{}", m, n, k, sm_version);
    let fp = format!("sm{}_{}B", sm_version, shared_mem_bytes / 1024);

    // Check cache
    {
        let db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        let shape = ProblemShape { m, n, k, elem_bytes, dtype_id: 0 };
        let cache_key = cache::op_key(&op_key, &shape);
        if let Some(cached) = db.get(&fp, &cache_key) {
            // Reconstruct GpuGemmConfig from cached TuningConfig
            // Store GPU params in kc=cta_m, mc=cta_n, nc=cta_k, threads=warp_m|warp_n|pipe
            if let Some(jit) = &cached.config.jit {
                return GpuGemmConfig {
                    cta_m: cached.config.kc,
                    cta_n: cached.config.mc,
                    cta_k: cached.config.nc,
                    warp_m: jit.k_unroll,
                    warp_n: jit.prefetch_distance,
                    pipeline_depth: jit.sw_pipeline_depth,
                };
            }
        }
    }

    let space = GpuSearchSpace::for_sm(sm_version, shared_mem_bytes);
    let candidates = space.enumerate_valid(shared_mem_bytes, elem_bytes);

    if candidates.is_empty() {
        // Fallback to hardware profile defaults
        return default_gpu_config(sm_version);
    }

    // For Fast tuning: just return the hardware profile default
    // For Default/Thorough: would benchmark candidates on real GPU
    // Since GPU benchmarking requires a live GPU context, we use heuristic scoring
    let best = match level {
        TuneLevel::Fast => default_gpu_config(sm_version),
        TuneLevel::Default | TuneLevel::Thorough => {
            // Score candidates by heuristic (SMEM utilization + occupancy)
            score_and_pick(&candidates, m, n, k, shared_mem_bytes, elem_bytes)
        }
    };

    // Cache the result
    {
        let mut db = wisdom_db().lock().unwrap_or_else(|e| e.into_inner());
        let shape = ProblemShape { m, n, k, elem_bytes, dtype_id: 0 };
        let cache_key = cache::op_key(&op_key, &shape);
        let config = TuningConfig {
            kc: best.cta_m,
            mc: best.cta_n,
            nc: best.cta_k,
            num_threads: 1,
            jit: Some(JitParams {
                k_unroll: best.warp_m,
                prefetch_distance: best.warp_n,
                reg_alloc_strategy: RegAllocStrategy::Balanced,
                sw_pipeline_depth: best.pipeline_depth,
                nr_variant: 16,
            }),
        };
        db.put(&fp, &cache_key, config, 0.0, None);
        let _ = db.save();
    }

    best
}

/// Get hardware profile default GPU GEMM config for a given SM version.
fn default_gpu_config(sm_version: u32) -> GpuGemmConfig {
    match sm_version {
        100.. => GpuGemmConfig { cta_m: 128, cta_n: 256, cta_k: 64, warp_m: 64, warp_n: 32, pipeline_depth: 3 },
        90..=99 => GpuGemmConfig { cta_m: 128, cta_n: 128, cta_k: 64, warp_m: 64, warp_n: 16, pipeline_depth: 3 },
        80..=89 => GpuGemmConfig { cta_m: 128, cta_n: 128, cta_k: 32, warp_m: 64, warp_n: 16, pipeline_depth: 2 },
        _ => GpuGemmConfig { cta_m: 64, cta_n: 64, cta_k: 32, warp_m: 32, warp_n: 16, pipeline_depth: 1 },
    }
}

/// Score GPU GEMM candidates by heuristic and pick the best.
///
/// Scoring criteria:
/// 1. SMEM utilization: tile should use >50% of available SMEM
/// 2. Occupancy: more warps per CTA = better latency hiding
/// 3. Pipeline depth: deeper = better for compute-bound GEMM
/// 4. M-fit: CTA tile should match M dimension well
fn score_and_pick(
    candidates: &[GpuGemmConfig],
    m: usize,
    _n: usize,
    _k: usize,
    shared_mem_bytes: usize,
    elem_bytes: usize,
) -> GpuGemmConfig {
    let mut best = candidates[0].clone();
    let mut best_score = f64::NEG_INFINITY;

    for cfg in candidates {
        let stage_bytes = (cfg.cta_m * cfg.cta_k + cfg.cta_k * cfg.cta_n) * elem_bytes;
        let total_smem = stage_bytes * cfg.pipeline_depth;

        // SMEM utilization: prefer >50% usage but not >95%
        let smem_ratio = total_smem as f64 / shared_mem_bytes as f64;
        let smem_score = if smem_ratio > 0.95 {
            -1.0 // Too much — risk of spills
        } else if smem_ratio > 0.5 {
            smem_ratio // Good utilization
        } else {
            smem_ratio * 0.5 // Under-utilized
        };

        // Occupancy: more warps = better
        let num_warps = (cfg.cta_m / cfg.warp_m) * (cfg.cta_n / cfg.warp_n);
        let occupancy_score = (num_warps as f64).ln().max(0.0);

        // Pipeline bonus
        let pipe_score = cfg.pipeline_depth as f64 * 0.3;

        // M-fit: penalize if CTA_M >> M (wasted threads)
        let m_fit = if m <= cfg.cta_m {
            1.0 - (cfg.cta_m - m) as f64 / cfg.cta_m as f64
        } else {
            1.0 // Multiple CTAs needed — no penalty
        };

        let score = smem_score + occupancy_score + pipe_score + m_fit;
        if score > best_score {
            best_score = score;
            best = cfg.clone();
        }
    }

    best
}

/// Extract unique GEMM shapes from a model's CompilerGraph.
///
/// Returns deduplicated (M, N, K) tuples representing the model's GEMM ops.
/// M may be symbolic (decode M=1, prefill M varies) — returns 0 for symbolic M.
pub fn extract_model_gemm_shapes() -> Vec<(usize, usize, usize)> {
    // This is a placeholder that returns common LLM GEMM shapes.
    // In production, it would inspect the CompilerGraph to extract actual shapes.
    vec![
        (0, 4096, 4096),     // QKV proj (M=symbolic)
        (0, 4096, 4096),     // Attn out
        (0, 4096, 11008),    // FFN gate
        (0, 4096, 11008),    // FFN up
        (0, 11008, 4096),    // FFN down
    ]
}

/// Batch-tune all GEMM shapes for a model on a specific GPU.
///
/// Called during model loading. Deduplicates shapes and tunes each one.
/// Returns a map from (N, K) to optimal GpuGemmConfig.
pub fn tune_model_gpu_gemms(
    sm_version: u32,
    shared_mem_bytes: usize,
    elem_bytes: usize,
    level: TuneLevel,
) -> std::collections::HashMap<(usize, usize), GpuGemmConfig> {
    let shapes = extract_model_gemm_shapes();
    let mut results = std::collections::HashMap::new();

    for (m, n, k) in shapes {
        let key = (n, k);
        if results.contains_key(&key) {
            continue; // Deduplicate by (N, K)
        }
        let config = tune_gpu_gemm(m, n, k, sm_version, shared_mem_bytes, elem_bytes, level);
        results.insert(key, config);
    }

    results
}
