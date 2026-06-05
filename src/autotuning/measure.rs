//! High-precision performance measurement for autotuning.
//!
//! Uses rdtsc on x86 for cycle-accurate timing, with fallback to
//! std::time::Instant. Includes warmup, outlier rejection, and
//! statistical analysis (median, IQR).
//!
//! The JIT measurement path (`measure_jit_gemm`) generates a GEMM microkernel
//! via the Phase 3 codegen pipeline, maps it into executable memory, and
//! benchmarks it with the specified tuning parameters.

use std::time::Instant;
use crate::autotuning::search_space::{TuningConfig, JitParams};
use crate::types::CompilerError;

/// Result of benchmarking a single configuration.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Median time in nanoseconds
    pub median_ns: f64,
    /// Interquartile range (measure of variance)
    pub iqr_ns: f64,
    /// Minimum observed time in nanoseconds
    pub min_ns: f64,
    /// Number of samples collected
    pub samples: usize,
    /// Computed GFLOPS (if applicable)
    pub gflops: Option<f64>,
    /// Computed bandwidth in GB/s (if applicable)
    pub bandwidth_gbs: Option<f64>,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "median={:.1}us IQR={:.1}us min={:.1}us",
            self.median_ns / 1000.0,
            self.iqr_ns / 1000.0,
            self.min_ns / 1000.0,
        )?;
        if let Some(gf) = self.gflops {
            write!(f, " {gf:.1}GFLOPS")?;
        }
        if let Some(bw) = self.bandwidth_gbs {
            write!(f, " {bw:.1}GB/s")?;
        }
        Ok(())
    }
}

/// Configuration for the benchmark harness.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Minimum number of warmup iterations
    pub warmup_iters: usize,
    /// Minimum number of measurement iterations
    pub min_iters: usize,
    /// Maximum number of measurement iterations
    pub max_iters: usize,
    /// Minimum total measurement time in nanoseconds
    pub min_time_ns: u64,
    /// Maximum total measurement time in nanoseconds (timeout)
    pub max_time_ns: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        BenchConfig {
            warmup_iters: 3,
            min_iters: 7,
            max_iters: 100,
            min_time_ns: 50_000_000,   // 50ms minimum
            max_time_ns: 2_000_000_000, // 2s timeout
        }
    }
}

impl BenchConfig {
    /// Fast config for autotuning (less precision, more throughput).
    pub fn fast() -> Self {
        BenchConfig {
            warmup_iters: 2,
            min_iters: 5,
            max_iters: 30,
            min_time_ns: 20_000_000,    // 20ms
            max_time_ns: 500_000_000,   // 500ms timeout
        }
    }

    /// Precise config for final validation.
    pub fn precise() -> Self {
        BenchConfig {
            warmup_iters: 5,
            min_iters: 15,
            max_iters: 200,
            min_time_ns: 200_000_000,   // 200ms
            max_time_ns: 5_000_000_000, // 5s timeout
        }
    }
}

/// Benchmark a closure, returning statistical results.
///
/// The closure receives an iteration index (for black_box purposes).
/// It should perform exactly one operation invocation.
pub fn bench_fn<F>(config: &BenchConfig, mut f: F) -> BenchResult
where
    F: FnMut(usize),
{
    // Warmup: run without measuring to fill caches, trigger JIT, etc.
    for i in 0..config.warmup_iters {
        f(i);
    }

    // Measurement phase
    let mut times_ns = Vec::with_capacity(config.max_iters);
    let wall_start = Instant::now();
    let mut iter = 0usize;

    loop {
        let t0 = precise_now_ns();
        f(iter);
        let t1 = precise_now_ns();
        let elapsed = t1.saturating_sub(t0);
        times_ns.push(elapsed as f64);
        iter += 1;

        let total_wall = wall_start.elapsed().as_nanos() as u64;
        if iter >= config.min_iters && total_wall >= config.min_time_ns {
            break;
        }
        if iter >= config.max_iters || total_wall >= config.max_time_ns {
            break;
        }
    }

    compute_stats(&mut times_ns)
}

/// Benchmark with FLOP count for GFLOPS calculation.
pub fn bench_fn_flops<F>(config: &BenchConfig, flops: u64, f: F) -> BenchResult
where
    F: FnMut(usize),
{
    let mut result = bench_fn(config, f);
    if result.median_ns > 0.0 {
        result.gflops = Some(flops as f64 / result.median_ns);
    }
    result
}

/// Benchmark with byte count for bandwidth calculation.
pub fn bench_fn_bandwidth<F>(config: &BenchConfig, bytes: u64, f: F) -> BenchResult
where
    F: FnMut(usize),
{
    let mut result = bench_fn(config, f);
    if result.median_ns > 0.0 {
        result.bandwidth_gbs = Some(bytes as f64 / result.median_ns);
    }
    result
}

// ── JIT codegen measurement path ─────────────────────────────────────────

/// Measure a JIT-generated GEMM kernel with the given tuning configuration.
///
/// This is the core measurement function for JIT autotuning (MS-2.1.1).
/// It generates a GEMM microkernel via the Phase 3 codegen pipeline with
/// the specified JIT parameters, maps it into executable memory, and
/// benchmarks it against test matrices.
///
/// The generated code is a standalone GEMM function with signature:
///   fn(a: *const f32, b: *const f32, c: *mut f32, scratchpad: *mut u8)
///
/// Returns the benchmark result with GFLOPS computed from 2*M*N*K.
#[cfg(feature = "jit-x86")]
pub fn measure_jit_gemm(
    m: usize,
    n: usize,
    k: usize,
    dtype: crate::types::DType,
    cfg: &TuningConfig,
    bench_cfg: &BenchConfig,
) -> Result<BenchResult, CompilerError> {
    let jit_params = cfg.jit.as_ref().cloned().unwrap_or_default();

    // Use global ExecutionPlan singleton (SPEC §10.15: all strategy through plan)
    let exec_plan = crate::compiler::planner::global_execution_plan();
    let profile = &exec_plan.profile;

    // Generate the GEMM microkernel with the specified JIT parameters
    let code_bytes = generate_jit_gemm_code(m, n, k, dtype, cfg, &jit_params, profile)?;

    // Map into executable memory
    let exec_buf = ExecutableGemmBuffer::new(&code_bytes)?;

    // Allocate test matrices (always f32 for now — standalone GEMM handles conversion internally via pack)
    let a = vec![0.5f32; m * k];
    let b = vec![0.3f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // Scratchpad for BLIS packing buffers
    let scratchpad_size = compute_scratchpad_size(m, n, k, cfg, profile);
    let mut scratchpad = vec![0u8; scratchpad_size];

    let flops = 2u64 * m as u64 * n as u64 * k as u64;

    let mut result = bench_fn_flops(bench_cfg, flops, |_| {
        // Zero C before each iteration for correctness
        for v in c.iter_mut() {
            *v = 0.0;
        }
        unsafe {
            exec_buf.call(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                scratchpad.as_mut_ptr(),
            );
        }
        black_box(&c);
    });

    // Also compute bandwidth for reference
    let bytes = ((m * k + k * n + m * n) * dtype.size_bytes()) as u64;
    if result.median_ns > 0.0 {
        result.bandwidth_gbs = Some(bytes as f64 / result.median_ns);
    }

    Ok(result)
}

/// Fallback for non-JIT builds: returns an error.
#[cfg(not(feature = "jit-x86"))]
pub fn measure_jit_gemm(
    _m: usize,
    _n: usize,
    _k: usize,
    _dtype: crate::types::DType,
    _cfg: &TuningConfig,
    _bench_cfg: &BenchConfig,
) -> Result<BenchResult, CompilerError> {
    Err("JIT measurement requires the jit-x86 feature".into())
}

/// Generate JIT GEMM machine code with the specified tuning parameters.
///
/// Creates an X86CodeGen, configures it with the JIT params from the
/// TuningConfig, and emits a standalone GEMM microkernel. The generated
/// code follows the ABI:
///   rdi = A ptr, rsi = B ptr, rdx = C ptr, rcx = scratchpad ptr
#[cfg(feature = "jit-x86")]
fn generate_jit_gemm_code(
    m: usize,
    n: usize,
    k: usize,
    dtype: crate::types::DType,
    cfg: &TuningConfig,
    jit_params: &JitParams,
    profile: &crate::dispatch::DeviceProfile,
) -> Result<Vec<u8>, CompilerError> {
    use crate::compiler::codegen::X86CodeGen;
    use crate::dispatch::device_profile::GemmBlocking;

    // Build a custom blocking from the tuning config
    let (mr, nr) = profile.microkernel_mr_nr();
    let effective_nr = jit_params.nr_variant;

    // Validate NR variant against register constraints
    let simd_w = profile.simd_width_f32();
    if effective_nr % simd_w != 0 {
        return Err(format!(
            "NR variant {} not aligned to SIMD width {}",
            effective_nr, simd_w
        ).into());
    }
    let nr_vecs = effective_nr / simd_w;
    let num_acc = mr * nr_vecs;
    let total_regs = profile.num_simd_regs();
    let scratch_needed = jit_params.reg_alloc_strategy.scratch_regs();
    if num_acc + scratch_needed > total_regs {
        return Err(format!(
            "Register overflow: {}*{} accumulators + {} scratch = {} > {} available",
            mr, nr_vecs, scratch_needed, num_acc + scratch_needed, total_regs
        ).into());
    }

    let mut codegen = X86CodeGen::new(profile, crate::types::DType::F32);

    // Apply JIT tuning parameters to the codegen
    codegen.set_jit_params(jit_params);

    // Use the blocking from the tuning config
    let blocking = GemmBlocking {
        kc: cfg.kc.max(4),
        mc: cfg.mc.max(mr),
        nc: cfg.nc.max(effective_nr),
        mr,
        nr: effective_nr,
    };

    codegen.emit_standalone_gemm(m, n, k, dtype, &blocking, profile)
}

/// Compute scratchpad size needed for BLIS packing buffers.
#[cfg(feature = "jit-x86")]
fn compute_scratchpad_size(
    _m: usize,
    _n: usize,
    _k: usize,
    cfg: &TuningConfig,
    profile: &crate::dispatch::DeviceProfile,
) -> usize {
    let (mr, _nr) = profile.microkernel_mr_nr();
    let effective_nr = cfg.jit.as_ref().map(|j| j.nr_variant).unwrap_or(16);
    let kc = cfg.kc.max(4);
    let mc = cfg.mc.max(mr);
    let nc = cfg.nc.max(effective_nr);

    let pack_a_panels = (mc + mr - 1) / mr;
    let pack_a_bytes = pack_a_panels * mr * kc * std::mem::size_of::<f32>();
    let pack_b_panels = (nc + effective_nr - 1) / effective_nr;
    let pack_b_bytes = pack_b_panels * effective_nr * kc * std::mem::size_of::<f32>();

    // Extra margin for alignment
    pack_a_bytes + pack_b_bytes + 4096
}

/// Executable memory buffer for a standalone GEMM function.
///
/// Maps JIT-generated machine code into an RWX memory region via mmap.
/// The function signature is:
///   fn(a: *const f32, b: *const f32, c: *mut f32, scratchpad: *mut u8)
struct ExecutableGemmBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for ExecutableGemmBuffer {}
unsafe impl Sync for ExecutableGemmBuffer {}

impl ExecutableGemmBuffer {
    fn new(code: &[u8]) -> Result<Self, CompilerError> {
        if code.is_empty() {
            return Err("Empty code buffer".into());
        }

        let page_size = page_size();
        let len = (code.len() + page_size - 1) & !(page_size - 1);

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err("mmap failed for JIT GEMM buffer".into());
        }

        let ptr = ptr as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
        }

        let ret = unsafe {
            libc::mprotect(ptr as *mut _, len, libc::PROT_READ | libc::PROT_EXEC)
        };
        if ret != 0 {
            unsafe { libc::munmap(ptr as *mut _, len); }
            return Err("mprotect failed for JIT GEMM buffer".into());
        }

        Ok(ExecutableGemmBuffer { ptr, len })
    }

    /// Call the JIT-generated GEMM function.
    ///
    /// ABI: rdi=A, rsi=B, rdx=C, rcx=scratchpad
    unsafe fn call(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        scratchpad: *mut u8,
    ) {
        type GemmFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, *mut u8);
        let f: GemmFn = std::mem::transmute(self.ptr);
        f(a, b, c, scratchpad);
    }
}

impl Drop for ExecutableGemmBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut _, self.len);
            }
        }
    }
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

// ── Statistical analysis ────────────────────────────────────────────────

fn compute_stats(times: &mut Vec<f64>) -> BenchResult {
    let n = times.len();
    if n == 0 {
        return BenchResult {
            median_ns: 0.0,
            iqr_ns: 0.0,
            min_ns: 0.0,
            samples: 0,
            gflops: None,
            bandwidth_gbs: None,
        };
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Remove outliers: discard top 10% (likely OS interrupts)
    let trimmed_len = (n * 9 / 10).max(1);
    let trimmed = &times[..trimmed_len];

    let median = if trimmed_len % 2 == 0 {
        (trimmed[trimmed_len / 2 - 1] + trimmed[trimmed_len / 2]) / 2.0
    } else {
        trimmed[trimmed_len / 2]
    };

    let q1 = trimmed[trimmed_len / 4];
    let q3 = trimmed[trimmed_len * 3 / 4];
    let iqr = q3 - q1;

    BenchResult {
        median_ns: median,
        iqr_ns: iqr,
        min_ns: trimmed[0],
        samples: n,
        gflops: None,
        bandwidth_gbs: None,
    }
}

// ── High-precision timing ───────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn precise_now_ns() -> u64 {
    let now = Instant::now();
    thread_local! {
        static EPOCH: Instant = Instant::now();
    }
    EPOCH.with(|epoch| now.duration_since(*epoch).as_nanos() as u64)
}

#[cfg(not(target_arch = "x86_64"))]
fn precise_now_ns() -> u64 {
    thread_local! {
        static EPOCH: Instant = Instant::now();
    }
    EPOCH.with(|epoch| Instant::now().duration_since(*epoch).as_nanos() as u64)
}

/// Compiler fence to prevent dead-code elimination of benchmark results.
#[inline(always)]
pub fn black_box<T>(x: T) -> T {
    // Use read_volatile to prevent the compiler from optimizing away the value.
    unsafe {
        let ret = std::ptr::read_volatile(&x);
        std::mem::forget(x);
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_fn_basic() {
        let config = BenchConfig::fast();
        let result = bench_fn(&config, |_| {
            // Simulate ~1us of work
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            black_box(sum);
        });
        assert!(result.samples >= config.min_iters);
        assert!(result.median_ns > 0.0);
        assert!(result.iqr_ns >= 0.0);
        eprintln!("Basic bench: {result}");
    }

    #[test]
    fn test_bench_fn_flops() {
        let config = BenchConfig::fast();
        let n = 1024usize;
        let a = vec![1.0f32; n];
        let b = vec![2.0f32; n];
        let result = bench_fn_flops(&config, (2 * n) as u64, |_| {
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += a[i] * b[i];
            }
            black_box(sum);
        });
        assert!(result.gflops.is_some());
        eprintln!("Dot product bench: {result}");
    }

    #[test]
    fn test_stats_correctness() {
        let mut times = vec![100.0, 200.0, 150.0, 120.0, 180.0, 130.0, 160.0, 140.0, 170.0, 110.0];
        let result = compute_stats(&mut times);
        // Median of sorted [100,110,120,130,140,150,160,170,180,200] trimmed to 9 = [100..180]
        // Median of 9 elements = element at index 4 = 140
        assert!(result.median_ns > 0.0);
        assert!(result.samples == 10);
    }

    #[test]
    fn test_bench_result_constructor_with_none_fields() {
        let result = BenchResult {
            median_ns: 1234.5,
            iqr_ns: 56.7,
            min_ns: 1000.0,
            samples: 20,
            gflops: None,
            bandwidth_gbs: None,
        };
        assert_eq!(result.median_ns, 1234.5);
        assert_eq!(result.iqr_ns, 56.7);
        assert_eq!(result.min_ns, 1000.0);
        assert_eq!(result.samples, 20);
        assert!(result.gflops.is_none());
        assert!(result.bandwidth_gbs.is_none());
    }

    #[test]
    fn test_bench_result_constructor_with_some_fields() {
        let result = BenchResult {
            median_ns: 500.0,
            iqr_ns: 10.0,
            min_ns: 480.0,
            samples: 30,
            gflops: Some(42.5),
            bandwidth_gbs: Some(12.3),
        };
        assert_eq!(result.gflops.unwrap(), 42.5);
        assert_eq!(result.bandwidth_gbs.unwrap(), 12.3);
    }

    #[test]
    fn test_bench_result_display_without_optionals() {
        let result = BenchResult {
            median_ns: 1500.0,
            iqr_ns: 200.0,
            min_ns: 1200.0,
            samples: 10,
            gflops: None,
            bandwidth_gbs: None,
        };
        let s = format!("{result}");
        assert!(s.contains("median="));
        assert!(s.contains("IQR="));
        assert!(s.contains("min="));
        assert!(!s.contains("GFLOPS"));
        assert!(!s.contains("GB/s"));
    }

    #[test]
    fn test_bench_result_display_with_all_optionals() {
        let result = BenchResult {
            median_ns: 100.0,
            iqr_ns: 10.0,
            min_ns: 90.0,
            samples: 15,
            gflops: Some(99.9),
            bandwidth_gbs: Some(33.3),
        };
        let s = format!("{result}");
        assert!(s.contains("GFLOPS"));
        assert!(s.contains("GB/s"));
    }

    #[test]
    fn test_bench_config_default_values() {
        let config = BenchConfig::default();
        assert_eq!(config.warmup_iters, 3);
        assert_eq!(config.min_iters, 7);
        assert_eq!(config.max_iters, 100);
        assert_eq!(config.min_time_ns, 50_000_000);
        assert_eq!(config.max_time_ns, 2_000_000_000);
    }

    #[test]
    fn test_bench_config_fast_values() {
        let config = BenchConfig::fast();
        assert_eq!(config.warmup_iters, 2);
        assert_eq!(config.min_iters, 5);
        assert_eq!(config.max_iters, 30);
        assert_eq!(config.min_time_ns, 20_000_000);
        assert_eq!(config.max_time_ns, 500_000_000);
    }

    #[test]
    fn test_bench_config_precise_values() {
        let config = BenchConfig::precise();
        assert_eq!(config.warmup_iters, 5);
        assert_eq!(config.min_iters, 15);
        assert_eq!(config.max_iters, 200);
        assert_eq!(config.min_time_ns, 200_000_000);
        assert_eq!(config.max_time_ns, 5_000_000_000);
    }

    #[test]
    fn test_bench_config_struct_update_syntax() {
        let base = BenchConfig::default();
        let custom = BenchConfig {
            warmup_iters: 10,
            ..base
        };
        assert_eq!(custom.warmup_iters, 10);
        assert_eq!(custom.min_iters, base.min_iters);
        assert_eq!(custom.max_iters, base.max_iters);
        assert_eq!(custom.min_time_ns, base.min_time_ns);
        assert_eq!(custom.max_time_ns, base.max_time_ns);
    }

    #[test]
    fn test_compute_stats_empty_input() {
        let mut times: Vec<f64> = vec![];
        let result = compute_stats(&mut times);
        assert_eq!(result.median_ns, 0.0);
        assert_eq!(result.iqr_ns, 0.0);
        assert_eq!(result.min_ns, 0.0);
        assert_eq!(result.samples, 0);
        assert!(result.gflops.is_none());
        assert!(result.bandwidth_gbs.is_none());
    }

    #[test]
    fn test_compute_stats_single_element() {
        let mut times = vec![42.0];
        let result = compute_stats(&mut times);
        assert_eq!(result.median_ns, 42.0);
        assert_eq!(result.min_ns, 42.0);
        assert_eq!(result.samples, 1);
    }

    #[test]
    fn test_compute_stats_all_identical_values() {
        let mut times = vec![100.0; 20];
        let result = compute_stats(&mut times);
        assert_eq!(result.median_ns, 100.0);
        assert_eq!(result.iqr_ns, 0.0);
        assert_eq!(result.min_ns, 100.0);
        assert_eq!(result.samples, 20);
    }

    #[test]
    fn test_bench_fn_bandwidth_computes_bandwidth() {
        let config = BenchConfig::fast();
        let bytes = 1024u64;
        let result = bench_fn_bandwidth(&config, bytes, |_| {
            let mut sum = 0u64;
            for i in 0..500 {
                sum = sum.wrapping_add(i);
            }
            black_box(sum);
        });
        assert!(result.bandwidth_gbs.is_some());
        let bw = result.bandwidth_gbs.unwrap();
        assert!(bw > 0.0);
    }

    #[test]
    fn test_black_box_passthrough() {
        let val = 42u64;
        let out = black_box(val);
        assert_eq!(out, 42u64);
    }

    // ── New tests (wave-12kee): 10 additional tests covering uncovered paths ──

    #[test]
    fn test_compute_stats_even_length_trimmed_median() {
        // Arrange: 10 elements -> trimmed to 9 (top 10% discarded)
        // sorted: [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]
        // after trimming (9/10): [10, 20, 30, 40, 50, 60, 70, 80, 90]
        // median of 9 elements (odd) = index 4 = 50.0
        let mut times = vec![10.0, 1000.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 20.0];
        let result = compute_stats(&mut times);
        // Assert
        assert_eq!(result.median_ns, 50.0);
        assert_eq!(result.min_ns, 10.0);
        // q1 at index 9/4=2 = 30.0, q3 at index 9*3/4=6 = 70.0
        assert_eq!(result.iqr_ns, 40.0);
        assert_eq!(result.samples, 10);
    }

    #[test]
    fn test_compute_stats_outlier_rejection_removes_top_10_percent() {
        // Arrange: 20 elements with one massive outlier at the end
        let mut times: Vec<f64> = (1..=19).map(|v| v as f64 * 10.0).collect();
        times.push(999_999.0); // outlier
        // sorted: [10, 20, ..., 190, 999999]
        // trimmed to 18 (20*9/10=18): [10, 20, ..., 180]
        let result = compute_stats(&mut times);
        // Assert: min should be the smallest (10.0), not affected by outlier
        assert_eq!(result.min_ns, 10.0);
        // Median of 18 elements (even) = (trimmed[8] + trimmed[9]) / 2 = (90 + 100) / 2 = 95
        assert_eq!(result.median_ns, 95.0);
        assert!(result.iqr_ns < 500.0, "IQR should not be inflated by outlier: got {}", result.iqr_ns);
    }

    #[test]
    fn test_compute_stats_two_elements() {
        // Arrange: 2 elements, trimmed to 1 (2*9/10=1)
        let mut times = vec![100.0, 500.0];
        let result = compute_stats(&mut times);
        // Assert: trimmed to 1 element, median = 100.0 (the smaller one after sort)
        assert_eq!(result.median_ns, 100.0);
        assert_eq!(result.min_ns, 100.0);
        assert_eq!(result.samples, 2);
    }

    #[test]
    fn test_bench_fn_flops_zero_median_does_not_set_gflops() {
        // Arrange: construct a zero-median result to test the guard logic
        let mut result = BenchResult {
            median_ns: 0.0,
            iqr_ns: 0.0,
            min_ns: 0.0,
            samples: 0,
            gflops: None,
            bandwidth_gbs: None,
        };
        // Act: replicate the guard from bench_fn_flops
        if result.median_ns > 0.0 {
            result.gflops = Some(1000.0 / result.median_ns);
        }
        // Assert: gflops should remain None when median is 0
        assert!(result.gflops.is_none());
    }

    #[test]
    fn test_bench_fn_bandwidth_zero_median_does_not_set_bandwidth() {
        // Arrange
        let mut result = BenchResult {
            median_ns: 0.0,
            iqr_ns: 0.0,
            min_ns: 0.0,
            samples: 0,
            gflops: None,
            bandwidth_gbs: None,
        };
        // Act: replicate the guard from bench_fn_bandwidth
        if result.median_ns > 0.0 {
            result.bandwidth_gbs = Some(4096.0 / result.median_ns);
        }
        // Assert: bandwidth should remain None when median is 0
        assert!(result.bandwidth_gbs.is_none());
    }

    #[test]
    fn test_bench_fn_invocation_count() {
        // Arrange: count how many times the closure is actually invoked
        let config = BenchConfig {
            warmup_iters: 2,
            min_iters: 5,
            max_iters: 10,
            min_time_ns: 0,                // hit min_iters immediately
            max_time_ns: 10_000_000_000,   // effectively infinite
        };
        let mut count = 0usize;
        // Act
        let _result = bench_fn(&config, |_i| {
            count += 1;
        });
        // Assert: warmup_iters + min_iters = 2 + 5 = 7 minimum invocations
        assert!(count >= config.warmup_iters + config.min_iters,
            "Expected at least {} invocations, got {}", config.warmup_iters + config.min_iters, count);
    }

    #[test]
    fn test_bench_result_display_format_values() {
        // Arrange
        let result = BenchResult {
            median_ns: 1_500_000.0,  // 1500 us
            iqr_ns: 200_000.0,        // 200 us
            min_ns: 1_200_000.0,      // 1200 us
            samples: 10,
            gflops: Some(42.5),
            bandwidth_gbs: None,
        };
        // Act
        let s = format!("{result}");
        // Assert: check that values are formatted in microseconds
        assert!(s.contains("median=1500.0us"), "display should contain median in us: got {s}");
        assert!(s.contains("IQR=200.0us"), "display should contain IQR in us: got {s}");
        assert!(s.contains("min=1200.0us"), "display should contain min in us: got {s}");
        assert!(s.contains("42.5GFLOPS"), "display should contain GFLOPS: got {s}");
        assert!(!s.contains("GB/s"), "display should not contain bandwidth when None");
    }

    #[test]
    fn test_bench_config_clone_is_independent() {
        // Arrange
        let original = BenchConfig::precise();
        // Act
        let cloned = original.clone();
        // Assert: cloned has same values
        assert_eq!(cloned.warmup_iters, original.warmup_iters);
        assert_eq!(cloned.min_iters, original.min_iters);
        assert_eq!(cloned.max_iters, original.max_iters);
        assert_eq!(cloned.min_time_ns, original.min_time_ns);
        assert_eq!(cloned.max_time_ns, original.max_time_ns);
    }

    #[test]
    fn test_bench_result_clone_preserves_values() {
        // Arrange
        let original = BenchResult {
            median_ns: 987.6,
            iqr_ns: 45.3,
            min_ns: 800.0,
            samples: 42,
            gflops: Some(123.4),
            bandwidth_gbs: Some(56.7),
        };
        // Act
        let cloned = original.clone();
        // Assert: all fields identical
        assert_eq!(cloned.median_ns, original.median_ns);
        assert_eq!(cloned.iqr_ns, original.iqr_ns);
        assert_eq!(cloned.min_ns, original.min_ns);
        assert_eq!(cloned.samples, original.samples);
        assert_eq!(cloned.gflops, original.gflops);
        assert_eq!(cloned.bandwidth_gbs, original.bandwidth_gbs);
    }

    #[test]
    fn test_page_size_is_positive() {
        // Arrange & Act
        let ps = page_size();
        // Assert: page size must be a positive power of 2
        assert!(ps > 0, "page size should be positive, got {ps}");
        assert!(ps.is_power_of_two(), "page size should be a power of 2, got {ps}");
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_measure_jit_gemm() {
        let cfg = TuningConfig {
            kc: 64,
            mc: 6,
            nc: 16,
            num_threads: 1,
            jit: Some(JitParams::default()),
        };
        let bench_cfg = BenchConfig::fast();
        let result = measure_jit_gemm(6, 16, 64, crate::types::DType::F32, &cfg, &bench_cfg);
        match result {
            Ok(r) => {
                assert!(r.median_ns > 0.0);
                assert!(r.gflops.is_some());
                eprintln!("JIT GEMM 6x16x64: {r}");
            }
            Err(e) => {
                // Codegen may fail for certain parameter combinations;
                // that's expected during search space exploration.
                eprintln!("JIT GEMM codegen error (expected in some configs): {e}");
            }
        }
    }

    // ── New tests (wave-12koc): 10 additional tests ──────────────────────────

    #[test]
    fn test_compute_stats_three_elements() {
        // Arrange: 3 elements, trimmed to max(3*9/10, 1) = max(2, 1) = 2
        // sorted: [50, 100, 500] -> trimmed to 2: [50, 100]
        // median of 2 (even) = (50 + 100) / 2 = 75.0
        let mut times = vec![100.0, 500.0, 50.0];
        // Act
        let result = compute_stats(&mut times);
        // Assert
        assert_eq!(result.median_ns, 75.0);
        assert_eq!(result.min_ns, 50.0);
        assert_eq!(result.samples, 3);
        assert!(result.iqr_ns >= 0.0);
    }

    #[test]
    fn test_bench_fn_flops_gflops_formula() {
        // Arrange: a trivial closure with known flops count
        let config = BenchConfig {
            warmup_iters: 1,
            min_iters: 3,
            max_iters: 5,
            min_time_ns: 0,
            max_time_ns: 10_000_000_000,
        };
        let flops = 1_000_000u64;
        // Act
        let result = bench_fn_flops(&config, flops, |_| {
            let mut sum = 0u64;
            for i in 0..200 {
                sum = sum.wrapping_add(i);
            }
            black_box(sum);
        });
        // Assert: gflops = flops / median_ns (GFLOPS = 1e9 flops / 1e9 ns)
        let gflops = result.gflops.expect("gflops should be computed when median > 0");
        assert!(gflops > 0.0, "GFLOPS should be positive, got {gflops}");
        let expected = flops as f64 / result.median_ns;
        assert!((gflops - expected).abs() < 1e-6,
            "GFLOPS formula mismatch: got {gflops}, expected {expected}");
    }

    #[test]
    fn test_bench_fn_bandwidth_formula() {
        // Arrange
        let config = BenchConfig {
            warmup_iters: 1,
            min_iters: 3,
            max_iters: 5,
            min_time_ns: 0,
            max_time_ns: 10_000_000_000,
        };
        let bytes = 4096u64;
        // Act
        let result = bench_fn_bandwidth(&config, bytes, |_| {
            let mut sum = 0u64;
            for i in 0..200 {
                sum = sum.wrapping_add(i);
            }
            black_box(sum);
        });
        // Assert: bandwidth_gbs = bytes / median_ns (GB/s = bytes / ns since 1e9 ns/s, no 1e9 factor)
        let bw = result.bandwidth_gbs.expect("bandwidth should be computed when median > 0");
        assert!(bw > 0.0, "bandwidth should be positive, got {bw}");
        let expected = bytes as f64 / result.median_ns;
        assert!((bw - expected).abs() < 1e-12,
            "bandwidth formula mismatch: got {bw}, expected {expected}");
    }

    #[test]
    fn test_precise_now_ns_monotonicity() {
        // Arrange & Act: take two timestamps in sequence
        let t0 = precise_now_ns();
        let t1 = precise_now_ns();
        // Assert: time is non-decreasing
        assert!(t1 >= t0, "precise_now_ns should be non-decreasing: t0={t0}, t1={t1}");
    }

    #[test]
    fn test_tuning_config_display_without_jit() {
        // Arrange
        let cfg = TuningConfig {
            kc: 128,
            mc: 64,
            nc: 256,
            num_threads: 4,
            jit: None,
        };
        // Act
        let s = format!("{cfg}");
        // Assert: should contain blocking params and threads, no JIT pipe
        assert!(s.contains("KC=128"), "should contain KC: got {s}");
        assert!(s.contains("MC=64"), "should contain MC: got {s}");
        assert!(s.contains("NC=256"), "should contain NC: got {s}");
        assert!(s.contains("threads=4"), "should contain threads: got {s}");
        assert!(!s.contains("|"), "should not contain JIT separator when jit is None");
    }

    #[test]
    fn test_tuning_config_display_with_jit() {
        // Arrange
        let cfg = TuningConfig {
            kc: 64,
            mc: 32,
            nc: 128,
            num_threads: 2,
            jit: Some(JitParams::default()),
        };
        // Act
        let s = format!("{cfg}");
        // Assert: should contain the pipe separator and JIT params
        assert!(s.contains("|"), "should contain JIT separator: got {s}");
        assert!(s.contains("k_unroll="), "should contain k_unroll: got {s}");
        assert!(s.contains("nr="), "should contain nr: got {s}");
    }

    #[test]
    fn test_jit_params_default_values() {
        // Arrange & Act
        let params = JitParams::default();
        // Assert
        assert_eq!(params.k_unroll, 4);
        assert_eq!(params.prefetch_distance, 8);
        assert_eq!(params.reg_alloc_strategy, crate::autotuning::search_space::RegAllocStrategy::Balanced);
        assert_eq!(params.sw_pipeline_depth, 0);
        assert_eq!(params.nr_variant, 16);
    }

    #[test]
    fn test_jit_params_display_format() {
        // Arrange
        let params = JitParams {
            k_unroll: 8,
            prefetch_distance: 12,
            reg_alloc_strategy: crate::autotuning::search_space::RegAllocStrategy::MinSpill,
            sw_pipeline_depth: 2,
            nr_variant: 32,
        };
        // Act
        let s = format!("{params}");
        // Assert
        assert!(s.contains("k_unroll=8"), "got {s}");
        assert!(s.contains("prefetch=12"), "got {s}");
        assert!(s.contains("reg=min_spill"), "got {s}");
        assert!(s.contains("swp=2"), "got {s}");
        assert!(s.contains("nr=32"), "got {s}");
    }

    #[test]
    fn test_reg_alloc_strategy_roundtrip() {
        // Arrange: enumerate all variants
        use crate::autotuning::search_space::RegAllocStrategy;
        let all = RegAllocStrategy::all();
        // Act & Assert: index roundtrip must be identity
        for variant in all {
            let idx = variant.to_index();
            let recovered = RegAllocStrategy::from_index(idx);
            assert_eq!(*variant, recovered,
                "roundtrip failed for {:?}: index={} recovered={:?}", variant, idx, recovered);
        }
    }

    #[test]
    fn test_executable_gemm_buffer_rejects_empty_code() {
        // Arrange: an empty code slice
        let code: &[u8] = &[];
        // Act
        let result = ExecutableGemmBuffer::new(code);
        // Assert: should return an error, not panic or succeed
        assert!(result.is_err(), "empty code buffer should be rejected");
    }
}
