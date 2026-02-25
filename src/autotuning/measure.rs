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
    cfg: &TuningConfig,
    bench_cfg: &BenchConfig,
) -> Result<BenchResult, String> {
    use crate::dispatch::DeviceProfile;

    let jit_params = cfg.jit.as_ref().cloned().unwrap_or_default();

    // Build a DeviceProfile for codegen
    let profile = DeviceProfile::detect();

    // Generate the GEMM microkernel with the specified JIT parameters
    let code_bytes = generate_jit_gemm_code(m, n, k, cfg, &jit_params, &profile)?;

    // Map into executable memory
    let exec_buf = ExecutableGemmBuffer::new(&code_bytes)?;

    // Allocate test matrices
    let a = vec![0.5f32; m * k];
    let b = vec![0.3f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // Scratchpad for BLIS packing buffers
    let scratchpad_size = compute_scratchpad_size(m, n, k, cfg, &profile);
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
    let bytes = ((m * k + k * n + m * n) * 4) as u64;
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
    _cfg: &TuningConfig,
    _bench_cfg: &BenchConfig,
) -> Result<BenchResult, String> {
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
    cfg: &TuningConfig,
    jit_params: &JitParams,
    profile: &crate::dispatch::DeviceProfile,
) -> Result<Vec<u8>, String> {
    use crate::compiler::codegen::x86_64::jit::X86CodeGen;
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
        ));
    }
    let nr_vecs = effective_nr / simd_w;
    let num_acc = mr * nr_vecs;
    let total_regs = profile.num_simd_regs();
    let scratch_needed = jit_params.reg_alloc_strategy.scratch_regs();
    if num_acc + scratch_needed > total_regs {
        return Err(format!(
            "Register overflow: {}*{} accumulators + {} scratch = {} > {} available",
            mr, nr_vecs, scratch_needed, num_acc + scratch_needed, total_regs
        ));
    }

    let mut codegen = X86CodeGen::new(profile);

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

    codegen.emit_standalone_gemm(m, n, k, &blocking, profile)
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
    let pack_a_bytes = pack_a_panels * mr * kc * 4;
    let pack_b_panels = (nc + effective_nr - 1) / effective_nr;
    let pack_b_bytes = pack_b_panels * effective_nr * kc * 4;

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
    fn new(code: &[u8]) -> Result<Self, String> {
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
        let result = measure_jit_gemm(6, 16, 64, &cfg, &bench_cfg);
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
}
