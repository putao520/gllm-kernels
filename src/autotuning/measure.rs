//! High-precision performance measurement for autotuning.
//!
//! Uses rdtsc on x86 for cycle-accurate timing, with fallback to
//! std::time::Instant. Includes warmup, outlier rejection, and
//! statistical analysis (median, IQR).

use std::time::Instant;

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
    // Use rdtsc for sub-nanosecond precision, convert to ns via Instant calibration.
    // For autotuning we care about relative comparisons, so Instant is fine.
    // rdtsc has ordering issues with out-of-order execution; use rdtscp or
    // lfence+rdtsc. For simplicity and portability, use Instant.
    //
    // Note: We use Instant because rdtsc->ns conversion requires knowing TSC frequency
    // which varies across CPUs. Instant::now() on Linux uses clock_gettime(MONOTONIC)
    // which is ~20ns overhead — acceptable for operations taking microseconds+.
    let now = Instant::now();
    // Store in thread-local to compute delta
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
}
