//! FLOPS counting and memory bandwidth measurement.
//!
//! Provides accounting for compute-bound and memory-bound operators:
//! - FLOPS: floating-point operations per second (for GEMM, dot products)
//! - Bandwidth: bytes/second actually transferred (for memory-bound ops)
//!
//! The counters are purely arithmetic — they compute theoretical FLOP/byte
//! counts from operator parameters, then divide by measured wall time.

use crate::profiling::timer::CycleTimer;

/// Describes the workload of a profiled operation.
#[derive(Debug, Clone, Copy)]
pub enum OpWorkload {
    /// Compute-bound: total floating-point operations.
    /// GEMM(m,n,k) = 2*m*n*k FLOPs (multiply + add per element).
    Compute { flops: u64 },

    /// Memory-bound: total bytes read + written.
    /// E.g. vec_add(n, f32): read 2*n*4 + write n*4 = 12*n bytes.
    Memory { bytes: u64 },

    /// Mixed (quantized GEMV): both compute and memory are relevant.
    Mixed { flops: u64, bytes: u64 },
}

/// Performance metrics computed from workload + timing.
#[derive(Debug, Clone, Copy)]
pub struct PerfMetrics {
    /// Wall-clock seconds.
    pub elapsed_secs: f64,
    /// Elapsed TSC cycles.
    pub elapsed_cycles: u64,
    /// GFLOPS (billions of FLOP/s). Zero for pure memory-bound ops.
    pub gflops: f64,
    /// Bandwidth in GB/s. Zero for pure compute-bound ops.
    pub bandwidth_gbs: f64,
    /// Efficiency vs theoretical peak (0.0 - 1.0). Requires peak to be set.
    pub efficiency: f64,
}

/// Theoretical peak performance of the current hardware.
#[derive(Debug, Clone, Copy)]
pub struct HwPeak {
    /// Peak GFLOPS (single-precision).
    pub gflops: f64,
    /// Peak memory bandwidth in GB/s (from STREAM or spec).
    pub bandwidth_gbs: f64,
}

impl HwPeak {
    /// Detect hardware peak from CPU info.
    /// Uses: cores * frequency * FMA_throughput * SIMD_width * 2 (mul+add).
    pub fn detect() -> Self {
        let (gflops, bw) = detect_hw_peak();
        Self { gflops, bandwidth_gbs: bw }
    }

    /// Manually specify peaks (for when auto-detection is inaccurate).
    pub fn manual(gflops: f64, bandwidth_gbs: f64) -> Self {
        Self { gflops, bandwidth_gbs }
    }
}

/// Compute performance metrics from workload, timing, and optional peak.
pub fn compute_metrics(workload: OpWorkload, timer: &CycleTimer, peak: Option<&HwPeak>) -> PerfMetrics {
    let secs = timer.elapsed_secs();
    let cycles = timer.elapsed_cycles();

    let (gflops, bw_gbs) = match workload {
        OpWorkload::Compute { flops } => {
            let gf = if secs > 0.0 { flops as f64 / secs / 1e9 } else { 0.0 };
            (gf, 0.0)
        }
        OpWorkload::Memory { bytes } => {
            let bw = if secs > 0.0 { bytes as f64 / secs / 1e9 } else { 0.0 };
            (0.0, bw)
        }
        OpWorkload::Mixed { flops, bytes } => {
            let gf = if secs > 0.0 { flops as f64 / secs / 1e9 } else { 0.0 };
            let bw = if secs > 0.0 { bytes as f64 / secs / 1e9 } else { 0.0 };
            (gf, bw)
        }
    };

    let efficiency = match (workload, peak) {
        (OpWorkload::Compute { .. } | OpWorkload::Mixed { .. }, Some(p)) if p.gflops > 0.0 => {
            gflops / p.gflops
        }
        (OpWorkload::Memory { .. }, Some(p)) if p.bandwidth_gbs > 0.0 => {
            bw_gbs / p.bandwidth_gbs
        }
        _ => 0.0,
    };

    PerfMetrics {
        elapsed_secs: secs,
        elapsed_cycles: cycles,
        gflops,
        bandwidth_gbs: bw_gbs,
        efficiency,
    }
}

// ---------------------------------------------------------------------------
// Workload helpers for common operators
// ---------------------------------------------------------------------------

/// GEMM workload: 2*M*N*K FLOPs.
pub fn gemm_workload(m: usize, n: usize, k: usize) -> OpWorkload {
    OpWorkload::Compute { flops: 2 * m as u64 * n as u64 * k as u64 }
}

/// GEMV workload: 2*M*N FLOPs.
pub fn gemv_workload(m: usize, n: usize) -> OpWorkload {
    OpWorkload::Compute { flops: 2 * m as u64 * n as u64 }
}

/// Element-wise unary op (silu, relu, exp, etc.): read N + write N elements.
pub fn elementwise_unary_workload(n: usize, elem_bytes: usize) -> OpWorkload {
    OpWorkload::Memory { bytes: 2 * n as u64 * elem_bytes as u64 }
}

/// Element-wise binary op (add, mul, etc.): read 2N + write N elements.
pub fn elementwise_binary_workload(n: usize, elem_bytes: usize) -> OpWorkload {
    OpWorkload::Memory { bytes: 3 * n as u64 * elem_bytes as u64 }
}

/// Dot product: read 2N elements, compute 2N FLOPs (mul + add).
pub fn dot_workload(n: usize, elem_bytes: usize) -> OpWorkload {
    OpWorkload::Mixed {
        flops: 2 * n as u64,
        bytes: 2 * n as u64 * elem_bytes as u64,
    }
}

/// RMS norm: read N (input) + N (weight) + write N = 3N elements, ~5N FLOPs.
pub fn rms_norm_workload(n: usize, elem_bytes: usize) -> OpWorkload {
    OpWorkload::Mixed {
        flops: 5 * n as u64,
        bytes: 3 * n as u64 * elem_bytes as u64,
    }
}

/// Softmax: read N + write N = 2N elements, ~5N FLOPs (max + exp + sum + div).
pub fn softmax_workload(n: usize, elem_bytes: usize) -> OpWorkload {
    OpWorkload::Mixed {
        flops: 5 * n as u64,
        bytes: 2 * n as u64 * elem_bytes as u64,
    }
}

/// Quantized GEMV: reads quantized weights + f32 input, compute-heavy decode+FMA.
pub fn quant_gemv_workload(m: usize, n: usize, weight_bytes_per_row: usize, elem_bytes: usize) -> OpWorkload {
    OpWorkload::Mixed {
        flops: 2 * m as u64 * n as u64,
        bytes: m as u64 * weight_bytes_per_row as u64 + n as u64 * elem_bytes as u64,
    }
}

// ---------------------------------------------------------------------------
// Hardware peak detection
// ---------------------------------------------------------------------------

fn detect_hw_peak() -> (f64, f64) {
    let cores = detect_physical_cores();
    let freq_ghz = detect_max_freq_ghz();
    let simd_flops_per_cycle = detect_simd_flops_per_cycle();
    let peak_gflops = cores as f64 * freq_ghz * simd_flops_per_cycle;
    let peak_bw = detect_memory_bandwidth();
    (peak_gflops, peak_bw)
}

fn detect_physical_cores() -> usize {
    // Try to get physical cores (not hyperthreads)
    #[cfg(target_os = "linux")]
    {
        // NOTE: /sys/.../core_siblings_list parsing is unreliable across kernel versions;
        // using heuristic below instead.
        // Simpler: count online CPUs / 2 (assuming HT)
        let online = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Heuristic: if > 1, assume HT and halve
        if online > 1 { return online / 2; }
        return online;
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get() / 2)
            .unwrap_or(1)
            .max(1)
    }
}

fn detect_max_freq_ghz() -> f64 {
    // Use TSC frequency as proxy (close to max turbo on modern CPUs)
    let tsc = super::timer::tsc_freq_hz();
    tsc as f64 / 1e9
}

fn detect_simd_flops_per_cycle() -> f64 {
    // FLOPs per cycle per core for FMA:
    // AVX-512: 2 FMA units * 16 f32/zmm * 2 (mul+add) = 64
    // AVX2:    2 FMA units * 8 f32/ymm * 2 (mul+add) = 32
    // SSE:     1 FMA unit * 4 f32/xmm * 2 = 8
    // NEON:    2 FMLA units * 4 f32/v * 2 = 16
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return 64.0;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return 32.0;
        }
        return 8.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return 16.0;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return 4.0; // conservative scalar
    }
}

fn detect_memory_bandwidth() -> f64 {
    // Rough estimate from DDR spec. Real measurement would need STREAM benchmark.
    // DDR4-3200: ~25 GB/s per channel, dual channel = ~50 GB/s
    // DDR5-4800: ~38 GB/s per channel, dual channel = ~76 GB/s
    // Conservative default: 40 GB/s
    #[cfg(target_os = "linux")]
    {
        // Try to read from DMI or meminfo for a hint
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let total_kb: u64 = content.lines()
                .find(|l| l.starts_with("MemTotal"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            // Heuristic: servers with >64GB likely have higher bandwidth
            if total_kb > 64 * 1024 * 1024 {
                return 80.0; // likely multi-channel
            }
        }
    }
    40.0 // conservative default for desktop dual-channel DDR4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::timer::CycleTimer;

    #[test]
    fn test_hw_peak_detect() {
        let peak = HwPeak::detect();
        eprintln!("Detected peak: {:.1} GFLOPS, {:.1} GB/s", peak.gflops, peak.bandwidth_gbs);
        assert!(peak.gflops > 0.0);
        assert!(peak.bandwidth_gbs > 0.0);
    }

    #[test]
    fn test_gemm_workload() {
        let w = gemm_workload(1024, 1024, 1024);
        match w {
            OpWorkload::Compute { flops } => {
                assert_eq!(flops, 2 * 1024 * 1024 * 1024);
            }
            _ => panic!("expected Compute"),
        }
    }

    #[test]
    fn test_compute_metrics() {
        let mut timer = CycleTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.stop();

        let workload = gemm_workload(512, 512, 512);
        let peak = HwPeak::manual(100.0, 50.0);
        let metrics = compute_metrics(workload, &timer, Some(&peak));

        assert!(metrics.elapsed_secs > 0.0);
        assert!(metrics.gflops > 0.0);
        // Efficiency can exceed 1.0 in this synthetic test because we sleep
        // instead of doing real compute — just verify it's positive and finite.
        assert!(metrics.efficiency > 0.0);
        assert!(metrics.efficiency.is_finite());
    }
}
