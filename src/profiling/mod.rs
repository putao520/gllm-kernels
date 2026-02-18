//! Built-in performance profiling for gllm-kernels.
//!
//! Provides cycle-accurate timing, hardware counter integration, FLOPS/bandwidth
//! measurement, and structured report generation (JSON/HTML).
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use gllm_kernels::profiling::{Profiler, counters};
//!
//! let mut profiler = Profiler::new();
//!
//! // Profile a GEMM operation
//! let (m, n, k) = (1024, 1024, 1024);
//! let workload = counters::gemm_workload(m, n, k);
//!
//! profiler.begin("gemm_f32_1024x1024x1024", workload);
//! // ... run kernel ...
//! profiler.end();
//!
//! // Generate reports
//! profiler.report().write_json("/tmp/perf.json").unwrap();
//! profiler.report().write_html("/tmp/perf.html").unwrap();
//! ```
//!
//! # Scoped API
//!
//! ```rust,no_run
//! use gllm_kernels::profiling::{Profiler, counters};
//!
//! let mut profiler = Profiler::new();
//! let metrics = profiler.measure("silu_4096",
//!     counters::elementwise_unary_workload(4096, 4),
//!     || {
//!         // ... run kernel ...
//!     },
//! );
//! println!("Bandwidth: {:.1} GB/s", metrics.bandwidth_gbs);
//! ```

pub mod timer;
pub mod perf_events;
pub mod counters;
pub mod report;

use timer::CycleTimer;
use perf_events::PerfEventGroup;
use counters::{OpWorkload, PerfMetrics, HwPeak, compute_metrics};
use report::{ProfileReport, ProfileEntry};
use crate::cpu_kernels::get_isa_level;

/// High-level profiler that collects timing + HW counters for kernel runs.
pub struct Profiler {
    peak: HwPeak,
    perf_group: PerfEventGroup,
    entries: Vec<ProfileEntry>,
    // Active measurement state
    active_name: Option<String>,
    active_workload: Option<OpWorkload>,
    active_timer: Option<CycleTimer>,
}

impl Profiler {
    /// Create a new profiler with auto-detected hardware peaks.
    pub fn new() -> Self {
        Self {
            peak: HwPeak::detect(),
            perf_group: PerfEventGroup::new(),
            entries: Vec::new(),
            active_name: None,
            active_workload: None,
            active_timer: None,
        }
    }

    /// Create a profiler with manually specified hardware peaks.
    pub fn with_peak(peak_gflops: f64, peak_bandwidth_gbs: f64) -> Self {
        Self {
            peak: HwPeak::manual(peak_gflops, peak_bandwidth_gbs),
            perf_group: PerfEventGroup::new(),
            entries: Vec::new(),
            active_name: None,
            active_workload: None,
            active_timer: None,
        }
    }

    /// Returns true if hardware performance counters are available.
    pub fn has_hw_counters(&self) -> bool {
        self.perf_group.is_available()
    }

    /// Returns the detected hardware peak.
    pub fn hw_peak(&self) -> &HwPeak {
        &self.peak
    }

    /// Begin a profiled region. Must be paired with `end()`.
    pub fn begin(&mut self, name: &str, workload: OpWorkload) {
        self.active_name = Some(name.to_string());
        self.active_workload = Some(workload);
        self.perf_group.start();
        self.active_timer = Some(CycleTimer::start());
    }

    /// End the profiled region started by `begin()`.
    /// Returns the computed metrics.
    pub fn end(&mut self) -> PerfMetrics {
        let mut timer = self.active_timer.take().expect("end() called without begin()");
        timer.stop();
        let hw = self.perf_group.stop();
        let workload = self.active_workload.take().expect("end() called without begin()");
        let name = self.active_name.take().expect("end() called without begin()");

        let metrics = compute_metrics(workload, &timer, Some(&self.peak));
        let hw_counters = if self.perf_group.is_available() { Some(hw) } else { None };

        self.entries.push(ProfileEntry {
            name,
            workload,
            metrics,
            hw_counters,
        });

        metrics
    }

    /// Profile a closure, returning its metrics. Convenience wrapper around begin/end.
    pub fn measure<F, R>(&mut self, name: &str, workload: OpWorkload, f: F) -> PerfMetrics
    where
        F: FnOnce() -> R,
    {
        self.begin(name, workload);
        let result = f();
        std::hint::black_box(result);
        self.end()
    }

    /// Profile a closure N times, returning the median metrics.
    pub fn measure_median<F>(&mut self, name: &str, workload: OpWorkload, iterations: usize, mut f: F) -> PerfMetrics
    where
        F: FnMut(),
    {
        let mut all_metrics: Vec<PerfMetrics> = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            self.begin(name, workload);
            f();
            let m = self.end();
            // Remove the auto-added entry; we'll add the median manually
            self.entries.pop();
            all_metrics.push(m);
        }

        // Sort by elapsed time, pick median
        all_metrics.sort_by(|a, b| a.elapsed_secs.partial_cmp(&b.elapsed_secs).unwrap());
        let median = all_metrics[iterations / 2];

        // Collect HW counters from the median run (re-run once with counters)
        self.begin(name, workload);
        f();
        let final_metrics = self.end();
        // Replace the last entry's timing with median
        if let Some(last) = self.entries.last_mut() {
            last.metrics = PerfMetrics {
                elapsed_secs: median.elapsed_secs,
                elapsed_cycles: median.elapsed_cycles,
                gflops: median.gflops,
                bandwidth_gbs: median.bandwidth_gbs,
                efficiency: median.efficiency,
            };
            // Keep hw_counters from the final run (they're representative)
            let _ = final_metrics;
        }

        median
    }

    /// Number of profiled entries collected so far.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Build a report from all collected entries.
    pub fn report(&self) -> ProfileReport {
        let isa_str = match get_isa_level() {
            crate::cpu_kernels::IsaLevel::Avx512Fp16 => "avx512fp16",
            crate::cpu_kernels::IsaLevel::Avx512 => "avx512",
            crate::cpu_kernels::IsaLevel::Avx2 => "avx2",
            crate::cpu_kernels::IsaLevel::Neon => "neon",
            crate::cpu_kernels::IsaLevel::Scalar => "scalar",
        };
        let mut report = ProfileReport::new(self.peak, isa_str);
        for entry in &self.entries {
            report.add(entry.clone());
        }
        report
    }

    /// Print a summary table to stderr (for quick terminal inspection).
    pub fn print_summary(&self) {
        eprintln!("{}", self.summary_string());
    }

    /// Format a summary table as a string.
    pub fn summary_string(&self) -> String {
        let mut s = String::with_capacity(2048);
        s.push_str(&format!(
            "\n{:=<90}\n gllm-kernels profiling summary  |  ISA: {:?}  |  Peak: {:.1} GFLOPS, {:.1} GB/s\n{:=<90}\n",
            "", get_isa_level(), self.peak.gflops, self.peak.bandwidth_gbs, ""
        ));
        s.push_str(&format!(
            " {:<40} {:>8} {:>10} {:>10} {:>8} {:>6}\n",
            "Kernel", "Time(us)", "GFLOPS", "BW(GB/s)", "Eff%", "IPC"
        ));
        s.push_str(&format!("{:-<90}\n", ""));

        for e in &self.entries {
            let ipc_str = e.hw_counters.as_ref()
                .map(|h| format!("{:.2}", h.ipc()))
                .unwrap_or_else(|| "-".to_string());
            s.push_str(&format!(
                " {:<40} {:>8.1} {:>10.2} {:>10.2} {:>7.1}% {:>6}\n",
                e.name,
                e.metrics.elapsed_secs * 1e6,
                e.metrics.gflops,
                e.metrics.bandwidth_gbs,
                e.metrics.efficiency * 100.0,
                ipc_str,
            ));
        }
        s.push_str(&format!("{:=<90}\n", ""));
        s
    }

    /// Clear all collected entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_begin_end() {
        let mut profiler = Profiler::new();
        let workload = counters::gemm_workload(256, 256, 256);

        profiler.begin("test_gemm", workload);
        // Simulate work
        let mut x = 0.0f64;
        for i in 0..100_000 {
            x += (i as f64).sqrt();
        }
        std::hint::black_box(x);
        let metrics = profiler.end();

        assert!(metrics.elapsed_secs > 0.0);
        assert!(metrics.elapsed_cycles > 0);
        assert_eq!(profiler.entry_count(), 1);
    }

    #[test]
    fn test_profiler_measure() {
        let mut profiler = Profiler::new();
        let workload = counters::elementwise_unary_workload(4096, 4);

        let metrics = profiler.measure("test_silu", workload, || {
            let mut v = vec![0.0f32; 4096];
            for i in 0..4096 {
                v[i] = (i as f32).sin();
            }
            std::hint::black_box(&v);
        });

        assert!(metrics.elapsed_secs > 0.0);
        assert!(metrics.bandwidth_gbs > 0.0);
        assert_eq!(profiler.entry_count(), 1);
    }

    #[test]
    fn test_profiler_report_generation() {
        let mut profiler = Profiler::new();

        // Add a few entries
        for size in [256, 512, 1024] {
            let workload = counters::gemm_workload(size, size, size);
            profiler.measure(&format!("gemm_{size}x{size}x{size}"), workload, || {
                std::thread::sleep(std::time::Duration::from_micros(100));
            });
        }

        let report = profiler.report();
        assert_eq!(report.entries.len(), 3);

        let json = report.to_json();
        assert!(json.contains("gemm_256x256x256"));
        assert!(json.contains("gemm_1024x1024x1024"));

        let html = report.to_html();
        assert!(html.contains("gllm-kernels Performance Report"));
        assert!(html.contains("Efficiency Overview"));
    }

    #[test]
    fn test_profiler_summary() {
        let mut profiler = Profiler::new();
        let workload = counters::gemm_workload(512, 512, 512);
        profiler.measure("gemm_512", workload, || {
            std::thread::sleep(std::time::Duration::from_micros(50));
        });

        let summary = profiler.summary_string();
        assert!(summary.contains("gemm_512"));
        assert!(summary.contains("GFLOPS"));
    }
}
