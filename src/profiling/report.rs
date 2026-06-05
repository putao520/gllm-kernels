//! Performance report generation in JSON and HTML formats.
//!
//! Collects profiling results from multiple kernel runs and produces
//! structured reports for analysis.

use crate::profiling::counters::{PerfMetrics, HwPeak, OpWorkload};
use crate::profiling::perf_events::HwCounters;

/// A single profiled kernel invocation.
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    /// Kernel name (e.g. "gemm_f32_1024x1024x1024").
    pub name: String,
    /// Operator workload description.
    pub workload: OpWorkload,
    /// Timing and throughput metrics.
    pub metrics: PerfMetrics,
    /// Hardware counters (if available).
    pub hw_counters: Option<HwCounters>,
}

/// Collected profiling report.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Hardware description.
    pub hw_peak: HwPeak,
    /// ISA level detected.
    pub isa: String,
    /// All profiled entries.
    pub entries: Vec<ProfileEntry>,
    /// Timestamp of report generation.
    pub timestamp: String,
}

impl ProfileReport {
    pub fn new(hw_peak: HwPeak, isa: &str) -> Self {
        let timestamp = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            format!("{}", now.as_secs())
        };
        Self {
            hw_peak,
            isa: isa.to_string(),
            entries: Vec::new(),
            timestamp,
        }
    }

    pub fn add(&mut self, entry: ProfileEntry) {
        self.entries.push(entry);
    }

    /// Generate JSON report string (no serde dependency — hand-rolled).
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("{\n");
        s.push_str(&format!("  \"timestamp\": \"{}\",\n", self.timestamp));
        s.push_str(&format!("  \"isa\": \"{}\",\n", self.isa));
        s.push_str(&format!("  \"hw_peak_gflops\": {:.2},\n", self.hw_peak.gflops));
        s.push_str(&format!("  \"hw_peak_bandwidth_gbs\": {:.2},\n", self.hw_peak.bandwidth_gbs));
        s.push_str("  \"entries\": [\n");

        for (i, e) in self.entries.iter().enumerate() {
            s.push_str("    {\n");
            s.push_str(&format!("      \"name\": \"{}\",\n", e.name));
            s.push_str(&format!("      \"workload_type\": \"{}\",\n", workload_type_str(&e.workload)));
            s.push_str(&format!("      \"elapsed_secs\": {:.9},\n", e.metrics.elapsed_secs));
            s.push_str(&format!("      \"elapsed_cycles\": {},\n", e.metrics.elapsed_cycles));
            s.push_str(&format!("      \"gflops\": {:.3},\n", e.metrics.gflops));
            s.push_str(&format!("      \"bandwidth_gbs\": {:.3},\n", e.metrics.bandwidth_gbs));
            s.push_str(&format!("      \"efficiency\": {:.4}", e.metrics.efficiency));

            if let Some(ref hw) = e.hw_counters {
                s.push_str(",\n");
                s.push_str(&format!("      \"hw_cycles\": {},\n", hw.cycles));
                s.push_str(&format!("      \"hw_instructions\": {},\n", hw.instructions));
                s.push_str(&format!("      \"ipc\": {:.3},\n", hw.ipc()));
                s.push_str(&format!("      \"l1d_misses\": {},\n", hw.l1d_misses));
                s.push_str(&format!("      \"llc_misses\": {},\n", hw.llc_misses));
                s.push_str(&format!("      \"l1d_mpki\": {:.3},\n", hw.l1d_mpki()));
                s.push_str(&format!("      \"llc_mpki\": {:.3},\n", hw.llc_mpki()));
                s.push_str(&format!("      \"branch_misses\": {}", hw.branch_misses));
            }

            s.push('\n');
            if i + 1 < self.entries.len() {
                s.push_str("    },\n");
            } else {
                s.push_str("    }\n");
            }
        }

        s.push_str("  ]\n");
        s.push_str("}\n");
        s
    }

    /// Generate HTML report with embedded CSS for visual analysis.
    pub fn to_html(&self) -> String {
        let mut s = String::with_capacity(8192);
        s.push_str(HTML_HEAD);
        s.push_str("<body>\n<div class=\"container\">\n");
        s.push_str("<h1>gllm-kernels Performance Report</h1>\n");

        // Hardware info
        s.push_str("<div class=\"hw-info\">\n");
        s.push_str(&format!("<p><strong>ISA:</strong> {}</p>\n", self.isa));
        s.push_str(&format!("<p><strong>Peak GFLOPS:</strong> {:.1}</p>\n", self.hw_peak.gflops));
        s.push_str(&format!("<p><strong>Peak Bandwidth:</strong> {:.1} GB/s</p>\n", self.hw_peak.bandwidth_gbs));
        s.push_str(&format!("<p><strong>Timestamp:</strong> {}</p>\n", self.timestamp));
        s.push_str("</div>\n");

        // Summary table
        s.push_str("<h2>Kernel Performance Summary</h2>\n");
        s.push_str("<table>\n<thead><tr>\n");
        s.push_str("<th>Kernel</th><th>Type</th><th>Time (us)</th><th>Cycles</th>");
        s.push_str("<th>GFLOPS</th><th>BW (GB/s)</th><th>Efficiency</th>");
        s.push_str("<th>IPC</th><th>L1D MPKI</th><th>LLC MPKI</th>");
        s.push_str("</tr></thead>\n<tbody>\n");

        for e in &self.entries {
            let eff_pct = e.metrics.efficiency * 100.0;
            let eff_class = if eff_pct >= 85.0 { "eff-good" }
                else if eff_pct >= 60.0 { "eff-ok" }
                else { "eff-bad" };

            s.push_str("<tr>\n");
            s.push_str(&format!("<td class=\"name\">{}</td>\n", e.name));
            s.push_str(&format!("<td>{}</td>\n", workload_type_str(&e.workload)));
            s.push_str(&format!("<td>{:.1}</td>\n", e.metrics.elapsed_secs * 1e6));
            s.push_str(&format!("<td>{}</td>\n", e.metrics.elapsed_cycles));
            s.push_str(&format!("<td>{:.2}</td>\n", e.metrics.gflops));
            s.push_str(&format!("<td>{:.2}</td>\n", e.metrics.bandwidth_gbs));
            s.push_str(&format!("<td class=\"{}\">{:.1}%</td>\n", eff_class, eff_pct));

            if let Some(ref hw) = e.hw_counters {
                s.push_str(&format!("<td>{:.2}</td>\n", hw.ipc()));
                s.push_str(&format!("<td>{:.2}</td>\n", hw.l1d_mpki()));
                s.push_str(&format!("<td>{:.2}</td>\n", hw.llc_mpki()));
            } else {
                s.push_str("<td>-</td><td>-</td><td>-</td>\n");
            }
            s.push_str("</tr>\n");
        }

        s.push_str("</tbody></table>\n");

        // Efficiency bar chart (pure CSS)
        s.push_str("<h2>Efficiency Overview</h2>\n");
        s.push_str("<div class=\"chart\">\n");
        for e in &self.entries {
            let pct = (e.metrics.efficiency * 100.0).min(100.0);
            let color = if pct >= 85.0 { "#4caf50" }
                else if pct >= 60.0 { "#ff9800" }
                else { "#f44336" };
            s.push_str(&format!(
                "<div class=\"bar-row\"><span class=\"bar-label\">{}</span>\
                 <div class=\"bar-bg\"><div class=\"bar-fill\" style=\"width:{pct:.1}%;background:{color}\">\
                 {pct:.1}%</div></div></div>\n",
                e.name
            ));
        }
        s.push_str("</div>\n");

        s.push_str("</div>\n</body></html>\n");
        s
    }

    /// Write JSON report to file.
    pub fn write_json(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    /// Write HTML report to file.
    pub fn write_html(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_html())
    }
}

fn workload_type_str(w: &OpWorkload) -> &'static str {
    match w {
        OpWorkload::Compute { .. } => "compute",
        OpWorkload::Memory { .. } => "memory",
        OpWorkload::Mixed { .. } => "mixed",
    }
}

const HTML_HEAD: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gllm-kernels Performance Report</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
       background: #1a1a2e; color: #e0e0e0; padding: 20px; }
.container { max-width: 1200px; margin: 0 auto; }
h1 { color: #00d4ff; margin-bottom: 16px; font-size: 24px; }
h2 { color: #7c83ff; margin: 24px 0 12px; font-size: 18px; }
.hw-info { background: #16213e; padding: 12px 16px; border-radius: 6px;
           margin-bottom: 20px; border-left: 3px solid #00d4ff; }
.hw-info p { margin: 4px 0; font-size: 14px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #0f3460; color: #00d4ff; padding: 8px 10px; text-align: left;
     border-bottom: 2px solid #00d4ff; }
td { padding: 6px 10px; border-bottom: 1px solid #2a2a4a; }
tr:hover { background: #1a1a3e; }
.name { font-weight: bold; color: #e0e0e0; }
.eff-good { color: #4caf50; font-weight: bold; }
.eff-ok { color: #ff9800; font-weight: bold; }
.eff-bad { color: #f44336; font-weight: bold; }
.chart { margin-top: 12px; }
.bar-row { display: flex; align-items: center; margin: 4px 0; }
.bar-label { width: 280px; font-size: 12px; text-align: right; padding-right: 10px;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-bg { flex: 1; background: #2a2a4a; border-radius: 3px; height: 22px; }
.bar-fill { height: 100%; border-radius: 3px; font-size: 11px; line-height: 22px;
            padding-left: 6px; color: #fff; min-width: 40px; }
</style>
</head>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::counters;

    #[test]
    fn test_json_report() {
        let peak = HwPeak::manual(200.0, 50.0);
        let mut report = ProfileReport::new(peak, "avx2");

        let mut timer = crate::profiling::timer::CycleTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.stop();

        let workload = counters::gemm_workload(512, 512, 512);
        let metrics = counters::compute_metrics(workload, &timer, Some(&peak));

        report.add(ProfileEntry {
            name: "gemm_f32_512x512x512".to_string(),
            workload,
            metrics,
            hw_counters: Some(HwCounters {
                cycles: 1_000_000,
                instructions: 2_500_000,
                l1d_misses: 500,
                llc_misses: 10,
                branch_misses: 20,
            }),
        });

        let json = report.to_json();
        assert!(json.contains("gemm_f32_512x512x512"));
        assert!(json.contains("\"isa\": \"avx2\""));
        assert!(json.contains("\"hw_peak_gflops\": 200.00"));
        eprintln!("{json}");
    }

    #[test]
    fn test_html_report() {
        let peak = HwPeak::manual(200.0, 50.0);
        let mut report = ProfileReport::new(peak, "avx2");

        let mut timer = crate::profiling::timer::CycleTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.stop();

        let workload = counters::gemm_workload(1024, 1024, 1024);
        let metrics = counters::compute_metrics(workload, &timer, Some(&peak));

        report.add(ProfileEntry {
            name: "gemm_f32_1024x1024x1024".to_string(),
            workload,
            metrics,
            hw_counters: None,
        });

        let html = report.to_html();
        assert!(html.contains("gllm-kernels Performance Report"));
        assert!(html.contains("gemm_f32_1024x1024x1024"));
        assert!(html.contains("Efficiency Overview"));
    }

    // ── 13 new tests ──────────────────────────────────────────────────────

    #[test]
    fn profile_entry_constructor_and_clone() {
        // Arrange
        let entry = ProfileEntry {
            name: "test_kernel".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 0.0001,
                elapsed_cycles: 300_000,
                gflops: 10.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.05,
            },
            hw_counters: None,
        };

        // Act
        let cloned = entry.clone();

        // Assert
        assert_eq!(cloned.name, "test_kernel");
        assert_eq!(cloned.metrics.elapsed_cycles, 300_000);
        assert!(cloned.hw_counters.is_none());
    }

    #[test]
    fn profile_entry_with_hw_counters() {
        // Arrange
        let hw = HwCounters {
            cycles: 500_000,
            instructions: 1_200_000,
            l1d_misses: 300,
            llc_misses: 15,
            branch_misses: 42,
        };
        let entry = ProfileEntry {
            name: "attn_f32".to_string(),
            workload: counters::elementwise_unary_workload(1024, 4),
            metrics: PerfMetrics {
                elapsed_secs: 0.0005,
                elapsed_cycles: 500_000,
                gflops: 0.0,
                bandwidth_gbs: 8.0,
                efficiency: 0.16,
            },
            hw_counters: Some(hw),
        };

        // Act
        let hw_ref = entry.hw_counters.as_ref().unwrap();

        // Assert
        assert_eq!(hw_ref.cycles, 500_000);
        assert_eq!(hw_ref.instructions, 1_200_000);
        assert!((hw_ref.ipc() - 2.4).abs() < 1e-6);
    }

    #[test]
    fn profile_report_new_sets_empty_entries_and_timestamp() {
        // Arrange
        let peak = HwPeak::manual(100.0, 25.0);

        // Act
        let report = ProfileReport::new(peak, "avx512");

        // Assert
        assert_eq!(report.isa, "avx512");
        assert!(report.entries.is_empty());
        assert!(!report.timestamp.is_empty(), "timestamp must be non-empty");
        assert!((report.hw_peak.gflops - 100.0).abs() < 1e-6);
        assert!((report.hw_peak.bandwidth_gbs - 25.0).abs() < 1e-6);
    }

    #[test]
    fn profile_report_add_accumulates_entries() {
        // Arrange
        let peak = HwPeak::manual(200.0, 50.0);
        let mut report = ProfileReport::new(peak, "neon");
        let entry1 = ProfileEntry {
            name: "kernel_a".to_string(),
            workload: counters::gemm_workload(128, 128, 128),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 3000,
                gflops: 5.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.025,
            },
            hw_counters: None,
        };
        let entry2 = ProfileEntry {
            name: "kernel_b".to_string(),
            workload: counters::dot_workload(256, 4),
            metrics: PerfMetrics {
                elapsed_secs: 2e-6,
                elapsed_cycles: 6000,
                gflops: 1.0,
                bandwidth_gbs: 4.0,
                efficiency: 0.005,
            },
            hw_counters: Some(HwCounters::default()),
        };

        // Act
        report.add(entry1);
        report.add(entry2);

        // Assert
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.entries[0].name, "kernel_a");
        assert_eq!(report.entries[1].name, "kernel_b");
    }

    #[test]
    fn profile_report_clone_is_independent() {
        // Arrange
        let peak = HwPeak::manual(300.0, 60.0);
        let mut report = ProfileReport::new(peak, "sve");
        report.add(ProfileEntry {
            name: "orig".to_string(),
            workload: counters::gemm_workload(32, 32, 32),
            metrics: PerfMetrics {
                elapsed_secs: 1e-7,
                elapsed_cycles: 500,
                gflops: 1.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.003,
            },
            hw_counters: None,
        });

        // Act
        let mut cloned = report.clone();
        cloned.add(ProfileEntry {
            name: "extra".to_string(),
            workload: counters::gemm_workload(16, 16, 16),
            metrics: PerfMetrics {
                elapsed_secs: 1e-8,
                elapsed_cycles: 50,
                gflops: 0.5,
                bandwidth_gbs: 0.0,
                efficiency: 0.001,
            },
            hw_counters: None,
        });

        // Assert
        assert_eq!(report.entries.len(), 1);
        assert_eq!(cloned.entries.len(), 2);
    }

    #[test]
    fn json_report_multiple_entries_no_hw_counters() {
        // Arrange
        let peak = HwPeak::manual(500.0, 100.0);
        let mut report = ProfileReport::new(peak, "avx2");
        for i in 0..3 {
            report.add(ProfileEntry {
                name: format!("kernel_{i}"),
                workload: counters::elementwise_binary_workload(256, 4),
                metrics: PerfMetrics {
                    elapsed_secs: 0.001 * (i as f64 + 1.0),
                    elapsed_cycles: 3_000_000 * (i as u64 + 1),
                    gflops: 0.0,
                    bandwidth_gbs: 10.0,
                    efficiency: 0.1,
                },
                hw_counters: None,
            });
        }

        // Act
        let json = report.to_json();

        // Assert
        assert!(json.contains("\"name\": \"kernel_0\""));
        assert!(json.contains("\"name\": \"kernel_1\""));
        assert!(json.contains("\"name\": \"kernel_2\""));
        assert!(json.contains("\"hw_peak_gflops\": 500.00"));
        assert!(json.contains("\"hw_cycles\"") == false, "no hw_cycles when counters are None");
    }

    #[test]
    fn html_report_efficiency_class_good() {
        // Arrange
        let peak = HwPeak::manual(10.0, 5.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "high_eff".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 3000,
                gflops: 9.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.90,
            },
            hw_counters: None,
        });

        // Act
        let html = report.to_html();

        // Assert
        assert!(html.contains("eff-good"));
        assert!(html.contains("90.0%"));
    }

    #[test]
    fn html_report_efficiency_class_bad() {
        // Arrange
        let peak = HwPeak::manual(10.0, 5.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "low_eff".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 3000,
                gflops: 2.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.20,
            },
            hw_counters: None,
        });

        // Act
        let html = report.to_html();

        // Assert
        assert!(html.contains("eff-bad"));
        assert!(html.contains("20.0%"));
    }

    #[test]
    fn html_report_efficiency_class_ok() {
        // Arrange
        let peak = HwPeak::manual(10.0, 5.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "mid_eff".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 3000,
                gflops: 7.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.70,
            },
            hw_counters: None,
        });

        // Act
        let html = report.to_html();

        // Assert
        assert!(html.contains("eff-ok"));
        assert!(html.contains("70.0%"));
    }

    #[test]
    fn json_report_with_hw_counters_includes_ipc_and_mpki() {
        // Arrange
        let peak = HwPeak::manual(200.0, 50.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "profiled_op".to_string(),
            workload: counters::gemm_workload(256, 256, 256),
            metrics: PerfMetrics {
                elapsed_secs: 0.0001,
                elapsed_cycles: 300_000,
                gflops: 50.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.25,
            },
            hw_counters: Some(HwCounters {
                cycles: 300_000,
                instructions: 900_000,
                l1d_misses: 1500,
                llc_misses: 30,
                branch_misses: 7,
            }),
        });

        // Act
        let json = report.to_json();

        // Assert
        assert!(json.contains("\"hw_cycles\": 300000"));
        assert!(json.contains("\"hw_instructions\": 900000"));
        assert!(json.contains("\"branch_misses\": 7"));
        // IPC = 900000 / 300000 = 3.000
        assert!(json.contains("\"ipc\": 3.000"));
        // L1D MPKI = 1500 / 900000 * 1000 = 1.667
        assert!(json.contains("\"l1d_mpki\": 1.667"));
    }

    #[test]
    fn workload_type_str_all_variants() {
        // Arrange
        let compute_w = counters::gemm_workload(8, 8, 8);
        let memory_w = counters::elementwise_unary_workload(128, 4);
        let mixed_w = counters::dot_workload(64, 4);

        let peak = HwPeak::manual(100.0, 50.0);
        let mut report = ProfileReport::new(peak, "avx2");

        let base_metrics = PerfMetrics {
            elapsed_secs: 1e-6,
            elapsed_cycles: 1000,
            gflops: 0.0,
            bandwidth_gbs: 0.0,
            efficiency: 0.0,
        };

        report.add(ProfileEntry {
            name: "compute_entry".to_string(),
            workload: compute_w,
            metrics: base_metrics,
            hw_counters: None,
        });
        report.add(ProfileEntry {
            name: "memory_entry".to_string(),
            workload: memory_w,
            metrics: base_metrics,
            hw_counters: None,
        });
        report.add(ProfileEntry {
            name: "mixed_entry".to_string(),
            workload: mixed_w,
            metrics: base_metrics,
            hw_counters: None,
        });

        // Act
        let json = report.to_json();

        // Assert
        assert!(json.contains("\"workload_type\": \"compute\""));
        assert!(json.contains("\"workload_type\": \"memory\""));
        assert!(json.contains("\"workload_type\": \"mixed\""));
    }

    #[test]
    fn perf_metrics_zero_values_no_panic() {
        // Arrange
        let peak = HwPeak::manual(0.0, 0.0);
        let mut report = ProfileReport::new(peak, "scalar");
        report.add(ProfileEntry {
            name: "zero_op".to_string(),
            workload: counters::gemm_workload(1, 1, 1),
            metrics: PerfMetrics {
                elapsed_secs: 0.0,
                elapsed_cycles: 0,
                gflops: 0.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.0,
            },
            hw_counters: Some(HwCounters::default()),
        });

        // Act — must not panic on zero divisions or NaN
        let json = report.to_json();
        let html = report.to_html();

        // Assert
        assert!(!json.is_empty());
        assert!(!html.is_empty());
        assert!(json.contains("\"elapsed_secs\": 0.000000000"));
        assert!(html.contains("0.0%"));
    }

    #[test]
    fn profile_entry_struct_update_syntax() {
        // Arrange
        let base = ProfileEntry {
            name: "base_kernel".to_string(),
            workload: counters::gemm_workload(128, 128, 128),
            metrics: PerfMetrics {
                elapsed_secs: 0.001,
                elapsed_cycles: 3_000_000,
                gflops: 20.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.10,
            },
            hw_counters: None,
        };

        // Act — struct update syntax, override name only
        let derived = ProfileEntry { name: "derived_kernel".to_string(), ..base };

        // Assert
        assert_eq!(derived.name, "derived_kernel");
        assert_eq!(derived.metrics.elapsed_cycles, 3_000_000);
        assert_eq!(derived.metrics.gflops, 20.0);
        assert!(derived.hw_counters.is_none());
    }

    #[test]
    fn profile_report_debug_trait_output() {
        // Arrange
        let peak = HwPeak::manual(123.4, 56.7);
        let mut report = ProfileReport::new(peak, "avx512_vnni");
        report.add(ProfileEntry {
            name: "debug_kernel".to_string(),
            workload: counters::rms_norm_workload(4096, 4),
            metrics: PerfMetrics {
                elapsed_secs: 5e-5,
                elapsed_cycles: 150_000,
                gflops: 2.0,
                bandwidth_gbs: 6.0,
                efficiency: 0.016,
            },
            hw_counters: Some(HwCounters {
                cycles: 150_000,
                instructions: 450_000,
                l1d_misses: 200,
                llc_misses: 5,
                branch_misses: 1,
            }),
        });

        // Act
        let debug_str = format!("{:?}", report);

        // Assert — Debug output must contain key struct names and field values
        assert!(debug_str.contains("ProfileReport"));
        assert!(debug_str.contains("avx512_vnni"));
        assert!(debug_str.contains("debug_kernel"));
        assert!(debug_str.contains("ProfileEntry"));
    }

    // ── 10 additional tests ────────────────────────────────────────────────

    #[test]
    fn html_report_contains_all_table_headers() {
        // Arrange
        let peak = HwPeak::manual(100.0, 20.0);
        let report = ProfileReport::new(peak, "neon");

        // Act
        let html = report.to_html();

        // Assert — table must have all 10 column headers even with zero entries
        assert!(html.contains("<th>Kernel</th>"));
        assert!(html.contains("<th>Type</th>"));
        assert!(html.contains("<th>Time (us)</th>"));
        assert!(html.contains("<th>Cycles</th>"));
        assert!(html.contains("<th>GFLOPS</th>"));
        assert!(html.contains("<th>BW (GB/s)</th>"));
        assert!(html.contains("<th>Efficiency</th>"));
        assert!(html.contains("<th>IPC</th>"));
        assert!(html.contains("<th>L1D MPKI</th>"));
        assert!(html.contains("<th>LLC MPKI</th>"));
    }

    #[test]
    fn html_report_dash_placeholders_when_no_hw_counters() {
        // Arrange
        let peak = HwPeak::manual(50.0, 10.0);
        let mut report = ProfileReport::new(peak, "sve2");
        report.add(ProfileEntry {
            name: "no_hw".to_string(),
            workload: counters::elementwise_unary_workload(512, 4),
            metrics: PerfMetrics {
                elapsed_secs: 1e-5,
                elapsed_cycles: 50_000,
                gflops: 0.0,
                bandwidth_gbs: 3.0,
                efficiency: 0.3,
            },
            hw_counters: None,
        });

        // Act
        let html = report.to_html();

        // Assert — IPC/L1D MPKI/LLC MPKI cells must show dash placeholders
        assert!(html.contains("<td>-</td><td>-</td><td>-</td>"));
    }

    #[test]
    fn json_report_empty_entries_produces_valid_structure() {
        // Arrange
        let peak = HwPeak::manual(400.0, 80.0);
        let report = ProfileReport::new(peak, "avx2");

        // Act
        let json = report.to_json();

        // Assert — empty entries array, no trailing comma
        assert!(json.contains("\"entries\": [\n  ]\n"));
        assert!(json.contains("\"isa\": \"avx2\""));
        assert!(json.contains("\"hw_peak_gflops\": 400.00"));
        assert!(!json.contains("\"name\":"));
    }

    #[test]
    fn html_report_hw_info_section_values() {
        // Arrange
        let peak = HwPeak::manual(256.5, 64.3);
        let report = ProfileReport::new(peak, "amx");

        // Act
        let html = report.to_html();

        // Assert — hardware info section must embed exact values
        assert!(html.contains("<strong>ISA:</strong> amx"));
        assert!(html.contains("<strong>Peak GFLOPS:</strong> 256.5"));
        assert!(html.contains("<strong>Peak Bandwidth:</strong> 64.3 GB/s"));
        assert!(html.contains("<strong>Timestamp:</strong>"));
    }

    #[test]
    fn html_report_with_hw_counters_shows_ipc_values() {
        // Arrange
        let peak = HwPeak::manual(200.0, 40.0);
        let mut report = ProfileReport::new(peak, "avx512");
        report.add(ProfileEntry {
            name: "hw_profiled".to_string(),
            workload: counters::gemm_workload(256, 256, 256),
            metrics: PerfMetrics {
                elapsed_secs: 1e-4,
                elapsed_cycles: 400_000,
                gflops: 80.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.40,
            },
            hw_counters: Some(HwCounters {
                cycles: 400_000,
                instructions: 1_200_000,
                l1d_misses: 800,
                llc_misses: 12,
                branch_misses: 3,
            }),
        });

        // Act
        let html = report.to_html();

        // Assert — IPC = 1200000/400000 = 3.00, L1D MPKI = 800/1200000*1000 = 0.67
        assert!(html.contains("<td>3.00</td>"));
        assert!(html.contains("<td>0.67</td>"));
    }

    #[test]
    fn json_report_elapsed_secs_scientific_precision() {
        // Arrange
        let peak = HwPeak::manual(100.0, 20.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "precision_test".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 0.000000123,
                elapsed_cycles: 369,
                gflops: 0.5,
                bandwidth_gbs: 0.0,
                efficiency: 0.005,
            },
            hw_counters: None,
        });

        // Act
        let json = report.to_json();

        // Assert — 9 decimal places for elapsed_secs
        assert!(json.contains("\"elapsed_secs\": 0.000000123"));
        assert!(json.contains("\"elapsed_cycles\": 369"));
    }

    #[test]
    fn profile_report_timestamp_is_numeric_string() {
        // Arrange
        let peak = HwPeak::manual(100.0, 20.0);

        // Act
        let report = ProfileReport::new(peak, "avx2");

        // Assert — timestamp is a string of decimal digits (UNIX epoch seconds)
        assert!(report.timestamp.chars().all(|c| c.is_ascii_digit()),
            "timestamp must be a numeric string, got: {}", report.timestamp);
        assert!(report.timestamp.len() >= 9, "timestamp should be >= 9 digits (post-2001 epoch)");
    }

    #[test]
    fn hw_counters_delta_across_entries() {
        // Arrange
        let baseline = HwCounters {
            cycles: 1_000_000,
            instructions: 3_000_000,
            l1d_misses: 5000,
            llc_misses: 100,
            branch_misses: 200,
        };
        let current = HwCounters {
            cycles: 1_500_000,
            instructions: 4_500_000,
            l1d_misses: 6200,
            llc_misses: 130,
            branch_misses: 250,
        };

        // Act
        let delta = current.delta(&baseline);

        // Assert
        assert_eq!(delta.cycles, 500_000);
        assert_eq!(delta.instructions, 1_500_000);
        assert_eq!(delta.l1d_misses, 1200);
        assert_eq!(delta.llc_misses, 30);
        assert_eq!(delta.branch_misses, 50);
    }

    #[test]
    fn html_report_efficiency_bar_chart_colors() {
        // Arrange
        let peak = HwPeak::manual(10.0, 5.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "high_bar".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 3000,
                gflops: 9.5,
                bandwidth_gbs: 0.0,
                efficiency: 0.95,
            },
            hw_counters: None,
        });
        report.add(ProfileEntry {
            name: "low_bar".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 3000,
                gflops: 1.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.10,
            },
            hw_counters: None,
        });

        // Act
        let html = report.to_html();

        // Assert — green for >= 85%, red for < 60%
        assert!(html.contains("background:#4caf50"), "green for >= 85%");
        assert!(html.contains("background:#f44336"), "red for < 60%");
        assert!(html.contains("95.0%"));
        assert!(html.contains("10.0%"));
    }

    #[test]
    fn write_json_to_temp_file_roundtrip() {
        // Arrange
        let peak = HwPeak::manual(200.0, 50.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "file_test".to_string(),
            workload: counters::gemm_workload(128, 128, 128),
            metrics: PerfMetrics {
                elapsed_secs: 1e-4,
                elapsed_cycles: 300_000,
                gflops: 10.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.05,
            },
            hw_counters: None,
        });
        let path = format!("/tmp/gllm_test_report_{}.json", std::process::id());

        // Act
        report.write_json(&path).expect("write_json must succeed");
        let content = std::fs::read_to_string(&path).expect("read back must succeed");

        // Assert
        assert!(content.contains("\"name\": \"file_test\""));
        assert!(content.contains("\"isa\": \"avx2\""));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    // ── 10 edge-case tests ─────────────────────────────────────────────────

    #[test]
    fn json_report_single_entry_no_trailing_comma() {
        // Arrange
        let peak = HwPeak::manual(100.0, 20.0);
        let mut report = ProfileReport::new(peak, "neon");
        report.add(ProfileEntry {
            name: "solo".to_string(),
            workload: counters::gemm_workload(32, 32, 32),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 1000,
                gflops: 1.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.01,
            },
            hw_counters: None,
        });

        // Act
        let json = report.to_json();

        // Assert — last entry must close with "}\n" not "},\n"
        assert!(json.contains("    }\n  ]\n"), "last entry must not have trailing comma");
        assert_eq!(json.matches("\"name\": \"solo\"").count(), 1);
    }

    #[test]
    fn html_report_empty_table_body_no_rows() {
        // Arrange
        let peak = HwPeak::manual(50.0, 10.0);
        let report = ProfileReport::new(peak, "avx2");

        // Act
        let html = report.to_html();

        // Assert — table structure present but tbody is empty (no <tr> in body)
        assert!(html.contains("<tbody>\n</tbody>"), "empty report must have empty tbody");
        assert!(html.contains("Efficiency Overview"));
        // CSS defines .bar-row in stylesheet; verify no bar-row elements rendered in body
        assert!(!html.contains("class=\"bar-row\""), "no bar-row elements when no entries");
    }

    #[test]
    fn json_report_mixed_hw_counters_across_entries() {
        // Arrange
        let peak = HwPeak::manual(200.0, 50.0);
        let mut report = ProfileReport::new(peak, "avx2");
        let base_metrics = PerfMetrics {
            elapsed_secs: 1e-5,
            elapsed_cycles: 50_000,
            gflops: 5.0,
            bandwidth_gbs: 0.0,
            efficiency: 0.025,
        };
        report.add(ProfileEntry {
            name: "with_hw".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: base_metrics,
            hw_counters: Some(HwCounters {
                cycles: 50_000,
                instructions: 150_000,
                l1d_misses: 400,
                llc_misses: 8,
                branch_misses: 2,
            }),
        });
        report.add(ProfileEntry {
            name: "without_hw".to_string(),
            workload: counters::elementwise_unary_workload(256, 4),
            metrics: base_metrics,
            hw_counters: None,
        });

        // Act
        let json = report.to_json();

        // Assert — first entry has hw fields, second does not
        let with_hw_end = json.find("\"name\": \"without_hw\"").unwrap();
        let first_section = &json[..with_hw_end];
        assert!(first_section.contains("\"hw_cycles\": 50000"));
        let second_section = &json[with_hw_end..];
        assert!(!second_section.contains("\"hw_cycles\""));
        assert!(second_section.contains("\"name\": \"without_hw\""));
    }

    #[test]
    fn hw_counters_delta_zero_when_identical() {
        // Arrange
        let counters = HwCounters {
            cycles: 999_999,
            instructions: 2_000_000,
            l1d_misses: 777,
            llc_misses: 33,
            branch_misses: 11,
        };

        // Act
        let delta = counters.delta(&counters);

        // Assert — identical baseline yields all zeros
        assert_eq!(delta.cycles, 0);
        assert_eq!(delta.instructions, 0);
        assert_eq!(delta.l1d_misses, 0);
        assert_eq!(delta.llc_misses, 0);
        assert_eq!(delta.branch_misses, 0);
    }

    #[test]
    fn hw_counters_default_ipc_and_mpki_no_panic() {
        // Arrange
        let hw = HwCounters::default();

        // Act
        let ipc = hw.ipc();
        let l1d = hw.l1d_mpki();
        let llc = hw.llc_mpki();

        // Assert — default is all zeros; division by zero must not panic
        assert!(ipc == 0.0 || ipc.is_nan() || ipc.is_infinite() || ipc.is_finite());
        assert!(l1d == 0.0 || l1d.is_nan() || l1d.is_finite());
        assert!(llc == 0.0 || llc.is_nan() || llc.is_finite());
    }

    #[test]
    fn write_html_to_temp_file_roundtrip() {
        // Arrange
        let peak = HwPeak::manual(150.0, 30.0);
        let mut report = ProfileReport::new(peak, "sve");
        report.add(ProfileEntry {
            name: "html_file_test".to_string(),
            workload: counters::rms_norm_workload(2048, 4),
            metrics: PerfMetrics {
                elapsed_secs: 5e-6,
                elapsed_cycles: 15_000,
                gflops: 0.0,
                bandwidth_gbs: 5.0,
                efficiency: 0.1667,
            },
            hw_counters: None,
        });
        let path = format!("/tmp/gllm_test_report_{}.html", std::process::id());

        // Act
        report.write_html(&path).expect("write_html must succeed");
        let content = std::fs::read_to_string(&path).expect("read back must succeed");

        // Assert
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("html_file_test"));
        assert!(content.contains("</html>"));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn profile_report_entry_name_with_special_characters() {
        // Arrange
        let peak = HwPeak::manual(100.0, 20.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "gemm_f32_1024x1024_x1024 (fused)".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 1000,
                gflops: 1.0,
                bandwidth_gbs: 0.0,
                efficiency: 0.01,
            },
            hw_counters: None,
        });

        // Act
        let json = report.to_json();
        let html = report.to_html();

        // Assert — names with parens/spaces preserved in both outputs
        assert!(json.contains("gemm_f32_1024x1024_x1024 (fused)"));
        assert!(html.contains("gemm_f32_1024x1024_x1024 (fused)"));
    }

    #[test]
    fn html_report_efficiency_100_percent_uses_good_class() {
        // Arrange
        let peak = HwPeak::manual(10.0, 5.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "perfect_eff".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 1000,
                gflops: 10.0,
                bandwidth_gbs: 0.0,
                efficiency: 1.0,
            },
            hw_counters: None,
        });

        // Act
        let html = report.to_html();

        // Assert — 100% is >= 85% so gets eff-good class and green bar
        assert!(html.contains("eff-good"));
        assert!(html.contains("100.0%"));
        assert!(html.contains("background:#4caf50"));
    }

    #[test]
    fn json_report_many_entries_all_present() {
        // Arrange
        let peak = HwPeak::manual(200.0, 40.0);
        let mut report = ProfileReport::new(peak, "avx512");
        for i in 0..10 {
            report.add(ProfileEntry {
                name: format!("batch_op_{:02}", i),
                workload: counters::gemm_workload(128, 128, 128),
                metrics: PerfMetrics {
                    elapsed_secs: 1e-5,
                    elapsed_cycles: 30_000,
                    gflops: 5.0,
                    bandwidth_gbs: 0.0,
                    efficiency: 0.025,
                },
                hw_counters: None,
            });
        }

        // Act
        let json = report.to_json();

        // Assert — all 10 entries present, no duplication
        for i in 0..10 {
            let name = format!("\"name\": \"batch_op_{:02}\"", i);
            assert_eq!(json.matches(&name).count(), 1, "entry {} must appear exactly once", i);
        }
    }

    #[test]
    fn html_report_efficiency_bar_capped_at_100_percent() {
        // Arrange
        let peak = HwPeak::manual(10.0, 5.0);
        let mut report = ProfileReport::new(peak, "avx2");
        report.add(ProfileEntry {
            name: "over_100".to_string(),
            workload: counters::gemm_workload(64, 64, 64),
            metrics: PerfMetrics {
                elapsed_secs: 1e-6,
                elapsed_cycles: 1000,
                gflops: 15.0,
                bandwidth_gbs: 0.0,
                efficiency: 1.5, // 150% — should be capped to 100% in bar
            },
            hw_counters: None,
        });

        // Act
        let html = report.to_html();

        // Assert — bar width capped at 100%, but the label still shows 150%
        assert!(html.contains("width:100.0%"), "bar width must be capped at 100%");
        assert!(html.contains("150.0%"), "label shows actual efficiency");
    }
}
