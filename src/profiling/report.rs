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
}
