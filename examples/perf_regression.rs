//! Performance regression test tool.
//!
//! Usage:
//!   cargo run --release --example perf_regression           # run and display
//!   cargo run --release --example perf_regression -- --save # save baseline
//!   cargo run --release --example perf_regression -- --compare # compare vs baseline

use gllm_kernels::cpu_kernels::{get_isa_level, CpuKernels};
use gllm_kernels::traits::Kernels;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn median_time(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup { f(); }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iters / 2]
}

// ---------------------------------------------------------------------------
// CPU detection
// ---------------------------------------------------------------------------

fn detect_freq_ghz() -> f64 {
    // Try /proc/cpuinfo
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in s.lines() {
            if line.starts_with("cpu MHz") {
                if let Some(v) = line.split(':').nth(1) {
                    if let Ok(mhz) = v.trim().parse::<f64>() {
                        return mhz / 1000.0;
                    }
                }
            }
        }
    }
    3.7 // fallback
}

fn detect_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn fma_width() -> usize {
    match get_isa_level() {
        #[cfg(target_arch = "x86_64")]
        gllm_kernels::cpu_kernels::IsaLevel::Avx512 => 16,
        #[cfg(target_arch = "x86_64")]
        gllm_kernels::cpu_kernels::IsaLevel::Avx2 => 8,
        _ => 4,
    }
}

fn fma_ports() -> usize {
    // Most modern x86 have 2 FMA ports; NEON has 2 FMLA pipes on A76+
    2
}

struct HwParams {
    freq_ghz: f64,
    cores: usize,
    lanes: usize,
    ports: usize,
}

impl HwParams {
    fn detect() -> Self {
        Self {
            freq_ghz: detect_freq_ghz(),
            cores: detect_cores(),
            lanes: fma_width(),
            ports: fma_ports(),
        }
    }
    fn peak_gflops_1core(&self) -> f64 {
        (self.ports * self.lanes * 2) as f64 * self.freq_ghz
    }
    fn peak_gflops_all(&self) -> f64 {
        self.peak_gflops_1core() * self.cores as f64
    }
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BenchResult {
    name: String,
    metric: f64,      // GFLOPS or GB/s
    unit: &'static str,
    efficiency: f64,   // percent of theoretical peak
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_gemm(hw: &HwParams) -> Vec<BenchResult> {
    let kernels = CpuKernels::<f32>::new();
    let shapes: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("gemm 256x256x256",       256,  256,  256,  5, 100),
        ("gemm 512x512x512",       512,  512,  512,  3, 30),
        ("gemm 1024x1024x1024",    1024, 1024, 1024, 2, 10),
        ("gemm 128x4096x4096",     128,  4096, 4096, 2, 5),
        ("gemv 1x4096x4096",       1,    4096, 4096, 5, 30),
    ];
    let mut results = Vec::new();
    for &(name, m, n, k, warmup, iters) in shapes {
        let a: Vec<f32> = (0..m*k).map(|i| (i % 97) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i % 89) as f32 * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        let secs = median_time(warmup, iters, || {
            kernels.gemm(&a, &b, &mut c, m, n, k);
        });
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
        let peak = hw.peak_gflops_all();
        results.push(BenchResult {
            name: name.to_string(),
            metric: gflops,
            unit: "GFLOPS",
            efficiency: gflops / peak * 100.0,
        });
    }
    results
}

fn bench_quant_gemv(_hw: &HwParams) -> Vec<BenchResult> {
    let kernels = CpuKernels::<f32>::new();
    let k = 4096usize;   // input features
    let m = 4096usize;   // output features (weight rows)
    let n = 1usize;      // batch size
    let mut results = Vec::new();

    // Q4_K GEMV via gemm_q4 (m=batch, n=out_features, k=in_features — different convention)
    {
        let block_size = std::mem::size_of::<gllm_kernels::quant::BlockQ4K>();
        let blocks_per_row = k / 256;
        let weight = vec![0x55u8; m * blocks_per_row * block_size];
        let input = vec![1.0f32; n * k]; // [batch x k]
        let scales = vec![1.0f32; m];
        let mut output = vec![0.0f32; n * m];
        let secs = median_time(3, 20, || {
            kernels.gemm_q4(&weight, &input, &mut output, &scales, n, m, k);
        });
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
        let weight_bytes = (m * blocks_per_row * block_size) as f64;
        let input_bytes = (n * k * 4) as f64;
        let gb_s = (weight_bytes + input_bytes) / secs / 1e9;
        results.push(BenchResult {
            name: "gemv_q4k 1x4096x4096".to_string(),
            metric: gflops,
            unit: "GFLOPS",
            efficiency: 0.0,
        });
        results.push(BenchResult {
            name: "gemv_q4k 1x4096x4096 BW".to_string(),
            metric: gb_s,
            unit: "GB/s",
            efficiency: 0.0,
        });
    }

    // Q8_K GEMV via kquant_matmul (m=weight_rows, n=batch, k=shared_dim)
    {
        let block_size = std::mem::size_of::<gllm_kernels::quant::BlockQ8K>();
        let blocks_per_row = k / 256;
        let weight_bytes_total = m * blocks_per_row * block_size;
        let weight_raw = vec![0u8; weight_bytes_total];
        let input = vec![1.0f32; k * n]; // [k x n]
        let mut output = vec![0.0f32; m * n];
        let secs = median_time(3, 20, || {
            kernels.kquant_matmul(&weight_raw, &input, &mut output,
                gllm_kernels::quant::QuantType::Q8K, m, n, k);
        });
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
        let bw = (weight_bytes_total as f64 + (k * n * 4) as f64) / secs / 1e9;
        results.push(BenchResult {
            name: "gemv_q8k 1x4096x4096".to_string(),
            metric: gflops,
            unit: "GFLOPS",
            efficiency: 0.0,
        });
        results.push(BenchResult {
            name: "gemv_q8k 1x4096x4096 BW".to_string(),
            metric: bw,
            unit: "GB/s",
            efficiency: 0.0,
        });
    }

    results
}

fn bench_memory_ops(_hw: &HwParams) -> Vec<BenchResult> {
    let kernels = CpuKernels::<f32>::new();
    let dim = 4096usize;
    let mut results = Vec::new();

    // rms_norm
    {
        let input = vec![1.0f32; dim];
        let weight = vec![1.0f32; dim];
        let mut output = vec![0.0f32; dim];
        let secs = median_time(5, 100, || {
            kernels.rms_norm(&input, &weight, &mut output, 1e-5);
        });
        let bytes = (dim * 4 * 3) as f64; // read input + weight, write output
        let gb_s = bytes / secs / 1e9;
        results.push(BenchResult {
            name: "rms_norm 4096".to_string(),
            metric: gb_s,
            unit: "GB/s",
            efficiency: 0.0,
        });
    }

    // softmax
    {
        let data = (0..dim).map(|i| (i % 100) as f32 * 0.01).collect::<Vec<_>>();
        let mut out = vec![0.0f32; dim];
        let secs = median_time(5, 100, || {
            kernels.softmax(&data, &mut out);
        });
        let bytes = (dim * 4 * 3) as f64; // 2 passes read + 1 write
        let gb_s = bytes / secs / 1e9;
        results.push(BenchResult {
            name: "softmax 4096".to_string(),
            metric: gb_s,
            unit: "GB/s",
            efficiency: 0.0,
        });
    }

    // rope
    {
        let head_dim = 128usize;
        let n_heads = 32usize;
        let total = n_heads * head_dim;
        let mut qk = vec![1.0f32; total];
        let cos = vec![1.0f32; head_dim / 2];
        let sin = vec![0.0f32; head_dim / 2];
        let secs = median_time(5, 100, || {
            kernels.rope(&mut qk, &cos, &sin, head_dim, false);
        });
        let bytes = (total * 4 * 2) as f64; // read + write qk
        let gb_s = bytes / secs / 1e9;
        results.push(BenchResult {
            name: "rope 32heads dim128".to_string(),
            metric: gb_s,
            unit: "GB/s",
            efficiency: 0.0,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// JSON baseline (hand-rolled, no serde)
// ---------------------------------------------------------------------------

fn results_to_json(results: &[BenchResult]) -> String {
    let mut s = String::from("{\n  \"results\": [\n");
    for (i, r) in results.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"metric\": {:.4}, \"unit\": \"{}\"}}",
            r.name, r.metric, r.unit
        ));
        if i + 1 < results.len() { s.push(','); }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

fn load_baseline(path: &str) -> Option<Vec<BenchResult>> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut results = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if !line.starts_with("{\"name\"") { continue; }
        let line = line.trim_end_matches(',');
        // Parse: {"name": "...", "metric": ..., "unit": "..."}
        let name = extract_json_str(line, "name")?;
        let metric = extract_json_num(line, "metric")?;
        let unit = extract_json_str(line, "unit")?;
        let unit_static: &'static str = match unit.as_str() {
            "GFLOPS" => "GFLOPS",
            "GB/s" => "GB/s",
            _ => "?",
        };
        results.push(BenchResult { name, metric, unit: unit_static, efficiency: 0.0 });
    }
    Some(results)
}

fn extract_json_str(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\":", key);
    let idx = line.find(&pat)? + pat.len();
    let rest = &line[idx..].trim_start();
    if !rest.starts_with('"') { return None; }
    let end = rest[1..].find('"')? + 1;
    Some(rest[1..end].to_string())
}

fn extract_json_num(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\":", key);
    let idx = line.find(&pat)? + pat.len();
    let rest = line[idx..].trim_start();
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let save = args.iter().any(|a| a == "--save");
    let compare = args.iter().any(|a| a == "--compare");
    let baseline_path = "benches/baseline.json";

    let hw = HwParams::detect();
    let isa = get_isa_level();

    println!("=== gllm-kernels Performance Regression Test ===");
    println!("ISA: {:?}  Cores: {}  Freq: {:.1} GHz  FMA: {}x{}",
             isa, hw.cores, hw.freq_ghz, hw.ports, hw.lanes);
    println!("Peak: {:.1} GFLOPS (1-core)  {:.0} GFLOPS (all-core)",
             hw.peak_gflops_1core(), hw.peak_gflops_all());
    println!("Rayon threads: {}", rayon::current_num_threads());
    println!();

    let baseline = if compare { load_baseline(baseline_path) } else { None };

    let mut all_results = Vec::new();

    // GEMM
    println!("{:<42} {:>10} {:>6} {:>8}", "Benchmark", "Result", "Unit", "Eff%");
    println!("{}", "-".repeat(70));

    let gemm = bench_gemm(&hw);
    for r in &gemm {
        print_result(r, &baseline);
    }
    all_results.extend(gemm);

    println!();
    let quant = bench_quant_gemv(&hw);
    for r in &quant {
        print_result(r, &baseline);
    }
    all_results.extend(quant);

    println!();
    let mem = bench_memory_ops(&hw);
    for r in &mem {
        print_result(r, &baseline);
    }
    all_results.extend(mem);

    if save {
        let json = results_to_json(&all_results);
        std::fs::create_dir_all("benches").ok();
        std::fs::write(baseline_path, &json).expect("failed to write baseline");
        println!("\nBaseline saved to {baseline_path}");
    }
}

fn print_result(r: &BenchResult, baseline: &Option<Vec<BenchResult>>) {
    let eff_str = if r.efficiency > 0.0 {
        format!("{:.1}%", r.efficiency)
    } else {
        "—".to_string()
    };

    let delta = baseline.as_ref().and_then(|bl| {
        bl.iter().find(|b| b.name == r.name).map(|b| {
            let pct = (r.metric - b.metric) / b.metric * 100.0;
            if pct >= 0.0 { format!(" +{:.1}%", pct) } else { format!(" {:.1}%", pct) }
        })
    }).unwrap_or_default();

    println!("{:<42} {:>8.1} {:>6} {:>8}{}", r.name, r.metric, r.unit, eff_str, delta);
}
