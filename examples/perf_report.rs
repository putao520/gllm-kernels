#!/usr/bin/env -S cargo +nightly -Zscript
//! gllm-kernels Baseline Performance Audit Report
//! Generated from criterion benchmark results on i9-10900KF

fn main() {
    // ========================================================================
    // Hardware Parameters
    // ========================================================================
    let cpu = "Intel i9-10900KF";
    let freq_ghz = 4.9_f64;       // turbo single-core
    let cores = 10_usize;
    let fma_ports = 2_usize;       // Comet Lake has 2 FMA units on port 0+1
    let avx2_lanes = 8_usize;      // 256-bit / 32-bit
    let fp32_peak_1core = (fma_ports * avx2_lanes * 2) as f64 * freq_ghz; // 156.8 GFLOPS
    let fp32_peak_all = fp32_peak_1core * cores as f64;
    let dram_bw_theoretical = 51.2_f64; // GB/s, DDR4-3200 dual channel
    let dram_bw_practical = 40.0_f64;   // ~78% of theoretical is typical

    let sep = "=".repeat(90);
    let thin = "-".repeat(90);

    println!("{sep}");
    println!("  gllm-kernels BASELINE PERFORMANCE AUDIT");
    println!("{sep}");
    println!("  CPU:           {cpu} @ 3.70 GHz (turbo {freq_ghz} GHz)");
    println!("  Cores:         {cores} physical / {} threads", cores * 2);
    println!("  ISA:           AVX2 + FMA (no AVX-512)");
    println!("  Cache:         L1D=32KB, L2=256KB, L3=20MB");
    println!("  FP32 Peak:     {fp32_peak_1core:.1} GFLOPS (1 core) / {fp32_peak_all:.0} GFLOPS ({cores} cores)");
    println!("  DRAM BW:       {dram_bw_theoretical} GB/s theoretical / ~{dram_bw_practical} GB/s practical");
    println!("{sep}\n");

    // ========================================================================
    // Memory-Bound Operators
    // ========================================================================
    // Criterion reports GiB/s. Convert to GB/s: multiply by 1.0737
    let gib_to_gb = 1.073741824_f64;

    println!("  MEMORY-BOUND OPERATORS (bandwidth efficiency)");
    println!("{thin}");
    println!("  {:30} {:>8} {:>8} {:>10} {:>8} {:>8}",
             "Operator", "Size", "GiB/s", "GB/s", "Eff%", "Rating");
    println!("{thin}");

    struct MemResult {
        name: &'static str,
        size: &'static str,
        gib_s: f64,
        // For small sizes that fit in cache, the "limit" is cache BW, not DRAM.
        // We'll flag cache-resident results separately.
        cache_resident: bool,
    }

    let mem_results = vec![
        // vec_dot
        MemResult { name: "vec_dot",       size: "1K",   gib_s: 136.0, cache_resident: true },
        MemResult { name: "vec_dot",       size: "4K",   gib_s: 182.0, cache_resident: true },
        MemResult { name: "vec_dot",       size: "64K",  gib_s: 61.0,  cache_resident: false },
        MemResult { name: "vec_dot",       size: "1M",   gib_s: 67.0,  cache_resident: false },
        // vec_add
        MemResult { name: "vec_add",       size: "1K",   gib_s: 256.0, cache_resident: true },
        MemResult { name: "vec_add",       size: "4K",   gib_s: 116.0, cache_resident: true },
        MemResult { name: "vec_add",       size: "64K",  gib_s: 65.0,  cache_resident: false },
        MemResult { name: "vec_add",       size: "1M",   gib_s: 52.0,  cache_resident: false },
        // vec_axpy
        MemResult { name: "vec_axpy",      size: "4K",   gib_s: 234.0, cache_resident: true },
        MemResult { name: "vec_axpy",      size: "64K",  gib_s: 86.0,  cache_resident: false },
        // silu
        MemResult { name: "silu",          size: "4K",   gib_s: 9.1,   cache_resident: true },
        MemResult { name: "silu",          size: "32K",  gib_s: 9.3,   cache_resident: true },
        // softmax
        MemResult { name: "softmax",       size: "1K",   gib_s: 8.3,   cache_resident: true },
        MemResult { name: "softmax",       size: "4K",   gib_s: 8.7,   cache_resident: true },
        MemResult { name: "softmax",       size: "32K",  gib_s: 8.6,   cache_resident: true },
        // exp
        MemResult { name: "exp",           size: "4K",   gib_s: 12.2,  cache_resident: true },
        MemResult { name: "exp",           size: "32K",  gib_s: 12.6,  cache_resident: true },
        // rms_norm
        MemResult { name: "rms_norm",      size: "1K",   gib_s: 176.0, cache_resident: true },
        MemResult { name: "rms_norm",      size: "4K",   gib_s: 120.0, cache_resident: true },
        MemResult { name: "rms_norm",      size: "8K",   gib_s: 119.0, cache_resident: true },
        // rope
        MemResult { name: "rope",          size: "1tok", gib_s: 8.8,   cache_resident: true },
        MemResult { name: "rope",          size: "32tok",gib_s: 5.0,   cache_resident: false },
        // dequant
        MemResult { name: "dequant_q4k",   size: "256",  gib_s: 41.0,  cache_resident: true },
        MemResult { name: "dequant_q4k",   size: "4K",   gib_s: 41.0,  cache_resident: true },
        MemResult { name: "dequant_q8k",   size: "256",  gib_s: 92.0,  cache_resident: true },
        MemResult { name: "dequant_q8k",   size: "4K",   gib_s: 87.0,  cache_resident: true },
        // gelu
        MemResult { name: "gelu",          size: "4K",   gib_s: 7.7,   cache_resident: true },
        MemResult { name: "gelu",          size: "32K",  gib_s: 7.8,   cache_resident: true },
        // swiglu
        MemResult { name: "swiglu",        size: "4K",   gib_s: 13.2,  cache_resident: true },
        MemResult { name: "swiglu",        size: "11K",  gib_s: 13.3,  cache_resident: true },
        // layer_norm
        MemResult { name: "layer_norm",    size: "1K",   gib_s: 105.0, cache_resident: true },
        MemResult { name: "layer_norm",    size: "4K",   gib_s: 86.0,  cache_resident: true },
        MemResult { name: "layer_norm",    size: "8K",   gib_s: 81.0,  cache_resident: true },
    ];

    for r in &mem_results {
        let gb_s = r.gib_s * gib_to_gb;
        // For DRAM-bound sizes, compare vs practical DRAM BW
        // For cache-resident, the limit is much higher, so we note it differently
        let (eff, rating) = if r.cache_resident {
            // Cache-resident: these numbers are expected to be high.
            // Compare against a rough L1/L2 BW estimate (~200 GB/s for L1, ~100 GB/s for L2)
            // But for compute-limited ops (silu/gelu/softmax/exp), the bottleneck is ALU, not BW.
            let is_compute_limited = ["silu", "softmax", "exp", "gelu", "rope", "swiglu"].contains(&r.name);
            if is_compute_limited {
                // These are ALU-bound (transcendentals), not memory-bound
                // Rate them by how well they use the ALU
                (gb_s, "ALU-LIM")
            } else {
                // Pure memory ops in cache: very high BW expected
                (gb_s / 200.0 * 100.0, "CACHE")
            }
        } else {
            // DRAM-bound
            let eff = gb_s / dram_bw_practical * 100.0;
            let rating = if eff >= 90.0 { "GREEN" }
                        else if eff >= 70.0 { "YELLOW" }
                        else { "RED" };
            (eff, rating)
        };

        let eff_str = if ["ALU-LIM"].contains(&rating) {
            format!("{:.1}GB/s", gb_s)
        } else {
            format!("{:.1}%", eff)
        };

        println!("  {:30} {:>8} {:>7.1} {:>9.1} {:>8} {:>8}",
                 r.name, r.size, r.gib_s, gb_s, eff_str, rating);
    }

    // ========================================================================
    // Compute-Bound Operators
    // ========================================================================
    println!("\n\n  COMPUTE-BOUND OPERATORS (FLOPS efficiency vs {fp32_peak_1core:.1} GFLOPS single-core peak)");
    println!("{thin}");
    println!("  {:40} {:>10} {:>10} {:>8} {:>8}",
             "Operator", "GFLOPS", "Eff%", "Rating", "Notes");
    println!("{thin}");

    struct ComputeResult {
        name: &'static str,
        gflops: f64,
        uses_mt: bool,  // multi-threaded?
        notes: &'static str,
    }

    let compute_results = vec![
        // GEMM (multi-threaded for large sizes)
        ComputeResult { name: "gemm 128x128x128",           gflops: 105.0,  uses_mt: false, notes: "small, setup overhead" },
        ComputeResult { name: "gemm 256x256x256",           gflops: 258.0,  uses_mt: true,  notes: "MT kicks in" },
        ComputeResult { name: "gemm 512x512x512",           gflops: 494.0,  uses_mt: true,  notes: "" },
        ComputeResult { name: "gemm 1024x1024x1024",        gflops: 591.0,  uses_mt: true,  notes: "" },
        ComputeResult { name: "gemm 2048x2048x2048",        gflops: 630.0,  uses_mt: true,  notes: "best case" },
        // GEMM prepacked
        ComputeResult { name: "gemm_prepacked 128x128x128",  gflops: 205.0,  uses_mt: false, notes: "2x vs unpacked" },
        ComputeResult { name: "gemm_prepacked 256x256x256",  gflops: 347.0,  uses_mt: true,  notes: "" },
        ComputeResult { name: "gemm_prepacked 512x512x512",  gflops: 515.0,  uses_mt: true,  notes: "" },
        ComputeResult { name: "gemm_prepacked 1024x1024x1024",gflops: 610.0, uses_mt: true,  notes: "" },
        ComputeResult { name: "gemm_prepacked 2048x2048x2048",gflops: 673.0, uses_mt: true,  notes: "best case" },
        // GEMV (memory-bound in practice)
        ComputeResult { name: "gemv 4096x4096",              gflops: 18.4,   uses_mt: true,  notes: "mem-bound" },
        ComputeResult { name: "gemv 4096x11008",             gflops: 16.8,   uses_mt: true,  notes: "mem-bound" },
        // Quantized dot
        ComputeResult { name: "gemv_q4k dot 4096",           gflops: 13.7,   uses_mt: false, notes: "single row" },
        ComputeResult { name: "gemv_q8k dot 4096",           gflops: 21.5,   uses_mt: false, notes: "single row" },
    ];

    for r in &compute_results {
        let peak = if r.uses_mt { fp32_peak_all } else { fp32_peak_1core };
        let eff = r.gflops / peak * 100.0;
        let rating = if r.notes.contains("mem-bound") {
            "MEM-LIM"
        } else if eff >= 90.0 {
            "GREEN"
        } else if eff >= 70.0 {
            "YELLOW"
        } else if eff >= 50.0 {
            "ORANGE"
        } else {
            "RED"
        };

        println!("  {:40} {:>9.1} {:>9.1}% {:>8} {:>8}",
                 r.name, r.gflops, eff, rating, r.notes);
    }

    // ========================================================================
    // Summary & Bottleneck Analysis
    // ========================================================================
    println!("\n\n  EFFICIENCY SUMMARY");
    println!("{thin}");

    println!("
  GEMM (compute-bound, the most critical operator):
    Best single-core:  105 GFLOPS @ 128x128  = {:.1}% of {:.1} GFLOPS peak
    Best multi-core:   673 GFLOPS @ 2048x2048 prepacked = {:.1}% of {:.0} GFLOPS peak
    Scaling:           673 / 156.8 = {:.1}x across {cores} cores ({:.1}% parallel efficiency)

  GEMV (memory-bound, critical for LLM decode):
    4096x4096:         18.4 GFLOPS
    Arithmetic intensity: 2 FLOPS/8 bytes = 0.25 FLOP/byte
    Expected peak at 40 GB/s DRAM: 40 * 0.25 = 10 GFLOPS
    Measured 18.4 GFLOPS > 10 GFLOPS => benefiting from L3 cache (matrix fits in 20MB L3 partially)

  Activation functions (ALU-limited, transcendental-heavy):
    silu:    ~9.3 GiB/s  (exp + div bottleneck)
    gelu:    ~7.8 GiB/s  (tanh + mul chain)
    softmax: ~8.6 GiB/s  (2-pass: max+exp+sum, then div)
    exp:     ~12.5 GiB/s (fastest transcendental)
    swiglu:  ~13.3 GiB/s (silu + mul, better than standalone silu due to fusion)

  Normalization (well-optimized, single-pass fused):
    rms_norm:   119-176 GiB/s (excellent, single-pass fused)
    layer_norm: 81-105 GiB/s  (good, 2-pass for mean+var)

  Dequantization:
    Q4_K decode: 41 GiB/s  (good, bit-unpacking + scale)
    Q8_K decode: 87 GiB/s  (excellent, simpler format)

  Quantized dot products:
    Q4_K dot 4096: 13.7 GFLOPS (on-the-fly dequant + FMA)
    Q8_K dot 4096: 21.5 GFLOPS (simpler dequant, 1.6x faster)
",
        105.0 / fp32_peak_1core * 100.0,
        fp32_peak_1core,
        673.0 / fp32_peak_all * 100.0,
        fp32_peak_all,
        673.0 / fp32_peak_1core,
        673.0 / fp32_peak_all * 100.0 * (cores as f64) / (673.0 / fp32_peak_1core),
    );

    println!("  TOP BOTTLENECKS (priority order for optimization)");
    println!("{thin}");
    println!("
  1. [RED]    GEMM 128x128 single-core: 105 GFLOPS = {:.1}% peak
              Root cause: setup/dispatch overhead dominates at small sizes.
              Impact: Prefill with short sequences, attention score matmul.
              Fix: Reduce dispatch overhead, consider inline microkernel for small M.

  2. [RED]    RoPE 32-token: 5.0 GiB/s
              Root cause: Likely non-contiguous memory access pattern (head-interleaved).
              Impact: Every prefill step.
              Fix: Investigate memory layout, consider head-contiguous RoPE.

  3. [ORANGE] Activation functions (silu/gelu/softmax): 7.8-9.3 GiB/s
              Root cause: Transcendental functions (exp, tanh) are ALU-bound.
              Impact: Every FFN layer.
              Fix: Polynomial approximation for exp/tanh (already done?), check vectorization.

  4. [YELLOW] GEMM 2048x2048 multi-core: 673 GFLOPS = {:.1}% of {:.0} GFLOPS peak
              This is actually decent for a self-implemented GEMM.
              For reference, MKL typically achieves ~85-90% on this hardware.
              Fix: Further tuning of MC/KC blocking, prefetch distances, pack_a.

  5. [GREEN]  vec_dot/vec_add/rms_norm/layer_norm: All excellent at cache-resident sizes.
              These are well-optimized and not bottlenecks.
",
        105.0 / fp32_peak_1core * 100.0,
        673.0 / fp32_peak_all * 100.0,
        fp32_peak_all,
    );

    println!("{sep}");
    println!("  Benchmark file: benches/baseline_audit.rs");
    println!("  Run command: RUSTFLAGS=\"-C target-cpu=native\" cargo bench --bench baseline_audit");
    println!("{sep}");
}
