use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use gllm_kernels::cpu_kernels::CpuKernels;
use gllm_kernels::traits::Kernels;
use rand::Rng;

fn rand_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

// ============================================================
// GEMV: LLM decode shapes (M=1)
// ============================================================
fn bench_gemv(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("gemv_llm");

    // Typical LLM decode: single-token GEMV through weight matrices
    // Llama-7B: hidden=4096, ffn=11008
    // Llama-13B: hidden=5120, ffn=13824
    for &(m, k) in &[
        (4096, 4096),   // Q/K/V projection
        (4096, 11008),  // gate/up projection
        (11008, 4096),  // down projection
        (5120, 13824),  // Llama-13B gate/up
    ] {
        let mat = rand_vec(m * k);
        let x = rand_vec(k);
        let mut out = vec![0.0f32; m];
        group.throughput(Throughput::Elements((2 * m * k) as u64));
        group.bench_function(BenchmarkId::new("gemv", format!("{m}x{k}")), |bench| {
            bench.iter(|| {
                kernels.gemv(black_box(&mat), black_box(&x), black_box(&mut out), m, k);
            })
        });
    }
    group.finish();
}

// ============================================================
// Skinny GEMM: LLM prefill shapes (M=2..32)
// ============================================================
fn bench_skinny_gemm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("skinny_gemm_llm");
    group.sample_size(20);

    // Prefill: small batch of tokens through weight matrices
    for &(m, n, k) in &[
        (4, 4096, 4096),    // 4-token prefill, Q projection
        (8, 4096, 4096),    // 8-token prefill
        (16, 4096, 4096),   // 16-token prefill
        (32, 4096, 4096),   // 32-token prefill
        (16, 11008, 4096),  // 16-token gate/up
        (32, 11008, 4096),  // 32-token gate/up
    ] {
        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let mut out = vec![0.0f32; m * n];
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));
        group.bench_function(BenchmarkId::new("gemm", format!("{m}x{n}x{k}")), |bench| {
            bench.iter(|| {
                kernels.gemm(black_box(&a), black_box(&b), black_box(&mut out), m, n, k);
            })
        });
    }
    group.finish();
}

// ============================================================
// Large GEMM: square and transformer shapes
// ============================================================
fn bench_large_gemm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("large_gemm");
    group.sample_size(10);

    for &(m, n, k) in &[
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (128, 4096, 4096),   // 128-token prefill
        (256, 4096, 4096),   // 256-token prefill
    ] {
        let a = rand_vec(m * k);
        let b = rand_vec(k * n);
        let mut out = vec![0.0f32; m * n];
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));
        group.bench_function(BenchmarkId::new("gemm", format!("{m}x{n}x{k}")), |bench| {
            bench.iter(|| {
                kernels.gemm(black_box(&a), black_box(&b), black_box(&mut out), m, n, k);
            })
        });

        // Prepacked variant
        let packed_b = kernels.pack_b(&b, n, k);
        group.bench_function(BenchmarkId::new("prepacked", format!("{m}x{n}x{k}")), |bench| {
            bench.iter(|| {
                kernels.gemm_prepacked(
                    black_box(&a), black_box(&packed_b), black_box(&mut out), m, n, k,
                );
            })
        });
    }
    group.finish();
}

// ============================================================
// Softmax: online (2-pass) vs 3-pass comparison
// ============================================================
// Quantized GEMV: Q4_K and Q8_K at LLM sizes
// ============================================================
fn bench_quant_gemv(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("quant_gemv");
    group.sample_size(20);
    let mut rng = rand::thread_rng();

    // Q4_K dot products
    let q4_block_bytes = std::mem::size_of::<gllm_kernels::quant::BlockQ4K>();
    for &n in &[4096, 11008] {
        let blocks = n / 256;
        let weight = vec![0u8; blocks * q4_block_bytes];
        let x = rand_vec(n);
        group.throughput(Throughput::Elements((2 * n) as u64));
        group.bench_function(BenchmarkId::new("q4k_dot", n), |bench| {
            bench.iter(|| {
                black_box(kernels.gemv_q4(black_box(&weight), black_box(&x), 1.0, n))
            })
        });
    }

    // Q8_K dot products
    let q8_block_bytes = std::mem::size_of::<gllm_kernels::quant::BlockQ8K>();
    for &n in &[4096, 11008] {
        let blocks = n / 256;
        let mut weight_raw = vec![0u8; blocks * q8_block_bytes];
        for b in 0..blocks {
            let base = b * q8_block_bytes;
            let d: f32 = rng.gen_range(0.01..0.1);
            weight_raw[base..base + 4].copy_from_slice(&d.to_le_bytes());
            for j in 4..q8_block_bytes {
                weight_raw[base + j] = rng.gen::<u8>();
            }
        }
        let weight = unsafe {
            std::slice::from_raw_parts(weight_raw.as_ptr() as *const i8, weight_raw.len())
        };
        let x = rand_vec(n);
        group.throughput(Throughput::Elements((2 * n) as u64));
        group.bench_function(BenchmarkId::new("q8k_dot", n), |bench| {
            bench.iter(|| {
                black_box(kernels.gemv_q8(black_box(weight), black_box(&x), 1.0, n))
            })
        });
    }

    // kquant_matmul: multi-row Q8K GEMM (simulates batched decode)
    // Semantics: output[m,n] = weight[m,k] · input[k,n]
    // weight is [m, k] quantized, input is [k, n] col-major
    for &(m, n, k) in &[(1, 1, 4096), (1, 4, 4096), (4, 1, 4096)] {
        let blocks = k / 256;
        let mut weight_raw = vec![0u8; m * blocks * q8_block_bytes];
        for i in 0..(m * blocks) {
            let base = i * q8_block_bytes;
            let d: f32 = 0.05;
            weight_raw[base..base + 4].copy_from_slice(&d.to_le_bytes());
            for j in 4..q8_block_bytes {
                weight_raw[base + j] = rng.gen::<u8>();
            }
        }
        // input is [k, n] col-major: n column vectors of length k
        let input = rand_vec(k * n);
        let mut output = vec![0.0f32; m * n];
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));
        group.bench_function(
            BenchmarkId::new("kquant_q8k", format!("{m}x{n}x{k}")),
            |bench| {
                bench.iter(|| {
                    kernels.kquant_matmul(
                        black_box(&weight_raw), black_box(&input), black_box(&mut output),
                        gllm_kernels::quant::QuantType::Q8K, m, n, k,
                    );
                })
            },
        );
    }

    // kquant_matmul: multi-row Q4K GEMM (simulates batched decode)
    for &(m, n, k) in &[(1, 1, 4096), (4, 1, 4096)] {
        let blocks = k / 256;
        let mut weight_raw = vec![0u8; m * blocks * q4_block_bytes];
        for i in 0..(m * blocks) {
            let base = i * q4_block_bytes;
            // d as f16 at offset 0
            let d_f16 = half::f16::from_f32(0.05);
            weight_raw[base..base + 2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
            for j in 2..q4_block_bytes {
                weight_raw[base + j] = rng.gen::<u8>();
            }
        }
        let input = rand_vec(k * n);
        let mut output = vec![0.0f32; m * n];
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));
        group.bench_function(
            BenchmarkId::new("kquant_q4k", format!("{m}x{n}x{k}")),
            |bench| {
                bench.iter(|| {
                    kernels.kquant_matmul(
                        black_box(&weight_raw), black_box(&input), black_box(&mut output),
                        gllm_kernels::quant::QuantType::Q4K, m, n, k,
                    );
                })
            },
        );
    }

    group.finish();
}
fn bench_swiglu(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("swiglu_llm");

    for &n in &[4096, 11008, 13824] {
        let gate = rand_vec(n);
        let up = rand_vec(n);
        let mut out = vec![0.0f32; n];
        // Read gate + up, write out = 3*n*4 bytes
        group.throughput(Throughput::Bytes((3 * n * 4) as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |bench| {
            bench.iter(|| kernels.swiglu(black_box(&gate), black_box(&up), black_box(&mut out)))
        });
    }
    group.finish();
}

// ============================================================
// Activations at LLM sizes
// ============================================================
fn bench_activations(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("activations_llm");

    for &n in &[4096, 11008, 32000] {
        let input = rand_vec(n);
        let mut out = vec![0.0f32; n];
        group.throughput(Throughput::Bytes((2 * n * 4) as u64));

        group.bench_function(BenchmarkId::new("silu", n), |bench| {
            bench.iter(|| kernels.silu(black_box(&input), black_box(&mut out)))
        });
        group.bench_function(BenchmarkId::new("gelu", n), |bench| {
            bench.iter(|| kernels.gelu(black_box(&input), black_box(&mut out)))
        });
    }
    group.finish();
}

// ============================================================
// RMS Norm at LLM hidden sizes
// ============================================================
fn bench_rms_norm(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("rms_norm_llm");

    for &n in &[4096, 5120, 8192] {
        let input = rand_vec(n);
        let weight = rand_vec(n);
        let mut out = vec![0.0f32; n];
        group.throughput(Throughput::Bytes((4 * n * 4) as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |bench| {
            bench.iter(|| {
                kernels.rms_norm(
                    black_box(&input), black_box(&weight),
                    black_box(&mut out), 1e-5,
                );
            })
        });
    }
    group.finish();
}

// ============================================================
// RoPE at LLM shapes
// ============================================================
fn bench_rope(c: &mut Criterion) {
    let kernels = CpuKernels::<f32>::new();
    let mut group = c.benchmark_group("rope_llm");

    for &(seq_len, num_heads, head_dim) in &[
        (1, 32, 128),    // single-token decode
        (32, 32, 128),   // short prefill
        (512, 32, 128),  // long prefill
    ] {
        let total = seq_len * num_heads * head_dim;
        let mut qk = rand_vec(total);
        // rope treats data as (total / head_dim) positions, each needing head_dim/2 cos/sin
        let rope_positions = total / head_dim; // = seq_len * num_heads
        let cos = rand_vec(rope_positions * head_dim / 2);
        let sin = rand_vec(rope_positions * head_dim / 2);
        group.throughput(Throughput::Elements(total as u64));
        group.bench_function(
            BenchmarkId::new("rope", format!("s{seq_len}_h{num_heads}_d{head_dim}")),
            |bench| {
                bench.iter(|| {
                    kernels.rope(
                        black_box(&mut qk),
                        black_box(&cos), black_box(&sin),
                        head_dim, false,
                    );
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = llm_benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_gemv,
        bench_skinny_gemm,
        bench_large_gemm,
        bench_quant_gemv,
        bench_swiglu,
        bench_activations,
        bench_rms_norm,
        bench_rope,
);
criterion_main!(llm_benches);
