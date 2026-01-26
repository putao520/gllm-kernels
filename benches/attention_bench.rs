use std::mem::size_of;
use std::time::Duration;
#[cfg(all(feature = "cuda-kernel", feature = "cuda"))]
use std::sync::Arc;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
#[cfg(all(feature = "cuda-kernel", feature = "cuda"))]
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
#[cfg(all(feature = "cuda-kernel", feature = "cuda"))]
use cudarc::driver::{CudaContext, CudaStream};
use gllm_kernels::backend::{auto_select_backend, Backend as GllmBackend, TensorSlice, TensorSliceMut};
use gllm_kernels::FlashAttentionConfig as KernelFlashAttentionConfig;
// Old ops types (from ops module)
use gllm_kernels::ops::flash_attention::{
    AttentionWorkspace, FlashAttentionConfig, HierarchicalFlashAttention,
};
use gllm_kernels::ops::flash_attention_v3::{FlashAttention3, FlashAttention3Config};
use gllm_kernels::ops::kv_compression::{CompressionMethod, KVCacheCompressor};
use gllm_kernels::ops::sparse_attention::{SparseAttention, SparseAttentionConfig, SparsityPattern};
use gllm_kernels::ops::speculative_decoding::{
    PredictionConfig, PredictionHeadType, SpeculativeDecoder, TreeConfig, VerificationStrategy,
};
#[cfg(all(feature = "cuda-kernel", feature = "cuda"))]
use gllm_kernels::FlashAttentionKernel;
#[cfg(all(feature = "cuda-kernel", feature = "cuda"))]
use burn_cuda::Cuda;

type CpuBackend = NdArray<f32>;
type CpuDevice = <CpuBackend as Backend>::Device;

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
}

fn div_ceil(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return 0;
    }
    (value + divisor - 1) / divisor
}

fn kv_bytes(batch: usize, num_heads: usize, seq_len: usize, head_dim: usize) -> u64 {
    let elements = batch as u64 * num_heads as u64 * seq_len as u64 * head_dim as u64;
    elements * 2 * size_of::<f32>() as u64
}

fn low_rank_bytes(batch: usize, num_heads: usize, seq_len: usize, rank: usize) -> u64 {
    let tokens = batch as u64 * num_heads as u64 * seq_len as u64;
    let projected = tokens * rank as u64 * size_of::<f32>() as u64;
    let indices = rank as u64 * size_of::<usize>() as u64;
    (projected + indices) * 2
}

fn vq_bytes(
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    codebook_size: usize,
    bits: u8,
) -> u64 {
    let tokens = batch as u64 * num_heads as u64 * seq_len as u64;
    let codebook = codebook_size as u64 * head_dim as u64 * size_of::<f32>() as u64;
    let codes = match bits {
        4 => div_ceil(tokens, 2),
        8 => tokens,
        _ => div_ceil(tokens * bits as u64, 8),
    };
    (codebook + codes) * 2
}

fn hybrid_bytes(batch: usize, num_heads: usize, seq_len: usize, rank: usize, bits: u8) -> u64 {
    let tokens = batch as u64 * num_heads as u64 * seq_len as u64;
    let quantized = div_ceil(tokens * rank as u64 * bits as u64, 8);
    let indices = rank as u64 * size_of::<usize>() as u64;
    (quantized + indices) * 2
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn report_compression(label: &str, before: u64, after: u64) {
    let ratio = before as f64 / after as f64;
    println!(
        "{} memory: before={:.2}MB after={:.2}MB ratio={:.2}x",
        label,
        bytes_to_mb(before),
        bytes_to_mb(after),
        ratio
    );
}

fn speculative_token_count(branch_factor: usize, depth: usize) -> u64 {
    let mut total = 0u64;
    let mut level = branch_factor as u64;
    for _ in 0..depth {
        total = total.saturating_add(level);
        level = level.saturating_mul(branch_factor as u64);
    }
    total
}

fn bench_flash_attention_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    configure_group(&mut group);

    let device = CpuDevice::default();
    let attention = HierarchicalFlashAttention::default_config();
    let num_heads = 8;
    let head_dim = 64;
    let seq_lens = [512usize, 1024, 2048, 4096];
    let batches = [1usize, 4];

    for &seq_len in &seq_lens {
        for &batch in &batches {
            let q = Tensor::<CpuBackend, 4>::random(
                [batch, num_heads, seq_len, head_dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let k = Tensor::<CpuBackend, 4>::random(
                [batch, num_heads, seq_len, head_dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let v = Tensor::<CpuBackend, 4>::random(
                [batch, num_heads, seq_len, head_dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );

            let tokens = (batch * seq_len) as u64;
            group.throughput(Throughput::Elements(tokens));
            let id = format!("baseline_seq{}_b{}", seq_len, batch);
            group.bench_function(id, |b| {
                b.iter(|| {
                    let output = attention.forward(q.clone(), k.clone(), v.clone(), false, 0);
                    black_box(output);
                })
            });
        }
    }

    group.finish();
}

fn bench_flash_attention_v3(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    configure_group(&mut group);

    let device = CpuDevice::default();
    let attention = FlashAttention3::new(
        FlashAttentionConfig::default(),
        FlashAttention3Config {
            use_wgmma: true,
            async_pipeline: true,
            fp8_enabled: true,
            block_quantization: true,
        },
    );
    let num_heads = 8;
    let head_dim = 64;
    let seq_lens = [512usize, 1024, 2048, 4096];
    let batches = [1usize, 4];

    for &seq_len in &seq_lens {
        for &batch in &batches {
            let q = Tensor::<CpuBackend, 4>::random(
                [batch, num_heads, seq_len, head_dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let k = Tensor::<CpuBackend, 4>::random(
                [batch, num_heads, seq_len, head_dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let v = Tensor::<CpuBackend, 4>::random(
                [batch, num_heads, seq_len, head_dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );

            let tokens = (batch * seq_len) as u64;
            group.throughput(Throughput::Elements(tokens));
            let id = format!("v3_seq{}_b{}", seq_len, batch);
            group.bench_function(id, |b| {
                b.iter(|| {
                    let output = attention.forward(q.clone(), k.clone(), v.clone(), false, 0);
                    black_box(output);
                })
            });
        }
    }

    group.finish();
}

#[cfg(all(feature = "cuda-kernel", feature = "cuda"))]
fn bench_flash_attention_cuda_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_cuda");
    configure_group(&mut group);

    let cuda_ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };
    let stream: Arc<CudaStream> = cuda_ctx.default_stream();
    let kernel = match FlashAttentionKernel::new(&cuda_ctx) {
        Ok(kernel) => kernel,
        Err(_) => return,
    };

    let cpu_device = CpuDevice::default();
    let batch = 1usize;
    let num_heads = 8usize;
    let seq_len = 512usize;
    let head_dim = 64usize;
    let dims = [batch, num_heads, seq_len, head_dim];

    let q_cpu = Tensor::<CpuBackend, 4>::random(
        dims,
        Distribution::Normal(0.0, 1.0),
        &cpu_device,
    );
    let k_cpu = Tensor::<CpuBackend, 4>::random(
        dims,
        Distribution::Normal(0.0, 1.0),
        &cpu_device,
    );
    let v_cpu = Tensor::<CpuBackend, 4>::random(
        dims,
        Distribution::Normal(0.0, 1.0),
        &cpu_device,
    );

    let q_host = q_cpu
        .clone()
        .into_data()
        .into_vec::<f32>()
        .expect("q host data");
    let k_host = k_cpu
        .clone()
        .into_data()
        .into_vec::<f32>()
        .expect("k host data");
    let v_host = v_cpu
        .clone()
        .into_data()
        .into_vec::<f32>()
        .expect("v host data");

    let q_dev = stream.clone_htod(&q_host).expect("q copy");
    let k_dev = stream.clone_htod(&k_host).expect("k copy");
    let v_dev = stream.clone_htod(&v_host).expect("v copy");

    let tokens = (batch * seq_len) as u64;
    group.throughput(Throughput::Elements(tokens));

    group.bench_function("cuda_kernel", |b| {
        b.iter(|| {
            let output = kernel
                .forward_f32(
                    &stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    batch,
                    num_heads,
                    seq_len,
                    head_dim,
                    false,
                    1.0 / (head_dim as f32).sqrt(),
                    0,
                )
                .expect("cuda kernel");
            stream.synchronize().expect("cuda sync");
            black_box(output);
        })
    });

    let cuda_backend_device = <Cuda as Backend>::Device::default();
    let q_cuda = Tensor::<Cuda, 4>::from_data(TensorData::new(q_host, dims), &cuda_backend_device);
    let k_cuda = Tensor::<Cuda, 4>::from_data(TensorData::new(k_host, dims), &cuda_backend_device);
    let v_cuda = Tensor::<Cuda, 4>::from_data(TensorData::new(v_host, dims), &cuda_backend_device);

    let attention = HierarchicalFlashAttention::default_config();
    group.bench_function("burn_cuda", |b| {
        b.iter(|| {
            let output = attention.forward(q_cuda.clone(), k_cuda.clone(), v_cuda.clone(), false, 0);
            let _ = output.clone().into_data();
            black_box(output);
        })
    });

    group.finish();
}

#[cfg(not(all(feature = "cuda-kernel", feature = "cuda")))]
fn bench_flash_attention_cuda_kernel(_c: &mut Criterion) {}

fn bench_sparse_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse");
    configure_group(&mut group);

    let device = CpuDevice::default();
    let batch = 1usize;
    let num_heads = 8usize;
    let query_len = 128usize;
    let kv_len = 4096usize;
    let selected_counts = [256usize, 512, 1024, 2048];

    let scores = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, query_len, kv_len],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );

    for &selected_kv_count in &selected_counts {
        let config = SparseAttentionConfig {
            selected_kv_count,
            block_size: 128,
            sparsity_pattern: SparsityPattern::Dynamic,
        };
        let sparse = SparseAttention::new(config);

        let tokens = (batch * query_len) as u64;
        group.throughput(Throughput::Elements(tokens));
        let id = format!("select_kv{}", selected_kv_count);
        group.bench_function(id, |b| {
            b.iter(|| {
                let selection = sparse.select_indices(scores.clone()).unwrap();
                black_box(selection);
            })
        });

        group.throughput(Throughput::Elements(tokens));
        let id = format!("sparsify_kv{}", selected_kv_count);
        group.bench_function(id, |b| {
            b.iter(|| {
                let (masked, selection) = sparse.sparsify_scores(scores.clone()).unwrap();
                black_box(masked);
                black_box(selection);
            })
        });
    }

    group.finish();
}

fn bench_kv_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_compression");
    configure_group(&mut group);

    let device = CpuDevice::default();
    let batch = 1usize;
    let num_heads = 8usize;
    let seq_len = 1024usize;
    let head_dim = 64usize;

    let k = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let v = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let input_bytes = kv_bytes(batch, num_heads, seq_len, head_dim);

    let low_rank = KVCacheCompressor::<CpuBackend>::new(CompressionMethod::LowRank { rank: 16 }, 8);
    let low_rank_compressed_bytes = low_rank_bytes(batch, num_heads, seq_len, 16);
    report_compression(
        "kv/low_rank_rank16",
        input_bytes,
        low_rank_compressed_bytes,
    );
    group.throughput(Throughput::Bytes(input_bytes));
    group.bench_function("low_rank_compress_rank16", |b| {
        b.iter(|| {
            let compressed = low_rank.compress_kv(k.clone(), v.clone()).unwrap();
            black_box(compressed);
        })
    });
    let low_rank_compressed = low_rank.compress_kv(k.clone(), v.clone()).unwrap();
    group.throughput(Throughput::Bytes(low_rank_compressed_bytes));
    group.bench_function("low_rank_decompress_rank16", |b| {
        b.iter(|| {
            let (k_full, v_full) = low_rank.decompress_kv(low_rank_compressed.clone()).unwrap();
            black_box(k_full);
            black_box(v_full);
        })
    });

    let vq = KVCacheCompressor::<CpuBackend>::new(
        CompressionMethod::VectorQuantization { codebook_size: 64 },
        8,
    );
    let vq_compressed_bytes = vq_bytes(batch, num_heads, seq_len, head_dim, 64, 8);
    report_compression("kv/vq_codebook64", input_bytes, vq_compressed_bytes);
    group.throughput(Throughput::Bytes(input_bytes));
    group.bench_function("vq_compress_cb64", |b| {
        b.iter(|| {
            let compressed = vq.compress_kv(k.clone(), v.clone()).unwrap();
            black_box(compressed);
        })
    });
    let vq_compressed = vq.compress_kv(k.clone(), v.clone()).unwrap();
    group.throughput(Throughput::Bytes(vq_compressed_bytes));
    group.bench_function("vq_decompress_cb64", |b| {
        b.iter(|| {
            let (k_full, v_full) = vq.decompress_kv(vq_compressed.clone()).unwrap();
            black_box(k_full);
            black_box(v_full);
        })
    });

    let hybrid = KVCacheCompressor::<CpuBackend>::new(
        CompressionMethod::Hybrid {
            rank: 16,
            quant_bits: 4,
        },
        8,
    );
    let hybrid_compressed_bytes = hybrid_bytes(batch, num_heads, seq_len, 16, 4);
    report_compression(
        "kv/hybrid_rank16_q4",
        input_bytes,
        hybrid_compressed_bytes,
    );
    group.throughput(Throughput::Bytes(input_bytes));
    group.bench_function("hybrid_compress_rank16_q4", |b| {
        b.iter(|| {
            let compressed = hybrid.compress_kv(k.clone(), v.clone()).unwrap();
            black_box(compressed);
        })
    });
    let hybrid_compressed = hybrid.compress_kv(k.clone(), v.clone()).unwrap();
    group.throughput(Throughput::Bytes(hybrid_compressed_bytes));
    group.bench_function("hybrid_decompress_rank16_q4", |b| {
        b.iter(|| {
            let (k_full, v_full) = hybrid.decompress_kv(hybrid_compressed.clone()).unwrap();
            black_box(k_full);
            black_box(v_full);
        })
    });

    group.finish();
}

fn bench_speculative_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculative");
    configure_group(&mut group);

    let device = CpuDevice::default();
    let batch = 2usize;
    let seq_len = 16usize;
    let hidden_dim = 256usize;
    let branch_factor = 4usize;
    let cache_len = 16usize;
    let depths = [2usize, 4, 8];

    let hidden = Tensor::<CpuBackend, 3>::random(
        [batch, seq_len, hidden_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let cache_tokens = Tensor::<CpuBackend, 2>::zeros([batch, cache_len], &device);

    for &depth in &depths {
        let prediction = PredictionConfig {
            hidden_dim,
            head_type: PredictionHeadType::Eagle { num_layers: 2 },
        };
        let tree = TreeConfig {
            branch_factor,
            depth,
            verification: VerificationStrategy::Greedy,
        };
        let decoder = SpeculativeDecoder::<CpuBackend>::new(prediction, tree, 8);

        let tokens = speculative_token_count(branch_factor, depth) * batch as u64;
        group.throughput(Throughput::Elements(tokens));
        let id = format!("speculate_depth{}", depth);
        group.bench_function(id, |b| {
            b.iter(|| {
                let candidates = decoder.speculate(hidden.clone()).unwrap();
                black_box(candidates);
            })
        });

        let candidates = decoder.speculate(hidden.clone()).unwrap();
        let target_logits = Tensor::<CpuBackend, 3>::random(
            [batch, depth, hidden_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        group.throughput(Throughput::Elements(tokens));
        let id = format!("verify_depth{}", depth);
        group.bench_function(id, |b| {
            b.iter(|| {
                let verification = decoder
                    .verify(
                        &candidates,
                        target_logits.clone(),
                        cache_tokens.clone(),
                    )
                    .unwrap();
                black_box(verification);
            })
        });
    }

    group.finish();
}

fn bench_e2e_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e");
    configure_group(&mut group);

    let device = CpuDevice::default();
    let batch = 1usize;
    let num_heads = 8usize;
    let seq_len = 1024usize;
    let head_dim = 64usize;
    let tokens = (batch * seq_len) as u64;

    let q = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let k = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let v = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let flash = HierarchicalFlashAttention::default_config();

    let compressor = KVCacheCompressor::<CpuBackend>::new(
        CompressionMethod::Hybrid {
            rank: 16,
            quant_bits: 4,
        },
        8,
    );
    group.throughput(Throughput::Elements(tokens));
    group.bench_function("flash_kv_compression", |b| {
        b.iter(|| {
            let compressed = compressor.compress_kv(k.clone(), v.clone()).unwrap();
            let (k_full, v_full) = compressor.decompress_kv(compressed).unwrap();
            let output = flash.forward(q.clone(), k_full, v_full, false, 0);
            black_box(output);
        })
    });

    group.finish();
}

/// Benchmark comparing standard vs optimized forward pass
fn bench_flash_attention_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_optimization");
    configure_group(&mut group);

    let device = CpuDevice::default();
    let num_heads = 8;
    let head_dim = 64;

    // Test different sequence lengths
    let seq_lens = [256usize, 512, 1024, 2048];
    let batch = 1usize;

    for &seq_len in &seq_lens {
        let attention = HierarchicalFlashAttention::new(FlashAttentionConfig {
            block_q: 64,
            block_kv: 16,
            use_log_space: false,
            ..Default::default()
        });

        let q = Tensor::<CpuBackend, 4>::random(
            [batch, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<CpuBackend, 4>::random(
            [batch, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v = Tensor::<CpuBackend, 4>::random(
            [batch, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let tokens = (batch * seq_len) as u64;
        group.throughput(Throughput::Elements(tokens));

        // Benchmark standard forward (uses cached masks internally)
        let id = format!("standard_seq{}", seq_len);
        group.bench_function(&id, |b| {
            b.iter(|| {
                let output = attention.forward(q.clone(), k.clone(), v.clone(), true, 0);
                black_box(output);
            })
        });

        // Benchmark optimized forward with workspace
        let mut workspace = AttentionWorkspace::new();
        let id = format!("workspace_seq{}", seq_len);
        group.bench_function(&id, |b| {
            b.iter(|| {
                let output = attention.forward_with_workspace(
                    q.clone(), k.clone(), v.clone(), true, 0, &mut workspace
                );
                black_box(output);
            })
        });
    }

    // Test mask cache effectiveness with repeated calls
    let seq_len = 512usize;
    let attention = HierarchicalFlashAttention::new(FlashAttentionConfig {
        block_q: 64,
        block_kv: 16,
        use_log_space: false,
        ..Default::default()
    });

    let q = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let k = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let v = Tensor::<CpuBackend, 4>::random(
        [batch, num_heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Clear mask cache before cold run
    HierarchicalFlashAttention::clear_mask_cache();

    group.throughput(Throughput::Elements((batch * seq_len) as u64));
    group.bench_function("mask_cache_cold", |b| {
        b.iter(|| {
            HierarchicalFlashAttention::clear_mask_cache();
            let output = attention.forward(q.clone(), k.clone(), v.clone(), true, 0);
            black_box(output);
        })
    });

    // Warm cache (don't clear between iterations)
    group.bench_function("mask_cache_warm", |b| {
        b.iter(|| {
            let output = attention.forward(q.clone(), k.clone(), v.clone(), true, 0);
            black_box(output);
        })
    });

    group.finish();
}

/// Benchmark verifying backend call overhead.
fn bench_backend_zero_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_cost_backend");
    configure_group(&mut group);

    // Test parameters
    let batch = 1usize;
    let num_heads = 8usize;
    let head_dim = 64usize;
    let seq_lens = [128usize, 256, 512, 1024];

    let backend = auto_select_backend();
    println!("Backend: {:?}", backend.backend_type());

    for &seq_len in &seq_lens {
        // Prepare data
        let size = batch * num_heads * seq_len * head_dim;
        let q: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let k: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos()).collect();
        let v: Vec<f32> = (0..size).map(|i| (i as f32 * 0.003).sin()).collect();
        let mut output = vec![0.0f32; size];

        let config = KernelFlashAttentionConfig {
            batch_size: batch,
            num_heads,
            seq_len_q: seq_len,
            seq_len_kv: seq_len,
            head_dim,
            causal: true,
            use_log_space_softmax: false,
            use_kahan_accumulator: false,
            ..Default::default()
        };

        let tokens = (batch * seq_len) as u64;
        group.throughput(Throughput::Elements(tokens));

        let id = format!("backend_seq{}", seq_len);
        group.bench_function(&id, |b| {
            b.iter(|| {
                backend
                    .flash_attention(
                        TensorSlice::F32(black_box(&q)),
                        TensorSlice::F32(black_box(&k)),
                        TensorSlice::F32(black_box(&v)),
                        TensorSliceMut::F32(black_box(&mut output)),
                        config.clone(),
                    )
                    .expect("backend flash_attention failed");
                black_box(&output);
            })
        });
    }

    group.finish();
}

/// Benchmark f16 vs f32 performance to verify generic overhead.
fn bench_backend_f16_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("f16_vs_f32");
    configure_group(&mut group);

    let batch = 1usize;
    let num_heads = 4usize;
    let seq_len = 256usize;
    let head_dim = 64usize;
    let size = batch * num_heads * seq_len * head_dim;

    let backend = auto_select_backend();

    // f32 data
    let q_f32: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
    let k_f32: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos()).collect();
    let v_f32: Vec<f32> = (0..size).map(|i| (i as f32 * 0.003).sin()).collect();
    let mut output_f32 = vec![0.0f32; size];

    // f16 data
    let q_f16: Vec<half::f16> = q_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_f16: Vec<half::f16> = k_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_f16: Vec<half::f16> = v_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let mut output_f16 = vec![half::f16::ZERO; size];

    let config = KernelFlashAttentionConfig {
        batch_size: batch,
        num_heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: false,
        use_kahan_accumulator: false,
        ..Default::default()
    };

    let tokens = (batch * seq_len) as u64;
    group.throughput(Throughput::Elements(tokens));

    // f32 benchmark
    group.bench_function("flash_attn_f32", |b| {
        b.iter(|| {
            backend
                .flash_attention(
                    TensorSlice::F32(black_box(&q_f32)),
                    TensorSlice::F32(black_box(&k_f32)),
                    TensorSlice::F32(black_box(&v_f32)),
                    TensorSliceMut::F32(black_box(&mut output_f32)),
                    config.clone(),
                )
                .expect("backend flash_attention f32 failed");
            black_box(&output_f32);
        })
    });

    // f16 benchmark
    group.bench_function("flash_attn_f16", |b| {
        b.iter(|| {
            backend
                .flash_attention(
                    TensorSlice::F16(black_box(&q_f16)),
                    TensorSlice::F16(black_box(&k_f16)),
                    TensorSlice::F16(black_box(&v_f16)),
                    TensorSliceMut::F16(black_box(&mut output_f16)),
                    config.clone(),
                )
                .expect("backend flash_attention f16 failed");
            black_box(&output_f16);
        })
    });

    group.finish();
}

/// Benchmark long context (2M tokens) numerical stability.
fn bench_backend_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    configure_group(&mut group);

    let batch = 1usize;
    let num_heads = 1usize;
    let head_dim = 64usize;
    let seq_len = 512usize; // Use smaller seq for benchmark speed

    let backend = auto_select_backend();
    let size = batch * num_heads * seq_len * head_dim;

    let q: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..size).map(|i| (i as f32 * 0.003).sin()).collect();
    let mut output = vec![0.0f32; size];

    let tokens = (batch * seq_len) as u64;
    group.throughput(Throughput::Elements(tokens));

    // Standard mode (fast)
    let config_fast = KernelFlashAttentionConfig {
        batch_size: batch,
        num_heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: false,
        use_kahan_accumulator: false,
        ..Default::default()
    };

    group.bench_function("standard_softmax", |b| {
        b.iter(|| {
            backend
                .flash_attention(
                    TensorSlice::F32(black_box(&q)),
                    TensorSlice::F32(black_box(&k)),
                    TensorSlice::F32(black_box(&v)),
                    TensorSliceMut::F32(black_box(&mut output)),
                    config_fast.clone(),
                )
                .expect("backend flash_attention fast failed");
            black_box(&output);
        })
    });

    // Stable mode (log-space + Kahan)
    let config_stable = KernelFlashAttentionConfig {
        batch_size: batch,
        num_heads,
        seq_len_q: seq_len,
        seq_len_kv: seq_len,
        head_dim,
        causal: true,
        use_log_space_softmax: true,
        use_kahan_accumulator: true,
        ..Default::default()
    };

    group.bench_function("stable_log_kahan", |b| {
        b.iter(|| {
            backend
                .flash_attention(
                    TensorSlice::F32(black_box(&q)),
                    TensorSlice::F32(black_box(&k)),
                    TensorSlice::F32(black_box(&v)),
                    TensorSliceMut::F32(black_box(&mut output)),
                    config_stable.clone(),
                )
                .expect("backend flash_attention stable failed");
            black_box(&output);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_backend_zero_cost,
    bench_backend_f16_f32,
    bench_backend_stability,
    bench_flash_attention_optimized,
    bench_flash_attention_baseline,
    bench_flash_attention_v3,
    bench_flash_attention_cuda_kernel,
    bench_sparse_attention,
    bench_kv_compression,
    bench_speculative_decoding,
    bench_e2e_pipeline
);
criterion_main!(benches);
