# gllm-kernels

Low-level attention kernels for [gllm](https://github.com/putao520/gllm) with CUDA/ROCm support.

[![Crates.io](https://img.shields.io/crates/v/gllm-kernels.svg)](https://crates.io/crates/gllm-kernels)
[![Documentation](https://docs.rs/gllm-kernels/badge.svg)](https://docs.rs/gllm-kernels)
[![License](https://img.shields.io/crates/l/gllm-kernels.svg)](LICENSE)

## Features

- **FlashAttention**: Memory-efficient attention with O(N) memory complexity
- **Hierarchical Attention**: Multi-level attention for ultra-long contexts (2M+ tokens)
- **CUDA Kernels**: Native CUDA implementation with PTX for NVIDIA GPUs
- **ROCm/HIP Kernels**: AMD GPU support (experimental)
- **Multiple Backends**: CPU (ndarray), CUDA, WebGPU via Burn

## Performance

| Implementation | Time (seq=512) | vs burn_cuda |
|----------------|----------------|--------------|
| **cuda_kernel** | 21.27ms | **37% faster** |
| burn_cuda | 33.83ms | baseline |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gllm-kernels = "0.1"
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `cpu` | CPU backend via burn-ndarray | Yes |
| `cuda` | CUDA backend via burn-cuda | No |
| `cuda-kernel` | Native CUDA kernels (requires CUDA toolkit) | No |
| `wgpu` | WebGPU backend | No |
| `rocm-kernel` | ROCm/HIP kernels (experimental) | No |

## Usage

### Basic FlashAttention

```rust
use gllm_kernels::ops::flash_attention::{
    HierarchicalFlashAttention, AttentionConfig
};

// Create attention module
let attention = HierarchicalFlashAttention::new(
    num_heads,
    head_dim,
    AttentionConfig::default(),
);

// Forward pass
let output = attention.forward(q, k, v, mask)?;
```

### CUDA Kernel (Native)

```rust
use gllm_kernels::cuda_kernels::FlashAttentionKernel;
use cudarc::driver::CudaContext;
use std::sync::Arc;

let ctx = Arc::new(CudaContext::new(0)?);
let kernel = FlashAttentionKernel::new(&ctx)?;

let output = kernel.forward(
    &stream, &q, &k, &v,
    batch_size, num_heads, seq_len, head_dim,
    is_causal, scale, position_offset,
)?;
```

### Deterministic Mode

For reproducible results in ultra-long context scenarios:

```rust
use gllm_kernels::ops::flash_attention::DeterministicConfig;

let config = AttentionConfig {
    determinism: DeterministicConfig::strict(),
    ..Default::default()
};
```

## Architecture

```
gllm-kernels
├── ops/
│   ├── flash_attention.rs      # HierarchicalFlashAttention
│   ├── flash_attention_v3.rs   # Advanced attention variants
│   ├── paged_attention.rs      # KV cache paging
│   ├── ring_attention.rs       # Distributed attention
│   ├── sparse_attention.rs     # Sparse patterns
│   ├── mla.rs                  # Multi-head Latent Attention
│   ├── mamba.rs                # State space models
│   └── kv_compression.rs       # KV cache compression
├── cuda_kernels/
│   ├── flash_attn.rs           # CUDA kernel bindings
│   └── kernels/
│       ├── tiled_attention.cu  # CUDA source
│       └── tiled_attention.ptx # Compiled PTX (sm_61)
├── hip_kernels/                # ROCm/HIP (experimental)
└── comm/                       # Distributed communication
```

## Building CUDA Kernels

If you need to recompile PTX for a different GPU architecture:

```bash
cd src/cuda_kernels/kernels
nvcc -ptx -arch=sm_XX tiled_attention.cu -o tiled_attention.ptx
```

Replace `sm_XX` with your GPU's compute capability (e.g., `sm_61` for GTX 1060, `sm_86` for RTX 3090).

## License

Apache-2.0
