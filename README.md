# gllm-kernels — JIT Compiler Backend for LLM Inference

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The JIT-compiled fusion kernel engine behind [gllm](https://github.com/putao520/gllm). Generates hardware-optimal machine code at model load time for x86_64, AArch64, and GPU (CUDA/ROCm/Metal).

## Core Architecture

### 4-Stage JIT Pipeline

```
Scalar (Rust) → SymExec → TraceOp IR → ISA Lowering → Machine Code
```

| Stage | Module | Output |
|-------|--------|--------|
| Phase 0 | `registry` + `symexec/` | Scalar reference + OpTrace + ComputePattern |
| Phase 1 | `semantic_dag` | OpClass classification (ElemWise/Injective/Reduction/Gemm/Opaque) |
| Phase 2 | `fusion` + `hw_constraints` | FusionPlan with 7 fusion rules |
| Phase 3 | `codegen/x86_64`, `codegen/aarch64`, `codegen/gpu_ir` | Device-specific machine code |

### Auto Instruction Selection

All operators route through `auto_select` — a lookup-table-based TraceOp → VmInstr mapping (similar to LLVM SelectionDAG). No hand-written per-opcode lowering. Adding a new operator requires only: register scalar impl → SymExec extracts ComputePattern → automatic routing.

### Fusion Modes

- **EpilogueInjection**: GEMM + activation/bias/residual fused into accumulator registers
- **LoopFusion**: Elementwise operator chains merged into single loop
- **TileLevelFusion**: Predecessor ops embedded in GEMM MC loop
- **ComputeRoot**: Predecessor fully computed, stays in L1/L2
- **QkvSharedInput**: Q/K/V GEMMs share packed input
- **NormIntoGemm**: RmsNorm output fed directly into GEMM (no intermediate writeback)

## Backend Support

| Backend | ISA Targets | Key Features |
|---------|-------------|-------------|
| **x86_64** | AVX2, AVX-512 (VNNI, BF16, FP16), AMX | BLIS-style GEMM, tile computation |
| **AArch64** | NEON, SVE, SME2 | Outer product GEMM, ZA storage |
| **CUDA** | PTX (sm_70/80/90/100+) | Tensor Core, TMA, FP4 MMA, warp specialization |
| **ROCm** | HIP (AMDGPU) | MFMA instructions, CDNA architecture |
| **Metal** | MSL (Apple GPU) | Threadgroup shared memory, SIMD-group operations |

## VmInstr Architecture

Virtual instruction set with ~100 dtype-polymorphic opcodes. Dtype is in the type signature, not the instruction name (similar to Triton TTIR). ISA lowering maps each VmInstr to hardware-specific instructions via `DeviceProfile`.

### Quantization Support

22 quantization types with JIT dequantization fusion:
- **4-bit**: INT4, AWQ4, GPTQ4, NVFP4, MXFP4, FP4 (E2M1)
- **8-bit**: INT8, FP8 (E4M3/E5M2)
- **GGUF types**: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ series
- **Block-level**: QuantGather prologue with double buffering

## Operator Coverage

| Category | Operators |
|----------|-----------|
| **GEMM** | Dense, Quantized, MoE Expert, MLA Absorbed |
| **Attention** | MHA, GQA, MLA, FlashAttention, Sliding Window |
| **Normalization** | RmsNorm, LayerNorm, QkNorm, ValueNorm |
| **Activation** | SiLU, GELU, SwiGLU, ReLU |
| **Position Encoding** | RoPE, DualRoPE, p-RoPE, YaRN |
| **MoE** | Gate, Router, Dispatch, Shared Experts |
| **Sampling** | Temperature, Top-K, Top-P, Argmax, Min-P |
| **Structural** | Gather, ColumnSlice, Concat, Reshape, Transpose |
| **AltUp** | Predict, Correct, Inject (PLE gating) |
| **KV Cache** | Paged KV read/write, compression, migration |

## License

MIT License
