# gllm-kernels 需求清单

> **📌 SSOT**: 本文档是 gllm-kernels 算子库的需求唯一真源。

## 定位

**纯 CPU 算子库**：提供逼近硬件理论极限的底层计算原语。不含任何业务逻辑（无 Attention、无 KV Cache、无推理流程、无 GPU 后端）。

---

## 1. 性能需求（REQ-PERF）🚨 铁律

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-PERF-001** | Compute-bound 算子逼近计算峰值 | GEMM 达到理论 FLOPS 峰值的 **≥ 85%** | 🔴 当前 41%，需手写汇编 |
| **REQ-PERF-002** | Memory-bound 算子逼近带宽峰值 | 激活/归一化/BLAS-1 达到内存带宽的 **≥ 90%** | 🟡 待测 |
| **REQ-PERF-003** | 量化算子逼近瓶颈极限 | 量化 GEMV/GEMM 达到 **≥ 85%** 瓶颈极限 | 🔴 需手写汇编 |
| **REQ-PERF-004** | 手写汇编微内核 | GEMM、量化 GEMV/GEMM 必须使用 `global_asm!` 手写汇编 | 🔴 待实现 |
| **REQ-PERF-005** | 运行时 CPUID 分发 | 启动时一次检测 ISA，之后零开销分发到最优微内核 | 🟢 已完成 |

---

## 2. 算子需求（REQ-OPS）

### 2.1 BLAS 算子

| ID | 算子 | 签名 | 瓶颈 | 状态 |
|----|------|------|------|------|
| **REQ-OPS-001** | vec_dot | `(a: &[E], b: &[E]) -> E` | Memory | 🟢 已完成 |
| **REQ-OPS-002** | vec_add | `(a: &[E], b: &[E], out: &mut [E])` | Memory | 🟢 已完成 |
| **REQ-OPS-003** | vec_sub | `(a: &[E], b: &[E], out: &mut [E])` | Memory | 🟢 已完成 |
| **REQ-OPS-004** | vec_mul | `(a: &[E], b: &[E], out: &mut [E])` | Memory | 🟢 已完成 |
| **REQ-OPS-005** | vec_scale | `(x: &mut [E], s: E)` | Memory | 🟢 已完成 |
| **REQ-OPS-006** | vec_axpy | `(y: &mut [E], a: E, x: &[E])` | Memory | 🟢 已完成 |
| **REQ-OPS-007** | vec_sum | `(x: &[E]) -> E` | Memory | 🟢 已完成 |
| **REQ-OPS-008** | vec_max | `(x: &[E]) -> E` | Memory | 🟢 已完成 |
| **REQ-OPS-009** | vec_sum_squares | `(x: &[E]) -> E` | Memory | 🟢 已完成 |
| **REQ-OPS-010** | gemv | `(a: &[E], x: &[E], y: &mut [E], m, n)` | Memory | 🟢 已完成 |
| **REQ-OPS-011** | gemm | `(a: &[E], b: &[E], c: &mut [E], m, n, k)` | Compute | 🟡 intrinsics 实现，需手写 asm |
| **REQ-OPS-012** | gemm_bias | `(a, b, bias, c, m, n, k)` | Compute | 🟡 同上 |
| **REQ-OPS-013** | gemm_prepacked | `(a, packed_b, c, m, n, k)` | Compute | 🟡 同上 |
| **REQ-OPS-014** | pack_b | `(b, packed_b, n, k)` | Memory | 🟢 已完成 |

### 2.2 激活函数

| ID | 算子 | 瓶颈 | 状态 |
|----|------|------|------|
| **REQ-OPS-020** | silu | Memory | 🟢 已完成 |
| **REQ-OPS-021** | gelu | Memory | 🟢 已完成 |
| **REQ-OPS-022** | relu | Memory | 🟢 已完成 |
| **REQ-OPS-023** | tanh | Memory | 🟢 已完成 |
| **REQ-OPS-024** | swiglu | Memory | 🟢 已完成 |
| **REQ-OPS-025** | softmax | Memory | 🟢 已完成 |
| **REQ-OPS-026** | exp | Memory | 🟢 已完成 |

### 2.3 归一化

| ID | 算子 | 瓶颈 | 状态 |
|----|------|------|------|
| **REQ-OPS-030** | rms_norm | Memory | 🟢 已完成 |
| **REQ-OPS-031** | layer_norm | Memory | 🟢 已完成 |

### 2.4 位置编码

| ID | 算子 | 瓶颈 | 状态 |
|----|------|------|------|
| **REQ-OPS-040** | rope | Memory | 🟢 已完成 |
| **REQ-OPS-041** | rope_with_pos | Memory | 🟢 已完成 |

### 2.5 量化解码（18 种格式）

| ID | 格式 | 块大小 | 块字节 | 位宽 | 状态 |
|----|------|--------|--------|------|------|
| **REQ-OPS-050** | Q2_K | 256 | 84 | 2 | 🟢 已完成 |
| **REQ-OPS-051** | Q3_K | 256 | 110 | 3 | 🟢 已完成 |
| **REQ-OPS-052** | Q4_K | 256 | 144 | 4 | 🟢 已完成 |
| **REQ-OPS-053** | Q5_K | 256 | 176 | 5 | 🟢 已完成 |
| **REQ-OPS-054** | Q6_K | 256 | 210 | 6 | 🟢 已完成 |
| **REQ-OPS-055** | Q8_K | 256 | 292 | 8 | 🟢 已完成 |
| **REQ-OPS-056** | IQ1_S | 256 | 50 | 1 | 🟢 已完成 |
| **REQ-OPS-057** | IQ1_M | 256 | 56 | 1 | 🟢 已完成 |
| **REQ-OPS-058** | IQ2_XXS | 256 | 66 | 2 | 🟢 已完成 |
| **REQ-OPS-059** | IQ2_XS | 256 | 74 | 2 | 🟢 已完成 |
| **REQ-OPS-060** | IQ2_S | 256 | 82 | 2 | 🟢 已完成 |
| **REQ-OPS-061** | IQ3_XXS | 256 | 98 | 3 | 🟢 已完成 |
| **REQ-OPS-062** | IQ3_S | 256 | 110 | 3 | 🟢 已完成 |
| **REQ-OPS-063** | IQ4_NL | 32 | 18 | 4 | 🟢 已完成 |
| **REQ-OPS-064** | IQ4_XS | 256 | 136 | 4 | 🟢 已完成 |
| **REQ-OPS-065** | AWQ4 | 128 | 72 | 4 | 🟢 已完成 |
| **REQ-OPS-066** | GPTQ4 | 128 | 72 | 4 | 🟢 已完成 |
| **REQ-OPS-067** | SqueezeLLM | 256 | 130 | 3 | 🟢 已完成 |

### 2.6 量化 GEMV/GEMM

| ID | 算子 | 权重格式 | 状态 |
|----|------|----------|------|
| **REQ-OPS-070** | gemv_q8 | INT8 | 🟡 intrinsics，需手写 asm |
| **REQ-OPS-071** | gemv_q4 | INT4 packed | 🟡 同上 |
| **REQ-OPS-072** | gemv_q2 | INT2 packed | 🟡 同上 |
| **REQ-OPS-073** | gemv_q1 | INT1 packed | 🟡 同上 |
| **REQ-OPS-074** | gemm_q8 | INT8 | 🟡 同上 |
| **REQ-OPS-075** | gemm_q4 | INT4 packed | 🟡 同上 |
| **REQ-OPS-076** | kquant_matmul | K-Quant 系列 | 🟡 同上 |
| **REQ-OPS-077** | iq_matmul | IQ 系列 | 🟡 同上 |
| **REQ-OPS-078** | awq_matmul | AWQ4 | 🟡 同上 |
| **REQ-OPS-079** | gptq_matmul | GPTQ4 | 🟡 同上 |
| **REQ-OPS-080** | squeeze_matmul | SqueezeLLM | 🟡 同上 |

---

## 3. 架构需求（REQ-ARCH）

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-ARCH-001** | 纯 Rust 零外部依赖 | `cargo install` 一键安装，禁止 faer/OpenBLAS/MKL | 🟢 已完成 |
| **REQ-ARCH-002** | 手写汇编微内核 | GEMM/量化 GEMV 使用 `global_asm!` | 🔴 待实现 |
| **REQ-ARCH-003** | 运行时 ISA 分发 | OnceLock + CPUID 检测，启动后零开销 | 🟢 已完成 |
| **REQ-ARCH-004** | 泛型精度支持 | f32/f16/bf16 通过 `<E: Element>` 编译时单态化 | 🟢 已完成 |
| **REQ-ARCH-005** | 宏驱动代码生成 | 非热路径通过四层宏架构批量展开 | 🟢 已完成 |
| **REQ-ARCH-006** | 多 ISA 支持 | AVX2 / AVX-512 / NEON / Scalar | 🟢 已完成 |

---

## 4. ISA 覆盖需求（REQ-ISA）

| ISA | f32 | f16 | bf16 | 手写 asm | 状态 |
|-----|-----|-----|------|---------|------|
| **Scalar** | ✅ | ✅ 软件转换 | ✅ 软件转换 | N/A | 🟢 |
| **AVX2** | ✅ | ✅ F16C | ✅ 位转换 | 🔴 待实现 | 🟡 |
| **AVX-512** | ✅ | ⚡ FP16 原生 | ⚡ BF16 原生 | 🔴 待实现 | 🟡 |
| **NEON** | ✅ | ⚡ FP16 原生 | ✅ 位转换 | 🔴 待实现 | 🟡 |

---

## 5. 不在范围内（OUT-OF-SCOPE）

以下功能属于上层推理引擎，不在本算子库范围内：

- FlashAttention / Paged Attention
- KV Cache 管理与调度
- 融合算子（fused_qkv_rope, fused_ffn, fused_gate_up_swiglu 等）
- Embedding lookup
- Sampling (argmax, top-k, top-p)
- CUDA / Metal / ROCm GPU 后端
- L3 GPU-Pure API
- 推理调度、批处理、Swap 管理
- ONNX 加载与图优化
- Tree Attention / 推测解码
