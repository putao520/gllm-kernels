# gllm-kernels 功能需求清单

> **📌 SSOT**: 本文档是 gllm-kernels 项目的功能需求唯一真源。

## 1. 核心架构需求 (REQ-ARCH)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-ARCH-001** | L3 GPU-Pure API | 实现全流程 GPU 常驻推理 API | 1. 权重一次上传<br>2. KV Cache GPU 常驻<br>3. Logits GPU 常驻<br>4. 仅传输 Token ID (8 bytes/step) | 🟢 已完成 |
| **REQ-ARCH-002** | 算子融合 | 必须使用融合算子替代独立算子串联 | 实现 `fused_qkv_rope`, `fused_gate_up_silu` | 🟢 已完成 |
| **REQ-ARCH-003** | 静态工作空间 | 消除生成循环中的显存分配 | 使用预分配的 Workspace Buffer | 🟢 已完成 |
| **REQ-ARCH-004** | AOT Only | 仅使用预编译二进制 (CUBIN)，禁止 PTX JIT | 1. 针对 sm_80/86/89/90 预编译<br>2. 运行时根据架构加载对应 CUBIN | 🟢 已完成 |
| **REQ-ARCH-005** | Driver API Only | 仅依赖 `libcuda.so` | 不链接 `libcudart.so` | 🟢 已完成 |

## 2. 后端支持需求 (REQ-BACKEND)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-BACKEND-001** | CUDA 后端 | NVIDIA GPU 高性能实现 | 实现所有 L3 API 和核心算子 | 🟢 已完成 |
| **REQ-BACKEND-002** | CPU 后端 | 纯 Rust 高性能实现 | 1. **F32/F16**: 使用 `faer` 库实现 SIMD 加速 (AVX2/AVX-512/NEON)<br>2. **Quantized**: 手写 SIMD 反量化内核 | 🟢 已完成 |
| **REQ-BACKEND-003** | 自动后端选择 | 运行时自动检测可用硬件 | 优先选择 CUDA，回退到 CPU | 🟢 已完成 |

## 3. CPU 性能优化 (REQ-CPU-OPT)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-CPU-001** | SIMD/AVX 加速 | 充分利用 CPU 向量化指令 | 核心算子覆盖 AVX2 (x86) 和 NEON (ARM) | 🟢 已完成 |
| **REQ-CPU-002** | 线性代数库集成 | 使用 `faer` 替代手写 BLAS | 1. 稠密计算 (f32/f64) 委托给 `faer`<br>2. 确保无 C/C++ 链接依赖 | 🟢 已完成 |
| **REQ-CPU-003** | CPU 融合算子 | CPU 端也必须实现算子融合 | 减少内存带宽压力，提升 L2 Cache 命中率 | 🟢 已完成 |
| **REQ-CPU-004** | 小模型快路径 | 小模型/Batch=1 绕过复杂调度 | 直接在 CPU 上执行完整推理，零开销 | 🟢 已完成 |

## 4. 量化支持需求 (REQ-QUANT)

> 针对 1/2/4/8/16-bit 不同精度的支持方案。
> **核心策略**: 存储使用 Packed u8，计算使用 Block-wise Dequantization。
> **架构决策** (ARCH-QUANT-TEMPLATE): 使用 CUDA C++ 模板统一实现量化内核，避免每种位宽单独实现。

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-QUANT-001** | 多精度类型系统 | 支持 <16bit 数据的类型表达 | 1. `half` (f16/bf16)<br>2. `u8` (Int8)<br>3. `Packed<u8>` (Int4/2/1) | 🟢 已完成 |
| **REQ-QUANT-002** | 块式量化格式 | 定义 Block-wise 存储结构 | 支持类似 GGUF/AWQ 的块结构：<br>`struct Block { scales: f16, data: [u8; N] }` | 🟢 已完成 |
| **REQ-QUANT-003** | 即时反量化内核 | SIMD/CUDA 解包计算 | 1. **CPU**: SIMD 加载 u8 -> 寄存器解包 -> FMA<br>2. **CUDA**: 模板化量化内核 (int8/4/2/1) | 🟢 已完成 |
| **REQ-QUANT-004** | 模板化量化内核 | 使用 C++ 模板统一实现 | 1. `template<int BITS>` 量化矩阵乘法<br>2. 一套代码覆盖 1/2/4/8-bit<br>3. 编译时实例化，零运行时开销 | 🟢 已完成 |

## 5. 核心算子需求 (REQ-OPS)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-OPS-001** | FlashAttention | 实现 FlashAttention-2 算法 | 支持 VarLen, Causal Masking | 🟢 已完成 |
| **REQ-OPS-002** | RoPE | 旋转位置编码 | 支持 ALiBi/RoPE, 预计算/在线计算 | 🟢 已完成 |
| **REQ-OPS-003** | RMSNorm | Root Mean Square Normalization | 支持 epsilon 配置 | 🟢 已完成 |
| **REQ-OPS-004** | SiLU/SwiGLU | 激活函数 | 融合实现 (Gate+Up+SiLU+Mul) | 🟢 已完成 |
| **REQ-OPS-005** | Sampling | GPU 端采样 | 支持 Argmax, Top-K, Top-P, Temperature | 🟢 已完成 |

## 6. 性能指标需求 (REQ-PERF)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-PERF-001** | 高吞吐生成 | 消除 PCIe 瓶颈 | GTX 1060 (0.6B) TPS > 50 (待基准测试验证) | 🟡 待验证 |
| **REQ-PERF-002** | 超长上下文 | 支持长文本推理 | 架构支持 2M+ Token (受限于显存) | 🟡 架构就绪 |
| **REQ-PERF-003** | 精度支持 | 多精度推理支持 | 支持 FP32, FP16, BF16 | 🟢 已完成 |

## 7. 后端调度需求 (REQ-SCHED)

> **关联文档**: [gllm SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md](../gllm/SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md)

| ID | 需求标题 | 描述 | 验收标准 | 状态 |
|----|----------|------|----------|------|
| **REQ-SCHED-K001** | 页面状态管理 | 支持页面状态机 (Active/Standby/Swapped/Warm/Protected) | 1. `get_page_states()` 返回正确状态<br>2. **Warm 状态页面不被换出**<br>3. Protected 状态页面不被换出 | 🟢 已实现 (2026-02-02) [commit: d667c6b] |
| **REQ-SCHED-K002** | Swap 操作接口 | 实现 swap_out_pages 和 swap_in_pages | 1. **swap_out_pages** 将页面从 GPU 搬运到 CPU<br>2. **swap_in_pages** 将页面从 CPU 搬运回 GPU<br>3. 操作不影响正确性 | 🟢 已实现 |
| **REQ-SCHED-K003** | 批处理前向传播 | 实现 batch_forward_gpu_pure | 1. 支持多序列同时 forward<br>2. 每个序列返回独立的 LogitsTensor<br>3. 位置参数正确 | 🟢 已实现 |
| **REQ-SCHED-K004** | 内存压力检测 | 实现 get_memory_pressure() | 1. 返回 0.0-1.0 的内存使用率<br>2. 使用 CUDA API 查询当前内存<br>3. 精度 < 1% | 🟢 已实现 |
| **REQ-SCHED-K005** | 自动 Swap 触发 | 内存不足时自动 swap-out | 1. **检测内存压力超过阈值**<br>2. **自动选择 LRU 受害者页面**<br>3. **执行 swap-out 操作** | 🟢 已实现 (2026-02-02) |
| **REQ-SCHED-K006** | SwapManager | 集成 SwapManager | 1. **LRU 受害者选择**<br>2. **CPU 端页面备份**<br>3. **页面状态跟踪** (含 Warm/Protected) | 🟢 已实现 (2026-02-02) [commit: d667c6b] |
| **REQ-SCHED-K007** | Chunked Prefill 后端支持 | 支持 Chunked Prefill 调度的后端接口 | 1. `batch_forward_gpu_pure()` 支持不同序列不同位置<br>2. 支持部分 KV Cache 复用<br>3. **AOT CUBIN 兼容** (无需新 Kernel) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |
| **REQ-SCHED-K008** | SwiftKV 后端支持 | 支持 KV Cache 蒸馏和压缩 | 1. SwapManager 支持蒸馏模式 (`distill_kv_pages`)<br>2. 支持跨层 KV 相似度计算 (`compute_kv_similarity`)<br>3. CPU 端蒸馏算法实现<br>4. **AOT CUBIN 兼容** (蒸馏在 CPU 端执行) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |
| **REQ-SCHED-K009** | LMCache 后端支持 | 支持跨请求 KV Cache 共享 | 1. GPU ↔ CPU DMA 复制接口<br>2. KV Cache 序列化/反序列化<br>3. 支持 Redis/LocalDisk 后端<br>4. **AOT CUBIN 兼容** (使用现有 Memcpy Kernel) | 🟢 已实现 (2026-02-02) [commit: 085bbf8] |

> **详细设计**: 见 [gllm SPEC/02-ARCHITECTURE.md §2024 vLLM 优化](../gllm/SPEC/02-ARCHITECTURE.md#2024-vllm-优化-arch-sched-2024)
