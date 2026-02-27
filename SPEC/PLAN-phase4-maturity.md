# Phase 4 成熟度突破规划（JIT 编译器核心）

> **目标**：从成熟度 3.5/5 推进到 4.5/5，以 JIT 编译器为唯一性能交付路径。
> **时间跨度**：3 个里程碑，预计 8-12 周。
> **核心策略**：JIT Phase 3 生成完整高性能代码 → Autotuning 闭环调优 → 推理栈可用。

---

## 架构定位（与 CLAUDE.md 一致）

> 所有性能优化通过 JIT 编译器实现。算子的唯一定义来源是 `extern "C"` 纯标量函数，
> 编译器通过二进制符号执行自动提取计算结构（OpTrace），然后根据 DeviceProfile 生成最优融合 SIMD 代码。
> 项目中现有的手写 asm / intrinsics / 宏生成实现作为正确性基准和性能参考。

**关键区别（vs 旧规划）**：
- 不再有"优化手写 ASM 微内核"的任务
- pack_a/pack_b 向量化由 JIT 生成，不是手写
- 软件流水线由 JIT 生成，不是手写
- 手写 ASM 只用于验证 JIT 输出的正确性和性能对标
- 性能目标是 JIT 生成代码的性能，不是手写 ASM 的性能

---

## 现状诊断

| 维度 | 当前 | 目标 | 差距根因 |
|------|------|------|---------|
| GEMM 效率 | unpacked ~42%, prepacked ~59% (ASM 参考路径) | JIT 生成代码尽可能逼近理论峰值 | JIT BLIS 循环的 pack 例程未向量化；K-loop 未做软件流水线；blocking 参数未经 autotuning |
| Phase 3 x86_64 AVX2 | 完整实现（6307 行） | 生成完整高性能 GEMM | pack 例程 JIT 生成、K-loop 软件流水线、边界处理 |
| Phase 3 x86_64 AVX-512 | 基础框架 | 完整 BLIS 循环 + epilogue | zmm 寄存器分配、14×32 微内核 JIT 生成未完成 |
| Phase 3 aarch64 | BLIS 5 级循环 + TraceOp→NEON 映射 | 完整 elementwise 链 + E2E 验证 | emit_plan 的 LoopFusion 分支未接入 |
| Autotuning 闭环 | 框架完整（2177 行），tune_gemm 可用 | 搜索→JIT 编译→测量→反馈 | autotuning 结果未反馈到 JIT codegen |
| E2E 推理 | cpu_backend 1356 行，attention 简化 | 可跑 Llama-7B 推理 | KV Cache 骨架、attention 未完整 |

---

## 里程碑总览

```
MS-1: Phase 3 Codegen 完整化（JIT 能生成完整的高性能代码）
  │
  ├── MS-2: Autotuning→JIT 闭环（搜索→编译→测量→反馈最优参数）
  │
  └── MS-3: 推理栈可用（KV Cache + Attention + decoder_forward）
```

---

## MS-1: Phase 3 Codegen 完整化

> **优先级**：P0 🔴
> **预计工期**：4-5 周
> **依赖**：无（可立即开始）
> **核心目标**：JIT 生成的 GEMM 代码数值正确（vs 标量 reference），具备达到高性能的代码结构。

### 1.1 JIT 自动生成 BLIS 5 级循环（含 pack）

**现状**：`emit_gemm_blis()`（x86_64.rs）已有 BLIS 循环框架，但 pack_a/pack_b 调用外部函数而非 JIT 内联生成。

**目标**：JIT 程序化生成完整的 BLIS 循环，包括向量化的 pack 例程。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 1.1.1 | JIT 生成 `pack_b` AVX2 向量化循环（NR=16 → 16×8 转置） | pack_b 吞吐 ≥ L1 带宽 80% | 高 — 转置模式明确 |
| 1.1.2 | JIT 生成 `pack_a` AVX2 向量化循环（MR=6 → 6×8 转置） | pack_a 吞吐 ≥ L1 带宽 80% | 高 |
| 1.1.3 | BLIS 5 级循环完整内联（NC/MC/KC 循环 + pack + 微内核） | JIT GEMM 1024×1024×1024 数值正确（vs scalar_gemm ≤1e-4） | 高 |
| 1.1.4 | AVX-512 pack_a/pack_b（MR=14, NR=32）JIT 生成 | AVX-512 GEMM 数值正确 | 高 — 模式与 AVX2 一致 |

### 1.2 JIT 自动生成软件流水线化 K-loop

**现状**：JIT 生成的 K-loop 是简单的 load→FMA 序列，无预取、无 load/compute 重叠。

**目标**：JIT 自动生成 load(k+1) 与 compute(k) 重叠的流水线化 K-loop。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 1.2.1 | K-loop 展开 2 次，交错 load/FMA（AVX2 6×16） | IPC 提升 ≥10%（perf stat 验证） | 中 — 指令调度需实验 |
| 1.2.2 | 添加 L1/L2 prefetch 指令生成（`prefetcht0`/`prefetcht1`） | 大矩阵（M,N,K≥1024）效率提升 | 中 — prefetch 距离参数化 |
| 1.2.3 | AVX-512 14×32 K-loop 软件流水线 | AVX-512 路径同等优化 | 中 |

### 1.3 JIT Epilogue Injection 完整化

**现状**：AVX2 路径 `emit_trace_on_accumulator()` 已实现。AVX-512 路径 epilogue 未适配 zmm。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 1.3.1 | AVX-512 `emit_trace_on_accumulator` zmm 路径 | GEMM+SiLU epilogue AVX-512 数值正确 | 高 — 模式与 AVX2 一致 |
| 1.3.2 | AVX-512 `emit_elementwise_trace_body` zmm 路径 | elementwise 链 AVX-512 数值正确 | 高 |
| 1.3.3 | 边界处理：M/N 尾部 masked store（AVX2 + AVX-512） | 非对齐矩阵数值正确 | 中 |

### 1.4 aarch64 路径补全

**现状**：BLIS 5 级循环 + TraceOp→NEON 映射已实现，但 `emit_plan` 的 LoopFusion 分支未接入。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 1.4.1 | `emit_plan` LoopFusion 分支接入 `emit_elementwise_loop` | SiLU→VecMul→VecAdd 链 E2E 数值正确 | 高 |
| 1.4.2 | aarch64 GEMM epilogue injection 接线 | GEMM+SiLU epilogue NEON 数值正确 | 高 |
| 1.4.3 | aarch64 E2E 测试：至少 5 个 fusion pattern | 测试全绿 | 高 |

### MS-1 验收标准

| 指标 | 当前 | 目标 | 测量方法 |
|------|------|------|---------|
| JIT GEMM 数值正确（AVX2） | 部分验证 | 全尺寸正确（tolerance ≤1e-4 vs scalar_gemm） | `cargo test` |
| JIT GEMM 数值正确（AVX-512） | 基础框架 | 全尺寸正确 | `cargo test` |
| JIT GEMM+epilogue 数值正确 | AVX2 已验证 | AVX2 + AVX-512 + NEON 全验证 | `cargo test` |
| aarch64 elementwise 链 E2E | 未接线 | ≥5 个 fusion pattern 测试通过 | `cargo test` |
| JIT 生成代码含向量化 pack | 无 | pack_a + pack_b 均为 JIT 内联生成 | 代码审查 |
| JIT 生成代码含软件流水线 | 无 | K-loop load/compute 交错 | 反汇编验证 |

### MS-1 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| JIT 生成的 pack 例程性能不如手写 | 中 | 可能需要多轮迭代 | 以手写 ASM pack 为性能参考，逐步逼近 |
| K-loop 软件流水线的最优展开因子难以确定 | 中 | 可能只达到 75-80% | 留给 MS-2 autotuning 自动搜索 |
| aarch64 dynasm-rs 指令覆盖不全 | 低 | 某些 NEON 指令需要 raw encoding | 已有 tanh/log 的完整 NEON 生成作为参考 |

---

## MS-2: Autotuning→JIT 闭环

> **优先级**：P0 🔴
> **预计工期**：3-4 周
> **依赖**：MS-1（JIT 能生成完整代码后才能 autotuning）
> **核心目标**：JIT GEMM 效率尽可能逼近理论峰值。

### 2.1 Autotuning 搜索空间扩展

**现状**：autotuning 框架可搜索 blocking 参数（KC/MC/NC），但搜索空间只覆盖 ASM driver 路径。

**目标**：TVM 式闭环 — 搜索参数 → JIT 编译 → 测量 → 反馈最优参数。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 2.1.1 | `measure.rs` 增加 JIT codegen 测量路径 | `tune_gemm()` 可测量 JIT 生成的 GEMM | 高 |
| 2.1.2 | 搜索空间增加 JIT 特有参数（K-loop 展开因子、prefetch 距离、寄存器分配策略） | 搜索空间 ≥ 5 维 | 中 |
| 2.1.3 | `DeviceProfile::gemm_blocking()` 增加 `WisdomDb` 查询路径 | 有 wisdom 时用 autotuning 结果，无 wisdom 时回退公式计算 | 高 |

### 2.2 JIT Codegen 参数化

**现状**：JIT 生成代码的 blocking 参数来自 `DeviceProfile` 的公式计算，不可调。

**目标**：JIT codegen 的所有性能关键参数均可由 autotuning 注入。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 2.2.1 | `GemmUnit` 增加 `TuningParams`（KC/MC/NC/unroll_factor/prefetch_distance） | JIT 生成代码使用 autotuned 参数 | 高 |
| 2.2.2 | JIT K-loop 展开因子参数化（1/2/4） | 不同展开因子生成不同代码 | 高 |
| 2.2.3 | JIT prefetch 距离参数化 | 不同 prefetch 距离生成不同代码 | 高 |

### 2.3 测量与反馈闭环

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 2.3.1 | `tune_gemm()` 结果持久化到 `~/.gllm/wisdom.json` | 重启后自动加载最优参数 | 高 |
| 2.3.2 | JIT `emit_gemm_blis()` 从 `WisdomDb` 读取 autotuned 参数 | JIT GEMM 使用 autotuned KC/MC/NC + 展开因子 | 高 |
| 2.3.3 | `cargo bench --bench bench_fusion` 自动使用 autotuned 参数 | fusion benchmark 效率 ≥ 手动参数的 95% | 高 |

### MS-2 验收标准

| 指标 | 当前 | 目标 | 测量方法 |
|------|------|------|---------|
| JIT GEMM 效率 (1024³, AVX2) | 未测 | 尽可能逼近理论峰值 | `cargo bench` vs 理论峰值 |
| JIT GEMM 效率 (4096³, AVX2) | 未测 | 尽可能逼近理论峰值 | 同上 |
| JIT GEMM 效率 vs 手写 ASM | 未测 | ≥ 手写 ASM 的 95% | JIT vs ASM 对比 benchmark |
| autotuning→JIT 闭环 | 未连接 | tune_gemm 可测量 JIT 代码并反馈参数 | 手动验证 |
| 非对齐矩阵 (1000³) | 未测 | 尽可能逼近理论峰值 | `cargo bench` |

### MS-2 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| autotuning 搜索空间过大导致搜索时间长 | 中 | 首次 tuning 可能需要数分钟 | 分层搜索：先粗搜 blocking，再细搜展开因子 |
| AVX-512 降频导致实际效率低于预期 | 中 | AVX-512 路径可能不如 AVX2 | autotuning 中加入 AVX-512 vs AVX2 对比，自动选择 |
| JIT 生成代码的指令调度不如手写 ASM | 中 | 可能停在 80% | 分析手写 ASM 的指令调度模式，在 JIT 中复现 |

---

## MS-3: 推理栈可用

> **优先级**：P1 🟡
> **预计工期**：3-4 周
> **依赖**：MS-1（GEMM 数值正确是推理正确的基础）；MS-2（GEMM 性能是推理性能的基础）
> **核心目标**：Llama-7B 可跑，使用 JIT 编译的算子。

### 3.1 KV Cache 完善

**现状**：`kv_cache.rs` 278 行，Page 结构 + 基础 alloc/append/reset。缺少 swap_out/swap_in 实现、多 batch 支持。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 3.1.1 | KV Cache append/read 完整实现 | 单 batch 单 sequence 正确 | 高 |
| 3.1.2 | swap_out/swap_in 实现 | 长序列不 OOM | 高 |

### 3.2 Attention 实现

**现状**：`flash_attn.rs` 389 行，FlashAttention 算法框架。`cpu_backend.rs` 中 attention 使用简化实现。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 3.2.1 | GQA (Grouped Query Attention) 完整实现 | 数值正确（vs PyTorch reference） | 高 |
| 3.2.2 | FlashAttention CPU 实现（tiled softmax + 分块 QK^T） | 长序列（seq_len=2048）内存 O(N) 而非 O(N²) | 中 — 算法复杂 |
| 3.2.3 | Attention + KV Cache 集成 | 自回归生成正确 | 高 |

### 3.3 decoder_forward 接通

**现状**：`cpu_backend.rs` 的 `decoder_forward` 有完整的 operator-by-operator 路径，但 attention 部分简化。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 3.3.1 | decoder_forward 完整实现（RMSNorm→QKV→RoPE→Attention→FFN） | 单 layer forward 数值正确 | 高 |
| 3.3.2 | 多 layer 堆叠 + final_norm + lm_head | 完整 Llama-7B forward pass | 高 |
| 3.3.3 | 简单 greedy sampling | 可生成文本（质量不要求） | 高 |

### 3.4 C FFI 补全

**现状**：FFI 函数签名已定义，`gllm_decoder_forward` 未实现。

| # | 任务 | 验收标准 | AI 可辅助度 |
|---|------|---------|-----------|
| 3.4.1 | `gllm_decoder_forward` 实现 | C 调用方可执行推理 | 高 |
| 3.4.2 | `cbindgen` 生成 C 头文件 | `gllm_kernels.h` 自动生成 | 高 |

### MS-3 验收标准

| 指标 | 当前 | 目标 | 测量方法 |
|------|------|------|---------|
| Llama-7B 单 token 推理 | 不可用 | 可运行，输出合理 token | 手动验证 |
| decoder_forward 数值正确 | attention 简化 | vs PyTorch reference ≤1e-3 | 自动化测试 |
| C FFI 可用 | 骨架 | C 程序可调用推理 | 集成测试 |
| 推理吞吐 (tokens/s) | N/A | ≥ llama.cpp 的 60% | benchmark |

### MS-3 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| FlashAttention CPU 实现复杂度高 | 中 | 可能延期 2 周 | 先用 naive attention 跑通 E2E，再优化为 flash |
| 模型权重加载格式兼容性 | 中 | 需要适配 GGUF/safetensors | 先支持 safetensors（简单），GGUF 后续 |
| 多模型架构差异 | 低 | Mistral/Phi 需要额外工作 | Phase 4 只做 Llama，多模型留给 Phase 5 |

---

## 不做的事（Phase 4 范围外）

| 事项 | 原因 | 何时做 |
|------|------|--------|
| 手写更多 ASM 微内核 | JIT 是唯一性能交付路径，手写 ASM 只作参考 | 不做 |
| 手动优化 pack 例程 | pack 由 JIT 自动生成 | 不做 |
| 手动优化 K-loop 软件流水线 | 由 JIT 自动生成 + autotuning 搜索 | 不做 |
| GPU 后端（CUDA/Metal） | ROI 低，CPU JIT 路径未成熟 | Phase 6+ |
| 多模型架构（Mistral/Phi/Qwen/Gemma） | 先把 Llama 做到极致 | Phase 5 |
| 多 GPU / 张量并行 | 超出当前范围 | Phase 7+ |
| SVE (aarch64) | dynasm-rs 不支持 | 等上游支持 |
| AMX (x86_64) | x86_amx.rs 489 行已有探索，但 AMX 硬件覆盖率低 | Phase 5 评估 |
| Tree Attention / 推测解码 | 业务层优化 | 上层引擎负责 |

---

## AI 辅助编码的务实评估

| 工作类型 | AI 可辅助度 | 说明 |
|----------|-----------|------|
| JIT codegen 框架代码（BLIS 循环、pack 生成、寄存器分配） | 高 | 模式化代码，可参考 AVX2 路径生成 AVX-512 路径 |
| TraceOp→SIMD 映射 | 高 | 查表式映射，AI 可批量生成 |
| JIT K-loop 软件流水线生成 | 中 | 指令交错模式可参考手写 ASM，但最优调度需实验 |
| JIT pack 例程生成 | 高 | 转置模式明确，可参考现有 pack_a AVX2 实现 |
| autotuning 闭环集成 | 高 | 纯 Rust 逻辑代码 |
| 推理栈（KV Cache、Attention、decoder_forward） | 高 | 算法明确，参考实现丰富 |
| Prefetch 距离参数化 | 中 | 参数化框架 AI 可写，最优值需 autotuning |
| 性能回归诊断 | 低 | 需要 perf/vtune 分析，AI 无法直接操作 |

**结论**：MS-1 约 75% 的工作 AI 可高效辅助（JIT codegen 框架、pack 生成、epilogue 适配），25%（K-loop 调度优化）需要人工实验。MS-2 大部分 AI 可辅助（autotuning 框架是纯逻辑代码）。MS-3 约 80% AI 可辅助。

---

## 依赖关系图

```
MS-1.1 (JIT BLIS 循环 + pack 生成) ─────────────┐
MS-1.2 (JIT K-loop 软件流水线) ──────────────────┤
MS-1.3 (JIT epilogue AVX-512 完整化) ────────────┤
MS-1.4 (aarch64 路径补全) ───────────────────────┤
                                                  ▼
                                     MS-1 验收: JIT GEMM 数值正确
                                                  │
                                 ┌────────────────┤
                                 ▼                ▼
                    MS-2.1 (搜索空间扩展)    MS-3.1 (KV Cache)
                    MS-2.2 (JIT 参数化)      MS-3.2 (Attention)
                    MS-2.3 (测量反馈闭环)    MS-3.3 (decoder_forward)
                                 │            MS-3.4 (C FFI)
                                 ▼                    │
                    MS-2 验收: JIT GEMM 逼近峰值          ▼
                                             MS-3 验收: Llama-7B 推理

注: MS-1.4 (aarch64) 可与 MS-1.1-1.3 (x86_64) 并行。
    MS-3 的 KV Cache/Attention 实现可与 MS-1 并行启动（不依赖 GEMM 性能）。
    MS-3 的性能验收依赖 MS-2。
```

---

## 测试策略

### 新增测试需求

| 里程碑 | 测试类型 | 数量 | 说明 |
|--------|---------|------|------|
| MS-1 | JIT GEMM 数值正确性（vs scalar_gemm） | 10 | 各尺寸 × AVX2/AVX-512/NEON |
| MS-1 | JIT GEMM+epilogue 数值正确性 | 8 | SiLU/GELU/ReLU × 各 ISA |
| MS-1 | JIT elementwise 链 E2E | 5 | 各 fusion pattern |
| MS-1 | JIT pack 例程正确性 | 4 | pack_a/pack_b × AVX2/AVX-512 |
| MS-2 | autotuning→JIT 集成测试 | 3 | wisdom 读写、参数注入、JIT 使用 |
| MS-2 | JIT GEMM 性能回归测试 | 5 | 不同尺寸的效率下限断言 |
| MS-2 | JIT vs 手写 ASM 对比测试 | 3 | 效率比 ≥ 95% |
| MS-3 | KV Cache 单元测试 | 5 | append/read/swap/多 seq |
| MS-3 | Attention 数值正确性 | 5 | GQA、不同 head 配置 |
| MS-3 | decoder_forward E2E | 3 | 单 layer、多 layer、完整模型 |

**目标**：从当前 668 个测试增长到 ≥720 个，全绿。

---

## 成功标准总结

| 维度 | Phase 4 完成时 | 对标 |
|------|---------------|------|
| JIT GEMM 效率 | 尽可能逼近理论峰值（JIT 生成代码） | 追平 OpenBLAS |
| JIT vs 手写 ASM | ≥95% 手写 ASM 效率 | 证明 JIT 路径可行 |
| 编译器 E2E | x86_64 AVX2+AVX-512 + aarch64 NEON 全路径可用 | 接近 TVM 编译闭环 |
| 推理能力 | Llama-7B 可跑，≥ llama.cpp 60% 吞吐 | 基础可用 |
| 测试覆盖 | ≥720 测试全绿 | 企业级 |
| 成熟度评分 | 4.5/5 | 从 3.5 提升 1.0 |
