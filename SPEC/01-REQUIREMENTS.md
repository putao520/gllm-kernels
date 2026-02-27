# gllm-kernels 需求清单

> **📌 SSOT**: 本文档是 gllm-kernels 算子库的需求唯一真源。

## 定位

**极限性能算子库 + JIT 编译器**：提供逼近硬件理论极限的底层计算原语，以及算法意图编译器（JIT）自动融合优化。当前聚焦 CPU 后端，GPU 后端（CUDA/Metal）为规划中的未来工作（见 SPEC/04-GPU-BACKEND.md）。推理后端（见 SPEC/05-LAYER2-INFERENCE.md）提供 Layer 2 抽象层。

---

## 1. 性能需求（REQ-PERF）🚨 铁律

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-PERF-001** | Compute-bound 算子逼近计算峰值 | GEMM 尽可能逼近理论 FLOPS 峰值 | 🟡 unpacked 42%, prepacked 59%。差距根因：blocking 参数未经 autotuning 闭环、pack_b 标量路径、微内核无软件流水线。见 PLAN-phase4 MS-1 |
| **REQ-PERF-002** | Memory-bound 算子逼近带宽峰值 | 激活/归一化/BLAS-1 尽可能逼近内存带宽峰值 | 🟡 GEMV 67-76%，activation ALU-limited 7-13 GiB/s（需 JIT Loop Fusion 消除中间 writeback） |
| **REQ-PERF-003** | 量化算子逼近瓶颈极限 | 量化 GEMV/GEMM 尽可能逼近瓶颈极限 | 🟡 ASM 微内核已写（Q4K/Q8K），效率未经系统测量。intrinsics 路径（Q2/Q1/IQ/AWQ/GPTQ/Squeeze）效率未知 |
| **REQ-PERF-004** | 手写汇编微内核 | GEMM、量化 GEMV/GEMM 必须使用 `global_asm!` 手写汇编 | 🟢 已完成（8 个 global_asm! 微内核） |
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
| **REQ-OPS-011** | gemm | `(a: &[E], b: &[E], c: &mut [E], m, n, k)` | Compute | 🟢 ASM 微内核 (AVX2 6×16, AVX-512 14×32, NEON 8×12) |
| **REQ-OPS-012** | gemm_bias | `(a, b, bias, c, m, n, k)` | Compute | 🟢 ASM fused path |
| **REQ-OPS-013** | gemm_prepacked | `(a, packed_b, c, m, n, k)` | Compute | 🟢 ASM driver |
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
| **REQ-OPS-070** | gemv_q8 | INT8 | 🟢 ASM 微内核 (AVX2/AVX-512/NEON) |
| **REQ-OPS-071** | gemv_q4 | INT4 packed | 🟢 ASM 微内核 (AVX2/AVX-512/NEON) |
| **REQ-OPS-072** | gemv_q2 | INT2 packed | 🟡 intrinsics |
| **REQ-OPS-073** | gemv_q1 | INT1 packed | 🟡 intrinsics |
| **REQ-OPS-074** | gemm_q8 | INT8 | 🟡 intrinsics |
| **REQ-OPS-075** | gemm_q4 | INT4 packed | 🟡 intrinsics |
| **REQ-OPS-076** | kquant_matmul | K-Quant 系列 | 🟢 Q4K/Q8K ASM, 其余 intrinsics |
| **REQ-OPS-077** | iq_matmul | IQ 系列 | 🟡 intrinsics |
| **REQ-OPS-078** | awq_matmul | AWQ4 | 🟡 intrinsics |
| **REQ-OPS-079** | gptq_matmul | GPTQ4 | 🟡 intrinsics |
| **REQ-OPS-080** | squeeze_matmul | SqueezeLLM | 🟡 intrinsics |

---

## 3. 架构需求（REQ-ARCH）

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-ARCH-001** | 纯 Rust 零外部依赖 | `cargo install` 一键安装，禁止 faer/OpenBLAS/MKL | 🟢 已完成 |
| **REQ-ARCH-002** | 手写汇编微内核 | GEMM/量化 GEMV 使用 `global_asm!` | 🟢 已完成（8 个微内核） |
| **REQ-ARCH-003** | 运行时 ISA 分发 | OnceLock + CPUID 检测，启动后零开销 | 🟢 已完成 |
| **REQ-ARCH-004** | 泛型精度支持 | f32/f16/bf16 通过 `<E: Element>` 编译时单态化 | 🟢 已完成 |
| **REQ-ARCH-005** | 宏驱动代码生成 | 非热路径通过四层宏架构批量展开 | 🟢 已完成 |
| **REQ-ARCH-006** | 多 ISA 支持 | AVX2 / AVX-512 / NEON / Scalar | 🟢 已完成 |

---

## 4. ISA 覆盖需求（REQ-ISA）

| ISA | f32 | f16 | bf16 | 手写 asm | 状态 |
|-----|-----|-----|------|---------|------|
| **Scalar** | ✅ | ✅ 软件转换 | ✅ 软件转换 | N/A | 🟢 |
| **AVX2** | ✅ | ✅ F16C | ✅ 位转换 | ✅ GEMM 6×16 + Q4K/Q8K GEMV | 🟢 |
| **AVX-512** | ✅ | ⚡ FP16 原生 | ⚡ BF16 原生 | ✅ GEMM 14×32 + Q4K/Q8K GEMV | 🟢 |
| **NEON** | ✅ | ⚡ FP16 原生 | ✅ 位转换 | ✅ GEMM 8×12 + Q4K/Q8K GEMV | 🟢 |

---

## 5. 范围边界

### 5.1 不在 Layer 1 算子库范围内

以下功能属于上层推理引擎，不在 Layer 1 算子库范围内：

- FlashAttention / Paged Attention
- KV Cache 管理与调度
- 融合算子（fused_qkv_rope, fused_ffn, fused_gate_up_swiglu 等）
- Embedding lookup
- Sampling (argmax, top-k, top-p)
- 推理调度、批处理、Swap 管理
- ONNX 文件加载与解析（由 GLLM 负责）
- Tree Attention / 推测解码

### 5.2 规划中的未来工作

以下功能在本项目范围内，但尚未完成：

- CUDA / Metal GPU 后端（见 SPEC/04-GPU-BACKEND.md）🔴
- Layer 2 推理后端（见 SPEC/05-LAYER2-INFERENCE.md）🟡 基础实现

---

## 6. 编译器需求（REQ-COMPILER）

> 算法意图编译器：算子的唯一定义来源是 `extern "C"` 纯标量函数，编译器通过二进制符号执行自动提取计算结构（OpTrace），然后根据 DeviceProfile 生成最优融合 SIMD 代码。核心原则：**标量定义 → 二进制分析 → 融合决策 → 全新代码生成**。

### 6.1 标量算子定义

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-SCALAR-001** | 标量算子 C ABI | 所有算子必须有 `extern "C"` 纯标量实现，注册到 `ScalarOpRegistry`。纯标量运算，无 SIMD 指令 | 🟢 已完成（`scalar-ops/src/` 17 个 `extern "C"` 函数，全部注册到 `ScalarOpRegistry::with_defaults()`） |
| **REQ-SCALAR-002** | 编译约束 | 标量函数编译用 `opt-level=1`（保留循环结构，消除冗余，不做向量化），确保符号执行可分析 | 🟢 已完成（`gllm-scalar-ops` 独立 subcrate，`[profile.release.package.gllm-scalar-ops] opt-level = 1`，objdump 验证零 ymm/zmm 指令） |
| **REQ-SCALAR-003** | 正确性基准 | 标量实现作为 JIT 代码的 golden reference。JIT 生成的代码与标量实现数值误差 ≤ 1e-4（f32）/ 1e-2（f16） | 🟢 已完成（E2E 测试对比标量 reference，tolerance ≤ 1e-4） |

### 6.2 二进制符号执行（Phase 0）

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-SYMEXEC-001** | 符号执行覆盖 | 正确提取 elementwise / binary_elementwise / reduction / normlike / gemm 五类计算模式的 OpTrace | 🟢 已完成（`src/compiler/symexec/engine.rs` 1104 行，支持 FMA/比较/位运算/栈溢出/常量池/libm 识别） |
| **REQ-SYMEXEC-002** | OpTrace 缓存 | 同一算子不重复分析。`ScalarOpRegistry::get_trace()` 首次调用触发分析，之后从缓存返回 | 🟢 已完成（`ScalarOpRegistry::with_defaults()` 预填充 17 个算子的 OpTrace） |
| **REQ-SYMEXEC-003** | 分析延迟 | 单个算子符号执行 < 1ms（标量函数通常 < 100 条指令） | 🟢 已完成（symexec_benchmark.rs 验证：2.5-15μs/算子，远低于 1ms 预算） |

### 6.3 语义 DAG 构筑与数据流分析（Phase 1）

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-COMPILER-001** | 算子绑定 | 对 CompilerGraph 中每个算子查 `ScalarOpRegistry` 获取 `OpTrace`，从 `OpTrace.pattern` 自动推导算子分类（kElemWise/kReduction/kGemm/kOpaque）。未注册算子返回 `CompileError::UnsupportedOp` | 🟢 已完成（`SemanticDAG::from_graph()` 查 registry → 自动推导 `OpClass`） |
| **REQ-COMPILER-003** | 图级数据流分析 | 对 CompilerGraph 构建张量 def-use 链：每条边标注数据量（bytes）、生产者-消费者关系、是否可寄存器传递。构建后支配树用于融合组划分 | 🟢 已完成（`SemanticDAG` 含 `TensorEdge` def-use 链、消费者计数、寄存器传递标注） |

### 6.4 融合决策（Phase 2，Profile-Driven）

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-COMPILER-004** | 接收 CompilerGraph | `compile_graph(graph: &CompilerGraph, profile: &DeviceProfile) -> Result<CompiledLayer>` 接口可用，支持 GLLM 传入的任意合法 CompilerGraph（由 GLLM 将 FusedGraph 展开为原子算子 DAG） | 🟢 已完成（`InferenceCompiler::compile_graph()` 接口可用） |
| **REQ-COMPILER-005** | 算子分类自动推导 | 算子分类从 `OpTrace.pattern` 自动推导，不手动维护映射表。`ComputePattern::Elementwise` → kElemWise，`Reduction` → kReduction，`NormLike` → kReduction，`Gemm` → kGemm，`QuantDecode` → kOpaque | 🟢 已完成（`semantic_dag.rs::derive_op_class()` 从 `ComputePattern` 自动映射） |
| **REQ-COMPILER-006** | Profile-Driven 融合决策 | 基于 DeviceProfile（cache 容量、roofline ridge point、寄存器数量、SIMD 宽度）和 OpTrace（计算结构、数据量）做融合决策。三种融合模式：(a) Epilogue Injection — 取消费者 OpTrace.body 的 TraceOp 序列，在 GEMM 累加器上原地生成 SIMD 指令；(b) Tile-Level Fusion — 在 GEMM MC 循环内按 tile 计算前驱算子；(c) Loop Fusion — 多个 elementwise 算子合并为单循环，遍历每个算子的 OpTrace.body 生成指令。融合决策必须考虑：中间张量是否放得进目标 cache 层级、融合后寄存器压力是否超出可用寄存器数、生产者是否有多个消费者 | 🟢 融合决策框架完成（`fuse_with_dag()` 5 种模式 + `HwConstraintChecker` 寄存器/L1/epilogue 深度验证）。Epilogue Injection + Loop Fusion codegen 已接线。TileLevelFusion/ComputeRoot codegen 已完成。融合代价模型已实现（`Cost` 结构体：roofline FLOP 计数 + 签名字节数 → compute/memory cycles，`fusion_benefit()` 消除中间内存访问收益计算，`is_compute_bound()` roofline 分类）。L1 阈值决策：TileLevelFusion（norm 输出 > 75% L1）/ ComputeRoot（≤ 75% L1）。Elementwise 链 L1 split 防止 cache thrashing |
| **REQ-COMPILER-007** | 张量活性分析 | 对 SemanticDAG 执行张量生命周期分析（birth = 生产指令拓扑序位置，death = 最后消费指令拓扑序位置），通过区间图着色贪心算法最大化 buffer 原地复用，输出 BufferPlan | 🟢 已完成（`buffer_alloc.rs` 285 行，区间图着色 + 贪心分配） |

### 6.5 代码生成（Phase 3）

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-COMPILER-008** | TraceOp → SIMD 代码生成 | 根据 FusionPlan 通过 `MachineCodeEmitter` trait 程序化生成全新机器码。核心机制：遍历 OpTrace.body 中的 `Vec<TraceOp>`，每个 TraceOp 映射到对应的 SIMD 指令（如 `TraceOp::Add` → `vaddps`，`TraceOp::Exp` → 多项式逼近指令序列）。底层使用 iced-x86 CodeAssembler（x86_64）/ dynasm-rs（aarch64） | 🟡 x86_64 AVX2 完整实现（6307 行）：`emit_trace_ops_avx2()`、`emit_trace_on_accumulator()`、`emit_elementwise_trace_body()` 三条路径均覆盖所有 TraceOp 变体，Epilogue injection + Loop fusion 已接线。x86_64 AVX-512：基础框架（emit_gemm_tile_avx512 存在），BLIS 循环 zmm 分支未完整接线，epilogue injection 未适配 zmm。aarch64（2246 行）：BLIS 5 级循环嵌套 + tanh/log 真实 NEON 指令生成，但 emit_plan 的 LoopFusion 分支未接入 elementwise 链 |
| **REQ-COMPILER-009** | GEMM Tile-Level Fusion | GEMM 的 MC 循环内可嵌入前驱算子的 tile 计算。嵌入决策由 profile 驱动：当前驱输出 > L1 容量时启用 tile-level fusion，否则 compute_root | 🟡 aarch64 已实现（`emit_gemm_blis_neon` BLIS 5 级循环嵌套 + TileLevelFusion/ComputeRoot codegen），x86_64 待实现 |
| **REQ-COMPILER-010** | 数值一致性 | 编译器生成的 CompiledLayer 与标量函数实现（golden reference）数值误差 ≤ 1e-4（f32）/ 1e-2（f16），通过自动化回归测试验证 | 🟡 x86_64 AVX2 路径：GEMM + elementwise 链 + epilogue fusion bias 已验证（tolerance ≤ 1e-4）。AVX-512 路径：仅基础 GEMM tile 验证，epilogue/elementwise 未覆盖。aarch64 路径：GEMM 微内核数值验证通过，elementwise 链 E2E 未验证（emit_plan 未接线） |
| **REQ-COMPILER-011** | 算子知识自动提取 | 算子计算结构通过二进制符号执行自动提取（OpTrace），编译器不内置算子计算逻辑。新增算子只需写 `extern "C"` 标量函数并注册，编译器自动分析 + 生成最优代码 | 🟢 已完成（生产路径已接通：fn_ptr → iced-x86 Decoder → SymbolicExecutor → OpTrace。`decoder.rs` 927 行，自动提取 SiLU/GELU/ReLU 等激活函数的 OpTrace。`auto_register_from_symexec()` 从编译后的 scalar_ops 二进制自动分析） |

### 6.6 外部依赖与平台支持

| ID | 需求 | 验收标准 | 状态 |
|----|------|----------|------|
| **REQ-COMPILER-012** | 汇编器后端 | 两层 trait 架构：`PlatformBackend`（统一入口）→ `MachineCodeEmitter`（Phase 3 代码生成）。x86_64 使用 `iced-x86`（CodeAssembler + Decoder）；aarch64 使用 `dynasm-rs`（Assembler）。iced-x86 同时用于 Phase 0（Decoder 反汇编）和 Phase 3（CodeAssembler 代码生成） | 🟡 x86_64 iced-x86 集成完成（Phase 0 Decoder + Phase 3 CodeAssembler，AVX2 完整 + AVX-512 基础）。aarch64 dynasm-rs 实质实现（BLIS 5 级循环嵌套 + tanh/log 真实 NEON 指令），但 `MachineCodeEmitter::emit_plan()` 未完整接线（LoopFusion 分支缺失） |
| **REQ-COMPILER-013** | 平台后端 ISA 覆盖 | x86_64 后端（iced-x86）：AVX2 全指令集 + AVX-512（EVEX 编码, zmm, mask）+ FMA/F16C/VNNI/BF16/FP16。AVX2 和 AVX-512 是同一后端的寄存器宽度分支（由 DeviceProfile.isa 选择），不是独立实现。aarch64 后端（dynasm-rs）：NEON 全指令集（fmla, fmul, fadd, ld1/st1/ldp/stp, dup, sdot/udot）+ ARMv8.4+ | 🟡 x86_64: AVX2 完整支持（GEMM BLIS 5 级循环 + epilogue injection + elementwise 链），AVX-512 基础路径（emit_gemm_tile_avx512 存在，BLIS 循环/epilogue/elementwise 未完整适配 zmm）。aarch64: GEMM BLIS 5 级循环 + TraceOp→NEON 映射完整（含 tanh 15 条、log 36 条指令），elementwise 链 emit_plan 接线待补全 |
| **REQ-COMPILER-016** | JIT 延迟 | 单个 transformer layer 的编译延迟 < 100ms（含 Phase 0-3）。Phase 0 符号执行 < 1ms/算子，iced-x86 和 dynasm-rs 均为 μs 级指令编码 | 🟢 已完成（基准测试验证：symexec 2.5-15μs/算子，JIT 编译 16-237μs/layer，均远低于 100ms 预算） |
