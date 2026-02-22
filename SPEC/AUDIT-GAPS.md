# SPEC 设计细节缺口审计（修订版）

> 审计范围：SPEC/01-05 全部文档 vs gllm-kernels + gllm 代码库实际状态
> 修订：基于 gllm/src/graph/types.rs 的 FusedGraph 实际定义、dispatch/device_profile.rs 实际实现
> 再修订：Phase 0 重新引入为「二进制符号执行」，OpSemanticsKind 删除，改为 OpTrace 自动提取

---

## 修正：已排除的误判

- ~~GAP-12: FusedGraph 未定义~~ → **已存在于 gllm 项目**，但当前未传给 gllm-kernels（见下方 GAP-A）
- ~~GAP-14: DeviceProfile 不一致~~ → **代码已完整**，SPEC 只是用了简写（`profile.l1_cache_bytes` = `profile.kernel_config.l1d`），唯一缺 `num_simd_regs` 字段
- ~~GAP-16/17: 多线程未设计~~ → 代码已依赖 rayon，GEMM driver 已有并行，属于工程实现细节而非 SPEC 缺口
- ~~GAP-08: CodeFragment 寄存器重映射~~ → CodeFragment 已删除，编译器从 OpTrace 程序化生成全新代码，无需重映射
- ~~GAP-09: Relocation 结构体~~ → CodeFragment 已删除，常量池由 MachineCodeEmitter 直接管理
- ~~§C: 类型定义缺失（InstrMnemonic/Operand/DecodedInstr 等）~~ → 这些类型不再需要，符号执行使用 iced-x86 Decoder 原生类型

---

## A. 接口桥接 — GLLM FusedGraph → gllm-kernels 编译器入口

### GAP-A: FusedGraph 传递路径不存在

**现状**：
- gllm 有完整的 `FusedGraph`（`gllm/src/graph/types.rs`），包含 7 种 FusedOp + AtomicOp
- 但 gllm 当前只用 FusedGraph 做内部优化统计，**不传给 gllm-kernels**
- gllm → gllm-kernels 的接口是 `ModelConfig → LayerIR`
- SPEC 设计的编译器入口 `compile_fused_graph(graph: &FusedGraph, ...)` 无对应代码

**需要你决策**：
1. gllm 的 FusedGraph 直接作为编译器输入？还是 gllm-kernels 定义自己的图 IR？
2. 如果复用 gllm 的 FusedGraph，gllm-kernels 需要依赖 gllm 的类型（循环依赖风险），还是抽取到共享 crate？
3. gllm 的 FusedOp 粒度（FlashAttention/SwiGLU/GQA 等高层融合）vs SPEC 设计的粒度（MatMul/RmsNorm/SiLU 等原子算子）— 编译器期望哪个粒度的输入？

**分析**：gllm 的 FusedOp 已经是高层融合（FlashAttention = QKV + Softmax + Attention + Output），而编译器的 Phase 1 期望的是原子算子级别的 DAG。两者粒度不匹配。可能的方案：
- gllm 传 FusedGraph 前先 "展开" 高层融合为原子算子 DAG
- 或 gllm-kernels 定义自己的 `CompilerGraph`，由 gllm 负责转换

### GAP-B: AtomicOp 的 op_type 是 String，缺乏类型安全

**现状**：gllm 的 `AtomicOp { op_type: String }` 用字符串标识算子类型（"MatMul"、"Add" 等）。Phase 1 需要将每个算子查 `ScalarOpRegistry` 获取 `OpTrace`，字符串匹配容易出错。

**建议**：gllm-kernels 侧定义强类型 `OpKind` 枚举，gllm 负责 String → OpKind 转换。

---

## B. DeviceProfile 补充

### GAP-15: DeviceProfile 补充 num_simd_regs

- 代码 `DeviceProfile` 缺 `num_simd_regs` 字段（可从 `isa` 推导：AVX2→16, AVX-512→32, NEON→32）
- 编译器的寄存器压力检查需要此字段
- **建议**：加一个 helper method `fn num_simd_regs(&self) -> usize` 到 DeviceProfile

---

## C. 决策记录

| GAP | 决策 | 理由 |
|-----|------|------|
| **GAP-A** | **gllm 负责展开**：gllm 将高层 FusedOp（FlashAttention/SwiGLU 等）展开为原子算子 DAG 后传入 gllm-kernels | gllm 拥有模型结构知识，展开逻辑属于图优化层；gllm-kernels 只关心原子算子的语义和融合 |
| **Phase 0 回归** | **重新引入 Phase 0 为「二进制符号执行」**：对 `extern "C"` 纯标量函数做二进制分析，自动提取 OpTrace | 消除编译器内置算子知识的维护负担；新增算子只需写标量函数并注册，编译器自动分析 + 生成最优代码 |
| **OpSemanticsKind 删除** | **替换为 OpTrace + ComputePattern**：算子分类从 OpTrace.pattern 自动推导，不再手动维护映射表 | OpSemanticsKind 是硬编码分类标签，每新增算子需改编译器内部；OpTrace 是自动提取的完整计算结构 |
| **ActivationKind 删除** | **替换为 TraceOp 序列**：激活函数不再是枚举标签，而是 OpTrace.body 中的 TraceOp 指令序列 | 消除 emit_activation 的 match 分支，代码生成统一为 TraceOp → SIMD 指令映射 |
| **FusedLoopOpKind 删除** | **替换为 OpTrace**：FusedLoopOp 直接携带 OpTrace，代码生成遍历 TraceOp 序列 | 与上同理，消除硬编码算子类型枚举 |
| **标量函数形态** | **由数学定义决定，不是设计选择**：SiLU 就是 `x/(1+exp(-x))`，RMSNorm 天然带循环和归约，GEMM 天然是三重循环 | 符号执行面对的二进制复杂度由算子的数学结构决定，不需要人为约束函数形态 |
| **fn_ptr 双重角色** | **ScalarOpRegistry 的 fn_ptr 同时是执行入口和分析入口**：DAG 中 `OpKind::Silu → registry.get(Silu) → fn_ptr` 既可调用（正确性基准）又可反汇编（符号执行输入） | 解决「DAG 里的算子对应的函数实现在哪」和「要分析的二进制在哪」是同一个问题 |
| **GEMM 模式识别** | **通过三重循环 + FMA 累加模式自动识别**：符号执行看到三重嵌套循环 + 内层 `a*b+c` 累加 → `ComputePattern::Gemm` | 不需要"知道这是矩阵乘法"，只需识别循环嵌套结构和累加模式；与 elementwise（单循环）、reduction（循环+累加器）的识别逻辑一致 |
| **opt-level 控制** | **不需要单独控制**：标量函数与 gllm-kernels 共享 crate 的 opt-level，符号执行引擎处理编译器优化后的输出 | 标量函数结构极简（几条指令或简单循环），即使 opt-level=2/3 也不会产生难以分析的代码 |
| **Gemm.epilogue 字段** | **笔误，已删除**：`ComputePattern::Gemm` 不再携带 epilogue 字段。Epilogue 是 Phase 2 融合决策的结果，由消费者算子的 OpTrace.body 提供 | GEMM 标量函数本身不含 epilogue，epilogue 来自融合后的消费者算子 |
| **数学函数逼近** | **从现有手写 asm 提取多项式系数**：exp/tanh/gelu 的 SIMD 多项式逼近系数从已有手写汇编微内核中提取，硬编码为 `MathApprox` 查找表 | 现有手写 asm 的系数经过仔细调优，直接复用；编译器内置标准逼近方案（Cephes/SLEEF 级别精度） |
| **Tile-Level Fusion scratch buffer** | **MC 行结果写入 scratchpad normed 区域**：TileLevelFusion 时 MC×K×sizeof(E) 写入 scratchpad，紧接着被 pack_a 消费；weight 通过 JIT 函数参数传入（graph input）；RMSNorm 逐行独立，按 MC 切分正确 | scratchpad 已由 planner 分配；weight 是只读 graph input，L2 热驻留；逐行独立保证 tile 切分的数学等价性 |
| **TileLevelFusion vs ComputeRoot** | **硬件驱动二选一**：前驱输出 > L1×0.75 → TileLevelFusion（嵌入 MC 循环）；≤ L1×0.75 → ComputeRoot（先算完，结果留 L1）。tile_rows = MC，由 DeviceProfile 的 cache 层级决定 | 同一 DAG 在不同硬件上产生不同融合策略（SPEC §8.5 Step 3 的三个硬件示例）；fuse() 必须接收 DeviceProfile |
| **EpilogueKind → OpTrace** | **删除 EpilogueKind 枚举**：EpilogueOp 不再用 BiasAdd/Activation/ResidualAdd 标签，直接携带消费者算子的 OpTrace，Phase 3 遍历 TraceOp 序列生成 SIMD 指令 | 与 OpSemanticsKind/ActivationKind 删除同理，消除硬编码枚举，统一为 OpTrace 驱动 |
| **GAP-12: extra_input_idx** | **Option\<usize\> 够用**：原子算子最多一个额外输入（VecMul→weight, VecAdd→residual, BiasAdd→bias），不存在单个原子算子需要两个额外输入的情况 | 保持简单，如果未来出现多额外输入算子再扩展为 Vec |
| **GAP-13: emit_trace_ops scratch** | **签名改为 `emit_trace_ops(ops, reg, scratch: &[Register])`**：调用方告知可用 scratch 寄存器集合，GEMM epilogue 场景下累加器占大部分寄存器，scratch 只有剩余几个 | emitter 内部无法知道上下文已占用哪些寄存器，必须由调用方传入 |
| **GAP-14: Injective ComputePattern** | **新增 `ComputePattern::Injective { body, num_inputs, num_outputs }`**：覆盖 RoPE 等多输入多输出逐元素变换。OpClass 推导为 Injective（可融合进消费者，但不能作为 epilogue 注入 GEMM） | RoPE 是 4 输入 2 输出，不适合 Elementwise/BinaryElementwise；Injective 在 TVM 融合规则中与 ElemWise 行为相同但语义更精确 |
| **GAP-15: 并行化层级** | **Phase 2 Step 3.5 决定并行策略**：GEMM→NC 循环并行，Elementwise→按元素均分，小数据→Sequential。JIT 代码是单线程的，调用方 thread pool 按 ParallelStrategy 分发 tile | JIT 代码不含线程同步逻辑（简化代码生成），并行维度选择需要 cache 层级信息（Phase 2 已有 DeviceProfile） |

## D. 执行状态

| 项目 | 状态 | 修改文件 |
|------|------|---------|
| Phase 0 符号执行引入 | ✅ 完成 | SPEC/02 §8 新增 Phase 0 + ScalarOpRegistry；SPEC/03 §12.1 新增 OpTrace/ComputePattern/TraceOp |
| OpSemanticsKind → OpTrace | ✅ 完成 | SPEC/01 §6、SPEC/02 §8、SPEC/03 §12、SPEC/05 §7、CLAUDE.md 全部更新 |
| §D-10/11 路径 B 清理 | ✅ 完成 | 路径 B 代码（templates.rs/emitter.rs/reloc.rs）及 SPEC 引用已删除 |
| GAP-A FusedGraph 桥接 | ✅ 完成 | SPEC/03 新增 §11.5（CompilerGraph/CompilerNode/CompilerOp/TensorRef/展开规则表） |
| Phase 0 澄清：fn_ptr 双重角色 + 标量函数形态 + GEMM 识别 | ✅ 完成 | SPEC/02 §8.3 补充核心洞察、数学定义决定形态、GEMM 模式识别 |
| GAP-1/2/10 决策落地 | ✅ 完成 | opt-level 不需单独控制；Gemm.epilogue 笔误已删（SPEC/03）；数学逼近从手写 asm 提取 |
| Phase 2 硬件驱动融合 | ✅ 完成 | SPEC/02 §8.5 补充 fuse() 签名 + TileLevelFusion vs ComputeRoot 决策规则 + scratch buffer 方案；SPEC/03 §12.3 新增 ComputeRoot 变体 + tile_rows 字段 |
| EpilogueKind → OpTrace | ✅ 完成 | SPEC/03 §12.3 删除 EpilogueKind 枚举，EpilogueOp 直接携带 OpTrace |
| GAP-12/13/14/15 决策落地 | ✅ 完成 | GAP-13: emit_trace_ops 加 scratch 参数（SPEC/03 §12.4）；GAP-14: ComputePattern 新增 Injective + OpClass 推导表（SPEC/03 §12.1/12.2）；GAP-15: Phase 2 Step 3.5 并行化 + ParallelStrategy 枚举（SPEC/02 §8.5 + SPEC/03 §12.3） |
