# Layer 2-3 推理后端与 JIT 编译器设计

> **📌 SSOT**: 本文档定义 Layer 2（推理后端）和 Layer 3（JIT 编译器）的接口规范。

---

## 1. 架构总览

```
Layer 1: 原子算子 (Kernels<E> trait)
    ↑ 调用
Layer 2: 推理后端 (InferenceBackend trait)
    ↑ 调用
Layer 3: JIT 编译器 (InferenceCompiler)
    ↑ 调用
上层推理引擎 (gllm)
```

- Layer 2 提供 **fallback 路径**：逐算子组合实现 transformer forward pass
- Layer 3 提供 **JIT 路径**：将整个 transformer layer 编译为单一高性能函数
- 上层引擎优先使用 JIT 路径，不可用时回退到 fallback

---

## 2. InferenceBackend Trait（Layer 2 核心接口）

```rust
pub trait InferenceBackend: Send + Sync {
    fn init(config: &ModelConfig) -> Result<Self, InferenceError> where Self: Sized;
    fn device_kind(&self) -> DeviceKind;

    // Memory
    fn alloc(&self, num_elements: usize, dtype: DType) -> Result<DeviceTensor, InferenceError>;
    fn upload_f32(&self, src: &[f32], dst: &mut DeviceTensor) -> Result<(), InferenceError>;
    fn download_f32(&self, src: &DeviceTensor, dst: &mut [f32]) -> Result<(), InferenceError>;

    // KV Cache
    fn alloc_kv_cache(&self, batch_size: usize, max_seq_len: usize) -> Result<KvCache, InferenceError>;

    // Forward pass
    fn decoder_forward(&self, input: &DeviceTensor, positions: &DeviceTensor,
        kv_cache: &mut KvCache, weights: &ModelWeights, seq_lens: &[usize],
        output: &mut DeviceTensor) -> Result<(), InferenceError>;
    fn encoder_forward(&self, input: &DeviceTensor, positions: &DeviceTensor,
        attention_mask: &DeviceTensor, weights: &ModelWeights,
        output: &mut DeviceTensor) -> Result<(), InferenceError>;

    // Sampling
    fn sample(&self, logits: &DeviceTensor, temperature: f32, top_k: usize,
        top_p: f32, output_ids: &mut [u32]) -> Result<(), InferenceError>;

    fn sync(&self) -> Result<(), InferenceError>;
}
```

### 实现

| 后端 | 模块 | 状态 |
|------|------|------|
| `CpuInferenceBackend` | `inference/cpu_backend.rs` | 🟡 基础实现（final_norm+lm_head 已完成，attention 简化） |

> **GPU 后端**: GPU 推理不通过独立的 `InferenceBackend` 实现，而是复用同一 JIT 编译器管线（Phase 0-2 完全复用，Phase 3 新增 GPU CodeGen）。详见 SPEC/04-GPU-BACKEND.md。

---

## 3. DeviceTensor（统一张量句柄）

```rust
pub struct DeviceTensor {
    ptr: *mut u8,           // CPU: host pointer; GPU: device pointer
    len_bytes: usize,
    num_elements: usize,
    dtype: DType,
    device: DeviceKind,     // Cpu | Cuda(id) | Metal(id)
    owned: bool,            // true = Drop 时释放
}
```

- CPU 路径零开销：`as_slice<E>()` 直接返回 `&[E]`
- GPU 路径：数据留在设备端，通过 `upload_f32` / `download_f32` 传输
- 64 字节对齐分配（cache line aligned）

---

## 4. KvCache（分页 KV 缓存）

```rust
pub struct KvCache {
    pages: Vec<Page>,               // 物理页池
    free_pages: Vec<usize>,         // 空闲页栈
    layer_tables: Vec<Vec<SeqPageTable>>,  // [layer][seq] → 页表
}
```

- 页大小：16 tokens
- 每页存储：`[2(K+V), num_kv_heads, PAGE_SIZE, head_dim]`
- 支持：append / reset_seq / swap_out / swap_in
- 设计灵感：vLLM PagedAttention

---

## 5. ModelWeights（权重存储）

```rust
pub struct ModelWeights {
    pub embedding: DeviceTensor,     // [vocab_size, hidden_size]
    pub layers: Vec<LayerWeights>,   // 每层权重
    pub final_norm: DeviceTensor,    // [hidden_size]
    pub lm_head: DeviceTensor,       // [hidden_size, vocab_size]
}

pub struct LayerWeights {
    pub attn_norm: DeviceTensor,     // RMSNorm / LayerNorm weight
    pub wq, wk, wv, wo: DeviceTensor,  // Attention projections
    pub ffn_norm: DeviceTensor,      // FFN norm weight
    pub w_gate, w_up, w_down: DeviceTensor,  // FFN projections
}
```

---

## 6. FFI C ABI 规范

### 错误码

```c
enum GllmStatus {
    GLLM_OK           =  0,
    GLLM_INVALID_ARG  = -1,
    GLLM_OUT_OF_MEMORY = -2,
    GLLM_COMPILE_ERROR = -3,
    GLLM_RUNTIME_ERROR = -4,
    GLLM_UNSUPPORTED   = -5,
    GLLM_IO_ERROR      = -6,
};
```

### 函数签名

| 函数 | 签名 | 状态 |
|------|------|------|
| `gllm_backend_init` | `(config*, out*) -> i32` | 🟢 |
| `gllm_backend_free` | `(backend) -> void` | 🟢 |
| `gllm_tensor_alloc` | `(backend, n, dtype, out*) -> i32` | 🟢 |
| `gllm_tensor_free` | `(tensor) -> void` | 🟢 |
| `gllm_tensor_upload_f32` | `(backend, src*, n, tensor) -> i32` | 🟢 |
| `gllm_tensor_download_f32` | `(backend, tensor, dst*, n) -> i32` | 🟢 |
| `gllm_kv_cache_alloc` | `(backend, batch, seq, out*) -> i32` | 🟢 |
| `gllm_kv_cache_free` | `(cache) -> void` | 🟢 |
| `gllm_version` | `() -> char*` | 🟢 |
| `gllm_hw_info` | `(buf*, len) -> usize` | 🟢 |
| `gllm_decoder_forward` | `(backend, input, pos, kv, weights, seq_lens, out) -> i32` | 🔴 未实现 |

### 线程安全

- 所有 `gllm_*` 函数线程安全（`InferenceBackend: Send + Sync`）
- 同一 `GllmKvCache` 不可并发写入（需外部同步）

---

## 7. JIT 编译器架构（Layer 3）

### 编译流水线

```
JIT 编译流水线（标量函数 + 符号执行驱动编译器）
═══════════════════════════════════════════════════════════════════

ScalarOpRegistry (extern "C" 标量函数)
    │
    ▼
Phase 0: 二进制符号执行
    · iced-x86 Decoder 反汇编标量函数
    · 符号执行提取计算结构 → OpTrace
    · 首次分析后缓存
    │
    ▼
CompilerGraph (from GLLM)    DeviceProfile
    │                         │
    ▼                         ▼
Phase 1: 语义 DAG 构筑
    · CompilerOp → 查 ScalarOpRegistry → 取 OpTrace
    · OpTrace.pattern 自动推导算子分类
    · 张量 def-use 链 + 后支配树
    │
    ▼
Phase 2: Profile-Driven 融合决策
    · 融合组划分（后支配树 + TVM 规则）
    · Profile 约束检查（cache/寄存器/消费者数）
    · 三种融合模式:
      - Epilogue Injection: 取消费者 OpTrace.body，在 GEMM 累加器上原地生成 SIMD 指令
      - Loop Fusion: 遍历每个算子的 OpTrace.body 生成单循环
      - Tile-Level Fusion: 前驱 tile 计算嵌入 GEMM MC 循环
    · Tiling 参数计算 + Buffer 规划
    │
    ▼
Phase 3: 全新代码生成
    · 从 OpTrace 的 Vec<TraceOp> 直接映射到 SIMD 指令
    · iced-x86 CodeAssembler (x86_64) / dynasm-rs Assembler (aarch64)
    │
    ▼
CompiledLayer (mmap RWX)
    ↕
CompilationCache
```

编译失败（UnsupportedOp 等）时自动回退到 Layer 2 fallback（逐算子调用）。

### 语义驱动编译器

编译器接收 CompilerGraph（由 GLLM 将 FusedGraph 展开为原子算子 DAG）和 DeviceProfile，
通过二进制符号执行自动提取算子计算结构，然后根据硬件特征做融合决策，用平台汇编器程序化生成全新机器码。

**Phase 0: 二进制符号执行**

对 `ScalarOpRegistry` 中注册的 `extern "C"` 纯标量函数做二进制分析。iced-x86 Decoder 反汇编函数体，
符号执行引擎追踪 load → compute → store 数据流，识别循环结构、归约模式、多 pass 结构。
输出 OpTrace（完整计算结构描述），首次分析后缓存。

标量函数只有简单的标量指令 + 循环，符号执行复杂度极低（对比 SIMD 模板的数百条向量指令）。

**Phase 1: 语义 DAG 构筑**

将 CompilerGraph 中的每个算子查 ScalarOpRegistry 获取已缓存的 OpTrace，
从 OpTrace.pattern 自动推导算子分类（不再手动维护映射表），
构建张量 def-use 链和后支配树，为融合决策提供基础。

**Phase 2: Profile-Driven 融合决策**

融合决策完全由硬件 profile 和 OpTrace 驱动，不依赖"模板是否存在"。

**Step 1: 融合组划分（后支配树算法）**

基于 TVM 的融合规则，在后支配树上从叶子向根遍历：

| 生产者类型 | 消费者类型 | 融合规则 |
|-----------|-----------|---------|
| kElemWise | 任意 | 可融合进消费者 |
| kInjective | 任意 | 可融合进消费者 |
| kReduction | kElemWise/kInjective | 可将消费者融合为 epilogue |
| kGemm | kElemWise | 可将消费者融合为 epilogue（Epilogue Injection） |
| kGemm | kReduction | 不融合 |
| kOpaque | 任意 | 不融合 |

额外约束：
- 生产者有多个消费者 → 不融合（避免重复计算）
- 融合组内总寄存器压力 > 可用寄存器数 → 拒绝融合或拆分

**Step 2: 三种融合模式**

| 模式 | 触发条件 | 效果 |
|------|---------|------|
| Epilogue Injection | GEMM 后紧跟 elemwise 消费者，且 scratch 寄存器足够 | 在累加器寄存器上原地执行，消除 1-2 次内存往返 |
| Loop Fusion | 连续 elemwise 链，单消费者 | 单循环，数据在寄存器中流过整个链，消除中间写+读 |
| Tile-Level Fusion | 前驱输出 > L1 容量 × 0.75 且有 GEMM 消费者 | 前驱 tile 计算嵌入 GEMM MC 循环，结果留在 L1 |

当前驱输出 ≤ L1 × 0.75 时，使用 ComputeRoot（先算完，结果留在 L1），不做 Tile-Level Fusion。

**Step 3: 硬件差异示例**

同一 DAG 在不同硬件上可能产生不同的融合策略（取决于 L1 容量、寄存器数量、SIMD 宽度）。详见 SPEC/02 §8.5 Phase 2 的完整示例。

**Phase 3: 代码生成**

使用平台特定汇编器（x86_64: iced-x86 CodeAssembler / aarch64: dynasm-rs Assembler）程序化生成每一条指令。核心机制：遍历 OpTrace.body 中的 `Vec<TraceOp>`，每个 TraceOp 映射到对应的 SIMD 指令（如 `TraceOp::Add` → `vaddps`，`TraceOp::Exp` → 多项式逼近指令序列）。两个后端通过 `MachineCodeEmitter` trait 统一接口。

**外部依赖**：
- `iced-x86` — x86_64: Phase 0 反汇编（Decoder）+ Phase 3 代码生成（CodeAssembler）
- `dynasm-rs` / `dynasmrt` — aarch64: Phase 3 代码生成（Assembler）
- 编译流水线通过 `PlatformBackend` trait 统一访问，不直接依赖底层库

### LayerIR（中间表示）

描述一个 transformer layer 的计算图：
- `Decoder { hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, norm_type, activation, rope_config }`
- `Encoder { ... }`
- `MoE { num_experts, top_k, ... }`（未来）

### ExecutionPlan（执行计划）

融合决策 + 资源分配：
- `fusion_groups: Vec<FusionGroup>` — 已绑定 ISA 的算子序列
- 每个 `FusionGroup` 包含语义描述（FusionDecision）+ 具体算子序列（`Vec<ResolvedOp>`）
- `ResolvedOp` = 已确定 ISA 的算子 + 参数
- Tiling 参数（基于 DeviceProfile 的 cache 大小）
- Scratchpad 大小
- 线程分配策略
- GEMM blocking 参数（KC/MC/NC）

---

## 8. 模型架构支持

| 架构 | 特征 | 状态 |
|------|------|------|
| Llama | RMSNorm + GQA + SwiGLU | 🟡 CPU fallback 基础实现 |
| GPT-2 | LayerNorm + MHA + GELU | 🔴 encoder_forward 未实现 |
| Mistral | Llama + Sliding Window | 🔴 未实现 |
| Phi | GPT + Partial Rotary | 🔴 未实现 |
| Qwen | Llama-like | 🔴 未实现 |
| Gemma | Llama + GeGLU | 🔴 未实现 |
