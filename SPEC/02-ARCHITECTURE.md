# gllm-kernels 架构设计

## 定位

**gllm-kernels = 计算后端**

提供 GPU/CPU 计算能力，gllm 作为客户端调用它完成推理计算。

---

## 当前状态（2026-02）

| 后端 | 状态 | 说明 |
|------|------|------|
| **CPU** | 🔴 待实现 | 高性能 SIMD 后端，编译时自动选择架构 |
| ├─ x86_64 | | AVX2 / AVX-512 / VNNI |
| ├─ ARM | | NEON / dotprod / SVE |
| └─ Apple Silicon | | NEON / AMX |
| **CUDA** | 🔴 待实现 | NVIDIA GPU，L3 GPU-Pure API |
| Metal | 📋 规划中 | Apple GPU |
| ROCm | 📋 规划中 | AMD GPU |

---

## 泛型化核心架构（ARCH-GENERIC-CORE）🚨 铁律

> **设计理念**：gllm-kernels 采用「**一次编写，全精度覆盖**」的泛型架构。所有后端算子和 API 必须对任意满足 trait bounds 的类型工作，禁止为具体类型（f32/f16/i8）单独实现。

### 0.1 Element Trait（ARCH-ELEMENT）

**核心定义**：使用 blanket implementation 自动覆盖所有满足约束的类型。

```rust
/// 所有可用于计算的元素类型必须满足的约束
pub trait Element: Debug + Clone + Copy + Send + Sync + Default + 'static + DeviceRepr {}

/// Blanket Implementation - 自动为所有满足约束的类型实现 Element
impl<T> Element for T
where
    T: Debug + Clone + Copy + Send + Sync + Default + 'static + DeviceRepr,
{}
```

**设计原则**：
- ✅ **自动实现**：任何满足 bounds 的类型自动成为 Element，无需手动列举
- ✅ **零开销抽象**：编译期单态化，运行时无动态分发
- ✅ **类型安全**：编译器保证类型约束

**禁止的做法**：
```rust
// ❌ 错误：手动列举类型实现
impl Element for f32 {}
impl Element for f16 {}
impl Element for i8 {}
// 这违反泛型设计，blanket impl 已自动覆盖
```

### 0.2 Backend Trait 泛型设计（ARCH-BACKEND-GENERIC）

**核心设计**：Backend trait 必须对 Element 类型参数化，一次实现覆盖所有精度。

```rust
/// 后端 trait - 对 Element 类型泛型
pub trait Backend<E: Element> {
    type Tensor: Send + Sync;
    type KvCache: Send + Sync;
    type LogitsHandle: Send + Sync;

    // 权重上传（泛型）
    fn upload_weights(&self, data: &[E]) -> BackendResult<Self::Tensor>;

    // 前向传播（泛型）
    fn forward_batch(
        &self,
        inputs: &[BatchInput],
        weights: &Self::Tensor,
        kv_cache: &mut Self::KvCache,
    ) -> BackendResult<Self::LogitsHandle>;

    // 采样
    fn sample_next_token(
        &self,
        logits: &Self::LogitsHandle,
        config: &SamplingConfig,
    ) -> BackendResult<u32>;
}
```

**正确的实现方式**：
```rust
// ✅ 正确：泛型实现，一次覆盖所有精度
impl<E: Element> Backend<E> for CpuBackend {
    type Tensor = Vec<E>;
    type KvCache = CpuKvCache<E>;
    type LogitsHandle = Vec<E>;

    fn upload_weights(&self, data: &[E]) -> BackendResult<Self::Tensor> {
        Ok(data.to_vec())
    }
    // ...
}

impl<E: Element> Backend<E> for CudaBackend {
    type Tensor = GpuBuffer<E>;
    type KvCache = GpuKvCache<E>;
    type LogitsHandle = GpuLogits<E>;
    // ...
}
```

**禁止的实现方式**：
```rust
// ❌ 错误：为每个精度分别实现（代码重复，违反泛型本质）
impl Backend<f32> for CpuBackend { ... }
impl Backend<f16> for CpuBackend { ... }
impl Backend<i8> for CpuBackend { ... }

// ❌ 错误：使用枚举 + match 分发
fn forward(&self, dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 => self.forward_f32(),
        DType::F16 => self.forward_f16(),
    }
}
```

### 0.3 CPU 内核架构（ARCH-CPU-KERNELS）

> **权威设计**：见 `03-DATA-STRUCTURE.md` 三层树状分发架构

**核心原则**：三层零成本分发

```
Layer 1: Backend    → 用户指定（CpuBackend / CudaBackend）
Layer 2: ISA        → 启动时一次检测（Scalar / AVX2 / AVX-512 / NEON）
Layer 3: Precision  → 编译时泛型单态化（<E: Element>）
```

**ISA 检测只在程序启动时发生一次**，之后整棵算子树都是静态确定的。

### 0.4 泛型化禁止清单（ARCH-GENERIC-FORBIDDEN）

| 禁止行为 | 原因 | 正确做法 |
|----------|------|----------|
| `impl Backend<f32>` 单独实现 | 代码重复 | `impl<E: Element> Backend<E>` |
| `fn gemv_int8_f32()` 每精度单独函数 | 硬编码类型 | `fn gemv_q8<E: Element>()` |
| `TypeId::of::<E>()` 运行时类型检测 | 运行时开销 | 三层树状静态分发 |
| `dyn Kernels` 动态分发 | vtable 开销 | 静态泛型 |
| 手动 `impl Element for T` | 违反 blanket impl | 自动推导 |

---

## 核心原则（🚨 铁律）

### 1. L3 GPU-Pure 架构（ARCH-GPU-PURE）

**正确的生成循环数据流**：

```
模型加载（一次）：
┌─────────────────────────────────────────────────────────────┐
│  CPU                           GPU                          │
│  weights ──upload_weights()──► GPU 常驻权重                 │
│           ──alloc_kv_cache()─► GPU 常驻 KV Cache            │
└─────────────────────────────────────────────────────────────┘

生成循环（每 token）：
┌─────────────────────────────────────────────────────────────┐
│  CPU                           GPU                          │
│  token_id (4B) ──────────────► forward_gpu_pure()           │
│                                 ├── embedding lookup        │
│                                 ├── attention (融合)        │
│                                 ├── ffn (融合)              │
│                                 ├── lm_head                 │
│                                 └── LogitsTensor (GPU 常驻) │
│                                          │                  │
│                                 sample_from_tensor()        │
│                                 └── GPU argmax/topk         │
│  token_id (4B) ◄─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

数据传输：每 token 仅 8 bytes（上传 4B + 下载 4B）
```

**禁止的错误模式**：

```
❌ L2 API 用于生成循环：
   每次 attention_block() 调用都上传权重 → 灾难性性能

❌ Logits 下载到 CPU：
   每 token 下载 128KB+ logits → PCIe 瓶颈

❌ 中途 GPU↔CPU 往返：
   GPU → readback → CPU 计算 → upload → GPU
```

### 2. API 层级定义（ARCH-API-LEVELS）

| 层级 | API | 用途 | 数据位置 |
|------|-----|------|----------|
| **L3 GPU-Pure** | `*_forward_gpu_pure()` | 生产推理 | 权重/KV/Logits 全 GPU |
| L3 CPU | `*_forward()` | CPU 推理 | 全 CPU |
| L2 Block | `attention_block()` 等 | 调试/测试 | 每次调用传输 |
| L1 Atomic | `ops::*` | 内部实现 | 不暴露 |

**生成循环必须使用 L3 GPU-Pure API**

### 3. 算子融合（ARCH-FUSED-KERNELS）

**必须使用融合内核，禁止独立算子串联**：

| 融合内核 | 替代的独立算子 | 内核名称 |
|----------|----------------|----------|
| `fused_qkv_rope` | Q_proj + K_proj + V_proj + RoPE | `fused_qkv_attention` |
| `fused_gate_up_silu` | Gate_proj + Up_proj + SiLU + Mul | `linear` |
| `flash_attention` | Q×K + Softmax + ×V | `flash_attention` |

### 4. 静态工作空间（ARCH-STATIC-WORKSPACE）

```rust
// 禁止生成循环中 cudaMalloc
struct LayerWorkspace {
    q_buf: CudaSlice<f32>,    // 预分配
    k_buf: CudaSlice<f32>,    // 预分配
    v_buf: CudaSlice<f32>,    // 预分配
    attn_out: CudaSlice<f32>, // 预分配
    ffn_buf: CudaSlice<f32>,  // 预分配
}
```

### 5. AOT Only (ARCH-AOT-CUBIN)

为了实现极致启动速度和降低用户驱动环境依赖，我们**放弃 PTX/JIT**，全面采用 **Ahead-of-Time (AOT)** 编译策略。

- **机制**:
  - 针对特定 GPU 微架构离线编译机器码 (`.cubin` / `.hsaco`)。
  - Rust 运行时检测设备架构 (e.g., SM 8.9)，加载对应的二进制。
- **构建流**:
  - 开发者: `compile_kernels.sh` -> 生成 `kernels_sm80.cubin`, `kernels_sm89.cubin`...
  - 编译器: `include_bytes!("kernels/kernels_sm80.cubin")` 嵌入 Rust 二进制。
  - 用户: 无需 CUDA Toolkit，无需 JIT，直接执行。
- **支持架构 (Allowlist)**:
  - `sm_80` (Ampere: A100, RTX 3090)
  - `sm_86` (Ampere: RTX 3060/3070/3080)
  - `sm_89` (Ada Lovelace: RTX 4090, L40)
  - `sm_90` (Hopper: H100)
  - *注：不支持 Pascal/Volta 等旧架构，以减少维护成本。*

### 6. Driver API Only (ARCH-DRIVER-ONLY)

为了确保全平台 "Pure Rust" 编译体验（无外部 SDK 依赖），所有后端必须直接对接系统级驱动接口：

| 后端 | 目标库 (System Driver) | 绑定策略 | 禁止项 |
|------|------------------------|----------|--------|
| **CUDA** | `libcuda.so` / `nvcuda.dll` | 使用 `cudarc` 动态加载符号 | ❌ `libcudart.so` (Runtime) |
| **Metal** | `Metal.framework` | 使用 `metal-rs` (ObjC Runtime 桥接) | ❌ C++ Metal Wrapper |
| **ROCm** | `libhsa-runtime64.so` | HSA Runtime FFI (底层驱动接口) | ❌ `libhip.so` (HIP Runtime) |
| **Intel** | `libze_loader.so` | Level Zero FFI (oneAPI 底层接口) | ❌ OpenCL / SYCL Runtime |

**原则**: 编译产物必须是独立二进制，仅在运行时动态链接系统驱动。

### 7. 统一内存架构优化 (ARCH-UMA)

针对 **Apple Silicon**, **NVIDIA Grace-Hopper**, **Intel Integrated/Arc** 等统一内存架构，系统必须实现**物理级零拷贝**：

- **机制**:
  - 检测到 UMA 设备时，`alloc_weights` 不分配显存，而是分配 **Shared Memory (USM)**。
  - `upload` 操作退化为 `No-Op` 或简单的指针传递。
  - CPU 和 GPU 共享同一块物理内存页，彻底消除 PCIe 传输开销。
- **收益**: 模型加载速度提升 10x-100x (受限于内存带宽而非 PCIe)。

### 8. CPU SIMD 架构（ARCH-CPU-SIMD）🚨 已更新

> **核心原则**：CPU 后端采用 **自研泛型 + SIMD 特化** 策略，**禁止任何外部 BLAS 依赖**。

**禁止的依赖**：
- ❌ `faer` - Pure Rust BLAS
- ❌ `OpenBLAS` - C BLAS
- ❌ `MKL` - Intel 数学库
- ❌ `Accelerate` - Apple 数学库

**自研内核架构**：

1. **泛型接口层**：
   ```rust
   pub fn matmul<E: Element>(a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize)
   ```

2. **SIMD 特化层** (f32/f64)：
   - 运行时 ISA 检测 (AVX2 vs AVX-512 vs NEON)
   - Cache-aware 分块 (Tiling)
   - 寄存器阻塞 (Register Blocking)

3. **标量回退层** (其他类型)：
   - 泛型标量实现
   - 保证正确性

**分块常量**：
```rust
const TILE_M: usize = 64;   // L1 cache 友好
const TILE_N: usize = 64;
const TILE_K: usize = 256;  // L2 cache 友好
```

**量化计算**：
- **手写 SIMD Micro-kernels**
- 架构：`Packed<u8>` -> SIMD Load -> Unpack to f32 registers -> FMA
- On-the-fly dequantization (不生成完整 f32 矩阵)

### 8. 量化架构（ARCH-QUANT）

> **📌 详细设计**: [SPEC/DOCS/quantization/generic-quantization-kernels.md](./DOCS/quantization/generic-quantization-kernels.md)

采用 **Block-wise Quantization** 与 **Bit-Packing** 结合：

#### A. Rust 泛型架构（ARCH-QUANT-GENERIC）

**核心设计原则**：使用 Rust 泛型系统 emulate CUDA C++ 模板，实现零运行时开销的类型抽象。

| 层级 | CUDA C++ | Rust |
|------|----------|------|
| **类型抽象** | `template<int BITS>` | `trait QuantizedMatMul<T: DTypeTrait>` |
| **实例化** | 编译时模板展开 | 编译时单态化 |
| **性能** | 零运行时开销 | 零运行时开销 |

**核心 Trait 定义**：
```rust
pub trait DTypeTrait: Sized + Copy + 'static {
    type Storage: Copy;
    fn dequantize(scaled: Self::Storage, scale: f16) -> f32;
    const BITS: u8;
    const IS_PACKED: bool;
}

pub trait QuantizedMatMul<T: DTypeTrait> {
    fn matmul(
        input: &[f32],
        weight: &[T::Storage],
        scales: &[f16],
        bias: Option<&[f32]>,
        output: &mut [f32],
        m: usize, n: usize, k: usize,
    ) -> Result<(), BackendError>;
}
```

**支持的类型映射**：
| DType | Storage | Bits | Trait 实现 |
|-------|---------|------|------------|
| F32 | f32 | 32 | `QuantizedMatMul<F32>` → **自研 SIMD 内核** |
| F16 | f16 | 16 | `QuantizedMatMul<F16>` → 半精度 SIMD |
| BF16 | bf16 | 16 | `QuantizedMatMul<BF16>` → bfloat SIMD |
| I8 | i8 | 8 | `QuantizedMatMul<I8>` → int8 SIMD 反量化 |
| PackedI4 | u8 | 4 | `QuantizedMatMul<PackedI4>` → 解包 + FMA |
| PackedI2 | u8 | 2 | `QuantizedMatMul<PackedI2>` → 解包 + FMA |
| PackedI1 | u8 | 1 | `QuantizedMatMul<PackedI1>` → 解包 + FMA |

**禁止行为**：
- ❌ 为每个量化位宽实现独立的函数 (如 `matmul_int8`, `matmul_int4` 等)
- ❌ 使用 `dyn Trait` 动态分发
- ❌ 运行时 `match dtype` 分派

#### B. QKV 投影统一 API（ARCH-QKV-GENERIC）

**自动路径选择**：
```rust
pub enum QkvWeightFormat<'a, T: DTypeTrait> {
    Separated { q_weight, k_weight, v_weight, scales },  // 最优: 3×小矩阵
    Fused { qkv_weight, scales },                        // 回退: 1×大矩阵
}

pub fn qkv_projection_rope_generic<T: DTypeTrait>(
    input: &[f32],
    qkv_weights: &QkvWeightFormat<T>,
    // ...
) -> Result<()>
where
    CpuBackend: QuantizedMatMul<T>;
```

**性能约束**：
- 分离权重路径必须使用 3×独立 `linear_generic<T>` 调用
- 每次调用都是小的矩阵乘法，L1/L2 Cache 友好
- 禁止使用单次大矩阵乘法然后分割结果

#### C. 统一存储抽象 (QuantizedStorage Trait)
为了统一 CPU 和 GPU 的量化行为，定义底层存储契约：

```rust
// 核心抽象：屏蔽 Device 差异
pub trait QuantizedStorage {
    type PackedData; // e.g., Vec<u8> or CudaSlice<u8>

    // 获取原始 Packed 数据（不反量化）
    fn as_packed(&self) -> &Self::PackedData;

    // 获取量化元数据
    fn scales(&self) -> &[f16];
}
```

- **存储层 (Storage)**:
  - 不引入新类型，统一使用 `u8` 容器。
  - **Int4**: 2 elements per `u8`.
  - **Int2**: 4 elements per `u8`.
  - **Block**: 每 block (e.g. 128 params) 共享一个 `f16` scale。

- **计算层 (Compute)**:
  - **On-the-fly Dequantization**: 永不在内存中还原完整 f32 矩阵。
  - 核心循环：
    ```rust
    // 伪代码
    let packed = load_u128(ptr);
    let floats = simd_unpack_int4_to_f32(packed);
    acc = simd_fma(floats, input, acc);
    ```

### 9. CUDA Graphs 加速 (ARCH-GPU-GRAPH)

为了彻底消除 CPU 发射开销 (Launch Overhead)，生成循环必须支持 **CUDA Graph Capture**：

- **机制**:
  - `start_capture()` -> 运行一次完整的 forward 流程 -> `end_capture()`。
  - 后续生成步骤只需调用 `graph.launch()`。
- **约束**:
  - 图内部的显存指针必须是固定的 (Static Workspace)。
  - `cudarc` 提供了完善的 Graph 支持，直接利用。
- **收益**: 小 Batch 推理延迟降低 30%-50%。

---

### 10. Tree Attention (ARCH-TREE-ATTN)

为了即时支持 EAGLE-2 / Medusa-2 等推测解码算法，L3 API 必须原生支持非线性拓扑：

- **Token 序列**: 不再是一维数组，而是 `(token_id, parent_index, position_id)` 的结构。
- **Attention Mask**: 内核必须支持基于拓扑生成的 2D Mask，而不仅仅是 Causal Mask。
- **KV Cache**: 写入时需根据 `parent_index` 进行 Scatter 写，而不是简单的 Append。

### 11. 后端自动探测 (ARCH-AUTO-DETECT)

为了实现"零配置启动"，系统必须在运行时动态探测可用硬件，按优先级自动选择最佳后端：

**探测优先级 (Priority Strategy)**:

1.  **显式覆盖 (Override)**:
    - 检查环境变量 `GLLM_DEVICE` (e.g. `cuda:0`, `cpu`).
    - 如果设定，强制使用指定后端，失败则报错。

2.  **CUDA (P0 - NVIDIA)**:
    - 尝试动态加载 `libcuda.so` / `nvcuda.dll`。
    - 调用 `cuInit(0)` 成功且检测到设备 -> **Selected**。

3.  **Metal (P1 - Apple Silicon)**:
    - (MacOS Only) 检查 `Metal.framework` 可用性。
    - 检测到 Apple GPU -> **Selected**。

4.  **ROCm (P2 - AMD)**:
    - 尝试动态加载 `libhsa-runtime64.so`。
    - 检测到 HSA Agent -> **Selected**。

5.  **CPU (Fallback)**:
    - 如果以上均失败，回退到 **CPU Backend** (自研泛型 SIMD 内核)。
    - 即使有 GPU，CPU 后端也必须始终可用（作为参考实现）。

### 12. 软件工程抽象模式 (ARCH-SOFTWARE-PATTERNS)

为了管理多后端、多精度、多模型的复杂性，系统采用以下高阶软件设计模式，严禁过程式“面条代码”。

#### A. 类型驱动后端 (Type-Driven Backend)
利用 Rust 强大的类型系统，将后端差异编码在类型签名中，而非运行时的 `match`。

```rust
// ❌ 禁止：运行时枚举判断
// fn forward(device: DeviceType) { match device { ... } }

// ✅ 推荐：泛型特化 (Static Dispatch)
trait Backend {
    type Tensor<T>;
    type Graph;
}

struct CudaBackend;
impl Backend for CudaBackend { ... }

// 业务逻辑对具体后端无感知
fn forward<B: Backend>(engine: &Engine<B>) { ... }
```

#### B. 算子构建者模式 (Operator Builder)
为了支持 CUDA Graphs 和算子融合，计算逻辑与执行逻辑分离。

```rust
// 1. 录制阶段 (不执行)
let mut graph = builder.begin_graph();
let c = graph.matmul(a, b);
let d = graph.silu(c);

// 2. 编译阶段 (后端优化)
let exec_plan = graph.compile(); // CUDA: Graph Capture, CPU: Loop Fusion

// 3. 执行阶段
exec_plan.launch();
```

#### C. 资源句柄化 (Opaque Handles)
所有设备侧资源（显存、图）必须封装为不透明句柄 (NewType Pattern)，物理阻断错误访问。

```rust
pub struct LogitsHandle(usize); // 仅持有 ID
pub struct KvCacheHandle(usize);

// Client 无法解引用 Handle 获取数据，只能传回 Backend
```

---

## L3 GPU-Pure API 定义

### Generator（文本生成）

```rust
// 拓扑描述 (用于 Tree Attention)
pub struct AttentionTopology {
    // 如果是 None，则是标准线性解码 (Causal)
    // 如果是 Some，则包含每个 token 的父节点索引和位置偏移
    pub tree_structure: Option<CudaSlice<i32>>,
}

// 一次上传
fn upload_generator_weights(...) -> Result<GeneratorModelWeightsGpu, String>;

fn alloc_kv_cache_gpu(...) -> Result<KVCacheGpu, String>;

// 零拷贝 forward (支持 Tree Attention)
fn generator_forward_gpu_pure(
    tokens: &[u32],
    topology: &AttentionTopology, // 新增：支持非线性解码
    weights: &GeneratorModelWeightsGpu,
    kv_cache: &mut KVCacheGpu,
    config: &GeneratorForwardConfig,
) -> Result<LogitsTensor, String>;

// GPU 采样 (支持 Tree)
fn sample_from_tensor(
    logits: &LogitsTensor,
    topology: &AttentionTopology, // 新增：采样可能需要树结构信息
    vocab_size: usize,
    config: &SamplingConfig,
) -> Result<Vec<u32>, String>;
```

### Embedding（文本向量化）

```rust
fn upload_embedding_weights(...) -> Result<EmbeddingModelWeightsGpu, String>;
fn embedding_forward_gpu_pure(...) -> Result<Vec<f32>, String>;
```

### Rerank（文本重排序）

```rust
fn upload_reranker_weights(...) -> Result<RerankerModelWeightsGpu, String>;
fn rerank_forward_gpu_pure(...) -> Result<Vec<f32>, String>;
```

---

## 目录结构

```
src/
├── backend.rs          # 入口 + auto_select_backend()
├── backend_trait.rs    # Backend trait 定义
├── cpu_backend.rs      # CPU 实现（参考 + fallback）
├── cuda_backend.rs     # CUDA 实现（唯一优先）
├── cuda_kernels/       # CUDA CUBIN + Driver API
│   ├── kernels/        # 预编译 CUBIN 文件 (sm_80/86/89/90)
│   ├── flash_attn.rs   # FlashAttention
│   ├── linear.rs       # 融合 Linear
│   ├── sampling/       # GPU 采样
│   └── ...
├── kernel_types.rs     # 配置类型
├── gpu_types.rs        # GPU 张量类型
└── ops/                # CPU 参考实现
```

---

## 未来计划

| 后端 | 优先级 | 依赖 |
|------|--------|------|
| ROCm | P2 | CUDA 完成后 |
| Metal | P3 | ROCm 完成后 |

**当前专注 CUDA，确保 L3 GPU-Pure API 100% 正确实现。**

---

## 后端调度架构 (ARCH-SCHED-BACKEND)

> **关联文档**: [gllm SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md](../gllm/SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md)

### Backend Trait 调度接口

调度器需要 Backend 提供以下接口：

| 接口 | 功能 | 约束 |
|------|------|------|
| `swap_out_pages()` | GPU → CPU 页面搬运 | 必须保证数据完整性 |
| `swap_in_pages()` | CPU → GPU 页面搬运 | 必须在 Warm-up 期后调用 |
| `get_memory_pressure()` | 获取 GPU 内存使用率 | 返回 0.0-1.0 精度 |
| `get_page_states()` | 获取页面状态快照 | 返回 (page_id, PageState) 列表 |
| `batch_forward_gpu_pure()` | 批处理前向传播 | 支持多序列、独立 Logits |

### Swap 约束 (ARCH-SCHED-SWAP)

| 约束 | 说明 | 违规后果 |
|------|------|----------|
| **禁止生成循环中 Swap** | Swap 只能在批次间执行 | 违反 ARCH-GPU-001 零拷贝原则 |
| **禁止热页 Swap** | Protected/Warm 状态页面禁止 Swap | 导致 Cache Thrashing |
| **异步 Swap** | Swap 操作必须在后台流执行 | 避免阻塞生成循环 |

### KV Cache 约束 (ARCH-SCHED-KVCACHE)

| 约束 | 说明 |
|------|------|
| **页面大小对齐** | 所有分配必须按 page_size 对齐 |
| **预分配策略** | KV Cache 必须预分配，避免生成循环分配 |
| **双缓冲支持** | 支持 Front/Back 双缓冲调度 |
