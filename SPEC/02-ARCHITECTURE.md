# gllm-kernels 架构设计

## 概述

gllm-kernels 采用**运行时后端选择 + 零成本抽象 + Fat Binary**架构：
- **Fat Binary**: 编译时嵌入所有后端的预编译中间态
- **运行时检测**: 启动时检测可用后端，无 feature flag
- **零成本派发**: enum + match 实现零 vtable 开销
- **Driver API Only**: 只依赖 GPU 驱动，无需开发工具包

### 🚨 核心设计原则（铁律）

| 原则 | 说明 | 理由 |
|------|------|------|
| **只做应用级算子** | 只实现 FlashAttention、PagedAttention、RMSNorm 等 LLM 专用算子 | 底层通用算子（matmul、softmax）由硬件库（cuBLAS、MPS）更高效 |
| **不做底层通用算子** | ❌ 禁止实现通用 matmul、gemm、conv2d 等 | 自己实现性能必然差于硬件厂商优化 |
| **CPU 是兜底后端** | CPU 实现仅作 fallback，不是性能目标 | GPU 才是主战场，CPU 只保证可用性 |
| **全流程单后端** | GPU 任务全程 GPU，CPU 任务全程 CPU，禁止混合 | GPU↔CPU 数据传输开销极大，会抵消计算加速 |

**全流程单后端原则详解**：

```
✅ 正确：全流程 GPU
   upload → flash_attn(GPU) → rms_norm(GPU) → linear(GPU) → readback

✅ 正确：全流程 CPU（无 GPU 时）
   flash_attn(CPU) → rms_norm(CPU) → linear(CPU)

❌ 错误：混合执行
   flash_attn(GPU) → readback → rms_norm(CPU) → upload → linear(GPU)
   ↑ 每次 readback/upload 都是巨大开销！
```

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        gllm-kernels                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Public API Layer                                                    │
│  ├── KernelDispatcher         → 零成本算子派发                       │
│  ├── detect_backend()         → 运行时检测（CUDA>ROCm>Metal>WGPU>CPU）│
│  └── BackendType enum         → 后端类型枚举                         │
├─────────────────────────────────────────────────────────────────────┤
│  Application-Level Operators（只做 LLM 专用算子）                     │
│  ├── flash_attention          → Flash Attention v2/v3               │
│  ├── paged_attention          → PagedKV Attention (vLLM)            │
│  ├── flash_tree_attention     → Tree-structured Attention           │
│  ├── moe_forward              → MoE 前向计算                         │
│  ├── rms_norm / layer_norm    → 归一化层                             │
│  ├── rope                     → 旋转位置编码                         │
│  ├── softmax                  → Log-Space Softmax（数值稳定）        │
│  └── sampling                 → Top-K/Top-P 采样                     │
├─────────────────────────────────────────────────────────────────────┤
│  GPU Backend Kernels（主战场）                                        │
│  ├── cuda_kernels/            → PTX + CUDA Driver API                │
│  ├── hip_kernels/             → HSACO + HSA Runtime (Driver API)     │
│  ├── metal_kernels/           → metallib + Metal Framework           │
│  └── wgpu_kernels/            → WGSL + wgpu                          │
├─────────────────────────────────────────────────────────────────────┤
│  CPU Fallback（兜底实现，非性能目标）                                 │
│  └── ops/                     → 纯 Rust 参考实现（无 GPU 时使用）     │
├─────────────────────────────────────────────────────────────────────┤
│  Runtime Detection Layer                                             │
│  ├── runtime_detection.rs     → 后端可用性检测                       │
│  └── kernel_dispatcher.rs     → 统一派发入口                         │
└─────────────────────────────────────────────────────────────────────┘
```

> ⚠️ **已废弃/未实现的算子**：ring_attention、mla、mamba（性能不满足要求或无需求）

---

## 核心架构决策

### ARCH-BACKEND-001: 简化后端架构（🚨 铁律 - 2026-01）

**核心理念**：Backend = 硬件算法工具库，就这么简单。

**设计原则**：
- 启动时选择一次后端，之后直接使用
- 每个后端是独立的 `Backend` trait 实现
- 热路径内无任何间接调用
- `dyn Trait` 只允许在启动时（一次 vtable 查找可忽略）

#### 正确架构

```rust
// 1. Backend trait - 十几个核心应用级算子
pub trait Backend: Send + Sync {
    fn flash_attention(&self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
                       output: &mut GpuTensor, config: FlashAttentionConfig) -> Result<(), String>;
    fn paged_attention(&self, ...) -> Result<(), String>;
    fn moe_forward(&self, ...) -> Result<(), String>;
    fn rms_norm(&self, ...) -> Result<(), String>;
    fn linear_forward(&self, ...) -> Result<(), String>;
    fn upload<T>(&self, host: &[T], gpu: &mut GpuTensor) -> Result<(), String>;
    fn readback<T>(&self, gpu: &GpuTensor, host: &mut [T]) -> Result<(), String>;
    // ~15 个核心算子，不需要更多
}

// 2. 每个后端各自实现 - 独立文件
// wgpu_backend.rs
impl Backend for WgpuBackend { ... }

// cuda_backend.rs
impl Backend for CudaBackend { ... }

// cpu_backend.rs
impl Backend for CpuBackend { ... }

// 3. 启动时选一次，直接用
pub fn auto_select_backend() -> Arc<dyn Backend> {
    if cuda_available() { return Arc::new(CudaBackend::new()); }
    if rocm_available() { return Arc::new(RocmBackend::new()); }
    if metal_available() { return Arc::new(MetalBackend::new()); }
    if wgpu_available() { return Arc::new(WgpuBackend::new()); }
    Arc::new(CpuBackend::new())
}

// 4. 使用 - 一次动态分发，完事
let backend = auto_select_backend();
backend.flash_attention(...);  // 直接调用，没有中间层
```

#### 实现状态（2026-01-26 更新）

**阶段1 已完成** ✅：外层架构重组

| 目标 | 重构前 | 重构后 | 状态 |
|------|--------|--------|------|
| 派发机制 | `DispatchedBackend` enum | `Arc<dyn Backend>` | ✅ |
| backend.rs 行数 | 6,270 行 | 20 行 | ✅ |
| 后端文件 | 全在 backend.rs | 独立文件 | ✅ |
| `auto_select_backend()` | 无 | 返回 `Arc<dyn Backend>` | ✅ |
| Backend trait 方法数 | 52 个 | 15 个 | ✅ |

**阶段2 待完成** ⚠️：消除 KernelDispatcher 中间层

**当前问题**：
```
Backend trait
    ↓ 委托
BackendCore
    ↓ 委托
KernelDispatcher（7,646 行）
    ↓ match self.backend 分发（每个算子内部）
cuda_kernels / wgpu_kernels / metal_kernels / ops
```

KernelDispatcher 内部仍有大量 `match self.backend` 分发，这违反了"启动时选一次"原则。

**正确架构**（阶段2目标）：
```
Backend trait
    ↓ 直接实现
CpuBackend   → 直接调用 ops/
CudaBackend  → 直接调用 cuda_kernels/
WgpuBackend  → 直接调用 wgpu_kernels/
MetalBackend → 直接调用 metal_kernels/
RocmBackend  → 直接调用 hip_kernels/
```

**阶段2 任务清单**：
- [ ] 提取 KernelDispatcher 中的 CPU 实现到 `CpuBackend`
- [ ] 提取 CUDA 分发逻辑到 `CudaBackend`
- [ ] 提取 WGPU 分发逻辑到 `WgpuBackend`
- [ ] 提取 Metal 分发逻辑到 `MetalBackend`
- [ ] 提取 ROCm 分发逻辑到 `RocmBackend`
- [ ] 删除 KernelDispatcher 或将其降级为纯测试工具
- [ ] 删除 BackendCore 中间层

#### 算子实现状态矩阵（KernelDispatcher 内部）

**说明**：以下矩阵记录 KernelDispatcher 内部各算子在不同后端的实现状态。阶段2完成后，这些实现将迁移到各独立 Backend 文件中。

##### Backend trait 核心算子（15个）

| 算子 | CPU | CUDA | Metal | WGPU | ROCm | 说明 |
|------|-----|------|-------|------|------|------|
| flash_attention | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | FlashAttention-2 |
| paged_attention | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | PagedAttention |
| softmax | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 独立 softmax |
| matmul | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 通用矩阵乘 |
| rope_precompute | ✅ | ✅ | ✅ | ✅ | ✅ | RoPE 预计算 |
| rope_apply | ✅ | ✅ | ✅ | ✅ | ✅ | RoPE 应用 |
| rope_apply_inplace | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | RoPE 原地应用 |
| topk | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | Top-K 采样 |
| apply_temperature | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | 温度缩放 |
| sample_tokens | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | Token 采样 |
| argmax | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | 贪婪解码 |
| moe_route | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | MoE 路由 |
| compute_routing_logits | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | 路由 logits |
| add_bias | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | 偏置加法 |
| backend_type | ✅ | ✅ | ✅ | ✅ | ✅ | 类型查询 |

##### KernelDispatcher 其他算子

| 算子 | CPU | CUDA | Metal | WGPU | ROCm | 说明 |
|------|-----|------|-------|------|------|------|
| rms_norm | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | RMS 归一化 |
| layer_norm | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | Layer 归一化 |
| silu | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | SiLU 激活 |
| gelu | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | GELU 激活 |
| embedding | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 词嵌入查表 |
| causal_mask | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 因果掩码 |
| quantize | ✅ | ✅ | ⚠️ partial | ⚠️ partial | ❌ | 量化操作 |
| dequantize | ✅ | ✅ | ⚠️ partial | ⚠️ partial | ❌ | 反量化操作 |
| fused_attention | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 融合注意力 |
| moe_forward | ✅ | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | MoE 前向 |
| flash_tree_attention | ✅ | ⚠️ partial | ❌ | ❌ | ❌ | 树结构注意力 |
| concat | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 张量拼接 |
| split | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 张量分割 |
| transpose | ✅ | ✅ | ✅ | ✅ | ⚠️ fallback | 转置 |
| copy | ✅ | ✅ | ✅ | ✅ | ✅ | 内存拷贝 |

##### 实现覆盖率统计

| 后端 | 完整实现 | CPU Fallback | 部分实现 | 未实现 | 覆盖率 |
|------|---------|--------------|----------|--------|--------|
| CPU | 30/30 | - | - | 0 | **100%** |
| CUDA | 18/30 | 10 | 2 | 0 | **60%** (含fallback: 100%) |
| Metal | 16/30 | 10 | 2 | 2 | **53%** (含fallback: 93%) |
| WGPU | 16/30 | 10 | 2 | 2 | **53%** (含fallback: 93%) |
| ROCm | 2/30 | 26 | 0 | 2 | **7%** (含fallback: 93%) |

**图例**：
- ✅ = 原生 GPU/CPU 实现
- ⚠️ CPU = 回退到 CPU 实现
- ⚠️ fallback = 回退到其他后端
- ⚠️ partial = 部分实现
- ❌ = 未实现

**阶段2优先级**（基于覆盖率）：
1. CPU（100%）→ 直接迁移到 CpuBackend
2. CUDA（60%）→ 迁移已有实现，保持 CPU fallback
3. Metal/WGPU（53%）→ 迁移已有实现
4. ROCm（7%）→ 迁移 RoPE，其他保持 fallback

#### 关于 `dyn Trait`

| 场景 | 是否允许 | 原因 |
|------|----------|------|
| 启动时选一次后端 | ✅ 允许 | 一次 vtable 查找，忽略不计 |
| 热路径内部（每次 matmul） | ❌ 禁止 | vtable = 运行时开销 |

#### 当前文件结构（阶段1完成后）

```
src/
├── backend.rs              # 入口模块 + auto_select_backend()（20 行）
├── backend_trait.rs        # Backend trait 定义（15 个方法）
├── backend_core.rs         # ⚠️ 中间层，待删除
├── backend_core_*.rs       # ⚠️ 中间层，待删除
├── cpu_backend.rs          # CpuBackend（委托到 BackendCore）
├── cuda_backend.rs         # CudaBackend（委托到 BackendCore）
├── wgpu_backend.rs         # WgpuBackend（委托到 BackendCore）
├── metal_backend.rs        # MetalBackend（委托到 BackendCore）
├── rocm_backend.rs         # RocmBackend（委托到 BackendCore）
│
├── kernel_dispatcher.rs    # ⚠️ 7,646 行，包含实际分发逻辑，待拆解
│
├── cuda_kernels/           # CUDA GPU kernel 实现
├── hip_kernels/            # ROCm GPU kernel 实现
├── metal_kernels/          # Metal GPU kernel 实现
├── wgpu_kernels/           # WGPU GPU kernel 实现
│
└── ops/                    # CPU 实现（纯 Rust + SIMD）
```

---

### ARCH-SCOPE-001: 算子实现范围（🚨 铁律）

**我们只实现 LLM 推理专用的应用级算子，不实现底层通用算子。**

#### ✅ 实现的算子（应用级，LLM 专用）

| 算子 | 用途 | 说明 |
|------|------|------|
| flash_attention | Transformer 注意力 | FlashAttention-2/3 算法 |
| paged_attention | KV Cache 分页 | vLLM PagedAttention |
| flash_tree_attention | 树结构注意力 | 推测解码用 |
| moe_forward | MoE 专家前向 | Mixtral/DeepSeek 等 |
| rms_norm | RMS 归一化 | LLaMA 等模型 |
| layer_norm | Layer 归一化 | GPT/BERT 等模型 |
| rope | 旋转位置编码 | 大多数现代 LLM |
| silu/gelu | 激活函数 | LLM 常用激活 |
| sampling | Top-K/Top-P 采样 | 文本生成 |

#### ❌ 不实现的算子（底层通用）

| 算子 | 原因 |
|------|------|
| matmul / gemm | cuBLAS / MPS / BLAS 已极致优化，自己写必然更慢 |
| conv2d / pooling | 图像算子，与 LLM 无关 |
| batch_norm | CV 用，LLM 用 RMSNorm/LayerNorm |
| 通用 softmax | 已融合到 FlashAttention 内部 |
| transpose / reshape | 零成本视图操作，不需要 kernel |

#### 设计理由

```
❌ 错误思路：
"我们来实现一个高性能 matmul kernel"
→ NVIDIA 花了 20 年优化 cuBLAS，你不可能超越

✅ 正确思路：
"我们实现 FlashAttention，内部调用 cuBLAS matmul"
→ 算法创新 + 复用硬件库 = 最佳性能
```

#### 废弃的算子

| 算子 | 状态 | 原因 |
|------|------|------|
| ring_attention | ❌ 废弃 | 分布式场景未验证，暂不需要 |
| mla | ❌ 废弃 | 与 flash_attention 合并处理 |
| mamba | ❌ 废弃 | SSM 模型支持优先级低 |

---

### ARCH-ROUTE-001: 优化路由决策（小模型/小批量）

**核心问题**：GPU 不是万能的，某些场景下 CPU 反而更快。

#### 1. 模型级路由：800M 参数阈值

**实现位置**：`gllm/src/engine.rs:588-614`

```rust
// 自动路由逻辑（Device::Auto 时生效）
if let Ok((config, _)) = ModelConfig::load(repo_name, Some(&config_path)) {
    let params = estimate_model_params(&config);
    // 阈值: 800M 参数。低于此值时，CPU 对小批量更快，且节省 GPU 显存
    if params < 800_000_000 {
        log::debug!("Auto-routing small model ({} params) to CPU", params);
        return false;  // 使用 CPU
    }
}
return true;  // 使用 GPU
```

**参数估算公式**：

```rust
fn estimate_model_params(config: &ModelConfig) -> usize {
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let layers = config.num_hidden_layers;

    // Embedding 参数: vocab × hidden
    let embedding_params = vocab * hidden;
    // Transformer 层参数: layers × 12 × hidden²
    // (4×h² attention + 8×h² MLP 的近似)
    let layer_params = layers * 12 * hidden * hidden;

    embedding_params + layer_params
}
```

**路由决策表**：

| 模型规模 | 参数量 | 路由目标 | 原因 |
|----------|--------|----------|------|
| 小模型 | < 800M | CPU | 计算密度低，PCIe 开销占主导 |
| 中大模型 | ≥ 800M | GPU | 计算量大，GPU 并行优势明显 |

#### 2. Decode Mode 优化：单 Token 生成

**实现位置**：`gllm/src/causal_attention.rs:428-437`

```rust
// Decode 阶段：每次只生成 1 个 token
self.dispatcher.flash_attention(
    q_buf, cached_k, cached_v, attn_out_buf,
    FlashAttentionConfig {
        causal: true,
        num_heads: self.num_attention_heads,
        head_dim: self.head_dim,
        seq_len_q: 1,        // 只有 1 个查询 token
        seq_len_kv: key_len, // KV cache 长度
        batch_size: 1,       // 单批次
        ..Default::default()
    },
);
```

**场景说明**：
- `seq_len_q: 1` = 单 token 查询（自回归生成的每一步）
- `batch_size: 1` = 单请求推理
- 此配置下 GPU kernel launch 开销可能接近实际计算时间

#### 3. 算子级路由：始终 CPU 的操作

**实现位置**：`gllm-kernels/src/kernel_dispatcher.rs:2995-3009`

```rust
pub fn matryoshka_truncate(&self, embeddings: &[f32], output: &mut [f32], config: MatryoshkaConfig) {
    // ALWAYS use CPU - GPU overhead exceeds computation time
    // for this simple memory-bound operation (dimension truncation + optional normalize)
    crate::ops::embedding::matryoshka_truncate(embeddings, output, &config);
}
```

**始终 CPU 的算子**：

| 算子 | 原因 | GPU 开销 |
|------|------|----------|
| `matryoshka_truncate` | 纯内存拷贝 + 可选 L2 归一化 | kernel launch (~5-50μs) + PCIe (~100-400μs) > 计算时间 |

#### 4. CPU SIMD 加速路径

**检测位置**：`ops/simd_asm.rs::current_simd_path()`

| 架构 | SIMD 指令集 | 吞吐量 |
|------|------------|--------|
| x86_64 | AVX-512 | 64 floats/iter |
| x86_64 | AVX2+FMA | 32 floats/iter |
| ARM64 | NEON | 16 floats/iter |
| 通用 | wide crate | 8 floats/iter |

---

### ARCH-KERNELS-001: Kernel 目录结构

> **⚠️ 说明**：以下是基础结构规范，实际实现包含更多 kernel 文件（eagle3、medusa、spec_ee、chunked_prefill、evic_press、int2_quantizer、prompt_cache、flash_tree_attn、moe_ffn、linear、rms_norm 等）

**每个后端的 kernel 模块基本目录结构**：

```
src/{backend}_kernels/
├── mod.rs                      # 模块导出
├── {runtime}.rs                # Runtime 抽象（命名因后端而异）
├── flash_attn.rs               # Flash Attention Kernel
├── paged_attn.rs               # Paged Attention Kernel
├── [其他 kernel].rs            # 更多 kernel 实现...
└── kernels/                    # 预编译中间态
    ├── flash_attention.{ext}   # 嵌入的中间态
    └── [其他 kernel].{ext}     # 更多预编译中间态
```

**中间态格式对照表**：

| 后端 | 中间态扩展名 | 源码扩展名 | 编译工具 | 运行时依赖 |
|------|-------------|-----------|----------|-----------|
| CUDA | `.ptx` | `.cu` | `nvcc -ptx` | `libcuda.so` |
| ROCm | `.hsaco` | `.hip` | `hipcc --genco` | `libhsa-runtime64.so` |
| Metal | `.metallib` | `.metal` | `xcrun metallib` | Metal.framework |
| WGPU | `.wgsl` | N/A | N/A（WGSL 即中间态） | wgpu |
| CPU | N/A | N/A | N/A | 无 |

**说明**：WGSL 是 WebGPU 的标准着色器语言，本身就是中间态格式，无需预编译。通过 `include_str!` 嵌入后由 wgpu 运行时编译为各平台原生格式。

### ARCH-KERNELS-002: Runtime 接口契约

> **⚠️ 说明**：以下是核心接口规范，各后端实际实现可能有差异（如 CUDA 使用 cudarc、ROCm 使用 HSA Runtime）

**每个后端的 `runtime` 模块应提供以下核心功能**：

#### 检测接口

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `is_available()` | bool | 检测后端是否可用（驱动已安装且能正常初始化） |
| `device_count()` | usize | 获取可用设备数量 |

#### Buffer 管理接口

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `Buffer.new` | device: usize, size: usize | Result\<Buffer, Error\> | 在指定设备上分配内存 |
| `Buffer.from_slice` | device: usize, data: slice | Result\<Buffer, Error\> | 从 host 数据创建并复制到 GPU |
| `Buffer.to_vec` | - | Result\<Vec, Error\> | 从 GPU 复制数据回 host |
| `Buffer.len` | - | usize | 获取 buffer 长度（元素数量） |

#### Kernel 加载接口（零配置）

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `KernelLibrary.from_bytes` | data: bytes | Result\<KernelLibrary, Error\> | 从嵌入的字节加载（主要路径） |
| `KernelLibrary.get_function` | name: string | Result\<Function, Error\> | 获取 kernel 函数 |

**🚨 禁止的接口**：
- ❌ `compile_source(source)` - 禁止从源码编译

### ARCH-KERNELS-003: Kernel 接口契约

> **⚠️ 说明**：以下是核心 Attention Kernel 的接口规范示例，实际实现通过 `Backend` trait 统一调度

#### FlashAttentionKernel 接口

| 方法 | 说明 |
|------|------|
| `new(device)` | 创建 Kernel 实例（加载预编译中间态） |
| `forward_f32(...)` | f32 前向计算 |
| `forward_f16(...)` | f16 前向计算 |

**forward 参数表**：

| 参数 | 类型 | 说明 |
|------|------|------|
| q | Buffer | Query: [batch, heads, seq_len, head_dim] |
| k | Buffer | Key: [batch, heads, seq_len, head_dim] |
| v | Buffer | Value: [batch, heads, seq_len, head_dim] |
| batch_size | usize | 批大小 |
| num_heads | usize | 注意力头数 |
| seq_len | usize | 序列长度 |
| head_dim | usize | 头维度 |
| is_causal | bool | 是否因果掩码 |
| scale | f32 | 缩放因子 |

#### PagedAttentionKernel 接口

| 方法 | 说明 |
|------|------|
| `new(device)` | 创建 Kernel 实例 |
| `forward_f32(...)` | f32 前向计算 |
| `forward_f16(...)` | f16 前向计算 |

**forward 参数表**：

| 参数 | 类型 | 说明 |
|------|------|------|
| q | Buffer | Query |
| k_cache | Buffer | Key 缓存 |
| v_cache | Buffer | Value 缓存 |
| block_tables | Buffer\<i32\> | 块索引表 |
| block_offsets | Buffer\<i32\> | 块内偏移 |
| batch_size | usize | 批大小 |
| num_heads | usize | 注意力头数 |
| head_dim | usize | 头维度 |
| block_size | usize | 块大小 |
| seq_len | usize | 序列长度 |

---

## ARCH-LOAD-001: 统一 Kernel 加载策略（🚨 零配置 + Fat Binary + 中间态运行时加载）

**核心设计理念**：
- **完全自动化，零配置**：用户不需要配置任何东西，即插即用，无感知体验
- **Fat Binary**：编译时嵌入所有平台的预编译中间态（PTX/HSACO/metallib/WGSL）
- **中间态运行时加载**：运行时加载预编译的中间态，由 Driver/Runtime 转换为本地代码
- **🚨 禁止源码运行时编译**：不允许从 .cu/.hip/.metal 源码运行时编译
- **❌ 禁止环境变量**：删除所有 `GLLM_*` 环境变量支持

**中间态编译架构**：

```
编译时（CI/开发者机器）                     运行时（用户机器）
========================                   ========================
.cu  ──nvcc -ptx──→ PTX      ─┐
.hip ──hipcc --genco──→ HSACO  ├─ include_bytes! ─→ Fat Binary
.metal ──xcrun metallib──→ metallib            │
.wgsl ────────────────→ WGSL ─┘                ↓
                                           加载中间态
                                               │
                            ┌──────────────────┼──────────────────┐
                            ↓                  ↓                  ↓
                      CUDA Driver         HSA Runtime       Metal/wgpu
                      (PTX→SASS)         (HSACO→执行)    (metallib/WGSL→执行)
```

**统一加载流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 读取嵌入的中间态 | 编译时通过 `include_bytes!` 嵌入 |
| 2 | 检查中间态有效性 | 如果为空，返回错误 |
| 3 | 加载到 Driver/Runtime | Driver 负责 JIT 转换为本地代码 |

**错误处理**：
- 中间态为空 → `KernelNotFound` 错误
- 加载失败 → `KernelLoadFailed` 错误
- 🚨 无回退路径，禁止源码编译

**各后端中间态加载**：

| 后端 | 中间态 | 嵌入方式 | 运行时加载 | 源码编译 |
|------|--------|----------|-----------|----------|
| CUDA | PTX | `include_bytes!` | ✅ CUDA Driver JIT | ❌ 禁止 |
| ROCm | HSACO | `include_bytes!` | ✅ HSA Runtime | ❌ 禁止 |
| Metal | metallib | `include_bytes!` | ✅ Metal Framework | ❌ 禁止 |
| WGPU | WGSL | `include_str!` | ✅ wgpu 转换 | ❌ 禁止 |
| CPU | N/A | N/A | N/A | N/A |

**🚨 铁律约束**：
- ✅ **中间态运行时加载**：PTX/HSACO/metallib/WGSL 在运行时由 Driver/Runtime 加载执行
- ❌ **禁止源码运行时编译**：不允许从 .cu/.hip/.metal 源码运行时编译
- ❌ **禁止环境变量配置**：删除所有 `GLLM_*` 环境变量
- ❌ **禁止配置文件**：无需任何外部配置文件
- ✅ **自动检测硬件**：启动时自动检测 GPU 类型
- ✅ **自动选择最优**：按检测顺序（CUDA > ROCm > Metal > WGPU > CPU）选择

**中间态 vs 源码编译的区别**：

| 类型 | 定义 | 示例 | 允许 |
|------|------|------|------|
| 中间态加载 | 加载预编译的中间表示 | PTX→SASS, HSACO→执行, WGSL→SPIR-V | ✅ |
| 源码编译 | 从高级语言源码编译 | .cu→PTX, .metal→metallib | ❌ |

**WGSL 说明**：WGSL 是 WebGPU 的标准中间表示（IR），wgpu 在运行时将其转换为各平台原生格式（Vulkan SPIR-V、Metal MSL、DirectX DXIL）。这是"中间态到原生码的转换"，类似 PTX 被 CUDA Driver JIT 为 SASS，不是"源码编译"。

---

## ARCH-ROCM-001: ROCm 后端（HSA Runtime Only）

**决策**：只使用 HSA Runtime (Driver API)，无 HIP Runtime

**理由**：
1. HSA Runtime = Driver API，只需 AMD GPU 驱动
2. HIP Runtime = Runtime API，需要完整 ROCm 开发工具包
3. 与 CUDA (cudarc Driver API) 和 Metal (Metal.framework) 架构一致
4. 零配置：用户无需安装 ROCm toolkit

**目录结构**：
```
src/hip_kernels/
├── mod.rs              # 模块导出
├── hsa_runtime.rs      # HSA Runtime 动态加载（libhsa-runtime64.so）
├── hsa_flash_attn.rs   # HSA Flash Attention Kernel
├── hsa_paged_attn.rs   # HSA Paged Attention Kernel
└── kernels/
    ├── flash_attention.hsaco   # 预编译 HSACO（include_bytes!）
    └── paged_attention.hsaco   # 预编译 HSACO（include_bytes!）
```

**公开接口**：

| 模块 | 导出 | 说明 |
|------|------|------|
| `hsa_runtime` | `get_hsa_lib`, `is_hsa_available`, `HsaLib`, `GpuAgent`, `find_gpu_agents` | HSA 运行时抽象 |
| `hsa_flash_attn` | `HsaFlashAttentionKernel`, `HsaBuffer`, `HsaQueueWrapper` | Flash Attention |
| `hsa_paged_attn` | `HsaPagedAttentionKernel` | Paged Attention |
| 便捷函数 | `is_amd_gpu_available()` | 检测 AMD GPU 可用性 |

**HSA Runtime 核心抽象**：

| 类型 | 说明 |
|------|------|
| `HsaLib` | HSA Library 动态加载（通过 libloading） |
| `GpuAgent` | GPU Agent 抽象（handle + name） |
| `HsaKernelModule` | HSA Kernel Module（HSACO 加载） |
| `HsaQueue` | HSA 命令队列 |
| `HsaSignal` | HSA 同步信号 |

| 函数 | 返回类型 | 说明 |
|------|----------|------|
| `is_hsa_available()` | bool | 检测 HSA 是否可用 |
| `get_hsa_lib()` | Result\<HsaLib\> | 获取 HSA 库实例 |
| `find_gpu_agents()` | Vec\<GpuAgent\> | 查找所有 GPU Agent |

**Kernel 实现模式**：

| 组件 | 说明 |
|------|------|
| 预编译 HSACO | 通过 `include_bytes!` 嵌入 |
| `HsaFlashAttentionKernel` | 包含 agent + module_f32 + module_f16 |

| 方法 | 说明 |
|------|------|
| `new(agent)` | 从 GpuAgent 创建 Kernel 实例 |
| `forward_f32(...)` | f32 前向计算 |
| `forward_f16(...)` | f16 前向计算 |

---

## ARCH-METAL-001: Metal 后端（零配置 + metallib 中间态加载）

**决策**：移除 `metal-precompiled` feature，metallib 嵌入为默认行为，无环境变量配置

**设计原则**（遵循 ARCH-LOAD-001）：
- 用户零配置，即插即用
- 只从嵌入的 metallib 中间态加载
- 🚨 禁止从 .metal 源码运行时编译

**加载流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 读取嵌入的 metallib | `include_bytes!("kernels/flash_attention.metallib")` |
| 2 | 检查有效性 | 如果为空，返回 `KernelNotFound` 错误 |
| 3 | 加载到 Metal Device | `device.new_library_with_data()` |
| 4 | 🚨 无回退 | 禁止从 .metal 源码编译！ |

**Metal 目录结构**：
```
src/metal_kernels/
├── mod.rs
├── metal_runtime.rs        # Metal Runtime 抽象
├── metallib_loader.rs      # MetallibCollection 中间态加载
├── flash_attn.rs           # Flash Attention
├── paged_attn.rs           # Paged Attention
└── kernels/
    ├── flash_attention.metallib  # 预编译中间态（CI 生成，include_bytes!）
    └── paged_attention.metallib  # 预编译中间态（CI 生成，include_bytes!）
```

---

## ARCH-DISPATCH-001: 后端派发（已简化）

> ⚠️ **已简化**：参见 ARCH-BACKEND-001，使用 `Arc<dyn Backend>` 替代 enum 派发。
>
> 启动时选一次后端，返回 `Arc<dyn Backend>`，之后直接调用方法。
> 热路径内无任何额外分发开销。

---

## ARCH-RUNTIME-001: 运行时后端检测（全自动）

**设计原则**：
- **完全自动化**：无需用户配置，自动检测最优后端
- **零配置**：不读取任何环境变量
- **确定性顺序**：按性能优先级检测（CUDA > ROCm > Metal > WGPU > CPU）
- **一次检测**：返回 `Arc<dyn Backend>`，后续直接使用

**检测流程**（参见 ARCH-BACKEND-001）：

```rust
pub fn auto_select_backend() -> Arc<dyn Backend> {
    if cuda_available() { return Arc::new(CudaBackend::new()); }
    if rocm_available() { return Arc::new(RocmBackend::new()); }
    if metal_available() { return Arc::new(MetalBackend::new()); }
    if wgpu_available() { return Arc::new(WgpuBackend::new()); }
    Arc::new(CpuBackend::new())
}
```

**检测顺序表**：

| 优先级 | 检测 | 返回 |
|--------|------|------|
| 1 | CUDA | `Arc<CudaBackend>` |
| 2 | ROCm | `Arc<RocmBackend>` |
| 3 | Metal | `Arc<MetalBackend>` |
| 4 | WGPU | `Arc<WgpuBackend>` |
| 5 | CPU | `Arc<CpuBackend>` |

**🚨 设计约束**：
- ❌ **禁止 `GLLM_BACKEND` 环境变量**：用户不应手动指定后端
- ✅ **自动选择最优**：始终选择检测到的最高性能后端

---

## ARCH-FLOW-001: 执行流程（🚨 全流程单后端）

> ⚠️ **核心原则**：选定后端后，整个推理流程在同一后端执行，禁止混合！

**核心流程**：

```
程序启动 → auto_select_backend() → Arc<dyn Backend> → 全流程使用同一 backend
```

| 阶段 | 操作 |
|------|------|
| 1. 后端选择 | `auto_select_backend()` 返回 `Arc<dyn Backend>` |
| 2. 数据上传 | `backend.upload(host_data, &mut gpu_tensor)` |
| 3. **全流程计算** | `backend.flash_attention(...)` → `backend.rms_norm(...)` → ... |
| 4. 结果下载 | `backend.readback(&gpu_tensor, &mut host_data)` |

**🚨 禁止的模式**：

```rust
// ❌ 错误：GPU/CPU 混合执行
let gpu_out = backend.flash_attention(gpu_q, gpu_k, gpu_v);
backend.readback(&gpu_out, &mut host_data);  // 下载到 CPU
let cpu_out = cpu_rms_norm(&host_data);       // CPU 计算
backend.upload(&cpu_out, &mut gpu_tensor);    // 再上传
// ↑ 每次 readback/upload 都是毫秒级开销！

// ✅ 正确：全流程 GPU
let gpu_out = backend.flash_attention(gpu_q, gpu_k, gpu_v);
backend.rms_norm(&gpu_out, &mut gpu_norm);    // GPU 继续
backend.linear_forward(&gpu_norm, ...);        // GPU 继续
backend.readback(&final_gpu, &mut host_result); // 最后才下载
```

**分布式执行**：
- ⚠️ **未实现**：Ring Attention 等分布式算子暂未实现

---

## ARCH-SLICE-001: 原生切片接口

**问题**: Tensor 抽象层会引入额外开销

**决策**: 算子接口使用原生切片

#### flash_attention 接口

| 参数 | 类型 | 布局 |
|------|------|------|
| q | slice\<T\> | [batch, heads, seq_len, head_dim] |
| k | slice\<T\> | [batch, heads, seq_len, head_dim] |
| v | slice\<T\> | [batch, heads, seq_len, head_dim] |
| config | FlashAttentionConfig | 配置参数 |
| 返回值 | Result\<Vec\<T\>\> | 输出数据 |

#### 调用流程

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | Tensor → Slice | 调用方从 Tensor 提取切片 |
| 2 | 执行 Kernel | 传入原生切片调用算子 |
| 3 | Slice → Tensor | 调用方从返回的切片构造 Tensor |

**设计原则**：调用方负责内存管理，gllm-kernels 不依赖任何 Tensor 抽象

---

## 高级算子架构（2025-2026 技术升级）

> **实现状态总览**（2026-01 更新）：
>
> | ID | 算子 | 状态 | 代码位置 | 代码行数 |
> |-----|------|------|----------|----------|
> | ARCH-OP-007 | StreamingLLM | ❌ 未实现 | - | - |
> | ARCH-OP-008 | EAGLE-3 | ✅ 已实现 | `ops/eagle3/` | ~976 行 |
> | ARCH-OP-009 | SpecEE/LayerSkip | ✅ 已实现 | `ops/spec_ee/` | ~937 行 |
> | ARCH-OP-010 | Flash Tree-attention | ✅ 已实现 | `ops/flash_tree_attn.rs` | ~1122 行 |
> | ARCH-OP-011 | INT2/EvicPress | ✅ 已实现 | `ops/int2_quantizer.rs`, `ops/evic_press.rs` | ~1333 行 |
> | ARCH-OP-012 | Infinite Retrieval | ❌ 未实现 | - | - |
> | ARCH-OP-013 | Medusa | ✅ 已实现 | `ops/medusa/` | ~1008 行 |
> | ARCH-OP-014 | Prompt Cache | ✅ 已实现 | `ops/prompt_cache.rs` | ~1333 行 |
> | ARCH-OP-015 | Chunked Prefill | ✅ 已实现 | `ops/chunked_prefill.rs` | ~1048 行 |

### ARCH-OP-007: StreamingLLM / Attention Sink

> **⚠️ 实现状态**：❌ 未实现（设计文档）

**设计目标**: 支持无限长度序列推理，通过 Attention Sink 机制保持生成质量

**核心原理**:
- 初始 token（Attention Sink）聚集大量注意力权重，对生成质量至关重要
- 滑动窗口维护最近 L 个 token 的 KV
- 内存复杂度从 O(T) 降至 O(N+L) 常数

**架构设计**:

```
KV Cache 布局:
┌────────────────┬─────────────────────────────────────┐
│  Sink Tokens   │         Sliding Window              │
│   (固定 N 个)   │      (滑动，最近 L 个)              │
└────────────────┴─────────────────────────────────────┘
   Position: 0..N          Position: T-L..T

Attention 计算:
Q_current × [K_sink | K_window]^T → Attention Scores
```

**StreamingKVCache 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| sink_k | Buffer | Sink token 的 Key: [num_layers, sink_size, num_heads, head_dim] |
| sink_v | Buffer | Sink token 的 Value: [num_layers, sink_size, num_heads, head_dim] |
| window_k | CircularBuffer | 滑动窗口 Key: [num_layers, window_size, num_heads, head_dim] |
| window_v | CircularBuffer | 滑动窗口 Value: [num_layers, window_size, num_heads, head_dim] |
| sink_size | usize | Sink token 数量（默认 4） |
| window_size | usize | 滑动窗口大小（默认 512 或 1024） |
| current_pos | usize | 当前写入位置 |

**CircularBuffer 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| data | Buffer | 底层存储 |
| head | usize | 环形缓冲区头指针 |
| capacity | usize | 环形缓冲区容量 |

**接口设计**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `new` | config: StreamingConfig | Self | 创建 StreamingKVCache |
| `append` | k: Buffer, v: Buffer, pos: usize | () | 追加 KV（自动处理 sink/window 分配） |
| `get_attention_kv` | - | (Buffer, Buffer) | 获取用于 Attention 计算的 KV（sink + window） |
| `clear_window` | - | () | 清空滑动窗口（保留 sink） |

**StreamingConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| sink_size | usize | 4 | Attention Sink token 数量 |
| window_size | usize | 512 | 滑动窗口大小 |
| num_layers | usize | - | 模型层数 |
| num_heads | usize | - | 注意力头数 |
| head_dim | usize | - | 头维度 |

**与现有组件集成**:

| 集成点 | 方式 |
|--------|------|
| PagedAttention | window_k/v 可使用 Paged 存储 |
| KV Cache 压缩 | sink 保持高精度，window 可使用 Low-rank/VQ |
| Ring Attention | 每个节点独立维护 StreamingKVCache |

---

### ARCH-OP-008: EAGLE-3 自适应草稿长度

> **✅ 实现状态**：已实现
> - 代码位置：`src/ops/eagle3/`
> - 核心模块：`decoder.rs`(343行), `predictor.rs`(136行), `scheduler.rs`(110行)
> - 导出：`AdaptiveDraftConfig`, `ConfidencePredictor`, `Eagle3Decoder`, `LengthScheduler`

**设计目标**: 升级投机解码，基于置信度动态调整草稿长度，实现 2-6x 加速

**基于论文**: [EAGLE-3](https://arxiv.org/abs/2504.08850) (NeurIPS'25)

**核心原理**:
- 基于隐藏状态预测 target model 接受概率
- 置信度低时提前终止草稿生成，避免浪费计算
- 学习最优草稿长度分布，自适应不同输入
- **多层特征融合**：融合多个 Transformer 层的隐藏状态（vs EAGLE-2 单层）
- **Token 级预测**：从序列级预测升级为 token 级置信度预测
- **Training-time Test**：训练时模拟测试分布，提升泛化

**架构设计**:

```
Draft Generation Flow:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Hidden State h_t                                           │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │ Confidence Head │ → p_accept = σ(W_c · h_t)              │
│  └─────────────────┘                                        │
│       │                                                     │
│       ▼                                                     │
│  p_accept < threshold? ──Yes──→ Stop Draft (Early Exit)     │
│       │                                                     │
│       No                                                    │
│       │                                                     │
│       ▼                                                     │
│  Generate Next Draft Token                                  │
│       │                                                     │
│       ▼                                                     │
│  draft_length < max_length? ──Yes──→ Continue               │
│       │                                                     │
│       No                                                    │
│       │                                                     │
│       ▼                                                     │
│  Submit to Verification                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**AdaptiveDraftConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| min_draft_length | usize | 1 | 最小草稿长度 |
| max_draft_length | usize | 8 | 最大草稿长度 |
| confidence_threshold | f32 | 0.5 | 置信度阈值 |
| fallback_length | usize | 3 | 验证失败后的回退长度 |
| enable_length_scheduler | bool | true | 是否启用长度调度器 |

**ConfidencePredictor 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| weight | Buffer | 线性层权重: [hidden_dim, 1] |
| bias | f32 | 偏置项 |

**ConfidencePredictor 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `predict` | hidden_state: Buffer | f32 | 预测接受概率（sigmoid 输出） |

**LengthScheduler 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| length_distribution | Vec\<f32\> | 各长度的历史接受率 |
| ema_alpha | f32 | 指数移动平均系数 |
| sample_count | Vec\<usize\> | 各长度的采样次数 |

**LengthScheduler 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `suggest_length` | - | usize | 基于历史建议草稿长度 |
| `update` | length: usize, accepted: usize | () | 更新统计（接受了多少 token） |

**回退策略**:

| 验证结果 | 下次草稿长度 |
|----------|-------------|
| 全部接受 | min(current + 1, max) |
| 部分接受 | max(accepted_count, min) |
| 全部拒绝 | fallback_length |

---

### ARCH-OP-009: SpecEE / LayerSkip 早退出推测

> **✅ 实现状态**：已实现
> - 代码位置：`src/ops/spec_ee/`
> - 核心模块：`engine.rs`(205行), `head.rs`(186行), `cache.rs`(178行)
> - 导出：`SpecEEConfig`, `SpecEEEngine`, `EarlyExitHead`, `LayerDropoutSchedule`, `SharedActivations`

**设计目标**: 整合早退出机制与投机解码，同一模型内部完成草稿-验证循环，目标 2.25-4x 延迟降低

**基于论文**:
- [SpecEE](https://dl.acm.org/doi/10.1145/3695053.3730996) (ISCA'25) - 2.25-2.43x 加速
- [LayerSkip](https://arxiv.org/abs/2404.16710) (ACL'24) - Self-Speculative Decoding

**核心原理**:
- 每层设置 Early Exit Head，计算置信度
- 高置信度时从早期层退出作为"草稿"
- 完整前向作为"验证"，接受或拒绝早退出结果
- **Layer Dropout 训练**：训练时低层低 dropout rate，高层高 dropout rate，增强早期层独立性
- **共享激活优化**：草稿和验证阶段共享早期层的计算和 KV Cache

**架构设计**:

```
Self-Speculation Flow:
┌─────────────────────────────────────────────────────────────┐
│  Layer 0  ──→  Layer 1  ──→  ...  ──→  Layer N-1  ──→ Final │
│     │            │                         │                │
│     ▼            ▼                         ▼                │
│  [EE Head]   [EE Head]               [EE Head]              │
│     │            │                         │                │
│     ▼            ▼                         ▼                │
│  conf_0       conf_1                   conf_N-1             │
│     │            │                         │                │
│     └───────────┴──────────┬──────────────┘                │
│                            │                                │
│                            ▼                                │
│              Max confidence layer = exit_layer              │
│                            │                                │
│                            ▼                                │
│              conf[exit_layer] > threshold?                  │
│                   │                  │                      │
│                  Yes                 No                     │
│                   │                  │                      │
│                   ▼                  ▼                      │
│           Use Early Exit      Continue Full Forward        │
│           (Draft Output)       (Verification)              │
│                   │                  │                      │
│                   └──────────────────┘                      │
│                            │                                │
│                            ▼                                │
│              Verify: EE output == Full output?              │
│                   │                  │                      │
│                  Yes                 No                     │
│                   │                  │                      │
│                   ▼                  ▼                      │
│               Accept EE         Reject, use Full           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**EarlyExitHead 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| lm_head | Buffer | 语言模型头: [hidden_dim, vocab_size] |
| confidence_head | Buffer | 置信度头: [hidden_dim, 1] |
| layer_idx | usize | 所属层索引 |

**EarlyExitHead 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `forward` | hidden: Buffer | (logits: Buffer, confidence: f32) | 输出 logits 和置信度 |

**SpecEEConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| exit_layers | Vec\<usize\> | [6, 12, 18] | 配置早退出的层 |
| confidence_threshold | f32 | 0.8 | 早退出置信度阈值 |
| min_exit_layer | usize | 6 | 最小退出层（保证质量） |
| speculation_depth | usize | 4 | 自推测深度 |
| enable_layer_dropout | bool | true | 启用 Layer Dropout 训练模式 |
| layer_dropout_rate | fn(usize)->f32 | linear(0.1, 0.5) | 层级 dropout rate 函数 |
| share_activations | bool | true | 启用草稿-验证激活共享 |

**SpecEEEngine 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| early_exit_heads | Vec\<EarlyExitHead\> | 各层的早退出头 |
| config | SpecEEConfig | 配置 |
| stats | SpecEEStats | 运行时统计 |

**SpecEEStats 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| exit_layer_counts | Vec\<usize\> | 各层退出次数 |
| acceptance_rate | f32 | 早退出接受率 |
| avg_exit_layer | f32 | 平均退出层 |

---

### ARCH-OP-010: DeFT / Talon Flash Tree-attention

> **✅ 实现状态**：已实现
> - 代码位置：`src/ops/flash_tree_attn.rs`（1122 行）
> - 导出：`FlashTreeAttention`, `TokenTree`, `TreeMask`, `TalonController`, `BatchTreeConfig`, `TalonConfig`

**设计目标**: 优化树验证算法，实现 O(n+m) 复杂度的 Flash Tree-attention，批处理场景加速 2-4x

**基于论文**:
- [DeFT](https://arxiv.org/abs/2404.00242) (ICLR'25) - Flash Tree-attention
- [Talon](https://arxiv.org/abs/2501.08076) (ICLR'26) - 置信度自适应 Token Tree
- [SEQUOIA](https://arxiv.org/abs/2402.12374) - 动态树结构优化

**核心原理**:
- 投机解码生成 token tree（多个候选路径）
- 传统方法逐路径验证，O(n*m) 复杂度
- DeFT 通过树结构分解，一次 Attention 计算所有路径
- **DeFT-Flatten**：均匀分割树结构到 GPU SMs，最大化并行度
- **DeFT-Node**：节点级并行，每个 SM 处理多个节点
- **Talon 置信度自适应**：根据历史接受率动态调整树结构
- **Traversal Verification**：序列级验证替代 token 级，减少验证开销

**架构设计**:

```
Token Tree Structure:
                 [root]
                /  |   \
            [a]   [b]   [c]
           / |     |    / \
        [d] [e]   [f] [g] [h]

Linearized Sequence (DFS order):
[root, a, d, e, b, f, c, g, h]

Tree Mask (Causal + Tree Structure):
        root  a  d  e  b  f  c  g  h
root  [  1   0  0  0  0  0  0  0  0 ]
a     [  1   1  0  0  0  0  0  0  0 ]
d     [  1   1  1  0  0  0  0  0  0 ]
e     [  1   1  0  1  0  0  0  0  0 ]
b     [  1   0  0  0  1  0  0  0  0 ]
f     [  1   0  0  0  1  1  0  0  0 ]
c     [  1   0  0  0  0  0  1  0  0 ]
g     [  1   0  0  0  0  0  1  1  0 ]
h     [  1   0  0  0  0  0  1  0  1 ]
```

**TokenTree 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| tokens | Vec\<TokenId\> | DFS 线性化的 token 序列 |
| parent_indices | Vec\<i32\> | 每个节点的父节点索引（root=-1） |
| depth | Vec\<usize\> | 每个节点的深度 |
| num_nodes | usize | 节点总数 |

**TreeMask 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| mask_data | Buffer | 压缩的掩码数据（bit-packed） |
| num_nodes | usize | 节点数 |

**TreeMask 生成规则**:

| 规则 | 公式 |
|------|------|
| mask[i][j] = 1 当且仅当 | j 是 i 的祖先（含自身） |

**FlashTreeAttention 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| flash_attn_kernel | FlashAttentionKernel | 底层 Flash Attention |
| tree_mask_builder | TreeMaskBuilder | 树掩码构建器 |

**FlashTreeAttention 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `forward` | q: Buffer, k: Buffer, v: Buffer, tree: TokenTree | Buffer | 树注意力前向 |
| `batch_forward` | batch_trees: Vec\<TokenTree\> | Vec\<Buffer\> | 批量树注意力 |

**BatchTreeConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_batch_size | usize | 8 | 最大批大小 |
| max_tree_depth | usize | 8 | 最大树深度 |
| max_nodes_per_tree | usize | 64 | 每棵树最大节点数 |
| partition_strategy | PartitionStrategy | DeFTFlatten | 树分割策略（Flatten/Node） |
| enable_talon | bool | true | 启用 Talon 置信度自适应 |
| traversal_verification | bool | true | 启用序列级 Traversal Verification |

**TalonConfig 结构（置信度自适应树）**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| acceptance_history_size | usize | 100 | 历史接受率窗口大小 |
| tree_expansion_threshold | f32 | 0.8 | 接受率高于此值时扩展树 |
| tree_shrink_threshold | f32 | 0.3 | 接受率低于此值时收缩树 |
| min_branches | usize | 2 | 最小分支数 |
| max_branches | usize | 8 | 最大分支数 |

**复杂度分析**:

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 逐路径验证 | O(n * m * d) | O(n + d) |
| DeFT Flash Tree | O(n + m) | O(m) |

> n = prompt length, m = tree nodes, d = tree depth

---

### ARCH-OP-011: PM-KVQ / EvicPress INT2 量化与驱逐

> **✅ 实现状态**：已实现
> - INT2 量化：`src/ops/int2_quantizer.rs`（501 行）
> - EvicPress 驱逐：`src/ops/evic_press.rs`（832 行）
> - 导出：`Int2Quantizer`, `Int2PackedBuffer`, `ProgressiveKVCache`, `EvicPressConfig`, `TokenImportance`

**设计目标**: 极端 KV Cache 量化，支持 INT2 精度，结合智能驱逐策略，额外节省 25-75% 内存

**基于论文**:
- [PM-KVQ](https://arxiv.org/abs/2406.02069) - Progressive Mixed-precision Quantization
- [EvicPress](https://arxiv.org/abs/2503.00909) - 联合压缩与驱逐策略
- [MiniKV](https://arxiv.org/abs/2411.14625) - 2-bit 极端量化

**核心原理**:
- 2-bit 量化：每个元素只需 2 bits（4 个量化级别）
- 渐进式量化：热 token 保持高精度，冷 token 逐渐降级
- 与现有压缩方案（Low-rank, VQ）兼容
- **EvicPress 联合策略**：压缩和驱逐协同决策，而非独立操作
- **重要性评分**：结合注意力分数、位置衰减、语义重要性

**架构设计**:

```
Progressive Quantization:
┌──────────────────────────────────────────────────────────┐
│                    KV Cache 布局                          │
├──────────────────────────────────────────────────────────┤
│  Hot Zone (FP16)  │  Warm Zone (INT8)  │  Cold Zone (INT2) │
│  [最近 64 tokens] │  [64-256 tokens]   │  [>256 tokens]    │
│   内存: 4B/elem   │   内存: 1B/elem    │   内存: 0.25B/elem│
└──────────────────────────────────────────────────────────┘
         │                   │                    │
         │                   │                    │
         ▼                   ▼                    ▼
      最高精度            4x 压缩             16x 压缩

生成过程中动态降级:
Token 进入 → Hot Zone → (64步后) → Warm Zone → (256步后) → Cold Zone
```

**INT2Quantizer 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| scale | f32 | 量化缩放因子 |
| zero_point | i8 | 零点（对称量化为 0） |

**INT2 编码表**:

| 2-bit 值 | 映射浮点（对称） | 说明 |
|----------|-----------------|------|
| 00 (0) | -1.5 * scale | 最小值 |
| 01 (1) | -0.5 * scale | 负小值 |
| 10 (2) | +0.5 * scale | 正小值 |
| 11 (3) | +1.5 * scale | 最大值 |

**INT2PackedBuffer 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| data | Vec\<u8\> | 打包数据（4 个 INT2 = 1 byte） |
| scales | Vec\<f32\> | 每组的缩放因子 |
| group_size | usize | 量化组大小（默认 128） |
| num_elements | usize | 原始元素数量 |

**打包/解包操作**:

| 操作 | 公式 | 说明 |
|------|------|------|
| pack_4_int2 | byte = (a<<6) \| (b<<4) \| (c<<2) \| d | 4 个 INT2 打包为 1 byte |
| unpack_4_int2 | a=(byte>>6)&3, b=(byte>>4)&3, c=(byte>>2)&3, d=byte&3 | 从 1 byte 解包 4 个 INT2 |

**ProgressiveKVCache 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| hot_k | Buffer\<f16\> | 热区 Key（FP16） |
| hot_v | Buffer\<f16\> | 热区 Value（FP16） |
| warm_k | Buffer\<i8\> | 温区 Key（INT8） |
| warm_v | Buffer\<i8\> | 温区 Value（INT8） |
| cold_k | INT2PackedBuffer | 冷区 Key（INT2） |
| cold_v | INT2PackedBuffer | 冷区 Value（INT2） |
| config | ProgressiveQuantConfig | 配置 |

**ProgressiveQuantConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| hot_size | usize | 64 | 热区大小 |
| warm_size | usize | 192 | 温区大小（64-256） |
| group_size | usize | 128 | INT2 量化组大小 |
| enable_int2 | bool | true | 是否启用 INT2 冷区 |
| enable_evicpress | bool | true | 启用 EvicPress 联合策略 |

**EvicPressConfig 结构（联合压缩驱逐策略）**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_cache_size | usize | 4096 | KV Cache 最大 token 数 |
| eviction_threshold | f32 | 0.9 | 缓存占用率达到此值时触发驱逐 |
| importance_decay | f32 | 0.99 | 位置衰减因子（越远越小） |
| attention_weight | f32 | 0.6 | 注意力分数权重 |
| semantic_weight | f32 | 0.4 | 语义重要性权重 |
| min_keep_tokens | usize | 128 | 最少保留 token 数（sink + 关键 token） |

**EvicPress 决策流程**:

| 缓存状态 | 决策 | 操作 |
|----------|------|------|
| 占用率 < 50% | 无操作 | 正常写入 |
| 50% ≤ 占用率 < 90% | 压缩 | 低重要性 token 降级（FP16→INT8→INT2） |
| 占用率 ≥ 90% | 联合 | 同时压缩 + 驱逐最低重要性 token |

**内存节省分析**:

| 配置 | 1K tokens 内存 | vs FP16 |
|------|---------------|---------|
| 全 FP16 | 4 KB | 基准 |
| INT8 | 1 KB | 4x |
| INT2 | 0.25 KB | 16x |
| 渐进式（64+192+冷区） | 0.5-1 KB | 4-8x |

---

### ARCH-OP-012: Infinite Retrieval 长上下文

> **⚠️ 实现状态**：❌ 未实现（设计文档）
> - 依赖 ARCH-OP-007 StreamingLLM（也未实现）

**设计目标**: 超长上下文检索增强，支持 100K+ token 上下文，不丢失远端重要信息

**核心原理**:
- 与 StreamingLLM（ARCH-OP-007）配合使用
- 对被淘汰的 KV 建立检索索引
- 生成时按需检索并加载历史重要 token

**架构设计**:

```
Infinite Context Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Active Context (StreamingLLM)           │   │
│  │  ┌──────────┬─────────────────────────────────────┐ │   │
│  │  │  Sinks   │        Sliding Window                │ │   │
│  │  │  (N=4)   │        (L=512)                       │ │   │
│  │  └──────────┴─────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           │ 超出窗口的 KV                    │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Historical KV Store                     │   │
│  │  ┌────────────┬─────────────────────────────────┐   │   │
│  │  │  KV Index  │        KV Data                   │   │   │
│  │  │  (HNSW/    │        (Compressed or Paged)     │   │   │
│  │  │   FAISS)   │                                  │   │   │
│  │  └────────────┴─────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           │ Query-based Retrieval           │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Retrieved Context Injection             │   │
│  │  Attention(Q, [K_sink | K_window | K_retrieved])     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**HistoricalKVStore 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| kv_data | PagedBuffer | 分页存储的历史 KV |
| index | KVIndex | 检索索引（基于 Key 的 embedding） |
| metadata | Vec\<KVMetadata\> | KV 元数据（position, importance） |
| config | RetrievalConfig | 检索配置 |

**KVIndex 结构（可选实现）**:

| 实现方式 | 说明 | 适用场景 |
|----------|------|----------|
| LinearScan | 线性扫描 | 历史 KV < 1000 |
| HNSW | 层次化导航小世界图 | 历史 KV > 1000 |
| LSH | 局部敏感哈希 | 超大规模，近似检索 |

**KVMetadata 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| position | usize | 原始位置 |
| importance | f32 | 重要性分数（基于 attention 权重） |
| layer_mask | u32 | 存储了哪些层的 KV（位掩码） |

**RetrievalConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| top_k | usize | 32 | 每次检索返回 token 数 |
| retrieval_interval | usize | 64 | 检索触发间隔（每 N 步检索一次） |
| importance_threshold | f32 | 0.1 | 重要性阈值（低于此值不存储） |
| max_historical_size | usize | 100000 | 最大历史存储 |

**InfiniteContext 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| streaming_cache | StreamingKVCache | 活跃上下文（ARCH-OP-007） |
| historical_store | HistoricalKVStore | 历史存储 |
| retriever | ContextRetriever | 检索器 |

**InfiniteContext 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `append` | k: Buffer, v: Buffer, importance: f32 | () | 追加 KV（自动管理存储位置） |
| `get_attention_kv` | query: Buffer | (Buffer, Buffer) | 获取 Attention KV（sink + window + retrieved） |
| `evict_to_historical` | k: Buffer, v: Buffer, meta: KVMetadata | () | 将淘汰的 KV 存入历史 |
| `retrieve` | query: Buffer, top_k: usize | Vec\<(Buffer, Buffer)\> | 检索相关历史 KV |

**与其他组件集成**:

| 集成点 | 方式 |
|--------|------|
| StreamingLLM | 共享 sink + window，淘汰时触发 evict_to_historical |
| PM-KVQ | 历史 KV 可使用 INT2 量化存储 |
| PagedAttention | historical_store 使用 Paged 存储 |

**性能参数**:

| 指标 | 目标值 |
|------|--------|
| 最大上下文长度 | 100K+ tokens |
| 检索延迟 | < 1ms（GPU）, < 10ms（CPU） |
| 历史存储开销 | < 10% 原始 KV 内存 |

---

### ARCH-OP-013: Assisted Generation / Self-Speculative Decoding

> **✅ 实现状态**：已实现（Medusa）
> - 代码位置：`src/ops/medusa/`
> - 核心模块：`engine.rs`(262行), `head.rs`(223行), `cache.rs`(NgramCache)
> - 导出：`MedusaEngine`, `MedusaHead`, `MedusaDraft`, `NgramCache`, `AssistedGenerationConfig`

**设计目标**: 使用辅助模型或模型自身低层加速生成，无需独立草稿模型，目标 1.5-2x 延迟降低

**基于论文**:
- [Draft & Verify](https://arxiv.org/abs/2309.08168) - Lossless Large Language Model Acceleration
- [Medusa](https://arxiv.org/abs/2401.10774) - Multiple Decode Heads
- [Lookahead Decoding](https://arxiv.org/abs/2402.02057) - N-gram 预测加速

**核心原理**:
- **辅助头方案（Medusa）**：在 LLM 顶层添加多个解码头，并行预测未来 token
- **N-gram 预测**：利用历史 N-gram 统计预测未来 token
- **自推测方案**：使用模型早期层的输出作为草稿（与 LayerSkip 配合）
- **无损加速**：验证阶段保证输出与原始模型完全一致

**架构设计**:

```
Assisted Generation Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Main LLM (Target Model)                    │   │
│  │  [Layer 0] → [Layer 1] → ... → [Layer N-1] → [LM Head]│   │
│  │                                        │              │   │
│  │                                        ▼              │   │
│  │                              ┌────────────────┐      │   │
│  │                              │  Medusa Heads  │      │   │
│  │                              │  [H1][H2][H3]  │      │   │
│  │                              │   ↓   ↓   ↓   │      │   │
│  │                              │  t+1 t+2 t+3  │      │   │
│  │                              └────────────────┘      │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Verification Phase                      │   │
│  │  Run full forward on [t+1, t+2, t+3] candidates     │   │
│  │  Accept matching tokens, reject divergent ones      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**MedusaHead 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| weights | Buffer | 预测权重: [hidden_dim, vocab_size] |
| position_offset | usize | 预测位置偏移（1=下一个，2=下下个） |
| temperature | f32 | 采样温度 |

**AssistedGenerationConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| num_medusa_heads | usize | 3 | Medusa 头数量 |
| speculation_depth | usize | 4 | 推测深度 |
| candidate_count | usize | 8 | 候选 token 数量 |
| use_ngram_draft | bool | true | 是否使用 N-gram 辅助草稿 |
| ngram_size | usize | 3 | N-gram 大小 |
| tree_attention | bool | true | 是否使用树注意力验证（配合 DeFT） |

**AssistedGeneration 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `generate_draft` | hidden: Buffer | Vec\<TokenId\> | 生成草稿 token |
| `verify` | draft_tokens: Vec\<TokenId\> | (Vec\<TokenId\>, usize) | 验证并返回接受的 token |
| `update_ngram` | accepted_tokens: Vec\<TokenId\> | () | 更新 N-gram 统计 |

---

### ARCH-OP-014: Prompt Caching / KV Cache Reuse

> **✅ 实现状态**：已实现
> - 代码位置：`src/ops/prompt_cache.rs`（1333 行）
> - 导出：`PromptCacheManager`, `PromptCacheEntry`, `BlendedKVCache`, `CacheHit`, `EvictionPolicy`, `StorageTier`

**设计目标**: 跨请求复用 KV Cache，减少重复计算，目标 2-15x 吞吐提升

**基于论文**:
- [CacheBlend](https://arxiv.org/abs/2405.16444) (EuroSys'25 Best Paper) - 3.9x RAG 吞吐提升
- [LMCache](https://arxiv.org/abs/2505.12125) - 15x 吞吐，2x 延迟降低
- [vLLM](https://arxiv.org/abs/2309.06180) - PagedAttention 与前缀缓存

**核心原理**:
- **前缀缓存**：相同 prompt 前缀的 KV Cache 可跨请求复用
- **语义融合（CacheBlend）**：不同知识片段的 KV 通过位置重编码融合
- **分层存储**：热 KV 在 GPU，温 KV 在 CPU，冷 KV 在磁盘/网络
- **引用计数**：自动管理 KV 生命周期，支持 CoW（Copy-on-Write）

**架构设计**:

```
Prompt Caching Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Prompt Cache Manager                    │   │
│  │                                                      │   │
│  │   Request A: "System: You are an assistant..."       │   │
│  │   Request B: "System: You are an assistant..."       │   │
│  │             ↘      ↙                                 │   │
│  │        Cache Hit! Reuse KV                           │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Hierarchical KV Storage                 │   │
│  │  ┌──────────┬──────────┬──────────────────────────┐ │   │
│  │  │  GPU L1  │  CPU L2  │    Disk/Network L3       │ │   │
│  │  │  (Hot)   │  (Warm)  │    (Cold)                │ │   │
│  │  │  1GB     │  16GB    │    Unlimited             │ │   │
│  │  └──────────┴──────────┴──────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              CacheBlend Position Reencoding          │   │
│  │  Knowledge A (pos 0-100) + Knowledge B (pos 0-50)   │   │
│  │  → Merged with reencoded positions (0-150)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**PromptCacheEntry 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| hash | u64 | Prompt 内容的哈希值 |
| kv_blocks | Vec\<KVBlockId\> | KV 数据块 ID 列表 |
| token_count | usize | 缓存的 token 数量 |
| ref_count | AtomicUsize | 引用计数 |
| last_access | Instant | 最后访问时间 |
| storage_tier | StorageTier | 存储层级（GPU/CPU/Disk） |

**PromptCacheConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| gpu_cache_size | usize | 1 GB | GPU 缓存大小 |
| cpu_cache_size | usize | 16 GB | CPU 缓存大小 |
| enable_disk_cache | bool | true | 是否启用磁盘缓存 |
| eviction_policy | EvictionPolicy | LRU | 驱逐策略 |
| enable_cacheblend | bool | true | 启用 CacheBlend 语义融合 |
| min_prefix_length | usize | 64 | 最小缓存前缀长度 |
| hash_algorithm | HashAlgorithm | xxHash64 | 哈希算法 |

**PromptCacheManager 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `lookup` | prompt_tokens: Vec\<TokenId\> | Option\<CacheHit\> | 查找前缀缓存 |
| `insert` | prompt_tokens: Vec\<TokenId\>, kv: KVCache | CacheEntryId | 插入缓存 |
| `evict_lru` | target_size: usize | usize | 驱逐最少使用的条目 |
| `blend_knowledge` | entries: Vec\<CacheEntryId\> | KVCache | CacheBlend 融合多个知识片段 |
| `prefetch` | entry_id: CacheEntryId | () | 预取到高层存储 |

---

### ARCH-OP-015: Chunked Prefill Attention

> **✅ 实现状态**：已实现
> - 代码位置：`src/ops/chunked_prefill.rs`（1048 行）
> - 导出：`ChunkedPrefillScheduler`, `ChunkConfig`, `PODAttentionConfig`, `PrefillRequest`, `DecodeRequest`, `ScheduledBatch`

**设计目标**: 分块预填充与解码重叠执行，优化长 prompt 场景，目标 10-22% 吞吐提升

**基于论文**:
- [POD-Attention](https://arxiv.org/abs/2411.13369) (ASPLOS'25) - 22% 吞吐提升
- [FlashInfer](https://flashinfer.ai/) - Customizable Attention Engine
- [Sarathi](https://arxiv.org/abs/2308.16369) - Chunked Prefill 与流水线

**核心原理**:
- **分块预填充**：将长 prompt 分成多个 chunk，逐块计算 KV
- **Prefill-Decode 重叠**：prefill 某些请求的同时 decode 其他请求
- **动态调度**：根据 GPU 利用率动态分配 prefill/decode 资源
- **内存效率**：chunk 级别的 KV 管理，避免大块内存分配

**架构设计**:

```
Chunked Prefill Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Request Queue                           │   │
│  │  [Prefill A: 8K tokens] [Decode B] [Decode C] ...   │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Chunk Scheduler                         │   │
│  │  Prefill A: [Chunk 0-2K] [Chunk 2K-4K] [Chunk 4K-6K]│   │
│  │             [Chunk 6K-8K]                            │   │
│  │                                                      │   │
│  │  Interleave with decodes:                           │   │
│  │  [Chunk 0] → [Decode B,C] → [Chunk 1] → [Decode] ...│   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              POD-Attention Kernel                    │   │
│  │                                                      │   │
│  │  ┌──────────────────┬──────────────────┐            │   │
│  │  │   Prefill SMs    │    Decode SMs    │            │   │
│  │  │   (60%)          │    (40%)         │            │   │
│  │  │   Chunk Attn     │    Token Attn    │            │   │
│  │  └──────────────────┴──────────────────┘            │   │
│  │                                                      │   │
│  │  Dynamic SM allocation based on load                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**ChunkConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| chunk_size | usize | 2048 | 每个 chunk 的 token 数 |
| max_chunks_per_batch | usize | 4 | 每批最大 chunk 数 |
| interleave_decodes | bool | true | 是否与 decode 交织执行 |
| dynamic_chunk_size | bool | true | 根据 prompt 长度动态调整 chunk 大小 |

**PODAttentionConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prefill_sm_ratio | f32 | 0.6 | Prefill 分配的 SM 比例 |
| decode_sm_ratio | f32 | 0.4 | Decode 分配的 SM 比例 |
| enable_dynamic_allocation | bool | true | 动态 SM 分配 |
| min_sm_per_task | usize | 4 | 每任务最小 SM 数 |

**ChunkedPrefillScheduler 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| prefill_queue | VecDeque\<PrefillRequest\> | 待处理的 prefill 请求 |
| decode_queue | VecDeque\<DecodeRequest\> | 待处理的 decode 请求 |
| chunk_config | ChunkConfig | 分块配置 |
| pod_config | PODAttentionConfig | POD-Attention 配置 |

**ChunkedPrefillScheduler 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `submit_prefill` | request: PrefillRequest | RequestId | 提交 prefill 请求 |
| `submit_decode` | request: DecodeRequest | RequestId | 提交 decode 请求 |
| `schedule_batch` | () | ScheduledBatch | 调度下一批次（混合 prefill chunk + decode） |
| `execute_batch` | batch: ScheduledBatch | Vec\<Output\> | 执行批次 |

**性能对比**:

| 场景 | 无 Chunked Prefill | 有 Chunked Prefill | 提升 |
|------|-------------------|-------------------|------|
| 长 prompt (8K) | 基准 | +15% 吞吐 | POD-Attention 重叠 |
| 混合负载 | 基准 | +22% 吞吐 | 资源利用率提升 |
| 批处理延迟 | 基准 | -30% P99 延迟 | 更平滑的调度 |

---

## 数值稳定性算法

### ARCH-ALGO-001: Kahan 补偿求和

**问题**: 浮点累加误差随序列长度线性增长 O(n)

**解决方案**: Kahan 算法将误差降至 O(1)

#### KahanAccumulator 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| sum | Float | 累加结果 |
| c | Float | 补偿项（lost bits） |

#### add 方法步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | y = value - c | 补偿上次丢失的位 |
| 2 | t = sum + y | 可能丢失低位 |
| 3 | c = (t - sum) - y | 计算本次丢失的位 |
| 4 | sum = t | 更新累加结果 |

**位置**: `ops/stable_accumulator.rs`

### ARCH-ALGO-002: Log-Space Softmax

**问题**: 2M+ token 序列的 exp() 会溢出

**解决方案**: 在对数空间计算，使用 log-add-exp 技巧

#### LogSpaceSoftmax 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| m | f32 | 当前最大值 |
| log_l | f32 | log(Σ exp(x_i - m)) |

#### update 方法流程

| 条件 | 操作 | 说明 |
|------|------|------|
| x > m | 更新 log_l 和 m | 发现新最大值，重新计算 |
| x ≤ m | log_add_exp | 使用 log-add-exp 累加 |

**位置**: `ops/softmax.rs`

### ARCH-ALGO-003: 分层累加器

**问题**: 超长序列即使使用 Kahan 也会有误差累积

**解决方案**: 多级分层累加，定期合并部分和

#### HierarchicalAccumulator 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| levels | Vec\<KahanAccumulator\> | 多级累加器 |
| counts | Vec\<usize\> | 各级计数 |
| config | AccumulatorConfig | 配置参数 |

**位置**: `ops/stable_accumulator.rs`

---

## 中间态编译架构（Fat Binary）

### 核心设计原则

1. 所有后端的 kernel 预编译为中间态（PTX/HSACO/metallib）
2. 中间态通过 `include_bytes!` 嵌入到可执行文件
3. 运行时通过 Driver API 加载中间态并执行
4. 无编译时链接依赖，只需目标平台的 GPU 驱动

### 编译流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Kernel 编译流程（CI/离线）                        │
├─────────────────────────────────────────────────────────────────────┤
│  CUDA Kernel (.cu)     ──nvcc -ptx──→     PTX (.ptx)     ──┐        │
│  HIP Kernel (.hip)     ──hipcc --genco──→ HSACO (.hsaco)   ├→ embed │
│  Metal Shader (.metal) ──xcrun metallib─→ metallib (.metallib)      │
│  WGSL Shader (.wgsl)   ──────────────────→ 直接嵌入（include_str!）──┘│
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    运行时加载（用户机器）                             │
├─────────────────────────────────────────────────────────────────────┤
│  嵌入的 PTX      + CUDA Driver API (cudarc)  → NVIDIA GPU 执行      │
│  嵌入的 HSACO    + HSA Runtime (libloading)  → AMD GPU 执行         │
│  嵌入的 metallib + Metal Framework           → Apple GPU 执行       │
│  嵌入的 WGSL     + wgpu 运行时编译           → 跨平台 GPU 执行      │
└─────────────────────────────────────────────────────────────────────┘
```

### Driver API vs Runtime API

| 层级 | NVIDIA | AMD | Apple | 特点 |
|------|--------|-----|-------|------|
| **Driver API** | `libcuda.so` | `libhsa-runtime64.so` | Metal.framework | 只需驱动 |
| **Runtime API** | `libcudart.so` | `libamdhip64.so` | - | 需要开发工具包 |

**gllm-kernels 使用 Driver API**：用户机器只需安装 GPU 驱动，无需完整开发工具包。

---

## 目录结构

```
gllm-kernels/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # 公共导出
│   ├── backend.rs                # 后端类型定义
│   ├── runtime_detection.rs      # 运行时检测
│   ├── kernel_dispatcher.rs      # 零成本派发
│   ├── types.rs                  # 公共类型
│   │
│   ├── cuda_kernels/             # CUDA 后端（统一结构）
│   │   ├── mod.rs
│   │   ├── runtime.rs            # CUDA Driver API
│   │   ├── flash_attn.rs
│   │   ├── paged_attn.rs
│   │   └── kernels/
│   │       ├── flash_attention.ptx
│   │       └── paged_attention.ptx
│   │
│   ├── hip_kernels/              # ROCm 后端（HSA Runtime Only）
│   │   ├── mod.rs
│   │   ├── hsa_runtime.rs        # HSA Runtime 动态加载
│   │   ├── hsa_flash_attn.rs     # HSA Flash Attention
│   │   ├── hsa_paged_attn.rs     # HSA Paged Attention
│   │   └── kernels/
│   │       ├── flash_attention.hsaco
│   │       └── paged_attention.hsaco
│   │
│   ├── metal_kernels/            # Metal 后端
│   │   ├── mod.rs
│   │   ├── metal_runtime.rs      # Metal Framework 抽象
│   │   ├── metallib_loader.rs    # MetallibCollection 统一加载
│   │   ├── flash_attn.rs         # Flash Attention
│   │   ├── paged_attn.rs         # Paged Attention
│   │   └── kernels/
│   │       ├── flash_attention.metallib
│   │       ├── flash_attention.metal
│   │       ├── paged_attention.metallib
│   │       └── paged_attention.metal
│   │
│   ├── wgpu_kernels/             # WGPU 后端
│   │   ├── mod.rs
│   │   ├── runtime.rs            # wgpu 抽象
│   │   ├── flash_attn.rs
│   │   ├── paged_attn.rs
│   │   └── kernels/
│   │       ├── flash_attention.wgsl
│   │       └── paged_attention.wgsl
│   │
│   ├── ops/                      # 纯 Rust 算子实现（与后端无关）
│   │   ├── mod.rs
│   │   ├── softmax.rs            # Log-Space Softmax
│   │   ├── stable_accumulator.rs # Kahan/层级累加器
│   │   ├── linear.rs             # 线性层
│   │   ├── rms_norm.rs           # RMS 归一化
│   │   ├── layer_norm.rs         # Layer 归一化
│   │   ├── activations.rs        # 激活函数 (SiLU, GELU, ReLU, etc.)
│   │   ├── rope.rs               # RoPE 位置编码
│   │   ├── sampling.rs           # 采样算子
│   │   ├── moe_routing.rs        # MoE 路由
│   │   ├── embedding.rs          # 嵌入算子
│   │   ├── engram*.rs            # Engram 条件记忆
│   │   ├── eagle3/               # EAGLE-3 自适应草稿 (ARCH-OP-008)
│   │   ├── spec_ee/              # SpecEE/LayerSkip (ARCH-OP-009)
│   │   ├── medusa/               # Medusa 辅助生成 (ARCH-OP-013)
│   │   ├── flash_tree_attn.rs    # Flash Tree-attention (ARCH-OP-010)
│   │   ├── int2_quantizer.rs     # INT2 量化 (ARCH-OP-011)
│   │   ├── evic_press.rs         # EvicPress 驱逐 (ARCH-OP-011)
│   │   ├── prompt_cache.rs       # Prompt Cache (ARCH-OP-014)
│   │   └── chunked_prefill.rs    # Chunked Prefill (ARCH-OP-015)
│   │
│   └── comm/                     # 通信后端
│       ├── mod.rs
│       ├── nccl.rs
│       └── tcp.rs
│
├── scripts/
│   ├── compile_cuda_kernels.sh   # CUDA PTX 编译
│   ├── compile_hip_kernels.sh    # HIP HSACO 编译
│   └── compile_metal_kernels.sh  # Metal metallib 编译
│
├── SPEC/                         # 设计文档
└── benches/                      # 性能测试
```

---

## 与 gllm 的集成

### ARCH-INT-001: 集成架构

> ⚠️ **Burn-Free 架构**：根据 ADR-001，本项目已完全移除 Burn 依赖，
> 使用原始切片 `&[T]` + KernelDispatcher 实现零成本抽象。

```
┌─────────────────────────────────────────────────────────────────────┐
│                           gllm                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Model Layer                                                         │
│    └── 原始切片 &[T] + WeightMatrix/WeightVector 用于权重加载         │
├─────────────────────────────────────────────────────────────────────┤
│  Attention Layer                                                     │
│    ├── 检测 gllm-kernels 可用性（运行时后端检测）                     │
│    ├── 可用 → 调用 KernelDispatcher::flash_attention()               │
│    └── 不可用 → fallback 到 CPU 参考实现                             │
└─────────────────────────────────────────────────────────────────────┘
           │
           │ 调用
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        gllm-kernels                                  │
├─────────────────────────────────────────────────────────────────────┤
│  KernelDispatcher                                                    │
│    ├── 运行时选择后端（CUDA/ROCm/Metal/WGPU/CPU）                    │
│    ├── enum + match 零成本派发（无 vtable）                          │
│    └── 使用 2M 上下文优化算法                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature 配置（最小化）

> ⚠️ **同步状态**：以下配置与 Cargo.toml 保持同步（2025-01）

```toml
[features]
# 默认：完整支持
default = ["full"]

# 完整版：所有后端 + 所有 kernel（Fat Binary，全部嵌入）
full = ["all-backends", "all-kernels"]

# 所有后端（运行时检测，动态加载）
all-backends = []

# 所有自定义 kernel
all-kernels = []

# Fusion 支持（已移除 Burn 依赖，保留 feature 以兼容）
fusion = []

# NCCL 多 GPU（需要 CUDA 环境）
nccl = ["cudarc/nccl"]

# RCCL 多 GPU（需要 ROCm 环境）
rccl = []

# Flash Attention v3 优化（Hopper+）
flash-attention-v3 = [
    "flash-attention-v3-wgmma",
    "flash-attention-v3-async",
    "flash-attention-v3-fp8",
    "flash-attention-v3-block-quant",
]

# 精简版（仅 CPU，用于测试/CI）
minimal = []

# Fat Binary 预编译内核（嵌入 ~10MB）
embedded-kernels = []

# 运行时下载内核（首次使用时从 GitHub Release 下载）
download-kernels = []
```

---

## ARCH-EMBED-001: Embedding/Rerank 快路径架构

### 核心目标

基于学术前沿研究实现 Embedding 和 Rerank 的硬件加速快路径：
- **Binary Quantization**: 1-bit embedding，POPCNT+SIMD，32x 吞吐提升
- **Int8 Quantization**: AVX512-VNNI/CUDA INT8，4x 吞吐提升
- **Int4 Packed**: 2 values/byte，8x 内存带宽提升
- **Matryoshka Truncation**: 运行时维度选择（1024→512→256→128）
- **三阶段 Rerank**: Binary 粗筛 → Int8 精排 → Cross-encoder 终排

### 实际目录结构

> ⚠️ **实现位置**：Embedding 操作统一在 `ops/embedding.rs`，使用纯 Rust + SIMD 实现。

```
src/ops/
├── embedding.rs                # 统一的 Embedding 操作模块
│   ├── BinaryIpConfig          # Binary Quantization 配置
│   ├── Int8DotConfig           # Int8 点积配置
│   ├── Int4PackedConfig        # Int4 打包配置
│   ├── MatryoshkaConfig        # Matryoshka 截断配置
│   ├── RerankPipelineConfig    # 三阶段 Rerank 配置
│   └── 纯 Rust SIMD 实现       # 无需单独 GPU kernel
```

**设计决策**：
- Embedding 操作主要是内存带宽受限，纯 Rust + SIMD 已足够高效
- 使用 `packed_simd` / 手动 SIMD 内联汇编优化
- 避免独立 kernel 模块的额外复杂度

### ARCH-EMBED-002: Binary Quantization Inner Product

**学术来源**: Matryoshka Quantization (2025.02), SimSIMD

**核心原理**:
- 将 f32 embedding 量化为 1-bit（sign bit）
- 使用 POPCNT 指令计算 Hamming distance
- 转换为 cosine 相似度近似

**数学基础**:
```
# 原始内积
IP(a, b) = Σ aᵢ * bᵢ

# Binary 量化
q(x) = sign(x) ∈ {-1, +1} → 存储为 0/1

# Binary 内积（POPCNT）
IP_bin(a, b) = popcount(a XOR b)
cosine ≈ 1 - 2 * IP_bin / dim
```

**BinaryEmbedding 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| data | Vec\<u64\> | 打包的 bit 向量（dim/64 个 u64） |
| dim | usize | 原始维度 |

**BinaryEmbedding 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `from_f32` | embedding: slice\<f32\> | Self | 从 f32 embedding 量化 |
| `binary_inner_product` | other: Self | u32 | Binary 内积（返回 Hamming distance） |
| `batch_inner_product` | candidates: slice, dispatcher | Vec\<u32\> | 批量内积（GPU 加速） |

**各后端实现**:

| 后端 | 指令/API | 预期吞吐 |
|------|----------|----------|
| CPU (x86) | `_mm_popcnt_u64` + AVX2 | 10B ops/s |
| CUDA | `__popc()` + warp reduce | 100B+ ops/s |
| ROCm | `__builtin_popcount` | 100B+ ops/s |
| Metal | `simd_popcnt` | 50B+ ops/s |
| WGPU | bit 操作模拟 | 5B ops/s |

### ARCH-EMBED-003: Int8 Embedding Dot Product

**学术来源**: PilotANN (2025.03), Intel VNNI

**核心原理**:
- f32 embedding 量化为 int8（scale+zero_point）
- 使用 VNNI/DOT4 指令 4x 加速
- 反量化恢复 f32 结果

**Int8Embedding 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| data | Vec\<i8\> | 量化后的 int8 向量 |
| scale | f32 | 缩放因子 |
| zero_point | i8 | 零点偏移（对称量化时为 0） |
| dim | usize | 原始维度 |

**量化方法 - 对称量化**:

| 步骤 | 操作 | 公式 |
|------|------|------|
| 1 | 计算最大绝对值 | `max_abs = max(abs(embedding))` |
| 2 | 计算缩放因子 | `scale = max_abs / 127.0` |
| 3 | 量化每个元素 | `data[i] = round(embedding[i] / scale)` |
| 4 | 设置零点 | `zero_point = 0`（对称量化） |

**各后端实现**:

| 后端 | 指令/API | 吞吐提升 |
|------|----------|----------|
| CPU (AVX512) | `_mm512_dpbusd_epi32` (VNNI) | 4x |
| CPU (AVX2) | `_mm256_maddubs_epi16` | 2x |
| CUDA | `__dp4a()` (INT8 DP4A) | 4x |
| ROCm | `__builtin_amdgcn_sdot4` | 4x |
| Metal | `simd_dot` (int8x4) | 4x |
| WGPU | 手动展开 | 1.5x |

### ARCH-EMBED-004: Int4 Packed Embedding

**学术来源**: Matryoshka Quantization multi-precision (int8→int4→int2)

**核心原理**:
- 2 个 int4 值打包到 1 个 byte
- 8x 内存带宽提升
- 计算时解包为 int8 或 f16

**Int4PackedEmbedding 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| data | Vec\<u8\> | 打包数据（dim/2 bytes） |
| scale | f32 | 缩放因子 |
| dim | usize | 原始维度 |

**打包/解包操作**:

| 操作 | 输入 | 输出 | 公式 |
|------|------|------|------|
| pack_int4 | a: i8, b: i8 | u8 | `((a & 0x0F) << 4) \| (b & 0x0F)` |
| unpack_int4 | packed: u8 | (i8, i8) | `high = (packed >> 4) - 8`, `low = (packed & 0x0F) - 8` |

> **注**：int4 范围 [-8, 7]，存储时加 8 变为 [0, 15] 无符号。

### ARCH-EMBED-005: Matryoshka Dimension Truncation

**学术来源**: Matryoshka Representation Learning (MRL)

**核心原理**:
- Embedding 模型训练时使用嵌套损失
- 前 N 维包含最重要信息
- 运行时按需截断维度（1024→512→256→128）

**MatryoshkaConfig 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| full_dim | usize | 原始维度 |
| truncation_points | Vec\<usize\> | 可用截断点（必须是训练时使用的） |

**常见配置**:

| 配置名 | full_dim | truncation_points |
|--------|----------|-------------------|
| default_1024 | 1024 | [1024, 512, 256, 128] |

**truncate 操作**:

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| truncate | embedding: slice, target_dim: usize | slice[..target_dim] | 零成本切片，仅改变范围 |

**与量化结合**:
```
阶段1 (粗筛): Binary @ 128 dim → 32x 加速，4bit/vector
阶段2 (精排): Int8 @ 512 dim → 4x 加速，512B/vector
阶段3 (终排): FP32 @ 1024 dim → 基准精度，4KB/vector
```

### ARCH-EMBED-006: 三阶段 Rerank 管道

**学术来源**: PE-Rank (2024.06), Cohere Rerank v3

**流程设计**:
```
Query Embedding
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  阶段1: Binary 粗筛 (GPU Kernel)                          │
│  - 输入: Query (binary) vs 100K candidates (binary)       │
│  - 算法: POPCNT Hamming distance                          │
│  - 输出: Top-1000 candidates                              │
│  - 耗时: ~1ms (GPU), ~10ms (CPU)                          │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  阶段2: Int8 精排 (GPU Kernel)                            │
│  - 输入: Query (int8) vs Top-1000 candidates (int8)       │
│  - 算法: INT8 dot product (VNNI/DP4A)                     │
│  - 输出: Top-100 candidates                               │
│  - 耗时: ~0.5ms (GPU), ~5ms (CPU)                         │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  阶段3: Cross-encoder 终排 (可选，LLM 调用)               │
│  - 输入: Query text + Top-100 passages                    │
│  - 算法: BERT/T5 cross-attention                          │
│  - 输出: Top-10 ranked results                            │
│  - 耗时: ~50ms (GPU)                                      │
└──────────────────────────────────────────────────────────┘
```

**RerankPipeline 结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| binary_kernel | BinaryIpKernel | Binary 内积 Kernel |
| int8_kernel | Int8DpKernel | Int8 点积 Kernel |
| config | RerankConfig | 管道配置 |

**RerankConfig 结构**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| stage1_topk | usize | 1000 | 阶段1输出数量 |
| stage2_topk | usize | 100 | 阶段2输出数量 |
| enable_cross_encoder | bool | false | 是否启用阶段3（需要 LLM） |

**RerankPipeline 接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `rerank` | query: Embedding, candidates: EmbeddingIndex, dispatcher | Vec\<RerankResult\> | 执行三阶段 Rerank |

### ARCH-EMBED-007: 实现策略

> ⚠️ **纯 Rust 实现**：Embedding 操作使用纯 Rust + SIMD，无需 GPU kernel。

**设计理由**：
- Embedding 操作是内存带宽受限，CPU SIMD 已足够高效
- 避免额外的 GPU kernel 模块复杂度
- 减少编译时间和二进制大小

**SIMD 加速**：
- Binary IP: 使用 POPCNT 指令（`std::arch::x86_64::_popcnt64`）
- Int8 Dot: 使用 AVX2/AVX512 向量化
- Int4 Packed: 使用位操作优化

### ARCH-EMBED-008: 公开 API

**ops/embedding.rs 模块导出**:

| 导出项 | 类型 | 说明 |
|--------|------|------|
| BinaryIpConfig | 结构 | Binary 内积配置 |
| pack_binary_f32 | 函数 | f32 → binary 打包 |
| binary_ip_hamming | 函数 | Hamming distance 计算 |
| binary_ip_hamming_simd | 函数 | SIMD 加速版本 |
| binary_ip_asymmetric | 函数 | 非对称 Binary IP |
| Int8DotConfig | 结构 | Int8 点积配置 |
| quantize_to_int8 | 函数 | f32 → int8 量化 |
| int8_dot_product | 函数 | Int8 点积 |
| int8_dot_product_unrolled | 函数 | 展开优化版本 |
| Int4PackedConfig | 结构 | Int4 打包配置 |
| pack_int4 | 函数 | Int4 打包 |
| unpack_int4 | 函数 | Int4 解包 |
| quantize_to_int4_packed | 函数 | f32 → int4 打包 |
| int4_packed_dot_product | 函数 | Int4 打包点积 |
| MatryoshkaConfig | 结构 | Matryoshka 配置 |
| matryoshka_truncate | 函数 | 维度截断 |
| select_matryoshka_dim | 函数 | 选择截断维度 |
| RerankPipelineConfig | 结构 | Rerank 管道配置 |
| RerankResult | 结构 | Rerank 结果 |
| rerank_binary_stage | 函数 | Binary 阶段 |
| rerank_int8_stage | 函数 | Int8 阶段 |

**KernelDispatcher Embedding 扩展接口**:

| 方法 | 参数 | 返回类型 | 说明 |
|------|------|----------|------|
| `binary_inner_product_batch` | query: BinaryEmbedding, candidates: slice | Result\<Vec\<u32\>, Error\> | Binary 内积批量计算 |
| `int8_dot_product_batch` | query: Int8Embedding, candidates: slice | Result\<Vec\<i32\>, Error\> | Int8 点积批量计算 |
| `int4_dot_product_batch` | query: Int4PackedEmbedding, candidates: slice | Result\<Vec\<i32\>, Error\> | Int4 点积批量计算 |

---

## ARCH-API-001: 统一泛型算子 API 规范（🚨 FROZEN - 零成本铁律）

### 核心原则

**所有实现必须是零成本的，所有算子必须使用泛型 `<T: Float>`**。

### 零成本保证机制

| 机制 | 说明 | 开销 |
|------|------|------|
| 泛型单态化 | `<T: Float>` 编译时展开为具体类型函数 | 零 |
| const TYPE_ID | `T::TYPE_ID` 是编译时常量，分支被优化消除 | 零 |
| `#[inline(always)]` | 强制内联，无函数调用开销 | 零 |
| 原始切片 | `&[T]` 无任何抽象层 | 零 |

### 泛型 Float Trait

```rust
pub trait Float: Copy + Send + Sync + 'static + Default {
    /// 编译时类型标识，用于 GPU kernel 选择（const = 零成本）
    const TYPE_ID: FloatType;

    fn zero() -> Self;
    fn one() -> Self;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl Float for f32 { const TYPE_ID: FloatType = FloatType::F32; /* ... */ }
impl Float for f16 { const TYPE_ID: FloatType = FloatType::F16; /* ... */ }
impl Float for bf16 { const TYPE_ID: FloatType = FloatType::BF16; /* ... */ }
```

### 统一算子签名

```rust
// 纯泛型 API - 编译时单态化 = 零成本
pub fn flash_attention<T: Float>(
    q: &[T], k: &[T], v: &[T],
    output: &mut [T],
    batch: usize, seq_len: usize, num_heads: usize, head_dim: usize,
) -> Result<(), KernelError>;

pub fn softmax<T: Float>(
    input: &[T],
    output: &mut [T],
    shape: (usize, usize),
) -> Result<(), KernelError>;
```

### 禁止的模式

| 禁止 | 原因 | 正确做法 |
|------|------|----------|
| ❌ `flash_attention_f32()` | 类型后缀 = 代码重复 | ✅ `flash_attention::<f32>()` |
| ❌ `Tensor<B, D>` | Burn 抽象 = 运行时开销 | ✅ `&[T]` 原始切片 |
| ❌ 运行时类型判断 | `if type == f32` = 分支开销 | ✅ `T::TYPE_ID` const 分支 |
| ❌ trait object | `dyn Float` = vtable 开销 | ✅ `<T: Float>` 静态分发 |

### GPU Kernel 实现细节（内部，用户不可见）

GPU kernel 文件按类型分开（PTX/HSACO 不支持泛型），但这是**实现细节**：

```rust
// 内部实现 - const 分支被编译器完全优化掉
#[inline(always)]
fn dispatch_kernel<T: Float>(...) {
    // T::TYPE_ID 是 const，编译器直接消除其他分支
    match T::TYPE_ID {
        FloatType::F32 => /* 直接内联 f32 kernel 调用 */,
        FloatType::F16 => /* 直接内联 f16 kernel 调用 */,
        FloatType::BF16 => /* 直接内联 bf16 kernel 调用 */,
    }
}
```

> **编译结果**：`flash_attention::<f32>()` 直接编译为 f32 kernel 调用，无任何分支判断。

### 验收标准

- [ ] 所有公开 API 使用 `<T: Float>` 泛型
- [ ] 无任何 `_f32`、`_f16`、`_bf16` 后缀的公开函数
- [ ] 无任何 `dyn` trait object
- [ ] 无任何运行时类型判断（仅 const 分支）
- [ ] Float trait 支持 f32、f16、bf16 三种类型
- [ ] 所有热路径函数标记 `#[inline(always)]`

---

## ADR-001: 删除 Burn 依赖，统一到 kernel_dispatcher（2025-01-16）

### 背景

gllm-kernels 存在两套独立实现：
- **ops/ 层**：Burn Tensor 抽象，包含论文优化算法
- **kernel_dispatcher**：原始切片 `&[T]`，GPU 调度层

### 问题

| 问题 | 影响 |
|------|------|
| Burn Tensor 开销 | 内存布局、trait dispatch、无法内联 |
| GPU 调度分离 | ops/ 论文算法无法使用 GPU 加速 |

### 🚨 重要发现：ops/ 包含论文优化，不能删除

ops/ 层包含多轮基于论文的算法优化实现：

| 算法 | 论文来源 | ops/ 核心优化 | kernel_dispatcher 现状 |
|------|----------|--------------|----------------------|
| EAGLE-3 | NeurIPS'25 | 多层融合 + 自适应调度（984行） | 仅 GPU 调度层 |
| Medusa | ICML'24 | N-gram 缓存 + 树生成（818行） | 仅 GPU 调度层 |
| FlashAttention | FlashAttention-2 | 分层块 + MaskCache LRU（1326行） | 简化 CPU fallback |
| Softmax | 数值稳定性 | Log-space + Kahan 累加（433行） | 基础实现 |
| PagedAttention | vLLM | 多级层级 + CoW 管理 | 基础实现 |

### 决策：迁移 ops/ 算法到 kernel_dispatcher（🚨 铁律）

**正确做法**：迁移论文优化算法，去除 Burn 依赖，而非删除

```
迁移策略：
├── 阶段1: 升级 KernelFloat trait
│   ├── 添加 const TYPE_ID: FloatType（零成本分支消除）
│   └── 添加 bf16 支持
│
├── 阶段2: 迁移轻度依赖模块（Burn 仅作数据容器）
│   ├── int2_quantizer.rs → 转换 Tensor<B,D> 为 &[T]
│   └── evic_press.rs     → 转换 Tensor<B,D> 为 &[T]
│
├── 阶段3: 迁移中度依赖模块（保留论文核心算法）
│   ├── eagle3.rs   → 迁移多层融合 + 自适应调度逻辑
│   ├── spec_ee.rs  → 迁移推测执行优化
│   └── medusa.rs   → 迁移 N-gram 缓存 + 树生成
│
├── 阶段4: 迁移重度依赖模块（论文核心实现）
│   ├── flash_attention.rs → 迁移分层块 + MaskCache
│   ├── softmax.rs         → 迁移 Log-space + Kahan
│   └── paged_attention.rs → 迁移多级层级 + CoW
│
└── 阶段5: 清理
    └── 删除空的 ops/ 模块（迁移完成后）

保留（已是纯 Rust，无需迁移）：
├── ops/engram*.rs              → 语义记忆，纯 SIMD
├── ops/embedding.rs            → 向量操作，纯 SIMD
└── ops/stable_accumulator.rs   → 数值工具
```

### 迁移模式：Burn Tensor → 原始切片

```rust
// 迁移前（Burn Tensor）
pub fn flash_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4>

// 迁移后（零成本泛型）
pub fn flash_attention<T: Float>(
    q: &[T], k: &[T], v: &[T],
    output: &mut [T],
    batch: usize, seq_len: usize, num_heads: usize, head_dim: usize,
) -> Result<(), KernelError>
```

### 统一架构

```
用户 API
    │
    ▼
KernelDispatcher（唯一入口）
    │
    ├── BackendType::Cuda  → cuda_kernels/
    ├── BackendType::Rocm  → hip_kernels/
    ├── BackendType::Metal → metal_kernels/
    ├── BackendType::Wgpu  → wgpu_kernels/
    └── BackendType::Cpu   → 论文优化算法（从 ops/ 迁移）
```

### 零成本保证

| 要求 | 实现 |
|------|------|
| 无 Tensor 抽象 | 原始切片 `&[T]` |
| 无 vtable | enum + match 派发 |
| 强制内联 | `#[inline(always)]` |
| 编译时单态化 | 泛型 `T: Float`（见 ARCH-API-001） |
| const 分支消除 | `T::TYPE_ID` 编译时确定 |

### 关联约束

- **ARCH-API-001**：统一泛型算子 API 规范（🚨 FROZEN - 零成本铁律）

### 状态（2025-01 更新）

**已完成**：
- ✅ GPU kernel wrappers（4 平台 × 8 算法）
- ✅ ARCH-API-001 零成本铁律定义
- ✅ Burn 依赖已完全移除
- ✅ KernelFloat trait 已实现（const TYPE_ID + bf16）

**当前实现**：
- ops/eagle3/ - EAGLE-3 模块（目录结构）
- ops/medusa/ - Medusa 模块（目录结构）
- ops/spec_ee/ - SpecEE 模块（目录结构）
- ops/softmax.rs - Log-space + Kahan 实现
- ops/paged_attn.rs - PagedAttention 纯 Rust 实现
- ops/int2_quantizer.rs, ops/evic_press.rs - 已迁移到纯 Rust

**待完成**：
- 🚨 flash_attention.rs - 未实现（当前通过 KernelDispatcher 调用 GPU kernel）
- 🚨 部分算子的 CPU fallback 优化

---

## 附录：Engram 条件记忆支持

DeepSeek Engram 实现规划，详见 `SPEC/DOCS/ENGRAM-DESIGN.md`。

| 组件 | 实现方式 | 位置 |
|------|----------|------|
| 核心模块 | 条件记忆管理 | `ops/engram.rs` |
| N-gram 哈希 | SIMD 优化哈希 | `ops/engram_hash.rs` |
| Embedding 查找 | 内存映射 + prefetch | `ops/engram_lookup.rs` |

> ⚠️ **当前状态**：Engram 使用纯 Rust + SIMD 实现，GPU kernel 尚未实现。
