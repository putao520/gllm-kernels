# gllm-kernels

**High-Performance Compute Backend** — The computational engine for `gllm`.

> **🚨 TABULA RASA (2026-02)**: This project has been reset. All legacy code has been removed to enforce strict architectural compliance.

---

## 优先级铁律（PRIORITY HIERARCHY）

| 优先级 | 原则 | 含义 |
|--------|------|------|
| **P0 🔴 性能最大化** | 每条指令、每个寄存器、每行 Cache 都不可浪费 | 不同 ISA 必须有结构不同的最优微内核；禁止"一份通用代码适配所有硬件" |
| **P1 🟡 代码量最少** | 宏驱动批量生成，杜绝手写 1700+ 函数 | 在 P0 不受损的前提下，通过宏模板 + 批量展开最大化代码复用 |
| **P2 🟢 可维护性** | 新增 ISA/量化格式/算子的变更路径清晰 | 遵循 `SPEC/03 §8.9` 检查清单 |
| **P3 ⚪ 编译速度** | 可接受较长编译时间 | 使用 Fat LTO + codegen-units=1 追求极致运行性能 |

> **核心判断准则**：当代码简洁性与性能冲突时，**永远选择性能**。宏模板的存在是为了避免手写重复代码，而不是为了统一不同硬件的算法逻辑。

---

## SPEC 导航（Single Source of Truth）

| 文件 | 内容 | 核心章节 |
|------|------|----------|
| `SPEC/01-REQUIREMENTS.md` | 功能需求清单 | REQ-ARCH / REQ-BACKEND / REQ-QUANT / REQ-OPS |
| `SPEC/02-ARCHITECTURE.md` | 核心架构设计 | ARCH-GPU-PURE / ARCH-GENERIC-CORE / ARCH-CPU-SIMD |
| `SPEC/03-DATA-STRUCTURE.md` | **算子全清单 + 宏驱动分发架构** | **§1 三层分发** / **§8 宏架构** / **§8.9 维护清单** |
| `SPEC/DOCS/quantization/` | 量化内核详细设计 | 模板化 CUDA / 泛型 Rust 量化 |

---

## Technology Stack (Strict)

| Component | Technology | Constraint |
|-----------|------------|------------|
| **Language** | Rust (1.93.0+) | **Pure Rust Only** (No C/C++ build scripts) |
| **GPU API** | CUDA Driver API | Via `cudarc` (No Runtime API `libcudart.so`) |
| **CPU Kernels** | 自研泛型实现 | **禁止外部 BLAS 依赖**（无 faer/OpenBLAS/MKL） |
| **Kernel Dist** | AOT Binary | Embed `.cubin` (sm_80/86/89/90/90a/100). **No PTX/JIT**. |

---

## 🚨 元编程/宏编程核心机制（FROZEN — ARCH-METAPROGRAMMING）

> **📌 权威设计**：`SPEC/03-DATA-STRUCTURE.md` §8

### 为什么必须用宏

算子组合矩阵：**71 算子模板 × 8 ISA × 3 精度 + 18 量化格式 ≈ 1,824 个函数实例**。
手写不可能，Trait 泛型无法表达"算法随硬件变化"的需求（见下文 §ISA 差异），因此必须用宏。

### 四层宏架构 (ARCH-MACRO-LAYERS)

```
Layer 1: simd_primitive!     — 硬件原语映射表（每 ISA × 精度 22 个操作）
            ↓ 被调用
Layer 2: define_xxx!         — 算子逻辑模板（可引用 ISA 常量调整分块/展开策略）
            ↓ 被调用
Layer 3: quant_primitive!    — 量化特化原语（位解包/码本查表/On-the-fly 解量化）
         decode_block!       — 块解码宏（每种量化格式的解码逻辑）
            ↓ 被调用
Layer 4: expand_all_xxx!     — 批量展开（一次性生成全量 ISA × 精度 × 量化格式实例）
```

**层级调用规则**：
- ✅ 上层可以调用下层
- ❌ 禁止跨层调用（如 Layer 4 直接调用 Layer 1）
- ❌ 禁止在算子模板中直接使用裸 Intrinsic（必须通过 `simd_primitive!`）

### 算子分类分发 (MACRO-DISPATCH)

> **📌 详细分类逻辑**：`SPEC/03 §8.2`

| 分类 | 权重参数 | 输出类型 | 展开维度 | 示例 |
|------|----------|----------|----------|------|
| **表 A** (纯浮点) | 无 或 `&[E]` | `E` / `&mut [E]` | ISA × 精度 | silu, gemm, flash_attention |
| **表 B** (解量化) | `&[u8]` | 固定 `&mut [f32]` | ISA | dequant_q4_k |
| **表 C** (量化计算) | `&[u8]` / `&[i8]` | `E` / `&mut [E]` | ISA × 精度 × 格式 | gemv_q4, awq_matmul |
| **表 D** (量化融合) | `&[u8]` | `&mut [E]` | ISA × 精度 × 格式 | fused_ffn_q4 |

### 禁止项

- ❌ 禁止在热路径 (Hot Path) 使用 `match quant_type` 或 `TypeId` 做运行时分发
- ❌ 禁止使用 `dyn Trait` 动态分发（ISA 入口处的 `OnceLock` 分发除外）
- ❌ 禁止脱离宏体系手写单体算子（极端热点手写覆写除外，见 §8.7.4）

---

## 🚨 ISA 差异性与性能最大化原则（FROZEN — ARCH-ISA-PERF）

> **核心立场**：不同 ISA 的最优算法**结构不同**，不仅仅是"换指令"。

### 为什么 Trait 泛型不能替代宏

| 差异维度 | AVX2 (16×256b) | AVX-512 (32×512b) | NEON (32×128b) |
|----------|----------------|-------------------|----------------|
| **GEMM 最优微内核** | 6×16 (12 累加器) | 14×32 (28 累加器) | 8×12 (24 累加器) |
| **水平求和** | 手动 shuffle 4 步 | 原生 `reduce_add` | 原生 `vaddvq` |
| **f16 计算** | F16C 转换→f32 FMA | AVX512-FP16 **原生 FMA** | NEON FP16 **原生 FMA** |
| **INT8 点积** | 无原生支持 | VNNI `vpdpbusd` | `sdot` |
| **bf16 点积** | 位转换→f32 | `dpbf16_ps` 原生 | 位转换→f32 |
| **预取距离** | 256B ahead | 512B ahead | 128B ahead |

**关键洞察**：
- `fn gemm_impl<S: SimdOps>()` 一个泛型函数**无法同时**对 AVX2 用 6×16 微内核、对 AVX-512 用 14×32 微内核。
- f16 在 AVX-512 FP16 扩展上可以完全跳过 f32 转换，这是**算法路径不同**，不是参数不同。
- 宏 `define_gemm!(avx512, f32)` 可以展开为与 `define_gemm!(avx2, f32)` **结构完全不同**的代码。

### 宏模板中的 ISA 感知机制

```rust
macro_rules! define_gemm {
    ($isa:ident, $elem:ty) => {
        // 通过 simd_primitive! 获取 ISA 特定常量
        const LANES: usize = simd_primitive!($isa, $elem, lanes);
        const NUM_REGS: usize = simd_primitive!($isa, $elem, num_regs);

        // 分块因子根据 ISA 寄存器数量和 SIMD 宽度自动调整
        const TILE_M: usize = NUM_REGS / 2;     // 一半寄存器做累加器
        const TILE_N: usize = LANES * 2;         // 2 个 SIMD 向量宽

        #[inline(always)]
        pub fn gemm(/* ... */) {
            // 循环结构由 TILE_M/TILE_N 决定
            // 编译器在常量折叠后完全展开内层循环
        }
    };
}
```

### 极端热点手写覆写规则

对于 GEMM、FlashAttention 等核心热点，允许在宏生成的基线之上手写覆写：

```rust
mod avx512_f32 {
    define_gemm!(avx512, f32);  // 宏生成的基线

    // 手写覆写（更激进的寄存器阻塞 + 预取）
    // 仅在基准测试证明手写比宏生成快 >10% 时允许
    #[inline(always)]
    pub fn gemm_optimized(/* ... */) { /* ... */ }
}
```

> **📌 覆写规则**：`SPEC/03 §8.7.4`

---

## 🚨 三层零成本分发架构（FROZEN — ARCH-DISPATCH）

> **📌 权威设计**：`SPEC/03 §1`

```
Layer 1: Backend    → 用户指定（CpuBackend / CudaBackend）     — 编译时泛型，零开销
Layer 2: ISA        → 启动时一次检测（Scalar/AVX2/AVX-512/NEON）— OnceLock，一次性
Layer 3: Precision  → 编译时泛型单态化（<E: Element>）          — 零开销
```

ISA 检测只在程序启动时发生一次，之后整棵算子树都是静态确定的。

---

## 🚨 CPU 内核自研架构（FROZEN — ARCH-CPU-SELF-IMPL）

### 核心原则

**自研优于依赖**：CPU 内核是 gllm-kernels 的核心职责，必须自己实现。

### 禁止的外部依赖

```rust
// ❌ 禁止：任何外部 BLAS/数学库
use faer::matmul;      // 禁止
use openblas::*;        // 禁止
use mkl::*;             // 禁止
use ndarray::linalg::*; // 禁止
```

### 性能优化要求

| 优化技术 | 适用算子 | 说明 |
|----------|----------|------|
| **寄存器阻塞** | GEMM, GEMV | 微内核尺寸适配 ISA 寄存器文件 |
| **Cache 分块** | GEMM, Flash Attention | L1/L2/L3 分级分块 |
| **SIMD 运行时检测** | 全部 | `OnceLock` + `is_x86_feature_detected!` |
| **软件预取** | GEMM, 量化 GEMV | `_mm_prefetch` / `__builtin_prefetch` |
| **数值稳定** | Softmax, RMSNorm | Online 最大值跟踪，避免 overflow |
| **On-the-fly 解量化** | 量化 GEMV/GEMM | 寄存器内解包→FMA，不生成中间 f32 矩阵 |

---

## 🚨 Backend Trait 泛型设计（FROZEN — ARCH-GENERIC-CORE）

> **📌 权威设计**：`SPEC/02-ARCHITECTURE.md` §0

### Element Trait（blanket implementation）

```rust
pub trait Element: Copy + Send + Sync + Default + 'static {
    const ZERO: Self;
    const ONE: Self;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn mul_add(self, a: Self, b: Self) -> Self;
    // ... 完整定义见 SPEC/03 §2.1
}
```

### Backend + Kernels Trait

```rust
pub trait Backend: Send + Sync + 'static {
    type Kernels<E: Element>: Kernels<E>;
    fn init<E: Element>() -> Self::Kernels<E>;
}

pub trait Kernels<E: Element>: Send + Sync {
    // 71 个算子签名 — 见 SPEC/03 §2.3
}
```

### 禁止的实现方式

```rust
// ❌ 为每个精度分别实现
impl Backend<f32> for CpuBackend { ... }
impl Backend<f16> for CpuBackend { ... }

// ❌ 手动列举类型
impl Element for f32 { ... }

// ❌ 运行时类型枚举分发
match dtype { DType::F32 => ..., DType::F16 => ... }
```

---

## Core Architecture (FROZEN)

### 1. L3 GPU-Pure Architecture (ARCH-GPU-PURE)

> **📌 详细设计**：`SPEC/02` §1

- **Weights**: Uploaded once to GPU memory
- **KV Cache**: Permanently resident on GPU
- **Logits**: Generated and sampled on GPU
- **Data Transfer**: Only 8 bytes/step (TokenID in → TokenID out)
- **Violation**: Any `Vec<f32>` transfer during generation loop is a critical bug

### 2. Quantization Kernel Template (ARCH-QUANT-TEMPLATE)

> **📌 详细设计**：`SPEC/DOCS/quantization/`

- CUDA: C++ `template<int BITS>` 统一实现，编译时实例化
- CPU: 宏 `define_quant_gemv!($isa, $elem, $quant_fmt, $block_size)` 批量展开
- **Violation**: 为每种位宽单独编写内核

### 3. Fused-First Architecture (ARCH-FUSED-FIRST)

- 调度层**优先选择融合算子**，仅在无法匹配融合模式时降级使用原子算子
- ONNX Loader 必须实现 Graph Pattern Matching，严禁 naive 1:1 翻译

### 4. Build & Distribution

- **No `build.rs` compilation**: No `cc` crate, no `nvcc`
- **Pre-compiled Kernels**: `.cubin` checked into repo (`src/cuda_kernels/kernels/`)
- **Embed**: `include_bytes!("kernels/kernels_smXX.cubin")`

---

## Directory Structure

```
src/
├── lib.rs                  # Crate 入口
├── element.rs              # Element trait 定义
├── backend.rs              # Backend/Kernels trait + auto_select_backend()
├── quant_types.rs           # QuantType 枚举 + 块常量
│
├── macros/                 # 🚨 宏架构核心
│   ├── mod.rs
│   ├── simd_primitive.rs   # Layer 1: ISA 原语映射表
│   ├── operator_templates.rs # Layer 2: 算子逻辑模板
│   ├── quant_primitive.rs  # Layer 3: 量化特化原语
│   └── expand.rs           # Layer 4: 批量展开
│
├── cpu_kernels/            # CPU 后端实现
│   ├── mod.rs              # CpuKernels 结构 + ISA 检测
│   ├── scalar/             # Scalar 回退实现
│   ├── avx2/               # AVX2 优化实现
│   ├── avx512/             # AVX-512 优化实现
│   └── neon/               # NEON 优化实现
│
├── cuda_kernels/           # CUDA 后端实现
│   ├── mod.rs              # CudaKernels 结构 + CUBIN 加载
│   └── kernels/            # *.cubin 文件 (Git tracked)
│
└── codebooks.rs            # IQ 量化码本常量
```

---

## Common Commands

```bash
cargo check                           # 类型检查
cargo test                            # 运行测试
cargo bench                           # 性能基准测试
RUSTFLAGS="-C target-cpu=native" cargo bench  # 启用本机 ISA
```

## Cargo Profile (Release)

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```
