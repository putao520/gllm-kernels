# gllm-kernels

**极限性能 CPU 算子库** — 逼近硬件理论峰值的底层计算原语。

> **定位**：纯算子库（Operator Library），不含任何业务逻辑（无 Attention、无 KV Cache、无推理流程）。上层推理引擎通过组合这些算子构建完整推理管线。

---

## 优先级铁律（PRIORITY HIERARCHY）

| 优先级 | 原则 | 含义 |
|--------|------|------|
| **P0 🔴 逼近理论极限** | 每个算子必须达到硬件理论峰值的 85%+ | compute-bound 算子逼近 FLOPS 峰值；memory-bound 算子逼近带宽峰值 |
| **P1 🟡 手写汇编微内核** | 核心热路径必须使用 `global_asm!` / `naked_fn` 手写汇编 | 精确控制寄存器分配、指令调度、软件流水线，不依赖编译器 |
| **P2 🟢 代码量最少** | 宏驱动批量生成非热路径代码 | 在 P0/P1 不受损的前提下，通过宏模板最大化代码复用 |
| **P3 ⚪ 可维护性** | 新增 ISA/量化格式/算子的变更路径清晰 | 遵循维护检查清单 |

> **核心判断准则**：当任何因素与性能冲突时，**永远选择性能**。手写汇编优先于宏生成，宏生成优先于泛型抽象。

---

## 性能目标（PERF-TARGET）🚨 铁律

### 理论峰值计算方法

| 瓶颈类型 | 理论峰值公式 | 目标效率 |
|----------|-------------|----------|
| **Compute-bound** (GEMM) | `核心数 × 频率 × FMA吞吐 × SIMD宽度 × 2` | **≥ 85%** |
| **Memory-bound** (GEMV, 激活, 归一化) | `内存带宽 / (输入+输出字节数)` | **≥ 90%** |
| **量化 GEMV** | `min(计算峰值, 带宽/量化字节数)` | **≥ 85%** |

### 参考对标

| 库 | 典型效率 | 我们的目标 |
|---|---|---|
| Intel MKL (GEMM) | 85-95% | **≥ 85%** |
| OpenBLAS (GEMM) | 70-85% | 超越 |
| llama.cpp (量化 GEMV) | 60-75% | **≥ 85%** |

### 当前状态

| 算子 | 当前效率 | 目标 | 手段 |
|------|---------|------|------|
| F32 GEMM (intrinsics) | ~41% | 85%+ | 手写汇编微内核 |
| 量化 GEMV | 待测 | 85%+ | 手写汇编微内核 |
| Softmax/RMSNorm/SiLU | 待测 | 90%+ 带宽 | 验证是否已达带宽瓶颈 |

---

## SPEC 导航（Single Source of Truth）

| 文件 | 内容 |
|------|------|
| `SPEC/01-REQUIREMENTS.md` | 算子清单 + 性能需求 |
| `SPEC/02-ARCHITECTURE.md` | 核心架构：手写汇编 + 宏驱动 + 运行时分发 |
| `SPEC/03-DATA-STRUCTURE.md` | 数据结构 + 宏架构详细设计 |

---

## Technology Stack

| Component | Technology | Constraint |
|-----------|------------|------------|
| **Language** | Rust nightly (1.93.0+) | `global_asm!`, `naked_fn`, `target_feature` |
| **CPU Kernels** | 自研手写汇编 + intrinsics | **禁止外部 BLAS 依赖** |
| **汇编微内核** | `global_asm!` / `core::arch::asm!` | 核心热路径必须手写 |
| **非热路径** | Rust intrinsics + 宏生成 | 宏驱动批量展开 |
| **分发** | `cargo install` 一键安装 | 零外部依赖，纯 Rust crate |

---

## 🚨 手写汇编微内核架构（ARCH-ASM-MICROKERNEL）

### 为什么必须手写汇编

Rust intrinsics 经过编译器后无法保证：
1. **寄存器分配最优** — 编译器可能 spill 关键累加器到栈
2. **指令调度最优** — FMA/load/store 的交错顺序影响流水线利用率
3. **软件流水线** — 手动安排 load(k+1) 与 compute(k) 重叠

### 微内核规格

| ISA | 微内核尺寸 | 累加器 | 临时寄存器 | 实现方式 |
|-----|-----------|--------|-----------|---------|
| **AVX2** | 6×16 (6M × 2×ymm) | 12 ymm | 4 ymm | `global_asm!` |
| **AVX-512** | 14×32 (14M × 2×zmm) | 28 zmm | 4 zmm | `global_asm!` |
| **NEON** | 8×12 (8M × 3×v) | 24 v | 8 v | `global_asm!` |

### 运行时 CPUID 分发

```rust
// 启动时一次检测，之后零开销
static ISA: OnceLock<IsaLevel> = OnceLock::new();

fn gemm(a, b, c, m, n, k) {
    match *ISA.get().unwrap() {
        IsaLevel::Avx512 => gemm_avx512_asm(a, b, c, m, n, k),
        IsaLevel::Avx2   => gemm_avx2_asm(a, b, c, m, n, k),
        IsaLevel::Neon   => gemm_neon_asm(a, b, c, m, n, k),
        IsaLevel::Scalar => gemm_scalar(a, b, c, m, n, k),
    }
}
```

---

## 🚨 算子边界定义（ARCH-SCOPE）

### 属于本库的算子（纯计算原语）

| 类别 | 算子 | 瓶颈类型 |
|------|------|---------|
| **BLAS-1** | vec_dot, vec_add, vec_mul, vec_scale, vec_axpy, vec_sum, vec_max | Memory-bound |
| **BLAS-2** | gemv | Memory-bound |
| **BLAS-3** | gemm, gemm_bias, gemm_prepacked, pack_b | Compute-bound |
| **激活函数** | silu, gelu, relu, tanh, swiglu, softmax, exp | Memory-bound |
| **归一化** | rms_norm, layer_norm | Memory-bound |
| **位置编码** | rope | Memory-bound |
| **量化解码** | dequant_* (18 种格式) | Memory-bound |
| **量化 GEMV/GEMM** | gemv_q4, gemv_q8, gemm_q4, gemm_q8, kquant_matmul, iq_matmul 等 | 带宽/计算混合 |

### 不属于本库的（上层业务）

- ❌ FlashAttention / Paged Attention
- ❌ KV Cache 管理
- ❌ 融合算子（fused_qkv_rope, fused_ffn 等）
- ❌ Embedding lookup
- ❌ Sampling (argmax, top-k, top-p)
- ❌ CUDA/GPU 后端
- ❌ 推理调度、批处理

---

## 🚨 四层宏架构（ARCH-MACRO-LAYERS）

> 非热路径代码通过宏批量生成，热路径手写汇编覆写。

```
Layer 1: simd_primitive!     — 硬件原语映射表（每 ISA × 精度 22 个操作）
            ↓ 被调用
Layer 2: define_xxx!         — 算子逻辑模板（基线实现）
            ↓ 被调用
Layer 3: quant_primitive!    — 量化特化原语（位解包/码本查表）
            ↓ 被调用
Layer 4: expand_all_xxx!     — 批量展开

热路径覆写：
  gemm_avx2_asm()     — 手写汇编，替代宏生成的 GEMM
  gemv_q4_avx2_asm()  — 手写汇编，替代宏生成的量化 GEMV
```

### 覆写规则

- 手写汇编微内核**必须**用于：GEMM、量化 GEMV/GEMM
- 宏生成的基线实现作为**正确性参考**和**非热路径兜底**
- 覆写必须通过 benchmark 证明优于宏生成版本

---

## 🚨 三层零成本分发架构（ARCH-DISPATCH）

```
Layer 1: Backend    → CpuBackend（本库唯一后端）
Layer 2: ISA        → 启动时一次检测（Scalar/AVX2/AVX-512/NEON）— OnceLock
Layer 3: Precision  → 编译时泛型单态化（<E: Element>）— 零开销
```

---

## 🚨 ISA 差异性原则（ARCH-ISA-PERF）

> 不同 ISA 的最优算法**结构不同**，不仅仅是"换指令"。

| 差异维度 | AVX2 (16×256b) | AVX-512 (32×512b) | NEON (32×128b) |
|----------|----------------|-------------------|----------------|
| **GEMM 微内核** | 6×16 手写 asm | 14×32 手写 asm | 8×12 手写 asm |
| **水平求和** | 手动 shuffle 4 步 | 原生 `reduce_add` | 原生 `vaddvq` |
| **f16 计算** | F16C 转换→f32 FMA | AVX512-FP16 原生 FMA | NEON FP16 原生 FMA |
| **INT8 点积** | 无原生支持 | VNNI `vpdpbusd` | `sdot` |

---

## Directory Structure

```
src/
├── lib.rs                  # Crate 入口
├── traits.rs               # Element/Backend/Kernels trait
├── quant.rs                # QuantType 枚举 + 块常量
├── codebooks.rs            # IQ 量化码本常量
│
├── macros/                 # 宏架构
│   ├── simd_primitive.rs   # Layer 1: ISA 原语映射表
│   ├── operator_templates.rs # Layer 2: 算子逻辑模板（基线）
│   ├── quant_primitive/    # Layer 3: 量化特化原语
│   └── expand.rs           # Layer 4: 批量展开
│
├── cpu_kernels/            # CPU 后端
│   ├── mod.rs              # ISA 检测 + 分发
│   ├── scalar/             # Scalar 兜底
│   ├── avx2/               # AVX2（含手写 asm 微内核）
│   ├── avx512/             # AVX-512（含手写 asm 微内核）
│   └── neon/               # NEON（含手写 asm 微内核）
│
└── asm/                    # 手写汇编微内核（新增）
    ├── x86_64/
    │   ├── gemm_avx2.S     # AVX2 GEMM 6×16 微内核
    │   ├── gemm_avx512.S   # AVX-512 GEMM 14×32 微内核
    │   ├── gemv_q4_avx2.S  # AVX2 Q4 GEMV 微内核
    │   └── gemv_q8_avx2.S  # AVX2 Q8 GEMV 微内核
    └── aarch64/
        ├── gemm_neon.S     # NEON GEMM 8×12 微内核
        └── gemv_q4_neon.S  # NEON Q4 GEMV 微内核
```

---

## Common Commands

```bash
cargo test --lib                      # 运行测试
cargo bench --bench gemm_benchmark    # GEMM 基准测试
cargo bench --bench kernels_benchmark # 全算子基准测试
RUSTFLAGS="-C target-cpu=native" cargo bench  # 启用本机 ISA
```

## Cargo Profile

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```
