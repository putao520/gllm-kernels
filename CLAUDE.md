# gllm-kernels

**极限性能 CPU 算子库** — 逼近硬件理论峰值的底层计算原语。

> **定位**：纯算子库（Operator Library），不含任何业务逻辑（无 Attention、无 KV Cache、无推理流程）。上层推理引擎通过组合这些算子构建完整推理管线。

---

## 优先级铁律（PRIORITY HIERARCHY）

| 优先级 | 原则 | 含义 |
|--------|------|------|
| **P0 🔴 逼近理论极限** | 每个算子必须达到硬件理论峰值的 85%+ | compute-bound 算子逼近 FLOPS 峰值；memory-bound 算子逼近带宽峰值 |
| **P1 🟡 JIT 编译器自动优化** | 所有算子通过编译器融合决策 + 程序化代码生成达到最优 | Phase 3 用 iced-x86/dynasm-rs 生成每条指令，自动 epilogue injection / loop fusion / tile-level fusion |
| **P2 🟢 代码量最少** | 编译器代码本身保持精简 | 宏/泛型复用编译器内部逻辑，避免重复代码 |
| **P3 ⚪ 可维护性** | 新增 ISA/量化格式/算子的变更路径清晰 | 遵循维护检查清单 |

> **核心判断准则**：所有性能优化通过 JIT 编译器实现。不存在"热路径/非热路径"区分 — 全部走 JIT 最优化生成。

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

| 算子 | 当前效率 | 目标 | 达成路径 |
|------|---------|------|---------|
| F32 GEMM | unpacked ~42%, prepacked ~59% (ASM 微内核路径) | 85%+ | JIT 编译器 Phase 3 自动生成最优代码 |
| 量化 GEMV | intrinsics 路径 | 85%+ | JIT 编译器 Phase 3 自动生成最优代码 |
| Softmax/RMSNorm/SiLU | ALU-limited 7-13 GiB/s | 90%+ 带宽 | JIT 编译器 Loop Fusion 消除中间 writeback |

> **性能调优路线**：全部算子通过 JIT 编译器 Phase 3 自动生成最优代码，不存在手动调优路径。

---

## SPEC 导航（Single Source of Truth）

| 文件 | 内容 |
|------|------|
| `SPEC/01-REQUIREMENTS.md` | 算子清单 + 性能需求 + 编译器需求 |
| `SPEC/02-ARCHITECTURE.md` | 核心架构：手写汇编 + 宏驱动 + 运行时分发 + §8 编译器架构 |
| `SPEC/03-DATA-STRUCTURE.md` | 数据结构 + 宏架构详细设计 |
| `SPEC/05-LAYER2-INFERENCE.md` | 推理后端 + §7 JIT 编译器流水线 |

---

## 🚨 算法意图编译器（ARCH-COMPILER）— 最易偏离的设计

> **核心原则：分析语义 → 决策融合 → 生成新代码。**
> **融合 = 全新代码生成。不是 trampoline 调度，不是模板拼接。**

### 三阶段编译流水线

```
CompilerGraph (from GLLM) + DeviceProfile
    │
    ▼
Phase 1: 语义 DAG 构筑
    · 算子 → 内置语义描述绑定（OpSemanticsKind）
    · 张量 def-use 链 + 后支配树
    · 算子分类: elemwise / injective / reduction / gemm / opaque
    │
    ▼
Phase 2: Profile-Driven 融合决策
    · 后支配树 + TVM 规则 → 融合组划分
    · Profile 约束检查（L1 容量、寄存器压力、消费者数）
    · 三种融合模式:
      - Epilogue Injection: GEMM 累加器写回前，在寄存器上原地执行 activation
      - Loop Fusion: elementwise 链 → 单循环，数据在寄存器中流过整个链
      - Tile-Level Fusion: 前驱 tile 计算嵌入 GEMM MC 循环，结果留在 L1
    │
    ▼
Phase 3: 全新代码生成（iced-x86 / dynasm-rs）
    · 程序化生成每一条指令（vfmadd231ps, vbroadcastss, ...）
    · GEMM: 完整 K-loop + FMA 序列 + epilogue 在累加器上原地执行 + store
    · Elementwise: 单循环体，数据在 ymm 寄存器中流过整个算子链
    · 输出: CompiledLayer (mmap RWX)
```

### 🚫 绝对禁止的实现模式

| 禁止模式 | 为什么错 | 正确做法 |
|----------|---------|---------|
| `mov rax, trampoline_addr; call rax` | 数据落地内存，融合收益为零 | iced-x86 程序化生成 FMA/activation 指令序列 |
| 预编译微内核变体（gemm_silu, gemm_gelu） | 组合爆炸，不可扩展 | Phase 3 根据融合决策动态生成 epilogue |
| EmitAction::CallGemm / CallElementwise | "调度器"不是"编译器" | MachineCodeEmitter trait 生成新代码 |
| 模板字节拼接（复制 body bytes） | 融合后算法结构变了，不能拼 | 根据算子数学语义程序化生成新循环 |

### 正确的 Phase 3 代码结构

```
GEMM + SiLU epilogue（JIT 生成，非模板）:
  prologue
  NC loop:
    pack_b
    MC loop:
      pack_a
      NR loop:
        微内核:
          vxorps ymm0..ymm11          // 累加器清零
          K-loop:                       // 程序化生成 FMA 序列
            vbroadcastss ymm12, [A]
            vmovups ymm14, [B]
            vfmadd231ps ymm0, ymm12, ymm14
            ...
          // ★ epilogue 在 store 前执行，数据不落地
          SiLU on ymm0..ymm11 (用 ymm12-14 做 scratch)
          vmovups [C], ymm0..ymm11     // 一次 store
  epilogue
```

### 关键 trait 架构（SPEC §8.6）

```rust
trait PlatformBackend {
    type Emitter: MachineCodeEmitter;
    fn new_emitter(&self) -> Self::Emitter;
}

trait MachineCodeEmitter {
    fn emit_gemm_unit(&mut self, unit: &GemmUnit) -> Result<Vec<u8>>;
    fn emit_fused_loop(&mut self, unit: &FusedLoop) -> Result<Vec<u8>>;
    fn emit_activation(&mut self, kind: ActivationKind, reg: Register) -> Result<()>;
    fn finalize(self) -> Result<Vec<u8>>;
}
```

### 当前状态

Phase 1（graph.rs, semantics.rs）和 Phase 2（fusion.rs）的基础已实现。
Phase 3 当前是 stub — 等待按上述设计正确实现。
详见 `SPEC/02-ARCHITECTURE.md` §8 和 `SPEC/01-REQUIREMENTS.md` §6。

---

## Technology Stack

| Component | Technology | Constraint |
|-----------|------------|------------|
| **Language** | Rust nightly (1.93.0+) | `global_asm!`, `naked_fn`, `target_feature` |
| **JIT 编译器 (主路径)** | iced-x86 (x86_64) / dynasm-rs (aarch64) | 程序化生成每条指令，全部算子 JIT 最优化 |
| **Layer 1 算子库** | `global_asm!` 微内核 + intrinsics + 宏生成 | 正确性参考 + 编译器测试基准 |
| **分发** | `cargo install` 一键安装 | 零外部依赖，纯 Rust crate |

---

## Layer 1 算子库（ARCH-ASM-MICROKERNEL）

> **定位**：JIT 编译器的正确性参考基准 + 测试 oracle。
> 所有算子的生产路径走 JIT 编译器 Phase 3 自动生成。

### 现有微内核规格（正确性参考）

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
| **BLAS-2** | gemv, streaming GEMV (M=1 路径) | Memory-bound |
| **BLAS-3** | gemm, gemm_bias, gemm_prepacked, pack_b, gemm_bt (B-transposed skinny GEMM) | Compute-bound |
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

> Layer 1 算子库的内部代码组织。宏批量生成基线实现，手写 asm 作为正确性参考。

```
Layer 1: simd_primitive!     — 硬件原语映射表（每 ISA × 精度 22 个操作）
            ↓ 被调用
Layer 2: define_xxx!         — 算子逻辑模板（基线实现）
            ↓ 被调用
Layer 3: quant_primitive!    — 量化特化原语（位解包/码本查表）
            ↓ 被调用
Layer 4: expand_all_xxx!     — 批量展开

正确性参考实现：
  gemm_avx2_asm()     — 手写汇编 GEMM（JIT 编译器测试 oracle）
  gemv_q4_avx2_asm()  — 手写汇编量化 GEMV（JIT 编译器测试 oracle）
```

### 路径选择（Layer 1 算子库内部）

- M=1 走 streaming GEMV 路径
- M≤32 走 skinny GEMM intrinsics 路径
- M>32 走 ASM 微内核路径

> **注意**：以上路径选择仅描述 Layer 1 算子库的内部逻辑。生产路径全部由 JIT 编译器 Phase 3 自动生成最优代码。

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
├── asm/                    # 手写汇编微内核
│   ├── x86_64/
│   │   ├── gemm_avx2.rs    # AVX2 GEMM 6×16 微内核 (global_asm!)
│   │   ├── gemm_avx512.rs  # AVX-512 GEMM 14×32 微内核 (global_asm!)
│   │   ├── gemm_driver.rs  # 缓存分块驱动 (pack_b + MC/KC/NC blocking)
│   │   └── quant_gemv.rs   # 量化 GEMV
│   └── aarch64/
│       └── gemm_neon.rs    # NEON GEMM 8×12 微内核 (global_asm!)
│
└── compiler/               # 算法意图编译器（JIT）
    ├── mod.rs              # InferenceCompiler 入口
    ├── graph.rs            # Phase 1: CompilerGraph DAG ✅
    ├── semantics.rs        # Phase 1: 算子语义分析 ✅
    ├── fusion.rs           # Phase 2: 融合决策（需增强）
    ├── planner.rs          # Phase 2: ExecutionPlan（需增强）
    ├── executable.rs       # CompiledLayer mmap RWX ✅
    ├── cache.rs            # 编译缓存 ✅
    ├── ir.rs               # LayerIR 中间表示 ✅
    └── codegen/            # Phase 3: 代码生成（当前 stub，待实现）
        ├── emitter.rs      # ScratchpadLayout + buffer 规划
        ├── x86_64.rs       # iced-x86 后端（待实现）
        └── aarch64.rs      # dynasm-rs 后端（待实现）
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
