# gllm-kernels 架构设计

## 定位

**gllm-kernels = 极限性能算子库 + JIT 编译器**

提供逼近硬件理论峰值的底层计算原语，以及算法意图编译器（JIT）自动融合优化。上层推理引擎（gllm）通过组合这些算子构建完整推理管线。

当前聚焦 CPU 后端，GPU 后端为规划中的未来工作（见 SPEC/04-GPU-BACKEND.md）。

---

## 核心架构原则

### 1. 手写汇编优先（ARCH-ASM-FIRST）🚨 铁律

核心热路径**必须**使用手写汇编微内核，不依赖编译器的寄存器分配和指令调度。

**必须手写汇编的算子**：
- F32/F16/BF16 GEMM 微内核
- 量化 GEMV/GEMM 微内核（Q4_K, Q8_K 等）

**可以用 intrinsics 的算子**（memory-bound，瓶颈在带宽不在计算）：
- BLAS-1（vec_dot, vec_add 等）
- 激活函数（silu, gelu, softmax 等）
- 归一化（rms_norm, layer_norm）
- 位置编码（rope）
- 量化解码（dequant_*）

**判断标准**：如果算子是 compute-bound 且 intrinsics 版本达不到 85% 理论峰值，就必须手写汇编。

### 2. 逼近理论极限（ARCH-PEAK-PERF）🚨 铁律

| 瓶颈类型 | 目标 | 验证方法 |
|----------|------|---------|
| Compute-bound | ≥ 85% FLOPS 峰值 | `实测 GFLOPS / 理论峰值 GFLOPS` |
| Memory-bound | ≥ 90% 带宽峰值 | `实测 GB/s / STREAM 带宽` |

理论峰值计算（以 AVX2 FMA 为例）：
```
单核: 频率 × 2(FMA ports) × 8(f32/ymm) × 2(mul+add) = 频率 × 32 FLOP/cycle
全核: 单核 × 核心数 × 全核 Turbo 频率
```

### 3. CPU 优先（ARCH-CPU-FIRST）

当前聚焦 CPU 后端。GPU 后端（CUDA/Metal）为规划中的未来工作（见 SPEC/04-GPU-BACKEND.md）。

**禁止的外部依赖**：
```rust
// ❌ 禁止
use faer::*;        // 外部 BLAS
use openblas::*;    // 外部 BLAS
use mkl::*;         // 外部 BLAS
use cudarc::*;      // GPU
```

### 4. 算子边界（ARCH-SCOPE）

**属于本库**：纯计算原语（BLAS、激活、归一化、位置编码、量化解码、量化 GEMV/GEMM）

**不属于 Layer 1 算子库**：
- FlashAttention / Paged Attention（业务算法）
- KV Cache 管理（业务状态）
- 融合算子 fused_qkv_rope / fused_ffn 等（业务组合）
- Embedding lookup（查表，非计算密集）
- Sampling（业务逻辑）

**规划中的未来工作**：
- GPU 后端（CUDA/Metal）— 见 SPEC/04-GPU-BACKEND.md
- Layer 2 推理后端 — 见 SPEC/05-LAYER2-INFERENCE.md

---

## 手写汇编微内核架构（ARCH-ASM-MICROKERNEL）

### 为什么 intrinsics 不够

| 问题 | 说明 | 影响 |
|------|------|------|
| 寄存器 spill | 编译器可能将累加器 spill 到栈 | 额外 load/store，降低 IPC |
| 指令调度 | 编译器不一定交错 FMA 和 load | 流水线气泡 |
| 软件流水线 | 无法手动安排 load(k+1) 与 compute(k) 重叠 | 访存延迟暴露 |
| 寄存器分配 | 无法指定哪个 ymm/zmm 做累加器 | 可能用到高编号寄存器导致 VEX→EVEX 切换 |

### 微内核设计

GEMM 的核心是一个 MR×NR 的微内核，在 K 维度上循环累加：

```
for k in 0..KC:
    load A panel column (MR elements)
    load B panel row (NR elements)
    outer product: C[MR×NR] += A[MR] × B[NR]
```

**各 ISA 微内核规格**：

| ISA | MR×NR | 累加器寄存器 | 临时寄存器 | 总寄存器 |
|-----|-------|------------|-----------|---------|
| AVX2 | 6×16 | 12 ymm (6×2) | 4 ymm | 16 ymm |
| AVX-512 | 14×32 | 28 zmm (14×2) | 4 zmm | 32 zmm |
| NEON | 8×12 | 24 v (8×3) | 8 v | 32 v |

### 汇编微内核接口约定

```rust
// Rust 侧声明
extern "C" {
    /// AVX2 GEMM 6×16 微内核
    /// 在 K 维度上循环，累加 C[6×16] += A[6×KC] × B[KC×16]
    fn gemm_microkernel_avx2_6x16(
        kc: usize,           // K 维度循环次数
        a: *const f32,        // A panel, MR×KC, 列主序
        b: *const f32,        // B panel, KC×NR, 行主序（已 pack）
        c: *mut f32,          // C tile, MR×NR
        c_stride: usize,      // C 行步长（字节）
    );
}

// 汇编侧通过 global_asm! 实现
global_asm!(include_str!("asm/x86_64/gemm_avx2.S"));
```

### GEMM 分层结构

```
gemm(A, B, C, M, N, K)
│
├── L3 分块: NC×KC 块（适配 L3 Cache）
│   ├── pack_b: B[KC×NC] → packed_b[KC×NC]（连续内存）
│   │
│   ├── L2 分块: MC×KC 块（适配 L2 Cache）
│   │   ├── pack_a: A[MC×KC] → packed_a[MC×KC]
│   │   │
│   │   └── L1 分块: MR×NR 微内核（适配 L1 Cache + 寄存器）
│   │       └── gemm_microkernel_asm(KC, packed_a, packed_b, C)
│   │           └── 手写汇编：精确控制寄存器、指令调度、预取
```

**分块参数**：

| 参数 | 含义 | AVX2 典型值 | 适配 |
|------|------|-----------|------|
| MR | 微内核行数 | 6 | 寄存器文件 |
| NR | 微内核列数 | 16 (2×ymm) | 寄存器文件 |
| MC | L2 分块行数 | 72-144 | L2 Cache |
| KC | L2 分块 K 维 | 256-512 | L1 Cache |
| NC | L3 分块列数 | 4096+ | L3 Cache |

---

## 量化微内核架构（ARCH-QUANT-MICROKERNEL）

量化 GEMV/GEMM 的核心是 on-the-fly dequantization + FMA：

```
for each block (256 elements):
    SIMD load packed weights (u8/u4)
    SIMD unpack to int8/int16
    SIMD convert to f32
    SIMD scale by block scale factor
    SIMD FMA with input activation
```

**关键优化**：
1. 不生成完整 f32 矩阵（on-the-fly）
2. 块级解码，L1 Cache 友好
3. 输入向量 SIMD 广播复用
4. 手写汇编精确控制解包+FMA 交错

---

## 三层零成本分发（ARCH-DISPATCH）

```
Layer 1: Backend    → CpuBackend（唯一后端）         — 编译时确定
Layer 2: ISA        → Scalar/AVX2/AVX-512/NEON       — 启动时一次检测（OnceLock）
Layer 3: Precision  → f32/f16/bf16                    — 编译时泛型单态化
```

```rust
pub struct CpuKernels;

impl CpuKernels {
    pub fn gemm<E: Element>(a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        match get_isa_level() {
            IsaLevel::Avx512 => avx512::gemm::<E>(a, b, c, m, n, k),
            IsaLevel::Avx2   => avx2::gemm::<E>(a, b, c, m, n, k),
            IsaLevel::Neon   => neon::gemm::<E>(a, b, c, m, n, k),
            IsaLevel::Scalar => scalar::gemm::<E>(a, b, c, m, n, k),
        }
    }
}

fn get_isa_level() -> IsaLevel {
    static LEVEL: OnceLock<IsaLevel> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") { return IsaLevel::Avx512; }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return IsaLevel::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        { return IsaLevel::Neon; }
        IsaLevel::Scalar
    })
}
```

---

## 泛型精度架构（ARCH-GENERIC）

### Element Trait

```rust
pub trait Element: Copy + Send + Sync + Default + 'static {
    const ZERO: Self;
    const ONE: Self;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn recip(self) -> Self;
}
```

### 精度处理策略

| 精度 | AVX2 | AVX-512 | NEON |
|------|------|---------|------|
| f32 | 原生 FMA | 原生 FMA | 原生 FMA |
| f16 | F16C load→f32 FMA→F16C store | AVX512-FP16 原生 或 转换 | NEON FP16 原生 |
| bf16 | 位移 load→f32 FMA→位移 store | AVX512-BF16 `dpbf16_ps` | 位移 load→f32 FMA→位移 store |

---

## 四层宏架构（ARCH-MACRO）

> 非热路径代码通过宏批量生成。热路径手写汇编覆写。

```
Layer 1: simd_primitive!     — 硬件原语映射表
Layer 2: define_xxx!         — 算子逻辑模板（基线实现）
Layer 3: quant_primitive!    — 量化特化原语
Layer 4: expand_all_xxx!     — 批量展开
```

**覆写规则**：
- 宏生成的实现是**基线**（保证正确性）
- 手写汇编微内核**覆写**热路径（保证性能）
- 覆写必须通过 benchmark 证明优于基线

**详细宏设计**：见 `03-DATA-STRUCTURE.md` §8

---

## ISA 差异性（ARCH-ISA-DIFF）

> 不同 ISA 的最优算法**结构不同**，不仅仅是"换指令"。

| 差异维度 | AVX2 | AVX-512 | NEON |
|----------|------|---------|------|
| GEMM 微内核 | 6×16 | 14×32 | 8×12 |
| 寄存器数 | 16 ymm | 32 zmm | 32 v |
| 水平求和 | 4 步 shuffle | 原生 reduce | 原生 vaddvq |
| INT8 点积 | 无 | VNNI vpdpbusd | sdot |
| 预取距离 | 256B | 512B | 128B |

这就是为什么必须用宏（或手写汇编）而不是泛型 trait：一个 `fn gemm<S: SimdOps>()` 无法同时对 AVX2 用 6×16、对 AVX-512 用 14×32 微内核。

---

## 目录结构

```
src/
├── lib.rs                  # Crate 入口
├── traits.rs               # Element trait
├── quant.rs                # QuantType 枚举
├── codebooks.rs            # IQ 码本常量
│
├── macros/                 # 宏架构（非热路径基线）
│   ├── simd_primitive.rs   # Layer 1
│   ├── operator_templates.rs # Layer 2
│   ├── quant_primitive/    # Layer 3
│   └── expand.rs           # Layer 4
│
├── cpu_kernels/            # CPU 后端
│   ├── mod.rs              # ISA 分发
│   ├── scalar/             # Scalar 兜底
│   ├── avx2/               # AVX2 实现
│   ├── avx512/             # AVX-512 实现
│   └── neon/               # NEON 实现
│
└── asm/                    # 手写汇编微内核
    ├── x86_64/             # AVX2 / AVX-512 汇编
    └── aarch64/            # NEON 汇编
```

---

## 8. 算法意图编译器架构（ARCH-COMPILER）

> 标量定义 → 二进制分析 → 融合决策 → 全新代码生成。算子的唯一定义来源是 `extern "C"` 纯标量函数，编译器通过二进制符号执行自动提取计算结构（OpTrace），然后根据 DeviceProfile 生成最优融合 SIMD 代码。

### 8.1 设计哲学

**标量优先。** 算子开发者只写 `extern "C"` 纯标量函数（数学公式的直接翻译），编译器对其编译后的二进制做符号执行，自动提取完整计算结构（OpTrace）。

**为什么标量 C ABI：**
1. **编译器可分析性** — 标量代码编译后只有简单的 x87/SSE 标量指令 + 循环，符号执行复杂度极低（对比 SIMD 模板的数百条向量指令）
2. **算子开发者零门槛** — 只需写数学公式的直接翻译，无需理解 SIMD/寄存器分配
3. **自动正确性基准** — 标量实现本身就是 golden reference
4. **新算子即插即用** — 写一个 `extern "C"` 函数，编译器自动分析 + 生成最优代码

**融合 = 全新代码生成。** 编译器根据 OpTrace（计算结构）和 DeviceProfile（硬件特征），直接用平台汇编器程序化生成全新的融合内核。不从模板中提取代码片段。

**Profile 驱动。** 同一算子 DAG 在不同硬件上可能产生完全不同的融合策略和代码结构。融合决策完全由 DeviceProfile 驱动，不依赖"模板是否存在"。

**ISA 无关性。** AVX2、AVX-512、NEON 不是独立功能。它们是 MachineCodeEmitter 后端的寄存器宽度选择。x86_64 后端根据 DeviceProfile.isa 在 ymm (256-bit) 和 zmm (512-bit) 之间选择。aarch64 后端使用 v 寄存器 (128-bit)。Phase 0-2 完全不感知 ISA。

### 8.2 四阶段编译流水线

```
标量函数注册表 (ScalarOpRegistry)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 0: 二进制符号执行（标量函数分析）                       │
│  · 取 extern "C" 函数指针，用 iced-x86 Decoder 反汇编        │
│  · 符号执行引擎追踪 load → compute → store 数据流             │
│  · 识别循环结构、归约模式、多 pass 结构                       │
│  · 输出: OpTrace（完整计算结构描述）                          │
│  · 首次分析后缓存，同一算子不重复分析                         │
└──────────────────────────┬───────────────────────────────────┘
                           │
CompilerGraph + DeviceProfile + OpTrace 缓存
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: 语义 DAG 构筑                                       │
│  · 算子 → 查 ScalarOpRegistry → 取已缓存的 OpTrace           │
│  · OpTrace.pattern 自动推导算子分类（不再手动映射）            │
│  · 构建张量 def-use 链（每条边标注数据量、形状）               │
│  · 构建后支配树（用于融合组划分）                              │
│  · 输出: SemanticDAG（节点携带 OpTrace）                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 2: Profile-Driven 融合决策                              │
│                                                               │
│  输入: SemanticDAG + DeviceProfile                            │
│                                                               │
│  Step 1: 融合组划分（后支配树 + 算子分类规则）                 │
│    · elemwise/injective 链 → Loop Fusion 候选                 │
│    · GEMM 后的 elemwise 消费者 → Epilogue Injection 候选      │
│    · GEMM 前的 norm 生产者 → Tile-Level Fusion 候选           │
│                                                               │
│  Step 2: Profile 约束检查                                     │
│    · 中间张量 > L1 容量? → 必须 tile-level fusion             │
│    · 中间张量 ≤ L1 容量? → compute_root 即可                  │
│    · 融合后寄存器压力 > 可用寄存器? → 拒绝融合或插入 spill    │
│    · 生产者有多个消费者? → 不融合（避免重复计算）             │
│                                                               │
│  Step 3: Tiling 参数计算                                      │
│    · GEMM: BLIS 三级分块 KC/MC/NC（适配 L1/L2/L3）           │
│    · Elementwise: tile 大小适配 L1                             │
│    · Tile-Level Fusion: MC 对齐前驱算子的 tile 边界           │
│                                                               │
│  Step 3.5: 并行化决策                                         │
│    · GEMM: NC 循环并行（每个 NC tile 独立，无数据依赖）       │
│    · Elementwise/LoopFusion: 按元素数均分到线程               │
│    · 数据量 < 阈值 → Sequential（并行开销不值得）             │
│    · JIT 代码是单线程的，调用方 thread pool 按策略分发 tile   │
│                                                               │
│  Step 4: Buffer 规划                                          │
│    · 张量活性分析: birth/death 拓扑序位置                     │
│    · 区间图着色贪心算法: 最大化 buffer 原地复用               │
│                                                               │
│  输出: FusionPlan（策略 + TileConfig + ParallelStrategy + BufferPlan）│
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 3: 全新代码生成                                        │
│  · 从 OpTrace 的 Vec<TraceOp> 直接映射到 SIMD 指令           │
│  · TraceOp::Add → vaddps, TraceOp::Mul → vmulps, ...        │
│  · TraceOp::Exp → 多项式逼近指令序列                         │
│  · Epilogue Injection: 取消费者 OpTrace.body，               │
│    对每个 TraceOp 生成 SIMD 指令，在累加器上原地执行          │
│  · GEMM → 完整 BLIS 三重循环 + epilogue 在累加器写回前执行   │
│  · Tile-Level Fusion → 前驱算子的 tile 计算嵌入               │
│    GEMM MC 循环内部                                           │
│  · 输出: CompiledLayer (mmap RWX 可执行页)                    │
└──────────────────────────────────────────────────────────────┘
```

### 8.3 Phase 0: 二进制符号执行（标量函数分析）

**输入**：`extern "C"` 标量函数指针（从 `ScalarOpRegistry` 获取）

**核心洞察：fn_ptr 的双重角色**

`ScalarOpRegistry` 注册的 `fn_ptr` 同时是：
1. **执行入口** — 可直接调用，作为正确性基准（golden reference）
2. **分析入口** — iced-x86 Decoder 从此地址开始反汇编，提取计算结构

DAG 中每个 `CompilerOp { kind: OpKind::Silu }` 通过 `registry.get(OpKind::Silu)` 获取 fn_ptr，
既知道「这个算子的函数实现在哪」（可执行），也知道「要分析的二进制在哪」（可分析）。

**标量函数的形态由数学定义决定**

标量函数不是"设计选择"，而是算法的直接翻译：
- SiLU 的数学定义是 `x·σ(x)`，标量实现就是 `x / (1 + exp(-x))`，编译后几条标量指令 + `call expf`
- RMSNorm 天然需要向量输入（要算方差），标量实现自然带循环和归约
- GEMM 天然是三重循环 + FMA 累加

每个算子的二进制复杂度由其数学结构决定，符号执行面对的不是任意二进制，
而是我们自己编译的、已知入口地址的、结构由数学公式决定的简单函数。

**处理流程**：

1. **反汇编**：iced-x86 Decoder 从函数地址开始，反汇编到 `ret` 指令。`extern "C"` 保证函数以标准 prologue/epilogue 包裹。

2. **循环检测**：找 backward jump（目标地址 < 当前地址）= back-edge = 循环。标量函数的循环结构简单（单层或两层嵌套），不需要完整的 CFG 分析。

3. **符号执行**：`SymState`（寄存器 → `SymValue` 映射）逐指令推进，追踪每条指令的数据流（load → compute → store）。

4. **libm 调用识别**：`call` 指令的目标地址解析为 `expf`, `sqrtf`, `tanhf` 等。通过符号表或已知地址映射。

5. **归约模式识别**：循环内有 `addss xmm_acc, xmm_temp` 且 `xmm_acc` 跨迭代存活 → 归约累加器。

6. **多 pass 识别**：多个连续循环（同一个 `n` 参数控制边界）→ multi-pass 算子（如 rms_norm 的 sum_squares + scale）。

7. **GEMM 模式识别**：三重嵌套循环 + 内层是 `a * b + c` 累加模式 → `ComputePattern::Gemm`。不需要"知道这是矩阵乘法"，只需识别循环嵌套结构和 FMA 累加模式。

**输出**：`OpTrace { pattern: ComputePattern, signature: ScalarFnSignature }`

**缓存**：`OpKind → OpTrace` 的 HashMap，首次分析后缓存，同一算子不重复分析。

**与旧 Phase 0（已删除）的区别**：
- 旧 Phase 0 对 SIMD 模板二进制做符号执行 → 复杂度太高，被删除
- 新 Phase 0 对纯标量 C ABI 函数做符号执行 → 只有标量指令 + 简单循环，复杂度极低

**标量函数约束**（编译器可分析的前提）：
- `extern "C"` ABI — 干净的调用约定，无 Rust name mangling
- 只用标量算术：`+`, `-`, `*`, `/`, `exp()`, `sqrt()`, `tanh()` 等
- 不调用其他自定义函数（libm 函数除外）
- 不做堆分配
- 循环结构清晰：`for i in 0..n` 或等价的 while 循环
- 编译时用 `-C opt-level=1`（保留循环结构，消除冗余，不做向量化）

**标量函数示例**：

```rust
// src/scalar_ops/activations.rs

/// SiLU: out[i] = x[i] / (1 + exp(-x[i]))
#[no_mangle]
pub extern "C" fn scalar_silu(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        unsafe {
            let v = *x.add(i);
            *out.add(i) = v / (1.0 + (-v).exp());
        }
    }
}

/// RMSNorm: two-pass — sum_squares then scale
#[no_mangle]
pub extern "C" fn scalar_rms_norm(
    x: *const f32, weight: *const f32, out: *mut f32, n: usize, eps: f32,
) {
    unsafe {
        // Pass 1: sum of squares
        let mut ss: f32 = 0.0;
        for i in 0..n {
            let v = *x.add(i);
            ss += v * v;
        }
        let inv_rms = 1.0 / ((ss / n as f32) + eps).sqrt();
        // Pass 2: scale
        for i in 0..n {
            *out.add(i) = *x.add(i) * inv_rms * *weight.add(i);
        }
    }
}
```

符号执行对 `scalar_silu` 的分析结果：
```
OpTrace {
    pattern: ComputePattern::Elementwise {
        body: [Input(0), Neg(0), Exp(1), Const(1.0), Add(2, 3), Div(0, 4)]
    },
    // SiLU: v / (1 + exp(-v))
}
```

### 8.4 Phase 1: 语义 DAG 构筑

```rust
/// 编译器入口
pub fn compile_graph(
    graph: &CompilerGraph,
    profile: &DeviceProfile,
) -> Result<CompiledLayer, CompileError>;
```

**CompilerGraph 来源**：由 GLLM 将 FusedGraph 展开为原子算子 DAG 后传入。gllm-kernels 不负责 ONNX 文件加载与解析。

**构筑过程**：

```
CompilerGraph
    │
    ├── 1. 算子绑定: 每个算子 → 查 ScalarOpRegistry → 取已缓存的 OpTrace
    │      · OpTrace 包含完整计算结构（不只是分类标签）
    │      · 首次分析后缓存，同一算子不重复分析
    │
    ├── 2. 算子分类（从 OpTrace.pattern 自动推导，不再手动映射）:
    │      · ComputePattern::Elementwise / BinaryElementwise → kElemWise
    │      · ComputePattern::Reduction → kReduction
    │      · ComputePattern::NormLike → kReduction
    │      · ComputePattern::Gemm → kGemm
    │      · ComputePattern::QuantDecode → kOpaque
    │      · kInjective: rope, reshape, transpose（从 CompilerOp 类型推导）
    │
    ├── 3. 张量 def-use 链: 每条边标注
    │      · data_bytes: 数据量（元素数 × sizeof(E)）
    │      · consumers: 消费者节点列表
    │      · can_register_pass: 是否可寄存器传递（单消费者 + elemwise）
    │
    └── 4. 后支配树构建（用于 Phase 2 融合组划分）
```

**未映射算子处理**：返回 `CompileError::UnsupportedOp(op_name)`，GLLM 回退到 fallback 路径。

### 8.5 Phase 2: Profile-Driven 融合决策

融合决策完全由硬件 profile 和算子语义驱动，不依赖"模板是否存在"。

**入口签名**：

```rust
/// Phase 2 入口 — 必须接收 DeviceProfile，融合决策由硬件状态驱动
pub fn fuse(
    graph: &CompilerGraph,
    profile: &DeviceProfile,
) -> FusionPlan;
```

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

**额外约束（硬件驱动）**：
- 生产者有多个消费者 → 不融合（避免重复计算）
- 融合组内总寄存器压力 > `profile.num_simd_regs()` → 拒绝融合或拆分
- Epilogue 需要的 scratch 寄存器 > 微内核剩余寄存器 → 拒绝 epilogue injection

**Step 2: 五种融合模式（FusionMode）**

每个 FusionGroup 携带一个 `FusionMode`，由硬件 profile 和数据量共同决定：

```rust
pub enum FusionMode {
    /// GEMM store 阶段注入 epilogue（消费者 OpTrace.body → SIMD）
    EpilogueInjection { epilogue_ops: Vec<OpId> },
    /// 多个 elementwise 合并为单循环（数据在寄存器中流过整个链）
    LoopFusion { chain: Vec<OpId> },
    /// 前驱算子嵌入 GEMM MC 循环（硬件驱动：output > L1 * 0.75）
    TileLevelFusion { predecessor: OpId, tile_rows: usize },
    /// 前驱算子先算完，结果留 L1（硬件驱动：output ≤ L1 * 0.75）
    ComputeRoot { predecessor: OpId },
    /// 不融合
    Standalone,
}
```

**Step 2a: TileLevelFusion vs ComputeRoot 的硬件驱动决策**

```
决策规则（以 RMSNorm → GEMM 为例）:

  let output_bytes = hidden_dim * dtype.size_bytes();  // RMSNorm 输出大小
  let (l1, _, _) = profile.cache_sizes();              // 从 DeviceProfile 获取 L1 容量

  if output_bytes > l1 * 3 / 4 {
      // 输出超过 L1 的 75% → 先算完再读时已被逐出 L1
      // 必须嵌入 GEMM MC 循环，每次只算 MC 行，输出留在 L1 热区
      let (_, mc, _) = profile.gemm_blocking(shape);
      FusionMode::TileLevelFusion { predecessor: norm_op, tile_rows: mc }
  } else {
      // 输出 ≤ L1 的 75% → 先算完，结果整体留在 L1
      // GEMM 读 A 矩阵时 RMSNorm 结果仍然热
      FusionMode::ComputeRoot { predecessor: norm_op }
  }
```

**TileLevelFusion 的 scratch buffer 方案**：

```
数据流:
  scratchpad (由 Phase 2 Step 4 buffer planning 分配)
      │
      ├── normed 区域: MC × K × sizeof(E) bytes
      │   · MC 行的 RMSNorm 结果写入此处
      │   · 紧接着被 pack_a 消费，pack_a 按 KC 列切片读取
      │   · 每次只读 MC × KC × sizeof(E)，在 L2 内
      │
      └── 其他中间张量...

  weight 向量 (hidden_dim × sizeof(E)):
      · 是 graph input，通过 JIT 函数参数传入（不在 scratchpad 里）
      · 每个 MC tile 都读完整 weight 向量（只读，L2 热驻留）

  正确性保证:
      · RMSNorm 是逐行独立的（每行的 norm 只依赖该行自身）
      · 按 MC 行切分不影响正确性
      · 每个 MC tile 独立做完整的两 pass（pass 1: sum_squares, pass 2: scale）
```

**模式 A: Epilogue Injection（GEMM 后注入）**

```
场景: GEMM → Bias → SiLU

未融合:
  GEMM: C[m][n] = Σ A[m][k]*B[k][n]  → 写回内存
  Bias: C[m][n] += bias[n]             → 读内存，写回内存
  SiLU: C[m][n] = silu(C[m][n])        → 读内存，写回内存
  内存往返: 3 次写 + 2 次读

融合后（单一微内核 epilogue）:
  GEMM K-loop: ymm0..ymm11 累加完毕
  注入 bias:   vaddps ymm_i, ymm_i, [bias + j*32]
  注入 SiLU:   对 ymm_i 原地执行 SiLU（程序化生成的指令序列）
  store:       vmovups [C + offset], ymm_i
  内存往返: 1 次写

实现: 编译器根据 SiLU 的数学语义（x * sigmoid(x)）直接用汇编器生成对单个 ymm 执行
     SiLU 的 ~15 条指令，注入到 GEMM 微内核的 store 阶段之前。由于 GEMM 6x16 有
     12 个累加器(ymm0-11) 和 4 个 scratch(ymm12-15)，SiLU 需要 3 个 scratch，
     可以逐行处理：对 ymm0-ymm1 执行 SiLU（用 ymm12-14 做 scratch），
     然后 ymm2-ymm3，以此类推。
```

**模式 B: Loop Fusion（Elementwise 链合并）**

```
场景: SiLU → VecMul → VecAdd

未融合（3 个独立循环）:
  Loop1: for i { out1[i] = silu(in[i]) }        → 读 in, 写 out1
  Loop2: for i { out2[i] = out1[i] * w[i] }     → 读 out1+w, 写 out2
  Loop3: for i { out3[i] = out2[i] + res[i] }   → 读 out2+res, 写 out3
  内存往返: 3 次读 + 3 次写 + 2 次中间读

融合后（单循环，数据在寄存器中流过整个链）:
  for i in (0..n).step_by(8):
    ymm0 = load(in[i..i+8])
    ymm0 = silu(ymm0)           // 寄存器内，不写内存
    ymm0 = ymm0 * load(w[i])    // 寄存器内
    ymm0 = ymm0 + load(res[i])  // 寄存器内
    store(out[i..i+8], ymm0)
  内存往返: 3 次读 + 1 次写（消除 2 次中间写 + 2 次中间读）

实现: 用 iced-x86 CodeAssembler (x86_64) / dynasm-rs (aarch64) 程序化生成单循环体。每个算子的核心计算
     根据数学语义直接生成指令序列，按顺序注入循环体。数据始终在 ymm0 中，
     不经过内存。
```

**模式 C: Tile-Level Fusion（前驱算子嵌入 GEMM 循环）**

```
场景: RMSNorm → GEMM, hidden_dim=16384

Profile 分析:
  RMSNorm 输出 = 16384 * 4B = 64KB > L1(32KB)
  → 如果先算完 RMSNorm 再算 GEMM，GEMM 读 A 矩阵时 RMSNorm 结果已被逐出 L1
  → 必须 tile-level fusion

融合后（RMSNorm tile 嵌入 GEMM MC 循环）:
  for nc in 0..N/NC:
    pack_b(B[0..K, nc..nc+NC])
    for mc in 0..M/MC:                          // MC ≈ 72-144
      // ★ 嵌入: 只算 MC 行的 RMSNorm
      rmsnorm_tile(x[mc..mc+MC], w, scratch_a)  // 输出 MC*K*4B ≈ 72*4096*4 = 1.1MB
      pack_a(scratch_a[mc..mc+MC])               // pack 后立即被微内核消费，L1 热
      for nr in 0..NC/NR:
        microkernel(KC, packed_a, packed_b, C)

对比 compute_root（hidden_dim=4096, 输出=16KB < L1）:
  rmsnorm(x, w, normed_x)    // 整体算完，结果 16KB 留在 L1
  for nc in 0..N/NC:
    pack_b(...)
    for mc in 0..M/MC:
      pack_a(normed_x[mc..mc+MC])  // 从 L1 读，仍然热
      ...

决策规则:
  if rmsnorm_output_bytes > profile.l1_cache_bytes * 0.75 {
      TileLevelFusion  // 嵌入 MC 循环
  } else {
      ComputeRoot      // 先算完，结果留在 L1
  }
```

**Step 3: 同一 DAG 在不同硬件上的融合差异**

```
DAG: RMSNorm(4096) → Wq_GEMM → RoPE → Attention

硬件 A (L1=32KB, 16 ymm, AVX2):
  RMSNorm 输出 = 16KB < L1*0.75 → ComputeRoot
  Wq_GEMM + RoPE → Epilogue Injection（RoPE 注入 GEMM store 阶段）
  生成: rmsnorm() + gemm_with_rope_epilogue()

硬件 B (L1=32KB, 但 hidden=16384):
  RMSNorm 输出 = 64KB > L1*0.75 → Tile-Level Fusion
  Wq_GEMM + RoPE → Epilogue Injection
  生成: gemm_with_rmsnorm_tile_and_rope_epilogue()（三算子融合）

硬件 C (L1=48KB, 32 zmm, AVX-512):
  RMSNorm 输出 = 64KB > L1*0.75 → Tile-Level Fusion
  但 32 个 zmm 寄存器 → GEMM 14x32 用 28 个累加器 + 4 scratch
  RoPE epilogue 需要 4 scratch → 寄存器不够 → 拒绝 epilogue injection
  生成: gemm_with_rmsnorm_tile() + rope_standalone()
```

### 8.5 Phase 3: 全新代码生成

**核心原则**：用平台特定汇编器（x86_64: iced-x86 CodeAssembler / aarch64: dynasm-rs Assembler）程序化生成每一条指令。编译器从 OpTrace 的 `Vec<TraceOp>` 直接映射到 SIMD 指令，不从模板中提取片段。两个后端通过 `MachineCodeEmitter` trait 统一接口。

**TraceOp → SIMD 指令映射**：

两个后端通过 `MachineCodeEmitter` trait 统一接口，ISA 变体是同一后端的寄存器宽度分支。

**x86_64 后端 (iced-x86)**：根据 DeviceProfile.isa 选择 ymm (AVX2) 或 zmm (AVX-512)

| TraceOp | AVX2 (ymm) / AVX-512 (zmm) |
|---------|----------------------------|
| Add(a,b) | vaddps |
| Sub(a,b) | vsubps |
| Mul(a,b) | vmulps |
| Div(a,b) | vdivps |
| Fma(a,b,c) | vfmadd231ps |
| Neg(a) | vxorps(sign_mask) |
| Exp(a) | 多项式逼近 ~12 条（寄存器宽度由 DeviceProfile 决定） |
| Recip(a) | vrcpps/vrcp14ps + Newton |
| Rsqrt(a) | vrsqrtps/vrsqrt14ps + Newton |
| Sqrt(a) | vsqrtps |
| Tanh(a) | 有理逼近 ~15 条 |
| Abs(a) | vandps(abs_mask) |
| Max(a,b) | vmaxps |
| Min(a,b) | vminps |

注: AVX2 和 AVX-512 是同一后端的寄存器宽度分支，不是独立实现。

**aarch64 后端 (dynasm-rs)**：NEON v 寄存器 (128-bit)

| TraceOp | NEON (v.4s) |
|---------|-------------|
| Add(a,b) | fadd |
| Sub(a,b) | fsub |
| Mul(a,b) | fmul |
| Div(a,b) | fdiv |
| Fma(a,b,c) | fmla |
| Neg(a) | fneg |
| Exp(a) | 多项式逼近（v 寄存器） |
| Recip(a) | frecpe + Newton |
| Rsqrt(a) | frsqrte + Newton |
| Sqrt(a) | fsqrt |
| Tanh(a) | 有理逼近 |
| Abs(a) | fabs |
| Max(a,b) | fmax |
| Min(a,b) | fmin |

**Epilogue Injection 的数据来源**：
- GEMM 微内核累加完毕后，取消费者算子的 `OpTrace.body`（`Vec<TraceOp>`）
- 对 `body` 中每个 `TraceOp` 生成对应 SIMD 指令
- 在累加器寄存器上原地执行（数据不落地内存）
- 例：SiLU epilogue = 从 OpTrace 提取 `[Neg, Exp, Add(1.0), Recip, Mul]`，对每个 TraceOp 生成 ~15 条 AVX2 指令

**GEMM + Epilogue 生成**：

```
JIT 生成的代码结构（全新，非模板拼接）:

prologue (save callee-saved: rbx, r12-r15, rbp)
│
├── NC loop (r12 = nc_counter)
│   ├── pack_b: 程序化生成的 pack 循环
│   │
│   ├── MC loop (r13 = mc_counter)
│   │   ├── [可选] rmsnorm_tile: 嵌入的前驱算子 tile 计算
│   │   ├── pack_a: 程序化生成的 pack 循环
│   │   │
│   │   └── NR loop (r14 = nr_counter)
│   │       └── 微内核:
│   │           ├── 累加器清零: vxorps ymm0..ymm11
│   │           ├── K-loop: 程序化生成 FMA 序列
│   │           │   · vbroadcastss ymm12, [A + k*4]
│   │           │   · vmovups ymm14, [B + k*NR*4]
│   │           │   · vfmadd231ps ymm0, ymm12, ymm14
│   │           │   · ... (MR 行展开)
│   │           ├── [可选] bias: vaddps ymm_i, ymm_i, [bias + j*32]
│   │           ├── [可选] activation: 程序化生成激活函数指令序列
│   │           │   · 对 ymm0..ymm11 逐对执行，用 ymm12-14 做 scratch
│   │           └── store: vmovups [C + offset], ymm0..ymm11
│   │
│   └── edge tile 处理 (M/N 尾部)
│
epilogue (restore + ret)

所有循环边界（KC/MC/NC）在 JIT 时已知，作为立即数 bake 进机器码。
外层循环只使用微内核不碰的寄存器（编译器根据微内核规格确定的安全集合）。
```

**Elementwise 链生成**：

```
JIT 生成的代码结构:

prologue
│
├── 主循环 (rbx = element_counter, step = 8 for ymm)
│   ├── ymm0 = vmovups [rdi]              // 加载输入
│   ├── 遍历 OpTrace.body 中每个 TraceOp，生成对应 SIMD 指令
│   │   · TraceOp::Neg → vxorps ymm0, ymm0, sign_mask
│   │   · TraceOp::Exp → 多项式逼近指令序列 on ymm0
│   │   · TraceOp::Mul → vmulps ymm0, ymm0, [rsi]
│   │   · TraceOp::Add → vaddps ymm0, ymm0, [rdx]
│   ├── vmovups [rcx], ymm0               // 存储输出
│   ├── add rdi/rsi/rdx/rcx, 32           // 推进指针
│   └── dec rbx; jnz loop                 // 循环
│
├── 尾部处理 (标量或 masked store)
│
epilogue
```

**与旧架构的关键区别**：

| 维度 | 旧架构（内置语义知识） | 新架构（标量函数 + 符号执行） |
|------|----------------------|------------------------------|
| 算子知识来源 | 编译器内置 OpSemanticsKind（4 个分类标签） | extern "C" 标量函数 → 符号执行 → OpTrace（完整计算结构） |
| 算子分类 | 手动映射表（CompilerOp → OpSemanticsKind） | 从 OpTrace.pattern 自动推导 |
| 代码生成数据 | 编译器"知道" SiLU 是什么（硬编码） | 从 OpTrace.body 的 TraceOp 序列逐条映射到 SIMD 指令 |
| 新增算子 | 需修改编译器内部映射表 + 代码生成逻辑 | 只需写一个 extern "C" 标量函数，编译器自动分析 |
| 融合决策 | profile + 语义分类 | profile + OpTrace.pattern（不变） |
| GEMM epilogue | 编译器需要"知道"每种激活函数的指令序列 | 从消费者 OpTrace.body 自动生成 |
| 正确性基准 | 需要单独维护 golden reference | 标量函数本身就是 golden reference |

### 8.6 汇编器后端（PlatformBackend）

Phase 1-2（DAG 构筑、融合决策）完全平台无关。平台差异封装在 `MachineCodeEmitter` trait 中，通过 `PlatformBackend` 统一入口提供：

```rust
/// 平台后端 — 提供 Phase 3 代码生成能力
trait PlatformBackend {
    type Emitter: MachineCodeEmitter;

    fn new_emitter(&self) -> Self::Emitter;
    fn platform(&self) -> Platform;
    fn num_simd_regs(&self) -> usize;
}

/// Phase 3: 代码生成
trait MachineCodeEmitter {
    /// 从 OpTrace.body 的 TraceOp 序列生成 SIMD 指令（对指定寄存器原地执行）
    fn emit_trace_ops(&mut self, ops: &[TraceOp], regs: &[VirtReg], scratch: &[VirtReg]) -> Result<()>;
    fn emit_gemm_unit(&mut self, unit: &GemmUnit) -> Result<Vec<u8>>;
    fn emit_fused_loop(&mut self, unit: &FusedLoop) -> Result<Vec<u8>>;
    fn emit_prologue(&mut self) -> Result<()>;
    fn emit_epilogue(&mut self) -> Result<()>;
    fn finalize(self) -> Result<Vec<u8>>;
}
```

编译流水线只依赖 `PlatformBackend`，不感知底层是 iced-x86 还是 dynasm-rs：

```
compile_graph(graph, profile, backend: &dyn PlatformBackend)
│
├── Phase 1-2 [平台无关]: 纯数据结构操作（DAG 构筑 + 融合决策）
│
└── Phase 3 [平台特定]: let mut emitter = backend.new_emitter();
                         emitter.emit_gemm_unit(&gemm_unit)?;
                         emitter.emit_trace_ops(&silu_trace.body, &regs, &scratch)?;
                         let code = emitter.finalize()?;
```

平台特定代码只存在于 Phase 3 的代码生成。DAG 构筑（Phase 1）和融合决策（Phase 2）全部平台无关。

**x86_64 后端（iced-x86）**：

```rust
struct X86Emitter {
    asm: iced_x86::code_asm::CodeAssembler,  // 64-bit mode
}

// Phase 3 代码生成:
//   asm.vfmadd231ps(ymm0, ymm12, ymm14)?;  // AVX2 FMA
//   asm.vmovups(ymm0, ptr(rdi))?;           // AVX2 load
//   asm.vaddps(zmm0, zmm0, zmm1)?;          // AVX-512
//   let code = asm.assemble(0x0)?;           // → Vec<u8>
```

**aarch64 后端（dynasm-rs）**：

```rust
struct Arm64Emitter {
    ops: dynasmrt::aarch64::Assembler,
}

// Phase 3 代码生成:
//   dynasm!(ops
//     ; fmla v0.4s, v24.4s, v28.s[0]   // NEON FMA (by element)
//     ; ldp q24, q25, [x1], #32         // load pair + post-index
//     ; dup v28.4s, v16.s[0]            // broadcast
//     ; st1 {v0.4s, v1.4s}, [x0], #32  // store + post-index
//   );
//   let buf = ops.finalize().unwrap();   // → ExecutableBuffer (mmap RWX)
```

**aarch64 GEMM 微内核示例（8x12 NEON）**：

```
prologue (save x19-x28, d8-d15)
│
├── NC loop (x19 = nc_counter)
│   ├── pack_b
│   ├── MC loop (x20 = mc_counter)
│   │   ├── pack_a
│   │   └── NR loop (x21 = nr_counter)
│   │       └── 微内核 (8x12, 24 个累加器 v0-v23):
│   │           ├── 累加器清零: movi v0.4s, #0 ... movi v23.4s, #0
│   │           ├── K-loop: 程序化生成 FMA 序列
│   │           │   · ldr q24, [x1]           // load B[k][0:4]
│   │           │   · ldr q25, [x1, #16]      // load B[k][4:8]
│   │           │   · ldr q26, [x1, #32]      // load B[k][8:12]
│   │           │   · ld1r {v28.4s}, [x0]     // broadcast A[0][k]
│   │           │   · fmla v0.4s, v24.4s, v28.4s   // C[0][0:4] += A[0]*B[0:4]
│   │           │   · fmla v1.4s, v25.4s, v28.4s   // C[0][4:8]
│   │           │   · fmla v2.4s, v26.4s, v28.4s   // C[0][8:12]
│   │           │   · ... (8 行展开)
│   │           ├── [可选] epilogue: 程序化生成激活函数指令序列
│   │           └── store: stp q0, q1, [x2] ...
│   └── edge tile
│
epilogue (restore + ret)
```

**平台差异总结**：

| 维度 | x86_64 (iced-x86) | aarch64 (dynasm-rs) |
|------|-------------------|---------------------|
| SIMD 宽度 | 256-bit ymm / 512-bit zmm | 128-bit v (NEON) |
| GEMM 微内核 | 6x16 (AVX2) / 14x32 (AVX-512) | 8x12 (NEON) |
| 累加器数 | 12 ymm / 28 zmm | 24 v |
| Scratch 寄存器 | ymm12-15 / zmm28-31 | v24-v31 |
| FMA 指令 | vfmadd231ps (3 操作数) | fmla (by-element 变体) |
| Broadcast | vbroadcastss (专用指令) | dup / ld1r |
| 调用约定 | System V AMD64 | AAPCS64 |
| AVX-512 | ✅ 完整 EVEX 编码 | N/A |
| SVE | N/A | ❌ dynasm-rs 暂不支持 |
