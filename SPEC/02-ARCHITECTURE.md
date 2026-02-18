# gllm-kernels 架构设计

## 定位

**gllm-kernels = 极限性能 CPU 算子库**

提供逼近硬件理论峰值的底层计算原语。上层推理引擎（gllm）通过组合这些算子构建完整推理管线。

**不是**推理引擎、不含业务逻辑、不含 GPU 后端。

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

### 3. 纯 CPU 算子库（ARCH-CPU-ONLY）

本库只实现 CPU 后端。GPU 后端（CUDA/Metal/ROCm）由上层推理引擎负责。

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

**不属于本库**：
- FlashAttention / Paged Attention（业务算法）
- KV Cache 管理（业务状态）
- 融合算子 fused_qkv_rope / fused_ffn 等（业务组合）
- Embedding lookup（查表，非计算密集）
- Sampling（业务逻辑）
- GPU 后端（独立项目）

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
