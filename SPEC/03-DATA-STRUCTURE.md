# gllm-kernels 数据结构与算子架构

> **📌 SSOT**: 本文档定义 gllm-kernels 的核心数据结构、算子清单、分发架构。
> 当前聚焦 CPU 后端，GPU 后端（CUDA/HIP/Metal）为规划中的未来工作（见 SPEC/04-GPU-BACKEND.md）。

---

## 1. 三层零成本分发架构（ARCH-DISPATCH）🚨 铁律

> **📌 SSOT**: 分发架构的权威定义见 SPEC/02-ARCHITECTURE.md §ARCH-DISPATCH。本节为数据结构视角的简要描述。

### 1.1 架构总览

```
Layer 1: Backend    → CpuBackend（当前唯一后端）     — 编译时确定
Layer 2: ISA        → Scalar/AVX2/AVX-512/NEON       — 启动时一次检测（OnceLock）
Layer 3: Precision  → f32/f16/bf16                    — 编译时泛型单态化
```

### 1.2 零成本分发原则

| 层级 | 分发时机 | 机制 | 开销 |
|------|----------|------|------|
| **Layer 1** | 编译时 | Cargo feature 选择后端 | 零 |
| **Layer 2** | 程序启动时一次 | `OnceLock` + ISA 检测 | 启动时一次 |
| **Layer 3** | 编译时 | Rust 单态化 (monomorphization) | 零 |

**关键**：ISA 检测只在程序启动时发生一次，之后整棵算子树都是静态确定的。

---

## 2. 核心 Trait 定义

### 2.1 Element Trait（DATA-ELEMENT）

```rust
/// 计算精度 Trait
///
/// 设计原则：
/// - 编译时单态化，零运行时开销
/// - 覆盖推理常用精度：f32, f16, bf16
pub trait Element: Copy + Send + Sync + Default + 'static {
    /// 加法单位元
    const ZERO: Self;
    /// 乘法单位元
    const ONE: Self;
    /// 类型判别：0=f32, 1=f16, 2=bf16
    const ELEM_ID: u8;

    /// 从 f32 转换（解量化后的标准格式）
    fn from_f32(v: f32) -> Self;
    /// 转换为 f32（最终输出或高精度计算）
    fn to_f32(self) -> f32;

    /// 融合乘加：self + a * b
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// 基础算术
    fn elem_add(self, other: Self) -> Self;
    fn elem_sub(self, other: Self) -> Self;
    fn elem_mul(self, other: Self) -> Self;
    fn elem_div(self, other: Self) -> Self;
    fn neg(self) -> Self;

    /// 比较
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    /// 数学函数
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn recip(self) -> Self;  // 1/x
    fn abs(self) -> Self;
    fn tanh(self) -> Self;

    /// 零成本 f32 切片转换（仅 Self=f32 时返回 Some）
    fn as_f32_slice(s: &[Self]) -> Option<&[f32]>;
    fn as_f32_slice_mut(s: &mut [Self]) -> Option<&mut [f32]>;
    fn as_f32_ref(v: &Self) -> Option<&f32>;
}
```

### 2.2 CpuKernels 结构（DATA-CPU-KERNELS）

```rust
/// CPU 内核（包含 ISA 分发）
///
/// 本库唯一的后端实现。不存在 Backend trait 抽象层。
pub struct CpuKernels<E: Element> {
    inner: &'static dyn IsaKernels<E>,  // 启动时选择的 ISA 实现
}

impl<E: Element> CpuKernels<E> {
    /// 检测最优 ISA 并初始化（程序启动时调用一次）
    pub fn new() -> Self {
        static DETECTED: OnceLock<IsaLevel> = OnceLock::new();
        let isa = DETECTED.get_or_init(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") { return IsaLevel::Avx512; }
                if is_x86_feature_detected!("avx2") { return IsaLevel::Avx2; }
            }
            #[cfg(target_arch = "aarch64")]
            { return IsaLevel::Neon; }
            IsaLevel::Scalar
        });

        let inner: &'static dyn IsaKernels<E> = match isa {
            IsaLevel::Avx512 => &Avx512Impl::<E>,
            IsaLevel::Avx2 => &Avx2Impl::<E>,
            IsaLevel::Neon => &NeonImpl::<E>,
            IsaLevel::Scalar => &ScalarImpl::<E>,
        };
        Self { inner }
    }
}

/// ISA 类型枚举（仅用于 OnceLock 存储）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsaLevel {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}
```

### 2.3 Kernels Trait（DATA-KERNELS）🚨 核心

```rust
/// 内核算子接口 - 所有 ISA 实现此 Trait
///
/// E 是精度泛型，编译时单态化
pub trait Kernels<E: Element>: Send + Sync {

    // ========================================================================
    // 向量运算 (BLAS-1)
    // ========================================================================
    fn vec_dot(&self, a: &[E], b: &[E]) -> E;
    fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]);
    fn vec_sub(&self, a: &[E], b: &[E], out: &mut [E]);
    fn vec_mul(&self, a: &[E], b: &[E], out: &mut [E]);
    fn vec_scale(&self, x: &mut [E], s: E);
    fn vec_axpy(&self, y: &mut [E], a: E, x: &[E]);
    fn vec_sum(&self, x: &[E]) -> E;
    fn vec_max(&self, x: &[E]) -> E;
    fn vec_sum_squares(&self, x: &[E]) -> E;

    // ========================================================================
    // 矩阵运算 (BLAS-2/3)
    // ========================================================================
    fn gemv(&self, a: &[E], x: &[E], y: &mut [E], m: usize, n: usize);
    fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize);
    fn gemm_bias(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E],
                 m: usize, n: usize, k: usize);
    fn gemm_bias_act(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E],
                     m: usize, n: usize, k: usize, act: Activation);
    fn pack_b(&self, b: &[E], n: usize, k: usize) -> Vec<E>;
    fn gemm_prepacked(&self, a: &[E], packed_b: &[E], c: &mut [E],
                      m: usize, n: usize, k: usize);
    fn gemm_bias_prepacked(&self, a: &[E], packed_b: &[E], bias: &[E],
                           c: &mut [E], m: usize, n: usize, k: usize);

    // ========================================================================
    // 激活函数
    // ========================================================================
    fn silu(&self, x: &[E], out: &mut [E]);
    fn gelu(&self, x: &[E], out: &mut [E]);
    fn relu(&self, x: &[E], out: &mut [E]);
    fn tanh(&self, x: &[E], out: &mut [E]);
    fn swiglu(&self, gate: &[E], up: &[E], out: &mut [E]);
    fn softmax(&self, x: &[E], out: &mut [E]);
    fn exp(&self, x: &[E], out: &mut [E]);

    // ========================================================================
    // 归一化
    // ========================================================================
    fn rms_norm(&self, x: &[E], weight: &[E], out: &mut [E], eps: f32);
    fn layer_norm(&self, x: &[E], gamma: &[E], beta: &[E], out: &mut [E], eps: f32);

    // ========================================================================
    // 位置编码
    // ========================================================================
    fn rope(&self, qk: &mut [E], cos: &[E], sin: &[E],
            head_dim: usize, interleaved: bool);
    fn rope_with_pos(&self, qk: &mut [E], cos: &[E], sin: &[E],
                     head_dim: usize, position: usize, interleaved: bool);

    // ========================================================================
    // 解量化 (输出固定 f32)
    // ========================================================================
    // K-Quant 系列
    fn dequant_q2_k(&self, block: &[u8], out: &mut [f32]);
    fn dequant_q3_k(&self, block: &[u8], out: &mut [f32]);
    fn dequant_q4_k(&self, block: &[u8], out: &mut [f32]);
    fn dequant_q5_k(&self, block: &[u8], out: &mut [f32]);
    fn dequant_q6_k(&self, block: &[u8], out: &mut [f32]);
    fn dequant_q8_k(&self, block: &[u8], out: &mut [f32]);

    // IQ 系列
    fn dequant_iq1_s(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq1_m(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq2_xxs(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq2_xs(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq2_s(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq3_xxs(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq3_s(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq4_nl(&self, block: &[u8], out: &mut [f32]);
    fn dequant_iq4_xs(&self, block: &[u8], out: &mut [f32]);

    // 商业格式
    fn dequant_awq4(&self, packed: &[u8], zeros: &[u8],
                    scales: &[half::f16], out: &mut [f32]);
    fn dequant_gptq4(&self, packed: &[u8], g_idx: &[i32],
                     scales: &[half::f16], out: &mut [f32]);
    fn dequant_squeeze(&self, block: &[u8], out: &mut [f32]);

    // ========================================================================
    // 量化 GEMV/GEMM
    // ========================================================================
    fn gemv_q8(&self, weight: &[i8], input: &[E], scale: f32, n: usize) -> E;
    fn gemv_q4(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E;
    fn gemv_q2(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E;
    fn gemv_q1(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E;

    fn gemm_q8(&self, weight: &[i8], input: &[E], output: &mut [E],
               scales: &[f32], m: usize, n: usize, k: usize);
    fn gemm_q4(&self, weight: &[u8], input: &[E], output: &mut [E],
               scales: &[f32], m: usize, n: usize, k: usize);

    // ========================================================================
    // 量化格式专用 Matmul
    // ========================================================================
    fn kquant_matmul(&self, weight_blocks: &[u8], input: &[E], output: &mut [E],
                     quant_type: QuantType, m: usize, n: usize, k: usize);
    fn iq_matmul(&self, weight_blocks: &[u8], input: &[E], output: &mut [E],
                 quant_type: QuantType, m: usize, n: usize, k: usize);
    fn awq_matmul(&self, weight: &[u8], zeros: &[u8], scales: &[half::f16],
                  input: &[E], output: &mut [E], m: usize, n: usize, k: usize);
    fn gptq_matmul(&self, weight: &[u8], g_idx: &[i32], scales: &[half::f16],
                   input: &[E], output: &mut [E], m: usize, n: usize, k: usize);
    fn squeeze_matmul(&self, weight_blocks: &[u8], input: &[E], output: &mut [E],
                      m: usize, n: usize, k: usize);
}
```

---

## 3. CPU ISA 实现架构（DATA-CPU-ISA）

### 3.1 ISA 内核结构

```rust
/// ISA 级内核 Trait（内部使用，与 Kernels<E> 方法一致）
trait IsaKernels<E: Element>: Send + Sync + 'static {
    // ... 与 Kernels<E> 相同的方法签名
}

/// 各 ISA 实现（泛型 + 内部特化）
struct ScalarImpl<E>(PhantomData<E>);
struct Avx2Impl<E>(PhantomData<E>);
struct Avx512Impl<E>(PhantomData<E>);
struct NeonImpl<E>(PhantomData<E>);

impl<E: Element> IsaKernels<E> for ScalarImpl<E> { ... }
impl<E: Element> IsaKernels<E> for Avx2Impl<E> { ... }
impl<E: Element> IsaKernels<E> for Avx512Impl<E> { ... }
impl<E: Element> IsaKernels<E> for NeonImpl<E> { ... }
```

### 3.2 SIMD 精度处理策略

| 精度 | AVX2 | AVX-512 | NEON |
|------|------|---------|------|
| **f32** | `_mm256_fmadd_ps` (8-wide) | `_mm512_fmadd_ps` (16-wide) | `vfmaq_f32` (4-wide) |
| **f16** | F16C 转换 + f32 SIMD | AVX512-FP16 原生 或 转换 | NEON FP16 原生 |
| **bf16** | 位转换 + f32 SIMD | AVX512-BF16 原生 或 转换 | 位转换 + f32 SIMD |

---

## 4. 算子清单（DATA-OPS）

### 4.1 基础算子

| 类别 | 算子 | 数量 |
|------|------|------|
| **向量运算** | vec_dot, vec_add, vec_sub, vec_mul, vec_scale, vec_axpy, vec_sum, vec_max, vec_sum_squares | 9 |
| **矩阵运算** | gemv, gemm, gemm_bias, gemm_bias_act, pack_b, gemm_prepacked, gemm_bias_prepacked | 7 |
| **激活函数** | silu, gelu, relu, tanh, swiglu, softmax, exp | 7 |
| **归一化** | rms_norm, layer_norm | 2 |
| **位置编码** | rope, rope_with_pos | 2 |

### 4.2 解量化算子

| 类别 | 格式 | 块大小 | 块字节 | 位宽 |
|------|------|--------|--------|------|
| **K-Quant** | Q2_K | 256 | 84 | 2 |
| | Q3_K | 256 | 110 | 3 |
| | Q4_K | 256 | 144 | 4 |
| | Q5_K | 256 | 176 | 5 |
| | Q6_K | 256 | 210 | 6 |
| | Q8_K | 256 | 292 | 8 |
| **IQ 系列** | IQ1_S | 256 | 50 | 1 |
| | IQ1_M | 256 | 56 | 1 |
| | IQ2_XXS | 256 | 66 | 2 |
| | IQ2_XS | 256 | 74 | 2 |
| | IQ2_S | 256 | 82 | 2 |
| | IQ3_XXS | 256 | 98 | 3 |
| | IQ3_S | 256 | 110 | 3 |
| | IQ4_NL | 32 | 18 | 4 |
| | IQ4_XS | 256 | 136 | 4 |
| **商业格式** | AWQ4 | 128 | 72 | 4 |
| | GPTQ4 | 128 | 72 | 4 |
| | SqueezeLLM | 256 | 130 | 3 |

### 4.3 量化 GEMV/GEMM 算子

| 算子 | 权重格式 | 输入精度 |
|------|----------|----------|
| gemv_q8 | INT8 | E: f32/f16/bf16 |
| gemv_q4 | INT4 packed | E: f32/f16/bf16 |
| gemv_q2 | INT2 packed | E: f32/f16/bf16 |
| gemv_q1 | INT1 packed | E: f32/f16/bf16 |
| gemm_q8 | INT8 | E: f32/f16/bf16 |
| gemm_q4 | INT4 packed | E: f32/f16/bf16 |

### 4.4 量化格式专用 Matmul

| 算子 | 量化格式 | 说明 |
|------|----------|------|
| `kquant_matmul<E>` | Q2_K ~ Q8_K | K-Quant 系列融合解量化+matmul |
| `iq_matmul<E>` | IQ1_S ~ IQ4_XS | IQ 系列融合解量化+matmul |
| `awq_matmul<E>` | AWQ4 | AWQ 融合解量化+matmul |
| `gptq_matmul<E>` | GPTQ4 | GPTQ 融合解量化+matmul |
| `squeeze_matmul<E>` | SqueezeLLM | SqueezeLLM 融合解量化+matmul |

---

## 5. 性能目标（PERF-TARGETS）🚨 铁律

### 5.1 性能达标基准

| 算子类别 | 瓶颈类型 | 目标 | 参考基准 |
|----------|----------|------|----------|
| **GEMM (compute-bound)** | 计算密集 | 逼近理论 FLOPS 峰值 | MKL/OpenBLAS 同规模 |
| **GEMV (memory-bound)** | 内存带宽 | 逼近带宽峰值 | STREAM benchmark |
| **激活/归一化 (memory-bound)** | 内存带宽 | 逼近带宽峰值 | 单次遍历理论值 |
| **量化 GEMV/GEMM** | 混合瓶颈 | 逼近瓶颈极限 | llama.cpp 同格式 |
| **解量化** | 内存带宽 | 逼近带宽峰值 | 理论解码吞吐 |

### 5.2 性能分析方法论

```
算子瓶颈判定：
  Arithmetic Intensity (AI) = FLOPs / Bytes

  AI > Machine Balance → Compute-bound → 目标: FLOPS 利用率
  AI < Machine Balance → Memory-bound  → 目标: 带宽利用率

  Machine Balance = Peak FLOPS / Peak Bandwidth
  典型值：
    AVX2 (Zen4):    ~8 FLOP/Byte
    AVX-512 (SPR):  ~12 FLOP/Byte
    NEON (M2):      ~6 FLOP/Byte
```

### 5.3 GEMM 性能公式

```
理论峰值 GFLOPS = 频率(GHz) × SIMD宽度 × 2(FMA) × 核心数

效率 = 实测 GFLOPS / 理论峰值 GFLOPS

影响效率的因素：
  1. 微内核寄存器利用率（累加器占比）
  2. Cache 分块命中率（L1/L2/L3 三级）
  3. 尾部处理开销（M/N/K 非 tile 倍数）
  4. 多线程负载均衡
```

---

## 6. 量化类型定义（DATA-QUANT）

```rust
/// 量化类型枚举
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantType {
    // K-Quant
    Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
    // IQ
    IQ1S, IQ1M, IQ2XXS, IQ2XS, IQ2S, IQ3XXS, IQ3S, IQ4NL, IQ4XS,
    // 商业
    AWQ4, GPTQ4, Squeeze,
}

impl QuantType {
    /// 每块元素数
    pub const fn block_size(self) -> usize {
        match self {
            Self::IQ4NL => 32,
            _ => 256,
        }
    }

    /// 每块字节数
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ1S => 50,
            Self::IQ1M => 56,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ4NL => 18,
            Self::IQ4XS => 136,
            Self::AWQ4 | Self::GPTQ4 => 72,
            Self::Squeeze => 130,
        }
    }

    /// 有效位宽
    pub const fn bits(self) -> u8 {
        match self {
            Self::IQ1S | Self::IQ1M => 1,
            Self::Q2K | Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 2,
            Self::Q3K | Self::IQ3XXS | Self::IQ3S | Self::Squeeze => 3,
            Self::Q4K | Self::IQ4NL | Self::IQ4XS | Self::AWQ4 | Self::GPTQ4 => 4,
            Self::Q5K => 5,
            Self::Q6K => 6,
            Self::Q8K => 8,
        }
    }
}
```

---

## 7. 完整展开树（DATA-TREE）

```
CpuKernels
│
├─► x86_64 (#[cfg(target_arch = "x86_64")])
│   └─► ISA (运行时检测)
│       ├─► Scalar   → 兜底（仅限无 SIMD 硬件）
│       ├─► AVX2     → 256-bit SIMD
│       │   └─► 手写汇编微内核: GEMM, 量化 GEMV/GEMM
│       ├─► AVX-512  → 512-bit SIMD
│       │   └─► 手写汇编微内核: GEMM, 量化 GEMV/GEMM
│       └─► VNNI     → INT8 点积加速
│
├─► ARM (#[cfg(target_arch = "aarch64")])
│   └─► ISA (运行时检测)
│       ├─► NEON     → 128-bit SIMD (基线)
│       │   └─► 手写汇编微内核: GEMM, 量化 GEMV/GEMM
│       ├─► dotprod  → INT8 点积
│       └─► SVE      → 可变宽度 SIMD (规划中)
│
│   每个 ISA 实现：
│   └─► impl<E: Element>
│       ├── E = f32  (编译时展开)
│       ├── E = f16  (编译时展开)
│       └── E = bf16 (编译时展开)
│       └── [45 个算子模板]
```

---

## 8. 宏驱动零成本代码生成（ARCH-MACRO）🚨 核心策略

### 8.1 设计原则

**问题**：ISA × 精度 × 量化格式 的组合爆炸

```
CPU 最坏情况：
- ISA:  Scalar, AVX2, AVX-512, NEON, VNNI, ... ≈ 6
- 精度: f32, f16, bf16 = 3
- 量化: 18 种格式
- 算子: 45 个模板

暴力实现: 6 × 3 × 45 = 810+ 函数（不含量化组合）
```

**解法**：宏驱动代码生成 + 手写汇编微内核覆写，零性能妥协

```
┌─────────────────────────────────────────────────────────────────┐
│  simd_primitive! 宏                                             │
│  ─────────────────────────────────────────────────────────────  │
│  定义 ISA × 精度 的原子操作映射表                                │
│  (avx2, f32, fma, a, b, c) → _mm256_fmadd_ps(a, b, c)          │
│  (neon, f32, fma, a, b, c) → vfmaq_f32(c, a, b)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  define_xxx! 算子模板宏                                         │
│  ─────────────────────────────────────────────────────────────  │
│  用 simd_primitive! 编写一次算子逻辑                            │
│  define_vec_dot!(avx2, f32) → 展开为 AVX2 f32 实现              │
│  define_vec_dot!(neon, f16) → 展开为 NEON f16 实现              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  quant_primitive! / decode_block!                               │
│  ─────────────────────────────────────────────────────────────  │
│  量化特化原语（位解包/码本查表/On-the-fly 解量化）              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  批量展开 + 汇编覆写                                            │
│  ─────────────────────────────────────────────────────────────  │
│  mod avx2_f32  { define_vec_dot!(avx2, f32);  ... }            │
│  mod avx2_f32  { pub fn gemm_ukernel() { global_asm!(...) } }  │
│  ...                                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 算子分类判断规则（MACRO-CLASSIFY）🚨 关键

> AI CODER 必须理解：**量化模型只量化权重，不量化激活值**

```
神经网络计算本质：

  输入激活 (f32/f16/bf16)  ───┐
                              ├──► 算子 ──► 输出激活 (f32/f16/bf16)
  权重 (f32 或 量化格式)  ────┘

关键洞察：
  • 激活值 = 中间计算结果 → 永远是浮点（f32/f16/bf16）
  • 权重 = 模型参数 → 可能是浮点，也可能是量化格式（Q4_K, AWQ4, ...）
```

#### 判断流程图

```
                    ┌─────────────────────────┐
                    │ 新算子签名中有权重参数吗？ │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │ 无权重参数                         │ 有权重参数
              │ (只有激活值输入)                   │
              ▼                                   ▼
        ┌───────────┐                   ┌─────────────────────┐
        │  表 A     │                   │ 权重是什么类型？      │
        │ 纯浮点算子 │                   └──────────┬──────────┘
        └───────────┘                              │
                                    ┌──────────────┴──────────────┐
                                    │ 浮点权重 (&[E])              │ 量化权重 (&[u8])
                                    ▼                             ▼
                              ┌───────────┐               ┌─────────────────┐
                              │  表 A     │               │ 输出是什么？      │
                              │ 纯浮点算子 │               └────────┬────────┘
                              └───────────┘                        │
                                                    ┌──────────────┴──────────────┐
                                                    │ 输出是 f32                   │ 输出是 E (激活)
                                                    │ (纯解量化)                   │ (量化计算)
                                                    ▼                             ▼
                                              ┌───────────┐               ┌───────────┐
                                              │  表 B     │               │  表 C     │
                                              │ 解量化算子 │               │ 量化计算   │
                                              └───────────┘               └───────────┘
```

#### 签名特征速查表

| 分类 | 权重参数 | 输入类型 | 输出类型 | 示例签名 |
|------|----------|----------|----------|----------|
| **表 A** | 无 或 `&[E]` | `&[E]` | `E` 或 `&mut [E]` | `fn silu(x: &[E], out: &mut [E])` |
| **表 A** | `&[E]` | `&[E]` | `&mut [E]` | `fn gemv(w: &[E], x: &[E], y: &mut [E], ...)` |
| **表 B** | `&[u8]` | - | `&mut [f32]` | `fn dequant_q4_k(block: &[u8], out: &mut [f32])` |
| **表 C** | `&[u8]`/`&[i8]` | `&[E]` | `E` 或 `&mut [E]` | `fn gemv_q4(w: &[u8], x: &[E], scale: f32) -> E` |

#### 快速判断口诀

```
1. 看签名有没有 &[u8] 或 &[i8] 作为权重 → 有则是量化相关（表 B/C）
2. 量化相关中，输出是 &mut [f32] 固定 → 表 B（纯解量化）
3. 量化相关中，输出是 &mut [E] 泛型 → 表 C（量化计算）
4. 其余全是表 A（纯浮点）
```

---

### 8.3 算子分类表（MACRO-OPS-TABLE）

#### 表 A：纯浮点算子（27 个）

> 输入/输出都是激活值（或浮点权重），只需 ISA × 精度 展开

| 类别 | 算子 | 展开维度 | 组合数 |
|------|------|----------|--------|
| **向量运算** | vec_dot, vec_add, vec_sub, vec_mul, vec_scale, vec_axpy, vec_sum, vec_max, vec_sum_squares | ISA × 精度 | 9×6×3=162 |
| **矩阵运算** | gemv, gemm, gemm_bias, gemm_bias_act, pack_b, gemm_prepacked, gemm_bias_prepacked | ISA × 精度 | 7×6×3=126 |
| **激活函数** | silu, gelu, relu, tanh, swiglu, softmax, exp | ISA × 精度 | 7×6×3=126 |
| **归一化** | rms_norm, layer_norm | ISA × 精度 | 2×6×3=36 |
| **位置编码** | rope, rope_with_pos | ISA × 精度 | 2×6×3=36 |
| **小计** | | | **~486** |

**宏策略**：`define_xxx!(isa, elem)` 模板，一次定义 27 个模板，批量展开

#### 表 B：解量化算子（18 个）

> 输入是量化块 `&[u8]`，输出固定为 `f32`，只需 ISA 展开

| 格式 | 算子 | 展开维度 | 组合数 |
|------|------|----------|--------|
| **K-Quant** | dequant_q2_k, dequant_q3_k, dequant_q4_k, dequant_q5_k, dequant_q6_k, dequant_q8_k | ISA | 6×6=36 |
| **IQ 系列** | dequant_iq1_s, dequant_iq1_m, dequant_iq2_xxs, dequant_iq2_xs, dequant_iq2_s, dequant_iq3_xxs, dequant_iq3_s, dequant_iq4_nl, dequant_iq4_xs | ISA | 9×6=54 |
| **商业格式** | dequant_awq4, dequant_gptq4, dequant_squeeze | ISA | 3×6=18 |
| **小计** | | | **~108** |

**宏策略**：`decode_block!(quant_fmt, block, out)` 解码逻辑独立，SIMD 存储共用

#### 表 C：量化计算算子（11 个）

> 权重是量化格式，输入是浮点，需要 ISA × 输入精度 × 量化格式 展开

| 类别 | 算子 | 展开维度 | 组合数 |
|------|------|----------|--------|
| **通用量化 GEMV** | gemv_q8, gemv_q4, gemv_q2, gemv_q1 | ISA × 精度 | 4×6×3=72 |
| **通用量化 GEMM** | gemm_q8, gemm_q4 | ISA × 精度 | 2×6×3=36 |
| **格式专用 Matmul** | kquant_matmul, iq_matmul, awq_matmul, gptq_matmul, squeeze_matmul | ISA × 精度 × 格式子集 | ~90 |
| **小计** | | | **~198** |

**宏策略**：
```rust
macro_rules! define_quant_gemv {
    ($isa:ident, $input_elem:ty, $quant_fmt:ident, $block_size:expr) => {
        // 主循环共用，decode_block! 分发格式差异
    };
}
```

### 8.4 量化宏详细设计（MACRO-QUANT-DESIGN）🚨 核心

> 量化算子的宏化是整个架构最复杂的部分，需要处理 **18 种格式 × 6 ISA × 3 精度** 的组合。

#### 8.4.1 量化原语表（quant_primitive!）

```rust
/// 量化专用原语 - 与 simd_primitive! 配合使用
///
/// 核心操作：位解包、查表、scale 应用
macro_rules! quant_primitive {
    // ========================================================================
    // INT4 解包（每 u8 包含 2 个 4-bit 值）
    // ========================================================================

    // AVX2: 一次解包 32 个 INT4 → 32 个 f32
    (avx2, unpack_int4, $packed:expr) => {{
        let lo_mask = _mm256_set1_epi8(0x0F);
        let lo = _mm256_and_si256($packed, lo_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16($packed, 4), lo_mask);
        _mm256_unpacklo_epi8(lo, hi)
    }};

    // AVX-512: 一次解包 64 个 INT4 → 64 个 f32
    (avx512, unpack_int4, $packed:expr) => {{
        let lo_mask = _mm512_set1_epi8(0x0F);
        let lo = _mm512_and_si512($packed, lo_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16($packed, 4), lo_mask);
        _mm512_unpacklo_epi8(lo, hi)
    }};

    // NEON: 一次解包 16 个 INT4 → 16 个 f32
    (neon, unpack_int4, $packed:expr) => {{
        let lo_mask = vdupq_n_u8(0x0F);
        let lo = vandq_u8($packed, lo_mask);
        let hi = vandq_u8(vshrq_n_u8($packed, 4), lo_mask);
        vzip1q_u8(lo, hi)
    }};

    // Scalar: 逐个解包
    (scalar, unpack_int4, $byte:expr, $idx:expr) => {{
        if $idx & 1 == 0 { ($byte & 0x0F) as i8 } else { (($byte >> 4) & 0x0F) as i8 }
    }};

    // ========================================================================
    // INT2 解包（每 u8 包含 4 个 2-bit 值）
    // ========================================================================

    (avx2, unpack_int2, $packed:expr) => {{
        let mask = _mm256_set1_epi8(0x03);
        let v0 = _mm256_and_si256($packed, mask);
        let v1 = _mm256_and_si256(_mm256_srli_epi16($packed, 2), mask);
        let v2 = _mm256_and_si256(_mm256_srli_epi16($packed, 4), mask);
        let v3 = _mm256_and_si256(_mm256_srli_epi16($packed, 6), mask);
        (v0, v1, v2, v3)
    }};

    (scalar, unpack_int2, $byte:expr, $idx:expr) => {{
        (($byte >> (($idx & 3) * 2)) & 0x03) as i8
    }};

    // ========================================================================
    // INT1 解包（每 u8 包含 8 个 1-bit 值）
    // ========================================================================

    (scalar, unpack_int1, $byte:expr, $idx:expr) => {{
        (($byte >> ($idx & 7)) & 1) as i8
    }};

    // ========================================================================
    // Scale 应用（解量化核心）
    // ========================================================================

    (avx2, f32, apply_scale, $int_vec:expr, $scale:expr, $zero:expr) => {{
        let float_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32($int_vec));
        let zero_vec = _mm256_set1_ps($zero);
        let scale_vec = _mm256_set1_ps($scale);
        _mm256_mul_ps(_mm256_sub_ps(float_vec, zero_vec), scale_vec)
    }};

    (scalar, f32, apply_scale, $int_val:expr, $scale:expr, $zero:expr) => {{
        (($int_val as f32) - $zero) * $scale
    }};

    // ========================================================================
    // IQ 码本查表（IQ 系列专用）
    // ========================================================================

    (any, iq1_lookup, $grid_idx:expr) => {{ IQ1_S_GRID[$grid_idx as usize] }};
    (any, iq2_xxs_lookup, $grid_idx:expr) => {{ IQ2_XXS_GRID[$grid_idx as usize] }};
    (any, iq3_xxs_lookup, $grid_idx:expr) => {{ IQ3_XXS_GRID[$grid_idx as usize] }};
    (any, iq4_nl_lookup, $idx:expr) => {{ IQ4_NL_GRID[$idx as usize] }};
}
```

#### 8.4.2 块解码宏（decode_block!）

```rust
/// 块解码宏 - 每种量化格式的解码逻辑
///
/// 输入: 原始字节块 &[u8]
/// 输出: 解量化后的 f32 数组
///
/// 关键：解码逻辑与 ISA 无关，只有存储操作用 simd_primitive!
macro_rules! decode_block {
    // K-Quant 系列
    (q4_k, $isa:ident, $block:expr, $out:expr) => {{ /* 144 bytes */ }};
    (q8_k, $isa:ident, $block:expr, $out:expr) => {{ /* 292 bytes */ }};
    (q2_k, $isa:ident, $block:expr, $out:expr) => {{ /* 84 bytes */ }};
    (q3_k, $isa:ident, $block:expr, $out:expr) => {{ /* 110 bytes */ }};
    (q5_k, $isa:ident, $block:expr, $out:expr) => {{ /* 176 bytes */ }};
    (q6_k, $isa:ident, $block:expr, $out:expr) => {{ /* 210 bytes */ }};

    // IQ 系列
    (iq1_s, $isa:ident, $block:expr, $out:expr) => {{ /* IQ1_S_GRID 查表 */ }};
    (iq4_nl, $isa:ident, $block:expr, $out:expr) => {{ /* IQ4_NL_GRID 查表 */ }};
    // ... 其他 IQ 格式

    // 商业格式
    (awq4, $isa:ident, $packed:expr, $zeros:expr, $scales:expr, $out:expr, $group_idx:expr) => {{ /* ... */ }};
    (gptq4, $isa:ident, $packed:expr, $g_idx:expr, $scales:expr, $out:expr) => {{ /* ... */ }};
}
```

#### 8.4.3 量化 GEMV 模板（define_quant_gemv!）

```rust
/// 量化 GEMV 模板 - 融合解量化 + 矩阵向量乘法
///
/// 核心优化：
/// 1. 不生成完整 f32 矩阵（On-the-fly dequantization）
/// 2. 块级解码，L1 Cache 友好
/// 3. 输入向量 SIMD 广播复用
macro_rules! define_quant_gemv {
    ($isa:ident, $input_elem:ty, $quant_fmt:ident, $block_size:expr) => {
        #[inline(always)]
        pub fn gemv(
            weight_blocks: &[u8],
            input: &[$input_elem],
            output: &mut [f32],
            m: usize, k: usize,
        ) {
            const BLOCK_SIZE: usize = $block_size;
            const BLOCK_BYTES: usize = block_bytes!($quant_fmt);
            let blocks_per_row = k / BLOCK_SIZE;

            let mut dequant_buf: [f32; BLOCK_SIZE] = [0.0; BLOCK_SIZE];

            for row in 0..m {
                let mut acc = simd_primitive!($isa, f32, zero);

                for blk_idx in 0..blocks_per_row {
                    let blk_offset = (row * blocks_per_row + blk_idx) * BLOCK_BYTES;
                    let block = &weight_blocks[blk_offset..blk_offset + BLOCK_BYTES];

                    decode_block!($quant_fmt, $isa, block, &mut dequant_buf);

                    let input_offset = blk_idx * BLOCK_SIZE;
                    for i in (0..BLOCK_SIZE).step_by(simd_primitive!($isa, f32, lanes)) {
                        let w = simd_primitive!($isa, f32, load, dequant_buf[i..].as_ptr());
                        let x = simd_primitive!($isa, $input_elem, load_cvt,
                                               input[input_offset + i..].as_ptr());
                        acc = simd_primitive!($isa, f32, fma, w, x, acc);
                    }
                }

                output[row] = simd_primitive!($isa, f32, reduce_sum, acc);
            }
        }
    };
}
```

#### 8.4.4 量化格式常量表（QUANT-CONST-TABLE）

```rust
macro_rules! block_bytes {
    (q2_k)    => { 84 };    (q3_k)    => { 110 };
    (q4_k)    => { 144 };   (q5_k)    => { 176 };
    (q6_k)    => { 210 };   (q8_k)    => { 292 };
    (iq1_s)   => { 50 };    (iq1_m)   => { 56 };
    (iq2_xxs) => { 66 };    (iq2_xs)  => { 74 };
    (iq2_s)   => { 82 };    (iq3_xxs) => { 98 };
    (iq3_s)   => { 110 };   (iq4_nl)  => { 18 };
    (iq4_xs)  => { 136 };   (awq4)    => { 72 };
    (gptq4)   => { 72 };    (squeeze) => { 130 };
}

macro_rules! block_size {
    (iq4_nl) => { 32 };
    ($other:ident) => { 256 };
}
```

#### 8.4.5 批量展开量化算子

```rust
macro_rules! expand_all_quant_formats {
    ($macro_name:ident, $isa:ident, $elem:ty) => {
        mod q2_k  { $macro_name!($isa, $elem, q2_k, 256); }
        mod q3_k  { $macro_name!($isa, $elem, q3_k, 256); }
        mod q4_k  { $macro_name!($isa, $elem, q4_k, 256); }
        mod q5_k  { $macro_name!($isa, $elem, q5_k, 256); }
        mod q6_k  { $macro_name!($isa, $elem, q6_k, 256); }
        mod q8_k  { $macro_name!($isa, $elem, q8_k, 256); }
        mod iq1_s { $macro_name!($isa, $elem, iq1_s, 256); }
        mod iq4_nl { $macro_name!($isa, $elem, iq4_nl, 32); }
        // ... 其他 10 种格式
    };
}

macro_rules! expand_quant_kernels {
    () => {
        #[cfg(target_arch = "x86_64")]
        mod avx2 {
            mod f32 { expand_all_quant_formats!(define_quant_gemv, avx2, f32); }
            mod f16 { expand_all_quant_formats!(define_quant_gemv, avx2, f16); }
        }
        #[cfg(target_arch = "x86_64")]
        mod avx512 {
            mod f32 { expand_all_quant_formats!(define_quant_gemv, avx512, f32); }
        }
        #[cfg(target_arch = "aarch64")]
        mod neon {
            mod f32 { expand_all_quant_formats!(define_quant_gemv, neon, f32); }
        }
        mod scalar {
            mod f32 { expand_all_quant_formats!(define_quant_gemv, scalar, f32); }
        }
    };
}
```

#### 8.4.6 IQ 码本常量

```rust
// IQ4_NL: 16 个非线性量化值（llama.cpp 标准）
pub static IQ4_NL_GRID: [f32; 16] = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
];

// IQ1_S, IQ2_XXS, IQ3_XXS 等码本从 llama.cpp 移植
pub static IQ1_S_GRID: [f32; 2048] = [ /* ... */ ];
pub static IQ2_XXS_GRID: [[f32; 8]; 256] = [ /* ... */ ];
```

### 8.5 simd_primitive! 完整映射表（MACRO-PRIMITIVE-COMPLETE）🚨 核心维护点

> **AI CODER 注意**：这是整个宏架构的核心！添加新 ISA 只需扩展此表。

#### 8.5.1 操作清单（每个 ISA × 精度 组合必须实现）

**A. 计算操作（22 个）**

| 操作 | 签名 | 说明 |
|------|------|------|
| `lanes` | `() -> usize` | SIMD 向量宽度（编译时常量） |
| `zero` | `() -> Vec` | 零向量 |
| `splat` | `(val) -> Vec` | 标量广播到所有通道 |
| `load` / `loadu` | `(ptr) -> Vec` | 从内存加载（对齐/非对齐） |
| `store` / `storeu` | `(ptr, vec)` | 存储到内存（对齐/非对齐） |
| `stream` | `(ptr, vec)` | NT 存储（绕过 Cache） |
| `maskload` | `(ptr, count) -> Vec` | 带掩码加载（尾部处理） |
| `maskstore` | `(ptr, count, vec)` | 带掩码存储（尾部处理） |
| `load_cvt` | `(ptr) -> Vec<f32>` | 加载 f16/bf16 并转换为 f32 |
| `store_cvt` | `(ptr, vec)` | 将 f32 转换并存储为 f16/bf16 |
| `add` | `(a, b) -> Vec` | 向量加法 |
| `sub` | `(a, b) -> Vec` | 向量减法 |
| `mul` | `(a, b) -> Vec` | 向量乘法 |
| `div` | `(a, b) -> Vec` | 向量除法 |
| `fma` | `(a, b, c) -> Vec` | 融合乘加：c + a * b |
| `neg` | `(a) -> Vec` | 取负 |
| `max` | `(a, b) -> Vec` | 逐元素最大 |
| `min` | `(a, b) -> Vec` | 逐元素最小 |
| `reduce_sum` | `(vec) -> Scalar` | 水平求和 |
| `reduce_max` | `(vec) -> Scalar` | 水平最大 |
| `exp` | `(a) -> Vec` | 指数函数 |
| `recip` | `(a) -> Vec` | 倒数 (1/x) |
| `sqrt` | `(a) -> Vec` | 平方根 |
| `rsqrt` | `(a) -> Vec` | 倒数平方根 |
| `prefetch` | `(ptr, distance)` | 软件预取到 L1 Cache |

**B. 架构常量（🚨 性能最大化关键 — 算子模板据此生成结构不同的微内核）**

| 常量 | 说明 | Scalar | AVX2 | AVX-512 | NEON |
|------|------|--------|------|---------|------|
| `num_regs` | 可用 SIMD 寄存器数 | ∞ | 16 | 32 | 32 |
| `optimal_tile_m` | GEMM 微内核行数 | 1 | 6 | 14 | 8 |
| `optimal_tile_n_vecs` | GEMM 微内核列向量数 | 1 | 2 | 2 | 3 |
| `prefetch_distance` | 预取字节距离 | 0 | 256 | 512 | 128 |
| `has_native_fp16` | 是否支持原生 f16 运算 | false | false | * | true |
| `has_native_bf16` | 是否支持原生 bf16 点积 | false | false | * | false |
| `has_vnni` | 是否支持 INT8 点积加速 | false | false | * | false |
| `has_dot_prod` | ARM dotprod 支持 | false | - | - | * |

> `*` = 运行时检测子特性（如 AVX512-FP16 需要额外检测 `is_x86_feature_detected!("avx512fp16")`）

**设计意图**：`define_gemm!($isa, $elem)` 内部通过 `simd_primitive!($isa, $elem, optimal_tile_m)` 获取最优分块因子，使得 AVX2 展开为 6×16 微内核、AVX-512 展开为 14×32 微内核——**循环结构本身随 ISA 变化**，而非只替换指令。

### 8.6 ISA × 精度 支持矩阵

| ISA | f32 | f16 | bf16 | 说明 |
|-----|-----|-----|------|------|
| **Scalar** | ✅ 原生 | ✅ 软件转换 | ✅ 软件转换 | 兜底（仅限无 SIMD 硬件） |
| **AVX2** | ✅ 原生 | ✅ F16C 转换 | ✅ 位转换 | x86_64 基线 |
| **AVX-512** | ✅ 原生 | ⚡ AVX512-FP16 | ⚡ AVX512-BF16 | 需运行时检测扩展 |
| **VNNI** | - | - | - | INT8 点积加速 |
| **NEON** | ✅ 原生 | ⚡ FP16 原生 | ✅ 位转换 | ARM 基线 |
| **SVE** | ✅ 原生 | ⚡ FP16 原生 | ⚡ BF16 原生 | ARM 服务器（规划中） |

**图例**：✅ 必须实现 | ⚡ 硬件原生支持 | - 不适用

---

## 9. 手写汇编微内核架构（ARCH-ASM-UKERNEL）🚨 性能核心

### 9.1 设计原则

**为什么必须手写汇编**：

编译器（LLVM）在以下场景无法生成最优代码：
1. **寄存器分配**：GEMM 微内核需要精确控制累加器寄存器，编译器的寄存器分配器无法保证零溢出
2. **指令调度**：FMA 流水线延迟隐藏需要精确的指令交错，编译器的调度器不够激进
3. **预取插入**：软件预取的位置和距离需要根据微架构精确调整
4. **量化解包**：位操作序列有特定的最优指令选择，编译器可能选择次优路径

**强制规则**：
- GEMM 微内核（内层循环）：**必须手写汇编**
- 量化 GEMV/GEMM 的内层点积：**必须手写汇编**
- 其他算子（激活/归一化/BLAS-1）：宏生成 intrinsic 即可，编译器能处理好

### 9.2 汇编微内核接口约定（ASM-UKERNEL-ABI）

#### 9.2.1 Rust 集成方式

```rust
// 方式 1: global_asm! — 完整汇编文件嵌入（推荐用于大型微内核）
use std::arch::global_asm;

global_asm!(
    include_str!("asm/avx2_f32_gemm_6x16.S"),
    options(att_syntax)  // 或 intel_syntax
);

extern "C" {
    /// AVX2 f32 GEMM 6x16 微内核
    /// 计算 C[6×16] += A[6×k] * B[k×16]，k 步迭代
    fn gk_gemm_avx2_f32_6x16(
        k: usize,
        a: *const f32,       // A 面板指针，行主序，lda = k
        b: *const f32,       // B 面板指针，已 pack 为列主序 [k][16]
        c: *mut f32,         // C 输出指针，行主序，ldc = n
        ldc: usize,          // C 的列步长（字节或元素数，按约定）
        alpha: f32,          // 缩放因子（通常 1.0）
    );
}

// 方式 2: naked_fn — 小型微内核（Rust nightly）
#[naked]
#[target_feature(enable = "avx2,fma")]
unsafe extern "C" fn gk_gemm_avx2_f32_6x16(
    k: usize, a: *const f32, b: *const f32,
    c: *mut f32, ldc: usize, alpha: f32,
) {
    core::arch::asm!(
        // ... 汇编指令 ...
        options(noreturn)
    );
}
```

#### 9.2.2 命名约定

```
gk_{op}_{isa}_{elem}_{tile}

gk_       — gllm-kernels 前缀
{op}      — 操作名: gemm, gemv, qdot (量化点积)
{isa}     — ISA: avx2, avx512, neon
{elem}    — 精度: f32, f16, bf16, i8
{tile}    — 微内核尺寸: 6x16, 14x32, 8x12

示例：
  gk_gemm_avx2_f32_6x16      — AVX2 f32 GEMM 6行×16列微内核
  gk_gemm_avx512_f32_14x32   — AVX-512 f32 GEMM 14行×32列微内核
  gk_gemm_neon_f32_8x12      — NEON f32 GEMM 8行×12列微内核
  gk_qdot_avx2_q4k_f32       — AVX2 Q4_K 量化点积（输入 f32）
  gk_qdot_avx512_q8k_f32     — AVX-512 Q8_K 量化点积
```

#### 9.2.3 调用约定

所有汇编微内核使用 **C ABI** (`extern "C"`)，参数传递遵循平台 ABI：

| 平台 | 整数/指针参数 | 浮点参数 | 返回值 |
|------|--------------|----------|--------|
| **x86_64 SysV** | rdi, rsi, rdx, rcx, r8, r9 | xmm0-xmm7 | rax / xmm0 |
| **aarch64** | x0-x7 | v0-v7 (标量部分) | x0 / v0 |

**寄存器使用约定**（x86_64 GEMM 微内核）：

```
被调用者保存（callee-saved）：rbx, rbp, r12-r15
  → 微内核如果使用这些寄存器，必须 push/pop

调用者保存（caller-saved）：rax, rcx, rdx, rsi, rdi, r8-r11
  → 微内核可以自由使用

SIMD 寄存器：
  AVX2:    ymm0-ymm15 全部 caller-saved
  AVX-512: zmm0-zmm31 全部 caller-saved
  → 微内核可以自由使用所有 SIMD 寄存器
```

### 9.3 GEMM 汇编微内核设计

#### 9.3.1 微内核尺寸选择

| ISA | 寄存器数 | SIMD 宽度 | 微内核 (TM×TN) | 累加器数 | 临时寄存器 |
|-----|----------|-----------|----------------|----------|------------|
| **AVX2** | 16 ymm | 8×f32 | 6×16 (6×2vec) | 12 | 4 (A广播+B加载+预取) |
| **AVX-512** | 32 zmm | 16×f32 | 14×32 (14×2vec) | 28 | 4 |
| **NEON** | 32 v-reg | 4×f32 | 8×12 (8×3vec) | 24 | 8 |

**选择原则**：
```
累加器数 = TM × (TN / LANES) = TM × NV
临时寄存器 ≥ 3（1个A广播 + NV个B加载）
总寄存器 = 累加器 + 临时 ≤ 可用寄存器数

最大化 TM × TN 以提高计算/访存比
```

#### 9.3.2 AVX2 f32 6×16 微内核伪代码

```asm
; gk_gemm_avx2_f32_6x16
; 输入: k(rdi), a(rsi), b(rdx), c(rcx), ldc(r8), alpha(xmm0)
;
; 寄存器分配:
;   ymm0-ymm11:  6×2 = 12 个累加器 (c_i_j)
;   ymm12:       A 元素广播
;   ymm13-ymm14: B 列向量加载
;   ymm15:       临时/预取

    ; 初始化 12 个累加器为零
    vxorps ymm0, ymm0, ymm0    ; c_0_0
    vxorps ymm1, ymm1, ymm1    ; c_0_1
    ; ... ymm2-ymm11

    ; K 循环
.Lk_loop:
    ; 加载 B 的两个向量 (16 个 f32)
    vmovups ymm13, [rdx]        ; B[k][0:8]
    vmovups ymm14, [rdx + 32]   ; B[k][8:16]

    ; 预取下一个 B 面板
    prefetcht0 [rdx + 256]

    ; 对 A 的每一行广播并 FMA
    vbroadcastss ymm12, [rsi]           ; A[0][k]
    vfmadd231ps  ymm0, ymm12, ymm13    ; c_0_0 += A[0][k] * B[k][0:8]
    vfmadd231ps  ymm1, ymm12, ymm14    ; c_0_1 += A[0][k] * B[k][8:16]

    vbroadcastss ymm12, [rsi + 4]       ; A[1][k]
    vfmadd231ps  ymm2, ymm12, ymm13
    vfmadd231ps  ymm3, ymm12, ymm14

    ; ... A[2]-A[5] 同理 (ymm4-ymm11)

    ; 步进
    add rsi, 24        ; A 面板: 6 个 f32 = 24 bytes
    add rdx, 64        ; B 面板: 16 个 f32 = 64 bytes
    dec rdi
    jnz .Lk_loop

    ; 写回 C（可选 alpha 缩放）
    ; vmovups [rcx], ymm0
    ; vmovups [rcx + 32], ymm1
    ; ... 按 ldc 步进写回 6 行
    ret
```

#### 9.3.3 宏生成的外层循环 + 汇编微内核

```rust
/// GEMM 外层循环（宏生成）调用汇编微内核（手写）
///
/// 三层分块: MC × KC × NC
///   MC: A 面板行数（适配 L2 Cache）
///   KC: 公共维度分块（适配 L1 Cache）
///   NC: B 面板列数（适配 L3 Cache）
macro_rules! define_gemm_driver {
    ($isa:ident, $elem:ty, $TM:literal, $TN:literal, $ukernel:path) => {
        pub fn gemm(
            a: &[$elem], b: &[$elem], c: &mut [$elem],
            m: usize, n: usize, k: usize,
        ) {
            let bp = blocking_params($TM, $TN / simd_primitive!($isa, $elem, lanes),
                                     simd_primitive!($isa, $elem, lanes),
                                     std::mem::size_of::<$elem>());

            // Pack B into column-panel layout [KC][NC]
            let packed_b = pack_b(b, n, k, bp.kc, $TN);

            // MC loop (over rows of A)
            for mc_start in (0..m).step_by(bp.mc) {
                let mc = bp.mc.min(m - mc_start);

                // KC loop (over common dimension)
                for kc_start in (0..k).step_by(bp.kc) {
                    let kc = bp.kc.min(k - kc_start);

                    // Pack A panel [MC][KC]
                    let packed_a = pack_a(&a, m, k, mc_start, kc_start, mc, kc, $TM);

                    // NC loop (over columns of B) → TM×TN 微内核
                    for nc_start in (0..n).step_by($TN) {
                        let nc = $TN.min(n - nc_start);

                        // TM loop (over micro-rows)
                        for mr in (0..mc).step_by($TM) {
                            let tm = $TM.min(mc - mr);
                            if tm == $TM && nc == $TN {
                                // 完整微内核：调用手写汇编
                                unsafe {
                                    $ukernel(
                                        kc,
                                        packed_a[mr * kc..].as_ptr(),
                                        packed_b[nc_start * kc..].as_ptr(),
                                        c[(mc_start + mr) * n + nc_start..].as_mut_ptr(),
                                        n,  // ldc
                                        1.0,
                                    );
                                }
                            } else {
                                // 尾部处理：标量或 masked SIMD
                                gemm_tail(/* ... */);
                            }
                        }
                    }
                }
            }
        }
    };
}
```

### 9.4 量化汇编微内核设计

#### 9.4.1 量化点积微内核

量化 GEMV/GEMM 的核心是**融合解量化+点积**，在寄存器内完成解包→FMA，不写回中间 f32 矩阵。

```rust
extern "C" {
    /// AVX2 Q4_K 量化点积
    /// 计算 sum(dequant(weight_block) * input_f32)
    /// 一次处理一个 256 元素块
    fn gk_qdot_avx2_q4k_f32(
        block: *const u8,     // Q4_K 块指针 (144 bytes)
        input: *const f32,    // f32 输入向量 (256 elements)
        block_count: usize,   // 块数量
    ) -> f32;                 // 点积结果
}
```

#### 9.4.2 量化微内核与宏的协作

```
宏生成的外层循环（行遍历、块遍历、输出累加）
    │
    └─► 内层调用汇编微内核（单块解量化+点积）
        │
        ├─ gk_qdot_avx2_q4k_f32   — Q4_K 格式
        ├─ gk_qdot_avx2_q8k_f32   — Q8_K 格式
        ├─ gk_qdot_avx2_iq4nl_f32 — IQ4_NL 格式
        └─ ...

每种量化格式 × 每种 ISA = 一个专用汇编微内核
宏负责：行循环、块索引计算、输出写回
汇编负责：单块内的解包+FMA 流水线
```

### 9.5 汇编文件组织

```
src/
├── asm/                          # 手写汇编微内核
│   ├── x86_64/
│   │   ├── avx2_f32_gemm_6x16.S
│   │   ├── avx512_f32_gemm_14x32.S
│   │   ├── avx2_qdot_q4k.S
│   │   ├── avx2_qdot_q8k.S
│   │   ├── avx512_qdot_q4k.S
│   │   └── ...
│   └── aarch64/
│       ├── neon_f32_gemm_8x12.S
│       ├── neon_qdot_q4k.S
│       └── ...
```

### 9.6 汇编覆写规则

| 算子 | 宏生成基线 | 汇编覆写 | 覆写条件 |
|------|-----------|----------|----------|
| **GEMM 微内核** | `define_matmul_x86!` | **强制覆写** | 始终使用汇编 |
| **量化 GEMV 点积** | `define_quant_gemv!` | **强制覆写** | 始终使用汇编 |
| **量化 GEMM 点积** | `define_quant_gemm!` | **强制覆写** | 始终使用汇编 |
| BLAS-1 (vec_dot 等) | `define_blas1_ops!` | 可选覆写 | 基准测试证明 >10% 提升 |
| 激活函数 | `define_element_wise_ops!` | 不覆写 | 编译器足够好 |
| 归一化 | `define_norm_ops!` | 不覆写 | 内存带宽瓶颈 |

---

## 10. AI CODER 维护指南

### 10.1 添加新 ISA

```
□ 步骤 1：扩展 simd_primitive! 表
  ├─ 添加所有 22+ 个操作的映射（见 §8.5.1 操作清单）
  ├─ 每个操作必须有对应的 intrinsic 或软件实现
  └─ 验证：grep -c "(new_isa, f32," 应该 >= 22

□ 步骤 2：扩展 simd_lanes! 宏
  ├─ 添加 (new_isa, f32), (new_isa, f16), (new_isa, bf16) 三条规则
  └─ 验证：编译通过

□ 步骤 3：扩展 simd_align! 宏
  └─ 添加 (new_isa, $elem:ty) => { 对齐字节数 }

□ 步骤 4：更新 expand_all_isa! 宏
  ├─ 添加 #[cfg(target_arch = "xxx")] mod new_isa { ... }
  └─ 验证：所有算子自动获得新 ISA 支持

□ 步骤 5：编写汇编微内核
  ├─ GEMM 微内核（必须）
  ├─ 量化点积微内核（必须）
  └─ 放置于 src/asm/{arch}/ 目录

□ 步骤 6：更新 §8.6 ISA × 精度 支持矩阵
  └─ 添加新行，标注支持的精度和硬件特性

□ 步骤 7：测试
  ├─ cargo test --features new_isa
  ├─ 正确性：与 scalar 实现对比
  └─ 性能：基准测试验证达标（§5 性能目标）
```

### 10.2 添加新量化格式

```
□ 步骤 1：定义格式常量
  ├─ 在 block_bytes! 宏中添加 (new_fmt) => { 字节数 }
  ├─ 在 block_size! 宏中添加（如果不是 256）
  └─ 在 QuantType 枚举中添加新变体

□ 步骤 2：实现 decode_block! 规则
  ├─ 添加 (new_fmt, $isa:ident, $block:expr, $out:expr) => {{ ... }}
  ├─ 解析块头（scale, zero 等元数据）
  ├─ 使用 quant_primitive! 解包数据
  └─ 验证：与参考实现（llama.cpp）数值一致

□ 步骤 3：添加解量化函数
  ├─ 在 Kernels trait 中添加 fn dequant_new_fmt(...)
  └─ 在各 ISA 实现中调用 decode_block!(new_fmt, ...)

□ 步骤 4：编写汇编量化点积微内核
  ├─ 每个 ISA 一个专用微内核
  └─ 放置于 src/asm/{arch}/

□ 步骤 5：生成量化 GEMV
  ├─ expand_all_quant_formats! 中添加 mod new_fmt { ... }
  └─ 验证：所有 ISA × 精度 组合自动生成

□ 步骤 6：更新 dispatch_quant_type! 宏
  └─ 添加 QuantType::NewFmt => kernels.dequant_new_fmt(...)

□ 步骤 7：测试
  ├─ 单元测试：decode 正确性
  ├─ 集成测试：GEMV 输出与参考一致
  └─ 性能测试：与 llama.cpp 对比，达标 §5 目标
```

### 10.3 添加新算子

```
□ 步骤 1：判断算子类别
  ├─ 签名无量化权重 → 表 A（纯浮点）
  ├─ 输出固定 f32 → 表 B（解量化）
  └─ 量化权重 + 泛型输出 → 表 C（量化计算）

□ 步骤 2：编写算子模板宏
  ├─ 命名：define_xxx!(isa, elem)
  ├─ 使用 simd_primitive! 原语，不直接使用 intrinsic
  ├─ 包含尾部处理（非 LANES 对齐部分）
  └─ 验证：scalar 实现正确

□ 步骤 3：批量展开
  ├─ 在对应模块中调用 expand_all_isa!(define_xxx)
  └─ 验证：编译通过

□ 步骤 4：添加到 Kernels trait
  └─ 在各 ISA 实现中添加调用

□ 步骤 5：判断是否需要汇编覆写
  ├─ 计算密集型（GEMM 类）→ 必须汇编
  ├─ 内存带宽瓶颈 → 不需要
  └─ 基准测试决定

□ 步骤 6：更新 §11 算子统计表
  └─ 更新对应类别数量

□ 步骤 7：测试
  ├─ 正确性测试（与标量/参考实现对比）
  └─ 性能测试（各 ISA 加速比，对照 §5 目标）
```

### 10.4 常见错误检查

```
❌ 错误 1：直接使用 intrinsic 而不是 simd_primitive!
   → 导致新 ISA 无法自动支持
   → 检查：grep -r "_mm256\|_mm512\|vaddq" src/cpu_kernels/*.rs
   → 例外：手写汇编微内核（src/asm/）不受此限制

❌ 错误 2：忘记尾部处理
   → 数组长度非 LANES 倍数时结果错误
   → 检查：所有循环后是否有 remainder 处理

❌ 错误 3：decode_block! 中硬编码 ISA
   → 解码逻辑应该对 ISA 透明
   → 检查：decode_block! 内部只用 quant_primitive! 或标量操作

❌ 错误 4：忘记更新 dispatch_quant_type!
   → 新格式无法被分发
   → 检查：QuantType 枚举和 dispatch 宏分支数一致

❌ 错误 5：f16/bf16 直接计算而不转换
   → Rust 没有 f16 原生算术
   → 检查：f16 操作必须经过 load_cvt/store_cvt

❌ 错误 6：GEMM/量化 GEMV 使用宏生成而非汇编
   → 性能无法达标
   → 检查：GEMM 和量化 GEMV 的内层循环必须调用 gk_* 汇编函数
```

### 10.5 性能验证基准

| 操作 | 期望加速比（vs Scalar） | 性能目标 | 备注 |
|------|------------------------|----------|------|
| GEMM (f32, large) | AVX2: 6-8×, AVX512: 12-16× | 逼近峰值 FLOPS | 汇编微内核 |
| GEMV (f32) | AVX2: 5-7×, AVX512: 10-12× | 逼近带宽峰值 | 内存带宽瓶颈 |
| vec_dot (f32) | AVX2: 6-8×, AVX512: 12-14× | 逼近带宽峰值 | SIMD 宽度 |
| rms_norm | AVX2: 4-6×, AVX512: 8-10× | 逼近带宽峰值 | 两次遍历 |
| softmax | AVX2: 3-5× | 逼近带宽峰值 | exp 近似开销 |
| dequant_q4_k | AVX2: 3-4× | 逼近带宽峰值 | 解码开销 |
| quant_gemv (q4) | AVX2: 4-6×, AVX512: 8-12× | 逼近瓶颈极限 | 汇编微内核 |
| quant_gemm (q4) | AVX2: 5-7×, AVX512: 10-14× | 逼近瓶颈极限 | 汇编微内核 |

---

## 11. 算子统计

### 11.1 算子模板数（需维护）

| 类别 | 数量 | 宏策略 |
|------|------|--------|
| 向量运算 | 9 | 表 A |
| 矩阵运算 | 7 | 表 A（外层宏 + 汇编微内核） |
| 激活函数 | 7 | 表 A |
| 归一化 | 2 | 表 A |
| 位置编码 | 2 | 表 A |
| 解量化 | 18 | 表 B |
| 量化 GEMV/GEMM | 6 | 表 C（外层宏 + 汇编微内核） |
| 量化格式专用 Matmul | 5 | 表 C（外层宏 + 汇编微内核） |
| **模板总计** | **56** | |

### 11.2 宏展开后实现数（自动生成）

| 类别 | 展开公式 | 实现数 |
|------|----------|--------|
| 表 A 纯浮点 | 27 算子 × 6 ISA × 3 精度 | ~486 |
| 表 B 解量化 | 18 格式 × 6 ISA | ~108 |
| 表 C 量化计算 | 11 算子 × 6 ISA × 3 精度 | ~198 |
| **展开总计** | | **~792** |

### 11.3 手写汇编微内核数

| 类别 | 每 ISA 数量 | ISA 数 | 总计 |
|------|------------|--------|------|
| GEMM 微内核 (f32) | 1 | 4 | 4 |
| GEMM 微内核 (f16/bf16) | 2 | 4 | 8 |
| 量化点积 (每格式) | 18 | 4 | 72 |
| **汇编总计** | | | **~84** |

> 注：实际汇编数量取决于格式合并策略。同位宽格式（如 Q4_K/IQ4_NL/IQ4_XS）可共享解包逻辑，
> 只在 scale/zero 处理上分支，减少实际汇编文件数。

---

## 11.5 FusedGraph 桥接设计（DATA-FUSED-GRAPH）

> 路径 A 编译器的输入契约。gllm 负责将高层 FusedOp 展开为原子算子 DAG 后传入 gllm-kernels。

### 接口边界

```
gllm 侧:
  ONNX Graph → GraphOptimizer → FusedGraph（高层融合: FlashAttention/SwiGLU/GQA/...）
                                     │
                                     ▼ expand_for_compiler()
                              CompilerGraph（原子算子: MatMul/RmsNorm/SiLU/Add/...）
                                     │
                                     ▼ 传入 gllm-kernels
gllm-kernels 侧:
  compile_graph(graph: &CompilerGraph, profile: &DeviceProfile) → CompiledLayer
```

**设计决策**：gllm 负责展开，因为 gllm 拥有模型结构知识（哪些算子可以拆分、拆分后的形状推导）。gllm-kernels 只关心原子算子的语义和融合。

### CompilerGraph（gllm-kernels 接收的原子算子图）

```rust
/// 原子算子图 — gllm 展开高层融合后传入 gllm-kernels 的编译器输入
/// 节点为原子算子，边为张量数据流
pub struct CompilerGraph {
    /// 原子算子节点列表（拓扑序）
    pub nodes: Vec<CompilerNode>,
    /// 图输入张量描述
    pub inputs: Vec<TensorDesc>,
    /// 图输出张量描述
    pub outputs: Vec<TensorDesc>,
}

/// 原子算子节点
pub struct CompilerNode {
    /// 节点名称（调试用）
    pub name: String,
    /// 原子算子类型
    pub op: CompilerOp,
    /// 输入张量索引列表（指向其他节点的输出或图输入）
    pub inputs: Vec<TensorRef>,
    /// 输出张量描述
    pub output: TensorDesc,
}

/// 张量引用
pub enum TensorRef {
    /// 图输入（第 n 个）
    GraphInput(usize),
    /// 其他节点的输出: (node_idx, output_idx)
    NodeOutput(usize, usize),
    /// 权重张量（名称引用，运行时由 WeightsHandle 提供）
    Weight(String),
}

/// 原子算子类型 — gllm-kernels 编译器理解的最小粒度
pub enum CompilerOp {
    /// 矩阵乘: C[m,n] = A[m,k] × B[k,n]
    MatMul { m: usize, n: usize, k: usize, transpose_b: bool },
    /// RMSNorm: out = x * w * rsqrt(mean(x²) + eps)
    RmsNorm { hidden_size: usize, eps: f32 },
    /// LayerNorm: out = (x - mean) / sqrt(var + eps) * gamma + beta
    LayerNorm { hidden_size: usize, eps: f32 },
    /// 激活函数（通过 OpKind 引用 → registry → OpTrace）
    Activation(OpKind),
    /// 逐元素加
    Add,
    /// 逐元素乘
    Mul,
    /// RoPE 位置编码
    Rope { head_dim: usize, max_seq_len: usize, theta: f64 },
    /// Softmax
    Softmax { axis: i32 },
    /// 量化矩阵乘
    QuantMatMul { quant_type: QuantType, m: usize, n: usize, k: usize },
    /// Reshape（零拷贝，仅改变逻辑形状）
    Reshape { target_shape: Vec<usize> },
    /// Transpose（可能需要物理重排）
    Transpose { perm: Vec<usize> },
}

// ActivationType 已删除 — 激活函数通过 OpKind 引用 ScalarOpRegistry，
// 编译器从 OpTrace.body 的 TraceOp 序列自动生成 SIMD 指令，
// 不再硬编码算子语义。
//
// 原 CompilerOp::Activation(ActivationType::SiLU)
// → CompilerOp::Activation(OpKind::Silu)
// → registry.get_trace(OpKind::Silu) → OpTrace { pattern: Elementwise { body: [...] } }

/// 张量描述
pub struct TensorDesc {
    pub shape: Vec<usize>,
    pub dtype: DType,
}
```

### gllm 侧展开规则

| gllm FusedOp | 展开为 CompilerOp 序列 |
|--------------|----------------------|
| `FlashAttention` | Reshape(Q) → Reshape(K) → Reshape(V) → MatMul(Q,K^T) → Softmax → MatMul(attn,V) → Reshape(out) |
| `SwiGLU` | MatMul(gate) → Activation(SiLU) → MatMul(up) → Mul → MatMul(down) |
| `FusedRMSLinear` | RmsNorm → MatMul |
| `FusedQkvRope` | MatMul(Wq) → Rope → MatMul(Wk) → Rope → MatMul(Wv) |
| `RoPE` | Rope（直接映射） |
| `GQA` | 展开为 FlashAttention 等价序列 |
| `Atomic("MatMul")` | MatMul（直接映射） |
| `Atomic("Add")` | Add（直接映射） |
| `Atomic("Softmax")` | Softmax（直接映射） |

### 与编译器的关系

- 基础入口：`compile_model(config: &ModelConfig)` → `LayerIR` → `ExecutionPlan` → `CompiledLayer`
- 语义驱动入口：`compile_graph(graph: &CompilerGraph, profile: &DeviceProfile)` → `SemanticDAG` → `FusionPlan` → `CompiledLayer`
- 共享基础设施：`CompiledLayer`、`CompilationCache`
- 编译失败时回退到 Layer 2 fallback（逐算子调用）

---

## 12. 编译器数据结构（DATA-COMPILER）

> 语义驱动编译器的核心数据结构。四阶段编译流水线（符号执行 → DAG 构筑 → 融合决策 → 代码生成）的中间表示。

### 12.1 Phase 0 数据结构：标量函数分析

编译器通过二进制符号执行自动提取算子的计算结构。算子的唯一定义来源是 `extern "C"` 纯标量函数。

```rust
/// Phase 0 输出：算子的完整计算结构描述
/// 由二进制符号执行从 extern "C" 标量函数自动提取
pub struct OpTrace {
    /// 算子类型标识
    pub op_kind: OpKind,
    /// 计算模式（完整计算结构，非分类标签）
    pub pattern: ComputePattern,
    /// 标量函数签名
    pub signature: ScalarFnSignature,
}

/// 计算模式 — 从标量函数的循环结构和数据流自动识别
pub enum ComputePattern {
    /// out[i] = f(in[i]) — 单输入逐元素变换
    /// 符号执行识别：单循环，每次迭代 load 1 → compute → store 1
    Elementwise { body: Vec<TraceOp> },

    /// out[i] = f(a[i], b[i]) — 双输入逐元素运算
    /// 符号执行识别：单循环，每次迭代 load 2 → compute → store 1
    BinaryElementwise { body: Vec<TraceOp> },

    /// out[i] = f(in[i], extra_0[i], extra_1[i], ...) — 带额外参数的逐元素变换
    /// 符号执行识别：单循环，每次迭代 load N (N≥2) → compute → store M (M≥1)
    /// 典型算子：RoPE (4 输入 2 输出)、带位置编码的变换
    /// 与 Elementwise/BinaryElementwise 的区别：输入/输出数量不固定
    /// OpClass 推导为 Injective（可融合进消费者，但不能作为 epilogue 注入 GEMM）
    Injective {
        body: Vec<TraceOp>,
        num_inputs: usize,
        num_outputs: usize,
    },

    /// result = fold(input, identity, combine) — 归约
    /// 符号执行识别：循环内累加器跨迭代存活
    Reduction { identity: f64, combine: Vec<TraceOp> },

    /// Pass 1: reduce, Pass 2: elementwise with reduction result — 归一化类
    /// 符号执行识别：两个连续循环，第二个循环使用第一个循环的归约结果
    NormLike {
        reduce: Vec<TraceOp>,
        finalize: Vec<TraceOp>,
        transform: Vec<TraceOp>,
    },

    /// 三重循环矩阵乘
    /// 符号执行识别：三层嵌套循环 + FMA 累加
    /// 注意：epilogue 不在此处 — GEMM 的 epilogue 是 Phase 2 融合决策的结果，
    /// 存储在 FusionPlan 中，由消费者算子的 OpTrace.body 提供
    Gemm,

    /// 量化解码 + 计算
    /// 符号执行识别：块级循环 + 位操作解包 + scale 应用
    QuantDecode { block_size: usize, decode: Vec<TraceOp> },
}

/// 计算操作（SSA 形式，u32 引用前序操作的输出索引）
/// Phase 3 代码生成时，每个 TraceOp 映射到对应的 SIMD 指令
#[derive(Debug, Clone)]
pub enum TraceOp {
    /// 输入值（参数索引）
    Input(u32),
    /// 常量
    Const(f64),
    /// 算术运算
    Add(u32, u32), Sub(u32, u32), Mul(u32, u32), Div(u32, u32),
    /// 融合乘加: a * b + c
    Fma(u32, u32, u32),
    /// 一元运算
    Neg(u32), Abs(u32),
    /// 超越函数（符号执行通过 libm 调用识别）
    Exp(u32), Sqrt(u32), Rsqrt(u32), Tanh(u32),
    /// 快速近似
    Recip(u32),
    /// 比较
    Max(u32, u32), Min(u32, u32),
}

/// 标量函数签名 — 描述 extern "C" 函数的参数布局
pub struct ScalarFnSignature {
    /// 函数指针（编译后的标量函数地址）
    pub fn_ptr: *const u8,
    /// 参数列表
    pub params: Vec<ScalarParam>,
}

/// 标量函数参数类型
pub enum ScalarParam {
    /// 输入数据指针 (*const f32)
    InputPtr,
    /// 输出数据指针 (*mut f32)
    OutputPtr,
    /// 权重数据指针 (*const f32)
    WeightPtr,
    /// 维度参数 (usize)
    Dim(usize),
    /// 标量参数（如 eps）
    Scalar(f32),
}

/// 标量算子注册表 — 所有算子的 extern "C" 标量函数集中注册
pub struct ScalarOpRegistry {
    /// OpKind → 标量函数指针
    entries: HashMap<OpKind, ScalarFnSignature>,
    /// OpKind → 已缓存的 OpTrace（首次分析后缓存）
    trace_cache: HashMap<OpKind, OpTrace>,
}

impl ScalarOpRegistry {
    /// 注册标量函数
    pub fn register(&mut self, op: OpKind, sig: ScalarFnSignature);
    /// 获取 OpTrace（首次调用时触发符号执行，之后从缓存返回）
    pub fn get_trace(&mut self, op: &OpKind) -> Result<&OpTrace, CompileError>;
}
```

#### 符号执行内部类型（不导出，仅引擎内部使用）

```rust
/// 符号值 — 追踪寄存器/内存中数据的来源
enum SymValue {
    /// 函数参数（第 n 个）
    Input(usize),
    /// 常量
    Const(f64),
    /// 从内存加载
    Load(Box<SymValue>),
    /// 算术运算
    Add(Box<SymValue>, Box<SymValue>),
    Mul(Box<SymValue>, Box<SymValue>),
    Div(Box<SymValue>, Box<SymValue>),
    Neg(Box<SymValue>),
    /// libm 函数调用
    Call(LibmFn, Vec<SymValue>),
}

/// 识别的 libm 函数
enum LibmFn {
    Expf, Sqrtf, Tanhf, Logf, Fabsf,
}

/// 符号执行状态
struct SymState {
    /// 寄存器 → 符号值映射
    regs: HashMap<iced_x86::Register, SymValue>,
    /// 栈偏移 → 符号值映射
    stack: HashMap<i64, SymValue>,
    /// 内存操作记录（用于识别 load/store 模式）
    memory: Vec<SymMemOp>,
}

/// 内存操作记录
struct SymMemOp {
    kind: MemOpKind,
    addr: SymValue,
    value: SymValue,
}

enum MemOpKind { Load, Store }
```

### 12.2 Phase 1 数据结构：语义 DAG

#### SemanticDAG（语义标注图）

```rust
/// 语义 DAG — CompilerGraph + 语义标注 + 数据流分析
pub struct SemanticDAG {
    /// 语义标注节点列表（保持拓扑序）
    pub nodes: Vec<SemanticNode>,
    /// 张量 def-use 链
    pub tensor_edges: Vec<TensorEdge>,
    /// 全局输入张量 ID 列表
    pub inputs: Vec<TensorId>,
    /// 全局输出张量 ID 列表
    pub outputs: Vec<TensorId>,
    /// 后支配树（用于融合组划分）
    pub post_dominator_tree: PostDomTree,
}

/// 语义标注节点
pub struct SemanticNode {
    /// 节点 ID（对应 CompilerGraph 中的索引）
    pub node_id: usize,
    /// 算子计算结构（Phase 0 符号执行提取，替代原来的 OpSemanticsKind）
    pub op_trace: OpTrace,
    /// TVM 算子分类（从 op_trace.pattern 自动推导）
    pub op_class: OpClass,
    /// 瓶颈类型
    pub bottleneck: Bottleneck,
    /// 算术强度 (FLOPs / Bytes)
    pub arithmetic_intensity: f32,
    /// 每元素字节数
    pub bytes_per_elem: usize,
    /// 每元素浮点运算数
    pub flops_per_elem: usize,
    /// 输入张量 ID 列表
    pub inputs: Vec<TensorId>,
    /// 输出张量 ID 列表
    pub outputs: Vec<TensorId>,
    /// 输出形状（元素数）
    pub output_elems: usize,
}

/// TVM 算子分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// 逐元素: vec_add, silu, gelu, relu, exp, vec_mul
    ElemWise,
    /// 注入式: rope, reshape, transpose（带额外参数的逐元素变换）
    Injective,
    /// 归约: softmax, rms_norm, layer_norm
    Reduction,
    /// 矩阵乘: gemm, gemv
    Gemm,
    /// 不透明: 量化 matmul 等
    Opaque,
}

/// ComputePattern → OpClass 自动推导规则
///
/// | ComputePattern      | OpClass    | 说明 |
/// |---------------------|------------|------|
/// | Elementwise         | ElemWise   | 单输入逐元素 |
/// | BinaryElementwise   | ElemWise   | 双输入逐元素 |
/// | Injective           | Injective  | 多输入/多输出逐元素（如 RoPE） |
/// | Reduction           | Reduction  | 纯归约 |
/// | NormLike            | Reduction  | 归约 + 逐元素（两 pass） |
/// | Gemm                | Gemm       | 三重循环矩阵乘 |
/// | QuantDecode         | Opaque     | 量化解码，不参与融合 |

/// 瓶颈类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Mixed,
}

/// 张量 def-use 边
pub struct TensorEdge {
    /// 张量 ID
    pub tensor_id: TensorId,
    /// 生产者节点 ID
    pub producer: usize,
    /// 消费者节点 ID 列表
    pub consumers: Vec<usize>,
    /// 数据量（字节）
    pub data_bytes: usize,
    /// 是否可寄存器传递（单消费者 + 生产者/消费者均为 elemwise）
    pub can_register_pass: bool,
}

/// 张量标识符（图内唯一）
pub type TensorId = u32;

/// 后支配树（简化表示）
pub struct PostDomTree {
    /// 每个节点的直接后支配者
    pub ipost_dom: Vec<Option<usize>>,
}
```

### 12.3 Phase 2 数据结构：融合决策

#### FusionPlan（融合计划）

```rust
/// 融合计划 — Phase 2 的完整输出
pub struct FusionPlan {
    /// 融合组列表（拓扑序）
    pub groups: Vec<FusionGroup>,
    /// 分块配置（每个融合组一个）
    pub tile_configs: Vec<TileConfig>,
    /// 缓冲区规划
    pub buffer_plan: BufferPlan,
}

/// 融合组 — 一组将被编译为单一代码块的算子
pub struct FusionGroup {
    /// 组内的节点 ID 列表（拓扑序）
    pub node_ids: Vec<usize>,
    /// 融合策略
    pub strategy: FusionStrategy,
    /// 组的聚合瓶颈类型
    pub bottleneck: Bottleneck,
    /// 组输入张量 ID
    pub inputs: Vec<TensorId>,
    /// 组输出张量 ID
    pub outputs: Vec<TensorId>,
    /// 融合后的寄存器压力估算
    pub register_pressure: RegisterPressure,
}

/// 融合策略（Profile-Driven，非模板能力驱动）
///
/// 每个 FusionGroup 携带一个 FusionStrategy，由硬件 profile 和数据量共同决定。
/// Phase 2 入口签名: `fuse(graph: &CompilerGraph, profile: &DeviceProfile) -> FusionPlan`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionStrategy {
    /// 单算子，不融合
    Single,

    /// Loop Fusion: 多个 elemwise 算子合并为单循环
    /// 数据在寄存器中流过整个链，消除中间内存往返
    LoopFusion {
        /// 链中每个算子的节点 ID（按执行顺序）
        chain_nodes: Vec<usize>,
    },

    /// Epilogue Injection: 将 elemwise 消费者注入 GEMM store 阶段
    /// 在累加器寄存器上原地执行，不经过内存
    /// epilogue 的指令序列从消费者算子的 OpTrace.body 自动生成
    EpilogueInjection {
        /// GEMM 节点 ID
        gemm_node: usize,
        /// 注入的 epilogue 算子列表（按执行顺序）
        epilogue_ops: Vec<EpilogueOp>,
    },

    /// Tile-Level Fusion: 前驱算子的 tile 计算嵌入 GEMM MC 循环
    ///
    /// 硬件驱动决策: 当前驱输出 > L1 * 0.75 时使用
    /// 例: RMSNorm(hidden=16384) 输出 64KB > L1(32KB)*0.75 → 嵌入 MC 循环
    ///
    /// scratch buffer 方案:
    /// · MC 行的前驱结果写入 scratchpad 的 normed 区域（MC × K × sizeof(E) bytes）
    /// · 紧接着被 pack_a 消费，pack_a 按 KC 列切片读取（MC × KC × sizeof(E)，在 L2 内）
    /// · weight 向量通过 JIT 函数参数传入（graph input，不在 scratchpad 里）
    /// · 前驱算子逐行独立（如 RMSNorm），按 MC 行切分不影响正确性
    /// · 每个 MC tile 独立做完整的多 pass（如 RMSNorm: pass1 sum_squares + pass2 scale）
    TileLevelFusion {
        /// GEMM 节点 ID
        gemm_node: usize,
        /// 嵌入 MC 循环的前驱算子
        tiled_predecessor: usize,
        /// MC tile 行数（由 GEMM blocking 参数决定，来自 DeviceProfile 的 cache 层级）
        tile_rows: usize,
        /// 可选的 epilogue 注入
        epilogue_ops: Vec<EpilogueOp>,
    },

    /// ComputeRoot: 前驱算子先整体算完，结果留在 L1
    ///
    /// 硬件驱动决策: 当前驱输出 ≤ L1 * 0.75 时使用
    /// 例: RMSNorm(hidden=4096) 输出 16KB ≤ L1(32KB)*0.75 → 先算完，GEMM 读时仍热
    ///
    /// 与 TileLevelFusion 互斥 — 同一个 (前驱, GEMM) 对只选其一
    ComputeRoot {
        /// 先整体执行的前驱算子
        predecessor: usize,
        /// 后续 GEMM 节点 ID
        gemm_node: usize,
    },

    /// Fallback: 调用 Kernels<E> 方法（Opaque 算子）
    KernelCall {
        method: KernelMethod,
    },
}

/// Epilogue 操作
/// 不再使用硬编码的 EpilogueKind 枚举（BiasAdd/Activation 等），
/// 而是直接携带消费者算子的 OpTrace，Phase 3 代码生成遍历 TraceOp 序列生成 SIMD 指令
#[derive(Debug, Clone)]
pub struct EpilogueOp {
    /// 对应的算子节点 ID
    pub node_id: usize,
    /// 消费者算子的 OpTrace（包含完整计算结构）
    /// Phase 3 从 trace.pattern 的 body/TraceOp 序列直接映射到 SIMD 指令
    pub trace: OpTrace,
    /// 额外参数指针（如 bias 向量地址）
    pub extra_ptr: Option<PtrSource>,
}

/// 寄存器压力估算
#[derive(Debug, Clone)]
pub struct RegisterPressure {
    /// 需要的 SIMD 寄存器数（ymm/zmm/v）
    pub simd_regs_needed: usize,
    /// 需要的通用寄存器数
    pub gpr_regs_needed: usize,
    /// 是否超出可用寄存器（需要 spill）
    pub needs_spill: bool,
}

/// Kernels<E> 方法标识（用于 fallback 路径）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelMethod {
    Gemm, GemmBias, GemmBiasAct, GemmPrepacked, Gemv, PackB,
    RmsNorm, LayerNorm, Silu, Gelu, Relu, Swiglu,
    Softmax, Rope, RopeWithPos, VecAdd, VecMul, Exp,
    KquantMatmul, IqMatmul, AwqMatmul, GptqMatmul, SqueezeMatmul,
}
```

#### TileConfig（分块配置）

```rust
/// 分块配置 — 根据 DeviceProfile cache 层级 + 融合策略确定
pub struct TileConfig {
    /// GEMM BLIS 三级分块（仅 Gemm/EpilogueInjection/TileLevelFusion）
    pub gemm_blocking: Option<GemmBlocking>,
    /// Elementwise tile 大小（元素数，适配 L1）
    pub elem_tile: usize,
    /// Tile-Level Fusion 的前驱 tile 大小（MC 对齐）
    pub predecessor_tile: Option<usize>,
    /// 线程数
    pub num_threads: usize,
    /// 并行策略（Phase 2 决定，Phase 3 生成对应代码）
    pub parallel: ParallelStrategy,
    /// 预取距离 (bytes)
    pub prefetch_distance: usize,
}

/// 并行策略 — Phase 2 决定哪个循环层级并行化
///
/// JIT 生成的代码是单线程的（纯计算，无线程同步逻辑）。
/// 调用方（InferenceBackend）负责线程调度：
///   1. Phase 2 决定并行维度和 tile 划分
///   2. Phase 3 生成单个 tile 的计算函数
///   3. 运行时 thread pool 按 ParallelStrategy 分发 tile 到各线程
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// GEMM: NC 循环并行（每个 NC tile 独立，无数据依赖）
    /// 线程 i 处理 NC tiles [i*chunk .. (i+1)*chunk]
    GemmNcParallel {
        /// 每线程处理的 NC tile 数
        tiles_per_thread: usize,
    },
    /// Elementwise/LoopFusion: 按元素数均分
    /// 线程 i 处理 elements [i*chunk .. (i+1)*chunk]
    ElemParallel {
        /// 每线程处理的元素数（对齐到 SIMD 宽度）
        elems_per_thread: usize,
    },
    /// 单线程执行（数据量太小，并行开销不值得）
    Sequential,
}

/// GEMM BLIS 分块参数
pub struct GemmBlocking {
    /// K 维度分块（适配 L1 Cache）
    pub kc: usize,
    /// M 维度分块（适配 L2 Cache）
    pub mc: usize,
    /// N 维度分块（适配 L3 Cache）
    pub nc: usize,
    /// 微内核行数
    pub mr: usize,
    /// 微内核列数
    pub nr: usize,
    /// K 维度展开因子
    pub k_unroll: usize,
}
```

#### BufferPlan（缓冲区规划）

```rust
/// 缓冲区规划 — 通过张量活性分析 + 区间图着色生成
pub struct BufferPlan {
    /// 总 scratchpad 字节数
    pub scratchpad_bytes: usize,
    /// 每个张量的 buffer 分配
    pub allocations: Vec<BufferAlloc>,
}

/// 单个 buffer 分配
pub struct BufferAlloc {
    /// 张量 ID
    pub tensor_id: TensorId,
    /// scratchpad 内的字节偏移
    pub offset: usize,
    /// 字节大小
    pub size_bytes: usize,
    /// 是否原地复用其他 buffer（Some = 复用的源张量 ID）
    pub reuses: Option<TensorId>,
    /// 张量生命周期: (birth 拓扑序位置, death 拓扑序位置)
    pub lifetime: (usize, usize),
}
```

**缓冲区分配算法（张量活性分析 + 区间图着色）**

Phase 2 Step 4 执行以下流程，输入为 SemanticDAG 的拓扑排序结果：

```
Step 1: 张量活性分析
  对 SemanticDAG 做拓扑排序，为每条边（张量）计算生命周期：
    birth = 生产者节点的拓扑序位置
    death = 所有消费者节点中最大的拓扑序位置
  输出: Vec<(TensorId, birth, death, size_bytes)>

Step 2: 按 size_bytes 降序排序（大张量优先分配，减少碎片）

Step 3: 区间图着色贪心分配
  维护 free_list: Vec<(offset, size)>，初始为空
  对每个张量 t（按 Step 2 排序）:
    a. 检查原地复用：如果 t 的生产者是 elemwise 且 t.size_bytes == input.size_bytes
       且 input.death == t.birth（输入在此处死亡），则复用 input 的 offset
       → BufferAlloc { reuses: Some(input_id), offset: input.offset, ... }
    b. 否则在 free_list 中找 first-fit 空闲区间（offset 对齐到 64 字节）
    c. 找不到则在 scratchpad 末尾追加，更新 scratchpad_bytes
    d. 当张量到达 death 位置时，将其区间归还 free_list（合并相邻空闲区间）

Step 4: 输出 BufferPlan { scratchpad_bytes, allocations }
```

约束：
- 所有 offset 按 64 字节对齐（SIMD 对齐要求）
- 图的输入/输出张量不参与 scratchpad 分配（由调用方提供）
- 仅中间张量（融合组内部产生且内部消费的张量）参与分配

### 12.4 Phase 3 数据结构：代码生成

#### CodeGenPlan（代码生成计划）

FusionPlan 到机器码的最终中间表示。每个 FusionGroup 对应一个 CodeGenUnit。

```rust
/// 代码生成计划 — 驱动 MachineCodeEmitter (x86_64: iced-x86 / aarch64: dynasm-rs)
pub struct CodeGenPlan {
    /// 代码生成单元列表（与 FusionGroup 一一对应）
    pub units: Vec<CodeGenUnit>,
    /// 缓冲区规划（从 FusionPlan 继承）
    pub buffer_plan: BufferPlan,
    /// 常量池（SIMD 对齐常量，如 SiLU Horner 系数）
    pub constant_pool: ConstantPool,
}

/// 代码生成单元
pub enum CodeGenUnit {
    /// Loop Fusion: 生成单循环，数据在寄存器中流过整个链
    FusedLoop {
        /// 循环元素数
        num_elements: usize,
        /// 循环体内的算子列表（按执行顺序，编译器根据语义生成指令）
        body_ops: Vec<FusedLoopOp>,
        /// 输入指针
        input: PtrSource,
        /// 输出指针
        output: PtrSource,
        /// 额外输入指针（如 VecMul 的权重、VecAdd 的残差）
        extra_inputs: Vec<PtrSource>,
    },

    /// GEMM + 可选 epilogue + 可选 tile-level fusion
    GemmUnit {
        /// BLIS 分块参数
        blocking: GemmBlocking,
        /// 可选 epilogue 注入（在 store 之前执行）
        epilogue_ops: Vec<EpilogueOp>,
        /// 可选 tile-level fusion（嵌入 MC 循环的前驱算子）
        tiled_predecessor: Option<TiledPredecessor>,
        /// M, N, K 维度
        m: usize, n: usize, k: usize,
        /// 指针
        a: PtrSource, b: PtrSource, c: PtrSource,
        /// 可选 bias 指针
        bias: Option<PtrSource>,
    },

    /// Fallback: 调用 Kernels<E> 方法
    KernelCall {
        method: KernelMethod,
        args: Vec<PtrSource>,
        output: PtrSource,
    },
}

/// Tile-Level Fusion 的前驱算子描述
pub struct TiledPredecessor {
    /// 前驱算子的计算结构（从 OpTrace 获取，编译器据此生成 tile 代码）
    pub op_trace: OpTrace,
    /// 前驱算子的节点 ID
    pub node_id: usize,
    /// 前驱输出写入的 scratch buffer
    pub scratch: PtrSource,
    /// tile 大小（与 MC 对齐）
    pub tile_size: usize,
}

/// 常量池
pub struct ConstantPool {
    /// 对齐的常量数据
    pub data: Vec<u8>,
    /// 常量条目: (偏移, 大小, 对齐)
    pub entries: Vec<ConstantEntry>,
}

pub struct ConstantEntry {
    pub id: usize,
    pub offset: usize,
    pub size: usize,
    pub align: usize,
}
```

#### PtrSource（指针来源）

```rust
/// 指针来源 — 描述运行时数据的位置
#[derive(Debug, Clone)]
pub enum PtrSource {
    /// CompiledLayer 入口参数
    EntryArg(EntryArgKind),
    /// Scratchpad 内的偏移
    Scratch { offset: usize },
    /// 权重张量
    Weight { layer_idx: usize, kind: WeightKind },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryArgKind {
    Input, Output, Weights, KvCache,
    Scratchpad, ScratchA, ScratchB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightKind {
    AttnNorm, Wq, Wk, Wv, Wo,
    FfnNorm, WGate, WUp, WDown,
}
```

#### 平台后端统一接口

Phase 3（代码生成）的平台差异通过 `MachineCodeEmitter` trait 封装，Phase 1/2 完全平台无关。

```rust
// ============================================================
// Phase 3: 代码生成（平台特定）
// ============================================================

/// 平台无关的代码生成接口
/// x86_64: X86Emitter (iced-x86 CodeAssembler)
/// aarch64: Arm64Emitter (dynasm-rs Assembler)
pub trait MachineCodeEmitter {
    /// 生成 GEMM 单元（三重循环 + 微内核 + 可选 epilogue/tile-fusion）
    fn emit_gemm_unit(&mut self, unit: &GemmUnit) -> Result<Vec<u8>>;
    /// 生成融合 Elementwise 循环
    fn emit_fused_loop(&mut self, unit: &FusedLoop) -> Result<Vec<u8>>;
    /// 从 OpTrace.body 的 TraceOp 序列生成 SIMD 指令（对指定寄存器原地执行）
    /// reg: 主数据寄存器（输入/输出）
    /// scratch: 可用的 scratch 寄存器集合（GEMM epilogue 场景下只有累加器剩余的几个）
    fn emit_trace_ops(&mut self, ops: &[TraceOp], reg: Register, scratch: &[Register]) -> Result<()>;
    /// 生成 prologue（保存 callee-saved 寄存器）
    fn emit_prologue(&mut self) -> Result<()>;
    /// 生成 epilogue（恢复 + ret）
    fn emit_epilogue(&mut self) -> Result<()>;
    /// 完成并返回可执行字节
    fn finalize(self) -> Result<Vec<u8>>;
}

/// 融合循环体内的算子描述
/// 代码生成时遍历 op_trace.body 中的 TraceOp，逐条映射到 SIMD 指令
#[derive(Debug, Clone)]
pub struct FusedLoopOp {
    /// 算子的计算结构（从 OpTrace 获取）
    pub op_trace: OpTrace,
    /// 额外输入指针索引（对应 FusedLoop::extra_inputs）
    pub extra_input_idx: Option<usize>,
}

// ============================================================
// 统一入口：PlatformBackend
// ============================================================

/// 平台后端 — 提供 Phase 3 代码生成能力
/// 编译流水线通过此 trait 获取当前平台的代码生成器
pub trait PlatformBackend {
    type Emitter: MachineCodeEmitter;

    fn new_emitter(&self) -> Self::Emitter;
    fn platform(&self) -> Platform;
    fn num_simd_regs(&self) -> usize;
}

#[derive(Debug, Clone, Copy)]
pub enum Platform {
    X86_64 { avx512: bool },
    Aarch64 { sve: bool },
}

// ============================================================
// 平台具体实现
// ============================================================

/// x86_64 后端
pub struct X86Backend;

pub struct X86Emitter {
    asm: iced_x86::code_asm::CodeAssembler,
}

/// aarch64 后端
pub struct Arm64Backend;

pub struct Arm64Emitter {
    ops: dynasmrt::aarch64::Assembler,
}
```

#### Register（平台寄存器抽象）

```rust
/// 平台无关的寄存器标识
/// 在 Phase 3 代码生成时映射到平台具体寄存器
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Register {
    /// x86_64 通用寄存器 (rax=0, rcx=1, ..., r15=15)
    X86Gpr(u8),
    /// x86_64 SIMD 寄存器 (ymm0-ymm15 / zmm0-zmm31)
    X86Simd(u8),
    /// aarch64 通用寄存器 (x0-x30)
    Arm64Gpr(u8),
    /// aarch64 NEON 寄存器 (v0-v31)
    Arm64Neon(u8),
}
```

## 13. 推理层数据结构（DATA-INFERENCE）

> 路径 B（推理执行）使用的核心类型。详细语义见 SPEC/05。

### 13.1 ModelConfig（模型配置）

```rust
/// 模型配置 — 由 gllm 传入，描述模型架构参数
pub struct ModelConfig {
    pub arch: ModelArch,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub norm_type: NormType,
    /// FFN 激活函数的 OpTrace body（从 ScalarOpRegistry 获取，不再使用 ActivationKind 枚举）
    pub activation_trace: Option<Vec<TraceOp>>,
    pub rope_config: Option<RopeConfig>,
    pub dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama, Gpt2, Mistral, Phi, Qwen, Gemma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RmsNorm, LayerNorm,
}

#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub base: f32,
    pub scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone)]
pub enum RopeScaling {
    Linear(f32),
    Dynamic { factor: f32, max_seq_len: usize },
}
```

### 13.2 DeviceTensor（统一张量句柄）

```rust
/// 统一张量句柄 — CPU/GPU 透明
pub struct DeviceTensor {
    /// CPU: host pointer; GPU: device pointer
    ptr: *mut u8,
    len_bytes: usize,
    num_elements: usize,
    dtype: DType,
    device: DeviceKind,
    /// true = Drop 时释放
    owned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda(u32),
    Metal(u32),
}
```

- CPU 路径零开销：`as_slice::<E>()` 直接返回 `&[E]`
- GPU 路径：数据留在设备端，通过 `upload_f32` / `download_f32` 传输
- 64 字节对齐分配（cache line aligned）

### 13.3 KvCache（分页 KV 缓存）

```rust
/// 分页 KV 缓存 — 设计灵感: vLLM PagedAttention
pub struct KvCache {
    /// 物理页池
    pages: Vec<Page>,
    /// 空闲页栈
    free_pages: Vec<usize>,
    /// [layer][seq] → 页表
    layer_tables: Vec<Vec<SeqPageTable>>,
}
```

- 页大小：16 tokens
- 每页存储：`[2(K+V), num_kv_heads, PAGE_SIZE, head_dim]`
- 支持：append / reset_seq / swap_out / swap_in

### 13.4 ModelWeights（权重存储）

```rust
pub struct ModelWeights {
    pub embedding: DeviceTensor,     // [vocab_size, hidden_size]
    pub layers: Vec<LayerWeights>,
    pub final_norm: DeviceTensor,    // [hidden_size]
    pub lm_head: DeviceTensor,       // [hidden_size, vocab_size]
}

pub struct LayerWeights {
    pub attn_norm: DeviceTensor,     // RMSNorm / LayerNorm weight
    pub wq: DeviceTensor,
    pub wk: DeviceTensor,
    pub wv: DeviceTensor,
    pub wo: DeviceTensor,
    pub ffn_norm: DeviceTensor,
    pub w_gate: DeviceTensor,
    pub w_up: DeviceTensor,
    pub w_down: DeviceTensor,
}
```

### 13.5 InferenceError（推理错误）

```rust
#[derive(Debug)]
pub enum InferenceError {
    OutOfMemory { requested: usize, available: usize },
    InvalidArg(String),
    CompileError(String),
    RuntimeError(String),
    Unsupported(String),
    IoError(std::io::Error),
}
```

与 FFI 层 `GllmStatus` 错误码一一对应（见 SPEC/05 §6）。
