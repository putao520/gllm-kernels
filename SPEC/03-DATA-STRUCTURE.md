# gllm-kernels 数据结构与算子架构

> **📌 SSOT**: 本文档定义 gllm-kernels 的核心数据结构、算子清单、分发架构。

---

## 1. 三层树状分发架构（ARCH-DISPATCH）🚨 铁律

### 1.1 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Backend Device (后端设备) - 运行时用户指定                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              Backend                                        │
│                                 │                                           │
│              ┌──────────────────┼──────────────────┐                        │
│              ▼                  ▼                  ▼                        │
│         CpuBackend         CudaBackend      Metal/ROCm (规划中)             │
│              │                  │                                           │
│              ▼                  ▼                                           │
│   L1.5 CPU架构(编译时)    (直接泛型)                                         │
│    ├─ x86_64                                                                │
│    ├─ ARM                                                                   │
│    └─ AppleSilicon                                                          │
│              │                                                              │
│              ▼                                                              │
│         Layer 2                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 2: ISA (仅 CPU，启动时一次检测)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            CpuBackend                                       │
│                                │                                            │
│          ┌─────────────┬───────┴───────┬─────────────┐                      │
│          ▼             ▼               ▼             ▼                      │
│       Scalar         AVX2          AVX-512         NEON                     │
│          │             │               │             │                      │
│          ▼             ▼               ▼             ▼                      │
│      Layer 3       Layer 3         Layer 3       Layer 3                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 3: Precision (精度，编译时泛型单态化)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         impl<E: Element>                                    │
│                                │                                            │
│              ┌─────────────────┼─────────────────┐                          │
│              ▼                 ▼                 ▼                          │
│          E = f32           E = f16           E = bf16                       │
│       (编译时展开)        (编译时展开)       (编译时展开)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 零成本分发原则

| 层级 | 分发时机 | 机制 | 开销 |
|------|----------|------|------|
| **Layer 1** | 用户指定 | 编译时泛型 `B: Backend` | 零 |
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

    /// 从 f32 转换（解量化后的标准格式）
    fn from_f32(v: f32) -> Self;
    /// 转换为 f32（最终输出或高精度计算）
    fn to_f32(self) -> f32;

    /// 融合乘加：self + a * b
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// 基础算术
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn neg(self) -> Self;

    /// 比较
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    /// 数学函数
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn recip(self) -> Self;  // 1/x
}
```

### 2.2 Backend Trait（DATA-BACKEND）

```rust
/// 后端设备 Trait
pub trait Backend: Send + Sync + 'static {
    const NAME: &'static str;

    /// 关联的内核实现类型
    type Kernels<E: Element>: Kernels<E>;

    /// 初始化后端，返回内核实例
    fn init<E: Element>() -> Self::Kernels<E>;
}

// 后端实现
pub struct CpuBackend;           // ✅ 实现中 (内部按 target_arch 分发)
pub struct CudaBackend;          // ✅ 实现中
pub struct MetalBackend;         // 📋 规划中
pub struct RocmBackend;          // 📋 规划中
```

### 2.3 Kernels Trait（DATA-KERNELS）🚨 核心

```rust
/// 内核算子接口 - 所有后端/ISA 实现此 Trait
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
    fn gemm_bias(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize);

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
    fn rope(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, interleaved: bool);
    fn rope_with_pos(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, position: usize, interleaved: bool);

    // ========================================================================
    // 查表
    // ========================================================================
    fn embedding_lookup(&self, ids: &[u32], table: &[E], output: &mut [E], vocab_size: usize, hidden_size: usize);

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
    fn dequant_awq4(&self, packed: &[u8], zeros: &[u8], scales: &[half::f16], out: &mut [f32]);
    fn dequant_gptq4(&self, packed: &[u8], g_idx: &[i32], scales: &[half::f16], out: &mut [f32]);
    fn dequant_squeeze(&self, block: &[u8], out: &mut [f32]);

    // ========================================================================
    // 量化 GEMV/GEMM
    // ========================================================================
    fn gemv_q8(&self, weight: &[i8], input: &[E], scale: f32, n: usize) -> E;
    fn gemv_q4(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E;
    fn gemv_q2(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E;
    fn gemv_q1(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E;

    fn gemm_q8(&self, weight: &[i8], input: &[E], output: &mut [E], scales: &[f32], m: usize, n: usize, k: usize);
    fn gemm_q4(&self, weight: &[u8], input: &[E], output: &mut [E], scales: &[f32], m: usize, n: usize, k: usize);

    // ========================================================================
    // 融合算子
    // ========================================================================
    fn fused_qkv_rope(
        &self,
        input: &[E], wq: &[E], wk: &[E], wv: &[E],
        cos: &[E], sin: &[E],
        q_out: &mut [E], k_out: &mut [E], v_out: &mut [E],
        seq_len: usize, hidden_size: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        rotary_dim: usize, interleaved: bool,
    );

    fn fused_gate_up_swiglu(
        &self,
        input: &[E], gate_weight: &[E], up_weight: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize,
    );

    fn fused_ffn(
        &self,
        input: &[E],
        gate_weight: &[E], up_weight: &[E], down_weight: &[E],
        residual: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize,
    );

    fn fused_linear_residual_rmsnorm(
        &self,
        input: &[E], weight: &[E],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    );

    fn flash_attention(
        &self,
        q: &[E], k: &[E], v: &[E], output: &mut [E],
        seq_len: usize, num_heads: usize, head_dim: usize,
        scale: f32, causal: bool,
    );

    fn flash_attention_paged(
        &self,
        q: &[E], k_cache: &[E], v_cache: &[E],
        page_table: &[usize], output: &mut [E],
        seq_len: usize, cache_len: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        page_size: usize, scale: f32,
    );

    fn fused_ffn_rmsnorm(
        &self,
        input: &[E],
        gate_weight: &[E], up_weight: &[E], down_weight: &[E],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize, eps: f32,
    );

    fn fused_linear_bias_residual_rmsnorm(
        &self,
        input: &[E], weight: &[E], bias: &[E],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    );

    // ========================================================================
    // 量化融合算子
    // ========================================================================
    fn fused_qkv_rope_q4(
        &self,
        input: &[E],
        wq: &[u8], wk: &[u8], wv: &[u8],
        scales_q: &[f32], scales_k: &[f32], scales_v: &[f32],
        cos: &[E], sin: &[E],
        q_out: &mut [E], k_out: &mut [E], v_out: &mut [E],
        seq_len: usize, hidden_size: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        rotary_dim: usize, interleaved: bool,
    );

    fn fused_ffn_q4(
        &self,
        input: &[E],
        gate: &[u8], up: &[u8], down: &[u8],
        gate_scales: &[f32], up_scales: &[f32], down_scales: &[f32],
        residual: &[E], output: &mut [E],
        seq_len: usize, hidden_size: usize, ffn_dim: usize,
    );

    fn fused_dequant_gemv(
        &self,
        weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: QuantType, m: usize, n: usize, k: usize,
    );

    fn fused_int8_linear_residual_rmsnorm(
        &self,
        input: &[E], weight: &[i8], scales: &[f32],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    );

    fn fused_int4_linear_residual_rmsnorm(
        &self,
        input: &[E], weight: &[u8], scales: &[f32],
        residual: &[E], norm_weight: &[E], output: &mut [E],
        seq_len: usize, in_features: usize, out_features: usize, eps: f32,
    );

    // ========================================================================
    // 量化格式专用 Matmul
    // ========================================================================
    fn kquant_matmul(
        &self,
        weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: QuantType, m: usize, n: usize, k: usize,
    );

    fn iq_matmul(
        &self,
        weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: QuantType, m: usize, n: usize, k: usize,
    );

    fn awq_matmul(
        &self,
        weight: &[u8], zeros: &[u8], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );

    fn gptq_matmul(
        &self,
        weight: &[u8], g_idx: &[i32], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );

    fn squeeze_matmul(
        &self,
        weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );

    fn fused_iq1_s_matmul(
        &self,
        weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );

    fn fused_iq2_xxs_matmul(
        &self,
        weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );

    fn fused_awq4_matmul(
        &self,
        weight: &[u8], zeros: &[u8], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );

    fn fused_gptq4_matmul(
        &self,
        weight: &[u8], g_idx: &[i32], scales: &[half::f16],
        input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );

    fn fused_squeeze_matmul(
        &self,
        weight_blocks: &[u8], input: &[E], output: &mut [E],
        m: usize, n: usize, k: usize,
    );
}
```

---

## 3. CPU ISA 实现架构（DATA-CPU-ISA）

### 3.1 ISA 内核结构

```rust
/// CPU 内核（包含 ISA 分发）
pub struct CpuKernels<E: Element> {
    inner: &'static dyn IsaKernels<E>,  // 启动时选择的 ISA 实现
}

impl<E: Element> CpuKernels<E> {
    /// 检测最优 ISA 并初始化（程序启动时调用一次）
    pub fn detect_best() -> Self {
        static DETECTED: OnceLock<IsaType> = OnceLock::new();
        let isa = DETECTED.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            {
                if is_avx512_supported() { return IsaType::Avx512; }
                if is_avx2_supported() { return IsaType::Avx2; }
            }
            #[cfg(target_arch = "aarch64")]
            { return IsaType::Neon; }
            IsaType::Scalar
        });

        let inner: &'static dyn IsaKernels<E> = match isa {
            IsaType::Avx512 => &Avx512Impl::<E>,
            IsaType::Avx2 => &Avx2Impl::<E>,
            IsaType::Neon => &NeonImpl::<E>,
            IsaType::Scalar => &ScalarImpl::<E>,
        };
        Self { inner }
    }
}

/// ISA 类型枚举（仅用于 OnceLock 存储）
#[derive(Clone, Copy)]
enum IsaType {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

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
| **矩阵运算** | gemv, gemm, gemm_bias | 3 |
| **激活函数** | silu, gelu, relu, tanh, swiglu, softmax, exp | 7 |
| **归一化** | rms_norm, layer_norm | 2 |
| **位置编码** | rope, rope_with_pos | 2 |
| **查表** | embedding_lookup | 1 |

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

---

## 5. 融合算子清单（DATA-FUSED）

### 5.1 Transformer 核心融合

| 融合算子 | 组成 | 收益 |
|----------|------|------|
| `fused_qkv_rope` | QKV 投影 + RoPE | 省 3 次 K/V 遍历 |
| `fused_gate_up_swiglu` | Gate 投影 + Up 投影 + SwiGLU | 省中间激活存储 |
| `fused_ffn` | Gate/Up + SwiGLU + Down + Residual | FFN 单次遍历 |
| `fused_ffn_rmsnorm` | FFN + RMSNorm 融合 | 省一次遍历 |
| `fused_linear_residual_rmsnorm` | Linear + Residual + RMSNorm | 后处理融合 |
| `fused_linear_bias_residual_rmsnorm` | Linear + Bias + Residual + RMSNorm | 带 bias 版本 |
| `flash_attention` | QK^T + Softmax + V | O(1) 额外内存 |
| `flash_attention_paged` | 分页 KV Cache 的 Flash Attention | 支持长序列 |

### 5.2 量化融合

| 融合算子 | 组成 | 收益 |
|----------|------|------|
| `fused_qkv_rope_q4` | INT4 QKV 投影 + RoPE | 省解量化中间 f32 |
| `fused_ffn_q4` | INT4 FFN 全流程 | 省解量化中间 f32 |
| `fused_int8_linear_residual_rmsnorm` | INT8 Linear + Residual + RMSNorm | INT8 量化版本 |
| `fused_int4_linear_residual_rmsnorm` | INT4 Linear + Residual + RMSNorm | INT4 量化版本 |

### 5.3 量化格式专用 Matmul

| 融合算子 | 量化格式 | 说明 |
|----------|----------|------|
| `kquant_matmul<E>` | Q2_K ~ Q8_K | K-Quant 系列融合解量化+matmul |
| `iq_matmul<E>` | IQ1_S ~ IQ4_XS | IQ 系列融合解量化+matmul |
| `awq_matmul<E>` | AWQ4 | AWQ 融合解量化+matmul |
| `gptq_matmul<E>` | GPTQ4 | GPTQ 融合解量化+matmul |
| `squeeze_matmul<E>` | SqueezeLLM | SqueezeLLM 融合解量化+matmul |
| `fused_iq1_s_matmul<E>` | IQ1_S | IQ1_S 专用融合 matmul |
| `fused_iq2_xxs_matmul<E>` | IQ2_XXS | IQ2_XXS 专用融合 matmul |
| `fused_awq4_matmul<E>` | AWQ4 | AWQ4 专用融合 matmul |
| `fused_gptq4_matmul<E>` | GPTQ4 | GPTQ4 专用融合 matmul |
| `fused_squeeze_matmul<E>` | SqueezeLLM | SqueezeLLM 专用融合 matmul |

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
Backend (用户指定)
│
├─► CpuBackend
│   │
│   ├─► [L1.5 CPU架构] (编译时 #[cfg] 分支，对用户透明)
│   │
│   ├─► x86_64 (#[cfg(target_arch = "x86_64")])
│   │   └─► ISA (运行时检测)
│   │       ├─► Scalar   → 兜底（仅限无 SIMD 硬件）
│   │       ├─► AVX2     → 256-bit SIMD
│   │       ├─► AVX-512  → 512-bit SIMD
│   │       └─► VNNI     → INT8 点积加速
│   │
│   ├─► ARM (#[cfg(target_arch = "aarch64")])
│   │   └─► ISA (运行时检测)
│   │       ├─► NEON     → 128-bit SIMD (基线)
│   │       ├─► dotprod  → INT8 点积
│   │       └─► SVE      → 可变宽度 SIMD
│   │
│   └─► AppleSilicon (#[cfg(target_os = "macos", target_arch = "aarch64")])
│       └─► ISA (运行时检测)
│           ├─► NEON     → 128-bit SIMD (基线)
│           └─► AMX      → Apple Matrix Extensions
│
│   每个 ISA 实现：
│   └─► impl<E: Element>
│       ├── E = f32  (编译时展开)
│       ├── E = f16  (编译时展开)
│       └── E = bf16 (编译时展开)
│       └── [71 个算子]
│
├─► CudaBackend
│   └─► CudaKernels<E>
│       ├── impl<E: Element> for CudaKernels<E>
│       │   ├── E = f32
│       │   ├── E = f16
│       │   └── E = bf16
│       └── [CUDA kernel 调用]
│
├─► MetalBackend (📋 规划中)
│   └─► [Apple GPU shader 调用]
│
└─► RocmBackend (📋 规划中)
    └─► [AMD GPU HIP kernel 调用]
```

---

## 8. 宏驱动零成本代码生成（ARCH-MACRO）🚨 核心策略

### 8.1 设计原则

**问题**：后端 × ISA × 精度 × 量化格式 的组合爆炸

```
CPU 后端最坏情况：
- 架构: x86_64, ARM, AppleSilicon = 3
- ISA:  Scalar, AVX2, AVX-512, NEON, AMX, ... ≈ 8
- 精度: f32, f16, bf16 = 3
- 量化: 18 种格式
- 算子: 71 个

暴力实现: 8 × 3 × 71 = 1,704+ 函数（不含量化组合）
```

**解法**：宏驱动代码生成，零性能妥协

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
│  批量展开                                                       │
│  ─────────────────────────────────────────────────────────────  │
│  mod avx2_f32  { define_vec_dot!(avx2, f32);  ... }            │
│  mod avx2_f16  { define_vec_dot!(avx2, f16);  ... }            │
│  mod neon_f32  { define_vec_dot!(neon, f32);  ... }            │
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
                                                    │ (纯解量化)                   │ (融合计算)
                                                    ▼                             ▼
                                              ┌───────────┐               ┌───────────┐
                                              │  表 B     │               │  表 C/D   │
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
| **表 D** | `&[u8]` | `&[E]` | `&mut [E]` | `fn fused_ffn_q4(x: &[E], gate: &[u8], ..., out: &mut [E])` |

#### 快速判断口诀

```
1. 看签名有没有 &[u8] 或 &[i8] 作为权重 → 有则是量化相关（表 B/C/D）
2. 量化相关中，输出是 &mut [f32] 固定 → 表 B（纯解量化）
3. 量化相关中，输出是 &mut [E] 泛型 → 表 C/D（量化计算/融合）
4. 表 C vs 表 D：单一操作 vs 多步融合
5. 其余全是表 A（纯浮点）
```

---

### 8.3 算子分类表（MACRO-OPS-TABLE）

#### 表 A：纯浮点算子（32 个）

> 输入/输出都是激活值（或浮点权重），只需 ISA × 精度 展开

| 类别 | 算子 | 展开维度 | 组合数 |
|------|------|----------|--------|
| **向量运算** | vec_dot, vec_add, vec_sub, vec_mul, vec_scale, vec_axpy, vec_sum, vec_max, vec_sum_squares | ISA × 精度 | 9×8×3=216 |
| **矩阵运算** | gemv, gemm, gemm_bias | ISA × 精度 | 3×8×3=72 |
| **激活函数** | silu, gelu, relu, tanh, swiglu, softmax, exp | ISA × 精度 | 7×8×3=168 |
| **归一化** | rms_norm, layer_norm | ISA × 精度 | 2×8×3=48 |
| **位置编码** | rope, rope_with_pos | ISA × 精度 | 2×8×3=48 |
| **查表** | embedding_lookup | ISA × 精度 | 1×8×3=24 |
| **Attention** | flash_attention, flash_attention_paged | ISA × 精度 | 2×8×3=48 |
| **融合(FP权重)** | fused_qkv_rope, fused_gate_up_swiglu, fused_ffn, fused_ffn_rmsnorm, fused_linear_residual_rmsnorm, fused_linear_bias_residual_rmsnorm | ISA × 精度 | 6×8×3=144 |
| **小计** | | | **~768** |

**宏策略**：`define_xxx!(isa, elem)` 模板，一次定义 50 个模板，批量展开

#### 表 B：解量化算子（18 个）

> 输入是量化块 `&[u8]`，输出固定为 `f32`，只需 ISA 展开

| 格式 | 算子 | 展开维度 | 组合数 |
|------|------|----------|--------|
| **K-Quant** | dequant_q2_k, dequant_q3_k, dequant_q4_k, dequant_q5_k, dequant_q6_k, dequant_q8_k | ISA | 6×8=48 |
| **IQ 系列** | dequant_iq1_s, dequant_iq1_m, dequant_iq2_xxs, dequant_iq2_xs, dequant_iq2_s, dequant_iq3_xxs, dequant_iq3_s, dequant_iq4_nl, dequant_iq4_xs | ISA | 9×8=72 |
| **商业格式** | dequant_awq4, dequant_gptq4, dequant_squeeze | ISA | 3×8=24 |
| **小计** | | | **~144** |

**宏策略**：`decode_block!(quant_fmt, block, out)` 解码逻辑独立，SIMD 存储共用

#### 表 C：量化 GEMV/GEMM 算子（6 + 10 = 16 个）

> 权重是量化格式，输入是浮点，需要 ISA × 输入精度 × 量化格式 展开

| 类别 | 算子 | 展开维度 | 组合数 |
|------|------|----------|--------|
| **通用量化 GEMV** | gemv_q8, gemv_q4, gemv_q2, gemv_q1 | ISA × 精度 | 4×8×3=96 |
| **通用量化 GEMM** | gemm_q8, gemm_q4 | ISA × 精度 | 2×8×3=48 |
| **格式专用 Matmul** | kquant_matmul, iq_matmul, awq_matmul, gptq_matmul, squeeze_matmul | ISA × 精度 × 格式子集 | ~120 |
| **IQ 专用融合** | fused_iq1_s_matmul, fused_iq2_xxs_matmul | ISA × 精度 | 2×8×3=48 |
| **商业格式融合** | fused_awq4_matmul, fused_gptq4_matmul, fused_squeeze_matmul | ISA × 精度 | 3×8×3=72 |
| **小计** | | | **~384** |

**宏策略**：
```rust
macro_rules! define_quant_gemv {
    ($isa:ident, $input_elem:ty, $quant_fmt:ident, $block_size:expr) => {
        // 主循环共用，decode_block! 分发格式差异
    };
}
```

#### 表 D：量化融合算子（7 个）

> 完整的量化推理流程融合

| 算子 | 展开维度 | 组合数 |
|------|----------|--------|
| fused_qkv_rope_q4 | ISA × 精度 | 8×3=24 |
| fused_ffn_q4 | ISA × 精度 | 8×3=24 |
| fused_dequant_gemv | ISA × 精度 × 格式 | 8×3×18=432 |
| fused_int8_linear_residual_rmsnorm | ISA × 精度 | 8×3=24 |
| fused_int4_linear_residual_rmsnorm | ISA × 精度 | 8×3=24 |
| **小计** | | **~528** |

### 8.3 量化宏详细设计（MACRO-QUANT-DESIGN）🚨 核心

> 量化算子的宏化是整个架构最复杂的部分，需要处理 **18 种格式 × 8 ISA × 3 精度** 的组合。

#### 8.3.1 量化原语表（quant_primitive!）

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

#### 8.3.2 块解码宏（decode_block!）

```rust
/// 块解码宏 - 每种量化格式的解码逻辑
///
/// 输入: 原始字节块 &[u8]
/// 输出: 解量化后的 f32 数组
///
/// 关键：解码逻辑与 ISA 无关，只有存储操作用 simd_primitive!
macro_rules! decode_block {
    // ========================================================================
    // K-Quant 系列（GGUF 标准格式）
    // ========================================================================

    // Q4_K: 256 元素块，144 字节
    (q4_k, $isa:ident, $block:expr, $out:expr) => {{
        let d = f16::from_le_bytes([$block[0], $block[1]]).to_f32();
        let dmin = f16::from_le_bytes([$block[2], $block[3]]).to_f32();
        let scales = &$block[4..16];
        let qs = &$block[16..144];

        for j in 0..32 {
            let scale_idx = j / 4;
            let sc = (scales[scale_idx] & 0x3F) as f32;
            let m = (scales[scale_idx + 6] & 0x3F) as f32;

            for i in 0..8 {
                let idx = j * 8 + i;
                let q = quant_primitive!(scalar, unpack_int4, qs[idx / 2], idx);
                $out[idx] = d * sc * (q as f32) - dmin * m;
            }
        }
    }};

    // Q8_K: 256 元素块，292 字节
    (q8_k, $isa:ident, $block:expr, $out:expr) => {{
        let d = f32::from_le_bytes([$block[0], $block[1], $block[2], $block[3]]);
        let qs = &$block[4..260];
        for i in 0..256 {
            $out[i] = d * (qs[i] as i8 as f32);
        }
    }};

    (q2_k, $isa:ident, $block:expr, $out:expr) => {{ /* 84 bytes */ }};
    (q3_k, $isa:ident, $block:expr, $out:expr) => {{ /* 110 bytes */ }};
    (q5_k, $isa:ident, $block:expr, $out:expr) => {{ /* 176 bytes */ }};
    (q6_k, $isa:ident, $block:expr, $out:expr) => {{ /* 210 bytes */ }};

    // ========================================================================
    // IQ 系列（超低比特码本量化）
    // ========================================================================

    (iq1_s, $isa:ident, $block:expr, $out:expr) => {{
        let d = f16::from_le_bytes([$block[0], $block[1]]).to_f32();
        // 使用 IQ1_S_GRID 查表
    }};

    (iq4_nl, $isa:ident, $block:expr, $out:expr) => {{
        let d = f16::from_le_bytes([$block[0], $block[1]]).to_f32();
        let qs = &$block[2..18];
        for i in 0..32 {
            let q = quant_primitive!(scalar, unpack_int4, qs[i / 2], i);
            $out[i] = d * quant_primitive!(any, iq4_nl_lookup, q);
        }
    }};

    // ========================================================================
    // 商业格式（AWQ / GPTQ）
    // ========================================================================

    (awq4, $isa:ident, $packed:expr, $zeros:expr, $scales:expr, $out:expr, $group_idx:expr) => {{
        let scale = $scales[$group_idx].to_f32();
        let zero = quant_primitive!(scalar, unpack_int4, $zeros[$group_idx / 2], $group_idx) as f32;
        for i in 0..128 {
            let idx = $group_idx * 128 + i;
            let q = quant_primitive!(scalar, unpack_int4, $packed[idx / 2], idx);
            $out[i] = quant_primitive!(scalar, f32, apply_scale, q, scale, zero);
        }
    }};

    (gptq4, $isa:ident, $packed:expr, $g_idx:expr, $scales:expr, $out:expr) => {{
        for i in 0..128 {
            let group = $g_idx[i] as usize;
            let scale = $scales[group].to_f32();
            let q = quant_primitive!(scalar, unpack_int4, $packed[i / 2], i);
            $out[i] = (q as f32) * scale;
        }
    }};
}
```

#### 8.3.3 量化 GEMV 模板（define_quant_gemv!）

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

#### 8.3.4 量化格式常量表（QUANT-CONST-TABLE）

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

#### 8.3.5 批量展开量化算子

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

#### 8.3.6 IQ 码本常量

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

### 8.4 simd_primitive! 完整映射表（MACRO-PRIMITIVE-COMPLETE）🚨 核心维护点

> **AI CODER 注意**：这是整个宏架构的核心！添加新 ISA 只需扩展此表。

#### 8.4.1 操作清单（每个 ISA × 精度 组合必须实现）

**A. 计算操作（22 个）**

| 操作 | 签名 | 说明 |
|------|------|------|
| `lanes` | `() -> usize` | SIMD 向量宽度（编译时常量） |
| `zero` | `() -> Vec` | 零向量 |
| `splat` | `(val) -> Vec` | 标量广播到所有通道 |
| `load` | `(ptr) -> Vec` | 从内存加载（可能非对齐） |
| `store` | `(ptr, vec)` | 存储到内存（可能非对齐） |
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

#### 8.4.2 完整映射表实现

```rust
/// simd_primitive! 宏 - ISA 抽象的核心
///
/// 设计原则：
/// 1. 每个 (ISA, 精度, 操作) 三元组映射到一个 intrinsic
/// 2. 算子模板只使用此宏，对 ISA 完全透明
/// 3. 添加新 ISA 只需扩展此表，所有算子自动获得支持
macro_rules! simd_primitive {
    // ========================================================================
    // Scalar 兜底（仅限无 SIMD 硬件，禁止在有 SIMD 能力的硬件上使用）
    // ========================================================================

    // --- f32 架构常量 ---
    (scalar, f32, num_regs) => { usize::MAX };         // 标量无寄存器压力
    (scalar, f32, optimal_tile_m) => { 1 };
    (scalar, f32, optimal_tile_n_vecs) => { 1 };
    (scalar, f32, prefetch_distance) => { 0 };          // 标量不做预取
    (scalar, f32, has_native_fp16) => { false };
    (scalar, f32, has_native_bf16) => { false };

    // --- f32 计算操作 ---
    (scalar, f32, lanes) => { 1 };
    (scalar, f32, zero) => { 0.0f32 };
    (scalar, f32, splat, $v:expr) => { $v };
    (scalar, f32, load, $p:expr) => { unsafe { *$p } };
    (scalar, f32, store, $p:expr, $v:expr) => { unsafe { *$p = $v } };
    (scalar, f32, add, $a:expr, $b:expr) => { $a + $b };
    (scalar, f32, sub, $a:expr, $b:expr) => { $a - $b };
    (scalar, f32, mul, $a:expr, $b:expr) => { $a * $b };
    (scalar, f32, div, $a:expr, $b:expr) => { $a / $b };
    (scalar, f32, fma, $a:expr, $b:expr, $c:expr) => { $c + $a * $b };
    (scalar, f32, neg, $a:expr) => { -$a };
    (scalar, f32, max, $a:expr, $b:expr) => { $a.max($b) };
    (scalar, f32, min, $a:expr, $b:expr) => { $a.min($b) };
    (scalar, f32, reduce_sum, $v:expr) => { $v };
    (scalar, f32, reduce_max, $v:expr) => { $v };
    (scalar, f32, exp, $a:expr) => { $a.exp() };
    (scalar, f32, recip, $a:expr) => { 1.0 / $a };
    (scalar, f32, sqrt, $a:expr) => { $a.sqrt() };
    (scalar, f32, rsqrt, $a:expr) => { 1.0 / $a.sqrt() };
    (scalar, f32, prefetch, $p:expr, $dist:expr) => { /* no-op */ };

    // --- f16 (软件转换) ---
    (scalar, f16, lanes) => { 1 };
    (scalar, f16, load_cvt, $p:expr) => { unsafe { (*$p).to_f32() } };
    (scalar, f16, store_cvt, $p:expr, $v:expr) => { unsafe { *$p = f16::from_f32($v) } };
    // f16 的算术操作转换为 f32 计算

    // --- bf16 (软件转换) ---
    (scalar, bf16, lanes) => { 1 };
    (scalar, bf16, load_cvt, $p:expr) => { unsafe { (*$p).to_f32() } };
    (scalar, bf16, store_cvt, $p:expr, $v:expr) => { unsafe { *$p = bf16::from_f32($v) } };

    // ========================================================================
    // AVX2 (x86_64, 256-bit, 8×f32)
    // ========================================================================

    // --- f32 架构常量 ---
    (avx2, f32, num_regs) => { 16 };              // ymm0-ymm15
    (avx2, f32, optimal_tile_m) => { 6 };          // 6行 × 2列 = 12累加器, 留4临时
    (avx2, f32, optimal_tile_n_vecs) => { 2 };     // 2个ymm = 16列
    (avx2, f32, prefetch_distance) => { 256 };     // 256B = 4 cache lines
    (avx2, f32, has_native_fp16) => { false };     // F16C仅做转换，非原生运算
    (avx2, f32, has_native_bf16) => { false };

    // --- f32 计算操作 ---
    (avx2, f32, lanes) => { 8 };
    (avx2, f32, zero) => { _mm256_setzero_ps() };
    (avx2, f32, splat, $v:expr) => { _mm256_set1_ps($v) };
    (avx2, f32, load, $p:expr) => { _mm256_loadu_ps($p) };
    (avx2, f32, store, $p:expr, $v:expr) => { _mm256_storeu_ps($p, $v) };
    (avx2, f32, add, $a:expr, $b:expr) => { _mm256_add_ps($a, $b) };
    (avx2, f32, sub, $a:expr, $b:expr) => { _mm256_sub_ps($a, $b) };
    (avx2, f32, mul, $a:expr, $b:expr) => { _mm256_mul_ps($a, $b) };
    (avx2, f32, div, $a:expr, $b:expr) => { _mm256_div_ps($a, $b) };
    (avx2, f32, fma, $a:expr, $b:expr, $c:expr) => { _mm256_fmadd_ps($a, $b, $c) };
    (avx2, f32, neg, $a:expr) => { _mm256_xor_ps($a, _mm256_set1_ps(-0.0)) };
    (avx2, f32, max, $a:expr, $b:expr) => { _mm256_max_ps($a, $b) };
    (avx2, f32, min, $a:expr, $b:expr) => { _mm256_min_ps($a, $b) };
    (avx2, f32, reduce_sum, $v:expr) => { avx2_hsum_ps($v) };  // 辅助函数
    (avx2, f32, reduce_max, $v:expr) => { avx2_hmax_ps($v) };  // 辅助函数
    (avx2, f32, exp, $a:expr) => { avx2_exp_ps($a) };  // 多项式近似
    (avx2, f32, recip, $a:expr) => { _mm256_rcp_ps($a) };
    (avx2, f32, sqrt, $a:expr) => { _mm256_sqrt_ps($a) };
    (avx2, f32, rsqrt, $a:expr) => { _mm256_rsqrt_ps($a) };
    (avx2, f32, prefetch, $p:expr, $dist:expr) => { _mm_prefetch($p as *const i8, _MM_HINT_T0) };

    // --- f16 (F16C 转换) ---
    (avx2, f16, lanes) => { 8 };  // 一次处理 8 个 f16
    (avx2, f16, load_cvt, $p:expr) => {
        _mm256_cvtph_ps(_mm_loadu_si128($p as *const __m128i))
    };
    (avx2, f16, store_cvt, $p:expr, $v:expr) => {
        _mm_storeu_si128($p as *mut __m128i, _mm256_cvtps_ph($v, _MM_FROUND_TO_NEAREST_INT))
    };

    // --- bf16 (位转换) ---
    (avx2, bf16, lanes) => { 8 };
    (avx2, bf16, load_cvt, $p:expr) => {
        // bf16 左移 16 位变成 f32
        let raw = _mm_loadu_si128($p as *const __m128i);
        let expanded = _mm256_cvtepu16_epi32(raw);
        let shifted = _mm256_slli_epi32(expanded, 16);
        _mm256_castsi256_ps(shifted)
    };
    (avx2, bf16, store_cvt, $p:expr, $v:expr) => {
        // f32 右移 16 位变成 bf16
        let as_int = _mm256_castps_si256($v);
        let shifted = _mm256_srli_epi32(as_int, 16);
        let packed = _mm256_packus_epi32(shifted, shifted);
        let lo = _mm256_castsi256_si128(packed);
        _mm_storeu_si128($p as *mut __m128i, lo)
    };

    // ========================================================================
    // AVX-512 (x86_64, 512-bit, 16×f32)
    // ========================================================================

    // --- f32 架构常量 ---
    (avx512, f32, num_regs) => { 32 };             // zmm0-zmm31
    (avx512, f32, optimal_tile_m) => { 14 };       // 14行 × 2列 = 28累加器, 留4临时
    (avx512, f32, optimal_tile_n_vecs) => { 2 };   // 2个zmm = 32列
    (avx512, f32, prefetch_distance) => { 512 };   // 512B = 8 cache lines
    (avx512, f32, has_native_fp16) => { /* runtime: is_x86_feature_detected!("avx512fp16") */ };
    (avx512, f32, has_native_bf16) => { /* runtime: is_x86_feature_detected!("avx512bf16") */ };

    // --- f32 计算操作 ---
    (avx512, f32, lanes) => { 16 };
    (avx512, f32, zero) => { _mm512_setzero_ps() };
    (avx512, f32, splat, $v:expr) => { _mm512_set1_ps($v) };
    (avx512, f32, load, $p:expr) => { _mm512_loadu_ps($p) };
    (avx512, f32, store, $p:expr, $v:expr) => { _mm512_storeu_ps($p, $v) };
    (avx512, f32, add, $a:expr, $b:expr) => { _mm512_add_ps($a, $b) };
    (avx512, f32, sub, $a:expr, $b:expr) => { _mm512_sub_ps($a, $b) };
    (avx512, f32, mul, $a:expr, $b:expr) => { _mm512_mul_ps($a, $b) };
    (avx512, f32, div, $a:expr, $b:expr) => { _mm512_div_ps($a, $b) };
    (avx512, f32, fma, $a:expr, $b:expr, $c:expr) => { _mm512_fmadd_ps($a, $b, $c) };
    (avx512, f32, neg, $a:expr) => { _mm512_xor_ps($a, _mm512_set1_ps(-0.0)) };
    (avx512, f32, max, $a:expr, $b:expr) => { _mm512_max_ps($a, $b) };
    (avx512, f32, min, $a:expr, $b:expr) => { _mm512_min_ps($a, $b) };
    (avx512, f32, reduce_sum, $v:expr) => { _mm512_reduce_add_ps($v) };
    (avx512, f32, reduce_max, $v:expr) => { _mm512_reduce_max_ps($v) };
    (avx512, f32, exp, $a:expr) => { avx512_exp_ps($a) };
    (avx512, f32, recip, $a:expr) => { _mm512_rcp14_ps($a) };
    (avx512, f32, sqrt, $a:expr) => { _mm512_sqrt_ps($a) };
    (avx512, f32, rsqrt, $a:expr) => { _mm512_rsqrt14_ps($a) };
    (avx512, f32, prefetch, $p:expr, $dist:expr) => { _mm_prefetch($p as *const i8, _MM_HINT_T0) };

    // --- f16 (AVX512-FP16 或回退到 F16C) ---
    (avx512, f16, lanes) => { 16 };
    (avx512, f16, load_cvt, $p:expr) => {
        _mm512_cvtph_ps(_mm256_loadu_si256($p as *const __m256i))
    };
    (avx512, f16, store_cvt, $p:expr, $v:expr) => {
        _mm256_storeu_si256($p as *mut __m256i,
            _mm512_cvtps_ph($v, _MM_FROUND_TO_NEAREST_INT))
    };

    // ========================================================================
    // NEON (ARM, 128-bit, 4×f32)
    // ========================================================================

    // --- f32 架构常量 ---
    (neon, f32, num_regs) => { 32 };              // v0-v31
    (neon, f32, optimal_tile_m) => { 8 };          // 8行 × 3列 = 24累加器, 留8临时
    (neon, f32, optimal_tile_n_vecs) => { 3 };     // 3个vq = 12列
    (neon, f32, prefetch_distance) => { 128 };     // 128B = 2 cache lines
    (neon, f32, has_native_fp16) => { true };       // NEON FP16 原生支持
    (neon, f32, has_native_bf16) => { false };
    (neon, f32, has_dot_prod) => { /* runtime: is_aarch64_feature_detected!("dotprod") */ };

    // --- f32 计算操作 ---
    (neon, f32, lanes) => { 4 };
    (neon, f32, zero) => { vdupq_n_f32(0.0) };
    (neon, f32, splat, $v:expr) => { vdupq_n_f32($v) };
    (neon, f32, load, $p:expr) => { vld1q_f32($p) };
    (neon, f32, store, $p:expr, $v:expr) => { vst1q_f32($p, $v) };
    (neon, f32, add, $a:expr, $b:expr) => { vaddq_f32($a, $b) };
    (neon, f32, sub, $a:expr, $b:expr) => { vsubq_f32($a, $b) };
    (neon, f32, mul, $a:expr, $b:expr) => { vmulq_f32($a, $b) };
    (neon, f32, div, $a:expr, $b:expr) => { vdivq_f32($a, $b) };
    (neon, f32, fma, $a:expr, $b:expr, $c:expr) => { vfmaq_f32($c, $a, $b) };
    (neon, f32, neg, $a:expr) => { vnegq_f32($a) };
    (neon, f32, max, $a:expr, $b:expr) => { vmaxq_f32($a, $b) };
    (neon, f32, min, $a:expr, $b:expr) => { vminq_f32($a, $b) };
    (neon, f32, reduce_sum, $v:expr) => { vaddvq_f32($v) };
    (neon, f32, reduce_max, $v:expr) => { vmaxvq_f32($v) };
    (neon, f32, exp, $a:expr) => { neon_exp_f32($a) };  // 多项式近似
    (neon, f32, recip, $a:expr) => { vrecpeq_f32($a) };
    (neon, f32, sqrt, $a:expr) => { vsqrtq_f32($a) };
    (neon, f32, rsqrt, $a:expr) => { vrsqrteq_f32($a) };
    (neon, f32, prefetch, $p:expr, $dist:expr) => { __pld($p as *const u8) };

    // --- f16 (NEON FP16 原生支持) ---
    (neon, f16, lanes) => { 8 };  // float16x8_t
    (neon, f16, load_cvt, $p:expr) => {
        vcvt_f32_f16(vld1_f16($p))  // 4 个 f16 → 4 个 f32
    };
    (neon, f16, store_cvt, $p:expr, $v:expr) => {
        vst1_f16($p, vcvt_f16_f32($v))
    };
}

/// SIMD 宽度常量宏
macro_rules! simd_lanes {
    (scalar, $elem:ty) => { 1 };
    (avx2, f32) => { 8 };
    (avx2, f16) => { 8 };
    (avx2, bf16) => { 8 };
    (avx512, f32) => { 16 };
    (avx512, f16) => { 16 };
    (avx512, bf16) => { 16 };
    (neon, f32) => { 4 };
    (neon, f16) => { 8 };
}

/// SIMD 对齐要求宏
macro_rules! simd_align {
    (scalar, $elem:ty) => { 1 };
    (avx2, $elem:ty) => { 32 };
    (avx512, $elem:ty) => { 64 };
    (neon, $elem:ty) => { 16 };
}
```

#### 8.4.3 辅助函数（reduce 操作）

```rust
/// AVX2 水平求和（没有原生指令，需要手动实现）
#[inline(always)]
unsafe fn avx2_hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

/// AVX2 水平最大
#[inline(always)]
unsafe fn avx2_hmax_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(hi, lo);
    let hi64 = _mm_movehl_ps(max128, max128);
    let max64 = _mm_max_ps(max128, hi64);
    let hi32 = _mm_shuffle_ps(max64, max64, 1);
    _mm_cvtss_f32(_mm_max_ss(max64, hi32))
}

/// AVX2 指数函数（7 阶多项式近似）
#[inline(always)]
unsafe fn avx2_exp_ps(x: __m256) -> __m256 {
    // Cephes 风格的 exp 近似
    // 精度：|error| < 2e-7 for x ∈ [-88, 88]
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(0.166666666666666019037);
    let c4 = _mm256_set1_ps(0.0416666666665409524128);
    let c5 = _mm256_set1_ps(0.00833333333332249791693);
    // ... 完整实现
    c1 // 占位符
}

/// NEON 指数函数
#[inline(always)]
unsafe fn neon_exp_f32(x: float32x4_t) -> float32x4_t {
    // 类似的多项式近似
    vdupq_n_f32(1.0) // 占位符
}
```

### 8.5 后端统一架构（UNIFIED-BACKEND-MACRO）🚨 跨后端复用

> 宏策略不仅适用于 CPU，也可统一 CPU + CUDA 的分发逻辑。

#### 8.5.1 统一架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         宏驱动统一后端架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 0: Kernels Trait 签名（宏生成，CPU/CUDA 共享）                       │
│  ────────────────────────────────────────────────────────────────────────   │
│  define_kernels_trait!() → 生成 71 个算子签名                               │
│                                                                             │
│                              │                                              │
│              ┌───────────────┴───────────────┐                              │
│              ▼                               ▼                              │
│                                                                             │
│  Layer 1: CPU 实现                    Layer 1: CUDA 实现                    │
│  ──────────────────                   ────────────────────                  │
│  simd_primitive!(isa, elem, op)       cubin_dispatch!(arch, quant_fmt)      │
│         │                                    │                              │
│         ▼                                    ▼                              │
│  Rust SIMD intrinsics                 FFI → .cubin entry point              │
│                                                                             │
│                                                                             │
│  Layer 2: 分发逻辑（宏生成）                                                │
│  ──────────────────────────                                                 │
│  match quant_type {                                                         │
│      Q4K => kernels.dequant_q4_k(...),                                      │
│      Q8K => kernels.dequant_q8_k(...),                                      │
│      ...                                                                    │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 8.5.2 Kernels Trait 签名生成宏

```rust
/// 统一 Kernels Trait 签名（CPU + CUDA 共享）
macro_rules! define_kernels_trait {
    () => {
        pub trait Kernels<E: Element>: Send + Sync {
            // ================================================================
            // 表 A：纯浮点算子（32 个）
            // ================================================================
            define_table_a_signatures!();

            // ================================================================
            // 表 B：解量化算子（18 个，输出固定 f32）
            // ================================================================
            fn dequant_q2_k(&self, block: &[u8], out: &mut [f32]);
            fn dequant_q3_k(&self, block: &[u8], out: &mut [f32]);
            fn dequant_q4_k(&self, block: &[u8], out: &mut [f32]);
            fn dequant_q5_k(&self, block: &[u8], out: &mut [f32]);
            fn dequant_q6_k(&self, block: &[u8], out: &mut [f32]);
            fn dequant_q8_k(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq1_s(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq1_m(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq2_xxs(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq2_xs(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq2_s(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq3_xxs(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq3_s(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq4_nl(&self, block: &[u8], out: &mut [f32]);
            fn dequant_iq4_xs(&self, block: &[u8], out: &mut [f32]);
            fn dequant_awq4(&self, packed: &[u8], zeros: &[u8], scales: &[f16], out: &mut [f32]);
            fn dequant_gptq4(&self, packed: &[u8], g_idx: &[i32], scales: &[f16], out: &mut [f32]);
            fn dequant_squeeze(&self, block: &[u8], out: &mut [f32]);

            // ================================================================
            // 表 C：量化计算算子（16 个）
            // ================================================================
            fn kquant_matmul(&self, weight: &[u8], input: &[E], output: &mut [E],
                            quant_type: QuantType, m: usize, n: usize, k: usize);
            fn iq_matmul(&self, weight: &[u8], input: &[E], output: &mut [E],
                        quant_type: QuantType, m: usize, n: usize, k: usize);
            fn awq_matmul(&self, weight: &[u8], zeros: &[u8], scales: &[f16],
                         input: &[E], output: &mut [E], m: usize, n: usize, k: usize);
            fn gptq_matmul(&self, weight: &[u8], g_idx: &[i32], scales: &[f16],
                          input: &[E], output: &mut [E], m: usize, n: usize, k: usize);
            // ... 其他量化 matmul

            // ================================================================
            // 表 D：量化融合算子（5 个）
            // ================================================================
            fn fused_qkv_rope_q4(&self, /* ... */);
            fn fused_ffn_q4(&self, /* ... */);
            fn fused_dequant_gemv(&self, weight: &[u8], input: &[E], output: &mut [E],
                                  quant_type: QuantType, m: usize, n: usize, k: usize);
        }
    };
}

/// 表 A 签名生成宏
macro_rules! define_table_a_signatures {
    () => {
        // 向量运算
        fn vec_dot(&self, a: &[E], b: &[E]) -> E;
        fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]);
        fn vec_sub(&self, a: &[E], b: &[E], out: &mut [E]);
        fn vec_mul(&self, a: &[E], b: &[E], out: &mut [E]);
        fn vec_scale(&self, x: &mut [E], s: E);
        fn vec_axpy(&self, y: &mut [E], a: E, x: &[E]);
        fn vec_sum(&self, x: &[E]) -> E;
        fn vec_max(&self, x: &[E]) -> E;
        fn vec_sum_squares(&self, x: &[E]) -> E;

        // 矩阵运算
        fn gemv(&self, a: &[E], x: &[E], y: &mut [E], m: usize, n: usize);
        fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize);

        // 激活函数
        fn silu(&self, x: &[E], out: &mut [E]);
        fn gelu(&self, x: &[E], out: &mut [E]);
        fn relu(&self, x: &[E], out: &mut [E]);
        fn swiglu(&self, gate: &[E], up: &[E], out: &mut [E]);
        fn softmax(&self, x: &[E], out: &mut [E]);

        // 归一化
        fn rms_norm(&self, x: &[E], weight: &[E], out: &mut [E], eps: f32);
        fn layer_norm(&self, x: &[E], gamma: &[E], beta: &[E], out: &mut [E], eps: f32);

        // 位置编码
        fn rope(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, interleaved: bool);

        // Attention
        fn flash_attention(&self, q: &[E], k: &[E], v: &[E], out: &mut [E],
                          seq_len: usize, num_heads: usize, head_dim: usize, scale: f32, causal: bool);
    };
}
```

#### 8.5.3 CUDA FFI 分发宏

```rust
/// CUDA 后端：宏生成 FFI 调用包装
macro_rules! impl_cuda_kernels {
    () => {
        impl<E: Element> Kernels<E> for CudaBackend {
            // 表 B：解量化（分发到对应 sm_XX cubin）
            fn dequant_q4_k(&self, block: &CudaSlice<u8>, out: &mut CudaSlice<f32>) {
                unsafe {
                    match self.sm_arch {
                        80 => cubin_sm80::dequant_q4_k(block.ptr(), out.ptr(), block.len()),
                        86 => cubin_sm86::dequant_q4_k(block.ptr(), out.ptr(), block.len()),
                        89 => cubin_sm89::dequant_q4_k(block.ptr(), out.ptr(), block.len()),
                        90 => cubin_sm90::dequant_q4_k(block.ptr(), out.ptr(), block.len()),
                        _ => panic!("Unsupported SM arch"),
                    }
                }
            }

            // 表 C/D：量化 matmul（使用 C++ 模板实例化）
            fn kquant_matmul(&self, weight: &CudaSlice<u8>, input: &CudaSlice<E>,
                            output: &mut CudaSlice<E>, quant_type: QuantType,
                            m: usize, n: usize, k: usize) {
                unsafe {
                    // C++ 模板：template<int BITS> void quant_gemm(...)
                    // 编译时已实例化 BITS=1,2,3,4,5,6,8
                    match quant_type.bits() {
                        4 => cubin_quant_gemm_4bit(self.sm_arch, ...),
                        8 => cubin_quant_gemm_8bit(self.sm_arch, ...),
                        _ => panic!("Unsupported quant bits"),
                    }
                }
            }
        }
    };
}
```

#### 8.5.4 分发逻辑生成宏

```rust
/// 量化类型分发宏（CPU/CUDA 共享）
macro_rules! dispatch_quant_type {
    ($kernels:expr, $quant_type:expr, $method:ident, $($args:expr),*) => {
        match $quant_type {
            QuantType::Q2K => $kernels.dequant_q2_k($($args),*),
            QuantType::Q3K => $kernels.dequant_q3_k($($args),*),
            QuantType::Q4K => $kernels.dequant_q4_k($($args),*),
            QuantType::Q5K => $kernels.dequant_q5_k($($args),*),
            QuantType::Q6K => $kernels.dequant_q6_k($($args),*),
            QuantType::Q8K => $kernels.dequant_q8_k($($args),*),
            QuantType::IQ1S => $kernels.dequant_iq1_s($($args),*),
            QuantType::IQ1M => $kernels.dequant_iq1_m($($args),*),
            QuantType::IQ2XXS => $kernels.dequant_iq2_xxs($($args),*),
            QuantType::IQ2XS => $kernels.dequant_iq2_xs($($args),*),
            QuantType::IQ2S => $kernels.dequant_iq2_s($($args),*),
            QuantType::IQ3XXS => $kernels.dequant_iq3_xxs($($args),*),
            QuantType::IQ3S => $kernels.dequant_iq3_s($($args),*),
            QuantType::IQ4NL => $kernels.dequant_iq4_nl($($args),*),
            QuantType::IQ4XS => $kernels.dequant_iq4_xs($($args),*),
            QuantType::AWQ4 => panic!("AWQ4 需要额外参数"),
            QuantType::GPTQ4 => panic!("GPTQ4 需要额外参数"),
            QuantType::Squeeze => $kernels.dequant_squeeze($($args),*),
        }
    };
}
```

### 8.6 ISA × 精度 支持矩阵

| ISA | f32 | f16 | bf16 | 说明 |
|-----|-----|-----|------|------|
| **Scalar** | ✅ 原生 | ✅ 软件转换 | ✅ 软件转换 | 兜底（仅限无 SIMD 硬件） |
| **AVX2** | ✅ 原生 | ✅ F16C 转换 | ✅ 位转换 | x86_64 基线 |
| **AVX-512** | ✅ 原生 | ⚡ AVX512-FP16 | ⚡ AVX512-BF16 | 需运行时检测扩展 |
| **VNNI** | - | - | - | INT8 点积加速 |
| **NEON** | ✅ 原生 | ⚡ FP16 原生 | ✅ 位转换 | ARM 基线 |
| **SVE** | ✅ 原生 | ⚡ FP16 原生 | ⚡ BF16 原生 | ARM 服务器 |
| **AMX** | - | - | ⚡ 原生 | Apple Silicon 矩阵加速 |

**图例**：✅ 必须实现 | ⚡ 硬件原生支持 | - 不适用

### 8.7 后端量化格式支持策略

| 后端 | 支持格式 | 策略 |
|------|----------|------|
| **CPU** | **全部 18 种** | 软件解码，兜底后端 |
| **CUDA** | Q4_K, Q8_K, AWQ4, GPTQ4 | Tensor Core 友好 |
| **Metal** | Q4_K, Q8_K | Apple GPU 常见 |
| **ROCm** | Q4_K, Q8_K, GPTQ4 | AMD 常见格式 |

### 8.8 AI CODER 维护指南

#### 添加新 ISA

1. 在 `simd_primitive!` 宏中添加该 ISA 的所有操作映射
2. 定义 `simd_lanes!(new_isa, elem)` 常量
3. 所有算子自动通过宏展开获得新 ISA 支持

```rust
// 示例：添加 SVE 支持
macro_rules! simd_primitive {
    // ... 现有规则 ...

    // SVE f32
    (sve, f32, zero) => { svdup_f32(0.0) };
    (sve, f32, load, $p:expr) => { svld1_f32(svptrue_b32(), $p) };
    (sve, f32, fma, $a:expr, $b:expr, $c:expr) => { svmla_f32_x(svptrue_b32(), $c, $a, $b) };
    // ...
}

macro_rules! simd_lanes {
    (sve, f32) => { svcntw() };  // SVE 运行时确定宽度
}
```

#### 添加新精度

1. 实现 `Element` trait
2. 在 `simd_primitive!` 中为每个 ISA 添加该精度的操作
3. 批量展开时包含新精度

#### 添加新量化格式

1. 在 `decode_block!` 宏中添加解码规则
2. 定义块大小常量
3. 使用 `define_quant_gemv!` 生成 GEMV 实现
4. 添加 `dequant_xxx` 函数

```rust
// 示例：添加新量化格式 Q3_S
macro_rules! decode_block {
    (q3_s, $block:expr, $out:expr) => {{
        // Q3_S 特定解码逻辑
    }};
}

const Q3_S_BLOCK_SIZE: usize = 256;
const Q3_S_BLOCK_BYTES: usize = 104;

// 自动获得所有 ISA × 精度 的 GEMV 实现
mod avx2_f32_q3s { define_quant_gemv!(avx2, f32, q3_s, 256); }
mod avx2_f16_q3s { define_quant_gemv!(avx2, f16, q3_s, 256); }
// ...
```

#### 添加新算子

1. 判断算子类别（表 A/B/C/D）
2. 编写 `define_xxx!` 模板宏，使用 `simd_primitive!` 原语
3. 批量展开

```rust
// 示例：添加 gelu_tanh 算子（表 A 类）
macro_rules! define_gelu_tanh {
    ($isa:ident, $elem:ty) => {
        #[inline(always)]
        pub fn gelu_tanh(x: &[$elem], out: &mut [$elem]) {
            const LANES: usize = simd_lanes!($isa, $elem);
            // 使用 simd_primitive! 实现
        }
    };
}

// 批量展开
mod avx2_f32  { define_gelu_tanh!(avx2, f32);  }
mod avx2_f16  { define_gelu_tanh!(avx2, f16);  }
mod neon_f32  { define_gelu_tanh!(neon, f32);  }
// ...
```

#### 性能调优某个 ISA × 精度 组合

宏生成的代码是基线实现。对于热点路径，可以覆写：

```rust
mod avx512_f32 {
    // 宏生成的基线
    define_gemm!(avx512, f32);

    // 手写覆写（更激进的优化）
    #[inline(always)]
    pub fn gemm_optimized(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        // 手写 AVX-512 GEMM，使用寄存器分块、预取等
    }
}
```

### 8.9 AI CODER 维护检查清单（MAINTENANCE-CHECKLIST）🚨 必读

> **每次修改宏系统前必须阅读此清单**

#### 8.9.1 添加新 ISA 检查清单

```
□ 步骤 1：扩展 simd_primitive! 表
  ├─ 添加所有 21 个操作的映射（见 §8.4.1 操作清单）
  ├─ 每个操作必须有对应的 intrinsic 或软件实现
  └─ 验证：grep -c "(new_isa, f32," 应该 >= 21

□ 步骤 2：扩展 simd_lanes! 宏
  ├─ 添加 (new_isa, f32), (new_isa, f16), (new_isa, bf16) 三条规则
  └─ 验证：编译通过

□ 步骤 3：扩展 simd_align! 宏
  └─ 添加 (new_isa, $elem:ty) => { 对齐字节数 }

□ 步骤 4：更新 expand_all_isa! 宏
  ├─ 添加 #[cfg(target_arch = "xxx")] mod new_isa { ... }
  └─ 验证：所有算子自动获得新 ISA 支持

□ 步骤 5：更新 §8.6 ISA × 精度 支持矩阵
  └─ 添加新行，标注支持的精度和硬件特性

□ 步骤 6：测试
  ├─ cargo test --features new_isa
  └─ 基准测试验证性能
```

#### 8.9.2 添加新量化格式检查清单

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

□ 步骤 4：生成量化 GEMV
  ├─ expand_all_quant_formats! 中添加 mod new_fmt { ... }
  └─ 验证：所有 ISA × 精度 组合自动生成

□ 步骤 5：更新 dispatch_quant_type! 宏
  └─ 添加 QuantType::NewFmt => kernels.dequant_new_fmt(...)

□ 步骤 6：测试
  ├─ 单元测试：decode 正确性
  ├─ 集成测试：GEMV 输出与参考一致
  └─ 性能测试：与 llama.cpp 对比
```

#### 8.9.3 添加新算子检查清单

```
□ 步骤 1：判断算子类别
  ├─ 签名无量化权重 → 表 A（纯浮点）
  ├─ 输出固定 f32 → 表 B（解量化）
  ├─ 量化权重 + 泛型输出 → 表 C（量化计算）
  └─ 多步融合 + 量化 → 表 D（量化融合）

□ 步骤 2：编写算子模板宏
  ├─ 命名：define_xxx!(isa, elem)
  ├─ 使用 simd_primitive! 原语，不直接使用 intrinsic
  ├─ 包含尾部处理（非 LANES 对齐部分）
  └─ 验证：scalar 实现正确

□ 步骤 3：批量展开
  ├─ 在对应模块中调用 expand_all_isa!(define_xxx)
  └─ 验证：编译通过

□ 步骤 4：添加到 Kernels trait
  ├─ 在 define_table_X_signatures! 中添加签名
  └─ 在各后端实现中添加调用

□ 步骤 5：更新 §9.1 算子统计表
  └─ 更新对应类别数量

□ 步骤 6：测试
  ├─ 正确性测试（与标量/参考实现对比）
  └─ 性能测试（各 ISA 加速比）
```

#### 8.9.4 常见错误检查

```
❌ 错误 1：直接使用 intrinsic 而不是 simd_primitive!
   → 导致新 ISA 无法自动支持
   → 检查：grep -r "_mm256\|_mm512\|vaddq" src/cpu_kernels/*.rs

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
```

#### 8.9.5 性能验证基准

| 操作 | 期望加速比（vs Scalar） | 备注 |
|------|------------------------|------|
| vec_dot (f32) | AVX2: 6-8×, AVX512: 12-14× | SIMD 宽度 |
| gemv (f32) | AVX2: 5-7×, AVX512: 10-12× | 内存带宽限制 |
| rms_norm | AVX2: 4-6×, AVX512: 8-10× | 两次遍历 |
| softmax | AVX2: 3-5× | exp 近似开销 |
| dequant_q4_k | AVX2: 3-4× | 解码开销 |
| quant_gemv | AVX2: 2-3× | 解码 + 计算平衡 |

---

## 9. 算子统计

### 9.1 算子模板数（需维护）

| 类别 | 数量 | 宏策略 |
|------|------|--------|
| 向量运算 | 9 | 表 A |
| 矩阵运算 | 3 | 表 A |
| 激活函数 | 7 | 表 A |
| 归一化 | 2 | 表 A |
| 位置编码 | 2 | 表 A |
| 查表 | 1 | 表 A |
| Attention | 2 | 表 A |
| 融合算子（FP 权重） | 6 | 表 A |
| 解量化 | 18 | 表 B |
| 量化 GEMV/GEMM | 6 | 表 C |
| 量化格式专用 Matmul | 10 | 表 C |
| 融合算子（量化权重） | 5 | 表 D |
| **模板总计** | **71** | |

### 9.2 宏展开后实现数（自动生成）

| 类别 | 展开公式 | 实现数 |
|------|----------|--------|
| 表 A 纯浮点 | 32 算子 × 8 ISA × 3 精度 | ~768 |
| 表 B 解量化 | 18 格式 × 8 ISA | ~144 |
| 表 C 量化计算 | 16 算子 × 8 ISA × 3 精度 | ~384 |
| 表 D 量化融合 | 5 算子 × 8 ISA × 3 精度 + 特殊 | ~528 |
| **展开总计** | | **~1,824** |

> 注：实际数量取决于后端支持矩阵，CPU 全覆盖，GPU 选择性支持
