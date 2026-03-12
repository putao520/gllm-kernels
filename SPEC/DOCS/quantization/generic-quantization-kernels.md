# 泛型量化内核架构设计

> **关联需求**: REQ-QUANT-004 (模板化量化内核)
> **关联约束**: CLAUDE.md ARCH-QUANT-TEMPLATE (FROZEN)
> **状态**: 🟡 设计中

---

## 1. 架构目标

### 1.1 核心问题

当前 `cpu_kernels/mod.rs` 中的量化计算实现存在以下问题：

| 问题 | 当前实现 | 影响 |
|------|----------|------|
| **非泛型** | 每个量化位宽独立函数 (`matmul_int8`, `matmul_int4`, `matmul_int2`, `matmul_int1`) | 代码重复，维护成本高 |
| **f32 特化** | `linear()` 只支持 f32 权重 | 无法利用量化加速 |
| **手动分派** | 调用方需要根据 dtype 选择函数 | 易出错，扩展性差 |

### 1.2 设计目标

| 目标 | 描述 | 验收标准 |
|------|------|----------|
| **泛型 MatMul** | 单一 trait 接口支持所有精度类型 | `MatMul<T>` for T in {F32, F16, BF16, I8, I4, I2, I1} |
| **零运行时开销** | 编译时单态化，无动态分发 | 性能等同于手写特化版本 |
| **统一存储抽象** | 量化权重统一使用 `u8` 容器 | 支持 `PackedU8(PackedBits)` |
| **自动路径选择** | 根据权重格式自动选择最优计算路径 | 分离权重 → 3×小矩阵乘法，融合权重 → 1×大矩阵乘法 |

---

## 2. Rust 泛型架构设计

### 2.1 核心 Trait 定义

```rust
/// 量化矩阵乘法 Trait
///
/// 泛型参数 T: 数据类型 (f32, f16, bf16, i8, PackedI4, PackedI2, PackedI1)
pub trait QuantizedMatMul<T: DTypeTrait> {
    /// 计算 output = input @ weight + bias
    ///
    /// # 参数
    /// - `input`: [m, k] 输入矩阵 (始终为 f32)
    /// - `weight`: 量化权重 [n, k] (存储格式由 T 决定)
    /// - `scales`: 每行/每块的量化 scale [n] 或 [n_blocks]
    /// - `output`: [m, n] 输出矩阵 (始终为 f32)
    /// - `m`: batch/seq_len
    /// - `n`: output_dim
    /// - `k`: input_dim
    fn matmul(
        input: &[f32],
        weight: &[T::Storage],
        scales: &[f16],
        bias: Option<&[f32]>,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError>;
}

/// 数据类型 Trait
pub trait DTypeTrait: Sized + Copy + 'static {
    /// 存储类型 (f32, f16, u8)
    type Storage: Copy;
    /// 反量化到 f32
    fn dequantize(scaled: Self::Storage, scale: f16) -> f32;
    /// 每个值占用的 bit 数
    const BITS: u8;
    /// 是否是打包类型 (int4/2/1)
    const IS_PACKED: bool;
}
```

### 2.2 Trait 实现模板

```rust
// F32 实现 (自研 SIMD 内核)
impl QuantizedMatMul<F32> for CpuBackend {
    fn matmul(...) {
        // 自研分块 SIMD matmul (禁止 faer/OpenBLAS/MKL)
        matmul_tiled_simd(&mut dst, &lhs, &rhs, m, n, k);
    }
}

// Int8 实现
impl QuantizedMatMul<I8> for CpuBackend {
    fn matmul(...) {
        // SIMD 加速的 int8 反量化 + FMA
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                // 展开内层循环，启用 SIMD
                for kk in (0..k).step_by(8) {
                    let w_vec = load_int8_weight(&weight[j * k + kk..]);
                    let x_vec = dequantize_to_f32(w_vec, scales[j]);
                    sum = simd_fma(x_vec, &input[i * k + kk..], sum);
                }
                output[i * n + j] = sum + bias.map_or(0.0, |b| b[j]);
            }
        }
    }
}

// Int4 实现 (使用 PackedBits)
impl QuantizedMatMul<PackedI4> for CpuBackend {
    fn matmul(...) {
        let values_per_byte = 2; // Int4
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let byte_idx = j * k + kk / 2;
                    let nibble = if kk % 2 == 0 {
                        weight[byte_idx] & 0x0f
                    } else {
                        weight[byte_idx] >> 4
                    };
                    let w = unpack_int4(nibble) as f32 * scales[j] as f32;
                    sum += input[i * k + kk] * w;
                }
                output[i * n + j] = sum + bias.map_or(0.0, |b| b[j]);
            }
        }
    }
}
```

### 2.3 类型系统映射

```rust
// DType → DTypeTrait 映射
impl DTypeTrait for F32Type {
    type Storage = f32;
    fn dequantize(s: f32, _: f16) -> f32 { s }
    const BITS: u8 = 32;
    const IS_PACKED: bool = false;
}

impl DTypeTrait for I8Type {
    type Storage = i8;
    fn dequantize(s: i8, scale: f16) -> f32 { s as f32 * scale as f32 }
    const BITS: u8 = 8;
    const IS_PACKED: bool = false;
}

impl DTypeTrait for PackedI4Type {
    type Storage = u8;
    fn dequantize(p: u8, scale: f16) -> f32 {
        let value = if p & 0x08 != 0 { (p as i8) - 16 } else { p as i8 };
        value as f32 * scale as f32
    }
    const BITS: u8 = 4;
    const IS_PACKED: bool = true;
}

// 类似定义 Int2, Int1
```

### 2.4 编译时单态化

```rust
/// 通用 Linear 层
///
/// 编译器为每个 T 生成独立代码，零运行时开销
pub fn linear_generic<T: DTypeTrait>(
    backend: &CpuBackend,
    input: &[f32],
    weight: &[T::Storage],
    scales: &[f16],
    bias: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), BackendError>
where
    CpuBackend: QuantizedMatMul<T>,
{
    <CpuBackend as QuantizedMatMul<T>>::matmul(input, weight, scales, bias, output, m, n, k)
}

// 使用示例：
// linear_generic::<F32>(...)     → 生成 f32 专用版本
// linear_generic::<I8>(...)      → 生成 int8 专用版本
// linear_generic::<PackedI4>(...) → 生成 int4 专用版本
```

---

## 3. QKV 投影统一 API

### 3.1 通用 QKV 投影接口

```rust
/// QKV 投影 + RoPE 统一接口
///
/// 自动选择最优路径：
/// - 分离权重 → 3×小矩阵乘法 (最优缓存性能)
/// - 融合权重 → 1×大矩阵乘法
pub enum QkvWeightFormat<'a, T: DTypeTrait> {
    /// 分离的 Q, K, V 权重 (最优)
    Separated {
        q_weight: &'a [T::Storage],
        k_weight: &'a [T::Storage],
        v_weight: &'a [T::Storage],
        q_scales: Option<&'a [f16]>,
        k_scales: Option<&'a [f16]>,
        v_scales: Option<&'a [f16]>,
    },
    /// 融合的 QKV 权重
    Fused {
        qkv_weight: &'a [T::Storage],
        scales: Option<&'a [f16]>,
    },
}

pub fn qkv_projection_rope_generic<T: DTypeTrait>(
    backend: &CpuBackend,
    input: &[f32],
    qkv_weights: &QkvWeightFormat<T>,
    positions: &[i32],
    q_output: &mut [f32],
    k_output: &mut [f32],
    v_output: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rope_scale: f32,
) -> Result<(), BackendError>
where
    CpuBackend: QuantizedMatMul<T>,
{
    match qkv_weights {
        QkvWeightFormat::Separated { q_weight, k_weight, v_weight, q_scales, k_scales, v_scales } => {
            // 最优路径：3×独立小矩阵乘法
            let q_out = num_heads * head_dim;
            let kv_out = num_kv_heads * head_dim;

            linear_generic::<T>(
                backend,
                input,
                q_weight,
                q_scales.unwrap_or(&[]),
                None,
                q_output,
                seq_len,
                q_out,
                hidden_size,
            )?;

            linear_generic::<T>(
                backend,
                input,
                k_weight,
                k_scales.unwrap_or(&[]),
                None,
                k_output,
                seq_len,
                kv_out,
                hidden_size,
            )?;

            linear_generic::<T>(
                backend,
                input,
                v_weight,
                v_scales.unwrap_or(&[]),
                None,
                v_output,
                seq_len,
                kv_out,
                hidden_size,
            )?;

            apply_rope_separated(
                q_output, k_output, positions,
                seq_len, num_heads, num_kv_heads,
                head_dim, rotary_dim, rope_theta, rope_scale,
            )?;
        }
        QkvWeightFormat::Fused { qkv_weight, scales } => {
            // 回退路径：1×大矩阵乘法 (兼容性)
            let qkv_stride = num_heads * head_dim + 2 * num_kv_heads * head_dim;
            let mut qkv_buf = vec![0.0f32; seq_len * qkv_stride];

            linear_generic::<T>(
                backend,
                input,
                qkv_weight,
                scales.unwrap_or(&[]),
                None,
                &mut qkv_buf,
                seq_len,
                qkv_stride,
                hidden_size,
            )?;

            split_and_apply_rope(
                &qkv_buf, q_output, k_output, v_output,
                positions, seq_len, num_heads, num_kv_heads,
                head_dim, rotary_dim, rope_theta, rope_scale,
            )?;
        }
    }
    Ok(())
}
```

### 3.2 自动格式检测

```rust
impl CpuBackend {
    /// 从权重名称自动检测 QKV 格式
    pub fn detect_qkv_format<T: DTypeTrait>(
        &self,
        weights: &TensorMap,
        layer: usize,
    ) -> Result<QkvWeightFormat<T>, BackendError> {
        // 优先检测分离权重
        let q_name = format!("model.layers.{}.self_attn.q_proj.weight", layer);
        let k_name = format!("model.layers.{}.self_attn.k_proj.weight", layer);
        let v_name = format!("model.layers.{}.self_attn.v_proj.weight", layer);

        if let (Some(q), Some(k), Some(v)) = (
            weights.get_tensor::<T>(&q_name),
            weights.get_tensor::<T>(&k_name),
            weights.get_tensor::<T>(&v_name),
        ) {
            return Ok(QkvWeightFormat::Separated {
                q_weight: q.data(),
                k_weight: k.data(),
                v_weight: v.data(),
                q_scales: q.scales(),
                k_scales: k.scales(),
                v_scales: v.scales(),
            });
        }

        // 回退到融合权重
        let qkv_name = format!("model.layers.{}.self_attn.qkv_proj.weight", layer);
        if let Some(qkv) = weights.get_tensor::<T>(&qkv_name) {
            return Ok(QkvWeightFormat::Fused {
                qkv_weight: qkv.data(),
                scales: qkv.scales(),
            });
        }

        Err(BackendError::MissingWeight(format!(
            "QKV weights not found for layer {}", layer
        )))
    }
}
```

---

## 4. GPU 端实现映射（JIT 统一路径）

GPU 量化 matmul 不使用手写 `.cu` 内核或 AOT CUBIN。量化解码 + FMA 逻辑通过 JIT 编译器统一路径实现：

1. 量化解码定义为 `extern "C"` 标量函数（与 CPU 路径共享 Phase 0-2）
2. Phase 3 根据 `Platform::Cuda { sm_version }` 生成 PTX 量化 matmul kernel
3. `cuModuleLoadData` 加载 PTX，driver 编译为目标 GPU 的 SASS

详见 `SPEC/04-GPU-BACKEND.md`。

---

## 5. 集成计划

### 5.1 阶段 1: Trait 定义 (REQ-QUANT-004.1)

| 文件 | 修改 | 状态 |
|------|------|------|
| `src/cpu_kernels/traits.rs` | 新增 `DTypeTrait`, `QuantizedMatMul` | 🔵 待实现 |
| `src/cpu_kernels/mod.rs` | 重新导出 traits | 🔵 待实现 |

### 5.2 阶段 2: Trait 实现 (REQ-QUANT-004.2)

| 文件 | 修改 | 状态 |
|------|------|------|
| `src/cpu_kernels/f32_impl.rs` | `QuantizedMatMul<F32>` for `CpuBackend` | 🔵 待实现 |
| `src/cpu_kernels/i8_impl.rs` | `QuantizedMatMul<I8>` for `CpuBackend` | 🔵 待实现 |
| `src/cpu_kernels/i4_impl.rs` | `QuantizedMatMul<PackedI4>` for `CpuBackend` | 🔵 待实现 |
| `src/cpu_kernels/i2_impl.rs` | `QuantizedMatMul<PackedI2>` for `CpuBackend` | 🔵 待实现 |
| `src/cpu_kernels/i1_impl.rs` | `QuantizedMatMul<PackedI1>` for `CpuBackend` | 🔵 待实现 |

### 5.3 阶段 3: QKV 统一 API (REQ-QUANT-004.3)

| 文件 | 修改 | 状态 |
|------|------|------|
| `src/cpu_kernels/qkv_generic.rs` | 新增 `qkv_projection_rope_generic` | 🔵 待实现 |
| `src/cpu_backend.rs` | 替换 `fused_qkv_rope` 为泛型版本 | 🔵 待实现 |

### 5.4 阶段 4: GPU 端集成 — JIT 路径 (REQ-QUANT-004.4)

| 文件 | 修改 | 状态 |
|------|------|------|
| `src/scalar_ops/quant_decode.rs` | 量化解码 `extern "C"` 标量函数 | 🔵 待实现 |
| `src/compiler/codegen/ptx.rs` | PtxCodeGen 量化 matmul kernel 生成 | 🔵 待实现 |
| `src/compiler/codegen/amdgpu.rs` | AmdgpuCodeGen 量化 matmul kernel 生成 | 🔵 待实现 |

---

## 6. 性能约束

### 6.1 编译时保证

| 约束 | 验证方法 |
|------|----------|
| **零运行时开销** | `cargo asm` 验证生成的汇编等同于手写版本 |
| **完全内联** | `#[inline(always)]` 确保单态化代码内联 |
| **无动态分发** | 禁止 `dyn Trait`，仅使用泛型 |

### 6.2 SIMD 优化要求

| 后端 | SIMD 指令 | 实现要求 |
|------|-----------|----------|
| **x86_64 AVX2** | `_mm256_fma_ps` | 8×f32 并行 |
| **x86_64 AVX-512** | `_mm512_fma_ps` | 16×f32 并行 |
| **AArch64 NEON** | `vfmaq_f32` | 4×f32 并行 |
| **量化反量化** | 批量解包到 F32 寄存器 | 避免标量循环 |

---

## 7. 验收标准

### 7.1 功能正确性

- [ ] 所有量化类型的 `matmul` 输出与 f32 参考误差 < 0.1%
- [ ] QKV 投影输出与 HuggingFace 参考匹配 (测试: `test_k_values_real_weights.rs`)
- [ ] RoPE 应用正确 (position 0 不变，position > 0 旋转)

### 7.2 性能验证

- [ ] 泛型版本性能 ≥ 手写特化版本 (基准测试)
- [ ] 分离权重路径 (3×小矩阵) 性能 > 融合权重路径
- [ ] Int4/Int2/Int1 内存占用 < F32 的 25%

### 7.3 编译验证

- [ ] `cargo check` 通过 (无类型错误)
- [ ] `cargo asm` 确认完全单态化 (无虚调用)
- [ ] 各个量化类型生成独立汇编

---

## 8. 参考文档

- **SPEC/02-ARCHITECTURE.md**: L3 GPU-Pure 架构定义
- **CLAUDE.md**: ARCH-QUANT-TEMPLATE 约束
- **tests/test_k_values_real_weights.rs**: QKV 正确性验证
