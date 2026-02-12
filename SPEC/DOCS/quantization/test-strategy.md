# 泛型量化内核测试策略

> **关联需求**: REQ-QUANT-004, REQ-QUANT-005
> **关联设计**: [generic-quantization-kernels.md](./generic-quantization-kernels.md)

---

## 1. 测试类型覆盖

### 1.1 单元测试 (TEST-UNIT-001)

| 测试 ID | 测试目标 | 输入 | 期望输出 |
|---------|----------|------|----------|
| TEST-UNIT-TRAIT-001 | DTypeTrait bits() 正确性 | 各 DType | 正确 bit 数 |
| TEST-UNIT-TRAIT-002 | DTypeTrait values_per_byte() | 各 DType | 正确打包比 |
| TEST-UNIT-TRAIT-003 | DTypeTrait storage_bytes_for() | 各 DType, 各种 value count | 正确存储大小 |
| TEST-UNIT-MATMUL-001 | QuantizedMatMul<F32> 正确性 | 随机输入, f32 权重 | 与标量参考实现结果一致 |
| TEST-UNIT-MATMUL-002 | QuantizedMatMul<I8> 正确性 | 随机输入, int8 权重 | 与 f32 参考误差 < 0.1% |
| TEST-UNIT-MATMUL-003 | QuantizedMatMul<PackedI4> 正确性 | 随机输入, int4 权重 | 与 f32 参考误差 < 0.1% |
| TEST-UNIT-MATMUL-004 | QuantizedMatMul<PackedI2> 正确性 | 随机输入, int2 权重 | 与 f32 参考误差 < 0.1% |
| TEST-UNIT-MATMUL-005 | QuantizedMatMul<PackedI1> 正确性 | 随机输入, int1 权重 | 与 f32 参考误差 < 0.1% |

### 1.2 集成测试 (TEST-INT-001)

| 测试 ID | 测试目标 | 验收标准 |
|---------|----------|----------|
| TEST-INT-QKV-001 | QKV 投影正确性 (分离权重) | 与 HuggingFace SmolLM2 参考 K 值匹配 |
| TEST-INT-QKV-002 | QKV 投影正确性 (融合权重) | 与分离权重路径输出一致 |
| TEST-INT-QKV-003 | QKV 自动格式检测 | 正确识别 Q/K/V 分离权重 |
| TEST-INT-QKV-004 | RoPE 应用正确性 | position 0 不变，position > 0 旋转 |
| TEST-INT-QKV-005 | 泛型 QKV 与非泛型输出一致 | `qkv_projection_rope_generic<T>` == `fused_qkv_rope` |

### 1.3 性能测试 (TEST-PERF-001)

| 测试 ID | 测试目标 | 验收标准 |
|---------|----------|----------|
| TEST-PERF-001 | 泛型 vs 手写性能对比 | 泛型版本 ≥ 手写版本 (100%) |
| TEST-PERF-002 | 分离权重 vs 融合权重 | 分离权重 > 融合权重 (20%+) |
| TEST-PERF-003 | 量化内存占用 | Int4 < F32 * 25% |

---

## 2. 测试用例规范

### 2.1 Trait 正确性测试

```rust
#[test]
fn test_dtype_trait_bits() {
    assert_eq!(F32Type::BITS, 32);
    assert_eq!(F16Type::BITS, 16);
    assert_eq!(I8Type::BITS, 8);
    assert_eq!(PackedI4Type::BITS, 4);
    assert_eq!(PackedI2Type::BITS, 2);
    assert_eq!(PackedI1Type::BITS, 1);
}

#[test]
fn test_dtype_trait_values_per_byte() {
    assert_eq!(F32Type::IS_PACKED, false);
    assert_eq!(I8Type::IS_PACKED, false);
    assert_eq!(PackedI4Type::IS_PACKED, true);
    assert_eq!(PackedI4Type::values_per_byte(), 2);
    assert_eq!(PackedI2Type::values_per_byte(), 4);
    assert_eq!(PackedI1Type::values_per_byte(), 8);
}
```

### 2.2 MatMul 正确性测试模板

```rust
/// MatMul 正确性测试宏
macro_rules! test_matmul_correctness {
    ($dtype:ty, $name:ident) => {
        #[test]
        fn $name() {
            // 准备测试数据
            let m = 2;
            let n = 3;
            let k = 4;
            let input = vec![1.0f32; m * k];
            let mut weight = vec![<$dtype as DTypeTrait>::Storage::default(); n * k];
            // ... 初始化权重 ...
            let scales = vec![f16::from_f32(0.1); n];
            let mut output = vec![0.0f32; m * n];

            // 调用泛型 matmul
            <CpuBackend as QuantizedMatMul<$dtype>>::matmul(
                &input, &weight, &scales, None, &mut output, m, n, k
            ).unwrap();

            // 与 f32 参考比较
            let mut expected = vec![0.0f32; m * n];
            matmul_f32_reference(&input, &weight, &scales, &mut expected, m, n, k);

            for (i, (out, exp)) in output.iter().zip(expected.iter()).enumerate() {
                let error = (out - exp).abs() / exp.abs().max(1e-5);
                assert!(error < 0.001, "Mismatch at {}: {} vs {}", i, out, exp);
            }
        }
    };
}

// 为每个类型实例化测试
test_matmul_correctness!(F32Type, test_matmul_f32_correctness);
test_matmul_correctness!(I8Type, test_matmul_i8_correctness);
test_matmul_correctness!(PackedI4Type, test_matmul_i4_correctness);
// ...
```

### 2.3 QKV 投影测试

```rust
#[test]
fn test_qkv_projection_separated_matches_fused() {
    // SmolLM2-135M config
    let num_heads = 9;
    let num_kv_heads = 3;
    let head_dim = 64;
    let hidden_size = 576;
    let seq_len = 5;

    // 准备输入
    let input = vec![0.1f32; seq_len * hidden_size];
    let positions = vec![0i32, 1, 2, 3, 4];

    // 分离权重
    let q_weight = vec![0.01f32; num_heads * head_dim * hidden_size];
    let k_weight = vec![0.01f32; num_kv_heads * head_dim * hidden_size];
    let v_weight = vec![0.01f32; num_kv_heads * head_dim * hidden_size];
    let separated_weights = QkvWeightFormat::Separated {
        q_weight: &q_weight,
        k_weight: &k_weight,
        v_weight: &v_weight,
        scales: None,
    };

    // 融合权重 (拼接)
    let qkv_stride = num_heads * head_dim + 2 * num_kv_heads * head_dim;
    let mut qkv_weight = vec![0.01f32; qkv_stride * hidden_size];
    // ... 复制权重 ...
    let fused_weights = QkvWeightFormat::Fused {
        qkv_weight: &qkv_weight,
        scales: None,
    };

    // 分别计算
    let mut q_sep = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k_sep = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut v_sep = vec![0.0f32; seq_len * num_kv_heads * head_dim];

    let mut q_fused = q_sep.clone();
    let mut k_fused = k_sep.clone();
    let mut v_fused = v_sep.clone();

    qkv_projection_rope_generic::<F32>(
        &input, &separated_weights, &positions,
        &mut q_sep, &mut k_sep, &mut v_sep,
        // ...
    ).unwrap();

    qkv_projection_rope_generic::<F32>(
        &input, &fused_weights, &positions,
        &mut q_fused, &mut k_fused, &mut v_fused,
        // ...
    ).unwrap();

    // 验证输出一致
    assert_close(&q_sep, &q_fused, 1e-5);
    assert_close(&k_sep, &k_fused, 1e-5);
    assert_close(&v_sep, &v_fused, 1e-5);
}

#[test]
fn test_qkv_rope_position_0_unchanged() {
    // Position 0: cos=1, sin=0, 值不变
    // ... 见现有 test_rope_separated_basic ...
}
```

---

## 3. HuggingFace 参考测试

### 3.1 测试框架

使用 `docs/hf_comparison.py` 生成参考值：

```python
# 生成 HuggingFace 参考值
python docs/hf_comparison.py > hf_ref.txt

# 示例输出:
# K AFTER RoPE (position 0, first 10): [-0.91015625, 0.30859375, ...]
```

### 3.2 Rust 测试断言

```rust
#[test]
fn test_qkv_matches_huggingface() {
    // 使用真实 SmolLM2-135M 权重
    let token_ids = [504, 3575, 282, 4649, 314]; // "The capital of France is"

    // ... 加载权重 ...

    // 计算 QKV
    qkv_projection_rope_generic::<I4>(...).unwrap();

    // 验证 position 0 的 K 值
    let hf_ref = [
        -0.91015625, 0.30859375, -0.06591796875, -0.328125,
        0.2578125, 0.055908203125, -0.46875, 0.74609375,
        -0.119140625, -0.376953125,
    ];

    for (i, (&actual, &expected)) in k_output.iter().zip(hf_ref.iter()).take(10).enumerate() {
        assert_close(actual, expected, 1e-2, "K[{}] mismatch", i);
    }
}
```

---

## 4. 编译时验证

### 4.1 汇编检查

```bash
# 验证泛型完全单态化
cargo asm --lib --cpu_kernels::matmul_generic --release | grep matmul

# 期望输出: 无虚调用，每个类型独立函数
# _ZN12gllm_kernels...matmul_genericHYPERFO17B8b_...
```

### 4.2 内联验证

```rust
#[test]
fn verify_mono_morphization() {
    // 确保编译器为每个类型生成独立代码
    // 使用 #[inline(always)] 强制内联
}
```

---

## 5. 性能基准

### 5.1 基准测试框架

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_matmul<T: DTypeTrait>(c: &mut Criterion, name: &str)
where
    CpuBackend: QuantizedMatMul<T>,
{
    c.bench_function(name, |b| {
        b.iter(|| {
            // ... 准备数据 ...
            <CpuBackend as QuantizedMatMul<T>>::matmul(
                black_box(&input),
                black_box(&weight),
                black_box(&scales),
                None,
                black_box(&mut output),
                m, n, k,
            )
        });
    });
}

criterion_group!(benches, bench_matmul::<F32>, bench_matmul::<I8>, ...);
criterion_main!(benches);
```

### 5.2 对比基准

| 测试 | 基线 | 目标 |
|------|------|------|
| 泛型 F32 | 手写 f32 matmul | ≥ 100% |
| 泛型 I8 | 手写 int8 matmul | ≥ 100% |
| 分离权重 QKV | 融合权重 QKV | ≥ 120% |

---

## 6. 禁止行为检查

### 6.1 代码审查清单

- [ ] 无 `dyn Trait` 动态分发
- [ ] 无运行时 `match dtype` 分派
- [ ] 所有量化计算使用泛型 trait
- [ ] 手写函数逐步迁移到泛型实现

### 6.2 自动化检测

```rust
#[cfg(doctest)]
doc_comment! {
    include_str!("../docs/quantization/generic-quantization-kernels.md");
}

// Clippy lint
#[deny(trivial_casts)]
#[deny(clippy::match_same_arms)]
```
