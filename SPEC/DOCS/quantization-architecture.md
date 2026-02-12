# 量化内核架构设计

## 背景

量化内核需要对不同位宽 (1/2/4/8-bit) 进行支持。传统做法是为每种位宽单独实现内核，导致代码重复和维护困难。

## 架构决策 (ARCH-QUANT-TEMPLATE)

### 使用 CUDA C++ 模板统一实现

#### 技术方案

```cpp
// 单一模板实现，覆盖所有位宽
template<int BITS>
__global__ void quantized_mm(
    const float* input,
    const uint8_t* weight,
    float* output,
    QuantizedConfig params
) {
    // 模板参数 BITS 在编译时确定
    // 编译器会为每个特化生成优化代码
    // ...
}

// 编译时实例化
template __global__ void quantized_mm<1>(...);
template __global__ void quantized_mm<2>(...);
template __global__ void quantized_mm<4>(...);
template __global__ void quantized_mm<8>(...);
```

#### 优势

| 优势 | 说明 |
|------|------|
| **代码复用** | 一套代码逻辑，覆盖所有位宽 |
| **类型安全** | 编译时检查，常量折叠优化 |
| **零运行时开销** | 模板实例化在编译时完成 |
| **易于维护** | 修改一处，所有位宽受益 |

### Rust 侧调度

```rust
pub enum QuantizedBits {
    Int1,
    Int2,
    Int4,
    Int8,
}

// 运行时调度到对应模板实例
fn launch_quantized_mm(
    bits: QuantizedBits,
    // ...
) -> BackendResult<()> {
    match bits {
        QuantizedBits::Int1 => cuda_backend.int1_mm_launch(...),
        QuantizedBits::Int2 => cuda_backend.int2_mm_launch(...),
        QuantizedBits::Int4 => cuda_backend.int4_mm_launch(...),
        QuantizedBits::Int8 => cuda_backend.int8_mm_launch(...),
    }
}
```

## 相关需求

- **REQ-QUANT-001**: 多精度类型系统
- **REQ-QUANT-002**: 块式量化格式
- **REQ-QUANT-003**: 即时反量化内核
- **REQ-QUANT-004**: 模板化量化内核

## 约束 (FROZEN)

- ❌ **禁止**: 为每种位宽单独编写内核代码
- ❌ **禁止**: 使用运行时 if-else 判断位宽进行条件编译
- ✅ **必须**: 使用 C++ 模板 `<int BITS>` 统一实现
- ✅ **必须**: 在 nvcc 编译时显式实例化所有需要的模板
