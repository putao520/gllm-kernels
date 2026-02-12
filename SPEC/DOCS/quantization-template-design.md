# 量化内核模板化实施设计

## 1. 当前状态分析

### 1.1 现有实现 (kernels.cu)

```cpp
// 当前: 每种位宽单独实现
extern "C" __global__ void int8_mm(...) { /* Int8 特定逻辑 */ }
extern "C" __global__ void int4_mm(...) { /* Int4 特定逻辑，几乎与 int8 相同 */ }
// int2_mm, int1_mm 未实现
```

**问题**:
- 代码重复严重
- int2/int1 位宽未实现
- 维护困难，修改需要同步多处

### 1.2 目标设计

```cpp
// 模板化实现
template<int BITS>
__global__ void quantized_mm(
    const float* input,
    const uint8_t* weight,
    float* output,
    QuantizedConfig params
) {
    // BITS 是编译时常量
    // 编译器会为每个特化生成优化代码
}

// 显式实例化 (编译时)
template __global__ void quantized_mm<1>(...);
template __global__ void quantized_mm<2>(...);
template __global__ void quantized_mm<4>(...);
template __global__ void quantized_mm<8>(...);
```

## 2. 技术实现方案

### 2.1 C++ 模板设计

#### 数据结构

```cpp
template<int BITS>
struct QuantizedTraits {
    static constexpr int VALUES_PER_BYTE = 8 / BITS;
    static constexpr int MAX_VALUE = (1 << (BITS - 1)) - 1;
    static constexpr int MIN_VALUE = -(1 << (BITS - 1));

    __device__ static float decode(uint8_t packed, int index);
};

template<> struct QuantizedTraits<8> {
    static constexpr int VALUES_PER_BYTE = 1;
    __device__ static float decode(uint8_t packed, int index) {
        return static_cast<float>(static_cast<int8_t>(packed));
    }
};

template<> struct QuantizedTraits<4> {
    static constexpr int VALUES_PER_BYTE = 2;
    __device__ static float decode(uint8_t packed, int index) {
        uint8_t nibble = (index & 1) ? (packed >> 4) : (packed & 0x0f);
        return static_cast<float>(static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xf0) : nibble));
    }
};

template<> struct QuantizedTraits<2> {
    static constexpr int VALUES_PER_BYTE = 4;
    __device__ static float decode(uint8_t packed, int index) {
        uint8_t pair = (packed >> ((index & 3) * 2)) & 0x03;
        // 2-bit 有符号: 00=-2, 01=-1, 10=0, 11=1
        return static_cast<float>(static_cast<int8_t>(pair) - 2);
    }
};

template<> struct QuantizedTraits<1> {
    static constexpr int VALUES_PER_BYTE = 8;
    __device__ static float decode(uint8_t packed, int index) {
        uint8_t bit = (packed >> (index & 7)) & 0x01;
        return (bit != 0) ? 1.0f : -1.0f;
    }
};
```

#### 内核实现

```cpp
template<int BITS>
__global__ void quantized_mm(
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,
    float* __restrict__ output,
    QuantizedConfig params
) {
    constexpr int VPB = QuantizedTraits<BITS>::VALUES_PER_BYTE;

    uint32_t total = params.m * params.n;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t row = i / params.n;
        uint32_t col = i - row * params.n;

        const float* in_row = input + row * params.input_stride;
        const uint8_t* w_row = weight + col * params.weight_stride;

        float sum = 0.0f;
        for (uint32_t kk = 0; kk < params.k; ++kk) {
            // 模板函数内联，零运行时开销
            float w = QuantizedTraits<BITS>::decode(
                w_row[kk / VPB],
                kk
            );
            sum = fmaf(in_row[kk], w * params.scale, sum);
        }
        output[row * params.output_stride + col] = sum;
    }
}

// 显式实例化
template __global__ void quantized_mm<1>(const float*, const uint8_t*, float*, QuantizedConfig);
template __global__ void quantized_mm<2>(const float*, const uint8_t*, float*, QuantizedConfig);
template __global__ void quantized_mm<4>(const float*, const uint8_t*, float*, QuantizedConfig);
template __global__ void quantized_mm<8>(const float*, const uint8_t*, float*, QuantizedConfig);
```

### 2.2 Rust 侧集成

#### 配置结构

```rust
#[repr(C)]
pub struct QuantizedConfig {
    pub m: u32,      // batch size
    pub n: u32,      // output columns
    pub k: u32,      // input columns / shared dimension
    pub input_stride: u32,
    pub weight_stride: u32,
    pub output_stride: u32,
    pub scale: f32,
}

pub enum QuantizedBits {
    Int1,
    Int2,
    Int4,
    Int8,
}
```

#### 调度接口

```rust
impl CudaBackend {
    pub fn quantized_mm_launch(
        &self,
        bits: QuantizedBits,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<u8>,
        output: &mut CudaSlice<f32>,
        config: QuantizedConfig,
    ) -> BackendResult<()> {
        let (func_name, m, n, k) = match bits {
            QuantizedBits::Int1 => ("quantized_mm<1>", ...),
            QuantizedBits::Int2 => ("quantized_mm<2>", ...),
            QuantizedBits::Int4 => ("quantized_mm<4>", ...),
            QuantizedBits::Int8 => ("quantized_mm<8>", ...),
        };

        // 启动对应模板实例
        self.kernels.quantized_mm(
            &self.stream,
            input,
            weight,
            output,
            &config,
            m, n, k,
        )?;
        Ok(())
    }
}
```

## 3. 验收标准

| 标准 | 验证方法 |
|------|----------|
| 代码复用 | 单个模板函数覆盖所有位宽 |
| 编译通过 | sm_80/86/89/90 所有架构编译成功 |
| 功能正确 | 与现有 int8/int4 内核输出一致 |
| 性能无损失 | 与手写内核性能相当 |
| int2/int1 可用 | 新增位宽功能正常工作 |

## 4. 实施步骤

1. **模板设计**: 在 kernels.cu 中添加模板实现
2. **特化实现**: 为每种位宽实现 Traits
3. **显式实例化**: 添加 template 声明
4. **Rust 集成**: 更新 cuda_kernels.rs 和 cuda_backend.rs
5. **测试验证**: 运行现有量化测试确保兼容性
6. **性能基准**: 对比手写内核性能

## 5. 约束 (FROZEN)

- ✅ **必须**: 使用 `template<int BITS>` 统一实现
- ✅ **必须**: 编译时显式实例化所有模板
- ❌ **禁止**: 运行时 if-else 判断位宽
- ❌ **禁止**: 为每种位宽单独编写内核代码
