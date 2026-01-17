# gllm-kernels

Low-level GPU attention kernels with runtime backend selection.

## SPEC 位置

- `./SPEC/`

## 核心架构约束（🚨 FROZEN - 铁律）

### 零成本抽象铁律（ARCH-API-001 🚨 最高优先级）

**所有实现必须是零成本的，违反即拒绝合并**。

| 零成本机制 | 说明 |
|------------|------|
| 泛型单态化 | `<T: Float>` 编译时展开，无运行时开销 |
| const 分支消除 | `T::TYPE_ID` 是 const，match 分支被编译器优化掉 |
| `#[inline(always)]` | 强制内联，无函数调用开销 |
| 原始切片 | `&[T]` 无任何抽象层 |

**禁止的模式**：

| 禁止 | 原因 |
|------|------|
| ❌ `dyn Trait` | vtable = 运行时开销 |
| ❌ `Box<dyn Trait>` | 堆分配 + vtable |
| ❌ 运行时类型判断 | `if type == f32` = 分支开销 |
| ❌ `Tensor<B, D>` (Burn) | 抽象层 = 运行时开销 |
| ❌ `_f32`/`_f16` 后缀 | 代码重复，应用泛型 |

**正确做法**：

```rust
// ✅ 纯泛型 - 编译时单态化 = 零成本
pub fn flash_attention<T: Float>(q: &[T], k: &[T], v: &[T], out: &mut [T], ...) -> Result<(), Error>;

// ✅ const 分支 - 编译器完全消除
match T::TYPE_ID {  // T::TYPE_ID 是 const
    FloatType::F32 => ...,  // 编译时只保留对应分支
}
```

### Fat Binary Only（ARCH-LOAD-001）

**所有后端都必须使用预编译中间态嵌入，绝对禁止运行时编译**：

| 后端 | 中间态格式 | 嵌入方式 | 运行时编译 |
|------|-----------|----------|-----------|
| CUDA | PTX | `include_bytes!` | ❌ 禁止 |
| ROCm | HSACO | `include_bytes!` | ❌ 禁止 |
| Metal | metallib | `include_bytes!` | ❌ 禁止 |
| WGPU | WGSL | `include_str!` | ❌ 禁止 |

**禁止的行为**：
- ❌ NVRTC 运行时编译 PTX
- ❌ hipcc 运行时编译 HSACO
- ❌ Metal 源码运行时编译
- ❌ 任何形式的运行时编译回退
- ❌ 环境变量配置（如 `GLLM_KERNEL_PATH`）

**WGSL 说明**：WGSL 是 WebGPU 的中间表示（IR），虽然 wgpu 会将其转换为原生格式，但这是"中间态到原生码的加载"（类似 PTX 到 GPU 机器码），不是"源码编译"。

### 零配置原则

- 用户不需要配置任何东西
- 自动检测硬件，自动选择后端
- 检测顺序：CUDA > ROCm > Metal > WGPU > CPU

### Driver API Only

- CUDA: 只依赖 `libcuda.so`（CUDA Driver API）
- ROCm: 只依赖 `libhsa-runtime64.so`（HSA Runtime）
- 无需安装完整 CUDA Toolkit 或 ROCm SDK

### 全后端实现铁律（🚨 FROZEN）

**任何算子/算法都必须提供所有支持后端的完整实现**：

| 后端 | 实现路径 | Loader 模块 | 必须实现 |
|------|----------|-------------|----------|
| CUDA | `src/cuda_kernels/kernels/*.ptx` | `src/cuda_kernels/ptx_loader.rs` | ✅ 必须 |
| ROCm | `src/hip_kernels/kernels/*.hsaco` | `src/hip_kernels/hsa_runtime.rs` | ✅ 必须 |
| Metal | `src/metal_kernels/kernels/*.metallib` | `src/metal_kernels/metallib_loader.rs` | ✅ 必须 |
| WGPU | `src/wgpu_kernels/shaders/*.wgsl` | `src/wgpu_kernels/` | ✅ 必须 |
| CPU | `src/cpu_kernels/` | 纯 Rust 实现 | ✅ 必须（参考实现） |

**算子实现检查清单**：
- [ ] CUDA PTX kernel 已实现并嵌入
- [ ] ROCm HSACO kernel 已实现并嵌入（通过 HSA Runtime 加载）
- [ ] Metal metallib 已实现并嵌入（通过 metal-rs 加载）
- [ ] WGPU WGSL shader 已实现并嵌入
- [ ] CPU 纯 Rust 参考实现已完成

**禁止的行为**：
- ❌ 只实现部分后端（如只有 CUDA 没有 Metal）
- ❌ 某后端使用 stub/TODO 占位
- ❌ 后端之间行为不一致
- ❌ 跳过 CPU 参考实现

**统一调度接口**（`src/ops/*.rs`）：
```
算子调用 → BackendSelector → 运行时检测 → 调用对应后端
                              ↓
          ┌─────────┬─────────┬─────────┬─────────┬────────┐
          │ CUDA    │ ROCm    │ Metal   │ WGPU    │ CPU    │
          │ PTX     │ HSACO   │ metallib│ WGSL    │ Rust   │
          └─────────┴─────────┴─────────┴─────────┴────────┘
```

## 目录结构

```
src/
├── cuda_kernels/      # CUDA PTX + Driver API
│   ├── ptx_loader.rs  # SM-aware PTX 加载（无 NVRTC）
│   └── kernels/*.ptx  # 预编译 PTX
├── hip_kernels/       # HSA Runtime + HSACO
│   ├── hsa_runtime.rs # HSA 动态加载
│   └── kernels/*.hsaco
├── metal_kernels/     # Metal Framework + metallib
│   ├── metallib_loader.rs
│   └── kernels/*.metallib
├── wgpu_kernels/      # wgpu + WGSL
│   └── shaders/*.wgsl
└── cpu_kernels/       # 纯 Rust 参考实现
```

## 编译 Kernels

```bash
# CUDA PTX（需要 CUDA Toolkit）
./scripts/compile_cuda_kernels.sh

# ROCm HSACO（需要 ROCm）
./scripts/compile_hip_kernels.sh

# Metal metallib（需要 Xcode）
./scripts/compile_metal_kernels.sh
```

## 常用命令

```bash
cargo check                    # 语法检查
cargo test                     # 运行测试
cargo test --test integration  # 集成测试
cargo bench                    # 性能基准
```

---

## 开发经验教训（🚨 常见陷阱）

### WGPU API 版本兼容性

**问题**：wgpu 版本更新导致 API 变化

| 错误 | 原因 | 修复 |
|------|------|------|
| `wgpu::Maintain::Wait` 未定义 | wgpu 0.19+ API 变更 | 改为 `wgpu::PollType::Wait` |
| `request_adapter` 返回 Option | wgpu 0.20+ 返回 Result | 使用 `.map_err()` 而非 `.ok_or()` |
| `DeviceDescriptor` 缺少字段 | wgpu 0.20+ 新增 `trace` 字段 | 添加 `trace: wgpu::Trace::Off` |

**预防**：升级 wgpu 版本时必须检查 CHANGELOG，特别关注 Breaking Changes。

### HSA Runtime Rust 生命周期

**问题**：ROCm HSA kernel 初始化时的 borrow checker 错误

```rust
// ❌ 错误：agent 被移动后又被借用
let agent = agents.into_iter().next().unwrap();
let module = HsaKernelModule::from_hsaco(&agent, ...); // 借用 agent
Ok(Self {
    agent,  // 移动 agent - 错误！
    module,
})

// ✅ 正确：先计算所有需要借用的值，再移动
let agent = agents.into_iter().next().unwrap();
let module = HsaKernelModule::from_hsaco(&agent, ...);
let queue = create_queue(&agent, ...);  // 所有借用在这里完成
// 现在安全移动
Ok(Self { agent, queue, module })
```

**规则**：结构体包含 `agent` 和从 `agent` 派生的字段时，所有派生操作必须在 `agent` 移动到结构体之前完成。

### kernel_name 生命周期

**问题**：`&str.leak()` 无效，`&'static str` 要求

```rust
// ❌ 错误：leak() 不能用于 &str
fn from_hsaco(kernel_name: &str) {
    let name = kernel_name.leak();  // 编译错误
}

// ✅ 正确：直接使用 &'static str
const KERNEL_NAME: &str = "flash_attention_f32";

fn from_hsaco(kernel_name: &'static str) {
    // 直接使用，无需 leak
}
```

**规则**：HSA kernel 名称必须是 `&'static str`（编译时常量），不能是运行时创建的字符串。

### Fat Binary 占位文件

**问题**：`include_bytes!` 引用的文件必须存在

**解决**：为尚未编译的 kernel 创建最小有效占位文件：

```bash
# PTX 占位（最小有效 PTX）
echo '.version 7.0
.target sm_80
.address_size 64' > kernel.ptx

# HSACO 占位（最小 ELF header）
echo -ne '\x7fELF...' > kernel.hsaco

# metallib 占位（Apple metallib magic）
echo -ne 'MTLB...' > kernel.metallib
```

### 类型一致性

**问题**：`usize` vs `u32` 混用

```rust
// ❌ 错误
let count: u32 = config.max_candidates;  // max_candidates 是 usize

// ✅ 正确
let count = config.max_candidates;  // 保持 usize
// 或显式转换
let count: u32 = config.max_candidates as u32;
```

**规则**：配置结构体的数值类型应该统一，避免隐式转换。

### 删除 Burn，统一到 kernel_dispatcher（ADR-001 🚨 铁律）

**问题**：Burn Tensor 效率低，ops/ 论文算法无法使用 GPU 加速

**🚨 重要**：ops/ 包含论文优化算法（EAGLE-3、Medusa、FlashAttention），**必须迁移而非删除**！

**决策**：迁移 ops/ 论文算法到 kernel_dispatcher，去除 Burn 依赖

```
迁移（保留论文优化）：
📦 ops/eagle3.rs (NeurIPS'25)     → kernel_dispatcher
📦 ops/medusa.rs (ICML'24)        → kernel_dispatcher
📦 ops/flash_attention.rs         → kernel_dispatcher（分层块+MaskCache）
📦 ops/softmax.rs                 → kernel_dispatcher（Log-space+Kahan）
📦 ops/paged_attention.rs         → kernel_dispatcher（多级层级+CoW）

保留（已是纯 Rust，无需迁移）：
✅ ops/engram*.rs, embedding.rs, stable_accumulator.rs
```

**迁移模式**：`Tensor<B, D>` → `&[T]` 原始切片（保留算法逻辑）

**唯一 API**：`KernelDispatcher`（原始切片 `&[T]` + GPU 加速 + 论文算法）

**零成本要求**：
- `#[inline(always)]` 强制内联
- 原始切片，无 Tensor 抽象
- enum + match 派发，无 vtable
- `T::TYPE_ID` const 分支消除

### 统一泛型算子 API（ARCH-API-001）

> 详见顶部「零成本抽象铁律」章节。

**核心要点**：
- 纯泛型 `<T: Float>`，编译时单态化
- `T::TYPE_ID` 是 const，分支被编译器消除
- 最终代码：`flash_attention::<f32>()` 直接调用 f32 kernel，零开销
