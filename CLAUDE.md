# gllm-kernels

Low-level GPU attention kernels with runtime backend selection.

## SPEC 位置

- `./SPEC/`

## 核心架构约束（🚨 FROZEN - 铁律）

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
