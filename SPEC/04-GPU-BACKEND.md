# SPEC/04 — GPU Backend: JIT 编译器统一路径

## §1 Overview

GPU 后端（CUDA/HIP/Metal）与 CPU 后端走**同一条 JIT 编译器路径**。Phase 0→1→2 完全复用，Phase 3 新增 GPU ISA 代码生成后端。

### §1.1 设计目标

1. **JIT 编译器统一路径** — Phase 0→1→2 完全复用（标量函数 → 符号执行 → OpTrace → SemanticDAG → FusionPlan），Phase 3 新增 `PtxCodeGen`（CUDA）、`AmdgpuCodeGen`（HIP）和 `AirCodeGen`（Metal），与 `X86CodeGen` / `DynasmAArch64CodeGen` 平级
2. **零外部依赖** — 仅依赖 GPU driver API（`libcuda.so` / `libamdhip64.so` / `Metal.framework`），运行时动态链接。不依赖 cudarc、cuBLAS、metal-rs、MPS 或任何第三方 GPU 库
3. **硬件能力驱动** — `GpuDeviceProfile` 检测 SM version / shared memory / register file / bandwidth（CUDA/HIP）或 GPU Family / threadgroup memory（Metal），驱动融合决策和代码生成
4. **Feature-gated** — `jit-cuda`、`jit-hip`、`jit-metal` Cargo features，编译时不需要 CUDA SDK / ROCm SDK / Xcode

### §1.2 Non-Goals

- Multi-GPU（tensor parallelism, pipeline parallelism）— 延后
- Vulkan/OpenCL — 不在范围内
- 反向传播 — 仅推理

---

## §2 架构总览

```
Phase 0-2 [完全复用，平台无关]
    标量函数 → 符号执行 → OpTrace → SemanticDAG → FusionPlan

Phase 3 [平台特定代码生成]
    ├── X86CodeGen (iced-x86)        → 机器码 → mmap+call
    ├── DynasmAArch64CodeGen         → 机器码 → mmap+call
    ├── PtxCodeGen (新增)            → PTX 文本 → cuModuleLoadData → cuLaunchKernel
    ├── AmdgpuCodeGen (新增)         → AMDGPU ISA → hipModuleLoadData → hipLaunchKernel
    └── AirCodeGen (新增)            → AIR bitcode → MTLLibrary → dispatch
```

所有代码生成后端实现同一个 `MachineCodeEmitter` trait。Phase 0-2 的 `FusionPlan` 是平台无关的，Phase 3 根据 `Platform` 选择对应的 CodeGen 后端。

---

## §3 Platform 枚举扩展

```rust
pub enum Platform {
    X86_64 { avx512: bool, amx: bool },   // Intel AMX: Sapphire Rapids+
    Aarch64 { sve: bool, amx: bool },      // Apple AMX: M1+ (未公开指令集)
    Cuda { sm_version: u32 },              // sm_70, sm_80, sm_89, sm_90
    Hip { gfx_arch: u32 },                 // gfx908, gfx90a, gfx942, gfx1100
    Metal { gpu_family: u32 },             // Apple GPU family (7=M1, 8=M2, 9=M3, 10=M4)
}
```

### §3.1 AMX 作为 CPU 平台的能力扩展

AMX 不是独立 Platform variant，而是 CPU 平台的微内核加速能力：

| AMX 变体 | 微内核规格 | 指令 | 对比基线 |
|----------|-----------|------|---------|
| Intel AMX (SPR+) | 16×16 tile | `TDPBF16PS` / `TDPFP16PS` / `TDPBSSD` | vs AVX-512 14×32 FMA |
| Apple AMX (M1+) | 32×32 block | `AMX_LDX` / `AMX_FMA64` (未公开) | vs NEON 8×12 FMA |

AMX 不改变 Phase 0-2 的任何逻辑。Phase 3 的 X86CodeGen / DynasmAArch64CodeGen 检测到 AMX 后，GEMM 微内核生成切换到 tile/block 指令，一条指令完成整个 MR×NR 块的乘累加。这是"更宽的微内核"，不是新的执行模型。

---

## §4 GpuDeviceProfile

与 CPU `DeviceProfile` 同构，通过 driver API 检测硬件能力。

### §4.1 CPU 侧 IsaLevel 扩展

与 GPU 无关，但属于同一次 Platform 扩展：

```rust
pub enum IsaLevel {
    Scalar,
    Avx2,
    Avx512,
    Avx512Amx,     // 新增: Sapphire Rapids+ (AVX-512 + AMX tile 指令)
    Neon,
    Sve,
    Sve2,
    NeonAmx,       // 新增: Apple M1+ (NEON + AMX 协处理器指令)
}
```

### §4.2 GPU 侧 GpuDeviceProfile

#### CUDA

| 属性 | CUDA Driver API | 用途 |
|------|----------------|------|
| SM version | `cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR/MINOR)` | 指令集选择（wmma/mma.sync/wgmma） |
| Shared memory/SM | `cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)` | Tile 尺寸决策（类比 CPU L1） |
| Register file/SM | `cuDeviceGetAttribute(MAX_REGISTERS_PER_MULTIPROCESSOR)` | Register blocking（类比 CPU SIMD regs） |
| SM count | `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)` | Grid 尺寸 |
| L2 cache | `cuDeviceGetAttribute(L2_CACHE_SIZE)` | Tiling 决策（类比 CPU L3） |
| Memory bandwidth | clock rate × bus width | Roofline 分析 |
| Warp size | `cuDeviceGetAttribute(WARP_SIZE)` | 线程组织（始终 32） |

#### HIP

HIP driver API 与 CUDA 几乎 1:1 映射（`hipDeviceGetAttribute`），属性名对应替换。

#### Metal

| 属性 | Metal API | 用途 |
|------|-----------|------|
| GPU Family | `MTLDevice.supportsFamily()` | 指令集能力 |
| Max threadgroup memory | `MTLDevice.maxThreadgroupMemoryLength` | Tile 尺寸决策 |
| Max threads per threadgroup | `MTLDevice.maxThreadsPerThreadgroup` | Block 尺寸 |
| Recommended working set size | `MTLDevice.recommendedMaxWorkingSetSize` | 内存预算 |

这些参数驱动 Phase 2 融合决策和 Phase 3 代码生成，与 CPU 路径的 `cache_sizes()` / `num_simd_regs()` / `gemm_blocking()` 完全同构。

---

## §5 Driver API FFI 绑定

零依赖，运行时 `dlopen`。

### §5.1 CUDA Driver API

```rust
struct CudaDriver {
    lib: *mut c_void,  // dlopen("libcuda.so.1")
    // 初始化
    cuInit: unsafe extern "C" fn(u32) -> CUresult,
    // 设备管理
    cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut i32, CUdevice_attribute, CUdevice) -> CUresult,
    // 上下文管理
    cuCtxCreate: unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult,
    cuCtxDestroy: unsafe extern "C" fn(CUcontext) -> CUresult,
    // 模块/内核
    cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cuLaunchKernel: unsafe extern "C" fn(
        CUfunction, u32, u32, u32, u32, u32, u32,
        u32, CUstream, *mut *mut c_void, *mut *mut c_void,
    ) -> CUresult,
    // 内存管理
    cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cuMemcpyHtoD: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    // 流管理
    cuStreamCreate: unsafe extern "C" fn(*mut CUstream, u32) -> CUresult,
    cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    cuStreamDestroy: unsafe extern "C" fn(CUstream) -> CUresult,
}
```

加载方式：

```rust
impl CudaDriver {
    pub fn load() -> Result<Self, DriverError> {
        let lib = unsafe { dlopen(b"libcuda.so.1\0".as_ptr() as _, RTLD_LAZY) };
        if lib.is_null() {
            return Err(DriverError::NotFound("libcuda.so.1"));
        }
        // dlsym 逐个解析函数指针...
        Ok(Self { lib, cuInit: ..., ... })
    }
}
```

### §5.2 HIP Driver API

```rust
struct HipDriver {
    lib: *mut c_void,  // dlopen("libamdhip64.so")
    hipInit: unsafe extern "C" fn(u32) -> hipError_t,
    hipModuleLoadData: unsafe extern "C" fn(*mut hipModule_t, *const c_void) -> hipError_t,
    hipModuleGetFunction: unsafe extern "C" fn(*mut hipFunction_t, hipModule_t, *const c_char) -> hipError_t,
    hipModuleLaunchKernel: unsafe extern "C" fn(
        hipFunction_t, u32, u32, u32, u32, u32, u32,
        u32, hipStream_t, *mut *mut c_void, *mut *mut c_void,
    ) -> hipError_t,
    hipMalloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> hipError_t,
    hipFree: unsafe extern "C" fn(*mut c_void) -> hipError_t,
    hipMemcpyHtoD: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> hipError_t,
    hipMemcpyDtoH: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> hipError_t,
    // ...
}
```

API 与 CUDA 几乎 1:1 映射，函数名 `cu` → `hip` 前缀替换。

### §5.3 Metal Framework 绑定

通过 Objective-C runtime 绑定（`dlopen("Metal.framework/Metal")`）：

```rust
struct MetalDriver {
    // Objective-C runtime
    objc_msgSend: unsafe extern "C" fn(*mut Object, Sel, ...) -> *mut Object,
    // Metal classes
    MTLCreateSystemDefaultDevice: unsafe extern "C" fn() -> *mut Object,
    // 通过 objc_msgSend 调用:
    // [device newCommandQueue]
    // [device newBufferWithLength:options:]
    // [device newLibraryWithData:error:]
    // [library newFunctionWithName:]
    // [device newComputePipelineStateWithFunction:error:]
    // [commandBuffer computeCommandEncoder]
    // [encoder setComputePipelineState:]
    // [encoder setBuffer:offset:atIndex:]
    // [encoder dispatchThreadgroups:threadsPerThreadgroup:]
    // [encoder endEncoding]
    // [commandBuffer commit]
    // [commandBuffer waitUntilCompleted]
}
```

macOS 上 Metal framework 始终可用，无需额外安装。

---

## §6 PtxCodeGen — Phase 3 PTX 代码生成

实现 `MachineCodeEmitter` trait，与 X86CodeGen / DynasmAArch64CodeGen 平级。

### §6.1 TraceOp → PTX 指令映射

| TraceOp | PTX 指令 | 备注 |
|---------|---------|------|
| Add | `add.f32` | |
| Mul | `mul.f32` | |
| Fma | `fma.rn.f32` | Round-to-nearest |
| Exp | `ex2.approx.f32` + 多项式修正 | `exp(x) = exp2(x * log2(e))` |
| Sqrt | `sqrt.rn.f32` | |
| Recip | `rcp.approx.f32` + Newton-Raphson | 1 次 Newton 迭代达到 fp32 精度 |
| Neg | `neg.f32` | |
| Max | `max.f32` | |
| Min | `min.f32` | |
| Load | `ld.global.f32` | Global memory load |
| Store | `st.global.f32` | Global memory store |

### §6.2 SM 版本感知指令选择

类比 CPU 的 AVX2 vs AVX-512 选择：

| SM | 关键能力 | GEMM 策略 |
|----|---------|----------|
| sm_70 (V100) | HMMA fp16 | `wmma.load` / `wmma.mma` / `wmma.store` PTX |
| sm_80 (A100) | async copy + mma.sync | `cp.async.cg.shared.global` + `mma.sync.aligned.m16n8k16` |
| sm_89 (L40/4090) | FP8 Tensor Core | FP8 `mma.sync` 变体 |
| sm_90 (H100) | WGMMA + TMA | `wgmma.mma_async` + TMA descriptor（硬件异步数据搬运） |

### §6.3 PTX 生成流程

```
FusionPlan
  → PtxCodeGen.emit_plan(plan, &gpu_profile)
  → PTX 文本 (String)
  → cuModuleLoadData(ptx.as_ptr())  // driver 编译 PTX → SASS
  → cuModuleGetFunction("kernel_name")
  → GpuCompiledLayer { module, kernel, block_dim, grid_calculator, ... }
```

PTX 是文本格式的虚拟 ISA，由 NVIDIA driver 在 `cuModuleLoadData` 时编译为目标 GPU 的 SASS 机器码。这意味着：
- 我们生成 PTX 文本，不需要 NVRTC 或 nvcc
- Driver 负责最终的指令调度和寄存器分配
- PTX 版本需匹配目标 SM（`.version 7.0` for sm_70, `.version 8.0` for sm_80, etc.）

---

## §7 Phase 2 融合决策 GPU 适配

Phase 2 融合规则完全复用，硬件约束参数替换为 GPU 等价物：

| CPU 概念 | GPU 等价 | 说明 |
|----------|---------|------|
| L1 cache → TileLevelFusion 阈值 | Shared memory → SharedMemoryFusion 阈值 | GPU shared memory 是显式管理的 L1 |
| L2 cache → ComputeRoot 容量 | L2 cache（GPU 通常 4-40MB） | GPU L2 是隐式缓存 |
| SIMD 寄存器数 → epilogue 可行性 | Per-thread 寄存器数 → epilogue 可行性 | GPU 寄存器文件远大于 CPU |
| BLIS MC/NC/KC blocking | GPU block_m/block_n/block_k/warp_m/warp_n | 分层 tiling 模型统一 |

融合决策的核心逻辑（EpilogueInjection、LoopFusion、TileLevelFusion）不变，只是参数来源从 `DeviceProfile` 切换到 `GpuDeviceProfile`。

---

## §8 GEMM Tiling — 统一的分层映射

所有平台的 GEMM 都是同一个分层 tiling 模型，只是每层映射到不同的硬件资源：

| 层次 | CPU (AVX2) | CPU (AVX-512) | CPU (Intel AMX) | CPU (Apple AMX) | GPU (CUDA) |
|------|-----------|--------------|----------------|----------------|-----------|
| 最外层 | NC×KC (L3) | NC×KC (L3) | NC×KC (L3) | NC×KC (L3) | Grid tile |
| 中间层 | MC×KC (L2) | MC×KC (L2) | MC×KC (L2) | MC×KC (L2) | Block tile (shared mem) |
| 微内核 | 6×16 FMA | 14×32 FMA | 16×16 TDPBF16PS | 32×32 AMX_FMA | Warp mma.sync |
| 数据搬运 | pack_a/pack_b | pack_a/pack_b | tile load/store | AMX_LDX/STX | Global→Shared (`cp.async`) |

Phase 2 的融合决策（EpilogueInjection、TileLevelFusion 等）对所有平台通用。差异只在 Phase 3 微内核指令选择。

---

## §9 CompiledLayer 执行层分叉

```rust
pub enum CompiledLayer {
    Cpu(CpuCompiledLayer),    // 现有: mmap + fn_ptr call
    Gpu(GpuCompiledLayer),    // 新增: module + kernel launch
}

pub struct GpuCompiledLayer {
    module: GpuModule,          // CUmodule / hipModule_t / MTLLibrary
    kernel: GpuFunction,        // CUfunction / hipFunction_t / MTLComputePipelineState
    block_dim: (u32, u32, u32), // threads per block/threadgroup
    grid_calculator: GridCalc,  // 根据输入尺寸计算 grid 维度
    shared_mem_bytes: u32,      // 动态 shared memory 大小
    scratchpad_bytes: usize,    // 额外设备内存需求
    config_hash: u64,           // 用于缓存查找
}

/// Grid 尺寸计算器
pub enum GridCalc {
    /// Elementwise: grid = ceil(n / block_size)
    Linear { elements_per_thread: u32 },
    /// GEMM: grid = (ceil(M/block_m), ceil(N/block_n))
    Tiled2D { block_m: u32, block_n: u32 },
    /// Reduction: grid = num_rows (每行一个 block)
    PerRow,
}
```

---

## §10 AmdgpuCodeGen — HIP 后端

与 PtxCodeGen 平级。生成 AMDGPU ISA 汇编，通过 `hipModuleLoadData` 加载。

### §10.1 GFX 架构感知

| GFX Arch | 代表 GPU | 关键能力 | GEMM 策略 |
|----------|---------|---------|----------|
| gfx908 | MI100 | MFMA fp32/fp16 | `v_mfma_f32_32x32x8f16` |
| gfx90a | MI210 | MFMA + unified memory | MFMA + 统一寻址 |
| gfx942 | MI300X | MFMA + HBM3 | MFMA + 更大 LDS |
| gfx1100 | RX 7900 | WMMA | `v_wmma_f32_16x16x16_f16` |

### §10.2 AMDGPU vs PTX 差异

| 维度 | CUDA PTX | AMDGPU ISA |
|------|---------|-----------|
| 虚拟 ISA | PTX（driver 编译为 SASS） | 直接生成 GCN/RDNA ISA |
| Warp 大小 | 32 (warp) | 64 (wavefront, CDNA) / 32 (wave32, RDNA) |
| Shared memory | `shared` 地址空间 | LDS (Local Data Share) |
| 同步 | `bar.sync` | `s_barrier` |
| Tensor Core | `mma.sync` / `wgmma` | `v_mfma_*` / `v_wmma_*` |

---

## §11 AirCodeGen — Metal 后端

与 PtxCodeGen 平级。生成 Apple IR (AIR) bitcode，通过 Metal framework 加载。

AIR 是 Metal 的中间表示（类似 PTX 之于 CUDA），是 LLVM bitcode 的 Apple 定制变体。JIT 路径：

```
AirCodeGen 生成 AIR bitcode
  → MTLDevice.makeLibrary(data:)    // 从 AIR bitcode 创建 MTLLibrary
  → MTLLibrary.makeFunction(name:)  // 获取 kernel 函数
  → MTLDevice.makeComputePipelineState(function:)  // 编译为 PSO
  → dispatch
```

### Apple GPU Family 感知

| GPU Family | 芯片 | 关键能力 |
|-----------|------|---------|
| Family 7 | M1 | 基础 SIMD-group 操作 |
| Family 8 | M2 | 增强 SIMD-group + 更大 threadgroup memory |
| Family 9 | M3 | 动态缓存 (Dynamic Caching) + mesh shading |
| Family 10 | M4 | 增强 AI 加速 |

### Metal 特有概念映射

| GPU 通用概念 | Metal 等价 |
|-------------|-----------|
| Thread block | Threadgroup |
| Shared memory | Threadgroup memory |
| Warp (32 threads) | SIMD-group (32 threads) |
| Grid | Dispatch grid |
| Register file | Per-thread registers |
| Global memory | Device memory (`device` address space) |

---

## §12 Cargo Features

```toml
[features]
default = []
jit-cuda = []     # 零外部依赖，运行时 dlopen libcuda.so.1
jit-hip = []      # 零外部依赖，运行时 dlopen libamdhip64.so
jit-metal = []    # 零外部依赖，运行时 dlopen Metal.framework (macOS only)
```

所有 GPU feature 均为零外部依赖。编译时不需要 CUDA SDK、ROCm SDK 或 Xcode。GPU 相关代码通过 `#[cfg(feature = "jit-cuda")]` 等条件编译。

运行时通过 `dlopen` 动态加载 driver 库。如果 driver 不存在，返回 `DriverError::NotFound` 而非 panic。

---

## §13 实现优先级

| 阶段 | 内容 | 依赖 |
|------|------|------|
| P0 | Driver API FFI 绑定（CUDA/HIP/Metal）+ GpuDeviceProfile 检测 | 无 |
| P1 | PtxCodeGen: Elementwise LoopFusion → 单 kernel | P0 |
| P2 | PtxCodeGen: GEMM tiled kernel (sm_80 mma.sync) | P1 |
| P3 | EpilogueInjection + SharedMemoryFusion | P2 |
| P4 | SM 版本特化 (sm_70 wmma / sm_90 wgmma+TMA) | P2 |
| P5 | AmdgpuCodeGen: 同 P1-P4 for HIP | P0 |
| P6 | AirCodeGen: 同 P1-P3 for Metal | P0 |

---

## §14 测试策略

### §14.1 CPU Golden Reference

GPU JIT 输出 vs CPU JIT 输出，逐元素比较：

```rust
// 伪代码
let cpu_output = cpu_jit_execute(fusion_plan, &cpu_profile, &input);
let gpu_output = gpu_jit_execute(fusion_plan, &gpu_profile, &input);
assert_approx_eq(&cpu_output, &gpu_output, tolerance);
```

### §14.2 Tolerance 阈值

| 精度 | Elementwise | Reduction | GEMM |
|------|------------|-----------|------|
| f32 | 1e-6 | 1e-5 | 1e-4（累积误差） |
| f16 | 1e-3 | 1e-2 | 1e-2 |

### §14.3 性能对标

自研 JIT PTX vs 理论峰值（Roofline model）。目标：
- Elementwise: 达到 memory bandwidth 上限的 90%+
- GEMM: 达到 Tensor Core 峰值的 70%+（初始目标）

### §14.4 集成测试

与 `test_local_pipeline` 模型测试集成，端到端验证 GPU 推理路径。

---

## §15 与其他 SPEC 的一致性

| SPEC | 约束 | 本文档对齐 |
|------|------|-----------|
| SPEC/01 (ARCH-CPU-FIRST) | CPU 优先，GPU 扩展 | Platform enum 扩展不破坏 CPU 路径 |
| SPEC/02 (ARCH-JIT-FIRST) | 禁止 cudarc/cuBLAS/MPS | 零外部依赖，仅 driver API |
| SPEC/02 (ARCH-PEAK-PERF) | 硬件能力驱动 | GpuDeviceProfile 检测 + Roofline |
| SPEC/03 (MachineCodeEmitter) | 统一 trait 接口 | PtxCodeGen/AmdgpuCodeGen/AirCodeGen 实现同一 trait |
