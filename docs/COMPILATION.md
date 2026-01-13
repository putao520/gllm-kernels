# gllm-kernels 多平台编译指南

## 概述

gllm-kernels 需要为不同的 GPU 架构预编译 kernels。本指南说明如何设置编译环境并编译所有平台。

## 编译环境要求

### 1. CUDA Kernels (NVIDIA GPUs)
- **操作系统**: Linux
- **工具**: NVIDIA CUDA Toolkit
- **安装**: https://developer.nvidia.com/cuda-downloads

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12

# 设置环境变量
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
```

### 2. ROCm Kernels (AMD GPUs)
- **操作系统**: Linux
- **工具**: AMD ROCm
- **安装**: https://rocm.docs.amd.com/

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    rocm-dev \
    hip-dev \
    rocblas-dev \
    miopen-dev

# 设置环境变量
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### 3. Metal Kernels (Apple Silicon)
- **操作系统**: macOS (11.0+)
- **工具**: Xcode Command Line Tools
- **安装**: 系统自带，或 `xcode-select --install`

```bash
# 验证安装
xcrun --sdk macosx --show-sdk-path
```

## 编译脚本使用

### 方式 1: 直接编译（需要本地编译器）

```bash
cd /path/to/gllm-kernels
./scripts/compile-kernels.sh
```

### 方式 2: Docker 编译（推荐）

使用 Docker 容器编译，避免污染主机环境。

#### CUDA 编译容器

```bash
docker build -t gllm-kernels-cuda -f docker/Dockerfile.cuda .
docker run --rm -v ~/.gsc:/output gllm-kernels-cuda
```

#### ROCm 编译容器

```bash
docker build -t gllm-kernels-rocm -f docker/Dockerfile.rocm .
docker run --rm -v ~/.gsc:/output gllm-kernels-rocm
```

#### Metal 编译容器

```bash
# 在 macOS 上直接编译，不需要 Docker
cd /path/to/gllm-kernels
./scripts/compile-kernels.sh
```

## 预编译的 GPU 架构

### CUDA (Compute Capability)

| 架构 | GPU 型号 | 用途 |
|------|---------|------|
| sm_75 | GTX 1660, RTX 2060/2070/2080 | Turing |
| sm_80 | A100 | Datacenter |
| sm_86 | RTX 3080/3080 Ti/3090 | High-end consumer |
| sm_87 | RTX A2000/A4000, Jetson Orin | Professional |
| sm_89 | RTX 4080/4090 | Ada Lovelace |
| sm_90 | H100 | Hopper |

### ROCm (AMD GPU Architecture)

| 架构 | GPU 型号 | 用途 |
|------|---------|------|
| gfx1030 | RX 6800/6800 XT/6900 XT | RDNA2 |
| gfx1031 | RX 6700 XT | RDNA2 |
| gfx1100 | RX 7900 XTX/XT | RDNA3 |
| gfx1101 | RX 7800 XT | RDNA3 |
| gfx1102 | RX 7700 XT | RDNA3 |

### Metal (Apple Silicon)

| 架构 | 芯片 | 用途 |
|------|------|------|
| apple-m1 | M1, M1 Pro, M1 Max, M1 Ultra | First Gen |
| apple-m2 | M2, M2 Pro, M2 Max, M2 Ultra | Second Gen |
| apple-m3 | M3, M3 Pro, M3 Max | Third Gen |

## 编译产物位置

所有编译的 kernels 存储在：

```
~/.gsc/gllm/kernels/
├── cuda/
│   └── flash_attention/
│       ├── sm_75.ptx
│       ├── sm_86.ptx
│       └── sm_89.ptx
├── rocm/
│   └── flash_attention/
│       ├── gfx1030.hsaco
│       └── gfx1100.hsaco
└── metal/
    └── flash_attention/
        ├── apple-m1.metallib
        └── apple-m2.metallib
```

## CI/CD 自动编译

在 GitHub Actions 中自动编译：

```yaml
# .github/workflows/compile-kernels.yml
name: Compile Kernels

on:
  push:
    paths:
      - 'src/*/kernels/**'
  workflow_dispatch:

jobs:
  compile-cuda:
    runs-on: [ubuntu-latest]
    steps:
      - uses: actions/checkout@v4
      - name: Install CUDA
        run: |
          wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
          sudo dpkg -i cuda-keyring_1.0-1_all.deb
          sudo apt-get update
          sudo apt-get -y install cuda-toolkit-12
      - name: Compile kernels
        run: ./scripts/compile-kernels.sh
      - name: Upload cache
        uses: actions/upload-artifact@v4
        with:
          name: kernels-cuda
          path: ~/.gsc/gllm/kernels/
```

## 故障排除

### 问题 1: nvcc 未找到

```bash
# 检查 CUDA 安装
which nvcc

# 如果未找到，添加到 PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### 问题 2: hipcc 未找到

```bash
# 检查 ROCm 安装
which hipcc

# 如果未找到，添加到 PATH
export PATH=/opt/rocm/bin:$PATH
```

### 问题 3: Metal 编译失败 (macOS)

```bash
# 安装 Xcode Command Line Tools
xcode-select --install

# 接受许可证
sudo xcodebuild -license accept
```

## 性能优化

### 编译选项

- `-O3`: 最高优化级别
- `--use_fast_math`: 快速数学运算（牺牲精度）
- `-ffast-math`: 同上（ROCm/Metal）

### 架构特定优化

为特定 GPU 架构编译可以显著提升性能：

```bash
# CUDA: 为 RTX 4090 优化
nvcc -ptx -arch=sm_89 -O3 ...

# ROCm: 为 RX 7900 XTX 优化
hipcc --amdgpu-target=gfx1100 -O3 ...

# Metal: 为 M3 Max 优化
xcrun -sdk macosx metal -O3 ...
```

## 下一步

1. 在开发机上安装编译工具
2. 运行 `./scripts/compile-kernels.sh`
3. 将 `~/.gsc/gllm/` 目录提交到版本控制或分发
4. 或设置 CI/CD 自动编译并发布

## 相关资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [HIP Programming Guide](https://rocm.docs.amd.com/)
- [Metal Shading Language Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification/)
