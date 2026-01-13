#!/bin/bash
# gllm-kernels 编译脚本 - 为不同 GPU 架构预编译 kernels
# 版本: 2.0
# 支持: 全版本 PTX/HSACO/metallib 编译

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KERNELS_DIR="$PROJECT_ROOT/src"
CACHE_DIR="$HOME/.gsc/gllm/kernels"

# 创建缓存目录
mkdir -p "$CACHE_DIR"

log_info "GLLM Kernel 编译工具 v2.0"
log_info "缓存目录: $CACHE_DIR"
log_info "编译模式: 全版本中间态 (PTX/HSACO/metallib)"
echo ""

# ========================================
# CUDA Kernels 编译
# ========================================
compile_cuda_kernels() {
    if ! command -v nvcc &> /dev/null; then
        log_warn "nvcc 未找到，跳过 CUDA kernels 编译"
        return
    fi

    log_step "编译 CUDA kernels (PTX 中间态)"

    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_info "CUDA 版本: $CUDA_VERSION"
    log_info "PTX 是中间表示，可在无 GPU 环境编译"
    echo ""

    # 完整的 GPU 架构列表
    # https://developer.nvidia.com/cuda-gpus
    declare -A ARCHS=(
        # Volta (Compute Capability 7.0)
        ["sm_70"]="V100, Tesla V100"

        # Turing (Compute Capability 7.5)
        ["sm_75"]="GTX 1650/1660, RTX 2060/2070/2080, Tegra"

        # Ampere (Compute Capability 8.0)
        ["sm_80"]="A100 (GA100)"
        ["sm_86"]="RTX 3080/3080 Ti/3090 (GA102)"
        ["sm_87"]="RTX A2000/A4000/A5000, Jetson Orin"

        # Ada Lovelace (Compute Capability 8.9)
        ["sm_89"]="RTX 4080/4080 Ti, RTX 4090, L4, L40"

        # Hopper (Compute Capability 9.0)
        ["sm_90"]="H100, H200"
    )

    for kernel in "$KERNELS_DIR"/cuda_kernels/kernels/*.cu; do
        [ -f "$kernel" ] || continue
        kernel_name=$(basename "$kernel" .cu)
        log_info "编译 $kernel_name"
        echo ""

        for arch in "${!ARCHS[@]}"; do
            gpu_desc="${ARCHS[$arch]}"
            log_info "  → $arch: $gpu_desc"

            output_dir="$CACHE_DIR/cuda/$kernel_name"
            mkdir -p "$output_dir"
            output_file="$output_dir/${arch}.ptx"

            # 编译为 PTX (中间汇编)
            # -ptx: 生成 PTX 而不是 cubin
            # -arch: 指定虚拟架构
            # -O3: 最高优化
            nvcc -ptx \
                -arch=$arch \
                -O3 \
                --std=c++17 \
                --use_fast_math \
                -DFN_WITH_HALF \
                -o "$output_file" \
                "$kernel" 2>&1 | while IFS= read -r line; do
                    if echo "$line" | grep -qi "warning"; then
                        echo -e "${YELLOW}      $line${NC}"
                    elif echo "$line" | grep -qi "error"; then
                        echo -e "${RED}      $line${NC}"
                    else
                        echo "      $line"
                    fi
                done || {
                    log_error "编译失败: $arch"
                    continue
                }

            if [ -f "$output_file" ]; then
                size=$(du -h "$output_file" | cut -f1)
                lines=$(wc -l < "$output_file")
                log_info "    ✓ 生成: $output_file ($size, $lines lines)"
            fi
            echo ""
        done
    done

    # 生成版本信息
    cat > "$CACHE_DIR/cuda/VERSION.txt" << EOF
CUDA Kernels (PTX Intermediate Representation)
==============================================

CUDA Version: $CUDA_VERSION
PTX Version: Compatible with CUDA 11.0+
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Git Commit: $(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

Compiled Architectures:
EOF
    for arch in "${!ARCHS[@]}"; do
        echo "  - $arch: ${ARCHS[$arch]}" >> "$CACHE_DIR/cuda/VERSION.txt"
    done

    log_info "CUDA kernels 编译完成"
    echo ""
}

# ========================================
# ROCm Kernels 编译
# ========================================
compile_rocm_kernels() {
    if ! command -v hipcc &> /dev/null; then
        log_warn "hipcc 未找到，跳过 ROCm kernels 编译"
        return
    fi

    log_step "编译 ROCm kernels (HSACO 中间态)"

    ROCm_VERSION=$(hipcc --version 2>/dev/null | grep "HIP version" | head -1 || hipcc --version | head -1)
    log_info "ROCm 版本: $ROCM_VERSION"
    log_info "HSACO 是中间表示，可在无 GPU 环境编译"
    echo ""

    # 完整的 AMD GPU 架构列表
    # https://github.com/RadeonOpenCompute/ROCm/blob/rocm-5.7/docs/ISA-support.md
    declare -A ARCHS=(
        # RDNA2
        ["gfx1030"]="RX 6800 / RX 6800 XT / RX 6900 XT"
        ["gfx1031"]="RX 6700 / RX 6700 XT"
        ["gfx1032"]="RX 6600 / RX 6600 XT"
        ["gfx1034"]="RX 6650 XT"
        ["gfx1035"]="RX 6750 XT"
        ["gfx1036"]="Pro W6600"

        # RDNA3
        ["gfx1100"]="RX 7900 XTX / RX 7900 XT"
        ["gfx1101"]="RX 7800 XT"
        ["gfx1102"]="RX 7700 XT / RX 7700 XT"
        ["gfx1103"]="RX 7600"

        # CDNA2 (Datacenter)
        ["gfx90a"]="MI200 (Aldebaran)"

        # CDNA3 (Datacenter)
        ["gfx940"]="MI300 (CDNA 3)"
        ["gfx941"]="MI300A (CDNA 3)"
        ["gfx942"]="MI325X"
    )

    for kernel in "$KERNELS_DIR"/hip_kernels/kernels/*.hip; do
        [ -f "$kernel" ] || continue
        kernel_name=$(basename "$kernel" .hip)
        log_info "编译 $kernel_name"
        echo ""

        for arch in "${!ARCHS[@]}"; do
            gpu_desc="${ARCHS[$arch]}"
            log_info "  → $arch: $gpu_desc"

            output_dir="$CACHE_DIR/rocm/$kernel_name"
            mkdir -p "$output_dir"
            output_file="$output_dir/${arch}.hsaco"

            # 编译为 HSACO (中间表示)
            # --amdgpu-target: 指定 GPU 架构
            # -O3: 最高优化
            hipcc \
                -O3 \
                --amdgpu-target=$arch \
                -std=c++17 \
                -ffast-math \
                -o "$output_file" \
                "$kernel" 2>&1 | while IFS= read -r line; do
                    if echo "$line" | grep -qi "warning"; then
                        echo -e "${YELLOW}      $line${NC}"
                    elif echo "$line" | grep -qi "error"; then
                        echo -e "${RED}      $line${NC}"
                    else
                        echo "      $line"
                    fi
                done || {
                    log_error "编译失败: $arch"
                    continue
                }

            if [ -f "$output_file" ]; then
                size=$(du -h "$output_file" | cut -f1)
                log_info "    ✓ 生成: $output_file ($size)"
            fi
            echo ""
        done
    done

    # 生成版本信息
    cat > "$CACHE_DIR/rocm/VERSION.txt" << EOF
ROCm Kernels (HSACO Intermediate Representation)
================================================

ROCm Version: $ROCM_VERSION
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Git Commit: $(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

Compiled Architectures:
EOF
    for arch in "${!ARCHS[@]}"; do
        echo "  - $arch: ${ARCHS[$arch]}" >> "$CACHE_DIR/rocm/VERSION.txt"
    done

    log_info "ROCm kernels 编译完成"
    echo ""
}

# ========================================
# Metal Kernels 编译
# ========================================
compile_metal_kernels() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_warn "Metal 只能在 macOS 上编译，跳过"
        return
    fi

    if ! command -v xcrun &> /dev/null; then
        log_warn "xcrun 未找到，跳过 Metal kernels 编译"
        log_warn "请安装 Xcode Command Line Tools: xcode-select --install"
        return
    fi

    log_step "编译 Metal kernels (metallib 中间态)"

    macOS_VERSION=$(sw_vers -productVersion)
    SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)
    log_info "macOS 版本: $macOS_VERSION"
    log_info "SDK 版本: $SDK_VERSION"
    log_info "metallib 是中间表示，可在无特定硬件编译"
    echo ""

    # 完整的 Apple 芯片列表
    declare -A ARCHS=(
        # M1 系列
        ["apple-m1"]="M1 (5nm CPU/GPU)"
        ["apple-m1-pro"]="M1 Pro"
        ["apple-m1-max"]="M1 Max"
        ["apple-m1-ultra"]="M1 Ultra"

        # M2 系列
        ["apple-m2"]="M2 (4nm CPU/GPU)"
        ["apple-m2-pro"]="M2 Pro"
        ["apple-m2-max"]="M2 Max"
        ["apple-m2-ultra"]="M2 Ultra"

        # M3 系列
        ["apple-m3"]="M3 (3nm CPU/GPU)"
        ["apple-m3-pro"]="M3 Pro"
        ["apple-m3-max"]="M3 Max"
    )

    for kernel in "$KERNELS_DIR"/metal_kernels/kernels/*.metal; do
        [ -f "$kernel" ] || continue
        kernel_name=$(basename "$kernel" .metal)
        log_info "编译 $kernel_name"
        echo ""

        for arch in "${!ARCHS[@]}"; do
            gpu_desc="${ARCHS[$arch]}"
            log_info "  → $arch: $gpu_desc"

            output_dir="$CACHE_DIR/metal/$kernel_name"
            mkdir -p "$output_dir"
            output_file="$output_dir/${arch}.metallib"
            air_file="${output_file}.air"

            # 编译为 AIR (Apple Intermediate Representation)
            # 然后转换为 metallib
            xcrun -sdk macosx metal \
                -O3 \
                -mmacosx-version-min=11.0 \
                -o "$air_file" \
                "$kernel" 2>&1 | while IFS= read -r line; do
                    if echo "$line" | grep -qi "warning"; then
                        echo -e "${YELLOW}      $line${NC}"
                    elif echo "$line" | grep -qi "error"; then
                        echo -e "${RED}      $line${NC}"
                    else
                        echo "      $line"
                    fi
                done || {
                    log_error "编译失败: $arch"
                    continue
                }

            # 将 AIR 转换为 metallib
            if [ -f "$air_file" ]; then
                xcrun -sdk macosx metallib \
                    -o "$output_file" \
                    "$air_file" 2>&1 | while IFS= read -r line; do
                        echo "      $line"
                    done

                # 清理 AIR 文件
                rm -f "$air_file"
            fi

            if [ -f "$output_file" ]; then
                size=$(du -h "$output_file" | cut -f1)
                log_info "    ✓ 生成: $output_file ($size)"
            fi
            echo ""
        done
    done

    # 生成版本信息
    cat > "$CACHE_DIR/metal/VERSION.txt" << EOF
Metal Kernels (metallib Intermediate Representation)
====================================================

macOS Version: $macOS_VERSION
SDK Version: $SDK_VERSION
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Git Commit: $(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

Compiled Architectures:
EOF
    for arch in "${!ARCHS[@]}"; do
        echo "  - $arch: ${ARCHS[$arch]}" >> "$CACHE_DIR/metal/VERSION.txt"
    done

    log_info "Metal kernels 编译完成"
    echo ""
}

# ========================================
# 统计和报告
# ========================================
generate_report() {
    log_step "生成编译报告"
    echo ""

    log_info "编译完成！统计信息："
    echo ""

    if [ -d "$CACHE_DIR" ]; then
        total_files=$(find "$CACHE_DIR" -type f \( -name "*.ptx" -o -name "*.hsaco" -o -name "*.metallib" \) | wc -l)
        total_size=$(du -sh "$CACHE_DIR" | cut -f1)

        log_info "总文件数: $total_files"
        log_info "总大小: $total_size"
        echo ""

        # 按后端分组统计
        for backend in cuda rocm metal; do
            backend_dir="$CACHE_DIR/$backend"
            if [ -d "$backend_dir" ]; then
                backend_files=$(find "$backend_dir" -type f \( -name "*.ptx" -o -name "*.hsaco" -o -name "*.metallib" \) | wc -l)
                backend_size=$(du -sh "$backend_dir" | cut -f1)

                echo "$(tr '[:lower:]' '[:upper:]' <<< ${backend}) 后端:"
                echo "  文件数: $backend_files"
                echo "  大小: $backend_size"
                echo ""
            fi
        done

        # 列出所有编译的 kernels
        log_info "已编译的 kernels："
        echo ""
        find "$CACHE_DIR" -type f \( -name "*.ptx" -o -name "*.hsaco" -o -name "*.metallib" \) | sort | while read -r file; do
            size=$(du -h "$file" | cut -f1)
            rel_path="${file#$CACHE_DIR/}"
            backend=$(echo "$rel_path" | cut -d'/' -f1)
            kernel=$(echo "$rel_path" | cut -d'/' -f2)
            arch=$(basename "$rel_path")
            ext="${arch##*.}"

            printf "  %-20s %-20s %-30s %8s\n" "$backend" "$kernel" "$arch" "($size)"
        done
    fi

    echo ""
    log_info "所有 kernels 已缓存到: $CACHE_DIR"
}

# ========================================
# 主流程
# ========================================
main() {
    log_info "开始编译 kernels..."
    echo ""

    # 编译 CUDA kernels
    compile_cuda_kernels
    echo ""

    # 编译 ROCm kernels
    compile_rocm_kernels
    echo ""

    # 编译 Metal kernels
    compile_metal_kernels
    echo ""

    # 生成报告
    generate_report
}

# 执行主流程
main "$@"
