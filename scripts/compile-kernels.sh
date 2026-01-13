#!/bin/bash
# gllm-kernels 编译脚本 - 为不同 GPU 架构预编译 kernels

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KERNELS_DIR="$PROJECT_ROOT/src"
CACHE_DIR="$HOME/.gsc/gllm/kernels"

# 创建缓存目录
mkdir -p "$CACHE_DIR"

log_info "GLLM Kernel 编译工具"
log_info "缓存目录: $CACHE_DIR"
echo ""

# ========================================
# CUDA Kernels 编译
# ========================================
compile_cuda_kernels() {
    if ! command -v nvcc &> /dev/null; then
        log_warn "nvcc 未找到，跳过 CUDA kernels 编译"
        return
    fi

    log_info "编译 CUDA kernels..."

    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_info "CUDA 版本: $CUDA_VERSION"

    # 定义目标 GPU 架构
    # https://developer.nvidia.com/cuda-gpus
    declare -A ARCHS=(
        ["sm_75"]="GTX 1660, RTX 2060/2070/2080 (Turing)"
        ["sm_80"]="A100 (Ampere)"
        ["sm_86"]="RTX 3080/3080 Ti/3090 (Ampere)"
        ["sm_87"]="Jetson Orin, RTX A2000/A4000"
        ["sm_89"]="RTX 4080/4090 (Ada Lovelace)"
        ["sm_90"]="H100 (Hopper)"
    )

    for kernel in "$KERNELS_DIR"/cuda_kernels/kernels/*.cu; do
        [ -f "$kernel" ] || continue
        kernel_name=$(basename "$kernel" .cu)
        log_info "  编译 $kernel_name"

        for arch in "${!ARCHS[@]}"; do
            log_info "    $arch - ${ARCHS[$arch]}"

            output_dir="$CACHE_DIR/cuda/$kernel_name"
            mkdir -p "$output_dir"
            output_file="$output_dir/${arch}.ptx"

            # 编译 PTX (虚拟汇编，不特定于实际 GPU)
            nvcc -ptx \
                -arch=$arch \
                -O3 \
                --std=c++17 \
                --use_fast_math \
                -DFN_WITH_HALF \
                -o "$output_file" \
                "$kernel" 2>&1 | while IFS= read -r line; do
                    if echo "$line" | grep -q "warning"; then
                        echo -e "${YELLOW}      $line${NC}"
                    elif echo "$line" | grep -q "error"; then
                        echo -e "${RED}      $line${NC}"
                    else
                        echo "      $line"
                    fi
                done

            if [ -f "$output_file" ]; then
                size=$(du -h "$output_file" | cut -f1)
                log_info "    ✓ 生成: $output_file ($size)"
            fi
        done
    done

    log_info "CUDA kernels 编译完成"
}

# ========================================
# ROCm Kernels 编译
# ========================================
compile_rocm_kernels() {
    if ! command -v hipcc &> /dev/null; then
        log_warn "hipcc 未找到，跳过 ROCm kernels 编译"
        return
    fi

    log_info "编译 ROCm kernels..."

    ROCM_VERSION=$(hipcc --version | grep "HIP version" | sed 's/.*HIP version: //')
    log_info "ROCm 版本: $ROCM_VERSION"

    # 定义目标 AMD GPU 架构
    # https://github.com/RadeonOpenCompute/ROCm/blob/rocm-5.0/docs/ISA-support.md
    declare -A ARCHS=(
        ["gfx1030"]="RX 6800/6800 XT/6900 XT (RDNA2)"
        ["gfx1031"]="RX 6700 XT"
        ["gfx1100"]="RX 7900 XTX/XT (RDNA3)"
        ["gfx1101"]="RX 7800 XT"
        ["gfx1102"]="RX 7700 XT"
    )

    for kernel in "$KERNELS_DIR"/hip_kernels/kernels/*.hip; do
        [ -f "$kernel" ] || continue
        kernel_name=$(basename "$kernel" .hip)
        log_info "  编译 $kernel_name"

        for arch in "${!ARCHS[@]}"; do
            log_info "    $arch - ${ARCHS[$arch]}"

            output_dir="$CACHE_DIR/rocm/$kernel_name"
            mkdir -p "$output_dir"
            output_file="$output_dir/${arch}.hsaco"

            # 编译 HSACO (HIP 的汇编格式)
            hipcc \
                -O3 \
                --amdgpu-target=$arch \
                -std=c++17 \
                -ffast-math \
                -o "$output_file" \
                "$kernel" 2>&1 | while IFS= read -r line; do
                    if echo "$line" | grep -q "warning"; then
                        echo -e "${YELLOW}      $line${NC}"
                    elif echo "$line" | grep -q "error"; then
                        echo -e "${RED}      $line${NC}"
                    else
                        echo "      $line"
                    fi
                done

            if [ -f "$output_file" ]; then
                size=$(du -h "$output_file" | cut -f1)
                log_info "    ✓ 生成: $output_file ($size)"
            fi
        done
    done

    log_info "ROCm kernels 编译完成"
}

# ========================================
# Metal Kernels 编译
# ========================================
compile_metal_kernels() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_warn "Metal 只能在 macOS 上编译，跳过"
        return
    fi

    if ! command -v metallib &> /dev/null && ! command -v xcrun &> /dev/null; then
        log_warn "Metal 编译工具未找到，跳过 Metal kernels 编译"
        return
    fi

    log_info "编译 Metal kernels..."

    # 定义目标 Apple 芯片
    declare -A ARCHS=(
        ["apple-m1"]="M1 / M1 Pro / M1 Max"
        ["apple-m2"]="M2 / M2 Pro / M2 Max / M2 Ultra"
        ["apple-m3"]="M3 / M3 Pro / M3 Max"
    )

    for kernel in "$KERNELS_DIR"/metal_kernels/kernels/*.metal; do
        [ -f "$kernel" ] || continue
        kernel_name=$(basename "$kernel" .metal)
        log_info "  编译 $kernel_name"

        for arch in "${!ARCHS[@]}"; do
            log_info "    $arch - ${ARCHS[$arch]}"

            output_dir="$CACHE_DIR/metal/$kernel_name"
            mkdir -p "$output_dir"
            output_file="$output_dir/${arch}.metallib"

            # 使用 xcrun 编译 Metal shader
            # -O: 优化级别
            # -mmacosx-version-min: 最低 macOS 版本
            xcrun -sdk macosx metal \
                -O3 \
                -mmacosx-version-min=11.0 \
                -o "${output_file}.air" \
                "$kernel" 2>&1 | while IFS= read -r line; do
                    if echo "$line" | grep -q "warning"; then
                        echo -e "${YELLOW}      $line${NC}"
                    elif echo "$line" | grep -q "error"; then
                        echo -e "${RED}      $line${NC}"
                    else
                        echo "      $line"
                    fi
                done

            # 将 AIR 转换为 metallib
            if [ -f "${output_file}.air" ]; then
                xcrun -sdk macosx metallib \
                    -o "$output_file" \
                    "${output_file}.air" 2>&1 | while IFS= read -r line; do
                        echo "      $line"
                    done

                rm "${output_file}.air"
            fi

            if [ -f "$output_file" ]; then
                size=$(du -h "$output_file" | cut -f1)
                log_info "    ✓ 生成: $output_file ($size)"
            fi
        done
    done

    log_info "Metal kernels 编译完成"
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

    # 统计编译结果
    log_info "编译完成！统计信息："
    echo ""

    if [ -d "$CACHE_DIR" ]; then
        total_files=$(find "$CACHE_DIR" -type f | wc -l)
        total_size=$(du -sh "$CACHE_DIR" | cut -f1)

        log_info "总文件数: $total_files"
        log_info "总大小: $total_size"
        echo ""

        # 列出所有编译的 kernels
        log_info "已编译的 kernels："
        find "$CACHE_DIR" -type f -name "*.bin" -o -name "*.ptx" -o -name "*.hsaco" -o -name "*.metallib" | sort | while read -r file; do
            size=$(du -h "$file" | cut -f1)
            rel_path="${file#$CACHE_DIR/}"
            echo "  $rel_path ($size)"
        done
    fi

    echo ""
    log_info "所有 kernels 已缓存到: $CACHE_DIR"
}

# 执行主流程
main "$@"
