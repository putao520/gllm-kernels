#!/bin/bash
# Compile CUDA kernels to PTX
# Usage: ./compile_cuda_kernels.sh [--sm ARCH] [--output DIR]
#
# Requirements:
#   - CUDA Toolkit with nvcc
#   - Environment variable CUDA_HOME or nvcc in PATH

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KERNEL_DIR="$PROJECT_ROOT/src/cuda_kernels/kernels"
OUTPUT_DIR="$KERNEL_DIR"

# Default SM architectures (Ampere, Ada, Hopper)
SM_ARCHS="80 86 89 90"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sm)
            SM_ARCHS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--sm ARCH] [--output DIR]"
            echo "  --sm ARCH     Space-separated SM architectures (default: 80 86 89 90)"
            echo "  --output DIR  Output directory for PTX files"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find nvcc
if [ -n "$CUDA_HOME" ]; then
    NVCC="$CUDA_HOME/bin/nvcc"
elif command -v nvcc &> /dev/null; then
    NVCC="nvcc"
else
    echo "Error: nvcc not found. Please install CUDA Toolkit or set CUDA_HOME."
    exit 1
fi

echo "Using nvcc: $NVCC"
$NVCC --version | head -n1

# Compile options
COMMON_FLAGS=(
    "-O3"
    "--use_fast_math"
    "-Xptxas=-v"
    "-lineinfo"
)

# Compile each kernel
compile_kernel() {
    local src="$1"
    local name=$(basename "$src" .cu)

    for sm in $SM_ARCHS; do
        local ptx_file="$OUTPUT_DIR/${name}_sm${sm}.ptx"
        echo "Compiling $name for SM $sm..."

        $NVCC "${COMMON_FLAGS[@]}" \
            -arch=sm_$sm \
            --ptx \
            -o "$ptx_file" \
            "$src"

        echo "  -> $ptx_file ($(stat -c%s "$ptx_file" 2>/dev/null || stat -f%z "$ptx_file") bytes)"
    done

    # Also compile default PTX (for runtime compilation fallback)
    local default_ptx="$OUTPUT_DIR/${name}.ptx"
    echo "Compiling $name default PTX..."
    $NVCC "${COMMON_FLAGS[@]}" \
        -arch=sm_80 \
        --ptx \
        -o "$default_ptx" \
        "$src"
}

# Main
echo "=== Compiling CUDA Kernels ==="
echo "Kernel directory: $KERNEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Target architectures: $SM_ARCHS"
echo ""

for cu_file in "$KERNEL_DIR"/*.cu; do
    if [ -f "$cu_file" ]; then
        compile_kernel "$cu_file"
    fi
done

echo ""
echo "=== Compilation Complete ==="
echo "PTX files generated in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.ptx 2>/dev/null || echo "No PTX files found"
