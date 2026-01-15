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

# Default SM architectures (2019-2026 full coverage)
# ═══════════════════════════════════════════════════════════════════
# Turing (2018-2019)
#   SM 75: GTX 1650/1660, RTX 2060/2070/2080, Tesla T4
# Ampere (2020-2021)
#   SM 80: A100, A30 (GA100)
#   SM 86: RTX 3060/3070/3080/3090, A10, A40, RTX A4000/A5000/A6000
#   SM 87: Jetson Orin (embedded, optional)
# Ada Lovelace (2022-2023)
#   SM 89: RTX 4060/4070/4080/4090, L4, L40, RTX 4000/5000/6000 Ada
# Hopper (2022-2023)
#   SM 90: H100, H200
# Blackwell (2024-2026) - requires CUDA 12.8+
#   SM 100: B100, B200, GB200
#   SM 101: Blackwell Thor, DIGITS
#   SM 120: RTX 5070/5080/5090 (Blackwell consumer)
# ═══════════════════════════════════════════════════════════════════
SM_ARCHS="75 80 86 89 90"

# Blackwell architectures (requires CUDA 12.8+)
SM_ARCHS_BLACKWELL="100 120"
INCLUDE_BLACKWELL=false

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
        --include-blackwell)
            INCLUDE_BLACKWELL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--sm ARCH] [--output DIR] [--include-blackwell]"
            echo "  --sm ARCH          Space-separated SM architectures"
            echo "  --output DIR       Output directory for PTX files"
            echo "  --include-blackwell  Include Blackwell architectures (requires CUDA 12.8+)"
            echo ""
            echo "Default architectures (2019-2026 coverage):"
            echo "  Turing (2018-2019):"
            echo "    SM 75 - GTX 1650/1660, RTX 2060/2070/2080, Tesla T4"
            echo "  Ampere (2020-2021):"
            echo "    SM 80 - A100, A30"
            echo "    SM 86 - RTX 3060/3070/3080/3090, A10, A40"
            echo "  Ada Lovelace (2022-2023):"
            echo "    SM 89 - RTX 4060/4070/4080/4090, L4, L40"
            echo "  Hopper (2022-2023):"
            echo "    SM 90 - H100, H200"
            echo "  Blackwell (2024-2026, requires CUDA 12.8+):"
            echo "    SM 100 - B100, B200, GB200"
            echo "    SM 120 - RTX 5070/5080/5090"
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

# Check CUDA version for Blackwell support
CUDA_VERSION=$($NVCC --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

echo "CUDA version: $CUDA_VERSION"

# Add Blackwell architectures if requested and CUDA >= 12.8
if [ "$INCLUDE_BLACKWELL" = true ]; then
    if [ "$CUDA_MAJOR" -gt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
        echo "Adding Blackwell architectures (SM 100, 120)..."
        SM_ARCHS="$SM_ARCHS $SM_ARCHS_BLACKWELL"
    else
        echo "Warning: Blackwell requires CUDA 12.8+, current version is $CUDA_VERSION"
        echo "         Skipping SM 100/120 architectures"
    fi
fi

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
