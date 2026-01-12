#!/bin/bash
# Compile HIP kernels to HSACO for AMD GPUs
# Usage: ./compile_hip_kernels.sh [--gfx ARCH] [--output DIR]
#
# Requirements:
#   - ROCm with hipcc
#   - Environment variable ROCM_PATH or hipcc in PATH

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KERNEL_DIR="$PROJECT_ROOT/src/hip_kernels/kernels"
OUTPUT_DIR="$KERNEL_DIR"

# Default GFX architectures (RDNA2, RDNA3, CDNA2, CDNA3)
GFX_ARCHS="gfx1030 gfx1100 gfx90a gfx942"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gfx)
            GFX_ARCHS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--gfx ARCH] [--output DIR]"
            echo "  --gfx ARCH    Space-separated GFX architectures (default: gfx1030 gfx1100 gfx90a gfx942)"
            echo "  --output DIR  Output directory for HSACO files"
            echo ""
            echo "Common architectures:"
            echo "  gfx1030 - RX 6800/6900 (RDNA2)"
            echo "  gfx1100 - RX 7900 (RDNA3)"
            echo "  gfx90a  - MI200 (CDNA2)"
            echo "  gfx942  - MI300 (CDNA3)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find hipcc
if [ -n "$ROCM_PATH" ]; then
    HIPCC="$ROCM_PATH/bin/hipcc"
elif [ -d "/opt/rocm" ]; then
    HIPCC="/opt/rocm/bin/hipcc"
elif command -v hipcc &> /dev/null; then
    HIPCC="hipcc"
else
    echo "Error: hipcc not found. Please install ROCm or set ROCM_PATH."
    exit 1
fi

echo "Using hipcc: $HIPCC"
$HIPCC --version 2>&1 | head -n1

# Compile options
COMMON_FLAGS=(
    "-O3"
    "-ffast-math"
    "-fgpu-rdc"
)

# Compile each kernel
compile_kernel() {
    local src="$1"
    local name=$(basename "$src" .hip)

    for gfx in $GFX_ARCHS; do
        local hsaco_file="$OUTPUT_DIR/${name}_${gfx}.hsaco"
        echo "Compiling $name for $gfx..."

        $HIPCC "${COMMON_FLAGS[@]}" \
            --offload-arch=$gfx \
            --genco \
            -o "$hsaco_file" \
            "$src"

        echo "  -> $hsaco_file ($(stat -c%s "$hsaco_file" 2>/dev/null || stat -f%z "$hsaco_file") bytes)"
    done

    # Also compile default HSACO
    local default_hsaco="$OUTPUT_DIR/${name}.hsaco"
    echo "Compiling $name default HSACO (gfx90a)..."
    $HIPCC "${COMMON_FLAGS[@]}" \
        --offload-arch=gfx90a \
        --genco \
        -o "$default_hsaco" \
        "$src"
}

# Main
echo "=== Compiling HIP Kernels ==="
echo "Kernel directory: $KERNEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Target architectures: $GFX_ARCHS"
echo ""

for hip_file in "$KERNEL_DIR"/*.hip; do
    if [ -f "$hip_file" ]; then
        compile_kernel "$hip_file"
    fi
done

echo ""
echo "=== Compilation Complete ==="
echo "HSACO files generated in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.hsaco 2>/dev/null || echo "No HSACO files found"
