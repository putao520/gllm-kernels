#!/bin/bash
# ROCm/HIP Kernel Build Script (Single-File Library Architecture)
# Target: Latest 3 architectures
# Output: One hsaco per architecture containing ALL kernels
# Strategy: Compile each kernel individually, then link with RDC
# Requires: ROCm SDK 6.0+ in CI runner

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/kernels/rocm"
SRC_DIR="${PROJECT_ROOT}/src/hip_kernels/kernels"
TMP_DIR="${PROJECT_ROOT}/build-tmp/rocm"

# Target architectures (latest 3 supported generations in ROCm 6.1)
# gfx90a: MI200 series (Instinct)
# gfx942: MI300 series (Instinct)
# gfx1100: RDNA3 (Navi 31)
ROCM_ARCHS="gfx90a gfx942 gfx1100"

# All ROCm kernels to include in each hsaco
KERNELS="chunked_prefill eagle3 embedding_ops evic_press flash_attention flash_tree_attn int2_quantizer medusa paged_attention prompt_cache spec_ee"

echo "=== ROCm Fat Binary Build (Single-File Library) ==="
echo "Output: $OUTPUT_DIR"
echo "Architectures: gfx90a (MI200), gfx942 (MI300), gfx1100 (RDNA3)"

# Check hipcc
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Please install ROCm SDK."
    exit 1
fi

hipcc --version 2>&1 | head -1 || true

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

# Collect all source files
SRC_FILES=()
FOUND_COUNT=0
for kernel in $KERNELS; do
    SRC_FILE="$SRC_DIR/$kernel.hip"
    if [[ -f "$SRC_FILE" ]]; then
        SRC_FILES+=("$SRC_FILE")
        FOUND_COUNT=$((FOUND_COUNT + 1))
        echo "  Found: $kernel.hip"
    else
        echo "  Warning: $kernel.hip not found, skipping..."
    fi
done

if [[ $FOUND_COUNT -eq 0 ]]; then
    echo "Error: No HIP source files found!"
    exit 1
fi

echo ""
echo "Building with $FOUND_COUNT kernels..."

# Build hsaco for each architecture
# Strategy: Compile each .hip to .hsaco individually (ROCm doesn't have fatbinary tool)
for arch in $ROCM_ARCHS; do
    echo ""
    echo "Building $arch hsaco files ($FOUND_COUNT kernels)..."

    HSACO_COUNT=0
    for src in "${SRC_FILES[@]}"; do
        kernel_name=$(basename "$src" .hip)
        hsaco_file="$OUTPUT_DIR/${kernel_name}_${arch}.hsaco"

        hipcc \
            --offload-arch=$arch \
            -O3 \
            -ffast-math \
            --genco \
            -o "$hsaco_file" \
            "$src"

        if [[ -f "$hsaco_file" ]]; then
            HSACO_COUNT=$((HSACO_COUNT + 1))
        fi
    done

    echo "  Created $HSACO_COUNT hsaco files for $arch"
done

# Cleanup temp files
rm -rf "$TMP_DIR"

echo ""
echo "=== ROCm Build Complete ==="
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
