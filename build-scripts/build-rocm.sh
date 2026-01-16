#!/bin/bash
# ROCm/HIP Kernel Build Script (Single-File Library Architecture)
# Target: Latest 3 architectures
# Output: One hsaco per architecture containing ALL kernels
# Requires: ROCm SDK 6.0+ in CI runner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/kernels/rocm"
SRC_DIR="${PROJECT_ROOT}/src/hip_kernels/kernels"

# Target architectures (latest 3 generations)
ROCM_ARCHS="gfx90a gfx1100 gfx1201"

# All ROCm kernels to include in each hsaco
KERNELS="chunked_prefill eagle3 embedding_ops evic_press flash_attention flash_tree_attn int2_quantizer medusa paged_attention prompt_cache spec_ee"

echo "=== ROCm Fat Binary Build (Single-File Library) ==="
echo "Output: $OUTPUT_DIR"
echo "Architectures: gfx90a (MI200), gfx1100 (RDNA3), gfx1201 (RDNA4)"

# Check hipcc
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Please install ROCm SDK."
    exit 1
fi

hipcc --version | head -1

mkdir -p "$OUTPUT_DIR"

# Collect all source files
SRC_FILES=""
FOUND_COUNT=0
for kernel in $KERNELS; do
    SRC_FILE="$SRC_DIR/$kernel.hip"
    if [[ -f "$SRC_FILE" ]]; then
        SRC_FILES="$SRC_FILES $SRC_FILE"
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

# Build hsaco for each architecture (containing all kernels)
for arch in $ROCM_ARCHS; do
    echo ""
    echo "Building $arch.hsaco (all $FOUND_COUNT kernels)..."
    hipcc \
        --offload-arch=$arch \
        -O3 \
        -ffast-math \
        --genco \
        -o "$OUTPUT_DIR/$arch.hsaco" \
        $SRC_FILES

    if [[ -f "$OUTPUT_DIR/$arch.hsaco" ]]; then
        SIZE=$(du -h "$OUTPUT_DIR/$arch.hsaco" | cut -f1)
        echo "  Created: $arch.hsaco ($SIZE)"
    fi
done

echo ""
echo "=== ROCm Build Complete ==="
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
