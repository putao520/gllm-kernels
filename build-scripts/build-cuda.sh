#!/bin/bash
# CUDA Kernel Build Script (Single-File Library Architecture)
# Target: Latest 3 architectures + PTX fallback
# Output: One fatbin per architecture containing ALL kernels
# Requires: CUDA SDK 12.0+ in CI runner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/kernels/cuda"
SRC_DIR="${PROJECT_ROOT}/src/cuda_kernels/kernels"

# Target architectures (latest 3 generations)
CUDA_ARCHS="86 89 90"

# All CUDA kernels to include in each fatbin
KERNELS="embedding_ops flash_attention fused_qkv_attention online_softmax paged_attention selective_scan tiled_attention"

echo "=== CUDA Fat Binary Build (Single-File Library) ==="
echo "Output: $OUTPUT_DIR"
echo "Architectures: sm_86 (Ampere), sm_89 (Ada), sm_90 (Hopper), PTX fallback"

# Check nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA SDK."
    exit 1
fi

nvcc --version | head -1

mkdir -p "$OUTPUT_DIR"

# Collect all source files
SRC_FILES=""
FOUND_COUNT=0
for kernel in $KERNELS; do
    SRC_FILE="$SRC_DIR/$kernel.cu"
    if [[ -f "$SRC_FILE" ]]; then
        SRC_FILES="$SRC_FILES $SRC_FILE"
        FOUND_COUNT=$((FOUND_COUNT + 1))
        echo "  Found: $kernel.cu"
    else
        echo "  Warning: $kernel.cu not found, skipping..."
    fi
done

if [[ $FOUND_COUNT -eq 0 ]]; then
    echo "Error: No CUDA source files found!"
    exit 1
fi

echo ""
echo "Building with $FOUND_COUNT kernels..."

# Build fatbin for each architecture (containing all kernels)
for arch in $CUDA_ARCHS; do
    echo ""
    echo "Building sm_$arch.fatbin (all $FOUND_COUNT kernels)..."
    nvcc -fatbin \
        -arch=sm_$arch \
        -O3 \
        --use_fast_math \
        -o "$OUTPUT_DIR/sm_$arch.fatbin" \
        $SRC_FILES

    if [[ -f "$OUTPUT_DIR/sm_$arch.fatbin" ]]; then
        SIZE=$(du -h "$OUTPUT_DIR/sm_$arch.fatbin" | cut -f1)
        echo "  Created: sm_$arch.fatbin ($SIZE)"
    fi
done

# Build PTX fallback (for older/unknown GPUs)
echo ""
echo "Building fallback.ptx (all $FOUND_COUNT kernels)..."
nvcc -ptx \
    -arch=sm_86 \
    -O3 \
    --use_fast_math \
    -o "$OUTPUT_DIR/fallback.ptx" \
    $SRC_FILES

if [[ -f "$OUTPUT_DIR/fallback.ptx" ]]; then
    SIZE=$(du -h "$OUTPUT_DIR/fallback.ptx" | cut -f1)
    echo "  Created: fallback.ptx ($SIZE)"
fi

echo ""
echo "=== CUDA Build Complete ==="
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
