#!/bin/bash
# Metal Kernel Build Script (Single-File Library Architecture)
# Target: Universal (M1/M2/M3/M4)
# Output: One metallib containing ALL kernels
# Requires: macOS 14+ with Xcode Command Line Tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/kernels/metal"
SRC_DIR="${PROJECT_ROOT}/src/metal_kernels/kernels"

# All Metal kernels to include
KERNELS="eagle3 embedding_ops flash_attention paged_attention"

echo "=== Metal Fat Binary Build (Single-File Library) ==="
echo "Output: $OUTPUT_DIR"
echo "Target: Universal (M1/M2/M3/M4)"

# Check xcrun/metal
if ! command -v xcrun &> /dev/null; then
    echo "Error: xcrun not found. Please install Xcode Command Line Tools."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Compile all kernels to .air intermediate files
AIR_FILES=""
FOUND_COUNT=0

for kernel in $KERNELS; do
    SRC_FILE="$SRC_DIR/$kernel.metal"

    if [[ ! -f "$SRC_FILE" ]]; then
        echo "  Warning: $kernel.metal not found, skipping..."
        continue
    fi

    FOUND_COUNT=$((FOUND_COUNT + 1))
    echo "  Compiling $kernel.metal to AIR..."

    xcrun -sdk macosx metal \
        -c "$SRC_FILE" \
        -o "/tmp/${kernel}.air" \
        -O3

    AIR_FILES="$AIR_FILES /tmp/${kernel}.air"
done

if [[ $FOUND_COUNT -eq 0 ]]; then
    echo "Error: No Metal source files found!"
    exit 1
fi

echo ""
echo "Linking $FOUND_COUNT kernels to universal.metallib..."

# Link all .air files into single metallib
xcrun -sdk macosx metallib \
    $AIR_FILES \
    -o "$OUTPUT_DIR/universal.metallib"

# Cleanup intermediate files
rm -f /tmp/*.air

if [[ -f "$OUTPUT_DIR/universal.metallib" ]]; then
    SIZE=$(du -h "$OUTPUT_DIR/universal.metallib" | cut -f1)
    echo "  Created: universal.metallib ($SIZE)"
fi

echo ""
echo "=== Metal Build Complete ==="
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
