#!/bin/bash
# SPIR-V Kernel Build Script (Single-File Library Architecture)
# Target: Universal (WebGPU/Vulkan)
# Output: One spv containing ALL kernels (or merged WGSL fallback)
# Requires: naga-cli (cargo install naga-cli)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/kernels/spirv"
SRC_DIR="${PROJECT_ROOT}/src/wgpu_kernels/kernels"

# All SPIR-V kernels to include
KERNELS="chunked_prefill eagle3 embedding_ops evic_press flash_attention flash_tree_attn int2_quantizer medusa paged_attention prompt_cache spec_ee"

echo "=== SPIR-V Fat Binary Build (Single-File Library) ==="
echo "Output: $OUTPUT_DIR"
echo "Target: Universal (WebGPU/Vulkan)"

mkdir -p "$OUTPUT_DIR"

# Check for compiler
if command -v naga &> /dev/null; then
    COMPILER="naga"
elif command -v glslc &> /dev/null; then
    COMPILER="glslc"
else
    echo "Warning: Neither naga nor glslc found."
    echo "Will create merged WGSL file as fallback."
    COMPILER="none"
fi

echo "Using compiler: $COMPILER"

# Merge all WGSL files into single file
MERGED_WGSL="/tmp/merged_kernels.wgsl"
echo "// Auto-generated merged WGSL kernels" > "$MERGED_WGSL"
echo "// Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MERGED_WGSL"

FOUND_COUNT=0
for kernel in $KERNELS; do
    SRC_FILE="$SRC_DIR/$kernel.wgsl"

    if [[ ! -f "$SRC_FILE" ]]; then
        echo "  Warning: $kernel.wgsl not found, skipping..."
        continue
    fi

    FOUND_COUNT=$((FOUND_COUNT + 1))
    echo "  Adding: $kernel.wgsl"

    echo "" >> "$MERGED_WGSL"
    echo "// ============================================================" >> "$MERGED_WGSL"
    echo "// Kernel: $kernel" >> "$MERGED_WGSL"
    echo "// ============================================================" >> "$MERGED_WGSL"
    cat "$SRC_FILE" >> "$MERGED_WGSL"
done

if [[ $FOUND_COUNT -eq 0 ]]; then
    echo "Error: No WGSL source files found!"
    rm -f "$MERGED_WGSL"
    exit 1
fi

echo ""
echo "Merged $FOUND_COUNT kernels."

# Compile or copy based on available compiler
if [[ "$COMPILER" == "naga" ]]; then
    echo "Compiling merged WGSL to universal.spv..."
    if naga "$MERGED_WGSL" "$OUTPUT_DIR/universal.spv" 2>/dev/null; then
        SIZE=$(du -h "$OUTPUT_DIR/universal.spv" | cut -f1)
        echo "  Created: universal.spv ($SIZE)"
    else
        echo "  Note: naga compilation failed (may need WGSL fixes)"
        echo "  Copying WGSL as fallback..."
        cp "$MERGED_WGSL" "$OUTPUT_DIR/universal.wgsl"
    fi
elif [[ "$COMPILER" == "glslc" ]]; then
    echo "Note: glslc doesn't support WGSL directly"
    echo "Copying merged WGSL as fallback..."
    cp "$MERGED_WGSL" "$OUTPUT_DIR/universal.wgsl"
else
    echo "Copying merged WGSL as fallback..."
    cp "$MERGED_WGSL" "$OUTPUT_DIR/universal.wgsl"
fi

rm -f "$MERGED_WGSL"

echo ""
echo "=== SPIR-V Build Complete ==="
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
