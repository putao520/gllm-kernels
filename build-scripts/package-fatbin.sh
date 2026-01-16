#!/bin/bash
# Package all pre-compiled kernels into Fat Binary archive
# Single-File Library Architecture: One file per platform/architecture
# Called by CI after all platform builds complete

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KERNELS_DIR="${PROJECT_ROOT}/kernels"
VERSION="${1:-dev}"

echo "=== Packaging Fat Binary (Single-File Library) ==="
echo "Version: $VERSION"

# Verify platform files exist
echo ""
echo "=== Platform Files ==="

CUDA_FILES=0
ROCM_FILES=0
METAL_FILES=0
SPIRV_FILES=0

# Enable nullglob to handle empty glob patterns
shopt -s nullglob

# Check CUDA
for file in "$KERNELS_DIR"/cuda/*.fatbin "$KERNELS_DIR"/cuda/*.ptx; do
    if [[ -f "$file" ]]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  CUDA: $(basename "$file") ($SIZE)"
        CUDA_FILES=$((CUDA_FILES + 1))
    fi
done

# Check ROCm
for file in "$KERNELS_DIR"/rocm/*.hsaco; do
    if [[ -f "$file" ]]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  ROCm: $(basename "$file") ($SIZE)"
        ROCM_FILES=$((ROCM_FILES + 1))
    fi
done

# Check Metal
for file in "$KERNELS_DIR"/metal/*.metallib; do
    if [[ -f "$file" ]]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  Metal: $(basename "$file") ($SIZE)"
        METAL_FILES=$((METAL_FILES + 1))
    fi
done

# Check SPIR-V
for file in "$KERNELS_DIR"/spirv/*.spv "$KERNELS_DIR"/spirv/*.wgsl; do
    if [[ -f "$file" ]]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  SPIR-V: $(basename "$file") ($SIZE)"
        SPIRV_FILES=$((SPIRV_FILES + 1))
    fi
done

# Restore default glob behavior
shopt -u nullglob

TOTAL_FILES=$((CUDA_FILES + ROCM_FILES + METAL_FILES + SPIRV_FILES))
echo ""
echo "Total files: $TOTAL_FILES"

# Create manifest
MANIFEST="$KERNELS_DIR/MANIFEST.json"
cat > "$MANIFEST" << EOF
{
  "version": "$VERSION",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "architecture": "single-file-library",
  "description": "Each file contains ALL kernels for that platform/architecture",
  "platforms": {
    "cuda": {
      "files": ["sm_86.fatbin", "sm_89.fatbin", "sm_90.fatbin", "fallback.ptx"],
      "architectures": {
        "sm_86": "Ampere (RTX 30xx Ti/Super, A10/A40)",
        "sm_89": "Ada Lovelace (RTX 40xx, L4/L40)",
        "sm_90": "Hopper (H100, H200)",
        "ptx": "Forward-compatible fallback"
      },
      "kernels_per_file": 7,
      "kernels": ["embedding_ops", "flash_attention", "fused_qkv_attention", "online_softmax", "paged_attention", "selective_scan", "tiled_attention"]
    },
    "rocm": {
      "file_pattern": "{kernel}_{arch}.hsaco",
      "architectures": {
        "gfx90a": "MI200 series (CDNA2)",
        "gfx942": "MI300 series (CDNA3)",
        "gfx1100": "RDNA3 (RX 7000 series)"
      },
      "kernels": ["chunked_prefill", "eagle3", "embedding_ops", "evic_press", "flash_attention", "flash_tree_attn", "int2_quantizer", "medusa", "paged_attention", "prompt_cache", "spec_ee"],
      "note": "ROCm produces individual hsaco per kernel per arch (no fatbinary tool)"
    },
    "metal": {
      "files": ["universal.metallib"],
      "architectures": {
        "universal": "Apple Silicon (M1/M2/M3/M4)"
      },
      "kernels_per_file": 4,
      "kernels": ["eagle3", "embedding_ops", "flash_attention", "paged_attention"]
    },
    "spirv": {
      "files": ["universal.spv"],
      "architectures": {
        "universal": "WebGPU/Vulkan fallback"
      },
      "kernels_per_file": 11,
      "kernels": ["chunked_prefill", "eagle3", "embedding_ops", "evic_press", "flash_attention", "flash_tree_attn", "int2_quantizer", "medusa", "paged_attention", "prompt_cache", "spec_ee"]
    }
  }
}
EOF

echo ""
echo "Created: $MANIFEST"

# Calculate sizes
echo ""
echo "=== Size Report ==="
for dir in cuda rocm metal spirv; do
    if [[ -d "$KERNELS_DIR/$dir" ]]; then
        SIZE=$(du -sh "$KERNELS_DIR/$dir" 2>/dev/null | cut -f1)
        echo "  $dir: $SIZE"
    fi
done
echo ""
TOTAL_SIZE=$(du -sh "$KERNELS_DIR" | cut -f1)
echo "  Total: $TOTAL_SIZE"

# Create archive
ARCHIVE_NAME="gllm-kernels-fatbin-${VERSION}.tar.gz"
echo ""
echo "Creating archive: $ARCHIVE_NAME"
tar -czvf "$PROJECT_ROOT/$ARCHIVE_NAME" -C "$PROJECT_ROOT" kernels/

echo ""
echo "=== Package Complete ==="
ls -lh "$PROJECT_ROOT/$ARCHIVE_NAME"
