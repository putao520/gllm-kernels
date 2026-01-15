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

# Default GFX architectures (2019-2026 full coverage)
# ═══════════════════════════════════════════════════════════════════
# RDNA 1 (2019)
#   gfx1010: RX 5600 XT, RX 5700, RX 5700 XT (Navi 10)
#   gfx1011: RX 5500, RX 5500 XT (Navi 14)
#   gfx1012: RX 5300 (Navi 14 variant)
# RDNA 2 (2020-2021)
#   gfx1030: RX 6800, RX 6800 XT, RX 6900 XT, RX 6950 XT (Navi 21)
#   gfx1031: RX 6700 XT, RX 6750 XT (Navi 22)
#   gfx1032: RX 6600, RX 6600 XT, RX 6650 XT (Navi 23)
#   gfx1034: RX 6500 XT (Navi 24)
#   gfx1035: Integrated (Rembrandt APU)
# RDNA 3 (2022-2024)
#   gfx1100: RX 7900 XT, RX 7900 XTX (Navi 31)
#   gfx1101: RX 7800 XT, RX 7700 XT (Navi 32)
#   gfx1102: RX 7600, RX 7600 XT (Navi 33)
#   gfx1103: Integrated (Phoenix APU)
# RDNA 3.5 (2024)
#   gfx1150: Strix Point APU
#   gfx1151: Strix Halo APU
# RDNA 4 (2025-2026)
#   gfx1200: RX 9070, RX 9070 XT (Navi 48)
#   gfx1201: Future RDNA 4 variants
# ───────────────────────────────────────────────────────────────────
# CDNA 1 (2020)
#   gfx908: MI100 (Arcturus)
# CDNA 2 (2021-2022)
#   gfx90a: MI210, MI250, MI250X (Aldebaran)
# CDNA 3 (2023-2024)
#   gfx940: MI300A (APU variant)
#   gfx941: MI300 variant
#   gfx942: MI300X, MI325X (Aqua Vanjaram)
# CDNA 4 (2025-2026)
#   MI350X, MI355X (expected gfx950 series)
# ═══════════════════════════════════════════════════════════════════

# Consumer GPUs (RDNA)
GFX_RDNA="gfx1010 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102"

# Data Center GPUs (CDNA)
GFX_CDNA="gfx908 gfx90a gfx942"

# Combined default (most common architectures)
GFX_ARCHS="$GFX_RDNA $GFX_CDNA"

# Extended architectures (newer/experimental)
GFX_RDNA4="gfx1200 gfx1201"
INCLUDE_RDNA4=false

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
        --include-rdna4)
            INCLUDE_RDNA4=true
            shift
            ;;
        --rdna-only)
            GFX_ARCHS="$GFX_RDNA"
            shift
            ;;
        --cdna-only)
            GFX_ARCHS="$GFX_CDNA"
            shift
            ;;
        --help)
            echo "Usage: $0 [--gfx ARCH] [--output DIR] [--include-rdna4] [--rdna-only] [--cdna-only]"
            echo "  --gfx ARCH       Space-separated GFX architectures (overrides default)"
            echo "  --output DIR     Output directory for HSACO files"
            echo "  --include-rdna4  Include RDNA4 architectures (requires ROCm 7.0+)"
            echo "  --rdna-only      Only compile for consumer GPUs (RDNA)"
            echo "  --cdna-only      Only compile for data center GPUs (CDNA)"
            echo ""
            echo "Supported architectures (2019-2026 coverage):"
            echo ""
            echo "  RDNA 1 (2019):"
            echo "    gfx1010 - RX 5600/5700 series"
            echo "  RDNA 2 (2020-2021):"
            echo "    gfx1030 - RX 6800/6900 series"
            echo "    gfx1031 - RX 6700 series"
            echo "    gfx1032 - RX 6600 series"
            echo "  RDNA 3 (2022-2024):"
            echo "    gfx1100 - RX 7900 series"
            echo "    gfx1101 - RX 7700/7800 series"
            echo "    gfx1102 - RX 7600 series"
            echo "  RDNA 4 (2025-2026, requires ROCm 7.0+):"
            echo "    gfx1200 - RX 9070 series"
            echo "    gfx1201 - Future RDNA4"
            echo ""
            echo "  CDNA 1 (2020):"
            echo "    gfx908  - MI100"
            echo "  CDNA 2 (2021-2022):"
            echo "    gfx90a  - MI200 series"
            echo "  CDNA 3 (2023-2024):"
            echo "    gfx942  - MI300 series"
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

# Check ROCm version for RDNA4 support
ROCM_VERSION=$($HIPCC --version 2>&1 | grep -oP 'HIP version: \K[0-9]+\.[0-9]+' | head -1)
if [ -z "$ROCM_VERSION" ]; then
    # Try alternative version detection
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d- -f1 || echo "0.0")
fi
ROCM_MAJOR=$(echo "$ROCM_VERSION" | cut -d. -f1)
ROCM_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f2)

echo "ROCm/HIP version: $ROCM_VERSION"

# Add RDNA4 architectures if requested and ROCm >= 7.0
if [ "$INCLUDE_RDNA4" = true ]; then
    if [ "$ROCM_MAJOR" -ge 7 ]; then
        echo "Adding RDNA4 architectures (gfx1200, gfx1201)..."
        GFX_ARCHS="$GFX_ARCHS $GFX_RDNA4"
    else
        echo "Warning: RDNA4 requires ROCm 7.0+, current version is $ROCM_VERSION"
        echo "         Skipping gfx1200/gfx1201 architectures"
    fi
fi

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
