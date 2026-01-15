#!/bin/bash
# Compile Metal shaders to metallib (precompiled binary)
#
# Usage: ./scripts/compile_metal_kernels.sh [--std VERSION] [--output DIR]
#
# Prerequisites:
# - macOS with Xcode Command Line Tools (xcrun metal, xcrun metallib)
#
# Output:
# - src/metal_kernels/kernels/flash_attention.metallib
# - src/metal_kernels/kernels/paged_attention.metallib

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KERNEL_DIR="$PROJECT_ROOT/src/metal_kernels/kernels"
OUTPUT_DIR="$KERNEL_DIR"

# Metal Language Standards (2019-2026 Apple Silicon coverage)
# ═══════════════════════════════════════════════════════════════════
# macos-metal2.2: macOS 10.15 Catalina (2019)
#   - A13 Bionic (iPhone 11), Intel Macs with AMD GPUs
# macos-metal2.3: macOS 11 Big Sur (2020)
#   - A14 Bionic, M1 (Apple7 GPU family)
# macos-metal2.4: macOS 13 Ventura (2022)
#   - A15/A16, M1 Pro/Max/Ultra, M2 (Apple8 GPU family)
# macos-metal3.0: macOS 14 Sonoma (2023)
#   - A17 Pro, M3 (Apple9 GPU family), mesh shaders
# macos-metal3.1: macOS 15 Sequoia (2024)
#   - A18, M4 (Apple9 GPU family), enhanced ray tracing
# metal3.2/4.0: macOS 16+ (2025-2026)
#   - M5 (Apple10 GPU family), Neural Engine in GPU cores
# ═══════════════════════════════════════════════════════════════════

# Default: metal2.4 for broad compatibility (covers all M-series)
METAL_STD="macos-metal2.4"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --std)
            METAL_STD="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--std VERSION] [--output DIR]"
            echo "  --std VERSION   Metal language standard (default: macos-metal2.4)"
            echo "  --output DIR    Output directory for metallib files"
            echo ""
            echo "Supported Metal standards (2019-2026 Apple Silicon):"
            echo ""
            echo "  macos-metal2.2 (macOS 10.15, 2019):"
            echo "    - A13 Bionic (iPhone 11)"
            echo "    - Intel Macs with AMD Radeon"
            echo ""
            echo "  macos-metal2.3 (macOS 11, 2020):"
            echo "    - A14 Bionic, M1"
            echo "    - Apple GPU family 7"
            echo ""
            echo "  macos-metal2.4 (macOS 13, 2022) [DEFAULT]:"
            echo "    - A15/A16, M1 Pro/Max/Ultra, M2"
            echo "    - Apple GPU family 8"
            echo ""
            echo "  macos-metal3.0 (macOS 14, 2023):"
            echo "    - A17 Pro, M3"
            echo "    - Apple GPU family 9, mesh shaders"
            echo ""
            echo "  macos-metal3.1 (macOS 15, 2024):"
            echo "    - A18, M4"
            echo "    - Enhanced ray tracing"
            echo ""
            echo "  macos-metal3.2 (macOS 16, 2025-2026):"
            echo "    - M5, future chips"
            echo "    - Neural Engine in GPU cores"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: Metal compilation requires macOS"
    exit 1
fi

# Check for xcrun
if ! command -v xcrun &> /dev/null; then
    echo "Error: xcrun not found. Install Xcode Command Line Tools:"
    echo "  xcode-select --install"
    exit 1
fi

# Get Xcode/Metal version info
XCODE_VERSION=$(xcrun --version 2>&1 | head -1)
METAL_VERSION=$(xcrun -sdk macosx metal --version 2>&1 | head -1)

echo "=== Metal Kernel Compilation ==="
echo "Kernel directory: $KERNEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Metal standard: $METAL_STD"
echo "Xcode: $XCODE_VERSION"
echo "Metal compiler: $METAL_VERSION"
echo ""

# Compile each .metal file to .metallib
compile_metal() {
    local name="$1"
    local metal_src="$KERNEL_DIR/${name}.metal"
    local air_file="$OUTPUT_DIR/${name}.air"
    local metallib_file="$OUTPUT_DIR/${name}.metallib"

    if [[ ! -f "$metal_src" ]]; then
        echo "Warning: $metal_src not found, skipping"
        return
    fi

    echo "Compiling $name (${METAL_STD})..."

    # Step 1: Compile .metal to .air (Metal IR)
    xcrun -sdk macosx metal \
        -c "$metal_src" \
        -o "$air_file" \
        -std="$METAL_STD" \
        -O2

    # Step 2: Link .air to .metallib
    xcrun -sdk macosx metallib \
        "$air_file" \
        -o "$metallib_file"

    # Clean up intermediate .air file
    rm -f "$air_file"

    echo "  Created: $metallib_file"
    ls -lh "$metallib_file"
}

# Compile all kernels
compile_metal "flash_attention"
compile_metal "paged_attention"

echo ""
echo "=== Compilation Complete ==="
echo ""
echo "To use precompiled metallib in Rust:"
echo '  const METALLIB: &[u8] = include_bytes!("kernels/flash_attention.metallib");'
echo '  let loader = MetalKernelLoader::from_bytes(METALLIB)?;'
