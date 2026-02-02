#!/bin/bash
set -e

# gllm-kernels compilation script
# Usage: ./compile_kernels.sh [cuda|rocm|all]

TARGET=${1:-all}

# ==============================================================================
# CUDA (NVIDIA) - Output .cubin
# ==============================================================================
if [[ "$TARGET" == "all" || "$TARGET" == "cuda" ]]; then
    if command -v nvcc &> /dev/null; then
        echo "[CUDA] Found nvcc, compiling..."

        SRC_DIR="src/cuda_kernels/kernels"

        # Supported Architectures for AOT
        ARCHS=("sm_80" "sm_86" "sm_89" "sm_90")

        for ARCH in "${ARCHS[@]}"; do
            echo "  -> Compiling for $ARCH..."
            mkdir -p "$SRC_DIR"
            OUT_ARCH="${ARCH//_/}"
            nvcc -cubin -O3 -std=c++14 -arch=$ARCH "$SRC_DIR/kernels.cu" -o "$SRC_DIR/kernels_${OUT_ARCH}.cubin"
        done

        echo "[CUDA] Done."
    else
        echo "[CUDA] nvcc not found, skipping."
    fi
fi

# ==============================================================================
# ROCm (AMD) - Output .hsaco
# ==============================================================================
if [[ "$TARGET" == "all" || "$TARGET" == "rocm" ]]; then
    if command -v hipcc &> /dev/null; then
        echo "[ROCm] Found hipcc, compiling..."
        # Logic to be implemented
    else
        echo "[ROCm] hipcc not found, skipping."
    fi
fi

echo "Done."
