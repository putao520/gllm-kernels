#!/bin/bash
# Compile all GPU kernels (CUDA and HIP)
# Usage: ./compile_all_kernels.sh [--cuda-only] [--hip-only]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMPILE_CUDA=true
COMPILE_HIP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-only)
            COMPILE_HIP=false
            shift
            ;;
        --hip-only)
            COMPILE_CUDA=false
            shift
            ;;
        --help)
            echo "Usage: $0 [--cuda-only] [--hip-only]"
            echo ""
            echo "Compiles GPU kernels for gllm-kernels:"
            echo "  CUDA: .cu -> .ptx (NVIDIA)"
            echo "  HIP:  .hip -> .hsaco (AMD)"
            echo ""
            echo "Options:"
            echo "  --cuda-only  Only compile CUDA kernels"
            echo "  --hip-only   Only compile HIP kernels"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "  gllm-kernels GPU Kernel Compiler"
echo "========================================"
echo ""

SUCCESS=0
FAILED=0

if [ "$COMPILE_CUDA" = true ]; then
    echo ">>> Compiling CUDA kernels..."
    if command -v nvcc &> /dev/null || [ -n "$CUDA_HOME" ]; then
        "$SCRIPT_DIR/compile_cuda_kernels.sh" && SUCCESS=$((SUCCESS+1)) || FAILED=$((FAILED+1))
    else
        echo "[SKIP] nvcc not found, skipping CUDA kernels"
    fi
    echo ""
fi

if [ "$COMPILE_HIP" = true ]; then
    echo ">>> Compiling HIP kernels..."
    if command -v hipcc &> /dev/null || [ -n "$ROCM_PATH" ] || [ -d "/opt/rocm" ]; then
        "$SCRIPT_DIR/compile_hip_kernels.sh" && SUCCESS=$((SUCCESS+1)) || FAILED=$((FAILED+1))
    else
        echo "[SKIP] hipcc not found, skipping HIP kernels"
    fi
    echo ""
fi

echo "========================================"
echo "  Compilation Summary"
echo "========================================"
echo "  Success: $SUCCESS"
echo "  Failed:  $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Some compilations failed!"
    exit 1
fi

echo "All kernel compilations successful!"
