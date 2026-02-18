#!/usr/bin/env python3
"""Benchmark MKL SGEMM via ctypes at LLM-relevant shapes."""
import ctypes
import time
import numpy as np
import os

# Load MKL
mkl_path = "/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so"
try:
    mkl = ctypes.CDLL(mkl_path)
except OSError as e:
    print(f"Cannot load MKL: {e}")
    exit(1)

# cblas_sgemm signature
# void cblas_sgemm(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
#                  MKL_INT M, MKL_INT N, MKL_INT K,
#                  float alpha, const float *A, MKL_INT lda,
#                  const float *B, MKL_INT ldb,
#                  float beta, float *C, MKL_INT ldc);
CblasRowMajor = 101
CblasNoTrans = 111

mkl.cblas_sgemm.restype = None
mkl.cblas_sgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # layout, transA, transB
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M, N, K
    ctypes.c_float,                              # alpha
    ctypes.c_void_p, ctypes.c_int,              # A, lda
    ctypes.c_void_p, ctypes.c_int,              # B, ldb
    ctypes.c_float,                              # beta
    ctypes.c_void_p, ctypes.c_int,              # C, ldc
]

# Set thread count
mkl.mkl_set_num_threads.restype = None
mkl.mkl_set_num_threads.argtypes = [ctypes.c_int]

def bench_mkl_sgemm(m, n, k, warmup=3, iters=10):
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    c = np.zeros((m, n), dtype=np.float32)

    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    for _ in range(warmup):
        mkl.cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, alpha,
                        a.ctypes.data, k,
                        b.ctypes.data, n,
                        beta, c.ctypes.data, n)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mkl.cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, alpha,
                        a.ctypes.data, k,
                        b.ctypes.data, n,
                        beta, c.ctypes.data, n)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median = times[len(times) // 2]
    gflops = 2.0 * m * n * k / median / 1e9
    return median, gflops

shapes = [
    ("M=1 K=4096 N=4096 (decode GEMV)",      1,    4096, 4096, 50),
    ("M=32 K=4096 N=4096 (small batch)",      32,   4096, 4096, 20),
    ("M=128 K=4096 N=11008 (FFN prefill)",    128,  4096, 11008, 10),
    ("M=512 K=4096 N=4096 (medium prefill)",  512,  4096, 4096, 10),
    ("M=2048 K=4096 N=4096 (large prefill)",  2048, 4096, 4096, 5),
]

def run_suite(label, threads):
    mkl.mkl_set_num_threads(threads)
    print(f"\n{'='*72}")
    print(f"  {label} (threads={threads})")
    print(f"{'='*72}")
    print(f"{'Shape':<45} {'Time':>10} {'GFLOPS':>10}")
    print("-" * 72)
    for name, m, n, k, iters in shapes:
        median, gflops = bench_mkl_sgemm(m, n, k, iters=iters)
        print(f"{name:<45} {median*1000:>8.3f}ms {gflops:>9.1f}")

if __name__ == "__main__":
    print("MKL loaded from:", mkl_path)
    run_suite("MKL SGEMM - Single Thread", 1)
    ncpu = os.cpu_count() or 10
    run_suite(f"MKL SGEMM - All Cores ({ncpu})", ncpu)
