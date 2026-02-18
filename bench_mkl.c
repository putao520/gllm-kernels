#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench(const char *label, int m, int n, int k, int warmup, int iters) {
    float *a = (float*)mkl_malloc(m * k * sizeof(float), 64);
    float *b = (float*)mkl_malloc(k * n * sizeof(float), 64);
    float *c = (float*)mkl_malloc(m * n * sizeof(float), 64);

    for (int i = 0; i < m*k; i++) a[i] = (float)(i % 97) * 0.01f - 0.5f;
    for (int i = 0; i < k*n; i++) b[i] = (float)(i % 89) * 0.01f - 0.5f;

    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmup; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);

    double *times = (double*)malloc(iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
        times[i] = now_sec() - t0;
    }

    for (int i = 0; i < iters-1; i++)
        for (int j = i+1; j < iters; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

    double median = times[iters/2];
    double gflops = 2.0 * m * n * k / median / 1e9;
    printf("%-45s %8.3fms %9.1f\n", label, median*1000, gflops);

    mkl_free(a); mkl_free(b); mkl_free(c); free(times);
}

static void run_suite(const char *label) {
    printf("\n========================================================================\n");
    printf("  %s (threads=%d)\n", label, mkl_get_max_threads());
    printf("========================================================================\n");
    printf("%-45s %10s %10s\n", "Shape", "Time", "GFLOPS");
    printf("------------------------------------------------------------------------\n");
    bench("M=1 K=4096 N=4096 (decode GEMV)",      1,    4096, 4096, 10, 50);
    bench("M=32 K=4096 N=4096 (small batch)",      32,   4096, 4096, 5, 20);
    bench("M=128 K=4096 N=11008 (FFN prefill)",    128,  4096, 11008, 3, 10);
    bench("M=512 K=4096 N=4096 (medium prefill)",  512,  4096, 4096, 3, 10);
    bench("M=2048 K=4096 N=4096 (large prefill)",  2048, 4096, 4096, 2, 5);
}

int main(void) {
    char ver[256];
    mkl_get_version_string(ver, 256);
    printf("MKL version: %s\n", ver);

    mkl_set_num_threads(1);
    run_suite("MKL SGEMM - Single Thread");

    mkl_set_num_threads(10);
    run_suite("MKL SGEMM - 10 Threads");

    mkl_set_num_threads(20);
    run_suite("MKL SGEMM - 20 Threads");

    return 0;
}
