extern "C" __global__ void sgemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Simple tiled SGEMM or naive for now?
    // Let's implement a naive one first to verify pipeline, then optimize.
    // Index: C[row, col]
    // Global ID
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A is MxK (row-major) -> A[row * K + k]
            // B is KxN (row-major) -> B[k * N + col]
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
