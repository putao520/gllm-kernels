#![allow(dead_code)]

use rand::Rng;

/// GEMM GFLOPS: 2*M*N*K / duration / 1e9
pub fn gemm_gflops(m: usize, n: usize, k: usize, duration_secs: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / duration_secs / 1e9
}

/// 内存吞吐量 GiB/s
pub fn throughput_gibs(bytes: usize, duration_secs: f64) -> f64 {
    bytes as f64 / duration_secs / (1024.0 * 1024.0 * 1024.0)
}

/// 元素吞吐量 elements/sec
pub fn throughput_elements(elements: usize, duration_secs: f64) -> f64 {
    elements as f64 / duration_secs
}

/// GEMM 的 FLOP 数 (multiply-add = 2 ops)
pub fn gemm_flops(m: usize, n: usize, k: usize) -> u64 {
    2 * m as u64 * n as u64 * k as u64
}

/// Elementwise 算子的读写字节数 (in + out, f32)
pub fn elementwise_rw_bytes(n: usize) -> u64 {
    2 * n as u64 * 4
}

/// RmsNorm 的读写字节数 (input + weight + output, f32)
pub fn rmsnorm_rw_bytes(n: usize) -> u64 {
    3 * n as u64 * 4
}

/// Softmax 的读写字节数 (3-pass: 读3次 + 写2次)
pub fn softmax_rw_bytes(n: usize) -> u64 {
    5 * n as u64 * 4
}

/// 生成随机 f32 向量 [-1.0, 1.0)
pub fn random_f32_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// 生成随机 i8 向量
pub fn random_i8_vec(n: usize) -> Vec<i8> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-128..127i8)).collect()
}

/// 生成随机 u8 向量
pub fn random_u8_vec(n: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0..=255u8)).collect()
}

/// 生成随机 f32 scale 向量 (0.001..0.1)
pub fn random_scale_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0.001..0.1)).collect()
}
