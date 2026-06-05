pub mod math;

// Expand AVX-512 float-32 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512_f32, avx512, f32);
// Expand AVX-512 f16 implementations (F16C conversion path)
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512_f16, avx512, f16);
// Expand AVX-512 bf16 implementations
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512_bf16, avx512, bf16);

// AVX-512 FP16 native implementations (Sapphire Rapids+)
// Uses __m512h (32 x f16) with native FMA — 2x throughput vs F16C path.
#[cfg(target_arch = "x86_64")]
crate::expand_isa_impls!(avx512fp16_f16, avx512fp16, f16);

// AVX-512 VNNI INT8 GEMM (standalone — not through expand_isa_impls)
#[cfg(target_arch = "x86_64")]
pub mod avx512_int8 {
    crate::define_matmul_x86_int8!();
}

// AMX BF16 GEMM (tile-based matrix multiply)
#[cfg(target_arch = "x86_64")]
pub mod amx_bf16 {
    crate::define_matmul_x86_amx_bf16!();
}

// AMX INT8 GEMM (tile-based integer matrix multiply)
#[cfg(target_arch = "x86_64")]
pub mod amx_int8 {
    crate::define_matmul_x86_amx_int8!();
}
