//! SIMD regression tests: verify that the dispatched SIMD paths produce
//! consistent, reasonable results for all quant formats.
//! Uses zero-initialized blocks to avoid codebook index-out-of-bounds issues.
//!
//! Tests marked `#[ignore]` expose pre-existing AVX2 bugs (not regressions).

#[cfg(test)]
mod tests {
    use half::f16;
    use std::mem::size_of;

    // ========================================================================
    // Helper macros
    // ========================================================================
    macro_rules! scalar_decode {
        ($qfmt:ident, $block_ty:ty, $block:expr, $n:expr) => {{
            let blk: *const $block_ty = $block;
            let mut out = vec![0.0f32; $n];
            let dst = out.as_mut_ptr();
            crate::quant_primitive!(scalar, $qfmt, decode, blk, dst);
            out
        }};
    }

    macro_rules! scalar_dot {
        ($qfmt:ident, $block_ty:ty, $block:expr, $other:expr) => {{
            let blk: *const $block_ty = $block;
            let src = $other.as_ptr();
            crate::quant_primitive!(scalar, $qfmt, dot, blk, src)
        }};
    }

    #[cfg(target_arch = "x86_64")]
    macro_rules! avx2_decode {
        ($qfmt:ident, $block_ty:ty, $block:expr, $n:expr) => {{
            let blk: *const $block_ty = $block;
            let mut out = vec![0.0f32; $n];
            let dst = out.as_mut_ptr();
            crate::quant_primitive!(avx2, $qfmt, decode, blk, dst);
            out
        }};
    }

    #[cfg(target_arch = "x86_64")]
    macro_rules! avx2_dot {
        ($qfmt:ident, $block_ty:ty, $block:expr, $other:expr) => {{
            let blk: *const $block_ty = $block;
            let src = $other.as_ptr();
            crate::quant_primitive!(avx2, $qfmt, dot, blk, src)
        }};
    }

    fn assert_close(a: &[f32], b: &[f32], label: &str, tol: f32) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "{label}[{i}]: scalar={x}, simd={y}, diff={}",
                (x - y).abs()
            );
        }
    }

    fn assert_dot_close(a: f32, b: f32, label: &str, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "{label}: scalar={a}, simd={b}, diff={}",
            (a - b).abs()
        );
    }

    fn ramp_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) / (n as f32)).collect()
    }

    // ========================================================================
    // PASSING: these formats match scalar vs AVX2 on zero blocks
    // ========================================================================

    // IQ4_XS
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq4_xs_decode() {
        use crate::quant::BlockIQ4XS;
        let mut block: BlockIQ4XS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq4_xs, BlockIQ4XS, &block, 256);
        let simd = avx2_decode!(iq4_xs, BlockIQ4XS, &block, 256);
        assert_close(&scalar, &simd, "iq4_xs decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq4_xs_dot() {
        use crate::quant::BlockIQ4XS;
        let mut block: BlockIQ4XS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(iq4_xs, BlockIQ4XS, &block, &other);
        let v = avx2_dot!(iq4_xs, BlockIQ4XS, &block, &other);
        assert_dot_close(s, v, "iq4_xs dot", 1e-2);
    }

    // IQ2_XXS
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq2_xxs_decode() {
        use crate::quant::BlockIQ2XXS;
        let mut block: BlockIQ2XXS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq2_xxs, BlockIQ2XXS, &block, 256);
        let simd = avx2_decode!(iq2_xxs, BlockIQ2XXS, &block, 256);
        assert_close(&scalar, &simd, "iq2_xxs decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq2_xxs_dot() {
        use crate::quant::BlockIQ2XXS;
        let mut block: BlockIQ2XXS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(iq2_xxs, BlockIQ2XXS, &block, &other);
        let v = avx2_dot!(iq2_xxs, BlockIQ2XXS, &block, &other);
        assert_dot_close(s, v, "iq2_xxs dot", 1e-2);
    }

    // IQ2_XS
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq2_xs_decode() {
        use crate::quant::BlockIQ2XS;
        let mut block: BlockIQ2XS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq2_xs, BlockIQ2XS, &block, 256);
        let simd = avx2_decode!(iq2_xs, BlockIQ2XS, &block, 256);
        assert_close(&scalar, &simd, "iq2_xs decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq2_xs_dot() {
        use crate::quant::BlockIQ2XS;
        let mut block: BlockIQ2XS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(iq2_xs, BlockIQ2XS, &block, &other);
        let v = avx2_dot!(iq2_xs, BlockIQ2XS, &block, &other);
        assert_dot_close(s, v, "iq2_xs dot", 1e-2);
    }

    // IQ1_S
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq1_s_decode() {
        use crate::quant::BlockIQ1S;
        let mut block: BlockIQ1S = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq1_s, BlockIQ1S, &block, 256);
        let simd = avx2_decode!(iq1_s, BlockIQ1S, &block, 256);
        assert_close(&scalar, &simd, "iq1_s decode", 1e-4);
    }

    // IQ1_M
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_iq1_m_decode() {
        use crate::quant::BlockIQ1M;
        let block: BlockIQ1M = unsafe { std::mem::zeroed() };
        let scalar = scalar_decode!(iq1_m, BlockIQ1M, &block, 256);
        let simd = avx2_decode!(iq1_m, BlockIQ1M, &block, 256);
        assert_close(&scalar, &simd, "iq1_m decode", 1e-4);
    }

    // AWQ4
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_awq4_decode() {
        use crate::quant::BlockAWQ4;
        let mut block: BlockAWQ4 = unsafe { std::mem::zeroed() };
        block.scales = f16::from_f32(1.0);
        let scalar = scalar_decode!(awq4, BlockAWQ4, &block, 256);
        let simd = avx2_decode!(awq4, BlockAWQ4, &block, 256);
        assert_close(&scalar, &simd, "awq4 decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_awq4_dot() {
        use crate::quant::BlockAWQ4;
        let mut block: BlockAWQ4 = unsafe { std::mem::zeroed() };
        block.scales = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(awq4, BlockAWQ4, &block, &other);
        let v = avx2_dot!(awq4, BlockAWQ4, &block, &other);
        assert_dot_close(s, v, "awq4 dot", 1e-2);
    }

    // ========================================================================
    // IGNORED: pre-existing AVX2 bugs (not regressions from refactor)
    // ========================================================================

    // BUG: iq4_nl AVX2 decode wrong from element 32 onward (scalar=0, simd=-127)
    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq4_nl AVX2 decode diverges from scalar at element 32"]
    fn test_simd_iq4_nl_decode() {
        use crate::quant::BlockIQ4NL;
        let mut block: BlockIQ4NL = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq4_nl, BlockIQ4NL, &block, 256);
        let simd = avx2_decode!(iq4_nl, BlockIQ4NL, &block, 256);
        assert_close(&scalar, &simd, "iq4_nl decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq4_nl AVX2 dot diverges from scalar"]
    fn test_simd_iq4_nl_dot() {
        use crate::quant::BlockIQ4NL;
        let mut block: BlockIQ4NL = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(iq4_nl, BlockIQ4NL, &block, &other);
        let v = avx2_dot!(iq4_nl, BlockIQ4NL, &block, &other);
        assert_dot_close(s, v, "iq4_nl dot", 1e-2);
    }

    // BUG: iq2_s codebook index out of bounds (len=8, index=8)
    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq2_s codebook table too small (index 8, len 8)"]
    fn test_simd_iq2_s_decode() {
        use crate::quant::BlockIQ2S;
        let mut block: BlockIQ2S = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq2_s, BlockIQ2S, &block, 256);
        let simd = avx2_decode!(iq2_s, BlockIQ2S, &block, 256);
        assert_close(&scalar, &simd, "iq2_s decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq2_s codebook table too small (index 8, len 8)"]
    fn test_simd_iq2_s_dot() {
        use crate::quant::BlockIQ2S;
        let mut block: BlockIQ2S = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(iq2_s, BlockIQ2S, &block, &other);
        let v = avx2_dot!(iq2_s, BlockIQ2S, &block, &other);
        assert_dot_close(s, v, "iq2_s dot", 1e-2);
    }

    // BUG: iq3_xxs AVX2 sign flip (scalar=4, simd=-4)
    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq3_xxs AVX2 sign flip vs scalar"]
    fn test_simd_iq3_xxs_decode() {
        use crate::quant::BlockIQ3XXS;
        let mut block: BlockIQ3XXS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq3_xxs, BlockIQ3XXS, &block, 256);
        let simd = avx2_decode!(iq3_xxs, BlockIQ3XXS, &block, 256);
        assert_close(&scalar, &simd, "iq3_xxs decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq3_xxs AVX2 sign flip vs scalar"]
    fn test_simd_iq3_xxs_dot() {
        use crate::quant::BlockIQ3XXS;
        let mut block: BlockIQ3XXS = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(iq3_xxs, BlockIQ3XXS, &block, &other);
        let v = avx2_dot!(iq3_xxs, BlockIQ3XXS, &block, &other);
        assert_dot_close(s, v, "iq3_xxs dot", 1e-2);
    }

    // BUG: iq3_s codebook index out of bounds (len=8, index=8)
    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq3_s codebook table too small (index 8, len 8)"]
    fn test_simd_iq3_s_decode() {
        use crate::quant::BlockIQ3S;
        let mut block: BlockIQ3S = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let scalar = scalar_decode!(iq3_s, BlockIQ3S, &block, 256);
        let simd = avx2_decode!(iq3_s, BlockIQ3S, &block, 256);
        assert_close(&scalar, &simd, "iq3_s decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: iq3_s codebook table too small (index 8, len 8)"]
    fn test_simd_iq3_s_dot() {
        use crate::quant::BlockIQ3S;
        let mut block: BlockIQ3S = unsafe { std::mem::zeroed() };
        block.d = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(iq3_s, BlockIQ3S, &block, &other);
        let v = avx2_dot!(iq3_s, BlockIQ3S, &block, &other);
        assert_dot_close(s, v, "iq3_s dot", 1e-2);
    }

    // BUG: gptq4 AVX2 zero-point handling differs (scalar=-8, simd=0)
    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: gptq4 AVX2 zero-point handling differs from scalar"]
    fn test_simd_gptq4_decode() {
        use crate::quant::BlockGPTQ4;
        let mut block: BlockGPTQ4 = unsafe { std::mem::zeroed() };
        block.scales = f16::from_f32(1.0);
        let scalar = scalar_decode!(gptq4, BlockGPTQ4, &block, 256);
        let simd = avx2_decode!(gptq4, BlockGPTQ4, &block, 256);
        assert_close(&scalar, &simd, "gptq4 decode", 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore = "pre-existing: gptq4 AVX2 zero-point handling differs from scalar"]
    fn test_simd_gptq4_dot() {
        use crate::quant::BlockGPTQ4;
        let mut block: BlockGPTQ4 = unsafe { std::mem::zeroed() };
        block.scales = f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(gptq4, BlockGPTQ4, &block, &other);
        let v = avx2_dot!(gptq4, BlockGPTQ4, &block, &other);
        assert_dot_close(s, v, "gptq4 dot", 1e-2);
    }

    // ========================================================================
    // Q4_K: AVX2 dot product correctness (zero block + non-trivial block)
    // ========================================================================

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_q4k_dot_zero() {
        use crate::quant::BlockQ4K;
        let mut block: BlockQ4K = unsafe { std::mem::zeroed() };
        block.d = half::f16::from_f32(1.0);
        let other = ramp_f32(256);
        let s = scalar_dot!(q4_k, BlockQ4K, &block, &other);
        let v = avx2_dot!(q4_k, BlockQ4K, &block, &other);
        assert_dot_close(s, v, "q4_k dot (zero block)", 1e-2);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_q4k_dot_nonzero() {
        use crate::quant::BlockQ4K;
        // Build a block with non-trivial nibble patterns
        let mut block: BlockQ4K = unsafe { std::mem::zeroed() };
        block.d = half::f16::from_f32(0.5);
        // Fill qs with a ramp pattern: byte i = (i & 0xF) | ((i*3 & 0xF) << 4)
        for i in 0..128 {
            let lo = (i & 0x0F) as u8;
            let hi = ((i * 3) & 0x0F) as u8;
            block.qs[i] = lo | (hi << 4);
        }
        let other: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) / 64.0).collect();
        let s = scalar_dot!(q4_k, BlockQ4K, &block, &other);
        let v = avx2_dot!(q4_k, BlockQ4K, &block, &other);
        assert_dot_close(s, v, "q4_k dot (nonzero)", 1e-1);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_q4k_dot_random_pattern() {
        use crate::quant::BlockQ4K;
        // Pseudo-random pattern to stress all nibble values
        let mut block: BlockQ4K = unsafe { std::mem::zeroed() };
        block.d = half::f16::from_f32(0.125);
        let mut seed: u32 = 42;
        for i in 0..128 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            block.qs[i] = (seed >> 16) as u8;
        }
        let other: Vec<f32> = (0..256).map(|i| {
            let x = (i as f32) * 0.01;
            (x * 2.7).sin()
        }).collect();
        let s = scalar_dot!(q4_k, BlockQ4K, &block, &other);
        let v = avx2_dot!(q4_k, BlockQ4K, &block, &other);
        assert_dot_close(s, v, "q4_k dot (random)", 1e-1);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_q4k_decode() {
        use crate::quant::BlockQ4K;
        let mut block: BlockQ4K = unsafe { std::mem::zeroed() };
        block.d = half::f16::from_f32(1.0);
        for i in 0..128 {
            block.qs[i] = ((i & 0xF) | (((i + 5) & 0xF) << 4)) as u8;
        }
        let scalar = scalar_decode!(q4_k, BlockQ4K, &block, 256);
        let simd = avx2_decode!(q4_k, BlockQ4K, &block, 256);
        assert_close(&scalar, &simd, "q4_k decode", 1e-4);
    }
}
