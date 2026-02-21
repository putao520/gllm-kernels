/// Commercial format quantization primitives (AWQ4, GPTQ4, Squeeze).
///
/// Sub-macro of the Layer 3 quant_primitive dispatcher.
/// Currently scalar-only.
#[macro_export]
macro_rules! quant_primitive_commercial {
    (scalar, awq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let out_ptr = $out_ptr;
            // AWQ4 block_size=128: 32 u32 words, each holding 8 nibbles
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { *out_ptr.add(w * 8 + nib) = d * (q - 8.0); }
                }
            }
        }
    };

    (scalar, awq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { sum += (d * (q - 8.0)) * *other.add(w * 8 + nib); }
                }
            }
            sum
        }
    };

    // GPTQ4: same layout as AWQ4 for decode/dot
    (scalar, gptq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let out_ptr = $out_ptr;
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { *out_ptr.add(w * 8 + nib) = d * (q - 8.0); }
                }
            }
        }
    };

    (scalar, gptq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.scales.to_f32();
            let qw = &block.qweight;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for w in 0..32 {
                let word = qw[w];
                for nib in 0..8 {
                    let q = ((word >> (nib * 4)) & 0xF) as f32;
                    unsafe { sum += (d * (q - 8.0)) * *other.add(w * 8 + nib); }
                }
            }
            sum
        }
    };

    // Squeeze: 3-bit SqueezeLLM
    (scalar, squeeze, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let out_ptr = $out_ptr;
            // 3-bit packed: 256 values in 96 bytes (256*3/8=96), but block has 128 bytes qs
            for i in 0..256 {
                let bit_offset = i * 3;
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let q = if bit_idx <= 5 {
                    (qs[byte_idx] >> bit_idx) & 0x07
                } else {
                    let lo = qs[byte_idx] >> bit_idx;
                    let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                    (lo | hi) & 0x07
                };
                unsafe { *out_ptr.add(i) = (q as f32).mul_add(d, -d * 4.0); }
            }
        }
    };

    (scalar, squeeze, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..256 {
                let bit_offset = i * 3;
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let q = if bit_idx <= 5 {
                    (qs[byte_idx] >> bit_idx) & 0x07
                } else {
                    let lo = qs[byte_idx] >> bit_idx;
                    let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                    (lo | hi) & 0x07
                };
                unsafe { sum += (q as f32).mul_add(d, -d * 4.0) * *other.add(i); }
            }
            sum
        }
    };

    // ========================================================================
    // AVX2 Squeeze decode/dot
    // ========================================================================

    // Squeeze decode: 256 × 3-bit values, 8-wide AVX2
    // out[i] = d * (q3 - 4.0)
    (avx2, squeeze, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;

                for i in 0..32 {
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let idx = i * 8 + j;
                        let bit_offset = idx * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_idx = bit_offset % 8;
                        let q = if bit_idx <= 5 {
                            (qs[byte_idx] >> bit_idx) & 0x07
                        } else {
                            let lo = qs[byte_idx] >> bit_idx;
                            let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                            (lo | hi) & 0x07
                        };
                        vals[j] = q as f32 - 4.0;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 8), res);
                }
            }
        }
    };

    // Squeeze dot: sum += d * (q3 - 4.0) * other[i], 8-wide AVX2
    (avx2, squeeze, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);

                for i in 0..32 {
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let idx = i * 8 + j;
                        let bit_offset = idx * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_idx = bit_offset % 8;
                        let q = if bit_idx <= 5 {
                            (qs[byte_idx] >> bit_idx) & 0x07
                        } else {
                            let lo = qs[byte_idx] >> bit_idx;
                            let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                            (lo | hi) & 0x07
                        };
                        vals[j] = q as f32 - 4.0;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(i * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx2, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // AVX-512 Squeeze decode/dot
    // ========================================================================

    // Squeeze decode: 256 × 3-bit values, 16-wide AVX-512
    // out[i] = d * (q3 - 4.0)
    (avx512, squeeze, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;

                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..16 {
                        let idx = i * 16 + j;
                        let bit_offset = idx * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_idx = bit_offset % 8;
                        let q = if bit_idx <= 5 {
                            (qs[byte_idx] >> bit_idx) & 0x07
                        } else {
                            let lo = qs[byte_idx] >> bit_idx;
                            let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                            (lo | hi) & 0x07
                        };
                        vals[j] = q as f32 - 4.0;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i * 16), res);
                }
            }
        }
    };

    // Squeeze dot: sum += d * (q3 - 4.0) * other[i], 16-wide AVX-512
    (avx512, squeeze, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..16 {
                        let idx = i * 16 + j;
                        let bit_offset = idx * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_idx = bit_offset % 8;
                        let q = if bit_idx <= 5 {
                            (qs[byte_idx] >> bit_idx) & 0x07
                        } else {
                            let lo = qs[byte_idx] >> bit_idx;
                            let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                            (lo | hi) & 0x07
                        };
                        vals[j] = q as f32 - 4.0;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(i * 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // NEON Squeeze decode/dot
    // ========================================================================

    // Squeeze decode: 256 × 3-bit values, 4-wide NEON
    // out[i] = d * (q3 - 4.0)
    (neon, squeeze, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;

                for i in 0..64 {
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        let idx = i * 4 + j;
                        let bit_offset = idx * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_idx = bit_offset % 8;
                        let q = if bit_idx <= 5 {
                            (qs[byte_idx] >> bit_idx) & 0x07
                        } else {
                            let lo = qs[byte_idx] >> bit_idx;
                            let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                            (lo | hi) & 0x07
                        };
                        vals[j] = q as f32 - 4.0;
                    }
                    let vf = $crate::simd_primitive!(neon, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(neon, f32, mul, vd, vf);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(i * 4), res);
                }
            }
        }
    };

    // Squeeze dot: sum += d * (q3 - 4.0) * other[i], 4-wide NEON
    (neon, squeeze, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, zero);

                for i in 0..64 {
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        let idx = i * 4 + j;
                        let bit_offset = idx * 3;
                        let byte_idx = bit_offset / 8;
                        let bit_idx = bit_offset % 8;
                        let q = if bit_idx <= 5 {
                            (qs[byte_idx] >> bit_idx) & 0x07
                        } else {
                            let lo = qs[byte_idx] >> bit_idx;
                            let hi = if byte_idx + 1 < qs.len() { qs[byte_idx + 1] << (8 - bit_idx) } else { 0 };
                            (lo | hi) & 0x07
                        };
                        vals[j] = q as f32 - 4.0;
                    }
                    let vf = $crate::simd_primitive!(neon, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(neon, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(neon, f32, load, other.add(i * 4));
                    acc = $crate::simd_primitive!(neon, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // AVX2 AWQ4/GPTQ4 decode/dot
    // ========================================================================

    // AWQ4 decode: 32 u32 words × 8 nibbles = 256 values
    // out[w*8+nib] = d * (nibble - 8.0)
    (avx2, awq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_neg_d_offset = $crate::simd_primitive!(avx2, f32, splat, -d * 8.0);
                let qw = &block.qweight;
                let out_ptr = $out_ptr;

                // Each u32 word contains 8 x 4-bit nibbles → 8 f32 outputs = one __m256
                for w in 0..32 {
                    let word = qw[w];
                    // Extract 8 nibbles into f32 array
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx2, f32, load, nibbles.as_ptr());
                    // d * (q - 8.0)
                    let res = $crate::simd_primitive!(avx2, f32, fma, vd, vq, v_neg_d_offset);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(w * 8), res);
                }
            }
        }
    };

    // AWQ4 dot: sum += d * (nibble - 8.0) * other[w*8+nib]
    (avx2, awq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_neg_d_offset = $crate::simd_primitive!(avx2, f32, splat, -d * 8.0);
                let qw = &block.qweight;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);

                for w in 0..32 {
                    let word = qw[w];
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx2, f32, load, nibbles.as_ptr());
                    // dequant = d * (q - 8.0)
                    let dq = $crate::simd_primitive!(avx2, f32, fma, vd, vq, v_neg_d_offset);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(w * 8));
                    // acc += dq * vo
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }

                // Horizontal reduce
                $crate::simd_primitive!(avx2, f32, reduce_sum, acc)
            }
        }
    };

    // GPTQ4 decode: 32 u32 words × 8 nibbles = 256 values
    // out[w*8+nib] = d * (nibble - zero), zero = low nibble of block.zeros
    (avx2, gptq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let zero = (block.zeros & 0xF) as f32;
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_neg_d_zero = $crate::simd_primitive!(avx2, f32, splat, -d * zero);
                let qw = &block.qweight;
                let out_ptr = $out_ptr;

                for w in 0..32 {
                    let word = qw[w];
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx2, f32, load, nibbles.as_ptr());
                    // d * (q - zero)
                    let res = $crate::simd_primitive!(avx2, f32, fma, vd, vq, v_neg_d_zero);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(w * 8), res);
                }
            }
        }
    };

    // GPTQ4 dot: sum += d * (nibble - zero) * other[w*8+nib]
    (avx2, gptq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let zero = (block.zeros & 0xF) as f32;
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let v_neg_d_zero = $crate::simd_primitive!(avx2, f32, splat, -d * zero);
                let qw = &block.qweight;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);

                for w in 0..32 {
                    let word = qw[w];
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx2, f32, load, nibbles.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, fma, vd, vq, v_neg_d_zero);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(w * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx2, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // AVX-512 AWQ4/GPTQ4 decode/dot
    // ========================================================================

    // AWQ4 decode: pair two u32 words → 16 nibbles → one __m512
    // out[pair*16..pair*16+16] = d * (nibbles - 8.0)
    (avx512, awq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 8.0);
                let qw = &block.qweight;
                let out_ptr = $out_ptr;

                // 32 words / 2 = 16 pairs, each pair → 16 nibbles → one __m512
                for pair in 0..16 {
                    let w0 = qw[pair * 2];
                    let w1 = qw[pair * 2 + 1];
                    let nibbles: [f32; 16] = [
                        ((w0      ) & 0xF) as f32, ((w0 >>  4) & 0xF) as f32,
                        ((w0 >>  8) & 0xF) as f32, ((w0 >> 12) & 0xF) as f32,
                        ((w0 >> 16) & 0xF) as f32, ((w0 >> 20) & 0xF) as f32,
                        ((w0 >> 24) & 0xF) as f32, ((w0 >> 28) & 0xF) as f32,
                        ((w1      ) & 0xF) as f32, ((w1 >>  4) & 0xF) as f32,
                        ((w1 >>  8) & 0xF) as f32, ((w1 >> 12) & 0xF) as f32,
                        ((w1 >> 16) & 0xF) as f32, ((w1 >> 20) & 0xF) as f32,
                        ((w1 >> 24) & 0xF) as f32, ((w1 >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx512, f32, load, nibbles.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd, vq, v_neg_d_offset);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(pair * 16), res);
                }
            }
        }
    };

    // AWQ4 dot: sum += d * (nibble - 8.0) * other[pair*16+i]
    (avx512, awq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_neg_d_offset = $crate::simd_primitive!(avx512, f32, splat, -d * 8.0);
                let qw = &block.qweight;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for pair in 0..16 {
                    let w0 = qw[pair * 2];
                    let w1 = qw[pair * 2 + 1];
                    let nibbles: [f32; 16] = [
                        ((w0      ) & 0xF) as f32, ((w0 >>  4) & 0xF) as f32,
                        ((w0 >>  8) & 0xF) as f32, ((w0 >> 12) & 0xF) as f32,
                        ((w0 >> 16) & 0xF) as f32, ((w0 >> 20) & 0xF) as f32,
                        ((w0 >> 24) & 0xF) as f32, ((w0 >> 28) & 0xF) as f32,
                        ((w1      ) & 0xF) as f32, ((w1 >>  4) & 0xF) as f32,
                        ((w1 >>  8) & 0xF) as f32, ((w1 >> 12) & 0xF) as f32,
                        ((w1 >> 16) & 0xF) as f32, ((w1 >> 20) & 0xF) as f32,
                        ((w1 >> 24) & 0xF) as f32, ((w1 >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx512, f32, load, nibbles.as_ptr());
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd, vq, v_neg_d_offset);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(pair * 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // GPTQ4 decode: pair two u32 words → 16 nibbles → one __m512
    // out[pair*16..pair*16+16] = d * (nibbles - zero)
    (avx512, gptq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let zero = (block.zeros & 0xF) as f32;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_neg_d_zero = $crate::simd_primitive!(avx512, f32, splat, -d * zero);
                let qw = &block.qweight;
                let out_ptr = $out_ptr;

                for pair in 0..16 {
                    let w0 = qw[pair * 2];
                    let w1 = qw[pair * 2 + 1];
                    let nibbles: [f32; 16] = [
                        ((w0      ) & 0xF) as f32, ((w0 >>  4) & 0xF) as f32,
                        ((w0 >>  8) & 0xF) as f32, ((w0 >> 12) & 0xF) as f32,
                        ((w0 >> 16) & 0xF) as f32, ((w0 >> 20) & 0xF) as f32,
                        ((w0 >> 24) & 0xF) as f32, ((w0 >> 28) & 0xF) as f32,
                        ((w1      ) & 0xF) as f32, ((w1 >>  4) & 0xF) as f32,
                        ((w1 >>  8) & 0xF) as f32, ((w1 >> 12) & 0xF) as f32,
                        ((w1 >> 16) & 0xF) as f32, ((w1 >> 20) & 0xF) as f32,
                        ((w1 >> 24) & 0xF) as f32, ((w1 >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx512, f32, load, nibbles.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, fma, vd, vq, v_neg_d_zero);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(pair * 16), res);
                }
            }
        }
    };

    // GPTQ4 dot: sum += d * (nibble - zero) * other[pair*16+i]
    (avx512, gptq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let zero = (block.zeros & 0xF) as f32;
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let v_neg_d_zero = $crate::simd_primitive!(avx512, f32, splat, -d * zero);
                let qw = &block.qweight;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                for pair in 0..16 {
                    let w0 = qw[pair * 2];
                    let w1 = qw[pair * 2 + 1];
                    let nibbles: [f32; 16] = [
                        ((w0      ) & 0xF) as f32, ((w0 >>  4) & 0xF) as f32,
                        ((w0 >>  8) & 0xF) as f32, ((w0 >> 12) & 0xF) as f32,
                        ((w0 >> 16) & 0xF) as f32, ((w0 >> 20) & 0xF) as f32,
                        ((w0 >> 24) & 0xF) as f32, ((w0 >> 28) & 0xF) as f32,
                        ((w1      ) & 0xF) as f32, ((w1 >>  4) & 0xF) as f32,
                        ((w1 >>  8) & 0xF) as f32, ((w1 >> 12) & 0xF) as f32,
                        ((w1 >> 16) & 0xF) as f32, ((w1 >> 20) & 0xF) as f32,
                        ((w1 >> 24) & 0xF) as f32, ((w1 >> 28) & 0xF) as f32,
                    ];
                    let vq = $crate::simd_primitive!(avx512, f32, load, nibbles.as_ptr());
                    let dq = $crate::simd_primitive!(avx512, f32, fma, vd, vq, v_neg_d_zero);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(pair * 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // NEON AWQ4/GPTQ4 decode/dot
    // ========================================================================

    // AWQ4 decode: one u32 word → 8 nibbles → two float32x4_t (4-wide each)
    // out[w*8..w*8+8] = d * (nibbles - 8.0)
    (neon, awq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 8.0);
                let qw = &block.qweight;
                let out_ptr = $out_ptr;

                for w in 0..32 {
                    let word = qw[w];
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq0 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr());
                    let vq1 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr().add(4));
                    let res0 = $crate::simd_primitive!(neon, f32, fma, vd, vq0, v_neg_d_offset);
                    let res1 = $crate::simd_primitive!(neon, f32, fma, vd, vq1, v_neg_d_offset);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(w * 8), res0);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(w * 8 + 4), res1);
                }
            }
        }
    };

    // AWQ4 dot: sum += d * (nibble - 8.0) * other[w*8+i]
    (neon, awq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_neg_d_offset = $crate::simd_primitive!(neon, f32, splat, -d * 8.0);
                let qw = &block.qweight;
                let other = $other_ptr;
                let mut acc0 = $crate::simd_primitive!(neon, f32, zero);
                let mut acc1 = $crate::simd_primitive!(neon, f32, zero);

                for w in 0..32 {
                    let word = qw[w];
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq0 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr());
                    let vq1 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr().add(4));
                    let dq0 = $crate::simd_primitive!(neon, f32, fma, vd, vq0, v_neg_d_offset);
                    let dq1 = $crate::simd_primitive!(neon, f32, fma, vd, vq1, v_neg_d_offset);
                    let vo0 = $crate::simd_primitive!(neon, f32, load, other.add(w * 8));
                    let vo1 = $crate::simd_primitive!(neon, f32, load, other.add(w * 8 + 4));
                    acc0 = $crate::simd_primitive!(neon, f32, fma, dq0, vo0, acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, dq1, vo1, acc1);
                }

                let combined = $crate::simd_primitive!(neon, f32, add, acc0, acc1);
                $crate::simd_primitive!(neon, f32, reduce_sum, combined)
            }
        }
    };

    // GPTQ4 decode: one u32 word → 8 nibbles → two float32x4_t
    // out[w*8..w*8+8] = d * (nibbles - zero)
    (neon, gptq4, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let zero = (block.zeros & 0xF) as f32;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_neg_d_zero = $crate::simd_primitive!(neon, f32, splat, -d * zero);
                let qw = &block.qweight;
                let out_ptr = $out_ptr;

                for w in 0..32 {
                    let word = qw[w];
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq0 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr());
                    let vq1 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr().add(4));
                    let res0 = $crate::simd_primitive!(neon, f32, fma, vd, vq0, v_neg_d_zero);
                    let res1 = $crate::simd_primitive!(neon, f32, fma, vd, vq1, v_neg_d_zero);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(w * 8), res0);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(w * 8 + 4), res1);
                }
            }
        }
    };

    // GPTQ4 dot: sum += d * (nibble - zero) * other[w*8+i]
    (neon, gptq4, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;

                let block = &*$block_ptr;
                let d: f32 = block.scales.to_f32();
                let zero = (block.zeros & 0xF) as f32;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);
                let v_neg_d_zero = $crate::simd_primitive!(neon, f32, splat, -d * zero);
                let qw = &block.qweight;
                let other = $other_ptr;
                let mut acc0 = $crate::simd_primitive!(neon, f32, zero);
                let mut acc1 = $crate::simd_primitive!(neon, f32, zero);

                for w in 0..32 {
                    let word = qw[w];
                    let nibbles: [f32; 8] = [
                        ((word      ) & 0xF) as f32,
                        ((word >>  4) & 0xF) as f32,
                        ((word >>  8) & 0xF) as f32,
                        ((word >> 12) & 0xF) as f32,
                        ((word >> 16) & 0xF) as f32,
                        ((word >> 20) & 0xF) as f32,
                        ((word >> 24) & 0xF) as f32,
                        ((word >> 28) & 0xF) as f32,
                    ];
                    let vq0 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr());
                    let vq1 = $crate::simd_primitive!(neon, f32, load, nibbles.as_ptr().add(4));
                    let dq0 = $crate::simd_primitive!(neon, f32, fma, vd, vq0, v_neg_d_zero);
                    let dq1 = $crate::simd_primitive!(neon, f32, fma, vd, vq1, v_neg_d_zero);
                    let vo0 = $crate::simd_primitive!(neon, f32, load, other.add(w * 8));
                    let vo1 = $crate::simd_primitive!(neon, f32, load, other.add(w * 8 + 4));
                    acc0 = $crate::simd_primitive!(neon, f32, fma, dq0, vo0, acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, dq1, vo1, acc1);
                }

                let combined = $crate::simd_primitive!(neon, f32, add, acc0, acc1);
                $crate::simd_primitive!(neon, f32, reduce_sum, combined)
            }
        }
    };
}
