/// IQ-series quantization primitives (IQ1_S through IQ4_XS).
///
/// Sub-macro of the Layer 3 quant_primitive dispatcher.
/// Covers scalar + partial SIMD (avx2, neon).
#[macro_export]
macro_rules! quant_primitive_iq {
    (scalar, iq1_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;      // [u8; 32]
            let qh = &block.qh;      // [u16; 8]
            let scales = &block.scales; // [u8; 16]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 32 (8 groups total)
            for group in 0..8 {
                // Extract grid index from qs (simplified: use byte pairs as index)
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF; // 11 bits for 2048 entries
                let grid_val = crate::codebooks::IQ1S_GRID[idx];

                // Extract scale for this group
                let scale = scales[group * 2] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    // Apply sign from qh
                    let sign_bit = (qh[group] >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 32 + j) = group_d * (signed_v as f32); }
                }

                // Second half of the group (next 8 values)
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_bit = (qh[group] >> (j + 8)) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 32 + 8 + j) = group_d * (signed_v as f32); }
                }

                // Remaining 16 values (simplified: reuse grid pattern)
                for j in 0..16 {
                    let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 32 + 16 + j) = group_d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq1_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let qh = &block.qh;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..8 {
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                let grid_val = crate::codebooks::IQ1S_GRID[idx];
                let scale = scales[group * 2] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_bit = (qh[group] >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 32 + j); }
                }

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_bit = (qh[group] >> (j + 8)) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 32 + 8 + j); }
                }

                for j in 0..16 {
                    let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                    unsafe { sum += (group_d * (v as f32)) * *other.add(group * 32 + 16 + j); }
                }
            }
            sum
        }
    };

    // IQ1_M: 1-bit with IQ1S_GRID codebook (same grid as IQ1_S)
    (scalar, iq1_m, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let qs = &block.qs;      // [u8; 32]
            let qh = &block.qh;      // [u8; 16]
            let scales = &block.scales; // [u8; 8]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 32 (8 groups total)
            for group in 0..8 {
                // Extract grid index from qs
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                let grid_val = crate::codebooks::IQ1S_GRID[idx];

                // Extract scale for this group
                let scale = scales[group] as f32 / 255.0;

                // Unpack 8 int8 values from u64 grid entry (4 times for 32 values)
                for rep in 0..4 {
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        // Apply sign from qh
                        let qh_idx = group * 2 + rep / 2;
                        let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                        let signed_v = if sign_bit == 0 { v } else { -v };
                        unsafe { *out_ptr.add(group * 32 + rep * 8 + j) = scale * (signed_v as f32); }
                    }
                }
            }
        }
    };

    (scalar, iq1_m, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let qs = &block.qs;
            let qh = &block.qh;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..8 {
                let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                let grid_val = crate::codebooks::IQ1S_GRID[idx];
                let scale = scales[group] as f32 / 255.0;

                for rep in 0..4 {
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let qh_idx = group * 2 + rep / 2;
                        let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                        let signed_v = if sign_bit == 0 { v } else { -v };
                        unsafe { sum += (scale * (signed_v as f32)) * *other.add(group * 32 + rep * 8 + j); }
                    }
                }
            }
            sum
        }
    };

    // IQ2_XXS: 2-bit with IQ2XXS_GRID codebook
    (scalar, iq2_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u16; 32]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 8 (32 groups total)
            for group in 0..32 {
                // Extract grid index from qs (low 8 bits)
                let idx = (qs[group] & 0xFF) as usize;
                let grid_val = crate::codebooks::IQ2XXS_GRID[idx];

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 8 + j) = d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq2_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u16; 32]
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..32 {
                let idx = (qs[group] & 0xFF) as usize;
                let grid_val = crate::codebooks::IQ2XXS_GRID[idx];

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { sum += (d * (v as f32)) * *other.add(group * 8 + j); }
                }
            }
            sum
        }
    };

    // IQ2_XS: 2-bit with IQ2XS_GRID codebook
    (scalar, iq2_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u16; 32]
            let scales = &block.scales; // [u8; 8]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 8 (32 groups total)
            for group in 0..32 {
                // Extract grid index from qs (low 9 bits for 512 entries)
                let idx = (qs[group] & 0x1FF) as usize;
                let grid_val = crate::codebooks::IQ2XS_GRID[idx];

                // Extract scale for this group
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    // Apply sign from KSIGNS_IQ2XS
                    let sign_byte = crate::codebooks::KSIGNS_IQ2XS[((qs[group] >> 9) & 0x7F) as usize];
                    let sign_bit = (sign_byte >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 8 + j) = group_d * (signed_v as f32); }
                }
            }
        }
    };

    (scalar, iq2_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..32 {
                let idx = (qs[group] & 0x1FF) as usize;
                let grid_val = crate::codebooks::IQ2XS_GRID[idx];
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_byte = crate::codebooks::KSIGNS_IQ2XS[((qs[group] >> 9) & 0x7F) as usize];
                    let sign_bit = (sign_byte >> j) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 8 + j); }
                }
            }
            sum
        }
    };

    // IQ2_S: 2-bit with IQ2S_GRID codebook
    (scalar, iq2_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 64]
            let qh = &block.qh; // [u8; 8]
            let scales = &block.scales; // [u8; 8]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 8 (32 groups total)
            for group in 0..32 {
                // Extract grid index from qs (10 bits: 8 from qs + 2 from qh)
                let qs_idx = group * 2;
                let qh_idx = group / 4;
                let qh_shift = (group % 4) * 2;
                let idx = ((qs[qs_idx] as usize) |
                          ((qs[qs_idx + 1] as usize) << 8) |
                          (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF; // 10 bits for 1024 entries
                let grid_val = crate::codebooks::IQ2S_GRID[idx];

                // Extract scale for this group
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 8 int8 values from u64 grid entry
                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 8 + j) = group_d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq2_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 64]
            let qh = &block.qh;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..32 {
                let qs_idx = group * 2;
                let qh_idx = group / 4;
                let qh_shift = (group % 4) * 2;
                let idx = ((qs[qs_idx] as usize) |
                          ((qs[qs_idx + 1] as usize) << 8) |
                          (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF;
                let grid_val = crate::codebooks::IQ2S_GRID[idx];
                let scale_idx = group / 4;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..8 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { sum += (group_d * (v as f32)) * *other.add(group * 8 + j); }
                }
            }
            sum
        }
    };

    // IQ3_XXS: 3-bit with IQ3XXS_GRID codebook
    (scalar, iq3_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 96] = 3*QK_K/8
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 4 (64 groups total)
            // Each grid entry (u32) contains 4 packed int8 values
            for group in 0..64 {
                // Extract 8-bit index from qs
                let idx = qs[group] as usize;
                let grid_val = crate::codebooks::IQ3XXS_GRID[idx];

                // Unpack 4 int8 values from u32 grid entry
                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { *out_ptr.add(group * 4 + j) = d * (v as f32); }
                }
            }
        }
    };

    (scalar, iq3_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..64 {
                let idx = qs[group] as usize;
                let grid_val = crate::codebooks::IQ3XXS_GRID[idx];

                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    unsafe { sum += (d * (v as f32)) * *other.add(group * 4 + j); }
                }
            }
            sum
        }
    };

    // IQ3_S: 3-bit with IQ3S_GRID codebook
    (scalar, iq3_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs; // [u8; 64]
            let qh = &block.qh; // [u8; 8]
            let signs = &block.signs; // [u8; 32]
            let scales = &block.scales; // [u8; 4]
            let out_ptr = $out_ptr;

            // Process 256 values in groups of 4 (64 groups total)
            for group in 0..64 {
                // Extract 9-bit index from qs (8 bits) + qh (1 bit)
                let qh_idx = group / 8;
                let qh_shift = group % 8;
                let idx = ((qs[group] as usize) |
                          (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF; // 9 bits for 512 entries
                let grid_val = crate::codebooks::IQ3S_GRID[idx];

                // Extract scale for this group
                let scale_idx = group / 16;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                // Unpack 4 int8 values from u32 grid entry
                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    // Apply sign from signs array
                    let sign_idx = group / 2;
                    let sign_shift = (group % 2) * 4 + j;
                    let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { *out_ptr.add(group * 4 + j) = group_d * (signed_v as f32); }
                }
            }
        }
    };

    (scalar, iq3_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let qh = &block.qh;
            let signs = &block.signs;
            let scales = &block.scales;
            let other = $other_ptr;
            let mut sum = 0.0f32;

            for group in 0..64 {
                let qh_idx = group / 8;
                let qh_shift = group % 8;
                let idx = ((qs[group] as usize) |
                          (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF;
                let grid_val = crate::codebooks::IQ3S_GRID[idx];
                let scale_idx = group / 16;
                let scale = scales[scale_idx] as f32 / 255.0;
                let group_d = d * scale;

                for j in 0..4 {
                    let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                    let sign_idx = group / 2;
                    let sign_shift = (group % 2) * 4 + j;
                    let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                    let signed_v = if sign_bit == 0 { v } else { -v };
                    unsafe { sum += (group_d * (signed_v as f32)) * *other.add(group * 4 + j); }
                }
            }
            sum
        }
    };

    // IQ4_NL: 4-bit non-linear codebook lookup
    (scalar, iq4_nl, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let out_ptr = $out_ptr;
            // IQ4_NL block_size=32, qs has 16 bytes
            for i in 0..16 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    *out_ptr.add(i * 2) = d * (crate::codebooks::KVALUES_IQ4NL[lo] as f32);
                    *out_ptr.add(i * 2 + 1) = d * (crate::codebooks::KVALUES_IQ4NL[hi] as f32);
                }
            }
        }
    };

    (scalar, iq4_nl, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..16 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[lo] as f32) * *other.add(i * 2);
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[hi] as f32) * *other.add(i * 2 + 1);
                }
            }
            sum
        }
    };

    // IQ4_XS: uses same codebook as IQ4_NL but block_size=256
    (scalar, iq4_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let out_ptr = $out_ptr;
            for i in 0..128 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    *out_ptr.add(i * 2) = d * (crate::codebooks::KVALUES_IQ4NL[lo] as f32);
                    *out_ptr.add(i * 2 + 1) = d * (crate::codebooks::KVALUES_IQ4NL[hi] as f32);
                }
            }
        }
    };

    (scalar, iq4_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        {
            let block = unsafe { &*$block_ptr };
            let d: f32 = block.d.to_f32();
            let qs = &block.qs;
            let other = $other_ptr;
            let mut sum = 0.0f32;
            for i in 0..128 {
                let b = qs[i];
                let lo = (b & 0x0F) as usize;
                let hi = (b >> 4) as usize;
                unsafe {
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[lo] as f32) * *other.add(i * 2);
                    sum += (d * crate::codebooks::KVALUES_IQ4NL[hi] as f32) * *other.add(i * 2 + 1);
                }
            }
            sum
        }
    };

    // ========================================================================
    // NEON IQ dot implementations (scalar decode + NEON FMA accumulation)
    // ========================================================================

    (neon, iq1_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 32];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..16 {
                        let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                        vals[16 + j] = v as f32;
                    }
                    let base = group * 32;
                    for c in 0..8 {
                        let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(c * 4)));
                        let vo = $crate::simd_primitive!(neon, f32, load, other.add(base + c * 4));
                        acc = $crate::simd_primitive!(neon, f32, fma, vf, vo, acc);
                    }
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq1_m, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, scale);
                    let mut vals = [0.0f32; 32];
                    for rep in 0..4 {
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                            vals[rep * 8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                    }
                    let base = group * 32;
                    for c in 0..8 {
                        let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(c * 4)));
                        let vo = $crate::simd_primitive!(neon, f32, load, other.add(base + c * 4));
                        acc = $crate::simd_primitive!(neon, f32, fma, vf, vo, acc);
                    }
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq2_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                for group in 0..32 {
                    let idx = (qs[group] & 0xFF) as usize;
                    let grid_val = crate::codebooks::IQ2XXS_GRID[idx];
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, vf1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq2_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..32 {
                    let idx = (qs[group] & 0x1FF) as usize;
                    let grid_val = crate::codebooks::IQ2XS_GRID[idx.min(511)];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, vf1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq2_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..32 {
                    let qs_idx = group * 2;
                    let qh_idx = group / 4;
                    let qh_shift = (group % 4) * 2;
                    let idx = ((qs[qs_idx] as usize) |
                              ((qs[qs_idx + 1] as usize) << 8) |
                              (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF;
                    let grid_val = crate::codebooks::IQ2S_GRID[idx];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf0, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                    acc = $crate::simd_primitive!(neon, f32, fma, vf1, $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq3_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                for group in 0..64 {
                    let idx = qs[group] as usize;
                    let grid_val = crate::codebooks::IQ3XXS_GRID[idx];
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 4;
                    let vf = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq3_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);

                for group in 0..64 {
                    let qh_idx = group / 8;
                    let qh_shift = group % 8;
                    let idx = ((qs[group] as usize) |
                              (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF;
                    let grid_val = crate::codebooks::IQ3S_GRID[idx];
                    let scale_idx = group / 16;
                    let scale = scales[scale_idx] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_idx = group / 2;
                        let sign_shift = (group % 2) * 4 + j;
                        let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let base = group * 4;
                    let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    acc = $crate::simd_primitive!(neon, f32, fma, vf, $crate::simd_primitive!(neon, f32, load, other.add(base)), acc);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq4_nl, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                // Load 16-entry codebook into NEON register for vqtbl1q lookup
                let lut = vld1q_s8(crate::codebooks::KVALUES_IQ4NL.as_ptr());
                let mask_lo = vdupq_n_u8(0x0F);

                // Process 16 bytes (32 values) at a time — IQ4_NL block_size=32, qs has 16 bytes
                let raw = vld1q_u8(qs.as_ptr());
                let lo_idx = vandq_u8(raw, mask_lo);
                let hi_idx = vshrq_n_u8::<4>(raw);

                // Table lookup: 16 parallel codebook lookups each
                let lo_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(lo_idx)));
                let hi_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(hi_idx)));

                // Interleave lo/hi to get correct order: [lo0, hi0, lo1, hi1, ...]
                let interleaved_lo = vzip1q_s8(lo_vals, hi_vals);
                let interleaved_hi = vzip2q_s8(lo_vals, hi_vals);

                // Convert i8 → f32 and FMA, 4 elements at a time
                // First 16 values (interleaved_lo)
                let lo8 = vget_low_s8(interleaved_lo);
                let hi8 = vget_high_s8(interleaved_lo);
                let lo16_0 = vmovl_s8(lo8);
                let lo16_1 = vmovl_s8(hi8);

                // 4 groups of 4 from interleaved_lo
                let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0)));
                let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0)));
                let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1)));
                let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1)));

                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f0), $crate::simd_primitive!(neon, f32, load, other), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f1), $crate::simd_primitive!(neon, f32, load, other.add(4)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f2), $crate::simd_primitive!(neon, f32, load, other.add(8)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f3), $crate::simd_primitive!(neon, f32, load, other.add(12)), acc);

                // Next 16 values (interleaved_hi)
                let lo8b = vget_low_s8(interleaved_hi);
                let hi8b = vget_high_s8(interleaved_hi);
                let lo16_2 = vmovl_s8(lo8b);
                let lo16_3 = vmovl_s8(hi8b);

                let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_2)));
                let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_2)));
                let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_3)));
                let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_3)));

                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f4), $crate::simd_primitive!(neon, f32, load, other.add(16)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f5), $crate::simd_primitive!(neon, f32, load, other.add(20)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f6), $crate::simd_primitive!(neon, f32, load, other.add(24)), acc);
                acc = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f7), $crate::simd_primitive!(neon, f32, load, other.add(28)), acc);

                $crate::simd_primitive!(neon, f32, reduce_sum, acc)
            }
        }
    };

    (neon, iq4_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc0 = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let mut acc1 = $crate::simd_primitive!(neon, f32, splat, 0.0);
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                // Load 16-entry codebook into NEON register
                let lut = vld1q_s8(crate::codebooks::KVALUES_IQ4NL.as_ptr());
                let mask_lo = vdupq_n_u8(0x0F);

                // IQ4_XS block_size=256, qs has 128 bytes → 16 groups of 8 bytes
                for i in 0..8 {
                    // Load 16 packed bytes → 32 values
                    let raw = vld1q_u8(qs.as_ptr().add(i * 16));
                    let lo_idx = vandq_u8(raw, mask_lo);
                    let hi_idx = vshrq_n_u8::<4>(raw);

                    let lo_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(lo_idx)));
                    let hi_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(hi_idx)));

                    let interleaved_lo = vzip1q_s8(lo_vals, hi_vals);
                    let interleaved_hi = vzip2q_s8(lo_vals, hi_vals);

                    let base = i * 32;

                    // interleaved_lo → 16 f32 values
                    let lo8 = vget_low_s8(interleaved_lo);
                    let hi8 = vget_high_s8(interleaved_lo);
                    let s16_0 = vmovl_s8(lo8);
                    let s16_1 = vmovl_s8(hi8);

                    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_0)));
                    let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_0)));
                    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_1)));
                    let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_1)));

                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f0), $crate::simd_primitive!(neon, f32, load, other.add(base)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f1), $crate::simd_primitive!(neon, f32, load, other.add(base + 4)), acc1);
                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f2), $crate::simd_primitive!(neon, f32, load, other.add(base + 8)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f3), $crate::simd_primitive!(neon, f32, load, other.add(base + 12)), acc1);

                    // interleaved_hi → 16 f32 values
                    let lo8b = vget_low_s8(interleaved_hi);
                    let hi8b = vget_high_s8(interleaved_hi);
                    let s16_2 = vmovl_s8(lo8b);
                    let s16_3 = vmovl_s8(hi8b);

                    let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_2)));
                    let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_2)));
                    let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_3)));
                    let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_3)));

                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f4), $crate::simd_primitive!(neon, f32, load, other.add(base + 16)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f5), $crate::simd_primitive!(neon, f32, load, other.add(base + 20)), acc1);
                    acc0 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f6), $crate::simd_primitive!(neon, f32, load, other.add(base + 24)), acc0);
                    acc1 = $crate::simd_primitive!(neon, f32, fma, $crate::simd_primitive!(neon, f32, mul, vd, f7), $crate::simd_primitive!(neon, f32, load, other.add(base + 28)), acc1);
                }
                $crate::simd_primitive!(neon, f32, reduce_sum, $crate::simd_primitive!(neon, f32, add, acc0, acc1))
            }
        }
    };

    // ========================================================================
    // NEON IQ1-IQ3 decode
    // ========================================================================

    (neon, iq1_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;

                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 32];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..16 {
                        let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                        vals[16 + j] = v as f32;
                    }
                    let base = group * 32;
                    for c in 0..8 {
                        let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(c * 4)));
                        $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + c * 4), vf);
                    }
                }
            }
        }
    };

    (neon, iq1_m, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;

                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, scale);
                    let mut vals = [0.0f32; 32];
                    for rep in 0..4 {
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                            vals[rep * 8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                    }
                    let base = group * 32;
                    for c in 0..8 {
                        let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(c * 4)));
                        $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + c * 4), vf);
                    }
                }
            }
        }
    };

    (neon, iq2_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                for group in 0..32 {
                    let idx = (qs[group] & 0xFF) as usize;
                    let grid_val = crate::codebooks::IQ2XXS_GRID[idx];
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), vf0);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 4), vf1);
                }
            }
        }
    };

    (neon, iq2_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let out_ptr = $out_ptr;

                for group in 0..32 {
                    let idx = (qs[group] & 0x1FF) as usize;
                    let grid_val = crate::codebooks::IQ2XS_GRID[idx.min(511)];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), vf0);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 4), vf1);
                }
            }
        }
    };

    (neon, iq2_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;

                for group in 0..32 {
                    let qs_idx = group * 2;
                    let qh_idx = group / 4;
                    let qh_shift = (group % 4) * 2;
                    let idx = ((qs[qs_idx] as usize) |
                              ((qs[qs_idx + 1] as usize) << 8) |
                              (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF;
                    let grid_val = crate::codebooks::IQ2S_GRID[idx];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 8;
                    let vf0 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    let vf1 = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr().add(4)));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), vf0);
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 4), vf1);
                }
            }
        }
    };

    (neon, iq3_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                for group in 0..64 {
                    let idx = qs[group] as usize;
                    let grid_val = crate::codebooks::IQ3XXS_GRID[idx];
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let base = group * 4;
                    let vf = $crate::simd_primitive!(neon, f32, mul, vd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), vf);
                }
            }
        }
    };

    (neon, iq3_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let out_ptr = $out_ptr;

                for group in 0..64 {
                    let qh_idx = group / 8;
                    let qh_shift = group % 8;
                    let idx = ((qs[group] as usize) |
                              (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF;
                    let grid_val = crate::codebooks::IQ3S_GRID[idx];
                    let scale_idx = group / 16;
                    let scale = scales[scale_idx] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(neon, f32, splat, group_d);
                    let mut vals = [0.0f32; 4];
                    for j in 0..4 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_idx = group / 2;
                        let sign_shift = (group % 2) * 4 + j;
                        let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let base = group * 4;
                    let vf = $crate::simd_primitive!(neon, f32, mul, vgd, $crate::simd_primitive!(neon, f32, load, vals.as_ptr()));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), vf);
                }
            }
        }
    };

    // ========================================================================
    // AVX2 IQ4_NL decode/dot (16-entry codebook, vpshufb lookup)
    // ========================================================================
    (avx2, iq4_nl, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr();
                let out_ptr = $out_ptr;
                let mask_lo = _mm_set1_epi8(0x0F);
                // IQ4_NL block_size=32: 16 bytes of qs → 32 values
                // Process the single 16-byte chunk
                for i in 0..1 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let lo_bytes = _mm_and_si128(v128, mask_lo);
                    let hi_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask_lo);
                    // Convert nibble indices to codebook values via scalar lookup
                    // (vpshufb only works for byte->byte, we need float output)
                    for chunk in 0..2 {
                        let bytes = if chunk == 0 { lo_bytes } else { hi_bytes };
                        // Extract 16 bytes, lookup codebook, convert to f32
                        let mut indices = [0u8; 16];
                        _mm_storeu_si128(indices.as_mut_ptr() as *mut _, bytes);
                        for g in 0..2 {
                            let mut vals = [0.0f32; 8];
                            for j in 0..8 {
                                vals[j] = crate::codebooks::KVALUES_IQ4NL[indices[g * 8 + j] as usize] as f32;
                            }
                            let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                            let res = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                            let out_idx = i * 32 + chunk * 16 + g * 8;
                            $crate::simd_primitive!(avx2, f32, store, out_ptr.add(out_idx), res);
                        }
                    }
                }
            }
        }
    };

    (avx2, iq4_nl, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs_ptr = block.qs.as_ptr();
                let other = $other_ptr;
                let mask_lo = _mm_set1_epi8(0x0F);
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // IQ4_NL block_size=32: 16 bytes of qs → 32 values
                for i in 0..1 {
                    let v128 = _mm_loadu_si128(qs_ptr.add(i * 16) as *const _);
                    let lo_bytes = _mm_and_si128(v128, mask_lo);
                    let hi_bytes = _mm_and_si128(_mm_srli_epi16(v128, 4), mask_lo);
                    for chunk in 0..2 {
                        let bytes = if chunk == 0 { lo_bytes } else { hi_bytes };
                        let mut indices = [0u8; 16];
                        _mm_storeu_si128(indices.as_mut_ptr() as *mut _, bytes);
                        for g in 0..2 {
                            let mut vals = [0.0f32; 8];
                            for j in 0..8 {
                                vals[j] = crate::codebooks::KVALUES_IQ4NL[indices[g * 8 + j] as usize] as f32;
                            }
                            let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                            let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                            let out_idx = i * 32 + chunk * 16 + g * 8;
                            let vo = $crate::simd_primitive!(avx2, f32, load, other.add(out_idx));
                            acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                        }
                    }
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    // ========================================================================
    // AVX2 IQ4_XS decode/dot (same codebook as IQ4_NL, block_size=256)
    // ========================================================================
    (avx2, iq4_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                // 128 bytes qs, 2 nibbles each = 256 values
                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[(b & 0x0F) as usize] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[(b >> 4) as usize] as f32;
                    }
                    let v0 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let v1 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr().add(8));
                    let r0 = $crate::simd_primitive!(avx2, f32, mul, vd, v0);
                    let r1 = $crate::simd_primitive!(avx2, f32, mul, vd, v1);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16), r0);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(i * 16 + 8), r1);
                }
            }
        }
    };

    (avx2, iq4_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[(b & 0x0F) as usize] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[(b >> 4) as usize] as f32;
                    }
                    let v0 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let v1 = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr().add(8));
                    let dq0 = $crate::simd_primitive!(avx2, f32, mul, vd, v0);
                    let dq1 = $crate::simd_primitive!(avx2, f32, mul, vd, v1);
                    let o0 = $crate::simd_primitive!(avx2, f32, load, other.add(i * 16));
                    let o1 = $crate::simd_primitive!(avx2, f32, load, other.add(i * 16 + 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq0, o0, acc);
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq1, o1, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    // ========================================================================
    // AVX2 IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S dot (vpgatherdd codebook)
    // ========================================================================
    // For grid-based IQ formats, the bit extraction is complex but the FMA
    // accumulation benefits from AVX2. We extract values to a temp buffer
    // then do vectorized FMA.

    (avx2, iq2_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // 32 groups of 8 values
                for group in 0..32 {
                    let idx = (qs[group] & 0xFF) as usize;
                    let grid_val = crate::codebooks::IQ2XXS_GRID[idx];
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq2_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for group in 0..32 {
                    let idx = (qs[group] & 0x1FF) as usize;
                    let grid_val = crate::codebooks::IQ2XS_GRID[idx.min(511)];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq3_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // IQ3_S: 64 groups of 4 values, codebook lookup via IQ3S_GRID
                for pair in 0..32 {
                    let mut vals = [0.0f32; 8];
                    for half in 0..2 {
                        let group = pair * 2 + half;
                        let qh_idx = group / 8;
                        let qh_shift = group % 8;
                        let idx = ((qs[group] as usize) |
                                  (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF;
                        let grid_val = crate::codebooks::IQ3S_GRID[idx];
                        let scale_idx = group / 16;
                        let scale = scales[scale_idx] as f32 / 255.0;
                        let group_d = d * scale;

                        for j in 0..4 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let sign_idx = group / 2;
                            let sign_shift = (group % 2) * 4 + j;
                            let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                            let signed_v = if sign_bit == 0 { v } else { -v };
                            vals[half * 4 + j] = group_d * (signed_v as f32);
                        }
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(pair * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, vf, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq3_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // IQ3_XXS: 64 groups of 4 values, codebook lookup via IQ3XXS_GRID
                for pair in 0..32 {
                    let mut vals = [0.0f32; 8];
                    for half in 0..2 {
                        let group = pair * 2 + half;
                        let idx = qs[group] as usize;
                        let grid_val = crate::codebooks::IQ3XXS_GRID[idx];
                        for j in 0..4 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            vals[half * 4 + j] = v as f32;
                        }
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(pair * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq2_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                // 256 values in 32 groups of 8 (matching scalar logic)
                for group in 0..32 {
                    let qs_idx = group * 2;
                    let qh_idx = group / 4;
                    let qh_shift = (group % 4) * 2;
                    let idx = ((qs[qs_idx] as usize) |
                              ((qs[qs_idx + 1] as usize) << 8) |
                              (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF;
                    let grid_val = crate::codebooks::IQ2S_GRID[idx];
                    let scale_idx = group / 4;
                    let scale = scales[scale_idx] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);

                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        vals[j] = v as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    (avx2, iq1_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    // First 8 values
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                    // Second 8 values
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32 + 8));
                    acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                    // Remaining 16 values (2 chunks of 8)
                    for chunk in 0..2 {
                        for j in 0..8 {
                            let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                            vals[j] = v as f32;
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32 + 16 + chunk * 8));
                        acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                    }
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    // ========================================================================
    // AVX-512 IQ4_NL decode/dot
    // ========================================================================

    (avx512, iq4_nl, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;

                // IQ4_NL block_size=32, qs has 16 bytes → 32 values
                // Process 8 bytes at a time → 16 low nibbles + 16 high nibbles
                // Iteration 0: low nibbles of all 16 bytes → 16 values → 1x __m512
                // Iteration 1: high nibbles of all 16 bytes → 16 values → 1x __m512
                // But values must be interleaved: [lo0, hi0, lo1, hi1, ...]
                // So instead: process 8 bytes → 8 low + 8 high = 16 values → 1x __m512
                for i in 0..2 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        let lo = (b & 0x0F) as usize;
                        let hi = (b >> 4) as usize;
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[lo] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[hi] as f32;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i * 16), res);
                }
            }
        }
    };

    (avx512, iq4_nl, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                // IQ4_NL block_size=32, qs has 16 bytes → 32 values
                // Process 8 bytes → 16 interleaved values per __m512
                for i in 0..2 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        let lo = (b & 0x0F) as usize;
                        let hi = (b >> 4) as usize;
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[lo] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[hi] as f32;
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
    // AVX-512 IQ4_XS decode/dot
    // ========================================================================

    (avx512, iq4_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;

                // IQ4_XS block_size=256, qs has 128 bytes → 256 values
                // Process 8 bytes per iteration → 16 interleaved values → 1x __m512
                // 128 / 8 = 16 iterations
                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        let lo = (b & 0x0F) as usize;
                        let hi = (b >> 4) as usize;
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[lo] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[hi] as f32;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(i * 16), res);
                }
            }
        }
    };

    (avx512, iq4_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                // IQ4_XS block_size=256, qs has 128 bytes → 256 values
                // Process 8 bytes per iteration → 16 interleaved values → 1x __m512
                // 128 / 8 = 16 iterations
                for i in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let b = qs[i * 8 + j];
                        let lo = (b & 0x0F) as usize;
                        let hi = (b >> 4) as usize;
                        vals[j * 2] = crate::codebooks::KVALUES_IQ4NL[lo] as f32;
                        vals[j * 2 + 1] = crate::codebooks::KVALUES_IQ4NL[hi] as f32;
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
    // AVX-512 IQ1/IQ2 decode/dot
    // ========================================================================

    // --- IQ1_S decode: 8 groups of 32 values, per-group scale ---
    // Pair (first8, second8) and (chunk0, chunk1) → 2 pairs/group × 8 groups = 16 iters
    (avx512, iq1_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);

                    // Pair 0: first 8 + second 8 → 16 values
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(group * 32), res);

                    // Pair 1: remaining 16 values (2 chunks of 8)
                    let mut vals2 = [0.0f32; 16];
                    for chunk in 0..2 {
                        for j in 0..8 {
                            let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                            vals2[chunk * 8 + j] = v as f32;
                        }
                    }
                    let vf2 = $crate::simd_primitive!(avx512, f32, load, vals2.as_ptr());
                    let res2 = $crate::simd_primitive!(avx512, f32, mul, vgd, vf2);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(group * 32 + 16), res2);
                }
            }
        }
    };

    // --- IQ1_S dot: same structure, FMA + reduce_sum ---
    (avx512, iq1_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);

                    // Pair 0: first 8 + second 8 → 16 values
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(group * 32));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);

                    // Pair 1: remaining 16 values (2 chunks of 8)
                    let mut vals2 = [0.0f32; 16];
                    for chunk in 0..2 {
                        for j in 0..8 {
                            let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                            vals2[chunk * 8 + j] = v as f32;
                        }
                    }
                    let vf2 = $crate::simd_primitive!(avx512, f32, load, vals2.as_ptr());
                    let dq2 = $crate::simd_primitive!(avx512, f32, mul, vgd, vf2);
                    let vo2 = $crate::simd_primitive!(avx512, f32, load, other.add(group * 32 + 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq2, vo2, acc);
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // --- IQ1_M decode: 8 groups of 32 = 4 reps of 8, per-group scale ---
    // Pair reps (0,1) and (2,3) → 2 pairs/group × 8 groups = 16 iters
    (avx512, iq1_m, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, scale);

                    // Pair reps (0,1) and (2,3) → 16 values each
                    for pair in 0..2 {
                        let rep0 = pair * 2;
                        let rep1 = pair * 2 + 1;
                        let mut vals = [0.0f32; 16];
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep0 / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep0 % 2) * 4 + j / 2)) & 1;
                            vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep1 / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep1 % 2) * 4 + j / 2)) & 1;
                            vals[8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                        let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                        let res = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                        $crate::simd_primitive!(avx512, f32, store, out_ptr.add(group * 32 + pair * 16), res);
                    }
                }
            }
        }
    };

    // --- IQ1_M dot: same structure, FMA + reduce_sum ---
    (avx512, iq1_m, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, scale);

                    for pair in 0..2 {
                        let rep0 = pair * 2;
                        let rep1 = pair * 2 + 1;
                        let mut vals = [0.0f32; 16];
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep0 / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep0 % 2) * 4 + j / 2)) & 1;
                            vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep1 / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep1 % 2) * 4 + j / 2)) & 1;
                            vals[8 + j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                        let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx512, f32, load, other.add(group * 32 + pair * 16));
                        acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                    }
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // --- IQ2_XXS decode: 32 groups of 8, global d ---
    // Pair adjacent groups → 16 iterations of 16 values
    (avx512, iq2_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                // 32 groups of 8 → 16 pairs of 16
                for pair in 0..16 {
                    let g0 = pair * 2;
                    let g1 = pair * 2 + 1;
                    let idx0 = (qs[g0] & 0xFF) as usize;
                    let idx1 = (qs[g1] & 0xFF) as usize;
                    let grid0 = crate::codebooks::IQ2XXS_GRID[idx0];
                    let grid1 = crate::codebooks::IQ2XXS_GRID[idx1];
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        vals[j] = ((grid0 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    for j in 0..8 {
                        vals[8 + j] = ((grid1 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(pair * 16), res);
                }
            }
        }
    };

    // --- IQ2_XXS dot: same pairing, FMA + reduce_sum ---
    (avx512, iq2_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                // 32 groups of 8 → 16 pairs of 16
                for pair in 0..16 {
                    let g0 = pair * 2;
                    let g1 = pair * 2 + 1;
                    let idx0 = (qs[g0] & 0xFF) as usize;
                    let idx1 = (qs[g1] & 0xFF) as usize;
                    let grid0 = crate::codebooks::IQ2XXS_GRID[idx0];
                    let grid1 = crate::codebooks::IQ2XXS_GRID[idx1];
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        vals[j] = ((grid0 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    for j in 0..8 {
                        vals[8 + j] = ((grid1 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(pair * 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // --- IQ2_XS decode: 32 groups of 8, per-group-of-4 scales ---
    // Adjacent pairs (2k, 2k+1) share scales[k/2], so pairing is safe
    (avx512, iq2_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                // 32 groups of 8 → 16 pairs of 16
                for pair in 0..16 {
                    let g0 = pair * 2;
                    let g1 = pair * 2 + 1;
                    // g0/4 == g1/4 since g0 and g1 are consecutive even/odd
                    let scale = scales[g0 / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);
                    let idx0 = (qs[g0] & 0x1FF) as usize;
                    let idx1 = (qs[g1] & 0x1FF) as usize;
                    let grid0 = crate::codebooks::IQ2XS_GRID[idx0.min(511)];
                    let grid1 = crate::codebooks::IQ2XS_GRID[idx1.min(511)];
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        vals[j] = ((grid0 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    for j in 0..8 {
                        vals[8 + j] = ((grid1 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(pair * 16), res);
                }
            }
        }
    };

    // --- IQ2_XS dot: same pairing, FMA + reduce_sum ---
    (avx512, iq2_xs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                // 32 groups of 8 → 16 pairs of 16
                for pair in 0..16 {
                    let g0 = pair * 2;
                    let g1 = pair * 2 + 1;
                    let scale = scales[g0 / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);
                    let idx0 = (qs[g0] & 0x1FF) as usize;
                    let idx1 = (qs[g1] & 0x1FF) as usize;
                    let grid0 = crate::codebooks::IQ2XS_GRID[idx0.min(511)];
                    let grid1 = crate::codebooks::IQ2XS_GRID[idx1.min(511)];
                    let mut vals = [0.0f32; 16];
                    for j in 0..8 {
                        vals[j] = ((grid0 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    for j in 0..8 {
                        vals[8 + j] = ((grid1 >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(pair * 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // --- IQ2_S decode: 8 groups of 32 (4 chunks of 8), per-group scale ---
    // Pair chunks (0,1) and (2,3) → 2 pairs/group × 8 groups = 16 iters
    (avx512, iq2_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                for group in 0..8 {
                    let scale = scales[group] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);
                    // 2 pairs of chunks: (0,1) and (2,3)
                    for cp in 0..2 {
                        let c0 = cp * 2;
                        let c1 = cp * 2 + 1;
                        let mut vals = [0.0f32; 16];
                        for j in 0..8 {
                            let idx = group * 32 + c0 * 8 + j;
                            let q = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                            let h = (qh[idx / 8] >> (idx % 8)) & 1;
                            vals[j] = ((h << 2) | q) as f32 - 4.0;
                        }
                        for j in 0..8 {
                            let idx = group * 32 + c1 * 8 + j;
                            let q = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                            let h = (qh[idx / 8] >> (idx % 8)) & 1;
                            vals[8 + j] = ((h << 2) | q) as f32 - 4.0;
                        }
                        let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                        let res = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                        $crate::simd_primitive!(avx512, f32, store, out_ptr.add(group * 32 + cp * 16), res);
                    }
                }
            }
        }
    };

    // --- IQ2_S dot: same chunk pairing, FMA + reduce_sum ---
    (avx512, iq2_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);
                for group in 0..8 {
                    let scale = scales[group] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);
                    for cp in 0..2 {
                        let c0 = cp * 2;
                        let c1 = cp * 2 + 1;
                        let mut vals = [0.0f32; 16];
                        for j in 0..8 {
                            let idx = group * 32 + c0 * 8 + j;
                            let q = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                            let h = (qh[idx / 8] >> (idx % 8)) & 1;
                            vals[j] = ((h << 2) | q) as f32 - 4.0;
                        }
                        for j in 0..8 {
                            let idx = group * 32 + c1 * 8 + j;
                            let q = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                            let h = (qh[idx / 8] >> (idx % 8)) & 1;
                            vals[8 + j] = ((h << 2) | q) as f32 - 4.0;
                        }
                        let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx512, f32, load, other.add(group * 32 + cp * 16));
                        acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                    }
                }
                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    (avx2, iq1_m, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx2, f32, zero);
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, scale);
                    for rep in 0..4 {
                        let mut vals = [0.0f32; 8];
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                            vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        let vo = $crate::simd_primitive!(avx2, f32, load, other.add(group * 32 + rep * 8));
                        acc = $crate::simd_primitive!(avx2, f32, fma, dq, vo, acc);
                    }
                }
                let t1 = _mm256_hadd_ps(acc, acc);
                let t2 = _mm256_hadd_ps(t1, t1);
                let hi128 = _mm256_extractf128_ps(t2, 1);
                let lo128 = _mm256_castps256_ps128(t2);
                _mm_cvtss_f32(_mm_add_ps(hi128, lo128))
            }
        }
    };

    // ========================================================================
    // AVX2 IQ decode (extracted from dot pattern logic)
    // ========================================================================

    (avx2, iq1_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group * 2] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    // First 8 values
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> j) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(group * 32), res);
                    // Second 8 values
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        let sign_bit = (qh[group] >> (j + 8)) & 1;
                        vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(group * 32 + 8), res);
                    // Remaining 16 values (2 chunks of 8)
                    for chunk in 0..2 {
                        for j in 0..8 {
                            let v = ((grid_val >> ((j % 8) * 8)) & 0xFF) as i8;
                            vals[j] = v as f32;
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let res = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        $crate::simd_primitive!(avx2, f32, store, out_ptr.add(group * 32 + 16 + chunk * 8), res);
                    }
                }
            }
        }
    };

    (avx2, iq1_m, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                for group in 0..8 {
                    let idx = ((qs[group * 4] as usize) | ((qs[group * 4 + 1] as usize) << 8)) & 0x7FF;
                    let grid_val = crate::codebooks::IQ1S_GRID[idx];
                    let scale = scales[group] as f32 / 255.0;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, scale);
                    for rep in 0..4 {
                        let mut vals = [0.0f32; 8];
                        for j in 0..8 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let qh_idx = group * 2 + rep / 2;
                            let sign_bit = (qh[qh_idx] >> ((rep % 2) * 4 + j / 2)) & 1;
                            vals[j] = if sign_bit == 0 { v as f32 } else { -(v as f32) };
                        }
                        let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                        let res = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                        $crate::simd_primitive!(avx2, f32, store, out_ptr.add(group * 32 + rep * 8), res);
                    }
                }
            }
        }
    };

    (avx2, iq2_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                // 32 groups of 8 values
                for group in 0..32 {
                    let idx = (qs[group] & 0xFF) as usize;
                    let grid_val = crate::codebooks::IQ2XXS_GRID[idx];
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(group * 8), res);
                }
            }
        }
    };

    (avx2, iq2_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                for group in 0..32 {
                    let idx = (qs[group] & 0x1FF) as usize;
                    let grid_val = crate::codebooks::IQ2XS_GRID[idx.min(511)];
                    let scale = scales[group / 4] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        vals[j] = ((grid_val >> (j * 8)) & 0xFF) as i8 as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(group * 8), res);
                }
            }
        }
    };

    (avx2, iq2_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                // 256 values in 32 groups of 8 (matching scalar logic)
                for group in 0..32 {
                    // Extract 10-bit grid index from qs + qh
                    let qs_idx = group * 2;
                    let qh_idx = group / 4;
                    let qh_shift = (group % 4) * 2;
                    let idx = ((qs[qs_idx] as usize) |
                              ((qs[qs_idx + 1] as usize) << 8) |
                              (((qh[qh_idx] >> qh_shift) & 0x03) as usize) << 8) & 0x3FF;
                    let grid_val = crate::codebooks::IQ2S_GRID[idx];

                    let scale_idx = group / 4;
                    let scale = scales[scale_idx] as f32 / 255.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx2, f32, splat, group_d);

                    // Unpack 8 int8 values from u64 grid entry
                    let mut vals = [0.0f32; 8];
                    for j in 0..8 {
                        let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                        vals[j] = v as f32;
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx2, f32, mul, vgd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(group * 8), res);
                }
            }
        }
    };

    (avx2, iq3_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx2, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                // IQ3_XXS: 64 groups of 4 values, codebook lookup via IQ3XXS_GRID
                // Process 2 groups (8 values) per iteration for AVX2
                for pair in 0..32 {
                    let mut vals = [0.0f32; 8];
                    for half in 0..2 {
                        let group = pair * 2 + half;
                        let idx = qs[group] as usize;
                        let grid_val = crate::codebooks::IQ3XXS_GRID[idx];
                        for j in 0..4 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            vals[half * 4 + j] = v as f32;
                        }
                    }
                    let vf = $crate::simd_primitive!(avx2, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx2, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(pair * 8), res);
                }
            }
        }
    };

    (avx2, iq3_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {

                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let out_ptr = $out_ptr;
                // IQ3_S: 64 groups of 4 values, codebook lookup via IQ3S_GRID
                // Process 2 groups (8 values) per iteration for AVX2
                for pair in 0..32 {
                    let mut vals = [0.0f32; 8];
                    for half in 0..2 {
                        let group = pair * 2 + half;
                        // Extract 9-bit index from qs (8 bits) + qh (1 bit)
                        let qh_idx = group / 8;
                        let qh_shift = group % 8;
                        let idx = ((qs[group] as usize) |
                                  (((qh[qh_idx] >> qh_shift) & 1) as usize) << 8) & 0x1FF;
                        let grid_val = crate::codebooks::IQ3S_GRID[idx];

                        let scale_idx = group / 16;
                        let scale = scales[scale_idx] as f32 / 255.0;
                        let group_d = d * scale;

                        for j in 0..4 {
                            let v = ((grid_val >> (j * 8)) & 0xFF) as i8;
                            let sign_idx = group / 2;
                            let sign_shift = (group % 2) * 4 + j;
                            let sign_bit = (signs[sign_idx] >> sign_shift) & 1;
                            let signed_v = if sign_bit == 0 { v } else { -v };
                            vals[half * 4 + j] = group_d * (signed_v as f32);
                        }
                    }
                    // Note: vals already scaled per-group, so just store directly
                    let mut out_vals = [0.0f32; 8];
                    out_vals.copy_from_slice(&vals);
                    let vf = $crate::simd_primitive!(avx2, f32, load, out_vals.as_ptr());
                    $crate::simd_primitive!(avx2, f32, store, out_ptr.add(pair * 8), vf);
                }
            }
        }
    };

    // ========================================================================
    // AVX-512 IQ3 decode/dot
    // ========================================================================

    (avx512, iq3_xxs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let out_ptr = $out_ptr;

                // IQ3_XXS: 3-bit packed in qs, 256 values, global d
                // 32 groups of 8 → pair adjacent groups → 16 iterations of 16 values
                for pair in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for half in 0..2 {
                        let group = pair * 2 + half;
                        let base = group * 8;
                        for j in 0..8 {
                            let idx = base + j;
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
                            vals[half * 8 + j] = q as f32 - 4.0;
                        }
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let res = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    $crate::simd_primitive!(avx512, f32, store, out_ptr.add(pair * 16), res);
                }
            }
        }
    };

    (avx512, iq3_xxs, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let vd = $crate::simd_primitive!(avx512, f32, splat, d);
                let qs = &block.qs;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                // IQ3_XXS: 3-bit packed in qs, 256 values, global d
                // 32 groups of 8 → pair adjacent groups → 16 iterations of 16 values
                for pair in 0..16 {
                    let mut vals = [0.0f32; 16];
                    for half in 0..2 {
                        let group = pair * 2 + half;
                        let base = group * 8;
                        for j in 0..8 {
                            let idx = base + j;
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
                            vals[half * 8 + j] = q as f32 - 4.0;
                        }
                    }
                    let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                    let dq = $crate::simd_primitive!(avx512, f32, mul, vd, vf);
                    let vo = $crate::simd_primitive!(avx512, f32, load, other.add(pair * 16));
                    acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    (avx512, iq3_s, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let out_ptr = $out_ptr;

                // IQ3_S: 256 values in 8 groups of 32, 4 chunks of 8 per group
                // Pair adjacent chunks within each group → 8 groups × 2 pairs = 16 iterations
                for group in 0..8 {
                    let scale = scales[group / 2] as f32 / 15.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);

                    for pair in 0..2 {
                        let mut vals = [0.0f32; 16];
                        for half in 0..2 {
                            let chunk = pair * 2 + half;
                            let base = group * 32 + chunk * 8;
                            for j in 0..8 {
                                let idx = base + j;
                                let low = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                                let high = (qh[idx / 8] >> (idx % 8)) & 1;
                                let val = (high << 2) | low;
                                let sign_byte = signs[idx / 8];
                                let sign_bit = (sign_byte >> (idx % 8)) & 1;
                                let signed_val = if sign_bit == 0 { val as f32 } else { -(val as f32) };
                                vals[half * 8 + j] = signed_val;
                            }
                        }
                        let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                        let res = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                        let out_offset = group * 32 + pair * 16;
                        $crate::simd_primitive!(avx512, f32, store, out_ptr.add(out_offset), res);
                    }
                }
            }
        }
    };

    (avx512, iq3_s, dot, $block_ptr:expr, $other_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let qh = &block.qh;
                let signs = &block.signs;
                let scales = &block.scales;
                let other = $other_ptr;
                let mut acc = $crate::simd_primitive!(avx512, f32, zero);

                // IQ3_S: 256 values in 8 groups of 32, 4 chunks of 8 per group
                // Pair adjacent chunks within each group → 8 groups × 2 pairs = 16 iterations
                for group in 0..8 {
                    let scale = scales[group / 2] as f32 / 15.0;
                    let group_d = d * scale;
                    let vgd = $crate::simd_primitive!(avx512, f32, splat, group_d);

                    for pair in 0..2 {
                        let mut vals = [0.0f32; 16];
                        for half in 0..2 {
                            let chunk = pair * 2 + half;
                            let base = group * 32 + chunk * 8;
                            for j in 0..8 {
                                let idx = base + j;
                                let low = (qs[idx / 4] >> ((idx % 4) * 2)) & 3;
                                let high = (qh[idx / 8] >> (idx % 8)) & 1;
                                let val = (high << 2) | low;
                                let sign_byte = signs[idx / 8];
                                let sign_bit = (sign_byte >> (idx % 8)) & 1;
                                let signed_val = if sign_bit == 0 { val as f32 } else { -(val as f32) };
                                vals[half * 8 + j] = signed_val;
                            }
                        }
                        let vf = $crate::simd_primitive!(avx512, f32, load, vals.as_ptr());
                        let dq = $crate::simd_primitive!(avx512, f32, mul, vgd, vf);
                        let other_offset = group * 32 + pair * 16;
                        let vo = $crate::simd_primitive!(avx512, f32, load, other.add(other_offset));
                        acc = $crate::simd_primitive!(avx512, f32, fma, dq, vo, acc);
                    }
                }

                $crate::simd_primitive!(avx512, f32, reduce_sum, acc)
            }
        }
    };

    // ========================================================================
    // NEON IQ4 decode
    // ========================================================================

    (neon, iq4_nl, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                // Load 16-entry codebook into NEON register for vqtbl1q lookup
                let lut = vld1q_s8(crate::codebooks::KVALUES_IQ4NL.as_ptr());
                let mask_lo = vdupq_n_u8(0x0F);

                // IQ4_NL block_size=32, qs has 16 bytes → 32 values in one shot
                let raw = vld1q_u8(qs.as_ptr());
                let lo_idx = vandq_u8(raw, mask_lo);
                let hi_idx = vshrq_n_u8::<4>(raw);

                // Table lookup: 16 parallel codebook lookups each
                let lo_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(lo_idx)));
                let hi_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(hi_idx)));

                // Interleave lo/hi to get correct order: [lo0, hi0, lo1, hi1, ...]
                let interleaved_lo = vzip1q_s8(lo_vals, hi_vals);
                let interleaved_hi = vzip2q_s8(lo_vals, hi_vals);

                // First 16 values (interleaved_lo): i8 → i16 → i32 → f32
                let lo8 = vget_low_s8(interleaved_lo);
                let hi8 = vget_high_s8(interleaved_lo);
                let lo16_0 = vmovl_s8(lo8);
                let lo16_1 = vmovl_s8(hi8);

                let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0)));
                let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0)));
                let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1)));
                let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1)));

                $crate::simd_primitive!(neon, f32, store, out_ptr, $crate::simd_primitive!(neon, f32, mul, vd, f0));
                $crate::simd_primitive!(neon, f32, store, out_ptr.add(4), $crate::simd_primitive!(neon, f32, mul, vd, f1));
                $crate::simd_primitive!(neon, f32, store, out_ptr.add(8), $crate::simd_primitive!(neon, f32, mul, vd, f2));
                $crate::simd_primitive!(neon, f32, store, out_ptr.add(12), $crate::simd_primitive!(neon, f32, mul, vd, f3));

                // Next 16 values (interleaved_hi): i8 → i16 → i32 → f32
                let lo8b = vget_low_s8(interleaved_hi);
                let hi8b = vget_high_s8(interleaved_hi);
                let lo16_2 = vmovl_s8(lo8b);
                let lo16_3 = vmovl_s8(hi8b);

                let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_2)));
                let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_2)));
                let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_3)));
                let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_3)));

                $crate::simd_primitive!(neon, f32, store, out_ptr.add(16), $crate::simd_primitive!(neon, f32, mul, vd, f4));
                $crate::simd_primitive!(neon, f32, store, out_ptr.add(20), $crate::simd_primitive!(neon, f32, mul, vd, f5));
                $crate::simd_primitive!(neon, f32, store, out_ptr.add(24), $crate::simd_primitive!(neon, f32, mul, vd, f6));
                $crate::simd_primitive!(neon, f32, store, out_ptr.add(28), $crate::simd_primitive!(neon, f32, mul, vd, f7));
            }
        }
    };

    (neon, iq4_xs, decode, $block_ptr:expr, $out_ptr:expr) => {
        unsafe {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let block = &*$block_ptr;
                let d: f32 = block.d.to_f32();
                let qs = &block.qs;
                let out_ptr = $out_ptr;
                let vd = $crate::simd_primitive!(neon, f32, splat, d);

                // Load 16-entry codebook into NEON register
                let lut = vld1q_s8(crate::codebooks::KVALUES_IQ4NL.as_ptr());
                let mask_lo = vdupq_n_u8(0x0F);

                // IQ4_XS block_size=256, qs has 128 bytes → 8 iterations of 16 bytes (32 values each)
                for i in 0..8 {
                    let raw = vld1q_u8(qs.as_ptr().add(i * 16));
                    let lo_idx = vandq_u8(raw, mask_lo);
                    let hi_idx = vshrq_n_u8::<4>(raw);

                    let lo_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(lo_idx)));
                    let hi_vals = vqtbl1q_s8(lut, vreinterpretq_u8_s8(vreinterpretq_s8_u8(hi_idx)));

                    let interleaved_lo = vzip1q_s8(lo_vals, hi_vals);
                    let interleaved_hi = vzip2q_s8(lo_vals, hi_vals);

                    let base = i * 32;

                    // interleaved_lo → 16 f32 values
                    let lo8 = vget_low_s8(interleaved_lo);
                    let hi8 = vget_high_s8(interleaved_lo);
                    let s16_0 = vmovl_s8(lo8);
                    let s16_1 = vmovl_s8(hi8);

                    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_0)));
                    let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_0)));
                    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_1)));
                    let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_1)));

                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base), $crate::simd_primitive!(neon, f32, mul, vd, f0));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 4), $crate::simd_primitive!(neon, f32, mul, vd, f1));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 8), $crate::simd_primitive!(neon, f32, mul, vd, f2));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 12), $crate::simd_primitive!(neon, f32, mul, vd, f3));

                    // interleaved_hi → 16 f32 values
                    let lo8b = vget_low_s8(interleaved_hi);
                    let hi8b = vget_high_s8(interleaved_hi);
                    let s16_2 = vmovl_s8(lo8b);
                    let s16_3 = vmovl_s8(hi8b);

                    let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_2)));
                    let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_2)));
                    let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_3)));
                    let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_3)));

                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 16), $crate::simd_primitive!(neon, f32, mul, vd, f4));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 20), $crate::simd_primitive!(neon, f32, mul, vd, f5));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 24), $crate::simd_primitive!(neon, f32, mul, vd, f6));
                    $crate::simd_primitive!(neon, f32, store, out_ptr.add(base + 28), $crate::simd_primitive!(neon, f32, mul, vd, f7));
                }
            }
        }
    };
}
