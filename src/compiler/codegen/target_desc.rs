//! Target descriptor — platform-independent hardware description for codegen.
//!
//! Provides the `TargetDesc` struct that captures SIMD width, register count,
//! and microkernel dimensions. The algorithm layer (`algorithm.rs`) uses this
//! to drive code generation without knowing the specific ISA.

use crate::dispatch::DeviceProfile;

/// Platform-independent description of the target's SIMD capabilities.
///
/// Constructed once per codegen invocation from `DeviceProfile`. The algorithm
/// layer reads these fields to determine loop tiling, register allocation
/// limits, and microkernel geometry.
#[derive(Debug, Clone, Copy)]
pub struct TargetDesc {
    /// Number of f32 elements per SIMD register (4=NEON, 8=AVX2, 16=AVX-512).
    pub simd_width_f32: usize,
    /// Total number of SIMD registers (16=AVX2, 32=NEON/AVX-512).
    pub num_simd_regs: usize,
    /// Scratch registers available (total - accumulators - reserved).
    pub n_scratch_regs: usize,
    /// Microkernel M dimension (rows per tile).
    pub mr: usize,
    /// Microkernel N dimension (columns per tile).
    pub nr: usize,
    /// SIMD register width in bytes (16=NEON, 32=AVX2, 64=AVX-512).
    pub simd_bytes: usize,
}

impl TargetDesc {
    /// Build a `TargetDesc` from the detected hardware profile.
    pub fn from_profile(profile: &DeviceProfile) -> Self {
        let simd_width_f32 = profile.simd_width_f32();
        let num_simd_regs = profile.num_simd_regs();
        // Use a default blocking to get mr/nr
        let blocking = profile.gemm_blocking(64, 64, 64);
        let mr = blocking.mr;
        let nr = blocking.nr;
        let nr_vecs = nr / simd_width_f32;
        let n_accum = mr * nr_vecs;
        // Reserve 3 scratch regs for math (exp/tanh/log)
        let n_scratch_regs = num_simd_regs.saturating_sub(n_accum + 3);
        let simd_bytes = simd_width_f32 * 4;

        TargetDesc {
            simd_width_f32,
            num_simd_regs,
            n_scratch_regs,
            mr,
            nr,
            simd_bytes,
        }
    }

    /// Number of SIMD vectors per NR columns.
    #[inline]
    pub fn nr_vecs(&self) -> usize {
        self.nr / self.simd_width_f32
    }

    /// Number of accumulator registers needed for an MR×NR tile.
    #[inline]
    pub fn n_accum_regs(&self) -> usize {
        self.mr * self.nr_vecs()
    }
}
