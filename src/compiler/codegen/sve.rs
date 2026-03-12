//! ARM SVE/SVE2 codegen for GEMM micro-kernels.
//!
//! SVE (Scalable Vector Extension) provides vector registers whose width is
//! determined at runtime (128-2048 bits in 128-bit increments). This module
//! emits VL-agnostic machine code that adapts to the hardware vector length.
//!
//! Key SVE instructions used:
//! - FMLA (vectors): Zd.S = Za.S + Zn.S * Zm.S[idx]  (indexed f32 FMA)
//! - LD1W / ST1W: predicated contiguous load/store of f32 elements
//! - PTRUE: create all-true predicate for the active VL
//! - WHILELT: create predicate for loop tails (elements < bound)
//! - CNTW: count f32 elements per vector (VL/32)
//! - PRFM: SVE prefetch with predicate
//!
//! Register allocation convention for GEMM micro-kernel:
//!   z0..z(Nr-1)   -- accumulator rows (one per N-register)
//!   z24..z27      -- A-side broadcast / loaded values
//!   z28..z31      -- B-side loaded vectors
//!   p0            -- all-true predicate (PTRUE p0.s)
//!   p1            -- tail predicate (WHILELT)

use super::emitter::Platform;

/// SVE vector-length query result.
#[derive(Debug, Clone, Copy)]
pub struct SveVl {
    /// Vector length in bytes (16, 32, 48, 64, ... up to 256).
    pub vl_bytes: usize,
    /// Number of f32 elements per Z register (vl_bytes / 4).
    pub f32_per_vec: usize,
}

impl SveVl {
    /// Create from a known VL in bytes.
    pub fn from_bytes(vl_bytes: usize) -> Self {
        debug_assert!(vl_bytes >= 16 && vl_bytes % 16 == 0);
        Self {
            vl_bytes,
            f32_per_vec: vl_bytes / 4,
        }
    }

    /// Minimum SVE VL (128-bit NEON-equivalent).
    pub fn min_128() -> Self {
        Self::from_bytes(16)
    }
}

/// SVE code generator for GEMM micro-kernels.
///
/// Emits AArch64 SVE/SVE2 instructions as raw `u32` words. The generated
/// code is VL-agnostic: it uses CNTW/WHILELT to handle arbitrary vector
/// lengths at runtime.
#[derive(Debug)]
pub struct SveCodeGen {
    /// Assembled instruction words.
    code: Vec<u32>,
    /// Whether SVE2 extensions are available.
    pub sve2: bool,
    /// Runtime vector length info (used for tile size calculations).
    pub vl: SveVl,
}

impl SveCodeGen {
    /// Create a new SVE code generator.
    ///
    /// - `sve2`: enable SVE2-specific instructions (FMMLA, etc.)
    /// - `vl_bytes`: hardware vector length in bytes (from CNTB/RDVL)
    pub fn new(sve2: bool, vl_bytes: usize) -> Self {
        Self {
            code: Vec::with_capacity(256),
            sve2,
            vl: SveVl::from_bytes(vl_bytes),
        }
    }

    /// Create from a DeviceProfile's detected platform.
    pub fn from_platform(platform: &Platform) -> Option<Self> {
        match platform {
            Platform::Aarch64 { sve: true, .. } => {
                // Default to 128-bit minimum; actual VL is set at runtime.
                Some(Self::new(false, 16))
            }
            _ => None,
        }
    }

    /// Emit a raw 32-bit instruction word.
    #[inline]
    fn emit(&mut self, insn: u32) {
        self.code.push(insn);
    }

    /// Get the assembled code as a byte slice.
    pub fn code_bytes(&self) -> &[u32] {
        &self.code
    }

    /// Total code size in bytes.
    pub fn code_size(&self) -> usize {
        self.code.len() * 4
    }

    // -- SVE instruction encodings --

    /// PTRUE Pd.S, pattern
    /// Initialize predicate register to all-true for 32-bit elements.
    ///
    /// Encoding: 0x2598E000 | (pattern << 5) | pd
    /// pattern=0x1F (ALL) -> all lanes active
    pub fn emit_ptrue_s(&mut self, pd: u8) {
        debug_assert!(pd < 16);
        let insn: u32 = 0x2598E000 | (0x1F << 5) | (pd as u32);
        self.emit(insn);
    }

    /// CNTW Xd -- count 32-bit elements per SVE vector.
    ///
    /// Encoding: 0x04A0E000 | (0x1F << 5) | rd
    pub fn emit_cntw(&mut self, rd: u8) {
        debug_assert!(rd < 31);
        let insn: u32 = 0x04A0E000 | (0x1F << 5) | (rd as u32);
        self.emit(insn);
    }

    /// WHILELT Pd.S, Xn, Xm -- set predicate lanes where Xn+i < Xm.
    ///
    /// Used for tail masking: only process remaining elements.
    /// Encoding: 0x25A00400 | (xm << 16) | (xn << 5) | pd
    pub fn emit_whilelt_s(&mut self, pd: u8, xn: u8, xm: u8) {
        debug_assert!(pd < 16 && xn < 31 && xm < 31);
        let insn: u32 = 0x25A00400 | ((xm as u32) << 16) | ((xn as u32) << 5) | (pd as u32);
        self.emit(insn);
    }

    /// LD1W {Zt.S}, Pg/Z, [Xn, Xm, LSL #2]
    /// Predicated contiguous load of f32 elements with register offset.
    ///
    /// Encoding: 0xA5404000 | (zm << 16) | (pg << 10) | (xn << 5) | zt
    pub fn emit_ld1w_scalar(&mut self, zt: u8, pg: u8, xn: u8, xm: u8) {
        debug_assert!(zt < 32 && pg < 8 && xn < 31 && xm < 31);
        let insn: u32 =
            0xA5404000 | ((xm as u32) << 16) | ((pg as u32) << 10) | ((xn as u32) << 5) | (zt as u32);
        self.emit(insn);
    }

    /// LD1W {Zt.S}, Pg/Z, [Xn, #imm, MUL VL]
    /// Predicated contiguous load with scaled immediate offset.
    ///
    /// imm range: -8..7 (4-bit signed, scaled by VL).
    /// Encoding: 0xA540A000 | ((imm & 0xF) << 16) | (pg << 10) | (xn << 5) | zt
    pub fn emit_ld1w_imm(&mut self, zt: u8, pg: u8, xn: u8, imm: i8) {
        debug_assert!(zt < 32 && pg < 8 && xn < 31 && (-8..=7).contains(&imm));
        let imm_bits = (imm as u32) & 0xF;
        let insn: u32 =
            0xA540A000 | (imm_bits << 16) | ((pg as u32) << 10) | ((xn as u32) << 5) | (zt as u32);
        self.emit(insn);
    }

    /// ST1W {Zt.S}, Pg, [Xn, #imm, MUL VL]
    /// Predicated contiguous store with scaled immediate offset.
    ///
    /// Encoding: 0xE500E000 | ((imm & 0xF) << 16) | (pg << 10) | (xn << 5) | zt
    pub fn emit_st1w_imm(&mut self, zt: u8, pg: u8, xn: u8, imm: i8) {
        debug_assert!(zt < 32 && pg < 8 && xn < 31 && (-8..=7).contains(&imm));
        let imm_bits = (imm as u32) & 0xF;
        let insn: u32 =
            0xE500E000 | (imm_bits << 16) | ((pg as u32) << 10) | ((xn as u32) << 5) | (zt as u32);
        self.emit(insn);
    }

    /// FMLA Zda.S, Zn.S, Zm.S[idx]
    /// Fused multiply-add: Zda += Zn * Zm[idx] (f32, indexed).
    ///
    /// Encoding: 0x64200000 | (idx << 19) | (zm << 16) | (zn << 5) | zda
    /// idx: 0..3 (2-bit index into 128-bit segment)
    pub fn emit_fmla_indexed_s(&mut self, zda: u8, zn: u8, zm: u8, idx: u8) {
        debug_assert!(zda < 32 && zn < 32 && zm < 8 && idx < 4);
        let insn: u32 = 0x64200000
            | ((idx as u32) << 19)
            | ((zm as u32) << 16)
            | ((zn as u32) << 5)
            | (zda as u32);
        self.emit(insn);
    }

    /// FMLA Zda.S, Pg/M, Zn.S, Zm.S
    /// Predicated fused multiply-add (non-indexed, full vector).
    ///
    /// Encoding: 0x65A00000 | (zm << 16) | (pg << 10) | (zn << 5) | zda
    pub fn emit_fmla_vec_s(&mut self, zda: u8, pg: u8, zn: u8, zm: u8) {
        debug_assert!(zda < 32 && pg < 8 && zn < 32 && zm < 32);
        let insn: u32 = 0x65A00000
            | ((zm as u32) << 16)
            | ((pg as u32) << 10)
            | ((zn as u32) << 5)
            | (zda as u32);
        self.emit(insn);
    }

    /// DUP Zd.S, Zn.S[idx]
    /// Broadcast element idx of Zn to all lanes of Zd.
    ///
    /// Encoding: 0x05200000 | (tsz_imm << 16) | (zn << 5) | zd
    pub fn emit_dup_element_s(&mut self, zd: u8, zn: u8, idx: u8) {
        debug_assert!(zd < 32 && zn < 32 && idx < (self.vl.f32_per_vec as u8));
        let tsz_imm = 0x04 | ((idx as u32) << 2);
        let insn: u32 = 0x05200000 | (tsz_imm << 16) | ((zn as u32) << 5) | (zd as u32);
        self.emit(insn);
    }

    /// MOV Zd.S, #0 (zero a Z register via DUP immediate).
    ///
    /// Encoding: 0x2538C000 | zd
    pub fn emit_zero_z(&mut self, zd: u8) {
        debug_assert!(zd < 32);
        let insn: u32 = 0x2538C000 | (zd as u32);
        self.emit(insn);
    }

    /// PRFB <prfop>, Pg, [Xn, #imm, MUL VL]
    /// SVE contiguous prefetch, scaled by VL.
    ///
    /// prfop: 0=PLDL1KEEP, 1=PLDL1STRM, 2=PLDL2KEEP, etc.
    pub fn emit_prfb(&mut self, prfop: u8, pg: u8, xn: u8, imm: i8) {
        debug_assert!(prfop < 16 && pg < 8 && xn < 31 && (-8..=7).contains(&imm));
        let imm_bits = (imm as u32) & 0x3F;
        let insn: u32 = 0x85C00000
            | (imm_bits << 16)
            | ((pg as u32) << 10)
            | ((xn as u32) << 5)
            | (prfop as u32);
        self.emit(insn);
    }

    // -- SVE2 instruction encodings --

    /// FMMLA Zda.S, Zn.S, Zm.S (SVE2)
    /// Matrix multiply-accumulate: 2x2 f32 sub-tiles.
    ///
    /// Encoding: 0x64A0E400 | (zm << 16) | (zn << 5) | zda
    pub fn emit_fmmla_s(&mut self, zda: u8, zn: u8, zm: u8) {
        debug_assert!(self.sve2, "FMMLA requires SVE2");
        debug_assert!(zda < 32 && zn < 32 && zm < 32);
        let insn: u32 = 0x64A0E400 | ((zm as u32) << 16) | ((zn as u32) << 5) | (zda as u32);
        self.emit(insn);
    }

    // -- High-level GEMM micro-kernel patterns --

    /// Emit prologue: set up all-true predicate and zero accumulators.
    ///
    /// - `num_acc`: number of accumulator Z registers to zero (starting at z0)
    /// - Returns: predicate register used (p0)
    pub fn emit_gemm_prologue(&mut self, num_acc: u8) -> u8 {
        let pg = 0u8;
        self.emit_ptrue_s(pg);
        for i in 0..num_acc {
            self.emit_zero_z(i);
        }
        pg
    }

    /// Emit a single GEMM inner-loop iteration (rank-1 update).
    ///
    /// Loads one A-column broadcast and Nr B-vectors, then FMAs into
    /// the accumulator tile.
    pub fn emit_rank1_update(
        &mut self,
        pg: u8,
        z_acc_base: u8,
        z_a: u8,
        z_b_base: u8,
        nr: u8,
    ) {
        for j in 0..nr {
            self.emit_fmla_vec_s(z_acc_base + j, pg, z_a, z_b_base + j);
        }
    }

    /// Emit the store-back of accumulator tiles to memory.
    pub fn emit_store_accumulators(
        &mut self,
        pg: u8,
        x_c_base: u8,
        z_acc_base: u8,
        nr: u8,
    ) {
        for j in 0..nr {
            self.emit_st1w_imm(z_acc_base + j, pg, x_c_base, j as i8);
        }
    }
}

/// SVE does not implement TileOps — it is a scalable vector ISA, not a
/// tile-register accelerator like AMX/SME. GEMM micro-kernels use the
/// direct emit_* methods above instead.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sve_vl_from_bytes() {
        let vl = SveVl::from_bytes(32);
        assert_eq!(vl.vl_bytes, 32);
        assert_eq!(vl.f32_per_vec, 8);
    }

    #[test]
    fn test_sve_vl_min() {
        let vl = SveVl::min_128();
        assert_eq!(vl.vl_bytes, 16);
        assert_eq!(vl.f32_per_vec, 4);
    }

    #[test]
    fn test_sve_codegen_new() {
        let gen = SveCodeGen::new(false, 32);
        assert!(!gen.sve2);
        assert_eq!(gen.vl.vl_bytes, 32);
        assert_eq!(gen.code.len(), 0);
    }

    #[test]
    fn test_emit_ptrue() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_ptrue_s(0);
        assert_eq!(gen.code.len(), 1);
        assert_eq!(gen.code[0], 0x2598E000 | (0x1F << 5) | 0);
    }

    #[test]
    fn test_emit_cntw() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_cntw(1);
        assert_eq!(gen.code.len(), 1);
        assert_eq!(gen.code[0], 0x04A0E000 | (0x1F << 5) | 1);
    }

    #[test]
    fn test_emit_zero_z() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_zero_z(5);
        assert_eq!(gen.code.len(), 1);
        assert_eq!(gen.code[0], 0x2538C000 | 5);
    }

    #[test]
    fn test_gemm_prologue() {
        let mut gen = SveCodeGen::new(false, 16);
        let pg = gen.emit_gemm_prologue(4);
        assert_eq!(pg, 0);
        assert_eq!(gen.code.len(), 5); // 1 PTRUE + 4 zeros
    }

    #[test]
    fn test_emit_zero_direct() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_zero_z(3);
        assert_eq!(gen.code.len(), 1);
        assert_eq!(gen.code[0], 0x2538C000 | 3);
    }

    #[test]
    fn test_emit_fmla_vec() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_fmla_vec_s(0, 0, 24, 28);
        assert_eq!(gen.code.len(), 1);
        let insn = gen.code[0];
        assert_eq!(insn & 0xFF000000, 0x65000000);
    }

    #[test]
    fn test_emit_fmmla_sve2() {
        let mut gen = SveCodeGen::new(true, 16);
        gen.emit_fmmla_s(0, 24, 28);
        assert_eq!(gen.code.len(), 1);
        let insn = gen.code[0];
        assert_eq!(insn & 0xFFE0FC00, 0x64A0E400);
    }

    #[test]
    fn test_code_size() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_gemm_prologue(2);
        assert_eq!(gen.code_size(), 12); // 3 * 4 bytes
    }

    #[test]
    fn test_rank1_update() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_rank1_update(0, 0, 24, 28, 4);
        assert_eq!(gen.code.len(), 4);
    }

    #[test]
    fn test_store_accumulators() {
        let mut gen = SveCodeGen::new(false, 16);
        gen.emit_store_accumulators(0, 0, 0, 3);
        assert_eq!(gen.code.len(), 3);
    }
}
