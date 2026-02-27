//! Intel AMX (Advanced Matrix Extensions) codegen for GEMM kernels.
//!
//! AMX provides 8 tile registers (tmm0-tmm7), each up to 16 rows x 64 bytes (1 KiB).
//! Supported operations:
//! - AMX-BF16: TDPBF16PS -- BF16 dot-product into f32 accumulator
//! - AMX-INT8: TDPBSSD/TDPBSUD/TDPBUSD/TDPBUUD -- INT8 dot-product into i32 accumulator
//!
//! Tile register allocation for 2x2 GEMM:
//!   tmm0..tmm3 -- accumulators (C tiles)
//!   tmm4, tmm5 -- A tiles (two row-blocks)
//!   tmm6, tmm7 -- B tiles (two col-blocks)

#[cfg(feature = "jit-x86")]
use iced_x86::code_asm::*;

/// TILECFG palette 1 configuration (64 bytes, loaded via LDTILECFG).
///
/// Layout (Intel SDM Vol. 2, Table 3-2):
///   byte  0:     palette (must be 1)
///   byte  1:     start_row (0 for fresh config)
///   bytes 2-15:  reserved (0)
///   bytes 16-31: colsb[0..7] as u16 -- columns in bytes per tile
///   bytes 32-47: reserved (0)
///   bytes 48-55: rows[0..7] as u8 -- row count per tile
///   bytes 56-63: reserved (0)
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct TileCfg {
    pub data: [u8; 64],
}

impl TileCfg {
    /// Create a zeroed TILECFG (palette 0 = tiles disabled).
    pub fn zeroed() -> Self {
        Self { data: [0u8; 64] }
    }

    /// Create a TILECFG for BF16 GEMM with 2x2 accumulator layout.
    ///
    /// - Accumulators (tmm0-3): `tile_m` rows x `tile_n * 4` bytes (f32)
    /// - A tiles (tmm4-5): `tile_m` rows x `tile_k * 4` bytes (bf16 pairs)
    /// - B tiles (tmm6-7): `tile_k` rows x `tile_n * 4` bytes (bf16 pairs)
    pub fn bf16_gemm_2x2(tile_m: u8, tile_n: u8, tile_k: u8) -> Self {
        let mut cfg = Self::zeroed();
        cfg.data[0] = 1; // palette 1

        let acc_colsb = (tile_n as u16) * 4; // f32 output columns
        let a_colsb = (tile_k as u16) * 4;   // bf16 pairs (K/2 dwords)
        let b_colsb = acc_colsb;             // bf16 pairs matching N

        // Accumulators: tmm0-3
        for i in 0..4u8 {
            cfg.set_tile(i, tile_m, acc_colsb);
        }
        // A tiles: tmm4-5
        cfg.set_tile(4, tile_m, a_colsb);
        cfg.set_tile(5, tile_m, a_colsb);
        // B tiles: tmm6-7
        cfg.set_tile(6, tile_k, b_colsb);
        cfg.set_tile(7, tile_k, b_colsb);

        cfg
    }

    /// Create a TILECFG for INT8 GEMM with 2x2 accumulator layout.
    pub fn int8_gemm_2x2(tile_m: u8, tile_n: u8, tile_k: u8) -> Self {
        // Layout is identical to BF16 -- only the TDPB instruction differs.
        Self::bf16_gemm_2x2(tile_m, tile_n, tile_k)
    }

    fn set_tile(&mut self, idx: u8, rows: u8, colsb: u16) {
        assert!(idx < 8);
        let i = idx as usize;
        let cb = colsb.to_le_bytes();
        self.data[16 + 2 * i] = cb[0];
        self.data[16 + 2 * i + 1] = cb[1];
        self.data[48 + i] = rows;
    }
}

/// Whether the given GEMM dimensions are suitable for AMX acceleration.
pub fn amx_gemm_eligible(m: usize, n: usize, k: usize, is_bf16: bool) -> bool {
    if is_bf16 {
        m >= 1 && n >= 1 && k >= 2 && k % 2 == 0
    } else {
        m >= 1 && n >= 1 && k >= 4 && k % 4 == 0
    }
}

// -- iced-x86 AMX code emission --

#[cfg(feature = "jit-x86")]
pub mod jit {
    use super::*;

    /// Map tile index 0..7 to iced-x86 AsmRegisterTmm.
    pub fn tmm_reg(idx: u8) -> Result<AsmRegisterTmm, String> {
        match idx {
            0 => Ok(tmm0),
            1 => Ok(tmm1),
            2 => Ok(tmm2),
            3 => Ok(tmm3),
            4 => Ok(tmm4),
            5 => Ok(tmm5),
            6 => Ok(tmm6),
            7 => Ok(tmm7),
            _ => Err(format!("tmm index {idx} out of range (0..7)")),
        }
    }

    /// Emit LDTILECFG from a stack-allocated 64-byte config block.
    pub fn emit_ldtilecfg(
        asm: &mut CodeAssembler,
        cfg: &TileCfg,
    ) -> Result<(), String> {
        asm.sub(rsp, 64i32).map_err(|e| e.to_string())?;

        for i in 0..8usize {
            let off = i * 8;
            let qw = u64::from_le_bytes([
                cfg.data[off],
                cfg.data[off + 1],
                cfg.data[off + 2],
                cfg.data[off + 3],
                cfg.data[off + 4],
                cfg.data[off + 5],
                cfg.data[off + 6],
                cfg.data[off + 7],
            ]);
            if qw != 0 {
                asm.mov(rax, qw).map_err(|e| e.to_string())?;
                asm.mov(qword_ptr(rsp + off as i32), rax)
                    .map_err(|e| e.to_string())?;
            } else {
                asm.mov(qword_ptr(rsp + off as i32), 0i32)
                    .map_err(|e| e.to_string())?;
            }
        }

        asm.ldtilecfg(ptr(rsp)).map_err(|e| e.to_string())?;
        asm.add(rsp, 64i32).map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Emit TILEZERO for tiles 0..n (accumulators).
    pub fn emit_tilezero_accumulators(
        asm: &mut CodeAssembler,
        count: u8,
    ) -> Result<(), String> {
        for i in 0..count {
            asm.tilezero(tmm_reg(i)?).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Emit TILERELEASE to free all tile state.
    pub fn emit_tilerelease(asm: &mut CodeAssembler) -> Result<(), String> {
        asm.tilerelease().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Emit a BF16 GEMM microkernel using AMX 2x2 tile layout.
    ///
    /// Register convention (System V AMD64 ABI):
    ///   rdi = A pointer (row-major bf16, stride = lda bytes)
    ///   rsi = B pointer (row-major bf16, stride = ldb bytes)
    ///   rdx = C pointer (row-major f32,  stride = ldc bytes)
    ///   rcx = K_pairs (number of dword-pairs = K_bf16 / 2)
    ///   r8  = lda (A row stride in bytes)
    ///   r9  = ldb (B row stride in bytes)
    ///   r10 = ldc (C row stride in bytes)
    ///
    /// Computes: C[0:2T, 0:2T] += A[0:2T, 0:K] * B[0:K, 0:2T]
    pub fn emit_amx_bf16_gemm_2x2(
        asm: &mut CodeAssembler,
        tile_m: u8,
        tile_n: u8,
        tile_k: u8,
    ) -> Result<(), String> {
        let cfg = TileCfg::bf16_gemm_2x2(tile_m, tile_n, tile_k);
        emit_ldtilecfg(asm, &cfg)?;

        // Save callee-saved registers
        asm.push(r12).map_err(|e| e.to_string())?;
        asm.push(r13).map_err(|e| e.to_string())?;
        asm.push(r14).map_err(|e| e.to_string())?;
        asm.push(r15).map_err(|e| e.to_string())?;

        // Zero accumulators tmm0-3
        emit_tilezero_accumulators(asm, 4)?;

        // r11 = A column byte offset (advances by tile_k * 4 per K-step)
        // r12 = B row byte offset (advances by tile_k * ldb per K-step)
        // r13 = K-pairs remaining
        asm.xor(r11d, r11d).map_err(|e| e.to_string())?;
        asm.xor(r12d, r12d).map_err(|e| e.to_string())?;
        asm.mov(r13, rcx).map_err(|e| e.to_string())?;

        // r14 = tile_m * lda (row offset for second A block)
        asm.imul_3(r14, r8, tile_m as i32).map_err(|e| e.to_string())?;
        // r15 = tile_n * 4 (column byte offset for second B/C block)
        asm.mov(r15d, (tile_n as i32) * 4).map_err(|e| e.to_string())?;

        let k_step = tile_k as i32;
        let a_step = k_step * 4;

        let mut k_loop = asm.create_label();
        let mut k_done = asm.create_label();

        // -- K loop --
        asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
        asm.cmp(r13d, k_step).map_err(|e| e.to_string())?;
        asm.jl(k_done).map_err(|e| e.to_string())?;

        // Load A tiles
        asm.lea(rax, ptr(rdi + r11)).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm4, ptr(rax + r8)).map_err(|e| e.to_string())?;

        asm.lea(rax, ptr(rdi + r14)).map_err(|e| e.to_string())?;
        asm.add(rax, r11).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm5, ptr(rax + r8)).map_err(|e| e.to_string())?;

        // Load B tiles
        asm.lea(rax, ptr(rsi + r12)).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm6, ptr(rax + r9)).map_err(|e| e.to_string())?;

        asm.lea(rax, ptr(rsi + r12)).map_err(|e| e.to_string())?;
        asm.add(rax, r15).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm7, ptr(rax + r9)).map_err(|e| e.to_string())?;

        // 2x2 outer product
        asm.tdpbf16ps(tmm0, tmm4, tmm6).map_err(|e| e.to_string())?;
        asm.tdpbf16ps(tmm1, tmm4, tmm7).map_err(|e| e.to_string())?;
        asm.tdpbf16ps(tmm2, tmm5, tmm6).map_err(|e| e.to_string())?;
        asm.tdpbf16ps(tmm3, tmm5, tmm7).map_err(|e| e.to_string())?;

        // Advance K counters
        asm.add(r11d, a_step).map_err(|e| e.to_string())?;
        asm.imul_3(rax, r9, k_step).map_err(|e| e.to_string())?;
        asm.add(r12, rax).map_err(|e| e.to_string())?;
        asm.sub(r13d, k_step).map_err(|e| e.to_string())?;
        asm.jmp(k_loop).map_err(|e| e.to_string())?;

        asm.set_label(&mut k_done).map_err(|e| e.to_string())?;

        // -- Store accumulators to C --
        asm.mov(rax, rdx).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm0).map_err(|e| e.to_string())?;

        asm.mov(rax, rdx).map_err(|e| e.to_string())?;
        asm.add(rax, r15).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm1).map_err(|e| e.to_string())?;

        asm.imul_3(rax, r10, tile_m as i32).map_err(|e| e.to_string())?;
        asm.add(rax, rdx).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm2).map_err(|e| e.to_string())?;

        asm.imul_3(rax, r10, tile_m as i32).map_err(|e| e.to_string())?;
        asm.add(rax, rdx).map_err(|e| e.to_string())?;
        asm.add(rax, r15).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm3).map_err(|e| e.to_string())?;

        emit_tilerelease(asm)?;

        // Restore callee-saved registers
        asm.pop(r15).map_err(|e| e.to_string())?;
        asm.pop(r14).map_err(|e| e.to_string())?;
        asm.pop(r13).map_err(|e| e.to_string())?;
        asm.pop(r12).map_err(|e| e.to_string())?;

        asm.ret().map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Emit an INT8 GEMM microkernel using AMX 2x2 tile layout.
    /// Uses TDPBSSD (signed x signed -> i32 accumulator).
    /// Same register convention as BF16 variant.
    pub fn emit_amx_int8_gemm_2x2(
        asm: &mut CodeAssembler,
        tile_m: u8,
        tile_n: u8,
        tile_k: u8,
    ) -> Result<(), String> {
        let cfg = TileCfg::int8_gemm_2x2(tile_m, tile_n, tile_k);
        emit_ldtilecfg(asm, &cfg)?;

        asm.push(r12).map_err(|e| e.to_string())?;
        asm.push(r13).map_err(|e| e.to_string())?;
        asm.push(r14).map_err(|e| e.to_string())?;
        asm.push(r15).map_err(|e| e.to_string())?;

        emit_tilezero_accumulators(asm, 4)?;

        asm.xor(r11d, r11d).map_err(|e| e.to_string())?;
        asm.xor(r12d, r12d).map_err(|e| e.to_string())?;
        asm.mov(r13, rcx).map_err(|e| e.to_string())?;

        asm.imul_3(r14, r8, tile_m as i32).map_err(|e| e.to_string())?;
        asm.mov(r15d, (tile_n as i32) * 4).map_err(|e| e.to_string())?;

        let k_step = tile_k as i32;
        let a_step = k_step * 4;

        let mut k_loop = asm.create_label();
        let mut k_done = asm.create_label();

        asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
        asm.cmp(r13d, k_step).map_err(|e| e.to_string())?;
        asm.jl(k_done).map_err(|e| e.to_string())?;

        asm.lea(rax, ptr(rdi + r11)).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm4, ptr(rax + r8)).map_err(|e| e.to_string())?;
        asm.lea(rax, ptr(rdi + r14)).map_err(|e| e.to_string())?;
        asm.add(rax, r11).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm5, ptr(rax + r8)).map_err(|e| e.to_string())?;

        asm.lea(rax, ptr(rsi + r12)).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm6, ptr(rax + r9)).map_err(|e| e.to_string())?;
        asm.lea(rax, ptr(rsi + r12)).map_err(|e| e.to_string())?;
        asm.add(rax, r15).map_err(|e| e.to_string())?;
        asm.tileloadd(tmm7, ptr(rax + r9)).map_err(|e| e.to_string())?;

        asm.tdpbssd(tmm0, tmm4, tmm6).map_err(|e| e.to_string())?;
        asm.tdpbssd(tmm1, tmm4, tmm7).map_err(|e| e.to_string())?;
        asm.tdpbssd(tmm2, tmm5, tmm6).map_err(|e| e.to_string())?;
        asm.tdpbssd(tmm3, tmm5, tmm7).map_err(|e| e.to_string())?;

        asm.add(r11d, a_step).map_err(|e| e.to_string())?;
        asm.imul_3(rax, r9, k_step).map_err(|e| e.to_string())?;
        asm.add(r12, rax).map_err(|e| e.to_string())?;
        asm.sub(r13d, k_step).map_err(|e| e.to_string())?;
        asm.jmp(k_loop).map_err(|e| e.to_string())?;

        asm.set_label(&mut k_done).map_err(|e| e.to_string())?;

        asm.mov(rax, rdx).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm0).map_err(|e| e.to_string())?;

        asm.mov(rax, rdx).map_err(|e| e.to_string())?;
        asm.add(rax, r15).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm1).map_err(|e| e.to_string())?;

        asm.imul_3(rax, r10, tile_m as i32).map_err(|e| e.to_string())?;
        asm.add(rax, rdx).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm2).map_err(|e| e.to_string())?;

        asm.imul_3(rax, r10, tile_m as i32).map_err(|e| e.to_string())?;
        asm.add(rax, rdx).map_err(|e| e.to_string())?;
        asm.add(rax, r15).map_err(|e| e.to_string())?;
        asm.tilestored(ptr(rax + r10), tmm3).map_err(|e| e.to_string())?;

        emit_tilerelease(asm)?;

        asm.pop(r15).map_err(|e| e.to_string())?;
        asm.pop(r14).map_err(|e| e.to_string())?;
        asm.pop(r13).map_err(|e| e.to_string())?;
        asm.pop(r12).map_err(|e| e.to_string())?;

        asm.ret().map_err(|e| e.to_string())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tilecfg_bf16_layout() {
        let cfg = TileCfg::bf16_gemm_2x2(16, 16, 16);
        assert_eq!(cfg.data[0], 1, "palette must be 1");

        // Accumulators tmm0-3: 16 rows, 64 colsb (16 f32)
        for i in 0..4 {
            let colsb = u16::from_le_bytes([cfg.data[16 + 2 * i], cfg.data[17 + 2 * i]]);
            assert_eq!(colsb, 64, "tmm{i} colsb");
            assert_eq!(cfg.data[48 + i], 16, "tmm{i} rows");
        }
        // A tiles tmm4-5: 16 rows, 64 colsb
        for i in 4..6 {
            let colsb = u16::from_le_bytes([cfg.data[16 + 2 * i], cfg.data[17 + 2 * i]]);
            assert_eq!(colsb, 64, "tmm{i} colsb");
            assert_eq!(cfg.data[48 + i], 16, "tmm{i} rows");
        }
        // B tiles tmm6-7: 16 rows, 64 colsb
        for i in 6..8 {
            let colsb = u16::from_le_bytes([cfg.data[16 + 2 * i], cfg.data[17 + 2 * i]]);
            assert_eq!(colsb, 64, "tmm{i} colsb");
            assert_eq!(cfg.data[48 + i], 16, "tmm{i} rows");
        }
    }

    #[test]
    fn test_tilecfg_int8_layout() {
        let cfg = TileCfg::int8_gemm_2x2(16, 16, 16);
        assert_eq!(cfg.data[0], 1);
        let cfg_bf16 = TileCfg::bf16_gemm_2x2(16, 16, 16);
        assert_eq!(cfg.data, cfg_bf16.data);
    }

    #[test]
    fn test_tilecfg_small_tiles() {
        let cfg = TileCfg::bf16_gemm_2x2(8, 4, 8);
        assert_eq!(cfg.data[48], 8);
        let colsb = u16::from_le_bytes([cfg.data[16], cfg.data[17]]);
        assert_eq!(colsb, 16);
    }

    #[test]
    fn test_amx_gemm_eligible_bf16() {
        assert!(amx_gemm_eligible(16, 16, 32, true));
        assert!(amx_gemm_eligible(1, 1, 2, true));
        assert!(!amx_gemm_eligible(16, 16, 3, true));
        assert!(!amx_gemm_eligible(0, 16, 32, true));
    }

    #[test]
    fn test_amx_gemm_eligible_int8() {
        assert!(amx_gemm_eligible(16, 16, 64, false));
        assert!(amx_gemm_eligible(1, 1, 4, false));
        assert!(!amx_gemm_eligible(16, 16, 3, false));
        assert!(!amx_gemm_eligible(16, 0, 64, false));
    }

    #[test]
    fn test_tmm_reg_mapping() {
        #[cfg(feature = "jit-x86")]
        {
            for i in 0..8u8 {
                assert!(jit::tmm_reg(i).is_ok());
            }
            assert!(jit::tmm_reg(8).is_err());
        }
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emit_tilecfg_assembles() {
        let cfg = TileCfg::bf16_gemm_2x2(16, 16, 16);
        let mut asm = CodeAssembler::new(64).unwrap();
        let result = jit::emit_ldtilecfg(&mut asm, &cfg);
        assert!(result.is_ok(), "emit_ldtilecfg failed: {:?}", result.err());
        let code = asm.assemble(0x1000).unwrap();
        assert!(!code.is_empty());
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emit_tilezero_assembles() {
        let mut asm = CodeAssembler::new(64).unwrap();
        let result = jit::emit_tilezero_accumulators(&mut asm, 4);
        assert!(result.is_ok());
        let code = asm.assemble(0x1000).unwrap();
        assert!(!code.is_empty());
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emit_tilerelease_assembles() {
        let mut asm = CodeAssembler::new(64).unwrap();
        let result = jit::emit_tilerelease(&mut asm);
        assert!(result.is_ok());
        let code = asm.assemble(0x1000).unwrap();
        assert!(!code.is_empty());
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emit_bf16_gemm_assembles() {
        let mut asm = CodeAssembler::new(64).unwrap();
        let result = jit::emit_amx_bf16_gemm_2x2(&mut asm, 16, 16, 16);
        assert!(result.is_ok(), "emit_amx_bf16_gemm_2x2 failed: {:?}", result.err());
        let code = asm.assemble(0x1000).unwrap();
        assert!(!code.is_empty());
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emit_int8_gemm_assembles() {
        let mut asm = CodeAssembler::new(64).unwrap();
        let result = jit::emit_amx_int8_gemm_2x2(&mut asm, 16, 16, 16);
        assert!(result.is_ok(), "emit_amx_int8_gemm_2x2 failed: {:?}", result.err());
        let code = asm.assemble(0x1000).unwrap();
        assert!(!code.is_empty());
    }
}
