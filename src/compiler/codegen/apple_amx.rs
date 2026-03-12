//! Apple AMX (Apple Matrix eXtensions) codegen for GEMM kernels.
//!
//! AMX is an undocumented coprocessor on Apple Silicon (M1+) accessed via
//! system register writes. It provides 32×32 matrix block operations with
//! dedicated register files:
//!   - X registers (x0-x7): 64 bytes each, hold matrix operand rows
//!   - Y registers (y0-y7): 64 bytes each, hold matrix operand columns
//!   - Z registers (z0-z63): 64 bytes each, hold accumulator tiles
//!
//! AMX instructions are encoded as `msr s0_<op>_c<crn>_c<crm>_<op2>, <Xn>`
//! where the operand register Xn carries the memory address or register index.
//!
//! Key operations:
//!   - LDX/LDY: load 64-byte row/column from memory into X/Y register
//!   - STZ: store 64-byte accumulator row to memory
//!   - FMA32: f32 outer-product accumulate Z += X ⊗ Y (32×32 block)
//!   - FMA16: f16 outer-product accumulate Z += X ⊗ Y with f32 accumulation
//!
//! References:
//!   - Dougall Johnson's reverse-engineering: https://github.com/corsix/amx
//!   - llama.cpp AMX backend

/// AMX operand encoding: pack register index and memory address into Xn.
///
/// For load/store ops, bits [63:56] encode the register index,
/// bits [55:0] encode the 64-byte-aligned memory address.
#[inline]
pub const fn amx_operand(reg_idx: u8, addr: u64) -> u64 {
    ((reg_idx as u64) << 56) | (addr & 0x00FFFFFFFFFFFFFF)
}

/// AMX instruction opcodes (encoded as MSR immediate fields).
///
/// Each AMX op is issued via `msr s0_<op>_c<crn>_c<crm>_<op2>, Xn`.
/// The 17-bit encoding packs: op0=0, op1, CRn, CRm, op2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AmxOp {
    /// Enable AMX coprocessor (must be called before any AMX operation).
    Enable = 0,
    /// Disable AMX coprocessor (should be called after AMX operations complete).
    Disable = 1,
    /// Load 64 bytes into X register from memory.
    Ldx = 2,
    /// Load 64 bytes into Y register from memory.
    Ldy = 3,
    /// Store 64 bytes from Z register to memory.
    Stz = 4,
    /// Load 64 bytes into X register, pair mode.
    LdxPair = 5,
    /// Load 64 bytes into Y register, pair mode.
    LdyPair = 6,
    /// Store 64 bytes from Z register, pair mode.
    StzPair = 7,
    /// f32 outer-product: Z += X ⊗ Y (32-bit float).
    Fma32 = 8,
    /// f16 outer-product: Z += X ⊗ Y (16-bit float, f32 accumulation).
    Fma16 = 9,
    /// Clear Z accumulator registers.
    ClearZ = 17,
}

/// Encode an AMX instruction as a raw AArch64 `msr` instruction word.
///
/// AMX instructions use the system register encoding space:
///   `msr s0_<amx_row>_c<crn>_c<crm>_<op2>, Xn`
///
/// The instruction word layout:
///   [31:22] = 0b1101010100_1 (MSR prefix)
///   [21:19] = op1 (AMX row / 2)
///   [18:15] = CRn
///   [14:11] = CRm
///   [10:8]  = op2
///   [7:5]   = 0
///   [4:0]   = Rt (source register)
#[inline]
pub const fn encode_amx_msr(op: AmxOp, rt: u8) -> u32 {
    // AMX instructions are encoded in the implementation-defined sysreg space.
    // Base: 0x00201000 in the MSR encoding, with AMX op in bits [9:5].
    let base: u32 = 0xD5000000; // MSR base
    let amx_row = op as u32;
    // Encode into o0=0, op1=amx_row[4:2], CRn=0b0001, CRm=amx_row[1:0]<<2, op2=0
    let op1 = (amx_row >> 2) & 0x7;
    let crn = 0x11u32; // AMX uses CRn=17 in Apple's encoding
    let crm = (amx_row & 0x3) << 2;
    let op2 = 0u32;
    base | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5) | (rt as u32 & 0x1F)
}

/// Whether the given GEMM dimensions are suitable for Apple AMX acceleration.
///
/// AMX operates on 32x32 f32 blocks (or 32x32 f16 with f32 accumulation).
/// M and N must be multiples of 32; K must be at least 1.
pub fn apple_amx_gemm_eligible(m: usize, n: usize, k: usize) -> bool {
    m >= 32 && n >= 32 && k >= 1 && m % 32 == 0 && n % 32 == 0
}

/// AMX tile dimensions for GEMM microkernels.
pub const AMX_TILE_M: usize = 32;
pub const AMX_TILE_N: usize = 32;
/// Number of f32 elements per AMX register (64 bytes / 4 bytes per f32).
pub const AMX_REG_F32: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amx_operand_encoding() {
        let op = amx_operand(0, 0x1000);
        assert_eq!(op & 0x00FFFFFFFFFFFFFF, 0x1000);
        assert_eq!(op >> 56, 0);

        let op = amx_operand(3, 0xDEAD_BEEF);
        assert_eq!(op >> 56, 3);
        assert_eq!(op & 0x00FFFFFFFFFFFFFF, 0xDEAD_BEEF);
    }

    #[test]
    fn test_amx_msr_encoding_has_msr_prefix() {
        let insn = encode_amx_msr(AmxOp::Enable, 0);
        // Top 10 bits should be MSR prefix: 0b1101010100
        assert_eq!(insn >> 22, 0b1101010100);
    }

    #[test]
    fn test_amx_msr_encoding_rt_field() {
        for rt in 0..31u8 {
            let insn = encode_amx_msr(AmxOp::Ldx, rt);
            assert_eq!(insn & 0x1F, rt as u32);
        }
    }

    #[test]
    fn test_apple_amx_gemm_eligible() {
        assert!(apple_amx_gemm_eligible(32, 32, 1));
        assert!(apple_amx_gemm_eligible(64, 64, 128));
        assert!(!apple_amx_gemm_eligible(16, 32, 1)); // M too small
        assert!(!apple_amx_gemm_eligible(32, 16, 1)); // N too small
        assert!(!apple_amx_gemm_eligible(48, 32, 1)); // M not multiple of 32
        assert!(!apple_amx_gemm_eligible(32, 32, 0)); // K = 0
    }

    #[test]
    fn test_amx_tile_constants() {
        assert_eq!(AMX_TILE_M, 32);
        assert_eq!(AMX_TILE_N, 32);
        assert_eq!(AMX_REG_F32, 16); // 64 bytes / 4
    }

    #[test]
    fn test_amx_fma32_operand_encoding() {
        // z_row=0, x_reg=0, y_reg=0 → only bit[1]=1 set
        let op = amx_fma32_operand(0, 0, 0);
        assert_eq!(op & 0x3, 1, "accumulate bit must be set");
        assert_eq!(op >> 56, 0, "z_row=0");

        // z_row=5, x_reg=2, y_reg=3
        let op = amx_fma32_operand(5, 2, 3);
        assert_eq!(op >> 56, 5, "z_row field");
        assert_eq!((op >> 20) & 0x7, 3, "y_reg field");
        assert_eq!((op >> 2) & 0x7, 2, "x_reg field");
        assert_eq!(op & 0x1, 1, "accumulate bit");

        // z_row=63 (max), x_reg=7, y_reg=7
        let op = amx_fma32_operand(63, 7, 7);
        assert_eq!(op >> 56, 63);
        assert_eq!((op >> 20) & 0x7, 7);
        assert_eq!((op >> 2) & 0x7, 7);
    }

    #[test]
    fn test_emit_apple_amx_f32_gemm_produces_code() {
        let words = emit_apple_amx_f32_gemm(32, 32, 1);
        assert!(!words.is_empty(), "should emit instructions");
        // All instructions must be 4-byte aligned (trivially true for Vec<u32>)
        // Last instruction must be ret (0xD65F03C0)
        assert_eq!(*words.last().unwrap(), 0xD65F03C0u32, "must end with ret");
    }

    #[test]
    fn test_emit_apple_amx_f32_gemm_64x64_k4() {
        let words = emit_apple_amx_f32_gemm(64, 64, 4);
        assert!(!words.is_empty());
        assert_eq!(*words.last().unwrap(), 0xD65F03C0u32);
        // 64×64 = 4 tiles, each tile has more instructions than 32×32×1
        let words_32 = emit_apple_amx_f32_gemm(32, 32, 1);
        assert!(words.len() > words_32.len(), "larger GEMM should emit more instructions");
    }

    #[test]
    fn test_emit_apple_amx_f32_gemm_contains_enable_disable() {
        let words = emit_apple_amx_f32_gemm(32, 32, 1);
        // AMX Enable and Disable must appear (encoded via encode_amx_msr)
        let enable_insn = encode_amx_msr(AmxOp::Enable, 8);
        let disable_insn = encode_amx_msr(AmxOp::Disable, 8);
        assert!(words.contains(&enable_insn), "must contain AMX Enable");
        assert!(words.contains(&disable_insn), "must contain AMX Disable");
    }

    #[test]
    fn test_emit_apple_amx_f32_gemm_contains_ldx_ldy_stz() {
        let words = emit_apple_amx_f32_gemm(32, 32, 1);
        let ldx_insn = encode_amx_msr(AmxOp::Ldx, 4);
        let ldy_insn = encode_amx_msr(AmxOp::Ldy, 5);
        let stz_insn = encode_amx_msr(AmxOp::Stz, 5);
        assert!(words.contains(&ldx_insn), "must contain LDX");
        assert!(words.contains(&ldy_insn), "must contain LDY");
        assert!(words.contains(&stz_insn), "must contain STZ");
    }
}

/// Encode the FMA32 operand word.
///
/// FMA32 Xn layout (from Dougall Johnson's reverse-engineering):
///   bits [63:56] = Z register row index (0..63)
///   bits [27:20] = Y register index (0..7)
///   bits [9:2]   = X register index (0..7)
///   bit  [1]     = 1: accumulate into Z (don't clear before FMA)
#[inline]
pub const fn amx_fma32_operand(z_row: u8, x_reg: u8, y_reg: u8) -> u64 {
    ((z_row as u64) << 56)
        | ((y_reg as u64 & 0x7) << 20)
        | ((x_reg as u64 & 0x7) << 2)
        | 1u64
}

/// Emit a complete Apple AMX f32 GEMM kernel as AArch64 instruction words.
///
/// Computes C[M×N] += A[M×K] × B[K×N] (row-major f32).
///
/// # ABI (matches NEON GEMM convention)
/// - x0 = A pointer (M×K row-major f32)
/// - x1 = B pointer (K×N row-major f32)
/// - x2 = C pointer (M×N row-major f32, output)
///
/// # Constraints
/// - M and N must be multiples of 32; K >= 1
///
/// # Strategy
/// For each 32×32 output tile (mt, nt):
///   1. ClearZ (zero Z[0..63])
///   2. For each k-step ki:
///      - Load B[ki][nt*32..nt*32+16] → Y[0], B[ki][nt*32+16..nt*32+32] → Y[1]
///      - For each A row r in [mt*32..mt*32+32]:
///          Load A[r][ki..ki+16] → X[0]
///          FMA32: Z[r]    += X[0] ⊗ Y[0]  (first 16 output cols)
///          FMA32: Z[r+32] += X[0] ⊗ Y[1]  (second 16 output cols)
///   3. Stz Z[0..31] → C[mt*32..mt*32+32][nt*32..nt*32+16]
///      Stz Z[32..63] → C[mt*32..mt*32+32][nt*32+16..nt*32+32]
pub fn emit_apple_amx_f32_gemm(m: usize, n: usize, k: usize) -> Vec<u32> {
    assert!(m % AMX_TILE_M == 0 && n % AMX_TILE_N == 0 && k >= 1,
        "AMX GEMM: m={m} n={n} k={k} must satisfy m%32==0, n%32==0, k>=1");

    let m_tiles = m / AMX_TILE_M;
    let n_tiles = n / AMX_TILE_N;
    let a_row_bytes = k * 4;
    let b_row_bytes = n * 4;
    let c_row_bytes = n * 4;

    let mut code: Vec<u32> = Vec::new();

    // ── Encoding helpers ──────────────────────────────────────────────

    // movz xN, #imm16
    let movz = |reg: u8, imm: u16| -> u32 {
        0xD2800000u32 | ((imm as u32) << 5) | (reg as u32 & 0x1F)
    };
    // movk xN, #imm16, lsl #(shift) where shift in {0,16,32,48}
    let movk = |reg: u8, imm: u16, shift: u8| -> u32 {
        let hw = (shift / 16) as u32;
        0xF2800000u32 | (hw << 21) | ((imm as u32) << 5) | (reg as u32 & 0x1F)
    };
    // mov xd, xs  (ORR xd, xzr, xs)
    let mov_reg = |dst: u8, src: u8| -> u32 {
        0xAA0003E0u32 | ((src as u32 & 0x1F) << 16) | (dst as u32 & 0x1F)
    };
    // add xd, xn, xm
    let add_reg = |dst: u8, n_r: u8, m_r: u8| -> u32 {
        0x8B000000u32 | ((m_r as u32 & 0x1F) << 16) | ((n_r as u32 & 0x1F) << 5) | (dst as u32 & 0x1F)
    };
    // add xd, xn, #imm12
    let add_imm12 = |dst: u8, n_r: u8, imm: u32| -> u32 {
        debug_assert!(imm < 4096, "add_imm12: imm={imm} >= 4096");
        0x91000000u32 | (imm << 10) | ((n_r as u32 & 0x1F) << 5) | (dst as u32 & 0x1F)
    };

    // Load 64-bit immediate into xN (movz + up to 3 movk)
    let load_u64 = |reg: u8, val: u64, out: &mut Vec<u32>| {
        out.push(movz(reg, (val & 0xFFFF) as u16));
        if (val >> 16) & 0xFFFF != 0 { out.push(movk(reg, ((val >> 16) & 0xFFFF) as u16, 16)); }
        if (val >> 32) & 0xFFFF != 0 { out.push(movk(reg, ((val >> 32) & 0xFFFF) as u16, 32)); }
        if (val >> 48) & 0xFFFF != 0 { out.push(movk(reg, ((val >> 48) & 0xFFFF) as u16, 48)); }
    };

    // ── AMX ENABLE ───────────────────────────────────────────────────
    load_u64(8, 0, &mut code);
    code.push(encode_amx_msr(AmxOp::Enable, 8));

    // ── Tile loops ───────────────────────────────────────────────────
    for mt in 0..m_tiles {
        for nt in 0..n_tiles {
            // ClearZ: zero all Z registers for this tile
            load_u64(8, 0, &mut code);
            code.push(encode_amx_msr(AmxOp::ClearZ, 8));

            // K-loop (unrolled — k is known at JIT time)
            for ki in 0..k {
                // ── Load B row ki, columns [nt*32..nt*32+32] into Y[0], Y[1] ──
                // B is row-major K×N: B[ki][col] at B_base + ki*b_row_bytes + col*4
                let b_base_off = ki * b_row_bytes + nt * AMX_TILE_N * 4;

                // x5 = B_base + b_base_off  (absolute address, reg_idx=0 → high bits=0)
                load_u64(9, b_base_off as u64, &mut code);
                code.push(add_reg(5, 1, 9));
                // Y[0] = mem[x5]  (reg_idx=0, bits[63:56]=0, already in x5)
                code.push(encode_amx_msr(AmxOp::Ldy, 5));

                // Y[1] = mem[x5 + 64]  (reg_idx=1 → bits[63:56]=1)
                // x6 = x5 + 64
                code.push(add_imm12(6, 5, 64));
                // OR reg_idx=1 into bits[63:48]: movk x6, #0x0001, lsl #48
                code.push(movk(6, 0x0001, 48));
                code.push(encode_amx_msr(AmxOp::Ldy, 6));

                // ── For each A row in this tile ──
                for row in 0u8..32u8 {
                    let a_row_idx = mt * AMX_TILE_M + row as usize;
                    // A[a_row_idx][ki]: A_base + a_row_idx*a_row_bytes + ki*4
                    let a_off = a_row_idx * a_row_bytes + ki * 4;

                    // x4 = A_base + a_off  (reg_idx=0 → bits[63:56]=0)
                    load_u64(9, a_off as u64, &mut code);
                    code.push(add_reg(4, 0, 9));
                    // X[0] = mem[x4]
                    code.push(encode_amx_msr(AmxOp::Ldx, 4));

                    // FMA32: Z[row] += X[0] ⊗ Y[0]  (first 16 output cols)
                    load_u64(8, amx_fma32_operand(row, 0, 0), &mut code);
                    code.push(encode_amx_msr(AmxOp::Fma32, 8));

                    // FMA32: Z[row+32] += X[0] ⊗ Y[1]  (second 16 output cols)
                    load_u64(8, amx_fma32_operand(row + 32, 0, 1), &mut code);
                    code.push(encode_amx_msr(AmxOp::Fma32, 8));
                }
            }

            // ── Store Z[0..63] to C tile ──────────────────────────────
            // C tile base: C_base + mt*32*c_row_bytes + nt*32*4
            let c_tile_base = mt * AMX_TILE_M * c_row_bytes + nt * AMX_TILE_N * 4;
            load_u64(9, c_tile_base as u64, &mut code);
            code.push(add_reg(6, 2, 9)); // x6 = C_base + c_tile_base

            for row in 0u8..32u8 {
                let c_row_off = row as usize * c_row_bytes;

                // STZ Z[row] → C[mt*32+row][nt*32..nt*32+16]
                // x5 = x6 + c_row_off  (address of first 16 f32 in this output row)
                load_u64(9, c_row_off as u64, &mut code);
                code.push(add_reg(5, 6, 9));
                // Encode reg_idx=row in bits[63:56] of x5
                // row < 32, so bits[63:48] = row (fits in low 6 bits of the 16-bit field)
                if row > 0 {
                    code.push(movk(5, row as u16, 48));
                }
                code.push(encode_amx_msr(AmxOp::Stz, 5));

                // STZ Z[row+32] → C[mt*32+row][nt*32+16..nt*32+32]
                let c_row_off2 = c_row_off + 64; // +64 bytes = +16 f32
                load_u64(9, c_row_off2 as u64, &mut code);
                code.push(add_reg(5, 6, 9));
                let z_row2 = row + 32;
                code.push(movk(5, z_row2 as u16, 48));
                code.push(encode_amx_msr(AmxOp::Stz, 5));
            }
        }
    }

    // ── AMX DISABLE ──────────────────────────────────────────────────
    load_u64(8, 0, &mut code);
    code.push(encode_amx_msr(AmxOp::Disable, 8));

    // ret
    code.push(0xD65F03C0u32);

    code
}
