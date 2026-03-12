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
}
