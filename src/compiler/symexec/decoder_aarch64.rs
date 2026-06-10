//! AArch64 instruction decoder for Scalar + SymExec binary symbolic execution.
//!
//! Decodes compiled `extern "C"` scalar functions on AArch64 by reading
//! fixed-width 32-bit instructions and feeding them to the SymbolicExecutor.

use super::engine::{SymbolicExecutor, SymExecError};
use super::sym_value::SymValue;
use crate::compiler::trace::{ScalarFnSignature, ScalarParam, TraceOp};
use std::collections::HashMap;
use crate::types::CompilerError;

const MAX_FN_BYTES: usize = 4096;
const MAX_INSTRUCTIONS: usize = 500;

// ---------------------------------------------------------------------------
// AArch64 decoded instruction
// ---------------------------------------------------------------------------

/// A single decoded AArch64 instruction.
#[derive(Debug, Clone)]
pub struct A64Insn {
    pub mnemonic: String,
    pub operands: Vec<String>,
    pub addr: u64,
    pub len: u32, // always 4
    /// Branch target (absolute address) if this is a branch instruction.
    pub branch_target: Option<u64>,
    /// True if this is a conditional branch (b.cond, cbz, cbnz, tbz, tbnz).
    pub is_cond_branch: bool,
    /// True if this is an unconditional branch (b, br).
    pub is_uncond_branch: bool,
    /// True if this is a return (ret).
    pub is_return: bool,
    /// True if this is a call (bl, blr).
    pub is_call: bool,
}

// ---------------------------------------------------------------------------
// Bit extraction helpers
// ---------------------------------------------------------------------------

fn bits(word: u32, hi: u32, lo: u32) -> u32 {
    (word >> lo) & ((1 << (hi - lo + 1)) - 1)
}

fn sign_extend(val: u32, bit_width: u32) -> i64 {
    let shift = 64 - bit_width;
    ((val as i64) << shift) >> shift
}

fn sreg(idx: u32) -> String {
    format!("s{idx}")
}

fn dreg(idx: u32) -> String {
    format!("d{idx}")
}

fn xreg(idx: u32) -> String {
    if idx == 31 { "sp".to_string() } else { format!("x{idx}") }
}

fn wreg(idx: u32) -> String {
    if idx == 31 { "wsp".to_string() } else { format!("w{idx}") }
}

/// Float register name based on type field (0=S, 1=D).
fn ftype_reg(idx: u32, ftype: u32) -> String {
    match ftype {
        0 => sreg(idx),
        1 => dreg(idx),
        _ => format!("v{idx}"),
    }
}

// ---------------------------------------------------------------------------
// Condition code names
// ---------------------------------------------------------------------------

fn cond_name(cond: u32) -> &'static str {
    match cond & 0xF {
        0x0 => "eq", 0x1 => "ne", 0x2 => "hs", 0x3 => "lo",
        0x4 => "mi", 0x5 => "pl", 0x6 => "vs", 0x7 => "vc",
        0x8 => "hi", 0x9 => "ls", 0xA => "ge", 0xB => "lt",
        0xC => "gt", 0xD => "le", 0xE => "al", _ => "nv",
    }
}

// ---------------------------------------------------------------------------
// Main decoder
// ---------------------------------------------------------------------------

/// Decode a single 32-bit AArch64 instruction at the given PC.
pub fn decode_one(word: u32, pc: u64) -> A64Insn {
    let mut insn = A64Insn {
        mnemonic: String::new(),
        operands: Vec::new(),
        addr: pc,
        len: 4,
        branch_target: None,
        is_cond_branch: false,
        is_uncond_branch: false,
        is_return: false,
        is_call: false,
    };

    let op0 = bits(word, 31, 25);

    // Branches: [28:26] = 101
    if bits(word, 28, 26) == 0b101 {
        decode_branch(word, pc, &mut insn);
        return insn;
    }

    // Data processing — float: [28:25] = 0b0111 with bit24=1 (FP/SIMD)
    // More precisely: bits [27:24] = 0b1110 or 0b1111 for FP scalar
    let op1 = bits(word, 27, 24);
    if op1 == 0b1110 || op1 == 0b1111 {
        decode_fp(word, pc, &mut insn);
        return insn;
    }

    // Load/store: bits [27:25] = 0b1x0 (where x can be 0 or 1)
    let ls_group = bits(word, 27, 25);
    if ls_group == 0b100 || ls_group == 0b110 {
        decode_load_store(word, pc, &mut insn);
        return insn;
    }

    // Data processing — immediate: [28:26] = 100
    if bits(word, 28, 26) == 0b100 {
        decode_dp_imm(word, &mut insn);
        return insn;
    }

    // Data processing — register: [27:25] = 0b101 already caught by branch
    // [28:25] = 0b0101 for DP reg
    if bits(word, 28, 25) == 0b0101 || bits(word, 28, 25) == 0b1101 {
        decode_dp_reg(word, &mut insn);
        return insn;
    }

    // Fallback: unknown instruction
    insn.mnemonic = format!(".word 0x{word:08x}");
    insn
}

// ---------------------------------------------------------------------------
// Branch decoding
// ---------------------------------------------------------------------------

fn decode_branch(word: u32, pc: u64, insn: &mut A64Insn) {
    let op = bits(word, 31, 29);

    match op {
        // b.cond: 0101_0100 imm19 0 cond
        _ if bits(word, 31, 24) == 0b0101_0100 => {
            let cond = bits(word, 3, 0);
            let imm19 = bits(word, 23, 5);
            let offset = sign_extend(imm19, 19) * 4;
            let target = (pc as i64 + offset) as u64;
            insn.mnemonic = format!("b.{}", cond_name(cond));
            insn.operands.push(format!("0x{target:x}"));
            insn.branch_target = Some(target);
            insn.is_cond_branch = true;
        }
        // cbz/cbnz: x011_010x
        _ if bits(word, 30, 25) == 0b01_1010 => {
            let sf = bits(word, 31, 31);
            let op_bit = bits(word, 24, 24);
            let imm19 = bits(word, 23, 5);
            let rt = bits(word, 4, 0);
            let offset = sign_extend(imm19, 19) * 4;
            let target = (pc as i64 + offset) as u64;
            insn.mnemonic = if op_bit == 0 { "cbz".into() } else { "cbnz".into() };
            insn.operands.push(if sf == 1 { xreg(rt) } else { wreg(rt) });
            insn.operands.push(format!("0x{target:x}"));
            insn.branch_target = Some(target);
            insn.is_cond_branch = true;
        }
        // tbz/tbnz: x011_011x
        _ if bits(word, 30, 25) == 0b01_1011 => {
            let op_bit = bits(word, 24, 24);
            let b5 = bits(word, 31, 31);
            let b40 = bits(word, 23, 19);
            let imm14 = bits(word, 18, 5);
            let rt = bits(word, 4, 0);
            let bit_pos = (b5 << 5) | b40;
            let offset = sign_extend(imm14, 14) * 4;
            let target = (pc as i64 + offset) as u64;
            insn.mnemonic = if op_bit == 0 { "tbz".into() } else { "tbnz".into() };
            insn.operands.push(xreg(rt));
            insn.operands.push(format!("#{bit_pos}"));
            insn.operands.push(format!("0x{target:x}"));
            insn.branch_target = Some(target);
            insn.is_cond_branch = true;
        }
        // b: 0001_01 imm26
        _ if bits(word, 31, 26) == 0b000101 => {
            let imm26 = bits(word, 25, 0);
            let offset = sign_extend(imm26, 26) * 4;
            let target = (pc as i64 + offset) as u64;
            insn.mnemonic = "b".into();
            insn.operands.push(format!("0x{target:x}"));
            insn.branch_target = Some(target);
            insn.is_uncond_branch = true;
        }
        // bl: 1001_01 imm26
        _ if bits(word, 31, 26) == 0b100101 => {
            let imm26 = bits(word, 25, 0);
            let offset = sign_extend(imm26, 26) * 4;
            let target = (pc as i64 + offset) as u64;
            insn.mnemonic = "bl".into();
            insn.operands.push(format!("0x{target:x}"));
            insn.branch_target = Some(target);
            insn.is_call = true;
        }
        // ret/br/blr: 1101_0110
        _ if bits(word, 31, 22) == 0b1101011000 || bits(word, 31, 22) == 0b1101011001
            || bits(word, 31, 22) == 0b1101011010 => {
            let opc = bits(word, 24, 21);
            let rn = bits(word, 9, 5);
            match opc {
                0b0000 => { // br
                    insn.mnemonic = "br".into();
                    insn.operands.push(xreg(rn));
                    insn.is_uncond_branch = true;
                }
                0b0001 => { // blr
                    insn.mnemonic = "blr".into();
                    insn.operands.push(xreg(rn));
                    insn.is_call = true;
                }
                0b0010 => { // ret
                    insn.mnemonic = "ret".into();
                    if rn != 30 { insn.operands.push(xreg(rn)); }
                    insn.is_return = true;
                }
                _ => {
                    insn.mnemonic = format!("branch_unknown_{opc}");
                }
            }
        }
        _ => {
            insn.mnemonic = format!(".branch 0x{word:08x}");
        }
    }
}

// ---------------------------------------------------------------------------
// Floating-point decoding
// ---------------------------------------------------------------------------

fn decode_fp(word: u32, pc: u64, insn: &mut A64Insn) {
    // FP data processing (2-source): 0001_1110_xx1x_xxxx_xxxx_10xx_xxxx_xxxx
    if bits(word, 31, 24) == 0b0001_1110 && bits(word, 21, 21) == 1 && bits(word, 11, 10) == 0b10 {
        let ftype = bits(word, 23, 22);
        let rm = bits(word, 20, 16);
        let opcode = bits(word, 15, 12);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mnem = match opcode {
            0b0000 => "fmul",
            0b0001 => "fdiv",
            0b0010 => "fadd",
            0b0011 => "fsub",
            0b0100 => "fmax",
            0b0101 => "fmin",
            0b0110 => "fmaxnm",
            0b0111 => "fminnm",
            _ => "fp2src_unknown",
        };
        insn.mnemonic = mnem.into();
        insn.operands.push(ftype_reg(rd, ftype));
        insn.operands.push(ftype_reg(rn, ftype));
        insn.operands.push(ftype_reg(rm, ftype));
        return;
    }

    // FP data processing (1-source): 0001_1110_xx1_00000_0xxxxx_xxxxx_xxxxx
    if bits(word, 31, 24) == 0b0001_1110 && bits(word, 21, 17) == 0b10000
        && bits(word, 14, 14) == 0
    {
        let ftype = bits(word, 23, 22);
        let opcode = bits(word, 20, 15);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mnem = match opcode {
            0b000000 => "fmov",
            0b000001 => "fabs",
            0b000010 => "fneg",
            0b000011 => "fsqrt",
            0b000100 => "fcvt", // S→D
            0b000101 => "fcvt", // D→S
            _ => "fp1src_unknown",
        };
        insn.mnemonic = mnem.into();
        // For fcvt, destination type differs from source
        if opcode == 0b000100 {
            insn.operands.push(dreg(rd));
            insn.operands.push(sreg(rn));
        } else if opcode == 0b000101 {
            insn.operands.push(sreg(rd));
            insn.operands.push(dreg(rn));
        } else {
            insn.operands.push(ftype_reg(rd, ftype));
            insn.operands.push(ftype_reg(rn, ftype));
        }
        return;
    }

    // FP compare: 0001_1110_xx1_xxxxx_0010_00_xxxxx_x0000
    if bits(word, 31, 24) == 0b0001_1110 && bits(word, 21, 21) == 1
        && bits(word, 13, 10) == 0b1000 && bits(word, 2, 0) == 0b000
    {
        let ftype = bits(word, 23, 22);
        let rm = bits(word, 20, 16);
        let rn = bits(word, 9, 5);
        let opc = bits(word, 4, 3);
        insn.mnemonic = if opc & 1 == 0 { "fcmp".into() } else { "fcmpe".into() };
        insn.operands.push(ftype_reg(rn, ftype));
        if bits(word, 4, 3) < 2 && rm == 0 && bits(word, 20, 16) == 0 {
            // fcmp with #0.0 has rm=0 and specific encoding
            if bits(word, 20, 16) == 0 && bits(word, 4, 4) == 0 {
                insn.operands.push(ftype_reg(rm, ftype));
            } else {
                insn.operands.push("#0.0".into());
            }
        } else {
            insn.operands.push(ftype_reg(rm, ftype));
        }
        return;
    }

    // FP conditional select: 0001_1110_xx1_xxxxx_xxxx_11_xxxxx_xxxxx
    if bits(word, 31, 24) == 0b0001_1110 && bits(word, 21, 21) == 1
        && bits(word, 11, 10) == 0b11
    {
        let ftype = bits(word, 23, 22);
        let rm = bits(word, 20, 16);
        let cond = bits(word, 15, 12);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        insn.mnemonic = "fcsel".into();
        insn.operands.push(ftype_reg(rd, ftype));
        insn.operands.push(ftype_reg(rn, ftype));
        insn.operands.push(ftype_reg(rm, ftype));
        insn.operands.push(cond_name(cond).into());
        return;
    }

    // FP data processing (3-source): 0001_1111_xx_x_xxxxx_x_xxxxx_xxxxx_xxxxx
    if bits(word, 31, 24) == 0b0001_1111 {
        let ftype = bits(word, 23, 22);
        let o1 = bits(word, 21, 21);
        let rm = bits(word, 20, 16);
        let o0 = bits(word, 15, 15);
        let ra = bits(word, 14, 10);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        let mnem = match (o1, o0) {
            (0, 0) => "fmadd",
            (0, 1) => "fmsub",
            (1, 0) => "fnmadd",
            (1, 1) => "fnmsub",
            _ => unreachable!(),
        };
        insn.mnemonic = mnem.into();
        insn.operands.push(ftype_reg(rd, ftype));
        insn.operands.push(ftype_reg(rn, ftype));
        insn.operands.push(ftype_reg(rm, ftype));
        insn.operands.push(ftype_reg(ra, ftype));
        return;
    }

    // Integer ↔ FP conversion: 0001_1110_xx1_xxxxx_000000_xxxxx_xxxxx
    if bits(word, 31, 24) == 0b0001_1110 && bits(word, 21, 21) == 1
        && bits(word, 15, 10) == 0b000000
    {
        let sf = bits(word, 31, 31);
        let ftype = bits(word, 23, 22);
        let rmode = bits(word, 20, 19);
        let opcode = bits(word, 18, 16);
        let rn = bits(word, 9, 5);
        let rd = bits(word, 4, 0);
        // scvtf/ucvtf: int→float
        if opcode == 0b010 || opcode == 0b011 {
            insn.mnemonic = if opcode == 0b010 { "scvtf".into() } else { "ucvtf".into() };
            insn.operands.push(ftype_reg(rd, ftype));
            insn.operands.push(if sf == 1 { xreg(rn) } else { wreg(rn) });
            return;
        }
        // fcvtzs/fcvtzu: float→int
        if opcode == 0b000 || opcode == 0b001 {
            insn.mnemonic = if rmode == 0b11 {
                if opcode == 0 { "fcvtzs".into() } else { "fcvtzu".into() }
            } else {
                format!("fcvt_rm{rmode}_{opcode}")
            };
            insn.operands.push(if sf == 1 { xreg(rd) } else { wreg(rd) });
            insn.operands.push(ftype_reg(rn, ftype));
            return;
        }
        // fmov between int and float regs
        if opcode == 0b110 || opcode == 0b111 {
            insn.mnemonic = "fmov".into();
            if opcode == 0b111 {
                // GP → FP
                insn.operands.push(ftype_reg(rd, ftype));
                insn.operands.push(if sf == 1 { xreg(rn) } else { wreg(rn) });
            } else {
                // FP → GP
                insn.operands.push(if sf == 1 { xreg(rd) } else { wreg(rd) });
                insn.operands.push(ftype_reg(rn, ftype));
            }
            return;
        }
    }

    // Fallback
    insn.mnemonic = format!(".fp 0x{word:08x}");
}


// ---------------------------------------------------------------------------
// Load/store decoding
// ---------------------------------------------------------------------------

fn decode_load_store(word: u32, pc: u64, insn: &mut A64Insn) {
    // LDR (literal, SIMD/FP): opc=xx 011 1 00 imm19 Rt
    // opc[1]=1 means SIMD/FP
    if bits(word, 29, 27) == 0b011 && bits(word, 26, 26) == 1 && bits(word, 25, 24) == 0b00 {
        let opc = bits(word, 31, 30);
        let imm19 = bits(word, 23, 5);
        let rt = bits(word, 4, 0);
        let offset = sign_extend(imm19, 19) * 4;
        let target = (pc as i64 + offset) as u64;
        insn.mnemonic = "ldr".into();
        insn.operands.push(match opc {
            0b00 => sreg(rt),
            0b01 => dreg(rt),
            _ => format!("q{rt}"),
        });
        insn.operands.push(format!("[pc, #0x{target:x}]"));
        // Store the literal address for constant pool resolution
        insn.branch_target = Some(target);
        return;
    }

    // LDR/STR (unsigned offset, SIMD/FP): xx 111 1 01 xx imm12 Rn Rt
    if bits(word, 29, 27) == 0b111 && bits(word, 26, 26) == 1 && bits(word, 25, 24) == 0b01 {
        let size = bits(word, 31, 30);
        let opc = bits(word, 23, 22);
        let imm12 = bits(word, 21, 10);
        let rn = bits(word, 9, 5);
        let rt = bits(word, 4, 0);
        let scale = if size == 0b00 { 4 } else if size == 0b01 { 8 } else { 16 };
        let offset = (imm12 as i64) * scale;
        let is_load = opc & 1 == 1;
        insn.mnemonic = if is_load { "ldr".into() } else { "str".into() };
        let reg = match size {
            0b00 => sreg(rt),
            0b01 => dreg(rt),
            _ => format!("q{rt}"),
        };
        let base = if rn == 31 { "sp".to_string() } else { format!("x{rn}") };
        if is_load {
            insn.operands.push(reg);
            insn.operands.push(format!("[{base}, #{offset}]"));
        } else {
            insn.operands.push(reg);
            insn.operands.push(format!("[{base}, #{offset}]"));
        }
        return;
    }

    // LDP/STP (SIMD/FP): opc 101 1 0xx imm7 Rt2 Rn Rt
    if bits(word, 29, 27) == 0b101 && bits(word, 26, 26) == 1 {
        let opc = bits(word, 31, 30);
        let is_load = bits(word, 22, 22) == 1;
        let imm7 = bits(word, 21, 15);
        let rt2 = bits(word, 14, 10);
        let rn = bits(word, 9, 5);
        let rt = bits(word, 4, 0);
        let scale = if opc == 0b00 { 4 } else if opc == 0b01 { 8 } else { 16 };
        let offset = sign_extend(imm7, 7) * scale;
        insn.mnemonic = if is_load { "ldp".into() } else { "stp".into() };
        let reg_fn = |idx: u32| -> String {
            match opc {
                0b00 => sreg(idx),
                0b01 => dreg(idx),
                _ => format!("q{idx}"),
            }
        };
        let base = if rn == 31 { "sp".to_string() } else { format!("x{rn}") };
        insn.operands.push(reg_fn(rt));
        insn.operands.push(reg_fn(rt2));
        insn.operands.push(format!("[{base}, #{offset}]"));
        return;
    }

    // LDR/STR (unsigned offset, integer): xx 111 0 01 xx imm12 Rn Rt
    if bits(word, 29, 27) == 0b111 && bits(word, 26, 26) == 0 && bits(word, 25, 24) == 0b01 {
        let size = bits(word, 31, 30);
        let opc = bits(word, 23, 22);
        let rn = bits(word, 9, 5);
        let rt = bits(word, 4, 0);
        let is_load = opc & 1 == 1;
        insn.mnemonic = if is_load { "ldr_int".into() } else { "str_int".into() };
        insn.operands.push(if size >= 0b11 { xreg(rt) } else { wreg(rt) });
        return;
    }

    // LDP/STP (integer): opc 101 0 0xx imm7 Rt2 Rn Rt
    if bits(word, 29, 27) == 0b101 && bits(word, 26, 26) == 0 {
        let is_load = bits(word, 22, 22) == 1;
        insn.mnemonic = if is_load { "ldp_int".into() } else { "stp_int".into() };
        return;
    }

    // LDR/STR (register offset): xx 111 x 00 xx 1 Rm opt S 10 Rn Rt
    if bits(word, 29, 27) == 0b111 && bits(word, 25, 24) == 0b00
        && bits(word, 21, 21) == 1 && bits(word, 11, 10) == 0b10
    {
        let is_fp = bits(word, 26, 26) == 1;
        let size = bits(word, 31, 30);
        let opc = bits(word, 23, 22);
        let rm = bits(word, 20, 16);
        let rn = bits(word, 9, 5);
        let rt = bits(word, 4, 0);
        let is_load = opc & 1 == 1;
        if is_fp {
            insn.mnemonic = if is_load { "ldr".into() } else { "str".into() };
            let reg = match size { 0b00 => sreg(rt), 0b01 => dreg(rt), _ => format!("q{rt}") };
            let base = if rn == 31 { "sp".to_string() } else { format!("x{rn}") };
            insn.operands.push(reg);
            insn.operands.push(format!("[{base}, x{rm}]"));
        } else {
            insn.mnemonic = if is_load { "ldr_int".into() } else { "str_int".into() };
            insn.operands.push(if size >= 0b11 { xreg(rt) } else { wreg(rt) });
        }
        return;
    }

    // Fallback: treat as integer load/store (no-op for symexec)
    insn.mnemonic = "ls_unknown".into();
}

// ---------------------------------------------------------------------------
// Data processing — immediate
// ---------------------------------------------------------------------------

fn decode_dp_imm(word: u32, insn: &mut A64Insn) {
    let op0 = bits(word, 25, 23);
    let sf = bits(word, 31, 31);
    let rd = bits(word, 4, 0);
    let rn = bits(word, 9, 5);

    match op0 {
        // MOVZ/MOVN/MOVK: 10x
        0b101 => {
            let opc = bits(word, 30, 29);
            insn.mnemonic = match opc { 0b00 => "movn", 0b10 => "movz", 0b11 => "movk", _ => "mov_imm" }.into();
            insn.operands.push(if sf == 1 { xreg(rd) } else { wreg(rd) });
        }
        // ADD/SUB immediate: 00x
        0b001 => {
            let op = bits(word, 30, 30);
            let s = bits(word, 29, 29);
            if s == 1 && rd == 31 {
                insn.mnemonic = "cmp".into();
                insn.operands.push(if sf == 1 { xreg(rn) } else { wreg(rn) });
            } else {
                insn.mnemonic = if op == 0 { "add".into() } else { "sub".into() };
                insn.operands.push(if sf == 1 { xreg(rd) } else { wreg(rd) });
            }
        }
        // ADRP/ADR: 0b000
        0b000 => {
            let op = bits(word, 31, 31);
            insn.mnemonic = if op == 1 { "adrp".into() } else { "adr".into() };
            insn.operands.push(xreg(rd));
        }
        // Logical immediate
        0b010 => {
            let opc = bits(word, 30, 29);
            insn.mnemonic = match opc {
                0b00 => "and", 0b01 => "orr", 0b10 => "eor", 0b11 => "tst",
                _ => "logic_imm",
            }.into();
            insn.operands.push(if sf == 1 { xreg(rd) } else { wreg(rd) });
        }
        _ => {
            insn.mnemonic = "dp_imm_unknown".into();
        }
    }
}

// ---------------------------------------------------------------------------
// Data processing — register
// ---------------------------------------------------------------------------

fn decode_dp_reg(word: u32, insn: &mut A64Insn) {
    let sf = bits(word, 31, 31);
    let rd = bits(word, 4, 0);
    let rn = bits(word, 9, 5);
    let rm = bits(word, 20, 16);

    // Conditional select: x0_11010100_xxxxx_xxxx_0x_xxxxx_xxxxx
    if bits(word, 29, 21) == 0b011010100 {
        let op = bits(word, 30, 30);
        let op2 = bits(word, 10, 10);
        let cond = bits(word, 15, 12);
        insn.mnemonic = match (op, op2) {
            (0, 0) => "csel",
            (0, 1) => "csinc",
            (1, 0) => "csinv",
            (1, 1) => "csneg",
            _ => "csel_unknown",
        }.into();
        insn.operands.push(if sf == 1 { xreg(rd) } else { wreg(rd) });
        insn.operands.push(if sf == 1 { xreg(rn) } else { wreg(rn) });
        insn.operands.push(if sf == 1 { xreg(rm) } else { wreg(rm) });
        insn.operands.push(cond_name(cond).into());
        return;
    }

    // Shifted register (add/sub/and/orr/eor)
    let op_kind = bits(word, 30, 29);
    let op54 = bits(word, 24, 21);
    insn.mnemonic = match op_kind {
        0b00 => "add",
        0b01 => "sub",
        0b10 => "and",
        0b11 => "orr",
        _ => "dp_reg",
    }.into();
    insn.operands.push(if sf == 1 { xreg(rd) } else { wreg(rd) });
}


// ---------------------------------------------------------------------------
// Instruction iterator
// ---------------------------------------------------------------------------

/// Iterator over AArch64 instructions from a byte slice.
pub struct A64Iter<'a> {
    bytes: &'a [u8],
    offset: usize,
    base_addr: u64,
}

impl<'a> A64Iter<'a> {
    pub fn new(bytes: &'a [u8], base_addr: u64) -> Self {
        Self { bytes, offset: 0, base_addr }
    }
}

impl<'a> Iterator for A64Iter<'a> {
    type Item = A64Insn;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + 4 > self.bytes.len() {
            return None;
        }
        let word = u32::from_le_bytes([
            self.bytes[self.offset],
            self.bytes[self.offset + 1],
            self.bytes[self.offset + 2],
            self.bytes[self.offset + 3],
        ]);
        let pc = self.base_addr + self.offset as u64;
        self.offset += 4;
        Some(decode_one(word, pc))
    }
}

// ---------------------------------------------------------------------------
// Pointer parameter map (AAPCS64)
// ---------------------------------------------------------------------------

/// AAPCS64: integer/pointer args go to x0-x7, float args to s0-s7/d0-d7.
#[cfg(target_arch = "aarch64")]
const INT_ARG_REGS: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Tracks which GP registers hold input vs output pointers.
#[cfg(target_arch = "aarch64")]
struct PtrParamMap {
    /// GP register index → input ordinal.
    input_regs: HashMap<u32, usize>,
    /// GP register indices holding output pointers.
    output_regs: Vec<u32>,
}

#[cfg(target_arch = "aarch64")]
impl PtrParamMap {
    fn propagate_mov(&mut self, dst: u32, src: u32) {
        if let Some(&ord) = self.input_regs.get(&src) {
            self.input_regs.insert(dst, ord);
        }
        if self.output_regs.contains(&src) && !self.output_regs.contains(&dst) {
            self.output_regs.push(dst);
        }
    }

    fn is_input(&self, reg: u32) -> Option<usize> {
        self.input_regs.get(&reg).copied()
    }

    fn is_output(&self, reg: u32) -> bool {
        self.output_regs.contains(&reg)
    }
}

#[cfg(target_arch = "aarch64")]
fn build_ptr_map(sig: &ScalarFnSignature) -> PtrParamMap {
    let mut input_regs = HashMap::new();
    let mut output_regs = Vec::new();
    let mut int_reg_idx = 0usize;
    let mut input_ordinal = 0usize;

    for param in &sig.params {
        match param {
            ScalarParam::InputPtr | ScalarParam::WeightPtr => {
                if int_reg_idx < 8 {
                    input_regs.insert(INT_ARG_REGS[int_reg_idx], input_ordinal);
                    input_ordinal += 1;
                }
                int_reg_idx += 1;
            }
            ScalarParam::OutputPtr => {
                if int_reg_idx < 8 {
                    output_regs.push(INT_ARG_REGS[int_reg_idx]);
                }
                int_reg_idx += 1;
            }
            ScalarParam::Dim(_) => {
                int_reg_idx += 1;
            }
            ScalarParam::Scalar(_) => {
                // Float params go to s0-s7, don't consume integer regs.
            }
        }
    }

    PtrParamMap { input_regs, output_regs }
}

fn count_inputs(sig: &ScalarFnSignature) -> usize {
    sig.params
        .iter()
        .filter(|p| matches!(p, ScalarParam::InputPtr | ScalarParam::WeightPtr))
        .count()
}

// ---------------------------------------------------------------------------
// Call target resolution (AArch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn resolve_call_target_a64(target_addr: u64) -> String {
    // Try dladdr to resolve symbol name.
    #[cfg(unix)]
    {
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::dladdr(target_addr as *const libc::c_void, &mut info) };
        if ret != 0 && !info.dli_sname.is_null() {
            let name = unsafe { std::ffi::CStr::from_ptr(info.dli_sname) };
            if let Ok(s) = name.to_str() {
                return s.to_string();
            }
        }
    }

    // Fallback: try matching against known libm function addresses.
    extern "C" {
        fn expf(x: f32) -> f32;
        fn tanhf(x: f32) -> f32;
        fn logf(x: f32) -> f32;
        fn sqrtf(x: f32) -> f32;
        fn fabsf(x: f32) -> f32;
    }

    let known: &[(*const (), &str)] = &[
        (expf as *const (), "expf"),
        (tanhf as *const (), "tanhf"),
        (logf as *const (), "logf"),
        (sqrtf as *const (), "sqrtf"),
        (fabsf as *const (), "fabsf"),
    ];

    for &(addr, name) in known {
        if target_addr == addr as u64 {
            return name.to_string();
        }
    }

    format!("0x{target_addr:x}")
}

// ---------------------------------------------------------------------------
// Helper: extract base register index from memory operand string
// ---------------------------------------------------------------------------

/// Parse `[x0, #16]` or `[sp, #-8]` → base register index (0-31, 31=sp).
fn parse_base_reg(mem_op: &str) -> Option<u32> {
    let inner = mem_op.strip_prefix('[')?.split(|c| c == ',' || c == ']').next()?;
    let base = inner.trim();
    if base == "sp" {
        Some(31)
    } else if let Some(num) = base.strip_prefix('x') {
        num.parse().ok()
    } else {
        None
    }
}

/// Check if a register name is a float register (s/d/q/v prefix).
fn is_float_reg(name: &str) -> bool {
    let first = name.as_bytes().first().copied().unwrap_or(0);
    matches!(first, b's' | b'd' | b'q' | b'v')
        && name.len() > 1
        && name[1..].chars().all(|c| c.is_ascii_digit())
}

// ---------------------------------------------------------------------------
// Main analysis: analyze_scalar_fn (AArch64)
// ---------------------------------------------------------------------------

/// Disassemble an AArch64 scalar function and symbolically execute it.
///
/// # Safety
/// `fn_ptr` must point to a valid, compiled `extern "C"` function.
#[cfg(target_arch = "aarch64")]
pub fn analyze_scalar_fn(
    fn_ptr: *const u8,
    sig: &ScalarFnSignature,
) -> Result<Vec<TraceOp>, SymExecError> {
    if fn_ptr.is_null() {
        return Err(SymExecError::DisassemblyFailed("null function pointer".into()));
    }

    let n_inputs = count_inputs(sig);
    let mut ptr_map = build_ptr_map(sig);
    let mut executor = SymbolicExecutor::new(n_inputs, 0);
    let mut last_output_value: Option<SymValue> = None;

    let bytes = unsafe { std::slice::from_raw_parts(fn_ptr, MAX_FN_BYTES) };
    let base_addr = fn_ptr as u64;
    let iter = A64Iter::new(bytes, base_addr);
    let mut instr_count = 0;

    for insn in iter {
        if instr_count >= MAX_INSTRUCTIONS {
            break;
        }
        instr_count += 1;

        // Stop at ret.
        if insn.is_return {
            break;
        }

        // Backward branch → end of first loop iteration.
        if let Some(target) = insn.branch_target {
            if !insn.is_call && target <= insn.addr {
                break;
            }
            // Forward branch (not a call) → skip.
            if !insn.is_call {
                continue;
            }
        }

        // Handle bl (call) instructions.
        if insn.is_call {
            if let Some(target) = insn.branch_target {
                let name = resolve_call_target_a64(target);
                executor.step("bl", &[&name])?;
            } else if insn.mnemonic == "blr" {
                executor.step("bl", &["unknown"])?;
            }
            continue;
        }

        let mnem = &insn.mnemonic;

        // PC-relative literal load → constant pool.
        if mnem == "ldr" && insn.operands.len() == 2
            && insn.operands[1].contains("pc,")
        {
            if let Some(lit_addr) = insn.branch_target {
                let dst = &insn.operands[0];
                if is_float_reg(dst) {
                    if !executor.has_constant(lit_addr) {
                        let value = unsafe { *(lit_addr as *const f32) };
                        executor.register_constant(lit_addr, value);
                    }
                    let value = unsafe { *(lit_addr as *const f32) };
                    executor.set(dst, SymValue::Const(value as f64));
                    continue;
                }
            }
        }

        // Float load from input pointer: ldr s0, [x0, #offset]
        if mnem == "ldr" && insn.operands.len() == 2 && is_float_reg(&insn.operands[0]) {
            let mem = &insn.operands[1];
            if let Some(base_idx) = parse_base_reg(mem) {
                if let Some(input_ord) = ptr_map.is_input(base_idx) {
                    executor.set(&insn.operands[0], SymValue::Param(input_ord));
                    continue;
                }
            }
        }

        // Float store to output pointer: str s0, [x8, #offset]
        if mnem == "str" && insn.operands.len() == 2 && is_float_reg(&insn.operands[0]) {
            let mem = &insn.operands[1];
            if let Some(base_idx) = parse_base_reg(mem) {
                if ptr_map.is_output(base_idx) {
                    last_output_value = Some(executor.get_value(&insn.operands[0]));
                    continue;
                }
            }
        }

        // Track GP register moves for pointer aliasing.
        // AArch64: mov x15, x0 is encoded as orr x15, xzr, x0
        if (mnem == "mov" || mnem == "orr") && insn.operands.len() >= 2 {
            let dst_str = &insn.operands[0];
            // SAFETY: guarded by operands.len() >= 2 above
            let src_str = insn.operands.last().expect("operands.len() >= 2");
            if let (Some(d), Some(s)) = (
                dst_str.strip_prefix('x').and_then(|n| n.parse::<u32>().ok()),
                src_str.strip_prefix('x').and_then(|n| n.parse::<u32>().ok()),
            ) {
                ptr_map.propagate_mov(d, s);
            }
        }

        // Skip integer-only instructions.
        if is_integer_mnemonic(mnem) {
            continue;
        }

        // General case: feed to symbolic executor.
        let op_refs: Vec<&str> = insn.operands.iter().map(|s| s.as_str()).collect();
        executor.step(mnem, &op_refs)?;
    }

    // For void functions that write results via output pointers.
    if let Some(output_val) = last_output_value {
        executor.set("s0", output_val);
    }

    executor.extract_trace()
}

/// Integer-only mnemonics that don't affect float symbolic state.
fn is_integer_mnemonic(m: &str) -> bool {
    matches!(
        m,
        "mov" | "movz" | "movk" | "movn"
            | "add" | "sub" | "mul" | "sdiv" | "udiv"
            | "and" | "orr" | "eor" | "tst"
            | "lsl" | "lsr" | "asr"
            | "cmp" | "cmn"
            | "adrp" | "adr"
            | "sxtw" | "uxtw"
            | "nop"
            | "stp_int" | "ldp_int" | "str_int" | "ldr_int"
            | "csel" | "csinc" | "csinv" | "csneg"
            | "ls_unknown" | "dp_imm_unknown" | "dp_reg"
    )
}

// ---------------------------------------------------------------------------
// Structured analysis (CFG-based)
// ---------------------------------------------------------------------------

/// Analyze a scalar function using CFG + loop detection.
#[cfg(target_arch = "aarch64")]
pub fn analyze_scalar_fn_structured(
    fn_ptr: *const u8,
    sig: &ScalarFnSignature,
) -> Result<Option<super::loop_analyzer::MultiPassAnalysis>, SymExecError> {
    use super::cfg::find_loops;
    use super::loop_analyzer::{analyze_single_loop, analyze_nested_loops, combine_passes};

    if fn_ptr.is_null() {
        return Err(SymExecError::DisassemblyFailed("null function pointer".into()));
    }

    let cfg = match build_cfg_from_fn_aarch64(fn_ptr, MAX_FN_BYTES) {
        Ok(c) => c,
        Err(e) => return Err(SymExecError::DisassemblyFailed(e)),
    };

    let forest = find_loops(&cfg);
    if forest.loops.is_empty() {
        return Ok(None);
    }

    let n_inputs = count_inputs(sig);
    let executor = SymbolicExecutor::new(n_inputs, 0);

    // Try nested loop analysis first (GEMM detection).
    if let Some(nested) = analyze_nested_loops(&forest, &cfg, &executor) {
        return Ok(Some(super::loop_analyzer::MultiPassAnalysis {
            loop_traces: nested.inner_trace.into_iter().collect(),
            pattern: nested.pattern,
            num_loops: forest.loops.len(),
        }));
    }

    // Flat analysis: analyze each top-level loop.
    let mut traces = Vec::new();
    for &loop_idx in &forest.top_level {
        let loop_info = &forest.loops[loop_idx];
        match analyze_single_loop(loop_info, &cfg, &executor) {
            Ok(trace) => {
                if !trace.reductions.is_empty() {
                    traces.push(trace);
                }
            }
            Err(_) => continue,
        }
    }

    if traces.is_empty() {
        return Ok(None);
    }

    let pattern = match combine_passes(&traces) {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };

    Ok(Some(super::loop_analyzer::MultiPassAnalysis {
        loop_traces: traces,
        pattern,
        num_loops: forest.loops.len(),
    }))
}

// ---------------------------------------------------------------------------
// CFG construction for AArch64
// ---------------------------------------------------------------------------

/// Build a control-flow graph from an AArch64 function.
#[cfg(target_arch = "aarch64")]
pub fn build_cfg_from_fn_aarch64(
    fn_ptr: *const u8,
    max_bytes: usize,
) -> Result<super::cfg::ControlFlowGraph, CompilerError> {
    use std::collections::{BTreeMap, BTreeSet};
    use super::cfg::*;

    if fn_ptr.is_null() {
        return Err("null function pointer".into());
    }

    let base_addr = fn_ptr as u64;
    let bytes = unsafe { std::slice::from_raw_parts(fn_ptr, max_bytes) };

    // Linear scan — collect all instructions and branch targets.
    let mut all_insns: Vec<A64Insn> = Vec::new();
    let mut branch_targets: BTreeSet<u64> = BTreeSet::new();
    let mut func_end: Option<u64> = None;

    for insn in A64Iter::new(bytes, base_addr) {
        let next_addr = insn.addr + 4;
        all_insns.push(insn.clone());

        if insn.is_return {
            func_end = Some(next_addr);
            break;
        }

        if let Some(target) = insn.branch_target {
            if !insn.is_call {
                branch_targets.insert(target);
                branch_targets.insert(next_addr);
            }
        }
    }

    if all_insns.is_empty() {
        return Err("no instructions decoded".into());
    }

    let end_addr = func_end.unwrap_or_else(|| {
        all_insns.last().map(|i| i.addr + 4).unwrap_or(base_addr)
    });

    // Determine block boundaries.
    let mut block_starts: BTreeSet<u64> = BTreeSet::new();
    block_starts.insert(base_addr);
    for &target in &branch_targets {
        if target >= base_addr && target < end_addr {
            block_starts.insert(target);
        }
    }

    let mut addr_to_block: BTreeMap<u64, BlockId> = BTreeMap::new();
    for (idx, &addr) in block_starts.iter().enumerate() {
        addr_to_block.insert(addr, BlockId(idx as u32));
    }

    let find_block = |addr: u64| -> Option<BlockId> {
        addr_to_block.get(&addr).copied()
    };

    // Build basic blocks.
    let block_start_vec: Vec<u64> = block_starts.iter().copied().collect();
    let mut blocks: BTreeMap<BlockId, BasicBlock> = BTreeMap::new();
    let mut successors: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
    let mut predecessors: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();

    for (blk_idx, &blk_start) in block_start_vec.iter().enumerate() {
        let blk_id = BlockId(blk_idx as u32);
        let blk_end = if blk_idx + 1 < block_start_vec.len() {
            block_start_vec[blk_idx + 1]
        } else {
            end_addr
        };

        let mut insns: Vec<DecodedInsn> = Vec::new();
        let mut last_insn: Option<&A64Insn> = None;

        for a64_insn in &all_insns {
            if a64_insn.addr >= blk_start && a64_insn.addr < blk_end {
                insns.push(DecodedInsn {
                    mnemonic: a64_insn.mnemonic.clone(),
                    operands: a64_insn.operands.clone(),
                    addr: a64_insn.addr,
                });
                last_insn = Some(a64_insn);
            }
        }

        let terminator = if let Some(last) = last_insn {
            if last.is_return {
                Terminator::Return
            } else if last.is_uncond_branch {
                if let Some(target) = last.branch_target {
                    if let Some(target_id) = find_block(target) {
                        Terminator::Jump(target_id)
                    } else {
                        Terminator::Return
                    }
                } else {
                    Terminator::Return
                }
            } else if last.is_cond_branch {
                if let Some(target) = last.branch_target {
                    let taken_id = find_block(target).unwrap_or(blk_id);
                    let fall_addr = last.addr + 4;
                    let fall_id = find_block(fall_addr).unwrap_or(blk_id);
                    let kind = a64_cond_to_branch_kind(&last.mnemonic);
                    Terminator::CondBranch {
                        kind,
                        taken: taken_id,
                        fallthrough: fall_id,
                    }
                } else {
                    Terminator::Fallthrough(BlockId((blk_idx + 1) as u32))
                }
            } else if blk_idx + 1 < block_start_vec.len() {
                Terminator::Fallthrough(BlockId((blk_idx + 1) as u32))
            } else {
                Terminator::Return
            }
        } else {
            Terminator::Return
        };

        let succs = match &terminator {
            Terminator::Fallthrough(next) => vec![*next],
            Terminator::Jump(target) => vec![*target],
            Terminator::CondBranch { taken, fallthrough, .. } => vec![*taken, *fallthrough],
            Terminator::Return => vec![],
        };

        for &s in &succs {
            predecessors.entry(s).or_default().push(blk_id);
        }
        successors.insert(blk_id, succs);

        let block_end_addr = last_insn.map(|i| i.addr + 4).unwrap_or(blk_end);

        blocks.insert(blk_id, BasicBlock {
            id: blk_id,
            start_addr: blk_start,
            end_addr: block_end_addr,
            instructions: insns,
            terminator,
        });
    }

    for &blk_id in blocks.keys() {
        successors.entry(blk_id).or_default();
        predecessors.entry(blk_id).or_default();
    }

    Ok(ControlFlowGraph {
        blocks,
        entry: BlockId(0),
        successors,
        predecessors,
    })
}

/// Map AArch64 condition mnemonic to BranchKind.
#[cfg(target_arch = "aarch64")]
fn a64_cond_to_branch_kind(mnemonic: &str) -> super::cfg::BranchKind {
    use super::cfg::BranchKind;
    match mnemonic {
        "b.eq" => BranchKind::Equal,
        "b.ne" => BranchKind::NotEqual,
        "b.gt" => BranchKind::Greater,
        "b.ge" => BranchKind::GreaterEqual,
        "b.lt" => BranchKind::Less,
        "b.le" => BranchKind::LessEqual,
        "b.hi" => BranchKind::Above,
        "b.hs" | "b.cs" => BranchKind::AboveEqual,
        "b.lo" | "b.cc" => BranchKind::Below,
        "b.ls" => BranchKind::BelowEqual,
        "b.mi" => BranchKind::Sign,
        "b.pl" => BranchKind::NotSign,
        "b.vs" => BranchKind::Parity,
        "b.vc" => BranchKind::NotParity,
        // cbz/cbnz/tbz/tbnz → Equal/NotEqual
        "cbz" => BranchKind::Equal,
        "cbnz" => BranchKind::NotEqual,
        "tbz" => BranchKind::Equal,
        "tbnz" => BranchKind::NotEqual,
        _ => BranchKind::Equal,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_ret() {
        // ret = 0xD65F03C0
        let insn = decode_one(0xD65F03C0, 0x1000);
        assert_eq!(insn.mnemonic, "ret");
        assert!(insn.is_return);
    }

    #[test]
    fn test_decode_b() {
        // b #8 at PC=0x1000 → target = 0x1008
        // Encoding: 000101 imm26, imm26 = 2 (offset = 2*4 = 8)
        let word = 0b000101_00_0000_0000_0000_0000_0000_0010u32;
        let insn = decode_one(word, 0x1000);
        assert_eq!(insn.mnemonic, "b");
        assert!(insn.is_uncond_branch);
        assert_eq!(insn.branch_target, Some(0x1008));
    }

    #[test]
    fn test_decode_bl() {
        // bl #-16 at PC=0x1000 → target = 0xFF0
        // Encoding: 100101 imm26, imm26 = -4 (sign-extended, offset = -4*4 = -16)
        let imm26 = ((-4i32) as u32) & 0x03FF_FFFF;
        let word = (0b100101u32 << 26) | imm26;
        let insn = decode_one(word, 0x1000);
        assert_eq!(insn.mnemonic, "bl");
        assert!(insn.is_call);
        assert_eq!(insn.branch_target, Some(0x0FF0));
    }

    #[test]
    fn test_decode_fadd() {
        // fadd s0, s1, s2: 0001_1110_001_00010_001010_00001_00000
        // ftype=00 (single), opcode=0010 (fadd), Rm=2, Rn=1, Rd=0
        let word: u32 = 0b0001_1110_0010_0010_0010_1000_0010_0000;
        let insn = decode_one(word, 0x1000);
        assert_eq!(insn.mnemonic, "fadd");
        assert_eq!(insn.operands, vec!["s0", "s1", "s2"]);
    }

    #[test]
    fn test_decode_fmadd() {
        // fmadd s0, s1, s2, s3: 0001_1111_00_0_00010_0_00011_00001_00000
        // ftype=00, o1=0, Rm=2, o0=0, Ra=3, Rn=1, Rd=0
        let word: u32 = 0b0001_1111_0000_0010_0000_1100_0010_0000;
        let insn = decode_one(word, 0x1000);
        assert_eq!(insn.mnemonic, "fmadd");
        assert_eq!(insn.operands.len(), 4);
        assert_eq!(insn.operands[0], "s0");
        assert_eq!(insn.operands[1], "s1");
        assert_eq!(insn.operands[2], "s2");
        assert_eq!(insn.operands[3], "s3");
    }

    #[test]
    fn test_a64_iter() {
        // Two instructions: fadd + ret
        let fadd: u32 = 0b0001_1110_0010_0010_0010_1000_0010_0000;
        let ret: u32 = 0xD65F03C0;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&fadd.to_le_bytes());
        bytes.extend_from_slice(&ret.to_le_bytes());

        let insns: Vec<A64Insn> = A64Iter::new(&bytes, 0x1000).collect();
        assert_eq!(insns.len(), 2);
        assert_eq!(insns[0].mnemonic, "fadd");
        assert_eq!(insns[1].mnemonic, "ret");
    }

    #[test]
    fn test_parse_base_reg() {
        assert_eq!(parse_base_reg("[x0, #16]"), Some(0));
        assert_eq!(parse_base_reg("[sp, #-8]"), Some(31));
        assert_eq!(parse_base_reg("[x15, #0]"), Some(15));
    }

    #[test]
    fn test_is_float_reg() {
        assert!(is_float_reg("s0"));
        assert!(is_float_reg("d15"));
        assert!(is_float_reg("q31"));
        assert!(!is_float_reg("x0"));
        assert!(!is_float_reg("sp"));
    }

    // ── New tests (target: 21 total) ──────────────────────────────

    #[test]
    fn test_a64_insn_default_fields() {
        // Arrange: construct A64Insn with all boolean fields false
        let insn = A64Insn {
            mnemonic: "nop".to_string(),
            operands: vec![],
            addr: 0,
            len: 4,
            branch_target: None,
            is_cond_branch: false,
            is_uncond_branch: false,
            is_return: false,
            is_call: false,
        };

        // Assert: verify every field
        assert_eq!(insn.mnemonic, "nop");
        assert!(insn.operands.is_empty());
        assert_eq!(insn.addr, 0);
        assert_eq!(insn.len, 4);
        assert_eq!(insn.branch_target, None);
        assert!(!insn.is_cond_branch);
        assert!(!insn.is_uncond_branch);
        assert!(!insn.is_return);
        assert!(!insn.is_call);
    }

    #[test]
    fn test_a64_insn_clone_roundtrip() {
        // Arrange
        let original = A64Insn {
            mnemonic: "fmadd".to_string(),
            operands: vec!["s0".to_string(), "s1".to_string()],
            addr: 0xDEAD,
            len: 4,
            branch_target: Some(0xBEEF),
            is_cond_branch: false,
            is_uncond_branch: false,
            is_return: false,
            is_call: false,
        };

        // Act
        let cloned = original.clone();

        // Assert: cloned matches original, mutation is independent
        assert_eq!(cloned.mnemonic, original.mnemonic);
        assert_eq!(cloned.operands, original.operands);
        assert_eq!(cloned.addr, original.addr);
        assert_eq!(cloned.branch_target, original.branch_target);
        assert_eq!(cloned.is_return, original.is_return);
    }

    #[test]
    fn test_a64_insn_debug_format() {
        // Arrange
        let insn = A64Insn {
            mnemonic: "ret".to_string(),
            operands: vec![],
            addr: 0x1000,
            len: 4,
            branch_target: None,
            is_cond_branch: false,
            is_uncond_branch: false,
            is_return: true,
            is_call: false,
        };

        // Act
        let debug_str = format!("{insn:?}");

        // Assert: Debug output contains key fields
        assert!(debug_str.contains("ret"));
        assert!(debug_str.contains("4096")); // addr: 0x1000 = 4096 in decimal
        assert!(debug_str.contains("is_return: true"));
    }

    #[test]
    fn test_a64_insn_struct_update_syntax() {
        // Arrange: base instruction
        let base = A64Insn {
            mnemonic: "b".to_string(),
            operands: vec!["0x2000".to_string()],
            addr: 0x1000,
            len: 4,
            branch_target: Some(0x2000),
            is_cond_branch: false,
            is_uncond_branch: true,
            is_return: false,
            is_call: false,
        };

        // Act: struct update syntax overriding only addr and branch_target
        let derived = A64Insn {
            addr: 0x5000,
            branch_target: Some(0x6000),
            ..base
        };

        // Assert: overridden fields changed, rest inherited
        assert_eq!(derived.addr, 0x5000);
        assert_eq!(derived.branch_target, Some(0x6000));
        assert_eq!(derived.mnemonic, "b");
        assert_eq!(derived.operands, vec!["0x2000"]);
        assert!(derived.is_uncond_branch);
    }

    #[test]
    fn test_bits_extraction_boundary() {
        // Arrange: word = 0xFF00_F0F0
        let word = 0xFF00_F0F0u32;

        // Act & Assert
        assert_eq!(bits(word, 31, 24), 0xFF);       // top byte
        assert_eq!(bits(word, 7, 0), 0xF0);          // bottom byte
        assert_eq!(bits(word, 15, 8), 0xF0);         // second byte
        assert_eq!(bits(word, 23, 16), 0x00);        // third byte
        assert_eq!(bits(word, 7, 7), 1);             // single bit (bit 7 of 0xF0 = 1)
        assert_eq!(bits(word, 6, 6), 1);             // single bit (bit 6 of 0xF0 = 1)
        assert_eq!(bits(word, 3, 3), 0);             // bit 3 of 0xF0 = 0
        assert_eq!(bits(word, 4, 4), 1);             // bit 4 of 0xF0 = 1
    }

    #[test]
    fn test_sign_extend_edge_cases() {
        // Arrange & Act & Assert
        // Positive value within range
        assert_eq!(sign_extend(0x00FF, 16), 0x00FF);

        // Negative value: top bit set in 16-bit
        assert_eq!(sign_extend(0xFFFF, 16), -1);
        assert_eq!(sign_extend(0x8000, 16), -32768);

        // 1-bit: 0 and 1
        assert_eq!(sign_extend(0, 1), 0);
        assert_eq!(sign_extend(1, 1), -1);

        // 32-bit: max positive (top bit clear)
        assert_eq!(sign_extend(0x7FFF_FFFF, 32), 0x7FFF_FFFF_i64);

        // 26-bit: branch offset encoding used in decode_branch
        let offset_neg_4: u32 = (-4i32 as u32) & 0x03FF_FFFF;
        assert_eq!(sign_extend(offset_neg_4, 26), -4);
    }

    #[test]
    fn test_register_name_helpers_boundary() {
        // Arrange & Act & Assert: xreg boundary values
        assert_eq!(xreg(0), "x0");
        assert_eq!(xreg(30), "x30");
        assert_eq!(xreg(31), "sp");          // x31 = sp

        // wreg boundary values
        assert_eq!(wreg(0), "w0");
        assert_eq!(wreg(30), "w30");
        assert_eq!(wreg(31), "wsp");          // w31 = wsp

        // sreg/dreg
        assert_eq!(sreg(0), "s0");
        assert_eq!(sreg(31), "s31");
        assert_eq!(dreg(0), "d0");
        assert_eq!(dreg(31), "d31");

        // ftype_reg: ftype=0→S, ftype=1→D, other→V
        assert_eq!(ftype_reg(5, 0), "s5");
        assert_eq!(ftype_reg(5, 1), "d5");
        assert_eq!(ftype_reg(5, 2), "v5");
        assert_eq!(ftype_reg(5, 99), "v5");
    }

    #[test]
    fn test_cond_name_all_codes() {
        // Arrange & Act & Assert: all 16 condition codes
        assert_eq!(cond_name(0x0), "eq");
        assert_eq!(cond_name(0x1), "ne");
        assert_eq!(cond_name(0x2), "hs");
        assert_eq!(cond_name(0x3), "lo");
        assert_eq!(cond_name(0x4), "mi");
        assert_eq!(cond_name(0x5), "pl");
        assert_eq!(cond_name(0x6), "vs");
        assert_eq!(cond_name(0x7), "vc");
        assert_eq!(cond_name(0x8), "hi");
        assert_eq!(cond_name(0x9), "ls");
        assert_eq!(cond_name(0xA), "ge");
        assert_eq!(cond_name(0xB), "lt");
        assert_eq!(cond_name(0xC), "gt");
        assert_eq!(cond_name(0xD), "le");
        assert_eq!(cond_name(0xE), "al");
        assert_eq!(cond_name(0xF), "nv");

        // cond_name masks upper bits: 0x1F & 0xF = 0xF → "nv"
        assert_eq!(cond_name(0x1F), "nv");
    }

    #[test]
    fn test_decode_unknown_instruction_fallback() {
        // Arrange: all zeros → unlikely to match any real encoding
        let insn = decode_one(0x0000_0000, 0x0);

        // Assert: should produce a .word fallback mnemonic, not panic
        assert!(insn.mnemonic.contains("0x00000000"));
        assert_eq!(insn.addr, 0);
        assert_eq!(insn.len, 4);
        assert!(!insn.is_return);
        assert!(!insn.is_call);
        assert!(!insn.is_cond_branch);
        assert!(!insn.is_uncond_branch);
    }

    #[test]
    fn test_a64_iter_empty_and_single() {
        // Arrange: empty byte slice
        let empty: Vec<u8> = vec![];
        let mut iter_empty = A64Iter::new(&empty, 0x1000);

        // Act & Assert: no items from empty slice
        assert!(iter_empty.next().is_none());

        // Arrange: single 4-byte instruction (ret)
        let single = 0xD65F03C0u32.to_le_bytes().to_vec();
        let insns: Vec<A64Insn> = A64Iter::new(&single, 0).collect();

        // Assert
        assert_eq!(insns.len(), 1);
        assert_eq!(insns[0].mnemonic, "ret");
        assert_eq!(insns[0].addr, 0);
    }

    #[test]
    fn test_a64_iter_truncated_bytes_ignored() {
        // Arrange: 7 bytes (one full instruction + 3 trailing bytes)
        let ret_bytes = 0xD65F03C0u32.to_le_bytes();
        let mut bytes = Vec::with_capacity(7);
        bytes.extend_from_slice(&ret_bytes);
        bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF]); // 3 trailing bytes

        // Act
        let insns: Vec<A64Insn> = A64Iter::new(&bytes, 0x8000).collect();

        // Assert: only the complete 4-byte instruction decoded
        assert_eq!(insns.len(), 1);
        assert_eq!(insns[0].addr, 0x8000);
    }

    #[test]
    fn test_parse_base_reg_invalid_inputs() {
        // Arrange & Act & Assert
        assert_eq!(parse_base_reg(""), None);                   // empty string
        assert_eq!(parse_base_reg("x0, #16]"), None);           // missing bracket
        assert_eq!(parse_base_reg("[y0, #16]"), None);          // not x/sp register
        assert_eq!(parse_base_reg("[x, #16]"), None);           // no digit after x
        assert_eq!(parse_base_reg("[sp, #0]"), Some(31));       // valid sp
        assert_eq!(parse_base_reg("[x0]"), Some(0));            // no offset, just base
    }

    #[test]
    fn test_is_integer_mnemonic_coverage() {
        // Arrange & Act & Assert: mnemonics classified as integer
        let int_mnemonics = [
            "mov", "movz", "movk", "movn",
            "add", "sub", "mul", "sdiv", "udiv",
            "and", "orr", "eor", "tst",
            "lsl", "lsr", "asr",
            "cmp", "cmn",
            "adrp", "adr",
            "sxtw", "uxtw",
            "nop",
            "stp_int", "ldp_int", "str_int", "ldr_int",
            "csel", "csinc", "csinv", "csneg",
            "ls_unknown", "dp_imm_unknown", "dp_reg",
        ];
        for m in &int_mnemonics {
            assert!(is_integer_mnemonic(m), "expected '{m}' to be integer mnemonic");
        }

        // Float mnemonics should NOT be classified as integer
        assert!(!is_integer_mnemonic("fadd"));
        assert!(!is_integer_mnemonic("fmul"));
        assert!(!is_integer_mnemonic("ret"));
        assert!(!is_integer_mnemonic("bl"));
    }

    #[test]
    fn test_count_inputs_with_signature() {
        // Arrange: signature with 2 inputs, 1 output, 1 dim
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(128),
            ],
        };

        // Act
        let count = count_inputs(&sig);

        // Assert: only InputPtr counted
        assert_eq!(count, 2);

        // Arrange: empty signature
        let empty_sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![],
        };

        // Act & Assert
        assert_eq!(count_inputs(&empty_sig), 0);

        // Arrange: signature with WeightPtr counted as input
        let weight_sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Scalar(1.0),
            ],
        };

        // Act & Assert
        assert_eq!(count_inputs(&weight_sig), 1);
    }

    // ── Wave 12k56 tests (+13) ──────────────────────────────────────

    #[test]
    fn test_decode_bcond_eq() {
        // b.eq at PC=0x2000 with forward offset of +8 bytes (imm19 = 2)
        // Encoding: 0101_0100 imm19 0 cond
        // cond=0 (eq), imm19=2, bit4=0
        let imm19: u32 = 2;
        let word = (0b0101_0100u32 << 24) | (imm19 << 5) | 0b0000;
        let insn = decode_one(word, 0x2000);

        assert_eq!(insn.mnemonic, "b.eq");
        assert!(insn.is_cond_branch);
        assert!(!insn.is_uncond_branch);
        assert!(!insn.is_return);
        assert!(!insn.is_call);
        assert_eq!(insn.branch_target, Some(0x2008));
        assert_eq!(insn.operands, vec!["0x2008"]);
    }

    #[test]
    fn test_decode_cbz_cbnz() {
        // cbz x0, #+8 at PC=0x1000
        // Encoding: sf=1 (64-bit), 011_0100_0 imm19 rt
        // bits[30:25] = 011010, op=0 (cbz), sf=1
        let imm19: u32 = 2; // offset = 2*4 = 8
        let rt: u32 = 0;
        let word = (1u32 << 31) | (0b011_0100u32 << 25) | (0u32 << 24) | (imm19 << 5) | rt;
        let insn = decode_one(word, 0x1000);

        assert_eq!(insn.mnemonic, "cbz");
        assert!(insn.is_cond_branch);
        assert_eq!(insn.branch_target, Some(0x1008));
        assert_eq!(insn.operands[0], "x0");

        // cbnz w3, #+12 at PC=0x2000
        // sf=0 (32-bit), op=1 (cbnz)
        let imm19_b: u32 = 3; // offset = 3*4 = 12
        let rt_b: u32 = 3;
        let word_b = (0u32 << 31) | (0b011_0100u32 << 25) | (1u32 << 24) | (imm19_b << 5) | rt_b;
        let insn_b = decode_one(word_b, 0x2000);

        assert_eq!(insn_b.mnemonic, "cbnz");
        assert!(insn_b.is_cond_branch);
        assert_eq!(insn_b.operands[0], "w3");
        assert_eq!(insn_b.branch_target, Some(0x200C));
    }

    #[test]
    fn test_decode_tbz_tbnz() {
        // tbz x0, #0, #+8 at PC=0x1000
        // Encoding: sf=1, 011_0110 b5 imm14 rt
        // bits[30:25] = 011011, op=0 (tbz), b5=0, b40=0 (bit_pos=0)
        let imm14: u32 = 2; // offset = 2*4 = 8
        let rt: u32 = 5;
        let b5: u32 = 0;
        let b40: u32 = 0;
        let word = (1u32 << 31) | (0b011_011u32 << 25) | (0u32 << 24)
            | (b5 << 19) | (b40 << 19) | (imm14 << 5) | rt;
        // Reconstruct properly: bit31=b5, bits[30:25]=011011, bit24=op, bits[23:19]=b40, bits[18:5]=imm14, bits[4:0]=rt
        let word = (b5 << 31) | (0b011_011u32 << 25) | (0u32 << 24)
            | (b40 << 19) | (imm14 << 5) | rt;
        let insn = decode_one(word, 0x1000);

        assert_eq!(insn.mnemonic, "tbz");
        assert!(insn.is_cond_branch);
        assert_eq!(insn.operands[0], "x5");
        assert_eq!(insn.operands[1], "#0");
    }

    #[test]
    fn test_decode_fp_2source_all_ops() {
        // Test fsub (opcode=0011), fmul (opcode=0000), fdiv (opcode=0001), fmax (opcode=0100), fmin (opcode=0101)
        let test_cases = [
            (0b0000u32, "fmul"),
            (0b0001, "fdiv"),
            (0b0010, "fadd"),
            (0b0011, "fsub"),
            (0b0100, "fmax"),
            (0b0101, "fmin"),
            (0b0110, "fmaxnm"),
            (0b0111, "fminnm"),
        ];

        for (opcode, expected_mnem) in test_cases {
            // FP 2-source: 0001_1110_fty_1_Rm_opcode_10_Rn_Rd
            let word = (0b0001_1110u32 << 24)
                | (0u32 << 22)      // ftype=0 (single)
                | (1u32 << 21)      // bit21=1
                | (2u32 << 16)      // Rm=2
                | (opcode << 12)    // opcode
                | (0b10u32 << 10)   // bits[11:10]=10
                | (1u32 << 5)       // Rn=1
                | 0u32;             // Rd=0
            let insn = decode_one(word, 0x1000);
            assert_eq!(insn.mnemonic, expected_mnem, "opcode {opcode} should decode to {expected_mnem}");
            assert_eq!(insn.operands[0], "s0");
            assert_eq!(insn.operands[1], "s1");
            assert_eq!(insn.operands[2], "s2");
        }
    }

    #[test]
    fn test_decode_fp_3source_fmsub_fnmadd_fnmsub() {
        // fmsub s0, s1, s2, s3: o1=0, o0=1
        // Encoding: 0001_1111_00_0_Rm_1_Ra_Rn_Rd
        let word_fmsub = (0b0001_1111u32 << 24)
            | (0u32 << 22)         // ftype=0
            | (0u32 << 21)         // o1=0
            | (2u32 << 16)         // Rm=2
            | (1u32 << 15)         // o0=1
            | (3u32 << 10)         // Ra=3
            | (1u32 << 5)          // Rn=1
            | 0u32;                // Rd=0
        let insn = decode_one(word_fmsub, 0);
        assert_eq!(insn.mnemonic, "fmsub");
        assert_eq!(insn.operands, vec!["s0", "s1", "s2", "s3"]);

        // fnmadd: o1=1, o0=0
        let word_fnmadd = (0b0001_1111u32 << 24)
            | (0u32 << 22) | (1u32 << 21) | (2u32 << 16)
            | (0u32 << 15) | (3u32 << 10) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_fnmadd, 0).mnemonic, "fnmadd");

        // fnmsub: o1=1, o0=1
        let word_fnmsub = (0b0001_1111u32 << 24)
            | (0u32 << 22) | (1u32 << 21) | (2u32 << 16)
            | (1u32 << 15) | (3u32 << 10) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_fnmsub, 0).mnemonic, "fnmsub");
    }

    #[test]
    fn test_decode_fp_1source_fmov_fabs_fneg_fsqrt_fcvt() {
        // fmov s0, s1: opcode=000000
        // Encoding: 0001_1110_fty_1_00000_0_opcode_Rn_Rd
        let word_fmov = (0b0001_1110u32 << 24)
            | (0u32 << 22)           // ftype=0 (single)
            | (0b10000u32 << 17)     // bits[21:17]=10000
            | (0u32 << 14)           // bit14=0
            | (0b000000u32 << 10)    // no overlap; opcode in bits[15:10]
            | (1u32 << 5)            // Rn=1
            | 0u32;                  // Rd=0
        // Rebuild properly: bits[15:10] = opcode=000000, bit14=0
        let word_fmov = (0b0001_1110u32 << 24)
            | (0u32 << 22)
            | (1u32 << 21)
            | (0b00000u32 << 17)
            | (0u32 << 14)           // bit14=0
            | (0b000000u32 << 10)    // bits[15:10] – but bit14 is already 0
            | (1u32 << 5) | 0u32;
        // Actually need: bits[21:17]=10000, bit14=0, bits[15:10]=opcode
        // So: bit21=1, bits[20:17]=0000, bits[15:10]=000000, bit14=0
        let word_fmov = (0b0001_1110u32 << 24) | (0u32 << 22) | (1u32 << 21)
            | (0u32 << 17) | (0b000000u32 << 10) | (1u32 << 5) | 0u32;
        // bits[20:17] = 0000, so bits[21:17] = 10000
        let insn = decode_one(word_fmov, 0);
        assert_eq!(insn.mnemonic, "fmov");
        assert_eq!(insn.operands, vec!["s0", "s1"]);

        // fabs: opcode=000001
        let word_fabs = (0b0001_1110u32 << 24) | (0u32 << 22) | (1u32 << 21)
            | (0u32 << 17) | (0b000001u32 << 10) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_fabs, 0).mnemonic, "fabs");

        // fneg: opcode=000010
        let word_fneg = (0b0001_1110u32 << 24) | (0u32 << 22) | (1u32 << 21)
            | (0u32 << 17) | (0b000010u32 << 10) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_fneg, 0).mnemonic, "fneg");

        // fsqrt: opcode=000011
        let word_fsqrt = (0b0001_1110u32 << 24) | (0u32 << 22) | (1u32 << 21)
            | (0u32 << 17) | (0b000011u32 << 10) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_fsqrt, 0).mnemonic, "fsqrt");

        // fcvt S->D: opcode=000100 → dest is dreg, src is sreg
        let word_fcvt_sd = (0b0001_1110u32 << 24) | (0u32 << 22) | (1u32 << 21)
            | (0u32 << 17) | (0b000100u32 << 10) | (1u32 << 5) | 0u32;
        let insn_fcvt = decode_one(word_fcvt_sd, 0);
        assert_eq!(insn_fcvt.mnemonic, "fcvt");
        assert_eq!(insn_fcvt.operands[0], "d0");  // dest: double
        assert_eq!(insn_fcvt.operands[1], "s1");  // src: single

        // fcvt D->S: opcode=000101 → dest is sreg, src is dreg
        let word_fcvt_ds = (0b0001_1110u32 << 24) | (0u32 << 22) | (1u32 << 21)
            | (0u32 << 17) | (0b000101u32 << 10) | (1u32 << 5) | 0u32;
        let insn_ds = decode_one(word_fcvt_ds, 0);
        assert_eq!(insn_ds.operands[0], "s0");  // dest: single
        assert_eq!(insn_ds.operands[1], "d1");  // src: double
    }

    #[test]
    fn test_decode_fp_compare_fcmp_fcmpe() {
        // fcmp s0, s1: opc=00 → fcmp
        // Encoding: 0001_1110_fty_1_Rm_0010_00_Rn_opc_000
        // bits[31:24]=00011110, ftype=00, bit21=1, bits[13:10]=1000, bits[2:0]=000
        let word_fcmp = (0b0001_1110u32 << 24)
            | (0u32 << 22)         // ftype=0
            | (1u32 << 21)
            | (1u32 << 16)         // Rm=1
            | (0b1000u32 << 10)    // bits[13:10]
            | (0u32 << 5)          // Rn=0
            | (0b000u32 << 3)      // bits[4:3]=opc=00
            | 0u32;                // bits[2:0]=000
        let insn = decode_one(word_fcmp, 0);
        assert_eq!(insn.mnemonic, "fcmp");
        assert!(insn.operands[0].starts_with('s'));

        // fcmpe: opc bits[4:3] where bit0=1 → opc=01 (bit3=0,bit4=1 → opc & 1 == 1)
        let word_fcmpe = (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21) | (1u32 << 16)
            | (0b1000u32 << 10) | (0u32 << 5)
            | (0b010u32 << 3) | 0u32;  // opc=01 → bit3=1
        // opc bits[4:3] = 01 → opc=1 → fcmpe (opc & 1 == 1)
        let word_fcmpe = (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21) | (1u32 << 16)
            | (0b1000u32 << 10) | (0u32 << 5)
            | (1u32 << 3) | 0u32;  // bit3=1 → opc=01
        let insn_e = decode_one(word_fcmpe, 0);
        assert_eq!(insn_e.mnemonic, "fcmpe");
    }

    #[test]
    fn test_decode_fp_fcsel() {
        // fcsel s0, s1, s2, eq: cond=0
        // Encoding: 0001_1110_fty_1_Rm_cond_11_Rn_Rd
        let word = (0b0001_1110u32 << 24)
            | (0u32 << 22)         // ftype=0 (single)
            | (1u32 << 21)
            | (2u32 << 16)         // Rm=2
            | (0u32 << 12)         // cond=0 (eq)
            | (0b11u32 << 10)      // bits[11:10]=11
            | (1u32 << 5)          // Rn=1
            | 0u32;                // Rd=0
        let insn = decode_one(word, 0);

        assert_eq!(insn.mnemonic, "fcsel");
        assert_eq!(insn.operands.len(), 4);
        assert_eq!(insn.operands[0], "s0");
        assert_eq!(insn.operands[1], "s1");
        assert_eq!(insn.operands[2], "s2");
        assert_eq!(insn.operands[3], "eq");
    }

    #[test]
    fn test_decode_dp_imm_movz_movn_movk_add_sub_cmp_adrp() {
        // movz x0, #0: bits[28:26]=100, bits[25:23]=101, opc=10, sf=1
        let word_movz = (1u32 << 31) | (0b100u32 << 26) | (0b101u32 << 23) | (0b10u32 << 29);
        let insn = decode_one(word_movz, 0);
        assert_eq!(insn.mnemonic, "movz");
        assert_eq!(insn.operands[0], "x0");

        // movn w0: opc=00, sf=0
        let word_movn = (0u32 << 31) | (0b100u32 << 26) | (0b101u32 << 23) | (0b00u32 << 29);
        let insn_n = decode_one(word_movn, 0);
        assert_eq!(insn_n.mnemonic, "movn");
        assert_eq!(insn_n.operands[0], "w0");

        // movk: opc=11
        let word_movk = (1u32 << 31) | (0b100u32 << 26) | (0b101u32 << 23) | (0b11u32 << 29);
        assert_eq!(decode_one(word_movk, 0).mnemonic, "movk");

        // add immediate: bits[28:26]=100, bits[25:23]=001, op=0, s=0
        let word_add = (1u32 << 31) | (0b100u32 << 26) | (0b001u32 << 23);
        let insn_add = decode_one(word_add, 0);
        assert_eq!(insn_add.mnemonic, "add");
        assert_eq!(insn_add.operands[0], "x0");

        // sub immediate: op=1
        let word_sub = (1u32 << 31) | (1u32 << 30) | (0b100u32 << 26) | (0b001u32 << 23);
        assert_eq!(decode_one(word_sub, 0).mnemonic, "sub");

        // cmp: s=1, rd=31 → "cmp" mnemonic
        let word_cmp = (1u32 << 31) | (0b11u32 << 29) | (0b100u32 << 26) | (0b001u32 << 23) | (31u32 << 0);
        let insn_cmp = decode_one(word_cmp, 0);
        assert_eq!(insn_cmp.mnemonic, "cmp");
        assert_eq!(insn_cmp.operands[0], "x5"); // rn = bits[9:5] = 0 with bit29=s=1

        // adrp: bits[28:26]=100, bits[25:23]=000, sf=1 (bit31=1)
        let word_adrp = (1u32 << 31) | (0b100u32 << 26) | (0b000u32 << 23);
        let insn_adrp = decode_one(word_adrp, 0);
        assert_eq!(insn_adrp.mnemonic, "adrp");
        assert_eq!(insn_adrp.operands[0], "x0");

        // adr: bit31=0
        let word_adr = (0u32 << 31) | (0b100u32 << 26) | (0b000u32 << 23);
        assert_eq!(decode_one(word_adr, 0).mnemonic, "adr");
    }

    #[test]
    fn test_decode_dp_imm_logical_ops() {
        // Logical immediate: bits[28:26]=100, bits[25:23]=010
        // and: opc=00
        let word_and = (1u32 << 31) | (0b00u32 << 29) | (0b100u32 << 26) | (0b010u32 << 23);
        assert_eq!(decode_one(word_and, 0).mnemonic, "and");

        // orr: opc=01
        let word_orr = (1u32 << 31) | (0b01u32 << 29) | (0b100u32 << 26) | (0b010u32 << 23);
        assert_eq!(decode_one(word_orr, 0).mnemonic, "orr");

        // eor: opc=10
        let word_eor = (1u32 << 31) | (0b10u32 << 29) | (0b100u32 << 26) | (0b010u32 << 23);
        assert_eq!(decode_one(word_eor, 0).mnemonic, "eor");

        // tst: opc=11
        let word_tst = (1u32 << 31) | (0b11u32 << 29) | (0b100u32 << 26) | (0b010u32 << 23);
        assert_eq!(decode_one(word_tst, 0).mnemonic, "tst");
    }

    #[test]
    fn test_decode_br_blr_ret_non_x30() {
        // br x5: 1101_0110_0000_0000_0000_00_00101_00000
        // bits[31:22] = 1101011000, opc=0000 (br), rn=5
        let word_br = (0b1101011000u32 << 22) | (0b0000u32 << 21) | (5u32 << 5);
        let insn = decode_one(word_br, 0x1000);

        assert_eq!(insn.mnemonic, "br");
        assert!(insn.is_uncond_branch);
        assert!(!insn.is_call);
        assert!(!insn.is_return);
        assert_eq!(insn.operands[0], "x5");

        // blr x2: opc=0001
        let word_blr = (0b1101011000u32 << 22) | (0b0001u32 << 21) | (2u32 << 5);
        let insn_blr = decode_one(word_blr, 0x1000);

        assert_eq!(insn_blr.mnemonic, "blr");
        assert!(insn_blr.is_call);
        assert!(!insn_blr.is_return);
        assert_eq!(insn_blr.operands[0], "x2");

        // ret x5 (non-x30): opc=0010, rn=5 → operands include x5
        let word_ret5 = (0b1101011001u32 << 22) | (0b0010u32 << 21) | (5u32 << 5);
        let insn_ret5 = decode_one(word_ret5, 0x1000);

        assert_eq!(insn_ret5.mnemonic, "ret");
        assert!(insn_ret5.is_return);
        assert_eq!(insn_ret5.operands[0], "x5");
    }

    #[test]
    fn test_decode_dp_reg_csel_family() {
        // csel x0, x1, x2, eq: op=0, op2=0
        // Encoding: x0_11010100_xxxxx_xxxx_0x_xxxxx_xxxxx
        // bits[29:21] = 011010100, sf=1 (x-reg), op=0, op2=0
        let word_csel = (1u32 << 31) | (0b011010100u32 << 21)
            | (0u32 << 30)          // op=0
            | (0u32 << 10)          // op2=0
            | (2u32 << 16)          // Rm=2
            | (0u32 << 12)          // cond=0 (eq)
            | (1u32 << 5)           // Rn=1
            | 0u32;                 // Rd=0
        let insn = decode_one(word_csel, 0);

        assert_eq!(insn.mnemonic, "csel");
        assert_eq!(insn.operands[0], "x0");
        assert_eq!(insn.operands[1], "x1");
        assert_eq!(insn.operands[2], "x2");
        assert_eq!(insn.operands[3], "eq");

        // csinc: op=0, op2=1
        let word_csinc = (1u32 << 31) | (0b011010100u32 << 21)
            | (0u32 << 30) | (1u32 << 10) | (2u32 << 16) | (0u32 << 12) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_csinc, 0).mnemonic, "csinc");

        // csinv: op=1, op2=0
        let word_csinv = (1u32 << 31) | (0b011010100u32 << 21)
            | (1u32 << 30) | (0u32 << 10) | (2u32 << 16) | (0u32 << 12) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_csinv, 0).mnemonic, "csinv");

        // csneg: op=1, op2=1
        let word_csneg = (1u32 << 31) | (0b011010100u32 << 21)
            | (1u32 << 30) | (1u32 << 10) | (2u32 << 16) | (0u32 << 12) | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_csneg, 0).mnemonic, "csneg");
    }

    #[test]
    fn test_decode_dp_reg_shifted_register_ops() {
        // add (shifted reg): op_kind=00
        // bits[28:25] = 0101 (DP register), op_kind bits[30:29]=00
        let word_add = (1u32 << 31) | (0b00u32 << 29) | (0b0101u32 << 25)
            | (1u32 << 5) | 0u32;   // Rn=1, Rd=0
        let insn = decode_one(word_add, 0);
        assert_eq!(insn.mnemonic, "add");

        // sub: op_kind=01
        let word_sub = (1u32 << 31) | (0b01u32 << 29) | (0b0101u32 << 25);
        assert_eq!(decode_one(word_sub, 0).mnemonic, "sub");

        // and: op_kind=10
        let word_and = (1u32 << 31) | (0b10u32 << 29) | (0b0101u32 << 25);
        assert_eq!(decode_one(word_and, 0).mnemonic, "and");

        // orr: op_kind=11
        let word_orr = (1u32 << 31) | (0b11u32 << 29) | (0b0101u32 << 25);
        assert_eq!(decode_one(word_orr, 0).mnemonic, "orr");

        // sf=0 → w registers
        let word_add_w = (0u32 << 31) | (0b00u32 << 29) | (0b0101u32 << 25) | (1u32 << 5) | 0u32;
        let insn_w = decode_one(word_add_w, 0);
        assert_eq!(insn_w.operands[0], "w0");
    }

    #[test]
    fn test_is_float_reg_edge_cases() {
        // Positive cases
        assert!(is_float_reg("s0"));
        assert!(is_float_reg("d31"));
        assert!(is_float_reg("q15"));
        assert!(is_float_reg("v7"));

        // Negative: wrong prefix
        assert!(!is_float_reg("x0"));
        assert!(!is_float_reg("w0"));
        assert!(!is_float_reg("sp"));

        // Negative: single character (no digit)
        assert!(!is_float_reg("s"));
        assert!(!is_float_reg("d"));

        // Negative: empty string
        assert!(!is_float_reg(""));

        // Negative: letter after prefix
        assert!(!is_float_reg("sab"));
        assert!(!is_float_reg("d1a"));
    }

    #[test]
    fn test_decode_fp_int_conversion_scvtf_ucvtf() {
        // scvtf d0, x1: sf=1, ftype=01 (double), rmode=00, opcode=010
        // Encoding: 0001_1110_sf_fty_1_rmode_opcode_000000_Rn_Rd
        // bits[31:24]=00011110, bit21=1, bits[15:10]=000000
        let word_scvtf = (1u32 << 31) | (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)
            | (0b00u32 << 19)        // rmode=00
            | (0b010u32 << 16)       // opcode=010 (scvtf)
            | (0b000000u32 << 10)
            | (1u32 << 5)            // Rn=1
            | 0u32;                  // Rd=0
        let insn = decode_one(word_scvtf, 0);
        assert_eq!(insn.mnemonic, "scvtf");
        assert_eq!(insn.operands[0], "d0");   // ftype=01 → double
        assert_eq!(insn.operands[1], "x1");   // sf=1 → x reg

        // ucvtf: opcode=011
        let word_ucvtf = (1u32 << 31) | (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21)
            | (0b00u32 << 19) | (0b011u32 << 16) | (0b000000u32 << 10)
            | (1u32 << 5) | 0u32;
        assert_eq!(decode_one(word_ucvtf, 0).mnemonic, "ucvtf");

        // scvtf s0, w1: sf=0, ftype=00
        let word_scvtf_w = (0u32 << 31) | (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21)
            | (0b00u32 << 19) | (0b010u32 << 16) | (0b000000u32 << 10)
            | (1u32 << 5) | 0u32;
        let insn_w = decode_one(word_scvtf_w, 0);
        assert_eq!(insn_w.operands[0], "s0");  // ftype=00 → single
        assert_eq!(insn_w.operands[1], "w1");  // sf=0 → w reg
    }

    // ── Wave 12k69 tests (+10) ──────────────────────────────────────

    // @trace TEST-12k69
    #[test]
    fn test_decode_ldr_literal_simd_fp() {
        // Arrange: LDR S0, [PC, #offset] — literal SIMD/FP load
        // Encoding: opc=00 011 1 00 imm19 Rt
        // bits[29:27]=011, bit[26]=1, bits[25:24]=00
        // opc=00 → s register, imm19=1 → offset = 1*4 = 4, rt=0
        let imm19: u32 = 1;
        let word = (0b00u32 << 30) | (0b011u32 << 27) | (1u32 << 26)
            | (0b00u32 << 24) | (imm19 << 5) | 0u32;
        let pc: u64 = 0x1000;

        // Act
        let insn = decode_one(word, pc);

        // Assert: decoded as ldr with s-register, branch_target holds literal address
        assert_eq!(insn.mnemonic, "ldr");
        assert_eq!(insn.operands[0], "s0");
        assert!(insn.operands[1].contains("pc,"));
        assert_eq!(insn.branch_target, Some(pc + 4)); // offset = imm19 * 4 = 4
        assert_eq!(insn.addr, pc);
        assert_eq!(insn.len, 4);
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_ldr_str_unsigned_offset_simd_fp() {
        // Arrange: LDR D5, [X2, #16] — unsigned offset SIMD/FP load
        // Encoding: size=01 111 1 01 opc=01 imm12 Rn Rt
        // bits[29:27]=111, bit[26]=1, bits[25:24]=01
        // size=01 (double), opc=01 (load), imm12=1, rn=2, rt=5
        let word_ldr = (0b01u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b01u32 << 24)
            | (0b01u32 << 22) | (1u32 << 10) | (2u32 << 5) | 5u32;

        // Act
        let insn_ldr = decode_one(word_ldr, 0x1000);

        // Assert: load decoded with d-register and offset
        assert_eq!(insn_ldr.mnemonic, "ldr");
        assert_eq!(insn_ldr.operands[0], "d5");
        assert!(insn_ldr.operands[1].contains("x2"));
        assert!(insn_ldr.operands[1].contains("#8")); // imm12=1 * scale=8 (size=01)

        // Arrange: STR S3, [SP, #0] — store to sp with zero offset
        // size=00, opc=00 (store), imm12=0, rn=31 (sp), rt=3
        let word_str = (0b00u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b01u32 << 24)
            | (0b00u32 << 22) | (0u32 << 10) | (31u32 << 5) | 3u32;

        // Act
        let insn_str = decode_one(word_str, 0x1000);

        // Assert: store decoded with s-register and sp base
        assert_eq!(insn_str.mnemonic, "str");
        assert_eq!(insn_str.operands[0], "s3");
        assert!(insn_str.operands[1].contains("sp"));
        assert!(insn_str.operands[1].contains("#0"));
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_ldp_stp_simd_fp_pair() {
        // Arrange: LDP S0, S1, [X2, #8] — load pair, single-precision
        // Encoding: opc=00 101 1 0xx imm7 Rt2 Rn Rt
        // bits[29:27]=101, bit[26]=1, bit[22]=1 (load)
        // opc=00 (s-reg), imm7=1, rt2=1, rn=2, rt=0
        // scale for opc=00 → 4, so offset = sign_extend(1,7) * 4 = 4
        let word_ldp = (0b00u32 << 30) | (0b101u32 << 27) | (1u32 << 26)
            | (1u32 << 22) | (1u32 << 15) | (1u32 << 10) | (2u32 << 5) | 0u32;

        // Act
        let insn_ldp = decode_one(word_ldp, 0x2000);

        // Assert
        assert_eq!(insn_ldp.mnemonic, "ldp");
        assert_eq!(insn_ldp.operands[0], "s0");
        assert_eq!(insn_ldp.operands[1], "s1");
        assert!(insn_ldp.operands[2].contains("x2"));

        // Arrange: STP D0, D1, [SP, #-16] — store pair, double-precision, sp base
        // opc=01 (d-reg), bit[22]=0 (store), imm7=-4 (sign-extended 7-bit), scale=8
        let imm7_neg4: u32 = ((-4i32) as u32) & 0x7F;
        let word_stp = (0b01u32 << 30) | (0b101u32 << 27) | (1u32 << 26)
            | (0u32 << 22) | (imm7_neg4 << 15) | (1u32 << 10) | (31u32 << 5) | 0u32;

        // Act
        let insn_stp = decode_one(word_stp, 0x3000);

        // Assert
        assert_eq!(insn_stp.mnemonic, "stp");
        assert_eq!(insn_stp.operands[0], "d0");
        assert_eq!(insn_stp.operands[1], "d1");
        assert!(insn_stp.operands[2].contains("sp"));
        assert!(insn_stp.operands[2].contains("-32")); // offset = -4 * 8 = -32
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_ldr_str_integer_unsigned_offset() {
        // Arrange: LDR X0, unsigned offset integer load
        // Encoding: size=11 111 0 01 opc=01 imm12 Rn Rt
        // bits[29:27]=111, bit[26]=0, bits[25:24]=01
        // size=11 (x-reg), opc=01 (load)
        let word_ldr_int = (0b11u32 << 30) | (0b111u32 << 27) | (0u32 << 26) | (0b01u32 << 24)
            | (0b01u32 << 22) | (0u32 << 10) | (0u32 << 5) | 0u32;

        // Act
        let insn = decode_one(word_ldr_int, 0);

        // Assert: integer load decoded, uses x register (size >= 0b11)
        assert_eq!(insn.mnemonic, "ldr_int");
        assert_eq!(insn.operands[0], "x0");

        // Arrange: STR W5, integer store (size=10, opc=00 store)
        let word_str_int = (0b10u32 << 30) | (0b111u32 << 27) | (0u32 << 26) | (0b01u32 << 24)
            | (0b00u32 << 22) | (0u32 << 10) | (0u32 << 5) | 5u32;

        // Act
        let insn_str = decode_one(word_str_int, 0);

        // Assert: integer store decoded, uses w register (size < 0b11)
        assert_eq!(insn_str.mnemonic, "str_int");
        assert_eq!(insn_str.operands[0], "w5");
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_ldp_stp_integer_pair() {
        // Arrange: LDP integer — opc 101 0 0xx, bit[22]=1 (load)
        // bits[29:27]=101, bit[26]=0
        let word_ldp_int = (0b00u32 << 30) | (0b101u32 << 27) | (0u32 << 26)
            | (1u32 << 22) | (0u32 << 15) | (0u32 << 10) | (0u32 << 5) | 0u32;

        // Act
        let insn = decode_one(word_ldp_int, 0);

        // Assert
        assert_eq!(insn.mnemonic, "ldp_int");

        // Arrange: STP integer — bit[22]=0 (store)
        let word_stp_int = (0b00u32 << 30) | (0b101u32 << 27) | (0u32 << 26)
            | (0u32 << 22);

        // Act
        let insn_stp = decode_one(word_stp_int, 0);

        // Assert
        assert_eq!(insn_stp.mnemonic, "stp_int");
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_load_store_register_offset() {
        // Arrange: LDR S0, [X1, X2] — register offset, SIMD/FP
        // Encoding: xx 111 x 00 xx 1 Rm opt S 10 Rn Rt
        // bits[29:27]=111, bits[25:24]=00, bit[21]=1, bits[11:10]=10
        // is_fp: bit[26]=1, size=00 (s-reg), opc=01 (load), rm=2, rn=1, rt=0
        let word_ldr_reg = (0b00u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b00u32 << 24)
            | (1u32 << 21) | (0b01u32 << 22) | (2u32 << 16) | (0b10u32 << 10)
            | (1u32 << 5) | 0u32;

        // Act
        let insn = decode_one(word_ldr_reg, 0);

        // Assert: register-offset load with x2 index register
        assert_eq!(insn.mnemonic, "ldr");
        assert_eq!(insn.operands[0], "s0");
        assert!(insn.operands[1].contains("x1"));
        assert!(insn.operands[1].contains("x2]"));

        // Arrange: STR D3, [X5, X7] — register offset, double store
        // size=01, bit[26]=1, opc=00 (store), rm=7, rn=5, rt=3
        let word_str_reg = (0b01u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b00u32 << 24)
            | (1u32 << 21) | (0b00u32 << 22) | (7u32 << 16) | (0b10u32 << 10)
            | (5u32 << 5) | 3u32;

        // Act
        let insn_str = decode_one(word_str_reg, 0);

        // Assert
        assert_eq!(insn_str.mnemonic, "str");
        assert_eq!(insn_str.operands[0], "d3");

        // Arrange: LDR integer register offset (bit[26]=0)
        // size=11 (x-reg), opc=01 (load)
        let word_ldr_int_reg = (0b11u32 << 30) | (0b111u32 << 27) | (0u32 << 26) | (0b00u32 << 24)
            | (1u32 << 21) | (0b01u32 << 22) | (2u32 << 16) | (0b10u32 << 10)
            | (1u32 << 5) | 0u32;

        // Act
        let insn_int = decode_one(word_ldr_int_reg, 0);

        // Assert: integer register-offset load uses x register
        assert_eq!(insn_int.mnemonic, "ldr_int");
        assert_eq!(insn_int.operands[0], "x0");
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_load_store_fallback() {
        // Arrange: a load/store encoding that falls through to the ls_unknown fallback
        // bits[27:25]=0b110 (ls_group), but no sub-pattern matches
        // We set bits[29:27]=101, bit[26]=0 (integer ldp/stp path) but mangle
        // to fall through. The ldp_int/stp_int path catches bit[26]=0 very early.
        // Instead use bits[27:25]=100 or 110 with bits that don't match any sub-pattern.
        // bits[29:27]=0b111, bit[26]=0, bits[25:24]=0b01 → integer unsigned offset
        // but let's find a path that hits the fallback. The fallback is "ls_unknown"
        // when none of the specific patterns match. Let's use bits[27:25]=100 with
        // a combination that doesn't hit any decoder path.
        // bits[29:27]=101, bit[26]=0 already goes to ldp_int/stp_int early.
        // To reach fallback, we need: ls_group=100 or 110, but all sub-patterns fail.
        // Use bits[29:27]=011, bit[26]=1, bits[25:24]=01 → would be SIMD unsigned offset
        // but only if bits[29:27]=111. Let's try: bits[29:27]=100, bit[26]=1, bits[25:24]=11
        // This gives ls_group=bits[27:25]=100, but the check is ls_group==0b100||0b110
        // So bits[27:25]=0b100 matches. Then inside decode_load_store, bits[29:27]=0b100
        // does not match any of the specific patterns (011, 111, 101, 111, 101, 111).
        let word = (0b01u32 << 30) | (0b100u32 << 27) | (1u32 << 26) | (0b11u32 << 24);

        // Act
        let insn = decode_one(word, 0);

        // Assert: falls to ls_unknown
        assert_eq!(insn.mnemonic, "ls_unknown");
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_fp_fallback_unknown() {
        // Arrange: bits[27:24]=0b1110 (enters decode_fp), but no sub-pattern matches
        // Use an encoding that hits the fallback ".fp 0x..." path.
        // bits[31:24]=0b0001_1110 (FP), bit[21]=0 (not 1 for 2-source/1-source),
        // and bits[11:10] != 0b10 (not 2-source), not matching compare or fcsel.
        // Also bits[31:24] != 0b0001_1111 (not 3-source).
        let word = 0b0001_1110_0000_0000_0000_0000_0000_0000u32;

        // Act
        let insn = decode_one(word, 0);

        // Assert: falls to FP fallback
        assert!(insn.mnemonic.starts_with(".fp"));
        assert!(insn.mnemonic.contains("0x0001e000"));
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_dp_imm_unknown_opcode() {
        // Arrange: bits[28:26]=100 (dp_imm), bits[25:23]=0b100 (unhandled op0)
        // op0 = bits[25:23] = 0b100 → hits the fallback "dp_imm_unknown"
        let word = (0u32 << 31) | (0b100u32 << 26) | (0b100u32 << 23);

        // Act
        let insn = decode_one(word, 0);

        // Assert
        assert_eq!(insn.mnemonic, "dp_imm_unknown");
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_one_always_len4_preserves_pc() {
        // Arrange: several different instruction encodings at different PCs
        let test_pcs: Vec<u64> = vec![0x0, 0x100, 0xFFFF_0000, 0x1_0000_0000];
        let test_words: Vec<u32> = vec![
            0xD65F03C0,   // ret
            0x0000_0000,  // unknown
            0b0001_1110_0010_0010_0010_1000_0010_0000, // fadd
        ];

        for &pc in &test_pcs {
            for &word in &test_words {
                // Act
                let insn = decode_one(word, pc);

                // Assert: len is always 4, addr is always pc
                assert_eq!(insn.len, 4, "len should be 4 for word 0x{word:08x} at pc 0x{pc:x}");
                assert_eq!(insn.addr, pc, "addr should be pc for word 0x{word:08x}");
            }
        }
    }

    // @trace TEST-12k69
    #[test]
    fn test_decode_fp_fmov_gp_to_fp_and_fp_to_gp() {
        // Arrange: fmov GP→FP (opcode=111): moves integer register to float register
        // Encoding: 0001_1110_sf_fty_1_rmode_opcode_000000_Rn_Rd
        // opcode=111, sf=1 (x-reg), ftype=00 (single), rmode=00
        let word_gp_to_fp = (1u32 << 31) | (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21)
            | (0b00u32 << 19) | (0b111u32 << 16) | (0b000000u32 << 10)
            | (1u32 << 5) | 0u32;

        // Act
        let insn_g2f = decode_one(word_gp_to_fp, 0);

        // Assert: fmov x1 → s0 (GP to FP)
        assert_eq!(insn_g2f.mnemonic, "fmov");
        assert_eq!(insn_g2f.operands[0], "s0");  // dest: float (ftype=0 → s)
        assert_eq!(insn_g2f.operands[1], "x1");  // src: integer (sf=1 → x)

        // Arrange: fmov FP→GP (opcode=110): moves float register to integer register
        // opcode=110, sf=0 (w-reg), ftype=01 (double)
        let word_fp_to_gp = (0u32 << 31) | (0b0001_1110u32 << 24)
            | (1u32 << 22) | (1u32 << 21)
            | (0b00u32 << 19) | (0b110u32 << 16) | (0b000000u32 << 10)
            | (1u32 << 5) | 0u32;

        // Act
        let insn_f2g = decode_one(word_fp_to_gp, 0);

        // Assert: fmov d1 → w0 (FP to GP)
        assert_eq!(insn_f2g.mnemonic, "fmov");
        assert_eq!(insn_f2g.operands[0], "w0");  // dest: integer (sf=0 → w)
        assert_eq!(insn_f2g.operands[1], "d1");  // src: float (ftype=1 → d)
    }

    // ── Wave 12kau tests (+10, target: 58 total) ───────────────────────

    // @trace TEST-12kau
    #[test]
    fn test_decode_fcvtzs_fcvtzu_float_to_int_sf0() {
        // Arrange: fcvtzs w0, s1 — sf=0, rmode=11, opcode=000
        // With sf=0, bits[31:24] = 0b0001_1110 = 0x1E, matching the FP int conversion path
        let word_fcvtzs = (0u32 << 31) | (0b0001_1110u32 << 24)
            | (0u32 << 22)           // ftype=00 (single)
            | (1u32 << 21)
            | (0b11u32 << 19)        // rmode=11
            | (0b000u32 << 16)       // opcode=000 (fcvtzs)
            | (0b000000u32 << 10)
            | (1u32 << 5)            // Rn=1
            | 0u32;                  // Rd=0

        // Act
        let insn = decode_one(word_fcvtzs, 0);

        // Assert: fcvtzs decoded with integer dest (w-reg), float source (s-reg)
        assert_eq!(insn.mnemonic, "fcvtzs");
        assert_eq!(insn.operands[0], "w0");
        assert_eq!(insn.operands[1], "s1");

        // Arrange: fcvtzu w3, d5 — sf=0, ftype=01, rmode=11, opcode=001
        let word_fcvtzu = (0u32 << 31) | (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)
            | (0b11u32 << 19)        // rmode=11
            | (0b001u32 << 16)       // opcode=001 (fcvtzu)
            | (0b000000u32 << 10)
            | (5u32 << 5)            // Rn=5
            | 3u32;                  // Rd=3

        // Act
        let insn_u = decode_one(word_fcvtzu, 0);

        // Assert: fcvtzu decoded with w-reg dest, d-reg source
        assert_eq!(insn_u.mnemonic, "fcvtzu");
        assert_eq!(insn_u.operands[0], "w3");
        assert_eq!(insn_u.operands[1], "d5");
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_fcvt_nonstandard_rmode_sf0() {
        // Arrange: float→int with sf=0, rmode=01, opcode=000
        // rmode != 0b11 produces "fcvt_rm{rmode}_{opcode}" mnemonic
        let word = (0u32 << 31) | (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21)
            | (0b01u32 << 19)        // rmode=01 (not 11)
            | (0b000u32 << 16)       // opcode=000
            | (0b000000u32 << 10)
            | (1u32 << 5) | 0u32;

        // Act
        let insn = decode_one(word, 0);

        // Assert: non-standard rounding mode produces formatted mnemonic
        assert!(insn.mnemonic.starts_with("fcvt_rm"));
        assert_eq!(insn.operands[0], "w0"); // sf=0 → w-reg
        assert_eq!(insn.operands[1], "s1"); // ftype=0 → s-reg
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_fp_2source_double_precision() {
        // Arrange: fmul d3, d7, d11 — double-precision FP 2-source
        // ftype=01 (double), opcode=0000 (fmul), Rm=11, Rn=7, Rd=3
        let word = (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)           // bit21=1
            | (11u32 << 16)          // Rm=11
            | (0b0000u32 << 12)      // opcode=fmul
            | (0b10u32 << 10)        // bits[11:10]=10
            | (7u32 << 5)            // Rn=7
            | 3u32;                  // Rd=3

        // Act
        let insn = decode_one(word, 0);

        // Assert: double-precision registers used
        assert_eq!(insn.mnemonic, "fmul");
        assert_eq!(insn.operands[0], "d3");
        assert_eq!(insn.operands[1], "d7");
        assert_eq!(insn.operands[2], "d11");
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_fp_2source_unknown_opcode() {
        // Arrange: FP 2-source with opcode >= 0b1000 → "fp2src_unknown"
        let word = (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21)
            | (2u32 << 16)
            | (0b1000u32 << 12)      // opcode=1000 (unhandled)
            | (0b10u32 << 10)
            | (1u32 << 5) | 0u32;

        // Act
        let insn = decode_one(word, 0);

        // Assert: unknown opcode produces fallback mnemonic
        assert_eq!(insn.mnemonic, "fp2src_unknown");
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_fp_1source_fabs_double_precision() {
        // Arrange: fabs d5, d7 — double-precision FP 1-source
        // opcode=000001 (fabs) in bits[20:15], ftype=01 (double)
        // 1-source path: bits[21:17]=10000, bit14=0
        let word = (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)           // bit21=1
            | (0b00000u32 << 17)     // bits[20:17]=0000 (combined with bit21 → bits[21:17]=10000)
            | (0u32 << 14)           // bit14=0
            | (0b000001u32 << 15)    // opcode=000001 (fabs) in bits[20:15]
            | (7u32 << 5)            // Rn=7
            | 5u32;                  // Rd=5

        // Act
        let insn = decode_one(word, 0);

        // Assert: fabs decoded with double-precision registers
        assert_eq!(insn.mnemonic, "fabs");
        assert_eq!(insn.operands[0], "d5");
        assert_eq!(insn.operands[1], "d7");
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_tbnz() {
        // Arrange: tbnz x5, #7, #+8 at PC=0x1000
        // Encoding: b5=0, bits[30:25]=011011, op=1 (tbnz), b40=7, imm14=2, rt=5
        // bit_pos = (b5 << 5) | b40 = 0 | 7 = 7
        let word = (0u32 << 31)           // b5=0
            | (0b011_011u32 << 25)        // bits[30:25]=011011
            | (1u32 << 24)                // op=1 (tbnz)
            | (7u32 << 19)                // b40=7
            | (2u32 << 5)                 // imm14=2 → offset = 2*4 = 8
            | 5u32;                       // rt=5

        // Act
        let insn = decode_one(word, 0x1000);

        // Assert: tbnz decoded correctly
        assert_eq!(insn.mnemonic, "tbnz");
        assert!(insn.is_cond_branch);
        assert!(!insn.is_uncond_branch);
        assert_eq!(insn.operands[0], "x5");
        assert_eq!(insn.operands[1], "#7");
        assert_eq!(insn.branch_target, Some(0x1008));
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_branch_unknown_fallback() {
        // Arrange: a branch encoding (bits[28:26]=0b101) that doesn't match any
        // specific sub-pattern in decode_branch.
        // op=bits[31:29]=0b110 with bits[28:26]=0b101 → 0xD4...
        // This is the HVC/SMC/BRK space, not decoded by any branch sub-pattern.
        // None of: b.cond, cbz/cbnz, tbz/tbnz, b, bl, br/blr/ret match.
        let word = 0xD400_00FFu32;

        // Act
        let insn = decode_one(word, 0);

        // Assert: falls to branch fallback ".branch 0x..."
        assert!(insn.mnemonic.starts_with(".branch"), "got: {}", insn.mnemonic);
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_ldr_literal_q_register() {
        // Arrange: LDR Q0, [PC, #offset] — opc=10 → q register
        // bits[29:27]=011, bit[26]=1, bits[25:24]=00
        // opc=10 (q-register), imm19=4, rt=0
        let word = (0b10u32 << 30) | (0b011u32 << 27) | (1u32 << 26)
            | (0b00u32 << 24) | (4u32 << 5) | 0u32;

        // Act
        let insn = decode_one(word, 0x1000);

        // Assert: decoded as ldr with q-register
        assert_eq!(insn.mnemonic, "ldr");
        assert_eq!(insn.operands[0], "q0");
        assert!(insn.operands[1].contains("pc,"));
        assert_eq!(insn.branch_target, Some(0x1000 + 4 * 4)); // imm19=4, offset=16
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_ldr_str_unsigned_offset_q_register() {
        // Arrange: LDR Q7, [X3, #64] — size=10 → q-register, scale=16
        // bits[29:27]=111, bit[26]=1, bits[25:24]=01
        // size=10, opc=01 (load), imm12=4, rn=3, rt=7
        let word_ldr = (0b10u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b01u32 << 24)
            | (0b01u32 << 22) | (4u32 << 10) | (3u32 << 5) | 7u32;

        // Act
        let insn_ldr = decode_one(word_ldr, 0);

        // Assert: q-register load with scaled offset (imm12=4 * scale=16 = 64)
        assert_eq!(insn_ldr.mnemonic, "ldr");
        assert_eq!(insn_ldr.operands[0], "q7");
        assert!(insn_ldr.operands[1].contains("x3"));
        assert!(insn_ldr.operands[1].contains("#64"));

        // Arrange: STR Q2, [X5, #32] — size=10, opc=00 (store), imm12=2, rn=5
        let word_str = (0b10u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b01u32 << 24)
            | (0b00u32 << 22) | (2u32 << 10) | (5u32 << 5) | 2u32;

        // Act
        let insn_str = decode_one(word_str, 0);

        // Assert: q-register store
        assert_eq!(insn_str.mnemonic, "str");
        assert_eq!(insn_str.operands[0], "q2");
        assert!(insn_str.operands[1].contains("x5"));
        assert!(insn_str.operands[1].contains("#32"));
    }

    // @trace TEST-12kau
    #[test]
    fn test_decode_ldp_stp_q_register_pair() {
        // Arrange: LDP Q0, Q1, [X2, #32] — opc=10 (q-register), load
        // Encoding: opc=10 101 1 0xx imm7 Rt2 Rn Rt
        // scale for opc=10 → 16, imm7=2 → offset = 2*16 = 32
        let word_ldp = (0b10u32 << 30) | (0b101u32 << 27) | (1u32 << 26)
            | (1u32 << 22)            // load
            | (2u32 << 15)            // imm7=2
            | (1u32 << 10)            // Rt2=1
            | (2u32 << 5)             // Rn=2
            | 0u32;                   // Rt=0

        // Act
        let insn_ldp = decode_one(word_ldp, 0x4000);

        // Assert: q-register pair load
        assert_eq!(insn_ldp.mnemonic, "ldp");
        assert_eq!(insn_ldp.operands[0], "q0");
        assert_eq!(insn_ldp.operands[1], "q1");
        assert!(insn_ldp.operands[2].contains("x2"));
        assert!(insn_ldp.operands[2].contains("#32")); // imm7=2 * scale=16 = 32

        // Arrange: STP Q3, Q4, [SP, #-16] — opc=10 (q-register), store
        // imm7=-1 → offset = -1*16 = -16
        let imm7_neg1: u32 = ((-1i32) as u32) & 0x7F;
        let word_stp = (0b10u32 << 30) | (0b101u32 << 27) | (1u32 << 26)
            | (0u32 << 22)            // store
            | (imm7_neg1 << 15)       // imm7=-1
            | (4u32 << 10)            // Rt2=4
            | (31u32 << 5)            // Rn=31 (sp)
            | 3u32;                   // Rt=3

        // Act
        let insn_stp = decode_one(word_stp, 0);

        // Assert: q-register pair store to sp
        assert_eq!(insn_stp.mnemonic, "stp");
        assert_eq!(insn_stp.operands[0], "q3");
        assert_eq!(insn_stp.operands[1], "q4");
        assert!(insn_stp.operands[2].contains("sp"));
        assert!(insn_stp.operands[2].contains("-16"));
    }

    // ── Wave 12khf tests (+10, target: 68 total) ───────────────────────

    // @trace TEST-12khf
    #[test]
    fn test_decode_b_negative_offset_backward_branch() {
        // Arrange: b #-32 at PC=0x2000 → target = 0x1FE0
        // Encoding: 000101 imm26, imm26 = -8 (sign-extended, offset = -8*4 = -32)
        let imm26 = ((-8i32) as u32) & 0x03FF_FFFF;
        let word = (0b000101u32 << 26) | imm26;

        // Act
        let insn = decode_one(word, 0x2000);

        // Assert: backward unconditional branch with correct negative offset
        assert_eq!(insn.mnemonic, "b");
        assert!(insn.is_uncond_branch);
        assert!(!insn.is_cond_branch);
        assert!(!insn.is_call);
        assert!(!insn.is_return);
        assert_eq!(insn.branch_target, Some(0x1FE0));
        assert_eq!(insn.operands, vec!["0x1fe0"]);
    }

    // @trace TEST-12khf
    #[test]
    fn test_decode_bcond_ne_and_gt_conditions() {
        // Arrange: b.ne at PC=0x3000 with offset +16 (imm19=4)
        // cond=1 (ne)
        let imm19_ne: u32 = 4;
        let word_ne = (0b0101_0100u32 << 24) | (imm19_ne << 5) | 0b0001u32;

        // Act
        let insn_ne = decode_one(word_ne, 0x3000);

        // Assert: b.ne decoded with correct condition and target
        assert_eq!(insn_ne.mnemonic, "b.ne");
        assert!(insn_ne.is_cond_branch);
        assert_eq!(insn_ne.branch_target, Some(0x3010)); // 0x3000 + 4*4

        // Arrange: b.gt at PC=0x4000 with offset -8 (imm19 = -2 sign-extended)
        // cond=0xC (gt)
        let imm19_gt: u32 = ((-2i32) as u32) & 0x7FFFF;
        let word_gt = (0b0101_0100u32 << 24) | (imm19_gt << 5) | 0b1100u32;

        // Act
        let insn_gt = decode_one(word_gt, 0x4000);

        // Assert: b.gt decoded with backward target
        assert_eq!(insn_gt.mnemonic, "b.gt");
        assert!(insn_gt.is_cond_branch);
        assert_eq!(insn_gt.branch_target, Some(0x3FF8)); // 0x4000 + (-2)*4
    }

    // @trace TEST-12khf
    #[test]
    fn test_decode_tbz_high_bit_position() {
        // Arrange: tbz x0, #39, #+16 at PC=0x1000
        // b5=1, b40=7 → bit_pos = (1 << 5) | 7 = 39
        // imm14=4 → offset = 4*4 = 16, rt=0, op=0 (tbz)
        let word = (1u32 << 31)              // b5=1
            | (0b011_011u32 << 25)            // bits[30:25]=011011
            | (0u32 << 24)                    // op=0 (tbz)
            | (7u32 << 19)                    // b40=7
            | (4u32 << 5)                     // imm14=4
            | 0u32;                           // rt=0

        // Act
        let insn = decode_one(word, 0x1000);

        // Assert: high bit position (>= 32) decoded correctly
        assert_eq!(insn.mnemonic, "tbz");
        assert!(insn.is_cond_branch);
        assert_eq!(insn.operands[0], "x0");
        assert_eq!(insn.operands[1], "#39");
        assert_eq!(insn.branch_target, Some(0x1010)); // 0x1000 + 16
    }

    // @trace TEST-12khf
    #[test]
    fn test_decode_fp_3source_double_precision_fmadd() {
        // Arrange: fmadd d0, d1, d2, d3 — double-precision 3-source
        // ftype=01 (double), o1=0, o0=0 → fmadd
        let word = (0b0001_1111u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (0u32 << 21)           // o1=0
            | (2u32 << 16)           // Rm=2
            | (0u32 << 15)           // o0=0
            | (3u32 << 10)           // Ra=3
            | (1u32 << 5)            // Rn=1
            | 0u32;                  // Rd=0

        // Act
        let insn = decode_one(word, 0);

        // Assert: double-precision fmadd with all d-registers
        assert_eq!(insn.mnemonic, "fmadd");
        assert_eq!(insn.operands.len(), 4);
        assert_eq!(insn.operands[0], "d0");
        assert_eq!(insn.operands[1], "d1");
        assert_eq!(insn.operands[2], "d2");
        assert_eq!(insn.operands[3], "d3");
    }

    // @trace TEST-12khf
    #[test]
    fn test_decode_bcond_always_and_never_conditions() {
        // Arrange: b.al (condition=0xE "always") at PC=0x5000 with offset +4
        let imm19_al: u32 = 1; // offset = 1*4 = 4
        let word_al = (0b0101_0100u32 << 24) | (imm19_al << 5) | 0b1110u32; // cond=0xE (al)

        // Act
        let insn_al = decode_one(word_al, 0x5000);

        // Assert: b.al decoded as conditional branch (the AL condition means always taken)
        assert_eq!(insn_al.mnemonic, "b.al");
        assert!(insn_al.is_cond_branch);
        assert_eq!(insn_al.branch_target, Some(0x5004));

        // Arrange: b.nv (condition=0xF "never") at PC=0x6000 with offset +8
        let imm19_nv: u32 = 2; // offset = 2*4 = 8
        let word_nv = (0b0101_0100u32 << 24) | (imm19_nv << 5) | 0b1111u32; // cond=0xF (nv)

        // Act
        let insn_nv = decode_one(word_nv, 0x6000);

        // Assert: b.nv decoded correctly
        assert_eq!(insn_nv.mnemonic, "b.nv");
        assert!(insn_nv.is_cond_branch);
        assert_eq!(insn_nv.branch_target, Some(0x6008));
    }

    // @trace TEST-12khf
    #[test]
    fn test_bits_single_bit_and_half_word_extraction() {
        // Arrange: word with known bit pattern
        let word = 0xA5A5_A5A5u32; // 1010_0101 pattern repeated

        // Act & Assert: single bit extraction at various positions
        assert_eq!(bits(word, 31, 31), 1); // bit 31 = 1
        assert_eq!(bits(word, 30, 30), 0); // bit 30 = 0
        assert_eq!(bits(word, 0, 0), 1);   // bit 0 = 1
        assert_eq!(bits(word, 1, 1), 0);   // bit 1 = 0

        // Act & Assert: half-word extraction (avoiding hi=31,lo=0 overflow in bits())
        assert_eq!(bits(word, 31, 16), 0xA5A5); // upper 16 bits
        assert_eq!(bits(word, 15, 0), 0xA5A5);  // lower 16 bits

        // Act & Assert: byte extraction
        assert_eq!(bits(word, 23, 16), 0xA5);   // third byte
        assert_eq!(bits(word, 7, 0), 0xA5);      // lowest byte

        // Act & Assert: multi-bit extraction with known values
        // 0xA5 = binary 10100101: bit7=1, bit6=0, bit5=1, bit4=0, bit3=0, bit2=1, bit1=0, bit0=1
        assert_eq!(bits(word, 7, 6), 0b10);   // bits 7:6 = 10 = 2
        assert_eq!(bits(word, 5, 4), 0b10);   // bits 5:4 = 10 = 2
        assert_eq!(bits(word, 3, 2), 0b01);   // bits 3:2 = 01 = 1
        assert_eq!(bits(word, 1, 0), 0b01);   // bits 1:0 = 01 = 1
    }

    // @trace TEST-12khf
    #[test]
    fn test_sign_extend_large_widths() {
        // Arrange & Act & Assert: 64-bit value (bit_width=64) stays unchanged
        // sign_extend with bit_width=64: shift = 0, no change
        assert_eq!(sign_extend(0, 64), 0);
        assert_eq!(sign_extend(1, 64), 1);
        assert_eq!(sign_extend(0xFFFF_FFFF, 64), 0xFFFF_FFFF_i64);

        // Arrange & Act & Assert: 8-bit signed values
        assert_eq!(sign_extend(0x7F, 8), 127);       // max positive 8-bit
        assert_eq!(sign_extend(0x80, 8), -128);       // min negative 8-bit
        assert_eq!(sign_extend(0xFF, 8), -1);          // -1 in 8-bit

        // Arrange & Act & Assert: 14-bit (used in tbz/tbnz imm14)
        // 14-bit all 1s = 0x3FFF, sign-extended = -1
        assert_eq!(sign_extend(0x3FFF, 14), -1);
        // 14-bit MSB set only = 0x2000 = -8192
        assert_eq!(sign_extend(0x2000, 14), -8192);
        // 14-bit max positive = 0x1FFF = 8191
        assert_eq!(sign_extend(0x1FFF, 14), 8191);
    }

    // @trace TEST-12khf
    #[test]
    fn test_a64_iter_sequential_address_increments() {
        // Arrange: 5 instructions at consecutive addresses
        let ret = 0xD65F03C0u32; // ret
        let instructions: Vec<u32> = vec![
            0b0001_1110_0010_0010_0010_1000_0010_0000, // fadd s0, s1, s2
            0b0001_1110_0010_0001_0010_1000_0010_0000, // fmul s0, s1, s1 (approx)
            0b0001_1111_0000_0010_0000_1100_0010_0000, // fmadd s0, s1, s2, s3
            0b0001_1110_0010_0011_0010_1000_0010_0000, // fsub s0, s1, s2
            ret,
        ];
        let mut bytes = Vec::new();
        for instr in &instructions {
            bytes.extend_from_slice(&instr.to_le_bytes());
        }
        let base: u64 = 0x8000;

        // Act
        let insns: Vec<A64Insn> = A64Iter::new(&bytes, base).collect();

        // Assert: all 5 instructions decoded with strictly increasing addresses
        assert_eq!(insns.len(), 5);
        for (i, insn) in insns.iter().enumerate() {
            assert_eq!(insn.addr, base + (i as u64) * 4);
            assert_eq!(insn.len, 4);
        }
        assert_eq!(insns[4].mnemonic, "ret");
        assert!(insns[4].is_return);
    }

    // @trace TEST-12khf
    #[test]
    fn test_is_integer_mnemonic_comprehensive_edge_cases() {
        // Arrange & Act & Assert: less common integer mnemonics
        assert!(is_integer_mnemonic("sxtw"), "sxtw should be integer");
        assert!(is_integer_mnemonic("uxtw"), "uxtw should be integer");
        assert!(is_integer_mnemonic("cmn"), "cmn should be integer");
        assert!(is_integer_mnemonic("asr"), "asr should be integer");
        assert!(is_integer_mnemonic("lsl"), "lsl should be integer");
        assert!(is_integer_mnemonic("lsr"), "lsr should be integer");
        assert!(is_integer_mnemonic("nop"), "nop should be integer");
        assert!(is_integer_mnemonic("mul"), "mul should be integer");
        assert!(is_integer_mnemonic("sdiv"), "sdiv should be integer");
        assert!(is_integer_mnemonic("udiv"), "udiv should be integer");

        // Float/load mnemonics should NOT be classified as integer
        assert!(!is_integer_mnemonic("ldr"), "ldr should not be integer");
        assert!(!is_integer_mnemonic("str"), "str should not be integer");
        assert!(!is_integer_mnemonic("ldp"), "ldp should not be integer");
        assert!(!is_integer_mnemonic("stp"), "stp should not be integer");
        assert!(!is_integer_mnemonic("fmadd"), "fmadd should not be integer");
        assert!(!is_integer_mnemonic("b"), "b should not be integer");
    }

    // @trace TEST-12khf
    #[test]
    fn test_parse_base_reg_x31_numeric_and_various_offsets() {
        // Arrange & Act & Assert: x31 parsed as index 31 (not "sp" string)
        assert_eq!(parse_base_reg("[x31, #0]"), Some(31));
        assert_eq!(parse_base_reg("[x31, #128]"), Some(31));
        assert_eq!(parse_base_reg("[x31]"), Some(31));

        // Various valid x registers
        assert_eq!(parse_base_reg("[x7, #256]"), Some(7));
        assert_eq!(parse_base_reg("[x16, #-16]"), Some(16));
        assert_eq!(parse_base_reg("[x30, #0]"), Some(30));

        // Invalid: bracket notation without register prefix
        assert_eq!(parse_base_reg("[, #16]"), None);
        assert_eq!(parse_base_reg("[,]"), None);
    }

    // ── Wave 12x33 tests (+10, target: 78 total) ───────────────────────

    // @trace TEST-12x33
    #[test]
    fn test_decode_bl_large_positive_offset() {
        // Arrange: bl #+1MB at PC=0x10000 → target = 0x10000 + 1048576 = 0x110000
        // imm26 = 262144 (offset = 262144*4 = 1048576)
        let imm26: u32 = 262144;
        let word = (0b100101u32 << 26) | imm26;

        // Act
        let insn = decode_one(word, 0x10000);

        // Assert: large forward call with correct target
        assert_eq!(insn.mnemonic, "bl");
        assert!(insn.is_call);
        assert!(!insn.is_return);
        assert_eq!(insn.branch_target, Some(0x110000));
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_bcond_mi_and_pl_conditions() {
        // Arrange: b.mi (condition=4 "minus/negative") at PC=0x1000, offset +12
        let imm19: u32 = 3; // offset = 3*4 = 12
        let word_mi = (0b0101_0100u32 << 24) | (imm19 << 5) | 0b0100u32;

        // Act
        let insn_mi = decode_one(word_mi, 0x1000);

        // Assert: b.mi decoded with minus condition
        assert_eq!(insn_mi.mnemonic, "b.mi");
        assert!(insn_mi.is_cond_branch);
        assert_eq!(insn_mi.branch_target, Some(0x100C));

        // Arrange: b.pl (condition=5 "plus/positive") at PC=0x2000, offset -4
        let imm19_neg: u32 = ((-1i32) as u32) & 0x7FFFF;
        let word_pl = (0b0101_0100u32 << 24) | (imm19_neg << 5) | 0b0101u32;

        // Act
        let insn_pl = decode_one(word_pl, 0x2000);

        // Assert: b.pl with backward branch
        assert_eq!(insn_pl.mnemonic, "b.pl");
        assert_eq!(insn_pl.branch_target, Some(0x1FFC));
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_fp_2source_single_fsub_registers() {
        // Arrange: fsub s7, s15, s31 — single-precision subtraction
        // opcode=0011 (fsub), ftype=00, Rm=31, Rn=15, Rd=7
        let word = (0b0001_1110u32 << 24)
            | (0u32 << 22)           // ftype=00 (single)
            | (1u32 << 21)
            | (31u32 << 16)          // Rm=31
            | (0b0011u32 << 12)      // opcode=fsub
            | (0b10u32 << 10)
            | (15u32 << 5)           // Rn=15
            | 7u32;                  // Rd=7

        // Act
        let insn = decode_one(word, 0);

        // Assert: high-numbered s-registers preserved correctly
        assert_eq!(insn.mnemonic, "fsub");
        assert_eq!(insn.operands, vec!["s7", "s15", "s31"]);
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_fp_1source_fneg_single_precision() {
        // Arrange: fneg s20, s10 — single-precision negate
        // opcode=000010 (fneg), ftype=00 (single)
        let word = (0b0001_1110u32 << 24)
            | (0u32 << 22)           // ftype=00
            | (1u32 << 21)
            | (0u32 << 17)           // bits[20:17]=0000
            | (0b000010u32 << 10)    // opcode=fneg
            | (10u32 << 5)           // Rn=10
            | 20u32;                 // Rd=20

        // Act
        let insn = decode_one(word, 0);

        // Assert: fneg decoded with correct single-precision registers
        assert_eq!(insn.mnemonic, "fneg");
        assert_eq!(insn.operands[0], "s20");
        assert_eq!(insn.operands[1], "s10");
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_fp_3source_fnmadd_double_precision() {
        // Arrange: fnmadd d1, d2, d3, d4 — double-precision negated multiply-add
        // ftype=01 (double), o1=1, o0=0
        let word = (0b0001_1111u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)           // o1=1
            | (3u32 << 16)           // Rm=3
            | (0u32 << 15)           // o0=0
            | (4u32 << 10)           // Ra=4
            | (2u32 << 5)            // Rn=2
            | 1u32;                  // Rd=1

        // Act
        let insn = decode_one(word, 0);

        // Assert: fnmadd with all double-precision registers
        assert_eq!(insn.mnemonic, "fnmadd");
        assert_eq!(insn.operands, vec!["d1", "d2", "d3", "d4"]);
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_fp_fcsel_double_precision_ge_condition() {
        // Arrange: fcsel d0, d1, d2, ge — double-precision conditional select
        // ftype=01 (double), cond=0xA (ge)
        let word = (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)
            | (2u32 << 16)           // Rm=2
            | (0b1010u32 << 12)      // cond=ge
            | (0b11u32 << 10)
            | (1u32 << 5)            // Rn=1
            | 0u32;                  // Rd=0

        // Act
        let insn = decode_one(word, 0);

        // Assert: double-precision fcsel with "ge" condition
        assert_eq!(insn.mnemonic, "fcsel");
        assert_eq!(insn.operands, vec!["d0", "d1", "d2", "ge"]);
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_cbz_32bit_w_register() {
        // Arrange: cbz w7, #+20 at PC=0x5000
        // sf=0 (32-bit), op=0 (cbz), imm19=5, rt=7
        let imm19: u32 = 5; // offset = 5*4 = 20
        let word = (0u32 << 31)                // sf=0
            | (0b011_0100u32 << 25)
            | (0u32 << 24)                     // op=0 (cbz)
            | (imm19 << 5)
            | 7u32;                            // rt=7

        // Act
        let insn = decode_one(word, 0x5000);

        // Assert: 32-bit cbz with w-register
        assert_eq!(insn.mnemonic, "cbz");
        assert!(insn.is_cond_branch);
        assert!(!insn.is_uncond_branch);
        assert_eq!(insn.operands[0], "w7");
        assert_eq!(insn.branch_target, Some(0x5014));
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_dp_imm_sub_32bit_w_register() {
        // Arrange: sub w3, ... — sf=0, bits[28:26]=100, bits[25:23]=001, op=1
        let word = (0u32 << 31)              // sf=0 (32-bit)
            | (1u32 << 30)                   // op=1 (sub)
            | (0b100u32 << 26)
            | (0b001u32 << 23)
            | (3u32 << 0);                   // Rd=3

        // Act
        let insn = decode_one(word, 0);

        // Assert: 32-bit sub uses w-register
        assert_eq!(insn.mnemonic, "sub");
        assert_eq!(insn.operands[0], "w3");
    }

    // @trace TEST-12x33
    #[test]
    fn test_decode_dp_reg_csel_32bit_w_registers() {
        // Arrange: csel w0, w1, w2, hi — 32-bit conditional select
        // sf=0, bits[29:21]=011010100, op=0, op2=0, cond=0x8 (hi)
        let word = (0u32 << 31)                // sf=0 (32-bit)
            | (0b011010100u32 << 21)
            | (0u32 << 30)                     // op=0
            | (0u32 << 10)                     // op2=0
            | (2u32 << 16)                     // Rm=2
            | (0b1000u32 << 12)                // cond=hi
            | (1u32 << 5)                      // Rn=1
            | 0u32;                            // Rd=0

        // Act
        let insn = decode_one(word, 0);

        // Assert: all operands use w-registers
        assert_eq!(insn.mnemonic, "csel");
        assert_eq!(insn.operands[0], "w0");
        assert_eq!(insn.operands[1], "w1");
        assert_eq!(insn.operands[2], "w2");
        assert_eq!(insn.operands[3], "hi");
    }

    // @trace TEST-12x33
    #[test]
    fn test_a64_iter_multiple_ret_instructions() {
        // Arrange: two ret instructions — iterator decodes both, neither stops iteration
        // (the iterator itself does not stop at ret; that logic is in analyze_scalar_fn)
        let ret_word = 0xD65F03C0u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ret_word.to_le_bytes());
        bytes.extend_from_slice(&ret_word.to_le_bytes());

        // Act
        let insns: Vec<A64Insn> = A64Iter::new(&bytes, 0x2000).collect();

        // Assert: both ret instructions decoded
        assert_eq!(insns.len(), 2);
        assert!(insns[0].is_return);
        assert!(insns[1].is_return);
        assert_eq!(insns[0].addr, 0x2000);
        assert_eq!(insns[1].addr, 0x2004);
    }

    // ── Wave 12x60 tests (+10, target: 88 total) ───────────────────────

    // @trace TEST-12x60
    #[test]
    fn test_decode_fdiv_single_and_double_precision() {
        // Arrange: fdiv s10, s20, s30 — single-precision FP divide
        // opcode=0001 (fdiv), ftype=00 (single), Rm=30, Rn=20, Rd=10
        let word_single = (0b0001_1110u32 << 24)
            | (0u32 << 22)           // ftype=00 (single)
            | (1u32 << 21)
            | (30u32 << 16)          // Rm=30
            | (0b0001u32 << 12)      // opcode=fdiv
            | (0b10u32 << 10)
            | (20u32 << 5)           // Rn=20
            | 10u32;                 // Rd=10

        // Act
        let insn_single = decode_one(word_single, 0);

        // Assert: single-precision fdiv with correct s-register operands
        assert_eq!(insn_single.mnemonic, "fdiv");
        assert_eq!(insn_single.operands, vec!["s10", "s20", "s30"]);

        // Arrange: fdiv d0, d1, d2 — double-precision FP divide
        // ftype=01 (double), Rm=2, Rn=1, Rd=0
        let word_double = (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)
            | (2u32 << 16)           // Rm=2
            | (0b0001u32 << 12)      // opcode=fdiv
            | (0b10u32 << 10)
            | (1u32 << 5)            // Rn=1
            | 0u32;                  // Rd=0

        // Act
        let insn_double = decode_one(word_double, 0);

        // Assert: double-precision fdiv uses d-registers
        assert_eq!(insn_double.mnemonic, "fdiv");
        assert_eq!(insn_double.operands, vec!["d0", "d1", "d2"]);
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_bl_negative_large_offset() {
        // Arrange: bl #-1MB at PC=0x200000 → target = 0x200000 - 1048576 = 0x100000
        // imm26 = -262144 (sign-extended), offset = -262144*4 = -1048576
        let imm26: u32 = ((-262144i32) as u32) & 0x03FF_FFFF;
        let word = (0b100101u32 << 26) | imm26;

        // Act
        let insn = decode_one(word, 0x200000);

        // Assert: large backward call decoded correctly
        assert_eq!(insn.mnemonic, "bl");
        assert!(insn.is_call);
        assert!(!insn.is_return);
        assert!(insn.branch_target.unwrap() < 0x200000);
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_cbz_cbnz_64bit_with_backward_branch() {
        // Arrange: cbnz x15, #-16 at PC=0x3000 → target = 0x2FF0
        // Encoding: sf[31]=1, bits[30:25]=011010, op[24]=1 (cbnz), imm19, Rt
        // bits[28:26]=101 ensures entry into decode_branch
        // bits[30:25]=011010 matches the cbz/cbnz pattern
        let imm19: u32 = ((-4i32) as u32) & 0x7FFFF;
        let word = (1u32 << 31)                  // sf=1 (64-bit)
            | (0b011010u32 << 25)                // bits[30:25] = 011010
            | (1u32 << 24)                       // op=1 (cbnz)
            | (imm19 << 5)
            | 15u32;                             // rt=15

        // Act
        let insn = decode_one(word, 0x3000);

        // Assert: 64-bit cbnz with x15 register and backward target
        assert_eq!(insn.mnemonic, "cbnz");
        assert!(insn.is_cond_branch);
        assert_eq!(insn.operands[0], "x15");
        assert!(insn.branch_target.unwrap() < 0x3000);
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_fp_1source_fsqrt_double_precision() {
        // Arrange: fsqrt d15, d20 — double-precision square root
        // FP 1-source: 0001_1110_xx1_00000_0xxxxx_xxxxx_xxxxx
        // bits[21:17]=10000, bit14=0, opcode in bits[20:15]
        // opcode=000011 (fsqrt): bits[20:17]=0000 (from constraint), bits[16:15]=11
        let word = (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)           // bit21=1
            | (0u32 << 17)           // bits[20:17]=0000 → bits[21:17]=10000
            | (0u32 << 14)           // bit14=0
            | (0b000011u32 << 15)    // opcode=fsqrt in bits[20:15] (bits 16:15 = 11)
            | (20u32 << 5)           // Rn=20
            | 15u32;                 // Rd=15

        // Act
        let insn = decode_one(word, 0);

        // Assert: fsqrt decoded with double-precision register pair
        assert_eq!(insn.mnemonic, "fsqrt");
        assert_eq!(insn.operands[0], "d15");
        assert_eq!(insn.operands[1], "d20");
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_fp_3source_fmsub_double_precision() {
        // Arrange: fmsub d5, d10, d15, d20 — double-precision fused multiply-subtract
        // ftype=01 (double), o1=0, o0=1 → fmsub
        let word = (0b0001_1111u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (0u32 << 21)           // o1=0
            | (15u32 << 16)          // Rm=15
            | (1u32 << 15)           // o0=1 → fmsub
            | (20u32 << 10)          // Ra=20
            | (10u32 << 5)           // Rn=10
            | 5u32;                  // Rd=5

        // Act
        let insn = decode_one(word, 0);

        // Assert: fmsub with all double-precision registers
        assert_eq!(insn.mnemonic, "fmsub");
        assert_eq!(insn.operands.len(), 4);
        assert_eq!(insn.operands[0], "d5");
        assert_eq!(insn.operands[1], "d10");
        assert_eq!(insn.operands[2], "d15");
        assert_eq!(insn.operands[3], "d20");
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_ret_default_x30_no_operands() {
        // Arrange: ret (with default x30 link register) — standard encoding 0xD65F03C0
        // rn=30 → no operands emitted (ret defaults to x30)
        let word = 0xD65F03C0u32;

        // Act
        let insn = decode_one(word, 0x1000);

        // Assert: ret with default x30 has no operands
        assert_eq!(insn.mnemonic, "ret");
        assert!(insn.is_return);
        assert!(!insn.is_call);
        assert!(!insn.is_cond_branch);
        assert!(!insn.is_uncond_branch);
        assert!(insn.operands.is_empty(), "ret x30 should have no operands");
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_bl_backward_call_to_self() {
        // Arrange: bl #-4 at PC=0x1000 → target = 0xFFC (calls itself)
        // Encoding: 100101 imm26, imm26 = -1 (sign-extended, offset = -1*4 = -4)
        let imm26: u32 = ((-1i32) as u32) & 0x03FF_FFFF;
        let word = (0b100101u32 << 26) | imm26;

        // Act
        let insn = decode_one(word, 0x1000);

        // Assert: backward bl with target just before PC
        assert_eq!(insn.mnemonic, "bl");
        assert!(insn.is_call);
        assert!(!insn.is_return);
        assert!(!insn.is_cond_branch);
        assert!(insn.branch_target.unwrap() < 0x1000);
        assert_eq!(insn.branch_target, Some(0x0FFC));
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_ldr_str_unsigned_offset_d_register_high_numbered() {
        // Arrange: STR D30, [X20, #80] — store high-numbered d-register
        // size=01 (d-reg), opc=00 (store), imm12=10, rn=20, rt=30
        let word = (0b01u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b01u32 << 24)
            | (0b00u32 << 22) | (10u32 << 10) | (20u32 << 5) | 30u32;

        // Act
        let insn = decode_one(word, 0);

        // Assert: d30 stored with correct base and offset (10 * 8 = 80)
        assert_eq!(insn.mnemonic, "str");
        assert_eq!(insn.operands[0], "d30");
        assert!(insn.operands[1].contains("x20"));
        assert!(insn.operands[1].contains("#80"));

        // Arrange: LDR D31, [SP, #96] — load from sp with high d-register
        // size=01, opc=01 (load), imm12=12, rn=31 (sp), rt=31
        let word_ldr = (0b01u32 << 30) | (0b111u32 << 27) | (1u32 << 26) | (0b01u32 << 24)
            | (0b01u32 << 22) | (12u32 << 10) | (31u32 << 5) | 31u32;

        // Act
        let insn_ldr = decode_one(word_ldr, 0);

        // Assert: d31 loaded from sp with offset 12*8=96
        assert_eq!(insn_ldr.mnemonic, "ldr");
        assert_eq!(insn_ldr.operands[0], "d31");
        assert!(insn_ldr.operands[1].contains("sp"));
        assert!(insn_ldr.operands[1].contains("#96"));
    }

    // @trace TEST-12x60
    #[test]
    fn test_decode_fp_compare_fcmp_double_precision() {
        // Arrange: fcmp d0, d16 — double-precision compare
        // FP compare: 0001_1110_xx1_xxxxx_0010_00_xxxxx_x0000
        // bits[31:24]=00011110, bit21=1, bits[13:10]=1000, bits[2:0]=000
        // Use Rm=16 (bit20=1) so bits[21:17]=11000 ≠ 10000 (avoids 1-source match)
        // ftype=01 (double), Rm=16, Rn=0, opc=00 (fcmp)
        let word = (0b0001_1110u32 << 24)
            | (1u32 << 22)           // ftype=01 (double)
            | (1u32 << 21)
            | (16u32 << 16)          // Rm=16 (bit20=1, avoids 1-source pattern)
            | (0b1000u32 << 10)      // bits[13:10]=1000
            | (0u32 << 5)            // Rn=0
            | (0b000u32 << 3)        // opc=00 (fcmp)
            | 0u32;                  // bits[2:0]=000

        // Act
        let insn = decode_one(word, 0);

        // Assert: double-precision fcmp uses d-registers
        assert_eq!(insn.mnemonic, "fcmp");
        assert_eq!(insn.operands[0], "d0", "first operand should be d0 (Rn=0, ftype=01)");
        assert_eq!(insn.operands[1], "d16", "second operand should be d16 (Rm=16, ftype=01)");
    }

    // @trace TEST-12x60
    #[test]
    fn test_a64_iter_mixed_opcode_sequence() {
        // Arrange: a realistic mixed sequence of 6 instructions:
        // fadd → fmul → ldr literal → b.cond → bl → ret
        let fadd: u32 = 0b0001_1110_0010_0010_0010_1000_0010_0000; // fadd s0, s1, s2
        let fmul_word = (0b0001_1110u32 << 24)
            | (0u32 << 22) | (1u32 << 21) | (2u32 << 16)
            | (0b0000u32 << 12) | (0b10u32 << 10) | (1u32 << 5) | 0u32; // fmul s0, s1, s2
        let ldr_lit = (0b00u32 << 30) | (0b011u32 << 27) | (1u32 << 26)
            | (0b00u32 << 24) | (1u32 << 5) | 0u32; // ldr s0, [pc, #4]
        let bcond_eq = (0b0101_0100u32 << 24) | (1u32 << 5) | 0b0000u32; // b.eq #+4
        let bl_word = (0b100101u32 << 26) | 16u32; // bl #+64
        let ret_word = 0xD65F03C0u32;

        let mut bytes = Vec::new();
        for instr in &[fadd, fmul_word, ldr_lit, bcond_eq, bl_word, ret_word] {
            bytes.extend_from_slice(&instr.to_le_bytes());
        }
        let base: u64 = 0x10000;

        // Act
        let insns: Vec<A64Insn> = A64Iter::new(&bytes, base).collect();

        // Assert: all 6 instructions decoded with sequential addresses
        assert_eq!(insns.len(), 6);

        // fadd at 0x10000
        assert_eq!(insns[0].mnemonic, "fadd");
        assert_eq!(insns[0].addr, 0x10000);

        // fmul at 0x10004
        assert_eq!(insns[1].mnemonic, "fmul");
        assert_eq!(insns[1].addr, 0x10004);

        // ldr literal at 0x10008 — has branch_target for constant pool
        assert_eq!(insns[2].mnemonic, "ldr");
        assert_eq!(insns[2].addr, 0x10008);
        assert!(insns[2].branch_target.is_some());

        // b.eq at 0x1000C — conditional branch
        assert_eq!(insns[3].mnemonic, "b.eq");
        assert!(insns[3].is_cond_branch);
        assert_eq!(insns[3].addr, 0x1000C);

        // bl at 0x10010 — call
        assert_eq!(insns[4].mnemonic, "bl");
        assert!(insns[4].is_call);
        assert_eq!(insns[4].addr, 0x10010);

        // ret at 0x10014
        assert_eq!(insns[5].mnemonic, "ret");
        assert!(insns[5].is_return);
        assert_eq!(insns[5].addr, 0x10014);
    }
}
