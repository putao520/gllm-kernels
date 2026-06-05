use std::collections::HashMap;
use super::sym_value::{SymValue, LibmFn, SelectKind};
use crate::compiler::trace::{TraceOp, ValueId};

#[derive(Debug)]
pub enum SymExecError {
    DisassemblyFailed(String),
    UnsupportedInstruction(String),
    NoReturnValue,
}

impl std::fmt::Display for SymExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymExecError::DisassemblyFailed(s) => write!(f, "disassembly failed: {s}"),
            SymExecError::UnsupportedInstruction(s) => write!(f, "unsupported instruction: {s}"),
            SymExecError::NoReturnValue => write!(f, "no return value found"),
        }
    }
}

impl std::error::Error for SymExecError {}

/// Comparison flags from ucomiss/comiss.
#[derive(Debug, Clone)]
struct CmpFlags {
    lhs: SymValue,
    rhs: SymValue,
}

/// Symbolic execution state: maps register names to symbolic values,
/// with stack spill tracking and constant pool support.
#[derive(Clone)]
pub struct SymbolicExecutor {
    regs: HashMap<String, SymValue>,
    /// Stack spill slots: offset from RSP → symbolic value.
    stack: HashMap<i64, SymValue>,
    /// Constant pool: absolute address → f32 value.
    constants: HashMap<u64, f32>,
    /// Last comparison flags (from ucomiss/comiss).
    flags: Option<CmpFlags>,
}

#[cfg(target_arch = "x86_64")]
const FLOAT_ARG_REGS: &[&str] = &["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"];
#[cfg(target_arch = "x86_64")]
const PTR_ARG_REGS: &[&str] = &["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

#[cfg(target_arch = "aarch64")]
const FLOAT_ARG_REGS: &[&str] = &["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"];
#[cfg(target_arch = "aarch64")]
const PTR_ARG_REGS: &[&str] = &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const FLOAT_ARG_REGS: &[&str] = &[];
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const PTR_ARG_REGS: &[&str] = &[];

/// ABS mask: 0x7FFFFFFF clears sign bit → absolute value.
const ABS_MASK_BITS: u32 = 0x7FFF_FFFF;
/// Sign mask: 0x80000000 flips sign bit → negation via xorps.
const SIGN_MASK_BITS: u32 = 0x8000_0000;

// ---------------------------------------------------------------------------
// Memory operand helpers
// ---------------------------------------------------------------------------

/// Parse `[rsp+0x10]`, `[rsp-8]`, `[rsp]` → stack offset.
fn parse_stack_offset(op: &str) -> Option<i64> {
    let inner = op.strip_prefix('[')?.strip_suffix(']')?;
    let rest = inner.strip_prefix("rsp")?;
    if rest.is_empty() {
        return Some(0);
    }
    if let Some(hex) = rest.strip_prefix("+0x") {
        return i64::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = rest.strip_prefix("-0x") {
        return i64::from_str_radix(hex, 16).ok().map(|v| -v);
    }
    if let Some(dec) = rest.strip_prefix('+') {
        return dec.parse().ok();
    }
    if let Some(dec) = rest.strip_prefix('-') {
        return dec.parse::<i64>().ok().map(|v| -v);
    }
    None
}

/// Parse `[rip+0x1234]` or `[0x7f...]` → absolute address for constant pool.
fn parse_rip_addr(op: &str) -> Option<u64> {
    let inner = op.strip_prefix('[')?.strip_suffix(']')?;
    if let Some(hex) = inner.strip_prefix("rip+0x") {
        return u64::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = inner.strip_prefix("0x") {
        return u64::from_str_radix(hex, 16).ok();
    }
    None
}

fn is_memory_operand(op: &str) -> bool {
    op.starts_with('[') && op.ends_with(']')
}

fn is_xmm_reg(op: &str) -> bool {
    op.starts_with("xmm") || op.starts_with("ymm") || op.starts_with("zmm")
}

fn is_aarch64_float_reg(op: &str) -> bool {
    op.starts_with('s') || op.starts_with('d') || op.starts_with('v') || op.starts_with('q')
}

/// Parse AArch64 memory operand `[x0, #16]` or `[sp, #-32]` → (base_reg, offset).
fn parse_aarch64_mem(op: &str) -> Option<(&str, i64)> {
    let inner = op.strip_prefix('[')?.strip_suffix(']')?;
    let parts: Vec<&str> = inner.splitn(2, ',').collect();
    let base = parts[0].trim();
    if parts.len() == 1 {
        return Some((base, 0));
    }
    let offset_str = parts[1].trim().strip_prefix('#').unwrap_or(parts[1].trim());
    let offset = offset_str.parse::<i64>().ok()?;
    Some((base, offset))
}

/// Check if an f32 constant (by bits) is the ABS mask.
fn is_abs_mask_f32(v: f32) -> bool {
    v.to_bits() == ABS_MASK_BITS
}

/// Check if an f32 constant (by bits) is the sign mask.
fn is_sign_mask_f32(v: f32) -> bool {
    v.to_bits() == SIGN_MASK_BITS
}

impl SymbolicExecutor {
    /// Create a new executor with `n_float_args` float params and `n_ptr_args` pointer params.
    pub fn new(n_float_args: usize, n_ptr_args: usize) -> Self {
        let mut regs = HashMap::new();
        for i in 0..n_float_args.min(FLOAT_ARG_REGS.len()) {
            regs.insert(FLOAT_ARG_REGS[i].to_string(), SymValue::Param(i));
        }
        for i in 0..n_ptr_args.min(PTR_ARG_REGS.len()) {
            regs.insert(
                PTR_ARG_REGS[i].to_string(),
                SymValue::Param(n_float_args + i),
            );
        }
        Self {
            regs,
            stack: HashMap::new(),
            constants: HashMap::new(),
            flags: None,
        }
    }

    /// Register a constant at an absolute address (for `movss xmm, [rip+X]` resolution).
    pub fn register_constant(&mut self, addr: u64, value: f32) {
        self.constants.insert(addr, value);
    }

    /// Check if a constant is already registered at an address.
    pub fn has_constant(&self, addr: u64) -> bool {
        self.constants.contains_key(&addr)
    }

    /// Get the symbolic value of a register, or Unknown.
    fn get(&self, reg: &str) -> SymValue {
        self.regs
            .get(reg)
            .cloned()
            .unwrap_or_else(|| SymValue::Unknown(reg.to_string()))
    }

    /// Public accessor for the symbolic value of a register.
    pub fn get_value(&self, reg: &str) -> SymValue {
        self.get(reg)
    }

    /// Set a register to a symbolic value.
    pub fn set(&mut self, reg: &str, val: SymValue) {
        self.regs.insert(reg.to_string(), val);
    }

    /// Resolve an operand: register name → get value, memory → load from stack/constants.
    fn resolve(&self, op: &str) -> SymValue {
        if is_memory_operand(op) {
            // Stack spill load
            if let Some(offset) = parse_stack_offset(op) {
                if let Some(val) = self.stack.get(&offset) {
                    return val.clone();
                }
            }
            // Constant pool load
            if let Some(addr) = parse_rip_addr(op) {
                if let Some(&val) = self.constants.get(&addr) {
                    return SymValue::Const(val as f64);
                }
            }
            return SymValue::Unknown(format!("mem:{op}"));
        }
        self.get(op)
    }

    /// Return the symbolic value in xmm0 (the return register for floats on x86_64).
    pub fn return_value(&self) -> Result<SymValue, SymExecError> {
        // x86_64: xmm0, AArch64: v0/s0/d0
        if let Some(v) = self.regs.get("xmm0") {
            return Ok(v.clone());
        }
        if let Some(v) = self.regs.get("s0") {
            return Ok(v.clone());
        }
        if let Some(v) = self.regs.get("d0") {
            return Ok(v.clone());
        }
        if let Some(v) = self.regs.get("v0") {
            return Ok(v.clone());
        }
        Err(SymExecError::NoReturnValue)
    }

    /// Execute a single instruction line (mnemonic + operands).
    pub fn step(&mut self, mnemonic: &str, operands: &[&str]) -> Result<(), SymExecError> {
        match mnemonic {
            // --- Arithmetic (scalar float) ---
            "addss" | "vaddss" => self.bin_op(operands, |a, b| SymValue::Add(Box::new(a), Box::new(b))),
            "subss" | "vsubss" => self.bin_op(operands, |a, b| SymValue::Sub(Box::new(a), Box::new(b))),
            "mulss" | "vmulss" => self.bin_op(operands, |a, b| SymValue::Mul(Box::new(a), Box::new(b))),
            "divss" | "vdivss" => self.bin_op(operands, |a, b| SymValue::Div(Box::new(a), Box::new(b))),
            "addps" | "vaddps" => self.bin_op(operands, |a, b| SymValue::Add(Box::new(a), Box::new(b))),
            "mulps" | "vmulps" => self.bin_op(operands, |a, b| SymValue::Mul(Box::new(a), Box::new(b))),
            "subps" | "vsubps" => self.bin_op(operands, |a, b| SymValue::Sub(Box::new(a), Box::new(b))),
            "divps" | "vdivps" => self.bin_op(operands, |a, b| SymValue::Div(Box::new(a), Box::new(b))),

            // --- FMA (vfmaddXXXss/ps) ---
            // vfmadd132ss dst, src1, src2 → dst = dst * src2 + src1
            "vfmadd132ss" | "vfmadd132ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Fma(Box::new(d), Box::new(s2), Box::new(s1))
            }),
            // vfmadd213ss dst, src1, src2 → dst = src1 * dst + src2
            "vfmadd213ss" | "vfmadd213ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Fma(Box::new(s1), Box::new(d), Box::new(s2))
            }),
            // vfmadd231ss dst, src1, src2 → dst = src1 * src2 + dst
            "vfmadd231ss" | "vfmadd231ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Fma(Box::new(s1), Box::new(s2), Box::new(d))
            }),
            // vfmsub variants: a*b - c
            "vfmsub132ss" | "vfmsub132ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Sub(
                    Box::new(SymValue::Mul(Box::new(d), Box::new(s2))),
                    Box::new(s1),
                )
            }),
            "vfmsub213ss" | "vfmsub213ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Sub(
                    Box::new(SymValue::Mul(Box::new(s1), Box::new(d))),
                    Box::new(s2),
                )
            }),
            "vfmsub231ss" | "vfmsub231ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Sub(
                    Box::new(SymValue::Mul(Box::new(s1), Box::new(s2))),
                    Box::new(d),
                )
            }),
            // vfnmadd variants: -a*b + c  (negate-multiply-add)
            "vfnmadd132ss" | "vfnmadd132ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Sub(
                    Box::new(s1),
                    Box::new(SymValue::Mul(Box::new(d), Box::new(s2))),
                )
            }),
            "vfnmadd213ss" | "vfnmadd213ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Sub(
                    Box::new(s2),
                    Box::new(SymValue::Mul(Box::new(s1), Box::new(d))),
                )
            }),
            "vfnmadd231ss" | "vfnmadd231ps" => self.fma_op(operands, |d, s1, s2| {
                SymValue::Sub(
                    Box::new(d),
                    Box::new(SymValue::Mul(Box::new(s1), Box::new(s2))),
                )
            }),

            // --- Max / Min ---
            "maxss" | "vmaxss" => self.bin_op(operands, |a, b| SymValue::Max(Box::new(a), Box::new(b))),
            "minss" | "vminss" => self.bin_op(operands, |a, b| SymValue::Min(Box::new(a), Box::new(b))),
            "maxps" | "vmaxps" => self.bin_op(operands, |a, b| SymValue::Max(Box::new(a), Box::new(b))),
            "minps" | "vminps" => self.bin_op(operands, |a, b| SymValue::Min(Box::new(a), Box::new(b))),

            // --- Unary math ---
            "sqrtss" | "vsqrtss" => {
                if operands.len() >= 2 {
                    // SAFETY: .last() is Some because operands.len() >= 2
                    let src = self.resolve(operands.last().expect("SAFETY: guarded by operands.len() >= 2"));
                    self.set(operands[0], SymValue::Sqrt(Box::new(src)));
                }
                Ok(())
            }
            "rcpss" | "vrcpss" => {
                if operands.len() >= 2 {
                    // SAFETY: .last() is Some because operands.len() >= 2
                    let src = self.resolve(operands.last().expect("SAFETY: guarded by operands.len() >= 2"));
                    self.set(operands[0], SymValue::Recip(Box::new(src)));
                }
                Ok(())
            }
            "rsqrtss" | "vrsqrtss" => {
                if operands.len() >= 2 {
                    // SAFETY: .last() is Some because operands.len() >= 2
                    let src = self.resolve(operands.last().expect("SAFETY: guarded by operands.len() >= 2"));
                    self.set(operands[0], SymValue::Rsqrt(Box::new(src)));
                }
                Ok(())
            }

            // --- Moves (with memory support) ---
            "movss" | "vmovss" | "movaps" | "vmovaps" | "movups" | "vmovups"
            | "movd" | "vmovd" => {
                self.mov_op(operands)
            }

            // --- Bitwise (abs / sign manipulation) ---
            "andps" | "vandps" | "andpd" | "vandpd" => self.bitwise_and_op(operands),
            "orps" | "vorps" | "orpd" | "vorpd" => {
                // orps: if one operand is sign mask → unknown (rare pattern)
                // For now, just propagate as unknown unless both are const.
                self.bitwise_generic_op(operands, "or")
            }
            "andnps" | "vandnps" | "andnpd" | "vandnpd" => {
                // andnps dst, src → dst = (~dst) & src
                // Used for sign extraction; treat as unknown for now.
                self.bitwise_generic_op(operands, "andnot")
            }

            // --- XOR (zero idiom + sign flip) ---
            "xorps" | "vxorps" | "pxor" | "vpxor" | "xorpd" | "vxorpd" => {
                self.xor_op(operands)
            }

            // --- Negation injected by decoder (xorps with sign mask) ---
            "neg_float" => {
                if !operands.is_empty() && is_xmm_reg(operands[0]) {
                    let val = self.get(operands[0]);
                    self.set(operands[0], SymValue::Neg(Box::new(val)));
                }
                Ok(())
            }

            // --- Comparison (flag tracking) ---
            "ucomiss" | "vucomiss" | "comiss" | "vcomiss" => {
                if operands.len() >= 2 {
                    let lhs = self.resolve(operands[0]);
                    let rhs = self.resolve(operands[1]);
                    self.flags = Some(CmpFlags { lhs, rhs });
                }
                Ok(())
            }

            // --- Conditional move (integer regs, but track for patterns) ---
            "cmova" | "cmovae" | "cmovb" | "cmovbe"
            | "cmovg" | "cmovge" | "cmovl" | "cmovle"
            | "cmove" | "cmovne" => {
                // cmov operates on integer regs; skip for float symbolic exec.
                Ok(())
            }

            // --- Integer conversion ---
            "cvtss2si" | "vcvtss2si" => {
                // float → int: result goes to integer reg, not tracked in float domain.
                Ok(())
            }
            "cvtsi2ss" | "vcvtsi2ss" => {
                // int → float: result is unknown (we don't track integer regs symbolically).
                if !operands.is_empty() && is_xmm_reg(operands[0]) {
                    self.set(operands[0], SymValue::Unknown("cvtsi2ss".into()));
                }
                Ok(())
            }
            "cvttss2si" | "vcvttss2si" => Ok(()),

            // --- libm calls ---
            "call" | "bl" => self.call_op(operands),

            // ===================================================================
            // AArch64 floating-point instructions
            // ===================================================================

            // --- AArch64 scalar float arithmetic (Sd = Sn op Sm) ---
            "fadd" => self.bin_op(operands, |a, b| SymValue::Add(Box::new(a), Box::new(b))),
            "fsub" => self.bin_op(operands, |a, b| SymValue::Sub(Box::new(a), Box::new(b))),
            "fmul" => self.bin_op(operands, |a, b| SymValue::Mul(Box::new(a), Box::new(b))),
            "fdiv" => self.bin_op(operands, |a, b| SymValue::Div(Box::new(a), Box::new(b))),

            // --- AArch64 FMA: fmadd Sd, Sn, Sm, Sa → Sd = Sn * Sm + Sa ---
            "fmadd" => {
                if operands.len() == 4 {
                    let sn = self.resolve(operands[1]);
                    let sm = self.resolve(operands[2]);
                    let sa = self.resolve(operands[3]);
                    self.set(operands[0], SymValue::Fma(Box::new(sn), Box::new(sm), Box::new(sa)));
                    Ok(())
                } else {
                    Err(SymExecError::UnsupportedInstruction("fmadd expects 4 operands".into()))
                }
            }
            // fmsub Sd, Sn, Sm, Sa → Sd = Sn * Sm - Sa
            "fmsub" => {
                if operands.len() == 4 {
                    let sn = self.resolve(operands[1]);
                    let sm = self.resolve(operands[2]);
                    let sa = self.resolve(operands[3]);
                    self.set(operands[0], SymValue::Sub(
                        Box::new(SymValue::Mul(Box::new(sn), Box::new(sm))),
                        Box::new(sa),
                    ));
                    Ok(())
                } else {
                    Err(SymExecError::UnsupportedInstruction("fmsub expects 4 operands".into()))
                }
            }
            // fnmadd Sd, Sn, Sm, Sa → Sd = -(Sn * Sm) + Sa = Sa - Sn*Sm
            "fnmadd" => {
                if operands.len() == 4 {
                    let sn = self.resolve(operands[1]);
                    let sm = self.resolve(operands[2]);
                    let sa = self.resolve(operands[3]);
                    self.set(operands[0], SymValue::Sub(
                        Box::new(sa),
                        Box::new(SymValue::Mul(Box::new(sn), Box::new(sm))),
                    ));
                    Ok(())
                } else {
                    Err(SymExecError::UnsupportedInstruction("fnmadd expects 4 operands".into()))
                }
            }
            // fnmsub Sd, Sn, Sm, Sa → Sd = -(Sn * Sm) - Sa
            "fnmsub" => {
                if operands.len() == 4 {
                    let sn = self.resolve(operands[1]);
                    let sm = self.resolve(operands[2]);
                    let sa = self.resolve(operands[3]);
                    self.set(operands[0], SymValue::Neg(Box::new(SymValue::Fma(
                        Box::new(sn), Box::new(sm), Box::new(sa),
                    ))));
                    Ok(())
                } else {
                    Err(SymExecError::UnsupportedInstruction("fnmsub expects 4 operands".into()))
                }
            }

            // --- AArch64 max/min ---
            "fmax" => self.bin_op(operands, |a, b| SymValue::Max(Box::new(a), Box::new(b))),
            "fmin" => self.bin_op(operands, |a, b| SymValue::Min(Box::new(a), Box::new(b))),
            "fmaxnm" => self.bin_op(operands, |a, b| SymValue::Max(Box::new(a), Box::new(b))),
            "fminnm" => self.bin_op(operands, |a, b| SymValue::Min(Box::new(a), Box::new(b))),

            // --- AArch64 unary ---
            "fsqrt" => {
                if operands.len() >= 2 {
                    let src = self.resolve(operands[1]);
                    self.set(operands[0], SymValue::Sqrt(Box::new(src)));
                }
                Ok(())
            }
            "fneg" => {
                if operands.len() >= 2 {
                    let src = self.resolve(operands[1]);
                    self.set(operands[0], SymValue::Neg(Box::new(src)));
                }
                Ok(())
            }
            "fabs" => {
                if operands.len() >= 2 {
                    let src = self.resolve(operands[1]);
                    self.set(operands[0], SymValue::Abs(Box::new(src)));
                }
                Ok(())
            }

            // --- AArch64 moves ---
            "fmov" => self.mov_op(operands),
            // ldr/str with float regs: ldr s0, [x0, #offset]
            "ldr" | "ldp" => self.aarch64_load_op(operands),
            "str" | "stp" => self.aarch64_store_op(operands),

            // --- AArch64 comparison (flag tracking) ---
            "fcmp" | "fcmpe" => {
                if operands.len() >= 2 {
                    let lhs = self.resolve(operands[0]);
                    let rhs = if operands[1] == "#0.0" || operands[1] == "0.0" {
                        SymValue::Const(0.0)
                    } else {
                        self.resolve(operands[1])
                    };
                    self.flags = Some(CmpFlags { lhs, rhs });
                }
                Ok(())
            }

            // --- AArch64 conditional select ---
            // fcsel Sd, Sn, Sm, cond → Sd = cond ? Sn : Sm
            "fcsel" => {
                if operands.len() >= 4 {
                    let true_val = self.resolve(operands[1]);
                    let false_val = self.resolve(operands[2]);
                    let cond = operands[3];
                    if let Some(flags) = &self.flags {
                        let kind = match cond {
                            "gt" | "hi" => SelectKind::Gt,
                            "ge" | "hs" | "cs" => SelectKind::Ge,
                            "lt" | "lo" | "cc" => SelectKind::Lt,
                            "le" | "ls" => SelectKind::Le,
                            "eq" => SelectKind::Eq,
                            "ne" => SelectKind::Ne,
                            _ => SelectKind::Ne,
                        };
                        self.set(operands[0], SymValue::Select {
                            kind,
                            cond_lhs: Box::new(flags.lhs.clone()),
                            cond_rhs: Box::new(flags.rhs.clone()),
                            true_val: Box::new(true_val),
                            false_val: Box::new(false_val),
                        });
                    } else {
                        self.set(operands[0], SymValue::Unknown("fcsel:no_flags".into()));
                    }
                }
                Ok(())
            }

            // --- AArch64 conversions ---
            "scvtf" | "ucvtf" => {
                // int → float: result is unknown
                if !operands.is_empty() && is_aarch64_float_reg(operands[0]) {
                    self.set(operands[0], SymValue::Unknown("scvtf".into()));
                }
                Ok(())
            }
            "fcvtzs" | "fcvtzu" | "fcvtas" | "fcvtau" => {
                // float → int: not tracked in float domain
                Ok(())
            }
            // float precision conversion: fcvt Dd, Sn or fcvt Sd, Dn
            "fcvt" => self.mov_op(operands),

            // --- No-ops for symbolic execution (x86_64 + AArch64 combined) ---
            // x86_64: integer ops, branches, flags, stack management
            "ret" | "nop" | "push" | "pop" | "endbr64" | "lea"
            | "mov" | "movz" | "movk" | "movn"
            | "add" | "sub" | "mul" | "sdiv" | "udiv"
            | "and" | "orr" | "eor" | "or" | "xor"
            | "lsl" | "lsr" | "asr" | "shr" | "shl" | "sar" | "sal"
            | "cmp" | "cmn" | "tst" | "test"
            | "je" | "jne" | "jmp" | "jb" | "ja" | "jbe" | "jae"
            | "seta" | "setb" | "setae" | "setbe" | "sete" | "setne"
            | "setg" | "setge" | "setl" | "setle"
            | "cdq" | "cdqe" | "cqo"
            // AArch64: branches, conditional ops, address gen
            | "b" | "b.eq" | "b.ne" | "b.gt" | "b.ge" | "b.lt" | "b.le"
            | "b.hi" | "b.hs" | "b.lo" | "b.ls" | "b.mi" | "b.pl"
            | "cbz" | "cbnz" | "tbz" | "tbnz"
            | "adrp" | "sxtw" | "uxtw"
            | "stp_int" | "ldp_int" | "str_int" | "ldr_int"
            | "csel" | "csinc" | "csinv" | "csneg" => {
                Ok(())
            }

            _ => {
                // Unknown instruction — don't fail, just skip.
                Ok(())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Instruction handler helpers
    // -----------------------------------------------------------------------

    fn bin_op(
        &mut self,
        operands: &[&str],
        f: impl FnOnce(SymValue, SymValue) -> SymValue,
    ) -> Result<(), SymExecError> {
        match operands.len() {
            2 => {
                let dst_val = self.resolve(operands[0]);
                let src_val = self.resolve(operands[1]);
                self.set(operands[0], f(dst_val, src_val));
                Ok(())
            }
            3 => {
                let src1 = self.resolve(operands[1]);
                let src2 = self.resolve(operands[2]);
                self.set(operands[0], f(src1, src2));
                Ok(())
            }
            _ => Err(SymExecError::UnsupportedInstruction(
                format!("unexpected operand count: {}", operands.len()),
            )),
        }
    }

    /// FMA: always 3 operands (dst, src1, src2). The closure receives
    /// (dst_value, src1_value, src2_value) and returns the result.
    fn fma_op(
        &mut self,
        operands: &[&str],
        f: impl FnOnce(SymValue, SymValue, SymValue) -> SymValue,
    ) -> Result<(), SymExecError> {
        if operands.len() != 3 {
            return Err(SymExecError::UnsupportedInstruction(
                format!("FMA expects 3 operands, got {}", operands.len()),
            ));
        }
        let dst_val = self.resolve(operands[0]);
        let src1 = self.resolve(operands[1]);
        let src2 = self.resolve(operands[2]);
        self.set(operands[0], f(dst_val, src1, src2));
        Ok(())
    }

    fn mov_op(&mut self, operands: &[&str]) -> Result<(), SymExecError> {
        if operands.len() < 2 {
            return Ok(());
        }
        let dst = operands[0];
        // For VEX 3-operand movss (vmovss xmm1, xmm2, xmm3), use last as src.
        let src = if operands.len() == 3 { operands[2] } else { operands[1] };

        // Store to stack: movss [rsp+X], xmmN
        if is_memory_operand(dst) {
            if let Some(offset) = parse_stack_offset(dst) {
                let val = self.resolve(src);
                self.stack.insert(offset, val);
                return Ok(());
            }
            // Store to other memory — ignore.
            return Ok(());
        }

        // Load from memory or register
        let val = self.resolve(src);
        self.set(dst, val);
        Ok(())
    }

    fn xor_op(&mut self, operands: &[&str]) -> Result<(), SymExecError> {
        // Zero idiom: xorps xmm0, xmm0 or vxorps xmm0, xmm1, xmm1
        let is_zero = match operands.len() {
            2 => operands[0] == operands[1],
            3 => operands[1] == operands[2],
            _ => false,
        };
        if is_zero {
            self.set(operands[0], SymValue::Const(0.0));
            return Ok(());
        }

        // Sign flip: xorps xmm0, [sign_mask] where sign_mask = 0x80000000
        let (dst, src) = match operands.len() {
            2 => (operands[0], operands[1]),
            3 => (operands[0], operands[2]),
            _ => return Ok(()),
        };

        if is_memory_operand(src) {
            if let Some(addr) = parse_rip_addr(src) {
                if let Some(&cval) = self.constants.get(&addr) {
                    if is_sign_mask_f32(cval) {
                        let val = self.resolve(if operands.len() == 3 { operands[1] } else { dst });
                        self.set(dst, SymValue::Neg(Box::new(val)));
                        return Ok(());
                    }
                }
            }
        }
        // Check register source for sign mask
        let src_val = self.resolve(src);
        if let SymValue::Const(v) = &src_val {
            if is_sign_mask_f32(*v as f32) {
                let base = self.resolve(if operands.len() == 3 { operands[1] } else { dst });
                self.set(dst, SymValue::Neg(Box::new(base)));
                return Ok(());
            }
        }

        // Generic xor — not a recognized pattern.
        Ok(())
    }

    fn bitwise_and_op(&mut self, operands: &[&str]) -> Result<(), SymExecError> {
        let (dst, src1_name, src2_name) = match operands.len() {
            2 => (operands[0], operands[0], operands[1]),
            3 => (operands[0], operands[1], operands[2]),
            _ => return Ok(()),
        };

        let src1 = self.resolve(src1_name);
        let src2 = self.resolve(src2_name);

        // Check if either operand is the ABS mask → Abs(other)
        if is_const_abs_mask(&src2) {
            self.set(dst, SymValue::Abs(Box::new(src1)));
            return Ok(());
        }
        if is_const_abs_mask(&src1) {
            self.set(dst, SymValue::Abs(Box::new(src2)));
            return Ok(());
        }

        // Unknown bitwise and
        self.set(dst, SymValue::Unknown(format!("and({src1}, {src2})")));
        Ok(())
    }

    fn bitwise_generic_op(&mut self, operands: &[&str], name: &str) -> Result<(), SymExecError> {
        if !operands.is_empty() && is_xmm_reg(operands[0]) {
            let desc = operands.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(", ");
            self.set(operands[0], SymValue::Unknown(format!("{name}({desc})")));
        }
        Ok(())
    }

    fn call_op(&mut self, operands: &[&str]) -> Result<(), SymExecError> {
        let target = operands.first().copied().unwrap_or("");
        let func = if target.contains("expf") {
            Some(LibmFn::Expf)
        } else if target.contains("sqrtf") {
            Some(LibmFn::Sqrtf)
        } else if target.contains("tanhf") {
            Some(LibmFn::Tanhf)
        } else if target.contains("logf") {
            Some(LibmFn::Logf)
        } else if target.contains("fabsf") {
            Some(LibmFn::Fabsf)
        } else {
            None
        };

        if let Some(f) = func {
            // x86_64: xmm0, AArch64: s0
            let (arg_reg, ret_reg) = self.float_call_regs();
            let arg = self.get(arg_reg);
            self.set(ret_reg, SymValue::Call(f, vec![arg]));
        } else {
            let (_, ret_reg) = self.float_call_regs();
            self.set(ret_reg, SymValue::Unknown(format!("call:{target}")));
        }
        Ok(())
    }

    /// Return (arg_reg, ret_reg) for float function calls per platform ABI.
    fn float_call_regs(&self) -> (&'static str, &'static str) {
        // Check if we have AArch64 float regs in state
        if self.regs.contains_key("s0") || self.regs.contains_key("d0") || self.regs.contains_key("v0") {
            ("s0", "s0")
        } else {
            ("xmm0", "xmm0")
        }
    }

    // -----------------------------------------------------------------------
    // AArch64 load/store helpers
    // -----------------------------------------------------------------------

    /// AArch64 load: `ldr s0, [x0, #16]` or `ldr s0, [sp, #-8]`
    fn aarch64_load_op(&mut self, operands: &[&str]) -> Result<(), SymExecError> {
        if operands.len() < 2 {
            return Ok(());
        }
        let dst = operands[0];
        // Only track float register loads
        if !is_aarch64_float_reg(dst) {
            return Ok(());
        }
        let mem = operands[1];
        if is_memory_operand(mem) {
            if let Some((base, offset)) = parse_aarch64_mem(mem) {
                // Stack spill load
                if base == "sp" || base == "x29" {
                    if let Some(val) = self.stack.get(&offset) {
                        self.set(dst, val.clone());
                        return Ok(());
                    }
                }
            }
            self.set(dst, SymValue::Unknown(format!("load:{mem}")));
        } else {
            let val = self.resolve(mem);
            self.set(dst, val);
        }
        Ok(())
    }

    /// AArch64 store: `str s0, [sp, #-8]`
    fn aarch64_store_op(&mut self, operands: &[&str]) -> Result<(), SymExecError> {
        if operands.len() < 2 {
            return Ok(());
        }
        let src = operands[0];
        let mem = operands[1];
        // Only track float register stores to stack
        if !is_aarch64_float_reg(src) {
            return Ok(());
        }
        if is_memory_operand(mem) {
            if let Some((base, offset)) = parse_aarch64_mem(mem) {
                if base == "sp" || base == "x29" {
                    let val = self.resolve(src);
                    self.stack.insert(offset, val);
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // State snapshot / restore (for loop analysis)
    // -----------------------------------------------------------------------

    /// Take a snapshot of the current executor state.
    pub fn snapshot(&self) -> SymbolicExecutor {
        self.clone()
    }

    /// Restore from a previously taken snapshot.
    pub fn restore(&mut self, snap: &SymbolicExecutor) {
        self.regs = snap.regs.clone();
        self.stack = snap.stack.clone();
        self.constants = snap.constants.clone();
        self.flags = snap.flags.clone();
    }

    /// Return the current XMM register state (xmm0..xmm15).
    pub fn xmm_state(&self) -> HashMap<String, SymValue> {
        self.regs
            .iter()
            .filter(|(k, _)| k.starts_with("xmm"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Return the current comparison flags (lhs, rhs) if set.
    pub fn get_flags(&self) -> Option<(SymValue, SymValue)> {
        self.flags.as_ref().map(|f| (f.lhs.clone(), f.rhs.clone()))
    }

    /// Return the current stack spill state.
    pub fn stack_state(&self) -> &HashMap<i64, SymValue> {
        &self.stack
    }

    /// Set a stack spill slot to a symbolic value.
    pub fn set_stack(&mut self, offset: i64, val: SymValue) {
        self.stack.insert(offset, val);
    }

    // -----------------------------------------------------------------------
    // OpTrace extraction
    // -----------------------------------------------------------------------

    /// Extract a linear `Vec<TraceOp>` from the return value's symbolic expression tree.
    ///
    /// Simplifies the expression first, then linearizes into SSA form with
    /// deduplication of identical sub-expressions.
    pub fn extract_trace(&self) -> Result<Vec<TraceOp>, SymExecError> {
        let ret = self.return_value()?;
        let simplified = ret.simplify();
        let mut ops = Vec::new();
        let mut cache = HashMap::new();
        linearize(&simplified, &mut ops, &mut cache);
        Ok(ops)
    }
}

/// Check if a SymValue is a constant with ABS mask bits (0x7FFFFFFF).
fn is_const_abs_mask(v: &SymValue) -> bool {
    if let SymValue::Const(c) = v {
        is_abs_mask_f32(*c as f32)
    } else {
        false
    }
}

/// Recursively linearize a SymValue tree into SSA `Vec<TraceOp>`.
///
/// Returns the index of the emitted op. Uses `cache` (keyed by Display string)
/// to deduplicate identical sub-expressions.
fn linearize(
    val: &SymValue,
    ops: &mut Vec<TraceOp>,
    cache: &mut HashMap<String, ValueId>,
) -> ValueId {
    let key = format!("{val}");
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let idx = match val {
        SymValue::Param(n) => {
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Input(*n as u32));
            i
        }
        SymValue::Const(v) => {
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Const(*v));
            i
        }
        SymValue::Add(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Add(ai, bi));
            i
        }
        SymValue::Sub(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Sub(ai, bi));
            i
        }
        SymValue::Mul(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Mul(ai, bi));
            i
        }
        SymValue::Div(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Div(ai, bi));
            i
        }
        SymValue::Fma(a, b, c) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let ci = linearize(c, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Fma(ai, bi, ci));
            i
        }
        SymValue::Neg(a) => {
            let ai = linearize(a, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Neg(ai));
            i
        }
        SymValue::Abs(a) => {
            let ai = linearize(a, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Abs(ai));
            i
        }
        SymValue::Max(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Max(ai, bi));
            i
        }
        SymValue::Min(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Min(ai, bi));
            i
        }
        SymValue::Sqrt(a) => {
            let ai = linearize(a, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Sqrt(ai));
            i
        }
        SymValue::Recip(a) => {
            let ai = linearize(a, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Recip(ai));
            i
        }
        SymValue::Rsqrt(a) => {
            let ai = linearize(a, ops, cache);
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Rsqrt(ai));
            i
        }
        SymValue::Call(func, args) => {
            match func {
                LibmFn::Expf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ValueId(ops.len() as u32);
                    ops.push(TraceOp::Exp(ai));
                    i
                }
                LibmFn::Tanhf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ValueId(ops.len() as u32);
                    ops.push(TraceOp::Tanh(ai));
                    i
                }
                LibmFn::Sqrtf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ValueId(ops.len() as u32);
                    ops.push(TraceOp::Sqrt(ai));
                    i
                }
                LibmFn::Fabsf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ValueId(ops.len() as u32);
                    ops.push(TraceOp::Abs(ai));
                    i
                }
                LibmFn::Logf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ValueId(ops.len() as u32);
                    ops.push(TraceOp::Log(ai));
                    i
                }
            }
        }
        SymValue::Select { kind, true_val, false_val, .. } => {
            // Select that survived simplification — linearize as Max/Min
            // based on the comparison kind, or fall back to true_val.
            let ti = linearize(true_val, ops, cache);
            let fi = linearize(false_val, ops, cache);
            let i = ValueId(ops.len() as u32);
            match kind {
                SelectKind::Gt | SelectKind::Ge => {
                    ops.push(TraceOp::Max(ti, fi));
                }
                SelectKind::Lt | SelectKind::Le => {
                    ops.push(TraceOp::Min(ti, fi));
                }
                _ => {
                    // For Eq/Ne selects, just use the true branch as best effort.
                    return ti;
                }
            }
            i
        }
        SymValue::Load { .. } | SymValue::Unknown(_) => {
            // Unresolvable — emit as Input(0) placeholder.
            let i = ValueId(ops.len() as u32);
            ops.push(TraceOp::Input(0));
            i
        }
    };

    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::trace::TraceOp;

    // -----------------------------------------------------------------------
    // Basic instruction tests (existing + enhanced)
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_add() {
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.step("addss", &["xmm0", "xmm1"]).unwrap();
        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        assert!(s.contains("param(0)"));
        assert!(s.contains("param(1)"));
        assert!(s.contains("+"));
    }

    #[test]
    fn test_vex_mul() {
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.step("vmulss", &["xmm2", "xmm0", "xmm1"]).unwrap();
        exec.step("vmovss", &["xmm0", "xmm2"]).unwrap();
        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        assert!(s.contains("*"));
    }

    #[test]
    fn test_call_expf() {
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("call", &["expf@PLT"]).unwrap();
        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        assert!(s.contains("expf"));
        assert!(s.contains("param(0)"));
    }

    #[test]
    fn test_zero_via_xorps() {
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("xorps", &["xmm0", "xmm0"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Const(v) if v == 0.0));
    }

    #[test]
    fn test_zero_via_vxorps_3op() {
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.step("vxorps", &["xmm0", "xmm1", "xmm1"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Const(v) if v == 0.0));
    }

    #[test]
    fn test_vxorps_3op_not_zero() {
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.step("vxorps", &["xmm0", "xmm0", "xmm1"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(!matches!(ret, SymValue::Const(v) if v == 0.0));
    }

    // -----------------------------------------------------------------------
    // FMA instructions
    // -----------------------------------------------------------------------

    #[test]
    fn test_fma231() {
        // vfmadd231ss xmm0, xmm1, xmm2 → xmm0 = xmm1 * xmm2 + xmm0
        let mut exec = SymbolicExecutor::new(3, 0);
        exec.step("vfmadd231ss", &["xmm0", "xmm1", "xmm2"]).unwrap();
        let ret = exec.return_value().unwrap();
        // Should be Fma(param(1), param(2), param(0))
        assert!(matches!(ret, SymValue::Fma(_, _, _)));
        let s = format!("{ret}");
        assert!(s.contains("param(1)"));
        assert!(s.contains("param(2)"));
        assert!(s.contains("param(0)"));
    }

    #[test]
    fn test_fma213() {
        // vfmadd213ss xmm0, xmm1, xmm2 → xmm0 = xmm1 * xmm0 + xmm2
        let mut exec = SymbolicExecutor::new(3, 0);
        exec.step("vfmadd213ss", &["xmm0", "xmm1", "xmm2"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Fma(_, _, _)));
    }

    #[test]
    fn test_fma132() {
        // vfmadd132ss xmm0, xmm1, xmm2 → xmm0 = xmm0 * xmm2 + xmm1
        let mut exec = SymbolicExecutor::new(3, 0);
        exec.step("vfmadd132ss", &["xmm0", "xmm1", "xmm2"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Fma(_, _, _)));
    }

    // -----------------------------------------------------------------------
    // Stack spill / reload
    // -----------------------------------------------------------------------

    #[test]
    fn test_stack_spill_reload() {
        let mut exec = SymbolicExecutor::new(1, 0);
        // Spill xmm0 to [rsp+0x10]
        exec.step("movss", &["[rsp+0x10]", "xmm0"]).unwrap();
        // Clobber xmm0
        exec.step("xorps", &["xmm0", "xmm0"]).unwrap();
        assert!(matches!(exec.return_value().unwrap(), SymValue::Const(v) if v == 0.0));
        // Reload from [rsp+0x10]
        exec.step("movss", &["xmm0", "[rsp+0x10]"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Param(0)));
    }

    #[test]
    fn test_stack_negative_offset() {
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("movss", &["[rsp-8]", "xmm0"]).unwrap();
        exec.step("xorps", &["xmm0", "xmm0"]).unwrap();
        exec.step("movss", &["xmm0", "[rsp-8]"]).unwrap();
        assert!(matches!(exec.return_value().unwrap(), SymValue::Param(0)));
    }

    // -----------------------------------------------------------------------
    // Constant pool
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_load() {
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.register_constant(0x1234, 1.0);
        exec.step("movss", &["xmm1", "[rip+0x1234]"]).unwrap();
        exec.step("addss", &["xmm0", "xmm1"]).unwrap();
        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        assert!(s.contains("param(0)"));
        assert!(s.contains("1.0")); // constant folded or present
    }

    #[test]
    fn test_constant_pool_multiple() {
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.register_constant(0xA000, 2.5);
        exec.register_constant(0xB000, 3.0);
        exec.step("movss", &["xmm1", "[rip+0xa000]"]).unwrap();
        exec.step("movss", &["xmm2", "[rip+0xb000]"]).unwrap();
        exec.step("mulss", &["xmm0", "xmm1"]).unwrap();
        exec.step("addss", &["xmm0", "xmm2"]).unwrap();
        // result = param(0) * 2.5 + 3.0
        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        assert!(s.contains("param(0)"));
        assert!(s.contains("2.5"));
        assert!(s.contains("3.0"));
    }

    // -----------------------------------------------------------------------
    // Bitwise abs pattern
    // -----------------------------------------------------------------------

    #[test]
    fn test_andps_abs() {
        let mut exec = SymbolicExecutor::new(1, 0);
        let abs_mask = f32::from_bits(0x7FFF_FFFF);
        exec.register_constant(0x5000, abs_mask);
        exec.step("movss", &["xmm1", "[rip+0x5000]"]).unwrap();
        exec.step("andps", &["xmm0", "xmm1"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Abs(_)));
    }

    // -----------------------------------------------------------------------
    // XOR sign flip (negation)
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorps_sign_flip() {
        let mut exec = SymbolicExecutor::new(1, 0);
        let sign_mask = f32::from_bits(0x8000_0000);
        exec.register_constant(0x6000, sign_mask);
        exec.step("movss", &["xmm1", "[rip+0x6000]"]).unwrap();
        exec.step("xorps", &["xmm0", "xmm1"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Neg(_)));
    }

    // -----------------------------------------------------------------------
    // ReLU: max(0, x)
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_maxss() {
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("xorps", &["xmm1", "xmm1"]).unwrap();
        exec.step("maxss", &["xmm0", "xmm1"]).unwrap();
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Max(_, _)));
        let s = format!("{ret}");
        assert!(s.contains("param(0)"));
        assert!(s.contains("0")); // Const(0.0)
    }

    // -----------------------------------------------------------------------
    // SiLU: x / (1 + exp(-x))
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_symbolic() {
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.register_constant(0x100, 1.0);
        let sign_mask = f32::from_bits(0x8000_0000);
        exec.register_constant(0x200, sign_mask);

        // xmm0 = param(0) = x
        // Negate: xmm1 = -x
        exec.step("movss", &["xmm1", "xmm0"]).unwrap();
        exec.step("movss", &["xmm2", "[rip+0x200]"]).unwrap();
        exec.step("xorps", &["xmm1", "xmm2"]).unwrap();
        // exp(-x)
        exec.step("movss", &["xmm0", "xmm1"]).unwrap();
        exec.step("call", &["expf@PLT"]).unwrap();
        // xmm0 = exp(-x), add 1.0
        exec.step("movss", &["xmm1", "[rip+0x100]"]).unwrap();
        exec.step("addss", &["xmm0", "xmm1"]).unwrap();
        // xmm0 = 1 + exp(-x), now divide x / (1+exp(-x))
        // Save denominator
        exec.step("movss", &["[rsp+0x10]", "xmm0"]).unwrap();
        // Reload x (was spilled or we use param trick)
        // For this test, let's set xmm1 = param(0) directly
        exec.set("xmm1", SymValue::Param(0));
        exec.step("movss", &["xmm0", "xmm1"]).unwrap();
        exec.step("divss", &["xmm0", "[rsp+0x10]"]).unwrap();

        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        // Should contain division, exp, param(0)
        assert!(s.contains("param(0)"));
        assert!(s.contains("expf"));
        assert!(s.contains("/"));
    }

    // -----------------------------------------------------------------------
    // OpTrace extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_trace_add_one() {
        // f(x) = x + 1
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.register_constant(0x100, 1.0);
        exec.step("movss", &["xmm1", "[rip+0x100]"]).unwrap();
        exec.step("addss", &["xmm0", "xmm1"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        // Expected: [Input(0), Const(1.0), Add(0, 1)]
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0], TraceOp::Input(0));
        assert_eq!(trace[1], TraceOp::Const(1.0));
        assert_eq!(trace[2], TraceOp::Add(ValueId(0), ValueId(1)));
    }

    #[test]
    fn test_extract_trace_relu() {
        // f(x) = max(0, x)
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("xorps", &["xmm1", "xmm1"]).unwrap();
        exec.step("maxss", &["xmm0", "xmm1"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        // Expected: [Input(0), Const(0.0), Max(0, 1)]
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0], TraceOp::Input(0));
        assert_eq!(trace[1], TraceOp::Const(0.0));
        assert_eq!(trace[2], TraceOp::Max(ValueId(0), ValueId(1)));
    }

    #[test]
    fn test_extract_trace_silu() {
        // f(x) = x / (1 + exp(-x))
        // Build the SymValue directly for a clean test.
        let mut exec = SymbolicExecutor::new(1, 0);
        let x = SymValue::Param(0);
        let neg_x = SymValue::Neg(Box::new(x.clone()));
        let exp_neg_x = SymValue::Call(LibmFn::Expf, vec![neg_x]);
        let one_plus = SymValue::Add(
            Box::new(exp_neg_x),
            Box::new(SymValue::Const(1.0)),
        );
        let silu = SymValue::Div(Box::new(x), Box::new(one_plus));
        exec.set("xmm0", silu);

        let trace = exec.extract_trace().unwrap();
        // Verify SSA validity
        for (i, op) in trace.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::Neg(a) | TraceOp::Exp(a) | TraceOp::Abs(a)
                | TraceOp::Sqrt(a) | TraceOp::Rsqrt(a) | TraceOp::Tanh(a)
                | TraceOp::Recip(a) | TraceOp::Log(a) => {
                    assert!((a.0 as usize) < i, "SSA violation at {i}");
                }
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b)
                | TraceOp::Div(a, b) | TraceOp::Max(a, b) | TraceOp::Min(a, b) => {
                    assert!((a.0 as usize) < i, "SSA violation at {i}");
                    assert!((b.0 as usize) < i, "SSA violation at {i}");
                }
                TraceOp::Fma(a, b, c) => {
                    assert!((a.0 as usize) < i, "SSA violation at {i}");
                    assert!((b.0 as usize) < i, "SSA violation at {i}");
                    assert!((c.0 as usize) < i, "SSA violation at {i}");
                }
                TraceOp::ConditionalBranch(mask, t_val, f_val) => {
                    assert!((mask.0 as usize) < i, "SSA violation at {i}");
                    assert!((t_val.0 as usize) < i, "SSA violation at {i}");
                    assert!((f_val.0 as usize) < i, "SSA violation at {i}");
                }
                // Extended §12+§14 variants: SSA validation for these is handled
                // per-variant in their own tests; skip here to keep this test focused.
                _ => {}
            }
        }

        // Should contain: Input(0), Neg, Exp, Const(1.0), Add, Div
        let has_input = trace.iter().any(|op| matches!(op, TraceOp::Input(0)));
        let has_neg = trace.iter().any(|op| matches!(op, TraceOp::Neg(_)));
        let has_exp = trace.iter().any(|op| matches!(op, TraceOp::Exp(_)));
        let has_div = trace.iter().any(|op| matches!(op, TraceOp::Div(_, _)));
        assert!(has_input, "missing Input(0)");
        assert!(has_neg, "missing Neg");
        assert!(has_exp, "missing Exp");
        assert!(has_div, "missing Div");
    }

    #[test]
    fn test_extract_trace_fma() {
        // f(x, y, z) = x * y + z via FMA
        let mut exec = SymbolicExecutor::new(3, 0);
        exec.step("vfmadd231ss", &["xmm0", "xmm1", "xmm2"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        // Should contain Fma
        let has_fma = trace.iter().any(|op| matches!(op, TraceOp::Fma(_, _, _)));
        assert!(has_fma, "missing Fma op");
    }

    #[test]
    fn test_extract_trace_dedup() {
        // f(x) = x + x — Input(0) should appear only once.
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("addss", &["xmm0", "xmm0"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        let input_count = trace.iter().filter(|op| matches!(op, TraceOp::Input(0))).count();
        assert_eq!(input_count, 1, "Input(0) should be deduplicated");
        // Expected: [Input(0), Add(0, 0)]
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[1], TraceOp::Add(ValueId(0), ValueId(0)));
    }

    #[test]
    fn test_extract_trace_simplification() {
        // f(x) = (x + 0) * 1 should simplify to just x → [Input(0)]
        let mut exec = SymbolicExecutor::new(1, 0);
        let v = SymValue::Mul(
            Box::new(SymValue::Add(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Const(0.0)),
            )),
            Box::new(SymValue::Const(1.0)),
        );
        exec.set("xmm0", v);

        let trace = exec.extract_trace().unwrap();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0], TraceOp::Input(0));
    }

    #[test]
    fn test_ucomiss_sets_flags() {
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.step("ucomiss", &["xmm0", "xmm1"]).unwrap();
        assert!(exec.flags.is_some());
    }

    #[test]
    fn test_noop_instructions_dont_fail() {
        let mut exec = SymbolicExecutor::new(1, 0);
        for instr in &["ret", "push", "pop", "endbr64", "nop", "lea", "mov",
                       "sub", "add", "cmp", "test", "je", "jne", "jmp",
                       "cvtss2si", "cvtsi2ss", "cmova", "seta"] {
            exec.step(instr, &["rax", "rbx"]).unwrap();
        }
        // xmm0 should still be param(0)
        assert!(matches!(exec.return_value().unwrap(), SymValue::Param(0)));
    }

    // -----------------------------------------------------------------------
    // WI-21: Additional symexec engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_symexec_log_extraction() {
        // f(x) = log(x) via call to logf
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("call", &["logf@PLT"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        // Expected: [Input(0), Log(0)]
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0], TraceOp::Input(0));
        assert_eq!(trace[1], TraceOp::Log(ValueId(0)));
    }

    #[test]
    fn test_symexec_constant_folding() {
        // f(x) = x + 0.0 should simplify to just x via extract_trace
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.register_constant(0x100, 0.0);
        exec.step("movss", &["xmm1", "[rip+0x100]"]).unwrap();
        exec.step("addss", &["xmm0", "xmm1"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        // After simplification: x + 0.0 -> x -> [Input(0)]
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0], TraceOp::Input(0));
    }

    #[test]
    fn test_symexec_identity_mul() {
        // f(x) = x * 1.0 should simplify to just x
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.register_constant(0x100, 1.0);
        exec.step("movss", &["xmm1", "[rip+0x100]"]).unwrap();
        exec.step("mulss", &["xmm0", "xmm1"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        // After simplification: x * 1.0 -> x -> [Input(0)]
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0], TraceOp::Input(0));
    }

    #[test]
    fn test_symexec_zero_sub() {
        // f(x) = x - x should simplify to Const(0.0)
        let mut exec = SymbolicExecutor::new(1, 0);
        exec.step("subss", &["xmm0", "xmm0"]).unwrap();

        let trace = exec.extract_trace().unwrap();
        // After simplification: x - x -> 0.0 -> [Const(0.0)]
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0], TraceOp::Const(0.0));
    }

    #[test]
    fn test_symexec_stack_spill_reload() {
        // Compute x + 1, spill to stack, clobber registers, reload, verify preserved
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.register_constant(0x100, 1.0);

        // Compute param(0) + 1.0 into xmm3
        exec.step("movss", &["xmm2", "[rip+0x100]"]).unwrap();
        exec.step("vaddss", &["xmm3", "xmm0", "xmm2"]).unwrap();
        // Spill computed value to stack
        exec.step("movss", &["[rsp+0x20]", "xmm3"]).unwrap();
        // Clobber xmm3 with unrelated work
        exec.step("vmulss", &["xmm3", "xmm1", "xmm1"]).unwrap();
        // Clobber xmm0 too
        exec.step("xorps", &["xmm0", "xmm0"]).unwrap();
        // Reload the spilled value into xmm0
        exec.step("movss", &["xmm0", "[rsp+0x20]"]).unwrap();

        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        // Should be param(0) + 1.0, not 0.0 or param(1)*param(1)
        assert!(s.contains("param(0)"), "spilled value should reference param(0)");
        assert!(s.contains("1.0"), "spilled value should reference constant 1.0");
    }

    #[test]
    fn test_symexec_fma_variants() {
        // Test all 3 FMA operand orderings and verify correct operand placement.

        // vfmadd132ss xmm0, xmm1, xmm2 -> dst = dst * src2 + src1
        let mut exec132 = SymbolicExecutor::new(3, 0);
        exec132.step("vfmadd132ss", &["xmm0", "xmm1", "xmm2"]).unwrap();
        let ret132 = exec132.return_value().unwrap();
        if let SymValue::Fma(a, b, c) = &ret132 {
            assert!(matches!(**a, SymValue::Param(0)), "132: a should be dst=param(0)");
            assert!(matches!(**b, SymValue::Param(2)), "132: b should be src2=param(2)");
            assert!(matches!(**c, SymValue::Param(1)), "132: c should be src1=param(1)");
        } else {
            panic!("vfmadd132ss should produce Fma, got: {ret132:?}");
        }

        // vfmadd213ss xmm0, xmm1, xmm2 -> dst = src1 * dst + src2
        let mut exec213 = SymbolicExecutor::new(3, 0);
        exec213.step("vfmadd213ss", &["xmm0", "xmm1", "xmm2"]).unwrap();
        let ret213 = exec213.return_value().unwrap();
        if let SymValue::Fma(a, b, c) = &ret213 {
            assert!(matches!(**a, SymValue::Param(1)), "213: a should be src1=param(1)");
            assert!(matches!(**b, SymValue::Param(0)), "213: b should be dst=param(0)");
            assert!(matches!(**c, SymValue::Param(2)), "213: c should be src2=param(2)");
        } else {
            panic!("vfmadd213ss should produce Fma, got: {ret213:?}");
        }

        // vfmadd231ss xmm0, xmm1, xmm2 -> dst = src1 * src2 + dst
        let mut exec231 = SymbolicExecutor::new(3, 0);
        exec231.step("vfmadd231ss", &["xmm0", "xmm1", "xmm2"]).unwrap();
        let ret231 = exec231.return_value().unwrap();
        if let SymValue::Fma(a, b, c) = &ret231 {
            assert!(matches!(**a, SymValue::Param(1)), "231: a should be src1=param(1)");
            assert!(matches!(**b, SymValue::Param(2)), "231: b should be src2=param(2)");
            assert!(matches!(**c, SymValue::Param(0)), "231: c should be dst=param(0)");
        } else {
            panic!("vfmadd231ss should produce Fma, got: {ret231:?}");
        }
    }

    #[test]
    fn test_linearize_dedup() {
        // Build (x+1) * (x+1) — the shared sub-expression (x+1) should be emitted once
        let mut exec = SymbolicExecutor::new(1, 0);
        let x_plus_1 = SymValue::Add(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(1.0)),
        );
        let product = SymValue::Mul(
            Box::new(x_plus_1.clone()),
            Box::new(x_plus_1),
        );
        exec.set("xmm0", product);

        let trace = exec.extract_trace().unwrap();
        // Expected: [Input(0), Const(1.0), Add(0,1), Mul(2,2)]
        // The Add should appear only once due to dedup
        let add_count = trace.iter().filter(|op| matches!(op, TraceOp::Add(_, _))).count();
        assert_eq!(add_count, 1, "shared (x+1) sub-expression should be deduplicated");
        assert_eq!(trace.len(), 4, "expected 4 ops: Input, Const, Add, Mul");
        assert_eq!(trace[3], TraceOp::Mul(ValueId(2), ValueId(2)), "Mul should reference the same Add index twice");
    }

    // -----------------------------------------------------------------------
    // WI-12k77: Additional symexec engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_symexec_error_display_messages() {
        // Arrange: create each SymExecError variant
        let err1 = SymExecError::DisassemblyFailed("bad bytes".into());
        let err2 = SymExecError::UnsupportedInstruction("fancy_op".into());
        let err3 = SymExecError::NoReturnValue;

        // Assert: Display produces expected prefix strings
        assert!(err1.to_string().starts_with("disassembly failed:"), "DisassemblyFailed display");
        assert!(err1.to_string().contains("bad bytes"));
        assert!(err2.to_string().starts_with("unsupported instruction:"), "UnsupportedInstruction display");
        assert!(err2.to_string().contains("fancy_op"));
        assert_eq!(err3.to_string(), "no return value found", "NoReturnValue display");
    }

    #[test]
    fn test_parse_stack_offset_bare_rsp() {
        // Arrange: bare [rsp] with no offset
        // Act
        let offset = parse_stack_offset("[rsp]");
        // Assert
        assert_eq!(offset, Some(0), "bare [rsp] should parse as offset 0");
    }

    #[test]
    fn test_parse_stack_offset_hex_and_decimal() {
        // Arrange: various stack offset formats
        // Act & Assert
        assert_eq!(parse_stack_offset("[rsp+0x10]"), Some(0x10));
        assert_eq!(parse_stack_offset("[rsp-0x8]"), Some(-8));
        assert_eq!(parse_stack_offset("[rsp+16]"), Some(16));
        assert_eq!(parse_stack_offset("[rsp-4]"), Some(-4));
        assert_eq!(parse_stack_offset("[rax+0x10]"), None, "non-rsp base should return None");
        assert_eq!(parse_stack_offset("rsp+0x10]"), None, "missing opening bracket");
    }

    #[test]
    fn test_parse_rip_addr_variants() {
        // Arrange: various RIP-relative and absolute address formats
        // Act & Assert
        assert_eq!(parse_rip_addr("[rip+0x1234]"), Some(0x1234));
        assert_eq!(parse_rip_addr("[0x7f000000]"), Some(0x7f000000));
        assert_eq!(parse_rip_addr("[rsp+0x10]"), None, "non-rip non-absolute should return None");
        assert_eq!(parse_rip_addr("rip+0x10]"), None, "missing opening bracket");
    }

    #[test]
    fn test_new_executor_registers_float_and_ptr_args() {
        // Arrange: create executor with 2 float args and 3 pointer args
        let exec = SymbolicExecutor::new(2, 3);

        // Act
        let xmm0 = exec.get_value("xmm0");
        let xmm1 = exec.get_value("xmm1");
        let rdi = exec.get_value("rdi");
        let rsi = exec.get_value("rsi");
        let rdx = exec.get_value("rdx");
        let xmm2 = exec.get_value("xmm2");

        // Assert: float args mapped to Param(0..1), ptr args to Param(2..4)
        assert!(matches!(xmm0, SymValue::Param(0)), "xmm0 should be Param(0)");
        assert!(matches!(xmm1, SymValue::Param(1)), "xmm1 should be Param(1)");
        assert!(matches!(rdi, SymValue::Param(2)), "rdi should be Param(2)");
        assert!(matches!(rsi, SymValue::Param(3)), "rsi should be Param(3)");
        assert!(matches!(rdx, SymValue::Param(4)), "rdx should be Param(4)");
        assert!(matches!(xmm2, SymValue::Unknown(_)), "xmm2 should be Unknown");
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        // Arrange: executor with computed state
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.step("mulss", &["xmm0", "xmm1"]).unwrap();
        let snap = exec.snapshot();

        // Act: clobber state, then restore
        exec.step("xorps", &["xmm0", "xmm0"]).unwrap();
        assert!(matches!(exec.return_value().unwrap(), SymValue::Const(0.0)));
        exec.restore(&snap);

        // Assert: restored state has the multiplication result
        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        assert!(s.contains("param(0)"), "restored state should have param(0)");
        assert!(s.contains("param(1)"), "restored state should have param(1)");
        assert!(s.contains("*"), "restored state should contain multiplication");
    }

    #[test]
    fn test_has_constant_and_register_constant() {
        // Arrange: fresh executor
        let mut exec = SymbolicExecutor::new(0, 0);

        // Act & Assert: before registration
        assert!(!exec.has_constant(0x1000), "no constant at 0x1000 yet");
        exec.register_constant(0x1000, 42.0);
        assert!(exec.has_constant(0x1000), "constant at 0x1000 registered");
        assert!(!exec.has_constant(0x2000), "no constant at 0x2000");
    }

    #[test]
    fn test_set_stack_and_stack_state() {
        // Arrange: executor with one float arg
        let mut exec = SymbolicExecutor::new(1, 0);
        assert!(exec.stack_state().is_empty(), "stack should start empty");

        // Act: manually spill a symbolic value
        let val = SymValue::Param(0);
        exec.set_stack(16, val.clone());

        // Assert: stack_state shows the spill
        assert_eq!(exec.stack_state().len(), 1, "stack should have one entry");
        assert!(matches!(exec.stack_state().get(&16), Some(v) if matches!(v, SymValue::Param(0))));
    }

    #[test]
    fn test_xmm_state_returns_only_xmm_regs() {
        // Arrange: executor with float and ptr args
        let exec = SymbolicExecutor::new(2, 2);

        // Act
        let state = exec.xmm_state();

        // Assert: only xmm* registers, no rdi/rsi
        assert!(state.contains_key("xmm0"), "xmm0 in state");
        assert!(state.contains_key("xmm1"), "xmm1 in state");
        assert!(!state.contains_key("rdi"), "rdi should not be in xmm_state");
        assert!(!state.contains_key("rsi"), "rsi should not be in xmm_state");
    }

    #[test]
    fn test_return_value_empty_executor_errors() {
        // Arrange: executor with zero args — no registers populated
        let exec = SymbolicExecutor::new(0, 0);

        // Act
        let result = exec.return_value();

        // Assert: should return NoReturnValue error
        assert!(result.is_err(), "empty executor should fail return_value");
        let err = result.unwrap_err();
        assert!(matches!(err, SymExecError::NoReturnValue), "error should be NoReturnValue");
    }

    // -----------------------------------------------------------------------
    // Wave-12kec: 10 additional symexec engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sqrtss_and_rcpss_and_rsqrtss() {
        // Arrange: executor with 1 float arg x
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: compute 1/sqrt(x) via rsqrtss, then sqrt(x), then 1/x
        exec.step("rsqrtss", &["xmm1", "xmm0"]).unwrap();
        exec.step("sqrtss", &["xmm2", "xmm0"]).unwrap();
        exec.step("rcpss", &["xmm3", "xmm0"]).unwrap();

        // Assert: each register holds the correct symbolic variant
        let v1 = exec.get_value("xmm1");
        let v2 = exec.get_value("xmm2");
        let v3 = exec.get_value("xmm3");
        assert!(matches!(v1, SymValue::Rsqrt(_)), "rsqrtss should produce Rsqrt");
        assert!(matches!(v2, SymValue::Sqrt(_)), "sqrtss should produce Sqrt");
        assert!(matches!(v3, SymValue::Recip(_)), "rcpss should produce Recip");
    }

    #[test]
    fn test_minss_produces_min() {
        // Arrange: executor with 2 float args
        let mut exec = SymbolicExecutor::new(2, 0);

        // Act: minss xmm0, xmm1 → Min(param(0), param(1))
        exec.step("minss", &["xmm0", "xmm1"]).unwrap();

        // Assert
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Min(_, _)), "minss should produce Min");
        let s = format!("{ret}");
        assert!(s.contains("param(0)") && s.contains("param(1)"));
    }

    #[test]
    fn test_neg_float_pseudo_instruction() {
        // Arrange: executor with 1 float arg
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: apply neg_float pseudo-instruction
        exec.step("neg_float", &["xmm0"]).unwrap();

        // Assert: xmm0 should hold Neg(param(0))
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Neg(_)), "neg_float should produce Neg");
    }

    #[test]
    fn test_vfmsub231_produces_sub_mul() {
        // Arrange: executor with 3 float args (a, b, c)
        let mut exec = SymbolicExecutor::new(3, 0);

        // Act: vfmsub231ss xmm0, xmm1, xmm2 → xmm0 = xmm1*xmm2 - xmm0
        exec.step("vfmsub231ss", &["xmm0", "xmm1", "xmm2"]).unwrap();

        // Assert: result should be Sub(Mul(...), ...)
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Sub(_, _)), "vfmsub should produce Sub");
        let s = format!("{ret}");
        assert!(s.contains("param(1)"), "should reference src1 (param(1))");
        assert!(s.contains("param(2)"), "should reference src2 (param(2))");
    }

    #[test]
    fn test_vfnmadd231_produces_sub_from_dst() {
        // Arrange: executor with 3 float args (a, b, c)
        let mut exec = SymbolicExecutor::new(3, 0);

        // Act: vfnmadd231ss xmm0, xmm1, xmm2 → xmm0 = xmm0 - xmm1*xmm2
        exec.step("vfnmadd231ss", &["xmm0", "xmm1", "xmm2"]).unwrap();

        // Assert: result should be Sub(param(0), Mul(param(1), param(2)))
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Sub(_, _)), "vfnmadd should produce Sub");
    }

    #[test]
    fn test_call_sqrtf_and_tanhf_and_fabsf() {
        // Arrange: executor with 1 float arg
        // Act & Assert for each libm function

        let mut exec_sqrt = SymbolicExecutor::new(1, 0);
        exec_sqrt.step("call", &["sqrtf@PLT"]).unwrap();
        let trace_sqrt = exec_sqrt.extract_trace().unwrap();
        assert!(trace_sqrt.iter().any(|op| matches!(op, TraceOp::Sqrt(_))),
            "call sqrtf should extract to TraceOp::Sqrt");

        let mut exec_tanh = SymbolicExecutor::new(1, 0);
        exec_tanh.step("call", &["tanhf@PLT"]).unwrap();
        let trace_tanh = exec_tanh.extract_trace().unwrap();
        assert!(trace_tanh.iter().any(|op| matches!(op, TraceOp::Tanh(_))),
            "call tanhf should extract to TraceOp::Tanh");

        let mut exec_fabs = SymbolicExecutor::new(1, 0);
        exec_fabs.step("call", &["fabsf@PLT"]).unwrap();
        let trace_fabs = exec_fabs.extract_trace().unwrap();
        assert!(trace_fabs.iter().any(|op| matches!(op, TraceOp::Abs(_))),
            "call fabsf should extract to TraceOp::Abs");
    }

    #[test]
    fn test_call_unknown_function_produces_unknown() {
        // Arrange: executor with 1 float arg
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: call unrecognized function
        exec.step("call", &["custom_func@PLT"]).unwrap();

        // Assert: return value should be Unknown containing function name
        let ret = exec.return_value().unwrap();
        let s = format!("{ret}");
        assert!(s.contains("call:custom_func"), "unknown function should produce Unknown with call: prefix");
    }

    #[test]
    fn test_get_flags_after_ucomiss() {
        // Arrange: executor with 2 float args
        let mut exec = SymbolicExecutor::new(2, 0);

        // Act: perform comparison, then read flags
        assert!(exec.get_flags().is_none(), "flags should be None before comparison");
        exec.step("ucomiss", &["xmm0", "xmm1"]).unwrap();
        let flags = exec.get_flags();

        // Assert: flags should contain (param(0), param(1))
        assert!(flags.is_some(), "flags should be set after ucomiss");
        let (lhs, rhs) = flags.unwrap();
        assert!(matches!(lhs, SymValue::Param(0)), "flag lhs should be param(0)");
        assert!(matches!(rhs, SymValue::Param(1)), "flag rhs should be param(1)");
    }

    #[test]
    fn test_andps_abs_reverse_operand_order() {
        // Arrange: executor with 1 float arg, register abs mask into xmm1 first
        let mut exec = SymbolicExecutor::new(1, 0);
        let abs_mask = f32::from_bits(0x7FFF_FFFF);
        exec.register_constant(0x5000, abs_mask);
        exec.step("movss", &["xmm1", "[rip+0x5000]"]).unwrap();

        // Act: andps xmm1, xmm0 → src1=abs_mask, src2=param(0)
        // The handler checks src2 first, then src1 for abs mask
        exec.step("andps", &["xmm1", "xmm0"]).unwrap();

        // Assert: result should be Abs(param(0)) because src1 is abs mask
        let v = exec.get_value("xmm1");
        assert!(matches!(v, SymValue::Abs(_)), "andps with abs mask in src1 should produce Abs");
    }

    #[test]
    fn test_cvtsi2ss_produces_unknown() {
        // Arrange: executor with 1 float arg
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: int-to-float conversion — we don't track integer registers
        exec.step("cvtsi2ss", &["xmm0", "eax"]).unwrap();

        // Assert: xmm0 should now be Unknown("cvtsi2ss")
        let ret = exec.return_value().unwrap();
        match &ret {
            SymValue::Unknown(s) => assert!(s.contains("cvtsi2ss"),
                "cvtsi2ss should set Unknown with cvtsi2ss label, got: {s}"),
            other => panic!("expected Unknown, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Wave-12kmb: 10 additional symexec engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_divss_trace_extraction() {
        // Arrange: executor with 2 float args, compute x / y
        let mut exec = SymbolicExecutor::new(2, 0);

        // Act: divss xmm0, xmm1 → param(0) / param(1)
        exec.step("divss", &["xmm0", "xmm1"]).unwrap();

        // Assert: extract_trace should produce [Input(0), Input(1), Div(0, 1)]
        let trace = exec.extract_trace().unwrap();
        assert_eq!(trace.len(), 3, "div trace should have 3 ops");
        assert_eq!(trace[0], TraceOp::Input(0));
        assert_eq!(trace[1], TraceOp::Input(1));
        assert_eq!(trace[2], TraceOp::Div(ValueId(0), ValueId(1)));
    }

    #[test]
    fn test_vandps_vex_form_abs() {
        // Arrange: executor with 1 float arg, abs mask constant
        let mut exec = SymbolicExecutor::new(1, 0);
        let abs_mask = f32::from_bits(0x7FFF_FFFF);
        exec.register_constant(0x5000, abs_mask);
        exec.step("movss", &["xmm1", "[rip+0x5000]"]).unwrap();

        // Act: VEX 3-operand form vandps xmm2, xmm0, xmm1
        // dst=xmm2, src1=xmm0=param(0), src2=xmm1=abs_mask
        exec.step("vandps", &["xmm2", "xmm0", "xmm1"]).unwrap();

        // Assert: xmm2 should be Abs(param(0))
        let v = exec.get_value("xmm2");
        assert!(matches!(v, SymValue::Abs(_)), "vandps with abs mask in src2 should produce Abs");
    }

    #[test]
    fn test_orps_produces_unknown() {
        // Arrange: executor with 1 float arg
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: orps xmm0, xmm1 — not a recognized pattern
        exec.step("orps", &["xmm0", "xmm1"]).unwrap();

        // Assert: result should be Unknown containing "or"
        let ret = exec.return_value().unwrap();
        match &ret {
            SymValue::Unknown(s) => assert!(s.contains("or"),
                "orps should produce Unknown with 'or' label, got: {s}"),
            other => panic!("expected Unknown from orps, got: {other:?}"),
        }
    }

    #[test]
    fn test_andnps_produces_unknown() {
        // Arrange: executor with 1 float arg
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: andnps xmm0, xmm1 — andnot pattern
        exec.step("andnps", &["xmm0", "xmm1"]).unwrap();

        // Assert: result should be Unknown containing "andnot"
        let ret = exec.return_value().unwrap();
        match &ret {
            SymValue::Unknown(s) => assert!(s.contains("andnot"),
                "andnps should produce Unknown with 'andnot' label, got: {s}"),
            other => panic!("expected Unknown from andnps, got: {other:?}"),
        }
    }

    #[test]
    fn test_bin_op_wrong_operand_count_errors() {
        // Arrange: executor with 1 float arg
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: addss with 4 operands (invalid)
        let result = exec.step("addss", &["xmm0", "xmm1", "xmm2", "xmm3"]);

        // Assert: should return UnsupportedInstruction error
        assert!(result.is_err(), "bin_op with 4 operands should fail");
        let err = result.unwrap_err();
        assert!(matches!(err, SymExecError::UnsupportedInstruction(_)),
            "error should be UnsupportedInstruction");
        assert!(err.to_string().contains("unexpected operand count"),
            "error message should mention operand count");
    }

    #[test]
    fn test_fma_wrong_operand_count_errors() {
        // Arrange: executor with 2 float args
        let mut exec = SymbolicExecutor::new(2, 0);

        // Act: vfmadd231ss with only 2 operands (FMA requires 3)
        let result = exec.step("vfmadd231ss", &["xmm0", "xmm1"]);

        // Assert: should return UnsupportedInstruction error about operand count
        assert!(result.is_err(), "FMA with 2 operands should fail");
        let err = result.unwrap_err();
        assert!(matches!(err, SymExecError::UnsupportedInstruction(_)));
        assert!(err.to_string().contains("3 operands"),
            "error should mention FMA requires 3 operands");
    }

    #[test]
    fn test_snapshot_preserves_constants_and_flags() {
        // Arrange: executor with constants and flags set
        let mut exec = SymbolicExecutor::new(2, 0);
        exec.register_constant(0xABCD, 3.14);
        exec.step("ucomiss", &["xmm0", "xmm1"]).unwrap();
        assert!(exec.get_flags().is_some(), "flags should be set before snapshot");
        let snap = exec.snapshot();

        // Act: clobber constants and flags by creating a fresh executor and restoring
        let mut exec2 = SymbolicExecutor::new(0, 0);
        assert!(exec2.get_flags().is_none(), "fresh executor has no flags");
        exec2.restore(&snap);

        // Assert: restored executor has flags from snapshot
        let flags = exec2.get_flags();
        assert!(flags.is_some(), "restored executor should have flags");
        let (lhs, rhs) = flags.unwrap();
        assert!(matches!(lhs, SymValue::Param(0)));
        assert!(matches!(rhs, SymValue::Param(1)));
    }

    #[test]
    fn test_resolve_unknown_stack_memory() {
        // Arrange: executor with 1 float arg, no stack spills at offset 0x40
        let mut exec = SymbolicExecutor::new(1, 0);

        // Act: load from uninitialized stack slot
        exec.step("movss", &["xmm0", "[rsp+0x40]"]).unwrap();

        // Assert: should resolve to Unknown with mem: prefix
        let ret = exec.return_value().unwrap();
        match &ret {
            SymValue::Unknown(s) => assert!(s.contains("mem:"),
                "load from unknown stack should produce Unknown with mem: prefix, got: {s}"),
            other => panic!("expected Unknown, got: {other:?}"),
        }
    }

    #[test]
    fn test_movss_store_to_non_stack_memory_noop() {
        // Arrange: executor with 1 float arg
        let mut exec = SymbolicExecutor::new(1, 0);
        let original = exec.return_value().unwrap();

        // Act: store to non-RSP memory address — should be silently ignored
        exec.step("movss", &["[rdi+0x10]", "xmm0"]).unwrap();

        // Assert: xmm0 is unchanged (not clobbered by the store)
        let ret = exec.return_value().unwrap();
        assert!(matches!(ret, SymValue::Param(0)),
            "xmm0 should still be param(0) after store to non-stack memory");
        // Stack should be empty (store didn't go to [rsp+...] slot)
        assert!(exec.stack_state().is_empty(),
            "stack should be empty after store to non-stack memory");
    }

    #[test]
    fn test_extract_trace_sqrt_and_recip() {
        // Arrange: build 1/sqrt(x) as a symbolic expression and set as return
        let mut exec = SymbolicExecutor::new(1, 0);
        let x = SymValue::Param(0);
        let sqrt_x = SymValue::Sqrt(Box::new(x.clone()));
        let recip_sqrt_x = SymValue::Recip(Box::new(sqrt_x));
        exec.set("xmm0", recip_sqrt_x);

        // Act
        let trace = exec.extract_trace().unwrap();

        // Assert: should have Input(0), Sqrt, Recip in that SSA order
        assert_eq!(trace.len(), 3, "expected 3 ops: Input, Sqrt, Recip");
        assert_eq!(trace[0], TraceOp::Input(0));
        assert_eq!(trace[1], TraceOp::Sqrt(ValueId(0)));
        assert_eq!(trace[2], TraceOp::Recip(ValueId(1)));
    }
}
