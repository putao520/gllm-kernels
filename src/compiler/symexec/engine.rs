use std::collections::HashMap;
use super::sym_value::{SymValue, LibmFn};
use crate::compiler::trace::TraceOp;

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

    /// Return the symbolic value in xmm0 (the return register for floats).
    pub fn return_value(&self) -> Result<SymValue, SymExecError> {
        self.regs
            .get("xmm0")
            .cloned()
            .ok_or(SymExecError::NoReturnValue)
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
                    let src = self.resolve(operands.last().expect("guarded by operands.len() >= 2"));
                    self.set(operands[0], SymValue::Sqrt(Box::new(src)));
                }
                Ok(())
            }
            "rcpss" | "vrcpss" => {
                if operands.len() >= 2 {
                    let src = self.resolve(operands.last().expect("guarded by operands.len() >= 2"));
                    self.set(operands[0], SymValue::Recip(Box::new(src)));
                }
                Ok(())
            }
            "rsqrtss" | "vrsqrtss" => {
                if operands.len() >= 2 {
                    let src = self.resolve(operands.last().expect("guarded by operands.len() >= 2"));
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
            "call" => self.call_op(operands),

            // --- No-ops for symbolic execution ---
            "ret" | "push" | "pop" | "endbr64" | "nop" | "lea" | "mov" | "sub" | "add"
            | "cmp" | "test" | "je" | "jne" | "jmp" | "jb" | "ja" | "jbe" | "jae"
            | "seta" | "setb" | "setae" | "setbe" | "sete" | "setne"
            | "setg" | "setge" | "setl" | "setle"
            | "and" | "or" | "xor" | "shr" | "shl" | "sar" | "sal"
            | "cdq" | "cdqe" | "cqo" => {
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
            let arg = self.get("xmm0");
            self.set("xmm0", SymValue::Call(f, vec![arg]));
        } else {
            self.set("xmm0", SymValue::Unknown(format!("call:{target}")));
        }
        Ok(())
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
    cache: &mut HashMap<String, u32>,
) -> u32 {
    let key = format!("{val}");
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let idx = match val {
        SymValue::Param(n) => {
            let i = ops.len() as u32;
            ops.push(TraceOp::Input(*n as u32));
            i
        }
        SymValue::Const(v) => {
            let i = ops.len() as u32;
            ops.push(TraceOp::Const(*v));
            i
        }
        SymValue::Add(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Add(ai, bi));
            i
        }
        SymValue::Sub(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Sub(ai, bi));
            i
        }
        SymValue::Mul(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Mul(ai, bi));
            i
        }
        SymValue::Div(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Div(ai, bi));
            i
        }
        SymValue::Fma(a, b, c) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let ci = linearize(c, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Fma(ai, bi, ci));
            i
        }
        SymValue::Neg(a) => {
            let ai = linearize(a, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Neg(ai));
            i
        }
        SymValue::Abs(a) => {
            let ai = linearize(a, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Abs(ai));
            i
        }
        SymValue::Max(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Max(ai, bi));
            i
        }
        SymValue::Min(a, b) => {
            let ai = linearize(a, ops, cache);
            let bi = linearize(b, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Min(ai, bi));
            i
        }
        SymValue::Sqrt(a) => {
            let ai = linearize(a, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Sqrt(ai));
            i
        }
        SymValue::Recip(a) => {
            let ai = linearize(a, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Recip(ai));
            i
        }
        SymValue::Rsqrt(a) => {
            let ai = linearize(a, ops, cache);
            let i = ops.len() as u32;
            ops.push(TraceOp::Rsqrt(ai));
            i
        }
        SymValue::Call(func, args) => {
            match func {
                LibmFn::Expf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ops.len() as u32;
                    ops.push(TraceOp::Exp(ai));
                    i
                }
                LibmFn::Tanhf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ops.len() as u32;
                    ops.push(TraceOp::Tanh(ai));
                    i
                }
                LibmFn::Sqrtf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ops.len() as u32;
                    ops.push(TraceOp::Sqrt(ai));
                    i
                }
                LibmFn::Fabsf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ops.len() as u32;
                    ops.push(TraceOp::Abs(ai));
                    i
                }
                LibmFn::Logf => {
                    let ai = linearize(&args[0], ops, cache);
                    let i = ops.len() as u32;
                    ops.push(TraceOp::Log(ai));
                    i
                }
            }
        }
        SymValue::Load { .. } | SymValue::Unknown(_) => {
            // Unresolvable — emit as Input(0) placeholder.
            let i = ops.len() as u32;
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
        assert_eq!(trace[2], TraceOp::Add(0, 1));
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
        assert_eq!(trace[2], TraceOp::Max(0, 1));
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
                    assert!((*a as usize) < i, "SSA violation at {i}");
                }
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b)
                | TraceOp::Div(a, b) | TraceOp::Max(a, b) | TraceOp::Min(a, b) => {
                    assert!((*a as usize) < i, "SSA violation at {i}");
                    assert!((*b as usize) < i, "SSA violation at {i}");
                }
                TraceOp::Fma(a, b, c) => {
                    assert!((*a as usize) < i, "SSA violation at {i}");
                    assert!((*b as usize) < i, "SSA violation at {i}");
                    assert!((*c as usize) < i, "SSA violation at {i}");
                }
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
        assert_eq!(trace[1], TraceOp::Add(0, 0));
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
        assert_eq!(trace[1], TraceOp::Log(0));
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
        assert_eq!(trace[3], TraceOp::Mul(2, 2), "Mul should reference the same Add index twice");
    }
}
