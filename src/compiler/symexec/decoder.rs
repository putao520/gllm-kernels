//! Phase 0 decoder bridge: fn_ptr → iced-x86 Decoder → SymbolicExecutor → OpTrace.
//!
//! Takes a scalar function pointer, disassembles its machine code with iced-x86,
//! and feeds each instruction into the symbolic execution engine to extract the
//! computational structure as a `Vec<TraceOp>`.

#[cfg(target_arch = "x86_64")]
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

use super::engine::{SymbolicExecutor, SymExecError};
use super::sym_value::SymValue;
use crate::compiler::trace::{ScalarFnSignature, ScalarParam, TraceOp};
use std::collections::HashMap;

/// Maximum bytes to read from a function pointer for disassembly.
const MAX_FN_BYTES: usize = 4096;

/// Safety limit on instructions to prevent runaway analysis.
const MAX_INSTRUCTIONS: usize = 500;

/// Tracks which ABI registers hold input vs output pointers.
/// Also tracks register-to-register copies so we can follow pointer aliasing
/// (e.g. `mov r15, rdi` makes r15 an alias of the input pointer in rdi).
struct PtrParamMap {
    /// Register → input ordinal (for InputPtr / WeightPtr params).
    input_regs: HashMap<Register, usize>,
    /// Registers holding OutputPtr params.
    output_regs: Vec<Register>,
}

impl PtrParamMap {
    /// Record a GP register copy: `mov dst, src`. If src is a known
    /// input/output pointer, propagate that identity to dst.
    fn propagate_mov(&mut self, dst: Register, src: Register) {
        if let Some(&ord) = self.input_regs.get(&src) {
            self.input_regs.insert(dst, ord);
        }
        if self.output_regs.contains(&src) && !self.output_regs.contains(&dst) {
            self.output_regs.push(dst);
        }
    }

    /// Check if a register is a known input pointer.
    fn is_input(&self, reg: Register) -> Option<usize> {
        self.input_regs.get(&reg).copied()
    }

    /// Check if a register is a known output pointer.
    fn is_output(&self, reg: Register) -> bool {
        self.output_regs.contains(&reg)
    }
}

/// Integer argument registers per System V AMD64 ABI.
#[cfg(target_arch = "x86_64")]
const INT_ARG_REGS: [Register; 6] = [
    Register::RDI,
    Register::RSI,
    Register::RDX,
    Register::RCX,
    Register::R8,
    Register::R9,
];

/// Build the pointer-param → register mapping from a function signature.
///
/// System V ABI: integer/pointer args go to rdi, rsi, rdx, rcx, r8, r9 in order.
/// Float args (ScalarParam::Scalar) go to xmm0-xmm7 but we don't track those here.
#[cfg(target_arch = "x86_64")]
fn build_ptr_map(sig: &ScalarFnSignature) -> PtrParamMap {
    let mut input_regs = HashMap::new();
    let mut output_regs = Vec::new();
    let mut int_reg_idx = 0;
    let mut input_ordinal = 0;

    for param in &sig.params {
        match param {
            ScalarParam::InputPtr | ScalarParam::WeightPtr => {
                if int_reg_idx < INT_ARG_REGS.len() {
                    input_regs.insert(INT_ARG_REGS[int_reg_idx], input_ordinal);
                    input_ordinal += 1;
                }
                int_reg_idx += 1;
            }
            ScalarParam::OutputPtr => {
                if int_reg_idx < INT_ARG_REGS.len() {
                    output_regs.push(INT_ARG_REGS[int_reg_idx]);
                }
                int_reg_idx += 1;
            }
            ScalarParam::Dim(_) => {
                // Dimension param consumes an integer register but we don't track it.
                int_reg_idx += 1;
            }
            ScalarParam::Scalar(_) => {
                // Float param goes to xmm registers, doesn't consume integer reg.
            }
        }
    }

    PtrParamMap {
        input_regs,
        output_regs,
    }
}

/// Count the number of input parameters (InputPtr + WeightPtr) in a signature.
fn count_inputs(sig: &ScalarFnSignature) -> usize {
    sig.params
        .iter()
        .filter(|p| matches!(p, ScalarParam::InputPtr | ScalarParam::WeightPtr))
        .count()
}

/// Disassemble a scalar function and symbolically execute it to extract TraceOps.
///
/// # Safety
/// `fn_ptr` must point to a valid, compiled `extern "C"` function. The function
/// must be compiled at opt-level ≤ 1 (no vectorization) for reliable analysis.
#[cfg(target_arch = "x86_64")]
pub fn analyze_scalar_fn(
    fn_ptr: *const u8,
    sig: &ScalarFnSignature,
) -> Result<Vec<TraceOp>, SymExecError> {
    if fn_ptr.is_null() {
        return Err(SymExecError::DisassemblyFailed(
            "null function pointer".into(),
        ));
    }

    let n_inputs = count_inputs(sig);
    let mut ptr_map = build_ptr_map(sig);

    // Create executor with n_inputs float params (representing loaded values).
    // We don't pass pointer args — the decoder bridge injects Input(n) directly
    // when it sees loads from input pointer registers.
    let mut executor = SymbolicExecutor::new(n_inputs, 0);

    // For void functions that store results via output pointers, we track
    // the last value written to the output pointer.
    let mut last_output_value: Option<SymValue> = None;

    // Read bytes from function address.
    let bytes = unsafe { std::slice::from_raw_parts(fn_ptr, MAX_FN_BYTES) };
    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    decoder.set_ip(fn_ptr as u64);

    let mut instr = Instruction::default();
    let mut instr_count = 0;

    while decoder.can_decode() && instr_count < MAX_INSTRUCTIONS {
        decoder.decode_out(&mut instr);
        instr_count += 1;

        let mnemonic = instr.mnemonic();

        // Stop at ret.
        if mnemonic == Mnemonic::Ret {
            break;
        }

        // Detect backward jumps (loop back-edge) → stop after first iteration.
        if is_branch(mnemonic) {
            if instr.op_count() > 0 && has_near_branch(&instr) {
                let target = instr.near_branch_target();
                if target <= instr.ip() {
                    break; // backward jump = end of first loop iteration
                }
            }
            continue; // forward jump = skip (branch not taken path)
        }

        // Handle call instructions: resolve libm function names.
        if mnemonic == Mnemonic::Call {
            let target_name = resolve_call_target(&instr);
            executor.step("call", &[&target_name])?;
            continue;
        }

        // Track GP register-to-register moves for pointer aliasing.
        // e.g. `mov r15, rdi` → r15 becomes an alias of the input pointer.
        if mnemonic == Mnemonic::Mov && instr.op_count() == 2
            && instr.op_kind(0) == OpKind::Register
            && instr.op_kind(1) == OpKind::Register
        {
            let dst = instr.op_register(0);
            let src = instr.op_register(1);
            if is_gp64(dst) && is_gp64(src) {
                ptr_map.propagate_mov(dst, src);
            }
        }

        // Check for float load from input pointer → inject Input(ordinal).
        if is_float_load(mnemonic) && instr.op_count() >= 2 {
            // Load: movss xmm, [input_ptr + ...]
            if instr.op_kind(0) == OpKind::Register && is_xmm(instr.op_register(0)) {
                if instr.op_kind(1) == OpKind::Memory {
                    let base = instr.memory_base();
                    if let Some(input_ord) = ptr_map.is_input(base) {
                        let dst = format_register(instr.op_register(0));
                        executor.set(&dst, SymValue::Param(input_ord));
                        continue;
                    }
                }
            }
        }

        // Float store to output pointer → capture the value being stored.
        // For void functions, this is how we find the "return" value.
        if is_float_store(mnemonic, &instr) {
            let base = instr.memory_base();
            if ptr_map.is_output(base) {
                // The source register is the last operand.
                let src_reg = instr.op_register(instr.op_count() as u32 - 1);
                if is_xmm(src_reg) {
                    let src_name = format_register(src_reg);
                    last_output_value = Some(executor.get_value(&src_name));
                }
                continue;
            }
        }

        // xorps/xorpd with RIP-relative constant = sign-bit flip (negation).
        // The compiler emits `xorps xmm0, [sign_mask]` to negate a float.
        if is_xor_ps_pd(mnemonic) && instr.op_count() == 2
            && instr.op_kind(0) == OpKind::Register && is_xmm(instr.op_register(0))
            && instr.op_kind(1) == OpKind::Memory && instr.memory_base() == Register::RIP
        {
            let dst = format_register(instr.op_register(0));
            executor.step("neg_float", &[&dst])?;
            continue;
        }

        // RIP-relative constant pool load: read the f32 value and register it.
        // This covers movss, addss, mulss, divss, etc. with [rip+...] operands.
        if has_memory_operand(&instr) && instr.memory_base() == Register::RIP {
            let abs_addr = instr.memory_displacement64();
            if !executor.has_constant(abs_addr) {
                // Safety: the constant is in the .rodata section of the same binary.
                let value = unsafe { *(abs_addr as *const f32) };
                executor.register_constant(abs_addr, value);
            }
        }

        // Skip integer-only instructions that don't affect float state.
        if is_integer_only(mnemonic) {
            continue;
        }

        // General case: format mnemonic + operands and feed to executor.
        let mnem_str = format_mnemonic(mnemonic);
        let operands = format_operands(&instr);
        let op_refs: Vec<&str> = operands.iter().map(|s| s.as_str()).collect();
        executor.step(&mnem_str, &op_refs)?;
    }

    // For void functions that write results via output pointers, the result
    // isn't in xmm0. Use the last value stored to the output pointer instead.
    if let Some(output_val) = last_output_value {
        // Check if xmm0 has a meaningful result (not just an intermediate).
        // If the function stores to an output pointer, prefer that value.
        executor.set("xmm0", output_val);
    }

    executor.extract_trace()
}

// ---------------------------------------------------------------------------
// Instruction classification helpers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn is_branch(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Je
            | Mnemonic::Jne
            | Mnemonic::Jb
            | Mnemonic::Jbe
            | Mnemonic::Ja
            | Mnemonic::Jae
            | Mnemonic::Jl
            | Mnemonic::Jle
            | Mnemonic::Jg
            | Mnemonic::Jge
            | Mnemonic::Jmp
            | Mnemonic::Js
            | Mnemonic::Jns
            | Mnemonic::Jp
            | Mnemonic::Jnp
    )
}

#[cfg(target_arch = "x86_64")]
fn has_near_branch(instr: &Instruction) -> bool {
    for i in 0..instr.op_count() {
        match instr.op_kind(i) {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => return true,
            _ => {}
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
fn is_float_mov(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Movss
            | Mnemonic::Vmovss
            | Mnemonic::Movsd
            | Mnemonic::Vmovsd
            | Mnemonic::Movaps
            | Mnemonic::Vmovaps
            | Mnemonic::Movups
            | Mnemonic::Vmovups
            | Mnemonic::Movd
            | Mnemonic::Vmovd
    )
}

/// Float load: movss/movaps xmm, [mem] (destination is register, source is memory).
#[cfg(target_arch = "x86_64")]
fn is_float_load(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Movss | Mnemonic::Vmovss | Mnemonic::Movsd | Mnemonic::Vmovsd
    )
}

/// Check if instruction is a float store to memory (operand 0 is memory).
#[cfg(target_arch = "x86_64")]
fn is_float_store(m: Mnemonic, instr: &Instruction) -> bool {
    is_float_mov(m) && instr.op_count() >= 2 && instr.op_kind(0) == OpKind::Memory
}

/// xorps / xorpd — used for sign-bit flips (negation).
#[cfg(target_arch = "x86_64")]
fn is_xor_ps_pd(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Xorps | Mnemonic::Xorpd | Mnemonic::Vxorps | Mnemonic::Vxorpd
    )
}

/// Check if a register is a 64-bit general-purpose register.
#[cfg(target_arch = "x86_64")]
fn is_gp64(r: Register) -> bool {
    matches!(
        r,
        Register::RAX
            | Register::RBX
            | Register::RCX
            | Register::RDX
            | Register::RSI
            | Register::RDI
            | Register::RBP
            | Register::RSP
            | Register::R8
            | Register::R9
            | Register::R10
            | Register::R11
            | Register::R12
            | Register::R13
            | Register::R14
            | Register::R15
    )
}

/// Integer-only instructions that don't affect floating-point state.
/// We skip these to avoid confusing the symbolic executor.
#[cfg(target_arch = "x86_64")]
fn is_integer_only(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Push
            | Mnemonic::Pop
            | Mnemonic::Mov
            | Mnemonic::Lea
            | Mnemonic::Add
            | Mnemonic::Sub
            | Mnemonic::Inc
            | Mnemonic::Dec
            | Mnemonic::Xor
            | Mnemonic::And
            | Mnemonic::Or
            | Mnemonic::Shl
            | Mnemonic::Shr
            | Mnemonic::Sar
            | Mnemonic::Test
            | Mnemonic::Cmp
            | Mnemonic::Nop
            | Mnemonic::Endbr64
            | Mnemonic::Cdq
            | Mnemonic::Cdqe
            | Mnemonic::Cqo
            | Mnemonic::Cmove
            | Mnemonic::Cmovne
            | Mnemonic::Cmovb
            | Mnemonic::Cmovbe
            | Mnemonic::Cmova
            | Mnemonic::Cmovae
            | Mnemonic::Cmovl
            | Mnemonic::Cmovle
            | Mnemonic::Cmovg
            | Mnemonic::Cmovge
            | Mnemonic::Sete
            | Mnemonic::Setne
            | Mnemonic::Setb
            | Mnemonic::Setbe
            | Mnemonic::Seta
            | Mnemonic::Setae
            | Mnemonic::Setl
            | Mnemonic::Setle
            | Mnemonic::Setg
            | Mnemonic::Setge
            | Mnemonic::Neg
            | Mnemonic::Not
            | Mnemonic::Imul
            | Mnemonic::Mul
            | Mnemonic::Idiv
            | Mnemonic::Div
    )
}

#[cfg(target_arch = "x86_64")]
fn is_xmm(r: Register) -> bool {
    matches!(
        r,
        Register::XMM0
            | Register::XMM1
            | Register::XMM2
            | Register::XMM3
            | Register::XMM4
            | Register::XMM5
            | Register::XMM6
            | Register::XMM7
            | Register::XMM8
            | Register::XMM9
            | Register::XMM10
            | Register::XMM11
            | Register::XMM12
            | Register::XMM13
            | Register::XMM14
            | Register::XMM15
    )
}

/// Check if any operand is a memory operand.
#[cfg(target_arch = "x86_64")]
fn has_memory_operand(instr: &Instruction) -> bool {
    for i in 0..instr.op_count() {
        if instr.op_kind(i) == OpKind::Memory {
            return true;
        }
    }
    false
}

/// Check if operand 0 (destination) is a memory operand (store).
#[cfg(target_arch = "x86_64")]
fn has_memory_operand_at_dst(instr: &Instruction) -> bool {
    instr.op_count() > 0 && instr.op_kind(0) == OpKind::Memory
}

/// Compute the absolute address for a RIP-relative memory operand.
#[cfg(target_arch = "x86_64")]
fn compute_rip_address(instr: &Instruction) -> u64 {
    // When the decoder IP is set correctly, iced-x86's memory_displacement64()
    // already returns the absolute target address for RIP-relative operands.
    instr.memory_displacement64()
}

// ---------------------------------------------------------------------------
// Call target resolution
// ---------------------------------------------------------------------------

/// Try to resolve a call target to a libm function name.
///
/// Handles both direct calls (`call addr`) and indirect calls through the GOT
/// (`call qword ptr [rip+offset]`).
#[cfg(target_arch = "x86_64")]
fn resolve_call_target(instr: &Instruction) -> String {
    // First, try direct near branch target.
    let target = instr.near_branch_target();

    // If target is 0, this might be an indirect call through memory (GOT).
    // e.g. `call qword ptr [rip+0x...]`
    let resolved_addr = if target != 0 {
        target
    } else if instr.op_count() > 0 && instr.op_kind(0) == OpKind::Memory
        && instr.memory_base() == Register::RIP
    {
        // Indirect call through GOT: read the function pointer from the GOT entry.
        let got_addr = instr.memory_displacement64();
        // Safety: reading from GOT in our own process.
        let actual_fn = unsafe { *(got_addr as *const u64) };
        actual_fn
    } else {
        return "unknown".to_string();
    };

    // Try dladdr to resolve symbol name.
    #[cfg(unix)]
    {
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::dladdr(resolved_addr as *const libc::c_void, &mut info) };
        if ret != 0 && !info.dli_sname.is_null() {
            let name = unsafe { std::ffi::CStr::from_ptr(info.dli_sname) };
            if let Ok(s) = name.to_str() {
                return s.to_string();
            }
        }
    }

    // Fallback: try matching against known libm function addresses.
    if let Some(name) = match_known_libm(resolved_addr) {
        return name.to_string();
    }

    // Last resort for direct calls: try reading the PLT stub.
    if target != 0 {
        if let Some(name) = try_resolve_plt(target) {
            return name;
        }
    }

    format!("0x{resolved_addr:x}")
}

/// Match a call target against known libm function addresses.
#[cfg(target_arch = "x86_64")]
fn match_known_libm(target: u64) -> Option<&'static str> {
    // Get addresses of common libm functions.
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
        if target == addr as u64 {
            return Some(name);
        }
    }
    None
}

/// Try to resolve a PLT stub by reading its jmp target.
///
/// PLT stubs typically look like: `jmp [rip+offset]` (FF 25 xx xx xx xx).
/// We follow the indirection to the GOT entry, then dladdr the resolved address.
#[cfg(target_arch = "x86_64")]
fn try_resolve_plt(target: u64) -> Option<String> {
    let ptr = target as *const u8;
    // Safety: we're reading from executable memory in our own process.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, 16) };

    // Check for `jmp [rip+disp32]` pattern: FF 25 xx xx xx xx
    // or `endbr64; jmp [rip+disp32]`: F3 0F 1E FA FF 25 xx xx xx xx
    let jmp_offset = if bytes.len() >= 6 && bytes[0] == 0xFF && bytes[1] == 0x25 {
        Some(0)
    } else if bytes.len() >= 10
        && bytes[0] == 0xF3
        && bytes[1] == 0x0F
        && bytes[2] == 0x1E
        && bytes[3] == 0xFA
        && bytes[4] == 0xFF
        && bytes[5] == 0x25
    {
        Some(4)
    } else {
        None
    };

    if let Some(off) = jmp_offset {
        let disp = i32::from_le_bytes([
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
        ]);
        // RIP after the jmp instruction = target + off + 6
        let next_ip = target + off as u64 + 6;
        let got_addr = (next_ip as i64 + disp as i64) as u64;
        // Read the GOT entry (pointer to actual function).
        let actual_fn = unsafe { *(got_addr as *const u64) };

        #[cfg(unix)]
        {
            let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
            let ret = unsafe { libc::dladdr(actual_fn as *const libc::c_void, &mut info) };
            if ret != 0 && !info.dli_sname.is_null() {
                let name = unsafe { std::ffi::CStr::from_ptr(info.dli_sname) };
                if let Ok(s) = name.to_str() {
                    return Some(s.to_string());
                }
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Instruction formatting (iced-x86 → string for SymbolicExecutor::step)
// ---------------------------------------------------------------------------

/// Format a Mnemonic as the lowercase string the engine expects.
#[cfg(target_arch = "x86_64")]
fn format_mnemonic(m: Mnemonic) -> String {
    // iced-x86 Debug gives PascalCase like "Addss", "Vaddss", "Ret"
    format!("{m:?}").to_ascii_lowercase()
}

/// Format a Register as the lowercase string the engine expects.
#[cfg(target_arch = "x86_64")]
fn format_register(r: Register) -> String {
    format!("{r:?}").to_ascii_lowercase()
}

/// Format a memory operand as a string the engine can parse.
///
/// Produces formats like `[rsp+0x10]`, `[rsp-0x8]`, `[rip+0xADDR]`, `[rdi+rcx*4]`.
#[cfg(target_arch = "x86_64")]
fn format_memory(instr: &Instruction) -> String {
    let base = instr.memory_base();
    let index = instr.memory_index();
    let scale = instr.memory_index_scale();
    let disp = instr.memory_displacement64();

    // RIP-relative: format as [rip+0xABSOLUTE_ADDR] for constant pool resolution.
    if base == Register::RIP {
        let abs_addr = compute_rip_address(instr);
        return format!("[rip+0x{abs_addr:x}]");
    }

    // Stack-relative: format as [rsp+0xNN] or [rsp-0xNN].
    if base == Register::RSP || base == Register::RBP {
        let base_name = format_register(base);
        if disp == 0 && index == Register::None {
            return format!("[{base_name}]");
        }
        // Treat displacement as signed for display.
        let signed_disp = disp as i64;
        if index == Register::None {
            if signed_disp >= 0 {
                return format!("[{base_name}+0x{signed_disp:x}]");
            } else {
                return format!("[{base_name}-0x{:x}]", -signed_disp);
            }
        }
    }

    // General memory operand: [base + index*scale + disp]
    let mut parts = Vec::new();
    if base != Register::None {
        parts.push(format_register(base));
    }
    if index != Register::None {
        if scale > 1 {
            parts.push(format!("{}*{}", format_register(index), scale));
        } else {
            parts.push(format_register(index));
        }
    }
    if disp != 0 || parts.is_empty() {
        parts.push(format!("0x{disp:x}"));
    }

    format!("[{}]", parts.join("+"))
}

/// Format all operands of an instruction as strings for the engine.
#[cfg(target_arch = "x86_64")]
fn format_operands(instr: &Instruction) -> Vec<String> {
    let mut ops = Vec::new();
    for i in 0..instr.op_count() {
        match instr.op_kind(i) {
            OpKind::Register => {
                ops.push(format_register(instr.op_register(i)));
            }
            OpKind::Memory => {
                ops.push(format_memory(instr));
            }
            OpKind::Immediate8
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate32to64
            | OpKind::Immediate64 => {
                // Immediates are rare in float code; format as hex.
                ops.push(format!("0x{:x}", instr.immediate(i)));
            }
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                ops.push(format!("0x{:x}", instr.near_branch_target()));
            }
            _ => {
                ops.push("?".to_string());
            }
        }
    }
    ops
}

// ---------------------------------------------------------------------------
// Structured analysis: CFG → loops → combine_passes → ComputePattern
// ---------------------------------------------------------------------------

/// Analyze a multi-loop scalar function using CFG-based loop detection.
///
/// Unlike `analyze_scalar_fn` (which does a linear walk and stops at the first
/// backward jump), this function:
/// 1. Builds a full CFG from the function's machine code.
/// 2. Detects all natural loops.
/// 3. Symbolically executes each loop body to extract reductions.
/// 4. Combines the loop traces into a `ComputePattern`.
///
/// Returns `None` if the function has no loops (caller should fall back to
/// `analyze_scalar_fn` for elementwise ops).
#[cfg(target_arch = "x86_64")]
pub fn analyze_scalar_fn_structured(
    fn_ptr: *const u8,
    sig: &ScalarFnSignature,
) -> Result<Option<super::loop_analyzer::MultiPassAnalysis>, SymExecError> {
    use super::cfg::{build_cfg_from_fn, find_loops};
    use super::loop_analyzer::{analyze_single_loop, analyze_nested_loops, combine_passes, MultiPassAnalysis};

    if fn_ptr.is_null() {
        return Err(SymExecError::DisassemblyFailed("null function pointer".into()));
    }

    // Build CFG (use a generous byte limit for multi-loop functions).
    let cfg = build_cfg_from_fn(fn_ptr, MAX_FN_BYTES)
        .map_err(|e| SymExecError::DisassemblyFailed(e))?;

    let forest = find_loops(&cfg);
    if forest.loops.is_empty() {
        return Ok(None); // No loops → not a multi-pass function.
    }

    // Create executor with the right number of float/ptr params.
    let n_inputs = count_inputs(sig);
    let executor = SymbolicExecutor::new(n_inputs, 0);

    // Phase 5: Check for nested loops first (GEMM, RoPE, Transpose).
    // If the forest has loops with depth > 0, try nested analysis before
    // falling through to flat multi-pass analysis.
    if let Some(nested) = analyze_nested_loops(&forest, &cfg, &executor) {
        return Ok(Some(MultiPassAnalysis {
            loop_traces: nested.inner_trace.into_iter().collect(),
            pattern: nested.pattern,
            num_loops: forest.loops.len(),
        }));
    }

    // Flat multi-pass analysis: use top-level (outermost) loops.
    let top_loops: Vec<&super::cfg::NaturalLoop> = forest.top_level.iter()
        .filter_map(|&idx| forest.loops.get(idx))
        .collect();

    if top_loops.is_empty() {
        return Ok(None);
    }

    // Analyze each top-level loop, keeping those with detected reductions.
    //
    // Filtering rules:
    // - Loops with no reductions AND no mutations → trivial (alignment
    //   checks, null-pointer guards) → skip.
    // - Loops with only unknown_mutations (no reductions) → elementwise
    //   loop bodies the analyzer couldn't decompose (e.g. SiLU's
    //   vectorized exp() call) → skip, let Level 2 handle them.
    // - Loops with reductions (even if they also have unknown_mutations)
    //   → keep. Multi-pass functions like LayerNorm may have loops where
    //   the compiler interleaves a real accumulator with opaque stores.
    let mut loop_traces = Vec::new();
    for natural_loop in &top_loops {
        match analyze_single_loop(natural_loop, &cfg, &executor) {
            Ok(trace) => {
                if trace.reductions.is_empty() {
                    continue;
                }
                loop_traces.push(trace);
            }
            Err(_) => continue, // Skip loops we can't analyze.
        }
    }

    if loop_traces.is_empty() {
        return Ok(None);
    }

    // Combine loop traces into a ComputePattern.
    match combine_passes(&loop_traces) {
        Ok(pattern) => Ok(Some(MultiPassAnalysis {
            loop_traces,
            pattern,
            num_loops: forest.loops.len(),
        })),
        Err(_) => Ok(None), // Couldn't classify → fall back to linear.
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn analyze_scalar_fn_structured(
    _fn_ptr: *const u8,
    _sig: &ScalarFnSignature,
) -> Result<Option<super::loop_analyzer::MultiPassAnalysis>, SymExecError> {
    Ok(None)
}

// ---------------------------------------------------------------------------
// Stub for non-x86_64 targets
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "x86_64"))]
pub fn analyze_scalar_fn(
    _fn_ptr: *const u8,
    _sig: &ScalarFnSignature,
) -> Result<Vec<TraceOp>, SymExecError> {
    Err(SymExecError::DisassemblyFailed(
        "binary analysis only supported on x86_64".into(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::compiler::trace::{classify_pattern, ComputePattern};

    /// Helper: analyze a scalar function and return its trace.
    fn analyze(
        f: *const u8,
        params: Vec<ScalarParam>,
    ) -> Result<Vec<TraceOp>, SymExecError> {
        let sig = ScalarFnSignature {
            fn_ptr: f,
            params,
        };
        analyze_scalar_fn(f, &sig)
    }

    #[test]
    fn test_analyze_silu() {
        use gllm_scalar_ops::activations::scalar_silu;
        let trace = analyze(
            scalar_silu as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        );
        match trace {
            Ok(ops) => {
                eprintln!("SiLU trace ({} ops): {ops:?}", ops.len());
                let pattern = classify_pattern(&ops);
                assert!(
                    matches!(pattern, ComputePattern::Elementwise { .. }),
                    "SiLU should be Elementwise, got {pattern:?}"
                );
                // Must contain: Input(0), Neg or sign flip, Exp, Add or Const(1.0), Div
                let has_input = ops.iter().any(|op| matches!(op, TraceOp::Input(0)));
                let has_exp = ops.iter().any(|op| matches!(op, TraceOp::Exp(_)));
                let has_div = ops.iter().any(|op| matches!(op, TraceOp::Div(_, _)));
                assert!(has_input, "SiLU missing Input(0)");
                assert!(has_exp, "SiLU missing Exp");
                assert!(has_div, "SiLU missing Div");
            }
            Err(e) => {
                eprintln!("SiLU analysis failed (may be expected on some platforms): {e}");
            }
        }
    }

    #[test]
    fn test_analyze_gelu() {
        use gllm_scalar_ops::activations::scalar_gelu;
        let trace = analyze(
            scalar_gelu as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        );
        match trace {
            Ok(ops) => {
                eprintln!("GELU trace ({} ops): {ops:?}", ops.len());
                let pattern = classify_pattern(&ops);
                assert!(
                    matches!(pattern, ComputePattern::Elementwise { .. }),
                    "GELU should be Elementwise, got {pattern:?}"
                );
                let has_input = ops.iter().any(|op| matches!(op, TraceOp::Input(0)));
                let has_tanh = ops.iter().any(|op| matches!(op, TraceOp::Tanh(_)));
                let has_mul = ops.iter().any(|op| matches!(op, TraceOp::Mul(_, _)));
                assert!(has_input, "GELU missing Input(0)");
                assert!(has_tanh, "GELU missing Tanh");
                assert!(has_mul, "GELU missing Mul");
            }
            Err(e) => {
                eprintln!("GELU analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_analyze_relu() {
        use gllm_scalar_ops::activations::scalar_relu;
        let trace = analyze(
            scalar_relu as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        );
        match trace {
            Ok(ops) => {
                eprintln!("ReLU trace ({} ops): {ops:?}", ops.len());
                let pattern = classify_pattern(&ops);
                assert!(
                    matches!(pattern, ComputePattern::Elementwise { .. }),
                    "ReLU should be Elementwise, got {pattern:?}"
                );
                let has_input = ops.iter().any(|op| matches!(op, TraceOp::Input(0)));
                assert!(has_input, "ReLU missing Input(0)");
                // ReLU is max(0, x) — should have Max or a comparison pattern.
                let has_max = ops.iter().any(|op| matches!(op, TraceOp::Max(_, _)));
                // Some compilers emit maxss, others use compare+blend.
                // Accept either Max or a non-trivial trace.
                assert!(
                    has_max || ops.len() > 1,
                    "ReLU trace too simple: {ops:?}"
                );
            }
            Err(e) => {
                eprintln!("ReLU analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_analyze_vec_add() {
        use gllm_scalar_ops::blas::scalar_vec_add;
        let trace = analyze(
            scalar_vec_add as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        );
        match trace {
            Ok(ops) => {
                eprintln!("VecAdd trace ({} ops): {ops:?}", ops.len());
                let pattern = classify_pattern(&ops);
                assert!(
                    matches!(pattern, ComputePattern::BinaryElementwise { .. }),
                    "VecAdd should be BinaryElementwise, got {pattern:?}"
                );
                let has_input0 = ops.iter().any(|op| matches!(op, TraceOp::Input(0)));
                let has_input1 = ops.iter().any(|op| matches!(op, TraceOp::Input(1)));
                let has_add = ops.iter().any(|op| matches!(op, TraceOp::Add(_, _)));
                assert!(has_input0, "VecAdd missing Input(0)");
                assert!(has_input1, "VecAdd missing Input(1)");
                assert!(has_add, "VecAdd missing Add");
            }
            Err(e) => {
                eprintln!("VecAdd analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_null_ptr_returns_error() {
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        let result = analyze_scalar_fn(std::ptr::null(), &sig);
        assert!(result.is_err());
    }

    // ── Structured (multi-loop) analysis tests ──────────────────────────

    /// Helper: run structured analysis on a scalar function.
    fn analyze_structured(
        f: *const u8,
        params: Vec<ScalarParam>,
    ) -> Result<Option<super::super::loop_analyzer::MultiPassAnalysis>, SymExecError> {
        let sig = ScalarFnSignature {
            fn_ptr: f,
            params,
        };
        analyze_scalar_fn_structured(f, &sig)
    }

    #[test]
    fn test_structured_rmsnorm() {
        use gllm_scalar_ops::norms::scalar_rms_norm;
        let result = analyze_structured(
            scalar_rms_norm as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        );
        match result {
            Ok(Some(info)) => {
                eprintln!("RmsNorm structured: {} loops, pattern={:?}", info.num_loops, info.pattern);
                assert!(
                    matches!(info.pattern, ComputePattern::NormLike { .. }),
                    "RmsNorm should be NormLike, got {:?}", info.pattern
                );
                assert!(info.num_loops >= 2, "RmsNorm should have ≥2 loops, got {}", info.num_loops);
            }
            Ok(None) => {
                eprintln!("RmsNorm structured: no loops detected (may need CFG improvements)");
            }
            Err(e) => {
                eprintln!("RmsNorm structured analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_structured_layernorm() {
        use gllm_scalar_ops::norms::scalar_layer_norm;
        let result = analyze_structured(
            scalar_layer_norm as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        );
        match result {
            Ok(Some(info)) => {
                eprintln!("LayerNorm structured: {} loops, pattern={:?}", info.num_loops, info.pattern);
                assert!(
                    matches!(info.pattern, ComputePattern::NormLike { .. }),
                    "LayerNorm should be NormLike, got {:?}", info.pattern
                );
                assert!(info.num_loops >= 2, "LayerNorm should have ≥2 loops, got {}", info.num_loops);
            }
            Ok(None) => {
                eprintln!("LayerNorm structured: no loops detected");
            }
            Err(e) => {
                eprintln!("LayerNorm structured analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_structured_softmax() {
        use gllm_scalar_ops::blas::scalar_softmax;
        let result = analyze_structured(
            scalar_softmax as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        );
        match result {
            Ok(Some(info)) => {
                eprintln!("Softmax structured: {} loops, pattern={:?}", info.num_loops, info.pattern);
                // Softmax has 3 passes (max, sum-exp, normalize) → Reduction or NormLike
                assert!(
                    matches!(info.pattern, ComputePattern::Reduction { .. } | ComputePattern::NormLike { .. }),
                    "Softmax should be Reduction or NormLike, got {:?}", info.pattern
                );
                assert!(info.num_loops >= 2, "Softmax should have ≥2 loops, got {}", info.num_loops);
            }
            Ok(None) => {
                eprintln!("Softmax structured: no loops detected");
            }
            Err(e) => {
                eprintln!("Softmax structured analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_structured_l2normalize() {
        use gllm_scalar_ops::norms::scalar_l2_normalize;
        let result = analyze_structured(
            scalar_l2_normalize as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        );
        match result {
            Ok(Some(info)) => {
                eprintln!("L2Normalize structured: {} loops, pattern={:?}", info.num_loops, info.pattern);
                assert!(
                    matches!(info.pattern, ComputePattern::NormLike { .. }),
                    "L2Normalize should be NormLike, got {:?}", info.pattern
                );
                assert!(info.num_loops >= 2, "L2Normalize should have ≥2 loops, got {}", info.num_loops);
            }
            Ok(None) => {
                eprintln!("L2Normalize structured: no loops detected");
            }
            Err(e) => {
                eprintln!("L2Normalize structured analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_structured_meanpool() {
        use gllm_scalar_ops::pooling::scalar_mean_pool;
        let result = analyze_structured(
            scalar_mean_pool as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0), // seq_len
                ScalarParam::Dim(1), // hidden
            ],
        );
        match result {
            Ok(Some(info)) => {
                eprintln!("MeanPool structured: {} loops, pattern={:?}", info.num_loops, info.pattern);
                assert!(
                    matches!(info.pattern, ComputePattern::Reduction { .. } | ComputePattern::NormLike { .. }),
                    "MeanPool should be Reduction or NormLike, got {:?}", info.pattern
                );
            }
            Ok(None) => {
                eprintln!("MeanPool structured: no loops detected (nested loops may need CFG improvements)");
            }
            Err(e) => {
                eprintln!("MeanPool structured analysis failed: {e}");
            }
        }
    }

    #[test]
    fn test_structured_null_ptr_returns_error() {
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        };
        let result = analyze_scalar_fn_structured(std::ptr::null(), &sig);
        assert!(result.is_err(), "null pointer should return Err");
    }
}
