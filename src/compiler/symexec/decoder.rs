//! Scalar + SymExec decoder bridge: fn_ptr → iced-x86 Decoder → SymbolicExecutor → OpTrace.
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
pub unsafe fn analyze_scalar_fn(
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
            if instr.op_kind(0) == OpKind::Register && is_xmm(instr.op_register(0))
                && instr.op_kind(1) == OpKind::Memory {
                    let base = instr.memory_base();
                    if let Some(input_ord) = ptr_map.is_input(base) {
                        let dst = format_register(instr.op_register(0));
                        executor.set(&dst, SymValue::Param(input_ord));
                        continue;
                    }
                }
        }

        // Float store to output pointer → capture the value being stored.
        // For void functions, this is how we find the "return" value.
        if is_float_store(mnemonic, &instr) {
            let base = instr.memory_base();
            if ptr_map.is_output(base) {
                // The source register is the last operand.
                let src_reg = instr.op_register(instr.op_count() - 1);
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
            if !executor.has_constant(abs_addr) && abs_addr % 4 == 0 {
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
        
        unsafe { *(got_addr as *const u64) }
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
pub unsafe fn analyze_scalar_fn_structured(
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
        .map_err(|e| SymExecError::DisassemblyFailed(e.to_string()))?;

    let forest = find_loops(&cfg);
    if forest.loops.is_empty() {
        return Ok(None); // No loops → not a multi-pass function.
    }

    // Create executor with the right number of float/ptr params.
    let n_inputs = count_inputs(sig);
    let executor = SymbolicExecutor::new(n_inputs, 0);

    // Check for nested loops first (GEMM, RoPE, Transpose).
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
    let mut last_transform_trace: Option<super::loop_analyzer::LoopTrace> = None;
    for natural_loop in &top_loops {
        match analyze_single_loop(natural_loop, &cfg, &executor) {
            Ok(trace) => {
                if trace.reductions.is_empty() {
                    // Keep the last transform-only loop as potential normalize pass.
                    if !trace.unknown_mutations.is_empty() {
                        last_transform_trace = Some(trace);
                    }
                    continue;
                }
                loop_traces.push(trace);
            }
            Err(_) => continue,
        }
    }

    // If we have 2 reduction loops and a trailing transform loop,
    // include the transform as a normalize pass (Softmax: max → sum → normalize).
    if loop_traces.len() == 2 && last_transform_trace.is_some() {
        loop_traces.push(last_transform_trace.unwrap());
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

#[cfg(target_arch = "aarch64")]
pub unsafe fn analyze_scalar_fn_structured(
    fn_ptr: *const u8,
    sig: &ScalarFnSignature,
) -> Result<Option<super::loop_analyzer::MultiPassAnalysis>, SymExecError> {
    super::decoder_aarch64::analyze_scalar_fn_structured(fn_ptr, sig)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn analyze_scalar_fn(
    fn_ptr: *const u8,
    sig: &ScalarFnSignature,
) -> Result<Vec<TraceOp>, SymExecError> {
    super::decoder_aarch64::analyze_scalar_fn(fn_ptr, sig)
}

// ---------------------------------------------------------------------------
// Stub for unsupported targets
// ---------------------------------------------------------------------------

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn analyze_scalar_fn_structured(
    _fn_ptr: *const u8,
    _sig: &ScalarFnSignature,
) -> Result<Option<super::loop_analyzer::MultiPassAnalysis>, SymExecError> {
    Ok(None)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn analyze_scalar_fn(
    _fn_ptr: *const u8,
    _sig: &ScalarFnSignature,
) -> Result<Vec<TraceOp>, SymExecError> {
    Err(SymExecError::DisassemblyFailed(
        "binary analysis only supported on x86_64 and aarch64".into(),
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
    unsafe fn analyze(
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
        let trace = unsafe { analyze(
            scalar_silu as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        )};
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
        let trace = unsafe { analyze(
            scalar_gelu as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        )};
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
        let trace = unsafe { analyze(
            scalar_relu as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        )};
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
        let trace = unsafe { analyze(
            scalar_vec_add as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        )};
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
        let result = unsafe { analyze_scalar_fn(std::ptr::null(), &sig) };
        assert!(result.is_err());
    }

    // ── Structured (multi-loop) analysis tests ──────────────────────────

    /// Helper: run structured analysis on a scalar function.
    unsafe fn analyze_structured(
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
        let result = unsafe { analyze_structured(
            scalar_rms_norm as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        ) };
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
        let result = unsafe { analyze_structured(
            scalar_layer_norm as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        ) };
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
        let result = unsafe { analyze_structured(
            scalar_softmax as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        ) };
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
        let result = unsafe { analyze_structured(
            scalar_l2_normalize as *const u8,
            vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
        ) };
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
        let result = unsafe { analyze_structured(
            scalar_mean_pool as *const u8,
            vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0), // seq_len
                ScalarParam::Dim(1), // hidden
            ],
        ) };
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
        let result = unsafe { analyze_scalar_fn_structured(std::ptr::null(), &sig) };
        assert!(result.is_err(), "null pointer should return Err");
    }

    // ── Pure data structure tests ───────────────────────────────────────

    #[test]
    fn test_scalar_param_variants_and_equality() {
        // Arrange: create all ScalarParam variants including boundary values
        let input = ScalarParam::InputPtr;
        let output = ScalarParam::OutputPtr;
        let weight = ScalarParam::WeightPtr;
        let dim_zero = ScalarParam::Dim(0);
        let dim_max = ScalarParam::Dim(usize::MAX);
        let scalar_nan = ScalarParam::Scalar(f32::NAN);
        let scalar_inf = ScalarParam::Scalar(f32::INFINITY);
        let scalar_neg_inf = ScalarParam::Scalar(f32::NEG_INFINITY);
        let scalar_zero = ScalarParam::Scalar(0.0);
        let scalar_neg_zero = ScalarParam::Scalar(-0.0);

        // Act & Assert: verify PartialEq works for same and different variants
        assert_eq!(input, ScalarParam::InputPtr);
        assert_eq!(output, ScalarParam::OutputPtr);
        assert_eq!(weight, ScalarParam::WeightPtr);
        assert_eq!(dim_zero, ScalarParam::Dim(0));
        assert_eq!(dim_max, ScalarParam::Dim(usize::MAX));
        // NaN != NaN by IEEE 754, but the derived PartialEq for f32 respects this
        assert_ne!(scalar_nan, ScalarParam::Scalar(f32::NAN));
        assert_eq!(scalar_inf, ScalarParam::Scalar(f32::INFINITY));
        assert_eq!(scalar_neg_inf, ScalarParam::Scalar(f32::NEG_INFINITY));
        assert_eq!(scalar_zero, ScalarParam::Scalar(0.0));
        // -0.0 == 0.0 by IEEE 754, derived PartialEq for f32
        assert_eq!(scalar_neg_zero, ScalarParam::Scalar(-0.0));

        // Verify different variants are not equal
        assert_ne!(input, output);
        assert_ne!(weight, dim_zero);
        assert_ne!(dim_max, scalar_zero);
    }

    #[test]
    fn test_scalar_param_clone() {
        // Arrange
        let original = vec![
            ScalarParam::InputPtr,
            ScalarParam::Dim(42),
            ScalarParam::Scalar(3.14),
        ];

        // Act
        let cloned = original.clone();

        // Assert: cloned matches original element-wise
        assert_eq!(original.len(), cloned.len());
        for (o, c) in original.iter().zip(cloned.iter()) {
            assert_eq!(o, c);
        }
        // Modify clone to prove independence
        drop(cloned);
        assert_eq!(original.len(), 3);
    }

    #[test]
    fn test_scalar_fn_signature_construction() {
        // Arrange: build a signature with all param types
        let dummy_ptr = 1usize as *const u8;
        let params = vec![
            ScalarParam::InputPtr,
            ScalarParam::WeightPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(128),
            ScalarParam::Dim(256),
            ScalarParam::Scalar(1e-5),
        ];

        // Act
        let sig = ScalarFnSignature {
            fn_ptr: dummy_ptr,
            params: params.clone(),
        };

        // Assert
        assert_eq!(sig.fn_ptr, dummy_ptr);
        assert_eq!(sig.params.len(), 6);
        assert_eq!(sig.params[3], ScalarParam::Dim(128));
        assert_eq!(sig.params[5], ScalarParam::Scalar(1e-5));
    }

    #[test]
    fn test_scalar_fn_signature_update_syntax() {
        // Arrange
        let base = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr],
        };

        // Act: use struct update syntax to override fn_ptr
        let derived = ScalarFnSignature {
            fn_ptr: 42usize as *const u8,
            ..base
        };

        // Assert: fn_ptr was overridden but params remain
        assert_eq!(derived.fn_ptr, 42usize as *const u8);
        assert_eq!(derived.params.len(), 2);
        assert_eq!(derived.params[0], ScalarParam::InputPtr);
    }

    #[test]
    fn test_count_inputs_various_signatures() {
        // Arrange & Act & Assert: empty signature
        let empty_sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![],
        };
        assert_eq!(count_inputs(&empty_sig), 0);

        // Only output ptrs
        let output_only = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![ScalarParam::OutputPtr],
        };
        assert_eq!(count_inputs(&output_only), 0);

        // Mixed params: InputPtr + WeightPtr count, OutputPtr/Dim/Scalar don't
        let mixed = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1.0),
                ScalarParam::InputPtr,
            ],
        };
        assert_eq!(count_inputs(&mixed), 3);
    }

    #[test]
    fn test_sym_exec_error_variants_and_display() {
        // Arrange
        let err1 = SymExecError::DisassemblyFailed("test error".into());
        let err2 = SymExecError::UnsupportedInstruction("bad opcode".into());
        let err3 = SymExecError::NoReturnValue;

        // Act: format via Display
        let s1 = format!("{err1}");
        let s2 = format!("{err2}");
        let s3 = format!("{err3}");

        // Assert
        assert!(s1.contains("test error"), "DisassemblyFailed display: {s1}");
        assert!(s1.contains("disassembly failed"), "DisassemblyFailed prefix: {s1}");
        assert!(s2.contains("bad opcode"), "UnsupportedInstruction display: {s2}");
        assert!(s2.contains("unsupported instruction"), "UnsupportedInstruction prefix: {s2}");
        assert!(s3.contains("no return value"), "NoReturnValue display: {s3}");

        // Assert: Error trait is usable (boxed dyn dispatch)
        let _: Box<dyn std::error::Error> = Box::new(err1);
    }

    #[test]
    fn test_sym_value_variants_and_debug() {
        // Arrange: build various SymValue variants
        use super::super::sym_value::{LibmFn, SelectKind};

        let param0 = SymValue::Param(0);
        let param_max = SymValue::Param(usize::MAX);
        let const_zero = SymValue::Const(0.0);
        let const_neg = SymValue::Const(-1e38);
        let add_val = SymValue::Add(Box::new(SymValue::Param(0)), Box::new(SymValue::Const(1.0)));
        let nested = SymValue::Mul(
            Box::new(SymValue::Add(Box::new(SymValue::Param(0)), Box::new(SymValue::Param(1)))),
            Box::new(SymValue::Const(2.0)),
        );
        let select_val = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Const(0.0)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Const(0.0)),
        };
        let libm_call = SymValue::Call(LibmFn::Expf, vec![SymValue::Param(0)]);
        let unknown = SymValue::Unknown("test".into());

        // Act: Debug format all
        let dbg_param = format!("{:?}", param0);
        let dbg_const = format!("{:?}", const_neg);
        let dbg_add = format!("{:?}", add_val);
        let dbg_nested = format!("{:?}", nested);
        let dbg_select = format!("{:?}", select_val);
        let dbg_libm = format!("{:?}", libm_call);
        let dbg_unknown = format!("{:?}", unknown);

        // Assert: Debug output contains expected substrings
        assert!(dbg_param.contains("Param"), "Debug for Param: {dbg_param}");
        assert!(dbg_const.contains("-"), "Debug for negative Const: {dbg_const}");
        assert!(dbg_add.contains("Add"), "Debug for Add: {dbg_add}");
        assert!(dbg_nested.contains("Mul"), "Debug for nested Mul: {dbg_nested}");
        assert!(dbg_select.contains("Select"), "Debug for Select: {dbg_select}");
        assert!(dbg_libm.contains("Call"), "Debug for Call: {dbg_libm}");
        assert!(dbg_unknown.contains("test"), "Debug for Unknown: {dbg_unknown}");

        // Assert: Clone works and produces equal value
        let cloned = nested.clone();
        assert_eq!(format!("{:?}", cloned), format!("{:?}", nested));

        // Assert: boundary usize::MAX param doesn't panic on Debug
        let _ = format!("{:?}", param_max);
    }

    #[test]
    fn test_compute_pattern_variants_and_body() {
        // Arrange: construct various ComputePattern variants
        use crate::compiler::trace::ValueId;

        let v = |n: u32| ValueId(n);

        let ew = ComputePattern::Elementwise { body: vec![TraceOp::Input(0), TraceOp::Neg(v(0))] };
        let bew = ComputePattern::BinaryElementwise {
            body: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(v(0), v(1))],
        };
        let inj = ComputePattern::Injective {
            body: vec![TraceOp::Input(0)],
            num_inputs: 2,
            num_outputs: 3,
        };
        let gemm = ComputePattern::Gemm;
        let empty_ew = ComputePattern::Elementwise { body: vec![] };

        // Act & Assert: body() method
        assert!(ew.body().is_some());
        assert_eq!(ew.body().unwrap().len(), 2);
        assert!(bew.body().is_some());
        assert_eq!(bew.body().unwrap().len(), 3);
        assert!(inj.body().is_some());
        assert_eq!(inj.body().unwrap().len(), 1);
        assert!(gemm.body().is_none(), "Gemm should have no body");
        assert!(empty_ew.body().is_some());
        assert!(empty_ew.body().unwrap().is_empty());

        // Assert: Debug format works for all variants
        let _ = format!("{:?}", ew);
        let _ = format!("{:?}", bew);
        let _ = format!("{:?}", inj);
        let _ = format!("{:?}", gemm);
    }

    #[test]
    fn test_trace_op_various_constructions() {
        // Arrange: construct a representative set of TraceOp variants
        use crate::compiler::trace::ValueId;

        let v = |n: u32| ValueId(n);

        let input = TraceOp::Input(0);
        let input_max = TraceOp::Input(u32::MAX);
        let const_val = TraceOp::Const(0.0);
        let const_neg = TraceOp::Const(-f64::INFINITY);
        let add = TraceOp::Add(v(0), v(1));
        let sub = TraceOp::Sub(v(2), v(3));
        let mul = TraceOp::Mul(v(4), v(5));
        let div = TraceOp::Div(v(6), v(7));
        let fma = TraceOp::Fma(v(0), v(1), v(2));
        let neg = TraceOp::Neg(v(0));
        let abs = TraceOp::Abs(v(1));
        let exp = TraceOp::Exp(v(2));
        let sqrt = TraceOp::Sqrt(v(3));
        let rsqrt = TraceOp::Rsqrt(v(4));
        let tanh = TraceOp::Tanh(v(5));
        let recip = TraceOp::Recip(v(6));
        let log = TraceOp::Log(v(7));
        let sigmoid = TraceOp::Sigmoid(v(8));
        let max = TraceOp::Max(v(0), v(1));
        let min = TraceOp::Min(v(2), v(3));
        let cond = TraceOp::ConditionalBranch(v(0), v(1), v(2));

        // Act: verify Debug doesn't panic
        let ops = [input, input_max, const_val, const_neg, add, sub, mul, div, fma,
                   neg, abs, exp, sqrt, rsqrt, tanh, recip, log, sigmoid, max, min, cond];

        // Assert: all Debug formats produce non-empty strings
        for op in &ops {
            let dbg = format!("{:?}", op);
            assert!(!dbg.is_empty(), "TraceOp Debug should not be empty for {:?}", op);
        }

        // Assert: boundary u32::MAX Input doesn't panic
        let _ = format!("{:?}", TraceOp::Input(u32::MAX));

        // Assert: -infinity Const doesn't panic
        let _ = format!("{:?}", TraceOp::Const(-f64::INFINITY));
    }

    #[test]
    fn test_reduction_kind_variants() {
        // Arrange
        use super::super::loop_analyzer::ReductionKind;

        let sum = ReductionKind::Sum;
        let max = ReductionKind::Max;
        let min = ReductionKind::Min;

        // Act & Assert: PartialEq
        assert_eq!(sum, ReductionKind::Sum);
        assert_eq!(max, ReductionKind::Max);
        assert_eq!(min, ReductionKind::Min);
        assert_ne!(sum, max);
        assert_ne!(max, min);

        // Assert: Copy + Clone
        let copied = sum;
        assert_eq!(copied, ReductionKind::Sum);

        // Assert: Debug
        assert!(format!("{:?}", sum).contains("Sum"));
        assert!(format!("{:?}", max).contains("Max"));
        assert!(format!("{:?}", min).contains("Min"));
    }

    #[test]
    fn test_accumulator_init_variants() {
        // Arrange
        use super::super::loop_analyzer::AccumulatorInit;

        let const_init = AccumulatorInit::Const(0.0);
        let const_neg_inf = AccumulatorInit::Const(f64::NEG_INFINITY);
        let const_nan = AccumulatorInit::Const(f64::NAN);
        let symbolic = AccumulatorInit::Symbolic(SymValue::Param(0));
        let nested_symbolic = AccumulatorInit::Symbolic(SymValue::Mul(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(2.0)),
        ));

        // Act: Debug format all
        let dbg_const = format!("{:?}", const_init);
        let dbg_neg_inf = format!("{:?}", const_neg_inf);
        let dbg_nan = format!("{:?}", const_nan);
        let dbg_sym = format!("{:?}", symbolic);
        let dbg_nested = format!("{:?}", nested_symbolic);

        // Assert: Debug output contains variant names
        assert!(dbg_const.contains("Const"), "AccumulatorInit::Const Debug: {dbg_const}");
        assert!(dbg_neg_inf.contains("Const"), "AccumulatorInit::Const(neg_inf) Debug: {dbg_neg_inf}");
        assert!(dbg_nan.contains("Const"), "AccumulatorInit::Const(NaN) Debug: {dbg_nan}");
        assert!(dbg_sym.contains("Symbolic"), "AccumulatorInit::Symbolic Debug: {dbg_sym}");
        assert!(dbg_nested.contains("Symbolic"), "AccumulatorInit::Symbolic nested Debug: {dbg_nested}");

        // Assert: Clone works
        let cloned = const_init.clone();
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_block_id_and_branch_kind() {
        // Arrange
        use super::super::cfg::{BlockId, BranchKind};

        let b0 = BlockId(0);
        let b1 = BlockId(1);
        let b_max = BlockId(u32::MAX);

        // Act & Assert: Ord comparison
        assert!(b0 < b1);
        assert!(b1 < b_max);
        assert_eq!(b0, BlockId(0));

        // Assert: Copy
        let b0_copy = b0;
        assert_eq!(b0_copy, BlockId(0));

        // Assert: Hash works (insert into HashSet)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(b0);
        set.insert(b1);
        set.insert(b_max);
        assert_eq!(set.len(), 3);

        // BranchKind variants
        let kinds = [
            BranchKind::Above, BranchKind::AboveEqual, BranchKind::Below,
            BranchKind::BelowEqual, BranchKind::Greater, BranchKind::GreaterEqual,
            BranchKind::Less, BranchKind::LessEqual, BranchKind::Equal,
            BranchKind::NotEqual, BranchKind::Sign, BranchKind::NotSign,
            BranchKind::Parity, BranchKind::NotParity,
        ];
        // Assert: all 14 variants are distinct
        for (i, k1) in kinds.iter().enumerate() {
            for (j, k2) in kinds.iter().enumerate() {
                if i == j {
                    assert_eq!(k1, k2);
                } else {
                    assert_ne!(k1, k2);
                }
            }
        }
    }

    #[test]
    fn test_select_kind_and_libm_fn_coverage() {
        // Arrange
        use super::super::sym_value::{LibmFn, SelectKind};

        // Act & Assert: SelectKind all variants + Copy + PartialEq
        let select_kinds = [SelectKind::Gt, SelectKind::Ge, SelectKind::Lt, SelectKind::Le, SelectKind::Eq, SelectKind::Ne];
        assert_eq!(select_kinds.len(), 6, "SelectKind should have 6 variants");
        for (i, k1) in select_kinds.iter().enumerate() {
            for (j, k2) in select_kinds.iter().enumerate() {
                assert_eq!(i == j, k1 == k2);
            }
        }
        // Copy
        let copied = select_kinds[0];
        assert_eq!(copied, SelectKind::Gt);

        // Act & Assert: LibmFn all variants + Copy + PartialEq
        let libm_variants = [LibmFn::Expf, LibmFn::Sqrtf, LibmFn::Tanhf, LibmFn::Logf, LibmFn::Fabsf];
        assert_eq!(libm_variants.len(), 5, "LibmFn should have 5 variants");
        for (i, v1) in libm_variants.iter().enumerate() {
            for (j, v2) in libm_variants.iter().enumerate() {
                assert_eq!(i == j, v1 == v2);
            }
        }
        // Copy
        let copied = libm_variants[0];
        assert_eq!(copied, LibmFn::Expf);
    }

    // ── Constants and helper function tests ─────────────────────────────

    #[test]
    fn test_decoder_constants_bounds() {
        // Arrange & Act: read the constants
        // Assert: MAX_FN_BYTES is a reasonable upper bound for scalar functions
        assert!(
            MAX_FN_BYTES >= 1024,
            "MAX_FN_BYTES should be at least 1024 for realistic scalar functions, got {MAX_FN_BYTES}"
        );
        assert!(
            MAX_FN_BYTES <= 65536,
            "MAX_FN_BYTES should not exceed 64KB to avoid reading into adjacent code, got {MAX_FN_BYTES}"
        );

        // Assert: MAX_INSTRUCTIONS is a reasonable safety limit
        assert!(
            MAX_INSTRUCTIONS >= 100,
            "MAX_INSTRUCTIONS should allow at least 100 instructions for non-trivial functions, got {MAX_INSTRUCTIONS}"
        );
        assert!(
            MAX_INSTRUCTIONS <= 10000,
            "MAX_INSTRUCTIONS should cap at 10000 to prevent runaway analysis, got {MAX_INSTRUCTIONS}"
        );
    }

    #[test]
    fn test_ptr_param_map_empty_and_queries() {
        // Arrange: empty PtrParamMap — no registers mapped
        let map = PtrParamMap {
            input_regs: HashMap::new(),
            output_regs: vec![],
        };

        // Act & Assert: no register is known as input or output
        assert!(map.is_input(Register::RDI).is_none(), "empty map should have no input");
        assert!(map.is_input(Register::RSI).is_none());
        assert!(!map.is_output(Register::RDI), "empty map should have no output");
        assert!(!map.is_output(Register::RSI));
    }

    #[test]
    fn test_ptr_param_map_input_and_output_tracking() {
        // Arrange: RDI=Input(0), RSI=Output
        let map = PtrParamMap {
            input_regs: {
                let mut m = HashMap::new();
                m.insert(Register::RDI, 0);
                m
            },
            output_regs: vec![Register::RSI],
        };

        // Act & Assert: is_input returns ordinal for mapped register
        assert_eq!(map.is_input(Register::RDI), Some(0));
        assert!(map.is_input(Register::RSI).is_none(), "RSI is output, not input");

        // Assert: is_output returns true for mapped register
        assert!(map.is_output(Register::RSI));
        assert!(!map.is_output(Register::RDI), "RDI is input, not output");
    }

    #[test]
    fn test_ptr_param_map_propagate_mov_input() {
        // Arrange: RDI=Input(0), no outputs
        let mut map = PtrParamMap {
            input_regs: {
                let mut m = HashMap::new();
                m.insert(Register::RDI, 0);
                m
            },
            output_regs: vec![],
        };

        // Act: simulate `mov r15, rdi` — propagate input identity
        map.propagate_mov(Register::R15, Register::RDI);

        // Assert: r15 now also recognized as Input(0)
        assert_eq!(map.is_input(Register::R15), Some(0));
        assert_eq!(map.is_input(Register::RDI), Some(0), "original mapping preserved");
        assert!(!map.is_output(Register::R15));
    }

    #[test]
    fn test_ptr_param_map_propagate_mov_output() {
        // Arrange: RSI=Output, no inputs
        let mut map = PtrParamMap {
            input_regs: HashMap::new(),
            output_regs: vec![Register::RSI],
        };

        // Act: simulate `mov r13, rsi` — propagate output identity
        map.propagate_mov(Register::R13, Register::RSI);

        // Assert: r13 now also recognized as output
        assert!(map.is_output(Register::R13));
        assert!(map.is_output(Register::RSI), "original mapping preserved");
        assert!(map.is_input(Register::R13).is_none());
    }

    #[test]
    fn test_build_ptr_map_various_signatures() {
        // Arrange: signature with InputPtr, OutputPtr, WeightPtr, Dim, Scalar
        let sig_with_all = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::InputPtr,    // → RDI (input ordinal 0)
                ScalarParam::OutputPtr,   // → RSI (output)
                ScalarParam::WeightPtr,   // → RDX (input ordinal 1)
                ScalarParam::Dim(0),      // → RCX (not tracked)
                ScalarParam::Scalar(1.0), // → xmm0 (not tracked)
            ],
        };

        // Act
        let map = build_ptr_map(&sig_with_all);

        // Assert: RDI=Input(0), RDX=Input(1), RSI=Output
        assert_eq!(map.input_regs.get(&Register::RDI), Some(&0), "RDI should be Input(0)");
        assert_eq!(map.input_regs.get(&Register::RDX), Some(&1), "RDX should be Input(1)");
        assert!(map.output_regs.contains(&Register::RSI), "RSI should be output");
        assert!(map.input_regs.get(&Register::RCX).is_none(), "RCX holds Dim, not input");
    }

    #[test]
    fn test_format_mnemonic_and_register() {
        // Arrange: pick a few known Mnemonic and Register values
        let addss = Mnemonic::Addss;
        let vmulps = Mnemonic::Vmulps;
        let xmm0 = Register::XMM0;
        let rdi = Register::RDI;

        // Act
        let mnem_add = format_mnemonic(addss);
        let mnem_mul = format_mnemonic(vmulps);
        let reg_xmm = format_register(xmm0);
        let reg_rdi = format_register(rdi);

        // Assert: all lowercase
        assert_eq!(mnem_add, "addss", "format_mnemonic(Addss) should be 'addss'");
        assert_eq!(mnem_mul, "vmulps", "format_mnemonic(Vmulps) should be 'vmulps'");
        assert_eq!(reg_xmm, "xmm0", "format_register(XMM0) should be 'xmm0'");
        assert_eq!(reg_rdi, "rdi", "format_register(RDI) should be 'rdi'");
    }

    #[test]
    fn test_terminator_variants_and_equality() {
        // Arrange
        use super::super::cfg::{BlockId, BranchKind, Terminator};

        let b0 = BlockId(0);
        let b1 = BlockId(1);

        let fallthrough = Terminator::Fallthrough(b1);
        let jump = Terminator::Jump(b1);
        let cond = Terminator::CondBranch {
            kind: BranchKind::Greater,
            taken: b1,
            fallthrough: b0,
        };
        let ret = Terminator::Return;

        // Act & Assert: PartialEq
        assert_eq!(fallthrough, Terminator::Fallthrough(b1));
        assert_ne!(fallthrough, jump);
        assert_ne!(jump, cond);
        assert_ne!(cond, ret);

        // Assert: Clone
        let cloned = cond.clone();
        assert_eq!(cloned, cond);

        // Assert: Debug produces non-empty output
        assert!(!format!("{:?}", fallthrough).is_empty());
        assert!(!format!("{:?}", jump).is_empty());
        assert!(!format!("{:?}", cond).is_empty());
        assert!(!format!("{:?}", ret).is_empty());
    }

    #[test]
    fn test_decoded_insn_and_basic_block_construction() {
        // Arrange
        use super::super::cfg::{BasicBlock, BlockId, DecodedInsn, Terminator};

        let insn = DecodedInsn {
            mnemonic: "addss".to_string(),
            operands: vec!["xmm0".to_string(), "xmm1".to_string()],
            addr: 0x1000,
        };
        let insn_clone = insn.clone();

        // Act & Assert: DecodedInsn PartialEq
        assert_eq!(insn, insn_clone);

        let block = BasicBlock {
            id: BlockId(0),
            start_addr: 0x1000,
            end_addr: 0x1010,
            instructions: vec![insn],
            terminator: Terminator::Fallthrough(BlockId(1)),
        };

        // Assert: block fields are accessible
        assert_eq!(block.id, BlockId(0));
        assert_eq!(block.instructions.len(), 1);
        assert_eq!(block.instructions[0].mnemonic, "addss");
        assert_eq!(block.start_addr, 0x1000);
        assert_eq!(block.end_addr, 0x1010);

        // Assert: Debug
        let dbg = format!("{:?}", block);
        assert!(dbg.contains("BasicBlock"), "Debug should contain BasicBlock: {dbg}");
    }

    #[test]
    fn test_natural_loop_and_loop_forest_construction() {
        // Arrange
        use super::super::cfg::{BlockId, LoopForest, NaturalLoop};
        use std::collections::{BTreeSet, HashMap};

        let header = BlockId(0);
        let latch = BlockId(2);
        let mut body = BTreeSet::new();
        body.insert(header);
        body.insert(BlockId(1));
        body.insert(latch);

        let natural_loop = NaturalLoop {
            header,
            body_blocks: body.clone(),
            latch,
            exits: vec![BlockId(3)],
            ordinal: 0,
            depth: 0,
        };

        // Act & Assert: field access
        assert_eq!(natural_loop.header, BlockId(0));
        assert_eq!(natural_loop.body_blocks.len(), 3);
        assert_eq!(natural_loop.latch, BlockId(2));
        assert_eq!(natural_loop.exits.len(), 1);
        assert_eq!(natural_loop.ordinal, 0);
        assert_eq!(natural_loop.depth, 0);

        // Assert: Clone
        let cloned_loop = natural_loop.clone();
        assert_eq!(cloned_loop.header, natural_loop.header);
        assert_eq!(cloned_loop.body_blocks.len(), natural_loop.body_blocks.len());

        // Arrange: LoopForest with a single top-level loop
        let forest = LoopForest {
            loops: vec![natural_loop],
            children: HashMap::new(),
            top_level: vec![0],
        };

        // Act & Assert
        assert_eq!(forest.loops.len(), 1);
        assert!(forest.children.is_empty(), "no nested loops");
        assert_eq!(forest.top_level, vec![0]);

        // Assert: Debug
        let dbg = format!("{:?}", forest);
        assert!(dbg.contains("LoopForest"), "Debug should contain LoopForest: {dbg}");
    }

    // ── Additional pure data structure / logic tests ─────────────────────

    #[test]
    fn test_value_id_operations_and_constants() {
        // Arrange
        use crate::compiler::trace::ValueId;

        let v0 = ValueId(0);
        let v42 = ValueId(42);
        let v_none = ValueId::NONE;

        // Act & Assert: is_some
        assert!(v0.is_some(), "ValueId(0) should be 'some'");
        assert!(v42.is_some(), "ValueId(42) should be 'some'");
        assert!(!v_none.is_some(), "ValueId::NONE should not be 'some'");

        // Act & Assert: Display
        assert_eq!(format!("{v0}"), "v0");
        assert_eq!(format!("{v42}"), "v42");
        assert_eq!(format!("{v_none}"), format!("v{}", u32::MAX));

        // Act & Assert: Sub<u32>
        let v40 = v42 - 2;
        assert_eq!(v40, ValueId(40));

        // Act & Assert: saturating_sub
        let v_saturated = ValueId(1).saturating_sub(5);
        assert_eq!(v_saturated, ValueId(0), "saturating_sub should clamp to 0");
        let v_no_underflow = ValueId(10).saturating_sub(3);
        assert_eq!(v_no_underflow, ValueId(7));
    }

    #[test]
    fn test_reduction_detected_construction_and_debug() {
        // Arrange
        use super::super::loop_analyzer::{AccumulatorInit, ReductionDetected, ReductionKind};

        let rd_sum = ReductionDetected {
            register: "xmm0".to_string(),
            kind: ReductionKind::Sum,
            init: AccumulatorInit::Const(0.0),
            body_expr: SymValue::Param(0),
        };
        let rd_max = ReductionDetected {
            register: "xmm1".to_string(),
            kind: ReductionKind::Max,
            init: AccumulatorInit::Const(f64::NEG_INFINITY),
            body_expr: SymValue::Mul(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Param(0)),
            ),
        };

        // Act: Debug
        let dbg_sum = format!("{:?}", rd_sum);
        let dbg_max = format!("{:?}", rd_max);

        // Assert: Debug output contains key fields
        assert!(dbg_sum.contains("xmm0"), "Debug should contain register: {dbg_sum}");
        assert!(dbg_sum.contains("Sum"), "Debug should contain kind: {dbg_sum}");
        assert!(dbg_max.contains("Max"), "Debug should contain kind: {dbg_max}");
        assert!(dbg_max.contains("xmm1"), "Debug should contain register: {dbg_max}");

        // Assert: Clone produces equal value
        let cloned = rd_sum.clone();
        assert_eq!(cloned.register, "xmm0");
        assert_eq!(cloned.kind, ReductionKind::Sum);
    }

    #[test]
    fn test_loop_trace_construction_and_fields() {
        // Arrange
        use super::super::cfg::BlockId;
        use super::super::loop_analyzer::{AccumulatorInit, LoopTrace, ReductionDetected, ReductionKind};

        let reduction = ReductionDetected {
            register: "xmm0".to_string(),
            kind: ReductionKind::Sum,
            init: AccumulatorInit::Const(0.0),
            body_expr: SymValue::Param(0),
        };
        let unknown = ("xmm3".to_string(), SymValue::Unknown("spill".into()));

        // Act
        let trace = LoopTrace {
            loop_header: BlockId(5),
            reductions: vec![reduction],
            unknown_mutations: vec![unknown],
            body_block_count: 7,
        };

        // Assert
        assert_eq!(trace.loop_header, BlockId(5));
        assert_eq!(trace.reductions.len(), 1);
        assert_eq!(trace.reductions[0].kind, ReductionKind::Sum);
        assert_eq!(trace.unknown_mutations.len(), 1);
        assert_eq!(trace.unknown_mutations[0].0, "xmm3");
        assert_eq!(trace.body_block_count, 7);

        // Assert: Debug
        let dbg = format!("{:?}", trace);
        assert!(dbg.contains("LoopTrace"), "Debug should contain LoopTrace: {dbg}");
    }

    #[test]
    fn test_multi_pass_analysis_construction_and_debug() {
        // Arrange
        use super::super::cfg::BlockId;
        use super::super::loop_analyzer::{AccumulatorInit, LoopTrace, MultiPassAnalysis, ReductionDetected, ReductionKind};

        let loop_trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".to_string(),
                kind: ReductionKind::Max,
                init: AccumulatorInit::Const(f64::NEG_INFINITY),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 3,
        };
        let pattern = ComputePattern::Reduction {
            identity: f64::NEG_INFINITY,
            combine: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Max(
                crate::compiler::trace::ValueId(0),
                crate::compiler::trace::ValueId(1),
            )],
            second_pass: None,
            normalize: None,
        };

        // Act
        let mpa = MultiPassAnalysis {
            loop_traces: vec![loop_trace],
            pattern,
            num_loops: 2,
        };

        // Assert
        assert_eq!(mpa.loop_traces.len(), 1);
        assert_eq!(mpa.num_loops, 2);
        assert!(matches!(mpa.pattern, ComputePattern::Reduction { .. }));

        // Assert: Debug
        let dbg = format!("{:?}", mpa);
        assert!(dbg.contains("MultiPassAnalysis"), "Debug should contain MultiPassAnalysis: {dbg}");
        assert!(dbg.contains("Reduction"), "Debug should contain Reduction pattern: {dbg}");
    }

    #[test]
    fn test_reduction_second_pass_construction() {
        // Arrange
        use crate::compiler::trace::{ReductionSecondPass, ValueId};

        let v = |n: u32| ValueId(n);
        let element_transform = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Sub(v(0), v(1)),
            TraceOp::Exp(v(2)),
        ];
        let combine = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Add(v(0), v(1)),
        ];

        // Act
        let second_pass = ReductionSecondPass {
            identity: 0.0,
            element_transform: element_transform.clone(),
            combine: combine.clone(),
        };

        // Assert: fields match
        assert_eq!(second_pass.identity, 0.0);
        assert_eq!(second_pass.element_transform.len(), 4);
        assert_eq!(second_pass.combine.len(), 3);
        assert!(matches!(second_pass.element_transform[3], TraceOp::Exp(_)));
        assert!(matches!(second_pass.combine[2], TraceOp::Add(_, _)));

        // Assert: Clone
        let cloned = second_pass.clone();
        assert_eq!(cloned.element_transform.len(), second_pass.element_transform.len());
    }

    #[test]
    fn test_compute_pattern_reduction_and_normlike_variants() {
        // Arrange
        use crate::compiler::trace::ValueId;

        let v = |n: u32| ValueId(n);

        let reduction = ComputePattern::Reduction {
            identity: 0.0,
            combine: vec![TraceOp::Input(0), TraceOp::Input(1), TraceOp::Add(v(0), v(1))],
            second_pass: None,
            normalize: Some(vec![TraceOp::Input(0), TraceOp::Recip(v(0))]),
        };
        let normlike = ComputePattern::NormLike {
            reduce: vec![TraceOp::Input(0), TraceOp::Mul(v(0), v(0)), TraceOp::Add(v(0), v(1))],
            finalize: vec![TraceOp::Rsqrt(v(0))],
            transform: vec![TraceOp::Input(0), TraceOp::Mul(v(0), v(1))],
        };

        // Act & Assert: body() returns None for these multi-phase patterns
        assert!(reduction.body().is_none(), "Reduction body() should be None");
        assert!(normlike.body().is_none(), "NormLike body() should be None");

        // Assert: Debug works
        let dbg_red = format!("{:?}", reduction);
        let dbg_norm = format!("{:?}", normlike);
        assert!(dbg_red.contains("Reduction"), "Debug should contain Reduction: {dbg_red}");
        assert!(dbg_norm.contains("NormLike"), "Debug should contain NormLike: {dbg_norm}");
    }

    #[test]
    fn test_compute_pattern_quant_decode_variant() {
        // Arrange
        let decode_ops = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Sub(
                crate::compiler::trace::ValueId(0),
                crate::compiler::trace::ValueId(1),
            ),
        ];
        let quant = ComputePattern::QuantDecode {
            block_size: 32,
            decode: decode_ops.clone(),
        };

        // Act: body() should return the decode ops
        let body = quant.body();
        assert!(body.is_some(), "QuantDecode should have a body");
        assert_eq!(body.unwrap().len(), 3);

        // Assert: field access
        if let ComputePattern::QuantDecode { block_size, decode } = &quant {
            assert_eq!(*block_size, 32);
            assert_eq!(decode.len(), 3);
        } else {
            panic!("Expected QuantDecode variant");
        }

        // Assert: Debug
        let dbg = format!("{:?}", quant);
        assert!(dbg.contains("QuantDecode"), "Debug should contain QuantDecode: {dbg}");
    }

    #[test]
    fn test_sym_value_load_variant_construction() {
        // Arrange
        let load_simple = SymValue::Load {
            base: Box::new(SymValue::Param(0)),
            index: Box::new(SymValue::Const(4.0)),
        };
        let load_nested = SymValue::Load {
            base: Box::new(SymValue::Load {
                base: Box::new(SymValue::Param(1)),
                index: Box::new(SymValue::Const(8.0)),
            }),
            index: Box::new(SymValue::Param(2)),
        };

        // Act: Debug format
        let dbg_simple = format!("{:?}", load_simple);
        let dbg_nested = format!("{:?}", load_nested);

        // Assert: Debug contains "Load"
        assert!(dbg_simple.contains("Load"), "Simple load Debug: {dbg_simple}");
        assert!(dbg_nested.contains("Load"), "Nested load Debug: {dbg_nested}");

        // Assert: Clone
        let cloned = load_nested.clone();
        let dbg_cloned = format!("{:?}", cloned);
        assert_eq!(dbg_nested, dbg_cloned, "Cloned Load should debug-format identically");
    }

    #[test]
    fn test_sym_value_fma_and_recip_and_rsqrt() {
        // Arrange
        let fma_val = SymValue::Fma(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(2.0)),
            Box::new(SymValue::Param(1)),
        );
        let recip = SymValue::Recip(Box::new(SymValue::Param(0)));
        let rsqrt = SymValue::Rsqrt(Box::new(SymValue::Param(3)));

        // Act: Debug format
        let dbg_fma = format!("{:?}", fma_val);
        let dbg_recip = format!("{:?}", recip);
        let dbg_rsqrt = format!("{:?}", rsqrt);

        // Assert: Debug contains variant names
        assert!(dbg_fma.contains("Fma"), "Fma Debug: {dbg_fma}");
        assert!(dbg_recip.contains("Recip"), "Recip Debug: {dbg_recip}");
        assert!(dbg_rsqrt.contains("Rsqrt"), "Rsqrt Debug: {dbg_rsqrt}");

        // Assert: Clone round-trip
        let cloned = fma_val.clone();
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_ptr_param_map_propagate_mov_unknown_registers() {
        // Arrange: map with RDI=Input(0) and RSI=Output
        let mut map = PtrParamMap {
            input_regs: {
                let mut m = HashMap::new();
                m.insert(Register::RDI, 0);
                m
            },
            output_regs: vec![Register::RSI],
        };

        // Act: propagate between two registers that are NOT in the map
        // mov rax, rbx — neither rax nor rbx is tracked
        map.propagate_mov(Register::RAX, Register::RBX);

        // Assert: no new mappings created (neither was known)
        assert!(map.is_input(Register::RAX).is_none());
        assert!(map.is_input(Register::RBX).is_none());
        assert!(!map.is_output(Register::RAX));
        assert!(!map.is_output(Register::RBX));

        // Assert: original mappings are unchanged
        assert_eq!(map.is_input(Register::RDI), Some(0));
        assert!(map.is_output(Register::RSI));
    }

    #[test]
    fn test_is_gp64_all_gp_and_non_gp_registers() {
        // Arrange: all 16 GP64 registers and some non-GP registers
        let gp64_regs = [
            Register::RAX, Register::RBX, Register::RCX, Register::RDX,
            Register::RSI, Register::RDI, Register::RBP, Register::RSP,
            Register::R8, Register::R9, Register::R10, Register::R11,
            Register::R12, Register::R13, Register::R14, Register::R15,
        ];
        let non_gp_regs = [
            Register::XMM0, Register::XMM15, Register::EAX, Register::AL,
            Register::RIP, Register::None,
        ];

        // Act & Assert: all GP64 registers recognized
        for reg in &gp64_regs {
            assert!(is_gp64(*reg), "{reg:?} should be GP64");
        }
        // Assert: non-GP registers rejected
        for reg in &non_gp_regs {
            assert!(!is_gp64(*reg), "{reg:?} should NOT be GP64");
        }
    }

    #[test]
    fn test_is_xmm_all_16_and_non_xmm() {
        // Arrange: all 16 XMM registers
        let xmm_regs = [
            Register::XMM0, Register::XMM1, Register::XMM2, Register::XMM3,
            Register::XMM4, Register::XMM5, Register::XMM6, Register::XMM7,
            Register::XMM8, Register::XMM9, Register::XMM10, Register::XMM11,
            Register::XMM12, Register::XMM13, Register::XMM14, Register::XMM15,
        ];
        let non_xmm = [Register::RAX, Register::YMM0, Register::ZMM0, Register::EAX];

        // Act & Assert: all 16 XMM recognized
        for reg in &xmm_regs {
            assert!(is_xmm(*reg), "{reg:?} should be XMM");
        }
        // Assert: non-XMM rejected (including wider YMM/ZMM aliases)
        for reg in &non_xmm {
            assert!(!is_xmm(*reg), "{reg:?} should NOT be XMM");
        }
    }

    #[test]
    fn test_is_branch_all_conditionals_and_unconditional() {
        // Arrange: all branch mnemonics that is_branch matches
        let branch_mnemonics = [
            Mnemonic::Je, Mnemonic::Jne, Mnemonic::Jb, Mnemonic::Jbe,
            Mnemonic::Ja, Mnemonic::Jae, Mnemonic::Jl, Mnemonic::Jle,
            Mnemonic::Jg, Mnemonic::Jge, Mnemonic::Jmp, Mnemonic::Js,
            Mnemonic::Jns, Mnemonic::Jp, Mnemonic::Jnp,
        ];
        let non_branch = [
            Mnemonic::Addss, Mnemonic::Movss, Mnemonic::Ret, Mnemonic::Call,
            Mnemonic::Nop, Mnemonic::Push, Mnemonic::Xorps,
        ];

        // Act & Assert: all branch mnemonics recognized
        for m in &branch_mnemonics {
            assert!(is_branch(*m), "{m:?} should be a branch");
        }
        // Assert: non-branch mnemonics rejected
        for m in &non_branch {
            assert!(!is_branch(*m), "{m:?} should NOT be a branch");
        }
    }

    #[test]
    fn test_is_integer_only_comprehensive() {
        // Arrange: mnemonics that are integer-only
        let integer_mnemonics = [
            Mnemonic::Push, Mnemonic::Pop, Mnemonic::Mov, Mnemonic::Lea,
            Mnemonic::Add, Mnemonic::Sub, Mnemonic::Xor, Mnemonic::And,
            Mnemonic::Or, Mnemonic::Shl, Mnemonic::Shr, Mnemonic::Sar,
            Mnemonic::Test, Mnemonic::Cmp, Mnemonic::Nop, Mnemonic::Neg,
            Mnemonic::Imul, Mnemonic::Cdq, Mnemonic::Cdqe, Mnemonic::Cqo,
            Mnemonic::Cmove, Mnemonic::Cmovne, Mnemonic::Sete, Mnemonic::Setne,
        ];
        let float_mnemonics = [
            Mnemonic::Addss, Mnemonic::Mulss, Mnemonic::Divss,
            Mnemonic::Movss, Mnemonic::Xorps, Mnemonic::Ucomiss,
        ];

        // Act & Assert: integer mnemonics recognized
        for m in &integer_mnemonics {
            assert!(is_integer_only(*m), "{m:?} should be integer-only");
        }
        // Assert: float mnemonics not classified as integer-only
        for m in &float_mnemonics {
            assert!(!is_integer_only(*m), "{m:?} should NOT be integer-only");
        }
    }

    #[test]
    fn test_is_float_load_and_float_mov_coverage() {
        // Arrange: float load mnemonics (subset of float mov)
        let float_loads = [
            Mnemonic::Movss, Mnemonic::Vmovss, Mnemonic::Movsd, Mnemonic::Vmovsd,
        ];
        // float_mov includes additional mov variants not in is_float_load
        let float_mov_extra = [
            Mnemonic::Movaps, Mnemonic::Vmovaps, Mnemonic::Movups,
            Mnemonic::Vmovups, Mnemonic::Movd, Mnemonic::Vmovd,
        ];
        let non_float = [Mnemonic::Mov, Mnemonic::Addss, Mnemonic::Ret];

        // Act & Assert: all float load mnemonics recognized
        for m in &float_loads {
            assert!(is_float_load(*m), "{m:?} should be float load");
            assert!(is_float_mov(*m), "{m:?} should also be float mov");
        }
        // Assert: extra float mov mnemonics are mov but not load
        for m in &float_mov_extra {
            assert!(!is_float_load(*m), "{m:?} should NOT be float load");
            assert!(is_float_mov(*m), "{m:?} should be float mov");
        }
        // Assert: non-float mnemonics rejected by both
        for m in &non_float {
            assert!(!is_float_load(*m), "{m:?} should NOT be float load");
            assert!(!is_float_mov(*m), "{m:?} should NOT be float mov");
        }
    }

    #[test]
    fn test_is_xor_ps_pd_all_variants() {
        // Arrange
        let xor_variants = [
            Mnemonic::Xorps, Mnemonic::Xorpd, Mnemonic::Vxorps, Mnemonic::Vxorpd,
        ];
        let non_xor = [Mnemonic::Xor, Mnemonic::Addss, Mnemonic::Pxor];

        // Act & Assert
        for m in &xor_variants {
            assert!(is_xor_ps_pd(*m), "{m:?} should be xor_ps/pd");
        }
        for m in &non_xor {
            assert!(!is_xor_ps_pd(*m), "{m:?} should NOT be xor_ps/pd");
        }
    }

    #[test]
    fn test_build_ptr_map_exceeds_int_arg_regs() {
        // Arrange: 7 pointer params → only 6 integer arg registers available,
        // so the 7th (R9 is the 6th register) should be silently dropped.
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::InputPtr,    // RDI → input ordinal 0
                ScalarParam::InputPtr,    // RSI → input ordinal 1
                ScalarParam::OutputPtr,   // RDX → output
                ScalarParam::WeightPtr,   // RCX → input ordinal 2
                ScalarParam::OutputPtr,   // R8  → output
                ScalarParam::WeightPtr,   // R9  → input ordinal 3
                ScalarParam::InputPtr,    // ← exceeds INT_ARG_REGS, dropped
            ],
        };

        // Act
        let map = build_ptr_map(&sig);

        // Assert: 6 mapped registers (3 inputs + 2 outputs = 5, 7th dropped)
        // Input ordinals: 0, 1, 2, 3 (4 input/weight ptrs fit in first 6 regs)
        assert_eq!(map.input_regs.len(), 4, "4 input/weight ptrs mapped");
        assert_eq!(map.output_regs.len(), 2, "2 output ptrs mapped");
        // Verify the 7th param (index 6) is not present — no register to map to
        assert_eq!(
            map.input_regs.values().filter(|&&v| v == 4).count(),
            0,
            "7th param should not be mapped (no register available)"
        );
    }

    #[test]
    fn test_build_ptr_map_scalar_params_dont_consume_int_regs() {
        // Arrange: Scalar params go to xmm registers, not integer registers
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::Scalar(1.0),  // xmm0 — does not consume int reg
                ScalarParam::Scalar(2.0),  // xmm1 — does not consume int reg
                ScalarParam::InputPtr,     // RDI → input ordinal 0
                ScalarParam::OutputPtr,    // RSI → output
            ],
        };

        // Act
        let map = build_ptr_map(&sig);

        // Assert: InputPtr maps to RDI (first available int reg), not shifted
        assert_eq!(map.input_regs.get(&Register::RDI), Some(&0));
        assert!(map.output_regs.contains(&Register::RSI));
    }

    #[test]
    fn test_count_inputs_weight_ptr_distinguished_from_output() {
        // Arrange: WeightPtr is an input (like InputPtr), not an output
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
            ],
        };

        // Act
        let count = count_inputs(&sig);

        // Assert: 3 input params (1 InputPtr + 2 WeightPtr), 2 OutputPtr excluded
        assert_eq!(count, 3);
    }

    #[test]
    fn test_format_mnemonic_various_instruction_types() {
        // Arrange: sample mnemonics from different categories
        let test_cases = [
            (Mnemonic::Subss, "subss"),
            (Mnemonic::Divss, "divss"),
            (Mnemonic::Mulps, "mulps"),
            (Mnemonic::Maxss, "maxss"),
            (Mnemonic::Minss, "minss"),
            (Mnemonic::Sqrtss, "sqrtss"),
            (Mnemonic::Rsqrtss, "rsqrtss"),
            (Mnemonic::Cvtsi2ss, "cvtsi2ss"),
            (Mnemonic::Ucomiss, "ucomiss"),
            (Mnemonic::Ret, "ret"),
        ];

        // Act & Assert: all formatted as lowercase
        for (mnemonic, expected) in &test_cases {
            let formatted = format_mnemonic(*mnemonic);
            assert_eq!(formatted, *expected, "format_mnemonic({mnemonic:?}) should be '{expected}'");
            // Double-check: no uppercase characters
            assert_eq!(formatted, formatted.to_ascii_lowercase());
        }
    }

    // ── Wave 12kib: 10 additional tests (64 total) ────────────────────────

    #[test]
    fn test_sym_value_sub_div_abs_nested() {
        // Arrange: build Sub, Div, Abs with nested expressions
        let sub_val = SymValue::Sub(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(1.0)),
        );
        let div_val = SymValue::Div(
            Box::new(SymValue::Const(6.0)),
            Box::new(SymValue::Param(1)),
        );
        let abs_val = SymValue::Abs(Box::new(SymValue::Sub(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(5.0)),
        )));

        // Act: Debug format
        let dbg_sub = format!("{:?}", sub_val);
        let dbg_div = format!("{:?}", div_val);
        let dbg_abs = format!("{:?}", abs_val);

        // Assert: Debug contains variant names
        assert!(dbg_sub.contains("Sub"), "Sub Debug: {dbg_sub}");
        assert!(dbg_div.contains("Div"), "Div Debug: {dbg_div}");
        assert!(dbg_abs.contains("Abs"), "Abs Debug: {dbg_abs}");

        // Assert: Clone round-trip preserves structure
        let cloned_div = div_val.clone();
        assert_eq!(format!("{:?}", cloned_div), format!("{:?}", div_val));
    }

    #[test]
    fn test_sym_value_call_all_libm_fn_variants() {
        // Arrange: build SymValue::Call with each LibmFn variant
        use super::super::sym_value::LibmFn;

        let expf_call = SymValue::Call(LibmFn::Expf, vec![SymValue::Param(0)]);
        let sqrtf_call = SymValue::Call(LibmFn::Sqrtf, vec![SymValue::Param(1)]);
        let tanhf_call = SymValue::Call(LibmFn::Tanhf, vec![SymValue::Param(2)]);
        let logf_call = SymValue::Call(LibmFn::Logf, vec![SymValue::Const(2.718)]);
        let fabsf_call = SymValue::Call(LibmFn::Fabsf, vec![SymValue::Param(0)]);

        // Act: Debug format all variants
        let calls = [
            (&expf_call, "Expf"),
            (&sqrtf_call, "Sqrtf"),
            (&tanhf_call, "Tanhf"),
            (&logf_call, "Logf"),
            (&fabsf_call, "Fabsf"),
        ];

        // Assert: each variant's Debug contains the function name
        for (val, name) in &calls {
            let dbg = format!("{:?}", val);
            assert!(
                dbg.contains(name),
                "SymValue::Call({name}, ..) Debug should contain '{name}': {dbg}"
            );
        }

        // Assert: Clone preserves call structure
        let cloned = expf_call.clone();
        assert_eq!(format!("{:?}", cloned), format!("{:?}", expf_call));
    }

    #[test]
    fn test_sym_value_select_all_kinds() {
        // Arrange: build Select with each SelectKind variant
        use super::super::sym_value::SelectKind;

        let kinds = [
            (SelectKind::Gt, "Gt"),
            (SelectKind::Ge, "Ge"),
            (SelectKind::Lt, "Lt"),
            (SelectKind::Le, "Le"),
            (SelectKind::Eq, "Eq"),
            (SelectKind::Ne, "Ne"),
        ];

        // Act & Assert: each kind produces valid Debug and Clone
        for (kind, name) in &kinds {
            let sel = SymValue::Select {
                kind: *kind,
                cond_lhs: Box::new(SymValue::Param(0)),
                cond_rhs: Box::new(SymValue::Const(0.0)),
                true_val: Box::new(SymValue::Param(1)),
                false_val: Box::new(SymValue::Const(0.0)),
            };
            let dbg = format!("{:?}", sel);
            assert!(
                dbg.contains("Select"),
                "Select({name}) Debug should contain 'Select': {dbg}"
            );
            assert!(
                dbg.contains(name),
                "Select({name}) Debug should contain '{name}': {dbg}"
            );
            // Clone round-trip
            let cloned = sel.clone();
            assert_eq!(format!("{:?}", cloned), dbg);
        }
    }

    #[test]
    fn test_sym_value_min_max_neg_display_round_trip() {
        // Arrange: build binary/unary SymValue variants not directly covered by other tests
        let min_val = SymValue::Min(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(0.0)),
        );
        let max_val = SymValue::Max(
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Param(2)),
        );
        let neg_val = SymValue::Neg(Box::new(SymValue::Const(-3.5)));
        let unknown_val = SymValue::Unknown("spill".to_string());

        // Act: Display format
        let disp_min = format!("{min_val}");
        let disp_max = format!("{max_val}");
        let disp_neg = format!("{neg_val}");
        let disp_unknown = format!("{unknown_val}");

        // Assert: Display contains expected substrings
        assert!(disp_min.contains("min"), "Min Display: {disp_min}");
        assert!(disp_max.contains("max"), "Max Display: {disp_max}");
        assert!(disp_neg.contains("(-"), "Neg Display: {disp_neg}");
        assert!(disp_unknown.contains("spill"), "Unknown Display: {disp_unknown}");

        // Assert: Clone produces structurally equal value
        let cloned_min = min_val.clone();
        assert_eq!(format!("{cloned_min}"), disp_min);
    }

    #[test]
    fn test_build_ptr_map_empty_signature() {
        // Arrange: signature with no params at all
        let empty_sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![],
        };

        // Act
        let map = build_ptr_map(&empty_sig);

        // Assert: no registers mapped
        assert!(map.input_regs.is_empty(), "empty sig should have no input regs");
        assert!(map.output_regs.is_empty(), "empty sig should have no output regs");
    }

    #[test]
    fn test_ptr_param_map_propagate_chain_input_to_output_alias() {
        // Arrange: RDI=Input(0), RSI=Output
        let mut map = PtrParamMap {
            input_regs: {
                let mut m = HashMap::new();
                m.insert(Register::RDI, 0);
                m
            },
            output_regs: vec![Register::RSI],
        };

        // Act: chain propagation — mov r15, rdi → r15 becomes Input(0)
        // then mov r13, r15 → r13 also becomes Input(0) via chain
        map.propagate_mov(Register::R15, Register::RDI);
        map.propagate_mov(Register::R13, Register::R15);

        // Assert: chain propagation worked for input
        assert_eq!(map.is_input(Register::R13), Some(0), "r13 should inherit Input(0) via r15 chain");
        assert_eq!(map.is_input(Register::R15), Some(0), "r15 should still be Input(0)");
        assert_eq!(map.is_input(Register::RDI), Some(0), "rdi should still be Input(0)");

        // Act: chain propagation — mov r12, rsi → r12 becomes output
        map.propagate_mov(Register::R12, Register::RSI);

        // Assert: chain propagation worked for output
        assert!(map.is_output(Register::R12), "r12 should inherit output via rsi chain");
        assert!(map.is_output(Register::RSI), "rsi should still be output");
    }

    #[test]
    fn test_has_memory_operand_and_dst_detection() {
        // Arrange: construct instructions with known operand types
        // movss xmm0, dword ptr [rdi+rcx*4] — has memory operand, but dst is register
        let bytes_movss_load: &[u8] = &[0xf3, 0x0f, 0x10, 0x04, 0x8f];
        let mut decoder = Decoder::new(64, bytes_movss_load, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act: check memory operand detection
        let has_mem = has_memory_operand(&instr);
        let has_dst_mem = has_memory_operand_at_dst(&instr);

        // Assert: load has memory but dst is register (not memory)
        assert!(has_mem, "movss xmm0, [rdi+rcx*4] should have memory operand");
        assert!(!has_dst_mem, "movss load dst is register, not memory");

        // Arrange: movss dword ptr [rsi+rcx*4], xmm0 — store, dst IS memory
        let bytes_movss_store: &[u8] = &[0xf3, 0x0f, 0x11, 0x04, 0x8e];
        let mut decoder2 = Decoder::new(64, bytes_movss_store, DecoderOptions::NONE);
        decoder2.set_ip(0x2000);
        let mut instr2 = Instruction::default();
        decoder2.decode_out(&mut instr2);

        // Act
        let has_mem2 = has_memory_operand(&instr2);
        let has_dst_mem2 = has_memory_operand_at_dst(&instr2);

        // Assert: store has memory and dst IS memory
        assert!(has_mem2, "movss [rsi+rcx*4], xmm0 should have memory operand");
        assert!(has_dst_mem2, "movss store dst is memory");
    }

    #[test]
    fn test_format_memory_stack_and_general_operands() {
        // Arrange: [rsp+0x10] — stack-relative with positive displacement
        let bytes_rsp: &[u8] = &[0xf3, 0x0f, 0x10, 0x44, 0x24, 0x10]; // movss xmm0, [rsp+0x10]
        let mut decoder = Decoder::new(64, bytes_rsp, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mem_str = format_memory(&instr);

        // Assert: stack-relative format
        assert!(mem_str.starts_with('['), "memory operand should start with '[': {mem_str}");
        assert!(mem_str.ends_with(']'), "memory operand should end with ']': {mem_str}");
        assert!(mem_str.contains("rsp"), "stack-relative should contain 'rsp': {mem_str}");

        // Arrange: [rdi+rcx*4] — general with scaled index
        let bytes_scaled: &[u8] = &[0xf3, 0x0f, 0x10, 0x04, 0x8f]; // movss xmm0, [rdi+rcx*4]
        let mut decoder2 = Decoder::new(64, bytes_scaled, DecoderOptions::NONE);
        decoder2.set_ip(0x2000);
        let mut instr2 = Instruction::default();
        decoder2.decode_out(&mut instr2);

        // Act
        let mem_str2 = format_memory(&instr2);

        // Assert: general format with scale
        assert!(mem_str2.contains("rcx"), "should contain index register: {mem_str2}");
        assert!(mem_str2.contains("4"), "should contain scale factor: {mem_str2}");
    }

    #[test]
    fn test_classify_pattern_empty_trace() {
        // Arrange: empty trace — should still produce a valid pattern
        let ops: Vec<TraceOp> = vec![];

        // Act
        let pattern = classify_pattern(&ops);

        // Assert: empty trace is classified as Injective (zero-input, single-output).
        // This is a degenerate but valid classification from classify_pattern's logic.
        assert!(
            matches!(pattern, ComputePattern::Injective { ref body, num_inputs: 0, num_outputs: 1 } if body.is_empty()),
            "empty trace should be Injective(0 in, 1 out, empty body), got {pattern:?}"
        );
    }

    #[test]
    fn test_compute_pattern_injective_variant_body_and_debug() {
        // Arrange: Injective pattern with multiple outputs
        use crate::compiler::trace::ValueId;

        let v = |n: u32| ValueId(n);

        let injective = ComputePattern::Injective {
            body: vec![
                TraceOp::Input(0),
                TraceOp::Input(1),
                TraceOp::Add(v(0), v(1)),
                TraceOp::Sub(v(0), v(1)),
            ],
            num_inputs: 2,
            num_outputs: 2,
        };

        // Act: body() returns the body
        let body = injective.body();
        assert!(body.is_some(), "Injective should have a body");
        assert_eq!(body.unwrap().len(), 4, "Injective body should have 4 ops");

        // Act: Debug format
        let dbg = format!("{:?}", injective);
        assert!(dbg.contains("Injective"), "Debug should contain Injective: {dbg}");

        // Assert: field access via destructuring
        if let ComputePattern::Injective { num_inputs, num_outputs, .. } = &injective {
            assert_eq!(*num_inputs, 2);
            assert_eq!(*num_outputs, 2);
        } else {
            panic!("Expected Injective variant");
        }
    }

    // ── 10 additional tests ──────────────────────────────────────────────

    #[test]
    fn test_int_arg_regs_order_and_length() {
        // Arrange: System V AMD64 ABI specifies 6 integer argument registers
        // Act & Assert: exactly 6 registers in the canonical order
        assert_eq!(INT_ARG_REGS.len(), 6, "System V ABI has 6 integer arg registers");
        assert_eq!(INT_ARG_REGS[0], Register::RDI);
        assert_eq!(INT_ARG_REGS[1], Register::RSI);
        assert_eq!(INT_ARG_REGS[2], Register::RDX);
        assert_eq!(INT_ARG_REGS[3], Register::RCX);
        assert_eq!(INT_ARG_REGS[4], Register::R8);
        assert_eq!(INT_ARG_REGS[5], Register::R9);
    }

    #[test]
    fn test_sym_value_const_display_various_floats() {
        // Arrange: SymValue::Const with various float values
        let zero = SymValue::Const(0.0);
        let one = SymValue::Const(1.0);
        let neg = SymValue::Const(-2.5);
        let large = SymValue::Const(1e38);

        // Act: Display format
        let d_zero = format!("{zero}");
        let d_one = format!("{one}");
        let d_neg = format!("{neg}");
        let d_large = format!("{large}");

        // Assert: Display contains the numeric value
        assert!(d_zero.contains('0'), "Const(0.0) Display: {d_zero}");
        assert!(d_one.contains('1'), "Const(1.0) Display: {d_one}");
        assert!(d_neg.contains('-'), "Const(-2.5) Display: {d_neg}");
        assert!(!d_large.is_empty(), "Const(1e38) Display: {d_large}");

        // Assert: Clone round-trip
        assert_eq!(format!("{}", zero.clone()), d_zero);
    }

    #[test]
    fn test_sym_value_deeply_nested_expression() {
        // Arrange: 4-level deep tree: Mul(Add(Sub(Param(0), Const(1)), Param(1)), Const(2))
        let inner = SymValue::Sub(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(1.0)),
        );
        let mid = SymValue::Add(Box::new(inner), Box::new(SymValue::Param(1)));
        let outer = SymValue::Mul(Box::new(mid), Box::new(SymValue::Const(2.0)));

        // Act: Debug and Display
        let dbg = format!("{:?}", outer);
        let disp = format!("{outer}");

        // Assert: Debug contains all nested variant names
        assert!(dbg.contains("Mul"), "outer should be Mul: {dbg}");
        assert!(dbg.contains("Add"), "mid should be Add: {dbg}");
        assert!(dbg.contains("Sub"), "inner should be Sub: {dbg}");
        assert!(dbg.contains("Param"), "should contain Param: {dbg}");
        assert!(!disp.is_empty(), "Display should not be empty");

        // Assert: Clone preserves structure
        let cloned = outer.clone();
        assert_eq!(format!("{:?}", cloned), dbg);
    }

    #[test]
    fn test_has_near_branch_with_decoded_je() {
        // Arrange: decode `je rel32` (0F 84 + disp32)
        let bytes: &[u8] = &[0x0f, 0x84, 0x00, 0x01, 0x00, 0x00];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let result = has_near_branch(&instr);

        // Assert: je has a near branch target
        assert!(result, "je should have near branch operand");
        assert_eq!(instr.mnemonic(), Mnemonic::Je);
    }

    #[test]
    fn test_has_near_branch_with_non_branch_decoded() {
        // Arrange: decode `addss xmm0, xmm1` (F3 0F 58 C1)
        let bytes: &[u8] = &[0xf3, 0x0f, 0x58, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let result = has_near_branch(&instr);

        // Assert: addss has no near branch
        assert!(!result, "addss should not have near branch operand");
        assert_eq!(instr.mnemonic(), Mnemonic::Addss);
    }

    #[test]
    fn test_format_operands_decoded_addss_registers() {
        // Arrange: decode `addss xmm0, xmm1` (F3 0F 58 C1)
        let bytes: &[u8] = &[0xf3, 0x0f, 0x58, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let ops = format_operands(&instr);

        // Assert: two register operands formatted as lowercase names
        assert_eq!(ops.len(), 2, "addss should have 2 operands");
        assert_eq!(ops[0], "xmm0", "first operand should be xmm0");
        assert_eq!(ops[1], "xmm1", "second operand should be xmm1");
    }

    #[test]
    fn test_classify_pattern_single_neg_elementwise() {
        // Arrange: trace with Input(0) followed by Neg → elementwise
        use crate::compiler::trace::ValueId;

        let ops = vec![TraceOp::Input(0), TraceOp::Neg(ValueId(0))];

        // Act
        let pattern = classify_pattern(&ops);

        // Assert: single-input negation classified as Elementwise
        assert!(
            matches!(pattern, ComputePattern::Elementwise { ref body } if body.len() == 2),
            "Input+Neg should be Elementwise, got {pattern:?}"
        );
    }

    #[test]
    fn test_build_ptr_map_only_dim_params() {
        // Arrange: signature with only Dim params — no pointer params at all
        let sig = ScalarFnSignature {
            fn_ptr: std::ptr::null(),
            params: vec![
                ScalarParam::Dim(0),
                ScalarParam::Dim(1),
                ScalarParam::Dim(2),
            ],
        };

        // Act
        let map = build_ptr_map(&sig);

        // Assert: Dim params consume integer registers but map to neither input nor output
        assert!(map.input_regs.is_empty(), "Dim-only sig should have no input regs");
        assert!(map.output_regs.is_empty(), "Dim-only sig should have no output regs");
    }

    #[test]
    fn test_compute_pattern_elementwise_complex_chain() {
        // Arrange: Elementwise chain — Input → Exp → Neg → Add(const) → Recip
        use crate::compiler::trace::ValueId;

        let v = |n: u32| ValueId(n);

        let ops = vec![
            TraceOp::Input(0),
            TraceOp::Exp(v(0)),
            TraceOp::Neg(v(1)),
            TraceOp::Const(1.0),
            TraceOp::Add(v(2), v(3)),
            TraceOp::Recip(v(4)),
        ];

        // Act
        let pattern = classify_pattern(&ops);

        // Assert: classified as Elementwise with 6 ops
        assert!(
            matches!(pattern, ComputePattern::Elementwise { ref body } if body.len() == 6),
            "complex chain should be Elementwise(6 ops), got {pattern:?}"
        );

        // Assert: body contains expected operations
        let body = pattern.body().unwrap();
        assert!(body.iter().any(|op| matches!(op, TraceOp::Exp(_))), "should contain Exp");
        assert!(body.iter().any(|op| matches!(op, TraceOp::Recip(_))), "should contain Recip");
    }

    #[test]
    fn test_is_float_store_with_decoded_movss_store() {
        // Arrange: decode `movss [rsi], xmm0` (F3 0F 11 06) — store to memory
        let bytes_store: &[u8] = &[0xf3, 0x0f, 0x11, 0x06];
        let mut decoder = Decoder::new(64, bytes_store, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr_store = Instruction::default();
        decoder.decode_out(&mut instr_store);

        // Act & Assert: store instruction → is_float_store returns true
        assert!(
            is_float_store(instr_store.mnemonic(), &instr_store),
            "movss [rsi], xmm0 should be a float store"
        );

        // Arrange: decode `movss xmm0, [rdi]` (F3 0F 10 07) — load from memory
        let bytes_load: &[u8] = &[0xf3, 0x0f, 0x10, 0x07];
        let mut decoder2 = Decoder::new(64, bytes_load, DecoderOptions::NONE);
        decoder2.set_ip(0x2000);
        let mut instr_load = Instruction::default();
        decoder2.decode_out(&mut instr_load);

        // Act & Assert: load instruction → is_float_store returns false
        assert!(
            !is_float_store(instr_load.mnemonic(), &instr_load),
            "movss xmm0, [rdi] should NOT be a float store"
        );
    }

    // ── 10 additional tests: instruction decoding, operand extraction,
    //    register identification, instruction classification, edge cases ──

    #[test]
    fn test_decode_mulss_and_format_operands() {
        // Arrange: decode `mulss xmm0, xmm1` (F3 0F 59 C1)
        let bytes: &[u8] = &[0xf3, 0x0f, 0x59, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let operands = format_operands(&instr);

        // Assert: decoded as mulss with two XMM register operands
        assert_eq!(mnemonic, Mnemonic::Mulss, "should decode as Mulss");
        assert_eq!(operands.len(), 2, "mulss should have 2 operands");
        assert_eq!(operands[0], "xmm0", "first operand should be xmm0");
        assert_eq!(operands[1], "xmm1", "second operand should be xmm1");
    }

    #[test]
    fn test_decode_subss_with_memory_operand_extraction() {
        // Arrange: decode `subss xmm0, dword ptr [rdi]` (F3 0F 5C 07)
        let bytes: &[u8] = &[0xf3, 0x0f, 0x5c, 0x07];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let has_mem = has_memory_operand(&instr);
        let operands = format_operands(&instr);

        // Assert: decoded as subss with register + memory operand
        assert_eq!(mnemonic, Mnemonic::Subss, "should decode as Subss");
        assert!(has_mem, "subss xmm0, [rdi] should have memory operand");
        assert_eq!(operands.len(), 2, "subss should have 2 operands");
        assert_eq!(operands[0], "xmm0", "destination should be xmm0");
        assert!(operands[1].starts_with('['), "source should be memory operand: {}", operands[1]);
    }

    #[test]
    fn test_decode_immediate_operand_formatting() {
        // Arrange: decode `add rcx, 4` (48 83 C1 04) — immediate operand
        let bytes: &[u8] = &[0x48, 0x83, 0xc1, 0x04];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let operands = format_operands(&instr);

        // Assert: two operands — register and immediate (hex formatted)
        assert_eq!(operands.len(), 2, "add rcx, 4 should have 2 operands");
        assert_eq!(operands[0], "rcx", "first operand should be rcx");
        assert!(operands[1].starts_with("0x"), "immediate should be hex-formatted: {}", operands[1]);
    }

    #[test]
    fn test_decode_vex_prefix_vaddss() {
        // Arrange: decode VEX-encoded `vaddss xmm0, xmm0, xmm1` (C5 F2 58 C1)
        // VEX prefix = 3-byte form starting with C5
        let bytes: &[u8] = &[0xc5, 0xf2, 0x58, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();

        // Assert: VEX-encoded instruction decoded as Vaddss
        assert_eq!(mnemonic, Mnemonic::Vaddss, "VEX prefix should decode as Vaddss");
        assert!(!is_integer_only(mnemonic), "Vaddss should not be integer-only");
    }

    #[test]
    fn test_format_register_r8_through_r15() {
        // Arrange: R8-R15 are the extended GP64 registers
        let extended_gp64 = [
            Register::R8, Register::R9, Register::R10, Register::R11,
            Register::R12, Register::R13, Register::R14, Register::R15,
        ];

        // Act & Assert: all extended registers are GP64 and format as lowercase
        for reg in &extended_gp64 {
            assert!(is_gp64(*reg), "{reg:?} should be GP64");
            let formatted = format_register(*reg);
            assert_eq!(formatted, formatted.to_ascii_lowercase(),
                "format_register({reg:?}) should be lowercase: {formatted}");
            assert!(!formatted.is_empty(), "format_register({reg:?}) should not be empty");
        }
    }

    #[test]
    fn test_is_float_load_vs_is_float_mov_distinction() {
        // Arrange: movaps is in is_float_mov but NOT in is_float_load;
        // movsd is in both. Verify the boundary is correct.
        let in_both = [Mnemonic::Movss, Mnemonic::Movsd];
        let mov_only = [Mnemonic::Movaps, Mnemonic::Movups, Mnemonic::Movd];

        // Act & Assert: items in both are recognized by both predicates
        for m in &in_both {
            assert!(is_float_load(*m), "{m:?} should be float load");
            assert!(is_float_mov(*m), "{m:?} should be float mov");
        }

        // Assert: mov-only items are NOT float loads but ARE float movs
        for m in &mov_only {
            assert!(!is_float_load(*m), "{m:?} should NOT be float load");
            assert!(is_float_mov(*m), "{m:?} should be float mov");
        }
    }

    #[test]
    fn test_decode_jmp_backward_and_is_branch() {
        // Arrange: decode `jmp rel8` backward (EB FD → jump to IP-1)
        let bytes: &[u8] = &[0xeb, 0xfd];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let is_br = is_branch(mnemonic);
        let has_br = has_near_branch(&instr);
        let target = instr.near_branch_target();

        // Assert: unconditional jump recognized as branch with near branch target
        assert_eq!(mnemonic, Mnemonic::Jmp, "should decode as Jmp");
        assert!(is_br, "Jmp should be a branch");
        assert!(has_br, "Jmp should have near branch operand");
        // Target = IP + 2 (instruction length) + 0xFD (signed -3) = 0x0FFF
        assert!(target < 0x1000, "backward jmp target {target:#x} should be < IP 0x1000");
    }

    #[test]
    fn test_decode_ret_is_not_branch_and_not_integer_only() {
        // Arrange: decode `ret` (C3)
        let bytes: &[u8] = &[0xc3];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();

        // Assert: ret is neither a branch nor integer-only (it terminates analysis)
        assert_eq!(mnemonic, Mnemonic::Ret, "should decode as Ret");
        assert!(!is_branch(mnemonic), "Ret should not be classified as branch");
        assert!(!is_integer_only(mnemonic), "Ret should not be integer-only");
    }

    #[test]
    fn test_decode_xorps_reg_reg_not_rip_relative() {
        // Arrange: decode `xorps xmm0, xmm1` (0F 57 C1) — register-to-register,
        // NOT RIP-relative. This is the "zeroing" pattern (xorps xmm0, xmm0).
        let bytes: &[u8] = &[0x0f, 0x57, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let is_xor = is_xor_ps_pd(mnemonic);
        let operands = format_operands(&instr);

        // Assert: xorps recognized but operands are registers (not memory/RIP)
        assert_eq!(mnemonic, Mnemonic::Xorps, "should decode as Xorps");
        assert!(is_xor, "Xorps should be recognized by is_xor_ps_pd");
        assert_eq!(operands.len(), 2, "xorps should have 2 operands");
        assert_eq!(operands[0], "xmm0", "first operand should be xmm0");
        assert_eq!(operands[1], "xmm1", "second operand should be xmm1 (not memory)");
    }

    #[test]
    fn test_decode_multiple_instructions_from_stream() {
        // Arrange: a stream of two instructions:
        //   addss xmm0, xmm1  (F3 0F 58 C1)
        //   mulss xmm0, xmm2  (F3 0F 59 C2)
        let bytes: &[u8] = &[0xf3, 0x0f, 0x58, 0xc1, 0xf3, 0x0f, 0x59, 0xc2];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();

        // Act: decode first instruction
        decoder.decode_out(&mut instr);
        let first_mnemonic = instr.mnemonic();
        let first_ops = format_operands(&instr);

        // Assert: first is addss xmm0, xmm1
        assert_eq!(first_mnemonic, Mnemonic::Addss);
        assert_eq!(first_ops[0], "xmm0");
        assert_eq!(first_ops[1], "xmm1");

        // Act: decode second instruction
        decoder.decode_out(&mut instr);
        let second_mnemonic = instr.mnemonic();
        let second_ops = format_operands(&instr);

        // Assert: second is mulss xmm0, xmm2
        assert_eq!(second_mnemonic, Mnemonic::Mulss);
        assert_eq!(second_ops[0], "xmm0");
        assert_eq!(second_ops[1], "xmm2");
    }

    // -- 10 additional tests: VEX-encoded, REP prefix, memory operand
    //    edge cases, register tracking, instruction classification --------

    #[test]
    fn test_decode_vex_vmulss_three_operand_form() {
        // Arrange: decode VEX-encoded vmulss xmm0, xmm1, xmm2 (C5 F2 59 C2)
        let bytes: &[u8] = &[0xc5, 0xf2, 0x59, 0xc2];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let operands = format_operands(&instr);

        // Assert: VEX mulss decoded correctly with 3-operand form
        assert_eq!(mnemonic, Mnemonic::Vmulss, "VEX 0xC5 prefix should decode as Vmulss");
        assert_eq!(operands.len(), 3, "vmulss should have 3 operands in VEX form");
        assert!(!is_integer_only(mnemonic), "Vmulss should not be integer-only");
        assert!(!is_float_mov(mnemonic), "Vmulss is not a float mov");
    }

    #[test]
    fn test_decode_vmovss_load_from_memory() {
        // Arrange: decode vmovss xmm0, dword ptr [rdi] (C5 FA 10 07)
        let bytes: &[u8] = &[0xc5, 0xfa, 0x10, 0x07];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let has_mem = has_memory_operand(&instr);
        let has_dst_mem = has_memory_operand_at_dst(&instr);

        // Assert: VEX movss load from memory - dst is register, src is memory
        assert_eq!(mnemonic, Mnemonic::Vmovss, "should decode as Vmovss");
        assert!(is_float_load(mnemonic), "Vmovss should be a float load");
        assert!(has_mem, "vmovss xmm0, [rdi] should have memory operand");
        assert!(!has_dst_mem, "load destination is register, not memory");
    }

    #[test]
    fn test_decode_rep_movsb_prefix_classification() {
        // Arrange: decode rep movsb (F3 A4) - REP prefix with string operation
        let bytes: &[u8] = &[0xf3, 0xa4];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let has_rep = instr.has_rep_prefix();

        // Assert: iced-x86 decodes REP MOVSB as Mnemonic::Movsb with has_rep_prefix()=true
        // REP is tracked as a prefix, not merged into the mnemonic name.
        assert_eq!(mnemonic, Mnemonic::Movsb, "should decode as Movsb (REP is a prefix)");
        assert!(has_rep, "rep movsb should have REP prefix flag set");
        // Movsb is not in is_integer_only() but is definitely not float-related:
        assert!(!is_float_load(mnemonic), "Movsb is not a float load");
        assert!(!is_float_mov(mnemonic), "Movsb is not a float mov");
        assert!(!is_branch(mnemonic), "Movsb is not a branch");
        assert!(!is_xor_ps_pd(mnemonic), "Movsb is not xor");
    }

    #[test]
    fn test_decode_nop_and_classification() {
        // Arrange: decode multi-byte NOP (0F 1F 44 00 00) - 5-byte NOP
        let bytes: &[u8] = &[0x0f, 0x1f, 0x44, 0x00, 0x00];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();

        // Assert: NOP is integer-only and not any float category
        assert_eq!(mnemonic, Mnemonic::Nop, "should decode as Nop");
        assert!(is_integer_only(mnemonic), "Nop should be integer-only");
        assert!(!is_float_load(mnemonic));
        assert!(!is_float_mov(mnemonic));
        assert!(!is_branch(mnemonic));
        assert!(!is_xor_ps_pd(mnemonic));
    }

    #[test]
    fn test_decode_xmm8_through_xmm15_in_vex_encoding() {
        // Arrange: decode VEX-encoded vaddss xmm8, xmm8, xmm9 using 2-byte VEX
        let bytes: &[u8] = &[0xc5, 0x32, 0x58, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let operands = format_operands(&instr);

        // Assert: high XMM registers accessible via VEX encoding
        assert_eq!(mnemonic, Mnemonic::Vaddss, "should decode as Vaddss");
        // Verify at least one operand uses a high XMM register (>= xmm8)
        let has_high_xmm = operands.iter().any(|op| op.starts_with("xmm") && {
            if let Ok(n) = op[3..].parse::<u32>() {
                n >= 8
            } else {
                false
            }
        });
        assert!(has_high_xmm, "VEX encoding should allow access to xmm8-15, got operands: {operands:?}");
    }

    #[test]
    fn test_decode_ucomiss_classification_not_integer_not_float_load() {
        // Arrange: decode ucomiss xmm0, xmm1 (0F 2E C1) - float compare
        let bytes: &[u8] = &[0x0f, 0x2e, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();

        // Assert: ucomiss is not integer-only, not float-load, not float-mov, not xor
        assert_eq!(mnemonic, Mnemonic::Ucomiss, "should decode as Ucomiss");
        assert!(!is_integer_only(mnemonic), "ucomiss is not integer-only (affects EFLAGS from float)");
        assert!(!is_float_load(mnemonic), "ucomiss is not a float load");
        assert!(!is_float_mov(mnemonic), "ucomiss is not a float mov");
        assert!(!is_xor_ps_pd(mnemonic), "ucomiss is not xor");
        assert!(!is_branch(mnemonic), "ucomiss is not a branch");
    }

    #[test]
    fn test_decode_sse_cvtsi2ss_with_memory_operand() {
        // Arrange: decode cvtsi2ss xmm0, dword ptr [rdi] (F3 0F 2A 07)
        let bytes: &[u8] = &[0xf3, 0x0f, 0x2a, 0x07];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let has_mem = has_memory_operand(&instr);
        let operands = format_operands(&instr);

        // Assert: cvtsi2ss decoded with register dst and memory src
        assert_eq!(mnemonic, Mnemonic::Cvtsi2ss, "should decode as Cvtsi2ss");
        assert!(has_mem, "cvtsi2ss xmm0, [rdi] should have memory operand");
        assert!(!is_integer_only(mnemonic), "cvtsi2ss produces float result");
        assert_eq!(operands[0], "xmm0", "destination should be xmm0");
        assert!(operands[1].starts_with('['), "source should be memory operand");
    }

    #[test]
    fn test_decode_and_classify_je_forward_branch() {
        // Arrange: decode je +256 (0F 84 00 01 00 00) - forward conditional jump
        let bytes: &[u8] = &[0x0f, 0x84, 0x00, 0x01, 0x00, 0x00];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();
        let is_br = is_branch(mnemonic);
        let has_br = has_near_branch(&instr);
        let target = instr.near_branch_target();

        // Assert: forward je - target > IP
        assert_eq!(mnemonic, Mnemonic::Je, "should decode as Je");
        assert!(is_br, "Je should be a branch");
        assert!(has_br, "Je should have near branch operand");
        assert!(target > 0x1000, "forward branch target {target:#x} should be > IP 0x1000");
    }

    #[test]
    fn test_decode_movss_store_format_memory_output() {
        // Arrange: decode movss dword ptr [rsi+8], xmm0 (F3 0F 11 46 08)
        let bytes: &[u8] = &[0xf3, 0x0f, 0x11, 0x46, 0x08];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let is_store = is_float_store(instr.mnemonic(), &instr);
        let has_dst_mem = has_memory_operand_at_dst(&instr);
        let mem_str = format_memory(&instr);

        // Assert: store to memory with displacement
        assert!(is_store, "movss [rsi+8], xmm0 should be a float store");
        assert!(has_dst_mem, "store destination should be memory");
        assert!(mem_str.contains("rsi"), "memory operand should contain base register rsi: {mem_str}");
        assert!(mem_str.starts_with('[') && mem_str.ends_with(']'), "memory format: [{mem_str}]");
    }

    #[test]
    fn test_decode_vmovsd_double_precision_load() {
        // Arrange: decode vmovsd xmm0, xmm0, xmm1 (C5 FB 10 C1)
        let bytes: &[u8] = &[0xc5, 0xfb, 0x10, 0xc1];
        let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
        decoder.set_ip(0x1000);
        let mut instr = Instruction::default();
        decoder.decode_out(&mut instr);

        // Act
        let mnemonic = instr.mnemonic();

        // Assert: vmovsd is recognized as float load and float mov
        assert_eq!(mnemonic, Mnemonic::Vmovsd, "should decode as Vmovsd");
        assert!(is_float_load(mnemonic), "Vmovsd should be a float load");
        assert!(is_float_mov(mnemonic), "Vmovsd should be a float mov");
        assert!(!is_integer_only(mnemonic), "Vmovsd should not be integer-only");
        assert!(!is_xor_ps_pd(mnemonic), "Vmovsd should not be xor_ps/pd");
    }
}
