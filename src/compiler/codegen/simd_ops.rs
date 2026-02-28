//! SimdOps trait — virtual SIMD instruction set for platform-agnostic codegen.
//!
//! The algorithm layer (`algorithm.rs`) generates code by calling methods on
//! `SimdOps`. Each backend (x86_64, aarch64) provides a concrete implementation
//! that emits real machine instructions.
//!
//! Key design decisions:
//! - `VReg(u8)` is a lightweight virtual register index; backends map to physical regs.
//! - All methods return `Result<(), String>` for uniform error handling.
//! - Generic `<E: SimdOps>` ensures monomorphization (no vtable overhead).

use super::CodegenOutput;

// ── Virtual register model ──────────────────────────────────────────────────

/// Virtual SIMD register — a lightweight index mapped to physical registers
/// by each backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(pub u8);

/// Base register for memory addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseReg {
    /// Function argument register (0=first arg, 1=second, etc.)
    Arg(u8),
    /// Stack pointer.
    StackPtr,
    /// Scratchpad base pointer (loaded from stack at function entry).
    ScratchpadBase,
    /// BLIS loop variable (backend allocates callee-saved GPR).
    /// 0=jc, 1=pc, 2=ic, 3=jr, 4=ir
    LoopVar(u8),
    /// Output pointer (r8 on x86, x7 on aarch64).
    OutputPtr,
    /// Temporary GPR (backend picks a free caller-saved register).
    Scratch(u8),
}

/// Memory operand for load/store.
#[derive(Debug, Clone, Copy)]
pub struct MemOperand {
    pub base: BaseReg,
    pub offset: i32,
}

/// Label for branch targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

// ── SimdOps trait ───────────────────────────────────────────────────────────

/// Platform-agnostic SIMD instruction interface.
///
/// Backends implement this trait to emit real machine instructions. The
/// algorithm layer calls these methods to generate code without knowing
/// the target ISA.
///
/// Register conventions:
/// - VReg(0..n_accum) are accumulator registers for GEMM tiles
/// - VReg(n_accum..n_accum+3) are scratch registers for math approximations
/// - The backend maps VReg indices to physical SIMD registers
pub trait SimdOps {
    // ── Vector arithmetic ───────────────────────────────────────────────

    /// dst = a + b
    fn vadd(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a - b
    fn vsub(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a * b
    fn vmul(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a / b
    fn vdiv(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a * b + c  (fused multiply-add)
    fn vfma(&mut self, dst: VReg, a: VReg, b: VReg, c: VReg) -> Result<(), String>;
    /// dst = -a
    fn vneg(&mut self, dst: VReg, a: VReg) -> Result<(), String>;
    /// dst = |a|
    fn vabs(&mut self, dst: VReg, a: VReg) -> Result<(), String>;
    /// dst = sqrt(a)
    fn vsqrt(&mut self, dst: VReg, a: VReg) -> Result<(), String>;
    /// dst = max(a, b)
    fn vmax(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = min(a, b)
    fn vmin(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;

    // ── Approximate reciprocals ─────────────────────────────────────────

    /// dst ≈ 1/a  (fast reciprocal, ~12-bit precision)
    fn vrecip(&mut self, dst: VReg, a: VReg) -> Result<(), String>;
    /// dst ≈ 1/sqrt(a)  (fast inverse sqrt, ~12-bit precision)
    fn vrsqrt(&mut self, dst: VReg, a: VReg) -> Result<(), String>;

    // ── Memory operations ───────────────────────────────────────────────

    /// dst = load SIMD vector from memory
    fn vload(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String>;
    /// store SIMD vector to memory
    fn vstore(&mut self, mem: MemOperand, src: VReg) -> Result<(), String>;
    /// dst = broadcast scalar f32 from memory to all lanes
    fn vbroadcast(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String>;
    /// dst = broadcast compile-time f32 constant to all lanes
    fn vbroadcast_const(&mut self, dst: VReg, val: f32) -> Result<(), String>;
    /// dst = all zeros
    fn vzero(&mut self, dst: VReg) -> Result<(), String>;
    /// dst = copy src
    fn vmov(&mut self, dst: VReg, src: VReg) -> Result<(), String>;

    // ── Bitwise / integer operations (for exp/tanh/log) ─────────────────

    /// dst = a & b  (bitwise AND)
    fn vand(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a | b  (bitwise OR)
    fn vor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a ^ b  (bitwise XOR)
    fn vxor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a >> imm  (logical right shift, 32-bit lanes)
    fn vshr_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String>;
    /// dst = a << imm  (logical left shift, 32-bit lanes)
    fn vshl_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String>;
    /// dst = convert i32 lanes to f32
    fn vcvt_i32_f32(&mut self, dst: VReg, a: VReg) -> Result<(), String>;
    /// dst = convert f32 lanes to i32 (truncate toward zero)
    fn vcvt_f32_i32(&mut self, dst: VReg, a: VReg) -> Result<(), String>;
    /// dst = round f32 to nearest integer (as f32)
    fn vround(&mut self, dst: VReg, a: VReg) -> Result<(), String>;
    /// dst = a + b  (32-bit integer lanes)
    fn vadd_i32(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;

    // ── FMA variants for Horner polynomial evaluation ───────────────────

    /// dst = dst * a + b  (in-place FMA for Horner chains)
    fn vfmadd213(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;
    /// dst = a * b + dst  (accumulating FMA)
    fn vfmadd231(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String>;

    // ── Loop control ────────────────────────────────────────────────────

    /// Allocate a new label (returns unique label ID).
    fn alloc_label(&mut self) -> Label;
    /// Define a label at the current code position.
    fn define_label(&mut self, label: Label) -> Result<(), String>;
    /// Unconditional jump to label.
    fn jump(&mut self, label: Label) -> Result<(), String>;
    /// Decrement GPR and branch to label if non-zero.
    fn dec_and_branch_nz(&mut self, counter: BaseReg, label: Label) -> Result<(), String>;
    /// Compare GPR to immediate and branch if less than.
    fn cmp_and_branch_lt(&mut self, reg: BaseReg, imm: i64, label: Label) -> Result<(), String>;
    /// Compare GPR to immediate and branch if greater-or-equal.
    fn cmp_and_branch_ge(&mut self, reg: BaseReg, imm: i64, label: Label) -> Result<(), String>;

    // ── GPR operations ──────────────────────────────────────────────────

    /// Load immediate value into GPR.
    fn gpr_load_imm(&mut self, dst: BaseReg, imm: i64) -> Result<(), String>;
    /// dst = dst + imm
    fn gpr_add_imm(&mut self, dst: BaseReg, imm: i32) -> Result<(), String>;
    /// dst = src
    fn gpr_mov(&mut self, dst: BaseReg, src: BaseReg) -> Result<(), String>;

    // ── Function frame ──────────────────────────────────────────────────

    /// Emit function prologue (save callee-saved registers, set up frame).
    fn emit_prologue(&mut self) -> Result<(), String>;
    /// Emit function epilogue (restore registers, return).
    fn emit_epilogue(&mut self) -> Result<(), String>;
    /// Finalize code generation and return assembled machine code.
    fn finalize(&mut self) -> Result<CodegenOutput, String>;

    // ── Prefetch ────────────────────────────────────────────────────────

    /// Prefetch memory to L1 cache.
    fn prefetch_l1(&mut self, mem: MemOperand) -> Result<(), String>;

    // ── Non-temporal store ──────────────────────────────────────────────

    /// Store SIMD vector using non-temporal hint (bypass cache).
    fn vstore_nt(&mut self, mem: MemOperand, src: VReg) -> Result<(), String>;

    // ── Memory fence ────────────────────────────────────────────────────

    /// Store fence (ensure all prior stores are visible).
    fn sfence(&mut self) -> Result<(), String>;

    // ── Scalar operations (for tail handling) ───────────────────────────

    /// Load single f32 scalar into lowest lane of dst (upper lanes zeroed).
    fn scalar_load(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String>;
    /// Store lowest lane of src as single f32 scalar.
    fn scalar_store(&mut self, mem: MemOperand, src: VReg) -> Result<(), String>;

    // ── External function calls ─────────────────────────────────────────

    /// Call an external function pointer (for pack_a/pack_b/norm).
    fn call_fn_ptr(&mut self, addr: u64) -> Result<(), String>;

    // ── NOP placeholder ─────────────────────────────────────────────────

    /// Emit a no-op instruction.
    fn emit_nop(&mut self) -> Result<(), String>;
}
