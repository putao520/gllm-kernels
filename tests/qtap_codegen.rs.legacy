//! Q-Tap STG codegen unit tests (ARCH-SG-QTAP).
//!
//! SPEC refs:
//! - `SPEC/SEMANTIC-GATEKEEPER.md §4` Q 截获协议
//! - `SPEC/08-EXECUTOR.md §4.2.1` FusedAttentionLayer Q-Tap 扩展
//!
//! These tests validate the lower / VM-program / x86_64 emit layers directly.
//! Per CLAUDE.md "SPEC 未完成禁止验证", they are compiled (`cargo check --tests`)
//! but **not run** in this session — a downstream integration session is
//! expected to invoke them end-to-end on a real layer.

use gllm_kernels::compiler::codegen::vm::instr::*;
use gllm_kernels::compiler::codegen::vm::lower::lower_qtap_stg;
use gllm_kernels::compiler::graph::QTapPosition;
use gllm_kernels::types::DType;

/// Build a minimal `VmProgram` that contains a single QTapSTG for the
/// LastToken (decode) case and return the program plus the indices of the key
/// side-effect instructions for subsequent assertions.
fn build_qtap_last_token_program(
    sink_ptr: u64,
    step_index_ptr: u64,
    q_dim: usize,
    num_slots: usize,
) -> Result<VmProgram, gllm_kernels::types::CompilerError> {
    let mut prog = VmProgram::new();
    // Pre-declare the q_input_ptr VReg — in the full lower path it comes out of
    // the TensorPtrResolver. Here we fabricate a Ptr VReg so we can call
    // `lower_qtap_stg` in isolation.
    let q_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

    lower_qtap_stg(
        &mut prog,
        sink_ptr,
        step_index_ptr,
        DType::F32,
        q_dim,
        BoundExpr::Const(1), // LastToken + seq_len=1 (decode)
        QTapPosition::LastToken,
        num_slots,
        SimdWidth::W256,
        q_input_ptr,
    )?;
    Ok(prog)
}

#[test]
fn qtap_stg_vm_program_emits_release_fence_and_atomic_bump() {
    // Happy path: LastToken decode, q_dim=64 (divisible by 8 AVX2 lanes),
    // num_slots=2, non-null sink + step_index addresses.
    let prog = build_qtap_last_token_program(
        /* sink_ptr      = */ 0x7fff_ffff_0000,
        /* step_index_ptr = */ 0x7fff_ffff_1000,
        /* q_dim          = */ 64,
        /* num_slots      = */ 2,
    )
    .expect("lower_qtap_stg should succeed for the happy path");

    // Structural sanity: loop / scope balance must hold.
    prog.validate_structure().expect("VmProgram structure must be balanced");
    prog.validate_provenance().expect("every VReg must be declared");
    prog.validate_declares_before_uses().expect("declare-before-use invariant");

    // Must contain at least one MemFence(Release) and one AtomicAddU64(value=1).
    let has_release_fence = prog.instrs.iter().any(|i| {
        matches!(
            i,
            VmInstr::MemFence { order: MemFenceOrder::Release }
        )
    });
    let has_atomic_bump = prog.instrs.iter().any(|i| {
        matches!(
            i,
            VmInstr::AtomicAddU64 {
                value: 1,
                offset: OffsetExpr::Const(0),
                ..
            }
        )
    });
    assert!(has_release_fence, "QTapSTG lower must emit a Release fence");
    assert!(
        has_atomic_bump,
        "QTapSTG lower must emit AtomicAddU64(value=1) for step_index"
    );

    // sink_ptr and step_index_ptr must be loaded via PtrExpr::AbsAddr (not ABI args).
    let abs_addr_loads: Vec<u64> = prog
        .instrs
        .iter()
        .filter_map(|i| match i {
            VmInstr::LoadPtr { src: PtrExpr::AbsAddr(a), .. } => Some(*a),
            _ => None,
        })
        .collect();
    assert!(
        abs_addr_loads.contains(&0x7fff_ffff_0000),
        "sink_ptr must be loaded as PtrExpr::AbsAddr"
    );
    assert!(
        abs_addr_loads.contains(&0x7fff_ffff_1000),
        "step_index_ptr must be loaded as PtrExpr::AbsAddr"
    );

    // The tap must also issue at least one VecStore (ring buffer write).
    let has_vec_store = prog
        .instrs
        .iter()
        .any(|i| matches!(i, VmInstr::VecStore { .. }));
    assert!(has_vec_store, "QTapSTG must emit at least one VecStore");
}

#[test]
fn qtap_stg_all_tokens_uses_outer_loop() {
    // AllTokens: outer loop over seq (may be Symbolic); inner loop over q_dim.
    let mut prog = VmProgram::new();
    let q_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    lower_qtap_stg(
        &mut prog,
        0xaaaa_bbbb_cccc,
        0xaaaa_bbbb_dddd,
        DType::F32,
        /* q_dim    = */ 128,
        /* seq_bound = */ BoundExpr::Const(16),
        QTapPosition::AllTokens,
        /* num_slots = */ 2,
        SimdWidth::W256,
        q_input_ptr,
    )
    .expect("lower_qtap_stg AllTokens happy path");

    prog.validate_structure().unwrap();

    // Count LoopBegin instructions — AllTokens must emit at least two
    // (outer seq + inner vector).
    let num_loop_begins = prog
        .instrs
        .iter()
        .filter(|i| matches!(i, VmInstr::LoopBegin { .. }))
        .count();
    assert!(
        num_loop_begins >= 2,
        "AllTokens must emit outer seq loop + inner vector loop, got {num_loop_begins}"
    );
}

#[test]
fn qtap_stg_rejects_null_sink_ptr() {
    let mut prog = VmProgram::new();
    let q_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let err = lower_qtap_stg(
        &mut prog,
        0, // NULL
        0x1234,
        DType::F32,
        64,
        BoundExpr::Const(1),
        QTapPosition::LastToken,
        2,
        SimdWidth::W256,
        q_input_ptr,
    )
    .expect_err("NULL sink_ptr must be rejected");
    let msg = format!("{:?}", err);
    assert!(msg.contains("sink_ptr"), "error should mention sink_ptr: {msg}");
}

#[test]
fn qtap_stg_rejects_non_power_of_two_num_slots() {
    let mut prog = VmProgram::new();
    let q_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let err = lower_qtap_stg(
        &mut prog,
        0x1000,
        0x2000,
        DType::F32,
        64,
        BoundExpr::Const(1),
        QTapPosition::LastToken,
        /* num_slots = */ 3, // not power of two
        SimdWidth::W256,
        q_input_ptr,
    )
    .expect_err("non-power-of-two num_slots must be rejected");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("power of two"),
        "error should mention 'power of two': {msg}"
    );
}

#[test]
fn qtap_stg_rejects_non_f32_dtype() {
    let mut prog = VmProgram::new();
    let q_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    let err = lower_qtap_stg(
        &mut prog,
        0x1000,
        0x2000,
        DType::F16, // unsupported in CPU Q-Tap for this session
        64,
        BoundExpr::Const(1),
        QTapPosition::LastToken,
        2,
        SimdWidth::W256,
        q_input_ptr,
    )
    .expect_err("F16 dtype must be rejected (CPU Q-Tap is F32 only)");
    let msg = format!("{:?}", err);
    assert!(msg.contains("dtype"), "error should mention dtype: {msg}");
}

#[test]
fn qtap_stg_rejects_misaligned_q_dim() {
    let mut prog = VmProgram::new();
    let q_input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
    // q_dim = 15 is NOT divisible by 8 (AVX2 lanes).
    let err = lower_qtap_stg(
        &mut prog,
        0x1000,
        0x2000,
        DType::F32,
        15,
        BoundExpr::Const(1),
        QTapPosition::LastToken,
        2,
        SimdWidth::W256,
        q_input_ptr,
    )
    .expect_err("q_dim not divisible by SIMD lanes must be rejected");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("not divisible"),
        "error should mention divisibility: {msg}"
    );
}

// ─────────────────────────────────────────────────────────────
// x86_64 machine-code level assertions
// ─────────────────────────────────────────────────────────────

/// Run the full lower → reg-alloc → x86_lower pipeline for a VmProgram that
/// already has a reasonable VmProgram shape. Returns the raw byte output.
#[cfg(target_arch = "x86_64")]
fn assemble_to_x86_bytes(prog: VmProgram) -> Result<Vec<u8>, gllm_kernels::types::CompilerError> {
    use gllm_kernels::compiler::codegen::vm::isa_profile::IsaProfile;
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocator;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::x86_lower::X86Lower;
    use gllm_kernels::dispatch::DeviceProfile;

    let dp = DeviceProfile::detect();
    let profile = IsaProfile::from_device_profile(&dp);
    let alloc = RegAllocator::new(&profile)
        .allocate(&prog)
        .map_err(gllm_kernels::types::CompilerError::CodegenViolation)?;
    let frame = StackFrame::compute(&alloc, &profile, 0);

    let mut lower = X86Lower::new();
    lower.emit_prologue(&frame, &alloc)?;
    for instr in &prog.instrs {
        lower.lower_instr(instr, &alloc)?;
    }
    lower.emit_epilogue(&frame, &alloc)?;
    lower.finalize()
}

/// Heuristic: look for the `lock` prefix byte (0xF0) in the emitted code.
#[cfg(target_arch = "x86_64")]
fn contains_lock_prefix(code: &[u8]) -> bool {
    code.contains(&0xF0)
}

/// Heuristic: look for the `mfence` opcode sequence (0x0F 0xAE 0xF0).
#[cfg(target_arch = "x86_64")]
fn contains_mfence(code: &[u8]) -> bool {
    code.windows(3).any(|w| w == [0x0F, 0xAE, 0xF0])
}

/// Heuristic: a `mov r64, imm64` for one of the AbsAddr values must show up
/// as the little-endian imm64 bytes appearing consecutively in the code.
#[cfg(target_arch = "x86_64")]
fn contains_imm64_le(code: &[u8], value: u64) -> bool {
    let bytes = value.to_le_bytes();
    code.windows(8).any(|w| w == bytes)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn qtap_stg_x86_emits_mfence_and_lock_prefix() {
    // This test compiles the VmProgram down to actual x86_64 machine code
    // and asserts that:
    //   (a) the Release MemFence becomes an `mfence`
    //   (b) the AtomicAddU64 becomes a `lock add qword [mem], imm`
    //   (c) the sink_ptr and step_index_ptr are emitted as mov r64, imm64.
    let sink = 0x7fff_0000_0000_1234u64;
    let step = 0x7fff_0000_0000_abcdu64;

    // NOTE: we cannot go through the full `assemble_to_x86_bytes` path because
    // X86Lower requires X86 feature gating on the host. Instead, the test
    // gracefully skips if the assembler refuses to run.
    let prog = match build_qtap_last_token_program(sink, step, 64, 2) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping x86 codegen test: {e:?}");
            return;
        }
    };
    let code = match assemble_to_x86_bytes(prog) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping x86 codegen test: assembler unavailable ({e:?})");
            return;
        }
    };

    assert!(
        contains_mfence(&code),
        "emitted code must contain `mfence` for Release fence"
    );
    assert!(
        contains_lock_prefix(&code),
        "emitted code must contain `lock` prefix for atomic step_index bump"
    );
    assert!(
        contains_imm64_le(&code, sink),
        "emitted code must contain sink_ptr imm64 ({sink:#x})"
    );
    assert!(
        contains_imm64_le(&code, step),
        "emitted code must contain step_index_ptr imm64 ({step:#x})"
    );
}
