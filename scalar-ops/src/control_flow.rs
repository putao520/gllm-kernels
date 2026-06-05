//! Scalar implementations for control-flow and metadata OpKinds.
//!
//! These operators have real control-flow logic in JIT (conditional jumps,
//! shared memory reads/writes, loop exits). Their scalar reference
//! implementations model the default "no side-effect" path:
//! - Data-passing ops: output = input (identity through)
//! - Condition-check ops: output = [0.0] (default: not triggered)
//!
//! Phase 0 reference only. NOT runtime callable (CLAUDE.md NO_SCALAR).

/// StoreToken: write generated token to output buffer.
/// Scalar: pass-through (identity). JIT handles the actual buffer write.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_store_token(
    input: *const f32,
    output: *mut f32,
    _len: usize,
) {
    if input.is_null() || output.is_null() || _len == 0 {
        return;
    }
    unsafe {
        for i in 0.._len {
            *output.add(i) = *input.add(i);
        }
    }
}

/// CheckStopCondition: check EOS / max_tokens for generate loop exit.
/// Scalar: always returns [0.0] (not stopped). JIT handles the real check.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_check_stop_condition(
    _input: *const f32,
    output: *mut f32,
) {
    if output.is_null() {
        return;
    }
    unsafe {
        *output = 0.0;
    }
}

/// WriteLogits: write selected logits to output buffer.
/// Scalar: pass-through (identity copy).
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_write_logits(
    input: *const f32,
    output: *mut f32,
    len: usize,
) {
    if input.is_null() || output.is_null() || len == 0 {
        return;
    }
    unsafe {
        for i in 0..len {
            *output.add(i) = *input.add(i);
        }
    }
}

/// EarlyExit: conditional early exit at anchor layer.
/// Scalar: pass-through (hidden state unchanged).
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_early_exit(
    input: *const f32,
    output: *mut f32,
    len: usize,
) {
    if input.is_null() || output.is_null() || len == 0 {
        return;
    }
    unsafe {
        for i in 0..len {
            *output.add(i) = *input.add(i);
        }
    }
}

/// GuardrailCheck: post-node veto probe (in-flight safety check).
/// Scalar: pass-through (hidden state unchanged). JIT handles veto logic.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_guardrail_check(
    input: *const f32,
    output: *mut f32,
    len: usize,
) {
    if input.is_null() || output.is_null() || len == 0 {
        return;
    }
    unsafe {
        for i in 0..len {
            *output.add(i) = *input.add(i);
        }
    }
}

/// SgInject: Semantic Gatekeeper knowledge residual vector injection.
/// Scalar: pass-through (hidden state unchanged). JIT handles the ADD.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_sg_inject(
    input: *const f32,
    output: *mut f32,
    len: usize,
) {
    if input.is_null() || output.is_null() || len == 0 {
        return;
    }
    unsafe {
        for i in 0..len {
            *output.add(i) = *input.add(i);
        }
    }
}

/// SgDetect: Semantic Gatekeeper hidden state extraction for detection.
/// Scalar: output = [0.0] (not detected). JIT copies hidden to shared memory.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_sg_detect(
    _input: *const f32,
    output: *mut f32,
) {
    if output.is_null() {
        return;
    }
    unsafe {
        *output = 0.0;
    }
}

/// CotStepCheck: CoT Step Hook step control flags check.
/// Scalar: output = [0.0] (continue). JIT reads shared memory for real check.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_cot_step_check(
    _input: *const f32,
    output: *mut f32,
) {
    if output.is_null() {
        return;
    }
    unsafe {
        *output = 0.0;
    }
}

/// SessionKvRestore: session KV cache cross-turn restore.
/// Scalar: pass-through (input unchanged). JIT handles pointer arithmetic.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_session_kv_restore(
    input: *const f32,
    output: *mut f32,
    len: usize,
) {
    if input.is_null() || output.is_null() || len == 0 {
        return;
    }
    unsafe {
        for i in 0..len {
            *output.add(i) = *input.add(i);
        }
    }
}

/// MmHiddenInject: multimodal fused hidden state injection.
/// Scalar: pass-through (embedding unchanged). JIT handles the ADD.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_mm_hidden_inject(
    input: *const f32,
    output: *mut f32,
    len: usize,
) {
    if input.is_null() || output.is_null() || len == 0 {
        return;
    }
    unsafe {
        for i in 0..len {
            *output.add(i) = *input.add(i);
        }
    }
}
