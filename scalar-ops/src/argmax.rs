//! Scalar Argmax — find index of maximum element in a vector.
//!
//! Phase 0 reference implementation for SymExec trace extraction.
//! Argmax is a Reduction op (OpClass::Reduction).

/// Find the index of the maximum value in `data[0..len]`.
///
/// Returns the index as f32 (JIT convention: scalar results in f32 registers).
/// For empty input, returns 0.0.
#[no_mangle]
pub unsafe extern "C" fn scalar_argmax(data: *const f32, out: *mut f32, len: usize) {
    if len == 0 || data.is_null() || out.is_null() {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mut best_idx: usize = 0;
    let mut best_val: f32 = slice[0];
    for i in 1..len {
        if slice[i] > best_val {
            best_val = slice[i];
            best_idx = i;
        }
    }
    unsafe {
        *out = best_idx as f32;
    }
}
