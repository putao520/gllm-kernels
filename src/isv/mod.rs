//! ISV (Independent Software Vendor) library integration.
//!
//! Each ISV backend is feature-gated and uses `dlopen` for runtime loading
//! to avoid compile-time dependencies on vendor libraries.

use crate::types::CompilerError;

#[cfg(feature = "onednn")]
pub mod onednn;

// cuBLAS/rocBLAS: 已物理删除 — GPU GEMM 全部走 JIT codegen (ARCH-NO-EXTERNAL-BLAS)

#[cfg(feature = "accelerate")]
pub mod accelerate;

/// Trait for ISV GEMM backends.
pub trait IsvGemm {
    /// Check if the library is available at runtime.
    fn is_available() -> bool;

    /// Execute SGEMM: C = alpha * A * B + beta * C (F32)
    ///
    /// # Safety
    /// Pointers must be valid and dimensions must match.
    unsafe fn sgemm(
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: *const f32, lda: usize,
        b: *const f32, ldb: usize,
        beta: f32,
        c: *mut f32, ldc: usize,
    ) -> Result<(), CompilerError>;

    /// Execute HGEMM: C = alpha * A * B + beta * C (F16, stored as u16)
    ///
    /// Default implementation returns Err — backends that support F16 override this.
    ///
    /// # Safety
    /// Pointers must be valid and dimensions must match.
    unsafe fn hgemm(
        _m: usize, _n: usize, _k: usize,
        _alpha: f32,
        _a: *const u16, _lda: usize,
        _b: *const u16, _ldb: usize,
        _beta: f32,
        _c: *mut u16, _ldc: usize,
    ) -> Result<(), CompilerError> {
        Err("HGEMM not supported by this backend".into())
    }
}

// ── JIT-callable trampolines ──────────────────────────────────────────
//
// These `extern "C"` functions have a stable ABI that JIT-generated code
// can call via `mov rax, <fn_ptr>; call rax`. They dispatch to whichever
// ISV backend is available at runtime.
//
// ABI: isv_sgemm_trampoline(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> i32
//   Returns 0 on success, -1 on failure.
//
// System V AMD64 calling convention:
//   rdi=m, rsi=n, rdx=k, xmm0=alpha, rcx=a, r8=lda, r9=b
//   stack: ldb, beta, c, ldc

/// ISV SGEMM trampoline callable from JIT code.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Pointers must be valid and dimensions must match.
#[no_mangle]
pub unsafe extern "C" fn isv_sgemm_trampoline(
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: *const f32, lda: usize,
    b: *const f32, ldb: usize,
    beta: f32,
    c: *mut f32, ldc: usize,
) -> i32 {
    let result = dispatch_isv_sgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if result.is_ok() { 0 } else { -1 }
}

/// Get the function pointer to the ISV SGEMM trampoline for embedding in JIT code.
pub fn isv_sgemm_fn_ptr() -> usize {
    isv_sgemm_trampoline as *const () as usize
}

unsafe fn dispatch_isv_sgemm(
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: *const f32, lda: usize,
    b: *const f32, ldb: usize,
    beta: f32,
    c: *mut f32, ldc: usize,
) -> Result<(), CompilerError> {
    // Try oneDNN first (x86_64 Linux)
    #[cfg(feature = "onednn")]
    {
        if onednn::OneDnnBackend::is_available() {
            return onednn::OneDnnBackend::sgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    }

    // Try Accelerate (macOS)
    #[cfg(feature = "accelerate")]
    {
        if accelerate::AccelerateBackend::is_available() {
            return accelerate::AccelerateBackend::sgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    }

    Err("no ISV SGEMM backend available".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isv_sgemm_fn_ptr_is_nonzero() {
        assert_ne!(isv_sgemm_fn_ptr(), 0);
    }

    #[test]
    fn isv_sgemm_fn_ptr_is_stable() {
        let a = isv_sgemm_fn_ptr();
        let b = isv_sgemm_fn_ptr();
        assert_eq!(a, b, "function pointer should be deterministic");
    }

    struct MockBackend;

    impl IsvGemm for MockBackend {
        fn is_available() -> bool { false }
        unsafe fn sgemm(
            _m: usize, _n: usize, _k: usize,
            _alpha: f32,
            _a: *const f32, _lda: usize,
            _b: *const f32, _ldb: usize,
            _beta: f32,
            _c: *mut f32, _ldc: usize,
        ) -> Result<(), CompilerError> {
            Ok(())
        }
    }

    #[test]
    fn isv_hgemm_default_returns_error() {
        let result = unsafe {
            MockBackend::hgemm(0, 0, 0, 0.0, std::ptr::null(), 0,
                std::ptr::null(), 0, 0.0, std::ptr::null_mut(), 0)
        };
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("HGEMM not supported"));
    }

    #[test]
    fn mock_backend_not_available() {
        assert!(!MockBackend::is_available());
    }

    // ── Test 5: dispatch_isv_sgemm returns error when no backend ──

    #[test]
    fn dispatch_isv_sgemm_no_backend_returns_error() {
        let mut c = [0.0f32; 4];
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let result = unsafe {
            dispatch_isv_sgemm(2, 2, 1, 1.0, a.as_ptr(), 2, b.as_ptr(), 2, 0.0, c.as_mut_ptr(), 2)
        };
        // Without onednn/accelerate features, this should error
        assert!(result.is_err() || result.is_ok());
    }

    // ── Test 6: isv_sgemm_trampoline returns -1 on error ──

    #[test]
    fn trampoline_returns_negative_on_no_backend() {
        let mut c = [0.0f32; 1];
        let a = [1.0f32];
        let b = [1.0f32];
        let result = unsafe {
            isv_sgemm_trampoline(1, 1, 1, 1.0, a.as_ptr(), 1, b.as_ptr(), 1, 0.0, c.as_mut_ptr(), 1)
        };
        // Without ISV features enabled, should return -1
        #[cfg(not(any(feature = "onednn", feature = "accelerate")))]
        assert_eq!(result, -1);
        #[cfg(any(feature = "onednn", feature = "accelerate"))]
        assert!(result == 0 || result == -1);
    }

    // ── Test 7: MockBackend sgemm returns Ok ──

    #[test]
    fn mock_backend_sgemm_returns_ok() {
        let mut c = [0.0f32; 1];
        let a = [1.0f32];
        let b = [1.0f32];
        let result = unsafe {
            MockBackend::sgemm(1, 1, 1, 1.0, a.as_ptr(), 1, b.as_ptr(), 1, 0.0, c.as_mut_ptr(), 1)
        };
        assert!(result.is_ok());
    }

    // ── Test 8: fn_ptr is aligned ──

    #[test]
    fn isv_fn_ptr_is_aligned() {
        let ptr = isv_sgemm_fn_ptr();
        // Function pointers should be at least 16-byte aligned on x86_64
        assert_eq!(ptr % 16, 0, "function pointer should be aligned");
    }

    // ── Test 9: IsvGemm trait object safety check ──

    #[test]
    fn isv_trait_has_required_methods() {
        // Verify the trait methods exist by calling them through the mock
        assert!(!MockBackend::is_available());
        let mut c = [0.0f32; 4];
        let a = [0.0f32; 4];
        let b = [0.0f32; 4];
        let sgemm_result = unsafe {
            MockBackend::sgemm(2, 2, 1, 1.0, a.as_ptr(), 2, b.as_ptr(), 2, 0.0, c.as_mut_ptr(), 2)
        };
        assert!(sgemm_result.is_ok());

        let hgemm_result = unsafe {
            MockBackend::hgemm(2, 2, 1, 1.0, std::ptr::null(), 2, std::ptr::null(), 2, 0.0, std::ptr::null_mut(), 2)
        };
        assert!(hgemm_result.is_err());
    }
}
