//! Intel oneDNN (MKL-DNN) FFI wrapper.
//!
//! Uses `dlopen` to load libdnnl.so (or libmkl_rt.so) at runtime.
//! oneDNN's `dnnl_sgemm` uses row-major layout natively, so no transpose trick is needed.

use super::IsvGemm;
use std::sync::OnceLock;
use crate::types::CompilerError;

/// Signature: `dnnl_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> i32`
type DnnlSgemmFn = unsafe extern "C" fn(
    transa: std::ffi::c_char,
    transb: std::ffi::c_char,
    m: i64,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    b: *const f32,
    ldb: i64,
    beta: f32,
    c: *mut f32,
    ldc: i64,
) -> i32;

struct OneDnnFfi {
    _handle: *mut std::ffi::c_void,
    sgemm: DnnlSgemmFn,
}

unsafe impl Send for OneDnnFfi {}
unsafe impl Sync for OneDnnFfi {}

static FFI: OnceLock<Option<OneDnnFfi>> = OnceLock::new();

fn load_ffi() -> Option<OneDnnFfi> {
    const LIB_NAMES: &[&[u8]] = &[
        b"libdnnl.so\0",
        b"libmkl_rt.so\0",
    ];

    for name in LIB_NAMES {
        unsafe {
            let handle = libc::dlopen(name.as_ptr() as *const _, libc::RTLD_LAZY);
            if handle.is_null() {
                continue;
            }
            let sym = libc::dlsym(handle, b"dnnl_sgemm\0".as_ptr() as *const _);
            if sym.is_null() {
                libc::dlclose(handle);
                continue;
            }
            return Some(OneDnnFfi {
                _handle: handle,
                sgemm: std::mem::transmute(sym),
            });
        }
    }
    None
}

fn get_ffi() -> Option<&'static OneDnnFfi> {
    FFI.get_or_init(load_ffi).as_ref()
}

pub struct OneDnnBackend;

impl IsvGemm for OneDnnBackend {
    fn is_available() -> bool {
        #[cfg(target_os = "linux")]
        {
            return get_ffi().is_some();
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    unsafe fn sgemm(
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: *const f32, lda: usize,
        b: *const f32, ldb: usize,
        beta: f32,
        c: *mut f32, ldc: usize,
    ) -> Result<(), CompilerError> {
        let ffi = get_ffi().ok_or_else(|| CompilerError::Internal("oneDNN library not loaded".to_string()))?;
        let ret = (ffi.sgemm)(
            b'N' as std::ffi::c_char,
            b'N' as std::ffi::c_char,
            m as i64, n as i64, k as i64,
            alpha,
            a, lda as i64,
            b, ldb as i64,
            beta,
            c, ldc as i64,
        );
        if ret == 0 {
            Ok(())
        } else {
            Err(CompilerError::Internal(format!("dnnl_sgemm returned error code {}", ret)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: is_available returns a concrete bool without panicking ──

    #[test]
    fn is_available_returns_bool_without_panicking() {
        // Arrange: OneDnnBackend is a unit struct, no setup needed
        // Act: call is_available
        let available = OneDnnBackend::is_available();
        // Assert: it returns a bool (compiles) and does not panic
        let _ = available;
    }

    // ── Test 2: is_available is false when library is not installed ──

    #[test]
    fn is_available_is_false_without_library() {
        // Arrange: no libdnnl.so / libmkl_rt.so on this system
        // Act
        let available = OneDnnBackend::is_available();
        // Assert: without the oneDNN shared library, it must report unavailable
        assert!(!available, "oneDNN should not be available without libdnnl.so");
    }

    // ── Test 3: sgemm returns error when library is not loaded ──

    #[test]
    fn sgemm_returns_error_without_library() {
        // Arrange: small 2x2 GEMM buffers
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        // Act
        let result = unsafe {
            OneDnnBackend::sgemm(
                2, 2, 2, 1.0,
                a.as_ptr(), 2,
                b.as_ptr(), 2,
                0.0,
                c.as_mut_ptr(), 2,
            )
        };
        // Assert: without the library, must be an error
        assert!(result.is_err(), "sgemm should fail when oneDNN is not loaded");
    }

    // ── Test 4: sgemm error message mentions oneDNN ──

    #[test]
    fn sgemm_error_message_mentions_onednn() {
        // Arrange: minimal valid pointers
        let a = [0.0f32];
        let b = [0.0f32];
        let mut c = [0.0f32];
        // Act
        let result = unsafe {
            OneDnnBackend::sgemm(
                1, 1, 1, 1.0,
                a.as_ptr(), 1,
                b.as_ptr(), 1,
                0.0,
                c.as_mut_ptr(), 1,
            )
        };
        // Assert: error message should contain "oneDNN"
        let err = result.expect_err("should be error without library");
        let msg = err.to_string();
        assert!(
            msg.contains("oneDNN"),
            "error message should mention oneDNN, got: {msg}"
        );
    }

    // ── Test 5: sgemm error is CompilerError::Internal variant ──

    #[test]
    fn sgemm_error_is_internal_variant() {
        // Arrange
        let a = [0.0f32];
        let b = [0.0f32];
        let mut c = [0.0f32];
        // Act
        let result = unsafe {
            OneDnnBackend::sgemm(
                1, 1, 1, 1.0,
                a.as_ptr(), 1,
                b.as_ptr(), 1,
                0.0,
                c.as_mut_ptr(), 1,
            )
        };
        // Assert: error must be the Internal variant
        let err = result.expect_err("should be error");
        match &err {
            CompilerError::Internal(msg) => {
                assert!(msg.contains("oneDNN"), "Internal message should mention oneDNN");
            }
            other => panic!("expected CompilerError::Internal, got: {other:?}"),
        }
    }

    // ── Test 6: sgemm with zero dimensions still returns library-not-loaded error ──

    #[test]
    fn sgemm_zero_dims_returns_library_error() {
        // Arrange: zero-dimension GEMM, valid pointers
        let a = [0.0f32];
        let b = [0.0f32];
        let mut c = [0.0f32];
        // Act
        let result = unsafe {
            OneDnnBackend::sgemm(
                0, 0, 0, 0.0,
                a.as_ptr(), 0,
                b.as_ptr(), 0,
                0.0,
                c.as_mut_ptr(), 0,
            )
        };
        // Assert: still errors because library is not loaded (not because of dimensions)
        assert!(result.is_err());
    }

    // ── Test 7: is_available is consistent across repeated calls ──

    #[test]
    fn is_available_is_consistent_across_calls() {
        // Arrange: call once to initialize OnceLock
        let first = OneDnnBackend::is_available();
        // Act: call again
        let second = OneDnnBackend::is_available();
        let third = OneDnnBackend::is_available();
        // Assert: OnceLock guarantees deterministic result
        assert_eq!(first, second, "is_available should be consistent");
        assert_eq!(second, third, "is_available should be consistent");
    }

    // ── Test 8: default hgemm returns error ──

    #[test]
    fn hgemm_default_returns_error() {
        // Arrange: OneDnnBackend uses the default hgemm impl from the trait
        // Act
        let result = unsafe {
            OneDnnBackend::hgemm(
                1, 1, 1, 1.0,
                std::ptr::null(), 1,
                std::ptr::null(), 1,
                0.0,
                std::ptr::null_mut(), 1,
            )
        };
        // Assert: default hgemm should return error
        assert!(result.is_err(), "default hgemm should not be supported");
    }

    // ── Test 9: default hgemm error message mentions unsupported ──

    #[test]
    fn hgemm_default_error_message() {
        // Arrange
        // Act
        let result = unsafe {
            OneDnnBackend::hgemm(
                2, 2, 2, 1.0,
                std::ptr::null(), 2,
                std::ptr::null(), 2,
                0.0,
                std::ptr::null_mut(), 2,
            )
        };
        // Assert: error message from the trait default impl
        let msg = result.expect_err("hgemm should error").to_string();
        assert!(
            msg.contains("HGEMM not supported"),
            "error should mention HGEMM not supported, got: {msg}"
        );
    }

    // ── Test 10: OneDnnBackend is a zero-sized unit struct ──

    #[test]
    fn one_dnn_backend_is_zero_sized() {
        // Arrange
        let backend = OneDnnBackend;
        // Act
        let size = std::mem::size_of_val(&backend);
        // Assert: unit struct should be zero-sized
        assert_eq!(size, 0, "OneDnnBackend should be a zero-sized unit struct");
    }
}
