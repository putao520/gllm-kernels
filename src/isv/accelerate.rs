//! Apple Accelerate (vecLib/BLAS) FFI wrapper.
//!
//! Directly links against the Accelerate framework on macOS.
//! Uses row-major layout natively via `CblasRowMajor`.

use super::IsvGemm;
use crate::types::CompilerError;

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;

#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

pub struct AccelerateBackend;

impl IsvGemm for AccelerateBackend {
    fn is_available() -> bool {
        cfg!(target_os = "macos")
    }

    unsafe fn sgemm(
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: *const f32, lda: usize,
        b: *const f32, ldb: usize,
        beta: f32,
        c: *mut f32, ldc: usize,
    ) -> Result<(), CompilerError> {
        #[cfg(target_os = "macos")]
        {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32, n as i32, k as i32,
                alpha,
                a, lda as i32,
                b, ldb as i32,
                beta,
                c, ldc as i32,
            );
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            Err("Accelerate is only available on macOS".into())
        }
    }
}
