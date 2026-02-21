//! C ABI wrapper for gllm-kernels inference backend.
//!
//! Provides `#[no_mangle] extern "C"` functions for use from C/C++/Python.
//! All functions return `GllmStatus` error codes and use opaque handles.

pub mod types;

pub use types::{GllmStatus, GllmBackend, GllmTensor, GllmKvCache, GllmWeights, GllmModelConfig};

use crate::inference::cpu_backend::CpuInferenceBackend;
use crate::inference::tensor::DeviceTensor;
use crate::inference::kv_cache::KvCache;
use crate::inference::InferenceBackend;

// ── Backend lifecycle ───────────────────────────────────────────────

/// Initialize a CPU inference backend.
///
/// # Safety
/// `config` must point to a valid `GllmModelConfig`.
/// `out` must point to a valid `GllmBackend` pointer.
#[no_mangle]
pub unsafe extern "C" fn gllm_backend_init(
    config: *const GllmModelConfig,
    out: *mut GllmBackend,
) -> i32 {
    if config.is_null() || out.is_null() {
        return GllmStatus::InvalidArg as i32;
    }

    let cfg = match (*config).to_model_config() {
        Ok(c) => c,
        Err(s) => return s as i32,
    };

    match CpuInferenceBackend::init(&cfg) {
        Ok(backend) => {
            *out = Box::into_raw(Box::new(backend)) as GllmBackend;
            GllmStatus::Ok as i32
        }
        Err(e) => GllmStatus::from(e) as i32,
    }
}

/// Destroy a backend handle.
///
/// # Safety
/// `backend` must be a handle returned by `gllm_backend_init`, or null.
#[no_mangle]
pub unsafe extern "C" fn gllm_backend_free(backend: GllmBackend) {
    if !backend.is_null() {
        drop(Box::from_raw(backend as *mut CpuInferenceBackend));
    }
}

// ── Tensor operations ───────────────────────────────────────────────

/// Allocate a tensor.
///
/// # Safety
/// `backend` must be a valid handle. `out` must point to a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn gllm_tensor_alloc(
    backend: GllmBackend,
    num_elements: usize,
    dtype: i32,
    out: *mut GllmTensor,
) -> i32 {
    if backend.is_null() || out.is_null() {
        return GllmStatus::InvalidArg as i32;
    }

    let b = &*(backend as *const CpuInferenceBackend);
    let dt = match dtype {
        0 => crate::inference::types::DType::F32,
        1 => crate::inference::types::DType::F16,
        2 => crate::inference::types::DType::BF16,
        _ => return GllmStatus::InvalidArg as i32,
    };

    match b.alloc(num_elements, dt) {
        Ok(tensor) => {
            *out = Box::into_raw(Box::new(tensor)) as GllmTensor;
            GllmStatus::Ok as i32
        }
        Err(e) => GllmStatus::from(e) as i32,
    }
}

/// Free a tensor.
///
/// # Safety
/// `tensor` must be a handle returned by `gllm_tensor_alloc`, or null.
#[no_mangle]
pub unsafe extern "C" fn gllm_tensor_free(tensor: GllmTensor) {
    if !tensor.is_null() {
        drop(Box::from_raw(tensor as *mut DeviceTensor));
    }
}

/// Upload f32 data to a tensor.
///
/// # Safety
/// `src` must point to `num_elements` valid f32 values.
#[no_mangle]
pub unsafe extern "C" fn gllm_tensor_upload_f32(
    backend: GllmBackend,
    src: *const f32,
    num_elements: usize,
    tensor: GllmTensor,
) -> i32 {
    if backend.is_null() || src.is_null() || tensor.is_null() {
        return GllmStatus::InvalidArg as i32;
    }

    let b = &*(backend as *const CpuInferenceBackend);
    let t = &mut *(tensor as *mut DeviceTensor);
    let slice = std::slice::from_raw_parts(src, num_elements);

    match b.upload_f32(slice, t) {
        Ok(()) => GllmStatus::Ok as i32,
        Err(e) => GllmStatus::from(e) as i32,
    }
}

/// Download f32 data from a tensor.
///
/// # Safety
/// `dst` must point to space for at least `num_elements` f32 values.
#[no_mangle]
pub unsafe extern "C" fn gllm_tensor_download_f32(
    backend: GllmBackend,
    tensor: GllmTensor,
    dst: *mut f32,
    num_elements: usize,
) -> i32 {
    if backend.is_null() || tensor.is_null() || dst.is_null() {
        return GllmStatus::InvalidArg as i32;
    }

    let b = &*(backend as *const CpuInferenceBackend);
    let t = &*(tensor as *const DeviceTensor);
    let slice = std::slice::from_raw_parts_mut(dst, num_elements);

    match b.download_f32(t, slice) {
        Ok(()) => GllmStatus::Ok as i32,
        Err(e) => GllmStatus::from(e) as i32,
    }
}

// ── KV Cache ────────────────────────────────────────────────────────

/// Allocate a KV cache.
///
/// # Safety
/// `backend` must be valid. `out` must point to a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn gllm_kv_cache_alloc(
    backend: GllmBackend,
    batch_size: usize,
    max_seq_len: usize,
    out: *mut GllmKvCache,
) -> i32 {
    if backend.is_null() || out.is_null() {
        return GllmStatus::InvalidArg as i32;
    }

    let b = &*(backend as *const CpuInferenceBackend);
    match b.alloc_kv_cache(batch_size, max_seq_len) {
        Ok(cache) => {
            *out = Box::into_raw(Box::new(cache)) as GllmKvCache;
            GllmStatus::Ok as i32
        }
        Err(e) => GllmStatus::from(e) as i32,
    }
}

/// Free a KV cache.
///
/// # Safety
/// `cache` must be a handle returned by `gllm_kv_cache_alloc`, or null.
#[no_mangle]
pub unsafe extern "C" fn gllm_kv_cache_free(cache: GllmKvCache) {
    if !cache.is_null() {
        drop(Box::from_raw(cache as *mut KvCache));
    }
}

// ── Version info ────────────────────────────────────────────────────

/// Get the library version string.
///
/// Returns a pointer to a static null-terminated string.
#[no_mangle]
pub extern "C" fn gllm_version() -> *const std::ffi::c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const std::ffi::c_char
}

/// Get detected hardware info as a string.
///
/// Writes up to `buf_len` bytes into `buf`. Returns the number of bytes written.
///
/// # Safety
/// `buf` must point to at least `buf_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn gllm_hw_info(buf: *mut u8, buf_len: usize) -> usize {
    if buf.is_null() || buf_len == 0 {
        return 0;
    }

    let profile = crate::dispatch::device_profile();
    let info = format!("{profile}");
    let bytes = info.as_bytes();
    let copy_len = bytes.len().min(buf_len - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, copy_len);
    *buf.add(copy_len) = 0; // null terminate
    copy_len
}
