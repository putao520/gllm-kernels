//! C ABI wrapper for gllm-kernels inference backend.
//!
//! Provides `#[no_mangle] extern "C"` functions for use from C/C++/Python.
//! All functions return `GllmStatus` error codes and use opaque handles.

pub mod types;

pub use types::{
    GllmStatus, GllmBackend, GllmTensor, GllmKvCache, GllmWeights, GllmModelConfig,
    GllmWeightField,
};

use crate::inference::cpu_backend::CpuInferenceBackend;
use crate::inference::tensor::DeviceTensor;
use crate::inference::kv_cache::KvCache;
use crate::inference::weights::ModelWeights;
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

// ── Model weights ────────────────────────────────────────────────────

/// Allocate zeroed model weights for the given configuration.
///
/// The caller must populate the weight data via `gllm_weights_get_ptr`
/// before calling `gllm_decoder_forward`.
///
/// # Safety
/// `config` must point to a valid `GllmModelConfig`.
/// `out` must point to a valid `GllmWeights` pointer.
#[no_mangle]
pub unsafe extern "C" fn gllm_weights_alloc(
    config: *const GllmModelConfig,
    out: *mut GllmWeights,
) -> i32 {
    if config.is_null() || out.is_null() {
        return GllmStatus::InvalidArg as i32;
    }

    let cfg = match (*config).to_model_config() {
        Ok(c) => c,
        Err(s) => return s as i32,
    };

    match ModelWeights::alloc_cpu(&cfg) {
        Ok(weights) => {
            *out = Box::into_raw(Box::new(weights)) as GllmWeights;
            GllmStatus::Ok as i32
        }
        Err(e) => GllmStatus::from(e) as i32,
    }
}

/// Free model weights.
///
/// # Safety
/// `weights` must be a handle returned by `gllm_weights_alloc`, or null.
#[no_mangle]
pub unsafe extern "C" fn gllm_weights_free(weights: GllmWeights) {
    if !weights.is_null() {
        drop(Box::from_raw(weights as *mut ModelWeights));
    }
}

/// Get a mutable pointer to a weight tensor's raw data for direct writing.
///
/// `field` identifies which weight tensor (see `GllmWeightField` constants).
/// `layer_idx` is required for per-layer fields (field >= 10), ignored for global fields.
///
/// On success, writes the data pointer to `*out_ptr` and the element count to
/// `*out_num_elements`, then returns `GLLM_OK`.
///
/// Returns `GLLM_INVALID_ARG` if the field is unknown, the layer index is out
/// of range, or the requested optional field (e.g. QKV bias) is absent.
///
/// # Safety
/// `weights` must be a valid handle. `out_ptr` and `out_num_elements` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn gllm_weights_get_ptr(
    weights: GllmWeights,
    field: i32,
    layer_idx: i32,
    out_ptr: *mut *mut u8,
    out_num_elements: *mut usize,
) -> i32 {
    if weights.is_null() || out_ptr.is_null() || out_num_elements.is_null() {
        return GllmStatus::InvalidArg as i32;
    }

    let wf = match GllmWeightField::from_i32(field) {
        Some(f) => f,
        None => return GllmStatus::InvalidArg as i32,
    };

    let w = &mut *(weights as *mut ModelWeights);

    // Helper: extract ptr + num_elements from a DeviceTensor
    macro_rules! tensor_out {
        ($tensor:expr) => {{
            let t: &mut DeviceTensor = $tensor;
            *out_ptr = t.as_mut_ptr();
            *out_num_elements = t.num_elements();
            GllmStatus::Ok as i32
        }};
    }

    // Global fields
    if wf.is_global() {
        return match wf {
            GllmWeightField::Embedding => tensor_out!(&mut w.embedding),
            GllmWeightField::FinalNorm => tensor_out!(&mut w.final_norm),
            GllmWeightField::LmHead => tensor_out!(&mut w.lm_head),
            _ => GllmStatus::InvalidArg as i32,
        };
    }

    // Per-layer fields
    let li = layer_idx as usize;
    if li >= w.layers.len() {
        return GllmStatus::InvalidArg as i32;
    }
    let lw = &mut w.layers[li];

    match wf {
        GllmWeightField::AttnNorm => tensor_out!(&mut lw.attn_norm),
        GllmWeightField::Wq => tensor_out!(&mut lw.wq),
        GllmWeightField::Wk => tensor_out!(&mut lw.wk),
        GllmWeightField::Wv => tensor_out!(&mut lw.wv),
        GllmWeightField::Wo => tensor_out!(&mut lw.wo),
        GllmWeightField::FfnNorm => tensor_out!(&mut lw.ffn_norm),
        GllmWeightField::WGate => tensor_out!(&mut lw.w_gate),
        GllmWeightField::WUp => tensor_out!(&mut lw.w_up),
        GllmWeightField::WDown => tensor_out!(&mut lw.w_down),
        GllmWeightField::QkvBias => {
            match lw.qkv_bias {
                Some(ref mut t) => tensor_out!(t),
                None => GllmStatus::InvalidArg as i32,
            }
        }
        GllmWeightField::AttnNormBias => tensor_out!(&mut lw.attn_norm_bias),
        GllmWeightField::FfnNormBias => tensor_out!(&mut lw.ffn_norm_bias),
        _ => GllmStatus::InvalidArg as i32,
    }
}

// ── Inference ────────────────────────────────────────────────────────

/// Run decoder forward pass (all transformer layers).
///
/// Computes the full decoder stack: for each layer, applies attention norm,
/// QKV projection, RoPE, KV cache update, multi-head attention, output
/// projection, residual, FFN norm, gated FFN, and residual.
///
/// `input`:     opaque tensor handle, shape [batch_size, seq_len, hidden_size]
/// `positions`: opaque tensor handle, shape [batch_size, seq_len] (f32 position indices)
/// `kv_cache`:  opaque KV cache handle (mutated in place)
/// `weights`:   opaque model weights handle
/// `seq_lens`:  pointer to per-sequence lengths array
/// `num_seqs`:  number of sequences (length of `seq_lens`)
/// `output`:    opaque tensor handle, shape [batch_size, seq_len, hidden_size]
///
/// Returns `GLLM_OK` on success.
///
/// # Safety
/// All handles must be valid and non-null. `seq_lens` must point to `num_seqs` elements.
#[no_mangle]
pub unsafe extern "C" fn gllm_decoder_forward(
    backend: GllmBackend,
    input: GllmTensor,
    positions: GllmTensor,
    kv_cache: GllmKvCache,
    weights: GllmWeights,
    seq_lens: *const usize,
    num_seqs: usize,
    output: GllmTensor,
) -> i32 {
    if backend.is_null()
        || input.is_null()
        || positions.is_null()
        || kv_cache.is_null()
        || weights.is_null()
        || seq_lens.is_null()
        || output.is_null()
    {
        return GllmStatus::InvalidArg as i32;
    }

    let b = &*(backend as *const CpuInferenceBackend);
    let inp = &*(input as *const DeviceTensor);
    let pos = &*(positions as *const DeviceTensor);
    let kv = &mut *(kv_cache as *mut KvCache);
    let w = &*(weights as *const ModelWeights);
    let sl = std::slice::from_raw_parts(seq_lens, num_seqs);
    let out = &mut *(output as *mut DeviceTensor);

    match b.decoder_forward(inp, pos, kv, w, sl, out) {
        Ok(()) => GllmStatus::Ok as i32,
        Err(e) => GllmStatus::from(e) as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny GllmModelConfig for testing.
    fn tiny_c_config() -> GllmModelConfig {
        GllmModelConfig {
            arch: 0,              // Llama
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            num_layers: 1,
            vocab_size: 10,
            max_seq_len: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: 0,             // F32
            quant_type: -1,       // None
            has_qkv_bias: 0,
            partial_rotary_factor: 1.0,
        }
    }

    #[test]
    fn test_ffi_weights_lifecycle() {
        unsafe {
            let cfg = tiny_c_config();
            let mut weights: GllmWeights = std::ptr::null_mut();
            let rc = gllm_weights_alloc(&cfg, &mut weights);
            assert_eq!(rc, GllmStatus::Ok as i32);
            assert!(!weights.is_null());

            // Access a global field
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut n: usize = 0;
            let rc = gllm_weights_get_ptr(weights, GllmWeightField::Embedding as i32, 0, &mut ptr, &mut n);
            assert_eq!(rc, GllmStatus::Ok as i32);
            assert_eq!(n, 10 * 8); // vocab_size * hidden_size

            // Access a per-layer field
            let rc = gllm_weights_get_ptr(weights, GllmWeightField::Wq as i32, 0, &mut ptr, &mut n);
            assert_eq!(rc, GllmStatus::Ok as i32);
            assert_eq!(n, 8 * 8); // hidden_size * (num_heads * head_dim)

            // Invalid layer index
            let rc = gllm_weights_get_ptr(weights, GllmWeightField::Wq as i32, 99, &mut ptr, &mut n);
            assert_eq!(rc, GllmStatus::InvalidArg as i32);

            // Invalid field
            let rc = gllm_weights_get_ptr(weights, 999, 0, &mut ptr, &mut n);
            assert_eq!(rc, GllmStatus::InvalidArg as i32);

            // QkvBias absent for Llama
            let rc = gllm_weights_get_ptr(weights, GllmWeightField::QkvBias as i32, 0, &mut ptr, &mut n);
            assert_eq!(rc, GllmStatus::InvalidArg as i32);

            gllm_weights_free(weights);
        }
    }

    #[test]
    fn test_ffi_decoder_forward_roundtrip() {
        unsafe {
            let cfg = tiny_c_config();

            // Init backend
            let mut backend: GllmBackend = std::ptr::null_mut();
            assert_eq!(gllm_backend_init(&cfg, &mut backend), GllmStatus::Ok as i32);

            // Alloc weights and fill norm weights with 1.0
            let mut weights: GllmWeights = std::ptr::null_mut();
            assert_eq!(gllm_weights_alloc(&cfg, &mut weights), GllmStatus::Ok as i32);

            let h = cfg.hidden_size as usize;
            let ones = vec![1.0f32; h];
            for field in [GllmWeightField::AttnNorm, GllmWeightField::FfnNorm] {
                let mut ptr: *mut u8 = std::ptr::null_mut();
                let mut n: usize = 0;
                let rc = gllm_weights_get_ptr(weights, field as i32, 0, &mut ptr, &mut n);
                assert_eq!(rc, GllmStatus::Ok as i32);
                std::ptr::copy_nonoverlapping(ones.as_ptr() as *const u8, ptr, h * 4);
            }

            // Alloc input tensor and upload data
            let mut input: GllmTensor = std::ptr::null_mut();
            assert_eq!(gllm_tensor_alloc(backend, h, 0, &mut input), GllmStatus::Ok as i32);
            let input_data = vec![0.1f32; h];
            assert_eq!(
                gllm_tensor_upload_f32(backend, input_data.as_ptr(), h, input),
                GllmStatus::Ok as i32,
            );

            // Alloc positions tensor
            let mut positions: GllmTensor = std::ptr::null_mut();
            assert_eq!(gllm_tensor_alloc(backend, 1, 0, &mut positions), GllmStatus::Ok as i32);
            let pos_data = [0.0f32];
            assert_eq!(
                gllm_tensor_upload_f32(backend, pos_data.as_ptr(), 1, positions),
                GllmStatus::Ok as i32,
            );

            // Alloc KV cache
            let mut kv_cache: GllmKvCache = std::ptr::null_mut();
            assert_eq!(
                gllm_kv_cache_alloc(backend, 1, cfg.max_seq_len as usize, &mut kv_cache),
                GllmStatus::Ok as i32,
            );

            // Alloc output tensor
            let mut output: GllmTensor = std::ptr::null_mut();
            assert_eq!(gllm_tensor_alloc(backend, h, 0, &mut output), GllmStatus::Ok as i32);

            // Run decoder forward
            let seq_lens = [1usize];
            let rc = gllm_decoder_forward(
                backend,
                input,
                positions,
                kv_cache,
                weights,
                seq_lens.as_ptr(),
                1,
                output,
            );
            assert_eq!(rc, GllmStatus::Ok as i32);

            // Download and verify output is finite and non-trivial
            let mut out_data = vec![0.0f32; h];
            assert_eq!(
                gllm_tensor_download_f32(backend, output, out_data.as_mut_ptr(), h),
                GllmStatus::Ok as i32,
            );
            assert!(out_data.iter().all(|v| v.is_finite()), "output has non-finite values");

            // Cleanup
            gllm_tensor_free(input);
            gllm_tensor_free(positions);
            gllm_tensor_free(output);
            gllm_kv_cache_free(kv_cache);
            gllm_weights_free(weights);
            gllm_backend_free(backend);
        }
    }

    #[test]
    fn test_ffi_decoder_forward_null_checks() {
        unsafe {
            let seq_lens = [1usize];

            // All-null should return InvalidArg
            let rc = gllm_decoder_forward(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            );
            assert_eq!(rc, GllmStatus::InvalidArg as i32);

            // Null seq_lens with valid-looking (but fake) handles
            let fake = 1usize as *mut std::ffi::c_void;
            let rc = gllm_decoder_forward(
                fake, fake, fake, fake, fake,
                std::ptr::null(),
                1,
                fake,
            );
            assert_eq!(rc, GllmStatus::InvalidArg as i32);
        }
    }

    #[test]
    fn test_ffi_weights_null_checks() {
        unsafe {
            let cfg = tiny_c_config();
            let mut weights: GllmWeights = std::ptr::null_mut();

            // Null config
            assert_eq!(
                gllm_weights_alloc(std::ptr::null(), &mut weights),
                GllmStatus::InvalidArg as i32,
            );

            // Null out
            assert_eq!(
                gllm_weights_alloc(&cfg, std::ptr::null_mut()),
                GllmStatus::InvalidArg as i32,
            );

            // Null weights handle for get_ptr
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut n: usize = 0;
            assert_eq!(
                gllm_weights_get_ptr(std::ptr::null_mut(), 0, 0, &mut ptr, &mut n),
                GllmStatus::InvalidArg as i32,
            );

            // Free null is safe (no-op)
            gllm_weights_free(std::ptr::null_mut());
        }
    }
}
