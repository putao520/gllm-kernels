use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::traits::{Backend, Element, Kernels};
use crate::quant::{QK_K, QuantType};
use half::f16;

// ==========================================================================
// IQ4_NL non-linear lookup table (from llama.cpp ggml-common.h)
// ==========================================================================

/// Pre-computed f32 IQ4_NL codebook. Each 4-bit nibble indexes into this
/// table to produce a dequantized weight value (before scaling by d).
/// Raw i8 values: {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}
const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
];

// ==========================================================================
// IQ stub helpers
// ==========================================================================

/// Per-format one-shot warning flags for IQ grid-based formats awaiting
/// codebook table embedding.
static IQ1S_WARNED: AtomicBool = AtomicBool::new(false);
static IQ1M_WARNED: AtomicBool = AtomicBool::new(false);
static IQ2XXS_WARNED: AtomicBool = AtomicBool::new(false);
static IQ2XS_WARNED: AtomicBool = AtomicBool::new(false);
static IQ2S_WARNED: AtomicBool = AtomicBool::new(false);
static IQ3XXS_WARNED: AtomicBool = AtomicBool::new(false);
static IQ3S_WARNED: AtomicBool = AtomicBool::new(false);

/// Non-panicking stub for IQ grid-based formats whose E8/D4 lattice codebook
/// tables are not yet embedded. Outputs zeros and emits a one-shot warning
/// via eprintln. This ensures downstream code never panics on encountering
/// these quant types.
fn iq_stub_dequant(name: &str, block: &[u8], out: &mut [f32], block_bytes: usize, block_size: usize) {
    let warned_flag = match name {
        "IQ1_S" => &IQ1S_WARNED,
        "IQ1_M" => &IQ1M_WARNED,
        "IQ2_XXS" => &IQ2XXS_WARNED,
        "IQ2_XS" => &IQ2XS_WARNED,
        "IQ2_S" => &IQ2S_WARNED,
        "IQ3_XXS" => &IQ3XXS_WARNED,
        "IQ3_S" => &IQ3S_WARNED,
        _ => &IQ1S_WARNED, // fallback, should not happen
    };
    if !warned_flag.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[gllm-kernels] WARNING: dequant_{} outputs zeros — \
             E8/D4 lattice codebook tables not yet embedded. \
             IQ4_NL and IQ4_XS are fully functional.",
            name
        );
    }
    let num_blocks = block.len() / block_bytes;
    let total = num_blocks * block_size;
    let len = out.len();
    for v in out[..total.min(len)].iter_mut() {
        *v = 0.0;
    }
}

/// Scalar CPU kernel implementation.
///
/// This is the reference implementation that uses pure scalar operations.
/// The JIT compiler (Phase 3) generates optimized SIMD code at runtime;
/// this scalar path serves as the correctness baseline and fallback.
pub struct CpuKernels<E: Element> {
    _phantom: PhantomData<E>,
}

impl<E: Element> CpuKernels<E> {
    pub fn new() -> Self {
        CpuKernels { _phantom: PhantomData }
    }
}

impl<E: Element> Kernels<E> for CpuKernels<E> {
    // ── BLAS-1 ──

    fn vec_add(&self, a: &[E], b: &[E], out: &mut [E]) {
        for i in 0..out.len() {
            out[i] = a[i].elem_add(b[i]);
        }
    }

    fn vec_mul(&self, a: &[E], b: &[E], out: &mut [E]) {
        for i in 0..out.len() {
            out[i] = a[i].elem_mul(b[i]);
        }
    }

    fn vec_dot(&self, a: &[E], b: &[E]) -> E {
        let mut acc = E::ZERO;
        for i in 0..a.len() {
            acc = acc.elem_add(a[i].elem_mul(b[i]));
        }
        acc
    }

    fn vec_sub(&self, a: &[E], b: &[E], out: &mut [E]) {
        for i in 0..out.len() {
            out[i] = a[i].elem_sub(b[i]);
        }
    }

    fn vec_scale(&self, x: &mut [E], s: E) {
        for v in x.iter_mut() {
            *v = v.elem_mul(s);
        }
    }

    fn vec_axpy(&self, y: &mut [E], a: E, x: &[E]) {
        for i in 0..y.len() {
            y[i] = y[i].elem_add(a.elem_mul(x[i]));
        }
    }

    fn vec_sum(&self, x: &[E]) -> E {
        let mut acc = E::ZERO;
        for &v in x {
            acc = acc.elem_add(v);
        }
        acc
    }

    fn vec_max(&self, x: &[E]) -> E {
        let mut m = x[0];
        for &v in &x[1..] {
            m = m.max(v);
        }
        m
    }

    fn vec_sum_squares(&self, x: &[E]) -> E {
        let mut acc = E::ZERO;
        for &v in x {
            acc = acc.elem_add(v.elem_mul(v));
        }
        acc
    }

    // ── BLAS-2/3 ──

    fn gemv(&self, a: &[E], x: &[E], y: &mut [E], m: usize, n: usize) {
        for i in 0..m {
            let mut acc = E::ZERO;
            for j in 0..n {
                acc = acc.elem_add(a[i * n + j].elem_mul(x[j]));
            }
            y[i] = acc;
        }
    }

    fn gemm(&self, a: &[E], b: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = E::ZERO;
                for p in 0..k {
                    acc = acc.elem_add(a[i * k + p].elem_mul(b[p * n + j]));
                }
                c[i * n + j] = acc;
            }
        }
    }

    fn gemm_bt(&self, a: &[E], b_t: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = E::ZERO;
                for p in 0..k {
                    acc = acc.elem_add(a[i * k + p].elem_mul(b_t[j * k + p]));
                }
                c[i * n + j] = acc;
            }
        }
    }

    fn gemm_bias(&self, a: &[E], b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        self.gemm(a, b, c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = c[i * n + j].elem_add(bias[j]);
            }
        }
    }

    fn pack_b(&self, b: &[E], n: usize, k: usize) -> Vec<E> {
        // Identity packing for scalar path
        b[..n * k].to_vec()
    }

    fn gemm_prepacked(&self, a: &[E], packed_b: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        self.gemm(a, packed_b, c, m, n, k);
    }

    fn gemm_bias_prepacked(&self, a: &[E], packed_b: &[E], bias: &[E], c: &mut [E], m: usize, n: usize, k: usize) {
        self.gemm_bias(a, packed_b, bias, c, m, n, k);
    }

    // ── Activations ──

    fn silu(&self, a: &[E], out: &mut [E]) {
        for i in 0..a.len() {
            // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            let x = a[i];
            let neg_x = x.neg();
            let exp_neg = neg_x.exp();
            let denom = E::ONE.elem_add(exp_neg);
            out[i] = x.elem_div(denom);
        }
    }

    fn gelu(&self, x: &[E], out: &mut [E]) {
        // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = E::from_f32(0.7978845608);
        let coeff = E::from_f32(0.044715);
        let half = E::from_f32(0.5);
        for i in 0..x.len() {
            let v = x[i];
            let v3 = v.elem_mul(v).elem_mul(v);
            let inner = sqrt_2_over_pi.elem_mul(v.elem_add(coeff.elem_mul(v3)));
            let t = inner.tanh();
            out[i] = half.elem_mul(v).elem_mul(E::ONE.elem_add(t));
        }
    }

    fn relu(&self, x: &[E], out: &mut [E]) {
        for i in 0..x.len() {
            out[i] = x[i].max(E::ZERO);
        }
    }

    fn tanh(&self, x: &[E], out: &mut [E]) {
        for i in 0..x.len() {
            out[i] = x[i].tanh();
        }
    }

    fn swiglu(&self, gate: &[E], up: &[E], out: &mut [E]) {
        for i in 0..gate.len() {
            let g = gate[i];
            let neg_g = g.neg();
            let sigmoid = E::ONE.elem_div(E::ONE.elem_add(neg_g.exp()));
            out[i] = g.elem_mul(sigmoid).elem_mul(up[i]);
        }
    }

    fn softmax(&self, x: &[E], out: &mut [E]) {
        let n = x.len();
        let mut max_val = x[0];
        for i in 1..n {
            max_val = max_val.max(x[i]);
        }
        let mut sum = E::ZERO;
        for i in 0..n {
            let e = x[i].elem_sub(max_val).exp();
            out[i] = e;
            sum = sum.elem_add(e);
        }
        let inv_sum = sum.recip();
        for i in 0..n {
            out[i] = out[i].elem_mul(inv_sum);
        }
    }

    fn exp(&self, x: &[E], out: &mut [E]) {
        for i in 0..x.len() {
            out[i] = x[i].exp();
        }
    }

    // ── Normalization ──

    fn rms_norm(&self, x: &[E], weight: &[E], out: &mut [E], eps: f32) {
        let n = x.len();
        let mut sum_sq = E::ZERO;
        for i in 0..n {
            sum_sq = sum_sq.elem_add(x[i].elem_mul(x[i]));
        }
        let mean_sq = sum_sq.to_f32() / n as f32;
        let inv_rms = E::from_f32(1.0 / (mean_sq + eps).sqrt());
        for i in 0..n {
            out[i] = x[i].elem_mul(inv_rms).elem_mul(weight[i]);
        }
    }

    fn layer_norm(&self, x: &[E], gamma: &[E], beta: &[E], out: &mut [E], eps: f32) {
        let n = x.len();
        let mut sum = E::ZERO;
        for i in 0..n {
            sum = sum.elem_add(x[i]);
        }
        let mean = sum.to_f32() / n as f32;
        let mut var_sum = 0.0f32;
        for i in 0..n {
            let d = x[i].to_f32() - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();
        for i in 0..n {
            let normed = (x[i].to_f32() - mean) * inv_std;
            out[i] = E::from_f32(normed).elem_mul(gamma[i]).elem_add(beta[i]);
        }
    }

    // ── Positional encoding ──

    fn rope(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, interleaved: bool) {
        let half = head_dim / 2;
        if interleaved {
            // Interleaved: pairs are (qk[2i], qk[2i+1])
            for i in (0..qk.len()).step_by(head_dim) {
                for j in 0..half {
                    let x0 = qk[i + 2 * j];
                    let x1 = qk[i + 2 * j + 1];
                    let c = cos[j];
                    let s = sin[j];
                    qk[i + 2 * j] = x0.elem_mul(c).elem_sub(x1.elem_mul(s));
                    qk[i + 2 * j + 1] = x0.elem_mul(s).elem_add(x1.elem_mul(c));
                }
            }
        } else {
            // Non-interleaved: pairs are (qk[j], qk[j+half])
            for i in (0..qk.len()).step_by(head_dim) {
                for j in 0..half {
                    let x0 = qk[i + j];
                    let x1 = qk[i + j + half];
                    let c = cos[j];
                    let s = sin[j];
                    qk[i + j] = x0.elem_mul(c).elem_sub(x1.elem_mul(s));
                    qk[i + j + half] = x0.elem_mul(s).elem_add(x1.elem_mul(c));
                }
            }
        }
    }

    fn rope_with_pos(&self, qk: &mut [E], cos: &[E], sin: &[E], head_dim: usize, position: usize, interleaved: bool) {
        let half = head_dim / 2;
        if interleaved {
            for i in (0..qk.len()).step_by(head_dim) {
                for j in 0..half {
                    let x0 = qk[i + 2 * j];
                    let x1 = qk[i + 2 * j + 1];
                    let idx = position * half + j;
                    let c = cos[idx];
                    let s = sin[idx];
                    qk[i + 2 * j] = x0.elem_mul(c).elem_sub(x1.elem_mul(s));
                    qk[i + 2 * j + 1] = x0.elem_mul(s).elem_add(x1.elem_mul(c));
                }
            }
        } else {
            for i in (0..qk.len()).step_by(head_dim) {
                for j in 0..half {
                    let x0 = qk[i + j];
                    let x1 = qk[i + j + half];
                    let idx = position * half + j;
                    let c = cos[idx];
                    let s = sin[idx];
                    qk[i + j] = x0.elem_mul(c).elem_sub(x1.elem_mul(s));
                    qk[i + j + half] = x0.elem_mul(s).elem_add(x1.elem_mul(c));
                }
            }
        }
    }

    // ── Dequantization ──

    fn dequant_q4_k(&self, block: &[u8], out: &mut [f32]) {
        // Q4_K: 256 values per block, block_size = 144 bytes
        // Layout: d(f16) + dmin(f16) + scales(12 bytes) + qs(128 bytes)
        const BLOCK_SIZE: usize = 144;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([b[2], b[3]]).to_f32();
            let scales = &b[4..16];
            let qs = &b[16..144];
            let out_block = &mut out[bi * QK_K..(bi + 1) * QK_K];

            for j in 0..QK_K / 64 {
                let sc_idx = j;
                let sc = if sc_idx < 4 {
                    (scales[sc_idx] & 0x3F) as f32
                } else {
                    ((scales[sc_idx + 4] & 0xF) | ((scales[sc_idx - 4] >> 6) << 4)) as f32
                };
                let m = if sc_idx < 4 {
                    (scales[sc_idx + 4] & 0x3F) as f32
                } else {
                    ((scales[sc_idx + 4] >> 4) | ((scales[sc_idx - 0] >> 6) << 4)) as f32
                };
                let d_sc = d * sc;
                let d_m = dmin * m;
                for l in 0..32 {
                    let q = qs[j * 32 + l];
                    out_block[j * 64 + l] = d_sc * (q & 0xF) as f32 - d_m;
                    out_block[j * 64 + l + 32] = d_sc * (q >> 4) as f32 - d_m;
                }
            }
        }
    }

    fn dequant_q8_k(&self, block: &[u8], out: &mut [f32]) {
        // Q8_K: 256 values per block, block_size = 292 bytes
        // Layout: d(f32) + qs(256 i8) + bsums(32 i16)
        const BLOCK_SIZE: usize = 292;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let qs = &b[4..260];
            let out_block = &mut out[bi * QK_K..(bi + 1) * QK_K];
            for i in 0..QK_K {
                out_block[i] = d * (qs[i] as i8) as f32;
            }
        }
    }

    // ── K-Quant dequantization (optional) ──

    fn dequant_q2_k(&self, block: &[u8], out: &mut [f32]) {
        const BLOCK_SIZE: usize = 84;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let scales = &b[0..16];
            let qs = &b[16..80];
            let d = f16::from_le_bytes([b[80], b[81]]).to_f32();
            let dmin = f16::from_le_bytes([b[82], b[83]]).to_f32();
            let out_block = &mut out[bi * QK_K..(bi + 1) * QK_K];
            for j in 0..QK_K / 16 {
                let sc = (scales[j] & 0xF) as f32;
                let m = (scales[j] >> 4) as f32;
                let d_sc = d * sc;
                let d_m = dmin * m;
                for l in 0..16 {
                    let idx = j * 16 + l;
                    let byte_idx = idx / 4;
                    let shift = (idx % 4) * 2;
                    let q = ((qs[byte_idx] >> shift) & 0x03) as f32;
                    out_block[idx] = d_sc * q - d_m;
                }
            }
        }
    }

    fn dequant_q3_k(&self, block: &[u8], out: &mut [f32]) {
        const BLOCK_SIZE: usize = 110;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let hmask = &b[0..32];
            let qs = &b[32..96];
            let raw_scales = &b[96..108];
            let d = f16::from_le_bytes([b[108], b[109]]).to_f32();
            let out_block = &mut out[bi * QK_K..(bi + 1) * QK_K];

            // Unpack 12 raw bytes → 16 x 6-bit scales
            let mut unpacked_scales = [0u8; 16];
            for i in 0..8 {
                unpacked_scales[i] = (raw_scales[i] & 0x0F) | ((raw_scales[8 + i / 2] >> ((i % 2) * 4)) & 0x03) << 4;
            }
            for i in 0..8 {
                unpacked_scales[8 + i] = (raw_scales[i] >> 4) | ((raw_scales[8 + i / 2] >> ((i % 2) * 4 + 2)) & 0x03) << 4;
            }

            for j in 0..QK_K / 16 {
                let sc = unpacked_scales[j] as i8 as f32;
                let dl = d * (sc - 32.0);
                for l in 0..16 {
                    let idx = j * 16 + l;
                    let byte_idx = idx / 4;
                    let shift = (idx % 4) * 2;
                    let q_lo = ((qs[byte_idx] >> shift) & 0x03) as i32;
                    let hbit = if (hmask[idx % 32] >> (idx / 32)) & 1 != 0 { 0 } else { 4 };
                    out_block[idx] = dl * (q_lo + hbit) as f32;
                }
            }
        }
    }

    fn dequant_q5_k(&self, block: &[u8], out: &mut [f32]) {
        // BlockQ5K layout: d(f16)@0, dmin(f16)@2, scales([u8;12])@4, qh([u8;32])@16, qs([u8;128])@48
        const BLOCK_SIZE: usize = 176;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
            let scales = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..176];
            let out_block = &mut out[bi * QK_K..(bi + 1) * QK_K];

            for j in 0..QK_K / 64 {
                let sc_idx = j;
                let sc = if sc_idx < 4 {
                    (scales[sc_idx] & 0x3F) as f32
                } else {
                    ((scales[sc_idx + 4] & 0xF) | ((scales[sc_idx - 4] >> 6) << 4)) as f32
                };
                let m = if sc_idx < 4 {
                    (scales[sc_idx + 4] & 0x3F) as f32
                } else {
                    ((scales[sc_idx + 4] >> 4) | ((scales[sc_idx - 0] >> 6) << 4)) as f32
                };
                let d_sc = d * sc;
                let d_m = dmin * m;
                for l in 0..32 {
                    let q_lo = qs[j * 32 + l] & 0xF;
                    let u1 = (qh[l] >> j) & 1;
                    let val = (q_lo as u32 + (u1 as u32) * 16) as f32;
                    out_block[j * 64 + l] = d_sc * val - d_m;
                }
                for l in 0..32 {
                    let q_hi = qs[j * 32 + l] >> 4;
                    let u2 = (qh[l] >> (j + 4)) & 1;
                    let val = (q_hi as u32 + (u2 as u32) * 16) as f32;
                    out_block[j * 64 + l + 32] = d_sc * val - d_m;
                }
            }
        }
    }

    fn dequant_q6_k(&self, block: &[u8], out: &mut [f32]) {
        const BLOCK_SIZE: usize = 210;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let ql = &b[0..128];
            let qh_bytes = &b[128..192];
            let scales = &b[192..208];
            let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
            let out_block = &mut out[bi * QK_K..(bi + 1) * QK_K];

            for j in 0..QK_K / 16 {
                let is = j;
                let sc = scales[is] as i8 as f32;
                for l in 0..16 {
                    let idx = j * 16 + l;
                    let ql_byte = ql[idx % 128];
                    let shift_lo = if idx < 128 { 0 } else { 4 };
                    let q_lo = (ql_byte >> shift_lo) & 0xF;
                    let qh_byte = qh_bytes[idx % 64];
                    let shift_hi = (idx / 64) * 2;
                    let q_hi = (qh_byte >> shift_hi) & 0x03;
                    let q = ((q_lo as u8) | (q_hi << 4)) as i8 as i32 - 32;
                    out_block[idx] = d * sc * q as f32;
                }
            }
        }
    }

    // ── Classic GGML dequantization (block_size=32) ──

    fn dequant_q4_0(&self, block: &[u8], out: &mut [f32]) {
        // BlockQ4_0 layout: d(f16)@0, qs([u8;16])@2
        // Element i: qs[i/2] lo/hi nibble, subtract 8
        const BLOCK_SIZE: usize = 18;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let out_block = &mut out[bi * 32..(bi + 1) * 32];
            for i in 0..32 {
                let byte = b[2 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                out_block[i] = d * (nibble as f32 - 8.0);
            }
        }
    }

    fn dequant_q4_1(&self, block: &[u8], out: &mut [f32]) {
        // BlockQ4_1 layout: d(f16)@0, m(f16)@2, qs([u8;16])@4
        // Element i: qs[i/2] lo/hi nibble
        const BLOCK_SIZE: usize = 20;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let m = f16::from_le_bytes([b[2], b[3]]).to_f32();
            let out_block = &mut out[bi * 32..(bi + 1) * 32];
            for i in 0..32 {
                let byte = b[4 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                out_block[i] = d * nibble as f32 + m;
            }
        }
    }

    fn dequant_q5_0(&self, block: &[u8], out: &mut [f32]) {
        // BlockQ5_0 layout: d(f16)@0, qh([u8;4])@2, qs([u8;16])@6
        // Element i: qs[i/2] lo/hi nibble + qh bit i → 5-bit value, subtract 16
        const BLOCK_SIZE: usize = 22;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            let out_block = &mut out[bi * 32..(bi + 1) * 32];
            for i in 0..32 {
                let byte = b[6 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let hbit = (qh >> i) & 1;
                let q = (nibble as u32) | (hbit << 4);
                out_block[i] = d * (q as f32 - 16.0);
            }
        }
    }

    fn dequant_q5_1(&self, block: &[u8], out: &mut [f32]) {
        // BlockQ5_1 layout: d(f16)@0, m(f16)@2, qh([u8;4])@4, qs([u8;16])@8
        // Element i: qs[i/2] lo/hi nibble + qh bit i → 5-bit value
        const BLOCK_SIZE: usize = 24;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let m = f16::from_le_bytes([b[2], b[3]]).to_f32();
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let out_block = &mut out[bi * 32..(bi + 1) * 32];
            for i in 0..32 {
                let byte = b[8 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let hbit = (qh >> i) & 1;
                let q = (nibble as u32) | (hbit << 4);
                out_block[i] = d * q as f32 + m;
            }
        }
    }

    fn dequant_q8_0(&self, block: &[u8], out: &mut [f32]) {
        const BLOCK_SIZE: usize = 34; // f16 + 32 i8
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let out_block = &mut out[bi * 32..(bi + 1) * 32];
            for i in 0..32 {
                out_block[i] = d * (b[2 + i] as i8) as f32;
            }
        }
    }

    fn dequant_q8_1(&self, block: &[u8], out: &mut [f32]) {
        const BLOCK_SIZE: usize = 36; // f16 + f16 + 32 i8
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            // b[2..4] is sum (unused in decode)
            let out_block = &mut out[bi * 32..(bi + 1) * 32];
            for i in 0..32 {
                out_block[i] = d * (b[4 + i] as i8) as f32;
            }
        }
    }

    // ── AWQ/GPTQ/Squeeze dequantization ──

    fn dequant_awq4(&self, packed: &[u8], zeros: &[u8], scales: &[f16], out: &mut [f32]) {
        // Simple scalar decode: each u32 packs 8 x 4-bit values
        let packed_u32 = unsafe {
            std::slice::from_raw_parts(packed.as_ptr() as *const u32, packed.len() / 4)
        };
        let zeros_u32 = unsafe {
            std::slice::from_raw_parts(zeros.as_ptr() as *const u32, zeros.len() / 4)
        };
        for (i, word) in packed_u32.iter().enumerate() {
            for j in 0..8 {
                let idx = i * 8 + j;
                if idx >= out.len() { return; }
                let q = ((*word >> (j * 4)) & 0xF) as f32;
                let z_word = if i < zeros_u32.len() { zeros_u32[i] } else { 0 };
                let z = ((z_word >> (j * 4)) & 0xF) as f32;
                let s = if idx < scales.len() { scales[idx].to_f32() } else { 1.0 };
                out[idx] = (q - z) * s;
            }
        }
    }

    fn dequant_gptq4(&self, packed: &[u8], g_idx: &[i32], scales: &[f16], out: &mut [f32]) {
        let packed_u32 = unsafe {
            std::slice::from_raw_parts(packed.as_ptr() as *const u32, packed.len() / 4)
        };
        for (i, word) in packed_u32.iter().enumerate() {
            for j in 0..8 {
                let idx = i * 8 + j;
                if idx >= out.len() { return; }
                let q = ((*word >> (j * 4)) & 0xF) as f32;
                let group = if idx < g_idx.len() { g_idx[idx] as usize } else { 0 };
                let s = if group < scales.len() { scales[group].to_f32() } else { 1.0 };
                out[idx] = (q - 8.0) * s;
            }
        }
    }

    fn dequant_squeeze(&self, block: &[u8], out: &mut [f32]) {
        // SqueezeLLM: f16 scale + 128 bytes packed 3-bit
        const BLOCK_SIZE: usize = 130;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let out_block = &mut out[bi * 256..(bi + 1).min(num_blocks) * 256];
            for e in 0..out_block.len().min(256) {
                let bit_offset = e * 3;
                let byte_idx = bit_offset / 8;
                let bit_shift = bit_offset % 8;
                let lo = b[2 + byte_idx] as u16;
                let hi = if 2 + byte_idx + 1 < BLOCK_SIZE { b[2 + byte_idx + 1] as u16 } else { 0 };
                let combined = lo | (hi << 8);
                let q = ((combined >> bit_shift) & 0x07) as i8 - 4;
                out_block[e] = d * q as f32;
            }
        }
    }

    // ── Quantized GEMV ──

    fn gemv_q8(&self, weight: &[i8], input: &[E], scale: f32, n: usize) -> E {
        let mut acc = 0.0f32;
        for i in 0..n.min(weight.len()).min(input.len()) {
            acc += (weight[i] as f32) * input[i].to_f32();
        }
        E::from_f32(acc * scale)
    }

    fn gemv_q4(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        let mut acc = 0.0f32;
        for i in 0..n / 2 {
            if i >= weight.len() { break; }
            let byte = weight[i];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = (byte >> 4) as i8 - 8;
            if 2 * i < input.len() { acc += (lo as f32) * input[2 * i].to_f32(); }
            if 2 * i + 1 < input.len() { acc += (hi as f32) * input[2 * i + 1].to_f32(); }
        }
        E::from_f32(acc * scale)
    }

    fn gemv_q2(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        let mut acc = 0.0f32;
        for i in 0..n / 4 {
            if i >= weight.len() { break; }
            let byte = weight[i];
            for j in 0..4 {
                let q = ((byte >> (j * 2)) & 0x03) as i8 - 2;
                let idx = 4 * i + j;
                if idx < input.len() { acc += (q as f32) * input[idx].to_f32(); }
            }
        }
        E::from_f32(acc * scale)
    }

    fn gemv_q1(&self, weight: &[u8], input: &[E], scale: f32, n: usize) -> E {
        let mut acc = 0.0f32;
        for i in 0..n / 8 {
            if i >= weight.len() { break; }
            let byte = weight[i];
            for j in 0..8 {
                let bit = (byte >> j) & 1;
                let q: f32 = if bit == 1 { 1.0 } else { -1.0 };
                let idx = 8 * i + j;
                if idx < input.len() { acc += q * input[idx].to_f32(); }
            }
        }
        E::from_f32(acc * scale)
    }

    // ── Quantized GEMM ──

    fn gemm_q8(&self, weight: &[i8], input: &[E], output: &mut [E], scales: &[f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let w_idx = p * n + j;
                    let w = if w_idx < weight.len() { weight[w_idx] as f32 } else { 0.0 };
                    let inp = if i * k + p < input.len() { input[i * k + p].to_f32() } else { 0.0 };
                    acc += w * inp;
                }
                let s = if j < scales.len() { scales[j] } else { 1.0 };
                output[i * n + j] = E::from_f32(acc * s);
            }
        }
    }

    fn gemm_q4(&self, weight: &[u8], input: &[E], output: &mut [E], scales: &[f32], m: usize, n: usize, k: usize) {
        // Weight is n rows of Q4_K blocks, dequantize-on-the-fly
        let block_bytes = QuantType::Q4K.block_bytes();
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * block_bytes;

        for i in 0..m {
            for j in 0..n {
                // Dequantize row j
                let row_start = j * row_bytes;
                let row_end = row_start + row_bytes;
                if row_end > weight.len() { continue; }
                let mut dequant = vec![0.0f32; k];
                self.dequant_q4_k(&weight[row_start..row_end], &mut dequant);

                let mut acc = 0.0f32;
                for p in 0..k {
                    let inp = if i * k + p < input.len() { input[i * k + p].to_f32() } else { 0.0 };
                    acc += dequant[p] * inp;
                }
                let s = if j < scales.len() { scales[j] } else { 1.0 };
                output[i * n + j] = E::from_f32(acc * s);
            }
        }
    }

    // ── K-Quant matmul ──

    fn kquant_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: QuantType, m: usize, n: usize, k: usize,
    ) {
        let block_bytes = quant_type.block_bytes();
        let block_size = quant_type.block_size();
        let blocks_per_row = k / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        for j in 0..m {
            let row_start = j * row_bytes;
            let row_end = row_start + row_bytes;
            if row_end > weight_blocks.len() { continue; }
            let mut dequant = vec![0.0f32; k];
            match quant_type {
                QuantType::Q4K => self.dequant_q4_k(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q8K => self.dequant_q8_k(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q2K => self.dequant_q2_k(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q3K => self.dequant_q3_k(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q5K => self.dequant_q5_k(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q6K => self.dequant_q6_k(&weight_blocks[row_start..row_end], &mut dequant),
                _ => panic!("unsupported QuantType {:?} in kquant_matmul, routing bug", quant_type),
            }
            for i in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let inp = if i * k + p < input.len() { input[i * k + p].to_f32() } else { 0.0 };
                    acc += dequant[p] * inp;
                }
                output[j * n + i] = E::from_f32(acc);
            }
        }
    }

    // ── Classic GGML matmul ──

    fn classic_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: QuantType, m: usize, n: usize, k: usize,
    ) {
        let block_bytes = quant_type.block_bytes();
        let block_size = quant_type.block_size();
        let blocks_per_row = k / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        for j in 0..m {
            let row_start = j * row_bytes;
            let row_end = row_start + row_bytes;
            if row_end > weight_blocks.len() { continue; }
            let mut dequant = vec![0.0f32; k];
            match quant_type {
                QuantType::Q4_0 => self.dequant_q4_0(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q4_1 => self.dequant_q4_1(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q5_0 => self.dequant_q5_0(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q5_1 => self.dequant_q5_1(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q8_0 => self.dequant_q8_0(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::Q8_1 => self.dequant_q8_1(&weight_blocks[row_start..row_end], &mut dequant),
                _ => panic!("unsupported QuantType {:?} in classic_matmul, routing bug", quant_type),
            }
            for i in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let inp = if i * k + p < input.len() { input[i * k + p].to_f32() } else { 0.0 };
                    acc += dequant[p] * inp;
                }
                output[j * n + i] = E::from_f32(acc);
            }
        }
    }

    // ── IQ dequantization ──

    fn dequant_iq4_nl(&self, block: &[u8], out: &mut [f32]) {
        // IQ4_NL: 4-bit non-linear quantization, block_size=32, block_bytes=18
        // Layout: d(f16, 2 bytes) + qs(16 bytes, 32 x 4-bit indices into kvalues_iq4nl)
        // Decode: output[i] = d * kvalues_iq4nl[nibble[i]]
        const BLOCK_SIZE: usize = 18;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qs = &b[2..18]; // 16 bytes = 32 nibbles
            let out_block = &mut out[bi * 32..(bi + 1) * 32];
            // Lower nibbles go to first 16 elements, upper nibbles to last 16
            for j in 0..16 {
                out_block[j] = d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize];
                out_block[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize];
            }
        }
    }

    fn dequant_iq4_xs(&self, block: &[u8], out: &mut [f32]) {
        // IQ4_XS: 4-bit importance quantization (extra small), block_size=256, block_bytes=136
        // Layout: d(f16, 2) + scales_h(u16, 2) + scales_l([u8; 4]) + qs([u8; 128])
        // Contains 8 sub-blocks of 32 elements each, each with its own 6-bit scale
        const BLOCK_SIZE: usize = 136;
        let num_blocks = block.len() / BLOCK_SIZE;
        for bi in 0..num_blocks {
            let b = &block[bi * BLOCK_SIZE..];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let scales_h = u16::from_le_bytes([b[2], b[3]]);
            let scales_l = &b[4..8];
            let qs = &b[8..136]; // 128 bytes
            let out_block = &mut out[bi * QK_K..(bi + 1) * QK_K];

            for ib in 0..8 {
                // Unpack 6-bit scale: 4 bits from scales_l, 2 bits from scales_h
                let ls_lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
                let ls_hi = ((scales_h >> (2 * ib)) & 3) as u8;
                let ls = (ls_lo | (ls_hi << 4)) as i32;
                let dl = d * (ls - 32) as f32;

                let sub_qs = &qs[ib * 16..(ib + 1) * 16];
                let sub_out = &mut out_block[ib * 32..(ib + 1) * 32];
                for j in 0..16 {
                    sub_out[j] = dl * KVALUES_IQ4NL[(sub_qs[j] & 0x0F) as usize];
                    sub_out[j + 16] = dl * KVALUES_IQ4NL[(sub_qs[j] >> 4) as usize];
                }
            }
        }
    }

    fn dequant_iq1_s(&self, block: &[u8], out: &mut [f32]) {
        // IQ1_S: E8-lattice codebook, block_bytes=50, QK_K=256
        // Layout: d(f16,2) + qs([u8;32]) + qh([u16;8]=16 bytes)
        use crate::codebooks::IQ1S_GRID;
        const BLOCK_BYTES: usize = 50;
        let nblocks = block.len() / BLOCK_BYTES;
        for bi in 0..nblocks {
            let b = &block[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qs = &b[2..34];   // 32 bytes
            let qh = &b[34..50];  // 16 bytes = 8 x u16
            let base = bi * QK_K;
            for ib in 0..8 {
                let qh_val = u16::from_le_bytes([qh[ib * 2], qh[ib * 2 + 1]]);
                for j in 0..4 {
                    let idx = ib * 4 + j;
                    let grid_idx = (qs[idx] as usize) | ((((qh_val >> (3 * j)) & 7) as usize) << 8);
                    let grid_val = IQ1S_GRID[grid_idx.min(2047)];
                    let delta = if (qh_val >> (12 + j)) & 1 != 0 { -1.0_f32 } else { 1.0 };
                    for k in 0..8 {
                        let pos = base + ib * 32 + j * 8 + k;
                        if pos < out.len() {
                            let val = ((grid_val >> (k * 8)) & 0xFF) as i8;
                            out[pos] = d * delta * val as f32;
                        }
                    }
                }
            }
        }
    }

    fn dequant_iq1_m(&self, block: &[u8], out: &mut [f32]) {
        // IQ1_M: E8-lattice codebook with per-block scales, block_bytes=56, QK_K=256
        // Layout: qs([u8;32]) + qh([u16;8]=16 bytes) + scales([u8;6]) + d(f16,2)
        use crate::codebooks::IQ1S_GRID;
        const BLOCK_BYTES: usize = 56;
        let nblocks = block.len() / BLOCK_BYTES;
        for bi in 0..nblocks {
            let b = &block[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
            let qs = &b[0..32];
            let qh = &b[32..48];  // 16 bytes = 8 x u16
            let scales = &b[48..54];
            let d = f16::from_le_bytes([b[54], b[55]]).to_f32();
            let base = bi * QK_K;

            // Unpack 16 x 4-bit scales from 6 bytes (packed nibbles + extra bits)
            let mut sc = [0u8; 8];
            for i in 0..4 {
                sc[2 * i] = (scales[i] & 0x0F) | (((scales[4 + i / 2] >> (4 * (i % 2))) & 0x0F) << 4);
                sc[2 * i + 1] = (scales[i] >> 4) | (((scales[4 + i / 2] >> (4 * (i % 2) + 2)) & 0x03) << 4);
            }

            for ib in 0..8 {
                let qh_val = u16::from_le_bytes([qh[ib * 2], qh[ib * 2 + 1]]);
                let scale = d * (2.0 * sc[ib] as f32 + 1.0) / 64.0;
                for j in 0..4 {
                    let idx = ib * 4 + j;
                    let grid_idx = (qs[idx] as usize) | ((((qh_val >> (3 * j)) & 7) as usize) << 8);
                    let grid_val = IQ1S_GRID[grid_idx.min(2047)];
                    let delta = if (qh_val >> (12 + j)) & 1 != 0 { -1.0_f32 } else { 1.0 };
                    for k in 0..8 {
                        let pos = base + ib * 32 + j * 8 + k;
                        if pos < out.len() {
                            let val = ((grid_val >> (k * 8)) & 0xFF) as i8;
                            out[pos] = scale * delta * val as f32;
                        }
                    }
                }
            }
        }
    }

    fn dequant_iq2_xxs(&self, block: &[u8], out: &mut [f32]) {
        // IQ2_XXS: D4-lattice codebook (256 x u64), block_bytes=66, QK_K=256
        // Layout: d(f16,2) + qs([u16;32]=64 bytes)
        // Each u16 in qs: low 8 bits = grid index, high 8 bits = signs+scale
        use crate::codebooks::{IQ2XXS_GRID, KSIGNS_IQ2XS};
        const BLOCK_BYTES: usize = 66;
        let nblocks = block.len() / BLOCK_BYTES;
        for bi in 0..nblocks {
            let b = &block[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qs = &b[2..66]; // 64 bytes = 32 x u16
            let base = bi * QK_K;
            for ib in 0..8 {
                // Each sub-block: 4 x u16 = 8 bytes, producing 32 values
                let sub = &qs[ib * 8..(ib + 1) * 8];
                // Last u16 contains scale info
                let aux16 = u16::from_le_bytes([sub[6], sub[7]]);
                let scale = d * (0.5 + (aux16 >> 12) as f32) * 0.25;
                for j in 0..3 {
                    let q2 = u16::from_le_bytes([sub[j * 2], sub[j * 2 + 1]]);
                    let grid_idx = (q2 & 0xFF) as usize;
                    let sign_idx = ((q2 >> 8) & 0x7F) as usize;
                    let grid_val = IQ2XXS_GRID[grid_idx.min(255)];
                    let signs = KSIGNS_IQ2XS[sign_idx.min(127)];
                    for k in 0..8 {
                        let pos = base + ib * 32 + j * 8 + k;
                        if pos < out.len() {
                            let val = ((grid_val >> (k * 8)) & 0xFF) as i8;
                            let sign = if (signs >> k) & 1 != 0 { -1.0_f32 } else { 1.0 };
                            out[pos] = scale * sign * val as f32;
                        }
                    }
                }
                // 4th group from aux16
                let grid_idx = (aux16 & 0xFF) as usize;
                let sign_idx = ((aux16 >> 8) & 0x0F) as usize;
                let grid_val = IQ2XXS_GRID[grid_idx.min(255)];
                for k in 0..8 {
                    let pos = base + ib * 32 + 24 + k;
                    if pos < out.len() {
                        let val = ((grid_val >> (k * 8)) & 0xFF) as i8;
                        let sign = if (sign_idx >> k) & 1 != 0 { -1.0_f32 } else { 1.0 };
                        out[pos] = scale * sign * val as f32;
                    }
                }
            }
        }
    }

    fn dequant_iq2_xs(&self, block: &[u8], out: &mut [f32]) {
        // IQ2_XS: D4-lattice codebook (512 x u64), block_bytes=74, QK_K=256
        // Layout: d(f16,2) + qs([u16;32]=64 bytes) + scales([u8;8])
        use crate::codebooks::{IQ2XS_GRID, KSIGNS_IQ2XS, KMASK_IQ2XS};
        const BLOCK_BYTES: usize = 74;
        let nblocks = block.len() / BLOCK_BYTES;
        for bi in 0..nblocks {
            let b = &block[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qs = &b[2..66];     // 64 bytes = 32 x u16
            let scales = &b[66..74]; // 8 bytes
            let base = bi * QK_K;
            for ib in 0..8 {
                let scale = d * (0.5 + (scales[ib] & 0x0F) as f32) * 0.25;
                for j in 0..4 {
                    let idx = ib * 4 + j;
                    let q2 = u16::from_le_bytes([qs[idx * 2], qs[idx * 2 + 1]]);
                    let grid_idx = (q2 & 0x1FF) as usize;
                    let sign_idx = ((q2 >> 9) & 0x7F) as usize;
                    let grid_val = IQ2XS_GRID[grid_idx.min(511)];
                    let signs = KSIGNS_IQ2XS[sign_idx.min(127)];
                    for k in 0..8 {
                        let pos = base + ib * 32 + j * 8 + k;
                        if pos < out.len() {
                            let val = ((grid_val >> (k * 8)) & 0xFF) as i8;
                            let sign = if signs & KMASK_IQ2XS[k] != 0 { -1.0_f32 } else { 1.0 };
                            out[pos] = scale * sign * val as f32;
                        }
                    }
                }
            }
        }
    }

    fn dequant_iq2_s(&self, block: &[u8], out: &mut [f32]) {
        // IQ2_S: D4-lattice codebook (1024 x u64), block_bytes=82, QK_K=256
        // Layout: d(f16,2) + qs([u8;32]) + qh([u8;16]) + scales([u8;16]) + signs([u8;16])
        use crate::codebooks::IQ2S_GRID;
        const BLOCK_BYTES: usize = 82;
        let nblocks = block.len() / BLOCK_BYTES;
        for bi in 0..nblocks {
            let b = &block[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qs = &b[2..34];     // 32 bytes
            let qh = &b[34..50];    // 16 bytes
            let scales = &b[50..66]; // 16 bytes
            let signs = &b[66..82]; // 16 bytes
            let base = bi * QK_K;
            for ib in 0..8 {
                let scale = d * (0.5 + (scales[ib] & 0x0F) as f32) * 0.25;
                for j in 0..4 {
                    let idx = ib * 4 + j;
                    let grid_idx = (qs[idx] as usize) | (((qh[idx / 2] >> (4 * (idx % 2))) & 0x0F) as usize) << 8;
                    let grid_val = IQ2S_GRID[grid_idx.min(1023)];
                    let sign_byte = signs[ib * 2 + j / 2];
                    let sign_nibble = if j % 2 == 0 { sign_byte & 0x0F } else { sign_byte >> 4 };
                    for k in 0..8 {
                        let pos = base + ib * 32 + j * 8 + k;
                        if pos < out.len() {
                            let val = ((grid_val >> (k * 8)) & 0xFF) as i8;
                            let sign = if (sign_nibble >> (k % 4)) & 1 != 0 { -1.0_f32 } else { 1.0 };
                            out[pos] = scale * sign * val as f32;
                        }
                    }
                }
            }
        }
    }

    fn dequant_iq3_xxs(&self, block: &[u8], out: &mut [f32]) {
        // IQ3_XXS: D4-lattice codebook (256 x u32), block_bytes=98, QK_K=256
        // Layout: d(f16,2) + qs([u8;48]) + signs+scales(48 bytes)
        use crate::codebooks::IQ3XXS_GRID;
        const BLOCK_BYTES: usize = 98;
        let nblocks = block.len() / BLOCK_BYTES;
        for bi in 0..nblocks {
            let b = &block[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qs = &b[2..50];      // 48 bytes: 3-bit indices
            let signs_scales = &b[50..98]; // 48 bytes: signs + scales
            let base = bi * QK_K;
            for ib in 0..8 {
                // Each sub-block: 6 bytes of qs (32 x 3-bit = 96 bits = 12 bytes per 2 sub-blocks)
                // signs_scales: 6 bytes per sub-block (4 bytes signs + 2 bytes scale)
                let ss = &signs_scales[ib * 6..(ib + 1) * 6];
                let scale_bits = u16::from_le_bytes([ss[4], ss[5]]);
                let scale = d * (0.5 + scale_bits as f32) / 32.0;
                let sign_bytes = &ss[0..4]; // 32 sign bits
                for j in 0..32 {
                    let q_idx = ib * 6 + j * 3 / 8;
                    let bit_off = (j * 3) % 8;
                    let grid_idx = if q_idx + 1 < qs.len() {
                        let combined = (qs[q_idx] as u16) | ((qs[q_idx + 1] as u16) << 8);
                        ((combined >> bit_off) & 0xFF) as usize
                    } else {
                        ((qs[q_idx.min(47)] >> bit_off) & 0xFF) as usize
                    };
                    let grid_val = IQ3XXS_GRID[grid_idx.min(255)];
                    let sign = if (sign_bytes[j / 8] >> (j % 8)) & 1 != 0 { -1.0_f32 } else { 1.0 };
                    let pos = base + ib * 32 + j;
                    if pos < out.len() {
                        // IQ3 grid values are u32, each encoding a single dequantized value
                        let val = grid_val as i32;
                        out[pos] = scale * sign * val as f32;
                    }
                }
            }
        }
    }

    fn dequant_iq3_s(&self, block: &[u8], out: &mut [f32]) {
        // IQ3_S: D4-lattice codebook (512 x u32), block_bytes=110, QK_K=256
        // Layout: d(f16,2) + qs([u8;64]) + qh([u8;16]) + signs([u8;16]) + scales([u8;12])
        use crate::codebooks::IQ3S_GRID;
        const BLOCK_BYTES: usize = 110;
        let nblocks = block.len() / BLOCK_BYTES;
        for bi in 0..nblocks {
            let b = &block[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            let qs = &b[2..66];      // 64 bytes
            let qh = &b[66..82];     // 16 bytes (high bits)
            let signs = &b[82..98];  // 16 bytes
            let scales = &b[98..110]; // 12 bytes
            let base = bi * QK_K;
            for ib in 0..8 {
                let scale_byte = scales[ib * 3 / 2];
                let ls = if ib % 2 == 0 { scale_byte & 0x0F } else { scale_byte >> 4 };
                let scale = d * (1.0 + 2.0 * ls as f32);
                // signs: 4 bytes per sub-block (32 bits)
                let sign_u32 = u32::from_le_bytes([
                    signs[ib * 2],
                    signs[ib * 2 + 1],
                    if ib * 2 + 2 < signs.len() { signs[ib * 2 + 2] } else { 0 },
                    if ib * 2 + 3 < signs.len() { signs[ib * 2 + 3] } else { 0 },
                ]);
                for j in 0..32 {
                    let idx = ib * 8 + j / 4;
                    let shift = (j % 4) * 2;
                    let full_idx = if idx < qs.len() {
                        ((qs[idx] as usize) >> shift) & 0xFF
                    } else { 0 };
                    let qh_byte = qh[ib * 2 + j / 16];
                    let hi = ((qh_byte >> (j % 8)) & 1) as usize;
                    let grid_idx = (full_idx & 0xFF) | (hi << 8);
                    let grid_val = IQ3S_GRID[grid_idx.min(511)];
                    let sign = if (sign_u32 >> j) & 1 != 0 { -1.0_f32 } else { 1.0 };
                    let pos = base + ib * 32 + j;
                    if pos < out.len() {
                        let val = grid_val as i32;
                        out[pos] = scale * sign * val as f32;
                    }
                }
            }
        }
    }

    // ── IQ matmul ──

    fn iq_matmul(
        &self, weight_blocks: &[u8], input: &[E], output: &mut [E],
        quant_type: QuantType, m: usize, n: usize, k: usize,
    ) {
        // Dequantize-then-matmul: functional correctness path.
        // Weight layout: m rows of quantized blocks, each row encodes k elements.
        let block_bytes = quant_type.block_bytes();
        let block_size = quant_type.block_size();
        let blocks_per_row = k / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        for j in 0..m {
            let row_start = j * row_bytes;
            let row_end = row_start + row_bytes;
            if row_end > weight_blocks.len() { continue; }
            let mut dequant = vec![0.0f32; k];
            match quant_type {
                QuantType::IQ4NL => self.dequant_iq4_nl(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ4XS => self.dequant_iq4_xs(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ1S => self.dequant_iq1_s(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ1M => self.dequant_iq1_m(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ2XXS => self.dequant_iq2_xxs(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ2XS => self.dequant_iq2_xs(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ2S => self.dequant_iq2_s(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ3XXS => self.dequant_iq3_xxs(&weight_blocks[row_start..row_end], &mut dequant),
                QuantType::IQ3S => self.dequant_iq3_s(&weight_blocks[row_start..row_end], &mut dequant),
                _ => panic!("unsupported QuantType {:?} in iq_matmul, routing bug", quant_type),
            }
            for i in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let inp = if i * k + p < input.len() { input[i * k + p].to_f32() } else { 0.0 };
                    acc += dequant[p] * inp;
                }
                output[j * n + i] = E::from_f32(acc);
            }
        }
    }
}

/// The CPU backend (SPEC/03 §2.2).
pub struct CpuBackend;

impl Backend for CpuBackend {
    const NAME: &'static str = "cpu";

    type Kernels<E: Element> = CpuKernels<E>;

    fn init<E: Element>() -> Self::Kernels<E> {
        CpuKernels::new()
    }
}
