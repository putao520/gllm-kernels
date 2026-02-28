//! Shared math approximations — ISA-independent transcendental functions.
//!
//! Provides polynomial coefficients and generic code-generation functions for
//! exp, tanh, and log. These are called by `algorithm.rs` and emit SIMD
//! instructions via the `SimdOps` trait, eliminating per-backend duplication.
//!
//! All approximations target f32 precision (≤1 ULP error for exp/log,
//! ≤2 ULP for tanh).

use super::simd_ops::{SimdOps, VReg};

// ── Exp coefficients (Cephes degree-5 polynomial) ───────────────────────────

/// Input clamp range for exp (prevents overflow/underflow).
pub const EXP_CLAMP_LO: f32 = -88.376;
pub const EXP_CLAMP_HI: f32 = 88.376;
/// log2(e) for range reduction.
pub const EXP_LOG2E: f32 = 1.4426950408889634;
/// Cody-Waite constants for range reduction.
pub const EXP_C1: f32 = -0.693359375;
pub const EXP_C2: f32 = 2.12194440e-4;
/// Horner polynomial coefficients (degree 5).
pub const EXP_P0: f32 = 1.9875691500e-4;
pub const EXP_P1: f32 = 1.3981999507e-3;
pub const EXP_P2: f32 = 8.3334519073e-3;
pub const EXP_P3: f32 = 4.1665795894e-2;
pub const EXP_P4: f32 = 1.6666665459e-1;
pub const EXP_P5: f32 = 5.0000001201e-1;
/// Magic constant for 2^k construction via integer shift.
pub const EXP_MAGIC127: f32 = 127.0;

// ── Tanh coefficients (via 2*sigmoid(2x) - 1) ──────────────────────────────

pub const TANH_NEG2: f32 = -2.0;
pub const TANH_TWO: f32 = 2.0;

// ── Log coefficients (degree-4 minimax polynomial) ──────────────────────────

/// Mantissa extraction mask: 0x007FFFFF
pub const LOG_MANTISSA_MASK: f32 = f32::from_bits(0x007F_FFFF);
/// ln(2) for exponent scaling.
pub const LOG_LN2: f32 = 0.6931471805599453;
/// Minimax coefficients for ln(1+t), t in [0, 1).
pub const LOG_C1: f32 = 0.99999934;
pub const LOG_C2: f32 = -0.49987412;
pub const LOG_C3: f32 = 0.33179903;
pub const LOG_C4: f32 = -0.24073381;

// ── Generic emit functions ──────────────────────────────────────────────────

/// Emit exp(x) approximation using Cephes degree-5 polynomial.
///
/// Algorithm:
/// 1. Clamp input to [-88.376, 88.376]
/// 2. Compute k = round(x * log2(e))
/// 3. Cody-Waite range reduction: r = x - k * ln(2)
/// 4. Horner polynomial: p = p0 + r*(p1 + r*(p2 + r*(p3 + r*(p4 + r*p5))))
/// 5. Reconstruct: exp(x) = p * 2^k
///
/// Uses `dst` for output, `src` for input, `s[0..3]` as scratch.
pub fn emit_exp<E: SimdOps>(
    e: &mut E,
    dst: VReg,
    src: VReg,
    s: [VReg; 3],
) -> Result<(), String> {
    // Clamp input
    e.vbroadcast_const(s[1], EXP_CLAMP_LO)?;
    e.vmax(s[0], src, s[1])?;
    e.vbroadcast_const(s[1], EXP_CLAMP_HI)?;
    e.vmin(s[0], s[0], s[1])?;

    // x * log2(e)
    e.vbroadcast_const(s[1], EXP_LOG2E)?;
    e.vmul(s[1], s[0], s[1])?;

    // k = round(x * log2(e))
    e.vround(s[2], s[1])?;

    // Cody-Waite range reduction: s[0] = x - k*c1 - k*c2
    e.vbroadcast_const(s[1], EXP_C1)?;
    e.vfmadd231(s[0], s[2], s[1])?;
    e.vbroadcast_const(s[1], EXP_C2)?;
    e.vfmadd231(s[0], s[2], s[1])?;

    // Horner polynomial evaluation
    e.vbroadcast_const(dst, EXP_P0)?;
    e.vbroadcast_const(s[1], EXP_P1)?;
    e.vfmadd213(dst, s[0], s[1])?;
    e.vbroadcast_const(s[1], EXP_P2)?;
    e.vfmadd213(dst, s[0], s[1])?;
    e.vbroadcast_const(s[1], EXP_P3)?;
    e.vfmadd213(dst, s[0], s[1])?;
    e.vbroadcast_const(s[1], EXP_P4)?;
    e.vfmadd213(dst, s[0], s[1])?;
    e.vbroadcast_const(s[1], EXP_P5)?;
    e.vfmadd213(dst, s[0], s[1])?;
    e.vmul(dst, dst, s[0])?;
    e.vmul(dst, dst, s[0])?;
    e.vadd(dst, dst, s[0])?;
    e.vbroadcast_const(s[1], 1.0)?;
    e.vadd(dst, dst, s[1])?;

    // Construct 2^k: convert k to int, add 127, shift left 23
    e.vcvt_f32_i32(s[1], s[2])?;
    e.vbroadcast_const(s[2], EXP_MAGIC127)?;
    e.vcvt_f32_i32(s[2], s[2])?;
    e.vadd_i32(s[1], s[1], s[2])?;
    e.vshl_i32(s[1], s[1], 23)?;

    // dst = polynomial * 2^k
    e.vmul(dst, dst, s[1])?;

    Ok(())
}

/// Emit tanh(x) approximation via `2*sigmoid(2x) - 1`.
///
/// Uses exp(-2x) internally, then Newton-Raphson reciprocal.
///
/// Uses `dst` for output, `src` for input, `s[0..3]` as scratch.
pub fn emit_tanh<E: SimdOps>(
    e: &mut E,
    dst: VReg,
    src: VReg,
    s: [VReg; 3],
) -> Result<(), String> {
    // s[0] = -2 * src
    e.vbroadcast_const(s[1], TANH_NEG2)?;
    e.vmul(s[0], src, s[1])?;

    // dst = exp(-2x)
    emit_exp(e, dst, s[0], s)?;

    // dst = 1 + exp(-2x)
    e.vbroadcast_const(s[1], 1.0)?;
    e.vadd(dst, dst, s[1])?;

    // Newton-Raphson reciprocal: r0 = recip(dst), r1 = dst*r0, r = r0*(2-r1)
    e.vrecip(s[0], dst)?;
    e.vmul(s[1], dst, s[0])?;
    e.vbroadcast_const(s[2], TANH_TWO)?;
    e.vsub(s[1], s[2], s[1])?;
    e.vmul(s[0], s[0], s[1])?;
    // dst = 2 * reciprocal
    e.vmul(dst, s[0], s[2])?;

    // dst = 2*sigmoid(2x) - 1
    e.vbroadcast_const(s[1], 1.0)?;
    e.vsub(dst, dst, s[1])?;

    Ok(())
}

/// Emit log(x) approximation using degree-4 minimax polynomial.
///
/// Algorithm:
/// 1. Extract exponent: e = float(x >> 23) - 127
/// 2. Extract mantissa: m = (x & 0x007FFFFF) | 0x3F800000 → [1.0, 2.0)
/// 3. t = m - 1.0
/// 4. Horner: p = ((c4*t + c3)*t + c2)*t + c1
/// 5. result = e * ln(2) + p * t
///
/// Uses `dst` for output, `src` for input, `s[0..3]` as scratch.
pub fn emit_log<E: SimdOps>(
    e: &mut E,
    dst: VReg,
    src: VReg,
    s: [VReg; 3],
) -> Result<(), String> {
    // Extract exponent: e = float(x >> 23) - 127.0
    e.vshr_i32(s[0], src, 23)?;
    e.vcvt_i32_f32(s[0], s[0])?;
    e.vbroadcast_const(s[2], EXP_MAGIC127)?;
    e.vsub(s[0], s[0], s[2])?;

    // Extract mantissa: m = (x & 0x007FFFFF) | 0x3F800000 → [1.0, 2.0)
    e.vbroadcast_const(s[2], LOG_MANTISSA_MASK)?;
    e.vand(s[1], src, s[2])?;
    e.vbroadcast_const(s[2], 1.0)?; // 0x3F800000 as f32
    e.vor(s[1], s[1], s[2])?;

    // t = m - 1.0
    e.vsub(s[1], s[1], s[2])?;

    // Horner: p = ((c4 * t + c3) * t + c2) * t + c1
    e.vbroadcast_const(dst, LOG_C4)?;
    e.vbroadcast_const(s[2], LOG_C3)?;
    e.vfmadd213(dst, s[1], s[2])?;
    e.vbroadcast_const(s[2], LOG_C2)?;
    e.vfmadd213(dst, s[1], s[2])?;
    e.vbroadcast_const(s[2], LOG_C1)?;
    e.vfmadd213(dst, s[1], s[2])?;
    // p = p * t
    e.vmul(dst, dst, s[1])?;

    // result = e * ln(2) + p
    e.vbroadcast_const(s[2], LOG_LN2)?;
    e.vfmadd231(dst, s[0], s[2])?;

    Ok(())
}
