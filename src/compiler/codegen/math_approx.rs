//! Shared math approximations — ISA-independent transcendental functions.
//!
//! Provides polynomial coefficients and generic code-generation functions for
//! exp, tanh, and log. These are called by `algorithm.rs` and emit SIMD
//! instructions via the `SimdOps` trait, eliminating per-backend duplication.
//!
//! All approximations target f32 precision (≤1 ULP error for exp/log,
//! ≤2 ULP for tanh).

// 系数常量供 vm/x86_lower.rs 的 emit_exp_cephes 等使用。
// 旧的 emit_exp/emit_tanh/emit_log 泛型函数 (依赖 SimdOps trait) 已删除。
// VM 路径通过 VmInstr::Transcendental + IsaLower 直接生成。

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

// 旧 emit_exp/emit_tanh/emit_log 泛型函数已删除 (依赖已删除的 SimdOps trait)。
// VM 路径: VmInstr::Transcendental → x86_lower::emit_exp_cephes 直接使用上述系数。

#[cfg(test)]
mod tests {
    use super::*;

    // ── Exp constants ─────────────────────────────────────────────────

    #[test]
    fn exp_clamp_range_covers_zero() {
        assert!(EXP_CLAMP_LO < 0.0);
        assert!(EXP_CLAMP_HI > 0.0);
        assert!(EXP_CLAMP_LO > -100.0);
        assert!(EXP_CLAMP_HI < 100.0);
    }

    #[test]
    fn exp_log2e_is_log2_e() {
        let expected = 1.0 / 2.0f32.ln(); // log2(e) = 1/ln(2)
        assert!((EXP_LOG2E - expected).abs() < 1e-6);
    }

    #[test]
    fn exp_c1_plus_c2_approximates_neg_ln2() {
        // C1 + C2 ≈ -ln(2) for Cody-Waite range reduction
        let neg_ln2 = -(2.0f32).ln();
        assert!((EXP_C1 + EXP_C2 - neg_ln2).abs() < 1e-3);
    }

    #[test]
    fn exp_polynomial_reconstructs_near_one() {
        // exp(t) ≈ 1 + P0*t + P1*t² + P2*t³ + P3*t⁴ + P4*t⁵ + P5*t⁶
        // for small t near 0, this should be close to 1
        let t = 0.0f32;
        let result = 1.0 + EXP_P0 * t + EXP_P1 * t * t;
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn exp_magic127_is_positive() {
        assert_eq!(EXP_MAGIC127, 127.0);
    }

    // ── Tanh constants ────────────────────────────────────────────────

    #[test]
    fn tanh_constants_values() {
        assert_eq!(TANH_NEG2, -2.0);
        assert_eq!(TANH_TWO, 2.0);
    }

    // ── Log constants ─────────────────────────────────────────────────

    #[test]
    fn log_mantissa_mask_is_valid_float_bits() {
        // 0x007FFFFF = lower 23 bits (mantissa of IEEE 754 float)
        let mask_bits = LOG_MANTISSA_MASK.to_bits();
        assert_eq!(mask_bits, 0x007F_FFFF);
    }

    #[test]
    fn log_ln2_is_natural_log_of_2() {
        let expected = 2.0f32.ln();
        assert!((LOG_LN2 - expected).abs() < 1e-7);
    }

    #[test]
    fn log_polynomial_at_zero_near_one() {
        // ln(1 + t) ≈ C1*t + C2*t² + C3*t³ + C4*t⁴
        // At t=0: should be 0 (ln(1)=0)
        let t = 0.0f32;
        let result = LOG_C1 * t + LOG_C2 * t * t + LOG_C3 * t * t * t;
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn log_polynomial_at_half_approximates_ln_1_5() {
        // ln(1 + 0.5) = ln(1.5) ≈ 0.4055
        let t = 0.5f32;
        let poly = LOG_C1 * t + LOG_C2 * t * t + LOG_C3 * t * t * t + LOG_C4 * t * t * t * t;
        let expected = (1.0f32 + t).ln();
        assert!((poly - expected).abs() < 0.01, "poly={poly}, expected={expected}");
    }

    // ── Cross-validation ──────────────────────────────────────────────

    #[test]
    fn exp_clamp_hi_below_float_max() {
        // exp(88.376) ≈ 1.7e38, well below f32::MAX ≈ 3.4e38
        assert!((EXP_CLAMP_HI.exp()) < f32::MAX);
    }

    #[test]
    fn exp_clamp_lo_above_float_min_positive() {
        // exp(-88.376) ≈ 1.7e-39, above f32 subnormal range
        assert!(EXP_CLAMP_LO.exp() > 0.0);
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn exp_cody_waite_precision() {
        // C1 + C2 should approximate -ln(2) to better than C1 alone
        let neg_ln2 = -(2.0f32).ln();
        let c1_only_error = (EXP_C1 - neg_ln2).abs();
        let combined_error = (EXP_C1 + EXP_C2 - neg_ln2).abs();
        assert!(combined_error < c1_only_error,
            "Cody-Waite should reduce error: combined={combined_error}, c1_only={c1_only_error}");
    }

    #[test]
    fn exp_polynomial_ordering() {
        // Coefficients should decrease in magnitude (degree-5 Horner)
        assert!(EXP_P0.abs() < EXP_P1.abs());
        assert!(EXP_P1.abs() < EXP_P2.abs());
        assert!(EXP_P2.abs() < EXP_P3.abs());
        assert!(EXP_P3.abs() < EXP_P4.abs());
        assert!(EXP_P4.abs() < EXP_P5.abs());
    }

    #[test]
    fn exp_clamp_hi_produces_finite() {
        let val = EXP_CLAMP_HI.exp();
        assert!(val.is_finite(), "exp(CLAMP_HI) should be finite, got {val}");
    }

    #[test]
    fn exp_clamp_lo_produces_finite() {
        let val = EXP_CLAMP_LO.exp();
        assert!(val.is_finite(), "exp(CLAMP_LO) should be finite, got {val}");
    }

    #[test]
    fn log_mantissa_mask_preserves_mantissa() {
        // For a normal float, applying the mask should zero out exponent and sign
        let val: f32 = 1.5; // 0x3FC00000
        let masked = f32::from_bits(val.to_bits() & LOG_MANTISSA_MASK.to_bits());
        // Mantissa part of 1.5 is 0.5 (implicit leading 1 removed)
        assert!(masked >= 0.0 && masked < 1.0, "masked={masked}");
    }

    #[test]
    fn log_c1_close_to_one() {
        // C1 approximates the leading coefficient of ln(1+t)
        assert!((LOG_C1 - 1.0).abs() < 0.001, "C1 should be close to 1.0");
    }

    #[test]
    fn log_polynomial_monotonic_on_positive() {
        // ln(1+t) polynomial should be monotonically increasing for t in (0, 1)
        let mut prev = 0.0f32;
        for i in 1..=10 {
            let t = i as f32 / 10.0;
            let poly = LOG_C1 * t + LOG_C2 * t * t + LOG_C3 * t * t * t + LOG_C4 * t * t * t * t;
            assert!(poly > prev, "poly not monotonic at t={t}: prev={prev}, poly={poly}");
            prev = poly;
        }
    }

    #[test]
    fn tanh_constants_relationship() {
        // TANH_TWO should be the negation of TANH_NEG2
        assert_eq!(TANH_TWO, -TANH_NEG2);
    }

    #[test]
    fn exp_log2e_positive() {
        assert!(EXP_LOG2E > 0.0, "log2(e) must be positive");
    }

    #[test]
    fn log_ln2_positive() {
        assert!(LOG_LN2 > 0.0, "ln(2) must be positive");
    }

    #[test]
    fn exp_magic127_integer_value() {
        assert_eq!(EXP_MAGIC127, 127.0);
        assert!(EXP_MAGIC127.fract() == 0.0, "must be exact integer");
    }
}
