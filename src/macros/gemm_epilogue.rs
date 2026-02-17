/// GEMM epilogue macros — fused activation applied to SIMD vectors before store.
///
/// `apply_act!(isa, elem, vec, relu/silu/gelu/none)` transforms a SIMD register
/// in-place, intended for use between the K-loop accumulation and the final storeu.
/// This avoids an extra read/write pass over the C matrix.

/// Apply activation function to a single SIMD vector.
/// Used inside the microkernel store phase for fused GEMM+activation.
#[macro_export]
macro_rules! apply_act {
    // No-op: pass through
    ($isa:ident, $elem:ident, $v:expr, none) => { $v };

    // ReLU: max(0, x)
    ($isa:ident, $elem:ident, $v:expr, relu) => {{
        let vz = $crate::simd_primitive!($isa, $elem, zero);
        $crate::simd_primitive!($isa, $elem, max, $v, vz)
    }};

    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    ($isa:ident, $elem:ident, $v:expr, silu) => {{
        let neg_v = $crate::simd_primitive!($isa, $elem, neg, $v);
        let exp_neg = $crate::simd_primitive!($isa, $elem, exp, neg_v);
        let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
        let one_plus_exp = $crate::simd_primitive!($isa, $elem, add, one, exp_neg);
        let sigmoid = $crate::simd_primitive!($isa, $elem, recip, one_plus_exp);
        $crate::simd_primitive!($isa, $elem, mul, $v, sigmoid)
    }};

    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ($isa:ident, $elem:ident, $v:expr, gelu) => {{
        let half = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(0.5));
        let one = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::ONE);
        let vc = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(0.044715));
        let vs = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(0.7978845608));
        let two = $crate::simd_primitive!($isa, $elem, splat, <$elem as Element>::from_f32(2.0));
        // x^3
        let x2 = $crate::simd_primitive!($isa, $elem, mul, $v, $v);
        let x3 = $crate::simd_primitive!($isa, $elem, mul, x2, $v);
        // x + 0.044715 * x^3
        let inner = $crate::simd_primitive!($isa, $elem, fma, vc, x3, $v);
        // sqrt(2/pi) * inner
        let scaled = $crate::simd_primitive!($isa, $elem, mul, vs, inner);
        // tanh via (exp(2x)-1)/(exp(2x)+1)
        let two_x = $crate::simd_primitive!($isa, $elem, mul, two, scaled);
        let exp_2x = $crate::simd_primitive!($isa, $elem, exp, two_x);
        let num = $crate::simd_primitive!($isa, $elem, sub, exp_2x, one);
        let den = $crate::simd_primitive!($isa, $elem, add, exp_2x, one);
        let tanh_val = $crate::simd_primitive!($isa, $elem, div, num, den);
        // 0.5 * x * (1 + tanh)
        let one_plus_tanh = $crate::simd_primitive!($isa, $elem, add, one, tanh_val);
        let half_x = $crate::simd_primitive!($isa, $elem, mul, half, $v);
        $crate::simd_primitive!($isa, $elem, mul, half_x, one_plus_tanh)
    }};
}

/// Runtime dispatch variant of `apply_act!` — accepts an `Activation` enum value
/// instead of a compile-time token. Uses precise `div` for silu (not approximate `recip`)
/// to match the numerical behavior of INT8/AMX dequantization kernels.
#[macro_export]
macro_rules! apply_act_runtime {
    ($isa:ident, $elem:ident, $v:expr, $act:expr) => {
        match $act {
            $crate::Activation::None => $v,
            $crate::Activation::Relu => {
                let vz = $crate::simd_primitive!($isa, $elem, zero);
                $crate::simd_primitive!($isa, $elem, max, $v, vz)
            }
            $crate::Activation::Silu => {
                // x * sigmoid(x) = x / (1 + exp(-x))  — precise div, not recip
                let neg_v = $crate::simd_primitive!($isa, $elem, neg, $v);
                let exp_neg = $crate::simd_primitive!($isa, $elem, exp, neg_v);
                let one = $crate::simd_primitive!($isa, $elem, splat, 1.0f32);
                let denom = $crate::simd_primitive!($isa, $elem, add, one, exp_neg);
                $crate::simd_primitive!($isa, $elem, div, $v, denom)
            }
            $crate::Activation::Gelu => {
                // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let half = $crate::simd_primitive!($isa, $elem, splat, 0.5f32);
                let one = $crate::simd_primitive!($isa, $elem, splat, 1.0f32);
                let vc = $crate::simd_primitive!($isa, $elem, splat, 0.044715f32);
                let vs = $crate::simd_primitive!($isa, $elem, splat, 0.7978845608f32);
                let two = $crate::simd_primitive!($isa, $elem, splat, 2.0f32);
                let x2 = $crate::simd_primitive!($isa, $elem, mul, $v, $v);
                let x3 = $crate::simd_primitive!($isa, $elem, mul, x2, $v);
                let inner = $crate::simd_primitive!($isa, $elem, fma, vc, x3, $v);
                let scaled = $crate::simd_primitive!($isa, $elem, mul, vs, inner);
                let two_x = $crate::simd_primitive!($isa, $elem, mul, two, scaled);
                let exp_2x = $crate::simd_primitive!($isa, $elem, exp, two_x);
                let num = $crate::simd_primitive!($isa, $elem, sub, exp_2x, one);
                let den = $crate::simd_primitive!($isa, $elem, add, exp_2x, one);
                let tanh_val = $crate::simd_primitive!($isa, $elem, div, num, den);
                let one_plus_tanh = $crate::simd_primitive!($isa, $elem, add, one, tanh_val);
                let half_x = $crate::simd_primitive!($isa, $elem, mul, half, $v);
                $crate::simd_primitive!($isa, $elem, mul, half_x, one_plus_tanh)
            }
        }
    };
}

/// Runtime dispatch scalar activation. Accepts an `Activation` enum value.
#[macro_export]
macro_rules! apply_act_scalar_runtime {
    ($v:expr, $act:expr) => {
        match $act {
            $crate::Activation::None => $v,
            $crate::Activation::Relu => if $v < 0.0 { 0.0 } else { $v },
            $crate::Activation::Silu => $v / (1.0 + (-$v).exp()),
            $crate::Activation::Gelu => {
                let inner = 0.7978845608f32 * ($v + 0.044715f32 * $v * $v * $v);
                0.5 * $v * (1.0 + inner.tanh())
            }
        }
    };
}

/// Apply activation to a scalar value. Used in remainder loops.
#[macro_export]
macro_rules! apply_act_scalar {
    ($v:expr, $elem:ident, none) => { $v };
    ($v:expr, $elem:ident, relu) => { <$elem as Element>::max($v, <$elem as Element>::ZERO) };
    ($v:expr, $elem:ident, silu) => {{
        let val_f = $v.to_f32();
        let sigmoid = 1.0f32 / (1.0f32 + (-val_f).exp());
        <$elem as Element>::from_f32(val_f * sigmoid)
    }};
    ($v:expr, $elem:ident, gelu) => {{
        let x = $v.to_f32();
        let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
        let tanh_val = inner.tanh();
        <$elem as Element>::from_f32(0.5f32 * x * (1.0f32 + tanh_val))
    }};
}
