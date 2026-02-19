/// Layer 3 quant_primitive dispatcher.
///
/// Routes `quant_primitive!(isa, format, op, ...)` to the appropriate sub-macro:
/// - Classic (Q4_0~Q8_1) → `quant_primitive_classic!`
/// - K-Quant (Q2_K~Q8_K) → `quant_primitive_kquant!`
/// - IQ series (IQ1_S~IQ4_XS) → `quant_primitive_iq!`
/// - Commercial (AWQ4/GPTQ4/Squeeze) → `quant_primitive_commercial!`

#[macro_use]
pub mod classic;
#[macro_use]
pub mod k_quant;
#[macro_use]
pub mod iq_series;
#[macro_use]
pub mod commercial;

#[macro_export]
macro_rules! quant_primitive {
    // Classic GGML formats → quant_primitive_classic!
    ($isa:ident, q4_0, $($rest:tt)*) => { $crate::quant_primitive_classic!($isa, q4_0, $($rest)*) };
    ($isa:ident, q4_1, $($rest:tt)*) => { $crate::quant_primitive_classic!($isa, q4_1, $($rest)*) };
    ($isa:ident, q5_0, $($rest:tt)*) => { $crate::quant_primitive_classic!($isa, q5_0, $($rest)*) };
    ($isa:ident, q5_1, $($rest:tt)*) => { $crate::quant_primitive_classic!($isa, q5_1, $($rest)*) };
    ($isa:ident, q8_0, $($rest:tt)*) => { $crate::quant_primitive_classic!($isa, q8_0, $($rest)*) };
    ($isa:ident, q8_1, $($rest:tt)*) => { $crate::quant_primitive_classic!($isa, q8_1, $($rest)*) };

    // K-Quant formats → quant_primitive_kquant!
    ($isa:ident, q4_k, $($rest:tt)*) => { $crate::quant_primitive_kquant!($isa, q4_k, $($rest)*) };
    ($isa:ident, q8_k, $($rest:tt)*) => { $crate::quant_primitive_kquant!($isa, q8_k, $($rest)*) };
    ($isa:ident, q2_k, $($rest:tt)*) => { $crate::quant_primitive_kquant!($isa, q2_k, $($rest)*) };
    ($isa:ident, q3_k, $($rest:tt)*) => { $crate::quant_primitive_kquant!($isa, q3_k, $($rest)*) };
    ($isa:ident, q5_k, $($rest:tt)*) => { $crate::quant_primitive_kquant!($isa, q5_k, $($rest)*) };
    ($isa:ident, q6_k, $($rest:tt)*) => { $crate::quant_primitive_kquant!($isa, q6_k, $($rest)*) };

    // IQ series → quant_primitive_iq!
    ($isa:ident, iq1_s, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq1_s, $($rest)*) };
    ($isa:ident, iq1_m, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq1_m, $($rest)*) };
    ($isa:ident, iq2_xxs, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq2_xxs, $($rest)*) };
    ($isa:ident, iq2_xs, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq2_xs, $($rest)*) };
    ($isa:ident, iq2_s, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq2_s, $($rest)*) };
    ($isa:ident, iq3_xxs, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq3_xxs, $($rest)*) };
    ($isa:ident, iq3_s, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq3_s, $($rest)*) };
    ($isa:ident, iq4_nl, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq4_nl, $($rest)*) };
    ($isa:ident, iq4_xs, $($rest:tt)*) => { $crate::quant_primitive_iq!($isa, iq4_xs, $($rest)*) };

    // Commercial formats → quant_primitive_commercial!
    ($isa:ident, awq4, $($rest:tt)*) => { $crate::quant_primitive_commercial!($isa, awq4, $($rest)*) };
    ($isa:ident, gptq4, $($rest:tt)*) => { $crate::quant_primitive_commercial!($isa, gptq4, $($rest)*) };
    ($isa:ident, squeeze, $($rest:tt)*) => { $crate::quant_primitive_commercial!($isa, squeeze, $($rest)*) };
}
