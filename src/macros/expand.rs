/// Expands operator templates for a specific ISA and Element type.
///
/// This macro is the "Layer 4" of the architecture. It takes a module name,
/// an ISA identifier, and an element type, and generates a module containing
/// all the operator implementations for that specific combination.
#[macro_export]
macro_rules! expand_isa_impls {
    ($module_name:ident, $isa:ident, $elem:ident) => {
        pub mod $module_name {
            use crate::traits::Element;
            #[allow(unused_imports)]
            // use crate::$isa::*; // Removed to avoid error for scalar
            use crate::macros::simd_primitive; // We might need this if we use it unqualified, but we switch to $crate::simd_primitive

            // Include Layer 2 templates
            // Layer 2 templates: Element-wise ops (add, mul, silu)
            crate::define_element_wise_ops!($isa, $elem);
            // BLAS-1 ops (sub, div, scale, axpy, sum, max_val, sum_squares)
            crate::define_blas1_ops!($isa, $elem);
            // Activation ops (relu, gelu, tanh, exp, softmax)
            crate::define_activation_ops!($isa, $elem);
            // Normalization ops (rms_norm, layer_norm)
            crate::define_norm_ops!($isa, $elem);
            // Position ops (rope, embedding_lookup)
            crate::define_position_ops!($isa, $elem);
            // GEMV
            crate::define_gemv_op!($isa, $elem);
            // GEMM (matmul)
            crate::define_matmul_op!($isa, $elem);
        }
    };
}
