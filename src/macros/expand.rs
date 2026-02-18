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
            use half::{f16, bf16};
            #[allow(unused_imports)]
            use crate::macros::simd_primitive;

            // Include Layer 2 templates
            crate::define_element_wise_ops!($isa, $elem);
            crate::define_blas1_ops!($isa, $elem);
            crate::define_activation_ops!($isa, $elem);
            crate::define_norm_ops!($isa, $elem);
            crate::define_position_ops!($isa, $elem);
            crate::define_gemv_op!($isa, $elem);
            crate::define_matmul_op!($isa, $elem);

        }
    };
}
