#![allow(
    clippy::too_many_arguments,
    clippy::crate_in_macro_def,
    clippy::new_without_default,
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::excessive_precision,
    clippy::missing_safety_doc,
    clippy::approx_constant,
    clippy::uninit_vec,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::redundant_closure_for_method_calls,
    clippy::macro_metavars_in_unsafe,
)]

pub mod traits;
pub mod macros;
pub mod cpu_kernels;
pub mod cuda_kernels;
pub mod cuda_backend;
pub mod backend;
pub mod quant;
pub mod codebooks;

#[cfg(test)]
pub mod tests;
#[cfg(test)]
mod tests_quant;
#[cfg(test)]
mod tests_simd;
#[cfg(test)]
mod check_isa;

pub use traits::{Element, Backend, Kernels};
