#![cfg_attr(target_arch = "x86_64", feature(x86_amx_intrinsics))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512_f16))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_f16))]
#![feature(f16)]
#![allow(
    dead_code,
    unused_macros,
    unused_variables,
    unused_assignments,
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
pub use traits::Activation;
pub mod cache_params;
pub mod microarch;
pub mod numa;
pub mod macros;
pub mod cpu_kernels;
pub mod backend;
pub mod gpu;
pub mod quant;
pub mod codebooks;
pub mod asm;
pub mod profiling;
pub mod autotuning;
pub mod dispatch;
pub mod inference;
pub mod compiler;
pub mod ffi;

#[cfg(test)]
pub mod tests;
#[cfg(test)]
mod tests_quant;
#[cfg(test)]
mod tests_simd;
#[cfg(test)]
mod tests_int8;
#[cfg(test)]
mod check_isa;
#[cfg(test)]
mod tests_amx;

pub use traits::{Element, Backend, Kernels};

/// Type alias for the primitive `f16` type (core::f16).
/// Needed because `half::f16` shadows the primitive in macro expansion contexts.
/// Both are IEEE 754 binary16 with identical memory layout.
#[allow(non_camel_case_types)]
pub type prim_f16 = f16;
