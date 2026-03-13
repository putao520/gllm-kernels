//! GPU IR primitive types shared across all GPU backends.
//!
//! These types form the vocabulary for the `GpuDialect` trait, abstracting
//! over PTX / HIP / MSL differences at the instruction level.

/// Kernel parameter descriptor.
#[derive(Debug, Clone)]
pub struct KernelParam {
    /// Parameter name (e.g. "input", "output").
    pub name: String,
    /// Parameter type.
    pub ty: ParamType,
    /// Input/output qualifier.
    pub qualifier: ParamQualifier,
}

/// Parameter type for kernel signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    /// Pointer to float buffer.
    FloatPtr,
    /// Unsigned 32-bit integer (e.g. dimension N).
    Uint,
    /// Scalar float value.
    Float,
}

/// Parameter qualifier — input vs output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamQualifier {
    /// Read-only input.
    Input,
    /// Write-only output.
    Output,
    /// Scalar value parameter.
    Value,
}

/// Backend capability flags.
#[derive(Debug, Clone, Copy)]
pub struct GpuCapabilities {
    /// Whether the backend has matrix unit support (Tensor Cores / MFMA / simdgroup_matrix).
    pub has_matrix_unit: bool,
    /// Whether the backend supports injective (multi-input/output) codegen.
    pub has_injective_codegen: bool,
}
