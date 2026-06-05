
// Expand scalar float-32 implementations
crate::expand_isa_impls!(scalar_f32, scalar, f32);
// Expand scalar f16 implementations (compute in f32, load/store with conversion)
crate::expand_isa_impls!(scalar_f16, scalar, f16);
// Expand scalar bf16 implementations (compute in f32, load/store with conversion)
crate::expand_isa_impls!(scalar_bf16, scalar, bf16);
