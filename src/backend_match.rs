use crate::backend_trait::{TensorSlice, TensorSliceMut};

pub(crate) fn match_float1<R>(
    input: TensorSlice<'_>,
    f32_fn: impl FnOnce(&[f32]) -> R,
    f16_fn: impl FnOnce(&[half::f16]) -> R,
    bf16_fn: impl FnOnce(&[half::bf16]) -> R,
) -> Result<R, String> {
    match input {
        TensorSlice::F32(data) => Ok(f32_fn(data)),
        TensorSlice::F16(data) => Ok(f16_fn(data)),
        TensorSlice::BF16(data) => Ok(bf16_fn(data)),
    }
}

pub(crate) fn match_float1_mut(
    input: TensorSliceMut<'_>,
    f32_fn: impl FnOnce(&mut [f32]),
    f16_fn: impl FnOnce(&mut [half::f16]),
    bf16_fn: impl FnOnce(&mut [half::bf16]),
) -> Result<(), String> {
    match input {
        TensorSliceMut::F32(data) => {
            f32_fn(data);
            Ok(())
        }
        TensorSliceMut::F16(data) => {
            f16_fn(data);
            Ok(())
        }
        TensorSliceMut::BF16(data) => {
            bf16_fn(data);
            Ok(())
        }
    }
}

pub(crate) fn match_float1_out(
    op: &str,
    input: TensorSlice<'_>,
    output: TensorSliceMut<'_>,
    f32_fn: impl FnOnce(&[f32], &mut [f32]),
    f16_fn: impl FnOnce(&[half::f16], &mut [half::f16]),
    bf16_fn: impl FnOnce(&[half::bf16], &mut [half::bf16]),
) -> Result<(), String> {
    match (input, output) {
        (TensorSlice::F32(input), TensorSliceMut::F32(output)) => {
            f32_fn(input, output);
            Ok(())
        }
        (TensorSlice::F16(input), TensorSliceMut::F16(output)) => {
            f16_fn(input, output);
            Ok(())
        }
        (TensorSlice::BF16(input), TensorSliceMut::BF16(output)) => {
            bf16_fn(input, output);
            Ok(())
        }
        _ => Err(dtype_mismatch(op)),
    }
}

pub(crate) fn match_float2_out(
    op: &str,
    a: TensorSlice<'_>,
    b: TensorSlice<'_>,
    output: TensorSliceMut<'_>,
    f32_fn: impl FnOnce(&[f32], &[f32], &mut [f32]),
    f16_fn: impl FnOnce(&[half::f16], &[half::f16], &mut [half::f16]),
    bf16_fn: impl FnOnce(&[half::bf16], &[half::bf16], &mut [half::bf16]),
) -> Result<(), String> {
    match (a, b, output) {
        (TensorSlice::F32(a), TensorSlice::F32(b), TensorSliceMut::F32(output)) => {
            f32_fn(a, b, output);
            Ok(())
        }
        (TensorSlice::F16(a), TensorSlice::F16(b), TensorSliceMut::F16(output)) => {
            f16_fn(a, b, output);
            Ok(())
        }
        (TensorSlice::BF16(a), TensorSlice::BF16(b), TensorSliceMut::BF16(output)) => {
            bf16_fn(a, b, output);
            Ok(())
        }
        _ => Err(dtype_mismatch(op)),
    }
}

pub(crate) fn match_float2_out2(
    op: &str,
    a: TensorSlice<'_>,
    b: TensorSlice<'_>,
    out_a: TensorSliceMut<'_>,
    out_b: TensorSliceMut<'_>,
    f32_fn: impl FnOnce(&[f32], &[f32], &mut [f32], &mut [f32]),
    f16_fn: impl FnOnce(&[half::f16], &[half::f16], &mut [half::f16], &mut [half::f16]),
    bf16_fn: impl FnOnce(&[half::bf16], &[half::bf16], &mut [half::bf16], &mut [half::bf16]),
) -> Result<(), String> {
    match (a, b, out_a, out_b) {
        (TensorSlice::F32(a), TensorSlice::F32(b), TensorSliceMut::F32(out_a), TensorSliceMut::F32(out_b)) => {
            f32_fn(a, b, out_a, out_b);
            Ok(())
        }
        (TensorSlice::F16(a), TensorSlice::F16(b), TensorSliceMut::F16(out_a), TensorSliceMut::F16(out_b)) => {
            f16_fn(a, b, out_a, out_b);
            Ok(())
        }
        (TensorSlice::BF16(a), TensorSlice::BF16(b), TensorSliceMut::BF16(out_a), TensorSliceMut::BF16(out_b)) => {
            bf16_fn(a, b, out_a, out_b);
            Ok(())
        }
        _ => Err(dtype_mismatch(op)),
    }
}

pub(crate) fn match_float3_out(
    op: &str,
    a: TensorSlice<'_>,
    b: TensorSlice<'_>,
    c: TensorSlice<'_>,
    output: TensorSliceMut<'_>,
    f32_fn: impl FnOnce(&[f32], &[f32], &[f32], &mut [f32]),
    f16_fn: impl FnOnce(&[half::f16], &[half::f16], &[half::f16], &mut [half::f16]),
    bf16_fn: impl FnOnce(&[half::bf16], &[half::bf16], &[half::bf16], &mut [half::bf16]),
) -> Result<(), String> {
    match (a, b, c, output) {
        (TensorSlice::F32(a), TensorSlice::F32(b), TensorSlice::F32(c), TensorSliceMut::F32(output)) => {
            f32_fn(a, b, c, output);
            Ok(())
        }
        (TensorSlice::F16(a), TensorSlice::F16(b), TensorSlice::F16(c), TensorSliceMut::F16(output)) => {
            f16_fn(a, b, c, output);
            Ok(())
        }
        (TensorSlice::BF16(a), TensorSlice::BF16(b), TensorSlice::BF16(c), TensorSliceMut::BF16(output)) => {
            bf16_fn(a, b, c, output);
            Ok(())
        }
        _ => Err(dtype_mismatch(op)),
    }
}

fn dtype_mismatch(op: &str) -> String {
    format!("{op}: dtype mismatch")
}
