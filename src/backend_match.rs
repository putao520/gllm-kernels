use crate::backend_trait::{TensorSlice, TensorSliceMut};
use crate::kernel_types::KernelFloat;

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

pub(crate) fn match_float1_mut_weight(
    op: &str,
    data: TensorSliceMut<'_>,
    weight: TensorSlice<'_>,
    f32_fn: impl FnOnce(&mut [f32], &[f32]),
    f16_fn: impl FnOnce(&mut [half::f16], &[half::f16]),
    bf16_fn: impl FnOnce(&mut [half::bf16], &[half::bf16]),
) -> Result<(), String> {
    match (data, weight) {
        (TensorSliceMut::F32(data), TensorSlice::F32(weight)) => {
            f32_fn(data, weight);
            Ok(())
        }
        (TensorSliceMut::F16(data), TensorSlice::F16(weight)) => {
            f16_fn(data, weight);
            Ok(())
        }
        (TensorSliceMut::BF16(data), TensorSlice::BF16(weight)) => {
            bf16_fn(data, weight);
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

pub(crate) fn apply_f32_unary_inplace<T: KernelFloat>(
    data: &mut [T],
    f32_fn: impl FnOnce(&mut [f32]),
) {
    let mut data_f32: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    f32_fn(&mut data_f32);
    for (dst, val) in data.iter_mut().zip(data_f32.iter()) {
        *dst = T::from_f32(*val);
    }
}

pub(crate) fn apply_f32_unary_out<T: KernelFloat>(
    input: &[T],
    output: &mut [T],
    f32_fn: impl FnOnce(&[f32], &mut [f32]),
) {
    debug_assert_eq!(input.len(), output.len());
    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    let mut output_f32 = vec![0.0f32; output.len()];
    f32_fn(&input_f32, &mut output_f32);
    for (dst, val) in output.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }
}

pub(crate) fn apply_f32_binary_out<T: KernelFloat>(
    input: &[T],
    weight: &[T],
    output: &mut [T],
    f32_fn: impl FnOnce(&[f32], &[f32], &mut [f32]),
) {
    debug_assert_eq!(input.len(), output.len());
    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
    let mut output_f32 = vec![0.0f32; output.len()];
    f32_fn(&input_f32, &weight_f32, &mut output_f32);
    for (dst, val) in output.iter_mut().zip(output_f32.iter()) {
        *dst = T::from_f32(*val);
    }
}

pub(crate) fn apply_f32_inplace_weight<T: KernelFloat>(
    data: &mut [T],
    weight: &[T],
    f32_fn: impl FnOnce(&mut [f32], &[f32]),
) {
    let mut data_f32: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
    f32_fn(&mut data_f32, &weight_f32);
    for (dst, val) in data.iter_mut().zip(data_f32.iter()) {
        *dst = T::from_f32(*val);
    }
}
