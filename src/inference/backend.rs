use std::collections::HashMap;

use crate::compiler::graph::{CompilerGraph, OpKind, TensorId};
use crate::types::InferenceError;
use super::types::DeviceTensor;

/// Fallback inference backend trait.
///
/// Executes a `CompilerGraph` by walking ops in topological order.
/// Used when JIT compilation fails or is unavailable.
pub trait InferenceBackend: Send + Sync {
    /// Execute `graph` with the given input tensors, writing results into `outputs`.
    ///
    /// `inputs[i]` corresponds to `graph.inputs[i]` (same order).
    /// `outputs[i]` corresponds to `graph.outputs[i]`.
    fn forward(
        &self,
        graph: &CompilerGraph,
        inputs: &[&DeviceTensor],
        outputs: &mut [DeviceTensor],
    ) -> Result<(), InferenceError>;
}

/// CPU fallback backend: executes each op via scalar functions.
pub struct CpuFallbackBackend;

impl CpuFallbackBackend {
    pub fn new() -> Self {
        CpuFallbackBackend
    }
}

impl Default for CpuFallbackBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a TensorId to its f32 data, checking graph inputs first,
/// then the intermediate results map.
fn resolve_tensor<'a>(
    tid: TensorId,
    graph: &CompilerGraph,
    inputs: &'a [&'a DeviceTensor],
    intermediates: &'a HashMap<TensorId, DeviceTensor>,
) -> Result<&'a [f32], InferenceError> {
    if let Some(idx) = graph.inputs.iter().position(|&id| id == tid) {
        if idx < inputs.len() {
            return Ok(inputs[idx].as_f32_slice());
        }
    }
    intermediates
        .get(&tid)
        .map(|t| t.as_f32_slice())
        .ok_or_else(|| {
            InferenceError::RuntimeError(format!("tensor {:?} not produced yet", tid))
        })
}

/// Allocate an output DeviceTensor matching the graph tensor metadata.
fn alloc_output(graph: &CompilerGraph, tid: TensorId) -> Result<DeviceTensor, InferenceError> {
    let meta = graph.tensor(tid).ok_or_else(|| {
        InferenceError::RuntimeError(format!("unknown tensor {:?}", tid))
    })?;
    let numel: usize = meta.shape.iter().product::<usize>().max(1);
    Ok(DeviceTensor::from_slice(
        &vec![0.0_f32; numel],
        meta.shape.clone(),
        meta.dtype,
    ))
}

// ---------------------------------------------------------------------------
// InferenceBackend impl
// ---------------------------------------------------------------------------

impl InferenceBackend for CpuFallbackBackend {
    fn forward(
        &self,
        graph: &CompilerGraph,
        inputs: &[&DeviceTensor],
        outputs: &mut [DeviceTensor],
    ) -> Result<(), InferenceError> {
        if inputs.len() != graph.inputs.len() {
            return Err(InferenceError::ShapeMismatch {
                expected: format!("{} graph inputs", graph.inputs.len()),
                got: format!("{} provided", inputs.len()),
            });
        }
        if outputs.len() != graph.outputs.len() {
            return Err(InferenceError::ShapeMismatch {
                expected: format!("{} graph outputs", graph.outputs.len()),
                got: format!("{} provided", outputs.len()),
            });
        }

        let sorted = graph.topological_sort();
        let mut intermediates: HashMap<TensorId, DeviceTensor> = HashMap::new();

        for op_id in sorted {
            let op = graph.op(op_id).ok_or_else(|| {
                InferenceError::RuntimeError(format!("op {:?} not found", op_id))
            })?;
            let kind = op.kind.clone();
            let in_ids = op.inputs.clone();
            let out_ids = op.outputs.clone();

            match kind {
                OpKind::Silu => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = x.len();
                    crate::scalar_ops::activations::scalar_silu(
                        x.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Gelu => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = x.len();
                    crate::scalar_ops::activations::scalar_gelu(
                        x.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::SwiGlu => {
                    let gate = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let up = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = gate.len();
                    crate::scalar_ops::activations::scalar_swiglu(
                        gate.as_ptr(),
                        up.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::GeGlu => {
                    let gate = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let up = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = gate.len();
                    crate::scalar_ops::activations::scalar_geglu(
                        gate.as_ptr(),
                        up.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Softmax => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = x.len();
                    crate::scalar_ops::blas::scalar_softmax(
                        x.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Add | OpKind::Residual => {
                    let a = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let b = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = a.len();
                    crate::scalar_ops::blas::scalar_vec_add(
                        a.as_ptr(),
                        b.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Mul => {
                    let a = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let b = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = a.len();
                    crate::scalar_ops::blas::scalar_vec_mul(
                        a.as_ptr(),
                        b.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Gemm { m, n, k } => {
                    let a = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let b = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let c = out.as_f32_slice_mut();
                    c.fill(0.0);
                    crate::scalar_ops::blas::scalar_gemm(
                        a.as_ptr(),
                        b.as_ptr(),
                        c.as_mut_ptr(),
                        m, n, k,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::GemmBias { m, n, k } => {
                    let a = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let b = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let bias = resolve_tensor(in_ids[2], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    crate::scalar_ops::blas::scalar_gemm_bias(
                        a.as_ptr(),
                        b.as_ptr(),
                        bias.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        m, n, k,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::RmsNorm { eps } => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let weight = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = x.len();
                    crate::scalar_ops::norms::scalar_rms_norm(
                        x.as_ptr(),
                        weight.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                        eps,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::LayerNorm { eps } => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let weight = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let bias = resolve_tensor(in_ids[2], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = x.len();
                    crate::scalar_ops::norms::scalar_layer_norm(
                        x.as_ptr(),
                        weight.as_ptr(),
                        bias.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                        eps,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::RoPE { head_dim, .. } => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let cos_sin = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    // cos_sin tensor has [half_dim] elements — treat as cos values,
                    // derive sin = sqrt(1 - cos^2).
                    let cos_vals: Vec<f32> = cos_sin.to_vec();
                    let sin_vals: Vec<f32> = cos_vals
                        .iter()
                        .map(|&c| (1.0 - c * c).max(0.0).sqrt())
                        .collect();
                    let n_heads = x.len() / head_dim;
                    crate::scalar_ops::rope::scalar_rope(
                        x.as_ptr(),
                        cos_vals.as_ptr(),
                        sin_vals.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        head_dim,
                        n_heads,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Transpose { ref perm } => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let meta = graph.tensor(in_ids[0]).ok_or_else(|| {
                        InferenceError::RuntimeError(format\!("tensor {:?} not found in graph", in_ids[0]))
                    })?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    // Only 2D transpose supported in scalar fallback.
                    if meta.shape.len() == 2 && perm == &[1, 0] {
                        let rows = meta.shape[0];
                        let cols = meta.shape[1];
                        crate::scalar_ops::blas::scalar_transpose_2d(
                            x.as_ptr(),
                            out.as_f32_slice_mut().as_mut_ptr(),
                            rows,
                            cols,
                        );
                    } else {
                        // General case: identity copy for unsupported permutations.
                        out.as_f32_slice_mut().copy_from_slice(x);
                    }
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Reshape { .. } => {
                    let x = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    let n = x.len();
                    crate::scalar_ops::blas::scalar_reshape(
                        x.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        n,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::Dequantize {
                    num_elements,
                    block_size,
                    ..
                } => {
                    let quant = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let scale = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    crate::scalar_ops::blas::scalar_dequantize(
                        quant.as_ptr(),
                        scale.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        num_elements,
                        block_size,
                    );
                    intermediates.insert(out_ids[0], out);
                }

                OpKind::QuantGemm {
                    m, n, k, block_size, ..
                } => {
                    let a = resolve_tensor(in_ids[0], graph, inputs, &intermediates)?;
                    let b_quant = resolve_tensor(in_ids[1], graph, inputs, &intermediates)?;
                    let b_scale = resolve_tensor(in_ids[2], graph, inputs, &intermediates)?;
                    let mut out = alloc_output(graph, out_ids[0])?;
                    crate::scalar_ops::blas::scalar_quant_gemm(
                        a.as_ptr(),
                        b_quant.as_ptr(),
                        b_scale.as_ptr(),
                        out.as_f32_slice_mut().as_mut_ptr(),
                        m, n, k, block_size,
                    );
                    intermediates.insert(out_ids[0], out);
                }
            }
        }

        // Copy graph outputs to caller's output tensors.
        for (i, &tid) in graph.outputs.iter().enumerate() {
            let src = intermediates.get(&tid).ok_or_else(|| {
                InferenceError::RuntimeError(format!(
                    "graph output {:?} was not produced",
                    tid
                ))
            })?;
            outputs[i].copy_from(src)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, OpKind};
    use crate::types::DType;

    fn make_backend() -> CpuFallbackBackend {
        CpuFallbackBackend::new()
    }

    /// input -> Silu -> output
    #[test]
    fn test_forward_silu() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let inp = g.add_tensor("input", vec![4], dt);
        let out = g.add_tensor("output", vec![4], dt);
        g.add_op(OpKind::Silu, vec![inp], vec![out], "silu");
        g.inputs = vec![inp];
        g.outputs = vec![out];

        let input_data = DeviceTensor::from_slice(&[0.0, 1.0, -1.0, 2.0], vec![4], dt);
        let mut output = DeviceTensor::zeros(vec![4], dt);

        make_backend()
            .forward(&g, &[&input_data], std::slice::from_mut(&mut output))
            .unwrap();

        let result = output.as_f32_slice();
        // silu(0) = 0, silu(x) = x * sigmoid(x)
        assert!((result[0]).abs() < 1e-6);
        let expected_1 = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!((result[1] - expected_1).abs() < 1e-5);
    }

    /// a, b -> Add -> output
    #[test]
    fn test_forward_add() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![4], dt);
        let b = g.add_tensor("b", vec![4], dt);
        let out = g.add_tensor("out", vec![4], dt);
        g.add_op(OpKind::Add, vec![a, b], vec![out], "add");
        g.inputs = vec![a, b];
        g.outputs = vec![out];

        let ta = DeviceTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4], dt);
        let tb = DeviceTensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4], dt);
        let mut output = DeviceTensor::zeros(vec![4], dt);

        make_backend()
            .forward(&g, &[&ta, &tb], std::slice::from_mut(&mut output))
            .unwrap();

        assert_eq!(output.as_f32_slice(), &[11.0, 22.0, 33.0, 44.0]);
    }

    /// A[1,2] * B[2,2] -> C[1,2]
    #[test]
    fn test_forward_gemm() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![1, 2], dt);
        let b = g.add_tensor("b", vec![2, 2], dt);
        let c = g.add_tensor("c", vec![1, 2], dt);
        g.add_op(OpKind::Gemm { m: 1, n: 2, k: 2 }, vec![a, b], vec![c], "gemm");
        g.inputs = vec![a, b];
        g.outputs = vec![c];

        let ta = DeviceTensor::from_slice(&[1.0, 2.0], vec![1, 2], dt);
        let tb = DeviceTensor::from_slice(&[3.0, 4.0, 5.0, 6.0], vec![2, 2], dt);
        let mut output = DeviceTensor::zeros(vec![1, 2], dt);

        make_backend()
            .forward(&g, &[&ta, &tb], std::slice::from_mut(&mut output))
            .unwrap();

        // C = [1,2] * [[3,4],[5,6]] = [1*3+2*5, 1*4+2*6] = [13, 16]
        assert_eq!(output.as_f32_slice(), &[13.0, 16.0]);
    }

    /// input -> RmsNorm -> Silu -> output (chain of two ops)
    #[test]
    fn test_forward_chain_rmsnorm_silu() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let inp = g.add_tensor("input", vec![4], dt);
        let w = g.add_tensor("weight", vec![4], dt);
        let normed = g.add_tensor("normed", vec![4], dt);
        let out = g.add_tensor("output", vec![4], dt);

        g.add_op(
            OpKind::RmsNorm { eps: 1e-5 },
            vec![inp, w],
            vec![normed],
            "rms_norm",
        );
        g.add_op(OpKind::Silu, vec![normed], vec![out], "silu");
        g.inputs = vec![inp, w];
        g.outputs = vec![out];

        let t_inp = DeviceTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4], dt);
        let t_w = DeviceTensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4], dt);
        let mut output = DeviceTensor::zeros(vec![4], dt);

        make_backend()
            .forward(&g, &[&t_inp, &t_w], std::slice::from_mut(&mut output))
            .unwrap();

        let result = output.as_f32_slice();

        // Verify: RmsNorm then Silu produces finite non-zero values.
        assert!(result.iter().all(|v| v.is_finite()));
        assert!(result.iter().any(|&v| v.abs() > 1e-6));

        // Cross-check with manual scalar calls.
        let mut normed_buf = vec![0.0_f32; 4];
        let mut expected = vec![0.0_f32; 4];
        crate::scalar_ops::norms::scalar_rms_norm(
            t_inp.as_f32_slice().as_ptr(),
            t_w.as_f32_slice().as_ptr(),
            normed_buf.as_mut_ptr(),
            4,
            1e-5,
        );
        crate::scalar_ops::activations::scalar_silu(
            normed_buf.as_ptr(),
            expected.as_mut_ptr(),
            4,
        );
        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "mismatch at {i}: got {}, expected {}",
                result[i],
                expected[i]
            );
        }
    }

    /// Diamond DAG: input -> Silu -> left, input -> Gelu -> right, left + right -> output
    #[test]
    fn test_forward_diamond() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let inp = g.add_tensor("input", vec![4], dt);
        let left = g.add_tensor("left", vec![4], dt);
        let right = g.add_tensor("right", vec![4], dt);
        let out = g.add_tensor("output", vec![4], dt);

        g.add_op(OpKind::Silu, vec![inp], vec![left], "silu");
        g.add_op(OpKind::Gelu, vec![inp], vec![right], "gelu");
        g.add_op(OpKind::Add, vec![left, right], vec![out], "add");
        g.inputs = vec![inp];
        g.outputs = vec![out];

        let t_inp = DeviceTensor::from_slice(&[0.0, 1.0, -1.0, 2.0], vec![4], dt);
        let mut output = DeviceTensor::zeros(vec![4], dt);

        make_backend()
            .forward(&g, &[&t_inp], std::slice::from_mut(&mut output))
            .unwrap();

        let result = output.as_f32_slice();

        // Cross-check: silu(x) + gelu(x)
        let mut silu_buf = vec![0.0_f32; 4];
        let mut gelu_buf = vec![0.0_f32; 4];
        crate::scalar_ops::activations::scalar_silu(
            t_inp.as_f32_slice().as_ptr(),
            silu_buf.as_mut_ptr(),
            4,
        );
        crate::scalar_ops::activations::scalar_gelu(
            t_inp.as_f32_slice().as_ptr(),
            gelu_buf.as_mut_ptr(),
            4,
        );
        for i in 0..4 {
            let expected = silu_buf[i] + gelu_buf[i];
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "diamond[{i}]: got {}, expected {}",
                result[i],
                expected
            );
        }
    }

    /// Softmax correctness: output sums to 1.
    #[test]
    fn test_forward_softmax() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let inp = g.add_tensor("input", vec![4], dt);
        let out = g.add_tensor("output", vec![4], dt);
        g.add_op(OpKind::Softmax, vec![inp], vec![out], "softmax");
        g.inputs = vec![inp];
        g.outputs = vec![out];

        let t_inp = DeviceTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4], dt);
        let mut output = DeviceTensor::zeros(vec![4], dt);

        make_backend()
            .forward(&g, &[&t_inp], std::slice::from_mut(&mut output))
            .unwrap();

        let sum: f32 = output.as_f32_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    /// Input count mismatch returns error.
    #[test]
    fn test_forward_input_mismatch() {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let a = g.add_tensor("a", vec![4], dt);
        let b = g.add_tensor("b", vec![4], dt);
        let out = g.add_tensor("out", vec![4], dt);
        g.add_op(OpKind::Add, vec![a, b], vec![out], "add");
        g.inputs = vec![a, b];
        g.outputs = vec![out];

        let ta = DeviceTensor::from_slice(&[1.0; 4], vec![4], dt);
        let mut output = DeviceTensor::zeros(vec![4], dt);

        // Only 1 input provided, graph expects 2.
        let err = make_backend().forward(&g, &[&ta], std::slice::from_mut(&mut output));
        assert!(err.is_err());
    }
}
