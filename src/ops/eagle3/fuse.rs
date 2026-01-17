use crate::kernel_dispatcher::KernelFloat;
use crate::ops::math;

use super::config::AdaptiveDraftConfig;

/// Fuse hidden states from multiple layers (standalone function).
#[inline(always)]
pub fn fuse_multi_layer_hidden<T: KernelFloat>(
    layer_hidden_states: &[&[T]],
    config: &AdaptiveDraftConfig,
    batch: usize,
    seq_len: usize,
    output: &mut [T],
) -> Result<(), &'static str> {
    if batch == 0 || seq_len == 0 {
        return Err("batch and seq_len must be > 0");
    }
    if layer_hidden_states.is_empty() {
        return Err("no hidden states provided");
    }
    if layer_hidden_states.len() < config.fusion_layers {
        return Err("insufficient layers for fusion");
    }

    let hidden_dim = config.hidden_dim;
    let fused_dim = config.fused_dim();
    let expected_layer = batch * seq_len * hidden_dim;
    let expected_output = batch * seq_len * fused_dim;
    if output.len() != expected_output {
        return Err("output length mismatch");
    }

    for layer in layer_hidden_states.iter() {
        if layer.len() != expected_layer {
            return Err("layer length mismatch");
        }
    }

    let start_idx = layer_hidden_states.len() - config.fusion_layers;
    for (layer_idx, layer) in layer_hidden_states[start_idx..].iter().enumerate() {
        for pos in 0..(batch * seq_len) {
            let src_offset = pos * hidden_dim;
            let dst_offset = pos * fused_dim + layer_idx * hidden_dim;
            for d in 0..hidden_dim {
                output[dst_offset + d] = layer[src_offset + d];
            }
        }
    }

    Ok(())
}

/// Predict token-level confidence from fused hidden states (standalone function).
#[inline(always)]
pub fn predict_confidence<T: KernelFloat>(
    fused_hidden: &[T],
    weight: &[T],
    bias: f32,
    config: &AdaptiveDraftConfig,
    output: &mut [T],
) -> Result<(), &'static str> {
    let fused_dim = config.fused_dim();
    if fused_dim == 0 {
        return Err("fused_dim must be > 0");
    }
    if weight.len() != fused_dim {
        return Err("weight length mismatch");
    }
    if fused_hidden.len() % fused_dim != 0 {
        return Err("fused_hidden length must be a multiple of fused_dim");
    }

    let positions = fused_hidden.len() / fused_dim;
    if output.len() != positions {
        return Err("output length mismatch");
    }

    for pos in 0..positions {
        let base = pos * fused_dim;
        let mut logit = bias;
        for d in 0..fused_dim {
            logit += fused_hidden[base + d].to_f32() * weight[d].to_f32();
        }
        output[pos] = T::from_f32(math::sigmoid(logit));
    }

    Ok(())
}
