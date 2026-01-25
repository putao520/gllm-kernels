use crate::backend_core::{match_float1, match_float1_out, BackendCore};
use crate::backend_trait::{TensorSlice, TensorSliceMut};
use crate::ops::moe_routing::{MoERoutingConfig, MoERoutingResult};

impl BackendCore {
    pub(crate) fn moe_route(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<MoERoutingResult, String> {
        match_float1(
            hidden_states,
            |hidden_states| {
                self.dispatcher
                    .moe_route::<f32>(hidden_states, gate_weights, batch_size, seq_len, config)
            },
            |hidden_states| {
                self.dispatcher
                    .moe_route::<half::f16>(hidden_states, gate_weights, batch_size, seq_len, config)
            },
            |hidden_states| {
                self.dispatcher
                    .moe_route::<half::bf16>(hidden_states, gate_weights, batch_size, seq_len, config)
            },
        )
    }

    pub(crate) fn compute_routing_logits(
        &self,
        hidden_states: TensorSlice<'_>,
        gate_weights: &[f32],
        batch_size: usize,
        seq_len: usize,
        config: &MoERoutingConfig,
    ) -> Result<Vec<f32>, String> {
        match_float1(
            hidden_states,
            |hidden_states| {
                self.dispatcher.compute_routing_logits::<f32>(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                self.dispatcher.compute_routing_logits::<half::f16>(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
            |hidden_states| {
                self.dispatcher.compute_routing_logits::<half::bf16>(
                    hidden_states,
                    gate_weights,
                    batch_size,
                    seq_len,
                    config,
                )
            },
        )
    }

    pub(crate) fn add_bias(
        &self,
        output: TensorSliceMut<'_>,
        bias: TensorSlice<'_>,
        batch: usize,
        features: usize,
    ) -> Result<(), String> {
        match_float1_out(
            "add_bias",
            bias,
            output,
            |bias, output| self.dispatcher.add_bias::<f32>(output, bias, batch, features),
            |bias, output| self.dispatcher.add_bias::<half::f16>(output, bias, batch, features),
            |bias, output| self.dispatcher.add_bias::<half::bf16>(output, bias, batch, features),
        )
    }
}
