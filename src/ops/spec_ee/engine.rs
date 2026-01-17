use crate::kernel_dispatcher::KernelFloat;
use crate::ops::math;

use super::cache::SharedActivations;
use super::config::SpecEEConfig;
use super::dropout::LayerDropoutSchedule;
use super::head::EarlyExitHead;
use super::stats::SpecEEStats;
use super::types::EarlyExitDecision;

/// SpecEE / LayerSkip engine for self-speculative decoding.
#[derive(Debug)]
pub struct SpecEEEngine<T: KernelFloat> {
    /// Configuration.
    config: SpecEEConfig,
    /// Early exit heads (one per exit layer).
    early_exit_heads: Vec<EarlyExitHead<T>>,
    /// Layer dropout schedule (for training).
    dropout_schedule: Option<LayerDropoutSchedule>,
    /// Shared activations cache.
    shared_activations: SharedActivations<T>,
    /// Runtime statistics.
    stats: SpecEEStats,
}

impl<T: KernelFloat> SpecEEEngine<T> {
    /// Create a new SpecEE engine.
    #[inline(always)]
    pub fn new(config: SpecEEConfig) -> Result<Self, &'static str> {
        config.validate()?;

        let mut early_exit_heads = Vec::with_capacity(config.exit_layers.len());
        for &layer_idx in &config.exit_layers {
            early_exit_heads.push(EarlyExitHead::new(
                config.hidden_dim,
                config.vocab_size,
                layer_idx,
            )?);
        }

        let dropout_schedule = if config.enable_layer_dropout {
            Some(LayerDropoutSchedule::linear(0.1, 0.5, config.num_layers)?)
        } else {
            None
        };

        let shared_activations = SharedActivations::new(config.num_layers)?;
        let stats = SpecEEStats::new(config.num_layers);

        Ok(Self {
            config,
            early_exit_heads,
            dropout_schedule,
            shared_activations,
            stats,
        })
    }

    /// Get configuration.
    #[inline(always)]
    pub fn config(&self) -> &SpecEEConfig {
        &self.config
    }

    /// Get statistics.
    #[inline(always)]
    pub fn stats(&self) -> &SpecEEStats {
        &self.stats
    }

    /// Get mutable reference to shared activations.
    #[inline(always)]
    pub fn shared_activations_mut(&mut self) -> &mut SharedActivations<T> {
        &mut self.shared_activations
    }

    /// Evaluate early exit decision for a layer.
    #[inline(always)]
    pub fn evaluate_exit(
        &mut self,
        hidden: &[T],
        batch: usize,
        seq_len: usize,
        layer_idx: usize,
    ) -> Result<Option<EarlyExitDecision>, &'static str> {
        let head_idx = self.config.exit_layers.iter().position(|&l| l == layer_idx);
        let head_idx = match head_idx {
            Some(idx) => idx,
            None => return Ok(None),
        };
        if layer_idx < self.config.min_exit_layer {
            return Ok(None);
        }
        if batch == 0 || seq_len == 0 {
            return Err("batch and seq_len must be > 0");
        }

        let head = &self.early_exit_heads[head_idx];
        let positions = batch * seq_len;
        let mut logits = vec![T::zero(); positions * self.config.vocab_size];
        let mut confidence = vec![T::zero(); positions];
        head.forward(hidden, batch, seq_len, &mut logits, &mut confidence)?;

        if self.config.share_activations {
            self.shared_activations.store_hidden(layer_idx, hidden, batch, seq_len)?;
            self.shared_activations.store_exit_output(
                layer_idx,
                &logits,
                &confidence,
                batch,
                seq_len,
            )?;
        }

        let last_idx = (batch - 1) * seq_len + (seq_len - 1);
        let last_conf = confidence[last_idx].to_f32();

        if last_conf >= self.config.confidence_threshold {
            let last_logits_start = last_idx * self.config.vocab_size;
            let last_logits_end = last_logits_start + self.config.vocab_size;
            let last_logits = &logits[last_logits_start..last_logits_end];
            let (token_id, log_prob) = math::find_top_token(last_logits);

            return Ok(Some(EarlyExitDecision {
                exit_layer: layer_idx,
                confidence: last_conf,
                token_id,
                log_prob,
                is_early_exit: true,
            }));
        }

        Ok(None)
    }

    /// Verify early exit decision against full forward result.
    #[inline(always)]
    pub fn verify_exit(&mut self, early_exit: &EarlyExitDecision, full_token_id: usize) -> bool {
        let accepted = early_exit.token_id == full_token_id;
        self.stats.record_early_exit(early_exit.exit_layer, accepted);
        accepted
    }

    /// Generate draft tokens using early exit.
    #[inline(always)]
    pub fn generate_draft(
        &mut self,
        layer_hidden_states: &[&[T]],
        batch: usize,
        seq_len: usize,
    ) -> Result<Vec<EarlyExitDecision>, &'static str> {
        if layer_hidden_states.len() != self.config.num_layers {
            return Err("layer_hidden_states length mismatch");
        }
        if batch == 0 || seq_len == 0 {
            return Err("batch and seq_len must be > 0");
        }

        let mut drafts = Vec::with_capacity(self.config.speculation_depth);
        let mut best_exit: Option<EarlyExitDecision> = None;
        let exit_layers = self.config.exit_layers.clone();

        for exit_layer in exit_layers {
            if exit_layer >= layer_hidden_states.len() {
                continue;
            }

            let hidden = layer_hidden_states[exit_layer];
            if let Some(decision) = self.evaluate_exit(hidden, batch, seq_len, exit_layer)? {
                match &best_exit {
                    None => best_exit = Some(decision),
                    Some(current) => {
                        if decision.exit_layer < current.exit_layer
                            && decision.confidence >= self.config.confidence_threshold
                        {
                            best_exit = Some(decision);
                        }
                    }
                }
            }
        }

        if let Some(exit) = best_exit {
            drafts.push(exit);
        }

        Ok(drafts)
    }

    /// Get dropout rate for a layer (training mode).
    #[inline(always)]
    pub fn get_dropout_rate(&self, layer_idx: usize) -> f32 {
        self.dropout_schedule
            .as_ref()
            .map(|s| s.get_rate(layer_idx))
            .unwrap_or(0.0)
    }

    /// Reset engine state.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.shared_activations.clear();
        self.stats = SpecEEStats::new(self.config.num_layers);
    }
}
