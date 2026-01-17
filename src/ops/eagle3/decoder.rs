use crate::kernel_dispatcher::KernelFloat;
use crate::ops::math;

use super::config::AdaptiveDraftConfig;
use super::fuse::fuse_multi_layer_hidden;
use super::predictor::ConfidencePredictor;
use super::scheduler::LengthScheduler;
use super::types::{Eagle3Draft, Eagle3DraftToken, Eagle3Stats, Eagle3Verification};

/// EAGLE-3 adaptive speculative decoder.
#[derive(Debug, Clone)]
pub struct Eagle3Decoder<T: KernelFloat> {
    /// Configuration.
    config: AdaptiveDraftConfig,
    /// Confidence predictor.
    confidence_predictor: ConfidencePredictor<T>,
    /// Length scheduler (optional).
    length_scheduler: Option<LengthScheduler>,
    /// Current draft length target.
    current_draft_length: usize,
    /// Runtime statistics.
    stats: Eagle3Stats,
}

impl<T: KernelFloat> Eagle3Decoder<T> {
    /// Create a new EAGLE-3 decoder.
    #[inline(always)]
    pub fn new(config: AdaptiveDraftConfig) -> Result<Self, &'static str> {
        config.validate()?;

        let confidence_predictor = ConfidencePredictor::new(
            config.hidden_dim,
            config.fusion_layers,
        )?;

        let length_scheduler = if config.enable_length_scheduler {
            Some(LengthScheduler::new(
                config.min_draft_length,
                config.max_draft_length,
                0.1,
            )?)
        } else {
            None
        };

        let initial_length = config.min_draft_length +
            (config.max_draft_length - config.min_draft_length) / 2;

        Ok(Self {
            config,
            confidence_predictor,
            length_scheduler,
            current_draft_length: initial_length,
            stats: Eagle3Stats::default(),
        })
    }

    /// Get current configuration.
    #[inline(always)]
    pub fn config(&self) -> &AdaptiveDraftConfig {
        &self.config
    }

    /// Get current statistics.
    #[inline(always)]
    pub fn stats(&self) -> &Eagle3Stats {
        &self.stats
    }

    /// Get current target draft length.
    #[inline(always)]
    pub fn current_draft_length(&self) -> usize {
        self.current_draft_length
    }

    /// Fuse hidden states from multiple layers.
    #[inline(always)]
    pub fn fuse_hidden_states(
        &self,
        layer_hidden_states: &[&[T]],
        batch: usize,
        seq_len: usize,
        output: &mut [T],
    ) -> Result<(), &'static str> {
        fuse_multi_layer_hidden(layer_hidden_states, &self.config, batch, seq_len, output)
    }

    /// Generate draft tokens with confidence-based early termination.
    #[inline(always)]
    pub fn generate_draft(
        &self,
        draft_logits: &[T],
        fused_hidden: &[T],
        vocab_size: usize,
        max_length: Option<usize>,
    ) -> Result<Eagle3Draft, &'static str> {
        if vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }

        let max_len = self.clamp_draft_length(max_length);
        let fused_dim = self.config.fused_dim();
        let num_positions = checked_positions(fused_hidden.len(), fused_dim, "fused_hidden")?;
        let num_logit_positions = checked_positions(draft_logits.len(), vocab_size, "draft_logits")?;

        if num_positions == 0 || num_logit_positions == 0 {
            return Err("empty hidden states or logits");
        }

        let max_positions = max_len.min(num_positions).min(num_logit_positions);
        if max_positions == 0 {
            return Err("max_positions must be > 0");
        }

        let (tokens, total_confidence, early_terminated) =
            self.collect_draft_tokens(draft_logits, fused_hidden, vocab_size, max_positions)?;

        let avg_confidence = if tokens.is_empty() {
            0.0
        } else {
            total_confidence / tokens.len() as f32
        };
        let suggested_next_length = self.suggested_next_length(avg_confidence);

        Ok(Eagle3Draft {
            tokens,
            avg_confidence,
            early_terminated,
            suggested_next_length,
        })
    }

    /// Verify draft against target model logits.
    #[inline(always)]
    pub fn verify_draft(
        &self,
        draft: &Eagle3Draft,
        target_logits: &[T],
        vocab_size: usize,
    ) -> Result<Eagle3Verification, &'static str> {
        if vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }
        if draft.tokens.is_empty() {
            return Ok(Eagle3Verification {
                accepted_count: 0,
                accepted_tokens: Vec::new(),
                bonus_token: None,
                rejection_confidence: None,
            });
        }

        let num_positions = checked_positions(target_logits.len(), vocab_size, "target_logits")?;
        if num_positions < draft.tokens.len() {
            return Err("target logits too short for draft");
        }

        let (accepted_tokens, rejection_confidence) =
            self.accept_draft_tokens(draft, target_logits, vocab_size)?;
        let bonus_token = self.select_bonus_token(
            &accepted_tokens,
            target_logits,
            vocab_size,
            num_positions,
        );

        Ok(Eagle3Verification {
            accepted_count: accepted_tokens.len(),
            accepted_tokens,
            bonus_token,
            rejection_confidence,
        })
    }

    /// Update decoder state after verification.
    #[inline(always)]
    pub fn update_after_verification(
        &mut self,
        draft: &Eagle3Draft,
        verification: &Eagle3Verification,
    ) {
        let draft_len = draft.tokens.len();
        let accepted = verification.accepted_count;

        if let Some(scheduler) = &mut self.length_scheduler {
            scheduler.update(draft_len, accepted);
            self.current_draft_length = scheduler.suggest_length();
        } else {
            self.current_draft_length = self.fallback_length_update(draft_len, accepted);
        }

        self.stats.update(draft_len, accepted, draft.early_terminated);
    }

    /// Reset decoder state (statistics and scheduler).
    #[inline(always)]
    pub fn reset(&mut self) {
        self.stats = Eagle3Stats::default();
        if let Some(scheduler) = &mut self.length_scheduler {
            scheduler.reset();
        }
        self.current_draft_length = self.config.min_draft_length +
            (self.config.max_draft_length - self.config.min_draft_length) / 2;
    }

    #[inline(always)]
    fn clamp_draft_length(&self, max_length: Option<usize>) -> usize {
        let candidate = max_length.unwrap_or(self.current_draft_length);
        candidate.clamp(self.config.min_draft_length, self.config.max_draft_length)
    }

    #[inline(always)]
    fn collect_draft_tokens(
        &self,
        draft_logits: &[T],
        fused_hidden: &[T],
        vocab_size: usize,
        max_positions: usize,
    ) -> Result<(Vec<Eagle3DraftToken>, f32, bool), &'static str> {
        let fused_dim = self.config.fused_dim();
        let mut tokens = Vec::with_capacity(max_positions);
        let mut total_confidence = 0.0f32;
        let mut early_terminated = false;

        for pos in 0..max_positions {
            let hidden_start = pos * fused_dim;
            let hidden_end = hidden_start + fused_dim;
            let position_hidden = &fused_hidden[hidden_start..hidden_end];
            let confidence = self.confidence_predictor.predict_single(position_hidden)?;

            if confidence < self.config.confidence_threshold && pos > 0 {
                early_terminated = true;
                break;
            }

            let logit_start = pos * vocab_size;
            let logit_end = logit_start + vocab_size;
            let position_logits = &draft_logits[logit_start..logit_end];
            let (token_id, log_prob) = math::find_top_token(position_logits);

            tokens.push(Eagle3DraftToken {
                token_id,
                log_prob,
                confidence,
                position: pos,
            });
            total_confidence += confidence;
        }

        Ok((tokens, total_confidence, early_terminated))
    }

    #[inline(always)]
    fn suggested_next_length(&self, avg_confidence: f32) -> usize {
        if avg_confidence > 0.8 {
            (self.current_draft_length + 1).min(self.config.max_draft_length)
        } else if avg_confidence < 0.4 {
            (self.current_draft_length - 1).max(self.config.min_draft_length)
        } else {
            self.current_draft_length
        }
    }

    #[inline(always)]
    fn accept_draft_tokens(
        &self,
        draft: &Eagle3Draft,
        target_logits: &[T],
        vocab_size: usize,
    ) -> Result<(Vec<usize>, Option<f32>), &'static str> {
        let mut accepted_tokens = Vec::new();
        let mut rejection_confidence = None;

        for (pos, draft_token) in draft.tokens.iter().enumerate() {
            if draft_token.token_id >= vocab_size {
                return Err("draft token_id out of range");
            }
            let logit_start = pos * vocab_size;
            let logit_end = logit_start + vocab_size;
            let position_logits = &target_logits[logit_start..logit_end];

            let log_probs = math::log_softmax(position_logits);
            let target_prob = log_probs[draft_token.token_id].exp();
            let draft_prob = draft_token.log_prob.exp();

            if target_prob >= draft_prob {
                accepted_tokens.push(draft_token.token_id);
            } else {
                rejection_confidence = Some(target_prob);
                break;
            }
        }

        Ok((accepted_tokens, rejection_confidence))
    }

    #[inline(always)]
    fn select_bonus_token(
        &self,
        accepted_tokens: &[usize],
        target_logits: &[T],
        vocab_size: usize,
        num_positions: usize,
    ) -> Option<usize> {
        let bonus_pos = accepted_tokens.len();
        if bonus_pos >= num_positions {
            return None;
        }

        let logit_start = bonus_pos * vocab_size;
        let logit_end = logit_start + vocab_size;
        let position_logits = &target_logits[logit_start..logit_end];
        let (token_id, _) = math::find_top_token(position_logits);
        Some(token_id)
    }

    #[inline(always)]
    fn fallback_length_update(&self, draft_len: usize, accepted: usize) -> usize {
        if accepted == draft_len {
            (self.current_draft_length + 1).min(self.config.max_draft_length)
        } else if accepted == 0 {
            self.config.fallback_length
        } else {
            accepted.max(self.config.min_draft_length)
        }
    }
}

#[inline(always)]
fn checked_positions(len: usize, stride: usize, label: &'static str) -> Result<usize, &'static str> {
    if stride == 0 {
        return Err("stride must be > 0");
    }
    if len % stride != 0 {
        return Err(match label {
            "fused_hidden" => "fused_hidden length must be a multiple of fused_dim",
            "draft_logits" => "draft_logits length must be a multiple of vocab_size",
            "target_logits" => "target_logits length must be a multiple of vocab_size",
            _ => "input length mismatch",
        });
    }
    Ok(len / stride)
}
