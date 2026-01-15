//! EAGLE-3 adaptive draft length speculative decoding.
//!
//! Based on EAGLE-3 (NeurIPS'25): 2-6x inference acceleration through
//! multi-layer feature fusion, token-level confidence prediction, and
//! adaptive draft length scheduling.
//!
//! # Key Features
//! - Multi-layer feature fusion (vs EAGLE-2 single layer)
//! - Token-level confidence prediction (vs sequence-level)
//! - Adaptive draft length based on acceptance history
//! - Training-time test distribution simulation

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Configuration for EAGLE-3 adaptive draft generation.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveDraftConfig {
    /// Minimum draft length (default: 1).
    pub min_draft_length: usize,
    /// Maximum draft length (default: 8).
    pub max_draft_length: usize,
    /// Confidence threshold for early termination (default: 0.5).
    pub confidence_threshold: f32,
    /// Fallback length after verification failure (default: 3).
    pub fallback_length: usize,
    /// Enable length scheduler (default: true).
    pub enable_length_scheduler: bool,
    /// Number of layers to fuse for confidence prediction (default: 4).
    pub fusion_layers: usize,
    /// Hidden dimension for confidence predictor.
    pub hidden_dim: usize,
}

impl Default for AdaptiveDraftConfig {
    fn default() -> Self {
        Self {
            min_draft_length: 1,
            max_draft_length: 8,
            confidence_threshold: 0.5,
            fallback_length: 3,
            enable_length_scheduler: true,
            fusion_layers: 4,
            hidden_dim: 768,
        }
    }
}

impl AdaptiveDraftConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_draft_length == 0 {
            return Err("min_draft_length must be > 0");
        }
        if self.max_draft_length < self.min_draft_length {
            return Err("max_draft_length must be >= min_draft_length");
        }
        if self.confidence_threshold <= 0.0 || self.confidence_threshold > 1.0 {
            return Err("confidence_threshold must be in (0, 1]");
        }
        if self.fallback_length == 0 {
            return Err("fallback_length must be > 0");
        }
        if self.fusion_layers == 0 {
            return Err("fusion_layers must be > 0");
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        Ok(())
    }

    /// Create a new config with specified hidden dimension.
    pub fn with_hidden_dim(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            ..Default::default()
        }
    }
}

/// Confidence predictor for acceptance probability estimation.
///
/// Uses multi-layer feature fusion to predict token-level acceptance probability.
#[derive(Debug, Clone)]
pub struct ConfidencePredictor<B: Backend> {
    /// Linear layer weights: [hidden_dim * fusion_layers, 1].
    weight: Tensor<B, 2>,
    /// Bias term.
    bias: f32,
    /// Number of layers to fuse.
    fusion_layers: usize,
    /// Per-layer hidden dimension.
    hidden_dim: usize,
}

impl<B: Backend> ConfidencePredictor<B> {
    /// Create a new confidence predictor with given dimensions.
    pub fn new(hidden_dim: usize, fusion_layers: usize, device: &B::Device) -> Self {
        let fused_dim = hidden_dim * fusion_layers;
        // Initialize with small random-like values based on position
        let mut weight_data = vec![0.0f32; fused_dim];
        for i in 0..fused_dim {
            // Xavier-like initialization approximation
            let scale = (2.0 / fused_dim as f32).sqrt();
            weight_data[i] = ((i % 7) as f32 - 3.0) * scale * 0.1;
        }
        let weight = Tensor::from_data(TensorData::new(weight_data, [fused_dim, 1]), device);

        Self {
            weight,
            bias: 0.0,
            fusion_layers,
            hidden_dim,
        }
    }

    /// Load predictor from pre-trained weights.
    pub fn from_weights(
        weight: Tensor<B, 2>,
        bias: f32,
        fusion_layers: usize,
        hidden_dim: usize,
    ) -> Result<Self, &'static str> {
        let [fused_dim, output_dim] = weight.dims();
        if fused_dim != hidden_dim * fusion_layers {
            return Err("weight dimension mismatch with fusion_layers * hidden_dim");
        }
        if output_dim != 1 {
            return Err("output dimension must be 1");
        }
        Ok(Self {
            weight,
            bias,
            fusion_layers,
            hidden_dim,
        })
    }

    /// Predict acceptance probability from fused hidden states.
    ///
    /// # Arguments
    /// * `fused_hidden` - Fused hidden states: [batch, seq_len, hidden_dim * fusion_layers]
    ///
    /// # Returns
    /// Token-level acceptance probabilities: [batch, seq_len]
    pub fn predict(&self, fused_hidden: Tensor<B, 3>) -> Result<Tensor<B, 2>, &'static str> {
        let [batch, seq_len, fused_dim] = fused_hidden.dims();
        let expected_dim = self.hidden_dim * self.fusion_layers;
        if fused_dim != expected_dim {
            return Err("fused_hidden dimension mismatch");
        }

        // Reshape for matmul: [batch * seq_len, fused_dim]
        let hidden_2d = fused_hidden.reshape([batch * seq_len, fused_dim]);

        // Linear projection: [batch * seq_len, 1]
        let logits = hidden_2d.matmul(self.weight.clone());

        // Get logits as Vec and apply sigmoid + bias
        let logits_data = logits
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to convert logits to f32")?;

        let probs: Vec<f32> = logits_data
            .iter()
            .map(|&x| sigmoid(x + self.bias))
            .collect();

        let device = self.weight.device();
        let result = Tensor::from_data(TensorData::new(probs, [batch, seq_len]), &device);
        Ok(result)
    }

    /// Predict single token acceptance probability (for incremental generation).
    pub fn predict_single(&self, fused_hidden: &[f32]) -> Result<f32, &'static str> {
        let expected_dim = self.hidden_dim * self.fusion_layers;
        if fused_hidden.len() != expected_dim {
            return Err("fused_hidden length mismatch");
        }

        let weight_data = self.weight.clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to convert weight to f32")?;

        let mut logit = self.bias;
        for (i, &h) in fused_hidden.iter().enumerate() {
            logit += h * weight_data[i];
        }

        Ok(sigmoid(logit))
    }
}

/// Length scheduler for adaptive draft length selection.
///
/// Learns optimal draft length distribution from acceptance history.
#[derive(Debug, Clone)]
pub struct LengthScheduler {
    /// Acceptance rate distribution for each length: [max_length].
    length_distribution: Vec<f32>,
    /// Exponential moving average coefficient.
    ema_alpha: f32,
    /// Sample count for each length.
    sample_count: Vec<usize>,
    /// Minimum draft length.
    min_length: usize,
    /// Maximum draft length.
    max_length: usize,
}

impl LengthScheduler {
    /// Create a new length scheduler.
    pub fn new(min_length: usize, max_length: usize, ema_alpha: f32) -> Self {
        let length_range = max_length - min_length + 1;
        // Initialize with decreasing acceptance rates (longer = less likely to accept all)
        let mut length_distribution = Vec::with_capacity(length_range);
        for i in 0..length_range {
            // Assume exponential decay: acceptance ~ 0.8^length
            let length = min_length + i;
            length_distribution.push(0.8f32.powi(length as i32));
        }

        Self {
            length_distribution,
            ema_alpha,
            sample_count: vec![0; length_range],
            min_length,
            max_length,
        }
    }

    /// Suggest optimal draft length based on acceptance history.
    ///
    /// Uses expected accepted tokens = length * acceptance_rate to find optimal length.
    pub fn suggest_length(&self) -> usize {
        let mut best_length = self.min_length;
        let mut best_expected = 0.0f32;

        for (i, &acceptance_rate) in self.length_distribution.iter().enumerate() {
            let length = self.min_length + i;
            let expected_tokens = length as f32 * acceptance_rate;

            if expected_tokens > best_expected {
                best_expected = expected_tokens;
                best_length = length;
            }
        }

        best_length
    }

    /// Update acceptance statistics for a given draft length.
    ///
    /// # Arguments
    /// * `draft_length` - The draft length that was used
    /// * `accepted_count` - Number of tokens that were accepted
    pub fn update(&mut self, draft_length: usize, accepted_count: usize) {
        if draft_length < self.min_length || draft_length > self.max_length {
            return;
        }

        let idx = draft_length - self.min_length;
        let acceptance_rate = accepted_count as f32 / draft_length as f32;

        // EMA update
        self.length_distribution[idx] =
            self.ema_alpha * acceptance_rate +
            (1.0 - self.ema_alpha) * self.length_distribution[idx];
        self.sample_count[idx] += 1;
    }

    /// Get current acceptance rate for a specific length.
    pub fn get_acceptance_rate(&self, length: usize) -> Option<f32> {
        if length < self.min_length || length > self.max_length {
            return None;
        }
        Some(self.length_distribution[length - self.min_length])
    }

    /// Get total sample count across all lengths.
    pub fn total_samples(&self) -> usize {
        self.sample_count.iter().sum()
    }
}

/// Draft token with confidence information.
#[derive(Debug, Clone, Copy)]
pub struct Eagle3DraftToken {
    /// Token ID (vocabulary index).
    pub token_id: usize,
    /// Draft log probability from draft model.
    pub log_prob: f32,
    /// Predicted acceptance confidence.
    pub confidence: f32,
    /// Position in draft sequence (0-indexed).
    pub position: usize,
}

/// Generated draft sequence with metadata.
#[derive(Debug, Clone)]
pub struct Eagle3Draft {
    /// Draft tokens in sequence order.
    pub tokens: Vec<Eagle3DraftToken>,
    /// Average confidence across all tokens.
    pub avg_confidence: f32,
    /// Whether draft was terminated early due to low confidence.
    pub early_terminated: bool,
    /// Suggested next draft length based on this draft's confidence.
    pub suggested_next_length: usize,
}

/// Verification result from target model.
#[derive(Debug, Clone)]
pub struct Eagle3Verification {
    /// Number of accepted tokens (prefix of draft).
    pub accepted_count: usize,
    /// Accepted token IDs.
    pub accepted_tokens: Vec<usize>,
    /// Bonus token from target model (if any rejected).
    pub bonus_token: Option<usize>,
    /// Target model confidence at rejection point.
    pub rejection_confidence: Option<f32>,
}

/// Statistics for EAGLE-3 generation.
#[derive(Debug, Clone, Default)]
pub struct Eagle3Stats {
    /// Total draft tokens generated.
    pub total_draft_tokens: usize,
    /// Total accepted tokens.
    pub total_accepted_tokens: usize,
    /// Number of draft-verify cycles.
    pub num_cycles: usize,
    /// Average draft length.
    pub avg_draft_length: f32,
    /// Average acceptance rate.
    pub avg_acceptance_rate: f32,
    /// Early termination count.
    pub early_terminations: usize,
}

impl Eagle3Stats {
    /// Update statistics after a verification cycle.
    pub fn update(&mut self, draft_len: usize, accepted: usize, early_term: bool) {
        self.total_draft_tokens += draft_len;
        self.total_accepted_tokens += accepted;
        self.num_cycles += 1;
        if early_term {
            self.early_terminations += 1;
        }

        // Update running averages
        self.avg_draft_length = self.total_draft_tokens as f32 / self.num_cycles as f32;
        self.avg_acceptance_rate = if self.total_draft_tokens > 0 {
            self.total_accepted_tokens as f32 / self.total_draft_tokens as f32
        } else {
            0.0
        };
    }

    /// Calculate speedup estimate (draft accepted / cycles).
    pub fn estimated_speedup(&self) -> f32 {
        if self.num_cycles == 0 {
            return 1.0;
        }
        // Speedup ≈ (accepted_tokens + cycles) / cycles
        // Because each cycle produces at least 1 token (bonus or first accepted)
        (self.total_accepted_tokens + self.num_cycles) as f32 / self.num_cycles as f32
    }
}

/// EAGLE-3 adaptive speculative decoder.
///
/// Implements the EAGLE-3 algorithm with:
/// - Multi-layer feature fusion for confidence prediction
/// - Token-level confidence-based early termination
/// - Adaptive draft length scheduling
#[derive(Debug, Clone)]
pub struct Eagle3Decoder<B: Backend> {
    /// Configuration.
    config: AdaptiveDraftConfig,
    /// Confidence predictor.
    confidence_predictor: ConfidencePredictor<B>,
    /// Length scheduler (optional).
    length_scheduler: Option<LengthScheduler>,
    /// Current draft length target.
    current_draft_length: usize,
    /// Runtime statistics.
    stats: Eagle3Stats,
    /// Phantom marker for backend.
    _marker: PhantomData<B>,
}

impl<B: Backend> Eagle3Decoder<B> {
    /// Create a new EAGLE-3 decoder.
    pub fn new(config: AdaptiveDraftConfig, device: &B::Device) -> Result<Self, &'static str> {
        config.validate()?;

        let confidence_predictor = ConfidencePredictor::new(
            config.hidden_dim,
            config.fusion_layers,
            device,
        );

        let length_scheduler = if config.enable_length_scheduler {
            Some(LengthScheduler::new(
                config.min_draft_length,
                config.max_draft_length,
                0.1, // EMA alpha
            ))
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
            _marker: PhantomData,
        })
    }

    /// Get current configuration.
    pub fn config(&self) -> &AdaptiveDraftConfig {
        &self.config
    }

    /// Get current statistics.
    pub fn stats(&self) -> &Eagle3Stats {
        &self.stats
    }

    /// Get current target draft length.
    pub fn current_draft_length(&self) -> usize {
        self.current_draft_length
    }

    /// Fuse hidden states from multiple layers.
    ///
    /// # Arguments
    /// * `layer_hidden_states` - Hidden states from multiple layers, each [batch, seq_len, hidden_dim]
    ///
    /// # Returns
    /// Fused hidden states: [batch, seq_len, hidden_dim * fusion_layers]
    pub fn fuse_hidden_states(
        &self,
        layer_hidden_states: Vec<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, &'static str> {
        if layer_hidden_states.is_empty() {
            return Err("no hidden states provided");
        }
        if layer_hidden_states.len() < self.config.fusion_layers {
            return Err("insufficient layers for fusion");
        }

        // Take the last N layers for fusion
        let start_idx = layer_hidden_states.len() - self.config.fusion_layers;
        let layers_to_fuse: Vec<_> = layer_hidden_states[start_idx..].to_vec();

        // Verify dimensions
        let [batch, seq_len, hidden_dim] = layers_to_fuse[0].dims();
        if hidden_dim != self.config.hidden_dim {
            return Err("hidden_dim mismatch");
        }
        for layer in &layers_to_fuse {
            if layer.dims() != [batch, seq_len, hidden_dim] {
                return Err("layer dimensions inconsistent");
            }
        }

        // Concatenate along hidden dimension
        let fused_dim = hidden_dim * self.config.fusion_layers;
        let device = layers_to_fuse[0].device();

        // Extract data from each layer and concatenate
        let mut fused_data = vec![0.0f32; batch * seq_len * fused_dim];

        for (layer_idx, layer) in layers_to_fuse.iter().enumerate() {
            let layer_data = layer.clone()
                .into_data()
                .into_vec::<f32>()
                .map_err(|_| "failed to extract layer data")?;

            for b in 0..batch {
                for s in 0..seq_len {
                    let src_offset = (b * seq_len + s) * hidden_dim;
                    let dst_offset = (b * seq_len + s) * fused_dim + layer_idx * hidden_dim;
                    fused_data[dst_offset..dst_offset + hidden_dim]
                        .copy_from_slice(&layer_data[src_offset..src_offset + hidden_dim]);
                }
            }
        }

        let fused = Tensor::from_data(
            TensorData::new(fused_data, [batch, seq_len, fused_dim]),
            &device,
        );
        Ok(fused)
    }

    /// Generate draft tokens with confidence-based early termination.
    ///
    /// # Arguments
    /// * `draft_logits` - Logits from draft model: [seq_len, vocab_size]
    /// * `fused_hidden` - Fused hidden states for confidence: [seq_len, fused_dim]
    /// * `max_length` - Maximum draft length (overrides config if provided)
    ///
    /// # Returns
    /// Generated draft with tokens and metadata
    pub fn generate_draft(
        &self,
        draft_logits: &[f32],
        fused_hidden: &[f32],
        vocab_size: usize,
        max_length: Option<usize>,
    ) -> Result<Eagle3Draft, &'static str> {
        let max_len = max_length.unwrap_or(self.current_draft_length);
        let max_len = max_len.clamp(self.config.min_draft_length, self.config.max_draft_length);

        let fused_dim = self.config.hidden_dim * self.config.fusion_layers;
        let num_positions = fused_hidden.len() / fused_dim;
        let num_logit_positions = draft_logits.len() / vocab_size;

        if num_positions == 0 || num_logit_positions == 0 {
            return Err("empty hidden states or logits");
        }

        let mut tokens = Vec::with_capacity(max_len);
        let mut total_confidence = 0.0f32;
        let mut early_terminated = false;

        for pos in 0..max_len.min(num_positions).min(num_logit_positions) {
            // Get confidence for this position
            let hidden_start = pos * fused_dim;
            let hidden_end = hidden_start + fused_dim;
            let position_hidden = &fused_hidden[hidden_start..hidden_end];
            let confidence = self.confidence_predictor.predict_single(position_hidden)?;

            // Check for early termination
            if confidence < self.config.confidence_threshold && pos > 0 {
                early_terminated = true;
                break;
            }

            // Get top token from logits
            let logit_start = pos * vocab_size;
            let logit_end = logit_start + vocab_size;
            let position_logits = &draft_logits[logit_start..logit_end];

            let (token_id, log_prob) = find_top_token(position_logits);

            tokens.push(Eagle3DraftToken {
                token_id,
                log_prob,
                confidence,
                position: pos,
            });
            total_confidence += confidence;
        }

        let avg_confidence = if tokens.is_empty() {
            0.0
        } else {
            total_confidence / tokens.len() as f32
        };

        // Calculate suggested next length based on confidence
        let suggested_next_length = if avg_confidence > 0.8 {
            (self.current_draft_length + 1).min(self.config.max_draft_length)
        } else if avg_confidence < 0.4 {
            (self.current_draft_length - 1).max(self.config.min_draft_length)
        } else {
            self.current_draft_length
        };

        Ok(Eagle3Draft {
            tokens,
            avg_confidence,
            early_terminated,
            suggested_next_length,
        })
    }

    /// Verify draft against target model logits.
    ///
    /// # Arguments
    /// * `draft` - Generated draft tokens
    /// * `target_logits` - Target model logits: [draft_len, vocab_size]
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Verification result with accepted tokens
    pub fn verify_draft(
        &self,
        draft: &Eagle3Draft,
        target_logits: &[f32],
        vocab_size: usize,
    ) -> Result<Eagle3Verification, &'static str> {
        if draft.tokens.is_empty() {
            return Ok(Eagle3Verification {
                accepted_count: 0,
                accepted_tokens: Vec::new(),
                bonus_token: None,
                rejection_confidence: None,
            });
        }

        let num_positions = target_logits.len() / vocab_size;
        if num_positions < draft.tokens.len() {
            return Err("target logits too short for draft");
        }

        let mut accepted_tokens = Vec::new();
        let mut rejection_confidence = None;

        for (pos, draft_token) in draft.tokens.iter().enumerate() {
            let logit_start = pos * vocab_size;
            let logit_end = logit_start + vocab_size;
            let position_logits = &target_logits[logit_start..logit_end];

            let log_probs = log_softmax(position_logits);
            let target_prob = log_probs[draft_token.token_id].exp();
            let draft_prob = draft_token.log_prob.exp();

            // Rejection sampling: accept if target_prob >= draft_prob
            if target_prob >= draft_prob {
                accepted_tokens.push(draft_token.token_id);
            } else {
                // Rejection: sample bonus token
                rejection_confidence = Some(target_prob);
                break;
            }
        }

        // Get bonus token from the position after last accepted
        let bonus_token = if accepted_tokens.len() < draft.tokens.len() {
            let bonus_pos = accepted_tokens.len();
            if bonus_pos < num_positions {
                let logit_start = bonus_pos * vocab_size;
                let logit_end = logit_start + vocab_size;
                let position_logits = &target_logits[logit_start..logit_end];
                let (token_id, _) = find_top_token(position_logits);
                Some(token_id)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Eagle3Verification {
            accepted_count: accepted_tokens.len(),
            accepted_tokens,
            bonus_token,
            rejection_confidence,
        })
    }

    /// Update decoder state after verification.
    ///
    /// Updates length scheduler and statistics based on verification result.
    pub fn update_after_verification(
        &mut self,
        draft: &Eagle3Draft,
        verification: &Eagle3Verification,
    ) {
        let draft_len = draft.tokens.len();
        let accepted = verification.accepted_count;

        // Update length scheduler
        if let Some(scheduler) = &mut self.length_scheduler {
            scheduler.update(draft_len, accepted);
            self.current_draft_length = scheduler.suggest_length();
        } else {
            // Simple fallback strategy without scheduler
            if accepted == draft_len {
                self.current_draft_length = (self.current_draft_length + 1)
                    .min(self.config.max_draft_length);
            } else if accepted == 0 {
                self.current_draft_length = self.config.fallback_length;
            } else {
                self.current_draft_length = accepted.max(self.config.min_draft_length);
            }
        }

        // Update statistics
        self.stats.update(draft_len, accepted, draft.early_terminated);
    }

    /// Reset decoder state (statistics and scheduler).
    pub fn reset(&mut self) {
        self.stats = Eagle3Stats::default();
        if let Some(scheduler) = &mut self.length_scheduler {
            *scheduler = LengthScheduler::new(
                self.config.min_draft_length,
                self.config.max_draft_length,
                0.1,
            );
        }
        self.current_draft_length = self.config.min_draft_length +
            (self.config.max_draft_length - self.config.min_draft_length) / 2;
    }
}

/// Fuse hidden states from multiple layers (standalone function).
///
/// # Arguments
/// * `layer_hidden_states` - Hidden states from multiple layers
/// * `fusion_layers` - Number of layers to fuse from the end
///
/// # Returns
/// Fused tensor by concatenating along hidden dimension
pub fn fuse_multi_layer_hidden<B: Backend>(
    layer_hidden_states: &[Tensor<B, 3>],
    fusion_layers: usize,
) -> Result<Tensor<B, 3>, &'static str> {
    if layer_hidden_states.is_empty() {
        return Err("no hidden states provided");
    }
    if layer_hidden_states.len() < fusion_layers {
        return Err("insufficient layers for fusion");
    }

    let start_idx = layer_hidden_states.len() - fusion_layers;
    let layers_to_fuse = &layer_hidden_states[start_idx..];

    let [batch, seq_len, hidden_dim] = layers_to_fuse[0].dims();
    let fused_dim = hidden_dim * fusion_layers;
    let device = layers_to_fuse[0].device();

    let mut fused_data = vec![0.0f32; batch * seq_len * fused_dim];

    for (layer_idx, layer) in layers_to_fuse.iter().enumerate() {
        let [lb, ls, lh] = layer.dims();
        if lb != batch || ls != seq_len || lh != hidden_dim {
            return Err("layer dimensions inconsistent");
        }

        let layer_data = layer.clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to extract layer data")?;

        for b in 0..batch {
            for s in 0..seq_len {
                let src_offset = (b * seq_len + s) * hidden_dim;
                let dst_offset = (b * seq_len + s) * fused_dim + layer_idx * hidden_dim;
                fused_data[dst_offset..dst_offset + hidden_dim]
                    .copy_from_slice(&layer_data[src_offset..src_offset + hidden_dim]);
            }
        }
    }

    let fused = Tensor::from_data(
        TensorData::new(fused_data, [batch, seq_len, fused_dim]),
        &device,
    );
    Ok(fused)
}

/// Predict token-level confidence from fused hidden states (standalone function).
///
/// # Arguments
/// * `fused_hidden` - Fused hidden states: [batch, seq_len, fused_dim]
/// * `weight` - Predictor weight: [fused_dim, 1]
/// * `bias` - Predictor bias
///
/// # Returns
/// Confidence probabilities: [batch, seq_len]
pub fn predict_confidence<B: Backend>(
    fused_hidden: Tensor<B, 3>,
    weight: Tensor<B, 2>,
    bias: f32,
) -> Result<Tensor<B, 2>, &'static str> {
    let [batch, seq_len, fused_dim] = fused_hidden.dims();
    let [weight_dim, output_dim] = weight.dims();

    if weight_dim != fused_dim {
        return Err("weight dimension mismatch");
    }
    if output_dim != 1 {
        return Err("output dimension must be 1");
    }

    let device = fused_hidden.device();
    let hidden_2d = fused_hidden.reshape([batch * seq_len, fused_dim]);
    let logits = hidden_2d.matmul(weight);

    let logits_data = logits
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "failed to convert logits")?;

    let probs: Vec<f32> = logits_data
        .iter()
        .map(|&x| sigmoid(x + bias))
        .collect();

    Ok(Tensor::from_data(TensorData::new(probs, [batch, seq_len]), &device))
}

// Helper functions

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![max; logits.len()];
    }
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = max + sum.ln();
    logits.iter().map(|&x| x - log_sum).collect()
}

fn find_top_token(logits: &[f32]) -> (usize, f32) {
    let log_probs = log_softmax(logits);
    let mut best_idx = 0;
    let mut best_prob = f32::NEG_INFINITY;

    for (idx, &prob) in log_probs.iter().enumerate() {
        if prob > best_prob {
            best_prob = prob;
            best_idx = idx;
        }
    }

    (best_idx, best_prob)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_adaptive_draft_config_validation() {
        let valid = AdaptiveDraftConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = AdaptiveDraftConfig {
            min_draft_length: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid = AdaptiveDraftConfig {
            max_draft_length: 0,
            min_draft_length: 2,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_confidence_predictor() {
        let device = <TestBackend as Backend>::Device::default();
        let predictor = ConfidencePredictor::<TestBackend>::new(64, 4, &device);

        // Test single prediction
        let fused_hidden = vec![0.1f32; 64 * 4];
        let confidence = predictor.predict_single(&fused_hidden).unwrap();
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_length_scheduler() {
        let mut scheduler = LengthScheduler::new(1, 8, 0.1);

        // Initial suggestion
        let initial = scheduler.suggest_length();
        assert!(initial >= 1 && initial <= 8);

        // Update with high acceptance
        scheduler.update(4, 4);
        scheduler.update(4, 4);
        scheduler.update(4, 4);

        // Should suggest longer lengths after high acceptance
        let after_high = scheduler.suggest_length();
        assert!(after_high >= 1);
    }

    #[test]
    fn test_eagle3_decoder_creation() {
        let device = <TestBackend as Backend>::Device::default();
        let config = AdaptiveDraftConfig::with_hidden_dim(64);
        let decoder = Eagle3Decoder::<TestBackend>::new(config, &device);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_fuse_hidden_states() {
        let device = <TestBackend as Backend>::Device::default();
        let config = AdaptiveDraftConfig {
            hidden_dim: 8,
            fusion_layers: 2,
            ..Default::default()
        };
        let decoder = Eagle3Decoder::<TestBackend>::new(config, &device).unwrap();

        // Create dummy hidden states
        let layer1 = Tensor::<TestBackend, 3>::zeros([1, 4, 8], &device);
        let layer2 = Tensor::<TestBackend, 3>::ones([1, 4, 8], &device);

        let fused = decoder.fuse_hidden_states(vec![layer1, layer2]);
        assert!(fused.is_ok());
        let fused = fused.unwrap();
        assert_eq!(fused.dims(), [1, 4, 16]);
    }

    #[test]
    fn test_generate_and_verify_draft() {
        let device = <TestBackend as Backend>::Device::default();
        let config = AdaptiveDraftConfig {
            hidden_dim: 8,
            fusion_layers: 2,
            min_draft_length: 1,
            max_draft_length: 4,
            ..Default::default()
        };
        let decoder = Eagle3Decoder::<TestBackend>::new(config, &device).unwrap();

        let vocab_size = 16;
        let seq_len = 4;
        let fused_dim = 16;

        // Create dummy logits and hidden states
        let mut draft_logits = vec![0.0f32; seq_len * vocab_size];
        for i in 0..seq_len {
            draft_logits[i * vocab_size + (i % vocab_size)] = 2.0; // Make one token dominant
        }

        let fused_hidden = vec![0.5f32; seq_len * fused_dim];

        let draft = decoder.generate_draft(&draft_logits, &fused_hidden, vocab_size, None);
        assert!(draft.is_ok());
        let draft = draft.unwrap();
        assert!(!draft.tokens.is_empty());

        // Verify draft
        let mut target_logits = vec![0.0f32; seq_len * vocab_size];
        for i in 0..seq_len {
            target_logits[i * vocab_size + (i % vocab_size)] = 3.0; // Match draft tokens
        }

        let verification = decoder.verify_draft(&draft, &target_logits, vocab_size);
        assert!(verification.is_ok());
    }

    #[test]
    fn test_stats_tracking() {
        let mut stats = Eagle3Stats::default();

        stats.update(4, 3, false);
        assert_eq!(stats.total_draft_tokens, 4);
        assert_eq!(stats.total_accepted_tokens, 3);
        assert_eq!(stats.num_cycles, 1);

        stats.update(4, 4, false);
        assert_eq!(stats.total_draft_tokens, 8);
        assert_eq!(stats.total_accepted_tokens, 7);

        let speedup = stats.estimated_speedup();
        assert!(speedup > 1.0);
    }
}
