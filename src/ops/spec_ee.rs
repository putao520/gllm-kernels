//! SpecEE / LayerSkip Early-Exit Speculative Decoding.
//!
//! Based on:
//! - SpecEE (ISCA'25): 2.25-2.43x acceleration through early exit speculation
//! - LayerSkip (ACL'24): Self-speculative decoding using layer dropout
//!
//! # Key Features
//! - Per-layer early exit heads with confidence prediction
//! - Layer dropout training for early exit robustness
//! - Shared activation optimization between draft and verify phases
//! - Three-level predictor: algorithm + system + mapping

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Configuration for SpecEE/LayerSkip early exit.
#[derive(Debug, Clone)]
pub struct SpecEEConfig {
    /// Layers configured for early exit (e.g., [6, 12, 18]).
    pub exit_layers: Vec<usize>,
    /// Confidence threshold for early exit (default: 0.8).
    pub confidence_threshold: f32,
    /// Minimum layer index for exit (ensures quality).
    pub min_exit_layer: usize,
    /// Self-speculation depth.
    pub speculation_depth: usize,
    /// Enable layer dropout training mode.
    pub enable_layer_dropout: bool,
    /// Enable shared activations between draft and verify.
    pub share_activations: bool,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Total number of layers in the model.
    pub num_layers: usize,
}

impl Default for SpecEEConfig {
    fn default() -> Self {
        Self {
            exit_layers: vec![6, 12, 18],
            confidence_threshold: 0.8,
            min_exit_layer: 6,
            speculation_depth: 4,
            enable_layer_dropout: true,
            share_activations: true,
            hidden_dim: 768,
            vocab_size: 32000,
            num_layers: 24,
        }
    }
}

impl SpecEEConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.exit_layers.is_empty() {
            return Err("exit_layers must not be empty");
        }
        if self.confidence_threshold <= 0.0 || self.confidence_threshold > 1.0 {
            return Err("confidence_threshold must be in (0, 1]");
        }
        if self.min_exit_layer == 0 {
            return Err("min_exit_layer must be > 0");
        }
        if self.speculation_depth == 0 {
            return Err("speculation_depth must be > 0");
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0");
        }
        for &layer in &self.exit_layers {
            if layer >= self.num_layers {
                return Err("exit_layer index exceeds num_layers");
            }
            if layer < self.min_exit_layer {
                return Err("exit_layer below min_exit_layer");
            }
        }
        Ok(())
    }

    /// Create config with custom exit layers.
    pub fn with_exit_layers(exit_layers: Vec<usize>, num_layers: usize) -> Self {
        let min_exit = exit_layers.iter().cloned().min().unwrap_or(6);
        Self {
            exit_layers,
            min_exit_layer: min_exit,
            num_layers,
            ..Default::default()
        }
    }
}

/// Layer dropout rate function.
///
/// Following LayerSkip paper: low layers have low dropout, high layers have high dropout.
#[derive(Debug, Clone, Copy)]
pub struct LayerDropoutSchedule {
    /// Minimum dropout rate (for layer 0).
    pub min_rate: f32,
    /// Maximum dropout rate (for last layer).
    pub max_rate: f32,
    /// Total number of layers.
    pub num_layers: usize,
}

impl LayerDropoutSchedule {
    /// Create a linear dropout schedule.
    pub fn linear(min_rate: f32, max_rate: f32, num_layers: usize) -> Self {
        Self {
            min_rate: min_rate.clamp(0.0, 1.0),
            max_rate: max_rate.clamp(0.0, 1.0),
            num_layers,
        }
    }

    /// Get dropout rate for a specific layer.
    pub fn get_rate(&self, layer_idx: usize) -> f32 {
        if self.num_layers <= 1 {
            return self.min_rate;
        }
        let t = layer_idx as f32 / (self.num_layers - 1) as f32;
        self.min_rate + t * (self.max_rate - self.min_rate)
    }

    /// Check if a layer should be dropped (training only).
    pub fn should_drop(&self, layer_idx: usize, random_value: f32) -> bool {
        random_value < self.get_rate(layer_idx)
    }
}

/// Early exit head for a specific layer.
///
/// Contains both LM head (for token prediction) and confidence head.
#[derive(Debug, Clone)]
pub struct EarlyExitHead<B: Backend> {
    /// LM head weights: [hidden_dim, vocab_size].
    lm_head: Tensor<B, 2>,
    /// Confidence head weights: [hidden_dim, 1].
    confidence_head: Tensor<B, 2>,
    /// Confidence bias.
    confidence_bias: f32,
    /// Layer index this head is attached to.
    layer_idx: usize,
}

impl<B: Backend> EarlyExitHead<B> {
    /// Create a new early exit head.
    pub fn new(hidden_dim: usize, vocab_size: usize, layer_idx: usize, device: &B::Device) -> Self {
        // Initialize LM head with small random-like values
        let mut lm_data = vec![0.0f32; hidden_dim * vocab_size];
        for i in 0..lm_data.len() {
            let scale = (2.0 / (hidden_dim + vocab_size) as f32).sqrt();
            lm_data[i] = ((i % 13) as f32 - 6.0) * scale * 0.1;
        }
        let lm_head = Tensor::from_data(
            TensorData::new(lm_data, [hidden_dim, vocab_size]),
            device,
        );

        // Initialize confidence head
        let mut conf_data = vec![0.0f32; hidden_dim];
        for i in 0..conf_data.len() {
            let scale = (2.0 / hidden_dim as f32).sqrt();
            conf_data[i] = ((i % 7) as f32 - 3.0) * scale * 0.1;
        }
        let confidence_head = Tensor::from_data(
            TensorData::new(conf_data, [hidden_dim, 1]),
            device,
        );

        Self {
            lm_head,
            confidence_head,
            confidence_bias: 0.0,
            layer_idx,
        }
    }

    /// Load from pre-trained weights.
    pub fn from_weights(
        lm_head: Tensor<B, 2>,
        confidence_head: Tensor<B, 2>,
        confidence_bias: f32,
        layer_idx: usize,
    ) -> Result<Self, &'static str> {
        let [lm_h, _lm_v] = lm_head.dims();
        let [conf_h, conf_out] = confidence_head.dims();

        if lm_h != conf_h {
            return Err("lm_head and confidence_head hidden dim mismatch");
        }
        if conf_out != 1 {
            return Err("confidence_head output dim must be 1");
        }

        Ok(Self {
            lm_head,
            confidence_head,
            confidence_bias,
            layer_idx,
        })
    }

    /// Forward pass: compute logits and confidence.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states: [batch, seq_len, hidden_dim]
    ///
    /// # Returns
    /// * logits: [batch, seq_len, vocab_size]
    /// * confidence: [batch, seq_len]
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 2>), &'static str> {
        let [batch, seq_len, hidden_dim] = hidden.dims();
        let [lm_h, vocab_size] = self.lm_head.dims();
        let [conf_h, _] = self.confidence_head.dims();

        if hidden_dim != lm_h || hidden_dim != conf_h {
            return Err("hidden dimension mismatch");
        }

        // Reshape for matmul
        let hidden_2d = hidden.clone().reshape([batch * seq_len, hidden_dim]);

        // Compute logits
        let logits_2d = hidden_2d.clone().matmul(self.lm_head.clone());
        let logits = logits_2d.reshape([batch, seq_len, vocab_size]);

        // Compute confidence
        let conf_2d = hidden_2d.matmul(self.confidence_head.clone());
        let conf_data = conf_2d
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to convert confidence")?;

        let confidence: Vec<f32> = conf_data
            .iter()
            .map(|&x| sigmoid(x + self.confidence_bias))
            .collect();

        let device = hidden.device();
        let confidence = Tensor::from_data(
            TensorData::new(confidence, [batch, seq_len]),
            &device,
        );

        Ok((logits, confidence))
    }

    /// Forward pass returning only confidence (faster for exit decision).
    pub fn forward_confidence_only(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<Tensor<B, 2>, &'static str> {
        let [batch, seq_len, hidden_dim] = hidden.dims();
        let [conf_h, _] = self.confidence_head.dims();

        if hidden_dim != conf_h {
            return Err("hidden dimension mismatch");
        }

        let hidden_2d = hidden.clone().reshape([batch * seq_len, hidden_dim]);
        let conf_2d = hidden_2d.matmul(self.confidence_head.clone());

        let conf_data = conf_2d
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to convert confidence")?;

        let confidence: Vec<f32> = conf_data
            .iter()
            .map(|&x| sigmoid(x + self.confidence_bias))
            .collect();

        let device = hidden.device();
        Ok(Tensor::from_data(
            TensorData::new(confidence, [batch, seq_len]),
            &device,
        ))
    }

    /// Get layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
}

/// Early exit decision result.
#[derive(Debug, Clone)]
pub struct EarlyExitDecision {
    /// Layer index to exit from.
    pub exit_layer: usize,
    /// Confidence at exit layer.
    pub confidence: f32,
    /// Predicted token ID.
    pub token_id: usize,
    /// Log probability of predicted token.
    pub log_prob: f32,
    /// Whether this is an early exit (vs full forward).
    pub is_early_exit: bool,
}

/// Shared activations cache for draft-verify optimization.
#[derive(Debug, Clone)]
pub struct SharedActivations<B: Backend> {
    /// Cached hidden states per layer: layer_idx -> [batch, seq_len, hidden_dim].
    layer_hidden: Vec<Option<Tensor<B, 3>>>,
    /// Cached logits from early exit heads: layer_idx -> [batch, seq_len, vocab_size].
    layer_logits: Vec<Option<Tensor<B, 3>>>,
    /// Cached confidence scores: layer_idx -> [batch, seq_len].
    layer_confidence: Vec<Option<Tensor<B, 2>>>,
    /// Number of layers.
    num_layers: usize,
}

impl<B: Backend> SharedActivations<B> {
    /// Create a new shared activations cache.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layer_hidden: vec![None; num_layers],
            layer_logits: vec![None; num_layers],
            layer_confidence: vec![None; num_layers],
            num_layers,
        }
    }

    /// Store hidden states for a layer.
    pub fn store_hidden(&mut self, layer_idx: usize, hidden: Tensor<B, 3>) {
        if layer_idx < self.num_layers {
            self.layer_hidden[layer_idx] = Some(hidden);
        }
    }

    /// Store logits and confidence for a layer.
    pub fn store_exit_output(
        &mut self,
        layer_idx: usize,
        logits: Tensor<B, 3>,
        confidence: Tensor<B, 2>,
    ) {
        if layer_idx < self.num_layers {
            self.layer_logits[layer_idx] = Some(logits);
            self.layer_confidence[layer_idx] = Some(confidence);
        }
    }

    /// Get cached hidden states.
    pub fn get_hidden(&self, layer_idx: usize) -> Option<&Tensor<B, 3>> {
        self.layer_hidden.get(layer_idx)?.as_ref()
    }

    /// Get cached logits.
    pub fn get_logits(&self, layer_idx: usize) -> Option<&Tensor<B, 3>> {
        self.layer_logits.get(layer_idx)?.as_ref()
    }

    /// Get cached confidence.
    pub fn get_confidence(&self, layer_idx: usize) -> Option<&Tensor<B, 2>> {
        self.layer_confidence.get(layer_idx)?.as_ref()
    }

    /// Clear all cached activations.
    pub fn clear(&mut self) {
        for i in 0..self.num_layers {
            self.layer_hidden[i] = None;
            self.layer_logits[i] = None;
            self.layer_confidence[i] = None;
        }
    }

    /// Clear activations from a specific layer onwards (for partial recompute).
    pub fn clear_from(&mut self, start_layer: usize) {
        for i in start_layer..self.num_layers {
            self.layer_hidden[i] = None;
            self.layer_logits[i] = None;
            self.layer_confidence[i] = None;
        }
    }
}

/// Statistics for SpecEE inference.
#[derive(Debug, Clone, Default)]
pub struct SpecEEStats {
    /// Exit counts per layer.
    pub exit_layer_counts: Vec<usize>,
    /// Total early exits.
    pub total_early_exits: usize,
    /// Total full forwards.
    pub total_full_forwards: usize,
    /// Acceptance count (early exit matched full forward).
    pub accepted_count: usize,
    /// Rejection count (early exit differed from full forward).
    pub rejected_count: usize,
    /// Average exit layer.
    pub avg_exit_layer: f32,
    /// Acceptance rate.
    pub acceptance_rate: f32,
}

impl SpecEEStats {
    /// Initialize with number of layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            exit_layer_counts: vec![0; num_layers],
            ..Default::default()
        }
    }

    /// Update stats after an early exit.
    pub fn record_early_exit(&mut self, exit_layer: usize, accepted: bool) {
        if exit_layer < self.exit_layer_counts.len() {
            self.exit_layer_counts[exit_layer] += 1;
        }
        self.total_early_exits += 1;

        if accepted {
            self.accepted_count += 1;
        } else {
            self.rejected_count += 1;
        }

        self.update_derived();
    }

    /// Update stats after a full forward.
    pub fn record_full_forward(&mut self) {
        self.total_full_forwards += 1;
        self.update_derived();
    }

    fn update_derived(&mut self) {
        // Calculate average exit layer
        let total_exits: usize = self.exit_layer_counts.iter().sum();
        if total_exits > 0 {
            let weighted_sum: usize = self.exit_layer_counts
                .iter()
                .enumerate()
                .map(|(i, &count)| i * count)
                .sum();
            self.avg_exit_layer = weighted_sum as f32 / total_exits as f32;
        }

        // Calculate acceptance rate
        let total_attempts = self.accepted_count + self.rejected_count;
        if total_attempts > 0 {
            self.acceptance_rate = self.accepted_count as f32 / total_attempts as f32;
        }
    }

    /// Estimate speedup based on average exit layer.
    pub fn estimated_speedup(&self, num_layers: usize) -> f32 {
        if self.total_early_exits == 0 {
            return 1.0;
        }
        // Speedup ≈ num_layers / avg_exit_layer * acceptance_rate
        let exit_ratio = num_layers as f32 / (self.avg_exit_layer + 1.0);
        // Account for verification overhead and rejections
        let effective_rate = self.acceptance_rate * 0.9 + 0.1; // At least some benefit
        exit_ratio * effective_rate
    }
}

/// SpecEE / LayerSkip engine for self-speculative decoding.
#[derive(Debug)]
pub struct SpecEEEngine<B: Backend> {
    /// Configuration.
    config: SpecEEConfig,
    /// Early exit heads (one per exit layer).
    early_exit_heads: Vec<EarlyExitHead<B>>,
    /// Layer dropout schedule (for training).
    dropout_schedule: Option<LayerDropoutSchedule>,
    /// Shared activations cache.
    shared_activations: SharedActivations<B>,
    /// Runtime statistics.
    stats: SpecEEStats,
    /// Phantom marker.
    _marker: PhantomData<B>,
}

impl<B: Backend> SpecEEEngine<B> {
    /// Create a new SpecEE engine.
    pub fn new(config: SpecEEConfig, device: &B::Device) -> Result<Self, &'static str> {
        config.validate()?;

        let early_exit_heads: Vec<_> = config
            .exit_layers
            .iter()
            .map(|&layer_idx| {
                EarlyExitHead::new(
                    config.hidden_dim,
                    config.vocab_size,
                    layer_idx,
                    device,
                )
            })
            .collect();

        let dropout_schedule = if config.enable_layer_dropout {
            Some(LayerDropoutSchedule::linear(0.1, 0.5, config.num_layers))
        } else {
            None
        };

        let shared_activations = SharedActivations::new(config.num_layers);
        let stats = SpecEEStats::new(config.num_layers);

        Ok(Self {
            config,
            early_exit_heads,
            dropout_schedule,
            shared_activations,
            stats,
            _marker: PhantomData,
        })
    }

    /// Get configuration.
    pub fn config(&self) -> &SpecEEConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &SpecEEStats {
        &self.stats
    }

    /// Get mutable reference to shared activations.
    pub fn shared_activations_mut(&mut self) -> &mut SharedActivations<B> {
        &mut self.shared_activations
    }

    /// Evaluate early exit decision for a layer.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states from layer: [batch, seq_len, hidden_dim]
    /// * `layer_idx` - Current layer index
    ///
    /// # Returns
    /// Early exit decision if confidence exceeds threshold
    pub fn evaluate_exit(
        &mut self,
        hidden: Tensor<B, 3>,
        layer_idx: usize,
    ) -> Result<Option<EarlyExitDecision>, &'static str> {
        // Check if this is an exit layer
        let head_idx = self.config.exit_layers.iter().position(|&l| l == layer_idx);
        let head_idx = match head_idx {
            Some(idx) => idx,
            None => return Ok(None), // Not an exit layer
        };

        // Must be at or above minimum exit layer
        if layer_idx < self.config.min_exit_layer {
            return Ok(None);
        }

        let head = &self.early_exit_heads[head_idx];
        let (logits, confidence) = head.forward(hidden.clone())?;

        // Store in shared activations cache
        if self.config.share_activations {
            self.shared_activations.store_hidden(layer_idx, hidden);
            self.shared_activations
                .store_exit_output(layer_idx, logits.clone(), confidence.clone());
        }

        // Check confidence threshold (take last position for autoregressive)
        let conf_data = confidence
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to extract confidence")?;

        let [batch, seq_len, vocab_size] = logits.dims();
        let last_conf = conf_data[(batch - 1) * seq_len + (seq_len - 1)];

        if last_conf >= self.config.confidence_threshold {
            // Extract predicted token from last position
            let logits_data = logits
                .into_data()
                .into_vec::<f32>()
                .map_err(|_| "failed to extract logits")?;

            let last_logits_start = ((batch - 1) * seq_len + (seq_len - 1)) * vocab_size;
            let last_logits = &logits_data[last_logits_start..last_logits_start + vocab_size];
            let (token_id, log_prob) = find_top_token(last_logits);

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
    ///
    /// # Arguments
    /// * `early_exit` - Early exit decision
    /// * `full_token_id` - Token ID from full forward
    ///
    /// # Returns
    /// Whether the early exit was accepted
    pub fn verify_exit(&mut self, early_exit: &EarlyExitDecision, full_token_id: usize) -> bool {
        let accepted = early_exit.token_id == full_token_id;
        self.stats.record_early_exit(early_exit.exit_layer, accepted);
        accepted
    }

    /// Generate draft tokens using early exit.
    ///
    /// # Arguments
    /// * `layer_hidden_states` - Hidden states from all layers: [num_layers][batch, seq_len, hidden_dim]
    ///
    /// # Returns
    /// List of draft decisions (one per speculation step)
    pub fn generate_draft(
        &mut self,
        layer_hidden_states: &[Tensor<B, 3>],
    ) -> Result<Vec<EarlyExitDecision>, &'static str> {
        if layer_hidden_states.len() != self.config.num_layers {
            return Err("layer_hidden_states length mismatch");
        }

        let mut drafts = Vec::with_capacity(self.config.speculation_depth);

        // Find best exit layer
        let mut best_exit: Option<EarlyExitDecision> = None;

        // Clone exit_layers to avoid borrow conflict with self.evaluate_exit
        let exit_layers = self.config.exit_layers.clone();
        let confidence_threshold = self.config.confidence_threshold;

        for exit_layer in exit_layers {
            if exit_layer >= layer_hidden_states.len() {
                continue;
            }

            let hidden = layer_hidden_states[exit_layer].clone();
            if let Some(decision) = self.evaluate_exit(hidden, exit_layer)? {
                match &best_exit {
                    None => best_exit = Some(decision),
                    Some(current) => {
                        // Prefer earlier exit with sufficient confidence
                        if decision.exit_layer < current.exit_layer
                            && decision.confidence >= confidence_threshold
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
    pub fn get_dropout_rate(&self, layer_idx: usize) -> f32 {
        self.dropout_schedule
            .as_ref()
            .map(|s| s.get_rate(layer_idx))
            .unwrap_or(0.0)
    }

    /// Reset engine state.
    pub fn reset(&mut self) {
        self.shared_activations.clear();
        self.stats = SpecEEStats::new(self.config.num_layers);
    }
}

/// Self-speculation round result.
#[derive(Debug, Clone)]
pub struct SelfSpeculationResult {
    /// Final accepted token IDs.
    pub accepted_tokens: Vec<usize>,
    /// Number of tokens from early exit that were accepted.
    pub early_exit_accepted: usize,
    /// Bonus token from full forward (if early exit rejected).
    pub bonus_token: Option<usize>,
    /// Layers saved by early exit.
    pub layers_saved: usize,
}

/// Perform a self-speculation round.
///
/// # Arguments
/// * `early_exit_token` - Token from early exit
/// * `early_exit_layer` - Layer index of early exit
/// * `full_forward_token` - Token from full forward
/// * `num_layers` - Total layers in model
pub fn self_speculate(
    early_exit_token: usize,
    early_exit_layer: usize,
    full_forward_token: usize,
    num_layers: usize,
) -> SelfSpeculationResult {
    let accepted = early_exit_token == full_forward_token;

    if accepted {
        SelfSpeculationResult {
            accepted_tokens: vec![early_exit_token],
            early_exit_accepted: 1,
            bonus_token: None,
            layers_saved: num_layers - early_exit_layer - 1,
        }
    } else {
        SelfSpeculationResult {
            accepted_tokens: vec![full_forward_token],
            early_exit_accepted: 0,
            bonus_token: Some(full_forward_token),
            layers_saved: 0,
        }
    }
}

// Helper functions

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_spec_ee_config_validation() {
        let valid = SpecEEConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = SpecEEConfig {
            exit_layers: vec![],
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid = SpecEEConfig {
            confidence_threshold: 0.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_layer_dropout_schedule() {
        let schedule = LayerDropoutSchedule::linear(0.1, 0.5, 24);

        assert!((schedule.get_rate(0) - 0.1).abs() < 0.001);
        assert!((schedule.get_rate(23) - 0.5).abs() < 0.001);
        assert!(schedule.get_rate(12) > 0.1 && schedule.get_rate(12) < 0.5);
    }

    #[test]
    fn test_early_exit_head() {
        let device = <TestBackend as Backend>::Device::default();
        let head = EarlyExitHead::<TestBackend>::new(64, 100, 6, &device);

        let hidden = Tensor::zeros([1, 4, 64], &device);
        let result = head.forward(hidden);
        assert!(result.is_ok());

        let (logits, confidence) = result.unwrap();
        assert_eq!(logits.dims(), [1, 4, 100]);
        assert_eq!(confidence.dims(), [1, 4]);
    }

    #[test]
    fn test_shared_activations() {
        let device = <TestBackend as Backend>::Device::default();
        let mut cache = SharedActivations::<TestBackend>::new(24);

        let hidden = Tensor::zeros([1, 4, 64], &device);
        cache.store_hidden(6, hidden.clone());

        assert!(cache.get_hidden(6).is_some());
        assert!(cache.get_hidden(12).is_none());

        cache.clear_from(6);
        assert!(cache.get_hidden(6).is_none());
    }

    #[test]
    fn test_spec_ee_engine_creation() {
        let device = <TestBackend as Backend>::Device::default();
        let config = SpecEEConfig {
            hidden_dim: 64,
            vocab_size: 100,
            num_layers: 12,
            exit_layers: vec![4, 8],
            min_exit_layer: 4,
            ..Default::default()
        };

        let engine = SpecEEEngine::<TestBackend>::new(config, &device);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_self_speculation() {
        // Test accepted case
        let result = self_speculate(42, 12, 42, 24);
        assert_eq!(result.early_exit_accepted, 1);
        assert!(result.bonus_token.is_none());
        assert_eq!(result.layers_saved, 11);

        // Test rejected case
        let result = self_speculate(42, 12, 100, 24);
        assert_eq!(result.early_exit_accepted, 0);
        assert_eq!(result.bonus_token, Some(100));
        assert_eq!(result.layers_saved, 0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut stats = SpecEEStats::new(24);

        stats.record_early_exit(6, true);
        stats.record_early_exit(12, true);
        stats.record_early_exit(6, false);

        assert_eq!(stats.total_early_exits, 3);
        assert_eq!(stats.accepted_count, 2);
        assert_eq!(stats.rejected_count, 1);
        assert!(stats.acceptance_rate > 0.6);
    }
}
