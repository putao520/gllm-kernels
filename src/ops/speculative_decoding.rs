//! Speculative decoding utilities for draft/target verification.

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Speculative decoder for draft/target model verification.
pub struct SpeculativeDecoder<B: Backend> {
    /// Prediction head configuration.
    prediction_config: PredictionConfig,
    /// Tree structure configuration.
    tree_config: TreeConfig,
    /// Maximum speculation length (hard cap).
    max_speculation_length: usize,
    _marker: PhantomData<B>,
}

/// Configuration for the prediction head.
#[derive(Debug, Clone, Copy)]
pub struct PredictionConfig {
    /// Hidden dimension (treated as vocab dimension for logits).
    pub hidden_dim: usize,
    /// Prediction head type.
    pub head_type: PredictionHeadType,
}

/// Prediction head choices.
#[derive(Debug, Clone, Copy)]
pub enum PredictionHeadType {
    /// EAGLE-style lightweight MLP head.
    Eagle { num_layers: usize },
    /// Early-exit head with probability threshold.
    EarlyExit { exit_threshold: f32 },
}

/// Configuration for speculative tree expansion.
#[derive(Debug, Clone, Copy)]
pub struct TreeConfig {
    /// Branch factor per node.
    pub branch_factor: usize,
    /// Tree depth.
    pub depth: usize,
    /// Verification strategy for acceptance.
    pub verification: VerificationStrategy,
}

/// Verification strategy for speculative decoding.
#[derive(Debug, Clone, Copy)]
pub enum VerificationStrategy {
    /// Greedy verification (deterministic).
    Greedy,
    /// Sampling-based verification with temperature.
    Sampling { temperature: f32 },
}

/// Token entry in the speculative tree.
#[derive(Debug, Clone, Copy)]
pub struct SpeculativeToken {
    /// Token id (vocab index).
    pub id: usize,
    /// Draft log-probability.
    pub log_prob: f32,
}

/// Speculative token tree arranged by depth levels.
#[derive(Debug, Clone)]
pub struct SpeculativeTree {
    /// Tokens per depth level (BFS order, repeated per parent).
    pub levels: Vec<Vec<SpeculativeToken>>,
}

/// Batch of speculative trees.
#[derive(Debug, Clone)]
pub struct SpeculativeCandidates {
    /// Trees for each batch item.
    pub trees: Vec<SpeculativeTree>,
    /// Branch factor used to build trees.
    pub branch_factor: usize,
    /// Maximum depth across the batch.
    pub max_depth: usize,
    /// Vocabulary size inferred from logits.
    pub vocab_size: usize,
}

/// Verification output with accepted tokens and updated cache.
#[derive(Debug)]
pub struct SpeculativeVerification<B: Backend> {
    /// Accepted tokens per batch item.
    pub accepted_tokens: Vec<Vec<usize>>,
    /// Updated cache tokens with padding for shorter accept sequences.
    pub updated_cache: Tensor<B, 2>,
}

impl<B: Backend> SpeculativeDecoder<B> {
    /// Create a new speculative decoder.
    pub fn new(
        prediction_config: PredictionConfig,
        tree_config: TreeConfig,
        max_speculation_length: usize,
    ) -> Self {
        Self {
            prediction_config,
            tree_config,
            max_speculation_length,
            _marker: PhantomData,
        }
    }

    /// Access prediction head configuration.
    pub fn prediction_config(&self) -> &PredictionConfig {
        &self.prediction_config
    }

    /// Access tree configuration.
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    /// Maximum speculation length.
    pub fn max_speculation_length(&self) -> usize {
        self.max_speculation_length
    }

    /// Generate speculative token trees from hidden states.
    ///
    /// # Shapes
    /// * `hidden`: [batch, seq_len, hidden_dim]
    pub fn speculate(&self, hidden: Tensor<B, 3>) -> Result<SpeculativeCandidates, &'static str> {
        self.prediction_config.validate()?;
        self.tree_config.validate()?;
        if self.max_speculation_length == 0 {
            return Err("max speculation length must be > 0");
        }

        let [batch, seq_len, hidden_dim] = hidden.dims();
        if seq_len == 0 {
            return Err("sequence length must be > 0");
        }
        if hidden_dim != self.prediction_config.hidden_dim {
            return Err("hidden dimension mismatch");
        }

        let hidden_data = hidden
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "hidden data conversion failed")?;

        let depth_cap = self
            .tree_config
            .depth
            .min(self.max_speculation_length);
        if depth_cap == 0 {
            return Err("speculation depth must be > 0");
        }

        let mut trees = Vec::with_capacity(batch);
        let mut max_depth = 0;

        for batch_idx in 0..batch {
            let base = batch_idx * seq_len * hidden_dim;
            let offset = base + (seq_len - 1) * hidden_dim;
            let mut logits = hidden_data[offset..offset + hidden_dim].to_vec();
            apply_prediction_head(self.prediction_config.head_type, &mut logits)?;
            let log_probs = log_softmax(&logits);

            let mut effective_depth = depth_cap;
            if let PredictionHeadType::EarlyExit { exit_threshold } =
                self.prediction_config.head_type
            {
                let max_log_prob = log_probs
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let max_prob = max_log_prob.exp();
                if max_prob >= exit_threshold {
                    effective_depth = 1;
                }
            }

            let mut levels = Vec::with_capacity(effective_depth);
            let top_k = top_k_indices(&log_probs, self.tree_config.branch_factor);
            let mut parents = 1usize;
            for _depth in 0..effective_depth {
                let mut level = Vec::with_capacity(parents * top_k.len());
                for _ in 0..parents {
                    for &token in &top_k {
                        level.push(SpeculativeToken {
                            id: token,
                            log_prob: log_probs[token],
                        });
                    }
                }
                parents = parents.saturating_mul(top_k.len().max(1));
                levels.push(level);
            }

            max_depth = max_depth.max(effective_depth);
            trees.push(SpeculativeTree { levels });
        }

        Ok(SpeculativeCandidates {
            trees,
            branch_factor: self.tree_config.branch_factor,
            max_depth,
            vocab_size: hidden_dim,
        })
    }

    /// Verify candidates against target logits with rejection-style acceptance.
    ///
    /// # Shapes
    /// * `target_logits`: [batch, depth, vocab]
    /// * `cache_tokens`: [batch, cache_len]
    pub fn verify(
        &self,
        candidates: &SpeculativeCandidates,
        target_logits: Tensor<B, 3>,
        cache_tokens: Tensor<B, 2>,
    ) -> Result<SpeculativeVerification<B>, &'static str> {
        let [batch, depth, vocab] = target_logits.dims();
        if batch != candidates.trees.len() {
            return Err("target batch mismatch");
        }
        if vocab != candidates.vocab_size {
            return Err("target vocab mismatch");
        }
        if depth < candidates.max_depth {
            return Err("target logits depth too small");
        }

        let [cache_batch, cache_len] = cache_tokens.dims();
        if cache_batch != batch {
            return Err("cache batch mismatch");
        }

        let target_data = target_logits
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "target logits conversion failed")?;
        let cache_device = cache_tokens.device();
        let cache_data = cache_tokens
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "cache conversion failed")?;

        let mut accepted_tokens = Vec::with_capacity(batch);
        for (batch_idx, tree) in candidates.trees.iter().enumerate() {
            let mut accepted = Vec::new();
            for (depth_idx, level) in tree.levels.iter().enumerate() {
                let offset = (batch_idx * depth + depth_idx) * vocab;
                let mut logits = target_data[offset..offset + vocab].to_vec();
                if let VerificationStrategy::Sampling { temperature } = self.tree_config.verification
                {
                    if temperature <= 0.0 {
                        return Err("temperature must be > 0");
                    }
                    for value in logits.iter_mut() {
                        *value /= temperature;
                    }
                }
                let log_probs = log_softmax(&logits);

                let mut best_token = None;
                let mut best_prob = f32::NEG_INFINITY;
                let mut best_draft_prob = 0.0f32;
                for token in level {
                    let target_prob = log_probs[token.id].exp();
                    if target_prob > best_prob {
                        best_prob = target_prob;
                        best_token = Some(token.id);
                        best_draft_prob = token.log_prob.exp();
                    }
                }

                let token_id = match best_token {
                    Some(id) => id,
                    None => break,
                };

                if best_prob >= best_draft_prob {
                    accepted.push(token_id);
                } else {
                    break;
                }
            }
            accepted_tokens.push(accepted);
        }

        let max_accept = accepted_tokens
            .iter()
            .map(|tokens| tokens.len())
            .max()
            .unwrap_or(0);
        let new_len = cache_len + max_accept;
        let mut updated = vec![-1.0f32; batch * new_len];
        for batch_idx in 0..batch {
            let src_offset = batch_idx * cache_len;
            let dst_offset = batch_idx * new_len;
            updated[dst_offset..dst_offset + cache_len]
                .copy_from_slice(&cache_data[src_offset..src_offset + cache_len]);
            for (idx, token) in accepted_tokens[batch_idx].iter().enumerate() {
                updated[dst_offset + cache_len + idx] = *token as f32;
            }
        }

        let updated_cache =
            Tensor::from_data(TensorData::new(updated, [batch, new_len]), &cache_device);

        Ok(SpeculativeVerification {
            accepted_tokens,
            updated_cache,
        })
    }
}

impl PredictionConfig {
    /// Validate prediction head configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        self.head_type.validate()
    }
}

impl PredictionHeadType {
    fn validate(&self) -> Result<(), &'static str> {
        match *self {
            PredictionHeadType::Eagle { num_layers } => {
                if num_layers == 0 {
                    return Err("num_layers must be > 0");
                }
            }
            PredictionHeadType::EarlyExit { exit_threshold } => {
                if exit_threshold <= 0.0 || exit_threshold > 1.0 {
                    return Err("exit_threshold must be in (0, 1]");
                }
            }
        }
        Ok(())
    }
}

impl TreeConfig {
    /// Validate tree configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.branch_factor == 0 {
            return Err("branch_factor must be > 0");
        }
        if self.depth == 0 {
            return Err("depth must be > 0");
        }
        if let VerificationStrategy::Sampling { temperature } = self.verification {
            if temperature <= 0.0 {
                return Err("temperature must be > 0");
            }
        }
        Ok(())
    }
}

fn apply_prediction_head(
    head_type: PredictionHeadType,
    logits: &mut [f32],
) -> Result<(), &'static str> {
    match head_type {
        PredictionHeadType::Eagle { num_layers } => {
            for _ in 0..num_layers {
                for value in logits.iter_mut() {
                    let gate = 1.0 / (1.0 + (-*value).exp());
                    *value = value.tanh() * gate;
                }
            }
        }
        PredictionHeadType::EarlyExit { .. } => {}
    }
    Ok(())
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![max; logits.len()];
    }
    let mut sum = 0.0f32;
    for value in logits {
        sum += (value - max).exp();
    }
    let log_sum = max + sum.ln();
    logits.iter().map(|value| value - log_sum).collect()
}

fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .map(|(idx, &score)| {
            let score = if score.is_nan() {
                f32::NEG_INFINITY
            } else {
                score
            };
            (idx, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k.min(scored.len()));
    scored.into_iter().map(|(idx, _)| idx).collect()
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_speculate_tree_depth() {
        let config = PredictionConfig {
            hidden_dim: 4,
            head_type: PredictionHeadType::Eagle { num_layers: 2 },
        };
        let tree_config = TreeConfig {
            branch_factor: 2,
            depth: 3,
            verification: VerificationStrategy::Greedy,
        };
        let decoder = SpeculativeDecoder::<NdArray<f32>>::new(config, tree_config, 2);
        let device = <NdArray<f32> as Backend>::Device::default();
        let data = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.0, -0.1,
        ];
        let hidden = Tensor::from_data(TensorData::new(data, [1, 3, 4]), &device);

        let candidates = decoder.speculate(hidden).expect("speculate");
        assert_eq!(candidates.trees.len(), 1);
        assert_eq!(candidates.max_depth, 2);
        assert_eq!(candidates.trees[0].levels.len(), 2);
        assert_eq!(candidates.trees[0].levels[0].len(), 2);
        assert_eq!(candidates.trees[0].levels[1].len(), 4);
    }

    #[test]
    fn test_verify_rejects_on_low_target_prob() {
        let config = PredictionConfig {
            hidden_dim: 3,
            head_type: PredictionHeadType::EarlyExit { exit_threshold: 0.5 },
        };
        let tree_config = TreeConfig {
            branch_factor: 2,
            depth: 2,
            verification: VerificationStrategy::Greedy,
        };
        let decoder = SpeculativeDecoder::<NdArray<f32>>::new(config, tree_config, 2);
        let device = <NdArray<f32> as Backend>::Device::default();
        let hidden = Tensor::from_data(
            TensorData::new(vec![0.2, 0.1, 0.0], [1, 1, 3]),
            &device,
        );
        let candidates = decoder.speculate(hidden).expect("speculate");

        let target_logits = Tensor::from_data(
            TensorData::new(vec![0.0, 2.0, 0.0, -2.0, -2.0, 5.0], [1, 2, 3]),
            &device,
        );
        let cache_tokens =
            Tensor::from_data(TensorData::new(vec![1.0, 2.0], [1, 2]), &device);

        let result = decoder
            .verify(&candidates, target_logits, cache_tokens)
            .expect("verify");
        assert_eq!(result.accepted_tokens.len(), 1);
        assert_eq!(result.accepted_tokens[0].len(), 1);
        let updated = result
            .updated_cache
            .into_data()
            .into_vec::<f32>()
            .expect("cache data");
        assert_eq!(updated.len(), 3);
    }
}
