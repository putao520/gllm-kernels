//! Medusa / Assisted Generation for parallel token prediction.
//!
//! Based on:
//! - Medusa (ICML'24): Multiple decode heads for parallel prediction
//! - Lookahead Decoding: N-gram assisted draft generation
//!
//! # Key Features
//! - Multiple Medusa heads for predicting future tokens
//! - N-gram cache for assisted drafting
//! - Tree-structured candidate generation
//! - Compatible with DeFT tree attention for verification

use std::collections::HashMap;
use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Configuration for Medusa assisted generation.
#[derive(Debug, Clone)]
pub struct AssistedGenerationConfig {
    /// Number of Medusa heads (default: 3).
    pub num_medusa_heads: usize,
    /// Speculation depth per head (default: 4).
    pub speculation_depth: usize,
    /// Number of candidate tokens per position (default: 8).
    pub candidate_count: usize,
    /// Use N-gram assisted drafting.
    pub use_ngram_draft: bool,
    /// N-gram size (default: 3).
    pub ngram_size: usize,
    /// Use tree attention for verification.
    pub tree_attention: bool,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Temperature for sampling (0 = greedy).
    pub temperature: f32,
}

impl Default for AssistedGenerationConfig {
    fn default() -> Self {
        Self {
            num_medusa_heads: 3,
            speculation_depth: 4,
            candidate_count: 8,
            use_ngram_draft: true,
            ngram_size: 3,
            tree_attention: true,
            hidden_dim: 768,
            vocab_size: 32000,
            temperature: 0.0,
        }
    }
}

impl AssistedGenerationConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_medusa_heads == 0 {
            return Err("num_medusa_heads must be > 0");
        }
        if self.speculation_depth == 0 {
            return Err("speculation_depth must be > 0");
        }
        if self.candidate_count == 0 {
            return Err("candidate_count must be > 0");
        }
        if self.ngram_size < 2 {
            return Err("ngram_size must be >= 2");
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0");
        }
        Ok(())
    }
}

/// A single Medusa head for predicting tokens at a specific offset.
#[derive(Debug, Clone)]
pub struct MedusaHead<B: Backend> {
    /// Prediction weights: [hidden_dim, vocab_size].
    weights: Tensor<B, 2>,
    /// Position offset (1 = next token, 2 = token after next, etc.).
    position_offset: usize,
    /// Sampling temperature.
    temperature: f32,
}

impl<B: Backend> MedusaHead<B> {
    /// Create a new Medusa head.
    pub fn new(
        hidden_dim: usize,
        vocab_size: usize,
        position_offset: usize,
        temperature: f32,
        device: &B::Device,
    ) -> Self {
        // Initialize with small random-like values
        let mut weight_data = vec![0.0f32; hidden_dim * vocab_size];
        for i in 0..weight_data.len() {
            let scale = (2.0 / (hidden_dim + vocab_size) as f32).sqrt();
            weight_data[i] = ((i % 17) as f32 - 8.0) * scale * 0.1;
        }
        let weights = Tensor::from_data(
            TensorData::new(weight_data, [hidden_dim, vocab_size]),
            device,
        );

        Self {
            weights,
            position_offset,
            temperature,
        }
    }

    /// Load from pre-trained weights.
    pub fn from_weights(
        weights: Tensor<B, 2>,
        position_offset: usize,
        temperature: f32,
    ) -> Result<Self, &'static str> {
        let [_hidden_dim, _vocab_size] = weights.dims();
        Ok(Self {
            weights,
            position_offset,
            temperature,
        })
    }

    /// Forward pass to get logits for predicted tokens.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states: [batch, seq_len, hidden_dim]
    ///
    /// # Returns
    /// Logits: [batch, seq_len, vocab_size]
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>, &'static str> {
        let [batch, seq_len, hidden_dim] = hidden.dims();
        let [weight_h, vocab_size] = self.weights.dims();

        if hidden_dim != weight_h {
            return Err("hidden dimension mismatch");
        }

        let hidden_2d = hidden.reshape([batch * seq_len, hidden_dim]);
        let logits_2d = hidden_2d.matmul(self.weights.clone());
        let logits = logits_2d.reshape([batch, seq_len, vocab_size]);

        Ok(logits)
    }

    /// Get top-k candidate tokens from last position.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states: [batch, seq_len, hidden_dim]
    /// * `k` - Number of candidates
    ///
    /// # Returns
    /// (token_ids, log_probs) for each batch item
    pub fn get_candidates(
        &self,
        hidden: Tensor<B, 3>,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>, &'static str> {
        let logits = self.forward(hidden)?;
        let [batch, seq_len, vocab_size] = logits.dims();

        let logits_data = logits
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "failed to convert logits")?;

        let mut results = Vec::with_capacity(batch);

        for b in 0..batch {
            // Get last position logits
            let start = (b * seq_len + (seq_len - 1)) * vocab_size;
            let position_logits = &logits_data[start..start + vocab_size];

            // Apply temperature
            let scaled_logits: Vec<f32> = if self.temperature > 0.0 {
                position_logits.iter().map(|&x| x / self.temperature).collect()
            } else {
                position_logits.to_vec()
            };

            // Get top-k
            let candidates = top_k_with_probs(&scaled_logits, k);
            results.push(candidates);
        }

        Ok(results)
    }

    /// Position offset for this head.
    pub fn position_offset(&self) -> usize {
        self.position_offset
    }
}

/// N-gram cache for assisted drafting.
#[derive(Debug, Clone)]
pub struct NgramCache {
    /// N-gram to next token mapping: (context_hash) -> [(token_id, count)].
    cache: HashMap<u64, Vec<(usize, usize)>>,
    /// N-gram size.
    n: usize,
    /// Maximum cache entries.
    max_entries: usize,
}

impl NgramCache {
    /// Create a new N-gram cache.
    pub fn new(n: usize, max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            n,
            max_entries,
        }
    }

    /// Hash a token sequence.
    fn hash_context(&self, tokens: &[usize]) -> u64 {
        let mut hash = 0u64;
        for (i, &token) in tokens.iter().enumerate() {
            hash = hash.wrapping_mul(31).wrapping_add(token as u64);
            hash = hash.wrapping_add((i as u64).wrapping_mul(17));
        }
        hash
    }

    /// Update cache with observed tokens.
    pub fn update(&mut self, tokens: &[usize]) {
        if tokens.len() < self.n {
            return;
        }

        for i in 0..=tokens.len() - self.n {
            let context = &tokens[i..i + self.n - 1];
            let next_token = tokens[i + self.n - 1];
            let hash = self.hash_context(context);

            let entry = self.cache.entry(hash).or_insert_with(Vec::new);

            // Update count for this token
            if let Some(pos) = entry.iter().position(|(t, _)| *t == next_token) {
                entry[pos].1 += 1;
            } else {
                entry.push((next_token, 1));
            }

            // Keep top entries by count
            entry.sort_by(|a, b| b.1.cmp(&a.1));
            entry.truncate(16);
        }

        // Evict if too large
        if self.cache.len() > self.max_entries {
            let keys_to_remove: Vec<_> = self.cache.keys().take(self.max_entries / 4).cloned().collect();
            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
    }

    /// Get predicted next tokens for a context.
    pub fn predict(&self, context: &[usize], k: usize) -> Vec<usize> {
        if context.len() < self.n - 1 {
            return Vec::new();
        }

        let recent_context = &context[context.len() - (self.n - 1)..];
        let hash = self.hash_context(recent_context);

        match self.cache.get(&hash) {
            Some(predictions) => predictions.iter().take(k).map(|(t, _)| *t).collect(),
            None => Vec::new(),
        }
    }

    /// Get cache size.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Candidate token with metadata.
#[derive(Debug, Clone, Copy)]
pub struct MedusaCandidate {
    /// Token ID.
    pub token_id: usize,
    /// Log probability.
    pub log_prob: f32,
    /// Source head index (None for N-gram).
    pub head_idx: Option<usize>,
    /// Position offset from current token.
    pub position_offset: usize,
}

/// Generated draft tree from Medusa heads.
#[derive(Debug, Clone)]
pub struct MedusaDraft {
    /// Root token (from main LM head).
    pub root_token: usize,
    /// Root log probability.
    pub root_log_prob: f32,
    /// Candidate tokens per position offset.
    /// candidates[i] contains candidates for position current+i+1.
    pub candidates: Vec<Vec<MedusaCandidate>>,
    /// Total number of candidate paths.
    pub num_paths: usize,
}

/// Verification result for Medusa draft.
#[derive(Debug, Clone)]
pub struct MedusaVerification {
    /// Accepted tokens in sequence.
    pub accepted_tokens: Vec<usize>,
    /// Number of tokens from draft that were accepted.
    pub draft_accepted: usize,
    /// Whether all draft tokens were accepted.
    pub all_accepted: bool,
}

/// Statistics for Medusa generation.
#[derive(Debug, Clone, Default)]
pub struct MedusaStats {
    /// Total draft tokens generated.
    pub total_draft_tokens: usize,
    /// Total tokens accepted.
    pub total_accepted: usize,
    /// Number of generation rounds.
    pub num_rounds: usize,
    /// Tokens from N-gram cache.
    pub ngram_contributions: usize,
    /// Average accepted per round.
    pub avg_accepted_per_round: f32,
}

impl MedusaStats {
    /// Update stats after a round.
    pub fn update(&mut self, draft_len: usize, accepted: usize, ngram_count: usize) {
        self.total_draft_tokens += draft_len;
        self.total_accepted += accepted;
        self.num_rounds += 1;
        self.ngram_contributions += ngram_count;

        if self.num_rounds > 0 {
            self.avg_accepted_per_round = self.total_accepted as f32 / self.num_rounds as f32;
        }
    }

    /// Estimated speedup.
    pub fn estimated_speedup(&self) -> f32 {
        if self.num_rounds == 0 {
            return 1.0;
        }
        // Each round produces at least 1 token
        (self.total_accepted + self.num_rounds) as f32 / self.num_rounds as f32
    }
}

/// Medusa assisted generation engine.
#[derive(Debug)]
pub struct MedusaEngine<B: Backend> {
    /// Configuration.
    config: AssistedGenerationConfig,
    /// Medusa heads.
    heads: Vec<MedusaHead<B>>,
    /// N-gram cache.
    ngram_cache: Option<NgramCache>,
    /// Statistics.
    stats: MedusaStats,
    /// Phantom marker.
    _marker: PhantomData<B>,
}

impl<B: Backend> MedusaEngine<B> {
    /// Create a new Medusa engine.
    pub fn new(config: AssistedGenerationConfig, device: &B::Device) -> Result<Self, &'static str> {
        config.validate()?;

        // Create Medusa heads (one per position offset)
        let heads: Vec<_> = (1..=config.num_medusa_heads)
            .map(|offset| {
                MedusaHead::new(
                    config.hidden_dim,
                    config.vocab_size,
                    offset,
                    config.temperature,
                    device,
                )
            })
            .collect();

        let ngram_cache = if config.use_ngram_draft {
            Some(NgramCache::new(config.ngram_size, 10000))
        } else {
            None
        };

        Ok(Self {
            config,
            heads,
            ngram_cache,
            stats: MedusaStats::default(),
            _marker: PhantomData,
        })
    }

    /// Get configuration.
    pub fn config(&self) -> &AssistedGenerationConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &MedusaStats {
        &self.stats
    }

    /// Generate draft using Medusa heads.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states from last layer: [batch, seq_len, hidden_dim]
    /// * `main_logits` - Logits from main LM head: [batch, seq_len, vocab_size]
    /// * `context_tokens` - Previous tokens for N-gram lookup
    ///
    /// # Returns
    /// Draft with candidates at each position
    pub fn generate_draft(
        &self,
        hidden: Tensor<B, 3>,
        main_logits: &[f32],
        vocab_size: usize,
        context_tokens: &[usize],
    ) -> Result<MedusaDraft, &'static str> {
        let [batch, seq_len, _hidden_dim] = hidden.dims();
        if batch != 1 {
            return Err("batch size must be 1 for draft generation");
        }

        // Get root token from main logits (last position)
        let last_logits_start = (seq_len - 1) * vocab_size;
        if main_logits.len() < last_logits_start + vocab_size {
            return Err("main_logits too short");
        }
        let last_logits = &main_logits[last_logits_start..last_logits_start + vocab_size];
        let (root_token, root_log_prob) = find_top_token(last_logits);

        // Get candidates from each Medusa head
        let mut all_candidates = Vec::with_capacity(self.config.num_medusa_heads);

        for head in &self.heads {
            let head_candidates = head.get_candidates(hidden.clone(), self.config.candidate_count)?;
            // Take first batch item
            let candidates: Vec<MedusaCandidate> = head_candidates[0]
                .iter()
                .map(|&(token_id, log_prob)| MedusaCandidate {
                    token_id,
                    log_prob,
                    head_idx: Some(head.position_offset() - 1),
                    position_offset: head.position_offset(),
                })
                .collect();
            all_candidates.push(candidates);
        }

        // Add N-gram predictions if available
        if let Some(ngram) = &self.ngram_cache {
            let ngram_preds = ngram.predict(context_tokens, self.config.candidate_count);
            if !ngram_preds.is_empty() && !all_candidates.is_empty() {
                // Add to first position candidates
                for (i, token_id) in ngram_preds.iter().enumerate() {
                    all_candidates[0].push(MedusaCandidate {
                        token_id: *token_id,
                        log_prob: -1.0 - i as f32 * 0.1, // Lower priority than Medusa
                        head_idx: None,
                        position_offset: 1,
                    });
                }
            }
        }

        // Calculate total paths
        let num_paths = all_candidates
            .iter()
            .map(|c| c.len().max(1))
            .product::<usize>();

        Ok(MedusaDraft {
            root_token,
            root_log_prob,
            candidates: all_candidates,
            num_paths,
        })
    }

    /// Verify draft against target model logits.
    ///
    /// # Arguments
    /// * `draft` - Generated draft
    /// * `target_logits` - Target model logits for draft positions
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Verification result with accepted tokens
    pub fn verify_draft(
        &mut self,
        draft: &MedusaDraft,
        target_logits: &[f32],
        vocab_size: usize,
    ) -> Result<MedusaVerification, &'static str> {
        let mut accepted_tokens = vec![draft.root_token];

        // Verify candidates at each position
        let mut position = 0;
        let mut ngram_count = 0;

        for candidates in &draft.candidates {
            if candidates.is_empty() {
                break;
            }

            // Get target token at this position
            let logit_start = position * vocab_size;
            if target_logits.len() < logit_start + vocab_size {
                break;
            }
            let position_logits = &target_logits[logit_start..logit_start + vocab_size];
            let (target_token, _) = find_top_token(position_logits);

            // Check if any candidate matches
            let matched = candidates.iter().find(|c| c.token_id == target_token);

            if let Some(candidate) = matched {
                accepted_tokens.push(candidate.token_id);
                if candidate.head_idx.is_none() {
                    ngram_count += 1;
                }
                position += 1;
            } else {
                // No match, add target token as bonus
                accepted_tokens.push(target_token);
                break;
            }
        }

        let draft_accepted = accepted_tokens.len() - 1; // Exclude root
        let all_accepted = draft_accepted == draft.candidates.len();

        // Update stats
        let total_draft = draft.candidates.iter().map(|c| c.len()).sum();
        self.stats.update(total_draft, draft_accepted, ngram_count);

        Ok(MedusaVerification {
            accepted_tokens,
            draft_accepted,
            all_accepted,
        })
    }

    /// Update N-gram cache with accepted tokens.
    pub fn update_ngram_cache(&mut self, tokens: &[usize]) {
        if let Some(ngram) = &mut self.ngram_cache {
            ngram.update(tokens);
        }
    }

    /// Reset engine state.
    pub fn reset(&mut self) {
        if let Some(ngram) = &mut self.ngram_cache {
            ngram.clear();
        }
        self.stats = MedusaStats::default();
    }
}

/// Build a tree of candidate paths from Medusa draft.
///
/// # Arguments
/// * `draft` - Medusa draft with candidates
/// * `max_paths` - Maximum number of paths to generate
///
/// # Returns
/// List of candidate paths (each path is a sequence of token IDs)
pub fn build_candidate_tree(draft: &MedusaDraft, max_paths: usize) -> Vec<Vec<usize>> {
    let mut paths = vec![vec![draft.root_token]];

    for candidates in &draft.candidates {
        if candidates.is_empty() {
            break;
        }

        let mut new_paths = Vec::new();
        for path in &paths {
            for candidate in candidates {
                if new_paths.len() >= max_paths {
                    break;
                }
                let mut new_path = path.clone();
                new_path.push(candidate.token_id);
                new_paths.push(new_path);
            }
            if new_paths.len() >= max_paths {
                break;
            }
        }
        paths = new_paths;

        if paths.len() >= max_paths {
            break;
        }
    }

    paths.truncate(max_paths);
    paths
}

/// Flatten candidate tree to linear sequence for batch verification.
///
/// # Arguments
/// * `paths` - Candidate paths
///
/// # Returns
/// (flattened_tokens, path_lengths, path_starts)
pub fn flatten_candidate_tree(paths: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut tokens = Vec::new();
    let mut lengths = Vec::with_capacity(paths.len());
    let mut starts = Vec::with_capacity(paths.len());

    for path in paths {
        starts.push(tokens.len());
        lengths.push(path.len());
        tokens.extend(path.iter().cloned());
    }

    (tokens, lengths, starts)
}

// Helper functions

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

fn top_k_with_probs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let log_probs = log_softmax(logits);
    let mut scored: Vec<(usize, f32)> = log_probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
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
    fn test_assisted_generation_config() {
        let config = AssistedGenerationConfig::default();
        assert!(config.validate().is_ok());

        let invalid = AssistedGenerationConfig {
            num_medusa_heads: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_medusa_head() {
        let device = <TestBackend as Backend>::Device::default();
        let head = MedusaHead::<TestBackend>::new(64, 100, 1, 0.0, &device);

        let hidden = Tensor::zeros([1, 4, 64], &device);
        let logits = head.forward(hidden);
        assert!(logits.is_ok());
        assert_eq!(logits.unwrap().dims(), [1, 4, 100]);
    }

    #[test]
    fn test_ngram_cache() {
        let mut cache = NgramCache::new(3, 1000);

        // Add some sequences
        cache.update(&[1, 2, 3, 4, 5]);
        cache.update(&[1, 2, 3, 6, 7]);
        cache.update(&[1, 2, 3, 4, 8]);

        // Predict next token after [2, 3]
        let predictions = cache.predict(&[1, 2, 3], 3);
        assert!(!predictions.is_empty());
        // Token 4 should be most common
        assert_eq!(predictions[0], 4);
    }

    #[test]
    fn test_medusa_engine_creation() {
        let device = <TestBackend as Backend>::Device::default();
        let config = AssistedGenerationConfig {
            hidden_dim: 64,
            vocab_size: 100,
            num_medusa_heads: 2,
            ..Default::default()
        };

        let engine = MedusaEngine::<TestBackend>::new(config, &device);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_build_candidate_tree() {
        let draft = MedusaDraft {
            root_token: 0,
            root_log_prob: 0.0,
            candidates: vec![
                vec![
                    MedusaCandidate {
                        token_id: 1,
                        log_prob: -0.1,
                        head_idx: Some(0),
                        position_offset: 1,
                    },
                    MedusaCandidate {
                        token_id: 2,
                        log_prob: -0.2,
                        head_idx: Some(0),
                        position_offset: 1,
                    },
                ],
                vec![
                    MedusaCandidate {
                        token_id: 3,
                        log_prob: -0.1,
                        head_idx: Some(1),
                        position_offset: 2,
                    },
                ],
            ],
            num_paths: 2,
        };

        let paths = build_candidate_tree(&draft, 4);
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], vec![0, 1, 3]);
        assert_eq!(paths[1], vec![0, 2, 3]);
    }

    #[test]
    fn test_flatten_candidate_tree() {
        let paths = vec![vec![0, 1, 2], vec![0, 3, 4, 5]];

        let (tokens, lengths, starts) = flatten_candidate_tree(&paths);

        assert_eq!(tokens, vec![0, 1, 2, 0, 3, 4, 5]);
        assert_eq!(lengths, vec![3, 4]);
        assert_eq!(starts, vec![0, 3]);
    }

    #[test]
    fn test_stats_tracking() {
        let mut stats = MedusaStats::default();

        stats.update(8, 3, 1);
        stats.update(8, 4, 0);

        assert_eq!(stats.total_draft_tokens, 16);
        assert_eq!(stats.total_accepted, 7);
        assert_eq!(stats.num_rounds, 2);
        assert_eq!(stats.ngram_contributions, 1);
        assert!(stats.estimated_speedup() > 1.0);
    }
}
