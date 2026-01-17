use std::marker::PhantomData;

use crate::kernel_dispatcher::KernelFloat;

use super::cache::NgramCache;
use super::config::AssistedGenerationConfig;
use super::head::MedusaHead;
use super::types::{find_top_token, MedusaCandidate, MedusaDraft, MedusaStats, MedusaVerification};

/// Medusa assisted generation engine.
#[derive(Debug)]
pub struct MedusaEngine<T: KernelFloat> {
    /// Configuration.
    config: AssistedGenerationConfig,
    /// Medusa heads.
    heads: Vec<MedusaHead<T>>,
    /// N-gram cache.
    ngram_cache: Option<NgramCache>,
    /// Statistics.
    stats: MedusaStats,
    /// Phantom marker.
    _marker: PhantomData<T>,
}

impl<T: KernelFloat> MedusaEngine<T> {
    /// Create a new Medusa engine.
    #[inline(always)]
    pub fn new(config: AssistedGenerationConfig) -> Result<Self, &'static str> {
        config.validate()?;

        // Create Medusa heads (one per position offset)
        let mut heads = Vec::with_capacity(config.num_medusa_heads);
        for offset in 1..=config.num_medusa_heads {
            heads.push(MedusaHead::new(
                config.hidden_dim,
                config.vocab_size,
                offset,
                config.temperature,
            )?);
        }

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

    /// Create engine from pre-trained heads.
    #[inline(always)]
    pub fn from_heads(
        config: AssistedGenerationConfig,
        heads: Vec<MedusaHead<T>>,
    ) -> Result<Self, &'static str> {
        config.validate()?;

        if heads.len() != config.num_medusa_heads {
            return Err("number of heads mismatch");
        }

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
    #[inline(always)]
    pub fn config(&self) -> &AssistedGenerationConfig {
        &self.config
    }

    /// Get statistics.
    #[inline(always)]
    pub fn stats(&self) -> &MedusaStats {
        &self.stats
    }

    /// Generate draft using Medusa heads.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states from last layer: [batch * seq_len * hidden_dim]
    /// * `batch` - Batch size
    /// * `seq_len` - Sequence length
    /// * `main_logits` - Logits from main LM head: [batch * seq_len * vocab_size]
    /// * `context_tokens` - Previous tokens for N-gram lookup
    ///
    /// # Returns
    /// Draft with candidates at each position
    #[inline(always)]
    pub fn generate_draft(
        &self,
        hidden: &[T],
        batch: usize,
        seq_len: usize,
        main_logits: &[f32],
        context_tokens: &[usize],
    ) -> Result<MedusaDraft, &'static str> {
        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;

        if batch != 1 {
            return Err("batch size must be 1 for draft generation");
        }
        if hidden.len() != batch * seq_len * hidden_dim {
            return Err("hidden length mismatch");
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
            let head_candidates = head.get_candidates(hidden, batch, seq_len, self.config.candidate_count)?;
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
    ///
    /// # Returns
    /// Verification result with accepted tokens
    #[inline(always)]
    pub fn verify_draft(
        &mut self,
        draft: &MedusaDraft,
        target_logits: &[f32],
    ) -> Result<MedusaVerification, &'static str> {
        let vocab_size = self.config.vocab_size;
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
    #[inline(always)]
    pub fn update_ngram_cache(&mut self, tokens: &[usize]) {
        if let Some(ngram) = &mut self.ngram_cache {
            ngram.update(tokens);
        }
    }

    /// Reset engine state.
    #[inline(always)]
    pub fn reset(&mut self) {
        if let Some(ngram) = &mut self.ngram_cache {
            ngram.clear();
        }
        self.stats = MedusaStats::default();
    }
}
