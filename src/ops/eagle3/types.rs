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
    #[inline(always)]
    pub fn update(&mut self, draft_len: usize, accepted: usize, early_term: bool) {
        self.total_draft_tokens += draft_len;
        self.total_accepted_tokens += accepted;
        self.num_cycles += 1;
        if early_term {
            self.early_terminations += 1;
        }

        self.avg_draft_length = self.total_draft_tokens as f32 / self.num_cycles as f32;
        self.avg_acceptance_rate = if self.total_draft_tokens > 0 {
            self.total_accepted_tokens as f32 / self.total_draft_tokens as f32
        } else {
            0.0
        };
    }

    /// Calculate speedup estimate (draft accepted / cycles).
    #[inline(always)]
    pub fn estimated_speedup(&self) -> f32 {
        if self.num_cycles == 0 {
            return 1.0;
        }
        (self.total_accepted_tokens + self.num_cycles) as f32 / self.num_cycles as f32
    }
}
