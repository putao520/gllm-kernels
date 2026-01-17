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
    #[inline(always)]
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
    #[inline(always)]
    pub fn estimated_speedup(&self) -> f32 {
        if self.num_rounds == 0 {
            return 1.0;
        }
        // Each round produces at least 1 token
        (self.total_accepted + self.num_rounds) as f32 / self.num_rounds as f32
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
#[inline(always)]
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
#[inline(always)]
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

/// Find the top token from logits.
#[inline(always)]
pub fn find_top_token(logits: &[f32]) -> (usize, f32) {
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

/// Compute log-softmax over logits.
#[inline(always)]
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
