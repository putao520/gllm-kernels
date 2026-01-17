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
#[inline(always)]
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
            layers_saved: num_layers.saturating_sub(early_exit_layer + 1),
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
