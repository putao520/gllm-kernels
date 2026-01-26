//! Mixture-of-Experts (MoE) Routing implementation.
//!
//! MoE routing selects which experts should process each token based on
//! routing logits computed from hidden states.
//!
//! This is a pure Rust implementation without framework dependencies.
//!
//! Reference: Switch Transformers, Mixtral, DeepSeek-MoE

use crate::kernel_types::KernelFloat;

/// Configuration for MoE routing operations.
#[derive(Debug, Clone)]
pub struct MoERoutingConfig {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of experts to select per token (top-k).
    pub num_experts_per_tok: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
}

impl Default for MoERoutingConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_tok: 2,
            hidden_size: 4096,
        }
    }
}

/// Result of MoE routing operation.
#[derive(Debug, Clone)]
pub struct MoERoutingResult {
    /// Selected expert indices for each token: [batch * seq, top_k]
    pub expert_indices: Vec<u32>,
    /// Normalized weights for selected experts: [batch * seq, top_k]
    pub expert_weights: Vec<f32>,
    /// Number of tokens.
    pub num_tokens: usize,
    /// Top-k value (experts per token).
    pub top_k: usize,
}

/// Compute MoE routing: select top-k experts for each token.
///
/// # Arguments
/// * `hidden_states` - Input hidden states: [batch * seq, hidden_size]
/// * `gate_weights` - Router gate weights: [hidden_size, num_experts]
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `config` - Routing configuration
///
/// # Returns
/// MoERoutingResult with expert indices and weights.
///
/// # Algorithm
/// 1. Compute routing logits: hidden_states @ gate_weights
/// 2. Select top-k experts per token
/// 3. Apply softmax to normalize weights
pub fn moe_route<T: KernelFloat>(
    hidden_states: &[T],
    gate_weights: &[f32],
    batch_size: usize,
    seq_len: usize,
    config: &MoERoutingConfig,
) -> MoERoutingResult {
    let num_tokens = batch_size * seq_len;
    let hidden_size = config.hidden_size;
    let num_experts = config.num_experts;
    let top_k = config.num_experts_per_tok;

    // Validate dimensions
    assert_eq!(
        hidden_states.len(),
        num_tokens * hidden_size,
        "hidden_states size mismatch: expected {}, got {}",
        num_tokens * hidden_size,
        hidden_states.len()
    );
    assert_eq!(
        gate_weights.len(),
        hidden_size * num_experts,
        "gate_weights size mismatch: expected {}, got {}",
        hidden_size * num_experts,
        gate_weights.len()
    );
    assert!(
        top_k <= num_experts,
        "top_k ({}) cannot exceed num_experts ({})",
        top_k,
        num_experts
    );

    let mut expert_indices = Vec::with_capacity(num_tokens * top_k);
    let mut expert_weights = Vec::with_capacity(num_tokens * top_k);

    // Process each token
    for token_idx in 0..num_tokens {
        let hidden_offset = token_idx * hidden_size;
        let hidden_slice = &hidden_states[hidden_offset..hidden_offset + hidden_size];

        // Step 1: Compute routing logits for this token
        // logits[e] = sum_h(hidden[h] * gate[h, e])
        let mut logits = vec![0.0f32; num_experts];
        for (h, &hval) in hidden_slice.iter().enumerate() {
            let hval_f32 = hval.to_f32();
            for e in 0..num_experts {
                // gate_weights is [hidden_size, num_experts] row-major
                logits[e] += hval_f32 * gate_weights[h * num_experts + e];
            }
        }

        // Step 2: Find top-k experts
        let top_k_result = find_topk(&logits, top_k);

        // Step 3: Apply softmax to top-k logits
        let softmax_weights = softmax_slice(&top_k_result.values);

        // Store results
        for &idx in &top_k_result.indices {
            expert_indices.push(idx as u32);
        }
        for &w in &softmax_weights {
            expert_weights.push(w);
        }
    }

    MoERoutingResult {
        expert_indices,
        expert_weights,
        num_tokens,
        top_k,
    }
}

/// Compute routing logits without selecting experts.
///
/// Useful when you want to analyze routing patterns or implement custom selection.
///
/// # Arguments
/// * `hidden_states` - Input hidden states: [batch * seq, hidden_size]
/// * `gate_weights` - Router gate weights: [hidden_size, num_experts]
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `config` - Routing configuration
///
/// # Returns
/// Routing logits: [batch * seq, num_experts]
pub fn compute_routing_logits<T: KernelFloat>(
    hidden_states: &[T],
    gate_weights: &[f32],
    batch_size: usize,
    seq_len: usize,
    config: &MoERoutingConfig,
) -> Vec<f32> {
    let num_tokens = batch_size * seq_len;
    let hidden_size = config.hidden_size;
    let num_experts = config.num_experts;

    let mut logits = vec![0.0f32; num_tokens * num_experts];

    for token_idx in 0..num_tokens {
        let hidden_offset = token_idx * hidden_size;
        let hidden_slice = &hidden_states[hidden_offset..hidden_offset + hidden_size];
        let logit_offset = token_idx * num_experts;

        for (h, &hval) in hidden_slice.iter().enumerate() {
            let hval_f32 = hval.to_f32();
            for e in 0..num_experts {
                logits[logit_offset + e] += hval_f32 * gate_weights[h * num_experts + e];
            }
        }
    }

    logits
}

/// Get expert load statistics (how many tokens are routed to each expert).
///
/// Useful for load balancing analysis and auxiliary loss computation.
pub fn compute_expert_load(routing_result: &MoERoutingResult, num_experts: usize) -> Vec<usize> {
    let mut load = vec![0usize; num_experts];
    for &expert_idx in &routing_result.expert_indices {
        if (expert_idx as usize) < num_experts {
            load[expert_idx as usize] += 1;
        }
    }
    load
}

/// Compute load balancing loss (auxiliary loss for training).
///
/// L_aux = num_experts * sum_e(f_e * P_e)
/// where f_e = fraction of tokens routed to expert e
///       P_e = mean routing probability for expert e
pub fn compute_load_balance_loss(
    routing_logits: &[f32],
    num_tokens: usize,
    num_experts: usize,
) -> f32 {
    if num_tokens == 0 || num_experts == 0 {
        return 0.0;
    }

    // Compute softmax probabilities for all experts
    let mut probs = vec![0.0f32; num_tokens * num_experts];
    for t in 0..num_tokens {
        let offset = t * num_experts;
        let logit_slice = &routing_logits[offset..offset + num_experts];
        let prob_slice = &mut probs[offset..offset + num_experts];

        // Softmax
        let max_logit = logit_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (i, &l) in logit_slice.iter().enumerate() {
            prob_slice[i] = (l - max_logit).exp();
            sum += prob_slice[i];
        }
        if sum > 0.0 {
            for p in prob_slice.iter_mut() {
                *p /= sum;
            }
        }
    }

    // Compute mean probability per expert (P_e)
    let mut mean_probs = vec![0.0f32; num_experts];
    for t in 0..num_tokens {
        for e in 0..num_experts {
            mean_probs[e] += probs[t * num_experts + e];
        }
    }
    for p in mean_probs.iter_mut() {
        *p /= num_tokens as f32;
    }

    // Compute fraction of tokens per expert (f_e) - using argmax routing
    let mut fractions = vec![0.0f32; num_experts];
    for t in 0..num_tokens {
        let offset = t * num_experts;
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for e in 0..num_experts {
            if probs[offset + e] > max_val {
                max_val = probs[offset + e];
                max_idx = e;
            }
        }
        fractions[max_idx] += 1.0;
    }
    for f in fractions.iter_mut() {
        *f /= num_tokens as f32;
    }

    // L_aux = num_experts * sum_e(f_e * P_e)
    let mut loss = 0.0f32;
    for e in 0..num_experts {
        loss += fractions[e] * mean_probs[e];
    }
    loss * (num_experts as f32)
}

// ============================================================================
// Internal helpers
// ============================================================================

struct TopKInternalResult {
    indices: Vec<usize>,
    values: Vec<f32>,
}

/// Find top-k largest values and their indices.
fn find_topk(values: &[f32], k: usize) -> TopKInternalResult {
    let k = k.min(values.len());

    // Create (value, index) pairs
    let mut pairs: Vec<(f32, usize)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Partial sort to get top-k
    pairs.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_k_pairs = &mut pairs[..k];
    // Sort top-k by value (descending) for consistent output
    top_k_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    TopKInternalResult {
        indices: top_k_pairs.iter().map(|&(_, i)| i).collect(),
        values: top_k_pairs.iter().map(|&(v, _)| v).collect(),
    }
}

/// Apply softmax to a slice of values.
fn softmax_slice(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }

    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_values.iter().sum();

    if sum > 0.0 {
        for v in exp_values.iter_mut() {
            *v /= sum;
        }
    }

    exp_values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_route_basic() {
        let config = MoERoutingConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            hidden_size: 8,
        };

        let batch_size = 2;
        let seq_len = 3;
        let num_tokens = batch_size * seq_len;

        // Random hidden states
        let hidden_states: Vec<f32> = (0..num_tokens * config.hidden_size)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        // Random gate weights
        let gate_weights: Vec<f32> = (0..config.hidden_size * config.num_experts)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();

        let result = moe_route(&hidden_states, &gate_weights, batch_size, seq_len, &config);

        // Check output dimensions
        assert_eq!(result.num_tokens, num_tokens);
        assert_eq!(result.top_k, config.num_experts_per_tok);
        assert_eq!(result.expert_indices.len(), num_tokens * config.num_experts_per_tok);
        assert_eq!(result.expert_weights.len(), num_tokens * config.num_experts_per_tok);

        // Check expert indices are valid
        for &idx in &result.expert_indices {
            assert!((idx as usize) < config.num_experts);
        }

        // Check weights are normalized (sum ~= 1 per token)
        for t in 0..num_tokens {
            let start = t * config.num_experts_per_tok;
            let end = start + config.num_experts_per_tok;
            let weight_sum: f32 = result.expert_weights[start..end].iter().sum();
            assert!(
                (weight_sum - 1.0).abs() < 1e-5,
                "Weight sum for token {}: {} (expected 1.0)",
                t,
                weight_sum
            );
        }
    }

    #[test]
    fn test_moe_route_single_expert() {
        let config = MoERoutingConfig {
            num_experts: 8,
            num_experts_per_tok: 1, // Select only 1 expert
            hidden_size: 16,
        };

        let batch_size = 1;
        let seq_len = 4;

        let hidden_states: Vec<f32> = vec![0.5; batch_size * seq_len * config.hidden_size];
        let gate_weights: Vec<f32> = vec![0.1; config.hidden_size * config.num_experts];

        let result = moe_route(&hidden_states, &gate_weights, batch_size, seq_len, &config);

        // With top_k=1, each token should have exactly 1 expert with weight 1.0
        assert_eq!(result.expert_indices.len(), batch_size * seq_len);
        assert_eq!(result.expert_weights.len(), batch_size * seq_len);

        for &w in &result.expert_weights {
            assert!((w - 1.0).abs() < 1e-5, "Single expert weight should be 1.0");
        }
    }

    #[test]
    fn test_topk_internal() {
        let values = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let result = find_topk(&values, 3);

        assert_eq!(result.indices.len(), 3);
        assert_eq!(result.values.len(), 3);

        // Top 3 should be indices 3 (0.8), 1 (0.5), 4 (0.3)
        assert_eq!(result.indices[0], 3);
        assert_eq!(result.indices[1], 1);
        assert_eq!(result.indices[2], 4);

        assert!((result.values[0] - 0.8).abs() < 1e-6);
        assert!((result.values[1] - 0.5).abs() < 1e-6);
        assert!((result.values[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_slice() {
        let values = vec![1.0, 2.0, 3.0];
        let probs = softmax_slice(&values);

        // Check sum is 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check ordering (higher value -> higher prob)
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_expert_load() {
        let result = MoERoutingResult {
            expert_indices: vec![0, 1, 0, 2, 1, 1], // 6 assignments
            expert_weights: vec![0.6, 0.4, 0.7, 0.3, 0.5, 0.5],
            num_tokens: 3,
            top_k: 2,
        };

        let load = compute_expert_load(&result, 4);

        assert_eq!(load[0], 2); // Expert 0 selected twice
        assert_eq!(load[1], 3); // Expert 1 selected three times
        assert_eq!(load[2], 1); // Expert 2 selected once
        assert_eq!(load[3], 0); // Expert 3 not selected
    }

    #[test]
    fn test_load_balance_loss() {
        // Uniform routing should have lower loss than skewed routing
        let num_tokens = 4;
        let num_experts = 4;

        // Uniform logits (all experts equally likely)
        let uniform_logits: Vec<f32> = vec![1.0; num_tokens * num_experts];
        let uniform_loss = compute_load_balance_loss(&uniform_logits, num_tokens, num_experts);

        // Skewed logits (one expert much higher)
        let mut skewed_logits = vec![0.0f32; num_tokens * num_experts];
        for t in 0..num_tokens {
            skewed_logits[t * num_experts] = 10.0; // Expert 0 dominates
        }
        let skewed_loss = compute_load_balance_loss(&skewed_logits, num_tokens, num_experts);

        // Skewed routing should have higher loss (worse balance)
        assert!(
            skewed_loss > uniform_loss,
            "Skewed loss {} should be > uniform loss {}",
            skewed_loss,
            uniform_loss
        );
    }

    #[test]
    fn test_compute_routing_logits() {
        let config = MoERoutingConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            hidden_size: 8,
        };

        let batch_size = 1;
        let seq_len = 2;

        // Simple hidden states
        let hidden_states: Vec<f32> = vec![1.0; batch_size * seq_len * config.hidden_size];

        // Identity-ish gate weights (expert e gets contribution from all hidden dims)
        let mut gate_weights = vec![0.0f32; config.hidden_size * config.num_experts];
        for h in 0..config.hidden_size {
            for e in 0..config.num_experts {
                gate_weights[h * config.num_experts + e] = (e + 1) as f32 * 0.1;
            }
        }

        let logits = compute_routing_logits(&hidden_states, &gate_weights, batch_size, seq_len, &config);

        assert_eq!(logits.len(), batch_size * seq_len * config.num_experts);

        // Expert with higher weights should have higher logits
        // Expert 0: 0.1 * 8 = 0.8
        // Expert 1: 0.2 * 8 = 1.6
        // Expert 2: 0.3 * 8 = 2.4
        // Expert 3: 0.4 * 8 = 3.2
        for t in 0..batch_size * seq_len {
            let offset = t * config.num_experts;
            assert!(logits[offset + 3] > logits[offset + 2]);
            assert!(logits[offset + 2] > logits[offset + 1]);
            assert!(logits[offset + 1] > logits[offset]);
        }
    }
}
