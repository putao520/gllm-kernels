use crate::kernel_types::KernelFloat;

/// A single Medusa head for predicting tokens at a specific offset.
#[derive(Debug, Clone)]
pub struct MedusaHead<T: KernelFloat> {
    /// Prediction weights: [hidden_dim, vocab_size].
    weights: Vec<T>,
    /// Weight shape: [hidden_dim, vocab_size].
    weight_shape: [usize; 2],
    /// Position offset (1 = next token, 2 = token after next, etc.).
    position_offset: usize,
    /// Sampling temperature.
    temperature: f32,
}

impl<T: KernelFloat> MedusaHead<T> {
    /// Create a new Medusa head.
    #[inline(always)]
    pub fn new(
        hidden_dim: usize,
        vocab_size: usize,
        position_offset: usize,
        temperature: f32,
    ) -> Result<Self, &'static str> {
        if hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }
        if position_offset == 0 {
            return Err("position_offset must be > 0");
        }

        // Initialize with small random-like values
        let mut weights = Vec::with_capacity(hidden_dim * vocab_size);
        for i in 0..(hidden_dim * vocab_size) {
            let scale = (2.0 / (hidden_dim + vocab_size) as f32).sqrt();
            let value = ((i % 17) as f32 - 8.0) * scale * 0.1;
            weights.push(T::from_f32(value));
        }

        Ok(Self {
            weights,
            weight_shape: [hidden_dim, vocab_size],
            position_offset,
            temperature,
        })
    }

    /// Load from pre-trained weights.
    #[inline(always)]
    pub fn from_weights(
        weights: Vec<T>,
        weight_shape: [usize; 2],
        position_offset: usize,
        temperature: f32,
    ) -> Result<Self, &'static str> {
        let [hidden_dim, vocab_size] = weight_shape;
        if hidden_dim == 0 || vocab_size == 0 {
            return Err("weight_shape must be non-empty");
        }
        if weights.len() != hidden_dim * vocab_size {
            return Err("weights length mismatch");
        }
        if position_offset == 0 {
            return Err("position_offset must be > 0");
        }

        Ok(Self {
            weights,
            weight_shape,
            position_offset,
            temperature,
        })
    }

    /// Forward pass to get logits for predicted tokens.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states: [batch * seq_len * hidden_dim]
    /// * `batch` - Batch size
    /// * `seq_len` - Sequence length
    /// * `logits_out` - Output logits: [batch * seq_len * vocab_size]
    #[inline(always)]
    pub fn forward(
        &self,
        hidden: &[T],
        batch: usize,
        seq_len: usize,
        logits_out: &mut [T],
    ) -> Result<(), &'static str> {
        let [hidden_dim, vocab_size] = self.weight_shape;
        let positions = batch * seq_len;

        if hidden.len() != positions * hidden_dim {
            return Err("hidden length mismatch");
        }
        if logits_out.len() != positions * vocab_size {
            return Err("logits_out length mismatch");
        }

        // Matrix multiplication: hidden @ weights -> logits
        // hidden: [positions, hidden_dim]
        // weights: [hidden_dim, vocab_size]
        // logits: [positions, vocab_size]
        for pos in 0..positions {
            let hidden_base = pos * hidden_dim;
            let logits_base = pos * vocab_size;
            for v in 0..vocab_size {
                let mut acc = 0.0f32;
                for d in 0..hidden_dim {
                    let weight_idx = d * vocab_size + v;
                    acc += hidden[hidden_base + d].to_f32() * self.weights[weight_idx].to_f32();
                }
                logits_out[logits_base + v] = T::from_f32(acc);
            }
        }

        Ok(())
    }

    /// Get top-k candidate tokens from last position.
    ///
    /// # Arguments
    /// * `hidden` - Hidden states: [batch * seq_len * hidden_dim]
    /// * `batch` - Batch size
    /// * `seq_len` - Sequence length
    /// * `k` - Number of candidates
    ///
    /// # Returns
    /// (token_ids, log_probs) for each batch item
    #[inline(always)]
    pub fn get_candidates(
        &self,
        hidden: &[T],
        batch: usize,
        seq_len: usize,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>, &'static str> {
        let [hidden_dim, vocab_size] = self.weight_shape;

        if hidden.len() != batch * seq_len * hidden_dim {
            return Err("hidden length mismatch");
        }

        // Compute logits for last position of each batch item
        let mut results = Vec::with_capacity(batch);

        for b in 0..batch {
            // Get last position hidden state
            let last_pos = (b * seq_len + (seq_len - 1)) * hidden_dim;
            let last_hidden = &hidden[last_pos..last_pos + hidden_dim];

            // Compute logits for this position
            let mut logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let mut acc = 0.0f32;
                for d in 0..hidden_dim {
                    let weight_idx = d * vocab_size + v;
                    acc += last_hidden[d].to_f32() * self.weights[weight_idx].to_f32();
                }
                logits[v] = acc;
            }

            // Apply temperature
            if self.temperature > 0.0 {
                for logit in &mut logits {
                    *logit /= self.temperature;
                }
            }

            // Get top-k
            let candidates = top_k_with_probs(&logits, k);
            results.push(candidates);
        }

        Ok(results)
    }

    /// Position offset for this head.
    #[inline(always)]
    pub fn position_offset(&self) -> usize {
        self.position_offset
    }

    /// Get hidden dimension.
    #[inline(always)]
    pub fn hidden_dim(&self) -> usize {
        self.weight_shape[0]
    }

    /// Get vocabulary size.
    #[inline(always)]
    pub fn vocab_size(&self) -> usize {
        self.weight_shape[1]
    }
}

/// Get top-k tokens with their log probabilities.
#[inline(always)]
fn top_k_with_probs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let log_probs = log_softmax(logits);
    let mut scored: Vec<(usize, f32)> = log_probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
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
