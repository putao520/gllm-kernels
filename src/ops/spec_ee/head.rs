use crate::kernel_dispatcher::KernelFloat;
use crate::ops::math;

/// Early exit head for a specific layer.
///
/// Contains both LM head (for token prediction) and confidence head.
#[derive(Debug, Clone)]
pub struct EarlyExitHead<T: KernelFloat> {
    /// LM head weights: [hidden_dim, vocab_size].
    lm_head: Vec<T>,
    lm_shape: [usize; 2],
    /// Confidence head weights: [hidden_dim, 1].
    confidence_head: Vec<T>,
    confidence_shape: [usize; 2],
    /// Confidence bias.
    confidence_bias: f32,
    /// Layer index this head is attached to.
    layer_idx: usize,
}

impl<T: KernelFloat> EarlyExitHead<T> {
    /// Create a new early exit head.
    #[inline(always)]
    pub fn new(hidden_dim: usize, vocab_size: usize, layer_idx: usize) -> Result<Self, &'static str> {
        if hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }

        let mut lm_head = Vec::with_capacity(hidden_dim * vocab_size);
        for i in 0..(hidden_dim * vocab_size) {
            let scale = (2.0 / (hidden_dim + vocab_size) as f32).sqrt();
            let value = ((i % 13) as f32 - 6.0) * scale * 0.1;
            lm_head.push(T::from_f32(value));
        }

        let mut confidence_head = Vec::with_capacity(hidden_dim);
        for i in 0..hidden_dim {
            let scale = (2.0 / hidden_dim as f32).sqrt();
            let value = ((i % 7) as f32 - 3.0) * scale * 0.1;
            confidence_head.push(T::from_f32(value));
        }

        Ok(Self {
            lm_head,
            lm_shape: [hidden_dim, vocab_size],
            confidence_head,
            confidence_shape: [hidden_dim, 1],
            confidence_bias: 0.0,
            layer_idx,
        })
    }

    /// Load from pre-trained weights.
    #[inline(always)]
    pub fn from_weights(
        lm_head: Vec<T>,
        lm_shape: [usize; 2],
        confidence_head: Vec<T>,
        confidence_shape: [usize; 2],
        confidence_bias: f32,
        layer_idx: usize,
    ) -> Result<Self, &'static str> {
        if lm_shape[0] == 0 || lm_shape[1] == 0 {
            return Err("lm_head shape must be non-empty");
        }
        if confidence_shape[0] == 0 || confidence_shape[1] != 1 {
            return Err("confidence_head shape must be [hidden_dim, 1]");
        }
        if lm_shape[0] != confidence_shape[0] {
            return Err("lm_head and confidence_head hidden dim mismatch");
        }
        if lm_head.len() != lm_shape[0] * lm_shape[1] {
            return Err("lm_head length mismatch");
        }
        if confidence_head.len() != confidence_shape[0] * confidence_shape[1] {
            return Err("confidence_head length mismatch");
        }

        Ok(Self {
            lm_head,
            lm_shape,
            confidence_head,
            confidence_shape,
            confidence_bias,
            layer_idx,
        })
    }

    /// Forward pass: compute logits and confidence.
    #[inline(always)]
    pub fn forward(
        &self,
        hidden: &[T],
        batch: usize,
        seq_len: usize,
        logits_out: &mut [T],
        confidence_out: &mut [T],
    ) -> Result<(), &'static str> {
        let hidden_dim = self.lm_shape[0];
        let vocab_size = self.lm_shape[1];
        if self.confidence_shape[0] != hidden_dim {
            return Err("confidence_head hidden_dim mismatch");
        }
        if batch == 0 || seq_len == 0 {
            return Err("batch and seq_len must be > 0");
        }

        let positions = batch * seq_len;
        let expected_hidden = positions * hidden_dim;
        if hidden.len() != expected_hidden {
            return Err("hidden length mismatch");
        }
        if logits_out.len() != positions * vocab_size {
            return Err("logits_out length mismatch");
        }
        if confidence_out.len() != positions {
            return Err("confidence_out length mismatch");
        }

        for pos in 0..positions {
            let hidden_base = pos * hidden_dim;
            let logits_base = pos * vocab_size;
            for v in 0..vocab_size {
                let mut acc = 0.0f32;
                for d in 0..hidden_dim {
                    let weight_idx = d * vocab_size + v;
                    acc += hidden[hidden_base + d].to_f32() * self.lm_head[weight_idx].to_f32();
                }
                logits_out[logits_base + v] = T::from_f32(acc);
            }
        }

        for pos in 0..positions {
            let hidden_base = pos * hidden_dim;
            let mut logit = self.confidence_bias;
            for d in 0..hidden_dim {
                logit += hidden[hidden_base + d].to_f32() * self.confidence_head[d].to_f32();
            }
            confidence_out[pos] = T::from_f32(math::sigmoid(logit));
        }

        Ok(())
    }

    /// Forward pass returning only confidence (faster for exit decision).
    #[inline(always)]
    pub fn forward_confidence_only(
        &self,
        hidden: &[T],
        batch: usize,
        seq_len: usize,
        confidence_out: &mut [T],
    ) -> Result<(), &'static str> {
        let hidden_dim = self.confidence_shape[0];
        if batch == 0 || seq_len == 0 {
            return Err("batch and seq_len must be > 0");
        }
        if hidden.len() != batch * seq_len * hidden_dim {
            return Err("hidden length mismatch");
        }
        if confidence_out.len() != batch * seq_len {
            return Err("confidence_out length mismatch");
        }

        let positions = batch * seq_len;
        for pos in 0..positions {
            let hidden_base = pos * hidden_dim;
            let mut logit = self.confidence_bias;
            for d in 0..hidden_dim {
                logit += hidden[hidden_base + d].to_f32() * self.confidence_head[d].to_f32();
            }
            confidence_out[pos] = T::from_f32(math::sigmoid(logit));
        }

        Ok(())
    }

    /// Get layer index.
    #[inline(always)]
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
}
