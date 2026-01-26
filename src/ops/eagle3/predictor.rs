use crate::kernel_types::KernelFloat;
use crate::ops::math;

/// Confidence predictor for acceptance probability estimation.
///
/// Uses multi-layer feature fusion to predict token-level acceptance probability.
#[derive(Debug, Clone)]
pub struct ConfidencePredictor<T: KernelFloat> {
    /// Linear layer weights: [fused_dim, 1].
    weight: Vec<T>,
    /// Weight shape: [fused_dim, 1].
    weight_shape: [usize; 2],
    /// Bias term.
    bias: f32,
    /// Number of layers to fuse.
    fusion_layers: usize,
    /// Per-layer hidden dimension.
    hidden_dim: usize,
}

impl<T: KernelFloat> ConfidencePredictor<T> {
    /// Create a new confidence predictor with given dimensions.
    #[inline(always)]
    pub fn new(hidden_dim: usize, fusion_layers: usize) -> Result<Self, &'static str> {
        if hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if fusion_layers == 0 {
            return Err("fusion_layers must be > 0");
        }

        let fused_dim = hidden_dim * fusion_layers;
        let mut weight = Vec::with_capacity(fused_dim);

        for i in 0..fused_dim {
            let scale = (2.0 / fused_dim as f32).sqrt();
            let value = ((i % 7) as f32 - 3.0) * scale * 0.1;
            weight.push(T::from_f32(value));
        }

        Ok(Self {
            weight,
            weight_shape: [fused_dim, 1],
            bias: 0.0,
            fusion_layers,
            hidden_dim,
        })
    }

    /// Load predictor from pre-trained weights.
    #[inline(always)]
    pub fn from_weights(
        weight: Vec<T>,
        weight_shape: [usize; 2],
        bias: f32,
        fusion_layers: usize,
        hidden_dim: usize,
    ) -> Result<Self, &'static str> {
        if weight_shape[1] != 1 {
            return Err("output dimension must be 1");
        }
        let fused_dim = hidden_dim * fusion_layers;
        if weight_shape[0] != fused_dim {
            return Err("weight dimension mismatch with fusion_layers * hidden_dim");
        }
        if weight.len() != fused_dim {
            return Err("weight length mismatch with fused_dim");
        }

        Ok(Self {
            weight,
            weight_shape,
            bias,
            fusion_layers,
            hidden_dim,
        })
    }

    /// Predict acceptance probabilities from fused hidden states.
    #[inline(always)]
    pub fn predict(
        &self,
        fused_hidden: &[T],
        batch: usize,
        seq_len: usize,
        output: &mut [T],
    ) -> Result<(), &'static str> {
        let fused_dim = self.fused_dim();
        let expected_hidden = batch * seq_len * fused_dim;
        if fused_hidden.len() != expected_hidden {
            return Err("fused_hidden length mismatch");
        }
        let expected_output = batch * seq_len;
        if output.len() != expected_output {
            return Err("output length mismatch");
        }
        if self.weight.len() != fused_dim {
            return Err("weight length mismatch");
        }

        for pos in 0..expected_output {
            let base = pos * fused_dim;
            let mut logit = self.bias;
            for d in 0..fused_dim {
                logit += fused_hidden[base + d].to_f32() * self.weight[d].to_f32();
            }
            output[pos] = T::from_f32(math::sigmoid(logit));
        }

        Ok(())
    }

    /// Predict single token acceptance probability (for incremental generation).
    #[inline(always)]
    pub fn predict_single(&self, fused_hidden: &[T]) -> Result<f32, &'static str> {
        let fused_dim = self.fused_dim();
        if fused_hidden.len() != fused_dim {
            return Err("fused_hidden length mismatch");
        }
        if self.weight.len() != fused_dim {
            return Err("weight length mismatch");
        }

        let mut logit = self.bias;
        for d in 0..fused_dim {
            logit += fused_hidden[d].to_f32() * self.weight[d].to_f32();
        }

        Ok(math::sigmoid(logit))
    }

    #[inline(always)]
    fn fused_dim(&self) -> usize {
        self.hidden_dim * self.fusion_layers
    }
}
