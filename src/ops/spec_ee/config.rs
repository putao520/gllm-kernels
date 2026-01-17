/// Configuration for SpecEE/LayerSkip early exit.
#[derive(Debug, Clone)]
pub struct SpecEEConfig {
    /// Layers configured for early exit (e.g., [6, 12, 18]).
    pub exit_layers: Vec<usize>,
    /// Confidence threshold for early exit (default: 0.8).
    pub confidence_threshold: f32,
    /// Minimum layer index for exit (ensures quality).
    pub min_exit_layer: usize,
    /// Self-speculation depth.
    pub speculation_depth: usize,
    /// Enable layer dropout training mode.
    pub enable_layer_dropout: bool,
    /// Enable shared activations between draft and verify.
    pub share_activations: bool,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Total number of layers in the model.
    pub num_layers: usize,
}

impl Default for SpecEEConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            exit_layers: vec![6, 12, 18],
            confidence_threshold: 0.8,
            min_exit_layer: 6,
            speculation_depth: 4,
            enable_layer_dropout: true,
            share_activations: true,
            hidden_dim: 768,
            vocab_size: 32000,
            num_layers: 24,
        }
    }
}

impl SpecEEConfig {
    /// Validate configuration.
    #[inline(always)]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.exit_layers.is_empty() {
            return Err("exit_layers must not be empty");
        }
        if self.confidence_threshold <= 0.0 || self.confidence_threshold > 1.0 {
            return Err("confidence_threshold must be in (0, 1]");
        }
        if self.min_exit_layer == 0 {
            return Err("min_exit_layer must be > 0");
        }
        if self.speculation_depth == 0 {
            return Err("speculation_depth must be > 0");
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0");
        }
        for &layer in &self.exit_layers {
            if layer >= self.num_layers {
                return Err("exit_layer index exceeds num_layers");
            }
            if layer < self.min_exit_layer {
                return Err("exit_layer below min_exit_layer");
            }
        }
        Ok(())
    }

    /// Create config with custom exit layers.
    #[inline(always)]
    pub fn with_exit_layers(exit_layers: Vec<usize>, num_layers: usize) -> Self {
        let min_exit = exit_layers.iter().cloned().min().unwrap_or(6);
        Self {
            exit_layers,
            min_exit_layer: min_exit,
            num_layers,
            ..Default::default()
        }
    }
}
