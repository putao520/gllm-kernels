/// Configuration for EAGLE-3 adaptive draft generation.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveDraftConfig {
    /// Minimum draft length (default: 1).
    pub min_draft_length: usize,
    /// Maximum draft length (default: 8).
    pub max_draft_length: usize,
    /// Confidence threshold for early termination (default: 0.5).
    pub confidence_threshold: f32,
    /// Fallback length after verification failure (default: 3).
    pub fallback_length: usize,
    /// Enable length scheduler (default: true).
    pub enable_length_scheduler: bool,
    /// Number of layers to fuse for confidence prediction (default: 4).
    pub fusion_layers: usize,
    /// Hidden dimension for confidence predictor.
    pub hidden_dim: usize,
}

impl Default for AdaptiveDraftConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            min_draft_length: 1,
            max_draft_length: 8,
            confidence_threshold: 0.5,
            fallback_length: 3,
            enable_length_scheduler: true,
            fusion_layers: 4,
            hidden_dim: 768,
        }
    }
}

impl AdaptiveDraftConfig {
    /// Validate configuration parameters.
    #[inline(always)]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_draft_length == 0 {
            return Err("min_draft_length must be > 0");
        }
        if self.max_draft_length < self.min_draft_length {
            return Err("max_draft_length must be >= min_draft_length");
        }
        if self.confidence_threshold <= 0.0 || self.confidence_threshold > 1.0 {
            return Err("confidence_threshold must be in (0, 1]");
        }
        if self.fallback_length == 0 {
            return Err("fallback_length must be > 0");
        }
        if self.fusion_layers == 0 {
            return Err("fusion_layers must be > 0");
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        Ok(())
    }

    /// Create a new config with specified hidden dimension.
    #[inline(always)]
    pub fn with_hidden_dim(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            ..Default::default()
        }
    }

    /// Fused hidden dimension (hidden_dim * fusion_layers).
    #[inline(always)]
    pub fn fused_dim(&self) -> usize {
        self.hidden_dim * self.fusion_layers
    }
}
