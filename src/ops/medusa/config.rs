/// Configuration for Medusa assisted generation.
#[derive(Debug, Clone)]
pub struct AssistedGenerationConfig {
    /// Number of Medusa heads (default: 3).
    pub num_medusa_heads: usize,
    /// Speculation depth per head (default: 4).
    pub speculation_depth: usize,
    /// Number of candidate tokens per position (default: 8).
    pub candidate_count: usize,
    /// Use N-gram assisted drafting.
    pub use_ngram_draft: bool,
    /// N-gram size (default: 3).
    pub ngram_size: usize,
    /// Use tree attention for verification.
    pub tree_attention: bool,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Temperature for sampling (0 = greedy).
    pub temperature: f32,
}

impl Default for AssistedGenerationConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            num_medusa_heads: 3,
            speculation_depth: 4,
            candidate_count: 8,
            use_ngram_draft: true,
            ngram_size: 3,
            tree_attention: true,
            hidden_dim: 768,
            vocab_size: 32000,
            temperature: 0.0,
        }
    }
}

impl AssistedGenerationConfig {
    /// Validate configuration.
    #[inline(always)]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_medusa_heads == 0 {
            return Err("num_medusa_heads must be > 0");
        }
        if self.speculation_depth == 0 {
            return Err("speculation_depth must be > 0");
        }
        if self.candidate_count == 0 {
            return Err("candidate_count must be > 0");
        }
        if self.ngram_size < 2 {
            return Err("ngram_size must be >= 2");
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0");
        }
        Ok(())
    }
}
