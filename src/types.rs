//! Types and configuration for attention operations.

/// Result type for configuration validation.
pub type ConfigResult<T> = std::result::Result<T, String>;

/// Precision mode for kernel computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelPrecision {
    /// FP32 throughout (highest precision, slowest).
    FP32,
    /// FP16 compute with FP32 accumulation (balanced).
    #[default]
    FP16,
    /// BF16 compute with FP32 accumulation (best for training).
    BF16,
    /// FP8 compute with FP16 accumulation (fastest, Hopper+).
    FP8,
}

/// Configuration for standard attention.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Query sequence length.
    pub query_len: usize,
    /// Key/Value sequence length.
    pub kv_len: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Whether to apply causal masking.
    pub causal: bool,
    /// Softmax scale factor (usually 1/sqrt(head_dim)).
    pub scale: f32,
    /// Computation precision.
    pub precision: KernelPrecision,
    /// Block size for query tiling.
    pub block_q: usize,
    /// Block size for KV tiling.
    pub block_kv: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 8,
            query_len: 1,
            kv_len: 1024,
            head_dim: 64,
            causal: true,
            scale: 0.125,
            precision: KernelPrecision::FP16,
            block_q: 64,
            block_kv: 64,
        }
    }
}

impl AttentionConfig {
    /// Create config for the given dimensions.
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        query_len: usize,
        kv_len: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            query_len,
            kv_len,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        }
    }

    /// Set causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set precision mode.
    pub fn with_precision(mut self, precision: KernelPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set block sizes for tiling.
    pub fn with_block_sizes(mut self, block_q: usize, block_kv: usize) -> Self {
        self.block_q = block_q;
        self.block_kv = block_kv;
        self
    }

    /// Validate configuration values.
    pub fn validate(&self) -> ConfigResult<()> {
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".to_string());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".to_string());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".to_string());
        }
        if self.head_dim > 256 {
            return Err("head_dim > 256 not supported".to_string());
        }
        if self.block_q == 0 || self.block_kv == 0 {
            return Err("block sizes must be > 0".to_string());
        }
        Ok(())
    }
}

/// Configuration for paged attention.
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Base attention config.
    pub attention: AttentionConfig,
    /// Block size in the page table (tokens per block).
    pub page_block_size: usize,
    /// Maximum number of blocks per sequence.
    pub max_blocks_per_seq: usize,
    /// Maximum number of sequences.
    pub max_num_seqs: usize,
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            attention: AttentionConfig::default(),
            page_block_size: 16,
            max_blocks_per_seq: 128,
            max_num_seqs: 256,
        }
    }
}

impl PagedAttentionConfig {
    /// Create config with the given block size.
    pub fn new(page_block_size: usize) -> Self {
        Self {
            page_block_size,
            ..Default::default()
        }
    }

    /// Set attention config.
    pub fn with_attention(mut self, attention: AttentionConfig) -> Self {
        self.attention = attention;
        self
    }

    /// Set max blocks per sequence.
    pub fn with_max_blocks(mut self, max_blocks: usize) -> Self {
        self.max_blocks_per_seq = max_blocks;
        self
    }

    /// Maximum context length supported.
    pub fn max_context_len(&self) -> usize {
        self.page_block_size * self.max_blocks_per_seq
    }

    /// Validate configuration values.
    pub fn validate(&self) -> ConfigResult<()> {
        self.attention.validate()?;
        if self.page_block_size == 0 {
            return Err("page_block_size must be > 0".to_string());
        }
        if self.max_blocks_per_seq == 0 {
            return Err("max_blocks_per_seq must be > 0".to_string());
        }
        Ok(())
    }
}
