//! Engram conditional memory module for O(1) static knowledge lookup.
//!
//! Reference: DeepSeek "Conditional Memory via Scalable Lookup" (https://arxiv.org/abs/2601.07372)
//!
//! ## Overview
//!
//! Engram separates static knowledge (factual, encyclopedic) from dynamic reasoning
//! by using O(1) hash-based lookup instead of attention for static knowledge retrieval.
//!
//! ```text
//! Input Tokens
//!      │
//!      ├────────────────┬───────────────────────────────┐
//!      │                │                               │
//!      ▼                ▼                               │
//! ┌─────────┐    ┌────────────────┐                    │
//! │ N-gram  │    │   Attention    │                    │
//! │ Hashing │    │   (动态推理)    │                    │
//! └────┬────┘    └───────┬────────┘                    │
//!      │                 │                              │
//!      ▼                 │                              │
//! ┌─────────────┐        │  75-80% 参数                 │
//! │  Engram     │◄───────┤                              │
//! │  Lookup     │        │                              │
//! │  (DRAM)     │        │                              │
//! └──────┬──────┘        │                              │
//!        │               │                              │
//!        └───────────────┼──────────────────────────────┘
//!                        ▼
//!                 ┌────────────┐
//!                 │   Output   │
//!                 └────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use gllm_kernels::ops::engram::{Engram, EngramConfig};
//!
//! // Initialize with configuration
//! let config = EngramConfig::default();
//! let engram = Engram::from_file("embeddings.bin", config)?;
//!
//! // Lookup embeddings for token sequence
//! let tokens = [100, 200, 300, 400, 500];
//! let embeddings = engram.forward(&tokens);
//! ```
//!
//! ## Memory Efficiency
//!
//! - Static knowledge stored in system DRAM (not GPU HBM)
//! - Memory-mapped files for very large tables (can exceed RAM)
//! - Only active embeddings transferred to GPU
//! - Reduces HBM usage by 20-25% for factual knowledge

use std::path::Path;

use super::engram_hash::{EngramHashConfig, EngramHasher};
use super::engram_lookup::{EngramEmbeddingTable, EngramError, EngramLookupConfig, EngramModule};

/// Configuration for the Engram module.
#[derive(Clone, Debug)]
pub struct EngramConfig {
    /// Hash configuration.
    pub hash: EngramHashConfig,
    /// Lookup configuration.
    pub lookup: EngramLookupConfig,
    /// Whether to fuse Engram output with attention output.
    pub fuse_with_attention: bool,
    /// Scaling factor for Engram output (before addition).
    pub output_scale: f32,
}

impl Default for EngramConfig {
    fn default() -> Self {
        Self {
            hash: EngramHashConfig::default(),
            lookup: EngramLookupConfig::default(),
            fuse_with_attention: true,
            output_scale: 1.0,
        }
    }
}

impl EngramConfig {
    /// Create a small configuration for testing.
    pub fn small() -> Self {
        Self {
            hash: EngramHashConfig {
                ngram_size: 2,
                num_buckets: 1 << 16, // 64K buckets
                ..Default::default()
            },
            lookup: EngramLookupConfig {
                embedding_dim: 512,
                num_buckets: 1 << 16,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a large configuration for production.
    pub fn large() -> Self {
        Self {
            hash: EngramHashConfig {
                ngram_size: 4,
                num_buckets: 1 << 24, // 16M buckets
                ..Default::default()
            },
            lookup: EngramLookupConfig {
                embedding_dim: 4096,
                num_buckets: 1 << 24,
                prefetch_distance: 16,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Engram conditional memory module.
///
/// Provides O(1) hash-based lookup for static knowledge retrieval.
pub struct Engram {
    module: EngramModule,
    config: EngramConfig,
}

impl Engram {
    /// Create a new Engram module from an embedding file.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        config: EngramConfig,
    ) -> Result<Self, EngramError> {
        let module = EngramModule::from_file(path, config.hash.clone(), config.lookup.clone())?;
        Ok(Self { module, config })
    }

    /// Create a new Engram module from in-memory data.
    pub fn from_bytes(data: Vec<f32>, config: EngramConfig) -> Result<Self, EngramError> {
        let hasher = EngramHasher::new(config.hash.clone());
        let table = EngramEmbeddingTable::from_bytes(data, config.lookup.clone())?;
        let module = EngramModule::new(hasher, table);
        Ok(Self { module, config })
    }

    /// Forward pass: lookup embeddings for a token sequence.
    ///
    /// Returns a tensor of shape [num_ngrams, embedding_dim].
    pub fn forward(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.module.lookup(tokens)
    }

    /// Forward pass with output scaling.
    pub fn forward_scaled(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let mut embeddings = self.forward(tokens);
        if (self.config.output_scale - 1.0).abs() > 1e-6 {
            for embedding in &mut embeddings {
                for val in embedding.iter_mut() {
                    *val *= self.config.output_scale;
                }
            }
        }
        embeddings
    }

    /// Forward pass into pre-allocated buffer.
    ///
    /// Buffer must have size >= num_ngrams * embedding_dim.
    pub fn forward_into(&self, tokens: &[u32], output: &mut [f32]) {
        self.module.lookup_into(tokens, output);

        // Apply scaling if needed
        if (self.config.output_scale - 1.0).abs() > 1e-6 {
            for val in output.iter_mut() {
                *val *= self.config.output_scale;
            }
        }
    }

    /// Get aggregated embedding for the entire sequence.
    ///
    /// Returns the mean of all N-gram embeddings.
    pub fn aggregate_mean(&self, tokens: &[u32]) -> Vec<f32> {
        let embeddings = self.forward(tokens);
        if embeddings.is_empty() {
            return vec![0.0; self.config.lookup.embedding_dim];
        }

        let mut result = vec![0.0; self.config.lookup.embedding_dim];
        for embedding in &embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                result[i] += val;
            }
        }

        let scale = 1.0 / embeddings.len() as f32;
        for val in &mut result {
            *val *= scale;
        }

        result
    }

    /// Get the number of N-grams for a given token sequence length.
    #[inline]
    pub fn num_ngrams(&self, seq_len: usize) -> usize {
        if seq_len < self.config.hash.ngram_size {
            0
        } else {
            seq_len - self.config.hash.ngram_size + 1
        }
    }

    /// Get the embedding dimension.
    #[inline]
    pub fn embedding_dim(&self) -> usize {
        self.config.lookup.embedding_dim
    }

    /// Get the N-gram size.
    #[inline]
    pub fn ngram_size(&self) -> usize {
        self.config.hash.ngram_size
    }

    /// Get the configuration.
    pub fn config(&self) -> &EngramConfig {
        &self.config
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.module.table().size_bytes()
    }
}

/// Fuse Engram output with attention output.
///
/// out = attention_out + scale * engram_out
pub fn fuse_engram_attention(
    attention_out: &mut [f32],
    engram_out: &[f32],
    scale: f32,
) {
    debug_assert_eq!(attention_out.len(), engram_out.len());

    // SIMD-friendly loop
    for (a, &e) in attention_out.iter_mut().zip(engram_out.iter()) {
        *a += scale * e;
    }
}

/// Fuse Engram output with attention output (batch version).
///
/// For each position i: out[i] = attention_out[i] + scale * engram_out[i]
#[cfg(target_arch = "x86_64")]
pub fn fuse_engram_attention_simd(
    attention_out: &mut [f32],
    engram_out: &[f32],
    scale: f32,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(attention_out.len(), engram_out.len());

    let len = attention_out.len();
    let simd_len = len / 8 * 8;

    unsafe {
        let scale_vec = _mm256_set1_ps(scale);

        for i in (0..simd_len).step_by(8) {
            let a = _mm256_loadu_ps(attention_out.as_ptr().add(i));
            let e = _mm256_loadu_ps(engram_out.as_ptr().add(i));
            let result = _mm256_fmadd_ps(e, scale_vec, a); // a + scale * e
            _mm256_storeu_ps(attention_out.as_mut_ptr().add(i), result);
        }
    }

    // Handle remainder
    for i in simd_len..len {
        attention_out[i] += scale * engram_out[i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn fuse_engram_attention_simd(
    attention_out: &mut [f32],
    engram_out: &[f32],
    scale: f32,
) {
    fuse_engram_attention(attention_out, engram_out, scale);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_engram() -> Engram {
        let config = EngramConfig::small();
        let num_elements = config.lookup.num_buckets * config.lookup.embedding_dim;
        let data: Vec<f32> = (0..num_elements)
            .map(|i| (i as f32) / 1000.0)
            .collect();

        Engram::from_bytes(data, config).unwrap()
    }

    #[test]
    fn test_forward() {
        let engram = create_test_engram();
        let tokens = [1u32, 2, 3, 4, 5];

        let embeddings = engram.forward(&tokens);

        // With ngram_size=2 and 5 tokens, should get 4 embeddings
        assert_eq!(embeddings.len(), 4);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), engram.embedding_dim());
        }
    }

    #[test]
    fn test_forward_scaled() {
        let mut config = EngramConfig::small();
        config.output_scale = 0.5;

        let num_elements = config.lookup.num_buckets * config.lookup.embedding_dim;
        let data: Vec<f32> = (0..num_elements)
            .map(|i| (i as f32) / 1000.0)
            .collect();

        let engram = Engram::from_bytes(data, config).unwrap();
        let tokens = [1u32, 2, 3];

        let scaled = engram.forward_scaled(&tokens);
        let unscaled = engram.forward(&tokens);

        // Check that scaling was applied
        for (s, u) in scaled.iter().zip(unscaled.iter()) {
            for (sv, uv) in s.iter().zip(u.iter()) {
                assert!((sv - uv * 0.5).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_aggregate_mean() {
        let engram = create_test_engram();
        let tokens = [1u32, 2, 3, 4, 5];

        let mean = engram.aggregate_mean(&tokens);

        assert_eq!(mean.len(), engram.embedding_dim());
        // Mean should be non-zero
        assert!(mean.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_num_ngrams() {
        let engram = create_test_engram();

        assert_eq!(engram.num_ngrams(5), 4); // 5 - 2 + 1 = 4
        assert_eq!(engram.num_ngrams(2), 1); // 2 - 2 + 1 = 1
        assert_eq!(engram.num_ngrams(1), 0); // Too short
    }

    #[test]
    fn test_fuse_engram_attention() {
        let mut attention_out = vec![1.0f32, 2.0, 3.0, 4.0];
        let engram_out = vec![0.5f32, 0.5, 0.5, 0.5];

        fuse_engram_attention(&mut attention_out, &engram_out, 2.0);

        assert!((attention_out[0] - 2.0).abs() < 1e-6); // 1.0 + 2.0 * 0.5 = 2.0
        assert!((attention_out[1] - 3.0).abs() < 1e-6); // 2.0 + 2.0 * 0.5 = 3.0
    }

    #[test]
    fn test_fuse_simd() {
        // Test with larger array to exercise SIMD path
        let mut attention_out: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let engram_out: Vec<f32> = vec![1.0; 100];
        let original = attention_out.clone();

        fuse_engram_attention_simd(&mut attention_out, &engram_out, 0.5);

        for (i, (&a, &o)) in attention_out.iter().zip(original.iter()).enumerate() {
            let expected = o + 0.5 * 1.0;
            assert!((a - expected).abs() < 1e-6, "Mismatch at {}: {} vs {}", i, a, expected);
        }
    }
}
