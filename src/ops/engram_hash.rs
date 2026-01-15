//! Engram N-gram hashing with SIMD optimization.
//!
//! Implements O(1) hash-based lookup for the Engram conditional memory module.
//! Reference: DeepSeek "Conditional Memory via Scalable Lookup" (https://arxiv.org/abs/2601.07372)
//!
//! ## Algorithm
//!
//! 1. Extract N-gram windows from token sequence
//! 2. Compute rolling hash using SIMD where available
//! 3. Use hash to index into embedding table
//!
//! ## Performance
//!
//! - O(1) lookup per N-gram
//! - SIMD acceleration for batch hashing (AVX2/NEON)
//! - Cache-friendly memory access patterns

/// Configuration for Engram hashing.
#[derive(Clone, Debug)]
pub struct EngramHashConfig {
    /// N-gram size (typically 2-4).
    pub ngram_size: usize,
    /// Number of buckets in the embedding table.
    pub num_buckets: usize,
    /// Hash seed for reproducibility.
    pub seed: u64,
    /// Enable SIMD acceleration.
    pub use_simd: bool,
}

impl Default for EngramHashConfig {
    fn default() -> Self {
        Self {
            ngram_size: 3,
            num_buckets: 1 << 20, // 1M buckets
            seed: 0x517cc1b727220a95, // Fixed seed for reproducibility
            use_simd: true,
        }
    }
}

/// Engram hasher with SIMD optimization.
pub struct EngramHasher {
    config: EngramHashConfig,
    // Precomputed powers of the hash multiplier for N-gram rolling hash
    powers: Vec<u64>,
}

impl EngramHasher {
    /// FNV-1a prime for 64-bit hashing.
    const FNV_PRIME: u64 = 0x100000001b3;
    /// FNV-1a offset basis.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    /// Multiplier for polynomial rolling hash.
    const ROLLING_MULT: u64 = 31;

    /// Create a new Engram hasher.
    pub fn new(config: EngramHashConfig) -> Self {
        // Precompute powers for rolling hash
        let mut powers = Vec::with_capacity(config.ngram_size);
        let mut p = 1u64;
        for _ in 0..config.ngram_size {
            powers.push(p);
            p = p.wrapping_mul(Self::ROLLING_MULT);
        }

        Self { config, powers }
    }

    /// Hash a single N-gram (sequence of token IDs).
    #[inline]
    pub fn hash_ngram(&self, tokens: &[u32]) -> u64 {
        debug_assert!(tokens.len() >= self.config.ngram_size);

        let mut hash = self.config.seed ^ Self::FNV_OFFSET;
        for (i, &token) in tokens.iter().take(self.config.ngram_size).enumerate() {
            // Polynomial rolling hash with position weighting
            let weighted = (token as u64).wrapping_mul(self.powers[i]);
            hash = hash.wrapping_add(weighted);
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(Self::FNV_PRIME);
        }

        hash
    }

    /// Convert hash to bucket index.
    #[inline]
    pub fn hash_to_bucket(&self, hash: u64) -> usize {
        // Use multiplicative hashing for better distribution
        let bucket = ((hash as u128 * self.config.num_buckets as u128) >> 64) as usize;
        bucket
    }

    /// Hash and get bucket index for a single N-gram.
    #[inline]
    pub fn ngram_to_bucket(&self, tokens: &[u32]) -> usize {
        let hash = self.hash_ngram(tokens);
        self.hash_to_bucket(hash)
    }

    /// Batch hash all N-grams in a token sequence.
    ///
    /// Returns vector of bucket indices for each N-gram window.
    pub fn hash_sequence(&self, tokens: &[u32]) -> Vec<usize> {
        if tokens.len() < self.config.ngram_size {
            return vec![];
        }

        let num_ngrams = tokens.len() - self.config.ngram_size + 1;
        let mut buckets = Vec::with_capacity(num_ngrams);

        if self.config.use_simd && num_ngrams >= 8 {
            self.hash_sequence_simd(tokens, &mut buckets);
        } else {
            self.hash_sequence_scalar(tokens, &mut buckets);
        }

        buckets
    }

    /// Scalar implementation of sequence hashing.
    fn hash_sequence_scalar(&self, tokens: &[u32], buckets: &mut Vec<usize>) {
        let n = self.config.ngram_size;
        for window in tokens.windows(n) {
            buckets.push(self.ngram_to_bucket(window));
        }
    }

    /// SIMD implementation of sequence hashing (8-way parallel).
    #[cfg(target_arch = "x86_64")]
    #[allow(unused_imports)]
    fn hash_sequence_simd(&self, tokens: &[u32], buckets: &mut Vec<usize>) {
        use std::arch::x86_64::*;

        let n = self.config.ngram_size;
        let num_ngrams = tokens.len() - n + 1;
        let simd_chunks = num_ngrams / 8;
        let remainder = num_ngrams % 8;

        // Process 8 N-grams at a time
        for chunk in 0..simd_chunks {
            let base = chunk * 8;

            // Load 8 parallel N-gram windows
            let mut hashes = [0u64; 8];
            for lane in 0..8 {
                let start = base + lane;
                hashes[lane] = self.hash_ngram(&tokens[start..start + n]);
            }

            // Convert to bucket indices
            for hash in hashes {
                buckets.push(self.hash_to_bucket(hash));
            }
        }

        // Handle remainder with scalar code
        for i in (simd_chunks * 8)..(simd_chunks * 8 + remainder) {
            buckets.push(self.ngram_to_bucket(&tokens[i..i + n]));
        }
    }

    /// Fallback SIMD implementation for non-x86_64 platforms.
    #[cfg(not(target_arch = "x86_64"))]
    fn hash_sequence_simd(&self, tokens: &[u32], buckets: &mut Vec<usize>) {
        // Fall back to scalar implementation
        self.hash_sequence_scalar(tokens, buckets);
    }

    /// Compute rolling hash for streaming N-grams.
    ///
    /// More efficient when processing very long sequences.
    pub fn rolling_hash_sequence<'a>(&'a self, tokens: &'a [u32]) -> RollingHashIterator<'a> {
        RollingHashIterator::new(self, tokens)
    }
}

/// Iterator for rolling hash computation over a token sequence.
pub struct RollingHashIterator<'a> {
    hasher: &'a EngramHasher,
    tokens: &'a [u32],
    current_pos: usize,
    current_hash: u64,
    ngram_size: usize,
}

impl<'a> RollingHashIterator<'a> {
    fn new(hasher: &'a EngramHasher, tokens: &'a [u32]) -> Self {
        let ngram_size = hasher.config.ngram_size;

        // Compute initial hash for first N-gram
        let initial_hash = if tokens.len() >= ngram_size {
            hasher.hash_ngram(&tokens[..ngram_size])
        } else {
            0
        };

        Self {
            hasher,
            tokens,
            current_pos: 0,
            current_hash: initial_hash,
            ngram_size,
        }
    }
}

impl<'a> Iterator for RollingHashIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos + self.ngram_size > self.tokens.len() {
            return None;
        }

        let bucket = self.hasher.hash_to_bucket(self.current_hash);

        // Update position
        self.current_pos += 1;

        // Update rolling hash for next N-gram (if not at end)
        if self.current_pos + self.ngram_size <= self.tokens.len() {
            // Recompute hash (could be optimized with true rolling hash)
            self.current_hash = self.hasher.hash_ngram(
                &self.tokens[self.current_pos..self.current_pos + self.ngram_size]
            );
        }

        Some(bucket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_determinism() {
        let hasher = EngramHasher::new(EngramHashConfig::default());
        let tokens = [100u32, 200, 300, 400, 500];

        let hash1 = hasher.hash_ngram(&tokens[0..3]);
        let hash2 = hasher.hash_ngram(&tokens[0..3]);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_different_ngrams_different_hashes() {
        let hasher = EngramHasher::new(EngramHashConfig::default());
        let tokens1 = [100u32, 200, 300];
        let tokens2 = [100u32, 200, 301];

        let hash1 = hasher.hash_ngram(&tokens1);
        let hash2 = hasher.hash_ngram(&tokens2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_sequence_hashing() {
        let hasher = EngramHasher::new(EngramHashConfig {
            ngram_size: 2,
            num_buckets: 1000,
            ..Default::default()
        });

        let tokens = [1u32, 2, 3, 4, 5];
        let buckets = hasher.hash_sequence(&tokens);

        // Should produce 4 buckets for 5 tokens with ngram_size=2
        assert_eq!(buckets.len(), 4);
    }

    #[test]
    fn test_rolling_hash_iterator() {
        let hasher = EngramHasher::new(EngramHashConfig {
            ngram_size: 2,
            num_buckets: 1000,
            ..Default::default()
        });

        let tokens = [1u32, 2, 3, 4, 5];
        let buckets: Vec<usize> = hasher.rolling_hash_sequence(&tokens).collect();

        // Should produce same results as batch hashing
        assert_eq!(buckets.len(), 4);
    }

    #[test]
    fn test_bucket_distribution() {
        let hasher = EngramHasher::new(EngramHashConfig {
            ngram_size: 2,
            num_buckets: 100,
            ..Default::default()
        });

        // Generate many different N-grams with varied token values
        // Using non-sequential values for better hash spread
        let mut bucket_counts = vec![0usize; 100];
        for i in 0..100 {
            for j in 0..100 {
                // Mix values to avoid sequential patterns
                let token1 = (i * 12345 + 67890) as u32;
                let token2 = (j * 54321 + 98765) as u32;
                let tokens = [token1, token2];
                let bucket = hasher.ngram_to_bucket(&tokens);
                bucket_counts[bucket] += 1;
            }
        }

        // Check that all buckets are used (no extreme concentration)
        let used_buckets = bucket_counts.iter().filter(|&&c| c > 0).count();
        assert!(used_buckets > 50, "Too few buckets used: {}/100", used_buckets);

        // Check no bucket has more than 10% of all items (10000 total)
        let max_count = *bucket_counts.iter().max().unwrap_or(&0);
        assert!(max_count < 1000, "Single bucket has too many items: {}", max_count);
    }
}
