//! Engram embedding lookup with memory mapping and prefetch.
//!
//! Implements efficient O(1) embedding retrieval for the Engram conditional memory module.
//! Reference: DeepSeek "Conditional Memory via Scalable Lookup" (https://arxiv.org/abs/2601.07372)
//!
//! ## Features
//!
//! - Memory-mapped embedding tables (DRAM/NVMe storage)
//! - Software prefetching for latency hiding
//! - Batch lookup with gather operations
//! - Zero-copy access to large embedding tables
//!
//! ## Memory Layout
//!
//! ```text
//! Embedding Table (mmap'ed):
//! ┌──────────────────────────────────────────────────────────────┐
//! │  Bucket 0: [dim0, dim1, ..., dim_n]                         │
//! │  Bucket 1: [dim0, dim1, ..., dim_n]                         │
//! │  ...                                                         │
//! │  Bucket M: [dim0, dim1, ..., dim_n]                         │
//! └──────────────────────────────────────────────────────────────┘
//! ```

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use super::engram_hash::{EngramHashConfig, EngramHasher};

/// Configuration for Engram lookup.
#[derive(Clone, Debug)]
pub struct EngramLookupConfig {
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Number of buckets (must match hasher config).
    pub num_buckets: usize,
    /// Prefetch distance (number of buckets to prefetch ahead).
    pub prefetch_distance: usize,
    /// Whether to use prefetching.
    pub enable_prefetch: bool,
}

impl Default for EngramLookupConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 4096,
            num_buckets: 1 << 20, // 1M buckets
            prefetch_distance: 8,
            enable_prefetch: true,
        }
    }
}

/// Error types for Engram operations.
#[derive(Debug)]
pub enum EngramError {
    /// IO error during file operations.
    Io(io::Error),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Bucket index out of bounds.
    BucketOutOfBounds(usize),
    /// Memory mapping failed.
    MmapFailed(String),
}

impl std::fmt::Display for EngramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::BucketOutOfBounds(idx) => write!(f, "Bucket index {} out of bounds", idx),
            Self::MmapFailed(msg) => write!(f, "Memory mapping failed: {}", msg),
        }
    }
}

impl std::error::Error for EngramError {}

impl From<io::Error> for EngramError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Memory-mapped Engram embedding table.
///
/// Uses mmap for zero-copy access to large embedding tables stored on disk.
pub struct EngramEmbeddingTable {
    config: EngramLookupConfig,
    mmap: Mmap,
    /// Bytes per bucket (embedding_dim * sizeof(f32)).
    bucket_size: usize,
}

impl EngramEmbeddingTable {
    /// Create a new embedding table from a file.
    ///
    /// The file must contain `num_buckets * embedding_dim * 4` bytes
    /// (f32 embeddings in row-major order).
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        config: EngramLookupConfig,
    ) -> Result<Self, EngramError> {
        let file = File::open(path)?;
        let bucket_size = config.embedding_dim * std::mem::size_of::<f32>();
        let expected_size = config.num_buckets * bucket_size;

        let file_len = file.metadata()?.len() as usize;
        if file_len < expected_size {
            return Err(EngramError::InvalidConfig(format!(
                "File size {} is less than expected {}",
                file_len, expected_size
            )));
        }

        let mmap = unsafe {
            MmapOptions::new()
                .len(expected_size)
                .map(&file)
                .map_err(|e| EngramError::MmapFailed(e.to_string()))?
        };

        Ok(Self {
            config,
            mmap,
            bucket_size,
        })
    }

    /// Create a new embedding table from raw bytes.
    ///
    /// Useful for testing or when embeddings are already in memory.
    pub fn from_bytes(data: Vec<f32>, config: EngramLookupConfig) -> Result<Self, EngramError> {
        let _bucket_size = config.embedding_dim * std::mem::size_of::<f32>();
        let expected_elements = config.num_buckets * config.embedding_dim;

        if data.len() < expected_elements {
            return Err(EngramError::InvalidConfig(format!(
                "Data length {} is less than expected {}",
                data.len(), expected_elements
            )));
        }

        // For in-memory data, we create a temporary file and mmap it
        // This allows unified interface but is less efficient than direct access
        // Use thread ID and timestamp for uniqueness in parallel tests
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_path = std::env::temp_dir().join(format!("engram_temp_{}_{:x}", std::process::id(), unique_id));
        let mut file = File::create(&temp_path)?;

        // Write f32 data as bytes
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);

        let result = Self::from_file(&temp_path, config);

        // Clean up temp file
        let _ = std::fs::remove_file(&temp_path);

        result
    }

    /// Get embedding at a specific bucket index.
    #[inline]
    pub fn get_embedding(&self, bucket: usize) -> Result<&[f32], EngramError> {
        if bucket >= self.config.num_buckets {
            return Err(EngramError::BucketOutOfBounds(bucket));
        }

        let offset = bucket * self.bucket_size;
        let bytes = &self.mmap[offset..offset + self.bucket_size];

        // Safety: we know the bytes are properly aligned f32 values
        let embedding = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.config.embedding_dim,
            )
        };

        Ok(embedding)
    }

    /// Prefetch embedding at a specific bucket index.
    #[inline]
    pub fn prefetch(&self, bucket: usize) {
        if bucket >= self.config.num_buckets {
            return;
        }

        let offset = bucket * self.bucket_size;
        let ptr = self.mmap.as_ptr().wrapping_add(offset);

        // Software prefetch using platform-specific intrinsics
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::_prefetch;
            _prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
        }
    }

    /// Batch lookup multiple embeddings.
    ///
    /// Returns a vector of embeddings corresponding to the bucket indices.
    /// Uses prefetching to hide memory latency.
    pub fn batch_lookup(&self, buckets: &[usize]) -> Vec<Vec<f32>> {
        let mut results = Vec::with_capacity(buckets.len());

        if self.config.enable_prefetch {
            // Prefetch ahead
            for (i, &bucket) in buckets.iter().enumerate() {
                // Prefetch future buckets
                if i + self.config.prefetch_distance < buckets.len() {
                    self.prefetch(buckets[i + self.config.prefetch_distance]);
                }

                // Lookup current bucket
                if let Ok(embedding) = self.get_embedding(bucket) {
                    results.push(embedding.to_vec());
                } else {
                    // Return zero embedding for invalid buckets
                    results.push(vec![0.0; self.config.embedding_dim]);
                }
            }
        } else {
            // No prefetching
            for &bucket in buckets {
                if let Ok(embedding) = self.get_embedding(bucket) {
                    results.push(embedding.to_vec());
                } else {
                    results.push(vec![0.0; self.config.embedding_dim]);
                }
            }
        }

        results
    }

    /// Batch lookup with output buffer (avoids allocation).
    pub fn batch_lookup_into(&self, buckets: &[usize], output: &mut [f32]) {
        let dim = self.config.embedding_dim;
        debug_assert!(output.len() >= buckets.len() * dim);

        if self.config.enable_prefetch {
            for (i, &bucket) in buckets.iter().enumerate() {
                // Prefetch future buckets
                if i + self.config.prefetch_distance < buckets.len() {
                    self.prefetch(buckets[i + self.config.prefetch_distance]);
                }

                // Lookup current bucket
                let out_slice = &mut output[i * dim..(i + 1) * dim];
                if let Ok(embedding) = self.get_embedding(bucket) {
                    out_slice.copy_from_slice(embedding);
                } else {
                    out_slice.fill(0.0);
                }
            }
        } else {
            for (i, &bucket) in buckets.iter().enumerate() {
                let out_slice = &mut output[i * dim..(i + 1) * dim];
                if let Ok(embedding) = self.get_embedding(bucket) {
                    out_slice.copy_from_slice(embedding);
                } else {
                    out_slice.fill(0.0);
                }
            }
        }
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get the number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.config.num_buckets
    }

    /// Get the total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.mmap.len()
    }
}

/// Engram module combining hasher and embedding table.
///
/// Provides end-to-end O(1) lookup for token sequences.
pub struct EngramModule {
    hasher: EngramHasher,
    table: EngramEmbeddingTable,
}

impl EngramModule {
    /// Create a new Engram module.
    pub fn new(hasher: EngramHasher, table: EngramEmbeddingTable) -> Self {
        Self { hasher, table }
    }

    /// Load an Engram module from file.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        hash_config: EngramHashConfig,
        lookup_config: EngramLookupConfig,
    ) -> Result<Self, EngramError> {
        let hasher = EngramHasher::new(hash_config);
        let table = EngramEmbeddingTable::from_file(path, lookup_config)?;
        Ok(Self { hasher, table })
    }

    /// Lookup embeddings for a token sequence.
    ///
    /// Returns one embedding per N-gram in the sequence.
    pub fn lookup(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let buckets = self.hasher.hash_sequence(tokens);
        self.table.batch_lookup(&buckets)
    }

    /// Lookup embeddings into a pre-allocated buffer.
    pub fn lookup_into(&self, tokens: &[u32], output: &mut [f32]) {
        let buckets = self.hasher.hash_sequence(tokens);
        self.table.batch_lookup_into(&buckets, output);
    }

    /// Aggregate embeddings for a token sequence.
    ///
    /// Returns the sum of all N-gram embeddings (can be used for pooling).
    pub fn aggregate(&self, tokens: &[u32]) -> Vec<f32> {
        let embeddings = self.lookup(tokens);
        if embeddings.is_empty() {
            return vec![0.0; self.table.embedding_dim()];
        }

        let mut result = vec![0.0; self.table.embedding_dim()];
        for embedding in &embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                result[i] += val;
            }
        }

        result
    }

    /// Get the hasher.
    pub fn hasher(&self) -> &EngramHasher {
        &self.hasher
    }

    /// Get the embedding table.
    pub fn table(&self) -> &EngramEmbeddingTable {
        &self.table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_table(num_buckets: usize, embedding_dim: usize) -> EngramEmbeddingTable {
        // Create random embeddings
        let data: Vec<f32> = (0..num_buckets * embedding_dim)
            .map(|i| (i as f32) / 1000.0)
            .collect();

        let config = EngramLookupConfig {
            embedding_dim,
            num_buckets,
            ..Default::default()
        };

        EngramEmbeddingTable::from_bytes(data, config).unwrap()
    }

    #[test]
    fn test_single_lookup() {
        let table = create_test_table(100, 16);
        let embedding = table.get_embedding(0).unwrap();

        assert_eq!(embedding.len(), 16);
        // First bucket should have values 0/1000, 1/1000, ..., 15/1000
        assert!((embedding[0] - 0.0).abs() < 1e-6);
        assert!((embedding[1] - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_batch_lookup() {
        let table = create_test_table(100, 16);
        let buckets = vec![0, 1, 2];
        let embeddings = table.batch_lookup(&buckets);

        assert_eq!(embeddings.len(), 3);
        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(embedding.len(), 16);
            // Check first value of each embedding
            let expected_first = (i * 16) as f32 / 1000.0;
            assert!((embedding[0] - expected_first).abs() < 1e-6);
        }
    }

    #[test]
    fn test_engram_module() {
        let table = create_test_table(1000, 16);
        let hasher = EngramHasher::new(EngramHashConfig {
            ngram_size: 2,
            num_buckets: 1000,
            ..Default::default()
        });

        let module = EngramModule::new(hasher, table);

        let tokens = [1u32, 2, 3, 4, 5];
        let embeddings = module.lookup(&tokens);

        // 5 tokens with ngram_size=2 should produce 4 embeddings
        assert_eq!(embeddings.len(), 4);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 16);
        }
    }

    #[test]
    fn test_aggregate() {
        let table = create_test_table(1000, 16);
        let hasher = EngramHasher::new(EngramHashConfig {
            ngram_size: 2,
            num_buckets: 1000,
            ..Default::default()
        });

        let module = EngramModule::new(hasher, table);

        let tokens = [1u32, 2, 3];
        let aggregated = module.aggregate(&tokens);

        assert_eq!(aggregated.len(), 16);
        // Should be non-zero (sum of embeddings)
        assert!(aggregated.iter().any(|&x| x.abs() > 1e-6));
    }
}
