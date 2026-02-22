//! Paged KV Cache for efficient attention computation.
//!
//! Uses a page-table design (inspired by vLLM) where KV data is stored in
//! fixed-size pages. This avoids pre-allocating max_seq_len for every sequence
//! and enables efficient memory sharing for beam search.

use crate::inference::types::{DType, InferenceError, ModelConfig};

/// Page size in tokens.
pub const PAGE_SIZE: usize = 16;

/// A single page of KV cache data.
struct Page {
    /// Raw data: [2 (K+V), num_kv_heads, PAGE_SIZE, head_dim] in compute dtype
    data: Vec<u8>,
    /// Number of tokens currently stored in this page
    used: usize,
}

/// Per-sequence page table mapping logical positions to physical pages.
struct SeqPageTable {
    /// Physical page indices for this sequence
    pages: Vec<usize>,
    /// Current sequence length
    seq_len: usize,
}

/// Paged KV Cache supporting dynamic sequence lengths.
pub struct KvCache {
    /// All physical pages (shared pool)
    pages: Vec<Page>,
    /// Free page indices (stack — pop from back)
    free_pages: Vec<usize>,
    /// Per-layer, per-sequence page tables: layer_tables[layer][seq]
    layer_tables: Vec<Vec<SeqPageTable>>,
    // Config
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    bytes_per_page: usize,
}

impl KvCache {
    /// Allocate a new KV cache.
    pub fn new(
        config: &ModelConfig,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        let bytes_per_page =
            2 * config.num_kv_heads * PAGE_SIZE * config.head_dim * config.dtype.size_bytes();

        let pages_per_seq = (max_seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
        let total_pages = pages_per_seq * batch_size * config.num_layers;

        let total_bytes = total_pages * bytes_per_page;
        if total_bytes > 256 * 1024 * 1024 * 1024 {
            return Err(InferenceError::OutOfMemory {
                requested: total_bytes,
                available: 0,
            });
        }

        let pages: Vec<Page> = (0..total_pages)
            .map(|_| Page {
                data: vec![0u8; bytes_per_page],
                used: 0,
            })
            .collect();

        let free_pages: Vec<usize> = (0..total_pages).rev().collect();

        let layer_tables = (0..config.num_layers)
            .map(|_| {
                (0..batch_size)
                    .map(|_| SeqPageTable {
                        pages: Vec::new(),
                        seq_len: 0,
                    })
                    .collect()
            })
            .collect();

        Ok(KvCache {
            pages,
            free_pages,
            layer_tables,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            dtype: config.dtype,
            bytes_per_page,
        })
    }

    /// Append tokens to a sequence's KV cache for a given layer.
    /// Returns the physical page index and offset within the page.
    pub fn append(
        &mut self,
        layer: usize,
        seq_idx: usize,
        num_new_tokens: usize,
    ) -> Result<Vec<(usize, usize)>, InferenceError> {
        let table = &mut self.layer_tables[layer][seq_idx];
        let mut positions = Vec::with_capacity(num_new_tokens);

        for _ in 0..num_new_tokens {
            let offset_in_page = table.seq_len % PAGE_SIZE;

            // Need a new page?
            if offset_in_page == 0 {
                let page_id = self.free_pages.pop().ok_or_else(|| {
                    InferenceError::OutOfMemory {
                        requested: self.bytes_per_page,
                        available: 0,
                    }
                })?;
                table.pages.push(page_id);
            }

            let page_id = *table.pages.last().unwrap();
            positions.push((page_id, offset_in_page));
            table.seq_len += 1;
            self.pages[page_id].used = offset_in_page + 1;
        }

        Ok(positions)
    }

    /// Get a raw pointer to a page's data.
    #[inline]
    pub fn page_ptr(&self, page_id: usize) -> *const u8 {
        self.pages[page_id].data.as_ptr()
    }

    /// Get a mutable raw pointer to a page's data.
    #[inline]
    pub fn page_mut_ptr(&mut self, page_id: usize) -> *mut u8 {
        self.pages[page_id].data.as_mut_ptr()
    }

    /// Current sequence length for a given layer and sequence.
    pub fn seq_len(&self, layer: usize, seq_idx: usize) -> usize {
        self.layer_tables[layer][seq_idx].seq_len
    }

    /// Page indices for a given layer and sequence.
    pub fn seq_pages(&self, layer: usize, seq_idx: usize) -> &[usize] {
        &self.layer_tables[layer][seq_idx].pages
    }

    /// Reset a sequence (free its pages back to the pool).
    pub fn reset_seq(&mut self, seq_idx: usize) {
        for layer in 0..self.num_layers {
            let table = &mut self.layer_tables[layer][seq_idx];
            for &page_id in &table.pages {
                self.pages[page_id].used = 0;
                self.free_pages.push(page_id);
            }
            table.pages.clear();
            table.seq_len = 0;
        }
    }

    /// Total number of physical pages.
    pub fn total_pages(&self) -> usize {
        self.pages.len()
    }

    /// Number of free pages remaining.
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Bytes per page.
    pub fn bytes_per_page(&self) -> usize {
        self.bytes_per_page
    }

    /// Swap out pages to an external buffer (for offloading).
    pub fn swap_out(&self, page_ids: &[usize], buf: &mut [u8]) -> Result<(), InferenceError> {
        let total = page_ids.len() * self.bytes_per_page;
        if buf.len() < total {
            return Err(InferenceError::RuntimeError(format!(
                "swap buffer too small: need {total}, got {}",
                buf.len()
            )));
        }
        for (i, &pid) in page_ids.iter().enumerate() {
            let offset = i * self.bytes_per_page;
            buf[offset..offset + self.bytes_per_page]
                .copy_from_slice(&self.pages[pid].data);
        }
        Ok(())
    }

    /// Swap in pages from an external buffer.
    pub fn swap_in(&mut self, page_ids: &[usize], buf: &[u8]) -> Result<(), InferenceError> {
        let total = page_ids.len() * self.bytes_per_page;
        if buf.len() < total {
            return Err(InferenceError::RuntimeError(format!(
                "swap buffer too small: need {total}, got {}",
                buf.len()
            )));
        }
        for (i, &pid) in page_ids.iter().enumerate() {
            let offset = i * self.bytes_per_page;
            self.pages[pid]
                .data
                .copy_from_slice(&buf[offset..offset + self.bytes_per_page]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            arch: crate::inference::types::ModelArch::Llama,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 2,
            vocab_size: 100,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    #[test]
    fn test_kv_cache_alloc() {
        let cfg = tiny_config();
        let cache = KvCache::new(&cfg, 2, 64).unwrap();
        assert!(cache.total_pages() > 0);
        assert_eq!(cache.free_page_count(), cache.total_pages());
    }

    #[test]
    fn test_kv_cache_append() {
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        let positions = cache.append(0, 0, 5).unwrap();
        assert_eq!(positions.len(), 5);
        assert_eq!(cache.seq_len(0, 0), 5);
        // Should have allocated 1 page (5 < PAGE_SIZE=16)
        assert_eq!(cache.free_page_count(), initial_free - 1);
    }

    #[test]
    fn test_kv_cache_reset() {
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        cache.append(0, 0, 20).unwrap(); // spans 2 pages
        assert!(cache.free_page_count() < initial_free);

        cache.reset_seq(0);
        assert_eq!(cache.seq_len(0, 0), 0);
        // Pages returned to pool (for both layers)
        // Note: reset_seq frees pages for all layers
    }
}
