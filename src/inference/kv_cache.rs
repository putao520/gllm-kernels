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
    /// Physical page indices for this sequence (in logical order)
    pages: Vec<usize>,
    /// Current sequence length
    seq_len: usize,
}

/// Metadata for a swapped-out sequence, needed to restore it via [`KvCache::swap_in_seq`].
pub struct SwapHandle {
    /// Per-layer: (seq_len, num_pages)
    layer_info: Vec<(usize, usize)>,
    /// Total bytes stored in the swap buffer
    total_bytes: usize,
}

impl SwapHandle {
    /// Total bytes stored in the swap buffer.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
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

    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Compute dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of tokens currently stored in a physical page.
    pub fn page_used(&self, page_id: usize) -> usize {
        self.pages[page_id].used
    }

    // ── Offset helpers ──────────────────────────────────────────────

    /// Byte offset within a page for a given (kv_index, head, token_in_page) coordinate.
    ///
    /// Page layout: `[2(K+V), num_kv_heads, PAGE_SIZE, head_dim]`.
    /// - `kv_index`: 0 = K, 1 = V
    /// - `head`: KV head index
    /// - `token_in_page`: token position within the page (0..PAGE_SIZE)
    #[inline]
    fn page_offset(&self, kv_index: usize, head: usize, token_in_page: usize) -> usize {
        let elem = self.dtype.size_bytes();
        let kv_plane = self.num_kv_heads * PAGE_SIZE * self.head_dim;
        (kv_index * kv_plane + head * PAGE_SIZE * self.head_dim + token_in_page * self.head_dim)
            * elem
    }

    /// Bytes per token per head (`head_dim * dtype_size`).
    #[inline]
    fn head_bytes(&self) -> usize {
        self.head_dim * self.dtype.size_bytes()
    }

    /// Bytes per token across all KV heads (`num_kv_heads * head_dim * dtype_size`).
    #[inline]
    fn token_kv_bytes(&self) -> usize {
        self.num_kv_heads * self.head_bytes()
    }

    // ── Write / Read KV data ────────────────────────────────────────

    /// Write K and V data for newly appended tokens.
    ///
    /// `positions` comes from [`KvCache::append`]. `k_data` and `v_data` are
    /// contiguous byte slices with layout `[num_tokens, num_kv_heads, head_dim]`
    /// in the cache's compute dtype.
    pub fn write_kv(
        &mut self,
        _layer: usize,
        _seq_idx: usize,
        positions: &[(usize, usize)],
        k_data: &[u8],
        v_data: &[u8],
    ) -> Result<(), InferenceError> {
        let expected = positions.len() * self.token_kv_bytes();
        if k_data.len() != expected || v_data.len() != expected {
            return Err(InferenceError::ShapeMismatch {
                expected: format!("{expected} bytes for {} tokens", positions.len()),
                got: format!("k={}, v={}", k_data.len(), v_data.len()),
            });
        }

        let hb = self.head_bytes();
        for (tok, &(page_id, off)) in positions.iter().enumerate() {
            for h in 0..self.num_kv_heads {
                let src = (tok * self.num_kv_heads + h) * hb;

                let k_dst = self.page_offset(0, h, off);
                self.pages[page_id].data[k_dst..k_dst + hb]
                    .copy_from_slice(&k_data[src..src + hb]);

                let v_dst = self.page_offset(1, h, off);
                self.pages[page_id].data[v_dst..v_dst + hb]
                    .copy_from_slice(&v_data[src..src + hb]);
            }
        }
        Ok(())
    }

    /// Read all K and V data for a sequence.
    ///
    /// Outputs are contiguous byte slices with layout
    /// `[seq_len, num_kv_heads, head_dim]` in the cache's compute dtype.
    /// Returns the current sequence length.
    pub fn read_kv(
        &self,
        layer: usize,
        seq_idx: usize,
        k_out: &mut [u8],
        v_out: &mut [u8],
    ) -> Result<usize, InferenceError> {
        let table = &self.layer_tables[layer][seq_idx];
        let seq_len = table.seq_len;
        if seq_len == 0 {
            return Ok(0);
        }

        let expected = seq_len * self.token_kv_bytes();
        if k_out.len() < expected || v_out.len() < expected {
            return Err(InferenceError::ShapeMismatch {
                expected: format!("{expected} bytes for seq_len={seq_len}"),
                got: format!("k_out={}, v_out={}", k_out.len(), v_out.len()),
            });
        }

        let hb = self.head_bytes();
        let mut global_tok = 0usize;

        for (pi, &page_id) in table.pages.iter().enumerate() {
            let tokens_in_page = if pi < table.pages.len() - 1 {
                PAGE_SIZE
            } else {
                let rem = seq_len % PAGE_SIZE;
                if rem == 0 { PAGE_SIZE } else { rem }
            };

            for t in 0..tokens_in_page {
                for h in 0..self.num_kv_heads {
                    let dst = (global_tok * self.num_kv_heads + h) * hb;

                    let k_src = self.page_offset(0, h, t);
                    k_out[dst..dst + hb]
                        .copy_from_slice(&self.pages[page_id].data[k_src..k_src + hb]);

                    let v_src = self.page_offset(1, h, t);
                    v_out[dst..dst + hb]
                        .copy_from_slice(&self.pages[page_id].data[v_src..v_src + hb]);
                }
                global_tok += 1;
            }
        }
        Ok(seq_len)
    }

    // ── Low-level page swap (arbitrary page list) ───────────────────

    /// Copy page data to an external buffer (low-level, does not free pages).
    pub fn swap_out(&self, page_ids: &[usize], buf: &mut [u8]) -> Result<(), InferenceError> {
        let total = page_ids.len() * self.bytes_per_page;
        if buf.len() < total {
            return Err(InferenceError::RuntimeError(format!(
                "swap buffer too small: need {total}, got {}",
                buf.len()
            )));
        }
        for (i, &pid) in page_ids.iter().enumerate() {
            let off = i * self.bytes_per_page;
            buf[off..off + self.bytes_per_page]
                .copy_from_slice(&self.pages[pid].data);
        }
        Ok(())
    }

    /// Restore page data from an external buffer (low-level, pages must already exist).
    pub fn swap_in(&mut self, page_ids: &[usize], buf: &[u8]) -> Result<(), InferenceError> {
        let total = page_ids.len() * self.bytes_per_page;
        if buf.len() < total {
            return Err(InferenceError::RuntimeError(format!(
                "swap buffer too small: need {total}, got {}",
                buf.len()
            )));
        }
        for (i, &pid) in page_ids.iter().enumerate() {
            let off = i * self.bytes_per_page;
            self.pages[pid]
                .data
                .copy_from_slice(&buf[off..off + self.bytes_per_page]);
        }
        Ok(())
    }

    // ── Sequence-level swap (full lifecycle) ────────────────────────

    /// Swap out an entire sequence across all layers: copy data to `buf`, free
    /// physical pages back to the pool, and clear the page tables.
    ///
    /// Returns a [`SwapHandle`] that must be passed to [`KvCache::swap_in_seq`]
    /// to restore the sequence later.
    pub fn swap_out_seq(
        &mut self,
        seq_idx: usize,
        buf: &mut Vec<u8>,
    ) -> Result<SwapHandle, InferenceError> {
        let mut layer_info = Vec::with_capacity(self.num_layers);
        let mut total_pages = 0usize;

        for layer in 0..self.num_layers {
            let t = &self.layer_tables[layer][seq_idx];
            layer_info.push((t.seq_len, t.pages.len()));
            total_pages += t.pages.len();
        }

        let total_bytes = total_pages * self.bytes_per_page;
        buf.resize(total_bytes, 0);

        let mut cursor = 0usize;
        for layer in 0..self.num_layers {
            let table = &mut self.layer_tables[layer][seq_idx];
            for &page_id in &table.pages {
                buf[cursor..cursor + self.bytes_per_page]
                    .copy_from_slice(&self.pages[page_id].data);
                // Zero out and return page to pool
                self.pages[page_id].data.iter_mut().for_each(|b| *b = 0);
                self.pages[page_id].used = 0;
                self.free_pages.push(page_id);
                cursor += self.bytes_per_page;
            }
            table.pages.clear();
            table.seq_len = 0;
        }

        Ok(SwapHandle {
            layer_info,
            total_bytes,
        })
    }

    /// Swap in a previously swapped-out sequence: allocate fresh physical pages,
    /// restore data from `buf`, and rebuild the page tables.
    pub fn swap_in_seq(
        &mut self,
        seq_idx: usize,
        handle: &SwapHandle,
        buf: &[u8],
    ) -> Result<(), InferenceError> {
        if buf.len() < handle.total_bytes {
            return Err(InferenceError::RuntimeError(format!(
                "swap buffer too small: need {}, got {}",
                handle.total_bytes,
                buf.len()
            )));
        }

        let pages_needed: usize = handle.layer_info.iter().map(|&(_, n)| n).sum();
        if self.free_pages.len() < pages_needed {
            return Err(InferenceError::OutOfMemory {
                requested: pages_needed * self.bytes_per_page,
                available: self.free_pages.len() * self.bytes_per_page,
            });
        }

        let mut cursor = 0usize;
        for (layer, &(seq_len, num_pages)) in handle.layer_info.iter().enumerate() {
            let table = &mut self.layer_tables[layer][seq_idx];
            table.seq_len = seq_len;
            table.pages.clear();
            table.pages.reserve(num_pages);

            for pi in 0..num_pages {
                let page_id = self.free_pages.pop().unwrap(); // availability checked above

                self.pages[page_id]
                    .data
                    .copy_from_slice(&buf[cursor..cursor + self.bytes_per_page]);

                // Restore used count
                let is_last = pi == num_pages - 1;
                self.pages[page_id].used = if is_last {
                    let rem = seq_len % PAGE_SIZE;
                    if rem == 0 { PAGE_SIZE } else { rem }
                } else {
                    PAGE_SIZE
                };

                table.pages.push(page_id);
                cursor += self.bytes_per_page;
            }
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

    // ── Helper: generate deterministic f32 KV data as bytes ─────────

    /// Build `[num_tokens, num_kv_heads, head_dim]` f32 data where each element
    /// is `base + token*1000 + head*100 + dim` so every value is unique and
    /// easy to verify.
    fn make_kv_bytes(
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        base: f32,
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(num_tokens * num_kv_heads * head_dim * 4);
        for t in 0..num_tokens {
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let val = base + t as f32 * 1000.0 + h as f32 * 100.0 + d as f32;
                    out.extend_from_slice(&val.to_ne_bytes());
                }
            }
        }
        out
    }

    /// Read a single f32 from a byte slice at the given element index.
    fn read_f32(buf: &[u8], elem_idx: usize) -> f32 {
        let off = elem_idx * 4;
        f32::from_ne_bytes(buf[off..off + 4].try_into().unwrap())
    }

    // ── New tests ───────────────────────────────────────────────────

    #[test]
    fn test_write_read_kv_single_page() {
        let cfg = tiny_config(); // 4 kv_heads, head_dim=16, 2 layers, F32
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Append 5 tokens to layer 0, seq 0 (fits in 1 page)
        let positions = cache.append(0, 0, 5).unwrap();
        assert_eq!(positions.len(), 5);

        let k_data = make_kv_bytes(5, 4, 16, 0.0);
        let v_data = make_kv_bytes(5, 4, 16, 50000.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        // Read back
        let token_kv_bytes = 4 * 16 * 4; // num_kv_heads * head_dim * sizeof(f32)
        let mut k_out = vec![0u8; 5 * token_kv_bytes];
        let mut v_out = vec![0u8; 5 * token_kv_bytes];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(len, 5);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);
    }

    #[test]
    fn test_write_read_kv_multi_page() {
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Append 20 tokens → spans 2 pages (PAGE_SIZE=16)
        let positions = cache.append(0, 0, 20).unwrap();
        assert_eq!(positions.len(), 20);
        // First 16 tokens in page 0, next 4 in page 1
        assert_eq!(positions[15].1, 15); // last slot of first page
        assert_eq!(positions[16].1, 0);  // first slot of second page

        let k_data = make_kv_bytes(20, 4, 16, 0.0);
        let v_data = make_kv_bytes(20, 4, 16, 90000.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0u8; 20 * tkb];
        let mut v_out = vec![0u8; 20 * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(len, 20);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);
    }

    #[test]
    fn test_read_kv_empty_seq() {
        let cfg = tiny_config();
        let cache = KvCache::new(&cfg, 1, 64).unwrap();

        let mut k_out = vec![0u8; 0];
        let mut v_out = vec![0u8; 0];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn test_swap_out_in_seq_roundtrip() {
        let cfg = tiny_config(); // 2 layers
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Populate both layers with 10 tokens each
        for layer in 0..2 {
            let pos = cache.append(layer, 0, 10).unwrap();
            let k = make_kv_bytes(10, 4, 16, layer as f32 * 100000.0);
            let v = make_kv_bytes(10, 4, 16, layer as f32 * 100000.0 + 50000.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }
        // 2 pages used (1 per layer)
        assert_eq!(cache.free_page_count(), initial_free - 2);

        // Swap out
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();
        assert_eq!(handle.total_bytes(), 2 * cache.bytes_per_page());

        // Pages freed
        assert_eq!(cache.free_page_count(), initial_free);
        assert_eq!(cache.seq_len(0, 0), 0);
        assert_eq!(cache.seq_len(1, 0), 0);

        // Swap in
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();
        assert_eq!(cache.seq_len(0, 0), 10);
        assert_eq!(cache.seq_len(1, 0), 10);
        assert_eq!(cache.free_page_count(), initial_free - 2);

        // Verify data integrity for both layers
        let tkb = 4 * 16 * 4;
        for layer in 0..2 {
            let mut k_out = vec![0u8; 10 * tkb];
            let mut v_out = vec![0u8; 10 * tkb];
            cache.read_kv(layer, 0, &mut k_out, &mut v_out).unwrap();

            let k_expected = make_kv_bytes(10, 4, 16, layer as f32 * 100000.0);
            let v_expected = make_kv_bytes(10, 4, 16, layer as f32 * 100000.0 + 50000.0);
            assert_eq!(k_out, k_expected, "K mismatch on layer {layer}");
            assert_eq!(v_out, v_expected, "V mismatch on layer {layer}");
        }
    }

    #[test]
    fn test_swap_out_in_long_seq_multi_page() {
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // 48 tokens → 3 pages per layer, 2 layers → 6 pages total
        for layer in 0..2 {
            let pos = cache.append(layer, 0, 48).unwrap();
            let k = make_kv_bytes(48, 4, 16, layer as f32 * 200000.0);
            let v = make_kv_bytes(48, 4, 16, layer as f32 * 200000.0 + 100000.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }

        let free_before_swap = cache.free_page_count();

        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();
        // 6 pages freed
        assert_eq!(cache.free_page_count(), free_before_swap + 6);

        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();
        assert_eq!(cache.free_page_count(), free_before_swap);

        // Verify data
        let tkb = 4 * 16 * 4;
        for layer in 0..2 {
            let mut k_out = vec![0u8; 48 * tkb];
            let mut v_out = vec![0u8; 48 * tkb];
            cache.read_kv(layer, 0, &mut k_out, &mut v_out).unwrap();

            let k_expected = make_kv_bytes(48, 4, 16, layer as f32 * 200000.0);
            let v_expected = make_kv_bytes(48, 4, 16, layer as f32 * 200000.0 + 100000.0);
            assert_eq!(k_out, k_expected, "K mismatch layer {layer}");
            assert_eq!(v_out, v_expected, "V mismatch layer {layer}");
        }
    }

    #[test]
    fn test_append_after_swap_in() {
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Phase 1: append 10 tokens
        let pos1 = cache.append(0, 0, 10).unwrap();
        let k1 = make_kv_bytes(10, 4, 16, 0.0);
        let v1 = make_kv_bytes(10, 4, 16, 50000.0);
        cache.write_kv(0, 0, &pos1, &k1, &v1).unwrap();

        // Swap out and back in
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();
        assert_eq!(cache.seq_len(0, 0), 10);

        // Phase 2: append 6 more tokens (still within the same page, slots 10..16)
        let pos2 = cache.append(0, 0, 6).unwrap();
        assert_eq!(pos2.len(), 6);
        assert_eq!(pos2[0].1, 10); // continues from offset 10

        let k2 = make_kv_bytes(6, 4, 16, 10000.0);
        let v2 = make_kv_bytes(6, 4, 16, 60000.0);
        cache.write_kv(0, 0, &pos2, &k2, &v2).unwrap();
        assert_eq!(cache.seq_len(0, 0), 16);

        // Read all 16 tokens and verify both phases
        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0u8; 16 * tkb];
        let mut v_out = vec![0u8; 16 * tkb];
        cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();

        // First 10 tokens from phase 1
        assert_eq!(&k_out[..10 * tkb], &k1[..]);
        assert_eq!(&v_out[..10 * tkb], &v1[..]);
        // Next 6 tokens from phase 2
        assert_eq!(&k_out[10 * tkb..16 * tkb], &k2[..]);
        assert_eq!(&v_out[10 * tkb..16 * tkb], &v2[..]);
    }

    #[test]
    fn test_swap_in_insufficient_pages() {
        let cfg = tiny_config();
        // Only 1 seq, max_seq_len=32 → limited pages
        let mut cache = KvCache::new(&cfg, 1, 32).unwrap();

        // Fill layer 0 with 32 tokens (uses all pages for layer 0)
        let pos = cache.append(0, 0, 32).unwrap();
        let k = make_kv_bytes(32, 4, 16, 0.0);
        let v = make_kv_bytes(32, 4, 16, 50000.0);
        cache.write_kv(0, 0, &pos, &k, &v).unwrap();

        // Also fill layer 1
        let pos1 = cache.append(1, 0, 32).unwrap();
        cache.write_kv(1, 0, &pos1, &k, &v).unwrap();

        // Swap out seq 0
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Now exhaust all free pages by creating a second cache scenario:
        // Actually, after swap_out all pages are free. Let's consume them.
        // Re-append to seq 0 to consume pages, then try swap_in which needs
        // pages that are no longer free.
        for layer in 0..2 {
            cache.append(layer, 0, 32).unwrap();
        }
        // No free pages left
        assert_eq!(cache.free_page_count(), 0);

        // swap_in should fail with OutOfMemory
        let result = cache.swap_in_seq(0, &handle, &swap_buf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::OutOfMemory { .. } => {} // expected
            other => panic!("expected OutOfMemory, got: {other:?}"),
        }
    }
}
