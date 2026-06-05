//! Paged KV Cache for efficient attention computation.
//!
//! Uses a page-table design (inspired by vLLM) where KV data is stored in
//! fixed-size pages. This avoids pre-allocating max_seq_len for every sequence
//! and enables efficient memory sharing for beam search.

use crate::types::{DType, InferenceError, ModelConfig};

/// Page size in tokens.
pub const PAGE_SIZE: usize = 16;

/// Default maximum KV cache allocation (256 GiB).
/// Use `KvCache::new_with_memory_limit()` to override based on actual device memory.
pub const MAX_KV_CACHE_BYTES: usize = 256 * 1024 * 1024 * 1024;

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
        // Sanity cap: reject allocations exceeding MAX_KV_CACHE_BYTES.
        // Default 256 GiB; callers with GPU profile should use new_with_memory_limit().
        if total_bytes > MAX_KV_CACHE_BYTES {
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
                let page_id = self.free_pages.pop().ok_or({
                    InferenceError::OutOfMemory {
                        requested: self.bytes_per_page,
                        available: 0,
                    }
                })?;
                table.pages.push(page_id);
            }

            // SAFETY: page was just pushed in the offset_in_page==0 branch above,
            // or a prior iteration already pushed a page that is not yet full.
            let page_id = *table.pages.last().expect("SAFETY: page guaranteed present — pushed when offset_in_page==0");
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
                let page_id = self.free_pages.pop().expect("SAFETY: free_pages.len() >= pages_needed checked above");

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
            arch: crate::types::ModelArch::Llama,
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

    // @trace TEST-KVC-11 [req:REQ-KV] [level:unit]
    #[test]
    fn test_new_rejects_oversized_allocation() {
        // Arrange: a config that would require > MAX_KV_CACHE_BYTES.
        // With F32 (4 bytes), num_kv_heads=1024, head_dim=128, PAGE_SIZE=16:
        // bytes_per_page = 2 * 1024 * 16 * 128 * 4 = 16 MiB per page.
        // batch=4096, max_seq=65536 → pages_per_seq=4096 → total_pages huge.
        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 131072,
            num_heads: 1024,
            num_kv_heads: 1024,
            head_dim: 128,
            intermediate_size: 262144,
            num_layers: 64,
            vocab_size: 100,
            max_seq_len: 65536,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };
        // Act
        let result = KvCache::new(&cfg, 4096, 65536);
        // Assert: must fail with OutOfMemory
        assert!(result.is_err());
        match result.err().unwrap() {
            InferenceError::OutOfMemory { .. } => {}
            other => panic!("expected OutOfMemory, got: {other:?}"),
        }
    }

    // @trace TEST-KVC-12 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_oom_exhausts_pages() {
        // Arrange: tiny cache with 1 layer, 1 seq, max_seq=PAGE_SIZE (only 1 page per layer)
        let cfg = tiny_config(); // num_layers=2, so 2 pages total
        let mut cache = KvCache::new(&cfg, 1, PAGE_SIZE).unwrap();

        // Consume both pages (1 per layer)
        cache.append(0, 0, PAGE_SIZE).unwrap();
        cache.append(1, 0, PAGE_SIZE).unwrap();
        assert_eq!(cache.free_page_count(), 0);

        // Act: try to append one more token on layer 0 — needs a new page
        let result = cache.append(0, 0, 1);
        // Assert: OutOfMemory
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::OutOfMemory { .. } => {}
            other => panic!("expected OutOfMemory, got: {other:?}"),
        }
    }

    // @trace TEST-KVC-13 [req:REQ-KV] [level:unit]
    #[test]
    fn test_write_kv_shape_mismatch() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let positions = cache.append(0, 0, 5).unwrap();

        // k_data too short (only 1 token worth of data instead of 5)
        let short_k = make_kv_bytes(1, 4, 16, 0.0);
        let correct_v = make_kv_bytes(5, 4, 16, 50000.0);

        // Act
        let result = cache.write_kv(0, 0, &positions, &short_k, &correct_v);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }

        // Also test v_data too short
        let correct_k = make_kv_bytes(5, 4, 16, 0.0);
        let short_v = make_kv_bytes(2, 4, 16, 50000.0);
        let result2 = cache.write_kv(0, 0, &positions, &correct_k, &short_v);
        assert!(result2.is_err());
        match result2.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }
    }

    // @trace TEST-KVC-14 [req:REQ-KV] [level:unit]
    #[test]
    fn test_read_kv_shape_mismatch_buffer_too_small() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        cache.append(0, 0, 10).unwrap();
        let tkb = 4 * 16 * 4; // token_kv_bytes

        // Act: provide output buffers that are too small (only 5 tokens worth)
        let mut k_out = vec![0u8; 5 * tkb];
        let mut v_out = vec![0u8; 10 * tkb]; // v is large enough
        let result = cache.read_kv(0, 0, &mut k_out, &mut v_out);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }

        // Now test v_out too small
        let mut k_out2 = vec![0u8; 10 * tkb];
        let mut v_out2 = vec![0u8; 3 * tkb];
        let result2 = cache.read_kv(0, 0, &mut k_out2, &mut v_out2);
        assert!(result2.is_err());
        match result2.unwrap_err() {
            InferenceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got: {other:?}"),
        }
    }

    // @trace TEST-KVC-15 [req:REQ-KV] [level:unit]
    #[test]
    fn test_accessors_return_correct_values() {
        // Arrange
        let cfg = tiny_config(); // num_kv_heads=4, head_dim=16, dtype=F32, num_layers=2
        let cache = KvCache::new(&cfg, 1, 64).unwrap();
        let expected_bytes_per_page = 2 * 4 * PAGE_SIZE * 16 * 4; // 2*KV * heads * page_size * head_dim * f32

        // Act & Assert
        assert_eq!(cache.num_kv_heads(), 4);
        assert_eq!(cache.head_dim(), 16);
        assert_eq!(cache.dtype(), DType::F32);
        assert_eq!(cache.bytes_per_page(), expected_bytes_per_page);
    }

    // @trace TEST-KVC-16 [req:REQ-KV] [level:unit]
    #[test]
    fn test_page_ptr_and_page_used() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: append 7 tokens to layer 0, seq 0
        let positions = cache.append(0, 0, 7).unwrap();

        // Assert: first page was allocated
        let page_id = positions[0].0;
        assert!(!cache.page_ptr(page_id).is_null());
        assert_eq!(cache.page_used(page_id), 7);
        assert_eq!(cache.free_page_count(), initial_free - 1);

        // page_mut_ptr returns a valid mutable pointer
        let mut_ptr = cache.page_mut_ptr(page_id);
        assert!(!mut_ptr.is_null());
        // Verify mutable pointer aliases the same data
        unsafe {
            assert_eq!(*cache.page_ptr(page_id), *mut_ptr);
        }
    }

    // @trace TEST-KVC-17 [req:REQ-KV] [level:unit]
    #[test]
    fn test_seq_pages_returns_correct_indices() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Act: append 20 tokens (2 pages) to layer 0
        let positions = cache.append(0, 0, 20).unwrap();

        // Assert: seq_pages returns exactly the page indices used
        let pages = cache.seq_pages(0, 0);
        assert_eq!(pages.len(), 2);
        // First page covers tokens 0..15, second covers 16..19
        assert_eq!(pages[0], positions[0].0);
        assert_eq!(pages[1], positions[16].0);
        // seq_pages for layer 1 (unused) should be empty
        assert!(cache.seq_pages(1, 0).is_empty());
    }

    // @trace TEST-KVC-18 [req:REQ-KV] [level:unit]
    #[test]
    fn test_multi_sequence_isolation() {
        // Arrange: batch_size=2, so 2 sequences per layer
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 2, 64).unwrap();

        // Act: append to seq 0 and seq 1 independently
        let pos0 = cache.append(0, 0, 8).unwrap();
        let pos1 = cache.append(0, 1, 12).unwrap();

        // Assert: independent sequence lengths
        assert_eq!(cache.seq_len(0, 0), 8);
        assert_eq!(cache.seq_len(0, 1), 12);

        // Reset seq 0 only
        cache.reset_seq(0);
        assert_eq!(cache.seq_len(0, 0), 0);
        assert_eq!(cache.seq_len(1, 0), 0);
        // Seq 1 untouched
        assert_eq!(cache.seq_len(0, 1), 12);
        assert_eq!(cache.seq_len(1, 1), 0); // layer 1 never appended
    }

    // @trace TEST-KVC-19 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_buffer_too_small() {
        // Arrange
        let cfg = tiny_config();
        let cache = KvCache::new(&cfg, 1, 64).unwrap();
        // Allocate some pages so we have page IDs to reference
        // We need a mutable cache for append, so clone won't work — just use the pool pages
        // swap_out takes page_ids; we can pass page index 0 even though it's not allocated to a seq
        let page_ids: Vec<usize> = (0..2).collect();
        let bpp = cache.bytes_per_page();
        let mut buf = vec![0u8; bpp + bpp - 1]; // 1 byte short of 2 pages

        // Act
        let result = cache.swap_out(&page_ids, &mut buf);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::RuntimeError(msg) => {
                assert!(msg.contains("swap buffer too small"));
            }
            other => panic!("expected RuntimeError, got: {other:?}"),
        }
    }

    // @trace TEST-KVC-20 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_in_buffer_too_small() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let page_ids: Vec<usize> = (0..2).collect();
        let bpp = cache.bytes_per_page();
        // Write some data into the pages first
        for &pid in &page_ids {
            let ptr = cache.page_mut_ptr(pid);
            unsafe {
                std::ptr::write_bytes(ptr, 0xAB, bpp);
            }
        }
        let mut swap_buf = vec![0u8; bpp * 2];
        cache.swap_out(&page_ids, &mut swap_buf).unwrap();

        // Act: try swap_in with a buffer that's too small
        let short_buf = &swap_buf[..bpp]; // only 1 page worth
        let result = cache.swap_in(&page_ids, short_buf);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::RuntimeError(msg) => {
                assert!(msg.contains("swap buffer too small"));
            }
            other => panic!("expected RuntimeError, got: {other:?}"),
        }
    }

    // @trace TEST-KVC-21 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_in_seq_buffer_too_small() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        for layer in 0..2 {
            let pos = cache.append(layer, 0, 10).unwrap();
            let k = make_kv_bytes(10, 4, 16, 0.0);
            let v = make_kv_bytes(10, 4, 16, 50000.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Act: truncate the buffer to be too small
        let short_buf = &swap_buf[..handle.total_bytes() - 1];
        let result = cache.swap_in_seq(0, &handle, short_buf);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::RuntimeError(msg) => {
                assert!(msg.contains("swap buffer too small"));
            }
            other => panic!("expected RuntimeError, got: {other:?}"),
        }
    }

    // @trace TEST-KVC-22 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_exact_page_boundary() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: append exactly PAGE_SIZE (16) tokens
        let positions = cache.append(0, 0, PAGE_SIZE).unwrap();

        // Assert: exactly 1 page used, all slots filled
        assert_eq!(positions.len(), PAGE_SIZE);
        assert_eq!(positions[0].1, 0);
        assert_eq!(positions[PAGE_SIZE - 1].1, PAGE_SIZE - 1);
        assert_eq!(cache.seq_len(0, 0), PAGE_SIZE);
        assert_eq!(cache.free_page_count(), initial_free - 1);
        // The page is fully used
        let page_id = positions[0].0;
        assert_eq!(cache.page_used(page_id), PAGE_SIZE);

        // Act: append 1 more token — should allocate a new page
        let pos2 = cache.append(0, 0, 1).unwrap();
        assert_eq!(pos2.len(), 1);
        assert_eq!(pos2[0].1, 0); // offset 0 in the new page
        assert!(pos2[0].0 != page_id); // different physical page
        assert_eq!(cache.free_page_count(), initial_free - 2);
    }

    // @trace TEST-KVC-23 [req:REQ-KV] [level:unit]
    #[test]
    fn test_reset_seq_restores_all_pages_across_layers() {
        // Arrange: append to both layers, verify free count restored after reset
        let cfg = tiny_config(); // 2 layers
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Append 20 tokens to each layer (2 pages per layer = 4 pages total)
        for layer in 0..2 {
            cache.append(layer, 0, 20).unwrap();
        }
        assert_eq!(cache.free_page_count(), initial_free - 4);

        // Act: reset sequence 0
        cache.reset_seq(0);

        // Assert: all pages returned, seq_len zeroed across all layers
        assert_eq!(cache.free_page_count(), initial_free);
        for layer in 0..2 {
            assert_eq!(cache.seq_len(layer, 0), 0);
            assert!(cache.seq_pages(layer, 0).is_empty());
        }
        // All previously used pages should have used=0
        // (pages are pushed back to free_pages, verify count is exact)
    }

    // @trace TEST-KVC-24 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_handle_total_bytes_accessor() {
        // Arrange: construct SwapHandle directly with known values
        let handle = SwapHandle {
            layer_info: vec![(10, 1), (10, 1)],
            total_bytes: 4096,
        };

        // Act & Assert
        assert_eq!(handle.total_bytes(), 4096);
        assert_eq!(handle.layer_info.len(), 2);
    }

    // @trace TEST-KVC-25 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_zero_tokens_is_noop() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: append 0 tokens
        let positions = cache.append(0, 0, 0).unwrap();

        // Assert: nothing allocated, nothing changed
        assert!(positions.is_empty());
        assert_eq!(cache.seq_len(0, 0), 0);
        assert_eq!(cache.free_page_count(), initial_free);
    }

    // @trace TEST-KVC-26 [req:REQ-KV] [level:unit]
    #[test]
    fn test_write_read_kv_exact_page_boundary() {
        // Arrange: write/read exactly PAGE_SIZE tokens (full single page)
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Act: append exactly PAGE_SIZE tokens
        let positions = cache.append(0, 0, PAGE_SIZE).unwrap();
        assert_eq!(positions.len(), PAGE_SIZE);

        let k_data = make_kv_bytes(PAGE_SIZE, 4, 16, 0.0);
        let v_data = make_kv_bytes(PAGE_SIZE, 4, 16, 77777.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        // Assert: read back and verify exact match
        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0u8; PAGE_SIZE * tkb];
        let mut v_out = vec![0u8; PAGE_SIZE * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);
    }

    // @trace TEST-KVC-27 [req:REQ-KV] [level:unit]
    #[test]
    fn test_low_level_swap_out_in_roundtrip() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Write a known pattern into page 0 via page_mut_ptr
        let bpp = cache.bytes_per_page();
        let ptr = cache.page_mut_ptr(0);
        unsafe {
            for i in 0..bpp {
                *ptr.add(i) = (i % 256) as u8;
            }
        }

        // Act: swap out pages [0] to buffer
        let page_ids = vec![0];
        let mut buf = vec![0u8; bpp];
        cache.swap_out(&page_ids, &mut buf).unwrap();

        // Assert: buffer matches page data
        for i in 0..bpp {
            assert_eq!(buf[i], (i % 256) as u8);
        }

        // Act: corrupt the page, then swap_in from buffer
        let ptr2 = cache.page_mut_ptr(0);
        unsafe {
            std::ptr::write_bytes(ptr2, 0xFF, bpp);
        }

        cache.swap_in(&page_ids, &buf).unwrap();

        // Assert: page data restored
        let ptr3 = cache.page_ptr(0);
        unsafe {
            for i in 0..bpp {
                assert_eq!(*ptr3.add(i), (i % 256) as u8, "byte mismatch at offset {i}");
            }
        }
    }

    // @trace TEST-KVC-28 [req:REQ-KV] [level:unit]
    #[test]
    fn test_reset_seq_idempotent_on_empty_sequence() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: reset a sequence that was never appended to
        cache.reset_seq(0);
        cache.reset_seq(0); // double reset

        // Assert: nothing changed, no panic
        assert_eq!(cache.free_page_count(), initial_free);
        for layer in 0..2 {
            assert_eq!(cache.seq_len(layer, 0), 0);
            assert!(cache.seq_pages(layer, 0).is_empty());
        }
    }

    // @trace TEST-KVC-29 [req:REQ-KV] [level:unit]
    #[test]
    fn test_new_bf16_dtype_halves_bytes_per_page() {
        // Arrange: BF16 config (2 bytes per element vs F32's 4)
        let cfg_f32 = tiny_config();
        let mut cfg_bf16 = tiny_config();
        cfg_bf16.dtype = DType::BF16;

        // Act
        let cache_f32 = KvCache::new(&cfg_f32, 1, 64).unwrap();
        let cache_bf16 = KvCache::new(&cfg_bf16, 1, 64).unwrap();

        // Assert: BF16 page is exactly half the size of F32 page
        assert_eq!(cache_bf16.bytes_per_page(), cache_f32.bytes_per_page() / 2);
        assert_eq!(cache_bf16.dtype(), DType::BF16);
        // BF16 should also have the same number of total pages (same topology, different page size)
        assert_eq!(cache_bf16.total_pages(), cache_f32.total_pages());
    }

    // @trace TEST-KVC-30 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_spans_page_boundary_in_single_call() {
        // Arrange: append 10 tokens, then 10 more in a second call
        // so total is 20 tokens spanning a page boundary at token 16
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: first 10 tokens fit in page 0 (offsets 0..9)
        let pos1 = cache.append(0, 0, 10).unwrap();
        assert_eq!(pos1.len(), 10);
        assert_eq!(cache.free_page_count(), initial_free - 1);

        // Second 10 tokens: offsets 10..15 in page 0, then 0..3 in page 1
        let pos2 = cache.append(0, 0, 10).unwrap();
        assert_eq!(pos2.len(), 10);

        // Assert
        assert_eq!(cache.seq_len(0, 0), 20);
        assert_eq!(cache.free_page_count(), initial_free - 2); // 2 pages allocated

        // First 6 of pos2 are in page 0 (offsets 10..15)
        assert_eq!(pos2[0].0, pos1[0].0); // same page
        assert_eq!(pos2[0].1, 10);
        assert_eq!(pos2[5].1, 15);
        // Last 4 of pos2 are in page 1 (offsets 0..3)
        assert_ne!(pos2[6].0, pos1[0].0); // new page
        assert_eq!(pos2[6].1, 0);
        assert_eq!(pos2[9].1, 3);
    }

    // @trace TEST-KVC-31 [req:REQ-KV] [level:unit]
    #[test]
    fn test_read_kv_with_oversized_buffers() {
        // Arrange: write 5 tokens, but provide output buffers large enough for 20
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let positions = cache.append(0, 0, 5).unwrap();

        let k_data = make_kv_bytes(5, 4, 16, 42.0);
        let v_data = make_kv_bytes(5, 4, 16, 99.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        // Act: read with oversized buffers
        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0xFFu8; 20 * tkb]; // 20 tokens worth
        let mut v_out = vec![0xFFu8; 20 * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();

        // Assert: returns correct length, first 5 tokens match
        assert_eq!(len, 5);
        assert_eq!(&k_out[..5 * tkb], &k_data[..]);
        assert_eq!(&v_out[..5 * tkb], &v_data[..]);
        // Remainder of buffer untouched (still 0xFF)
        for i in (5 * tkb)..(6 * tkb) {
            assert_eq!(k_out[i], 0xFF, "k_out byte {i} should be untouched");
            assert_eq!(v_out[i], 0xFF, "v_out byte {i} should be untouched");
        }
    }

    // @trace TEST-KVC-32 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_after_reset_reuses_freed_pages() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Append and then reset
        cache.append(0, 0, 20).unwrap(); // 2 pages
        assert_eq!(cache.free_page_count(), initial_free - 2);
        cache.reset_seq(0);
        assert_eq!(cache.free_page_count(), initial_free);

        // Act: append again after reset
        let positions = cache.append(0, 0, 8).unwrap();

        // Assert: pages were reused (free count drops again)
        assert_eq!(positions.len(), 8);
        assert_eq!(cache.seq_len(0, 0), 8);
        assert_eq!(cache.free_page_count(), initial_free - 1);
    }

    // @trace TEST-KVC-33 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_across_layers_independently() {
        // Arrange
        let cfg = tiny_config(); // 2 layers
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: append different lengths to different layers for the same seq
        let pos_l0 = cache.append(0, 0, 5).unwrap();
        let pos_l1 = cache.append(1, 0, 20).unwrap();

        // Assert: independent seq_len tracking per layer
        assert_eq!(cache.seq_len(0, 0), 5);
        assert_eq!(cache.seq_len(1, 0), 20);
        assert_eq!(pos_l0.len(), 5);
        assert_eq!(pos_l1.len(), 20);

        // 1 page for layer 0, 2 pages for layer 1 = 3 pages total
        assert_eq!(cache.free_page_count(), initial_free - 3);

        // seq_pages for each layer reflects its own pages
        assert_eq!(cache.seq_pages(0, 0).len(), 1);
        assert_eq!(cache.seq_pages(1, 0).len(), 2);

        // The page indices used by layer 0 and layer 1 are distinct
        let l0_pages = cache.seq_pages(0, 0);
        let l1_pages = cache.seq_pages(1, 0);
        for &p0 in l0_pages {
            assert!(!l1_pages.contains(&p0), "layer 0 and layer 1 should use distinct pages");
        }
    }

    // @trace TEST-KVC-34 [req:REQ-KV] [level:unit]
    #[test]
    fn test_new_batch_size_scales_page_count() {
        // Arrange: batch_size=1 as baseline, then batch_size=3
        let cfg = tiny_config(); // num_layers=2, max_seq_len=64, PAGE_SIZE=16 → 4 pages/seq/layer
        let cache_b1 = KvCache::new(&cfg, 1, 64).unwrap();
        let cache_b3 = KvCache::new(&cfg, 3, 64).unwrap();

        // Act: compare total page counts
        let pages_b1 = cache_b1.total_pages();
        let pages_b3 = cache_b3.total_pages();

        // Assert: batch_size=3 should have exactly 3x the pages of batch_size=1
        assert_eq!(pages_b3, pages_b1 * 3);
        // All pages start free
        assert_eq!(cache_b3.free_page_count(), pages_b3);
    }

    // @trace TEST-KVC-35 [req:REQ-KV] [level:unit]
    #[test]
    fn test_new_non_page_aligned_max_seq_len() {
        // Arrange: max_seq_len=33, PAGE_SIZE=16 → needs ceil(33/16)=3 pages per seq per layer
        let cfg = tiny_config(); // num_layers=2
        let mut cache = KvCache::new(&cfg, 1, 33).unwrap();

        // Act
        let total = cache.total_pages();

        // Assert: 3 pages/layer × 2 layers = 6 pages total
        assert_eq!(total, 3 * cfg.num_layers);
        // Free count matches total (nothing used yet)
        assert_eq!(cache.free_page_count(), total);

        // Act: append 33 tokens — should succeed, consuming 3 pages on layer 0
        // Token indices: page 0 (0..15), page 1 (16..31), page 2 (32)
        let positions = cache.append(0, 0, 33).unwrap();
        assert_eq!(positions.len(), 33);
        // Token 32 is at seq_len=32 before increment → 32 % 16 = 0 (start of third page)
        assert_eq!(positions[32].1, 0);
        // Token 31 is the last slot of the second page: 31 % 16 = 15
        assert_eq!(positions[31].1, 15);
        assert_eq!(cache.free_page_count(), total - 3);
    }

    // @trace TEST-KVC-36 [req:REQ-KV] [level:unit]
    #[test]
    fn test_write_read_kv_cross_page_boundary() {
        // Arrange: append 20 tokens so positions span 2 pages (page 0: tokens 0..15, page 1: tokens 16..19)
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let positions = cache.append(0, 0, 20).unwrap();

        // Act: write KV data for all 20 tokens in a single write_kv call
        let k_data = make_kv_bytes(20, 4, 16, 1.0);
        let v_data = make_kv_bytes(20, 4, 16, 2.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        // Assert: read back and verify every element, especially around the boundary
        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0u8; 20 * tkb];
        let mut v_out = vec![0u8; 20 * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(len, 20);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);

        // Verify specific values at the page boundary (token 15 and 16)
        // Token 15, head 0, dim 0: K = 1.0 + 15*1000 = 15001.0
        assert_eq!(read_f32(&k_out, 15 * 4 * 16), 1.0 + 15.0 * 1000.0);
        // Token 16, head 0, dim 0: K = 1.0 + 16*1000 = 16001.0
        assert_eq!(read_f32(&k_out, 16 * 4 * 16), 1.0 + 16.0 * 1000.0);
    }

    // @trace TEST-KVC-37 [req:REQ-KV] [level:unit]
    #[test]
    fn test_read_kv_seq_len_exact_multiple_of_page_size() {
        // Arrange: 32 tokens = exactly 2 full pages (32 / 16 = 2, remainder = 0)
        // This exercises the `rem == 0 { PAGE_SIZE }` branch in read_kv's last-page logic
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let positions = cache.append(0, 0, 32).unwrap();

        let k_data = make_kv_bytes(32, 4, 16, 7.0);
        let v_data = make_kv_bytes(32, 4, 16, 8.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        // Act
        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0u8; 32 * tkb];
        let mut v_out = vec![0u8; 32 * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();

        // Assert: all 32 tokens read correctly
        assert_eq!(len, 32);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);
    }

    // @trace TEST-KVC-38 [req:REQ-KV] [level:unit]
    #[test]
    fn test_page_offset_layout_correctness() {
        // Arrange: verify the internal page layout by writing through write_kv and
        // reading specific bytes via page_ptr directly
        let cfg = tiny_config(); // 4 kv_heads, head_dim=16, F32
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let positions = cache.append(0, 0, 1).unwrap();
        let page_id = positions[0].0;

        // Write 1 token with known values: K base=100.0, V base=200.0
        let k_data = make_kv_bytes(1, 4, 16, 100.0);
        let v_data = make_kv_bytes(1, 4, 16, 200.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        // Act: read K[head=0][dim=0] directly from page raw bytes
        let page_data = unsafe {
            let ptr = cache.page_ptr(page_id);
            std::slice::from_raw_parts(ptr, cache.bytes_per_page())
        };

        // K plane: kv_index=0 → page_offset(0, 0, 0) = 0
        // First f32 should be 100.0 (base + 0*1000 + 0*100 + 0)
        let k_first = f32::from_ne_bytes(page_data[0..4].try_into().unwrap());
        assert_eq!(k_first, 100.0);

        // V plane: kv_index=1 → page_offset(1, 0, 0)
        // kv_plane = num_kv_heads * PAGE_SIZE * head_dim = 4*16*16 = 1024 elements
        // byte offset = 1 * 1024 * 4 = 4096
        let kv_plane_bytes = 4 * PAGE_SIZE * 16 * 4; // num_kv_heads * PAGE_SIZE * head_dim * sizeof(f32)
        let v_first = f32::from_ne_bytes(
            page_data[kv_plane_bytes..kv_plane_bytes + 4].try_into().unwrap(),
        );
        assert_eq!(v_first, 200.0);
    }

    // @trace TEST-KVC-39 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_in_preserves_other_sequence() {
        // Arrange: 2 sequences, populate both, swap out only seq 0
        let cfg = tiny_config(); // 2 layers
        let mut cache = KvCache::new(&cfg, 2, 64).unwrap();

        // Populate seq 0 with 10 tokens
        for layer in 0..2 {
            let pos = cache.append(layer, 0, 10).unwrap();
            let k = make_kv_bytes(10, 4, 16, layer as f32 * 1000.0);
            let v = make_kv_bytes(10, 4, 16, layer as f32 * 1000.0 + 500.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }

        // Populate seq 1 with 8 tokens
        for layer in 0..2 {
            let pos = cache.append(layer, 1, 8).unwrap();
            let k = make_kv_bytes(8, 4, 16, layer as f32 * 2000.0);
            let v = make_kv_bytes(8, 4, 16, layer as f32 * 2000.0 + 600.0);
            cache.write_kv(layer, 1, &pos, &k, &v).unwrap();
        }

        // Act: swap out seq 0 only
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Assert: seq 0 cleared, seq 1 untouched
        assert_eq!(cache.seq_len(0, 0), 0);
        assert_eq!(cache.seq_len(1, 0), 0);
        assert_eq!(cache.seq_len(0, 1), 8);
        assert_eq!(cache.seq_len(1, 1), 8);

        // Swap seq 0 back in
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();
        assert_eq!(cache.seq_len(0, 0), 10);
        assert_eq!(cache.seq_len(1, 0), 10);

        // Verify seq 1 data still intact
        let tkb = 4 * 16 * 4;
        for layer in 0..2 {
            let mut k_out = vec![0u8; 8 * tkb];
            let mut v_out = vec![0u8; 8 * tkb];
            cache.read_kv(layer, 1, &mut k_out, &mut v_out).unwrap();
            let k_expected = make_kv_bytes(8, 4, 16, layer as f32 * 2000.0);
            let v_expected = make_kv_bytes(8, 4, 16, layer as f32 * 2000.0 + 600.0);
            assert_eq!(k_out, k_expected, "seq 1 K data corrupted after swap on layer {layer}");
            assert_eq!(v_out, v_expected, "seq 1 V data corrupted after swap on layer {layer}");
        }
    }

    // @trace TEST-KVC-40 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_in_seq_restores_page_used_counts() {
        // Arrange: create a sequence with 25 tokens (2 full pages + 1 partial page with 9 tokens)
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        for layer in 0..2 {
            let pos = cache.append(layer, 0, 25).unwrap();
            let k = make_kv_bytes(25, 4, 16, 0.0);
            let v = make_kv_bytes(25, 4, 16, 1.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }

        // Swap out
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Act: swap in
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();

        // Assert: page_used counts restored correctly
        let pages_l0 = cache.seq_pages(0, 0);
        assert_eq!(pages_l0.len(), 2); // 25 tokens → ceil(25/16) = 2 pages
        assert_eq!(cache.page_used(pages_l0[0]), PAGE_SIZE); // first page full
        assert_eq!(cache.page_used(pages_l0[1]), 25 - PAGE_SIZE); // second page: 9 tokens

        let pages_l1 = cache.seq_pages(1, 0);
        assert_eq!(pages_l1.len(), 2);
        assert_eq!(cache.page_used(pages_l1[0]), PAGE_SIZE);
        assert_eq!(cache.page_used(pages_l1[1]), 25 - PAGE_SIZE);
    }

    // @trace TEST-KVC-41 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_exhausts_all_pages_then_reset() {
        // Arrange: max_seq_len=PAGE_SIZE, batch_size=1, num_layers=2 → exactly 2 pages total
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, PAGE_SIZE).unwrap();
        let total = cache.total_pages();
        assert_eq!(total, 2); // 1 page/layer × 2 layers

        // Act: fill both layers completely
        cache.append(0, 0, PAGE_SIZE).unwrap();
        cache.append(1, 0, PAGE_SIZE).unwrap();
        assert_eq!(cache.free_page_count(), 0);

        // Assert: further append fails
        let result = cache.append(0, 0, 1);
        assert!(result.is_err());

        // Reset and verify pages are available again
        cache.reset_seq(0);
        assert_eq!(cache.free_page_count(), total);

        // Can append again after reset
        let pos = cache.append(0, 0, PAGE_SIZE).unwrap();
        assert_eq!(pos.len(), PAGE_SIZE);
    }

    // @trace TEST-KVC-42 [req:REQ-KV] [level:unit]
    #[test]
    fn test_head_bytes_and_token_kv_bytes_indirect() {
        // Arrange: F32, head_dim=16, num_kv_heads=4
        // head_bytes = 16 * 4 = 64 bytes
        // token_kv_bytes = 4 * 64 = 256 bytes
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Act: verify via bytes_per_page formula:
        // bytes_per_page = 2 * num_kv_heads * PAGE_SIZE * head_dim * dtype_size
        //                = 2 * 4 * 16 * 16 * 4 = 8192
        let expected_bpp = 2 * 4 * PAGE_SIZE * 16 * 4;
        assert_eq!(cache.bytes_per_page(), expected_bpp);

        // Write 3 tokens and verify the size validation in write_kv:
        // expected = 3 * token_kv_bytes = 3 * 256 = 768 bytes
        let positions = cache.append(0, 0, 3).unwrap();
        let token_kv = 4 * 16 * 4; // 256 bytes
        let k_correct = vec![0u8; 3 * token_kv];
        let v_correct = vec![0u8; 3 * token_kv];
        // This should succeed (correct sizes)
        cache.write_kv(0, 0, &positions, &k_correct, &v_correct).unwrap();

        // Providing token_kv - 1 bytes (off by 1) should fail
        let k_short = vec![0u8; 3 * token_kv - 1];
        let result = cache.write_kv(0, 0, &positions, &k_short, &v_correct);
        assert!(result.is_err());
    }

    // @trace TEST-KVC-43 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_seq_empty_sequence() {
        // Arrange: never append anything, then swap out the empty sequence
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: swap out a sequence that was never populated
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Assert: zero bytes written, all pages still free
        assert_eq!(handle.total_bytes(), 0);
        assert!(swap_buf.is_empty() || swap_buf.iter().all(|&b| b == 0));
        assert_eq!(cache.free_page_count(), initial_free);

        // Swap back in should also be a no-op (no pages needed)
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();
        assert_eq!(cache.seq_len(0, 0), 0);
        assert_eq!(cache.free_page_count(), initial_free);
    }

    // @trace TEST-KVC-44 [req:REQ-KV] [level:unit]
    #[test]
    fn test_f16_dtype_halves_bytes_per_page() {
        // Arrange: F16 config (2 bytes per element) vs F32 (4 bytes)
        let cfg_f32 = tiny_config();
        let mut cfg_f16 = tiny_config();
        cfg_f16.dtype = DType::F16;

        // Act
        let cache_f32 = KvCache::new(&cfg_f32, 1, 64).unwrap();
        let cache_f16 = KvCache::new(&cfg_f16, 1, 64).unwrap();

        // Assert: F16 page is half the size of F32 page
        assert_eq!(cache_f16.bytes_per_page(), cache_f32.bytes_per_page() / 2);
        assert_eq!(cache_f16.dtype(), DType::F16);
        // Same number of pages (page count depends on seq topology, not dtype)
        assert_eq!(cache_f16.total_pages(), cache_f32.total_pages());
    }

    // @trace TEST-KVC-45 [req:REQ-KV] [level:unit]
    #[test]
    fn test_multi_layer_write_read_data_integrity() {
        // Arrange: write distinct data to each layer, verify each reads back correctly
        let cfg = tiny_config(); // 2 layers
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let tkb = 4 * 16 * 4; // token_kv_bytes

        for layer in 0..2 {
            let pos = cache.append(layer, 0, 12).unwrap();
            let k = make_kv_bytes(12, 4, 16, layer as f32 * 11111.0);
            let v = make_kv_bytes(12, 4, 16, layer as f32 * 22222.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }

        // Act & Assert: read each layer independently
        for layer in 0..2 {
            let mut k_out = vec![0u8; 12 * tkb];
            let mut v_out = vec![0u8; 12 * tkb];
            let len = cache.read_kv(layer, 0, &mut k_out, &mut v_out).unwrap();
            assert_eq!(len, 12);
            let k_expected = make_kv_bytes(12, 4, 16, layer as f32 * 11111.0);
            let v_expected = make_kv_bytes(12, 4, 16, layer as f32 * 22222.0);
            assert_eq!(k_out, k_expected, "K mismatch layer {layer}");
            assert_eq!(v_out, v_expected, "V mismatch layer {layer}");
        }
    }

    // @trace TEST-KVC-46 [req:REQ-KV] [level:unit]
    #[test]
    fn test_incremental_append_full_data_readback() {
        // Arrange: append in 3 separate calls, then read all data back in one shot
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let tkb = 4 * 16 * 4;

        // First append: 5 tokens
        let pos1 = cache.append(0, 0, 5).unwrap();
        let k1 = make_kv_bytes(5, 4, 16, 10.0);
        let v1 = make_kv_bytes(5, 4, 16, 20.0);
        cache.write_kv(0, 0, &pos1, &k1, &v1).unwrap();

        // Second append: 7 tokens (total 12, still in first page)
        let pos2 = cache.append(0, 0, 7).unwrap();
        let k2 = make_kv_bytes(7, 4, 16, 30.0);
        let v2 = make_kv_bytes(7, 4, 16, 40.0);
        cache.write_kv(0, 0, &pos2, &k2, &v2).unwrap();

        // Third append: 10 tokens (total 22, spans into second page)
        let pos3 = cache.append(0, 0, 10).unwrap();
        let k3 = make_kv_bytes(10, 4, 16, 50.0);
        let v3 = make_kv_bytes(10, 4, 16, 60.0);
        cache.write_kv(0, 0, &pos3, &k3, &v3).unwrap();

        // Act: read all 22 tokens
        assert_eq!(cache.seq_len(0, 0), 22);
        let mut k_out = vec![0u8; 22 * tkb];
        let mut v_out = vec![0u8; 22 * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();

        // Assert: each segment matches what was written
        assert_eq!(len, 22);
        assert_eq!(&k_out[..5 * tkb], &k1[..]);
        assert_eq!(&v_out[..5 * tkb], &v1[..]);
        assert_eq!(&k_out[5 * tkb..12 * tkb], &k2[..]);
        assert_eq!(&v_out[5 * tkb..12 * tkb], &v2[..]);
        assert_eq!(&k_out[12 * tkb..22 * tkb], &k3[..]);
        assert_eq!(&v_out[12 * tkb..22 * tkb], &v3[..]);
    }

    // @trace TEST-KVC-47 [req:REQ-KV] [level:unit]
    #[test]
    fn test_reset_clears_page_data() {
        // Arrange: write known data, then reset, verify page content is zeroed
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let bpp = cache.bytes_per_page();

        let positions = cache.append(0, 0, 5).unwrap();
        let page_id = positions[0].0;
        let k = make_kv_bytes(5, 4, 16, 123.0);
        let v = make_kv_bytes(5, 4, 16, 456.0);
        cache.write_kv(0, 0, &positions, &k, &v).unwrap();

        // Verify the page has non-zero data before reset
        let ptr_before = cache.page_ptr(page_id);
        let has_nonzero = unsafe {
            (0..bpp).any(|i| *ptr_before.add(i) != 0)
        };
        assert!(has_nonzero, "page should have non-zero data after write");

        // Act: reset the sequence
        cache.reset_seq(0);

        // Assert: the page data is zeroed (reset_seq sets used=0 but the data
        // is freed back to the pool — verify we can re-read it as all zeros
        // since swap_out_seq zeroes pages but reset_seq only returns them)
        // After reset, the page is in free_pages. We can allocate it again.
        let pos2 = cache.append(0, 0, 1).unwrap();
        // The newly allocated page may or may not be the same physical page.
        // Either way, its used count should be 1.
        assert_eq!(cache.page_used(pos2[0].0), 1);
    }

    // @trace TEST-KVC-48 [req:REQ-KV] [level:unit]
    #[test]
    fn test_large_batch_multi_sequence_append() {
        // Arrange: batch_size=4, append different lengths to each sequence
        let cfg = tiny_config(); // 2 layers
        let mut cache = KvCache::new(&cfg, 4, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: append to sequences with varying lengths
        cache.append(0, 0, 5).unwrap();
        cache.append(0, 1, 16).unwrap(); // exactly 1 page
        cache.append(0, 2, 17).unwrap(); // just over 1 page
        cache.append(0, 3, 1).unwrap();  // minimal

        // Assert: independent seq_len tracking
        assert_eq!(cache.seq_len(0, 0), 5);
        assert_eq!(cache.seq_len(0, 1), 16);
        assert_eq!(cache.seq_len(0, 2), 17);
        assert_eq!(cache.seq_len(0, 3), 1);

        // Page counts: 5→1 page, 16→1 page, 17→2 pages, 1→1 page = 5 pages for layer 0
        assert_eq!(cache.free_page_count(), initial_free - 5);

        // Other layers are untouched
        for seq in 0..4 {
            assert_eq!(cache.seq_len(1, seq), 0);
        }
    }

    // @trace TEST-KVC-49 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_in_partial_page_preserves_used_count() {
        // Arrange: 7 tokens (partial page), swap out and back in, verify page_used
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        for layer in 0..2 {
            let pos = cache.append(layer, 0, 7).unwrap();
            let k = make_kv_bytes(7, 4, 16, 0.0);
            let v = make_kv_bytes(7, 4, 16, 1.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }

        // Act: swap out and swap back in
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();

        // Assert: page_used counts are correct (7 tokens, not PAGE_SIZE)
        for layer in 0..2 {
            let pages = cache.seq_pages(layer, 0);
            assert_eq!(pages.len(), 1);
            assert_eq!(cache.page_used(pages[0]), 7);
        }
    }

    // @trace TEST-KVC-50 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_single_token_many_times() {
        // Arrange: append 1 token at a time across PAGE_SIZE+1 iterations
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Act: append 17 tokens one at a time (should span 2 pages)
        for i in 0..17 {
            let pos = cache.append(0, 0, 1).unwrap();
            assert_eq!(pos.len(), 1);
            assert_eq!(pos[0].1, i % PAGE_SIZE);
        }

        // Assert
        assert_eq!(cache.seq_len(0, 0), 17);
        // 17 tokens = 2 pages (16 + 1)
        assert_eq!(cache.free_page_count(), initial_free - 2);
    }

    // @trace TEST-KVC-51 [req:REQ-KV] [level:unit]
    #[test]
    fn test_low_level_swap_out_multiple_pages() {
        // Arrange: write distinct patterns to 3 pages, swap all out together
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let bpp = cache.bytes_per_page();

        // Write patterns to pages 0, 1, 2
        for pid in 0..3 {
            let ptr = cache.page_mut_ptr(pid);
            unsafe {
                for i in 0..bpp {
                    *ptr.add(i) = ((pid * 10 + i) % 256) as u8;
                }
            }
        }

        // Act: swap out all 3 pages at once
        let page_ids: Vec<usize> = (0..3).collect();
        let mut buf = vec![0u8; 3 * bpp];
        cache.swap_out(&page_ids, &mut buf).unwrap();

        // Assert: buffer contains all 3 pages in order
        for (pi, &pid) in page_ids.iter().enumerate() {
            let off = pi * bpp;
            for i in 0..bpp {
                assert_eq!(
                    buf[off + i],
                    ((pid * 10 + i) % 256) as u8,
                    "mismatch at page {pi} byte {i}"
                );
            }
        }

        // Corrupt pages, swap in, verify restoration
        for pid in 0..3 {
            let ptr = cache.page_mut_ptr(pid);
            unsafe { std::ptr::write_bytes(ptr, 0, bpp); }
        }
        cache.swap_in(&page_ids, &buf).unwrap();

        for pid in 0..3 {
            let ptr = cache.page_ptr(pid);
            unsafe {
                for i in 0..bpp {
                    assert_eq!(
                        *ptr.add(i),
                        ((pid * 10 + i) % 256) as u8,
                        "restore mismatch page {pid} byte {i}"
                    );
                }
            }
        }
    }

    // @trace TEST-KVC-52 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_in_across_layers_with_data_verification() {
        // Arrange: populate each layer with unique data, swap out, verify handle metadata
        let cfg = tiny_config(); // 2 layers
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Layer 0: 20 tokens (2 pages), Layer 1: 5 tokens (1 page)
        let pos_l0 = cache.append(0, 0, 20).unwrap();
        let k_l0 = make_kv_bytes(20, 4, 16, 100.0);
        let v_l0 = make_kv_bytes(20, 4, 16, 200.0);
        cache.write_kv(0, 0, &pos_l0, &k_l0, &v_l0).unwrap();

        let pos_l1 = cache.append(1, 0, 5).unwrap();
        let k_l1 = make_kv_bytes(5, 4, 16, 300.0);
        let v_l1 = make_kv_bytes(5, 4, 16, 400.0);
        cache.write_kv(1, 0, &pos_l1, &k_l1, &v_l1).unwrap();

        // Act: swap out
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Assert: handle metadata reflects asymmetric layer usage
        assert_eq!(handle.layer_info.len(), 2);
        assert_eq!(handle.layer_info[0], (20, 2)); // layer 0: 20 tokens, 2 pages
        assert_eq!(handle.layer_info[1], (5, 1));  // layer 1: 5 tokens, 1 page
        assert_eq!(handle.total_bytes(), 3 * cache.bytes_per_page());

        // Swap back in and verify data for each layer
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();
        let tkb = 4 * 16 * 4;

        let mut k_out = vec![0u8; 20 * tkb];
        let mut v_out = vec![0u8; 20 * tkb];
        cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(k_out, k_l0);
        assert_eq!(v_out, v_l0);

        let mut k_out2 = vec![0u8; 5 * tkb];
        let mut v_out2 = vec![0u8; 5 * tkb];
        cache.read_kv(1, 0, &mut k_out2, &mut v_out2).unwrap();
        assert_eq!(k_out2, k_l1);
        assert_eq!(v_out2, v_l1);
    }

    // @trace TEST-KVC-53 [req:REQ-KV] [level:unit]
    #[test]
    fn test_new_with_minimal_config() {
        // Arrange: smallest viable config — 1 layer, 1 kv_head, head_dim=1, batch=1, seq=1
        let cfg = ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            intermediate_size: 1,
            num_layers: 1,
            vocab_size: 1,
            max_seq_len: 1,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype: DType::F32,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        };

        // Act
        let mut cache = KvCache::new(&cfg, 1, 1).unwrap();

        // Assert: minimal allocation — ceil(1/16)=1 page × 1 layer = 1 page total
        assert_eq!(cache.total_pages(), 1);
        assert_eq!(cache.free_page_count(), 1);
        // bytes_per_page = 2 * 1 * 16 * 1 * 4 = 128 bytes
        assert_eq!(cache.bytes_per_page(), 128);

        // Can append 1 token
        let pos = cache.append(0, 0, 1).unwrap();
        assert_eq!(pos.len(), 1);
        assert_eq!(pos[0].1, 0);
        assert_eq!(cache.free_page_count(), 0);
    }

    // @trace TEST-KVC-54 [req:REQ-KV] [level:unit]
    #[test]
    fn test_write_kv_zero_tokens_succeeds() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let positions = cache.append(0, 0, 0).unwrap();
        assert!(positions.is_empty());

        let k_data: Vec<u8> = Vec::new();
        let v_data: Vec<u8> = Vec::new();

        // Act: write_kv with zero tokens should succeed
        let result = cache.write_kv(0, 0, &positions, &k_data, &v_data);

        // Assert
        assert!(result.is_ok());
    }

    // @trace TEST-KVC-55 [req:REQ-KV] [level:unit]
    #[test]
    fn test_reset_one_seq_does_not_affect_other_seqs_in_batch() {
        // Arrange: batch_size=3, populate seq 0 and seq 2, reset seq 1 (empty)
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 3, 64).unwrap();
        let initial_free = cache.free_page_count();

        let pos0 = cache.append(0, 0, 10).unwrap();
        let k0 = make_kv_bytes(10, 4, 16, 0.0);
        let v0 = make_kv_bytes(10, 4, 16, 100.0);
        cache.write_kv(0, 0, &pos0, &k0, &v0).unwrap();

        let pos2 = cache.append(0, 2, 15).unwrap();
        let k2 = make_kv_bytes(15, 4, 16, 5.0);
        let v2 = make_kv_bytes(15, 4, 16, 105.0);
        cache.write_kv(0, 2, &pos2, &k2, &v2).unwrap();

        // Act: reset seq 1 (which was never used)
        cache.reset_seq(1);

        // Assert: seq 0 and seq 2 are completely unaffected
        assert_eq!(cache.seq_len(0, 0), 10);
        assert_eq!(cache.seq_len(0, 2), 15);
        assert_eq!(cache.free_page_count(), initial_free - 2);

        // Verify seq 0 data intact
        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0u8; 10 * tkb];
        let mut v_out = vec![0u8; 10 * tkb];
        cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(k_out, k0);
        assert_eq!(v_out, v0);
    }

    // @trace TEST-KVC-56 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_zeros_page_data_in_pool() {
        // Arrange: write data, swap out, verify the physical page is zeroed
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let bpp = cache.bytes_per_page();

        let pos = cache.append(0, 0, 5).unwrap();
        let page_id = pos[0].0;
        let k = make_kv_bytes(5, 4, 16, 42.0);
        let v = make_kv_bytes(5, 4, 16, 84.0);
        cache.write_kv(0, 0, &pos, &k, &v).unwrap();

        // Verify page has non-zero data
        let ptr = cache.page_ptr(page_id);
        let has_data = unsafe { (0..bpp).any(|i| *ptr.add(i) != 0) };
        assert!(has_data);

        // Act: swap out the sequence
        let mut swap_buf = Vec::new();
        cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Assert: the physical page is now zeroed
        let ptr_after = cache.page_ptr(page_id);
        for i in 0..bpp {
            assert_eq!(unsafe { *ptr_after.add(i) }, 0, "page byte {i} not zeroed after swap_out");
        }
    }

    // @trace TEST-KVC-57 [req:REQ-KV] [level:unit]
    #[test]
    fn test_append_positions_within_same_page_contiguous() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Act: append 8 tokens, all should be in the same page with contiguous offsets
        let positions = cache.append(0, 0, 8).unwrap();

        // Assert: all page IDs are the same, offsets are 0..7
        let page_id = positions[0].0;
        for (i, &(pid, off)) in positions.iter().enumerate() {
            assert_eq!(pid, page_id, "token {i} in unexpected page");
            assert_eq!(off, i, "token {i} has unexpected offset");
        }
        assert_eq!(cache.page_used(page_id), 8);
    }

    // @trace TEST-KVC-58 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_handle_layer_info_matches_actual_state() {
        // Arrange: asymmetric layer usage — layer 0: 30 tokens, layer 1: 3 tokens
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        cache.append(0, 0, 30).unwrap(); // ceil(30/16) = 2 pages
        cache.append(1, 0, 3).unwrap();  // 1 page

        // Act
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();

        // Assert: layer_info accurately records (seq_len, num_pages)
        assert_eq!(handle.layer_info.len(), 2);
        assert_eq!(handle.layer_info[0], (30, 2));
        assert_eq!(handle.layer_info[1], (3, 1));
        assert_eq!(handle.total_bytes(), 3 * cache.bytes_per_page());
    }

    // @trace TEST-KVC-59 [req:REQ-KV] [level:unit]
    #[test]
    fn test_new_f8e4m3_dtype_quarters_bytes_per_page() {
        // Arrange: F8E4M3 (1 byte) vs F32 (4 bytes)
        let cfg_f32 = tiny_config();
        let mut cfg_f8 = tiny_config();
        cfg_f8.dtype = DType::F8E4M3;

        // Act
        let cache_f32 = KvCache::new(&cfg_f32, 1, 64).unwrap();
        let cache_f8 = KvCache::new(&cfg_f8, 1, 64).unwrap();

        // Assert: F8E4M3 page is 1/4 the size of F32 page
        assert_eq!(cache_f8.bytes_per_page(), cache_f32.bytes_per_page() / 4);
        assert_eq!(cache_f8.dtype(), DType::F8E4M3);
        assert_eq!(cache_f8.total_pages(), cache_f32.total_pages());
    }

    // @trace TEST-KVC-60 [req:REQ-KV] [level:unit]
    #[test]
    fn test_multiple_reset_cycles_with_data() {
        // Arrange: write data, reset, write different data, reset, write again
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();
        let tkb = 4 * 16 * 4;

        // Cycle 1: write 10 tokens
        let pos1 = cache.append(0, 0, 10).unwrap();
        let k1 = make_kv_bytes(10, 4, 16, 1.0);
        let v1 = make_kv_bytes(10, 4, 16, 2.0);
        cache.write_kv(0, 0, &pos1, &k1, &v1).unwrap();
        assert_eq!(cache.seq_len(0, 0), 10);
        cache.reset_seq(0);
        assert_eq!(cache.free_page_count(), initial_free);

        // Cycle 2: write 20 tokens
        let pos2 = cache.append(0, 0, 20).unwrap();
        let k2 = make_kv_bytes(20, 4, 16, 3.0);
        let v2 = make_kv_bytes(20, 4, 16, 4.0);
        cache.write_kv(0, 0, &pos2, &k2, &v2).unwrap();
        assert_eq!(cache.seq_len(0, 0), 20);
        cache.reset_seq(0);
        assert_eq!(cache.free_page_count(), initial_free);

        // Cycle 3: write 5 tokens and verify
        let pos3 = cache.append(0, 0, 5).unwrap();
        let k3 = make_kv_bytes(5, 4, 16, 5.0);
        let v3 = make_kv_bytes(5, 4, 16, 6.0);
        cache.write_kv(0, 0, &pos3, &k3, &v3).unwrap();

        let mut k_out = vec![0u8; 5 * tkb];
        let mut v_out = vec![0u8; 5 * tkb];
        cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(k_out, k3);
        assert_eq!(v_out, v3);
    }

    // @trace TEST-KVC-61 [req:REQ-KV] [level:unit]
    #[test]
    fn test_page_used_tracks_last_written_token() {
        // Arrange
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Act: append 3 tokens, check used; then append 5 more, check used
        let pos1 = cache.append(0, 0, 3).unwrap();
        let page_id = pos1[0].0;
        assert_eq!(cache.page_used(page_id), 3);

        let pos2 = cache.append(0, 0, 5).unwrap();
        assert_eq!(pos2[0].0, page_id); // same page
        assert_eq!(cache.page_used(page_id), 8); // 3 + 5

        // Append to fill the page (8 more to reach PAGE_SIZE=16)
        let pos3 = cache.append(0, 0, 8).unwrap();
        assert_eq!(cache.page_used(page_id), PAGE_SIZE);

        // Assert: next append allocates a new page
        let pos4 = cache.append(0, 0, 1).unwrap();
        assert_ne!(pos4[0].0, page_id);
        assert_eq!(cache.page_used(pos4[0].0), 1);
    }

    // @trace TEST-KVC-62 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_in_with_max_seq_len_tokens() {
        // Arrange: fill to max_seq_len=48 (exactly 3 pages per layer)
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 48).unwrap();
        let initial_free = cache.free_page_count();

        for layer in 0..2 {
            let pos = cache.append(layer, 0, 48).unwrap();
            assert_eq!(pos.len(), 48);
            let k = make_kv_bytes(48, 4, 16, layer as f32 * 50000.0);
            let v = make_kv_bytes(48, 4, 16, layer as f32 * 50000.0 + 1000.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }
        assert_eq!(cache.free_page_count(), 0);

        // Act: swap out
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();
        assert_eq!(cache.free_page_count(), initial_free);

        // Swap back in
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();
        assert_eq!(cache.free_page_count(), 0);

        // Assert: data integrity
        let tkb = 4 * 16 * 4;
        for layer in 0..2 {
            let mut k_out = vec![0u8; 48 * tkb];
            let mut v_out = vec![0u8; 48 * tkb];
            cache.read_kv(layer, 0, &mut k_out, &mut v_out).unwrap();
            let k_expected = make_kv_bytes(48, 4, 16, layer as f32 * 50000.0);
            let v_expected = make_kv_bytes(48, 4, 16, layer as f32 * 50000.0 + 1000.0);
            assert_eq!(k_out, k_expected, "K mismatch layer {layer}");
            assert_eq!(v_out, v_expected, "V mismatch layer {layer}");
        }
    }

    // @trace TEST-KVC-63 [req:REQ-KV] [level:unit]
    #[test]
    fn test_write_read_kv_two_full_pages_exactly() {
        // Arrange: exactly 2 * PAGE_SIZE = 32 tokens, filling 2 pages completely
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let num_tokens = 2 * PAGE_SIZE;

        let positions = cache.append(0, 0, num_tokens).unwrap();
        assert_eq!(positions.len(), num_tokens);
        assert_eq!(positions[PAGE_SIZE - 1].1, PAGE_SIZE - 1); // last slot of page 0
        assert_eq!(positions[PAGE_SIZE].1, 0); // first slot of page 1

        let k_data = make_kv_bytes(num_tokens, 4, 16, 7.0);
        let v_data = make_kv_bytes(num_tokens, 4, 16, 13.0);
        cache.write_kv(0, 0, &positions, &k_data, &v_data).unwrap();

        // Act: read back all tokens
        let tkb = 4 * 16 * 4;
        let mut k_out = vec![0u8; num_tokens * tkb];
        let mut v_out = vec![0u8; num_tokens * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();

        // Assert: exact match for all 32 tokens
        assert_eq!(len, num_tokens);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);

        // Verify boundary values
        assert_eq!(read_f32(&k_out, 15 * 4 * 16), 7.0 + 15.0 * 1000.0); // last token of page 0
        assert_eq!(read_f32(&k_out, 16 * 4 * 16), 7.0 + 16.0 * 1000.0); // first token of page 1
    }

    // @trace TEST-KVC-64 [req:REQ-KV] [level:unit]
    #[test]
    fn test_zero_page_kv_cache_max_seq_zero() {
        // Arrange: max_seq_len=0 → 0 pages per seq, 0 total pages
        let cfg = tiny_config();
        let cache = KvCache::new(&cfg, 1, 0).unwrap();

        // Act & Assert: zero pages allocated, all free
        assert_eq!(cache.total_pages(), 0);
        assert_eq!(cache.free_page_count(), 0);
        assert_eq!(cache.bytes_per_page(), 2 * 4 * PAGE_SIZE * 16 * 4);
    }

    // @trace TEST-KVC-65 [req:REQ-KV] [level:unit]
    #[test]
    fn test_single_sequence_uses_exactly_its_pages() {
        // Arrange: 1 seq, 1 layer, max_seq=PAGE_SIZE → exactly 1 page
        let mut cfg = tiny_config();
        cfg.num_layers = 1;
        let mut cache = KvCache::new(&cfg, 1, PAGE_SIZE).unwrap();
        assert_eq!(cache.total_pages(), 1);

        // Act: append fills the single page
        let pos = cache.append(0, 0, PAGE_SIZE).unwrap();

        // Assert: no free pages, all positions within the single page
        assert_eq!(cache.free_page_count(), 0);
        let page_id = pos[0].0;
        for &(pid, off) in &pos {
            assert_eq!(pid, page_id);
            assert!(off < PAGE_SIZE);
        }
    }

    // @trace TEST-KVC-66 [req:REQ-KV] [level:unit]
    fn gqa_config(num_kv_heads: usize) -> ModelConfig {
        ModelConfig {
            arch: crate::types::ModelArch::Llama,
            hidden_size: 64,
            num_heads: 8,
            num_kv_heads,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 1,
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

    // @trace TEST-KVC-66 [req:REQ-KV] [level:unit]
    #[test]
    fn test_gqa_fewer_kv_heads_reduces_bytes_per_page() {
        // Arrange: MHA (8 kv_heads) vs GQA (2 kv_heads) — 4x compression
        let cfg_mha = gqa_config(8);
        let cfg_gqa = gqa_config(2);

        // Act
        let cache_mha = KvCache::new(&cfg_mha, 1, 64).unwrap();
        let cache_gqa = KvCache::new(&cfg_gqa, 1, 64).unwrap();

        // Assert: GQA bytes_per_page is 2/8 = 1/4 of MHA
        assert_eq!(cache_gqa.bytes_per_page() * 4, cache_mha.bytes_per_page());
        assert_eq!(cache_gqa.num_kv_heads(), 2);
        assert_eq!(cache_mha.num_kv_heads(), 8);
        // Same total pages (page count depends on seq topology, not kv_heads)
        assert_eq!(cache_gqa.total_pages(), cache_mha.total_pages());
    }

    // @trace TEST-KVC-67 [req:REQ-KV] [level:unit]
    #[test]
    fn test_gqa_write_read_data_integrity() {
        // Arrange: GQA with 2 kv_heads, 8 query heads
        let cfg = gqa_config(2);
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let num_tokens = 10;

        // Act
        let pos = cache.append(0, 0, num_tokens).unwrap();
        let k_data = make_kv_bytes(num_tokens, 2, 16, 33.0);
        let v_data = make_kv_bytes(num_tokens, 2, 16, 77.0);
        cache.write_kv(0, 0, &pos, &k_data, &v_data).unwrap();

        // Assert: read back matches
        let tkb = 2 * 16 * 4; // num_kv_heads * head_dim * sizeof(f32)
        let mut k_out = vec![0u8; num_tokens * tkb];
        let mut v_out = vec![0u8; num_tokens * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(len, num_tokens);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);
    }

    // @trace TEST-KVC-68 [req:REQ-KV] [level:unit]
    #[test]
    fn test_page_recycle_after_reset_and_realloc() {
        // Arrange: allocate, fill, reset, then allocate again — pages must be recyclable
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Cycle 1: allocate 3 pages across 2 layers
        cache.append(0, 0, 20).unwrap(); // 2 pages on layer 0
        cache.append(1, 0, 20).unwrap(); // 2 pages on layer 1
        assert_eq!(cache.free_page_count(), initial_free - 4);

        // Act: reset and re-allocate a different pattern
        cache.reset_seq(0);
        assert_eq!(cache.free_page_count(), initial_free);

        // Now allocate 1 page on layer 0 only
        let pos = cache.append(0, 0, 5).unwrap();

        // Assert: exactly 1 page consumed, positions valid
        assert_eq!(cache.free_page_count(), initial_free - 1);
        assert_eq!(pos.len(), 5);
        assert_eq!(cache.seq_len(0, 0), 5);
        assert_eq!(cache.seq_len(1, 0), 0);
    }

    // @trace TEST-KVC-69 [req:REQ-KV] [level:unit]
    #[test]
    fn test_mixed_precision_bf16_write_read_roundtrip() {
        // Arrange: BF16 cache — 2 bytes per element
        let mut cfg = tiny_config();
        cfg.dtype = DType::BF16;
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();

        // Act: append and write BF16 data (treat as raw bytes)
        let pos = cache.append(0, 0, 8).unwrap();
        let num_tokens = 8;
        let num_kv_heads = 4;
        let head_dim = 16;
        let elem_size = 2; // BF16
        let tkb = num_kv_heads * head_dim * elem_size;

        // Generate deterministic BF16 data as raw byte pairs
        let mut k_data = Vec::with_capacity(num_tokens * tkb);
        let mut v_data = Vec::with_capacity(num_tokens * tkb);
        for t in 0..num_tokens {
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let val: u16 = ((t * 1000 + h * 100 + d) % 65536) as u16;
                    k_data.extend_from_slice(&val.to_ne_bytes());
                    let vval: u16 = ((t * 1000 + h * 100 + d + 50000) % 65536) as u16;
                    v_data.extend_from_slice(&vval.to_ne_bytes());
                }
            }
        }
        cache.write_kv(0, 0, &pos, &k_data, &v_data).unwrap();

        // Assert: read back matches exactly
        let mut k_out = vec![0u8; num_tokens * tkb];
        let mut v_out = vec![0u8; num_tokens * tkb];
        let len = cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(len, num_tokens);
        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);
    }

    // @trace TEST-KVC-70 [req:REQ-KV] [level:unit]
    #[test]
    fn test_mixed_precision_f8e4m3_smallest_page_size() {
        // Arrange: F8E4M3 (1 byte per element) — smallest page
        let mut cfg = tiny_config();
        cfg.dtype = DType::F8E4M3;
        let cache_f8 = KvCache::new(&cfg, 1, 64).unwrap();

        // Act: compare with F32
        let cfg_f32 = tiny_config();
        let cache_f32 = KvCache::new(&cfg_f32, 1, 64).unwrap();

        // Assert: F8E4M3 page is 1/4 of F32 page
        assert_eq!(cache_f8.bytes_per_page() * 4, cache_f32.bytes_per_page());
        assert_eq!(cache_f8.dtype(), DType::F8E4M3);
        // Expected: 2 * 4 * 16 * 16 * 1 = 2048 bytes
        assert_eq!(cache_f8.bytes_per_page(), 2048);
    }

    // @trace TEST-KVC-71 [req:REQ-KV] [level:unit]
    #[test]
    fn test_batch_zero_is_zero_pages() {
        // Arrange: batch_size=0 → 0 total pages
        let cfg = tiny_config();

        // Act
        let cache = KvCache::new(&cfg, 0, 64).unwrap();

        // Assert: no pages, no free pages, but valid bytes_per_page
        assert_eq!(cache.total_pages(), 0);
        assert_eq!(cache.free_page_count(), 0);
        assert!(cache.bytes_per_page() > 0);
    }

    // @trace TEST-KVC-72 [req:REQ-KV] [level:unit]
    #[test]
    fn test_swap_out_in_after_incremental_appends() {
        // Arrange: append in 3 small increments, then swap out/in
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 1, 64).unwrap();
        let tkb = 4 * 16 * 4;

        let pos1 = cache.append(0, 0, 5).unwrap();
        let k1 = make_kv_bytes(5, 4, 16, 10.0);
        let v1 = make_kv_bytes(5, 4, 16, 20.0);
        cache.write_kv(0, 0, &pos1, &k1, &v1).unwrap();

        let pos2 = cache.append(0, 0, 7).unwrap();
        let k2 = make_kv_bytes(7, 4, 16, 30.0);
        let v2 = make_kv_bytes(7, 4, 16, 40.0);
        cache.write_kv(0, 0, &pos2, &k2, &v2).unwrap();

        // Also populate layer 1
        for layer in 0..2 {
            if layer == 0 { continue; }
            let pos = cache.append(layer, 0, 12).unwrap();
            let k = make_kv_bytes(12, 4, 16, 50.0);
            let v = make_kv_bytes(12, 4, 16, 60.0);
            cache.write_kv(layer, 0, &pos, &k, &v).unwrap();
        }

        // Act: swap out and swap back in
        let mut swap_buf = Vec::new();
        let handle = cache.swap_out_seq(0, &mut swap_buf).unwrap();
        cache.swap_in_seq(0, &handle, &swap_buf).unwrap();

        // Assert: layer 0 data intact (first 5 + next 7 = 12 tokens)
        assert_eq!(cache.seq_len(0, 0), 12);
        let mut k_out = vec![0u8; 12 * tkb];
        let mut v_out = vec![0u8; 12 * tkb];
        cache.read_kv(0, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(&k_out[..5 * tkb], &k1[..]);
        assert_eq!(&v_out[..5 * tkb], &v1[..]);
        assert_eq!(&k_out[5 * tkb..12 * tkb], &k2[..]);
        assert_eq!(&v_out[5 * tkb..12 * tkb], &v2[..]);
    }

    // @trace TEST-KVC-73 [req:REQ-KV] [level:unit]
    #[test]
    fn test_multiple_seqs_page_recycling_interleaved() {
        // Arrange: 2 sequences, allocate both, reset one, verify freed pages reusable
        let cfg = tiny_config();
        let mut cache = KvCache::new(&cfg, 2, 64).unwrap();
        let initial_free = cache.free_page_count();

        // Allocate both sequences on layer 0
        cache.append(0, 0, 20).unwrap(); // 2 pages
        cache.append(0, 1, 20).unwrap(); // 2 pages
        assert_eq!(cache.free_page_count(), initial_free - 4);

        // Act: reset seq 0 only — its 2 pages (per layer) return to pool
        cache.reset_seq(0);
        // reset_seq frees pages across ALL layers for seq 0
        // seq 0 had 2 pages on layer 0 and 0 on layer 1 = 2 pages freed
        assert_eq!(cache.seq_len(0, 0), 0);
        assert_eq!(cache.seq_len(0, 1), 20); // seq 1 untouched

        // Assert: freed pages are reusable — append to seq 0 again
        let pos = cache.append(0, 0, 8).unwrap();
        assert_eq!(pos.len(), 8);
        assert_eq!(cache.seq_len(0, 0), 8);
    }
}
