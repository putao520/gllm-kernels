use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
#[cfg(feature = "paged-kernel")]
use burn::tensor::{Int, TensorData};
use std::collections::HashMap;

use crate::ops::flash_attention::{FlashAttentionConfig, HierarchicalFlashAttention};
#[cfg(feature = "cuda-kernel")]
use crate::cuda_kernels::PagedAttentionKernel as CudaPagedAttentionKernel;
#[cfg(feature = "rocm-kernel")]
use crate::hip_kernels::paged_attn::{HipBuffer, HipStreamWrapper};
#[cfg(feature = "rocm-kernel")]
use crate::hip_kernels::PagedAttentionKernel as HipPagedAttentionKernel;
#[cfg(feature = "metal-kernel")]
use crate::metal_kernels::PagedAttentionKernel as MetalPagedAttentionKernel;
#[cfg(feature = "cuda-kernel")]
use cudarc::driver::CudaContext;
#[cfg(any(feature = "cuda-kernel", feature = "rocm-kernel", feature = "metal-kernel"))]
use half::f16;
#[cfg(feature = "metal-kernel")]
use metal::Device;
#[cfg(feature = "metal-kernel")]
use std::mem;
#[cfg(feature = "paged-kernel")]
use std::any::{Any, TypeId};

/// Tokens per physical block.
pub const BLOCK_SIZE: usize = 16;

/// Physical KV block that stores a fixed number of tokens.
#[derive(Debug, Clone)]
pub struct KVBlock<B: Backend> {
    /// Keys: [num_heads, BLOCK_SIZE, head_dim].
    pub keys: Tensor<B, 3>,
    /// Values: [num_heads, BLOCK_SIZE, head_dim].
    pub values: Tensor<B, 3>,
    /// Number of tokens filled in this block.
    pub num_tokens: usize,
}

impl<B: Backend> KVBlock<B> {
    pub fn new(num_heads: usize, head_dim: usize, device: &B::Device) -> Self {
        Self {
            keys: Tensor::zeros([num_heads, BLOCK_SIZE, head_dim], device),
            values: Tensor::zeros([num_heads, BLOCK_SIZE, head_dim], device),
            num_tokens: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.num_tokens >= BLOCK_SIZE
    }

    pub fn remaining_capacity(&self) -> usize {
        BLOCK_SIZE - self.num_tokens
    }
}

/// Manages allocation and reuse of physical blocks.
#[derive(Debug, Clone)]
pub struct BlockManager<B: Backend> {
    free_blocks: Vec<usize>,
    blocks: Vec<Option<KVBlock<B>>>,
    num_heads: usize,
    head_dim: usize,
    device: B::Device,
}

impl<B: Backend> BlockManager<B> {
    pub fn new(max_blocks: usize, num_heads: usize, head_dim: usize, device: B::Device) -> Self {
        let mut blocks = Vec::with_capacity(max_blocks);
        let mut free_blocks = Vec::with_capacity(max_blocks);

        for i in 0..max_blocks {
            blocks.push(Some(KVBlock::new(num_heads, head_dim, &device)));
            free_blocks.push(i);
        }

        Self {
            free_blocks,
            blocks,
            num_heads,
            head_dim,
            device,
        }
    }

    /// Allocate a new block, returning its index.
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop()
    }

    /// Release a block back to the free list.
    pub fn free(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            if let Some(block) = &mut self.blocks[block_id] {
                block.num_tokens = 0;
            }
            self.free_blocks.push(block_id);
        }
    }

    pub fn get(&self, block_id: usize) -> Option<&KVBlock<B>> {
        self.blocks.get(block_id)?.as_ref()
    }

    pub fn get_mut(&mut self, block_id: usize) -> Option<&mut KVBlock<B>> {
        self.blocks.get_mut(block_id)?.as_mut()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

/// Block table mapping logical blocks to physical blocks.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// block_table[logical_block_idx] = physical_block_id.
    block_table: Vec<usize>,
    /// Current sequence length.
    seq_len: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            block_table: Vec::new(),
            seq_len: 0,
        }
    }

    /// Get physical block id and offset for a token index.
    pub fn get_physical_location(&self, token_idx: usize) -> Option<(usize, usize)> {
        if token_idx >= self.seq_len {
            return None;
        }
        let logical_block = token_idx / BLOCK_SIZE;
        let offset = token_idx % BLOCK_SIZE;
        self.block_table
            .get(logical_block)
            .map(|&block_id| (block_id, offset))
    }

    pub fn add_block(&mut self, physical_block_id: usize) {
        self.block_table.push(physical_block_id);
    }

    pub fn extend_seq_len(&mut self, new_tokens: usize) {
        self.seq_len += new_tokens;
    }

    pub fn physical_blocks(&self) -> &[usize] {
        &self.block_table
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn needs_new_block(&self) -> bool {
        let current_capacity = self.block_table.len() * BLOCK_SIZE;
        self.seq_len >= current_capacity
    }
}

/// Reference to a KV block without copying data.
pub struct KVBlockRef<'a, B: Backend> {
    /// Keys slice: [num_heads, valid_tokens, head_dim].
    pub keys: &'a Tensor<B, 3>,
    /// Values slice: [num_heads, valid_tokens, head_dim].
    pub values: &'a Tensor<B, 3>,
    /// Number of valid tokens in this block.
    pub num_tokens: usize,
    /// Block index in the sequence (0-based).
    pub block_idx: usize,
}

/// Iterator over KV blocks for a sequence.
pub struct KVBlockIterator<'a, B: Backend> {
    block_manager: &'a BlockManager<B>,
    block_ids: &'a [usize],
    current_idx: usize,
    remaining_tokens: usize,
}

impl<'a, B: Backend> Iterator for KVBlockIterator<'a, B> {
    type Item = KVBlockRef<'a, B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_tokens == 0 || self.current_idx >= self.block_ids.len() {
            return None;
        }

        let block_id = self.block_ids[self.current_idx];
        let block = self.block_manager.get(block_id)?;

        let valid_tokens = block.num_tokens.min(self.remaining_tokens);
        if valid_tokens == 0 {
            return None;
        }

        let block_ref = KVBlockRef {
            keys: &block.keys,
            values: &block.values,
            num_tokens: valid_tokens,
            block_idx: self.current_idx,
        };

        self.remaining_tokens -= valid_tokens;
        self.current_idx += 1;

        Some(block_ref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.block_ids.len() - self.current_idx;
        (remaining.min(1), Some(remaining))
    }
}

/// Paged KV cache with dynamic block allocation.
#[derive(Debug, Clone)]
pub struct PagedKVCache<B: Backend> {
    block_manager: BlockManager<B>,
    page_tables: Vec<HashMap<usize, BlockTable>>,
    num_layers: usize,
    next_seq_id: usize,
}

impl<B: Backend> PagedKVCache<B> {
    pub fn new(
        max_blocks: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let block_manager = BlockManager::new(max_blocks, num_heads, head_dim, device.clone());
        let page_tables = (0..num_layers).map(|_| HashMap::new()).collect();

        Self {
            block_manager,
            page_tables,
            num_layers,
            next_seq_id: 0,
        }
    }

    /// Allocate a new sequence and return its id.
    pub fn allocate_sequence(&mut self) -> usize {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        for layer in 0..self.num_layers {
            self.page_tables[layer].insert(seq_id, BlockTable::new());
        }

        seq_id
    }

    /// Append KV tensors for a given layer and sequence.
    pub fn append(
        &mut self,
        layer: usize,
        seq_id: usize,
        keys: Tensor<B, 3>,
        values: Tensor<B, 3>,
    ) -> Result<(), &'static str> {
        let (block_manager, page_tables) = (&mut self.block_manager, &mut self.page_tables);
        let layer_tables = page_tables.get_mut(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get_mut(&seq_id).ok_or("unknown sequence")?;

        let [num_heads, new_len, head_dim] = keys.dims();
        let values_dims = values.dims();
        if values_dims != [num_heads, new_len, head_dim] {
            return Err("keys/values shape mismatch");
        }
        if num_heads != block_manager.num_heads() || head_dim != block_manager.head_dim() {
            return Err("keys/values head dimensions mismatch");
        }
        if new_len == 0 {
            return Ok(());
        }

        let mut offset = 0usize;
        while offset < new_len {
            let needs_block = match page_table.physical_blocks().last() {
                Some(&block_id) => {
                    let block = block_manager.get(block_id).ok_or("block not found")?;
                    block.is_full()
                }
                None => true,
            };

            if needs_block {
                let block_id = block_manager
                    .allocate()
                    .ok_or("no free blocks available")?;
                page_table.add_block(block_id);
            }

            let block_id = *page_table
                .physical_blocks()
                .last()
                .ok_or("no block allocated")?;
            let block = block_manager.get_mut(block_id).ok_or("block not found")?;
            let write_len = (new_len - offset).min(block.remaining_capacity());
            if write_len == 0 {
                return Err("block has no remaining capacity");
            }

            let keys_slice = keys.clone().slice([
                0..num_heads,
                offset..(offset + write_len),
                0..head_dim,
            ]);
            let values_slice = values.clone().slice([
                0..num_heads,
                offset..(offset + write_len),
                0..head_dim,
            ]);

            let block_offset = block.num_tokens;
            block.keys = block.keys.clone().slice_assign(
                [
                    0..num_heads,
                    block_offset..(block_offset + write_len),
                    0..head_dim,
                ],
                keys_slice,
            );
            block.values = block.values.clone().slice_assign(
                [
                    0..num_heads,
                    block_offset..(block_offset + write_len),
                    0..head_dim,
                ],
                values_slice,
            );
            block.num_tokens += write_len;
            page_table.extend_seq_len(write_len);

            offset += write_len;
        }

        Ok(())
    }

    /// Gather all cached KV tensors for a layer/sequence into contiguous tensors.
    pub fn get_kv(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        let seq_len = page_table.seq_len();
        if seq_len == 0 {
            return Err("sequence is empty");
        }

        let mut remaining = seq_len;
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for &block_id in page_table.physical_blocks() {
            if remaining == 0 {
                break;
            }
            let block = self
                .block_manager
                .get(block_id)
                .ok_or("block not found")?;
            let [num_heads, _block_size, head_dim] = block.keys.dims();
            let take = block.num_tokens.min(remaining);
            if take == 0 {
                continue;
            }

            keys.push(
                block
                    .keys
                    .clone()
                    .slice([0..num_heads, 0..take, 0..head_dim]),
            );
            values.push(
                block
                    .values
                    .clone()
                    .slice([0..num_heads, 0..take, 0..head_dim]),
            );
            remaining -= take;
        }

        if keys.is_empty() || remaining != 0 {
            return Err("incomplete kv data");
        }

        Ok((Tensor::cat(keys, 1), Tensor::cat(values, 1)))
    }

    pub fn seq_len(&self, layer: usize, seq_id: usize) -> Result<usize, &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        Ok(page_table.seq_len())
    }

    pub fn num_free_blocks(&self) -> usize {
        self.block_manager.num_free_blocks()
    }

    /// Iterate over KV blocks for a sequence without concatenation.
    pub fn iter_kv_blocks(
        &self,
        layer: usize,
        seq_id: usize,
    ) -> Result<KVBlockIterator<'_, B>, &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        let seq_len = page_table.seq_len();

        Ok(KVBlockIterator {
            block_manager: &self.block_manager,
            block_ids: page_table.physical_blocks(),
            current_idx: 0,
            remaining_tokens: seq_len,
        })
    }

    /// Get the number of blocks allocated for a sequence.
    pub fn num_blocks(&self, layer: usize, seq_id: usize) -> Result<usize, &'static str> {
        let layer_tables = self.page_tables.get(layer).ok_or("invalid layer")?;
        let page_table = layer_tables.get(&seq_id).ok_or("unknown sequence")?;
        Ok(page_table.physical_blocks().len())
    }

    /// Get block manager configuration.
    pub fn num_heads(&self) -> usize {
        self.block_manager.num_heads()
    }

    pub fn head_dim(&self) -> usize {
        self.block_manager.head_dim()
    }

    pub fn device(&self) -> &B::Device {
        self.block_manager.device()
    }

    /// Release all blocks associated with a sequence id.
    pub fn free_sequence(&mut self, seq_id: usize) -> Result<(), &'static str> {
        let mut found = false;
        for layer_tables in &mut self.page_tables {
            if let Some(table) = layer_tables.remove(&seq_id) {
                for &block_id in table.physical_blocks() {
                    self.block_manager.free(block_id);
                }
                found = true;
            }
        }

        if found {
            Ok(())
        } else {
            Err("unknown sequence")
        }
    }
}

/// Wrapper for computing attention with a paged KV cache.
pub struct PagedAttention {
    attention: HierarchicalFlashAttention,
}

impl PagedAttention {
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self {
            attention: HierarchicalFlashAttention::new(config),
        }
    }

    pub fn config(&self) -> &FlashAttentionConfig {
        self.attention.config()
    }

    pub fn forward<B: Backend>(
        &self,
        cache: &mut PagedKVCache<B>,
        layer: usize,
        seq_id: usize,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        causal: bool,
    ) -> Result<Tensor<B, 4>, &'static str> {
        let [batch, num_heads, new_seq_len, head_dim] = key.dims();
        if batch != 1 {
            return Err("paged attention expects batch=1");
        }
        if value.dims() != [batch, num_heads, new_seq_len, head_dim] {
            return Err("keys/values shape mismatch");
        }

        let k_reshaped = key.clone().reshape([num_heads, new_seq_len, head_dim]);
        let v_reshaped = value.clone().reshape([num_heads, new_seq_len, head_dim]);
        cache.append(layer, seq_id, k_reshaped, v_reshaped)?;

        let total_seq_len = cache.seq_len(layer, seq_id)?;
        let position_offset = total_seq_len.saturating_sub(new_seq_len);

        #[cfg(feature = "paged-kernel")]
        if causal && self.should_use_kernel::<B>() {
            if let Some(output) = (|| {
                let layer_tables = cache.page_tables.get(layer)?;
                let page_table = layer_tables.get(&seq_id)?;
                let kv_len = total_seq_len;
                if kv_len == 0 {
                    return None;
                }

                let block_ids = page_table.physical_blocks();
                let mut table_data = Vec::with_capacity(kv_len);
                for (block_idx, &block_id) in block_ids.iter().enumerate() {
                    let start = block_idx * BLOCK_SIZE;
                    if start >= kv_len {
                        break;
                    }
                    let end = (start + BLOCK_SIZE).min(kv_len);
                    for _ in start..end {
                        table_data.push(block_id as i64);
                    }
                }
                if table_data.len() != kv_len {
                    log::warn!(
                        "Paged kernel fallback: block table length mismatch ({} != {})",
                        table_data.len(),
                        kv_len
                    );
                    return None;
                }

                let block_tables = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(table_data, [batch, kv_len]),
                    &query.device(),
                );

                let mut k_blocks = Vec::with_capacity(cache.block_manager.blocks.len());
                let mut v_blocks = Vec::with_capacity(cache.block_manager.blocks.len());
                for block in cache.block_manager.blocks.iter() {
                    let block = block.as_ref()?;
                    k_blocks.push(block.keys.clone().swap_dims(0, 1).unsqueeze_dim(0));
                    v_blocks.push(block.values.clone().swap_dims(0, 1).unsqueeze_dim(0));
                }
                if k_blocks.is_empty() {
                    log::warn!("Paged kernel fallback: no KV blocks available");
                    return None;
                }

                let k_cache = Tensor::cat(k_blocks, 0);
                let v_cache = Tensor::cat(v_blocks, 0);
                let block_offsets = vec![position_offset];

                self.try_paged_kernel(
                    &query,
                    &k_cache,
                    &v_cache,
                    &block_tables,
                    &block_offsets,
                    causal,
                )
            })() {
                B::sync(&output.device());
                return Ok(output);
            }
        }

        let (cached_k, cached_v) = cache.get_kv(layer, seq_id)?;
        let cached_seq_len = cached_k.dims()[1];
        let cached_k = cached_k.reshape([1, num_heads, cached_seq_len, head_dim]);
        let cached_v = cached_v.reshape([1, num_heads, cached_seq_len, head_dim]);

        let output = self
            .attention
            .forward(query, cached_k, cached_v, causal, position_offset);
        B::sync(&output.device());
        Ok(output)
    }

    pub fn forward_fused<B: Backend>(
        &self,
        cache: &mut PagedKVCache<B>,
        layer: usize,
        seq_id: usize,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        causal: bool,
    ) -> Result<Tensor<B, 4>, &'static str> {
        let [batch, num_heads, new_seq_len, head_dim] = key.dims();
        if batch != 1 {
            return Err("paged attention expects batch=1");
        }
        if value.dims() != [batch, num_heads, new_seq_len, head_dim] {
            return Err("keys/values shape mismatch");
        }

        let k_reshaped = key.clone().reshape([num_heads, new_seq_len, head_dim]);
        let v_reshaped = value.clone().reshape([num_heads, new_seq_len, head_dim]);
        cache.append(layer, seq_id, k_reshaped, v_reshaped)?;

        let total_seq_len = cache.seq_len(layer, seq_id)?;
        let position_offset = total_seq_len.saturating_sub(new_seq_len);

        let kv_iter = cache.iter_kv_blocks(layer, seq_id)?;
        let kv_blocks: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = kv_iter
            .map(|block| {
                let [num_heads, _, head_dim] = block.keys.dims();
                let k = block
                    .keys
                    .clone()
                    .slice([0..num_heads, 0..block.num_tokens, 0..head_dim]);
                let v = block
                    .values
                    .clone()
                    .slice([0..num_heads, 0..block.num_tokens, 0..head_dim]);
                (k, v)
            })
            .collect();

        let output = self.attention.forward_fused_iter(
            query,
            kv_blocks.into_iter(),
            causal,
            position_offset,
            total_seq_len,
        );
        B::sync(&output.device());
        Ok(output)
    }

    #[cfg(feature = "paged-kernel")]
    fn should_use_kernel<B: Backend + 'static>(&self) -> bool {
        is_cuda_backend::<B>() || is_rocm_backend::<B>() || is_metal_backend::<B>()
    }

    #[cfg(feature = "paged-kernel")]
    fn try_paged_kernel<B: Backend + 'static>(
        &self,
        q: &Tensor<B, 4>,
        k_cache: &Tensor<B, 4>,
        v_cache: &Tensor<B, 4>,
        block_tables: &Tensor<B, 2, Int>,
        block_offsets: &[usize],
        causal: bool,
    ) -> Option<Tensor<B, 4>>
    where
        B::FloatElem: 'static,
    {
        if !causal {
            return None;
        }

        let q_dims = q.dims();
        let [batch_size, num_heads, seq_len, head_dim] = q_dims;
        if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
            return None;
        }
        if block_offsets.len() != batch_size {
            log::warn!(
                "Paged kernel fallback: block_offsets length {} != batch_size {}",
                block_offsets.len(),
                batch_size
            );
            return None;
        }

        let k_dims = k_cache.dims();
        let v_dims = v_cache.dims();
        if k_dims != v_dims {
            log::warn!("Paged kernel fallback: K/V cache shape mismatch");
            return None;
        }

        let [_, block_size, k_heads, k_dim] = k_dims;
        if k_heads != num_heads || k_dim != head_dim {
            log::warn!("Paged kernel fallback: cache head dims mismatch");
            return None;
        }

        let kv_len = block_offsets[0].saturating_add(seq_len);
        if block_offsets
            .iter()
            .any(|offset| offset.saturating_add(seq_len) != kv_len)
        {
            log::warn!("Paged kernel fallback: varying KV lengths across batch");
            return None;
        }
        if block_tables.dims() != [batch_size, kv_len] {
            log::warn!(
                "Paged kernel fallback: block_tables dims {:?} != [{}, {}]",
                block_tables.dims(),
                batch_size,
                kv_len
            );
            return None;
        }

        #[cfg(feature = "cuda-kernel")]
        if is_cuda_backend::<B>() {
            if let Some(output) = self.try_paged_cuda_kernel(
                q,
                k_cache,
                v_cache,
                block_tables,
                block_offsets,
                batch_size,
                num_heads,
                head_dim,
                block_size,
                seq_len,
            ) {
                return Some(output);
            }
        }

        #[cfg(feature = "rocm-kernel")]
        if is_rocm_backend::<B>() {
            if let Some(output) = self.try_paged_rocm_kernel(
                q,
                k_cache,
                v_cache,
                block_tables,
                block_offsets,
                batch_size,
                num_heads,
                head_dim,
                block_size,
                seq_len,
            ) {
                return Some(output);
            }
        }

        #[cfg(feature = "metal-kernel")]
        if is_metal_backend::<B>() {
            if let Some(output) = self.try_paged_metal_kernel(
                q,
                k_cache,
                v_cache,
                block_tables,
                block_offsets,
                batch_size,
                num_heads,
                head_dim,
                block_size,
                seq_len,
            ) {
                return Some(output);
            }
        }

        None
    }

    #[cfg(feature = "cuda-kernel")]
    fn try_paged_cuda_kernel<B: Backend + 'static>(
        &self,
        q: &Tensor<B, 4>,
        k_cache: &Tensor<B, 4>,
        v_cache: &Tensor<B, 4>,
        block_tables: &Tensor<B, 2, Int>,
        block_offsets: &[usize],
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
    ) -> Option<Tensor<B, 4>>
    where
        B::FloatElem: 'static,
    {
        let device = q.device();
        let cuda_index = {
            #[cfg(feature = "cuda")]
            {
                let device_any = &device as &dyn Any;
                if let Some(cuda_device) = device_any.downcast_ref::<burn_cuda::CudaDevice>() {
                    cuda_device.index
                } else {
                    0
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                0
            }
        };

        let cuda_ctx = match CudaContext::new(cuda_index) {
            Ok(ctx) => std::sync::Arc::new(ctx),
            Err(err) => {
                log::warn!("CUDA paged kernel fallback: device init failed: {err}");
                return None;
            }
        };

        let stream = cuda_ctx.default_stream();
        let kernel = match CudaPagedAttentionKernel::new(&cuda_ctx) {
            Ok(kernel) => kernel,
            Err(err) => {
                log::warn!("CUDA paged kernel fallback: kernel load failed: {err}");
                return None;
            }
        };

        let elem_type = TypeId::of::<B::FloatElem>();
        if elem_type == TypeId::of::<f32>() {
            let q_host = q.clone().into_data().into_vec::<f32>().ok()?;
            let k_host = k_cache.clone().into_data().into_vec::<f32>().ok()?;
            let v_host = v_cache.clone().into_data().into_vec::<f32>().ok()?;

            let table_host = block_tables
                .clone()
                .into_data()
                .into_vec::<i64>()
                .ok()?;
            let table_i32: Vec<i32> = table_host
                .into_iter()
                .map(|value| i32::try_from(value).ok())
                .collect::<Option<Vec<_>>>()?;
            let offsets_i32: Vec<i32> = block_offsets
                .iter()
                .map(|value| i32::try_from(*value).ok())
                .collect::<Option<Vec<_>>>()?;

            let q_dev = stream.clone_htod(&q_host).ok()?;
            let k_dev = stream.clone_htod(&k_host).ok()?;
            let v_dev = stream.clone_htod(&v_host).ok()?;
            let table_dev = stream.clone_htod(&table_i32).ok()?;
            let offsets_dev = stream.clone_htod(&offsets_i32).ok()?;

            let output = kernel
                .forward_f32(
                    &stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &table_dev,
                    &offsets_dev,
                    batch_size,
                    num_heads,
                    head_dim,
                    block_size,
                    seq_len,
                )
                .ok()?;

            let out_host = stream.clone_dtoh(&output).ok()?;
            return Some(Tensor::<B, 4>::from_data(
                TensorData::new(out_host, q.dims()),
                &device,
            ));
        }

        if elem_type == TypeId::of::<f16>() {
            let q_host = q.clone().into_data().into_vec::<f16>().ok()?;
            let k_host = k_cache.clone().into_data().into_vec::<f16>().ok()?;
            let v_host = v_cache.clone().into_data().into_vec::<f16>().ok()?;

            let table_host = block_tables
                .clone()
                .into_data()
                .into_vec::<i64>()
                .ok()?;
            let table_i32: Vec<i32> = table_host
                .into_iter()
                .map(|value| i32::try_from(value).ok())
                .collect::<Option<Vec<_>>>()?;
            let offsets_i32: Vec<i32> = block_offsets
                .iter()
                .map(|value| i32::try_from(*value).ok())
                .collect::<Option<Vec<_>>>()?;

            let q_dev = stream.clone_htod(&q_host).ok()?;
            let k_dev = stream.clone_htod(&k_host).ok()?;
            let v_dev = stream.clone_htod(&v_host).ok()?;
            let table_dev = stream.clone_htod(&table_i32).ok()?;
            let offsets_dev = stream.clone_htod(&offsets_i32).ok()?;

            let output = kernel
                .forward_f16(
                    &stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &table_dev,
                    &offsets_dev,
                    batch_size,
                    num_heads,
                    head_dim,
                    block_size,
                    seq_len,
                )
                .ok()?;

            let out_host = stream.clone_dtoh(&output).ok()?;
            return Some(Tensor::<B, 4>::from_data(
                TensorData::new(out_host, q.dims()),
                &device,
            ));
        }

        log::warn!("CUDA paged kernel fallback: unsupported dtype");
        None
    }

    #[cfg(feature = "rocm-kernel")]
    fn try_paged_rocm_kernel<B: Backend + 'static>(
        &self,
        q: &Tensor<B, 4>,
        k_cache: &Tensor<B, 4>,
        v_cache: &Tensor<B, 4>,
        block_tables: &Tensor<B, 2, Int>,
        block_offsets: &[usize],
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
    ) -> Option<Tensor<B, 4>>
    where
        B::FloatElem: 'static,
    {
        let device = q.device();
        let rocm_index = {
            #[cfg(feature = "rocm")]
            {
                let device_any = &device as &dyn Any;
                if let Some(rocm_device) = device_any.downcast_ref::<burn_rocm::RocmDevice>() {
                    rocm_device.index
                } else {
                    0
                }
            }
            #[cfg(not(feature = "rocm"))]
            {
                0
            }
        };

        let kernel = match HipPagedAttentionKernel::new(rocm_index as i32) {
            Ok(kernel) => kernel,
            Err(err) => {
                log::warn!("ROCm paged kernel fallback: kernel load failed: {err}");
                return None;
            }
        };

        let stream = HipStreamWrapper::new().ok()?;

        let elem_type = TypeId::of::<B::FloatElem>();
        if elem_type == TypeId::of::<f32>() {
            let q_host = q.clone().into_data().into_vec::<f32>().ok()?;
            let k_host = k_cache.clone().into_data().into_vec::<f32>().ok()?;
            let v_host = v_cache.clone().into_data().into_vec::<f32>().ok()?;

            let table_host = block_tables
                .clone()
                .into_data()
                .into_vec::<i64>()
                .ok()?;
            let table_i32: Vec<i32> = table_host
                .into_iter()
                .map(|value| i32::try_from(value).ok())
                .collect::<Option<Vec<_>>>()?;
            let offsets_i32: Vec<i32> = block_offsets
                .iter()
                .map(|value| i32::try_from(*value).ok())
                .collect::<Option<Vec<_>>>()?;

            let q_dev = HipBuffer::from_slice(&q_host).ok()?;
            let k_dev = HipBuffer::from_slice(&k_host).ok()?;
            let v_dev = HipBuffer::from_slice(&v_host).ok()?;
            let table_dev = HipBuffer::from_slice(&table_i32).ok()?;
            let offsets_dev = HipBuffer::from_slice(&offsets_i32).ok()?;

            let output = kernel
                .forward_f32(
                    &stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &table_dev,
                    &offsets_dev,
                    batch_size,
                    num_heads,
                    head_dim,
                    block_size,
                    seq_len,
                )
                .ok()?;
            stream.synchronize().ok()?;

            let out_host = output.to_vec().ok()?;
            return Some(Tensor::<B, 4>::from_data(
                TensorData::new(out_host, q.dims()),
                &device,
            ));
        }

        if elem_type == TypeId::of::<f16>() {
            let q_host = q.clone().into_data().into_vec::<f16>().ok()?;
            let k_host = k_cache.clone().into_data().into_vec::<f16>().ok()?;
            let v_host = v_cache.clone().into_data().into_vec::<f16>().ok()?;

            let table_host = block_tables
                .clone()
                .into_data()
                .into_vec::<i64>()
                .ok()?;
            let table_i32: Vec<i32> = table_host
                .into_iter()
                .map(|value| i32::try_from(value).ok())
                .collect::<Option<Vec<_>>>()?;
            let offsets_i32: Vec<i32> = block_offsets
                .iter()
                .map(|value| i32::try_from(*value).ok())
                .collect::<Option<Vec<_>>>()?;

            let q_bits: Vec<u16> = q_host.iter().map(|v| v.to_bits()).collect();
            let k_bits: Vec<u16> = k_host.iter().map(|v| v.to_bits()).collect();
            let v_bits: Vec<u16> = v_host.iter().map(|v| v.to_bits()).collect();

            let q_dev = HipBuffer::from_slice(&q_bits).ok()?;
            let k_dev = HipBuffer::from_slice(&k_bits).ok()?;
            let v_dev = HipBuffer::from_slice(&v_bits).ok()?;
            let table_dev = HipBuffer::from_slice(&table_i32).ok()?;
            let offsets_dev = HipBuffer::from_slice(&offsets_i32).ok()?;

            let output = kernel
                .forward_f16(
                    &stream,
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &table_dev,
                    &offsets_dev,
                    batch_size,
                    num_heads,
                    head_dim,
                    block_size,
                    seq_len,
                )
                .ok()?;
            stream.synchronize().ok()?;

            let out_bits = output.to_vec().ok()?;
            let out_host: Vec<f16> = out_bits.into_iter().map(f16::from_bits).collect();
            return Some(Tensor::<B, 4>::from_data(
                TensorData::new(out_host, q.dims()),
                &device,
            ));
        }

        log::warn!("ROCm paged kernel fallback: unsupported dtype");
        None
    }

    #[cfg(feature = "metal-kernel")]
    fn try_paged_metal_kernel<B: Backend + 'static>(
        &self,
        q: &Tensor<B, 4>,
        k_cache: &Tensor<B, 4>,
        v_cache: &Tensor<B, 4>,
        block_tables: &Tensor<B, 2, Int>,
        block_offsets: &[usize],
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
    ) -> Option<Tensor<B, 4>>
    where
        B::FloatElem: 'static,
    {
        let device = q.device();
        let metal_device = Device::system_default()?;
        let kernel = MetalPagedAttentionKernel::new(&metal_device).ok()?;

        let elem_type = TypeId::of::<B::FloatElem>();
        if elem_type == TypeId::of::<f32>() {
            let q_host = q.clone().into_data().into_vec::<f32>().ok()?;
            let k_host = k_cache.clone().into_data().into_vec::<f32>().ok()?;
            let v_host = v_cache.clone().into_data().into_vec::<f32>().ok()?;

            let table_host = block_tables
                .clone()
                .into_data()
                .into_vec::<i64>()
                .ok()?;
            let table_i32: Vec<i32> = table_host
                .into_iter()
                .map(|value| i32::try_from(value).ok())
                .collect::<Option<Vec<_>>>()?;
            let offsets_i32: Vec<i32> = block_offsets
                .iter()
                .map(|value| i32::try_from(*value).ok())
                .collect::<Option<Vec<_>>>()?;

            let q_buf = metal_device.new_buffer_with_data(
                q_host.as_ptr() as *const _,
                (q_host.len() * mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let k_buf = metal_device.new_buffer_with_data(
                k_host.as_ptr() as *const _,
                (k_host.len() * mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let v_buf = metal_device.new_buffer_with_data(
                v_host.as_ptr() as *const _,
                (v_host.len() * mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let table_buf = metal_device.new_buffer_with_data(
                table_i32.as_ptr() as *const _,
                (table_i32.len() * mem::size_of::<i32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let offsets_buf = metal_device.new_buffer_with_data(
                offsets_i32.as_ptr() as *const _,
                (offsets_i32.len() * mem::size_of::<i32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let output = kernel
                .forward_f32(
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &table_buf,
                    &offsets_buf,
                    batch_size,
                    num_heads,
                    head_dim,
                    block_size,
                    seq_len,
                )
                .ok()?;

            let out_len = batch_size * num_heads * seq_len * head_dim;
            let out_ptr = output.contents() as *const f32;
            let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
            let out_host = out_slice.to_vec();
            return Some(Tensor::<B, 4>::from_data(
                TensorData::new(out_host, q.dims()),
                &device,
            ));
        }

        if elem_type == TypeId::of::<f16>() {
            let q_host = q.clone().into_data().into_vec::<f16>().ok()?;
            let k_host = k_cache.clone().into_data().into_vec::<f16>().ok()?;
            let v_host = v_cache.clone().into_data().into_vec::<f16>().ok()?;

            let table_host = block_tables
                .clone()
                .into_data()
                .into_vec::<i64>()
                .ok()?;
            let table_i32: Vec<i32> = table_host
                .into_iter()
                .map(|value| i32::try_from(value).ok())
                .collect::<Option<Vec<_>>>()?;
            let offsets_i32: Vec<i32> = block_offsets
                .iter()
                .map(|value| i32::try_from(*value).ok())
                .collect::<Option<Vec<_>>>()?;

            let q_bits: Vec<u16> = q_host.iter().map(|v| v.to_bits()).collect();
            let k_bits: Vec<u16> = k_host.iter().map(|v| v.to_bits()).collect();
            let v_bits: Vec<u16> = v_host.iter().map(|v| v.to_bits()).collect();

            let q_buf = metal_device.new_buffer_with_data(
                q_bits.as_ptr() as *const _,
                (q_bits.len() * mem::size_of::<u16>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let k_buf = metal_device.new_buffer_with_data(
                k_bits.as_ptr() as *const _,
                (k_bits.len() * mem::size_of::<u16>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let v_buf = metal_device.new_buffer_with_data(
                v_bits.as_ptr() as *const _,
                (v_bits.len() * mem::size_of::<u16>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let table_buf = metal_device.new_buffer_with_data(
                table_i32.as_ptr() as *const _,
                (table_i32.len() * mem::size_of::<i32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let offsets_buf = metal_device.new_buffer_with_data(
                offsets_i32.as_ptr() as *const _,
                (offsets_i32.len() * mem::size_of::<i32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let output = kernel
                .forward_f16(
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &table_buf,
                    &offsets_buf,
                    batch_size,
                    num_heads,
                    head_dim,
                    block_size,
                    seq_len,
                )
                .ok()?;

            let out_len = batch_size * num_heads * seq_len * head_dim;
            let out_ptr = output.contents() as *const u16;
            let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
            let out_host: Vec<f16> = out_slice.iter().copied().map(f16::from_bits).collect();
            return Some(Tensor::<B, 4>::from_data(
                TensorData::new(out_host, q.dims()),
                &device,
            ));
        }

        log::warn!("Metal paged kernel fallback: unsupported dtype");
        None
    }
}

#[cfg(feature = "paged-kernel")]
fn is_cuda_backend<B: Backend + 'static>() -> bool {
    #[cfg(feature = "cuda")]
    {
        let type_id = TypeId::of::<B>();
        if type_id == TypeId::of::<burn_cuda::Cuda>() {
            return true;
        }
        #[cfg(feature = "fusion")]
        {
            if type_id == TypeId::of::<burn_fusion::Fusion<burn_cuda::Cuda>>() {
                return true;
            }
        }
        false
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(feature = "paged-kernel")]
fn is_rocm_backend<B: Backend + 'static>() -> bool {
    #[cfg(feature = "rocm")]
    {
        let type_id = TypeId::of::<B>();
        if type_id == TypeId::of::<burn_rocm::Rocm>() {
            return true;
        }
        #[cfg(feature = "fusion")]
        {
            if type_id == TypeId::of::<burn_fusion::Fusion<burn_rocm::Rocm>>() {
                return true;
            }
        }
        false
    }
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}

#[cfg(feature = "paged-kernel")]
fn is_metal_backend<B: Backend + 'static>() -> bool {
    #[cfg(feature = "metal")]
    {
        let type_id = TypeId::of::<B>();
        if type_id == TypeId::of::<burn_mlx::Mlx>() {
            return true;
        }
        #[cfg(feature = "fusion")]
        {
            if type_id == TypeId::of::<burn_fusion::Fusion<burn_mlx::Mlx>>() {
                return true;
            }
        }
        false
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_block_manager_allocate_free() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut manager = BlockManager::<NdArray<f32>>::new(2, 2, 4, device);

        let first = manager.allocate().expect("first allocate");
        assert_eq!(manager.num_free_blocks(), 1);
        let second = manager.allocate().expect("second allocate");
        assert_eq!(manager.num_free_blocks(), 0);
        assert!(manager.allocate().is_none());

        if let Some(block) = manager.get_mut(first) {
            block.num_tokens = BLOCK_SIZE;
        }
        manager.free(first);
        assert_eq!(manager.num_free_blocks(), 1);

        let reused = manager.allocate().expect("reuse allocate");
        let reused_block = manager.get(reused).expect("reused block");
        assert_eq!(reused_block.num_tokens, 0);

        manager.free(second);
        manager.free(reused);
        assert_eq!(manager.num_free_blocks(), 2);
    }

    #[test]
    fn test_block_table_mapping() {
        let mut table = BlockTable::new();
        table.add_block(3);
        table.add_block(7);
        table.extend_seq_len(BLOCK_SIZE + 1);

        assert_eq!(table.get_physical_location(0), Some((3, 0)));
        assert_eq!(
            table.get_physical_location(BLOCK_SIZE - 1),
            Some((3, BLOCK_SIZE - 1))
        );
        assert_eq!(table.get_physical_location(BLOCK_SIZE), Some((7, 0)));
        assert_eq!(table.get_physical_location(BLOCK_SIZE + 1), None);
    }

    #[test]
    fn test_paged_kv_cache_basic() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(4, 2, 2, 4, &device);
        let seq_id = cache.allocate_sequence();

        let keys = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        let values = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        cache.append(1, seq_id, keys, values).expect("append");

        assert_eq!(cache.seq_len(1, seq_id).expect("seq len"), BLOCK_SIZE + 3);
        assert_eq!(cache.num_free_blocks(), 2);

        let (k, v) = cache.get_kv(1, seq_id).expect("get kv");
        assert_eq!(k.dims(), [2, BLOCK_SIZE + 3, 4]);
        assert_eq!(v.dims(), [2, BLOCK_SIZE + 3, 4]);

        cache.free_sequence(seq_id).expect("free");
        assert_eq!(cache.num_free_blocks(), 4);
    }

    #[test]
    fn test_iter_kv_blocks() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(8, 1, 2, 4, &device);
        let seq_id = cache.allocate_sequence();

        let total_tokens = BLOCK_SIZE * 2 + 5;
        let keys = Tensor::<NdArray<f32>, 3>::zeros([2, total_tokens, 4], &device);
        let values = Tensor::<NdArray<f32>, 3>::zeros([2, total_tokens, 4], &device);
        cache.append(0, seq_id, keys, values).expect("append");

        let iter = cache.iter_kv_blocks(0, seq_id).expect("iter");
        let blocks: Vec<_> = iter.collect();

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].num_tokens, BLOCK_SIZE);
        assert_eq!(blocks[0].block_idx, 0);
        assert_eq!(blocks[1].num_tokens, BLOCK_SIZE);
        assert_eq!(blocks[1].block_idx, 1);
        assert_eq!(blocks[2].num_tokens, 5);
        assert_eq!(blocks[2].block_idx, 2);

        let total: usize = blocks.iter().map(|b| b.num_tokens).sum();
        assert_eq!(total, total_tokens);
    }

    #[test]
    fn test_kv_blocks_to_tensors() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(4, 2, 2, 4, &device);
        let seq_id = cache.allocate_sequence();

        let keys = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        let values = Tensor::<NdArray<f32>, 3>::zeros([2, BLOCK_SIZE + 3, 4], &device);
        cache.append(1, seq_id, keys, values).expect("append");

        let iter = cache.iter_kv_blocks(1, seq_id).expect("iter");
        let kv_blocks: Vec<_> = iter
            .map(|block| {
                let [num_heads, _, head_dim] = block.keys.dims();
                let k = block
                    .keys
                    .clone()
                    .slice([0..num_heads, 0..block.num_tokens, 0..head_dim]);
                let v = block
                    .values
                    .clone()
                    .slice([0..num_heads, 0..block.num_tokens, 0..head_dim]);
                (k, v)
            })
            .collect();

        assert_eq!(kv_blocks.len(), 2);
    }

    #[test]
    fn test_paged_attention_forward() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut cache = PagedKVCache::<NdArray<f32>>::new(4, 1, 2, 4, &device);
        let seq_id = cache.allocate_sequence();
        let attention = PagedAttention::new(FlashAttentionConfig::default());

        let q = Tensor::<NdArray<f32>, 4>::random(
            [1, 2, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k = Tensor::<NdArray<f32>, 4>::random(
            [1, 2, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v = Tensor::<NdArray<f32>, 4>::random(
            [1, 2, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let output = attention
            .forward(&mut cache, 0, seq_id, q.clone(), k, v, true)
            .expect("paged attention forward");
        assert_eq!(output.dims(), [1, 2, 4, 4]);

        let q2 = Tensor::<NdArray<f32>, 4>::random(
            [1, 2, 2, 4],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k2 = Tensor::<NdArray<f32>, 4>::random(
            [1, 2, 2, 4],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v2 = Tensor::<NdArray<f32>, 4>::random(
            [1, 2, 2, 4],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let output2 = attention
            .forward_fused(&mut cache, 0, seq_id, q2, k2, v2, true)
            .expect("paged attention fused");
        assert_eq!(output2.dims(), [1, 2, 2, 4]);
        assert_eq!(cache.seq_len(0, seq_id).expect("seq_len"), 6);
    }
}
