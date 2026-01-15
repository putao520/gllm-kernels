use std::mem;

use wgpu::{BindGroupEntry, BindGroupLayoutEntry, Buffer, Device};

use super::kernel::{PagedAttentionError, PagedAttentionParams};

pub(super) fn buffer_layout_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn uniform_layout_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn buffer_binding(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(super) fn build_params(
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    seq_len: usize,
    block_offsets: &[i32],
    block_tables: &[i32],
) -> Result<PagedAttentionParams, PagedAttentionError> {
    if batch_size == 0 || num_heads == 0 || head_dim == 0 || block_size == 0 || seq_len == 0 {
        return Err(PagedAttentionError::InvalidConfig(
            "Dimensions must be > 0".into(),
        ));
    }
    if head_dim > super::kernel::MAX_HEAD_DIM {
        return Err(PagedAttentionError::InvalidConfig(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim,
            super::kernel::MAX_HEAD_DIM
        )));
    }
    if block_offsets.len() != batch_size {
        return Err(PagedAttentionError::InvalidConfig(
            "block_offsets length mismatch".into(),
        ));
    }
    if block_offsets.iter().any(|&offset| offset != block_offsets[0]) {
        return Err(PagedAttentionError::InvalidConfig(
            "block_offsets must be uniform".into(),
        ));
    }
    let kv_len = block_offsets[0]
        .checked_add(seq_len as i32)
        .ok_or_else(|| PagedAttentionError::InvalidConfig("kv_len overflow".into()))?;
    if kv_len <= 0 {
        return Err(PagedAttentionError::InvalidConfig(
            "kv_len must be > 0".into(),
        ));
    }
    let kv_len = kv_len as usize;
    let expected_tables = batch_size
        .checked_mul(kv_len)
        .ok_or_else(|| PagedAttentionError::InvalidConfig("block_tables overflow".into()))?;
    if block_tables.len() != expected_tables {
        return Err(PagedAttentionError::InvalidConfig(
            "block_tables length mismatch".into(),
        ));
    }

    let batch_size_u32 = u32::try_from(batch_size)
        .map_err(|_| PagedAttentionError::InvalidConfig("batch_size exceeds u32".into()))?;
    let num_heads_u32 = u32::try_from(num_heads)
        .map_err(|_| PagedAttentionError::InvalidConfig("num_heads exceeds u32".into()))?;
    let head_dim_u32 = u32::try_from(head_dim)
        .map_err(|_| PagedAttentionError::InvalidConfig("head_dim exceeds u32".into()))?;
    let block_size_u32 = u32::try_from(block_size)
        .map_err(|_| PagedAttentionError::InvalidConfig("block_size exceeds u32".into()))?;
    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|_| PagedAttentionError::InvalidConfig("seq_len exceeds u32".into()))?;

    Ok(PagedAttentionParams {
        batch_size: batch_size_u32,
        num_heads: num_heads_u32,
        head_dim: head_dim_u32,
        block_size: block_size_u32,
        seq_len: seq_len_u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    })
}

pub(super) fn validate_cache_len<T>(
    k_cache: &[T],
    v_cache: &[T],
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), PagedAttentionError> {
    if k_cache.len() != v_cache.len() {
        return Err(PagedAttentionError::InvalidConfig(
            "K/V cache length mismatch".into(),
        ));
    }
    let block_stride = block_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| PagedAttentionError::InvalidConfig("block stride overflow".into()))?;
    if block_stride == 0 || k_cache.len() % block_stride != 0 {
        return Err(PagedAttentionError::InvalidConfig(
            "KV cache stride mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn expected_elements(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<usize, PagedAttentionError> {
    let num_queries = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or_else(|| PagedAttentionError::InvalidConfig("num_queries overflow".into()))?;

    num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| PagedAttentionError::InvalidConfig("output_len overflow".into()))
}

pub(super) fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T) as *const u8, mem::size_of::<T>()) }
}

pub(super) fn slice_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * mem::size_of::<T>())
    }
}

pub(super) fn bytes_to_vec<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let len = bytes.len() / mem::size_of::<T>();
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

pub(super) fn align_up(value: u64, align: u64) -> u64 {
    if align == 0 {
        return value;
    }
    (value + align - 1) / align * align
}

pub(super) fn max_align() -> u64 {
    let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
    let map_align = wgpu::MAP_ALIGNMENT;
    if copy_align > map_align {
        copy_align
    } else {
        map_align
    }
}

pub(super) fn read_buffer_sync(
    device: &Device,
    buffer: &Buffer,
    size: u64,
) -> Result<Vec<u8>, PagedAttentionError> {
    let slice = buffer.slice(0..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            return Err(PagedAttentionError::Wgpu(format!("map_async failed: {err}")))
        }
        Err(_) => {
            return Err(PagedAttentionError::Wgpu("map_async channel closed".into()));
        }
    }

    let data = slice.get_mapped_range();
    let bytes = data.to_vec();
    drop(data);
    buffer.unmap();
    Ok(bytes)
}
