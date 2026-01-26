use std::mem;

use half::f16;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BufferUsages, CommandEncoder, ComputePipeline};

use super::kernel::{PagedAttentionError, PagedAttentionKernel, PagedAttentionParams, WORKGROUP_SIZE};
use super::utils::{
    align_up, buffer_binding, bytes_of, bytes_to_vec, build_params, expected_elements,
    max_align, read_buffer_sync, slice_as_bytes, validate_cache_len,
};

impl PagedAttentionKernel {
    /// Paged attention forward for f32 inputs.
    pub fn forward_f32(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        block_tables: &[i32],
        block_offsets: &[i32],
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, PagedAttentionError> {
        let params = build_params(
            batch_size,
            num_heads,
            head_dim,
            block_size,
            seq_len,
            block_offsets,
            block_tables,
        )?;
        let expected = expected_elements(batch_size, num_heads, seq_len, head_dim)?;
        if q.len() != expected {
            return Err(PagedAttentionError::InvalidConfig(
                "Q length mismatch".into(),
            ));
        }
        validate_cache_len(k_cache, v_cache, block_size, num_heads, head_dim)?;

        let output_bytes = expected
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| PagedAttentionError::InvalidConfig("output size overflow".into()))?;

        let output = self.dispatch_bytes(
            slice_as_bytes(q),
            slice_as_bytes(k_cache),
            slice_as_bytes(v_cache),
            slice_as_bytes(block_tables),
            slice_as_bytes(block_offsets),
            output_bytes as u64,
            params,
            &self.pipeline_f32,
        )?;

        Ok(bytes_to_vec(&output))
    }

    /// Paged attention forward for f16 inputs.
    pub fn forward_f16(
        &self,
        q: &[f16],
        k_cache: &[f16],
        v_cache: &[f16],
        block_tables: &[i32],
        block_offsets: &[i32],
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f16>, PagedAttentionError> {
        let pipeline = self.pipeline_f16.as_ref().ok_or_else(|| {
            PagedAttentionError::Unsupported("device does not support f16 kernels".into())
        })?;
        let params = build_params(
            batch_size,
            num_heads,
            head_dim,
            block_size,
            seq_len,
            block_offsets,
            block_tables,
        )?;
        let expected = expected_elements(batch_size, num_heads, seq_len, head_dim)?;
        if q.len() != expected {
            return Err(PagedAttentionError::InvalidConfig(
                "Q length mismatch".into(),
            ));
        }
        validate_cache_len(k_cache, v_cache, block_size, num_heads, head_dim)?;

        let output_bytes = expected
            .checked_mul(mem::size_of::<f16>())
            .ok_or_else(|| PagedAttentionError::InvalidConfig("output size overflow".into()))?;

        let output = self.dispatch_bytes(
            slice_as_bytes(q),
            slice_as_bytes(k_cache),
            slice_as_bytes(v_cache),
            slice_as_bytes(block_tables),
            slice_as_bytes(block_offsets),
            output_bytes as u64,
            params,
            pipeline,
        )?;

        Ok(bytes_to_vec(&output))
    }

    fn dispatch_bytes(
        &self,
        q_bytes: &[u8],
        k_bytes: &[u8],
        v_bytes: &[u8],
        block_tables_bytes: &[u8],
        block_offsets_bytes: &[u8],
        output_bytes: u64,
        params: PagedAttentionParams,
        pipeline: &ComputePipeline,
    ) -> Result<Vec<u8>, PagedAttentionError> {
        if output_bytes == 0 {
            return Ok(Vec::new());
        }

        let padded_bytes = align_up(output_bytes, max_align());
        let q_size = q_bytes.len() as u64;
        let k_size = k_bytes.len() as u64;
        let v_size = v_bytes.len() as u64;
        let bt_size = block_tables_bytes.len() as u64;
        let bo_size = block_offsets_bytes.len() as u64;

        // P0-MEM-3: Get buffers from pool
        let (q_buffer, k_buffer, v_buffer, block_tables, block_offsets, output_buffer, readback) = {
            let mut pool = self.buffer_pool.lock().map_err(|_| {
                PagedAttentionError::Wgpu("buffer pool lock poisoned".into())
            })?;

            let q_buf = pool.get_storage(
                &self.device,
                q_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                "paged_attention_q",
            );
            let k_buf = pool.get_storage(
                &self.device,
                k_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                "paged_attention_k_cache",
            );
            let v_buf = pool.get_storage(
                &self.device,
                v_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                "paged_attention_v_cache",
            );
            let bt_buf = pool.get_storage(
                &self.device,
                bt_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                "paged_attention_block_tables",
            );
            let bo_buf = pool.get_storage(
                &self.device,
                bo_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                "paged_attention_block_offsets",
            );
            let out_buf = pool.get_storage(
                &self.device,
                padded_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                "paged_attention_output",
            );
            let rb_buf = pool.get_staging(&self.device, padded_bytes, "paged_attention_readback");

            (q_buf, k_buf, v_buf, bt_buf, bo_buf, out_buf, rb_buf)
        };

        // Upload data to buffers
        self.queue.write_buffer(&q_buffer, 0, q_bytes);
        self.queue.write_buffer(&k_buffer, 0, k_bytes);
        self.queue.write_buffer(&v_buffer, 0, v_bytes);
        self.queue.write_buffer(&block_tables, 0, block_tables_bytes);
        self.queue.write_buffer(&block_offsets, 0, block_offsets_bytes);

        // Params buffer (small, not pooled)
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paged_attention_params"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paged_attention_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                buffer_binding(0, &q_buffer),
                buffer_binding(1, &k_buffer),
                buffer_binding(2, &v_buffer),
                buffer_binding(3, &block_tables),
                buffer_binding(4, &block_offsets),
                buffer_binding(5, &output_buffer),
                buffer_binding(6, &params_buffer),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("paged_attention_encoder"),
            });

        self.encode_pass(
            &mut encoder,
            pipeline,
            &bind_group,
            params.seq_len,
            params.batch_size,
            params.num_heads,
        );

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, padded_bytes);
        self.queue.submit(Some(encoder.finish()));

        let data = read_buffer_sync(&self.device, &readback, padded_bytes)?;

        // P0-MEM-3: Release buffers back to pool
        if let Ok(mut pool) = self.buffer_pool.lock() {
            pool.release_storage(q_buffer, q_size);
            pool.release_storage(k_buffer, k_size);
            pool.release_storage(v_buffer, v_size);
            pool.release_storage(block_tables, bt_size);
            pool.release_storage(block_offsets, bo_size);
            pool.release_storage(output_buffer, padded_bytes);
            pool.release_staging(readback, padded_bytes);
        }

        let mut output = data;
        output.truncate(output_bytes as usize);
        Ok(output)
    }

    fn encode_pass(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        seq_len: u32,
        batch_size: u32,
        num_heads: u32,
    ) {
        let workgroups_x = (seq_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = batch_size.saturating_mul(num_heads);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("paged_attention_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// GPU-pure paged attention forward - works directly with GPU buffers.
    /// No readback/upload - result stays in GPU memory.
    pub fn forward_gpu_pure(
        &self,
        query: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        block_tables: &wgpu::Buffer,
        block_offsets: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: PagedAttentionParams,
        use_f16: bool,
    ) -> Result<(), PagedAttentionError> {
        let pipeline = if use_f16 {
            self.pipeline_f16.as_ref().ok_or_else(|| {
                PagedAttentionError::Unsupported("device does not support f16 kernels".into())
            })?
        } else {
            &self.pipeline_f32
        };

        // Create params buffer
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paged_attention_params_gpu_pure"),
            contents: bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group with existing buffers
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paged_attention_bind_group_gpu_pure"),
            layout: &self.bind_group_layout,
            entries: &[
                buffer_binding(0, query),
                buffer_binding(1, k_cache),
                buffer_binding(2, v_cache),
                buffer_binding(3, block_tables),
                buffer_binding(4, block_offsets),
                buffer_binding(5, output),
                buffer_binding(6, &params_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("paged_attention_encoder_gpu_pure"),
        });

        self.encode_pass(
            &mut encoder,
            pipeline,
            &bind_group,
            params.seq_len,
            params.batch_size,
            params.num_heads,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}
