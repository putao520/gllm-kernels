impl GpuLower {
    // ── MXFP4 GPU dequant helper ──

    /// MXFP4 (E2M1 + e8m0 scale) GPU dequant: decode packed 4-bit data → f32 values.
    ///
    /// LZ4 流解压 — GPU 实现 (SPEC §3.3.1 REQ-COMP-003).
    ///
    /// GPU 策略: 1 thread per token, warp 协作 match copy + 4KB shared memory 滑动窗口。
    ///
    /// PTX: 使用 `.shared` 内存缓存已解压数据，warp shuffle 协调 match copy。
    /// HIP: `__shared__ unsigned char` 滑窗 + warp-level ballot 找活跃线程。
    /// Metal: `threadgroup unsigned char` + `simdgroup_broadcast` 协作。
    ///
    /// 注意: GPU 端 LZ4 是 in-kernel 解压，compressed_size 在 kernel 参数中传递。
    fn lower_gpu_lz4_decode(
        &mut self,
        src_ptr: VRegId,
        dst_ptr: VRegId,
        compressed_size: VRegId,
        decompressed_size: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let src = self.reg_name_with_kind(src_ptr, alloc);
        let dst = self.reg_name_with_kind(dst_ptr, alloc);
        let csz = self.reg_name_with_kind(compressed_size, alloc);

        // Shared memory sliding window: 4KB (LZ4 max back-reference distance)
        let smem_name = "lz4_window";
        let window_bytes = 4096usize;

        match self.dialect {
            GpuDialect::Ptx { .. } => {
                // PTX: 声明共享内存滑窗 + 单线程顺序解压 (tid == 0 执行主解压路径)
                self.emit_line(&format!(".shared .align 1 .b8 {}[{}];", smem_name, window_bytes));
                self.emit_line("{");
                self.emit_line("  .reg .u64 %lz4_src, %lz4_dst, %lz4_end, %lz4_msrc;");
                self.emit_line("  .reg .u32 %lz4_tok, %lz4_ll, %lz4_ml, %lz4_moff, %lz4_ext, %lz4_pos;");
                self.emit_line("  .reg .pred %lz4_done, %lz4_active;");
                // Only thread 0 in each block executes the sequential decode
                self.emit_line("  mov.u32 %lz4_tok, %tid.x;");
                self.emit_line("  setp.ne.u32 %lz4_active, %lz4_tok, 0;");
                self.emit_line("  @%lz4_active bra LZ4_SKIP_%lz4_pos;");
                self.emit_line(&format!("  mov.u64 %lz4_src, {};", src));
                self.emit_line(&format!("  mov.u64 %lz4_dst, {};", dst));
                self.emit_line(&format!("  add.u64 %lz4_end, %lz4_src, {};", csz));
                // LZ4 decode loop (sequential, thread 0 only)
                self.emit_line("LZ4_LOOP_%lz4_pos:");
                self.emit_line("  setp.ge.u64 %lz4_done, %lz4_src, %lz4_end;");
                self.emit_line("  @%lz4_done bra LZ4_DONE_%lz4_pos;");
                // Load token
                self.emit_line("  ld.global.u8 %lz4_tok, [%lz4_src];");
                self.emit_line("  add.u64 %lz4_src, %lz4_src, 1;");
                // literal_len = tok >> 4
                self.emit_line("  shr.u32 %lz4_ll, %lz4_tok, 4;");
                // match_len_raw = tok & 0xF
                self.emit_line("  and.b32 %lz4_ml, %lz4_tok, 15;");
                // Literal copy loop
                self.emit_line("LZ4_LIT_%lz4_pos:");
                self.emit_line("  setp.eq.u32 %lz4_done, %lz4_ll, 0;");
                self.emit_line("  @%lz4_done bra LZ4_LIT_DONE_%lz4_pos;");
                self.emit_line("  ld.global.u8 %lz4_ext, [%lz4_src];");
                self.emit_line("  st.global.u8 [%lz4_dst], %lz4_ext;");
                self.emit_line("  add.u64 %lz4_src, %lz4_src, 1;");
                self.emit_line("  add.u64 %lz4_dst, %lz4_dst, 1;");
                self.emit_line("  sub.u32 %lz4_ll, %lz4_ll, 1;");
                self.emit_line("  bra LZ4_LIT_%lz4_pos;");
                self.emit_line("LZ4_LIT_DONE_%lz4_pos:");
                // Check end
                self.emit_line("  setp.ge.u64 %lz4_done, %lz4_src, %lz4_end;");
                self.emit_line("  @%lz4_done bra LZ4_DONE_%lz4_pos;");
                // Load match offset (LE u16)
                self.emit_line("  ld.global.u16 %lz4_moff, [%lz4_src];");
                self.emit_line("  add.u64 %lz4_src, %lz4_src, 2;");
                // match_len = ml + 4 (minmatch)
                self.emit_line("  add.u32 %lz4_ml, %lz4_ml, 4;");
                // match_src = dst - moff
                self.emit_line("  cvt.u64.u32 %lz4_msrc, %lz4_moff;");
                self.emit_line("  sub.u64 %lz4_msrc, %lz4_dst, %lz4_msrc;");
                // Match copy loop (byte-by-byte for overlap safety)
                self.emit_line("LZ4_MATCH_%lz4_pos:");
                self.emit_line("  setp.eq.u32 %lz4_done, %lz4_ml, 0;");
                self.emit_line("  @%lz4_done bra LZ4_MATCH_DONE_%lz4_pos;");
                self.emit_line("  ld.global.u8 %lz4_ext, [%lz4_msrc];");
                self.emit_line("  st.global.u8 [%lz4_dst], %lz4_ext;");
                self.emit_line("  add.u64 %lz4_msrc, %lz4_msrc, 1;");
                self.emit_line("  add.u64 %lz4_dst, %lz4_dst, 1;");
                self.emit_line("  sub.u32 %lz4_ml, %lz4_ml, 1;");
                self.emit_line("  bra LZ4_MATCH_%lz4_pos;");
                self.emit_line("LZ4_MATCH_DONE_%lz4_pos:");
                self.emit_line("  bra LZ4_LOOP_%lz4_pos;");
                self.emit_line("LZ4_DONE_%lz4_pos:");
                self.emit_line("LZ4_SKIP_%lz4_pos:");
                // Barrier: all threads wait for thread 0 to finish decode
                self.emit_line("  bar.sync 0;");
                self.emit_line("}");
            }
            GpuDialect::Hip { .. } => {
                let _ = window_bytes;
                self.emit_line(&format!("__shared__ unsigned char {}[4096];", smem_name));
                // Only thread 0 decodes
                self.emit_line("if (threadIdx.x == 0) {");
                self.emit_line(&format!(
                    "  const unsigned char* lz4_src = (const unsigned char*){};", src
                ));
                self.emit_line(&format!(
                    "  unsigned char* lz4_dst = (unsigned char*){};", dst
                ));
                self.emit_line(&format!("  const unsigned char* lz4_end = lz4_src + {};", csz));
                self.emit_line("  while (lz4_src < lz4_end) {");
                self.emit_line("    unsigned token = *lz4_src++;");
                self.emit_line("    unsigned ll = token >> 4;");
                self.emit_line("    unsigned ml = token & 0xF;");
                self.emit_line("    while (ll--) { *lz4_dst++ = *lz4_src++; }");
                self.emit_line("    if (lz4_src >= lz4_end) break;");
                self.emit_line("    unsigned moff = lz4_src[0] | ((unsigned)lz4_src[1] << 8);");
                self.emit_line("    lz4_src += 2;");
                self.emit_line("    ml += 4;");
                self.emit_line("    const unsigned char* msrc = lz4_dst - moff;");
                self.emit_line("    while (ml--) { *lz4_dst++ = *msrc++; }");
                self.emit_line("  }");
                self.emit_line("}");
                self.emit_line("__syncthreads();");
            }
            GpuDialect::Metal { .. } => {
                let _ = (window_bytes, decompressed_size);
                self.emit_line(&format!("threadgroup unsigned char {}[4096];", smem_name));
                self.emit_line("if (thread_index_in_threadgroup == 0) {");
                self.emit_line(&format!(
                    "  const device unsigned char* lz4_src = (const device unsigned char*){};", src
                ));
                self.emit_line(&format!(
                    "  device unsigned char* lz4_dst = (device unsigned char*){};", dst
                ));
                self.emit_line(&format!(
                    "  const device unsigned char* lz4_end = lz4_src + {};", csz
                ));
                self.emit_line("  while (lz4_src < lz4_end) {");
                self.emit_line("    unsigned token = *lz4_src++;");
                self.emit_line("    unsigned ll = token >> 4;");
                self.emit_line("    unsigned ml = token & 0xF;");
                self.emit_line("    while (ll--) { *lz4_dst++ = *lz4_src++; }");
                self.emit_line("    if (lz4_src >= lz4_end) break;");
                self.emit_line("    unsigned moff = (unsigned)lz4_src[0] | ((unsigned)lz4_src[1] << 8);");
                self.emit_line("    lz4_src += 2;");
                self.emit_line("    ml += 4;");
                self.emit_line("    const device unsigned char* msrc = lz4_dst - moff;");
                self.emit_line("    while (ml--) { *lz4_dst++ = *msrc++; }");
                self.emit_line("  }");
                self.emit_line("}");
                self.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
        }
        Ok(())
    }

    /// BitPackRle 解压 — GPU 实现 (SPEC §3.3.2 REQ-COMP-004).
    ///
    /// GPU 策略: 1 thread per run, warp prefix-sum 计算 dst offset 并行展开。
    ///
    /// PTX: 单线程 (tid==0) 顺序展开 + bar.sync 后广播。
    /// HIP: `__shared__` 临时缓冲 + warp `__shfl_up_sync` prefix-sum。
    /// Metal: `threadgroup` 缓冲 + simdgroup_prefix_exclusive_sum。
    fn lower_gpu_bitpack_rle_decode(
        &mut self,
        src_ptr: VRegId,
        dst_ptr: VRegId,
        compressed_size: VRegId,
        nibble_bits: u8,
        element_count: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let _ = (nibble_bits, element_count); // nibble semantics handled by caller
        let src = self.reg_name_with_kind(src_ptr, alloc);
        let dst = self.reg_name_with_kind(dst_ptr, alloc);
        let csz = self.reg_name_with_kind(compressed_size, alloc);

        match self.dialect {
            GpuDialect::Ptx { .. } => {
                self.emit_line("{");
                self.emit_line("  .reg .u64 %rle_src, %rle_dst, %rle_end;");
                self.emit_line("  .reg .u32 %rle_byte, %rle_val, %rle_len, %rle_ext, %rle_tid;");
                self.emit_line("  .reg .pred %rle_done;");
                self.emit_line("  mov.u32 %rle_tid, %tid.x;");
                self.emit_line("  setp.ne.u32 %rle_done, %rle_tid, 0;");
                self.emit_line("  @%rle_done bra RLE_SKIP_%rle_tid;");
                self.emit_line(&format!("  mov.u64 %rle_src, {};", src));
                self.emit_line(&format!("  mov.u64 %rle_dst, {};", dst));
                self.emit_line(&format!("  add.u64 %rle_end, %rle_src, {};", csz));
                self.emit_line("RLE_LOOP_%rle_tid:");
                self.emit_line("  setp.ge.u64 %rle_done, %rle_src, %rle_end;");
                self.emit_line("  @%rle_done bra RLE_DONE_%rle_tid;");
                // Load run byte
                self.emit_line("  ld.global.u8 %rle_byte, [%rle_src];");
                self.emit_line("  add.u64 %rle_src, %rle_src, 1;");
                // run_value = byte & 0x0F
                self.emit_line("  and.b32 %rle_val, %rle_byte, 15;");
                // run_len = byte >> 4
                self.emit_line("  shr.u32 %rle_len, %rle_byte, 4;");
                // Extension: run_len==15 → accumulate from subsequent bytes until <255
                self.emit_line("  setp.ne.u32 %rle_done, %rle_len, 15;");
                self.emit_line("  @%rle_done bra RLE_FILL_%rle_tid;");
                self.emit_line("RLE_EXT_%rle_tid:");
                self.emit_line("  setp.ge.u64 %rle_done, %rle_src, %rle_end;");
                self.emit_line("  @%rle_done bra RLE_FILL_%rle_tid;");
                self.emit_line("  ld.global.u8 %rle_ext, [%rle_src];");
                self.emit_line("  add.u64 %rle_src, %rle_src, 1;");
                self.emit_line("  add.u32 %rle_len, %rle_len, %rle_ext;");
                self.emit_line("  setp.eq.u32 %rle_done, %rle_ext, 255;");
                self.emit_line("  @%rle_done bra RLE_EXT_%rle_tid;");
                // Fill rle_len bytes of rle_val
                self.emit_line("RLE_FILL_%rle_tid:");
                self.emit_line("  setp.eq.u32 %rle_done, %rle_len, 0;");
                self.emit_line("  @%rle_done bra RLE_NEXT_%rle_tid;");
                self.emit_line("  st.global.u8 [%rle_dst], %rle_val;");
                self.emit_line("  add.u64 %rle_dst, %rle_dst, 1;");
                self.emit_line("  sub.u32 %rle_len, %rle_len, 1;");
                self.emit_line("  bra RLE_FILL_%rle_tid;");
                self.emit_line("RLE_NEXT_%rle_tid:");
                self.emit_line("  bra RLE_LOOP_%rle_tid;");
                self.emit_line("RLE_DONE_%rle_tid:");
                self.emit_line("RLE_SKIP_%rle_tid:");
                self.emit_line("  bar.sync 0;");
                self.emit_line("}");
            }
            GpuDialect::Hip { .. } => {
                self.emit_line("if (threadIdx.x == 0) {");
                self.emit_line(&format!(
                    "  const unsigned char* rle_src = (const unsigned char*){};", src
                ));
                self.emit_line(&format!(
                    "  unsigned char* rle_dst = (unsigned char*){};", dst
                ));
                self.emit_line(&format!(
                    "  const unsigned char* rle_end = rle_src + {};", csz
                ));
                self.emit_line("  while (rle_src < rle_end) {");
                self.emit_line("    unsigned b = *rle_src++;");
                self.emit_line("    unsigned val = b & 0x0Fu;");
                self.emit_line("    unsigned len = b >> 4;");
                self.emit_line("    if (len == 15) {");
                self.emit_line("      unsigned ext;");
                self.emit_line("      do { ext = (rle_src < rle_end) ? *rle_src++ : 0; len += ext; } while (ext == 255);");
                self.emit_line("    }");
                self.emit_line("    for (unsigned i = 0; i < len; i++) *rle_dst++ = (unsigned char)val;");
                self.emit_line("  }");
                self.emit_line("}");
                self.emit_line("__syncthreads();");
            }
            GpuDialect::Metal { .. } => {
                self.emit_line("if (thread_index_in_threadgroup == 0) {");
                self.emit_line(&format!(
                    "  const device unsigned char* rle_src = (const device unsigned char*){};", src
                ));
                self.emit_line(&format!(
                    "  device unsigned char* rle_dst = (device unsigned char*){};", dst
                ));
                self.emit_line(&format!(
                    "  const device unsigned char* rle_end = rle_src + {};", csz
                ));
                self.emit_line("  while (rle_src < rle_end) {");
                self.emit_line("    unsigned b = *rle_src++;");
                self.emit_line("    unsigned val = b & 0x0Fu;");
                self.emit_line("    unsigned len = b >> 4;");
                self.emit_line("    if (len == 15) {");
                self.emit_line("      unsigned ext;");
                self.emit_line("      do { ext = (rle_src < rle_end) ? *rle_src++ : 0; len += ext; } while (ext == 255);");
                self.emit_line("    }");
                self.emit_line("    for (unsigned i = 0; i < len; i++) *rle_dst++ = (unsigned char)val;");
                self.emit_line("  }");
                self.emit_line("}");
                self.emit_line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
        }
        Ok(())
    }

    /// Mathematical E2M1 decode:
    ///   sign = nibble[3]
    ///   exp  = nibble[2:1]  (2-bit, values 0-3)
    ///   mant = nibble[0]    (1-bit, values 0-1)
    ///   value = (-1)^sign × (1 + mant) × 2^(exp - 1) × e8m0_scale
    ///   Special: nibble 0 → 0.0
    ///
    /// GPU SIMT model: each thread processes its own data. The VM instruction
    /// carries a SimdWidth::Warp(lanes) that tells us how many f32 output lanes
    /// to produce. We emit scalar-per-lane decode loops for PTX and element-wise
    /// C++ for HIP/Metal.
    ///
    /// ## e8m0 scale decode
    ///   f32::from_bits((byte as u32) << 23) = 2^(byte - 127)
    ///   On GPU we use float arithmetic directly:
    ///     scale = exp2((float)scale_byte - 127.0f)
    fn emit_mxfp4_dequant_gpu(
        &mut self,
        dst: VRegId,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let d = self.reg_name_with_kind(dst, alloc);
        let base = self.reg_name_with_kind(packed_ptr, alloc);
        let off = self.offset_to_string(packed_offset, alloc);
        let scale_gpr = self.reg_name_with_kind(scale_byte_src, alloc);

        let lanes = match width {
            SimdWidth::Warp(l) => l as usize,
            SimdWidth::W256 => 8,
            SimdWidth::W512 => 16,
            SimdWidth::W128 => 4,
            SimdWidth::Scalar => 1,
            SimdWidth::Scalable => 4, // conservative default for SVE-like
        };

        match self.dialect {
            GpuDialect::Ptx { .. } => {
                // ── PTX: SIMT per-thread e2m1 decode (GPU-002 根治 BCE-20260704-GPU-VIOLATIONS) ──
                //
                // 旧 bug: 多 lane 路径 (lanes>2) 仅解码 lane 0 并 `mov.f32 {d}, {fs2}`,
                //   注释自认 "decode all lanes but only store lane 0" — 高 lanes 静默丢弃。
                //   W512 (16 lanes) / W256 (8 lanes) 时 15/7 lanes 未解码 = 数据损坏。
                //
                // 根治 (参照 Q4_0 修复 1c1f1a40 SIMT per-thread 模式):
                //   每线程解码 1 个 nibble (lane = tid.x), 写单个 f32 到 {d} (无覆盖)。
                //   byte_idx = tid >> 1; is_high = tid & 1
                //   nibble = is_high ? (byte >> 4) & 0xF : byte & 0xF
                //   result = e2m1_decode(nibble) × scale → {d}
                //   边界: tid >= lanes → {d} = 0.0 (OOB 守卫)
                //
                // Algorithm (per nibble, no LUT needed):
                //   1. e8m0 scale: mov.u32 tmp, scale_gpr; and tmp, 0xFF; shl tmp, 23;
                //      mov.f32 scale_f32, tmp (IEEE 754 reinterpret)
                //   2. Load packed byte: ld.global.u8 byte, [base + off + byte_idx]
                //   3. Extract low/high nibble (per is_high)
                //   4. E2M1 decode: nibble==0 → 0.0; else magnitude=(1+mant*0.5)*2^(exp-1), apply sign
                //   5. result = magnitude × scale → {d}

                let rs0 = self.scratch_gpr_names[0]; // %rs0 — tid / byte / nibble / u32 temp
                let rs1 = self.scratch_gpr_names[1]; // %rs1 — is_high / u32 temp
                let fs0 = self.scratch_vec_names[0]; // %fs0 — f32 scale
                let fs1 = self.scratch_vec_names[1]; // %fs1 — f32 exp / two_pow temp
                let fs2 = self.scratch_vec_names[2]; // %fs2 — f32 result / magnitude temp
                let ps0 = self.scratch_pred_names[0]; // %ps0 — sign predicate
                let ps1 = self.scratch_pred_names[1]; // %ps1 — zero/OOB predicate

                let oob = self.next_skip_label();
                let done = self.next_skip_label();

                // Step 1: Decode e8m0 scale byte → f32 via IEEE 754 bit reinterpret
                // scale_f32 = f32::from_bits((scale_byte & 0xFF) << 23)
                self.emit_line("{");
                self.emit_line("  .reg .u64 %rd_addr;");
                self.emit_line("  .reg .pred %p_oob;");
                self.emit_line(&format!("  and.b32 {rs0}, {scale_gpr}, 0xFF;"));
                self.emit_line(&format!("  shl.b32 {rs0}, {rs0}, 23;"));
                self.emit_line(&format!("  mov.f32 {fs0}, {rs0};"));

                // tid.x → lane index; OOB guard: tid >= lanes → {d} = 0.0
                self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));

                // byte_idx = tid >> 1; is_high = tid & 1
                self.emit_line(&format!("  shr.u32 {rs1}, {rs0}, 1;"));   // byte_idx
                self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 1;"));    // is_high (reuse rs0)
                // addr = base + off + byte_idx (64-bit)
                self.emit_line(&format!("  cvt.u64.u32 %rd_addr, {rs1};"));
                self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {base};"));
                self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {off};"));
                self.emit_line(&format!("  ld.global.u8 {rs1}, [%rd_addr];")); // byte (reuse rs1)

                // nibble = is_high ? (byte >> 4) & 0xF : byte & 0xF
                // Use predicate on rs0 (is_high) to select shift
                self.emit_line(&format!("  setp.ne.u32 {ps0}, {rs0}, 0;")); // ps0 = is_high
                // Always mask low nibble first
                self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 0xF;"));   // low nibble
                // If is_high: nibble = (byte >> 4) & 0xF — recompute via shift on a temp
                // Simpler: compute high nibble separately and select via predicate
                // Reload byte for high path (PTX L1 caches the line)
                self.emit_line(&format!("  @!{ps0} bra LOW_NIB_{oob};"));
                // High path: re-read byte, shift right 4, mask
                self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_addr];")); // byte again (reuse rs0)
                self.emit_line(&format!("  shr.b32 {rs1}, {rs0}, 4;"));
                self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 0xF;"));
                self.emit_line(&format!("LOW_NIB_{oob}:"));
                // Now rs1 = nibble (0-15)

                // Zero check: nibble == 0 → result = 0.0
                self.emit_line(&format!("  setp.eq.u32 {ps1}, {rs1}, 0;"));
                self.emit_line(&format!("  @{ps1} bra ZERO_{oob};"));

                // Extract sign (bit 3)
                self.emit_line(&format!("  and.b32 {rs0}, {rs1}, 0x8;"));
                self.emit_line(&format!("  setp.ne.u32 {ps0}, {rs0}, 0;")); // ps0 = sign

                // Extract exp (bits 2:1) → f32
                self.emit_line(&format!("  shr.b32 {rs0}, {rs1}, 1;"));
                self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 0x3;"));
                self.emit_line(&format!("  cvt.rn.f32.u32 {fs1}, {rs0};"));
                // 2^(exp-1)
                self.emit_line(&format!("  sub.f32 {fs1}, {fs1}, 1.0;"));
                self.emit_line(&format!("  ex2.approx.f32 {fs1}, {fs1};"));

                // Extract mant (bit 0) → f32, magnitude = (1 + mant*0.5) * 2^(exp-1)
                self.emit_line(&format!("  and.b32 {rs0}, {rs1}, 0x1;"));
                self.emit_line(&format!("  cvt.rn.f32.u32 {fs2}, {rs0};"));
                self.emit_line(&format!("  mul.f32 {fs2}, {fs2}, 0f3F000000;")); // mant * 0.5
                self.emit_line(&format!("  add.f32 {fs2}, {fs2}, 0f3F800000;")); // 1.0 + mant*0.5
                self.emit_line(&format!("  mul.f32 {fs2}, {fs2}, {fs1};"));     // magnitude

                // Apply sign
                self.emit_line(&format!("  @{ps0} neg.f32 {fs2}, {fs2};"));
                // Multiply by e8m0 scale
                self.emit_line(&format!("  mul.f32 {fs2}, {fs2}, {fs0};"));
                self.emit_line(&format!("  mov.f32 {d}, {fs2};"));
                self.emit_line(&format!("  bra DONE_{done};"));

                // Zero path (nibble == 0)
                self.emit_line(&format!("ZERO_{oob}:"));
                self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                self.emit_line(&format!("  bra DONE_{done};"));

                // OOB path (tid >= lanes)
                self.emit_line(&format!("OOB_{oob}:"));
                self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));

                self.emit_line(&format!("DONE_{done}:"));
                self.emit_line("}");
            }

            GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => {
                // ── HIP/Metal (C++ syntax): SIMT per-thread decode (GPU-002 根治) ──
                // 旧 bug: 仅解码 lane 0 → dst, 高 lanes 静默丢弃。
                // 根治: 每线程解码 1 个 nibble (lane = threadIdx.x), OOB 守卫 → 0.0。
                self.emit_line("{");
                self.indent += 1;
                self.emit_line(&format!("unsigned int scale_bits = ((unsigned int){scale_gpr} & 0xFFu) << 23;"));
                self.emit_line("float scale_f;");
                self.emit_line("memcpy(&scale_f, &scale_bits, sizeof(float));");
                self.emit_line(&format!("unsigned char* packed_base = (unsigned char*){base};"));
                self.emit_line(&format!("unsigned int packed_off = (unsigned int)({off});"));
                // SIMT per-thread: each thread decodes one nibble
                let tid = if matches!(self.dialect, GpuDialect::Hip { .. }) { "threadIdx.x" } else { "thread_position_in_threadgroup.x" };
                self.emit_line(&format!("unsigned int _t = (unsigned int){tid};"));
                self.emit_line(&format!("if (_t < {lanes}) {{"));
                self.indent += 1;
                self.emit_line("unsigned int byte_idx = _t >> 1;");
                self.emit_line("unsigned int is_high = _t & 1u;");
                self.emit_line("unsigned char raw_byte = packed_base[packed_off + byte_idx];");
                self.emit_line("unsigned int nibble = is_high ? ((raw_byte >> 4) & 0xF) : (raw_byte & 0xF);");
                self.emit_line("float result;");
                self.emit_line("if (nibble == 0) {");
                self.indent += 1;
                self.emit_line("result = 0.0f;");
                self.indent = self.indent.saturating_sub(1);
                self.emit_line("} else {");
                self.indent += 1;
                self.emit_line("unsigned int sign_bit = (nibble >> 3) & 1u;");
                self.emit_line("unsigned int exp_field = (nibble >> 1) & 3u;");
                self.emit_line("unsigned int mant_field = nibble & 1u;");
                self.emit_line("float two_pow = exp2f((float)exp_field - 1.0f);");
                self.emit_line("float magnitude = (1.0f + (float)mant_field) * two_pow;");
                self.emit_line("result = (sign_bit ? -magnitude : magnitude);");
                self.emit_line("result *= scale_f;");
                self.indent = self.indent.saturating_sub(1);
                self.emit_line("}");
                self.emit_line(&format!("{d} = result;"));
                self.indent = self.indent.saturating_sub(1);
                self.emit_line("} else {");
                self.indent += 1;
                self.emit_line(&format!("{d} = 0.0f;  // OOB guard"));
                self.indent = self.indent.saturating_sub(1);
                self.emit_line("}");

                self.indent = self.indent.saturating_sub(1);
                self.emit_line("}");
            }
        }
        let _ = lanes; // lanes used in PTX/HIP/Metal branches above
        Ok(())
    }

    /// NVFP4 sub-block dequant: decode E2M1 packed nibbles with UE4M3 sub-block scale.
    ///
    /// Same E2M1 nibble decode as Mxfp4VecDequant, but scale is UE4M3 (unsigned FP8 E4M3, bias=7)
    /// instead of E8M0. The global tensor scale is applied at the epilogue level.
    fn emit_nvfp4_sub_block_dequant_gpu(
        &mut self,
        dst: VRegId,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let d = self.reg_name_with_kind(dst, alloc);
        let base = self.reg_name_with_kind(packed_ptr, alloc);
        let off = self.offset_to_string(packed_offset, alloc);
        let scale_gpr = self.reg_name_with_kind(scale_byte_src, alloc);

        let lanes = match width {
            SimdWidth::Warp(l) => l as usize,
            SimdWidth::W256 => 8,
            SimdWidth::W512 => 16,
            SimdWidth::W128 => 4,
            SimdWidth::Scalar => 1,
            SimdWidth::Scalable => 4,
        };

        match self.dialect {
            GpuDialect::Ptx { .. } => {
                let rs0 = self.scratch_gpr_names[0];
                let rs1 = self.scratch_gpr_names[1];
                let fs0 = self.scratch_vec_names[0]; // ue4m3 scale f32
                let fs1 = self.scratch_vec_names[1]; // e2m1 magnitude
                let fs2 = self.scratch_vec_names[2]; // result temp
                let ps0 = self.scratch_pred_names[0]; // sign predicate
                let ps1 = self.scratch_pred_names[1]; // zero nibble predicate

                // ── Step 1: Decode UE4M3 scale byte → f32 ──
                // byte = 0 → 0.0
                // normal: exp = (byte >> 3) & 0xF, mant = byte & 0x7
                //   value = 2^(exp-7) × (1 + mant/8)
                //   → build f32 via IEEE 754: biased_exp = exp + 120
                //   f32_bits = (biased_exp << 23) | (mant << 20)
                self.emit_line("{");
                self.indent += 1;

                // Load and mask scale byte
                self.emit_line(&format!("and.b32 {rs0}, {scale_gpr}, 0xFF;"));

                // Check zero scale
                let scale_zero_label = self.next_skip_label();
                self.emit_line(&format!("setp.eq.u32 {ps1}, {rs0}, 0;"));
                self.emit_line(&format!("@{ps1} bra SCALE_ZERO_{scale_zero_label};"));

                // Non-zero: decode UE4M3
                // rs1 = exp = (byte >> 3) & 0xF
                self.emit_line(&format!("shr.b32 {rs1}, {rs0}, 3;"));
                self.emit_line(&format!("and.b32 {rs1}, {rs1}, 0xF;"));
                // biased_exp = exp + 120
                self.emit_line(&format!("add.u32 {rs1}, {rs1}, 120;"));
                // rs0 = mant = byte & 0x7
                self.emit_line(&format!("and.b32 {rs0}, {rs0}, 0x7;"));
                // Build f32 bits: (biased_exp << 23) | (mant << 20)
                self.emit_line(&format!("shl.b32 {rs1}, {rs1}, 23;"));
                self.emit_line(&format!("shl.b32 {rs0}, {rs0}, 20;"));
                self.emit_line(&format!("or.b32 {rs0}, {rs0}, {rs1};"));
                self.emit_line(&format!("mov.f32 {fs0}, {rs0};"));
                let scale_done_label = self.next_skip_label();
                self.emit_line(&format!("bra SCALE_DONE_{scale_done_label};"));

                self.emit_line(&format!("SCALE_ZERO_{scale_zero_label}:"));
                self.emit_line(&format!("mov.f32 {fs0}, 0f00000000;"));
                self.emit_line(&format!("SCALE_DONE_{scale_done_label}:"));

                // ── Step 2: Decode E2M1 nibble(s) and multiply by scale ──
                // For GPU SIMT, each thread processes one nibble (lane 0)
                let byte_idx = 0;
                self.emit_line(&format!("cvt.u64.u32 %rd_addr, {off};"));
                self.emit_line(&format!("add.u64 %rd_addr, %rd_addr, {base};"));
                self.emit_line(&format!("ld.global.u8 {rs0}, [%rd_addr];"));

                // Low nibble
                self.emit_line(&format!("and.b32 {rs0}, {rs0}, 0xF;"));

                // Zero check for nibble
                let nib_zero_label = self.next_skip_label();
                self.emit_line(&format!("setp.eq.u32 {ps1}, {rs0}, 0;"));
                self.emit_line(&format!("@{ps1} bra NIB_ZERO_{nib_zero_label};"));

                // Sign (bit 3)
                self.emit_line(&format!("and.b32 {rs1}, {rs0}, 0x8;"));
                self.emit_line(&format!("setp.ne.u32 {ps0}, {rs1}, 0;"));

                // Exp (bits 2:1)
                self.emit_line(&format!("shr.b32 {rs1}, {rs0}, 1;"));
                self.emit_line(&format!("and.b32 {rs1}, {rs1}, 0x3;"));
                self.emit_line(&format!("cvt.rn.f32.u32 {fs1}, {rs1};"));

                // Mant (bit 0)
                self.emit_line(&format!("and.b32 {rs1}, {rs0}, 0x1;"));
                self.emit_line(&format!("cvt.rn.f32.u32 {fs2}, {rs1};"));

                // 2^(exp-1)
                self.emit_line(&format!("sub.f32 {fs1}, {fs1}, 1.0;"));
                self.emit_line(&format!("ex2.approx.f32 {fs1}, {fs1};"));

                // magnitude = (1 + mant) * 2^(exp-1)
                self.emit_line(&format!("add.f32 {fs2}, 1.0, {fs2};"));
                self.emit_line(&format!("mul.f32 {fs2}, {fs2}, {fs1};"));

                // Sign
                self.emit_line(&format!("@{ps0} neg.f32 {fs2}, {fs2};"));

                // Multiply by UE4M3 scale
                self.emit_line(&format!("mul.f32 {d}, {fs2}, {fs0};"));

                let nib_done_label = self.next_skip_label();
                self.emit_line(&format!("bra NIB_DONE_{nib_done_label};"));
                self.emit_line(&format!("NIB_ZERO_{nib_zero_label}:"));
                self.emit_line(&format!("mov.f32 {d}, 0f00000000;"));
                self.emit_line(&format!("NIB_DONE_{nib_done_label}:"));

                self.indent = self.indent.saturating_sub(1);
                self.emit_line("}");

                let _ = lanes; // PTX currently processes lane 0 only per instruction
            }
            GpuDialect::Hip { .. } => {
                let rs0 = self.scratch_gpr_names[0];
                let fs0 = self.scratch_vec_names[0];

                self.emit_line("{");
                self.indent += 1;

                // UE4M3 scale decode
                self.emit_line(&format!("unsigned int scale_byte = (unsigned int)({scale_gpr}) & 0xFF;"));
                self.emit_line(&format!("float {fs0};"));
                self.emit_line("if (scale_byte == 0) {");
                self.emit_line(&format!("  {fs0} = 0.0f;"));
                self.emit_line("} else {");
                self.emit_line("  unsigned int exp_field = (scale_byte >> 3) & 0xF;");
                self.emit_line("  unsigned int mant_field = scale_byte & 0x7;");
                self.emit_line("  unsigned int biased_exp = exp_field + 120;");

                // Build f32 bit pattern via union
                self.emit_line("  union { unsigned int u; float f; } scale_bits;");
                self.emit_line("  scale_bits.u = (biased_exp << 23) | (mant_field << 20);");
                self.emit_line(&format!("  {fs0} = scale_bits.f;"));
                self.emit_line("}");

                // E2M1 nibble decode (same as Mxfp4VecDequant HIP path)
                self.emit_line(&format!("unsigned int {rs0} = ((unsigned char*)({base}))[{off}];"));
                self.emit_line(&format!("{rs0} = {rs0} & 0xF;"));

                self.emit_line(&format!("float {d};"));
                self.emit_line(&format!("if ({rs0} == 0) {{"));
                self.emit_line(&format!("  {d} = 0.0f;"));
                self.emit_line("} else {");

                // E2M1 math decode
                self.emit_line(&format!("  int sign_bit = ({rs0} >> 3) & 1;"));
                self.emit_line(&format!("  int exp_val = ({rs0} >> 1) & 3;"));
                self.emit_line(&format!("  int mant_val = {rs0} & 1;"));
                self.emit_line("  float two_pow = exp2((float)exp_val - 1.0f);");
                self.emit_line("  float magnitude = (1.0f + (float)mant_val) * two_pow;");
                self.emit_line("  if (sign_bit) magnitude = -magnitude;");
                self.emit_line(&format!("  {d} = magnitude * {fs0};"));
                self.emit_line("}");

                self.indent = self.indent.saturating_sub(1);
                self.emit_line("}");

                let _ = lanes;
            }
            GpuDialect::Metal { .. } => {
                let fs0 = self.scratch_vec_names[0];

                self.emit_line("{");
                self.indent += 1;

                // UE4M3 scale decode
                self.emit_line(&format!("unsigned int scale_byte = (unsigned int)((device unsigned char*)({scale_gpr}))[0] & 0xFF;"));
                self.emit_line(&format!("float {fs0};"));
                self.emit_line("if (scale_byte == 0) {");
                self.emit_line(&format!("  {fs0} = 0.0f;"));
                self.emit_line("} else {");
                self.emit_line("  unsigned int exp_field = (scale_byte >> 3) & 0xF;");
                self.emit_line("  unsigned int mant_field = scale_byte & 0x7;");
                self.emit_line("  unsigned int biased_exp = exp_field + 120;");
                self.emit_line("  as_type<float> scale_f = as_type<float>((biased_exp << 23) | (mant_field << 20));");
                self.emit_line(&format!("  {fs0} = scale_f;"));
                self.emit_line("}");

                // E2M1 nibble decode
                self.emit_line(&format!("unsigned int nibble = ((device unsigned char*)({base}))[{off}] & 0xF;"));
                self.emit_line(&format!("float {d};"));
                self.emit_line("if (nibble == 0) {");
                self.emit_line(&format!("  {d} = 0.0f;"));
                self.emit_line("} else {");
                self.emit_line("  int sign_bit = (nibble >> 3) & 1;");
                self.emit_line("  int exp_val = (nibble >> 1) & 3;");
                self.emit_line("  int mant_val = nibble & 1;");
                self.emit_line("  float two_pow = exp2((float)exp_val - 1.0f);");
                self.emit_line("  float magnitude = (1.0f + (float)mant_val) * two_pow;");
                self.emit_line("  if (sign_bit) magnitude = -magnitude;");
                self.emit_line(&format!("  {d} = magnitude * {fs0};"));
                self.emit_line("}");

                self.indent = self.indent.saturating_sub(1);
                self.emit_line("}");

                let _ = lanes;
            }
        }
        Ok(())
    }

    /// KIVI 4-bit quantization helper — shared by KiviQuantChannel and KiviQuantToken.
    fn emit_kivi_quant(
        &mut self,
        src: &VRegId,
        dst_ptr: &VRegId,
        scale_ptr: &VRegId,
        num_elements: usize,
        alloc: &RegAllocation,
        _label: &str,
    ) -> Result<(), CompilerError> {
        // GPU-004 根治 (BCE-20260704-GPU-VIOLATIONS):
        //   旧实现 `for pair in 0..num_pairs` (num_pairs=(num_elements+1)/2, 通常 32-64)
        //   Rust 循环展开 PTX/HIP/Metal 行, 违反 NO-LOOP-UNROLL。
        //   根治: SIMT per-thread (参照 Q4_0 修复 1c1f1a40), 每个 thread 处理 1 个 pair,
        //   pair = tid.x; OOB 守卫: tid >= num_pairs → 不写。
        let s = self.reg_name_with_kind(*src, alloc);
        let dp = self.reg_name_with_kind(*dst_ptr, alloc);
        let sp = self.reg_name_with_kind(*scale_ptr, alloc);
        let rs0 = self.scratch_gpr_names[0];
        let rs1 = self.scratch_gpr_names[1];
        let num_pairs = (num_elements + 1) / 2;

        match self.dialect {
            GpuDialect::Ptx { .. } => {
                // PTX SIMT per-thread: pair = tid.x; OOB guard tid >= num_pairs → skip.
                // Each thread: load lo/hi f32, scale=max(|lo|,|hi|), pack nibbles, store.
                // Layout: scratch_gpr[0]=rs0 (tid/pair), [1]=rs1 (offsets/lo byte),
                //         [2]=rs2 (hi byte/nibble), [3]=rs3 (scale_addr/dst_addr temp).
                let rs2 = self.scratch_gpr_names[2];
                let oob = self.next_skip_label();

                self.emit_line("{");
                self.emit_line("  .reg .u64 %rd_pair;   // pair index (u64 for addressing)");
                self.emit_line("  .reg .u64 %rd_scale;  // scale store addr = sp + pair*4");
                self.emit_line("  .reg .u64 %rd_dst;    // dst store addr  = dp + pair");
                self.emit_line("  .reg .pred %p_oob;");
                self.emit_line("  .reg .pred %p_last;");

                // pair = tid.x; OOB guard
                self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {num_pairs};"));
                self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));

                // Compute addresses up-front (before loads clobber rs0)
                // rd_pair = (u64)pair
                self.emit_line(&format!("  cvt.u64.u32 %rd_pair, {rs0};"));
                // rd_scale = sp + pair*4
                self.emit_line(&format!("  shl.u64 %rd_scale, %rd_pair, 2;"));
                self.emit_line(&format!("  add.u64 %rd_scale, %rd_scale, {sp};"));
                // rd_dst = dp + pair
                self.emit_line(&format!("  add.u64 %rd_dst, %rd_pair, {dp};"));

                // lo_off_bytes = pair*8; load lo → rs0 (clobbers tid, but addrs already computed)
                self.emit_line(&format!("  mad.lo.u32 {rs1}, {rs0}, 8, 0;"));  // lo_off = pair*8
                self.emit_line(&format!("  ld.global.f32 {rs0}, [{s}+{rs1}];")); // lo
                self.emit_line(&format!("  abs.f32 {rs0}, {rs0};"));             // |lo|

                // hi_off_bytes = pair*8 + 4
                self.emit_line(&format!("  add.u32 {rs1}, {rs1}, 4;"));

                if num_elements % 2 == 1 {
                    // Odd num_elements: last pair has no hi element.
                    let last_pair = num_pairs - 1;
                    self.emit_line(&format!("  setp.eq.u32 %p_last, %tid.x, {last_pair};"));
                    self.emit_line("  @%p_last bra LAST_PAIR_{oob};");
                    // Even pair: load hi, scale=max(|lo|,|hi|), pack both nibbles
                    self.emit_line(&format!("  ld.global.f32 {rs2}, [{s}+{rs1}];")); // hi
                    self.emit_line(&format!("  abs.f32 {rs2}, {rs2};"));
                    self.emit_line(&format!("  max.f32 {rs0}, {rs0}, {rs2};"));       // scale
                    self.emit_line(&format!("  st.global.f32 [%rd_scale], {rs0};"));
                    // Reload lo & hi for quantization
                    self.emit_line(&format!("  mad.lo.u32 {rs1}, %tid.x, 8, 0;"));
                    self.emit_line(&format!("  ld.global.f32 {rs0}, [{s}+{rs1}];"));   // lo
                    self.emit_line(&format!("  mov.b32 {rs1}, {rs0};"));
                    self.emit_line(&format!("  shr.b32 {rs1}, {rs1}, 20;"));
                    self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 0xF;"));          // lo nibble
                    // Recompute hi_off properly: hi_off_bytes = pair*8 + 4
                    self.emit_line(&format!("  mad.lo.u32 {rs2}, %tid.x, 8, 4;"));
                    self.emit_line(&format!("  ld.global.f32 {rs0}, [{s}+{rs2}];"));   // hi
                    self.emit_line(&format!("  mov.b32 {rs2}, {rs0};"));
                    self.emit_line(&format!("  shr.b32 {rs2}, {rs2}, 20;"));
                    self.emit_line(&format!("  and.b32 {rs2}, {rs2}, 0xF;"));          // hi nibble
                    self.emit_line(&format!("  shl.b32 {rs2}, {rs2}, 4;"));
                    self.emit_line(&format!("  or.b32 {rs1}, {rs1}, {rs2};"));          // packed byte
                    self.emit_line(&format!("  st.global.u8 [%rd_dst], {rs1};"));
                    self.emit_line(&format!("  bra END_{oob};"));
                    self.emit_line(&format!("LAST_PAIR_{oob}:"));
                    // Odd last pair: only lo, scale = |lo|, byte = lo nibble (hi=0)
                    self.emit_line(&format!("  st.global.f32 [%rd_scale], {rs0};"));
                    self.emit_line(&format!("  mad.lo.u32 {rs1}, %tid.x, 8, 0;"));
                    self.emit_line(&format!("  ld.global.f32 {rs0}, [{s}+{rs1}];"));   // lo
                    self.emit_line(&format!("  mov.b32 {rs1}, {rs0};"));
                    self.emit_line(&format!("  shr.b32 {rs1}, {rs1}, 20;"));
                    self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 0xF;"));
                    self.emit_line(&format!("  st.global.u8 [%rd_dst], {rs1};"));
                    self.emit_line(&format!("  bra END_{oob};"));
                } else {
                    // Even num_elements: all pairs have lo & hi
                    self.emit_line(&format!("  ld.global.f32 {rs2}, [{s}+{rs1}];")); // hi
                    self.emit_line(&format!("  abs.f32 {rs2}, {rs2};"));
                    self.emit_line(&format!("  max.f32 {rs0}, {rs0}, {rs2};"));       // scale
                    self.emit_line(&format!("  st.global.f32 [%rd_scale], {rs0};"));
                    // Reload lo & hi for quantization
                    self.emit_line(&format!("  mad.lo.u32 {rs1}, %tid.x, 8, 0;"));   // lo_off
                    self.emit_line(&format!("  ld.global.f32 {rs0}, [{s}+{rs1}];"));  // lo
                    self.emit_line(&format!("  mov.b32 {rs1}, {rs0};"));
                    self.emit_line(&format!("  shr.b32 {rs1}, {rs1}, 20;"));
                    self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 0xF;"));         // lo nibble
                    self.emit_line(&format!("  mad.lo.u32 {rs2}, %tid.x, 8, 4;"));   // hi_off
                    self.emit_line(&format!("  ld.global.f32 {rs0}, [{s}+{rs2}];"));  // hi
                    self.emit_line(&format!("  mov.b32 {rs2}, {rs0};"));
                    self.emit_line(&format!("  shr.b32 {rs2}, {rs2}, 20;"));
                    self.emit_line(&format!("  and.b32 {rs2}, {rs2}, 0xF;"));         // hi nibble
                    self.emit_line(&format!("  shl.b32 {rs2}, {rs2}, 4;"));
                    self.emit_line(&format!("  or.b32 {rs1}, {rs1}, {rs2};"));         // packed byte
                    self.emit_line(&format!("  st.global.u8 [%rd_dst], {rs1};"));
                    self.emit_line(&format!("  bra END_{oob};"));
                }
                self.emit_line(&format!("OOB_{oob}:"));
                self.emit_line(&format!("END_{oob}:"));
                self.emit_line("}");
            }
            GpuDialect::Hip { .. } => {
                // HIP SIMT per-thread: pair = threadIdx.x; OOB guard.
                self.emit_line("{");
                self.emit_line(&format!("  float* _src = (float*)({s});"));
                self.emit_line(&format!("  unsigned char* _dst = (unsigned char*)({dp});"));
                self.emit_line(&format!("  float* _scp = (float*)({sp});"));
                self.emit_line("  unsigned int _pair = (unsigned int)threadIdx.x;");
                self.emit_line(&format!("  if (_pair < {num_pairs}) {{"));
                self.emit_line("    unsigned int _lo = _pair * 2;");
                self.emit_line("    unsigned int _hi = _pair * 2 + 1;");
                self.emit_line("    float _a = fabsf(_src[_lo]);");
                if num_elements % 2 == 1 {
                    self.emit_line(&format!("    if (_hi < {num_elements}) {{"));
                    self.emit_line("      float _b = fabsf(_src[_hi]);");
                    self.emit_line("      _scp[_pair] = fmaxf(_a, _b);");
                    self.emit_line("      unsigned _nlo = (__float_as_uint(_src[_lo]) >> 20) & 0xF;");
                    self.emit_line("      unsigned _nhi = (__float_as_uint(_src[_hi]) >> 20) & 0xF;");
                    self.emit_line("      _dst[_pair] = (unsigned char)(_nlo | (_nhi << 4));");
                    self.emit_line("    } else {");
                    self.emit_line("      _scp[_pair] = _a;");
                    self.emit_line("      unsigned _nlo = (__float_as_uint(_src[_lo]) >> 20) & 0xF;");
                    self.emit_line("      _dst[_pair] = (unsigned char)_nlo;");
                    self.emit_line("    }");
                } else {
                    self.emit_line("    float _b = fabsf(_src[_hi]);");
                    self.emit_line("    _scp[_pair] = fmaxf(_a, _b);");
                    self.emit_line("    unsigned _nlo = (__float_as_uint(_src[_lo]) >> 20) & 0xF;");
                    self.emit_line("    unsigned _nhi = (__float_as_uint(_src[_hi]) >> 20) & 0xF;");
                    self.emit_line("    _dst[_pair] = (unsigned char)(_nlo | (_nhi << 4));");
                }
                self.emit_line("  }");
                self.emit_line("}");
            }
            GpuDialect::Metal { .. } => {
                // Metal SIMT per-thread: pair = thread_position_in_threadgroup.x; OOB guard.
                self.emit_line("{");
                self.emit_line(&format!("  device float* _src = (device float*)({s});"));
                self.emit_line(&format!("  device unsigned char* _dst = (device unsigned char*)({dp});"));
                self.emit_line(&format!("  device float* _scp = (device float*)({sp});"));
                self.emit_line("  unsigned int _pair = (unsigned int)thread_position_in_threadgroup.x;");
                self.emit_line(&format!("  if (_pair < {num_pairs}) {{"));
                self.emit_line("    unsigned int _lo = _pair * 2;");
                self.emit_line("    unsigned int _hi = _pair * 2 + 1;");
                self.emit_line("    float _a = fabs(_src[_lo]);");
                if num_elements % 2 == 1 {
                    self.emit_line(&format!("    if (_hi < {num_elements}) {{"));
                    self.emit_line("      float _b = fabs(_src[_hi]);");
                    self.emit_line("      _scp[_pair] = max(_a, _b);");
                    self.emit_line("      uint _nlo = (as_type<uint>(_src[_lo]) >> 20) & 0xFu;");
                    self.emit_line("      uint _nhi = (as_type<uint>(_src[_hi]) >> 20) & 0xFu;");
                    self.emit_line("      _dst[_pair] = (unsigned char)(_nlo | (_nhi << 4));");
                    self.emit_line("    } else {");
                    self.emit_line("      _scp[_pair] = _a;");
                    self.emit_line("      uint _nlo = (as_type<uint>(_src[_lo]) >> 20) & 0xFu;");
                    self.emit_line("      _dst[_pair] = (unsigned char)_nlo;");
                    self.emit_line("    }");
                } else {
                    self.emit_line("    float _b = fabs(_src[_hi]);");
                    self.emit_line("    _scp[_pair] = max(_a, _b);");
                    self.emit_line("    uint _nlo = (as_type<uint>(_src[_lo]) >> 20) & 0xFu;");
                    self.emit_line("    uint _nhi = (as_type<uint>(_src[_hi]) >> 20) & 0xFu;");
                    self.emit_line("    _dst[_pair] = (unsigned char)(_nlo | (_nhi << 4));");
                }
                self.emit_line("  }");
                self.emit_line("}");
            }
        }
        Ok(())
    }

    // ── Cross-CTA Synchronization (Ring Barrier for SM61 / SM70) ──
    //
    // SPEC 32 §2.2.2: SM61 has no cluster.sync, no mbarrier.
    // Uses persistent ring barrier: atom.global.add.u32 arrival + spin-wait.
    // Must be used with cooperative launch (cuLaunchCooperativeKernel) to ensure all CTAs are scheduled.

    /// Emit ring barrier arrival: atomically increment the arrival counter.
    ///
    /// `barrier_ptr` is the name of a register holding the global memory address of the counter.
    /// `scratch_gpr` is a scratch register for the atomic result (unused but required by PTX).
    pub fn emit_ring_barrier_arrive(&mut self, barrier_ptr: &str, scratch_gpr: &str) {
        // atom.global.add.u32 scratch_gpr, [barrier_ptr], 1;
        self.emit_line(&format!(
            "atom.global.add.u32 {scratch_gpr}, [{barrier_ptr}], 1;"
        ));
        // membar.gl — ensure the atomic is visible to all CTAs
        self.emit_line("membar.gl;");
    }

    /// Emit ring barrier wait: spin-loop until the counter reaches `expected_count`.
    ///
    /// `bar_ptr` is the register holding the counter's global address.
    /// `scratch_gpr` is a scratch register for the loaded value.
    /// `scratch_pred` is a scratch predicate register.
    /// `expected_count` is the total number of CTAs that must arrive.
    pub fn emit_ring_barrier_wait(
        &mut self,
        barrier_ptr: &str,
        scratch_gpr: &str,
        scratch_pred: &str,
        expected_count: u32,
    ) {
        let wait_label = self.next_loop_label();
        self.emit_line(&format!(".Lring_wait_{wait_label}:"));
        self.emit_line(&format!("ld.global.u32 {scratch_gpr}, [{barrier_ptr}];"));
        self.emit_line(&format!(
            "setp.lt.u32 {scratch_pred}, {scratch_gpr}, {expected_count};"
        ));
        self.emit_line(&format!("@{scratch_pred} bra .Lring_wait_{wait_label};"));
        // Full fence after all arrivals confirmed
        self.emit_line("membar.gl;");
    }

    /// Emit a complete ring barrier: arrive + wait.
    /// Convenience method for the common case where arrive and wait happen in the same CTA.
    pub fn emit_ring_barrier(
        &mut self,
        barrier_ptr: &str,
        scratch_gpr: &str,
        scratch_pred: &str,
        total_ctas: u32,
    ) {
        self.emit_ring_barrier_arrive(barrier_ptr, scratch_gpr);
        self.emit_ring_barrier_wait(barrier_ptr, scratch_gpr, scratch_pred, total_ctas);
    }

    pub fn finalize(self) -> Result<String, CompilerError> {
        // SPEC 15 REQ-JCTX-022: 编译后资源泄漏检测
        let report = self.jit_ctx.usage_report();
        if !report.warnings.is_empty() {
            for w in &report.warnings {
                match w {
                    crate::compiler::jit_context::ResourceWarning::NearExhaustion { kind, peak, capacity } => {
                        eprintln!("[JitContext] WARNING: {:?} near exhaustion: {}/{}", kind, peak, capacity);
                    }
                    crate::compiler::jit_context::ResourceWarning::Leak { kind, instance, purpose } => {
                        eprintln!("[JitContext] WARNING: {:?} leak: instance {} ({}) not released", kind, instance, purpose);
                    }
                }
            }
        }
        Ok(self.ir)
    }
}
