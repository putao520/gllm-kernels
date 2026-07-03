impl GpuLower {
    fn lower_quant_block_load(&mut self, dst: VRegId, base: VRegId, offset: &OffsetExpr, unpack: &BlockUnpackMode, width: SimdWidth, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match unpack {
            BlockUnpackMode::Int8 => {
                // Load INT8 bytes, sign-extend to F32.
                // GPU SIMT per-thread model (与 VecLoad/KiviDequantLoad 一致):
                //   每个线程解码 1 个元素 (lane = tid.x), 写单个 f32 到 {d} (无覆盖)。
                //   tid → byte_idx = tid; byte = base[off + tid]; result = (float)(s8)byte → {d}
                //   边界: tid >= lanes → {d} = 0.0 (OOB 守卫, 避免越界读)
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // %rs0 — tid / byte_idx
                        let rs1 = self.scratch_gpr_names[1]; // %rs1 — s32 byte
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_addr;");
                        self.emit_line("  .reg .pred %p_oob;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        // addr = base + off + tid (64-bit; PTX [reg+imm] 合法, off 作为 imm 加到 u64)
                        self.emit_line(&format!("  cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {off};"));
                        self.emit_line(&format!("  ld.global.s8 {rs1}, [%rd_addr];"));
                        self.emit_line(&format!("  cvt.rn.f32.s32 {d}, {rs1};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const signed char* _bp = (const signed char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line(&format!("    {d} = (float)(_bp[_t]);"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const device signed char* _bp = (const device signed char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line(&format!("    {d} = (float)(_bp[_t]);"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::F16Broadcast => {
                // Load one F16 from base+offset, convert to F32, broadcast to all lanes.
                // 单元素加载, 不涉及 lanes 维度, 保持原实现 (无 for 循环, 无覆盖问题)。
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // Load b16 (f16), cvt to f32, broadcast via mov.
                        let rs0 = self.scratch_gpr_names[0];
                        self.emit_line(&format!("{{ .reg .b16 %tmp_h16; ld.global.u16 %tmp_h16, [{b}+{off}]; cvt.rn.f32.f16 {d}, %tmp_h16; }}"));
                        let _ = rs0;
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line(&format!("{d} = __half2float(*((__half*)({b}+({off})));"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = (float)(*((device half*)({b}+({off})));"));
                    }
                }
                Ok(())
            }
            BlockUnpackMode::SignedNibbleLow => {
                // Q4_0 4-bit packed dequant (symmetric zero-point = 8.0).
                // GPU SIMT per-thread model (BCE-20260711-GPU-REG-OVERWRITE 根治):
                //   每个线程解码 1 个 nibble (lane = tid.x), 写单个 f32 到 {d} (无覆盖)。
                //   byte_idx = tid >> 1; is_high = tid & 1
                //   nibble = is_high ? (byte >> 4) & 0xF : byte & 0xF
                //   result = (float)(nibble) - 8.0 → {d}
                //   边界: tid >= lanes → {d} = 0.0
                //
                // 旧 bug: for lane in 0..lanes 循环展开 PTX, 每个 lane 的 cvt+sub 写同一 {d},
                //   后覆盖前 → 只剩 lane[lanes-1] 值。根治: per-thread SIMT, 单线程单 {d}。
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // %rs0 — tid / byte_idx / byte
                        let rs1 = self.scratch_gpr_names[1]; // %rs1 — is_high
                        let fs0 = self.scratch_vec_names[0]; // %fs0 — bias 8.0
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_addr;");
                        self.emit_line("  .reg .pred %p_oob, %p_hi;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        // byte_idx = tid >> 1
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 1;"));
                        // addr = base + off + byte_idx (64-bit)
                        self.emit_line(&format!("  cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {off};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_addr];")); // byte (rs0 不再是 byte_idx)
                        // is_high = tid & 1 (rs0 已被 byte 占用, 用 rs1 重新读 tid)
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 1;"));
                        self.emit_line(&format!("  setp.eq.u32 %p_hi, {rs1}, 1;"));
                        self.emit_line(&format!("  @%p_hi shr.u32 {rs0}, {rs0}, 4;"));
                        self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 0xF;"));
                        // cvt + sub 8.0 (单线程单 {d}, 无覆盖)
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line("  mov.u32 %r_bias, 0x41000000;"); // 8.0f
                        self.emit_line(&format!("  mov.f32 {fs0}, %r_bias;"));
                        self.emit_line(&format!("  sub.rn.f32 {d}, {d}, {fs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line("  const float _bias = 8.0f;");
                        self.emit_line(&format!("  const unsigned char* _bp = (const unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _bi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _byte = _bp[_bi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_byte >> 4) & 0xF) : (_byte & 0xF));");
                        self.emit_line(&format!("    {d} = (float)_nib - _bias;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line("  constant float _bias = 8.0f;");
                        self.emit_line(&format!("  const device unsigned char* _bp = (const device unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _bi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _byte = _bp[_bi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_byte >> 4) & 0xF) : (_byte & 0xF));");
                        self.emit_line(&format!("    {d} = (float)_nib - _bias;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::SignedNibbleHigh => {
                // GGUF PackedNibbles high-nibble: extract (byte >> 4), subtract 8, convert to F32.
                // GPU SIMT per-thread (BCE-20260711 根治): lane = tid.x, 1 byte/thread → 1 nibble → {d}。
                //   byte_idx = tid; nibble = (byte[tid] >> 4) & 0xF; result = (float)nibble - 8.0
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // tid / byte
                        let fs0 = self.scratch_vec_names[0]; // bias 8.0
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_addr;");
                        self.emit_line("  .reg .pred %p_oob;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        self.emit_line(&format!("  cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {off};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_addr];"));
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 4;"));
                        self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 0xF;"));
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line("  mov.u32 %r_bias, 0x41000000;");
                        self.emit_line(&format!("  mov.f32 {fs0}, %r_bias;"));
                        self.emit_line(&format!("  sub.rn.f32 {d}, {d}, {fs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line("  const float _bias = 8.0f;");
                        self.emit_line(&format!("  const unsigned char* _bp = (const unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line(&format!("    {d} = (float)(_bp[_t] >> 4) - _bias;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line("  constant float _bias = 8.0f;");
                        self.emit_line(&format!("  const device unsigned char* _bp = (const device unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line(&format!("    {d} = (float)(_bp[_t] >> 4) - _bias;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::UnsignedNibbleLow => {
                // Unsigned 4-bit low-nibble (Q4_1): extract (& 0x0F), NO subtract-8.
                // GPU SIMT per-thread (BCE-20260711 根治): lane = tid.x, byte_idx = tid>>1, is_high = tid&1。
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // tid / byte_idx / byte
                        let rs1 = self.scratch_gpr_names[1]; // is_high
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_addr;");
                        self.emit_line("  .reg .pred %p_oob, %p_hi;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 1;")); // byte_idx
                        self.emit_line(&format!("  cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {off};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_addr];"));
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 1;"));
                        self.emit_line(&format!("  setp.eq.u32 %p_hi, {rs1}, 1;"));
                        self.emit_line(&format!("  @%p_hi shr.u32 {rs0}, {rs0}, 4;"));
                        self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 0xF;"));
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const unsigned char* _bp = (const unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _bi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _byte = _bp[_bi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_byte >> 4) & 0xF) : (_byte & 0xF));");
                        self.emit_line(&format!("    {d} = (float)_nib;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const device unsigned char* _bp = (const device unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _bi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _byte = _bp[_bi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_byte >> 4) & 0xF) : (_byte & 0xF));");
                        self.emit_line(&format!("    {d} = (float)_nib;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::UnsignedNibbleHigh => {
                // Unsigned 4-bit high-nibble (Q4_1): extract (>>4), NO subtract-8.
                // GPU SIMT per-thread (BCE-20260711 根治): lane = tid.x, 1 byte/thread。
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // tid / byte
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_addr;");
                        self.emit_line("  .reg .pred %p_oob;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        self.emit_line(&format!("  cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {b};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {off};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_addr];"));
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 4;"));
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const unsigned char* _bp = (const unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line(&format!("    {d} = (float)(_bp[_t] >> 4);"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const device unsigned char* _bp = (const device unsigned char*)({b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line(&format!("    {d} = (float)(_bp[_t] >> 4);"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::Bitpack2 { bias } => {
                // Q2K 2-bit packed: each byte has 4 × 2-bit values.
                // GPU SIMT per-thread (BCE-20260711 根治): lane = tid.x,
                //   byte_idx = tid >> 2; bit_shift = (tid & 3) * 2
                //   val = (byte >> bit_shift) & 3; result = (float)val - bias → {d}
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(*bias);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // tid / byte_idx / byte
                        let rs1 = self.scratch_gpr_names[1]; // bit_shift
                        let fs0 = self.scratch_vec_names[0]; // bias
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_addr;");
                        self.emit_line("  .reg .pred %p_oob;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        // byte_idx = tid >> 2; bit_shift = (tid & 3) * 2
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 2;")); // byte_idx
                        self.emit_line(&format!("  cvt.u64.u32 %rd_addr, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {qs_b};"));
                        self.emit_line(&format!("  add.u64 %rd_addr, %rd_addr, {off};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_addr];")); // byte
                        // bit_shift = (tid & 3) * 2 — 重新读 tid
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 3;"));
                        self.emit_line(&format!("  shl.u32 {rs1}, {rs1}, 1;")); // ×2
                        self.emit_line(&format!("  shr.b32 {rs0}, {rs0}, {rs1};"));
                        self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 3;"));
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line(&format!("  mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("  mov.f32 {fs0}, {rs0};"));
                        self.emit_line(&format!("  sub.rn.f32 {d}, {d}, {fs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const unsigned char* _bp = (const unsigned char*)({qs_b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _bi = _t >> 2;");
                        self.emit_line("    unsigned int _sh = (_t & 3u) * 2u;");
                        self.emit_line(&format!("    {d} = (float)((_bp[_bi] >> _sh) & 3u) - {bias_bits}_f32;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const device unsigned char* _bp = (const device unsigned char*)({qs_b}) + ({off});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _bi = _t >> 2;");
                        self.emit_line("    unsigned int _sh = (_t & 3u) * 2u;");
                        self.emit_line(&format!("    {d} = (float)((_bp[_bi] >> _sh) & 3u) - {bias_bits}_f32;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::Mxfp4 { scale_src } => {
                self.emit_mxfp4_dequant_gpu(dst, base, offset, *scale_src, width, alloc)
            }
            BlockUnpackMode::Nvfp4 { scale_src } => {
                self.emit_nvfp4_sub_block_dequant_gpu(dst, base, offset, *scale_src, width, alloc)
            }
            BlockUnpackMode::QhBitExpand { .. } => {
                Err(CompilerError::CodegenViolation("QhBitExpand: GPU not yet implemented".into()))
            }
        }
    }

    fn lower_quant_biplane_load(&mut self, dst: VRegId, qs_base: VRegId, extra_base: VRegId, bias: f32, mode: &BiPlaneMode, width: SimdWidth, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match mode {
            BiPlaneMode::Low5 => {
                // GGUF Q5_0/Q5_1: qs (4-bit low nibbles) + qh (1-bit high plane).
                // GPU SIMT per-thread (BCE-20260711 根治): lane = tid.x,
                //   qs_byte_idx = tid >> 1; is_high = tid & 1
                //   qh_byte_idx = tid >> 8; qh_bit = tid & 7
                //   value = (nibble | (qh_bit << 4)) - bias → {d}
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(qs_base, alloc);
                let qh_b = self.reg_name_with_kind(extra_base, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(bias);
                let qs_off_str = String::new(); // BiPlane offset 内嵌在 base 计算 (无独立 offset 参数)
                let _ = qs_off_str;
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // qs byte_idx / qs byte / merged
                        let rs1 = self.scratch_gpr_names[1]; // tid / is_high / qh byte
                        let fs0 = self.scratch_vec_names[0]; // bias
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_q, %rd_h;");
                        self.emit_line("  .reg .pred %p_oob, %p_hi;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        // qs_byte_idx = tid >> 1
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 1;"));
                        // load qs byte
                        self.emit_line(&format!("  cvt.u64.u32 %rd_q, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_q, %rd_q, {qs_b};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_q];"));
                        // is_high = tid & 1
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 1;"));
                        self.emit_line(&format!("  setp.eq.u32 %p_hi, {rs1}, 1;"));
                        self.emit_line(&format!("  @%p_hi shr.u32 {rs0}, {rs0}, 4;"));
                        self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 0xF;"));
                        // qh_byte_idx = tid >> 8; qh_bit = tid & 7
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  shr.u32 {rs1}, {rs1}, 8;")); // qh_byte_idx
                        self.emit_line(&format!("  cvt.u64.u32 %rd_h, {rs1};"));
                        self.emit_line(&format!("  add.u64 %rd_h, %rd_h, {qh_b};"));
                        self.emit_line(&format!("  ld.global.u8 {rs1}, [%rd_h];")); // qh byte
                        // qh_bit = tid & 7
                        self.emit_line(&format!("  mov.u32 {fs0}, %tid.x;"));
                        // 借用 fs0 的 u32 位作临时 (PTX 允许 .b32 重解释 mov.u32)
                        // 为避免污染 f32 bias, 用 rs1 的位先存 qh_bit: 改用 %r_tmp
                        self.emit_line("  .reg .u32 %r_qhbit, %r_qhval;");
                        self.emit_line("  mov.u32 %r_qhbit, %tid.x;");
                        self.emit_line("  and.b32 %r_qhbit, %r_qhbit, 7;");
                        self.emit_line("  shr.u32 {rs1}, {rs1}, %r_qhbit;");
                        self.emit_line("  and.b32 %r_qhval, {rs1}, 1;");
                        // merge: nibble | (qh_bit << 4)
                        self.emit_line("  shl.b32 %r_qhval, %r_qhval, 4;");
                        self.emit_line(&format!("  or.b32 {rs0}, {rs0}, %r_qhval;"));
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line(&format!("  mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("  mov.f32 {fs0}, {rs0};"));
                        self.emit_line(&format!("  sub.rn.f32 {d}, {d}, {fs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const float _bias = {bias_bits}_f32;"));
                        self.emit_line(&format!("  const unsigned char* _qs = (const unsigned char*)({qs_b});"));
                        self.emit_line(&format!("  const unsigned char* _qh = (const unsigned char*)({qh_b});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _qbi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _qb = _qs[_qbi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_qb >> 4) & 0xF) : (_qb & 0xF));");
                        self.emit_line("    unsigned int _hbi = _t >> 8;");
                        self.emit_line("    unsigned int _hbit = _t & 7u;");
                        self.emit_line("    int _qhv = (int)((_qh[_hbi] >> _hbit) & 1u);");
                        self.emit_line(&format!("    {d} = (float)(_nib | (_qhv << 4)) - _bias;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  constant float _bias = {bias_bits}_f32;"));
                        self.emit_line(&format!("  const device unsigned char* _qs = (const device unsigned char*)({qs_b});"));
                        self.emit_line(&format!("  const device unsigned char* _qh = (const device unsigned char*)({qh_b});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _qbi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _qb = _qs[_qbi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_qb >> 4) & 0xF) : (_qb & 0xF));");
                        self.emit_line("    unsigned int _hbi = _t >> 8;");
                        self.emit_line("    unsigned int _hbit = _t & 7u;");
                        self.emit_line("    int _qhv = (int)((_qh[_hbi] >> _hbit) & 1u);");
                        self.emit_line(&format!("    {d} = (float)(_nib | (_qhv << 4)) - _bias;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BiPlaneMode::Low6 => {
                // GGUF Q6K: qs (4-bit low) + qh (2-bit high plane).
                // GPU SIMT per-thread (BCE-20260711 根治): lane = tid.x,
                //   qs_byte_idx = tid >> 1; is_high = tid & 1
                //   qh_byte_idx = tid >> 2; qh_bit_shift = (tid & 3) * 2
                //   value = (nibble | (qh_2bit << 4)) - bias → {d}
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(qs_base, alloc);
                let qh_b = self.reg_name_with_kind(extra_base, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(bias);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // qs byte_idx / qs byte / merged
                        let rs1 = self.scratch_gpr_names[1]; // tid / qh byte / qh_2bit
                        let fs0 = self.scratch_vec_names[0]; // bias
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_q, %rd_h;");
                        self.emit_line("  .reg .pred %p_oob, %p_hi;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 1;")); // qs_byte_idx
                        self.emit_line(&format!("  cvt.u64.u32 %rd_q, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_q, %rd_q, {qs_b};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_q];"));
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  and.b32 {rs1}, {rs1}, 1;"));
                        self.emit_line(&format!("  setp.eq.u32 %p_hi, {rs1}, 1;"));
                        self.emit_line(&format!("  @%p_hi shr.u32 {rs0}, {rs0}, 4;"));
                        self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 0xF;"));
                        // qh_byte_idx = tid >> 2; qh_bit_shift = (tid & 3) * 2
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  shr.u32 {rs1}, {rs1}, 2;")); // qh_byte_idx
                        self.emit_line(&format!("  cvt.u64.u32 %rd_h, {rs1};"));
                        self.emit_line(&format!("  add.u64 %rd_h, %rd_h, {qh_b};"));
                        self.emit_line(&format!("  ld.global.u8 {rs1}, [%rd_h];")); // qh byte
                        self.emit_line("  .reg .u32 %r_sh, %r_qhv;");
                        self.emit_line("  mov.u32 %r_sh, %tid.x;");
                        self.emit_line("  and.b32 %r_sh, %r_sh, 3;");
                        self.emit_line("  shl.b32 %r_sh, %r_sh, 1;"); // ×2
                        self.emit_line(&format!("  shr.u32 {rs1}, {rs1}, %r_sh;"));
                        self.emit_line(&format!("  and.b32 %r_qhv, {rs1}, 3;"));
                        self.emit_line("  shl.b32 %r_qhv, %r_qhv, 4;");
                        self.emit_line(&format!("  or.b32 {rs0}, {rs0}, %r_qhv;"));
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line(&format!("  mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("  mov.f32 {fs0}, {rs0};"));
                        self.emit_line(&format!("  sub.rn.f32 {d}, {d}, {fs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const unsigned char* _qs = (const unsigned char*)({qs_b});"));
                        self.emit_line(&format!("  const unsigned char* _qh = (const unsigned char*)({qh_b});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _qbi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _qb = _qs[_qbi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_qb >> 4) & 0xF) : (_qb & 0xF));");
                        self.emit_line("    unsigned int _hbi = _t >> 2;");
                        self.emit_line("    unsigned int _hsh = (_t & 3u) * 2u;");
                        self.emit_line("    int _qh2 = (int)((_qh[_hbi] >> _hsh) & 3u);");
                        self.emit_line(&format!("    {d} = (float)(_nib | (_qh2 << 4)) - {bias_bits}_f32;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const device unsigned char* _qs = (const device unsigned char*)({qs_b});"));
                        self.emit_line(&format!("  const device unsigned char* _qh = (const device unsigned char*)({qh_b});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _qbi = _t >> 1;");
                        self.emit_line("    unsigned int _hi = _t & 1u;");
                        self.emit_line("    unsigned char _qb = _qs[_qbi];");
                        self.emit_line("    int _nib = (int)(_hi ? ((_qb >> 4) & 0xF) : (_qb & 0xF));");
                        self.emit_line("    unsigned int _hbi = _t >> 2;");
                        self.emit_line("    unsigned int _hsh = (_t & 3u) * 2u;");
                        self.emit_line("    int _qh2 = (int)((_qh[_hbi] >> _hsh) & 3u);");
                        self.emit_line(&format!("    {d} = (float)(_nib | (_qh2 << 4)) - {bias_bits}_f32;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BiPlaneMode::Q3Merge => {
                // Q3K 3-bit: qs(2-bit) + hmask(1-bit) merge.
                // GPU SIMT per-thread (BCE-20260711 根治): lane = tid.x,
                //   qs_byte_idx = tid >> 2; qs_bit_shift = (tid & 3) * 2
                //   hmask_byte_idx = tid >> 8; hmask_bit = tid & 7
                //   value = (qs_2bit | (hmask_bit << 2)) - bias → {d}
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(qs_base, alloc);
                let hmask_b = self.reg_name_with_kind(extra_base, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(bias);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // qs byte_idx / qs byte / merged
                        let rs1 = self.scratch_gpr_names[1]; // tid / hmask byte / hmask_bit
                        let fs0 = self.scratch_vec_names[0]; // bias
                        let oob = self.next_skip_label();
                        let done = self.next_skip_label();
                        self.emit_line("{");
                        self.emit_line("  .reg .u64 %rd_q, %rd_h;");
                        self.emit_line("  .reg .pred %p_oob;");
                        self.emit_line(&format!("  mov.u32 {rs0}, %tid.x;"));
                        self.emit_line(&format!("  setp.ge.u32 %p_oob, {rs0}, {lanes};"));
                        self.emit_line(&format!("  @%p_oob bra OOB_{oob};"));
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, 2;")); // qs_byte_idx
                        self.emit_line(&format!("  cvt.u64.u32 %rd_q, {rs0};"));
                        self.emit_line(&format!("  add.u64 %rd_q, %rd_q, {qs_b};"));
                        self.emit_line(&format!("  ld.global.u8 {rs0}, [%rd_q];"));
                        // qs_bit_shift = (tid & 3) * 2
                        self.emit_line("  .reg .u32 %r_sh;");
                        self.emit_line("  mov.u32 %r_sh, %tid.x;");
                        self.emit_line("  and.b32 %r_sh, %r_sh, 3;");
                        self.emit_line("  shl.b32 %r_sh, %r_sh, 1;");
                        self.emit_line(&format!("  shr.u32 {rs0}, {rs0}, %r_sh;"));
                        self.emit_line(&format!("  and.b32 {rs0}, {rs0}, 3;"));
                        // hmask_byte_idx = tid >> 8; hmask_bit = tid & 7
                        self.emit_line(&format!("  mov.u32 {rs1}, %tid.x;"));
                        self.emit_line(&format!("  shr.u32 {rs1}, {rs1}, 8;"));
                        self.emit_line(&format!("  cvt.u64.u32 %rd_h, {rs1};"));
                        self.emit_line(&format!("  add.u64 %rd_h, %rd_h, {hmask_b};"));
                        self.emit_line(&format!("  ld.global.u8 {rs1}, [%rd_h];"));
                        self.emit_line("  mov.u32 %r_sh, %tid.x;");
                        self.emit_line("  and.b32 %r_sh, %r_sh, 7;");
                        self.emit_line(&format!("  shr.u32 {rs1}, {rs1}, %r_sh;"));
                        self.emit_line("  .reg .u32 %r_hv;");
                        self.emit_line("  and.b32 %r_hv, {rs1}, 1;");
                        self.emit_line("  shl.b32 %r_hv, %r_hv, 2;");
                        self.emit_line(&format!("  or.b32 {rs0}, {rs0}, %r_hv;"));
                        self.emit_line(&format!("  cvt.rn.f32.u32 {d}, {rs0};"));
                        self.emit_line(&format!("  mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("  mov.f32 {fs0}, {rs0};"));
                        self.emit_line(&format!("  sub.rn.f32 {d}, {d}, {fs0};"));
                        self.emit_line(&format!("  bra DONE_{done};"));
                        self.emit_line(&format!("OOB_{oob}:"));
                        self.emit_line(&format!("  mov.f32 {d}, 0f00000000;"));
                        self.emit_line(&format!("DONE_{done}:"));
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const unsigned char* _qs = (const unsigned char*)({qs_b});"));
                        self.emit_line(&format!("  const unsigned char* _hm = (const unsigned char*)({hmask_b});"));
                        self.emit_line("  unsigned int _t = (unsigned int)threadIdx.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _qbi = _t >> 2;");
                        self.emit_line("    unsigned int _qsh = (_t & 3u) * 2u;");
                        self.emit_line("    int _qs2 = (int)((_qs[_qbi] >> _qsh) & 3u);");
                        self.emit_line("    unsigned int _hbi = _t >> 8;");
                        self.emit_line("    unsigned int _hbit = _t & 7u;");
                        self.emit_line("    int _hv = (int)((_hm[_hbi] >> _hbit) & 1u);");
                        self.emit_line(&format!("    {d} = (float)(_qs2 | (_hv << 2)) - {bias_bits}_f32;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const device unsigned char* _qs = (const device unsigned char*)({qs_b});"));
                        self.emit_line(&format!("  const device unsigned char* _hm = (const device unsigned char*)({hmask_b});"));
                        self.emit_line("  unsigned int _t = (unsigned int)thread_position_in_threadgroup.x;");
                        self.emit_line(&format!("  if (_t < {lanes}) {{"));
                        self.emit_line("    unsigned int _qbi = _t >> 2;");
                        self.emit_line("    unsigned int _qsh = (_t & 3u) * 2u;");
                        self.emit_line("    int _qs2 = (int)((_qs[_qbi] >> _qsh) & 3u);");
                        self.emit_line("    unsigned int _hbi = _t >> 8;");
                        self.emit_line("    unsigned int _hbit = _t & 7u;");
                        self.emit_line("    int _hv = (int)((_hm[_hbi] >> _hbit) & 1u);");
                        self.emit_line(&format!("    {d} = (float)(_qs2 | (_hv << 2)) - {bias_bits}_f32;"));
                        self.emit_line("  } else {");
                        self.emit_line(&format!("    {d} = 0.0f;"));
                        self.emit_line("  }");
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
        }
    }
}
