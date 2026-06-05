impl GpuLower {
    fn lower_quant_block_load(&mut self, dst: VRegId, base: VRegId, offset: &OffsetExpr, unpack: &BlockUnpackMode, width: SimdWidth, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match unpack {
            BlockUnpackMode::Int8 => {
                // Load `lanes` INT8 bytes, sign-extend to F32.
                // PTX: ld.global.s8 + cvt.rn.f32.s32 per lane.
                // HIP/Metal: (float)(signed char) cast.
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // PTX scalar-per-thread: emit per-lane load + sign-extend + cvt.
                        // Each lane loads one s8 from base + offset + lane.
                        let rs0 = self.scratch_gpr_names[0]; // %rs0 — byte/s32 temp
                        for lane in 0..lanes {
                            let byte_off = if lane > 0 { format!("+{}", lane) } else { String::new() };
                            // Load s8 (PTX sign-extends s8 → s32 in .s32 register)
                            self.emit_line(&format!("ld.global.s8 {rs0}, [{b}+{off}{byte_off}];"));
                            // Convert s32 → f32
                            if lane == 0 {
                                self.emit_line(&format!("cvt.rn.f32.s32 {d}, {rs0};"));
                            } else {
                                // Subsequent lanes need separate naming —
                                // GPU SIMT: each thread holds one f32 per VReg.
                                // For >1 lane the compiler emits per-thread f32 values.
                                // Use array indexing notation: dst is conceptually a register array.
                                self.emit_line(&format!("cvt.rn.f32.s32 {d}, {rs0};  // lane {lane}"));
                            }
                        }
                        if lanes == 0 {
                            // Single element fallback
                            self.emit_line(&format!("ld.global.s8 {rs0}, [{b}+{off}];"));
                            self.emit_line(&format!("cvt.rn.f32.s32 {d}, {rs0};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        // HIP: each thread loads lanes bytes, converts to float.
                        if lanes == 1 {
                            self.emit_line(&format!("{d} = (float)(*((signed char*)({b}+({off})));"));
                        } else {
                            self.emit_line(&format!("for (int _li = 0; _li < {lanes}; ++_li) {{"));
                            self.emit_line(&format!("  {d} = (float)(((signed char*)({b}+({off})))[_li]);"));
                            self.emit_line("}");
                        }
                    }
                    GpuDialect::Metal { .. } => {
                        if lanes == 1 {
                            self.emit_line(&format!("{d} = (float)(*((device signed char*)({b}+({off})));"));
                        } else {
                            self.emit_line(&format!("for (int _li = 0; _li < {lanes}; ++_li) {{"));
                            self.emit_line(&format!("  {d} = (float)(((device signed char*)({b}+({off})))[_li]);"));
                            self.emit_line("}");
                        }
                    }
                }
                Ok(())
            }
            BlockUnpackMode::F16Broadcast => {
                // Load one F16 from base+offset, convert to F32, broadcast to all lanes.
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
                        self.emit_line(&format!("{d} = __half2float(*((__half*)({b}+({off}))));"));
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line(&format!("{d} = (float)(*((device half*)({b}+({off}))));"));
                    }
                }
                Ok(())
            }
            BlockUnpackMode::SignedNibbleLow => {
                // Load packed 4-bit values, unpack nibbles, subtract 8, convert to F32.
                // Each byte has 2 nibbles: low = byte & 0xF, high = (byte >> 4).
                // Output value = (nibble - 8.0) per Q4_0 symmetric zero-point.
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        // PTX scalar-per-thread: emit per-nibble decode.
                        // lanes/2 bytes produce lanes nibbles.
                        // lane i → byte[i/2], nibble = i%2 (low=0, high=1)
                        let rs0 = self.scratch_gpr_names[0]; // %rs0 — byte / nibble temp
                        let fs0 = self.scratch_vec_names[0]; // %fs0 — f32 bias (8.0)
                        // Emit the -8.0 constant once
                        // 8.0f32 = 0x41000000
                        self.emit_line("{ .reg .u32 %tmp_bias_u32;");
                        self.emit_line("mov.u32 %tmp_bias_u32, 0x41000000;");
                        self.emit_line(&format!("mov.f32 {fs0}, %tmp_bias_u32;"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            // Load byte
                            let byte_off = if byte_idx > 0 { format!("+{}", byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{b}+{off}{byte_off}];"));
                            // Extract nibble
                            if is_high {
                                self.emit_line(&format!("shr.b32 {rs0}, {rs0}, 4;"));
                            }
                            self.emit_line(&format!("and.b32 {rs0}, {rs0}, 0xF;"));
                            // Convert u32 nibble to f32, subtract 8.0
                            if lane == 0 {
                                self.emit_line(&format!("cvt.rn.f32.u32 {d}, {rs0};"));
                                self.emit_line(&format!("sub.rn.f32 {d}, {d}, {fs0};"));
                            } else {
                                self.emit_line(&format!("cvt.rn.f32.u32 {d}, {rs0};  // nibble lane {lane}"));
                                self.emit_line(&format!("sub.rn.f32 {d}, {d}, {fs0};"));
                            }
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line("  const float _bias = 8.0f;");
                        self.emit_line(&format!("  unsigned char* _bp = (unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let nibble_expr = if is_high {
                                format!("(_bp[{byte_idx}] >> 4) & 0xF")
                            } else {
                                format!("_bp[{byte_idx}] & 0xF")
                            };
                            self.emit_line(&format!("  {d} = (float)({nibble_expr}) - _bias;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line("  constant float _bias = 8.0f;");
                        self.emit_line(&format!("  device unsigned char* _bp = (device unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let nibble_expr = if is_high {
                                format!("(_bp[{byte_idx}] >> 4) & 0xF")
                            } else {
                                format!("_bp[{byte_idx}] & 0xF")
                            };
                            self.emit_line(&format!("  {d} = (float)({nibble_expr}) - _bias;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::SignedNibbleHigh => {
                // GGUF PackedNibbles high-nibble load: extract (byte >> 4), subtract 8, convert to F32.
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0];
                        let fs0 = self.scratch_vec_names[0];
                        self.emit_line("{ .reg .u32 %tmp_bias_u32;");
                        self.emit_line("mov.u32 %tmp_bias_u32, 0x41000000;");
                        self.emit_line(&format!("mov.f32 {fs0}, %tmp_bias_u32;"));
                        for lane in 0..lanes {
                            self.emit_line(&format!("ld.u8 {rs0}, [{b}+{off}+{lane}];"));
                            self.emit_line(&format!("shr.u32 {rs0}, {rs0}, 4;"));
                            self.emit_line(&format!("cvt.rn.f32.u32 {d}<{}>, {rs0};", lane));
                            self.emit_line(&format!("sub.f32 {d}<{}>, {d}<{}>, {fs0};", lane, lane));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line("  const float _bias = 8.0f;");
                        self.emit_line(&format!("  unsigned char* _bp = (unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            self.emit_line(&format!(
                                "  {d} = (float)(_bp[{lane}] >> 4) - _bias;  // lane {lane}"
                            ));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line("  constant float _bias = 8.0f;");
                        self.emit_line(&format!("  device unsigned char* _bp = (device unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            self.emit_line(&format!(
                                "  {d} = (float)(_bp[{lane}] >> 4) - _bias;  // lane {lane}"
                            ));
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::UnsignedNibbleLow => {
                // Unsigned 4-bit low-nibble load (Q4_1): extract (& 0x0F) or (>>4), NO subtract-8.
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0];
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let byte_off = if byte_idx > 0 { format!("+{}", byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{b}+{off}{byte_off}];"));
                            if is_high {
                                self.emit_line(&format!("shr.b32 {rs0}, {rs0}, 4;"));
                            }
                            self.emit_line(&format!("and.b32 {rs0}, {rs0}, 0xF;"));
                            self.emit_line(&format!("cvt.rn.f32.u32 {d}<{}>, {rs0};", lane));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  unsigned char* _bp = (unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let nibble_expr = if is_high {
                                format!("_bp[{byte_idx}] >> 4")
                            } else {
                                format!("_bp[{byte_idx}] & 0xF")
                            };
                            self.emit_line(&format!("  {d} = (float)({nibble_expr});  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  device unsigned char* _bp = (device unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let nibble_expr = if is_high {
                                format!("_bp[{byte_idx}] >> 4")
                            } else {
                                format!("_bp[{byte_idx}] & 0xF")
                            };
                            self.emit_line(&format!("  {d} = (float)({nibble_expr});  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::UnsignedNibbleHigh => {
                // Unsigned 4-bit high-nibble load (Q4_1): extract (>>4), NO subtract-8.
                let d = self.reg_name_with_kind(dst, alloc);
                let b = self.reg_name_with_kind(base, alloc);
                let off = self.offset_to_string(offset, alloc);
                let lanes = width.f32_lanes().max(1);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0];
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            let byte_off = if byte_idx > 0 { format!("+{}", byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{b}+{off}{byte_off}];"));
                            self.emit_line(&format!("shr.b32 {rs0}, {rs0}, 4;"));
                            self.emit_line(&format!("cvt.rn.f32.u32 {d}<{}>, {rs0};", lane));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  unsigned char* _bp = (unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            self.emit_line(&format!(
                                "  {d} = (float)(_bp[{byte_idx}] >> 4);  // lane {lane}"
                            ));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  device unsigned char* _bp = (device unsigned char*)({b}+({off}));"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 2;
                            self.emit_line(&format!(
                                "  {d} = (float)(_bp[{byte_idx}] >> 4);  // lane {lane}"
                            ));
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BlockUnpackMode::Bitpack2 { bias } => {
                // Q2K 2-bit packed: each byte has 4 × 2-bit values.
                // Extract: (byte >> (2*(i%4) + 8*(i/4))) & 3
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(base, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(*bias);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0];
                        let fs0 = self.scratch_vec_names[0];
                        // Load bias constant once
                        self.emit_line(&format!("mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("mov.f32 {fs0}, {rs0};"));
                        for lane in 0..lanes {
                            let byte_idx = lane / 4;
                            let bit_shift = (lane % 4) * 2;
                            let byte_off = if byte_idx > 0 { format!("+{}", byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{qs_b}{byte_off}];"));
                            self.emit_line(&format!("shr.b32 {rs0}, {rs0}, {bit_shift};"));
                            self.emit_line(&format!("and.b32 {rs0}, {rs0}, 3;"));
                            self.emit_line(&format!("cvt.rn.f32.u32 {d}, {rs0};  // lane {lane}"));
                            self.emit_line(&format!("sub.rn.f32 {d}, {d}, {fs0};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        for lane in 0..lanes {
                            let byte_idx = lane / 4;
                            let bit_shift = (lane % 4) * 2;
                            let expr = format!("((((unsigned char*)({qs_b}))[{byte_idx}]) >> {bit_shift}) & 3");
                            self.emit_line(&format!("  {d} = (float)({expr}) - {bias_bits}_f32;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        for lane in 0..lanes {
                            let byte_idx = lane / 4;
                            let bit_shift = (lane % 4) * 2;
                            let expr = format!("((((device unsigned char*)({qs_b}))[{byte_idx}]) >> {bit_shift}) & 3");
                            self.emit_line(&format!("  {d} = (float)({expr}) - {bias_bits}_f32;  // lane {lane}"));
                        }
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
        }
    }

    fn lower_quant_biplane_load(&mut self, dst: VRegId, qs_base: VRegId, extra_base: VRegId, bias: f32, mode: &BiPlaneMode, width: SimdWidth, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match mode {
            BiPlaneMode::Low5 => {
                // GGUF Q5_0/Q5_1: load qs (4-bit low nibbles) from qs_base + qh (1-bit high plane) from qh_base,
                // merge: value = (nibble | (qh_bit << 4)) - bias, convert to F32.
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(qs_base, alloc);
                let qh_b = self.reg_name_with_kind(extra_base, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(bias);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // qs nibble
                        let rs1 = self.scratch_gpr_names[1]; // qh byte
                        let fs0 = self.scratch_vec_names[0]; // bias f32
                        // Load bias constant once
                        self.emit_line(&format!("mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("mov.f32 {fs0}, {rs0};"));
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            // Load qs byte
                            let qs_byte_off = if qs_byte_idx > 0 { format!("+{}", qs_byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{qs_b}{qs_byte_off}];"));
                            if is_high {
                                self.emit_line(&format!("shr.b32 {rs0}, {rs0}, 4;"));
                            }
                            self.emit_line(&format!("and.b32 {rs0}, {rs0}, 0xF;"));
                            // Load qh bit: qh byte index = lane/8, bit position = lane%8
                            let qh_byte_idx = lane / 8;
                            let qh_bit = lane % 8;
                            let qh_byte_off = if qh_byte_idx > 0 { format!("+{}", qh_byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs1}, [{qh_b}{qh_byte_off}];"));
                            self.emit_line(&format!("shr.b32 {rs1}, {rs1}, {qh_bit};"));
                            self.emit_line(&format!("and.b32 {rs1}, {rs1}, 1;"));
                            // Merge: nibble | (qh_bit << 4)
                            self.emit_line(&format!("shl.b32 {rs1}, {rs1}, 4;"));
                            self.emit_line(&format!("or.b32 {rs0}, {rs0}, {rs1};"));
                            // Convert to f32 and subtract bias
                            self.emit_line(&format!("cvt.rn.f32.u32 {d}, {rs0};  // lane {lane}"));
                            self.emit_line(&format!("sub.rn.f32 {d}, {d}, {fs0};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  const float _bias = {bias_bits}_f32;"));
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let qh_byte_idx = lane / 8;
                            let qh_bit = lane % 8;
                            let nibble_expr = if is_high {
                                format!("(((unsigned char*)({qs_b}))[{qs_byte_idx}]) >> 4")
                            } else {
                                format!("(((unsigned char*)({qs_b}))[{qs_byte_idx}]) & 0xF")
                            };
                            let qh_expr = format!("((((unsigned char*)({qh_b}))[{qh_byte_idx}]) >> {qh_bit}) & 1");
                            self.emit_line(&format!("  {d} = (float)({nibble_expr} | ({qh_expr} << 4)) - _bias;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        self.emit_line(&format!("  constant float _bias = {bias_bits}_f32;"));
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let qh_byte_idx = lane / 8;
                            let qh_bit = lane % 8;
                            let nibble_expr = if is_high {
                                format!("(((device unsigned char*)({qs_b}))[{qs_byte_idx}]) >> 4")
                            } else {
                                format!("(((device unsigned char*)({qs_b}))[{qs_byte_idx}]) & 0xF")
                            };
                            let qh_expr = format!("((((device unsigned char*)({qh_b}))[{qh_byte_idx}]) >> {qh_bit}) & 1");
                            self.emit_line(&format!("  {d} = (float)({nibble_expr} | ({qh_expr} << 4)) - _bias;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BiPlaneMode::Low6 => {
                // GGUF Q6K: load qs (4-bit low) from qs_base + qh (2-bit high plane) from qh_base,
                // merge: value = (nibble | (qh_2bit << 4)) - bias, convert to F32.
                // Each qs byte has 2 nibbles, each qh byte has 4 x 2-bit values.
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(qs_base, alloc);
                let qh_b = self.reg_name_with_kind(extra_base, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(bias);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // qs nibble
                        let rs1 = self.scratch_gpr_names[1]; // qh 2-bit
                        let fs0 = self.scratch_vec_names[0]; // bias f32
                        // Load bias constant once
                        self.emit_line(&format!("mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("mov.f32 {fs0}, {rs0};"));
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            // Load qs nibble
                            let qs_byte_off = if qs_byte_idx > 0 { format!("+{}", qs_byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{qs_b}{qs_byte_off}];"));
                            if is_high {
                                self.emit_line(&format!("shr.b32 {rs0}, {rs0}, 4;"));
                            }
                            self.emit_line(&format!("and.b32 {rs0}, {rs0}, 0xF;"));
                            // Load qh 2-bit: qh byte index = lane/4, bit position = (lane%4)*2
                            let qh_byte_idx = lane / 4;
                            let qh_bit_shift = (lane % 4) * 2;
                            let qh_byte_off = if qh_byte_idx > 0 { format!("+{}", qh_byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs1}, [{qh_b}{qh_byte_off}];"));
                            self.emit_line(&format!("shr.b32 {rs1}, {rs1}, {qh_bit_shift};"));
                            self.emit_line(&format!("and.b32 {rs1}, {rs1}, 3;"));
                            // Merge: nibble | (qh_2bit << 4)
                            self.emit_line(&format!("shl.b32 {rs1}, {rs1}, 4;"));
                            self.emit_line(&format!("or.b32 {rs0}, {rs0}, {rs1};"));
                            // Convert to f32 and subtract bias
                            self.emit_line(&format!("cvt.rn.f32.u32 {d}, {rs0};  // lane {lane}"));
                            self.emit_line(&format!("sub.rn.f32 {d}, {d}, {fs0};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let qh_byte_idx = lane / 4;
                            let qh_bit_shift = (lane % 4) * 2;
                            let nibble_expr = if is_high {
                                format!("(((unsigned char*)({qs_b}))[{qs_byte_idx}]) >> 4")
                            } else {
                                format!("(((unsigned char*)({qs_b}))[{qs_byte_idx}]) & 0xF")
                            };
                            let qh_expr = format!("((((unsigned char*)({qh_b}))[{qh_byte_idx}]) >> {qh_bit_shift}) & 3");
                            self.emit_line(&format!("  {d} = (float)({nibble_expr} | ({qh_expr} << 4)) - {bias_bits}_f32;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 2;
                            let is_high = lane % 2 == 1;
                            let qh_byte_idx = lane / 4;
                            let qh_bit_shift = (lane % 4) * 2;
                            let nibble_expr = if is_high {
                                format!("(((device unsigned char*)({qs_b}))[{qs_byte_idx}]) >> 4")
                            } else {
                                format!("(((device unsigned char*)({qs_b}))[{qs_byte_idx}]) & 0xF")
                            };
                            let qh_expr = format!("((((device unsigned char*)({qh_b}))[{qh_byte_idx}]) >> {qh_bit_shift}) & 3");
                            self.emit_line(&format!("  {d} = (float)({nibble_expr} | ({qh_expr} << 4)) - {bias_bits}_f32;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
            BiPlaneMode::Q3Merge => {
                // Q3K 3-bit: qs(2-bit) + hmask(1-bit) merge.
                // Each byte has 4 × 2-bit qs values, each byte has 8 × 1-bit hmask values.
                // Merged value = qs_2bit | (hmask_bit << 2)
                let d = self.reg_name_with_kind(dst, alloc);
                let qs_b = self.reg_name_with_kind(qs_base, alloc);
                let hmask_b = self.reg_name_with_kind(extra_base, alloc);
                let lanes = width.f32_lanes().max(1);
                let bias_bits = f32::to_bits(bias);
                match self.dialect {
                    GpuDialect::Ptx { .. } => {
                        let rs0 = self.scratch_gpr_names[0]; // qs 2-bit
                        let rs1 = self.scratch_gpr_names[1]; // hmask bit
                        let fs0 = self.scratch_vec_names[0]; // bias f32
                        // Load bias constant once
                        self.emit_line(&format!("mov.u32 {rs0}, {bias_bits};"));
                        self.emit_line(&format!("mov.f32 {fs0}, {rs0};"));
                        for lane in 0..lanes {
                            // qs 2-bit: byte[lane/4], shift = (lane%4)*2
                            let qs_byte_idx = lane / 4;
                            let qs_bit_shift = (lane % 4) * 2;
                            let qs_byte_off = if qs_byte_idx > 0 { format!("+{}", qs_byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs0}, [{qs_b}{qs_byte_off}];"));
                            self.emit_line(&format!("shr.b32 {rs0}, {rs0}, {qs_bit_shift};"));
                            self.emit_line(&format!("and.b32 {rs0}, {rs0}, 3;"));
                            // hmask bit: byte[lane/8], shift = lane%8
                            let hmask_byte_idx = lane / 8;
                            let hmask_bit = lane % 8;
                            let hmask_byte_off = if hmask_byte_idx > 0 { format!("+{}", hmask_byte_idx) } else { String::new() };
                            self.emit_line(&format!("ld.global.u8 {rs1}, [{hmask_b}{hmask_byte_off}];"));
                            self.emit_line(&format!("shr.b32 {rs1}, {rs1}, {hmask_bit};"));
                            self.emit_line(&format!("and.b32 {rs1}, {rs1}, 1;"));
                            // Merge: qs_2bit | (hmask_bit << 2)
                            self.emit_line(&format!("shl.b32 {rs1}, {rs1}, 2;"));
                            self.emit_line(&format!("or.b32 {rs0}, {rs0}, {rs1};"));
                            // Convert and subtract bias
                            self.emit_line(&format!("cvt.rn.f32.u32 {d}, {rs0};  // lane {lane}"));
                            self.emit_line(&format!("sub.rn.f32 {d}, {d}, {fs0};"));
                        }
                    }
                    GpuDialect::Hip { .. } => {
                        self.emit_line("{");
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 4;
                            let qs_bit_shift = (lane % 4) * 2;
                            let hmask_byte_idx = lane / 8;
                            let hmask_bit = lane % 8;
                            let qs_expr = format!("((((unsigned char*)({qs_b}))[{qs_byte_idx}]) >> {qs_bit_shift}) & 3");
                            let hmask_expr = format!("((((unsigned char*)({hmask_b}))[{hmask_byte_idx}]) >> {hmask_bit}) & 1");
                            self.emit_line(&format!("  {d} = (float)({qs_expr} | ({hmask_expr} << 2)) - {bias_bits}_f32;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                    GpuDialect::Metal { .. } => {
                        self.emit_line("{");
                        for lane in 0..lanes {
                            let qs_byte_idx = lane / 4;
                            let qs_bit_shift = (lane % 4) * 2;
                            let hmask_byte_idx = lane / 8;
                            let hmask_bit = lane % 8;
                            let qs_expr = format!("((((device unsigned char*)({qs_b}))[{qs_byte_idx}]) >> {qs_bit_shift}) & 3");
                            let hmask_expr = format!("((((device unsigned char*)({hmask_b}))[{hmask_byte_idx}]) >> {hmask_bit}) & 1");
                            self.emit_line(&format!("  {d} = (float)({qs_expr} | ({hmask_expr} << 2)) - {bias_bits}_f32;  // lane {lane}"));
                        }
                        self.emit_line("}");
                    }
                }
                Ok(())
            }
        }
    }
}

