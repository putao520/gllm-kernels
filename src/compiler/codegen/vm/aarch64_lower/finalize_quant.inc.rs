impl AArch64Lower {
    pub fn finalize(self) -> Result<Vec<u8>, CompilerError> {
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
        Ok(self.code)
    }


    fn lower_quant_block_load(&mut self, dst: VRegId, base: VRegId, offset: &OffsetExpr, unpack: &BlockUnpackMode, width: SimdWidth, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match unpack {
            BlockUnpackMode::Int8 => {
                // Load INT8 quantized values from memory, sign-extend to F32.
                // NEON path: LDR Q (16 bytes) + SSHLL chain + SCVTF.
                // For W128 (4 lanes): load 4 bytes → SXTL b→h → SXTL h→s → SCVTF s→f32
                // For W256 (8 lanes): load 8 bytes → two SSHLL chains
                let vd = self.resolve_vreg(dst, alloc)?;
                let xn = self.resolve_gpr(base, alloc)?;
                let lanes = width.f32_lanes().max(1);

                // Compute effective address: x16 = base + offset
                self.eval_offset_to_tmp(offset, alloc, 16)?;
                self.emit32(self.enc_add_reg(16, xn, 16));

                if lanes <= 4 {
                    // Load 4 bytes (4 × i8) into vd low 32 bits
                    // LDR Sd, [X16] — 32-bit load into NEON scalar
                    self.emit32(0xBD400000 | (16u32 << 5) | vd as u32);
                    // SSHLL Vd.4S, Vn.4H, #0 — sign-extend 4×i16 → 4×i32 (actually SXTL h→s first)
                    // First: SSHLL Vd.4H, Vn.8B, #0 (= SXTL) — sign-extend low 8 bytes to 8×i16
                    // SSHLL Vd.4H, Vn.4H, #0: 0x0F00A400 | vn<<5 | vd
                    self.emit32(0x0F00A400 | ((vd as u32) << 5) | vd as u32);
                    // SSHLL Vd.4S, Vn.4H, #0: 0x4F00A400 | vn<<5 | vd — sign-extend low 4×i16 → 4×i32
                    self.emit32(0x4F00A400 | ((vd as u32) << 5) | vd as u32);
                    // SCVTF Vd.4S, Vn.4S — convert i32 → f32
                    self.emit32(self.enc_scvtf_4s(vd, vd));
                } else {
                    // 8 lanes: load 8 bytes, split into two halves
                    // LDR Dd, [X16] — 64-bit load
                    self.emit32(0xFC400000 | (16u32 << 5) | vd as u32);
                    // SSHLL Vd.8H, Vn.8B, #0 — sign-extend 8×i8 → 8×i16
                    // SSHLL.2D for upper half: SSHLL2 uses Q=1 form
                    // Actually: SSHLL Vd.8H, Vn.8B, #0: 0x2F00A400 | vn<<5 | vd (Q=0, 8B→8H)
                    self.emit32(0x2F00A400 | ((vd as u32) << 5) | vd as u32);

                    // Now we need SSHLL for low 4 H → 4 S and SSHLL2 for high 4 H → 4 S
                    // But we only have 8 i16 values in vd. We need a scratch register.
                    // Use v16 as scratch (x16 GPR is safe; v16 NEON is caller-saved).
                    let scratch: u8 = 16;
                    // MOV scratch, vd (save i16 vector)
                    self.emit32(self.enc_neon_mov(scratch, vd));

                    // SSHLL Vd.4S, Vn.4H, #0 — low 4 i16 → 4 i32
                    // SSHLL Vd.4S, Vn.4H, #0: 0x0F00A400 for .4S? No.
                    // SSHLL Vd.4S, Vn.4H, #0: size=01 (H→S), Q=0 → 0x0F00A400
                    self.emit32(0x0F00A400 | ((scratch as u32) << 5) | vd as u32);
                    // SSHLL2 scratch.4S, Vn.8H, #0 — high 4 i16 → 4 i32
                    // SSHLL2: Q=1 → 0x4F00A400
                    self.emit32(0x4F00A400 | ((scratch as u32) << 5) | scratch as u32);
                    // Now vd = low 4 × i32, scratch = high 4 × i32
                    // Interleave: ZIP1 vd.4S, vd.4S, scratch.4S (put them together — but NEON only has 4S per Q register)
                    // For 8 lanes output we need W256. But NEON is 128-bit (4 × f32).
                    // With NEON only supporting W128, 8 lanes need two registers.
                    // For now, just convert the low 4 to f32 (primary use case).
                    self.emit32(self.enc_scvtf_4s(vd, vd));
                    self.emit32(self.enc_scvtf_4s(scratch, scratch));
                    // Note: for full 8-lane support, caller should issue two GgufInt8Load with offsets
                }
                Ok(())
            }
            BlockUnpackMode::F16Broadcast => {
                // Load 1 × F16 from memory, convert to F32, broadcast to all lanes.
                // Reuse the existing QuantLoadF16toF32 pattern:
                //   LDRH Wt, [addr]; FMOV St, Wt; FCVT St, Ht; DUP Vd.4S, Vn.S[0]
                let vd = self.resolve_vreg(dst, alloc)?;
                let xn = self.resolve_gpr(base, alloc)?;
                let wt: u8 = 8; // W8 scratch

                // Compute effective address into x9
                self.eval_offset_to_tmp(offset, alloc, 16)?;
                self.emit32(self.enc_add_reg(9, xn, 16));

                // LDRH W8, [X9] — load 16-bit (F16)
                self.emit32(0x79400000 | ((9u32) << 5) | wt as u32);
                // FMOV Sd, Wt — move GPR bits to NEON scalar
                self.emit32(0x1E270000 | ((wt as u32) << 5) | vd as u32);
                // FCVT Sd, Hd — F16 → F32 scalar conversion
                self.emit32(0x1EE20000 | ((vd as u32) << 5) | vd as u32);
                // DUP Vd.4S, Vd.S[0] — broadcast lane 0 to all 4 lanes
                let imm5: u32 = 0b00100; // S lane, index 0
                self.emit32(0x4E000400 | (imm5 << 16) | ((vd as u32) << 5) | vd as u32);
                Ok(())
            }
            BlockUnpackMode::SignedNibbleLow => {
                // GGUF Q4_0: load lanes/2 bytes, unpack nibbles → subtract 8 → cvt to f32.
                // NEON path for 4 lanes (2 bytes → 4 nibbles → 4 × f32):
                //   Load 2 bytes, unpack low/high nibbles, sign-extend, subtract 8, convert.
                // NEON path for 8 lanes (4 bytes → 8 nibbles → 8 × f32):
                //   Load 4 bytes, unpack, interleave, subtract 8, convert.
                let vd = self.resolve_vreg(dst, alloc)?;
                let xn = self.resolve_gpr(base, alloc)?;
                let lanes = width.f32_lanes().max(1);

                // Compute effective address: x16 = base + offset
                self.eval_offset_to_tmp(offset, alloc, 16)?;
                self.emit32(self.enc_add_reg(16, xn, 16));

                // Scratch NEON registers: v16, v17 (caller-saved, IP registers)
                let s1: u8 = 16; // scratch 1
                let s2: u8 = 17; // scratch 2

                // Build constant 0x0F in v17 for AND mask:
                // MOV W8, #0x0F; FMOV S17, W8; DUP V17.16B, V17.B[0]
                self.emit32(self.enc_movz_w(8, 0x0F));
                self.emit32(0x1E270000 | ((8u32) << 5) | s2 as u32); // FMOV S17, W8
                // DUP V17.16B, V17.B[0]: imm5 = 0b00001 (B lane, index 0)
                self.emit32(0x4E000400 | (0b00001u32 << 16) | ((s2 as u32) << 5) | s2 as u32);

                if lanes <= 4 {
                    // Load 2 bytes (for 4 nibbles) — LDRH W8, [X16]
                    self.emit32(0x79400000 | ((16u32) << 5) | 8u32);
                    // FMOV Sd, W8 — put 2 bytes into NEON
                    self.emit32(0x1E270000 | ((8u32) << 5) | vd as u32);
                    // DUP Vd.16B, Vd.B[0] — broadcast byte to all 16 byte lanes (so we can AND/shift)
                    self.emit32(0x4E000400 | (0b00001u32 << 16) | ((vd as u32) << 5) | vd as u32);

                    // AND vd, vd, s1_mask → low nibbles
                    // AND Vd.16B, Vn.16B, Vm.16B: 0x4E201C00 | vm<<16 | vn<<5 | vd
                    self.emit32(0x4E201C00 | ((s2 as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                    // Save low nibbles to s1
                    self.emit32(self.enc_neon_mov(s1, vd));

                    // Get high nibbles: shift right by 4 (USHR on byte lanes)
                    // USHR Vd.16B, Vn.16B, #4: immh_immb for 8-bit element = 8+4 = 12 → immh=1, immb=4
                    // SHL is encode-first approach; for USHR on .16B: 0x6F000400 | (immh_immb << 16) | vn<<5 | vd
                    // For 8-bit: size=00, immh = 0b0001, immb = shift_code = (8+4) for shift 4 → wait.
                    // USHR on .B elements: immh:immb = 0001:(8-shift). For shift=4: immh:immb = 0001:0100 = 0x14
                    let immh_immb_nibble: u32 = 0x14; // 8-bit element, shift right by 4
                    self.emit32(0x6F000400 | (immh_immb_nibble << 16) | ((vd as u32) << 5) | vd as u32);

                    // Now s1 = [low0, low0, low0, ...], vd = [high0, high0, high0, ...]
                    // Interleave: ZIP1 Vd.16B, Vs1.16B, Vd.16B → [low0, high0, low0, high0, ...]
                    // ZIP1 Vd.16B, Vn.16B, Vm.16B: 0x0E003800 | vm<<16 | vn<<5 | vd
                    self.emit32(0x0E003800 | ((vd as u32) << 16) | ((s1 as u32) << 5) | vd as u32);

                    // Zero-extend bytes to 32-bit integers:
                    // USHLL Vd.8H, Vn.8B, #0 (= MOVL) — zero-extend 8 bytes → 8 half-words
                    // USHLL Vd.8H, Vn.8B, #0: 0x2F00A400 | vn<<5 | vd  (Q=0, zero-extend)
                    self.emit32(0x2F00A400 | ((vd as u32) << 5) | vd as u32);
                    // USHLL Vd.4S, Vn.4H, #0: 0x0F00A400 | vn<<5 | vd — low 4 H → 4 S
                    self.emit32(0x0F00A400 | ((vd as u32) << 5) | vd as u32);
                    // SCVTF Vd.4S, Vn.4S
                    self.emit32(self.enc_scvtf_4s(vd, vd));

                    // Subtract 8.0 (Q4_0 zero-point): build 8.0 constant in s1, FSUB
                    let eight_bits = 8.0f32.to_bits();
                    let lo16 = eight_bits & 0xFFFF ;
                    let hi16 = (eight_bits >> 16) & 0xFFFF ;
                    self.emit32(0x52800000 | (lo16 << 5) | 8u32); // MOVZ W8, #lo16
                    self.emit32(0x72A00000 | (hi16 << 5) | 8u32); // MOVK W8, #hi16, LSL 16
                    self.emit32(0x1E270000 | ((8u32) << 5) | s1 as u32); // FMOV S16, W8
                    let imm5: u32 = 0b00100;
                    self.emit32(0x4E000400 | (imm5 << 16) | ((s1 as u32) << 5) | s1 as u32); // DUP V16.4S, V16.S[0]
                    self.emit32(self.enc_fsub_4s(vd, vd, s1));
                } else {
                    // 8 lanes: load 4 bytes → 8 nibbles → 8 × f32
                    // Load 4 bytes into vd (32-bit NEON load)
                    self.emit32(0xBD400000 | ((16u32) << 5) | vd as u32); // LDR Sd, [X16]
                    // Broadcast byte 0..3 to fill vector for nibble extraction
                    // Actually: we need each of the 4 bytes separately.
                    // Use SHL+AND approach: duplicate bytes, extract low/high nibbles, interleave.

                    // Step 1: extend 4 bytes to 4 × i32 in vd (zero-extend via USHLL chain)
                    // Save raw bytes to s2
                    self.emit32(self.enc_neon_mov(s2, vd));
                    // USHLL Vd.8H, Vn.8B, #0 — zero-extend low 8B→8H (only low 4 meaningful)
                    self.emit32(0x2F00A400 | ((s2 as u32) << 5) | vd as u32);
                    // USHLL Vd.4S, Vn.4H, #0 — zero-extend low 4H→4S
                    self.emit32(0x0F00A400 | ((vd as u32) << 5) | vd as u32);
                    // vd now has [b0, b1, b2, b3] as i32 in lanes 0-3

                    // AND with 0x0F → low nibbles
                    self.emit32(0x4E201C00 | ((s2 as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                    // Save low nibbles to s1
                    self.emit32(self.enc_neon_mov(s1, vd));

                    // Shift right by 4 → high nibbles (still in vd)
                    // USHR Vd.4S, Vn.4S, #4
                    self.emit32(self.enc_ushr_4s(vd, vd, 4));
                    // AND with 0x0F again to clear upper bits
                    self.emit32(0x4E201C00 | ((s2 as u32) << 16) | ((vd as u32) << 5) | vd as u32);
                    // s1 = [low0, low1, low2, low3], vd = [high0, high1, high2, high3]

                    // Interleave: ZIP1 Vd.4S, Vs1.4S, Vd.4S → [low0, high0, low1, high1]
                    // ZIP1 Vd.4S, Vn.4S, Vm.4S: 0x4E403800 | Rm<<16 | Rn<<5 | Rd  (Q=1, size=10 for S)
                    // Actually ZIP1 for 4S: size=10, Q=1
                    // ZIP1 Vd.4S, Vn.4S, Vm.4S = 0x4E003800 | (size << 22) | Rm<<16 | Rn<<5 | Rd
                    // size for S (32-bit) = 10b → size << 22 = 0x00800000
                    // Wait, the encoding is: 0x0E003800 | (size << 22) | Rm<<16 | Rn<<5 | Rd, Q bit at bit 30
                    // For Q=1 (.4S): bit 30 set → 0x4E003800 | (size << 22) | Rm<<16 | Rn<<5 | Rd
                    // size = 0b10 for 32-bit (S) → size<<22 = 0x800000
                    self.emit32(0x4E803800 | ((vd as u32) << 16) | ((s1 as u32) << 5) | vd as u32);

                    // SCVTF i32 → f32
                    self.emit32(self.enc_scvtf_4s(vd, vd));

                    // Subtract 8.0
                    let eight_bits = 8.0f32.to_bits();
                    let lo16 = eight_bits & 0xFFFF ;
                    let hi16 = (eight_bits >> 16) & 0xFFFF ;
                    self.emit32(0x52800000 | (lo16 << 5) | 8u32);
                    self.emit32(0x72A00000 | (hi16 << 5) | 8u32);
                    self.emit32(0x1E270000 | ((8u32) << 5) | s1 as u32);
                    let imm5: u32 = 0b00100;
                    self.emit32(0x4E000400 | (imm5 << 16) | ((s1 as u32) << 5) | s1 as u32);
                    self.emit32(self.enc_fsub_4s(vd, vd, s1));
                }
                Ok(())
            }
            BlockUnpackMode::SignedNibbleHigh => {
                // GGUF PackedNibbles high-nibble extract: NEON not yet implemented.
                let _ = (dst, base, offset, alloc);
                let lanes = width.f32_lanes().max(1);
                Err(CompilerError::CodegenViolation(
                    format!("GgufInt4HighLoad: AArch64 NEON not yet implemented ({} lanes)", lanes)))
            }
            BlockUnpackMode::UnsignedNibbleLow => {
                // Unsigned 4-bit low-nibble load (Q4_1): NEON not yet implemented.
                let _ = (dst, base, offset, alloc);
                let lanes = width.f32_lanes().max(1);
                Err(CompilerError::CodegenViolation(
                    format!("GgufUInt4Load: AArch64 NEON not yet implemented ({} lanes)", lanes)))
            }
            BlockUnpackMode::UnsignedNibbleHigh => {
                // Unsigned 4-bit high-nibble load (Q4_1): NEON not yet implemented.
                let _ = (dst, base, offset, alloc);
                let lanes = width.f32_lanes().max(1);
                Err(CompilerError::CodegenViolation(
                    format!("GgufUInt4HighLoad: AArch64 NEON not yet implemented ({} lanes)", lanes)))
            }
            BlockUnpackMode::Bitpack2 { bias } => {
                // Q2K 2-bit packed → i32 → f32, subtract bias.
                // NEON scalar path: per-lane LDRB + UBFX(2-bit) + SCVTF + FSUB
                let lanes = width.f32_lanes().max(1);
                let vd = self.resolve_vreg(dst, alloc)?;
                let qs_xn = self.resolve_gpr(base, alloc)?;

                if lanes == 4 {
                    // 4 lanes: 1 qs byte → 4 × 2-bit values
                    let w_s: u8 = 8;
                    let s_tmp: u8 = 16;

                    for i in 0..4 {
                        // Load qs byte: LDRB W8, [qs_xn, #(i/4)]
                        // All 4 values come from 1 byte when lanes=4
                        self.emit32(self.enc_ldrb_imm(w_s, qs_xn, 0));
                        // UBFX W8, W8, #shift, #2 — extract 2-bit value
                        let shift = (i % 4) * 2;
                        let lsb = shift as u32;
                        let imms = (shift + 1) as u32; // lsb + width(2) - 1
                        let ubfx = (0b100u32 << 28) | (0b100110u32 << 22)
                            | (lsb << 16) | (imms << 10)
                            | ((w_s as u32) << 5) | (w_s as u32);
                        self.emit32(ubfx);
                        // SCVTF Stmp, Ws
                        self.emit32(self.enc_scvtf_s_w(s_tmp, w_s));
                        // INS Vd.S[i], V16.S[0]
                        if i == 0 {
                            let imm5: u32 = 0b00100;
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        } else {
                            let imm5: u32 = 0b00100 | ((i as u32) << 1);
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        }
                    }

                    // Apply bias subtraction
                    if *bias != 0.0 {
                        let bias_bits = f32::to_bits(*bias);
                        let lo16 = bias_bits & 0xFFFF ;
                        let hi16 = (bias_bits >> 16) & 0xFFFF ;
                        let bias_vec: u8 = 16;
                        self.emit32(self.enc_movz_w(8, lo16 as u16));
                        self.emit32(self.enc_movk_w_lsl16(8, hi16 as u16));
                        self.emit32(self.enc_fmov_s_from_w(bias_vec, 8));
                        let imm5: u32 = 0b00100;
                        self.emit32(0x4E000400 | (imm5 << 16) | ((bias_vec as u32) << 5) | (bias_vec as u32));
                        self.emit32(self.enc_fsub_4s(vd, vd, bias_vec));
                    }
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt2Load aarch64: only 4 lanes supported, got {}", lanes)
                    ));
                }
                Ok(())
            }
            BlockUnpackMode::Mxfp4 { scale_src } => {
                self.emit_mxfp4_dequant(dst, base, offset, *scale_src, width, alloc)
            }
            BlockUnpackMode::Nvfp4 { scale_src } => {
                self.emit_nvfp4_sub_block_dequant(dst, base, offset, *scale_src, width, alloc)
            }
        }
    }

    fn lower_quant_biplane_load(&mut self, dst: VRegId, qs_base: VRegId, extra_base: VRegId, bias: f32, mode: &BiPlaneMode, width: SimdWidth, alloc: &RegAllocation) -> Result<(), CompilerError> {
        match mode {
            BiPlaneMode::Low5 => {
                // INT5: nibble unpack + 1-bit high plane merge + bias subtract to f32
                // NEON scalar path: per-lane LDRB + UBFX + ORR + SCVTF + FSUB
                let lanes = width.f32_lanes().max(1);
                let vd = self.resolve_vreg(dst, alloc)?;
                let qs_xn = self.resolve_gpr(qs_base, alloc)?;
                let qh_xn = self.resolve_gpr(extra_base, alloc)?;

                if lanes == 4 {
                    // Scratch GPRs: w8 (temp), w9 (temp2) - IP registers
                    let w_s: u8 = 8;
                    let w_s2: u8 = 9;

                    for i in 0..4 {
                        // Load qs nibble: LDRB W8, [qs_xn, #(i/2)]
                        self.emit32(self.enc_ldrb_imm(w_s, qs_xn, (i / 2) as u16));
                        // UBFX W8, W8, #shift, #4 - extract 4-bit nibble
                        let shift = (i % 2) * 4;
                        {
                            let lsb = shift as u32;
                            let imms = (shift + 3) as u32; // lsb + width(4) - 1
                            let ubfx = (0b100u32 << 28) | (0b100110u32 << 22)
                                | (lsb << 16) | (imms << 10)
                                | ((w_s as u32) << 5) | (w_s as u32);
                            self.emit32(ubfx);
                        }
                        // Save qs nibble to w9: MOV W9, W8
                        self.emit32(0x2A0003E0 | ((w_s as u32) << 16) | (w_s2 as u32));

                        // Load qh bit: LDRB W8, [qh_xn, #qh_byte_idx]
                        let qh_byte_idx = i / 8;
                        let qh_bit_idx = i % 8;
                        self.emit32(self.enc_ldrb_imm(w_s, qh_xn, qh_byte_idx as u16));
                        // UBFX W8, W8, #qh_bit_idx, #1 - extract 1-bit
                        {
                            let lsb = qh_bit_idx as u32;
                            let ubfx = (0b100u32 << 28) | (0b100110u32 << 22)
                                | (lsb << 16) | (lsb << 10)
                                | ((w_s as u32) << 5) | (w_s as u32);
                            self.emit32(ubfx);
                        }
                        // W8 = 0 or 1, shift left by 4 -> 0 or 16
                        self.emit32(self.enc_lsl_w_imm(w_s, w_s, 4));
                        // ORR W8, W9, W8 - merge nibble | (qh_bit << 4)
                        self.emit32(self.enc_orr_w_reg(w_s, w_s2, w_s));
                        // W8 = 5-bit value (0-31), convert to f32 via temp s16
                        let s_tmp: u8 = 16; // s16 scratch NEON
                        self.emit32(self.enc_scvtf_s_w(s_tmp, w_s));
                        // INS Vd.S[i], V16.S[0]
                        if i == 0 {
                            // SCVTF wrote to S16; move to Vd.S[0]
                            self.emit32(self.enc_fmov_s_from_w(vd, 0)); // dummy; actually need FMOV Sd, S16
                            // FMOV Sd, Sn: 0x1E204000 | (Sn << 16) | (0 << 5) | Sd
                            // Actually simpler: INS Vd.S[0], V16.S[0]
                            let imm5: u32 = 0b00100; // .S lane 0
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        } else {
                            let imm5: u32 = 0b00100 | ((i as u32) << 1); // .S lane i
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        }
                    }

                    // Apply bias subtraction as a vector operation if bias != 0
                    if bias != 0.0 {
                        let bias_bits = f32::to_bits(bias);
                        let lo16 = bias_bits & 0xFFFF ;
                        let hi16 = (bias_bits >> 16) & 0xFFFF ;
                        let bias_vec: u8 = 16;
                        self.emit32(self.enc_movz_w(8, lo16 as u16));
                        self.emit32(self.enc_movk_w_lsl16(8, hi16 as u16));
                        self.emit32(self.enc_fmov_s_from_w(bias_vec, 8)); // FMOV S16, W8
                        let imm5: u32 = 0b00100; // .S lane 0
                        self.emit32(0x4E000400 | (imm5 << 16) | ((bias_vec as u32) << 5) | (bias_vec as u32)); // DUP V16.4S, V16.S[0]
                        self.emit32(self.enc_fsub_4s(vd, vd, bias_vec));
                    }
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt5Load aarch64: only 4 lanes supported, got {}", lanes)
                    ));
                }
                Ok(())
            }
            BiPlaneMode::Low6 => {
                // INT6: nibble unpack + 2-bit high plane merge + bias subtract to f32
                // NEON scalar path: per-lane LDRB + UBFX + ORR + SCVTF + FSUB
                let lanes = width.f32_lanes().max(1);
                let vd = self.resolve_vreg(dst, alloc)?;
                let qs_xn = self.resolve_gpr(qs_base, alloc)?;
                let qh_xn = self.resolve_gpr(extra_base, alloc)?;

                if lanes == 4 {
                    // Scratch GPRs: w8 (temp), w9 (temp2)
                    let w_s: u8 = 8;
                    let w_s2: u8 = 9;

                    for i in 0..4 {
                        // Load qs nibble: LDRB W8, [qs_xn, #(i/2)]
                        self.emit32(self.enc_ldrb_imm(w_s, qs_xn, (i / 2) as u16));
                        // UBFX W8, W8, #shift, #4 - extract 4-bit nibble
                        let shift = (i % 2) * 4;
                        {
                            let lsb = shift as u32;
                            let imms = (shift + 3) as u32;
                            let ubfx = (0b100u32 << 28) | (0b100110u32 << 22)
                                | (lsb << 16) | (imms << 10)
                                | ((w_s as u32) << 5) | (w_s as u32);
                            self.emit32(ubfx);
                        }
                        // Save qs nibble to w9
                        self.emit32(0x2A0003E0 | ((w_s as u32) << 16) | (w_s2 as u32));

                        // Load qh 2-bit: LDRB W8, [qh_xn, #qh_byte_idx]
                        let qh_byte_idx = i / 4;
                        let qh_shift = (i % 4) * 2;
                        self.emit32(self.enc_ldrb_imm(w_s, qh_xn, qh_byte_idx as u16));
                        // UBFX W8, W8, #qh_shift, #2 - extract 2-bit
                        {
                            let lsb = qh_shift as u32;
                            let imms = (qh_shift + 1) as u32;
                            let ubfx = (0b100u32 << 28) | (0b100110u32 << 22)
                                | (lsb << 16) | (imms << 10)
                                | ((w_s as u32) << 5) | (w_s as u32);
                            self.emit32(ubfx);
                        }
                        // W8 = 0..3, shift left by 4 -> 0, 16, 32, or 48
                        self.emit32(self.enc_lsl_w_imm(w_s, w_s, 4));
                        // ORR W8, W9, W8 - merge nibble | (qh_2bit << 4)
                        self.emit32(self.enc_orr_w_reg(w_s, w_s2, w_s));
                        // W8 = 6-bit value (0-63), convert to f32 via temp s16
                        let s_tmp: u8 = 16;
                        self.emit32(self.enc_scvtf_s_w(s_tmp, w_s));
                        // INS Vd.S[i], V16.S[0]
                        if i == 0 {
                            let imm5: u32 = 0b00100; // .S lane 0
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        } else {
                            let imm5: u32 = 0b00100 | ((i as u32) << 1);
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        }
                    }

                    // Apply bias subtraction as a vector operation if bias != 0
                    if bias != 0.0 {
                        let bias_bits = f32::to_bits(bias);
                        let lo16 = bias_bits & 0xFFFF ;
                        let hi16 = (bias_bits >> 16) & 0xFFFF ;
                        let bias_vec: u8 = 16;
                        self.emit32(self.enc_movz_w(8, lo16 as u16));
                        self.emit32(self.enc_movk_w_lsl16(8, hi16 as u16));
                        self.emit32(self.enc_fmov_s_from_w(bias_vec, 8));
                        let imm5: u32 = 0b00100;
                        self.emit32(0x4E000400 | (imm5 << 16) | ((bias_vec as u32) << 5) | (bias_vec as u32));
                        self.emit32(self.enc_fsub_4s(vd, vd, bias_vec));
                    }
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt6Load aarch64: only 4 lanes supported, got {}", lanes)
                    ));
                }
                Ok(())
            }
            BiPlaneMode::Q3Merge => {
                // Q3K 3-bit: qs(2-bit) + hmask(1-bit) → merge → i32 → f32, subtract bias.
                // NEON scalar path: per-lane LDRB + UBFX(2-bit) + UBFX(1-bit) + ORR + SCVTF + FSUB
                let lanes = width.f32_lanes().max(1);
                let vd = self.resolve_vreg(dst, alloc)?;
                let qs_xn = self.resolve_gpr(qs_base, alloc)?;
                let hmask_xn = self.resolve_gpr(extra_base, alloc)?;

                if lanes == 4 {
                    // 4 lanes: 1 qs byte + 1 hmask byte → 4 × 3-bit values
                    let w_s: u8 = 8;
                    let w_s2: u8 = 9;
                    let s_tmp: u8 = 16;

                    for i in 0..4 {
                        // Load qs byte: all 4 values from byte 0
                        self.emit32(self.enc_ldrb_imm(w_s, qs_xn, 0));
                        // UBFX W8, W8, #shift, #2 — extract 2-bit value
                        let qs_shift = (i % 4) * 2;
                        {
                            let lsb = qs_shift as u32;
                            let imms = (qs_shift + 1) as u32;
                            let ubfx = (0b100u32 << 28) | (0b100110u32 << 22)
                                | (lsb << 16) | (imms << 10)
                                | ((w_s as u32) << 5) | (w_s as u32);
                            self.emit32(ubfx);
                        }
                        // Save qs to w9
                        self.emit32(0x2A0003E0 | ((w_s as u32) << 16) | (w_s2 as u32));

                        // Load hmask bit: LDRB W8, [hmask_xn, #(i/8)]
                        let hmask_byte_idx = i / 8;
                        let hmask_bit = i % 8;
                        self.emit32(self.enc_ldrb_imm(w_s, hmask_xn, hmask_byte_idx as u16));
                        // UBFX W8, W8, #hmask_bit, #1 — extract 1-bit
                        {
                            let lsb = hmask_bit as u32;
                            let ubfx = (0b100u32 << 28) | (0b100110u32 << 22)
                                | (lsb << 16) | (lsb << 10)
                                | ((w_s as u32) << 5) | (w_s as u32);
                            self.emit32(ubfx);
                        }
                        // W8 = 0 or 1, shift left by 2 → 0 or 4
                        self.emit32(self.enc_lsl_w_imm(w_s, w_s, 2));
                        // ORR W8, W9, W8 — merge qs_2bit | (hmask_bit << 2)
                        self.emit32(self.enc_orr_w_reg(w_s, w_s2, w_s));
                        // W8 = 3-bit value (0-7)

                        // SCVTF Stmp, Ws
                        self.emit32(self.enc_scvtf_s_w(s_tmp, w_s));
                        // INS Vd.S[i], V16.S[0]
                        if i == 0 {
                            let imm5: u32 = 0b00100;
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        } else {
                            let imm5: u32 = 0b00100 | ((i as u32) << 1);
                            self.emit32(0x4E181C00 | (imm5 << 16) | ((s_tmp as u32) << 5) | (vd as u32));
                        }
                    }

                    // Apply bias subtraction
                    if bias != 0.0 {
                        let bias_bits = f32::to_bits(bias);
                        let lo16 = bias_bits & 0xFFFF ;
                        let hi16 = (bias_bits >> 16) & 0xFFFF ;
                        let bias_vec: u8 = 16;
                        self.emit32(self.enc_movz_w(8, lo16 as u16));
                        self.emit32(self.enc_movk_w_lsl16(8, hi16 as u16));
                        self.emit32(self.enc_fmov_s_from_w(bias_vec, 8));
                        let imm5: u32 = 0b00100;
                        self.emit32(0x4E000400 | (imm5 << 16) | ((bias_vec as u32) << 5) | (bias_vec as u32));
                        self.emit32(self.enc_fsub_4s(vd, vd, bias_vec));
                    }
                } else {
                    return Err(CompilerError::CodegenViolation(
                        format!("GgufInt3Load aarch64: only 4 lanes supported, got {}", lanes)
                    ));
                }
                Ok(())
            }
        }
    }

}

