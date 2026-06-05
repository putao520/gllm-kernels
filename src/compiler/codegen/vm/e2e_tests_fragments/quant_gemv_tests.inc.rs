#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod quant_gemv_tests {
    use crate::compiler::codegen::vm::instr::{BoundExpr, PtrExpr, SimdWidth, VmInstr, VmProgram, VRegKind};
    use crate::compiler::codegen::vm::isa_profile::{IsaProfile, PhysReg};
    use crate::compiler::codegen::vm::reg_alloc::RegAllocator;
    use crate::compiler::codegen::vm::stack_frame::StackFrame;
    use crate::compiler::codegen::vm::x86_lower::X86Lower;
    use crate::compiler::codegen::vm::moe_quant_emit::emit_quant_gemm_inline;
    use crate::compiler::trace::QuantPrecision;
    use crate::compiler::executable::CompiledLayer;
    use crate::dispatch::DeviceProfile;
    use crate::dispatch::device_profile::DotProductCap;
    use crate::quant::QuantType;
    use half::f16;

    fn build_q4_0_gemv_prog(k: usize, n: usize) -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // ARCH-LOADPTR-ORDER: AbiArg sources first, then StackArg.
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) }); // arg 7 = output

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(1),
            n, k,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q4_0 GEMV emit should succeed");

        prog
    }

    fn build_q8_0_gemv_prog(k: usize, n: usize) -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        // ARCH-LOADPTR-ORDER: AbiArg sources first, then StackArg.
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) }); // arg 7 = output

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(1),
            n, k,
            QuantType::Q8_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q8_0 GEMV emit should succeed");

        prog
    }

    fn compile_vm_prog(prog: &VmProgram) -> Vec<u8> {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(prog).unwrap();
        let frame = StackFrame::compute(&alloc, &profile, 0);

        let mut lower = X86Lower::new();
        lower.emit_prologue(&frame, &alloc).unwrap();
        for instr in &prog.instrs {
            lower.lower_instr(instr, &alloc).unwrap();
        }
        lower.emit_epilogue(&frame, &alloc).unwrap();
        lower.finalize().unwrap()
    }

    const Q4_0_BLOCK_SIZE: usize = 32;
    const Q4_0_BLOCK_BYTES: usize = 18;

    fn pack_q4_0_blocks(values: &[i8], scale: f16) -> Vec<u8> {
        assert!(values.len() % Q4_0_BLOCK_SIZE == 0);
        let num_blocks = values.len() / Q4_0_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks * Q4_0_BLOCK_BYTES);

        for blk in 0..num_blocks {
            let scale_bits = scale.to_bits();
            blocks.push((scale_bits & 0xFF) as u8);
            blocks.push(((scale_bits >> 8) & 0xFF) as u8);

            for i in 0..16 {
                let lo = (values[blk * 32 + i] as u8) & 0x0F;
                let hi = (values[blk * 32 + 16 + i] as u8) & 0x0F;
                blocks.push(lo | (hi << 4));
            }
        }
        blocks
    }

    fn pack_q8_0_blocks(values: &[i8], scale: f16) -> Vec<u8> {
        assert!(values.len() % Q4_0_BLOCK_SIZE == 0);
        let num_blocks = values.len() / Q4_0_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks * 34);

        for blk in 0..num_blocks {
            let scale_bits = scale.to_bits();
            blocks.push((scale_bits & 0xFF) as u8);
            blocks.push(((scale_bits >> 8) & 0xFF) as u8);

            for i in 0..32 {
                blocks.push(values[blk * 32 + i] as u8);
            }
        }
        blocks
    }

    fn ref_q4_0_gemv(activation: &[f32], weight_blocks: &[u8], n: usize, k: usize) -> Vec<f32> {
        let num_blocks_per_row = k / Q4_0_BLOCK_SIZE;
        let mut output = vec![0.0f32; n];

        for j in 0..n {
            let row_offset = j * num_blocks_per_row * Q4_0_BLOCK_BYTES;
            let mut sum = 0.0f32;

            for blk in 0..num_blocks_per_row {
                let blk_offset = row_offset + blk * Q4_0_BLOCK_BYTES;
                let scale_bits = (weight_blocks[blk_offset] as u16)
                    | ((weight_blocks[blk_offset + 1] as u16) << 8);
                let scale_f32 = f16::from_bits(scale_bits).to_f32();

                for i in 0..16 {
                    let byte = weight_blocks[blk_offset + 2 + i];
                    let lo = (byte & 0x0F) as i8 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                    let idx_lo = blk * 32 + i;
                    let idx_hi = blk * 32 + 16 + i;
                    sum += activation[idx_lo] * (lo as f32 * scale_f32);
                    sum += activation[idx_hi] * (hi as f32 * scale_f32);
                }
            }
            output[j] = sum;
        }
        output
    }

    fn ref_q8_0_gemv(activation: &[f32], weight_blocks: &[u8], n: usize, k: usize) -> Vec<f32> {
        let num_blocks_per_row = k / Q4_0_BLOCK_SIZE;
        let mut output = vec![0.0f32; n];

        for j in 0..n {
            let row_offset = j * num_blocks_per_row * 34;
            let mut sum = 0.0f32;

            for blk in 0..num_blocks_per_row {
                let blk_offset = row_offset + blk * 34;
                let scale_bits = (weight_blocks[blk_offset] as u16)
                    | ((weight_blocks[blk_offset + 1] as u16) << 8);
                let scale_f32 = f16::from_bits(scale_bits).to_f32();

                for i in 0..32 {
                    let qval = weight_blocks[blk_offset + 2 + i] as i8;
                    let idx = blk * 32 + i;
                    sum += activation[idx] * (qval as f32 * scale_f32);
                }
            }
            output[j] = sum;
        }
        output
    }

    unsafe fn exec_gemv(
        layer: &CompiledLayer,
        activation: &[f32],
        weights: &[u8],
        output: &mut [f32],
    ) {
        let mut telemetry = vec![0u8; crate::compiler::graph::telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES];
        let f = layer.entry_point();
        f(
            activation.as_ptr() as *const u8,
            weights.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            1,
            output.as_mut_ptr() as *mut u8,
            std::ptr::null_mut(),
            telemetry.as_mut_ptr() as *mut u8,
        );
    }

    #[test]
    fn test_q8_0_gemv_k32_n2_jit_execution() {
        let k: usize = 32;
        let n: usize = 2;

        let prog = build_q8_0_gemv_prog(k, n);
        let code = compile_vm_prog(&prog);

        eprintln!("[Q8_0 JIT] code_size={} bytes, prog_instrs={}", code.len(), prog.instrs.len());

        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

        let weight_values: Vec<i8> = vec![1i8; k * n];
        let scale = f16::from_f32(1.0);
        let weight_blocks = pack_q8_0_blocks(&weight_values, scale);

        let activation: Vec<f32> = (0..k).map(|i| (i + 1) as f32).collect();

        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };

        let expected = ref_q8_0_gemv(&activation, &weight_blocks, n, k);

        eprintln!("[Q8_0 JIT] output: {:?}", &output[..n.min(4)]);
        eprintln!("[Q8_0 REF] expected: {:?}", &expected[..n.min(4)]);

        for j in 0..n {
            let diff = (output[j] - expected[j]).abs();
            let rel = if expected[j].abs() > 1e-6 { diff / expected[j].abs() } else { diff };
            assert!(
                rel < 1e-3,
                "Q8_0 GEMV[{}] got={}, expected={}, diff={}, rel={}",
                j, output[j], expected[j], diff, rel
            );
        }
    }

    #[test]
    fn test_q4_0_gemv_k32_n2_jit_execution() {
        let k: usize = 32;
        let n: usize = 2;

        let prog = build_q4_0_gemv_prog(k, n);
        let code = compile_vm_prog(&prog);

        eprintln!("[Q4_0 JIT] code_size={} bytes, prog_instrs={}", code.len(), prog.instrs.len());

        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

        let weight_values: Vec<i8> = (0..k * n).map(|i| ((i % 16) as i8) - 8).collect();
        let scale = f16::from_f32(0.5);
        let weight_blocks = pack_q4_0_blocks(&weight_values, scale);

        let activation: Vec<f32> = (0..k).map(|i| (i + 1) as f32).collect();

        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };

        let expected = ref_q4_0_gemv(&activation, &weight_blocks, n, k);

        eprintln!("[Q4_0 JIT] output: {:?}", &output[..n.min(4)]);
        eprintln!("[Q4_0 REF] expected: {:?}", &expected[..n.min(4)]);

        for j in 0..n {
            let diff = (output[j] - expected[j]).abs();
            let rel = if expected[j].abs() > 1e-6 { diff / expected[j].abs() } else { diff };
            assert!(
                rel < 0.01,
                "Q4_0 GEMV[{}] got={}, expected={}, diff={}, rel={}",
                j, output[j], expected[j], diff, rel
            );
        }
    }

    #[test]
    fn test_q4_0_gemv_k64_n2_jit_execution() {
        let k: usize = 64;
        let n: usize = 2;

        let prog = build_q4_0_gemv_prog(k, n);
        let code = compile_vm_prog(&prog);

        eprintln!("[Q4_0 JIT K=64] code_size={} bytes, prog_instrs={}", code.len(), prog.instrs.len());

        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

        let weight_values: Vec<i8> = vec![5i8; k * n];
        let scale = f16::from_f32(1.0);
        let weight_blocks = pack_q4_0_blocks(&weight_values, scale);

        let activation: Vec<f32> = vec![1.0f32; k];

        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };

        let expected = ref_q4_0_gemv(&activation, &weight_blocks, n, k);

        eprintln!("[Q4_0 JIT K=64] output: {:?}", &output[..n.min(4)]);
        eprintln!("[Q4_0 REF K=64] expected: {:?}", &expected[..n.min(4)]);

        for j in 0..n {
            let diff = (output[j] - expected[j]).abs();
            let rel = if expected[j].abs() > 1e-6 { diff / expected[j].abs() } else { diff };
            assert!(
                rel < 0.01,
                "Q4_0 GEMV K=64 [{}] got={}, expected={}, diff={}, rel={}",
                j, output[j], expected[j], diff, rel
            );
        }
    }

    #[test]
    fn test_q4_0_gemv_k256_n4_jit_execution() {
        let k: usize = 256;
        let n: usize = 4;

        let prog = build_q4_0_gemv_prog(k, n);
        let code = compile_vm_prog(&prog);

        eprintln!("[Q4_0 JIT K=256 N=4] code_size={} bytes, prog_instrs={}", code.len(), prog.instrs.len());

        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

        let weight_values: Vec<i8> = (0..k * n).map(|i| (i % 16) as i8 - 8).collect();
        let scale = f16::from_f32(0.25);
        let weight_blocks = pack_q4_0_blocks(&weight_values, scale);

        let activation: Vec<f32> = (0..k).map(|i| ((i % 10) as f32 + 1.0) * 0.1).collect();

        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };

        let expected = ref_q4_0_gemv(&activation, &weight_blocks, n, k);

        eprintln!("[Q4_0 JIT K=256 N=4] output: {:?}", &output[..n.min(4)]);
        eprintln!("[Q4_0 REF K=256 N=4] expected: {:?}", &expected[..n.min(4)]);

        for j in 0..n {
            let diff = (output[j] - expected[j]).abs();
            let rel = if expected[j].abs() > 1e-6 { diff / expected[j].abs() } else { diff };
            assert!(
                rel < 0.05,
                "Q4_0 GEMV K=256 N=4 [{}] got={}, expected={}, diff={}, rel={}",
                j, output[j], expected[j], diff, rel
            );
        }
    }

    // ── Q3_K GEMV ──────────────────────────────────────────────────────────

    fn build_q3_k_gemv_prog(k: usize, n: usize) -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(1),
            n, k,
            QuantType::Q3K,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q3_K GEMV emit should succeed");

        prog
    }

    const Q3_K_BLOCK_SIZE: usize = 256;
    const Q3_K_BLOCK_BYTES: usize = 110;

    /// Inverse of Q3KExtended rearrangement: given 16 desired int8 scale values
    /// (interpreted as unsigned 6-bit values stored in byte bits [0..5]),
    /// compute the 12 raw bytes that dequantize to those values.
    ///
    /// The forward (dequant) rearrangement is:
    ///   tmp = aux[2]
    ///   aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    ///   aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    ///   aux[0] = (aux[0] & kmask2)         | (((tmp >> 0) & kmask1) << 4)
    ///   aux[1] = (aux[1] & kmask2)         | (((tmp >> 2) & kmask1) << 4)
    ///
    /// Each resulting aux[k] byte b contains: lo nibble = scale[4k+b] bits[0:3],
    ///                                     hi nibble bits[4:5] = scale high 2 bits.
    fn pack_q3k_extended_scales(target_scales: &[u8; 16]) -> [u8; 12] {
        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;

        // Build the 4 desired aux values (after rearrangement)
        let mut desired_aux = [0u32; 4];
        for k in 0..4 {
            let mut v: u32 = 0;
            for b in 0..4 {
                let s = target_scales[k * 4 + b] as u32; // 6-bit value [0,63]
                // lo nibble = s & 0xF, hi 2 bits at positions 4-5
                v |= (s & 0x0F) << (b * 8);
                v |= ((s >> 4) & 0x03) << (b * 8 + 4);
            }
            desired_aux[k] = v;
        }

        // Now invert the rearrangement:
        // desired_aux[0] = (orig[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        // desired_aux[1] = (orig[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        // desired_aux[2] = ((orig[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        // desired_aux[3] = ((orig[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        //
        // From aux[0] lo = orig[0] lo (kmask2 bits)
        // From aux[2] lo = orig[0] hi (>>4 then kmask2 bits)
        // So orig[0] = (aux[0] & kmask2) | ((aux[2] & kmask2) << 4)
        // Similarly orig[1] = (aux[1] & kmask2) | ((aux[3] & kmask2) << 4)
        //
        // From aux[0] hi = (tmp >> 0) & kmask1) << 4 → tmp bits[0..7] lo-2 of each byte
        // From aux[1] hi = (tmp >> 2) & kmask1) << 4
        // From aux[2] hi = (tmp >> 4) & kmask1) << 4
        // From aux[3] hi = (tmp >> 6) & kmask1) << 4

        let orig0 = (desired_aux[0] & kmask2) | ((desired_aux[2] & kmask2) << 4);
        let orig1 = (desired_aux[1] & kmask2) | ((desired_aux[3] & kmask2) << 4);

        // Extract hi nibbles from desired_aux to reconstruct tmp
        // desired_aux[0] hi bits (>>4 & kmask1) = (tmp >> 0) & kmask1
        // desired_aux[1] hi bits (>>4 & kmask1) = (tmp >> 2) & kmask1
        // desired_aux[2] hi bits (>>4 & kmask1) = (tmp >> 4) & kmask1
        // desired_aux[3] hi bits (>>4 & kmask1) = (tmp >> 6) & kmask1
        let tmp_part0 = (desired_aux[0] >> 4) & kmask1;
        let tmp_part2 = (desired_aux[1] >> 4) & kmask1;
        let tmp_part4 = (desired_aux[2] >> 4) & kmask1;
        let tmp_part6 = (desired_aux[3] >> 4) & kmask1;
        let orig2 = tmp_part0 | (tmp_part2 << 2) | (tmp_part4 << 4) | (tmp_part6 << 6);

        // orig[3] is unused in the dequant (only orig[0], orig[1], orig[2] matter)
        let orig3 = 0u32;

        let mut result = [0u8; 12];
        let orig = [orig0.to_le(), orig1.to_le(), orig2.to_le(), orig3.to_le()];
        let bytes = unsafe { &*(&orig as *const u32 as *const [u8; 16]) };
        result.copy_from_slice(&bytes[..12]);
        result
    }

    /// Dequantize a single Q3_K block to f32 values (llama.cpp reference algorithm).
    fn dequantize_q3_k_block(block: &[u8]) -> [f32; 256] {
        assert_eq!(block.len(), Q3_K_BLOCK_BYTES);
        let mut output = [0.0f32; 256];

        let d_bits = (block[108] as u16) | ((block[109] as u16) << 8);
        let d = f16::from_bits(d_bits).to_f32();

        // Q3KExtended rearrangement
        let mut aux = [0u32; 4];
        let aux_bytes = unsafe { &mut *(aux.as_mut_ptr() as *mut [u8; 12]) };
        aux_bytes.copy_from_slice(&block[96..108]);

        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scales = unsafe { &*(aux.as_ptr() as *const [i8; 16]) };

        let q = &block[32..96];   // qs[64]
        let hm = &block[0..32];   // hmask[32]

        let mut is = 0usize;
        let mut m = 1u8;

        for seg in 0..2 {
            for j in 0..4 {
                let shift = j * 2;

                let dl = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qs_val = (q[seg * 32 + l] >> shift) & 3;
                    let hmask_bit = (hm[l] & m) != 0;
                    let bias = if hmask_bit { 0i8 } else { 4i8 };
                    output[seg * 128 + j * 32 + l] = (qs_val as i8 - bias) as f32 * dl;
                }

                let dl2 = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qs_val = (q[seg * 32 + 16 + l] >> shift) & 3;
                    let hmask_bit = (hm[l + 16] & m) != 0;
                    let bias = if hmask_bit { 0i8 } else { 4i8 };
                    output[seg * 128 + j * 32 + 16 + l] = (qs_val as i8 - bias) as f32 * dl2;
                }

                m <<= 1;
            }
        }
        output
    }

    /// Build a Q3_K block with known values for testing.
    /// Uses simple quantization: each element is rounded to nearest quant level.
    fn build_simple_q3_k_block(values: &[f32], d: f16) -> [u8; Q3_K_BLOCK_BYTES] {
        assert_eq!(values.len(), Q3_K_BLOCK_SIZE);
        let mut block = [0u8; Q3_K_BLOCK_BYTES];
        let d_f32 = d.to_f32();

        // First pass: compute per-sub-block scale values
        let mut raw_scales = [0u8; 16]; // 6-bit unsigned, bias -32 during dequant
        for is in 0..16 {
            let start = is * 16;
            let mut max_abs = 0.0f32;
            for l in 0..16 {
                let v = values[start + l].abs();
                if v > max_abs { max_abs = v; }
            }
            // quant range = [-4, 3] (8 levels), center at 0
            // dl = d * (scale - 32), we want dl * max_quant ≈ max_abs
            // max_quant ≈ 3.5 (midpoint of range)
            // So scale = max_abs / (d * 3.5) + 32
            if d_f32.abs() > 1e-10 {
                let s = max_abs / (d_f32 * 3.5) + 32.0;
                raw_scales[is] = s.round().clamp(0.0, 63.0) as u8;
            } else {
                raw_scales[is] = 32;
            }
        }

        // Pack scales using inverse Q3KExtended
        let packed_scales = pack_q3k_extended_scales(&raw_scales);
        block[96..108].copy_from_slice(&packed_scales);

        // Write d (f16 at offset 108)
        let d_bits = d.to_bits();
        block[108] = (d_bits & 0xFF) as u8;
        block[109] = ((d_bits >> 8) & 0xFF) as u8;

        // Second pass: quantize values into qs[] and hmask[]
        for seg in 0..2 {
            for j in 0..4 {
                for run in 0..2 {
                    let is_idx = seg * 8 + j * 2 + run;
                    let dl = d_f32 * (raw_scales[is_idx] as f32 - 32.0);
                    let start = seg * 128 + j * 32 + run * 16;

                    for l in 0..16 {
                        let v = values[start + l];
                        let quant = if dl.abs() > 1e-10 { v / dl } else { 0.0 };
                        let qi = quant.round().clamp(-4.0, 3.0) as i8;

                        // Map qi to qs_2bit + hmask:
                        // if qi >= 0: qs=qi, hmask_bit=1 (bias=0)
                        // if qi < 0:  qs=qi+4 (in [0,3]), hmask_bit=0 (bias=4)
                        let (qs_val, set_hmask) = if qi >= 0 {
                            (qi as u8, true)
                        } else {
                            ((qi + 4) as u8, false)
                        };

                        // qs byte: seg*32 + run*16 + l, shift = j*2
                        let qs_byte_idx = seg * 32 + run * 16 + l;
                        block[32 + qs_byte_idx] |= (qs_val & 0x03) << (j * 2);

                        // hmask byte: l (first run) or l+16 (second run)
                        // bit position: seg*4 + j (m accumulates across segments)
                        // NOTE: hmask does NOT advance with seg — same 32 bytes for both segments
                        // seg=0 uses bits 0-3, seg=1 uses bits 4-7
                        if set_hmask {
                            let hmask_byte_idx = l + run * 16;
                            block[hmask_byte_idx] |= 1 << (seg * 4 + j);
                        }
                    }
                }
            }
        }

        block
    }

    /// Scalar reference Q3_K GEMV using dequantize_q3_k_block.
    fn ref_q3_k_gemv(activation: &[f32], weight_blocks: &[u8], n: usize, k: usize) -> Vec<f32> {
        let num_blocks_per_row = k / Q3_K_BLOCK_SIZE;
        let mut output = vec![0.0f32; n];

        for row in 0..n {
            let mut sum = 0.0f32;
            for blk in 0..num_blocks_per_row {
                let blk_offset = (row * num_blocks_per_row + blk) * Q3_K_BLOCK_BYTES;
                let block_data = &weight_blocks[blk_offset..blk_offset + Q3_K_BLOCK_BYTES];
                let dequant = dequantize_q3_k_block(block_data);
                let base = blk * Q3_K_BLOCK_SIZE;
                for i in 0..Q3_K_BLOCK_SIZE {
                    if base + i < k {
                        sum += activation[base + i] * dequant[i];
                    }
                }
            }
            output[row] = sum;
        }
        output
    }

    /// Test that pack_q3_k_blocks round-trips correctly through dequantize_q3_k_block.
    #[test]
    fn test_q3_k_pack_roundtrip() {
        let d = f16::from_f32(2.0);
        // Use d=0.1 so that scales have good dynamic range away from 32
        let d = f16::from_f32(0.1);
        let values: Vec<f32> = (0..256)
            .map(|i| (((i as f32 * 0.37) % 8.0) - 4.0))  // range [-4, 4]
            .collect();

        let block = build_simple_q3_k_block(&values, d);
        let dequant = dequantize_q3_k_block(&block);

        // Verify the dequantized values are close to original (within quantization error)
        let mut max_err = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..256 {
            let err = (dequant[i] - values[i]).abs();
            if err > max_err { max_err = err; worst_idx = i; }
        }
        eprintln!("[Q3_K roundtrip] max_error={} at idx={}", max_err, worst_idx);
        eprintln!("[Q3_K roundtrip] values[{}]={}, dequant[{}]={}", worst_idx, values[worst_idx], worst_idx, dequant[worst_idx]);
        // Also dump a few values for comparison
        for i in 0..8 {
            eprintln!("  [{}] val={:.4} dequant={:.4} err={:.4}", i, values[i], dequant[i], (dequant[i]-values[i]).abs());
        }
        // Dump raw block data for debugging
        eprintln!("  d_f16_bits = {:02x}{:02x}", block[109], block[108]);
        eprintln!("  scales_raw[96..108] = {:02x?}", &block[96..108]);
        eprintln!("  hmask[0..32] = {:02x?}", &block[0..32]);
        eprintln!("  qs[32..64] first 16 = {:02x?}", &block[32..48]);
        // Manually check first sub-block dequant
        {
            let d_f32 = d.to_f32();
            let raw_scales_bytes = &block[96..108];
            let mut aux = [0u32; 4];
            let aux_bytes = unsafe { &mut *(aux.as_mut_ptr() as *mut [u8; 12]) };
            aux_bytes.copy_from_slice(raw_scales_bytes);
            let kmask1: u32 = 0x03030303;
            let kmask2: u32 = 0x0f0f0f0f;
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            let scales = unsafe { &*(aux.as_ptr() as *const [i8; 16]) };
            eprintln!("  dequant_scales[0..4] = {:?}", &scales[0..4]);
            let dl = d_f32 * (scales[0] as f32 - 32.0);
            eprintln!("  dl[0] = {} * ({}) = {}", d_f32, scales[0], dl);
            let q0 = block[32]; // first qs byte
            let qs0 = (q0 >> 0) & 3;
            let hm0 = block[0]; // first hmask byte
            let hmask_bit = (hm0 & 1) != 0;
            eprintln!("  q[0]={:02x} qs0={} hm[0]={:02x} hmask_bit={}", q0, qs0, hm0, hmask_bit);
            let bias = if hmask_bit { 0i8 } else { 4i8 };
            eprintln!("  dequant[0] = {} * ({} - {}) = {}", dl, qs0, bias, dl * (qs0 as i8 - bias) as f32);
        }
        // 3-bit quantization has limited range; values outside the representable range
        // are clamped, leading to large errors. This is expected behavior.
        assert!(max_err < 1.0, "Q3_K roundtrip max_error={} too large at idx={}", max_err, worst_idx);
    }

    /// Direct test of the native helper function to verify it works independently.
    #[test]
    fn test_q3k_native_helper_direct() {
        let d = f16::from_f32(0.1);
        let values: Vec<f32> = (0..256)
            .map(|i| (((i as f32 * 0.37) % 8.0) - 4.0))
            .collect();

        let block = build_simple_q3_k_block(&values, d);
        let expected = dequantize_q3_k_block(&block);

        // Test all 32 iterations (lane_offset=0,8,16,...,248; lanes=8 each)
        let mut all_output = vec![0.0f32; 256];
        for iter in 0..32 {
            let lane_offset = iter * 8;
            let mut buf = [0.0f32; 8];
            unsafe {
                crate::asm::x86_64::quant_gemv::q3k_decode_step_native(
                    block.as_ptr(),
                    lane_offset as u64,
                    0.0,    // d_f32 (unused)
                    32,     // qs_offset
                    0,      // hmask_offset
                    8,      // lanes
                    buf.as_mut_ptr(),
                );
            }
            all_output[lane_offset..lane_offset + 8].copy_from_slice(&buf);
        }

        let mut max_err = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..256 {
            let err = (all_output[i] - expected[i]).abs();
            if err > 0.01 {
                eprintln!("[NATIVE-HELPER] MISMATCH at [{}]: got={}, expected={}", i, all_output[i], expected[i]);
            }
            if err > max_err {
                max_err = err;
                worst_idx = i;
            }
        }
        eprintln!("[NATIVE-HELPER] max_err={} at idx={}", max_err, worst_idx);
        assert!(max_err < 1e-5, "Native helper output differs from reference: max_err={} at idx={}", max_err, worst_idx);
    }

    #[test]
    fn test_q3_k_gemv_k256_n2_jit_execution() {
        let k: usize = 256;
        let n: usize = 2;

        let prog = build_q3_k_gemv_prog(k, n);
        let code = compile_vm_prog(&prog);

        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();

        // Create test values that exercise the quantization range with large values
        let d = f16::from_f32(1.0);
        let mut weight_blocks = Vec::with_capacity(n * Q3_K_BLOCK_BYTES);
        for row in 0..n {
            let values: Vec<f32> = (0..k)
                .map(|i| ((((row * k + i) as f32 * 0.37) % 8.0) - 4.0))
                .collect();
            let block = build_simple_q3_k_block(&values, d);
            weight_blocks.extend_from_slice(&block);
        }

        let activation: Vec<f32> = (0..k).map(|i| ((i % 10) as f32 + 1.0)).collect();

        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };
        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };

        let expected = ref_q3_k_gemv(&activation, &weight_blocks, n, k);

        for j in 0..n {
            let diff = (output[j] - expected[j]).abs();
            let rel = if expected[j].abs() > 1e-6 { diff / expected[j].abs() } else { diff };
            assert!(
                rel < 0.05,
                "Q3_K GEMV [{}] got={}, expected={}, diff={}, rel={}",
                j, output[j], expected[j], diff, rel
            );
        }
    }

    // ── Q4_K GEMV build ───────────────────────────────────────────────────

    fn build_q4_k_gemv_prog(k: usize, n: usize) -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(1),
            n, k,
            QuantType::Q4K,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q4_K GEMV emit should succeed");

        prog
    }

    // ── Q5_0 GEMV build ───────────────────────────────────────────────────

    fn build_q5_0_gemv_prog(k: usize, n: usize) -> VmProgram {
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(1),
            n, k,
            QuantType::Q5_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q5_0 GEMV emit should succeed");

        prog
    }

    /// Diagnostic test: dump VmInstr sequence + register allocation for Q4_0 GEMV K=32 N=1
    /// to identify why the FMA accumulator is not accumulating across ei loop iterations.
    #[test]
    fn test_q4_0_gemv_diagnostic_vm_instr_dump() {
        let k: usize = 32;
        let n: usize = 1;

        let prog = build_q4_0_gemv_prog(k, n);

        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let alloc = RegAllocator::new(&profile).allocate(&prog).unwrap();

        // Dump register allocation mapping
        eprintln!("=== Q4_0 GEMV K=32 N=1 Register Allocation ===");
        for (vreg_id, phys) in &alloc.mapping {
            match phys {
                PhysReg::Vec(v) => eprintln!("  VReg({}) -> ymm{}", vreg_id.0, v.0),
                PhysReg::Gpr(g) => eprintln!("  VReg({}) -> gpr{}", vreg_id.0, g.0),
                PhysReg::Spilled(slot) => eprintln!("  VReg({}) -> spill_slot={}", vreg_id.0, slot),
                _ => eprintln!("  VReg({}) -> {:?}", vreg_id.0, phys),
            }
        }
        eprintln!("  Spill slots:");
        for slot in &alloc.spills {
            eprintln!("    VReg({}) -> spill offset={} size={}", slot.vreg.0, slot.offset, slot.size);
        }

        // Dump all instructions with register info
        eprintln!("\n=== VmInstr Sequence ===");
        for (i, instr) in prog.instrs.iter().enumerate() {
            let vreg_info = match instr {
                VmInstr::VecLoad { dst, base, .. } => format!("dst=VReg({}) base=VReg({})", dst.0, base.0),
                VmInstr::VecBinOp { dst, a, b, .. } => format!("dst=VReg({}) a=VReg({}) b=VReg({})", dst.0, a.0, b.0),
                VmInstr::Fma { dst, acc, a, b, .. } => {
                    let dst_phys = alloc.get_vec(*dst).map(|v| format!("ymm{}", v.0)).unwrap_or("?".into());
                    let acc_phys = alloc.get_vec(*acc).map(|v| format!("ymm{}", v.0)).unwrap_or("?".into());
                    let a_phys = alloc.get_vec(*a).map(|v| format!("ymm{}", v.0)).unwrap_or("?".into());
                    let b_phys = alloc.get_vec(*b).map(|v| format!("ymm{}", v.0)).unwrap_or("?".into());
                    let acc_spilled = alloc.spills.iter().any(|s| s.vreg == *acc);
                    format!(
                        "dst=VReg({})[{}] acc=VReg({})[{}]{} a=VReg({})[{}] b=VReg({})[{}]",
                        dst.0, dst_phys, acc.0, acc_phys,
                        if acc_spilled { " SPILLED" } else { "" },
                        a.0, a_phys, b.0, b_phys
                    )
                },
                VmInstr::QuantBlockLoad { dst, base, .. } => format!("dst=VReg({}) base=VReg({})", dst.0, base.0),
                VmInstr::QuantInterleave { dst, lo, hi, .. } => {
                    let dst_phys = alloc.get_vec(*dst).map(|v| format!("ymm{}", v.0)).unwrap_or("?".into());
                    let lo_phys = alloc.get_vec(*lo).map(|v| format!("ymm{}", v.0)).unwrap_or("?".into());
                    let hi_phys = alloc.get_vec(*hi).map(|v| format!("ymm{}", v.0)).unwrap_or("?".into());
                    format!("dst=VReg({})[{}] lo=VReg({})[{}] hi=VReg({})[{}]",
                        dst.0, dst_phys, lo.0, lo_phys, hi.0, hi_phys)
                },
                VmInstr::Broadcast { dst, .. } => format!("dst=VReg({})", dst.0),
                VmInstr::HReduce { dst, src, .. } => format!("dst=VReg({}) src=VReg({})", dst.0, src.0),
                VmInstr::VecScalarStore { base, src, .. } => format!("base=VReg({}) src=VReg({})", base.0, src.0),
                VmInstr::GprBinOp { dst, .. } => format!("dst=VReg({})", dst.0),
                VmInstr::GprLoadImm { dst, .. } => format!("dst=VReg({})", dst.0),
                VmInstr::LoadPtr { dst, .. } => format!("dst=VReg({})", dst.0),
                VmInstr::LoopBegin { .. } => "LoopBegin".into(),
                VmInstr::LoopEnd { .. } => "LoopEnd".into(),
                _ => format!("{:?}", instr),
            };
            eprintln!("[{:3}] {:?} | {}", i, std::mem::discriminant(instr), vreg_info);
        }
    }

    // ── Q4_K GEMV emit + compile smoke test ────────────────────────────────

    #[test]
    fn test_q4_k_gemv_emit_compile() {
        let k: usize = 256;
        let n: usize = 2;

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        let result = emit_quant_gemm_inline(
            &mut prog, BoundExpr::Const(1), n, k,
            QuantType::Q4K, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        );
        assert!(result.is_ok(), "Q4_K emit failed: {:?}", result.err());
        eprintln!("[Q4_K] emit OK, {} instrs", prog.instrs.len());

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_K] compiled {} bytes", code.len());
        assert!(code.len() > 0, "Q4_K compiled code should not be empty");
    }

    // ── Q5_0 GEMV emit + compile smoke test ────────────────────────────────

    #[test]
    fn test_q5_0_gemv_emit_compile() {
        let k: usize = 32;
        let n: usize = 2;

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        let result = emit_quant_gemm_inline(
            &mut prog, BoundExpr::Const(1), n, k,
            QuantType::Q5_0, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        );
        assert!(result.is_ok(), "Q5_0 emit failed: {:?}", result.err());
        eprintln!("[Q5_0] emit OK, {} instrs", prog.instrs.len());

        let code = compile_vm_prog(&prog);
        eprintln!("[Q5_0] compiled {} bytes", code.len());
        assert!(code.len() > 0, "Q5_0 compiled code should not be empty");
    }

    // ── Q6_K GEMV emit + compile smoke test ────────────────────────────────

    #[test]
    fn test_q6_k_gemv_emit_compile() {
        let k: usize = 256;
        let n: usize = 2;

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        let result = emit_quant_gemm_inline(
            &mut prog, BoundExpr::Const(1), n, k,
            QuantType::Q6K, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        );
        assert!(result.is_ok(), "Q6_K emit failed: {:?}", result.err());
        eprintln!("[Q6_K] emit OK, {} instrs", prog.instrs.len());

        let code = compile_vm_prog(&prog);
        eprintln!("[Q6_K] compiled {} bytes", code.len());
        assert!(code.len() > 0, "Q6_K compiled code should not be empty");
    }

    // ── Q2_K GEMV emit + compile smoke test ────────────────────────────────

    #[test]
    fn test_q2_k_gemv_emit_compile() {
        let k: usize = 256;
        let n: usize = 2;

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        let result = emit_quant_gemm_inline(
            &mut prog, BoundExpr::Const(1), n, k,
            QuantType::Q2K, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        );
        assert!(result.is_ok(), "Q2_K emit failed: {:?}", result.err());
        eprintln!("[Q2_K] emit OK, {} instrs", prog.instrs.len());

        let code = compile_vm_prog(&prog);
        eprintln!("[Q2_K] compiled {} bytes", code.len());
        assert!(code.len() > 0, "Q2_K compiled code should not be empty");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Q4_K GEMV — Full JIT execution test with numerical validation
    // ═══════════════════════════════════════════════════════════════════════════

    const Q4_K_BLOCK_SIZE: usize = 256;
    const Q4_K_BLOCK_BYTES: usize = 144;

    /// Build a Q4_K block with known values for testing.
    ///
    /// Q4_K layout (144 bytes):
    ///   d(f16)       [offset 0, 2 bytes] — super-block scale
    ///   dmin(f16)    [offset 2, 2 bytes] — super-block minimum scale
    ///   scales[12]   [offset 4, 12 bytes] — 6-bit packed sub-block scales (KQuant6Bit)
    ///   qs[128]      [offset 16, 128 bytes] — 4-bit packed quant data (PackedNibbles)
    ///
    /// Dequant: For each sub-block (groups of 2 per 64-element run):
    ///   effective_scale = d * get_scale_min_k4(is).0
    ///   effective_min   = dmin * get_scale_min_k4(is).1
    ///   value = nibble * effective_scale - effective_min
    fn pack_q4_k_block(values: &[f32], d: f16, dmin: f16) -> [u8; Q4_K_BLOCK_BYTES] {
        assert_eq!(values.len(), Q4_K_BLOCK_SIZE);
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        let d_f32 = d.to_f32();

        // Write d and dmin at offsets 0 and 2
        let d_bits = d.to_bits();
        block[0] = (d_bits & 0xFF) as u8;
        block[1] = ((d_bits >> 8) & 0xFF) as u8;
        let dmin_bits = dmin.to_bits();
        block[2] = (dmin_bits & 0xFF) as u8;
        block[3] = ((dmin_bits >> 8) & 0xFF) as u8;

        // Compute raw sc[8] (6-bit values) for 8 sub-blocks of 32 elements.
        // Set m=0 for all sub-blocks (dmin=0 in our test).
        // nibble range [0, 15]: value = nibble * (d * sc)
        let mut raw_sc = [0u8; 8];
        for is in 0..8 {
            let start = is * 32;
            let mut max_val = 0.0f32;
            for l in 0..32 {
                let v = values[start + l];
                if v > max_val { max_val = v; }
            }
            if d_f32.abs() > 1e-10 {
                let sc = max_val / (15.0 * d_f32);
                raw_sc[is] = sc.round().clamp(0.0, 63.0) as u8;
            }
        }

        // Pack sc[8] and m[8]=0 into scales[12] using inverse KQuant6Bit.
        // Forward decode (get_scale_min_k4):
        //   j < 4:  sc[j] = scales[j] & 63,    m[j] = scales[j+4] & 63
        //   j >= 4: sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        //           m[j]  = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
        // Inverse (j < 4): scales[j] = sc[j], scales[j+4] = m[j] = 0
        // Inverse (j >= 4): scales[j+4] low4 = sc[j]&0xF; scales[j-4] high2 = sc[j]>>4
        for j in 0..4 {
            block[4 + j] = raw_sc[j] & 0x3F;
            block[4 + j + 4] = 0; // m[j] = 0
        }
        for j in 4..8 {
            block[4 + j + 4] = (block[4 + j + 4] & 0xF0) | (raw_sc[j] & 0x0F) as u8;
            block[4 + j - 4] = (block[4 + j - 4] & 0x3F) | (((raw_sc[j] >> 4) & 0x03) << 6) as u8;
        }

        // Quantize values into qs[128].
        // Q4_K: 4 groups x 2 sub-blocks per group x 32 elements per sub-block = 256
        for g in 0..4 {
            for run in 0..2 {
                let is = g * 2 + run;
                let effective_scale = d_f32 * raw_sc[is] as f32;
                for l in 0..32 {
                    let v = values[g * 64 + run * 32 + l];
                    let nibble = if effective_scale.abs() > 1e-10 {
                        (v / effective_scale).round().clamp(0.0, 15.0) as u8
                    } else {
                        0
                    };
                    let qs_byte_idx = g * 32 + l;
                    if run == 0 {
                        block[16 + qs_byte_idx] = (block[16 + qs_byte_idx] & 0xF0) | (nibble & 0x0F);
                    } else {
                        block[16 + qs_byte_idx] = (block[16 + qs_byte_idx] & 0x0F) | ((nibble & 0x0F) << 4);
                    }
                }
            }
        }

        block
    }

    /// Dequantize a single Q4_K block to f32 values (llama.cpp reference algorithm).
    fn dequantize_q4_k_block(block: &[u8]) -> [f32; 256] {
        assert_eq!(block.len(), Q4_K_BLOCK_BYTES);
        let mut output = [0.0f32; 256];

        let d = f16::from_bits((block[0] as u16) | ((block[1] as u16) << 8)).to_f32();
        let dmin = f16::from_bits((block[2] as u16) | ((block[3] as u16) << 8)).to_f32();
        let scales: [u8; 12] = block[4..16].try_into().unwrap();

        #[inline(always)]
        fn get_scale_min_k4(j: usize, sc: &[u8; 12]) -> (f32, f32) {
            if j < 4 {
                ((sc[j] & 63) as f32, (sc[j + 4] & 63) as f32)
            } else {
                let s = ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as f32;
                let m = ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f32;
                (s, m)
            }
        }

        let mut is = 0usize;
        for g in 0..4 {
            let q_off = g * 32;
            let out_off = g * 64;
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let d1 = d * sc1;
            let neg_m1 = dmin * m1;
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc2;
            let neg_m2 = dmin * m2;

            for l in 0..32 {
                output[out_off + l] = d1 * (block[16 + q_off + l] & 0xF) as f32 - neg_m1;
            }
            for l in 0..32 {
                output[out_off + 32 + l] = d2 * (block[16 + q_off + l] >> 4) as f32 - neg_m2;
            }
            is += 2;
        }
        output
    }

    /// Scalar reference Q4_K GEMV.
    fn ref_q4_k_gemv(activation: &[f32], weight_blocks: &[u8], n: usize, k: usize) -> Vec<f32> {
        let num_blocks_per_row = k / Q4_K_BLOCK_SIZE;
        let mut output = vec![0.0f32; n];
        for row in 0..n {
            let mut sum = 0.0f32;
            for blk in 0..num_blocks_per_row {
                let blk_offset = (row * num_blocks_per_row + blk) * Q4_K_BLOCK_BYTES;
                let block_data = &weight_blocks[blk_offset..blk_offset + Q4_K_BLOCK_BYTES];
                let dequant = dequantize_q4_k_block(block_data);
                let base = blk * Q4_K_BLOCK_SIZE;
                for i in 0..Q4_K_BLOCK_SIZE {
                    if base + i < k {
                        sum += activation[base + i] * dequant[i];
                    }
                }
            }
            output[row] = sum;
        }
        output
    }

    #[test]
    fn test_q4_k_gemv_k256_n2_jit_execution() {
        let k: usize = 256;
        let n: usize = 2;

        eprintln!("[Q4_K JIT] Step 1: building program...");
        let prog = build_q4_k_gemv_prog(k, n);
        eprintln!("[Q4_K JIT] prog_instrs={}", prog.instrs.len());

        eprintln!("[Q4_K JIT] Step 2: compiling...");
        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_K JIT] code_size={} bytes", code.len());

        eprintln!("[Q4_K JIT] Step 3: creating layer...");
        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
        eprintln!("[Q4_K JIT] Step 4: creating test data...");

        let d = f16::from_f32(1.0);
        let dmin = f16::from_f32(0.0);
        let mut weight_blocks = Vec::with_capacity(n * Q4_K_BLOCK_BYTES);
        for row in 0..n {
            let values: Vec<f32> = (0..k)
                .map(|i| ((((row * k + i) as f32 * 0.37) % 16.0)))
                .collect();
            let block = pack_q4_k_block(&values, d, dmin);
            weight_blocks.extend_from_slice(&block);
        }

        let activation: Vec<f32> = (0..k).map(|i| ((i % 10) as f32 + 1.0) * 0.1).collect();

        eprintln!("[Q4_K JIT] Step 5: executing JIT code...");
        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };
        eprintln!("[Q4_K JIT] Step 6: JIT execution complete.");

        let expected = ref_q4_k_gemv(&activation, &weight_blocks, n, k);

        eprintln!("[Q4_K JIT] output: {:?}", &output[..n.min(4)]);
        eprintln!("[Q4_K REF] expected: {:?}", &expected[..n.min(4)]);

        for j in 0..n {
            let diff = (output[j] - expected[j]).abs();
            let rel = if expected[j].abs() > 1e-6 { diff / expected[j].abs() } else { diff };
            assert!(
                rel < 0.05,
                "Q4_K GEMV [{}] got={}, expected={}, diff={}, rel={}",
                j, output[j], expected[j], diff, rel
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Q5_0 GEMV — Full JIT execution test with numerical validation
    // ═══════════════════════════════════════════════════════════════════════════

    const Q5_0_BLOCK_SIZE: usize = 32;
    const Q5_0_BLOCK_BYTES: usize = 22;

    /// Build Q5_0 blocks from values and scale.
    ///
    /// Q5_0 layout (22 bytes):
    ///   d(f16)    [offset 0, 2 bytes]  — block scale
    ///   qh[4]    [offset 2, 4 bytes]  — high bit plane (32 bits, one per element)
    ///   qs[16]   [offset 6, 16 bytes] — low nibbles (4 bits each, 2 per byte)
    ///
    /// Dequant: value = ((qs_nibble) | (qh_bit << 4)) - 16) * d
    fn pack_q5_0_blocks(values: &[i8], scale: f16) -> Vec<u8> {
        assert!(values.len() % Q5_0_BLOCK_SIZE == 0);
        let num_blocks = values.len() / Q5_0_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks * Q5_0_BLOCK_BYTES);

        for blk in 0..num_blocks {
            let scale_bits = scale.to_bits();
            blocks.push((scale_bits & 0xFF) as u8);
            blocks.push(((scale_bits >> 8) & 0xFF) as u8);

            // Build high bit plane: one bit per element, packed into 4 bytes (32 bits)
            let mut qh: u32 = 0;
            for i in 0..Q5_0_BLOCK_SIZE {
                let raw = values[blk * Q5_0_BLOCK_SIZE + i].clamp(-16, 15);
                let unsigned_val = (raw + 16) as u8;
                if (unsigned_val & 0x10) != 0 {
                    qh |= 1u32 << i;
                }
            }
            blocks.push((qh & 0xFF) as u8);
            blocks.push(((qh >> 8) & 0xFF) as u8);
            blocks.push(((qh >> 16) & 0xFF) as u8);
            blocks.push(((qh >> 24) & 0xFF) as u8);

            // Pack low nibbles (bits 0-3 of unsigned value)
            for i in 0..16 {
                let raw_lo = values[blk * Q5_0_BLOCK_SIZE + i].clamp(-16, 15);
                let raw_hi = values[blk * Q5_0_BLOCK_SIZE + 16 + i].clamp(-16, 15);
                let lo = ((raw_lo + 16) as u8) & 0x0F;
                let hi = ((raw_hi + 16) as u8) & 0x0F;
                blocks.push(lo | (hi << 4));
            }
        }
        blocks
    }

    /// Scalar reference Q5_0 GEMV.
    fn ref_q5_0_gemv(activation: &[f32], weight_blocks: &[u8], n: usize, k: usize) -> Vec<f32> {
        let num_blocks_per_row = k / Q5_0_BLOCK_SIZE;
        let mut output = vec![0.0f32; n];

        for j in 0..n {
            let row_offset = j * num_blocks_per_row * Q5_0_BLOCK_BYTES;
            let mut sum = 0.0f32;

            for blk in 0..num_blocks_per_row {
                let blk_offset = row_offset + blk * Q5_0_BLOCK_BYTES;
                let scale_bits = (weight_blocks[blk_offset] as u16)
                    | ((weight_blocks[blk_offset + 1] as u16) << 8);
                let scale_f32 = f16::from_bits(scale_bits).to_f32();

                let qh = (weight_blocks[blk_offset + 2] as u32)
                    | ((weight_blocks[blk_offset + 3] as u32) << 8)
                    | ((weight_blocks[blk_offset + 4] as u32) << 16)
                    | ((weight_blocks[blk_offset + 5] as u32) << 24);

                for i in 0..16 {
                    let byte = weight_blocks[blk_offset + 6 + i];

                    // Element i: low nibble + high bit from qh bit i
                    let lo_full = (byte & 0x0F) as u32 | (((qh >> i) & 1) << 4);
                    sum += activation[blk * 32 + i] * (lo_full as i32 - 16) as f32 * scale_f32;

                    // Element 16+i: hi nibble + high bit from qh bit (16+i)
                    let idx_hi = 16 + i;
                    let hi_full = ((byte >> 4) & 0x0F) as u32 | (((qh >> idx_hi) & 1) << 4);
                    sum += activation[blk * 32 + idx_hi] * (hi_full as i32 - 16) as f32 * scale_f32;
                }
            }
            output[j] = sum;
        }
        output
    }

    #[test]
    fn test_q5_0_gemv_k32_n2_jit_execution() {
        let k: usize = 32;
        let n: usize = 2;

        eprintln!("[Q5_0 JIT] Step 1: building program...");
        let prog = build_q5_0_gemv_prog(k, n);
        eprintln!("[Q5_0 JIT] prog_instrs={}", prog.instrs.len());

        eprintln!("[Q5_0 JIT] Step 2: compiling...");
        let code = compile_vm_prog(&prog);
        eprintln!("[Q5_0 JIT] code_size={} bytes", code.len());

        eprintln!("[Q5_0 JIT] Step 3: creating layer...");
        let layer = CompiledLayer::from_code(&code, 0, 0).unwrap();
        eprintln!("[Q5_0 JIT] Step 4: creating test data...");

        // Values in 5-bit signed range [-16, 15]
        let weight_values: Vec<i8> = (0..k * n).map(|i| ((i % 32) as i8) - 16).collect();
        let scale = f16::from_f32(0.5);
        let weight_blocks = pack_q5_0_blocks(&weight_values, scale);

        let activation: Vec<f32> = (0..k).map(|i| (i + 1) as f32).collect();

        eprintln!("[Q5_0 JIT] Step 5: executing JIT code...");
        let mut output = vec![0.0f32; n];
        unsafe { exec_gemv(&layer, &activation, &weight_blocks, &mut output) };
        eprintln!("[Q5_0 JIT] Step 6: JIT execution complete.");

        let expected = ref_q5_0_gemv(&activation, &weight_blocks, n, k);

        eprintln!("[Q5_0 JIT] output: {:?}", &output[..n.min(4)]);
        eprintln!("[Q5_0 REF] expected: {:?}", &expected[..n.min(4)]);

        for j in 0..n {
            let diff = (output[j] - expected[j]).abs();
            let rel = if expected[j].abs() > 1e-6 { diff / expected[j].abs() } else { diff };
            assert!(
                rel < 0.05,
                "Q5_0 GEMV [{}] got={}, expected={}, diff={}, rel={}",
                j, output[j], expected[j], diff, rel
            );
        }
    }
}
