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

    // ── Q3_K GEMV ──────────────────────────────────────────────────────────

    const Q3_K_BLOCK_SIZE: usize = 256;
    const Q3_K_BLOCK_BYTES: usize = 110;

    /// Inverse of Q3KExtended rearrangement: given 16 desired int8 scale values
    /// (interpreted as unsigned 6-bit values stored in byte bits [0..5]),
    /// compute the 12 raw bytes that dequantize to those values.
    fn pack_q3k_extended_scales(target_scales: &[u8; 16]) -> [u8; 12] {
        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;

        let mut desired_aux = [0u32; 4];
        for k in 0..4 {
            let mut v: u32 = 0;
            for b in 0..4 {
                let s = target_scales[k * 4 + b] as u32;
                v |= (s & 0x0F) << (b * 8);
                v |= ((s >> 4) & 0x03) << (b * 8 + 4);
            }
            desired_aux[k] = v;
        }

        let orig0 = (desired_aux[0] & kmask2) | ((desired_aux[2] & kmask2) << 4);
        let orig1 = (desired_aux[1] & kmask2) | ((desired_aux[3] & kmask2) << 4);

        let tmp_part0 = (desired_aux[0] >> 4) & kmask1;
        let tmp_part2 = (desired_aux[1] >> 4) & kmask1;
        let tmp_part4 = (desired_aux[2] >> 4) & kmask1;
        let tmp_part6 = (desired_aux[3] >> 4) & kmask1;
        let orig2 = tmp_part0 | (tmp_part2 << 2) | (tmp_part4 << 4) | (tmp_part6 << 6);

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

        let q = &block[32..96];
        let hm = &block[0..32];

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
    fn build_simple_q3_k_block(values: &[f32], d: f16) -> [u8; Q3_K_BLOCK_BYTES] {
        assert_eq!(values.len(), Q3_K_BLOCK_SIZE);
        let mut block = [0u8; Q3_K_BLOCK_BYTES];
        let d_f32 = d.to_f32();

        let mut raw_scales = [0u8; 16];
        for is in 0..16 {
            let start = is * 16;
            let mut max_abs = 0.0f32;
            for l in 0..16 {
                let v = values[start + l].abs();
                if v > max_abs { max_abs = v; }
            }
            if d_f32.abs() > 1e-10 {
                let s = max_abs / (d_f32 * 3.5) + 32.0;
                raw_scales[is] = s.round().clamp(0.0, 63.0) as u8;
            } else {
                raw_scales[is] = 32;
            }
        }

        let packed_scales = pack_q3k_extended_scales(&raw_scales);
        block[96..108].copy_from_slice(&packed_scales);

        let d_bits = d.to_bits();
        block[108] = (d_bits & 0xFF) as u8;
        block[109] = ((d_bits >> 8) & 0xFF) as u8;

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

                        let (qs_val, set_hmask) = if qi >= 0 {
                            (qi as u8, true)
                        } else {
                            ((qi + 4) as u8, false)
                        };

                        let qs_byte_idx = seg * 32 + run * 16 + l;
                        block[32 + qs_byte_idx] |= (qs_val & 0x03) << (j * 2);

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

    /// Test that pack_q3_k_blocks round-trips correctly through dequantize_q3_k_block.
    #[test]
    fn test_q3_k_pack_roundtrip() {
        let d = f16::from_f32(2.0);
        let d = f16::from_f32(0.1);
        let values: Vec<f32> = (0..256)
            .map(|i| (((i as f32 * 0.37) % 8.0) - 4.0))
            .collect();

        let block = build_simple_q3_k_block(&values, d);
        let dequant = dequantize_q3_k_block(&block);

        let mut max_err = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..256 {
            let err = (dequant[i] - values[i]).abs();
            if err > max_err { max_err = err; worst_idx = i; }
        }
        eprintln!("[Q3_K roundtrip] max_error={} at idx={}", max_err, worst_idx);
        eprintln!("[Q3_K roundtrip] values[{}]={}, dequant[{}]={}", worst_idx, values[worst_idx], worst_idx, dequant[worst_idx]);
        for i in 0..8 {
            eprintln!("  [{}] val={:.4} dequant={:.4} err={:.4}", i, values[i], dequant[i], (dequant[i]-values[i]).abs());
        }
        eprintln!("  d_f16_bits = {:02x}{:02x}", block[109], block[108]);
        eprintln!("  scales_raw[96..108] = {:02x?}", &block[96..108]);
        eprintln!("  hmask[0..32] = {:02x?}", &block[0..32]);
        eprintln!("  qs[32..64] first 16 = {:02x?}", &block[32..48]);
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
            let q0 = block[32];
            let qs0 = (q0 >> 0) & 3;
            let hm0 = block[0];
            let hmask_bit = (hm0 & 1) != 0;
            eprintln!("  q[0]={:02x} qs0={} hm[0]={:02x} hmask_bit={}", q0, qs0, hm0, hmask_bit);
            let bias = if hmask_bit { 0i8 } else { 4i8 };
            eprintln!("  dequant[0] = {} * ({} - {}) = {}", dl, qs0, bias, dl * (qs0 as i8 - bias) as f32);
        }
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

        let mut all_output = vec![0.0f32; 256];
        for iter in 0..32 {
            let lane_offset = iter * 8;
            let mut buf = [0.0f32; 8];
            unsafe {
                crate::asm::x86_64::quant_gemv::q3k_decode_step_native(
                    block.as_ptr(),
                    lane_offset as u64,
                    0.0,
                    32,
                    0,
                    8,
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

    // ── Q4_0 QuantGather x86 real-execution oracle ────────────────────────
    //
    // BCE-20260708-Q4_0 saga 定位：Qwen3 BF16 E2E PASS("Paris") + Q4_0 E2E FAIL(乱码) →
    // bug 100% Q4_0-quant 特有。QuantGather(embed) 与 QuantGemm(layer weights) 共用同一
    // DecodeTraceBuilder，但 saga 16+ 轮未能用可信 oracle 验证 JIT 执行数值。
    // 本测试构造已知 Q4_0 embed weight，emit QuantGather，编译后真实执行，对比手算参考。
    //
    // 测试规格：hidden_dim=32 (1 block), vocab=2 (token 0/1), seq_len=1 (查 token 0)。
    // Q4_0 block = 2 字节 f16 scale d + 16 字节 packed nibbles = 18 字节。
    // SPLIT 布局 (llama.cpp 权威)：byte j 低 nibble → elem[j]，高 nibble → elem[j+16]。
    //   token 0 block: d=f16(1.0), qs[0]=0x9A (lo=0xA=10, hi=0x9=9), 其余 qs[i]=0x88 (lo=hi=8=zero)。
    //   手算：elem[0]=1.0*(10-8)=2.0；elem[16]=1.0*(9-8)=1.0；其余 elem=1.0*(8-8)=0.0。
    //
    // ABI (build_q4_0_gemv_prog 同款)：AbiArg(0)=indices_ptr(rdi), AbiArg(1)=embed_ptr(rsi),
    //   StackArg(24)=output_ptr。StackArg(24) 在 prologue `push rbp; mov rbp,rsp` 后读 [rbp+24]，
    //   对应 SysV ABI 第 8 个参数 (arg 7, 0-indexed)。故 extern "C" fn 需 8 参数，output 放第 8 位。

    use crate::compiler::codegen::vm::quant_gather_emit::emit_quant_gather_inline;

    /// mmap 一段 RWX 内存并拷贝 JIT 机器码进去 (参考 autotuning/measure.rs ExecutableGemmBuffer::new)。
    /// 返回 (ptr, len) 用于调用与 munmap。
    fn make_exec_buffer(code: &[u8]) -> (*mut u8, usize) {
        assert!(!code.is_empty(), "empty JIT code");
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        let len = (code.len() + page_size - 1) & !(page_size - 1);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mmap failed");
        let ptr = ptr as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()) };
        let ret = unsafe { libc::mprotect(ptr as *mut _, len, libc::PROT_READ | libc::PROT_EXEC) };
        assert_eq!(ret, 0, "mprotect RX failed");
        (ptr, len)
    }

    #[test]
    fn test_q4_0_quant_gather_x86_oracle() {
        let hidden_dim: usize = 32; // 1 block (block_size=32)
        let vocab: usize = 2;
        // Q4_0 block_bytes = 18, row_stride = (32/32)*18 = 18 字节/token。
        let block_bytes: usize = 18;

        // ── 构造已知 Q4_0 embed weight buffer ──
        // token 0 block (18 字节): d=f16(1.0) + qs[0]=0x9A + qs[1..16]=0x88
        //   SPLIT: elem[0]=1.0*(10-8)=2.0, elem[16]=1.0*(9-8)=1.0, 其余=0.0
        // token 1 block (18 字节): d=f16(1.0) + qs[*]=0x88 (全 zero point)
        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits(); // little-endian 2 bytes
        let d_lo = (d_bits & 0xFF) as u8;
        let d_hi = ((d_bits >> 8) & 0xFF) as u8;

        let mut weight = vec![0u8; vocab * block_bytes];
        // token 0 at offset 0
        weight[0] = d_lo;
        weight[1] = d_hi;
        weight[2] = 0x9A; // qs[0]: lo=0xA=10, hi=0x9=9
        for j in 1..16 {
            weight[2 + j] = 0x88; // qs[1..16]: lo=hi=8 = zero point
        }
        // token 1 at offset 18: all zero point + d=1.0
        weight[18] = d_lo;
        weight[19] = d_hi;
        for j in 0..16 {
            weight[20 + j] = 0x88;
        }

        // ── 构造 indices: [0u32] (查 token 0, seq_len=1) ──
        let indices: [u32; 1] = [0u32];

        // ── 输出 buffer: seq_len * hidden_dim * 4 字节 (F32) ──
        let mut output = vec![0u8; 1 * hidden_dim * std::mem::size_of::<f32>()];

        // ── emit VmProgram (build_q4_0_gemv_prog 同款 ABI) ──
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            vocab,
            hidden_dim,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            None,
        ).expect("Q4_0 QuantGather emit should succeed");

        // ── 编译 ──
        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-ORACLE] JIT code {} bytes, {} instrs", code.len(), prog.instrs.len());

        // ── mmap + 执行 ──
        // ABI: AbiArg(0)=indices(rdi), AbiArg(1)=embed(rsi), StackArg(24)=output。
        // StackArg(24) = [rbp+24] = 第 8 个参数 (第 7 个参数在 [rbp+16] 被 skip)。
        // extern "C" 8 参数: (indices, embed, _p1, _p2, _p3, _p4, _p5, output)
        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GatherFn = unsafe extern "C" fn(
            *const u8, *const u8, usize, usize, usize, usize, usize, *mut u8,
        ) -> usize;
        let f: GatherFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe {
            f(
                indices.as_ptr() as *const u8,
                weight.as_ptr() as *const u8,
                0, 0, 0, 0, 0,            // args 3..7 unused (rdx/rcx/r8/r9/stack[16])
                output.as_mut_ptr() as *mut u8, // arg 8 = StackArg(24)
            )
        };
        eprintln!("[Q4_0-ORACLE] kernel returned {}", _ret);

        // ── 断言 ──
        let out_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output.as_ptr() as *const f32, hidden_dim)
        };
        eprintln!("[Q4_0-ORACLE] out[0..8]   = {:?}", &out_f32[0..8]);
        eprintln!("[Q4_0-ORACLE] out[8..16]  = {:?}", &out_f32[8..16]);
        eprintln!("[Q4_0-ORACLE] out[16..24] = {:?}", &out_f32[16..24]);
        eprintln!("[Q4_0-ORACLE] out[24..32] = {:?}", &out_f32[24..32]);
        eprintln!("[Q4_0-ORACLE] want: out[0]=2.0, out[16]=1.0, 其余=0.0");

        let mut pass = true;
        if (out_f32[0] - 2.0).abs() >= 1e-5 {
            eprintln!("[Q4_0-ORACLE] FAIL elem[0]: got {} want 2.0", out_f32[0]);
            pass = false;
        }
        if (out_f32[16] - 1.0).abs() >= 1e-5 {
            eprintln!("[Q4_0-ORACLE] FAIL elem[16]: got {} want 1.0", out_f32[16]);
            pass = false;
        }
        for i in 0..hidden_dim {
            if i != 0 && i != 16 && out_f32[i].abs() >= 1e-5 {
                eprintln!("[Q4_0-ORACLE] FAIL elem[{}]: got {} want 0.0", i, out_f32[i]);
                pass = false;
            }
        }

        // 释放 exec 内存
        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        if pass {
            eprintln!("[Q4_0-ORACLE] PASS — JIT emit+lower 数值正确 → bug 在接线 (caller 喂 ptr/offset)");
        } else {
            eprintln!("[Q4_0-ORACLE] FAIL — bug 在 emit/lower (actual vs expected 见上)");
        }
        assert!(pass, "Q4_0 QuantGather x86 oracle 数值不匹配 (actual vs expected 见 eprintln)");
    }

    /// 诊断变体：多字节非平凡 nibble，区分「4-byte 组内 INTERLEAVED」vs 其他模式。
    ///
    /// 构造：d=f16(1.0)，byte0=0x90 (lo=0, hi=9)，byte1=0xA0 (lo=0, hi=10)，
    /// byte2=0xB0 (lo=0, hi=11)，byte3=0xC0 (lo=0, hi=12)，其余 byte=0x88。
    /// 若是「4-byte 组内 INTERLEAVED」(lo 前 4 + hi 后 4)：out=[0,0,0,0, 9,10,11,12, 0...]
    /// 若是块级 SPLIT (lo→0..15, hi→16..31)：out[16..20]=[9,10,11,12]，out[0..4]=0
    #[test]
    fn test_q4_0_quant_gather_x86_oracle_pattern_diag() {
        let hidden_dim: usize = 32;
        let vocab: usize = 2;
        let block_bytes: usize = 18;

        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let d_lo = (d_bits & 0xFF) as u8;
        let d_hi = ((d_bits >> 8) & 0xFF) as u8;

        let mut weight = vec![0u8; vocab * block_bytes];
        weight[0] = d_lo;
        weight[1] = d_hi;
        // byte 0..3 各放一个独特 hi nibble，lo nibble 全 0 (zero point 偏移后 = -8，但为对比模式仍可识别)
        weight[2] = 0x90; // byte0: lo=0, hi=9
        weight[3] = 0xA0; // byte1: lo=0, hi=10
        weight[4] = 0xB0; // byte2: lo=0, hi=11
        weight[5] = 0xC0; // byte3: lo=0, hi=12
        for j in 4..16 {
            weight[2 + j] = 0x88; // byte4..15: lo=hi=8 = zero point
        }
        // token 1
        weight[18] = d_lo;
        weight[19] = d_hi;
        for j in 0..16 {
            weight[20 + j] = 0x88;
        }

        let indices: [u32; 1] = [0u32];
        let mut output = vec![0u8; 1 * hidden_dim * std::mem::size_of::<f32>()];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gather_inline(
            &mut prog, BoundExpr::Const(1), vocab, hidden_dim,
            QuantType::Q4_0, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, None,
        ).expect("emit should succeed");

        let code = compile_vm_prog(&prog);
        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GatherFn = unsafe extern "C" fn(
            *const u8, *const u8, usize, usize, usize, usize, usize, *mut u8,
        ) -> usize;
        let f: GatherFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ = unsafe { f(indices.as_ptr() as *const u8, weight.as_ptr() as *const u8,
                           0, 0, 0, 0, 0, output.as_mut_ptr() as *mut u8) };

        let out_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output.as_ptr() as *const f32, hidden_dim)
        };
        eprintln!("[Q4_0-DIAG] out[0..8]   = {:?}", &out_f32[0..8]);
        eprintln!("[Q4_0-DIAG] out[8..16]  = {:?}", &out_f32[8..16]);
        eprintln!("[Q4_0-DIAG] out[16..24] = {:?}", &out_f32[16..24]);
        eprintln!("[Q4_0-DIAG] out[24..32] = {:?}", &out_f32[24..32]);
        // hi nibbles 9,10,11,12 → 偏移 -8 后 = 1.0, 2.0, 3.0, 4.0
        eprintln!("[Q4_0-DIAG] 期望(4-byte 组内 INTERLEAVED): out[4..8]=[1,2,3,4], 其余=-8 或 0");
        eprintln!("[Q4_0-DIAG] 期望(块级 SPLIT): out[16..20]=[1,2,3,4], out[0..4]=-8");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }
        // 仅诊断，不断言（目的是打印实际模式供根因分析）
    }

    // ── Q4_0 QuantGemm x86 real-execution oracle ─────────────────────────
    //
    // BCE-20260709-Q4_0-SPLIT 序列第 2 步 (architect round 19):
    // QuantGather SPLIT 修复后 E2E 仍乱码 → QuantGemm (层权重 dequant, 走
    // QuantDequantFma 微内核, 非 DecodeTraceBuilder) 可能有同样 SPLIT bug.
    // 本测试喂已知 Q4_0 weight + 已知 activation, GEMV 输出对比手算 SPLIT 参考.
    //
    // 测试: m=1 (1 output), n=1, k=32 (1 block).
    // weight block: d=f16(1.0), qs[0]=0x9A(lo=10,hi=9), 其余 qs=0x88(zero=8).
    // SPLIT: elem[0]=1.0*(10-8)=2.0, elem[16]=1.0*(9-8)=1.0, 其余=0.
    // activation[0]=1.0, activation[16]=1.0, 其余=0.
    // output = act·weight = 1.0*2.0 + 1.0*1.0 = 3.0.
    #[test]
    fn test_q4_0_quant_gemm_x86_oracle() {
        use crate::compiler::codegen::vm::moe_quant_emit::emit_quant_gemm_inline;

        let m: usize = 1; // 1 output row
        let n: usize = 1; // 1 batch
        let k: usize = 32; // inner dim (1 block)
        let block_bytes: usize = 18;

        // ── weight: 1 block (m=1 row) ──
        // d=f16(1.0), qs[0]=0x9A, 其余 0x88
        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let mut weight = vec![0u8; m * block_bytes];
        weight[0] = (d_bits & 0xFF) as u8;
        weight[1] = ((d_bits >> 8) & 0xFF) as u8;
        weight[2] = 0x9A; // qs[0]: lo=10, hi=9
        for j in 1..16 {
            weight[2 + j] = 0x88;
        }

        // ── activation: [k, n] = [32, 1] col-major ──
        // act[0]=1.0, act[16]=1.0, 其余 0
        let mut act = vec![0.0f32; k * n];
        act[0] = 1.0;
        act[16] = 1.0;

        // ── output: [m, n] = [1, 1] ──
        let mut output = vec![0.0f32; m * n];

        // ── emit VmProgram (build_q4_0_gemv_prog 同款 ABI) ──
        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(m),
            n, k,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q4_0 QuantGemm emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-GEMM-ORACLE] JIT code {} bytes, {} instrs", code.len(), prog.instrs.len());

        // ── mmap + 执行 (同 QuantGather oracle ABI) ──
        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GemmFn = unsafe extern "C" fn(
            *const u8, *const u8, usize, usize, usize, usize, usize, *mut u8,
        ) -> usize;
        let f: GemmFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe {
            f(
                act.as_ptr() as *const u8,
                weight.as_ptr() as *const u8,
                0, 0, 0, 0, 0,
                output.as_mut_ptr() as *mut u8,
            )
        };
        eprintln!("[Q4_0-GEMM-ORACLE] kernel returned {}", _ret);

        let out_f32: &[f32] = &output;
        eprintln!("[Q4_0-GEMM-ORACLE] out = {:?}", out_f32);
        eprintln!("[Q4_0-GEMM-ORACLE] want: out[0] = 3.0 (1.0*2.0 + 1.0*1.0, SPLIT elem[0]=2.0 elem[16]=1.0)");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        // 断言: output[0] ≈ 3.0
        // 若 QuantGemm SPLIT 正确: weight elem[0]=2.0 配 act[0]=1.0, elem[16]=1.0 配 act[16]=1.0 → 3.0
        // 若 QuantGemm SPLIT 错误 (如 lo/hi 配错 activation): output 会是其他值 (如 2.0 或 1.0 或 0.0)
        let pass = (out_f32[0] - 3.0).abs() < 1e-3;
        if pass {
            eprintln!("[Q4_0-GEMM-ORACLE] PASS — QuantGemm SPLIT 正确, 层 dequant 对 → E2E 乱码另有根因");
        } else {
            eprintln!("[Q4_0-GEMM-ORACLE] FAIL — QuantGemm SPLIT bug, actual={} want 3.0", out_f32[0]);
        }
        assert!(pass, "Q4_0 QuantGemm x86 oracle: got {} want 3.0 (SPLIT act·weight)", out_f32[0]);
    }

    // ── Q4_0 QuantGemm x86 oracle — BF16 accum_dtype (真实推理配置) ───────
    //
    // 真实 Qwen3 Q4_0: geometry.dtype=BF16, compute_dtype=BF16 → accum_dtype=BF16.
    // 上面的 oracle 用 F32 accum, 真实推理用 BF16 accum. 本测试验 BF16 accum 下 QuantGemm 是否对.
    // Q4_0 decode 出 F32 → BF16 accum 累加. 2.0/1.0 等 BF16 可精确表示, 应仍 output=3.0.
    #[test]
    fn test_q4_0_quant_gemm_x86_oracle_bf16_accum() {
        use crate::compiler::codegen::vm::moe_quant_emit::emit_quant_gemm_inline;

        let m: usize = 1;
        let n: usize = 1;
        let k: usize = 32;
        let block_bytes: usize = 18;

        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let mut weight = vec![0u8; m * block_bytes];
        weight[0] = (d_bits & 0xFF) as u8;
        weight[1] = ((d_bits >> 8) & 0xFF) as u8;
        weight[2] = 0x9A;
        for j in 1..16 { weight[2 + j] = 0x88; }

        // activation F32 存储 (真实推理 act_dt=F32, 即使 compute_dtype=BF16)
        // act[0]=1.0, act[16]=1.0, 其余 0
        let mut act = vec![0.0f32; k * n];
        act[0] = 1.0;
        act[16] = 1.0;

        let mut output = vec![0.0f32; m * n];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        // BF16 accum_dtype (真实推理配置)
        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(m),
            n, k,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::BF16,
            DotProductCap::SimdAssisted,
        ).expect("Q4_0 QuantGemm BF16 emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-GEMM-BF16] JIT code {} bytes, {} instrs", code.len(), prog.instrs.len());

        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GemmFn = unsafe extern "C" fn(*const u8, *const u8, usize, usize, usize, usize, usize, *mut u8) -> usize;
        let f: GemmFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe {
            f(act.as_ptr() as *const u8, weight.as_ptr() as *const u8,
              0, 0, 0, 0, 0, output.as_mut_ptr() as *mut u8)
        };

        let out_f32: &[f32] = &output;
        eprintln!("[Q4_0-GEMM-BF16] out = {:?} (want 3.0, BF16 accum + F32 act 真实配置)", out_f32);

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        // BF16 可精确表示 2.0/1.0/3.0, 容差放宽到 1e-2 (BF16 累加噪声)
        let pass = (out_f32[0] - 3.0).abs() < 1e-2;
        if pass {
            eprintln!("[Q4_0-GEMM-BF16] PASS — BF16 accum 下 QuantGemm 对");
        } else {
            eprintln!("[Q4_0-GEMM-BF16] FAIL — BF16 accum bug, actual={} want 3.0", out_f32[0]);
        }
        assert!(pass, "Q4_0 QuantGemm BF16 accum oracle: got {} want 3.0", out_f32[0]);
    }

    // ── Q4_0 QuantGemm real-scale x86 oracle (architect round 23) ────────
    //
    // 6 单位 oracle 全过但 E2E 错 → architect round 23: 唯一未执行的 Q4_0 特有面
    // = 真实规模 in-op (m=5, n=2048, k=1024). BF16 过清共享, bug Q4_0 特有@规模.
    //
    // 设计: m=2, n=4, k=1024 (32 block). 每 (m,n) 输出唯一值便于定位错位.
    // weight[m][n] block0: qs[0] lo=(m+n+8)%16, hi=8(zero). d=1.0.
    // act[0]=1, act[16]=0 → output[m][n] = act[0]×elem[0] = (lo-8) = m+n (若 lo=m+n+8).
    //   即 output[i][j] = i+j (唯一, 易验错位).
    // output 布局 [m, n] row-major: output[m*n + n_idx].
    #[test]
    fn test_q4_0_quant_gemm_x86_oracle_realscale() {
        use crate::compiler::codegen::vm::moe_quant_emit::emit_quant_gemm_inline;

        let m: usize = 2;
        let n: usize = 4;
        let k: usize = 1024; // 32 blocks (真实规模)
        let block_bytes: usize = 18;
        let blocks_per_row: usize = k / 32; // 32

        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let d_lo = (d_bits & 0xFF) as u8;
        let d_hi = ((d_bits >> 8) & 0xFF) as u8;

        // weight: m 行 × n 列 × blocks_per_row blocks/row. layout [m, n, blocks, 18B]
        // weight_row_stride (n 维) = blocks_per_row * block_bytes = 576
        let row_bytes = blocks_per_row * block_bytes;
        let mut weight = vec![0u8; m * n * row_bytes];
        for mi in 0..m {
            for ni in 0..n {
                let base = (mi * n + ni) * row_bytes;
                // block0: elem[0]=ni (weight 与 mi 无关, 诊断: output[m][n]=n → weight_row_ptr 没随 m 前进)
                weight[base] = d_lo;
                weight[base + 1] = d_hi;
                let lo_val = ((ni + 8) % 16) as u8; // elem[0] = ni
                weight[base + 2] = lo_val | (0x80);
                for j in 1..16 { weight[base + 2 + j] = 0x88; }
                for b in 1..blocks_per_row {
                    let bs = base + b * block_bytes;
                    weight[bs] = d_lo;
                    weight[bs + 1] = d_hi;
                    for j in 0..16 { weight[bs + 2 + j] = 0x88; }
                }
            }
        }

        // activation [m, k] row-major: act[m_idx][k_idx] at act[m_idx*k + k_idx]
        // act[0][0]=1, act[1][0]=1 (两个 m 行的 k=0 处). a_row_stride = k*elem = 1024*4.
        let mut act = vec![0.0f32; m * k];
        act[0 * k + 0] = 1.0;  // m=0, k=0
        act[1 * k + 0] = 1.0;  // m=1, k=0

        let mut output = vec![0.0f32; m * n];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(m),
            n, k,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q4_0 QuantGemm real-scale emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-GEMM-RS] JIT code {} bytes, {} instrs", code.len(), prog.instrs.len());

        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GemmFn = unsafe extern "C" fn(*const u8, *const u8, usize, usize, usize, usize, usize, *mut u8) -> usize;
        let f: GemmFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe {
            f(act.as_ptr() as *const u8, weight.as_ptr() as *const u8,
              0, 0, 0, 0, 0, output.as_mut_ptr() as *mut u8)
        };

        let out_f32: &[f32] = &output;
        eprintln!("[Q4_0-GEMM-RS] out = {:?}", out_f32);
        eprintln!("[Q4_0-GEMM-RS] want: out[i][j] = j (0,1,2,3, 0,1,2,3) [weight 与 mi 无关]");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        // output[m*n + n_idx] = n_idx (weight elem[0]=n_idx, act=1)
        let mut pass = true;
        for mi in 0..m {
            for ni in 0..n {
                let idx = mi * n + ni;
                let want = ni as f32;
                if (out_f32[idx] - want).abs() >= 1e-2 {
                    eprintln!("[Q4_0-GEMM-RS] FAIL out[{}][{}] (idx={}): got {} want {}", mi, ni, idx, out_f32[idx], want);
                    pass = false;
                }
            }
        }
        if pass {
            eprintln!("[Q4_0-GEMM-RS] PASS — 真实规模 QuantGemm 对 (m={},n={},k={})", m, n, k);
        } else {
            eprintln!("[Q4_0-GEMM-RS] FAIL — 真实规模 in-op bug (architect round 23 TOP)");
        }
        assert!(pass, "Q4_0 QuantGemm real-scale oracle: 真实规模错位 (见 eprintln)");
    }

    // ── Q4_0 QuantGather real-scale x86 oracle (architect round 25) ──────
    //
    // 0 层截断乱码 → bug 在 embed(QuantGather) 或 lm_head(QuantGemm) 真实规模.
    // oracle 之前只测 hidden=32/64, 真实 hidden=1024 (32 block) 未测.
    // 本测试: hidden=1024, vocab=2, token 0, 每 block elem[0]=block_idx (唯一).
    //
    // 32 block × 32 elem/block. block b 的 byte0 lo→elem[b*32+0]=b, hi→elem[b*32+16]=-...
    // 设 d=1.0, qs[0] lo=(b+8)%16, hi=8(zero). elem[b*32+0]=b, 其余 0.
    #[test]
    fn test_q4_0_quant_gather_x86_oracle_realscale() {
        let hidden_dim: usize = 1024; // 32 blocks (真实 Qwen3 hidden)
        let vocab: usize = 2;
        let block_bytes: usize = 18;
        let blocks_per_row: usize = hidden_dim / 32; // 32

        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let d_lo = (d_bits & 0xFF) as u8;
        let d_hi = ((d_bits >> 8) & 0xFF) as u8;

        let row_bytes = blocks_per_row * block_bytes;
        let mut weight = vec![0u8; vocab * row_bytes];
        // token 0: 32 blocks, 每 block qs[0] lo=0xA(elem=2), hi=8(zero), 其余 qs=0x88
        // 验位置: out[b*32+0]=2 对所有 b, out[b*32+16]=0 (hi zero)
        for b in 0..blocks_per_row {
            let base = b * block_bytes;
            weight[base] = d_lo;
            weight[base + 1] = d_hi;
            weight[base + 2] = 0x8A; // lo=0xA=10→elem=2, hi=8→zero
            for j in 1..16 { weight[base + 2 + j] = 0x88; }
        }
        // token 1: all zero point
        for b in 0..blocks_per_row {
            let base = row_bytes + b * block_bytes;
            weight[base] = d_lo;
            weight[base + 1] = d_hi;
            for j in 0..16 { weight[base + 2 + j] = 0x88; }
        }

        let indices: [u32; 1] = [0u32];
        let mut output = vec![0u8; 1 * hidden_dim * std::mem::size_of::<f32>()];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            vocab,
            hidden_dim,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            None,
        ).expect("Q4_0 QuantGather real-scale emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-GATHER-RS] JIT code {} bytes, {} instrs", code.len(), prog.instrs.len());

        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GatherFn = unsafe extern "C" fn(*const u8, *const u8, usize, usize, usize, usize, usize, *mut u8) -> usize;
        let f: GatherFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe {
            f(indices.as_ptr() as *const u8, weight.as_ptr() as *const u8,
              0, 0, 0, 0, 0, output.as_mut_ptr() as *mut u8)
        };

        let out_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output.as_ptr() as *const f32, hidden_dim)
        };
        eprintln!("[Q4_0-GATHER-RS] out[0]={} out[16]={} out[32]={} out[48]={} (block0/1/2 lo)", out_f32[0], out_f32[16], out_f32[32], out_f32[48]);
        eprintln!("[Q4_0-GATHER-RS] want: out[b*32+0]=b for b in 0..32, 其余=0 (hi=8→0)");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        // SPLIT: block b 的 byte0 lo→elem[b*32+0]=2, hi→elem[b*32+16]=0(hi=8 zero)
        let mut pass = true;
        for b in 0..blocks_per_row {
            let lo_idx = b * 32;
            if (out_f32[lo_idx] - 2.0).abs() >= 1e-3 {
                eprintln!("[Q4_0-GATHER-RS] FAIL block{} lo (idx={}): got {} want 2", b, lo_idx, out_f32[lo_idx]);
                pass = false;
            }
            let hi_idx = b * 32 + 16;
            if out_f32[hi_idx].abs() >= 1e-3 {
                eprintln!("[Q4_0-GATHER-RS] FAIL block{} hi (idx={}): got {} want 0", b, hi_idx, out_f32[hi_idx]);
                pass = false;
            }
        }
        if pass {
            eprintln!("[Q4_0-GATHER-RS] PASS — QuantGather 真实规模 (hidden=1024, 32 block) 正确");
        } else {
            eprintln!("[Q4_0-GATHER-RS] FAIL — QuantGather 真实规模 bug");
        }
        assert!(pass, "Q4_0 QuantGather real-scale oracle: 真实规模错位 (见 eprintln)");
    }

    // ── Q4_0 QuantGemm large-n x86 oracle (lm_head 真实规模, architect round 25) ──
    //
    // 0 层截断乱码 + QuantGather/QuantGemm real-scale(m=2,n=4) 过 →
    // 剩余未测: lm_head 真实 n=vocab(151936). 本测试 n=2048 (足够大测 j 维迭代).
    // Qwen3 tie_word_embeddings=True → lm_head = embed (Q4_0, tied).
    // weight [n, k] = [2048, 1024]. m=1, act[0]=1.
    // 每 n 行的 block0 elem[0]=n_idx%8 (唯一标识, 验 j 维 weight_row_stride).
    #[test]
    fn test_q4_0_quant_gemm_x86_oracle_large_n() {
        use crate::compiler::codegen::vm::moe_quant_emit::emit_quant_gemm_inline;

        let m: usize = 1;
        let n: usize = 2048; // 大 n (lm_head 真实 vocab 量级)
        let k: usize = 1024;
        let block_bytes: usize = 18;
        let blocks_per_row: usize = k / 32;

        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        // weight [n, k]: 每 n 行 block0 qs[0] lo=(ni%8)+8, hi=8. elem[0]=ni%8.
        // 其余 block zero. weight_row_stride = blocks_per_row * block_bytes = 576.
        let row_bytes = blocks_per_row * block_bytes;
        let mut weight = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let base = ni * row_bytes;
            weight[base] = (d_bits & 0xFF) as u8;
            weight[base + 1] = ((d_bits >> 8) & 0xFF) as u8;
            let lo_val = ((ni % 8) + 8) as u8; // elem[0] = ni%8
            weight[base + 2] = lo_val | 0x80;
            for j in 1..16 { weight[base + 2 + j] = 0x88; }
            // block1..31 zero
            for b in 1..blocks_per_row {
                let bs = base + b * block_bytes;
                weight[bs] = (d_bits & 0xFF) as u8;
                weight[bs + 1] = ((d_bits >> 8) & 0xFF) as u8;
                for j in 0..16 { weight[bs + 2 + j] = 0x88; }
            }
        }

        // act [m, k]: act[0][0]=1, 其余 0
        let mut act = vec![0.0f32; m * k];
        act[0] = 1.0;

        let mut output = vec![0.0f32; m * n];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog, BoundExpr::Const(m), n, k,
            QuantType::Q4_0, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-GEMM-LARGEN] JIT {} bytes, {} instrs, n={}", code.len(), prog.instrs.len(), n);

        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GemmFn = unsafe extern "C" fn(*const u8, *const u8, usize, usize, usize, usize, usize, *mut u8) -> usize;
        let f: GemmFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe { f(act.as_ptr() as *const u8, weight.as_ptr() as *const u8, 0,0,0,0,0, output.as_mut_ptr() as *mut u8) };

        let out_f32: &[f32] = &output;
        // 验前 8 + 几个中间 n: output[ni] = ni%8
        eprintln!("[Q4_0-GEMM-LARGEN] out[0..8]={:?}", &out_f32[0..8]);
        eprintln!("[Q4_0-GEMM-LARGEN] out[1000..1004]={:?}", &out_f32[1000..1004]);
        eprintln!("[Q4_0-GEMM-LARGEN] want: out[ni]=ni%8");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        let mut pass = true;
        for ni in [0usize, 1, 7, 1000, 1500, 2047].iter() {
            let want = (*ni % 8) as f32;
            if (out_f32[*ni] - want).abs() >= 1e-2 {
                eprintln!("[Q4_0-GEMM-LARGEN] FAIL out[{}]: got {} want {}", ni, out_f32[*ni], want);
                pass = false;
            }
        }
        if pass {
            eprintln!("[Q4_0-GEMM-LARGEN] PASS — 大 n QuantGemm 对 (j 维 weight_row_stride 对)");
        } else {
            eprintln!("[Q4_0-GEMM-LARGEN] FAIL — 大 n j 维 bug");
        }
        assert!(pass, "Q4_0 QuantGemm large-n oracle: j 维错位");
    }

    // ── Q4_0 QuantGather multi-block x86 oracle (architect round 20) ──────
    //
    // 两个 1-block oracle 都过但 E2E 错 → architect round 20: 多 block 错序嫌疑 TOP.
    // 真实推理 hidden=1024=32 block. 1-block oracle 漏测 blk_ctr 跨 block 乘子.
    // 本测试 hidden=64 (2 block), 每 block 不同 nibble, 验证跨 block SPLIT 序.
    //
    // block 0 (offset 0): d=1.0, qs[0]=0x9A(lo=10,hi=9) → elem[0]=2.0, elem[16]=1.0
    // block 1 (offset 18): d=1.0, qs[0]=0xB6(lo=6,hi=11) → elem[32]=-2.0, elem[48]=3.0
    // 其余 elem=0 (qs=0x88 zero point)
    #[test]
    fn test_q4_0_quant_gather_x86_oracle_multiblock() {
        let hidden_dim: usize = 64; // 2 blocks (block_size=32)
        let vocab: usize = 2;
        let block_bytes: usize = 18;

        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let d_lo = (d_bits & 0xFF) as u8;
        let d_hi = ((d_bits >> 8) & 0xFF) as u8;

        let mut weight = vec![0u8; vocab * block_bytes * 2 / 2]; // vocab 行, 每行 2 block
        // 实际行 stride = (hidden/block_size)*block_bytes = 2*18 = 36 字节/token
        let row_stride = (hidden_dim / 32) * block_bytes; // 36
        let mut weight = vec![0u8; vocab * row_stride];
        // token 0: block 0 at offset 0, block 1 at offset 18
        weight[0] = d_lo; weight[1] = d_hi;
        weight[2] = 0x9A; // block0 qs[0]: lo=10, hi=9
        for j in 1..16 { weight[2 + j] = 0x88; }
        weight[18] = d_lo; weight[19] = d_hi;
        weight[20] = 0xB6; // block1 qs[0]: lo=6, hi=11
        for j in 1..16 { weight[20 + j] = 0x88; }
        // token 1 at offset 36: all zero point
        weight[36] = d_lo; weight[37] = d_hi;
        for j in 0..16 { weight[38 + j] = 0x88; }
        weight[54] = d_lo; weight[55] = d_hi;
        for j in 0..16 { weight[56 + j] = 0x88; }

        let indices: [u32; 1] = [0u32];
        let mut output = vec![0u8; 1 * hidden_dim * std::mem::size_of::<f32>()];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gather_inline(
            &mut prog,
            BoundExpr::Const(1),
            vocab,
            hidden_dim,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            None,
        ).expect("Q4_0 QuantGather multi-block emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-MB] JIT code {} bytes, {} instrs", code.len(), prog.instrs.len());

        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GatherFn = unsafe extern "C" fn(
            *const u8, *const u8, usize, usize, usize, usize, usize, *mut u8,
        ) -> usize;
        let f: GatherFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe {
            f(indices.as_ptr() as *const u8, weight.as_ptr() as *const u8,
              0, 0, 0, 0, 0, output.as_mut_ptr() as *mut u8)
        };

        let out_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output.as_ptr() as *const f32, hidden_dim)
        };
        eprintln!("[Q4_0-MB] out[0..8]   = {:?}", &out_f32[0..8]);
        eprintln!("[Q4_0-MB] out[16..24] = {:?}", &out_f32[16..24]);
        eprintln!("[Q4_0-MB] out[32..40] = {:?}", &out_f32[32..40]);
        eprintln!("[Q4_0-MB] out[48..56] = {:?}", &out_f32[48..56]);
        eprintln!("[Q4_0-MB] want: out[0]=2.0, out[16]=1.0, out[32]=-2.0, out[48]=3.0, 其余=0.0");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        let mut pass = true;
        let checks = [(0usize, 2.0f32), (16, 1.0), (32, -2.0), (48, 3.0)];
        for &(idx, want) in &checks {
            if (out_f32[idx] - want).abs() >= 1e-3 {
                eprintln!("[Q4_0-MB] FAIL elem[{}]: got {} want {}", idx, out_f32[idx], want);
                pass = false;
            }
        }
        for i in 0..hidden_dim {
            if ![0, 16, 32, 48].contains(&i) && out_f32[i].abs() >= 1e-3 {
                eprintln!("[Q4_0-MB] FAIL elem[{}]: got {} want 0.0", i, out_f32[i]);
                pass = false;
            }
        }
        if pass {
            eprintln!("[Q4_0-MB] PASS — 多 block SPLIT 正确 (跨 block blk_ctr 乘子对)");
        } else {
            eprintln!("[Q4_0-MB] FAIL — 多 block 错序 (architect round 20 top 嫌疑坐实)");
        }
        assert!(pass, "Q4_0 QuantGather multi-block oracle: 跨 block SPLIT 错序 (见 eprintln)");
    }

    // ── Q4_0 QuantGemm multi-block x86 oracle (architect round 20 step 2) ─
    //
    // QuantGather 多 block oracle 过 → 扩 QuantGemm 到多 block 验证层 GEMM 累加.
    // 真实推理 q_proj K=1024=32 block. 1-block oracle 漏测多 block 累加 + weight_row stride.
    //
    // m=1, n=1, k=64 (2 block). weight 2 block, act[0/16/32/48]=1.
    // block0: elem[0]=2.0, elem[16]=1.0; block1: elem[32]=-2.0, elem[48]=3.0
    // output = 1*2 + 1*1 + 1*(-2) + 1*3 = 4.0
    #[test]
    fn test_q4_0_quant_gemm_x86_oracle_multiblock() {
        use crate::compiler::codegen::vm::moe_quant_emit::emit_quant_gemm_inline;

        let m: usize = 1;
        let n: usize = 1;
        let k: usize = 64; // 2 blocks
        let block_bytes: usize = 18;

        // weight: 2 blocks (1 row, k=64)
        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let mut weight = vec![0u8; m * (k / 32) * block_bytes];
        // block 0 at offset 0
        weight[0] = (d_bits & 0xFF) as u8;
        weight[1] = ((d_bits >> 8) & 0xFF) as u8;
        weight[2] = 0x9A; // lo=10, hi=9
        for j in 1..16 { weight[2 + j] = 0x88; }
        // block 1 at offset 18
        weight[18] = (d_bits & 0xFF) as u8;
        weight[19] = ((d_bits >> 8) & 0xFF) as u8;
        weight[20] = 0xB6; // lo=6, hi=11
        for j in 1..16 { weight[20 + j] = 0x88; }

        // activation [k, n] = [64, 1] col-major: act[0/16/32/48]=1
        let mut act = vec![0.0f32; k * n];
        act[0] = 1.0;
        act[16] = 1.0;
        act[32] = 1.0;
        act[48] = 1.0;

        let mut output = vec![0.0f32; m * n];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog,
            BoundExpr::Const(m),
            n, k,
            QuantType::Q4_0,
            SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32,
            DotProductCap::SimdAssisted,
        ).expect("Q4_0 QuantGemm multi-block emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q4_0-GEMM-MB] JIT code {} bytes, {} instrs", code.len(), prog.instrs.len());

        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GemmFn = unsafe extern "C" fn(
            *const u8, *const u8, usize, usize, usize, usize, usize, *mut u8,
        ) -> usize;
        let f: GemmFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe {
            f(act.as_ptr() as *const u8, weight.as_ptr() as *const u8,
              0, 0, 0, 0, 0, output.as_mut_ptr() as *mut u8)
        };

        let out_f32: &[f32] = &output;
        eprintln!("[Q4_0-GEMM-MB] out = {:?}", out_f32);
        eprintln!("[Q4_0-GEMM-MB] want: out[0] = 4.0 (1*2 + 1*1 + 1*(-2) + 1*3, 多 block SPLIT 累加)");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        let pass = (out_f32[0] - 4.0).abs() < 1e-3;
        if pass {
            eprintln!("[Q4_0-GEMM-MB] PASS — QuantGemm 多 block SPLIT 累加正确");
        } else {
            eprintln!("[Q4_0-GEMM-MB] FAIL — QuantGemm 多 block 错序, actual={} want 4.0", out_f32[0]);
        }
        assert!(pass, "Q4_0 QuantGemm multi-block oracle: got {} want 4.0", out_f32[0]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Q6_K 真实执行 oracle (BCE-20260710-Q6K-HIGHBITS)
    // bug: emit_unpack NibbleWithHighBits 的 qh<<6 & 0x30 丢高 2 bit, Q6_K quarter 结构错位.
    // 构造已知 Q6_K 块 (d=1.0, sc=1, 全 quarter 有非零高 bit), 手算 dot vs JIT 真实执行.
    // ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_q6_k_quant_gemm_x86_oracle() {
        use crate::compiler::codegen::vm::moe_quant_emit::emit_quant_gemm_inline;

        let m: usize = 1;
        let n: usize = 2; // 2 vocab rows
        let k: usize = 256; // 1 Q6_K block/row (block_size=256)
        let block_bytes: usize = 210;

        // Q6_K block: qs[128] + qh[64] + scales[16] + d(f16) = 210B
        // 构造: d=f16(1.0), scales[0..16]=1 (i8), 每 quarter 设 q1=1,q2=2,q3=3,q4=4 (全高 bit 参与).
        // 6bit_val=33→lo4=1,hi2=2 (1|32=33), value=1; 34→lo4=2,hi2=2,value=2; 同理 3,4.
        // 对 n_group=0, l=0: q1(位0)=1,q2(位0)=2,q3(位0)=3,q4(位0)=4 → 输出位 [0,32,64,96]=[1,2,3,4]
        // 其余元素 0 (6bit=32: lo4=0,hi2=2 → value=0).
        // 手算: act=[1,1,1,1,0..0]. dot(row0)=1+2+3+4=10. dot(row1)=0.
        let d_f16 = f16::from_f32(1.0);
        let d_bits = d_f16.to_bits();
        let d_lo = (d_bits & 0xFF) as u8;
        let d_hi = ((d_bits >> 8) & 0xFF) as u8;

        let mut weight = vec![0u8; n * block_bytes];
        // row 0: q1=1,q2=2,q3=3,q4=4 at l=0 (n_group=0)
        // qs[0] = lo4(q1) | (lo4(q3)<<4) = 1 | (3<<4) = 0x31
        // qs[32] = lo4(q2) | (lo4(q4)<<4) = 2 | (4<<4) = 0x42
        // qh[0]: bit0-1=hi2(q1)=2, bit2-3=hi2(q2)=2, bit4-5=hi2(q3)=2, bit6-7=hi2(q4)=2 = 0xAA
        {
            let r0 = &mut weight[0..block_bytes];
            r0[0] = 0x31;   // qs[0]
            r0[32] = 0x42;  // qs[32]
            for j in 0..64 { r0[128 + j] = 0xAA; } // qh all 0xAA (hi2=2)
            for j in 0..16 { r0[192 + j] = 1; }     // scales=1
            r0[208] = d_lo; r0[209] = d_hi;
        }
        // row 1: 全 0 (lo4=0, hi2=2 → value=0)
        {
            let r1 = &mut weight[block_bytes..2*block_bytes];
            for j in 0..128 { r1[j] = 0x00; }
            for j in 0..64 { r1[128 + j] = 0xAA; }
            for j in 0..16 { r1[192 + j] = 1; }
            r1[208] = d_lo; r1[209] = d_hi;
        }

        let mut act = vec![0.0f32; m * k];
        // q1=1 at pos 0, q2=2 at pos 32, q3=3 at pos 64, q4=4 at pos 96
        act[0] = 1.0; act[32] = 1.0; act[64] = 1.0; act[96] = 1.0;

        let mut output = vec![0.0f32; m * n];

        let mut prog = VmProgram::new();
        let input_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let weight_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let output_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        prog.emit(VmInstr::LoadPtr { dst: input_ptr, src: PtrExpr::AbiArg(0) });
        prog.emit(VmInstr::LoadPtr { dst: weight_ptr, src: PtrExpr::AbiArg(1) });
        prog.emit(VmInstr::LoadPtr { dst: output_ptr, src: PtrExpr::StackArg(24) });

        emit_quant_gemm_inline(
            &mut prog, BoundExpr::Const(m), n, k,
            QuantType::Q6K, SimdWidth::W256,
            input_ptr, weight_ptr, output_ptr,
            QuantPrecision::F32, DotProductCap::SimdAssisted,
        ).expect("emit should succeed");

        let code = compile_vm_prog(&prog);
        eprintln!("[Q6_K-ORACLE] JIT {} bytes, {} instrs", code.len(), prog.instrs.len());

        let (exec_ptr, exec_len) = make_exec_buffer(&code);
        type GemmFn = unsafe extern "C" fn(*const u8, *const u8, usize, usize, usize, usize, usize, *mut u8) -> usize;
        let f: GemmFn = unsafe { std::mem::transmute(exec_ptr) };
        let _ret = unsafe { f(act.as_ptr() as *const u8, weight.as_ptr() as *const u8, 0,0,0,0,0, output.as_mut_ptr() as *mut u8) };

        let out_f32: &[f32] = &output;
        // dot(row0) = act[0]*1 + act[32]*2 + act[64]*3 + act[96]*4 = 1+2+3+4 = 10
        eprintln!("[Q6_K-ORACLE] out = {:?} (want [10.0, 0.0])", out_f32);
        eprintln!("[Q6_K-ORACLE] want: out[0]=1*1+1*2+1*3+1*4=10, out[1]=0");

        unsafe { libc::munmap(exec_ptr as *mut _, exec_len); }

        let pass0 = (out_f32[0] - 10.0).abs() < 1e-2;
        let pass1 = out_f32[1].abs() < 1e-2;
        if pass0 && pass1 {
            eprintln!("[Q6_K-ORACLE] PASS — Q6_K quarter 高 2 bit 解码正确");
        } else {
            eprintln!("[Q6_K-ORACLE] FAIL — Q6_K quarter 高 bit 提取错, got [{}, {}] want [10, 0]", out_f32[0], out_f32[1]);
        }
        assert!(pass0 && pass1, "Q6_K oracle: got [{}, {}] want [10, 0]", out_f32[0], out_f32[1]);
    }
}
