impl X86Lower {
    /// E2M1 decode (same math as emit_mxfp4_dequant, no LUT needed):
    ///   nibble bits: [sign(1)][exp(2)][mant(1)]
    ///   exp_field = (nibble >> 1) & 3
    ///   mant_field = nibble & 1
    ///   magnitude = (1 + mant×0.5) × 2^(exp-1)
    ///   nibble == 0 → 0.0
    ///
    /// Register allocation:
    ///   scratch_ymm(0) = s0 (nibble temp, then decoded a)
    ///   scratch_ymm(1) = s1 (exp bits temp, then decoded b)
    ///   scratch_ymm(2) = s2 (zero mask / sign temp)
    fn emit_fp4dot_ymm(
        &mut self,
        acc: VRegId,
        a: VRegId,
        b: VRegId,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let (acc_ymm, acc_spilled) = self.resolve_ymm_or_spill_write(acc, alloc, 0)?;
        let (a_ymm, _) = self.resolve_ymm_or_spill(a, alloc, 1)?;
        let (b_ymm, _) = self.resolve_ymm_or_spill(b, alloc, 2)?;

        let s0 = self.scratch_ymm(0); // decoded a
        let s1 = self.scratch_ymm(1); // decoded b
        let s2 = self.scratch_ymm(2); // temp

        // ── Decode a: e2m1 nibble → f32 ──
        // Copy nibbles to s0
        self.asm.vmovdqa(s0, a_ymm).map_err(Self::err)?;
        // Zero mask: nibble == 0 → 0.0
        self.asm.vpxor(s2, s2, s2).map_err(Self::err)?;
        self.asm.vpcmpeqd(s1, s0, s2).map_err(Self::err)?;
        // s1 = zero_mask_a (all-ones where nibble==0)
        let zero_mask_a = self.spill_scratch_ymm(1)?;
        self.asm.vmovdqa(zero_mask_a, s1).map_err(Self::err)?;

        // Extract exp: (nibble >> 1) & 3 — use shift trick: <<30 >>30
        self.asm.vpsrld(s1, s0, 1).map_err(Self::err)?;
        self.asm.vpslld(s1, s1, 30).map_err(Self::err)?;
        self.asm.vpsrld(s1, s1, 30).map_err(Self::err)?;
        // Build 2^(exp-2) via IEEE 754: (exp << 23) + 0x3E800000
        self.asm.vpslld(s1, s1, 23).map_err(Self::err)?;
        let base_025 = self.const_f32(0.25f32);
        self.asm.vbroadcastss(s2, dword_ptr(base_025)).map_err(Self::err)?;
        self.asm.vpaddd(s1, s1, s2).map_err(Self::err)?;
        // s1 = 2^(exp-2) as f32 bits
        // Extract sign → s2
        self.asm.vpsrld(s2, s0, 3).map_err(Self::err)?;
        self.asm.vpslld(s2, s2, 31).map_err(Self::err)?;
        // s2 = sign bit in MSB position
        // Extract mantissa
        self.asm.vpslld(s0, s0, 31).map_err(Self::err)?;
        self.asm.vpsrld(s0, s0, 31).map_err(Self::err)?;
        self.asm.vcvtdq2ps(s0, s0).map_err(Self::err)?;
        // FMA: (mant+1) × 2^(exp-2)
        self.asm.vfmadd213ps(s0, s1, s1).map_err(Self::err)?;
        // +2^(exp-2): gives (mant+2) × 2^(exp-2)
        self.asm.vaddps(s0, s0, s1).map_err(Self::err)?;
        // Apply sign
        self.asm.vxorps(s0, s0, s2).map_err(Self::err)?;
        // Apply zero mask: nibble==0 → 0.0
        self.asm.vandnps(s0, zero_mask_a, s0).map_err(Self::err)?;
        // s0 = decoded a (f32)

        // ── Decode b: e2m1 nibble → f32 (same sequence) ──
        self.asm.vmovdqa(s1, b_ymm).map_err(Self::err)?;
        // Zero mask
        self.asm.vpxor(s2, s2, s2).map_err(Self::err)?;
        self.asm.vpcmpeqd(s2, s1, s2).map_err(Self::err)?;
        let zero_mask_b = self.spill_scratch_ymm(2)?;
        self.asm.vmovdqa(zero_mask_b, s2).map_err(Self::err)?;
        // Need a temp for exp — use the now-free zero_mask_a register
        let exp_tmp = zero_mask_a;
        self.asm.vpsrld(exp_tmp, s1, 1).map_err(Self::err)?;
        self.asm.vpslld(exp_tmp, exp_tmp, 30).map_err(Self::err)?;
        self.asm.vpsrld(exp_tmp, exp_tmp, 30).map_err(Self::err)?;
        self.asm.vpslld(exp_tmp, exp_tmp, 23).map_err(Self::err)?;
        let base_025 = self.const_f32(0.25f32);
        self.asm.vbroadcastss(s2, dword_ptr(base_025)).map_err(Self::err)?;
        self.asm.vpaddd(exp_tmp, exp_tmp, s2).map_err(Self::err)?;
        // Sign
        self.asm.vpsrld(s2, s1, 3).map_err(Self::err)?;
        self.asm.vpslld(s2, s2, 31).map_err(Self::err)?;
        // Mantissa
        self.asm.vpslld(s1, s1, 31).map_err(Self::err)?;
        self.asm.vpsrld(s1, s1, 31).map_err(Self::err)?;
        self.asm.vcvtdq2ps(s1, s1).map_err(Self::err)?;
        // FMA: (mant+1) × 2^(exp-2)
        self.asm.vfmadd213ps(s1, exp_tmp, exp_tmp).map_err(Self::err)?;
        // +2^(exp-2)
        self.asm.vaddps(s1, s1, exp_tmp).map_err(Self::err)?;
        // Apply sign
        self.asm.vxorps(s1, s1, s2).map_err(Self::err)?;
        // Apply zero mask
        self.asm.vandnps(s1, zero_mask_b, s1).map_err(Self::err)?;
        // s1 = decoded b (f32)

        // ── Accumulate: acc += decoded_a × decoded_b ──
        self.asm.vfmadd231ps(acc_ymm, s0, s1).map_err(Self::err)?;
        if acc_spilled { self.spill_store_ymm(acc, alloc, 0)?; }
        Ok(())
    }

    /// DotProduct { input_dtype: Fp4, .. } AVX-512 (ZMM) — software e2m1 decode + FMA accumulate.
    ///
    /// Same math as emit_fp4dot_ymm but uses 512-bit zmm registers.
    /// AVX-512 comparisons write to opmask (k) registers instead of zmm directly,
    /// so we use knotd + vmovdqa32{k} merge for the zero-masking step.
    fn emit_fp4dot_zmm(
        &mut self,
        acc: VRegId,
        a: VRegId,
        b: VRegId,
        _width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let (acc_zmm, acc_spilled) = self.resolve_zmm_or_spill_write(acc, alloc, 0)?;
        let (a_zmm, _) = self.resolve_zmm_or_spill(a, alloc, 1)?;
        let (b_zmm, _) = self.resolve_zmm_or_spill(b, alloc, 2)?;

        let s0 = self.scratch_zmm(0); // decoded a
        let s1 = self.scratch_zmm(1); // decoded b
        let s2 = self.scratch_zmm(2); // temp / zero register

        // ── Decode a: e2m1 nibble → f32 ──
        self.asm.vmovdqa64(s0, a_zmm).map_err(Self::err)?;
        // Zero mask: nibble == 0 → 0.0
        self.asm.vpxord(s2, s2, s2).map_err(Self::err)?;
        // vpcmpeqd k1, s0, s2 → k1=1 where nibble==0
        let k_zero = iced_x86::code_asm::registers::k1;
        self.asm.vpcmpeqd(k_zero, s0, s2).map_err(Self::err)?;
        // Extract exp: (nibble >> 1) & 3 — shift trick: <<30 >>30
        self.asm.vpsrld(s1, s0, 1).map_err(Self::err)?;
        self.asm.vpslld(s1, s1, 30).map_err(Self::err)?;
        self.asm.vpsrld(s1, s1, 30).map_err(Self::err)?;
        // Build 2^(exp-2) via IEEE 754: (exp << 23) + 0x3E800000
        self.asm.vpslld(s1, s1, 23).map_err(Self::err)?;
        let base_025 = self.const_f32(0.25f32);
        self.asm.vbroadcastss(s2, dword_ptr(base_025)).map_err(Self::err)?;
        self.asm.vpaddd(s1, s1, s2).map_err(Self::err)?;
        // Sign
        self.asm.vpsrld(s2, s0, 3).map_err(Self::err)?;
        self.asm.vpslld(s2, s2, 31).map_err(Self::err)?;
        // Mantissa
        self.asm.vpslld(s0, s0, 31).map_err(Self::err)?;
        self.asm.vpsrld(s0, s0, 31).map_err(Self::err)?;
        self.asm.vcvtdq2ps(s0, s0).map_err(Self::err)?;
        // FMA: (mant+1) × 2^(exp-2)
        self.asm.vfmadd213ps(s0, s1, s1).map_err(Self::err)?;
        // +2^(exp-2)
        self.asm.vaddps(s0, s0, s1).map_err(Self::err)?;
        // Apply sign
        self.asm.vxorps(s0, s0, s2).map_err(Self::err)?;
        // Apply zero mask: where nibble==0 (k_zero=1), zero the result.
        // knotd k3, k_zero → k3=1 where nibble!=0
        // Save decoded to s1, zero s0, merge back: s0{k3} = s1
        self.asm.vmovdqa32(s1, s0).map_err(Self::err)?;
        self.asm.vpxord(s0, s0, s0).map_err(Self::err)?;
        let k_not_zero = iced_x86::code_asm::registers::k3;
        self.asm.knotd(k_not_zero, k_zero).map_err(Self::err)?;
        // vmovdqa32 s0{k3}, s1: where k3=1 (nibble!=0), s0[i]=s1[i]; where k3=0, s0[i] stays 0
        self.asm.vmovdqa32(s0.k3(), s1).map_err(Self::err)?;

        // ── Decode b: e2m1 nibble → f32 (same sequence) ──
        self.asm.vmovdqa64(s1, b_zmm).map_err(Self::err)?;
        // Zero mask: k1 for b
        self.asm.vpxord(s2, s2, s2).map_err(Self::err)?;
        let k_zero_b = iced_x86::code_asm::registers::k2;
        self.asm.vpcmpeqd(k_zero_b, s1, s2).map_err(Self::err)?;
        // Exp — reuse s2 as temp
        self.asm.vpsrld(s2, s1, 1).map_err(Self::err)?;
        self.asm.vpslld(s2, s2, 30).map_err(Self::err)?;
        self.asm.vpsrld(s2, s2, 30).map_err(Self::err)?;
        self.asm.vpslld(s2, s2, 23).map_err(Self::err)?;
        let base_025 = self.const_f32(0.25f32);
        // Need a broadcast scratch — use spill_scratch_zmm(1) (not used by decoded_a which is in s0)
        let bcast = self.spill_scratch_zmm(1)?;
        self.asm.vbroadcastss(bcast, dword_ptr(base_025)).map_err(Self::err)?;
        self.asm.vpaddd(s2, s2, bcast).map_err(Self::err)?;
        // Sign (reuse bcast as sign temp)
        self.asm.vpsrld(bcast, s1, 3).map_err(Self::err)?;
        self.asm.vpslld(bcast, bcast, 31).map_err(Self::err)?;
        // Mantissa
        self.asm.vpslld(s1, s1, 31).map_err(Self::err)?;
        self.asm.vpsrld(s1, s1, 31).map_err(Self::err)?;
        self.asm.vcvtdq2ps(s1, s1).map_err(Self::err)?;
        // FMA: (mant+1) × 2^(exp-2)
        self.asm.vfmadd213ps(s1, s2, s2).map_err(Self::err)?;
        // +2^(exp-2)
        self.asm.vaddps(s1, s1, s2).map_err(Self::err)?;
        // Apply sign (bcast holds sign bits)
        self.asm.vxorps(s1, s1, bcast).map_err(Self::err)?;
        // Apply zero mask: save decoded_b, zero, merge
        let b_save = self.spill_scratch_zmm(2)?;
        self.asm.vmovdqa32(b_save, s1).map_err(Self::err)?;
        self.asm.vpxord(s1, s1, s1).map_err(Self::err)?;
        let k_not_zero_b = iced_x86::code_asm::registers::k3;
        self.asm.knotd(k_not_zero_b, k_zero_b).map_err(Self::err)?;
        self.asm.vmovdqa32(s1.k3(), b_save).map_err(Self::err)?;

        // ── Accumulate: acc += decoded_a × decoded_b ──
        self.asm.vfmadd231ps(acc_zmm, s0, s1).map_err(Self::err)?;
        if acc_spilled { self.spill_store_zmm(acc, alloc, 0)?; }
        Ok(())
    }

    /// Helper: create a XMM register with a constant i32 value (broadcasted to all lanes).
    fn xmm_const_i32(&mut self, val: i32) -> Result<AsmRegisterXmm, CompilerError> {
        let label = self.const_f32(f32::from_bits(val as u32));
        let xmm = self.scratch_xmm(0); // temporary
        self.asm.vbroadcastss(xmm, dword_ptr(label)).map_err(Self::err)?;
        Ok(xmm)
    }

    /// Helper: create a YMM register with a constant i32 value (broadcasted to all lanes).
    fn ymm_const_i32(&mut self, val: i32) -> Result<AsmRegisterYmm, CompilerError> {
        let label = self.const_f32(f32::from_bits(val as u32));
        let ymm = self.scratch_ymm(0); // temporary
        self.asm.vbroadcastss(ymm, dword_ptr(label)).map_err(Self::err)?;
        Ok(ymm)
    }

    /// F32→BF16 窄化 (AVX2 软件路径, CR-TIER-SOVEREIGNTY-004)。
    ///
    /// 无 vcvtneps2bf16 的硬件 (i9-10900KF / CometLake 等 AVX2-only CPU) 上，
    /// 用向量化整数序列完成 round-to-nearest-even (RNE) 截断：
    ///   rounded = f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)
    ///   bf16    = rounded >> 16
    ///
    /// `+0x7FFF` 实现四舍五入到最近；`+lsb` 处理 RNE 平局 (tie-to-even)。
    /// NaN/Inf 保持语义 (exp 全 1 不受低位影响)。
    ///
    /// 输入: ymm(8×F32)。输出: xmm(8×BF16, 即 16 字节)。
    /// scratch 约定: s0=输入副本, s1=bias, s2=lsb-mask 结果。
    // @trace REQ-HW-TIER-004 [req:BF16-AVX2-SoftwareNarrow] AVX2 (无原生 BF16) F32→BF16 向量化 RNE 窄化
    fn emit_f32_to_bf16_ymm_to_xmm_avx2(
        &mut self,
        dst_xmm: AsmRegisterXmm,
        src_ymm: AsmRegisterYmm,
    ) -> Result<(), CompilerError> {
        let s0 = self.scratch_ymm(0); // 输入副本 / 累加
        let s1 = self.scratch_ymm(1); // bias 常量
        let s2 = self.scratch_ymm(2); // lsb 提取

        // s0 = src (复制，避免破坏源)
        self.asm.vmovups(s0, src_ymm).map_err(Self::err)?;

        // s2 = (s0 >> 16) & 1  —— BF16 lsb for RNE tie-break
        self.asm.vpsrld(s2, s0, 16).map_err(Self::err)?;
        // s1 = 广播 0x00000001 (作为 f32 bit pattern)
        let one = f32::from_bits(1u32);
        let one_label = self.const_f32(one);
        self.asm.vbroadcastss(s1, dword_ptr(one_label)).map_err(Self::err)?;
        self.asm.vpand(s2, s2, s1).map_err(Self::err)?;

        // s0 = s0 + 0x7FFF (round-half-up bias) + lsb
        let bias = f32::from_bits(0x7FFFu32);
        let bias_label = self.const_f32(bias);
        self.asm.vbroadcastss(s1, dword_ptr(bias_label)).map_err(Self::err)?;
        self.asm.vpaddd(s0, s0, s1).map_err(Self::err)?;
        self.asm.vpaddd(s0, s0, s2).map_err(Self::err)?;

        // s0 = s0 >> 16  —— 现在每个 u32 lane 的高 16 位是 BF16，低 16 位是 0
        self.asm.vpsrld(s0, s0, 16).map_err(Self::err)?;

        // dst_xmm = pack 8×u32(低16位有效) → 8×u16
        // vpackusdw: 将两个 ymm/xmm 的 32-bit lanes 饱和打包成 16-bit。
        // 这里所有 8 个 lane 都在 s0 的低半 (xmm 视图)，用同一源做 in-place pack。
        // vpackusdw xmm, xmm, xmm → 8×u16 = BF16，写入 dst_xmm。
        let s0_xmm = Self::ymm_to_xmm(s0);
        self.asm.vpackusdw(dst_xmm, s0_xmm, s0_xmm).map_err(Self::err)?;
        Ok(())
    }

    /// F32→BF16 窄化 (AVX2 软件路径, Scalar 变体)。
    /// 输入: xmm (lane 0 = 1×F32)。输出: xmm (lane 0 = 1×BF16, 其余 lane 无效)。
    fn emit_f32_to_bf16_xmm_avx2(
        &mut self,
        dst_xmm: AsmRegisterXmm,
        src_xmm: AsmRegisterXmm,
    ) -> Result<(), CompilerError> {
        // 复用 ymm 变体: 把 src_xmm broadcast/复制到 ymm，窄化，再取低 xmm。
        // 但更高效: 直接在 xmm 上做标量级序列 (4 lanes，只用 lane 0)。
        let s0 = self.scratch_xmm(0); // 输入副本
        let s1 = self.scratch_xmm(1); // bias/lsb 常量
        let s2 = self.scratch_xmm(2); // lsb

        self.asm.vmovups(s0, src_xmm).map_err(Self::err)?;

        // s2 = (s0 >> 16) & 1
        self.asm.vpsrld(s2, s0, 16).map_err(Self::err)?;
        let one = f32::from_bits(1u32);
        let one_label = self.const_f32(one);
        self.asm.vbroadcastss(s1, dword_ptr(one_label)).map_err(Self::err)?;
        self.asm.vpand(s2, s2, s1).map_err(Self::err)?;

        // s0 += 0x7FFF + lsb
        let bias = f32::from_bits(0x7FFFu32);
        let bias_label = self.const_f32(bias);
        self.asm.vbroadcastss(s1, dword_ptr(bias_label)).map_err(Self::err)?;
        self.asm.vpaddd(s0, s0, s1).map_err(Self::err)?;
        self.asm.vpaddd(s0, s0, s2).map_err(Self::err)?;

        // s0 >>= 16
        self.asm.vpsrld(s0, s0, 16).map_err(Self::err)?;

        // pack: dst_xmm 低 16-bit = lane 0 的 BF16
        self.asm.vpackusdw(dst_xmm, s0, s0).map_err(Self::err)?;
        Ok(())
    }

    /// Cephes degree-5 exp(x) 多项式。
    fn emit_exp_cephes(&mut self, dst: AsmRegisterYmm, src: AsmRegisterYmm) -> Result<(), CompilerError> {
        use crate::compiler::codegen::math_approx::*;
        // Cephes scratch: ymm13/ymm14/ymm15 (分配器保留: rax 排除, 高编号 ymm 不分配给短活跃 VReg)
        let s0 = ymm14;
        let s1 = ymm15;

        // Clamp
        let lo = self.const_f32(EXP_CLAMP_LO);
        self.asm.vbroadcastss(s0, dword_ptr(lo)).map_err(Self::err)?;
        self.asm.vmaxps(dst, src, s0).map_err(Self::err)?;
        let hi = self.const_f32(EXP_CLAMP_HI);
        self.asm.vbroadcastss(s0, dword_ptr(hi)).map_err(Self::err)?;
        self.asm.vminps(dst, dst, s0).map_err(Self::err)?;

        // k = round(x * log2e)
        let log2e = self.const_f32(EXP_LOG2E);
        self.asm.vbroadcastss(s0, dword_ptr(log2e)).map_err(Self::err)?;
        self.asm.vmulps(s0, dst, s0).map_err(Self::err)?;
        self.asm.vroundps(s1, s0, 0i32).map_err(Self::err)?; // s1 = k

        // Range reduction
        let c1 = self.const_f32(EXP_C1);
        self.asm.vbroadcastss(s0, dword_ptr(c1)).map_err(Self::err)?;
        self.asm.vfmadd231ps(dst, s1, s0).map_err(Self::err)?;
        let c2 = self.const_f32(EXP_C2);
        self.asm.vbroadcastss(s0, dword_ptr(c2)).map_err(Self::err)?;
        self.asm.vfmadd231ps(dst, s1, s0).map_err(Self::err)?;

        // Horner polynomial
        let p0 = self.const_f32(EXP_P0);
        self.asm.vbroadcastss(s0, dword_ptr(p0)).map_err(Self::err)?;
        for coeff in [EXP_P1, EXP_P2, EXP_P3, EXP_P4, EXP_P5] {
            let c = self.const_f32(coeff);
            let tmp = self.const_f32(coeff); // reuse label
            self.asm.vbroadcastss(ymm13, dword_ptr(c)).map_err(Self::err)?;
            self.asm.vfmadd213ps(s0, dst, ymm13).map_err(Self::err)?;
        }
        // result = poly*r^2 + r + 1
        self.asm.vmulps(s0, s0, dst).map_err(Self::err)?;
        self.asm.vmulps(s0, s0, dst).map_err(Self::err)?;
        self.asm.vaddps(s0, s0, dst).map_err(Self::err)?;
        let one = self.const_f32(1.0);
        self.asm.vbroadcastss(ymm13, dword_ptr(one)).map_err(Self::err)?;
        self.asm.vaddps(s0, s0, ymm13).map_err(Self::err)?;

        // 2^k
        self.asm.vcvtps2dq(ymm13, s1).map_err(Self::err)?;
        let m127 = self.const_f32(127.0);
        self.asm.vbroadcastss(dst, dword_ptr(m127)).map_err(Self::err)?;
        self.asm.vcvtps2dq(dst, dst).map_err(Self::err)?;
        self.asm.vpaddd(ymm13, ymm13, dst).map_err(Self::err)?;
        self.asm.vpslld(ymm13, ymm13, 23i32).map_err(Self::err)?;

        self.asm.vmulps(dst, s0, ymm13).map_err(Self::err)?;
        Ok(())
    }

    // ── FP8 → F32 conversion (software AVX2 path) ──

    /// FP8 E4M3 → F32 vector conversion (AVX2 software path).
    ///
    /// E4M3 layout: [sign(1)][exp(4)][mant(3)] — bias = 7
    /// F32 layout:  [sign(1)][exp(8)][mant(23)] — bias = 127
    ///
    /// Strategy: spill exp/mant early, build "as-if-normal" bits, then fix
    /// edge cases (zero, subnormal, Inf, NaN) with masks and blends.
    ///
    /// Register budget: s0=sign, s1=exp, s2=mant, d=result, va=temp
    fn emit_fp8_e4m3_to_f32_ymm(
        &mut self,
        d: AsmRegisterYmm,
        va: AsmRegisterYmm,
    ) -> Result<(), CompilerError> {
        let s0 = self.scratch_ymm(0);
        let s1 = self.scratch_ymm(1);
        let s2 = self.scratch_ymm(2);

        // ── Phase 1: Extract sign/exp/mant, spill exp and mant ──

        // sign = (x >> 7) << 31
        self.asm.vmovdqa(s0, va).map_err(Self::err)?;
        self.asm.vpsrld(s0, s0, 7).map_err(Self::err)?;
        self.asm.vpslld(s0, s0, 31).map_err(Self::err)?;

        // exp = (x >> 3) & 0xF
        self.asm.vmovdqa(s1, va).map_err(Self::err)?;
        self.asm.vpsrld(s1, s1, 3).map_err(Self::err)?;
        self.asm.vpslld(s2, s1, 28).map_err(Self::err)?;
        self.asm.vpsrld(s1, s2, 28).map_err(Self::err)?;

        // mant = x & 0x7
        self.asm.vmovdqa(s2, va).map_err(Self::err)?;
        self.asm.vpslld(s2, s2, 29).map_err(Self::err)?;
        self.asm.vpsrld(s2, s2, 29).map_err(Self::err)?;

        // Spill exp and mant for later mask computation
        let exp_spill = self.spill_scratch_ymm(0)?;
        self.asm.vmovdqa(exp_spill, s1).map_err(Self::err)?;
        let mant_spill = self.spill_scratch_ymm(1)?;
        self.asm.vmovdqa(mant_spill, s2).map_err(Self::err)?;

        // ── Phase 2: Build "as-if-normal" F32 bits ──
        // d = sign | ((exp + 120) << 23) | (mant << 20)
        let bias_120 = self.const_f32(f32::from_bits(120u32 << 23));
        self.asm.vbroadcastss(d, dword_ptr(bias_120)).map_err(Self::err)?;
        self.asm.vpaddd(d, s1, d).map_err(Self::err)?;
        self.asm.vpslld(d, d, 23).map_err(Self::err)?;
        self.asm.vpor(d, d, s0).map_err(Self::err)?;
        self.asm.vpslld(va, s2, 20).map_err(Self::err)?;
        self.asm.vpor(d, d, va).map_err(Self::err)?;
        // d correct for normal range (1 ≤ exp ≤ 14)

        // ── Phase 3: Fix zero (exp==0 & mant==0 → sign << 31) ──
        self.asm.vpxor(va, va, va).map_err(Self::err)?;
        self.asm.vpcmpeqd(va, exp_spill, va).map_err(Self::err)?;
        self.asm.vpcmpeqd(s1, mant_spill, self.scratch_ymm(1)).map_err(Self::err)?;
        // s1 = (mant==0). Need zero const for this comparison.
        self.asm.vpxor(s1, s1, s1).map_err(Self::err)?;
        self.asm.vpcmpeqd(s1, mant_spill, s1).map_err(Self::err)?;
        self.asm.vpand(va, va, s1).map_err(Self::err)?;
        // va = is_zero mask
        self.asm.vblendvps(d, d, s0, va).map_err(Self::err)?;
        // s0 still has sign<<31

        // ── Phase 4: Fix subnormal (exp==0 & mant!=0 → mant × 2^(-9)) ──
        // Reload mant for subnormal conversion
        self.asm.vmovdqa(s2, mant_spill).map_err(Self::err)?;
        self.asm.vcvtdq2ps(s2, s2).map_err(Self::err)?;
        let two_neg9 = self.const_f32(1.953125e-3); // 2^(-9)
        self.asm.vbroadcastss(s1, dword_ptr(two_neg9)).map_err(Self::err)?;
        self.asm.vmulps(s2, s2, s1).map_err(Self::err)?;
        // Apply sign via XOR
        self.asm.vxorps(s2, s2, s0).map_err(Self::err)?;
        // Build is_subnormal mask: exp==0 & mant!=0
        self.asm.vpxor(va, va, va).map_err(Self::err)?;
        self.asm.vpcmpeqd(va, exp_spill, va).map_err(Self::err)?;
        // va = (exp==0). Now AND with NOT(mant==0)
        self.asm.vpxor(s1, s1, s1).map_err(Self::err)?;
        self.asm.vpcmpeqd(s1, mant_spill, s1).map_err(Self::err)?;
        self.asm.vpandn(va, s1, va).map_err(Self::err)?;
        // va = is_subnormal mask
        self.asm.vblendvps(d, d, s2, va).map_err(Self::err)?;

        // ── Phase 5: Fix Inf/NaN (exp==15 → F32 Inf/NaN) ──
        // Normal bits: (15+120)<<23 = 135<<23. Need 255<<23.
        // Add (255-135)<<23 = 120<<23 to bump exp to 255.
        let delta_120 = self.const_f32(f32::from_bits(120u32 << 23));
        self.asm.vbroadcastss(va, dword_ptr(delta_120)).map_err(Self::err)?;
        self.asm.vpaddd(s2, d, va).map_err(Self::err)?;
        // s2 = Inf/NaN bits (exp=255). For mant==0 → Inf, mant!=0 → NaN.
        // Force quiet NaN: OR bit 22 (0x00400000)
        let quiet_nan = self.const_f32(f32::from_bits(0x00400000u32));
        self.asm.vbroadcastss(s1, dword_ptr(quiet_nan)).map_err(Self::err)?;
        self.asm.vpor(s2, s2, s1).map_err(Self::err)?;
        // Build is_special mask: exp==15
        let const_15 = self.const_f32(f32::from_bits(15u32));
        self.asm.vbroadcastss(va, dword_ptr(const_15)).map_err(Self::err)?;
        self.asm.vpcmpeqd(va, exp_spill, va).map_err(Self::err)?;
        // va = is_special mask
        self.asm.vblendvps(d, d, s2, va).map_err(Self::err)?;

        Ok(())
    }

    /// FP8 E5M2 → F32 vector conversion (AVX2 software path).
    ///
    /// E5M2 layout: [sign(1)][exp(5)][mant(2)] — bias = 15
    /// F32 layout:  [sign(1)][exp(8)][mant(23)] — bias = 127
    ///
    /// Normal (1 ≤ exp ≤ 30): F32 = sign | ((exp + 112) << 23) | (mant << 21)
    /// Zero  (exp=0, mant=0):  F32 = sign << 31
    /// Subnormal (exp=0, mant≠0): F32 = (-1)^sign × mant × 2^(-16)
    /// Inf   (exp=31, mant=0): F32 = sign | 0x7F800000
    /// NaN   (exp=31, mant≠0): F32 = sign | 0x7FC00000
    fn emit_fp8_e5m2_to_f32_ymm(
        &mut self,
        d: AsmRegisterYmm,
        va: AsmRegisterYmm,
    ) -> Result<(), CompilerError> {
        let s0 = self.scratch_ymm(0);
        let s1 = self.scratch_ymm(1);
        let s2 = self.scratch_ymm(2);

        // ── Phase 1: Extract sign/exp/mant, spill exp and mant ──

        // sign = (x >> 7) << 31
        self.asm.vmovdqa(s0, va).map_err(Self::err)?;
        self.asm.vpsrld(s0, s0, 7).map_err(Self::err)?;
        self.asm.vpslld(s0, s0, 31).map_err(Self::err)?;

        // exp = (x >> 2) & 0x1F  (5 bits)
        self.asm.vmovdqa(s1, va).map_err(Self::err)?;
        self.asm.vpsrld(s1, s1, 2).map_err(Self::err)?;
        self.asm.vpslld(s2, s1, 27).map_err(Self::err)?;
        self.asm.vpsrld(s1, s2, 27).map_err(Self::err)?;

        // mant = x & 0x3  (2 bits)
        self.asm.vmovdqa(s2, va).map_err(Self::err)?;
        self.asm.vpslld(s2, s2, 30).map_err(Self::err)?;
        self.asm.vpsrld(s2, s2, 30).map_err(Self::err)?;

        let exp_spill = self.spill_scratch_ymm(0)?;
        self.asm.vmovdqa(exp_spill, s1).map_err(Self::err)?;
        let mant_spill = self.spill_scratch_ymm(1)?;
        self.asm.vmovdqa(mant_spill, s2).map_err(Self::err)?;

        // ── Phase 2: Build "as-if-normal" F32 bits ──
        // d = sign | ((exp + 112) << 23) | (mant << 21)
        let bias_112 = self.const_f32(f32::from_bits(112u32 << 23));
        self.asm.vbroadcastss(d, dword_ptr(bias_112)).map_err(Self::err)?;
        self.asm.vpaddd(d, s1, d).map_err(Self::err)?;
        self.asm.vpslld(d, d, 23).map_err(Self::err)?;
        self.asm.vpor(d, d, s0).map_err(Self::err)?;
        self.asm.vpslld(va, s2, 21).map_err(Self::err)?;
        self.asm.vpor(d, d, va).map_err(Self::err)?;

        // ── Phase 3: Fix zero ──
        self.asm.vpxor(va, va, va).map_err(Self::err)?;
        self.asm.vpcmpeqd(va, exp_spill, va).map_err(Self::err)?;
        self.asm.vpxor(s1, s1, s1).map_err(Self::err)?;
        self.asm.vpcmpeqd(s1, mant_spill, s1).map_err(Self::err)?;
        self.asm.vpand(va, va, s1).map_err(Self::err)?;
        self.asm.vblendvps(d, d, s0, va).map_err(Self::err)?;

        // ── Phase 4: Fix subnormal (mant × 2^(-16)) ──
        self.asm.vmovdqa(s2, mant_spill).map_err(Self::err)?;
        self.asm.vcvtdq2ps(s2, s2).map_err(Self::err)?;
        let two_neg16 = self.const_f32(1.5258789e-5); // 2^(-16)
        self.asm.vbroadcastss(s1, dword_ptr(two_neg16)).map_err(Self::err)?;
        self.asm.vmulps(s2, s2, s1).map_err(Self::err)?;
        self.asm.vxorps(s2, s2, s0).map_err(Self::err)?;
        self.asm.vpxor(va, va, va).map_err(Self::err)?;
        self.asm.vpcmpeqd(va, exp_spill, va).map_err(Self::err)?;
        self.asm.vpxor(s1, s1, s1).map_err(Self::err)?;
        self.asm.vpcmpeqd(s1, mant_spill, s1).map_err(Self::err)?;
        self.asm.vpandn(va, s1, va).map_err(Self::err)?;
        self.asm.vblendvps(d, d, s2, va).map_err(Self::err)?;

        // ── Phase 5: Fix Inf/NaN (exp==31) ──
        // Normal: (31+112)<<23 = 143<<23. Need 255<<23. Delta = 112<<23.
        let delta_112 = self.const_f32(f32::from_bits(112u32 << 23));
        self.asm.vbroadcastss(va, dword_ptr(delta_112)).map_err(Self::err)?;
        self.asm.vpaddd(s2, d, va).map_err(Self::err)?;
        let quiet_nan = self.const_f32(f32::from_bits(0x00400000u32));
        self.asm.vbroadcastss(s1, dword_ptr(quiet_nan)).map_err(Self::err)?;
        self.asm.vpor(s2, s2, s1).map_err(Self::err)?;
        let const_31 = self.const_f32(f32::from_bits(31u32));
        self.asm.vbroadcastss(va, dword_ptr(const_31)).map_err(Self::err)?;
        self.asm.vpcmpeqd(va, exp_spill, va).map_err(Self::err)?;
        self.asm.vblendvps(d, d, s2, va).map_err(Self::err)?;

        Ok(())
    }

    // ── 常量池 + 最终汇编 ──

    /// Log(x) degree-4 minimax polynomial。
    fn emit_log_minimax(&mut self, dst: AsmRegisterYmm, src: AsmRegisterYmm) -> Result<(), CompilerError> {
        use crate::compiler::codegen::math_approx::*;
        // Extract exponent: e = float(x >> 23) - 127
        self.asm.vpsrld(ymm13, src, 23i32).map_err(Self::err)?;
        self.asm.vcvtdq2ps(ymm13, ymm13).map_err(Self::err)?;
        let m127 = self.const_f32(127.0);
        self.asm.vbroadcastss(ymm14, dword_ptr(m127)).map_err(Self::err)?;
        self.asm.vsubps(ymm13, ymm13, ymm14).map_err(Self::err)?;

        // Extract mantissa: m = (x & 0x007FFFFF) | 0x3F800000
        let mask = self.const_f32(LOG_MANTISSA_MASK);
        self.asm.vbroadcastss(ymm14, dword_ptr(mask)).map_err(Self::err)?;
        self.asm.vandps(ymm15, src, ymm14).map_err(Self::err)?;
        let one = self.const_f32(1.0);
        self.asm.vbroadcastss(ymm14, dword_ptr(one)).map_err(Self::err)?;
        self.asm.vorps(ymm15, ymm15, ymm14).map_err(Self::err)?;
        self.asm.vsubps(ymm15, ymm15, ymm14).map_err(Self::err)?; // t = m - 1

        // Horner: p = ((C4*t + C3)*t + C2)*t + C1
        let c4 = self.const_f32(LOG_C4);
        self.asm.vbroadcastss(dst, dword_ptr(c4)).map_err(Self::err)?;
        let c3 = self.const_f32(LOG_C3);
        self.asm.vbroadcastss(ymm14, dword_ptr(c3)).map_err(Self::err)?;
        self.asm.vfmadd213ps(dst, ymm15, ymm14).map_err(Self::err)?;
        let c2 = self.const_f32(LOG_C2);
        self.asm.vbroadcastss(ymm14, dword_ptr(c2)).map_err(Self::err)?;
        self.asm.vfmadd213ps(dst, ymm15, ymm14).map_err(Self::err)?;
        let c1 = self.const_f32(LOG_C1);
        self.asm.vbroadcastss(ymm14, dword_ptr(c1)).map_err(Self::err)?;
        self.asm.vfmadd213ps(dst, ymm15, ymm14).map_err(Self::err)?;
        self.asm.vmulps(dst, dst, ymm15).map_err(Self::err)?;

        // result = e * ln(2) + p
        let ln2 = self.const_f32(LOG_LN2);
        self.asm.vbroadcastss(ymm14, dword_ptr(ln2)).map_err(Self::err)?;
        self.asm.vfmadd231ps(dst, ymm13, ymm14).map_err(Self::err)?;
        Ok(())
    }
}