impl AArch64Lower {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            const_pool: Vec::new(),
            loop_stack: Vec::new(),
            platform: AArch64Features::default(),
            labels: std::collections::HashMap::new(),
            jit_ctx: crate::compiler::jit_context::JitContext::new(&IsaProfile::from_device_profile(
                &crate::dispatch::device_profile::DeviceProfile::detect(),
            )),
        }
    }

    /// 从 IsaProfile 构造 — 使 SVE2/SME2 路径可用。
    pub fn with_profile(profile: &IsaProfile) -> Self {
        let platform = match &profile.platform {
            Platform::AArch64 { has_sve2, has_sme2, sve_vl, .. } => AArch64Features {
                has_sve2: *has_sve2,
                has_sme2: *has_sme2,
                sve_vl: *sve_vl,
            },
            _ => AArch64Features::default(),
        };
        Self {
            code: Vec::new(),
            const_pool: Vec::new(),
            loop_stack: Vec::new(),
            platform,
            labels: std::collections::HashMap::new(),
            jit_ctx: crate::compiler::jit_context::JitContext::new(profile),
        }
    }

    fn emit32(&mut self, instr: u32) {
        self.code.extend_from_slice(&instr.to_le_bytes());
    }

    fn current_offset(&self) -> usize {
        self.code.len()
    }

    /// 回填指定偏移处的 32-bit 指令。
    fn patch32(&mut self, offset: usize, instr: u32) {
        let bytes = instr.to_le_bytes();
        self.code[offset..offset + 4].copy_from_slice(&bytes);
    }

    // ══════════════════════════════════════════════════════════════════
    //  AArch64 寄存器编号
    // ══════════════════════════════════════════════════════════════════

    /// GPR: x0-x30, sp=31
    fn gpr_num(phys: PhysGpr) -> u8 { phys.0 }

    /// NEON v0-v31 / SVE z0-z31 (共享编号空间)
    fn vreg_num(phys: PhysVec) -> u8 { phys.0 }

    fn resolve_gpr(&self, vreg: VRegId, alloc: &RegAllocation) -> Result<u8, CompilerError> {
        alloc.get_gpr(vreg)
            .map(Self::gpr_num)
            .ok_or_else(|| CompilerError::CodegenViolation(format!("v{} not allocated to GPR", vreg.0)))
    }

    fn resolve_vreg(&self, vreg: VRegId, alloc: &RegAllocation) -> Result<u8, CompilerError> {
        alloc.get_vec(vreg)
            .map(Self::vreg_num)
            .ok_or_else(|| CompilerError::CodegenViolation(format!("v{} not allocated to NEON/SVE", vreg.0)))
    }

    // ══════════════════════════════════════════════════════════════════
    //  NEON (128-bit) 指令编码
    // ══════════════════════════════════════════════════════════════════

    /// LDR Xd, [Xn, #imm] (64-bit load, imm scaled by 8)
    fn enc_ldr_x(&self, rd: u8, rn: u8, imm12: u16) -> u32 {
        0xF9400000 | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// STR Xd, [Xn, #imm] (64-bit store, imm scaled by 8)
    fn enc_str_x(&self, rd: u8, rn: u8, imm12: u16) -> u32 {
        0xF9000000 | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// MOV Xd, Xn (register move) — ORR Xd, XZR, Xn
    fn enc_mov_x(&self, rd: u8, rn: u8) -> u32 {
        0xAA0003E0 | ((rn as u32 & 0x1F) << 16) | (rd as u32 & 0x1F)
    }

    /// LD1 {Vt.4S}, [Xn] — NEON 128-bit vector load
    fn enc_ld1_4s(&self, vt: u8, xn: u8) -> u32 {
        0x4C407800 | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
    }

    /// ST1 {Vt.4S}, [Xn] — NEON 128-bit vector store
    fn enc_st1_4s(&self, vt: u8, xn: u8) -> u32 {
        0x4C007800 | ((xn as u32 & 0x1F) << 5) | (vt as u32 & 0x1F)
    }

    /// ORR Vd.16B, Vn.16B, Vm.16B — NEON register copy (Vd = Vn | Vm, used as mov)
    fn enc_orr_vv(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E201C00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FADD Vd.4S, Vn.4S, Vm.4S — NEON f32 add
    fn enc_fadd_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E20D400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FSUB Vd.4S, Vn.4S, Vm.4S
    fn enc_fsub_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4EA0D400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FMUL Vd.4S, Vn.4S, Vm.4S
    fn enc_fmul_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x6E20DC00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FDIV Vd.4S, Vn.4S, Vm.4S
    fn enc_fdiv_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x6E20FC00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FMLA Vd.4S, Vn.4S, Vm.4S — FMA: Vd += Vn * Vm
    fn enc_fmla_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E20CC00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FNEG Vd.4S, Vn.4S
    fn enc_fneg_4s(&self, vd: u8, vn: u8) -> u32 {
        0x6EA0F800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FSQRT Vd.4S, Vn.4S
    fn enc_fsqrt_4s(&self, vd: u8, vn: u8) -> u32 {
        0x6EA1F800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FRECPE Vd.4S, Vn.4S — fast reciprocal estimate
    fn enc_frecpe_4s(&self, vd: u8, vn: u8) -> u32 {
        0x4EA1D800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FRSQRTE Vd.4S, Vn.4S — fast reciprocal square root estimate
    fn enc_frsqrte_4s(&self, vd: u8, vn: u8) -> u32 {
        0x6EA1D800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FMAX Vd.4S, Vn.4S, Vm.4S
    fn enc_fmax_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E20F400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FMIN Vd.4S, Vn.4S, Vm.4S
    fn enc_fmin_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4EA0F400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FRINTN Vd.4S, Vn.4S (round to nearest)
    fn enc_frintn_4s(&self, vd: u8, vn: u8) -> u32 {
        0x4E218800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FCVTZS Vd.4S, Vn.4S (float → signed int)
    fn enc_fcvtzs_4s(&self, vd: u8, vn: u8) -> u32 {
        0x4EA1B800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// SCVTF Vd.4S, Vn.4S (signed int → float)
    fn enc_scvtf_4s(&self, vd: u8, vn: u8) -> u32 {
        0x4E21D800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// SHL Vd.4S, Vn.4S, #shift (左移 imm)
    fn enc_shl_4s(&self, vd: u8, vn: u8, shift: u8) -> u32 {
        let immh_immb = (shift as u32 + 32) & 0x3F;
        0x4F005400 | (immh_immb << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// ADD Vd.4S, Vn.4S, Vm.4S (整数加法)
    fn enc_add_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4EA08400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// MOVI Vd.4S, #0 (向量清零)
    fn enc_movi_zero_4s(&self, vd: u8) -> u32 {
        0x4F000400 | (vd as u32 & 0x1F)
    }

    // ── Scalar byte / GPR→NEON / NEON integer operations ──

    /// LDRB Wd, [Xn, #imm] — zero-extend byte load (unsigned offset)
    fn enc_ldrb_imm(&self, wd: u8, xn: u8, imm12: u16) -> u32 {
        0x39400000 | ((imm12 as u32 & 0xFFF) << 10) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// LDRB Wd, [Xn, Xm] — zero-extend byte load (register offset)
    fn enc_ldrb_reg(&self, wd: u8, xn: u8, xm: u8) -> u32 {
        0x38600800 | ((xm as u32 & 0x1F) << 16) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// FMOV Sd, Wn — move W register low 32 bits to NEON scalar S register.
    /// Encoding: 0x1E270000 | (Wn << 5) | Sd
    fn enc_fmov_s_from_w(&self, sd: u8, wn: u8) -> u32 {
        0x1E270000 | ((wn as u32 & 0x1F) << 5) | (sd as u32 & 0x1F)
    }

    /// FMOV Wd, Sn — move NEON scalar S register to W register (bits preserved).
    /// Encoding: 0x1E260000 | (Sn << 5) | Wd
    fn enc_fmov_w_from_s(&self, wd: u8, sn: u8) -> u32 {
        0x1E260000 | ((sn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// LDR Sd, [Xn, #offset] — load 32-bit float from memory, scaled offset (imm9).
    /// Encoding: 0xBD400000 | (imm9 << 12) | (Rn << 5) | Rt
    fn enc_ldr_s_imm(&self, sd: u8, xn: u8, imm9: u16) -> u32 {
        0xBD400000 | ((imm9 as u32 & 0x1FF) << 12) | ((xn as u32 & 0x1F) << 5) | (sd as u32 & 0x1F)
    }

    /// STR Sd, [Xn, #offset] — store 32-bit float to memory, scaled offset (imm9).
    /// Encoding: 0xBD000000 | (imm9 << 12) | (Rn << 5) | Rt
    fn enc_str_s_imm(&self, sd: u8, xn: u8, imm9: u16) -> u32 {
        0xBD000000 | ((imm9 as u32 & 0x1FF) << 12) | ((xn as u32 & 0x1F) << 5) | (sd as u32 & 0x1F)
    }

    /// FCVTZS Wd, Sn — convert float to signed integer (round toward zero).
    /// Encoding: 0x1E380000 | (Sn << 5) | Wd
    fn enc_fcvtzs_w_s(&self, wd: u8, sn: u8) -> u32 {
        0x1E380000 | ((sn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// SCVTF Sd, Wn — signed 32-bit int → float (scalar)
    /// Encoding: 0 0 0 11100 0 01 00 110 0010 10 Rn Rd = 0x1E220000 | (Rn << 5) | Rd
    fn enc_scvtf_s_w(&self, sd: u8, wn: u8) -> u32 {
        0x1E220000 | ((wn as u32 & 0x1F) << 5) | (sd as u32 & 0x1F)
    }

    /// MUL Wd, Wn, Wm — 32-bit integer multiply.
    /// Encoding: 0x1B007C00 | (Wm << 16) | (Wn << 5) | Wd
    fn enc_mul_w(&self, wd: u8, wn: u8, wm: u8) -> u32 {
        0x1B007C00 | ((wm as u32 & 0x1F) << 16) | ((wn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// USHR Vd.4S, Vn.4S, #shift — NEON unsigned right shift by immediate.
    /// For .4S elements: immh=001, shift encoded as 64-shift_amount.
    fn enc_ushr_4s(&self, vd: u8, vn: u8, shift: u8) -> u32 {
        debug_assert!(shift > 0 && shift < 32);
        let immh_immb = (64 - shift as u32) & 0x3F;
        0x6F000400 | (immh_immb << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// SUB Vd.4S, Vn.4S, Vm.4S — NEON integer subtract.
    fn enc_sub_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x6EA08400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// CMEQ Vd.4S, Vn.4S, Vm.4S — compare equal (all-1s if eq, else 0).
    fn enc_cmeq_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E208C00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FCMEQ Vd.4S, Vn.4S, Vm.4S — float compare equal (all-1s if eq, else 0)
    fn enc_fcmeq_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E20E400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FCMGT Vd.4S, Vn.4S, Vm.4S — float compare greater-than (Vn > Vm)
    fn enc_fcmgt_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4EA0EC00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FCMGE Vd.4S, Vn.4S, Vm.4S — float compare greater-or-equal (Vn >= Vm)
    fn enc_fcmge_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4EA0E400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FCMLT Vd.4S, Vn.4S, Vm.4S — float compare less-than (Vn < Vm)
    fn enc_fcmlt_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E20EC00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FCMLE Vd.4S, Vn.4S, Vm.4S — float compare less-or-equal (Vn <= Vm)
    fn enc_fcmle_4s(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x4E20E400 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// BSL Vd.16B, Vn.16B, Vm.16B — bit select: Vd[i] = Vd[i] ? Vn[i] : Vm[i]
    fn enc_bsl(&self, vd: u8, vn: u8, vm: u8) -> u32 {
        0x6E002C00 | ((vm as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FABS Vd.4S, Vn.4S — float absolute value
    fn enc_fabs_4s(&self, vd: u8, vn: u8) -> u32 {
        0x6EA0F800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FRINTP Vd.4S, Vn.4S — round to +infinity (ceil)
    fn enc_frintp_4s(&self, vd: u8, vn: u8) -> u32 {
        0x4EA19800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FRINTM Vd.4S, Vn.4S — round to -infinity (floor)
    fn enc_frintm_4s(&self, vd: u8, vn: u8) -> u32 {
        0x4EA18800 | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

/// LSL Xd, Xn, #shift — GPR logical shift left by immediate (UBFM alias).
    fn enc_lsl_x_imm(&self, rd: u8, rn: u8, shift: u8) -> u32 {
        debug_assert!(shift > 0 && shift < 64);
        let immr = (64 - shift as u32) & 0x3F;
        let imms = (63 - shift as u32) & 0x3F;
        0xD3400000 | (immr << 16) | (imms << 10) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// MOVZ Wd, #imm16 — move 16-bit immediate to W register (zero-extended).
    fn enc_movz_w(&self, wd: u8, imm16: u16) -> u32 {
        0x52800000 | ((imm16 as u32 & 0xFFFF) << 5) | (wd as u32 & 0x1F)
    }

    /// MOVK Wd, #imm16, LSL #16 — move keep (merge 16-bit at bit 16).
    fn enc_movk_w_lsl16(&self, wd: u8, imm16: u16) -> u32 {
        0x72A00000 | ((imm16 as u32 & 0xFFFF) << 5) | (wd as u32 & 0x1F)
    }

    /// AND Xd, Xn, Xm — GPR bitwise AND (64-bit).
    fn enc_and_reg(&self, rd: u8, rn: u8, rm: u8) -> u32 {
        0x8A000000 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// AND Wd, Wn, Wm — 32-bit GPR bitwise AND.
    fn enc_and_w_reg(&self, wd: u8, wn: u8, wm: u8) -> u32 {
        0x0A000000 | ((wm as u32 & 0x1F) << 16) | ((wn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// ORR Wd, Wn, Wm — 32-bit GPR bitwise OR.
    fn enc_orr_w_reg(&self, wd: u8, wn: u8, wm: u8) -> u32 {
        0x2A000000 | ((wm as u32 & 0x1F) << 16) | ((wn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// LSR Wd, Wn, #shift — 32-bit GPR logical right shift by immediate.
    /// Encoding: UBFM Wd, Wn, #shift, #(31) = 0x53000000 | (immr << 16) | (imms << 10) | (Rn << 5) | Rd
    /// immr = shift, imms = 31 for 32-bit LSR
    fn enc_lsr_w_imm(&self, wd: u8, wn: u8, shift: u8) -> u32 {
        debug_assert!(shift > 0 && shift < 32);
        0x53000000 | ((shift as u32 & 0x3F) << 16) | (31u32 << 10) | ((wn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// LSR Xd, Xn, #shift — 64-bit GPR logical right shift by immediate.
    /// Encoding: UBFM Xd, Xn, #shift, #(63)
    fn enc_lsr_x_imm(&self, xd: u8, xn: u8, shift: u8) -> u32 {
        debug_assert!(shift > 0 && shift < 64);
        0xD3400000 | ((shift as u32 & 0x3F) << 16) | (63u32 << 10) | ((xn as u32 & 0x1F) << 5) | (xd as u32 & 0x1F)
    }

    /// LSL Wd, Wn, #shift — 32-bit GPR logical left shift by immediate.
    fn enc_lsl_w_imm(&self, wd: u8, wn: u8, shift: u8) -> u32 {
        debug_assert!(shift > 0 && shift < 32);
        let immr = (32 - shift as u32) & 0x3F;
        0x53000000 | (immr << 16) | ((31 - shift as u32) << 10) | ((wn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// STRB Wd, [Xn] — store byte (immediate offset 0).
    fn enc_strb_imm(&self, wd: u8, xn: u8, imm12: u16) -> u32 {
        0x39000000 | ((imm12 as u32 & 0xFFF) << 10) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// STRB Wd, [Xn, Xm] — store byte (register offset).
    fn enc_strb_reg(&self, wd: u8, xn: u8, xm: u8) -> u32 {
        0x38200800 | ((xm as u32 & 0x1F) << 16) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// LDR Wd, [Xn, Xm, LSL #2] — 32-bit load with register offset, scaled by 4.
    fn enc_ldr_w_reg_scaled(&self, wd: u8, xn: u8, xm: u8) -> u32 {
        0xB8600800 | ((xm as u32 & 0x1F) << 16) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// LDR Wd, [Xn, #imm] — 32-bit load with unsigned immediate offset (scaled by 4).
    fn enc_ldr_w_imm(&self, wd: u8, xn: u8, imm12: u16) -> u32 {
        0xB9400000 | ((imm12 as u32 & 0xFFF) << 10) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// STR Wd, [Xn, Xm, LSL #2] — 32-bit store with register offset, scaled by 4.
    fn enc_str_w_reg_scaled(&self, wd: u8, xn: u8, xm: u8) -> u32 {
        0xB8200800 | ((xm as u32 & 0x1F) << 16) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// CSEL Wd, Wn, Wm, cond — 32-bit conditional select.
    fn enc_csel_w(&self, wd: u8, wn: u8, wm: u8, cond: u8) -> u32 {
        0x1A800000 | ((wm as u32 & 0x1F) << 16) | ((wn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F) | ((cond as u32 & 0x0F) << 12)
    }

    /// CSINC Wd, Wn, Wm, cond — conditional select increment (Wd = cond ? Wn : Wm+1).
    fn enc_csinc_w(&self, wd: u8, wn: u8, wm: u8, cond: u8) -> u32 {
        0x1A800400 | ((wm as u32 & 0x1F) << 16) | ((wn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F) | ((cond as u32 & 0x0F) << 12)
    }
}

