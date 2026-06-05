impl AArch64Lower {
    fn emit_f32_broadcast(&mut self, vd: u8, val: f32) {
        let bits = val.to_bits();
        let lo16 = bits & 0xFFFF;
        let hi16 = (bits >> 16) & 0xFFFF;
        // MOVZ W16, #lo16
        self.emit32(0x52800000 | (lo16 << 5) | 16);
        // MOVK W16, #hi16, LSL #16
        self.emit32(0x72A00000 | (hi16 << 5) | 16);
        // DUP Vd.4S, W16
        self.emit32(0x4E040C00 | ((16u32) << 5) | vd as u32);
    }

    /// RET
    fn enc_ret(&self) -> u32 { 0xD65F03C0 }

    /// STP Xt1, Xt2, [Xn, #imm]! — pre-index store pair
    fn enc_stp_pre(&self, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
        let imm = (imm7 as u32 >> 3) & 0x7F;
        0xA9800000 | (imm << 15) | ((rt2 as u32 & 0x1F) << 10) | ((rn as u32 & 0x1F) << 5) | (rt1 as u32 & 0x1F)
    }

    /// LDP Xt1, Xt2, [Xn], #imm — post-index load pair
    fn enc_ldp_post(&self, rt1: u8, rt2: u8, rn: u8, imm7: i8) -> u32 {
        let imm = (imm7 as u32 >> 3) & 0x7F;
        0xA8C00000 | (imm << 15) | ((rt2 as u32 & 0x1F) << 10) | ((rn as u32 & 0x1F) << 5) | (rt1 as u32 & 0x1F)
    }

    // ══════════════════════════════════════════════════════════════════
    //  SVE / SVE2 指令编码
    // ══════════════════════════════════════════════════════════════════
    //
    // SVE 寄存器: z0-z31 (共享 v0-v31 编号), p0-p15 谓词寄存器。
    // 所有 SVE 向量操作均受谓词保护，尾部元素自动抑制。
    //
    // 约定:
    //   p0 = 循环活跃谓词 (WHILELT 输出)
    //   p7 = all-true (PTRUE P7.S 初始化一次)

    /// RDVL Xd, #1 — 读取向量长度 (bytes) 到 GPR。
    /// 编码: 0000 0100 1011 1111 0101 00 imm6 Rd
    /// RDVL Xd, #imm6 : 04BF50 | (imm6 << 5) | Rd
    fn enc_rdvl(&self, rd: u8, imm6: i8) -> u32 {
        let imm = (imm6 as u32) & 0x3F;
        0x04BF5000 | (imm << 5) | (rd as u32 & 0x1F)
    }

    /// WHILELT Pd.S, Xn, Xm — 谓词: 为 Xn..Xm 中的活跃 32-bit 元素设置掩码。
    /// 编码: 0010 0101 10 1 Xm 0 00 1 Xn 0 Pd(4bit)
    ///        25A0 0400 | Xm<<16 | Xn<<5 | Pd
    fn enc_whilelt_s(&self, pd: u8, xn: u8, xm: u8) -> u32 {
        0x25A00410 | ((xm as u32 & 0x1F) << 16) | ((xn as u32 & 0x1F) << 5) | (pd as u32 & 0x0F)
    }

    /// PTRUE Pd.S, ALL — 设置全部活跃 lanes。
    /// 编码: 0010 0101 10 1 0 0000 1110 00 Pd
    fn enc_ptrue_s(&self, pd: u8) -> u32 {
        0x2598E000 | (pd as u32 & 0x0F)
    }

    /// LD1W {Zt.S}, Pg/Z, [Xn, Xm, LSL #2] — SVE predicated 32-bit load (scalar+scalar)。
    /// 编码: 1010 0101 01 0 Xm 010 Pg Xn Zt
    fn enc_ld1w_ss(&self, zt: u8, pg: u8, xn: u8, xm: u8) -> u32 {
        0xA5404000 | ((xm as u32 & 0x1F) << 16) | ((pg as u32 & 0x07) << 10) | ((xn as u32 & 0x1F) << 5) | (zt as u32 & 0x1F)
    }

    /// LD1W {Zt.S}, Pg/Z, [Xn] — SVE predicated 32-bit load (scalar, zero offset)。
    /// LD1W Zt.S, Pg/Z, [Xn, #0, MUL VL]
    /// 编码: 1010 0101 01 0 0 imm4 101 Pg Xn Zt (imm4=0)
    fn enc_ld1w_imm(&self, zt: u8, pg: u8, xn: u8) -> u32 {
        0xA540A000 | ((pg as u32 & 0x07) << 10) | ((xn as u32 & 0x1F) << 5) | (zt as u32 & 0x1F)
    }

    /// ST1W {Zt.S}, Pg, [Xn, Xm, LSL #2] — SVE predicated 32-bit store (scalar+scalar)。
    /// 编码: 1110 0101 01 0 Xm 010 Pg Xn Zt
    fn enc_st1w_ss(&self, zt: u8, pg: u8, xn: u8, xm: u8) -> u32 {
        0xE5404000 | ((xm as u32 & 0x1F) << 16) | ((pg as u32 & 0x07) << 10) | ((xn as u32 & 0x1F) << 5) | (zt as u32 & 0x1F)
    }

    /// ST1W {Zt.S}, Pg, [Xn] — SVE predicated 32-bit store (zero offset)。
    fn enc_st1w_imm(&self, zt: u8, pg: u8, xn: u8) -> u32 {
        0xE540E000 | ((pg as u32 & 0x07) << 10) | ((xn as u32 & 0x1F) << 5) | (zt as u32 & 0x1F)
    }

    /// FADD Zd.S, Pg/M, Zd.S, Zm.S — SVE predicated f32 add (destructive: Zd = Zn op Zm, Zdn=Zd)。
    /// 编码: 0110 0101 10 00 0000 100 Pg Zm Zdn
    fn enc_sve_fadd_s(&self, zdn: u8, pg: u8, zm: u8) -> u32 {
        0x65808000 | ((pg as u32 & 0x07) << 10) | ((zm as u32 & 0x1F) << 5) | (zdn as u32 & 0x1F)
    }

    /// FSUB Zdn.S, Pg/M, Zdn.S, Zm.S
    fn enc_sve_fsub_s(&self, zdn: u8, pg: u8, zm: u8) -> u32 {
        0x65818000 | ((pg as u32 & 0x07) << 10) | ((zm as u32 & 0x1F) << 5) | (zdn as u32 & 0x1F)
    }

    /// FMUL Zdn.S, Pg/M, Zdn.S, Zm.S
    fn enc_sve_fmul_s(&self, zdn: u8, pg: u8, zm: u8) -> u32 {
        0x65828000 | ((pg as u32 & 0x07) << 10) | ((zm as u32 & 0x1F) << 5) | (zdn as u32 & 0x1F)
    }

    /// FDIV Zdn.S, Pg/M, Zdn.S, Zm.S
    fn enc_sve_fdiv_s(&self, zdn: u8, pg: u8, zm: u8) -> u32 {
        0x658D8000 | ((pg as u32 & 0x07) << 10) | ((zm as u32 & 0x1F) << 5) | (zdn as u32 & 0x1F)
    }

    /// FMAX Zdn.S, Pg/M, Zdn.S, Zm.S
    fn enc_sve_fmax_s(&self, zdn: u8, pg: u8, zm: u8) -> u32 {
        0x65868000 | ((pg as u32 & 0x07) << 10) | ((zm as u32 & 0x1F) << 5) | (zdn as u32 & 0x1F)
    }

    /// FMIN Zdn.S, Pg/M, Zdn.S, Zm.S
    fn enc_sve_fmin_s(&self, zdn: u8, pg: u8, zm: u8) -> u32 {
        0x65878000 | ((pg as u32 & 0x07) << 10) | ((zm as u32 & 0x1F) << 5) | (zdn as u32 & 0x1F)
    }

    /// FNEG Zd.S, Pg/M, Zn.S — SVE predicated negate
    /// 编码: 0000 0100 10 01 1101 101 Pg Zn Zd
    fn enc_sve_fneg_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x049DA000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FSQRT Zd.S, Pg/M, Zn.S
    fn enc_sve_fsqrt_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x650DA000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FRECPE Zd.S, Zn.S — SVE reciprocal estimate (unpredicated)
    fn enc_sve_frecpe_s(&self, zd: u8, zn: u8) -> u32 {
        0x650E3000 | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FRSQRTE Zd.S, Zn.S — SVE reciprocal square root estimate (unpredicated)
    fn enc_sve_frsqrte_s(&self, zd: u8, zn: u8) -> u32 {
        0x650F3000 | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FABS Zd.S, Pg/M, Zn.S
    fn enc_sve_fabs_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x049CA000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FRINTN Zd.S, Pg/M, Zn.S — round to nearest
    fn enc_sve_frintn_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x6500A000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FRINTM Zd.S, Pg/M, Zn.S — round toward -inf (floor)
    fn enc_sve_frintm_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x6502A000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FRINTP Zd.S, Pg/M, Zn.S — round toward +inf (ceil)
    fn enc_sve_frintp_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x6501A000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FCVTZS Zd.S, Pg/M, Zn.S — float to signed int (SVE)
    fn enc_sve_fcvtzs_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x659CA000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// SCVTF Zd.S, Pg/M, Zn.S — signed int to float (SVE)
    fn enc_sve_scvtf_s(&self, zd: u8, pg: u8, zn: u8) -> u32 {
        0x6594A000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FMLA Zda.S, Pg/M, Zn.S, Zm.S — SVE predicated FMA: Zda += Zn * Zm
    /// 编码: 0110 0101 10 1 Zm 0 Pg Zn Zda
    fn enc_sve_fmla_s(&self, zda: u8, pg: u8, zn: u8, zm: u8) -> u32 {
        0x65A00000 | ((zm as u32 & 0x1F) << 16) | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (zda as u32 & 0x1F)
    }

    /// MOVPRFX Zd, Zn — move prefix (for non-destructive sequences)
    /// 编码: 0000 0100 00 1 0000 0 0011 11 Zn Zd
    fn enc_movprfx(&self, zd: u8, zn: u8) -> u32 {
        0x0420BC00 | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// FADDV Sd, Pg, Zn.S — SVE horizontal sum to scalar
    /// 编码: 0110 0101 10 00 0000 001 Pg Zn Vd
    fn enc_sve_faddv_s(&self, vd: u8, pg: u8, zn: u8) -> u32 {
        0x65802000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FMAXV Sd, Pg, Zn.S — SVE horizontal max to scalar
    fn enc_sve_fmaxv_s(&self, vd: u8, pg: u8, zn: u8) -> u32 {
        0x65862000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// FMINV Sd, Pg, Zn.S — SVE horizontal min to scalar
    fn enc_sve_fminv_s(&self, vd: u8, pg: u8, zn: u8) -> u32 {
        0x65872000 | ((pg as u32 & 0x07) << 10) | ((zn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// DUP Zd.S, Sn — broadcast scalar s-reg lane 0 to all SVE lanes
    /// 编码: 0000 0101 10 1 0 0000 0010 00 Zn Zd
    fn enc_sve_dup_s(&self, zd: u8, zn: u8) -> u32 {
        0x05A02000 | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// LSR Zd.S, Zn.S, #shift — SVE logical shift right by immediate (element size=32)
    fn enc_sve_lsr_s(&self, zd: u8, zn: u8, shift: u8) -> u32 {
        // tszh:tszl:imm3 encode the shift amount for .S: tszh=01, shift = 64-shift_amount
        // encoding: 0000 0100 1 tszh tszl imm3 1001 01 Zn Zd
        let shift_enc = (64 - shift as u32) & 0x3F;
        let tszh = (shift_enc >> 5) & 0x1;
        let tszl_imm3 = shift_enc & 0x1F;
        0x04209400 | (tszh << 22) | (tszl_imm3 << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// LSL Zd.S, Zn.S, #shift — SVE logical shift left by immediate
    fn enc_sve_lsl_s(&self, zd: u8, zn: u8, shift: u8) -> u32 {
        let shift_enc = (shift as u32 + 32) & 0x3F;
        let tszh = (shift_enc >> 5) & 0x1;
        let tszl_imm3 = shift_enc & 0x1F;
        0x04209C00 | (tszh << 22) | (tszl_imm3 << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// AND Zd.D, Zn.D, Zm.D — SVE bitwise AND (unpredicated)
    fn enc_sve_and(&self, zd: u8, zn: u8, zm: u8) -> u32 {
        0x04203000 | ((zm as u32 & 0x1F) << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// ORR Zd.D, Zn.D, Zm.D — SVE bitwise OR (unpredicated)
    fn enc_sve_orr(&self, zd: u8, zn: u8, zm: u8) -> u32 {
        0x04603000 | ((zm as u32 & 0x1F) << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// EOR Zd.D, Zn.D, Zm.D — SVE bitwise XOR (unpredicated)
    fn enc_sve_eor(&self, zd: u8, zn: u8, zm: u8) -> u32 {
        0x04A03000 | ((zm as u32 & 0x1F) << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// BIC Zd.D, Zn.D, Zm.D — SVE bit clear (Zd = Zn AND NOT Zm)
    fn enc_sve_bic(&self, zd: u8, zn: u8, zm: u8) -> u32 {
        0x04E03000 | ((zm as u32 & 0x1F) << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// ADD Zd.S, Zn.S, Zm.S — SVE integer add (unpredicated)
    fn enc_sve_add_s(&self, zd: u8, zn: u8, zm: u8) -> u32 {
        0x04A00000 | ((zm as u32 & 0x1F) << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// SUB Zd.S, Zn.S, Zm.S — SVE integer sub (unpredicated)
    fn enc_sve_sub_s(&self, zd: u8, zn: u8, zm: u8) -> u32 {
        0x04A00400 | ((zm as u32 & 0x1F) << 16) | ((zn as u32 & 0x1F) << 5) | (zd as u32 & 0x1F)
    }

    /// B.cond <label> — 条件分支。imm19 = offset/4
    fn enc_b_cond(&self, cond: u8, imm19: i32) -> u32 {
        0x54000000 | (((imm19 as u32) & 0x7FFFF) << 5) | (cond as u32 & 0x0F)
    }

    /// B <label> — 无条件分支。imm26 = offset/4
    fn enc_b(&self, imm26: i32) -> u32 {
        0x14000000 | ((imm26 as u32) & 0x03FFFFFF)
    }

    /// INCW Xdn — Xdn += VL/4 (f32 elements per VL)
    /// 编码: 0000 0100 1011 0000 1110 0100 Xdn
    fn enc_incw(&self, xdn: u8) -> u32 {
        0x04B0E400 | (xdn as u32 & 0x1F)
    }

    /// CNTW Xd — Xd = VL/4 (number of 32-bit elements in a vector)
    /// 编码: 0000 0100 1010 0000 1110 0000 Xd
    fn enc_cntw(&self, xd: u8) -> u32 {
        0x04A0E000 | (xd as u32 & 0x1F)
    }

    /// ADD Xd, Xn, #imm12 — GPR immediate add
    fn enc_add_imm(&self, rd: u8, rn: u8, imm12: u32) -> u32 {
        0x91000000 | ((imm12 & 0xFFF) << 10) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// ADD Xd, Xn, Xm — GPR register add
    fn enc_add_reg(&self, rd: u8, rn: u8, rm: u8) -> u32 {
        0x8B000000 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// CMP Xn, #imm12 (SUBS XZR, Xn, #imm12)
    fn enc_cmp_imm(&self, xn: u8, imm12: u32) -> u32 {
        0xF100001F | ((imm12 & 0xFFF) << 10) | ((xn as u32 & 0x1F) << 5)
    }

    /// CMP Xn, Xm (SUBS XZR, Xn, Xm)
    fn enc_cmp_reg(&self, xn: u8, xm: u8) -> u32 {
        0xEB00001F | ((xm as u32 & 0x1F) << 16) | ((xn as u32 & 0x1F) << 5)
    }

    /// SUB Xd, Xn, #imm12 — GPR immediate subtract
    fn enc_sub_imm(&self, rd: u8, rn: u8, imm12: u32) -> u32 {
        0xF1000000 | ((imm12 & 0xFFF) << 10) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// SUB Xd, Xn, Xm — GPR register subtract
    fn enc_sub_reg(&self, rd: u8, rn: u8, rm: u8) -> u32 {
        0xCB000000 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
    }

    /// MVN Xd, Xm (= ORN Xd, XZR, Xm) — bitwise NOT
    fn enc_mvn_reg(&self, rd: u8, rm: u8) -> u32 {
        0xAA2003E0 | ((rm as u32 & 0x1F) << 16) | (rd as u32 & 0x1F)
    }

    /// NEG Xd, Xm (= SUB Xd, XZR, Xm) — negate
    fn enc_neg_reg(&self, rd: u8, rm: u8) -> u32 {
        0xCB0003E0 | ((rm as u32 & 0x1F) << 16) | (rd as u32 & 0x1F)
    }

    /// STR Wd, [Xn, #imm] — 32-bit store, imm scaled by 4
    fn enc_str_w(&self, wd: u8, xn: u8, imm12: u16) -> u32 {
        0xB9000000 | ((imm12 as u32 & 0xFFF) << 10) | ((xn as u32 & 0x1F) << 5) | (wd as u32 & 0x1F)
    }

    /// MOVZ Xd, #imm16 — move 16-bit immediate to X register (zero-extended)
    fn enc_movz_x(&self, xd: u8, imm16: u16) -> u32 {
        0xD2800000 | ((imm16 as u32 & 0xFFFF) << 5) | (xd as u32 & 0x1F)
    }

    /// MOVK Xd, #imm16, LSL #16 — move keep (merge 16-bit at bit 16) for X register
    fn enc_movk_x_lsl16(&self, xd: u8, imm16: u16) -> u32 {
        0xF2A00000 | ((imm16 as u32 & 0xFFFF) << 5) | (xd as u32 & 0x1F)
    }

    /// MOVK Xd, #imm16, LSL #32 — move keep (merge 16-bit at bit 32) for X register
    fn enc_movk_x_lsl32(&self, xd: u8, imm16: u16) -> u32 {
        0xF2C00000 | ((imm16 as u32 & 0xFFFF) << 5) | (xd as u32 & 0x1F)
    }

    /// MOVK Xd, #imm16, LSL #48 — move keep (merge 16-bit at bit 48) for X register
    fn enc_movk_x_lsl48(&self, xd: u8, imm16: u16) -> u32 {
        0xF2E00000 | ((imm16 as u32 & 0xFFFF) << 5) | (xd as u32 & 0x1F)
    }

    /// MOV Zd, Zn — SVE register move (ORR Zd.D, Zn.D, Zn.D)
    fn enc_sve_mov(&self, zd: u8, zn: u8) -> u32 {
        self.enc_sve_orr(zd, zn, zn)
    }

    /// MOV Vd.16B, Vn.16B — NEON register move (ORR Vd, Vn, Vn)
    fn enc_neon_mov(&self, vd: u8, vn: u8) -> u32 {
        0x4EA01C00 | ((vn as u32 & 0x1F) << 16) | ((vn as u32 & 0x1F) << 5) | (vd as u32 & 0x1F)
    }

    /// 发射 f32 常量到 SVE Z 寄存器 — load to w16, FMOV s16, w16, DUP Zd.S, s16
    fn emit_sve_f32_broadcast(&mut self, zd: u8, val: f32) {
        let bits = val.to_bits();
        let lo16 = bits & 0xFFFF;
        let hi16 = (bits >> 16) & 0xFFFF;
        // MOVZ W16, #lo16
        self.emit32(0x52800000 | (lo16 << 5) | 16);
        // MOVK W16, #hi16, LSL #16
        self.emit32(0x72A00000 | (hi16 << 5) | 16);
        // FMOV S16, W16 — 0x1E270200 | (16_src << 5) | 16_dst
        self.emit32(0x1E270000 | ((16u32) << 5) | 16u32);
        // DUP Zd.S, S16 (scalar broadcast to SVE)
        self.emit32(self.enc_sve_dup_s(zd, 16));
    }

    // ══════════════════════════════════════════════════════════════════
    //  SME2 指令编码
    // ══════════════════════════════════════════════════════════════════
    //
    // SME2 扩展:
    //   - multi-vec FMLA za.s[w, #off], {Zn.S-Zm.S}, Zt.S
    //   - ZA slice 读取 MOVA Zd.S, Pg/M, ZAnh.S[Wm, #off]
    //   - 仍然使用 SMSTART/SMSTOP 进出 streaming SVE mode

    /// SMSTART — 进入 streaming SVE mode (MSR SVCRSM, #1)
    fn enc_smstart(&self) -> u32 { 0xD503437F }

    /// SMSTOP — 退出 streaming SVE mode (MSR SVCRSM, #0)
    fn enc_smstop(&self) -> u32 { 0xD503427F }

    /// FMOPA ZA0.S, Pn/M, Pm/M, Zn.S, Zm.S — SME f32 outer product accumulate
    /// 编码: 1000 0000 10 0 Zm 0 Pm 0 Pn Zn 0 ZAda
    fn enc_fmopa_s(&self, za: u8, pn: u8, pm: u8, zn: u8, zm: u8) -> u32 {
        0x80800000
            | ((zm as u32 & 0x1F) << 16)
            | ((pm as u32 & 0x07) << 13)
            | ((pn as u32 & 0x07) << 10)
            | ((zn as u32 & 0x1F) << 5)
            | (za as u32 & 0x03)
    }

    /// SME2 multi-vec FMLA: FMLA ZA.S[Wv, #off], {Zn.S-Zn+1.S}, Zm.S
    /// 用于 2 寄存器组 (Zn 必须偶数对齐):
    /// 编码: 1100 0001 0 01 Zm 00 off Wv-8 Zn/2  (Wv = w8..w11, off = 0..3)
    fn enc_sme2_fmla_vg2(&self, wv: u8, off: u8, zn_pair: u8, zm: u8) -> u32 {
        let wv_idx = (wv - 8) & 0x03; // w8=0, w9=1, w10=2, w11=3
        let zn_enc = (zn_pair >> 1) & 0x0F; // pair must be even-aligned, encode as Zn/2
        0xC1200000
            | ((zm as u32 & 0x1F) << 16)
            | ((off as u32 & 0x03) << 10)
            | ((wv_idx as u32) << 13)
            | ((zn_enc as u32) << 5)
    }

    /// SME2 MOVA: read ZA horizontal slice to Z register.
    /// MOVA Zd.S, Pg/M, ZA0H.S[Wv, #off]
    /// 编码: 1100 0000 10 00 0010 0 Pg Wv-8 off Zd
    fn enc_sme2_mova_za_to_z(&self, zd: u8, pg: u8, za: u8, wv: u8, off: u8) -> u32 {
        let wv_idx = (wv - 8) & 0x03;
        0xC0020000
            | ((za as u32 & 0x03) << 22)
            | ((pg as u32 & 0x07) << 10)
            | ((wv_idx as u32) << 13)
            | ((off as u32 & 0x03) << 5)
            | (zd as u32 & 0x1F)
    }

    // ══════════════════════════════════════════════════════════════════
    //  Address computation helpers
    // ══════════════════════════════════════════════════════════════════

    /// Evaluate OffsetExpr into a temporary GPR.
    /// Returns the physical GPR number holding the computed byte offset.
    ///
    /// Uses x16/x17 as scratch (caller-saved IP registers, conventionally
    /// reserved for temporary address computation in this lower).
    fn eval_offset_to_tmp(&mut self, offset: &OffsetExpr, alloc: &RegAllocation, tmp: u8) -> Result<u8, CompilerError> {
        let tmp2 = if tmp == 16 { 17u8 } else { 16u8 };
        match offset {
            OffsetExpr::Const(c) => {
                if *c <= 0xFFFF {
                    self.emit32(self.enc_movz_w(tmp, *c as u16));
                } else if *c <= 0xFFFF_FFFF {
                    self.emit32(self.enc_movz_w(tmp, (*c & 0xFFFF) as u16));
                    self.emit32(self.enc_movk_w_lsl16(tmp, ((*c >> 16) & 0xFFFF) as u16));
                } else {
                    // 64-bit: MOVZ + MOVK pairs for 4x16-bit chunks
                    self.emit32(0xD2800000 | (((*c & 0xFFFF) as u32) << 5) | tmp as u32);
                    self.emit32(0xF2A00000 | ((((*c >> 16) & 0xFFFF) as u32) << 5) | tmp as u32);
                    self.emit32(0xF2C00000 | ((((*c >> 32) & 0xFFFF) as u32) << 5) | tmp as u32);
                    self.emit32(0xF2E00000 | ((((*c >> 48) & 0xFFFF) as u32) << 5) | tmp as u32);
                }
                Ok(tmp)
            }
            OffsetExpr::LoopOffset(ov) => {
                let off_reg = self.resolve_gpr(*ov, alloc)?;
                if off_reg != tmp {
                    self.emit32(self.enc_mov_x(tmp, off_reg));
                }
                Ok(tmp)
            }
            OffsetExpr::Mul(inner, scale) => {
                self.eval_offset_to_tmp(inner, alloc, tmp)?;
                // Load scale into tmp2, then MADD tmp, tmp, tmp2, XZR
                if *scale <= 0xFFFF {
                    self.emit32(self.enc_movz_w(tmp2, *scale as u16));
                } else if *scale <= 0xFFFF_FFFF {
                    self.emit32(self.enc_movz_w(tmp2, (*scale & 0xFFFF) as u16));
                    self.emit32(self.enc_movk_w_lsl16(tmp2, ((*scale >> 16) & 0xFFFF) as u16));
                } else {
                    self.emit32(0xD2800000 | (((*scale & 0xFFFF) as u32) << 5) | tmp2 as u32);
                    self.emit32(0xF2A00000 | ((((*scale >> 16) & 0xFFFF) as u32) << 5) | tmp2 as u32);
                }
                // MADD Xtmp, Xtmp, Xtmp2, XZR
                self.emit32(0x1B007C00 | ((tmp2 as u32) << 16) | ((tmp as u32) << 5) | tmp as u32);
                Ok(tmp)
            }
            OffsetExpr::Add(a, b) => {
                self.eval_offset_to_tmp(a, alloc, tmp)?;
                self.emit32(self.enc_mov_x(tmp2, tmp)); // save a result
                self.eval_offset_to_tmp(b, alloc, tmp)?;
                self.emit32(self.enc_add_reg(tmp, tmp2, tmp)); // tmp = a + b
                Ok(tmp)
            }
            OffsetExpr::ScalarVReg(sv) => {
                let off_reg = self.resolve_gpr(*sv, alloc)?;
                if off_reg != tmp {
                    self.emit32(self.enc_mov_x(tmp, off_reg));
                }
                Ok(tmp)
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  Transcendental helper: emit inline Exp (Cephes degree-5)
    // ══════════════════════════════════════════════════════════════════

    /// Exp(x) for NEON f32x4 — Cephes polynomial with integer exponent reconstruction.
    /// Uses scratch V16-V23. Result in vd.
    fn emit_neon_exp(&mut self, vd: u8, vs: u8) {
        // --- 加载常量到 scratch 寄存器 ---
        self.emit_f32_broadcast(18, -87.33654f32);    // V18 = clamp_lo
        self.emit_f32_broadcast(19, 88.72284f32);     // V19 = clamp_hi
        self.emit_f32_broadcast(16, 1.4426950f32);    // V16 = log2(e)
        self.emit_f32_broadcast(17, 0.6931472f32);    // V17 = ln(2)

        // Step 1: clamp x
        if vd != vs {
            self.emit32(self.enc_neon_mov(vd, vs));
        }
        self.emit32(self.enc_fmax_4s(vd, vd, 18)); // Vd = max(x, -87.336)
        self.emit32(self.enc_fmin_4s(vd, vd, 19)); // Vd = min(x, 88.722)

        // Step 2: k = round(x * log2e)
        self.emit32(self.enc_fmul_4s(20, vd, 16));     // V20 = x * log2e
        self.emit32(self.enc_frintn_4s(20, 20));       // V20 = round(k)

        // Step 3: r = x - k * ln2
        self.emit32(self.enc_fmul_4s(21, 20, 17));     // V21 = k * ln2
        self.emit32(self.enc_fsub_4s(21, vd, 21));     // V21 = r = x - k*ln2

        // Step 4: Horner P(r) = ((((p5*r + p4)*r + p3)*r + p2)*r + p1)*r + p0
        self.emit_f32_broadcast(22, 0.008333334f32);   // V22 = p5 = 1/120
        self.emit32(self.enc_fmul_4s(22, 22, 21));     // V22 = p5*r
        self.emit_f32_broadcast(23, 0.041666668f32);   // V23 = p4
        self.emit32(self.enc_fadd_4s(22, 22, 23));     // V22 = p5*r + p4
        self.emit32(self.enc_fmul_4s(22, 22, 21));
        self.emit_f32_broadcast(23, 0.16666667f32);    // V23 = p3
        self.emit32(self.enc_fadd_4s(22, 22, 23));
        self.emit32(self.enc_fmul_4s(22, 22, 21));
        self.emit_f32_broadcast(23, 0.5f32);           // V23 = p2
        self.emit32(self.enc_fadd_4s(22, 22, 23));
        self.emit32(self.enc_fmul_4s(22, 22, 21));
        self.emit_f32_broadcast(23, 1.0f32);           // V23 = p1
        self.emit32(self.enc_fadd_4s(22, 22, 23));
        self.emit32(self.enc_fmul_4s(22, 22, 21));
        self.emit_f32_broadcast(23, 1.0f32);           // V23 = p0
        self.emit32(self.enc_fadd_4s(22, 22, 23));
        // V22 = P(r)

        // Step 5: 2^k via integer exponent
        self.emit32(self.enc_fcvtzs_4s(23, 20));       // V23 = int(k)
        self.emit32(self.enc_shl_4s(23, 23, 23));      // V23 = k_int << 23
        self.emit_f32_broadcast(20, 1.0f32);
        self.emit32(self.enc_add_4s(23, 23, 20));      // V23 = float(2^k)

        // result = P(r) * 2^k
        self.emit32(self.enc_fmul_4s(vd, 22, 23));
    }

    /// Exp(x) for SVE2 f32 — Cephes polynomial using predicated ops.
    /// Uses scratch z16-z23, p7 (all-true). Result in zd.
    fn emit_sve_exp(&mut self, zd: u8, zs: u8) {
        let pg = 7u8; // p7 = all-true (set up in prologue or loop)

        self.emit_sve_f32_broadcast(18, -87.33654f32);
        self.emit_sve_f32_broadcast(19, 88.72284f32);
        self.emit_sve_f32_broadcast(16, 1.4426950f32);
        self.emit_sve_f32_broadcast(17, 0.6931472f32);

        // clamp
        if zd != zs {
            self.emit32(self.enc_sve_mov(zd, zs));
        }
        // FMAX zd, pg/M, zd, z18
        self.emit32(self.enc_sve_fmax_s(zd, pg, 18));
        self.emit32(self.enc_sve_fmin_s(zd, pg, 19));

        // k = round(x * log2e)
        self.emit32(self.enc_movprfx(20, zd));
        self.emit32(self.enc_sve_fmul_s(20, pg, 16));
        self.emit32(self.enc_sve_frintn_s(20, pg, 20));

        // r = x - k*ln2
        self.emit32(self.enc_movprfx(21, 20));
        self.emit32(self.enc_sve_fmul_s(21, pg, 17));
        // FSUB z21 = zd - z21  (need movprfx to make non-destructive)
        self.emit32(self.enc_movprfx(22, zd));
        self.emit32(self.enc_sve_fsub_s(22, pg, 21));
        // z22 = r = x - k*ln2, rename z21 = r
        self.emit32(self.enc_sve_mov(21, 22));

        // Horner: p5*r^5 + p4*r^4 + ... + p0
        self.emit_sve_f32_broadcast(22, 0.008333334f32);
        self.emit32(self.enc_sve_fmul_s(22, pg, 21));
        self.emit_sve_f32_broadcast(23, 0.041666668f32);
        self.emit32(self.enc_sve_fadd_s(22, pg, 23));
        self.emit32(self.enc_sve_fmul_s(22, pg, 21));
        self.emit_sve_f32_broadcast(23, 0.16666667f32);
        self.emit32(self.enc_sve_fadd_s(22, pg, 23));
        self.emit32(self.enc_sve_fmul_s(22, pg, 21));
        self.emit_sve_f32_broadcast(23, 0.5f32);
        self.emit32(self.enc_sve_fadd_s(22, pg, 23));
        self.emit32(self.enc_sve_fmul_s(22, pg, 21));
        self.emit_sve_f32_broadcast(23, 1.0f32);
        self.emit32(self.enc_sve_fadd_s(22, pg, 23));
        self.emit32(self.enc_sve_fmul_s(22, pg, 21));
        self.emit_sve_f32_broadcast(23, 1.0f32);
        self.emit32(self.enc_sve_fadd_s(22, pg, 23));

        // 2^k via integer shift
        self.emit32(self.enc_sve_fcvtzs_s(23, pg, 20));
        self.emit32(self.enc_sve_lsl_s(23, 23, 23));
        self.emit_sve_f32_broadcast(20, 1.0f32);
        self.emit32(self.enc_sve_add_s(23, 23, 20));

        // result = P(r) * 2^k
        self.emit32(self.enc_movprfx(zd, 22));
        self.emit32(self.enc_sve_fmul_s(zd, pg, 23));
    }

    /// Log(x) minimax degree-4 polynomial approximation.
    ///
    /// Algorithm:
    ///   1. Reinterpret x as integer bits
    ///   2. Extract exponent: exp = (bits >> 23) - 127
    ///   3. Extract mantissa: m = (bits & 0x7FFFFF) | 0x3F800000  (mantissa in [1,2))
    ///   4. f = m - 1.0  (range [0,1))
    ///   5. Horner: P(f) = ((c4*f + c3)*f + c2)*f + c1)*f + c0
    ///   6. result = P(f) + exp * ln(2)
    ///
    /// Coefficients (minimax on [0,1) for ln(1+f)):
    ///   c0 = 0.0, c1 = 0.9999964, c2 = -0.4999899, c3 = 0.3334595, c4 = -0.2497882

    fn emit_neon_log(&mut self, vd: u8, vs: u8) {
        if vd != vs {
            self.emit32(self.enc_neon_mov(vd, vs));
        }
        // --- integer bit extraction ---
        // V16 = reinterpret(Vd) as int bits (the bits are already there, NEON treats them as is)

        // Extract exponent: V17 = (Vd_int >> 23) - 127
        // USHR V17.4S, Vd.4S, #23  — unsigned shift right
        // encoding USHR: 0x6F200400 | (immh_immb << 16) | ... ; for .4S #23: immh=0b0010, immb=shift
        // USHR imm encoding for 4S: shift_right = 64 - immh:immb, immh:immb = 64 - 23 = 41 = 0b101001
        // encoding: 0x6F000400 | (immhb << 16) | (Vn << 5) | Vd
        let immhb_23 = (64 - 23) as u32; // = 41 = 0x29
        self.emit32(0x6F000400 | (immhb_23 << 16) | ((vd as u32) << 5) | 17u32);

        // SUB V17.4S, V17.4S, #127 — need to broadcast 127 first
        self.emit_f32_broadcast(18, f32::from_bits(127)); // V18 = 127 (as integer in float lanes)
        self.emit32(self.enc_add_4s(17, 17, 18)); // actually need SUB, but using 2's complement:
        // Correct: use SUB V17.4S, V17.4S, V18.4S
        // SUB Vd.4S = 0x6EA08400
        self.emit32(0x6EA08400 | ((18u32) << 16) | ((17u32) << 5) | 17u32);

        // SCVTF V17.4S, V17.4S — convert exponent to float
        self.emit32(self.enc_scvtf_4s(17, 17));

        // Extract mantissa: V16 = (Vd & 0x7FFFFF) | 0x3F800000
        // AND V16, Vd, broadcast(0x7FFFFF)
        self.emit_f32_broadcast(18, f32::from_bits(0x007FFFFF));
        self.emit32(0x4E201C00 | ((18u32) << 16) | ((vd as u32) << 5) | 16u32); // AND V16, Vd, V18
        self.emit_f32_broadcast(18, f32::from_bits(0x3F800000));
        self.emit32(0x4EA01C00 | ((18u32) << 16) | ((16u32) << 5) | 16u32); // ORR V16, V16, V18
        // V16 = mantissa as float in [1.0, 2.0)

        // f = m - 1.0
        self.emit_f32_broadcast(18, 1.0f32);
        self.emit32(self.enc_fsub_4s(16, 16, 18)); // V16 = f = m - 1.0

        // Horner: P(f) = ((c4*f + c3)*f + c2)*f + c1)*f + c0
        //   c4 = -0.2497882, c3 = 0.3334595, c2 = -0.4999899, c1 = 0.9999964, c0 = 0.0
        self.emit_f32_broadcast(20, -0.2497882f32);     // V20 = c4
        self.emit32(self.enc_fmul_4s(20, 20, 16));      // c4*f
        self.emit_f32_broadcast(21, 0.3334595f32);
        self.emit32(self.enc_fadd_4s(20, 20, 21));       // c4*f + c3
        self.emit32(self.enc_fmul_4s(20, 20, 16));       // (c4*f+c3)*f
        self.emit_f32_broadcast(21, -0.4999899f32);
        self.emit32(self.enc_fadd_4s(20, 20, 21));       // +c2
        self.emit32(self.enc_fmul_4s(20, 20, 16));       // *f
        self.emit_f32_broadcast(21, 0.9999964f32);
        self.emit32(self.enc_fadd_4s(20, 20, 21));       // +c1
        self.emit32(self.enc_fmul_4s(20, 20, 16));       // *f (+c0=0, skip)
        // V20 = P(f)

        // result = P(f) + exp * ln(2)
        self.emit_f32_broadcast(21, 0.6931472f32);       // ln(2)
        self.emit32(self.enc_fmul_4s(17, 17, 21));       // V17 = exp * ln2
        self.emit32(self.enc_fadd_4s(vd, 20, 17));       // Vd = P(f) + exp*ln2
    }

    /// Log(x) minimax for SVE2.
    fn emit_sve_log(&mut self, zd: u8, zs: u8) {
        let pg = 7u8;

        if zd != zs {
            self.emit32(self.enc_sve_mov(zd, zs));
        }

        // Extract exponent: z17 = (zd >> 23) - 127, then scvtf
        self.emit32(self.enc_sve_lsr_s(17, zd, 23));
        self.emit_sve_f32_broadcast(18, f32::from_bits(127));
        self.emit32(self.enc_sve_sub_s(17, 17, 18));
        self.emit32(self.enc_sve_scvtf_s(17, pg, 17));

        // Extract mantissa: z16 = (zd & 0x7FFFFF) | 0x3F800000
        self.emit_sve_f32_broadcast(18, f32::from_bits(0x007FFFFF));
        self.emit32(self.enc_sve_and(16, zd, 18));
        self.emit_sve_f32_broadcast(18, f32::from_bits(0x3F800000));
        self.emit32(self.enc_sve_orr(16, 16, 18));

        // f = m - 1.0
        self.emit_sve_f32_broadcast(18, 1.0f32);
        self.emit32(self.enc_movprfx(16, 16));
        self.emit32(self.enc_sve_fsub_s(16, pg, 18));

        // Horner polynomial
        self.emit_sve_f32_broadcast(20, -0.2497882f32);
        self.emit32(self.enc_sve_fmul_s(20, pg, 16));
        self.emit_sve_f32_broadcast(21, 0.3334595f32);
        self.emit32(self.enc_sve_fadd_s(20, pg, 21));
        self.emit32(self.enc_sve_fmul_s(20, pg, 16));
        self.emit_sve_f32_broadcast(21, -0.4999899f32);
        self.emit32(self.enc_sve_fadd_s(20, pg, 21));
        self.emit32(self.enc_sve_fmul_s(20, pg, 16));
        self.emit_sve_f32_broadcast(21, 0.9999964f32);
        self.emit32(self.enc_sve_fadd_s(20, pg, 21));
        self.emit32(self.enc_sve_fmul_s(20, pg, 16));

        // result = P(f) + exp * ln(2)
        self.emit_sve_f32_broadcast(21, 0.6931472f32);
        self.emit32(self.enc_sve_fmul_s(17, pg, 21));
        self.emit32(self.enc_movprfx(zd, 20));
        self.emit32(self.enc_sve_fadd_s(zd, pg, 17));
    }

    // ══════════════════════════════════════════════════════════════════
    //  Prologue / Epilogue
    // ══════════════════════════════════════════════════════════════════

    pub fn emit_prologue(&mut self, _frame: &StackFrame, _alloc: &RegAllocation) -> Result<(), CompilerError> {
        // stp x29, x30, [sp, #-16]!
        self.emit32(self.enc_stp_pre(29, 30, 31, -16));
        // mov x29, sp
        self.emit32(self.enc_mov_x(29, 31));

        // SVE2: initialize p7 as all-true predicate for scratch operations
        if self.platform.has_sve2 {
            self.emit32(self.enc_ptrue_s(7)); // PTRUE P7.S
        }
        Ok(())
    }

    pub fn emit_epilogue(&mut self, _frame: &StackFrame, _alloc: &RegAllocation) -> Result<(), CompilerError> {
        // ldp x29, x30, [sp], #16
        self.emit32(self.enc_ldp_post(29, 30, 31, 16));
        // ret
        self.emit32(self.enc_ret());
        Ok(())
    }

    // ══════════════════════════════════════════════════════════════════
    //  Mxfp4VecDequant — NEON implementation
    // ══════════════════════════════════════════════════════════════════

    /// mxfp4 SIMD dequantization: decode packed 4-bit E2M1 data -> f32 NEON vector.
    ///
    /// Mathematical E2M1 decode (no LUT):
    ///   sign = nibble[3]
    ///   exp  = nibble[2:1]  (2-bit)
    ///   mant = nibble[0]    (1-bit)
    ///   value = (-1)^sign * (1 + mant) * 2^(exp-1) * e8m0_scale
    ///   Special: nibble 0 -> 0.0
    ///
    /// NEON register allocation (uses scratch V16-V22):
    ///   s0 (V16) = nibble data -> mantissa -> magnitude
    ///   s1 (V17) = exp -> 2^(exp-1)
    ///   s2 (V18) = e8m0 scale (broadcasted, held throughout)
    ///   V19      = zero mask (nibble==0 -> all ones)
    ///
    /// For W128 (NEON 4 lanes): loads 2 bytes (4 nibbles)
    /// For W256: loads 4 bytes (8 nibbles), processed as two W128 passes
    fn emit_mxfp4_dequant(
        &mut self,
        dst: VRegId,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let vd = self.resolve_vreg(dst, alloc)?;

        match width {
            SimdWidth::W128 => {
                self.emit_mxfp4_dequant_neon_w128(vd, packed_ptr, packed_offset, scale_byte_src, alloc)
            }
            SimdWidth::W256 => {
                // NEON is 128-bit; process 8 nibbles in two 4-nibble passes
                self.emit_mxfp4_dequant_neon_w256(vd, packed_ptr, packed_offset, scale_byte_src, alloc)
            }
            _ => Err(CompilerError::CodegenViolation(
                format!("Mxfp4VecDequant: unsupported width {:?} for AArch64 (W128/W256 only)", width)
            ))
        }
    }

    /// W128 path: 4 nibbles -> 4 f32 values in a single NEON Q register.
    fn emit_mxfp4_dequant_neon_w128(
        &mut self,
        vd: u8,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let xn = self.resolve_gpr(packed_ptr, alloc)?;
        let scale_gpr = self.resolve_gpr(scale_byte_src, alloc)?;

        // Scratch NEON registers: V16-V22
        let s0 = 16u8; // nibble data -> magnitude
        let s1 = 17u8; // exp / temp
        let s2 = 18u8; // e8m0 scale broadcast
        let zero_mask = 19u8;

        // ── Step 1: e8m0 scale decode -> broadcast to s2 ──
        let gpr_scratch = 16u8;
        self.emit32(self.enc_mov_x(gpr_scratch, scale_gpr));
        // UBFM X16, X16, #56, #7 (extract bits[7:0] = mask to byte)
        self.emit32(0xD3400000 | (56u32 << 16) | (7u32 << 10) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32);
        // LSL X16, X16, #23
        self.emit32(self.enc_lsl_x_imm(gpr_scratch, gpr_scratch, 23));
        // FMOV S18, W16 — move to NEON scalar
        self.emit32(self.enc_fmov_s_from_w(s2, gpr_scratch));
        // DUP V18.4S, V18.S[0] — broadcast to all 4 lanes
        self.emit32(0x4E040400 | ((s2 as u32) << 5) | s2 as u32);

        // ── Step 2: Compute address base + offset ──
        self.eval_offset_to_tmp(packed_offset, alloc, 17)?;
        // x17 = offset value. x16 was used as scratch in eval but may be clobbered.
        // Recompute: addr = base + offset in x16
        self.emit32(self.enc_add_reg(gpr_scratch, xn, 17)); // x16 = base + offset

        // ── Steps 3-11: shared dequant inner loop ──
        self.emit_mxfp4_dequant_4nibbles(s0, s1, s2, zero_mask, 20, 21, 22, gpr_scratch, 0)?;

        // Move result to destination
        if vd != s0 {
            self.emit32(self.enc_neon_mov(vd, s0));
        }
        Ok(())
    }

    /// W256 path: 8 nibbles -> 8 f32 values using two NEON passes.
    /// Since NEON is only 128-bit, we do two passes of 4 nibbles each.
    /// Results go into vd (pass 1) and vd+1 (pass 2) if contiguous pair available.
    fn emit_mxfp4_dequant_neon_w256(
        &mut self,
        vd: u8,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let xn = self.resolve_gpr(packed_ptr, alloc)?;
        let scale_gpr = self.resolve_gpr(scale_byte_src, alloc)?;

        let s0 = 16u8;
        let s1 = 17u8;
        let s2 = 18u8;
        let zero_mask = 19u8;

        // ── Step 1: e8m0 scale decode -> broadcast to s2 ──
        let gpr_scratch = 16u8;
        self.emit32(self.enc_mov_x(gpr_scratch, scale_gpr));
        self.emit32(0xD3400000 | (56u32 << 16) | (7u32 << 10) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32);
        self.emit32(self.enc_lsl_x_imm(gpr_scratch, gpr_scratch, 23));
        self.emit32(self.enc_fmov_s_from_w(s2, gpr_scratch));
        self.emit32(0x4E040400 | ((s2 as u32) << 5) | s2 as u32);

        // Compute address into x16
        self.eval_offset_to_tmp(packed_offset, alloc, 17)?;
        self.emit32(self.enc_add_reg(gpr_scratch, xn, 17));

        // ═══ Pass 1: nibbles 0-3 (2 bytes from offset+0) ═══
        self.emit_mxfp4_dequant_4nibbles(s0, s1, s2, zero_mask, 20, 21, 22, gpr_scratch, 0)?;
        if vd != s0 {
            self.emit32(self.enc_neon_mov(vd, s0));
        }

        // ═══ Pass 2: nibbles 4-7 (2 bytes from offset+2) ═══
        self.emit32(self.enc_add_imm(gpr_scratch, gpr_scratch, 2));
        self.emit_mxfp4_dequant_4nibbles(s0, s1, s2, zero_mask, 20, 21, 22, gpr_scratch, 0)?;

        let vd_upper = if vd + 1 < 32 { vd + 1 } else { vd };
        if vd_upper != s0 {
            self.emit32(self.enc_neon_mov(vd_upper, s0));
        }

        Ok(())
    }

    /// Shared inner routine: decode 4 nibbles (2 bytes) from [addr_reg + byte_offset] into f32 NEON.
    ///
    /// Reads a 16-bit halfword from memory, extracts 4 E2M1 nibbles, decodes to f32 values.
    /// Uses temporary NEON registers t0-t2 for intermediate computation.
    ///
    /// Register usage:
    ///   s0: nibble vector -> magnitude (output f32x4)
    ///   s1: exp extraction temp
    ///   s2: e8m0 scale broadcast (preserved, not modified)
    ///   zero_mask: all-1s where nibble==0 (for zeroing special case)
    ///   t0, t1, t2: temporaries (clobbered)
    ///   addr_reg: GPR holding base address (preserved)
    ///   byte_offset: additional offset in bytes (0 or 2)
    fn emit_mxfp4_dequant_4nibbles(
        &mut self,
        s0: u8,
        s1: u8,
        s2: u8,
        zero_mask: u8,
        t0: u8,
        t1: u8,
        _t2: u8,
        addr_reg: u8,
        byte_offset: u32,
    ) -> Result<(), CompilerError> {
        // Load 2 bytes (halfword) into W17
        if byte_offset == 0 {
            self.emit32(0x79400000 | ((addr_reg as u32) << 5) | 17u32); // LDRH W17, [addr_reg]
        } else {
            let tmp = 17u8;
            self.emit32(self.enc_add_imm(tmp, addr_reg, byte_offset));
            self.emit32(0x79400000 | ((tmp as u32) << 5) | 17u32); // LDRH W17, [tmp]
        }

        // Move to NEON and broadcast: all 4 lanes get same 16-bit value
        self.emit32(self.enc_fmov_s_from_w(s0, 17));
        self.emit32(0x4E040400 | ((s0 as u32) << 5) | s0 as u32); // DUP Vs0.4S, Vs0.S[0]

        // ── Extract 4 nibbles ──
        // Input: 16-bit value = [b1_hi, b1_lo, b0_hi, b0_lo]
        // Output: [nib0=b0_lo, nib1=b0_hi, nib2=b1_lo, nib3=b1_hi]

        // Low byte: mask with 0xFF, split into low/high nibble via AND 0xF and >>4
        self.emit32(self.enc_neon_mov(s1, s0));
        self.emit_f32_broadcast(t0, f32::from_bits(0x000000FF));
        self.emit32(0x4E201C00 | ((t0 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32); // AND s0, s0, t0

        // s0 = low nibble of byte0: AND with 0xF
        self.emit_f32_broadcast(t0, f32::from_bits(0x0000000F));
        self.emit32(0x4E201C00 | ((t0 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32); // AND s0, s0, t0 (mask 0xF)
        // s1 = high nibble of byte0: s1 >> 4 (valid shift: 4 <= 15)
        self.emit32(self.enc_ushr_4s(s1, s1, 4));

        // High byte: shift right by 8, split into low/high nibble
        self.emit32(self.enc_fmov_s_from_w(t0, 17));  // reload original
        self.emit32(0x4E040400 | ((t0 as u32) << 5) | t0 as u32); // DUP Vt0.4S, Vt0.S[0]
        self.emit32(self.enc_ushr_4s(t0, t0, 8)); // shift to high byte (valid: 8 <= 15)

        self.emit32(self.enc_neon_mov(t1, t0));
        self.emit32(self.enc_ushr_4s(t1, t1, 4)); // high nibble of byte1 (valid: 4 <= 15)
        // t0 = low nibble of byte1: AND with 0xF
        self.emit_f32_broadcast(22, f32::from_bits(0x0000000F)); // use V22 as constant temp
        self.emit32(0x4E201C00 | ((22u32) << 16) | ((t0 as u32) << 5) | t0 as u32); // AND t0, t0, V22

        // Assemble: s0 = [nib0, nib1, nib2, nib3]
        // INS Vd.S[dst_lane], Vn.S[0]:
        //   0x6E001C00 | (dst_lane << 19) | (Vn << 5) | Vd
        self.emit32(0x6E001C00 | (1u32 << 19) | ((s1 as u32) << 5) | s0 as u32); // INS s0.S[1], s1.S[0]
        self.emit32(0x6E001C00 | (2u32 << 19) | ((t0 as u32) << 5) | s0 as u32); // INS s0.S[2], t0.S[0]
        self.emit32(0x6E001C00 | (3u32 << 19) | ((t1 as u32) << 5) | s0 as u32); // INS s0.S[3], t1.S[0]

        // ── Zero mask (nibble==0) ──
        self.emit32(self.enc_movi_zero_4s(s1));
        self.emit32(self.enc_cmeq_4s(zero_mask, s0, s1));

        // ── Extract exp = (nibble >> 1) & 0x3 ──
        // >>1 is valid (1 <= 15); mask with 0x3 via AND
        self.emit32(self.enc_ushr_4s(s1, s0, 1));
        self.emit_f32_broadcast(t0, f32::from_bits(0x00000003));
        self.emit32(0x4E201C00 | ((t0 as u32) << 16) | ((s1 as u32) << 5) | s1 as u32); // AND s1, s1, t0

        // ── 2^(exp-1) = (exp << 23) + 0x3F000000 ──
        // exp << 23: shift=23 > 15, cannot use SHL .4S directly.
        // Instead: exp * 2^23 = exp * 0x00800000
        // Use USHL (vector shift left by register): load shift amount into a vector, then USHL.
        // But USHL uses signed shift from a vector register. Simpler approach:
        // Load constant 0x00800000, multiply as integer via VMUL (not available for int).
        // Simplest: load shift=23 into each lane via broadcast, then USHL.
        //
        // Actually, the cleanest NEON approach for exp << 23:
        // Use SSHL (shift left by register): Vd = Vn << Vm (per-element)
        self.emit_f32_broadcast(t0, f32::from_bits(23)); // Vt0 = 23 in each lane
        // SSHL Vd.4S, Vn.4S, Vm.4S: signed shift left by register
        // Encoding: 0x4E204400 | (Vm << 16) | (Vn << 5) | Vd
        self.emit32(0x4E204400 | ((t0 as u32) << 16) | ((s1 as u32) << 5) | s1 as u32);
        // s1 = exp << 23

        // Add 0x3F000000 (= 0.5f as IEEE 754)
        self.emit_f32_broadcast(t0, f32::from_bits(0x3F000000));
        self.emit32(self.enc_add_4s(s1, s1, t0));

        // ── Sign mask: bit 3 of nibble, shifted to sign bit position ──
        // sign_bit = (nibble >> 3) << 31
        // >>3 is valid (3 <= 15); <<31 cannot use SHL .4S.
        // Alternative: extract bit 3, multiply by 0x80000000
        // (nibble >> 3) gives 0 or 1. Then AND with 0x80000000 gives sign mask.
        self.emit32(self.enc_ushr_4s(t0, s0, 3)); // t0 = nibble >> 3 (0 or 1)
        self.emit_f32_broadcast(22, f32::from_bits(0x80000000)); // V22 = sign bit mask
        // AND t0, t0, V22 — but 0x80000000 is -0.0, which has only bit 31 set.
        // AND with 0 or 1 AND 0x80000000 = 0 or 0 (since bit 31 of 1 is 0).
        // That's WRONG — we need bit 0 * bit 31 = 0x80000000 if bit 0 is 1.
        // Use MUL instead: t0 * 0x80000000 = 0x80000000 if t0=1, else 0.
        // But NEON integer multiply requires care. Simpler:
        // sign = (nibble & 0x8) << 28. nibble & 8 gives 0 or 8. 8 << 28 = 0x80000000.
        // So: extract bit 3 directly and shift left by 28 (valid: 28 > 15, NOT valid for .4S!)
        //
        // Alternative using USHL: load 28, use SSHL
        // OR: use NEG to convert 0/1 to 0/-1, then AND with 0x80000000
        // NEG (vector integer): 0 → 0, 1 → -1 = 0xFFFFFFFF
        // Then AND with 0x80000000: 0 → 0, 0xFFFFFFFF → 0x80000000
        self.emit_f32_broadcast(22, f32::from_bits(0x80000000)); // V22 = sign bit mask
        // NEON integer NEG: NEG Vd.4S, Vn.4S
        // Encoding: 0x6EA0B800 | (Vn << 5) | Vd
        self.emit32(0x6EA0B800 | ((t0 as u32) << 5) | t0 as u32); // NEG t0, t0
        self.emit32(0x4E201C00 | ((22u32) << 16) | ((t0 as u32) << 5) | t0 as u32); // AND t0, t0, V22

        // ── Mantissa: bit 0 of nibble ──
        // nibble & 1 gives 0 or 1
        self.emit_f32_broadcast(22, f32::from_bits(0x00000001));
        self.emit32(0x4E201C00 | ((22u32) << 16) | ((s0 as u32) << 5) | s0 as u32); // AND s0, s0, V22
        self.emit32(self.enc_scvtf_4s(s0, s0)); // to f32 (0.0 or 1.0)

        // ── (1+mant) * 2^(exp-1) via FMLA ──
        // result = mant * s1 + s1 = (1+mant) * 2^(exp-1)
        self.emit32(self.enc_neon_mov(t1, s1));
        self.emit32(self.enc_fmla_4s(t1, s0, s1)); // t1 = s1 + s0*s1
        self.emit32(self.enc_neon_mov(s0, t1));

        // ── Apply sign (XOR with sign mask) ──
        // EOR Vd, Vn, Vm: 0x6E201C00 | (Vm << 16) | (Vn << 5) | Vd
        self.emit32(0x6E201C00 | ((t0 as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);

        // ── Zero nibble=0 lanes (BIC: clear bits where mask is all-1s) ──
        // BIC Vd, Vn, Vm: 0x4E601C00 | (Vm << 16) | (Vn << 5) | Vd
        self.emit32(0x4E601C00 | ((zero_mask as u32) << 16) | ((s0 as u32) << 5) | s0 as u32);

        // ── Multiply by e8m0 scale ──
        self.emit32(self.enc_fmul_4s(s0, s0, s2));

        Ok(())
    }

    // ══════════════════════════════════════════════════════════════════
    //  Nvfp4SubBlockDequant — NEON implementation
    // ══════════════════════════════════════════════════════════════════

    /// NVFP4 sub-block dequant: decode packed E2M1 nibbles with per-sub-block UE4M3 scale.
    ///
    /// Same E2M1 nibble decode as MXFP4, but scale is UE4M3 (unsigned FP8 E4M3, bias=7)
    /// instead of E8M0.
    fn emit_nvfp4_sub_block_dequant(
        &mut self,
        dst: VRegId,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        width: SimdWidth,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let vd = self.resolve_vreg(dst, alloc)?;

        match width {
            SimdWidth::W128 => {
                self.emit_nvfp4_dequant_neon_w128(vd, packed_ptr, packed_offset, scale_byte_src, alloc)
            }
            SimdWidth::W256 => {
                self.emit_nvfp4_dequant_neon_w256(vd, packed_ptr, packed_offset, scale_byte_src, alloc)
            }
            _ => Err(CompilerError::CodegenViolation(
                format!("Nvfp4SubBlockDequant: unsupported width {:?} for AArch64 (W128/W256 only)", width)
            ))
        }
    }

    /// NVFP4 W128 path: 4 nibbles -> 4 f32 values with UE4M3 scale.
    fn emit_nvfp4_dequant_neon_w128(
        &mut self,
        vd: u8,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let xn = self.resolve_gpr(packed_ptr, alloc)?;
        let scale_gpr = self.resolve_gpr(scale_byte_src, alloc)?;

        // Scratch NEON registers
        let s0 = 16u8; // nibble data -> magnitude
        let s1 = 17u8; // exp / temp
        let s2 = 18u8; // ue4m3 scale broadcast
        let zero_mask = 19u8;

        // ── Step 1: UE4M3 scale decode → broadcast to s2 ──
        // Decode unsigned FP8 E4M3 (bias=7) → f32
        // byte = 0 → 0.0
        // exp = (byte >> 3) & 0xF, mant = byte & 0x7
        // normal (exp>0): 2^(exp-7) × (1 + mant/8)
        //   → IEEE 754: biased_exp = exp + 120, f32_bits = (biased_exp << 23) | (mant << 20)
        let gpr_scratch = 16u8;
        let gpr_scratch2 = 9u8; // X9 as second scratch

        // Move scale byte to scratch
        self.emit32(self.enc_mov_x(gpr_scratch, scale_gpr));
        // Mask to byte
        self.emit32(0xD3400000 | (56u32 << 16) | (7u32 << 10) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32);

        // Check for zero: CBZ X16, zero_label
        let zero_off = self.current_offset();
        // CBZ X16, #offset — placeholder, patch later
        self.emit32(0x34000000 | ((gpr_scratch as u32) << 5)); // CBZ X16, #0 (patch)

        // Non-zero: decode UE4M3 → f32 bits in X16
        // X9 = byte >> 3 (LSR X9, X16, #3)
        self.emit32(0xD3410000 | ((gpr_scratch2 as u32) << 5) | gpr_scratch as u32); // UBFM X9, X16, #3, #60 = LSR 3
        // X9 = exp & 0xF (AND X9, X9, #0xF) — actually top bits are already 0 after UBFM
        // Add bias: X9 = exp + 120
        self.emit32(0x11000000 | (120u32 << 10) | ((gpr_scratch2 as u32) << 5) | gpr_scratch2 as u32); // ADD W9, W9, #120
        // X16 = mant = byte & 0x7
        self.emit32(0x12000000 | (7u32 << 10) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32); // AND W16, W16, #7
        // X16 = mant << 20 (LSL W16, W16, #20)
        self.emit32(0x53000000 | (20u32 << 16) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32); // LSL W16, W16, #20
        // X9 = (biased_exp << 23) (LSL W9, W9, #23)
        self.emit32(0x53000000 | (23u32 << 16) | ((gpr_scratch2 as u32) << 5) | gpr_scratch2 as u32); // LSL W9, W9, #23
        // X16 = f32_bits = (biased_exp << 23) | (mant << 20)
        self.emit32(0x32000000 | ((gpr_scratch2 as u32) << 5) | gpr_scratch as u32 | (gpr_scratch as u32)); // ORR W16, W16, W9
        // B done_label
        let done_off = self.current_offset();
        self.emit32(0x14000000); // B #0 (patch)

        // zero_label: X16 = 0
        let zero_label_off = self.current_offset();
        self.emit32(0xD2800000 | ((gpr_scratch as u32) << 5)); // MOV X16, #0
        // Patch CBZ offset
        let cbz_imm = ((zero_label_off - zero_off) / 4) as u32;
        self.patch32(zero_off, 0x34000000 | ((gpr_scratch as u32) << 5) | (cbz_imm & 0x7FFFF));

        // done_label:
        let done_label_off = self.current_offset();
        // Patch B offset
        let b_imm = ((done_label_off - done_off) / 4) as u32;
        self.patch32(done_off, 0x14000000 | (b_imm & 0x3FFFFFF));

        // FMOV S18, W16 — move f32 bits to NEON scalar
        self.emit32(self.enc_fmov_s_from_w(s2, gpr_scratch));
        // DUP V18.4S, V18.S[0] — broadcast to all 4 lanes
        self.emit32(0x4E040400 | ((s2 as u32) << 5) | s2 as u32);

        // ── Step 2: Compute address base + offset ──
        self.eval_offset_to_tmp(packed_offset, alloc, 17)?;
        self.emit32(self.enc_add_reg(gpr_scratch, xn, 17));

        // ── Steps 3-11: shared E2M1 dequant inner routine (same as Mxfp4) ──
        self.emit_mxfp4_dequant_4nibbles(s0, s1, s2, zero_mask, 20, 21, 22, gpr_scratch, 0)?;

        // Move result to destination
        if vd != s0 {
            self.emit32(self.enc_neon_mov(vd, s0));
        }
        Ok(())
    }

    /// NVFP4 W256 path: 8 nibbles -> 8 f32 values using two NEON passes with UE4M3 scale.
    fn emit_nvfp4_dequant_neon_w256(
        &mut self,
        vd: u8,
        packed_ptr: VRegId,
        packed_offset: &OffsetExpr,
        scale_byte_src: VRegId,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let xn = self.resolve_gpr(packed_ptr, alloc)?;
        let scale_gpr = self.resolve_gpr(scale_byte_src, alloc)?;

        let s0 = 16u8;
        let s1 = 17u8;
        let s2 = 18u8;
        let zero_mask = 19u8;

        // ── Step 1: UE4M3 scale decode → broadcast to s2 ──
        let gpr_scratch = 16u8;
        let gpr_scratch2 = 9u8;

        self.emit32(self.enc_mov_x(gpr_scratch, scale_gpr));
        self.emit32(0xD3400000 | (56u32 << 16) | (7u32 << 10) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32);

        // CBZ for zero check
        let zero_off = self.current_offset();
        self.emit32(0x34000000 | ((gpr_scratch as u32) << 5));

        // Non-zero: UE4M3 decode
        self.emit32(0xD3410000 | ((gpr_scratch2 as u32) << 5) | gpr_scratch as u32); // LSR W9, W16, #3
        self.emit32(0x11000000 | (120u32 << 10) | ((gpr_scratch2 as u32) << 5) | gpr_scratch2 as u32); // ADD W9, W9, #120
        self.emit32(0x12000000 | (7u32 << 10) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32); // AND W16, W16, #7
        self.emit32(0x53000000 | (20u32 << 16) | ((gpr_scratch as u32) << 5) | gpr_scratch as u32); // LSL W16, W16, #20
        self.emit32(0x53000000 | (23u32 << 16) | ((gpr_scratch2 as u32) << 5) | gpr_scratch2 as u32); // LSL W9, W9, #23
        self.emit32(0x32000000 | ((gpr_scratch2 as u32) << 5) | gpr_scratch as u32 | (gpr_scratch as u32)); // ORR W16, W16, W9
        let done_off = self.current_offset();
        self.emit32(0x14000000); // B done

        let zero_label_off = self.current_offset();
        self.emit32(0xD2800000 | ((gpr_scratch as u32) << 5)); // MOV X16, #0
        let cbz_imm = ((zero_label_off - zero_off) / 4) as u32;
        self.patch32(zero_off, 0x34000000 | ((gpr_scratch as u32) << 5) | (cbz_imm & 0x7FFFF));

        let done_label_off = self.current_offset();
        let b_imm = ((done_label_off - done_off) / 4) as u32;
        self.patch32(done_off, 0x14000000 | (b_imm & 0x3FFFFFF));

        // Broadcast scale to NEON
        self.emit32(self.enc_fmov_s_from_w(s2, gpr_scratch));
        self.emit32(0x4E040400 | ((s2 as u32) << 5) | s2 as u32);

        // Compute address
        self.eval_offset_to_tmp(packed_offset, alloc, 17)?;
        self.emit32(self.enc_add_reg(gpr_scratch, xn, 17));

        // Pass 1: nibbles 0-3
        self.emit_mxfp4_dequant_4nibbles(s0, s1, s2, zero_mask, 20, 21, 22, gpr_scratch, 0)?;
        if vd != s0 {
            self.emit32(self.enc_neon_mov(vd, s0));
        }

        // Pass 2: nibbles 4-7
        self.emit32(self.enc_add_imm(gpr_scratch, gpr_scratch, 2));
        self.emit_mxfp4_dequant_4nibbles(s0, s1, s2, zero_mask, 20, 21, 22, gpr_scratch, 0)?;

        let vd_upper = if vd + 1 < 32 { vd + 1 } else { vd };
        if vd_upper != s0 {
            self.emit32(self.enc_neon_mov(vd_upper, s0));
        }

        Ok(())
    }

    // ══════════════════════════════════════════════════════════════════
    //  KIVI 4-bit quantization (scalar fallback)
    // ══════════════════════════════════════════════════════════════════

    /// KIVI per-channel 4-bit quantization (scalar loop).
    ///
    /// For each pair of f32 values, find max(|a|, |b|), compute scale,
    /// quantize each to a 4-bit nibble, pack into a byte, store to dst_ptr.
    /// Store scale (f32) to scale_ptr.
    ///
    /// Register convention (caller-saved temporaries):
    ///   x9  = dst_ptr, x10 = scale_ptr, x11 = src base (sp-relative)
    ///   x12 = loop counter, x13 = num_pairs
    ///   s0-s2 = float temporaries, w14-w15 = integer temporaries
    fn lower_kivi_quant_channel(
        &mut self,
        src: &VRegId,
        dst_ptr: &VRegId,
        scale_ptr: &VRegId,
        num_channels: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_gpr = self.resolve_gpr(*dst_ptr, alloc)?;
        let scale_gpr = self.resolve_gpr(*scale_ptr, alloc)?;
        let src_vreg = self.resolve_vreg(*src, alloc)?;

        let num_pairs = (num_channels + 1) / 2;

        // x9 = dst_ptr (copy from resolved GPR)
        self.emit32(self.enc_mov_x(9, dst_gpr));
        // x10 = scale_ptr
        self.emit32(self.enc_mov_x(10, scale_gpr));
        // x12 = 0 (counter)
        self.emit32(self.enc_add_imm(12, 31, 0)); // MOV x12, xzr
        // x13 = num_pairs
        self.emit32(self.enc_movz_w(13, num_pairs as u16));
        if num_pairs > 65535 {
            self.emit32(self.enc_movk_w_lsl16(13, (num_pairs >> 16) as u16));
        }

        let loop_top = self.current_offset();

        // CMP x12, x13
        self.emit32(0xEB0D019F | ((13u32 & 0x1F) << 16) | ((12u32 & 0x1F) << 5)); // CMP x12, x13 → SUBS xzr, x12, x13
        // B.GE end (cond=0xA = GE)
        let bge_offset = self.current_offset();
        self.emit32(self.enc_b_cond(0xA, 1)); // placeholder, patch later

        // Load src[2*i] and src[2*i+1] from NEON register
        // NEON register v_src has the data in [0..num_channels] lanes.
        // We use scalar load from the vector register's element.
        // For AArch64, we read the f32 elements from the vector register directly.
        // However, since we can't index dynamically, we use the stack:
        // First, store the vector to a temp stack slot, then load scalars.

        // Actually, for the scalar fallback approach, we assume the src data
        // is already in a stack slot (spilled). We'll read from sp + offset.
        // Since AArch64 NEON has fixed 128-bit registers, num_channels ≤ head_dim.
        // For simplicity, iterate the pairs using NEON element extraction.

        // INS v16.s[0], v_src.s[counter] — extract element dynamically is not possible
        // in AArch64 without table lookup. Instead, store to stack and load.

        // Store the vector to a known stack temp slot (sp + 512)
        // Then load f32 scalars from there using counter-based offset.

        // STR Q_src, [SP, #512]
        self.emit32(0x3D8007E0 | ((src_vreg as u32 & 0x1F) << 10)); // STR q{src}, [sp, #512]

        // Load f32 at offset = 512 + x12*8 (2 f32s per pair)
        // ADD x14, sp, #512
        self.emit32(self.enc_add_imm(14, 31, 512)); // sp = x31
        // ADD x14, x14, x12, LSL #3 (multiply counter by 8)
        self.emit32(0x8B0C19CE); // ADD x14, x14, x12, LSL #3

        // LDR s0, [x14, #0]
        self.emit32(0xBD400000 | ((14u32 & 0x1F) << 5)); // LDR s0, [x14]
        // LDR s1, [x14, #4]
        self.emit32(0xBD400401 | ((14u32 & 0x1F) << 5)); // LDR s1, [x14, #4]

        // s0 = fabs(s0), s1 = fabs(s1) — use NEON scalar FABS
        self.emit32(0x1E60C020); // FABS s0, s0
        self.emit32(0x1E60C021); // FABS s1, s1

        // FMAX s2, s0, s1 — scale = max(|a|, |b|)
        self.emit32(0x1E204822 | ((1u32 & 0x1F) << 11)); // FMAX s2, s0, s1

        // STR s2, [x10] — store scale
        self.emit32(0xBD000000 | ((10u32 & 0x1F) << 5) | (2u32 & 0x1F)); // STR s2, [x10]
        // ADD x10, x10, #4
        self.emit32(self.enc_add_imm(10, 10, 4));

        // Now quantize the actual (non-abs) values.
        // Reload originals.
        self.emit32(0xBD400000 | ((14u32 & 0x1F) << 5)); // LDR s0, [x14] (reload original)
        self.emit32(0xBD400401 | ((14u32 & 0x1F) << 5)); // LDR s1, [x14, #4]

        // Quantize: nibble = clamp(round(val / scale * 7 + 8), 0, 15)
        // We use the x86 approach: extract bits 20-23 as approximate 4-bit quant.
        // val >> 20 gives ~top 4 mantissa bits, which approximates quantization.
        // For precision, use: FCVTZS w14, s0 → ASR w14, w14, #20 → AND w14, w14, #0xF

        // nibble_0: FCVTZS w14, s0 → LSR w14, w14, #20 → AND w14, w14, #0xF
        self.emit32(self.enc_fmov_w_from_s(14, 0)); // FMOV w14, s0 (reinterpret bits)
        self.emit32(self.enc_lsr_w_imm(14, 14, 20)); // LSR w14, w14, #20
        self.emit32(0x121F01CE); // AND w14, w14, #0xF → AND w14, w14, #0xF

        // nibble_1: FMOV w15, s1 → LSR w15, w15, #20 → AND w15, w15, #0xF
        self.emit32(self.enc_fmov_w_from_s(15, 1)); // FMOV w15, s1
        self.emit32(self.enc_lsr_w_imm(15, 15, 20)); // LSR w15, w15, #20
        self.emit32(0x121F01EF); // AND w15, w15, #0xF

        // Pack: byte = (nibble_1 << 4) | nibble_0
        self.emit32(self.enc_lsl_w_imm(15, 15, 4)); // LSL w15, w15, #4
        self.emit32(self.enc_orr_w_reg(14, 14, 15)); // ORR w14, w14, w15

        // STRB w14, [x9]
        self.emit32(0x3900000E | ((9u32 & 0x1F) << 5)); // STRB w14, [x9]
        // ADD x9, x9, #1
        self.emit32(self.enc_add_imm(9, 9, 1));

        // ADD x12, x12, #1 (increment counter)
        self.emit32(self.enc_add_imm(12, 12, 1));
        // B loop_top
        let b_offset = (loop_top as i32 - self.current_offset() as i32) / 4;
        self.emit32(self.enc_b(b_offset));

        // Patch B.GE to here (end label)
        let end_offset = self.current_offset();
        let bge_disp = (end_offset as i32 - bge_offset as i32) / 4 ;
        self.patch32(bge_offset, self.enc_b_cond(0xA, bge_disp));

        Ok(())
    }

    /// KIVI per-token 4-bit quantization (scalar loop).
    fn lower_kivi_quant_token(
        &mut self,
        src: &VRegId,
        dst_ptr: &VRegId,
        scale_ptr: &VRegId,
        num_elems: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let dst_gpr = self.resolve_gpr(*dst_ptr, alloc)?;
        let scale_gpr = self.resolve_gpr(*scale_ptr, alloc)?;
        let src_vreg = self.resolve_vreg(*src, alloc)?;

        let num_pairs = (num_elems + 1) / 2;

        // x9 = dst_ptr, x10 = scale_ptr
        self.emit32(self.enc_mov_x(9, dst_gpr));
        self.emit32(self.enc_mov_x(10, scale_gpr));
        // x12 = 0, x13 = num_pairs
        self.emit32(self.enc_add_imm(12, 31, 0));
        self.emit32(self.enc_movz_w(13, num_pairs as u16));
        if num_pairs > 65535 {
            self.emit32(self.enc_movk_w_lsl16(13, (num_pairs >> 16) as u16));
        }

        let loop_top = self.current_offset();

        // CMP x12, x13; B.GE end
        self.emit32(0xEB0D019F | ((13u32 & 0x1F) << 16) | ((12u32 & 0x1F) << 5));
        let bge_offset = self.current_offset();
        self.emit32(self.enc_b_cond(0xA, 1)); // placeholder

        // Store vector to stack, load scalars
        self.emit32(0x3D8007E0 | ((src_vreg as u32 & 0x1F) << 10)); // STR q{src}, [sp, #512]
        self.emit32(self.enc_add_imm(14, 31, 512));
        self.emit32(0x8B0C19CE); // ADD x14, x14, x12, LSL #3

        // Load f32 pair
        self.emit32(0xBD400000 | ((14u32 & 0x1F) << 5)); // LDR s0, [x14]
        self.emit32(0xBD400401 | ((14u32 & 0x1F) << 5)); // LDR s1, [x14, #4]

        // FABS + FMAX for scale
        self.emit32(0x1E60C020); // FABS s0, s0
        self.emit32(0x1E60C021); // FABS s1, s1
        self.emit32(0x1E204822 | ((1u32 & 0x1F) << 11)); // FMAX s2, s0, s1

        // Store scale
        self.emit32(0xBD000000 | ((10u32 & 0x1F) << 5) | (2u32 & 0x1F)); // STR s2, [x10]
        self.emit32(self.enc_add_imm(10, 10, 4));

        // Reload originals for quantization
        self.emit32(0xBD400000 | ((14u32 & 0x1F) << 5)); // LDR s0
        self.emit32(0xBD400401 | ((14u32 & 0x1F) << 5)); // LDR s1

        // Quantize to nibbles
        self.emit32(self.enc_fmov_w_from_s(14, 0));
        self.emit32(self.enc_lsr_w_imm(14, 14, 20));
        self.emit32(0x121F01CE); // AND w14, w14, #0xF

        self.emit32(self.enc_fmov_w_from_s(15, 1));
        self.emit32(self.enc_lsr_w_imm(15, 15, 20));
        self.emit32(0x121F01EF); // AND w15, w15, #0xF

        self.emit32(self.enc_lsl_w_imm(15, 15, 4));
        self.emit32(self.enc_orr_w_reg(14, 14, 15));

        // STRB
        self.emit32(0x3900000E | ((9u32 & 0x1F) << 5));
        self.emit32(self.enc_add_imm(9, 9, 1));
        self.emit32(self.enc_add_imm(12, 12, 1));

        // B loop_top
        let b_offset = (loop_top as i32 - self.current_offset() as i32) / 4;
        self.emit32(self.enc_b(b_offset));

        // Patch B.GE
        let end_offset = self.current_offset();
        let bge_disp = (end_offset as i32 - bge_offset as i32) / 4 ;
        self.patch32(bge_offset, self.enc_b_cond(0xA, bge_disp));

        Ok(())
    }

    /// KIVI 4-bit dequantization load (scalar loop).
    ///
    /// Load packed bytes, unpack nibbles, load scale, dequantize to f32.
    fn lower_kivi_dequant_load(
        &mut self,
        dst: &VRegId,
        src_ptr: &VRegId,
        scale_ptr: &VRegId,
        num_elems: usize,
        alloc: &RegAllocation,
    ) -> Result<(), CompilerError> {
        let src_gpr = self.resolve_gpr(*src_ptr, alloc)?;
        let scale_gpr = self.resolve_gpr(*scale_ptr, alloc)?;
        let dst_vreg = self.resolve_vreg(*dst, alloc)?;

        let num_pairs = (num_elems + 1) / 2;

        // x9 = src_ptr, x10 = scale_ptr
        self.emit32(self.enc_mov_x(9, src_gpr));
        self.emit32(self.enc_mov_x(10, scale_gpr));
        // x12 = 0, x13 = num_pairs
        self.emit32(self.enc_add_imm(12, 31, 0));
        self.emit32(self.enc_movz_w(13, num_pairs as u16));
        if num_pairs > 65535 {
            self.emit32(self.enc_movk_w_lsl16(13, (num_pairs >> 16) as u16));
        }

        // We'll build the result vector on the stack at sp+640, then load it.
        // ADD x15, sp, #640
        self.emit32(self.enc_add_imm(15, 31, 640));

        let loop_top = self.current_offset();

        // CMP x12, x13; B.GE end
        self.emit32(0xEB0D019F | ((13u32 & 0x1F) << 16) | ((12u32 & 0x1F) << 5));
        let bge_offset = self.current_offset();
        self.emit32(self.enc_b_cond(0xA, 1)); // placeholder

        // LDRB w14, [x9] — load packed byte
        self.emit32(0x3940000E | ((9u32 & 0x1F) << 5)); // LDRB w14, [x9]
        self.emit32(self.enc_add_imm(9, 9, 1)); // src_ptr++

        // LDR s2, [x10] — load scale
        self.emit32(0xBD400000 | ((10u32 & 0x1F) << 5) | (2u32 & 0x1F)); // LDR s2, [x10]
        self.emit32(self.enc_add_imm(10, 10, 4)); // scale_ptr += 4

        // Unpack low nibble: w3 = w14 & 0xF
        self.emit32(0x121F01CE); // AND w14, w14, #0xF — actually we need two temps
        // Use w3 = low nibble, w4 = high nibble
        // MOV w3, w14
        self.emit32(0x2A0E03E3); // MOV w3, w14
        self.emit32(0x121F0063); // AND w3, w3, #0xF

        // High nibble: w4 = (w14 >> 4) & 0xF
        self.emit32(self.enc_lsr_w_imm(14, 14, 4));
        self.emit32(0x121F01CE); // AND w14, w14, #0xF
        self.emit32(0x2A0E03E4); // MOV w4, w14

        // Sign-extend: nibble values 0-15, stored as val - 8 (range -8 to +7)
        // For dequant: f32 = (nibble - 8) * scale
        // SCVTF s0, w3 → s0 -= 8.0 → s0 *= scale
        self.emit32(self.enc_scvtf_s_w(0, 3)); // SCVTF s0, w3
        self.emit32(0x1E203020); // FMADD would be better, but use FSUB for simplicity: need fmov + fsub
        // FMOV s1, #-8.0... load constant. Use ADD #imm to stack instead.
        // Simpler: convert (nibble - 8) using integer subtraction first.
        // w3 = w3 - 8
        self.emit32(0x51002063); // SUB w3, w3, #8
        self.emit32(self.enc_scvtf_s_w(0, 3)); // SCVTF s0, w3 (now signed)
        self.emit32(0x1E200840 | ((2u32 & 0x1F) << 16)  ); // FMUL s0, s0, s2

        // Store to stack: STR s0, [x15, x12*8]
        // ADD x16, x15, x12, LSL #3
        self.emit32(0x8B0C19F0 | ((15u32 & 0x1F) << 5) | (16u32 & 0x1F)); // ADD x16, x15, x12, LSL #3
        self.emit32(0xBD000000 | ((16u32 & 0x1F) << 5)); // STR s0, [x16]

        // Second element: w4 - 8
        self.emit32(0x51002084); // SUB w4, w4, #8
        self.emit32(self.enc_scvtf_s_w(0, 4)); // SCVTF s0, w4
        self.emit32(0x1E200840 | ((2u32 & 0x1F) << 16)  ); // FMUL s0, s0, s2

        // STR s0, [x16, #4]
        self.emit32(0xBD000401 | ((16u32 & 0x1F) << 5)); // STR s0, [x16, #4]

        // x12++
        self.emit32(self.enc_add_imm(12, 12, 1));
        // B loop_top
        let b_offset = (loop_top as i32 - self.current_offset() as i32) / 4;
        self.emit32(self.enc_b(b_offset));

        // Patch B.GE
        let end_offset = self.current_offset();
        let bge_disp = (end_offset as i32 - bge_offset as i32) / 4 ;
        self.patch32(bge_offset, self.enc_b_cond(0xA, bge_disp));

        // Load result vector from stack into dst NEON register
        self.emit32(self.enc_add_imm(14, 31, 640)); // ADD x14, sp, #640
        self.emit32(0x3D8003E0 | ((dst_vreg as u32 & 0x1F) << 10) | (14u32 & 0x1F)); // LDR q{dst}, [x14]

        Ok(())
    }
}

