//! ISA Hook — 后端特化注入机制 (REGISTER-VM SPEC §13)
//!
//! 优化决策不硬编码在 lower_*() 或 IsaLower 中，
//! 而是通过 IsaHook trait 由每个后端注入最优策略。
//!
//! ## 支持的后端
//!
//! | 后端 | Hook | 核心策略 |
//! |------|------|---------|
//! | x86 AVX2 | X86Avx2Hook | FMA3, 6×2 微内核 |
//! | x86 AVX-512 | X86Avx512Hook | FMA3, 14×2 微内核, 32 zmm |
//! | x86 AMX/AMX+ | X86AmxHook | TileMma 16×16, FP16/FP8/COMPLEX |
//! | NVIDIA SM70 | GpuSm70Hook | wmma 16×16×16, FP16 TC |
//! | NVIDIA SM80 | GpuSm80Hook | mma.sync 16×8×16, cp.async |
//! | NVIDIA SM90 | GpuSm90Hook | WGMMA 16×16×64, TMA, warp_spec |
//! | NVIDIA SM100+ | GpuSm100Hook | tcgen05, TMEM, block-scaled, FP4 |
//! | AMD CDNA2 | GpuCdna2Hook | MFMA 16×16×16, wave64 |
//! | AMD CDNA3 | GpuCdna3Hook | MFMA 16×16×16, XCD topo |
//! | AMD CDNA4 | GpuCdna4Hook | MFMAv2 32×32×16, FP8/FP4 |
//! | ARM NEON | ArmNeonHook | FMLA v.4s, 128-bit |
//! | ARM SVE2 | ArmSveHook | predicated FMLA z.s |
//! | ARM SME2 | ArmSmeHook | FMOPA ZA, outer product |
//!
//! ## GEMM-FMA 族已迁移到 OpImpl (CR-TIER-SOVEREIGNTY-001..004)
//!
//! `FmaStrategy` / `select_fma` / `select_fma_best` / `select_fma_candidates` /
//! `estimate_strategy_cost` / `TileConfig` / `WgmmaConfig` / `Tcgen05Config` / `MfmaConfig`
//! 已全部删除。GEMM-FMA 算子族实现迁移到 `super::op_impl::OpImpl<GemmOpLayout>`
//! + `super::gemm_impls::select_gemm_impl` (select-then-emit 两阶段)。
//!
//! 本 IsaHook trait 仍保留 Attention / Transcendental / MoE 等其他算子族的方法
//! (它们尚未迁移到 OpImpl 框架, 属于不同算子族的并行演进, 不违反
//! ARCH-UNCONSTITUTION-CONTAGION)。

use super::isa_profile::{IsaProfile, Platform};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1.5 硬件资源预算 (供 CompileSession 持有, throughput_refine 用)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 硬件资源预算 — 用于 GEMM 策略成本微调 (throughput_refine)。
///
/// 保留: FmaStrategy 删除时, estimate_strategy_cost 抽成 throughput_refine,
/// ResourceBudget 仍被 CompileSession.budget 字段持有。
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    /// L1D 可用字节
    pub l1d_bytes: usize,
    /// L2 可用字节
    pub l2_bytes: usize,
    /// 可分配向量寄存器数量
    pub vec_reg_count: usize,
    /// 可分配 GPR 数量
    pub gpr_count: usize,
}

impl ResourceBudget {
    /// 从 IsaProfile 提取资源预算。
    pub fn from_isa_profile(profile: &super::isa_profile::IsaProfile) -> Self {
        Self {
            l1d_bytes: profile.cache.l1d_bytes,
            l2_bytes: profile.cache.l2_bytes,
            vec_reg_count: profile.vec_regs.len(),
            gpr_count: profile.gpr_regs.len(),
        }
    }
}


/// 超越函数实现策略。
#[derive(Debug, Clone)]
pub enum TransImpl {
    Polynomial { degree: u8 },
    HardwareInstr,
    TableLookup { table_size: usize },
}

/// Epilogue 执行位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpiloguePlace {
    OnAccumulators,
    AfterStore,
}

/// 预取配置。
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    pub distance: usize,
    pub hint: PrefetchHint,
}

#[derive(Debug, Clone, Copy)]
pub enum PrefetchHint { T0, T1, T2, Nta }

/// 内存访问模式。
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub stride: usize,
    pub total_bytes: usize,
    pub reuse_count: usize,
}

/// §14.5 Attention 策略。
#[derive(Debug, Clone)]
pub enum AttentionStrategy {
    Naive,
    FlashV2 { tile_q: usize, tile_kv: usize },
    /// SM90 Hopper: WGMMA + TMA pipeline
    FlashV3 { tile_q: usize, tile_kv: usize, tma: bool, warp_spec: bool },
    /// SM100+ Blackwell: tcgen05 + TMEM
    FlashV4 { tile_q: usize, tile_kv: usize, tmem: bool, block_scaled: bool },
    /// FlashDecoding: Split-K decode attention parallelism.
    /// Each warp computes partial (max, sum, O) over a KV chunk, then warp-reduce merges.
    FlashDecoding { split_k: usize, tile_kv: usize },
    SlidingWindow { window_size: usize },
}

/// §15 MoE 分发策略。
#[derive(Debug, Clone, Copy)]
pub enum MoeDispatchStrategy {
    CmpChain,
    JmpTable,
    Predicated,
    /// GPU 核内分发 (§15.1: 汇编 jmp 到专家权重区)
    InKernelJmp,
}

/// §11 KV 量化实现。
#[derive(Debug, Clone, Copy)]
pub enum KvQuantImpl { PerChannel, PerToken }

/// §16 残差总线端口。
#[derive(Debug, Clone)]
pub struct BusPortConfig {
    pub layer: usize,
    pub port_offset: usize,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 IsaHook trait (GEMM-FMA 族 select_fma/tile_config 已移除)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub trait IsaHook: Send + Sync {
    fn gemm_microkernel_shape(&self) -> (usize, usize);
    fn transcendental_impl(&self, func: super::instr::TranscendentalFn) -> TransImpl;
    fn epilogue_strategy(&self, acc_count: usize, epi_ops: usize) -> EpiloguePlace;
    fn prefetch_hint(&self, access: &AccessPattern) -> Option<PrefetchConfig>;

    /// Returns true for GPU backends (CUDA/HIP/Metal) that support shared memory.
    fn is_gpu(&self) -> bool { false }

    fn moe_dispatch(&self, expert_count: usize) -> MoeDispatchStrategy {
        if expert_count <= 8 { MoeDispatchStrategy::CmpChain }
        else { MoeDispatchStrategy::JmpTable }
    }

    fn kv_quant_codegen(&self) -> KvQuantImpl { KvQuantImpl::PerToken }

    fn select_attention(&self, seq_len: usize, head_dim: usize) -> AttentionStrategy {
        let _ = head_dim;
        if seq_len <= 32 { AttentionStrategy::Naive }
        else { AttentionStrategy::FlashV2 { tile_q: 64, tile_kv: 64 } }
    }

    /// 若返回 Some(cfg)，ResidualBusPass 在匹配到 VecBinOp(Add) → VecStore
    /// 残差结构时，会在中间插入 `[telemetry + port_offset] = residual_value`
    /// 探针 store，供外部 ResidualBus 消费 (§16)。
    ///
    /// **默认 None**：ResidualBusPass 默认跳过。只有明确需要 residual 探针
    /// 且能保证 caller 传入有效 telemetry 指针的 Hook 才应返回 Some。
    /// 原默认实现 (`Some(..)`) 对所有 Add+Store 都插入 store 到 telemetry，
    /// 在 caller 传 NULL 的主推理路径上会 SIGSEGV (ARCH-TELEMETRY-NONNULL)。
    fn residual_bus_port(&self, _injection_layer: usize) -> Option<BusPortConfig> {
        None
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 x86_64 Hook 实现
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct X86Avx2Hook;
impl IsaHook for X86Avx2Hook {
    // ARCH-AVX2-REGALLOC-BUDGET: AVX2 有 16 YMM — 减去 3 个 scratch (spill/reduce/const
    // broadcast) 剩 13 可分配。BLIS mr×nr 个累加器 + a_broadcast + b_vec = mr*nr + 2
    // 必须 ≤ 13。历史 (6,2) = 14 超池,触发 "v16 not allocated to YMM"。改为 (4,2) = 10
    // + 2 scratch = 12,留 1 slot 缓冲;微内核带宽损失由 AVX2 L1 带宽本身有限
    // (32 B/cycle) 吸收。AVX-512 有 32 ZMM,保持 (14, 2) 不变。
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (4, 2) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, acc: usize, epi: usize) -> EpiloguePlace {
        if acc + epi * 2 <= 16 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, a: &AccessPattern) -> Option<PrefetchConfig> {
        if a.total_bytes > 4096 { Some(PrefetchConfig { distance: 512, hint: PrefetchHint::T0 }) } else { None }
    }
}

pub struct X86Avx512Hook;
impl IsaHook for X86Avx512Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (14, 2) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, acc: usize, epi: usize) -> EpiloguePlace {
        if acc + epi * 2 <= 32 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 768, hint: PrefetchHint::T0 })
    }
    fn select_attention(&self, seq_len: usize, _: usize) -> AttentionStrategy {
        if seq_len <= 64 { AttentionStrategy::Naive } else { AttentionStrategy::FlashV2 { tile_q: 128, tile_kv: 128 } }
    }
}

/// x86_64 AMX+ Hook (SPR / Granite Rapids / Diamond Rapids)。
///
/// 保留 `has_amx_fp16` 等字段供 OpImpl selector 之外的诊断/调试使用;
/// GEMM-FMA 选择已迁移到 `super::gemm_impls::select_gemm_impl`,
/// 该 hook 不再决定 FmaStrategy。
pub struct X86AmxHook {
    pub has_amx_fp16: bool,
    pub has_amx_complex: bool,
    pub has_amx_fp8: bool,
    pub has_amx_transpose: bool,
}

impl IsaHook for X86AmxHook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (16, 16) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, acc: usize, epi: usize) -> EpiloguePlace {
        if acc + epi * 2 <= 32 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 1024, hint: PrefetchHint::T0 })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 NVIDIA GPU Hook 实现 (SM70 / SM80 / SM90 / SM100+)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// SM70 (Volta/Turing): wmma 16×16×16, FP16 TC, 无异步。
pub struct GpuSm70Hook;
impl IsaHook for GpuSm70Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (16, 16) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::HardwareInstr }
    fn epilogue_strategy(&self, _: usize, _: usize) -> EpiloguePlace { EpiloguePlace::AfterStore }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> { None }
    fn moe_dispatch(&self, _: usize) -> MoeDispatchStrategy { MoeDispatchStrategy::InKernelJmp }
    fn select_attention(&self, seq_len: usize, _: usize) -> AttentionStrategy {
        if seq_len <= 1 { AttentionStrategy::FlashDecoding { split_k: 4, tile_kv: 512 } }
        else if seq_len <= 64 { AttentionStrategy::Naive }
        else { AttentionStrategy::FlashV2 { tile_q: 64, tile_kv: 64 } }
    }
    fn is_gpu(&self) -> bool { true }
}

/// SM80-89 (Ampere/Ada): mma.sync 16×8×16, cp.async, BF16/TF32。
pub struct GpuSm80Hook { pub sm_version: u32 }
impl IsaHook for GpuSm80Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (16, 8) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::HardwareInstr }
    fn epilogue_strategy(&self, _: usize, epi: usize) -> EpiloguePlace {
        if epi <= 4 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 2048, hint: PrefetchHint::T0 }) // cp.async 预取
    }
    fn moe_dispatch(&self, _: usize) -> MoeDispatchStrategy { MoeDispatchStrategy::InKernelJmp }
    fn select_attention(&self, seq_len: usize, _: usize) -> AttentionStrategy {
        if seq_len <= 1 { AttentionStrategy::FlashDecoding { split_k: 4, tile_kv: 1024 } }
        else if seq_len <= 64 { AttentionStrategy::Naive }
        else { AttentionStrategy::FlashV2 { tile_q: 128, tile_kv: 128 } }
    }
    fn is_gpu(&self) -> bool { true }
}

/// SM90 (Hopper H100): WGMMA 16×N×K, TMA 2D/5D, warp specialization。
pub struct GpuSm90Hook;
impl IsaHook for GpuSm90Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (64, 16) } // Warpgroup 64×N
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::HardwareInstr }
    fn epilogue_strategy(&self, _: usize, epi: usize) -> EpiloguePlace {
        if epi <= 6 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 4096, hint: PrefetchHint::T0 }) // TMA 预取
    }
    fn moe_dispatch(&self, _: usize) -> MoeDispatchStrategy { MoeDispatchStrategy::InKernelJmp }
    fn select_attention(&self, seq_len: usize, _: usize) -> AttentionStrategy {
        if seq_len <= 1 { AttentionStrategy::FlashDecoding { split_k: 8, tile_kv: 1024 } }
        else if seq_len <= 64 { AttentionStrategy::Naive }
        else { AttentionStrategy::FlashV3 { tile_q: 128, tile_kv: 128, tma: true, warp_spec: true } }
    }
    fn is_gpu(&self) -> bool { true }
}

/// SM100+ (Blackwell B100/B200): tcgen05.mma, TMEM, block-scaled, FP4/FP6。
pub struct GpuSm100Hook;
impl IsaHook for GpuSm100Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (64, 64) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::HardwareInstr }
    fn epilogue_strategy(&self, _: usize, epi: usize) -> EpiloguePlace {
        if epi <= 8 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 8192, hint: PrefetchHint::T0 }) // TMEM + TMA
    }
    fn moe_dispatch(&self, _: usize) -> MoeDispatchStrategy { MoeDispatchStrategy::InKernelJmp }
    fn select_attention(&self, seq_len: usize, _: usize) -> AttentionStrategy {
        if seq_len <= 1 { AttentionStrategy::FlashDecoding { split_k: 8, tile_kv: 2048 } }
        else if seq_len <= 64 { AttentionStrategy::Naive }
        else { AttentionStrategy::FlashV4 { tile_q: 128, tile_kv: 256, tmem: true, block_scaled: true } }
    }
    fn is_gpu(&self) -> bool { true }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5 AMD GPU Hook 实现 (CDNA2 / CDNA3 / CDNA4)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// CDNA2 (gfx908/MI250): MFMA 16×16×16, wave64。
pub struct GpuCdna2Hook;
impl IsaHook for GpuCdna2Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (16, 16) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, _: usize, epi: usize) -> EpiloguePlace {
        if epi <= 4 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> { None }
    fn moe_dispatch(&self, _: usize) -> MoeDispatchStrategy { MoeDispatchStrategy::InKernelJmp }
    fn is_gpu(&self) -> bool { true }
}

/// CDNA3 (gfx942/MI300): MFMA 16×16×16, XCD 拓扑隔离。
pub struct GpuCdna3Hook;
impl IsaHook for GpuCdna3Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (16, 16) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, _: usize, epi: usize) -> EpiloguePlace {
        if epi <= 4 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> { None }
    fn moe_dispatch(&self, _: usize) -> MoeDispatchStrategy { MoeDispatchStrategy::InKernelJmp }
    fn is_gpu(&self) -> bool { true }
}

/// CDNA4 (gfx950/MI400): MFMA v2 32×32×16, FP8/FP4, wave64, 128KB LDS。
pub struct GpuCdna4Hook;
impl IsaHook for GpuCdna4Hook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (32, 32) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, _: usize, epi: usize) -> EpiloguePlace {
        if epi <= 6 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 4096, hint: PrefetchHint::T0 })
    }
    fn moe_dispatch(&self, _: usize) -> MoeDispatchStrategy { MoeDispatchStrategy::InKernelJmp }
    fn select_attention(&self, seq_len: usize, _: usize) -> AttentionStrategy {
        if seq_len <= 1 { AttentionStrategy::FlashDecoding { split_k: 4, tile_kv: 1024 } }
        else if seq_len <= 64 { AttentionStrategy::Naive }
        else { AttentionStrategy::FlashV2 { tile_q: 128, tile_kv: 128 } }
    }
    fn is_gpu(&self) -> bool { true }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §6 ARM Hook 实现 (NEON / SVE2 / SME2)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ARM NEON Hook: 128-bit 固定宽度, FMLA v.4s。
pub struct ArmNeonHook;
impl IsaHook for ArmNeonHook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) { (8, 12) }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, acc: usize, epi: usize) -> EpiloguePlace {
        if acc + epi * 2 <= 32 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, a: &AccessPattern) -> Option<PrefetchConfig> {
        if a.total_bytes > 4096 { Some(PrefetchConfig { distance: 512, hint: PrefetchHint::T0 }) } else { None }
    }
}

/// ARM SVE2 Hook: scalable vector, predicated ops, BLIS-style GEMM。
pub struct ArmSveHook { pub sve_vl: usize }
impl IsaHook for ArmSveHook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) {
        let lanes = self.sve_vl / 4; // f32 lanes
        (lanes * 2, 2) // 类似 BLIS 但利用 SVE 宽度
    }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, acc: usize, epi: usize) -> EpiloguePlace {
        if acc + epi * 2 <= 32 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 1024, hint: PrefetchHint::T0 })
    }
}

/// ARM SME2 Hook: ZA tile outer product, multi-vec FMLA, streaming SVE。
pub struct ArmSmeHook { pub sme_vl: usize }
impl IsaHook for ArmSmeHook {
    fn gemm_microkernel_shape(&self) -> (usize, usize) {
        let za_dim = self.sme_vl / 4;
        (za_dim, za_dim)
    }
    fn transcendental_impl(&self, _: super::instr::TranscendentalFn) -> TransImpl { TransImpl::Polynomial { degree: 5 } }
    fn epilogue_strategy(&self, acc: usize, epi: usize) -> EpiloguePlace {
        if acc + epi * 2 <= 32 { EpiloguePlace::OnAccumulators } else { EpiloguePlace::AfterStore }
    }
    fn prefetch_hint(&self, _: &AccessPattern) -> Option<PrefetchConfig> {
        Some(PrefetchConfig { distance: 2048, hint: PrefetchHint::T0 })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §7 Hook 选择器
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 从 IsaProfile 自动选择最优 Hook。
pub fn select_hook(profile: &IsaProfile) -> Box<dyn IsaHook> {
    match &profile.platform {
        // ── x86_64 ──
        Platform::X86_64 { has_amx: true, has_amx_fp16, has_amx_complex, has_amx_fp8, has_amx_transpose, .. } => {
            Box::new(X86AmxHook {
                has_amx_fp16: *has_amx_fp16,
                has_amx_complex: *has_amx_complex,
                has_amx_fp8: *has_amx_fp8,
                has_amx_transpose: *has_amx_transpose,
            })
        }
        Platform::X86_64 { has_avx512: true, .. } => Box::new(X86Avx512Hook),
        Platform::X86_64 { .. } => Box::new(X86Avx2Hook),

        // ── NVIDIA CUDA ──
        Platform::Cuda { sm_version, .. } => {
            match *sm_version {
                100.. => Box::new(GpuSm100Hook),
                90..=99 => Box::new(GpuSm90Hook),
                80..=89 => Box::new(GpuSm80Hook { sm_version: *sm_version }),
                _ => Box::new(GpuSm70Hook),
            }
        }

        // ── AMD HIP ──
        Platform::Hip { gfx_arch, .. } => {
            match *gfx_arch {
                950.. => Box::new(GpuCdna4Hook),
                940..=949 => Box::new(GpuCdna3Hook),
                908..=939 => Box::new(GpuCdna2Hook),
                _ => Box::new(GpuCdna2Hook), // RDNA fallback to CDNA2 strategies
            }
        }

        // ── ARM AArch64 ──
        Platform::AArch64 { has_sme2: true, sme_vl, .. } => {
            Box::new(ArmSmeHook { sme_vl: *sme_vl })
        }
        Platform::AArch64 { has_sve: true, sve_vl, .. } => {
            Box::new(ArmSveHook { sve_vl: *sve_vl })
        }
        Platform::AArch64 { .. } => Box::new(ArmNeonHook),

        // ── Apple Metal ──
        Platform::Metal { .. } => Box::new(GpuSm80Hook { sm_version: 80 }), // Metal 策略近似 SM80
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_hook_auto() {
        let dp = crate::dispatch::DeviceProfile::detect();
        let profile = super::super::isa_profile::IsaProfile::from_device_profile(&dp);
        let hook = select_hook(&profile);
        let (mr, nr) = hook.gemm_microkernel_shape();
        assert!(mr >= 1 && nr >= 1);
    }

    #[test]
    fn test_avx2_hook() {
        let h = X86Avx2Hook;
        // ARCH-AVX2-REGALLOC-BUDGET: microkernel 锁定 (4, 2) 以匹配 13-slot YMM 池。
        assert_eq!(h.gemm_microkernel_shape(), (4, 2));
    }

    #[test]
    fn test_avx512_hook_microkernel_and_epilogue() {
        let h = X86Avx512Hook;
        assert_eq!(h.gemm_microkernel_shape(), (14, 2));
        assert_eq!(h.epilogue_strategy(10, 4), EpiloguePlace::OnAccumulators);
        assert_eq!(h.epilogue_strategy(28, 3), EpiloguePlace::AfterStore);
        let pf = h.prefetch_hint(&AccessPattern { stride: 64, total_bytes: 8192, reuse_count: 2 }).unwrap();
        assert_eq!(pf.distance, 768);
    }

    #[test]
    fn test_sm90_hook_attention() {
        let h = GpuSm90Hook;
        match h.select_attention(2048, 128) {
            AttentionStrategy::FlashV3 { tma, warp_spec, .. } => {
                assert!(tma);
                assert!(warp_spec);
            }
            _ => panic!("expected FlashV3 on SM90"),
        }
    }

    #[test]
    fn test_sm100_hook_attention() {
        let h = GpuSm100Hook;
        match h.select_attention(4096, 128) {
            AttentionStrategy::FlashV4 { tmem, block_scaled, .. } => {
                assert!(tmem);
                assert!(block_scaled);
            }
            _ => panic!("expected FlashV4 on SM100+"),
        }
    }

    #[test]
    fn test_hook_selection_cuda_profiles() {
        for (sm, expected_type) in [(70, "Sm70"), (80, "Sm80"), (90, "Sm90"), (100, "Sm100")] {
            let profile = super::super::isa_profile::IsaProfile::cuda(sm);
            let hook = select_hook(&profile);
            let strategy = hook.select_attention(2048, 128);
            match (sm, &strategy) {
                (70, AttentionStrategy::FlashV2 { .. }) => {}
                (80, AttentionStrategy::FlashV2 { .. }) => {}
                (90, AttentionStrategy::FlashV3 { .. }) => {}
                (100, AttentionStrategy::FlashV4 { .. }) => {}
                _ => panic!("unexpected attention strategy for SM{}: {:?} (expected {})", sm, strategy, expected_type),
            }
        }
    }

    #[test]
    fn test_all_gpu_hooks_moe_in_kernel() {
        for hook in [
            select_hook(&super::super::isa_profile::IsaProfile::cuda(70)),
            select_hook(&super::super::isa_profile::IsaProfile::cuda(90)),
            select_hook(&super::super::isa_profile::IsaProfile::cuda(100)),
            select_hook(&super::super::isa_profile::IsaProfile::hip(950)),
        ] {
            assert!(matches!(hook.moe_dispatch(64), MoeDispatchStrategy::InKernelJmp));
        }
    }

    #[test]
    fn test_default_moe_dispatch_threshold() {
        let h = X86Avx2Hook;
        assert!(matches!(h.moe_dispatch(4), MoeDispatchStrategy::CmpChain));
        assert!(matches!(h.moe_dispatch(8), MoeDispatchStrategy::CmpChain));
        assert!(matches!(h.moe_dispatch(9), MoeDispatchStrategy::JmpTable));
        assert!(matches!(h.moe_dispatch(64), MoeDispatchStrategy::JmpTable));
    }

    #[test]
    fn test_default_kv_quant_and_residual_bus() {
        let h = X86Avx2Hook;
        assert!(matches!(h.kv_quant_codegen(), KvQuantImpl::PerToken));
        assert!(h.residual_bus_port(0).is_none());
        assert!(h.residual_bus_port(42).is_none());
    }

    #[test]
    fn test_default_select_attention_threshold() {
        let h = ArmNeonHook;
        assert!(matches!(h.select_attention(16, 128), AttentionStrategy::Naive));
        assert!(matches!(h.select_attention(32, 128), AttentionStrategy::Naive));
        match h.select_attention(33, 128) {
            AttentionStrategy::FlashV2 { tile_q, tile_kv } => {
                assert_eq!(tile_q, 64);
                assert_eq!(tile_kv, 64);
            }
            other => panic!("expected FlashV2 for seq_len=33, got {:?}", other),
        }
    }

    #[test]
    fn test_arm_neon_hook_epilogue_and_prefetch() {
        let h = ArmNeonHook;
        assert_eq!(h.gemm_microkernel_shape(), (8, 12));
        assert_eq!(h.epilogue_strategy(10, 4), EpiloguePlace::OnAccumulators);
        assert!(h.prefetch_hint(&AccessPattern { stride: 64, total_bytes: 1024, reuse_count: 1 }).is_none());
        let pf = h.prefetch_hint(&AccessPattern { stride: 64, total_bytes: 8192, reuse_count: 2 }).unwrap();
        assert_eq!(pf.distance, 512);
        assert!(!h.is_gpu());
    }

    #[test]
    fn test_arm_sve_hook_microkernel_scales_with_vl() {
        let h256 = ArmSveHook { sve_vl: 32 };
        let (mr256, nr256) = h256.gemm_microkernel_shape();
        assert_eq!(mr256, 16);
        assert_eq!(nr256, 2);

        let h512 = ArmSveHook { sve_vl: 64 };
        let (mr512, nr512) = h512.gemm_microkernel_shape();
        assert_eq!(mr512, 32);
        assert_eq!(nr512, 2);
        assert!(mr512 > mr256);
    }

    #[test]
    fn test_access_pattern_prefetch_variants() {
        let h = X86Avx2Hook;
        let small = AccessPattern { stride: 16, total_bytes: 256, reuse_count: 1 };
        assert!(h.prefetch_hint(&small).is_none());
        let large = AccessPattern { stride: 256, total_bytes: 65536, reuse_count: 4 };
        let pf = h.prefetch_hint(&large).unwrap();
        assert!(matches!(pf.hint, PrefetchHint::T0));
    }

    #[test]
    fn test_gpu_hooks_is_gpu_true_cpu_hooks_false() {
        let cpu_hooks: Vec<Box<dyn IsaHook>> = vec![
            Box::new(X86Avx2Hook),
            Box::new(X86Avx512Hook),
            Box::new(ArmNeonHook),
        ];
        let gpu_hooks: Vec<Box<dyn IsaHook>> = vec![
            Box::new(GpuSm70Hook),
            Box::new(GpuSm80Hook { sm_version: 86 }),
            Box::new(GpuSm90Hook),
            Box::new(GpuSm100Hook),
            Box::new(GpuCdna2Hook),
            Box::new(GpuCdna3Hook),
            Box::new(GpuCdna4Hook),
        ];
        for hook in &cpu_hooks {
            assert!(!hook.is_gpu(), "CPU hook should return is_gpu=false");
        }
        for hook in &gpu_hooks {
            assert!(hook.is_gpu(), "GPU hook should return is_gpu=true");
        }
    }

    #[test]
    fn test_sm70_hook_flash_decoding_for_decode_step() {
        let h = GpuSm70Hook;
        match h.select_attention(1, 128) {
            AttentionStrategy::FlashDecoding { split_k, tile_kv } => {
                assert_eq!(split_k, 4);
                assert_eq!(tile_kv, 512);
            }
            other => panic!("expected FlashDecoding for seq_len=1, got {:?}", other),
        }
        assert!(h.is_gpu());
        assert!(matches!(h.moe_dispatch(8), MoeDispatchStrategy::InKernelJmp));
        assert!(matches!(h.transcendental_impl(super::super::instr::TranscendentalFn::Exp), TransImpl::HardwareInstr));
    }

    #[test]
    fn test_avx2_epilogue_strategy_boundary() {
        let h = X86Avx2Hook;
        assert_eq!(h.epilogue_strategy(8, 4), EpiloguePlace::OnAccumulators); // 8 + 4*2 = 16
        assert_eq!(h.epilogue_strategy(9, 4), EpiloguePlace::AfterStore);     // 9 + 4*2 = 17 > 16
        assert_eq!(h.epilogue_strategy(14, 1), EpiloguePlace::OnAccumulators); // 14 + 1*2 = 16
        assert_eq!(h.epilogue_strategy(15, 1), EpiloguePlace::AfterStore);     // 15 + 1*2 = 17 > 16
    }

    #[test]
    fn test_epilogue_place_boundary_on_gpu_hooks() {
        let sm80 = GpuSm80Hook { sm_version: 80 };
        let sm90 = GpuSm90Hook;
        let sm100 = GpuSm100Hook;
        assert_eq!(sm80.epilogue_strategy(0, 4), EpiloguePlace::OnAccumulators);
        assert_eq!(sm80.epilogue_strategy(0, 5), EpiloguePlace::AfterStore);
        assert_eq!(sm90.epilogue_strategy(0, 6), EpiloguePlace::OnAccumulators);
        assert_eq!(sm90.epilogue_strategy(0, 7), EpiloguePlace::AfterStore);
        assert_eq!(sm100.epilogue_strategy(0, 8), EpiloguePlace::OnAccumulators);
        assert_eq!(sm100.epilogue_strategy(0, 9), EpiloguePlace::AfterStore);
    }

    #[test]
    fn test_sm80_sm90_flash_decoding_for_decode_step() {
        let sm80 = GpuSm80Hook { sm_version: 86 };
        let sm90 = GpuSm90Hook;
        match sm80.select_attention(1, 128) {
            AttentionStrategy::FlashDecoding { split_k, tile_kv } => {
                assert_eq!(split_k, 4);
                assert_eq!(tile_kv, 1024);
            }
            other => panic!("expected FlashDecoding on SM80 for seq=1, got {:?}", other),
        }
        match sm90.select_attention(1, 128) {
            AttentionStrategy::FlashDecoding { split_k, tile_kv } => {
                assert_eq!(split_k, 8);
                assert_eq!(tile_kv, 1024);
            }
            other => panic!("expected FlashDecoding on SM90 for seq=1, got {:?}", other),
        }
    }

    #[test]
    fn test_select_hook_aarch64_neon_sve_sme_profiles() {
        let neon_profile = super::super::isa_profile::IsaProfile::aarch64(false, false, 0, false, false, true);
        let sve_profile = super::super::isa_profile::IsaProfile::aarch64(true, false, 32, false, false, true);
        let sme_profile = super::super::isa_profile::IsaProfile::aarch64(true, true, 64, true, true, true);

        let neon_hook = select_hook(&neon_profile);
        let sve_hook = select_hook(&sve_profile);
        let sme_hook = select_hook(&sme_profile);

        assert_eq!(neon_hook.gemm_microkernel_shape(), (8, 12));
        let (mr_sve, _) = sve_hook.gemm_microkernel_shape();
        assert_eq!(mr_sve, 16); // vl=32 → 32/4=8 lanes * 2 = 16
        let (mr_sme, nr_sme) = sme_hook.gemm_microkernel_shape();
        assert_eq!(mr_sme, 16); // sme_vl=64 → 64/4=16
        assert_eq!(nr_sme, 16);
    }

    #[test]
    fn test_sm100_attention_strategy_thresholds() {
        let h = GpuSm100Hook;
        match h.select_attention(1, 128) {
            AttentionStrategy::FlashDecoding { split_k, tile_kv } => {
                assert_eq!(split_k, 8);
                assert!(tile_kv > 0);
            }
            other => panic!("expected FlashDecoding for seq=1, got {:?}", other),
        }
        assert!(matches!(h.select_attention(64, 128), AttentionStrategy::Naive));
        match h.select_attention(1024, 128) {
            AttentionStrategy::FlashV4 { tile_q, tile_kv, tmem, block_scaled } => {
                assert!(tile_q > 0);
                assert!(tile_kv > 0);
                assert!(tmem);
                assert!(block_scaled);
            }
            other => panic!("expected FlashV4 for long seq on SM100, got {:?}", other),
        }
    }
}
