//! ISA Profile — 硬件规则加载 (REGISTER-VM SPEC §6)
//!
//! Register VM 加载 IsaProfile 来决定：
//! - 寄存器数量和分配策略
//! - ABI 约定 (入参寄存器/callee-saved/栈对齐/red zone)
//! - 硬件特性 (FMA/BF16/Tile/AsyncCopy/WGMMA/TMA/MFMA)
//! - 缓存层级 (L1/L2/L3/TMEM/LDS)
//!
//! ## 支持的硬件世代
//!
//! | 平台 | 世代 | 核心能力 |
//! |------|------|---------|
//! | NVIDIA | SM70 (Volta) | wmma, FP16 TC |
//! | NVIDIA | SM80 (Ampere) | mma.sync, cp.async, BF16/TF32 |
//! | NVIDIA | SM89 (Ada) | mma.sync, cp.async, FP8 |
//! | NVIDIA | SM90 (Hopper) | WGMMA, TMA, warp_spec, FP8, cuda::barrier |
//! | NVIDIA | SM100+ (Blackwell) | tcgen05.mma, TMEM, block-scaled, FP4/FP6 |
//! | AMD | gfx908 (CDNA2/MI250) | MFMA v1, wave64 |
//! | AMD | gfx942 (CDNA3/MI300) | MFMA v1, wave64, XCD topo |
//! | AMD | gfx950 (CDNA4/MI400) | MFMA v2, FP8/FP4, wave64 |
//! | AMD | gfx1100 (RDNA3) | wave32 |
//! | Intel | AVX2 | 16 ymm, FMA |
//! | Intel | AVX-512 | 32 zmm, VNNI, BF16 |
//! | Intel | AMX (SPR) | tile 16×16, BF16, INT8 |
//! | Intel | AMX+ (GR/DR) | AMX-FP16, AMX-COMPLEX, AMX-Transpose, FP8 |
//! | Intel | AVX10.2 + APX | 31 GPR, VP2INTERSECT, BF16-256 |
//! | ARM | NEON | 128-bit fixed |
//! | ARM | SVE2 (Neoverse V2) | scalable 128-2048 bit |
//! | ARM | SME2 (Neoverse V3) | ZA tile, outer product, multi-vec |

use crate::dispatch::DeviceProfile;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 物理寄存器标识
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 物理 GPR 寄存器。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysGpr(pub u8);

/// 物理向量寄存器。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysVec(pub u8);

/// 物理 Tile 寄存器 (AMX tmm / SME ZA / Tensor Core)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysTile(pub u8);

/// 物理掩码寄存器 (AVX-512 k / SVE p)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysMask(pub u8);

/// 物理寄存器 (统一枚举，RegAllocator 的输出)。
///
/// `Spilled(slot_id)` 表示该 VReg 没分到物理寄存器，被 RegAllocator 放到栈
/// spill slot `slot_id`（对应 `RegAllocation::spills[slot_id]`）。
/// ISA Lower 通过 `resolve_gpr_read/write/rw` 在每次使用时从栈 load 到
/// scratch GPR / store 回栈。ARCH-REGALLOC-GPR-SPILL。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysReg {
    Gpr(PhysGpr),
    Vec(PhysVec),
    Tile(PhysTile),
    Mask(PhysMask),
    /// 栈 spill（slot index into RegAllocation.spills）
    Spilled(u32),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 平台
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone)]
pub enum Platform {
    X86_64 {
        // ── 基础 ISA ──
        has_avx512: bool,
        has_bf16: bool,          // AVX-512 BF16 (VDPBF16PS)
        has_vnni: bool,          // VNNI INT8 (vpdpbusd)
        has_avx512fp16: bool,    // AVX-512 FP16
        // ── AMX 世代 ──
        has_amx: bool,           // AMX-BF16 + AMX-INT8 (Sapphire Rapids)
        has_amx_fp16: bool,      // AMX-FP16 (Granite Rapids: TDPFP16PS)
        has_amx_complex: bool,   // AMX-COMPLEX (Diamond Rapids: TCMMIMFP16PS)
        has_amx_transpose: bool, // AMX-TRANSPOSE (Diamond Rapids: T2RPNTLVWZ)
        has_amx_fp8: bool,       // AMX-FP8 (Diamond Rapids)
        // ── AVX10 / APX ──
        has_avx10_2: bool,       // AVX10.2 (256-bit 统一, VMINMAXPS)
        has_apx: bool,           // APX 31 GPR (Extended GPR)
        has_vp2intersect: bool,  // VP2INTERSECT (sparse mask 硬件化)
    },
    AArch64 {
        // ── NEON (基线) ──
        has_bf16: bool,          // FEAT_BF16 (bfdot/bfmmla)
        has_dotprod: bool,       // FEAT_DotProd (sdot/udot)
        has_i8mm: bool,          // FEAT_I8MM (smmla/ummla)
        // ── SVE ──
        has_sve: bool,           // SVE 基础
        has_sve2: bool,          // SVE2 (整数/浮点扩展)
        sve_vl: usize,          // SVE 向量长度 (bytes, 16-256)
        // ── SME ──
        has_sme: bool,           // SME 基础 (ZA tile, SMSTART/SMSTOP)
        has_sme2: bool,          // SME2 (multi-vec FMLA, streaming SVE mode)
        has_sme_f16f16: bool,    // FEAT_SME_F16F16
        has_sme_i16i64: bool,    // FEAT_SME_I16I64
        sme_vl: usize,          // SME streaming 向量长度 (bytes)
    },
    Cuda {
        sm_version: u32,         // 70/75/80/86/89/90/100
        warp_size: u32,          // 32
        shared_mem_kb: usize,    // per-block SMEM (48-228 KB)
        reg_file_per_sm: usize,  // 寄存器文件大小
        max_regs_per_thread: usize,
        // ── SM90 Hopper 特性 ──
        has_wgmma: bool,         // Warpgroup MMA (16×16×64)
        has_tma: bool,           // Tensor Memory Accelerator (2D/5D)
        has_warp_spec: bool,     // Warp Specialization (producer/consumer)
        has_fp8: bool,           // FP8 native Tensor Core
        // ── SM100+ Blackwell 特性 ──
        has_tmem: bool,          // Tensor Memory (256KB/SM, 替代 SMEM)
        has_block_scaled: bool,  // Block-scaled GEMM (per-block 缩放因子内置)
        has_native_fp4: bool,    // FP4 native Tensor Core
        has_native_fp6: bool,    // FP6 native Tensor Core
        has_cluster: bool,       // Thread Block Cluster (跨 CTA 协同)
        has_2cta_mma: bool,      // 2-CTA 协同 MMA
        tmem_size_kb: usize,     // TMEM 大小 (0 if !has_tmem)
    },
    Hip {
        gfx_arch: u32,          // 908/942/950/1100 (替代纯 wave_size)
        wave_size: u32,          // 64 (CDNA) / 32 (RDNA)
        has_mfma: bool,          // Matrix Fused Multiply-Add v1 (gfx908+)
        has_mfma_v2: bool,       // MFMA v2 (gfx950: 更大 tile, FP8)
        has_fp8_mfma: bool,      // FP8 MFMA (gfx950)
        has_fp4_mfma: bool,      // FP4 MFMA (gfx950)
        vgpr_per_cu: usize,     // 向量 GPR 数/CU
        lds_size_kb: usize,     // Local Data Share (类 SMEM)
        infinity_cache_mb: usize, // Infinity Cache 大小
    },
    Metal {
        simd_width: u32,         // 32 (Apple GPU)
        gpu_family: u32,         // Apple GPU family (7=M1, 8=M2, 9=M3, 10=M4)
        has_simdgroup_matrix: bool, // simdgroup_matrix_multiply
        threadgroup_mem_kb: usize,  // Threadgroup memory
    },
}

impl Platform {
    /// NVIDIA SM 版本快捷方法
    pub fn cuda_sm(&self) -> Option<u32> {
        match self {
            Platform::Cuda { sm_version, .. } => Some(*sm_version),
            _ => None,
        }
    }

    /// AMD GFX 架构版本快捷方法
    pub fn hip_gfx(&self) -> Option<u32> {
        match self {
            Platform::Hip { gfx_arch, .. } => Some(*gfx_arch),
            _ => None,
        }
    }

    /// 是否为 GPU 平台
    pub fn is_gpu(&self) -> bool {
        matches!(self, Platform::Cuda { .. } | Platform::Hip { .. } | Platform::Metal { .. })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 ABI 约定
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone)]
pub struct AbiConvention {
    /// 函数入参 GPR (x86 SysV: rdi=0,rsi=1,rdx=2,rcx=3,r8=4,r9=5)
    pub arg_regs: Vec<PhysGpr>,
    /// 栈参数起始偏移 ([rbp+16] for x86 SysV)
    pub stack_arg_offset: i32,
    /// Callee-saved GPR
    pub callee_saved: Vec<PhysGpr>,
    /// Caller-saved GPR
    pub caller_saved: Vec<PhysGpr>,
    /// Callee-saved 向量寄存器 (x86 SysV: 无; Windows: xmm6-15)
    pub callee_saved_vec: Vec<PhysVec>,
    /// 栈帧对齐要求 (x86: 16)
    pub stack_alignment: usize,
    /// Red Zone 大小 (x86 SysV: 128; Windows: 0)
    pub red_zone_bytes: usize,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 硬件特性
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsaFeature {
    // ── 通用 ──
    Fma,
    NativeBf16,
    NativeFp16,
    NativeFp8,
    NativeFp4,
    NativeFp6,
    TileGemm { m: u8, n: u8, k: u8 },
    AsyncCopy,
    PredicatedExec,
    ScalableVector { min_vl: usize, max_vl: usize },
    WarpShuffle,
    HardwareTranscendental,

    // ── Intel x86 特性 ──
    /// VNNI INT8 点积 (vpdpbusd)
    Vnni,
    /// AMX-TRANSPOSE (Diamond Rapids: T2RPNTLVWZ)
    AmxTranspose,
    /// AMX-FP16 (Granite Rapids: TDPFP16PS)
    AmxFp16,
    /// AMX-COMPLEX (Diamond Rapids: TCMMIMFP16PS / TCMMRLFP16PS)
    AmxComplex,
    /// AMX-FP8 (Diamond Rapids)
    AmxFp8,
    /// Intel MOVRS read-shared hint (Diamond Rapids)
    Movrs,
    /// AVX10.2 256-bit 统一 SIMD
    Avx10_2,
    /// APX 31 GPR (Extended General Purpose Registers)
    Apx31Gpr,
    /// VP2INTERSECT 稀疏掩码硬件化
    Vp2Intersect,

    // ── NVIDIA CUDA 特性 ──
    /// Warpgroup MMA (SM90+: wgmma.mma_async 16×16×64)
    Wgmma,
    /// Tensor Memory Accelerator (SM90+: TMA 2D/5D prefetch)
    Tma,
    /// Tensor Memory (SM100+: 256KB/SM, 替代 SMEM)
    Tmem,
    /// Block-scaled GEMM (SM100+: per-block 缩放因子内置)
    BlockScaled,
    /// Thread Block Cluster (SM100+: 跨 CTA 协同)
    ThreadBlockCluster,
    /// 2-CTA 协同 MMA (SM100+)
    TwoCta,
    /// Warp Specialization (SM90+: producer/consumer 双线程组)
    WarpSpecialization,
    /// cuda::barrier (SM90+)
    CudaBarrier,
    /// L2 Multicast (SM90+)
    L2Multicast,

    // ── AMD HIP 特性 ──
    /// Matrix Fused Multiply-Add (gfx908+)
    Mfma,
    /// MFMA v2 (gfx950: 更大 tile, 新精度)
    MfmaV2,
    /// FP8 MFMA (gfx950)
    Fp8Mfma,
    /// FP4 MFMA (gfx950)
    Fp4Mfma,
    /// XCD/GCD 拓扑隔离 (CDNA3+)
    XcdTopology,

    // ── ARM 特性 ──
    /// SVE2 整数/浮点扩展
    Sve2,
    /// SME2 Multi-Vector (multi-vec FMLA)
    Sme2MultiVec,
    /// SME Tile Operations (ZA 2D outer product)
    SmeTileOp,
    /// SME F16×F16 → F16 (FEAT_SME_F16F16)
    SmeF16F16,
    /// SME I16×I64 (FEAT_SME_I16I64)
    SmeI16I64,
    /// BF16 点积 (FEAT_BF16: bfdot/bfmmla)
    ArmBf16,
    /// Integer dot product (FEAT_DotProd: sdot/udot)
    ArmDotProd,
    /// I8MM (FEAT_I8MM: smmla/ummla)
    ArmI8mm,
    /// Apple simdgroup_matrix_multiply
    SiMDGroupMatrix,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5 缓存层级
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone)]
pub struct CacheHierarchy {
    pub l1d_bytes: usize,
    pub l1i_bytes: usize,
    pub l2_bytes: usize,
    pub l3_bytes: usize,
    pub cacheline_bytes: usize,
    /// GPU TMEM (SM100+) 或 0
    pub tmem_bytes: usize,
    /// GPU SMEM per block 或 0
    pub smem_bytes: usize,
    /// AMD LDS per CU 或 0
    pub lds_bytes: usize,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §6 IsaProfile
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ISA 硬件 Profile——Register VM 的全部硬件规则。
#[derive(Debug, Clone)]
pub struct IsaProfile {
    pub platform: Platform,
    /// 可用 GPR (分配器从中选择)
    pub gpr_regs: Vec<PhysGpr>,
    /// ISA Lower 独占的 scratch GPR（**不可分配**）。
    ///
    /// ARCH-ISA-SCRATCH: ISA Lower 翻译 `OffsetExpr`/`ScalarExpr` 这类带内部中间值的
    /// VmInstr 时，需要若干临时寄存器承载求值栈顶值。这些寄存器由架构保留，
    /// RegAllocator 不得分配给 VReg，否则 ISA Lower 写入会覆盖活跃 VReg。
    ///
    /// 典型约定 (x86_64 SysV)：
    /// - `scratch_gprs[0]` = 主 scratch（OffsetExpr 求值结果）— 传统为 rax
    /// - `scratch_gprs[1]` = 副 scratch（Add 左子临时保存）— 传统为 r11
    ///
    /// 数量和物理编号由 ISA 决定，但必须 ≥ 求值最大嵌套深度所需。
    pub scratch_gprs: Vec<PhysGpr>,
    /// ISA Lower 独占的 scratch 向量寄存器（**不可分配**）。
    ///
    /// ARCH-ISA-SCRATCH-VEC: 用于向量 reduce / ScalarLoad→ScalarToIndex 位模式转换 /
    /// 横向归约 / spill scratch 等需要临时向量承载中间值的场景。
    ///
    /// 典型约定 (x86_64 AVX2, 16 ymm 中选 6)：
    /// - `scratch_vec_regs[0]` = 内部 scratch 0 (HReduce 的 vextractf128 + FWHT permute, 传统 ymm15)
    /// - `scratch_vec_regs[1]` = 内部 scratch 1 (FWHT add/sub, 传统 ymm14)
    /// - `scratch_vec_regs[2]` = 内部 scratch 2 (常量广播 / ScalarLoad 位模式容器 / Sigmoid·Tanh, 传统 ymm13)
    /// - `scratch_vec_regs[3]` = spill scratch A (VecBinOp/Fma 输入 a 装载, 传统 ymm12)
    /// - `scratch_vec_regs[4]` = spill scratch B (VecBinOp/Fma 输入 b 装载, 传统 ymm11)
    /// - `scratch_vec_regs[5]` = spill scratch C (Fma 累加器 acc / 输出 dst 中转, 传统 ymm10)
    ///
    /// 设计原因 (ARCH-MHA-VEC-SPILL): MHA register-only 路径在 hd_vecs > 物理寄存器池
    /// 时会 spill 长生命周期的 o_acc[d]。一条 VecBinOp/Fma 可能 dst/a/b 全部 spilled,
    /// 需要 3 个互不冲突的 spill scratch；同时 HReduce/Cephes 等内部计算仍需自身 scratch,
    /// 故拆分成"内部 scratch"和"spill scratch"两组。
    pub scratch_vec_regs: Vec<PhysVec>,
    /// 可用向量寄存器
    pub vec_regs: Vec<PhysVec>,
    /// 可用 tile 寄存器
    pub tile_regs: Vec<PhysTile>,
    /// 可用掩码寄存器
    pub mask_regs: Vec<PhysMask>,
    /// ABI 约定
    pub abi: AbiConvention,
    /// 缓存层级
    pub cache: CacheHierarchy,
    /// 硬件特性集
    pub features: Vec<IsaFeature>,
    /// §0.2.10: GEMM K循环展开因子 (from DeviceProfile::k_unroll_factor)
    pub k_unroll_factor: usize,
    /// Hardware dot-product capability (from DeviceProfile), drives quant microkernel selection.
    pub dot_cap: crate::dispatch::device_profile::DotProductCap,
}

impl IsaProfile {
    /// 检查是否有某个特性
    pub fn has_feature(&self, feat: &IsaFeature) -> bool {
        self.features.iter().any(|f| std::mem::discriminant(f) == std::mem::discriminant(feat))
    }

    /// 最优 SIMD 宽度
    pub fn optimal_simd_width(&self) -> super::instr::SimdWidth {
        match &self.platform {
            Platform::X86_64 { has_avx512: true, .. } => super::instr::SimdWidth::W512,
            Platform::X86_64 { has_avx10_2: true, .. } => super::instr::SimdWidth::W256,
            Platform::X86_64 { .. } => super::instr::SimdWidth::W256,
            Platform::AArch64 { has_sve: true, .. } => super::instr::SimdWidth::Scalable,
            Platform::AArch64 { .. } => super::instr::SimdWidth::W128,
            Platform::Cuda { warp_size, .. } => super::instr::SimdWidth::Warp(*warp_size),
            Platform::Hip { wave_size, .. } => super::instr::SimdWidth::Warp(*wave_size),
            Platform::Metal { simd_width, .. } => super::instr::SimdWidth::Warp(*simd_width),
        }
    }

    /// 从 DeviceProfile 构造 x86_64 IsaProfile。
    pub fn from_device_profile(dp: &DeviceProfile) -> Self {
        let kc = &dp.kernel_config;
        let hw = &dp.hw_info;

        // x86_64 System V ABI GPR 编号: rax=0 rcx=1 rdx=2 rbx=3 rsp=4 rbp=5 rsi=6 rdi=7 r8-r15=8-15
        let arg_regs = vec![
            PhysGpr(7),  // rdi
            PhysGpr(6),  // rsi
            PhysGpr(2),  // rdx
            PhysGpr(1),  // rcx
            PhysGpr(8),  // r8
            PhysGpr(9),  // r9
        ];
        let callee_saved = vec![
            PhysGpr(3),   // rbx
            PhysGpr(12),  // r12
            PhysGpr(13),  // r13
            PhysGpr(14),  // r14
            PhysGpr(15),  // r15
        ];
        // ARCH-ISA-SCRATCH: ISA Lower 独占 scratch (不进入 RegAllocator 可分配池)。
        // rax(0): OffsetExpr 求值主 scratch
        // r10(10): spill load/store scratch #1
        // r11(11): spill load/store scratch #2 + Add 左子临时保存
        // rsp(4), rbp(5): 栈帧寄存器，任何 ISA 都不可分配
        // APX 扩展: r16-r30 额外 15 个 GPR
        let scratch_gprs: Vec<PhysGpr> = vec![PhysGpr(0), PhysGpr(10), PhysGpr(11)];

        // caller_saved/callee_saved 必须剔除 scratch_gprs，否则 RegAllocator
        // 会把 scratch 分给普通 VReg，与 ISA Lower 的 scratch 使用冲突 (ARCH-ISA-SCRATCH)。
        let caller_saved = vec![
            PhysGpr(1),   // rcx
            PhysGpr(2),   // rdx
            PhysGpr(6),   // rsi
            PhysGpr(7),   // rdi
            PhysGpr(8),   // r8
            PhysGpr(9),   // r9
        ];
        let frame_gprs: Vec<u8> = vec![4, 5];
        let max_gpr = if kc.has_apx { 31u8 } else { 16u8 };
        let gpr_regs: Vec<PhysGpr> = (0..max_gpr)
            .filter(|r| !frame_gprs.contains(r) && !scratch_gprs.iter().any(|s| s.0 == *r))
            .map(PhysGpr)
            .collect();

        // 向量寄存器 (ARCH-ISA-SCRATCH-VEC: 末六个保留为 Lower scratch)
        // 见 IsaProfile.scratch_vec_regs 文档说明拆分原因。
        let vec_count: u8 = if kc.use_avx512 { 32 } else { 16 };
        let scratch_vec_regs: Vec<PhysVec> = vec![
            PhysVec(vec_count - 1), // ymm/zmm 15 or 31 — 内部 scratch 0 (HReduce/FWHT)
            PhysVec(vec_count - 2), // ymm/zmm 14 or 30 — 内部 scratch 1 (FWHT)
            PhysVec(vec_count - 3), // ymm/zmm 13 or 29 — 内部 scratch 2 (broadcast/Sigmoid)
            PhysVec(vec_count - 4), // ymm/zmm 12 or 28 — spill scratch A (输入 a)
            PhysVec(vec_count - 5), // ymm/zmm 11 or 27 — spill scratch B (输入 b)
            PhysVec(vec_count - 6), // ymm/zmm 10 or 26 — spill scratch C (acc/dst 中转)
        ];
        let vec_regs: Vec<PhysVec> = (0..vec_count)
            .filter(|&r| !scratch_vec_regs.iter().any(|s| s.0 == r))
            .map(PhysVec)
            .collect();

        // 特性
        let mut features = vec![IsaFeature::Fma];
        if kc.use_avx512 { features.push(IsaFeature::PredicatedExec); }
        if kc.has_bf16 { features.push(IsaFeature::NativeBf16); }
        if kc.has_avx512fp16 { features.push(IsaFeature::NativeFp16); }
        if kc.has_vnni { features.push(IsaFeature::Vnni); }
        if kc.has_amx {
            features.push(IsaFeature::TileGemm { m: 16, n: 16, k: 32 });
        }
        // AMX+ (Granite Rapids / Diamond Rapids)
        if kc.has_amx_fp16 { features.push(IsaFeature::AmxFp16); }
        if kc.has_amx_complex { features.push(IsaFeature::AmxComplex); }
        if kc.has_amx_transpose { features.push(IsaFeature::AmxTranspose); }
        if kc.has_amx_fp8 {
            features.push(IsaFeature::AmxFp8);
            features.push(IsaFeature::NativeFp8);
        }
        // AVX10.2 / APX
        if kc.has_avx10_2 { features.push(IsaFeature::Avx10_2); }
        if kc.has_apx {
            features.push(IsaFeature::Apx31Gpr);
        }
        if kc.has_vp2intersect { features.push(IsaFeature::Vp2Intersect); }

        let tile_regs = if kc.has_amx { (0..8).map(PhysTile).collect() } else { vec![] };
        let mask_regs = if kc.use_avx512 { (0..8).map(PhysMask).collect() } else { vec![] };

        Self {
            platform: Platform::X86_64 {
                has_avx512: kc.use_avx512,
                has_bf16: kc.has_bf16,
                has_vnni: kc.has_vnni,
                has_avx512fp16: kc.has_avx512fp16,
                has_amx: kc.has_amx,
                has_amx_fp16: kc.has_amx_fp16,
                has_amx_complex: kc.has_amx_complex,
                has_amx_transpose: kc.has_amx_transpose,
                has_amx_fp8: kc.has_amx_fp8,
                has_avx10_2: kc.has_avx10_2,
                has_apx: kc.has_apx,
                has_vp2intersect: kc.has_vp2intersect,
            },
            gpr_regs,
            scratch_gprs,
            vec_regs,
            scratch_vec_regs,
            tile_regs,
            mask_regs,
            abi: AbiConvention {
                arg_regs,
                stack_arg_offset: 16,
                callee_saved,
                caller_saved,
                callee_saved_vec: vec![],
                stack_alignment: 16,
                red_zone_bytes: 128,
            },
            cache: CacheHierarchy {
                l1d_bytes: hw.l1d_bytes,
                l1i_bytes: 32768,
                l2_bytes: hw.l2_bytes,
                l3_bytes: hw.l3_bytes,
                cacheline_bytes: hw.cacheline_bytes,
                tmem_bytes: 0,
                smem_bytes: 0,
                lds_bytes: 0,
            },
            features,
            k_unroll_factor: dp.k_unroll_factor(),
            dot_cap: dp.dot_product_cap(),
        }
    }

    /// 从 SM 版本构造 NVIDIA CUDA IsaProfile。
    pub fn cuda(sm_version: u32) -> Self {
        let is_hopper = sm_version >= 90;
        let is_blackwell = sm_version >= 100;
        let warp_size = 32;

        let (smem_kb, reg_file, max_regs) = match sm_version {
            100.. => (228, 65536, 255),  // Blackwell
            90..=99 => (228, 65536, 255),  // Hopper
            80..=89 => (164, 65536, 255),  // Ampere/Ada
            70..=79 => (96, 65536, 255),   // Volta/Turing
            60..=69 => (48, 65536, 255),   // Pascal (GTX 1060 = SM 6.1)
            _ => (48, 65536, 255),         // Kepler/Maxwell (保守默认)
        };

        let tmem_kb = if is_blackwell { 256 } else { 0 };

        let mut features = vec![
            IsaFeature::Fma,
            IsaFeature::WarpShuffle,
        ];

        // SM70+: wmma (Tensor Core v1)
        if sm_version >= 70 {
            features.push(IsaFeature::TileGemm { m: 16, n: 16, k: 16 });
        }
        // SM80+: mma.sync, cp.async, BF16/TF32
        if sm_version >= 80 {
            features.push(IsaFeature::AsyncCopy);
            features.push(IsaFeature::NativeBf16);
        }
        // SM89+: FP8
        if sm_version >= 89 {
            features.push(IsaFeature::NativeFp8);
        }
        // SM90 Hopper: WGMMA, TMA, Warp Specialization
        if is_hopper {
            features.push(IsaFeature::Wgmma);
            features.push(IsaFeature::Tma);
            features.push(IsaFeature::WarpSpecialization);
            features.push(IsaFeature::CudaBarrier);
            features.push(IsaFeature::L2Multicast);
        }
        // SM100+ Blackwell: TMEM, Block-Scaled, FP4/FP6, Cluster, 2-CTA
        if is_blackwell {
            features.push(IsaFeature::Tmem);
            features.push(IsaFeature::BlockScaled);
            features.push(IsaFeature::NativeFp4);
            features.push(IsaFeature::NativeFp6);
            features.push(IsaFeature::ThreadBlockCluster);
            features.push(IsaFeature::TwoCta);
        }

        Self {
            platform: Platform::Cuda {
                sm_version,
                warp_size,
                shared_mem_kb: smem_kb,
                reg_file_per_sm: reg_file,
                max_regs_per_thread: max_regs,
                has_wgmma: is_hopper,
                has_tma: is_hopper,
                has_warp_spec: is_hopper,
                has_fp8: sm_version >= 89,
                has_tmem: is_blackwell,
                has_block_scaled: is_blackwell,
                has_native_fp4: is_blackwell,
                has_native_fp6: is_blackwell,
                has_cluster: is_blackwell,
                has_2cta_mma: is_blackwell,
                tmem_size_kb: tmem_kb,
            },
            gpr_regs: vec![],   // GPU: 无 GPR 概念
            scratch_gprs: vec![], // GPU: ISA Lower 不走 x86 scratch 路径
            vec_regs: vec![],   // GPU: 由 reg_file 管理
            scratch_vec_regs: vec![],
            tile_regs: vec![],  // GPU: Tensor Core 由硬件调度
            mask_regs: vec![],
            abi: AbiConvention {
                arg_regs: vec![],
                stack_arg_offset: 0,
                callee_saved: vec![],
                caller_saved: vec![],
                callee_saved_vec: vec![],
                stack_alignment: 0,
                red_zone_bytes: 0,
            },
            cache: CacheHierarchy {
                l1d_bytes: smem_kb * 1024,
                l1i_bytes: 8192,       // GPU L1i ~8KB
                l2_bytes: 0,           // GPU L2 由 driver 管理
                l3_bytes: 0,
                cacheline_bytes: 128,  // GPU cacheline 128B
                tmem_bytes: tmem_kb * 1024,
                smem_bytes: smem_kb * 1024,
                lds_bytes: 0,
            },
            features,
            k_unroll_factor: if sm_version >= 90 { 8 } else if sm_version >= 70 { 4 } else { 2 },
            dot_cap: if sm_version >= 80 {
                crate::dispatch::device_profile::DotProductCap::NativeBf16
            } else if sm_version >= 70 {
                crate::dispatch::device_profile::DotProductCap::SimdAssisted
            } else {
                // SM61 Pascal: FP32 FMA only, no Tensor Core, no native BF16
                crate::dispatch::device_profile::DotProductCap::SimdAssisted
            },
        }
    }

    /// 从 gfx arch 构造 AMD HIP IsaProfile。
    pub fn hip(gfx_arch: u32) -> Self {
        let is_cdna = gfx_arch < 1100;
        let wave_size = if is_cdna { 64 } else { 32 };
        let is_cdna4 = gfx_arch >= 950;
        let is_cdna3 = (940..950).contains(&gfx_arch);

        let (lds_kb, vgpr, infinity_mb) = match gfx_arch {
            950.. => (128, 512, 256),    // CDNA4 MI400
            940..=949 => (64, 512, 128), // CDNA3 MI300
            908..=939 => (64, 256, 0),   // CDNA2 MI250
            _ => (64, 256, 0),           // RDNA3
        };

        let mut features = vec![
            IsaFeature::Fma,
            IsaFeature::WarpShuffle,
        ];
        if is_cdna && gfx_arch >= 908 {
            features.push(IsaFeature::Mfma);
            features.push(IsaFeature::TileGemm { m: 16, n: 16, k: 16 });
        }
        if is_cdna3 {
            features.push(IsaFeature::XcdTopology);
        }
        if is_cdna4 {
            features.push(IsaFeature::MfmaV2);
            features.push(IsaFeature::Fp8Mfma);
            features.push(IsaFeature::Fp4Mfma);
            features.push(IsaFeature::NativeFp8);
            features.push(IsaFeature::NativeFp4);
            features.push(IsaFeature::XcdTopology);
        }

        Self {
            platform: Platform::Hip {
                gfx_arch,
                wave_size,
                has_mfma: is_cdna && gfx_arch >= 908,
                has_mfma_v2: is_cdna4,
                has_fp8_mfma: is_cdna4,
                has_fp4_mfma: is_cdna4,
                vgpr_per_cu: vgpr,
                lds_size_kb: lds_kb,
                infinity_cache_mb: infinity_mb,
            },
            gpr_regs: vec![],
            scratch_gprs: vec![],
            vec_regs: vec![],
            scratch_vec_regs: vec![],
            tile_regs: vec![],
            mask_regs: vec![],
            abi: AbiConvention {
                arg_regs: vec![],
                stack_arg_offset: 0,
                callee_saved: vec![],
                caller_saved: vec![],
                callee_saved_vec: vec![],
                stack_alignment: 0,
                red_zone_bytes: 0,
            },
            cache: CacheHierarchy {
                l1d_bytes: 16384,
                l1i_bytes: 16384,
                l2_bytes: 0,
                l3_bytes: infinity_mb * 1024 * 1024,
                cacheline_bytes: 64,
                tmem_bytes: 0,
                smem_bytes: 0,
                lds_bytes: lds_kb * 1024,
            },
            features,
            k_unroll_factor: if gfx_arch >= 940 { 8 } else { 4 },
            dot_cap: crate::dispatch::device_profile::DotProductCap::NativeBf16,
        }
    }

    /// 构造 AArch64 IsaProfile。
    pub fn aarch64(
        has_sve: bool,
        has_sve2: bool,
        sve_vl: usize,
        has_sme: bool,
        has_sme2: bool,
        has_bf16: bool,
    ) -> Self {
        let mut features = vec![IsaFeature::Fma];
        if has_bf16 {
            features.push(IsaFeature::ArmBf16);
            features.push(IsaFeature::NativeBf16);
        }
        features.push(IsaFeature::ArmDotProd);
        if has_sve {
            features.push(IsaFeature::ScalableVector { min_vl: 16, max_vl: sve_vl });
            features.push(IsaFeature::PredicatedExec);
        }
        if has_sve2 {
            features.push(IsaFeature::Sve2);
        }
        if has_sme {
            features.push(IsaFeature::SmeTileOp);
            features.push(IsaFeature::TileGemm { m: 16, n: 16, k: 16 });
        }
        if has_sme2 {
            features.push(IsaFeature::Sme2MultiVec);
            features.push(IsaFeature::SmeF16F16);
        }

        // AArch64: x0-x30 (31 GPR), v0-v31 (32 NEON/SVE)
        // ARCH-ISA-SCRATCH: x16(ip0), x17(ip1) = ABI 约定的 intra-procedure-call scratch
        let scratch_gprs: Vec<PhysGpr> = vec![PhysGpr(16), PhysGpr(17)];
        let gpr_regs: Vec<PhysGpr> = (0..31u8)
            .filter(|&r| r != 29 && r != 30 && r != 31 && r != 16 && r != 17)
            .map(PhysGpr)
            .collect();
        // ARCH-ISA-SCRATCH-VEC: v16-v23 保留为 transcendental scratch (emit_neon_exp/sve_exp 使用)
        //                       v29-v31 保留为通用 Lower scratch (spill/reduce/broadcast)
        // 共 11 个，RegAllocator 剩余 21 个可分配（仍充裕）
        let scratch_vec_regs: Vec<PhysVec> = (16u8..=23).chain(29u8..=31)
            .map(PhysVec)
            .collect();
        let vec_regs: Vec<PhysVec> = (0..32)
            .filter(|&r| !scratch_vec_regs.iter().any(|s| s.0 == r))
            .map(PhysVec)
            .collect();
        let tile_regs = if has_sme { (0..4).map(PhysTile).collect() } else { vec![] };
        // ARCH-ISA-SCRATCH-MASK: SVE p7 保留为 "all-true predicate" (emit_sve_exp 使用)
        let mask_regs = if has_sve {
            (0..7).map(PhysMask).collect() // p0-p6 (p7 保留)
        } else {
            vec![]
        };

        Self {
            platform: Platform::AArch64 {
                has_bf16,
                has_dotprod: true,
                has_i8mm: has_sve2,
                has_sve,
                has_sve2,
                sve_vl,
                has_sme,
                has_sme2,
                has_sme_f16f16: has_sme2,
                has_sme_i16i64: has_sme2,
                sme_vl: if has_sme { sve_vl } else { 0 },
            },
            gpr_regs,
            scratch_gprs,
            vec_regs,
            scratch_vec_regs,
            tile_regs,
            mask_regs,
            abi: AbiConvention {
                arg_regs: (0..8u8).map(PhysGpr).collect(), // x0-x7
                stack_arg_offset: 0,
                callee_saved: (19..=28).map(PhysGpr).collect(), // x19-x28
                caller_saved: (0..=18).map(PhysGpr).collect(),  // x0-x18
                callee_saved_vec: (8..=15).map(PhysVec).collect(), // v8-v15
                stack_alignment: 16,
                red_zone_bytes: 0,
            },
            cache: CacheHierarchy {
                l1d_bytes: 65536,    // 典型 64KB
                l1i_bytes: 65536,
                l2_bytes: 524288,    // 典型 512KB
                l3_bytes: 8388608,   // 典型 8MB
                cacheline_bytes: 64,
                tmem_bytes: 0,
                smem_bytes: 0,
                lds_bytes: 0,
            },
            features,
            k_unroll_factor: if has_sme { 8 } else if has_sve { 4 } else { 4 },
            dot_cap: if has_sve2 {
                crate::dispatch::device_profile::DotProductCap::NativeInt8Simd
            } else {
                crate::dispatch::device_profile::DotProductCap::SimdAssisted
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::DeviceProfile;

    #[test]
    fn test_x86_profile_from_device() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);

        // ARCH-ISA-SCRATCH: scratch_gprs 与 allocatable gpr_regs 必须互斥。
        // 总数守恒: allocatable + scratch + frame(rsp/rbp) = max_gpr (16 基线 / 31 APX)
        let allocatable: std::collections::HashSet<u8> =
            profile.gpr_regs.iter().map(|r| r.0).collect();
        let scratch: std::collections::HashSet<u8> =
            profile.scratch_gprs.iter().map(|r| r.0).collect();
        assert!(allocatable.is_disjoint(&scratch),
            "ISA scratch 与可分配池重叠: alloc={:?} scratch={:?}", allocatable, scratch);
        // rsp=4, rbp=5 不能出现在两者中
        for frame in [4u8, 5u8] {
            assert!(!allocatable.contains(&frame), "rsp/rbp 不能进入可分配池");
            assert!(!scratch.contains(&frame), "rsp/rbp 不能作为 scratch");
        }
        let max_gpr = if matches!(&profile.platform, Platform::X86_64 { has_apx: true, .. }) { 31 } else { 16 };
        assert_eq!(profile.gpr_regs.len() + profile.scratch_gprs.len() + 2, max_gpr,
            "GPR 守恒: allocatable + scratch + {{rsp,rbp}} = {}", max_gpr);
        // GEMM 微核 & 融合算子对可分配 GPR 数有最小预算: 基线 ≥11 (16-2-3 scratch),
        // APX ≥26 (31-2-3 scratch)。若 scratch 集合过度膨胀会破坏此保证。
        let min_allocatable = if matches!(&profile.platform, Platform::X86_64 { has_apx: true, .. }) { 26 } else { 11 };
        assert!(profile.gpr_regs.len() >= min_allocatable,
            "可分配 GPR 不足: {} < {}, scratch 集合可能过度膨胀", profile.gpr_regs.len(), min_allocatable);

        // ARCH-ISA-SCRATCH-VEC: vec_regs + scratch_vec_regs = vec_count (16 基线 / 32 AVX-512)。
        // scratch 末 6 条 ymm/zmm: 3 内部 scratch (HReduce/FWHT/broadcast) + 3 spill scratch (a/b/dst)
        assert_eq!(profile.scratch_vec_regs.len(), 6);
        let vec_count = if matches!(&profile.platform, Platform::X86_64 { has_avx512: true, .. }) { 32 } else { 16 };
        assert_eq!(profile.vec_regs.len() + profile.scratch_vec_regs.len(), vec_count,
            "Vec 守恒: allocatable + scratch = {}", vec_count);

        // ABI 常量 (SysV x86-64, 与 CPU 无关)
        assert_eq!(profile.abi.arg_regs.len(), 6);
        assert_eq!(profile.abi.callee_saved.len(), 5);
        assert_eq!(profile.abi.stack_alignment, 16);
        assert_eq!(profile.abi.red_zone_bytes, 128);
        // x86_64 必有 FMA (SSE 家族中 FMA3 自 Haswell 后成为基线)
        assert!(profile.has_feature(&IsaFeature::Fma));
        assert!(profile.cache.l1d_bytes > 0);
    }

    #[test]
    fn test_optimal_simd_width() {
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);
        let width = profile.optimal_simd_width();
        assert!(width.f32_lanes() >= 8);
    }

    #[test]
    fn test_cuda_sm90_profile() {
        let profile = IsaProfile::cuda(90);
        assert!(profile.has_feature(&IsaFeature::Wgmma));
        assert!(profile.has_feature(&IsaFeature::Tma));
        assert!(profile.has_feature(&IsaFeature::WarpSpecialization));
        assert!(profile.has_feature(&IsaFeature::CudaBarrier));
        assert!(profile.has_feature(&IsaFeature::NativeFp8));
        assert!(!profile.has_feature(&IsaFeature::Tmem)); // Hopper 无 TMEM
        assert!(!profile.has_feature(&IsaFeature::BlockScaled));
    }

    #[test]
    fn test_cuda_sm100_blackwell_profile() {
        let profile = IsaProfile::cuda(100);
        // Blackwell 继承 Hopper 全部特性
        assert!(profile.has_feature(&IsaFeature::Wgmma));
        assert!(profile.has_feature(&IsaFeature::Tma));
        // Blackwell 专有
        assert!(profile.has_feature(&IsaFeature::Tmem));
        assert!(profile.has_feature(&IsaFeature::BlockScaled));
        assert!(profile.has_feature(&IsaFeature::NativeFp4));
        assert!(profile.has_feature(&IsaFeature::NativeFp6));
        assert!(profile.has_feature(&IsaFeature::ThreadBlockCluster));
        assert!(profile.has_feature(&IsaFeature::TwoCta));
        // 验证 TMEM 大小
        assert_eq!(profile.cache.tmem_bytes, 256 * 1024);
    }

    #[test]
    fn test_hip_gfx950_cdna4_profile() {
        let profile = IsaProfile::hip(950);
        assert!(profile.has_feature(&IsaFeature::Mfma));
        assert!(profile.has_feature(&IsaFeature::MfmaV2));
        assert!(profile.has_feature(&IsaFeature::Fp8Mfma));
        assert!(profile.has_feature(&IsaFeature::Fp4Mfma));
        assert!(profile.has_feature(&IsaFeature::NativeFp8));
        assert!(profile.has_feature(&IsaFeature::NativeFp4));
        assert!(profile.has_feature(&IsaFeature::XcdTopology));
        // wave64
        match &profile.platform {
            Platform::Hip { wave_size, .. } => assert_eq!(*wave_size, 64),
            _ => panic!("expected Hip platform"),
        }
        // LDS 128KB
        assert_eq!(profile.cache.lds_bytes, 128 * 1024);
    }

    #[test]
    fn test_hip_rdna3_profile() {
        let profile = IsaProfile::hip(1100);
        // RDNA3: wave32, 无 MFMA
        match &profile.platform {
            Platform::Hip { wave_size, has_mfma, .. } => {
                assert_eq!(*wave_size, 32);
                assert!(!has_mfma);
            }
            _ => panic!("expected Hip platform"),
        }
    }

    #[test]
    fn test_aarch64_sme2_profile() {
        let profile = IsaProfile::aarch64(true, true, 64, true, true, true);
        assert!(profile.has_feature(&IsaFeature::Sve2));
        assert!(profile.has_feature(&IsaFeature::Sme2MultiVec));
        assert!(profile.has_feature(&IsaFeature::SmeTileOp));
        assert!(profile.has_feature(&IsaFeature::SmeF16F16));
        assert!(profile.has_feature(&IsaFeature::ArmBf16));

        // ARCH-ISA-SCRATCH: scratch_gprs 与 allocatable gpr_regs 必须互斥。
        // AArch64 GPR: x0..x30 (31 个),frame 保留 {x29=fp, x30=lr}。
        // allocatable + scratch + {fp,lr} = 31
        let allocatable: std::collections::HashSet<u8> =
            profile.gpr_regs.iter().map(|r| r.0).collect();
        let scratch: std::collections::HashSet<u8> =
            profile.scratch_gprs.iter().map(|r| r.0).collect();
        assert!(allocatable.is_disjoint(&scratch),
            "AArch64 scratch 与可分配池重叠: alloc={:?} scratch={:?}", allocatable, scratch);
        for frame in [29u8, 30u8] {
            assert!(!allocatable.contains(&frame), "x29/x30 不能进入可分配池");
            assert!(!scratch.contains(&frame), "x29/x30 不能作为 scratch");
        }
        assert_eq!(profile.gpr_regs.len() + profile.scratch_gprs.len() + 2, 31,
            "AArch64 GPR 守恒: allocatable + scratch + {{fp,lr}} = 31");

        // ARCH-ISA-SCRATCH-VEC: v0..v31 (32 个) = allocatable + scratch。
        let vec_alloc: std::collections::HashSet<u8> =
            profile.vec_regs.iter().map(|r| r.0).collect();
        let vec_scratch: std::collections::HashSet<u8> =
            profile.scratch_vec_regs.iter().map(|r| r.0).collect();
        assert!(vec_alloc.is_disjoint(&vec_scratch),
            "AArch64 vec scratch 与可分配池重叠");
        assert_eq!(profile.vec_regs.len() + profile.scratch_vec_regs.len(), 32,
            "AArch64 Vec 守恒: allocatable + scratch = 32");

        // SVE predicate masks: 总数 8 (p0..p7),p7 保留 → 可分配 7
        assert_eq!(profile.mask_regs.len(), 7);
        // SME tiles 必须非空
        assert!(!profile.tile_regs.is_empty());
    }

    #[test]
    fn test_platform_helpers() {
        let cuda = IsaProfile::cuda(90);
        assert_eq!(cuda.platform.cuda_sm(), Some(90));
        assert!(cuda.platform.is_gpu());

        let hip = IsaProfile::hip(950);
        assert_eq!(hip.platform.hip_gfx(), Some(950));
        assert!(hip.platform.is_gpu());

        let dp = DeviceProfile::detect();
        let x86 = IsaProfile::from_device_profile(&dp);
        assert!(!x86.platform.is_gpu());
    }

    // ── New tests: struct constructors, enum variants, derives, edge cases ──

    #[test]
    fn test_phys_reg_variants_and_equality() {
        // Arrange: construct each PhysReg variant
        let gpr = PhysReg::Gpr(PhysGpr(0));
        let vec = PhysReg::Vec(PhysVec(1));
        let tile = PhysReg::Tile(PhysTile(2));
        let mask = PhysReg::Mask(PhysMask(3));
        let spilled = PhysReg::Spilled(42);

        // Act & Assert: distinct discriminants, reflexivity, Copy
        assert_ne!(gpr, vec);
        assert_ne!(vec, tile);
        assert_ne!(tile, mask);
        assert_ne!(mask, spilled);
        assert_eq!(gpr, PhysReg::Gpr(PhysGpr(0)));
        assert_eq!(spilled, PhysReg::Spilled(42));

        // Copy semantics: assignment copies, no move
        let spilled2 = spilled;
        assert_eq!(spilled, spilled2);
    }

    #[test]
    fn test_phys_register_tuple_newtypes_copy_and_hash() {
        // Arrange
        let gpr0 = PhysGpr(0);
        let gpr0_copy = gpr0;
        let gpr1 = PhysGpr(1);
        let vec0 = PhysVec(0);
        let tile0 = PhysTile(0);
        let mask0 = PhysMask(0);

        // Act & Assert: Copy + PartialEq + Eq + Hash
        assert_eq!(gpr0, gpr0_copy);
        assert_ne!(gpr0, gpr1);
        assert_eq!(vec0, PhysVec(0));
        assert_eq!(tile0, PhysTile(0));
        assert_eq!(mask0, PhysMask(0));

        // Hash consistency: equal values must produce equal hashes
        use std::collections::HashSet;
        let set: HashSet<PhysGpr> = [gpr0, gpr0_copy, gpr1].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_phys_reg_boundary_values() {
        // Arrange: u8 boundary values for PhysGpr/PhysVec/PhysTile/PhysMask
        let gpr_max = PhysGpr(u8::MAX);
        let vec_max = PhysVec(u8::MAX);
        let tile_min = PhysTile(0);
        let mask_min = PhysMask(0);
        let spilled_max = PhysReg::Spilled(u32::MAX);
        let spilled_zero = PhysReg::Spilled(0);

        // Act & Assert
        assert_eq!(gpr_max.0, 255);
        assert_eq!(vec_max.0, 255);
        assert_eq!(tile_min.0, 0);
        assert_eq!(mask_min.0, 0);
        assert_ne!(spilled_max, spilled_zero);
    }

    #[test]
    fn test_platform_metal_construction_and_fields() {
        // Arrange & Act: construct a Metal platform profile via IsaProfile fields
        let profile = IsaProfile {
            platform: Platform::Metal {
                simd_width: 32,
                gpu_family: 9,
                has_simdgroup_matrix: true,
                threadgroup_mem_kb: 32,
            },
            gpr_regs: vec![],
            scratch_gprs: vec![],
            vec_regs: vec![],
            scratch_vec_regs: vec![],
            tile_regs: vec![],
            mask_regs: vec![],
            abi: AbiConvention {
                arg_regs: vec![],
                stack_arg_offset: 0,
                callee_saved: vec![],
                caller_saved: vec![],
                callee_saved_vec: vec![],
                stack_alignment: 0,
                red_zone_bytes: 0,
            },
            cache: CacheHierarchy {
                l1d_bytes: 0,
                l1i_bytes: 0,
                l2_bytes: 0,
                l3_bytes: 0,
                cacheline_bytes: 128,
                tmem_bytes: 0,
                smem_bytes: 32 * 1024,
                lds_bytes: 0,
            },
            features: vec![IsaFeature::Fma, IsaFeature::SiMDGroupMatrix],
            k_unroll_factor: 4,
            dot_cap: crate::dispatch::device_profile::DotProductCap::SimdAssisted,
        };

        // Assert
        assert!(profile.platform.is_gpu());
        assert_eq!(profile.platform.cuda_sm(), None);
        assert_eq!(profile.platform.hip_gfx(), None);
        assert!(profile.has_feature(&IsaFeature::SiMDGroupMatrix));
        assert_eq!(profile.cache.smem_bytes, 32 * 1024);
    }

    #[test]
    fn test_abi_construction_and_field_access() {
        // Arrange: construct AbiConvention for a hypothetical calling convention
        let abi = AbiConvention {
            arg_regs: vec![PhysGpr(0), PhysGpr(1), PhysGpr(2)],
            stack_arg_offset: 8,
            callee_saved: vec![PhysGpr(5), PhysGpr(6)],
            caller_saved: vec![PhysGpr(3), PhysGpr(4)],
            callee_saved_vec: vec![PhysVec(8), PhysVec(9)],
            stack_alignment: 32,
            red_zone_bytes: 0,
        };

        // Act & Assert
        assert_eq!(abi.arg_regs.len(), 3);
        assert_eq!(abi.stack_arg_offset, 8);
        assert_eq!(abi.callee_saved.len(), 2);
        assert_eq!(abi.caller_saved.len(), 2);
        assert_eq!(abi.callee_saved_vec.len(), 2);
        assert_eq!(abi.stack_alignment, 32);
        assert_eq!(abi.red_zone_bytes, 0);

        // Verify Debug trait produces non-empty output
        let debug_str = format!("{:?}", abi);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_cache_hierarchy_construction_and_zero_fields() {
        // Arrange: GPU-style CacheHierarchy where some fields are zero
        let cache = CacheHierarchy {
            l1d_bytes: 16384,
            l1i_bytes: 8192,
            l2_bytes: 0,
            l3_bytes: 0,
            cacheline_bytes: 128,
            tmem_bytes: 256 * 1024,
            smem_bytes: 228 * 1024,
            lds_bytes: 0,
        };

        // Act & Assert
        assert_eq!(cache.l1d_bytes, 16384);
        assert_eq!(cache.l2_bytes, 0);
        assert_eq!(cache.l3_bytes, 0);
        assert_eq!(cache.tmem_bytes, 256 * 1024);
        assert_eq!(cache.smem_bytes, 228 * 1024);
        assert_eq!(cache.lds_bytes, 0);

        // Verify Clone derive
        let cloned = cache.clone();
        assert_eq!(cloned.l1d_bytes, cache.l1d_bytes);
        assert_eq!(cloned.tmem_bytes, cache.tmem_bytes);
    }

    #[test]
    fn test_isa_feature_equality_and_tile_gemm_discriminant() {
        // Arrange: IsaFeature derives PartialEq, Eq
        assert_eq!(IsaFeature::Fma, IsaFeature::Fma);
        assert_ne!(IsaFeature::Fma, IsaFeature::NativeBf16);
        assert_eq!(IsaFeature::TileGemm { m: 16, n: 16, k: 16 }, IsaFeature::TileGemm { m: 16, n: 16, k: 16 });

        // TileGemm with different parameters is still equal (same discriminant used by has_feature)
        assert_ne!(
            IsaFeature::TileGemm { m: 16, n: 16, k: 16 },
            IsaFeature::TileGemm { m: 8, n: 8, k: 8 }
        );

        // Act: has_feature uses discriminant matching, so different k values still match
        let profile = IsaProfile::cuda(80);
        assert!(profile.has_feature(&IsaFeature::TileGemm { m: 8, n: 8, k: 8 }));
    }

    #[test]
    fn test_cuda_sm70_volta_minimal_profile() {
        // Arrange & Act
        let profile = IsaProfile::cuda(70);

        // Assert: SM70 baseline features
        assert!(profile.has_feature(&IsaFeature::Fma));
        assert!(profile.has_feature(&IsaFeature::WarpShuffle));
        assert!(profile.has_feature(&IsaFeature::TileGemm { m: 16, n: 16, k: 16 }));
        // SM70 lacks async copy and BF16
        assert!(!profile.has_feature(&IsaFeature::AsyncCopy));
        assert!(!profile.has_feature(&IsaFeature::NativeBf16));
        assert!(!profile.has_feature(&IsaFeature::Wgmma));
        assert!(!profile.has_feature(&IsaFeature::NativeFp8));
        // Platform field checks
        assert_eq!(profile.platform.cuda_sm(), Some(70));
        assert!(profile.platform.is_gpu());
        // SM70 shared memory should be 96 KB
        match &profile.platform {
            Platform::Cuda { shared_mem_kb, .. } => assert_eq!(*shared_mem_kb, 96),
            _ => panic!("expected Cuda platform"),
        }
        assert_eq!(profile.cache.smem_bytes, 96 * 1024);
    }

    #[test]
    fn test_cuda_sm80_ampere_profile() {
        // Arrange & Act
        let profile = IsaProfile::cuda(80);

        // Assert: SM80 adds async copy + BF16
        assert!(profile.has_feature(&IsaFeature::AsyncCopy));
        assert!(profile.has_feature(&IsaFeature::NativeBf16));
        assert!(profile.has_feature(&IsaFeature::TileGemm { m: 16, n: 16, k: 16 }));
        // SM80 lacks Hopper features
        assert!(!profile.has_feature(&IsaFeature::Wgmma));
        assert!(!profile.has_feature(&IsaFeature::Tma));
        assert!(!profile.has_feature(&IsaFeature::NativeFp8));
        assert!(!profile.has_feature(&IsaFeature::Tmem));
        // SM80 shared memory
        match &profile.platform {
            Platform::Cuda { shared_mem_kb, .. } => assert_eq!(*shared_mem_kb, 164),
            _ => panic!("expected Cuda platform"),
        }
        assert!(!profile.has_feature(&IsaFeature::CudaBarrier));
    }

    #[test]
    fn test_hip_gfx908_cdna2_profile() {
        // Arrange & Act
        let profile = IsaProfile::hip(908);

        // Assert: CDNA2 has MFMA v1 but not v2
        assert!(profile.has_feature(&IsaFeature::Mfma));
        assert!(!profile.has_feature(&IsaFeature::MfmaV2));
        assert!(!profile.has_feature(&IsaFeature::Fp8Mfma));
        assert!(!profile.has_feature(&IsaFeature::Fp4Mfma));
        assert!(!profile.has_feature(&IsaFeature::XcdTopology));
        // wave64 for CDNA
        match &profile.platform {
            Platform::Hip { wave_size, gfx_arch, .. } => {
                assert_eq!(*wave_size, 64);
                assert_eq!(*gfx_arch, 908);
            }
            _ => panic!("expected Hip platform"),
        }
        // LDS 64KB for CDNA2
        assert_eq!(profile.cache.lds_bytes, 64 * 1024);
    }

    #[test]
    fn test_hip_gfx942_cdna3_profile() {
        // Arrange & Act
        let profile = IsaProfile::hip(942);

        // Assert: CDNA3 has MFMA v1 + XCD topology, but not v2
        assert!(profile.has_feature(&IsaFeature::Mfma));
        assert!(!profile.has_feature(&IsaFeature::MfmaV2));
        assert!(profile.has_feature(&IsaFeature::XcdTopology));
        // wave64
        match &profile.platform {
            Platform::Hip { wave_size, gfx_arch, vgpr_per_cu, .. } => {
                assert_eq!(*wave_size, 64);
                assert_eq!(*gfx_arch, 942);
                assert_eq!(*vgpr_per_cu, 512);
            }
            _ => panic!("expected Hip platform"),
        }
    }

    #[test]
    fn test_aarch64_baseline_neon_only() {
        // Arrange & Act: no SVE, no SME, no BF16
        let profile = IsaProfile::aarch64(false, false, 16, false, false, false);

        // Assert: baseline NEON has only FMA + DotProd
        assert!(profile.has_feature(&IsaFeature::Fma));
        assert!(!profile.has_feature(&IsaFeature::Sve2));
        assert!(!profile.has_feature(&IsaFeature::SmeTileOp));
        assert!(!profile.has_feature(&IsaFeature::ArmBf16));
        assert!(!profile.has_feature(&IsaFeature::NativeBf16));
        assert!(!profile.has_feature(&IsaFeature::PredicatedExec));
        // No tiles, no masks
        assert!(profile.tile_regs.is_empty());
        assert!(profile.mask_regs.is_empty());
        // NEON has 32 vec regs, 11 scratch, 21 allocatable
        assert_eq!(profile.scratch_vec_regs.len(), 11);
        assert_eq!(profile.vec_regs.len(), 21);
    }

    #[test]
    fn test_isa_profile_struct_update_syntax() {
        // Arrange: create a CUDA profile, then use struct update to modify only k_unroll_factor
        let base = IsaProfile::cuda(90);
        let original_k = base.k_unroll_factor;

        // Act: struct update syntax copies all fields except the specified ones
        let modified = IsaProfile { k_unroll_factor: 1, ..base.clone() };

        // Assert
        assert_eq!(modified.k_unroll_factor, 1);
        assert_eq!(original_k, 8); // SM90 default
        assert_eq!(modified.platform.cuda_sm(), base.platform.cuda_sm());
        assert_eq!(modified.features.len(), base.features.len());
        assert_eq!(modified.cache.tmem_bytes, base.cache.tmem_bytes);
    }

    #[test]
    fn test_optimal_simd_width_across_platforms() {
        // Arrange: profiles for each platform
        let cuda = IsaProfile::cuda(90);
        let hip = IsaProfile::hip(950);
        let aarch64_neon = IsaProfile::aarch64(false, false, 16, false, false, false);
        let aarch64_sve = IsaProfile::aarch64(true, true, 64, true, true, false);

        // Act & Assert: CUDA warp=32, HIP wave=64
        match cuda.optimal_simd_width() {
            super::super::instr::SimdWidth::Warp(w) => assert_eq!(w, 32),
            other => panic!("expected Warp(32), got {:?}", other),
        }
        match hip.optimal_simd_width() {
            super::super::instr::SimdWidth::Warp(w) => assert_eq!(w, 64),
            other => panic!("expected Warp(64), got {:?}", other),
        }
        // AArch64 NEON baseline = W128
        assert_eq!(aarch64_neon.optimal_simd_width().f32_lanes(), 4); // 128/32=4
        // AArch64 SVE = Scalable
        match aarch64_sve.optimal_simd_width() {
            super::super::instr::SimdWidth::Scalable => {}
            other => panic!("expected Scalable for SVE, got {:?}", other),
        }
    }

    // ── 13 New Tests ──

    #[test]
    fn test_cuda_sm89_ada_fp8_profile() {
        // Arrange & Act: SM89 Ada Lovelace — first generation with FP8, but no Hopper features
        let profile = IsaProfile::cuda(89);

        // Assert: Ada inherits Ampere async copy + BF16, adds FP8
        assert!(profile.has_feature(&IsaFeature::Fma));
        assert!(profile.has_feature(&IsaFeature::AsyncCopy));
        assert!(profile.has_feature(&IsaFeature::NativeBf16));
        assert!(profile.has_feature(&IsaFeature::NativeFp8));
        // Ada lacks Hopper WGMMA/TMA/barrier
        assert!(!profile.has_feature(&IsaFeature::Wgmma));
        assert!(!profile.has_feature(&IsaFeature::Tma));
        assert!(!profile.has_feature(&IsaFeature::CudaBarrier));
        assert!(!profile.has_feature(&IsaFeature::L2Multicast));
        // Ada lacks Blackwell TMEM/FP4/FP6
        assert!(!profile.has_feature(&IsaFeature::Tmem));
        assert!(!profile.has_feature(&IsaFeature::NativeFp4));
        // SM89 shared memory = 164 KB (same as SM80)
        match &profile.platform {
            Platform::Cuda { shared_mem_kb, .. } => assert_eq!(*shared_mem_kb, 164),
            _ => panic!("expected Cuda platform"),
        }
        // k_unroll_factor for SM89 (< SM90) = 4
        assert_eq!(profile.k_unroll_factor, 4);
    }

    #[test]
    fn test_aarch64_sve_without_sme_no_tiles() {
        // Arrange & Act: SVE2 enabled but no SME — should have predication but no tile regs
        let profile = IsaProfile::aarch64(true, true, 32, false, false, true);

        // Assert: SVE2 features present
        assert!(profile.has_feature(&IsaFeature::Sve2));
        assert!(profile.has_feature(&IsaFeature::PredicatedExec));
        assert!(profile.has_feature(&IsaFeature::ScalableVector { min_vl: 16, max_vl: 32 }));
        assert!(profile.has_feature(&IsaFeature::ArmBf16));
        assert!(profile.has_feature(&IsaFeature::NativeBf16));
        // No SME means no tile ops, no multi-vec
        assert!(!profile.has_feature(&IsaFeature::SmeTileOp));
        assert!(!profile.has_feature(&IsaFeature::Sme2MultiVec));
        assert!(!profile.has_feature(&IsaFeature::SmeF16F16));
        // Tile regs empty, but SVE masks present (p0-p6, p7 reserved)
        assert!(profile.tile_regs.is_empty());
        assert_eq!(profile.mask_regs.len(), 7);
        // DotProd always present on AArch64
        assert!(profile.has_feature(&IsaFeature::ArmDotProd));
        // dot_cap for SVE2 = NativeInt8Simd
        assert!(matches!(profile.dot_cap, crate::dispatch::device_profile::DotProductCap::NativeInt8Simd));
    }

    #[test]
    fn test_cuda_platform_field_consistency() {
        // Arrange & Act: SM100 Blackwell — verify all Cuda platform boolean fields
        let profile = IsaProfile::cuda(100);

        match &profile.platform {
            Platform::Cuda {
                sm_version, warp_size, shared_mem_kb, reg_file_per_sm, max_regs_per_thread,
                has_wgmma, has_tma, has_warp_spec, has_fp8,
                has_tmem, has_block_scaled, has_native_fp4, has_native_fp6,
                has_cluster, has_2cta_mma, tmem_size_kb,
            } => {
                // Assert: numeric fields
                assert_eq!(*sm_version, 100);
                assert_eq!(*warp_size, 32);
                assert_eq!(*shared_mem_kb, 228);
                assert_eq!(*reg_file_per_sm, 65536);
                assert_eq!(*max_regs_per_thread, 255);
                assert_eq!(*tmem_size_kb, 256);
                // Hopper features inherited
                assert!(has_wgmma);
                assert!(has_tma);
                assert!(has_warp_spec);
                assert!(has_fp8);
                // Blackwell-exclusive features
                assert!(has_tmem);
                assert!(has_block_scaled);
                assert!(has_native_fp4);
                assert!(has_native_fp6);
                assert!(has_cluster);
                assert!(has_2cta_mma);
            }
            _ => panic!("expected Cuda platform"),
        }
    }

    #[test]
    fn test_hip_platform_gfx908_cache_hierarchy() {
        // Arrange & Act
        let profile = IsaProfile::hip(908);

        // Assert: CDNA2 cache hierarchy
        assert_eq!(profile.cache.l1d_bytes, 16384);
        assert_eq!(profile.cache.l1i_bytes, 16384);
        assert_eq!(profile.cache.cacheline_bytes, 64);
        assert_eq!(profile.cache.lds_bytes, 64 * 1024);
        // CDNA2 has no Infinity Cache
        assert_eq!(profile.cache.l3_bytes, 0);
        // TMEM/SMEM are GPU-specific, not HIP
        assert_eq!(profile.cache.tmem_bytes, 0);
        assert_eq!(profile.cache.smem_bytes, 0);
        // GPU profiles have empty GPR/vec regs
        assert!(profile.gpr_regs.is_empty());
        assert!(profile.vec_regs.is_empty());
        assert!(profile.scratch_gprs.is_empty());
        assert!(profile.scratch_vec_regs.is_empty());
        assert!(profile.tile_regs.is_empty());
        assert!(profile.mask_regs.is_empty());
    }

    #[test]
    fn test_aarch64_abi_convention_fields() {
        // Arrange & Act: AArch64 with full SVE2+SME2
        let profile = IsaProfile::aarch64(true, true, 64, true, true, true);

        // Assert: AArch64 ABI — x0..x7 arg regs, x19..x28 callee-saved, 16-byte alignment
        assert_eq!(profile.abi.arg_regs.len(), 8);
        assert_eq!(profile.abi.arg_regs[0], PhysGpr(0));
        assert_eq!(profile.abi.arg_regs[7], PhysGpr(7));
        assert_eq!(profile.abi.stack_arg_offset, 0);
        assert_eq!(profile.abi.callee_saved.len(), 10); // x19..x28
        assert_eq!(profile.abi.callee_saved[0], PhysGpr(19));
        assert_eq!(profile.abi.callee_saved[9], PhysGpr(28));
        assert_eq!(profile.abi.caller_saved.len(), 19); // x0..x18
        assert_eq!(profile.abi.callee_saved_vec.len(), 8); // v8..v15
        assert_eq!(profile.abi.callee_saved_vec[0], PhysVec(8));
        assert_eq!(profile.abi.callee_saved_vec[7], PhysVec(15));
        assert_eq!(profile.abi.stack_alignment, 16);
        assert_eq!(profile.abi.red_zone_bytes, 0); // AArch64 has no red zone in AAPCS
    }

    #[test]
    fn test_cuda_k_unroll_factor_gradient() {
        // Arrange & Act: k_unroll_factor increases at SM90 boundary
        let sm80 = IsaProfile::cuda(80);
        let sm89 = IsaProfile::cuda(89);
        let sm90 = IsaProfile::cuda(90);
        let sm100 = IsaProfile::cuda(100);

        // Assert: pre-Hopper = 4, Hopper+ = 8
        assert_eq!(sm80.k_unroll_factor, 4);
        assert_eq!(sm89.k_unroll_factor, 4);
        assert_eq!(sm90.k_unroll_factor, 8);
        assert_eq!(sm100.k_unroll_factor, 8);
    }

    #[test]
    fn test_hip_k_unroll_factor_and_dot_cap() {
        // Arrange & Act: CDNA2 vs CDNA3 vs CDNA4
        let cdna2 = IsaProfile::hip(908);
        let cdna3 = IsaProfile::hip(942);
        let cdna4 = IsaProfile::hip(950);

        // Assert: k_unroll_factor increases at gfx940
        assert_eq!(cdna2.k_unroll_factor, 4);
        assert_eq!(cdna3.k_unroll_factor, 8);
        assert_eq!(cdna4.k_unroll_factor, 8);
        // All HIP profiles use NativeBf16 dot_cap
        assert!(matches!(cdna2.dot_cap, crate::dispatch::device_profile::DotProductCap::NativeBf16));
        assert!(matches!(cdna3.dot_cap, crate::dispatch::device_profile::DotProductCap::NativeBf16));
        assert!(matches!(cdna4.dot_cap, crate::dispatch::device_profile::DotProductCap::NativeBf16));
    }

    #[test]
    fn test_isa_feature_has_feature_discriminant_semantics() {
        // Arrange: construct a profile with TileGemm { m: 16, n: 16, k: 16 }
        let profile = IsaProfile::cuda(70);

        // Act & Assert: has_feature uses discriminant, so TileGemm with any params matches
        assert!(profile.has_feature(&IsaFeature::TileGemm { m: 16, n: 16, k: 16 }));
        assert!(profile.has_feature(&IsaFeature::TileGemm { m: 99, n: 99, k: 99 }));
        // Non-matching discriminant returns false
        assert!(!profile.has_feature(&IsaFeature::Tmem));
        assert!(!profile.has_feature(&IsaFeature::SmeTileOp));
        // WarpShuffle present on all CUDA profiles
        assert!(profile.has_feature(&IsaFeature::WarpShuffle));
    }

    #[test]
    fn test_aarch64_k_unroll_factor_by_capability() {
        // Arrange & Act: three capability levels
        let neon = IsaProfile::aarch64(false, false, 16, false, false, false);
        let sve = IsaProfile::aarch64(true, false, 32, false, false, false);
        let sme = IsaProfile::aarch64(true, true, 64, true, true, false);

        // Assert: NEON = 4, SVE = 4, SME = 8
        assert_eq!(neon.k_unroll_factor, 4);
        assert_eq!(sve.k_unroll_factor, 4);
        assert_eq!(sme.k_unroll_factor, 8);
    }

    #[test]
    fn test_hip_infinity_cache_scaling() {
        // Arrange & Act: CDNA2 (no infinity), CDNA3 (128MB), CDNA4 (256MB)
        let cdna2 = IsaProfile::hip(908);
        let cdna3 = IsaProfile::hip(942);
        let cdna4 = IsaProfile::hip(950);

        // Assert: L3 = infinity_cache_mb * 1024 * 1024
        assert_eq!(cdna2.cache.l3_bytes, 0); // no infinity cache
        assert_eq!(cdna3.cache.l3_bytes, 128 * 1024 * 1024);
        assert_eq!(cdna4.cache.l3_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn test_cuda_sm100_cache_tmem_and_smem() {
        // Arrange & Act
        let profile = IsaProfile::cuda(100);

        // Assert: Blackwell has both TMEM and SMEM in cache hierarchy
        assert_eq!(profile.cache.tmem_bytes, 256 * 1024);
        assert_eq!(profile.cache.smem_bytes, 228 * 1024);
        assert_eq!(profile.cache.l1d_bytes, 228 * 1024); // l1d = smem for CUDA
        assert_eq!(profile.cache.cacheline_bytes, 128);
        // GPU L2/L3 managed by driver, reported as 0
        assert_eq!(profile.cache.l2_bytes, 0);
        assert_eq!(profile.cache.l3_bytes, 0);
    }

    #[test]
    fn test_aarch64_gpr_scratch_excludes_frame_and_ip() {
        // Arrange & Act: any AArch64 profile — frame regs and IP scratch must be excluded
        let profile = IsaProfile::aarch64(false, false, 16, false, false, false);

        let allocatable: std::collections::HashSet<u8> =
            profile.gpr_regs.iter().map(|r| r.0).collect();
        let scratch: std::collections::HashSet<u8> =
            profile.scratch_gprs.iter().map(|r| r.0).collect();

        // Assert: x16, x17 are scratch (IP0, IP1)
        assert!(scratch.contains(&16));
        assert!(scratch.contains(&17));
        // x29 (fp) and x30 (lr) must not appear in either pool
        for frame in [29u8, 30u8] {
            assert!(!allocatable.contains(&frame), "x{} must not be allocatable", frame);
            assert!(!scratch.contains(&frame), "x{} must not be scratch", frame);
        }
        // x31 (SP/ZR) must not appear in either pool
        assert!(!allocatable.contains(&31));
        assert!(!scratch.contains(&31));
        // Total: allocatable + scratch + {fp, lr} = 31 GPR (x0..x30)
        assert_eq!(profile.gpr_regs.len() + profile.scratch_gprs.len() + 2, 31);
    }

    #[test]
    fn test_metal_simd_width_is_warp() {
        // Arrange: construct a minimal Metal profile directly
        let profile = IsaProfile {
            platform: Platform::Metal {
                simd_width: 32,
                gpu_family: 10,
                has_simdgroup_matrix: true,
                threadgroup_mem_kb: 64,
            },
            gpr_regs: vec![],
            scratch_gprs: vec![],
            vec_regs: vec![],
            scratch_vec_regs: vec![],
            tile_regs: vec![],
            mask_regs: vec![],
            abi: AbiConvention {
                arg_regs: vec![],
                stack_arg_offset: 0,
                callee_saved: vec![],
                caller_saved: vec![],
                callee_saved_vec: vec![],
                stack_alignment: 0,
                red_zone_bytes: 0,
            },
            cache: CacheHierarchy {
                l1d_bytes: 0,
                l1i_bytes: 0,
                l2_bytes: 0,
                l3_bytes: 0,
                cacheline_bytes: 64,
                tmem_bytes: 0,
                smem_bytes: 64 * 1024,
                lds_bytes: 0,
            },
            features: vec![IsaFeature::Fma, IsaFeature::SiMDGroupMatrix],
            k_unroll_factor: 4,
            dot_cap: crate::dispatch::device_profile::DotProductCap::SimdAssisted,
        };

        // Act & Assert: Metal simd_width=32 maps to Warp(32)
        match profile.optimal_simd_width() {
            super::super::instr::SimdWidth::Warp(w) => assert_eq!(w, 32),
            other => panic!("expected Warp(32) for Metal, got {:?}", other),
        }
        assert!(profile.platform.is_gpu());
        assert_eq!(profile.platform.cuda_sm(), None);
        assert_eq!(profile.platform.hip_gfx(), None);
        assert!(profile.has_feature(&IsaFeature::SiMDGroupMatrix));
    }

    // ── 10 Additional Tests (wave-12k91) ──

    #[test]
    fn test_phys_reg_debug_format_non_empty() {
        // Arrange: one of each PhysReg variant
        let gpr = PhysReg::Gpr(PhysGpr(7));
        let vec = PhysReg::Vec(PhysVec(3));
        let tile = PhysReg::Tile(PhysTile(1));
        let mask = PhysReg::Mask(PhysMask(2));
        let spilled = PhysReg::Spilled(99);

        // Act: format with Debug trait
        // Assert: each Debug output is non-empty and contains expected content
        let gpr_dbg = format!("{:?}", gpr);
        let vec_dbg = format!("{:?}", vec);
        let tile_dbg = format!("{:?}", tile);
        let mask_dbg = format!("{:?}", mask);
        let spill_dbg = format!("{:?}", spilled);

        assert!(!gpr_dbg.is_empty(), "Gpr Debug must not be empty");
        assert!(!vec_dbg.is_empty(), "Vec Debug must not be empty");
        assert!(!tile_dbg.is_empty(), "Tile Debug must not be empty");
        assert!(!mask_dbg.is_empty(), "Mask Debug must not be empty");
        assert!(!spill_dbg.is_empty(), "Spilled Debug must not be empty");
        // Spilled variant contains the slot id
        assert!(spill_dbg.contains("99"), "Spilled Debug should contain slot id");
    }

    #[test]
    fn test_isa_feature_debug_all_major_variants() {
        // Arrange: representative IsaFeature variants from each category
        let features = [
            IsaFeature::Fma,
            IsaFeature::NativeBf16,
            IsaFeature::TileGemm { m: 16, n: 16, k: 32 },
            IsaFeature::AsyncCopy,
            IsaFeature::PredicatedExec,
            IsaFeature::ScalableVector { min_vl: 16, max_vl: 256 },
            IsaFeature::Wgmma,
            IsaFeature::Tmem,
            IsaFeature::Mfma,
            IsaFeature::Sve2,
            IsaFeature::SmeTileOp,
            IsaFeature::AmxFp16,
            IsaFeature::Vp2Intersect,
            IsaFeature::SiMDGroupMatrix,
        ];

        // Act & Assert: every variant must produce non-empty Debug output
        for feat in &features {
            let dbg = format!("{:?}", feat);
            assert!(!dbg.is_empty(), "IsaFeature {:?} Debug output is empty", feat);
        }

        // TileGemm and ScalableVector should contain their parameters
        let tile_dbg = format!("{:?}", IsaFeature::TileGemm { m: 16, n: 16, k: 32 });
        assert!(tile_dbg.contains("16"), "TileGemm Debug should contain m=16");
        let sv_dbg = format!("{:?}", IsaFeature::ScalableVector { min_vl: 16, max_vl: 256 });
        assert!(sv_dbg.contains("256"), "ScalableVector Debug should contain max_vl=256");
    }

    #[test]
    fn test_cache_hierarchy_clone_independence() {
        // Arrange
        let original = CacheHierarchy {
            l1d_bytes: 32768,
            l1i_bytes: 32768,
            l2_bytes: 262144,
            l3_bytes: 8388608,
            cacheline_bytes: 64,
            tmem_bytes: 0,
            smem_bytes: 0,
            lds_bytes: 0,
        };

        // Act: clone and mutate the clone
        let mut cloned = original.clone();
        cloned.l1d_bytes = 999;

        // Assert: original unaffected
        assert_eq!(original.l1d_bytes, 32768);
        assert_eq!(cloned.l1d_bytes, 999);
        assert_eq!(original.l3_bytes, cloned.l3_bytes);
    }

    #[test]
    fn test_abi_convention_clone_independence() {
        // Arrange
        let original = AbiConvention {
            arg_regs: vec![PhysGpr(0), PhysGpr(1)],
            stack_arg_offset: 16,
            callee_saved: vec![PhysGpr(3)],
            caller_saved: vec![PhysGpr(0)],
            callee_saved_vec: vec![PhysVec(8)],
            stack_alignment: 16,
            red_zone_bytes: 128,
        };

        // Act: clone and mutate
        let mut cloned = original.clone();
        cloned.arg_regs.push(PhysGpr(2));
        cloned.stack_alignment = 32;

        // Assert: original unaffected
        assert_eq!(original.arg_regs.len(), 2);
        assert_eq!(cloned.arg_regs.len(), 3);
        assert_eq!(original.stack_alignment, 16);
        assert_eq!(cloned.stack_alignment, 32);
        assert_eq!(original.red_zone_bytes, cloned.red_zone_bytes);
    }

    #[test]
    fn test_platform_is_gpu_false_for_cpu_variants() {
        // Arrange: x86 and AArch64 platforms
        let x86 = Platform::X86_64 {
            has_avx512: true, has_bf16: true, has_vnni: true, has_avx512fp16: false,
            has_amx: false, has_amx_fp16: false, has_amx_complex: false,
            has_amx_transpose: false, has_amx_fp8: false,
            has_avx10_2: false, has_apx: false, has_vp2intersect: false,
        };
        let aarch64 = Platform::AArch64 {
            has_bf16: false, has_dotprod: true, has_i8mm: false,
            has_sve: false, has_sve2: false, sve_vl: 16,
            has_sme: false, has_sme2: false, has_sme_f16f16: false,
            has_sme_i16i64: false, sme_vl: 0,
        };

        // Act & Assert
        assert!(!x86.is_gpu(), "X86_64 is not a GPU platform");
        assert!(!aarch64.is_gpu(), "AArch64 is not a GPU platform");
        assert_eq!(x86.cuda_sm(), None, "X86_64 has no SM version");
        assert_eq!(x86.hip_gfx(), None, "X86_64 has no GFX arch");
        assert_eq!(aarch64.cuda_sm(), None, "AArch64 has no SM version");
        assert_eq!(aarch64.hip_gfx(), None, "AArch64 has no GFX arch");
    }

    #[test]
    fn test_cuda_sm89_has_fp8_field_but_not_hopper() {
        // Arrange & Act: SM89 Ada — has_fp8=true, but no Hopper features
        let profile = IsaProfile::cuda(89);

        // Assert: platform boolean fields
        match &profile.platform {
            Platform::Cuda { has_fp8, has_wgmma, has_tma, has_tmem, .. } => {
                assert!(has_fp8, "SM89 must have FP8");
                assert!(!has_wgmma, "SM89 must not have WGMMA");
                assert!(!has_tma, "SM89 must not have TMA");
                assert!(!has_tmem, "SM89 must not have TMEM");
            }
            _ => panic!("expected Cuda platform"),
        }
    }

    #[test]
    fn test_hip_gfx950_platform_all_boolean_fields() {
        // Arrange & Act
        let profile = IsaProfile::hip(950);

        // Assert: CDNA4 gfx950 — all MFMA flags true
        match &profile.platform {
            Platform::Hip {
                gfx_arch, wave_size, has_mfma, has_mfma_v2,
                has_fp8_mfma, has_fp4_mfma, vgpr_per_cu, lds_size_kb,
                infinity_cache_mb,
            } => {
                assert_eq!(*gfx_arch, 950);
                assert_eq!(*wave_size, 64);
                assert!(has_mfma, "CDNA4 must have MFMA v1");
                assert!(has_mfma_v2, "CDNA4 must have MFMA v2");
                assert!(has_fp8_mfma, "CDNA4 must have FP8 MFMA");
                assert!(has_fp4_mfma, "CDNA4 must have FP4 MFMA");
                assert_eq!(*vgpr_per_cu, 512);
                assert_eq!(*lds_size_kb, 128);
                assert_eq!(*infinity_cache_mb, 256);
            }
            _ => panic!("expected Hip platform"),
        }
    }

    #[test]
    fn test_aarch64_neon_baseline_dot_cap_is_simd_assisted() {
        // Arrange & Act: pure NEON baseline (no SVE2)
        let profile = IsaProfile::aarch64(false, false, 16, false, false, false);

        // Assert: NEON without SVE2 → SimdAssisted dot_cap
        assert!(matches!(
            profile.dot_cap,
            crate::dispatch::device_profile::DotProductCap::SimdAssisted
        ), "Baseline NEON must use SimdAssisted dot_cap");
    }

    #[test]
    fn test_isa_profile_clone_deep_copy_independence() {
        // Arrange: CUDA profile with features
        let original = IsaProfile::cuda(90);
        let original_feature_count = original.features.len();
        let original_smem = original.cache.smem_bytes;

        // Act: clone and mutate
        let mut cloned = original.clone();
        cloned.features.push(IsaFeature::HardwareTranscendental);
        cloned.cache.smem_bytes = 0;

        // Assert: original unaffected
        assert_eq!(original.features.len(), original_feature_count);
        assert_eq!(cloned.features.len(), original_feature_count + 1);
        assert_eq!(original.cache.smem_bytes, original_smem);
        assert_eq!(cloned.cache.smem_bytes, 0);
        // Platform still the same
        assert_eq!(original.platform.cuda_sm(), cloned.platform.cuda_sm());
    }

    #[test]
    fn test_phys_newtype_hash_set_dedup_across_types() {
        // Arrange: collect mixed newtypes into individual HashSets
        use std::collections::HashSet;
        let gprs: HashSet<PhysGpr> = [PhysGpr(0), PhysGpr(1), PhysGpr(0), PhysGpr(1)].into_iter().collect();
        let vecs: HashSet<PhysVec> = [PhysVec(0), PhysVec(0), PhysVec(2)].into_iter().collect();
        let tiles: HashSet<PhysTile> = [PhysTile(0), PhysTile(1), PhysTile(0)].into_iter().collect();
        let masks: HashSet<PhysMask> = [PhysMask(0), PhysMask(0)].into_iter().collect();

        // Act & Assert: HashSet dedup reflects unique values
        assert_eq!(gprs.len(), 2, "PhysGpr dedup: {{0, 1}}");
        assert_eq!(vecs.len(), 2, "PhysVec dedup: {{0, 2}}");
        assert_eq!(tiles.len(), 2, "PhysTile dedup: {{0, 1}}");
        assert_eq!(masks.len(), 1, "PhysMask dedup: {{0}}");
    }

    // ── 10 Additional Tests (wave-12x34) ──

    #[test]
    fn test_has_feature_scalable_vector_discriminant_matching() {
        // Arrange: AArch64 SVE profile has ScalableVector { min_vl: 16, max_vl: 64 }
        let profile = IsaProfile::aarch64(true, false, 64, false, false, false);

        // Act & Assert: has_feature uses discriminant, so different max_vl still matches
        assert!(profile.has_feature(&IsaFeature::ScalableVector { min_vl: 16, max_vl: 64 }));
        assert!(profile.has_feature(&IsaFeature::ScalableVector { min_vl: 1, max_vl: 999 }));
        // But a different feature kind does not match
        assert!(!profile.has_feature(&IsaFeature::WarpShuffle));
    }

    #[test]
    fn test_hip_gfx1000_rdna4_k_unroll_and_features() {
        // Arrange & Act: gfx1000 — is_cdna=false (gfx_arch >= 1100), is_cdna4=true (gfx_arch >= 950)
        // This tests the boundary behavior where CDNA4 feature flags are applied
        // even though is_cdna is false.
        let profile = IsaProfile::hip(1000);

        // Assert: k_unroll_factor = 8 (gfx_arch=1000 >= 940 threshold)
        assert_eq!(profile.k_unroll_factor, 8);
        // is_cdna = false (1000 < 1100 is false? No: is_cdna = gfx_arch < 1100 → true for 1000)
        // Actually is_cdna = gfx_arch < 1100 → 1000 < 1100 → true. So MFMA v1 applies.
        assert!(profile.has_feature(&IsaFeature::Mfma));
        // Baseline features always present
        assert!(profile.has_feature(&IsaFeature::Fma));
        assert!(profile.has_feature(&IsaFeature::WarpShuffle));
        // LDS = 128KB (gfx1000 matches 950.. range)
        assert_eq!(profile.cache.lds_bytes, 128 * 1024);
        // wave64 for CDNA
        match &profile.platform {
            Platform::Hip { wave_size, gfx_arch, .. } => {
                assert_eq!(*wave_size, 64);
                assert_eq!(*gfx_arch, 1000);
            }
            _ => panic!("expected Hip platform"),
        }
    }

    #[test]
    fn test_aarch64_sme_vl_zero_when_sme_disabled() {
        // Arrange & Act: two profiles — one with SME, one without
        let no_sme = IsaProfile::aarch64(true, true, 64, false, false, true);
        let with_sme = IsaProfile::aarch64(true, true, 128, true, true, true);

        // Assert: sme_vl = 0 when SME disabled, = sve_vl when enabled
        match &no_sme.platform {
            Platform::AArch64 { sme_vl, .. } => assert_eq!(*sme_vl, 0),
            _ => panic!("expected AArch64"),
        }
        match &with_sme.platform {
            Platform::AArch64 { sme_vl, sve_vl, .. } => assert_eq!(*sme_vl, *sve_vl),
            _ => panic!("expected AArch64"),
        }
    }

    #[test]
    fn test_cuda_sm70_legacy_reg_file_and_max_regs() {
        // Arrange & Act: SM70 uses the legacy fallback path (sm_version < 80)
        let profile = IsaProfile::cuda(70);

        // Assert: legacy GPU parameters
        match &profile.platform {
            Platform::Cuda { reg_file_per_sm, max_regs_per_thread, shared_mem_kb, .. } => {
                assert_eq!(*reg_file_per_sm, 65536);
                assert_eq!(*max_regs_per_thread, 255);
                assert_eq!(*shared_mem_kb, 96);
            }
            _ => panic!("expected Cuda platform"),
        }
        // l1d_bytes = smem_bytes for CUDA
        assert_eq!(profile.cache.l1d_bytes, 96 * 1024);
        assert_eq!(profile.cache.l1i_bytes, 8192);
    }

    #[test]
    fn test_x86_platform_cuda_sm_and_hip_gfx_return_none() {
        // Arrange: construct X86_64 platform directly
        let x86 = Platform::X86_64 {
            has_avx512: true, has_bf16: true, has_vnni: true, has_avx512fp16: false,
            has_amx: false, has_amx_fp16: false, has_amx_complex: false,
            has_amx_transpose: false, has_amx_fp8: false,
            has_avx10_2: false, has_apx: false, has_vp2intersect: false,
        };

        // Act & Assert: CPU platforms return None for GPU-specific queries
        assert_eq!(x86.cuda_sm(), None);
        assert_eq!(x86.hip_gfx(), None);
        assert!(!x86.is_gpu());
    }

    #[test]
    fn test_aarch64_has_i8mm_tied_to_sve2() {
        // Arrange & Act: two AArch64 profiles — SVE1 vs SVE2
        let sve1 = IsaProfile::aarch64(true, false, 32, false, false, false);
        let sve2 = IsaProfile::aarch64(true, true, 32, false, false, false);

        // Assert: has_i8mm = has_sve2 per aarch64() constructor
        match &sve1.platform {
            Platform::AArch64 { has_i8mm, .. } => assert!(!has_i8mm),
            _ => panic!("expected AArch64"),
        }
        match &sve2.platform {
            Platform::AArch64 { has_i8mm, .. } => assert!(has_i8mm),
            _ => panic!("expected AArch64"),
        }
    }

    #[test]
    fn test_aarch64_sme_f16f16_and_i16i64_tied_to_sme2() {
        // Arrange & Act: SME1 vs SME2 profiles
        let sme1 = IsaProfile::aarch64(true, true, 64, true, false, true);
        let sme2 = IsaProfile::aarch64(true, true, 64, true, true, true);

        // Assert: FEAT_SME_F16F16 and FEAT_SME_I16I64 only appear with SME2
        match &sme1.platform {
            Platform::AArch64 { has_sme_f16f16, has_sme_i16i64, .. } => {
                assert!(!has_sme_f16f16, "SME1 should not have F16F16");
                assert!(!has_sme_i16i64, "SME1 should not have I16I64");
            }
            _ => panic!("expected AArch64"),
        }
        match &sme2.platform {
            Platform::AArch64 { has_sme_f16f16, has_sme_i16i64, .. } => {
                assert!(has_sme_f16f16, "SME2 should have F16F16");
                assert!(has_sme_i16i64, "SME2 should have I16I64");
            }
            _ => panic!("expected AArch64"),
        }
    }

    #[test]
    fn test_x86_from_device_profile_mask_and_tile_regs_depend_on_features() {
        // Arrange: detect DeviceProfile (real hardware)
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);

        // Act & Assert: mask_regs non-empty only when use_avx512
        let has_avx512 = matches!(&profile.platform, Platform::X86_64 { has_avx512: true, .. });
        if has_avx512 {
            assert_eq!(profile.mask_regs.len(), 8, "AVX-512 must provide 8 mask regs (k0-k7)");
        } else {
            assert!(profile.mask_regs.is_empty(), "Non-AVX-512 must have no mask regs");
        }
        // tile_regs non-empty only when AMX
        let has_amx = matches!(&profile.platform, Platform::X86_64 { has_amx: true, .. });
        if has_amx {
            assert_eq!(profile.tile_regs.len(), 8, "AMX must provide 8 tile regs (tmm0-tmm7)");
        } else {
            assert!(profile.tile_regs.is_empty(), "Non-AMX must have no tile regs");
        }
    }

    #[test]
    fn test_cuda_abi_convention_all_zero_for_gpu() {
        // Arrange & Act: CUDA profile has empty/zero ABI (GPU kernel calling convention)
        let profile = IsaProfile::cuda(90);

        // Assert
        assert!(profile.abi.arg_regs.is_empty());
        assert_eq!(profile.abi.stack_arg_offset, 0);
        assert!(profile.abi.callee_saved.is_empty());
        assert!(profile.abi.caller_saved.is_empty());
        assert!(profile.abi.callee_saved_vec.is_empty());
        assert_eq!(profile.abi.stack_alignment, 0);
        assert_eq!(profile.abi.red_zone_bytes, 0);
    }

    #[test]
    fn test_hip_abi_convention_all_zero_for_gpu() {
        // Arrange & Act: HIP profile has empty/zero ABI
        let profile = IsaProfile::hip(942);

        // Assert: same as CUDA — GPU kernels do not use CPU calling convention
        assert!(profile.abi.arg_regs.is_empty());
        assert_eq!(profile.abi.stack_arg_offset, 0);
        assert!(profile.abi.callee_saved.is_empty());
        assert!(profile.abi.caller_saved.is_empty());
        assert!(profile.abi.callee_saved_vec.is_empty());
        assert_eq!(profile.abi.stack_alignment, 0);
        assert_eq!(profile.abi.red_zone_bytes, 0);
        // All register pools empty for GPU
        assert!(profile.gpr_regs.is_empty());
        assert!(profile.vec_regs.is_empty());
        assert!(profile.scratch_gprs.is_empty());
        assert!(profile.scratch_vec_regs.is_empty());
        assert!(profile.tile_regs.is_empty());
        assert!(profile.mask_regs.is_empty());
    }

    // ── 10 Additional Tests (wave-12kma) ──

    #[test]
    fn test_phys_reg_hash_consistency_across_variants() {
        // Arrange: PhysReg derives Hash; build a HashSet to verify dedup
        use std::collections::HashSet;
        let regs = [
            PhysReg::Gpr(PhysGpr(0)),
            PhysReg::Gpr(PhysGpr(0)),
            PhysReg::Vec(PhysVec(0)),
            PhysReg::Tile(PhysTile(0)),
            PhysReg::Mask(PhysMask(0)),
            PhysReg::Spilled(0),
            PhysReg::Spilled(0),
        ];

        // Act
        let set: HashSet<PhysReg> = regs.into_iter().collect();

        // Assert: duplicates collapsed, each variant contributes one unique entry
        assert_eq!(set.len(), 5);
        assert!(set.contains(&PhysReg::Gpr(PhysGpr(0))));
        assert!(set.contains(&PhysReg::Vec(PhysVec(0))));
        assert!(set.contains(&PhysReg::Tile(PhysTile(0))));
        assert!(set.contains(&PhysReg::Mask(PhysMask(0))));
        assert!(set.contains(&PhysReg::Spilled(0)));
    }

    #[test]
    fn test_has_feature_returns_false_for_empty_features() {
        // Arrange: construct IsaProfile with an empty features list
        let profile = IsaProfile {
            platform: Platform::X86_64 {
                has_avx512: false, has_bf16: false, has_vnni: false, has_avx512fp16: false,
                has_amx: false, has_amx_fp16: false, has_amx_complex: false,
                has_amx_transpose: false, has_amx_fp8: false,
                has_avx10_2: false, has_apx: false, has_vp2intersect: false,
            },
            gpr_regs: vec![],
            scratch_gprs: vec![],
            vec_regs: vec![],
            scratch_vec_regs: vec![],
            tile_regs: vec![],
            mask_regs: vec![],
            abi: AbiConvention {
                arg_regs: vec![], stack_arg_offset: 0, callee_saved: vec![],
                caller_saved: vec![], callee_saved_vec: vec![],
                stack_alignment: 0, red_zone_bytes: 0,
            },
            cache: CacheHierarchy {
                l1d_bytes: 0, l1i_bytes: 0, l2_bytes: 0, l3_bytes: 0,
                cacheline_bytes: 64, tmem_bytes: 0, smem_bytes: 0, lds_bytes: 0,
            },
            features: vec![],
            k_unroll_factor: 1,
            dot_cap: crate::dispatch::device_profile::DotProductCap::SimdAssisted,
        };

        // Act & Assert: no feature matches on empty list
        assert!(!profile.has_feature(&IsaFeature::Fma));
        assert!(!profile.has_feature(&IsaFeature::TileGemm { m: 16, n: 16, k: 16 }));
        assert!(!profile.has_feature(&IsaFeature::WarpShuffle));
    }

    #[test]
    fn test_cuda_sm90_l2_multicast_feature_present() {
        // Arrange & Act: SM90 Hopper has L2Multicast (req for TMA multicast prefetch)
        let sm90 = IsaProfile::cuda(90);
        let sm80 = IsaProfile::cuda(80);

        // Assert: L2Multicast only on Hopper+
        assert!(sm90.has_feature(&IsaFeature::L2Multicast));
        assert!(!sm80.has_feature(&IsaFeature::L2Multicast));
        // SM100 inherits L2Multicast from Hopper
        let sm100 = IsaProfile::cuda(100);
        assert!(sm100.has_feature(&IsaFeature::L2Multicast));
    }

    #[test]
    fn test_cache_hierarchy_debug_output_contains_fields() {
        // Arrange: construct a CacheHierarchy with distinctive values
        let cache = CacheHierarchy {
            l1d_bytes: 111111,
            l1i_bytes: 222222,
            l2_bytes: 333333,
            l3_bytes: 444444,
            cacheline_bytes: 128,
            tmem_bytes: 555555,
            smem_bytes: 666666,
            lds_bytes: 777777,
        };

        // Act
        let debug = format!("{:?}", cache);

        // Assert: Debug output contains the distinctive values as evidence all fields are rendered
        assert!(debug.contains("111111"), "Debug must contain l1d_bytes");
        assert!(debug.contains("555555"), "Debug must contain tmem_bytes");
        assert!(debug.contains("777777"), "Debug must contain lds_bytes");
        assert!(!debug.is_empty());
    }

    #[test]
    fn test_metal_platform_is_gpu_and_gpu_helpers_return_none() {
        // Arrange: Metal platform (Apple M4 gpu_family=10)
        let platform = Platform::Metal {
            simd_width: 32,
            gpu_family: 10,
            has_simdgroup_matrix: true,
            threadgroup_mem_kb: 64,
        };

        // Act & Assert: Metal is GPU, but has no CUDA SM or AMD GFX
        assert!(platform.is_gpu());
        assert_eq!(platform.cuda_sm(), None);
        assert_eq!(platform.hip_gfx(), None);
    }

    #[test]
    fn test_aarch64_sve1_with_sme_has_tiles_but_no_sve2_features() {
        // Arrange & Act: SVE1 + SME but no SVE2 — an unusual but valid combination
        let profile = IsaProfile::aarch64(true, false, 32, true, false, true);

        // Assert: has SVE predication and SME tile, but lacks SVE2
        assert!(profile.has_feature(&IsaFeature::PredicatedExec));
        assert!(profile.has_feature(&IsaFeature::SmeTileOp));
        assert!(profile.has_feature(&IsaFeature::TileGemm { m: 16, n: 16, k: 16 }));
        assert!(!profile.has_feature(&IsaFeature::Sve2));
        assert!(!profile.has_feature(&IsaFeature::Sme2MultiVec));
        // Tile regs present from SME
        assert_eq!(profile.tile_regs.len(), 4);
        // Mask regs present from SVE (p0-p6, p7 reserved)
        assert_eq!(profile.mask_regs.len(), 7);
        // has_i8mm = has_sve2 = false
        match &profile.platform {
            Platform::AArch64 { has_i8mm, has_sme2, .. } => {
                assert!(!has_i8mm);
                assert!(!has_sme2);
            }
            _ => panic!("expected AArch64"),
        }
    }

    #[test]
    fn test_isa_feature_arm_i8mm_not_in_any_profile_features_list() {
        // Arrange & Act: IsaFeature::ArmI8mm is a defined variant but never pushed to features
        // by any constructor. Verify it is absent from all platform profiles.
        let aarch64_neon = IsaProfile::aarch64(false, false, 16, false, false, false);
        let aarch64_sve2 = IsaProfile::aarch64(true, true, 64, true, true, true);
        let cuda = IsaProfile::cuda(90);
        let hip = IsaProfile::hip(950);

        // Assert: ArmI8mm variant is never matched by has_feature on any real profile
        assert!(!aarch64_neon.has_feature(&IsaFeature::ArmI8mm));
        assert!(!aarch64_sve2.has_feature(&IsaFeature::ArmI8mm));
        assert!(!cuda.has_feature(&IsaFeature::ArmI8mm));
        assert!(!hip.has_feature(&IsaFeature::ArmI8mm));
    }

    #[test]
    fn test_aarch64_vec_scratch_partition_ranges() {
        // Arrange & Act: AArch64 profile — scratch_vec_regs = [16..=23] + [29..=31]
        let profile = IsaProfile::aarch64(true, true, 64, true, true, true);
        let scratch_ids: Vec<u8> = profile.scratch_vec_regs.iter().map(|r| r.0).collect();

        // Assert: two disjoint ranges
        let expected: Vec<u8> = (16u8..=23).chain(29u8..=31).collect();
        assert_eq!(scratch_ids, expected);
        // Total scratch = 11, allocatable = 21
        assert_eq!(profile.scratch_vec_regs.len(), 11);
        assert_eq!(profile.vec_regs.len(), 21);
        // No overlap
        let alloc_set: std::collections::HashSet<u8> =
            profile.vec_regs.iter().map(|r| r.0).collect();
        let scratch_set: std::collections::HashSet<u8> =
            profile.scratch_vec_regs.iter().map(|r| r.0).collect();
        assert!(alloc_set.is_disjoint(&scratch_set));
    }

    #[test]
    fn test_cuda_shared_memory_gradient_across_sm_generations() {
        // Arrange & Act: shared memory size changes across SM generations
        let sm70 = IsaProfile::cuda(70);
        let sm80 = IsaProfile::cuda(80);
        let sm90 = IsaProfile::cuda(90);
        let sm100 = IsaProfile::cuda(100);

        // Assert: SM70=96KB, SM80/SM89=164KB, SM90/SM100=228KB
        assert_eq!(sm70.cache.smem_bytes, 96 * 1024);
        assert_eq!(sm80.cache.smem_bytes, 164 * 1024);
        assert_eq!(sm90.cache.smem_bytes, 228 * 1024);
        assert_eq!(sm100.cache.smem_bytes, 228 * 1024);
        // l1d_bytes mirrors smem_bytes for CUDA
        assert_eq!(sm70.cache.l1d_bytes, sm70.cache.smem_bytes);
        assert_eq!(sm100.cache.l1d_bytes, sm100.cache.smem_bytes);
    }

    #[test]
    fn test_aarch64_abi_callee_saved_vec_range() {
        // Arrange & Act: AArch64 callee_saved_vec = v8..v15 per AAPCS
        let profile = IsaProfile::aarch64(false, false, 16, false, false, false);

        // Assert: exactly 8 callee-saved vec regs, contiguous range
        assert_eq!(profile.abi.callee_saved_vec.len(), 8);
        for (i, reg) in profile.abi.callee_saved_vec.iter().enumerate() {
            assert_eq!(reg.0, 8 + i as u8, "callee_saved_vec[{}] should be v{}", i, 8 + i);
        }
        // caller_saved = x0..x18 (19 regs)
        assert_eq!(profile.abi.caller_saved.len(), 19);
        assert_eq!(profile.abi.caller_saved[0], PhysGpr(0));
        assert_eq!(profile.abi.caller_saved[18], PhysGpr(18));
    }

    // ── 10 Additional Tests (wave-12x59) ──

    #[test]
    fn test_simd_width_f32_lanes_across_all_variants() {
        // Arrange: exercise SimdWidth::f32_lanes() for every variant via platform profiles
        let aarch64_neon = IsaProfile::aarch64(false, false, 16, false, false, false);
        let cuda = IsaProfile::cuda(90);
        let hip = IsaProfile::hip(950);

        // Act & Assert: NEON = W128 → 4 lanes, CUDA Warp(32) → 32, HIP Warp(64) → 64
        assert_eq!(aarch64_neon.optimal_simd_width().f32_lanes(), 4);
        assert_eq!(cuda.optimal_simd_width().f32_lanes(), 32);
        assert_eq!(hip.optimal_simd_width().f32_lanes(), 64);

        // Scalable variant returns 0 (runtime VL unknown at compile time)
        let aarch64_sve = IsaProfile::aarch64(true, false, 64, false, false, false);
        assert_eq!(aarch64_sve.optimal_simd_width().f32_lanes(), 0);
    }

    #[test]
    fn test_cuda_feature_progression_sm70_to_sm100() {
        // Arrange: profiles spanning the SM generation range
        let sm70 = IsaProfile::cuda(70);
        let sm80 = IsaProfile::cuda(80);
        let sm90 = IsaProfile::cuda(90);
        let sm100 = IsaProfile::cuda(100);

        // Act & Assert: feature count strictly non-decreasing across generations
        assert!(sm70.features.len() > 0, "SM70 must have baseline features");
        assert!(sm80.features.len() >= sm70.features.len(),
            "SM80 feature count ({}) must >= SM70 ({})", sm80.features.len(), sm70.features.len());
        assert!(sm90.features.len() >= sm80.features.len(),
            "SM90 feature count ({}) must >= SM80 ({})", sm90.features.len(), sm80.features.len());
        assert!(sm100.features.len() >= sm90.features.len(),
            "SM100 feature count ({}) must >= SM90 ({})", sm100.features.len(), sm90.features.len());

        // AsyncCopy appears at SM80+, never on SM70
        assert!(!sm70.has_feature(&IsaFeature::AsyncCopy));
        assert!(sm80.has_feature(&IsaFeature::AsyncCopy));
        assert!(sm90.has_feature(&IsaFeature::AsyncCopy));
        assert!(sm100.has_feature(&IsaFeature::AsyncCopy));
    }

    #[test]
    fn test_hip_feature_wave_size_cdna_vs_rdna() {
        // Arrange: CDNA (compute) vs RDNA (graphics) architectures
        let cdna2 = IsaProfile::hip(908);
        let cdna3 = IsaProfile::hip(942);
        let rdna3 = IsaProfile::hip(1100);

        // Act & Assert: CDNA = wave64, RDNA = wave32
        let cdna2_ws = match &cdna2.platform { Platform::Hip { wave_size, .. } => *wave_size, _ => 0 };
        let cdna3_ws = match &cdna3.platform { Platform::Hip { wave_size, .. } => *wave_size, _ => 0 };
        let rdna3_ws = match &rdna3.platform { Platform::Hip { wave_size, .. } => *wave_size, _ => 0 };

        assert_eq!(cdna2_ws, 64);
        assert_eq!(cdna3_ws, 64);
        assert_eq!(rdna3_ws, 32);

        // MFMA only on CDNA, not RDNA
        assert!(cdna2.has_feature(&IsaFeature::Mfma));
        assert!(cdna3.has_feature(&IsaFeature::Mfma));
        assert!(!rdna3.has_feature(&IsaFeature::Mfma));
    }

    #[test]
    fn test_aarch64_sme_tile_reg_count_is_four() {
        // Arrange & Act: SME provides exactly 4 ZA tile registers
        let with_sme = IsaProfile::aarch64(true, true, 64, true, true, true);
        let without_sme = IsaProfile::aarch64(true, true, 64, false, false, true);

        // Assert
        assert_eq!(with_sme.tile_regs.len(), 4);
        // Verify tile register IDs are sequential
        assert_eq!(with_sme.tile_regs[0], PhysTile(0));
        assert_eq!(with_sme.tile_regs[3], PhysTile(3));
        // No SME means no tile registers
        assert!(without_sme.tile_regs.is_empty());
    }

    #[test]
    fn test_x86_abi_sysv_arg_registers_ordered() {
        // Arrange: x86 SysV ABI has 6 integer argument registers in specific order
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);

        // Act & Assert: arg_regs = [rdi, rsi, rdx, rcx, r8, r9] = [7, 6, 2, 1, 8, 9]
        assert_eq!(profile.abi.arg_regs.len(), 6);
        assert_eq!(profile.abi.arg_regs[0], PhysGpr(7)); // rdi
        assert_eq!(profile.abi.arg_regs[1], PhysGpr(6)); // rsi
        assert_eq!(profile.abi.arg_regs[2], PhysGpr(2)); // rdx
        assert_eq!(profile.abi.arg_regs[3], PhysGpr(1)); // rcx
        assert_eq!(profile.abi.arg_regs[4], PhysGpr(8)); // r8
        assert_eq!(profile.abi.arg_regs[5], PhysGpr(9)); // r9

        // Callee-saved = [rbx, r12, r13, r14, r15] = [3, 12, 13, 14, 15]
        assert_eq!(profile.abi.callee_saved.len(), 5);
        assert_eq!(profile.abi.callee_saved[0], PhysGpr(3)); // rbx
        assert_eq!(profile.abi.callee_saved[4], PhysGpr(15)); // r15
    }

    #[test]
    fn test_cuda_sm86_intermediate_ampere_profile() {
        // Arrange & Act: SM86 is the A100 variant, between SM80 and SM89
        let profile = IsaProfile::cuda(86);

        // Assert: inherits Ampere features (async copy, BF16), lacks Ada FP8
        assert!(profile.has_feature(&IsaFeature::Fma));
        assert!(profile.has_feature(&IsaFeature::AsyncCopy));
        assert!(profile.has_feature(&IsaFeature::NativeBf16));
        assert!(profile.has_feature(&IsaFeature::TileGemm { m: 16, n: 16, k: 16 }));
        // SM86 < 89 → no FP8
        assert!(!profile.has_feature(&IsaFeature::NativeFp8));
        // SM86 < 90 → no Hopper features
        assert!(!profile.has_feature(&IsaFeature::Wgmma));
        assert!(!profile.has_feature(&IsaFeature::WarpSpecialization));
        // SM86 shared memory = 164 KB (same as SM80 range)
        match &profile.platform {
            Platform::Cuda { shared_mem_kb, sm_version, .. } => {
                assert_eq!(*shared_mem_kb, 164);
                assert_eq!(*sm_version, 86);
            }
            _ => panic!("expected Cuda platform"),
        }
        // k_unroll_factor = 4 (pre-Hopper)
        assert_eq!(profile.k_unroll_factor, 4);
    }

    #[test]
    fn test_aarch64_sve_mask_regs_exclude_p7_reserved() {
        // Arrange & Act: SVE has 8 predicate registers p0-p7, p7 is reserved as all-true
        let sve_profile = IsaProfile::aarch64(true, true, 64, false, false, true);
        let neon_profile = IsaProfile::aarch64(false, false, 16, false, false, false);

        // Assert: SVE provides 7 allocatable masks (p0-p6)
        assert_eq!(sve_profile.mask_regs.len(), 7);
        // Verify mask IDs are sequential starting from 0
        for (i, mask) in sve_profile.mask_regs.iter().enumerate() {
            assert_eq!(mask.0, i as u8, "mask_regs[{}] should be p{}", i, i);
        }
        // p7 is not in the allocatable set
        let mask_ids: std::collections::HashSet<u8> =
            sve_profile.mask_regs.iter().map(|m| m.0).collect();
        assert!(!mask_ids.contains(&7), "p7 must not be allocatable (reserved for all-true)");

        // NEON has no predicate masks
        assert!(neon_profile.mask_regs.is_empty());
    }

    #[test]
    fn test_isa_profile_has_feature_ignores_field_values_for_simple_variants() {
        // Arrange: verify discriminant-based matching for non-parameterized IsaFeature variants
        let profile = IsaProfile::cuda(100);

        // Act & Assert: has_feature matches by discriminant, so any Fma == Fma
        assert!(profile.has_feature(&IsaFeature::Fma));
        assert!(profile.has_feature(&IsaFeature::WarpShuffle));
        assert!(profile.has_feature(&IsaFeature::AsyncCopy));
        assert!(profile.has_feature(&IsaFeature::NativeBf16));
        assert!(profile.has_feature(&IsaFeature::NativeFp8));
        assert!(profile.has_feature(&IsaFeature::Wgmma));
        assert!(profile.has_feature(&IsaFeature::Tma));
        assert!(profile.has_feature(&IsaFeature::Tmem));
        assert!(profile.has_feature(&IsaFeature::BlockScaled));
        assert!(profile.has_feature(&IsaFeature::NativeFp4));
        assert!(profile.has_feature(&IsaFeature::NativeFp6));
        assert!(profile.has_feature(&IsaFeature::ThreadBlockCluster));
        assert!(profile.has_feature(&IsaFeature::TwoCta));
        // Features NOT present on CUDA
        assert!(!profile.has_feature(&IsaFeature::Sve2));
        assert!(!profile.has_feature(&IsaFeature::Mfma));
        assert!(!profile.has_feature(&IsaFeature::Vnni));
        assert!(!profile.has_feature(&IsaFeature::AmxFp16));
    }

    #[test]
    fn test_x86_vec_scratch_highest_indices_regardless_of_width() {
        // Arrange: x86 scratch_vec_regs are the last 6 vector registers
        let dp = DeviceProfile::detect();
        let profile = IsaProfile::from_device_profile(&dp);

        // Act & Assert: scratch regs are the highest-numbered vec regs
        let scratch_ids: Vec<u8> = profile.scratch_vec_regs.iter().map(|r| r.0).collect();
        // All scratch IDs must be distinct
        let unique: std::collections::HashSet<u8> = scratch_ids.iter().copied().collect();
        assert_eq!(unique.len(), 6, "scratch_vec_regs must have 6 unique entries");

        // No scratch reg may appear in the allocatable vec_regs
        let alloc_ids: std::collections::HashSet<u8> =
            profile.vec_regs.iter().map(|r| r.0).collect();
        for sid in &scratch_ids {
            assert!(!alloc_ids.contains(sid), "scratch vec {} must not be in allocatable pool", sid);
        }

        // The highest scratch reg + 1 should equal the total vec count (contiguous from top)
        let max_scratch = *scratch_ids.iter().max().unwrap();
        let vec_count = if matches!(&profile.platform, Platform::X86_64 { has_avx512: true, .. }) { 32 } else { 16 };
        assert_eq!(max_scratch + 1, vec_count, "highest scratch reg should be vec_count - 1");
    }

    #[test]
    fn test_cuda_hip_metal_all_report_as_gpu() {
        // Arrange: one profile per GPU platform
        let cuda = IsaProfile::cuda(90);
        let hip = IsaProfile::hip(942);
        let metal = IsaProfile {
            platform: Platform::Metal {
                simd_width: 32,
                gpu_family: 9,
                has_simdgroup_matrix: true,
                threadgroup_mem_kb: 32,
            },
            gpr_regs: vec![],
            scratch_gprs: vec![],
            vec_regs: vec![],
            scratch_vec_regs: vec![],
            tile_regs: vec![],
            mask_regs: vec![],
            abi: AbiConvention {
                arg_regs: vec![],
                stack_arg_offset: 0,
                callee_saved: vec![],
                caller_saved: vec![],
                callee_saved_vec: vec![],
                stack_alignment: 0,
                red_zone_bytes: 0,
            },
            cache: CacheHierarchy {
                l1d_bytes: 0,
                l1i_bytes: 0,
                l2_bytes: 0,
                l3_bytes: 0,
                cacheline_bytes: 64,
                tmem_bytes: 0,
                smem_bytes: 0,
                lds_bytes: 0,
            },
            features: vec![],
            k_unroll_factor: 4,
            dot_cap: crate::dispatch::device_profile::DotProductCap::SimdAssisted,
        };

        // Act & Assert: all GPU platforms return true for is_gpu()
        assert!(cuda.platform.is_gpu(), "CUDA must be GPU");
        assert!(hip.platform.is_gpu(), "HIP must be GPU");
        assert!(metal.platform.is_gpu(), "Metal must be GPU");

        // Cross-platform identity: each has a unique non-overlapping SM/GFX query
        assert!(cuda.platform.cuda_sm().is_some());
        assert!(cuda.platform.hip_gfx().is_none());
        assert!(hip.platform.hip_gfx().is_some());
        assert!(hip.platform.cuda_sm().is_none());
        assert!(metal.platform.cuda_sm().is_none());
        assert!(metal.platform.hip_gfx().is_none());
    }
}
