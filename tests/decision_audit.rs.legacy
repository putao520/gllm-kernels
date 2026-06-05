//! Decision Audit — validates JIT decision chains across 7 synthetic hardware profiles.
//!
//! Modules:
//! 1. Device simulators (7 profiles)
//! 2. GEMM strategy audit
//! 3. Attention strategy audit
//! 4. BLIS blocking invariants
//! 5. Register pressure audit

use gllm_kernels::autotuning::HwInfo;
use gllm_kernels::autotuning::hw_info::IsaFeatures;
use gllm_kernels::compiler::codegen::attention_strategy::select_attention_strategy;
use gllm_kernels::compiler::codegen::gemm_dispatch::{select_gemm_strategy, GemmStrategy};
use gllm_kernels::compiler::graph::AttentionStrategy;
use gllm_kernels::dispatch::device_profile::{DeviceProfile, IsaLevel, IsvCapabilities};
use gllm_kernels::gpu::{GpuDeviceProfile, GpuIsvCapabilities};
use gllm_kernels::microarch::{KernelConfig, MicroArch};
use gllm_kernels::numa::{NumaNode, NumaTopology};
use gllm_kernels::types::DType;

// ═══════════════════════════════════════════════════════════════════════
// Module 1: Device Simulators
// ═══════════════════════════════════════════════════════════════════════

/// Shared ISA features: all false.
fn isa_none() -> IsaFeatures {
    IsaFeatures {
        avx2: false, fma: false, avx512f: false, avx512bw: false,
        avx512vnni: false, avx512fp16: false, avx512bf16: false,
        neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
    }
}

fn single_numa() -> NumaTopology {
    NumaTopology {
        nodes: vec![NumaNode { id: 0, cpus: vec![0, 1, 2, 3], mem_total: 32 * 1024 * 1024 * 1024, l3_size: 0 }],
        distances: vec![vec![10]],
    }
}

fn make_hw_info(vendor: &str, model: &str, phys: usize, log: usize,
                l1d: usize, l2: usize, l3: usize, isa: IsaFeatures) -> HwInfo {
    HwInfo {
        vendor: vendor.into(), model_name: model.into(),
        physical_cores: phys, logical_cores: log,
        l1d_bytes: l1d, l2_bytes: l2, l3_bytes: l3,
        cacheline_bytes: 64, isa,
    }
}

fn make_profile(arch: MicroArch, isa_level: IsaLevel, l1d: usize, l2: usize, l3: usize,
                hw: HwInfo, isv: IsvCapabilities) -> DeviceProfile {
    let kc = KernelConfig::from_arch_with_cache(arch, l1d, l2, l3);
    let (_, _, simd_w) = arch.microkernel_geometry();
    let fma_ports = arch.fma_ports() as f64;
    let cores = hw.physical_cores;
    let peak_gflops_f32 = cores as f64 * arch.estimated_freq_ghz() * fma_ports * simd_w as f64 * 2.0;
    DeviceProfile {
        arch, isa: isa_level, kernel_config: kc, hw_info: hw,
        numa: single_numa(),
        peak_gflops_f32, peak_bandwidth_gbs: 50.0,
        physical_cores: cores, logical_cores: cores * 2, isv,
    }
}

/// Haswell: AVX2, 32K/256K/8M, no ISV
fn profile_haswell() -> DeviceProfile {
    let isa = IsaFeatures { avx2: true, fma: true, ..isa_none() };
    let hw = make_hw_info("GenuineIntel", "Haswell-sim", 4, 8, 32768, 262144, 8388608, isa);
    make_profile(MicroArch::Haswell, IsaLevel::Avx2, 32768, 262144, 8388608, hw, IsvCapabilities::default())
}

/// Sapphire Rapids: AVX-512+AMX, 48K/2M/30M, amx+onednn
fn profile_spr() -> DeviceProfile {
    let isa = IsaFeatures { avx2: true, fma: true, avx512f: true, avx512bw: true,
        avx512vnni: true, avx512bf16: true, ..isa_none() };
    let hw = make_hw_info("GenuineIntel", "SPR-sim", 8, 16, 49152, 2097152, 31457280, isa);
    let isv = IsvCapabilities { onednn_available: true, accelerate_available: false };
    make_profile(MicroArch::SapphireRapids, IsaLevel::Avx512Amx, 49152, 2097152, 31457280, hw, isv)
}

/// Zen4: AVX2 (no AVX-512 in our dispatch), 32K/1M/32M, no ISV
fn profile_zen4() -> DeviceProfile {
    let isa = IsaFeatures { avx2: true, fma: true, ..isa_none() };
    let hw = make_hw_info("AuthenticAMD", "Zen4-sim", 8, 16, 32768, 1048576, 33554432, isa);
    make_profile(MicroArch::Zen4, IsaLevel::Avx2, 32768, 1048576, 33554432, hw, IsvCapabilities::default())
}

/// Apple M2: NEON+AMX, 128K/4M/8M, accelerate
fn profile_m2() -> DeviceProfile {
    let isa = IsaFeatures { neon: true, ..isa_none() };
    let hw = make_hw_info("Apple", "M2-sim", 8, 8, 131072, 4194304, 8388608, isa);
    let isv = IsvCapabilities { accelerate_available: true, ..Default::default() };
    // MicroArch::Scalar as placeholder for ARM (no ARM-specific MicroArch variant)
    make_profile(MicroArch::Scalar, IsaLevel::NeonAmx, 131072, 4194304, 8388608, hw, isv)
}

/// Graviton3: SVE 256-bit, 64K/1M/32M, no ISV
fn profile_graviton3() -> DeviceProfile {
    let isa = IsaFeatures { neon: true, sve: true, sve_vl_bytes: 32, ..isa_none() };
    let hw = make_hw_info("ARM", "Graviton3-sim", 64, 64, 65536, 1048576, 33554432, isa);
    make_profile(MicroArch::Scalar, IsaLevel::Sve, 65536, 1048576, 33554432, hw, IsvCapabilities::default())
}

/// GPU mock helper
fn mock_gpu(sm_version: u32) -> GpuDeviceProfile {
    use gllm_kernels::compiler::codegen::emitter::Platform;
    GpuDeviceProfile {
        platform: Platform::X86_64 { avx512: true, amx: false }, // placeholder
        compute_units: 108,
        shared_mem_per_block: 49152,
        max_registers_per_thread: 255,
        warp_size: 32,
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [2147483647, 65535, 65535],
        total_memory: 40 * 1024 * 1024 * 1024,
        memory_bandwidth_gbs: 1555.0,
        peak_gflops_f32: 19500.0,
        peak_gflops_f16: 39000.0,
        has_matrix_unit: true,
        clock_mhz: 1410,
        isv: GpuIsvCapabilities { tensor_core_gen: if sm_version >= 90 { 3 } else if sm_version >= 80 { 2 } else if sm_version >= 70 { 1 } else { 0 }, ..Default::default() },
    }
}

/// A100: sm_80
fn gpu_a100() -> GpuDeviceProfile { mock_gpu(80) }
/// H100: sm_90
fn gpu_h100() -> GpuDeviceProfile { mock_gpu(90) }

// ═══════════════════════════════════════════════════════════════════════
// Module 1 Tests: Device Profile Construction
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn device_haswell_fields() {
    let p = profile_haswell();
    assert_eq!(p.isa, IsaLevel::Avx2);
    assert_eq!(p.kernel_config.mr, 6);
    assert_eq!(p.kernel_config.nr, 16);
    let (l1, l2, l3) = p.cache_sizes();
    assert_eq!(l1, 32768);
    assert_eq!(l2, 262144);
    assert_eq!(l3, 8388608);
    assert_eq!(p.num_simd_regs(), 16);
    eprintln!("[Haswell] MR={} NR={} L1={}K L2={}K L3={}M regs={}",
        p.kernel_config.mr, p.kernel_config.nr, l1/1024, l2/1024, l3/1048576, p.num_simd_regs());
}

#[test]
fn device_spr_fields() {
    let p = profile_spr();
    assert_eq!(p.isa, IsaLevel::Avx512Amx);
    assert_eq!(p.kernel_config.mr, 14);
    assert_eq!(p.kernel_config.nr, 32);
    assert_eq!(p.num_simd_regs(), 32);
    assert!(p.kernel_config.has_amx);
    assert!(p.isv.onednn_available);
    eprintln!("[SPR] MR={} NR={} regs={} isa={:?} onednn={}",
        p.kernel_config.mr, p.kernel_config.nr, p.num_simd_regs(), p.isa, p.isv.onednn_available);
}

#[test]
fn device_zen4_fields() {
    let p = profile_zen4();
    assert_eq!(p.isa, IsaLevel::Avx2);
    assert_eq!(p.kernel_config.mr, 6);
    assert_eq!(p.kernel_config.nr, 16);
    let (l1, l2, l3) = p.cache_sizes();
    assert_eq!(l1, 32768);
    assert_eq!(l2, 1048576);
    assert_eq!(l3, 33554432);
    eprintln!("[Zen4] L1={}K L2={}K L3={}M", l1/1024, l2/1024, l3/1048576);
}

#[test]
fn device_m2_fields() {
    let p = profile_m2();
    assert_eq!(p.isa, IsaLevel::NeonAmx);
    assert_eq!(p.num_simd_regs(), 32);
    assert!(p.isv.accelerate_available);
    let (l1, _, _) = p.cache_sizes();
    assert_eq!(l1, 131072);
    eprintln!("[M2] ISA={:?} regs={} accelerate={}", p.isa, p.num_simd_regs(), p.isv.accelerate_available);
}

#[test]
fn device_graviton3_fields() {
    let p = profile_graviton3();
    assert_eq!(p.isa, IsaLevel::Sve);
    assert_eq!(p.num_simd_regs(), 32);
    let (l1, l2, l3) = p.cache_sizes();
    assert_eq!(l1, 65536);
    assert_eq!(l2, 1048576);
    assert_eq!(l3, 33554432);
    eprintln!("[Graviton3] ISA={:?} L1={}K", p.isa, l1/1024);
}

#[test]
fn device_a100_gpu_fields() {
    let g = gpu_a100();
    assert_eq!(g.isv.tensor_core_gen, 2);
    assert_eq!(g.total_memory, 40 * 1024 * 1024 * 1024);
    eprintln!("[A100] tc_gen={} mem={}GB",
        g.isv.tensor_core_gen, g.total_memory / (1024*1024*1024));
}

#[test]
fn device_h100_gpu_fields() {
    let g = gpu_h100();
    assert_eq!(g.isv.tensor_core_gen, 3);
    eprintln!("[H100] tc_gen={}", g.isv.tensor_core_gen);
}

// ═══════════════════════════════════════════════════════════════════════
// Module 2: GEMM Strategy Audit
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn gemm_haswell_all_jitblis() {
    let p = profile_haswell();
    for &(m, n, k) in &[(8,8,8), (64,64,64), (128,128,128)] {
        let s = select_gemm_strategy(m, n, k, DType::F32, &p, None);
        assert_eq!(s, GemmStrategy::JitBlis, "Haswell {m}x{n}x{k} F32 should be JitBlis");
        eprintln!("[GEMM] Haswell {m}x{n}x{k} F32 → {:?}", s);
    }
}

#[test]
fn gemm_spr_bf16_amx() {
    let p = profile_spr();
    // small: 8^3=512 < 65536 → JitBlis
    let s_small = select_gemm_strategy(8, 8, 8, DType::BF16, &p, None);
    assert_eq!(s_small, GemmStrategy::JitBlis, "SPR 8^3 BF16 too small for AMX");
    // medium: 64^3=262144 > 65536 → Amx
    let s_med = select_gemm_strategy(64, 64, 64, DType::BF16, &p, None);
    assert_eq!(s_med, GemmStrategy::Amx, "SPR 64^3 BF16 should use AMX");
    // large: 128^3=2097152 > 65536 → Amx (AMX checked before oneDNN)
    let s_large = select_gemm_strategy(128, 128, 128, DType::BF16, &p, None);
    assert_eq!(s_large, GemmStrategy::Amx, "SPR 128^3 BF16 should use AMX");
    eprintln!("[GEMM] SPR BF16: 8^3→{:?}, 64^3→{:?}, 128^3→{:?}", s_small, s_med, s_large);
}

#[test]
fn gemm_spr_f32_onednn() {
    let p = profile_spr();
    // small: JitBlis (AMX requires BF16)
    let s_small = select_gemm_strategy(8, 8, 8, DType::F32, &p, None);
    assert_eq!(s_small, GemmStrategy::JitBlis);
    // SPR l1d=49152, oneDNN threshold = (l1d/4)*4 = l1d = 49152 (min 32768) * 4 = 196608
    // 64^3=262144 > 196608 → OneDnn
    let s_med = select_gemm_strategy(64, 64, 64, DType::F32, &p, None);
    assert_eq!(s_med, GemmStrategy::OneDnn, "SPR 64^3 F32 should use OneDnn (dynamic threshold)");
    // large: 128^3=2097152 → OneDnn
    let s_large = select_gemm_strategy(128, 128, 128, DType::F32, &p, None);
    assert_eq!(s_large, GemmStrategy::OneDnn, "SPR 128^3 F32 should use OneDnn");
    eprintln!("[GEMM] SPR F32: 8^3→{:?}, 64^3→{:?}, 128^3→{:?}", s_small, s_med, s_large);
}

#[test]
fn gemm_zen4_all_jitblis() {
    let p = profile_zen4();
    for &(m, n, k) in &[(8,8,8), (64,64,64), (128,128,128)] {
        let s = select_gemm_strategy(m, n, k, DType::F32, &p, None);
        assert_eq!(s, GemmStrategy::JitBlis, "Zen4 {m}x{n}x{k} should be JitBlis (no ISV)");
    }
}

#[test]
fn gemm_m2_apple_amx() {
    let p = profile_m2();
    // small: 8^3=512 < 65536 → JitBlis
    let s_small = select_gemm_strategy(8, 8, 8, DType::F32, &p, None);
    assert_eq!(s_small, GemmStrategy::JitBlis);
    // medium: 64^3=262144 > 65536 → AppleAmx
    let s_med = select_gemm_strategy(64, 64, 64, DType::F32, &p, None);
    assert_eq!(s_med, GemmStrategy::AppleAmx, "M2 64^3 should use AppleAmx");
    eprintln!("[GEMM] M2: 8^3→{:?}, 64^3→{:?}", s_small, s_med);
}

#[test]
fn gemm_gpu_jit() {
    let p = profile_haswell(); // CPU profile is placeholder for GPU tests
    let a100 = gpu_a100();
    // small: 8^3=512 < 1M → JitBlis (falls through GPU path)
    let s_small = select_gemm_strategy(8, 8, 8, DType::F32, &p, Some(&a100));
    assert_eq!(s_small, GemmStrategy::JitBlis, "GPU 8^3 too small for GPU dispatch");
    // large: 128^3=2M > 1M, A100 tc_gen=2 → JitGpuTensorCore
    let s_large = select_gemm_strategy(128, 128, 128, DType::F32, &p, Some(&a100));
    assert_eq!(s_large, GemmStrategy::JitGpuTensorCore, "A100 128^3 should use JitGpuTensorCore");
    let h100 = gpu_h100();
    // H100 tc_gen=3 → JitGpuTensorCore
    let s_h100 = select_gemm_strategy(128, 128, 128, DType::F32, &p, Some(&h100));
    assert_eq!(s_h100, GemmStrategy::JitGpuTensorCore, "H100 128^3 should use JitGpuTensorCore");
    eprintln!("[GEMM] GPU: 8^3→{:?}, A100 128^3→{:?}, H100 128^3→{:?}", s_small, s_large, s_h100);
}

// ═══════════════════════════════════════════════════════════════════════
// Module 3: Attention Strategy Audit
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn attn_short_sequence_naive() {
    let p = profile_haswell();
    let s = select_attention_strategy(1, 32, 128, 32, DType::F32, &p, None, None);
    assert_eq!(s, AttentionStrategy::Naive);
    eprintln!("[Attn] seq=1,total=32 → {:?}", s);
}

#[test]
fn attn_long_sequence_flash() {
    let p = profile_haswell();
    // seq=2048, total=2048: scores_bytes = 2048*2048*4 = 16MB >> L1/2
    let s = select_attention_strategy(2048, 2048, 128, 32, DType::F32, &p, None, None);
    match s {
        AttentionStrategy::FlashV2 { block_m, block_n } => {
            // Cache-driven tiles: power of 2, in [32, 256]
            assert!(block_m.is_power_of_two(), "block_m must be power of 2, got {block_m}");
            assert!(block_m >= 32 && block_m <= 256, "block_m out of range: {block_m}");
            assert_eq!(block_m, block_n, "square tiles expected");
        }
        other => panic!("Expected FlashV2, got {:?}", other),
    }
    eprintln!("[Attn] seq=2048,total=2048 F32 → {:?}", s);
}

#[test]
fn attn_f16_larger_tiles() {
    let p = profile_haswell();
    let s_f16 = select_attention_strategy(2048, 2048, 128, 32, DType::F16, &p, None, None);
    let s_f32 = select_attention_strategy(2048, 2048, 128, 32, DType::F32, &p, None, None);
    match (&s_f16, &s_f32) {
        (AttentionStrategy::FlashV2 { block_m: m16, .. }, AttentionStrategy::FlashV2 { block_m: m32, .. }) => {
            // F16 elements are half the size → larger tiles or equal
            assert!(*m16 >= *m32, "F16 tiles ({m16}) should be >= F32 tiles ({m32})");
            assert!(m16.is_power_of_two(), "F16 block_m must be power of 2, got {m16}");
        }
        _ => panic!("Expected FlashV2 for both, got F16={:?}, F32={:?}", s_f16, s_f32),
    }
    eprintln!("[Attn] F16 → {:?}, F32 → {:?}", s_f16, s_f32);
}

#[test]
fn attn_gpu_paged() {
    let p = profile_haswell();
    let gpu = gpu_a100();
    // A100 40GiB → paged threshold = 8192, use total_seq > 8192 to trigger
    let s = select_attention_strategy(1, 16384, 128, 32, DType::F32, &p, Some(&gpu), None);
    assert_eq!(s, AttentionStrategy::Paged { page_size: 16 });
    eprintln!("[Attn] GPU total=16384 → {:?}", s);
}

#[test]
fn attn_gpu_short_not_paged() {
    let p = profile_haswell();
    let gpu = gpu_a100();
    // GPU + total_seq <= 2048 → not Paged, falls through to FlashV2 or Naive
    let s = select_attention_strategy(1, 32, 128, 32, DType::F32, &p, Some(&gpu), None);
    assert!(
        !matches!(s, AttentionStrategy::Paged { .. }),
        "GPU short seq should not be Paged, got {:?}", s
    );
    eprintln!("[Attn] GPU total=32 → {:?}", s);
}

#[test]
fn attn_sliding_window_priority() {
    let p = profile_haswell();
    let gpu = gpu_a100();
    // Sliding window overrides everything, even GPU+long seq
    let s = select_attention_strategy(1, 4096, 128, 32, DType::F32, &p, Some(&gpu), Some(512));
    assert_eq!(s, AttentionStrategy::SlidingWindow { window_size: 512 });
    eprintln!("[Attn] sliding_window=512 overrides GPU → {:?}", s);
}

// ═══════════════════════════════════════════════════════════════════════
// Module 4: BLIS Blocking Invariants
// ═══════════════════════════════════════════════════════════════════════

const PROBLEM_SIZES: &[(usize, usize, usize)] = &[
    (8, 8, 8), (64, 64, 64), (256, 256, 256),
    (1024, 1024, 1024), (4096, 4096, 4096),
    (1, 4096, 4096), (4096, 1, 4096), (32, 4096, 4096), (512, 2048, 768),
];

fn assert_blocking_invariants(profile: &DeviceProfile, label: &str) {
    let (l1, l2, l3) = profile.cache_sizes();
    let (mr, nr) = profile.microkernel_mr_nr();

    for &(m, n, k) in PROBLEM_SIZES {
        let b = profile.gemm_blocking(m, n, k);
        let tag = format!("{label} {m}x{n}x{k}");

        // 1. KC*(MR+NR)*4 <= L1*85% — micropanels fit in L1
        let micropanel = b.kc * (mr + nr) * 4;
        assert!(micropanel <= l1 * 85 / 100,
            "{tag}: micropanels {micropanel}B > 85% L1 {}B", l1 * 85 / 100);

        // 2. MC*KC*4 <= L2*85% — A panel fits in L2
        let a_panel = b.mc * b.kc * 4;
        assert!(a_panel <= l2 * 85 / 100,
            "{tag}: A panel {a_panel}B > 85% L2 {}B", l2 * 85 / 100);

        // 3. KC*NC*4 <= L3*65% — B panel fits in L3 (skip if L3 < 1MB)
        if l3 >= 1024 * 1024 {
            let b_panel = b.kc * b.nc * 4;
            assert!(b_panel <= l3 * 65 / 100,
                "{tag}: B panel {b_panel}B > 65% L3 {}B", l3 * 65 / 100);
        }

        // 4. MC % MR == 0 (when MC < m)
        if b.mc < m {
            assert_eq!(b.mc % mr, 0, "{tag}: MC={} not aligned to MR={mr}", b.mc);
        }

        // 5. NC % NR == 0 (when NC < n)
        if b.nc < n {
            assert_eq!(b.nc % nr, 0, "{tag}: NC={} not aligned to NR={nr}", b.nc);
        }

        // 6. Minimum values (clamped to actual dimension when dim < tile)
        assert!(b.kc >= 1, "{tag}: KC={} < 1", b.kc);
        assert!(b.mc >= m.min(mr).min(b.mc.max(1)), "{tag}: MC={} unexpectedly zero", b.mc);
        assert!(b.nc >= n.min(nr).min(b.nc.max(1)), "{tag}: NC={} unexpectedly zero", b.nc);
        // When dimension is large enough, blocking should reach MR/NR
        if m >= mr {
            assert!(b.mc >= mr, "{tag}: MC={} < MR={mr} (m={m} >= MR)", b.mc);
        }
        if n >= nr {
            assert!(b.nc >= nr, "{tag}: NC={} < NR={nr} (n={n} >= NR)", b.nc);
        }

        // 7. Don't exceed dimensions
        assert!(b.kc <= k, "{tag}: KC={} > k={k}", b.kc);
        assert!(b.mc <= m, "{tag}: MC={} > m={m}", b.mc);
        assert!(b.nc <= n, "{tag}: NC={} > n={n}", b.nc);

        eprintln!("[BLIS] {tag}: KC={} MC={} NC={} MR={mr} NR={nr}", b.kc, b.mc, b.nc);
    }
}

#[test]
fn blis_invariants_haswell() { assert_blocking_invariants(&profile_haswell(), "Haswell"); }

#[test]
fn blis_invariants_spr() { assert_blocking_invariants(&profile_spr(), "SPR"); }

#[test]
fn blis_invariants_zen4() { assert_blocking_invariants(&profile_zen4(), "Zen4"); }

#[test]
fn blis_invariants_m2() { assert_blocking_invariants(&profile_m2(), "M2"); }

#[test]
fn blis_invariants_graviton3() { assert_blocking_invariants(&profile_graviton3(), "Graviton3"); }

// ═══════════════════════════════════════════════════════════════════════
// Module 5: Register Pressure Audit
// ═══════════════════════════════════════════════════════════════════════

/// Verify microkernel geometry fits within the register file.
/// base_regs = MR * ceil(NR / simd_w) + ceil(MR / simd_w) + 1
/// Must have: base_regs + scratch <= num_simd_regs
fn assert_register_pressure(profile: &DeviceProfile, label: &str) {
    let (mr, nr) = profile.microkernel_mr_nr();
    let simd_w = profile.simd_width_f32();
    let num_regs = profile.num_simd_regs();

    if num_regs == 0 {
        eprintln!("[Regs] {label}: Scalar ISA, no SIMD register file to audit");
        return;
    }

    // Accumulator tiles: MR rows × ceil(NR/simd_w) columns
    let acc_cols = (nr + simd_w - 1) / simd_w;
    let acc_regs = mr * acc_cols;

    // A broadcast registers: ceil(MR / simd_w)
    let a_regs = (mr + simd_w - 1) / simd_w;

    // B load register: 1
    let b_regs = 1;

    let base_regs = acc_regs + a_regs + b_regs;
    let scratch = num_regs.saturating_sub(base_regs);

    eprintln!("[Regs] {label}: MR={mr} NR={nr} simd_w={simd_w} → acc={acc_regs} a={a_regs} b={b_regs} base={base_regs}/{num_regs} scratch={scratch}");

    assert!(base_regs <= num_regs,
        "{label}: base_regs={base_regs} > num_simd_regs={num_regs} (MR={mr} NR={nr} simd_w={simd_w})");
    assert!(scratch >= 1,
        "{label}: no scratch registers left (base={base_regs} total={num_regs})");
}

#[test]
fn regs_haswell_avx2() {
    // AVX2: 16 regs, MR=6, NR=16, simd_w=8 → acc=12, a=1, b=1 → base=14, scratch=2
    let p = profile_haswell();
    assert_register_pressure(&p, "Haswell/AVX2");
}

#[test]
fn regs_spr_avx512() {
    // AVX-512: 32 regs, MR=14, NR=32, simd_w=16 → acc=28, a=1, b=1 → base=30, scratch=2
    let p = profile_spr();
    assert_register_pressure(&p, "SPR/AVX-512");
}

#[test]
fn regs_m2_neon() {
    let p = profile_m2();
    assert_register_pressure(&p, "M2/NEON");
}

#[test]
fn regs_graviton3_sve() {
    let p = profile_graviton3();
    assert_register_pressure(&p, "Graviton3/SVE");
}

// ═══════════════════════════════════════════════════════════════════════
// Module 6: FMA Port Parameterization
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn fma_ports_zen5_has_3() {
    assert_eq!(MicroArch::Zen5.fma_ports(), 3);
}

#[test]
fn fma_ports_scalar_has_1() {
    assert_eq!(MicroArch::Scalar.fma_ports(), 1);
}

#[test]
fn fma_ports_mainstream_has_2() {
    for arch in &[
        MicroArch::Haswell, MicroArch::Broadwell,
        MicroArch::SkylakeClient, MicroArch::CometLake,
        MicroArch::SkylakeX, MicroArch::CascadeLake,
        MicroArch::IceLakeClient, MicroArch::IceLakeServer,
        MicroArch::TigerLake, MicroArch::AlderLake, MicroArch::RaptorLake,
        MicroArch::SapphireRapids, MicroArch::GraniteRapids,
        MicroArch::Zen2, MicroArch::Zen3, MicroArch::Zen4,
        MicroArch::GenericAvx512, MicroArch::GenericAvx2,
    ] {
        assert_eq!(arch.fma_ports(), 2, "{:?} should have 2 FMA ports", arch);
    }
}

#[test]
fn fma_ports_affects_peak_gflops() {
    // Zen5 (3 FMA ports) should have higher peak GFLOPS than Zen4 (2 FMA ports)
    // when core count and frequency are equal.
    let zen4_cfg = KernelConfig::from_arch_with_cache(MicroArch::Zen4, 32768, 1048576, 33554432);
    let zen5_cfg = KernelConfig::from_arch_with_cache(MicroArch::Zen5, 32768, 1048576, 33554432);

    let zen4_hw = make_hw_info("AuthenticAMD", "Zen4-sim", 8, 16, 32768, 1048576, 33554432,
        IsaFeatures { avx2: true, fma: true, ..isa_none() });
    let zen5_hw = make_hw_info("AuthenticAMD", "Zen5-sim", 8, 16, 32768, 1048576, 33554432,
        IsaFeatures { avx2: true, fma: true, avx512f: true, ..isa_none() });

    let zen4_p = make_profile(MicroArch::Zen4, IsaLevel::Avx2, 32768, 1048576, 33554432,
        zen4_hw, IsvCapabilities::default());
    let zen5_p = make_profile(MicroArch::Zen5, IsaLevel::Avx512, 32768, 1048576, 33554432,
        zen5_hw, IsvCapabilities::default());

    // Zen5 has 3 FMA ports vs Zen4's 2, and wider SIMD (16 vs 8)
    assert!(zen5_p.peak_gflops_f32 > zen4_p.peak_gflops_f32,
        "Zen5 peak {:.0} should exceed Zen4 peak {:.0}",
        zen5_p.peak_gflops_f32, zen4_p.peak_gflops_f32);
    eprintln!("[FMA] Zen4 peak={:.0} GFLOPS, Zen5 peak={:.0} GFLOPS",
        zen4_p.peak_gflops_f32, zen5_p.peak_gflops_f32);
}

// ═══════════════════════════════════════════════════════════════════════
// Module 7: AMX ISA Path Verification
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn amx_strategy_via_isa_not_isv() {
    // SPR profile: AMX is detected via ISA level, not ISV
    let p = profile_spr();
    assert_eq!(p.isa, IsaLevel::Avx512Amx);
    assert!(p.kernel_config.has_amx);

    // BF16 large → Amx (via ISA path)
    let s = select_gemm_strategy(64, 64, 64, DType::BF16, &p, None);
    assert_eq!(s, GemmStrategy::Amx);

    // F32 → not Amx (AMX requires BF16)
    let s = select_gemm_strategy(64, 64, 64, DType::F32, &p, None);
    assert_ne!(s, GemmStrategy::Amx);
}

#[test]
fn non_amx_isa_no_amx_strategy() {
    // Haswell: Avx2 ISA, has_amx=false → never selects Amx
    let p = profile_haswell();
    assert_eq!(p.isa, IsaLevel::Avx2);
    assert!(!p.kernel_config.has_amx);

    let s = select_gemm_strategy(64, 64, 64, DType::BF16, &p, None);
    assert_ne!(s, GemmStrategy::Amx, "Haswell should never select AMX strategy");
}
