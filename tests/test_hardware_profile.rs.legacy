//! Unit tests for hardware profile detection and fusion constraints.

use gllm_kernels::compiler::{HardwareProfile, FusionEngine};
use gllm_kernels::dispatch::DeviceProfile;

#[test]
fn test_hardware_profile_detection() {
    let profile = DeviceProfile::detect();
    let hw_profile = HardwareProfile::detect(&profile);

    // Should detect a specific profile (not Generic on real hardware)
    assert_ne!(hw_profile, HardwareProfile::Generic);
}

#[test]
fn test_fusion_aggressiveness_ordering() {
    // GPU profiles should be more aggressive than CPU
    assert!(HardwareProfile::CudaSM100.fusion_aggressiveness() >
            HardwareProfile::CpuAvx512.fusion_aggressiveness());
    assert!(HardwareProfile::CudaSM90.fusion_aggressiveness() >
            HardwareProfile::CpuAvx2.fusion_aggressiveness());
    assert!(HardwareProfile::RocmMI300.fusion_aggressiveness() >
            HardwareProfile::ArmNeoverse.fusion_aggressiveness());
}

#[test]
fn test_min_fusion_benefit_ordering() {
    // CPU should require higher benefit threshold than GPU
    assert!(HardwareProfile::CpuAvx2.min_fusion_benefit() >
            HardwareProfile::CudaSM90.min_fusion_benefit());
    assert!(HardwareProfile::CpuAvx512.min_fusion_benefit() >
            HardwareProfile::RocmMI300.min_fusion_benefit());
}

#[test]
fn test_max_fusion_depth_ordering() {
    // GPU should allow deeper fusion than CPU
    assert!(HardwareProfile::CudaSM100.max_fusion_depth() >
            HardwareProfile::CpuAvx2.max_fusion_depth());
    assert!(HardwareProfile::CudaSM90.max_fusion_depth() >
            HardwareProfile::CpuAvx512.max_fusion_depth());

    // Verify specific values
    assert_eq!(HardwareProfile::CudaSM100.max_fusion_depth(), 8);
    assert_eq!(HardwareProfile::CpuAvx2.max_fusion_depth(), 3);
}

#[test]
fn test_prefer_gemm_fusion() {
    // GPU and Apple Silicon prefer GEMM fusion
    assert!(HardwareProfile::CudaSM90.prefer_gemm_fusion());
    assert!(HardwareProfile::RocmMI300.prefer_gemm_fusion());
    assert!(HardwareProfile::AppleM3.prefer_gemm_fusion());

    // CPU x86 and ARM do not
    assert!(!HardwareProfile::CpuAvx512.prefer_gemm_fusion());
    assert!(!HardwareProfile::CpuAvx2.prefer_gemm_fusion());
    assert!(!HardwareProfile::ArmNeoverse.prefer_gemm_fusion());
}

#[test]
fn test_fusion_engine_with_hardware_profile() {
    let profile = DeviceProfile::detect();
    let engine = FusionEngine::with_hardware_profile(&profile);

    let hw_profile = engine.hardware_profile();

    // Verify constraints are reasonable
    assert!(hw_profile.max_fusion_depth() >= 2);
    assert!(hw_profile.max_fusion_depth() <= 8);
    assert!(hw_profile.min_fusion_benefit() >= 1.0);
    assert!(hw_profile.min_fusion_benefit() <= 2.0);
    assert!(hw_profile.fusion_aggressiveness() >= 0.0);
    assert!(hw_profile.fusion_aggressiveness() <= 1.0);
}

#[test]
fn test_all_profiles_have_valid_constraints() {
    let profiles = [
        HardwareProfile::CudaSM80,
        HardwareProfile::CudaSM90,
        HardwareProfile::CudaSM100,
        HardwareProfile::RocmMI200,
        HardwareProfile::RocmMI300,
        HardwareProfile::CpuAvx2,
        HardwareProfile::CpuAvx512,
        HardwareProfile::CpuAvx10_2,
        HardwareProfile::AppleM1,
        HardwareProfile::AppleM2,
        HardwareProfile::AppleM3,
        HardwareProfile::ArmNeoverse,
        HardwareProfile::Generic,
    ];

    for profile in &profiles {
        // All profiles should have valid constraints
        assert!(profile.max_fusion_depth() >= 2, "{:?} depth too low", profile);
        assert!(profile.max_fusion_depth() <= 8, "{:?} depth too high", profile);
        assert!(profile.min_fusion_benefit() >= 1.0, "{:?} benefit < 1.0", profile);
        assert!(profile.min_fusion_benefit() <= 2.0, "{:?} benefit too high", profile);
        assert!(profile.fusion_aggressiveness() >= 0.0, "{:?} aggressiveness < 0", profile);
        assert!(profile.fusion_aggressiveness() <= 1.0, "{:?} aggressiveness > 1", profile);
    }
}
