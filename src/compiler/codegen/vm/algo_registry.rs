//! Template Registry — strategy → template selection (SPEC 27 REQ-AT-007)
//!
//! Selects the best AlgoTemplate for a given (strategy, device) pair.
//! Priority: highest DeviceReq satisfied by current hardware wins.

use crate::compiler::codegen::vm::algo_template::*;
use crate::compiler::codegen::vm::algo_templates;
use crate::dispatch::device_profile::{DeviceProfile, IsaLevel};

/// Map IsaLevel to a capability level for DeviceReq matching.
fn isa_capability_level(isa: &IsaLevel) -> u32 {
    match isa {
        IsaLevel::Scalar => 0,
        IsaLevel::Avx2 => 10,
        IsaLevel::Avx512 => 20,
        IsaLevel::Avx512Amx => 30,
        IsaLevel::Neon => 5,
        IsaLevel::Sve => 15,
        IsaLevel::Sve2 => 25,
        IsaLevel::NeonAmx => 30,
    }
}

/// Resolve a strategy to its registered template (single template per strategy).
/// Returns None if the strategy has no registered template.
fn resolve_template(strategy: &AlgoStrategy) -> Option<&'static AlgoTemplate> {
    match strategy {
        AlgoStrategy::GemmNaive => Some(&algo_templates::GEMM_NAIVE),
        AlgoStrategy::GemmBlis => Some(&algo_templates::GEMM_BLIS),
        AlgoStrategy::GemmHardwareTile => Some(&algo_templates::GEMM_AMX_TILE),
        AlgoStrategy::GemmGpuTiled => Some(&algo_templates::GEMM_GPU_TILED),
        AlgoStrategy::GemmGpuPipelined => Some(&algo_templates::GEMM_GPU_PIPELINED),
        AlgoStrategy::AttnMha => Some(&algo_templates::attention_norm_rope_moe::ATTN_MHA),
        AlgoStrategy::AttnGqa => Some(&algo_templates::attention_norm_rope_moe::ATTN_GQA),
        AlgoStrategy::AttnMla => Some(&algo_templates::attention_norm_rope_moe::ATTN_MLA),
        AlgoStrategy::NormRms => Some(&algo_templates::attention_norm_rope_moe::NORM_RMS),
        AlgoStrategy::NormLayer => Some(&algo_templates::attention_norm_rope_moe::NORM_LAYER),
        AlgoStrategy::RopeStandard => Some(&algo_templates::attention_norm_rope_moe::ROPE_STANDARD),
        AlgoStrategy::RopePartial => Some(&algo_templates::attention_norm_rope_moe::ROPE_PARTIAL),
        AlgoStrategy::MoeRouterTopk => Some(&algo_templates::attention_norm_rope_moe::MOE_ROUTER_TOPK),
        AlgoStrategy::MoePackedDispatch => Some(&algo_templates::attention_norm_rope_moe::MOE_PACKED_DISPATCH),
        AlgoStrategy::SamplingArgmax => Some(&algo_templates::sampling::SAMPLING_ARGMAX),
        AlgoStrategy::SamplingTemperature => Some(&algo_templates::sampling::SAMPLING_TEMPERATURE),
        AlgoStrategy::SamplingSoftmax => Some(&algo_templates::sampling::SAMPLING_SOFTMAX),
        AlgoStrategy::SamplingTopK => Some(&algo_templates::sampling::SAMPLING_TOP_K),
        AlgoStrategy::SamplingTopP => Some(&algo_templates::sampling::SAMPLING_TOP_P),
        AlgoStrategy::SamplingMultinomial => Some(&algo_templates::sampling::SAMPLING_MULTINOMIAL),
        _ => None,
    }
}

/// Select the best template for a given strategy and device profile.
///
/// Checks that the template's DeviceReq is satisfied by the current device.
/// Returns None if no template matches or device requirements aren't met.
pub fn select_template(
    strategy: &AlgoStrategy,
    profile: &DeviceProfile,
) -> Option<&'static AlgoTemplate> {
    let cap_level = isa_capability_level(&profile.isa);
    let tmpl = resolve_template(strategy)?;
    if tmpl.device_req.is_satisfied_by(cap_level) {
        Some(tmpl)
    } else {
        None
    }
}

/// List all strategies that have registered templates.
pub fn available_strategies() -> Vec<AlgoStrategy> {
    vec![
        AlgoStrategy::GemmNaive,
        AlgoStrategy::GemmBlis,
        AlgoStrategy::GemmHardwareTile,
        AlgoStrategy::GemmGpuTiled,
        AlgoStrategy::GemmGpuPipelined,
        AlgoStrategy::AttnMha,
        AlgoStrategy::AttnGqa,
        AlgoStrategy::AttnMla,
        AlgoStrategy::NormRms,
        AlgoStrategy::NormLayer,
        AlgoStrategy::RopeStandard,
        AlgoStrategy::RopePartial,
        AlgoStrategy::MoeRouterTopk,
        AlgoStrategy::MoePackedDispatch,
        AlgoStrategy::SamplingArgmax,
        AlgoStrategy::SamplingTemperature,
        AlgoStrategy::SamplingSoftmax,
        AlgoStrategy::SamplingTopK,
        AlgoStrategy::SamplingTopP,
        AlgoStrategy::SamplingMultinomial,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemnaive_matches_any_device() {
        let profile = DeviceProfile::detect();
        let tmpl = select_template(&AlgoStrategy::GemmNaive, &profile);
        assert!(tmpl.is_some());
        assert_eq!(tmpl.unwrap().name, "GEMM_NAIVE");
    }

    #[test]
    fn test_norm_rms_registered() {
        let tmpl = resolve_template(&AlgoStrategy::NormRms);
        assert!(tmpl.is_some());
        assert_eq!(tmpl.unwrap().name, "NORM_RMS");
    }

    #[test]
    fn test_available_strategies_nonempty() {
        let strategies = available_strategies();
        assert!(!strategies.is_empty());
        assert!(strategies.contains(&AlgoStrategy::GemmNaive));
        assert!(strategies.contains(&AlgoStrategy::NormRms));
    }

    // ── resolve_template coverage for all strategies ──

    #[test]
    fn test_resolve_template_all_gemm_strategies() {
        for strategy in [
            AlgoStrategy::GemmNaive,
            AlgoStrategy::GemmBlis,
            AlgoStrategy::GemmHardwareTile,
            AlgoStrategy::GemmGpuTiled,
            AlgoStrategy::GemmGpuPipelined,
        ] {
            assert!(resolve_template(&strategy).is_some(), "{:?} should resolve", strategy);
        }
    }

    #[test]
    fn test_resolve_template_all_attention_strategies() {
        for strategy in [
            AlgoStrategy::AttnMha,
            AlgoStrategy::AttnGqa,
            AlgoStrategy::AttnMla,
        ] {
            assert!(resolve_template(&strategy).is_some(), "{:?} should resolve", strategy);
        }
    }

    #[test]
    fn test_resolve_template_all_norm_strategies() {
        for strategy in [AlgoStrategy::NormRms, AlgoStrategy::NormLayer] {
            let tmpl = resolve_template(&strategy).unwrap_or_else(|| panic!("{:?} should resolve", strategy));
            assert!(tmpl.name.contains("NORM"), "{}", tmpl.name);
        }
    }

    #[test]
    fn test_resolve_template_rope_strategies() {
        for strategy in [AlgoStrategy::RopeStandard, AlgoStrategy::RopePartial] {
            assert!(resolve_template(&strategy).is_some(), "{:?} should resolve", strategy);
        }
    }

    #[test]
    fn test_resolve_template_moe_strategies() {
        for strategy in [AlgoStrategy::MoeRouterTopk, AlgoStrategy::MoePackedDispatch] {
            assert!(resolve_template(&strategy).is_some(), "{:?} should resolve", strategy);
        }
    }

    #[test]
    fn test_resolve_template_all_sampling_strategies() {
        for strategy in [
            AlgoStrategy::SamplingArgmax,
            AlgoStrategy::SamplingTemperature,
            AlgoStrategy::SamplingSoftmax,
            AlgoStrategy::SamplingTopK,
            AlgoStrategy::SamplingTopP,
            AlgoStrategy::SamplingMultinomial,
        ] {
            assert!(resolve_template(&strategy).is_some(), "{:?} should resolve", strategy);
        }
    }

    #[test]
    fn test_resolve_template_quant_gather_returns_none() {
        // QuantGather is not in resolve_template match — should return None
        assert!(resolve_template(&AlgoStrategy::QuantGather).is_none());
    }

    // ── isa_capability_level ordering ──

    #[test]
    fn test_isa_capability_level_scalar_is_zero() {
        assert_eq!(isa_capability_level(&IsaLevel::Scalar), 0);
    }

    #[test]
    fn test_isa_capability_level_ordering() {
        assert!(isa_capability_level(&IsaLevel::Avx2) < isa_capability_level(&IsaLevel::Avx512));
        assert!(isa_capability_level(&IsaLevel::Avx512) < isa_capability_level(&IsaLevel::Avx512Amx));
        assert!(isa_capability_level(&IsaLevel::Neon) < isa_capability_level(&IsaLevel::Sve));
        assert!(isa_capability_level(&IsaLevel::Sve) < isa_capability_level(&IsaLevel::Sve2));
    }

    #[test]
    fn test_isa_capability_level_amx_tie() {
        assert_eq!(isa_capability_level(&IsaLevel::Avx512Amx), isa_capability_level(&IsaLevel::NeonAmx));
    }

    // ── select_template integration ──

    #[test]
    fn test_select_template_blis_with_scalar_profile() {
        let profile = DeviceProfile { isa: IsaLevel::Scalar, ..DeviceProfile::detect() };
        let tmpl = select_template(&AlgoStrategy::GemmBlis, &profile);
        // GemmBlis has DeviceReq::Avx2Plus, scalar should not satisfy
        assert!(tmpl.is_none(), "GemmBlis should not match scalar ISA");
    }

    #[test]
    fn test_select_template_naive_matches_scalar() {
        let profile = DeviceProfile { isa: IsaLevel::Scalar, ..DeviceProfile::detect() };
        let tmpl = select_template(&AlgoStrategy::GemmNaive, &profile);
        assert!(tmpl.is_some(), "GemmNaive should match scalar ISA");
    }

    #[test]
    fn test_select_template_norm_rms_matches_scalar() {
        let profile = DeviceProfile { isa: IsaLevel::Scalar, ..DeviceProfile::detect() };
        let tmpl = select_template(&AlgoStrategy::NormRms, &profile);
        assert!(tmpl.is_some(), "NormRms should match scalar ISA");
    }
}
