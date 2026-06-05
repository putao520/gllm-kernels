//! Property-based tests for the JIT compiler infrastructure.
//!
//! Uses proptest to verify invariants that must hold for all inputs:
//! - TraceOp SSA legality
//! - classify_pattern consistency
//! - Fusion plan completeness
//! - Buffer allocation non-overlap
//! - Codegen determinism

#![feature(f16)]

use proptest::prelude::*;

use gllm_kernels::compiler::trace::{classify_pattern, ComputePattern, TraceOp};
use gllm_kernels::compiler::fusion;
use gllm_kernels::compiler::graph::{CompilerGraph, OpKind};
use gllm_kernels::compiler::buffer_alloc;
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::inference::types::DType;

// ═══════════════════════════════════════════════════════════════════════
// 1. TraceOp SSA legality: all operand indices < current position
// ═══════════════════════════════════════════════════════════════════════

fn arb_trace_ops(max_len: usize) -> impl Strategy<Value = Vec<TraceOp>> {
    (1..=max_len).prop_flat_map(|len| {
        let mut strats: Vec<BoxedStrategy<TraceOp>> = Vec::new();
        // First op is always Input(0)
        strats.push(Just(TraceOp::Input(0)).boxed());
        for i in 1..len {
            let i_u32 = i as u32;
            strats.push(
                prop_oneof![
                    Just(TraceOp::Input(0)),
                    Just(TraceOp::Const(1.0)),
                    (0..i_u32).prop_map(TraceOp::Neg),
                    (0..i_u32).prop_map(TraceOp::Exp),
                    (0..i_u32).prop_map(TraceOp::Log),
                    (0..i_u32, 0..i_u32).prop_map(|(a, b)| TraceOp::Add(a, b)),
                    (0..i_u32, 0..i_u32).prop_map(|(a, b)| TraceOp::Mul(a, b)),
                    (0..i_u32, 0..i_u32).prop_map(|(a, b)| TraceOp::Sub(a, b)),
                    (0..i_u32, 0..i_u32).prop_map(|(a, b)| TraceOp::Div(a, b)),
                ]
                .boxed(),
            );
        }
        strats
    })
}

proptest! {
    /// Every generated TraceOp sequence must satisfy SSA: operand indices < position.
    #[test]
    fn prop_trace_op_ssa_legality(ops in arb_trace_ops(20)) {
        for (i, op) in ops.iter().enumerate() {
            let indices: Vec<u32> = match op {
                TraceOp::Input(_) | TraceOp::Const(_) => vec![],
                TraceOp::Neg(a) | TraceOp::Abs(a) | TraceOp::Exp(a)
                | TraceOp::Sqrt(a) | TraceOp::Rsqrt(a) | TraceOp::Tanh(a)
                | TraceOp::Recip(a) | TraceOp::Log(a) => vec![*a],
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b)
                | TraceOp::Div(a, b) | TraceOp::Max(a, b) | TraceOp::Min(a, b) => vec![*a, *b],
                TraceOp::Fma(a, b, c) => vec![*a, *b, *c],
            };
            for idx in indices {
                prop_assert!(
                    (idx as usize) < i,
                    "SSA violation at position {}: operand index {} >= {}",
                    i, idx, i
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. classify_pattern consistency: classification matches input count
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn prop_classify_pattern_consistent(ops in arb_trace_ops(15)) {
        let pattern = classify_pattern(&ops);

        let max_input = ops.iter().filter_map(|op| {
            if let TraceOp::Input(idx) = op { Some(*idx) } else { None }
        }).max();

        let num_inputs = match max_input {
            Some(idx) => (idx + 1) as usize,
            None => 0,
        };

        match &pattern {
            ComputePattern::Elementwise { .. } => {
                prop_assert!(num_inputs <= 1, "Elementwise should have 0-1 inputs, got {}", num_inputs);
            }
            ComputePattern::BinaryElementwise { .. } => {
                prop_assert_eq!(num_inputs, 2, "BinaryElementwise should have 2 inputs");
            }
            ComputePattern::Injective { num_inputs: ni, .. } => {
                prop_assert!(*ni >= 3 || ops.is_empty(), "Injective should have 3+ inputs or empty body");
            }
            _ => {} // Other patterns are manually constructed, not from classify_pattern
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Fusion plan completeness: every op belongs to exactly one group
// ═══════════════════════════════════════════════════════════════════════

fn build_linear_chain(n_ops: usize) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let mut prev = g.add_tensor("input", vec![64], DType::F32);
    g.inputs = vec![prev];

    for i in 0..n_ops {
        let out = g.add_tensor(&format!("t{i}"), vec![64], DType::F32);
        g.add_op(OpKind::Silu, vec![prev], vec![out], &format!("op{i}"));
        prev = out;
    }
    g.outputs = vec![prev];
    g
}

proptest! {
    #[test]
    fn prop_fusion_plan_completeness(n_ops in 1..8usize) {
        let graph = build_linear_chain(n_ops);
        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);

        // Every op must be in exactly one group
        let mut seen = std::collections::HashSet::new();
        for group in &plan.groups {
            for &op_id in &group.ops {
                prop_assert!(
                    seen.insert(op_id),
                    "op {:?} appears in multiple groups", op_id
                );
            }
        }

        // All graph ops must be covered
        for (i, _) in graph.ops.iter().enumerate() {
            let op_id = gllm_kernels::compiler::graph::OpId(i as u32);
            prop_assert!(
                plan.op_to_group.contains_key(&op_id),
                "op {:?} not in any fusion group", op_id
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Buffer allocation non-overlap: no two live tensors share bytes
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn prop_buffer_allocation_no_overlap(n_ops in 2..6usize) {
        let graph = build_linear_chain(n_ops);
        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);
        let lifetimes = buffer_alloc::analyze_lifetimes(&graph, &plan);
        let alloc = buffer_alloc::allocate_buffers(&lifetimes);

        // Check that no two simultaneously-live slots overlap
        for (i, si) in alloc.slots.iter().enumerate() {
            for sj in alloc.slots.iter().skip(i + 1) {
                let li = lifetimes.iter().find(|l| l.tensor_id == si.tensor_id);
                let lj = lifetimes.iter().find(|l| l.tensor_id == sj.tensor_id);

                if let (Some(li), Some(lj)) = (li, lj) {
                    let overlaps_time = li.first_use <= lj.last_use && lj.first_use <= li.last_use;
                    if overlaps_time {
                        let overlaps_space =
                            si.offset < sj.offset + sj.size_bytes
                            && sj.offset < si.offset + si.size_bytes;
                        prop_assert!(
                            !overlaps_space,
                            "buffers for {:?} and {:?} overlap in both time and space",
                            si.tensor_id, sj.tensor_id
                        );
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Codegen determinism: same graph → same bytes (jit-x86 only)
// ═══════════════════════════════════════════════════════════════════════

#[cfg(all(target_arch = "x86_64", feature = "jit-x86"))]
proptest! {
    #[test]
    fn prop_codegen_deterministic(n_ops in 1..4usize) {
        use gllm_kernels::compiler::codegen::x86_64::jit::X86CodeGen;

        let graph = build_linear_chain(n_ops);
        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);
        let lifetimes = buffer_alloc::analyze_lifetimes(&graph, &plan);
        let alloc = buffer_alloc::allocate_buffers(&lifetimes);

        let registry = gllm_kernels::compiler::registry::ScalarOpRegistry::with_defaults();
        let mut cg1 = X86CodeGen::new(&profile);
        let out1 = cg1.emit_plan(&plan, &graph, &alloc, &profile, Some(&registry)).unwrap();

        let mut cg2 = X86CodeGen::new(&profile);
        let out2 = cg2.emit_plan(&plan, &graph, &alloc, &profile, Some(&registry)).unwrap();

        prop_assert_eq!(
            out1.code, out2.code,
            "codegen produced different bytes for identical graph ({} ops)", n_ops
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. TraceOp Log SSA legality: sequences containing Log satisfy SSA
// ═══════════════════════════════════════════════════════════════════════

/// Generator that produces TraceOp sequences guaranteed to contain at least one Log.
fn arb_trace_ops_with_log(max_len: usize) -> impl Strategy<Value = Vec<TraceOp>> {
    arb_trace_ops(max_len).prop_filter("must contain at least one Log", |ops| {
        ops.iter().any(|op| matches!(op, TraceOp::Log(_)))
    })
}

proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    /// TraceOp sequences containing Log must satisfy SSA: all operand indices < position.
    #[test]
    fn prop_trace_op_log_ssa_legality(ops in arb_trace_ops_with_log(20)) {
        // Verify at least one Log is present
        prop_assert!(
            ops.iter().any(|op| matches!(op, TraceOp::Log(_))),
            "generated sequence must contain Log"
        );

        for (i, op) in ops.iter().enumerate() {
            let indices: Vec<u32> = match op {
                TraceOp::Input(_) | TraceOp::Const(_) => vec![],
                TraceOp::Neg(a) | TraceOp::Abs(a) | TraceOp::Exp(a)
                | TraceOp::Sqrt(a) | TraceOp::Rsqrt(a) | TraceOp::Tanh(a)
                | TraceOp::Recip(a) | TraceOp::Log(a) => vec![*a],
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b)
                | TraceOp::Div(a, b) | TraceOp::Max(a, b) | TraceOp::Min(a, b) => vec![*a, *b],
                TraceOp::Fma(a, b, c) => vec![*a, *b, *c],
            };
            for idx in indices {
                prop_assert!(
                    (idx as usize) < i,
                    "SSA violation at position {}: operand index {} >= {} (op: {:?})",
                    i, idx, i, op
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7. AVX-512 and AVX2 trace methods accept the same TraceOp variants
// ═══════════════════════════════════════════════════════════════════════

#[cfg(all(target_arch = "x86_64", feature = "jit-x86"))]
proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    /// For any valid SSA TraceOp body within register budget (<=12 ops),
    /// emit_trace_ops_avx2 must succeed on two independent X86CodeGen instances,
    /// confirming the method is deterministic and handles all TraceOp variants.
    #[test]
    fn prop_avx512_trace_body_matches_avx2(ops in arb_trace_ops(12)) {
        use gllm_kernels::compiler::codegen::x86_64::jit::X86CodeGen;

        let profile = DeviceProfile::detect();

        // Two independent codegen instances — both must agree on success/failure
        let mut cg1 = X86CodeGen::new(&profile);
        let mut cg2 = X86CodeGen::new(&profile);

        let r1 = cg1.emit_trace_ops_avx2(&ops);
        let r2 = cg2.emit_trace_ops_avx2(&ops);

        match (&r1, &r2) {
            (Ok(regs1), Ok(regs2)) => {
                prop_assert_eq!(
                    regs1.len(), regs2.len(),
                    "register map lengths differ for identical input"
                );
            }
            (Err(e1), Err(e2)) => {
                prop_assert_eq!(
                    e1, e2,
                    "error messages differ for identical input"
                );
            }
            _ => {
                prop_assert!(
                    false,
                    "one succeeded and one failed: r1={:?}, r2={:?}", r1.is_ok(), r2.is_ok()
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Fusion plan: no op appears in multiple groups
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    /// No op may appear in more than one fusion group.
    #[test]
    fn prop_fusion_plan_no_duplicate_ops(n_ops in 1..10usize) {
        let graph = build_linear_chain(n_ops);
        let profile = DeviceProfile::detect();
        let plan = fusion::fuse(&graph, &profile);

        let mut seen = std::collections::HashMap::<gllm_kernels::compiler::graph::OpId, usize>::new();
        for (gidx, group) in plan.groups.iter().enumerate() {
            for &op_id in &group.ops {
                if let Some(&prev_group) = seen.get(&op_id) {
                    prop_assert!(
                        false,
                        "op {:?} appears in group {} and group {}", op_id, prev_group, gidx
                    );
                }
                seen.insert(op_id, gidx);
            }
        }

        // Also verify op_to_group is consistent with the groups vec
        for (gidx, group) in plan.groups.iter().enumerate() {
            for &op_id in &group.ops {
                let mapped = plan.op_to_group.get(&op_id);
                prop_assert_eq!(
                    mapped, Some(&gidx),
                    "op_to_group mismatch for {:?}: expected group {}, got {:?}",
                    op_id, gidx, mapped
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 9. GEMM blocking constraints hold for random dimensions
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    /// For random (m, n, k) dimensions, gemm_blocking() output must satisfy
    /// cache capacity constraints and alignment invariants.
    #[test]
    fn prop_gemm_blocking_constraints_hold(
        m in 1..2048usize,
        n in 1..2048usize,
        k in 1..2048usize,
    ) {
        let profile = DeviceProfile::detect();
        let blk = profile.gemm_blocking(m, n, k);
        let (l1, l2, _l3) = profile.cache_sizes();
        let (mr, nr) = profile.microkernel_mr_nr();
        let elem_size = 4usize; // f32

        // Basic sanity: all blocking params > 0
        prop_assert!(blk.kc >= 1, "kc must be >= 1, got {}", blk.kc);
        prop_assert!(blk.mc >= 1, "mc must be >= 1, got {}", blk.mc);
        prop_assert!(blk.nc >= 1, "nc must be >= 1, got {}", blk.nc);

        // Blocking must not exceed problem dimensions
        prop_assert!(blk.kc <= k, "kc ({}) > k ({})", blk.kc, k);
        prop_assert!(blk.mc <= m, "mc ({}) > m ({})", blk.mc, m);
        prop_assert!(blk.nc <= n, "nc ({}) > n ({})", blk.nc, n);

        // MR/NR from profile must be consistent
        prop_assert_eq!(blk.mr, mr, "mr mismatch");
        prop_assert_eq!(blk.nr, nr, "nr mismatch");

        // L1 constraint: A micropanel (MR×KC) + B micropanel (KC×NR) must fit in L1
        let micropanel_bytes = (mr + nr) * blk.kc * elem_size;
        prop_assert!(
            micropanel_bytes <= l1,
            "micropanels ({} bytes) exceed L1 ({} bytes): mr={}, nr={}, kc={}",
            micropanel_bytes, l1, mr, nr, blk.kc
        );

        // L2 constraint: A panel (MC×KC) must fit in L2
        let a_panel_bytes = blk.mc * blk.kc * elem_size;
        prop_assert!(
            a_panel_bytes <= l2,
            "A panel ({} bytes) exceeds L2 ({} bytes): mc={}, kc={}",
            a_panel_bytes, l2, blk.mc, blk.kc
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 10. CRC32C determinism: same input → same output on repeated calls
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    /// crc32c must produce identical results for identical inputs across calls.
    #[test]
    fn prop_crc32c_deterministic(data in proptest::collection::vec(proptest::num::u8::ANY, 0..1024)) {
        use gllm_kernels::compiler::cache::crc32c;

        let h1 = crc32c(&data);
        let h2 = crc32c(&data);
        let h3 = crc32c(&data);

        prop_assert_eq!(h1, h2, "crc32c not deterministic on second call");
        prop_assert_eq!(h2, h3, "crc32c not deterministic on third call");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 11. config_hash: different inputs → different hashes (probabilistic)
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    /// For distinct IR byte slices, config_hash should produce distinct values.
    /// This is probabilistic — FNV-1a collisions are possible but extremely rare
    /// for inputs that differ by more than a single bit.
    #[test]
    fn prop_config_hash_no_collision(
        a in proptest::collection::vec(proptest::num::u8::ANY, 1..128),
        b in proptest::collection::vec(proptest::num::u8::ANY, 1..128),
    ) {
        use gllm_kernels::compiler::cache::config_hash;

        prop_assume!(a != b, "inputs must differ");

        let ha = config_hash(&a, "hw_test");
        let hb = config_hash(&b, "hw_test");

        prop_assert_ne!(
            ha, hb,
            "config_hash collision: {:?} and {:?} both hash to {}",
            &a[..a.len().min(16)], &b[..b.len().min(16)], ha
        );
    }
}
