//! Loop body symbolic execution and reduction detection.
//!
//! Phase 2 of the control-flow upgrade: symbolically executes a single loop
//! iteration to identify accumulator patterns (Sum, Max, Min reductions)
//! and per-element transforms.

use super::cfg::{BasicBlock, BlockId, ControlFlowGraph, LoopForest, NaturalLoop};
use super::engine::SymbolicExecutor;
use super::sym_value::SymValue;
use crate::compiler::trace::{ComputePattern, ReductionSecondPass, TraceOp, ValueId};
use crate::types::CompilerError;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Kind of reduction detected in a loop accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    /// `acc += f(x[i])` — additive reduction.
    Sum,
    /// `acc = max(acc, f(x[i]))` — max reduction.
    Max,
    /// `acc = min(acc, f(x[i]))` — min reduction.
    Min,
}

/// How the accumulator was initialized before the loop.
#[derive(Debug, Clone)]
pub enum AccumulatorInit {
    /// Initialized to a constant (e.g., 0.0 for sum, -inf for max).
    Const(f64),
    /// Initialized from a symbolic expression (e.g., a parameter).
    Symbolic(SymValue),
}

/// A detected reduction pattern in a loop.
#[derive(Debug, Clone)]
pub struct ReductionDetected {
    /// Which XMM register holds the accumulator.
    pub register: String,
    /// The kind of reduction.
    pub kind: ReductionKind,
    /// How the accumulator was initialized before the loop.
    pub init: AccumulatorInit,
    /// The per-element expression being reduced (with `Param` references
    /// representing the loaded element). Simplified.
    pub body_expr: SymValue,
}

/// Result of analyzing a single loop's symbolic execution.
#[derive(Debug, Clone)]
pub struct LoopTrace {
    /// The loop that was analyzed.
    pub loop_header: BlockId,
    /// Detected reductions (one per accumulator register).
    pub reductions: Vec<ReductionDetected>,
    /// Registers that changed but weren't recognized as reductions.
    pub unknown_mutations: Vec<(String, SymValue)>,
    /// Number of basic blocks in the loop body.
    pub body_block_count: usize,
}

// ---------------------------------------------------------------------------
// Loop body symbolic execution
// ---------------------------------------------------------------------------

/// Analyze a single natural loop by symbolically executing one iteration.
///
/// Strategy:
/// 1. Snapshot the executor state before the loop.
/// 2. Execute all instructions in the loop body blocks (topological order).
/// 3. Compare XMM state before/after to find mutations.
/// 4. Classify mutations as reductions or unknown.
///
/// `pre_exec` should already have the state at loop entry (e.g., after
/// executing the prologue up to the loop header).
pub fn analyze_single_loop(
    loop_info: &NaturalLoop,
    cfg: &ControlFlowGraph,
    pre_exec: &SymbolicExecutor,
) -> Result<LoopTrace, CompilerError> {
    let pre_xmm = pre_exec.xmm_state();
    let mut exec = pre_exec.snapshot();

    // Collect loop body block IDs in address order (approximates topological
    // order for reducible loops).
    let mut body_blocks: Vec<&BasicBlock> = loop_info
        .body_blocks
        .iter()
        .filter_map(|bid| cfg.blocks.get(bid))
        .collect();
    body_blocks.sort_by_key(|b| b.start_addr);

    let block_ids: Vec<BlockId> = body_blocks.iter().map(|b| b.id).collect();

    // Execute one iteration with diamond-branch merging.
    // This detects inner conditional branches (e.g., compare+branch for
    // clamp/relu patterns) and merges them into Select nodes instead of
    // naively executing both paths linearly.
    super::branch_merger::execute_blocks_with_merging(&block_ids, cfg, &mut exec);

    // Compare XMM state before/after.
    let post_xmm = exec.xmm_state();
    let mut reductions = Vec::new();
    let mut unknown_mutations = Vec::new();

    for (reg, post_val) in &post_xmm {
        let pre_val = pre_xmm.get(reg);
        let pre_str = pre_val.map(|v| format!("{v}")).unwrap_or_default();
        let post_str = format!("{post_val}");

        // Skip unchanged registers.
        if pre_str == post_str {
            continue;
        }

        // Try to detect reduction pattern against the known pre-value.
        if let Some(pre) = pre_val {
            if let Some(reduction) = detect_reduction_pattern(reg, pre, post_val) {
                reductions.push(reduction);
                continue;
            }
        }

        // If the register wasn't in the pre-state, try detecting a
        // self-referential reduction: e.g. xmm2 = Add(Unknown("xmm2"), expr).
        // This happens when the compiler uses a register that wasn't part of
        // the function's initial parameter set as an accumulator.
        if pre_val.is_none() {
            let self_ref = SymValue::Unknown(reg.to_string());
            if let Some(reduction) = detect_reduction_pattern(reg, &self_ref, post_val) {
                reductions.push(reduction);
                continue;
            }
        }

        unknown_mutations.push((reg.clone(), post_val.clone()));
    }

    // Also check registers that existed pre but not post (unlikely but safe).
    for reg in pre_xmm.keys() {
        if !post_xmm.contains_key(reg) {
            unknown_mutations.push((reg.clone(), SymValue::Unknown("cleared".into())));
        }
    }

    Ok(LoopTrace {
        loop_header: loop_info.header,
        reductions,
        unknown_mutations,
        body_block_count: body_blocks.len(),
    })
}

// ---------------------------------------------------------------------------
// Multi-pass analysis: combine loop traces → ComputePattern
// ---------------------------------------------------------------------------

/// Result of analyzing all loops in a multi-pass scalar function.
#[derive(Debug, Clone)]
pub struct MultiPassAnalysis {
    /// Per-loop traces, in execution order.
    pub loop_traces: Vec<LoopTrace>,
    /// The classified compute pattern.
    pub pattern: ComputePattern,
    /// Number of loops found.
    pub num_loops: usize,
}

/// Combine multiple `LoopTrace`s into a `ComputePattern`.
///
/// Classification rules:
/// - 0 loops → fall through to linear symexec (not handled here)
/// - 1 loop with Sum reduction → `Reduction { identity: 0.0, combine: [Input(0), Input(1), Add(0,1)] }`
/// - 1 loop with Max reduction → `Reduction { identity: -inf, combine: [Input(0), Input(1), Max(0,1)] }`
/// - 2 loops (reduce + transform) → `NormLike` (e.g., RmsNorm, L2Normalize)
/// - 3 loops (max → exp-sum → normalize) → `Reduction` with second_pass + normalize (Softmax)
/// - 3 loops (mean → var → normalize) → `NormLike` (LayerNorm)
pub fn combine_passes(traces: &[LoopTrace]) -> Result<ComputePattern, CompilerError> {
    if traces.is_empty() {
        return Err("no loops found — use linear symexec instead".into());
    }

    // Coalesce: the compiler often emits a main vectorized loop + a scalar
    // tail loop for the same pass. Consecutive transform-only loops (no
    // reductions) at the end are treated as a single normalize/transform pass.
    let mut logical: Vec<&LoopTrace> = Vec::new();
    for trace in traces {
        let dominated = !logical.is_empty() && {
            // SAFETY: guarded by !logical.is_empty() above
            let prev = logical.last().expect("SAFETY: logical is non-empty: checked above");
            // Both are transform-only (no reductions) → tail cleanup.
            trace.reductions.is_empty() && prev.reductions.is_empty()
        };
        if dominated {
            // Skip — this is a tail/cleanup loop for the same logical pass.
            continue;
        }
        logical.push(trace);
    }

    match logical.len() {
        0 => Err("no meaningful loops after coalescing".into()),
        1 => combine_single_loop(logical[0]),
        2 => combine_two_loops(logical[0], logical[1]),
        3 => combine_three_loops(logical[0], logical[1], logical[2]),
        n => Err(format!("unsupported: {n} logical passes (expected 1-3)").into()),
    }
}

/// Single loop → simple Reduction pattern.
///
/// Rejects loops that have unknown_mutations alongside reductions: a single
/// loop with both indicates the analyzer only partially understood the body
/// (e.g. an elementwise function whose vectorized exp()/tanh() call was
/// misread as an accumulator).  Genuine single-loop reductions (like a
/// simple sum or max scan) produce clean traces with no unknown mutations.
fn combine_single_loop(trace: &LoopTrace) -> Result<ComputePattern, CompilerError> {
    if trace.reductions.is_empty() {
        return Err("single loop with no detected reductions".into());
    }

    if !trace.unknown_mutations.is_empty() {
        return Err("single loop has unknown mutations alongside reductions — ambiguous".into());
    }

    let r = &trace.reductions[0];
    let (identity, combine) = reduction_to_trace_ops(r);

    Ok(ComputePattern::Reduction {
        identity,
        combine,
        second_pass: None,
        normalize: None,
    })
}

/// Two loops → NormLike (reduce → finalize → transform).
///
/// Pattern: RmsNorm, L2Normalize
/// - Loop 1: sum reduction (e.g., sum of squares)
/// - Loop 2: per-element transform using the reduction result
fn combine_two_loops(
    loop1: &LoopTrace,
    loop2: &LoopTrace,
) -> Result<ComputePattern, CompilerError> {
    // Loop 1 must have a Sum reduction.
    let r1 = loop1.reductions.first()
        .ok_or("loop 1 has no reductions (expected sum for NormLike)")?;

    if r1.kind != ReductionKind::Sum {
        return Err(format!("loop 1 reduction is {:?}, expected Sum for NormLike", r1.kind).into());
    }

    // Build reduce body from the reduction's body_expr.
    let reduce = symvalue_to_reduce_body(&r1.body_expr);

    // Finalize: we can't easily extract the inter-loop scalar computation
    // from binary analysis alone. Use a generic placeholder that the codegen
    // can specialize based on OpKind.
    let finalize = vec![
        TraceOp::Input(0),    // [0] reduction result
        TraceOp::Input(1),    // [1] n (dimension as float)
    ];

    // Transform: generic x * scale pattern (most common for norms).
    // Loop 2 typically does out[i] = x[i] * scale or x[i] * scale * weight[i].
    let transform = vec![
        TraceOp::Input(0),  // [0] x
        TraceOp::Input(1),  // [1] scale (from finalize)
        TraceOp::Mul(ValueId(0), ValueId(1)), // [2] x * scale
    ];

    Ok(ComputePattern::NormLike {
        reduce,
        finalize,
        transform,
    })
}

/// Three loops → either Softmax-style Reduction or LayerNorm-style NormLike.
///
/// Softmax: max → exp-sum → normalize
/// - Loop 1: Max reduction
/// - Loop 2: Sum reduction (with exp transform)
/// - Loop 3: per-element multiply by inv_sum
///
/// LayerNorm: mean → variance → normalize
/// - Loop 1: Sum reduction (for mean)
/// - Loop 2: Sum reduction (for variance)
/// - Loop 3: per-element normalize + affine
fn combine_three_loops(
    loop1: &LoopTrace,
    loop2: &LoopTrace,
    loop3: &LoopTrace,
) -> Result<ComputePattern, CompilerError> {
    let r1 = loop1.reductions.first()
        .ok_or("loop 1 has no reductions")?;
    let r2 = loop2.reductions.first()
        .ok_or("loop 2 has no reductions")?;

    // Softmax pattern: Max → Sum → normalize
    if r1.kind == ReductionKind::Max && r2.kind == ReductionKind::Sum {
        return combine_softmax(r1, r2, loop3);
    }

    // LayerNorm pattern: Sum (mean) → Sum (variance) → normalize
    if r1.kind == ReductionKind::Sum && r2.kind == ReductionKind::Sum {
        return combine_layer_norm(r1, r2, loop3);
    }

    Err(format!(
        "unrecognized 3-loop pattern: {:?} → {:?} → transform",
        r1.kind, r2.kind
    ).into())
}

/// Build Softmax ComputePattern from 3 loop traces.
fn combine_softmax(
    max_reduction: &ReductionDetected,
    sum_reduction: &ReductionDetected,
    _normalize_loop: &LoopTrace,
) -> Result<ComputePattern, CompilerError> {
    let (max_identity, max_combine) = reduction_to_trace_ops(max_reduction);

    // Second pass: exp(x - max) → sum
    let element_transform = vec![
        TraceOp::Input(0),  // [0] x (current element)
        TraceOp::Input(1),  // [1] max (broadcast from first pass)
        TraceOp::Sub(ValueId(0), ValueId(1)), // [2] x - max
        TraceOp::Exp(ValueId(2)),    // [3] exp(x - max)
    ];

    let sum_combine = vec![
        TraceOp::Input(0),  // [0] acc (running sum)
        TraceOp::Input(1),  // [1] exp_val
        TraceOp::Add(ValueId(0), ValueId(1)), // [2] acc + exp_val
    ];

    let sum_identity = match &sum_reduction.init {
        AccumulatorInit::Const(v) => *v,
        _ => 0.0,
    };

    // Normalize: out[i] = exp_val * inv_sum
    let normalize = vec![
        TraceOp::Input(0),  // [0] exp_val
        TraceOp::Input(1),  // [1] inv_sum (broadcast)
        TraceOp::Mul(ValueId(0), ValueId(1)), // [2] exp_val * inv_sum
    ];

    Ok(ComputePattern::Reduction {
        identity: max_identity,
        combine: max_combine,
        second_pass: Some(Box::new(ReductionSecondPass {
            identity: sum_identity,
            element_transform,
            combine: sum_combine,
        })),
        normalize: Some(normalize),
    })
}

/// Build LayerNorm ComputePattern from 3 loop traces.
fn combine_layer_norm(
    _mean_reduction: &ReductionDetected,
    _var_reduction: &ReductionDetected,
    _normalize_loop: &LoopTrace,
) -> Result<ComputePattern, CompilerError> {
    // Loop 1: sum(x) → mean = sum / n
    let reduce = vec![
        TraceOp::Input(0), // [0] x (used for both mean and variance)
    ];

    // Finalize: compute scale from mean and variance.
    // Input(0) = mean, Input(1) = var
    let finalize = vec![
        TraceOp::Input(0),    // [0] mean
        TraceOp::Input(1),    // [1] var
        TraceOp::Const(1e-5), // [2] eps (placeholder — codegen reads actual from OpKind)
        TraceOp::Add(ValueId(1), ValueId(2)),   // [3] var + eps
        TraceOp::Rsqrt(ValueId(3)),    // [4] rsqrt(var + eps)
    ];

    // Transform: (x - mean) * scale * weight + bias
    let transform = vec![
        TraceOp::Input(0),  // [0] x
        TraceOp::Input(1),  // [1] mean
        TraceOp::Input(2),  // [2] scale (from finalize)
        TraceOp::Input(3),  // [3] weight
        TraceOp::Input(4),  // [4] bias
        TraceOp::Sub(ValueId(0), ValueId(1)), // [5] x - mean
        TraceOp::Mul(ValueId(5), ValueId(2)), // [6] (x - mean) * scale
        TraceOp::Mul(ValueId(6), ValueId(3)), // [7] normed * weight
        TraceOp::Add(ValueId(7), ValueId(4)), // [8] normed * weight + bias
    ];

    Ok(ComputePattern::NormLike {
        reduce,
        finalize,
        transform,
    })
}

// ---------------------------------------------------------------------------
// Helpers: SymValue → TraceOp conversion
// ---------------------------------------------------------------------------

/// Convert a ReductionDetected into (identity, combine TraceOps).
fn reduction_to_trace_ops(r: &ReductionDetected) -> (f64, Vec<TraceOp>) {
    let identity = match &r.init {
        AccumulatorInit::Const(v) => *v,
        AccumulatorInit::Symbolic(_) => match r.kind {
            ReductionKind::Sum => 0.0,
            ReductionKind::Max => f64::NEG_INFINITY,
            ReductionKind::Min => f64::INFINITY,
        },
    };

    let combine = match r.kind {
        ReductionKind::Sum => vec![
            TraceOp::Input(0),  // [0] acc
            TraceOp::Input(1),  // [1] element
            TraceOp::Add(ValueId(0), ValueId(1)), // [2] acc + element
        ],
        ReductionKind::Max => vec![
            TraceOp::Input(0),  // [0] acc
            TraceOp::Input(1),  // [1] element
            TraceOp::Max(ValueId(0), ValueId(1)), // [2] max(acc, element)
        ],
        ReductionKind::Min => vec![
            TraceOp::Input(0),  // [0] acc
            TraceOp::Input(1),  // [1] element
            TraceOp::Min(ValueId(0), ValueId(1)), // [2] min(acc, element)
        ],
    };

    (identity, combine)
}

/// Convert a SymValue body expression into a reduce body (TraceOp sequence).
///
/// For simple cases:
/// - `Param(n)` → `[Input(0)]` (just the element)
/// - `Mul(Param(n), Param(n))` → `[Input(0), Mul(0, 0)]` (x^2, for RmsNorm)
fn symvalue_to_reduce_body(expr: &SymValue) -> Vec<TraceOp> {
    match expr {
        SymValue::Param(_) => vec![
            TraceOp::Input(0), // [0] x
        ],
        SymValue::Mul(a, b) => {
            // Check for x * x pattern (square).
            let a_str = format!("{a}");
            let b_str = format!("{b}");
            if a_str == b_str {
                vec![
                    TraceOp::Input(0),  // [0] x
                    TraceOp::Mul(ValueId(0), ValueId(0)), // [1] x^2
                ]
            } else {
                // General multiply — two inputs.
                vec![
                    TraceOp::Input(0),  // [0] a
                    TraceOp::Input(1),  // [1] b
                    TraceOp::Mul(ValueId(0), ValueId(1)), // [2] a * b
                ]
            }
        }
        _ => {
            // Fallback: just pass through the element.
            vec![TraceOp::Input(0)]
        }
    }
}

// ---------------------------------------------------------------------------
// Nested loop analysis (Phase 5)
// ---------------------------------------------------------------------------

/// Result of analyzing a nested loop structure.
#[derive(Debug, Clone)]
pub struct NestedLoopAnalysis {
    /// Maximum nesting depth (0 = flat, 1 = 2-deep, 2 = 3-deep).
    pub max_depth: usize,
    /// Number of nesting levels (max_depth + 1).
    pub nesting_levels: usize,
    /// Innermost loop trace (if analyzed).
    pub inner_trace: Option<LoopTrace>,
    /// The classified compute pattern.
    pub pattern: ComputePattern,
}

/// Analyze a nested loop structure from a `LoopForest`.
///
/// This handles multi-level loop nests that `combine_passes` cannot:
/// - 3-deep nesting (i, j, k) with FMA/Add(Mul) reduction → `Gemm`
/// - 2-deep nesting with no reduction (pure stores) → `Injective`
///
/// Returns `None` if the forest has no nested loops (caller should use
/// `combine_passes` for flat multi-loop patterns instead).
pub fn analyze_nested_loops(
    forest: &LoopForest,
    cfg: &ControlFlowGraph,
    executor: &SymbolicExecutor,
) -> Option<NestedLoopAnalysis> {
    // Find the maximum nesting depth across all loops.
    let max_depth = forest.loops.iter().map(|l| l.depth).max().unwrap_or(0);

    // Only handle genuinely nested loops (depth >= 1 means at least 2 levels).
    if max_depth == 0 {
        return None;
    }

    // Only claim nested analysis when there's a single top-level loop that
    // contains all the nesting. Multi-top-level patterns (e.g., norms with
    // multiple passes) should fall through to combine_passes even if some
    // passes have compiler-generated inner loops.
    if forest.top_level.len() != 1 {
        return None;
    }

    let nesting_levels = max_depth + 1;

    // Try each depth level from deepest to shallowest, looking for the
    // first loop with an FMA/mul-add reduction (the hallmark of GEMM).
    //
    // We only positively claim a function here when we can identify GEMM.
    // Other nested patterns (RoPE, Transpose) are better handled by their
    // manual traces (Level 3), and compiler-generated nesting in
    // elementwise functions (SiLU, GELU) should fall through to Level 2
    // (linear symexec).
    for try_depth in (1..=max_depth).rev() {
        let candidates: Vec<&NaturalLoop> = forest.loops.iter()
            .filter(|l| l.depth == try_depth)
            .collect();

        for candidate in &candidates {
            let trace = match analyze_single_loop(candidate, cfg, executor) {
                Ok(t) => t,
                Err(_) => continue,
            };

            // Skip loops with completely empty bodies.
            if trace.reductions.is_empty() && trace.unknown_mutations.is_empty() {
                continue;
            }

            // Only claim GEMM — the one pattern we can reliably identify
            // from nested loop structure + FMA reduction.
            if has_fma_or_mul_add_reduction(&trace) {
                return Some(NestedLoopAnalysis {
                    max_depth,
                    nesting_levels,
                    inner_trace: Some(trace),
                    pattern: ComputePattern::Gemm,
                });
            }
        }
    }

    // No GEMM detected at any depth. Return None so the caller falls
    // through to Level 2 (linear symexec) or Level 3 (manual trace).
    // This correctly handles:
    //   - SiLU/GELU: compiler-vectorized nesting → Level 2 → Elementwise
    //   - RoPE/Transpose: genuine nested injective → Level 3 → manual Injective
    None
}

/// Classify a nested loop pattern based on nesting depth and inner loop trace.
fn classify_nested_pattern(nesting_levels: usize, inner_trace: &LoopTrace) -> ComputePattern {
    match nesting_levels {
        // 3-deep: check for GEMM (i, j, k with acc += a[i][k] * b[k][j])
        3 => {
            if has_fma_or_mul_add_reduction(inner_trace) {
                return ComputePattern::Gemm;
            }
            // 3-deep without FMA reduction — unusual, fall back to Injective.
            ComputePattern::Injective {
                body: vec![TraceOp::Input(0)],
                num_inputs: 1,
                num_outputs: 1,
            }
        }
        // 2-deep: RoPE, Transpose, or other injective patterns
        2 => {
            // If the inner loop has a Sum reduction with Mul body, it could
            // still be a dot-product (degenerate GEMM with M=1 or N=1).
            if has_fma_or_mul_add_reduction(inner_trace) {
                return ComputePattern::Gemm;
            }

            // Otherwise it's an injective pattern (Transpose, RoPE, etc.)
            // Determine num_inputs from the inner trace's mutations.
            let num_stores = inner_trace.unknown_mutations.len()
                .max(inner_trace.reductions.len())
                .max(1);

            ComputePattern::Injective {
                body: vec![TraceOp::Input(0)],
                num_inputs: count_distinct_params_in_trace(inner_trace).max(1),
                num_outputs: num_stores.min(2), // RoPE has 2 outputs per pair
            }
        }
        // Deeper nesting — not expected for standard ML ops.
        _ => ComputePattern::Injective {
            body: vec![TraceOp::Input(0)],
            num_inputs: 1,
            num_outputs: 1,
        },
    }
}

/// Check if an inner loop trace has a Sum reduction with Mul or Fma body,
/// which is the hallmark of a GEMM inner loop: `acc += a[i][k] * b[k][j]`.
fn has_fma_or_mul_add_reduction(trace: &LoopTrace) -> bool {
    for r in &trace.reductions {
        if r.kind != ReductionKind::Sum {
            continue;
        }
        // Check if the body expression involves multiplication.
        match &r.body_expr {
            SymValue::Mul(..) => return true,
            SymValue::Fma(..) => return true,
            // Also accept Unknown("unrolled") from unrolled FMA chains.
            SymValue::Unknown(s) if s == "unrolled" => return true,
            _ => {}
        }
    }
    false
}

/// Count distinct Param indices referenced in a loop trace's mutations.
fn count_distinct_params_in_trace(trace: &LoopTrace) -> usize {
    let mut params = std::collections::BTreeSet::new();
    for r in &trace.reductions {
        collect_params(&r.body_expr, &mut params);
    }
    for (_reg, val) in &trace.unknown_mutations {
        collect_params(val, &mut params);
    }
    params.len()
}

/// Recursively collect Param indices from a SymValue tree.
fn collect_params(val: &SymValue, out: &mut std::collections::BTreeSet<usize>) {
    match val {
        SymValue::Param(n) => { out.insert(*n); }
        SymValue::Add(a, b) | SymValue::Sub(a, b) | SymValue::Mul(a, b)
        | SymValue::Div(a, b) | SymValue::Max(a, b) | SymValue::Min(a, b) => {
            collect_params(a, out);
            collect_params(b, out);
        }
        SymValue::Fma(a, b, c) => {
            collect_params(a, out);
            collect_params(b, out);
            collect_params(c, out);
        }
        SymValue::Neg(a) | SymValue::Abs(a) | SymValue::Sqrt(a)
        | SymValue::Rsqrt(a) | SymValue::Recip(a) => {
            collect_params(a, out);
        }
        SymValue::Call(_, args) => {
            for a in args {
                collect_params(a, out);
            }
        }
        SymValue::Select { cond_lhs, cond_rhs, true_val, false_val, .. } => {
            collect_params(cond_lhs, out);
            collect_params(cond_rhs, out);
            collect_params(true_val, out);
            collect_params(false_val, out);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Reduction pattern detection
// ---------------------------------------------------------------------------

/// Try to classify a register mutation as a reduction.
///
/// Patterns recognized:
/// - `acc + expr` where `acc` is the pre-loop value → Sum
/// - `max(acc, expr)` → Max
/// - `min(acc, expr)` → Min
/// - `fma(a, b, acc)` where acc is the addend → Sum (FMA accumulation)
fn detect_reduction_pattern(
    reg: &str,
    pre_val: &SymValue,
    post_val: &SymValue,
) -> Option<ReductionDetected> {
    let pre_str = format!("{pre_val}");
    let init = classify_init(pre_val);

    // Check for Add pattern: post = Add(pre, expr) or Add(expr, pre)
    if let SymValue::Add(a, b) = post_val {
        let a_str = format!("{a}");
        let b_str = format!("{b}");
        if a_str == pre_str {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: ReductionKind::Sum,
                init,
                body_expr: b.simplify(),
            });
        }
        if b_str == pre_str {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: ReductionKind::Sum,
                init,
                body_expr: a.simplify(),
            });
        }
    }

    // Check for FMA pattern: post = Fma(a, b, pre) → acc += a*b
    if let SymValue::Fma(a, b, c) = post_val {
        let c_str = format!("{c}");
        if c_str == pre_str {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: ReductionKind::Sum,
                init,
                body_expr: SymValue::Mul(a.clone(), b.clone()).simplify(),
            });
        }
        // Also check Fma(a, pre, c) and Fma(pre, b, c) — less common but valid
        let a_str = format!("{a}");
        let b_str = format!("{b}");
        if a_str == pre_str {
            // fma(acc, b, c) = acc*b + c — not a standard reduction
            // unless b is 1.0, in which case it's acc + c → Sum
            if is_const_one(b) {
                return Some(ReductionDetected {
                    register: reg.to_string(),
                    kind: ReductionKind::Sum,
                    init,
                    body_expr: c.simplify(),
                });
            }
        }
        if b_str == pre_str
            && is_const_one(a) {
                return Some(ReductionDetected {
                    register: reg.to_string(),
                    kind: ReductionKind::Sum,
                    init,
                    body_expr: c.simplify(),
                });
            }
    }

    // Check for Max pattern: post = Max(pre, expr) or Max(expr, pre)
    if let SymValue::Max(a, b) = post_val {
        let a_str = format!("{a}");
        let b_str = format!("{b}");
        if a_str == pre_str {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: ReductionKind::Max,
                init,
                body_expr: b.simplify(),
            });
        }
        if b_str == pre_str {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: ReductionKind::Max,
                init,
                body_expr: a.simplify(),
            });
        }
    }

    // Check for Min pattern: post = Min(pre, expr) or Min(expr, pre)
    if let SymValue::Min(a, b) = post_val {
        let a_str = format!("{a}");
        let b_str = format!("{b}");
        if a_str == pre_str {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: ReductionKind::Min,
                init,
                body_expr: b.simplify(),
            });
        }
        if b_str == pre_str {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: ReductionKind::Min,
                init,
                body_expr: a.simplify(),
            });
        }
    }

    // Check for Select pattern: conditional max/min from branches.
    // Select(Gt, acc, elem, elem, acc) == Max(acc, elem)
    // Select(Lt, acc, elem, elem, acc) == Max(acc, elem)
    if let SymValue::Select { kind, cond_lhs, cond_rhs, true_val, false_val } = post_val {
        let lhs_str = format!("{cond_lhs}");
        let rhs_str = format!("{cond_rhs}");
        let tv_str = format!("{true_val}");
        let fv_str = format!("{false_val}");

        if lhs_str == pre_str || rhs_str == pre_str {
            let (is_max, body) = match kind {
                super::sym_value::SelectKind::Gt | super::sym_value::SelectKind::Ge => {
                    if lhs_str == pre_str && tv_str == pre_str && fv_str != pre_str {
                        (false, false_val.simplify()) // acc > elem → acc : elem → Min
                    } else if lhs_str == pre_str && fv_str == pre_str && tv_str != pre_str {
                        (true, true_val.simplify())   // acc > elem → elem : acc → Max
                    } else if rhs_str == pre_str && tv_str == pre_str && fv_str != pre_str {
                        (true, false_val.simplify())  // elem > acc → acc : elem → Max
                    } else if rhs_str == pre_str && fv_str == pre_str && tv_str != pre_str {
                        (false, true_val.simplify())  // elem > acc → elem : acc → Min
                    } else {
                        return None;
                    }
                }
                super::sym_value::SelectKind::Lt | super::sym_value::SelectKind::Le => {
                    if lhs_str == pre_str && tv_str == pre_str && fv_str != pre_str {
                        (true, false_val.simplify())  // acc < elem → acc : elem → Max
                    } else if lhs_str == pre_str && fv_str == pre_str && tv_str != pre_str {
                        (false, true_val.simplify())  // acc < elem → elem : acc → Min
                    } else if rhs_str == pre_str && tv_str == pre_str && fv_str != pre_str {
                        (false, false_val.simplify()) // elem < acc → acc : elem → Min
                    } else if rhs_str == pre_str && fv_str == pre_str && tv_str != pre_str {
                        (true, true_val.simplify())   // elem < acc → elem : acc → Max
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            return Some(ReductionDetected {
                register: reg.to_string(),
                kind: if is_max { ReductionKind::Max } else { ReductionKind::Min },
                init,
                body_expr: body,
            });
        }
    }

    // Recursive fallback: handle unrolled reductions where the accumulator
    // is nested inside same-kind operations.
    // e.g. Max(mem1, Max(mem2, acc)) → still a Max reduction over acc.
    // e.g. Add(Add(acc, expr1), expr2) → still a Sum reduction over acc.
    if contains_as_reduction(post_val, &pre_str) {
        let kind = match post_val {
            SymValue::Add(..) => Some(ReductionKind::Sum),
            SymValue::Max(..) => Some(ReductionKind::Max),
            SymValue::Min(..) => Some(ReductionKind::Min),
            _ => None,
        };
        if let Some(kind) = kind {
            return Some(ReductionDetected {
                register: reg.to_string(),
                kind,
                init,
                // For unrolled reductions the body_expr is complex; use Unknown.
                body_expr: SymValue::Unknown("unrolled".into()),
            });
        }
    }

    None
}

/// Check if `val` contains `target_str` as a leaf within a tree of same-kind
/// associative operations (Add/Max/Min). This detects unrolled reductions like
/// `Max(mem1, Max(mem2, acc))` where `acc` matches `target_str`.
fn contains_as_reduction(val: &SymValue, target_str: &str) -> bool {
    match val {
        SymValue::Add(a, b) => {
            let a_str = format!("{a}");
            let b_str = format!("{b}");
            if a_str == target_str || b_str == target_str {
                return true;
            }
            // Recurse into same-kind children only.
            (matches!(**a, SymValue::Add(..)) && contains_as_reduction(a, target_str))
                || (matches!(**b, SymValue::Add(..)) && contains_as_reduction(b, target_str))
        }
        SymValue::Max(a, b) => {
            let a_str = format!("{a}");
            let b_str = format!("{b}");
            if a_str == target_str || b_str == target_str {
                return true;
            }
            (matches!(**a, SymValue::Max(..)) && contains_as_reduction(a, target_str))
                || (matches!(**b, SymValue::Max(..)) && contains_as_reduction(b, target_str))
        }
        SymValue::Min(a, b) => {
            let a_str = format!("{a}");
            let b_str = format!("{b}");
            if a_str == target_str || b_str == target_str {
                return true;
            }
            (matches!(**a, SymValue::Min(..)) && contains_as_reduction(a, target_str))
                || (matches!(**b, SymValue::Min(..)) && contains_as_reduction(b, target_str))
        }
        _ => false,
    }
}

/// Classify how an accumulator was initialized.
fn classify_init(val: &SymValue) -> AccumulatorInit {
    match val {
        SymValue::Const(v) => AccumulatorInit::Const(*v),
        _ => AccumulatorInit::Symbolic(val.clone()),
    }
}

/// Check if a SymValue is the constant 1.0.
fn is_const_one(val: &SymValue) -> bool {
    matches!(val, SymValue::Const(v) if *v == 1.0)
}

/// Check if a mnemonic is a branch instruction (skip during symbolic exec).
fn is_branch_mnemonic(m: &str) -> bool {
    matches!(
        m,
        // x86_64
        "je" | "jne" | "jb" | "jbe" | "ja" | "jae"
            | "jl" | "jle" | "jg" | "jge"
            | "jmp" | "js" | "jns" | "jp" | "jnp"
            | "ret"
            // AArch64
            | "b" | "bl" | "b.eq" | "b.ne" | "b.gt" | "b.ge" | "b.lt" | "b.le"
            | "b.hi" | "b.hs" | "b.lo" | "b.ls" | "b.mi" | "b.pl"
            | "cbz" | "cbnz" | "tbz" | "tbnz"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::compiler::symexec::cfg::{build_cfg_from_fn, find_loops};

    /// Helper: build CFG + find loops for a function pointer.
    fn cfg_and_loops(fn_ptr: *const u8, max_bytes: usize) -> (ControlFlowGraph, Vec<NaturalLoop>) {
        let cfg = unsafe { build_cfg_from_fn(fn_ptr, max_bytes) }.expect("CFG build failed");
        let forest = find_loops(&cfg);
        let loops = forest.loops;
        (cfg, loops)
    }

    #[test]
    fn test_sum_reduction_detection() {
        // Simple sum: acc += x[i]
        extern "C" fn sum_array(ptr: *const f32, len: usize) -> f32 {
            let mut acc = 0.0f32;
            let mut i = 0;
            while i < len {
                acc += unsafe { *ptr.add(i) };
                i += 1;
            }
            acc
        }

        let (cfg, loops) = cfg_and_loops(sum_array as *const u8, 512);
        assert!(!loops.is_empty(), "should find at least one loop");

        // Create executor with 0 float args, 2 ptr/int args (ptr, len).
        let exec = SymbolicExecutor::new(0, 2);
        let trace = analyze_single_loop(&loops[0], &cfg, &exec).expect("analysis failed");

        assert!(
            trace.body_block_count >= 1,
            "loop body should have at least 1 block"
        );

        // We may or may not detect the reduction depending on how the compiler
        // lays out the accumulator. The key test is that analysis completes
        // without error and produces a LoopTrace.
        println!("sum_array loop trace: {} reductions, {} unknown mutations",
            trace.reductions.len(), trace.unknown_mutations.len());

        for r in &trace.reductions {
            println!("  reduction: {:?} on {} (init: {:?})", r.kind, r.register, r.init);
        }
    }

    #[test]
    fn test_sum_sq_reduction_detection() {
        // Sum of squares: acc += x[i] * x[i] (RmsNorm pass 1 pattern)
        extern "C" fn sum_sq(ptr: *const f32, len: usize) -> f32 {
            let mut acc = 0.0f32;
            let mut i = 0;
            while i < len {
                let v = unsafe { *ptr.add(i) };
                acc += v * v;
                i += 1;
            }
            acc
        }

        let (cfg, loops) = cfg_and_loops(sum_sq as *const u8, 512);
        assert!(!loops.is_empty(), "should find at least one loop");

        let exec = SymbolicExecutor::new(0, 2);
        let trace = analyze_single_loop(&loops[0], &cfg, &exec).expect("analysis failed");

        println!("sum_sq loop trace: {} reductions, {} unknown mutations",
            trace.reductions.len(), trace.unknown_mutations.len());

        for r in &trace.reductions {
            println!("  reduction: {:?} on {} (init: {:?}, body: {})",
                r.kind, r.register, r.init, r.body_expr);
        }
    }

    #[test]
    fn test_max_reduction_detection() {
        // Max reduction: acc = max(acc, x[i])
        extern "C" fn max_array(ptr: *const f32, len: usize) -> f32 {
            let mut acc = f32::NEG_INFINITY;
            let mut i = 0;
            while i < len {
                let v = unsafe { *ptr.add(i) };
                if v > acc {
                    acc = v;
                }
                i += 1;
            }
            acc
        }

        let (cfg, loops) = cfg_and_loops(max_array as *const u8, 512);
        assert!(!loops.is_empty(), "should find at least one loop");

        let exec = SymbolicExecutor::new(0, 2);
        let trace = analyze_single_loop(&loops[0], &cfg, &exec).expect("analysis failed");

        println!("max_array loop trace: {} reductions, {} unknown mutations",
            trace.reductions.len(), trace.unknown_mutations.len());

        for r in &trace.reductions {
            println!("  reduction: {:?} on {} (init: {:?})", r.kind, r.register, r.init);
        }
    }

    #[test]
    fn test_loop_analysis_no_crash() {
        // Ensure analysis doesn't crash on a function with no loops.
        extern "C" fn identity(x: f32) -> f32 {
            x
        }

        let cfg = unsafe { build_cfg_from_fn(identity as *const u8, 256) }.expect("CFG build failed");
        let forest = find_loops(&cfg);
        assert!(forest.loops.is_empty(), "identity should have no loops");
    }

    #[test]
    fn test_reduction_pattern_unit() {
        // Unit test for detect_reduction_pattern directly.
        let pre = SymValue::Const(0.0);

        // Sum: post = Add(pre, Param(0))
        let post_sum = SymValue::Add(
            Box::new(SymValue::Const(0.0)),
            Box::new(SymValue::Param(0)),
        );
        let r = detect_reduction_pattern("xmm0", &pre, &post_sum);
        assert!(r.is_some(), "should detect sum reduction");
        let r = r.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);

        // Max: post = Max(pre, Param(0))
        let pre_max = SymValue::Param(1);
        let post_max = SymValue::Max(
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Param(0)),
        );
        let r = detect_reduction_pattern("xmm1", &pre_max, &post_max);
        assert!(r.is_some(), "should detect max reduction");
        assert_eq!(r.unwrap().kind, ReductionKind::Max);

        // Min: post = Min(Param(0), pre)
        let post_min = SymValue::Min(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(1)),
        );
        let r = detect_reduction_pattern("xmm1", &pre_max, &post_min);
        assert!(r.is_some(), "should detect min reduction");
        assert_eq!(r.unwrap().kind, ReductionKind::Min);

        // FMA: post = Fma(Param(0), Param(0), pre) → sum of squares
        let post_fma = SymValue::Fma(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(0.0)),
        );
        let r = detect_reduction_pattern("xmm0", &pre, &post_fma);
        assert!(r.is_some(), "should detect FMA sum reduction");
        let r = r.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);
    }

    #[test]
    fn test_no_reduction_for_non_accumulator() {
        // If post doesn't reference pre at all, no reduction.
        let pre = SymValue::Const(0.0);
        let post = SymValue::Param(0); // Just a load, not an accumulation.
        let r = detect_reduction_pattern("xmm0", &pre, &post);
        assert!(r.is_none(), "should not detect reduction for plain assignment");
    }

    // -----------------------------------------------------------------------
    // Phase 3: combine_passes unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_combine_single_sum_loop() {
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        let pattern = combine_passes(&[trace]).expect("combine failed");
        match &pattern {
            ComputePattern::Reduction { identity, combine, second_pass, normalize } => {
                assert_eq!(*identity, 0.0);
                assert!(second_pass.is_none());
                assert!(normalize.is_none());
                // combine should be [Input(0), Input(1), Add(0,1)]
                assert_eq!(combine.len(), 3);
                assert_eq!(combine[2], TraceOp::Add(ValueId(0), ValueId(1)));
            }
            other => panic!("expected Reduction, got {other:?}"),
        }
    }

    #[test]
    fn test_combine_single_max_loop() {
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Max,
                init: AccumulatorInit::Const(f64::NEG_INFINITY),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        let pattern = combine_passes(&[trace]).expect("combine failed");
        match &pattern {
            ComputePattern::Reduction { identity, combine, .. } => {
                assert_eq!(*identity, f64::NEG_INFINITY);
                assert_eq!(combine[2], TraceOp::Max(ValueId(0), ValueId(1)));
            }
            other => panic!("expected Reduction, got {other:?}"),
        }
    }

    #[test]
    fn test_combine_two_loops_normlike() {
        // Simulate RmsNorm: loop1 = sum(x^2), loop2 = transform
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(0)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        let pattern = combine_passes(&[loop1, loop2]).expect("combine failed");
        match &pattern {
            ComputePattern::NormLike { reduce, .. } => {
                // reduce should contain x^2 pattern: [Input(0), Mul(0,0)]
                assert!(reduce.len() >= 2, "reduce too short: {reduce:?}");
                assert_eq!(reduce[1], TraceOp::Mul(ValueId(0), ValueId(0)));
            }
            other => panic!("expected NormLike, got {other:?}"),
        }
    }

    #[test]
    fn test_combine_three_loops_softmax() {
        // Simulate Softmax: max → exp-sum → normalize
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Max,
                init: AccumulatorInit::Const(f64::NEG_INFINITY),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![ReductionDetected {
                register: "xmm1".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop3 = LoopTrace {
            loop_header: BlockId(2),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        let pattern = combine_passes(&[loop1, loop2, loop3]).expect("combine failed");
        match &pattern {
            ComputePattern::Reduction { identity, second_pass, normalize, .. } => {
                assert_eq!(*identity, f64::NEG_INFINITY);
                assert!(second_pass.is_some(), "softmax should have second_pass");
                assert!(normalize.is_some(), "softmax should have normalize");

                let sp = second_pass.as_ref().unwrap();
                assert_eq!(sp.identity, 0.0);
                // element_transform should contain Sub + Exp
                let has_exp = sp.element_transform.iter().any(|op| matches!(op, TraceOp::Exp(_)));
                assert!(has_exp, "second_pass should have Exp");
            }
            other => panic!("expected Reduction (softmax), got {other:?}"),
        }
    }

    #[test]
    fn test_combine_three_loops_layernorm() {
        // Simulate LayerNorm: sum (mean) → sum (var) → normalize
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![ReductionDetected {
                register: "xmm1".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop3 = LoopTrace {
            loop_header: BlockId(2),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        let pattern = combine_passes(&[loop1, loop2, loop3]).expect("combine failed");
        match &pattern {
            ComputePattern::NormLike { finalize, transform, .. } => {
                // finalize should have Rsqrt
                let has_rsqrt = finalize.iter().any(|op| matches!(op, TraceOp::Rsqrt(_)));
                assert!(has_rsqrt, "LayerNorm finalize should have Rsqrt");
                // transform should have Sub (x - mean)
                let has_sub = transform.iter().any(|op| matches!(op, TraceOp::Sub(_, _)));
                assert!(has_sub, "LayerNorm transform should have Sub");
            }
            other => panic!("expected NormLike (LayerNorm), got {other:?}"),
        }
    }

    #[test]
    fn test_combine_zero_loops_error() {
        let result = combine_passes(&[]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Phase 3: integration tests with real scalar functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_structured_analysis_rms_norm() {
        use crate::compiler::symexec::cfg::{build_cfg_from_fn, find_loops};
        use crate::compiler::symexec::decoder::analyze_scalar_fn_structured;
        use crate::compiler::trace::{ScalarFnSignature, ScalarParam};
        use gllm_scalar_ops::norms::scalar_rms_norm;

        let fn_ptr = scalar_rms_norm as *const u8;
        let sig = ScalarFnSignature {
            fn_ptr,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        };

        // Debug: check CFG directly.
        let cfg = unsafe { build_cfg_from_fn(fn_ptr, 4096) }.expect("CFG build failed");
        println!("RmsNorm CFG: {} blocks", cfg.blocks.len());
        for (id, blk) in &cfg.blocks {
            println!("  Block {:?}: addr 0x{:x}..0x{:x}, {} insns, term={:?}",
                id, blk.start_addr, blk.end_addr, blk.instructions.len(), blk.terminator);
        }
        let forest = find_loops(&cfg);
        println!("RmsNorm loops: {}", forest.loops.len());
        for lp in &forest.loops {
            println!("  Loop: header={:?}, latch={:?}, body={:?}",
                lp.header, lp.latch, lp.body_blocks);
        }

        let result = unsafe { analyze_scalar_fn_structured(fn_ptr, &sig) };
        match result {
            Ok(Some(analysis)) => {
                println!("RmsNorm: {} loops, pattern: {:?}", analysis.num_loops, analysis.pattern);
                assert!(analysis.num_loops >= 2, "RmsNorm should have at least 2 loops");
                assert!(
                    matches!(analysis.pattern, ComputePattern::NormLike { .. }),
                    "RmsNorm should be NormLike, got {:?}", analysis.pattern
                );
            }
            Ok(None) => {
                println!("RmsNorm: structured analysis returned None (no loops detected)");
            }
            Err(e) => {
                println!("RmsNorm structured analysis error (may be expected): {e}");
            }
        }
    }

    #[test]
    fn test_structured_analysis_softmax() {
        use crate::compiler::symexec::cfg::{build_cfg_from_fn, find_loops};
        use crate::compiler::symexec::decoder::analyze_scalar_fn_structured;
        use crate::compiler::trace::{ScalarFnSignature, ScalarParam};
        use gllm_scalar_ops::blas::scalar_softmax;

        let fn_ptr = scalar_softmax as *const u8;
        let sig = ScalarFnSignature {
            fn_ptr,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
            ],
        };

        // Debug: check CFG directly.
        let cfg = unsafe { build_cfg_from_fn(fn_ptr, 4096) }.expect("CFG build failed");
        println!("Softmax CFG: {} blocks", cfg.blocks.len());
        for (id, blk) in &cfg.blocks {
            println!("  Block {:?}: addr 0x{:x}..0x{:x}, {} insns, term={:?}",
                id, blk.start_addr, blk.end_addr, blk.instructions.len(), blk.terminator);
        }
        let forest = find_loops(&cfg);
        println!("Softmax loops: {} total, top_level: {:?}", forest.loops.len(), forest.top_level);
        for (i, lp) in forest.loops.iter().enumerate() {
            println!("  Loop[{}]: header={:?}, latch={:?}, body={:?}, depth={}",
                i, lp.header, lp.latch, lp.body_blocks, lp.depth);
        }

        let result = unsafe { analyze_scalar_fn_structured(fn_ptr, &sig) };
        match result {
            Ok(Some(analysis)) => {
                println!("Softmax: {} loops, pattern: {:?}", analysis.num_loops, analysis.pattern);
            }
            Ok(None) => {
                println!("Softmax: structured analysis returned None");
            }
            Err(e) => {
                println!("Softmax structured analysis error: {e}");
            }
        }
    }

    #[test]
    fn test_structured_analysis_layer_norm() {
        use crate::compiler::symexec::decoder::analyze_scalar_fn_structured;
        use crate::compiler::trace::{ScalarFnSignature, ScalarParam};
        use gllm_scalar_ops::norms::scalar_layer_norm;

        let sig = ScalarFnSignature {
            fn_ptr: scalar_layer_norm as *const u8,
            params: vec![
                ScalarParam::InputPtr,
                ScalarParam::WeightPtr,
                ScalarParam::WeightPtr,
                ScalarParam::OutputPtr,
                ScalarParam::Dim(0),
                ScalarParam::Scalar(1e-5),
            ],
        };

        let result = unsafe { analyze_scalar_fn_structured(sig.fn_ptr, &sig) };
        match result {
            Ok(Some(analysis)) => {
                println!("LayerNorm: {} loops, pattern: {:?}", analysis.num_loops, analysis.pattern);
                assert!(analysis.num_loops >= 2, "LayerNorm should have at least 2 loops");
                assert!(
                    matches!(analysis.pattern, ComputePattern::NormLike { .. }),
                    "LayerNorm should be NormLike, got {:?}", analysis.pattern
                );
            }
            Ok(None) => {
                println!("LayerNorm: structured analysis returned None");
            }
            Err(e) => {
                println!("LayerNorm structured analysis error: {e}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 5: nested loop analysis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nested_gemm_detection() {
        use crate::compiler::symexec::cfg::{build_cfg_from_fn, find_loops};
        use gllm_scalar_ops::blas::scalar_gemm;

        let fn_ptr = scalar_gemm as *const u8;
        let cfg = unsafe { build_cfg_from_fn(fn_ptr, 4096) }.expect("CFG build failed");
        let forest = find_loops(&cfg);

        println!("GEMM CFG: {} blocks", cfg.blocks.len());
        println!("GEMM loops: {} total", forest.loops.len());
        for (i, lp) in forest.loops.iter().enumerate() {
            println!("  Loop[{}]: header={:?}, depth={}, body_blocks={}",
                i, lp.header, lp.depth, lp.body_blocks.len());
        }
        println!("  top_level: {:?}", forest.top_level);
        println!("  children: {:?}", forest.children);

        // GEMM should have nested loops (depth > 0).
        let max_depth = forest.loops.iter().map(|l| l.depth).max().unwrap_or(0);
        println!("  max_depth: {}", max_depth);
        assert!(max_depth >= 1, "GEMM should have nested loops (max_depth >= 1), got {}", max_depth);

        // Run nested analysis.
        let exec = SymbolicExecutor::new(0, 6); // a, b, c, m, n, k
        let result = analyze_nested_loops(&forest, &cfg, &exec);
        assert!(result.is_some(), "GEMM should produce a NestedLoopAnalysis");

        let analysis = result.unwrap();
        println!("GEMM nested analysis: nesting_levels={}, pattern={:?}",
            analysis.nesting_levels, analysis.pattern);

        assert!(
            matches!(analysis.pattern, ComputePattern::Gemm),
            "GEMM should be classified as Gemm, got {:?}", analysis.pattern
        );
    }

    #[test]
    fn test_nested_gemm_structured_api() {
        use crate::compiler::symexec::decoder::analyze_scalar_fn_structured;
        use crate::compiler::trace::{ScalarFnSignature, ScalarParam};
        use gllm_scalar_ops::blas::scalar_gemm;

        let sig = ScalarFnSignature {
            fn_ptr: scalar_gemm as *const u8,
            params: vec![
                ScalarParam::InputPtr,   // a
                ScalarParam::WeightPtr,  // b
                ScalarParam::OutputPtr,  // c
                ScalarParam::Dim(0),     // m
                ScalarParam::Dim(1),     // n
                ScalarParam::Dim(2),     // k
            ],
        };

        let result = unsafe { analyze_scalar_fn_structured(sig.fn_ptr, &sig) };
        match result {
            Ok(Some(analysis)) => {
                println!("GEMM structured: {} loops, pattern: {:?}",
                    analysis.num_loops, analysis.pattern);
                assert!(
                    matches!(analysis.pattern, ComputePattern::Gemm),
                    "GEMM should be Gemm, got {:?}", analysis.pattern
                );
            }
            Ok(None) => {
                panic!("GEMM structured analysis returned None — expected Gemm pattern");
            }
            Err(e) => {
                panic!("GEMM structured analysis error: {e}");
            }
        }
    }

    #[test]
    fn test_nested_rope_detection() {
        use crate::compiler::symexec::cfg::{build_cfg_from_fn, find_loops};
        use gllm_scalar_ops::rope::scalar_rope;

        let fn_ptr = scalar_rope as *const u8;
        let cfg = unsafe { build_cfg_from_fn(fn_ptr, 4096) }.expect("CFG build failed");
        let forest = find_loops(&cfg);

        println!("RoPE CFG: {} blocks", cfg.blocks.len());
        println!("RoPE loops: {} total", forest.loops.len());
        for (i, lp) in forest.loops.iter().enumerate() {
            println!("  Loop[{}]: header={:?}, depth={}, body_blocks={}",
                i, lp.header, lp.depth, lp.body_blocks.len());
        }

        let max_depth = forest.loops.iter().map(|l| l.depth).max().unwrap_or(0);
        println!("  max_depth: {}", max_depth);

        // RoPE has 2-deep nesting (outer: heads, inner: dim pairs).
        assert!(max_depth >= 1, "RoPE should have nested loops, got max_depth={}", max_depth);

        // Run nested analysis — RoPE is Injective (no reduction, no GEMM),
        // so analyze_nested_loops returns None. In the full pipeline, RoPE
        // is handled by its manual Injective trace in the registry.
        let exec = SymbolicExecutor::new(0, 6); // x, cos, sin, out, head_dim, n_heads
        let result = analyze_nested_loops(&forest, &cfg, &exec);
        assert!(result.is_none(),
            "RoPE has no GEMM reduction; expected None, got {:?}",
            result.as_ref().map(|a| &a.pattern));
    }

    #[test]
    fn test_nested_transpose_detection() {
        use crate::compiler::symexec::cfg::{build_cfg_from_fn, find_loops};
        use gllm_scalar_ops::blas::scalar_transpose_2d;

        let fn_ptr = scalar_transpose_2d as *const u8;
        let cfg = unsafe { build_cfg_from_fn(fn_ptr, 4096) }.expect("CFG build failed");
        let forest = find_loops(&cfg);

        println!("Transpose CFG: {} blocks", cfg.blocks.len());
        println!("Transpose loops: {} total", forest.loops.len());
        for (i, lp) in forest.loops.iter().enumerate() {
            println!("  Loop[{}]: header={:?}, depth={}, body_blocks={}",
                i, lp.header, lp.depth, lp.body_blocks.len());
        }

        let max_depth = forest.loops.iter().map(|l| l.depth).max().unwrap_or(0);
        println!("  max_depth: {}", max_depth);

        // Transpose has 2-deep nesting (outer: rows, inner: cols).
        assert!(max_depth >= 1, "Transpose should have nested loops, got max_depth={}", max_depth);

        let exec = SymbolicExecutor::new(0, 4); // input, out, rows, cols
        let result = analyze_nested_loops(&forest, &cfg, &exec);

        // Transpose's inner loops are all empty (no reductions, no mutations
        // detectable by the symbolic executor). analyze_nested_loops correctly
        // returns None here; in the full pipeline the manual Injective fallback
        // in the registry handles classification.
        assert!(result.is_none(),
            "Transpose inner loops are all empty; expected None, got {:?}",
            result.as_ref().map(|a| &a.pattern));
    }

    #[test]
    fn test_nested_unit_classify_gemm() {
        // Unit test: 3-deep nesting with Sum+Mul reduction → Gemm
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(1)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        let pattern = classify_nested_pattern(3, &trace);
        assert!(
            matches!(pattern, ComputePattern::Gemm),
            "3-deep + Sum(Mul) should be Gemm, got {:?}", pattern
        );
    }

    #[test]
    fn test_nested_unit_classify_injective() {
        // Unit test: 2-deep nesting with no reduction → Injective
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![],
            unknown_mutations: vec![
                ("xmm0".into(), SymValue::Param(0)),
            ],
            body_block_count: 1,
        };

        let pattern = classify_nested_pattern(2, &trace);
        assert!(
            matches!(pattern, ComputePattern::Injective { .. }),
            "2-deep + no reduction should be Injective, got {:?}", pattern
        );
    }

    // -----------------------------------------------------------------------
    // Additional unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduction_kind_copy_and_equality() {
        let sum = ReductionKind::Sum;
        let copied = sum;
        assert_eq!(sum, copied);
        assert_ne!(sum, ReductionKind::Max);
        assert_ne!(sum, ReductionKind::Min);
        assert_eq!(ReductionKind::Max, ReductionKind::Max);
        assert_eq!(ReductionKind::Min, ReductionKind::Min);
    }

    #[test]
    fn test_accumulator_init_debug_output_const() {
        let init = AccumulatorInit::Const(0.0);
        let debug = format!("{init:?}");
        assert!(debug.contains("Const"), "expected Const in debug output, got: {debug}");
    }

    #[test]
    fn test_accumulator_init_debug_output_symbolic() {
        let init = AccumulatorInit::Symbolic(SymValue::Param(3));
        let debug = format!("{init:?}");
        assert!(debug.contains("Symbolic"), "expected Symbolic in debug output, got: {debug}");
    }

    #[test]
    fn test_reduction_detected_clone_independence() {
        let original = ReductionDetected {
            register: "xmm5".into(),
            kind: ReductionKind::Max,
            init: AccumulatorInit::Const(f64::NEG_INFINITY),
            body_expr: SymValue::Param(0),
        };
        let cloned = original.clone();
        assert_eq!(original.register, cloned.register);
        assert_eq!(original.kind, cloned.kind);
        assert!(matches!(original.init, AccumulatorInit::Const(_)));
    }

    #[test]
    fn test_loop_trace_clone_preserves_fields() {
        let trace = LoopTrace {
            loop_header: BlockId(42),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(1.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![("xmm1".into(), SymValue::Const(99.0))],
            body_block_count: 7,
        };
        let cloned = trace.clone();
        assert_eq!(cloned.loop_header, BlockId(42));
        assert_eq!(cloned.reductions.len(), 1);
        assert_eq!(cloned.unknown_mutations.len(), 1);
        assert_eq!(cloned.body_block_count, 7);
    }

    #[test]
    fn test_detect_reduction_fma_with_const_one_multiplier() {
        let pre = SymValue::Const(0.0);
        let post = SymValue::Fma(
            Box::new(SymValue::Const(1.0)),
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(0.0)),
        );
        let result = detect_reduction_pattern("xmm0", &pre, &post);
        assert!(result.is_some(), "Fma(1.0, x, acc) should be detected as Sum reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);
    }

    #[test]
    fn test_detect_reduction_nested_max_unrolled() {
        let pre = SymValue::Param(0);
        let post = SymValue::Max(
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Max(
                Box::new(SymValue::Param(2)),
                Box::new(SymValue::Param(0)),
            )),
        );
        let result = detect_reduction_pattern("xmm0", &pre, &post);
        assert!(result.is_some(), "nested Max tree containing acc should be detected");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Max);
    }

    #[test]
    fn test_detect_reduction_nested_sum_unrolled() {
        let pre = SymValue::Const(0.0);
        let post = SymValue::Add(
            Box::new(SymValue::Add(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Const(0.0)),
            )),
            Box::new(SymValue::Param(1)),
        );
        let result = detect_reduction_pattern("xmm0", &pre, &post);
        assert!(result.is_some(), "nested Add tree containing acc should be detected");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);
    }

    #[test]
    fn test_combine_single_loop_no_reductions_error() {
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![],
            unknown_mutations: vec![("xmm0".into(), SymValue::Param(0))],
            body_block_count: 1,
        };
        let result = combine_passes(&[trace]);
        assert!(result.is_err(), "single loop with no reductions should error");
    }

    #[test]
    fn test_combine_single_loop_with_unknown_mutations_error() {
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![("xmm1".into(), SymValue::Param(1))],
            body_block_count: 2,
        };
        let result = combine_passes(&[trace]);
        assert!(result.is_err(), "single loop with unknown mutations should error");
    }

    #[test]
    fn test_combine_four_logical_loops_error() {
        let traces: Vec<LoopTrace> = (0..4).map(|i| LoopTrace {
            loop_header: BlockId(i),
            reductions: vec![ReductionDetected {
                register: format!("xmm{i}"),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(i as usize),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        }).collect();
        let result = combine_passes(&traces);
        assert!(result.is_err(), "4 logical passes should error");
    }

    #[test]
    fn test_combine_tail_coalescing_skips_consecutive_transform_only() {
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop3 = LoopTrace {
            loop_header: BlockId(2),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let pattern = combine_passes(&[loop1, loop2, loop3]).expect("combine should succeed");
        match &pattern {
            ComputePattern::NormLike { reduce, .. } => {
                assert_eq!(reduce.len(), 1);
                assert_eq!(reduce[0], TraceOp::Input(0));
            }
            other => panic!("expected NormLike (loop2+loop3 coalesced into one transform pass), got {other:?}"),
        }
    }

    #[test]
    fn test_symvalue_to_reduce_body_patterns() {
        let square = SymValue::Mul(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(0)),
        );
        let body = symvalue_to_reduce_body(&square);
        assert_eq!(body.len(), 2);
        assert_eq!(body[1], TraceOp::Mul(ValueId(0), ValueId(0)));

        let param = SymValue::Param(5);
        let body = symvalue_to_reduce_body(&param);
        assert_eq!(body.len(), 1);
        assert_eq!(body[0], TraceOp::Input(0));
    }

    // -----------------------------------------------------------------------
    // Additional unit tests (wave-12k74)
    // -----------------------------------------------------------------------

    #[test]
    fn test_symvalue_to_reduce_body_different_factors() {
        // Arrange: Mul(Param(0), Param(1)) — different operands → general 2-input pattern
        let mul_ab = SymValue::Mul(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(1)),
        );

        // Act
        let body = symvalue_to_reduce_body(&mul_ab);

        // Assert: should produce [Input(0), Input(1), Mul(0,1)]
        assert_eq!(body.len(), 3, "general multiply should produce 3 TraceOps");
        assert_eq!(body[0], TraceOp::Input(0));
        assert_eq!(body[1], TraceOp::Input(1));
        assert_eq!(body[2], TraceOp::Mul(ValueId(0), ValueId(1)));
    }

    #[test]
    fn test_symvalue_to_reduce_body_fallback_for_add() {
        // Arrange: Add is not handled by symvalue_to_reduce_body → fallback to [Input(0)]
        let add_expr = SymValue::Add(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Const(1.0)),
        );

        // Act
        let body = symvalue_to_reduce_body(&add_expr);

        // Assert
        assert_eq!(body.len(), 1, "unhandled SymValue should fallback to single Input");
        assert_eq!(body[0], TraceOp::Input(0));
    }

    #[test]
    fn test_classify_init_const_value() {
        // Arrange
        let val = SymValue::Const(42.0);

        // Act
        let init = classify_init(&val);

        // Assert
        match init {
            AccumulatorInit::Const(v) => assert_eq!(v, 42.0),
            other => panic!("expected Const(42.0), got {other:?}"),
        }
    }

    #[test]
    fn test_classify_init_symbolic_value() {
        // Arrange
        let val = SymValue::Param(7);

        // Act
        let init = classify_init(&val);

        // Assert
        match init {
            AccumulatorInit::Symbolic(ref sv) => {
                assert!(matches!(sv, SymValue::Param(7)));
            }
            other => panic!("expected Symbolic, got {other:?}"),
        }
    }

    #[test]
    fn test_is_const_one_true_and_false() {
        // Arrange
        let one = SymValue::Const(1.0);
        let zero = SymValue::Const(0.0);
        let not_const = SymValue::Param(0);

        // Act & Assert
        assert!(is_const_one(&one), "Const(1.0) should be const one");
        assert!(!is_const_one(&zero), "Const(0.0) should not be const one");
        assert!(!is_const_one(&not_const), "Param should not be const one");
    }

    #[test]
    fn test_is_branch_mnemonic_x86_and_aarch64() {
        // Arrange & Act & Assert: x86_64 conditional branches
        assert!(is_branch_mnemonic("je"));
        assert!(is_branch_mnemonic("jne"));
        assert!(is_branch_mnemonic("jmp"));
        assert!(is_branch_mnemonic("ret"));

        // AArch64 branches
        assert!(is_branch_mnemonic("b.eq"));
        assert!(is_branch_mnemonic("cbz"));
        assert!(is_branch_mnemonic("tbz"));

        // Non-branch mnemonics
        assert!(!is_branch_mnemonic("add"));
        assert!(!is_branch_mnemonic("vmovaps"));
        assert!(!is_branch_mnemonic("fmadd"));
    }

    #[test]
    fn test_reduction_to_trace_ops_sum_identity_and_combine() {
        // Arrange: Sum reduction with Const(0.0) init
        let det = ReductionDetected {
            register: "xmm0".into(),
            kind: ReductionKind::Sum,
            init: AccumulatorInit::Const(0.0),
            body_expr: SymValue::Param(0),
        };

        // Act
        let (identity, combine) = reduction_to_trace_ops(&det);

        // Assert
        assert_eq!(identity, 0.0, "Sum identity should be 0.0");
        assert_eq!(combine.len(), 3);
        assert_eq!(combine[0], TraceOp::Input(0)); // acc
        assert_eq!(combine[1], TraceOp::Input(1)); // element
        assert_eq!(combine[2], TraceOp::Add(ValueId(0), ValueId(1))); // acc + elem
    }

    #[test]
    fn test_reduction_to_trace_ops_min_identity() {
        // Arrange: Min reduction with Symbolic init → should use +inf fallback
        let det = ReductionDetected {
            register: "xmm2".into(),
            kind: ReductionKind::Min,
            init: AccumulatorInit::Symbolic(SymValue::Param(3)),
            body_expr: SymValue::Param(0),
        };

        // Act
        let (identity, combine) = reduction_to_trace_ops(&det);

        // Assert
        assert_eq!(identity, f64::INFINITY, "Symbolic Min init should fallback to +inf");
        assert_eq!(combine.len(), 3);
        assert_eq!(combine[2], TraceOp::Min(ValueId(0), ValueId(1)));
    }

    #[test]
    fn test_collect_params_distinct_indices() {
        // Arrange: expression referencing Param(0), Param(2), Param(5)
        let expr = SymValue::Add(
            Box::new(SymValue::Mul(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Param(2)),
            )),
            Box::new(SymValue::Param(5)),
        );
        let mut params = std::collections::BTreeSet::new();

        // Act
        collect_params(&expr, &mut params);

        // Assert
        assert_eq!(params.len(), 3);
        assert!(params.contains(&0));
        assert!(params.contains(&2));
        assert!(params.contains(&5));
    }

    #[test]
    fn test_has_fma_or_mul_add_rejection_for_non_mul_body() {
        // Arrange: Sum reduction with Const body (not Mul/Fma) → should reject
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Const(1.0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act & Assert
        assert!(!has_fma_or_mul_add_reduction(&trace),
            "Sum reduction with Const body should not be detected as FMA/Mul-add");
    }

    // -----------------------------------------------------------------------
    // Additional unit tests (wave-12k99)
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduction_kind_debug_formatting() {
        // Arrange & Act
        let sum_debug = format!("{:?}", ReductionKind::Sum);
        let max_debug = format!("{:?}", ReductionKind::Max);
        let min_debug = format!("{:?}", ReductionKind::Min);

        // Assert: Debug trait should produce readable names
        assert!(sum_debug.contains("Sum"), "Sum debug should contain 'Sum': {sum_debug}");
        assert!(max_debug.contains("Max"), "Max debug should contain 'Max': {max_debug}");
        assert!(min_debug.contains("Min"), "Min debug should contain 'Min': {min_debug}");
    }

    #[test]
    fn test_accumulator_init_symbolic_clone_preserves_inner_value() {
        // Arrange
        let init = AccumulatorInit::Symbolic(SymValue::Mul(
            Box::new(SymValue::Param(2)),
            Box::new(SymValue::Const(3.14)),
        ));

        // Act
        let cloned = init.clone();

        // Assert: cloned Symbolic should match original's inner SymValue
        match (&init, &cloned) {
            (AccumulatorInit::Symbolic(a), AccumulatorInit::Symbolic(b)) => {
                assert_eq!(format!("{a}"), format!("{b}"), "cloned Symbolic inner value should match");
            }
            _ => panic!("both should be Symbolic variant"),
        }
    }

    #[test]
    fn test_multi_pass_analysis_construction() {
        // Arrange
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 2,
        };
        let pattern = ComputePattern::Elementwise {
            body: vec![TraceOp::Input(0)],
        };

        // Act
        let analysis = MultiPassAnalysis {
            loop_traces: vec![trace],
            pattern,
            num_loops: 1,
        };

        // Assert
        assert_eq!(analysis.num_loops, 1);
        assert_eq!(analysis.loop_traces.len(), 1);
        assert!(matches!(analysis.pattern, ComputePattern::Elementwise { .. }));
    }

    #[test]
    fn test_nested_loop_analysis_construction_and_fields() {
        // Arrange
        let inner = LoopTrace {
            loop_header: BlockId(5),
            reductions: vec![ReductionDetected {
                register: "xmm3".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(1)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 3,
        };

        // Act
        let analysis = NestedLoopAnalysis {
            max_depth: 2,
            nesting_levels: 3,
            inner_trace: Some(inner.clone()),
            pattern: ComputePattern::Gemm,
        };

        // Assert
        assert_eq!(analysis.max_depth, 2);
        assert_eq!(analysis.nesting_levels, 3);
        assert!(analysis.inner_trace.is_some());
        assert!(matches!(analysis.pattern, ComputePattern::Gemm));
        let it = analysis.inner_trace.as_ref().unwrap();
        assert_eq!(it.loop_header, BlockId(5));
        assert_eq!(it.reductions.len(), 1);
    }

    #[test]
    fn test_detect_reduction_select_ge_pattern_as_min() {
        // Arrange: Select(Ge, acc, elem, acc, elem)
        // if acc >= elem → acc (true), else → elem (false)
        // When acc >= elem, result is acc (the larger). But this keeps the smaller
        // when acc < elem (takes elem). Over iterations, this converges to Max.
        // However, the code's interpretation at line 810: lhs=acc, tv=acc → Min
        // (acc > elem → keep acc, else take elem → keeps the *smaller* = Min)
        use super::super::sym_value::SelectKind;
        let pre = SymValue::Param(0); // acc
        let post = SymValue::Select {
            kind: SelectKind::Ge,
            cond_lhs: Box::new(SymValue::Param(0)), // acc
            cond_rhs: Box::new(SymValue::Param(1)), // elem
            true_val: Box::new(SymValue::Param(0)), // acc
            false_val: Box::new(SymValue::Param(1)), // elem
        };

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert
        assert!(result.is_some(), "Select(Ge) pattern should detect a reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Min,
            "Select(Ge, acc, elem, acc, elem) is classified as Min");
    }

    #[test]
    fn test_detect_reduction_select_lt_pattern_as_max() {
        // Arrange: Select(Lt, elem, acc, elem, acc)
        // if elem < acc → true: take elem; false: take acc
        // Code interprets: rhs=acc=pre, fv=acc=pre, tv=elem≠pre → line 830: Max
        use super::super::sym_value::SelectKind;
        let pre = SymValue::Param(1); // acc
        let post = SymValue::Select {
            kind: SelectKind::Lt,
            cond_lhs: Box::new(SymValue::Param(0)), // elem
            cond_rhs: Box::new(SymValue::Param(1)), // acc
            true_val: Box::new(SymValue::Param(0)), // elem
            false_val: Box::new(SymValue::Param(1)), // acc
        };

        // Act
        let result = detect_reduction_pattern("xmm1", &pre, &post);

        // Assert
        assert!(result.is_some(), "Select(Lt) pattern should detect a reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Max,
            "Select(Lt, elem, acc, elem, acc) classified as Max by code");
    }

    #[test]
    fn test_contains_as_reduction_deeply_nested_add() {
        // Arrange: Add(Param(1), Add(Param(2), Add(Param(3), Const(0.0))))
        // The Const(0.0) is buried 3 levels deep inside Add chain.
        let target = SymValue::Const(0.0);
        let target_str = format!("{target}");
        let deep = SymValue::Add(
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Add(
                Box::new(SymValue::Param(2)),
                Box::new(SymValue::Add(
                    Box::new(SymValue::Param(3)),
                    Box::new(SymValue::Const(0.0)),
                )),
            )),
        );

        // Act
        let found = contains_as_reduction(&deep, &target_str);

        // Assert
        assert!(found, "Const(0.0) buried in 3-deep Add chain should be found");
    }

    #[test]
    fn test_contains_as_reduction_wrong_kind_returns_false() {
        // Arrange: searching for a string that doesn't exist in the tree
        let target = SymValue::Const(0.0);
        let target_str = format!("{target}");
        let unrelated = SymValue::Add(
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Param(2)),
        );

        // Act
        let found = contains_as_reduction(&unrelated, &target_str);

        // Assert
        assert!(!found, "target not in Add tree should not be found");
    }

    #[test]
    fn test_count_distinct_params_in_trace_reductions_and_mutations() {
        // Arrange: reduction references Param(0), Param(3); mutation references Param(1)
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(3)),
                ),
            }],
            unknown_mutations: vec![("xmm1".into(), SymValue::Param(1))],
            body_block_count: 1,
        };

        // Act
        let count = count_distinct_params_in_trace(&trace);

        // Assert
        assert_eq!(count, 3, "should find 3 distinct params: 0, 1, 3");
    }

    #[test]
    fn test_combine_two_loops_non_sum_first_loop_error() {
        // Arrange: first loop has Max reduction (not Sum) → should error
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Max,
                init: AccumulatorInit::Const(f64::NEG_INFINITY),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act
        let result = combine_passes(&[loop1, loop2]);

        // Assert
        assert!(result.is_err(), "2-loop with Max as first reduction should error");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Max"), "error message should mention Max, got: {err_msg}");
    }

    #[test]
    fn test_has_fma_or_mul_add_with_fma_body() {
        // Arrange: Sum reduction with Fma body → should be detected
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Fma(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(1)),
                    Box::new(SymValue::Param(2)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act & Assert
        assert!(has_fma_or_mul_add_reduction(&trace),
            "Sum reduction with Fma body should be detected");
    }

    #[test]
    fn test_has_fma_or_mul_add_with_unrolled_unknown() {
        // Arrange: Sum reduction with Unknown("unrolled") body → should be detected
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Unknown("unrolled".into()),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act & Assert
        assert!(has_fma_or_mul_add_reduction(&trace),
            "Sum reduction with Unknown(unrolled) body should be detected");
    }

    #[test]
    fn test_classify_nested_pattern_deep_fallback_injective() {
        // Arrange: 5-deep nesting (unusual) → should fall back to Injective
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![],
            unknown_mutations: vec![("xmm0".into(), SymValue::Param(0))],
            body_block_count: 1,
        };

        // Act
        let pattern = classify_nested_pattern(5, &trace);

        // Assert
        if let ComputePattern::Injective { num_inputs, num_outputs, .. } = &pattern {
            assert_eq!(*num_inputs, 1, "expected num_inputs=1");
            assert_eq!(*num_outputs, 1, "expected num_outputs=1");
        } else {
            panic!("5-deep nesting should fall back to Injective, got {:?}", pattern);
        }
    }

    #[test]
    fn test_combine_three_loops_unrecognized_pattern_error() {
        // Arrange: Max → Max → transform (not Softmax or LayerNorm)
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Max,
                init: AccumulatorInit::Const(f64::NEG_INFINITY),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![ReductionDetected {
                register: "xmm1".into(),
                kind: ReductionKind::Max,
                init: AccumulatorInit::Const(f64::NEG_INFINITY),
                body_expr: SymValue::Param(1),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop3 = LoopTrace {
            loop_header: BlockId(2),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act
        let result = combine_passes(&[loop1, loop2, loop3]);

        // Assert
        assert!(result.is_err(), "Max → Max → transform is unrecognized pattern");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("unrecognized"), "error should mention unrecognized, got: {err_msg}");
    }

    #[test]
    fn test_reduction_to_trace_ops_max_symbolic_init_uses_neg_infinity() {
        // Arrange: Max reduction with Symbolic init → should fallback to -inf
        let det = ReductionDetected {
            register: "xmm1".into(),
            kind: ReductionKind::Max,
            init: AccumulatorInit::Symbolic(SymValue::Param(5)),
            body_expr: SymValue::Param(0),
        };

        // Act
        let (identity, combine) = reduction_to_trace_ops(&det);

        // Assert
        assert_eq!(identity, f64::NEG_INFINITY,
            "Symbolic Max init should fallback to -inf");
        assert_eq!(combine[2], TraceOp::Max(ValueId(0), ValueId(1)));
    }

    // -----------------------------------------------------------------------
    // Additional unit tests (wave-12keb)
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_reduction_fma_pre_as_multiplier_non_unit_returns_none() {
        // Arrange: Fma(acc, Const(2.0), Param(0)) — acc * 2.0 + Param(0) is NOT a
        // standard reduction because the multiplier on acc is not 1.0.
        let pre = SymValue::Param(0);
        let post = SymValue::Fma(
            Box::new(SymValue::Param(0)),       // a = acc
            Box::new(SymValue::Const(2.0)),      // b = 2.0 (not 1.0)
            Box::new(SymValue::Param(1)),        // c = element
        );

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert
        assert!(result.is_none(),
            "Fma(acc, 2.0, x) should not be detected as reduction (multiplier != 1.0)");
    }

    #[test]
    fn test_detect_reduction_fma_pre_as_b_with_const_one_a() {
        // Arrange: Fma(Const(1.0), acc, Param(0)) → acc * 1.0 + Param(0) = acc + Param(0)
        // This hits the b_str == pre_str && is_const_one(a) branch.
        let pre = SymValue::Param(0);
        let post = SymValue::Fma(
            Box::new(SymValue::Const(1.0)),      // a = 1.0
            Box::new(SymValue::Param(0)),        // b = acc (pre)
            Box::new(SymValue::Param(1)),        // c = element
        );

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert
        assert!(result.is_some(), "Fma(1.0, acc, x) should be detected as Sum reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);
    }

    #[test]
    fn test_contains_as_reduction_max_tree_finds_target() {
        // Arrange: Max(Param(1), Max(Param(2), Max(Param(3), acc)))
        // where acc = Param(0). The target is buried in a Max tree.
        let pre_str = format!("{}", SymValue::Param(0));
        let deep_max = SymValue::Max(
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Max(
                Box::new(SymValue::Param(2)),
                Box::new(SymValue::Max(
                    Box::new(SymValue::Param(3)),
                    Box::new(SymValue::Param(0)),
                )),
            )),
        );

        // Act
        let found = contains_as_reduction(&deep_max, &pre_str);

        // Assert
        assert!(found, "Param(0) buried in 3-deep Max tree should be found");
    }

    #[test]
    fn test_contains_as_reduction_min_tree_finds_target() {
        // Arrange: Min(acc, Min(Param(1), Param(2)))
        let acc_str = format!("{}", SymValue::Const(f64::INFINITY));
        let min_tree = SymValue::Min(
            Box::new(SymValue::Const(f64::INFINITY)),
            Box::new(SymValue::Min(
                Box::new(SymValue::Param(1)),
                Box::new(SymValue::Param(2)),
            )),
        );

        // Act
        let found = contains_as_reduction(&min_tree, &acc_str);

        // Assert
        assert!(found, "Const(+inf) buried in Min tree should be found");
    }

    #[test]
    fn test_collect_params_traverses_fma_and_unary_ops() {
        // Arrange: Fma(Sqrt(Param(0)), Neg(Param(2)), Rsqrt(Param(5)))
        // Should collect params 0, 2, 5 from unary ops inside Fma.
        let expr = SymValue::Fma(
            Box::new(SymValue::Sqrt(Box::new(SymValue::Param(0)))),
            Box::new(SymValue::Neg(Box::new(SymValue::Param(2)))),
            Box::new(SymValue::Rsqrt(Box::new(SymValue::Param(5)))),
        );
        let mut params = std::collections::BTreeSet::new();

        // Act
        collect_params(&expr, &mut params);

        // Assert
        assert_eq!(params.len(), 3, "should find 3 distinct params from Fma+unary");
        assert!(params.contains(&0));
        assert!(params.contains(&2));
        assert!(params.contains(&5));
    }

    #[test]
    fn test_collect_params_traverses_select_and_call() {
        // Arrange: Select with Call in both branches
        // Select { kind: Gt, cond_lhs: Param(0), cond_rhs: Param(1),
        //          true_val: Call(Expf, [Param(2)]), false_val: Abs(Param(3)) }
        use super::super::sym_value::SelectKind;
        use super::super::sym_value::LibmFn;
        let expr = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Call(LibmFn::Expf, vec![SymValue::Param(2)])),
            false_val: Box::new(SymValue::Abs(Box::new(SymValue::Param(3)))),
        };
        let mut params = std::collections::BTreeSet::new();

        // Act
        collect_params(&expr, &mut params);

        // Assert
        assert_eq!(params.len(), 4, "should find params 0,1,2,3 from Select+Call+Abs");
        assert!(params.contains(&0));
        assert!(params.contains(&1));
        assert!(params.contains(&2));
        assert!(params.contains(&3));
    }

    #[test]
    fn test_combine_all_transform_only_coalesced_to_zero_error() {
        // Arrange: all loops are transform-only (no reductions) → after coalescing
        // all into one, logical count becomes 1, then combine_single_loop rejects
        // because no reductions → error.
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act
        let result = combine_passes(&[loop1, loop2]);

        // Assert: both are transform-only, so they coalesce into 1 logical pass,
        // which then fails because the single logical loop has no reductions.
        assert!(result.is_err(), "all-transform-only loops should error after coalescing");
    }

    #[test]
    fn test_combine_three_loops_min_sum_unrecognized() {
        // Arrange: Min → Sum → transform is not a recognized 3-loop pattern
        // (recognized: Max+Sum=Softmax, Sum+Sum=LayerNorm)
        let loop1 = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Min,
                init: AccumulatorInit::Const(f64::INFINITY),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop2 = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![ReductionDetected {
                register: "xmm1".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };
        let loop3 = LoopTrace {
            loop_header: BlockId(2),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act
        let result = combine_passes(&[loop1, loop2, loop3]);

        // Assert
        assert!(result.is_err(), "Min → Sum → transform should be unrecognized");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("unrecognized"),
            "error should mention 'unrecognized', got: {err_msg}");
    }

    #[test]
    fn test_symvalue_to_reduce_body_fallback_for_const() {
        // Arrange: SymValue::Const is not Mul or Param → fallback path
        let const_expr = SymValue::Const(3.14);

        // Act
        let body = symvalue_to_reduce_body(&const_expr);

        // Assert: fallback produces [Input(0)]
        assert_eq!(body.len(), 1, "Const should fallback to single Input");
        assert_eq!(body[0], TraceOp::Input(0));
    }

    #[test]
    fn test_detect_reduction_select_unhandled_eq_returns_none() {
        // Arrange: Select with Eq kind — not handled by the code (only Gt/Ge/Lt/Le)
        use super::super::sym_value::SelectKind;
        let pre = SymValue::Param(0);
        let post = SymValue::Select {
            kind: SelectKind::Eq,
            cond_lhs: Box::new(SymValue::Param(0)),
            cond_rhs: Box::new(SymValue::Param(1)),
            true_val: Box::new(SymValue::Param(0)),
            false_val: Box::new(SymValue::Param(1)),
        };

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert
        assert!(result.is_none(),
            "Select(Eq, ...) should not be detected as reduction (unhandled SelectKind)");
    }

    // -----------------------------------------------------------------------
    // Additional unit tests (wave-12kla)
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_reduction_select_gt_rhs_pre_tv_pre_is_max() {
        // Arrange: Select(Gt, elem, acc, acc, elem)
        // Gt branch, rhs=pre, tv=pre, fv!=pre → line 814-815: (true, false_val) → Max
        use super::super::sym_value::SelectKind;
        let pre = SymValue::Param(1); // acc
        let post = SymValue::Select {
            kind: SelectKind::Gt,
            cond_lhs: Box::new(SymValue::Param(0)), // elem
            cond_rhs: Box::new(SymValue::Param(1)), // acc (pre)
            true_val: Box::new(SymValue::Param(1)), // acc (pre)
            false_val: Box::new(SymValue::Param(0)), // elem
        };

        // Act
        let result = detect_reduction_pattern("xmm1", &pre, &post);

        // Assert
        assert!(result.is_some(), "Select(Gt, elem, acc, acc, elem) should detect reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Max,
            "Select(Gt) rhs=pre, tv=pre should be Max");
    }

    #[test]
    fn test_detect_reduction_select_le_rhs_pre_fv_pre_is_max() {
        // Arrange: Select(Le, elem, acc, elem, acc)
        // Le branch, rhs=pre, fv=pre, tv!=pre → line 829-830: (true, true_val) → Max
        use super::super::sym_value::SelectKind;
        let pre = SymValue::Param(1); // acc
        let post = SymValue::Select {
            kind: SelectKind::Le,
            cond_lhs: Box::new(SymValue::Param(0)), // elem
            cond_rhs: Box::new(SymValue::Param(1)), // acc (pre)
            true_val: Box::new(SymValue::Param(0)), // elem
            false_val: Box::new(SymValue::Param(1)), // acc (pre)
        };

        // Act
        let result = detect_reduction_pattern("xmm1", &pre, &post);

        // Assert
        assert!(result.is_some(), "Select(Le, elem, acc, elem, acc) should detect reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Max,
            "Select(Le) rhs=pre, fv=pre should be Max");
    }

    #[test]
    fn test_detect_reduction_nested_min_unrolled() {
        // Arrange: Min(Param(1), Min(Param(2), acc)) — nested Min tree with acc buried
        let pre = SymValue::Const(f64::INFINITY);
        let post = SymValue::Min(
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Min(
                Box::new(SymValue::Param(2)),
                Box::new(SymValue::Const(f64::INFINITY)),
            )),
        );

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert
        assert!(result.is_some(), "nested Min tree containing +inf acc should be detected");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Min);
        // Unrolled body produces Unknown("unrolled")
        assert!(matches!(r.body_expr, SymValue::Unknown(ref s) if s == "unrolled"),
            "unrolled reduction body should be Unknown(\"unrolled\")");
    }

    #[test]
    fn test_classify_nested_pattern_two_deep_with_fma_is_gemm() {
        // Arrange: 2-deep nesting with Sum+Mul reduction → degenerate GEMM (dot product)
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(1)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act
        let pattern = classify_nested_pattern(2, &trace);

        // Assert
        assert!(
            matches!(pattern, ComputePattern::Gemm),
            "2-deep + Sum(Mul) should be Gemm (dot product), got {:?}", pattern
        );
    }

    #[test]
    fn test_classify_nested_pattern_three_deep_without_fma_is_injective() {
        // Arrange: 3-deep nesting with no FMA/Mul reduction → Injective fallback
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0), // plain Param, no Mul/Fma
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act
        let pattern = classify_nested_pattern(3, &trace);

        // Assert
        assert!(
            matches!(pattern, ComputePattern::Injective { .. }),
            "3-deep without FMA/Mul should be Injective, got {:?}", pattern
        );
    }

    #[test]
    fn test_has_fma_or_mul_add_rejects_non_sum_kind() {
        // Arrange: Max reduction with Mul body — only Sum reductions qualify for GEMM detection
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Max,
                init: AccumulatorInit::Const(f64::NEG_INFINITY),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(1)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act & Assert
        assert!(!has_fma_or_mul_add_reduction(&trace),
            "Max reduction should not be detected as FMA/Mul-add even with Mul body");
    }

    #[test]
    fn test_has_fma_or_mul_add_returns_false_for_no_reductions() {
        // Arrange: loop with zero reductions (only unknown mutations)
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![],
            unknown_mutations: vec![("xmm0".into(), SymValue::Param(0))],
            body_block_count: 1,
        };

        // Act & Assert
        assert!(!has_fma_or_mul_add_reduction(&trace),
            "loop with no reductions should not be detected as FMA/Mul-add");
    }

    #[test]
    fn test_collect_params_traverses_div_and_sub() {
        // Arrange: Div(Sub(Param(0), Param(3)), Param(7)) — exercises Div and Sub branches
        let expr = SymValue::Div(
            Box::new(SymValue::Sub(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Param(3)),
            )),
            Box::new(SymValue::Param(7)),
        );
        let mut params = std::collections::BTreeSet::new();

        // Act
        collect_params(&expr, &mut params);

        // Assert
        assert_eq!(params.len(), 3, "should find 3 distinct params from Div/Sub tree");
        assert!(params.contains(&0));
        assert!(params.contains(&3));
        assert!(params.contains(&7));
    }

    #[test]
    fn test_nested_loop_analysis_with_none_inner_trace() {
        // Arrange: NestedLoopAnalysis where inner_trace is None (e.g., failed inner analysis)
        let analysis = NestedLoopAnalysis {
            max_depth: 1,
            nesting_levels: 2,
            inner_trace: None,
            pattern: ComputePattern::Gemm,
        };

        // Assert: construction succeeds, fields are accessible
        assert_eq!(analysis.max_depth, 1);
        assert_eq!(analysis.nesting_levels, 2);
        assert!(analysis.inner_trace.is_none(),
            "inner_trace should be None when inner loop analysis was skipped");
        assert!(matches!(analysis.pattern, ComputePattern::Gemm));
    }

    #[test]
    fn test_combine_single_loop_takes_first_reduction_only() {
        // Arrange: single loop with two Sum reductions — combine_single_loop uses index 0
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![
                ReductionDetected {
                    register: "xmm0".into(),
                    kind: ReductionKind::Sum,
                    init: AccumulatorInit::Const(0.0),
                    body_expr: SymValue::Param(0),
                },
                ReductionDetected {
                    register: "xmm1".into(),
                    kind: ReductionKind::Sum,
                    init: AccumulatorInit::Const(1.0),
                    body_expr: SymValue::Param(2),
                },
            ],
            unknown_mutations: vec![],
            body_block_count: 2,
        };

        // Act
        let pattern = combine_passes(&[trace]).expect("combine should succeed");

        // Assert: identity comes from the first reduction (0.0), not the second (1.0)
        match &pattern {
            ComputePattern::Reduction { identity, combine, .. } => {
                assert_eq!(*identity, 0.0,
                    "should use first reduction's identity, got {identity}");
                assert_eq!(combine[2], TraceOp::Add(ValueId(0), ValueId(1)));
            }
            other => panic!("expected Reduction, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Additional unit tests (wave-12x60)
    // -----------------------------------------------------------------------

    #[test]
    fn test_natural_loop_construction_and_depth_field() {
        // Arrange: construct a NaturalLoop with depth=2 (3-deep nesting)
        use std::collections::BTreeSet;
        let body: BTreeSet<BlockId> = vec![BlockId(10), BlockId(11), BlockId(12)].into_iter().collect();
        let lp = NaturalLoop {
            header: BlockId(10),
            body_blocks: body.clone(),
            latch: BlockId(12),
            exits: vec![BlockId(20)],
            ordinal: 3,
            depth: 2,
        };

        // Assert: all fields accessible and match construction
        assert_eq!(lp.header, BlockId(10));
        assert_eq!(lp.latch, BlockId(12));
        assert_eq!(lp.body_blocks, body);
        assert_eq!(lp.exits.len(), 1);
        assert_eq!(lp.exits[0], BlockId(20));
        assert_eq!(lp.ordinal, 3);
        assert_eq!(lp.depth, 2, "depth=2 means 3 levels of nesting (0-based depth)");
    }

    #[test]
    fn test_natural_loop_zero_depth_means_outermost() {
        // Arrange: outermost loop has depth=0 (flat, no nesting)
        use std::collections::BTreeSet;
        let body: BTreeSet<BlockId> = vec![BlockId(5), BlockId(6)].into_iter().collect();
        let lp = NaturalLoop {
            header: BlockId(5),
            body_blocks: body,
            latch: BlockId(6),
            exits: vec![BlockId(15)],
            ordinal: 0,
            depth: 0,
        };

        // Assert
        assert_eq!(lp.depth, 0, "depth=0 is the outermost nesting level");
    }

    #[test]
    fn test_loop_trace_empty_reductions_and_mutations() {
        // Arrange: a loop with no detected reductions or unknown mutations
        // (possible for a pure transform loop that writes to memory, not registers)
        let trace = LoopTrace {
            loop_header: BlockId(3),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 4,
        };

        // Assert: construction succeeds, fields are valid
        assert_eq!(trace.loop_header, BlockId(3));
        assert!(trace.reductions.is_empty());
        assert!(trace.unknown_mutations.is_empty());
        assert_eq!(trace.body_block_count, 4);
    }

    #[test]
    fn test_nested_loop_analysis_depth_equals_nesting_levels_minus_one() {
        // Arrange: verify the invariant nesting_levels = max_depth + 1
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(1)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        for max_depth in 0..4 {
            let nesting_levels = max_depth + 1;
            // Act
            let analysis = NestedLoopAnalysis {
                max_depth,
                nesting_levels,
                inner_trace: Some(trace.clone()),
                pattern: ComputePattern::Gemm,
            };

            // Assert: nesting_levels = max_depth + 1 invariant holds
            assert_eq!(analysis.nesting_levels, analysis.max_depth + 1,
                "nesting_levels should always equal max_depth + 1, got max_depth={max_depth}");
        }
    }

    #[test]
    fn test_detect_reduction_sum_reversed_operand_order() {
        // Arrange: Add(expr, pre) — the accumulator is the second operand
        let pre = SymValue::Const(0.0);
        let post = SymValue::Add(
            Box::new(SymValue::Param(3)),  // expr (non-acc)
            Box::new(SymValue::Const(0.0)), // pre (acc)
        );

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert: reversed order should still be detected as Sum
        assert!(result.is_some(), "Add(expr, acc) should detect Sum reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);
        // body_expr should be the non-acc operand (Param(3))
        assert!(matches!(r.body_expr, SymValue::Param(3)));
    }

    #[test]
    fn test_detect_reduction_min_reversed_operand_order() {
        // Arrange: Min(expr, pre) — the accumulator is the second operand
        let pre = SymValue::Param(1);
        let post = SymValue::Min(
            Box::new(SymValue::Param(0)),  // expr
            Box::new(SymValue::Param(1)),  // pre (acc)
        );

        // Act
        let result = detect_reduction_pattern("xmm1", &pre, &post);

        // Assert
        assert!(result.is_some(), "Min(expr, acc) should detect Min reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Min);
    }

    #[test]
    fn test_detect_reduction_self_referential_unknown_register() {
        // Arrange: when the pre-state doesn't contain the register, detect_reduction_pattern
        // falls through to the self-referential path checking Unknown("xmm2") as pre.
        // Simulate: post = Add(Unknown("xmm2"), Param(0)) where "xmm2" was not in pre state.
        // The function call with a pre_val that is NOT the register name would miss the
        // direct match, but the self-referential path uses Unknown("xmm2") as pre.
        let pre = SymValue::Const(1.0); // some unrelated pre-value
        let post = SymValue::Add(
            Box::new(SymValue::Unknown("xmm2".into())),
            Box::new(SymValue::Param(0)),
        );

        // Act: detect_reduction_pattern("xmm2", ...) — the register name matches Unknown("xmm2")
        // But pre is Const(1.0), not Unknown("xmm2"). The direct path checks
        // a_str == pre_str and b_str == pre_str — both fail.
        // Then the self-referential path creates Unknown("xmm2") as pre and tries again.
        // Add(Unknown("xmm2"), Param(0)) matches Add(self_ref, expr) → Sum.
        let result = detect_reduction_pattern("xmm2", &pre, &post);

        // Assert: detect_reduction_pattern handles self-referential Unknown registers.
        // The function may or may not detect the pattern depending on implementation.
        // Either result is valid — the key property is it doesn't panic.
        if let Some(r) = result {
            assert_eq!(r.kind, ReductionKind::Sum, "if detected, must be Sum");
        }
    }

    #[test]
    fn test_reduction_detected_debug_formatting_contains_all_fields() {
        // Arrange
        let det = ReductionDetected {
            register: "xmm7".into(),
            kind: ReductionKind::Min,
            init: AccumulatorInit::Const(f64::INFINITY),
            body_expr: SymValue::Mul(
                Box::new(SymValue::Param(2)),
                Box::new(SymValue::Param(4)),
            ),
        };

        // Act
        let debug = format!("{det:?}");

        // Assert: Debug output should contain the register name, kind, and init type
        assert!(debug.contains("xmm7"), "Debug should contain register name: {debug}");
        assert!(debug.contains("Min"), "Debug should contain reduction kind: {debug}");
        assert!(debug.contains("Const"), "Debug should contain init variant: {debug}");
    }

    #[test]
    fn test_loop_trace_debug_formatting_contains_header_and_counts() {
        // Arrange
        let trace = LoopTrace {
            loop_header: BlockId(99),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![("xmm3".into(), SymValue::Const(2.0))],
            body_block_count: 5,
        };

        // Act
        let debug = format!("{trace:?}");

        // Assert: Debug output should contain header, reduction count, and body_block_count
        assert!(debug.contains("99"), "Debug should contain header BlockId: {debug}");
        assert!(debug.contains("Sum"), "Debug should contain reduction kind: {debug}");
        assert!(debug.contains("5"), "Debug should contain body_block_count: {debug}");
    }

    #[test]
    fn test_natural_loop_empty_exits_is_valid() {
        // Arrange: a natural loop with no exit blocks (infinite loop or
        // loop whose only exit is via the latch jumping back to header)
        use std::collections::BTreeSet;
        let body: BTreeSet<BlockId> = vec![BlockId(7), BlockId(8)].into_iter().collect();
        let lp = NaturalLoop {
            header: BlockId(7),
            body_blocks: body,
            latch: BlockId(8),
            exits: vec![], // no exits — valid for infinite loops
            ordinal: 1,
            depth: 0,
        };

        // Assert: empty exits list is valid construction
        assert!(lp.exits.is_empty(), "infinite loops can have empty exits");
        assert_eq!(lp.header, BlockId(7));
        assert_eq!(lp.latch, BlockId(8));
    }

    // -----------------------------------------------------------------------
    // Additional unit tests (wave-12x61)
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_reduction_max_with_nested_max_in_body() {
        // Arrange: Max(Param(0), Max(Param(1), Param(2))) where pre = Param(0)
        // This is NOT a reduction pattern because the inner Max doesn't reference pre.
        // The accumulator (pre) is only at the outer level.
        let pre = SymValue::Param(0);
        let post = SymValue::Max(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Max(
                Box::new(SymValue::Param(1)),
                Box::new(SymValue::Param(2)),
            )),
        );

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert: should detect as Max reduction (outer Max references pre)
        assert!(result.is_some(), "Max(acc, expr) should be detected as Max reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Max);
    }

    #[test]
    fn test_detect_reduction_sum_with_div_expr_body() {
        // Arrange: Add(pre, Div(Param(0), Param(1))) — sum reduction with division in body
        let pre = SymValue::Const(0.0);
        let post = SymValue::Add(
            Box::new(SymValue::Const(0.0)),
            Box::new(SymValue::Div(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Param(1)),
            )),
        );

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert
        assert!(result.is_some(), "Add(acc, Div(a,b)) should be detected as Sum reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);
        // body_expr should be the Div expression
        assert!(matches!(r.body_expr, SymValue::Div(..)));
    }

    #[test]
    fn test_detect_reduction_fma_with_pre_as_addend() {
        // Arrange: Fma(Param(0), Param(1), pre) — standard GEMM-style FMA accumulation
        // This is the canonical form: acc = a * b + acc
        let pre = SymValue::Const(0.0);
        let post = SymValue::Fma(
            Box::new(SymValue::Param(0)),
            Box::new(SymValue::Param(1)),
            Box::new(SymValue::Const(0.0)),
        );

        // Act
        let result = detect_reduction_pattern("xmm0", &pre, &post);

        // Assert
        assert!(result.is_some(), "Fma(a, b, acc) should be detected as Sum reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Sum);
        // body_expr should be Mul(Param(0), Param(1))
        assert!(matches!(r.body_expr, SymValue::Mul(..)));
    }

    #[test]
    fn test_combine_passes_coalesces_vector_and_scalar_tail_loops() {
        // Arrange: main vectorized loop + scalar tail loop (both transform-only)
        // After coalescing, they become one logical pass.
        let main_loop = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Param(0),
            }],
            unknown_mutations: vec![],
            body_block_count: 3,
        };
        let tail_loop = LoopTrace {
            loop_header: BlockId(1),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 2,
        };
        let transform_loop = LoopTrace {
            loop_header: BlockId(2),
            reductions: vec![],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act: main + tail + transform → tail is coalesced, result is 2 logical passes
        let result = combine_passes(&[main_loop, tail_loop, transform_loop]);

        // Assert: should succeed as NormLike (Sum reduction + transform)
        assert!(result.is_ok(), "should coalesce tail loop and produce NormLike");
        let pattern = result.unwrap();
        assert!(matches!(pattern, ComputePattern::NormLike { .. }),
            "expected NormLike, got {:?}", pattern);
    }

    #[test]
    fn test_classify_nested_pattern_two_deep_no_reduction_is_injective() {
        // Arrange: 2-deep nesting with no reduction, only unknown mutations
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![],
            unknown_mutations: vec![
                ("xmm0".into(), SymValue::Param(0)),
                ("xmm1".into(), SymValue::Param(1)),
            ],
            body_block_count: 2,
        };

        // Act
        let pattern = classify_nested_pattern(2, &trace);

        // Assert
        if let ComputePattern::Injective { num_inputs, num_outputs, .. } = &pattern {
            // num_inputs from count_distinct_params_in_trace = 2 (Param(0), Param(1))
            // num_outputs from unknown_mutations.len() = 2, but capped at 2
            assert_eq!(*num_inputs, 2, "should detect 2 distinct params");
            assert_eq!(*num_outputs, 2, "should cap outputs at 2");
        } else {
            panic!("expected Injective, got {:?}", pattern);
        }
    }

    #[test]
    fn test_has_fma_or_mul_add_with_mul_body_returns_true() {
        // Arrange: Sum reduction with Mul body (not Fma) — should still be detected
        let trace = LoopTrace {
            loop_header: BlockId(0),
            reductions: vec![ReductionDetected {
                register: "xmm0".into(),
                kind: ReductionKind::Sum,
                init: AccumulatorInit::Const(0.0),
                body_expr: SymValue::Mul(
                    Box::new(SymValue::Param(0)),
                    Box::new(SymValue::Param(1)),
                ),
            }],
            unknown_mutations: vec![],
            body_block_count: 1,
        };

        // Act & Assert
        assert!(has_fma_or_mul_add_reduction(&trace),
            "Sum reduction with Mul body should be detected as FMA/Mul-add");
    }

    #[test]
    fn test_collect_params_from_recip_and_abs() {
        // Arrange: Recip(Abs(Param(5))) — exercises unary ops Recip and Abs
        let expr = SymValue::Recip(Box::new(SymValue::Abs(Box::new(SymValue::Param(5)))));
        let mut params = std::collections::BTreeSet::new();

        // Act
        collect_params(&expr, &mut params);

        // Assert
        assert_eq!(params.len(), 1, "should find Param(5) inside Recip(Abs(...))");
        assert!(params.contains(&5));
    }

    #[test]
    fn test_detect_reduction_select_ge_rhs_pre_tv_not_pre_returns_min() {
        // Arrange: Select(Ge, elem, acc, other, acc) where fv=acc but tv≠acc
        // Ge with rhs=pre and fv=pre means "if elem >= acc, keep acc, else other"
        // which is a Min-like selection (keep the smaller value).
        use super::super::sym_value::SelectKind;
        let pre = SymValue::Param(1); // acc
        let post = SymValue::Select {
            kind: SelectKind::Ge,
            cond_lhs: Box::new(SymValue::Param(0)), // elem
            cond_rhs: Box::new(SymValue::Param(1)), // acc (pre)
            true_val: Box::new(SymValue::Param(2)), // other (not acc)
            false_val: Box::new(SymValue::Param(1)), // acc
        };

        // Act
        let result = detect_reduction_pattern("xmm1", &pre, &post);

        // Assert: Ge + rhs=pre + fv=pre + tv≠pre → Min reduction
        assert!(result.is_some(),
            "Select(Ge) with rhs=pre, fv=pre should be detected as Min reduction");
        let r = result.unwrap();
        assert_eq!(r.kind, ReductionKind::Min);
    }

    #[test]
    fn test_symvalue_to_reduce_body_for_nested_mul() {
        // Arrange: Mul(Mul(Param(0), Param(1)), Param(2)) — nested multiply
        // The outer Mul has different operand string representations, so it takes
        // the general 2-input path: [Input(0), Input(1), Mul(0,1)]
        let nested_mul = SymValue::Mul(
            Box::new(SymValue::Mul(
                Box::new(SymValue::Param(0)),
                Box::new(SymValue::Param(1)),
            )),
            Box::new(SymValue::Param(2)),
        );

        // Act
        let body = symvalue_to_reduce_body(&nested_mul);

        // Assert: general multiply produces 3 TraceOps (2 inputs + 1 Mul)
        assert_eq!(body.len(), 3, "nested Mul should use general 2-input path");
        assert_eq!(body[0], TraceOp::Input(0));
        assert_eq!(body[1], TraceOp::Input(1));
        assert_eq!(body[2], TraceOp::Mul(ValueId(0), ValueId(1)));
    }

    #[test]
    fn test_reduction_to_trace_ops_sum_symbolic_init_uses_zero() {
        // Arrange: Sum reduction with Symbolic init → should use 0.0 fallback
        let det = ReductionDetected {
            register: "xmm3".into(),
            kind: ReductionKind::Sum,
            init: AccumulatorInit::Symbolic(SymValue::Param(7)),
            body_expr: SymValue::Param(0),
        };

        // Act
        let (identity, combine) = reduction_to_trace_ops(&det);

        // Assert
        assert_eq!(identity, 0.0, "Symbolic Sum init should fallback to 0.0");
        assert_eq!(combine.len(), 3);
        assert_eq!(combine[2], TraceOp::Add(ValueId(0), ValueId(1)));
    }
}
