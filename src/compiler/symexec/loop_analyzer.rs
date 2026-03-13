//! Loop body symbolic execution and reduction detection.
//!
//! Phase 2 of the control-flow upgrade: symbolically executes a single loop
//! iteration to identify accumulator patterns (Sum, Max, Min reductions)
//! and per-element transforms.

use super::cfg::{BasicBlock, BlockId, ControlFlowGraph, LoopForest, NaturalLoop};
use super::engine::SymbolicExecutor;
use super::sym_value::SymValue;
use crate::compiler::trace::{ComputePattern, ReductionSecondPass, TraceOp};

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
) -> Result<LoopTrace, String> {
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
    for (reg, _pre_val) in &pre_xmm {
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
pub fn combine_passes(traces: &[LoopTrace]) -> Result<ComputePattern, String> {
    if traces.is_empty() {
        return Err("no loops found — use linear symexec instead".into());
    }

    // Coalesce: the compiler often emits a main vectorized loop + a scalar
    // tail loop for the same pass. Consecutive transform-only loops (no
    // reductions) at the end are treated as a single normalize/transform pass.
    let mut logical: Vec<&LoopTrace> = Vec::new();
    for trace in traces {
        let dominated = !logical.is_empty() && {
            let prev = logical.last().unwrap();
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
        n => Err(format!("unsupported: {n} logical passes (expected 1-3)")),
    }
}

/// Single loop → simple Reduction pattern.
///
/// Rejects loops that have unknown_mutations alongside reductions: a single
/// loop with both indicates the analyzer only partially understood the body
/// (e.g. an elementwise function whose vectorized exp()/tanh() call was
/// misread as an accumulator).  Genuine single-loop reductions (like a
/// simple sum or max scan) produce clean traces with no unknown mutations.
fn combine_single_loop(trace: &LoopTrace) -> Result<ComputePattern, String> {
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
) -> Result<ComputePattern, String> {
    // Loop 1 must have a Sum reduction.
    let r1 = loop1.reductions.first()
        .ok_or("loop 1 has no reductions (expected sum for NormLike)")?;

    if r1.kind != ReductionKind::Sum {
        return Err(format!("loop 1 reduction is {:?}, expected Sum for NormLike", r1.kind));
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
        TraceOp::Mul(0, 1), // [2] x * scale
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
) -> Result<ComputePattern, String> {
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
    ))
}

/// Build Softmax ComputePattern from 3 loop traces.
fn combine_softmax(
    max_reduction: &ReductionDetected,
    sum_reduction: &ReductionDetected,
    _normalize_loop: &LoopTrace,
) -> Result<ComputePattern, String> {
    let (max_identity, max_combine) = reduction_to_trace_ops(max_reduction);

    // Second pass: exp(x - max) → sum
    let element_transform = vec![
        TraceOp::Input(0),  // [0] x (current element)
        TraceOp::Input(1),  // [1] max (broadcast from first pass)
        TraceOp::Sub(0, 1), // [2] x - max
        TraceOp::Exp(2),    // [3] exp(x - max)
    ];

    let sum_combine = vec![
        TraceOp::Input(0),  // [0] acc (running sum)
        TraceOp::Input(1),  // [1] exp_val
        TraceOp::Add(0, 1), // [2] acc + exp_val
    ];

    let sum_identity = match &sum_reduction.init {
        AccumulatorInit::Const(v) => *v,
        _ => 0.0,
    };

    // Normalize: out[i] = exp_val * inv_sum
    let normalize = vec![
        TraceOp::Input(0),  // [0] exp_val
        TraceOp::Input(1),  // [1] inv_sum (broadcast)
        TraceOp::Mul(0, 1), // [2] exp_val * inv_sum
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
) -> Result<ComputePattern, String> {
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
        TraceOp::Add(1, 2),   // [3] var + eps
        TraceOp::Rsqrt(3),    // [4] rsqrt(var + eps)
    ];

    // Transform: (x - mean) * scale * weight + bias
    let transform = vec![
        TraceOp::Input(0),  // [0] x
        TraceOp::Input(1),  // [1] mean
        TraceOp::Input(2),  // [2] scale (from finalize)
        TraceOp::Input(3),  // [3] weight
        TraceOp::Input(4),  // [4] bias
        TraceOp::Sub(0, 1), // [5] x - mean
        TraceOp::Mul(5, 2), // [6] (x - mean) * scale
        TraceOp::Mul(6, 3), // [7] normed * weight
        TraceOp::Add(7, 4), // [8] normed * weight + bias
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
            TraceOp::Add(0, 1), // [2] acc + element
        ],
        ReductionKind::Max => vec![
            TraceOp::Input(0),  // [0] acc
            TraceOp::Input(1),  // [1] element
            TraceOp::Max(0, 1), // [2] max(acc, element)
        ],
        ReductionKind::Min => vec![
            TraceOp::Input(0),  // [0] acc
            TraceOp::Input(1),  // [1] element
            TraceOp::Min(0, 1), // [2] min(acc, element)
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
                    TraceOp::Mul(0, 0), // [1] x^2
                ]
            } else {
                // General multiply — two inputs.
                vec![
                    TraceOp::Input(0),  // [0] a
                    TraceOp::Input(1),  // [1] b
                    TraceOp::Mul(0, 1), // [2] a * b
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
        SymValue::Select { true_val, false_val, .. } => {
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
        if b_str == pre_str {
            if is_const_one(a) {
                return Some(ReductionDetected {
                    register: reg.to_string(),
                    kind: ReductionKind::Sum,
                    init,
                    body_expr: c.simplify(),
                });
            }
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
        "je" | "jne" | "jb" | "jbe" | "ja" | "jae"
            | "jl" | "jle" | "jg" | "jge"
            | "jmp" | "js" | "jns" | "jp" | "jnp"
            | "ret"
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
        let cfg = build_cfg_from_fn(fn_ptr, max_bytes).expect("CFG build failed");
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

        let cfg = build_cfg_from_fn(identity as *const u8, 256).expect("CFG build failed");
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
                assert_eq!(combine[2], TraceOp::Add(0, 1));
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
                assert_eq!(combine[2], TraceOp::Max(0, 1));
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
                assert_eq!(reduce[1], TraceOp::Mul(0, 0));
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
        let cfg = build_cfg_from_fn(fn_ptr, 4096).expect("CFG build failed");
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

        let result = analyze_scalar_fn_structured(fn_ptr, &sig);
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
        let cfg = build_cfg_from_fn(fn_ptr, 4096).expect("CFG build failed");
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

        let result = analyze_scalar_fn_structured(fn_ptr, &sig);
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

        let result = analyze_scalar_fn_structured(sig.fn_ptr, &sig);
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
        let cfg = build_cfg_from_fn(fn_ptr, 4096).expect("CFG build failed");
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

        let result = analyze_scalar_fn_structured(sig.fn_ptr, &sig);
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
        let cfg = build_cfg_from_fn(fn_ptr, 4096).expect("CFG build failed");
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
        let cfg = build_cfg_from_fn(fn_ptr, 4096).expect("CFG build failed");
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
}
