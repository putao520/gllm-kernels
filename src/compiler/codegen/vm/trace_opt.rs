//! TraceOp IR 优化 Pass Pipeline
//!
//! 在 `auto_lower_trace_raw` 消费 body 之前执行 IR 层优化。
//! 与 `opt_pass.rs`（VmInstr 层 post-lowering 优化）互补，
//! 本模块在 TraceOp IR 层（pre-lowering）消除冗余计算。
//!
//! 架构: `TraceOptPass` trait + `TracePassPipeline`，
//! 镜像 `opt_pass.rs` 的 `VmOptPass` + `PassRegistry` 模式。
//!
//! 默认 pipeline: DCE → CSE → ConstFold → DCE

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::compiler::trace::{TraceOp, ValueId};

/// IR 层优化统计
#[derive(Debug, Default, Clone, Copy)]
pub struct TraceOptStats {
    pub ops_removed: usize,
    pub ops_folded: usize,
    pub ops_merged: usize,
}

impl std::ops::AddAssign for TraceOptStats {
    fn add_assign(&mut self, other: Self) {
        self.ops_removed += other.ops_removed;
        self.ops_folded += other.ops_folded;
        self.ops_merged += other.ops_merged;
    }
}

/// TraceOp IR 优化 Pass trait
pub trait TraceOptPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn priority(&self) -> u32;
    fn run(&self, body: &mut Vec<TraceOp>) -> TraceOptStats;
}

/// Pass pipeline — 注册、排序、执行
pub struct TracePassPipeline {
    passes: Vec<Box<dyn TraceOptPass>>,
}

impl TracePassPipeline {
    pub fn new() -> Self {
        Self { passes: vec![] }
    }

    pub fn register(&mut self, pass: Box<dyn TraceOptPass>) {
        self.passes.push(pass);
    }

    /// 默认 pipeline: DCE → CSE → ConstFold → DCE
    pub fn with_defaults() -> Self {
        let mut pipeline = Self::new();
        pipeline.register(Box::new(DeadCodeElimination));
        pipeline.register(Box::new(CommonSubexprElimination));
        pipeline.register(Box::new(ConstantFolding));
        pipeline.register(Box::new(DeadCodeElimination));
        pipeline
    }

    /// 按优先级排序后顺序执行
    pub fn optimize(&self, body: &mut Vec<TraceOp>) -> TraceOptStats {
        let mut total = TraceOptStats::default();
        let mut sorted: Vec<&dyn TraceOptPass> =
            self.passes.iter().map(|p| p.as_ref()).collect();
        sorted.sort_by_key(|p| p.priority());
        for pass in sorted {
            total += pass.run(body);
        }
        total
    }
}

/// Compute a structural hash of a pure TraceOp for CSE dedup.
/// Handles f64 by hashing its bit representation.
fn compute_op_hash(op: &TraceOp) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::mem::discriminant(op).hash(&mut hasher);
    op.visit_value_ids(|vid| vid.0.hash(&mut hasher));
    if let TraceOp::Const(v) = op {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 1: Dead Code Elimination
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct DeadCodeElimination;

impl TraceOptPass for DeadCodeElimination {
    fn name(&self) -> &'static str { "dce" }
    fn priority(&self) -> u32 { 10 }

    fn run(&self, body: &mut Vec<TraceOp>) -> TraceOptStats {
        if body.is_empty() {
            return TraceOptStats::default();
        }

        // 1. Collect all used ValueIds
        let mut used = HashSet::new();
        for op in body.iter() {
            op.visit_value_ids(|vid| {
                used.insert(vid);
            });
        }

        // 2. Mark alive (used or has side effects or is the last op = output)
        let last = body.len() - 1;
        let alive: Vec<bool> = body
            .iter()
            .enumerate()
            .map(|(i, op)| {
                i == last || used.contains(&ValueId(i as u32)) || op.has_side_effects()
            })
            .collect();

        let removed = alive.iter().filter(|&&a| !a).count();
        if removed == 0 {
            return TraceOptStats::default();
        }

        // 3. Build remap table: old index → new index
        let mut remap = vec![0u32; body.len()];
        let mut next = 0u32;
        for (i, is_alive) in alive.iter().enumerate() {
            if *is_alive {
                remap[i] = next;
                next += 1;
            } else {
                remap[i] = u32::MAX; // dead marker
            }
        }

        // 4. Compact: remove dead ops, remap ValueIds
        let map_fn = |vid: ValueId| -> ValueId {
            let idx = vid.0 as usize;
            if idx < remap.len() {
                ValueId(remap[idx])
            } else {
                vid
            }
        };

        let mut new_body = Vec::with_capacity(body.len() - removed);
        for (i, op) in body.drain(..).enumerate() {
            if alive[i] {
                new_body.push(op.map_value_ids(&map_fn));
            }
        }
        *body = new_body;

        TraceOptStats {
            ops_removed: removed,
            ..TraceOptStats::default()
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 2: Common Subexpression Elimination
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct CommonSubexprElimination;

impl TraceOptPass for CommonSubexprElimination {
    fn name(&self) -> &'static str { "cse" }
    fn priority(&self) -> u32 { 20 }

    fn run(&self, body: &mut Vec<TraceOp>) -> TraceOptStats {
        if body.len() < 2 {
            return TraceOptStats::default();
        }

        // 1. Build remap: duplicate → original (using hash-based dedup)
        let mut remap = vec![ValueId(u32::MAX); body.len()];
        let mut seen: HashMap<u64, u32> = HashMap::new(); // hash → first occurrence index
        let mut merged = 0usize;

        for i in 0..body.len() {
            let op = &body[i];
            if !op.is_pure() {
                continue;
            }
            let canonical = remap_op(op, &remap);
            let hash = compute_op_hash(&canonical);
            if let Some(&original_idx) = seen.get(&hash) {
                // Verify structural equality (hash collision defense)
                if ops_structurally_equal(&canonical, &body[original_idx as usize], &remap) {
                    remap[i] = ValueId(original_idx);
                    merged += 1;
                    continue;
                }
            }
            seen.insert(hash, i as u32);
        }

        if merged == 0 {
            return TraceOptStats::default();
        }

        // 2. Apply remap to all references
        let map_fn = |vid: ValueId| -> ValueId {
            let idx = vid.0 as usize;
            if idx < remap.len() && remap[idx].0 != u32::MAX {
                remap[idx]
            } else {
                vid
            }
        };
        for op in body.iter_mut() {
            let mapped = std::mem::replace(op, TraceOp::Const(0.0)).map_value_ids(&map_fn);
            *op = mapped;
        }

        TraceOptStats {
            ops_merged: merged,
            ..TraceOptStats::default()
        }
    }
}

/// Canonicalize an op's ValueId references through the remap table.
fn remap_op(op: &TraceOp, remap: &[ValueId]) -> TraceOp {
    let map_fn = |vid: ValueId| -> ValueId {
        let idx = vid.0 as usize;
        if idx < remap.len() && remap[idx].0 != u32::MAX {
            remap[idx]
        } else {
            vid
        }
    };
    op.clone().map_value_ids(&map_fn)
}

/// Structural equality check for CSE (avoids needing Eq trait on TraceOp).
/// Compares discriminant + all ValueId fields + f64 bits for Const.
fn ops_structurally_equal(a: &TraceOp, b: &TraceOp, _remap: &[ValueId]) -> bool {
    if std::mem::discriminant(a) != std::mem::discriminant(b) {
        return false;
    }
    // Compare ValueId fields
    let mut a_ids = Vec::new();
    let mut b_ids = Vec::new();
    a.visit_value_ids(|vid| a_ids.push(vid));
    b.visit_value_ids(|vid| b_ids.push(vid));
    if a_ids != b_ids {
        return false;
    }
    // For Const, compare bit-exact f64
    if let (TraceOp::Const(va), TraceOp::Const(vb)) = (a, b) {
        return va.to_bits() == vb.to_bits();
    }
    true
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 3: Constant Folding
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct ConstantFolding;

impl TraceOptPass for ConstantFolding {
    fn name(&self) -> &'static str { "const_fold" }
    fn priority(&self) -> u32 { 30 }

    fn run(&self, body: &mut Vec<TraceOp>) -> TraceOptStats {
        if body.is_empty() {
            return TraceOptStats::default();
        }

        let n = body.len();
        let mut known_constants: HashMap<u32, f64> = HashMap::new();
        let mut remap = vec![ValueId(u32::MAX); n];
        let mut folded = 0usize;

        // Two-phase: (1) collect + update known_constants, (2) apply body mutations
        // Phase 1: scan and record replacements
        let mut replacements: Vec<(usize, TraceOp)> = Vec::new();
        for i in 0..n {
            let vid = i as u32;
            // Clone to release borrow before mutation
            let op = body[i].clone();

            match &op {
                TraceOp::Const(v) => {
                    known_constants.insert(vid, *v);
                }
                TraceOp::Neg(a) => {
                    if let Some(va) = known_constants.get(&a.0) {
                        let result = -va;
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                TraceOp::Abs(a) => {
                    if let Some(va) = known_constants.get(&a.0) {
                        let result = va.abs();
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                TraceOp::Sqrt(a) => {
                    if let Some(va) = known_constants.get(&a.0) {
                        let result = va.sqrt();
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                TraceOp::Exp(a) => {
                    if let Some(va) = known_constants.get(&a.0) {
                        let result = va.exp();
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                TraceOp::Log(a) => {
                    if let Some(va) = known_constants.get(&a.0) {
                        let result = va.ln();
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                TraceOp::Rsqrt(a) => {
                    if let Some(va) = known_constants.get(&a.0) {
                        let result = 1.0 / va.sqrt();
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                TraceOp::Recip(a) => {
                    if let Some(va) = known_constants.get(&a.0) {
                        if *va != 0.0 {
                            let result = 1.0 / va;
                            known_constants.insert(vid, result);
                            replacements.push((i, TraceOp::Const(result)));
                            folded += 1;
                        }
                    }
                }
                TraceOp::Add(a, b) => {
                    let va = known_constants.get(&a.0);
                    let vb = known_constants.get(&b.0);
                    match (va, vb) {
                        (Some(va), Some(vb)) => {
                            let result = va + vb;
                            known_constants.insert(vid, result);
                            replacements.push((i, TraceOp::Const(result)));
                            folded += 1;
                        }
                        (Some(va), None) if *va == 0.0 => {
                            remap[i] = *b;
                            folded += 1;
                        }
                        (None, Some(vb)) if *vb == 0.0 => {
                            remap[i] = *a;
                            folded += 1;
                        }
                        _ => {}
                    }
                }
                TraceOp::Sub(a, b) => {
                    let va = known_constants.get(&a.0);
                    let vb = known_constants.get(&b.0);
                    if let (Some(va), Some(vb)) = (va, vb) {
                        let result = va - vb;
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    } else if let Some(vb) = vb {
                        if *vb == 0.0 {
                            remap[i] = *a;
                            folded += 1;
                        }
                    }
                }
                TraceOp::Mul(a, b) => {
                    let va = known_constants.get(&a.0);
                    let vb = known_constants.get(&b.0);
                    match (va, vb) {
                        (Some(va), Some(vb)) => {
                            let result = va * vb;
                            known_constants.insert(vid, result);
                            replacements.push((i, TraceOp::Const(result)));
                            folded += 1;
                        }
                        (Some(va), None) if *va == 1.0 => {
                            remap[i] = *b;
                            folded += 1;
                        }
                        (Some(_va), None)
                            if known_constants.get(&a.0).is_some_and(|v| *v == 0.0) =>
                        {
                            known_constants.insert(vid, 0.0);
                            replacements.push((i, TraceOp::Const(0.0)));
                            folded += 1;
                        }
                        (None, Some(vb)) if *vb == 1.0 => {
                            remap[i] = *a;
                            folded += 1;
                        }
                        (None, Some(_vb))
                            if known_constants.get(&b.0).is_some_and(|v| *v == 0.0) =>
                        {
                            known_constants.insert(vid, 0.0);
                            replacements.push((i, TraceOp::Const(0.0)));
                            folded += 1;
                        }
                        _ => {}
                    }
                }
                TraceOp::Div(a, b) => {
                    let va = known_constants.get(&a.0);
                    let vb = known_constants.get(&b.0);
                    if let (Some(va), Some(vb)) = (va, vb) {
                        if *vb != 0.0 {
                            let result = va / vb;
                            known_constants.insert(vid, result);
                            replacements.push((i, TraceOp::Const(result)));
                            folded += 1;
                        }
                    } else if let Some(vb) = vb {
                        if *vb == 1.0 {
                            remap[i] = *a;
                            folded += 1;
                        }
                    }
                }
                TraceOp::Max(a, b) => {
                    if a == b {
                        remap[i] = *a;
                        folded += 1;
                    } else if let (Some(va), Some(vb)) =
                        (known_constants.get(&a.0), known_constants.get(&b.0))
                    {
                        let result = va.max(*vb);
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                TraceOp::Min(a, b) => {
                    if a == b {
                        remap[i] = *a;
                        folded += 1;
                    } else if let (Some(va), Some(vb)) =
                        (known_constants.get(&a.0), known_constants.get(&b.0))
                    {
                        let result = va.min(*vb);
                        known_constants.insert(vid, result);
                        replacements.push((i, TraceOp::Const(result)));
                        folded += 1;
                    }
                }
                _ => {}
            }
        }

        if folded == 0 {
            return TraceOptStats::default();
        }

        // Phase 2: apply replacements
        for (i, new_op) in replacements {
            body[i] = new_op;
        }

        // Apply identity remaps
        let map_fn = |vid: ValueId| -> ValueId {
            let idx = vid.0 as usize;
            if idx < remap.len() && remap[idx].0 != u32::MAX {
                remap[idx]
            } else {
                vid
            }
        };
        for op in body.iter_mut() {
            let mapped = std::mem::replace(op, TraceOp::Const(0.0)).map_value_ids(&map_fn);
            *op = mapped;
        }

        TraceOptStats {
            ops_folded: folded,
            ..TraceOptStats::default()
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dce_removes_unused_ops() {
        let mut body = vec![
            TraceOp::Input(0),   // [0] x — used by Mul
            TraceOp::Input(1),   // [1] y — used by Add
            TraceOp::Mul(ValueId(0), ValueId(0)),  // [2] x*x — DEAD
            TraceOp::Add(ValueId(0), ValueId(1)),  // [3] x+y — output
        ];
        let stats = DeadCodeElimination.run(&mut body);
        assert_eq!(stats.ops_removed, 1);
        assert_eq!(body.len(), 3);
        assert!(matches!(body[2], TraceOp::Add(ValueId(0), ValueId(1))));
    }

    #[test]
    fn test_cse_merges_identical_ops() {
        let mut body = vec![
            TraceOp::Input(0),   // [0] x
            TraceOp::Input(1),   // [1] y
            TraceOp::Mul(ValueId(0), ValueId(1)),  // [2] x*y (original)
            TraceOp::Add(ValueId(0), ValueId(1)),  // [3] x+y (different)
            TraceOp::Mul(ValueId(0), ValueId(1)),  // [4] x*y (duplicate of [2])
        ];
        let stats = CommonSubexprElimination.run(&mut body);
        assert_eq!(stats.ops_merged, 1);
    }

    #[test]
    fn test_const_fold_binary() {
        let mut body = vec![
            TraceOp::Const(2.0),  // [0]
            TraceOp::Const(3.0),  // [1]
            TraceOp::Add(ValueId(0), ValueId(1)),   // [2] → Const(5.0)
            TraceOp::Const(0.0),  // [3]
            TraceOp::Add(ValueId(2), ValueId(3)),   // [4] → identity (x+0=x)
        ];
        let stats = ConstantFolding.run(&mut body);
        assert_eq!(stats.ops_folded, 2);
        assert!(matches!(body[2], TraceOp::Const(5.0)));
    }

    #[test]
    fn test_const_fold_mul_identity() {
        let mut body = vec![
            TraceOp::Input(0),     // [0] x
            TraceOp::Const(1.0),   // [1] 1.0
            TraceOp::Mul(ValueId(0), ValueId(1)),    // [2] → identity (x*1=x)
        ];
        let stats = ConstantFolding.run(&mut body);
        assert_eq!(stats.ops_folded, 1);
    }

    #[test]
    fn test_const_fold_mul_zero() {
        let mut body = vec![
            TraceOp::Input(0),     // [0] x
            TraceOp::Const(0.0),   // [1] 0.0
            TraceOp::Mul(ValueId(0), ValueId(1)),    // [2] → Const(0.0)
        ];
        let stats = ConstantFolding.run(&mut body);
        assert_eq!(stats.ops_folded, 1);
        assert!(matches!(body[2], TraceOp::Const(0.0)));
    }

    #[test]
    fn test_full_pipeline_silu() {
        let mut body = vec![
            TraceOp::Input(0),   // [0] x
            TraceOp::Input(0),   // [1] x (duplicate)
            TraceOp::Mul(ValueId(0), ValueId(0)),  // [2] x*x (unused)
            TraceOp::Neg(ValueId(0)),     // [3] -x
            TraceOp::Exp(ValueId(3)),     // [4] exp(-x)
            TraceOp::Const(1.0), // [5] 1.0
            TraceOp::Add(ValueId(4), ValueId(5)),  // [6] 1 + exp(-x)
            TraceOp::Div(ValueId(0), ValueId(6)),  // [7] x / (1 + exp(-x))
        ];
        let pipeline = TracePassPipeline::with_defaults();
        let stats = pipeline.optimize(&mut body);
        assert!(stats.ops_removed >= 1, "DCE should remove unused ops, got {:?}", stats);
        assert!(body.len() <= 7, "body should shrink: {} ops", body.len());
    }

    #[test]
    fn test_pipeline_noop_on_minimal_body() {
        let mut body = vec![
            TraceOp::Input(0),
            TraceOp::Const(2.0),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];
        let pipeline = TracePassPipeline::with_defaults();
        let stats = pipeline.optimize(&mut body);
        assert_eq!(stats.ops_removed, 0);
        assert_eq!(body.len(), 3);
    }

    #[test]
    fn test_dce_preserves_side_effects() {
        let mut body = vec![
            TraceOp::Input(0),                          // [0] base
            TraceOp::Input(1),                          // [1] offset
            TraceOp::Input(2),                          // [2] value
            TraceOp::Mul(ValueId(0), ValueId(1)),       // [3] dead (nobody refs ValueId(3))
            TraceOp::VecStoreIndexed { base: ValueId(0), offset: ValueId(1), value: ValueId(2) },
            // [4] side effect + last op → always preserved
        ];
        let stats = DeadCodeElimination.run(&mut body);
        assert_eq!(stats.ops_removed, 1); // Only Mul is dead
        assert_eq!(body.len(), 4);
        assert!(matches!(body[3], TraceOp::VecStoreIndexed { .. }));
    }

    // @trace TEST-TOPT-09 [req:REQ-VR] [level:unit]
    #[test]
    fn test_dce_empty_body() {
        // Arrange: empty body
        let mut body: Vec<TraceOp> = vec![];

        // Act
        let stats = DeadCodeElimination.run(&mut body);

        // Assert
        assert_eq!(stats.ops_removed, 0);
        assert!(body.is_empty());
    }

    // @trace TEST-TOPT-10 [req:REQ-VR] [level:unit]
    #[test]
    fn test_cse_skips_impure_ops() {
        // Arrange: VecStoreIndexed has side effects so CSE must not merge it
        let mut body = vec![
            TraceOp::Input(0),   // [0] base
            TraceOp::Input(1),   // [1] offset
            TraceOp::Input(2),   // [2] value
            TraceOp::VecStoreIndexed { base: ValueId(0), offset: ValueId(1), value: ValueId(2) },
            TraceOp::VecStoreIndexed { base: ValueId(0), offset: ValueId(1), value: ValueId(2) },
        ];

        // Act
        let stats = CommonSubexprElimination.run(&mut body);

        // Assert: no merges because VecStoreIndexed is impure
        assert_eq!(stats.ops_merged, 0);
        assert_eq!(body.len(), 5);
    }

    // @trace TEST-TOPT-11 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_unary_chain() {
        // Arrange: Neg(2.0) → Const(-2.0), Abs(-3.0) → Const(3.0)
        let mut body = vec![
            TraceOp::Const(2.0),   // [0]
            TraceOp::Const(-3.0),  // [1]
            TraceOp::Neg(ValueId(0)),      // [2] → Const(-2.0)
            TraceOp::Abs(ValueId(1)),      // [3] → Const(3.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 2);
        assert!(matches!(body[2], TraceOp::Const(v) if v == -2.0));
        assert!(matches!(body[3], TraceOp::Const(v) if v == 3.0));
    }

    // @trace TEST-TOPT-12 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_sqrt_exp_log() {
        // Arrange: Sqrt(4.0) → 2.0, Exp(0.0) → 1.0, Log(1.0) → 0.0
        let mut body = vec![
            TraceOp::Const(4.0),   // [0]
            TraceOp::Const(0.0),   // [1]
            TraceOp::Const(1.0),   // [2]
            TraceOp::Sqrt(ValueId(0)),      // [3] → Const(2.0)
            TraceOp::Exp(ValueId(1)),       // [4] → Const(1.0)
            TraceOp::Log(ValueId(2)),       // [5] → Const(0.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 3);
        assert!(matches!(body[3], TraceOp::Const(v) if (v - 2.0).abs() < 1e-10));
        assert!(matches!(body[4], TraceOp::Const(v) if (v - 1.0).abs() < 1e-10));
        assert!(matches!(body[5], TraceOp::Const(v) if v.abs() < 1e-10));
    }

    // @trace TEST-TOPT-13 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_rsqrt_recip() {
        // Arrange: Rsqrt(4.0) → 0.5, Recip(2.0) → 0.5
        let mut body = vec![
            TraceOp::Const(4.0),   // [0]
            TraceOp::Const(2.0),   // [1]
            TraceOp::Rsqrt(ValueId(0)),     // [2] → Const(0.5)
            TraceOp::Recip(ValueId(1)),     // [3] → Const(0.5)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 2);
        assert!(matches!(body[2], TraceOp::Const(v) if (v - 0.5).abs() < 1e-10));
        assert!(matches!(body[3], TraceOp::Const(v) if (v - 0.5).abs() < 1e-10));
    }

    // @trace TEST-TOPT-14 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_recip_zero_guard() {
        // Arrange: Recip(0.0) must NOT be folded (division by zero guard)
        let mut body = vec![
            TraceOp::Const(0.0),       // [0]
            TraceOp::Recip(ValueId(0)), // [1] → should remain Recip
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded, Recip preserved
        assert_eq!(stats.ops_folded, 0);
        assert!(matches!(body[1], TraceOp::Recip(ValueId(0))));
    }

    // @trace TEST-TOPT-15 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_sub_zero_identity() {
        // Arrange: x - 0 → x (identity), 5.0 - 3.0 → 2.0 (full fold)
        let mut body = vec![
            TraceOp::Input(0),       // [0] x
            TraceOp::Const(0.0),     // [1] zero
            TraceOp::Const(5.0),     // [2]
            TraceOp::Const(3.0),     // [3]
            TraceOp::Sub(ValueId(0), ValueId(1)),   // [4] → identity: remap to ValueId(0)
            TraceOp::Sub(ValueId(2), ValueId(3)),   // [5] → Const(2.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 2);
        assert!(matches!(body[5], TraceOp::Const(v) if (v - 2.0).abs() < 1e-10));
    }

    // @trace TEST-TOPT-16 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_div_identity_and_full() {
        // Arrange: x / 1 → x, 6.0 / 3.0 → 2.0
        let mut body = vec![
            TraceOp::Input(0),       // [0] x
            TraceOp::Const(1.0),     // [1]
            TraceOp::Const(6.0),     // [2]
            TraceOp::Const(3.0),     // [3]
            TraceOp::Div(ValueId(0), ValueId(1)),   // [4] → identity: remap to ValueId(0)
            TraceOp::Div(ValueId(2), ValueId(3)),   // [5] → Const(2.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 2);
        assert!(matches!(body[5], TraceOp::Const(v) if (v - 2.0).abs() < 1e-10));
    }

    // @trace TEST-TOPT-17 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_div_by_zero_guard() {
        // Arrange: 5.0 / 0.0 must NOT be folded
        let mut body = vec![
            TraceOp::Const(5.0),     // [0]
            TraceOp::Const(0.0),     // [1]
            TraceOp::Div(ValueId(0), ValueId(1)),   // [2] → should remain Div
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded because divisor is zero
        assert_eq!(stats.ops_folded, 0);
        assert!(matches!(body[2], TraceOp::Div(ValueId(0), ValueId(1))));
    }

    // @trace TEST-TOPT-18 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_max_min_same_operand() {
        // Arrange: Max(x, x) → x, Min(x, x) → x (same operand identity)
        let mut body = vec![
            TraceOp::Input(0),       // [0] x
            TraceOp::Max(ValueId(0), ValueId(0)),   // [1] → remap to ValueId(0)
            TraceOp::Min(ValueId(0), ValueId(0)),   // [2] → remap to ValueId(0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 2);
    }

    // @trace TEST-TOPT-19 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_max_min_both_constants() {
        // Arrange: Max(3.0, 7.0) → 7.0, Min(3.0, 7.0) → 3.0
        let mut body = vec![
            TraceOp::Const(3.0),     // [0]
            TraceOp::Const(7.0),     // [1]
            TraceOp::Max(ValueId(0), ValueId(1)),   // [2] → Const(7.0)
            TraceOp::Min(ValueId(0), ValueId(1)),   // [3] → Const(3.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 2);
        assert!(matches!(body[2], TraceOp::Const(v) if v == 7.0));
        assert!(matches!(body[3], TraceOp::Const(v) if v == 3.0));
    }

    // @trace TEST-TOPT-20 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_mul_zero_right_operand() {
        // Arrange: x * 0.0 (right operand is zero) → Const(0.0)
        let mut body = vec![
            TraceOp::Input(0),       // [0] x
            TraceOp::Const(0.0),     // [1] 0.0
            TraceOp::Mul(ValueId(0), ValueId(1)),   // [2] → Const(0.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert
        assert_eq!(stats.ops_folded, 1);
        assert!(matches!(body[2], TraceOp::Const(0.0)));
    }

    // @trace TEST-TOPT-21 [req:REQ-VR] [level:unit]
    #[test]
    fn test_pipeline_chained_folding() {
        // Arrange: (2+3)*1 → should fold add then fold mul identity
        // ConstFold phase 1: Add(2,3) → Const(5)
        // ConstFold phase 1: Mul(5,1) → identity remap
        // Then final DCE cleans up
        let mut body = vec![
            TraceOp::Const(2.0),     // [0]
            TraceOp::Const(3.0),     // [1]
            TraceOp::Add(ValueId(0), ValueId(1)),   // [2] → Const(5.0)
            TraceOp::Const(1.0),     // [3]
            TraceOp::Mul(ValueId(2), ValueId(3)),   // [4] → identity remap to ValueId(2)
        ];

        // Act
        let pipeline = TracePassPipeline::with_defaults();
        let stats = pipeline.optimize(&mut body);

        // Assert: at least 2 folds occurred
        assert!(stats.ops_folded >= 2, "expected >=2 folds, got {:?}", stats);
    }

    // @trace TEST-TOPT-22 [req:REQ-VR] [level:unit]
    #[test]
    fn test_dce_single_op_body() {
        // Arrange: body with a single op — always alive because it is the last op
        let mut body = vec![
            TraceOp::Const(42.0),
        ];

        // Act
        let stats = DeadCodeElimination.run(&mut body);

        // Assert: nothing removed, single op preserved as output
        assert_eq!(stats.ops_removed, 0);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], TraceOp::Const(v) if v == 42.0));
    }

    // @trace TEST-TOPT-23 [req:REQ-VR] [level:unit]
    #[test]
    fn test_dce_remaps_value_ids_after_removal() {
        // Arrange: [0]=Input, [1]=Const(dead), [2]=Add(0,0) (last)
        // Const(99) is dead — nothing references ValueId(1).
        // After DCE: [0]=Input, [1]=Add(0,0). The Add's args should stay ValueId(0).
        let mut body = vec![
            TraceOp::Input(0),                              // [0] — used by Add
            TraceOp::Const(99.0),                           // [1] — dead (nobody refs ValueId(1))
            TraceOp::Add(ValueId(0), ValueId(0)),           // [2] — last, alive
        ];

        // Act
        let stats = DeadCodeElimination.run(&mut body);

        // Assert: 1 dead op removed, Add is now at index 1
        assert_eq!(stats.ops_removed, 1);
        assert_eq!(body.len(), 2);
        assert!(matches!(body[0], TraceOp::Input(0)));
        // After compaction, Add was at index 2 → now at index 1
        // Its args ValueId(0) are unchanged because index 0 is alive and maps to 0
        assert!(matches!(body[1], TraceOp::Add(ValueId(0), ValueId(0))));
    }

    // @trace TEST-TOPT-24 [req:REQ-VR] [level:unit]
    #[test]
    fn test_cse_single_element_body() {
        // Arrange: body with 1 element — CSE early-returns (len < 2)
        let mut body = vec![
            TraceOp::Neg(ValueId(0)),
        ];

        // Act
        let stats = CommonSubexprElimination.run(&mut body);

        // Assert: nothing merged
        assert_eq!(stats.ops_merged, 0);
        assert_eq!(body.len(), 1);
    }

    // @trace TEST-TOPT-25 [req:REQ-VR] [level:unit]
    #[test]
    fn test_cse_remaps_downstream_consumers() {
        // Arrange: two identical Add ops; the second is consumed by downstream Mul.
        // CSE should remap [4] to point to the original [2].
        let mut body = vec![
            TraceOp::Input(0),                              // [0] x
            TraceOp::Input(1),                              // [1] y
            TraceOp::Add(ValueId(0), ValueId(1)),           // [2] x+y (original)
            TraceOp::Mul(ValueId(0), ValueId(1)),           // [3] x*y (different)
            TraceOp::Add(ValueId(0), ValueId(1)),           // [4] x+y (duplicate → remap to ValueId(2))
            TraceOp::Mul(ValueId(4), ValueId(3)),           // [5] (x+y)*(x*y) — should become Mul(ValueId(2), ValueId(3))
        ];

        // Act
        let stats = CommonSubexprElimination.run(&mut body);

        // Assert: 1 merge; downstream consumer remapped
        assert_eq!(stats.ops_merged, 1);
        // body[5] should now reference ValueId(2) instead of ValueId(4)
        assert!(matches!(body[5], TraceOp::Mul(ValueId(2), ValueId(3))));
    }

    // @trace TEST-TOPT-26 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_mul_one_right_operand() {
        // Arrange: x * 1.0 (right operand is 1.0) → identity remap to x
        let mut body = vec![
            TraceOp::Input(0),       // [0] x
            TraceOp::Const(1.0),     // [1] 1.0
            TraceOp::Mul(ValueId(0), ValueId(1)),   // [2] → identity: remap to ValueId(0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: 1 fold (identity remap)
        assert_eq!(stats.ops_folded, 1);
    }

    // @trace TEST-TOPT-27 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_fma_not_folded() {
        // Arrange: Fma with all-constant inputs — ConstantFolding does not handle Fma,
        // so it should remain unchanged.
        let mut body = vec![
            TraceOp::Const(2.0),   // [0]
            TraceOp::Const(3.0),   // [1]
            TraceOp::Const(1.0),   // [2]
            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),  // [3] 2*3+1
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded, Fma preserved as-is
        assert_eq!(stats.ops_folded, 0);
        assert!(matches!(body[3], TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2))));
    }

    // @trace TEST-TOPT-28 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_empty_body() {
        // Arrange: empty body
        let mut body: Vec<TraceOp> = vec![];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded, body still empty
        assert_eq!(stats.ops_folded, 0);
        assert!(body.is_empty());
    }

    // @trace TEST-TOPT-29 [req:REQ-VR] [level:unit]
    #[test]
    fn test_pipeline_stats_accumulation() {
        // Arrange: ConstFold alone folds Add(5,3)→8. Verify fold stat is reported.
        let mut body = vec![
            TraceOp::Const(5.0),                            // [0]
            TraceOp::Const(3.0),                            // [1]
            TraceOp::Add(ValueId(0), ValueId(1)),           // [2] → Const(8.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: 1 fold (Add→Const)
        assert_eq!(stats.ops_folded, 1);
        assert!(matches!(body[2], TraceOp::Const(v) if (v - 8.0).abs() < 1e-10));
    }

    // @trace TEST-TOPT-30 [req:REQ-VR] [level:unit]
    #[test]
    fn test_cse_different_const_values_not_merged() {
        // Arrange: two Const ops with different values — CSE must not merge them
        // even though they share the same discriminant.
        let mut body = vec![
            TraceOp::Const(1.0),   // [0]
            TraceOp::Const(2.0),   // [1]
            TraceOp::Neg(ValueId(0)),  // [2] -1.0
            TraceOp::Neg(ValueId(1)),  // [3] -2.0
        ];

        // Act
        let stats = CommonSubexprElimination.run(&mut body);

        // Assert: no merges — Const(1.0) and Const(2.0) are structurally different,
        // Neg(ValueId(0)) and Neg(ValueId(1)) have different input ValueIds
        assert_eq!(stats.ops_merged, 0);
        assert_eq!(body.len(), 4);
    }

    // @trace TEST-TOPT-31 [req:REQ-VR] [level:unit]
    #[test]
    fn test_pipeline_cse_then_dce_removes_dead() {
        // Arrange: CSE finds duplicate Neg ops where the second is consumed downstream.
        // Run CSE first (not through default pipeline, which runs DCE first at priority 10).
        // Then DCE removes any ops that become dead after remapping.
        let mut body = vec![
            TraceOp::Input(0),                              // [0] x
            TraceOp::Neg(ValueId(0)),                       // [1] -x (original)
            TraceOp::Neg(ValueId(0)),                       // [2] -x (CSE duplicate → remap to ValueId(1))
            TraceOp::Add(ValueId(1), ValueId(2)),           // [3] (-x)+(-x) — last, alive; refs both
        ];

        // Act: run CSE first, then DCE
        let cse_stats = CommonSubexprElimination.run(&mut body);
        let dce_stats = DeadCodeElimination.run(&mut body);

        // Assert: CSE merged 1 duplicate (remap ValueId(2) → ValueId(1))
        assert_eq!(cse_stats.ops_merged, 1);
        // After remap, body[3] becomes Add(ValueId(1), ValueId(1)).
        // ValueId(2) is now dead because its result was remapped away and nobody refs it.
        assert!(dce_stats.ops_removed >= 1, "DCE should remove dead duplicate, got {:?}", dce_stats);
        assert!(body.len() < 4, "body should shrink: {} ops", body.len());
    }

    // @trace TEST-TOPT-32 [req:REQ-VR] [level:unit]
    #[test]
    fn test_empty_pipeline_does_nothing() {
        // Arrange: a body with an unused op; empty pipeline should not touch it
        let mut body = vec![
            TraceOp::Input(0),
            TraceOp::Const(99.0),                          // dead
            TraceOp::Add(ValueId(0), ValueId(0)),          // last = output
        ];
        let original_len = body.len();

        // Act: pipeline with zero registered passes
        let pipeline = TracePassPipeline::new();
        let stats = pipeline.optimize(&mut body);

        // Assert: all counters zero, body untouched
        assert_eq!(stats.ops_removed, 0);
        assert_eq!(stats.ops_folded, 0);
        assert_eq!(stats.ops_merged, 0);
        assert_eq!(body.len(), original_len);
    }

    // @trace TEST-TOPT-33 [req:REQ-VR] [level:unit]
    #[test]
    fn test_pipeline_priority_ordering() {
        // Arrange: register passes with non-default priorities and verify execution order
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ORDER: AtomicUsize = AtomicUsize::new(0);

        struct PassA;
        struct PassB;

        impl TraceOptPass for PassA {
            fn name(&self) -> &'static str { "a" }
            fn priority(&self) -> u32 { 100 } // runs second
            fn run(&self, body: &mut Vec<TraceOp>) -> TraceOptStats {
                let order = ORDER.fetch_add(1, Ordering::SeqCst);
                assert_eq!(order, 1, "PassA should run second (priority 100)");
                let _ = body;
                TraceOptStats::default()
            }
        }

        impl TraceOptPass for PassB {
            fn name(&self) -> &'static str { "b" }
            fn priority(&self) -> u32 { 50 } // runs first
            fn run(&self, body: &mut Vec<TraceOp>) -> TraceOptStats {
                let order = ORDER.fetch_add(1, Ordering::SeqCst);
                assert_eq!(order, 0, "PassB should run first (priority 50)");
                let _ = body;
                TraceOptStats::default()
            }
        }

        let mut pipeline = TracePassPipeline::new();
        pipeline.register(Box::new(PassA)); // registered first, but higher priority
        pipeline.register(Box::new(PassB)); // registered second, but lower priority = runs first

        let mut body = vec![TraceOp::Const(1.0)];
        ORDER.store(0, Ordering::SeqCst);

        // Act: pipeline sorts by priority, so PassB (50) runs before PassA (100)
        pipeline.optimize(&mut body);

        // Assert: both passes ran (ORDER == 2), verified in assert above inside each pass
        assert_eq!(ORDER.load(Ordering::SeqCst), 2);
    }

    // @trace TEST-TOPT-34 [req:REQ-VR] [level:unit]
    #[test]
    fn test_stats_add_assign_accumulation() {
        // Arrange: manually construct stats and verify AddAssign
        let mut a = TraceOptStats {
            ops_removed: 2,
            ops_folded: 3,
            ops_merged: 1,
        };
        let b = TraceOptStats {
            ops_removed: 5,
            ops_folded: 7,
            ops_merged: 4,
        };

        // Act
        a += b;

        // Assert: all fields summed
        assert_eq!(a.ops_removed, 7);
        assert_eq!(a.ops_folded, 10);
        assert_eq!(a.ops_merged, 5);
    }

    // @trace TEST-TOPT-35 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_add_zero_left_operand() {
        // Arrange: 0 + x -> x (left operand is zero, identity remap)
        let mut body = vec![
            TraceOp::Input(0),       // [0] x
            TraceOp::Const(0.0),     // [1] 0.0
            TraceOp::Add(ValueId(1), ValueId(0)),   // [2] 0+x -> identity remap to ValueId(0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: 1 fold (identity remap)
        assert_eq!(stats.ops_folded, 1);
    }

    // @trace TEST-TOPT-36 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_tanh_not_folded() {
        // Arrange: Tanh with constant input -- ConstantFolding does not handle Tanh
        let mut body = vec![
            TraceOp::Const(1.0),      // [0]
            TraceOp::Tanh(ValueId(0)), // [1] -- not foldable by current pass
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded
        assert_eq!(stats.ops_folded, 0);
        assert!(matches!(body[1], TraceOp::Tanh(ValueId(0))));
    }

    // @trace TEST-TOPT-37 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_sigmoid_not_folded() {
        // Arrange: Sigmoid with constant input -- ConstantFolding does not handle Sigmoid
        let mut body = vec![
            TraceOp::Const(0.5),         // [0]
            TraceOp::Sigmoid(ValueId(0)), // [1] -- not foldable by current pass
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded
        assert_eq!(stats.ops_folded, 0);
        assert!(matches!(body[1], TraceOp::Sigmoid(ValueId(0))));
    }

    // @trace TEST-TOPT-38 [req:REQ-VR] [level:unit]
    #[test]
    fn test_dce_cascading_dead_chain() {
        // Arrange: Neg's result (ValueId(3)) is never referenced by anyone.
        // Mul's result (ValueId(2)) is referenced by Neg, so Mul is kept alive
        // in the single-pass "is my result referenced?" analysis.
        // Only Neg is removed. Input(0)/Input(1) are kept alive because Mul references them.
        let mut body = vec![
            TraceOp::Input(0),                              // [0] -- referenced by Mul, kept alive
            TraceOp::Input(1),                              // [1] -- referenced by Mul, kept alive
            TraceOp::Mul(ValueId(0), ValueId(1)),           // [2] -- alive (Neg refs ValueId(2))
            TraceOp::Neg(ValueId(2)),                       // [3] -- dead (nobody refs ValueId(3))
            TraceOp::Const(7.0),                            // [4] -- last = output, always alive
        ];

        // Act
        let stats = DeadCodeElimination.run(&mut body);

        // Assert: only Neg[3] is dead. Mul[2] stays because Neg referenced it.
        // Input[0], Input[1] survive because Mul references them.
        assert_eq!(stats.ops_removed, 1);
        assert_eq!(body.len(), 4);
        // After compaction: [0]=Input(0), [1]=Input(1), [2]=Mul(0,1), [3]=Const(7.0)
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Mul(ValueId(0), ValueId(1))));
        assert!(matches!(body[3], TraceOp::Const(7.0)));
    }

    // @trace TEST-TOPT-39 [req:REQ-VR] [level:unit]
    #[test]
    fn test_cse_merges_multiple_duplicates() {
        // Arrange: three identical Add ops -- CSE should merge the 2nd and 3rd into the 1st.
        let mut body = vec![
            TraceOp::Input(0),                              // [0] x
            TraceOp::Input(1),                              // [1] y
            TraceOp::Add(ValueId(0), ValueId(1)),           // [2] x+y (original)
            TraceOp::Add(ValueId(0), ValueId(1)),           // [3] x+y (duplicate 1)
            TraceOp::Add(ValueId(0), ValueId(1)),           // [4] x+y (duplicate 2)
            TraceOp::Mul(ValueId(2), ValueId(3)),           // [5] -- consumer of original + dup1
            TraceOp::Mul(ValueId(4), ValueId(0)),           // [6] -- consumer of dup2
        ];

        // Act
        let stats = CommonSubexprElimination.run(&mut body);

        // Assert: 2 merges (ops at index 3 and 4)
        assert_eq!(stats.ops_merged, 2);
        // Both downstream consumers should now reference ValueId(2)
        assert!(matches!(body[5], TraceOp::Mul(ValueId(2), ValueId(2))));
        assert!(matches!(body[6], TraceOp::Mul(ValueId(2), ValueId(0))));
    }

    // @trace TEST-TOPT-40 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_conditional_branch_not_folded() {
        // Arrange: ConditionalBranch with all-constant inputs --
        // ConstantFolding does not handle ConditionalBranch.
        let mut body = vec![
            TraceOp::Const(1.0),      // [0] mask (non-zero)
            TraceOp::Const(5.0),      // [1] true_val
            TraceOp::Const(9.0),      // [2] false_val
            TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2)), // [3]
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded, ConditionalBranch preserved
        assert_eq!(stats.ops_folded, 0);
        assert!(matches!(
            body[3],
            TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2))
        ));
    }

    // @trace TEST-TOPT-41 [req:REQ-VR] [level:unit]
    #[test]
    fn test_pipeline_deep_chain_simplification() {
        // Arrange: (2.0 * 3.0) + (4.0 - 0.0) -> Const(6.0) + Const(4.0) -> Const(10.0)
        // Tests that the full pipeline folds multi-level expressions in one pass.
        let mut body = vec![
            TraceOp::Const(2.0),     // [0]
            TraceOp::Const(3.0),     // [1]
            TraceOp::Mul(ValueId(0), ValueId(1)),    // [2] -> Const(6.0)
            TraceOp::Const(4.0),     // [3]
            TraceOp::Const(0.0),     // [4]
            TraceOp::Sub(ValueId(3), ValueId(4)),    // [5] -> Const(4.0)
            TraceOp::Add(ValueId(2), ValueId(5)),    // [6] -> Const(10.0)
        ];

        // Act
        let pipeline = TracePassPipeline::with_defaults();
        let stats = pipeline.optimize(&mut body);

        // Assert: at least 3 folds (Mul, Sub, Add)
        assert!(stats.ops_folded >= 3, "expected >=3 folds, got {:?}", stats);
        // After folding and DCE, the last op should be Const(10.0)
        let last = body.last().expect("body should not be empty");
        assert!(matches!(last, TraceOp::Const(v) if (v - 10.0).abs() < 1e-10));
    }

    // @trace TEST-TOPT-42 [req:REQ-VR] [level:unit]
    #[test]
    fn test_dce_all_alive_no_removal() {
        // Arrange: every op is either referenced or is the last op
        let mut body = vec![
            TraceOp::Input(0),                              // [0] — referenced by Add
            TraceOp::Input(1),                              // [1] — referenced by Add
            TraceOp::Add(ValueId(0), ValueId(1)),           // [2] — referenced by Neg
            TraceOp::Neg(ValueId(2)),                       // [3] — last = output
        ];

        // Act
        let stats = DeadCodeElimination.run(&mut body);

        // Assert: nothing removed, body untouched
        assert_eq!(stats.ops_removed, 0);
        assert_eq!(body.len(), 4);
    }

    // @trace TEST-TOPT-43 [req:REQ-VR] [level:unit]
    #[test]
    fn test_dce_multiple_consecutive_dead_ops() {
        // Arrange: three dead ops in a row (Const 10, Const 20, Const 30)
        // with only the last Const(99.0) alive as the output.
        let mut body = vec![
            TraceOp::Input(0),                              // [0] — dead (nobody refs ValueId(0))
            TraceOp::Const(10.0),                           // [1] — dead
            TraceOp::Const(20.0),                           // [2] — dead
            TraceOp::Const(30.0),                           // [3] — dead
            TraceOp::Const(99.0),                           // [4] — last = output, alive
        ];

        // Act
        let stats = DeadCodeElimination.run(&mut body);

        // Assert: 4 dead ops removed, only Const(99.0) survives
        assert_eq!(stats.ops_removed, 4);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], TraceOp::Const(v) if v == 99.0));
    }

    // @trace TEST-TOPT-44 [req:REQ-VR] [level:unit]
    #[test]
    fn test_cse_mixed_pure_and_impure_only_merges_pure() {
        // Arrange: two identical pure Neg ops and two identical impure VecStoreIndexed ops.
        // CSE should merge the Neg pair but leave both VecStoreIndexed untouched.
        let mut body = vec![
            TraceOp::Input(0),                              // [0] x
            TraceOp::Input(1),                              // [1] offset
            TraceOp::Input(2),                              // [2] value
            TraceOp::Neg(ValueId(0)),                       // [3] -x (original pure)
            TraceOp::Neg(ValueId(0)),                       // [4] -x (duplicate pure)
            TraceOp::VecStoreIndexed {                      // [5] impure
                base: ValueId(0),
                offset: ValueId(1),
                value: ValueId(2),
            },
            TraceOp::VecStoreIndexed {                      // [6] impure (identical but NOT merged)
                base: ValueId(0),
                offset: ValueId(1),
                value: ValueId(2),
            },
        ];

        // Act
        let stats = CommonSubexprElimination.run(&mut body);

        // Assert: exactly 1 merge (the two Neg ops), VecStoreIndexed pairs untouched
        assert_eq!(stats.ops_merged, 1);
        assert_eq!(body.len(), 7);
        // Verify the impure ops are still structurally present
        let store_count = body.iter().filter(|op| matches!(op, TraceOp::VecStoreIndexed { .. })).count();
        assert_eq!(store_count, 2);
    }

    // @trace TEST-TOPT-45 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_nested_unary_chain() {
        // Arrange: Neg(Abs(-5.0)) — two unary ops on constants should chain-fold.
        // Abs(-5.0) → Const(5.0), then Neg(5.0) → Const(-5.0).
        let mut body = vec![
            TraceOp::Const(-5.0),                           // [0]
            TraceOp::Abs(ValueId(0)),                       // [1] → Const(5.0)
            TraceOp::Neg(ValueId(1)),                       // [2] → Const(-5.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: both unary ops folded
        assert_eq!(stats.ops_folded, 2);
        assert!(matches!(body[1], TraceOp::Const(v) if v == 5.0));
        assert!(matches!(body[2], TraceOp::Const(v) if v == -5.0));
    }

    // @trace TEST-TOPT-46 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_mul_both_constants() {
        // Arrange: 4.0 * 0.25 → 1.0 (full binary fold, neither identity nor zero)
        let mut body = vec![
            TraceOp::Const(4.0),                            // [0]
            TraceOp::Const(0.25),                           // [1]
            TraceOp::Mul(ValueId(0), ValueId(1)),           // [2] → Const(1.0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: folded to Const(1.0)
        assert_eq!(stats.ops_folded, 1);
        assert!(matches!(body[2], TraceOp::Const(v) if (v - 1.0).abs() < 1e-10));
    }

    // @trace TEST-TOPT-47 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_combined_add_mul_identity() {
        // Arrange: (x + 0) * 1 should apply add-zero identity then mul-one identity.
        // Add(x, 0.0) → remap to x, then Mul(x, 1.0) → remap to x.
        let mut body = vec![
            TraceOp::Input(0),                              // [0] x
            TraceOp::Const(0.0),                            // [1] 0.0
            TraceOp::Add(ValueId(0), ValueId(1)),           // [2] → identity remap to ValueId(0)
            TraceOp::Const(1.0),                            // [3] 1.0
            TraceOp::Mul(ValueId(2), ValueId(3)),           // [4] → identity remap to ValueId(2) → ValueId(0)
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: 2 identity folds (Add-zero, Mul-one)
        assert_eq!(stats.ops_folded, 2);
    }

    // @trace TEST-TOPT-48 [req:REQ-VR] [level:unit]
    #[test]
    fn test_pipeline_second_run_stabilizes() {
        // Arrange: body with foldable Add. After first pipeline run, the Add is folded to Const(10.0),
        // and the original Const(3.0)/Const(7.0) become dead (no one references them).
        // Second pipeline run removes the dead constants. Third run is a complete no-op.
        let mut body = vec![
            TraceOp::Const(3.0),                            // [0]
            TraceOp::Const(7.0),                            // [1]
            TraceOp::Add(ValueId(0), ValueId(1)),           // [2] → foldable
        ];

        // Act: first run folds Add → Const(10.0)
        let pipeline = TracePassPipeline::with_defaults();
        let stats1 = pipeline.optimize(&mut body);
        assert!(stats1.ops_folded >= 1, "first run should fold Add, got {:?}", stats1);

        // Act: second run removes dead Const(3.0)/Const(7.0) left over from folding
        let stats2 = pipeline.optimize(&mut body);
        assert!(stats2.ops_removed >= 2, "second run should remove dead constants, got {:?}", stats2);

        // Act: third run — truly stable, no further changes
        let stats3 = pipeline.optimize(&mut body);
        assert_eq!(stats3.ops_removed + stats3.ops_folded + stats3.ops_merged, 0,
            "third run should be a no-op, got {:?}", stats3);
        // Body is just [Const(10.0)]
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], TraceOp::Const(v) if (v - 10.0).abs() < 1e-10));
    }

    // @trace TEST-TOPT-49 [req:REQ-VR] [level:unit]
    #[test]
    fn test_compute_op_hash_different_ops_differ() {
        // Arrange: two ops with different discriminants (Neg vs Abs) on same input
        let op_a = TraceOp::Neg(ValueId(0));
        let op_b = TraceOp::Abs(ValueId(0));

        // Act
        let hash_a = compute_op_hash(&op_a);
        let hash_b = compute_op_hash(&op_b);

        // Assert: hashes must differ because discriminants differ
        assert_ne!(hash_a, hash_b, "different op variants must produce different hashes");
    }

    // @trace TEST-TOPT-50 [req:REQ-VR] [level:unit]
    #[test]
    fn test_structural_equality_different_discriminants() {
        // Arrange: two ops with different discriminants
        let op_a = TraceOp::Neg(ValueId(0));
        let op_b = TraceOp::Exp(ValueId(0));
        let remap = vec![ValueId(u32::MAX); 2];

        // Act
        let equal = ops_structurally_equal(&op_a, &op_b, &remap);

        // Assert: must be false — different enum variants
        assert!(!equal);
    }

    // @trace TEST-TOPT-51 [req:REQ-VR] [level:unit]
    #[test]
    fn test_const_fold_input_only_body() {
        // Arrange: body with only Input ops and a final pure op on inputs —
        // nothing is foldable because no constants feed into the arithmetic.
        let mut body = vec![
            TraceOp::Input(0),                              // [0] x
            TraceOp::Input(1),                              // [1] y
            TraceOp::Add(ValueId(0), ValueId(1)),           // [2] x+y — not foldable
            TraceOp::Neg(ValueId(2)),                       // [3] -(x+y) — not foldable
        ];

        // Act
        let stats = ConstantFolding.run(&mut body);

        // Assert: nothing folded
        assert_eq!(stats.ops_folded, 0);
        assert_eq!(body.len(), 4);
        assert!(matches!(body[2], TraceOp::Add(ValueId(0), ValueId(1))));
        assert!(matches!(body[3], TraceOp::Neg(ValueId(2))));
    }
}
