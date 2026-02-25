//! Buffer lifetime analysis and interval graph coloring.
//!
//! Analyzes tensor lifetimes across the execution schedule and assigns
//! buffer offsets to minimize total scratchpad memory via greedy interval
//! graph coloring.

use std::collections::{HashMap, HashSet};
use crate::compiler::graph::{CompilerGraph, TensorId, OpId, OpKind};
use crate::compiler::fusion::{FusionPlan, FusionMode};

/// A tensor's lifetime interval: [first_use, last_use] in schedule order.
#[derive(Debug, Clone, Copy)]
pub struct Lifetime {
    /// Which tensor this lifetime belongs to.
    pub tensor_id: TensorId,
    /// Schedule step of first use (read or write).
    pub first_use: usize,
    /// Schedule step of last use (read or write).
    pub last_use: usize,
    /// Buffer size in bytes.
    pub size_bytes: usize,
}

/// Buffer assignment: tensor mapped to (offset, size) in the scratchpad.
#[derive(Debug, Clone, Copy)]
pub struct BufferSlot {
    /// Which tensor occupies this slot.
    pub tensor_id: TensorId,
    /// Byte offset within the scratchpad.
    pub offset: usize,
    /// Size of this buffer in bytes.
    pub size_bytes: usize,
}

/// Result of buffer allocation.
#[derive(Debug, Clone)]
pub struct BufferAllocation {
    /// Per-tensor buffer assignments.
    pub slots: Vec<BufferSlot>,
    /// Total scratchpad bytes required.
    pub total_bytes: usize,
    /// Number of intermediate tensors allocated.
    pub num_tensors: usize,
    /// Bytes saved compared to naive allocation (sum of all tensor sizes).
    pub bytes_saved: usize,
}

/// Analyze tensor lifetimes from the fusion plan's execution order.
///
/// Only intermediate tensors are included — graph inputs and outputs are
/// externally managed and excluded from scratchpad allocation.
pub fn analyze_lifetimes(graph: &CompilerGraph, plan: &FusionPlan) -> Vec<Lifetime> {
    // Build schedule order: flatten fusion groups
    let schedule: Vec<OpId> = plan
        .groups
        .iter()
        .flat_map(|g| g.ops.iter().copied())
        .collect();

    let mut first_use: HashMap<TensorId, usize> = HashMap::new();
    let mut last_use: HashMap<TensorId, usize> = HashMap::new();

    for (step, &op_id) in schedule.iter().enumerate() {
        if let Some(op) = graph.op(op_id) {
            for &tid in op.inputs.iter().chain(op.outputs.iter()) {
                first_use.entry(tid).or_insert(step);
                last_use.insert(tid, step);
            }
        }
    }

    // Exclude graph inputs/outputs (externally managed)
    let graph_io: HashSet<TensorId> = graph
        .inputs
        .iter()
        .chain(graph.outputs.iter())
        .copied()
        .collect();

    let mut lifetimes = Vec::new();
    for tensor in &graph.tensors {
        if graph_io.contains(&tensor.id) {
            continue;
        }
        if let (Some(&first), Some(&last)) =
            (first_use.get(&tensor.id), last_use.get(&tensor.id))
        {
            let size_bytes = tensor.shape.iter().product::<usize>() * tensor.dtype.size_bytes();
            if size_bytes > 0 {
                lifetimes.push(Lifetime {
                    tensor_id: tensor.id,
                    first_use: first,
                    last_use: last,
                    size_bytes,
                });
            }
        }
    }

    lifetimes
}

/// Greedy interval graph coloring: assign buffer offsets to minimize total memory.
///
/// Algorithm: sort by first_use (ties broken by larger size first for better
/// packing), then greedily assign each tensor to the lowest offset where it
/// doesn't overlap with any currently live tensor.
pub fn allocate_buffers(lifetimes: &[Lifetime]) -> BufferAllocation {
    if lifetimes.is_empty() {
        return BufferAllocation {
            slots: Vec::new(),
            total_bytes: 0,
            num_tensors: 0,
            bytes_saved: 0,
        };
    }

    // Sort by first_use, ties broken by larger size first
    let mut sorted: Vec<&Lifetime> = lifetimes.iter().collect();
    sorted.sort_by_key(|l| (l.first_use, std::cmp::Reverse(l.size_bytes)));

    // Active allocations: (offset, end_offset, last_use)
    let mut active: Vec<(usize, usize, usize)> = Vec::new();
    let mut slots = Vec::new();
    let mut total_bytes = 0usize;

    for lt in &sorted {
        let offset = find_offset(lt, &active);
        let end = offset + lt.size_bytes;
        total_bytes = total_bytes.max(end);

        active.push((offset, end, lt.last_use));
        slots.push(BufferSlot {
            tensor_id: lt.tensor_id,
            offset,
            size_bytes: lt.size_bytes,
        });
    }

    let naive_total: usize = lifetimes.iter().map(|l| l.size_bytes).sum();

    BufferAllocation {
        num_tensors: slots.len(),
        slots,
        total_bytes,
        bytes_saved: naive_total.saturating_sub(total_bytes),
    }
}

/// Find the lowest offset where a tensor can be placed without overlapping
/// any currently live allocation.
fn find_offset(lt: &Lifetime, active: &[(usize, usize, usize)]) -> usize {
    // Collect all live allocations that overlap with this tensor's lifetime
    let mut live_ranges: Vec<(usize, usize)> = active
        .iter()
        .filter(|(_, _, last)| *last >= lt.first_use)
        .map(|(start, end, _)| (*start, *end))
        .collect();

    live_ranges.sort_by_key(|(start, _)| *start);

    // Find first gap that fits
    let mut candidate = 0usize;
    for &(start, end) in &live_ranges {
        if candidate + lt.size_bytes <= start {
            return candidate; // fits in the gap
        }
        candidate = candidate.max(end);
    }

    // Align to 64 bytes for cache line alignment
    (candidate + 63) & !63
}

/// Per-group scratch buffer requirement for TileLevelFusion.
#[derive(Debug, Clone)]
pub struct GroupScratch {
    /// Fusion group ID.
    pub group_id: usize,
    /// Scratch bytes needed (tile_rows × K × sizeof(f32)).
    pub scratch_bytes: usize,
}

/// Compute scratch buffer requirements for TileLevelFusion groups.
///
/// Each TileLevelFusion group needs a scratch buffer to hold the tiled norm
/// output (tile_rows × K × element_size). This is separate from the
/// intermediate tensor allocation because it's a temporary within the
/// microkernel's MC loop, not a full tensor.
pub fn compute_scratch_requirements(
    plan: &FusionPlan,
    graph: &CompilerGraph,
) -> Vec<GroupScratch> {
    plan.groups
        .iter()
        .filter_map(|group| {
            if let FusionMode::TileLevelFusion { tile_rows, .. } = group.mode {
                // Find the GEMM op to get K dimension
                let k = group.ops.iter().find_map(|&oid| {
                    graph.op(oid).and_then(|o| match &o.kind {
                        OpKind::Gemm { k, .. }
                        | OpKind::GemmBias { k, .. }
                        | OpKind::QuantGemm { k, .. } => Some(*k),
                        _ => None,
                    })
                }).unwrap_or(0);

                let elem_size = group.ops.iter().find_map(|&oid| {
                    graph.op(oid).and_then(|o| {
                        o.outputs.first().and_then(|&tid| {
                            graph.tensor(tid).map(|t| t.dtype.size_bytes())
                        })
                    })
                }).unwrap_or(4);

                let scratch_bytes = tile_rows * k * elem_size;
                if scratch_bytes > 0 {
                    Some(GroupScratch {
                        group_id: group.id,
                        scratch_bytes,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{CompilerGraph, TensorId};
    use crate::compiler::ir::LayerIR;
    use crate::compiler::fusion;
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::ModelConfig;

    #[test]
    fn test_non_overlapping_reuse() {
        // Two tensors with non-overlapping lifetimes should share buffer space
        let lifetimes = vec![
            Lifetime {
                tensor_id: TensorId(0),
                first_use: 0,
                last_use: 1,
                size_bytes: 1024,
            },
            Lifetime {
                tensor_id: TensorId(1),
                first_use: 2,
                last_use: 3,
                size_bytes: 1024,
            },
        ];

        let alloc = allocate_buffers(&lifetimes);

        assert_eq!(alloc.num_tensors, 2);
        // Non-overlapping: second tensor can reuse the first's space
        // Total should be <= 1024 + alignment overhead, not 2048
        assert!(
            alloc.total_bytes < 2048,
            "Non-overlapping tensors should share buffer, total={} (expected < 2048)",
            alloc.total_bytes
        );
        assert!(alloc.bytes_saved > 0, "Should save bytes via reuse");
    }

    #[test]
    fn test_overlapping_no_reuse() {
        // Two tensors with overlapping lifetimes cannot share buffer space
        let lifetimes = vec![
            Lifetime {
                tensor_id: TensorId(0),
                first_use: 0,
                last_use: 3,
                size_bytes: 1024,
            },
            Lifetime {
                tensor_id: TensorId(1),
                first_use: 1,
                last_use: 2,
                size_bytes: 1024,
            },
        ];

        let alloc = allocate_buffers(&lifetimes);

        assert_eq!(alloc.num_tensors, 2);
        // Overlapping: both must be live simultaneously
        assert!(
            alloc.total_bytes >= 2048,
            "Overlapping tensors cannot share buffer, total={} (expected >= 2048)",
            alloc.total_bytes
        );
    }

    #[test]
    fn test_llama_buffer_allocation() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = ScalarOpRegistry::with_defaults();
        let plan = fusion::fuse_with_dag(&graph, &registry, &profile);

        let lifetimes = analyze_lifetimes(&graph, &plan);
        assert!(!lifetimes.is_empty(), "LLaMA graph should have intermediate tensors");

        let alloc = allocate_buffers(&lifetimes);

        assert!(alloc.num_tensors > 0);
        assert!(alloc.total_bytes > 0);
        assert!(
            alloc.bytes_saved > 0,
            "LLaMA graph should benefit from buffer reuse"
        );

        let naive: usize = lifetimes.iter().map(|l| l.size_bytes).sum();
        eprintln!(
            "LLaMA buffer alloc: {} tensors, naive={} bytes, optimized={} bytes, saved={} bytes ({:.0}%)",
            alloc.num_tensors,
            naive,
            alloc.total_bytes,
            alloc.bytes_saved,
            alloc.bytes_saved as f64 / naive as f64 * 100.0,
        );
    }

    #[test]
    fn test_empty_lifetimes() {
        let alloc = allocate_buffers(&[]);
        assert_eq!(alloc.total_bytes, 0);
        assert_eq!(alloc.num_tensors, 0);
        assert_eq!(alloc.bytes_saved, 0);
        assert!(alloc.slots.is_empty());
    }
}
