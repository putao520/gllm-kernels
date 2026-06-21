//! Compiler diagnostics dump — REQ-DUMP-001 / REQ-DUMP-002 / REQ-DUMP-003
//!
//! Provides four-level intermediate product dump for the compilation pipeline:
//! 1. CompilerGraph DOT/JSON (REQ-DUMP-001)
//! 2. TraceOp SSA (REQ-DUMP-002)
//! 3. VmInstr sequence (REQ-DUMP-003)
//!
//! All JSON output is hand-rolled (no serde dependency), following the
//! ProfileReport.to_json() convention in this crate.

use crate::compiler::codegen::vm::instr::{VRegId, VmInstr};
use crate::compiler::codegen::vm::isa_profile::PhysReg;
use crate::compiler::codegen::vm::reg_alloc::RegAllocation;
use crate::compiler::codegen::vm::topology::GraphTopologyAnalysis;
use crate::compiler::graph::{CompilerGraph, Op};
use crate::compiler::trace::{ComputePattern, OpTrace, TraceOp, ValueId};

// ── REQ-DUMP-001: CompilerGraph DOT/JSON ────────────────────────────

// @trace REQ-DUMP-001 [entity:CompilerGraph] dump_dot

/// Dump the CompilerGraph in Graphviz DOT format.
///
/// Nodes = ops (annotated with Op variant name, dtype, shape).
/// Edges = tensor connections (annotated with slot name).
/// ComputePattern annotation per op when `op_traces` is provided.
/// GraphTopologyAnalysis result when `topology` is provided.
pub fn graph_to_dot(
    graph: &CompilerGraph,
    op_traces: &[Option<OpTrace>],
    topology: Option<&GraphTopologyAnalysis>,
) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("digraph CompilerGraph {\n");
    s.push_str("  rankdir=TB;\n");
    s.push_str("  node [shape=box, style=filled, fillcolor=\"#f0f0f0\"];\n");
    s.push_str("  edge [fontsize=10];\n\n");

    // Topology annotation as graph-level comment
    if let Some(topo) = topology {
        s.push_str(&format!("  // topology: loop_topology={:?}\n", topo.loop_topology));
        s.push_str(&format!("  // topology: kv_cache_source={:?}\n", topo.kv_cache_source));
        s.push_str(&format!("  // topology: sg_ops={:?}\n", topo.sg_ops));
        s.push_str(&format!("  // topology: weight_source={:?}\n", topo.weight_source));
        s.push_str(&format!("  // topology: has_qk_norm={}\n", topo.has_qk_norm));
        if let Some(vs) = topo.vocab_size {
            s.push_str(&format!("  // topology: vocab_size={}\n", vs));
        }
        s.push('\n');
    }

    // Op nodes
    for (idx, op) in graph.ops.iter().enumerate() {
        let op_variant = op_variant_name(&op.op);
        let mut label = format!("op{}: {}", op.id.0, op.label);
        label.push_str(&format!("\\n{}", op_variant));

        // ComputePattern annotation
        if idx < op_traces.len() {
            if let Some(ref trace) = op_traces[idx] {
                label.push_str(&format!("\\n[{}]", compute_pattern_brief(&trace.pattern)));
            }
        }

        // Guard annotation
        if let Some(guard_str) = guard_label(&op.guard) {
            label.push_str(&format!("\\nguard={}", guard_str));
        }

        // Color by category
        let color = op_category_color(&op.op);
        s.push_str(&format!("  op{} [label=\"{}\", fillcolor=\"{}\"];\n", op.id.0, label, color));
    }

    s.push('\n');

    // Tensor edges
    for op in &graph.ops {
        for (slot_idx, &input_tid) in op.inputs.iter().enumerate() {
            if let Some(tensor) = graph.tensor(input_tid) {
                let producer_label = if let Some(producer_oid) = tensor.producer {
                    format!("op{}", producer_oid.0)
                } else {
                    "input".to_string()
                };
                let edge_label = format!("in{}: {} [{:?}]", slot_idx, tensor.name, tensor.dtype);
                s.push_str(&format!(
                    "  {} -> op{} [label=\"{}\"];\n",
                    producer_label, op.id.0, edge_label
                ));
            }
        }
    }

    s.push_str("}\n");
    s
}

// @trace REQ-DUMP-001 [entity:CompilerGraph] dump_json

/// Dump the CompilerGraph in JSON format (hand-rolled, no serde).
///
/// Full graph serialization with ops, tensors, ComputePattern annotations,
/// and GraphTopologyAnalysis result.
pub fn graph_to_json(
    graph: &CompilerGraph,
    op_traces: &[Option<OpTrace>],
    topology: Option<&GraphTopologyAnalysis>,
) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str("{\n");

    // Graph metadata
    s.push_str(&format!("  \"num_ops\": {},\n", graph.ops.len()));
    s.push_str(&format!("  \"num_tensors\": {},\n", graph.tensors.len()));

    // Topology
    if let Some(topo) = topology {
        s.push_str("  \"topology\": {\n");
        s.push_str(&format!("    \"loop_topology\": \"{}\",\n", format!("{:?}", topo.loop_topology)));
        s.push_str(&format!("    \"outer_loop_bound\": \"{}\",\n", format!("{:?}", topo.outer_loop_bound)));
        s.push_str(&format!("    \"seq_len_source\": \"{}\",\n", format!("{:?}", topo.seq_len_source)));
        s.push_str(&format!("    \"kv_cache_source\": \"{}\",\n", format!("{:?}", topo.kv_cache_source)));
        s.push_str(&format!("    \"sg_ops\": \"{}\",\n", format!("{:?}", topo.sg_ops)));
        s.push_str(&format!("    \"weight_source\": \"{}\",\n", format!("{:?}", topo.weight_source)));
        s.push_str(&format!("    \"has_qk_norm\": {},\n", topo.has_qk_norm));
        if let Some(vs) = topo.vocab_size {
            s.push_str(&format!("    \"vocab_size\": {},\n", vs));
        }
        if let Some(nl) = topo.layer_num_layers {
            s.push_str(&format!("    \"layer_num_layers\": {},\n", nl));
        }
        s.push_str("    \"placeholder\": null\n");
        s.push_str("  },\n");
    }

    // Ops
    s.push_str("  \"ops\": [\n");
    for (i, op) in graph.ops.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"id\": {},\n", op.id.0));
        s.push_str(&format!("      \"label\": {},\n", json_str(&op.label)));
        s.push_str(&format!("      \"op\": {},\n", json_str(&op_variant_name(&op.op))));
        s.push_str(&format!("      \"category\": {},\n", json_str(op.op.category())));
        s.push_str(&format!("      \"guard\": {},\n", json_str(&guard_label(&op.guard).unwrap_or_else(|| "Always".to_string()))));

        // Inputs
        s.push_str("      \"inputs\": [");
        for (j, &tid) in op.inputs.iter().enumerate() {
            if j > 0 { s.push_str(", "); }
            s.push_str(&format!("{}", tid.0));
        }
        s.push_str("],\n");

        // Outputs
        s.push_str("      \"outputs\": [");
        for (j, &tid) in op.outputs.iter().enumerate() {
            if j > 0 { s.push_str(", "); }
            s.push_str(&format!("{}", tid.0));
        }
        s.push_str("],\n");

        // ComputePattern annotation
        if i < op_traces.len() {
            if let Some(ref trace) = op_traces[i] {
                s.push_str(&format!("      \"compute_pattern\": {},\n", json_str(&compute_pattern_brief(&trace.pattern))));
            } else {
                s.push_str("      \"compute_pattern\": null,\n");
            }
        } else {
            s.push_str("      \"compute_pattern\": null,\n");
        }

        // Op details from Debug format (abbreviated)
        s.push_str(&format!("      \"op_detail\": {}\n", json_str(&format!("{:?}", op.op))));

        s.push_str(if i + 1 < graph.ops.len() { "    },\n" } else { "    }\n" });
    }
    s.push_str("  ],\n");

    // Tensors
    s.push_str("  \"tensors\": [\n");
    for (i, tensor) in graph.tensors.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"id\": {},\n", tensor.id.0));
        s.push_str(&format!("      \"name\": {},\n", json_str(&tensor.name)));
        s.push_str(&format!("      \"dtype\": {},\n", json_str(&format!("{:?}", tensor.dtype))));
        s.push_str(&format!("      \"shape\": {},\n", json_str(&symdim_vec_to_string(&tensor.shape))));

        if let Some(pid) = tensor.producer {
            s.push_str(&format!("      \"producer\": {},\n", pid.0));
        } else {
            s.push_str("      \"producer\": null,\n");
        }

        s.push_str("      \"consumers\": [");
        for (j, &cid) in tensor.consumers.iter().enumerate() {
            if j > 0 { s.push_str(", "); }
            s.push_str(&format!("{}", cid.0));
        }
        s.push_str("]\n");

        s.push_str(if i + 1 < graph.tensors.len() { "    },\n" } else { "    }\n" });
    }
    s.push_str("  ],\n");

    // Graph inputs/outputs
    s.push_str("  \"graph_inputs\": [");
    for (j, &tid) in graph.inputs.iter().enumerate() {
        if j > 0 { s.push_str(", "); }
        s.push_str(&format!("{}", tid.0));
    }
    s.push_str("],\n");

    s.push_str("  \"graph_outputs\": [");
    for (j, &tid) in graph.outputs.iter().enumerate() {
        if j > 0 { s.push_str(", "); }
        s.push_str(&format!("{}", tid.0));
    }
    s.push_str("]\n");

    s.push_str("}\n");
    s
}

// ── REQ-DUMP-002: TraceOp SSA dump ──────────────────────────────────

// @trace REQ-DUMP-002 [entity:TraceOp] dump_ssa

/// TraceOp SSA dump context — carries per-op trace information with
/// OpKind mapping and dtype/shape annotations.
pub struct TraceDumpContext<'a> {
    /// The graph being dumped (for OpKind and tensor metadata lookup).
    pub graph: &'a CompilerGraph,
    /// Per-op trace results (indexed by graph op index).
    pub op_traces: &'a [Option<OpTrace>],
}

/// Dump all TraceOp SSA forms in JSON format.
///
/// Each op's trace includes:
/// - Input/output SSA registers (ValueId)
/// - ComputePattern annotation
/// - dtype and shape information
/// - Mapping back to source OpKind
pub fn trace_ssa_to_json(ctx: &TraceDumpContext) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str("{\n");
    s.push_str(&format!("  \"num_ops\": {},\n", ctx.graph.ops.len()));
    s.push_str("  \"ops\": [\n");

    for (i, op) in ctx.graph.ops.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"op_id\": {},\n", op.id.0));
        s.push_str(&format!("      \"label\": {},\n", json_str(&op.label)));
        s.push_str(&format!("      \"source_op\": {},\n", json_str(&op_variant_name(&op.op))));

        // Input tensor dtype/shape
        s.push_str("      \"input_tensors\": [\n");
        for (j, &tid) in op.inputs.iter().enumerate() {
            if let Some(tensor) = ctx.graph.tensor(tid) {
                s.push_str("        {\n");
                s.push_str(&format!("          \"name\": {},\n", json_str(&tensor.name)));
                s.push_str(&format!("          \"dtype\": {},\n", json_str(&format!("{:?}", tensor.dtype))));
                s.push_str(&format!("          \"shape\": {}\n", json_str(&symdim_vec_to_string(&tensor.shape))));
                s.push_str(if j + 1 < op.inputs.len() { "        },\n" } else { "        }\n" });
            }
        }
        s.push_str("      ],\n");

        // Output tensor dtype/shape
        s.push_str("      \"output_tensors\": [\n");
        for (j, &tid) in op.outputs.iter().enumerate() {
            if let Some(tensor) = ctx.graph.tensor(tid) {
                s.push_str("        {\n");
                s.push_str(&format!("          \"name\": {},\n", json_str(&tensor.name)));
                s.push_str(&format!("          \"dtype\": {},\n", json_str(&format!("{:?}", tensor.dtype))));
                s.push_str(&format!("          \"shape\": {}\n", json_str(&symdim_vec_to_string(&tensor.shape))));
                s.push_str(if j + 1 < op.outputs.len() { "        },\n" } else { "        }\n" });
            }
        }
        s.push_str("      ],\n");

        // TraceOp SSA
        if i < ctx.op_traces.len() {
            if let Some(ref trace) = ctx.op_traces[i] {
                s.push_str(&format!("      \"compute_pattern\": {},\n", json_str(&compute_pattern_brief(&trace.pattern))));
                s.push_str("      \"trace_ops\": [\n");
                let body_ops = trace_ops_for_pattern(&trace.pattern);
                for (j, top) in body_ops.iter().enumerate() {
                    s.push_str(&format!("        {}\n", json_str(&format!("{:?}", top))));
                    if j + 1 < body_ops.len() {
                        s.push(',');
                    }
                }
                s.push_str("      ],\n");

                // SSA register info — extract ValueId references from TraceOps
                s.push_str("      \"ssa_registers\": [\n");
                let value_ids = extract_value_ids(body_ops);
                for (j, vid) in value_ids.iter().enumerate() {
                    s.push_str(&format!("        \"v{}\"", vid.0));
                    if j + 1 < value_ids.len() {
                        s.push_str(",\n");
                    } else {
                        s.push('\n');
                    }
                }
                s.push_str("      ]\n");
            } else {
                s.push_str("      \"compute_pattern\": null,\n");
                s.push_str("      \"trace_ops\": [],\n");
                s.push_str("      \"ssa_registers\": []\n");
            }
        } else {
            s.push_str("      \"compute_pattern\": null,\n");
            s.push_str("      \"trace_ops\": [],\n");
            s.push_str("      \"ssa_registers\": []\n");
        }

        s.push_str(if i + 1 < ctx.graph.ops.len() { "    },\n" } else { "    }\n" });
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

/// Dump all TraceOp SSA forms in human-readable text format.
pub fn trace_ssa_to_text(ctx: &TraceDumpContext) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str(&format!("TraceOp SSA Dump: {} ops\n", ctx.graph.ops.len()));
    s.push_str(&"=".repeat(60));
    s.push('\n');

    for (i, op) in ctx.graph.ops.iter().enumerate() {
        s.push_str(&format!("\n[{}] {} ({})\n", op.id.0, op.label, op_variant_name(&op.op)));

        // Input tensors
        for (j, &tid) in op.inputs.iter().enumerate() {
            if let Some(tensor) = ctx.graph.tensor(tid) {
                s.push_str(&format!("  in[{}]: {} [{:?}] shape={}\n",
                    j, tensor.name, tensor.dtype, symdim_vec_to_string(&tensor.shape)));
            }
        }

        // Output tensors
        for (j, &tid) in op.outputs.iter().enumerate() {
            if let Some(tensor) = ctx.graph.tensor(tid) {
                s.push_str(&format!("  out[{}]: {} [{:?}] shape={}\n",
                    j, tensor.name, tensor.dtype, symdim_vec_to_string(&tensor.shape)));
            }
        }

        // Trace
        if i < ctx.op_traces.len() {
            if let Some(ref trace) = ctx.op_traces[i] {
                s.push_str(&format!("  ComputePattern: {}\n", compute_pattern_brief(&trace.pattern)));
                let body_ops = trace_ops_for_pattern(&trace.pattern);
                if !body_ops.is_empty() {
                    s.push_str("  TraceOps:\n");
                    for top in body_ops {
                        s.push_str(&format!("    {:?}\n", top));
                    }
                } else {
                    s.push_str("  TraceOps: (none — pattern has no body)\n");
                }
            } else {
                s.push_str("  (no trace)\n");
            }
        }
    }

    s
}

// ── REQ-DUMP-003: VmInstr sequence dump ─────────────────────────────

// @trace REQ-DUMP-003 [entity:VmInstr] dump_instr

/// VmInstr sequence dump context — carries the program, register allocation,
/// and optional TraceOp mapping information.
pub struct VmInstrDumpContext<'a> {
    /// The VmInstr sequence.
    pub instrs: &'a [VmInstr],
    /// Register allocation results (if available).
    pub reg_alloc: Option<&'a RegAllocation>,
    /// ABI parameter binding: VRegId -> human-readable parameter name.
    pub abi_bindings: &'a [(VRegId, &'a str)],
    /// TraceOp mapping: VmInstr index -> source TraceOp description.
    pub trace_mapping: &'a [(usize, String)],
}

/// Dump the VmInstr sequence in JSON format.
///
/// Includes:
/// - VmInstr sequence with emit_loop boundaries
/// - Register allocation results
/// - ABI parameter binding information
/// - TraceOp mapping
pub fn vminstr_to_json(ctx: &VmInstrDumpContext) -> String {
    let mut s = String::with_capacity(16384);
    s.push_str("{\n");
    s.push_str(&format!("  \"num_instrs\": {},\n", ctx.instrs.len()));

    // ABI bindings
    s.push_str("  \"abi_bindings\": [\n");
    for (i, (vreg, name)) in ctx.abi_bindings.iter().enumerate() {
        s.push_str(&format!("    {{\"vreg\": {}, \"name\": {}}}", vreg.0, json_str(name)));
        if i + 1 < ctx.abi_bindings.len() { s.push_str(",\n"); } else { s.push('\n'); }
    }
    s.push_str("  ],\n");

    // Register allocation
    if let Some(alloc) = ctx.reg_alloc {
        s.push_str("  \"reg_alloc\": {\n");
        s.push_str(&format!("    \"num_vregs\": {},\n", alloc.num_vregs()));
        s.push_str(&format!("    \"num_spills\": {},\n", alloc.spills.len()));
        s.push_str("    \"mapping\": [\n");
        let mapping: Vec<_> = alloc.mapping.iter().collect();
        for (i, (vreg, phys)) in mapping.iter().enumerate() {
            s.push_str(&format!("      {{\"vreg\": {}, \"phys\": {}}}", vreg.0, json_str(&format!("{:?}", phys))));
            if i + 1 < mapping.len() { s.push_str(",\n"); } else { s.push('\n'); }
        }
        s.push_str("    ],\n");
        s.push_str("    \"spills\": [\n");
        for (i, slot) in alloc.spills.iter().enumerate() {
            s.push_str(&format!("      {{\"vreg\": {}, \"offset\": {}, \"size\": {}}}",
                slot.vreg.0, slot.offset, slot.size));
            if i + 1 < alloc.spills.len() { s.push_str(",\n"); } else { s.push('\n'); }
        }
        s.push_str("    ]\n");
        s.push_str("  },\n");
    }

    // Trace mapping
    if !ctx.trace_mapping.is_empty() {
        s.push_str("  \"trace_mapping\": [\n");
        for (i, (instr_idx, desc)) in ctx.trace_mapping.iter().enumerate() {
            s.push_str(&format!("    {{\"instr_idx\": {}, \"source\": {}}}", instr_idx, json_str(desc)));
            if i + 1 < ctx.trace_mapping.len() { s.push_str(",\n"); } else { s.push('\n'); }
        }
        s.push_str("  ],\n");
    }

    // VmInstr sequence
    s.push_str("  \"instrs\": [\n");
    let mut loop_depth = 0u32;
    for (i, instr) in ctx.instrs.iter().enumerate() {
        // Track loop boundaries
        if matches!(instr, VmInstr::LoopBegin { .. }) {
            loop_depth += 1;
        }

        s.push_str("    {\n");
        s.push_str(&format!("      \"idx\": {},\n", i));
        s.push_str(&format!("      \"name\": {},\n", json_str(&vminstr_name(instr))));
        s.push_str(&format!("      \"loop_depth\": {},\n", loop_depth));
        s.push_str(&format!("      \"detail\": {}\n", json_str(&format!("{:?}", instr))));

        s.push_str(if i + 1 < ctx.instrs.len() { "    },\n" } else { "    }\n" });

        if matches!(instr, VmInstr::LoopEnd) {
            loop_depth = loop_depth.saturating_sub(1);
        }
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

/// Dump the VmInstr sequence in human-readable text format.
pub fn vminstr_to_text(ctx: &VmInstrDumpContext) -> String {
    let mut s = String::with_capacity(16384);
    s.push_str(&format!("VmInstr Sequence Dump: {} instructions\n", ctx.instrs.len()));
    s.push_str(&"=".repeat(60));
    s.push('\n');

    // ABI bindings
    if !ctx.abi_bindings.is_empty() {
        s.push_str("\nABI Bindings:\n");
        for (vreg, name) in ctx.abi_bindings {
            s.push_str(&format!("  v{} = {}\n", vreg.0, name));
        }
    }

    // Register allocation summary
    if let Some(alloc) = ctx.reg_alloc {
        s.push_str(&format!("\nRegister Allocation: {} vregs, {} spills\n",
            alloc.num_vregs(), alloc.spills.len()));
        if !alloc.spills.is_empty() {
            s.push_str("Spills:\n");
            for slot in &alloc.spills {
                s.push_str(&format!("  v{} -> offset={} size={}\n",
                    slot.vreg.0, slot.offset, slot.size));
            }
        }
    }

    s.push_str("\nInstructions:\n");

    let mut loop_depth = 0u32;
    for (i, instr) in ctx.instrs.iter().enumerate() {
        if matches!(instr, VmInstr::LoopBegin { .. }) {
            loop_depth += 1;
            s.push_str(&format!("[{:>4}] {:>2} | >>> LOOP BEGIN\n", i, loop_depth));
        }

        let indent = "  ".repeat(loop_depth as usize);
        let name = vminstr_name(instr);
        let reg_info = vminstr_reg_summary(instr);
        s.push_str(&format!("[{:>4}] {:>2} | {}{} {}\n", i, loop_depth, indent, name, reg_info));

        // Trace mapping
        if let Some((_, desc)) = ctx.trace_mapping.iter().find(|(idx, _)| *idx == i) {
            s.push_str(&format!("         {}  <- trace: {}\n", indent, desc));
        }

        if matches!(instr, VmInstr::LoopEnd) {
            s.push_str(&format!("[{:>4}] {:>2} | <<< LOOP END\n", i, loop_depth));
            loop_depth = loop_depth.saturating_sub(1);
        }
    }

    s
}

// ── Helper functions ────────────────────────────────────────────────

/// Extract the Op variant name from Debug format.
fn op_variant_name(op: &Op) -> String {
    let debug = format!("{:?}", op);
    debug.split('(').next()
        .unwrap_or(&debug)
        .split('{').next()
        .unwrap_or(&debug)
        .to_string()
}

/// Compute a brief name for a ComputePattern.
fn compute_pattern_brief(pattern: &ComputePattern) -> String {
    match pattern {
        ComputePattern::Elementwise { .. } => "Elementwise".to_string(),
        ComputePattern::BinaryElementwise { .. } => "BinaryElementwise".to_string(),
        ComputePattern::Injective { num_inputs, num_outputs, .. } =>
            format!("Injective(inputs={}, outputs={})", num_inputs, num_outputs),
        ComputePattern::Reduction { .. } => "Reduction".to_string(),
        ComputePattern::NormLike { .. } => "NormLike".to_string(),
        ComputePattern::Gemm => "Gemm".to_string(),
        ComputePattern::QuantDecode { block_size, .. } =>
            format!("QuantDecode(block_size={})", block_size),
    }
}

/// Get the body TraceOps from a ComputePattern.
fn trace_ops_for_pattern(pattern: &ComputePattern) -> &[TraceOp] {
    match pattern {
        ComputePattern::Elementwise { body } => body,
        ComputePattern::BinaryElementwise { body } => body,
        ComputePattern::Injective { body, .. } => body,
        ComputePattern::QuantDecode { decode, .. } => decode,
        ComputePattern::Reduction { combine, .. } => combine,
        ComputePattern::NormLike { reduce, .. } => reduce,
        ComputePattern::Gemm => &[],
    }
}

/// Extract all ValueId references from a TraceOp slice.
///
/// Uses Debug string parsing — ValueId format is "v{N}" which is stable.
fn extract_value_ids(ops: &[TraceOp]) -> Vec<ValueId> {
    let mut ids = Vec::new();
    for op in ops {
        let debug = format!("{:?}", op);
        let mut pos = 0;
        while let Some(start) = debug[pos..].find("v") {
            let abs_pos = pos + start + 1;
            pos = abs_pos;
            if abs_pos < debug.len() {
                if start > 0 {
                    let prev_char = debug.as_bytes()[abs_pos - 2];
                    if prev_char.is_ascii_alphabetic() || prev_char == b'_' {
                        continue;
                    }
                }
                let num_start = abs_pos;
                let num_end = debug[num_start..]
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(debug.len() - num_start);
                if num_end > 0 {
                    if let Ok(n) = debug[num_start..num_start + num_end].parse::<u32>() {
                        if n != u32::MAX {
                            ids.push(ValueId(n));
                        }
                    }
                }
            }
        }
    }
    ids.sort_by_key(|v| v.0);
    ids.dedup();
    ids
}

/// Guard label for DOT/JSON annotation.
fn guard_label(guard: &crate::compiler::graph::LayerCondition) -> Option<String> {
    use crate::compiler::graph::LayerCondition;
    match guard {
        LayerCondition::Always => None,
        LayerCondition::LayerIdxLt(threshold) => Some(format!("layer_idx<{}", threshold)),
        LayerCondition::LayerIdxGe(threshold) => Some(format!("layer_idx>={}", threshold)),
    }
}

/// Op category color for DOT output.
fn op_category_color(op: &Op) -> &'static str {
    match op.category() {
        "norm" => "#dae8fc",
        "gemm" => "#f8cecc",
        "activation" => "#d5e8d4",
        "attention" => "#fff2cc",
        "rope" => "#e1d5e7",
        "moe" => "#ffe6cc",
        "sampling" => "#f5f5f5",
        "vision_audio" => "#fff0f0",
        "altup" => "#e0e0ff",
        _ => "#f0f0f0",
    }
}

/// Format SymDim vector to human-readable string.
fn symdim_vec_to_string(shape: &[crate::compiler::graph::SymDim]) -> String {
    use crate::compiler::graph::SymDim;
    let parts: Vec<String> = shape.iter().map(|d| match d {
        SymDim::Concrete(v) => format!("{}", v),
        SymDim::Symbolic { name, max_value } => {
            match max_value {
                Some(mv) => format!("{}(max={})", name, mv),
                None => name.clone(),
            }
        }
    }).collect();
    format!("[{}]", parts.join(", "))
}

/// Get the VmInstr variant name from Debug format.
fn vminstr_name(instr: &VmInstr) -> String {
    let debug = format!("{:?}", instr);
    debug.split('{').next()
        .unwrap_or(&debug)
        .split('(').next()
        .unwrap_or(&debug)
        .to_string()
}

/// Build a short register summary for a VmInstr using Debug format.
fn vminstr_reg_summary(instr: &VmInstr) -> String {
    let debug = format!("{:?}", instr);
    // Truncate long debug output for readability
    if debug.len() > 80 {
        format!(" {}", &debug[..80])
    } else {
        format!(" {}", debug)
    }
}

/// Brief physical register name.
fn phys_brief(phys: PhysReg) -> String {
    match phys {
        PhysReg::Gpr(g) => format!("r{}", g.0),
        PhysReg::Vec(v) => format!("ymm{}", v.0),
        PhysReg::Spilled(slot) => format!("spill{}", slot),
        PhysReg::Mask(m) => format!("k{}", m.0),
        PhysReg::Tile(t) => format!("tmm{}", t.0),
    }
}

/// JSON-escape a string value and wrap in double quotes.
fn json_str(s: impl AsRef<str>) -> String {
    let s = s.as_ref();
    let mut escaped = String::with_capacity(s.len() + 2);
    escaped.push('"');
    for c in s.chars() {
        match c {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_control() => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // @trace TEST-DUMP-001 [entity:CompilerGraph] test_graph_dot_basic

    /// TEST-DUMP-001: CompilerGraph DOT dump must contain op nodes and tensor edges.
    #[test]
    fn test_graph_dot_basic() {
        let graph = build_test_graph();
        let op_traces: Vec<Option<OpTrace>> = vec![];
        let dot = graph_to_dot(&graph, &op_traces, None);

        assert!(dot.starts_with("digraph CompilerGraph"), "DOT must start with digraph");
        assert!(dot.contains("}"), "DOT must end with closing brace");
        assert!(dot.contains("op0"), "DOT must contain op nodes");
        assert!(dot.contains("->"), "DOT must contain edges");
    }

    // @trace TEST-DUMP-001 [entity:CompilerGraph] test_graph_dot_with_topology

    /// TEST-DUMP-001: DOT dump must include GraphTopologyAnalysis result.
    #[test]
    fn test_graph_dot_with_topology() {
        let graph = build_test_graph();
        let topo = GraphTopologyAnalysis::analyze(&graph);
        let op_traces: Vec<Option<OpTrace>> = vec![];
        let dot = graph_to_dot(&graph, &op_traces, Some(&topo));

        assert!(dot.contains("loop_topology"), "DOT must contain loop_topology");
        assert!(dot.contains("kv_cache_source"), "DOT must contain kv_cache_source");
    }

    // @trace TEST-DUMP-001 [entity:CompilerGraph] test_graph_json_basic

    /// TEST-DUMP-001: CompilerGraph JSON dump must contain ops, tensors.
    #[test]
    fn test_graph_json_basic() {
        let graph = build_test_graph();
        let op_traces: Vec<Option<OpTrace>> = vec![];
        let json = graph_to_json(&graph, &op_traces, None);

        assert!(json.starts_with("{"), "JSON must start with {{");
        assert!(json.ends_with("}\n"), "JSON must end with }}\\n");
        assert!(json.contains("\"ops\":"), "JSON must contain ops");
        assert!(json.contains("\"tensors\":"), "JSON must contain tensors");
        assert!(json.contains("\"op\":"), "JSON must contain op variant name");
        assert!(json.contains("\"category\":"), "JSON must contain category");
    }

    // @trace TEST-DUMP-001 [entity:CompilerGraph] test_graph_json_with_topology_and_pattern

    /// TEST-DUMP-001: JSON dump must include topology and ComputePattern.
    #[test]
    fn test_graph_json_with_topology_and_pattern() {
        let graph = build_test_graph();
        let topo = GraphTopologyAnalysis::analyze(&graph);
        let op_traces: Vec<Option<OpTrace>> = vec![None];
        let json = graph_to_json(&graph, &op_traces, Some(&topo));

        assert!(json.contains("\"topology\":"), "JSON must contain topology");
        assert!(json.contains("\"compute_pattern\":"), "JSON must contain compute_pattern");
    }

    // @trace TEST-DUMP-002 [entity:TraceOp] test_trace_ssa_json

    /// TEST-DUMP-002: TraceOp SSA JSON dump must contain required fields.
    #[test]
    fn test_trace_ssa_json() {
        let graph = build_test_graph();
        let op_traces: Vec<Option<OpTrace>> = vec![None];
        let ctx = TraceDumpContext { graph: &graph, op_traces: &op_traces };
        let json = trace_ssa_to_json(&ctx);

        assert!(json.starts_with("{"), "JSON must start with {{");
        assert!(json.contains("\"source_op\":"), "JSON must contain source_op mapping");
        assert!(json.contains("\"compute_pattern\":"), "JSON must contain compute_pattern");
        assert!(json.contains("\"input_tensors\":"), "JSON must contain input_tensors");
        assert!(json.contains("\"output_tensors\":"), "JSON must contain output_tensors");
        assert!(json.contains("\"ssa_registers\":"), "JSON must contain ssa_registers");
    }

    // @trace TEST-DUMP-002 [entity:TraceOp] test_trace_ssa_text

    /// TEST-DUMP-002: TraceOp SSA text dump must be human-readable.
    #[test]
    fn test_trace_ssa_text() {
        let graph = build_test_graph();
        let op_traces: Vec<Option<OpTrace>> = vec![None];
        let ctx = TraceDumpContext { graph: &graph, op_traces: &op_traces };
        let text = trace_ssa_to_text(&ctx);

        assert!(text.contains("TraceOp SSA Dump"), "Text must contain header");
    }

    // @trace TEST-DUMP-003 [entity:VmInstr] test_vminstr_json

    /// TEST-DUMP-003: VmInstr JSON dump must contain required fields.
    #[test]
    fn test_vminstr_json() {
        let prog = build_test_vm_program();
        let alloc = build_test_reg_alloc();
        let abi_binding: [(VRegId, &str); 0] = [];
        let trace_mapping: [(usize, String); 0] = [];
        let ctx = VmInstrDumpContext {
            instrs: &prog,
            reg_alloc: Some(&alloc),
            abi_bindings: &abi_binding,
            trace_mapping: &trace_mapping,
        };
        let json = vminstr_to_json(&ctx);

        assert!(json.starts_with("{"), "JSON must start with {{");
        assert!(json.contains("\"num_instrs\":"), "JSON must contain num_instrs");
        assert!(json.contains("\"instrs\":"), "JSON must contain instrs array");
        assert!(json.contains("\"loop_depth\":"), "JSON must contain loop_depth");
        assert!(json.contains("\"reg_alloc\":"), "JSON must contain reg_alloc");
    }

    // @trace TEST-DUMP-003 [entity:VmInstr] test_vminstr_text

    /// TEST-DUMP-003: VmInstr text dump must show loop boundaries and register info.
    #[test]
    fn test_vminstr_text() {
        let prog = build_test_vm_program();
        let abi_bindings: [(VRegId, &str); 2] = [
            (VRegId(0), "input_ptr"),
            (VRegId(1), "weight_ptr"),
        ];
        let trace_mapping: [(usize, String); 0] = [];
        let ctx = VmInstrDumpContext {
            instrs: &prog,
            reg_alloc: None,
            abi_bindings: &abi_bindings,
            trace_mapping: &trace_mapping,
        };
        let text = vminstr_to_text(&ctx);

        assert!(text.contains("VmInstr Sequence Dump"), "Text must contain header");
        assert!(text.contains("ABI Bindings"), "Text must show ABI bindings");
        assert!(text.contains("LOOP BEGIN"), "Text must show loop boundaries");
        assert!(text.contains("LOOP END"), "Text must show loop boundaries");
        assert!(text.contains("input_ptr"), "Text must show ABI parameter names");
    }

    // @trace TEST-DUMP-003 [entity:VmInstr] test_vminstr_json_with_trace_mapping

    /// TEST-DUMP-003: VmInstr JSON must contain TraceOp mapping when provided.
    #[test]
    fn test_vminstr_json_with_trace_mapping() {
        let prog = build_test_vm_program();
        let trace_mapping = vec![
            (0usize, "TraceOp::Input(0)".to_string()),
            (1, "TraceOp::Mul(v0, v1)".to_string()),
        ];
        let abi_bindings: [(VRegId, &str); 0] = [];
        let ctx = VmInstrDumpContext {
            instrs: &prog,
            reg_alloc: None,
            abi_bindings: &abi_bindings,
            trace_mapping: &trace_mapping,
        };
        let json = vminstr_to_json(&ctx);

        assert!(json.contains("\"trace_mapping\":"), "JSON must contain trace_mapping");
        assert!(json.contains("TraceOp::Input(0)"), "JSON must contain trace mapping detail");
    }

    // @trace TEST-DUMP-001 [entity:CompilerGraph] test_json_str_escaping

    /// Verify JSON string escaping handles special characters.
    #[test]
    fn test_json_str_escaping() {
        assert_eq!(json_str("hello"), "\"hello\"");
        assert_eq!(json_str("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(json_str("line1\nline2"), "\"line1\\nline2\"");
        assert_eq!(json_str("tab\there"), "\"tab\\there\"");
        assert_eq!(json_str("back\\slash"), "\"back\\\\slash\"");
    }

    // ── Test fixtures ──

    fn build_test_graph() -> CompilerGraph {
        use crate::compiler::graph::{SymDim, NormSpec};
        use crate::types::DType;

        let mut graph = CompilerGraph::new();

        let input_tid = graph.add_tensor(
            "input",
            vec![SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(2048) }, SymDim::Concrete(4096)],
            DType::F32,
        );

        let weight_tid = graph.add_tensor(
            "weight",
            vec![SymDim::Concrete(4096), SymDim::Concrete(4096)],
            DType::F32,
        );

        let norm_out_tid = graph.add_tensor(
            "norm_out",
            vec![SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(2048) }, SymDim::Concrete(4096)],
            DType::F32,
        );

        graph.add_op(
            Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true }),
            vec![input_tid],
            vec![norm_out_tid],
            "layer.input_layernorm",
        );

        graph.inputs = vec![input_tid, weight_tid];
        graph.outputs = vec![norm_out_tid];

        graph
    }

    fn build_test_vm_program() -> Vec<VmInstr> {
        use crate::compiler::codegen::vm::instr::{OffsetExpr, SimdWidth, SymBound};
        use crate::compiler::codegen::vm::instr::BoundExpr;
        use crate::compiler::trace::QuantPrecision;

        vec![
            VmInstr::DeclareVReg {
                id: VRegId(0),
                kind: crate::compiler::codegen::vm::instr::VRegKind::Ptr,
                width: SimdWidth::Scalar,
            },
            VmInstr::DeclareVReg {
                id: VRegId(1),
                kind: crate::compiler::codegen::vm::instr::VRegKind::Ptr,
                width: SimdWidth::Scalar,
            },
            VmInstr::DeclareVReg {
                id: VRegId(2),
                kind: crate::compiler::codegen::vm::instr::VRegKind::Vec,
                width: SimdWidth::W256,
            },
            VmInstr::VecLoad {
                dst: VRegId(2),
                base: VRegId(0),
                offset: OffsetExpr::Const(0),
                width: SimdWidth::W256,
                dtype: QuantPrecision::F32,
                predicate: None,
            },
            VmInstr::LoopBegin {
                counter: VRegId(3),
                byte_offset: VRegId(4),
                bound: BoundExpr::Symbolic(SymBound { name: "seq_len".to_string(), max_alloc: 2048 }),
                step_bytes: 4,
            },
            VmInstr::Fma {
                dst: VRegId(5),
                acc: VRegId(2),
                a: VRegId(6),
                b: VRegId(7),
                dtype: QuantPrecision::F32,
            },
            VmInstr::LoopEnd,
            VmInstr::VecStore {
                base: VRegId(1),
                offset: OffsetExpr::Const(0),
                src: VRegId(5),
                width: SimdWidth::W256,
                dtype: QuantPrecision::F32,
                predicate: None,
            },
        ]
    }

    fn build_test_reg_alloc() -> RegAllocation {
        use std::collections::HashMap;
        use crate::compiler::codegen::vm::isa_profile::{PhysGpr, PhysVec};
        use crate::compiler::codegen::vm::reg_alloc::SpillSlot;

        let mut mapping = HashMap::new();
        mapping.insert(VRegId(0), PhysReg::Gpr(PhysGpr(0)));
        mapping.insert(VRegId(1), PhysReg::Gpr(PhysGpr(1)));
        mapping.insert(VRegId(2), PhysReg::Vec(PhysVec(0)));
        mapping.insert(VRegId(3), PhysReg::Gpr(PhysGpr(2)));
        mapping.insert(VRegId(4), PhysReg::Gpr(PhysGpr(3)));
        mapping.insert(VRegId(5), PhysReg::Vec(PhysVec(1)));
        mapping.insert(VRegId(6), PhysReg::Vec(PhysVec(2)));
        mapping.insert(VRegId(7), PhysReg::Vec(PhysVec(3)));

        RegAllocation {
            mapping,
            spills: vec![SpillSlot { vreg: VRegId(8), offset: 0, size: 32 }],
            callee_saved_used: vec![],
        }
    }
}
