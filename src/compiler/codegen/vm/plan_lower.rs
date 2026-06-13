//! FusionPlan → VmProgram 状态追踪 + compile_layer 全链路 (REGISTER-VM SPEC §13)
//!
//! 全管线: FusionPlan → 状态追踪 IR → VmOptPass → RegAlloc → StackFrame → IsaLower → 物理代码
//!
//! VmProgram 不是虚拟机——它是编译时半结构化状态追踪 IR，记录寄存器分配、
//! 内存布局、栈帧的约束关系，供后续 RegAlloc/StackFrame/IsaLower 消费。
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 5 个片段):
//! - `plan_lower/context.inc.rs` — LoweringContext, TensorPtrResolver, SymDimSlotMap, scratchpad 计算
//! - `plan_lower/pipeline.inc.rs` — lower_fusion_plan, emit_fusion_groups, emit_elementwise_inline
//! - `plan_lower/compile.inc.rs`  — compile_layer, emit_standalone_op, 辅助函数
//! - `plan_lower/template.inc.rs` — compile_layer_type_body, GroupDependencyAnalyzer
//! - `plan_lower/tests.inc.rs`    — 测试模块

use super::instr::*;
use super::isa_profile::IsaProfile;
use super::isa_hook;
use super::lower;
use super::reg_alloc::RegAllocator;
use super::stack_frame::StackFrame;
use super::opt_pass::PassRegistry;
use super::x86_lower::X86Lower;
use super::vm_state::{HeteroPhase, EmitState, AbiPtrs};

use crate::compiler::codegen::{CodegenOutput, DwcScratchRequirement, PleScratchRequirement, RopeCacheRequirement};
use crate::compiler::fusion::{FusionPlan, FusionMode, HeteroLayerType};
use crate::compiler::graph::{CompilerGraph, LayerCondition, OpKind, SymDim, TensorId};
use crate::compiler::buffer_alloc::{BufferAllocation, TensorPtrSource};
use crate::compiler::registry::ScalarOpRegistry;
use crate::compiler::trace::{QuantPrecision, TraceOp, ComputePattern, ValueId};
use crate::compiler::pain_point::{OpBottleneckMap, ExecPattern, ParallelismDesc};
use crate::compiler::virtual_activation::VirtualActivationMap;
use crate::compiler::virtual_tensor::VirtualTensorMap;
use crate::types::CompilerError;
use crate::dispatch::device_profile::DotProductCap;

use super::telemetry_emit::emit_silu_dead_neuron_telemetry;
use super::norm_softmax_emit::{emit_normlike_inline, NormKind};
use super::dispatch_emit::{dispatch_structural, dispatch_compute_pattern};
use super::fusion_group_emit::emit_fusion_group_by_mode;
use super::gemm_emit::emit_gemm_inline_with_hook;
use super::hetero_emit::compile_hetero_templates_parallel;

/// Re-export KvLoadMode from instr.rs
pub use super::instr::KvLoadMode;

include!("plan_lower/context.inc.rs");
include!("plan_lower/pipeline.inc.rs");
include!("plan_lower/compile.inc.rs");
include!("plan_lower/template.inc.rs");

include!("plan_lower/tests.inc.rs");
