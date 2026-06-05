//! GPU Backend Dialect Trait — GPU ISA 输出格式抽象
//!
//! 每种 GPU ISA (PTX/HIP/Metal) 实现此 trait，提供方言特定的代码生成方法。
//! `GpuLower` 通过 trait 对象 `&dyn GpuBackendDialect` 分发，消除 144 处 `match self.dialect` 分支。
//!
//! ## 设计原则
//!
//! - trait 方法只生成字符串（通过 `&mut GpuLowerContext` 写入 IR buffer）
//! - 方言不持有可变状态 — 所有状态在 `GpuLowerContext` 中
//! - 扩展新 GPU 后端只需 `impl GpuBackendDialect`，无需修改 `gpu_lower.rs`
//!
//! 代码组织 (include! 模式):
//! - `gpu_dialect_fragments/types.inc.rs` — types + trait + GpuLowerContext
//! - `gpu_dialect_fragments/ptx.inc.rs`   — PtxDialect impl
//! - `gpu_dialect_fragments/hip.inc.rs`   — HipDialect impl + shared helpers
//! - `gpu_dialect_fragments/metal.inc.rs` — MetalDialect impl + factory

use super::instr::*;
use crate::types::CompilerError;

include!("gpu_dialect_fragments/types.inc.rs");
include!("gpu_dialect_fragments/ptx.inc.rs");
include!("gpu_dialect_fragments/hip.inc.rs");
include!("gpu_dialect_fragments/metal.inc.rs");
