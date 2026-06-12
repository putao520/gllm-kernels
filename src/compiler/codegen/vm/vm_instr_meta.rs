//! VmInstrMeta — 融合优化元数据 (ISA 无关，编译时推导，运行时零开销)
//!
//! SPEC: `SPEC/26-VMINSTR-RATIONALIZATION.md §0.5`
//! REQ: REQ-VM-META-001~008
//!
//! 四维抽象模型:
//! 1. FusionCompat (loop_bound + data_flow) — 能否融合
//! 2. FusionValue (arithmetic_intensity) — 值不值得
//! 3. ResourceNeed (compute_regs + working_set_bytes) — 需要多少资源
//! 4. Boundary (quant_boundary + side_effect) — 有没有硬边界

use std::fmt;

// ============================================================================
// §0.5.2 LoopBoundAffinity — 循环绑定维度
// ============================================================================

/// VmInstr 的循环绑定维度 — 描述指令在哪个循环维度上执行。
#[derive(Clone, Debug)]
pub enum LoopBoundAffinity {
    /// 编译时固定维度 (如 head_dim/8 向量化内循环)
    Fixed { dim: usize },
    /// 运行时符号维度 (如 seq_len, batch)
    Symbolic { name: String },
    /// GEMM 三维绑定 (M, K, N)
    Gemm {
        m: DimBinding,
        k: DimBinding,
        n: DimBinding,
    },
    /// 嵌套循环 (外层 × 内层)
    Nested {
        outer: Box<LoopBoundAffinity>,
        inner: Box<LoopBoundAffinity>,
    },
    /// 无循环绑定 (控制流/标量操作)
    None,
}

/// 维度绑定 — 固定值或符号名
#[derive(Clone, Debug)]
pub enum DimBinding {
    Fixed(usize),
    Symbolic(String),
}

// ============================================================================
// §0.5.4 DataFlow — 数据流方向
// ============================================================================

/// VmInstr 的数据流方向 — 决定融合兼容性。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataFlow {
    /// 读一次不再重用 (VecMul, VecAdd, SiLU, Cast)
    Streaming,
    /// 多值→少值 (SoftmaxMax, MeanPool, Argmax, Sum)
    Reductive,
    /// 少值→多值 (Broadcast, MoEDispatch, EmbeddingExpand)
    Expansive,
    /// 输出覆盖输入 (Cast, Reshape)
    InPlace,
    /// 累加到目标 (GEMM accumulator, Epilogue chain, Attention O)
    Accumulative,
}

/// 融合兼容性矩阵: producer (行) × consumer (列) → 是否可融合
pub fn can_fuse_data_flow(producer: DataFlow, consumer: DataFlow) -> bool {
    match (producer, consumer) {
        // Streaming → Streaming: LoopFusion
        (DataFlow::Streaming, DataFlow::Streaming) => true,
        // Streaming → Reductive: 边界隔离
        (DataFlow::Streaming, DataFlow::Reductive) => false,
        // Streaming → Expansive: Broadcast 扩展
        (DataFlow::Streaming, DataFlow::Expansive) => true,
        // Streaming → Accumulative: Epilogue
        (DataFlow::Streaming, DataFlow::Accumulative) => true,
        // Reductive → *: 通常隔离 (除非合并 reduction)
        (DataFlow::Reductive, DataFlow::Streaming) => false,
        (DataFlow::Reductive, DataFlow::Reductive) => true, // 合并 reduction
        (DataFlow::Reductive, DataFlow::Expansive) => false,
        (DataFlow::Reductive, DataFlow::InPlace) => false,
        (DataFlow::Reductive, DataFlow::Accumulative) => false,
        // Expansive → *: 通常隔离
        (DataFlow::Expansive, _) => false,
        // InPlace → Streaming: 继续
        (DataFlow::InPlace, DataFlow::Streaming) => true,
        (DataFlow::InPlace, _) => false,
        // Accumulative → Streaming: Epilogue 完成
        (DataFlow::Accumulative, DataFlow::Streaming) => true,
        // Accumulative → Accumulative: 链式累加
        (DataFlow::Accumulative, DataFlow::Accumulative) => true,
        (DataFlow::Accumulative, _) => false,
    }
}

// ============================================================================
// §0.5.7 QuantBoundary — 量化边界
// ============================================================================

/// 量化边界标记 — 标记 VmInstr 的输入/输出量化类型。
#[derive(Clone, Debug)]
pub struct QuantBoundary {
    /// 输入量化类型 (None = FP32/默认)
    pub input_quant: Option<QuantType>,
    /// 输出量化类型 (None = FP32/默认)
    pub output_quant: Option<QuantType>,
    /// 反量化 FLOP 开销 (0.0 = 硬件原生/无额外开销)
    pub dequant_flops: f32,
}

impl QuantBoundary {
    /// 无量化边界 (纯 FP32 路径)
    pub const fn none() -> Self {
        Self {
            input_quant: None,
            output_quant: None,
            dequant_flops: 0.0,
        }
    }

    /// 是否跨越量化类型边界
    pub fn is_boundary(&self) -> bool {
        self.input_quant.is_some() || self.output_quant.is_some()
    }
}

/// 量化类型摘要 (用于 QuantBoundary 标记)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantType {
    Fp32,
    Bf16,
    Fp16,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    Int4,
    Q4_0,
    Q4_1,
    Q4K,
    Q5_0,
    Q5_1,
    Q5K,
    Q6K,
    Q2K,
    Q3K,
    Mxfp4,
    Nvfp4,
}

// ============================================================================
// §0.5.8 SideEffect — 副作用标记
// ============================================================================

/// Classification of a VM instruction's side effect.
///
/// ARCH-JIT-DATA-YIELDS: replaces `is_control_flow: bool` with a semantic enum.
/// The effect family drives scheduling and fence decisions, not a bool flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectFamily {
    /// Pure computation — no side effects, freely reorderable.
    Pure,
    /// Memory side effect — writes to memory, needs ordering but not control flow.
    Memory,
    /// Control flow — alters execution path (branches, loops, conditional jumps).
    ControlFlow,
}

/// VmInstr 副作用标记 — 标记融合边界约束。
#[derive(Clone, Debug)]
pub struct SideEffect {
    /// 写入 KV 缓存 → 强制融合边界
    pub writes_kv_cache: bool,
    /// 写入输出缓冲 → 强制融合边界
    pub writes_output_buffer: bool,
    /// 写入共享内存
    pub writes_shared_mem: bool,
    /// 读取共享状态 → 允许融合但需保留屏障
    pub reads_shared_state: bool,
    /// 副作用族 — 驱动调度和屏障决策
    pub effect_family: EffectFamily,
}

impl SideEffect {
    /// 无副作用
    pub const fn none() -> Self {
        Self {
            writes_kv_cache: false,
            writes_output_buffer: false,
            writes_shared_mem: false,
            reads_shared_state: false,
            effect_family: EffectFamily::Pure,
        }
    }

    /// 是否有强制融合边界的副作用
    pub fn is_fusion_barrier(&self) -> bool {
        self.writes_kv_cache
            || self.writes_output_buffer
            || self.effect_family == EffectFamily::ControlFlow
    }
}

// ============================================================================
// §0.5.1 VmInstrMeta — 融合优化元数据
// ============================================================================

/// VmInstr 融合优化元数据 — ISA 无关，编译时推导，运行时零开销。
///
/// 四维抽象模型:
/// - FusionCompat: loop_bound + data_flow — 能否融合
/// - FusionValue: arithmetic_intensity — 值不值得
/// - ResourceNeed: compute_regs + working_set_bytes — 需要多少资源
/// - Boundary: quant_boundary + side_effect — 有没有硬边界
#[derive(Clone, Debug)]
pub struct VmInstrMeta {
    // ── FusionCompat: 能否融合 ──
    pub loop_bound: LoopBoundAffinity,
    pub data_flow: DataFlow,

    // ── FusionValue: 值不值得 ──
    pub arithmetic_intensity: f32,

    // ── ResourceNeed: 需要多少资源 (抽象值，ISA 无关) ──
    pub compute_regs: u8,
    pub working_set_bytes: usize,

    // ── Boundary: 有没有硬边界 ──
    pub quant_boundary: QuantBoundary,
    pub side_effect: SideEffect,
}

impl VmInstrMeta {
    /// 算术密度分类
    pub fn bottleneck(&self) -> Bottleneck {
        if self.arithmetic_intensity < 1.0 {
            Bottleneck::MemoryBound
        } else if self.arithmetic_intensity > 10.0 {
            Bottleneck::ComputeBound
        } else {
            Bottleneck::Balanced
        }
    }
}

/// 算术密度分类
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bottleneck {
    MemoryBound,
    Balanced,
    ComputeBound,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bottleneck::MemoryBound => write!(f, "MemoryBound"),
            Bottleneck::Balanced => write!(f, "Balanced"),
            Bottleneck::ComputeBound => write!(f, "ComputeBound"),
        }
    }
}

// ============================================================================
// §0.5.9 DeviceProfile 派生函数
// ============================================================================

/// 寄存器预算
#[derive(Clone, Debug)]
pub struct RegBudget {
    pub need: u8,
    pub available: u8,
}

/// 缓存适配
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheFit {
    L1,
    L2,
    Global,
}

/// 从抽象 Meta + DeviceProfile 推导对齐要求
pub fn derive_alignment(simd_width_bytes: usize, compute_regs: u8) -> usize {
    if compute_regs == 0 {
        1 // scalar: 无对齐
    } else {
        simd_width_bytes // vector: SIMD 宽度即为对齐
    }
}

/// 从抽象 Meta 推导寄存器预算
pub fn derive_reg_budget(need: u8, available: u8) -> RegBudget {
    RegBudget { need, available }
}

/// 从工作集 + 缓存层级推导适配
pub fn derive_cache_fit(working_set_bytes: usize, l1_bytes: usize, l2_bytes: usize) -> CacheFit {
    if working_set_bytes <= l1_bytes * 3 / 4 {
        CacheFit::L1
    } else if working_set_bytes <= l2_bytes {
        CacheFit::L2
    } else {
        CacheFit::Global
    }
}

/// 融合策略选择 (基于 arithmetic_intensity)
pub fn derive_fusion_strategy(arithmetic_intensity: f32) -> FusionStrategy {
    if arithmetic_intensity < 1.0 {
        FusionStrategy::LoopFusion
    } else if arithmetic_intensity > 10.0 {
        FusionStrategy::EpilogueInjection
    } else {
        FusionStrategy::TileOrComputeRoot
    }
}

/// 融合策略
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionStrategy {
    LoopFusion,
    TileOrComputeRoot,
    EpilogueInjection,
}

// ============================================================================
// 常用 VmInstrMeta 工厂函数
// ============================================================================

/// 内存加载指令元数据
pub fn meta_vec_load(working_set_bytes: usize, quant: Option<QuantType>) -> VmInstrMeta {
    VmInstrMeta {
        loop_bound: LoopBoundAffinity::Fixed { dim: 0 },
        data_flow: DataFlow::Streaming,
        arithmetic_intensity: 0.0,
        compute_regs: 1,
        working_set_bytes,
        quant_boundary: QuantBoundary {
            input_quant: quant,
            output_quant: None,
            dequant_flops: quant.map_or(0.0, |q| dequant_flops_for(q)),
        },
        side_effect: SideEffect::none(),
    }
}

/// 内存存储指令元数据
pub fn meta_vec_store(working_set_bytes: usize) -> VmInstrMeta {
    VmInstrMeta {
        loop_bound: LoopBoundAffinity::Fixed { dim: 0 },
        data_flow: DataFlow::Streaming,
        arithmetic_intensity: 0.0,
        compute_regs: 1,
        working_set_bytes,
        quant_boundary: QuantBoundary::none(),
        side_effect: SideEffect {
            writes_output_buffer: true,
            effect_family: EffectFamily::Memory,
            ..SideEffect::none()
        },
    }
}

/// 逐元素二元操作元数据
pub fn meta_elem_binop(working_set_bytes: usize) -> VmInstrMeta {
    VmInstrMeta {
        loop_bound: LoopBoundAffinity::Symbolic { name: "batch".into() },
        data_flow: DataFlow::Streaming,
        arithmetic_intensity: 0.125,
        compute_regs: 3,
        working_set_bytes,
        quant_boundary: QuantBoundary::none(),
        side_effect: SideEffect::none(),
    }
}

/// GEMM 微核元数据
pub fn meta_gemm_microkernel(m: usize, k: usize, n: usize, working_set_bytes: usize) -> VmInstrMeta {
    VmInstrMeta {
        loop_bound: LoopBoundAffinity::Gemm {
            m: DimBinding::Fixed(m),
            k: DimBinding::Fixed(k),
            n: DimBinding::Fixed(n),
        },
        data_flow: DataFlow::Accumulative,
        arithmetic_intensity: 10.0,
        compute_regs: 8,
        working_set_bytes,
        quant_boundary: QuantBoundary::none(),
        side_effect: SideEffect::none(),
    }
}

/// 归约操作元数据
pub fn meta_reduction(working_set_bytes: usize) -> VmInstrMeta {
    VmInstrMeta {
        loop_bound: LoopBoundAffinity::Symbolic { name: "seq_len".into() },
        data_flow: DataFlow::Reductive,
        arithmetic_intensity: 0.5,
        compute_regs: 2,
        working_set_bytes,
        quant_boundary: QuantBoundary::none(),
        side_effect: SideEffect::none(),
    }
}

/// 控制流指令元数据
pub fn meta_control_flow() -> VmInstrMeta {
    VmInstrMeta {
        loop_bound: LoopBoundAffinity::None,
        data_flow: DataFlow::Streaming,
        arithmetic_intensity: 0.0,
        compute_regs: 2,
        working_set_bytes: 0,
        quant_boundary: QuantBoundary::none(),
        side_effect: SideEffect {
            effect_family: EffectFamily::ControlFlow,
            ..SideEffect::none()
        },
    }
}

/// KV cache 写入元数据
pub fn meta_kv_write(working_set_bytes: usize) -> VmInstrMeta {
    VmInstrMeta {
        loop_bound: LoopBoundAffinity::Symbolic { name: "seq_len".into() },
        data_flow: DataFlow::Streaming,
        arithmetic_intensity: 0.0,
        compute_regs: 1,
        working_set_bytes,
        quant_boundary: QuantBoundary::none(),
        side_effect: SideEffect {
            writes_kv_cache: true,
            effect_family: EffectFamily::Memory,
            ..SideEffect::none()
        },
    }
}

/// 反量化 FLOP 开销参考值 (SPEC §0.5.7)
fn dequant_flops_for(qt: QuantType) -> f32 {
    match qt {
        QuantType::Fp32 => 0.0,
        QuantType::Bf16 => 0.0,
        QuantType::Fp16 => 0.0,
        QuantType::Int8 => 0.5,
        QuantType::Q4_0 | QuantType::Q4_1 => 2.0,
        QuantType::Q4K => 3.0,
        QuantType::Q5_0 | QuantType::Q5_1 => 2.5,
        QuantType::Q5K | QuantType::Q6K => 3.0,
        QuantType::Q2K | QuantType::Q3K => 3.5,
        QuantType::Fp8E4M3 | QuantType::Fp8E5M2 => 0.0,
        QuantType::Int4 => 2.0,
        QuantType::Mxfp4 => 0.0,
        QuantType::Nvfp4 => 0.0,
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_flow_fusion_compatibility() {
        assert!(can_fuse_data_flow(DataFlow::Streaming, DataFlow::Streaming));
        assert!(can_fuse_data_flow(DataFlow::Streaming, DataFlow::Accumulative));
        assert!(can_fuse_data_flow(DataFlow::Accumulative, DataFlow::Accumulative));
        assert!(!can_fuse_data_flow(DataFlow::Reductive, DataFlow::Expansive));
        assert!(!can_fuse_data_flow(DataFlow::Expansive, DataFlow::Streaming));
    }

    #[test]
    fn bottleneck_classification() {
        let mem_bound = VmInstrMeta {
            loop_bound: LoopBoundAffinity::None,
            data_flow: DataFlow::Streaming,
            arithmetic_intensity: 0.125,
            compute_regs: 3,
            working_set_bytes: 1024,
            quant_boundary: QuantBoundary::none(),
            side_effect: SideEffect::none(),
        };
        assert_eq!(mem_bound.bottleneck(), Bottleneck::MemoryBound);

        let compute_bound = VmInstrMeta {
            arithmetic_intensity: 50.0,
            ..mem_bound.clone()
        };
        assert_eq!(compute_bound.bottleneck(), Bottleneck::ComputeBound);

        let balanced = VmInstrMeta {
            arithmetic_intensity: 5.0,
            ..mem_bound.clone()
        };
        assert_eq!(balanced.bottleneck(), Bottleneck::Balanced);
    }

    #[test]
    fn side_effect_barrier() {
        let none = SideEffect::none();
        assert!(!none.is_fusion_barrier());

        let kv_write = SideEffect {
            writes_kv_cache: true,
            ..SideEffect::none()
        };
        assert!(kv_write.is_fusion_barrier());

        let control = SideEffect {
            effect_family: EffectFamily::ControlFlow,
            ..SideEffect::none()
        };
        assert!(control.is_fusion_barrier());
    }

    #[test]
    fn cache_fit_classification() {
        assert_eq!(derive_cache_fit(1024, 32768, 262144), CacheFit::L1);
        assert_eq!(derive_cache_fit(25000, 32768, 262144), CacheFit::L2);
        assert_eq!(derive_cache_fit(300000, 32768, 262144), CacheFit::Global);
    }

    #[test]
    fn fusion_strategy_by_ai() {
        assert_eq!(derive_fusion_strategy(0.125), FusionStrategy::LoopFusion);
        assert_eq!(derive_fusion_strategy(5.0), FusionStrategy::TileOrComputeRoot);
        assert_eq!(derive_fusion_strategy(50.0), FusionStrategy::EpilogueInjection);
    }

    #[test]
    fn dequant_flops_lookup() {
        assert_eq!(dequant_flops_for(QuantType::Fp32), 0.0);
        assert_eq!(dequant_flops_for(QuantType::Bf16), 0.0);
        assert_eq!(dequant_flops_for(QuantType::Int8), 0.5);
        assert_eq!(dequant_flops_for(QuantType::Q4K), 3.0);
        assert_eq!(dequant_flops_for(QuantType::Mxfp4), 0.0);
        assert_eq!(dequant_flops_for(QuantType::Nvfp4), 0.0);
    }

    #[test]
    fn factory_functions() {
        let load = meta_vec_load(4096, Some(QuantType::Bf16));
        assert_eq!(load.compute_regs, 1);
        assert_eq!(load.data_flow, DataFlow::Streaming);

        let store = meta_vec_store(4096);
        assert!(store.side_effect.writes_output_buffer);

        let gemm = meta_gemm_microkernel(4, 16, 4, 1024);
        assert_eq!(gemm.data_flow, DataFlow::Accumulative);
        assert!(gemm.arithmetic_intensity > 1.0);

        let ctrl = meta_control_flow();
        assert_eq!(ctrl.side_effect.effect_family, EffectFamily::ControlFlow);
        assert!(ctrl.side_effect.is_fusion_barrier());
    }

    // ── DataFlow exhaustiveness ──

    #[test]
    fn data_flow_all_variants_distinct() {
        let variants = [
            DataFlow::Streaming, DataFlow::Reductive, DataFlow::Expansive,
            DataFlow::InPlace, DataFlow::Accumulative,
        ];
        for (i, &a) in variants.iter().enumerate() {
            for (j, &b) in variants.iter().enumerate() {
                if i != j { assert_ne!(a, b); }
            }
        }
    }

    #[test]
    fn data_flow_copy() {
        let a = DataFlow::Streaming;
        let b = a;
        assert_eq!(a, b);
    }

    // ── LoopBoundAffinity ──

    #[test]
    fn loop_bound_fixed_dim() {
        let lb = LoopBoundAffinity::Fixed { dim: 128 };
        if let LoopBoundAffinity::Fixed { dim } = lb { assert_eq!(dim, 128); }
        else { panic!("expected Fixed"); }
    }

    #[test]
    fn loop_bound_symbolic() {
        let lb = LoopBoundAffinity::Symbolic { name: "seq_len".into() };
        if let LoopBoundAffinity::Symbolic { name } = lb { assert_eq!(name, "seq_len"); }
        else { panic!("expected Symbolic"); }
    }

    #[test]
    fn loop_bound_gemm() {
        let lb = LoopBoundAffinity::Gemm {
            m: DimBinding::Fixed(4), k: DimBinding::Fixed(16), n: DimBinding::Fixed(4),
        };
        if let LoopBoundAffinity::Gemm { m, k, n } = lb {
            assert_eq!(m, DimBinding::Fixed(4));
            assert_eq!(k, DimBinding::Fixed(16));
        } else { panic!("expected Gemm"); }
    }

    #[test]
    fn loop_bound_nested() {
        let lb = LoopBoundAffinity::Nested {
            outer: Box::new(LoopBoundAffinity::Symbolic { name: "seq_len".into() }),
            inner: Box::new(LoopBoundAffinity::Fixed { dim: 8 }),
        };
        assert!(matches!(lb, LoopBoundAffinity::Nested { .. }));
    }

    #[test]
    fn loop_bound_none() {
        assert!(matches!(LoopBoundAffinity::None, LoopBoundAffinity::None));
    }

    #[test]
    fn loop_bound_clone() {
        let lb = LoopBoundAffinity::Gemm {
            m: DimBinding::Symbolic("M".into()),
            k: DimBinding::Fixed(64),
            n: DimBinding::Symbolic("N".into()),
        };
        let c = lb.clone();
        if let LoopBoundAffinity::Gemm { k, .. } = c { assert_eq!(k, DimBinding::Fixed(64)); }
        else { panic!("expected Gemm"); }
    }

    // ── DimBinding ──

    #[test]
    fn dim_binding_equality() {
        assert_eq!(DimBinding::Fixed(256), DimBinding::Fixed(256));
        assert_eq!(DimBinding::Symbolic("x".into()), DimBinding::Symbolic("x".into()));
    }

    // ── QuantBoundary ──

    #[test]
    fn quant_boundary_none_not_boundary() {
        let qb = QuantBoundary::none();
        assert!(!qb.is_boundary());
        assert_eq!(qb.dequant_flops, 0.0);
    }

    #[test]
    fn quant_boundary_input_is_boundary() {
        assert!(QuantBoundary { input_quant: Some(QuantType::Int8), output_quant: None, dequant_flops: 0.5 }.is_boundary());
    }

    #[test]
    fn quant_boundary_output_is_boundary() {
        assert!(QuantBoundary { input_quant: None, output_quant: Some(QuantType::Bf16), dequant_flops: 0.0 }.is_boundary());
    }

    // ── QuantType ──

    #[test]
    fn quant_type_18_variants_distinct() {
        use std::collections::HashSet;
        let variants: Vec<QuantType> = vec![
            QuantType::Fp32, QuantType::Bf16, QuantType::Fp16,
            QuantType::Fp8E4M3, QuantType::Fp8E5M2, QuantType::Int8,
            QuantType::Int4, QuantType::Q4_0, QuantType::Q4_1,
            QuantType::Q4K, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q5K, QuantType::Q6K, QuantType::Q2K,
            QuantType::Q3K, QuantType::Mxfp4, QuantType::Nvfp4,
        ];
        assert_eq!(variants.len(), 18);
        let set: HashSet<_> = variants.iter().collect();
        assert_eq!(set.len(), 18);
    }

    #[test]
    fn quant_type_copy() {
        let a = QuantType::Nvfp4;
        let b = a;
        assert_eq!(a, b);
    }

    // ── SideEffect ──

    #[test]
    fn side_effect_none_no_barrier() {
        let se = SideEffect::none();
        assert!(!se.writes_kv_cache);
        assert!(!se.writes_output_buffer);
        assert!(!se.writes_shared_mem);
        assert!(!se.reads_shared_state);
        assert_eq!(se.effect_family, EffectFamily::Pure);
        assert!(!se.is_fusion_barrier());
    }

    #[test]
    fn side_effect_shared_mem_no_barrier() {
        assert!(!SideEffect { writes_shared_mem: true, ..SideEffect::none() }.is_fusion_barrier());
    }

    #[test]
    fn side_effect_reads_state_no_barrier() {
        assert!(!SideEffect { reads_shared_state: true, ..SideEffect::none() }.is_fusion_barrier());
    }

    // ── Bottleneck Display & boundaries ──

    #[test]
    fn bottleneck_display() {
        assert_eq!(format!("{}", Bottleneck::MemoryBound), "MemoryBound");
        assert_eq!(format!("{}", Bottleneck::Balanced), "Balanced");
        assert_eq!(format!("{}", Bottleneck::ComputeBound), "ComputeBound");
    }

    #[test]
    fn bottleneck_boundary_values() {
        let base = VmInstrMeta {
            loop_bound: LoopBoundAffinity::None, data_flow: DataFlow::Streaming,
            arithmetic_intensity: 0.0, compute_regs: 1, working_set_bytes: 0,
            quant_boundary: QuantBoundary::none(), side_effect: SideEffect::none(),
        };
        assert_eq!(VmInstrMeta { arithmetic_intensity: 0.99, ..base.clone() }.bottleneck(), Bottleneck::MemoryBound);
        assert_eq!(VmInstrMeta { arithmetic_intensity: 1.0, ..base.clone() }.bottleneck(), Bottleneck::Balanced);
        assert_eq!(VmInstrMeta { arithmetic_intensity: 10.0, ..base.clone() }.bottleneck(), Bottleneck::Balanced);
        assert_eq!(VmInstrMeta { arithmetic_intensity: 10.01, ..base }.bottleneck(), Bottleneck::ComputeBound);
    }

    // ── derive_alignment ──

    #[test]
    fn alignment_scalar_one() { assert_eq!(derive_alignment(32, 0), 1); }
    #[test]
    fn alignment_vector_simd() { assert_eq!(derive_alignment(32, 4), 32); }

    // ── derive_cache_fit ──

    #[test]
    fn cache_fit_l1_boundary() {
        assert_eq!(derive_cache_fit(24576, 32768, 262144), CacheFit::L1);
        assert_eq!(derive_cache_fit(24577, 32768, 262144), CacheFit::L2);
    }

    #[test]
    fn cache_fit_l2_boundary() {
        assert_eq!(derive_cache_fit(262144, 32768, 262144), CacheFit::L2);
        assert_eq!(derive_cache_fit(262145, 32768, 262144), CacheFit::Global);
    }

    // ── FusionStrategy ──

    #[test]
    fn fusion_strategy_boundaries() {
        assert_eq!(derive_fusion_strategy(0.99), FusionStrategy::LoopFusion);
        assert_eq!(derive_fusion_strategy(1.0), FusionStrategy::TileOrComputeRoot);
        assert_eq!(derive_fusion_strategy(10.0), FusionStrategy::TileOrComputeRoot);
        assert_eq!(derive_fusion_strategy(10.01), FusionStrategy::EpilogueInjection);
    }

    // ── can_fuse_data_flow ──

    #[test]
    fn expansive_never_fuses() {
        for c in [DataFlow::Streaming, DataFlow::Reductive, DataFlow::Expansive,
                  DataFlow::InPlace, DataFlow::Accumulative] {
            assert!(!can_fuse_data_flow(DataFlow::Expansive, c));
        }
    }

    #[test]
    fn inplace_only_streaming() {
        assert!(can_fuse_data_flow(DataFlow::InPlace, DataFlow::Streaming));
        assert!(!can_fuse_data_flow(DataFlow::InPlace, DataFlow::Accumulative));
    }

    // ── Factory coverage ──

    #[test]
    fn meta_elem_binop() {
        let m = meta_elem_binop(8192);
        assert_eq!(m.data_flow, DataFlow::Streaming);
        assert_eq!(m.compute_regs, 3);
    }

    #[test]
    fn meta_reduction_props() {
        let m = meta_reduction(4096);
        assert_eq!(m.data_flow, DataFlow::Reductive);
        assert_eq!(m.compute_regs, 2);
    }

    #[test]
    fn meta_kv_write_barrier() {
        let m = meta_kv_write(2048);
        assert!(m.side_effect.writes_kv_cache);
        assert!(m.side_effect.is_fusion_barrier());
    }

    #[test]
    fn meta_vec_load_quant() {
        let m = meta_vec_load(1024, Some(QuantType::Q4K));
        assert!(m.quant_boundary.is_boundary());
        assert_eq!(m.quant_boundary.dequant_flops, 3.0);
    }

    #[test]
    fn meta_vec_load_no_quant() {
        assert!(!meta_vec_load(1024, None).quant_boundary.is_boundary());
    }

    // ── dequant_flops ──

    #[test]
    fn dequant_zero_for_native() {
        for q in [QuantType::Fp32, QuantType::Bf16, QuantType::Fp8E4M3, QuantType::Mxfp4, QuantType::Nvfp4] {
            assert_eq!(dequant_flops_for(q), 0.0);
        }
    }

    #[test]
    fn dequant_positive_for_quantized() {
        for q in [QuantType::Int8, QuantType::Int4, QuantType::Q4_0, QuantType::Q2K] {
            assert!(dequant_flops_for(q) > 0.0);
        }
    }

    // ── CacheFit / FusionStrategy / RegBudget traits ──

    #[test]
    fn cache_fit_eq() {
        assert_eq!(CacheFit::L1, CacheFit::L1);
        assert_ne!(CacheFit::L1, CacheFit::Global);
    }

    #[test]
    fn fusion_strategy_eq() {
        assert_eq!(FusionStrategy::LoopFusion, FusionStrategy::LoopFusion);
        assert_ne!(FusionStrategy::LoopFusion, FusionStrategy::EpilogueInjection);
    }

    #[test]
    fn reg_budget_clone() {
        let rb = RegBudget { need: 5, available: 16 };
        assert_eq!(rb.clone().need, 5);
    }

    #[test]
    fn vm_instr_meta_clone() {
        let m = meta_gemm_microkernel(4, 16, 4, 1024);
        assert_eq!(m.clone().compute_regs, 8);
    }
}
