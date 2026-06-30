//! VmInstr lowering 分类器 (REGISTER-VM SPEC §10 / PLAN-lower-instr-refactor §4)
//!
//! 关注点分离: 与 `vm_instr_meta.rs` 并列但语义不同。
//! - `vm_instr_meta.rs` = fusion 语义元信息 (OpKind/算子族 provenance)。
//! - `vm_instr_category.rs` = lowering 路由分类 (VmInstr → ISA lowering dispatch)。
//!
//! 跨三 ISA (x86_64 / aarch64 / gpu) 共享同一份定义,作为 L0 dispatch 的唯一依据。
//!
//! 分类原则 (ARCH-LOWER-DISPATCH-LAYERING):
//! - `InstrCategory` 把 153 个 VmInstr 变体按 lowering 关注点归为 8 类。
//! - `VmInstr::category()` 必须穷举 match 所有变体,禁止 `_ =>` 通配
//!   (新增 VmInstr 变体时编译器强制补全,防止漏 lowering 导致静默 NOP)。
//! - ISA 不支持的类别 (如 x86 不支持 GpuComm) 由各 ISA 的 L0 dispatch 走 `Err`,
//!   不在本分类器处理 (呼应 NO_SILENT_FALLBACK)。

use super::instr::VmInstr;

/// VmInstr 的 lowering 路由分类。
///
/// L0 dispatch 按 `VmInstr::category()` 将指令路由到对应类别的 `lower_<category>` 方法,
/// arm 体仅允许单表达式委托 (OCP 扩展点)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstrCategory {
    /// 内存 load/store/gather/scatter/atomic/prefetch/memcpy 类。
    /// VecLoad / VecStore / VecNarrow / VecWiden / Mov / Broadcast / LoadPtr /
    /// ScalarLoad / ScalarStore / VecScalarStore / ScalarToIndex / IndexToScalar /
    /// IntMulStride / GatherLoad / ScatterStore / TableLookup / ScalarByteLoad /
    /// MemCopy / AddPtr / StoreConstToStack / MemFence / AtomicAdd / AtomicCAS /
    /// ActivationSwap / Prefetch
    Memory,
    /// 向量/标量算术 + 转换 + reduction + transcendental 类。
    /// VecBinOp / VecShiftImm / VecUnaryOp / VecCmp / VecCast / ConditionalSelect /
    /// Fma / HReduce / Accumulate / Transcendental / DotProduct / ScaleApply /
    /// GprBinOp / GprUnaryOp / GprLoadImm / VecShuffle / VecExtractLane /
    /// VecInsertLane / VecLoadConst
    Arith,
    /// 控制流: 循环/作用域/分支/skip/label 类。
    /// LoopBegin / LoopEnd / ScopeBegin / ScopeEnd / ConditionalSkip /
    /// GprCondAction / IndirectJump / ConditionalExit / BranchIfPtrNonNull /
    /// BranchIfGprZero / BranchIfGprLtU / UnconditionalBranch / BreakLoop / MarkLabel
    Control,
    /// Tile/AMX/Tmem 矩阵单元类 (硬件矩阵加速单元)。
    /// TileConfig / TileLoad / TileMma / TileStore / TileRelease /
    /// TmemAlloc / TmemLoad / TmemStore / TmemDealloc
    Tile,
    /// 量化 load/dequant/decode/bitpack 类。
    /// KiviQuantChannel / KiviQuantToken / KiviDequantLoad / GgufSubScaleLoad /
    /// GgufKQuantScaleLoad / QuantBroadcastInt / QuantScalarCvtLoad / QuantBlockLoad /
    /// QuantBiPlaneLoad / QuantLoadBytesVec / QuantCodebookLookup / QuantExtractBits /
    /// QuantDequantFma / QuantInterleave / QuantConcatSeq / Q3KDecodeStep /
    /// HwQuantDequant / BitwiseGemm / SparseGemm / SparseFp8Gemm / NativeFp4Gemm /
    /// NativeFp8Gemm / SparseMaskIntersect
    Quant,
    /// GPU 专属: warp/shared-mem/async-copy/TMA/cluster/distributed-paging 通信类。
    /// WarpSync / AsyncCopy / AsyncWait / SharedMem* / WeightPrefetch* /
    /// WarpRoleDeclare / WarpBarrier* / TmaDescriptorInit / Tma2DCopy / BarrierInit /
    /// BlockSync / WarpReduce / SharedMemSwizzle / AllReduceChunk / CommBarrier /
    /// NvlinkAsyncCopy / RemotePageLookup / P2pPageFetch / RdmaPageFetch /
    /// RdmaPageFetchCompressed / RemotePageAttn / PageMigrationLock /
    /// PageMigrationUnlock / PageLocationUpdate / ClusterBarrierInit /
    /// ClusterStore / ClusterLoad / PageTableAddr / PageTableKVWrite /
    /// PageTableKVWriteQuant
    GpuComm,
    /// 采样/decode 控制: argmax/softmax/sample/stop-condition/seq-lookup 类。
    /// Argmax / TemperatureScale / StoreToken / CheckStopCondition /
    /// SoftmaxReduceMax / SoftmaxExpSum / SoftmaxNormalize / SampleTopKFilter /
    /// SampleTopPFilter / SampleMultinomial / WarpPRNG / BatchSeqIdLookup /
    /// BatchPerSeqArgmax / BatchPerSeqStopCheck / SeqIdLookup
    Sampling,
    /// 杂项: comment/debug/meta-ops/native-call/decode-helper 类。
    /// Comment / DebugBreakpoint / DebugMarker / DebugProbe / DebugBreakIf /
    /// DeclareVReg / ReleaseVReg / HotpatchSlot / LoadCallbackEntry / NativeCall /
    /// Lz4Decode / BitPackRleDecode
    Misc,
}

impl VmInstr {
    /// 返回该 VmInstr 的 lowering 路由分类。
    ///
    /// 穷举 match 所有 153 个 VmInstr 变体,无 `_ =>` 通配。
    /// 新增 VmInstr 变体时编译器强制补全此 match,防止漏 lowering 路由
    /// 导致静默 NOP (呼应 NO_SILENT_FALLBACK / ARCH-LOWER-DISPATCH-LAYERING §4)。
    pub fn category(&self) -> InstrCategory {
        match self {
            // ── Memory ──
            VmInstr::VecLoad { .. }
            | VmInstr::VecStore { .. }
            | VmInstr::VecNarrow { .. }
            | VmInstr::VecWiden { .. }
            | VmInstr::Mov { .. }
            | VmInstr::Broadcast { .. }
            | VmInstr::LoadPtr { .. }
            | VmInstr::ScalarLoad { .. }
            | VmInstr::ScalarStore { .. }
            | VmInstr::VecScalarStore { .. }
            | VmInstr::ScalarToIndex { .. }
            | VmInstr::IndexToScalar { .. }
            | VmInstr::IntMulStride { .. }
            | VmInstr::GatherLoad { .. }
            | VmInstr::ScatterStore { .. }
            | VmInstr::TableLookup { .. }
            | VmInstr::ScalarByteLoad { .. }
            | VmInstr::MemCopy { .. }
            | VmInstr::AddPtr { .. }
            | VmInstr::StoreConstToStack { .. }
            | VmInstr::MemFence { .. }
            | VmInstr::AtomicAdd { .. }
            | VmInstr::AtomicCAS { .. }
            | VmInstr::ActivationSwap { .. }
            | VmInstr::Prefetch { .. } => InstrCategory::Memory,

            // ── Arith ──
            VmInstr::VecBinOp { .. }
            | VmInstr::VecShiftImm { .. }
            | VmInstr::VecUnaryOp { .. }
            | VmInstr::VecCmp { .. }
            | VmInstr::VecCast { .. }
            | VmInstr::ConditionalSelect { .. }
            | VmInstr::Fma { .. }
            | VmInstr::HReduce { .. }
            | VmInstr::Accumulate { .. }
            | VmInstr::Transcendental { .. }
            | VmInstr::DotProduct { .. }
            | VmInstr::ScaleApply { .. }
            | VmInstr::GprBinOp { .. }
            | VmInstr::GprUnaryOp { .. }
            | VmInstr::GprLoadImm { .. }
            | VmInstr::VecShuffle { .. }
            | VmInstr::VecExtractLane { .. }
            | VmInstr::VecInsertLane { .. }
            | VmInstr::VecLoadConst { .. } => InstrCategory::Arith,

            // ── Control ──
            VmInstr::LoopBegin { .. }
            | VmInstr::LoopEnd
            | VmInstr::ScopeBegin { .. }
            | VmInstr::ScopeEnd { .. }
            | VmInstr::ConditionalSkip { .. }
            | VmInstr::GprCondAction { .. }
            | VmInstr::IndirectJump { .. }
            | VmInstr::ConditionalExit { .. }
            | VmInstr::BranchIfPtrNonNull { .. }
            | VmInstr::BranchIfGprZero { .. }
            | VmInstr::BranchIfGprLtU { .. }
            | VmInstr::UnconditionalBranch { .. }
            | VmInstr::BreakLoop { .. }
            | VmInstr::MarkLabel { .. } => InstrCategory::Control,

            // ── Tile ──
            VmInstr::TileConfig { .. }
            | VmInstr::TileLoad { .. }
            | VmInstr::TileMma { .. }
            | VmInstr::TileStore { .. }
            | VmInstr::TileRelease
            | VmInstr::TmemAlloc { .. }
            | VmInstr::TmemLoad { .. }
            | VmInstr::TmemStore { .. }
            | VmInstr::TmemDealloc { .. } => InstrCategory::Tile,

            // ── Quant ──
            VmInstr::KiviQuantChannel { .. }
            | VmInstr::KiviQuantToken { .. }
            | VmInstr::KiviDequantLoad { .. }
            | VmInstr::GgufSubScaleLoad { .. }
            | VmInstr::GgufKQuantScaleLoad { .. }
            | VmInstr::QuantBroadcastInt { .. }
            | VmInstr::QuantScalarCvtLoad { .. }
            | VmInstr::QuantBlockLoad { .. }
            | VmInstr::QuantBiPlaneLoad { .. }
            | VmInstr::QuantLoadBytesVec { .. }
            | VmInstr::QuantCodebookLookup { .. }
            | VmInstr::QuantExtractBits { .. }
            | VmInstr::QuantDequantFma { .. }
            | VmInstr::QuantInterleave { .. }
            | VmInstr::QuantConcatSeq { .. }
            | VmInstr::Q3KDecodeStep { .. }
            | VmInstr::HwQuantDequant { .. }
            | VmInstr::BitwiseGemm { .. }
            | VmInstr::SparseGemm { .. }
            | VmInstr::SparseFp8Gemm { .. }
            | VmInstr::NativeFp4Gemm { .. }
            | VmInstr::NativeFp8Gemm { .. }
            | VmInstr::SparseMaskIntersect { .. } => InstrCategory::Quant,

            // ── GpuComm (无条件部分) ──
            VmInstr::WarpSync
            | VmInstr::AsyncCopy { .. }
            | VmInstr::AsyncWait { .. }
            | VmInstr::SharedMemAlloc { .. }
            | VmInstr::SharedMemStore { .. }
            | VmInstr::SharedMemLoad { .. }
            | VmInstr::SharedMemAsyncStore { .. }
            | VmInstr::SharedMemAsyncWaitGroup { .. }
            | VmInstr::WeightPrefetchAsync { .. }
            | VmInstr::WeightPrefetchWait { .. }
            | VmInstr::WarpRoleDeclare { .. }
            | VmInstr::WarpBarrierArrive { .. }
            | VmInstr::WarpBarrierWait { .. }
            | VmInstr::TmaDescriptorInit { .. }
            | VmInstr::Tma2DCopy { .. }
            | VmInstr::BarrierInit { .. }
            | VmInstr::BlockSync
            | VmInstr::WarpReduce { .. }
            | VmInstr::SharedMemSwizzle { .. }
            | VmInstr::ClusterBarrierInit { .. }
            | VmInstr::ClusterStore { .. }
            | VmInstr::ClusterLoad { .. }
            | VmInstr::PageTableAddr { .. }
            | VmInstr::PageTableKVWrite { .. }
            | VmInstr::PageTableKVWriteQuant { .. } => InstrCategory::GpuComm,

            // ── GpuComm (nccl feature-gated 分布式通信变体) ──
            #[cfg(feature = "nccl")]
            VmInstr::AllReduceChunk { .. }
            | VmInstr::CommBarrier { .. }
            | VmInstr::NvlinkAsyncCopy { .. }
            | VmInstr::RemotePageLookup { .. }
            | VmInstr::P2pPageFetch { .. }
            | VmInstr::RdmaPageFetch { .. }
            | VmInstr::RdmaPageFetchCompressed { .. }
            | VmInstr::RemotePageAttn { .. }
            | VmInstr::PageMigrationLock { .. }
            | VmInstr::PageMigrationUnlock { .. }
            | VmInstr::PageLocationUpdate { .. } => InstrCategory::GpuComm,

            // ── Sampling ──
            VmInstr::Argmax { .. }
            | VmInstr::TemperatureScale { .. }
            | VmInstr::StoreToken { .. }
            | VmInstr::CheckStopCondition { .. }
            | VmInstr::SoftmaxReduceMax { .. }
            | VmInstr::SoftmaxExpSum { .. }
            | VmInstr::SoftmaxNormalize { .. }
            | VmInstr::SampleTopKFilter { .. }
            | VmInstr::SampleTopPFilter { .. }
            | VmInstr::SampleMultinomial { .. }
            | VmInstr::WarpPRNG { .. }
            | VmInstr::BatchSeqIdLookup { .. }
            | VmInstr::BatchPerSeqArgmax { .. }
            | VmInstr::BatchPerSeqStopCheck { .. }
            | VmInstr::SeqIdLookup { .. } => InstrCategory::Sampling,

            // ── Misc ──
            VmInstr::Comment(_)
            | VmInstr::DebugBreakpoint { .. }
            | VmInstr::DebugMarker { .. }
            | VmInstr::DebugProbe { .. }
            | VmInstr::DebugBreakIf { .. }
            | VmInstr::DeclareVReg { .. }
            | VmInstr::ReleaseVReg { .. }
            | VmInstr::HotpatchSlot { .. }
            | VmInstr::LoadCallbackEntry { .. }
            | VmInstr::NativeCall { .. }
            | VmInstr::Lz4Decode { .. }
            | VmInstr::BitPackRleDecode { .. } => InstrCategory::Misc,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;

    /// 穷举验证: 每个 VmInstr 变体必须命中且仅命中一个类别 (ARCH-LOWER-DISPATCH-LAYERING §4)。
    /// 此测试保证 category() 无 `_ =>` 通配,新增变体时若忘记补全会编译失败。
    #[test]
    fn test_category_exhaustive_no_wildcard() {
        // 只验证分类器对代表性变体的返回值正确,完整穷举由编译器 match 穷尽性保证。
        assert_eq!(VmInstr::LoopEnd.category(), InstrCategory::Control);
        assert_eq!(VmInstr::TileRelease.category(), InstrCategory::Tile);
        assert_eq!(VmInstr::BlockSync.category(), InstrCategory::GpuComm);
        assert_eq!(VmInstr::Comment("x".into()).category(), InstrCategory::Misc);
        assert_eq!(
            VmInstr::MarkLabel { label_id: 0 }.category(),
            InstrCategory::Control
        );
    }
}
