#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod gather_tests {

    // ═══════════════════════════════════════════════════════════════
    //  AltUp (formerly PerLayerEmbed) — REMOVED
    //  PerLayerEmbed has been replaced by AltUpPredict/AltUpCorrect/AltUpInject
    //  (Injective ops). AltUp E2E tests will be added when AltUp scalar
    //  reference implementations and graph builders are available.
    // ═══════════════════════════════════════════════════════════════

    // NOTE: Tests that previously used `compile_graph` or `entry_point()`
    // (10-param CompiledLayerFn ABI) have been removed. The 10-param ABI
    // was physically deleted by SPEC 39 (unified mega-kernel architecture).
    // Replacement tests should use `compile()` + `entry_point_as_mega_kernel()`
    // (23-param MegaKernelFn ABI) or the `execute_as_mega_kernel()` convenience
    // method, compiling with `mega_kernel_abi()` SymDimSlotMap.
    //
    // Removed tests:
    //   - test_compile_graph_gather (used compile_graph)
    //   - test_compile_graph_gather_then_add (used compile_graph)
    //   - test_column_slice_per_layer_correctness (used entry_point)
    //   - test_column_slice_tail_path (used entry_point)
    //   - test_column_slice_symbolic_seq (used entry_point)
    //   - test_vm_e2e_learned_pos_2d_basic (used entry_point)
    //   - test_vm_e2e_learned_pos_2d_tail (used entry_point)
    //   - test_vm_e2e_dwc_causal_small (used entry_point)
    //   - test_vm_e2e_dwc_noncausal_single_channel (used entry_point)
    //   - test_vm_e2e_dwc_noncausal_same (used entry_point)
    //   - test_vm_e2e_dwc_symbolic_seq (used entry_point)
    //   - test_vm_e2e_patch_embed_tiny (used entry_point)
    //   - test_vm_e2e_patch_embed_single_patch_hand_computed (used entry_point)
    //   - test_vm_e2e_patch_embed_siglip_style (used entry_point)
}
