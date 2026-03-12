//! x86_64 JIT code generation via iced-x86 CodeAssembler.
//!
//! Phase 3 of the JIT compiler pipeline: generates native x86_64 machine code
//! from FusionPlan + CompilerGraph + BufferAllocation.
//!
//! MVP implemented under the `jit-x86` feature flag (`jit::X86CodeGen`):
//! - TraceOp → AVX2 SIMD instruction mapping (elementwise kernel generation)
//! - GEMM microkernel generation (3-level blocked loop with nop body markers)
//! - Fused elementwise loops (nop markers per fused op; real SIMD loop is next)
//! - GEMM epilogue injection (nop markers; real epilogue on accumulators is next)
//! - Full `emit_plan()` pipeline: prologue → groups → epilogue → assemble
//!
//! Without the feature flag, only `emit_stub()` is available (minimal valid function).

use super::CodegenOutput;

/// Emit a minimal x86_64 stub (push rbp; mov rbp,rsp; nops; pop rbp; ret).
///
/// Used as fallback until the real Phase 3 code generator is implemented.
pub fn emit_stub() -> CodegenOutput {
    let mut code = Vec::with_capacity(16);
    code.push(0x55);                          // push rbp
    code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
    for _ in 0..8 {
        code.push(0x90);                      // nop
    }
    code.push(0x5D);                          // pop rbp
    code.push(0xC3);                          // ret
    CodegenOutput { code, scratchpad_bytes: 0 }
}

#[cfg(feature = "jit-x86")]
pub mod jit {
    use iced_x86::code_asm::*;
    use crate::compiler::trace::{TraceOp, ComputePattern};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::graph::{CompilerGraph, OpKind, OpId, WeightLayout, PtrSource, GroupPointerMap};
    use crate::compiler::registry::ScalarOpRegistry;
    use crate::compiler::buffer_alloc::{BufferAllocation, BufferSlot};
    use crate::dispatch::DeviceProfile;
    use super::CodegenOutput;
    use crate::compiler::codegen::simd_ops::{SimdOps, VReg, MemOperand, BaseReg, Label};

    /// x86_64 JIT code generator.
    ///
    /// Translates FusionPlan groups into native AVX2/AVX-512 machine code
    /// via iced-x86's `CodeAssembler`.
    pub struct X86CodeGen {
        asm: CodeAssembler,
        use_avx512: bool,
        /// Number of f32 elements per SIMD register (8 for AVX2, 16 for AVX-512).
        simd_width: usize,
        /// Constant pool: f32 values to be emitted after code section.
        /// Each entry is an 8×f32 (256-bit) broadcast-ready constant.
        const_pool: Vec<[f32; 8]>,
        /// Labels for each constant pool entry (RIP-relative addressing).
        const_labels: Vec<CodeLabel>,
        /// Extra scratchpad bytes needed by BLIS packing buffers.
        blis_scratchpad_bytes: usize,
        /// Offset within scratchpad where BLIS packing buffers start.
        blis_scratchpad_offset: usize,
        /// Initial value of blis_scratchpad_offset (= alloc.total_bytes).
        blis_base_offset: usize,
        /// Whether rdx (external tensor pointer) was saved to [rbp-48] for
        /// epilogue Input(1) access (e.g. bias vector in GEMM+AddBias).
        bias_saved: bool,
        /// JIT tuning parameters (set via `set_jit_params`).
        jit_params: Option<crate::autotuning::search_space::JitParams>,
        /// Weight blob layout for multi-weight graphs (BERT etc.)
        weight_layout: Option<WeightLayout>,
        /// Labels allocated by SimdOps::alloc_label / define_label / jump.
        simd_labels: Vec<CodeLabel>,
        /// Width stack for push_width/pop_width mixed-width emission.
        width_stack: Vec<crate::compiler::codegen::simd_ops::SimdWidth>,
        /// Whether the target has AVX-512 capability (immutable hardware flag).
        /// Unlike `use_avx512` which tracks the *current* emission width,
        /// this records the hardware capability for push_width(W512) validation.
        has_avx512: bool,
        /// Whether the target has AMX tile instructions (SPR+).
        has_amx: bool,
    }

    /// How `Input(idx)` nodes are materialised in trace evaluation.
    enum InputMode {
        /// All inputs map to the accumulator register.
        AccOnly,
        /// `Input(0)` = accumulator, `Input(1)` = `[rax + bias_disp]`.
        WithBias { bias_disp: i32 },
        /// `Input(0)` = accumulator, `Input(1)` = `[rsi + rbx]`.
        /// When `scalar_tail`, uses masked/scalar load for the second input.
        Elementwise { is_binary: bool, scalar_tail: bool },
        /// Reduction combine: Input(0)=accumulator ymm/zmm, Input(1)=current vector from memory [rdi+rbx].
        ReductionCombine,
        /// Dual-input: Input(0)=load from [rdi+rbx], Input(1)=broadcast register (ymm15/zmm31).
        WithBroadcast,
        /// NormLike transform: Input(0)=load [rdi+rbx], Input(1)=broadcast ymm14/zmm30 (mean or scale),
        /// Input(2)=broadcast ymm15/zmm31 (scale), Input(3)=load [rdx+rbx] (weight), Input(4)=load [rcx+rbx] (bias).
        NormTransform {
            has_mean: bool,
            has_weight: bool,
            has_bias: bool,
        },
    }

    /// Extract ymm register index (0-15) from an AsmRegisterYmm.
    fn ymm_index(reg: AsmRegisterYmm) -> u8 {
        let all = [
            ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
            ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
        ];
        for (i, r) in all.iter().enumerate() {
            if *r == reg { return i as u8; }
        }
        panic!("unknown ymm register")
    }

    /// Extract zmm register index (0-31) from an AsmRegisterZmm.
    fn zmm_index(reg: AsmRegisterZmm) -> u8 {
        let all: [AsmRegisterZmm; 32] = [
            zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7,
            zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15,
            zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23,
            zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31,
        ];
        for (i, r) in all.iter().enumerate() {
            if *r == reg { return i as u8; }
        }
        panic!("unknown zmm register")
    }

        /// Macro: generates trace-on-stack evaluator body for both AVX2 (ymm) and AVX-512 (zmm).
    ///
    /// The two ISA paths are structurally identical — they differ only in register
    /// width, scratch register numbers, a handful of ISA-specific instructions, and
    /// the scalar-tail load strategy.  This macro captures the shared logic once.
    macro_rules! trace_on_stack_body {
        (
            $self_:expr, $acc:expr, $body:expr, $mode:expr,
            simd_bytes: $simd_bytes:expr,
            ptr_fn: $ptr_fn:ident,
            s0: $s0:expr, s1: $s1:expr,
            math_s0: $math_s0:expr, math_s1: $math_s1:expr,
            broadcast_reg: $broadcast_reg:expr,
            mean_reg: $mean_reg:expr,
            scale_reg: $scale_reg:expr,
            zero_instr: $zero_instr:ident,
            rcp_instr: $rcp_instr:ident,
            rsqrt_instr: $rsqrt_instr:ident,
            emit_exp: $emit_exp:ident,
            emit_tanh: $emit_tanh:ident,
            emit_log: $emit_log:ident,
            scalar_tail_load: $scalar_tail_load:expr
        ) => {{
            let n = $body.len();
            if n == 0 {
                return Ok(());
            }

            let frame_size = (n as i32) * $simd_bytes;
            $self_.asm.sub(rsp, frame_size).map_err(|e| e.to_string())?;

            let slot_off = |i: u32| -> i32 { (i as i32) * $simd_bytes };

            for (i, op) in $body.iter().enumerate() {
                let i32_idx = i as u32;
                match op {
                    TraceOp::Input(idx) => {
                        match &$mode {
                            InputMode::AccOnly => {
                                $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $acc)
                                    .map_err(|e| e.to_string())?;
                            }
                            InputMode::WithBias { bias_disp } => {
                                if *idx == 0 {
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $acc)
                                        .map_err(|e| e.to_string())?;
                                } else {
                                    $self_.asm.vmovups($s0, $ptr_fn(rax + *bias_disp))
                                        .map_err(|e| e.to_string())?;
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                            InputMode::Elementwise { is_binary, scalar_tail } => {
                                if *idx == 0 {
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $acc)
                                        .map_err(|e| e.to_string())?;
                                } else if *idx == 1 && *is_binary {
                                    if *scalar_tail {
                                        #[allow(clippy::redundant_closure_call)]
                                        ($scalar_tail_load)(&mut $self_.asm)?;
                                    } else {
                                        $self_.asm.vmovups($s0, $ptr_fn(rsi + rbx))
                                            .map_err(|e| e.to_string())?;
                                    }
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                                        .map_err(|e| e.to_string())?;
                                } else {
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $acc)
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                            InputMode::ReductionCombine => {
                                if *idx == 0 {
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $acc)
                                        .map_err(|e| e.to_string())?;
                                } else {
                                    $self_.asm.vmovups($s0, $ptr_fn(rdi + rbx))
                                        .map_err(|e| e.to_string())?;
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                            InputMode::WithBroadcast => {
                                if *idx == 0 {
                                    $self_.asm.vmovups($s0, $ptr_fn(rdi + rbx))
                                        .map_err(|e| e.to_string())?;
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                                        .map_err(|e| e.to_string())?;
                                } else {
                                    $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $broadcast_reg)
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                            InputMode::NormTransform { has_mean, has_weight, has_bias } => {
                                match *idx {
                                    0 => {
                                        $self_.asm.vmovups($s0, $ptr_fn(rdi + rbx))
                                            .map_err(|e| e.to_string())?;
                                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    1 => {
                                        if *has_mean {
                                            $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $mean_reg)
                                                .map_err(|e| e.to_string())?;
                                        } else {
                                            // RmsNorm: Input(1) = scale
                                            $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $scale_reg)
                                                .map_err(|e| e.to_string())?;
                                        }
                                    }
                                    2 => {
                                        // scale — only used when has_mean (LayerNorm)
                                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $scale_reg)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    3 => {
                                        // weight from [rdx+rbx]
                                        if *has_weight {
                                            $self_.asm.vmovups($s0, $ptr_fn(rdx + rbx))
                                                .map_err(|e| e.to_string())?;
                                        } else {
                                            let one = $self_.const_f32(1.0);
                                            $self_.asm.vbroadcastss($s0, dword_ptr(one))
                                                .map_err(|e| e.to_string())?;
                                        }
                                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    4 => {
                                        // bias from [rcx+rbx]
                                        if *has_bias {
                                            $self_.asm.vmovups($s0, $ptr_fn(rcx + rbx))
                                                .map_err(|e| e.to_string())?;
                                        } else {
                                            $self_.asm.$zero_instr($s0, $s0, $s0)
                                                .map_err(|e| e.to_string())?;
                                        }
                                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    _ => {
                                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $acc)
                                            .map_err(|e| e.to_string())?;
                                    }
                                }
                            }
                        }
                    }
                    TraceOp::Const(v) => {
                        let label = $self_.const_f32(*v as f32);
                        $self_.asm.vbroadcastss($s0, dword_ptr(label))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Add(a, b) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vaddps($s0, $s0, $ptr_fn(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sub(a, b) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vsubps($s0, $s0, $ptr_fn(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Mul(a, b) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmulps($s0, $s0, $ptr_fn(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Div(a, b) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vdivps($s0, $s0, $ptr_fn(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Fma(a, b, c) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*c)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($s1, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vfmadd231ps($s0, $s1, $ptr_fn(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Neg(a) => {
                        $self_.asm.$zero_instr($s0, $s0, $s0)
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vsubps($s0, $s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Abs(a) => {
                        let abs_mask = $self_.const_f32(f32::from_bits(0x7FFF_FFFF));
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vbroadcastss($s1, dword_ptr(abs_mask))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vandps($s0, $s0, $s1)
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Exp(a) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.$emit_exp($s1, $s0, [$s0, $math_s0, $math_s1])?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s1)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Sqrt(a) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vsqrtps($s0, $s0)
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Rsqrt(a) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.$rsqrt_instr($s0, $s0)
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Tanh(a) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.$emit_tanh($s1, $s0, [$s0, $math_s0, $math_s1])?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s1)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Recip(a) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.$rcp_instr($s0, $s0)
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Log(a) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.$emit_log($s1, $s0, [$s0, $math_s0, $math_s1])?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s1)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Max(a, b) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmaxps($s0, $s0, $ptr_fn(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                    TraceOp::Min(a, b) => {
                        $self_.asm.vmovups($s0, $ptr_fn(rsp + slot_off(*a)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vminps($s0, $s0, $ptr_fn(rsp + slot_off(*b)))
                            .map_err(|e| e.to_string())?;
                        $self_.asm.vmovups($ptr_fn(rsp + slot_off(i32_idx)), $s0)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            $self_.asm.vmovups($acc, $ptr_fn(rsp + slot_off((n - 1) as u32)))
                .map_err(|e| e.to_string())?;

            $self_.asm.add(rsp, frame_size).map_err(|e| e.to_string())?;

            Ok(())
        }};
    }

    impl X86CodeGen {
        /// Create a new code generator configured for the detected hardware.
        pub fn new(profile: &DeviceProfile) -> Self {
            let use_avx512 = profile.kernel_config.use_avx512;
            X86CodeGen {
                asm: CodeAssembler::new(64).unwrap(),
                use_avx512,
                simd_width: if use_avx512 { 16 } else { 8 },
                const_pool: Vec::new(),
                const_labels: Vec::new(),
                blis_scratchpad_bytes: 0,
                blis_scratchpad_offset: 0,
                blis_base_offset: 0,
                bias_saved: false,
                jit_params: None,
                weight_layout: None,
                simd_labels: Vec::new(),
                width_stack: Vec::new(),
                has_avx512: use_avx512,
                has_amx: profile.kernel_config.has_amx,
            }
        }

        /// Set the weight blob layout for multi-weight graph compilation.
        pub fn set_weight_layout(&mut self, layout: WeightLayout) {
            self.weight_layout = Some(layout);
        }

        /// Number of f32 elements per SIMD register (8 for AVX2, 16 for AVX-512).
        pub fn simd_width(&self) -> usize {
            self.simd_width
        }

        /// Build HwCapabilityMatrix from the current codegen state.
        pub fn hw_capability_matrix(&self) -> crate::compiler::codegen::isa_scheduler::HwCapabilityMatrix {
            use crate::compiler::codegen::simd_ops::SimdWidth;
            use crate::compiler::codegen::tile_ops::TileAccelKind;
            let mut widths = vec![SimdWidth::W128, SimdWidth::W256];
            if self.use_avx512 { widths.push(SimdWidth::W512); }
            crate::compiler::codegen::isa_scheduler::HwCapabilityMatrix {
                simd_widths: widths,
                has_tile_accel: self.has_amx,
                tile_accel: if self.has_amx { Some(TileAccelKind::Amx) } else { None },
                peak_gflops_f32: 0.0,
                peak_bandwidth_gbs: 0.0,
                roofline_crossover: 0.0,
            }
        }

        /// Set JIT tuning parameters for code generation.
        ///
        /// These parameters control K-loop unrolling, prefetch distance,
        /// register allocation strategy, software pipelining, and NR tile variant.
        /// When set, `emit_gemm_microkernel` and `emit_standalone_gemm` use
        /// these values instead of the defaults.
        pub fn set_jit_params(&mut self, params: &crate::autotuning::search_space::JitParams) {
            self.jit_params = Some(params.clone());
        }

        /// K-loop unroll factor from JIT params, or default 1.
        fn k_unroll_factor(&self) -> usize {
            self.jit_params.as_ref().map(|p| p.k_unroll).unwrap_or(1)
        }

        /// Prefetch distance from JIT params, or default 0 (disabled).
        fn prefetch_distance(&self) -> usize {
            self.jit_params.as_ref().map(|p| p.prefetch_distance).unwrap_or(0)
        }

        /// Emit a standalone GEMM function for autotuning measurement.
        ///
        /// Generates a complete function with ABI:
        ///   rdi = A ptr (*const f32, row-major [m, k])
        ///   rsi = B ptr (*const f32, row-major [k, n])
        ///   rdx = C ptr (*mut f32, row-major [m, n])
        ///   rcx = scratchpad ptr (*mut u8)
        ///
        /// The function includes prologue, GEMM microkernel with the specified
        /// blocking parameters, and epilogue. Returns assembled machine code bytes.
        pub fn emit_standalone_gemm(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            blocking: &crate::dispatch::device_profile::GemmBlocking,
            profile: &DeviceProfile,
        ) -> Result<Vec<u8>, String> {
            // Prologue: save callee-saved registers
            self.emit_prologue()?;

            // Save scratchpad ptr (rcx) to rbp+32 area for BLIS packing
            // In standalone ABI: rdi=A, rsi=B, rdx=C, rcx=scratchpad
            // We need to remap to match the internal GEMM ABI where
            // rbp+32 holds the scratchpad pointer.
            self.asm.sub(rsp, 16i32).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rbp + 32), rcx).map_err(|e| e.to_string())?;

            // The internal GEMM emitter expects: rdi=A, rsi=B, rdx=C
            // which matches our standalone ABI (rdx=C, not scratchpad).
            // But we need to save rdx (C ptr) and use it for output.
            self.asm.mov(r15, rdx).map_err(|e| e.to_string())?;

            self.blis_scratchpad_offset = 0;
            self.blis_base_offset = 0;

            let mr = blocking.mr;
            let nr = blocking.nr;
            let simd_w = self.simd_width;
            let nr_vecs = nr / simd_w;
            let kc = blocking.kc;
            let mc = blocking.mc;
            let nc = blocking.nc;

            // Restore rdx = C ptr for the microkernel
            self.asm.mov(rdx, r15).map_err(|e| e.to_string())?;

            if m == 1 {
                self.emit_gemm_microkernel_direct(1, n, k, 1, nr, nr_vecs, simd_w, &[])?;
            } else if k <= kc && m <= mc && n <= nc {
                self.emit_gemm_microkernel_direct(m, n, k, mr, nr, nr_vecs, simd_w, &[])?;
            } else {
                self.emit_gemm_blis(m, n, k, mr, nr, nr_vecs, kc, mc, nc, &[])?;
            }

            self.asm.add(rsp, 16i32).map_err(|e| e.to_string())?;

            // Epilogue: restore callee-saved registers and return
            self.emit_epilogue()?;

            // Emit constant pool
            self.emit_const_pool()?;

            // Assemble
            let code = self.asm.assemble(0x0)
                .map_err(|e| format!("asm error: {e}"))?;

            Ok(code)
        }


        /// Deduplicates: returns existing label if value already present.
        fn const_f32(&mut self, val: f32) -> CodeLabel {
            let bits = val.to_bits();
            for (i, entry) in self.const_pool.iter().enumerate() {
                if entry[0].to_bits() == bits {
                    return self.const_labels[i];
                }
            }
            let label = self.asm.create_label();
            self.const_pool.push([val; 8]);
            self.const_labels.push(label);
            label
        }

        /// Emit the constant pool data after all code.
        /// Must be called before `assemble()`.
        fn emit_const_pool(&mut self) -> Result<(), String> {
            for i in 0..self.const_pool.len() {
                self.asm.set_label(&mut self.const_labels[i]).map_err(|e| e.to_string())?;
                for chunk in self.const_pool[i].chunks(2) {
                    self.asm.dd_f32(chunk).map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }

        /// Generate code for a complete fusion plan.
        ///
        /// Iterates over all fusion groups and emits the appropriate code
        /// for each pattern (standalone, elementwise chain, GEMM+epilogue).
        pub fn emit_plan(
            &mut self,
            plan: &FusionPlan,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<CodegenOutput, String> {
            self.emit_prologue()?;
            // When weight layout is present (multi-weight graph), save ABI
            // register values to callee-saved registers for use across groups.
            // r14 = weights blob base, r13 = activation input base
            if self.weight_layout.is_some() {
                self.asm.mov(r14, rsi).map_err(|e| e.to_string())?;  // r14 = weights blob base
                self.asm.mov(r13, rdi).map_err(|e| e.to_string())?;  // r13 = activation input base
            }

            self.blis_scratchpad_offset = alloc.total_bytes;
            self.blis_base_offset = alloc.total_bytes;

            eprintln!("[JIT-DBG] emit_plan: {} groups, weight_layout={}", plan.groups.len(), self.weight_layout.is_some());
            for group in &plan.groups {
                self.emit_group_pointer_setup(group, graph, alloc)?;
                // Save r13/r14 (activation input base / weights blob base) across
                // each group, since emit_group may clobber them (e.g. BLIS loops,
                // NormIntoGemm, TileLevelFusion all reuse r13/r14 internally).
                if self.weight_layout.is_some() {
                    self.asm.push(r13).map_err(|e| e.to_string())?;
                    self.asm.push(r14).map_err(|e| e.to_string())?;
                }
                self.emit_group(group, graph, alloc, profile, registry)?;
                if self.weight_layout.is_some() {
                    self.asm.pop(r14).map_err(|e| e.to_string())?;
                    self.asm.pop(r13).map_err(|e| e.to_string())?;
                }
            }

            self.emit_epilogue()?;

            self.emit_const_pool()?;

            let code = self.asm.assemble(0x0)
                .map_err(|e| format!("asm error: {e}"))?;

            eprintln!("[JIT-DBG] code_size={}, scratchpad_bytes={} (alloc={} + blis={})",
                code.len(), alloc.total_bytes + self.blis_scratchpad_bytes,
                alloc.total_bytes, self.blis_scratchpad_bytes);
            if let Ok(()) = std::fs::write(&format!("/tmp/gllm_jit_code_{}.bin", plan.groups.len() * 1000 + alloc.total_bytes), &code) {
                eprintln!("[JIT-DBG] dumped {} bytes to /tmp/gllm_jit_code.bin", code.len());
            }

            Ok(CodegenOutput {
                code,
                scratchpad_bytes: alloc.total_bytes + self.blis_scratchpad_bytes,
            })
        }

        /// System V AMD64 ABI prologue: save frame pointer + callee-saved registers.
        fn emit_prologue(&mut self) -> Result<(), String> {
            self.asm.push(rbp).map_err(|e| e.to_string())?;
            self.asm.mov(rbp, rsp).map_err(|e| e.to_string())?;
            self.asm.push(rbx).map_err(|e| e.to_string())?;
            self.asm.push(r12).map_err(|e| e.to_string())?;
            self.asm.push(r13).map_err(|e| e.to_string())?;
            self.asm.push(r14).map_err(|e| e.to_string())?;
            self.asm.push(r15).map_err(|e| e.to_string())?;
            self.asm.sub(rsp, 8i32).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Restore callee-saved registers and return.
        fn emit_epilogue(&mut self) -> Result<(), String> {
            self.asm.add(rsp, 8i32).map_err(|e| e.to_string())?;
            self.asm.pop(r15).map_err(|e| e.to_string())?;
            self.asm.pop(r14).map_err(|e| e.to_string())?;
            self.asm.pop(r13).map_err(|e| e.to_string())?;
            self.asm.pop(r12).map_err(|e| e.to_string())?;
            self.asm.pop(rbx).map_err(|e| e.to_string())?;
            self.asm.pop(rbp).map_err(|e| e.to_string())?;
            self.asm.ret().map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit x86_64 register setup for a fusion group's pointer bindings.
        ///
        /// When a weight layout is present (multi-weight graphs like BERT),
        /// this sets up rdi/rsi/rdx/r8 to point to the correct locations
        /// within the weight blob, scratchpad, or graph I/O buffers.
        ///
        /// Register convention (System V AMD64 ABI for CompiledLayerFn):
        ///   rdi = input (activation), rsi = weights blob base
        ///   r8  = output ptr, [rbp+32] = scratchpad ptr
        ///
        /// After pointer setup:
        ///   rdi = A ptr (activation or scratchpad intermediate)
        ///   rsi = B ptr (weight blob + offset OR scratchpad intermediate)
        ///   rdx = bias ptr (weight blob + offset, for GemmBias)
        ///   r8  = output ptr (graph output or scratchpad intermediate)
        fn emit_group_pointer_setup(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
        ) -> Result<(), String> {
            let wl = match &self.weight_layout {
                Some(wl) => wl,
                None => return Ok(()), // Single-weight graph, ABI ptrs are already correct
            };

            let map = Self::compute_group_pointer_map(group, graph, wl, alloc);
            eprintln!("[JIT-DBG] group {} anchor={:?} mode={:?}: a={:?} b={:?} bias={:?} out={:?} norm_w={:?} norm_b={:?}",
                group.id, group.anchor, group.mode,
                map.a_ptr, map.b_ptr, map.bias_ptr, map.output_ptr,
                map.norm_weight_ptr, map.norm_bias_ptr);

            // Set up A ptr (rdi)
            match &map.a_ptr {
                Some(PtrSource::ActivationInput) => {
                    // Restore activation input from callee-saved r13
                    self.asm.mov(rdi, r13).map_err(|e| e.to_string())?;
                }
                Some(PtrSource::Scratchpad(off)) => {
                    // Load scratchpad base and add offset
                    self.asm.mov(rdi, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(rdi, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                _ => {}
            }

            // Set up B ptr (rsi) — handles BOTH weight blob and scratchpad intermediates
            match &map.b_ptr {
                Some(PtrSource::WeightBlob(off)) => {
                    self.asm.mov(rsi, r14).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(rsi, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                Some(PtrSource::Scratchpad(off)) => {
                    // B matrix is an intermediate result in the scratchpad
                    self.asm.mov(rsi, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(rsi, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                _ => {}
            }

            // Set up bias ptr (rdx) for GemmBias
            match &map.bias_ptr {
                Some(PtrSource::WeightBlob(off)) => {
                    self.asm.mov(rdx, r14).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(rdx, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                Some(PtrSource::Scratchpad(off)) => {
                    self.asm.mov(rdx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(rdx, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                _ => {}
            }

            // Set up norm weight ptr (rdx — matches NormTransform trace convention)
            match &map.norm_weight_ptr {
                Some(PtrSource::WeightBlob(off)) => {
                    self.asm.mov(rdx, r14).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(rdx, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                _ => {}
            }

            // Set up norm bias ptr (rcx for LayerNorm)
            match &map.norm_bias_ptr {
                Some(PtrSource::WeightBlob(off)) => {
                    self.asm.mov(rcx, r14).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(rcx, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                _ => {}
            }

            // Set up output ptr (r8)
            match &map.output_ptr {
                Some(PtrSource::GraphOutput) => {
                    self.asm.mov(r8, qword_ptr(rbp + 24)).map_err(|e| e.to_string())?;
                }
                Some(PtrSource::Scratchpad(off)) => {
                    self.asm.mov(r8, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;
                    if *off > 0 {
                        self.asm.add(r8, *off as i32).map_err(|e| e.to_string())?;
                    }
                }
                _ => {}
            }


            Ok(())
        }

        /// Compute platform-independent pointer map for a fusion group.
        fn compute_group_pointer_map(
            group: &FusionGroup,
            graph: &CompilerGraph,
            weight_layout: &WeightLayout,
            alloc: &BufferAllocation,
        ) -> GroupPointerMap {
            let mut map = GroupPointerMap::default();

            let op = match graph.op(group.anchor) {
                Some(op) => op,
                None => return map,
            };

            // A matrix (input[0] of anchor op)
            if !op.inputs.is_empty() {
                let a_tid = op.inputs[0];
                if a_tid == graph.inputs[0] {
                    map.a_ptr = Some(PtrSource::ActivationInput);
                } else if let Some(off) = alloc.offset_of(a_tid) {
                    map.a_ptr = Some(PtrSource::Scratchpad(off));
                }
            }

            // B matrix (input[1] of anchor op, for GEMM/GemmBias)
            if op.inputs.len() > 1 {
                let b_tid = op.inputs[1];
                if let Some(off) = weight_layout.offset_of(b_tid) {
                    map.b_ptr = Some(PtrSource::WeightBlob(off));
                } else if let Some(off) = alloc.offset_of(b_tid) {
                    map.b_ptr = Some(PtrSource::Scratchpad(off));
                }
            }

            // Bias (input[2] of GemmBias)
            if matches!(op.kind, OpKind::GemmBias { .. }) && op.inputs.len() > 2 {
                let bias_tid = op.inputs[2];
                if let Some(off) = weight_layout.offset_of(bias_tid) {
                    map.bias_ptr = Some(PtrSource::WeightBlob(off));
                }
            }

            // V pointer (input[2] of MultiHeadAttention) -> bias_ptr (rdx)
            if matches!(op.kind, OpKind::MultiHeadAttention { .. }) && op.inputs.len() > 2 {
                let v_tid = op.inputs[2];
                if let Some(off) = weight_layout.offset_of(v_tid) {
                    map.bias_ptr = Some(PtrSource::WeightBlob(off));
                } else if let Some(off) = alloc.offset_of(v_tid) {
                    map.bias_ptr = Some(PtrSource::Scratchpad(off));
                }
            }

            // Norm weights/bias for NormIntoGemm groups: check predecessor ops
            for &op_id in &group.ops {
                if let Some(norm_op) = graph.op(op_id) {
                    match &norm_op.kind {
                        OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } => {
                            // norm weight = input[1]
                            if norm_op.inputs.len() > 1 {
                                let nw_tid = norm_op.inputs[1];
                                if let Some(off) = weight_layout.offset_of(nw_tid) {
                                    map.norm_weight_ptr = Some(PtrSource::WeightBlob(off));
                                }
                            }
                            // norm bias = input[2] (LayerNorm only)
                            if matches!(norm_op.kind, OpKind::LayerNorm { .. }) && norm_op.inputs.len() > 2 {
                                let nb_tid = norm_op.inputs[2];
                                if let Some(off) = weight_layout.offset_of(nb_tid) {
                                    map.norm_bias_ptr = Some(PtrSource::WeightBlob(off));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Output: use the final op in the group (last epilogue, or anchor if no epilogue)
            let final_op_id = group.epilogue.last().copied().unwrap_or(group.anchor);
            let final_op = graph.op(final_op_id).unwrap_or(op);
            if !final_op.outputs.is_empty() {
                let out_tid = final_op.outputs[0];
                if graph.outputs.contains(&out_tid) {
                    map.output_ptr = Some(PtrSource::GraphOutput);
                } else if let Some(off) = alloc.offset_of(out_tid) {
                    map.output_ptr = Some(PtrSource::Scratchpad(off));
                }
            }

            map
        }

        /// Dispatch a fusion group to the appropriate emitter.
        fn emit_group(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            match group.mode {
                FusionMode::Standalone => {
                    self.emit_standalone(group, graph, alloc, profile, registry)
                }
                FusionMode::LoopFusion => {
                    self.emit_elementwise_chain(group, graph, alloc, profile, registry)
                }
                FusionMode::EpilogueInjection => {
                    self.emit_gemm_with_epilogue(group, graph, alloc, profile, registry)
                }
                FusionMode::QkvSharedInput => {
                    for &op_id in &group.ops {
                        let single = FusionGroup {
                            id: group.id,
                            anchor: op_id,
                            epilogue: vec![],
                            mode: FusionMode::Standalone,
                            ops: vec![op_id],
                        };
                        self.emit_standalone(&single, graph, alloc, profile, registry)?;
                    }
                    Ok(())
                }
                FusionMode::NormIntoGemm => {
                    self.emit_norm_into_gemm(group, graph, profile)
                }
                FusionMode::TileLevelFusion { predecessor, tile_rows } => {
                    self.emit_tile_level_fusion(group, graph, profile, predecessor, tile_rows)
                }
                FusionMode::ComputeRoot { predecessor } => {
                    self.emit_compute_root(group, graph, profile, predecessor)
                }
            }
        }

        /// Emit a standalone op: GEMM, or trace-driven codegen for Reduction/NormLike/Elementwise.
        fn emit_standalone(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing op")?;
            eprintln!("[JIT-DBG] emit_standalone: group {} anchor={:?} op_kind={:?}", group.id, group.anchor, op.kind);
            match &op.kind {
                OpKind::Gemm { m, n, k } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile, &[])
                }
                OpKind::GemmBias { m, n, k } => {
                    // Save rdx (bias ptr) before GEMM since the microkernel clobbers it
                    self.asm.mov(qword_ptr(rbp - 48), rdx).map_err(|e| e.to_string())?;
                    self.emit_gemm_microkernel(*m, *n, *k, profile, &[])?;
                    // Restore rdx (bias ptr) and apply bias addition
                    self.asm.mov(rdx, qword_ptr(rbp - 48)).map_err(|e| e.to_string())?;
                    self.emit_bias_add(*m, *n)
                }
                OpKind::QuantGemm { m, n, k, block_size, bits } => {
                    self.emit_quant_gemm(*m, *n, *k, *block_size, *bits, profile, &[])
                }
                OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
                    self.emit_multi_head_attention(*seq_len, *num_heads, *head_dim)
                }
                OpKind::MeanPool { seq_len, hidden } => {
                    self.emit_mean_pool(*seq_len, *hidden)
                }
                _ => {
                    if let Some(reg) = registry {
                        let key = ScalarOpRegistry::key_from_op_kind(&op.kind);
                        if let Some(trace) = reg.get_trace(&key) {
                            return match &trace.pattern {
                                ComputePattern::Elementwise { .. }
                                | ComputePattern::BinaryElementwise { .. } => {
                                    self.emit_elementwise_chain(group, graph, alloc, profile, registry)
                                }
                                ComputePattern::NormLike { .. } => {
                                    self.emit_standalone_norm(group, graph, profile, reg)
                                }
                                ComputePattern::Reduction { .. } => {
                                    self.emit_standalone_reduction(group, graph, profile, reg)
                                }
                                other => Err(format!("x86_64: unhandled ComputePattern {:?} for {:?}", other, op.kind)),
                            };
                        }
                    }
                    Err(format!("x86_64: no registry entry for {:?}", op.kind))
                }
            }
        }

        /// Emit mean-pool: average `seq_len` rows of `hidden` f32 elements.
        ///
        /// ABI contract (same as other standalone ops):
        ///   rdi = input ptr  \[seq_len * hidden f32\]
        ///   r8  = output ptr \[hidden f32\]
        fn emit_mean_pool(
            &mut self,
            seq_len: usize,
            hidden: usize,
        ) -> Result<(), String> {
            // Broadcast 1.0 / seq_len for the final scaling multiply
            let inv_n = 1.0f32 / seq_len as f32;
            let inv_label = self.const_f32(inv_n);

            let simd_w = self.simd_width; // 8 for AVX2, 16 for AVX-512
            let vec_bytes = simd_w * 4;   // 32 for AVX2, 64 for AVX-512
            let vec_count = hidden / simd_w;
            let tail = hidden % simd_w;

            // ── SIMD columns (zmm for AVX-512, ymm for AVX2) ──
            for col in 0..vec_count {
                let col_byte_off = (col * vec_bytes) as i32;

                if self.use_avx512 {
                    // ── 16-wide (zmm) path ──
                    self.asm.vmovups(zmm0, zmmword_ptr(rdi + col_byte_off))
                        .map_err(|e| e.to_string())?;
                    for row in 1..seq_len {
                        let off = (row * hidden * 4 + col * vec_bytes) as i32;
                        self.asm.vaddps(zmm0, zmm0, zmmword_ptr(rdi + off))
                            .map_err(|e| e.to_string())?;
                    }
                    self.asm.vbroadcastss(zmm1, dword_ptr(inv_label))
                        .map_err(|e| e.to_string())?;
                    self.asm.vmulps(zmm0, zmm0, zmm1)
                        .map_err(|e| e.to_string())?;
                    self.asm.vmovups(zmmword_ptr(r8 + col_byte_off), zmm0)
                        .map_err(|e| e.to_string())?;
                } else {
                    // ── 8-wide (ymm) path ──
                    self.asm.vmovups(ymm0, ymmword_ptr(rdi + col_byte_off))
                        .map_err(|e| e.to_string())?;
                    for row in 1..seq_len {
                        let off = (row * hidden * 4 + col * vec_bytes) as i32;
                        self.asm.vaddps(ymm0, ymm0, ymmword_ptr(rdi + off))
                            .map_err(|e| e.to_string())?;
                    }
                    self.asm.vbroadcastss(ymm1, dword_ptr(inv_label))
                        .map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm0, ymm0, ymm1)
                        .map_err(|e| e.to_string())?;
                    self.asm.vmovups(ymmword_ptr(r8 + col_byte_off), ymm0)
                        .map_err(|e| e.to_string())?;
                }
            }

            // ── Scalar tail ──
            if tail > 0 {
                let base = (vec_count * vec_bytes) as i32;
                for i in 0..tail {
                    let col_off = base + (i * 4) as i32;
                    // First row
                    self.asm.vmovss(xmm0, dword_ptr(rdi + col_off))
                        .map_err(|e| e.to_string())?;
                    // Sum remaining rows
                    for row in 1..seq_len {
                        let off = (row * hidden * 4) as i32 + col_off;
                        self.asm.vaddss(xmm0, xmm0, dword_ptr(rdi + off))
                            .map_err(|e| e.to_string())?;
                    }
                    // Scale
                    self.asm.vmulss(xmm0, xmm0, dword_ptr(inv_label))
                        .map_err(|e| e.to_string())?;
                    // Store
                    self.asm.vmovss(dword_ptr(r8 + col_off), xmm0)
                        .map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Emit multi-head attention as fully JIT-compiled SIMD code.
        ///
        /// ABI contract:
        ///   rdi = Q ptr (input\[0\])
        ///   rsi = K ptr (input\[1\])
        ///   rdx = V ptr (input\[2\], mapped via bias_ptr)
        ///   r8  = output ptr
        ///   [rbp+32] = scratchpad ptr
        ///
        /// Scratchpad layout (all [seq_len, head_dim] or [seq_len, seq_len]):
        ///   q_head:   seq_len * head_dim * 4 bytes
        ///   k_head:   seq_len * head_dim * 4 bytes
        ///   v_head:   seq_len * head_dim * 4 bytes
        ///   scores:   seq_len * seq_len  * 4 bytes
        ///   out_head: seq_len * head_dim * 4 bytes
        fn emit_multi_head_attention(
            &mut self,
            seq_len: usize,
            num_heads: usize,
            head_dim: usize,
        ) -> Result<(), String> {
            let hidden = num_heads * head_dim;
            let hd_bytes = (head_dim * 4) as i32;
            let hidden_bytes = (hidden * 4) as i32;

            // Scratchpad allocation
            let head_mat_size = seq_len * head_dim * 4;
            let scores_size = seq_len * seq_len * 4;
            let total_scratch = head_mat_size * 4 + scores_size; // q, k, v, out + scores

            let q_off = self.blis_scratchpad_offset as i32;
            let k_off = q_off + head_mat_size as i32;
            let v_off = k_off + head_mat_size as i32;
            let sc_off = v_off + head_mat_size as i32;
            let out_off = sc_off + scores_size as i32;

            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset + total_scratch - self.blis_base_offset
            );

            let scale = 1.0 / (head_dim as f32).sqrt();
            let scale_label = self.const_f32(scale);

            // Load scratchpad base into rbx
            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            // Save input pointers onto local stack frame.
            // Layout: [rsp+0]=Q, [rsp+8]=K, [rsp+16]=V, [rsp+24]=output, [rsp+32]=head_counter
            // 48 bytes (6 qwords) keeps 16-byte alignment.
            self.asm.sub(rsp, 48i32).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp), rdi).map_err(|e| e.to_string())?;       // Q
            self.asm.mov(qword_ptr(rsp + 8), rsi).map_err(|e| e.to_string())?;   // K
            self.asm.mov(qword_ptr(rsp + 16), rdx).map_err(|e| e.to_string())?;  // V
            self.asm.mov(qword_ptr(rsp + 24), r8).map_err(|e| e.to_string())?;   // output

            // -- Head loop: for h in 0..num_heads --
            let mut h_loop = self.asm.create_label();
            let mut h_done = self.asm.create_label();

            self.asm.xor(eax, eax).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp + 32), rax).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut h_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(qword_ptr(rsp + 32), num_heads as i32).map_err(|e| e.to_string())?;
            self.asm.jge(h_done).map_err(|e| e.to_string())?;

            // r12 = h * head_dim * 4 (byte offset of this head within hidden dim)
            self.asm.mov(r12, qword_ptr(rsp + 32)).map_err(|e| e.to_string())?;
            self.asm.imul_3(r12, r12, hd_bytes).map_err(|e| e.to_string())?;

            // Step 1: Extract Q, K, V heads from [seq, hidden] to [seq, head_dim]
            self.emit_mha_extract_heads(seq_len, head_dim, hidden_bytes, q_off, k_off, v_off)?;

            // Step 2: scores[i][j] = dot(Q_head[i], K_head[j]) * scale
            self.emit_mha_compute_scores(seq_len, head_dim, q_off, k_off, sc_off, scale_label)?;

            // Step 3: Row-wise softmax on scores (in-place)
            self.emit_mha_softmax_inplace(seq_len, sc_off)?;

            // Step 4: out_head = scores @ V_head
            self.emit_mha_scores_times_v(seq_len, head_dim, sc_off, v_off, out_off)?;

            // Step 5: Scatter out_head back to output [seq, hidden]
            self.emit_mha_scatter_head(seq_len, head_dim, hidden_bytes, out_off)?;

            // Increment head counter
            self.asm.mov(rax, qword_ptr(rsp + 32)).map_err(|e| e.to_string())?;
            self.asm.inc(rax).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp + 32), rax).map_err(|e| e.to_string())?;
            self.asm.jmp(h_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut h_done).map_err(|e| e.to_string())?;

            // Restore stack
            self.asm.add(rsp, 48i32).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// MHA helper: extract Q, K, V heads from interleaved [seq, hidden] layout
        /// into contiguous [seq, head_dim] buffers in the scratchpad.
        ///
        /// Invariants on entry:
        ///   rbx = scratchpad base, r12 = h * head_dim * 4,
        ///   [rsp+0]=Q, [rsp+8]=K, [rsp+16]=V
        /// Clobbers: r13, r14, r15, rax, rdi, rsi, ymm0.
        fn emit_mha_extract_heads(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            hidden_bytes: i32,
            q_off: i32,
            k_off: i32,
            v_off: i32,
        ) -> Result<(), String> {
            let hd_bytes = (head_dim * 4) as i32;
            let vec_count = head_dim / 8;
            let tail = head_dim % 8;

            let mut s_loop = self.asm.create_label();
            let mut s_done = self.asm.create_label();

            self.asm.xor(r13d, r13d).map_err(|e| e.to_string())?;  // r13 = s
            self.asm.set_label(&mut s_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13d, seq_len as i32).map_err(|e| e.to_string())?;
            self.asm.jge(s_done).map_err(|e| e.to_string())?;

            // r14 = s * hidden_bytes (source row byte offset)
            self.asm.mov(r14, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(r14, r14, hidden_bytes).map_err(|e| e.to_string())?;
            // rax = src byte offset within row = r14 + r12 (s*hidden*4 + h*hd*4)
            self.asm.lea(rax, qword_ptr(r14 + r12)).map_err(|e| e.to_string())?;

            // r15 = s * hd_bytes (dest row byte offset in head buffer)
            self.asm.mov(r15, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(r15, r15, hd_bytes).map_err(|e| e.to_string())?;

            // --- Copy Q head ---
            self.asm.mov(rdi, qword_ptr(rsp)).map_err(|e| e.to_string())?;  // Q base
            self.asm.add(rdi, rax).map_err(|e| e.to_string())?;
            self.asm.lea(rsi, qword_ptr(rbx + q_off)).map_err(|e| e.to_string())?;
            self.asm.add(rsi, r15).map_err(|e| e.to_string())?;
            for v in 0..vec_count {
                let off = (v * 32) as i32;
                self.asm.vmovups(ymm0, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rsi + off), ymm0).map_err(|e| e.to_string())?;
            }
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm0, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovss(dword_ptr(rsi + off), xmm0).map_err(|e| e.to_string())?;
            }

            // --- Copy K head ---
            self.asm.mov(rdi, qword_ptr(rsp + 8)).map_err(|e| e.to_string())?;
            self.asm.add(rdi, rax).map_err(|e| e.to_string())?;
            self.asm.lea(rsi, qword_ptr(rbx + k_off)).map_err(|e| e.to_string())?;
            self.asm.add(rsi, r15).map_err(|e| e.to_string())?;
            for v in 0..vec_count {
                let off = (v * 32) as i32;
                self.asm.vmovups(ymm0, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rsi + off), ymm0).map_err(|e| e.to_string())?;
            }
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm0, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovss(dword_ptr(rsi + off), xmm0).map_err(|e| e.to_string())?;
            }

            // --- Copy V head ---
            self.asm.mov(rdi, qword_ptr(rsp + 16)).map_err(|e| e.to_string())?;
            self.asm.add(rdi, rax).map_err(|e| e.to_string())?;
            self.asm.lea(rsi, qword_ptr(rbx + v_off)).map_err(|e| e.to_string())?;
            self.asm.add(rsi, r15).map_err(|e| e.to_string())?;
            for v in 0..vec_count {
                let off = (v * 32) as i32;
                self.asm.vmovups(ymm0, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rsi + off), ymm0).map_err(|e| e.to_string())?;
            }
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm0, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovss(dword_ptr(rsi + off), xmm0).map_err(|e| e.to_string())?;
            }

            self.asm.inc(r13).map_err(|e| e.to_string())?;
            self.asm.jmp(s_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut s_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// MHA helper: compute scores[i][j] = dot(Q_head[i], K_head[j]) * scale.
        ///
        /// Performs (seq x hd) @ (hd x seq) = (seq x seq) via SIMD dot products.
        ///
        /// Invariants on entry: rbx = scratchpad base.
        /// Clobbers: r13, r14, r15, rax, rcx, rdi, rsi, ymm0-ymm3.
        fn emit_mha_compute_scores(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            q_off: i32,
            k_off: i32,
            sc_off: i32,
            scale_label: CodeLabel,
        ) -> Result<(), String> {
            let hd_bytes = (head_dim * 4) as i32;
            let seq_bytes = (seq_len * 4) as i32;
            let vec_count = head_dim / 8;
            let tail = head_dim % 8;

            let mut i_loop = self.asm.create_label();
            let mut i_done = self.asm.create_label();

            self.asm.xor(r13d, r13d).map_err(|e| e.to_string())?;  // r13 = i
            self.asm.set_label(&mut i_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13d, seq_len as i32).map_err(|e| e.to_string())?;
            self.asm.jge(i_done).map_err(|e| e.to_string())?;

            // r14 = &Q_head[i, 0] = rbx + q_off + i * hd_bytes
            self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, hd_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(r14, qword_ptr(rbx + q_off)).map_err(|e| e.to_string())?;
            self.asm.add(r14, rax).map_err(|e| e.to_string())?;

            // rdi = &scores[i, 0] = rbx + sc_off + i * seq_bytes
            self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, seq_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rdi, qword_ptr(rbx + sc_off)).map_err(|e| e.to_string())?;
            self.asm.add(rdi, rax).map_err(|e| e.to_string())?;

            // j loop
            let mut j_loop = self.asm.create_label();
            let mut j_done = self.asm.create_label();

            self.asm.xor(r15d, r15d).map_err(|e| e.to_string())?;  // r15 = j
            self.asm.set_label(&mut j_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r15d, seq_len as i32).map_err(|e| e.to_string())?;
            self.asm.jge(j_done).map_err(|e| e.to_string())?;

            // rsi = &K_head[j, 0] = rbx + k_off + j * hd_bytes
            self.asm.mov(rax, r15).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, hd_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rsi, qword_ptr(rbx + k_off)).map_err(|e| e.to_string())?;
            self.asm.add(rsi, rax).map_err(|e| e.to_string())?;

            // Compute dot product: sum_d Q_head[i,d] * K_head[j,d]
            // Accumulate in ymm1
            self.asm.vxorps(ymm1, ymm1, ymm1).map_err(|e| e.to_string())?;
            for v in 0..vec_count {
                let off = (v * 32) as i32;
                self.asm.vmovups(ymm2, ymmword_ptr(r14 + off)).map_err(|e| e.to_string())?;
                self.asm.vfmadd231ps(ymm1, ymm2, ymmword_ptr(rsi + off)).map_err(|e| e.to_string())?;
            }
            // Scalar tail
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm2, dword_ptr(r14 + off)).map_err(|e| e.to_string())?;
                self.asm.vfmadd231ss(xmm1, xmm2, dword_ptr(rsi + off)).map_err(|e| e.to_string())?;
            }

            // Horizontal sum ymm1 -> scalar in xmm0, broadcast to ymm1
            self.emit_horizontal_sum_ymm(ymm1, ymm1)?;
            // xmm0 now has the scalar sum; multiply by scale
            self.asm.vmulss(xmm0, xmm0, dword_ptr(scale_label)).map_err(|e| e.to_string())?;
            // Store to scores[i][j]
            self.asm.vmovss(dword_ptr(rdi + r15 * 4), xmm0).map_err(|e| e.to_string())?;

            self.asm.inc(r15).map_err(|e| e.to_string())?;
            self.asm.jmp(j_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut j_done).map_err(|e| e.to_string())?;

            self.asm.inc(r13).map_err(|e| e.to_string())?;
            self.asm.jmp(i_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut i_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// MHA helper: row-wise softmax on scores[seq_len, seq_len] in-place.
        ///
        /// For each row i:
        ///   1. Find row max (SIMD scan + horizontal reduce)
        ///   2. Subtract max, compute exp (Cephes AVX2 approximation)
        ///   3. Sum exp values
        ///   4. Multiply each element by 1/sum
        ///
        /// Invariants on entry: rbx = scratchpad base.
        /// Clobbers: r13, r14, rax, rcx, rdi, ymm0-ymm7.
        fn emit_mha_softmax_inplace(
            &mut self,
            seq_len: usize,
            sc_off: i32,
        ) -> Result<(), String> {
            let seq_bytes = (seq_len * 4) as i32;
            let vec_count = seq_len / 8;
            let tail = seq_len % 8;
            let neg_inf_label = self.const_f32(f32::NEG_INFINITY);
            let one_label = self.const_f32(1.0);

            let mut i_loop = self.asm.create_label();
            let mut i_done = self.asm.create_label();

            self.asm.xor(r13d, r13d).map_err(|e| e.to_string())?;  // r13 = i (row index)
            self.asm.set_label(&mut i_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13d, seq_len as i32).map_err(|e| e.to_string())?;
            self.asm.jge(i_done).map_err(|e| e.to_string())?;

            // rdi = &scores[i, 0] = rbx + sc_off + i * seq_bytes
            self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, seq_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rdi, qword_ptr(rbx + sc_off)).map_err(|e| e.to_string())?;
            self.asm.add(rdi, rax).map_err(|e| e.to_string())?;

            // --- Pass 1: find row max ---
            // Initialize ymm4 = -inf (accumulator for max)
            self.asm.vbroadcastss(ymm4, dword_ptr(neg_inf_label)).map_err(|e| e.to_string())?;
            for v in 0..vec_count {
                let off = (v * 32) as i32;
                self.asm.vmaxps(ymm4, ymm4, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
            }
            // Handle scalar tail for max
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vbroadcastss(ymm5, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmaxps(ymm4, ymm4, ymm5).map_err(|e| e.to_string())?;
            }
            // Horizontal max ymm4 -> broadcast in ymm4
            self.emit_horizontal_max_ymm(ymm4, ymm4)?;
            // ymm4 now has the row max broadcast across all lanes

            // --- Pass 2: exp(x - max) and accumulate sum ---
            // ymm5 = sum accumulator, starts at 0
            self.asm.vxorps(ymm5, ymm5, ymm5).map_err(|e| e.to_string())?;

            for v in 0..vec_count {
                let off = (v * 32) as i32;
                // ymm6 = scores[v] - max
                self.asm.vmovups(ymm6, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vsubps(ymm6, ymm6, ymm4).map_err(|e| e.to_string())?;
                // ymm6 = exp(ymm6), scratch = [ymm7, ymm3, ymm2]
                self.emit_exp_avx2(ymm6, ymm6, [ymm7, ymm3, ymm2])?;
                // Store back
                self.asm.vmovups(ymmword_ptr(rdi + off), ymm6).map_err(|e| e.to_string())?;
                // Accumulate sum
                self.asm.vaddps(ymm5, ymm5, ymm6).map_err(|e| e.to_string())?;
            }
            // Scalar tail for exp
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                // Load scalar, subtract max (scalar from xmm4 low lane)
                self.asm.vmovss(xmm6, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vsubss(xmm6, xmm6, xmm4).map_err(|e| e.to_string())?;
                // Broadcast to ymm for exp, compute, extract scalar
                self.asm.vbroadcastss(ymm6, xmm6).map_err(|e| e.to_string())?;
                self.emit_exp_avx2(ymm6, ymm6, [ymm7, ymm3, ymm2])?;
                // Store scalar back and accumulate
                self.asm.vmovss(dword_ptr(rdi + off), xmm6).map_err(|e| e.to_string())?;
                self.asm.vaddss(xmm5, xmm5, xmm6).map_err(|e| e.to_string())?;
            }

            // Horizontal sum ymm5 -> broadcast in ymm5
            self.emit_horizontal_sum_ymm(ymm5, ymm5)?;
            // Compute inv_sum = 1.0 / sum
            self.asm.vbroadcastss(ymm6, dword_ptr(one_label)).map_err(|e| e.to_string())?;
            self.asm.vdivss(xmm5, xmm6, xmm5).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(ymm5, xmm5).map_err(|e| e.to_string())?;

            // --- Pass 3: normalize (multiply by inv_sum) ---
            for v in 0..vec_count {
                let off = (v * 32) as i32;
                self.asm.vmulps(ymm6, ymm5, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rdi + off), ymm6).map_err(|e| e.to_string())?;
            }
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm6, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmulss(xmm6, xmm6, xmm5).map_err(|e| e.to_string())?;
                self.asm.vmovss(dword_ptr(rdi + off), xmm6).map_err(|e| e.to_string())?;
            }

            self.asm.inc(r13).map_err(|e| e.to_string())?;
            self.asm.jmp(i_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut i_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// MHA helper: compute out_head = scores @ V_head.
        ///
        /// out_head[i][d] = sum_j scores[i][j] * V_head[j][d]
        ///
        /// Optimized: for each i, for each j, broadcast scores[i][j] and
        /// vfmadd across the head_dim dimension of V_head[j].
        ///
        /// Invariants on entry: rbx = scratchpad base.
        /// Clobbers: r13, r14, r15, rax, rcx, rdi, rsi, ymm0-ymm3.
        fn emit_mha_scores_times_v(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            sc_off: i32,
            v_off: i32,
            out_off: i32,
        ) -> Result<(), String> {
            let hd_bytes = (head_dim * 4) as i32;
            let seq_bytes = (seq_len * 4) as i32;
            let hd_vec_count = head_dim / 8;
            let hd_tail = head_dim % 8;

            let mut i_loop = self.asm.create_label();
            let mut i_done = self.asm.create_label();

            self.asm.xor(r13d, r13d).map_err(|e| e.to_string())?;  // r13 = i
            self.asm.set_label(&mut i_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13d, seq_len as i32).map_err(|e| e.to_string())?;
            self.asm.jge(i_done).map_err(|e| e.to_string())?;

            // rdi = &out_head[i, 0] = rbx + out_off + i * hd_bytes
            self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, hd_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rdi, qword_ptr(rbx + out_off)).map_err(|e| e.to_string())?;
            self.asm.add(rdi, rax).map_err(|e| e.to_string())?;

            // Zero out_head[i] row
            self.asm.vxorps(ymm0, ymm0, ymm0).map_err(|e| e.to_string())?;
            for v in 0..hd_vec_count {
                let off = (v * 32) as i32;
                self.asm.vmovups(ymmword_ptr(rdi + off), ymm0).map_err(|e| e.to_string())?;
            }
            for t in 0..hd_tail {
                let off = ((hd_vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(dword_ptr(rdi + off), xmm0).map_err(|e| e.to_string())?;
            }

            // rcx = &scores[i, 0] = rbx + sc_off + i * seq_bytes
            self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, seq_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rcx, qword_ptr(rbx + sc_off)).map_err(|e| e.to_string())?;
            self.asm.add(rcx, rax).map_err(|e| e.to_string())?;

            // j loop: accumulate scores[i][j] * V_head[j]
            let mut j_loop = self.asm.create_label();
            let mut j_done = self.asm.create_label();

            self.asm.xor(r15d, r15d).map_err(|e| e.to_string())?;  // r15 = j
            self.asm.set_label(&mut j_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r15d, seq_len as i32).map_err(|e| e.to_string())?;
            self.asm.jge(j_done).map_err(|e| e.to_string())?;

            // ymm1 = broadcast(scores[i][j])
            self.asm.vbroadcastss(ymm1, dword_ptr(rcx + r15 * 4)).map_err(|e| e.to_string())?;

            // rsi = &V_head[j, 0] = rbx + v_off + j * hd_bytes
            self.asm.mov(rax, r15).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, hd_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rsi, qword_ptr(rbx + v_off)).map_err(|e| e.to_string())?;
            self.asm.add(rsi, rax).map_err(|e| e.to_string())?;

            // Accumulate: out_head[i][d] += scores[i][j] * V_head[j][d]
            for v in 0..hd_vec_count {
                let off = (v * 32) as i32;
                self.asm.vmovups(ymm2, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vfmadd231ps(ymm2, ymm1, ymmword_ptr(rsi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rdi + off), ymm2).map_err(|e| e.to_string())?;
            }
            for t in 0..hd_tail {
                let off = ((hd_vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm2, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vfmadd231ss(xmm2, xmm1, dword_ptr(rsi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovss(dword_ptr(rdi + off), xmm2).map_err(|e| e.to_string())?;
            }

            self.asm.inc(r15).map_err(|e| e.to_string())?;
            self.asm.jmp(j_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut j_done).map_err(|e| e.to_string())?;

            self.asm.inc(r13).map_err(|e| e.to_string())?;
            self.asm.jmp(i_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut i_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// MHA helper: scatter out_head[seq, head_dim] back to output[seq, hidden].
        ///
        /// Writes head_dim floats per seq position from out_head to
        /// output[s * hidden + h * head_dim].
        ///
        /// Invariants on entry:
        ///   rbx = scratchpad base, r12 = h * head_dim * 4,
        ///   [rsp+24] = output ptr
        /// Clobbers: r13, r14, r15, rax, rdi, rsi, ymm0.
        fn emit_mha_scatter_head(
            &mut self,
            seq_len: usize,
            head_dim: usize,
            hidden_bytes: i32,
            out_off: i32,
        ) -> Result<(), String> {
            let hd_bytes = (head_dim * 4) as i32;
            let vec_count = head_dim / 8;
            let tail = head_dim % 8;

            let mut s_loop = self.asm.create_label();
            let mut s_done = self.asm.create_label();

            self.asm.xor(r13d, r13d).map_err(|e| e.to_string())?;  // r13 = s
            self.asm.set_label(&mut s_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13d, seq_len as i32).map_err(|e| e.to_string())?;
            self.asm.jge(s_done).map_err(|e| e.to_string())?;

            // rdi = &out_head[s, 0] = rbx + out_off + s * hd_bytes (source)
            self.asm.mov(r15, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(r15, r15, hd_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rdi, qword_ptr(rbx + out_off)).map_err(|e| e.to_string())?;
            self.asm.add(rdi, r15).map_err(|e| e.to_string())?;

            // rsi = &output[s, h*hd] = output_base + s*hidden_bytes + r12
            self.asm.mov(r14, r13).map_err(|e| e.to_string())?;
            self.asm.imul_3(r14, r14, hidden_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(rax, qword_ptr(r14 + r12)).map_err(|e| e.to_string())?;
            self.asm.mov(rsi, qword_ptr(rsp + 24)).map_err(|e| e.to_string())?;
            self.asm.add(rsi, rax).map_err(|e| e.to_string())?;

            // Copy head_dim floats from out_head to output
            for v in 0..vec_count {
                let off = (v * 32) as i32;
                self.asm.vmovups(ymm0, ymmword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rsi + off), ymm0).map_err(|e| e.to_string())?;
            }
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm0, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                self.asm.vmovss(dword_ptr(rsi + off), xmm0).map_err(|e| e.to_string())?;
            }

            self.asm.inc(r13).map_err(|e| e.to_string())?;
            self.asm.jmp(s_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut s_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit fused RmsNorm → GEMM: normalize A in-place (via scratchpad),
        /// then run the GEMM with the normalized data as input.
        ///
        /// ABI contract for NormIntoGemm groups:
        ///   rdi = A input ptr (pre-norm)
        ///   rsi = B weight matrix ptr
        ///   rdx = norm weight ptr
        ///   r8  = C output ptr
        ///   [rbp+32] = scratchpad ptr
        ///
        /// The normalized A is written to the scratchpad at offset 0,
        /// consuming m*k*4 bytes. BLIS packing buffers (if used) start
        /// after this region.
        fn emit_norm_into_gemm(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
        ) -> Result<(), String> {
            let norm_op_id = group.ops.first().copied()
                .ok_or("NormIntoGemm: missing norm op")?;
            let norm_op = graph.op(norm_op_id)
                .ok_or("NormIntoGemm: norm op not in graph")?;
            let gemm_op = graph.op(group.anchor)
                .ok_or("NormIntoGemm: GEMM op not in graph")?;

            let eps = match &norm_op.kind {
                OpKind::RmsNorm { eps } => *eps,
                OpKind::LayerNorm { eps } => {
                    return Err("NormIntoGemm: LayerNorm not yet supported".into());
                }
                other => {
                    return Err(format!("NormIntoGemm: expected norm op, got {:?}", other));
                }
            };

            let (m, n, k) = match &gemm_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => {
                    return Err(format!("NormIntoGemm: expected GEMM op, got {:?}", other));
                }
            };

            let norm_buf_bytes = m * k * 4;

            let norm_scratch_offset = self.blis_scratchpad_offset;
            self.blis_scratchpad_offset += norm_buf_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset - self.blis_base_offset
            );

            let row_bytes = (k * 4) as i32;

            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            self.asm.sub(rsp, 16i32).map_err(|e| e.to_string())?;  // 16-byte aligned slot
            self.asm.mov(qword_ptr(rsp), r8).map_err(|e| e.to_string())?;  // save C output ptr
            self.asm.mov(r13, rdi).map_err(|e| e.to_string())?;  // A input base
            self.asm.mov(r14, rdx).map_err(|e| e.to_string())?;  // norm weight ptr
            self.asm.mov(r15, rsi).map_err(|e| e.to_string())?;  // B matrix ptr

            let mut row_loop = self.asm.create_label();
            let mut row_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;  // r12 = 0 (row counter)
            self.asm.set_label(&mut row_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(row_done).map_err(|e| e.to_string())?;

            self.asm.mov(rax, r12).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, row_bytes).map_err(|e| e.to_string())?;

            // rdi = input row, r10 = output (norm scratch), r14 = weight
            self.asm.lea(rdi, qword_ptr(r13 + rax)).map_err(|e| e.to_string())?;
            self.asm.lea(r10, qword_ptr(rbx + rax + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;
            self.emit_norm_row_jit(r10, r14, k, eps)?;

            self.asm.inc(r12).map_err(|e| e.to_string())?;
            self.asm.jmp(row_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut row_done).map_err(|e| e.to_string())?;

            self.asm.mov(r8, qword_ptr(rsp)).map_err(|e| e.to_string())?;  // restore C output ptr
            self.asm.add(rsp, 16i32).map_err(|e| e.to_string())?;
            self.asm.mov(rsi, r15).map_err(|e| e.to_string())?;  // restore B matrix ptr
            self.asm.lea(rdi, qword_ptr(rbx + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;  // A = normalized data

            self.emit_gemm_microkernel(m, n, k, profile, &[])
        }

        /// Emit TileLevelFusion: tile the predecessor norm into the GEMM's
        /// MC loop so each MC strip computes norm, pack_a, microkernel
        /// before moving to the next strip.
        ///
        /// Used when the full norm output exceeds 75% L1. By computing norm
        /// for only `tile_rows` (= MC) rows at a time, the norm output stays
        /// cache-hot when pack_a reads it immediately after.
        ///
        /// Loop structure (ic moved outside pc):
        /// ```text
        /// jc loop (NC):
        ///   ic loop (MC):
        ///     compute norm for mc_cur rows into norm_scratch
        ///     pc loop (KC):
        ///       pack_b: B[pc..pc+kc, jc..jc+nc] into packed_b
        ///       pack_a: norm_scratch[0..mc_cur, pc..pc+kc] into packed_a
        ///       jr loop (NR):
        ///         ir loop (MR):
        ///           microkernel tile
        /// ```
        ///
        /// ABI contract (same as NormIntoGemm):
        ///   rdi = A input ptr (pre-norm)
        ///   rsi = B weight matrix ptr
        ///   rdx = norm weight ptr
        ///   r8  = C output ptr
        ///   [rbp+32] = scratchpad ptr
        fn emit_tile_level_fusion(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
            tile_rows: usize,
        ) -> Result<(), String> {
            let norm_op = graph.op(predecessor)
                .ok_or("TileLevelFusion: predecessor norm op not in graph")?;
            let gemm_op = graph.op(group.anchor)
                .ok_or("TileLevelFusion: GEMM op not in graph")?;

            let eps = match &norm_op.kind {
                OpKind::RmsNorm { eps } => *eps,
                OpKind::LayerNorm { eps: _ } => {
                    return Err("TileLevelFusion: LayerNorm not yet supported".into());
                }
                other => {
                    return Err(format!("TileLevelFusion: expected norm op, got {:?}", other));
                }
            };

            let (m, n, k) = match &gemm_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => {
                    return Err(format!("TileLevelFusion: expected GEMM op, got {:?}", other));
                }
            };

            let blocking = profile.gemm_blocking(m, n, k);
            let mr = blocking.mr;
            let nr = blocking.nr;
            let nr_vecs = nr / 8; // AVX2: 8 f32 per ymm
            let kc = blocking.kc;
            let mc = tile_rows;
            let nc = blocking.nc;

            let c_row_bytes = (n * 4) as i32;
            let lda = k; // norm output is row-major [mc, k], stride = k
            let ldb = n; // B is row-major [k, n], stride = n

            // -- Scratchpad allocation --
            // norm_scratch: mc * k * 4 (norm output for one MC strip)
            let norm_scratch_bytes = mc * k * 4;
            let norm_scratch_off = self.blis_scratchpad_offset as i32;

            // pack_a / pack_b buffers (same layout as emit_gemm_blis)
            let pack_a_panels = (mc + mr - 1) / mr;
            let pack_a_bytes = pack_a_panels * mr * kc * 4;
            let pack_b_panels = (nc + nr - 1) / nr;
            let pack_b_bytes = pack_b_panels * nr * kc * 4;

            let pack_a_off = (self.blis_scratchpad_offset + norm_scratch_bytes) as i32;
            let pack_b_off = (self.blis_scratchpad_offset + norm_scratch_bytes + pack_a_bytes) as i32;

            let total_extra = self.blis_scratchpad_offset + norm_scratch_bytes
                + pack_a_bytes + pack_b_bytes - self.blis_base_offset;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(total_extra);

            // Load scratchpad base.
            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            // Allocate stack locals (48 bytes, 16-byte aligned):
            //   [rsp+0]  = nc_cur
            //   [rsp+8]  = kc_cur
            //   [rsp+16] = mc_cur
            //   [rsp+24] = ir (reused by inner loops)
            //   [rsp+32] = saved norm weight ptr (rdx)
            //   [rsp+40] = saved original A ptr (rdi)
            self.asm.sub(rsp, 48i32).map_err(|e| e.to_string())?;

            // Save norm weight ptr and original A ptr -- clobbered by inner loops.
            self.asm.mov(qword_ptr(rsp + 32), rdx).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp + 40), rdi).map_err(|e| e.to_string())?;

            let loc_nc: i32 = 0;
            let loc_kc: i32 = 8;
            let loc_mc: i32 = 16;
            let loc_ir: i32 = 24;

            // -- jc loop (NC) --
            let mut jc_loop = self.asm.create_label();
            let mut jc_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, n as i32).map_err(|e| e.to_string())?;
            self.asm.jge(jc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, n, nc, r12)?;
            self.asm.mov(qword_ptr(rsp + loc_nc), rax).map_err(|e| e.to_string())?;

            // -- ic loop (MC) -- moved outside pc for tile-level fusion
            let mut ic_loop = self.asm.create_label();
            let mut ic_done = self.asm.create_label();

            self.asm.xor(r14, r14).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r14, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(ic_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, m, mc, r14)?;
            self.asm.mov(qword_ptr(rsp + loc_mc), rax).map_err(|e| e.to_string())?;

            // -- Compute norm for mc_cur rows into norm_scratch --
            {
                let row_bytes_k = (k * 4) as i32;
                let mut norm_loop = self.asm.create_label();
                let mut norm_done = self.asm.create_label();

                // Use [rsp+24] (loc_ir) as row counter -- emit_norm_row_jit clobbers rcx
                self.asm.mov(qword_ptr(rsp + loc_ir), 0i32).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut norm_loop).map_err(|e| e.to_string())?;
                self.asm.mov(rcx, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
                self.asm.jge(norm_done).map_err(|e| e.to_string())?;

                // Byte offset in original A: (ic + row) * k * 4
                self.asm.mov(rax, r14).map_err(|e| e.to_string())?;
                self.asm.add(rax, rcx).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, row_bytes_k).map_err(|e| e.to_string())?;

                // Byte offset in norm_scratch: row * k * 4
                self.asm.mov(r9, rcx).map_err(|e| e.to_string())?;
                self.asm.imul_3(r9, r9, row_bytes_k).map_err(|e| e.to_string())?;

                // rdi = input row = original_A + byte_offset
                self.asm.mov(rdi, qword_ptr(rsp + 40)).map_err(|e| e.to_string())?; // original A ptr
                self.asm.add(rdi, rax).map_err(|e| e.to_string())?;

                // r10 = output row = norm_scratch + row_byte_offset
                self.asm.lea(r10, qword_ptr(rbx + norm_scratch_off)).map_err(|e| e.to_string())?;
                self.asm.add(r10, r9).map_err(|e| e.to_string())?;

                // r11 = weight ptr (saved on stack at [rsp+32])
                self.asm.mov(r11, qword_ptr(rsp + 32)).map_err(|e| e.to_string())?;

                self.emit_norm_row_jit(r10, r11, k, eps)?;

                // Increment row counter
                self.asm.mov(rax, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
                self.asm.inc(rax).map_err(|e| e.to_string())?;
                self.asm.mov(qword_ptr(rsp + loc_ir), rax).map_err(|e| e.to_string())?;
                self.asm.jmp(norm_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut norm_done).map_err(|e| e.to_string())?;
            }

            // -- pc loop (KC) --
            let mut pc_loop = self.asm.create_label();
            let mut pc_done = self.asm.create_label();

            self.asm.xor(r13, r13).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13, k as i32).map_err(|e| e.to_string())?;
            self.asm.jge(pc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, k, kc, r13)?;
            self.asm.mov(qword_ptr(rsp + loc_kc), rax).map_err(|e| e.to_string())?;

            // -- pack_b: B[pc..pc+kc, jc..jc+nc] into packed_b --
            {
                // B source ptr: &B[pc, jc] = rsi + (pc * ldb + jc) * 4
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, ldb as i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, r12).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, rsi).map_err(|e| e.to_string())?;
                self.asm.mov(r10, rax).map_err(|e| e.to_string())?;

                // r11 = packed_b destination
                self.asm.lea(r11, qword_ptr(rbx + pack_b_off)).map_err(|e| e.to_string())?;

                self.emit_jit_pack_b_avx2(ldb, nr, pack_b_off, loc_kc, loc_nc)?;
            }

            // -- pack_a: norm_scratch[0..mc_cur, pc..pc+kc] into packed_a --
            {
                // Source: &norm_scratch[0, pc] = rbx + norm_scratch_off + pc * 4
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.lea(r15, qword_ptr(rbx + norm_scratch_off)).map_err(|e| e.to_string())?;
                self.asm.add(r15, rax).map_err(|e| e.to_string())?;

                self.emit_jit_pack_a_avx2(lda, mr, pack_a_off, loc_kc, loc_mc)?;
            }

            // -- jr / ir loops (NR / MR) -- reuse existing BLIS inner loops
            {
                let m_rem = m % mr;
                let n_rem = n % nr;
                let simd_w = 8usize;
                let n_rem_vecs = n_rem / simd_w;

                let mut jr_loop = self.asm.create_label();
                let mut jr_done = self.asm.create_label();
                let mut jr_rem_start = self.asm.create_label();

                self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_loop).map_err(|e| e.to_string())?;

                self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
                self.asm.add(rax, nr as i32).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                self.asm.jg(jr_rem_start).map_err(|e| e.to_string())?;

                // r11 = packed_b ptr for this jr strip
                self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                if pack_b_off != 0 {
                    self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                }

                self.emit_blis_ir_loop(
                    mr, nr_vecs, m_rem, loc_kc, n, nr_vecs, nr,
                    loc_ir, loc_mc, pack_a_off, c_row_bytes,
                    &[], k,
                )?;

                self.asm.add(r9, nr as i32).map_err(|e| e.to_string())?;
                self.asm.jmp(jr_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut jr_rem_start).map_err(|e| e.to_string())?;

                if n_rem_vecs > 0 {
                    self.asm.cmp(r9, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                    self.asm.jge(jr_done).map_err(|e| e.to_string())?;

                    self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                    self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                    self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                    self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                    if pack_b_off != 0 {
                        self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                    }

                    self.emit_blis_ir_loop(
                        mr, nr_vecs, m_rem, loc_kc, n, n_rem_vecs, nr,
                        loc_ir, loc_mc, pack_a_off, c_row_bytes,
                        &[], k,
                    )?;
                }

                self.asm.nop().map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_done).map_err(|e| e.to_string())?;
            }

            // -- Close pc loop --
            self.asm.add(r13, kc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(pc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_done).map_err(|e| e.to_string())?;

            // -- Close ic loop --
            self.asm.add(r14, mc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(ic_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_done).map_err(|e| e.to_string())?;

            // -- Close jc loop --
            self.asm.add(r12, nc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(jc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_done).map_err(|e| e.to_string())?;

            self.asm.add(rsp, 48i32).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit ComputeRoot fusion: compute predecessor norm fully into a
        /// scratchpad buffer, then run the GEMM reading from that buffer.
        ///
        /// Used when the norm output fits in L1 (le 75% L1), so no tiling
        /// is needed -- the entire norm result stays cache-hot for the GEMM.
        ///
        /// ABI contract (same as NormIntoGemm):
        ///   rdi = A input ptr (pre-norm)
        ///   rsi = B weight matrix ptr
        ///   rdx = norm weight ptr
        ///   r8  = C output ptr
        ///   [rbp+32] = scratchpad ptr
        fn emit_compute_root(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            predecessor: OpId,
        ) -> Result<(), String> {
            let norm_op = graph.op(predecessor)
                .ok_or("ComputeRoot: predecessor norm op not in graph")?;
            let gemm_op = graph.op(group.anchor)
                .ok_or("ComputeRoot: GEMM op not in graph")?;

            let eps = match &norm_op.kind {
                OpKind::RmsNorm { eps } => *eps,
                OpKind::LayerNorm { eps: _ } => {
                    return Err("ComputeRoot: LayerNorm not yet supported".into());
                }
                other => {
                    return Err(format!("ComputeRoot: expected norm op, got {:?}", other));
                }
            };

            let (m, n, k) = match &gemm_op.kind {
                OpKind::Gemm { m, n, k } | OpKind::QuantGemm { m, n, k, .. } => (*m, *n, *k),
                other => {
                    return Err(format!("ComputeRoot: expected GEMM op, got {:?}", other));
                }
            };

            // Allocate scratchpad for the full norm output (m * k * 4 bytes).
            let norm_buf_bytes = m * k * 4;
            let norm_scratch_offset = self.blis_scratchpad_offset;
            self.blis_scratchpad_offset += norm_buf_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset - self.blis_base_offset
            );

            let row_bytes = (k * 4) as i32;

            // Load scratchpad base into rbx.
            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            // Save registers that the norm loop clobbers.
            self.asm.sub(rsp, 16i32).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp), r8).map_err(|e| e.to_string())?;   // save C output ptr
            self.asm.mov(r13, rdi).map_err(|e| e.to_string())?;  // A input base
            self.asm.mov(r14, rdx).map_err(|e| e.to_string())?;  // norm weight ptr
            self.asm.mov(r15, rsi).map_err(|e| e.to_string())?;  // B matrix ptr

            // Row loop: JIT RmsNorm for each row.
            let mut row_loop = self.asm.create_label();
            let mut row_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;  // r12 = row counter
            self.asm.set_label(&mut row_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(row_done).map_err(|e| e.to_string())?;

            // Compute byte offset for current row.
            self.asm.mov(rax, r12).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, row_bytes).map_err(|e| e.to_string())?;

            // rdi = input row, r10 = output (norm scratch), r14 = weight
            self.asm.lea(rdi, qword_ptr(r13 + rax)).map_err(|e| e.to_string())?;
            self.asm.lea(r10, qword_ptr(rbx + rax + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;
            self.emit_norm_row_jit(r10, r14, k, eps)?;

            self.asm.inc(r12).map_err(|e| e.to_string())?;
            self.asm.jmp(row_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut row_done).map_err(|e| e.to_string())?;

            // Restore registers and redirect A to the norm output buffer.
            self.asm.mov(r8, qword_ptr(rsp)).map_err(|e| e.to_string())?;   // restore C
            self.asm.add(rsp, 16i32).map_err(|e| e.to_string())?;
            self.asm.mov(rsi, r15).map_err(|e| e.to_string())?;  // restore B
            self.asm.lea(rdi, qword_ptr(rbx + norm_scratch_offset as i32)).map_err(|e| e.to_string())?;  // A = norm output

            // Run the GEMM with A pointing to the normalized data.
            self.emit_gemm_microkernel(m, n, k, profile, &[])
        }

        /// Emit a fused elementwise chain as a single SIMD loop.
        ///
        /// All ops in the chain execute on the same data in registers,
        /// no intermediate memory traffic.
        ///
        /// Generated code layout (System V AMD64 ABI):
        /// ```text
        ///   rdi = input ptr (arg 0)
        ///   r8  = scratchpad ptr (arg 4, used as output)
        ///   element_count baked as immediate
        ///
        ///   xor rbx, rbx              ; i = 0
        /// .loop:
        ///   cmp rbx, count
        ///   jge .done
        ///   vmovups ymm0, [rdi + rbx*4]
        ///   ; --- trace ops ---
        ///   vmovups [r8 + rbx*4], ymm_result
        ///   add rbx, 8
        ///   jmp .loop
        /// .done:
        /// ```
        /// Emit a fused elementwise chain as a single SIMD loop.
        ///
        /// All ops in the chain execute on the same data in registers,
        /// no intermediate memory traffic. Trace bodies are looked up
        /// from the `ScalarOpRegistry` when available; falls back to
        /// the hardcoded `emit_chain_body` path otherwise.
        ///
        /// Generated code layout (System V AMD64 ABI):
        /// ```text
        ///   rdi = input ptr (arg 0)
        ///   rsi = second input ptr (arg 1, for binary ops)
        ///   r8  = output ptr (arg 4)
        ///   rbx = byte offset counter
        ///
        ///   xor rbx, rbx              ; i = 0
        /// .loop:
        ///   cmp rbx, total_vec_bytes
        ///   jge .done
        ///   vmovups ymm0, [rdi + rbx]
        ///   ; --- trace ops (from registry) ---
        ///   vmovups [r8 + rbx], ymm0
        ///   add rbx, 32
        ///   jmp .loop
        /// .done:
        ///   ; scalar tail for remaining elements
        /// ```
        fn emit_elementwise_chain(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            _alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor op")?;

            let elem_count = if let Some(&out_id) = op.outputs.first() {
                graph.tensor_numel(out_id).unwrap_or(0)
            } else {
                0
            };

            if elem_count == 0 {
                return Err("elementwise chain: elem_count=0".into());
            }

            let mut trace_info: Vec<(Vec<TraceOp>, bool)> = Vec::new();

            if let Some(reg) = registry {
                let anchor_op = graph.op(group.anchor).ok_or("missing anchor")?;
                let key = ScalarOpRegistry::key_from_op_kind(&anchor_op.kind);
                if let Some(trace) = reg.get_trace(&key) {
                    if let Some(body) = trace.pattern.body() {
                        let is_binary = matches!(
                            trace.pattern,
                            ComputePattern::BinaryElementwise { .. }
                        );
                        trace_info.push((body.to_vec(), is_binary));
                    }
                }

                for &epi_id in &group.epilogue {
                    let epi_op = graph.op(epi_id).ok_or("missing epilogue op")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            let is_binary = matches!(
                                trace.pattern,
                                ComputePattern::BinaryElementwise { .. }
                            );
                            trace_info.push((body.to_vec(), is_binary));
                        }
                    }
                }
            }

            if trace_info.is_empty() {
                return self.emit_elementwise_chain_hardcoded(group, graph, elem_count);
            }

            let simd_w = self.simd_width; // 8 for AVX2, 16 for AVX-512
            let simd_bytes = (simd_w * 4) as i32; // 32 or 64
            let vec_count = elem_count / simd_w;
            let tail = elem_count % simd_w;

            let output_bytes = elem_count * 4;
            let (_, l2_size, _) = profile.cache_sizes();
            let use_nt_store = output_bytes > l2_size;

            let has_binary = trace_info.iter().any(|(_, b)| *b);

            let prefetch_dist = 512i32;

            let unroll = 4usize;
            let unrolled_vecs = (vec_count / unroll) * unroll;

            let unrolled_bytes = (unrolled_vecs * simd_w * 4) as i32;
            let total_vec_bytes = (vec_count * simd_w * 4) as i32;

            let mut unrolled_loop = self.asm.create_label();
            let mut remainder_loop = self.asm.create_label();
            let mut done_label = self.asm.create_label();

            self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;

            if unrolled_vecs > 0 {
                self.asm.set_label(&mut unrolled_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, unrolled_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(remainder_loop).map_err(|e| e.to_string())?;

                self.asm.prefetcht0(byte_ptr(rdi + rbx + prefetch_dist))
                    .map_err(|e| e.to_string())?;
                if has_binary {
                    self.asm.prefetcht0(byte_ptr(rsi + rbx + prefetch_dist))
                        .map_err(|e| e.to_string())?;
                }

                for _u in 0..unroll {
                    if self.use_avx512 {
                        self.asm.vmovups(zmm0, zmmword_ptr(rdi + rbx))
                            .map_err(|e| e.to_string())?;
                    } else {
                        self.asm.vmovups(ymm0, ymmword_ptr(rdi + rbx))
                            .map_err(|e| e.to_string())?;
                    }

                    for (body, is_binary) in &trace_info {
                        let mode = InputMode::Elementwise { is_binary: *is_binary, scalar_tail: false };
                        if self.use_avx512 {
                            self.emit_trace_on_stack_zmm(zmm0, body, mode)?;
                        } else {
                            self.emit_trace_on_stack_ymm(ymm0, body, mode)?;
                        }
                    }

                    if self.use_avx512 {
                        if use_nt_store {
                            self.asm.vmovntps(zmmword_ptr(r8 + rbx), zmm0)
                                .map_err(|e| e.to_string())?;
                        } else {
                            self.asm.vmovups(zmmword_ptr(r8 + rbx), zmm0)
                                .map_err(|e| e.to_string())?;
                        }
                    } else {
                        if use_nt_store {
                            self.asm.vmovntps(ymmword_ptr(r8 + rbx), ymm0)
                                .map_err(|e| e.to_string())?;
                        } else {
                            self.asm.vmovups(ymmword_ptr(r8 + rbx), ymm0)
                                .map_err(|e| e.to_string())?;
                        }
                    }

                    self.asm.add(rbx, simd_bytes).map_err(|e| e.to_string())?;
                }

                self.asm.jmp(unrolled_loop).map_err(|e| e.to_string())?;
            }

            self.asm.set_label(&mut remainder_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
            self.asm.jge(done_label).map_err(|e| e.to_string())?;

            self.asm.prefetcht0(byte_ptr(rdi + rbx + prefetch_dist))
                .map_err(|e| e.to_string())?;

            if self.use_avx512 {
                self.asm.vmovups(zmm0, zmmword_ptr(rdi + rbx))
                    .map_err(|e| e.to_string())?;
            } else {
                self.asm.vmovups(ymm0, ymmword_ptr(rdi + rbx))
                    .map_err(|e| e.to_string())?;
            }

            for (body, is_binary) in &trace_info {
                let mode = InputMode::Elementwise { is_binary: *is_binary, scalar_tail: false };
                if self.use_avx512 {
                    self.emit_trace_on_stack_zmm(zmm0, body, mode)?;
                } else {
                    self.emit_trace_on_stack_ymm(ymm0, body, mode)?;
                }
            }

            if self.use_avx512 {
                if use_nt_store {
                    self.asm.vmovntps(zmmword_ptr(r8 + rbx), zmm0)
                        .map_err(|e| e.to_string())?;
                } else {
                    self.asm.vmovups(zmmword_ptr(r8 + rbx), zmm0)
                        .map_err(|e| e.to_string())?;
                }
            } else {
                if use_nt_store {
                    self.asm.vmovntps(ymmword_ptr(r8 + rbx), ymm0)
                        .map_err(|e| e.to_string())?;
                } else {
                    self.asm.vmovups(ymmword_ptr(r8 + rbx), ymm0)
                        .map_err(|e| e.to_string())?;
                }
            }

            self.asm.add(rbx, simd_bytes).map_err(|e| e.to_string())?;
            self.asm.jmp(remainder_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut done_label).map_err(|e| e.to_string())?;

            if use_nt_store {
                self.asm.sfence().map_err(|e| e.to_string())?;
            }

            // Tail handling: AVX-512 uses masked load/store, AVX2 uses scalar loop
            if tail > 0 {
                let base_bytes = (vec_count * simd_w * 4) as i32;
                if self.use_avx512 {
                    self.emit_set_kmask(tail)?;
                    self.asm.mov(rbx, base_bytes as i64).map_err(|e| e.to_string())?;
                    self.asm.vmovups(zmm0.k1().z(), zmmword_ptr(rdi + base_bytes))
                        .map_err(|e| e.to_string())?;
                    for (body, is_binary) in &trace_info {
                        let mode = InputMode::Elementwise { is_binary: *is_binary, scalar_tail: true };
                        self.emit_trace_on_stack_zmm(zmm0, body, mode)?;
                    }
                    self.asm.vmovups(zmmword_ptr(r8 + base_bytes).k1(), zmm0)
                        .map_err(|e| e.to_string())?;
                } else {
                    for t in 0..tail as i32 {
                        let off = base_bytes + t * 4;
                        self.asm.mov(rbx, off as i64).map_err(|e| e.to_string())?;
                        self.asm.vmovss(xmm0, dword_ptr(rdi + off))
                            .map_err(|e| e.to_string())?;
                        for (body, is_binary) in &trace_info {
                            let mode = InputMode::Elementwise { is_binary: *is_binary, scalar_tail: true };
                            self.emit_trace_on_stack_ymm(ymm0, body, mode)?;
                        }
                        self.asm.vmovss(dword_ptr(r8 + off), xmm0)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            Ok(())
        }

        /// Hardcoded fallback for elementwise chain (no registry available).
        ///
        /// Uses the original OpKind-matching approach for SiLU, GELU, Add, Mul.
        fn emit_elementwise_chain_hardcoded(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            elem_count: usize,
        ) -> Result<(), String> {
            let simd_w = self.simd_width;
            let vec_count = elem_count / simd_w;
            let tail = elem_count % simd_w;

            let mut loop_label = self.asm.create_label();
            let mut done_label = self.asm.create_label();

            let total_vec_bytes = (vec_count * simd_w * 4) as i32;
            self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut loop_label).map_err(|e| e.to_string())?;
            self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
            self.asm.jge(done_label).map_err(|e| e.to_string())?;

            self.asm.vmovups(ymm0, ymmword_ptr(rdi + rbx))
                .map_err(|e| e.to_string())?;

            let result_reg = self.emit_chain_body(group, graph)?;

            self.asm.vmovups(ymmword_ptr(r8 + rbx), result_reg)
                .map_err(|e| e.to_string())?;

            self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
            self.asm.jmp(loop_label).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut done_label).map_err(|e| e.to_string())?;

            if tail > 0 {
                let base_offset = (vec_count * simd_w) as i32;
                for t in 0..tail as i32 {
                    let off = (base_offset + t) * 4;
                    self.asm.vmovss(xmm0, dword_ptr(rdi + off))
                        .map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r8 + off), xmm0)
                        .map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Emit the compute body for an elementwise chain (hardcoded fallback).
        ///
        /// Input is pre-loaded in ymm0. Returns the register holding the result.
        /// Uses ymm0-ymm5 for data, ymm13-ymm15 as scratch for exp/tanh.
        fn emit_chain_body(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
        ) -> Result<AsmRegisterYmm, String> {
            let op = graph.op(group.anchor).ok_or("missing anchor")?;
            match &op.kind {
                OpKind::Silu => {
                    self.asm.vxorps(ymm1, ymm1, ymm1).map_err(|e| e.to_string())?;
                    self.asm.vsubps(ymm1, ymm1, ymm0).map_err(|e| e.to_string())?;
                    self.emit_exp_avx2(ymm2, ymm1, [ymm13, ymm14, ymm15])?;
                    let one_label = self.const_f32(1.0);
                    self.asm.vbroadcastss(ymm3, dword_ptr(one_label)).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm2, ymm2, ymm3).map_err(|e| e.to_string())?;
                    self.asm.vdivps(ymm0, ymm0, ymm2).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                OpKind::Gelu => {
                    self.asm.vmulps(ymm1, ymm0, ymm0).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm1, ymm1, ymm0).map_err(|e| e.to_string())?;
                    let coeff_label = self.const_f32(0.044715);
                    self.asm.vbroadcastss(ymm2, dword_ptr(coeff_label)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm1, ymm2, ymm1).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm1, ymm0, ymm1).map_err(|e| e.to_string())?;
                    let sqrt2pi_label = self.const_f32(0.7978845608);
                    self.asm.vbroadcastss(ymm2, dword_ptr(sqrt2pi_label)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm1, ymm2, ymm1).map_err(|e| e.to_string())?;
                    self.emit_tanh_avx2(ymm1, ymm1, [ymm13, ymm14, ymm15])?;
                    let one_label = self.const_f32(1.0);
                    self.asm.vbroadcastss(ymm2, dword_ptr(one_label)).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm1, ymm1, ymm2).map_err(|e| e.to_string())?;
                    let half_label = self.const_f32(0.5);
                    self.asm.vbroadcastss(ymm2, dword_ptr(half_label)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm0, ymm2, ymm0).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm0, ymm0, ymm1).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                OpKind::Add => {
                    self.asm.vmovups(ymm1, ymmword_ptr(rsi + rbx)).map_err(|e| e.to_string())?;
                    self.asm.vaddps(ymm0, ymm0, ymm1).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                OpKind::Mul => {
                    self.asm.vmovups(ymm1, ymmword_ptr(rsi + rbx)).map_err(|e| e.to_string())?;
                    self.asm.vmulps(ymm0, ymm0, ymm1).map_err(|e| e.to_string())?;
                    Ok(ymm0)
                }
                _ => {
                    Ok(ymm0)
                }
            }
        }
        /// Emit GEMM with fused epilogue (activation injected before store).
        ///
        /// Generates the full GEMM microkernel, then applies epilogue ops
        /// (bias add, activation) on each accumulator register before storing
        /// to C. Epilogue bodies are looked up from the `ScalarOpRegistry`
        /// as `TraceOp` SSA traces, eliminating hand-coded per-OpKind logic.
        fn emit_gemm_with_epilogue(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            _alloc: &BufferAllocation,
            profile: &DeviceProfile,
            registry: Option<&ScalarOpRegistry>,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor")?;

            let mut epilogue_bodies: Vec<&[TraceOp]> = Vec::new();
            if let Some(reg) = registry {
                for &epi_id in &group.epilogue {
                    let epi_op = graph.op(epi_id).ok_or("missing epilogue op")?;
                    let key = ScalarOpRegistry::key_from_op_kind(&epi_op.kind);
                    if let Some(trace) = reg.get_trace(&key) {
                        if let Some(body) = trace.pattern.body() {
                            epilogue_bodies.push(body);
                        }
                    }
                }
            }
            let epi_refs: Vec<&[TraceOp]> = epilogue_bodies.iter().map(|s| *s).collect();
            eprintln!("[JIT-DBG] emit_gemm_with_epilogue: group {} anchor={:?} op={:?} epilogue_count={} epi_ops={:?}",
                group.id, group.anchor, op.kind, group.epilogue.len(),
                group.epilogue.iter().map(|&id| graph.op(id).map(|o| format!("{:?}", o.kind))).collect::<Vec<_>>());

            match &op.kind {
                OpKind::Gemm { m, n, k } => {
                    self.emit_gemm_microkernel(*m, *n, *k, profile, &epi_refs)?;
                }
                OpKind::GemmBias { m, n, k } => {
                    eprintln!("[JIT-DBG] GemmBias group {} emitting plain GEMM {}x{}x{}", group.id, m, n, k);
                    // Save rdx (bias ptr) before GEMM since the microkernel clobbers it
                    self.asm.mov(qword_ptr(rbp - 48), rdx).map_err(|e| e.to_string())?;
                    self.emit_gemm_microkernel(*m, *n, *k, profile, &[])?;
                    // Restore rdx (bias ptr) and apply bias addition
                    self.asm.mov(rdx, qword_ptr(rbp - 48)).map_err(|e| e.to_string())?;
                    self.emit_bias_add(*m, *n)?;
                }
                OpKind::QuantGemm { m, n, k, block_size, bits } => {
                    self.emit_quant_gemm(*m, *n, *k, *block_size, *bits, profile, &epi_refs)?;
                }
                _ => {}
            }
            Ok(())
        }

        /// Emit `reg = min(cap, total - counter)`.
        ///
        /// Computes `reg = total - counter_reg`, then clamps to `cap`.
        /// Used for runtime tail handling: `kc_cur = min(KC, k - pc)`.
        fn emit_runtime_min(
            &mut self,
            dst: iced_x86::code_asm::AsmRegister64,
            total: usize,
            cap: usize,
            counter: iced_x86::code_asm::AsmRegister64,
        ) -> Result<(), String> {
            self.asm.mov(dst, total as u64).map_err(|e| e.to_string())?;
            self.asm.sub(dst, counter).map_err(|e| e.to_string())?;
            self.asm.cmp(dst, cap as i32).map_err(|e| e.to_string())?;
            let mut skip = self.asm.create_label();
            self.asm.jle(skip).map_err(|e| e.to_string())?;
            self.asm.mov(dst, cap as u64).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut skip).map_err(|e| e.to_string())?;
            Ok(())
        }

        // ── JIT-inlined pack_b ──────────────────────────────────────
        //
        // Replaces the external `gllm_pack_b_f32` call with inline JIT code.
        //
        // Pack B from row-major [kc_cur, nc_cur] into panel-major layout:
        //   For each NR-wide column strip j (0..nc_cur step NR):
        //     For each row k (0..kc_cur):
        //       Copy NR floats from B[k, j..j+NR] to packed_b contiguously.
        //   Remainder strip (nc_cur % NR): zero-padded to NR width.
        //
        // On entry:
        //   r10 = B source base ptr (B + pc*ldb + jc, byte address)
        //   r11 = packed_b destination base ptr
        //   [rsp + loc_kc] = kc_cur
        //   [rsp + loc_nc] = nc_cur
        //
        // Clobbers: rax, rcx, rdx, r9, r10, r11, r15, ymm0, ymm1
        // Preserves: rdi, rsi, r8, r12, r13, r14, rbx, rbp
        fn emit_jit_pack_b_avx2(
            &mut self,
            ldb: usize,
            nr: usize,
            pack_b_off: i32,
            loc_kc: i32,
            loc_nc: i32,
        ) -> Result<(), String> {
            use iced_x86::code_asm::*;

            let ldb_bytes = (ldb * 4) as i32;
            let nr_bytes = (nr * 4) as i32;

            // ── j loop: iterate over full NR-wide strips ──
            let mut j_loop = self.asm.create_label();
            let mut j_rem = self.asm.create_label();
            let mut j_end = self.asm.create_label();

            // r9 = j (column offset in elements)
            self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
            // r15 = dst pointer (advances through packed_b)
            self.asm.mov(r15, r11).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut j_loop).map_err(|e| e.to_string())?;

            // if j + NR > nc_cur, go to remainder
            self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
            self.asm.add(rax, nr as i32).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
            self.asm.jg(j_rem).map_err(|e| e.to_string())?;

            // ── k loop for full NR panel: vectorized copy ──
            {
                let mut k_loop = self.asm.create_label();
                let mut k_done = self.asm.create_label();

                // rdx = B source for this strip = r10 + j * 4
                self.asm.lea(rdx, qword_ptr(r10 + r9 * 4)).map_err(|e| e.to_string())?;
                // rcx = k counter
                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(k_done).map_err(|e| e.to_string())?;

                // Load NR floats from B[k, j..j+NR] using nr/8 ymm registers
                let nr_vecs_local = nr / 8;
                for v in 0..nr_vecs_local {
                    let ymm_reg = Self::ymm_for_index(v)?;
                    let off = (v as i32) * 32;
                    self.asm.vmovups(ymm_reg, ymmword_ptr(rdx + off)).map_err(|e| e.to_string())?;
                }

                // Store to packed_b
                for v in 0..nr_vecs_local {
                    let ymm_reg = Self::ymm_for_index(v)?;
                    let off = (v as i32) * 32;
                    self.asm.vmovups(ymmword_ptr(r15 + off), ymm_reg).map_err(|e| e.to_string())?;
                }

                // Advance: src += ldb * 4, dst += NR * 4
                self.asm.add(rdx, ldb_bytes).map_err(|e| e.to_string())?;
                self.asm.add(r15, nr_bytes).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;
            }

            self.asm.add(r9, nr as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(j_loop).map_err(|e| e.to_string())?;

            // ── Remainder strip: nc_cur % NR columns, zero-padded ──
            self.asm.set_label(&mut j_rem).map_err(|e| e.to_string())?;

            // rax = nc_rem = nc_cur - j
            self.asm.mov(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, 0i32).map_err(|e| e.to_string())?;
            self.asm.jle(j_end).map_err(|e| e.to_string())?;

            // Save nc_rem in r9 (reuse, no longer need j)
            self.asm.mov(r9, rax).map_err(|e| e.to_string())?;

            // rdx = B source for remainder strip = r10 + (nc_cur - nc_rem) * 4
            self.asm.mov(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(r10 + rax * 4)).map_err(|e| e.to_string())?;

            {
                let mut rk_loop = self.asm.create_label();
                let mut rk_done = self.asm.create_label();

                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(rk_done).map_err(|e| e.to_string())?;

                // Zero the destination NR floats
                self.asm.vxorps(ymm0, ymm0, ymm0).map_err(|e| e.to_string())?;
                let nr_vecs_local = nr / 8;
                for v in 0..nr_vecs_local {
                    let off = (v as i32) * 32;
                    self.asm.vmovups(ymmword_ptr(r15 + off), ymm0).map_err(|e| e.to_string())?;
                }

                // Scalar copy of nc_rem floats
                {
                    let mut sc_loop = self.asm.create_label();
                    let mut sc_done = self.asm.create_label();

                    self.asm.xor(rax, rax).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_loop).map_err(|e| e.to_string())?;
                    self.asm.cmp(rax, r9).map_err(|e| e.to_string())?;
                    self.asm.jge(sc_done).map_err(|e| e.to_string())?;

                    self.asm.vmovss(xmm0, dword_ptr(rdx + rax * 4)).map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r15 + rax * 4), xmm0).map_err(|e| e.to_string())?;

                    self.asm.inc(rax).map_err(|e| e.to_string())?;
                    self.asm.jmp(sc_loop).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_done).map_err(|e| e.to_string())?;
                }

                self.asm.add(rdx, ldb_bytes).map_err(|e| e.to_string())?;
                self.asm.add(r15, nr_bytes).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(rk_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_done).map_err(|e| e.to_string())?;
            }

            // nop separates rk_done from j_end so iced-x86 doesn't see two labels on one insn
            self.asm.nop().map_err(|e| e.to_string())?;
            self.asm.set_label(&mut j_end).map_err(|e| e.to_string())?;

            Ok(())
        }

        // ── JIT-inlined pack_a ──────────────────────────────────────
        //
        // Replaces the external `gllm_pack_a_f32` call with inline JIT code.
        //
        // Pack A from row-major [mc_cur, kc_cur] into panel-major layout:
        //   For each MR-tall row strip i (0..mc_cur step MR):
        //     For each column k (0..kc_cur):
        //       Copy MR floats from A[i..i+MR, k] (column-gather, stride=lda)
        //       to packed_a contiguously.
        //   Remainder strip (mc_cur % MR): zero-padded to MR height.
        //
        // On entry:
        //   r15 = A source base ptr (A + ic*lda + pc, byte address)
        //   rbx = scratchpad base
        //   [rsp + loc_kc] = kc_cur
        //   [rsp + loc_mc] = mc_cur
        //
        // Clobbers: rax, rcx, rdx, r9, r10, r11, r15, ymm0..ymm5
        // Preserves: rdi, rsi, r8, r12, r13, r14, rbx, rbp
        fn emit_jit_pack_a_avx2(
            &mut self,
            lda: usize,
            mr: usize,
            pack_a_off: i32,
            loc_kc: i32,
            loc_mc: i32,
        ) -> Result<(), String> {
            use iced_x86::code_asm::*;

            let lda_bytes = (lda * 4) as i32;

            // ── i loop: iterate over full MR-tall strips ──
            let mut i_loop = self.asm.create_label();
            let mut i_rem = self.asm.create_label();
            let mut i_end = self.asm.create_label();

            // r9 = i (row offset in elements)
            self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
            // r11 = dst pointer (advances through packed_a)
            self.asm.lea(r11, qword_ptr(rbx + pack_a_off)).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut i_loop).map_err(|e| e.to_string())?;

            // if i + MR > mc_cur, go to remainder
            self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
            self.asm.add(rax, mr as i32).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.jg(i_rem).map_err(|e| e.to_string())?;

            // ── k loop for full MR panel: gather MR rows ──
            {
                let mut k_loop = self.asm.create_label();
                let mut k_done = self.asm.create_label();

                // r10 = A source for this strip = r15 + i * lda * 4
                self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, lda_bytes).map_err(|e| e.to_string())?;
                self.asm.lea(r10, qword_ptr(r15 + rax)).map_err(|e| e.to_string())?;

                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(k_done).map_err(|e| e.to_string())?;

                // Gather MR=6 floats from column k of rows i..i+MR
                // A[i+0, k], A[i+1, k], ..., A[i+5, k]
                // Each row is lda_bytes apart
                // Use scalar loads since rows are non-contiguous
                for row in 0..mr {
                    let row_off = (row as i32) * lda_bytes;
                    self.asm.vmovss(xmm0, dword_ptr(r10 + row_off)).map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r11 + (row * 4) as i32), xmm0).map_err(|e| e.to_string())?;
                }

                // Advance: src += 4 (next column), dst += MR * 4
                self.asm.add(r10, 4i32).map_err(|e| e.to_string())?;
                self.asm.add(r11, (mr * 4) as i32).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;
            }

            self.asm.add(r9, mr as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(i_loop).map_err(|e| e.to_string())?;

            // ── Remainder strip: mc_cur % MR rows, zero-padded ──
            self.asm.set_label(&mut i_rem).map_err(|e| e.to_string())?;

            // rax = mc_rem = mc_cur - i
            self.asm.mov(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.cmp(rax, 0i32).map_err(|e| e.to_string())?;
            self.asm.jle(i_end).map_err(|e| e.to_string())?;

            // Save mc_rem in r9 (reuse)
            self.asm.mov(r9, rax).map_err(|e| e.to_string())?;

            // r10 = A source for remainder = r15 + (mc_cur - mc_rem) * lda * 4
            self.asm.mov(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.sub(rax, r9).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, lda_bytes).map_err(|e| e.to_string())?;
            self.asm.lea(r10, qword_ptr(r15 + rax)).map_err(|e| e.to_string())?;

            {
                let mut rk_loop = self.asm.create_label();
                let mut rk_done = self.asm.create_label();

                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.jge(rk_done).map_err(|e| e.to_string())?;

                // Zero the destination MR floats
                self.asm.vxorps(ymm0, ymm0, ymm0).map_err(|e| e.to_string())?;
                // Zero MR floats (MR=6 → 24 bytes, use 1 ymm + partial)
                // Just zero all MR slots individually for simplicity
                for row in 0..mr {
                    self.asm.vmovss(dword_ptr(r11 + (row * 4) as i32), xmm0).map_err(|e| e.to_string())?;
                }

                // Scalar copy of mc_rem rows
                {
                    let mut sc_loop = self.asm.create_label();
                    let mut sc_done = self.asm.create_label();

                    self.asm.xor(rax, rax).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_loop).map_err(|e| e.to_string())?;
                    self.asm.cmp(rax, r9).map_err(|e| e.to_string())?;
                    self.asm.jge(sc_done).map_err(|e| e.to_string())?;

                    // Load A[rem_row, k] = r10 + rax * lda_bytes
                    self.asm.mov(rdx, rax).map_err(|e| e.to_string())?;
                    self.asm.imul_3(rdx, rdx, lda_bytes).map_err(|e| e.to_string())?;
                    self.asm.vmovss(xmm1, dword_ptr(r10 + rdx)).map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r11 + rax * 4), xmm1).map_err(|e| e.to_string())?;

                    self.asm.inc(rax).map_err(|e| e.to_string())?;
                    self.asm.jmp(sc_loop).map_err(|e| e.to_string())?;

                    self.asm.set_label(&mut sc_done).map_err(|e| e.to_string())?;
                }

                self.asm.add(r10, 4i32).map_err(|e| e.to_string())?;
                self.asm.add(r11, (mr * 4) as i32).map_err(|e| e.to_string())?;
                self.asm.inc(rcx).map_err(|e| e.to_string())?;
                self.asm.jmp(rk_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut rk_done).map_err(|e| e.to_string())?;
            }

            // nop separates rk_done from i_end so iced-x86 doesn't see two labels on one insn
            self.asm.nop().map_err(|e| e.to_string())?;
            self.asm.set_label(&mut i_end).map_err(|e| e.to_string())?;

            Ok(())
        }


        /// Emit a GEMM microkernel using the BLIS 5-level blocking scheme.
        ///
        /// For small matrices that fit within a single block (k ≤ kc, m ≤ mc,
        /// n ≤ nc), falls back to the direct unpacked tile loop (no packing
        /// overhead). Otherwise emits the full BLIS loop nest:
        ///
        /// ```text
        /// Loop 5 (jc over N, step NC):
        ///   Loop 4 (pc over K, step KC):
        ///     pack_b: B[pc..pc+kc, jc..jc+nc] → packed_b
        ///     Loop 3 (ic over M, step MC):
        ///       pack_a: A[ic..ic+mc, pc..pc+kc] → packed_a
        ///       Loop 2 (jr over NC, step NR):
        ///         Loop 1 (ir over MC, step MR):
        ///           emit_gemm_tile_packed(MR×NR, kc)
        /// ```
        ///
        /// Scratchpad layout (from [rbp+32]):
        ///   [0 .. mc*kc*4)           = packed_a buffer
        ///   [mc*kc*4 .. +kc*nc*4)    = packed_b buffer
        ///
        /// GPR plan (BLIS path):
        /// - rdi=A base, rsi=B base, r8=C base (ABI args, saved across calls)
        /// - rbx=scratchpad base (from [rbp+32])
        /// - r12=jc (NC counter), r13=pc (KC counter), r14=ic (MC counter)
        /// - r15=packed_a ptr, r11=packed_b ptr, r10=C tile ptr (inner loops)
        /// - r9=jr (NR counter), rcx=K counter (tile), rdx=packed_b ptr (tile)
        ///
        /// Stack locals (allocated via sub rsp, 32):
        /// - [rsp+0]  = nc_cur (runtime min(nc, n-jc))
        /// - [rsp+8]  = kc_cur (runtime min(kc, k-pc))
        /// - [rsp+16] = mc_cur (runtime min(mc, m-ic))
        /// - [rsp+24] = (reserved / alignment)
        /// Check whether any epilogue body references `Input(idx)` with idx > 0,
        /// indicating an external tensor (e.g. bias vector) that must be loaded
        /// from the rdx pointer saved at `[rbp-48]`.
        fn epilogue_has_external_input(bodies: &[&[TraceOp]]) -> bool {
            bodies.iter().any(|body| {
                body.iter().any(|op| matches!(op, TraceOp::Input(i) if *i > 0))
            })
        }

        pub fn emit_gemm_microkernel(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            profile: &DeviceProfile,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let blocking = profile.gemm_blocking(m, n, k);
            let mr = blocking.mr;
            let nr = blocking.nr;
            let kc = blocking.kc;
            let mc = blocking.mc;
            let nc = blocking.nc;
            let simd_w = 8usize; // AVX2: 8 f32 per ymm
            let nr_vecs = nr / simd_w;
            let num_acc = mr * nr_vecs;

            if num_acc + 1 > 13 {
                return Err(format!(
                    "GEMM {}x{} microkernel needs {} accumulators + 1 scratch, \
                     exceeds 13 usable ymm (ymm13-15 reserved)",
                    mr, nr, num_acc
                ));
            }

            // If epilogue references Input(1) (external tensor like bias),
            // save rdx (3rd ABI arg) to [rbp-48] so the epilogue can load from it.
            if Self::epilogue_has_external_input(epilogue_bodies) {
                self.asm.mov(qword_ptr(rbp - 48), rdx).map_err(|e| e.to_string())?;
                self.bias_saved = true;
            }

            if m == 1 {
                eprintln!("GEMV 1x{}x{}: direct path (mr=1, nr={})", n, k, nr);
                return self.emit_gemm_microkernel_direct(1, n, k, 1, nr, nr_vecs, simd_w, epilogue_bodies);
            }

            eprintln!("GEMM {}x{}x{}: mr={} nr={} kc={} mc={} nc={} → {}", m, n, k, mr, nr, kc, mc, nc,
                if k <= kc && m <= mc && n <= nc { "DIRECT" } else { "BLIS" });
            if k <= kc && m <= mc && n <= nc {
                return self.emit_gemm_microkernel_direct(m, n, k, mr, nr, nr_vecs, simd_w, epilogue_bodies);
            }

            self.emit_gemm_blis(m, n, k, mr, nr, nr_vecs, kc, mc, nc, epilogue_bodies)?;

            Ok(())
        }

        /// Emit a post-GEMM bias addition: C[m,n] += bias[n] (broadcast across rows).
        /// Expects: r8 = output ptr, rdx = bias ptr.
        fn emit_bias_add(&mut self, m: usize, n: usize) -> Result<(), String> {
            use iced_x86::code_asm::*;
            let row_bytes = (n * 4) as i32;
            let vec_count = n / 8;
            let tail = n % 8;

            // rax = row counter
            self.asm.xor(eax, eax).map_err(|e| e.to_string())?;

            let mut row_loop = self.asm.create_label();
            let mut row_done = self.asm.create_label();

            self.asm.set_label(&mut row_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(eax, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(row_done).map_err(|e| e.to_string())?;

            // rcx = row offset = rax * row_bytes
            self.asm.mov(ecx, eax).map_err(|e| e.to_string())?;
            self.asm.imul_3(ecx, eax, row_bytes).map_err(|e| e.to_string())?;

            // Vectorized bias add for this row
            for v in 0..vec_count {
                let bias_off = (v * 32) as i32;
                let out_off = bias_off;
                // ymm0 = bias[v*8..v*8+8]
                self.asm.vmovups(ymm0, ymmword_ptr(rdx + bias_off)).map_err(|e| e.to_string())?;
                // ymm1 = C[row, v*8..v*8+8]
                self.asm.vaddps(ymm1, ymm0, ymmword_ptr(r8 + rcx + out_off)).map_err(|e| e.to_string())?;
                // Store back
                self.asm.vmovups(ymmword_ptr(r8 + rcx + out_off), ymm1).map_err(|e| e.to_string())?;
            }

            // Scalar tail
            for t in 0..tail {
                let off = ((vec_count * 8 + t) * 4) as i32;
                self.asm.vmovss(xmm0, dword_ptr(rdx + off)).map_err(|e| e.to_string())?;
                self.asm.vaddss(xmm0, xmm0, dword_ptr(r8 + rcx + off)).map_err(|e| e.to_string())?;
                self.asm.vmovss(dword_ptr(r8 + rcx + off), xmm0).map_err(|e| e.to_string())?;
            }

            self.asm.inc(eax).map_err(|e| e.to_string())?;
            self.asm.jmp(row_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut row_done).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit the BLIS 5-level loop nest for large GEMMs.
        ///
        /// Separated from `emit_gemm_microkernel` for clarity. All blocking
        /// parameters are compile-time constants; runtime min() is emitted
        /// for tail handling.
        ///
        /// Stack locals layout (32 bytes, allocated via `sub rsp, 32`):
        /// - `[rsp+0]`  = nc_cur
        /// - `[rsp+8]`  = kc_cur
        /// - `[rsp+16]` = mc_cur
        /// - `[rsp+24]` = first_kc (1 if pc==0, 0 otherwise)
        fn emit_gemm_blis(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            mr: usize,
            nr: usize,
            nr_vecs: usize,
            kc: usize,
            mc: usize,
            nc: usize,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let c_row_bytes = (n * 4) as i32;
            let lda = k;  // A is row-major [m, k], stride = k
            let ldb = n;  // B is row-major [k, n], stride = n

            let pack_a_panels = (mc + mr - 1) / mr;
            let pack_a_bytes = pack_a_panels * mr * kc * 4;
            let pack_b_panels = (nc + nr - 1) / nr;
            let pack_b_bytes = pack_b_panels * nr * kc * 4;
            let blis_bytes = pack_a_bytes + pack_b_bytes;
            let total_extra = (self.blis_scratchpad_offset - self.blis_base_offset) + blis_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(total_extra);
            eprintln!("[JIT-DBG] BLIS {}x{}x{}: pack_a_off={}, pack_b_off={}, blis_base={}, blis_offset={}, total_extra={}, blis_bytes={}",
                m, n, k, self.blis_scratchpad_offset, self.blis_scratchpad_offset + pack_a_bytes,
                self.blis_base_offset, self.blis_scratchpad_offset, total_extra, blis_bytes);

            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            let pack_a_off = self.blis_scratchpad_offset as i32;
            let pack_b_off = (self.blis_scratchpad_offset + pack_a_bytes) as i32;

            self.asm.sub(rsp, 32i32).map_err(|e| e.to_string())?;

            let loc_nc: i32 = 0;
            let loc_kc: i32 = 8;
            let loc_mc: i32 = 16;

            let mut jc_loop = self.asm.create_label();
            let mut jc_done = self.asm.create_label();

            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, n as i32).map_err(|e| e.to_string())?;
            self.asm.jge(jc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, n, nc, r12)?;
            self.asm.mov(qword_ptr(rsp + loc_nc), rax).map_err(|e| e.to_string())?;

            let mut pc_loop = self.asm.create_label();
            let mut pc_done = self.asm.create_label();

            self.asm.xor(r13, r13).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13, k as i32).map_err(|e| e.to_string())?;
            self.asm.jge(pc_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, k, kc, r13)?;
            self.asm.mov(qword_ptr(rsp + loc_kc), rax).map_err(|e| e.to_string())?;

            // ── JIT pack_b: B[pc..pc+kc, jc..jc+nc] → packed_b ──
            {
                // r10 = &B[pc, jc] = rsi + (r13 * ldb + r12) * 4
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, ldb as i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, r12).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, rsi).map_err(|e| e.to_string())?;
                self.asm.mov(r10, rax).map_err(|e| e.to_string())?;

                // r11 = packed_b destination
                self.asm.lea(r11, qword_ptr(rbx + pack_b_off)).map_err(|e| e.to_string())?;

                self.emit_jit_pack_b_avx2(ldb, nr, pack_b_off, loc_kc, loc_nc)?;
            }

            let mut ic_loop = self.asm.create_label();
            let mut ic_done = self.asm.create_label();

            self.asm.xor(r14, r14).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r14, m as i32).map_err(|e| e.to_string())?;
            self.asm.jge(ic_done).map_err(|e| e.to_string())?;

            self.emit_runtime_min(rax, m, mc, r14)?;
            self.asm.mov(qword_ptr(rsp + loc_mc), rax).map_err(|e| e.to_string())?;

            // ── JIT pack_a: A[ic..ic+mc, pc..pc+kc] → packed_a ──
            {
                // r15 = &A[ic, pc] = rdi + (r14 * lda + r13) * 4
                self.asm.mov(rax, r14).map_err(|e| e.to_string())?;
                self.asm.imul_3(rax, rax, lda as i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, r13).map_err(|e| e.to_string())?;
                self.asm.shl(rax, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(rax, rdi).map_err(|e| e.to_string())?;
                self.asm.mov(r15, rax).map_err(|e| e.to_string())?;

                self.emit_jit_pack_a_avx2(lda, mr, pack_a_off, loc_kc, loc_mc)?;
            }

            {
                let m_rem = m % mr;
                let n_rem = n % nr;
                let simd_w = nr / nr_vecs;  // 8 for AVX2
                let n_rem_vecs = n_rem / simd_w;  // 0 if n % nr == 0

                let loc_ir: i32 = 24;

                let mut jr_loop = self.asm.create_label();
                let mut jr_done = self.asm.create_label();
                let mut jr_rem_start = self.asm.create_label();

                self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_loop).map_err(|e| e.to_string())?;

                self.asm.mov(rax, r9).map_err(|e| e.to_string())?;
                self.asm.add(rax, nr as i32).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                self.asm.jg(jr_rem_start).map_err(|e| e.to_string())?;

                self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                if pack_b_off != 0 {
                    self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                }

                self.emit_blis_ir_loop(
                    mr, nr_vecs, m_rem, loc_kc, n, nr_vecs, nr,
                    loc_ir, loc_mc, pack_a_off, c_row_bytes,
                    epilogue_bodies, k,
                )?;

                self.asm.add(r9, nr as i32).map_err(|e| e.to_string())?;
                self.asm.jmp(jr_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut jr_rem_start).map_err(|e| e.to_string())?;

                if n_rem_vecs > 0 {
                    self.asm.cmp(r9, qword_ptr(rsp + loc_nc)).map_err(|e| e.to_string())?;
                    self.asm.jge(jr_done).map_err(|e| e.to_string())?;

                    self.asm.mov(r11, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                    self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                    self.asm.imul_2(r11, r9).map_err(|e| e.to_string())?;
                    self.asm.add(r11, rbx).map_err(|e| e.to_string())?;
                    if pack_b_off != 0 {
                        self.asm.add(r11, pack_b_off).map_err(|e| e.to_string())?;
                    }

                    self.emit_blis_ir_loop(
                        mr, nr_vecs, m_rem, loc_kc, n, n_rem_vecs, nr,
                        loc_ir, loc_mc, pack_a_off, c_row_bytes,
                        epilogue_bodies, k,
                    )?;
                }

                self.asm.nop().map_err(|e| e.to_string())?;
                self.asm.set_label(&mut jr_done).map_err(|e| e.to_string())?;
            }

            self.asm.add(r14, mc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(ic_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ic_done).map_err(|e| e.to_string())?;

            self.asm.add(r13, kc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(pc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut pc_done).map_err(|e| e.to_string())?;

            self.asm.add(r12, nc as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(jc_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut jc_done).map_err(|e| e.to_string())?;

            self.asm.add(rsp, 32i32).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit the ir (M-dimension) loop for one jr strip in the BLIS path.
        ///
        /// Iterates ir = 0..mc_cur (step MR), emitting full MR tiles, then
        /// one M-remainder tile if `m_rem != 0`.
        ///
        /// Expects on entry:
        ///   r11 = packed_b ptr for this jr strip
        ///   r9  = jr (current N offset within nc_cur)
        ///   r12 = jc, r13 = pc, r14 = ic
        ///   r8  = C base ptr
        ///   rbx = scratchpad base
        ///   [rsp + loc_mc] = mc_cur
        ///   [rsp + loc_nc] = nc_cur (unused here but on stack)
        ///
        /// `tile_nr_vecs`: number of ymm vectors per row for this jr strip
        ///   (nr_vecs for full tiles, n_rem_vecs for the N remainder strip)
        fn emit_blis_ir_loop(
            &mut self,
            mr: usize,
            nr_vecs: usize,       // full NR in vectors (for packed_a panel width)
            m_rem: usize,         // m % mr, 0 if no M remainder
            loc_kc: i32,
            n: usize,             // full N dimension (for C row stride)
            tile_nr_vecs: usize,  // actual nr_vecs for this strip's tiles
            nr: usize,            // full NR (packed_b stride per K step)
            loc_ir: i32,
            loc_mc: i32,
            pack_a_off: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
            k_total: usize,
        ) -> Result<(), String> {
            let mut ir_loop = self.asm.create_label();
            let mut ir_done = self.asm.create_label();
            let mut ir_rem_start = self.asm.create_label();

            self.asm.mov(qword_ptr(rsp + loc_ir), 0i32).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ir_loop).map_err(|e| e.to_string())?;

            self.asm.mov(rax, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
            self.asm.mov(rcx, rax).map_err(|e| e.to_string())?;
            self.asm.add(rcx, mr as i32).map_err(|e| e.to_string())?;
            self.asm.cmp(rcx, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
            self.asm.jg(ir_rem_start).map_err(|e| e.to_string())?;

            self.emit_blis_tile_setup(mr, loc_kc, n, pack_a_off)?;

            self.emit_blis_tile_with_first_kc_branch(mr, tile_nr_vecs, loc_kc, c_row_bytes, mr, nr, epilogue_bodies, k_total)?;

            self.asm.mov(rax, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
            self.asm.add(rax, mr as i32).map_err(|e| e.to_string())?;
            self.asm.mov(qword_ptr(rsp + loc_ir), rax).map_err(|e| e.to_string())?;
            self.asm.jmp(ir_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut ir_rem_start).map_err(|e| e.to_string())?;

            if m_rem > 0 {
                self.asm.mov(rax, qword_ptr(rsp + loc_ir)).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, qword_ptr(rsp + loc_mc)).map_err(|e| e.to_string())?;
                self.asm.jge(ir_done).map_err(|e| e.to_string())?;

                self.emit_blis_tile_setup(mr, loc_kc, n, pack_a_off)?;

                self.emit_blis_tile_with_first_kc_branch(m_rem, tile_nr_vecs, loc_kc, c_row_bytes, mr, nr, epilogue_bodies, k_total)?;
            }

            self.asm.nop().map_err(|e| e.to_string())?;
            self.asm.set_label(&mut ir_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Set up r15 (packed_a ptr) and r10 (C tile ptr) for a BLIS tile.
        ///
        /// Expects rax = ir, r14 = ic, r12 = jc, r9 = jr, r8 = C base,
        /// rbx = scratchpad base.
        fn emit_blis_tile_setup(
            &mut self,
            _mr: usize,
            loc_kc: i32,
            n: usize,
            pack_a_off: i32,
        ) -> Result<(), String> {
            self.asm.mov(r15, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
            self.asm.shl(r15, 2i32).map_err(|e| e.to_string())?;
            self.asm.imul_2(r15, rax).map_err(|e| e.to_string())?;
            self.asm.add(r15, rbx).map_err(|e| e.to_string())?;
            if pack_a_off != 0 {
                self.asm.add(r15, pack_a_off).map_err(|e| e.to_string())?;
            }

            self.asm.mov(r10, r14).map_err(|e| e.to_string())?;
            self.asm.add(r10, rax).map_err(|e| e.to_string())?;  // ic + ir
            self.asm.imul_3(r10, r10, (n * 4) as i32).map_err(|e| e.to_string())?;
            self.asm.mov(rcx, r12).map_err(|e| e.to_string())?;
            self.asm.add(rcx, r9).map_err(|e| e.to_string())?;
            self.asm.shl(rcx, 2i32).map_err(|e| e.to_string())?;
            self.asm.add(r10, rcx).map_err(|e| e.to_string())?;
            self.asm.add(r10, r8).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Emit a BLIS tile with runtime first_kc_block branching.
        ///
        /// Branches on r13 (pc): if r13 == 0, zero accumulators; else load from C.
        fn emit_blis_tile_with_first_kc_branch(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            loc_kc: i32,
            c_row_bytes: i32,
            packed_mr: usize,
            packed_nr: usize,
            epilogue_bodies: &[&[TraceOp]],
            k_total: usize,
        ) -> Result<(), String> {
            let mut not_first = self.asm.create_label();
            let mut tile_done = self.asm.create_label();

            self.asm.test(r13, r13).map_err(|e| e.to_string())?;
            self.asm.jnz(not_first).map_err(|e| e.to_string())?;

            self.emit_gemm_tile_packed(tile_mr, tile_nr_vecs, loc_kc, c_row_bytes, true, packed_mr, packed_nr, epilogue_bodies, k_total)?;
            self.asm.jmp(tile_done).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut not_first).map_err(|e| e.to_string())?;
            self.emit_gemm_tile_packed(tile_mr, tile_nr_vecs, loc_kc, c_row_bytes, false, packed_mr, packed_nr, epilogue_bodies, k_total)?;

            self.asm.set_label(&mut tile_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit the direct (unpacked) GEMM microkernel for small matrices.
        ///
        /// Used when the entire problem fits in a single MC×KC×NC block,
        /// so packing overhead is not worthwhile. Generates a two-level
        /// tile loop (N-outer, M-inner) reading A and B directly.
        ///
        /// GPR plan (same as original microkernel):
        /// - rdi=A, rsi=B, r8=C (from calling convention)
        /// - r15=A_row_ptr, r11=B_col_ptr, r10=C_tile_ptr
        /// - r14=j (N counter), r13=i (M counter), r12=p (K counter)
        /// - r9=B_k_ptr (advances each K step)
        fn emit_gemm_microkernel_direct(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            mr: usize,
            nr: usize,
            nr_vecs: usize,
            simd_w: usize,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let a_row_bytes = (k * 4) as i32;
            let b_row_bytes = (n * 4) as i32;
            let c_row_bytes = (n * 4) as i32;

            let m_full = (m / mr) * mr;
            let m_rem = m % mr;
            let n_full = (n / nr) * nr;
            let n_rem = n % nr;
            let n_rem_vecs = n_rem / simd_w;

            if n_full > 0 {
                let mut n_loop = self.asm.create_label();
                let mut n_done = self.asm.create_label();

                self.asm.xor(r14, r14).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut n_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(r14, n_full as i32).map_err(|e| e.to_string())?;
                self.asm.jge(n_done).map_err(|e| e.to_string())?;

                self.asm.mov(r11, r14).map_err(|e| e.to_string())?;
                self.asm.shl(r11, 2i32).map_err(|e| e.to_string())?;
                self.asm.add(r11, rsi).map_err(|e| e.to_string())?;

                if m_full > 0 {
                    self.emit_gemm_m_loop(
                        mr, nr_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }
                if m_rem > 0 {
                    self.emit_gemm_m_remainder(
                        m_rem, nr_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }

                self.asm.add(r14, nr as i32).map_err(|e| e.to_string())?;
                self.asm.jmp(n_loop).map_err(|e| e.to_string())?;

                self.asm.set_label(&mut n_done).map_err(|e| e.to_string())?;
            }

            if n_rem_vecs > 0 {
                self.asm.mov(r11, rsi).map_err(|e| e.to_string())?;
                if n_full > 0 {
                    self.asm.add(r11, (n_full * 4) as i32).map_err(|e| e.to_string())?;
                }
                self.asm.mov(r14, n_full as i64).map_err(|e| e.to_string())?;

                if m_full > 0 {
                    self.emit_gemm_m_loop(
                        mr, n_rem_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }
                if m_rem > 0 {
                    self.emit_gemm_m_remainder(
                        m_rem, n_rem_vecs, k, m_full,
                        a_row_bytes, b_row_bytes, c_row_bytes,
                        epilogue_bodies,
                    )?;
                }
            }

            // Scalar N-tail: handle remaining columns that don't fill a SIMD vector
            let n_scalar_start = n_full + n_rem_vecs * simd_w;
            let n_scalar_rem = n - n_scalar_start;
            if n_scalar_rem > 0 {
                // For each remaining column j in [n_scalar_start, n):
                //   For each row i in [0, m):
                //     C[i,j] = sum_p A[i,p] * B[p,j]
                // Use scalar (ss) instructions.
                for col in 0..n_scalar_rem {
                    let j = n_scalar_start + col;
                    for row in 0..m {
                        // Compute dot product A[row, :] . B[:, j]
                        // xmm0 = accumulator
                        self.asm.vxorps(xmm0, xmm0, xmm0).map_err(|e| e.to_string())?;
                        for p in 0..k {
                            // A[row, p] at rdi + (row * k + p) * 4
                            let a_off = ((row * k + p) * 4) as i32;
                            // B[p, j] at rsi + (p * n + j) * 4
                            let b_off = ((p * n + j) * 4) as i32;
                            self.asm.vmovss(xmm1, dword_ptr(rdi + a_off)).map_err(|e| e.to_string())?;
                            self.asm.vmovss(xmm2, dword_ptr(rsi + b_off)).map_err(|e| e.to_string())?;
                            self.asm.vfmadd231ss(xmm0, xmm1, xmm2).map_err(|e| e.to_string())?;
                        }
                        // Store C[row, j] at r8 + (row * n + j) * 4
                        let c_off = ((row * n + j) * 4) as i32;
                        self.asm.vmovss(dword_ptr(r8 + c_off), xmm0).map_err(|e| e.to_string())?;
                    }
                }
            }

            Ok(())
        }

        /// Emit the M-dimension loop over full MR-sized tiles.
        ///
        /// Expects r11 = B_col_ptr, r14 = j (N offset in elements).
        /// Uses r15 for A_row_ptr, r10 for C_tile_ptr, r13 for M counter.
        fn emit_gemm_m_loop(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            k: usize,
            m_limit: usize,
            a_row_bytes: i32,
            b_row_bytes: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let mut m_loop = self.asm.create_label();
            let mut m_done = self.asm.create_label();

            self.asm.mov(r15, rdi).map_err(|e| e.to_string())?;
            self.asm.mov(r10, r14).map_err(|e| e.to_string())?;
            self.asm.shl(r10, 2i32).map_err(|e| e.to_string())?;
            self.asm.add(r10, r8).map_err(|e| e.to_string())?;
            self.asm.xor(r13, r13).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut m_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r13, m_limit as i32).map_err(|e| e.to_string())?;
            self.asm.jge(m_done).map_err(|e| e.to_string())?;

            self.emit_gemm_tile(tile_mr, tile_nr_vecs, k, a_row_bytes, b_row_bytes, c_row_bytes, epilogue_bodies)?;

            let mr_a_advance = (tile_mr as i32) * a_row_bytes;
            let mr_c_advance = (tile_mr as i32) * c_row_bytes;
            self.asm.add(r15, mr_a_advance).map_err(|e| e.to_string())?;
            self.asm.add(r10, mr_c_advance).map_err(|e| e.to_string())?;
            self.asm.add(r13, tile_mr as i32).map_err(|e| e.to_string())?;
            self.asm.jmp(m_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut m_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit a single M-remainder tile (fewer than MR rows).
        ///
        /// Sets up r15/r10 for the remainder starting row and emits one tile.
        fn emit_gemm_m_remainder(
            &mut self,
            rem_mr: usize,
            tile_nr_vecs: usize,
            k: usize,
            m_full: usize,
            a_row_bytes: i32,
            b_row_bytes: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            self.asm.mov(r15, rdi).map_err(|e| e.to_string())?;
            if m_full > 0 {
                self.asm.add(r15, (m_full as i32) * a_row_bytes).map_err(|e| e.to_string())?;
            }
            self.asm.mov(r10, r14).map_err(|e| e.to_string())?;
            self.asm.shl(r10, 2i32).map_err(|e| e.to_string())?;
            self.asm.add(r10, r8).map_err(|e| e.to_string())?;
            if m_full > 0 {
                self.asm.add(r10, (m_full as i32) * c_row_bytes).map_err(|e| e.to_string())?;
            }

            self.emit_gemm_tile(rem_mr, tile_nr_vecs, k, a_row_bytes, b_row_bytes, c_row_bytes, epilogue_bodies)
        }

        /// Emit one K-iteration of the FMA body for the direct (unpacked) path.
        ///
        /// Reads A from `[r15 + r12*4 + extra_a_disp + ii*a_row_bytes]` and
        /// B from `[r9 + b_base_disp + v*32]`.
        fn emit_direct_fma_body(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            a_scratch_0: AsmRegisterYmm,
            a_scratch_1: AsmRegisterYmm,
            extra_a_disp: i32,
            a_row_bytes: i32,
            b_base_disp: i32,
        ) -> Result<(), String> {
            self.asm.vbroadcastss(
                a_scratch_0,
                dword_ptr(r15 + r12 * 4 + extra_a_disp),
            ).map_err(|e| e.to_string())?;

            for ii in 0..tile_mr {
                let cur_scratch = if ii % 2 == 0 { a_scratch_0 } else { a_scratch_1 };
                let nxt_scratch = if ii % 2 == 0 { a_scratch_1 } else { a_scratch_0 };

                let acc0 = Self::ymm_for_index(ii * tile_nr_vecs)?;
                self.asm.vfmadd231ps(
                    acc0,
                    cur_scratch,
                    ymmword_ptr(r9 + b_base_disp),
                ).map_err(|e| e.to_string())?;

                if ii + 1 < tile_mr {
                    let a_disp_next = extra_a_disp + ((ii + 1) as i32) * a_row_bytes;
                    self.asm.vbroadcastss(
                        nxt_scratch,
                        dword_ptr(r15 + r12 * 4 + a_disp_next),
                    ).map_err(|e| e.to_string())?;
                }

                for v in 1..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let b_disp = b_base_disp + (v as i32) * 32;
                    self.asm.vfmadd231ps(
                        acc,
                        cur_scratch,
                        ymmword_ptr(r9 + b_disp),
                    ).map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }

        /// Emit one K-iteration of the FMA body for the packed GEMM path.
        ///
        /// Reads A from `[r15 + extra_a_off + ii*4]` and
        /// B from `[rdx + extra_b_off + v*32]`.
        fn emit_packed_fma_body(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            a_scratch_0: AsmRegisterYmm,
            a_scratch_1: AsmRegisterYmm,
            extra_a_off: i32,
            extra_b_off: i32,
        ) -> Result<(), String> {
            self.asm.vbroadcastss(
                a_scratch_0,
                dword_ptr(r15 + extra_a_off),
            ).map_err(|e| e.to_string())?;

            for ii in 0..tile_mr {
                let cur_scratch = if ii % 2 == 0 { a_scratch_0 } else { a_scratch_1 };
                let nxt_scratch = if ii % 2 == 0 { a_scratch_1 } else { a_scratch_0 };

                let acc0 = Self::ymm_for_index(ii * tile_nr_vecs)?;
                self.asm.vfmadd231ps(
                    acc0,
                    cur_scratch,
                    ymmword_ptr(rdx + extra_b_off),
                ).map_err(|e| e.to_string())?;

                if ii + 1 < tile_mr {
                    let a_disp_next = extra_a_off + ((ii + 1) as i32) * 4;
                    self.asm.vbroadcastss(
                        nxt_scratch,
                        dword_ptr(r15 + a_disp_next),
                    ).map_err(|e| e.to_string())?;
                }

                for v in 1..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let b_disp = extra_b_off + (v as i32) * 32;
                    self.asm.vfmadd231ps(
                        acc,
                        cur_scratch,
                        ymmword_ptr(rdx + b_disp),
                    ).map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }

        /// Emit a single MR×NR GEMM tile: zero accumulators, software-pipelined
        /// K-loop (unroll-2, prefetch 2 steps ahead), store results.
        ///
        /// Inputs (set by caller):
        /// - r15 = &A[tile_row, 0]
        /// - r11 = &B[0, tile_col]
        /// - r10 = &C[tile_row, tile_col]
        ///
        /// Clobbers: r9 (B_k_ptr), r12 (K counter), ymm0..ymm(num_acc),
        ///           two ymm scratch registers at indices `num_acc` and `num_acc+1`
        ///           (A broadcast interleaving).
        fn emit_gemm_tile(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            k: usize,
            a_row_bytes: i32,
            b_row_bytes: i32,
            c_row_bytes: i32,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let num_acc = tile_mr * tile_nr_vecs;
            if num_acc + 2 > 16 {
                return Err(format!(
                    "GEMM tile {}x{} needs {} accumulators + 2 scratch, exceeds 16 ymm registers",
                    tile_mr, tile_nr_vecs, num_acc
                ));
            }
            let a_scratch_0 = Self::ymm_any(num_acc)?;
            let a_scratch_1 = Self::ymm_any(num_acc + 1)?;

            for a in 0..num_acc {
                let reg = Self::ymm_for_index(a)?;
                self.asm.vxorps(reg, reg, reg).map_err(|e| e.to_string())?;
            }

            let k_main = (k & !1) as i32;
            let mut k_loop = self.asm.create_label();
            let mut k_tail = self.asm.create_label();
            let mut k_done = self.asm.create_label();

            self.asm.mov(r9, r11).map_err(|e| e.to_string())?;
            self.asm.xor(r12, r12).map_err(|e| e.to_string())?;

            // Main loop: 2 K-iterations per step (software-pipelined)
            self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, k_main).map_err(|e| e.to_string())?;
            self.asm.jge(k_tail).map_err(|e| e.to_string())?;

            // Prefetch B 2 steps ahead
            self.asm.prefetcht0(byte_ptr(r9 + 2 * b_row_bytes)).map_err(|e| e.to_string())?;
            if b_row_bytes > 64 {
                self.asm.prefetcht0(byte_ptr(r9 + 2 * b_row_bytes + 64)).map_err(|e| e.to_string())?;
            }

            // Iteration 0 (K = r12)
            self.emit_direct_fma_body(tile_mr, tile_nr_vecs, a_scratch_0, a_scratch_1, 0, a_row_bytes, 0)?;
            // Iteration 1 (K = r12 + 1)
            self.emit_direct_fma_body(tile_mr, tile_nr_vecs, a_scratch_0, a_scratch_1, 4, a_row_bytes, b_row_bytes)?;

            self.asm.add(r9, 2 * b_row_bytes).map_err(|e| e.to_string())?;
            self.asm.add(r12, 2i32).map_err(|e| e.to_string())?;
            self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

            // Tail: handle K-odd remainder (at most 1 iteration)
            self.asm.set_label(&mut k_tail).map_err(|e| e.to_string())?;
            self.asm.cmp(r12, k as i32).map_err(|e| e.to_string())?;
            self.asm.jge(k_done).map_err(|e| e.to_string())?;

            self.emit_direct_fma_body(tile_mr, tile_nr_vecs, a_scratch_0, a_scratch_1, 0, a_row_bytes, 0)?;
            self.asm.add(r9, b_row_bytes).map_err(|e| e.to_string())?;
            self.asm.inc(r12).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;

            // If epilogue has external input (bias), compute bias tile base
            // pointer into rax: rax = bias_ptr + r14 * 4
            // (r14 = current N offset in elements, r12/r9 are free after K-loop)
            if self.bias_saved && !epilogue_bodies.is_empty() {
                self.asm.mov(rax, qword_ptr(rbp - 48)).map_err(|e| e.to_string())?;
                self.asm.lea(rax, qword_ptr(rax + r14 * 4)).map_err(|e| e.to_string())?;
            }
            self.emit_epilogue_on_accumulators_inner(epilogue_bodies, num_acc, tile_nr_vecs)?;

            for ii in 0..tile_mr {
                for v in 0..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let c_disp = (ii as i32) * c_row_bytes + (v as i32) * 32;
                    self.asm.vmovups(
                        ymmword_ptr(r10 + c_disp),
                        acc,
                    ).map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Emit a single MR×NR GEMM tile reading from packed A/B buffers.
        ///
        /// This is the inner microkernel for the BLIS-style blocked GEMM.
        /// Packed layout:
        /// - packed_a: `[k * MR + ii]` (column-major MR-wide strips)
        /// - packed_b: `[k * NR + v*8]` (row-major NR-wide strips)
        ///
        /// Inputs (set by caller):
        /// - r15 = packed_a base for this MR strip
        /// - r11 = packed_b base for this NR strip
        /// - r10 = &C[tile_row, tile_col]
        ///
        /// Parameters:
        /// - `first_kc_block`: if true, zero accumulators; else load from C
        /// - `kc`: number of K iterations for this KC block
        ///
        /// Clobbers: rcx (K counter), rdx (B_k_ptr), rax (k_main scratch),
        ///           ymm0..ymm(num_acc), two ymm scratch registers at indices
        ///           `num_acc` and `num_acc+1` (A broadcast interleaving).
        /// Uses software-pipelined K-loop (unroll-2, prefetch 2 steps ahead).
        fn emit_gemm_tile_packed(
            &mut self,
            tile_mr: usize,
            tile_nr_vecs: usize,
            loc_kc: i32,
            c_row_bytes: i32,
            first_kc_block: bool,
            packed_mr: usize,     // MR used in packing (for A stride per K step)
            packed_nr: usize,     // NR used in packing (for B stride per K step)
            epilogue_bodies: &[&[TraceOp]],
            k_total: usize,
        ) -> Result<(), String> {
            let num_acc = tile_mr * tile_nr_vecs;
            if num_acc + 2 > 16 {
                return Err(format!(
                    "GEMM tile {}x{} needs {} accumulators + 2 scratch, exceeds 16 ymm registers",
                    tile_mr, tile_nr_vecs, num_acc
                ));
            }
            let a_scratch_0 = Self::ymm_any(num_acc)?;
            let a_scratch_1 = Self::ymm_any(num_acc + 1)?;
            let mr_stride_bytes = (packed_mr as i32) * 4; // packed_a advance per K step
            let nr_stride_bytes = (packed_nr as i32) * 4; // packed_b advance per K step

            if first_kc_block {
                for a in 0..num_acc {
                    let reg = Self::ymm_for_index(a)?;
                    self.asm.vxorps(reg, reg, reg).map_err(|e| e.to_string())?;
                }
            } else {
                for ii in 0..tile_mr {
                    for v in 0..tile_nr_vecs {
                        let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                        let c_disp = (ii as i32) * c_row_bytes + (v as i32) * 32;
                        self.asm.vmovups(acc, ymmword_ptr(r10 + c_disp))
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            let mut k_loop = self.asm.create_label();
            let mut k_tail = self.asm.create_label();
            let mut k_done = self.asm.create_label();

            self.asm.mov(rdx, r11).map_err(|e| e.to_string())?;
            self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;

            // rax = k_main = kc_cur & ~1 (even part for unrolled loop)
            self.asm.mov(rax, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
            self.asm.and(rax, -2i32).map_err(|e| e.to_string())?;

            // Main loop: 2 K-iterations per step (software-pipelined)
            self.asm.set_label(&mut k_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(rcx, rax).map_err(|e| e.to_string())?;
            self.asm.jge(k_tail).map_err(|e| e.to_string())?;

            // Prefetch A/B 2 steps ahead
            self.asm.prefetcht0(byte_ptr(r15 + 2 * mr_stride_bytes)).map_err(|e| e.to_string())?;
            self.asm.prefetcht0(byte_ptr(rdx + 2 * nr_stride_bytes)).map_err(|e| e.to_string())?;
            if nr_stride_bytes > 64 {
                self.asm.prefetcht0(byte_ptr(rdx + 2 * nr_stride_bytes + 64)).map_err(|e| e.to_string())?;
            }

            // Iteration 0 (K = rcx)
            self.emit_packed_fma_body(tile_mr, tile_nr_vecs, a_scratch_0, a_scratch_1, 0, 0)?;
            // Iteration 1 (K = rcx + 1)
            self.emit_packed_fma_body(tile_mr, tile_nr_vecs, a_scratch_0, a_scratch_1, mr_stride_bytes, nr_stride_bytes)?;

            self.asm.add(r15, 2 * mr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rdx, 2 * nr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rcx, 2i32).map_err(|e| e.to_string())?;
            self.asm.jmp(k_loop).map_err(|e| e.to_string())?;

            // Tail: handle K-odd remainder (at most 1 iteration)
            self.asm.set_label(&mut k_tail).map_err(|e| e.to_string())?;
            self.asm.cmp(rcx, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
            self.asm.jge(k_done).map_err(|e| e.to_string())?;

            self.asm.prefetcht0(byte_ptr(r15 + mr_stride_bytes)).map_err(|e| e.to_string())?;
            self.asm.prefetcht0(byte_ptr(rdx + nr_stride_bytes)).map_err(|e| e.to_string())?;
            if nr_stride_bytes > 64 {
                self.asm.prefetcht0(byte_ptr(rdx + nr_stride_bytes + 64)).map_err(|e| e.to_string())?;
            }

            self.emit_packed_fma_body(tile_mr, tile_nr_vecs, a_scratch_0, a_scratch_1, 0, 0)?;

            self.asm.add(r15, mr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rdx, nr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.inc(rcx).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut k_done).map_err(|e| e.to_string())?;

            self.asm.mov(rax, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
            self.asm.imul_3(rax, rax, mr_stride_bytes).map_err(|e| e.to_string())?;
            self.asm.sub(r15, rax).map_err(|e| e.to_string())?;

            if !epilogue_bodies.is_empty() {
                let mut skip_epi = self.asm.create_label();
                self.asm.mov(rax, r13).map_err(|e| e.to_string())?;
                self.asm.add(rax, qword_ptr(rsp + loc_kc)).map_err(|e| e.to_string())?;
                self.asm.cmp(rax, k_total as i32).map_err(|e| e.to_string())?;
                self.asm.jl(skip_epi).map_err(|e| e.to_string())?;
                // If epilogue has external input (bias), compute bias tile base
                // pointer into rax: rax = bias_ptr + (jc + jr) * 4
                // (rcx is free after K-loop; r12 = jc, r9 = jr are outer-loop regs)
                if self.bias_saved {
                    self.asm.mov(rax, qword_ptr(rbp - 48)).map_err(|e| e.to_string())?;
                    self.asm.mov(rcx, r12).map_err(|e| e.to_string())?;
                    self.asm.add(rcx, r9).map_err(|e| e.to_string())?;
                    self.asm.lea(rax, qword_ptr(rax + rcx * 4)).map_err(|e| e.to_string())?;
                }
                self.emit_epilogue_on_accumulators_inner(epilogue_bodies, num_acc, tile_nr_vecs)?;
                self.asm.set_label(&mut skip_epi).map_err(|e| e.to_string())?;
            }

            for ii in 0..tile_mr {
                for v in 0..tile_nr_vecs {
                    let acc = Self::ymm_for_index(ii * tile_nr_vecs + v)?;
                    let c_disp = (ii as i32) * c_row_bytes + (v as i32) * 32;
                    self.asm.vmovups(
                        ymmword_ptr(r10 + c_disp),
                        acc,
                    ).map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// Apply epilogue operations in-place on live accumulator registers
        /// ymm0..ymm(num_acc-1), driven entirely by `TraceOp` bodies.
        ///
        /// Each epilogue body is an SSA trace (from `OpTrace.pattern`).
        /// `Input(0)` in the body refers to the accumulator's current value.
        ///
        /// Strategy: for each accumulator, allocate `body.len() * 32` bytes
        /// of stack for SSA intermediates. Each SSA slot is a 32-byte ymm
        /// spill. We use ymm13-15 as scratch for compute, storing results
        /// back to stack slots. The final SSA value overwrites the accumulator.
        ///
        /// Must be called while accumulators hold GEMM results, BEFORE the
        /// store to C.
        fn emit_epilogue_on_accumulators(
            &mut self,
            epilogue_bodies: &[&[TraceOp]],
            num_acc: usize,
        ) -> Result<(), String> {
            self.emit_epilogue_on_accumulators_inner(epilogue_bodies, num_acc, 1)
        }

        /// Inner implementation that accepts `nr_vecs` so it can compute the
        /// per-vector bias byte offset for `Input(1)` loads.
        ///
        /// When `self.bias_saved` is true and a body contains `Input(1)`,
        /// the bias tile base pointer is expected in `rax` (set by the caller
        /// before invoking this method).  For accumulator index `a`, the
        /// vector index is `a % nr_vecs`, and the bias displacement is
        /// `(a % nr_vecs) * 32` bytes from `rax`.
        fn emit_epilogue_on_accumulators_inner(
            &mut self,
            epilogue_bodies: &[&[TraceOp]],
            num_acc: usize,
            nr_vecs: usize,
        ) -> Result<(), String> {
            if epilogue_bodies.is_empty() || num_acc == 0 {
                return Ok(());
            }

            for body in epilogue_bodies {
                if body.is_empty() {
                    continue;
                }
                let has_ext = body.iter().any(|op| matches!(op, TraceOp::Input(i) if *i > 0));
                if self.use_avx512 {
                    let acc_count_512 = num_acc.min(28); // zmm0-zmm27
                    for a in 0..acc_count_512 {
                        let acc = Self::zmm_for_index(a)?;
                        if has_ext && self.bias_saved {
                            let v = a % nr_vecs;
                            let bias_disp = (v as i32) * 64; // 64 bytes per zmm
                            self.emit_trace_on_accumulator_with_bias_avx512(acc, body, bias_disp)?;
                        } else {
                            self.emit_trace_on_accumulator_avx512(acc, body)?;
                        }
                    }
                } else {
                    let acc_count = num_acc.min(13); // ymm0-ymm12 max
                    for a in 0..acc_count {
                        let acc = Self::ymm_for_index(a)?;
                        if has_ext && self.bias_saved {
                            let v = a % nr_vecs;
                            let bias_disp = (v as i32) * 32;
                            self.emit_trace_on_accumulator_with_bias(acc, body, bias_disp)?;
                        } else {
                            self.emit_trace_on_accumulator(acc, body)?;
                        }
                    }
                }
            }

            Ok(())
        }

        /// AVX2 trace-on-stack evaluator.
        ///
        /// Allocates `n * 32` bytes on the stack, evaluates each `TraceOp`
        /// into its slot, then loads the final slot back into `acc`.
        /// `InputMode` controls how `Input(idx)` nodes are materialised.
        fn emit_trace_on_stack_ymm(
            &mut self,
            acc: AsmRegisterYmm,
            body: &[TraceOp],
            mode: InputMode,
        ) -> Result<(), String> {
            trace_on_stack_body!(
                self, acc, body, mode,
                simd_bytes: 32i32,
                ptr_fn: ymmword_ptr,
                s0: ymm13, s1: ymm14,
                math_s0: ymm15, math_s1: ymm12,
                broadcast_reg: ymm15,
                mean_reg: ymm14,
                scale_reg: ymm15,
                zero_instr: vxorps,
                rcp_instr: vrcpps,
                rsqrt_instr: vrsqrtps,
                emit_exp: emit_exp_avx2,
                emit_tanh: emit_tanh_avx2,
                emit_log: emit_log_avx2,
                scalar_tail_load: |asm: &mut CodeAssembler| -> Result<(), String> {
                    asm.vmovss(xmm13, dword_ptr(rsi + rbx))
                        .map_err(|e| e.to_string())
                }
            )
        }

        /// AVX2 trace-on-accumulator (acc-only). Thin wrapper.
        fn emit_trace_on_accumulator(
            &mut self,
            acc: AsmRegisterYmm,
            body: &[TraceOp],
        ) -> Result<(), String> {
            self.emit_trace_on_stack_ymm(acc, body, InputMode::AccOnly)
        }

        /// AVX2 trace-on-accumulator with bias. Thin wrapper.
        fn emit_trace_on_accumulator_with_bias(
            &mut self,
            acc: AsmRegisterYmm,
            body: &[TraceOp],
            bias_disp: i32,
        ) -> Result<(), String> {
            self.emit_trace_on_stack_ymm(acc, body, InputMode::WithBias { bias_disp })
        }

        /// AVX-512 trace-on-stack evaluator.
        ///
        /// Allocates `n * 64` bytes on the stack, evaluates each `TraceOp`
        /// into its slot, then loads the final slot back into `acc`.
        /// `InputMode` controls how `Input(idx)` nodes are materialised.
        fn emit_trace_on_stack_zmm(
            &mut self,
            acc: AsmRegisterZmm,
            body: &[TraceOp],
            mode: InputMode,
        ) -> Result<(), String> {
            trace_on_stack_body!(
                self, acc, body, mode,
                simd_bytes: 64i32,
                ptr_fn: zmmword_ptr,
                s0: zmm29, s1: zmm30,
                math_s0: zmm27, math_s1: zmm26,
                broadcast_reg: zmm31,
                mean_reg: zmm30,
                scale_reg: zmm31,
                zero_instr: vpxord,
                rcp_instr: vrcp14ps,
                rsqrt_instr: vrsqrt14ps,
                emit_exp: emit_exp_avx512,
                emit_tanh: emit_tanh_avx512,
                emit_log: emit_log_avx512,
                scalar_tail_load: |asm: &mut CodeAssembler| -> Result<(), String> {
                    asm.vmovups(zmm29.k1().z(), zmmword_ptr(rsi + rbx))
                        .map_err(|e| e.to_string())
                }
            )
        }

        /// AVX-512 trace-on-accumulator with bias. Thin wrapper.
        fn emit_trace_on_accumulator_with_bias_avx512(
            &mut self,
            acc: AsmRegisterZmm,
            body: &[TraceOp],
            bias_disp: i32,
        ) -> Result<(), String> {
            self.emit_trace_on_stack_zmm(acc, body, InputMode::WithBias { bias_disp })
        }

        /// AVX-512 trace-on-accumulator (acc-only). Thin wrapper.
        fn emit_trace_on_accumulator_avx512(
            &mut self,
            acc: AsmRegisterZmm,
            body: &[TraceOp],
        ) -> Result<(), String> {
            self.emit_trace_on_stack_zmm(acc, body, InputMode::AccOnly)
        }


        /// Emit exp(x) via shared SimdOps path.
        /// Wrapper for call sites that still use physical ymm registers.
        fn emit_exp_avx2(
            &mut self,
            dst: AsmRegisterYmm,
            src: AsmRegisterYmm,
            s: [AsmRegisterYmm; 3],
        ) -> Result<(), String> {
            use crate::compiler::codegen::simd_ops::VReg;
            let dst_v = VReg(ymm_index(dst));
            let src_v = VReg(ymm_index(src));
            let scratch = [VReg(ymm_index(s[0])), VReg(ymm_index(s[1])), VReg(ymm_index(s[2]))];
            crate::compiler::codegen::math_approx::emit_exp(self, dst_v, src_v, scratch)
        }

        /// Emit tanh(x) via shared SimdOps path.
        fn emit_tanh_avx2(
            &mut self,
            dst: AsmRegisterYmm,
            src: AsmRegisterYmm,
            s: [AsmRegisterYmm; 3],
        ) -> Result<(), String> {
            use crate::compiler::codegen::simd_ops::VReg;
            let dst_v = VReg(ymm_index(dst));
            let src_v = VReg(ymm_index(src));
            let scratch = [VReg(ymm_index(s[0])), VReg(ymm_index(s[1])), VReg(ymm_index(s[2]))];
            crate::compiler::codegen::math_approx::emit_tanh(self, dst_v, src_v, scratch)
        }

        /// Emit log(x) via shared SimdOps path.
        fn emit_log_avx2(
            &mut self,
            dst: AsmRegisterYmm,
            src: AsmRegisterYmm,
            s: [AsmRegisterYmm; 3],
        ) -> Result<(), String> {
            use crate::compiler::codegen::simd_ops::VReg;
            let dst_v = VReg(ymm_index(dst));
            let src_v = VReg(ymm_index(src));
            let scratch = [VReg(ymm_index(s[0])), VReg(ymm_index(s[1])), VReg(ymm_index(s[2]))];
            crate::compiler::codegen::math_approx::emit_log(self, dst_v, src_v, scratch)
        }

        /// Emit log(x) via shared SimdOps path (AVX-512).
        fn emit_log_avx512(
            &mut self,
            dst: AsmRegisterZmm,
            src: AsmRegisterZmm,
            s: [AsmRegisterZmm; 3],
        ) -> Result<(), String> {
            use crate::compiler::codegen::simd_ops::VReg;
            let dst_v = VReg(zmm_index(dst));
            let src_v = VReg(zmm_index(src));
            let scratch = [VReg(zmm_index(s[0])), VReg(zmm_index(s[1])), VReg(zmm_index(s[2]))];
            crate::compiler::codegen::math_approx::emit_log(self, dst_v, src_v, scratch)
        }


        /// Emit exp(x) via shared SimdOps path (AVX-512).
        fn emit_exp_avx512(
            &mut self,
            dst: AsmRegisterZmm,
            src: AsmRegisterZmm,
            s: [AsmRegisterZmm; 3],
        ) -> Result<(), String> {
            use crate::compiler::codegen::simd_ops::VReg;
            let dst_v = VReg(zmm_index(dst));
            let src_v = VReg(zmm_index(src));
            let scratch = [VReg(zmm_index(s[0])), VReg(zmm_index(s[1])), VReg(zmm_index(s[2]))];
            crate::compiler::codegen::math_approx::emit_exp(self, dst_v, src_v, scratch)
        }

        /// Emit tanh(x) via shared SimdOps path (AVX-512).
        fn emit_tanh_avx512(
            &mut self,
            dst: AsmRegisterZmm,
            src: AsmRegisterZmm,
            s: [AsmRegisterZmm; 3],
        ) -> Result<(), String> {
            use crate::compiler::codegen::simd_ops::VReg;
            let dst_v = VReg(zmm_index(dst));
            let src_v = VReg(zmm_index(src));
            let scratch = [VReg(zmm_index(s[0])), VReg(zmm_index(s[1])), VReg(zmm_index(s[2]))];
            crate::compiler::codegen::math_approx::emit_tanh(self, dst_v, src_v, scratch)
        }

        /// Emit a TraceOp sequence as AVX2 SIMD instructions.
        ///
        /// Delegates to the shared `algorithm::emit_trace_body` via the
        /// `SimdOps` trait implementation, then converts VRegs back to
        /// physical ymm registers. Scratch: ymm13-ymm15.
        pub fn emit_trace_ops_avx2(
            &mut self,
            ops: &[TraceOp],
        ) -> Result<Vec<AsmRegisterYmm>, String> {
            use crate::compiler::codegen::simd_ops::VReg;
            // Scratch registers: ymm13, ymm14, ymm15
            let scratch = [VReg(13), VReg(14), VReg(15)];
            let vregs = crate::compiler::codegen::algorithm::emit_trace_body(self, ops, 0, scratch)?;
            // Convert VRegs back to physical ymm registers
            vregs.iter()
                .map(|v| Self::ymm_any(v.0 as usize))
                .collect()
        }

        /// Emit a TraceOp sequence as AVX-512 SIMD instructions.
        ///
        /// Delegates to the shared `algorithm::emit_trace_body` via the
        /// `SimdOps` trait implementation, then converts VRegs back to
        /// physical zmm registers. Scratch: zmm29-zmm31.
        pub fn emit_trace_ops_avx512(
            &mut self,
            ops: &[TraceOp],
        ) -> Result<Vec<AsmRegisterZmm>, String> {
            use crate::compiler::codegen::simd_ops::VReg;
            // Scratch registers: zmm29, zmm30, zmm31
            let scratch = [VReg(29), VReg(30), VReg(31)];
            let vregs = crate::compiler::codegen::algorithm::emit_trace_body(self, ops, 0, scratch)?;
            // Convert VRegs back to physical zmm registers
            vregs.iter()
                .map(|v| Self::zmm_any(v.0 as usize))
                .collect()
        }

        /// Map a TraceOp SSA index to a ymm register (ymm0-ymm12).
        ///
        /// Returns `Err` if index >= 13 since ymm13-ymm15 are reserved as
        /// scratch registers for exp/tanh approximations.
        fn ymm_for_index(index: usize) -> Result<AsmRegisterYmm, String> {
            let regs = [
                ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                ymm8, ymm9, ymm10, ymm11, ymm12,
            ];
            if index >= regs.len() {
                return Err(format!(
                    "SSA index {} exceeds available ymm registers (13; ymm13-15 reserved for scratch)",
                    index
                ));
            }
            Ok(regs[index])
        }

        /// Map index 0-15 to xmm0-xmm15 (for scalar operations).
        fn xmm_from_index(index: usize) -> Result<AsmRegisterXmm, String> {
            let regs = [
                xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
                xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15,
            ];
            if index >= 16 {
                return Err(format!("xmm index {} out of range (0-15)", index));
            }
            Ok(regs[index])
        }

        /// Map index 0-15 to ymm0-ymm15 (unrestricted, for scratch registers).
        fn ymm_any(index: usize) -> Result<AsmRegisterYmm, String> {
            let regs = [
                ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,
            ];
            if index >= 16 {
                return Err(format!("ymm index {} out of range (0-15)", index));
            }
            Ok(regs[index])
        }

        // ── AVX-512 zmm register helpers ────────────────────────────────────

        /// Map a TraceOp SSA index to a zmm accumulator register (zmm0-zmm27).
        ///
        /// AVX-512 has 32 zmm registers. zmm28-zmm31 are reserved as scratch
        /// for A-panel broadcast and exp/tanh approximations.
        fn zmm_for_index(index: usize) -> Result<AsmRegisterZmm, String> {
            let regs: [AsmRegisterZmm; 28] = [
                zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7,
                zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15,
                zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23,
                zmm24, zmm25, zmm26, zmm27,
            ];
            if index >= regs.len() {
                return Err(format!(
                    "SSA index {} exceeds available zmm accumulators (28; zmm28-31 reserved)",
                    index
                ));
            }
            Ok(regs[index])
        }

        /// Map index 0-31 to zmm0-zmm31 (unrestricted, for scratch registers).
        fn zmm_any(index: usize) -> Result<AsmRegisterZmm, String> {
            let regs: [AsmRegisterZmm; 32] = [
                zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7,
                zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15,
                zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23,
                zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31,
            ];
            if index >= 32 {
                return Err(format!("zmm index {} out of range (0-31)", index));
            }
            Ok(regs[index])
        }

        /// SIMD register width in bytes for the current ISA (32 for AVX2, 64 for AVX-512).
        #[inline]
        fn simd_bytes(&self) -> i32 {
            if self.use_avx512 { 64 } else { 32 }
        }

        /// Set mask register k1 from a compile-time remainder count.
        ///
        /// Computes `(1 << remainder) - 1` and loads into k1 for masked
        /// load/store of tail elements.
        fn emit_set_kmask(&mut self, remainder: usize) -> Result<(), String> {
            let mask_val = (1u32 << remainder) - 1;
            self.asm.mov(eax, mask_val as i32).map_err(|e| e.to_string())?;
            self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit quantized GEMM: dequantize B into scratchpad, then regular GEMM.
        ///
        /// ABI: rdi=A (f32), rsi=B (quantized blocks), r8=C (f32), [rbp+32]=scratchpad
        ///
        /// Strategy: dequantize entire B matrix to f32 in scratchpad,
        /// redirect rsi to the dequantized buffer, then run regular GEMM.
        fn emit_quant_gemm(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            block_size: usize,
            bits: usize,
            profile: &DeviceProfile,
            epilogue_bodies: &[&[TraceOp]],
        ) -> Result<(), String> {
            let num_elements = k * n;
            if num_elements % block_size != 0 {
                return Err(format!(
                    "QuantGemm: k*n={} not divisible by block_size={}",
                    num_elements, block_size
                ));
            }
            let num_blocks = num_elements / block_size;
            let dequant_bytes = num_elements * 4; // f32 output

            let dequant_offset = self.blis_scratchpad_offset;
            self.blis_scratchpad_offset += dequant_bytes;
            self.blis_scratchpad_bytes = self.blis_scratchpad_bytes.max(
                self.blis_scratchpad_offset - self.blis_base_offset
            );

            self.asm.mov(rbx, qword_ptr(rbp + 32)).map_err(|e| e.to_string())?;

            eprintln!(
                "QuantGemm {}x{}x{}: bits={} blocks={} dequant_offset={} dequant_bytes={}",
                m, n, k, bits, num_blocks, dequant_offset, dequant_bytes
            );

            match bits {
                8 => self.emit_dequant_q8_0_loop(num_blocks, block_size, dequant_offset as i32)?,
                4 => self.emit_dequant_q4_0_loop(num_blocks, block_size, dequant_offset as i32)?,
                _ => return Err(format!("QuantGemm: unsupported bits={}", bits)),
            }

            self.asm.lea(rsi, qword_ptr(rbx + dequant_offset as i32))
                .map_err(|e| e.to_string())?;

            self.emit_gemm_microkernel(m, n, k, profile, epilogue_bodies)
        }

        /// Emit Q8_0 dequantize loop using AVX2 + F16C.
        ///
        /// BlockQ8_0: `d` (f16, 2 bytes) + `qs` (block_size × i8 bytes)
        /// Dequantize: `out[i] = qs[i] * f16_to_f32(d)`
        ///
        /// Processes 8 elements at a time using vpmovsxbd + vcvtdq2ps.
        ///
        /// Register plan:
        ///   rax = source ptr (quantized blocks, starts at rsi)
        ///   rdx = dest ptr (f32 output in scratchpad)
        ///   rcx = block counter (counts down)
        ///   ymm15 = broadcast f32 scale
        ///   ymm0 = temp for dequantized group
        fn emit_dequant_q8_0_loop(
            &mut self,
            num_blocks: usize,
            block_size: usize,
            dequant_offset: i32,
        ) -> Result<(), String> {
            let block_bytes = (2 + block_size) as i32; // f16 scale + i8 * block_size
            let groups_per_block = block_size / 8;

            self.asm.mov(rax, rsi).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(rbx + dequant_offset))
                .map_err(|e| e.to_string())?;
            self.asm.mov(rcx, num_blocks as u64).map_err(|e| e.to_string())?;

            let mut block_loop = self.asm.create_label();
            let mut block_done = self.asm.create_label();

            self.asm.set_label(&mut block_loop).map_err(|e| e.to_string())?;
            self.asm.test(rcx, rcx).map_err(|e| e.to_string())?;
            self.asm.jz(block_done).map_err(|e| e.to_string())?;

            self.asm.movzx(r9d, word_ptr(rax)).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm15, r9d).map_err(|e| e.to_string())?;
            self.asm.vcvtph2ps(xmm15, xmm15).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(ymm15, xmm15).map_err(|e| e.to_string())?;

            for g in 0..groups_per_block {
                let src_off = 2 + (g * 8) as i32; // skip f16 header
                let dst_off = (g * 8 * 4) as i32;  // 8 f32 = 32 bytes

                self.asm.vpmovsxbd(ymm0, qword_ptr(rax + src_off))
                    .map_err(|e| e.to_string())?;
                self.asm.vcvtdq2ps(ymm0, ymm0).map_err(|e| e.to_string())?;
                self.asm.vmulps(ymm0, ymm0, ymm15).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(rdx + dst_off), ymm0)
                    .map_err(|e| e.to_string())?;
            }

            self.asm.add(rax, block_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rdx, (block_size * 4) as i32).map_err(|e| e.to_string())?;
            self.asm.dec(rcx).map_err(|e| e.to_string())?;
            self.asm.jmp(block_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut block_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit Q4_0 dequantize loop using AVX2 + F16C.
        ///
        /// BlockQ4_0: `d` (f16, 2 bytes) + `qs` (block_size/2 × u8, packed nibbles)
        /// Element layout (GGML convention):
        ///   elements 0..15  = low nibbles of qs[0..15]
        ///   elements 16..31 = high nibbles of qs[0..15]
        /// Dequantize: `out[i] = (nibble - 8) * f16_to_f32(d)`
        ///
        /// Register plan:
        ///   rax = source ptr, rdx = dest ptr, rcx = block counter
        ///   ymm15 = broadcast f32 scale
        ///   ymm14 = broadcast i32(8) for subtraction
        ///   xmm13 = 0x0F byte mask
        ///   ymm0-ymm1 = temps
        fn emit_dequant_q4_0_loop(
            &mut self,
            num_blocks: usize,
            block_size: usize,
            dequant_offset: i32,
        ) -> Result<(), String> {
            if block_size != 32 {
                return Err(format!("Q4_0: only block_size=32 supported, got {}", block_size));
            }
            let block_bytes = 2 + (block_size / 2) as i32; // 18 bytes for block_size=32

            self.asm.mov(rax, rsi).map_err(|e| e.to_string())?;
            self.asm.lea(rdx, qword_ptr(rbx + dequant_offset))
                .map_err(|e| e.to_string())?;
            self.asm.mov(rcx, num_blocks as u64).map_err(|e| e.to_string())?;

            self.asm.mov(r9d, 8i32).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm14, r9d).map_err(|e| e.to_string())?;
            self.asm.vpbroadcastd(ymm14, xmm14).map_err(|e| e.to_string())?;

            self.asm.mov(r9d, 0x0F0F0F0Fi32).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm13, r9d).map_err(|e| e.to_string())?;
            self.asm.vpbroadcastd(xmm13, xmm13).map_err(|e| e.to_string())?;

            let mut block_loop = self.asm.create_label();
            let mut block_done = self.asm.create_label();

            self.asm.set_label(&mut block_loop).map_err(|e| e.to_string())?;
            self.asm.test(rcx, rcx).map_err(|e| e.to_string())?;
            self.asm.jz(block_done).map_err(|e| e.to_string())?;

            self.asm.movzx(r9d, word_ptr(rax)).map_err(|e| e.to_string())?;
            self.asm.vmovd(xmm15, r9d).map_err(|e| e.to_string())?;
            self.asm.vcvtph2ps(xmm15, xmm15).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(ymm15, xmm15).map_err(|e| e.to_string())?;

            self.asm.vmovdqu(xmm12, xmmword_ptr(rax + 2))
                .map_err(|e| e.to_string())?;

            self.asm.vpand(xmm10, xmm12, xmm13).map_err(|e| e.to_string())?;
            self.asm.vpsrlw(xmm11, xmm12, 4i32).map_err(|e| e.to_string())?;
            self.asm.vpand(xmm11, xmm11, xmm13).map_err(|e| e.to_string())?;

            self.emit_q4_dequant_group(xmm10, 0)?;    // → [rdx + 0]
            self.asm.vpsrldq(xmm0, xmm10, 8i32).map_err(|e| e.to_string())?;
            self.emit_q4_dequant_group(xmm0, 32)?;     // → [rdx + 32]

            self.emit_q4_dequant_group(xmm11, 64)?;    // → [rdx + 64]
            self.asm.vpsrldq(xmm0, xmm11, 8i32).map_err(|e| e.to_string())?;
            self.emit_q4_dequant_group(xmm0, 96)?;     // → [rdx + 96]

            self.asm.add(rax, block_bytes).map_err(|e| e.to_string())?;
            self.asm.add(rdx, (block_size * 4) as i32).map_err(|e| e.to_string())?;
            self.asm.dec(rcx).map_err(|e| e.to_string())?;
            self.asm.jmp(block_loop).map_err(|e| e.to_string())?;

            self.asm.set_label(&mut block_done).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Emit dequant for one group of 8 Q4_0 nibbles.
        ///
        /// Input: `src_xmm` lower 8 bytes contain nibble values (0-15).
        /// Output: 8 × f32 stored at `[rdx + dst_off]`.
        /// Uses ymm15 (scale), ymm14 (i32 broadcast 8), ymm0 as temp.
        fn emit_q4_dequant_group(
            &mut self,
            src_xmm: iced_x86::code_asm::AsmRegisterXmm,
            dst_off: i32,
        ) -> Result<(), String> {
            self.asm.vpmovzxbd(ymm0, src_xmm).map_err(|e| e.to_string())?;
            self.asm.vpsubd(ymm0, ymm0, ymm14).map_err(|e| e.to_string())?;
            self.asm.vcvtdq2ps(ymm0, ymm0).map_err(|e| e.to_string())?;
            self.asm.vmulps(ymm0, ymm0, ymm15).map_err(|e| e.to_string())?;
            self.asm.vmovups(ymmword_ptr(rdx + dst_off), ymm0)
                .map_err(|e| e.to_string())?;
            Ok(())
        }

        /// JIT emit: RmsNorm for one row of `row_size` f32 elements.
        ///
        /// Computes: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
        ///
        /// Register contract:
        ///   rdi = input row pointer
        ///   weight_reg = weight vector pointer
        ///   output_reg = output row pointer
        ///
        /// Clobbers: rcx (loop counter), ymm0-ymm3, ymm15, xmm0, xmm2
        /// Preserves: rdi (unchanged), rbx, r12-r15, all other GPRs not listed above
        fn emit_norm_row_jit(
            &mut self,
            output_reg: AsmRegister64,
            weight_reg: AsmRegister64,
            row_size: usize,
            eps: f32,
        ) -> Result<(), String> {
            let simd_w = self.simd_width; // 8 for AVX2
            let vec_count = row_size / simd_w;
            let tail = row_size % simd_w;
            let total_vec_bytes = (vec_count * simd_w * 4) as i32;
            let n_f32 = row_size as f32;

            // -- Phase 1: Reduce -- accumulate sum_sq = sum(x^2) --
            self.asm.vxorps(ymm0, ymm0, ymm0).map_err(|e| e.to_string())?; // ymm0 = accumulator

            if vec_count > 0 {
                let mut r_loop = self.asm.create_label();
                let mut r_done = self.asm.create_label();
                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut r_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(r_done).map_err(|e| e.to_string())?;

                self.asm.vmovups(ymm2, ymmword_ptr(rdi + rcx)).map_err(|e| e.to_string())?;
                self.asm.vfmadd231ps(ymm0, ymm2, ymm2).map_err(|e| e.to_string())?;

                self.asm.add(rcx, 32i32).map_err(|e| e.to_string())?;
                self.asm.jmp(r_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut r_done).map_err(|e| e.to_string())?;
            }

            // -- Phase 1.5: Horizontal sum --
            // Must reduce ymm0 to scalar BEFORE scalar tail, because VEX-encoded
            // scalar ops (vfmadd231ss etc.) zero bits [255:128] of the destination ymm.
            self.emit_horizontal_sum_ymm(ymm0, ymm0)?;

            // Scalar tail for reduce (now safe: ymm0 is already a broadcast scalar)
            if tail > 0 {
                let base = total_vec_bytes;
                for t in 0..tail as i32 {
                    let off = base + t * 4;
                    self.asm.vmovss(xmm2, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                    self.asm.vfmadd231ss(xmm0, xmm2, xmm2).map_err(|e| e.to_string())?;
                }
            }

            // -- Phase 2: Finalize -- scale = rsqrt(sum_sq / n + eps) --
            let n_label = self.const_f32(n_f32);
            let eps_label = self.const_f32(eps);

            // After hsum + scalar tail, sum_sq is in xmm0[0]. Broadcast to ymm for vectorized divide.
            self.asm.vbroadcastss(ymm0, xmm0).map_err(|e| e.to_string())?;
            self.asm.vbroadcastss(ymm3, dword_ptr(n_label)).map_err(|e| e.to_string())?;
            self.asm.vdivps(ymm2, ymm0, ymm3).map_err(|e| e.to_string())?;    // mean_sq = sum_sq / n
            self.asm.vbroadcastss(ymm3, dword_ptr(eps_label)).map_err(|e| e.to_string())?;
            self.asm.vaddps(ymm2, ymm2, ymm3).map_err(|e| e.to_string())?;     // mean_sq + eps
            self.asm.vrsqrtps(ymm15, ymm2).map_err(|e| e.to_string())?;        // ymm15 = scale

            // -- Phase 3: Transform -- output = x * scale * weight --
            if vec_count > 0 {
                let mut t_loop = self.asm.create_label();
                let mut t_done = self.asm.create_label();
                self.asm.xor(rcx, rcx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut t_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rcx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(t_done).map_err(|e| e.to_string())?;

                self.asm.vmovups(ymm2, ymmword_ptr(rdi + rcx)).map_err(|e| e.to_string())?;       // x
                self.asm.vmulps(ymm2, ymm2, ymm15).map_err(|e| e.to_string())?;                   // x * scale
                self.asm.vmovups(ymm3, ymmword_ptr(weight_reg + rcx)).map_err(|e| e.to_string())?; // weight
                self.asm.vmulps(ymm2, ymm2, ymm3).map_err(|e| e.to_string())?;                    // x * scale * weight
                self.asm.vmovups(ymmword_ptr(output_reg + rcx), ymm2).map_err(|e| e.to_string())?; // store

                self.asm.add(rcx, 32i32).map_err(|e| e.to_string())?;
                self.asm.jmp(t_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut t_done).map_err(|e| e.to_string())?;
            }

            // Scalar tail for transform
            if tail > 0 {
                let base = total_vec_bytes;
                for t in 0..tail as i32 {
                    let off = base + t * 4;
                    self.asm.vmovss(xmm0, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;       // x
                    self.asm.vmulss(xmm0, xmm0, xmm15).map_err(|e| e.to_string())?;                // x * scale
                    self.asm.vmovss(xmm2, dword_ptr(weight_reg + off)).map_err(|e| e.to_string())?; // weight
                    self.asm.vmulss(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;                 // x * scale * weight
                    self.asm.vmovss(dword_ptr(output_reg + off), xmm0).map_err(|e| e.to_string())?; // store
                }
            }

            Ok(())
        }



        /// Emit a standalone NormLike op (RmsNorm / LayerNorm).
        ///
        /// Three-phase structure, trace-driven:
        ///   Phase 1: Reduce — accumulate sum (and sum_sq for LayerNorm)
        ///   Phase 2: Finalize — compute scale (and mean for LayerNorm)
        ///   Phase 3: Transform — per-element normalization with weight/bias
        ///
        /// ABI: rdi = input ptr, rsi = weight ptr (rdx = bias ptr for LayerNorm), r8 = output ptr
        fn emit_standalone_norm(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            registry: &ScalarOpRegistry,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor op")?;
            let out_shape = if let Some(&out_id) = op.outputs.first() {
                graph.tensor(out_id).map(|t| t.shape.clone()).unwrap_or_default()
            } else {
                vec![]
            };
            let elem_count: usize = out_shape.iter().product::<usize>().max(1);
            if elem_count == 0 || out_shape.is_empty() {
                return Err("norm op: elem_count=0 or empty shape".into());
            }
            // LayerNorm/RmsNorm normalizes over the last dimension
            let row_size = *out_shape.last().unwrap();
            let num_rows = elem_count / row_size;

            let key = ScalarOpRegistry::key_from_op_kind(&op.kind);
            let trace = registry.get_trace(&key).ok_or("missing trace for norm op")?;
            let (reduce, finalize, transform) = match &trace.pattern {
                ComputePattern::NormLike { reduce, finalize, transform } => {
                    (reduce, finalize, transform)
                }
                _ => return Err("expected NormLike pattern".into()),
            };

            let is_layer_norm = matches!(op.kind, OpKind::LayerNorm { .. });
            let eps = match &op.kind {
                OpKind::RmsNorm { eps } | OpKind::LayerNorm { eps } => *eps,
                _ => 1e-5,
            };

            let simd_w = self.simd_width;
            let vec_count = row_size / simd_w;
            let tail = row_size % simd_w;
            let total_vec_bytes = (vec_count * simd_w * 4) as i32;
            let n_f32 = row_size as f32;
            let row_bytes = (row_size * 4) as i32;

            if self.use_avx512 {
                return self.emit_standalone_norm_avx512(
                    reduce, finalize, transform,
                    is_layer_norm, eps, elem_count, vec_count, tail, total_vec_bytes, n_f32,
                );
            }

            // Outer row loop: normalize each row independently
            let mut row_loop = self.asm.create_label();
            let mut row_done = self.asm.create_label();
            // Use r9 as row counter (not used by norm code)
            self.asm.xor(r9, r9).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut row_loop).map_err(|e| e.to_string())?;
            self.asm.cmp(r9, num_rows as i32).map_err(|e| e.to_string())?;
            self.asm.jge(row_done).map_err(|e| e.to_string())?;

            // ── AVX2 path ──

            // Phase 1: Reduce — accumulate sum_sq (RmsNorm) or sum + sum_sq (LayerNorm)
            self.asm.vxorps(ymm0, ymm0, ymm0).map_err(|e| e.to_string())?; // ymm0 = sum or sum_sq
            if is_layer_norm {
                self.asm.vxorps(ymm1, ymm1, ymm1).map_err(|e| e.to_string())?; // ymm1 = sum_sq (LN only)
            }

            if vec_count > 0 {
                let mut r_loop = self.asm.create_label();
                let mut r_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut r_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(r_done).map_err(|e| e.to_string())?;

                self.asm.vmovups(ymm2, ymmword_ptr(rdi + rbx)).map_err(|e| e.to_string())?;

                if is_layer_norm {
                    // sum += x
                    self.asm.vaddps(ymm0, ymm0, ymm2).map_err(|e| e.to_string())?;
                    // sum_sq += x * x
                    self.asm.vfmadd231ps(ymm1, ymm2, ymm2).map_err(|e| e.to_string())?;
                } else {
                    // RmsNorm: sum_sq += x * x
                    self.asm.vfmadd231ps(ymm0, ymm2, ymm2).map_err(|e| e.to_string())?;
                }

                self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
                self.asm.jmp(r_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut r_done).map_err(|e| e.to_string())?;
            }

            // Horizontal sum BEFORE scalar tail — VEX-encoded scalar ops
            // (vfmadd231ss, vaddss) zero bits [255:128] of the destination ymm.
            self.emit_horizontal_sum_ymm(ymm0, ymm0)?;
            if is_layer_norm {
                self.asm.vmovaps(ymm14, ymm0).map_err(|e| e.to_string())?;
                self.emit_horizontal_sum_ymm(ymm1, ymm1)?;
                self.asm.vmovaps(ymm0, ymm14).map_err(|e| e.to_string())?;
            }

            // Scalar tail for reduce (now safe: accumulators are already scalar)
            if tail > 0 {
                let base = total_vec_bytes;
                for t in 0..tail as i32 {
                    let off = base + t * 4;
                    self.asm.vmovss(xmm2, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                    if is_layer_norm {
                        self.asm.vaddss(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                        self.asm.vfmadd231ss(xmm1, xmm2, xmm2).map_err(|e| e.to_string())?;
                    } else {
                        self.asm.vfmadd231ss(xmm0, xmm2, xmm2).map_err(|e| e.to_string())?;
                    }
                }
            }

            // Broadcast scalar results back to ymm for vectorized finalize
            self.asm.vbroadcastss(ymm0, xmm0).map_err(|e| e.to_string())?;
            if is_layer_norm {
                self.asm.vbroadcastss(ymm1, xmm1).map_err(|e| e.to_string())?;
            }

            // Phase 2: Finalize — compute scale (and mean for LayerNorm)
            let n_label = self.const_f32(n_f32);
            let eps_label = self.const_f32(eps);

            if is_layer_norm {
                // mean = sum / n
                self.asm.vbroadcastss(ymm3, dword_ptr(n_label)).map_err(|e| e.to_string())?;
                self.asm.vdivps(ymm14, ymm0, ymm3).map_err(|e| e.to_string())?;
                // ymm14 = broadcast mean

                // var = sum_sq / n - mean^2
                self.asm.vdivps(ymm2, ymm1, ymm3).map_err(|e| e.to_string())?;
                self.asm.vfnmadd231ps(ymm2, ymm14, ymm14).map_err(|e| e.to_string())?;
                // ymm2 = var

                // scale = rsqrt(var + eps)
                self.asm.vbroadcastss(ymm3, dword_ptr(eps_label)).map_err(|e| e.to_string())?;
                self.asm.vaddps(ymm2, ymm2, ymm3).map_err(|e| e.to_string())?;
                self.asm.vrsqrtps(ymm15, ymm2).map_err(|e| e.to_string())?;
                // ymm14 = mean, ymm15 = scale
            } else {
                // RmsNorm: mean_sq = sum_sq / n, scale = rsqrt(mean_sq + eps)
                self.asm.vbroadcastss(ymm3, dword_ptr(n_label)).map_err(|e| e.to_string())?;
                self.asm.vdivps(ymm2, ymm0, ymm3).map_err(|e| e.to_string())?;
                self.asm.vbroadcastss(ymm3, dword_ptr(eps_label)).map_err(|e| e.to_string())?;
                self.asm.vaddps(ymm2, ymm2, ymm3).map_err(|e| e.to_string())?;
                self.asm.vrsqrtps(ymm15, ymm2).map_err(|e| e.to_string())?;
                // ymm15 = scale
            }

            // Phase 3: Transform — per-element normalization
            // Use NormTransform InputMode for trace evaluation
            let norm_mode = InputMode::NormTransform {
                has_mean: is_layer_norm,
                has_weight: true,
                has_bias: is_layer_norm,
            };

            if vec_count > 0 {
                let mut t_loop = self.asm.create_label();
                let mut t_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut t_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(t_done).map_err(|e| e.to_string())?;

                self.emit_trace_on_stack_ymm(ymm0, transform, norm_mode)?;
                self.asm.vmovups(ymmword_ptr(r8 + rbx), ymm0).map_err(|e| e.to_string())?;

                self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
                self.asm.jmp(t_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut t_done).map_err(|e| e.to_string())?;
            }

            // Scalar tail for transform
            if tail > 0 {
                let base = total_vec_bytes;
                for t in 0..tail as i32 {
                    let off = base + t * 4;
                    // x
                    self.asm.vmovss(xmm0, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                    if is_layer_norm {
                        // (x - mean) * scale * weight + bias
                        self.asm.vsubss(xmm0, xmm0, xmm14).map_err(|e| e.to_string())?;
                        self.asm.vmulss(xmm0, xmm0, xmm15).map_err(|e| e.to_string())?;
                        // weight
                        self.asm.vmovss(xmm2, dword_ptr(rdx + off)).map_err(|e| e.to_string())?;
                        self.asm.vmulss(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                        // bias
                        self.asm.vmovss(xmm2, dword_ptr(rcx + off)).map_err(|e| e.to_string())?;
                        self.asm.vaddss(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                    } else {
                        // x * scale * weight
                        self.asm.vmulss(xmm0, xmm0, xmm15).map_err(|e| e.to_string())?;
                        self.asm.vmovss(xmm2, dword_ptr(rdx + off)).map_err(|e| e.to_string())?;
                        self.asm.vmulss(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                    }
                    self.asm.vmovss(dword_ptr(r8 + off), xmm0).map_err(|e| e.to_string())?;
                }
            }

            // Advance input/output pointers for next row; weight/bias stay the same
            self.asm.add(rdi, row_bytes).map_err(|e| e.to_string())?;
            self.asm.add(r8, row_bytes).map_err(|e| e.to_string())?;
            self.asm.inc(r9).map_err(|e| e.to_string())?;
            self.asm.jmp(row_loop).map_err(|e| e.to_string())?;
            self.asm.set_label(&mut row_done).map_err(|e| e.to_string())?;

            Ok(())
        }

        /// AVX-512 path for standalone NormLike (RmsNorm / LayerNorm).
        fn emit_standalone_norm_avx512(
            &mut self,
            _reduce: &[TraceOp],
            _finalize: &[TraceOp],
            transform: &[TraceOp],
            is_layer_norm: bool,
            eps: f32,
            elem_count: usize,
            vec_count: usize,
            tail: usize,
            total_vec_bytes: i32,
            n_f32: f32,
        ) -> Result<(), String> {
            // Phase 1: Reduce
            self.asm.vpxord(zmm0, zmm0, zmm0).map_err(|e| e.to_string())?;
            if is_layer_norm {
                self.asm.vpxord(zmm1, zmm1, zmm1).map_err(|e| e.to_string())?;
            }

            if vec_count > 0 {
                let mut r_loop = self.asm.create_label();
                let mut r_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut r_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(r_done).map_err(|e| e.to_string())?;

                self.asm.vmovups(zmm2, zmmword_ptr(rdi + rbx)).map_err(|e| e.to_string())?;
                if is_layer_norm {
                    self.asm.vaddps(zmm0, zmm0, zmm2).map_err(|e| e.to_string())?;
                    self.asm.vfmadd231ps(zmm1, zmm2, zmm2).map_err(|e| e.to_string())?;
                } else {
                    self.asm.vfmadd231ps(zmm0, zmm2, zmm2).map_err(|e| e.to_string())?;
                }

                self.asm.add(rbx, 64i32).map_err(|e| e.to_string())?;
                self.asm.jmp(r_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut r_done).map_err(|e| e.to_string())?;
            }

            if tail > 0 {
                let mask = (1u32 << tail) - 1;
                self.asm.mov(eax, mask as i32).map_err(|e| e.to_string())?;
                self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
                self.asm.vpxord(zmm2, zmm2, zmm2).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmm2.k1(), zmmword_ptr(rdi + total_vec_bytes)).map_err(|e| e.to_string())?;
                if is_layer_norm {
                    self.asm.vaddps(zmm0, zmm0, zmm2).map_err(|e| e.to_string())?;
                    self.asm.vfmadd231ps(zmm1, zmm2, zmm2).map_err(|e| e.to_string())?;
                } else {
                    self.asm.vfmadd231ps(zmm0, zmm2, zmm2).map_err(|e| e.to_string())?;
                }
            }

            self.emit_horizontal_sum_zmm(zmm0, zmm0)?;
            if is_layer_norm {
                self.emit_horizontal_sum_zmm(zmm1, zmm1)?;
            }

            // Phase 2: Finalize
            let n_label = self.const_f32(n_f32);
            let eps_label = self.const_f32(eps);

            if is_layer_norm {
                self.asm.vbroadcastss(zmm3, dword_ptr(n_label)).map_err(|e| e.to_string())?;
                self.asm.vdivps(zmm30, zmm0, zmm3).map_err(|e| e.to_string())?; // zmm30 = mean
                self.asm.vdivps(zmm2, zmm1, zmm3).map_err(|e| e.to_string())?;
                self.asm.vfnmadd231ps(zmm2, zmm30, zmm30).map_err(|e| e.to_string())?; // var
                self.asm.vbroadcastss(zmm3, dword_ptr(eps_label)).map_err(|e| e.to_string())?;
                self.asm.vaddps(zmm2, zmm2, zmm3).map_err(|e| e.to_string())?;
                self.asm.vrsqrt14ps(zmm31, zmm2).map_err(|e| e.to_string())?; // zmm31 = scale
            } else {
                self.asm.vbroadcastss(zmm3, dword_ptr(n_label)).map_err(|e| e.to_string())?;
                self.asm.vdivps(zmm2, zmm0, zmm3).map_err(|e| e.to_string())?;
                self.asm.vbroadcastss(zmm3, dword_ptr(eps_label)).map_err(|e| e.to_string())?;
                self.asm.vaddps(zmm2, zmm2, zmm3).map_err(|e| e.to_string())?;
                self.asm.vrsqrt14ps(zmm31, zmm2).map_err(|e| e.to_string())?; // zmm31 = scale
            }

            // Phase 3: Transform
            let norm_mode = InputMode::NormTransform {
                has_mean: is_layer_norm,
                has_weight: true,
                has_bias: is_layer_norm,
            };

            if vec_count > 0 {
                let mut t_loop = self.asm.create_label();
                let mut t_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut t_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(t_done).map_err(|e| e.to_string())?;

                self.emit_trace_on_stack_zmm(zmm0, transform, norm_mode)?;
                self.asm.vmovups(zmmword_ptr(r8 + rbx), zmm0).map_err(|e| e.to_string())?;

                self.asm.add(rbx, 64i32).map_err(|e| e.to_string())?;
                self.asm.jmp(t_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut t_done).map_err(|e| e.to_string())?;
            }

            if tail > 0 {
                let mask = (1u32 << tail) - 1;
                self.asm.mov(eax, mask as i32).map_err(|e| e.to_string())?;
                self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
                // Scalar tail: load x, apply norm, store
                self.asm.vpxord(zmm0, zmm0, zmm0).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmm0.k1(), zmmword_ptr(rdi + total_vec_bytes)).map_err(|e| e.to_string())?;
                if is_layer_norm {
                    self.asm.vsubps(zmm0, zmm0, zmm30).map_err(|e| e.to_string())?;
                }
                self.asm.vmulps(zmm0, zmm0, zmm31).map_err(|e| e.to_string())?;
                // weight
                self.asm.vpxord(zmm2, zmm2, zmm2).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmm2.k1(), zmmword_ptr(rdx + total_vec_bytes)).map_err(|e| e.to_string())?;
                self.asm.vmulps(zmm0, zmm0, zmm2).map_err(|e| e.to_string())?;
                if is_layer_norm {
                    self.asm.vpxord(zmm2, zmm2, zmm2).map_err(|e| e.to_string())?;
                    self.asm.vmovups(zmm2.k1(), zmmword_ptr(rcx + total_vec_bytes)).map_err(|e| e.to_string())?;
                    self.asm.vaddps(zmm0, zmm0, zmm2).map_err(|e| e.to_string())?;
                }
                self.asm.vmovups(zmmword_ptr(r8 + total_vec_bytes).k1(), zmm0).map_err(|e| e.to_string())?;
            }

            Ok(())
        }

        /// Emit a standalone reduction op (e.g. Softmax).
        ///
        /// Three-pass structure, all trace-driven:
        ///   Pass 1: Reduce (e.g. max) — combine trace
        ///   Pass 2: Element transform + second reduce (e.g. exp-sum) — second_pass trace
        ///   Pass 3: Normalize (e.g. multiply by inv_sum) — normalize trace
        ///
        /// ABI: rdi = input ptr, r8 = output ptr, [rbp+32] = scratchpad ptr
        fn emit_standalone_reduction(
            &mut self,
            group: &FusionGroup,
            graph: &CompilerGraph,
            profile: &DeviceProfile,
            registry: &ScalarOpRegistry,
        ) -> Result<(), String> {
            let op = graph.op(group.anchor).ok_or("missing anchor op")?;
            let elem_count = if let Some(&out_id) = op.outputs.first() {
                graph.tensor_numel(out_id).unwrap_or(0)
            } else {
                0
            };
            if elem_count == 0 {
                eprintln!("[JIT-DBG] emit_standalone_reduction: elem_count=0, emitting NOP");
                return Err("reduction op: elem_count=0".into());
            }

            let key = ScalarOpRegistry::key_from_op_kind(&op.kind);
            let trace = registry.get_trace(&key).ok_or("missing trace for reduction op")?;
            let (identity, combine, second_pass, normalize) = match &trace.pattern {
                ComputePattern::Reduction { identity, combine, second_pass, normalize } => {
                    (*identity, combine, second_pass, normalize)
                }
                _ => return Err("expected Reduction pattern".into()),
            };

            let simd_w = self.simd_width;
            let vec_count = elem_count / simd_w;
            let tail = elem_count % simd_w;
            let total_vec_bytes = (vec_count * simd_w * 4) as i32;
            eprintln!("[JIT-DBG] emit_standalone_reduction: elem_count={elem_count}, simd_w={}, vec_count={vec_count}, tail={tail}, total_vec_bytes={total_vec_bytes}", self.simd_width);

            // We need a temp buffer for pass 2 output (exp values).
            // Use r8 (output ptr) as the temp buffer since we write final results there anyway.

            if self.use_avx512 {
                return self.emit_standalone_reduction_avx512(
                    combine, second_pass, normalize,
                    identity, elem_count, vec_count, tail, total_vec_bytes,
                );
            }

            // ── AVX2 path ──


            // Pass 1: Max reduction
            let identity_label = self.const_f32(identity as f32);
            self.asm.vbroadcastss(ymm0, dword_ptr(identity_label)).map_err(|e| e.to_string())?;

            if vec_count > 0 {
                let mut p1_loop = self.asm.create_label();
                let mut p1_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p1_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(p1_done).map_err(|e| e.to_string())?;

                // evaluate combine trace: Input(0)=ymm0 (acc), Input(1)=load [rdi+rbx]
                self.emit_trace_on_stack_ymm(ymm0, combine, InputMode::ReductionCombine)?;

                self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
                self.asm.jmp(p1_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p1_done).map_err(|e| e.to_string())?;
            }

            // Horizontal max BEFORE scalar tail — VEX-encoded vmaxss zeros
            // bits [255:128] of the destination ymm.
            self.emit_horizontal_max_ymm(ymm0, ymm15)?;

            // Scalar tail for pass 1 (now safe: max is scalar in xmm15)
            if tail > 0 {
                let base = total_vec_bytes;
                for t in 0..tail as i32 {
                    let off = base + t * 4;
                    self.asm.vmovss(xmm1, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                    self.asm.vmaxss(xmm15, xmm15, xmm1).map_err(|e| e.to_string())?;
                }
                // Re-broadcast updated max to ymm15
                self.asm.vbroadcastss(ymm15, xmm15).map_err(|e| e.to_string())?;
            }

            // If no second_pass, we're done with just the reduction
            let second = match second_pass.as_deref() {
                Some(s) => s,
                None => return Ok(()),
            };

            // Pass 2: exp-sum
            // ymm15 = broadcast max
            let sp_identity_label = self.const_f32(second.identity as f32);
            self.asm.vbroadcastss(ymm0, dword_ptr(sp_identity_label)).map_err(|e| e.to_string())?;
            // ymm0 = sum accumulator

            // Allocate 2 ymm stack slots to preserve ymm15 (broadcast max) and
            // accumulator across emit_trace_on_stack_ymm calls, which clobber
            // ymm12/ymm15 via emit_exp_avx2 scratch registers.
            self.asm.sub(rsp, 64i32).map_err(|e| e.to_string())?;
            self.asm.vmovups(ymmword_ptr(rsp), ymm15).map_err(|e| e.to_string())?;

            if vec_count > 0 {
                let mut p2_loop = self.asm.create_label();
                let mut p2_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p2_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(p2_done).map_err(|e| e.to_string())?;

                // Save accumulator to stack before trace call
                self.asm.vmovups(ymmword_ptr(rsp + 32), ymm0).map_err(|e| e.to_string())?;
                self.emit_trace_on_stack_ymm(ymm0, &second.element_transform, InputMode::WithBroadcast)?;
                // ymm0 = exp(x - max), ymm12/ymm15 may be clobbered by exp scratch
                self.asm.vmovups(ymmword_ptr(r8 + rbx), ymm0).map_err(|e| e.to_string())?;

                // Restore accumulator from stack, add exp value
                self.asm.vmovups(ymm12, ymmword_ptr(rsp + 32)).map_err(|e| e.to_string())?;
                self.asm.vaddps(ymm0, ymm12, ymm0).map_err(|e| e.to_string())?;
                // Restore ymm15 (broadcast max) for next iteration
                self.asm.vmovups(ymm15, ymmword_ptr(rsp)).map_err(|e| e.to_string())?;

                self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
                self.asm.jmp(p2_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p2_done).map_err(|e| e.to_string())?;
            }

            // Horizontal sum BEFORE scalar tail — VEX-encoded vaddss zeros
            // bits [255:128] of the destination ymm.
            self.emit_horizontal_sum_ymm(ymm0, ymm0)?;

            // Scalar tail for pass 2 (now safe: ymm0 is already a scalar)
            if tail > 0 {
                let base = total_vec_bytes;
                for t in 0..tail as i32 {
                    let off = base + t * 4;
                    // exp(x - max)
                    self.asm.vmovss(xmm1, dword_ptr(rdi + off)).map_err(|e| e.to_string())?;
                    self.asm.vsubss(xmm1, xmm1, xmm15).map_err(|e| e.to_string())?;
                    // scalar exp approximation: store, use vector exp, extract
                    self.asm.vbroadcastss(ymm1, xmm1).map_err(|e| e.to_string())?;
                    self.emit_exp_avx2(ymm2, ymm1, [ymm1, ymm3, ymm4])?;
                    self.asm.vmovss(dword_ptr(r8 + off), xmm2).map_err(|e| e.to_string())?;
                    self.asm.vaddss(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                }
            }

            self.asm.add(rsp, 64i32).map_err(|e| e.to_string())?;

            // Broadcast scalar sum back to ymm for reciprocal
            self.asm.vbroadcastss(ymm15, xmm0).map_err(|e| e.to_string())?;
            // ymm15 = broadcast sum, compute reciprocal
            self.asm.vrcpps(ymm15, ymm15).map_err(|e| e.to_string())?;
            // ymm15 = broadcast inv_sum

            // Pass 3: Normalize
            let norm_trace = match normalize {
                Some(n) => n,
                None => return Ok(()),
            };

            if vec_count > 0 {
                let mut p3_loop = self.asm.create_label();
                let mut p3_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p3_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(p3_done).map_err(|e| e.to_string())?;

                // Load exp value from output buffer, multiply by inv_sum
                self.asm.vmovups(ymm0, ymmword_ptr(r8 + rbx)).map_err(|e| e.to_string())?;
                self.asm.vmulps(ymm0, ymm0, ymm15).map_err(|e| e.to_string())?;
                self.asm.vmovups(ymmword_ptr(r8 + rbx), ymm0).map_err(|e| e.to_string())?;

                self.asm.add(rbx, 32i32).map_err(|e| e.to_string())?;
                self.asm.jmp(p3_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p3_done).map_err(|e| e.to_string())?;
            }

            // Scalar tail for pass 3
            if tail > 0 {
                let base = total_vec_bytes;
                for t in 0..tail as i32 {
                    let off = base + t * 4;
                    self.asm.vmovss(xmm0, dword_ptr(r8 + off)).map_err(|e| e.to_string())?;
                    self.asm.vmulss(xmm0, xmm0, xmm15).map_err(|e| e.to_string())?;
                    self.asm.vmovss(dword_ptr(r8 + off), xmm0).map_err(|e| e.to_string())?;
                }
            }

            Ok(())
        }

        /// AVX-512 path for standalone reduction (Softmax).
        fn emit_standalone_reduction_avx512(
            &mut self,
            combine: &[TraceOp],
            second_pass: &Option<Box<crate::compiler::trace::ReductionSecondPass>>,
            normalize: &Option<Vec<TraceOp>>,
            identity: f64,
            elem_count: usize,
            vec_count: usize,
            tail: usize,
            total_vec_bytes: i32,
        ) -> Result<(), String> {
            let simd_w = 16usize;

            // Pass 1: Max reduction
            let identity_label = self.const_f32(identity as f32);
            self.asm.vbroadcastss(zmm0, dword_ptr(identity_label)).map_err(|e| e.to_string())?;

            if vec_count > 0 {
                let mut p1_loop = self.asm.create_label();
                let mut p1_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p1_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(p1_done).map_err(|e| e.to_string())?;

                self.emit_trace_on_stack_zmm(zmm0, combine, InputMode::ReductionCombine)?;

                self.asm.add(rbx, 64i32).map_err(|e| e.to_string())?;
                self.asm.jmp(p1_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p1_done).map_err(|e| e.to_string())?;
            }

            // Tail for pass 1 (masked)
            if tail > 0 {
                let mask = (1u32 << tail) - 1;
                self.asm.mov(eax, mask as i32).map_err(|e| e.to_string())?;
                self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
                // Load identity into zmm1, then masked load over it
                self.asm.vbroadcastss(zmm1, dword_ptr(identity_label)).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmm1.k1(), zmmword_ptr(rdi + total_vec_bytes)).map_err(|e| e.to_string())?;
                self.asm.vmaxps(zmm0, zmm0, zmm1).map_err(|e| e.to_string())?;
            }

            self.emit_horizontal_max_zmm(zmm0, zmm31)?;
            // zmm31 = broadcast max

            let second = match second_pass.as_deref() {
                Some(s) => s,
                None => return Ok(()),
            };

            // Pass 2: exp-sum
            let sp_identity_label = self.const_f32(second.identity as f32);
            self.asm.vbroadcastss(zmm0, dword_ptr(sp_identity_label)).map_err(|e| e.to_string())?;

            if vec_count > 0 {
                let mut p2_loop = self.asm.create_label();
                let mut p2_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p2_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(p2_done).map_err(|e| e.to_string())?;

                self.asm.vmovups(zmm28, zmm0).map_err(|e| e.to_string())?; // save acc
                self.emit_trace_on_stack_zmm(zmm0, &second.element_transform, InputMode::WithBroadcast)?;
                self.asm.vmovups(zmmword_ptr(r8 + rbx), zmm0).map_err(|e| e.to_string())?;
                self.asm.vaddps(zmm0, zmm28, zmm0).map_err(|e| e.to_string())?;

                self.asm.add(rbx, 64i32).map_err(|e| e.to_string())?;
                self.asm.jmp(p2_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p2_done).map_err(|e| e.to_string())?;
            }

            // Tail for pass 2
            if tail > 0 {
                let mask = (1u32 << tail) - 1;
                self.asm.mov(eax, mask as i32).map_err(|e| e.to_string())?;
                self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
                // Load input masked
                self.asm.vpxord(zmm1, zmm1, zmm1).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmm1.k1(), zmmword_ptr(rdi + total_vec_bytes)).map_err(|e| e.to_string())?;
                // sub max, exp
                self.asm.vsubps(zmm1, zmm1, zmm31).map_err(|e| e.to_string())?;
                self.emit_exp_avx512(zmm2, zmm1, [zmm1, zmm3, zmm4])?;
                self.asm.vmovups(zmmword_ptr(r8 + total_vec_bytes).k1(), zmm2).map_err(|e| e.to_string())?;
                self.asm.vaddps(zmm0, zmm0, zmm2).map_err(|e| e.to_string())?;
            }

            self.emit_horizontal_sum_zmm(zmm0, zmm31)?;
            self.asm.vrcp14ps(zmm31, zmm31).map_err(|e| e.to_string())?;
            // zmm31 = broadcast inv_sum

            let _norm_trace = match normalize {
                Some(n) => n,
                None => return Ok(()),
            };

            // Pass 3: Normalize
            if vec_count > 0 {
                let mut p3_loop = self.asm.create_label();
                let mut p3_done = self.asm.create_label();
                self.asm.xor(rbx, rbx).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p3_loop).map_err(|e| e.to_string())?;
                self.asm.cmp(rbx, total_vec_bytes).map_err(|e| e.to_string())?;
                self.asm.jge(p3_done).map_err(|e| e.to_string())?;

                self.asm.vmovups(zmm0, zmmword_ptr(r8 + rbx)).map_err(|e| e.to_string())?;
                self.asm.vmulps(zmm0, zmm0, zmm31).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmmword_ptr(r8 + rbx), zmm0).map_err(|e| e.to_string())?;

                self.asm.add(rbx, 64i32).map_err(|e| e.to_string())?;
                self.asm.jmp(p3_loop).map_err(|e| e.to_string())?;
                self.asm.set_label(&mut p3_done).map_err(|e| e.to_string())?;
            }

            if tail > 0 {
                let mask = (1u32 << tail) - 1;
                self.asm.mov(eax, mask as i32).map_err(|e| e.to_string())?;
                self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmm0.k1().z(), zmmword_ptr(r8 + total_vec_bytes)).map_err(|e| e.to_string())?;
                self.asm.vmulps(zmm0, zmm0, zmm31).map_err(|e| e.to_string())?;
                self.asm.vmovups(zmmword_ptr(r8 + total_vec_bytes).k1(), zmm0).map_err(|e| e.to_string())?;
            }

            Ok(())
        }

        // ── Horizontal reduction helpers ──────────────────────────────────

        /// Horizontal max across ymm → scalar broadcast into ymm_dst.
        /// Thin wrapper around SimdOps::vhmax.
        fn emit_horizontal_max_ymm(
            &mut self,
            ymm_src: AsmRegisterYmm,
            ymm_dst: AsmRegisterYmm,
        ) -> Result<(), String> {
            self.vhmax(VReg(ymm_index(ymm_dst)), VReg(ymm_index(ymm_src)))
        }

        /// Horizontal sum across ymm → scalar broadcast into ymm_dst.
        /// Thin wrapper around SimdOps::vhsum.
        fn emit_horizontal_sum_ymm(
            &mut self,
            ymm_src: AsmRegisterYmm,
            ymm_dst: AsmRegisterYmm,
        ) -> Result<(), String> {
            self.vhsum(VReg(ymm_index(ymm_dst)), VReg(ymm_index(ymm_src)))
        }

        /// Horizontal max across zmm → scalar broadcast into zmm_dst.
        /// Thin wrapper around SimdOps::vhmax.
        fn emit_horizontal_max_zmm(
            &mut self,
            zmm_src: AsmRegisterZmm,
            zmm_dst: AsmRegisterZmm,
        ) -> Result<(), String> {
            self.vhmax(VReg(zmm_index(zmm_dst)), VReg(zmm_index(zmm_src)))
        }

        /// Horizontal sum across zmm → scalar broadcast into zmm_dst.
        /// Thin wrapper around SimdOps::vhsum.
        fn emit_horizontal_sum_zmm(
            &mut self,
            zmm_src: AsmRegisterZmm,
            zmm_dst: AsmRegisterZmm,
        ) -> Result<(), String> {
            self.vhsum(VReg(zmm_index(zmm_dst)), VReg(zmm_index(zmm_src)))
        }

        // ── SimdOps helper methods ──────────────────────────────────────

        /// Map a `BaseReg` to an x86_64 64-bit GPR.
        fn gpr64(&self, base: BaseReg) -> Result<AsmRegister64, String> {
            match base {
                BaseReg::Arg(0) => Ok(rdi),
                BaseReg::Arg(1) => Ok(rsi),
                BaseReg::Arg(2) => Ok(rdx),
                BaseReg::Arg(3) => Ok(rcx),
                BaseReg::Arg(4) => Ok(r8),
                BaseReg::Arg(5) => Ok(r9),
                BaseReg::Arg(n) => Err(format!("Arg({}) exceeds SysV ABI register args", n)),
                BaseReg::StackPtr => Ok(rbp),
                BaseReg::ScratchpadBase => Ok(r15),
                BaseReg::LoopVar(0) => Ok(rbx),  // jc
                BaseReg::LoopVar(1) => Ok(r12),  // pc
                BaseReg::LoopVar(2) => Ok(r13),  // ic
                BaseReg::LoopVar(3) => Ok(r14),  // jr
                BaseReg::LoopVar(4) => Ok(r11),  // ir
                BaseReg::LoopVar(n) => Err(format!("LoopVar({}) out of range (0-4)", n)),
                BaseReg::OutputPtr => Ok(r8),
                BaseReg::Scratch(0) => Ok(rax),
                BaseReg::Scratch(1) => Ok(r10),
                BaseReg::Scratch(2) => Ok(r11),
                BaseReg::Scratch(n) => Err(format!("Scratch({}) out of range (0-2)", n)),
            }
        }

        /// Build an iced-x86 `AsmMemoryOperand` for ymm-width (256-bit) access.
        fn ymm_mem(&self, mem: MemOperand) -> Result<AsmMemoryOperand, String> {
            let base = self.gpr64(mem.base)?;
            Ok(ymmword_ptr(base + mem.offset))
        }

        /// Build an iced-x86 `AsmMemoryOperand` for zmm-width (512-bit) access.
        fn zmm_mem(&self, mem: MemOperand) -> Result<AsmMemoryOperand, String> {
            let base = self.gpr64(mem.base)?;
            Ok(zmmword_ptr(base + mem.offset))
        }

        /// Build an iced-x86 `AsmMemoryOperand` for dword (32-bit) access.
        fn dword_mem(&self, mem: MemOperand) -> Result<AsmMemoryOperand, String> {
            let base = self.gpr64(mem.base)?;
            Ok(dword_ptr(base + mem.offset))
        }

        /// Build an iced-x86 `AsmMemoryOperand` for byte access (prefetch).
        fn byte_mem(&self, mem: MemOperand) -> Result<AsmMemoryOperand, String> {
            let base = self.gpr64(mem.base)?;
            Ok(byte_ptr(base + mem.offset))
        }

    }


    // ── SimdOps trait implementation ─────────────────────────────────────

    impl SimdOps for X86CodeGen {
        // ── Vector arithmetic ───────────────────────────────────────────

        fn vadd(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vaddps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vaddps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vsub(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vsubps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vsubps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vmul(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vmulps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vmulps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vdiv(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vdivps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vdivps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vfma(&mut self, dst: VReg, a: VReg, b: VReg, c: VReg) -> Result<(), String> {
            // dst = a * b + c  →  move c to dst, then vfmadd231(dst, a, b)
            if dst != c {
                self.vmov(dst, c)?;
            }
            self.vfmadd231(dst, a, b)
        }

        fn vneg(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            // 0 - x
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(src.0 as usize)?;
                self.asm.vpxord(d, d, d).map_err(|e| e.to_string())?;
                self.asm.vsubps(d, d, s).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(src.0 as usize)?;
                self.asm.vxorps(d, d, d).map_err(|e| e.to_string())?;
                self.asm.vsubps(d, d, s).map_err(|e| e.to_string())
            }
        }

        fn vabs(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            let abs_mask = self.const_f32(f32::from_bits(0x7FFF_FFFF));
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(src.0 as usize)?;
                self.asm.vbroadcastss(d, dword_ptr(abs_mask)).map_err(|e| e.to_string())?;
                self.asm.vandps(d, s, d).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(src.0 as usize)?;
                self.asm.vandps(d, s, ymmword_ptr(abs_mask)).map_err(|e| e.to_string())
            }
        }

        fn vsqrt(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(src.0 as usize)?;
                self.asm.vsqrtps(d, s).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(src.0 as usize)?;
                self.asm.vsqrtps(d, s).map_err(|e| e.to_string())
            }
        }

        fn vmax(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vmaxps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vmaxps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vmin(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vminps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vminps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        // ── Approximate reciprocals ─────────────────────────────────────

        fn vrecip(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(src.0 as usize)?;
                self.asm.vrcp14ps(d, s).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(src.0 as usize)?;
                self.asm.vrcpps(d, s).map_err(|e| e.to_string())
            }
        }

        fn vrsqrt(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(src.0 as usize)?;
                self.asm.vrsqrt14ps(d, s).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(src.0 as usize)?;
                self.asm.vrsqrtps(d, s).map_err(|e| e.to_string())
            }
        }

        // ── Memory operations ───────────────────────────────────────────

        fn vload(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let m = self.zmm_mem(mem)?;
                self.asm.vmovups(d, m).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let m = self.ymm_mem(mem)?;
                self.asm.vmovups(d, m).map_err(|e| e.to_string())
            }
        }

        fn vstore(&mut self, mem: MemOperand, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let s = Self::zmm_any(src.0 as usize)?;
                let m = self.zmm_mem(mem)?;
                self.asm.vmovups(m, s).map_err(|e| e.to_string())
            } else {
                let s = Self::ymm_any(src.0 as usize)?;
                let m = self.ymm_mem(mem)?;
                self.asm.vmovups(m, s).map_err(|e| e.to_string())
            }
        }

        fn vbroadcast(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String> {
            let m = self.dword_mem(mem)?;
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                self.asm.vbroadcastss(d, m).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                self.asm.vbroadcastss(d, m).map_err(|e| e.to_string())
            }
        }

        fn vbroadcast_const(&mut self, dst: VReg, val: f32) -> Result<(), String> {
            let label = self.const_f32(val);
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                self.asm.vbroadcastss(d, dword_ptr(label)).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                self.asm.vbroadcastss(d, dword_ptr(label)).map_err(|e| e.to_string())
            }
        }

        fn vzero(&mut self, dst: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                self.asm.vpxord(d, d, d).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                self.asm.vxorps(d, d, d).map_err(|e| e.to_string())
            }
        }

        fn vmov(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            if dst == src { return Ok(()); }
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(src.0 as usize)?;
                self.asm.vmovaps(d, s).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(src.0 as usize)?;
                self.asm.vmovaps(d, s).map_err(|e| e.to_string())
            }
        }

        // ── Bitwise / integer operations ────────────────────────────────

        fn vand(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vandps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vandps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vorps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vorps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vxor(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vpxord(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vxorps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vshr_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(a.0 as usize)?;
                self.asm.vpsrld(d, s, imm as i32).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(a.0 as usize)?;
                self.asm.vpsrld(d, s, imm as i32).map_err(|e| e.to_string())
            }
        }

        fn vshl_i32(&mut self, dst: VReg, a: VReg, imm: u8) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(a.0 as usize)?;
                self.asm.vpslld(d, s, imm as i32).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(a.0 as usize)?;
                self.asm.vpslld(d, s, imm as i32).map_err(|e| e.to_string())
            }
        }

        fn vcvt_i32_f32(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(a.0 as usize)?;
                self.asm.vcvtdq2ps(d, s).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(a.0 as usize)?;
                self.asm.vcvtdq2ps(d, s).map_err(|e| e.to_string())
            }
        }

        fn vcvt_f32_i32(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(a.0 as usize)?;
                self.asm.vcvttps2dq(d, s).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(a.0 as usize)?;
                self.asm.vcvttps2dq(d, s).map_err(|e| e.to_string())
            }
        }

        fn vround(&mut self, dst: VReg, a: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let s = Self::zmm_any(a.0 as usize)?;
                self.asm.vrndscaleps(d, s, 0x00).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let s = Self::ymm_any(a.0 as usize)?;
                self.asm.vroundps(d, s, 0x00).map_err(|e| e.to_string())
            }
        }

        fn vadd_i32(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vpaddd(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vpaddd(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        // ── FMA variants ────────────────────────────────────────────────

        fn vfmadd213(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vfmadd213ps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vfmadd213ps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vfmadd231(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vfmadd231ps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vfmadd231ps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        fn vfnmadd231(&mut self, dst: VReg, a: VReg, b: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let ra = Self::zmm_any(a.0 as usize)?;
                let rb = Self::zmm_any(b.0 as usize)?;
                self.asm.vfnmadd231ps(d, ra, rb).map_err(|e| e.to_string())
            } else {
                let d = Self::ymm_any(dst.0 as usize)?;
                let ra = Self::ymm_any(a.0 as usize)?;
                let rb = Self::ymm_any(b.0 as usize)?;
                self.asm.vfnmadd231ps(d, ra, rb).map_err(|e| e.to_string())
            }
        }

        // ── Loop control ────────────────────────────────────────────────

        fn alloc_label(&mut self) -> Label {
            let lbl = self.asm.create_label();
            let idx = self.simd_labels.len();
            self.simd_labels.push(lbl);
            Label(idx as u32)
        }

        fn define_label(&mut self, label: Label) -> Result<(), String> {
            let lbl = &mut self.simd_labels[label.0 as usize];
            self.asm.set_label(lbl).map_err(|e| e.to_string())
        }

        fn jump(&mut self, label: Label) -> Result<(), String> {
            let lbl = self.simd_labels[label.0 as usize];
            self.asm.jmp(lbl).map_err(|e| e.to_string())
        }

        fn dec_and_branch_nz(&mut self, counter: BaseReg, label: Label) -> Result<(), String> {
            let r = self.gpr64(counter)?;
            self.asm.dec(r).map_err(|e| e.to_string())?;
            let lbl = self.simd_labels[label.0 as usize];
            self.asm.jne(lbl).map_err(|e| e.to_string())
        }

        fn cmp_and_branch_lt(&mut self, reg: BaseReg, imm: i64, label: Label) -> Result<(), String> {
            let r = self.gpr64(reg)?;
            self.asm.cmp(r, imm as i32).map_err(|e| e.to_string())?;
            let lbl = self.simd_labels[label.0 as usize];
            self.asm.jl(lbl).map_err(|e| e.to_string())
        }

        fn cmp_and_branch_ge(&mut self, reg: BaseReg, imm: i64, label: Label) -> Result<(), String> {
            let r = self.gpr64(reg)?;
            self.asm.cmp(r, imm as i32).map_err(|e| e.to_string())?;
            let lbl = self.simd_labels[label.0 as usize];
            self.asm.jge(lbl).map_err(|e| e.to_string())
        }

        // ── GPR operations ──────────────────────────────────────────────

        fn gpr_load_imm(&mut self, dst: BaseReg, imm: i64) -> Result<(), String> {
            let r = self.gpr64(dst)?;
            self.asm.mov(r, imm).map_err(|e| e.to_string())
        }

        fn gpr_add_imm(&mut self, dst: BaseReg, imm: i32) -> Result<(), String> {
            let r = self.gpr64(dst)?;
            self.asm.add(r, imm).map_err(|e| e.to_string())
        }

        fn gpr_mov(&mut self, dst: BaseReg, src: BaseReg) -> Result<(), String> {
            let rd = self.gpr64(dst)?;
            let rs = self.gpr64(src)?;
            self.asm.mov(rd, rs).map_err(|e| e.to_string())
        }

        // ── Function frame ──────────────────────────────────────────────

        fn emit_prologue(&mut self) -> Result<(), String> {
            self.asm.push(rbp).map_err(|e| e.to_string())?;
            self.asm.mov(rbp, rsp).map_err(|e| e.to_string())?;
            self.asm.push(rbx).map_err(|e| e.to_string())?;
            self.asm.push(r12).map_err(|e| e.to_string())?;
            self.asm.push(r13).map_err(|e| e.to_string())?;
            self.asm.push(r14).map_err(|e| e.to_string())?;
            self.asm.push(r15).map_err(|e| e.to_string())?;
            self.asm.sub(rsp, 8i32).map_err(|e| e.to_string())?;
            Ok(())
        }

        fn emit_epilogue(&mut self) -> Result<(), String> {
            self.asm.add(rsp, 8i32).map_err(|e| e.to_string())?;
            self.asm.pop(r15).map_err(|e| e.to_string())?;
            self.asm.pop(r14).map_err(|e| e.to_string())?;
            self.asm.pop(r13).map_err(|e| e.to_string())?;
            self.asm.pop(r12).map_err(|e| e.to_string())?;
            self.asm.pop(rbx).map_err(|e| e.to_string())?;
            self.asm.pop(rbp).map_err(|e| e.to_string())?;
            self.asm.ret().map_err(|e| e.to_string())?;
            Ok(())
        }

        fn finalize(&mut self) -> Result<CodegenOutput, String> {
            self.emit_const_pool()?;
            let code = self.asm.assemble(0x0)
                .map_err(|e| format!("asm error: {e}"))?;
            Ok(CodegenOutput {
                code,
                scratchpad_bytes: 0, // caller adjusts as needed
            })
        }

        // ── Non-temporal store ──────────────────────────────────────────

        fn vstore_nt(&mut self, mem: MemOperand, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let s = Self::zmm_any(src.0 as usize)?;
                let m = self.zmm_mem(mem)?;
                self.asm.vmovntps(m, s).map_err(|e| e.to_string())
            } else {
                let s = Self::ymm_any(src.0 as usize)?;
                let m = self.ymm_mem(mem)?;
                self.asm.vmovntps(m, s).map_err(|e| e.to_string())
            }
        }

        // ── Memory fence ────────────────────────────────────────────────

        fn sfence(&mut self) -> Result<(), String> {
            self.asm.sfence().map_err(|e| e.to_string())
        }

        // ── Prefetch ────────────────────────────────────────────────────

        fn prefetch_l1(&mut self, mem: MemOperand) -> Result<(), String> {
            let m = self.byte_mem(mem)?;
            self.asm.prefetcht0(m).map_err(|e| e.to_string())
        }

        // ── Scalar operations ───────────────────────────────────────────

        fn scalar_load(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String> {
            let m = self.dword_mem(mem)?;
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                // Zero the register first, then load scalar into lowest lane
                self.asm.vpxord(d, d, d).map_err(|e| e.to_string())?;
                // vmovss from memory zeros upper bits of xmm, and zmm upper is already zero
                let xd = Self::xmm_from_index(dst.0 as usize)?;
                self.asm.vmovss(xd, m).map_err(|e| e.to_string())
            } else {
                let xd = Self::xmm_from_index(dst.0 as usize)?;
                // vmovss from memory zeros upper bits of xmm; upper ymm lanes are zeroed by VEX
                self.asm.vmovss(xd, m).map_err(|e| e.to_string())
            }
        }

        fn scalar_store(&mut self, mem: MemOperand, src: VReg) -> Result<(), String> {
            let m = self.dword_mem(mem)?;
            let xs = Self::xmm_from_index(src.0 as usize)?;
            self.asm.vmovss(m, xs).map_err(|e| e.to_string())
        }

        // ── External function calls ─────────────────────────────────────

        fn call_fn_ptr(&mut self, addr: u64) -> Result<(), String> {
            self.asm.mov(rax, addr as i64).map_err(|e| e.to_string())?;
            self.asm.call(rax).map_err(|e| e.to_string())
        }

        // ── Horizontal reductions ───────────────────────────────────────

        fn vhsum(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let zmm_src = Self::zmm_any(src.0 as usize)?;
                let zmm_dst = Self::zmm_any(dst.0 as usize)?;
                // 512→256
                self.asm.vextractf64x4(ymm2, zmm_src, 1u32).map_err(|e| e.to_string())?;
                self.asm.vaddps(ymm2, ymm2, ymm0).map_err(|e| e.to_string())?;
                // 256→128
                self.asm.vextractf128(xmm1, ymm2, 1u32).map_err(|e| e.to_string())?;
                self.asm.vaddps(xmm0, xmm0, xmm1).map_err(|e| e.to_string())?;
                // 128→64
                self.asm.vshufps(xmm1, xmm0, xmm0, 0x4Eu32).map_err(|e| e.to_string())?;
                self.asm.vaddps(xmm0, xmm0, xmm1).map_err(|e| e.to_string())?;
                // 64→32
                self.asm.vshufps(xmm1, xmm0, xmm0, 0xB1u32).map_err(|e| e.to_string())?;
                self.asm.vaddps(xmm0, xmm0, xmm1).map_err(|e| e.to_string())?;
                // Broadcast scalar to zmm
                self.asm.vbroadcastss(zmm_dst, xmm0).map_err(|e| e.to_string())?;
                Ok(())
            } else {
                let ymm_src = Self::ymm_any(src.0 as usize)?;
                let ymm_dst = Self::ymm_any(dst.0 as usize)?;
                // 256→128
                self.asm.vextractf128(xmm2, ymm_src, 1u32).map_err(|e| e.to_string())?;
                self.asm.vextractf128(xmm0, ymm_src, 0u32).map_err(|e| e.to_string())?;
                self.asm.vaddps(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                // 128→64
                self.asm.vshufps(xmm2, xmm0, xmm0, 0x4Eu32).map_err(|e| e.to_string())?;
                self.asm.vaddps(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                // 64→32
                self.asm.vshufps(xmm2, xmm0, xmm0, 0xB1u32).map_err(|e| e.to_string())?;
                self.asm.vaddps(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                // Broadcast scalar to ymm
                self.asm.vbroadcastss(ymm_dst, xmm0).map_err(|e| e.to_string())?;
                Ok(())
            }
        }

        fn vhmax(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let zmm_src = Self::zmm_any(src.0 as usize)?;
                let zmm_dst = Self::zmm_any(dst.0 as usize)?;
                // 512→256
                self.asm.vextractf64x4(ymm2, zmm_src, 1u32).map_err(|e| e.to_string())?;
                self.asm.vmaxps(ymm2, ymm2, ymm0).map_err(|e| e.to_string())?;
                // 256→128
                self.asm.vextractf128(xmm1, ymm2, 1u32).map_err(|e| e.to_string())?;
                self.asm.vmaxps(xmm0, xmm0, xmm1).map_err(|e| e.to_string())?;
                // 128→64
                self.asm.vshufps(xmm1, xmm0, xmm0, 0x4Eu32).map_err(|e| e.to_string())?;
                self.asm.vmaxps(xmm0, xmm0, xmm1).map_err(|e| e.to_string())?;
                // 64→32
                self.asm.vshufps(xmm1, xmm0, xmm0, 0xB1u32).map_err(|e| e.to_string())?;
                self.asm.vmaxps(xmm0, xmm0, xmm1).map_err(|e| e.to_string())?;
                // Broadcast scalar to zmm
                self.asm.vbroadcastss(zmm_dst, xmm0).map_err(|e| e.to_string())?;
                Ok(())
            } else {
                let ymm_src = Self::ymm_any(src.0 as usize)?;
                let ymm_dst = Self::ymm_any(dst.0 as usize)?;
                // 256→128
                self.asm.vextractf128(xmm2, ymm_src, 1u32).map_err(|e| e.to_string())?;
                self.asm.vextractf128(xmm0, ymm_src, 0u32).map_err(|e| e.to_string())?;
                self.asm.vmaxps(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                // 128→64
                self.asm.vshufps(xmm2, xmm0, xmm0, 0x4Eu32).map_err(|e| e.to_string())?;
                self.asm.vmaxps(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                // 64→32
                self.asm.vshufps(xmm2, xmm0, xmm0, 0xB1u32).map_err(|e| e.to_string())?;
                self.asm.vmaxps(xmm0, xmm0, xmm2).map_err(|e| e.to_string())?;
                // Broadcast scalar to ymm
                self.asm.vbroadcastss(ymm_dst, xmm0).map_err(|e| e.to_string())?;
                Ok(())
            }
        }

        // ── Masked operations ───────────────────────────────────────────

        fn set_tail_mask(&mut self, remainder: usize) -> Result<(), String> {
            if self.use_avx512 {
                // AVX-512: set k1 = (1 << remainder) - 1
                let mask_val = (1u32 << remainder) - 1;
                self.asm.mov(eax, mask_val as i32).map_err(|e| e.to_string())?;
                self.asm.kmovw(k1, eax).map_err(|e| e.to_string())?;
            }
            // AVX2/NEON: no-op (caller uses scalar fallback)
            Ok(())
        }

        fn vload_masked(&mut self, dst: VReg, mem: MemOperand) -> Result<(), String> {
            if self.use_avx512 {
                let d = Self::zmm_any(dst.0 as usize)?;
                let m = self.zmm_mem(mem)?;
                self.asm.vmovups(d.k1().z(), m).map_err(|e| e.to_string())
            } else {
                // AVX2: no hardware mask support; caller should use scalar fallback
                Err("vload_masked not supported on AVX2 (use scalar fallback)".into())
            }
        }

        fn vstore_masked(&mut self, mem: MemOperand, src: VReg) -> Result<(), String> {
            if self.use_avx512 {
                let s = Self::zmm_any(src.0 as usize)?;
                let m = self.zmm_mem(mem)?;
                self.asm.vmovups(m.k1(), s).map_err(|e| e.to_string())
            } else {
                Err("vstore_masked not supported on AVX2 (use scalar fallback)".into())
            }
        }

        // ── NOP ─────────────────────────────────────────────────────────

        fn emit_metadata_nop(&mut self) -> Result<(), String> {
            self.asm.nop().map_err(|e| e.to_string())
        }

        // ── Mixed-width support ─────────────────────────────────────────

        fn current_simd_width(&self) -> crate::compiler::codegen::simd_ops::SimdWidth {
            use crate::compiler::codegen::simd_ops::SimdWidth;
            if self.use_avx512 {
                SimdWidth::W512
            } else {
                SimdWidth::W256
            }
        }

        fn width_f32_lanes(&self) -> usize {
            self.simd_width
        }

        fn push_width(&mut self, w: crate::compiler::codegen::simd_ops::SimdWidth) -> Result<crate::compiler::codegen::simd_ops::SimdWidth, String> {
            use crate::compiler::codegen::simd_ops::SimdWidth;
            let prev = self.current_simd_width();
            match w {
                SimdWidth::W128 => {
                    self.width_stack.push(prev);
                    self.use_avx512 = false;
                    self.simd_width = 4;
                    Ok(prev)
                }
                SimdWidth::W256 => {
                    self.width_stack.push(prev);
                    self.use_avx512 = false;
                    self.simd_width = 8;
                    Ok(prev)
                }
                SimdWidth::W512 => {
                    if self.has_avx512 {
                        self.width_stack.push(prev);
                        self.use_avx512 = true;
                        self.simd_width = 16;
                        Ok(prev)
                    } else {
                        Err("AVX-512 not available on this hardware".into())
                    }
                }
                SimdWidth::Wvl => {
                    Err("SVE variable-length not supported on x86_64".into())
                }
            }
        }

        fn pop_width(&mut self, prev: crate::compiler::codegen::simd_ops::SimdWidth) -> Result<(), String> {
            use crate::compiler::codegen::simd_ops::SimdWidth;
            if self.width_stack.pop().is_none() {
                return Err("pop_width called without matching push_width".into());
            }
            match prev {
                SimdWidth::W128 => {
                    self.use_avx512 = false;
                    self.simd_width = 4;
                }
                SimdWidth::W256 => {
                    self.use_avx512 = false;
                    self.simd_width = 8;
                }
                SimdWidth::W512 => {
                    self.use_avx512 = true;
                    self.simd_width = 16;
                }
                SimdWidth::Wvl => {
                    return Err("SVE variable-length not supported on x86_64".into());
                }
            }
            Ok(())
        }
    }

    // ── TileOps implementation (AMX) ────────────────────────────────────

    impl crate::compiler::codegen::tile_ops::TileOps for X86CodeGen {
        fn tile_configure(&mut self, configs: &[(crate::compiler::codegen::tile_ops::TReg, crate::compiler::codegen::tile_ops::TileConfig)]) -> Result<(), String> {
            if !self.has_amx {
                return Err("AMX not available on this target".into());
            }
            use crate::compiler::codegen::x86_amx::TileCfg;
            let mut cfg = TileCfg::zeroed();
            cfg.data[0] = 1; // palette 1
            for (treg, tc) in configs {
                let idx = treg.0;
                if idx >= 8 {
                    return Err(format!("AMX tile index {} out of range (0..7)", idx));
                }
                let i = idx as usize;
                let cb = tc.cols_bytes.to_le_bytes();
                cfg.data[16 + 2 * i] = cb[0];
                cfg.data[16 + 2 * i + 1] = cb[1];
                cfg.data[48 + i] = tc.rows;
            }
            // Use stack-based LDTILECFG (same approach as x86_amx::jit::emit_ldtilecfg)
            crate::compiler::codegen::x86_amx::jit::emit_ldtilecfg(&mut self.asm, &cfg)
        }

        fn tile_release(&mut self) -> Result<(), String> {
            if !self.has_amx {
                return Err("AMX not available on this target".into());
            }
            self.asm.tilerelease().map_err(|e| format!("tilerelease: {e}"))
        }

        fn tile_zero(&mut self, dst: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            if !self.has_amx {
                return Err("AMX not available on this target".into());
            }
            let tmm = crate::compiler::codegen::x86_amx::jit::tmm_reg(dst.0)?;
            self.asm.tilezero(tmm).map_err(|e| format!("tilezero: {e}"))
        }

        fn tile_load(&mut self, dst: crate::compiler::codegen::tile_ops::TReg, base: BaseReg, stride: BaseReg) -> Result<(), String> {
            if !self.has_amx {
                return Err("AMX not available on this target".into());
            }
            let tmm = crate::compiler::codegen::x86_amx::jit::tmm_reg(dst.0)?;
            let base_gpr = self.gpr64(base)?;
            let stride_gpr = self.gpr64(stride)?;
            // TILELOADD tmm, [base + stride*1] — stride register encodes row stride
            self.asm.tileloadd(tmm, ptr(base_gpr + stride_gpr))
                .map_err(|e| format!("tileloadd: {e}"))
        }

        fn tile_store(&mut self, base: BaseReg, stride: BaseReg, src: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            if !self.has_amx {
                return Err("AMX not available on this target".into());
            }
            let tmm = crate::compiler::codegen::x86_amx::jit::tmm_reg(src.0)?;
            let base_gpr = self.gpr64(base)?;
            let stride_gpr = self.gpr64(stride)?;
            // TILESTORED [base + stride*1], tmm
            self.asm.tilestored(ptr(base_gpr + stride_gpr), tmm)
                .map_err(|e| format!("tilestored: {e}"))
        }

        fn tile_dpbf16(&mut self, dst: crate::compiler::codegen::tile_ops::TReg, a: crate::compiler::codegen::tile_ops::TReg, b: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            if !self.has_amx {
                return Err("AMX not available on this target".into());
            }
            let d = crate::compiler::codegen::x86_amx::jit::tmm_reg(dst.0)?;
            let ta = crate::compiler::codegen::x86_amx::jit::tmm_reg(a.0)?;
            let tb = crate::compiler::codegen::x86_amx::jit::tmm_reg(b.0)?;
            self.asm.tdpbf16ps(d, ta, tb).map_err(|e| format!("tdpbf16ps: {e}"))
        }

        fn tile_dpbssd(&mut self, dst: crate::compiler::codegen::tile_ops::TReg, a: crate::compiler::codegen::tile_ops::TReg, b: crate::compiler::codegen::tile_ops::TReg) -> Result<(), String> {
            if !self.has_amx {
                return Err("AMX not available on this target".into());
            }
            let d = crate::compiler::codegen::x86_amx::jit::tmm_reg(dst.0)?;
            let ta = crate::compiler::codegen::x86_amx::jit::tmm_reg(a.0)?;
            let tb = crate::compiler::codegen::x86_amx::jit::tmm_reg(b.0)?;
            self.asm.tdpbssd(d, ta, tb).map_err(|e| format!("tdpbssd: {e}"))
        }

        fn tile_max_rows(&self) -> u8 { 16 }
        fn tile_max_cols_bytes(&self) -> u16 { 64 }
        fn tile_count(&self) -> u8 { 8 }
        fn has_tile_ops(&self) -> bool { self.has_amx }

        fn tile_accel_kind(&self) -> Option<crate::compiler::codegen::tile_ops::TileAccelKind> {
            if self.has_amx {
                Some(crate::compiler::codegen::tile_ops::TileAccelKind::Amx)
            } else {
                None
            }
        }

        fn tile_supported_dtypes(&self) -> Vec<crate::compiler::codegen::tile_ops::TileDtype> {
            if self.has_amx {
                vec![
                    crate::compiler::codegen::tile_ops::TileDtype::BF16,
                    crate::compiler::codegen::tile_ops::TileDtype::INT8,
                ]
            } else {
                vec![]
            }
        }
    }



    // ── AVX-512 codegen unit tests ──────────────────────────────────────

    #[cfg(test)]
    #[cfg(target_arch = "x86_64")]
    mod avx512_tests {
        use super::*;
        use crate::compiler::trace::TraceOp;
        use crate::dispatch::DeviceProfile;
        use iced_x86::code_asm::*;

        /// Helper: create a DeviceProfile with use_avx512 forced on.
        fn avx512_profile() -> DeviceProfile {
            let mut profile = DeviceProfile::detect();
            profile.kernel_config.use_avx512 = true;
            profile
        }

        #[test]
        fn test_zmm_for_index_valid() {
            for i in 0..28 {
                let result = X86CodeGen::zmm_for_index(i);
                assert!(result.is_ok(), "zmm_for_index({i}) should succeed");
            }
        }

        #[test]
        fn test_zmm_for_index_overflow() {
            let result = X86CodeGen::zmm_for_index(28);
            assert!(result.is_err(), "zmm_for_index(28) should fail");
            let msg = result.unwrap_err();
            assert!(
                msg.contains("exceeds available zmm accumulators"),
                "unexpected error: {msg}"
            );
        }

        #[test]
        fn test_zmm_any_valid() {
            for i in 0..32 {
                let result = X86CodeGen::zmm_any(i);
                assert!(result.is_ok(), "zmm_any({i}) should succeed");
            }
        }

        #[test]
        fn test_zmm_any_overflow() {
            let result = X86CodeGen::zmm_any(32);
            assert!(result.is_err(), "zmm_any(32) should fail");
            let msg = result.unwrap_err();
            assert!(
                msg.contains("zmm index 32 out of range"),
                "unexpected error: {msg}"
            );
        }

        #[test]
        fn test_avx512_exp_compiles() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let result = codegen.emit_exp_avx512(zmm0, zmm1, [zmm29, zmm30, zmm31]);
            assert!(result.is_ok(), "emit_exp_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_tanh_compiles() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let result = codegen.emit_tanh_avx512(zmm0, zmm1, [zmm29, zmm30, zmm31]);
            assert!(result.is_ok(), "emit_tanh_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_log_compiles() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let result = codegen.emit_log_avx512(zmm0, zmm1, [zmm29, zmm30, zmm31]);
            assert!(result.is_ok(), "emit_log_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_trace_on_accumulator() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let body = vec![
                TraceOp::Input(0),
                TraceOp::Neg(0),
            ];
            let result = codegen.emit_trace_on_accumulator_avx512(zmm0, &body);
            assert!(result.is_ok(), "emit_trace_on_accumulator_avx512 failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_elementwise_trace_body() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let body = vec![
                TraceOp::Input(0),
                TraceOp::Neg(0),
            ];
            let result = codegen.emit_trace_on_stack_zmm(zmm0, &body, InputMode::Elementwise { is_binary: false, scalar_tail: false });
            assert!(result.is_ok(), "emit_trace_on_stack_zmm elementwise failed: {:?}", result.err());
        }

        #[test]
        fn test_avx512_trace_ops_silu() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let ops = vec![
                TraceOp::Input(0),   // [0] v
                TraceOp::Neg(0),     // [1] -v
                TraceOp::Exp(1),     // [2] exp(-v)
                TraceOp::Const(1.0), // [3] 1.0
                TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
                TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
            ];
            let result = codegen.emit_trace_ops_avx512(&ops);
            assert!(result.is_ok(), "emit_trace_ops_avx512 SiLU failed: {:?}", result.err());
            assert_eq!(result.unwrap().len(), 6);
        }

        #[test]
        fn test_avx512_trace_ops_all_ops() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let ops = vec![
                TraceOp::Input(0),     // [0]
                TraceOp::Input(1),     // [1]
                TraceOp::Neg(0),       // [2]
                TraceOp::Abs(0),       // [3]
                TraceOp::Exp(0),       // [4]
                TraceOp::Sqrt(0),      // [5]
                TraceOp::Rsqrt(0),     // [6]
                TraceOp::Tanh(0),      // [7]
                TraceOp::Recip(0),     // [8]
                TraceOp::Log(0),       // [9]
                TraceOp::Add(0, 1),    // [10]
                TraceOp::Sub(0, 1),    // [11]
                TraceOp::Mul(0, 1),    // [12]
                TraceOp::Div(0, 1),    // [13]
                TraceOp::Max(0, 1),    // [14]
                TraceOp::Min(0, 1),    // [15]
                TraceOp::Fma(0, 1, 2), // [16]
                TraceOp::Const(2.0),   // [17]
            ];
            let result = codegen.emit_trace_ops_avx512(&ops);
            assert!(result.is_ok(), "emit_trace_ops_avx512 all ops failed: {:?}", result.err());
            assert_eq!(result.unwrap().len(), 18);
        }

        #[test]
        fn test_avx512_trace_on_accumulator_with_bias() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            let body = vec![
                TraceOp::Input(0),   // accumulator
                TraceOp::Input(1),   // bias from external
                TraceOp::Add(0, 1),  // acc + bias
            ];
            let result = codegen.emit_trace_on_accumulator_with_bias_avx512(zmm0, &body, 0i32);
            assert!(
                result.is_ok(),
                "emit_trace_on_accumulator_with_bias_avx512 failed: {:?}",
                result.err()
            );
        }

        #[test]
        fn test_avx512_trace_ops_overflow() {
            let profile = avx512_profile();
            let mut codegen = X86CodeGen::new(&profile);
            // zmm0-zmm28 are usable (29 regs); zmm29-zmm31 reserved as scratch.
            // 29 inputs fill zmm0-zmm28, index 29 should fail.
            let mut ops: Vec<TraceOp> = (0..29).map(|i| TraceOp::Input(i as u32)).collect();
            ops.push(TraceOp::Add(0, 1)); // SSA index 29
            let result = codegen.emit_trace_ops_avx512(&ops);
            assert!(result.is_err());
            let msg = result.unwrap_err();
            assert!(
                msg.contains("SSA index 29 exceeds available registers"),
                "unexpected error: {msg}"
            );
        }

        #[test]
        fn test_simd_bytes_avx2_vs_avx512() {
            // AVX2 mode
            let mut p = DeviceProfile::detect();
            p.kernel_config.use_avx512 = false;
            let cg = X86CodeGen::new(&p);
            assert_eq!(cg.simd_bytes(), 32, "AVX2 simd_bytes should be 32");

            // AVX-512 mode
            let profile = avx512_profile();
            let cg512 = X86CodeGen::new(&profile);
            assert_eq!(cg512.simd_bytes(), 64, "AVX-512 simd_bytes should be 64");
        }
    }
}

#[cfg(feature = "jit-x86")]
impl crate::compiler::codegen::emitter::MachineCodeEmitter for jit::X86CodeGen {
    fn emit_plan(
        &mut self,
        plan: &crate::compiler::fusion::FusionPlan,
        graph: &crate::compiler::graph::CompilerGraph,
        alloc: &crate::compiler::buffer_alloc::BufferAllocation,
        profile: &crate::dispatch::DeviceProfile,
        registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, String> {
        self.emit_plan(plan, graph, alloc, profile, registry)
    }

    fn simd_width(&self) -> usize {
        self.simd_width()
    }
}

/// x86_64 platform backend factory.
#[cfg(feature = "jit-x86")]
pub struct X86Backend;

#[cfg(feature = "jit-x86")]
impl crate::compiler::codegen::emitter::PlatformBackend for X86Backend {
    type Emitter = jit::X86CodeGen;

    fn new_emitter(&self, profile: &crate::dispatch::DeviceProfile) -> Self::Emitter {
        jit::X86CodeGen::new(profile)
    }

    fn platform(&self) -> crate::compiler::codegen::emitter::Platform {
        #[cfg(target_arch = "x86_64")]
        let avx512 = std::is_x86_feature_detected!("avx512f");
        #[cfg(not(target_arch = "x86_64"))]
        let avx512 = false;
        #[cfg(target_arch = "x86_64")]
        let amx = std::is_x86_feature_detected!("amx-tile") && std::is_x86_feature_detected!("amx-bf16");
        #[cfg(not(target_arch = "x86_64"))]
        let amx = false;
        crate::compiler::codegen::emitter::Platform::X86_64 { avx512, amx }
    }

    fn num_simd_regs(&self) -> usize {
        16 // x86_64: ymm0-ymm15 (AVX2) or zmm0-zmm15 (AVX-512)
    }
}

#[cfg(feature = "jit-x86")]
pub use jit::X86CodeGen;

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::compiler::executable::CompiledLayer;

    #[test]
    fn test_x86_stub_callable() {
        let output = emit_stub();
        assert!(!output.code.is_empty());
        let layer = CompiledLayer::from_code(&output.code, 0, 0).unwrap();
        unsafe {
            let f = layer.entry_point();
            f(
                std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                std::ptr::null(), std::ptr::null(),
                0, 0,
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "jit-x86")]
    mod jit_tests {
        use crate::compiler::trace::{TraceOp, ComputePattern};
        use crate::dispatch::DeviceProfile;
        use super::super::jit;

        #[test]
        fn test_emit_trace_ops_silu() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),   // [0] v
                TraceOp::Neg(0),     // [1] -v
                TraceOp::Exp(1),     // [2] exp(-v)
                TraceOp::Const(1.0), // [3] 1.0
                TraceOp::Add(2, 3),  // [4] 1 + exp(-v)
                TraceOp::Div(0, 4),  // [5] v / (1 + exp(-v))
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 6);
        }

        #[test]
        fn test_emit_trace_ops_add() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),
                TraceOp::Input(1),
                TraceOp::Add(0, 1),
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 3);
        }

        #[test]
        fn test_emit_trace_ops_gelu() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),              // [0] x
                TraceOp::Mul(0, 0),             // [1] x^2
                TraceOp::Mul(1, 0),             // [2] x^3
                TraceOp::Const(0.044715),       // [3] 0.044715
                TraceOp::Mul(3, 2),             // [4] 0.044715 * x^3
                TraceOp::Add(0, 4),             // [5] x + 0.044715 * x^3
                TraceOp::Const(0.7978845608),   // [6] sqrt(2/pi)
                TraceOp::Mul(6, 5),             // [7] sqrt(2/pi) * (...)
                TraceOp::Tanh(7),               // [8] tanh(...)
                TraceOp::Const(1.0),            // [9] 1.0
                TraceOp::Add(9, 8),             // [10] 1 + tanh(...)
                TraceOp::Mul(0, 10),            // [11] x * (1 + tanh(...))
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 12);
        }

        #[test]
        fn test_emit_trace_ops_all_unary() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),  // [0]
                TraceOp::Neg(0),    // [1]
                TraceOp::Abs(0),    // [2]
                TraceOp::Exp(0),    // [3]
                TraceOp::Sqrt(0),   // [4]
                TraceOp::Rsqrt(0),  // [5]
                TraceOp::Tanh(0),   // [6]
                TraceOp::Recip(0),  // [7]
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 8);
        }

        #[test]
        fn test_emit_trace_ops_all_binary() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),   // [0]
                TraceOp::Input(1),   // [1]
                TraceOp::Add(0, 1),  // [2]
                TraceOp::Sub(0, 1),  // [3]
                TraceOp::Mul(0, 1),  // [4]
                TraceOp::Div(0, 1),  // [5]
                TraceOp::Max(0, 1),  // [6]
                TraceOp::Min(0, 1),  // [7]
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 8);
        }

        #[test]
        fn test_emit_trace_ops_fma() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let ops = vec![
                TraceOp::Input(0),    // [0] a
                TraceOp::Input(1),    // [1] b
                TraceOp::Input(2),    // [2] c
                TraceOp::Fma(0, 1, 2), // [3] a*b + c
            ];

            let regs = codegen.emit_trace_ops_avx2(&ops).unwrap();
            assert_eq!(regs.len(), 4);
        }

        #[test]
        fn test_ymm_for_index_overflow() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);

            let mut ops: Vec<TraceOp> = (0..13).map(|i| TraceOp::Input(i as u32)).collect();
            ops.push(TraceOp::Add(0, 1)); // SSA index 13

            let result = codegen.emit_trace_ops_avx2(&ops);
            assert!(result.is_err());
            let msg = result.unwrap_err();
            assert!(
                msg.contains("SSA index 13 exceeds available registers"),
                "unexpected error: {msg}"
            );
        }

        #[test]
        fn test_emit_gemm_microkernel() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);
            codegen.emit_gemm_microkernel(1, 4096, 4096, &profile, &[]).unwrap();
        }

        #[test]
        fn test_emit_gemm_microkernel_large() {
            let profile = DeviceProfile::detect();
            let mut codegen = jit::X86CodeGen::new(&profile);
            codegen.emit_gemm_microkernel(512, 4096, 4096, &profile, &[]).unwrap();
        }

        #[test]
        fn test_emit_full_plan() {
            use crate::compiler::graph::CompilerGraph;
            use crate::compiler::ir::LayerIR;
            use crate::compiler::fusion::fuse_with_dag;
            use crate::compiler::registry::ScalarOpRegistry;
            use crate::compiler::buffer_alloc::{analyze_lifetimes, allocate_buffers};
            use crate::inference::types::ModelConfig;

            let config = ModelConfig::llama_7b();
            let ir = LayerIR::from_model_config(&config, 1);
            let profile = DeviceProfile::detect();
            let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
            let registry = ScalarOpRegistry::with_defaults();
            let plan = fuse_with_dag(&graph, &registry, &profile);
            let lifetimes = analyze_lifetimes(&graph, &plan);
            let alloc = allocate_buffers(&lifetimes);

            let mut codegen = jit::X86CodeGen::new(&profile);
            let output = codegen.emit_plan(&plan, &graph, &alloc, &profile, Some(&registry)).unwrap();

            assert!(!output.code.is_empty());
            assert!(output.scratchpad_bytes > 0);

            eprintln!("Generated {} bytes of x86_64 code", output.code.len());
        }
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "jit-x86")]
mod e2e_tests {
    use crate::compiler::codegen::x86_64::jit::X86CodeGen;
    use crate::compiler::executable::CompiledLayer;
    use crate::compiler::graph::{CompilerGraph, OpKind, OpId, TensorId, WeightLayout};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::buffer_alloc::{BufferAllocation, BufferSlot};
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::DType;
    use std::collections::HashMap;
    use crate::compiler::registry::{ScalarOpRegistry, OpKindKey};
    use crate::compiler::trace::{TraceOp, OpTrace, ComputePattern, ScalarFnSignature, ScalarParam, ReductionSecondPass};

    /// Number of f32 elements for e2e tests (must be multiple of 8 for AVX2).
    const N: usize = 32;

    /// Build a minimal graph with one unary elementwise op.
    /// Returns (graph, op_id, output_tensor_id).
    fn build_unary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![input], vec![output], "op");
        (g, op_id, output)
    }

    /// Build a minimal graph with one binary elementwise op.
    /// Returns (graph, op_id, output_tensor_id).
    fn build_binary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![N], DType::F32);
        let b = g.add_tensor("b", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![a, b], vec![output], "op");
        (g, op_id, output)
    }

    /// Build a FusionPlan with a single ElementwiseChain group.
    fn build_plan(op_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        }
    }

    /// Build a minimal BufferAllocation (no intermediate tensors needed).
    fn build_alloc(total_bytes: usize) -> BufferAllocation {
        BufferAllocation {
            slots: vec![],
            total_bytes,
            num_tensors: 0,
            bytes_saved: 0,
        }
    }

    /// Compile a graph+plan into executable code and return the CompiledLayer.
    fn compile(
        graph: &CompilerGraph,
        plan: &FusionPlan,
        alloc: &BufferAllocation,
    ) -> CompiledLayer {
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        let output = codegen.emit_plan(plan, graph, alloc, &profile, None).unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
    }

    /// Execute a compiled unary op.
    ///
    /// Codegen ABI: rdi=input (arg0), r8=output (arg4=seq_lens slot).
    unsafe fn exec_unary(layer: &CompiledLayer, input: &[f32], output: &mut [f32]) {
        let f = layer.entry_point();
        f(
            input.as_ptr() as *const u8,       // rdi = input
            std::ptr::null(),                   // rsi = weights (unused)
            std::ptr::null_mut(),               // rdx = kv_cache (unused)
            std::ptr::null(),                   // rcx = positions (unused)
            output.as_mut_ptr() as *const usize, // r8 = output ptr
            0,                                  // r9 = batch_size (unused)
            0,                                  // stack = seq_len (unused)
            std::ptr::null_mut(),               // stack = output (unused)
            std::ptr::null_mut(),               // stack = scratchpad (unused)
        );
    }

    /// Execute a compiled binary op.
    ///
    /// Codegen ABI: rdi=input_a (arg0), rsi=input_b (arg1=weights slot),
    ///              r8=output (arg4=seq_lens slot).
    unsafe fn exec_binary(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) {
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,            // rdi = input a
            b.as_ptr() as *const u8,            // rsi = input b (weights slot)
            std::ptr::null_mut(),                // rdx = kv_cache (unused)
            std::ptr::null(),                    // rcx = positions (unused)
            output.as_mut_ptr() as *const usize, // r8 = output ptr
            0,                                   // r9 = batch_size (unused)
            0,                                   // stack = seq_len (unused)
            std::ptr::null_mut(),                // stack = output (unused)
            std::ptr::null_mut(),                // stack = scratchpad (unused)
        );
    }

    /// Test data: a spread of values including negatives, zero, and positives.
    fn test_input() -> Vec<f32> {
        let base = [
            -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        ];
        (0..N).map(|i| base[i % base.len()]).collect()
    }

    fn test_input_b() -> Vec<f32> {
        (0..N).map(|i| 0.1 * (i as f32) + 0.5).collect()
    }

    /// Scalar SiLU reference: x / (1 + exp(-x))
    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Scalar GELU reference: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn ref_gelu(x: f32) -> f32 {
        let inner = (2.0_f32 / std::f32::consts::PI).sqrt()
            * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn test_e2e_silu_correctness() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Silu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("SiLU e2e: all {} elements within 1e-3 tolerance", N);
    }

    #[test]
    fn test_e2e_gelu_correctness() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Gelu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_gelu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "GELU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("GELU e2e: all {} elements within 1e-3 tolerance", N);
    }

    #[test]
    fn test_e2e_add_correctness() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Add);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] + b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Add mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Add e2e: all {} elements within 1e-5 tolerance", N);
    }

    #[test]
    fn test_e2e_mul_correctness() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Mul);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] * b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Mul mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Mul e2e: all {} elements within 1e-5 tolerance", N);
    }

    /// Build a minimal graph with one Gemm op (M×K @ K×N → M×N).
    fn build_gemm_graph(m: usize, n: usize, k: usize) -> (CompilerGraph, OpId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let c = g.add_tensor("C", vec![m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![c];
        let op_id = g.add_op(OpKind::Gemm { m, n, k }, vec![a, b], vec![c], "gemm");
        (g, op_id)
    }

    /// Build a FusionPlan with a single Standalone GEMM group.
    fn build_gemm_plan(op_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![op_id],
            }],
            op_to_group,
        }
    }

    /// Execute a compiled GEMM.
    ///
    /// Codegen ABI: rdi=A (arg0), rsi=B (arg1), r8=C (arg4=seq_lens slot).
    unsafe fn exec_gemm(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) {
        let mut scratch = vec![0u8; layer.scratchpad_bytes];
        let scratch_ptr = if layer.scratchpad_bytes > 0 {
            scratch.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,             // rdi = A
            b.as_ptr() as *const u8,             // rsi = B
            std::ptr::null_mut(),                 // rdx = kv_cache (unused)
            std::ptr::null(),                     // rcx = positions (unused)
            c.as_mut_ptr() as *const usize,       // r8 = C output
            0,                                    // r9 = batch_size (unused)
            0,                                    // stack = seq_len (unused)
            std::ptr::null_mut(),                 // stack = output (unused)
            scratch_ptr,                          // stack = scratchpad
        );
    }

    /// Scalar reference matmul: C[i,j] = sum_p A[i,p] * B[p,j]
    fn ref_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Fill a matrix with deterministic pseudo-random values in [-1, 1].
    fn fill_matrix(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
        let len = rows * cols;
        let mut v = Vec::with_capacity(len);
        let mut s = seed;
        for _ in 0..len {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((s as f32) / (u32::MAX as f32) * 2.0 - 1.0);
        }
        v
    }

    /// Assert two matrices are element-wise close.
    fn assert_matrix_close(
        got: &[f32], expected: &[f32], m: usize, n: usize, tol: f32, label: &str,
    ) {
        assert_eq!(got.len(), expected.len());
        let mut max_diff = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let diff = (got[idx] - expected[idx]).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < tol,
                    "{} mismatch at [{},{}]: got={}, expected={}, diff={}",
                    label, i, j, got[idx], expected[idx], diff,
                );
            }
        }
        eprintln!("{} e2e: {}×{} max_diff={:.2e} (tol={:.0e})", label, m, n, max_diff, tol);
    }

    #[test]
    fn test_e2e_gemm_single_tile() {
        let (m, n, k) = (4, 8, 16);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);
        eprintln!("scratchpad_bytes = {}", layer.scratchpad_bytes);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 137);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-5;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-single-tile");
    }

    #[test]
    fn test_e2e_gemm_full_tiles() {
        let (m, n, k) = (12, 16, 32);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 99);
        let b = fill_matrix(k, n, 200);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-5;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-full-tiles");
    }

    #[test]
    fn test_e2e_gemm_with_remainders() {
        let (m, n, k) = (10, 24, 20);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 7);
        let b = fill_matrix(k, n, 13);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-5;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-remainders");
    }

    #[test]
    fn test_e2e_gemm_larger() {
        let (m, n, k) = (25, 40, 64);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);
        eprintln!("scratchpad_bytes = {}", layer.scratchpad_bytes);

        let a = fill_matrix(m, k, 31415);
        let b = fill_matrix(k, n, 27182);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        for i in 0..8.min(m) {
            eprint!("row {}: ", i);
            for j in 0..8.min(n) {
                eprint!("{:.4} ", c_jit[i * n + j]);
            }
            eprintln!();
        }
        eprintln!("--- ref ---");
        for i in 0..8.min(m) {
            eprint!("row {}: ", i);
            for j in 0..8.min(n) {
                eprint!("{:.4} ", c_ref[i * n + j]);
            }
            eprintln!();
        }

        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-larger");
    }

    #[test]
    fn test_e2e_gemm_large_k() {
        let (m, n, k) = (12, 16, 1024);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 54321);
        let b = fill_matrix(k, n, 12345);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-large-k");
    }

    #[test]
    fn test_e2e_gemm_blis_diag() {
        for &(m, n, k) in &[
            (6usize, 16usize, 297usize),    // single tile, 2 KC blocks
            (12, 32, 297),                    // 2x2 tiles, 2 KC blocks
            (24, 48, 297),                    // multi-tile
            (48, 64, 512),
            (72, 256, 297),
            (128, 16, 297),
            (128, 256, 297),
            (128, 256, 512),
        ] {
            println!("DIAG: testing {}x{}x{}", m, n, k);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            let (graph, op_id) = build_gemm_graph(m, n, k);
            let plan = build_gemm_plan(op_id);
            let alloc = build_alloc(m * n * 4);
            let layer = compile(&graph, &plan, &alloc);

            let a = fill_matrix(m, k, 271828);
            let b = fill_matrix(k, n, 314159);
            let mut c_jit = vec![0.0f32; m * n];
            let mut c_ref = vec![0.0f32; m * n];

            ref_matmul(&a, &b, &mut c_ref, m, n, k);
            println!("DIAG: calling JIT...");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };
            println!("DIAG: JIT returned");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let tol = k as f32 * 1e-3;
            let mut mismatches = 0;
            let mut max_diff = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let idx = i * n + j;
                    let diff = (c_jit[idx] - c_ref[idx]).abs();
                    max_diff = max_diff.max(diff);
                    if diff >= tol && mismatches < 5 {
                        println!(
                            "  MISMATCH [{},{}]: jit={:.6}, ref={:.6}, diff={:.6}",
                            i, j, c_jit[idx], c_ref[idx], diff,
                        );
                        mismatches += 1;
                    }
                }
            }
            println!(
                "DIAG {}x{}x{}: max_diff={:.2e}, tol={:.2e}, mismatches={}",
                m, n, k, max_diff, tol, mismatches,
            );
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            assert_eq!(mismatches, 0, "GEMM-blis-diag {}x{}x{} failed", m, n, k);
        }
    }

    #[test]
    fn test_e2e_gemm_full_blis() {
        let (m, n, k) = (128, 256, 512);
        let (graph, op_id) = build_gemm_graph(m, n, k);
        let plan = build_gemm_plan(op_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 271828);
        let b = fill_matrix(k, n, 314159);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM-full-blis");
    }

    /// Build a graph with GEMM followed by a unary epilogue op.
    /// Returns (graph, gemm_op_id, epilogue_op_id).
    fn build_gemm_epilogue_graph(
        m: usize, n: usize, k: usize, epi_kind: OpKind,
    ) -> (CompilerGraph, OpId, OpId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let gemm_out = g.add_tensor("gemm_out", vec![m, n], DType::F32);
        let epi_out = g.add_tensor("epi_out", vec![m, n], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![epi_out];
        let gemm_id = g.add_op(
            OpKind::Gemm { m, n, k }, vec![a, b], vec![gemm_out], "gemm",
        );
        let epi_id = g.add_op(epi_kind, vec![gemm_out], vec![epi_out], "epilogue");
        (g, gemm_id, epi_id)
    }

    /// Build a FusionPlan with GemmEpilogue pattern.
    fn build_gemm_epilogue_plan(gemm_id: OpId, epi_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(gemm_id, 0);
        op_to_group.insert(epi_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_id,
                epilogue: vec![epi_id],
                mode: FusionMode::EpilogueInjection,
                ops: vec![gemm_id, epi_id],
            }],
            op_to_group,
        }
    }

    /// Compile with a ScalarOpRegistry (needed for epilogue trace lookup).
    fn compile_with_registry(
        graph: &CompilerGraph,
        plan: &FusionPlan,
        alloc: &BufferAllocation,
        registry: &ScalarOpRegistry,
    ) -> CompiledLayer {
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        let output = codegen
            .emit_plan(plan, graph, alloc, &profile, Some(registry))
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
    }

    /// Build a registry with a ReLU trace: max(0, x).
    fn registry_with_relu() -> ScalarOpRegistry {
        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(
            OpKindKey::Silu,
            OpTrace {
                op_kind: OpKind::Silu,
                pattern: ComputePattern::Elementwise {
                    body: vec![
                        TraceOp::Input(0),   // [0] x
                        TraceOp::Const(0.0),  // [1] 0.0
                        TraceOp::Max(0, 1),   // [2] max(x, 0)
                    ],
                },
                signature: ScalarFnSignature {
                    fn_ptr: std::ptr::null(),
                    params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr],
                },
            },
        );
        reg
    }

    /// Build a registry with the real SiLU trace: x / (1 + exp(-x)).
    fn registry_with_silu() -> ScalarOpRegistry {
        ScalarOpRegistry::with_defaults()
    }

    /// Scalar reference ReLU.
    fn apply_relu_inplace(v: &mut [f32]) {
        for x in v.iter_mut() {
            *x = x.max(0.0);
        }
    }

    /// Scalar reference SiLU: x / (1 + exp(-x)).
    fn apply_silu_inplace(v: &mut [f32]) {
        for x in v.iter_mut() {
            *x = *x / (1.0 + (-*x).exp());
        }
    }

    #[test]
    fn gemm_relu_epilogue_small() {
        let m = 4;
        let n = 8;
        let k = 4;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_relu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 99);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_relu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        assert_matrix_close(&c_jit, &c_ref, m, n, 1e-4, "GEMM+ReLU-small");
    }

    #[test]
    fn gemm_relu_epilogue_medium() {
        let m = 16;
        let n = 16;
        let k = 8;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_relu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 7);
        let b = fill_matrix(k, n, 13);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_relu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+ReLU-medium");
    }

    #[test]
    fn gemm_silu_epilogue_small() {
        let m = 4;
        let n = 8;
        let k = 4;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_silu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 99);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_silu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        assert_matrix_close(&c_jit, &c_ref, m, n, 5e-3, "GEMM+SiLU-small");
    }

    #[test]
    fn gemm_silu_epilogue_medium() {
        let m = 16;
        let n = 16;
        let k = 8;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_silu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 7);
        let b = fill_matrix(k, n, 13);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_silu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 5e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+SiLU-medium");
    }

    /// Large GEMM+SiLU that exercises the BLIS blocking path.
    #[test]
    fn gemm_silu_epilogue_blis() {
        let m = 64;
        let n = 64;
        let k = 64;
        let (graph, gemm_id, epi_id) = build_gemm_epilogue_graph(m, n, k, OpKind::Silu);
        let plan = build_gemm_epilogue_plan(gemm_id, epi_id);
        let alloc = build_alloc(0);
        let registry = registry_with_silu();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = fill_matrix(m, k, 31);
        let b = fill_matrix(k, n, 59);
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_matmul(&a, &b, &mut c_ref, m, n, k);
        apply_silu_inplace(&mut c_ref);

        unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

        let tol = k as f32 * 5e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GEMM+SiLU-blis");
    }

    #[test]
    fn test_e2e_gemm_k_odd() {
        for &(m, n, k) in &[
            (4usize, 8usize, 3usize),
            (4, 8, 5),
            (4, 8, 7),
            (6, 16, 3),
            (6, 16, 5),
            (6, 16, 9),
            (12, 16, 3),
            (12, 16, 7),
            (12, 16, 11),
        ] {
            let (graph, op_id) = build_gemm_graph(m, n, k);
            let plan = build_gemm_plan(op_id);
            let alloc = build_alloc(m * n * 4);
            let layer = compile(&graph, &plan, &alloc);

            let a = fill_matrix(m, k, 42 + k as u32);
            let b = fill_matrix(k, n, 137 + k as u32);
            let mut c_jit = vec![0.0f32; m * n];
            let mut c_ref = vec![0.0f32; m * n];

            ref_matmul(&a, &b, &mut c_ref, m, n, k);
            unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

            let tol = k as f32 * 1e-5;
            assert_matrix_close(&c_jit, &c_ref, m, n, tol,
                &format!("GEMM-k-odd-{}x{}x{}", m, n, k));
        }
    }

    #[test]
    fn test_e2e_gemm_blis_k_odd() {
        for &(m, n, k) in &[
            (6usize, 16usize, 297usize),
            (12, 32, 297),
            (24, 48, 297),
            (48, 64, 513),
        ] {
            let (graph, op_id) = build_gemm_graph(m, n, k);
            let plan = build_gemm_plan(op_id);
            let alloc = build_alloc(m * n * 4);
            let layer = compile(&graph, &plan, &alloc);

            let a = fill_matrix(m, k, 271828 + k as u32);
            let b = fill_matrix(k, n, 314159 + k as u32);
            let mut c_jit = vec![0.0f32; m * n];
            let mut c_ref = vec![0.0f32; m * n];

            ref_matmul(&a, &b, &mut c_ref, m, n, k);
            unsafe { exec_gemm(&layer, &a, &b, &mut c_jit) };

            let tol = k as f32 * 1e-3;
            assert_matrix_close(&c_jit, &c_ref, m, n, tol,
                &format!("GEMM-blis-k-odd-{}x{}x{}", m, n, k));
        }
    }

    /// Build a graph with RmsNorm → Gemm.
    fn build_norm_gemm_graph(
        m: usize, n: usize, k: usize, eps: f32,
    ) -> (CompilerGraph, OpId, OpId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let norm_w = g.add_tensor("norm_w", vec![k], DType::F32);
        let normed = g.add_tensor("normed", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let c = g.add_tensor("C", vec![m, n], DType::F32);
        g.inputs = vec![a, b, norm_w];
        g.outputs = vec![c];
        let norm_id = g.add_op(
            OpKind::RmsNorm { eps }, vec![a, norm_w], vec![normed], "rms_norm",
        );
        let gemm_id = g.add_op(
            OpKind::Gemm { m, n, k }, vec![normed, b], vec![c], "gemm",
        );
        (g, norm_id, gemm_id)
    }

    /// Build a FusionPlan with NormIntoGemm pattern.
    fn build_norm_gemm_plan(norm_id: OpId, gemm_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(norm_id, 0);
        op_to_group.insert(gemm_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_id,
                epilogue: vec![],
                mode: FusionMode::NormIntoGemm,
                ops: vec![norm_id, gemm_id],
            }],
            op_to_group,
        }
    }

    /// Execute a compiled NormIntoGemm kernel.
    ///
    /// ABI: rdi=A, rsi=B, rdx=norm_weight, r8=C, scratchpad on stack.
    unsafe fn exec_norm_gemm(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        norm_w: &[f32],
        c: &mut [f32],
    ) {
        let mut scratch = vec![0u8; layer.scratchpad_bytes];
        let scratch_ptr = if layer.scratchpad_bytes > 0 {
            scratch.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,              // rdi = A input
            b.as_ptr() as *const u8,              // rsi = B weight matrix
            norm_w.as_ptr() as *mut u8,           // rdx = norm weight ptr
            std::ptr::null(),                     // rcx = positions (unused)
            c.as_mut_ptr() as *const usize,       // r8 = C output
            0,                                    // r9 = batch_size (unused)
            0,                                    // stack = seq_len (unused)
            std::ptr::null_mut(),                 // stack = output (unused)
            scratch_ptr,                          // stack = scratchpad
        );
    }

    /// Reference: row-wise RmsNorm then matmul.
    fn ref_norm_gemm(
        a: &[f32], b: &[f32], norm_w: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize, eps: f32,
    ) {
        let mut normed = vec![0.0f32; m * k];
        for row in 0..m {
            let off = row * k;
            let mut sum_sq = 0.0f32;
            for j in 0..k {
                sum_sq += a[off + j] * a[off + j];
            }
            let scale = 1.0 / (sum_sq / k as f32 + eps).sqrt();
            for j in 0..k {
                normed[off + j] = a[off + j] * scale * norm_w[j];
            }
        }
        ref_matmul(&normed, b, c, m, n, k);
    }

    #[test]
    fn test_e2e_norm_into_gemm_small() {
        let (m, n, k) = (4, 8, 16);
        let eps = 1e-5;
        let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
        let plan = build_norm_gemm_plan(norm_id, gemm_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 42);
        let b = fill_matrix(k, n, 137);
        let norm_w: Vec<f32> = (0..k).map(|i| 0.8 + 0.4 * (i as f32) / k as f32).collect();
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
        unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

        let tol = k as f32 * 1e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormIntoGemm-small");
    }

    #[test]
    fn test_e2e_norm_into_gemm_medium() {
        let (m, n, k) = (32, 64, 128);
        let eps = 1e-5;
        let (graph, norm_id, gemm_id) = build_norm_gemm_graph(m, n, k, eps);
        let plan = build_norm_gemm_plan(norm_id, gemm_id);
        let alloc = build_alloc(m * n * 4);
        let layer = compile(&graph, &plan, &alloc);

        let a = fill_matrix(m, k, 271828);
        let b = fill_matrix(k, n, 314159);
        let norm_w: Vec<f32> = (0..k).map(|i| 0.5 + 1.0 * (i as f32) / k as f32).collect();
        let mut c_jit = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        ref_norm_gemm(&a, &b, &norm_w, &mut c_ref, m, n, k, eps);
        unsafe { exec_norm_gemm(&layer, &a, &b, &norm_w, &mut c_jit) };

        let tol = k as f32 * 1e-3;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "NormIntoGemm-medium");
    }

    /// Build a unary graph with FusionMode::Standalone plan (for reductions like Softmax).
    fn build_standalone_plan(op_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::Standalone,
                ops: vec![op_id],
            }],
            op_to_group,
        }
    }

    /// Build a registry with the Softmax reduction trace manually injected.
    /// We bypass `with_defaults()` because symexec may misclassify the
    /// multi-pass Softmax scalar fn as Elementwise.
    fn registry_with_softmax() -> ScalarOpRegistry {
        let mut reg = ScalarOpRegistry::new();
        reg.inject_trace(
            OpKindKey::Softmax,
            OpTrace {
                op_kind: OpKind::Softmax,
                pattern: ComputePattern::Reduction {
                    identity: f64::NEG_INFINITY,
                    combine: vec![
                        TraceOp::Input(0),  // [0] a (running max)
                        TraceOp::Input(1),  // [1] b (new element)
                        TraceOp::Max(0, 1), // [2] max(a, b)
                    ],
                    second_pass: Some(Box::new(ReductionSecondPass {
                        identity: 0.0,
                        element_transform: vec![
                            TraceOp::Input(0),  // [0] x (current element)
                            TraceOp::Input(1),  // [1] max (broadcast)
                            TraceOp::Sub(0, 1), // [2] x - max
                            TraceOp::Exp(2),    // [3] exp(x - max)
                        ],
                        combine: vec![
                            TraceOp::Input(0),  // [0] acc (running sum)
                            TraceOp::Input(1),  // [1] exp_val
                            TraceOp::Add(0, 1), // [2] acc + exp_val
                        ],
                    })),
                    normalize: Some(vec![
                        TraceOp::Input(0),  // [0] exp_val
                        TraceOp::Input(1),  // [1] inv_sum (broadcast)
                        TraceOp::Mul(0, 1), // [2] exp_val * inv_sum
                    ]),
                },
                signature: ScalarFnSignature {
                    fn_ptr: std::ptr::null(),
                    params: vec![ScalarParam::InputPtr, ScalarParam::OutputPtr, ScalarParam::Dim(0)],
                },
            },
        );
        reg
    }

    /// Execute a compiled standalone reduction (Softmax).
    ///
    /// ABI: rdi=input, r8=output, [stack]=scratchpad.
    unsafe fn exec_reduction(
        layer: &CompiledLayer,
        input: &[f32],
        output: &mut [f32],
    ) {
        let mut scratch = vec![0u8; layer.scratchpad_bytes.max(64)];
        let scratch_ptr = scratch.as_mut_ptr();
        let f = layer.entry_point();
        f(
            input.as_ptr() as *const u8,             // rdi = input
            std::ptr::null(),                         // rsi = weights (unused)
            std::ptr::null_mut(),                     // rdx = kv_cache (unused)
            std::ptr::null(),                         // rcx = positions (unused)
            output.as_mut_ptr() as *const usize,      // r8 = output ptr
            0,                                        // r9 = batch_size (unused)
            0,                                        // stack = seq_len (unused)
            std::ptr::null_mut(),                     // stack = output (unused)
            scratch_ptr,                              // stack = scratchpad
        );
    }

    /// Scalar reference softmax: exp(x_i - max) / sum(exp(x_j - max)).
    fn ref_softmax(input: &[f32]) -> Vec<f32> {
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum).collect()
    }

    #[test]
    fn standalone_softmax_small() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Softmax);
        let plan = build_standalone_plan(op_id);
        let alloc = build_alloc(0);
        let registry = registry_with_softmax();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        let reference = ref_softmax(&input);

        unsafe { exec_reduction(&layer, &input, &mut output) };

        eprintln!("Softmax input:     {:?}", &input[..8]);
        eprintln!("Softmax output:    {:?}", &output[..8]);
        eprintln!("Softmax reference: {:?}", &reference[..8]);

        for i in 0..N {
            let diff = (output[i] - reference[i]).abs();
            assert!(
                diff < 1e-3,
                "Softmax mismatch at [{}]: got={}, expected={}, diff={}",
                i, output[i], reference[i], diff,
            );
        }
        eprintln!("Softmax e2e: all {} elements within 1e-3 tolerance", N);
    }

    /// Test GemmBias in EpilogueInjection mode with WeightLayout.
    ///
    /// Single group: GemmBias { m:4, n:8, k:4 } reads activation from rdi,
    /// B matrix and bias from weight blob (rsi), writes to graph output (r8).
    /// Uses small dimensions so the DIRECT microkernel path is taken.
    #[test]
    fn test_e2e_gemmbias_epilogue_with_weight_layout() {
        let (m, n, k) = (4, 8, 4);

        // Build graph: A is activation input, B and bias are weight tensors.
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let bias = g.add_tensor("bias", vec![n], DType::F32);
        let c = g.add_tensor("C", vec![m, n], DType::F32);
        g.inputs = vec![a];
        g.outputs = vec![c];
        let gemm_id = g.add_op(
            OpKind::GemmBias { m, n, k },
            vec![a, b, bias],
            vec![c],
            "gemmbias",
        );

        // Weight layout: B at offset 0, bias after B.
        let b_bytes = k * n * 4;
        let bias_bytes = n * 4;
        let weight_layout = WeightLayout {
            offsets: vec![(b, 0), (bias, b_bytes)],
            total_bytes: b_bytes + bias_bytes,
        };

        // FusionPlan: single group, EpilogueInjection mode.
        let mut op_to_group = HashMap::new();
        op_to_group.insert(gemm_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_id,
                epilogue: vec![],
                mode: FusionMode::EpilogueInjection,
                ops: vec![gemm_id],
            }],
            op_to_group,
        };

        let alloc = build_alloc(0);

        // Compile with weight layout.
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        codegen.set_weight_layout(weight_layout);
        let output = codegen
            .emit_plan(&plan, &g, &alloc, &profile, None)
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        let layer = CompiledLayer::from_code(
            &output.code, output.scratchpad_bytes, 0,
        ).unwrap();

        // Prepare data.
        let a_data = fill_matrix(m, k, 42);
        let b_data = fill_matrix(k, n, 99);
        let bias_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();

        // Pack weight blob: [B | bias].
        let mut weight_blob = vec![0u8; b_bytes + bias_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr(),
                b_bytes,
            );
            std::ptr::copy_nonoverlapping(
                bias_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(b_bytes),
                bias_bytes,
            );
        }

        // Execute.
        // With weight_layout, GraphOutput reads from [rbp+24] = arg7 (output).
        // ABI: rdi=input, rsi=weights, rdx=kv_cache, rcx=positions,
        //      r8=seq_lens, r9=batch_size, [rbp+16]=seq_len,
        //      [rbp+24]=output, [rbp+32]=scratchpad.
        let mut c_jit = vec![0.0f32; m * n];
        let mut scratch = vec![0u8; layer.scratchpad_bytes.max(64)];
        unsafe {
            let f = layer.entry_point();
            f(
                a_data.as_ptr() as *const u8,             // rdi = activation
                weight_blob.as_ptr() as *const u8,        // rsi = weight blob
                std::ptr::null_mut(),                      // rdx = kv_cache (unused)
                std::ptr::null(),                          // rcx = positions (unused)
                std::ptr::null(),                          // r8 = seq_lens (unused)
                0,                                         // r9 = batch_size (unused)
                0,                                         // stack[0] = seq_len (unused)
                c_jit.as_mut_ptr() as *mut u8,            // stack[1] = output ptr [rbp+24]
                scratch.as_mut_ptr(),                      // stack[2] = scratchpad [rbp+32]
            );
        }

        // Reference: matmul + bias (GemmBias = GEMM + bias add).
        let mut c_ref = vec![0.0f32; m * n];
        ref_matmul(&a_data, &b_data, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias_data[j];
            }
        }

        // Verify no NaN and matches reference.
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                assert!(
                    !c_jit[idx].is_nan(),
                    "GemmBias output NaN at [{},{}]", i, j,
                );
            }
        }
        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GemmBias-epilogue-weight-layout");
    }

    /// Same as above but with m=6, n=384, k=384 to force the BLIS tiling path (n > nr).
    #[test]
    fn test_e2e_gemmbias_blis_path() {
        let (m, n, k) = (6, 384, 384);

        // Build graph: A is activation input, B and bias are weight tensors.
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("A", vec![m, k], DType::F32);
        let b = g.add_tensor("B", vec![k, n], DType::F32);
        let bias = g.add_tensor("bias", vec![n], DType::F32);
        let c = g.add_tensor("C", vec![m, n], DType::F32);
        g.inputs = vec![a];
        g.outputs = vec![c];
        let gemm_id = g.add_op(
            OpKind::GemmBias { m, n, k },
            vec![a, b, bias],
            vec![c],
            "gemmbias",
        );

        // Weight layout: B at offset 0, bias after B.
        let b_bytes = k * n * 4; // 384*384*4 = 589824
        let bias_bytes = n * 4;  // 384*4 = 1536
        let weight_layout = WeightLayout {
            offsets: vec![(b, 0), (bias, b_bytes)],
            total_bytes: b_bytes + bias_bytes,
        };

        // FusionPlan: single group, EpilogueInjection mode.
        let mut op_to_group = HashMap::new();
        op_to_group.insert(gemm_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: gemm_id,
                epilogue: vec![],
                mode: FusionMode::EpilogueInjection,
                ops: vec![gemm_id],
            }],
            op_to_group,
        };

        let alloc = build_alloc(0);

        // Compile with weight layout.
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        codegen.set_weight_layout(weight_layout);
        let output = codegen
            .emit_plan(&plan, &g, &alloc, &profile, None)
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        let layer = CompiledLayer::from_code(
            &output.code, output.scratchpad_bytes, 0,
        ).unwrap();

        // Prepare data with random-ish values.
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.007).cos()).collect();
        let bias_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();

        // Pack weight blob: [B | bias].
        let mut weight_blob = vec![0u8; b_bytes + bias_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr(),
                b_bytes,
            );
            std::ptr::copy_nonoverlapping(
                bias_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(b_bytes),
                bias_bytes,
            );
        }

        // Execute.
        let mut c_jit = vec![0.0f32; m * n];
        let mut scratch = vec![0u8; layer.scratchpad_bytes.max(64)];
        unsafe {
            let f = layer.entry_point();
            f(
                a_data.as_ptr() as *const u8,             // rdi = activation
                weight_blob.as_ptr() as *const u8,        // rsi = weight blob
                std::ptr::null_mut(),                      // rdx = kv_cache (unused)
                std::ptr::null(),                          // rcx = positions (unused)
                std::ptr::null(),                          // r8 = seq_lens (unused)
                0,                                         // r9 = batch_size (unused)
                0,                                         // stack[0] = seq_len (unused)
                c_jit.as_mut_ptr() as *mut u8,            // stack[1] = output ptr [rbp+24]
                scratch.as_mut_ptr(),                      // stack[2] = scratchpad [rbp+32]
            );
        }

        // Reference: matmul + bias (GemmBias = GEMM + bias add).
        let mut c_ref = vec![0.0f32; m * n];
        ref_matmul(&a_data, &b_data, &mut c_ref, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c_ref[i * n + j] += bias_data[j];
            }
        }

        // Verify no NaN and matches reference.
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                assert!(
                    !c_jit[idx].is_nan(),
                    "GemmBias BLIS output NaN at [{},{}]", i, j,
                );
            }
        }
        let tol = k as f32 * 1e-4;
        assert_matrix_close(&c_jit, &c_ref, m, n, tol, "GemmBias-blis-path");
    }

    /// Two-group pipeline: Gemm (group 0, Standalone) -> GemmBias (group 1, EpilogueInjection).
    /// Both groups should use the BLIS path (large k=384).
    /// Intermediate tensor t2 lives in scratchpad.
    #[test]
    fn test_e2e_multi_group_gemmbias_blis() {
        let (m, k1, n1) = (6, 384, 384);
        let (k2, n2) = (n1, 384); // k2 = n1

        // Build graph.
        let mut g = CompilerGraph::new();
        let t0 = g.add_tensor("A", vec![m, k1], DType::F32);       // activation input
        let t1 = g.add_tensor("W1", vec![k1, n1], DType::F32);     // weight W1
        let t2 = g.add_tensor("mid", vec![m, n1], DType::F32);     // intermediate (scratchpad)
        let t3 = g.add_tensor("W2", vec![k2, n2], DType::F32);     // weight W2
        let t4 = g.add_tensor("b2", vec![n2], DType::F32);         // bias b2
        let t5 = g.add_tensor("out", vec![m, n2], DType::F32);     // output
        g.inputs = vec![t0];
        g.outputs = vec![t5];
        let op0 = g.add_op(OpKind::Gemm { m, n: n1, k: k1 }, vec![t0, t1], vec![t2], "gemm0");
        let op1 = g.add_op(OpKind::GemmBias { m, n: n2, k: k2 }, vec![t2, t3, t4], vec![t5], "gemmbias1");

        // Weight layout: W1 at 0, W2 after W1, b2 after W2.
        let w1_bytes = k1 * n1 * 4; // 384*384*4 = 589824
        let w2_bytes = k2 * n2 * 4; // 384*384*4 = 589824
        let b2_bytes = n2 * 4;      // 384*4 = 1536
        let weight_layout = WeightLayout {
            offsets: vec![(t1, 0), (t3, w1_bytes), (t4, w1_bytes + w2_bytes)],
            total_bytes: w1_bytes + w2_bytes + b2_bytes,
        };

        // Buffer allocation: t2 (intermediate) in scratchpad.
        let mid_bytes = m * n1 * 4; // 6*384*4 = 9216
        let alloc = BufferAllocation {
            slots: vec![BufferSlot {
                tensor_id: t2,
                offset: 0,
                size_bytes: mid_bytes,
            }],
            total_bytes: mid_bytes,
            num_tensors: 1,
            bytes_saved: 0,
        };

        // FusionPlan: 2 groups.
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op0, 0);
        op_to_group.insert(op1, 1);
        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0,
                    anchor: op0,
                    epilogue: vec![],
                    mode: FusionMode::Standalone,
                    ops: vec![op0],
                },
                FusionGroup {
                    id: 1,
                    anchor: op1,
                    epilogue: vec![],
                    mode: FusionMode::EpilogueInjection,
                    ops: vec![op1],
                },
            ],
            op_to_group,
        };

        // Compile.
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        codegen.set_weight_layout(weight_layout);
        let output = codegen
            .emit_plan(&plan, &g, &alloc, &profile, None)
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        let layer = CompiledLayer::from_code(
            &output.code, output.scratchpad_bytes, 0,
        ).unwrap();

        // Prepare data.
        let a_data: Vec<f32> = (0..m * k1).map(|i| (i as f32 * 0.01).sin()).collect();
        let w1_data: Vec<f32> = (0..k1 * n1).map(|i| (i as f32 * 0.007).cos()).collect();
        let w2_data: Vec<f32> = (0..k2 * n2).map(|i| (i as f32 * 0.005).sin()).collect();
        let b2_data: Vec<f32> = (0..n2).map(|i| 0.1 * i as f32).collect();

        // Pack weight blob: [W1 | W2 | b2].
        let total_weight_bytes = w1_bytes + w2_bytes + b2_bytes;
        let mut weight_blob = vec![0u8; total_weight_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                w1_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr(),
                w1_bytes,
            );
            std::ptr::copy_nonoverlapping(
                w2_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(w1_bytes),
                w2_bytes,
            );
            std::ptr::copy_nonoverlapping(
                b2_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(w1_bytes + w2_bytes),
                b2_bytes,
            );
        }

        // Execute.
        let mut out_jit = vec![0.0f32; m * n2];
        let mut scratch = vec![0u8; layer.scratchpad_bytes.max(64)];
        unsafe {
            let f = layer.entry_point();
            f(
                a_data.as_ptr() as *const u8,             // rdi = activation
                weight_blob.as_ptr() as *const u8,        // rsi = weight blob
                std::ptr::null_mut(),                      // rdx = kv_cache (unused)
                std::ptr::null(),                          // rcx = positions (unused)
                std::ptr::null(),                          // r8 = seq_lens (unused)
                0,                                         // r9 = batch_size (unused)
                0,                                         // stack[0] = seq_len (unused)
                out_jit.as_mut_ptr() as *mut u8,          // stack[1] = output ptr [rbp+24]
                scratch.as_mut_ptr(),                      // stack[2] = scratchpad [rbp+32]
            );
        }

        // Reference: two-stage matmul. Group 0 is plain Gemm, group 1 is GemmBias.
        let mut mid_ref = vec![0.0f32; m * n1];
        ref_matmul(&a_data, &w1_data, &mut mid_ref, m, n1, k1);
        let mut out_ref = vec![0.0f32; m * n2];
        ref_matmul(&mid_ref, &w2_data, &mut out_ref, m, n2, k2);
        for i in 0..m {
            for j in 0..n2 {
                out_ref[i * n2 + j] += b2_data[j];
            }
        }

        // Verify no NaN and matches reference.
        for i in 0..m {
            for j in 0..n2 {
                let idx = i * n2 + j;
                assert!(
                    !out_jit[idx].is_nan(),
                    "multi-group BLIS output NaN at [{},{}]", i, j,
                );
            }
        }
        let tol = (k1 + k2) as f32 * 1e-4;
        assert_matrix_close(&out_jit, &out_ref, m, n2, tol, "multi-group-gemmbias-blis");
    }

    /// 3-group pipeline: BLIS GEMM -> DIRECT GEMM -> BLIS GemmBias.
    /// Group 0 (Standalone Gemm, k=384 -> BLIS): [6,384] @ [384,32] -> Scratchpad(0) [6,32]
    /// Group 1 (Standalone Gemm, k=32  -> DIRECT): Scratchpad(0) [6,32] @ [32,384] -> Scratchpad(768) [6,384]
    /// Group 2 (EpilogueInjection GemmBias, k=384 -> BLIS): Scratchpad(768) [6,384] @ [384,384] + bias[384] -> output [6,384]
    #[test]
    fn test_e2e_three_group_direct_between_blis() {
        let m = 6;
        let (k1, n1) = (384, 32);
        let (k2, n2) = (32, 384);   // k2 = n1
        let (k3, n3) = (384, 384);  // k3 = n2

        // Build graph.
        let mut g = CompilerGraph::new();
        let t_a  = g.add_tensor("A",   vec![m, k1],  DType::F32);  // activation input [6,384]
        let t_w1 = g.add_tensor("W1",  vec![k1, n1], DType::F32);  // weight W1 [384,32]
        let t_mid1 = g.add_tensor("mid1", vec![m, n1], DType::F32); // scratchpad(0) [6,32]
        let t_w2 = g.add_tensor("W2",  vec![k2, n2], DType::F32);  // weight W2 [32,384]
        let t_mid2 = g.add_tensor("mid2", vec![m, n2], DType::F32); // scratchpad(768) [6,384]
        let t_w3 = g.add_tensor("W3",  vec![k3, n3], DType::F32);  // weight W3 [384,384]
        let t_b3 = g.add_tensor("b3",  vec![n3],     DType::F32);  // bias b3 [384]
        let t_out = g.add_tensor("out", vec![m, n3],  DType::F32);  // output [6,384]
        g.inputs = vec![t_a];
        g.outputs = vec![t_out];
        let op0 = g.add_op(OpKind::Gemm { m, n: n1, k: k1 }, vec![t_a, t_w1], vec![t_mid1], "gemm0");
        let op1 = g.add_op(OpKind::Gemm { m, n: n2, k: k2 }, vec![t_mid1, t_w2], vec![t_mid2], "gemm1");
        let op2 = g.add_op(OpKind::GemmBias { m, n: n3, k: k3 }, vec![t_mid2, t_w3, t_b3], vec![t_out], "gemmbias2");

        // Weight layout: W1 at 0, W2 after W1, W3 after W2, b3 after W3.
        let w1_bytes = k1 * n1 * 4; // 384*32*4 = 49152
        let w2_bytes = k2 * n2 * 4; // 32*384*4 = 49152
        let w3_bytes = k3 * n3 * 4; // 384*384*4 = 589824
        let b3_bytes = n3 * 4;      // 384*4 = 1536
        let weight_layout = WeightLayout {
            offsets: vec![
                (t_w1, 0),
                (t_w2, w1_bytes),
                (t_w3, w1_bytes + w2_bytes),
                (t_b3, w1_bytes + w2_bytes + w3_bytes),
            ],
            total_bytes: w1_bytes + w2_bytes + w3_bytes + b3_bytes, // 689664
        };

        // Buffer allocation: mid1 and mid2 in scratchpad.
        let mid1_bytes = m * n1 * 4; // 6*32*4 = 768
        let mid2_bytes = m * n2 * 4; // 6*384*4 = 9216
        let alloc = BufferAllocation {
            slots: vec![
                BufferSlot {
                    tensor_id: t_mid1,
                    offset: 0,
                    size_bytes: mid1_bytes,
                },
                BufferSlot {
                    tensor_id: t_mid2,
                    offset: mid1_bytes, // 768
                    size_bytes: mid2_bytes,
                },
            ],
            total_bytes: mid1_bytes + mid2_bytes, // 9984
            num_tensors: 2,
            bytes_saved: 0,
        };

        // FusionPlan: 3 groups.
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op0, 0);
        op_to_group.insert(op1, 1);
        op_to_group.insert(op2, 2);
        let plan = FusionPlan {
            groups: vec![
                FusionGroup {
                    id: 0,
                    anchor: op0,
                    epilogue: vec![],
                    mode: FusionMode::Standalone,
                    ops: vec![op0],
                },
                FusionGroup {
                    id: 1,
                    anchor: op1,
                    epilogue: vec![],
                    mode: FusionMode::Standalone,
                    ops: vec![op1],
                },
                FusionGroup {
                    id: 2,
                    anchor: op2,
                    epilogue: vec![],
                    mode: FusionMode::EpilogueInjection,
                    ops: vec![op2],
                },
            ],
            op_to_group,
        };

        // Compile.
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        codegen.set_weight_layout(weight_layout);
        let output = codegen
            .emit_plan(&plan, &g, &alloc, &profile, None)
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        let layer = CompiledLayer::from_code(
            &output.code, output.scratchpad_bytes, 0,
        ).unwrap();

        // Prepare data.
        let a_data: Vec<f32>  = (0..m * k1).map(|i| (i as f32 * 0.01).sin()).collect();
        let w1_data: Vec<f32> = (0..k1 * n1).map(|i| (i as f32 * 0.007).cos()).collect();
        let w2_data: Vec<f32> = (0..k2 * n2).map(|i| (i as f32 * 0.005).sin()).collect();
        let w3_data: Vec<f32> = (0..k3 * n3).map(|i| (i as f32 * 0.003).cos()).collect();
        let b3_data: Vec<f32> = (0..n3).map(|i| 0.1 * i as f32).collect();

        // Pack weight blob: [W1 | W2 | W3 | b3].
        let total_weight_bytes = w1_bytes + w2_bytes + w3_bytes + b3_bytes;
        let mut weight_blob = vec![0u8; total_weight_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                w1_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr(),
                w1_bytes,
            );
            std::ptr::copy_nonoverlapping(
                w2_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(w1_bytes),
                w2_bytes,
            );
            std::ptr::copy_nonoverlapping(
                w3_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(w1_bytes + w2_bytes),
                w3_bytes,
            );
            std::ptr::copy_nonoverlapping(
                b3_data.as_ptr() as *const u8,
                weight_blob.as_mut_ptr().add(w1_bytes + w2_bytes + w3_bytes),
                b3_bytes,
            );
        }

        // Execute.
        let mut out_jit = vec![0.0f32; m * n3];
        let mut scratch = vec![0u8; layer.scratchpad_bytes.max(64)];
        unsafe {
            let f = layer.entry_point();
            f(
                a_data.as_ptr() as *const u8,             // rdi = activation
                weight_blob.as_ptr() as *const u8,        // rsi = weight blob
                std::ptr::null_mut(),                      // rdx = kv_cache (unused)
                std::ptr::null(),                          // rcx = positions (unused)
                std::ptr::null(),                          // r8 = seq_lens (unused)
                0,                                         // r9 = batch_size (unused)
                0,                                         // stack[0] = seq_len (unused)
                out_jit.as_mut_ptr() as *mut u8,          // stack[1] = output ptr [rbp+24]
                scratch.as_mut_ptr(),                      // stack[2] = scratchpad [rbp+32]
            );
        }

        // Reference: three-stage matmul  out = ((A * W1) * W2) * W3 + b3
        // Groups 0 and 1 are plain Gemm, group 2 is GemmBias.
        let mut mid1_ref = vec![0.0f32; m * n1];
        ref_matmul(&a_data, &w1_data, &mut mid1_ref, m, n1, k1);
        let mut mid2_ref = vec![0.0f32; m * n2];
        ref_matmul(&mid1_ref, &w2_data, &mut mid2_ref, m, n2, k2);
        let mut out_ref = vec![0.0f32; m * n3];
        ref_matmul(&mid2_ref, &w3_data, &mut out_ref, m, n3, k3);
        for i in 0..m {
            for j in 0..n3 {
                out_ref[i * n3 + j] += b3_data[j];
            }
        }

        // Verify no NaN and matches reference.
        for i in 0..m {
            for j in 0..n3 {
                let idx = i * n3 + j;
                assert!(
                    !out_jit[idx].is_nan(),
                    "3-group BLIS/DIRECT/BLIS output NaN at [{},{}]", i, j,
                );
            }
        }
        let tol = (k1 + k2 + k3) as f32 * 1e-4;
        assert_matrix_close(&out_jit, &out_ref, m, n3, tol, "three-group-direct-between-blis");
    }

}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "jit-x86")]
mod registry_elementwise_tests {
    use crate::compiler::codegen::x86_64::jit::X86CodeGen;
    use crate::compiler::executable::CompiledLayer;
    use crate::compiler::graph::{CompilerGraph, OpKind, OpId, TensorId, WeightLayout};
    use crate::compiler::fusion::{FusionGroup, FusionPlan, FusionMode};
    use crate::compiler::buffer_alloc::{BufferAllocation, BufferSlot};
    use crate::dispatch::DeviceProfile;
    use crate::inference::types::DType;
    use crate::compiler::registry::ScalarOpRegistry;
    use std::collections::HashMap;

    const N: usize = 32;

    fn build_unary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![input], vec![output], "op");
        (g, op_id, output)
    }

    fn build_binary_graph(kind: OpKind) -> (CompilerGraph, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let a = g.add_tensor("a", vec![N], DType::F32);
        let b = g.add_tensor("b", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![a, b];
        g.outputs = vec![output];
        let op_id = g.add_op(kind, vec![a, b], vec![output], "op");
        (g, op_id, output)
    }

    /// Build a graph with two chained ops: unary anchor → binary epilogue.
    /// E.g. SiLU(x) + bias.
    fn build_chain_graph(
        anchor_kind: OpKind,
        epilogue_kind: OpKind,
    ) -> (CompilerGraph, OpId, OpId, TensorId) {
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![N], DType::F32);
        let bias = g.add_tensor("bias", vec![N], DType::F32);
        let mid = g.add_tensor("mid", vec![N], DType::F32);
        let output = g.add_tensor("output", vec![N], DType::F32);
        g.inputs = vec![input, bias];
        g.outputs = vec![output];
        let anchor_id = g.add_op(anchor_kind, vec![input], vec![mid], "anchor");
        let epi_id = g.add_op(epilogue_kind, vec![mid, bias], vec![output], "epilogue");
        (g, anchor_id, epi_id, output)
    }

    fn build_plan(op_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        }
    }

    fn build_chain_plan(anchor_id: OpId, epi_id: OpId) -> FusionPlan {
        let mut op_to_group = HashMap::new();
        op_to_group.insert(anchor_id, 0);
        op_to_group.insert(epi_id, 0);
        FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: anchor_id,
                epilogue: vec![epi_id],
                mode: FusionMode::LoopFusion,
                ops: vec![anchor_id, epi_id],
            }],
            op_to_group,
        }
    }

    fn build_alloc(total_bytes: usize) -> BufferAllocation {
        BufferAllocation {
            slots: vec![],
            total_bytes,
            num_tensors: 0,
            bytes_saved: 0,
        }
    }

    fn compile_with_registry(
        graph: &CompilerGraph,
        plan: &FusionPlan,
        alloc: &BufferAllocation,
        registry: &ScalarOpRegistry,
    ) -> CompiledLayer {
        let profile = DeviceProfile::detect();
        let mut codegen = X86CodeGen::new(&profile);
        let output = codegen
            .emit_plan(plan, graph, alloc, &profile, Some(registry))
            .unwrap();
        assert!(!output.code.is_empty(), "codegen produced empty code");
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap()
    }

    unsafe fn exec_unary(layer: &CompiledLayer, input: &[f32], output: &mut [f32]) {
        let f = layer.entry_point();
        f(
            input.as_ptr() as *const u8,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            output.as_mut_ptr() as *const usize,
            0, 0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }

    unsafe fn exec_binary(
        layer: &CompiledLayer,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) {
        let f = layer.entry_point();
        f(
            a.as_ptr() as *const u8,
            b.as_ptr() as *const u8,
            std::ptr::null_mut(),
            std::ptr::null(),
            output.as_mut_ptr() as *const usize,
            0, 0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }

    fn test_input() -> Vec<f32> {
        let base = [
            -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        ];
        (0..N).map(|i| base[i % base.len()]).collect()
    }

    fn test_input_b() -> Vec<f32> {
        (0..N).map(|i| 0.1 * (i as f32) + 0.5).collect()
    }

    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn ref_gelu(x: f32) -> f32 {
        let inner = (2.0_f32 / std::f32::consts::PI).sqrt()
            * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn test_registry_silu() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Silu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "Registry SiLU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_gelu() {
        let (graph, op_id, _) = build_unary_graph(OpKind::Gelu);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let mut output = vec![0.0f32; N];

        unsafe { exec_unary(&layer, &input, &mut output) };

        for i in 0..N {
            let expected = ref_gelu(input[i]);
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "Registry GELU mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, input[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry GELU: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_add() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Add);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] + b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Registry Add mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry Add: all {} elements within 1e-5", N);
    }

    #[test]
    fn test_registry_mul() {
        let (graph, op_id, _) = build_binary_graph(OpKind::Mul);
        let plan = build_plan(op_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let a = test_input();
        let b = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &a, &b, &mut output) };

        for i in 0..N {
            let expected = a[i] * b[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Registry Mul mismatch at [{}]: a={}, b={}, got={}, expected={}, diff={}",
                i, a[i], b[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry Mul: all {} elements within 1e-5", N);
    }

    #[test]
    fn test_registry_silu_add_chain() {
        let (graph, anchor_id, epi_id, _) =
            build_chain_graph(OpKind::Silu, OpKind::Add);
        let plan = build_chain_plan(anchor_id, epi_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let bias = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &input, &bias, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]) + bias[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU+Add chain mismatch at [{}]: input={}, bias={}, got={}, expected={}, diff={}",
                i, input[i], bias[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU+Add chain: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_silu_mul_chain() {
        let (graph, anchor_id, epi_id, _) =
            build_chain_graph(OpKind::Silu, OpKind::Mul);
        let plan = build_chain_plan(anchor_id, epi_id);
        let alloc = build_alloc(N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&graph, &plan, &alloc, &registry);

        let input = test_input();
        let scale = test_input_b();
        let mut output = vec![0.0f32; N];

        unsafe { exec_binary(&layer, &input, &scale, &mut output) };

        for i in 0..N {
            let expected = ref_silu(input[i]) * scale[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU+Mul chain mismatch at [{}]: input={}, scale={}, got={}, expected={}, diff={}",
                i, input[i], scale[i], output[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU+Mul chain: all {} elements within 1e-3", N);
    }

    #[test]
    fn test_registry_silu_with_tail() {
        const TAIL_N: usize = 19;
        let mut g = CompilerGraph::new();
        let input = g.add_tensor("input", vec![TAIL_N], DType::F32);
        let output = g.add_tensor("output", vec![TAIL_N], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(OpKind::Silu, vec![input], vec![output], "silu");

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
            }],
            op_to_group,
        };
        let alloc = build_alloc(TAIL_N * 4);
        let registry = ScalarOpRegistry::with_defaults();
        let layer = compile_with_registry(&g, &plan, &alloc, &registry);

        let base = [
            -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
            -0.3, 0.7, 2.2,
        ];
        let inp: Vec<f32> = base.to_vec();
        let mut out = vec![0.0f32; TAIL_N];

        unsafe { exec_unary(&layer, &inp, &mut out) };

        for i in 0..TAIL_N {
            let expected = ref_silu(inp[i]);
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-3,
                "SiLU tail mismatch at [{}]: input={}, got={}, expected={}, diff={}",
                i, inp[i], out[i], expected, diff,
            );
        }
        eprintln!("Registry SiLU with tail: all {} elements within 1e-3", TAIL_N);
    }
}

