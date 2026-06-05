//! Code generation — Register VM 统一代码生成 (REGISTER-VM SPEC)
//!
//! 全管线: TraceOp → lower → VmProgram → VmOptPass → RegAlloc → IsaLower → 物理代码
//!
//! 模块结构:
//! - `vm/` — Register VM 核心 (指令集/分配器/栈帧/优化/后端)
//! - `emitter.rs` — MachineCodeEmitter trait + X86CodeGen/X86Backend 实现
//! - `math_approx.rs` — 超越函数多项式系数 (供 VM Transcendental 使用)

pub mod emitter;
pub mod math_approx;
/// Register VM — 半虚拟机统一代码生成 (REGISTER-VM SPEC)
pub mod vm;

#[cfg(test)]
mod test_centroid;

pub use emitter::{MachineCodeEmitter, PlatformBackend, Platform};
#[cfg(feature = "jit-x86")]
pub use emitter::{X86CodeGen, X86Backend};

/// Code generation output 格式。
///
/// ARCH-CODEGEN-FORMAT: CPU Lower 产物是可直接执行的机器码；GPU Lower 产物是文本 IR，
/// 需要由设备 driver (nvrtc/HIPRTC/MSL compiler) 再编译为 cubin/hsaco/msllib。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeFormat {
    /// x86_64 / AArch64 机器码 — 可直接跳转执行
    MachineCode,
    /// NVIDIA PTX 文本 IR — 由 nvrtc/cuModuleLoadData 编译为 cubin
    Ptx,
    /// AMD HIP 源码文本 — 由 HIP runtime 编译
    Hip,
    /// Apple Metal Shading Language 文本
    Msl,
}

/// RoPE cos/sin cache requirement (ARCH-ROPE-CACHE).
///
/// JIT codegen 发现 RoPE 算子时,声明需要 caller 在 scratchpad 指定 offset
/// 处预填 cos/sin 表。布局: `[seq_len, head_dim]` row-major,
/// 每行前 `half` 个 f32 是 cos 值,后 `half` 个是 sin 值。
///
/// caller (gllm executor) 必须:
/// 1. 分配 scratchpad ≥ `cache_offset + max_seq_len * head_dim * sizeof::<f32>()`
/// 2. 根据 `positions` 数组,计算 `cos(p * theta^(-2i/head_dim))` 和 `sin(p * theta^(-2i/head_dim))`
///    填入 scratchpad,然后再调用 JIT kernel。
#[derive(Debug, Clone, PartialEq)]
pub struct RopeCacheRequirement {
    /// Scratchpad 内的 cos/sin 表起始 offset (字节)。
    pub cache_offset: usize,
    /// RoPE head_dim (= cos/sin 表的行内元素数)。所有 RoPE 算子共享该值。
    pub head_dim: usize,
    /// 旋转频率基数 (Llama 默认 10000.0)。
    pub theta: f64,
    /// 旋转比例 partial ∈ (0, 1]。只旋转前 `head_dim * partial` 维。
    pub partial: f32,
    /// 编译时分配的 seq_len 上界 (= SymDim::Symbolic.max_value)。
    pub max_seq_len: usize,
    /// RoPE 频率/温度缩放族 (None=标准, Yarn/Linear=长上下文扩展)。
    /// caller 必须用此参数调用 [`crate::compiler::rope_scaling::fill_cos_sin_table`]
    /// (或自行计算 `compute_inv_freq` + `compute_attention_scaling`),否则生成的
    /// cos/sin 表与编译期假设不一致,YaRN 长上下文行为退化为普通 RoPE。
    pub rope_scaling: Option<crate::compiler::graph::RopeScaling>,
    /// YaRN 注意力温度缩放 mscale = sqrt(0.1·ln(factor)+1.0)。
    /// caller 在 fill 时把 cos/sin 各乘以此值,使 Q·K^T 隐式 ×mscale²。
    /// None / Linear 缩放下恒等于 1.0;此字段冗余暴露便于断言与调试。
    pub attention_scaling: f32,
    /// Secondary RoPE cache for heterogeneous models with multiple head_dim values.
    /// Carries its own theta/partial/rope_scaling (e.g., Gemma-4 full layers use
    /// global_rope_theta=1M + partial=0.25, distinct from sliding theta=10K + partial=1.0).
    pub secondary_cache: Option<SecondaryRopeCache>,
}

/// Secondary RoPE cache parameters for heterogeneous models.
#[derive(Debug, Clone, PartialEq)]
pub struct SecondaryRopeCache {
    pub head_dim: usize,
    pub cache_offset: usize,
    pub theta: f64,
    pub partial: f32,
    pub rope_scaling: Option<crate::compiler::graph::RopeScaling>,
}

/// Per-Layer Embedding scratch requirement (Gemma 4 E2B/E4B).
///
/// PerLayerEmbed 算子内部拆成 2×GEMM + 2×Elementwise,中间张量
/// `ple_ctx` / `post_mlp_out` 必须驻留 scratchpad。JIT codegen 按本结构
/// 声明两个 buffer 在 scratchpad 内的 offset (字节,紧接在 RoPE cache /
/// BufferAllocation 之后)。
///
/// Shape 规则:
/// - `ple_ctx`    : `[max_seq, dim_per_layer]` f32
/// - `post_mlp`   : `[max_seq, hidden]` f32
///
/// Symbolic seq_len 场景按 `max_value` (= `SYMDIM_MAX_SEQ_LEN`) 预留上界。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PleScratchRequirement {
    /// `ple_ctx` 在 scratchpad 内的起始 offset (字节)。
    pub ctx_offset: usize,
    /// `post_mlp_out` 在 scratchpad 内的起始 offset (字节)。
    pub post_mlp_offset: usize,
    /// PLE 使用的总字节数 (ctx + post_mlp,对齐后)。
    pub total_bytes: usize,
    /// 编译时绑定的 seq_len 上界 (= SymDim::Symbolic.max_value)。
    pub max_seq_len: usize,
    /// `dim_per_layer` (PLE 投影维度)。
    pub dim_per_layer: usize,
    /// `hidden` (主路宽度)。
    pub hidden: usize,
}

/// DepthwiseConv1D scratch requirement (USM Conformer convolution module, T55).
///
/// DepthwiseConv1D 在 scratchpad 中预留 `[max_seq + left_pad + right_pad, channels]`
/// 的 padded input buffer,零初始化外圈 padding,中间 `[left_pad..left_pad+seq]`
/// 复制原始 input。之后的卷积循环直接从 padded buffer 按 `[t+k, c]` 读取,
/// 避免运行时 bound check。
///
/// Shape 规则:
/// - `padded_input` : `[max_seq + total_pad, channels]` f32
/// - `total_pad`    : causal 模式 = `kernel_size - 1` (全部在左侧)
///                   non-causal 模式 = `kernel_size - 1` (对称,前后各 `(K-1)/2`)
///
/// Symbolic seq_len 场景按 `max_value` 预留上界。多个 DWC op 共享同一 scratch 区
/// (签名一致时);不一致直接 Err。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DwcScratchRequirement {
    /// `padded_input` 在 scratchpad 内的起始 offset (字节,对齐后)。
    pub padded_offset: usize,
    /// DWC 使用的总字节数 (padded buffer 大小,含对齐)。
    pub total_bytes: usize,
    /// 编译时绑定的 seq_len 上界 (= SymDim::Symbolic.max_value 或 Concrete)。
    pub max_seq_len: usize,
    /// 通道数 (算子 `channels` 字段)。
    pub channels: usize,
    /// 卷积核尺寸 (算子 `kernel_size` 字段)。
    pub kernel_size: usize,
    /// Causal 模式: true → 左 pad `K-1`;false → 对称 SAME pad 前后各 `(K-1)/2`。
    pub causal: bool,
    /// 左侧 padding 元素数 (用于 lower 正确偏移 copy 起点)。
    pub left_pad: usize,
}

/// Output of code generation: machine code bytes or text IR (UTF-8).
pub struct CodegenOutput {
    /// 代码字节：MachineCode 时为机器码，Ptx/Hip/Msl 时为 UTF-8 文本 IR
    pub code: Vec<u8>,
    /// 代码格式（决定下游加载路径）
    pub format: CodeFormat,
    /// Required scratchpad size in bytes
    pub scratchpad_bytes: usize,
    /// Hot patch points for runtime modification (§9.2).
    /// Each entry: (offset_in_code, initial_target, alternatives)
    pub hotpatch_points: Vec<(usize, u64, Vec<u64>)>,
    /// RoPE cos/sin 表需求 (Some 当图中包含 RoPE 算子)。
    /// caller 必须在调用 kernel 前按该 layout 预填 scratchpad[cache_offset..]。
    pub rope_cache: Option<RopeCacheRequirement>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CodeFormat ──

    #[test]
    fn code_format_variants_equality() {
        assert_eq!(CodeFormat::MachineCode, CodeFormat::MachineCode);
        assert_eq!(CodeFormat::Ptx, CodeFormat::Ptx);
        assert_eq!(CodeFormat::Hip, CodeFormat::Hip);
        assert_eq!(CodeFormat::Msl, CodeFormat::Msl);
    }

    #[test]
    fn code_format_variants_inequality() {
        assert_ne!(CodeFormat::MachineCode, CodeFormat::Ptx);
        assert_ne!(CodeFormat::Ptx, CodeFormat::Hip);
        assert_ne!(CodeFormat::Hip, CodeFormat::Msl);
        assert_ne!(CodeFormat::MachineCode, CodeFormat::Msl);
    }

    // ── RopeCacheRequirement ──

    #[test]
    fn rope_cache_requirement_fields() {
        let r = RopeCacheRequirement {
            cache_offset: 1024,
            head_dim: 128,
            theta: 10000.0,
            partial: 1.0,
            max_seq_len: 4096,
            rope_scaling: None,
            attention_scaling: 1.0,
            secondary_cache: None,
        };
        assert_eq!(r.cache_offset, 1024);
        assert_eq!(r.head_dim, 128);
        assert_eq!(r.theta, 10000.0);
        assert_eq!(r.partial, 1.0);
        assert_eq!(r.max_seq_len, 4096);
        assert!(r.rope_scaling.is_none());
        assert_eq!(r.attention_scaling, 1.0);
        assert!(r.secondary_cache.is_none());
    }

    #[test]
    fn rope_cache_requirement_equality() {
        let a = RopeCacheRequirement {
            cache_offset: 0, head_dim: 64, theta: 500000.0,
            partial: 0.25, max_seq_len: 8192, rope_scaling: None,
            attention_scaling: 1.0, secondary_cache: None,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn rope_cache_requirement_with_secondary() {
        let sec = SecondaryRopeCache {
            head_dim: 64,
            cache_offset: 2048,
            theta: 1000000.0,
            partial: 0.25,
            rope_scaling: None,
        };
        let r = RopeCacheRequirement {
            cache_offset: 0, head_dim: 128, theta: 10000.0,
            partial: 1.0, max_seq_len: 8192, rope_scaling: None,
            attention_scaling: 1.0, secondary_cache: Some(sec.clone()),
        };
        assert_eq!(r.secondary_cache.as_ref().unwrap().head_dim, 64);
        assert_eq!(r.secondary_cache.as_ref().unwrap().theta, 1000000.0);
        let r2 = r.clone();
        assert_eq!(r, r2);
    }

    // ── SecondaryRopeCache ──

    #[test]
    fn secondary_rope_cache_fields() {
        let s = SecondaryRopeCache {
            head_dim: 96,
            cache_offset: 4096,
            theta: 500000.0,
            partial: 0.5,
            rope_scaling: Some(crate::compiler::graph::RopeScaling::Linear { factor: 4.0 }),
        };
        assert_eq!(s.head_dim, 96);
        assert_eq!(s.cache_offset, 4096);
        assert_eq!(s.theta, 500000.0);
        assert_eq!(s.partial, 0.5);
        assert!(s.rope_scaling.is_some());
    }

    #[test]
    fn secondary_rope_cache_equality() {
        let a = SecondaryRopeCache {
            head_dim: 64, cache_offset: 0, theta: 10000.0,
            partial: 1.0, rope_scaling: None,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── PleScratchRequirement ──

    #[test]
    fn ple_scratch_fields() {
        let p = PleScratchRequirement {
            ctx_offset: 1024,
            post_mlp_offset: 8192,
            total_bytes: 16384,
            max_seq_len: 4096,
            dim_per_layer: 256,
            hidden: 2048,
        };
        assert_eq!(p.ctx_offset, 1024);
        assert_eq!(p.post_mlp_offset, 8192);
        assert_eq!(p.total_bytes, 16384);
        assert_eq!(p.max_seq_len, 4096);
        assert_eq!(p.dim_per_layer, 256);
        assert_eq!(p.hidden, 2048);
    }

    #[test]
    fn ple_scratch_equality_and_copy() {
        let a = PleScratchRequirement {
            ctx_offset: 0, post_mlp_offset: 512, total_bytes: 1024,
            max_seq_len: 2048, dim_per_layer: 128, hidden: 1024,
        };
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn ple_scratch_inequality() {
        let a = PleScratchRequirement {
            ctx_offset: 0, post_mlp_offset: 512, total_bytes: 1024,
            max_seq_len: 2048, dim_per_layer: 128, hidden: 1024,
        };
        let mut b = a;
        b.total_bytes = 2048;
        assert_ne!(a, b);
    }

    // ── DwcScratchRequirement ──

    #[test]
    fn dwc_scratch_fields() {
        let d = DwcScratchRequirement {
            padded_offset: 2048,
            total_bytes: 4096,
            max_seq_len: 512,
            channels: 256,
            kernel_size: 7,
            causal: true,
            left_pad: 6,
        };
        assert_eq!(d.padded_offset, 2048);
        assert_eq!(d.total_bytes, 4096);
        assert_eq!(d.max_seq_len, 512);
        assert_eq!(d.channels, 256);
        assert_eq!(d.kernel_size, 7);
        assert!(d.causal);
        assert_eq!(d.left_pad, 6);
    }

    #[test]
    fn dwc_scratch_equality_and_copy() {
        let a = DwcScratchRequirement {
            padded_offset: 0, total_bytes: 1024, max_seq_len: 256,
            channels: 128, kernel_size: 3, causal: false, left_pad: 1,
        };
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn dwc_scratch_causal_vs_non_causal() {
        let causal = DwcScratchRequirement {
            padded_offset: 0, total_bytes: 1024, max_seq_len: 256,
            channels: 128, kernel_size: 5, causal: true, left_pad: 4,
        };
        let non_causal = DwcScratchRequirement {
            padded_offset: 0, total_bytes: 1024, max_seq_len: 256,
            channels: 128, kernel_size: 5, causal: false, left_pad: 2,
        };
        assert_ne!(causal, non_causal);
    }

    // ── CodegenOutput ──

    #[test]
    fn codegen_output_fields() {
        let out = CodegenOutput {
            code: vec![0x90, 0xC3],
            format: CodeFormat::MachineCode,
            scratchpad_bytes: 4096,
            hotpatch_points: vec![(16, 0xDEAD, vec![0xBEEF])],
            rope_cache: None,
        };
        assert_eq!(out.code, vec![0x90, 0xC3]);
        assert_eq!(out.format, CodeFormat::MachineCode);
        assert_eq!(out.scratchpad_bytes, 4096);
        assert_eq!(out.hotpatch_points.len(), 1);
        assert_eq!(out.hotpatch_points[0].0, 16);
        assert!(out.rope_cache.is_none());
    }

    #[test]
    fn codegen_output_ptx_format() {
        let out = CodegenOutput {
            code: b".version 8.0\n".to_vec(),
            format: CodeFormat::Ptx,
            scratchpad_bytes: 0,
            hotpatch_points: vec![],
            rope_cache: None,
        };
        assert_eq!(out.format, CodeFormat::Ptx);
        assert!(out.code.starts_with(b".version"));
        assert!(out.hotpatch_points.is_empty());
    }

    #[test]
    fn codegen_output_with_rope_cache() {
        let rope = RopeCacheRequirement {
            cache_offset: 256, head_dim: 128, theta: 10000.0,
            partial: 1.0, max_seq_len: 4096, rope_scaling: None,
            attention_scaling: 1.0, secondary_cache: None,
        };
        let out = CodegenOutput {
            code: vec![],
            format: CodeFormat::MachineCode,
            scratchpad_bytes: 65536,
            hotpatch_points: vec![],
            rope_cache: Some(rope),
        };
        let rc = out.rope_cache.unwrap();
        assert_eq!(rc.cache_offset, 256);
        assert_eq!(rc.head_dim, 128);
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn code_format_all_variants_are_distinct() {
        let variants = [
            CodeFormat::MachineCode,
            CodeFormat::Ptx,
            CodeFormat::Hip,
            CodeFormat::Msl,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "{:?} should != {:?}", variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn code_format_copy_preserves_value() {
        let a = CodeFormat::Hip;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn code_format_debug_output() {
        assert_eq!(format!("{:?}", CodeFormat::MachineCode), "MachineCode");
        assert_eq!(format!("{:?}", CodeFormat::Ptx), "Ptx");
        assert_eq!(format!("{:?}", CodeFormat::Hip), "Hip");
        assert_eq!(format!("{:?}", CodeFormat::Msl), "Msl");
    }

    #[test]
    fn rope_cache_with_yarn_scaling() {
        let r = RopeCacheRequirement {
            cache_offset: 0, head_dim: 64, theta: 10000.0,
            partial: 1.0, max_seq_len: 32768,
            rope_scaling: Some(crate::compiler::graph::RopeScaling::Yarn {
                factor: 4.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 8192,
            }),
            attention_scaling: 1.1892,
            secondary_cache: None,
        };
        assert!(r.rope_scaling.is_some());
        assert!((r.attention_scaling - 1.1892).abs() < 1e-4);
    }

    #[test]
    fn rope_cache_inequality_different_fields() {
        let a = RopeCacheRequirement {
            cache_offset: 0, head_dim: 64, theta: 10000.0,
            partial: 1.0, max_seq_len: 4096, rope_scaling: None,
            attention_scaling: 1.0, secondary_cache: None,
        };
        let mut b = a.clone();
        b.theta = 500000.0;
        assert_ne!(a, b);

        let mut c = a.clone();
        c.partial = 0.25;
        assert_ne!(a, c);
    }

    #[test]
    fn secondary_rope_cache_inequality() {
        let a = SecondaryRopeCache {
            head_dim: 64, cache_offset: 0, theta: 10000.0,
            partial: 1.0, rope_scaling: None,
        };
        let mut b = a.clone();
        b.partial = 0.25;
        assert_ne!(a, b);
    }

    #[test]
    fn ple_scratch_zero_offsets() {
        let p = PleScratchRequirement {
            ctx_offset: 0,
            post_mlp_offset: 0,
            total_bytes: 0,
            max_seq_len: 0,
            dim_per_layer: 0,
            hidden: 0,
        };
        assert_eq!(p.ctx_offset, 0);
        assert_eq!(p.total_bytes, 0);
    }

    #[test]
    fn dwc_scratch_non_causal_symmetric_pad() {
        let d = DwcScratchRequirement {
            padded_offset: 0, total_bytes: 2048, max_seq_len: 512,
            channels: 64, kernel_size: 5, causal: false, left_pad: 2,
        };
        assert!(!d.causal);
        assert_eq!(d.left_pad, 2);
        // Symmetric: total_pad = kernel_size - 1 = 4, left = (K-1)/2 = 2
    }

    #[test]
    fn codegen_output_hip_format_with_code() {
        let out = CodegenOutput {
            code: b"kernel void ...\0".to_vec(),
            format: CodeFormat::Hip,
            scratchpad_bytes: 0,
            hotpatch_points: vec![],
            rope_cache: None,
        };
        assert_eq!(out.format, CodeFormat::Hip);
        assert!(!out.code.is_empty());
        assert_eq!(out.scratchpad_bytes, 0);
    }

    #[test]
    fn codegen_output_msl_format() {
        let out = CodegenOutput {
            code: b"#include <metal>\n".to_vec(),
            format: CodeFormat::Msl,
            scratchpad_bytes: 1024,
            hotpatch_points: vec![(0, 100, vec![200, 300])],
            rope_cache: None,
        };
        assert_eq!(out.format, CodeFormat::Msl);
        assert_eq!(out.hotpatch_points.len(), 1);
        assert_eq!(out.hotpatch_points[0].2.len(), 2);
    }

    #[test]
    fn codegen_output_multiple_hotpatch_points() {
        let out = CodegenOutput {
            code: vec![],
            format: CodeFormat::MachineCode,
            scratchpad_bytes: 0,
            hotpatch_points: vec![
                (10, 0xAAAA, vec![0xBBBB]),
                (20, 0xCCCC, vec![0xDDDD, 0xEEEE]),
            ],
            rope_cache: None,
        };
        assert_eq!(out.hotpatch_points.len(), 2);
        assert_eq!(out.hotpatch_points[1].0, 20);
        assert_eq!(out.hotpatch_points[1].2.len(), 2);
    }

    #[test]
    fn dwc_scratch_inequality_different_channels() {
        let a = DwcScratchRequirement {
            padded_offset: 0, total_bytes: 1024, max_seq_len: 256,
            channels: 128, kernel_size: 3, causal: false, left_pad: 1,
        };
        let mut b = a;
        b.channels = 256;
        assert_ne!(a, b);
    }
}
