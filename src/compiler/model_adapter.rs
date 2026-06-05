//! ModelAdapter trait — 架构特定的配置和权重适配层。
//!
//! Layer 2 适配器负责：
//! 1. 将模型特定的配置转换为统一的 `ModelConfig`
//! 2. 适配权重布局以匹配 JIT 编译器的期望
//! 3. 提供架构特定的融合提示
//!
//! # 设计原则
//!
//! - **CPU/GPU 统一**: 同一个适配器适用于所有后端
//! - **编译时友好**: 配置适配在编译时完成，运行时零开销
//! - **架构隔离**: 每个架构族有独立的适配器实现

use crate::types::{ModelConfig, ModelArch, DType};
use crate::types::CompilerError;

/// 架构特定的融合提示。
///
/// 这些提示指导 JIT 编译器选择最优的融合策略。
#[derive(Debug, Clone, Default)]
pub struct FusionHints {
    /// 是否可以使用 FlashAttention（需要硬件支持）
    pub use_flash_attention: bool,
    /// 是否使用 GQA（Grouped Query Attention）
    pub use_gqa: bool,
    /// FFN 激活函数类型
    pub ffn_activation: FfnActivation,
    /// 是否需要滑动窗口注意力
    pub sliding_window: Option<usize>,
    /// RoPE 插值模式
    pub rope_mode: RopeMode,
}

/// FFN 激活函数类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum FfnActivation {
    /// SwiGLU (Llama, Qwen, Mistral)
    #[default]
    SwiGlu,
    /// GeGLU (Gemma)
    GeGlu,
    /// GELU (GPT-2, Phi)
    Gelu,
}


/// RoPE 模式。
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum RopeMode {
    /// 标准 RoPE (Llama)
    #[default]
    Standard,
    /// 部分旋转 (Phi)
    Partial { factor: f32 },
    /// 交错 RoPE (某些 GLM 变体)
    Interleaved,
}


/// 权重布局描述。
///
/// 描述权重张量在内存中的布局，以便 JIT 编译器生成正确的加载代码。
#[derive(Debug, Clone)]
pub struct AdapterWeightLayout {
    /// 权重名称 → 形状的映射
    pub shapes: Vec<(&'static str, Vec<usize>)>,
    /// 权重数据类型
    pub dtype: DType,
}

/// ModelAdapter trait — 架构特定的配置和权重适配。
///
/// 每个模型架构族（Llama, Qwen, Gemma 等）实现此 trait 以提供：
/// 1. 配置适配：从模型元数据生成 `ModelConfig`
/// 2. 融合提示：指导 JIT 编译器的融合策略
/// 3. 权重布局：描述权重的内存布局
pub trait ModelAdapter: Send + Sync + 'static {
    /// 适配器名称（用于调试和日志）
    fn name(&self) -> &'static str;

    /// 从架构元数据创建 `ModelConfig`。
    ///
    /// # 参数
    ///
    /// - `hidden_size`: 隐藏层维度
    /// - `num_layers`: 层数
    /// - `num_heads`: 注意力头数
    /// - `num_kv_heads`: KV 头数（GQA）
    /// - `head_dim`: 每个头的维度
    /// - `intermediate_size`: FFN 中间维度
    /// - `vocab_size`: 词汇表大小
    /// - `max_seq_len`: 最大序列长度
    /// - `dtype`: 计算数据类型
    ///
    /// # 返回
    ///
    /// 完整的 `ModelConfig`，包含架构特定的默认值（RoPE theta、norm eps 等）。
    fn adapt_config(
        &self,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> ModelConfig;

    /// 获取融合提示。
    ///
    /// 返回的提示基于架构特性指导 JIT 编译器：
    /// - 是否可以使用 FlashAttention
    /// - FFN 激活函数类型（影响融合策略）
    /// - RoPE 模式（影响 RoPE 融合）
    /// - 滑动窗口配置（影响注意力融合）
    fn fusion_hints(&self) -> FusionHints {
        FusionHints::default()
    }

    /// 获取权重布局描述。
    ///
    /// 返回权重张量的名称和形状，用于 JIT 编译器的权重加载代码生成。
    fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout;

    /// 验证配置是否与架构兼容。
    ///
    /// # 返回
    ///
    /// - `Ok(())`: 配置有效
    /// - `Err(String)`: 配置无效，包含错误描述
    fn validate_config(&self, config: &ModelConfig) -> Result<(), CompilerError> {
        // 基础验证：所有架构通用
        if config.hidden_size == 0 {
            return Err("hidden_size cannot be zero".to_string().into());
        }
        if config.num_layers == 0 {
            return Err("num_layers cannot be zero".to_string().into());
        }
        if config.num_heads == 0 {
            return Err("num_heads cannot be zero".to_string().into());
        }
        if config.num_kv_heads == 0 {
            return Err("num_kv_heads cannot be zero".to_string().into());
        }
        if config.num_kv_heads > config.num_heads {
            return Err(format!(
                "num_kv_heads ({}) cannot exceed num_heads ({})",
                config.num_kv_heads, config.num_heads
            ).into());
        }
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                config.num_heads, config.num_kv_heads
            ).into());
        }
        if config.head_dim == 0 {
            return Err("head_dim cannot be zero".to_string().into());
        }
        if config.vocab_size == 0 {
            return Err("vocab_size cannot be zero".to_string().into());
        }
        if config.max_seq_len == 0 {
            return Err("max_seq_len cannot be zero".to_string().into());
        }

        // 验证 arch 字段匹配
        let expected_arch = self.arch();
        if config.arch != expected_arch {
            return Err(format!(
                "config.arch ({:?}) does not match adapter arch ({:?})",
                config.arch, expected_arch
            ).into());
        }

        Ok(())
    }

    /// 获取此适配器对应的架构枚举值。
    fn arch(&self) -> ModelArch;
}

/// Llama 架构适配器（Llama 2/3/4，SmolLM2，InternLM3）。
///
/// 特性：
/// - RMSNorm
/// - GQA (num_kv_heads <= num_heads)
/// - SwiGLU FFN
/// - 标准 RoPE
#[derive(Debug, Clone, Copy, Default)]
pub struct LlamaAdapter;

impl ModelAdapter for LlamaAdapter {
    fn name(&self) -> &'static str {
        "llama"
    }

    fn adapt_config(
        &self,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    fn fusion_hints(&self) -> FusionHints {
        FusionHints {
            use_flash_attention: true,
            use_gqa: true,
            ffn_activation: FfnActivation::SwiGlu,
            sliding_window: None,
            rope_mode: RopeMode::Standard,
        }
    }

    fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout {
        AdapterWeightLayout {
            shapes: vec![
                ("embed_tokens", vec![config.vocab_size, config.hidden_size]),
                ("lm_head", vec![config.vocab_size, config.hidden_size]),
                ("norm.weight", vec![config.hidden_size]),
            ],
            dtype: config.dtype,
        }
    }

    fn arch(&self) -> ModelArch {
        ModelArch::Llama
    }
}

/// Qwen 架构适配器（Qwen 2/2.5/3）。
///
/// 特性：
/// - RMSNorm（更小的 epsilon）
/// - GQA
/// - SwiGLU FFN
/// - QKV 偏置
/// - 更大的 RoPE theta
#[derive(Debug, Clone, Copy, Default)]
pub struct QwenAdapter;

impl ModelAdapter for QwenAdapter {
    fn name(&self) -> &'static str {
        "qwen"
    }

    fn adapt_config(
        &self,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Qwen,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
            dtype,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: true,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    fn fusion_hints(&self) -> FusionHints {
        FusionHints {
            use_flash_attention: true,
            use_gqa: true,
            ffn_activation: FfnActivation::SwiGlu,
            sliding_window: None,
            rope_mode: RopeMode::Standard,
        }
    }

    fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout {
        AdapterWeightLayout {
            shapes: vec![
                ("embed_tokens", vec![config.vocab_size, config.hidden_size]),
                ("lm_head", vec![config.vocab_size, config.hidden_size]),
                ("norm.weight", vec![config.hidden_size]),
            ],
            dtype: config.dtype,
        }
    }

    fn arch(&self) -> ModelArch {
        ModelArch::Qwen
    }
}

/// Gemma 架构适配器（Gemma 1/2）。
///
/// 特性：
/// - RMSNorm（更小的 epsilon）
/// - GQA（Gemma 2）
/// - GeGLU FFN
/// - 标准 RoPE
#[derive(Debug, Clone, Copy, Default)]
pub struct GemmaAdapter;

impl ModelAdapter for GemmaAdapter {
    fn name(&self) -> &'static str {
        "gemma"
    }

    fn adapt_config(
        &self,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gemma,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
            dtype,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    fn fusion_hints(&self) -> FusionHints {
        FusionHints {
            use_flash_attention: true,
            use_gqa: true,
            ffn_activation: FfnActivation::GeGlu,
            sliding_window: None,
            rope_mode: RopeMode::Standard,
        }
    }

    fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout {
        AdapterWeightLayout {
            shapes: vec![
                ("embed_tokens", vec![config.vocab_size, config.hidden_size]),
                ("lm_head", vec![config.vocab_size, config.hidden_size]),
                ("norm.weight", vec![config.hidden_size]),
            ],
            dtype: config.dtype,
        }
    }

    fn arch(&self) -> ModelArch {
        ModelArch::Gemma
    }
}

/// Mistral 架构适配器（Mistral, Ministral）。
///
/// 特性：
/// - RMSNorm
/// - GQA
/// - SwiGLU FFN
/// - 滑动窗口注意力
/// - 大 RoPE theta
#[derive(Debug, Clone, Copy, Default)]
pub struct MistralAdapter {
    /// 滑动窗口大小（None = 无滑动窗口）
    pub sliding_window: Option<usize>,
}

impl ModelAdapter for MistralAdapter {
    fn name(&self) -> &'static str {
        "mistral"
    }

    fn adapt_config(
        &self,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Mistral,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
            dtype,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: 1.0,
            sliding_window: self.sliding_window,
        }
    }

    fn fusion_hints(&self) -> FusionHints {
        FusionHints {
            use_flash_attention: true,
            use_gqa: true,
            ffn_activation: FfnActivation::SwiGlu,
            sliding_window: self.sliding_window,
            rope_mode: RopeMode::Standard,
        }
    }

    fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout {
        AdapterWeightLayout {
            shapes: vec![
                ("embed_tokens", vec![config.vocab_size, config.hidden_size]),
                ("lm_head", vec![config.vocab_size, config.hidden_size]),
                ("norm.weight", vec![config.hidden_size]),
            ],
            dtype: config.dtype,
        }
    }

    fn arch(&self) -> ModelArch {
        ModelArch::Mistral
    }
}

/// Phi 架构适配器（Phi 2/3/4）。
///
/// 特性：
/// - LayerNorm
/// - MHA
/// - GELU FFN
/// - 部分旋转 RoPE
#[derive(Debug, Clone, Copy)]
pub struct PhiAdapter {
    /// 部分旋转因子（0.0-1.0）
    pub partial_rotary_factor: f32,
}

impl Default for PhiAdapter {
    fn default() -> Self {
        Self {
            partial_rotary_factor: 0.5,
        }
    }
}

impl ModelAdapter for PhiAdapter {
    fn name(&self) -> &'static str {
        "phi"
    }

    fn adapt_config(
        &self,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Phi,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads: num_heads, // Phi 使用 MHA
            head_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            dtype,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: false,
            partial_rotary_factor: self.partial_rotary_factor,
            sliding_window: None,
        }
    }

    fn fusion_hints(&self) -> FusionHints {
        FusionHints {
            use_flash_attention: true,
            use_gqa: false,
            ffn_activation: FfnActivation::Gelu,
            sliding_window: None,
            rope_mode: RopeMode::Partial {
                factor: self.partial_rotary_factor,
            },
        }
    }

    fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout {
        AdapterWeightLayout {
            shapes: vec![
                ("embed_tokens", vec![config.vocab_size, config.hidden_size]),
                ("lm_head", vec![config.vocab_size, config.hidden_size]),
                ("norm.weight", vec![config.hidden_size]),
            ],
            dtype: config.dtype,
        }
    }

    fn arch(&self) -> ModelArch {
        ModelArch::Phi
    }
}

/// GPT-2 架构适配器。
///
/// 特性：
/// - LayerNorm
/// - MHA
/// - GELU FFN
/// - 绝对位置编码（无 RoPE）
#[derive(Debug, Clone, Copy, Default)]
pub struct Gpt2Adapter;

impl ModelAdapter for Gpt2Adapter {
    fn name(&self) -> &'static str {
        "gpt2"
    }

    fn adapt_config(
        &self,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gpt2,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads: num_heads, // GPT-2 使用 MHA
            head_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta: 10000.0, // 不使用，但需要设置
            norm_eps: 1e-5,
            dtype,
            quant_type: None,
            rope_interleaved: false,
            has_qkv_bias: true,
            partial_rotary_factor: 1.0,
            sliding_window: None,
        }
    }

    fn fusion_hints(&self) -> FusionHints {
        FusionHints {
            use_flash_attention: true,
            use_gqa: false,
            ffn_activation: FfnActivation::Gelu,
            sliding_window: None,
            rope_mode: RopeMode::Standard,
        }
    }

    fn weight_layout(&self, config: &ModelConfig) -> AdapterWeightLayout {
        AdapterWeightLayout {
            shapes: vec![
                ("wte", vec![config.vocab_size, config.hidden_size]),
                ("lm_head", vec![config.vocab_size, config.hidden_size]),
            ],
            dtype: config.dtype,
        }
    }

    fn arch(&self) -> ModelArch {
        ModelArch::Gpt2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_adapter_config() {
        let adapter = LlamaAdapter;
        let config = adapter.adapt_config(
            4096, // hidden_size
            32,   // num_layers
            32,   // num_heads
            8,    // num_kv_heads (GQA)
            128,  // head_dim
            11008, // intermediate_size
            32000, // vocab_size
            4096,  // max_seq_len
            DType::F32,
        );

        assert_eq!(config.arch, ModelArch::Llama);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.rope_theta, 10000.0);
        assert_eq!(config.norm_eps, 1e-5);
        assert!(!config.has_qkv_bias);
        assert_eq!(config.partial_rotary_factor, 1.0);
    }

    #[test]
    fn test_qwen_adapter_config() {
        let adapter = QwenAdapter;
        let config = adapter.adapt_config(
            4096, 32, 32, 32, 128, 11008, 151936, 8192, DType::F16,
        );

        assert_eq!(config.arch, ModelArch::Qwen);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.norm_eps, 1e-6);
        assert!(config.has_qkv_bias);
    }

    #[test]
    fn test_gemma_adapter_config() {
        let adapter = GemmaAdapter;
        let config = adapter.adapt_config(
            2048, 18, 8, 1, 256, 16384, 256000, 8192, DType::BF16,
        );

        assert_eq!(config.arch, ModelArch::Gemma);
        assert_eq!(config.norm_eps, 1e-6);
    }

    #[test]
    fn test_mistral_adapter_sliding_window() {
        let adapter = MistralAdapter {
            sliding_window: Some(4096),
        };
        let config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 14336, 32000, 32768, DType::F32,
        );

        assert_eq!(config.arch, ModelArch::Mistral);
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn test_phi_adapter_partial_rotary() {
        let adapter = PhiAdapter {
            partial_rotary_factor: 0.5,
        };
        let config = adapter.adapt_config(
            2560, 32, 32, 32, 80, 10240, 51200, 2048, DType::F32,
        );

        assert_eq!(config.arch, ModelArch::Phi);
        assert_eq!(config.num_kv_heads, 32); // MHA
        assert_eq!(config.partial_rotary_factor, 0.5);
    }

    #[test]
    fn test_fusion_hints_llama() {
        let adapter = LlamaAdapter;
        let hints = adapter.fusion_hints();

        assert!(hints.use_flash_attention);
        assert!(hints.use_gqa);
        assert_eq!(hints.ffn_activation, FfnActivation::SwiGlu);
        assert!(hints.sliding_window.is_none());
    }

    #[test]
    fn test_fusion_hints_gemma() {
        let adapter = GemmaAdapter;
        let hints = adapter.fusion_hints();

        assert_eq!(hints.ffn_activation, FfnActivation::GeGlu);
    }

    #[test]
    fn test_fusion_hints_phi() {
        let adapter = PhiAdapter::default();
        let hints = adapter.fusion_hints();

        assert!(!hints.use_gqa); // MHA
        assert_eq!(hints.ffn_activation, FfnActivation::Gelu);
        assert!(matches!(hints.rope_mode, RopeMode::Partial { factor: 0.5 }));
    }

    #[test]
    fn test_validate_config_success() {
        let adapter = LlamaAdapter;
        let config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        assert!(adapter.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_invalid_kv_heads() {
        let adapter = LlamaAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        config.num_kv_heads = 33; // 不能超过 num_heads
        assert!(adapter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_not_divisible() {
        let adapter = LlamaAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        config.num_kv_heads = 7; // 32 不能被 7 整除
        assert!(adapter.validate_config(&config).is_err());
    }

    #[test]
    fn test_weight_layout_llama() {
        let adapter = LlamaAdapter;
        let config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        let layout = adapter.weight_layout(&config);

        assert_eq!(layout.dtype, DType::F32);
        assert_eq!(layout.shapes.len(), 3);
        assert_eq!(layout.shapes[0].0, "embed_tokens");
        assert_eq!(layout.shapes[0].1, vec![32000, 4096]);
    }

    // ── Additional 13 tests ──

    #[test]
    fn test_gpt2_adapter_config_forces_mha_and_qkv_bias() {
        // Arrange
        let adapter = Gpt2Adapter;

        // Act: pass num_kv_heads=1 but GPT-2 overrides to num_heads (MHA)
        let config = adapter.adapt_config(
            768, 12, 12, 1, 64, 3072, 50257, 1024, DType::F32,
        );

        // Assert
        assert_eq!(config.arch, ModelArch::Gpt2);
        assert_eq!(config.num_kv_heads, 12); // forced to num_heads
        assert!(config.has_qkv_bias);
        assert_eq!(adapter.name(), "gpt2");
    }

    #[test]
    fn test_gpt2_adapter_fusion_hints_no_gqa_gelu() {
        // Arrange
        let adapter = Gpt2Adapter;

        // Act
        let hints = adapter.fusion_hints();

        // Assert
        assert!(!hints.use_gqa); // MHA only
        assert_eq!(hints.ffn_activation, FfnActivation::Gelu);
        assert!(hints.sliding_window.is_none());
    }

    #[test]
    fn test_gpt2_adapter_weight_layout_uses_wte() {
        // Arrange
        let adapter = Gpt2Adapter;
        let config = adapter.adapt_config(
            768, 12, 12, 12, 64, 3072, 50257, 1024, DType::F32,
        );

        // Act
        let layout = adapter.weight_layout(&config);

        // Assert: GPT-2 uses "wte" not "embed_tokens"
        let names: Vec<_> = layout.shapes.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"wte"));
        assert!(!names.contains(&"embed_tokens"));
        assert_eq!(layout.dtype, DType::F32);
    }

    #[test]
    fn test_all_adapter_names_are_unique() {
        // Arrange
        let adapters: Vec<Box<dyn ModelAdapter>> = vec![
            Box::new(LlamaAdapter),
            Box::new(QwenAdapter),
            Box::new(GemmaAdapter),
            Box::new(MistralAdapter::default()),
            Box::new(PhiAdapter::default()),
            Box::new(Gpt2Adapter),
        ];

        // Act
        let names: Vec<&str> = adapters.iter().map(|a| a.name()).collect();

        // Assert: all names are distinct
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                assert_ne!(names[i], names[j], "duplicate adapter name: {}", names[i]);
            }
        }
    }

    #[test]
    fn test_all_adapter_arch_values_are_unique() {
        // Arrange
        let adapters: Vec<Box<dyn ModelAdapter>> = vec![
            Box::new(LlamaAdapter),
            Box::new(QwenAdapter),
            Box::new(GemmaAdapter),
            Box::new(MistralAdapter::default()),
            Box::new(PhiAdapter::default()),
            Box::new(Gpt2Adapter),
        ];

        // Act
        let archs: Vec<ModelArch> = adapters.iter().map(|a| a.arch()).collect();

        // Assert
        for i in 0..archs.len() {
            for j in (i + 1)..archs.len() {
                assert_ne!(archs[i], archs[j], "duplicate arch: {:?}", archs[i]);
            }
        }
    }

    #[test]
    fn test_mistral_adapter_default_has_no_sliding_window() {
        // Arrange
        let adapter = MistralAdapter::default();

        // Act
        let config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 14336, 32000, 32768, DType::BF16,
        );
        let hints = adapter.fusion_hints();

        // Assert
        assert_eq!(config.arch, ModelArch::Mistral);
        assert!(config.sliding_window.is_none());
        assert!(hints.sliding_window.is_none());
        assert_eq!(config.rope_theta, 1_000_000.0);
    }

    #[test]
    fn test_mistral_adapter_fusion_hints_carries_sliding_window() {
        // Arrange
        let adapter = MistralAdapter {
            sliding_window: Some(2048),
        };

        // Act
        let hints = adapter.fusion_hints();

        // Assert
        assert_eq!(hints.sliding_window, Some(2048));
        assert_eq!(hints.ffn_activation, FfnActivation::SwiGlu);
    }

    #[test]
    fn test_phi_adapter_custom_factor_propagates_to_rope_mode() {
        // Arrange
        let adapter = PhiAdapter {
            partial_rotary_factor: 0.25,
        };

        // Act
        let config = adapter.adapt_config(
            2560, 32, 32, 32, 80, 10240, 51200, 2048, DType::F16,
        );
        let hints = adapter.fusion_hints();

        // Assert
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert!(matches!(hints.rope_mode, RopeMode::Partial { factor } if (factor - 0.25).abs() < f32::EPSILON));
    }

    #[test]
    fn test_validate_config_rejects_zero_hidden_size() {
        // Arrange
        let adapter = LlamaAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        config.hidden_size = 0;

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_zero_max_seq_len() {
        // Arrange
        let adapter = QwenAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 32, 128, 11008, 151936, 8192, DType::F16,
        );
        config.max_seq_len = 0;

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_arch_mismatch() {
        // Arrange: create a Llama config but validate against QwenAdapter
        let adapter = QwenAdapter;
        let llama = LlamaAdapter;
        let config = llama.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_hints_default_values() {
        // Arrange & Act
        let hints = FusionHints::default();

        // Assert: verify all defaults are conservative
        assert!(!hints.use_flash_attention);
        assert!(!hints.use_gqa);
        assert_eq!(hints.ffn_activation, FfnActivation::SwiGlu);
        assert!(hints.sliding_window.is_none());
        assert_eq!(hints.rope_mode, RopeMode::Standard);
    }

    #[test]
    fn test_weight_layout_gpt2_has_two_entries() {
        // Arrange
        let adapter = Gpt2Adapter;
        let config = adapter.adapt_config(
            768, 12, 12, 12, 64, 3072, 50257, 1024, DType::F32,
        );

        // Act
        let layout = adapter.weight_layout(&config);

        // Assert: GPT-2 only has wte and lm_head, no norm.weight
        assert_eq!(layout.shapes.len(), 2);
        assert_eq!(layout.shapes[0].0, "wte");
        assert_eq!(layout.shapes[0].1, vec![50257, 768]);
        assert_eq!(layout.shapes[1].0, "lm_head");
    }

    // ── Additional 10 tests ──

    #[test]
    fn test_validate_config_rejects_zero_num_layers() {
        // Arrange
        let adapter = LlamaAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        config.num_layers = 0;

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_zero_num_heads() {
        // Arrange
        let adapter = GemmaAdapter;
        let mut config = adapter.adapt_config(
            2048, 18, 8, 1, 256, 16384, 256000, 8192, DType::BF16,
        );
        config.num_heads = 0;

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_zero_head_dim() {
        // Arrange
        let adapter = QwenAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 32, 128, 11008, 151936, 8192, DType::F16,
        );
        config.head_dim = 0;

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_zero_vocab_size() {
        // Arrange
        let adapter = MistralAdapter {
            sliding_window: Some(4096),
        };
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 14336, 32000, 32768, DType::F32,
        );
        config.vocab_size = 0;

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_zero_num_kv_heads() {
        // Arrange
        let adapter = LlamaAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        config.num_kv_heads = 0;

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_accepts_mha_equal_kv_heads() {
        // Arrange: num_kv_heads == num_heads is valid MHA (not just GQA)
        let adapter = QwenAdapter;
        let config = adapter.adapt_config(
            4096, 32, 32, 32, 128, 11008, 151936, 8192, DType::F16,
        );

        // Act
        let result = adapter.validate_config(&config);

        // Assert: num_heads % num_kv_heads == 0 is satisfied when they are equal
        assert!(result.is_ok());
    }

    #[test]
    fn test_phi_adapter_ignores_passed_kv_heads_and_forces_mha() {
        // Arrange: pass num_kv_heads=1 but PhiAdapter should override to num_heads
        let adapter = PhiAdapter {
            partial_rotary_factor: 0.4,
        };

        // Act
        let config = adapter.adapt_config(
            2560, 32, 32, 1, 80, 10240, 51200, 2048, DType::F32,
        );

        // Assert
        assert_eq!(config.num_kv_heads, 32); // forced to num_heads, ignoring the 1 we passed
        assert_eq!(config.partial_rotary_factor, 0.4);
    }

    #[test]
    fn test_qwen_weight_layout_propagates_f16_dtype() {
        // Arrange
        let adapter = QwenAdapter;
        let config = adapter.adapt_config(
            4096, 32, 32, 32, 128, 11008, 151936, 8192, DType::F16,
        );

        // Act
        let layout = adapter.weight_layout(&config);

        // Assert
        assert_eq!(layout.dtype, DType::F16);
        let embed_shape = &layout.shapes[0];
        assert_eq!(embed_shape.0, "embed_tokens");
        assert_eq!(embed_shape.1, vec![151936, 4096]);
    }

    #[test]
    fn test_gemma_fusion_hints_geglu_and_gqa() {
        // Arrange
        let adapter = GemmaAdapter;

        // Act
        let hints = adapter.fusion_hints();

        // Assert
        assert!(hints.use_flash_attention);
        assert!(hints.use_gqa);
        assert_eq!(hints.ffn_activation, FfnActivation::GeGlu);
        assert!(hints.sliding_window.is_none());
        assert_eq!(hints.rope_mode, RopeMode::Standard);
    }

    #[test]
    fn test_mistral_bf16_config_rope_theta_and_sliding_window_propagate() {
        // Arrange
        let adapter = MistralAdapter {
            sliding_window: Some(8192),
        };

        // Act
        let config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 14336, 32000, 65536, DType::BF16,
        );
        let layout = adapter.weight_layout(&config);

        // Assert
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.sliding_window, Some(8192));
        assert_eq!(config.dtype, DType::BF16);
        assert_eq!(layout.dtype, DType::BF16);
    }

    // ── Additional 10 tests (wave-12knb) ──

    #[test]
    fn test_rope_mode_interleaved_equality_and_copy() {
        // Arrange
        let mode = RopeMode::Interleaved;
        let copied = mode;

        // Assert: PartialEq works, Copy works, and Interleaved != Standard
        assert_eq!(mode, copied);
        assert_ne!(mode, RopeMode::Standard);
    }

    #[test]
    fn test_rope_mode_partial_equality_same_and_different_factor() {
        // Arrange
        let a = RopeMode::Partial { factor: 0.5 };
        let b = RopeMode::Partial { factor: 0.5 };
        let c = RopeMode::Partial { factor: 0.25 };

        // Assert: same factor => equal, different factor => not equal
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ffn_activation_copy_and_equality_all_variants() {
        // Arrange & Act
        let variants = [FfnActivation::SwiGlu, FfnActivation::GeGlu, FfnActivation::Gelu];

        // Assert: each variant equals itself via Copy
        for v in &variants {
            assert_eq!(*v, *v);
        }
        // Assert: all pairwise distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn test_fusion_hints_clone_is_independent() {
        // Arrange
        let original = FusionHints {
            use_flash_attention: true,
            use_gqa: true,
            ffn_activation: FfnActivation::GeGlu,
            sliding_window: Some(4096),
            rope_mode: RopeMode::Interleaved,
        };

        // Act
        let cloned = original.clone();

        // Assert: cloned matches original
        assert_eq!(cloned.use_flash_attention, original.use_flash_attention);
        assert_eq!(cloned.use_gqa, original.use_gqa);
        assert_eq!(cloned.ffn_activation, original.ffn_activation);
        assert_eq!(cloned.sliding_window, original.sliding_window);
        assert_eq!(cloned.rope_mode, original.rope_mode);
    }

    #[test]
    fn test_adapter_weight_layout_shapes_contain_correct_vocab_and_hidden() {
        // Arrange
        let adapter = GemmaAdapter;
        let config = adapter.adapt_config(
            1024, 6, 4, 1, 256, 4096, 50000, 2048, DType::BF16,
        );

        // Act
        let layout = adapter.weight_layout(&config);

        // Assert: embed_tokens shape is [vocab_size, hidden_size]
        let embed = layout.shapes.iter().find(|(n, _)| *n == "embed_tokens").unwrap();
        assert_eq!(embed.1, vec![50000, 1024]);
        // Assert: norm.weight shape is [hidden_size]
        let norm = layout.shapes.iter().find(|(n, _)| *n == "norm.weight").unwrap();
        assert_eq!(norm.1, vec![1024]);
    }

    #[test]
    fn test_llama_weight_layout_lm_head_shape_matches_vocab_and_hidden() {
        // Arrange
        let adapter = LlamaAdapter;
        let config = adapter.adapt_config(
            512, 4, 8, 2, 64, 1024, 1000, 512, DType::F32,
        );

        // Act
        let layout = adapter.weight_layout(&config);
        let lm_head = layout.shapes.iter().find(|(n, _)| *n == "lm_head").unwrap();

        // Assert
        assert_eq!(lm_head.1, vec![1000, 512]);
    }

    #[test]
    fn test_validate_config_error_message_contains_field_name() {
        // Arrange
        let adapter = LlamaAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );
        config.vocab_size = 0;

        // Act
        let err = adapter.validate_config(&config).unwrap_err();

        // Assert: error message mentions vocab_size
        let msg = format!("{:?}", err);
        assert!(msg.to_lowercase().contains("vocab_size"));
    }

    #[test]
    fn test_validate_config_error_message_mentions_arch_mismatch() {
        // Arrange
        let adapter = QwenAdapter;
        let llama = LlamaAdapter;
        let config = llama.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F32,
        );

        // Act
        let err = adapter.validate_config(&config).unwrap_err();

        // Assert: error message mentions arch mismatch
        let msg = format!("{:?}", err);
        assert!(msg.to_lowercase().contains("arch"));
    }

    #[test]
    fn test_mistral_weight_layout_shapes_with_sliding_window_config() {
        // Arrange
        let adapter = MistralAdapter {
            sliding_window: Some(4096),
        };
        let config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 14336, 32000, 32768, DType::F32,
        );

        // Act
        let layout = adapter.weight_layout(&config);

        // Assert: weight layout has 3 entries and correct embed shape
        assert_eq!(layout.shapes.len(), 3);
        let embed = layout.shapes.iter().find(|(n, _)| *n == "embed_tokens").unwrap();
        assert_eq!(embed.1, vec![32000, 4096]);
        assert_eq!(layout.dtype, DType::F32);
    }

    #[test]
    fn test_phi_adapter_default_partial_rotary_factor_is_half() {
        // Arrange
        let adapter = PhiAdapter::default();

        // Act
        let config = adapter.adapt_config(
            2560, 32, 32, 32, 80, 10240, 51200, 2048, DType::F32,
        );

        // Assert: default partial_rotary_factor is 0.5
        assert!((adapter.partial_rotary_factor - 0.5).abs() < f32::EPSILON);
        assert!((config.partial_rotary_factor - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.arch, ModelArch::Phi);
    }

    // ── Additional 10 tests (wave-12kod) ──

    #[test]
    fn test_llama_config_dtype_f8e4m3_propagates_correctly() {
        // Arrange
        let adapter = LlamaAdapter;

        // Act
        let config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 11008, 32000, 4096, DType::F8E4M3,
        );
        let layout = adapter.weight_layout(&config);

        // Assert: dtype propagates to both config and layout
        assert_eq!(config.dtype, DType::F8E4M3);
        assert_eq!(layout.dtype, DType::F8E4M3);
    }

    #[test]
    fn test_gpt2_validate_config_succeeds_with_own_arch() {
        // Arrange
        let adapter = Gpt2Adapter;
        let config = adapter.adapt_config(
            768, 12, 12, 12, 64, 3072, 50257, 1024, DType::F32,
        );

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn test_qwen_config_no_sliding_window_and_no_interleaved_rope() {
        // Arrange
        let adapter = QwenAdapter;

        // Act
        let config = adapter.adapt_config(
            4096, 32, 32, 32, 128, 11008, 151936, 8192, DType::F16,
        );

        // Assert
        assert!(config.sliding_window.is_none());
        assert!(!config.rope_interleaved);
        assert_eq!(config.quant_type, None);
    }

    #[test]
    fn test_gemma_config_intermediate_and_vocab_sizes_preserved() {
        // Arrange
        let adapter = GemmaAdapter;
        let hidden = 2048;
        let intermediate = 16384;
        let vocab = 256000;

        // Act
        let config = adapter.adapt_config(
            hidden, 18, 8, 1, 256, intermediate, vocab, 8192, DType::BF16,
        );

        // Assert: all dimensions passed through unchanged
        assert_eq!(config.hidden_size, hidden);
        assert_eq!(config.intermediate_size, intermediate);
        assert_eq!(config.vocab_size, vocab);
        assert_eq!(config.num_layers, 18);
        assert_eq!(config.head_dim, 256);
    }

    #[test]
    fn test_phi_weight_layout_shapes_match_config() {
        // Arrange
        let adapter = PhiAdapter {
            partial_rotary_factor: 0.5,
        };
        let config = adapter.adapt_config(
            2560, 32, 32, 32, 80, 10240, 51200, 2048, DType::F32,
        );

        // Act
        let layout = adapter.weight_layout(&config);

        // Assert: lm_head shape uses vocab_size and hidden_size
        let lm_head = layout.shapes.iter().find(|(n, _)| *n == "lm_head").unwrap();
        assert_eq!(lm_head.1, vec![51200, 2560]);
    }

    #[test]
    fn test_mistral_config_small_sliding_window_value() {
        // Arrange
        let adapter = MistralAdapter {
            sliding_window: Some(512),
        };

        // Act
        let config = adapter.adapt_config(
            1024, 8, 8, 2, 128, 4096, 8000, 2048, DType::F16,
        );
        let hints = adapter.fusion_hints();

        // Assert: sliding window value propagates to both config and hints
        assert_eq!(config.sliding_window, Some(512));
        assert_eq!(hints.sliding_window, Some(512));
        assert_eq!(config.rope_theta, 1_000_000.0);
    }

    #[test]
    fn test_validate_config_rejects_kv_heads_larger_than_heads_on_qwen() {
        // Arrange
        let adapter = QwenAdapter;
        let mut config = adapter.adapt_config(
            4096, 32, 32, 32, 128, 11008, 151936, 8192, DType::F16,
        );
        config.num_kv_heads = 64; // exceeds num_heads=32

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_non_divisible_gqa_on_mistral() {
        // Arrange
        let adapter = MistralAdapter {
            sliding_window: Some(4096),
        };
        let mut config = adapter.adapt_config(
            4096, 32, 32, 8, 128, 14336, 32000, 32768, DType::F32,
        );
        config.num_kv_heads = 5; // 32 % 5 != 0

        // Act
        let result = adapter.validate_config(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_hints_custom_non_default_values() {
        // Arrange
        let hints = FusionHints {
            use_flash_attention: true,
            use_gqa: false,
            ffn_activation: FfnActivation::Gelu,
            sliding_window: Some(1024),
            rope_mode: RopeMode::Interleaved,
        };

        // Assert: every field is set to a non-default value
        assert!(hints.use_flash_attention);
        assert!(!hints.use_gqa);
        assert_eq!(hints.ffn_activation, FfnActivation::Gelu);
        assert_eq!(hints.sliding_window, Some(1024));
        assert_eq!(hints.rope_mode, RopeMode::Interleaved);
    }

    #[test]
    fn test_llama_config_all_fields_consistent() {
        // Arrange
        let adapter = LlamaAdapter;

        // Act
        let config = adapter.adapt_config(
            512, 4, 8, 2, 64, 1024, 1000, 512, DType::BF16,
        );

        // Assert: verify full field consistency for a small model config
        assert_eq!(config.arch, ModelArch::Llama);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.intermediate_size, 1024);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.max_seq_len, 512);
        assert_eq!(config.dtype, DType::BF16);
        assert_eq!(config.rope_theta, 10000.0);
        assert_eq!(config.norm_eps, 1e-5);
        assert!(!config.rope_interleaved);
        assert!(!config.has_qkv_bias);
        assert_eq!(config.partial_rotary_factor, 1.0);
        assert!(config.sliding_window.is_none());
        assert!(config.quant_type.is_none());
    }
}
