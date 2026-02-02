use std::fmt;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::driver::DeviceRepr;

use crate::backend_trait::{
    AttentionTopology, Backend, BackendError, BackendResult, KvCacheHandle, LogitsTensor,
    TensorLookup,
};
use crate::cpu_kernels;
use crate::kernel_types::{
    alibi_slopes, GeneratorForwardConfig, KvCacheConfig, PositionEncoding, SamplingConfig,
};

#[derive(Debug, Clone, Copy, PartialEq)]
struct WorkspaceConfig {
    max_seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    position_encoding: PositionEncoding,
    rope_theta: f32,
    rope_scale: f32,
    rope_interleaved: bool,
    rope_precompute: bool,
}

struct CpuWorkspace {
    config: WorkspaceConfig,
    token_ids: Vec<u32>,
    positions: Vec<i32>,
    hidden: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_act: Vec<f32>,
    ffn_out: Vec<f32>,
    last_hidden: Vec<f32>,
    rope_cache: Option<cpu_kernels::RopeCache>,
    alibi_slopes: Option<Vec<f32>>,
}

impl CpuWorkspace {
    fn allocate(config: WorkspaceConfig) -> BackendResult<Self> {
        let hidden_len = config
            .max_seq_len
            .checked_mul(config.hidden_size)
            .ok_or_else(|| BackendError::InvalidConfig("hidden buffer size overflow".into()))?;
        let qkv_stride = config
            .num_heads
            .checked_mul(config.head_dim)
            .and_then(|q| q.checked_add(2 * config.num_kv_heads * config.head_dim))
            .ok_or_else(|| BackendError::InvalidConfig("qkv stride overflow".into()))?;
        let qkv_len = config
            .max_seq_len
            .checked_mul(qkv_stride)
            .ok_or_else(|| BackendError::InvalidConfig("qkv buffer size overflow".into()))?;
        let ffn_len = config
            .max_seq_len
            .checked_mul(config.ffn_dim)
            .ok_or_else(|| BackendError::InvalidConfig("ffn buffer size overflow".into()))?;
        let rope_cache =
            if config.position_encoding == PositionEncoding::Rope && config.rope_precompute {
                Some(cpu_kernels::RopeCache::new(
                    config.max_seq_len,
                    config.head_dim,
                    config.rope_theta,
                    config.rope_scale,
                ))
            } else {
                None
            };
        let alibi_slopes = if config.position_encoding == PositionEncoding::Alibi {
            Some(alibi_slopes(config.num_heads))
        } else {
            None
        };

        Ok(Self {
            config,
            token_ids: vec![0u32; config.max_seq_len],
            positions: vec![0i32; config.max_seq_len],
            hidden: vec![0f32; hidden_len],
            norm: vec![0f32; hidden_len],
            qkv: vec![0f32; qkv_len],
            attn_out: vec![0f32; hidden_len],
            ffn_gate: vec![0f32; ffn_len],
            ffn_up: vec![0f32; ffn_len],
            ffn_act: vec![0f32; ffn_len],
            ffn_out: vec![0f32; hidden_len],
            last_hidden: vec![0f32; config.hidden_size],
            rope_cache,
            alibi_slopes,
        })
    }
}

struct KvCacheEntry {
    storage: KvCacheStorage,
    config: KvCacheConfig,
    used: usize,
}

enum KvCacheStorage {
    Contiguous(Vec<f32>),
    Paged {
        pages: Vec<Vec<f32>>,
        page_size: usize,
        pages_per_layer: usize,
    },
}

impl KvCacheEntry {
    fn is_paged(&self) -> bool {
        matches!(self.storage, KvCacheStorage::Paged { .. })
    }

    fn page_size(&self) -> usize {
        match self.storage {
            KvCacheStorage::Paged { page_size, .. } => page_size,
            KvCacheStorage::Contiguous(_) => self.config.max_seq_len,
        }
    }

    fn pages_per_layer(&self) -> usize {
        match self.storage {
            KvCacheStorage::Paged {
                pages_per_layer, ..
            } => pages_per_layer,
            KvCacheStorage::Contiguous(_) => 1,
        }
    }
}

enum QkvLayerPlan {
    Direct {
        weight: String,
        bias: Option<String>,
    },
    Fused {
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    },
}

struct QkvPlan {
    layers: Vec<QkvLayerPlan>,
}

const EMBEDDING_WEIGHT_NAMES: &[&str] = &[
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
    "transformer.wte.weight",
    "model.tok_embeddings.weight",
    "embeddings.word_embeddings.weight",
    "model.embeddings.word_embeddings.weight",
];

const FINAL_NORM_NAMES: &[&str] = &[
    "model.norm.weight",
    "transformer.ln_f.weight",
    "norm.weight",
    "model.final_layernorm.weight",
    "encoder.layer_norm.weight",
];

const LM_HEAD_NAMES: &[&str] = &[
    "lm_head.weight",
    "model.lm_head.weight",
    "embed_out.weight",
    "model.embed_out.weight",
    "transformer.lm_head.weight",
];

const SCORE_HEAD_NAMES: &[&str] = &[
    "score.weight",
    "model.score.weight",
    "classifier.weight",
    "model.classifier.weight",
    "rerank_head.weight",
    "ranker.weight",
];

const ATTN_NORM_PATTERNS: &[&str] = &[
    "model.layers.{layer}.input_layernorm.weight",
    "model.layers.{layer}.attention_norm.weight",
    "model.layers.{layer}.attn_norm.weight",
    "transformer.h.{layer}.ln_1.weight",
    "model.layers.{layer}.ln1.weight",
    "layers.{layer}.input_layernorm.weight",
];

const FFN_NORM_PATTERNS: &[&str] = &[
    "model.layers.{layer}.post_attention_layernorm.weight",
    "model.layers.{layer}.ffn_norm.weight",
    "model.layers.{layer}.mlp_norm.weight",
    "transformer.h.{layer}.ln_2.weight",
    "model.layers.{layer}.ln2.weight",
    "layers.{layer}.post_attention_layernorm.weight",
];

const Q_PROJ_PATTERNS: &[&str] = &[
    "model.layers.{layer}.self_attn.q_proj.weight",
    "model.layers.{layer}.attention.q_proj.weight",
    "transformer.h.{layer}.attn.q_proj.weight",
    "layers.{layer}.self_attn.q_proj.weight",
];

const K_PROJ_PATTERNS: &[&str] = &[
    "model.layers.{layer}.self_attn.k_proj.weight",
    "model.layers.{layer}.attention.k_proj.weight",
    "transformer.h.{layer}.attn.k_proj.weight",
    "layers.{layer}.self_attn.k_proj.weight",
];

const V_PROJ_PATTERNS: &[&str] = &[
    "model.layers.{layer}.self_attn.v_proj.weight",
    "model.layers.{layer}.attention.v_proj.weight",
    "transformer.h.{layer}.attn.v_proj.weight",
    "layers.{layer}.self_attn.v_proj.weight",
];

const O_PROJ_PATTERNS: &[&str] = &[
    "model.layers.{layer}.self_attn.o_proj.weight",
    "model.layers.{layer}.self_attn.out_proj.weight",
    "model.layers.{layer}.attention.o_proj.weight",
    "transformer.h.{layer}.attn.c_proj.weight",
    "transformer.h.{layer}.attn.out_proj.weight",
];

const GATE_PROJ_PATTERNS: &[&str] = &[
    "model.layers.{layer}.mlp.gate_proj.weight",
    "model.layers.{layer}.mlp.w1.weight",
    "model.layers.{layer}.mlp.gate.weight",
    "transformer.h.{layer}.mlp.gate_proj.weight",
];

const UP_PROJ_PATTERNS: &[&str] = &[
    "model.layers.{layer}.mlp.up_proj.weight",
    "model.layers.{layer}.mlp.w3.weight",
    "model.layers.{layer}.mlp.up.weight",
    "transformer.h.{layer}.mlp.up_proj.weight",
];

const DOWN_PROJ_PATTERNS: &[&str] = &[
    "model.layers.{layer}.mlp.down_proj.weight",
    "model.layers.{layer}.mlp.w2.weight",
    "model.layers.{layer}.mlp.down.weight",
    "transformer.h.{layer}.mlp.c_proj.weight",
];

const QKV_FUSED_PATTERNS: &[&str] = &[
    "model.layers.{layer}.self_attn.qkv_proj.weight",
    "model.layers.{layer}.self_attn.W_pack.weight",
    "model.layers.{layer}.self_attn.qkv.weight",
    "model.layers.{layer}.self_attn.in_proj_weight",
    "transformer.h.{layer}.attn.c_attn.weight",
];

const FAST_PATH_MAX_LAYERS: usize = 8;
const FAST_PATH_MAX_HIDDEN: usize = 2048;
const FAST_PATH_MAX_FFN: usize = 8192;

fn layer_candidates(patterns: &[&str], layer: usize) -> Vec<String> {
    let layer = layer.to_string();
    patterns
        .iter()
        .map(|pattern| pattern.replace("{layer}", &layer))
        .collect()
}

fn embedding_candidates() -> Vec<String> {
    EMBEDDING_WEIGHT_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

fn final_norm_candidates() -> Vec<String> {
    FINAL_NORM_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

fn lm_head_candidates() -> Vec<String> {
    LM_HEAD_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

fn score_head_candidates() -> Vec<String> {
    SCORE_HEAD_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

fn find_tensor_name(
    weights: &dyn TensorLookup<CpuBackend>,
    candidates: &[String],
) -> Option<String> {
    for name in candidates {
        if weights.tensor_f32(name).is_some() {
            return Some(name.clone());
        }
    }
    None
}

fn bias_candidates(weight_name: &str) -> Vec<String> {
    let mut out = Vec::new();
    if weight_name.ends_with("in_proj_weight") {
        out.push(weight_name.replace("in_proj_weight", "in_proj_bias"));
    }
    if weight_name.ends_with(".weight") {
        out.push(weight_name.replace(".weight", ".bias"));
    } else if weight_name.ends_with("weight") {
        out.push(weight_name.replacen("weight", "bias", 1));
    }
    out
}

pub struct CpuBackend {
    kv_caches: Mutex<Vec<KvCacheEntry>>,
    workspace: Mutex<Option<CpuWorkspace>>,
    qkv_plan: Mutex<Option<QkvPlan>>,
    logits: Mutex<Vec<Vec<f32>>>,
    rng_state: Mutex<u64>,
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend").finish()
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|v| v.as_nanos() as u64)
            .unwrap_or(0x1234_5678_9abc_def0);
        Self {
            kv_caches: Mutex::new(Vec::new()),
            workspace: Mutex::new(None),
            qkv_plan: Mutex::new(None),
            logits: Mutex::new(Vec::new()),
            rng_state: Mutex::new(seed.max(1)),
        }
    }

    pub fn int8_mm(
        &self,
        input: &[f32],
        weight: &[i8],
        scale: f32,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        cpu_kernels::matmul_int8(input, weight, scale, output, m, n, k)
    }

    pub fn int4_mm(
        &self,
        input: &[f32],
        weight_packed: &[u8],
        scale: f32,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        cpu_kernels::matmul_int4(input, weight_packed, scale, output, m, n, k)
    }

    pub fn int2_mm(
        &self,
        input: &[f32],
        weight_packed: &[u8],
        scale: f32,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        cpu_kernels::matmul_int2(input, weight_packed, scale, output, m, n, k)
    }

    pub fn int1_mm(
        &self,
        input: &[f32],
        weight_packed: &[u8],
        scale: f32,
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        cpu_kernels::matmul_int1(input, weight_packed, scale, output, m, n, k)
    }

    fn hidden_size(config: &GeneratorForwardConfig) -> BackendResult<usize> {
        let hidden = config
            .num_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| BackendError::InvalidConfig("hidden size overflow".into()))?;
        Ok(hidden)
    }

    fn resolve_weight_name(
        weights: &dyn TensorLookup<Self>,
        candidates: &[String],
    ) -> BackendResult<String> {
        for name in candidates {
            if weights.tensor_f32(name).is_some() {
                return Ok(name.clone());
            }
        }
        Err(BackendError::MissingTensor(candidates.join(", ")))
    }

    fn resolve_weight<'a>(
        weights: &'a dyn TensorLookup<Self>,
        name: &str,
    ) -> BackendResult<&'a Vec<f32>> {
        weights
            .tensor_f32(name)
            .ok_or_else(|| BackendError::MissingTensor(name.to_string()))
    }

    fn resolve_bias<'a>(
        weights: &'a dyn TensorLookup<Self>,
        weight_name: &str,
    ) -> Option<&'a Vec<f32>> {
        for candidate in bias_candidates(weight_name) {
            if let Some(bias) = weights.tensor_f32(&candidate) {
                return Some(bias);
            }
        }
        None
    }

    fn resolve_weight_and_bias<'a>(
        weights: &'a dyn TensorLookup<Self>,
        candidates: &[String],
    ) -> BackendResult<(&'a Vec<f32>, Option<&'a Vec<f32>>)> {
        let name = Self::resolve_weight_name(weights, candidates)?;
        let weight = Self::resolve_weight(weights, &name)?;
        let bias = Self::resolve_bias(weights, &name);
        Ok((weight, bias))
    }

    fn resolve_shape(
        weights: &dyn TensorLookup<Self>,
        candidates: &[String],
    ) -> BackendResult<Vec<usize>> {
        let name = Self::resolve_weight_name(weights, candidates)?;
        let shape = weights
            .tensor_shape(&name)
            .ok_or_else(|| BackendError::MissingTensor(name.clone()))?;
        Ok(shape.to_vec())
    }

    fn deduce_out_dim(shape: &[usize], in_dim: usize) -> BackendResult<usize> {
        if shape.len() != 2 {
            return Err(BackendError::InvalidConfig(format!(
                "expected 2D weight shape, got {shape:?}"
            )));
        }
        if shape[1] == in_dim {
            Ok(shape[0])
        } else if shape[0] == in_dim {
            Ok(shape[1])
        } else {
            Err(BackendError::InvalidConfig(format!(
                "weight shape {shape:?} not compatible with input dim {in_dim}"
            )))
        }
    }

    fn ensure_workspace(
        &self,
        weights: &dyn TensorLookup<Self>,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<()> {
        let mut workspace = self.workspace.lock().expect("workspace lock poisoned");
        if let Some(existing) = workspace.as_ref() {
            if existing.config.max_seq_len != config.max_seq_len
                || existing.config.hidden_size != Self::hidden_size(config)?
                || existing.config.num_heads != config.num_heads
                || existing.config.num_kv_heads != config.num_kv_heads
                || existing.config.head_dim != config.head_dim
                || existing.config.vocab_size != config.vocab_size
                || existing.config.position_encoding != config.position_encoding
                || existing.config.rope_theta != config.rope_theta
                || existing.config.rope_scale != config.rope_scale
                || existing.config.rope_interleaved != config.rope_interleaved
                || existing.config.rope_precompute != config.rope_precompute
            {
                return Err(BackendError::InvalidConfig(
                    "workspace config mismatch".into(),
                ));
            }
            return Ok(());
        }

        let hidden_size = Self::hidden_size(config)?;
        let ffn_shape = Self::resolve_shape(weights, &layer_candidates(GATE_PROJ_PATTERNS, 0))?;
        let ffn_dim = Self::deduce_out_dim(&ffn_shape, hidden_size)?;
        let ws_config = WorkspaceConfig {
            max_seq_len: config.max_seq_len,
            hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            ffn_dim,
            vocab_size: config.vocab_size,
            position_encoding: config.position_encoding,
            rope_theta: config.rope_theta,
            rope_scale: config.rope_scale,
            rope_interleaved: config.rope_interleaved,
            rope_precompute: config.rope_precompute,
        };
        *workspace = Some(CpuWorkspace::allocate(ws_config)?);
        Ok(())
    }

    fn ensure_qkv_plan(
        &self,
        weights: &dyn TensorLookup<Self>,
        config: &GeneratorForwardConfig,
        hidden_size: usize,
    ) -> BackendResult<()> {
        let mut plan_guard = self.qkv_plan.lock().expect("qkv plan lock poisoned");
        if plan_guard.is_some() {
            return Ok(());
        }
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let fused_candidates = layer_candidates(QKV_FUSED_PATTERNS, layer);
            if let Some(fused_name) = find_tensor_name(weights, &fused_candidates) {
                let bias = bias_candidates(&fused_name)
                    .into_iter()
                    .find(|name| weights.tensor_f32(name).is_some());
                layers.push(QkvLayerPlan::Direct {
                    weight: fused_name,
                    bias,
                });
                continue;
            }

            let q_name =
                Self::resolve_weight_name(weights, &layer_candidates(Q_PROJ_PATTERNS, layer))?;
            let k_name =
                Self::resolve_weight_name(weights, &layer_candidates(K_PROJ_PATTERNS, layer))?;
            let v_name =
                Self::resolve_weight_name(weights, &layer_candidates(V_PROJ_PATTERNS, layer))?;

            let q_weight = Self::resolve_weight(weights, &q_name)?;
            let k_weight = Self::resolve_weight(weights, &k_name)?;
            let v_weight = Self::resolve_weight(weights, &v_name)?;

            let q_shape = weights
                .tensor_shape(&q_name)
                .ok_or_else(|| BackendError::MissingTensor(q_name.clone()))?;
            let k_shape = weights
                .tensor_shape(&k_name)
                .ok_or_else(|| BackendError::MissingTensor(k_name.clone()))?;
            let v_shape = weights
                .tensor_shape(&v_name)
                .ok_or_else(|| BackendError::MissingTensor(v_name.clone()))?;

            let q_out = Self::deduce_out_dim(q_shape, hidden_size)?;
            let k_out = Self::deduce_out_dim(k_shape, hidden_size)?;
            let v_out = Self::deduce_out_dim(v_shape, hidden_size)?;

            let expected_kv = config.num_kv_heads * config.head_dim;
            if q_out != hidden_size || k_out != expected_kv || v_out != expected_kv {
                return Err(BackendError::InvalidConfig(
                    "qkv projection dims mismatch".into(),
                ));
            }

            let fused_out = q_out + k_out + v_out;
            let fused_len = fused_out
                .checked_mul(hidden_size)
                .ok_or_else(|| BackendError::InvalidConfig("fused qkv size overflow".into()))?;
            let mut fused = vec![0f32; fused_len];

            let q_len = q_out * hidden_size;
            let k_len = k_out * hidden_size;
            let v_len = v_out * hidden_size;
            if q_weight.len() < q_len || k_weight.len() < k_len || v_weight.len() < v_len {
                return Err(BackendError::InvalidConfig(
                    "qkv weight size mismatch".into(),
                ));
            }
            fused[..q_len].copy_from_slice(&q_weight[..q_len]);
            fused[q_len..q_len + k_len].copy_from_slice(&k_weight[..k_len]);
            fused[q_len + k_len..q_len + k_len + v_len].copy_from_slice(&v_weight[..v_len]);

            let bias_present = Self::resolve_bias(weights, &q_name)
                .or_else(|| Self::resolve_bias(weights, &k_name))
                .or_else(|| Self::resolve_bias(weights, &v_name))
                .is_some();

            let fused_bias = if bias_present {
                let mut bias_buf = vec![0f32; fused_out];
                if let Some(q_bias) = Self::resolve_bias(weights, &q_name) {
                    if q_bias.len() < q_out {
                        return Err(BackendError::InvalidConfig("q bias size mismatch".into()));
                    }
                    bias_buf[..q_out].copy_from_slice(&q_bias[..q_out]);
                }
                if let Some(k_bias) = Self::resolve_bias(weights, &k_name) {
                    if k_bias.len() < k_out {
                        return Err(BackendError::InvalidConfig("k bias size mismatch".into()));
                    }
                    bias_buf[q_out..q_out + k_out].copy_from_slice(&k_bias[..k_out]);
                }
                if let Some(v_bias) = Self::resolve_bias(weights, &v_name) {
                    if v_bias.len() < v_out {
                        return Err(BackendError::InvalidConfig("v bias size mismatch".into()));
                    }
                    bias_buf[q_out + k_out..q_out + k_out + v_out]
                        .copy_from_slice(&v_bias[..v_out]);
                }
                Some(bias_buf)
            } else {
                None
            };

            layers.push(QkvLayerPlan::Fused {
                weight: fused,
                bias: fused_bias,
            });
        }
        *plan_guard = Some(QkvPlan { layers });
        Ok(())
    }

    fn use_fast_path(
        &self,
        weights: &dyn TensorLookup<Self>,
        config: &GeneratorForwardConfig,
        seq_len: usize,
    ) -> BackendResult<bool> {
        if seq_len != 1 {
            return Ok(false);
        }
        let hidden_size = Self::hidden_size(config)?;
        let ffn_shape = Self::resolve_shape(weights, &layer_candidates(GATE_PROJ_PATTERNS, 0))?;
        let ffn_dim = Self::deduce_out_dim(&ffn_shape, hidden_size)?;
        Ok(config.num_layers <= FAST_PATH_MAX_LAYERS
            && hidden_size <= FAST_PATH_MAX_HIDDEN
            && ffn_dim <= FAST_PATH_MAX_FFN)
    }

    fn forward_hidden_fast(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        mut kv_cache: Option<&mut KvCacheEntry>,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(BackendError::InvalidConfig("empty tokens".into()));
        }
        if tokens.len() > config.max_seq_len {
            return Err(BackendError::InvalidConfig(
                "tokens exceed max_seq_len".into(),
            ));
        }
        if topology.is_tree() {
            return Err(BackendError::Unimplemented(
                "cpu backend does not support tree topology",
            ));
        }

        let hidden_size = Self::hidden_size(config)?;
        if config.num_kv_heads == 0 || config.num_heads == 0 || config.head_dim == 0 {
            return Err(BackendError::InvalidConfig("invalid head config".into()));
        }

        let ffn_shape = Self::resolve_shape(weights, &layer_candidates(GATE_PROJ_PATTERNS, 0))?;
        let ffn_dim = Self::deduce_out_dim(&ffn_shape, hidden_size)?;
        self.ensure_qkv_plan(weights, config, hidden_size)?;

        let seq_len = tokens.len();
        let mut kv_start = 0usize;
        if let Some(cache) = kv_cache.as_deref_mut() {
            if cache.config.num_layers != config.num_layers
                || cache.config.num_heads != config.num_kv_heads
                || cache.config.head_dim != config.head_dim
            {
                return Err(BackendError::InvalidKvCache(
                    "kv cache config mismatch".into(),
                ));
            }
            if cache.config.dtype_size != std::mem::size_of::<f32>() {
                return Err(BackendError::InvalidKvCache(
                    "cpu kv cache expects f32 dtype".into(),
                ));
            }
            if seq_len > 1 && cache.used != 0 {
                cache.used = 0;
            }
            if seq_len > cache.config.max_seq_len.saturating_sub(cache.used) {
                return Err(BackendError::InvalidKvCache("kv cache exhausted".into()));
            }
            kv_start = cache.used;
        }

        let mut positions = vec![0i32; seq_len];
        for i in 0..seq_len {
            positions[i] = (kv_start + i) as i32;
        }

        let embed_name = Self::resolve_weight_name(weights, &embedding_candidates())?;
        let embed_weight = Self::resolve_weight(weights, &embed_name)?;
        let embed_shape = weights
            .tensor_shape(&embed_name)
            .ok_or_else(|| BackendError::MissingTensor(embed_name.clone()))?;
        if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
            return Err(BackendError::InvalidConfig(
                "embedding weight shape mismatch".into(),
            ));
        }

        let mut hidden = vec![0f32; seq_len * hidden_size];
        let mut norm = vec![0f32; seq_len * hidden_size];
        let mut attn_out = vec![0f32; seq_len * hidden_size];
        let mut ffn_out = vec![0f32; seq_len * hidden_size];

        let q_out = hidden_size;
        let kv_out = config.num_kv_heads * config.head_dim;
        let qkv_stride = q_out + 2 * kv_out;
        let mut qkv = vec![0f32; seq_len * qkv_stride];

        let mut ffn_gate = vec![0f32; seq_len * ffn_dim];
        let mut ffn_up = vec![0f32; seq_len * ffn_dim];
        let mut ffn_act = vec![0f32; seq_len * ffn_dim];

        cpu_kernels::embedding_lookup(
            tokens,
            embed_weight,
            &mut hidden[..seq_len * hidden_size],
            config.vocab_size,
            hidden_size,
        )?;

        let rope_cache =
            if config.position_encoding == PositionEncoding::Rope && config.rope_precompute {
                Some(cpu_kernels::RopeCache::new(
                    seq_len,
                    config.head_dim,
                    config.rope_theta,
                    config.rope_scale,
                ))
            } else {
                None
            };
        let alibi_slopes = if config.position_encoding == PositionEncoding::Alibi {
            Some(alibi_slopes(config.num_heads))
        } else {
            None
        };

        let qkv_plan_guard = self.qkv_plan.lock().expect("qkv plan lock poisoned");
        let qkv_plan = qkv_plan_guard
            .as_ref()
            .ok_or_else(|| BackendError::InvalidConfig("qkv plan missing".into()))?;

        let positions_ref = &positions[..seq_len];
        let rotary_dim = if config.position_encoding == PositionEncoding::Rope {
            config.head_dim
        } else {
            0
        };
        let rope_cache_ref = rope_cache.as_ref();

        let q_len = seq_len * q_out;
        let kv_len = seq_len * kv_out;

        for layer in 0..config.num_layers {
            let (attn_norm, _) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(ATTN_NORM_PATTERNS, layer),
            )?;
            let (ffn_norm, _) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(FFN_NORM_PATTERNS, layer),
            )?;

            cpu_kernels::rms_norm(
                &hidden[..seq_len * hidden_size],
                attn_norm,
                &mut norm[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                1e-5,
            )?;

            let (qkv_weight, qkv_bias) = match &qkv_plan.layers[layer] {
                QkvLayerPlan::Direct { weight, bias } => {
                    let weight = Self::resolve_weight(weights, weight)?;
                    let bias = bias.as_ref().and_then(|name| weights.tensor_f32(name));
                    (weight.as_slice(), bias.map(|b| b.as_slice()))
                }
                QkvLayerPlan::Fused { weight, bias } => {
                    (weight.as_slice(), bias.as_ref().map(|b| b.as_slice()))
                }
            };

            cpu_kernels::fused_qkv_rope(
                &norm[..seq_len * hidden_size],
                qkv_weight,
                qkv_bias,
                &mut qkv[..seq_len * qkv_stride],
                seq_len,
                hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                rotary_dim,
                config.rope_theta,
                config.rope_scale,
                config.rope_interleaved,
                rope_cache_ref,
                positions_ref,
            )?;

            if let Some(cache) = kv_cache.as_deref_mut() {
                let qkv_slice = &qkv[..seq_len * qkv_stride];
                let k_src = &qkv_slice[q_len..q_len + kv_len];
                let v_src = &qkv_slice[q_len + kv_len..q_len + 2 * kv_len];

                if cache.is_paged() {
                    let page_size = cache.page_size();
                    let pages_per_layer = cache.pages_per_layer();
                    let page_len = page_size * kv_out * 2;
                    let KvCacheStorage::Paged { pages, .. } = &mut cache.storage else {
                        return Err(BackendError::InvalidKvCache(
                            "expected paged kv cache".into(),
                        ));
                    };
                    if pages.len() < config.num_layers * pages_per_layer {
                        return Err(BackendError::InvalidKvCache(
                            "paged kv cache size mismatch".into(),
                        ));
                    }

                    let mut remaining = seq_len;
                    let mut src_token = 0usize;
                    let mut global = kv_start;
                    while remaining > 0 {
                        let page = global / page_size;
                        let offset = global - page * page_size;
                        let take = remaining.min(page_size - offset);
                        let page_index = layer * pages_per_layer + page;
                        let page_buf = pages
                            .get_mut(page_index)
                            .ok_or_else(|| BackendError::InvalidKvCache("page index".into()))?;
                        if page_buf.len() != page_len {
                            return Err(BackendError::InvalidKvCache(
                                "paged kv cache page size mismatch".into(),
                            ));
                        }

                        let src_offset = src_token * kv_out;
                        let src_len = take * kv_out;
                        let k_dst = offset * kv_out;
                        let v_dst = page_size * kv_out + offset * kv_out;
                        page_buf[k_dst..k_dst + src_len]
                            .copy_from_slice(&k_src[src_offset..src_offset + src_len]);
                        page_buf[v_dst..v_dst + src_len]
                            .copy_from_slice(&v_src[src_offset..src_offset + src_len]);

                        remaining -= take;
                        src_token += take;
                        global += take;
                    }

                    let kv_seq_len = kv_start + seq_len;
                    let q_view = &qkv[..q_len];
                    let layer_start = layer * pages_per_layer;
                    let layer_end = layer_start + pages_per_layer;
                    let layer_pages = &pages[layer_start..layer_end];
                    cpu_kernels::flash_attention_paged(
                        q_view,
                        layer_pages,
                        page_size,
                        &mut attn_out[..seq_len * hidden_size],
                        seq_len,
                        kv_seq_len,
                        config.num_heads,
                        config.num_kv_heads,
                        config.head_dim,
                        true,
                        1.0 / (config.head_dim as f32).sqrt(),
                        alibi_slopes.as_deref(),
                        kv_start,
                    )?;
                } else {
                    let KvCacheStorage::Contiguous(buffer) = &mut cache.storage else {
                        return Err(BackendError::InvalidKvCache(
                            "expected contiguous kv cache".into(),
                        ));
                    };
                    let layer_stride = cache.config.max_seq_len * kv_out * 2;
                    let layer_base = layer * layer_stride;
                    let k_dst_offset = layer_base + kv_start * kv_out;
                    let v_dst_offset =
                        layer_base + cache.config.max_seq_len * kv_out + kv_start * kv_out;

                    buffer[k_dst_offset..k_dst_offset + kv_len].copy_from_slice(k_src);
                    buffer[v_dst_offset..v_dst_offset + kv_len].copy_from_slice(v_src);

                    let kv_seq_len = kv_start + seq_len;
                    let k_cache_view = &buffer[..kv_seq_len * kv_out];
                    let v_cache_view = &buffer[cache.config.max_seq_len * kv_out
                        ..cache.config.max_seq_len * kv_out + kv_seq_len * kv_out];
                    let q_view = &qkv[..q_len];

                    cpu_kernels::flash_attention(
                        q_view,
                        k_cache_view,
                        v_cache_view,
                        &mut attn_out[..seq_len * hidden_size],
                        seq_len,
                        kv_seq_len,
                        config.num_heads,
                        config.num_kv_heads,
                        config.head_dim,
                        true,
                        1.0 / (config.head_dim as f32).sqrt(),
                        alibi_slopes.as_deref(),
                        kv_start,
                    )?;
                }
            } else {
                let q_view = &qkv[..q_len];
                let k_view = &qkv[q_len..q_len + kv_len];
                let v_view = &qkv[q_len + kv_len..q_len + 2 * kv_len];
                cpu_kernels::flash_attention(
                    q_view,
                    k_view,
                    v_view,
                    &mut attn_out[..seq_len * hidden_size],
                    seq_len,
                    seq_len,
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    true,
                    1.0 / (config.head_dim as f32).sqrt(),
                    alibi_slopes.as_deref(),
                    0,
                )?;
            }

            let (o_proj, o_bias) =
                Self::resolve_weight_and_bias(weights, &layer_candidates(O_PROJ_PATTERNS, layer))?;
            cpu_kernels::linear(
                &attn_out[..seq_len * hidden_size],
                o_proj,
                o_bias.map(|b| b.as_slice()),
                &mut norm[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                hidden_size,
            )?;
            cpu_kernels::add(
                &hidden[..seq_len * hidden_size],
                &norm[..seq_len * hidden_size],
                &mut attn_out[..seq_len * hidden_size],
            )?;
            hidden[..seq_len * hidden_size].copy_from_slice(&attn_out[..seq_len * hidden_size]);

            cpu_kernels::rms_norm(
                &hidden[..seq_len * hidden_size],
                ffn_norm,
                &mut norm[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                1e-5,
            )?;

            let (gate_proj, gate_bias) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(GATE_PROJ_PATTERNS, layer),
            )?;
            let (up_proj, up_bias) =
                Self::resolve_weight_and_bias(weights, &layer_candidates(UP_PROJ_PATTERNS, layer))?;
            let (down_proj, down_bias) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(DOWN_PROJ_PATTERNS, layer),
            )?;

            cpu_kernels::linear(
                &norm[..seq_len * hidden_size],
                gate_proj,
                gate_bias.map(|b| b.as_slice()),
                &mut ffn_gate[..seq_len * ffn_dim],
                seq_len,
                ffn_dim,
                hidden_size,
            )?;
            cpu_kernels::linear(
                &norm[..seq_len * hidden_size],
                up_proj,
                up_bias.map(|b| b.as_slice()),
                &mut ffn_up[..seq_len * ffn_dim],
                seq_len,
                ffn_dim,
                hidden_size,
            )?;
            cpu_kernels::fused_gate_up_silu(
                &ffn_gate[..seq_len * ffn_dim],
                &ffn_up[..seq_len * ffn_dim],
                &mut ffn_act[..seq_len * ffn_dim],
            )?;
            cpu_kernels::linear(
                &ffn_act[..seq_len * ffn_dim],
                down_proj,
                down_bias.map(|b| b.as_slice()),
                &mut ffn_out[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                ffn_dim,
            )?;
            cpu_kernels::add(
                &hidden[..seq_len * hidden_size],
                &ffn_out[..seq_len * hidden_size],
                &mut attn_out[..seq_len * hidden_size],
            )?;
            hidden[..seq_len * hidden_size].copy_from_slice(&attn_out[..seq_len * hidden_size]);
        }

        let final_norm_name = Self::resolve_weight_name(weights, &final_norm_candidates())?;
        let final_norm = Self::resolve_weight(weights, &final_norm_name)?;
        cpu_kernels::rms_norm(
            &hidden[..seq_len * hidden_size],
            final_norm,
            &mut norm[..seq_len * hidden_size],
            seq_len,
            hidden_size,
            1e-5,
        )?;

        let last_offset = (seq_len - 1) * hidden_size;
        let last_hidden = norm[last_offset..last_offset + hidden_size].to_vec();

        if let Some(cache) = kv_cache.as_deref_mut() {
            cache.used = kv_start + seq_len;
        }

        Ok(last_hidden)
    }

    fn ensure_logits_buffer(&self, vocab_size: usize) -> BackendResult<usize> {
        let mut logits = self.logits.lock().expect("logits lock poisoned");
        if logits.is_empty() {
            logits.push(vec![0f32; vocab_size]);
        }
        if logits[0].len() != vocab_size {
            return Err(BackendError::InvalidConfig(
                "logits buffer size mismatch".into(),
            ));
        }
        Ok(0)
    }

    fn forward_hidden(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        mut kv_cache: Option<&mut KvCacheEntry>,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<usize> {
        if tokens.is_empty() {
            return Err(BackendError::InvalidConfig("empty tokens".into()));
        }
        if tokens.len() > config.max_seq_len {
            return Err(BackendError::InvalidConfig(
                "tokens exceed max_seq_len".into(),
            ));
        }
        if topology.is_tree() {
            return Err(BackendError::Unimplemented(
                "cpu backend does not support tree topology",
            ));
        }

        let hidden_size = Self::hidden_size(config)?;
        if config.num_kv_heads == 0 || config.num_heads == 0 || config.head_dim == 0 {
            return Err(BackendError::InvalidConfig("invalid head config".into()));
        }

        self.ensure_workspace(weights, config)?;
        self.ensure_qkv_plan(weights, config, hidden_size)?;

        let mut workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
        let workspace = workspace_guard
            .as_mut()
            .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;
        let ws_config = workspace.config;
        if ws_config.hidden_size != hidden_size {
            return Err(BackendError::InvalidConfig("hidden size mismatch".into()));
        }

        let seq_len = tokens.len();
        let mut kv_start = 0usize;
        if let Some(cache) = kv_cache.as_deref_mut() {
            if cache.config.num_layers != config.num_layers
                || cache.config.num_heads != config.num_kv_heads
                || cache.config.head_dim != config.head_dim
            {
                return Err(BackendError::InvalidKvCache(
                    "kv cache config mismatch".into(),
                ));
            }
            if cache.config.dtype_size != std::mem::size_of::<f32>() {
                return Err(BackendError::InvalidKvCache(
                    "cpu kv cache expects f32 dtype".into(),
                ));
            }
            if seq_len > 1 && cache.used != 0 {
                cache.used = 0;
            }
            if seq_len > cache.config.max_seq_len.saturating_sub(cache.used) {
                return Err(BackendError::InvalidKvCache("kv cache exhausted".into()));
            }
            kv_start = cache.used;
        }

        workspace.token_ids[..seq_len].copy_from_slice(tokens);
        for i in 0..seq_len {
            workspace.positions[i] = (kv_start + i) as i32;
        }

        let embed_name = Self::resolve_weight_name(weights, &embedding_candidates())?;
        let embed_weight = Self::resolve_weight(weights, &embed_name)?;
        let embed_shape = weights
            .tensor_shape(&embed_name)
            .ok_or_else(|| BackendError::MissingTensor(embed_name.clone()))?;
        if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
            return Err(BackendError::InvalidConfig(
                "embedding weight shape mismatch".into(),
            ));
        }
        cpu_kernels::embedding_lookup(
            tokens,
            embed_weight,
            &mut workspace.hidden[..seq_len * hidden_size],
            config.vocab_size,
            hidden_size,
        )?;

        let qkv_plan_guard = self.qkv_plan.lock().expect("qkv plan lock poisoned");
        let qkv_plan = qkv_plan_guard
            .as_ref()
            .ok_or_else(|| BackendError::InvalidConfig("qkv plan missing".into()))?;

        let q_out = hidden_size;
        let kv_out = config.num_kv_heads * config.head_dim;
        let q_len = seq_len * q_out;
        let kv_len = seq_len * kv_out;
        let qkv_stride = q_out + 2 * kv_out;
        let positions = &workspace.positions[..seq_len];
        let rotary_dim = if config.position_encoding == PositionEncoding::Rope {
            config.head_dim
        } else {
            0
        };
        let rope_cache =
            if config.position_encoding == PositionEncoding::Rope && config.rope_precompute {
                workspace.rope_cache.as_ref()
            } else {
                None
            };

        for layer in 0..config.num_layers {
            let (attn_norm, _) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(ATTN_NORM_PATTERNS, layer),
            )?;
            let (ffn_norm, _) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(FFN_NORM_PATTERNS, layer),
            )?;

            cpu_kernels::rms_norm(
                &workspace.hidden[..seq_len * hidden_size],
                attn_norm,
                &mut workspace.norm[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                1e-5,
            )?;

            let (qkv_weight, qkv_bias) = match &qkv_plan.layers[layer] {
                QkvLayerPlan::Direct { weight, bias } => {
                    let weight = Self::resolve_weight(weights, weight)?;
                    let bias = bias.as_ref().and_then(|name| weights.tensor_f32(name));
                    (weight.as_slice(), bias.map(|b| b.as_slice()))
                }
                QkvLayerPlan::Fused { weight, bias } => {
                    (weight.as_slice(), bias.as_ref().map(|b| b.as_slice()))
                }
            };

            cpu_kernels::fused_qkv_rope(
                &workspace.norm[..seq_len * hidden_size],
                qkv_weight,
                qkv_bias,
                &mut workspace.qkv[..seq_len * qkv_stride],
                seq_len,
                hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                rotary_dim,
                config.rope_theta,
                config.rope_scale,
                config.rope_interleaved,
                rope_cache,
                positions,
            )?;

            if let Some(cache) = kv_cache.as_deref_mut() {
                let qkv_slice = &workspace.qkv[..seq_len * qkv_stride];
                let k_src = &qkv_slice[q_len..q_len + kv_len];
                let v_src = &qkv_slice[q_len + kv_len..q_len + 2 * kv_len];

                if cache.is_paged() {
                    let page_size = cache.page_size();
                    let pages_per_layer = cache.pages_per_layer();
                    let page_len = page_size * kv_out * 2;
                    let KvCacheStorage::Paged { pages, .. } = &mut cache.storage else {
                        return Err(BackendError::InvalidKvCache(
                            "expected paged kv cache".into(),
                        ));
                    };
                    if pages.len() < (config.num_layers * pages_per_layer) {
                        return Err(BackendError::InvalidKvCache(
                            "paged kv cache size mismatch".into(),
                        ));
                    }

                    let mut remaining = seq_len;
                    let mut src_token = 0usize;
                    let mut global = kv_start;
                    while remaining > 0 {
                        let page = global / page_size;
                        let offset = global - page * page_size;
                        let take = remaining.min(page_size - offset);
                        let page_index = layer * pages_per_layer + page;
                        let page_buf = pages
                            .get_mut(page_index)
                            .ok_or_else(|| BackendError::InvalidKvCache("page index".into()))?;
                        if page_buf.len() != page_len {
                            return Err(BackendError::InvalidKvCache(
                                "paged kv cache page size mismatch".into(),
                            ));
                        }

                        let src_offset = src_token * kv_out;
                        let src_len = take * kv_out;
                        let k_dst = offset * kv_out;
                        let v_dst = page_size * kv_out + offset * kv_out;
                        page_buf[k_dst..k_dst + src_len]
                            .copy_from_slice(&k_src[src_offset..src_offset + src_len]);
                        page_buf[v_dst..v_dst + src_len]
                            .copy_from_slice(&v_src[src_offset..src_offset + src_len]);

                        remaining -= take;
                        src_token += take;
                        global += take;
                    }

                    let kv_seq_len = kv_start + seq_len;
                    let q_view = &workspace.qkv[..q_len];
                    let layer_start = layer * pages_per_layer;
                    let layer_end = layer_start + pages_per_layer;
                    let layer_pages = &pages[layer_start..layer_end];
                    cpu_kernels::flash_attention_paged(
                        q_view,
                        layer_pages,
                        page_size,
                        &mut workspace.attn_out[..seq_len * hidden_size],
                        seq_len,
                        kv_seq_len,
                        config.num_heads,
                        config.num_kv_heads,
                        config.head_dim,
                        true,
                        1.0 / (config.head_dim as f32).sqrt(),
                        workspace.alibi_slopes.as_deref(),
                        kv_start,
                    )?;
                } else {
                    let KvCacheStorage::Contiguous(buffer) = &mut cache.storage else {
                        return Err(BackendError::InvalidKvCache(
                            "expected contiguous kv cache".into(),
                        ));
                    };
                    let layer_stride = cache.config.max_seq_len * kv_out * 2;
                    let layer_base = layer * layer_stride;
                    let k_dst_offset = layer_base + kv_start * kv_out;
                    let v_dst_offset =
                        layer_base + cache.config.max_seq_len * kv_out + kv_start * kv_out;

                    buffer[k_dst_offset..k_dst_offset + kv_len].copy_from_slice(k_src);
                    buffer[v_dst_offset..v_dst_offset + kv_len].copy_from_slice(v_src);

                    let kv_seq_len = kv_start + seq_len;
                    let k_cache_view = &buffer[..kv_seq_len * kv_out];
                    let v_cache_view = &buffer[cache.config.max_seq_len * kv_out
                        ..cache.config.max_seq_len * kv_out + kv_seq_len * kv_out];
                    let q_view = &workspace.qkv[..q_len];

                    cpu_kernels::flash_attention(
                        q_view,
                        k_cache_view,
                        v_cache_view,
                        &mut workspace.attn_out[..seq_len * hidden_size],
                        seq_len,
                        kv_seq_len,
                        config.num_heads,
                        config.num_kv_heads,
                        config.head_dim,
                        true,
                        1.0 / (config.head_dim as f32).sqrt(),
                        workspace.alibi_slopes.as_deref(),
                        kv_start,
                    )?;
                }
            } else {
                let q_view = &workspace.qkv[..q_len];
                let k_view = &workspace.qkv[q_len..q_len + kv_len];
                let v_view = &workspace.qkv[q_len + kv_len..q_len + 2 * kv_len];
                cpu_kernels::flash_attention(
                    q_view,
                    k_view,
                    v_view,
                    &mut workspace.attn_out[..seq_len * hidden_size],
                    seq_len,
                    seq_len,
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    true,
                    1.0 / (config.head_dim as f32).sqrt(),
                    workspace.alibi_slopes.as_deref(),
                    0,
                )?;
            }

            let (o_proj, o_bias) =
                Self::resolve_weight_and_bias(weights, &layer_candidates(O_PROJ_PATTERNS, layer))?;
            cpu_kernels::linear(
                &workspace.attn_out[..seq_len * hidden_size],
                o_proj,
                o_bias.map(|b| b.as_slice()),
                &mut workspace.norm[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                hidden_size,
            )?;
            cpu_kernels::add(
                &workspace.hidden[..seq_len * hidden_size],
                &workspace.norm[..seq_len * hidden_size],
                &mut workspace.attn_out[..seq_len * hidden_size],
            )?;
            workspace.hidden[..seq_len * hidden_size]
                .copy_from_slice(&workspace.attn_out[..seq_len * hidden_size]);

            cpu_kernels::rms_norm(
                &workspace.hidden[..seq_len * hidden_size],
                ffn_norm,
                &mut workspace.norm[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                1e-5,
            )?;

            let (gate_proj, gate_bias) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(GATE_PROJ_PATTERNS, layer),
            )?;
            let (up_proj, up_bias) =
                Self::resolve_weight_and_bias(weights, &layer_candidates(UP_PROJ_PATTERNS, layer))?;
            let (down_proj, down_bias) = Self::resolve_weight_and_bias(
                weights,
                &layer_candidates(DOWN_PROJ_PATTERNS, layer),
            )?;

            let ffn_dim = ws_config.ffn_dim;
            cpu_kernels::linear(
                &workspace.norm[..seq_len * hidden_size],
                gate_proj,
                gate_bias.map(|b| b.as_slice()),
                &mut workspace.ffn_gate[..seq_len * ffn_dim],
                seq_len,
                ffn_dim,
                hidden_size,
            )?;
            cpu_kernels::linear(
                &workspace.norm[..seq_len * hidden_size],
                up_proj,
                up_bias.map(|b| b.as_slice()),
                &mut workspace.ffn_up[..seq_len * ffn_dim],
                seq_len,
                ffn_dim,
                hidden_size,
            )?;
            cpu_kernels::fused_gate_up_silu(
                &workspace.ffn_gate[..seq_len * ffn_dim],
                &workspace.ffn_up[..seq_len * ffn_dim],
                &mut workspace.ffn_act[..seq_len * ffn_dim],
            )?;
            cpu_kernels::linear(
                &workspace.ffn_act[..seq_len * ffn_dim],
                down_proj,
                down_bias.map(|b| b.as_slice()),
                &mut workspace.ffn_out[..seq_len * hidden_size],
                seq_len,
                hidden_size,
                ffn_dim,
            )?;
            cpu_kernels::add(
                &workspace.hidden[..seq_len * hidden_size],
                &workspace.ffn_out[..seq_len * hidden_size],
                &mut workspace.attn_out[..seq_len * hidden_size],
            )?;
            workspace.hidden[..seq_len * hidden_size]
                .copy_from_slice(&workspace.attn_out[..seq_len * hidden_size]);
        }

        let final_norm_name = Self::resolve_weight_name(weights, &final_norm_candidates())?;
        let final_norm = Self::resolve_weight(weights, &final_norm_name)?;
        cpu_kernels::rms_norm(
            &workspace.hidden[..seq_len * hidden_size],
            final_norm,
            &mut workspace.norm[..seq_len * hidden_size],
            seq_len,
            hidden_size,
            1e-5,
        )?;

        let last_offset = (seq_len - 1) * hidden_size;
        workspace
            .last_hidden
            .copy_from_slice(&workspace.norm[last_offset..last_offset + hidden_size]);

        if let Some(cache) = kv_cache.as_deref_mut() {
            cache.used = kv_start + seq_len;
        }

        Ok(hidden_size)
    }

    fn next_random(&self) -> f32 {
        let mut state = self.rng_state.lock().expect("rng lock poisoned");
        let mut x = *state;
        if x == 0 {
            x = 0x9e37_79b9_7f4a_7c15;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        let v = (x as u32).wrapping_add(1) as f32;
        v * 2.3283064e-10
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type Tensor<T> = Vec<T>;

    fn upload_weights<T: DeviceRepr + Clone>(&self, data: &[T]) -> BackendResult<Self::Tensor<T>> {
        Ok(data.to_vec())
    }

    fn alloc_kv_cache(&self, config: &KvCacheConfig) -> BackendResult<KvCacheHandle> {
        if config.dtype_size != std::mem::size_of::<f32>() {
            return Err(BackendError::InvalidKvCache(
                "cpu kv cache expects f32 dtype".into(),
            ));
        }
        let kv_stride = config
            .num_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| BackendError::InvalidKvCache("kv stride overflow".into()))?;
        if kv_stride == 0 || config.max_seq_len == 0 || config.num_layers == 0 {
            return Err(BackendError::InvalidKvCache(
                "kv cache size must be > 0".into(),
            ));
        }
        let page_size = config.effective_page_size();
        let pages_per_layer = config
            .pages_per_layer()
            .ok_or_else(|| BackendError::InvalidKvCache("page size overflow".into()))?;
        let storage = if config.is_paged() {
            let page_len = page_size
                .checked_mul(kv_stride)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| BackendError::InvalidKvCache("page size overflow".into()))?;
            let total_pages = config
                .num_layers
                .checked_mul(pages_per_layer)
                .ok_or_else(|| BackendError::InvalidKvCache("page count overflow".into()))?;
            let mut pages = Vec::with_capacity(total_pages);
            for _ in 0..total_pages {
                pages.push(vec![0f32; page_len]);
            }
            KvCacheStorage::Paged {
                pages,
                page_size,
                pages_per_layer,
            }
        } else {
            let elements = config
                .num_layers
                .checked_mul(2)
                .and_then(|v| v.checked_mul(config.max_seq_len))
                .and_then(|v| v.checked_mul(kv_stride))
                .ok_or_else(|| BackendError::InvalidKvCache("size overflow".into()))?;
            KvCacheStorage::Contiguous(vec![0f32; elements])
        };
        let mut caches = self.kv_caches.lock().expect("kv cache lock poisoned");
        caches.push(KvCacheEntry {
            storage,
            config: *config,
            used: 0,
        });
        Ok(KvCacheHandle::new(caches.len() - 1))
    }

    fn generator_forward_gpu_pure(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        kv_cache: &mut KvCacheHandle,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<LogitsTensor> {
        let mut caches = self.kv_caches.lock().expect("kv cache lock poisoned");
        let cache = caches
            .get_mut(kv_cache.0)
            .ok_or_else(|| BackendError::InvalidHandle("kv cache handle".into()))?;

        if self.use_fast_path(weights, config, tokens.len())? {
            let last_hidden =
                self.forward_hidden_fast(tokens, topology, weights, Some(cache), config)?;
            let hidden_size = last_hidden.len();
            let logits_idx = self.ensure_logits_buffer(config.vocab_size)?;
            let mut logits_guard = self.logits.lock().expect("logits lock poisoned");
            let logits_buf = logits_guard
                .get_mut(logits_idx)
                .ok_or_else(|| BackendError::InvalidHandle("logits handle".into()))?;

            let lm_head_name = Self::resolve_weight_name(weights, &lm_head_candidates())
                .or_else(|_| Self::resolve_weight_name(weights, &embedding_candidates()))?;
            let lm_head = Self::resolve_weight(weights, &lm_head_name)?;
            let lm_bias = Self::resolve_bias(weights, &lm_head_name);

            cpu_kernels::linear(
                &last_hidden,
                lm_head,
                lm_bias.map(|b| b.as_slice()),
                logits_buf,
                1,
                config.vocab_size,
                hidden_size,
            )?;
            return Ok(LogitsTensor::new(logits_idx));
        }

        let hidden_size = self.forward_hidden(tokens, topology, weights, Some(cache), config)?;
        let logits_idx = self.ensure_logits_buffer(config.vocab_size)?;

        let mut logits_guard = self.logits.lock().expect("logits lock poisoned");
        let logits_buf = logits_guard
            .get_mut(logits_idx)
            .ok_or_else(|| BackendError::InvalidHandle("logits handle".into()))?;

        let lm_head_name = Self::resolve_weight_name(weights, &lm_head_candidates())
            .or_else(|_| Self::resolve_weight_name(weights, &embedding_candidates()))?;
        let lm_head = Self::resolve_weight(weights, &lm_head_name)?;
        let lm_bias = Self::resolve_bias(weights, &lm_head_name);

        let workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
        let workspace = workspace_guard
            .as_ref()
            .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;
        cpu_kernels::linear(
            &workspace.last_hidden,
            lm_head,
            lm_bias.map(|b| b.as_slice()),
            logits_buf,
            1,
            config.vocab_size,
            hidden_size,
        )?;

        Ok(LogitsTensor::new(logits_idx))
    }

    fn sample_from_tensor(
        &self,
        logits: &LogitsTensor,
        _topology: &AttentionTopology,
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> BackendResult<Vec<u32>> {
        let logits_guard = self.logits.lock().expect("logits lock poisoned");
        let logits_buf = logits_guard
            .get(logits.0)
            .ok_or_else(|| BackendError::InvalidHandle("logits handle".into()))?;

        if logits_buf.len() != vocab_size {
            return Err(BackendError::InvalidConfig(
                "logits buffer size mismatch".into(),
            ));
        }
        if vocab_size == 0 {
            return Ok(vec![0]);
        }

        let temp = if config.temperature.is_finite() && config.temperature > 0.0 {
            config.temperature
        } else {
            1.0
        };
        let top_p = if config.top_p.is_finite() && config.top_p > 0.0 && config.top_p <= 1.0 {
            config.top_p
        } else {
            1.0
        };
        let mut top_k = config.top_k;
        if top_k == 0 {
            if top_p >= 1.0 {
                let mut best_idx = 0usize;
                let mut best_val = logits_buf[0];
                for (i, &val) in logits_buf.iter().enumerate().skip(1) {
                    if val > best_val {
                        best_val = val;
                        best_idx = i;
                    }
                }
                return Ok(vec![best_idx as u32]);
            }
            top_k = vocab_size;
        }
        if top_k == 1 && top_p >= 1.0 {
            let mut best_idx = 0usize;
            let mut best_val = logits_buf[0];
            for (i, &val) in logits_buf.iter().enumerate().skip(1) {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            return Ok(vec![best_idx as u32]);
        }
        if top_k > vocab_size {
            top_k = vocab_size;
        }

        let mut candidates: Vec<(usize, f32)> = logits_buf
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val / temp))
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if top_k < candidates.len() {
            candidates.truncate(top_k);
        }

        let max_val = candidates[0].1;
        let mut exp_vals = Vec::with_capacity(candidates.len());
        let mut sum = 0.0f32;
        for &(_, val) in &candidates {
            let e = (val - max_val).exp();
            exp_vals.push(e);
            sum += e;
        }
        if sum == 0.0 {
            return Ok(vec![candidates[0].0 as u32]);
        }

        let mut cutoff = candidates.len();
        if top_p < 1.0 {
            let mut acc = 0.0f32;
            for (idx, &e) in exp_vals.iter().enumerate() {
                acc += e;
                if acc / sum >= top_p {
                    cutoff = idx + 1;
                    break;
                }
            }
        }

        let mut subset_sum = 0.0f32;
        for &e in exp_vals.iter().take(cutoff) {
            subset_sum += e;
        }
        if subset_sum == 0.0 {
            return Ok(vec![candidates[0].0 as u32]);
        }
        let r = self.next_random() * subset_sum;
        let mut acc = 0.0f32;
        let mut chosen = candidates[0].0;
        for i in 0..cutoff {
            acc += exp_vals[i];
            if r <= acc {
                chosen = candidates[i].0;
                break;
            }
        }
        Ok(vec![chosen as u32])
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<f32>> {
        if self.use_fast_path(weights, config, tokens.len())? {
            return self.forward_hidden_fast(tokens, topology, weights, None, config);
        }
        let hidden_size = self.forward_hidden(tokens, topology, weights, None, config)?;
        let workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
        let workspace = workspace_guard
            .as_ref()
            .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;
        Ok(workspace.last_hidden[..hidden_size].to_vec())
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<f32>> {
        if self.use_fast_path(weights, config, tokens.len())? {
            let last_hidden = self.forward_hidden_fast(tokens, topology, weights, None, config)?;
            let hidden_size = last_hidden.len();

            let score_name = find_tensor_name(weights, &score_head_candidates());
            if let Some(score_name) = score_name {
                let score_weight = Self::resolve_weight(weights, &score_name)?;
                let score_shape = weights
                    .tensor_shape(&score_name)
                    .ok_or_else(|| BackendError::MissingTensor(score_name.clone()))?;
                let out_dim = Self::deduce_out_dim(score_shape, hidden_size)?;
                let score_bias = Self::resolve_bias(weights, &score_name);

                let mut output = vec![0f32; out_dim];
                cpu_kernels::linear(
                    &last_hidden,
                    score_weight,
                    score_bias.map(|b| b.as_slice()),
                    &mut output,
                    1,
                    out_dim,
                    hidden_size,
                )?;
                return Ok(output);
            }
            return Ok(last_hidden);
        }

        let hidden_size = self.forward_hidden(tokens, topology, weights, None, config)?;

        let score_name = find_tensor_name(weights, &score_head_candidates());
        if let Some(score_name) = score_name {
            let score_weight = Self::resolve_weight(weights, &score_name)?;
            let score_shape = weights
                .tensor_shape(&score_name)
                .ok_or_else(|| BackendError::MissingTensor(score_name.clone()))?;
            let out_dim = Self::deduce_out_dim(score_shape, hidden_size)?;
            let score_bias = Self::resolve_bias(weights, &score_name);

            let workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
            let workspace = workspace_guard
                .as_ref()
                .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;
            let mut output = vec![0f32; out_dim];
            cpu_kernels::linear(
                &workspace.last_hidden,
                score_weight,
                score_bias.map(|b| b.as_slice()),
                &mut output,
                1,
                out_dim,
                hidden_size,
            )?;
            Ok(output)
        } else {
            let workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
            let workspace = workspace_guard
                .as_ref()
                .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;
            Ok(workspace.last_hidden[..hidden_size].to_vec())
        }
    }
}
