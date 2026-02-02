use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cudarc::driver::{
    result, sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
};

use crate::backend_trait::{
    AttentionTopology, Backend, BackendError, BackendResult, BatchInput, KvCacheHandle,
    LogitsTensor, TensorLookup,
};
use crate::cuda_kernels::{
    cubin_for, AddConfig, CudaKernels, EmbeddingConfig, FlashAttnConfig,
    FlashAttnPagedConfig, FusedQkvRopeConfig, LinearConfig, QuantizedBits, QuantizedConfig,
    RmsNormConfig, RopeConfig, SamplingKernelConfig, SiluConfig, SmVersion, SwiGluConfig,
};
use crate::gpu_types::GpuBuffer;
use crate::kernel_types::{
    alibi_slopes, precompute_rope_tables, GeneratorForwardConfig, KvCacheConfig, PositionEncoding,
    SamplingConfig,
};
use crate::swap_manager::SwapManager;

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

#[derive(Debug, Clone, Copy)]
pub struct PerfMetrics {
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub tokens_processed: u64,
    pub elapsed: Duration,
    pub last_forward: Duration,
    pub last_sample: Duration,
}

impl PerfMetrics {
    pub fn tps(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs > 0.0 {
            self.tokens_processed as f64 / secs
        } else {
            0.0
        }
    }
}

struct PerfState {
    start: Instant,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    tokens_processed: u64,
    last_forward: Duration,
    last_sample: Duration,
}

impl PerfState {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            host_to_device_bytes: 0,
            device_to_host_bytes: 0,
            tokens_processed: 0,
            last_forward: Duration::from_secs(0),
            last_sample: Duration::from_secs(0),
        }
    }

    fn snapshot(&self) -> PerfMetrics {
        PerfMetrics {
            host_to_device_bytes: self.host_to_device_bytes,
            device_to_host_bytes: self.device_to_host_bytes,
            tokens_processed: self.tokens_processed,
            elapsed: self.start.elapsed(),
            last_forward: self.last_forward,
            last_sample: self.last_sample,
        }
    }

    fn reset(&mut self) {
        self.start = Instant::now();
        self.host_to_device_bytes = 0;
        self.device_to_host_bytes = 0;
        self.tokens_processed = 0;
        self.last_forward = Duration::from_secs(0);
        self.last_sample = Duration::from_secs(0);
    }
}

struct LayerWorkspace {
    config: WorkspaceConfig,
    token_ids: CudaSlice<u32>,
    linear_positions: CudaSlice<i32>,
    positions: GpuBuffer<i32>,
    hidden: GpuBuffer<f32>,
    norm: GpuBuffer<f32>,
    qkv: GpuBuffer<f32>,
    attn_out: GpuBuffer<f32>,
    ffn_gate: GpuBuffer<f32>,
    ffn_up: GpuBuffer<f32>,
    ffn_act: GpuBuffer<f32>,
    ffn_out: GpuBuffer<f32>,
    last_hidden: GpuBuffer<f32>,
    rope_cos: Option<GpuBuffer<f32>>,
    rope_sin: Option<GpuBuffer<f32>>,
    alibi_slopes: Option<GpuBuffer<f32>>,
}

impl LayerWorkspace {
    fn allocate(stream: &Arc<CudaStream>, config: WorkspaceConfig) -> BackendResult<Self> {
        let hidden_len = config
            .max_seq_len
            .checked_mul(config.hidden_size)
            .ok_or_else(|| BackendError::InvalidConfig("hidden buffer size overflow".into()))?;
        let qkv_dim = config
            .num_heads
            .checked_mul(config.head_dim)
            .and_then(|q| q.checked_add(2 * config.num_kv_heads * config.head_dim))
            .ok_or_else(|| BackendError::InvalidConfig("qkv dimension overflow".into()))?;
        let qkv_len = config
            .max_seq_len
            .checked_mul(qkv_dim)
            .ok_or_else(|| BackendError::InvalidConfig("qkv buffer size overflow".into()))?;
        let ffn_len = config
            .max_seq_len
            .checked_mul(config.ffn_dim)
            .ok_or_else(|| BackendError::InvalidConfig("ffn buffer size overflow".into()))?;

        let token_ids = stream.alloc_zeros::<u32>(config.max_seq_len)?;
        let linear_positions = if config.max_seq_len == 0 {
            stream.alloc_zeros::<i32>(0)?
        } else {
            if config.max_seq_len > i32::MAX as usize {
                return Err(BackendError::InvalidConfig(
                    "max_seq_len exceeds i32 range".into(),
                ));
            }
            let mut host_positions = Vec::with_capacity(config.max_seq_len);
            for idx in 0..config.max_seq_len {
                host_positions.push(idx as i32);
            }
            stream.clone_htod(&host_positions)?
        };
        let positions = GpuBuffer::new(stream.alloc_zeros::<i32>(config.max_seq_len)?);
        let hidden = GpuBuffer::new(stream.alloc_zeros::<f32>(hidden_len)?);
        let norm = GpuBuffer::new(stream.alloc_zeros::<f32>(hidden_len)?);
        let qkv = GpuBuffer::new(stream.alloc_zeros::<f32>(qkv_len)?);
        let attn_out = GpuBuffer::new(stream.alloc_zeros::<f32>(hidden_len)?);
        let ffn_gate = GpuBuffer::new(stream.alloc_zeros::<f32>(ffn_len)?);
        let ffn_up = GpuBuffer::new(stream.alloc_zeros::<f32>(ffn_len)?);
        let ffn_act = GpuBuffer::new(stream.alloc_zeros::<f32>(ffn_len)?);
        let ffn_out = GpuBuffer::new(stream.alloc_zeros::<f32>(hidden_len)?);
        let last_hidden = GpuBuffer::new(stream.alloc_zeros::<f32>(config.hidden_size)?);
        let rope_cos = if config.position_encoding == PositionEncoding::Rope
            && config.rope_precompute
            && config.head_dim / 2 > 0
        {
            let (cos, sin) = precompute_rope_tables(
                config.max_seq_len,
                config.head_dim,
                config.rope_theta,
                config.rope_scale,
            );
            let cos_dev = GpuBuffer::new(stream.clone_htod(&cos)?);
            let sin_dev = GpuBuffer::new(stream.clone_htod(&sin)?);
            Some((cos_dev, sin_dev))
        } else {
            None
        };
        let (rope_cos, rope_sin) = match rope_cos {
            Some((cos, sin)) => (Some(cos), Some(sin)),
            None => (None, None),
        };
        let alibi_slopes = if config.position_encoding == PositionEncoding::Alibi {
            let slopes = alibi_slopes(config.num_heads);
            if slopes.is_empty() {
                None
            } else {
                Some(GpuBuffer::new(stream.clone_htod(&slopes)?))
            }
        } else {
            None
        };

        Ok(Self {
            config,
            token_ids,
            linear_positions,
            positions,
            hidden,
            norm,
            qkv,
            attn_out,
            ffn_gate,
            ffn_up,
            ffn_act,
            ffn_out,
            last_hidden,
            rope_cos,
            rope_sin,
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
    Contiguous(CudaSlice<f32>),
    Paged {
        pages: Vec<CudaSlice<f32>>,
        page_ptrs: CudaSlice<u64>,
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
        weight: GpuBuffer<f32>,
        bias: Option<GpuBuffer<f32>>,
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
    weights: &dyn TensorLookup<CudaBackend>,
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

pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: CudaModuleHandle,
    sm: SmVersion,
    kernels: CudaKernels,
    kv_caches: Mutex<Vec<KvCacheEntry>>,
    swap_managers: Mutex<Vec<Option<SwapManager>>>,
    workspace: Mutex<Option<LayerWorkspace>>,
    qkv_plan: Mutex<Option<QkvPlan>>,
    logits: Mutex<Vec<CudaSlice<f32>>>,
    sampling: Mutex<Option<CudaSlice<u32>>>,
    perf: Mutex<PerfState>,
}

struct CudaModuleHandle {
    module: sys::CUmodule,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaModuleHandle {}
unsafe impl Sync for CudaModuleHandle {}

impl Drop for CudaModuleHandle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = result::module::unload(self.module);
        }
    }
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
            .field("sm", &self.sm)
            .finish()
    }
}

impl CudaBackend {
    pub fn new(device_ordinal: usize) -> BackendResult<Self> {
        let device_count = CudaContext::device_count()?;
        if device_count <= 0 {
            return Err(BackendError::InvalidConfig(
                "no CUDA devices detected".into(),
            ));
        }
        if device_ordinal >= device_count as usize {
            return Err(BackendError::InvalidConfig(format!(
                "cuda device ordinal {device_ordinal} out of range (device_count={device_count})",
            )));
        }

        let ctx = CudaContext::new(device_ordinal)?;
        let (major, minor) = ctx.compute_capability()?;
        let sm = SmVersion::from_compute_capability(major, minor)
            .ok_or(BackendError::UnsupportedSm { major, minor })?;
        let cubin = cubin_for(sm);
        if cubin.is_empty() {
            return Err(BackendError::InvalidCubin(sm.as_str()));
        }
        ctx.bind_to_thread()?;
        let module = unsafe { result::module::load_data(cubin.as_ptr() as *const _) }?;
        let kernels = CudaKernels::load(module)?;
        let stream = ctx.default_stream();
        let module_handle = CudaModuleHandle {
            module,
            ctx: ctx.clone(),
        };
        Ok(Self {
            ctx,
            stream,
            _module: module_handle,
            sm,
            kernels,
            kv_caches: Mutex::new(Vec::new()),
            swap_managers: Mutex::new(Vec::new()),
            workspace: Mutex::new(None),
            qkv_plan: Mutex::new(None),
            logits: Mutex::new(Vec::new()),
            sampling: Mutex::new(None),
            perf: Mutex::new(PerfState::new()),
        })
    }

    pub fn device_count() -> BackendResult<i32> {
        Ok(CudaContext::device_count()?)
    }

    pub fn sm_version(&self) -> SmVersion {
        self.sm
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn kernels(&self) -> &CudaKernels {
        &self.kernels
    }

    pub fn perf_metrics(&self) -> PerfMetrics {
        let perf = self.perf.lock().expect("perf lock poisoned");
        perf.snapshot()
    }

    pub fn reset_perf_metrics(&self) {
        let mut perf = self.perf.lock().expect("perf lock poisoned");
        perf.reset();
    }

    fn record_htod(&self, bytes: usize) {
        let mut perf = self.perf.lock().expect("perf lock poisoned");
        perf.host_to_device_bytes = perf.host_to_device_bytes.saturating_add(bytes as u64);
    }

    fn record_dtoh(&self, bytes: usize) {
        let mut perf = self.perf.lock().expect("perf lock poisoned");
        perf.device_to_host_bytes = perf.device_to_host_bytes.saturating_add(bytes as u64);
    }

    fn record_tokens(&self, tokens: usize) {
        let mut perf = self.perf.lock().expect("perf lock poisoned");
        perf.tokens_processed = perf.tokens_processed.saturating_add(tokens as u64);
    }

    fn record_forward_time(&self, dur: Duration) {
        let mut perf = self.perf.lock().expect("perf lock poisoned");
        perf.last_forward = dur;
    }

    fn record_sample_time(&self, dur: Duration) {
        let mut perf = self.perf.lock().expect("perf lock poisoned");
        perf.last_sample = dur;
    }

    pub fn flash_attn<T: DeviceRepr>(
        &self,
        launch: LaunchConfig,
        params: &FlashAttnConfig,
        q: &GpuBuffer<T>,
        k: &GpuBuffer<T>,
        v: &GpuBuffer<T>,
        output: &mut GpuBuffer<T>,
    ) -> BackendResult<()> {
        self.kernels.flash_attn.launch(
            self.stream.as_ref(),
            launch,
            q.as_inner(),
            k.as_inner(),
            v.as_inner(),
            output.as_inner_mut(),
            None,
            params,
        )
    }

    pub fn rope<T: DeviceRepr>(
        &self,
        launch: LaunchConfig,
        params: &RopeConfig,
        q: &mut GpuBuffer<T>,
        k: &mut GpuBuffer<T>,
        positions: Option<&GpuBuffer<i32>>,
        cos_table: Option<&GpuBuffer<f32>>,
        sin_table: Option<&GpuBuffer<f32>>,
    ) -> BackendResult<()> {
        let positions = positions.map(GpuBuffer::as_inner);
        let cos_table = cos_table.map(GpuBuffer::as_inner);
        let sin_table = sin_table.map(GpuBuffer::as_inner);
        self.kernels.rope.launch(
            self.stream.as_ref(),
            launch,
            q.as_inner_mut(),
            k.as_inner_mut(),
            positions,
            cos_table,
            sin_table,
            params,
        )
    }

    pub fn fused_qkv_rope<T: DeviceRepr>(
        &self,
        launch: LaunchConfig,
        params: &FusedQkvRopeConfig,
        input: &GpuBuffer<T>,
        qkv_weight: &GpuBuffer<T>,
        qkv_bias: Option<&GpuBuffer<T>>,
        qkv_out: &mut GpuBuffer<T>,
        positions: Option<&GpuBuffer<i32>>,
        cos_table: Option<&GpuBuffer<f32>>,
        sin_table: Option<&GpuBuffer<f32>>,
    ) -> BackendResult<()> {
        let qkv_bias = qkv_bias.map(GpuBuffer::as_inner);
        let positions = positions.map(GpuBuffer::as_inner);
        let cos_table = cos_table.map(GpuBuffer::as_inner);
        let sin_table = sin_table.map(GpuBuffer::as_inner);
        self.kernels.fused_qkv_rope.launch(
            self.stream.as_ref(),
            launch,
            input.as_inner(),
            qkv_weight.as_inner(),
            qkv_bias,
            qkv_out.as_inner_mut(),
            positions,
            cos_table,
            sin_table,
            params,
        )
    }

    pub fn rms_norm<T: DeviceRepr>(
        &self,
        launch: LaunchConfig,
        params: &RmsNormConfig,
        input: &GpuBuffer<T>,
        weight: &GpuBuffer<T>,
        output: &mut GpuBuffer<T>,
    ) -> BackendResult<()> {
        self.kernels.rms_norm.launch(
            self.stream.as_ref(),
            launch,
            input.as_inner(),
            weight.as_inner(),
            output.as_inner_mut(),
            params,
        )
    }

    pub fn silu<T: DeviceRepr>(
        &self,
        launch: LaunchConfig,
        params: &SiluConfig,
        input: &GpuBuffer<T>,
        output: &mut GpuBuffer<T>,
    ) -> BackendResult<()> {
        self.kernels.silu.launch(
            self.stream.as_ref(),
            launch,
            input.as_inner(),
            output.as_inner_mut(),
            params,
        )
    }

    pub fn swiglu<T: DeviceRepr>(
        &self,
        launch: LaunchConfig,
        params: &SwiGluConfig,
        gate: &GpuBuffer<T>,
        up: &GpuBuffer<T>,
        output: &mut GpuBuffer<T>,
    ) -> BackendResult<()> {
        self.kernels.swiglu.launch(
            self.stream.as_ref(),
            launch,
            gate.as_inner(),
            up.as_inner(),
            output.as_inner_mut(),
            params,
        )
    }

    pub fn fused_gate_up_silu<T: DeviceRepr>(
        &self,
        launch: LaunchConfig,
        params: &SwiGluConfig,
        gate: &GpuBuffer<T>,
        up: &GpuBuffer<T>,
        output: &mut GpuBuffer<T>,
    ) -> BackendResult<()> {
        self.swiglu(launch, params, gate, up, output)
    }

    pub fn quantized_mm<W: DeviceRepr>(
        &self,
        bits: QuantizedBits,
        launch: LaunchConfig,
        params: &QuantizedConfig,
        input: &GpuBuffer<f32>,
        weight: &GpuBuffer<W>,
        output: &mut GpuBuffer<f32>,
    ) -> BackendResult<()> {
        let kernel = match bits {
            QuantizedBits::Int1 => self
                .kernels
                .quantized_mm
                .int1
                .ok_or(BackendError::Unimplemented("quantized_mm<1> kernel"))?,
            QuantizedBits::Int2 => self
                .kernels
                .quantized_mm
                .int2
                .ok_or(BackendError::Unimplemented("quantized_mm<2> kernel"))?,
            QuantizedBits::Int4 => self
                .kernels
                .quantized_mm
                .int4
                .ok_or(BackendError::Unimplemented("quantized_mm<4> kernel"))?,
            QuantizedBits::Int8 => self
                .kernels
                .quantized_mm
                .int8
                .ok_or(BackendError::Unimplemented("quantized_mm<8> kernel"))?,
        };
        kernel.launch(
            self.stream.as_ref(),
            launch,
            input.as_inner(),
            weight.as_inner(),
            output.as_inner_mut(),
            params,
        )
    }

    pub fn int8_mm(
        &self,
        launch: LaunchConfig,
        params: &QuantizedConfig,
        input: &GpuBuffer<f32>,
        weight: &GpuBuffer<i8>,
        output: &mut GpuBuffer<f32>,
    ) -> BackendResult<()> {
        self.quantized_mm(
            QuantizedBits::Int8,
            launch,
            params,
            input,
            weight,
            output,
        )
    }

    pub fn int4_mm(
        &self,
        launch: LaunchConfig,
        params: &QuantizedConfig,
        input: &GpuBuffer<f32>,
        weight: &GpuBuffer<u8>,
        output: &mut GpuBuffer<f32>,
    ) -> BackendResult<()> {
        self.quantized_mm(
            QuantizedBits::Int4,
            launch,
            params,
            input,
            weight,
            output,
        )
    }

    pub fn int2_mm(
        &self,
        launch: LaunchConfig,
        params: &QuantizedConfig,
        input: &GpuBuffer<f32>,
        weight: &GpuBuffer<u8>,
        output: &mut GpuBuffer<f32>,
    ) -> BackendResult<()> {
        self.quantized_mm(
            QuantizedBits::Int2,
            launch,
            params,
            input,
            weight,
            output,
        )
    }

    pub fn int1_mm(
        &self,
        launch: LaunchConfig,
        params: &QuantizedConfig,
        input: &GpuBuffer<f32>,
        weight: &GpuBuffer<u8>,
        output: &mut GpuBuffer<f32>,
    ) -> BackendResult<()> {
        self.quantized_mm(
            QuantizedBits::Int1,
            launch,
            params,
            input,
            weight,
            output,
        )
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
    ) -> BackendResult<&'a GpuBuffer<f32>> {
        weights
            .tensor_f32(name)
            .ok_or_else(|| BackendError::MissingTensor(name.to_string()))
    }

    fn resolve_bias<'a>(
        weights: &'a dyn TensorLookup<Self>,
        weight_name: &str,
    ) -> Option<&'a GpuBuffer<f32>> {
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
    ) -> BackendResult<(&'a GpuBuffer<f32>, Option<&'a GpuBuffer<f32>>)> {
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

        *workspace = Some(LayerWorkspace::allocate(&self.stream, ws_config)?);
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

            let in_dim = hidden_size;
            let fused_out = q_out + k_out + v_out;
            let fused_len = fused_out
                .checked_mul(in_dim)
                .ok_or_else(|| BackendError::InvalidConfig("fused qkv size overflow".into()))?;
            let mut fused = GpuBuffer::new(self.stream.alloc_zeros::<f32>(fused_len)?);

            let q_len = q_out * in_dim;
            let k_len = k_out * in_dim;
            let v_len = v_out * in_dim;

            let mut q_dst = fused.as_inner_mut().slice_mut(0..q_len);
            self.stream.memcpy_dtod(q_weight.as_inner(), &mut q_dst)?;
            let mut k_dst = fused.as_inner_mut().slice_mut(q_len..q_len + k_len);
            self.stream.memcpy_dtod(k_weight.as_inner(), &mut k_dst)?;
            let mut v_dst = fused
                .as_inner_mut()
                .slice_mut(q_len + k_len..q_len + k_len + v_len);
            self.stream.memcpy_dtod(v_weight.as_inner(), &mut v_dst)?;

            let bias_present = Self::resolve_bias(weights, &q_name)
                .or_else(|| Self::resolve_bias(weights, &k_name))
                .or_else(|| Self::resolve_bias(weights, &v_name))
                .is_some();

            let fused_bias = if bias_present {
                let mut bias_buf = GpuBuffer::new(self.stream.alloc_zeros::<f32>(fused_out)?);
                if let Some(q_bias) = Self::resolve_bias(weights, &q_name) {
                    let mut dst = bias_buf.as_inner_mut().slice_mut(0..q_out);
                    self.stream.memcpy_dtod(q_bias.as_inner(), &mut dst)?;
                }
                if let Some(k_bias) = Self::resolve_bias(weights, &k_name) {
                    let mut dst = bias_buf.as_inner_mut().slice_mut(q_out..q_out + k_out);
                    self.stream.memcpy_dtod(k_bias.as_inner(), &mut dst)?;
                }
                if let Some(v_bias) = Self::resolve_bias(weights, &v_name) {
                    let mut dst = bias_buf
                        .as_inner_mut()
                        .slice_mut(q_out + k_out..q_out + k_out + v_out);
                    self.stream.memcpy_dtod(v_bias.as_inner(), &mut dst)?;
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

    fn ensure_logits_buffer(&self, vocab_size: usize) -> BackendResult<usize> {
        self.ensure_logits_buffer_at(vocab_size, 0)?;
        Ok(0)
    }

    fn ensure_logits_buffer_at(&self, vocab_size: usize, index: usize) -> BackendResult<()> {
        let mut logits = self.logits.lock().expect("logits lock poisoned");
        while logits.len() <= index {
            logits.push(self.stream.alloc_zeros::<f32>(vocab_size)?);
        }
        if logits[index].len() != vocab_size {
            return Err(BackendError::InvalidConfig(
                "logits buffer size mismatch".into(),
            ));
        }
        Ok(())
    }

    fn validate_sequence_position(
        cache: &KvCacheEntry,
        seq_len: usize,
        position: usize,
    ) -> BackendResult<()> {
        if position > cache.config.max_seq_len {
            return Err(BackendError::InvalidConfig(
                "sequence position exceeds max_seq_len".into(),
            ));
        }
        if seq_len > 1 {
            if position != 0 {
                return Err(BackendError::InvalidConfig(
                    "prefill position must be 0".into(),
                ));
            }
            if cache.used != 0 {
                return Err(BackendError::InvalidKvCache(
                    "kv cache not empty for prefill".into(),
                ));
            }
        } else if position != cache.used {
            return Err(BackendError::InvalidKvCache(
                "sequence position mismatch".into(),
            ));
        }
        Ok(())
    }

    fn swap_out_pages_with_manager(
        &self,
        cache: &mut KvCacheEntry,
        manager: &mut SwapManager,
        page_indices: &[usize],
    ) -> BackendResult<()> {
        let KvCacheStorage::Paged { pages, .. } = &mut cache.storage else {
            return Err(BackendError::InvalidKvCache("cache is not paged".into()));
        };

        let prepared = manager.prepare_swap_out(page_indices)?;
        for &page_idx in &prepared {
            if page_idx >= pages.len() {
                continue;
            }
            let gpu_page = &pages[page_idx];
            let mut host_data = vec![0.0f32; gpu_page.len()];
            let gpu_src = gpu_page.slice(0..gpu_page.len());
            self.stream.memcpy_dtoh(&gpu_src, &mut host_data)?;
            manager.complete_swap_out(page_idx, host_data)?;
        }
        Ok(())
    }

    fn auto_swap_out_if_needed(
        &self,
        cache: &mut KvCacheEntry,
        manager: &mut SwapManager,
    ) -> BackendResult<()> {
        let Some(cfg) = cache.config.swap_config else {
            return Ok(());
        };
        if !cfg.enable_swap || !cache.is_paged() {
            return Ok(());
        }
        let pressure = self.get_memory_pressure()?;
        if !pressure.is_finite() {
            return Err(BackendError::InvalidConfig(
                "memory pressure is not finite".into(),
            ));
        }
        if pressure < cfg.swap_threshold {
            return Ok(());
        }
        let count = cfg.lru_granularity.max(1);
        let victims = manager.select_victim_pages(count);
        let prepared = manager.prepare_swap_out(&victims)?;
        if prepared.is_empty() {
            return Ok(());
        }
        self.swap_out_pages_with_manager(cache, manager, &prepared)?;
        Ok(())
    }

    fn mark_kv_pages_accessed(
        cache: &KvCacheEntry,
        manager: &mut SwapManager,
    ) -> BackendResult<()> {
        if !cache.is_paged() {
            return Ok(());
        }
        let kv_end = cache.used;
        if kv_end == 0 {
            return Ok(());
        }
        let page_size = cache.page_size();
        if page_size == 0 {
            return Ok(());
        }
        let pages_per_layer = cache.pages_per_layer();
        let pages_used = (kv_end + page_size - 1) / page_size;
        for layer in 0..cache.config.num_layers {
            let base = layer * pages_per_layer;
            for page in 0..pages_used {
                manager.mark_accessed(base + page)?;
            }
        }
        Ok(())
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
        let is_tree = topology.is_tree();

        let hidden_size = Self::hidden_size(config)?;
        if config.num_kv_heads == 0 || config.num_heads == 0 || config.head_dim == 0 {
            return Err(BackendError::InvalidConfig("invalid head config".into()));
        }

        let embedding_kernel = &self.kernels.embedding;
        let linear_kernel = &self.kernels.linear;
        let add_kernel = &self.kernels.add;

        self.ensure_workspace(weights, config)?;
        self.ensure_qkv_plan(weights, config, hidden_size)?;

        let mut workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
        let workspace = workspace_guard
            .as_mut()
            .expect("workspace should be allocated");
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
            if seq_len > 1 && cache.used != 0 {
                cache.used = 0;
            }
            if seq_len > cache.config.max_seq_len.saturating_sub(cache.used) {
                return Err(BackendError::InvalidKvCache("kv cache exhausted".into()));
            }
            kv_start = cache.used;
        }
        if kv_cache.is_some() {
            self.record_tokens(seq_len);
            self.record_htod(seq_len * std::mem::size_of::<u32>());
        }
        let mut token_view = workspace.token_ids.slice_mut(0..seq_len);
        self.stream.memcpy_htod(tokens, &mut token_view)?;

        if let Some(tree) = topology.tree_structure.as_ref() {
            if tree.len() < seq_len {
                return Err(BackendError::InvalidConfig(
                    "tree topology length mismatch".into(),
                ));
            }
            let src = tree.slice(0..seq_len);
            let mut dst = workspace.positions.as_inner_mut().slice_mut(0..seq_len);
            self.stream.memcpy_dtod(&src, &mut dst)?;
        } else {
            let end = kv_start
                .checked_add(seq_len)
                .ok_or_else(|| BackendError::InvalidConfig("positions range overflow".into()))?;
            if end > workspace.config.max_seq_len {
                return Err(BackendError::InvalidConfig(
                    "positions range exceeds workspace".into(),
                ));
            }
            let src = workspace.linear_positions.slice(kv_start..end);
            let mut dst = workspace.positions.as_inner_mut().slice_mut(0..seq_len);
            self.stream.memcpy_dtod(&src, &mut dst)?;
        }

        let embed_weight = Self::resolve_weight(
            weights,
            &Self::resolve_weight_name(weights, &embedding_candidates())?,
        )?;
        let emb_cfg = EmbeddingConfig {
            vocab_size: config.vocab_size as u32,
            hidden_size: hidden_size as u32,
            seq_len: seq_len as u32,
            stride: hidden_size as u32,
        };
        let emb_launch = LaunchConfig::for_num_elems((seq_len * hidden_size) as u32);
        embedding_kernel.launch(
            self.stream.as_ref(),
            emb_launch,
            &workspace.token_ids,
            embed_weight.as_inner(),
            workspace.hidden.as_inner_mut(),
            &emb_cfg,
        )?;

        let qkv_plan_guard = self.qkv_plan.lock().expect("qkv plan lock poisoned");
        let qkv_plan = qkv_plan_guard
            .as_ref()
            .expect("qkv plan should be allocated");

        let q_out = hidden_size;
        let kv_out = config.num_kv_heads * config.head_dim;
        let q_len = seq_len * q_out;
        let kv_len = seq_len * kv_out;
        let qkv_stride = q_out + 2 * kv_out;
        let rotary_dim = if config.position_encoding == PositionEncoding::Rope {
            config.head_dim
        } else {
            0
        };
        let rope_cos =
            if config.position_encoding == PositionEncoding::Rope && config.rope_precompute {
                workspace.rope_cos.as_ref()
            } else {
                None
            };
        let rope_sin =
            if config.position_encoding == PositionEncoding::Rope && config.rope_precompute {
                workspace.rope_sin.as_ref()
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

            let rms_cfg = RmsNormConfig {
                hidden_size: hidden_size as u32,
                stride: hidden_size as u32,
                eps: 1e-5,
                seq_len: seq_len as u32,
            };
            let rms_launch = LaunchConfig::for_num_elems(seq_len as u32);
            self.rms_norm(
                rms_launch,
                &rms_cfg,
                &workspace.hidden,
                attn_norm,
                &mut workspace.norm,
            )?;

            let (qkv_weight, qkv_bias) = match &qkv_plan.layers[layer] {
                QkvLayerPlan::Direct { weight, bias } => {
                    let weight = Self::resolve_weight(weights, weight)?;
                    let bias = bias.as_ref().and_then(|name| weights.tensor_f32(name));
                    (weight, bias)
                }
                QkvLayerPlan::Fused { weight, bias } => (weight, bias.as_ref()),
            };

            let qkv_cfg = FusedQkvRopeConfig {
                batch: 1,
                seq_len: seq_len as u32,
                num_heads: config.num_heads as u32,
                head_dim: config.head_dim as u32,
                rotary_dim: rotary_dim as u32,
                input_stride: hidden_size as u32,
                qkv_stride: qkv_stride as u32,
                base: config.rope_theta,
                scale: config.rope_scale,
                interleaved: if config.rope_interleaved { 1 } else { 0 },
                precompute_max_seq_len: config.max_seq_len as u32,
            };
            let qkv_launch = LaunchConfig::for_num_elems((seq_len * hidden_size) as u32);
            self.fused_qkv_rope(
                qkv_launch,
                &qkv_cfg,
                &workspace.norm,
                qkv_weight,
                qkv_bias,
                &mut workspace.qkv,
                Some(&workspace.positions),
                rope_cos,
                rope_sin,
            )?;

            if let Some(cache) = kv_cache.as_deref_mut() {
                let qkv_slice = workspace.qkv.as_inner();
                let k_src = qkv_slice.slice(q_len..q_len + kv_len);
                let v_src = qkv_slice.slice(q_len + kv_len..q_len + 2 * kv_len);

                if cache.is_paged() {
                    let page_size = cache.page_size();
                    let pages_per_layer = cache.pages_per_layer();
                    let page_len = page_size * kv_out * 2;
                    let KvCacheStorage::Paged {
                        pages, page_ptrs, ..
                    } = &mut cache.storage
                    else {
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
                        let k_seg = k_src.slice(src_offset..src_offset + src_len);
                        let v_seg = v_src.slice(src_offset..src_offset + src_len);
                        {
                            let mut k_dst_view = page_buf.slice_mut(k_dst..k_dst + src_len);
                            self.stream.memcpy_dtod(&k_seg, &mut k_dst_view)?;
                        }
                        {
                            let mut v_dst_view = page_buf.slice_mut(v_dst..v_dst + src_len);
                            self.stream.memcpy_dtod(&v_seg, &mut v_dst_view)?;
                        }

                        remaining -= take;
                        src_token += take;
                        global += take;
                    }

                    let kv_seq_len = kv_start + seq_len;
                    let q_view = workspace.qkv.as_inner().slice(0..q_len);
                    let table_start = layer * pages_per_layer;
                    let table_end = table_start + pages_per_layer;
                    let page_table = page_ptrs.slice(table_start..table_end);
                    let attn_cfg = FlashAttnPagedConfig {
                        batch: 1,
                        num_heads: config.num_heads as u32,
                        head_dim: config.head_dim as u32,
                        q_seq_len: seq_len as u32,
                        kv_seq_len: kv_seq_len as u32,
                        q_stride: q_out as u32,
                        kv_stride: kv_out as u32,
                        o_stride: q_out as u32,
                        causal: if is_tree { 0 } else { 1 },
                        scale: 1.0 / (config.head_dim as f32).sqrt(),
                        q_pos_offset: kv_start as u32,
                        page_size: page_size as u32,
                        pages_per_layer: pages_per_layer as u32,
                    };
                    let attn_launch =
                        LaunchConfig::for_num_elems((seq_len * config.num_heads) as u32);
                    self.kernels.flash_attn_paged.launch(
                        self.stream.as_ref(),
                        attn_launch,
                        &q_view,
                        &page_table,
                        workspace.attn_out.as_inner_mut(),
                        workspace.alibi_slopes.as_ref().map(GpuBuffer::as_inner),
                        &attn_cfg,
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

                    {
                        let mut k_dst = buffer.slice_mut(k_dst_offset..k_dst_offset + kv_len);
                        self.stream.memcpy_dtod(&k_src, &mut k_dst)?;
                    }
                    {
                        let mut v_dst = buffer.slice_mut(v_dst_offset..v_dst_offset + kv_len);
                        self.stream.memcpy_dtod(&v_src, &mut v_dst)?;
                    }

                    let kv_seq_len = kv_start + seq_len;
                    let k_cache_view = buffer.slice(0..kv_seq_len * kv_out);
                    let v_cache_view = buffer.slice(
                        cache.config.max_seq_len * kv_out
                            ..cache.config.max_seq_len * kv_out + kv_seq_len * kv_out,
                    );

                    let q_view = workspace.qkv.as_inner().slice(0..q_len);

                    let attn_cfg = FlashAttnConfig {
                        batch: 1,
                        num_heads: config.num_heads as u32,
                        head_dim: config.head_dim as u32,
                        q_seq_len: seq_len as u32,
                        kv_seq_len: kv_seq_len as u32,
                        q_stride: q_out as u32,
                        kv_stride: kv_out as u32,
                        o_stride: q_out as u32,
                        causal: if is_tree { 0 } else { 1 },
                        scale: 1.0 / (config.head_dim as f32).sqrt(),
                        q_pos_offset: kv_start as u32,
                    };
                    let attn_launch =
                        LaunchConfig::for_num_elems((seq_len * config.num_heads) as u32);
                    self.kernels.flash_attn.launch(
                        self.stream.as_ref(),
                        attn_launch,
                        &q_view,
                        &k_cache_view,
                        &v_cache_view,
                        workspace.attn_out.as_inner_mut(),
                        workspace.alibi_slopes.as_ref().map(GpuBuffer::as_inner),
                        &attn_cfg,
                    )?;
                }
            } else {
                let q_view = workspace.qkv.as_inner().slice(0..q_len);
                let k_view = workspace.qkv.as_inner().slice(q_len..q_len + kv_len);
                let v_view = workspace
                    .qkv
                    .as_inner()
                    .slice(q_len + kv_len..q_len + 2 * kv_len);

                let attn_cfg = FlashAttnConfig {
                    batch: 1,
                    num_heads: config.num_heads as u32,
                    head_dim: config.head_dim as u32,
                    q_seq_len: seq_len as u32,
                    kv_seq_len: seq_len as u32,
                    q_stride: q_out as u32,
                    kv_stride: kv_out as u32,
                    o_stride: q_out as u32,
                    causal: if is_tree { 0 } else { 1 },
                    scale: 1.0 / (config.head_dim as f32).sqrt(),
                    q_pos_offset: 0,
                };
                let attn_launch = LaunchConfig::for_num_elems((seq_len * config.num_heads) as u32);
                self.kernels.flash_attn.launch(
                    self.stream.as_ref(),
                    attn_launch,
                    &q_view,
                    &k_view,
                    &v_view,
                    workspace.attn_out.as_inner_mut(),
                    workspace.alibi_slopes.as_ref().map(GpuBuffer::as_inner),
                    &attn_cfg,
                )?;
            }

            let (o_proj, o_bias) =
                Self::resolve_weight_and_bias(weights, &layer_candidates(O_PROJ_PATTERNS, layer))?;
            let out_cfg = LinearConfig {
                m: seq_len as u32,
                n: hidden_size as u32,
                k: hidden_size as u32,
                input_stride: hidden_size as u32,
                weight_stride: hidden_size as u32,
                output_stride: hidden_size as u32,
                use_bias: if o_bias.is_some() { 1 } else { 0 },
            };
            let out_launch = LaunchConfig::for_num_elems((seq_len * hidden_size) as u32);
            linear_kernel.launch(
                self.stream.as_ref(),
                out_launch,
                workspace.attn_out.as_inner(),
                o_proj.as_inner(),
                o_bias.map(GpuBuffer::as_inner),
                workspace.norm.as_inner_mut(),
                &out_cfg,
            )?;

            let add_cfg = AddConfig {
                numel: (seq_len * hidden_size) as u32,
            };
            let add_launch = LaunchConfig::for_num_elems(add_cfg.numel);
            add_kernel.launch(
                self.stream.as_ref(),
                add_launch,
                workspace.hidden.as_inner(),
                workspace.norm.as_inner(),
                workspace.attn_out.as_inner_mut(),
                &add_cfg,
            )?;
            let src = workspace
                .attn_out
                .as_inner()
                .slice(0..seq_len * hidden_size);
            let mut dst = workspace
                .hidden
                .as_inner_mut()
                .slice_mut(0..seq_len * hidden_size);
            self.stream.memcpy_dtod(&src, &mut dst)?;

            self.rms_norm(
                rms_launch,
                &rms_cfg,
                &workspace.hidden,
                ffn_norm,
                &mut workspace.norm,
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
            let gate_cfg = LinearConfig {
                m: seq_len as u32,
                n: ffn_dim as u32,
                k: hidden_size as u32,
                input_stride: hidden_size as u32,
                weight_stride: hidden_size as u32,
                output_stride: ffn_dim as u32,
                use_bias: if gate_bias.is_some() { 1 } else { 0 },
            };
            let gate_launch = LaunchConfig::for_num_elems((seq_len * ffn_dim) as u32);
            linear_kernel.launch(
                self.stream.as_ref(),
                gate_launch,
                workspace.norm.as_inner(),
                gate_proj.as_inner(),
                gate_bias.map(GpuBuffer::as_inner),
                workspace.ffn_gate.as_inner_mut(),
                &gate_cfg,
            )?;

            let up_cfg = LinearConfig {
                m: seq_len as u32,
                n: ffn_dim as u32,
                k: hidden_size as u32,
                input_stride: hidden_size as u32,
                weight_stride: hidden_size as u32,
                output_stride: ffn_dim as u32,
                use_bias: if up_bias.is_some() { 1 } else { 0 },
            };
            linear_kernel.launch(
                self.stream.as_ref(),
                gate_launch,
                workspace.norm.as_inner(),
                up_proj.as_inner(),
                up_bias.map(GpuBuffer::as_inner),
                workspace.ffn_up.as_inner_mut(),
                &up_cfg,
            )?;

            let swiglu_cfg = SwiGluConfig {
                numel: (seq_len * ffn_dim) as u32,
            };
            let swiglu_launch = LaunchConfig::for_num_elems(swiglu_cfg.numel);
            self.fused_gate_up_silu(
                swiglu_launch,
                &swiglu_cfg,
                &workspace.ffn_gate,
                &workspace.ffn_up,
                &mut workspace.ffn_act,
            )?;

            let down_cfg = LinearConfig {
                m: seq_len as u32,
                n: hidden_size as u32,
                k: ffn_dim as u32,
                input_stride: ffn_dim as u32,
                weight_stride: ffn_dim as u32,
                output_stride: hidden_size as u32,
                use_bias: if down_bias.is_some() { 1 } else { 0 },
            };
            let down_launch = LaunchConfig::for_num_elems((seq_len * hidden_size) as u32);
            linear_kernel.launch(
                self.stream.as_ref(),
                down_launch,
                workspace.ffn_act.as_inner(),
                down_proj.as_inner(),
                down_bias.map(GpuBuffer::as_inner),
                workspace.ffn_out.as_inner_mut(),
                &down_cfg,
            )?;

            add_kernel.launch(
                self.stream.as_ref(),
                add_launch,
                workspace.hidden.as_inner(),
                workspace.ffn_out.as_inner(),
                workspace.attn_out.as_inner_mut(),
                &add_cfg,
            )?;
            let src = workspace
                .attn_out
                .as_inner()
                .slice(0..seq_len * hidden_size);
            let mut dst = workspace
                .hidden
                .as_inner_mut()
                .slice_mut(0..seq_len * hidden_size);
            self.stream.memcpy_dtod(&src, &mut dst)?;
        }

        if let Some(cache) = kv_cache.as_deref_mut() {
            cache.used = kv_start + seq_len;
        }

        let final_norm_name = Self::resolve_weight_name(weights, &final_norm_candidates())?;
        let final_norm = Self::resolve_weight(weights, &final_norm_name)?;
        let final_rms_cfg = RmsNormConfig {
            hidden_size: hidden_size as u32,
            stride: hidden_size as u32,
            eps: 1e-5,
            seq_len: seq_len as u32,
        };
        let final_launch = LaunchConfig::for_num_elems(seq_len as u32);
        self.rms_norm(
            final_launch,
            &final_rms_cfg,
            &workspace.hidden,
            final_norm,
            &mut workspace.norm,
        )?;

        let last_offset = (seq_len - 1) * hidden_size;
        let last_src = workspace
            .norm
            .as_inner()
            .slice(last_offset..last_offset + hidden_size);
        let mut last_dst = workspace
            .last_hidden
            .as_inner_mut()
            .slice_mut(0..hidden_size);
        self.stream.memcpy_dtod(&last_src, &mut last_dst)?;

        Ok(hidden_size)
    }

    fn forward_logits_with_index(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        kv_cache: &mut KvCacheHandle,
        config: &GeneratorForwardConfig,
        logits_idx: usize,
        expected_position: Option<usize>,
    ) -> BackendResult<LogitsTensor> {
        let start = Instant::now();
        if tokens.is_empty() {
            return Err(BackendError::InvalidConfig("empty tokens".into()));
        }

        let mut caches = self.kv_caches.lock().expect("kv cache lock poisoned");
        let cache = caches
            .get_mut(kv_cache.0)
            .ok_or_else(|| BackendError::InvalidHandle("kv cache handle".into()))?;

        if let Some(position) = expected_position {
            Self::validate_sequence_position(cache, tokens.len(), position)?;
        }

        {
            let mut managers = self.swap_managers.lock().expect("swap managers lock poisoned");
            if let Some(manager) = managers
                .get_mut(kv_cache.0)
                .and_then(|entry| entry.as_mut())
            {
                self.auto_swap_out_if_needed(cache, manager)?;
            }
        }

        let hidden_size = self.forward_hidden(tokens, topology, weights, Some(cache), config)?;

        if cache.is_paged() {
            let mut managers = self.swap_managers.lock().expect("swap managers lock poisoned");
            if let Some(manager) = managers
                .get_mut(kv_cache.0)
                .and_then(|entry| entry.as_mut())
            {
                Self::mark_kv_pages_accessed(cache, manager)?;
            }
        }

        drop(caches);

        self.ensure_logits_buffer_at(config.vocab_size, logits_idx)?;
        let linear_kernel = &self.kernels.linear;

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

        let lm_cfg = LinearConfig {
            m: 1,
            n: config.vocab_size as u32,
            k: hidden_size as u32,
            input_stride: hidden_size as u32,
            weight_stride: hidden_size as u32,
            output_stride: config.vocab_size as u32,
            use_bias: if lm_bias.is_some() { 1 } else { 0 },
        };
        let lm_launch = LaunchConfig::for_num_elems(config.vocab_size as u32);
        linear_kernel.launch(
            self.stream.as_ref(),
            lm_launch,
            workspace.last_hidden.as_inner(),
            lm_head.as_inner(),
            lm_bias.map(GpuBuffer::as_inner),
            logits_buf,
            &lm_cfg,
        )?;

        self.record_forward_time(start.elapsed());
        Ok(LogitsTensor::new(logits_idx))
    }
}

impl Backend for CudaBackend {
    type Tensor<T> = GpuBuffer<T>;

    fn upload_weights<T: DeviceRepr + Clone>(&self, data: &[T]) -> BackendResult<Self::Tensor<T>> {
        let slice = self.stream.clone_htod(data)?;
        Ok(GpuBuffer::new(slice))
    }

    fn alloc_kv_cache(&self, config: &KvCacheConfig) -> BackendResult<KvCacheHandle> {
        if config.dtype_size != std::mem::size_of::<f32>() {
            return Err(BackendError::InvalidKvCache(
                "cuda kv cache expects f32 dtype".into(),
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
            let mut host_ptrs = Vec::with_capacity(total_pages);
            for _ in 0..total_pages {
                let page = self.stream.alloc_zeros::<f32>(page_len)?;
                let (ptr, sync) = page.device_ptr(self.stream.as_ref());
                host_ptrs.push(ptr as u64);
                drop(sync);
                pages.push(page);
            }
            let page_ptrs = self.stream.clone_htod(&host_ptrs)?;
            KvCacheStorage::Paged {
                pages,
                page_ptrs,
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
            let slice = self.stream.alloc_zeros::<f32>(elements)?;
            KvCacheStorage::Contiguous(slice)
        };

        // 初始化 SwapManager (如果配置了)
        let swap_manager = config.swap_config.map(|cfg| {
            let total_pages = if config.is_paged() {
                config.num_layers * pages_per_layer
            } else {
                0
            };
            if total_pages > 0 && cfg.enable_swap {
                Some(SwapManager::new(total_pages, cfg))
            } else {
                None
            }
        }).flatten();

        let mut caches = self.kv_caches.lock().expect("kv cache lock poisoned");
        let mut managers = self.swap_managers.lock().expect("swap managers lock poisoned");
        caches.push(KvCacheEntry {
            storage,
            config: *config,
            used: 0,
        });
        managers.push(swap_manager);
        Ok(KvCacheHandle::new(caches.len() - 1))
    }

    fn swap_out_pages(
        &self,
        kv_cache: &mut KvCacheHandle,
        page_indices: &[usize],
    ) -> BackendResult<()> {
        let mut caches = self.kv_caches.lock().expect("kv cache lock poisoned");
        let mut managers = self.swap_managers.lock().expect("swap managers lock poisoned");

        let cache = caches
            .get_mut(kv_cache.0)
            .ok_or_else(|| BackendError::InvalidHandle("kv cache handle".into()))?;

        let manager = managers
            .get_mut(kv_cache.0)
            .ok_or_else(|| BackendError::InvalidHandle("swap manager handle".into()))?;

        let Some(manager) = manager.as_mut() else {
            return Err(BackendError::Unimplemented("swap not configured for this cache"));
        };

        self.swap_out_pages_with_manager(cache, manager, page_indices)
    }

    fn swap_in_pages(
        &self,
        kv_cache: &mut KvCacheHandle,
        page_indices: &[usize],
    ) -> BackendResult<()> {
        let mut caches = self.kv_caches.lock().expect("kv cache lock poisoned");
        let mut managers = self.swap_managers.lock().expect("swap managers lock poisoned");

        let cache = caches
            .get_mut(kv_cache.0)
            .ok_or_else(|| BackendError::InvalidHandle("kv cache handle".into()))?;

        let manager = managers
            .get_mut(kv_cache.0)
            .ok_or_else(|| BackendError::InvalidHandle("swap manager handle".into()))?;

        let Some(manager) = manager.as_mut() else {
            return Err(BackendError::Unimplemented("swap not configured for this cache"));
        };

        let KvCacheStorage::Paged { pages, .. } = &mut cache.storage else {
            return Err(BackendError::InvalidKvCache("cache is not paged".into()));
        };

        // 执行 swap-in
        for &page_idx in page_indices {
            if page_idx >= pages.len() {
                continue;
            }

            let gpu_page = &mut pages[page_idx];

            // 从 SwapManager 获取 CPU 数据
            let Some(host_data) = manager.get_swapped_data(page_idx) else {
                continue;
            };

            let copy_len = host_data.len().min(gpu_page.len());
            let mut dst = gpu_page.slice_mut(0..copy_len);
            self.stream.memcpy_htod(&host_data[0..copy_len], &mut dst)?;

            // 完成换入
            manager.complete_swap_in(page_idx)?;
        }

        Ok(())
    }

    fn get_memory_pressure(&self) -> BackendResult<f32> {
        self.ctx.bind_to_thread()?;
        let (free, total) = result::mem_get_info()?;
        if total == 0 {
            return Ok(0.0);
        }
        let used = total.saturating_sub(free);
        let pressure = (used as f64) / (total as f64);
        let pressure = pressure.clamp(0.0, 1.0) as f32;
        Ok(pressure)
    }

    fn get_page_states(
        &self,
        kv_cache: &KvCacheHandle,
    ) -> BackendResult<Vec<(usize, crate::kernel_types::PageState)>> {
        let managers = self.swap_managers.lock().expect("swap managers lock poisoned");
        let manager = managers
            .get(kv_cache.0)
            .ok_or_else(|| BackendError::InvalidHandle("swap manager handle".into()))?;

        if let Some(manager) = manager.as_ref() {
            let mut states = Vec::new();
            for idx in 0..manager.stats().total_pages {
                if let Some(state) = manager.get_page_state(idx) {
                    states.push((idx, state));
                }
            }
            Ok(states)
        } else {
            Ok(Vec::new())
        }
    }

    fn generator_forward_gpu_pure(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        kv_cache: &mut KvCacheHandle,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<LogitsTensor> {
        self.forward_logits_with_index(tokens, topology, weights, kv_cache, config, 0, None)
    }

    fn batch_forward_gpu_pure(
        &self,
        batch: &BatchInput,
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        kv_caches: &mut [KvCacheHandle],
        config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<LogitsTensor>> {
        if batch.sequences.is_empty() {
            return Err(BackendError::InvalidConfig("empty batch".into()));
        }
        if batch.sequences.len() != kv_caches.len() {
            return Err(BackendError::InvalidConfig(
                "batch size and kv cache count mismatch".into(),
            ));
        }

        let mut outputs = Vec::with_capacity(batch.sequences.len());
        for (idx, (sequence, cache)) in batch
            .sequences
            .iter()
            .zip(kv_caches.iter_mut())
            .enumerate()
        {
            if sequence.tokens.is_empty() {
                return Err(BackendError::InvalidConfig(
                    "empty sequence tokens".into(),
                ));
            }
            let logits = self.forward_logits_with_index(
                &sequence.tokens,
                topology,
                weights,
                cache,
                config,
                idx,
                Some(sequence.position),
            )?;
            outputs.push(logits);
        }
        Ok(outputs)
    }

    fn sample_from_tensor(
        &self,
        logits: &LogitsTensor,
        _topology: &AttentionTopology,
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> BackendResult<Vec<u32>> {
        let start = Instant::now();
        let sampling_kernel = &self.kernels.sampling;
        let logits_guard = self.logits.lock().expect("logits lock poisoned");
        let logits_buf = logits_guard
            .get(logits.0)
            .ok_or_else(|| BackendError::InvalidHandle("logits handle".into()))?;

        if logits_buf.len() != vocab_size {
            return Err(BackendError::InvalidConfig(
                "logits buffer size mismatch".into(),
            ));
        }

        let mut sampling_guard = self.sampling.lock().expect("sampling lock poisoned");
        if sampling_guard.is_none() {
            *sampling_guard = Some(self.stream.alloc_zeros::<u32>(1)?);
        }
        let out_buf = sampling_guard
            .as_mut()
            .expect("sampling buffer should exist");

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

        let params = SamplingKernelConfig {
            vocab_size: vocab_size as u32,
            top_k: config.top_k as u32,
            top_p,
            temperature: temp,
            stride: vocab_size as u32,
            batch: 1,
        };
        let launch = LaunchConfig::for_num_elems(1);
        sampling_kernel.launch(self.stream.as_ref(), launch, logits_buf, out_buf, &params)?;

        let mut host = vec![0u32; 1];
        self.stream.memcpy_dtoh(out_buf, &mut host)?;
        self.record_dtoh(std::mem::size_of::<u32>());
        self.record_sample_time(start.elapsed());
        Ok(host)
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<f32>> {
        let hidden_size = self.forward_hidden(tokens, topology, weights, None, config)?;

        let workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
        let workspace = workspace_guard
            .as_ref()
            .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;

        let mut host = vec![0f32; hidden_size];
        self.stream
            .memcpy_dtoh(workspace.last_hidden.as_inner(), &mut host)?;
        self.record_dtoh(host.len() * std::mem::size_of::<f32>());
        Ok(host)
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        topology: &AttentionTopology,
        weights: &dyn TensorLookup<Self>,
        config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<f32>> {
        let hidden_size = self.forward_hidden(tokens, topology, weights, None, config)?;

        let score_name = find_tensor_name(weights, &score_head_candidates());
        if let Some(score_name) = score_name {
            let score_weight = Self::resolve_weight(weights, &score_name)?;
            let score_shape = weights
                .tensor_shape(&score_name)
                .ok_or_else(|| BackendError::MissingTensor(score_name.clone()))?;
            let out_dim = Self::deduce_out_dim(score_shape, hidden_size)?;
            let score_bias = Self::resolve_bias(weights, &score_name);

            let logits_idx = self.ensure_logits_buffer(config.vocab_size)?;
            let mut logits_guard = self.logits.lock().expect("logits lock poisoned");
            let logits_buf = logits_guard
                .get_mut(logits_idx)
                .ok_or_else(|| BackendError::InvalidHandle("logits handle".into()))?;

            let workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
            let workspace = workspace_guard
                .as_ref()
                .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;

            let linear_kernel = &self.kernels.linear;

            let mut out_view = logits_buf.slice_mut(0..out_dim);
            let cfg = LinearConfig {
                m: 1,
                n: out_dim as u32,
                k: hidden_size as u32,
                input_stride: hidden_size as u32,
                weight_stride: hidden_size as u32,
                output_stride: out_dim as u32,
                use_bias: if score_bias.is_some() { 1 } else { 0 },
            };
            let launch = LaunchConfig::for_num_elems(out_dim as u32);
            linear_kernel.launch(
                self.stream.as_ref(),
                launch,
                workspace.last_hidden.as_inner(),
                score_weight.as_inner(),
                score_bias.map(GpuBuffer::as_inner),
                &mut out_view,
                &cfg,
            )?;

            let mut host = vec![0f32; out_dim];
            let out_read = out_view.as_view();
            self.stream.memcpy_dtoh(&out_read, &mut host)?;
            self.record_dtoh(host.len() * std::mem::size_of::<f32>());
            Ok(host)
        } else {
            let workspace_guard = self.workspace.lock().expect("workspace lock poisoned");
            let workspace = workspace_guard
                .as_ref()
                .ok_or_else(|| BackendError::InvalidConfig("workspace missing".into()))?;
            let mut host = vec![0f32; hidden_size];
            self.stream
                .memcpy_dtoh(workspace.last_hidden.as_inner(), &mut host)?;
            self.record_dtoh(host.len() * std::mem::size_of::<f32>());
            Ok(host)
        }
    }
}
