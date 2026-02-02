use crate::kernel_types::{
    GeneratorForwardConfig, KvCacheConfig, PageId, PageState, SamplingConfig,
};
use cudarc::driver::CudaSlice;
use cudarc::driver::DeviceRepr;
use cudarc::driver::DriverError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("cuda driver error: {0}")]
    Cuda(String),
    #[error("unsupported CUDA SM version {major}.{minor}")]
    UnsupportedSm { major: i32, minor: i32 },
    #[error("invalid kv cache config: {0}")]
    InvalidKvCache(String),
    #[error("invalid backend override: {0}")]
    InvalidBackendOverride(String),
    #[error("missing or empty cubin for {0}")]
    InvalidCubin(&'static str),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("invalid handle: {0}")]
    InvalidHandle(String),
    #[error("unimplemented backend feature: {0}")]
    Unimplemented(&'static str),
}

pub type BackendResult<T> = Result<T, BackendError>;

impl From<DriverError> for BackendError {
    fn from(err: DriverError) -> Self {
        BackendError::Cuda(format!("{err:?}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogitsTensor(pub(crate) usize);

impl LogitsTensor {
    pub(crate) fn new(id: usize) -> Self {
        Self(id)
    }
}

pub type LogitsHandle = LogitsTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCacheHandle(pub(crate) usize);

impl KvCacheHandle {
    pub(crate) fn new(id: usize) -> Self {
        Self(id)
    }
}

#[derive(Debug)]
pub struct AttentionTopology {
    pub tree_structure: Option<CudaSlice<i32>>,
}

impl AttentionTopology {
    pub fn linear() -> Self {
        Self {
            tree_structure: None,
        }
    }

    pub fn tree(tree_structure: CudaSlice<i32>) -> Self {
        Self {
            tree_structure: Some(tree_structure),
        }
    }

    pub fn is_tree(&self) -> bool {
        self.tree_structure.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct BatchInput {
    pub sequences: Vec<SequenceInput>,
}

#[derive(Debug, Clone)]
pub struct SequenceInput {
    pub tokens: Vec<u32>,
    pub position: usize,
}

pub trait Backend: Send + Sync {
    type Tensor<T>;

    fn upload_weights<T: DeviceRepr + Clone>(&self, data: &[T]) -> BackendResult<Self::Tensor<T>>;
    fn alloc_kv_cache(&self, config: &KvCacheConfig) -> BackendResult<KvCacheHandle>;

    /// Swap-out: 将 KV cache 页面从 GPU 搬运到 CPU
    fn swap_out_pages(
        &self,
        _kv_cache: &mut KvCacheHandle,
        _page_indices: &[usize],
    ) -> BackendResult<()> {
        Err(BackendError::Unimplemented("swap_out_pages"))
    }

    /// Swap-in: 将 KV cache 页面从 CPU 搬运回 GPU
    fn swap_in_pages(
        &self,
        _kv_cache: &mut KvCacheHandle,
        _page_indices: &[usize],
    ) -> BackendResult<()> {
        Err(BackendError::Unimplemented("swap_in_pages"))
    }

    /// 获取当前内存压力 (0.0-1.0)
    fn get_memory_pressure(&self) -> BackendResult<f32> {
        Ok(0.0)
    }

    /// 获取页面状态 (用于调试和监控)
    fn get_page_states(
        &self,
        _kv_cache: &KvCacheHandle,
    ) -> BackendResult<Vec<(PageId, PageState)>> {
        Ok(Vec::new())
    }

    fn generator_forward_gpu_pure(
        &self,
        _tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn TensorLookup<Self>,
        _kv_cache: &mut KvCacheHandle,
        _config: &GeneratorForwardConfig,
    ) -> BackendResult<LogitsTensor> {
        Err(BackendError::Unimplemented("generator_forward_gpu_pure"))
    }

    fn batch_forward_gpu_pure(
        &self,
        _batch: &BatchInput,
        _topology: &AttentionTopology,
        _weights: &dyn TensorLookup<Self>,
        _kv_caches: &mut [KvCacheHandle],
        _config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<LogitsTensor>> {
        Err(BackendError::Unimplemented("batch_forward_gpu_pure"))
    }

    fn sample_from_tensor(
        &self,
        _logits: &LogitsTensor,
        _topology: &AttentionTopology,
        _vocab_size: usize,
        _config: &SamplingConfig,
    ) -> BackendResult<Vec<u32>> {
        Err(BackendError::Unimplemented("sample_from_tensor"))
    }

    fn embedding_forward_gpu_pure(
        &self,
        _tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn TensorLookup<Self>,
        _config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<f32>> {
        Err(BackendError::Unimplemented("embedding_forward_gpu_pure"))
    }

    fn rerank_forward_gpu_pure(
        &self,
        _tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn TensorLookup<Self>,
        _config: &GeneratorForwardConfig,
    ) -> BackendResult<Vec<f32>> {
        Err(BackendError::Unimplemented("rerank_forward_gpu_pure"))
    }
}

pub trait TensorLookup<B: Backend> {
    fn tensor_f32(&self, name: &str) -> Option<&B::Tensor<f32>>;
    fn tensor_shape(&self, name: &str) -> Option<&[usize]>;
}
