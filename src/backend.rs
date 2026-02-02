use std::env;

use crate::backend_trait::{BackendError, BackendResult};
use crate::cpu_backend::CpuBackend;
use crate::cuda_backend::CudaBackend;

pub enum BackendKind {
    Cuda(CudaBackend),
    Cpu(CpuBackend),
}

pub fn auto_select_backend() -> BackendResult<BackendKind> {
    if let Ok(value) = env::var("GLLM_DEVICE") {
        let value = value.trim();
        if value.eq_ignore_ascii_case("cpu") {
            return Ok(BackendKind::Cpu(CpuBackend::new()));
        }
        if value.starts_with("cuda") {
            let ordinal = if let Some((_, idx)) = value.split_once(':') {
                idx.parse::<usize>()
                    .map_err(|_| BackendError::InvalidBackendOverride(value.to_string()))?
            } else {
                0
            };
            let cuda = CudaBackend::new(ordinal)?;
            return Ok(BackendKind::Cuda(cuda));
        }
        return Err(BackendError::InvalidBackendOverride(value.to_string()));
    }

    match CudaBackend::new(0) {
        Ok(cuda) => Ok(BackendKind::Cuda(cuda)),
        Err(_) => Ok(BackendKind::Cpu(CpuBackend::new())),
    }
}
