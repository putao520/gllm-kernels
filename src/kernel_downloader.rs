//! Kernel 自动下载器 - 从 GitHub Release 下载预编译 kernels

use crate::runtime_detection::{BackendType, DeviceInfo};
use std::path::PathBuf;
use std::time::Duration;

/// Kernel 下载器
pub struct KernelDownloader {
    client: ureq::Agent,
    repo_owner: String,
    repo_name: String,
    cache_dir: PathBuf,
}

impl KernelDownloader {
    /// 创建新的下载器
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let cache_dir = kernel_cache_dir();

        Ok(Self {
            client: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(30))
                .build(),
            repo_owner: "putao520".to_string(),
            repo_name: "gllm".to_string(),
            cache_dir,
        })
    }

    /// 检查 kernel 是否已缓存
    pub fn is_kernel_cached(&self, backend: &BackendType, kernel_name: &str, device_arch: &str) -> bool {
        let kernel_path = kernel_cache_path(backend.as_str(), kernel_name, device_arch);
        kernel_path.exists()
    }

    /// 下载指定的 kernel
    pub fn download_kernel(
        &self,
        backend: &BackendType,
        kernel_name: &str,
        device_arch: &str,
    ) -> Result<PathBuf, KernelDownloadError> {
        // 检查是否已缓存
        if self.is_kernel_cached(backend, kernel_name, device_arch) {
            let path = kernel_cache_path(backend.as_str(), kernel_name, device_arch);
            return Ok(path);
        }

        log::info!(
            "下载 {} kernel: {} for {}",
            backend.as_str(),
            kernel_name,
            device_arch
        );

        // 构建 GitHub release URL
        let url = self.build_release_url(backend, kernel_name, device_arch)?;

        // 下载 kernel
        let response = self
            .client
            .get(&url)
            .call()
            .map_err(|e| KernelDownloadError::NetworkError(e.to_string()))?;

        // 检查响应状态
        if response.status() != 200 {
            return Err(KernelDownloadError::NotFound(format!(
                "Kernel not found: {} (status: {})",
                url,
                response.status()
            )));
        }

        // 读取数据
        let data = response
            .into_string()
            .map_err(|e| KernelDownloadError::IoError(e.to_string()))?;

        // 保存到缓存
        let kernel_path = kernel_cache_path(backend.as_str(), kernel_name, device_arch);
        if let Some(parent) = kernel_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| KernelDownloadError::IoError(e.to_string()))?;
        }

        std::fs::write(&kernel_path, data)
            .map_err(|e| KernelDownloadError::IoError(e.to_string()))?;

        log::info!("✓ Kernel 下载完成: {:?}", kernel_path);

        Ok(kernel_path)
    }

    /// 批量下载所有需要的 kernels
    pub fn download_all_kernels(
        &self,
        backend: &BackendType,
        device_info: &DeviceInfo,
    ) -> Result<(), KernelDownloadError> {
        let kernels = match backend {
            BackendType::Cuda => vec!["flash_attention", "tiled_attention"],
            BackendType::Rocm => vec!["flash_attention"],
            BackendType::Metal => vec!["flash_attention"],
            _ => vec![],
        };

        let device_arch = device_info.arch.clone().unwrap_or_default();

        for kernel_name in kernels {
            self.download_kernel(backend, kernel_name, &device_arch)?;
        }

        Ok(())
    }

    /// 构建 GitHub release URL
    fn build_release_url(
        &self,
        backend: &BackendType,
        kernel_name: &str,
        device_arch: &str,
    ) -> Result<String, KernelDownloadError> {
        // 获取最新版本
        let version = self.get_latest_version()?;

        // 确定文件扩展名
        let ext = match backend {
            BackendType::Cuda => "ptx",
            BackendType::Rocm => "hsaco",
            BackendType::Metal => "metallib",
            _ => return Err(KernelDownloadError::InvalidBackend),
        };

        // 构建 URL
        // 格式: https://github.com/putao520/gllm/releases/download/v{version}/kernels/{backend}/{kernel_name}/{arch}.{ext}
        let url = format!(
            "https://github.com/{}/{}/releases/download/v{}/kernels/{}/{}/{}.{}",
            self.repo_owner, self.repo_name, version, backend.as_str(), kernel_name, device_arch, ext
        );

        Ok(url)
    }

    /// 获取最新 release 版本
    fn get_latest_version(&self) -> Result<String, KernelDownloadError> {
        // 简化版本：从 Cargo.toml 读取
        // 生产环境应该从 GitHub API 获取
        Ok("0.10.6".to_string())
    }
}

impl Default for KernelDownloader {
    fn default() -> Self {
        Self::new().expect("Failed to create KernelDownloader")
    }
}

/// Kernel 下载错误
#[derive(Debug)]
pub enum KernelDownloadError {
    NetworkError(String),
    IoError(String),
    NotFound(String),
    InvalidBackend,
}

impl std::fmt::Display for KernelDownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelDownloadError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            KernelDownloadError::IoError(msg) => write!(f, "IO error: {}", msg),
            KernelDownloadError::NotFound(msg) => write!(f, "Not found: {}", msg),
            KernelDownloadError::InvalidBackend => write!(f, "Invalid backend"),
        }
    }
}

impl std::error::Error for KernelDownloadError {}

/// 获取 kernel 缓存目录
fn kernel_cache_dir() -> PathBuf {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());

    PathBuf::from(home_dir)
        .join(".gsc")
        .join("gllm")
        .join("kernels")
}

/// 获取 kernel 缓存文件路径
fn kernel_cache_path(backend: &str, kernel_name: &str, device_arch: &str) -> PathBuf {
    let ext = match backend {
        "cuda" => "ptx",
        "rocm" => "hsaco",
        "metal" => "metallib",
        _ => "bin",
    };

    kernel_cache_dir()
        .join(backend)
        .join(kernel_name)
        .join(format!("{}.{}", device_arch, ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_cache_path() {
        let path = kernel_cache_path("cuda", "flash_attention", "sm_86");
        assert!(path.ends_with(".gsc/gllm/kernels/cuda/flash_attention/sm_86.ptx"));
    }
}
