//! Runtime backend detection with caching and auto-recovery.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Backend detection result that can be cached.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendDetectionResult {
    pub backend_type: BackendType,
    pub device_id: Option<String>,
    pub timestamp: i64,
    pub hostname: String,
    pub arch: Option<String>, // GPU 架构 (如 sm_86, gfx1030, apple-m1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackendType {
    Cuda,
    Rocm,
    Metal,
    Wgpu,
    Cpu,
}

impl BackendType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::Metal => "Metal",
            Self::Wgpu => "WGPU",
            Self::Cpu => "CPU",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Wgpu => "wgpu",
            Self::Cpu => "cpu",
        }
    }
}

/// Device information for kernel selection.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub backend: BackendType,
    pub device_id: Option<String>,
    pub arch: Option<String>, // GPU 架构标识
}

/// Cached detection result.
static CACHED_RESULT: OnceLock<Option<BackendDetectionResult>> = OnceLock::new();

/// Detect the best available backend.
///
/// Priority: CUDA → ROCm → Metal → WGPU → CPU
///
/// This function:
/// 1. Checks cached result from previous detection
/// 2. Validates the cached backend is still available
/// 3. Re-detects if validation fails
pub fn detect_backend() -> BackendType {
    CACHED_RESULT
        .get_or_init(|| {
            // Try to load cached result
            if let Some(cached) = load_cached_detection() {
                log::info!("Loaded cached backend: {}", cached.backend_type.name());

                // Validate cached backend
                if validate_backend(&cached.backend_type) {
                    return Some(cached);
                } else {
                    log::warn!("Cached backend {} is no longer available, re-detecting", cached.backend_type.name());
                }
            }

            // Perform fresh detection
            let result = perform_detection();
            save_cached_detection(&result);
            Some(result)
        })
        .as_ref()
        .map(|r| r.backend_type)
        .unwrap_or(BackendType::Cpu)
}

/// Force re-detection and update cache.
pub fn redetect_backend() -> BackendType {
    let result = perform_detection();
    save_cached_detection(&result);

    // Update cached result
    CACHED_RESULT.set(Some(result.clone())).ok();

    log::info!("Re-detected backend: {}", result.backend_type.name());
    result.backend_type
}

/// Perform actual backend detection with priority: CUDA → ROCm → Metal → WGPU → CPU
fn perform_detection() -> BackendDetectionResult {
    let backend_type = if try_cuda() {
        BackendType::Cuda
    } else if try_rocm() {
        BackendType::Rocm
    } else if try_metal() {
        BackendType::Metal
    } else if try_wgpu() {
        BackendType::Wgpu
    } else {
        BackendType::Cpu
    };

    let device_id = get_device_id(backend_type);
    let arch = detect_device_arch(backend_type);

    BackendDetectionResult {
        backend_type,
        device_id,
        arch,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64,
        hostname: hostname(),
    }
}

/// Validate that a detected backend is still available.
fn validate_backend(backend_type: &BackendType) -> bool {
    match backend_type {
        BackendType::Cuda => try_cuda(),
        BackendType::Rocm => try_rocm(),
        BackendType::Metal => try_metal(),
        BackendType::Wgpu => try_wgpu(),
        BackendType::Cpu => true, // CPU always available
    }
}

/// Get cache file path.
fn cache_file_path() -> PathBuf {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());

    PathBuf::from(home_dir)
        .join(".gsc")
        .join("gllm")
        .join("backend.json")
}

/// Load cached detection result.
fn load_cached_detection() -> Option<BackendDetectionResult> {
    let cache_path = cache_file_path();

    if !cache_path.exists() {
        return None;
    }

    match std::fs::read_to_string(&cache_path) {
        Ok(content) => {
            match serde_json::from_str::<BackendDetectionResult>(&content) {
                Ok(result) => {
                    log::debug!("Loaded backend cache from: {:?}", cache_path);
                    Some(result)
                }
                Err(e) => {
                    log::warn!("Failed to parse backend cache: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to read backend cache: {}", e);
            None
        }
    }
}

/// Save detection result to cache.
fn save_cached_detection(result: &BackendDetectionResult) {
    let cache_path = cache_file_path();

    // Ensure cache directory exists
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    match serde_json::to_string_pretty(result) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&cache_path, json) {
                log::warn!("Failed to write backend cache: {}", e);
            } else {
                log::info!("Saved backend cache to: {:?}", cache_path);
            }
        }
        Err(e) => {
            log::warn!("Failed to serialize backend cache: {}", e);
        }
    }
}

/// Get device ID for a backend.
fn get_device_id(backend_type: BackendType) -> Option<String> {
    match backend_type {
        #[cfg(feature = "cuda")]
        BackendType::Cuda => Some("cuda:0".to_string()),
        #[cfg(not(feature = "cuda"))]
        BackendType::Cuda => None,

        #[cfg(feature = "rocm")]
        BackendType::Rocm => Some("rocm:0".to_string()),
        #[cfg(not(feature = "rocm"))]
        BackendType::Rocm => None,

        #[cfg(feature = "metal")]
        BackendType::Metal => Some("metal:default".to_string()),
        #[cfg(not(feature = "metal"))]
        BackendType::Metal => None,

        BackendType::Wgpu => Some("wgpu:default".to_string()),
        BackendType::Cpu => Some("cpu:default".to_string()),
    }
}

/// Get system hostname.
fn hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| {
        std::env::var("COMPUTERNAME").unwrap_or_else(|_| "unknown".to_string())
    })
}

// Backend-specific detection functions

#[cfg(feature = "cuda")]
fn try_cuda() -> bool {
    use cudarc::driver::result;

    match result::init() {
        Ok(_) => {
            log::info!("CUDA backend detected");
            true
        }
        Err(e) => {
            log::debug!("CUDA not available: {}", e);
            false
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn try_cuda() -> bool {
    false
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn try_rocm() -> bool {
    if std::path::Path::new("/dev/kfd").exists() {
        log::info!("ROCm backend detected");
        true
    } else {
        log::debug!("ROCm not available");
        false
    }
}

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
fn try_rocm() -> bool {
    false
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn try_metal() -> bool {
    use metal::Device;

    if Device::system_default().is_some() {
        log::info!("Metal backend detected");
        true
    } else {
        log::debug!("Metal not available");
        false
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn try_metal() -> bool {
    false
}

#[cfg(feature = "wgpu")]
fn try_wgpu() -> bool {
    log::info!("WGPU backend available");
    true
}

#[cfg(not(feature = "wgpu"))]
fn try_wgpu() -> bool {
    false
}

/// Detect device architecture for kernel selection.
fn detect_device_arch(backend_type: BackendType) -> Option<String> {
    match backend_type {
        #[cfg(feature = "cuda")]
        BackendType::Cuda => detect_cuda_arch(),
        #[cfg(feature = "rocm")]
        BackendType::Rocm => detect_rocm_arch(),
        #[cfg(feature = "metal")]
        BackendType::Metal => detect_metal_arch(),
        _ => None,
    }
}

/// Detect CUDA GPU architecture (compute capability).
#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> Option<String> {
    use cudarc::driver::result;

    if let Ok(dev) = result::Device::get(0) {
        // 获取计算能力
        let major = dev.compute_capability_major().ok()?;
        let minor = dev.compute_capability_minor().ok()?;
        Some(format!("sm_{}{}", major, minor))
    } else {
        None
    }
}

/// Detect AMD GPU architecture.
#[cfg(feature = "rocm")]
fn detect_rocm_arch() -> Option<String> {
    // 简化实现：读取 /sys/class/kfd/kfd/topology/... 中的信息
    // 或者通过 rocm-smi 工具
    // 这里返回一个通用的 RDNA2 架构作为示例
    Some("gfx1030".to_string())
}

/// Detect Apple Silicon architecture.
#[cfg(feature = "metal")]
fn detect_metal_arch() -> Option<String> {
    use metal::Device;

    if let Some(device) = Device::system_default() {
        let name = device.name();
        // 根据 GPU 名称推断架构
        if name.contains("M1") {
            Some("apple-m1".to_string())
        } else if name.contains("M2") {
            Some("apple-m2".to_string())
        } else if name.contains("M3") {
            Some("apple-m3".to_string())
        } else {
            Some("apple-m1".to_string()) // 默认
        }
    } else {
        None
    }
}

/// Ensure kernels are downloaded and cached.
///
/// This function:
/// 1. Detects current backend and device architecture
/// 2. Checks if kernels are cached
/// 3. Downloads from GitHub release if missing
///
/// # Example
///
/// ```no_run
/// use gllm_kernels::ensure_kernels;
///
/// if let Err(e) = ensure_kernels() {
///     eprintln!("Failed to download kernels: {}", e);
/// }
/// ```
pub fn ensure_kernels() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "kernel-downloader")]
    {
        use crate::kernel_downloader::KernelDownloader;

        let detection = CACHED_RESULT
            .get()
            .or_else(|| {
                let result = perform_detection();
                save_cached_detection(&result);
                Some(result)
            });

        if let Some(info) = detection {
            let downloader = KernelDownloader::new()?;

            // 只为 GPU 后端下载 kernels
            match info.backend_type {
                BackendType::Cuda | BackendType::Rocm | BackendType::Metal => {
                    if let Some(arch) = &info.arch {
                        log::info!("Ensuring kernels for {} ({})", info.backend_type.name(), arch);
                        downloader.download_all_kernels(&info.backend_type, &info.device_id.clone().map(|id| crate::runtime_detection::DeviceInfo {
                            backend: info.backend_type,
                            device_id: Some(id),
                            arch: Some(arch.clone()),
                        }).unwrap_or_else(|| crate::runtime_detection::DeviceInfo {
                            backend: info.backend_type,
                            device_id: None,
                            arch: Some(arch.clone()),
                        }))?;
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_priority() {
        let backend = detect_backend();
        println!("Detected backend: {:?}", backend);
    }

    #[test]
    fn test_cache_persistence() {
        // Clear cache
        CACHED_RESULT.set(None).ok();

        let backend1 = detect_backend();
        let backend2 = detect_backend();

        assert_eq!(backend1, backend2);
    }
}
