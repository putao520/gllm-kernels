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

    BackendDetectionResult {
        backend_type,
        device_id,
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
