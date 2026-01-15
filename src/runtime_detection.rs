//! Runtime backend detection with caching and auto-recovery.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
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

/// GPU capabilities for device detection and model selection.
///
/// This struct provides detailed information about the available GPU/compute device,
/// including VRAM size, device name, and backend type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuCapabilities {
    /// Device name (e.g., "NVIDIA GeForce RTX 4090", "Apple M2 Pro")
    pub name: String,
    /// Backend type (CUDA, ROCm, Metal, WGPU, CPU)
    pub backend_type: BackendType,
    /// GPU architecture (e.g., "sm_89", "gfx1100", "apple-m2")
    pub arch: Option<String>,
    /// Total VRAM in bytes (0 for CPU)
    pub vram_bytes: u64,
    /// Whether GPU is available and usable
    pub gpu_available: bool,
    /// Recommended batch size based on VRAM
    pub recommended_batch_size: usize,
    /// Backend name for display
    pub backend_name: String,
}

/// Cached GPU capabilities (static singleton)
static CACHED_GPU_CAPS: OnceLock<GpuCapabilities> = OnceLock::new();

impl GpuCapabilities {
    /// Detect GPU capabilities (cached after first call).
    ///
    /// This is the primary entry point for GPU detection.
    /// Results are cached for the lifetime of the process.
    pub fn detect() -> &'static GpuCapabilities {
        CACHED_GPU_CAPS.get_or_init(|| Self::detect_impl())
    }

    /// Force fresh detection without using cache.
    ///
    /// Use this when you need to re-detect after hardware changes.
    pub fn detect_fresh() -> GpuCapabilities {
        Self::detect_impl()
    }

    /// Internal detection implementation.
    fn detect_impl() -> GpuCapabilities {
        let backend_type = detect_backend_impl();
        let arch = detect_device_arch(backend_type);
        let (name, vram_bytes) = detect_device_details(backend_type);
        let gpu_available = !matches!(backend_type, BackendType::Cpu);

        // Calculate recommended batch size based on VRAM
        let recommended_batch_size = if vram_bytes >= 24 * 1024 * 1024 * 1024 {
            64 // 24GB+ VRAM
        } else if vram_bytes >= 12 * 1024 * 1024 * 1024 {
            32 // 12GB+ VRAM
        } else if vram_bytes >= 8 * 1024 * 1024 * 1024 {
            16 // 8GB+ VRAM
        } else if vram_bytes >= 4 * 1024 * 1024 * 1024 {
            8 // 4GB+ VRAM
        } else if gpu_available {
            4 // GPU with limited VRAM
        } else {
            8 // CPU default
        };

        GpuCapabilities {
            name,
            backend_type,
            arch,
            vram_bytes,
            gpu_available,
            recommended_batch_size,
            backend_name: backend_type.name().to_string(),
        }
    }

    /// Get VRAM size in bytes.
    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    /// Check if GPU is available and usable.
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Generate a unique fingerprint for this device configuration.
    ///
    /// Used for cache invalidation when hardware changes.
    pub fn fingerprint(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        self.backend_type.as_str().hash(&mut hasher);
        self.arch.hash(&mut hasher);
        self.vram_bytes.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

/// Detect device name and VRAM for the given backend.
fn detect_device_details(backend_type: BackendType) -> (String, u64) {
    match backend_type {
        BackendType::Cuda => detect_cuda_details(),
        BackendType::Rocm => detect_rocm_details(),
        BackendType::Metal => detect_metal_details(),
        BackendType::Wgpu => detect_wgpu_details(),
        BackendType::Cpu => ("CPU".to_string(), 0),
    }
}

/// Detect CUDA device details (runtime via cudarc dynamic loading).
fn detect_cuda_details() -> (String, u64) {
    use cudarc::driver::result;

    if result::init().is_ok() {
        if let Ok(dev) = result::device::get(0) {
            let name = result::device::get_name(dev).unwrap_or_else(|_| "NVIDIA GPU".to_string());
            let vram = unsafe { result::device::total_mem(dev).unwrap_or(0) as u64 };
            return (name, vram);
        }
    }
    ("NVIDIA GPU (not available)".to_string(), 0)
}

/// Detect ROCm device details (Linux only).
#[cfg(target_os = "linux")]
fn detect_rocm_details() -> (String, u64) {
    // Try to read from sysfs
    let name = std::fs::read_to_string("/sys/class/drm/card0/device/product_name")
        .or_else(|_| std::fs::read_to_string("/sys/class/drm/card0/device/name"))
        .unwrap_or_else(|_| "AMD GPU".to_string())
        .trim()
        .to_string();

    // Try to get VRAM from sysfs
    let vram = std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_total")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(8 * 1024 * 1024 * 1024); // Default 8GB

    (name, vram)
}

#[cfg(not(target_os = "linux"))]
fn detect_rocm_details() -> (String, u64) {
    ("ROCm (not available)".to_string(), 0)
}

/// Detect Metal device details (macOS only).
#[cfg(target_os = "macos")]
fn detect_metal_details() -> (String, u64) {
    use metal::Device;

    if let Some(device) = Device::system_default() {
        let name = device.name().to_string();
        // Metal doesn't expose VRAM directly, estimate based on device
        let vram = if name.contains("M3 Max") || name.contains("M2 Ultra") {
            128 * 1024 * 1024 * 1024 // 128GB unified memory
        } else if name.contains("M3 Pro") || name.contains("M2 Max") {
            64 * 1024 * 1024 * 1024 // 64GB
        } else if name.contains("M2 Pro") || name.contains("M1 Max") {
            32 * 1024 * 1024 * 1024 // 32GB
        } else if name.contains("M2") || name.contains("M1 Pro") {
            16 * 1024 * 1024 * 1024 // 16GB
        } else {
            8 * 1024 * 1024 * 1024 // 8GB default
        };
        return (name, vram);
    }
    ("Apple GPU".to_string(), 8 * 1024 * 1024 * 1024)
}

#[cfg(not(target_os = "macos"))]
fn detect_metal_details() -> (String, u64) {
    ("Metal (not available)".to_string(), 0)
}

/// Detect WGPU device details (cross-platform, always available).
fn detect_wgpu_details() -> (String, u64) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) {
        let info = adapter.get_info();
        let name = info.name.clone();

        // Get memory limits
        let limits = adapter.limits();
        // WGPU doesn't expose total VRAM, use max buffer size as estimate
        let vram = limits.max_buffer_size;

        return (name, vram);
    }
    ("WGPU Device".to_string(), 4 * 1024 * 1024 * 1024) // Default 4GB
}

/// Internal backend detection (without caching).
fn detect_backend_impl() -> BackendType {
    if try_cuda() {
        BackendType::Cuda
    } else if try_rocm() {
        BackendType::Rocm
    } else if try_metal() {
        BackendType::Metal
    } else if try_wgpu() {
        BackendType::Wgpu
    } else {
        BackendType::Cpu
    }
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

/// Get device ID for a backend (runtime detection, no feature flags).
fn get_device_id(backend_type: BackendType) -> Option<String> {
    match backend_type {
        BackendType::Cuda => Some("cuda:0".to_string()),
        BackendType::Rocm => Some("rocm:0".to_string()),
        BackendType::Metal => Some("metal:default".to_string()),
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

// Backend-specific detection functions (runtime, no feature flags)

/// Try to initialize CUDA via cudarc's dynamic loading.
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

/// Try to detect ROCm (AMD GPU) via /dev/kfd.
#[cfg(target_os = "linux")]
fn try_rocm() -> bool {
    if std::path::Path::new("/dev/kfd").exists() {
        log::info!("ROCm backend detected");
        true
    } else {
        log::debug!("ROCm not available: /dev/kfd not found");
        false
    }
}

#[cfg(not(target_os = "linux"))]
fn try_rocm() -> bool {
    false
}

/// Try to detect Metal (macOS GPU).
#[cfg(target_os = "macos")]
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

#[cfg(not(target_os = "macos"))]
fn try_metal() -> bool {
    false
}

/// Try to detect WGPU (cross-platform GPU).
fn try_wgpu() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }));

    if adapter.is_ok() {
        log::info!("WGPU backend available");
        true
    } else {
        log::debug!("WGPU not available");
        false
    }
}

/// Detect device architecture for kernel selection.
fn detect_device_arch(backend_type: BackendType) -> Option<String> {
    match backend_type {
        BackendType::Cuda => detect_cuda_arch(),
        BackendType::Rocm => detect_rocm_arch(),
        BackendType::Metal => detect_metal_arch(),
        BackendType::Wgpu => detect_wgpu_arch(),
        BackendType::Cpu => None,
    }
}

/// Detect CUDA GPU architecture (compute capability).
fn detect_cuda_arch() -> Option<String> {
    use cudarc::driver::{result, sys};

    if let Ok(dev) = result::device::get(0) {
        let major = unsafe {
            result::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).ok()?
        };
        let minor = unsafe {
            result::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).ok()?
        };
        Some(format!("sm_{}{}", major, minor))
    } else {
        None
    }
}

/// Detect AMD GPU architecture.
#[cfg(target_os = "linux")]
fn detect_rocm_arch() -> Option<String> {
    // Read from sysfs
    std::fs::read_to_string("/sys/class/kfd/kfd/topology/nodes/0/name")
        .ok()
        .map(|s| s.trim().to_string())
        .or_else(|| Some("gfx1030".to_string())) // Default RDNA2
}

#[cfg(not(target_os = "linux"))]
fn detect_rocm_arch() -> Option<String> {
    None
}

/// Detect Apple Silicon architecture.
#[cfg(target_os = "macos")]
fn detect_metal_arch() -> Option<String> {
    use metal::Device;

    if let Some(device) = Device::system_default() {
        let name = device.name();
        if name.contains("M1") {
            Some("apple-m1".to_string())
        } else if name.contains("M2") {
            Some("apple-m2".to_string())
        } else if name.contains("M3") {
            Some("apple-m3".to_string())
        } else {
            Some("apple-m1".to_string())
        }
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_metal_arch() -> Option<String> {
    None
}

/// Detect WGPU adapter architecture.
fn detect_wgpu_arch() -> Option<String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).ok()?;

    let info = adapter.get_info();
    Some(format!("{:?}-{}", info.backend, info.device_type as u32))
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
///     eprintln!("Kernel verification failed: {}", e);
/// }
/// ```
///
/// Note: With Fat Binary architecture, all kernels are embedded at compile time.
/// This function only verifies backend availability, no downloading is needed.
pub fn ensure_kernels() -> Result<(), Box<dyn std::error::Error>> {
    // Fat Binary 架构：所有 kernel 在编译时已嵌入
    // 此函数仅验证后端可用性，无需下载
    let detection = detect_backend();
    log::info!(
        "Backend detection: {} (kernels embedded at compile time)",
        detection.name()
    );
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
