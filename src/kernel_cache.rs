//! Kernel compilation cache management.

use std::path::PathBuf;
use std::fs;
use std::io::Write;

/// Get kernel cache directory.
pub fn kernel_cache_dir() -> PathBuf {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());

    PathBuf::from(home_dir)
        .join(".gsc")
        .join("gllm")
        .join("kernels")
}

/// Get cached kernel file path.
///
/// # Arguments
/// * `backend` - Backend name (e.g., "cuda", "rocm", "metal")
/// * `kernel_name` - Kernel name (e.g., "flash_attention")
/// * `device_arch` - Device architecture (e.g., "sm_86", "gfx1100", "apple-m1")
///
/// # Example
/// ```
/// use gllm_kernels::kernel_cache_path;
/// let path = kernel_cache_path("cuda", "flash_attention", "sm_86");
/// // Returns: ~/.gsc/gllm/kernels/cuda/flash_attention/sm_86.bin
/// ```
pub fn kernel_cache_path(backend: &str, kernel_name: &str, device_arch: &str) -> PathBuf {
    kernel_cache_dir()
        .join(backend)
        .join(kernel_name)
        .join(format!("{}.bin", device_arch))
}

/// Load cached kernel binary.
///
/// Returns `None` if cache doesn't exist or is invalid.
pub fn load_cached_kernel(backend: &str, kernel_name: &str, device_arch: &str) -> Option<Vec<u8>> {
    let cache_path = kernel_cache_path(backend, kernel_name, device_arch);

    if !cache_path.exists() {
        log::debug!("Kernel cache miss: {:?}", cache_path);
        return None;
    }

    match fs::read(&cache_path) {
        Ok(data) => {
            log::info!("Loaded cached kernel: {:?}", cache_path);
            Some(data)
        }
        Err(e) => {
            log::warn!("Failed to read kernel cache {:?}: {}", cache_path, e);
            None
        }
    }
}

/// Save compiled kernel to cache.
pub fn save_kernel_to_cache(
    backend: &str,
    kernel_name: &str,
    device_arch: &str,
    kernel_binary: &[u8],
) -> Result<(), std::io::Error> {
    let cache_path = kernel_cache_path(backend, kernel_name, device_arch);

    // Ensure cache directory exists
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Write kernel binary
    let mut file = fs::File::create(&cache_path)?;
    file.write_all(kernel_binary)?;
    file.sync_all()?;

    log::info!("Saved kernel to cache: {:?}", cache_path);
    Ok(())
}

/// Clear all kernel cache.
pub fn clear_kernel_cache() -> Result<(), std::io::Error> {
    let cache_dir = kernel_cache_dir();

    if !cache_dir.exists() {
        return Ok(());
    }

    fs::remove_dir_all(&cache_dir)?;
    log::info!("Cleared kernel cache: {:?}", cache_dir);
    Ok(())
}

/// Get kernel cache size in bytes.
pub fn kernel_cache_size() -> Result<u64, std::io::Error> {
    let cache_dir = kernel_cache_dir();

    if !cache_dir.exists() {
        return Ok(0);
    }

    let mut total_size = 0u64;

    fn dir_size(dir: &PathBuf, total: &mut u64) -> Result<(), std::io::Error> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                dir_size(&path, total)?;
            } else {
                *total += entry.metadata()?.len();
            }
        }
        Ok(())
    }

    dir_size(&cache_dir, &mut total_size)?;
    Ok(total_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_path() {
        let path = kernel_cache_path("cuda", "flash_attention", "sm_86");
        assert!(path.ends_with(".gsc/gllm/kernels/cuda/flash_attention/sm_86.bin"));
    }

    #[test]
    fn test_save_and_load() {
        let test_data = b"test_kernel_binary";
        let backend = "test";
        let kernel = "test_kernel";
        let arch = "test_arch";

        // Save
        save_kernel_to_cache(backend, kernel, arch, test_data).unwrap();

        // Load
        let loaded = load_cached_kernel(backend, kernel, arch).unwrap();
        assert_eq!(loaded, test_data);

        // Cleanup
        let _ = clear_kernel_cache();
    }
}
