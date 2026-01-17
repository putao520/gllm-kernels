//! Embedded pre-compiled GPU kernels for Fat Binary distribution.
//!
//! # Single-File Library Architecture (ADR-001)
//!
//! Each platform/architecture has ONE file containing ALL kernels for that target.
//! This simplifies loading, enables code sharing, and reduces file count.
//!
//! ## File Structure
//!
//! ```text
//! kernels/
//! ├── cuda/
//! │   ├── sm_86.fatbin    # All 7 CUDA kernels for Ampere
//! │   ├── sm_89.fatbin    # All 7 CUDA kernels for Ada Lovelace
//! │   ├── sm_90.fatbin    # All 7 CUDA kernels for Hopper
//! │   └── fallback.ptx    # All 7 CUDA kernels as PTX
//! ├── rocm/
//! │   ├── gfx90a.hsaco    # All 11 HIP kernels for MI200
//! │   ├── gfx1100.hsaco   # All 11 HIP kernels for RDNA3
//! │   └── gfx1201.hsaco   # All 11 HIP kernels for RDNA4
//! ├── metal/
//! │   └── universal.metallib  # All 4 Metal kernels
//! └── spirv/
//!     └── universal.spv   # All 11 SPIR-V kernels
//! ```
//!
//! ## Supported Architectures (2026)
//!
//! | Platform | Files | Kernels per File |
//! |----------|-------|------------------|
//! | CUDA | 4 (3 fatbin + 1 ptx) | 7 |
//! | ROCm | 3 hsaco | 11 |
//! | Metal | 1 metallib | 4 |
//! | SPIR-V | 1 spv | 11 |

/// CUDA architecture identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaArch {
    /// Ampere: RTX 30xx Ti/Super, A10/A40
    Sm86,
    /// Ada Lovelace: RTX 40xx, L4/L40
    Sm89,
    /// Hopper: H100, H200
    Sm90,
    /// PTX fallback for forward compatibility
    Ptx,
}

impl CudaArch {
    /// Get compute capability as (major, minor)
    pub const fn compute_capability(&self) -> (u32, u32) {
        match self {
            Self::Sm86 => (8, 6),
            Self::Sm89 => (8, 9),
            Self::Sm90 => (9, 0),
            Self::Ptx => (8, 6), // PTX compiled for minimum supported
        }
    }

    /// Select best architecture for given compute capability
    pub fn select(major: u32, minor: u32) -> Self {
        let cc = major * 10 + minor;
        if cc >= 90 {
            Self::Sm90
        } else if cc >= 89 {
            Self::Sm89
        } else if cc >= 86 {
            Self::Sm86
        } else {
            Self::Ptx // Fallback for older GPUs
        }
    }

    /// Get file name for this architecture
    pub const fn file_name(&self) -> &'static str {
        match self {
            Self::Sm86 => "sm_86.fatbin",
            Self::Sm89 => "sm_89.fatbin",
            Self::Sm90 => "sm_90.fatbin",
            Self::Ptx => "fallback.ptx",
        }
    }

    /// Get architecture string
    pub const fn arch_string(&self) -> &'static str {
        match self {
            Self::Sm86 => "sm_86",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
            Self::Ptx => "ptx",
        }
    }
}

/// ROCm/HIP architecture identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RocmArch {
    /// MI200 series (CDNA2)
    Gfx90a,
    /// RDNA3: RX 7000 series
    Gfx1100,
    /// RDNA4: RX 8000 series (2025)
    Gfx1201,
}

impl RocmArch {
    /// Select best architecture for given GCN architecture string
    pub fn select(gcn_arch: &str) -> Option<Self> {
        match gcn_arch {
            s if s.starts_with("gfx120") => Some(Self::Gfx1201),
            s if s.starts_with("gfx110") => Some(Self::Gfx1100),
            s if s.starts_with("gfx90") => Some(Self::Gfx90a),
            _ => None,
        }
    }

    /// Get file name for this architecture
    pub const fn file_name(&self) -> &'static str {
        match self {
            Self::Gfx90a => "gfx90a.hsaco",
            Self::Gfx1100 => "gfx1100.hsaco",
            Self::Gfx1201 => "gfx1201.hsaco",
        }
    }

    /// Get architecture string
    pub const fn arch_string(&self) -> &'static str {
        match self {
            Self::Gfx90a => "gfx90a",
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1201 => "gfx1201",
        }
    }
}

/// Kernel type identifier (for function lookup within library)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// Chunked prefill for efficient batching
    ChunkedPrefill,
    /// Eagle3 speculative decoding (NeurIPS'25)
    Eagle3,
    /// Embedding operations (lookup, pooling)
    EmbeddingOps,
    /// EVIC cache eviction with pressure
    EvicPress,
    /// Flash Attention v2/v3
    FlashAttention,
    /// Flash Tree Attention for hierarchical structures
    FlashTreeAttn,
    /// Fused QKV attention projection
    FusedQkvAttention,
    /// INT2 quantization for extreme compression
    Int2Quantizer,
    /// Medusa multi-head speculative decoding (ICML'24)
    Medusa,
    /// Online softmax with Kahan summation
    OnlineSoftmax,
    /// Paged attention for KV cache
    PagedAttention,
    /// Prompt caching optimization
    PromptCache,
    /// Selective scan (Mamba SSM)
    SelectiveScan,
    /// Speculative execution engine
    SpecEe,
    /// Tiled attention for large contexts
    TiledAttention,
}

impl KernelType {
    /// Get function name within the library
    pub const fn function_name(&self) -> &'static str {
        match self {
            Self::ChunkedPrefill => "chunked_prefill",
            Self::Eagle3 => "eagle3",
            Self::EmbeddingOps => "embedding_ops",
            Self::EvicPress => "evic_press",
            Self::FlashAttention => "flash_attention",
            Self::FlashTreeAttn => "flash_tree_attn",
            Self::FusedQkvAttention => "fused_qkv_attention",
            Self::Int2Quantizer => "int2_quantizer",
            Self::Medusa => "medusa",
            Self::OnlineSoftmax => "online_softmax",
            Self::PagedAttention => "paged_attention",
            Self::PromptCache => "prompt_cache",
            Self::SelectiveScan => "selective_scan",
            Self::SpecEe => "spec_ee",
            Self::TiledAttention => "tiled_attention",
        }
    }

    /// Check if this kernel is available on CUDA
    pub const fn available_on_cuda(&self) -> bool {
        matches!(
            self,
            Self::EmbeddingOps
                | Self::FlashAttention
                | Self::FusedQkvAttention
                | Self::OnlineSoftmax
                | Self::PagedAttention
                | Self::SelectiveScan
                | Self::TiledAttention
        )
    }

    /// Check if this kernel is available on ROCm/HIP
    pub const fn available_on_rocm(&self) -> bool {
        matches!(
            self,
            Self::ChunkedPrefill
                | Self::Eagle3
                | Self::EmbeddingOps
                | Self::EvicPress
                | Self::FlashAttention
                | Self::FlashTreeAttn
                | Self::Int2Quantizer
                | Self::Medusa
                | Self::PagedAttention
                | Self::PromptCache
                | Self::SpecEe
        )
    }

    /// Check if this kernel is available on Metal
    pub const fn available_on_metal(&self) -> bool {
        matches!(
            self,
            Self::Eagle3 | Self::EmbeddingOps | Self::FlashAttention | Self::PagedAttention
        )
    }

    /// Check if this kernel is available on SPIR-V/WebGPU
    pub const fn available_on_spirv(&self) -> bool {
        matches!(
            self,
            Self::ChunkedPrefill
                | Self::Eagle3
                | Self::EmbeddingOps
                | Self::EvicPress
                | Self::FlashAttention
                | Self::FlashTreeAttn
                | Self::Int2Quantizer
                | Self::Medusa
                | Self::PagedAttention
                | Self::PromptCache
                | Self::SpecEe
        )
    }
}

// ============================================================================
// Kernel Library Loading (Single-File Architecture)
// ============================================================================

/// Kernel loading error
#[derive(Debug, Clone)]
pub enum KernelLoadError {
    /// Kernel library not found for platform/architecture
    LibraryNotFound {
        platform: String,
        arch: String,
    },
    /// Specific kernel function not found in library
    FunctionNotFound {
        kernel: String,
        platform: String,
    },
    /// Architecture not supported
    UnsupportedArchitecture(String),
    /// Download failed (if download-kernels feature enabled)
    DownloadFailed(String),
    /// Native backend (CPU) doesn't use kernel libraries
    NativeBackend,
}

impl std::fmt::Display for KernelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryNotFound { platform, arch } => {
                write!(f, "Kernel library not found: {}/{}", platform, arch)
            }
            Self::FunctionNotFound { kernel, platform } => {
                write!(f, "Kernel function '{}' not found in {} library", kernel, platform)
            }
            Self::UnsupportedArchitecture(arch) => {
                write!(f, "Unsupported architecture: {}", arch)
            }
            Self::DownloadFailed(msg) => write!(f, "Download failed: {}", msg),
            Self::NativeBackend => write!(f, "CPU backend doesn't use kernel libraries"),
        }
    }
}

impl std::error::Error for KernelLoadError {}

/// Get the kernel library for CUDA architecture
pub fn get_cuda_library(arch: CudaArch) -> Result<&'static [u8], KernelLoadError> {
    // Try embedded first
    #[cfg(feature = "embedded-kernels")]
    if let Some(bytes) = try_embedded_cuda(arch) {
        return Ok(bytes);
    }

    // Try cache
    if let Some(bytes) = try_cache_cuda(arch) {
        // Note: This returns owned Vec, but for simplicity we'd need
        // a different approach for cached data. For now, embedded is primary.
        let _ = bytes;
    }

    Err(KernelLoadError::LibraryNotFound {
        platform: "cuda".to_string(),
        arch: arch.arch_string().to_string(),
    })
}

/// Get the kernel library for ROCm architecture
pub fn get_rocm_library(arch: RocmArch) -> Result<&'static [u8], KernelLoadError> {
    #[cfg(feature = "embedded-kernels")]
    if let Some(bytes) = try_embedded_rocm(arch) {
        return Ok(bytes);
    }

    Err(KernelLoadError::LibraryNotFound {
        platform: "rocm".to_string(),
        arch: arch.arch_string().to_string(),
    })
}

/// Get the Metal kernel library (universal)
#[cfg(target_os = "macos")]
pub fn get_metal_library() -> Result<&'static [u8], KernelLoadError> {
    #[cfg(feature = "embedded-kernels")]
    if let Some(bytes) = try_embedded_metal() {
        return Ok(bytes);
    }

    Err(KernelLoadError::LibraryNotFound {
        platform: "metal".to_string(),
        arch: "universal".to_string(),
    })
}

/// Get the SPIR-V kernel library (universal)
pub fn get_spirv_library() -> Result<&'static [u8], KernelLoadError> {
    #[cfg(feature = "embedded-kernels")]
    if let Some(bytes) = try_embedded_spirv() {
        return Ok(bytes);
    }

    Err(KernelLoadError::LibraryNotFound {
        platform: "spirv".to_string(),
        arch: "universal".to_string(),
    })
}

// ============================================================================
// Embedded Kernels (only when feature enabled)
// ============================================================================

#[cfg(feature = "embedded-kernels")]
fn try_embedded_cuda(arch: CudaArch) -> Option<&'static [u8]> {
    // Note: In actual release builds, these would be populated via include_bytes!
    // For now, return None as kernels are not yet compiled
    let _ = arch;

    // Example of what this would look like when kernels exist:
    // match arch {
    //     CudaArch::Sm86 => Some(include_bytes!("../kernels/cuda/sm_86.fatbin")),
    //     CudaArch::Sm89 => Some(include_bytes!("../kernels/cuda/sm_89.fatbin")),
    //     CudaArch::Sm90 => Some(include_bytes!("../kernels/cuda/sm_90.fatbin")),
    //     CudaArch::Ptx => Some(include_bytes!("../kernels/cuda/fallback.ptx")),
    // }

    None
}

#[cfg(feature = "embedded-kernels")]
fn try_embedded_rocm(arch: RocmArch) -> Option<&'static [u8]> {
    let _ = arch;
    // Example:
    // match arch {
    //     RocmArch::Gfx90a => Some(include_bytes!("../kernels/rocm/gfx90a.hsaco")),
    //     RocmArch::Gfx1100 => Some(include_bytes!("../kernels/rocm/gfx1100.hsaco")),
    //     RocmArch::Gfx1201 => Some(include_bytes!("../kernels/rocm/gfx1201.hsaco")),
    // }
    None
}

#[cfg(all(feature = "embedded-kernels", target_os = "macos"))]
fn try_embedded_metal() -> Option<&'static [u8]> {
    // Example:
    // Some(include_bytes!("../kernels/metal/universal.metallib"))
    None
}

#[cfg(feature = "embedded-kernels")]
fn try_embedded_spirv() -> Option<&'static [u8]> {
    // Example:
    // Some(include_bytes!("../kernels/spirv/universal.spv"))
    None
}

fn try_cache_cuda(arch: CudaArch) -> Option<Vec<u8>> {
    let cache_dir = dirs::cache_dir()?.join("gllm-kernels");
    let version = env!("CARGO_PKG_VERSION");
    let path = cache_dir
        .join(version)
        .join("cuda")
        .join(arch.file_name());
    std::fs::read(&path).ok()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_arch_select() {
        assert_eq!(CudaArch::select(9, 0), CudaArch::Sm90);
        assert_eq!(CudaArch::select(8, 9), CudaArch::Sm89);
        assert_eq!(CudaArch::select(8, 6), CudaArch::Sm86);
        assert_eq!(CudaArch::select(7, 5), CudaArch::Ptx);
    }

    #[test]
    fn test_cuda_arch_file_names() {
        assert_eq!(CudaArch::Sm86.file_name(), "sm_86.fatbin");
        assert_eq!(CudaArch::Sm89.file_name(), "sm_89.fatbin");
        assert_eq!(CudaArch::Sm90.file_name(), "sm_90.fatbin");
        assert_eq!(CudaArch::Ptx.file_name(), "fallback.ptx");
    }

    #[test]
    fn test_rocm_arch_select() {
        assert_eq!(RocmArch::select("gfx90a"), Some(RocmArch::Gfx90a));
        assert_eq!(RocmArch::select("gfx1100"), Some(RocmArch::Gfx1100));
        assert_eq!(RocmArch::select("gfx1201"), Some(RocmArch::Gfx1201));
        assert_eq!(RocmArch::select("unknown"), None);
    }

    #[test]
    fn test_rocm_arch_file_names() {
        assert_eq!(RocmArch::Gfx90a.file_name(), "gfx90a.hsaco");
        assert_eq!(RocmArch::Gfx1100.file_name(), "gfx1100.hsaco");
        assert_eq!(RocmArch::Gfx1201.file_name(), "gfx1201.hsaco");
    }

    #[test]
    fn test_kernel_type_function_name() {
        assert_eq!(KernelType::FlashAttention.function_name(), "flash_attention");
        assert_eq!(KernelType::Eagle3.function_name(), "eagle3");
        assert_eq!(KernelType::PagedAttention.function_name(), "paged_attention");
    }

    #[test]
    fn test_kernel_platform_availability() {
        // FlashAttention is available everywhere
        assert!(KernelType::FlashAttention.available_on_cuda());
        assert!(KernelType::FlashAttention.available_on_rocm());
        assert!(KernelType::FlashAttention.available_on_metal());
        assert!(KernelType::FlashAttention.available_on_spirv());

        // SelectiveScan is CUDA only
        assert!(KernelType::SelectiveScan.available_on_cuda());
        assert!(!KernelType::SelectiveScan.available_on_rocm());
        assert!(!KernelType::SelectiveScan.available_on_metal());
        assert!(!KernelType::SelectiveScan.available_on_spirv());

        // Medusa is ROCm/SPIRV but not CUDA/Metal
        assert!(!KernelType::Medusa.available_on_cuda());
        assert!(KernelType::Medusa.available_on_rocm());
        assert!(!KernelType::Medusa.available_on_metal());
        assert!(KernelType::Medusa.available_on_spirv());
    }
}
