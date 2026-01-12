#[cfg(any(feature = "cuda", feature = "cpu"))]
use gllm_kernels::backend::DefaultBackend;

#[cfg(any(feature = "cuda", feature = "cpu"))]
use gllm_kernels::{default_device, select_device};

#[cfg(feature = "cuda")]
#[test]
fn default_backend_is_cuda() {
    let name = std::any::type_name::<DefaultBackend>();
    assert!(name.to_lowercase().contains("cuda"), "type name: {name}");
    let _ = default_device();
    let _ = select_device::<DefaultBackend>();
}

#[cfg(all(feature = "cpu", not(feature = "cuda")))]
#[test]
fn default_backend_is_cpu() {
    let name = std::any::type_name::<DefaultBackend>();
    assert!(name.to_lowercase().contains("ndarray"), "type name: {name}");
    let _ = default_device();
    let _ = select_device::<DefaultBackend>();
}
