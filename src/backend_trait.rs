//! Backend selector trait for runtime backend choice.

pub trait BackendSelector: Send + Sync {
    fn name(&self) -> &'static str;
}
