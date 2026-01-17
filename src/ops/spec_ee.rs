//! SpecEE / LayerSkip Early-Exit Speculative Decoding.
//!
//! Based on:
//! - SpecEE (ISCA'25): 2.25-2.43x acceleration through early exit speculation
//! - LayerSkip (ACL'24): Self-speculative decoding using layer dropout
//!
//! # Key Features
//! - Per-layer early exit heads with confidence prediction
//! - Layer dropout training for early exit robustness
//! - Shared activation optimization between draft and verify phases
//! - Three-level predictor: algorithm + system + mapping

#[path = "spec_ee/cache.rs"]
mod cache;
#[path = "spec_ee/config.rs"]
mod config;
#[path = "spec_ee/dropout.rs"]
mod dropout;
#[path = "spec_ee/engine.rs"]
mod engine;
#[path = "spec_ee/head.rs"]
mod head;
#[path = "spec_ee/stats.rs"]
mod stats;
#[path = "spec_ee/types.rs"]
mod types;

pub use cache::{Activation2, Activation3, SharedActivations};
pub use config::SpecEEConfig;
pub use dropout::LayerDropoutSchedule;
pub use engine::SpecEEEngine;
pub use head::EarlyExitHead;
pub use stats::SpecEEStats;
pub use types::{EarlyExitDecision, SelfSpeculationResult, self_speculate};

#[cfg(all(test, feature = "cpu"))]
#[path = "spec_ee/tests.rs"]
mod tests;
