//! Medusa / Assisted Generation for parallel token prediction.
//!
//! Based on:
//! - Medusa (ICML'24): Multiple decode heads for parallel prediction
//! - Lookahead Decoding: N-gram assisted draft generation
//!
//! # Key Features
//! - Multiple Medusa heads for predicting future tokens
//! - N-gram cache for assisted drafting
//! - Tree-structured candidate generation
//! - Compatible with DeFT tree attention for verification

#[path = "medusa/cache.rs"]
mod cache;
#[path = "medusa/config.rs"]
mod config;
#[path = "medusa/engine.rs"]
mod engine;
#[path = "medusa/head.rs"]
mod head;
#[path = "medusa/types.rs"]
mod types;

pub use cache::NgramCache;
pub use config::AssistedGenerationConfig;
pub use engine::MedusaEngine;
pub use head::MedusaHead;
pub use types::{
    build_candidate_tree, find_top_token, flatten_candidate_tree, MedusaCandidate, MedusaDraft,
    MedusaStats, MedusaVerification,
};

#[cfg(all(test, feature = "cpu"))]
#[path = "medusa/tests.rs"]
mod tests;
