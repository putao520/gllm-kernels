use super::*;

#[test]
fn test_assisted_generation_config() {
    let config = AssistedGenerationConfig::default();
    assert!(config.validate().is_ok());

    let invalid = AssistedGenerationConfig {
        num_medusa_heads: 0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    let invalid2 = AssistedGenerationConfig {
        vocab_size: 0,
        ..Default::default()
    };
    assert!(invalid2.validate().is_err());
}

#[test]
fn test_medusa_head_creation() {
    let head = MedusaHead::<f32>::new(64, 100, 1, 0.0);
    assert!(head.is_ok());
    let head = head.unwrap();
    assert_eq!(head.hidden_dim(), 64);
    assert_eq!(head.vocab_size(), 100);
    assert_eq!(head.position_offset(), 1);
}

#[test]
fn test_medusa_head_forward() {
    let head = MedusaHead::<f32>::new(64, 100, 1, 0.0).unwrap();

    let batch = 1;
    let seq_len = 4;
    let hidden_dim = 64;
    let vocab_size = 100;

    // Create hidden states with some variation
    let mut hidden = vec![0.0f32; batch * seq_len * hidden_dim];
    for i in 0..hidden.len() {
        hidden[i] = ((i % 13) as f32 - 6.0) * 0.1;
    }

    let mut logits = vec![0.0f32; batch * seq_len * vocab_size];

    let result = head.forward(&hidden, batch, seq_len, &mut logits);
    assert!(result.is_ok());

    // Verify logits are computed (not all zeros)
    let non_zero_count = logits.iter().filter(|&&x| x != 0.0).count();
    assert!(non_zero_count > 0);
}

#[test]
fn test_medusa_head_get_candidates() {
    let head = MedusaHead::<f32>::new(64, 100, 1, 0.0).unwrap();

    let batch = 1;
    let seq_len = 4;
    let hidden_dim = 64;

    let mut hidden = vec![0.0f32; batch * seq_len * hidden_dim];
    for i in 0..hidden.len() {
        hidden[i] = ((i % 13) as f32 - 6.0) * 0.1;
    }

    let result = head.get_candidates(&hidden, batch, seq_len, 5);
    assert!(result.is_ok());

    let candidates = result.unwrap();
    assert_eq!(candidates.len(), batch);
    assert!(candidates[0].len() <= 5);

    // Verify candidates are sorted by log_prob (descending)
    for i in 1..candidates[0].len() {
        assert!(candidates[0][i - 1].1 >= candidates[0][i].1);
    }
}

#[test]
fn test_ngram_cache() {
    let mut cache = NgramCache::new(3, 1000);

    // Add some sequences
    cache.update(&[1, 2, 3, 4, 5]);
    cache.update(&[1, 2, 3, 6, 7]);
    cache.update(&[1, 2, 3, 4, 8]);

    // Predict next token after [2, 3]
    let predictions = cache.predict(&[1, 2, 3], 3);
    assert!(!predictions.is_empty());
    // Token 4 should be most common
    assert_eq!(predictions[0], 4);
}

#[test]
fn test_ngram_cache_empty() {
    let cache = NgramCache::new(3, 1000);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    let predictions = cache.predict(&[1, 2], 3);
    assert!(predictions.is_empty());
}

#[test]
fn test_medusa_engine_creation() {
    let config = AssistedGenerationConfig {
        hidden_dim: 64,
        vocab_size: 100,
        num_medusa_heads: 2,
        ..Default::default()
    };

    let engine = MedusaEngine::<f32>::new(config);
    assert!(engine.is_ok());
}

#[test]
fn test_medusa_engine_draft_generation() {
    let config = AssistedGenerationConfig {
        hidden_dim: 64,
        vocab_size: 100,
        num_medusa_heads: 2,
        candidate_count: 4,
        use_ngram_draft: false,
        ..Default::default()
    };

    let engine = MedusaEngine::<f32>::new(config).unwrap();

    let batch = 1;
    let seq_len = 4;
    let hidden_dim = 64;
    let vocab_size = 100;

    let mut hidden = vec![0.0f32; batch * seq_len * hidden_dim];
    for i in 0..hidden.len() {
        hidden[i] = ((i % 13) as f32 - 6.0) * 0.1;
    }

    let mut main_logits = vec![0.0f32; batch * seq_len * vocab_size];
    for i in 0..main_logits.len() {
        main_logits[i] = ((i % 17) as f32 - 8.0) * 0.1;
    }

    let draft = engine.generate_draft(&hidden, batch, seq_len, &main_logits, &[]);
    assert!(draft.is_ok());

    let draft = draft.unwrap();
    assert!(draft.candidates.len() > 0);
}

#[test]
fn test_build_candidate_tree() {
    let draft = MedusaDraft {
        root_token: 0,
        root_log_prob: 0.0,
        candidates: vec![
            vec![
                MedusaCandidate {
                    token_id: 1,
                    log_prob: -0.1,
                    head_idx: Some(0),
                    position_offset: 1,
                },
                MedusaCandidate {
                    token_id: 2,
                    log_prob: -0.2,
                    head_idx: Some(0),
                    position_offset: 1,
                },
            ],
            vec![MedusaCandidate {
                token_id: 3,
                log_prob: -0.1,
                head_idx: Some(1),
                position_offset: 2,
            }],
        ],
        num_paths: 2,
    };

    let paths = build_candidate_tree(&draft, 4);
    assert_eq!(paths.len(), 2);
    assert_eq!(paths[0], vec![0, 1, 3]);
    assert_eq!(paths[1], vec![0, 2, 3]);
}

#[test]
fn test_flatten_candidate_tree() {
    let paths = vec![vec![0, 1, 2], vec![0, 3, 4, 5]];

    let (tokens, lengths, starts) = flatten_candidate_tree(&paths);

    assert_eq!(tokens, vec![0, 1, 2, 0, 3, 4, 5]);
    assert_eq!(lengths, vec![3, 4]);
    assert_eq!(starts, vec![0, 3]);
}

#[test]
fn test_stats_tracking() {
    let mut stats = MedusaStats::default();

    stats.update(8, 3, 1);
    stats.update(8, 4, 0);

    assert_eq!(stats.total_draft_tokens, 16);
    assert_eq!(stats.total_accepted, 7);
    assert_eq!(stats.num_rounds, 2);
    assert_eq!(stats.ngram_contributions, 1);
    assert!(stats.estimated_speedup() > 1.0);
}

#[test]
fn test_find_top_token() {
    let logits = vec![1.0, 2.0, 5.0, 1.5, 0.5];
    let (idx, _prob) = find_top_token(&logits);
    assert_eq!(idx, 2); // Index of max value (5.0)
}

#[test]
fn test_medusa_engine_verify_draft() {
    let config = AssistedGenerationConfig {
        hidden_dim: 64,
        vocab_size: 10,
        num_medusa_heads: 2,
        candidate_count: 4,
        use_ngram_draft: false,
        ..Default::default()
    };

    let mut engine = MedusaEngine::<f32>::new(config).unwrap();

    // Create a simple draft
    let draft = MedusaDraft {
        root_token: 1,
        root_log_prob: -0.1,
        candidates: vec![
            vec![
                MedusaCandidate {
                    token_id: 2,
                    log_prob: -0.2,
                    head_idx: Some(0),
                    position_offset: 1,
                },
                MedusaCandidate {
                    token_id: 3,
                    log_prob: -0.3,
                    head_idx: Some(0),
                    position_offset: 1,
                },
            ],
        ],
        num_paths: 2,
    };

    // Create target logits that select token 2 at position 0
    let vocab_size = 10;
    let mut target_logits = vec![0.0f32; vocab_size];
    target_logits[2] = 10.0; // Make token 2 the top token

    let verification = engine.verify_draft(&draft, &target_logits);
    assert!(verification.is_ok());

    let result = verification.unwrap();
    assert_eq!(result.accepted_tokens[0], 1); // Root token
    assert_eq!(result.accepted_tokens[1], 2); // Matched candidate
}
