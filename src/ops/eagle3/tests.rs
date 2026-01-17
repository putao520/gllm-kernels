use super::*;

#[test]
fn test_adaptive_draft_config_validation() {
    let valid = AdaptiveDraftConfig::default();
    assert!(valid.validate().is_ok());

    let invalid = AdaptiveDraftConfig {
        min_draft_length: 0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    let invalid = AdaptiveDraftConfig {
        max_draft_length: 0,
        min_draft_length: 2,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_confidence_predictor() {
    let predictor = ConfidencePredictor::<f32>::new(64, 4).unwrap();

    let fused_hidden = vec![0.1f32; 64 * 4];
    let confidence = predictor.predict_single(&fused_hidden).unwrap();
    assert!(confidence >= 0.0 && confidence <= 1.0);
}

#[test]
fn test_length_scheduler() {
    let mut scheduler = LengthScheduler::new(1, 8, 0.1).unwrap();

    let initial = scheduler.suggest_length();
    assert!(initial >= 1 && initial <= 8);

    scheduler.update(4, 4);
    scheduler.update(4, 4);
    scheduler.update(4, 4);

    let after_high = scheduler.suggest_length();
    assert!(after_high >= 1);
}

#[test]
fn test_eagle3_decoder_creation() {
    let config = AdaptiveDraftConfig::with_hidden_dim(64);
    let decoder = Eagle3Decoder::<f32>::new(config);
    assert!(decoder.is_ok());
}

#[test]
fn test_fuse_hidden_states() {
    let config = AdaptiveDraftConfig {
        hidden_dim: 8,
        fusion_layers: 2,
        ..Default::default()
    };
    let decoder = Eagle3Decoder::<f32>::new(config).unwrap();

    let batch = 1;
    let seq_len = 4;
    let hidden_dim = 8;
    let fused_dim = hidden_dim * 2;

    let layer1 = vec![0.0f32; batch * seq_len * hidden_dim];
    let layer2 = vec![1.0f32; batch * seq_len * hidden_dim];
    let mut fused = vec![0.0f32; batch * seq_len * fused_dim];

    decoder
        .fuse_hidden_states(&[layer1.as_slice(), layer2.as_slice()], batch, seq_len, &mut fused)
        .unwrap();

    assert_eq!(fused.len(), batch * seq_len * fused_dim);
    assert_eq!(fused[0], 0.0f32);
    assert_eq!(fused[hidden_dim], 1.0f32);
}

#[test]
fn test_generate_and_verify_draft() {
    let config = AdaptiveDraftConfig {
        hidden_dim: 8,
        fusion_layers: 2,
        min_draft_length: 1,
        max_draft_length: 4,
        ..Default::default()
    };
    let decoder = Eagle3Decoder::<f32>::new(config).unwrap();

    let vocab_size = 16;
    let seq_len = 4;
    let fused_dim = 16;

    let mut draft_logits = vec![0.0f32; seq_len * vocab_size];
    for i in 0..seq_len {
        draft_logits[i * vocab_size + (i % vocab_size)] = 2.0;
    }

    let fused_hidden = vec![0.5f32; seq_len * fused_dim];

    let draft = decoder.generate_draft(&draft_logits, &fused_hidden, vocab_size, None);
    assert!(draft.is_ok());
    let draft = draft.unwrap();
    assert!(!draft.tokens.is_empty());

    let mut target_logits = vec![0.0f32; seq_len * vocab_size];
    for i in 0..seq_len {
        target_logits[i * vocab_size + (i % vocab_size)] = 3.0;
    }

    let verification = decoder.verify_draft(&draft, &target_logits, vocab_size);
    assert!(verification.is_ok());
}

#[test]
fn test_stats_tracking() {
    let mut stats = Eagle3Stats::default();

    stats.update(4, 3, false);
    assert_eq!(stats.total_draft_tokens, 4);
    assert_eq!(stats.total_accepted_tokens, 3);
    assert_eq!(stats.num_cycles, 1);

    stats.update(4, 4, false);
    assert_eq!(stats.total_draft_tokens, 8);
    assert_eq!(stats.total_accepted_tokens, 7);
}
