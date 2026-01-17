use super::*;

#[test]
fn test_spec_ee_config_validation() {
    let valid = SpecEEConfig::default();
    assert!(valid.validate().is_ok());

    let invalid = SpecEEConfig {
        exit_layers: vec![],
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    let invalid = SpecEEConfig {
        confidence_threshold: 0.0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_layer_dropout_schedule() {
    let schedule = LayerDropoutSchedule::linear(0.1, 0.5, 24).unwrap();

    assert!((schedule.get_rate(0) - 0.1).abs() < 0.001);
    assert!((schedule.get_rate(23) - 0.5).abs() < 0.001);
    assert!(schedule.get_rate(12) > 0.1 && schedule.get_rate(12) < 0.5);
}

#[test]
fn test_early_exit_head() {
    let head = EarlyExitHead::<f32>::new(64, 100, 6).unwrap();
    let batch = 1;
    let seq_len = 4;
    let hidden = vec![0.0f32; batch * seq_len * 64];
    let mut logits = vec![0.0f32; batch * seq_len * 100];
    let mut confidence = vec![0.0f32; batch * seq_len];

    let result = head.forward(&hidden, batch, seq_len, &mut logits, &mut confidence);
    assert!(result.is_ok());
    assert_eq!(logits.len(), batch * seq_len * 100);
    assert_eq!(confidence.len(), batch * seq_len);
}

#[test]
fn test_shared_activations() {
    let mut cache = SharedActivations::<f32>::new(24).unwrap();

    let batch = 1;
    let seq_len = 4;
    let hidden = vec![0.0f32; batch * seq_len * 64];
    cache.store_hidden(6, &hidden, batch, seq_len).unwrap();

    assert!(cache.get_hidden(6).is_some());
    assert!(cache.get_hidden(12).is_none());

    cache.clear_from(6);
    assert!(cache.get_hidden(6).is_none());
}

#[test]
fn test_spec_ee_engine_creation() {
    let config = SpecEEConfig {
        hidden_dim: 64,
        vocab_size: 100,
        num_layers: 12,
        exit_layers: vec![4, 8],
        min_exit_layer: 4,
        ..Default::default()
    };

    let engine = SpecEEEngine::<f32>::new(config);
    assert!(engine.is_ok());
}

#[test]
fn test_self_speculation() {
    let result = self_speculate(42, 12, 42, 24);
    assert_eq!(result.early_exit_accepted, 1);
    assert!(result.bonus_token.is_none());
    assert_eq!(result.layers_saved, 11);

    let result = self_speculate(42, 12, 100, 24);
    assert_eq!(result.early_exit_accepted, 0);
    assert_eq!(result.bonus_token, Some(100));
    assert_eq!(result.layers_saved, 0);
}

#[test]
fn test_stats_tracking() {
    let mut stats = SpecEEStats::new(24);

    stats.record_early_exit(6, true);
    stats.record_early_exit(12, true);
    stats.record_early_exit(6, false);

    assert_eq!(stats.total_early_exits, 3);
    assert_eq!(stats.accepted_count, 2);
    assert_eq!(stats.rejected_count, 1);
    assert!(stats.acceptance_rate > 0.6);
}
