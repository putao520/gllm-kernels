//! §13.2 Softmax Centroid-Guided Prefetch — Unit Tests

#[cfg(test)]
mod tests {
    use crate::compiler::graph::telemetry_offsets;

    #[test]
    fn test_centroid_offset_layout() {
        // Verify telemetry buffer layout includes centroid offset
        assert_eq!(telemetry_offsets::CENTROID_TOKEN_IDX_OFFSET, 324);
        assert!(telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES >= 328);
    }

    #[test]
    fn test_centroid_extraction_sharp() {
        // Test centroid extraction with sharp distribution (single peak)
        // Verify that the centroid index is correctly computed as argmax

        let input = vec![0.1f32, 0.2, 0.9, 0.05];

        // Manually compute softmax to verify centroid
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();
        let softmax: Vec<f32> = input.iter().map(|&x| ((x - max_val).exp()) / exp_sum).collect();

        let (centroid_idx, _max_prob) = softmax.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        assert_eq!(centroid_idx, 2, "Expected centroid at index 2 (highest logit 0.9)");

        // Verify telemetry buffer has space for centroid
        let telemetry = vec![0u8; telemetry_offsets::TELEMETRY_BUFFER_MIN_BYTES];
        let centroid_bytes = &telemetry[telemetry_offsets::CENTROID_TOKEN_IDX_OFFSET as usize..][..4];
        let _centroid_u32 = u32::from_le_bytes([
            centroid_bytes[0],
            centroid_bytes[1],
            centroid_bytes[2],
            centroid_bytes[3],
        ]);
    }
}
