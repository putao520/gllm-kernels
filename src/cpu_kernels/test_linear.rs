#[cfg(test)]
mod test_linear_computation {
    use crate::cpu_kernels::linear;

    #[test]
    fn test_linear_simple() {
        // input: (m=1, k=3) = [1, 2, 3]
        // weight: (n=2, k=3) stored in row-major as [[10, 20, 30], [40, 50, 60]]
        // Expected output: (1, 2) = input @ weight.T = [140, 320]

        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut output = vec![0.0; 2];

        linear(&input, &weight, None, &mut output, 1, 2, 3).unwrap();

        println!("Input: {:?}", input);
        println!("Weight: {:?}", weight);
        println!("Output: {:?}", output);
        println!("Expected: [140.0, 320.0]");

        assert!((output[0] - 140.0).abs() < 1e-5, "output[0] = {}, expected 140.0", output[0]);
        assert!((output[1] - 320.0).abs() < 1e-5, "output[1] = {}, expected 320.0", output[1]);
    }

    #[test]
    fn test_linear_lm_head_style() {
        // Simulate lm_head: vocab_size=4, hidden_size=3
        // Weight shape: [4, 3] in row-major
        // Each row is the embedding for a token

        // Token embeddings:
        // Token 0: [1, 0, 0]
        // Token 1: [0, 1, 0]
        // Token 2: [0, 0, 1]
        // Token 3: [1, 1, 1]
        let weight = vec![
            1.0, 0.0, 0.0,  // Token 0 embedding
            0.0, 1.0, 0.0,  // Token 1 embedding
            0.0, 0.0, 1.0,  // Token 2 embedding
            1.0, 1.0, 1.0,  // Token 3 embedding
        ];

        // Hidden state: [2, 3, 4]
        let hidden = vec![2.0, 3.0, 4.0];
        let mut logits = vec![0.0; 4];

        linear(&hidden, &weight, None, &mut logits, 1, 4, 3).unwrap();

        println!("Hidden: {:?}", hidden);
        println!("Logits: {:?}", logits);

        // Expected:
        // logit[0] = 2*1 + 3*0 + 4*0 = 2
        // logit[1] = 2*0 + 3*1 + 4*0 = 3
        // logit[2] = 2*0 + 3*0 + 4*1 = 4
        // logit[3] = 2*1 + 3*1 + 4*1 = 9
        assert!((logits[0] - 2.0).abs() < 1e-5, "logits[0] = {}, expected 2.0", logits[0]);
        assert!((logits[1] - 3.0).abs() < 1e-5, "logits[1] = {}, expected 3.0", logits[1]);
        assert!((logits[2] - 4.0).abs() < 1e-5, "logits[2] = {}, expected 4.0", logits[2]);
        assert!((logits[3] - 9.0).abs() < 1e-5, "logits[3] = {}, expected 9.0", logits[3]);
    }
}
