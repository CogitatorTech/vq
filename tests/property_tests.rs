//! Property-based tests for the vq crate using proptest.

use proptest::prelude::*;
use vq::{BinaryQuantizer, Distance, ProductQuantizer, Quantizer, ScalarQuantizer, TSVQ};

// =============================================================================
// Strategies for generating test data
// =============================================================================

/// Generate a vector of f32 values within a specified range.
fn vec_f32(len: usize, min: f32, max: f32) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(min..max, len)
}

/// Generate a non-empty vector of f32 values.
fn non_empty_vec_f32(max_len: usize, min: f32, max: f32) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(min..max, 1..=max_len)
}

/// Generate training data: a collection of vectors with the same dimension.
fn training_data(
    n_vectors: usize,
    dim: usize,
    min: f32,
    max: f32,
) -> impl Strategy<Value = Vec<Vec<f32>>> {
    prop::collection::vec(vec_f32(dim, min, max), n_vectors)
}

// =============================================================================
// Binary Quantizer Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: BQ output length equals input length
    #[test]
    fn prop_bq_output_length_equals_input(
        input in non_empty_vec_f32(100, -1000.0, 1000.0),
        threshold in -100.0f32..100.0,
    ) {
        let bq = BinaryQuantizer::new(threshold, 0, 1).unwrap();
        let output = bq.quantize(&input).unwrap();
        prop_assert_eq!(output.len(), input.len());
    }

    /// Property: BQ output contains only low or high values
    #[test]
    fn prop_bq_output_is_binary(
        input in non_empty_vec_f32(100, -1000.0, 1000.0),
        threshold in -100.0f32..100.0,
    ) {
        let bq = BinaryQuantizer::new(threshold, 0, 1).unwrap();
        let output = bq.quantize(&input).unwrap();
        for val in output {
            prop_assert!(val == 0 || val == 1);
        }
    }

    /// Property: BQ is deterministic (same input produces same output)
    #[test]
    fn prop_bq_deterministic(
        input in non_empty_vec_f32(50, -100.0, 100.0),
        threshold in -50.0f32..50.0,
    ) {
        let bq = BinaryQuantizer::new(threshold, 0, 1).unwrap();
        let output1 = bq.quantize(&input).unwrap();
        let output2 = bq.quantize(&input).unwrap();
        prop_assert_eq!(output1, output2);
    }

    /// Property: BQ correctly classifies values above/below threshold
    #[test]
    fn prop_bq_threshold_correctness(
        input in non_empty_vec_f32(50, -100.0, 100.0),
        threshold in -50.0f32..50.0,
    ) {
        let bq = BinaryQuantizer::new(threshold, 0, 1).unwrap();
        let output = bq.quantize(&input).unwrap();

        for (i, &val) in input.iter().enumerate() {
            let expected = if val >= threshold { 1 } else { 0 };
            prop_assert_eq!(output[i], expected, "Mismatch at index {} for value {} with threshold {}", i, val, threshold);
        }
    }

    /// Property: BQ dequantize output length equals input length
    #[test]
    fn prop_bq_dequantize_length(
        input in non_empty_vec_f32(50, -100.0, 100.0),
    ) {
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let quantized = bq.quantize(&input).unwrap();
        let dequantized = bq.dequantize(&quantized).unwrap();
        prop_assert_eq!(dequantized.len(), input.len());
    }
}

// =============================================================================
// Scalar Quantizer Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: SQ output length equals input length
    #[test]
    fn prop_sq_output_length_equals_input(
        input in non_empty_vec_f32(100, -10.0, 10.0),
    ) {
        let sq = ScalarQuantizer::new(-10.0, 10.0, 256).unwrap();
        let output = sq.quantize(&input).unwrap();
        prop_assert_eq!(output.len(), input.len());
    }

    /// Property: SQ output values are within valid range [0, levels-1]
    #[test]
    fn prop_sq_output_in_valid_range(
        input in non_empty_vec_f32(100, -1000.0, 1000.0),
        levels in 2usize..=256,
    ) {
        let sq = ScalarQuantizer::new(-100.0, 100.0, levels).unwrap();
        let output = sq.quantize(&input).unwrap();
        for val in output {
            prop_assert!((val as usize) < levels, "Value {} exceeds max level {}", val, levels - 1);
        }
    }

    /// Property: SQ roundtrip error is bounded by half the step size
    #[test]
    fn prop_sq_roundtrip_error_bounded(
        input in vec_f32(20, -10.0, 10.0),
    ) {
        let sq = ScalarQuantizer::new(-10.0, 10.0, 256).unwrap();
        let quantized = sq.quantize(&input).unwrap();
        let reconstructed = sq.dequantize(&quantized).unwrap();

        let max_error = sq.step() / 2.0 + 1e-5;
        for (orig, recon) in input.iter().zip(reconstructed.iter()) {
            let clamped = orig.clamp(sq.min(), sq.max());
            let error = (clamped - recon).abs();
            prop_assert!(error <= max_error, "Error {} exceeds max {}", error, max_error);
        }
    }

    /// Property: SQ is deterministic
    #[test]
    fn prop_sq_deterministic(
        input in non_empty_vec_f32(50, -100.0, 100.0),
    ) {
        let sq = ScalarQuantizer::new(-100.0, 100.0, 256).unwrap();
        let output1 = sq.quantize(&input).unwrap();
        let output2 = sq.quantize(&input).unwrap();
        prop_assert_eq!(output1, output2);
    }

    /// Property: SQ dequantize produces values within [min, max]
    #[test]
    fn prop_sq_dequantize_in_range(
        input in non_empty_vec_f32(50, -100.0, 100.0),
    ) {
        let sq = ScalarQuantizer::new(-50.0, 50.0, 128).unwrap();
        let quantized = sq.quantize(&input).unwrap();
        let dequantized = sq.dequantize(&quantized).unwrap();

        for val in dequantized {
            prop_assert!(val >= sq.min() && val <= sq.max(),
                "Dequantized value {} outside range [{}, {}]", val, sq.min(), sq.max());
        }
    }
}

// =============================================================================
// Product Quantizer Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))] // Fewer cases due to training cost

    /// Property: PQ output dimension matches input dimension
    #[test]
    fn prop_pq_output_dimension(
        training in training_data(50, 8, -10.0, 10.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::new(&training_refs, 2, 4, 5, Distance::Euclidean, 42).unwrap();

        let test_vec = &training[0];
        let quantized = pq.quantize(test_vec).unwrap();
        prop_assert_eq!(quantized.len(), 8);
    }

    /// Property: PQ is deterministic (same input produces same output)
    #[test]
    fn prop_pq_deterministic(
        training in training_data(50, 8, -10.0, 10.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::new(&training_refs, 2, 4, 5, Distance::Euclidean, 42).unwrap();

        let test_vec = &training[0];
        let output1 = pq.quantize(test_vec).unwrap();
        let output2 = pq.quantize(test_vec).unwrap();
        prop_assert_eq!(output1, output2);
    }

    /// Property: PQ dequantize output dimension matches input
    #[test]
    fn prop_pq_dequantize_dimension(
        training in training_data(50, 12, -10.0, 10.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::new(&training_refs, 3, 4, 5, Distance::Euclidean, 42).unwrap();

        let test_vec = &training[0];
        let quantized = pq.quantize(test_vec).unwrap();
        let dequantized = pq.dequantize(&quantized).unwrap();
        prop_assert_eq!(dequantized.len(), 12);
    }

    /// Property: PQ reconstruction produces finite values
    #[test]
    fn prop_pq_reconstruction_finite(
        training in training_data(50, 8, -100.0, 100.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::new(&training_refs, 2, 4, 5, Distance::Euclidean, 42).unwrap();

        for vec in training.iter().take(10) {
            let quantized = pq.quantize(vec).unwrap();
            let dequantized = pq.dequantize(&quantized).unwrap();
            for val in dequantized {
                prop_assert!(val.is_finite(), "Non-finite value in PQ reconstruction");
            }
        }
    }
}

// =============================================================================
// TSVQ Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))] // Fewer cases due to tree building cost

    /// Property: TSVQ output dimension matches input dimension
    #[test]
    fn prop_tsvq_output_dimension(
        training in training_data(50, 6, -10.0, 10.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let tsvq = TSVQ::new(&training_refs, 3, Distance::Euclidean).unwrap();

        let test_vec = &training[0];
        let quantized = tsvq.quantize(test_vec).unwrap();
        prop_assert_eq!(quantized.len(), 6);
    }

    /// Property: TSVQ is deterministic
    #[test]
    fn prop_tsvq_deterministic(
        training in training_data(50, 6, -10.0, 10.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let tsvq = TSVQ::new(&training_refs, 3, Distance::Euclidean).unwrap();

        let test_vec = &training[0];
        let output1 = tsvq.quantize(test_vec).unwrap();
        let output2 = tsvq.quantize(test_vec).unwrap();
        prop_assert_eq!(output1, output2);
    }

    /// Property: TSVQ reconstruction produces finite values
    #[test]
    fn prop_tsvq_reconstruction_finite(
        training in training_data(50, 8, -100.0, 100.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let tsvq = TSVQ::new(&training_refs, 4, Distance::Euclidean).unwrap();

        for vec in training.iter().take(10) {
            let quantized = tsvq.quantize(vec).unwrap();
            let dequantized = tsvq.dequantize(&quantized).unwrap();
            for val in dequantized {
                prop_assert!(val.is_finite(), "Non-finite value in TSVQ reconstruction");
            }
        }
    }

    /// Property: TSVQ dequantize output dimension matches input
    #[test]
    fn prop_tsvq_dequantize_dimension(
        training in training_data(50, 10, -10.0, 10.0),
    ) {
        let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        let tsvq = TSVQ::new(&training_refs, 3, Distance::Euclidean).unwrap();

        let test_vec = &training[0];
        let quantized = tsvq.quantize(test_vec).unwrap();
        let dequantized = tsvq.dequantize(&quantized).unwrap();
        prop_assert_eq!(dequantized.len(), 10);
    }
}

// =============================================================================
// Cross-Algorithm Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: All quantizers preserve dimension in roundtrip
    #[test]
    fn prop_all_quantizers_preserve_dimension(
        input in vec_f32(10, -50.0, 50.0),
    ) {
        // BQ
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let bq_out = bq.quantize(&input).unwrap();
        let bq_recon = bq.dequantize(&bq_out).unwrap();
        prop_assert_eq!(bq_recon.len(), input.len());

        // SQ
        let sq = ScalarQuantizer::new(-50.0, 50.0, 256).unwrap();
        let sq_out = sq.quantize(&input).unwrap();
        let sq_recon = sq.dequantize(&sq_out).unwrap();
        prop_assert_eq!(sq_recon.len(), input.len());
    }

    /// Property: Empty input produces empty output for BQ and SQ
    #[test]
    fn prop_empty_input_empty_output(_dummy in 0..1i32) {
        let empty: Vec<f32> = vec![];

        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let bq_out = bq.quantize(&empty).unwrap();
        prop_assert!(bq_out.is_empty());

        let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();
        let sq_out = sq.quantize(&empty).unwrap();
        prop_assert!(sq_out.is_empty());
    }

    /// Property: Quantization output is reproducible across multiple calls
    #[test]
    fn prop_quantization_reproducible(
        input in vec_f32(20, -100.0, 100.0),
    ) {
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let sq = ScalarQuantizer::new(-100.0, 100.0, 256).unwrap();

        // Run multiple times and verify consistency
        for _ in 0..3 {
            let bq1 = bq.quantize(&input).unwrap();
            let bq2 = bq.quantize(&input).unwrap();
            prop_assert_eq!(bq1, bq2);

            let sq1 = sq.quantize(&input).unwrap();
            let sq2 = sq.quantize(&input).unwrap();
            prop_assert_eq!(sq1, sq2);
        }
    }
}

// =============================================================================
// Distance Metric Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: Distance to self is zero (or near-zero for Euclidean)
    #[test]
    fn prop_distance_to_self_is_zero(
        vec in vec_f32(10, -100.0, 100.0),
    ) {
        let distances = [
            Distance::Euclidean,
            Distance::SquaredEuclidean,
            Distance::Manhattan,
        ];

        for dist in distances {
            let result = dist.compute(&vec, &vec).unwrap();
            prop_assert!(result.abs() < 1e-6, "Distance to self should be zero for {:?}, got {}", dist, result);
        }
    }

    /// Property: Distance is symmetric
    #[test]
    fn prop_distance_symmetric(
        a in vec_f32(10, -100.0, 100.0),
        b in vec_f32(10, -100.0, 100.0),
    ) {
        let distances = [
            Distance::Euclidean,
            Distance::SquaredEuclidean,
            Distance::Manhattan,
            Distance::CosineDistance,
        ];

        for dist in distances {
            let d_ab = dist.compute(&a, &b).unwrap();
            let d_ba = dist.compute(&b, &a).unwrap();
            prop_assert!((d_ab - d_ba).abs() < 1e-5, "Distance not symmetric for {:?}: {} vs {}", dist, d_ab, d_ba);
        }
    }

    /// Property: Distance is non-negative
    #[test]
    fn prop_distance_non_negative(
        a in vec_f32(10, -100.0, 100.0),
        b in vec_f32(10, -100.0, 100.0),
    ) {
        let distances = [
            Distance::Euclidean,
            Distance::SquaredEuclidean,
            Distance::Manhattan,
        ];

        for dist in distances {
            let result = dist.compute(&a, &b).unwrap();
            prop_assert!(result >= 0.0, "Distance should be non-negative for {:?}, got {}", dist, result);
        }
    }

    /// Property: CosineDistance is in range [0, 2]
    #[test]
    fn prop_cosine_distance_in_range(
        a in vec_f32(10, 0.1, 100.0), // Avoid zero vectors
        b in vec_f32(10, 0.1, 100.0),
    ) {
        let result = Distance::CosineDistance.compute(&a, &b).unwrap();
        prop_assert!((-1e-6..=2.0 + 1e-6).contains(&result),
            "CosineDistance should be in [0, 2], got {}", result);
    }
}
