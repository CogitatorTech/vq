mod utils;

use utils::{generate_test_data, seeded_rng};
use vq::{BinaryQuantizer, Distance, ProductQuantizer, Quantizer, ScalarQuantizer, VqError, TSVQ};

// =============================================================================
// Basic Quantization Tests
// =============================================================================

#[test]
fn test_all_quantizers_on_same_data() {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..10).map(|j| ((i + j) % 100) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
    let test_vector = &training[0];

    // BQ
    let bq = BinaryQuantizer::new(50.0, 0, 1).unwrap();
    let bq_result = bq.quantize(test_vector).unwrap();
    assert_eq!(bq_result.len(), 10);

    // SQ
    let sq = ScalarQuantizer::new(0.0, 100.0, 256).unwrap();
    let sq_result = sq.quantize(test_vector).unwrap();
    assert_eq!(sq_result.len(), 10);

    // PQ
    let pq = ProductQuantizer::new(&training_refs, 2, 4, 10, Distance::Euclidean, 42).unwrap();
    let pq_result = pq.quantize(test_vector).unwrap();
    assert_eq!(pq_result.len(), 10);

    // TSVQ
    let tsvq = TSVQ::new(&training_refs, 3, Distance::Euclidean).unwrap();
    let tsvq_result = tsvq.quantize(test_vector).unwrap();
    assert_eq!(tsvq_result.len(), 10);
}

#[test]
fn test_quantization_consistency() {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..10).map(|j| ((i + j) % 100) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
    let test_vector = &training[0];

    let pq = ProductQuantizer::new(&training_refs, 2, 4, 10, Distance::Euclidean, 42).unwrap();

    let result1 = pq.quantize(test_vector).unwrap();
    let result2 = pq.quantize(test_vector).unwrap();

    assert_eq!(result1, result2, "Same input should produce same output");
}

// =============================================================================
// Roundtrip (Quantize + Dequantize) Tests
// =============================================================================

#[test]
fn test_bq_roundtrip() {
    let bq = BinaryQuantizer::new(0.5, 0, 1).unwrap();
    let input = vec![0.2, 0.8, 0.4, 0.9, 0.1];

    let quantized = bq.quantize(&input).unwrap();
    let reconstructed = bq.dequantize(&quantized).unwrap();

    assert_eq!(reconstructed.len(), input.len());
    // BQ dequantize returns 0.0 or 1.0
    for val in &reconstructed {
        assert!(*val == 0.0 || *val == 1.0);
    }
}

#[test]
fn test_sq_roundtrip_bounded_error() {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();
    let input = vec![-0.9, -0.5, 0.0, 0.5, 0.9];

    let quantized = sq.quantize(&input).unwrap();
    let reconstructed = sq.dequantize(&quantized).unwrap();

    assert_eq!(reconstructed.len(), input.len());
    // Error should be bounded by half the step size
    let max_error = sq.step() / 2.0 + 1e-6;
    for (orig, recon) in input.iter().zip(reconstructed.iter()) {
        let error = (orig - recon).abs();
        assert!(
            error <= max_error,
            "SQ roundtrip error {} exceeds max {}",
            error,
            max_error
        );
    }
}

#[test]
fn test_pq_roundtrip() {
    let mut rng = seeded_rng();
    let training = generate_test_data(&mut rng, 200, 16);
    let training_slices: Vec<Vec<f32>> = training.iter().map(|v| v.data.clone()).collect();
    let training_refs: Vec<&[f32]> = training_slices.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&training_refs, 4, 8, 20, Distance::Euclidean, 42).unwrap();

    let test_vec = &training_slices[0];
    let quantized = pq.quantize(test_vec).unwrap();
    let reconstructed = pq.dequantize(&quantized).unwrap();

    assert_eq!(reconstructed.len(), test_vec.len());
    // PQ reconstruction should be close to original (within training data variance)
}

#[test]
fn test_tsvq_roundtrip() {
    let mut rng = seeded_rng();
    let training = generate_test_data(&mut rng, 100, 8);
    let training_slices: Vec<Vec<f32>> = training.iter().map(|v| v.data.clone()).collect();
    let training_refs: Vec<&[f32]> = training_slices.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&training_refs, 4, Distance::Euclidean).unwrap();

    let test_vec = &training_slices[0];
    let quantized = tsvq.quantize(test_vec).unwrap();
    let reconstructed = tsvq.dequantize(&quantized).unwrap();

    assert_eq!(reconstructed.len(), test_vec.len());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_pq_dimension_mismatch() {
    let training: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..12).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&training_refs, 3, 4, 10, Distance::Euclidean, 42).unwrap();

    // Wrong dimension vector
    let wrong_dim = vec![1.0, 2.0, 3.0]; // 3 instead of 12
    let result = pq.quantize(&wrong_dim);
    assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
}

#[test]
fn test_tsvq_dimension_mismatch() {
    let training: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..8).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&training_refs, 3, Distance::Euclidean).unwrap();

    let wrong_dim = vec![1.0, 2.0]; // 2 instead of 8
    let result = tsvq.quantize(&wrong_dim);
    assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
}

#[test]
fn test_pq_empty_training_data() {
    let empty: Vec<&[f32]> = vec![];
    let result = ProductQuantizer::new(&empty, 2, 4, 10, Distance::Euclidean, 42);
    assert!(matches!(result, Err(VqError::EmptyInput)));
}

#[test]
fn test_tsvq_empty_training_data() {
    let empty: Vec<&[f32]> = vec![];
    let result = TSVQ::new(&empty, 3, Distance::Euclidean);
    assert!(matches!(result, Err(VqError::EmptyInput)));
}

#[test]
fn test_bq_invalid_levels() {
    // low >= high should fail
    assert!(BinaryQuantizer::new(0.0, 5, 5).is_err());
    assert!(BinaryQuantizer::new(0.0, 10, 5).is_err());
}

#[test]
fn test_sq_invalid_parameters() {
    // max <= min
    assert!(ScalarQuantizer::new(10.0, 5.0, 256).is_err());
    // levels < 2
    assert!(ScalarQuantizer::new(0.0, 1.0, 1).is_err());
    // levels > 256
    assert!(ScalarQuantizer::new(0.0, 1.0, 300).is_err());
}

#[test]
fn test_pq_dimension_not_divisible() {
    let training: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..10).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // dim=10 is not divisible by m=3
    let result = ProductQuantizer::new(&training_refs, 3, 4, 10, Distance::Euclidean, 42);
    assert!(matches!(result, Err(VqError::InvalidParameter(_))));
}

// =============================================================================
// Distance Metric Tests
// =============================================================================

#[test]
fn test_pq_with_cosine_distance() {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..8).map(|j| ((i + j) % 50 + 1) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&training_refs, 2, 4, 10, Distance::CosineDistance, 42).unwrap();
    let result = pq.quantize(&training[0]).unwrap();
    assert_eq!(result.len(), 8);
}

#[test]
fn test_tsvq_with_squared_euclidean() {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..6).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&training_refs, 3, Distance::SquaredEuclidean).unwrap();
    let result = tsvq.quantize(&training[0]).unwrap();
    assert_eq!(result.len(), 6);
}

#[test]
fn test_tsvq_with_manhattan_distance() {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..6).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&training_refs, 3, Distance::Manhattan).unwrap();
    let result = tsvq.quantize(&training[0]).unwrap();
    assert_eq!(result.len(), 6);
}

#[test]
fn test_all_distance_metrics_with_pq() {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..8).map(|j| ((i + j) % 50 + 1) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let distances = [
        Distance::Euclidean,
        Distance::SquaredEuclidean,
        Distance::CosineDistance,
        Distance::Manhattan,
    ];

    for distance in distances {
        let pq = ProductQuantizer::new(&training_refs, 2, 4, 10, distance, 42).unwrap();
        let result = pq.quantize(&training[0]).unwrap();
        assert_eq!(result.len(), 8, "Failed for {:?}", distance);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_sq_edge_values() {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();

    let edge_values = vec![-1.0, 1.0, 0.0];
    let result = sq.quantize(&edge_values).unwrap();
    assert_eq!(result.len(), 3);

    let outside_values = vec![-100.0, 100.0];
    let result = sq.quantize(&outside_values).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_bq_zero_threshold() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();

    let values = vec![0.0, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let result = bq.quantize(&values).unwrap();

    assert_eq!(result[0], 1);
    assert_eq!(result[1], 1);
    assert_eq!(result[2], 1);
    assert_eq!(result[3], 0);
}

#[test]
fn test_bq_empty_vector() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
    let empty: Vec<f32> = vec![];
    let result = bq.quantize(&empty).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_sq_empty_vector() {
    let sq = ScalarQuantizer::new(0.0, 1.0, 256).unwrap();
    let empty: Vec<f32> = vec![];
    let result = sq.quantize(&empty).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_bq_special_float_values() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();

    // Test with special float values
    let special = vec![f32::INFINITY, f32::NEG_INFINITY];
    let result = bq.quantize(&special).unwrap();
    assert_eq!(result[0], 1); // INFINITY >= 0
    assert_eq!(result[1], 0); // NEG_INFINITY < 0
}

#[test]
fn test_pq_single_training_vector() {
    let training = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Should work with a single training vector
    let pq = ProductQuantizer::new(&training_refs, 2, 1, 10, Distance::Euclidean, 42).unwrap();
    let result = pq.quantize(&training[0]).unwrap();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_tsvq_identical_training_vectors() {
    let vec = vec![1.0, 2.0, 3.0, 4.0];
    let training: Vec<Vec<f32>> = (0..20).map(|_| vec.clone()).collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&training_refs, 3, Distance::Euclidean).unwrap();
    let result = tsvq.quantize(&vec).unwrap();
    assert_eq!(result.len(), 4);
}

// =============================================================================
// Large Scale / Stress Tests
// =============================================================================

#[test]
fn test_high_dimensional_vectors() {
    let dim = 256;
    let mut rng = seeded_rng();
    let training = generate_test_data(&mut rng, 100, dim);
    let training_slices: Vec<Vec<f32>> = training.iter().map(|v| v.data.clone()).collect();
    let training_refs: Vec<&[f32]> = training_slices.iter().map(|v| v.as_slice()).collect();

    // PQ with many subspaces
    let pq = ProductQuantizer::new(&training_refs, 16, 8, 10, Distance::Euclidean, 42).unwrap();
    let result = pq.quantize(&training_slices[0]).unwrap();
    assert_eq!(result.len(), dim);
    assert_eq!(pq.dim(), dim);
    assert_eq!(pq.num_subspaces(), 16);
    assert_eq!(pq.sub_dim(), 16);
}

#[test]
fn test_large_training_set() {
    let dim = 16;
    let n = 1000;
    let mut rng = seeded_rng();
    let training = generate_test_data(&mut rng, n, dim);
    let training_slices: Vec<Vec<f32>> = training.iter().map(|v| v.data.clone()).collect();
    let training_refs: Vec<&[f32]> = training_slices.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&training_refs, 4, 16, 20, Distance::Euclidean, 42).unwrap();

    // Quantize multiple vectors
    for slice in training_slices.iter().take(100) {
        let result = pq.quantize(slice).unwrap();
        assert_eq!(result.len(), dim);
    }
}

#[test]
fn test_sq_large_vector() {
    let sq = ScalarQuantizer::new(-1000.0, 1000.0, 256).unwrap();
    let large_input: Vec<f32> = (0..10000).map(|i| ((i % 2000) as f32) - 1000.0).collect();

    let quantized = sq.quantize(&large_input).unwrap();
    assert_eq!(quantized.len(), 10000);

    let reconstructed = sq.dequantize(&quantized).unwrap();
    assert_eq!(reconstructed.len(), 10000);
}

#[test]
fn test_bq_large_vector() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
    let large_input: Vec<f32> = (0..10000)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let quantized = bq.quantize(&large_input).unwrap();
    assert_eq!(quantized.len(), 10000);

    for (i, &val) in quantized.iter().enumerate() {
        let expected = if i % 2 == 0 { 1 } else { 0 };
        assert_eq!(val, expected);
    }
}

// =============================================================================
// SIMD Consistency Tests (when simd feature is enabled)
// =============================================================================

#[cfg(feature = "simd")]
mod simd_tests {
    use super::*;

    #[test]
    fn test_simd_backend_available() {
        let backend = vq::get_simd_backend();
        // Should return a valid backend string
        assert!(!backend.is_empty());
    }

    #[test]
    fn test_pq_simd_produces_valid_results() {
        let mut rng = seeded_rng();
        let training = generate_test_data(&mut rng, 200, 32);
        let training_slices: Vec<Vec<f32>> = training.iter().map(|v| v.data.clone()).collect();
        let training_refs: Vec<&[f32]> = training_slices.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::new(&training_refs, 4, 8, 15, Distance::Euclidean, 42).unwrap();

        // Ensure SIMD-accelerated distance computations produce valid quantization
        for vec in training_slices.iter().take(50) {
            let quantized = pq.quantize(vec).unwrap();
            assert_eq!(quantized.len(), 32);

            let reconstructed = pq.dequantize(&quantized).unwrap();
            assert_eq!(reconstructed.len(), 32);

            // Values should be finite
            for val in &reconstructed {
                assert!(val.is_finite(), "Got non-finite value in reconstruction");
            }
        }
    }

    #[test]
    fn test_tsvq_simd_produces_valid_results() {
        let mut rng = seeded_rng();
        let training = generate_test_data(&mut rng, 150, 16);
        let training_slices: Vec<Vec<f32>> = training.iter().map(|v| v.data.clone()).collect();
        let training_refs: Vec<&[f32]> = training_slices.iter().map(|v| v.as_slice()).collect();

        let tsvq = TSVQ::new(&training_refs, 5, Distance::Euclidean).unwrap();

        for vec in training_slices.iter().take(50) {
            let quantized = tsvq.quantize(vec).unwrap();
            assert_eq!(quantized.len(), 16);

            let reconstructed = tsvq.dequantize(&quantized).unwrap();
            for val in &reconstructed {
                assert!(val.is_finite());
            }
        }
    }
}

// =============================================================================
// Special Float Value Edge Case Tests
// =============================================================================

#[test]
fn test_bq_with_nan_input() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();

    // NaN comparisons always return false, so NaN >= threshold is false
    let input = vec![f32::NAN, 1.0, -1.0, f32::NAN];
    let result = bq.quantize(&input).unwrap();

    // NaN >= 0.0 is false, so it maps to low (0)
    assert_eq!(result[0], 0); // NaN
    assert_eq!(result[1], 1); // 1.0 >= 0.0
    assert_eq!(result[2], 0); // -1.0 < 0.0
    assert_eq!(result[3], 0); // NaN
}

#[test]
fn test_bq_with_infinity_input() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();

    let input = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0];
    let result = bq.quantize(&input).unwrap();

    assert_eq!(result[0], 1); // +Inf >= 0.0
    assert_eq!(result[1], 0); // -Inf < 0.0
    assert_eq!(result[2], 1); // 0.0 >= 0.0
}

#[test]
fn test_sq_with_nan_input() {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();

    // NaN.clamp() returns NaN, and NaN comparisons produce undefined behavior
    // The current implementation will produce some output (likely 0 due to rounding)
    let input = vec![f32::NAN];
    let result = sq.quantize(&input).unwrap();
    assert_eq!(result.len(), 1);
    // Note: The exact value is implementation-defined for NaN
}

#[test]
fn test_sq_with_infinity_input() {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();

    let input = vec![f32::INFINITY, f32::NEG_INFINITY];
    let result = sq.quantize(&input).unwrap();

    // +Inf clamped to max (1.0) -> highest level (255)
    assert_eq!(result[0], 255);
    // -Inf clamped to min (-1.0) -> lowest level (0)
    assert_eq!(result[1], 0);
}

#[test]
fn test_sq_with_subnormal_floats() {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();

    // Subnormal (denormalized) floats are very small numbers close to zero
    let subnormal = f32::MIN_POSITIVE / 2.0; // This is subnormal
    let input = vec![subnormal, -subnormal, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let result = sq.quantize(&input).unwrap();

    assert_eq!(result.len(), 4);
    // All these values are very close to 0, so they should map to the middle level
    // Middle of [-1, 1] with 256 levels is around level 127-128
    for &val in &result {
        assert!(
            val >= 126 && val <= 129,
            "Subnormal should map near middle, got {}",
            val
        );
    }
}

#[test]
fn test_sq_with_extreme_values() {
    let sq = ScalarQuantizer::new(-1e10, 1e10, 256).unwrap();

    let input = vec![f32::MAX, f32::MIN_POSITIVE, -f32::MAX, 0.0];
    let result = sq.quantize(&input).unwrap();

    assert_eq!(result.len(), 4);
    // f32::MAX is clamped to 1e10 -> level 255
    assert_eq!(result[0], 255);
    // f32::MIN_POSITIVE is close to 0 -> middle level
    assert!(result[1] >= 126 && result[1] <= 129);
    // -f32::MAX is clamped to -1e10 -> level 0
    assert_eq!(result[2], 0);
    // 0.0 -> middle level
    assert!(result[3] >= 126 && result[3] <= 129);
}

#[test]
fn test_bq_dequantize_with_arbitrary_values() {
    let bq = BinaryQuantizer::new(0.0, 10, 20).unwrap();

    // Dequantize with values that don't match low/high
    let arbitrary = vec![0, 5, 10, 15, 20, 25, 255];
    let result = bq.dequantize(&arbitrary).unwrap();

    // Values >= high (20) map to 1.0, others to 0.0
    assert_eq!(result[0], 0.0); // 0 < 20
    assert_eq!(result[1], 0.0); // 5 < 20
    assert_eq!(result[2], 0.0); // 10 < 20
    assert_eq!(result[3], 0.0); // 15 < 20
    assert_eq!(result[4], 1.0); // 20 >= 20
    assert_eq!(result[5], 1.0); // 25 >= 20
    assert_eq!(result[6], 1.0); // 255 >= 20
}

#[test]
fn test_sq_dequantize_out_of_range_indices() {
    let sq = ScalarQuantizer::new(0.0, 10.0, 11).unwrap(); // step = 1.0

    // Dequantize with index larger than levels-1
    let out_of_range = vec![0, 5, 10, 100, 255];
    let result = sq.dequantize(&out_of_range).unwrap();

    // Index 0 -> 0.0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // Index 5 -> 5.0
    assert!((result[1] - 5.0).abs() < 1e-6);
    // Index 10 -> 10.0
    assert!((result[2] - 10.0).abs() < 1e-6);
    // Index 100 -> 100.0 (extrapolates beyond max, no clamping in dequantize)
    assert!((result[3] - 100.0).abs() < 1e-6);
    // Index 255 -> 255.0
    assert!((result[4] - 255.0).abs() < 1e-6);
}

#[test]
fn test_distance_with_nan() {
    let a = vec![1.0, f32::NAN, 3.0];
    let b = vec![1.0, 2.0, 3.0];

    // NaN in distance computation should propagate
    let result = Distance::Euclidean.compute(&a, &b).unwrap();
    assert!(result.is_nan(), "Distance with NaN input should return NaN");

    let result = Distance::Manhattan.compute(&a, &b).unwrap();
    assert!(result.is_nan());

    let result = Distance::SquaredEuclidean.compute(&a, &b).unwrap();
    assert!(result.is_nan());
}

#[test]
fn test_distance_with_infinity() {
    let a = vec![f32::INFINITY, 0.0];
    let b = vec![0.0, 0.0];

    let result = Distance::Euclidean.compute(&a, &b).unwrap();
    assert!(result.is_infinite() && result > 0.0);

    let result = Distance::Manhattan.compute(&a, &b).unwrap();
    assert!(result.is_infinite() && result > 0.0);
}

#[test]
fn test_distance_with_opposite_infinities() {
    let a = vec![f32::INFINITY];
    let b = vec![f32::NEG_INFINITY];

    let result = Distance::Euclidean.compute(&a, &b).unwrap();
    assert!(result.is_infinite());

    let result = Distance::Manhattan.compute(&a, &b).unwrap();
    assert!(result.is_infinite());
}

#[test]
fn test_cosine_distance_with_zero_vector() {
    let zero = vec![0.0, 0.0, 0.0];
    let nonzero = vec![1.0, 2.0, 3.0];

    // Cosine with zero vector should return 1.0 (maximum distance)
    let result = Distance::CosineDistance.compute(&zero, &nonzero).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-6,
        "Cosine with zero vector should be 1.0, got {}",
        result
    );

    let result = Distance::CosineDistance.compute(&zero, &zero).unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_distance_with_near_zero_vector() {
    // Very small values that are not exactly zero
    let small = vec![1e-38, 1e-38, 1e-38];
    let normal = vec![1.0, 1.0, 1.0];

    let result = Distance::CosineDistance.compute(&small, &normal).unwrap();
    // Should be close to 0 since vectors point in same direction
    assert!(result.is_finite());
    assert!(result >= 0.0 && result <= 2.0);
}

#[test]
fn test_sq_boundary_precision() {
    // Test exact boundary values don't cause off-by-one errors
    let sq = ScalarQuantizer::new(0.0, 1.0, 11).unwrap(); // 0.0, 0.1, 0.2, ..., 1.0

    let boundaries = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let result = sq.quantize(&boundaries).unwrap();

    for (i, &level) in result.iter().enumerate() {
        assert_eq!(
            level as usize, i,
            "Boundary {} should map to level {}",
            boundaries[i], i
        );
    }
}

#[test]
fn test_bq_negative_zero() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();

    // Both +0.0 and -0.0 should be >= 0.0
    let input = vec![0.0, -0.0];
    let result = bq.quantize(&input).unwrap();

    assert_eq!(result[0], 1); // 0.0 >= 0.0
    assert_eq!(result[1], 1); // -0.0 >= 0.0 (IEEE 754: -0.0 == 0.0)
}

#[test]
fn test_mixed_special_values() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();

    let input = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.0,
        -0.0,
        f32::MIN_POSITIVE / 2.0, // subnormal
    ];
    let result = bq.quantize(&input).unwrap();
    assert_eq!(result.len(), input.len());

    // All values produce valid binary output
    for &val in &result {
        assert!(val == 0 || val == 1);
    }
}
