//! Integration tests for the vq library.
//!
//! This file contains integration tests that exercise multiple components together,
//! regression tests, and property-based tests.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vq::bq::BinaryQuantizer;
use vq::distance::Distance;
use vq::pq::ProductQuantizer;
use vq::sq::ScalarQuantizer;
use vq::tsvq::TSVQ;
use vq::vector::Vector;

/// Shared test utilities
mod common {
    use super::*;

    pub const SEED: u64 = 42;
    pub const MIN_VAL: f32 = -1000.0;
    pub const MAX_VAL: f32 = 1000.0;

    pub fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(SEED)
    }

    pub fn generate_test_data(rng: &mut StdRng, n: usize, dim: usize) -> Vec<Vector<f32>> {
        (0..n)
            .map(|_| {
                let data: Vec<f32> = (0..dim)
                    .map(|_| rng.random_range(MIN_VAL..MAX_VAL))
                    .collect();
                Vector::new(data)
            })
            .collect()
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test that all quantizers can process the same dataset
#[test]
fn test_all_quantizers_on_same_data() {
    let mut rng = common::seeded_rng();
    let training_data = common::generate_test_data(&mut rng, 100, 10);
    let test_vector = &training_data[0];

    // BQ
    let bq = BinaryQuantizer::new(0.0, 0, 1);
    let bq_result = bq.quantize(&test_vector.data);
    assert_eq!(bq_result.len(), 10);

    // SQ
    let sq = ScalarQuantizer::new(-1000.0, 1000.0, 256);
    let sq_result = sq.quantize(&test_vector.data);
    assert_eq!(sq_result.len(), 10);

    // PQ (dim must be divisible by m)
    let pq = ProductQuantizer::fit(&training_data, 2, 4, 10, Distance::Euclidean, 42);
    let pq_result = pq.quantize(test_vector);
    assert_eq!(pq_result.len(), 10);

    // TSVQ
    let tsvq = TSVQ::new(&training_data, 3, Distance::Euclidean);
    let tsvq_result = tsvq.quantize(test_vector);
    assert_eq!(tsvq_result.len(), 10);
}

/// Test quantization consistency (same input should produce same output)
#[test]
fn test_quantization_consistency() {
    let mut rng = common::seeded_rng();
    let training_data = common::generate_test_data(&mut rng, 100, 10);
    let test_vector = &training_data[0];

    let pq = ProductQuantizer::fit(&training_data, 2, 4, 10, Distance::Euclidean, 42);

    let result1 = pq.quantize(test_vector);
    let result2 = pq.quantize(test_vector);

    assert_eq!(
        result1.data, result2.data,
        "Same input should produce same output"
    );
}

// =============================================================================
// Regression Tests
// =============================================================================

/// Regression test: SQ with edge values should not panic
#[test]
fn test_sq_edge_values() {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256);

    // Test with values exactly at boundaries
    let edge_values = vec![-1.0, 1.0, 0.0];
    let result = sq.quantize(&edge_values);
    assert_eq!(result.len(), 3);

    // Test with values outside boundaries (should be clamped)
    let outside_values = vec![-100.0, 100.0];
    let result = sq.quantize(&outside_values);
    assert_eq!(result.len(), 2);
}

/// Regression test: BQ should handle zero threshold correctly
#[test]
fn test_bq_zero_threshold() {
    let bq = BinaryQuantizer::new(0.0, 0, 1);

    // Zero should be >= threshold, so mapped to high
    let values = vec![0.0, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let result = bq.quantize(&values);

    assert_eq!(result.data[0], 1); // 0.0 >= 0.0
    assert_eq!(result.data[1], 1); // -0.0 >= 0.0
    assert_eq!(result.data[2], 1); // tiny positive >= 0.0
    assert_eq!(result.data[3], 0); // tiny negative < 0.0
}
