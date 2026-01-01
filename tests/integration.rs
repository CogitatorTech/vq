use vq::bq::BinaryQuantizer;
use vq::distance::Distance;
use vq::pq::ProductQuantizer;
use vq::sq::ScalarQuantizer;
use vq::tsvq::TSVQ;

#[test]
fn test_all_quantizers_on_same_data() {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..10).map(|j| ((i + j) % 100) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
    let test_vector = &training[0];

    // BQ
    let bq = BinaryQuantizer::new(50.0, 0, 1).unwrap();
    let bq_result = bq.quantize(test_vector);
    assert_eq!(bq_result.len(), 10);

    // SQ
    let sq = ScalarQuantizer::new(0.0, 100.0, 256).unwrap();
    let sq_result = sq.quantize(test_vector);
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

#[test]
fn test_sq_edge_values() {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();

    let edge_values = vec![-1.0, 1.0, 0.0];
    let result = sq.quantize(&edge_values);
    assert_eq!(result.len(), 3);

    let outside_values = vec![-100.0, 100.0];
    let result = sq.quantize(&outside_values);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_bq_zero_threshold() {
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();

    let values = vec![0.0, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let result = bq.quantize(&values);

    assert_eq!(result[0], 1);
    assert_eq!(result[1], 1);
    assert_eq!(result[2], 1);
    assert_eq!(result[3], 0);
}
