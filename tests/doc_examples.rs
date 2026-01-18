//! Test all code examples from Rust documentation markdown files.

use vq::{BinaryQuantizer, ScalarQuantizer, ProductQuantizer, Distance, Quantizer, VqResult};

/// Test examples from docs/index.md
fn test_index_example() -> VqResult<()> {
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;
    let quantized = bq.quantize(&[-1.0, 0.5, 1.0])?;
    assert_eq!(quantized, vec![0, 1, 1]);
    Ok(())
}

/// Test examples from docs/getting-started.md - Binary Quantization
fn test_gs_binary() -> VqResult<()> {
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;
    let vector = vec![-0.5, 0.0, 0.5, 1.0];
    let quantized = bq.quantize(&vector)?;
    assert_eq!(quantized, vec![0, 1, 1, 1]);
    Ok(())
}

/// Test examples from docs/getting-started.md - Scalar Quantization
fn test_gs_scalar() -> VqResult<()> {
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;
    let vector = vec![-1.0, 0.0, 0.5, 1.0];
    let quantized = sq.quantize(&vector)?;
    let reconstructed = sq.dequantize(&quantized)?;
    assert_eq!(reconstructed.len(), vector.len());
    Ok(())
}

/// Test examples from docs/getting-started.md - Product Quantization
fn test_gs_pq() -> VqResult<()> {
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..8).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(
        &refs,
        2,    // m: number of subspaces
        4,    // k: centroids per subspace
        10,   // max iterations
        Distance::Euclidean,
        42,   // random seed
    )?;

    let quantized = pq.quantize(&training[0])?;
    let _reconstructed = pq.dequantize(&quantized)?;

    assert_eq!(pq.dim(), 8);
    assert_eq!(pq.num_subspaces(), 2);
    Ok(())
}

/// Test examples from docs/getting-started.md - Distance
fn test_gs_distance() -> VqResult<()> {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    let euclidean = Distance::Euclidean.compute(&a, &b)?;
    let manhattan = Distance::Manhattan.compute(&a, &b)?;
    let cosine = Distance::CosineDistance.compute(&a, &b)?;

    assert!(euclidean > 0.0);
    assert!(manhattan > 0.0);
    assert!(cosine >= 0.0 && cosine <= 2.0);
    Ok(())
}

/// Test examples from docs/examples.md - Hamming distance
fn test_ex_hamming() -> VqResult<()> {
    fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }

    let bq = BinaryQuantizer::new(0.0, 0, 1)?;
    let embeddings = vec![
        vec![0.5, -0.3, 0.1, -0.8, 0.2],
        vec![0.4, -0.2, 0.0, -0.7, 0.3],  // Similar to first
        vec![-0.6, 0.4, -0.2, 0.9, -0.1], // Different
    ];

    let codes: Vec<_> = embeddings.iter()
        .map(|e| bq.quantize(e))
        .collect::<VqResult<_>>()?;

    let h01 = hamming_distance(&codes[0], &codes[1]);
    let h02 = hamming_distance(&codes[0], &codes[2]);
    assert!(h01 < h02, "Similar vectors should have lower Hamming distance");
    Ok(())
}

/// Test examples from docs/examples.md - Scalar error analysis
fn test_ex_scalar_error() -> VqResult<()> {
    let levels_to_test = [4, 16, 64, 256];
    let test_vector: Vec<f32> = (0..100)
        .map(|i| (i as f32 / 50.0) - 1.0)
        .collect();

    for levels in levels_to_test {
        let sq = ScalarQuantizer::new(-1.0, 1.0, levels)?;
        let quantized = sq.quantize(&test_vector)?;
        let reconstructed = sq.dequantize(&quantized)?;

        let mse: f32 = test_vector.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / test_vector.len() as f32;

        assert!(mse >= 0.0);
    }
    Ok(())
}

/// Test examples from docs/examples.md - PQ compression
fn test_ex_pq_compression() -> VqResult<()> {
    let embeddings: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            (0..128)
                .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 500.0 - 1.0)
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&refs, 16, 256, 15, Distance::SquaredEuclidean, 42)?;

    assert_eq!(pq.dim(), 128);
    assert_eq!(pq.num_subspaces(), 16);
    assert_eq!(pq.sub_dim(), 8);
    Ok(())
}

/// Test examples from docs/examples.md - Distance comparison
fn test_ex_distance_comparison() -> VqResult<()> {
    let a: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) + 0.1).collect();

    let metrics = [
        Distance::SquaredEuclidean,
        Distance::Euclidean,
        Distance::Manhattan,
        Distance::CosineDistance,
    ];

    for metric in metrics {
        let dist = metric.compute(&a, &b)?;
        assert!(dist >= 0.0);
    }
    Ok(())
}

/// Test examples from docs/examples.md - Chaining quantizers
fn test_ex_chaining() -> VqResult<()> {
    let test_vector = vec![0.1, -0.5, 0.8, -0.2, 0.6];

    let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;
    let bq = BinaryQuantizer::new(0.5, 0, 1)?;

    let sq_quantized = sq.quantize(&test_vector)?;
    let sq_reconstructed = sq.dequantize(&sq_quantized)?;
    let bq_quantized = bq.quantize(&sq_reconstructed)?;

    assert_eq!(bq_quantized.len(), test_vector.len());
    Ok(())
}

#[test]
fn test_all_doc_examples() {
    test_index_example().expect("index.md example failed");
    test_gs_binary().expect("getting-started.md binary example failed");
    test_gs_scalar().expect("getting-started.md scalar example failed");
    test_gs_pq().expect("getting-started.md PQ example failed");
    test_gs_distance().expect("getting-started.md distance example failed");
    test_ex_hamming().expect("examples.md hamming example failed");
    test_ex_scalar_error().expect("examples.md scalar error example failed");
    test_ex_pq_compression().expect("examples.md PQ compression example failed");
    test_ex_distance_comparison().expect("examples.md distance comparison failed");
    test_ex_chaining().expect("examples.md chaining example failed");
}
