# Basic Usage Examples

This page contains complete examples showing common Vq usage patterns.

## Binary Quantization

```rust
use vq::{BinaryQuantizer, Quantizer, VqResult};

/// Count the number of differing bits between two binary vectors
fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

fn main() -> VqResult<()> {
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;

    // Sample embeddings
    let embeddings = vec![
        vec![0.5, -0.3, 0.1, -0.8, 0.2],
        vec![0.4, -0.2, 0.0, -0.7, 0.3],  // Similar to first
        vec![-0.6, 0.4, -0.2, 0.9, -0.1], // Different
    ];

    // Quantize all embeddings
    let codes: Vec<_> = embeddings.iter()
        .map(|e| bq.quantize(e))
        .collect::<VqResult<_>>()?;

    // Compare using Hamming distance
    println!("Hamming(0, 1) = {}", hamming_distance(&codes[0], &codes[1]));
    println!("Hamming(0, 2) = {}", hamming_distance(&codes[0], &codes[2]));

    Ok(())
}
```

## Scalar Quantization with Error Analysis

```rust
use vq::{ScalarQuantizer, Quantizer, VqResult};

fn main() -> VqResult<()> {
    // Test different quantization levels
    let levels_to_test = [4, 16, 64, 256];
    let test_vector: Vec<f32> = (0..100)
        .map(|i| (i as f32 / 50.0) - 1.0)  // Values in [-1, 1]
        .collect();

    for levels in levels_to_test {
        let sq = ScalarQuantizer::new(-1.0, 1.0, levels)?;

        let quantized = sq.quantize(&test_vector)?;
        let reconstructed = sq.dequantize(&quantized)?;

        let mse: f32 = test_vector.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / test_vector.len() as f32;

        let max_error: f32 = test_vector.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        println!(
            "Levels: {:3} | MSE: {:.6} | Max Error: {:.4}",
            levels, mse, max_error
        );
    }

    Ok(())
}
```

## Product Quantization for Embedding Compression

```rust
use vq::{ProductQuantizer, Distance, Quantizer, VqResult};

fn main() -> VqResult<()> {
    // Simulate 1000 embeddings of dimension 128
    let embeddings: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            (0..128)
                .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 500.0 - 1.0)
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();

    // Train PQ: 16 subspaces (128/16 = 8 dims each), 256 centroids
    println!("Training PQ...");
    let pq = ProductQuantizer::new(&refs, 16, 256, 15, Distance::SquaredEuclidean, 42)?;

    println!("PQ Configuration:");
    println!("  Dimension: {}", pq.dim());
    println!("  Subspaces: {}", pq.num_subspaces());
    println!("  Sub-dimension: {}", pq.sub_dim());

    // Quantize and measure error
    let mut total_mse = 0.0;
    for emb in &embeddings[..100] {
        let quantized = pq.quantize(emb)?;
        let reconstructed = pq.dequantize(&quantized)?;

        let mse: f32 = emb.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / emb.len() as f32;
        total_mse += mse;
    }

    println!("Average MSE: {:.6}", total_mse / 100.0);

    // Storage comparison
    let original_bytes = 128 * 4;  // 128 floats * 4 bytes
    let quantized_bytes = 128 * 2; // 128 f16 values * 2 bytes
    println!(
        "Compression: {} bytes -> {} bytes ({:.0}% reduction)",
        original_bytes,
        quantized_bytes,
        (1.0 - quantized_bytes as f64 / original_bytes as f64) * 100.0
    );

    Ok(())
}
```

## Distance Computation Comparison

```rust
use vq::{Distance, VqResult};

fn main() -> VqResult<()> {
    // Create test vectors
    let a: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) + 0.1).collect();

    // Compare all distance metrics
    let metrics = [
        ("Squared Euclidean", Distance::SquaredEuclidean),
        ("Euclidean", Distance::Euclidean),
        ("Manhattan", Distance::Manhattan),
        ("Cosine Distance", Distance::CosineDistance),
    ];

    for (name, metric) in metrics {
        let dist = metric.compute(&a, &b)?;
        println!("{:20} = {:.6}", name, dist);
    }

    // Check SIMD backend (if enabled)
    #[cfg(feature = "simd")]
    {
        println!("\nSIMD Backend: {}", vq::get_simd_backend());
    }

    Ok(())
}
```

## Combining Multiple Quantizers

```rust
use vq::{BinaryQuantizer, ScalarQuantizer, Quantizer, VqResult};

fn main() -> VqResult<()> {
    let test_vector = vec![0.1, -0.5, 0.8, -0.2, 0.6];

    // Chain quantizers: first SQ, then BQ on reconstructed
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;
    let bq = BinaryQuantizer::new(0.5, 0, 1)?;

    // Step 1: Scalar quantization
    let sq_quantized = sq.quantize(&test_vector)?;
    let sq_reconstructed = sq.dequantize(&sq_quantized)?;

    // Step 2: Binary quantization on SQ output
    let bq_quantized = bq.quantize(&sq_reconstructed)?;

    println!("Original: {:?}", test_vector);
    println!("After SQ: {:?}", sq_reconstructed);
    println!("After BQ: {:?}", bq_quantized);

    Ok(())
}
```
