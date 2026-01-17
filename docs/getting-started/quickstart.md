# Quick Start

This guide shows you how to use Vq's quantization algorithms.

## Binary Quantization

Binary quantization maps values to 0 or 1 based on a threshold:

```rust
use vq::{BinaryQuantizer, Quantizer};

fn main() -> vq::VqResult<()> {
    // Values >= 0.0 map to 1, values < 0.0 map to 0
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;

    let vector = vec![-0.5, 0.0, 0.5, 1.0];
    let quantized = bq.quantize(&vector)?;

    println!("Quantized: {:?}", quantized);
    // Output: [0, 1, 1, 1]

    Ok(())
}
```

## Scalar Quantization

Scalar quantization maps a continuous range to discrete levels:

```rust
use vq::{ScalarQuantizer, Quantizer};

fn main() -> vq::VqResult<()> {
    // Map values from [-1.0, 1.0] to 256 levels
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;

    let vector = vec![-1.0, 0.0, 0.5, 1.0];
    let quantized = sq.quantize(&vector)?;

    // Reconstruct the vector
    let reconstructed = sq.dequantize(&quantized)?;

    println!("Original:      {:?}", vector);
    println!("Reconstructed: {:?}", reconstructed);

    Ok(())
}
```

## Product Quantization

Product quantization requires training on a dataset:

```rust
use vq::{ProductQuantizer, Distance, Quantizer};

fn main() -> vq::VqResult<()> {
    // Generate training data: 100 vectors of dimension 8
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..8).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Train PQ with 2 subspaces, 4 centroids each
    let pq = ProductQuantizer::new(
        &refs,
        2,    // m: number of subspaces
        4,    // k: centroids per subspace
        10,   // max iterations
        Distance::Euclidean,
        42,   // random seed
    )?;

    // Quantize and reconstruct
    let quantized = pq.quantize(&training[0])?;
    let reconstructed = pq.dequantize(&quantized)?;

    println!("Dimension: {}", pq.dim());
    println!("Subspaces: {}", pq.num_subspaces());

    Ok(())
}
```

## Distance Computation

Compute distances between vectors:

```rust
use vq::Distance;

fn main() -> vq::VqResult<()> {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    let euclidean = Distance::Euclidean.compute(&a, &b)?;
    let manhattan = Distance::Manhattan.compute(&a, &b)?;
    let cosine = Distance::CosineDistance.compute(&a, &b)?;

    println!("Euclidean: {}", euclidean);
    println!("Manhattan: {}", manhattan);
    println!("Cosine distance: {}", cosine);

    Ok(())
}
```

## Next Steps

- Learn about [vector quantization concepts](concepts.md)
- Explore individual algorithms in the [User Guide](../guide/bq.md)
- See detailed [API documentation](https://docs.rs/vq)
