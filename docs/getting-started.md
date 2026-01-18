# Getting Started

This guide covers installation and basic usage of Vq.

## Installation

Add Vq to your project:

```bash
cargo add vq --features parallel,simd
```

!!! note "Requirements"
    - Rust 1.85 or later
    - For `simd` feature: a C compiler (GCC or Clang) supporting C11

## Binary Quantization

Binary quantization maps values to 0 or 1 based on a threshold. It provides 75% storage reduction.

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

Scalar quantization maps a continuous range to discrete levels. It also provides 75% storage reduction.

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

Product quantization requires training on a dataset. It splits vectors into subspaces and learns codebooks.

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

Compute distances between vectors using various metrics:

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

## Concepts

### What is Vector Quantization?

Vector quantization is a lossy compression technique that approximates vectors using a smaller set of representative values. It's commonly used for:

- **Embedding compression**: Reduce memory for ML embeddings (128-1536 dimensions)
- **Approximate nearest neighbor search**: Speed up similarity searches
- **Data compression**: Reduce storage costs for vector databases

### Choosing an Algorithm

| Algorithm | Best For | Compression |
|-----------|----------|-------------|
| **Binary** | Fast similarity via Hamming distance | 75% (f32 → u8) |
| **Scalar** | Values with known min/max range | 75% (f32 → u8) |
| **Product** | High-dimensional embeddings | 50% (f32 → f16) |
| **TSVQ** | Hierarchical clustering | 50% (f32 → f16) |
