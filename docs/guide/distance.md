# Distance Metrics

Vq supports multiple distance metrics for vector comparisons. These are used both during training (PQ, TSVQ) and for finding nearest neighbors.

## Overview

The `Distance` enum provides four distance metrics:

| Metric | Formula | Range | Use Case |
|--------|---------|-------|----------|
| SquaredEuclidean | $\sum(a_i - b_i)^2$ | $[0, \infty)$ | Fast comparisons |
| Euclidean | $\sqrt{\sum(a_i - b_i)^2}$ | $[0, \infty)$ | General purpose |
| Manhattan | $\sum\|a_i - b_i\|$ | $[0, \infty)$ | Robust to outliers |
| CosineDistance | $1 - \frac{a \cdot b}{\|a\| \|b\|}$ | $[0, 2]$ | Normalized vectors |

## Using Distance

```rust
use vq::Distance;

let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

// Compute distances
let sq_euclidean = Distance::SquaredEuclidean.compute(&a, &b)?;
let euclidean = Distance::Euclidean.compute(&a, &b)?;
let manhattan = Distance::Manhattan.compute(&a, &b)?;
let cosine = Distance::CosineDistance.compute(&a, &b)?;

println!("Squared Euclidean: {}", sq_euclidean);  // 27.0
println!("Euclidean: {}", euclidean);              // ~5.196
println!("Manhattan: {}", manhattan);              // 9.0
println!("Cosine distance: {}", cosine);           // ~0.025
```

## Metric Details

### Squared Euclidean

$$d(a, b) = \sum_{i=1}^{n} (a_i - b_i)^2$$

- Fastest to compute (no square root)
- Same ordering as Euclidean distance
- Use for nearest neighbor comparisons when you don't need absolute distances

### Euclidean

$$d(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

- Standard "straight line" distance
- Use when you need actual distance values

### Manhattan

$$d(a, b) = \sum_{i=1}^{n} |a_i - b_i|$$

- Also called L1 distance or taxicab distance
- More robust to outliers than Euclidean
- Better for sparse vectors

### Cosine Distance

$$d(a, b) = 1 - \frac{a \cdot b}{\|a\| \|b\|}$$

- Measures angular difference, not magnitude
- Value of 0 means identical direction
- Value of 1 means orthogonal
- Value of 2 means opposite direction
- Use for normalized embeddings (e.g., sentence embeddings)

## SIMD Acceleration

When the `simd` feature is enabled, distance computations use SIMD instructions for faster performance:

- x86/x86_64: AVX, AVX2, AVX512, FMA
- ARM: NEON, SVE

The appropriate SIMD backend is automatically selected at runtime based on CPU capabilities.

```rust
#[cfg(feature = "simd")]
{
    let backend = vq::get_simd_backend();
    println!("SIMD backend: {}", backend);
    // e.g., "AVX2 (Auto)" or "NEON (Auto)"
}
```

## Error Handling

`Distance::compute()` returns an error if vectors have different lengths:

```rust
let a = vec![1.0, 2.0];
let b = vec![1.0, 2.0, 3.0];

let result = Distance::Euclidean.compute(&a, &b);
assert!(result.is_err());  // DimensionMismatch error
```

## Using with Quantizers

Distance metrics are used when creating PQ and TSVQ quantizers:

```rust
use vq::{ProductQuantizer, Distance};

let pq = ProductQuantizer::new(
    &training_refs,
    8,
    16,
    10,
    Distance::Euclidean,  // Use Euclidean distance for codebook training
    42,
)?;
```

## Choosing a Distance Metric

| Use Case | Recommended Metric |
|----------|-------------------|
| General nearest neighbors | SquaredEuclidean (fastest) |
| Need actual distances | Euclidean |
| Sparse vectors | Manhattan |
| Text/sentence embeddings | CosineDistance |
| Image embeddings | Euclidean or CosineDistance |
