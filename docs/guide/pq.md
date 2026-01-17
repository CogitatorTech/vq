# Product Quantizer (PQ)

Product quantization splits vectors into subspaces and quantizes each independently using learned codebooks, enabling efficient compression and approximate nearest neighbor search.

## Overview

Product quantization divides high-dimensional vectors into smaller subspaces and learns optimal codebooks for each subspace through training. This allows for efficient distance computation using precomputed lookup tables.

| Property | Value |
|----------|-------|
| Compression | 50% |
| Training | Required |
| Output type | `Vec<f16>` |
| Complexity | $O(nd)$ quantization, $O(nkd)$ training |

## Creating a Product Quantizer

PQ requires training data to learn codebooks:

```rust
use vq::{ProductQuantizer, Distance, Quantizer};

// Training data: 1000 vectors of dimension 128
let training: Vec<Vec<f32>> = generate_training_data();
let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

// Train PQ with 16 subspaces, 256 centroids each
let pq = ProductQuantizer::new(
    &refs,
    16,     // m: number of subspaces
    256,    // k: centroids per subspace
    20,     // max training iterations
    Distance::Euclidean,
    42,     // random seed for reproducibility
)?;
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_data` | `&[&[f32]]` | Training vectors for learning codebooks |
| `m` | `usize` | Number of subspaces to divide vectors into |
| `k` | `usize` | Number of centroids per subspace |
| `max_iters` | `usize` | Maximum iterations for codebook training |
| `distance` | `Distance` | Distance metric to use |
| `seed` | `u64` | Random seed for reproducibility |

### Validation

The constructor returns an error if:

- Training data is empty
- Vector dimension is less than `m`
- Vector dimension is not divisible by `m`

## How PQ Works

1. Divide each vector into `m` subvectors of dimension `d/m`
2. For each subspace, learn `k` centroids using LBG algorithm
3. During quantization, map each subvector to its nearest centroid
4. Store the centroid values (as f16) instead of centroid indices

## Quantization

```rust
let vector: Vec<f32> = get_vector();  // dimension 128
let quantized = pq.quantize(&vector)?;
// Result: Vec<f16> of length 128 (centroid values)
```

## Dequantization

```rust
let reconstructed = pq.dequantize(&quantized)?;
// Result: Vec<f32> of length 128
```

## Use Cases

Product quantization is ideal for:

- Large-scale approximate nearest neighbor search
- Billion-scale vector databases
- Memory-efficient embedding storage
- When training data is available

## Example: Training and Using PQ

```rust
use vq::{ProductQuantizer, Distance, Quantizer};

fn main() -> vq::VqResult<()> {
    // Generate synthetic training data
    let training: Vec<Vec<f32>> = (0..1000)
        .map(|i| (0..64).map(|j| ((i + j) % 100) as f32 / 100.0).collect())
        .collect();
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Train PQ: 8 subspaces (64/8 = 8 dims each), 16 centroids each
    let pq = ProductQuantizer::new(
        &refs,
        8,
        16,
        10,
        Distance::Euclidean,
        0,
    )?;

    println!("Dimension: {}", pq.dim());        // 64
    println!("Subspaces: {}", pq.num_subspaces()); // 8
    println!("Sub-dim: {}", pq.sub_dim());      // 8

    // Quantize a vector
    let query = &training[0];
    let quantized = pq.quantize(query)?;
    let reconstructed = pq.dequantize(&quantized)?;

    // Calculate reconstruction error
    let mse: f32 = query.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / query.len() as f32;
    println!("MSE: {}", mse);

    Ok(())
}
```

## Accessor Methods

```rust
let pq = ProductQuantizer::new(&refs, 8, 16, 10, Distance::Euclidean, 0)?;

assert_eq!(pq.dim(), 64);        // Expected input dimension
assert_eq!(pq.num_subspaces(), 8); // Number of subspaces (m)
assert_eq!(pq.sub_dim(), 8);     // Dimension per subspace (d/m)
```

## Choosing Parameters

### Number of Subspaces (m)

- Higher `m` = more subspaces = faster quantization but potentially lower accuracy
- Must divide the vector dimension evenly
- Common choices: 4, 8, 16, 32

### Number of Centroids (k)

- Higher `k` = more centroids = better accuracy but slower training
- Common choices: 16, 64, 256
- Memory per subspace: `k * sub_dim * sizeof(f32)`

### Training Iterations

- More iterations = better codebook quality
- Typically 10-20 iterations is sufficient
- Watch for convergence in reconstruction error
