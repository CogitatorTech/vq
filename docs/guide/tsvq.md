# Tree-Structured Vector Quantization (TSVQ)

Tree-structured vector quantization builds a hierarchical binary tree of cluster centroids, allowing efficient $O(\log k)$ quantization by traversing the tree to find the nearest leaf node.

## Overview

TSVQ organizes vectors in a hierarchical tree structure. Each node contains a centroid, and the tree is split based on the dimension with maximum variance. This enables fast quantization by traversing the tree rather than comparing against all centroids.

| Property | Value |
|----------|-------|
| Compression | 50% |
| Training | Required |
| Output type | `Vec<f16>` |
| Complexity | $O(d \log k)$ quantization, $O(n \log k)$ training |

## Creating a TSVQ

TSVQ requires training data to build the tree:

```rust
use vq::{TSVQ, Distance, Quantizer};

// Training data: 1000 vectors of dimension 64
let training: Vec<Vec<f32>> = generate_training_data();
let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

// Build TSVQ with max depth 6 (up to 64 leaf nodes)
let tsvq = TSVQ::new(
    &refs,
    6,     // max_depth
    Distance::Euclidean,
)?;
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_data` | `&[&[f32]]` | Training vectors for building the tree |
| `max_depth` | `usize` | Maximum depth of the tree |
| `distance` | `Distance` | Distance metric to use |

### Validation

The constructor returns an error if training data is empty.

## How TSVQ Works

1. Compute the centroid of all training vectors (root node)
2. Find the dimension with maximum variance
3. Split vectors at the median of that dimension
4. Recursively build left and right subtrees
5. Stop when reaching max depth or single vector

During quantization, traverse the tree by comparing distances to child centroids.

## Quantization

```rust
let vector: Vec<f32> = get_vector();  // dimension 64
let quantized = tsvq.quantize(&vector)?;
// Result: Vec<f16> containing the nearest leaf centroid
```

## Dequantization

```rust
let reconstructed = tsvq.dequantize(&quantized)?;
// Result: Vec<f32> of the same dimension
```

## Use Cases

TSVQ is ideal for:

- Fast quantization of high-dimensional vectors
- Hierarchical clustering applications
- When you need sub-linear quantization time
- Adaptive precision based on tree depth

## Example: Building and Using TSVQ

```rust
use vq::{TSVQ, Distance, Quantizer};

fn main() -> vq::VqResult<()> {
    // Generate synthetic training data
    let training: Vec<Vec<f32>> = (0..500)
        .map(|i| (0..32).map(|j| ((i * j) % 100) as f32 / 100.0).collect())
        .collect();
    let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Build TSVQ with depth 5
    let tsvq = TSVQ::new(&refs, 5, Distance::SquaredEuclidean)?;

    println!("Dimension: {}", tsvq.dim()); // 32

    // Quantize and reconstruct
    let query = &training[0];
    let quantized = tsvq.quantize(query)?;
    let reconstructed = tsvq.dequantize(&quantized)?;

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
let tsvq = TSVQ::new(&refs, 5, Distance::Euclidean)?;

assert_eq!(tsvq.dim(), 32);  // Expected input dimension
```

## Choosing Max Depth

The max depth determines the maximum number of leaf nodes (up to $2^{\text{depth}}$):

| Depth | Max Leaves | Quantization Comparisons |
|-------|------------|-------------------------|
| 4 | 16 | up to 4 |
| 5 | 32 | up to 5 |
| 6 | 64 | up to 6 |
| 8 | 256 | up to 8 |
| 10 | 1024 | up to 10 |

Deeper trees provide better accuracy but require more training data and memory.

## Comparison with PQ

| Aspect | TSVQ | PQ |
|--------|------|-----|
| Structure | Hierarchical tree | Flat subspaces |
| Quantization time | $O(d \log k)$ | $O(d)$ |
| Training | Recursive splitting | LBG per subspace |
| Accuracy | Depends on tree quality | Generally higher |
| Best for | Fast quantization | ANN search |
