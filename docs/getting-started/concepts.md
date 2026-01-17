# Vector Quantization Concepts

## What is Vector Quantization?

Vector quantization (VQ) is a data compression technique that represents high-dimensional vectors using a smaller set of representative values. Instead of storing full-precision floating point numbers, VQ maps vectors to compact codes that require less storage.

## Why Use Vector Quantization?

High-dimensional vectors (like embeddings from machine learning models) consume significant memory:

| Vectors | Dimensions | Precision | Storage |
|---------|------------|-----------|---------|
| 1M | 768 | float32 | 2.9 GB |
| 1M | 768 | BQ (1-bit) | 96 MB |
| 1M | 768 | SQ (8-bit) | 768 MB |

Common use cases:

- Similarity search at scale (approximate nearest neighbor)
- Embedding storage for retrieval systems
- Reducing memory footprint of ML models
- Faster distance computations with compact representations

## Compression vs. Accuracy Trade-off

All quantization introduces some loss of precision. The choice of algorithm depends on your accuracy requirements:

| Algorithm | Compression | Reconstruction Error | Training Required |
|-----------|-------------|---------------------|-------------------|
| Binary (BQ) | 75% | High (binary) | No |
| Scalar (SQ) | 75% | Low (bounded by step size) | No |
| Product (PQ) | 50% | Medium (learned codebook) | Yes |
| TSVQ | 50% | Medium (hierarchical) | Yes |

## Algorithm Selection Guide

### Binary Quantization (BQ)

Best for: Fast approximate similarity when you only need to know if values are "high" or "low".

- Fastest quantization
- Highest compression (1 bit per value)
- No training required
- Works with Hamming distance for fast similarity

### Scalar Quantization (SQ)

Best for: When values fall within a known range and you need predictable reconstruction error.

- Fast quantization
- 75% compression (8 bits per value)
- No training required
- Bounded reconstruction error

### Product Quantization (PQ)

Best for: Large-scale approximate nearest neighbor search.

- Divides vectors into subspaces
- Learns optimal codebooks from training data
- Good balance of compression and accuracy
- Supports efficient distance computations

### Tree-Structured VQ (TSVQ)

Best for: Hierarchical data organization and fast quantization.

- Logarithmic quantization time
- Builds a binary tree of centroids
- Good for very high-dimensional data
- Natural hierarchical structure

## The Quantizer Trait

All quantizers in Vq implement a common interface:

```rust
pub trait Quantizer {
    type QuantizedOutput;

    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput>;
    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>>;
}
```

This allows you to swap algorithms without changing your code structure.

## Distance Metrics

Vq supports four distance metrics:

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | $\sqrt{\sum(a_i - b_i)^2}$ | General purpose |
| Squared Euclidean | $\sum(a_i - b_i)^2$ | Faster (no sqrt) |
| Manhattan | $\sum\|a_i - b_i\|$ | Robust to outliers |
| Cosine | $1 - \frac{a \cdot b}{\|a\| \|b\|}$ | Normalized vectors |

With the `simd` feature enabled, distance computations use SIMD instructions (AVX/NEON) for faster performance.
