# Vector Quantization Concepts

## What is Vector Quantization?

Vector quantization (VQ) is a data compression technique that represents high-dimensional vectors using a smaller set of representative values. Instead of storing full-precision floating point numbers, VQ maps vectors to compact codes that require less storage.

## Why Use Vector Quantization?

High-dimensional vectors consume significant memory. For example, 1 million 768-dimensional embeddings:

| Format | Storage |
|--------|---------|
| float32 | 2.9 GB |
| 8-bit (SQ) | 768 MB |
| 1-bit (BQ) | 96 MB |

Common use cases:

- Embedding storage for retrieval systems
- Approximate nearest neighbor search
- Reducing memory footprint for large vector databases
- Speeding up similarity computations

## Compression vs. Accuracy

All quantization introduces some loss of precision:

| Algorithm | Compression | Accuracy | Training |
|-----------|-------------|----------|----------|
| Binary (BQ) | 75% | Low | No |
| Scalar (SQ) | 75% | High | No |
| Product (PQ) | 50% | Medium | Yes |
| TSVQ | 50% | Medium | Yes |

## Algorithm Overview

### Binary Quantization

Converts each value to 0 or 1 based on a threshold:

```python
# Values >= 0 become 1, values < 0 become 0
bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
```

Best for: Fast Hamming distance comparisons, when sign is meaningful.

### Scalar Quantization

Maps continuous values to discrete levels:

```python
# 256 levels in range [-1, 1]
sq = pyvq.ScalarQuantizer(min_val=-1.0, max_val=1.0, levels=256)
```

Best for: Known value ranges, predictable reconstruction error.

### Product Quantization

Divides vectors into subspaces and learns codebooks:

```python
pq = pyvq.ProductQuantizer(
    training_data=vectors,
    m=8,      # subspaces
    k=256,    # centroids per subspace
    ...
)
```

Best for: Large-scale nearest neighbor search, when training data is available.

### Tree-Structured VQ

Builds a hierarchical tree for fast quantization:

```python
tsvq = pyvq.TSVQ(
    training_data=vectors,
    max_depth=6,
    ...
)
```

Best for: Fast quantization time, hierarchical organization.

## Distance Metrics

PyVq supports four distance metrics:

| Metric | Use Case |
|--------|----------|
| Euclidean | General purpose |
| Squared Euclidean | Fast comparisons (no sqrt) |
| Manhattan | Robust to outliers |
| Cosine Distance | Normalized embeddings |

```python
dist = pyvq.Distance.euclidean()
result = dist.compute(vector_a, vector_b)
```

## NumPy Integration

All PyVq functions work with NumPy arrays:

- Input vectors should be `np.float32`
- Output is always `np.ndarray`
- Batch processing can be done with list comprehensions
