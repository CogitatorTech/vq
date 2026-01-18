# Getting Started

This guide covers installation and basic usage of PyVq.

## Installation

```bash
pip install pyvq
```

!!! note "Requirements"
    Python 3.10 or later is required.

## Binary Quantization

Binary quantization maps values to 0 or 1 based on a threshold. It provides 75% storage reduction.

```python
import numpy as np
import pyvq

# Create a binary quantizer
# Values >= threshold map to high, values < threshold map to low
bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

# Quantize a vector
vector = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
codes = bq.quantize(vector)
print(f"Input:  {vector}")
print(f"Output: {codes}")
# Output: [0, 0, 1, 1, 1]
```

## Scalar Quantization

Scalar quantization maps a continuous range to discrete levels.

```python
import numpy as np
import pyvq

# Create a scalar quantizer
# Maps values from [-1, 1] to 256 discrete levels
sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)

# Quantize and dequantize
vector = np.array([0.1, -0.3, 0.7, -0.9], dtype=np.float32)
quantized = sq.quantize(vector)
reconstructed = sq.dequantize(quantized)

print(f"Original:      {vector}")
print(f"Reconstructed: {reconstructed}")
```

## Product Quantization

Product quantization requires training on a dataset. It splits vectors into subspaces and learns codebooks.

```python
import numpy as np
import pyvq

# Generate training data: 100 vectors of dimension 16
training = np.random.randn(100, 16).astype(np.float32)

# Train a product quantizer
pq = pyvq.ProductQuantizer(
    training_data=training,
    num_subspaces=4,   # 4 subspaces (16/4 = 4 dims each)
    num_centroids=8,   # 8 centroids per subspace
    max_iters=10,
    distance=pyvq.Distance.euclidean(),
    seed=42
)

# Quantize a vector
vector = training[0]
quantized = pq.quantize(vector)
reconstructed = pq.dequantize(quantized)

print(f"Original dimension: {len(vector)}")
print(f"Quantized dimension: {len(quantized)}")
```

## Tree-Structured VQ

TSVQ builds a binary tree of centroids for hierarchical quantization.

```python
import numpy as np
import pyvq

# Generate training data
training = np.random.randn(100, 32).astype(np.float32)

# Create TSVQ with max depth 5
tsvq = pyvq.TSVQ(
    training_data=training,
    max_depth=5,
    distance=pyvq.Distance.squared_euclidean()
)

# Quantize
vector = training[0]
quantized = tsvq.quantize(vector)
reconstructed = tsvq.dequantize(quantized)
```

## Distance Computation

Compute distances between vectors using various metrics:

```python
import numpy as np
import pyvq

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

# Different distance metrics
euclidean = pyvq.Distance.euclidean()
manhattan = pyvq.Distance.manhattan()
cosine = pyvq.Distance.cosine()
sq_euclidean = pyvq.Distance.squared_euclidean()

print(f"Euclidean: {euclidean.compute(a, b)}")
print(f"Manhattan: {manhattan.compute(a, b)}")
print(f"Cosine: {cosine.compute(a, b)}")
print(f"Squared Euclidean: {sq_euclidean.compute(a, b)}")
```

## Concepts

### What is Vector Quantization?

Vector quantization is a lossy compression technique that approximates vectors using a smaller set of representative values. Common uses:

- **Embedding compression**: Reduce memory for ML embeddings
- **Approximate nearest neighbor search**: Speed up similarity searches
- **Data compression**: Reduce storage costs for vector databases

### Choosing an Algorithm

| Algorithm | Best For | Compression |
|-----------|----------|-------------|
| **Binary** | Fast similarity via Hamming distance | 75% |
| **Scalar** | Values with known min/max range | 75% |
| **Product** | High-dimensional embeddings | 50% |
| **TSVQ** | Hierarchical clustering | 50% |
