# Quick Start

This guide demonstrates how to use PyVq for vector quantization.

## Basic Quantization

### Binary Quantization

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

### Scalar Quantization

```python
import numpy as np
import pyvq

# Create a scalar quantizer
# Maps values from [-1, 1] to 256 discrete levels
sq = pyvq.ScalarQuantizer(min_val=-1.0, max_val=1.0, levels=256)

# Quantize and dequantize
vector = np.array([0.1, -0.3, 0.7, -0.9], dtype=np.float32)
quantized = sq.quantize(vector)
reconstructed = sq.dequantize(quantized)

print(f"Original:      {vector}")
print(f"Reconstructed: {reconstructed}")
```

### Product Quantization

```python
import numpy as np
import pyvq

# Generate training data: 100 vectors of dimension 16
training = np.random.randn(100, 16).astype(np.float32)

# Train a product quantizer
pq = pyvq.ProductQuantizer(
    training_data=training,
    m=4,           # 4 subspaces
    k=8,           # 8 centroids per subspace
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

### Tree-Structured VQ

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

```python
import numpy as np
import pyvq

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

# Different distance metrics
euclidean = pyvq.Distance.euclidean()
manhattan = pyvq.Distance.manhattan()
cosine = pyvq.Distance.cosine_distance()
sq_euclidean = pyvq.Distance.squared_euclidean()

print(f"Euclidean: {euclidean.compute(a, b)}")
print(f"Manhattan: {manhattan.compute(a, b)}")
print(f"Cosine: {cosine.compute(a, b)}")
print(f"Squared Euclidean: {sq_euclidean.compute(a, b)}")
```

## Working with NumPy Arrays

All PyVq functions accept NumPy arrays:

```python
import numpy as np
import pyvq

# Input must be float32
vectors = np.random.randn(100, 64).astype(np.float32)

sq = pyvq.ScalarQuantizer(-1.0, 1.0, 256)

# Quantize all vectors
quantized = [sq.quantize(v) for v in vectors]

# Results are NumPy arrays
print(type(quantized[0]))  # <class 'numpy.ndarray'>
```

## Next Steps

- Learn about [vector quantization concepts](concepts.md)
- See the [API Reference](../api/distance.md)
- Explore more [Examples](../examples/basic.md)
