# Examples

Complete code examples demonstrating PyVq usage patterns.

## Embedding Compression with Scalar Quantization

```python
import numpy as np
import pyvq

# Simulate embeddings (normally from a model)
embeddings = np.random.randn(1000, 768).astype(np.float32)

# Normalize to [-1, 1] range
embeddings = embeddings / np.abs(embeddings).max()

# Create scalar quantizer
sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)

# Compress all embeddings
compressed = [sq.quantize(e) for e in embeddings]

# Calculate compression ratio
original_bytes = embeddings.nbytes
compressed_bytes = sum(c.nbytes for c in compressed)
print(f"Original: {original_bytes:,} bytes")
print(f"Compressed: {compressed_bytes:,} bytes")
print(f"Ratio: {original_bytes / compressed_bytes:.1f}x")

# Verify reconstruction quality
reconstructed = np.array([sq.dequantize(c) for c in compressed])
mse = np.mean((embeddings - reconstructed) ** 2)
print(f"MSE: {mse:.6f}")
```

## Product Quantization for Similarity Search

```python
import numpy as np
import pyvq

# Create a database of vectors
database = np.random.randn(10000, 128).astype(np.float32)

# Train product quantizer
pq = pyvq.ProductQuantizer(
    training_data=database[:1000],  # Use subset for training
    num_subspaces=16,    # 16 subspaces
    num_centroids=256,   # 256 centroids each
    max_iters=10,
    distance=pyvq.Distance.squared_euclidean(),
    seed=42
)

# Quantize entire database
quantized_db = [pq.quantize(v) for v in database]

# Query - find approximate nearest neighbors
query = np.random.randn(128).astype(np.float32)
query_quantized = pq.quantize(query)

# Compare distances using reconstructed vectors
dist = pyvq.Distance.squared_euclidean()
distances = []
for i, qv in enumerate(quantized_db):
    recon = pq.dequantize(qv)
    d = dist.compute(query, recon)
    distances.append((i, d))

# Get top-5 nearest
nearest = sorted(distances, key=lambda x: x[1])[:5]
print("Top 5 nearest indices:", [n[0] for n in nearest])
```

## Binary Hashing for Fast Similarity

```python
import numpy as np
import pyvq

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing bits between two binary vectors."""
    return np.sum(a != b)

# Create binary quantizer
bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)

# Hash some vectors
vectors = [
    np.array([0.5, -0.3, 0.1, -0.8, 0.2], dtype=np.float32),
    np.array([0.4, -0.2, 0.0, -0.7, 0.3], dtype=np.float32),  # Similar
    np.array([-0.6, 0.4, -0.2, 0.9, -0.1], dtype=np.float32), # Different
]

hashes = [bq.quantize(v) for v in vectors]

# Compare using Hamming distance (fast!)
print(f"Hash 0 vs 1: {hamming_distance(hashes[0], hashes[1])}")  # Low
print(f"Hash 0 vs 2: {hamming_distance(hashes[0], hashes[2])}")  # High
```

## Comparing Distance Metrics

```python
import numpy as np
import pyvq

# Create test vectors
np.random.seed(42)
a = np.random.randn(100).astype(np.float32)
b = np.random.randn(100).astype(np.float32)

# All distance metrics
metrics = [
    ("Euclidean", pyvq.Distance.euclidean()),
    ("Squared Euclidean", pyvq.Distance.squared_euclidean()),
    ("Manhattan", pyvq.Distance.manhattan()),
    ("Cosine Distance", pyvq.Distance.cosine()),
]

print("Distance between random 100-d vectors:")
for name, dist in metrics:
    result = dist.compute(a, b)
    print(f"  {name:20s}: {result:.4f}")

# SIMD backend info
print(f"\nSIMD Backend: {pyvq.get_simd_backend()}")
```

## Error Analysis

```python
import numpy as np
import pyvq

# Test reconstruction errors for different quantizers
vector = np.random.randn(64).astype(np.float32)

# Binary Quantization
bq = pyvq.BinaryQuantizer(0.0, 0, 1)
bq_q = bq.quantize(vector)
bq_r = bq.dequantize(bq_q)
bq_mse = np.mean((vector - bq_r) ** 2)

# Scalar Quantization
sq = pyvq.ScalarQuantizer(min=-3.0, max=3.0, levels=256)
sq_q = sq.quantize(vector)
sq_r = sq.dequantize(sq_q)
sq_mse = np.mean((vector - sq_r) ** 2)

print(f"Binary Quantizer MSE: {bq_mse:.4f}")
print(f"Scalar Quantizer MSE: {sq_mse:.6f}")
```
