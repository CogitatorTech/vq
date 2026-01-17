# ProductQuantizer

Product quantization divides vectors into subspaces and learns codebooks.

## Constructor

```python
pyvq.ProductQuantizer(
    training_data: numpy.ndarray,
    m: int,
    k: int,
    max_iters: int,
    distance: pyvq.Distance,
    seed: int
)
```

**Parameters:**

- `training_data` - 2D numpy array of training vectors (float32)
- `m` - Number of subspaces to divide vectors into
- `k` - Number of centroids per subspace
- `max_iters` - Maximum iterations for codebook training
- `distance` - Distance metric to use
- `seed` - Random seed for reproducibility

**Raises:** `ValueError` if training data is empty, dimension < m, or dimension not divisible by m

**Example:**

```python
import numpy as np
import pyvq

# Training data: 100 vectors of dimension 64
training = np.random.randn(100, 64).astype(np.float32)

# 8 subspaces (64/8 = 8 dims each), 256 centroids
pq = pyvq.ProductQuantizer(
    training_data=training,
    m=8,
    k=256,
    max_iters=10,
    distance=pyvq.Distance.euclidean(),
    seed=42
)
```

## Methods

### `quantize`

```python
def quantize(self, vector: numpy.ndarray) -> numpy.ndarray
```

Quantizes a vector using the learned codebooks.

**Parameters:**

- `vector` - Input vector (numpy array of float32, must match training dimension)

**Returns:** Numpy array of float16 containing quantized centroid values

**Raises:** `ValueError` if dimension doesn't match

**Example:**

```python
vector = training[0]
quantized = pq.quantize(vector)
print(quantized.shape)  # (64,)
print(quantized.dtype)  # float16
```

### `dequantize`

```python
def dequantize(self, quantized: numpy.ndarray) -> numpy.ndarray
```

Reconstructs a vector from quantized representation.

**Parameters:**

- `quantized` - Quantized vector (numpy array of float16)

**Returns:** Numpy array of float32

**Example:**

```python
reconstructed = pq.dequantize(quantized)
mse = np.mean((vector - reconstructed) ** 2)
print(f"MSE: {mse}")
```

## Properties

- `dim` - Expected input vector dimension
- `num_subspaces` - Number of subspaces (m)
- `sub_dim` - Dimension per subspace (dim / m)

```python
print(pq.dim)           # 64
print(pq.num_subspaces) # 8
print(pq.sub_dim)       # 8
```

## Parameter Selection

### Number of Subspaces (m)

- Must divide the vector dimension evenly
- Higher m = faster quantization, potentially lower accuracy
- Common values: 4, 8, 16, 32

### Number of Centroids (k)

- Higher k = better accuracy, slower training
- Common values: 16, 64, 256
- Memory per subspace: k * sub_dim * 4 bytes

## Use Cases

- Large-scale approximate nearest neighbor search
- Embedding compression for vector databases
- Memory-efficient storage of ML embeddings
