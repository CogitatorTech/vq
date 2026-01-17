# TSVQ

Tree-structured vector quantization builds a hierarchical tree for fast quantization.

## Constructor

```python
pyvq.TSVQ(
    training_data: numpy.ndarray,
    max_depth: int,
    distance: pyvq.Distance
)
```

**Parameters:**

- `training_data` - 2D numpy array of training vectors (float32)
- `max_depth` - Maximum depth of the tree
- `distance` - Distance metric to use

**Raises:** `ValueError` if training data is empty

**Example:**

```python
import numpy as np
import pyvq

# Training data: 100 vectors of dimension 32
training = np.random.randn(100, 32).astype(np.float32)

tsvq = pyvq.TSVQ(
    training_data=training,
    max_depth=5,
    distance=pyvq.Distance.squared_euclidean()
)
```

## Methods

### `quantize`

```python
def quantize(self, vector: numpy.ndarray) -> numpy.ndarray
```

Quantizes a vector by traversing the tree to the nearest leaf.

**Parameters:**

- `vector` - Input vector (numpy array of float32)

**Returns:** Numpy array of float16 containing the leaf centroid

**Raises:** `ValueError` if dimension doesn't match

**Example:**

```python
vector = training[0]
quantized = tsvq.quantize(vector)
print(quantized.shape)  # (32,)
print(quantized.dtype)  # float16
```

### `dequantize`

```python
def dequantize(self, quantized: numpy.ndarray) -> numpy.ndarray
```

Reconstructs a vector from its quantized representation.

**Parameters:**

- `quantized` - Quantized vector (numpy array of float16)

**Returns:** Numpy array of float32

**Example:**

```python
reconstructed = tsvq.dequantize(quantized)
mse = np.mean((vector - reconstructed) ** 2)
print(f"MSE: {mse}")
```

## Properties

- `dim` - Expected input vector dimension

```python
print(tsvq.dim)  # 32
```

## How It Works

TSVQ builds a binary tree:

1. Compute centroid of all training vectors (root)
2. Find dimension with maximum variance
3. Split vectors at median of that dimension
4. Recursively build left and right subtrees
5. Stop at max_depth or single vector

Quantization traverses the tree by comparing distances to child centroids.

## Parameter Selection

### Max Depth

The depth controls the maximum number of leaf nodes (up to 2^depth):

| Depth | Max Leaves | Comparisons |
|-------|------------|-------------|
| 4 | 16 | up to 4 |
| 5 | 32 | up to 5 |
| 6 | 64 | up to 6 |
| 8 | 256 | up to 8 |

Deeper trees = better accuracy but more memory.

## Use Cases

- Fast quantization (logarithmic time)
- Hierarchical clustering
- When quantization speed is critical
