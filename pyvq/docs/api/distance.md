# Distance

The `Distance` class provides distance metric implementations.

## Creating Distance Objects

```python
import pyvq

# Factory methods
euclidean = pyvq.Distance.euclidean()
squared_euclidean = pyvq.Distance.squared_euclidean()
manhattan = pyvq.Distance.manhattan()
cosine = pyvq.Distance.cosine_distance()
```

## Methods

### `compute`

```python
def compute(self, a: numpy.ndarray, b: numpy.ndarray) -> float
```

Computes the distance between two vectors.

**Parameters:**

- `a` - First vector (numpy array of float32)
- `b` - Second vector (numpy array of float32)

**Returns:** Distance as a float

**Raises:** `ValueError` if vectors have different lengths

**Example:**

```python
import numpy as np
import pyvq

dist = pyvq.Distance.euclidean()

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

result = dist.compute(a, b)
print(f"Distance: {result}")  # ~5.196
```

## Distance Metrics

### Euclidean

$$d(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

```python
dist = pyvq.Distance.euclidean()
```

### Squared Euclidean

$$d(a, b) = \sum_{i=1}^{n} (a_i - b_i)^2$$

```python
dist = pyvq.Distance.squared_euclidean()
```

Faster than Euclidean (no square root), same ordering for nearest neighbor.

### Manhattan

$$d(a, b) = \sum_{i=1}^{n} |a_i - b_i|$$

```python
dist = pyvq.Distance.manhattan()
```

Also called L1 distance. More robust to outliers.

### Cosine Distance

$$d(a, b) = 1 - \frac{a \cdot b}{\|a\| \|b\|}$$

```python
dist = pyvq.Distance.cosine_distance()
```

Measures angular difference. Value of 0 = identical direction, 1 = orthogonal, 2 = opposite.

## SIMD Acceleration

When SIMD is available (AVX/NEON), distance computations are automatically accelerated.

```python
import pyvq

backend = pyvq.get_simd_backend()
print(f"SIMD Backend: {backend}")
# e.g., "AVX2 (Auto)" or "NEON (Auto)"
```
