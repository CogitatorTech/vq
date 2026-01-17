# ScalarQuantizer

Scalar quantization maps values to discrete levels within a range.

## Constructor

```python
pyvq.ScalarQuantizer(min_val: float, max_val: float, levels: int)
```

**Parameters:**

- `min_val` - Minimum value in the quantization range
- `max_val` - Maximum value in the quantization range  
- `levels` - Number of quantization levels (2-256)

**Raises:** `ValueError` if min_val >= max_val, levels < 2 or > 256, or values are NaN/Infinity

**Example:**

```python
import pyvq

# 256 levels in range [-1, 1]
sq = pyvq.ScalarQuantizer(min_val=-1.0, max_val=1.0, levels=256)
```

## Methods

### `quantize`

```python
def quantize(self, vector: numpy.ndarray) -> numpy.ndarray
```

Quantizes a vector to level indices.

**Parameters:**

- `vector` - Input vector (numpy array of float32)

**Returns:** Numpy array of uint8 containing level indices

Values are clamped to [min_val, max_val] before quantization.

**Example:**

```python
import numpy as np
import pyvq

sq = pyvq.ScalarQuantizer(-1.0, 1.0, 256)
vector = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)

indices = sq.quantize(vector)
print(indices)  # [0, 127, 191, 255] (approximately)
```

### `dequantize`

```python
def dequantize(self, quantized: numpy.ndarray) -> numpy.ndarray
```

Reconstructs a vector from level indices.

**Parameters:**

- `quantized` - Level indices (numpy array of uint8)

**Returns:** Numpy array of float32

**Example:**

```python
reconstructed = sq.dequantize(indices)
print(reconstructed)
# Values close to original, with bounded error
```

## Properties

- `min` - Minimum value
- `max` - Maximum value
- `levels` - Number of levels
- `step` - Step size between levels

```python
sq = pyvq.ScalarQuantizer(-1.0, 1.0, 256)
print(sq.min)    # -1.0
print(sq.max)    # 1.0
print(sq.levels) # 256
print(sq.step)   # ~0.0078
```

## Reconstruction Error

The maximum reconstruction error is bounded by `step / 2`:

```python
sq = pyvq.ScalarQuantizer(-1.0, 1.0, 256)
max_error = sq.step / 2  # ~0.0039

# Verify
vector = np.random.uniform(-1, 1, 100).astype(np.float32)
quantized = sq.quantize(vector)
reconstructed = sq.dequantize(quantized)
actual_max_error = np.max(np.abs(vector - reconstructed))
assert actual_max_error <= max_error + 1e-6
```

## Use Cases

- Embedding compression
- Gradient quantization
- Known-range data compression
