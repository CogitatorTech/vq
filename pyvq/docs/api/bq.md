# BinaryQuantizer

Binary quantization maps values to 0 or 1 based on a threshold.

## Constructor

```python
pyvq.BinaryQuantizer(threshold: float, low: int = 0, high: int = 1)
```

**Parameters:**

- `threshold` - Values >= threshold map to high, values < threshold map to low
- `low` - Output value for inputs below threshold (default: 0)
- `high` - Output value for inputs at or above threshold (default: 1)

**Raises:** `ValueError` if low >= high or threshold is NaN

**Example:**

```python
import pyvq

# Values >= 0.0 become 1, values < 0.0 become 0
bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
```

## Methods

### `quantize`

```python
def quantize(self, vector: numpy.ndarray) -> numpy.ndarray
```

Quantizes a vector to binary codes.

**Parameters:**

- `vector` - Input vector (numpy array of float32)

**Returns:** Numpy array of uint8 containing low/high values

**Example:**

```python
import numpy as np
import pyvq

bq = pyvq.BinaryQuantizer(0.0, 0, 1)
vector = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)

codes = bq.quantize(vector)
print(codes)  # [0, 1, 1, 1]
```

### `dequantize`

```python
def dequantize(self, quantized: numpy.ndarray) -> numpy.ndarray
```

Reconstructs a vector from binary codes.

**Parameters:**

- `quantized` - Quantized codes (numpy array of uint8)

**Returns:** Numpy array of float32 (0.0 for low, 1.0 for high)

**Example:**

```python
codes = np.array([0, 1, 1, 1], dtype=np.uint8)
reconstructed = bq.dequantize(codes)
print(reconstructed)  # [0.0, 1.0, 1.0, 1.0]
```

## Properties

- `threshold` - The threshold value
- `low` - The low quantization level
- `high` - The high quantization level

```python
bq = pyvq.BinaryQuantizer(0.5, 0, 1)
print(bq.threshold)  # 0.5
print(bq.low)        # 0
print(bq.high)       # 1
```

## Use Cases

- Fast Hamming distance similarity
- Sign-based feature hashing
- Memory-constrained environments
