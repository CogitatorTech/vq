# PyVq

PyVq provides Python bindings for the [Vq](https://github.com/CogitatorTech/vq) vector quantization library.

## Features

- High-performance Rust implementation with Python bindings
- NumPy array support for input and output
- All quantization algorithms: BinaryQuantizer, ScalarQuantizer, ProductQuantizer, TSVQ
- SIMD-accelerated distance computations
- Simple, Pythonic API

## Quick Example

```python
import numpy as np
import pyvq

# Binary Quantization
bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
vector = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
codes = bq.quantize(vector)
print(f"Quantized: {codes}")  # [0, 1, 1, 1]

# Scalar Quantization
sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
quantized = sq.quantize(vector)
reconstructed = sq.dequantize(quantized)
print(f"Reconstructed: {reconstructed}")

# Distance Computation
dist = pyvq.Distance.euclidean()
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
result = dist.compute(a, b)
print(f"Euclidean distance: {result}")
```

## Installation

```bash
pip install pyvq
```

Requires Python 3.10 or later.

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [Examples](examples.md) - Complete code examples
- [API Reference](api-reference.md) - Full API documentation

## Rust Library

For the Rust library documentation, see [docs.rs/vq](https://docs.rs/vq) or the [main documentation](https://cogitatortech.github.io/vq/).

!!! note "Early Development"
    PyVq is in early development. Please report bugs on [GitHub Issues](https://github.com/CogitatorTech/vq/issues).
