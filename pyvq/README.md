## PyVq

[![Python version](https://img.shields.io/badge/python-%3E=3.10-3776ab?style=flat&labelColor=282c34&logo=python)](https://github.com/CogitatorTech/vq)
[![PyPI version](https://img.shields.io/pypi/v/pyvq?style=flat&labelColor=282c34&color=3775a9&logo=pypi)](https://badge.fury.io/py/pyvq)
[![Documentation](https://img.shields.io/badge/docs-read-00acc1?style=flat&labelColor=282c34&logo=readthedocs)](https://CogitatorTech.github.io/vq/python)
[![License: MIT](https://img.shields.io/badge/license-MIT-0288d1?style=flat&labelColor=282c34&logo=open-source-initiative)](LICENSE)

PyVq provides Python bindings for [Vq](https://github.com/CogitatorTech/vq) vector quantization library.

> [!IMPORTANT]
> PyVq is in early development, so breaking changes and bugs are expected.
> Please report bugs on [GitHub issues](https://github.com/CogitatorTech/vq/issues).

### Installation

```bash
pip install pyvq
```

### Quickstart

```python
import numpy as np
import pyvq

# Binary Quantization
bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
vector = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
codes = bq.quantize(vector)
print(f"Binary codes: {codes}")  # [0, 1, 1, 1]

# Scalar Quantization  
sq = pyvq.ScalarQuantizer(min_val=-1.0, max_val=1.0, levels=256)
quantized = sq.quantize(vector)
reconstructed = sq.dequantize(quantized)
print(f"Reconstructed: {reconstructed}")

# Distance Computation
dist = pyvq.Distance.euclidean()
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
print(f"Distance: {dist.compute(a, b)}")
```

### Documentation

Visit PyVq's [documentation page](https://CogitatorTech.github.io/vq/python) for detailed information including examples and API references.

### License

PyVq is licensed under the [MIT License](LICENSE).
