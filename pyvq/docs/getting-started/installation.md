# Installation

## From PyPI

Install PyVq using pip:

```bash
pip install pyvq
```

## Requirements

- Python 3.10 or later
- NumPy (automatically installed as a dependency)

## Verifying Installation

```python
import pyvq

# Check version
print(pyvq.__version__)

# Check SIMD backend
backend = pyvq.get_simd_backend()
print(f"SIMD Backend: {backend}")
```

## Building from Source

To build from source, you need:

- Rust 1.85 or later
- Python 3.10 or later
- maturin

```bash
# Clone the repository
git clone https://github.com/CogitatorTech/vq.git
cd vq/pyvq

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install maturin
pip install maturin

# Build and install in development mode
maturin develop --release
```

## Platform Support

PyVq provides pre-built wheels for:

- Linux (x86_64, aarch64)
- macOS (ARM64)
- Windows (x86_64)

SIMD acceleration is automatically enabled on supported platforms:

| Platform | SIMD Support |
|----------|--------------|
| x86_64 | AVX, AVX2, AVX512 |
| ARM64 | NEON |
