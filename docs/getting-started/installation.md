# Installation

## Rust

Add `vq` to your project using Cargo:

```bash
cargo add vq
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
vq = "0.1"
```

### Optional Features

Enable optional features for better performance:

```bash
cargo add vq --features parallel,simd
```

| Feature | Description |
|---------|-------------|
| `parallel` | Multi-threaded training for PQ and TSVQ |
| `simd` | SIMD-accelerated distance computations |

!!! note "SIMD Requirements"
    The `simd` feature requires a C compiler supporting the C11 standard (GCC 4.9+ or Clang 3.1+).
    SIMD backends are automatically selected at runtime based on CPU capabilities.

### Minimum Supported Rust Version

Vq requires Rust 1.85 or later.

## Python

Install PyVq from PyPI:

```bash
pip install pyvq
```

### Requirements

- Python 3.10 or later
- NumPy (automatically installed)

### From Source

To build PyVq from source:

```bash
git clone https://github.com/CogitatorTech/vq.git
cd vq/pyvq
pip install maturin
maturin develop --release
```
