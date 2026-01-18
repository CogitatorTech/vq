# Vq

Vq (**v**ector **q**uantizer) is a vector quantization library for Rust 🦀.
It provides efficient implementations of popular quantization algorithms for compressing high-dimensional vectors.

## Features

- Simple and generic API via the `Quantizer` trait
- More than 50% reduction in storage size of input vectors
- SIMD acceleration support (AVX/AVX2/AVX512/NEON/SVE) via the `simd` feature
- Multi-threaded training via the `parallel` feature
- Multiple distance metrics: Euclidean, Manhattan, and cosine

## Supported Algorithms

| Algorithm              | Training Complexity | Quantization Complexity | Minimum Storage Reduction |
|------------------------|---------------------|-------------------------|---------------------------|
| Binary (BQ)            | $O(1)$              | $O(nd)$                 | 75%                       |
| Scalar (SQ)            | $O(1)$              | $O(nd)$                 | 75%                       |
| Product (PQ)           | $O(nkd)$            | $O(nd)$                 | 50%                       |
| Tree-Structured (TSVQ) | $O(n \log k)$       | $O(d \log k)$           | 50%                       |

Where $n$ is number of vectors, $d$ is the number of dimensions of a vector, and $k$ is the number of centroids used in clustering (for PQ and TSVQ).

## Quick Example

```rust
use vq::{BinaryQuantizer, Quantizer};

fn main() -> vq::VqResult<()> {
    // Create a binary quantizer with threshold 0.0
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;

    // Quantize a vector
    let quantized = bq.quantize(&[-1.0, 0.5, 1.0])?;
    assert_eq!(quantized, vec![0, 1, 1]);

    Ok(())
}
```

## Python Bindings

Python 🐍 bindings are available via [PyVq](https://pypi.org/project/pyvq/):

```bash
pip install pyvq
```

See the [PyVq documentation](https://cogitatortech.github.io/vq/python/) for Python-specific guides.

## Quick Links

- [Getting Started](getting-started.md) - Installation and first steps
- [Examples](examples.md) - Complete code examples
- [API Reference](api-reference.md) - API overview
- [docs.rs/vq](https://docs.rs/vq) - Full Rust API documentation
- [GitHub Repository](https://github.com/CogitatorTech/vq)
