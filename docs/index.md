# Vq

Vq (**v**ector **q**uantizer) is a vector quantization library for Rust.
It provides efficient implementations of popular quantization algorithms for compressing high-dimensional vectors.

## Features

- Simple and generic API via the `Quantizer` trait
- Morer than 50% reduction in storage size of input vectors
- SIMD acceleration support (AVX/AVX2/AVX512/NEON/SVE) via the `simd` feature
- Multi-threaded training via the `parallel` feature
- Multiple distance metrics: Euclidean, Manhattan, Cosine

## Supported Algorithms

| Algorithm                               | Training      | Quantization  | Compression | Use Case                |
|-----------------------------------------|---------------|---------------|-------------|-------------------------|
| [Binary (BQ)](guide/bq.md)              | $O(1)$        | $O(nd)$       | 75%         | Fast binary similarity  |
| [Scalar (SQ)](guide/sq.md)              | $O(1)$        | $O(nd)$       | 75%         | Uniform value ranges    |
| [Product (PQ)](guide/pq.md)             | $O(nkd)$      | $O(nd)$       | 50%         | Large-scale ANN search  |
| [Tree-Structured (TSVQ)](guide/tsvq.md) | $O(n \log k)$ | $O(d \log k)$ | 50%         | Hierarchical clustering |

Where $n$ = number of vectors, $d$ = dimensions, $k$ = centroids.

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

Python bindings are available via [PyVq](https://pypi.org/project/pyvq/):

```bash
pip install pyvq
```

See the [PyVq documentation](https://cogitatortech.github.io/vq/python/) for Python-specific guides.

## Quick Links

- [Getting Started](getting-started/installation.md)
- [API Reference (docs.rs)](https://docs.rs/vq)
- [GitHub Repository](https://github.com/CogitatorTech/vq)
- [PyPI Package](https://pypi.org/project/pyvq/)
