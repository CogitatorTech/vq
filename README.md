## Vq

[<img alt="tests" src="https://img.shields.io/github/actions/workflow/status/CogitatorTech/vq/tests.yml?label=tests&style=flat&labelColor=555555&logo=github" height="20">](https://github.com/CogitatorTech/vq/actions/workflows/tests.yml)
[<img alt="code coverage" src="https://img.shields.io/codecov/c/github/CogitatorTech/vq?style=flat&labelColor=555555&logo=codecov" height="20">](https://codecov.io/gh/CogitatorTech/vq)
[<img alt="crates.io" src="https://img.shields.io/crates/v/vq.svg?label=crates.io&style=flat&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-vq-66c2a5?label=docs.rs&style=flat&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/vq)
[![Documentation](https://img.shields.io/badge/docs-read-00acc1?style=flat&labelColor=555555&logo=readthedocs)](https://CogitatorTech.github.io/vq)
[<img alt="license" src="https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?label=license&style=flat&labelColor=555555&logo=open-source-initiative" height="20">](https://github.com/CogitatorTech/vq)

Vq (**v**[ector] **q**[uantizer]) is a vector quantization library for Rust.
It provides implementations of popular quantization algorithms, including binary quantization (BQ), scalar quantization (SQ),
product quantization (PQ), and tree-structured vector quantization (TSVQ).

Vector quantization is a technique to reduce the size of high-dimensional vectors by approximating them with a smaller set of representative vectors.
It can be used for various applications such as data compression and nearest neighbor search to reduce the memory footprint and speed up search.

### Features

- A simple and generic API for all quantizers
- Can reduce storage size by up to 75%
- Good performance via SIMD acceleration, multi-threading, and zero-copying
- Support for multiple distances including Euclidean, cosine, and Manhattan distances
- Python bindings via [PyVq](https://pypi.org/project/pyvq/) package

See [ROADMAP.md](ROADMAP.md) for the list of implemented and planned features.

> [!IMPORTANT]
> Vq is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/CogitatorTech/vq/issues) to report bugs or request features.

### Supported Algorithms

| Algorithm           | Training Complexity | Quantization Complexity | Supported Distances | Input Type | Output Type | Compression |
|---------------------|---------------------|-------------------------|---------------------|------------|-------------|-------------|
| [BQ](src/bq.rs)     | $O(1)$              | $O(nd)$                 | —                   | `&[f32]`   | `Vec<u8>`   | 75%         |
| [SQ](src/sq.rs)     | $O(1)$              | $O(nd)$                 | —                   | `&[f32]`   | `Vec<u8>`   | 75%         |
| [PQ](src/pq.rs)     | $O(nkd)$            | $O(nd)$                 | All                 | `&[f32]`   | `Vec<f16>`  | 50%         |
| [TSVQ](src/tsvq.rs) | $O(n \log k)$       | $O(d \log k)$           | All                 | `&[f32]`   | `Vec<f16>`  | 50%         |

- $n$: number of vectors
- $d$: dimensionality of vectors
- $k$: number of centroids or clusters

---

### Getting Started

#### Installation

Add `vq` to your `Cargo.toml`:

```bash
cargo add vq --features parallel simd
```

> [!NOTE]
> The `parallel` and `simd` features enables multi-threading support and SIMD acceleration support for training phase of PQ and TSVQ algorithms.
> This can significantly speed up training time, especially for large datasets.
> Note that the enable `simd` feature a modern C compiler (like GCC or Clang) that supports C11 standard is needed.

*Vq requires Rust 1.85 or later.*

---

### Python Bindings

Python bindings for Vq are available via [PyVq](https://pypi.org/project/pyvq/) package.

```bash
pip install pyvq
```

For more information, check out the [pyvq](pyvq) directory.

---

### Documentation

Check out the latest API documentation on [docs.rs](https://docs.rs/vq).

#### Quick Example

Here's a simple example using the BQ and SQ algorithms to quantize vectors:

```rust
use vq::{BinaryQuantizer, ScalarQuantizer, Quantizer, VqResult};

fn main() -> VqResult<()> {
    // Binary quantization
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;
    let quantized = bq.quantize(&[0.5, -0.3, 0.8])?;

    // Scalar quantization
    let sq = ScalarQuantizer::new(0.0, 1.0, 256)?;
    let quantized = sq.quantize(&[0.1, 0.5, 0.9])?;

    Ok(())
}
```

#### Product Quantizer Example

```rust
use vq::{ProductQuantizer, Distance, VqResult};

fn main() -> VqResult<()> {
    // Training data (each inner slice is a vector)
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..10).map(|j| ((i + j) % 50) as f32).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    // Train the quantizer
    let pq = ProductQuantizer::new(
        &training_refs,
        2,  // m: number of subspaces
        4,  // k: centroids per subspace
        10, // max_iters
        Distance::Euclidean,
        42, // seed
    )?;

    // Quantize a vector
    let quantized = pq.quantize(&training[0])?;

    Ok(())
}
```

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### License

Vq is available under either of the following licenses:

* MIT License ([LICENSE-MIT](LICENSE-MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

### Acknowledgements

* This project uses [Hsdlib](https://github.com/habedi/hsdlib) library for SIMD acceleration.
