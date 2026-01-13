## Vq

[<img alt="tests" src="https://img.shields.io/github/actions/workflow/status/CogitatorTech/vq/tests.yml?label=tests&style=flat&labelColor=555555&logo=github" height="20">](https://github.com/CogitatorTech/vq/actions/workflows/tests.yml)
[<img alt="code coverage" src="https://img.shields.io/codecov/c/github/CogitatorTech/vq?style=flat&labelColor=555555&logo=codecov" height="20">](https://codecov.io/gh/CogitatorTech/vq)
[<img alt="codefactor" src="https://img.shields.io/codefactor/grade/github/CogitatorTech/vq?style=flat&labelColor=555555&logo=codefactor" height="20">](https://www.codefactor.io/repository/github/CogitatorTech/vq)
[<img alt="crates.io" src="https://img.shields.io/crates/v/vq.svg?label=crates.io&style=flat&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-vq-66c2a5?label=docs.rs&style=flat&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/vq)
[<img alt="msrv" src="https://img.shields.io/badge/msrv-1.83.0-orange?label=msrv&style=flat&labelColor=555555&logo=rust" height="20">](https://github.com/rust-lang/rust/releases/tag/1.83.0)
[<img alt="license" src="https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?label=license&style=flat&labelColor=555555&logo=open-source-initiative" height="20">](https://github.com/CogitatorTech/vq)

Vq (**v**[ector] **q**[uantizer]) is a vector quantization library for Rust.
It provides implementations of popular quantization algorithms, including binary quantization (BQ), scalar
quantization (SQ), product quantization (PQ), and tree-structured vector quantization (TSVQ).

Vector quantization is a technique used in machine learning and data compression to reduce the size of high-dimensional
vectors by approximating them with a smaller set of representative vectors.
It can be used for various applications such as image compression and nearest neighbor search to speed up similarity
search in large datasets.

### Features

- Simple and uniform API for all quantization algorithms
- All operations return `Result` for proper error handling
- Supports multiple distance metrics (Euclidean, Cosine, and Manhattan)

### Quantization Algorithms

| Algorithm                                           | Training Complexity | Quantization Complexity | Supported Distances  | Input Type     | Output Type   |
|-----------------------------------------------------|---------------------|-------------------------|----------------------|----------------|---------------|
| [BQ](src/bq.rs)                                     | $O(1)$              | $O(nd)$                 | Cosine               | `&[f32]`       | `Vector<u8>`  |
| [SQ](src/sq.rs)                                     | $O(1)$              | $O(nd)$                 | Euclidean            | `&[f32]`       | `Vector<u8>`  |
| [PQ](https://ieeexplore.ieee.org/document/5432202)  | $O(nkd)$            | $O(nd)$                 | Euclidean and Cosine | `&Vector<f32>` | `Vector<f16>` |
| [TSVQ](https://ieeexplore.ieee.org/document/515493) | $O(n \log k)$       | $O(d \log k)$           | Euclidean            | `&Vector<f32>` | `Vector<f16>` |

- $n$: number of vectors
- $d$: dimensionality of vectors
- $k$: number of centroids or clusters

### Installation

```bash
cargo add vq
```

*Vq requires Rust 1.83 or later.*

### Documentation

Find the latest API documentation on [docs.rs](https://docs.rs/vq).

Check out [vq_examples.rs](src/bin/vq_examples.rs) the [tests](tests) directory for detailed examples of using Vq.

#### Quick Example

Here's a simple example using the SQ algorithm to quantize a vector:

```rust
use vq::bq::BinaryQuantizer;
use vq::sq::ScalarQuantizer;
use vq::{Quantizer, VqResult};

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

### Product Quantizer Example

```rust
use vq::distance::Distance;
use vq::pq::ProductQuantizer;
use vq::VqResult;

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

### License

Vq is available under the terms of either of the following licenses:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
