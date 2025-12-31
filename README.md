## Vq

[<img alt="tests" src="https://img.shields.io/github/actions/workflow/status/CogitatorTech/vq/tests.yml?label=tests&style=flat&labelColor=555555&logo=github" height="20">](https://github.com/CogitatorTech/vq/actions/workflows/tests.yml)
[<img alt="lints" src="https://img.shields.io/github/actions/workflow/status/CogitatorTech/vq/lints.yml?label=lints&style=flat&labelColor=555555&logo=github" height="20">](https://github.com/CogitatorTech/vq/actions/workflows/lints.yml)
[<img alt="code coverage" src="https://img.shields.io/codecov/c/github/CogitatorTech/vq?style=flat&labelColor=555555&logo=codecov" height="20">](https://codecov.io/gh/CogitatorTech/vq)
[<img alt="codefactor" src="https://img.shields.io/codefactor/grade/github/CogitatorTech/vq?style=flat&labelColor=555555&logo=codefactor" height="20">](https://www.codefactor.io/repository/github/CogitatorTech/vq)
[<img alt="crates.io" src="https://img.shields.io/crates/v/vq.svg?label=crates.io&style=flat&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-vq-66c2a5?label=docs.rs&style=flat&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/vq)
[<img alt="downloads" src="https://img.shields.io/crates/d/vq?label=downloads&style=flat&labelColor=555555&logo=rust" height="20">](https://crates.io/crates/vq)
[<img alt="msrv" src="https://img.shields.io/badge/msrv-1.83.0-orange?label=msrv&style=flat&labelColor=555555&logo=rust" height="20">](https://github.com/rust-lang/rust/releases/tag/1.83.0)
[<img alt="dependencies" src="https://deps.rs/repo/github/CogitatorTech/vq/status.svg">](https://deps.rs/repo/github/CogitatorTech/vq)
[<img alt="license" src="https://img.shields.io/badge/license-MIT%2FApache--2.0-007ec6?label=license&style=flat&labelColor=555555&logo=open-source-initiative" height="20">](https://github.com/CogitatorTech/vq)

Vq (**v**[ector] **q**[uantizer]) is a vector quantization library for Rust.
It provides implementations of popular quantization algorithms, including Binary Quantization (BQ), Scalar
Quantization (SQ), Product Quantization (PQ), Optimized Product Quantization (OPQ), Tree-structured Vector
Quantization (TSVQ), and Residual Vector Quantization (RVQ).

Vector quantization is a technique used in machine learning and data compression to reduce the size of high-dimensional
vectors by approximating them with a smaller set of representative vectors.
It can be used for various applications such as image compression and nearest neighbor search to speed up similarity
search in large datasets.

### Features

- Simple and uniform API for all quantization algorithms
- Fast distance computation using SIMD instructions (SSE and AVX) on AMD64 architecture
- Parallelized vector operations for large vectors using [Rayon](https://crates.io/crates/rayon)
- Different distance support, including Euclidean, Cosine, and Manhattan distances

### Quantization Algorithms

| Algorithm                                           | Training Complexity | Quantization Complexity | Supported Distances  | Input Type | Output Type |
|-----------------------------------------------------|---------------------|-------------------------|----------------------|------------|-------------|
| [BQ](src/bq.rs)                                     | $O(nd)$             | $O(nd)$                 | Hamming and Cosine   | `&[f32]`   | `Vec<u8>`   |
| [SQ](src/sq.rs)                                     | $O(n)$              | $O(nd)$                 | Euclidean            | `&[f32]`   | `Vec<u8>`   |
| [PQ](https://ieeexplore.ieee.org/document/5432202)  | $O(nkd)$            | $O(nd)$                 | Euclidean and Cosine | `&[f32]`   | `Vec<f16>`  |
| [TSVQ](https://ieeexplore.ieee.org/document/515493) | $O(n \log k)$       | $O(d \log k)$           | Euclidean            | `&[f32]`   | `Vec<f16>`  |

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
use vq::sq::ScalarQuantizer;
use vq::vector::Vector;

fn main() {
    // Create a scalar quantizer for values in the range [0.0, 1.0] with 256 levels.
    let quantizer = ScalarQuantizer::fit(0.0, 1.0, 256);

    // Create an input vector.
    let input = Vector::new(vec![0.1, 0.5, -0.8, -0.3, 0.9]);

    // Quantize the input vector.
    let quantized_input = quantizer.quantize(&input);

    println!("Quantized input vector: {}", quantized_input);
}
```

### Performance

Check out the [notebooks](notebooks/) directory for information on how to evaluate the performance of the implemented
algorithms.
Additionally, see the content of [src/bin](src/bin/) directory for the scripts used for the evaluation.

> On a ThinkPad T14 laptop with an Intel i7-1355U CPU and 32GB of RAM, the performance of the PQ algorithm for
> quantizing one million vectors of 128 dimensions (into 16 subspaces with 256 centroids per subspace) is as follows:
>   - Training Time: 232.5 seconds
>   - Quantization Time: 34.1 seconds
>   - Reconstruction Error (MSE): 0.02
>   - Recall@10: 0.19

### Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

### License

Vq is available under the terms of either of the following licenses:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
