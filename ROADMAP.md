## Feature Roadmap

This document includes the roadmap for the Vq project.
It outlines features to be implemented and their current status.

> [!IMPORTANT]
> This roadmap is a work in progress and is subject to change.

### 1. Quantization Algorithms

* [x] Binary Quantizer (BQ)
* [x] Scalar Quantizer (SQ)
* [x] Product Quantizer (PQ)
* [x] Tree-Structured Vector Quantizer (TSVQ)

### 2. Distance Metrics

* [x] Squared Euclidean (L2)
* [x] Euclidean (L2)
* [x] Manhattan (L1)
* [x] Cosine Similarity/Distance
* [ ] Dot Product (internal only currently)
* [ ] Hamming Distance

### 3. Core Features

* [x] Unified `Quantizer` trait
* [x] Generic `Vector` struct
* [x] Codebook training (LBG/k-means)
* [x] `dequantize` support
* [ ] Persistent serialization (save and load models)
* [ ] Streaming training support

### 4. Performance Optimizations

* [x] Parallel training (Rayon)
* [x] Inline hints for hot paths
* [x] Zero-copy training (allocation reduction)
* [x] SIMD Acceleration (x86_64 AVX/AVX2/AVX512)
* [x] SIMD Acceleration (ARM NEON)
* [x] Runtime CPU feature detection
* [ ] SIMD for `f16` (half-precision)

### 5. Language Bindings

* [x] Python bindings (`pyvq`) via Maturin
* [ ] C bindings
* [ ] Node.js bindings

### 6. Tools & Binaries

* [x] `eval` tool for algorithm comparison
* [ ] CLI for direct file quantization
* [ ] Benchmark suite (internal `cargo bench`)

### 7. Documentation & Testing

* [x] Rust unit tests
* [x] Integration tests
* [x] Documentation examples
* [x] Code coverage setup
* [ ] Performance benchmarks report
* [ ] Python API documentation
