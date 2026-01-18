# API Reference

This page provides an overview of Vq's public API. For detailed documentation, see [docs.rs/vq](https://docs.rs/vq).

## Core Trait

### `Quantizer`

All quantization algorithms implement this trait:

```rust
pub trait Quantizer {
    type QuantizedOutput;

    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput>;
    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>>;
}
```

## Quantizers

### `BinaryQuantizer`

Maps values above/below a threshold to two discrete levels.

```rust
// Constructor
BinaryQuantizer::new(threshold: f32, low: u8, high: u8) -> VqResult<Self>

// Getters
fn threshold(&self) -> f32
fn low(&self) -> u8
fn high(&self) -> u8

// Output type: Vec<u8>
```

### `ScalarQuantizer`

Uniformly quantizes values in a range to discrete levels (2-256).

```rust
// Constructor
ScalarQuantizer::new(min: f32, max: f32, levels: usize) -> VqResult<Self>

// Getters
fn min(&self) -> f32
fn max(&self) -> f32
fn levels(&self) -> usize
fn step(&self) -> f32

// Output type: Vec<u8>
```

### `ProductQuantizer`

Divides vectors into subspaces and quantizes each using learned codebooks.

```rust
// Constructor (requires training)
ProductQuantizer::new(
    training_data: &[&[f32]],
    m: usize,           // number of subspaces
    k: usize,           // centroids per subspace
    max_iters: usize,
    distance: Distance,
    seed: u64,
) -> VqResult<Self>

// Getters
fn num_subspaces(&self) -> usize
fn sub_dim(&self) -> usize
fn dim(&self) -> usize

// Output type: Vec<f16>
```

### `TSVQ`

Tree-structured vector quantizer using hierarchical clustering.

```rust
// Constructor (requires training)
TSVQ::new(
    training_data: &[&[f32]],
    max_depth: usize,
    distance: Distance,
) -> VqResult<Self>

// Getters
fn dim(&self) -> usize

// Output type: Vec<f16>
```

## Distance Metrics

### `Distance`

Enum for computing vector distances:

```rust
pub enum Distance {
    SquaredEuclidean,  // L2²
    Euclidean,         // L2
    Manhattan,         // L1
    CosineDistance,    // 1 - cosine_similarity
}

// Usage
Distance::Euclidean.compute(a: &[f32], b: &[f32]) -> VqResult<f32>
```

## Error Handling

### `VqError`

All operations return `VqResult<T>`, which is `Result<T, VqError>`:

```rust
pub enum VqError {
    DimensionMismatch { expected: usize, found: usize },
    EmptyInput,
    InvalidParameter(String),
    InvalidMetricParameter { metric: String, details: String },
    InvalidInput(String),
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Multi-threaded training for PQ and TSVQ |
| `simd` | SIMD acceleration (AVX/AVX2/AVX512/NEON/SVE) |

```bash
cargo add vq --features parallel,simd
```
