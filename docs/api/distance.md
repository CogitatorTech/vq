# Distance API Reference

The `Distance` enum provides distance metric implementations for vector comparisons.

## Enum Variants

```rust
pub enum Distance {
    /// Squared Euclidean distance (L2²)
    SquaredEuclidean,

    /// Euclidean distance (L2)
    Euclidean,

    /// Manhattan distance (L1)
    Manhattan,

    /// Cosine distance (1 - cosine similarity)
    CosineDistance,
}
```

## Methods

### `compute`

```rust
impl Distance {
    pub fn compute(&self, a: &[f32], b: &[f32]) -> VqResult<f32>
}
```

Computes the distance between two vectors.

#### Arguments

- `a: &[f32]` - First vector
- `b: &[f32]` - Second vector

#### Returns

- `Ok(f32)` - The computed distance
- `Err(VqError::DimensionMismatch)` - If vectors have different lengths

#### SIMD Acceleration

When the `simd` feature is enabled, this method uses SIMD instructions where available:

- x86/x86_64: AVX, AVX2, AVX512, FMA
- ARM: NEON, SVE

The backend is automatically selected at runtime.

## Examples

### Basic Usage

```rust
use vq::Distance;

let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

let dist = Distance::Euclidean.compute(&a, &b)?;
println!("Distance: {}", dist);
```

### All Metrics

```rust
use vq::Distance;

let a = vec![1.0, 0.0];
let b = vec![0.0, 1.0];

let metrics = [
    ("SquaredEuclidean", Distance::SquaredEuclidean),
    ("Euclidean", Distance::Euclidean),
    ("Manhattan", Distance::Manhattan),
    ("CosineDistance", Distance::CosineDistance),
];

for (name, metric) in metrics {
    let dist = metric.compute(&a, &b)?;
    println!("{}: {}", name, dist);
}
```

### Error Handling

```rust
use vq::Distance;

let a = vec![1.0, 2.0];
let b = vec![1.0, 2.0, 3.0];

match Distance::Euclidean.compute(&a, &b) {
    Ok(dist) => println!("Distance: {}", dist),
    Err(e) => println!("Error: {}", e),
}
```

## Related Functions

### `get_simd_backend`

```rust
#[cfg(feature = "simd")]
pub fn get_simd_backend() -> String
```

Returns the name of the active SIMD backend.

```rust
#[cfg(feature = "simd")]
{
    let backend = vq::get_simd_backend();
    println!("Using: {}", backend);
    // e.g., "AVX2 (Auto)", "NEON (Auto)", "Scalar"
}
```

## Trait Implementations

`Distance` implements:

- `Debug`
- `Clone`
- `Copy`
- `PartialEq`
- `Eq`
