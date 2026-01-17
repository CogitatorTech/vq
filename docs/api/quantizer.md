# Quantizer Trait

The `Quantizer` trait defines the common interface for all vector quantization algorithms in Vq.

## Definition

```rust
pub trait Quantizer {
    /// The type of the quantized output
    type QuantizedOutput;

    /// Quantize a vector into a compressed representation
    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput>;

    /// Reconstruct a vector from its quantized representation
    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>>;
}
```

## Methods

### `quantize`

```rust
fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput>
```

Converts a floating-point vector into a compressed representation.

- Input: `&[f32]` - the vector to quantize
- Output: `Self::QuantizedOutput` - the quantized representation
- Errors: May return `VqError` for dimension mismatches or other issues

### `dequantize`

```rust
fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>>
```

Reconstructs an approximate vector from its quantized representation.

- Input: `&Self::QuantizedOutput` - the quantized data
- Output: `Vec<f32>` - the reconstructed vector
- Errors: May return `VqError` for invalid quantized data

## Implementations

| Quantizer | QuantizedOutput | Compression |
|-----------|-----------------|-------------|
| `BinaryQuantizer` | `Vec<u8>` | 75% |
| `ScalarQuantizer` | `Vec<u8>` | 75% |
| `ProductQuantizer` | `Vec<f16>` | 50% |
| `TSVQ` | `Vec<f16>` | 50% |

## Generic Usage

The trait enables writing generic code that works with any quantizer:

```rust
use vq::{Quantizer, VqResult};

fn compress_vectors<Q: Quantizer>(
    quantizer: &Q,
    vectors: &[Vec<f32>],
) -> VqResult<Vec<Q::QuantizedOutput>> {
    vectors.iter()
        .map(|v| quantizer.quantize(v))
        .collect()
}

fn round_trip<Q: Quantizer>(
    quantizer: &Q,
    vector: &[f32],
) -> VqResult<Vec<f32>> {
    let quantized = quantizer.quantize(vector)?;
    quantizer.dequantize(&quantized)
}
```

## Example

```rust
use vq::{BinaryQuantizer, ScalarQuantizer, Quantizer, VqResult};

fn demonstrate_quantizer<Q: Quantizer>(name: &str, q: &Q, vector: &[f32]) -> VqResult<()>
where
    Q::QuantizedOutput: std::fmt::Debug,
{
    let quantized = q.quantize(vector)?;
    let reconstructed = q.dequantize(&quantized)?;

    let mse: f32 = vector.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / vector.len() as f32;

    println!("{}: MSE = {:.6}", name, mse);
    Ok(())
}

fn main() -> VqResult<()> {
    let vector = vec![0.1, -0.3, 0.7, -0.9, 0.5];

    let bq = BinaryQuantizer::new(0.0, 0, 1)?;
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;

    demonstrate_quantizer("BinaryQuantizer", &bq, &vector)?;
    demonstrate_quantizer("ScalarQuantizer", &sq, &vector)?;

    Ok(())
}
```
