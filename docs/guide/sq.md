# Scalar Quantizer (SQ)

Scalar quantization uniformly divides a value range into discrete levels, mapping each input value to its nearest quantization level.

## Overview

Scalar quantization maps floating-point values to a fixed number of discrete levels within a specified range. This achieves 75% compression by representing each value with 8 bits.

| Property | Value |
|----------|-------|
| Compression | 75% |
| Training | Not required |
| Output type | `Vec<u8>` |
| Complexity | $O(nd)$ |

## Creating a Scalar Quantizer

```rust
use vq::{ScalarQuantizer, Quantizer};

// Create a quantizer for range [-1.0, 1.0] with 256 levels
let sq = ScalarQuantizer::new(
    -1.0,  // minimum value
    1.0,   // maximum value
    256,   // number of levels
)?;
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `min` | `f32` | Minimum value in the quantization range |
| `max` | `f32` | Maximum value in the quantization range |
| `levels` | `usize` | Number of quantization levels (2-256) |

### Validation

The constructor returns an error if:

- `min` or `max` is NaN or Infinity
- `max <= min`
- `levels < 2` or `levels > 256`

## Quantization

Values are clamped to the `[min, max]` range before quantization:

```rust
let vector = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
let quantized = sq.quantize(&vector)?;
// Each value is mapped to a level index (0-255)
```

## Dequantization

Dequantization reconstructs approximate values from level indices:

```rust
let reconstructed = sq.dequantize(&quantized)?;

// Reconstruction error is bounded by step/2
for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
    let error = (orig - recon).abs();
    assert!(error <= sq.step() / 2.0 + 1e-6);
}
```

## Use Cases

Scalar quantization is ideal for:

- Values with a known, bounded range
- When you need predictable reconstruction error
- Embedding compression for retrieval systems
- Gradient quantization in distributed training

## Example: Embedding Compression

```rust
use vq::{ScalarQuantizer, Quantizer};

fn main() -> vq::VqResult<()> {
    // Assume embeddings are normalized to [-1, 1]
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;

    let embedding = vec![0.1, -0.3, 0.7, -0.9, 0.0];

    // Compress
    let compressed = sq.quantize(&embedding)?;
    println!("Compressed: {:?}", compressed);

    // Decompress
    let decompressed = sq.dequantize(&compressed)?;
    println!("Decompressed: {:?}", decompressed);

    // Calculate reconstruction error
    let mse: f32 = embedding.iter()
        .zip(decompressed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / embedding.len() as f32;
    println!("MSE: {}", mse);

    Ok(())
}
```

## Accessor Methods

```rust
let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;

assert_eq!(sq.min(), -1.0);
assert_eq!(sq.max(), 1.0);
assert_eq!(sq.levels(), 256);
println!("Step size: {}", sq.step());  // ~0.0078
```

## Choosing the Number of Levels

The number of levels affects both compression and accuracy:

| Levels | Bits per value | Max error (for range 2.0) |
|--------|----------------|---------------------------|
| 2 | 1 | 1.0 |
| 4 | 2 | 0.33 |
| 16 | 4 | 0.067 |
| 256 | 8 | 0.0039 |

For most applications, 256 levels (8-bit quantization) provides a good balance between compression and accuracy.
