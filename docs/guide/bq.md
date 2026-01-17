# Binary Quantizer (BQ)

Binary quantization is the simplest form of vector quantization, mapping each value to one of two discrete levels based on a threshold comparison.

## Overview

Binary quantization converts floating-point values to binary (0 or 1) based on a threshold. This achieves 75% compression by representing each value with a single bit.

| Property | Value |
|----------|-------|
| Compression | 75% |
| Training | Not required |
| Output type | `Vec<u8>` |
| Complexity | $O(nd)$ |

## Creating a Binary Quantizer

```rust
use vq::{BinaryQuantizer, Quantizer};

// Create a quantizer with threshold 0.0
// Values >= 0.0 map to 1, values < 0.0 map to 0
let bq = BinaryQuantizer::new(
    0.0,  // threshold
    0,    // low value
    1,    // high value
)?;
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | `f32` | Values >= threshold map to high, values < threshold map to low |
| `low` | `u8` | Output value for inputs below threshold |
| `high` | `u8` | Output value for inputs at or above threshold |

### Validation

The constructor returns an error if:

- `low >= high`
- `threshold` is NaN

## Quantization

```rust
let vector = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
let quantized = bq.quantize(&vector)?;
// Result: [0, 0, 1, 1, 1]
```

## Dequantization

Dequantization maps quantized values back to floating point:

```rust
let reconstructed = bq.dequantize(&quantized)?;
// Result: [0.0, 0.0, 1.0, 1.0, 1.0]
```

Note that dequantization does not recover the original values exactly. It maps:

- Values with `low` to 0.0
- Values with `high` to 1.0

## Use Cases

Binary quantization is ideal for:

- Fast approximate similarity using Hamming distance
- Feature hashing
- Memory-constrained environments
- When only the sign of values matters

## Example: Sign-Based Quantization

```rust
use vq::{BinaryQuantizer, Quantizer};

fn main() -> vq::VqResult<()> {
    // Quantize based on sign (positive = 1, negative = 0)
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;

    let embeddings = vec![
        vec![-0.5, 0.3, -0.1, 0.8],
        vec![0.2, -0.4, 0.6, -0.2],
    ];

    for emb in &embeddings {
        let codes = bq.quantize(emb)?;
        println!("{:?} -> {:?}", emb, codes);
    }

    Ok(())
}
```

## Accessor Methods

```rust
let bq = BinaryQuantizer::new(0.5, 0, 1)?;

assert_eq!(bq.threshold(), 0.5);
assert_eq!(bq.low(), 0);
assert_eq!(bq.high(), 1);
```
