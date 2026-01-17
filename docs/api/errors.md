# Error Types

Vq uses a custom error type for all fallible operations.

## VqError

```rust
pub enum VqError {
    /// Input vector or data is empty
    EmptyInput,

    /// Vector dimensions don't match
    DimensionMismatch { expected: usize, found: usize },

    /// Invalid parameter value
    InvalidParameter(String),
}
```

### Variants

#### `EmptyInput`

Returned when an operation receives empty input data.

```rust
let empty: Vec<&[f32]> = vec![];
let result = ProductQuantizer::new(&empty, 2, 4, 10, Distance::Euclidean, 0);
assert!(matches!(result, Err(VqError::EmptyInput)));
```

#### `DimensionMismatch`

Returned when vector dimensions don't match expected values.

```rust
let pq = ProductQuantizer::new(&training, 2, 4, 10, Distance::Euclidean, 0)?;
// pq expects dimension 8, but we pass dimension 4
let result = pq.quantize(&[1.0, 2.0, 3.0, 4.0]);
assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
```

#### `InvalidParameter`

Returned when a parameter value is invalid.

```rust
// low >= high is invalid
let result = BinaryQuantizer::new(0.0, 1, 0);
assert!(matches!(result, Err(VqError::InvalidParameter(_))));

// levels < 2 is invalid
let result = ScalarQuantizer::new(-1.0, 1.0, 1);
assert!(matches!(result, Err(VqError::InvalidParameter(_))));
```

## VqResult

A type alias for convenience:

```rust
pub type VqResult<T> = Result<T, VqError>;
```

## Error Handling

### Using `?` Operator

```rust
use vq::{BinaryQuantizer, Quantizer, VqResult};

fn process_vectors(vectors: &[Vec<f32>]) -> VqResult<Vec<Vec<u8>>> {
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;

    vectors.iter()
        .map(|v| bq.quantize(v))
        .collect()
}
```

### Using `match`

```rust
use vq::{ScalarQuantizer, VqError};

fn create_quantizer(min: f32, max: f32, levels: usize) {
    match ScalarQuantizer::new(min, max, levels) {
        Ok(sq) => println!("Created with step size: {}", sq.step()),
        Err(VqError::InvalidParameter(msg)) => {
            eprintln!("Invalid parameter: {}", msg);
        }
        Err(e) => eprintln!("Other error: {:?}", e),
    }
}
```

## Trait Implementations

`VqError` implements:

- `std::fmt::Debug`
- `std::fmt::Display`
- `std::error::Error`

This allows using Vq errors with standard Rust error handling patterns and libraries like `anyhow` or `thiserror`.
