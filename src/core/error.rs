use thiserror::Error;

/// Crate-specific error type for Vq operations.
#[derive(Debug, Error)]
pub enum VqError {
    /// Vectors have different dimensions where they were expected to match.
    #[error("Dimension mismatch: expected {expected}, got {found}")]
    DimensionMismatch { expected: usize, found: usize },

    /// Input data is empty when at least one element is required.
    #[error("Empty input: at least one vector is required.")]
    EmptyInput,

    /// A parameter provided to an algorithm is invalid (e.g. k=0).
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// A metric specific parameter is invalid.
    #[error("Invalid metric parameter for {metric}: {details}")]
    InvalidMetricParameter { metric: String, details: String },

    /// Input contains invalid values (like NaN or Infinity where it was not allowed).
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// A specialized Result type for Vq operations.
pub type VqResult<T> = std::result::Result<T, VqError>;
