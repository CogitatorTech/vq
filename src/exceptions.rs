use thiserror::Error;

#[derive(Debug, Error)]
pub enum VqError {
    #[error("Dimension mismatch: expected {expected}, got {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("Empty input: at least one vector is required.")]
    EmptyInput,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Invalid metric parameter for {metric}: {details}")]
    InvalidMetricParameter { metric: String, details: String },
}

pub type VqResult<T> = std::result::Result<T, VqError>;
