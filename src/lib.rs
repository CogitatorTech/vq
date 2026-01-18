//! # Vq
//!
//! `vq` is a vector quantization library for Rust.
//!
//! It provides efficient implementations of various vector quantization algorithms,
//! including Binary Quantization (BQ), Scalar Quantization (SQ), Product Quantization (PQ),
//! and Tree-Structured Vector Quantization (TSVQ).
//!
//! ## Features
//!
//! - Unified API: All algorithms implement the [`Quantizer`] trait.
//! - Distance metrics: Supports Euclidean, Squared Euclidean, Manhattan, and Cosine distance.
//! - Performance:
//!   - SIMD acceleration (AVX/AVX2/AVX512/NEON/SVE) via `simd` feature.
//!   - Parallel training via `parallel` feature.
//!
//! ## Example
//!
//! ```rust
//! use vq::{Quantizer, VqResult};
//! use vq::sq::ScalarQuantizer;
//!
//! fn main() -> VqResult<()> {
//!     let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;
//!     let vector = vec![0.5, -0.2, 0.9];
//!     let quantized = sq.quantize(&vector)?;
//!     let reconstructed = sq.dequantize(&quantized)?;
//!     Ok(())
//! }
//! ```

pub mod bq;
pub mod core;
pub mod pq;
pub mod sq;
pub mod tsvq;

pub use bq::BinaryQuantizer;
pub use pq::ProductQuantizer;
pub use sq::ScalarQuantizer;
pub use tsvq::TSVQ;

pub use core::distance::Distance;
pub use core::error::{VqError, VqResult};
pub use core::quantizer::Quantizer;
pub use core::vector::Vector;

#[cfg(feature = "simd")]
pub use core::hsdlib_ffi::get_simd_backend;
