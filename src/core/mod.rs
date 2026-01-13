//! Core types and traits for the vq library.
//!
//! This module contains:
//! - [`Quantizer`](quantizer::Quantizer) - The unified trait implemented by all quantizers
//! - [`Distance`](distance::Distance) - Distance metrics for vector comparisons
//! - [`VqError`](error::VqError) and [`VqResult`](error::VqResult) - Error handling types
//! - [`Vector`](vector::Vector) - Generic vector type used internally

pub mod distance;
pub mod error;
#[cfg(feature = "simd")]
pub mod hsdlib_ffi;
pub mod quantizer;
pub mod vector;
