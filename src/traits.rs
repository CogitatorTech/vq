//! Common traits for vector quantization algorithms.

use crate::exceptions::VqResult;

/// A trait representing a vector quantizer.
///
/// All quantization algorithms implement this trait, providing a uniform
/// interface for encoding vectors into compact representations and
/// reconstructing approximate vectors from those representations.
///
/// # Type Parameters
///
/// * `QuantizedOutput` - The type of the quantized representation (e.g., `Vec<u8>`, `Vec<f16>`)
///
/// # Example
///
/// ```rust
/// use vq::{Quantizer, VqResult};
/// use vq::sq::ScalarQuantizer;
///
/// fn quantize_and_reconstruct<Q: Quantizer>(
///     quantizer: &Q,
///     vector: &[f32],
/// ) -> VqResult<Vec<f32>> {
///     let quantized = quantizer.quantize(vector)?;
///     quantizer.dequantize(&quantized)
/// }
/// ```
pub trait Quantizer {
    /// The output type of the quantization process.
    type QuantizedOutput;

    /// Quantizes a vector into a compact representation.
    ///
    /// # Arguments
    ///
    /// * `vector` - The input vector to quantize
    ///
    /// # Returns
    ///
    /// The quantized representation of the input vector
    ///
    /// # Errors
    ///
    /// Returns an error if the input vector has an invalid dimension or
    /// other algorithm-specific validation fails.
    fn quantize(&self, vector: &[f32]) -> VqResult<Self::QuantizedOutput>;

    /// Reconstructs an approximate vector from its quantized representation.
    ///
    /// # Arguments
    ///
    /// * `quantized` - The quantized representation to decode
    ///
    /// # Returns
    ///
    /// An approximate reconstruction of the original vector
    ///
    /// # Errors
    ///
    /// Returns an error if the quantized representation is invalid.
    fn dequantize(&self, quantized: &Self::QuantizedOutput) -> VqResult<Vec<f32>>;
}
