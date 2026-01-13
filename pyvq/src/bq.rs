use pyo3::prelude::*;
use vq::bq::BinaryQuantizer as VqBinaryQuantizer;
use vq::vector::Vector;

/// A Python binding for the BinaryQuantizer.
///
/// This class maps floating-point values to one of two discrete levels
/// based on a threshold.
#[pyclass]
pub struct BinaryQuantizer {
    quantizer: VqBinaryQuantizer,
}

#[pymethods]
impl BinaryQuantizer {
    /// Create a new BinaryQuantizer.
    ///
    /// Parameters:
    /// - threshold (float): The threshold value for quantization.
    /// - low (int): The quantized value for inputs below the threshold.
    /// - high (int): The quantized value for inputs at or above the threshold.
    ///
    /// Panics if `low` is not less than `high`.
    #[new]
    fn new(threshold: f32, low: u8, high: u8) -> PyResult<Self> {
        // VqBinaryQuantizer::fit will panic if parameters are invalid.
        Ok(BinaryQuantizer {
            quantizer: VqBinaryQuantizer::new(threshold, low, high),
        })
    }

    /// Quantize a list of floats.
    ///
    /// Parameters:
    /// - v (List[float]): A list of floating-point values.
    ///
    /// Returns:
    /// - List[int]: The quantized values.
    fn quantize(&self, v: Vec<f32>) -> PyResult<Vec<u8>> {
        // Convert the input Vec<f32> into a slice and pass it to the quantizer.
        let result: Vector<u8> = self.quantizer.quantize(&v);
        Ok(result.data)
    }
}
