use pyo3::prelude::*;
use vq::sq::ScalarQuantizer as VqScalarQuantizer;
use vq::vector::Vector;

/// A Python binding for the ScalarQuantizer.
///
/// This class maps floating-point values to one of several discrete levels
/// based on uniform quantization.
#[pyclass]
pub struct ScalarQuantizer {
    quantizer: VqScalarQuantizer,
}

#[pymethods]
impl ScalarQuantizer {
    /// Create a new ScalarQuantizer.
    ///
    /// Parameters:
    /// - min (float): The minimum value in the quantizer's range.
    /// - max (float): The maximum value in the quantizer's range.
    /// - levels (int): The number of quantization levels (between 2 and 256).
    ///
    /// Panics if the parameters are invalid.
    #[new]
    fn new(min: f32, max: f32, levels: usize) -> PyResult<Self> {
        Ok(ScalarQuantizer {
            quantizer: VqScalarQuantizer::new(min, max, levels),
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
