use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vq::distance::Distance as VqDistance;

/// A Python class for computing vector distances.
#[pyclass]
pub struct Distance {
    metric: VqDistance,
}

#[pymethods]
impl Distance {
    #[new]
    fn new(metric: String) -> PyResult<Self> {
        let m = match metric.to_lowercase().as_str() {
            "euclidean" => VqDistance::Euclidean,
            "squaredeuclidean" | "squared_euclidean" => VqDistance::SquaredEuclidean,
            "cosine" | "cosine_distance" => VqDistance::CosineDistance,
            "manhattan" => VqDistance::Manhattan,
            _ => return Err(PyValueError::new_err("Invalid distance metric.\
             Choose from: euclidean, squared_euclidean, cosine, and manhattan.")),
        };
        Ok(Distance { metric: m })
    }

    /// Compute the distance between two vectors.
    fn compute(&self, a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
        if a.len() != b.len() {
            return Err(PyValueError::new_err(format!(
                "Vectors must have the same length (got {} and {})",
                a.len(),
                b.len()
            )));
        }
        Ok(self.metric.compute(&a, &b))
    }
}
