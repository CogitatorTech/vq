use half::f16;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vq::tsvq::TSVQ as VqTSVQ;
use vq::{Distance as VqDistance, Quantizer};

use crate::distance::Distance;

/// Tree-structured vector quantizer using hierarchical clustering.
///
/// TSVQ builds a binary tree where each node represents a cluster centroid.
/// Vectors are quantized by traversing the tree to find the nearest leaf node.
///
/// Example:
///     >>> import numpy as np
///     >>> training = np.random.rand(100, 8).astype(np.float32)
///     >>> tsvq = pyvq.TSVQ(
///     ...     training_data=training,
///     ...     max_depth=5,
///     ...     distance=pyvq.Distance.euclidean()
///     ... )
///     >>> codes = tsvq.quantize(training[0])
///     >>> reconstructed = tsvq.dequantize(codes)
#[pyclass]
pub struct TSVQ {
    quantizer: VqTSVQ,
}

#[pymethods]
impl TSVQ {
    /// Create a new Tree-Structured Vector Quantizer.
    ///
    /// Args:
    ///     training_data: 2D numpy array of training vectors (float32), shape (n_samples, dim).
    ///     max_depth: Maximum depth of the tree.
    ///     distance: Distance metric to use.
    ///
    /// Raises:
    ///     ValueError: If training data is empty.
    #[new]
    #[pyo3(signature = (training_data, max_depth, distance=None))]
    fn new(
        training_data: PyReadonlyArray2<f32>,
        max_depth: usize,
        distance: Option<Distance>,
    ) -> PyResult<Self> {
        let shape = training_data.shape();
        if shape[0] == 0 {
            return Err(PyValueError::new_err("Training data cannot be empty"));
        }

        // Convert 2D numpy array to Vec<Vec<f32>>
        let training_vec: Vec<Vec<f32>> = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| *training_data.get([i, j]).unwrap())
                    .collect()
            })
            .collect();

        let training_refs: Vec<&[f32]> = training_vec.iter().map(|v| v.as_slice()).collect();
        let dist = distance
            .map(|d| d.metric)
            .unwrap_or(VqDistance::Euclidean);

        VqTSVQ::new(&training_refs, max_depth, dist)
            .map(|q| TSVQ { quantizer: q })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a vector.
    ///
    /// Args:
    ///     vector: Input vector as numpy array (float32).
    ///
    /// Returns:
    ///     Quantized representation (leaf centroid) as numpy array (float16 stored as u16 bits).
    ///     Use `.view(np.float16)` to interpret as float16.
    fn quantize<'py>(
        &self,
        py: Python<'py>,
        vector: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<u16>>> {
        let input = vector.as_slice()?;
        let result: Vec<u16> = self
            .quantizer
            .quantize(input)
            .map(|codes| codes.iter().map(|c| c.to_bits()).collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Reconstruct a vector from its quantized representation.
    ///
    /// Args:
    ///     codes: Quantized representation as numpy array (float16 as u16 bits or via `.view(np.uint16)`).
    ///
    /// Returns:
    ///     Reconstructed vector as numpy array (float32).
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray1<u16>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input: Vec<f16> = codes
            .as_slice()?
            .iter()
            .map(|&bits| f16::from_bits(bits))
            .collect();
        let result = self
            .quantizer
            .dequantize(&input)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// The expected input vector dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    fn __repr__(&self) -> String {
        format!("TSVQ(dim={})", self.quantizer.dim())
    }
}
