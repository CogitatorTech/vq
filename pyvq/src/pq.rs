use half::f16;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vq::pq::ProductQuantizer as VqProductQuantizer;
use vq::{Distance as VqDistance, Quantizer};

use crate::distance::Distance;

/// Product quantizer that divides vectors into subspaces and quantizes each separately.
///
/// Product quantization (PQ) splits high-dimensional vectors into smaller subspaces
/// and quantizes each subspace independently using learned codebooks.
///
/// Example:
///     >>> import numpy as np
///     >>> training = np.random.rand(100, 16).astype(np.float32)
///     >>> pq = pyvq.ProductQuantizer(
///     ...     training_data=training,
///     ...     num_subspaces=4,
///     ...     num_centroids=8,
///     ...     max_iters=20,
///     ...     distance=pyvq.Distance.euclidean(),
///     ...     seed=42
///     ... )
///     >>> codes = pq.quantize(training[0])
///     >>> reconstructed = pq.dequantize(codes)
#[pyclass]
pub struct ProductQuantizer {
    quantizer: VqProductQuantizer,
}

#[pymethods]
impl ProductQuantizer {
    /// Create a new ProductQuantizer.
    ///
    /// Args:
    ///     training_data: 2D numpy array of training vectors (float32), shape (n_samples, dim).
    ///     num_subspaces: Number of subspaces to divide vectors into (m).
    ///     num_centroids: Number of centroids per subspace (k).
    ///     max_iters: Maximum iterations for codebook training.
    ///     distance: Distance metric to use.
    ///     seed: Random seed for reproducibility.
    ///
    /// Raises:
    ///     ValueError: If training data is empty, dimension < num_subspaces,
    ///                 or dimension not divisible by num_subspaces.
    #[new]
    #[pyo3(signature = (training_data, num_subspaces, num_centroids, max_iters=10, distance=None, seed=42))]
    fn new(
        training_data: PyReadonlyArray2<f32>,
        num_subspaces: usize,
        num_centroids: usize,
        max_iters: usize,
        distance: Option<Distance>,
        seed: u64,
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

        VqProductQuantizer::new(
            &training_refs,
            num_subspaces,
            num_centroids,
            max_iters,
            dist,
            seed,
        )
        .map(|q| ProductQuantizer { quantizer: q })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a vector.
    ///
    /// Args:
    ///     vector: Input vector as numpy array (float32).
    ///
    /// Returns:
    ///     Quantized representation as numpy array (float16 stored as u16 bits).
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

    /// The number of subspaces.
    #[getter]
    fn num_subspaces(&self) -> usize {
        self.quantizer.num_subspaces()
    }

    /// The dimension of each subspace.
    #[getter]
    fn sub_dim(&self) -> usize {
        self.quantizer.sub_dim()
    }

    /// The expected input vector dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    fn __repr__(&self) -> String {
        format!(
            "ProductQuantizer(dim={}, num_subspaces={}, sub_dim={})",
            self.quantizer.dim(),
            self.quantizer.num_subspaces(),
            self.quantizer.sub_dim()
        )
    }
}
