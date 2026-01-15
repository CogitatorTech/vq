mod bq;
mod distance;
mod pq;
mod sq;
mod tsvq;

use pyo3::prelude::*;

/// Python bindings for the Vq vector quantization library.
///
/// This module provides efficient implementations of various vector quantization
/// algorithms, including Binary Quantization (BQ), Scalar Quantization (SQ),
/// Product Quantization (PQ), and Tree-Structured Vector Quantization (TSVQ).
///
/// Example:
///     >>> import pyvq
///     >>>
///     >>> # Binary Quantization
///     >>> bq = pyvq.BinaryQuantizer(threshold=0.5)
///     >>> codes = bq.quantize([0.3, 0.7, 0.5])
///     >>> print(codes)  # [0, 1, 1]
///     >>>
///     >>> # Distance computation
///     >>> dist = pyvq.Distance.euclidean()
///     >>> result = dist.compute([1.0, 2.0], [3.0, 4.0])
#[pymodule]
fn pyvq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<distance::Distance>()?;
    m.add_class::<bq::BinaryQuantizer>()?;
    m.add_class::<sq::ScalarQuantizer>()?;
    m.add_class::<pq::ProductQuantizer>()?;
    m.add_class::<tsvq::TSVQ>()?;
    Ok(())
}
