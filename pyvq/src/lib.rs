mod distance;
mod bq;
mod sq;

use pyo3::prelude::*;

/// The PyPQ module provides classes for vector quantization.
#[pymodule]
fn pyvq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bq::BinaryQuantizer>()?;
    m.add_class::<distance::Distance>()?;
    m.add_class::<sq::ScalarQuantizer>()?;
    Ok(())
}
